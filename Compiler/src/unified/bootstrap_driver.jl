# This file is a part of Julia. License is MIT: https://julialang.org/license

# The sys-unified sysimage stage script (sysimage.mk): the structural twin of
# the standard `sys` stage (sysbase.so + contrib/generate_precompile.jl), but
# with the UnifiedIR-native compiler port baked into the image as
# `Base.UnifiedCompiler` and made the runtime's DEFAULT inference path
# BEFORE the precompile workload runs.
#
# The flip: in the sysbase image `jl_typeinf_func` already points at
# `Base.Compiler.typeinf_ext_toplevel` (basecompiler's `bootstrap!`; the
# pointer is serialized), and that entry consults `Compiler.UNIFIED_HOOKS`.
# `enable_pipeline!()` installs the REAL driver (`unified_typeinf`) there,
# so every inference request — the workload's, the sysimage output phase's,
# and (the Ref's value persists into the dumped image) every request in the
# booted sys-unified.so — tries the unified pipeline first and falls back
# to stock per body, counted in `pipeline_stats()` (the ledger printed at
# the checkpoints below).
#
# Warmup: before the flip, the UnifiedCompiler pkgimage's precompile
# workload (UnifiedCompiler/src/precompile.jl) is replayed in-process — its
# phase 1 drives the driver, the UNIFIED_HOOKS reflection entries, and the
# Queries/converter surface over the representative body corpus, compiling
# the driver's whole hot path under the still-stock (fully cached sysbase)
# compiler; its phase-2 stock `bootstrap!` sweep resolves from the sysbase
# cache in seconds. Post-flip reentrant driver-code compiles then only
# cover the long tail, under the reentrant valve/budgets.

Core.println("UNIFIED: baking Base.UnifiedCompiler")
const _t_bake = time_ns()
Core.eval(Base, Expr(:const, Expr(:(=), :_UNIFIED_BOOT_DIR, @__DIR__)))
Core.include(Base, joinpath(@__DIR__, "bootstrap.jl"))
let dt = (time_ns() - _t_bake) / 1e9
    Core.println("UNIFIED: bake wall time: ", round(dt; digits=1), " s")
end

const _U = Base.UnifiedCompiler

# refresh jl_typeinf_world right after the bake: the runtime consults the
# UNIFIED_HOOKS inside the pinned typeinf world, which basecompiler's
# bootstrap! captured long before this script baked Base.UnifiedCompiler —
# without the refresh every hooked compile is a world-age MethodError on
# unified_typeinf (the warmup's brief hooks-on window included). Same
# function, new world: jl_set_typeinf_func stores jl_get_tls_world_age().
ccall(:jl_set_typeinf_func, Cvoid, (Any,), Base.Compiler.typeinf_ext_toplevel)

# the pipeline ledger (pipeline_stats), one block per checkpoint
function _unified_ledger_line(tag::String)
    s = _U.pipeline_stats()
    total = s.unified + sum(values(s.fallbacks); init=0)
    Core.println("UNIFIED[", tag, "]: unified=", s.unified, "/", total, " bodies")
    for (reason, n) in sort!(collect(s.fallbacks); by=last, rev=true)
        Core.println("UNIFIED[", tag, "]:   fallback ", rpad(String(reason), 22), " ", n)
    end
    let m = s.memo
        Core.println("UNIFIED[", tag, "]:   memo hits=", m.hits, " misses=", m.misses,
                     " stale=", m.stale, " stores=", m.stores, " entries=", m.entries)
    end
    if s.last_error !== nothing
        reason, mi, err = s.last_error
        msg = try
            sprint(Base.showerror, err)
        catch
            string(typeof(err))
        end
        Core.println("UNIFIED[", tag, "]: last_error ", reason, " at ", mi, ": ",
                     first(msg, 400))
    end
    return nothing
end

Core.println("UNIFIED: warmup sweep (UnifiedCompiler/src/precompile.jl, pre-flip)")
const _t_warmup = time_ns()
let shim = Module(:UnifiedBootWarmup)
    Core.eval(shim, Expr(:const, Expr(:(=), :Unified, _U)))
    Core.eval(shim, Expr(:const, Expr(:(=), :Compiler, Base.Compiler)))
    # the pkgimage resets its session state before serialization; here the
    # ledger is reset explicitly below (after the post-warmup print)
    Core.eval(shim, :(_reset_session_state!() = nothing))
    Base.include(shim, joinpath(@__DIR__, "..", "..", "..",
                                "UnifiedCompiler", "src", "precompile.jl"))
end
let dt = (time_ns() - _t_warmup) / 1e9
    Core.println("UNIFIED: warmup wall time: ", round(dt; digits=1), " s")
end
_unified_ledger_line("post-warmup")
_U.reset_pipeline_stats!()

Core.println("UNIFIED: pipeline ON (Compiler.UNIFIED_HOOKS global mode -> unified_typeinf; ",
             "jl_typeinf_func stays Compiler.typeinf_ext_toplevel, world refreshed)")
_U.SHADOW_ENABLED[] = false
# global_mode is a FIELD of the UnifiedHooks object (the wave-12
# reflection/global mode split): without it typeinf_ext_toplevel never
# consults the hooks for C-driven inference and the whole stage — and the
# booted image — silently runs stock. enable_pipeline!(global_mode=true)
# also re-registers jl_typeinf_func AFTER installing the hooks, refreshing
# the pinned jl_typeinf_world to one that includes everything the warmup
# defined (staticdata serializes that world, so it persists into the
# dumped image — exactly what package-mode activate! does).
_U.enable_pipeline!(global_mode = true)
_U.GLOBAL_MODE[] = true

const _t_workload = time_ns()
include(joinpath(@__DIR__, "..", "..", "..", "contrib", "generate_precompile.jl"))
let dt = (time_ns() - _t_workload) / 1e9
    Core.println("UNIFIED: generate_precompile workload wall time: ", round(dt; digits=1), " s")
end
_unified_ledger_line("post-workload")

# Drop the session-scoped driver state so it is not serialized into the
# image — memo tables/query states keyed by build-process MethodInstances
# with world stamps, lock-owner task refs, the last_error exception.
# Inline, not `Base.atexit`: sysimage-stage processes never run module
# `__init__`s, so the `_atexit_hooks_finished = true` the sysbase dump
# baked is never reset (Base.__init__ does that) and atexit registration
# errors "already exiting" in ANY sys stage. Nothing runs between here and
# process exit except the exit path itself; the output phase recompiles
# through the (still enabled) pipeline, so entries it stores do land in
# the image — they revalidate per (world, counter) stamps at first use in
# the booted session (the A6 contract). The flip itself —
# UNIFIED_HOOKS/GLOBAL_MODE — and the ledger counters stay.
empty!(_U.DRIVER_MEMO)
_U.MEMO_OWNER[] = nothing
_U.STATS_OWNER[] = nothing
empty!(_U.EA_OPT_SUMMARIES); empty!(_U.EA_OPT_ACTIVE); _U.EA_OPT_WORLD[] = 0
empty!(_U.OPT_FX_MEMO);      empty!(_U.OPT_FX_ACTIVE); _U.OPT_FX_WORLD[] = 0
_U.OPT_NEST_DEPTH[] = 0; _U.OPT_WORK_LEFT[] = 0
empty!(_U.INLINE_COST_MEMO); empty!(_U.INLINE_COST_ACTIVE); _U.INLINE_COST_WORLD[] = 0
empty!(_U.QUERY_STATES)
_U.PIPELINE_STATS.last_error = nothing
_U.TOWER_FRAME_CAP[] = 0
# the reentrant valve's session counter must not be baked at its exhausted
# value, or the booted image would never admit reentrant unified passes
_U.REENTRANT_ADMITTED[] = 0

# the sysimage output phase compiles further bodies through the (still
# active) unified pipeline; report the final ledger after output
Base.postoutput() do
    _unified_ledger_line("post-output")
end
