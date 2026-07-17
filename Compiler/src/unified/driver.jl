# The real typeinf driver (COMPILER-PORT-PLAN A1): `unified_typeinf` runs the
# native unified pipeline — entry-convert → infer_ir! → optimize_ir! →
# ir_to_ircode → CodeInfo — behind the Compiler module's standard entry
# points, producing cache-grade CodeInstances with stock-encoded edges and
# sound world bounds. Every body the pipeline cannot (yet) handle falls back
# to the stock compiler, per body, with a counted reason (`pipeline_stats()`
# is the ratchet). Installed via `enable_pipeline!` (Compiler.UNIFIED_HOOKS);
# `activate!` additionally flips the runtime's jl_typeinf_func.
#
# Concurrency and reentrancy (A5): every request runs with a FRESH
# `UInferState`/`UEdges` pair, so there is no shared inference state between
# passes — each collector sees exactly the facts its own pass consumed. The
# per-mi serialization is the engine's (`engine_reserve`, whose C side
# resolves same-thread and cross-thread reservation cycles without
# deadlocking; a same-thread re-reservation returns a non-owning placeholder
# CodeInstance and `jl_engine_fulfill` ignores non-reservations). What the
# driver adds is a small per-TASK discipline:
#   - an `inflight` set declines requests for a MethodInstance this task is
#     already driving (`:reentrant_self`) — recursing on the same body can
#     only redo the same work against the engine placeholder;
#   - a depth counter bounds nested driver passes (`:reentrant_depth`).
#     Reentrant requests below the bound — the runtime compiling something
#     the driver's own execution needs, and the devirtualizer's callee
#     CodeInstance production — run the unified pipeline recursively; at the
#     bound they decline precisely and stock compiles the body (cached, so
#     each such body is compiled at most once per session). The runtime
#     itself additionally caps `jl_typeinf_func` reentrancy per task (gf.c
#     reentrant_timing), so runtime-initiated recursion is shallow by
#     construction; the driver bound mainly governs its own recursion.
#
# Soundness protocol (mirrors stock finish!/finish_nocycle):
#   - the collector starts at WorldRange(1, world_counter) and intersects the
#     validity window of every consulted fact (see UEdges in uinference.jl);
#   - at finish, if the intersection no longer reaches the CURRENT counter,
#     the result is NOT cached (reason :world_moved/:world_bounded) — unlike
#     stock we do not publish bounded CodeInstances, because the unified
#     OPTIMIZER still reads ambient global state (isconst/getglobal in
#     static_operand_value & co.), which is only provably world-consistent
#     when nothing moved during the pass. Lazy binding/partition
#     materialization bumps the counter once per binding per process, so the
#     driver retries once before giving up;
#   - otherwise: store_backedges (stock encoding via build_edges) →
#     jl_fill_codeinst → cache insert → engine fulfill → codegen-cache
#     insert → jl_promote_ci_to_current, exactly stock's sequence.

# ---------------------------------------------------------------------------
# Fallback ledger (the ratchet)
# ---------------------------------------------------------------------------

mutable struct PipelineLedger
    unified::Int                  # bodies fully through the unified pipeline
    fallbacks::Dict{Symbol,Int}   # reason -> count (stock handled the body)
    last_error::Any               # (reason, mi, exception) of the last error-class fallback
end
const PIPELINE_STATS = PipelineLedger(0, Dict{Symbol,Int}(), nothing)
# requests run concurrently (no global driver lock): ledger writes take this
const STATS_LOCK = Base.Threads.SpinLock()

function count_fallback!(reason::Symbol, @nospecialize(mi = nothing), @nospecialize(err = nothing))
    Base.@lock STATS_LOCK begin
        d = PIPELINE_STATS.fallbacks
        d[reason] = get(d, reason, 0) + 1
        err === nothing || (PIPELINE_STATS.last_error = (reason, mi, err))
    end
    return nothing
end

note_unified!() = (Base.@lock STATS_LOCK PIPELINE_STATS.unified += 1; nothing)

"""
    pipeline_stats() -> NamedTuple

The driver's ledger: `unified` counts bodies compiled end-to-end by the
unified pipeline, `fallbacks` maps fallback reason to count (those bodies
were handled by the stock compiler), `last_error` retains the most recent
`(reason, mi, exception)` for error-class fallbacks.
"""
pipeline_stats() = Base.@lock STATS_LOCK (; unified = PIPELINE_STATS.unified,
                    fallbacks = copy(PIPELINE_STATS.fallbacks),
                    last_error = PIPELINE_STATS.last_error)

function reset_pipeline_stats!()
    Base.@lock STATS_LOCK begin
        PIPELINE_STATS.unified = 0
        empty!(PIPELINE_STATS.fallbacks)
        PIPELINE_STATS.last_error = nothing
    end
    return nothing
end

"One-line ledger print (the demo/bench surface)."
function print_pipeline_stats(io::IO = Base.stdout)
    stats = pipeline_stats()
    total = stats.unified + sum(values(stats.fallbacks); init = 0)
    println(io, "unified pipeline: ", stats.unified, "/", total, " bodies")
    for (reason, n) in sort!(collect(stats.fallbacks); by = last, rev = true)
        println(io, "  fallback ", rpad(String(reason), 22), " ", n)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Reentrancy / concurrency guard (see the header comment)
# ---------------------------------------------------------------------------

"Per-task driver state: nesting depth, the MethodInstances this task is
currently driving (each holds an engine reservation up-stack), and whether a
devirtualization CodeInstance production is in progress (`devirt` — bounds
eager callee compilation to one level per root chain; see
`driver_ci_for_invoke`)."
mutable struct DriverTaskState
    depth::Int
    devirt::Int
    const inflight::Base.IdSet{Core.MethodInstance}
end

const DRIVER_TLS_KEY = :unified_compiler_driver_state

function driver_task_state()::DriverTaskState
    tls = Base.task_local_storage()
    v = get(tls, DRIVER_TLS_KEY, nothing)
    v isa DriverTaskState && return v
    st = DriverTaskState(0, 0, Base.IdSet{Core.MethodInstance}())
    tls[DRIVER_TLS_KEY] = st
    return st
end

"Nested driver passes this task admits before declining (`:reentrant_depth`).
Native recursion: each level stacks a full pipeline pass (which itself
recurses per DRIVER_MAX_DEPTH), so the bound stays small — declined bodies
are stock-compiled once and cached, and devirtualization targets degrade to
MethodInstance invokes whose CodeInstances materialize on first call."
const DRIVER_REENTRY_LIMIT = Base.RefValue(8)

"Per-session admission budget for REENTRANT passes (requests arriving while
this task is already inside the driver — the self-hosting burn-in). Every
pass re-infers its callee tree with fresh state (the pre-A6 soundness
basis), so unbounded admission makes first activation re-derive the
compiler's own call graph body by body — an hour-class burn-in. Admit up to
this many reentrant bodies through the pipeline per session (unified,
cached), then decline precisely (`:reentrant_budget` — stock compiles and
caches those, so a repeated workload is reentrant-quiet either way). A6's
cross-body memoization removes the need for this valve."
const DRIVER_REENTRANT_BUDGET = Base.RefValue(1_000)
const REENTRANT_ADMITTED = Base.Threads.Atomic{Int}(0)

":invoke emission switch (devirtualize_calls!)."
const DEVIRTUALIZE = Base.RefValue(true)

# Per-body inference budgets (v0): the driver re-infers each body's callee
# tree with a fresh state — the edge collector's soundness requires every
# consumed method-table/binding fact to be observed within this body's pass —
# so the budgets are deliberately tight. Depth/frame cutoffs resolve through
# the stock return_type oracle (fast, cached, and covered by the recorded
# match edge), trading callee-type precision for bounded per-body cost.
# Cross-body memoization with per-result edge replay is the A6 upgrade path.
const DRIVER_MAX_DEPTH = Base.RefValue(16)
const DRIVER_FRAME_BUDGET = Base.RefValue(3_000)
# Reentrant passes (nested driver work: the runtime compiling the driver's
# own code mid-pass, and devirtualization targets) run with narrower budgets:
# the same soundness protocol at lower callee-type precision, so the
# self-hosting burn-in costs a fraction of a root pass. Cutoffs stay sound
# (return_type oracle + recorded edges); A6's memoization removes the need.
const DRIVER_REENTRANT_MAX_DEPTH = Base.RefValue(4)
const DRIVER_REENTRANT_FRAME_BUDGET = Base.RefValue(400)

# ---------------------------------------------------------------------------
# One pipeline pass over one body
# ---------------------------------------------------------------------------

"A per-body decline: `reason` keys the ledger; `err` is retained evidence."
struct Fallback
    reason::Symbol
    err::Any
    Fallback(reason::Symbol, @nospecialize(err = nothing)) = new(reason, err)
end

"Everything one pipeline pass proves about a body (driver_infer's result)."
struct DriverResult
    src::Any                        # optimized CodeInfo (nothing when optimize=false)
    rt::Any                         # return lattice element (Const-precise)
    exct::Any
    effects::Compiler.Effects
    edges::Core.SimpleVector        # stock encoding (build_edges)
    valid_worlds::Compiler.WorldRange
    start_counter::UInt             # world counter at pass start
    rettype_const::Any
    const_flags::UInt8              # stock encoding: 0x2 rettype_const set, 0x3 const ABI
end

"""Per-axis upward refinement of the inference-time ipo effects with the
post-optimization recompute (stock `refine_effects!` semantics: an axis only
improves when the optimized body PROVES the better value — both computations
are sound for the emitted body, so taking the better bit per axis is too)."""
function refine_post_opt(base::Compiler.Effects, post::Compiler.Effects)
    return Compiler.Effects(base;
        consistent = post.consistent === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.consistent,
        effect_free = post.effect_free === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.effect_free,
        nothrow = base.nothrow | post.nothrow,
        terminates = base.terminates | post.terminates,
        notaskstate = base.notaskstate | post.notaskstate,
        inaccessiblememonly = post.inaccessiblememonly === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.inaccessiblememonly,
        noub = post.noub === Compiler.ALWAYS_TRUE ? Compiler.ALWAYS_TRUE :
            (post.noub === Compiler.NOUB_IF_NOINBOUNDS &&
             base.noub === Compiler.ALWAYS_FALSE ? Compiler.NOUB_IF_NOINBOUNDS :
             base.noub),
        nortcall = base.nortcall | post.nortcall)
end

"Encode the collector's records as a stock-format CodeInstance edges vector:
`user_edges` (a staged expansion's generator-declared edges, already in
stock encoding) first, then binding edges, per-lookup MethodMatchInfo
encodings (mi_edge=true, so match backedges land on MethodInstances), and
invoke edges (incl. the devirtualizer's CodeInstance targets)."
function build_edges(col::UEdges, @nospecialize(user_edges = nothing))
    edges = Any[]
    if user_edges !== nothing
        for e in user_edges
            push!(edges, e)
        end
    end
    for b in col.bindings
        push!(edges, b)
    end
    for (atype, result) in col.calls
        fullmatch = Base.any(m -> (m::Core.MethodMatch).fully_covers, result.matches)
        info = Compiler.MethodMatchInfo(result, Core.methodtable, atype, fullmatch)
        Compiler._add_edges_impl(edges, info, #=mi_edge=#true)
    end
    for (invokesig, target) in col.invokes
        if invokesig === nothing
            target isa Core.CodeInstance ? Compiler.add_one_edge!(edges, target) :
                                           Compiler.add_one_edge!(edges, target::Core.MethodInstance)
        else
            Compiler.add_invoke_edge!(edges, invokesig, target)
        end
    end
    return Core.svec(edges...)
end

"The stock inline_cost_model criteria over the driver's optimized IRCode
(a sane equivalent of compute_inlining_cost, so the stock inliner can
consume unified-produced CodeInstances when pipelines mix)."
function driver_inlining_cost(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                              src0::Core.CodeInfo, ircode, @nospecialize(rt))
    src0.inlining == 0x02 && return Compiler.MAX_INLINE_COST      # @noinline
    declared_inline = src0.inlining == 0x01
    sig = Base.unwrap_unionall(mi.specTypes)
    (sig isa DataType && sig.name === Tuple.name) || return Compiler.MAX_INLINE_COST
    !declared_inline && rt === Union{} && return Compiler.MAX_INLINE_COST
    if declared_inline && Base.isdispatchtuple(mi.specTypes)
        return Compiler.MIN_INLINE_COST
    end
    params = Compiler.OptimizationParams(interp)
    cost_threshold = params.inline_cost_threshold
    declared_inline && (cost_threshold += 19 * params.inline_cost_threshold)
    return try
        Compiler.inline_cost_model(ircode, params, Int(cost_threshold))
    catch
        Compiler.MAX_INLINE_COST
    end
end

"""
    driver_infer(interp, mi; optimize=true, emit_code=true)
        -> Union{DriverResult,Fallback}

One unified-pipeline pass over `mi`'s body. Pure with respect to the global
caches: nothing is cached or reserved here — the callers decide (the cache
entry wraps this with engine semantics; the reflection bridges use the
result directly). `optimize = false` stops after inference; `emit_code =
false` runs the optimizer (so rt/effects/exct see post-optimization
refinement, stock's `ipo_dataflow_analysis!` analog) but skips the
CodeInfo exit — for effects/exct queries, which need no code. `src` is
`nothing` in both reduced modes.
"""
function driver_infer(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance;
                      optimize::Bool = true, emit_code::Bool = true,
                      max_depth::Int = DRIVER_MAX_DEPTH[],
                      frame_budget::Int = DRIVER_FRAME_BUDGET[])
    world = Compiler.get_inference_world(interp)
    def = mi.def
    def isa Method || return Fallback(:toplevel)
    def.is_for_opaque_closure && return Fallback(:opaque_closure)
    Compiler.InferenceParams(interp).force_enable_inference && return Fallback(:trim)
    ccall(:jl_get_module_infer, Cint, (Any,), def.module) == 0 &&
        return Fallback(:inference_disabled)

    start_counter = Base.get_world_counter()
    col = UEdges(world)
    world <= start_counter || return Fallback(:world_unprovable)

    # Generated functions: `retrieve_code_info` expands the staged body
    # (jl_code_for_staged) — the expansion's validity window arrives as
    # `src.min_world/max_world` (clamped below, exactly stock InferenceState's
    # rule) and any generator-declared edges as `src.edges` (appended raw to
    # the CodeInstance edges, stock compute_edges!' user_edges rule; the
    # runtime registers them on the cached uninferred expansion as well).
    # The generator itself is ordinary user code: any compilation it needs
    # reenters the driver (bounded recursion) or stock. Its errors surface
    # as a per-body decline — the stock path reproduces stock's call-time
    # generator-error semantics.
    staged = isdefined(def, :generator)
    src0 = try
        Compiler.retrieve_code_info(mi, world)
    catch err
        return Fallback(staged ? :staged_source : :no_source, err)
    end
    src0 isa Core.CodeInfo || return Fallback(staged ? :staged_source : :no_source)
    clamp_world!(col, src0.min_world, src0.max_world)
    user_edges = src0.edges
    user_edges isa Core.SimpleVector && isempty(user_edges) && (user_edges = nothing)
    user_edges isa Vector{Any} && isempty(user_edges) && (user_edges = nothing)

    local uir
    try
        uir = codeinfo_to_ir(src0; nargs = Int(def.nargs), name = def.name)
    catch err
        err isa UnsupportedIR || return Fallback(:internal_error, err)
        return Fallback(:entry_convert, err)
    end
    uir.meta[:method_instance] = mi
    uir.meta[:mi] = mi
    uir.meta[:slotnames] = src0.slotnames
    uir.meta[:propagate_inbounds] = src0.propagate_inbounds
    uir.sptypes = Any[t for t in mi.sparam_vals]
    uir.meta[:sptypes_lat] = sptypes_lattice(mi)
    let und = sptypes_undef(mi)
        und === nothing || (uir.meta[:sptypes_undef] = und)
    end
    let reads = sparam_statement_reads(src0)
        isempty(reads) || (uir.meta[:sparam_reads] = reads)
    end

    st = UInferState(UInferConfig(; world,
        max_methods = Compiler.InferenceParams(interp).max_methods,
        max_depth, frame_budget))
    st.edges = col
    argl = method_arglattice(def, mi, Any[])
    argl === nothing && return Fallback(:arglattice)

    local rt, effects, exct
    try
        infer_ir!(uir, copy(argl); state = st)
        # the ipo effects baseline is the INFERENCE-time frame effects with
        # the method-level `@assume_effects` override (stock's finish order);
        # the optimizer's recompute below only REFINES it upward — a
        # recompute over the inlined body can lose callee-override precision
        # (inlining dissolves the callee frames the overrides applied to)
        effects = apply_effects_override(def, frame_effects_meta(uir))
        exct = get(uir.meta, :exct, Any)
        if optimize
            uir = optimize_ir!(uir, argl; state = st, inline = true)
            # stock's ipo_dataflow_analysis!/refine_effects! analog: the
            # post-optimization body (branches folded, dead throws gone)
            # re-inferred; upgrade any axis it proves
            effects = refine_post_opt(effects, frame_effects_meta(uir))
        end
        rt = get(uir.meta, :rettype, Any)
    catch err
        err isa UnsupportedIR || return Fallback(:inference_error, err)
        return Fallback(:inference_unsupported, err)
    end
    rt = sanitize_intercond(def, rt)
    rt isa UInterCond && (rt = Bool)
    Compiler.is_nothrow(effects) && (exct = Union{})

    src = nothing
    if optimize && emit_code
        # :invoke emission for residual statically-resolved calls (each
        # rewrite is individually sound, so a failure just leaves the
        # remaining sites as dynamic calls)
        if DEVIRTUALIZE[]
            try
                devirtualize_calls!(uir, st, interp)
            catch
            end
        end
        local ircode
        try
            ircode = ir_to_ircode(uir)
        catch err
            err isa UnsupportedIR || return Fallback(:exit_error, err)
            return Fallback(:typed_exit, err)
        end
        try
            nargs = Int(def.nargs)
            src = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
            slotnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), def.slot_syms)
            length(slotnames) < nargs && append!(slotnames,
                Symbol[Symbol("#arg", i) for i in (length(slotnames)+1):nargs])
            src.slotnames = slotnames
            src.slotflags = Base.fill(0x00, length(slotnames))
            src.slottypes = copy(ircode.argtypes)
            src.isva = def.isva
            src.nargs = UInt(nargs)
            ircode.debuginfo.def = mi
            Compiler.ir_to_codeinf!(src, ircode)
            src.rettype = CC.widenconst(rt)
            src.parent = mi
            src.min_world = col.valid_worlds.min_world
            src.max_world = col.valid_worlds.max_world
            src.inlining_cost = driver_inlining_cost(interp, mi, src0, ircode, CC.widenconst(rt))
        catch err
            return Fallback(:exit_error, err)
        end
    end

    col.ok || return Fallback(:world_unprovable)
    (col.valid_worlds.min_world <= world <= col.valid_worlds.max_world) ||
        return Fallback(:world_unprovable)
    edges = try
        build_edges(col, user_edges)
    catch err
        return Fallback(:internal_error, err)
    end
    src isa Core.CodeInfo && (src.edges = edges)

    rettype_const = nothing
    const_flags = 0x00
    if rt isa CC.Const
        rettype_const = rt.val
        constabi = Compiler.is_foldable_nothrow(effects) &&
                   Compiler.is_inlineable_constant(rt.val)
        const_flags = constabi ? 0x03 : 0x02
    elseif Compiler.isconstType(rt)
        rettype_const = Compiler.type_parameter(rt)
        const_flags = 0x02
    end

    return DriverResult(src, rt, exct, effects, edges, col.valid_worlds,
                        start_counter, rettype_const, const_flags)
end

# ---------------------------------------------------------------------------
# Devirtualization: statically-resolved residual calls become `:invoke`
# ---------------------------------------------------------------------------

"""
    driver_ci_for_invoke(interp, mi) -> Union{Nothing,CodeInstance}

A CodeInstance suitable as an `:invoke` target for `mi`: the world-covering
cache entry when one exists, else a recursive unified pass (per-task depth
bound and inflight set apply — mutual recursion and over-deep chains return
`nothing`, and the site degrades to a MethodInstance invoke, which the
runtime compiles on first call through the ordinary entry). No JIT work
happens here: the caller's `add_codeinsts_to_jit!` walk collects embedded
CodeInstance targets via `collectinvokes!`.
"""
function driver_ci_for_invoke(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance)
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance &&
           Compiler.ci_meets_requirement(interp, code, Compiler.SOURCE_MODE_ABI)
            return code
        end
    end
    dts = driver_task_state()
    (mi in dts.inflight || dts.depth >= DRIVER_REENTRY_LIMIT[]) && return nothing
    # eager production is bounded to ONE level, from ROOT passes only: a
    # nested (reentrant or production) pass embeds cached CodeInstances or
    # MethodInstance invokes. Without the root restriction the burn-in
    # compiles the STATIC call graph — far beyond the runtime-demand set —
    # eagerly; targets left as mi-invokes materialize (and cache) when the
    # runtime first needs them, so coverage converges by execution.
    (dts.devirt > 0 || dts.depth > 1) && return nothing
    local ci
    dts.depth += 1
    dts.devirt += 1
    push!(dts.inflight, mi)
    try
        ci = _unified_typeinf(interp, mi, Compiler.SOURCE_MODE_ABI)
    finally
        dts.depth -= 1
        dts.devirt -= 1
        delete!(dts.inflight, mi)
    end
    ci isa Core.CodeInstance || return nothing
    return ci
end

"""
    devirtualize_calls!(uir, st, interp) -> Int

Post-optimization `:invoke` emission (stock's inliner leaves
`Expr(:invoke, ci, ...)` at statically-resolved sites it does not inline):
for each residual `K"call"` whose signature — built from the final inferred
operand types, exactly what the last `infer_ir!` pass looked up — resolves
to a SINGLE, FULLY-COVERING method match in a world-clamped, edge-recorded
query (`resolve_single_match(st, sig)`), rewrite the statement to
`K"invoke"` targeting the callee's CodeInstance (produced through the
driver, bounded recursion) or, when a CI cannot be soundly produced right
now, the compilable MethodInstance. Soundness: the recorded match edge caps
this body's CodeInstance whenever the callee set changes, and the emitted
world bounds are additionally intersected with the callee CI's; sparams of
non-dispatch-tuple targets are re-derived per call by the runtime's invoke
convention, so the rewrite is dispatch-exact. Types/effects columns are
unchanged (the rewrite preserves semantics per statement).
"""
function devirtualize_calls!(uir, st::UInferState, interp::Compiler.AbstractInterpreter)
    n = 0
    col = st.edges
    for s in UnifiedIR.each_stmt(uir)
        UnifiedIR.is_tombstone(uir, s) && continue
        UnifiedIR.stmt_kind(uir, s) === K"call" || continue
        nop = UnifiedIR.nops(uir, s)
        nop >= 1 || continue
        args = Any[stmt_lattice(uir, UnifiedIR.getop(uir, s, i)) for i in 1:nop]
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
        f === nothing && continue
        (f isa Core.Builtin || f isa Core.IntrinsicFunction) && continue
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        Base.any(t -> t === Union{} || !(t isa Type) || CC.has_free_typevars(t), argts) && continue
        ft = f isa Type ? Type{f} : typeof(f)
        sig = try
            Tuple{ft, argts...}
        catch
            continue
        end
        match = resolve_single_match(st, sig)   # records the match edge
        match === nothing && continue
        match.fully_covers || continue
        mi = try
            CC.specialize_method(match)
        catch
            continue
        end
        mi isa Core.MethodInstance || continue
        target = ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)
        target isa Core.MethodInstance || continue
        ci = driver_ci_for_invoke(interp, target)
        tgt = ci === nothing ? target : ci
        if ci isa Core.CodeInstance && col isa UEdges
            # the embedded CI must cover every world this body claims
            clamp_world!(col, ci.min_world, ci.max_world) || continue
        end
        ops = UnifiedIR.Operand[UnifiedIR.vop(uir, tgt)]
        for i in 1:nop
            push!(ops, UnifiedIR.getop(uir, s, i))
        end
        UnifiedIR.replace_stmt!(uir, s, K"invoke", ops...;
                                type = UnifiedIR.stmt_type(uir, s),
                                flag = UnifiedIR.stmt_flag(uir, s))
        n += 1
    end
    return n
end

# ---------------------------------------------------------------------------
# The cache-grade entry (typeinf_ext_toplevel hook)
# ---------------------------------------------------------------------------

"Fill + publish a driver result following stock's finish!/promotecache!
sequence. Takes ownership of the engine-reserved `ci`."
function finish_unified!(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                         ci::Core.CodeInstance, result::DriverResult)
    valid_worlds = result.valid_worlds
    validation_world = Base.get_world_counter()
    if valid_worlds.max_world < validation_world
        # something moved (or was already bounded) during the pass: v0 never
        # publishes bounded CodeInstances (see the header comment)
        Compiler.engine_reject(interp, ci)
        count_fallback!(valid_worlds.max_world < result.start_counter ?
                        :world_bounded : :world_moved, mi)
        return nothing
    end
    src = result.src::Core.CodeInfo
    discard_src = result.const_flags == 0x03 && Compiler.may_discard_trees(interp)
    inferred = nothing
    debuginfo = nothing
    if !discard_src
        inferred = Compiler.maybe_compress_codeinfo(interp, mi, src)
        debuginfo = src.debuginfo
    end
    debuginfo === nothing && (debuginfo = Core.DebugInfo(mi))
    # all facts verified at validation_world: register invalidation edges
    Compiler.store_backedges(ci, result.edges)
    ipo = Compiler.encode_effects(result.effects)
    ccall(:jl_fill_codeinst, Cvoid,
          (Any, Any, Any, Any, Any, Int32, UInt, UInt, UInt32, Any,
           Float64, Float64, Float64, Any, Any),
          ci, CC.widenconst(result.rt), result.exct, result.rettype_const, inferred,
          Int32(result.const_flags), valid_worlds.min_world, valid_worlds.max_world,
          ipo, nothing, 0.0, 0.0, 0.0, debuginfo, result.edges)
    Compiler.code_cache(interp)[mi] = ci
    Compiler.engine_reject(interp, ci)          # fulfill: wake any waiters
    if !discard_src
        codegen = Compiler.codegen_cache(interp)
        codegen === nothing || (codegen[ci] = src)
    end
    ccall(:jl_promote_ci_to_current, Cvoid, (Any, UInt), ci, validation_world)
    note_unified!()
    return ci
end

function _unified_typeinf(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                          source_mode::UInt8)
    mi = ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)::Core.MethodInstance
    # fast cache path (stock typeinf_ext's)
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance && Compiler.ci_meets_requirement(interp, code, source_mode)
            return code
        end
    end
    ci = Compiler.engine_reserve(interp, mi)
    # check cache again if it is still new after reserving in the engine
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance && Compiler.ci_meets_requirement(interp, code, source_mode)
            Compiler.engine_reject(interp, ci)
            return code
        end
    end
    local result
    # nested passes (depth > 1: reentrant driver-code compiles and
    # devirtualization targets) run with the narrower budgets
    nested = driver_task_state().depth > 1
    max_depth = nested ? DRIVER_REENTRANT_MAX_DEPTH[] : DRIVER_MAX_DEPTH[]
    frame_budget = nested ? DRIVER_REENTRANT_FRAME_BUDGET[] : DRIVER_FRAME_BUDGET[]
    try
        result = driver_infer(interp, mi; max_depth, frame_budget)
        if result isa DriverResult && result.valid_worlds.max_world == result.start_counter &&
           Base.get_world_counter() > result.start_counter
            # the counter moved but no consulted fact was bounded below the
            # pass start: lazy binding/partition materialization (one bump
            # per binding per process). The bindings exist now — one retry
            # settles it.
            result = driver_infer(interp, mi; max_depth, frame_budget)
        end
        if result isa Fallback
            Compiler.engine_reject(interp, ci)
            count_fallback!(result.reason, mi, result.err)
            return nothing
        end
        return finish_unified!(interp, mi, ci, result::DriverResult)
    catch err
        # fallback discipline: NO unified-path error escapes the hook — the
        # reservation is released and stock compiles the body
        Compiler.engine_reject(interp, ci)
        count_fallback!(:internal_error, mi, err)
        return nothing
    end
end

"""
    unified_typeinf(interp::AbstractInterpreter, mi::MethodInstance, source_mode::UInt8)
        -> Union{Nothing,CodeInstance}

The `Compiler.UNIFIED_HOOKS.typeinf_ext_toplevel` implementation: the real
unified pipeline with stock cache/engine semantics. Returns `nothing` when
this body falls back (counted in `pipeline_stats()`); the caller —
`Compiler.typeinf_ext_toplevel` — then runs the stock path for it.
"""
function unified_typeinf(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                         source_mode::UInt8)
    dts = driver_task_state()
    if mi in dts.inflight
        # this task is already driving this exact body up-stack: recursing
        # can only redo the same pass against the engine placeholder
        count_fallback!(:reentrant_self)
        return nothing
    end
    if dts.depth >= DRIVER_REENTRY_LIMIT[]
        count_fallback!(:reentrant_depth)
        return nothing
    end
    if dts.depth > 0
        # reentrant request (the self-hosting burn-in): admit within the
        # session budget, else decline precisely — stock compiles + caches
        if REENTRANT_ADMITTED[] >= DRIVER_REENTRANT_BUDGET[]
            count_fallback!(:reentrant_budget)
            return nothing
        end
        Base.Threads.atomic_add!(REENTRANT_ADMITTED, 1)
    end
    local ci
    dts.depth += 1
    push!(dts.inflight, mi)
    try
        ci = _unified_typeinf(interp, mi, source_mode)
    finally
        dts.depth -= 1
        delete!(dts.inflight, mi)
    end
    ci isa Core.CodeInstance || return nothing
    # stock typeinf_ext_toplevel's JIT closure (needs no unified state; may
    # stock-infer un-JIT'd invoke targets)
    return Compiler.add_codeinsts_to_jit!(interp, ci, source_mode)
end

# ---------------------------------------------------------------------------
# Reflection bridges (typeinf_code / _infer_effects / _infer_exception_type)
# ---------------------------------------------------------------------------

# run `f(...)` under the driver's per-task depth accounting, declining
# (nothing) at the reentrancy bound or on any escaped unified-path error
# (the fallback discipline: the hook caller must always be able to continue
# on stock). Nested driver work stays bounded and every pass still builds
# its own fresh state.
function with_driver_guard(f)
    dts = driver_task_state()
    if dts.depth >= DRIVER_REENTRY_LIMIT[]
        count_fallback!(:reentrant_depth)
        return nothing
    end
    dts.depth += 1
    try
        return f()
    catch err
        count_fallback!(:internal_error, nothing, err)
        return nothing
    finally
        dts.depth -= 1
    end
end

"""
    unified_typeinf_code(interp, mi, run_optimizer) -> Union{Nothing,CodeInfo}

The `typeinf_code` bridge: `code_typed`/`@code_typed`/`code_warntype` show
the unified pipeline's optimized output. Like stock, a const-ABI result
renders as the synthetic `return <const>` CodeInfo. Unoptimized queries
(`optimize=false`) stay on stock (that view is representation-independent).
"""
function unified_typeinf_code(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                              run_optimizer::Bool)
    run_optimizer || return nothing   # uninferred/unoptimized view: stock
    return with_driver_guard() do
        result = driver_infer(interp, mi)
        if result isa Fallback
            count_fallback!(result.reason, mi, result.err)
            return nothing
        end
        note_unified!()
        if result.const_flags == 0x03 && Compiler.may_discard_trees(interp)
            return Compiler.codeinfo_for_const(interp, mi, result.valid_worlds,
                                               result.edges, result.rettype_const)
        end
        return result.src::Core.CodeInfo
    end
end

"""
    unified_infer_effects(interp, tt, optimize) -> Union{Nothing,Effects}

The `_infer_effects` bridge (`Base.infer_effects`): per-match driver
inference merged with stock's MethodError accounting. `optimize` mirrors
stock's `typeinf_frame(...; run_optimizer)` semantics — the optimizer runs
(post-opt effects refinement) but no code is emitted. Declines whole-query
on any per-match fallback.
"""
function unified_infer_effects(interp::Compiler.AbstractInterpreter, @nospecialize(tt),
                               optimize::Bool)
    return with_driver_guard() do
        matches = Compiler.findall(tt, Compiler.method_table(interp))
        matches === nothing && return nothing
        effects = Compiler.EFFECTS_TOTAL
        if Compiler._may_throw_methoderror(matches)
            effects = Compiler.Effects(effects; nothrow = false)
        end
        for match in matches.matches
            match = match::Core.MethodMatch
            result = driver_infer(interp, Compiler.specialize_method(match);
                                  optimize, emit_code = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            note_unified!()
            effects = Compiler.merge_effects(effects, result.effects)
        end
        return effects
    end
end

"""
    unified_infer_exception_type(interp, tt, optimize) -> Union{Nothing,Type}

The `_infer_exception_type` bridge: the per-match frame exception-type
bestguess (the thrown-escape join tracked by inference, `Union{}` for
proven-nothrow bodies), plus stock's MethodError account.
"""
function unified_infer_exception_type(interp::Compiler.AbstractInterpreter, @nospecialize(tt),
                                      optimize::Bool)
    return with_driver_guard() do
        matches = Compiler.findall(tt, Compiler.method_table(interp))
        matches === nothing && return nothing
        exct = Union{}
        if Compiler._may_throw_methoderror(matches)
            exct = MethodError
        end
        for match in matches.matches
            match = match::Core.MethodMatch
            result = driver_infer(interp, Compiler.specialize_method(match);
                                  optimize, emit_code = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            note_unified!()
            exct = CC.tmerge(CC.fallback_lattice, exct, result.exct)
        end
        return CC.widenconst(exct)
    end
end

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

"""
    enable_pipeline!() -> Nothing

Install the unified driver behind the Compiler module's standard entry
points (`Compiler.UNIFIED_HOOKS`): `typeinf_ext_toplevel`, `typeinf_code`,
`_infer_effects` and `_infer_exception_type` route NativeInterpreter
requests through the unified pipeline, falling back to stock per body
(`pipeline_stats()`). Combined with `@activate Compiler`-style reflection
activation, `code_typed`/`infer_effects` show unified results; combined
with [`activate!`](@ref)'s jl_set_typeinf_func flip, ALL runtime inference
routes here. Undo with [`disable_pipeline!`](@ref).
"""
function enable_pipeline!()
    Compiler.UNIFIED_HOOKS[] = Compiler.UnifiedHooks(
        unified_typeinf, unified_typeinf_code,
        unified_infer_effects, unified_infer_exception_type)
    return nothing
end

"Remove the driver from `Compiler.UNIFIED_HOOKS`: stock behavior, bit-identical."
function disable_pipeline!()
    Compiler.UNIFIED_HOOKS[] = nothing
    return nothing
end

pipeline_enabled() = Compiler.UNIFIED_HOOKS[] !== nothing
