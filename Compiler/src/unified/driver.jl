# The real typeinf driver (COMPILER-PORT-PLAN A1): `unified_typeinf` runs the
# native unified pipeline — entry-convert → infer_ir! → optimize_ir! →
# ir_to_ircode → CodeInfo — behind the Compiler module's standard entry
# points, producing cache-grade CodeInstances with stock-encoded edges and
# sound world bounds. Every body the pipeline cannot (yet) handle falls back
# to the stock compiler, per body, with a counted reason (`pipeline_stats()`
# is the ratchet). Installed via `enable_pipeline!` (Compiler.UNIFIED_HOOKS);
# `activate!` additionally flips the runtime's jl_typeinf_func.
#
# Concurrency (v0, documented choice): one driver pass at a time. Reentrant
# requests (the driver's own code compiling through the global hook) and
# concurrent requests from other tasks decline immediately to stock —
# `trylock` rather than `lock`, because blocking here while stock inference
# holds engine reservations on another thread could deadlock. The unified
# inference state is fresh per pass, so this also keeps every method-table/
# binding fact a pass consumes inside one collector.
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

function count_fallback!(reason::Symbol, @nospecialize(mi = nothing), @nospecialize(err = nothing))
    d = PIPELINE_STATS.fallbacks
    d[reason] = get(d, reason, 0) + 1
    err === nothing || (PIPELINE_STATS.last_error = (reason, mi, err))
    return nothing
end

"""
    pipeline_stats() -> NamedTuple

The driver's ledger: `unified` counts bodies compiled end-to-end by the
unified pipeline, `fallbacks` maps fallback reason to count (those bodies
were handled by the stock compiler), `last_error` retains the most recent
`(reason, mi, exception)` for error-class fallbacks.
"""
pipeline_stats() = (; unified = PIPELINE_STATS.unified,
                    fallbacks = copy(PIPELINE_STATS.fallbacks),
                    last_error = PIPELINE_STATS.last_error)

function reset_pipeline_stats!()
    PIPELINE_STATS.unified = 0
    empty!(PIPELINE_STATS.fallbacks)
    PIPELINE_STATS.last_error = nothing
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
# Reentrancy / concurrency guard
# ---------------------------------------------------------------------------

const DRIVER_LOCK = Base.ReentrantLock()
const DRIVER_ACTIVE = Base.RefValue(false)

# Per-body inference budgets (v0): the driver re-infers each body's callee
# tree with a fresh state — the edge collector's soundness requires every
# consumed method-table/binding fact to be observed within this body's pass —
# so the budgets are deliberately tight. Depth/frame cutoffs resolve through
# the stock return_type oracle (fast, cached, and covered by the recorded
# match edge), trading callee-type precision for bounded per-body cost.
# Cross-body memoization with per-result edge replay is the A6 upgrade path.
const DRIVER_MAX_DEPTH = Base.RefValue(16)
const DRIVER_FRAME_BUDGET = Base.RefValue(3_000)

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

"UnifiedIR frame effect mask -> Compiler.Effects (conservative on every axis
the unified flags do not model)."
function effects_from_mask(mask::UInt32)
    return Compiler.Effects(Compiler.EFFECTS_UNKNOWN;
        consistent = (mask & UnifiedIR.FLAG_CONSISTENT) != 0 ?
            Compiler.ALWAYS_TRUE : Compiler.ALWAYS_FALSE,
        effect_free = (mask & UnifiedIR.FLAG_EFFECT_FREE) != 0 ?
            Compiler.ALWAYS_TRUE : Compiler.ALWAYS_FALSE,
        nothrow = (mask & UnifiedIR.FLAG_NOTHROW) != 0,
        terminates = (mask & UnifiedIR.FLAG_TERMINATES) != 0)
end

"Encode the collector's records as a stock-format CodeInstance edges vector:
Binding edges, then per-lookup MethodMatchInfo encodings (mi_edge=true, so
backedges land on MethodInstances — the unified pipeline creates no callee
CodeInstances), then invoke edges."
function build_edges(col::UEdges)
    edges = Any[]
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
    driver_infer(interp, mi; optimize=true) -> Union{DriverResult,Fallback}

One unified-pipeline pass over `mi`'s body. Pure with respect to the global
caches: nothing is cached or reserved here — the callers decide (the cache
entry wraps this with engine semantics; the reflection bridges use the
result directly). `optimize = false` stops after inference (effects/exct
queries; `src` is `nothing` then).
"""
function driver_infer(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance;
                      optimize::Bool = true)
    world = Compiler.get_inference_world(interp)
    def = mi.def
    def isa Method || return Fallback(:toplevel)
    isdefined(def, :generator) && return Fallback(:generated)
    Compiler.InferenceParams(interp).force_enable_inference && return Fallback(:trim)
    ccall(:jl_get_module_infer, Cint, (Any,), def.module) == 0 &&
        return Fallback(:inference_disabled)

    start_counter = Base.get_world_counter()
    col = UEdges(world)
    world <= start_counter || return Fallback(:world_unprovable)

    src0 = try
        Compiler.retrieve_code_info(mi, world)
    catch err
        return Fallback(:no_source, err)
    end
    src0 isa Core.CodeInfo || return Fallback(:no_source)
    clamp_world!(col, src0.min_world, src0.max_world)

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
    uir.sptypes = Any[t for t in mi.sparam_vals]
    uir.meta[:sptypes_lat] = sptypes_lattice(mi)

    st = UInferState(UInferConfig(; world,
        max_methods = Compiler.InferenceParams(interp).max_methods,
        max_depth = DRIVER_MAX_DEPTH[],
        frame_budget = DRIVER_FRAME_BUDGET[]))
    st.edges = col
    argl = method_arglattice(def, mi, Any[])
    argl === nothing && return Fallback(:arglattice)

    local rt
    try
        infer_ir!(uir, copy(argl); state = st)
        optimize && (uir = optimize_ir!(uir, argl; state = st, inline = true))
        rt = get(uir.meta, :rettype, Any)
    catch err
        err isa UnsupportedIR || return Fallback(:inference_error, err)
        return Fallback(:inference_unsupported, err)
    end
    rt = sanitize_intercond(def, rt)
    rt isa UInterCond && (rt = Bool)
    mask = apply_effects_override(def, get(uir.meta, :effects, EFFECTS_NONE)::UInt32)
    effects = effects_from_mask(mask)
    exct = Compiler.is_nothrow(effects) ? Union{} : Any

    src = nothing
    if optimize
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
        build_edges(col)
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
    PIPELINE_STATS.unified += 1
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
    result = driver_infer(interp, mi)
    if result isa DriverResult && result.valid_worlds.max_world == result.start_counter &&
       Base.get_world_counter() > result.start_counter
        # the counter moved but no consulted fact was bounded below the pass
        # start: lazy binding/partition materialization (one bump per binding
        # per process). The bindings exist now — one retry settles it.
        result = driver_infer(interp, mi)
    end
    if result isa Fallback
        Compiler.engine_reject(interp, ci)
        count_fallback!(result.reason, mi, result.err)
        return nothing
    end
    return finish_unified!(interp, mi, ci, result::DriverResult)
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
    if DRIVER_ACTIVE[]
        count_fallback!(:reentrant)
        return nothing
    end
    if !Base.trylock(DRIVER_LOCK)
        count_fallback!(:concurrent)
        return nothing
    end
    local ci
    try
        DRIVER_ACTIVE[] = true
        ci = _unified_typeinf(interp, mi, source_mode)
    finally
        DRIVER_ACTIVE[] = false
        Base.unlock(DRIVER_LOCK)
    end
    ci isa Core.CodeInstance || return nothing
    # stock typeinf_ext_toplevel's JIT closure (needs no unified state; may
    # stock-infer un-JIT'd invoke targets)
    return Compiler.add_codeinsts_to_jit!(interp, ci, source_mode)
end

# ---------------------------------------------------------------------------
# Reflection bridges (typeinf_code / _infer_effects / _infer_exception_type)
# ---------------------------------------------------------------------------

# run `f(...)` under the driver guard, declining (nothing) on reentrance
function with_driver_guard(f)
    if DRIVER_ACTIVE[]
        count_fallback!(:reentrant)
        return nothing
    end
    if !Base.trylock(DRIVER_LOCK)
        count_fallback!(:concurrent)
        return nothing
    end
    try
        DRIVER_ACTIVE[] = true
        return f()
    finally
        DRIVER_ACTIVE[] = false
        Base.unlock(DRIVER_LOCK)
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
        PIPELINE_STATS.unified += 1
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
inference merged with stock's MethodError accounting. Declines whole-query
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
                                  optimize = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            PIPELINE_STATS.unified += 1
            effects = Compiler.merge_effects(effects, result.effects)
        end
        return effects
    end
end

"""
    unified_infer_exception_type(interp, tt, optimize) -> Union{Nothing,Type}

The `_infer_exception_type` bridge: v0 honesty — `Union{}` for bodies the
pipeline proves nothrow, `Any` otherwise, plus stock's MethodError account.
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
                                  optimize = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            PIPELINE_STATS.unified += 1
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
