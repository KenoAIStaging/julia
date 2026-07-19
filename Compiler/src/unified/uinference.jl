# The inference port (§10.3): abstract interpretation running natively on
# UnifiedIR — a structured fixed-point walk over the region tree (no
# reconstruction of basic blocks outside `cfg` islands). Reuses the Compiler
# package's lattice elements and tfuncs (Const, tmerge, builtin_tfunction);
# the IR-shape-dependent walker is what this file replaces.
#
# §10.3 mappings implemented (v1 scope):
#   (a) diverging-arm refinement: arms whose every exit is return/unreachable
#       contribute nothing to the if's result join;
#   (b) irinterp edge-killing: Const conditions select a single arm during
#       inference (the surgery form lives in fold_constant_branches!);
#   (c) backedge refinement: loop carried-arg states are joined from init and
#       `continue` values with bounded widening.

const CC = Compiler

struct UInferConfig
    world::UInt
    max_methods::Int
    max_depth::Int
    max_loop_iter::Int
    native_fallback::Bool   # delegate callees outside the entry-converter
                            # feature matrix to stock inference (documented seam)
    frame_budget::Int       # frames per top-level query (cutoffs resolve
                            # through native_fallback); default = the
                            # historical FRAME_BUDGET constant
    interp::CC.NativeInterpreter
end
function UInferConfig(; world::UInt = Base.get_world_counter(),
                      max_methods::Int = 3, max_depth::Int = 128,
                      max_loop_iter::Int = 8, native_fallback::Bool = true,
                      frame_budget::Int = FRAME_BUDGET)
    UInferConfig(world, max_methods, max_depth, max_loop_iter, native_fallback,
                 frame_budget, CC.NativeInterpreter(world))
end

mutable struct UInferStats
    frames::Int
    native_fallbacks::Int
    cycles::Int
end

"""
    UEdges

Edge and world-bound collector for cache-grade inference (the driver,
driver.jl). When attached to a `UInferState`, every fact the inference reads
from mutable global state — method-table queries, `invoke` target
resolutions, global-binding reads, staged-source expansions — is recorded
together with the world range it is valid for; `valid_worlds` is the running
intersection. The driver encodes the records into the stock CodeInstance
`edges` format (so `store_backedges`/invalidation work unchanged) and uses
the intersection as the CodeInstance's world bounds. `ok = false` means some
consulted fact could not be bounded (or the ranges became disjoint): the
result is NOT cacheable and the driver must fall back to stock.

Without a collector (`state.edges === nothing`, all pre-driver users), every
gated site keeps its historical behavior byte-for-byte.
"""
mutable struct UEdges
    world::UInt
    valid_worlds::CC.WorldRange
    ok::Bool
    calls::Vector{Any}                    # (atype, CC.MethodLookupResult) in discovery order
    callindex::Dict{Any,Int}              # atype -> index into calls (dedup)
    invokes::Vector{Any}                  # (invokesig, MethodInstance|CodeInstance)
    bindings::Vector{Core.Binding}        # order-preserving, dedup'd
    bindingset::Base.IdSet{Core.Binding}
    globmemo::Dict{Tuple{Module,Symbol},Any}  # partition-read lattice memo (one
                                              # pass reads each global many times;
                                              # sound: a partition change bumps the
                                              # world counter, which the driver's
                                              # finish protocol detects)
    # Cross-request memo support (driver.jl's global memo, A6): an append-only
    # per-request FACT TRACE — one event per consulted mutable-global-state
    # fact, INCLUDING re-consults the edge storage dedups away — so a callee
    # frame's trace window is exactly the fact set its result depends on.
    # Event encodings (tuples):
    #   (0x1, atype, result::MethodLookupResult, limit)   method-table lookup
    #   (0x2, mod, name, rte::RTEffects)                  binding-partition read
    #   (0x3, ci::CodeInstance)                           CodeInstance result read
    #   (0x4, sig, rt)                                    stock return_type oracle answer
    #   (0x5, lo, hi)                                     span reference: this request's
    #                                                     trace events lo..hi (a
    #                                                     per-request cache hit)
    # `spans` maps a per-request cache key (mi or const key) to the trace
    # window that justifies its cached result; `poison` counts consulted facts
    # the trace CANNOT represent (frames whose window saw one are excluded
    # from the global memo — the per-request collector still handles them).
    trace::Vector{Any}
    spans::Dict{Any,Tuple{Int,Int}}
    poison::Int
end
UEdges(world::UInt) =
    UEdges(world, CC.WorldRange(UInt(1), Base.get_world_counter()), true,
           Any[], Dict{Any,Int}(), Any[], Core.Binding[], Base.IdSet{Core.Binding}(),
           Dict{Tuple{Module,Symbol},Any}(), Any[], Dict{Any,Tuple{Int,Int}}(), 0)

"Intersect the collector's valid range with `[minw, maxw]`; a disjoint range
or one that no longer covers the inference world marks the collector unsound."
function clamp_world!(col::UEdges, minw::UInt, maxw::UInt)
    lo = max(col.valid_worlds.min_world, minw)
    hi = min(col.valid_worlds.max_world, maxw)
    if lo > hi || !(lo <= col.world <= hi)
        col.ok = false
        return false
    end
    col.valid_worlds = CC.WorldRange(lo, hi)
    return true
end
clamp_world!(col::UEdges, wr) = clamp_world!(col, wr.min_world, wr.max_world)

function record_call!(col::UEdges, @nospecialize(atype), result)
    clamp_world!(col, result.valid_worlds)
    if !haskey(col.callindex, atype)
        push!(col.calls, (atype, result))
        col.callindex[atype] = length(col.calls)
    end
    return nothing
end

function record_invoke!(col::UEdges, @nospecialize(invokesig), @nospecialize(target))
    push!(col.invokes, (invokesig, target))
    return nothing
end

function record_binding!(col::UEdges, b::Core.Binding)
    if !(b in col.bindingset)
        push!(col.bindingset, b)
        push!(col.bindings, b)
    end
    return nothing
end

# --- cross-request memo fact-trace helpers (see the UEdges field comment) ---

@inline trace!(col::UEdges, @nospecialize(ev)) = (push!(col.trace, ev); nothing)
@inline trace!(::Nothing, @nospecialize(ev)) = nothing

"Mark the current frame windows as containing a fact the trace cannot
represent: every enclosing frame becomes ineligible for the global memo."
@inline memo_poison!(col::UEdges) = (col.poison += 1; nothing)
@inline memo_poison!(::Nothing) = nothing

mutable struct UInferState
    cfg::UInferConfig
    cache::Dict{Core.MethodInstance,Any}        # mi -> UResult (rettype + effects)
    active::Dict{Core.MethodInstance,Int}       # cycle detection
    cycle_hit::Set{Core.MethodInstance}         # frames whose stale value was read
    stats::UInferStats
    constcache::Dict{Any,Any}                   # (mi, const-arg key) -> UResult
    budget_mark::Int                            # stats.frames at top-level query entry
    limited::Int                                # depth/budget cutoffs (taint counter)
    scratch::Dict{Any,Any}                      # per-top-level-query memo for
                                                # cutoff-tainted results
    cycle_scratch::Dict{Any,Any}                # transient memo for results that
                                                # depend on a stale approximation
                                                # (per-fixpoint-pass; the SCC's
                                                # membership table for the pass)
    scc_prev::Dict{Any,Any}                     # SCC joint-fixpoint state: last
                                                # pass's tmerge-accumulated result
                                                # per member (mi or const key);
                                                # seeds nested cycle roots and
                                                # detects joint convergence
    stale_depth::Int                            # min active-stack depth of any
                                                # outstanding stale (cycle) read
    stale_events::Int                           # stale-read event counter
    cyscr_hits::Int                             # cycle-scratch consumption counter
    nonconverged::Int                           # non-converged fixpoint exits
    resolutions::Int                            # resolved-cycle epoch (Bottom
                                                # scratch entries expire on bump)
    edges::Union{Nothing,UEdges}                # driver-mode edge/world collector
end
UInferState(cfg::UInferConfig = UInferConfig()) =
    UInferState(cfg, Dict{Core.MethodInstance,Any}(), Dict{Core.MethodInstance,Int}(),
                Set{Core.MethodInstance}(), UInferStats(0, 0, 0), Dict{Any,Any}(), 0, 0,
                Dict{Any,Any}(), Dict{Any,Any}(), Dict{Any,Any}(), typemax(Int), 0, 0, 0, 0,
                nothing)

"""Serve a per-request cached result to the trace: reference the span that
justified it, so enclosing frames' windows stay fact-complete. A hit whose
key has no recorded span (context-tainted results) poisons the window
instead."""
function memo_note_hit!(st::UInferState, @nospecialize(key))
    col = st.edges
    col === nothing && return nothing
    sp = get(col.spans, key, nothing)
    sp === nothing ? (col.poison += 1) : push!(col.trace, (0x5, sp[1], sp[2]))
    return nothing
end

⊔(st::UInferState, @nospecialize(a), @nospecialize(b)) =
    a === nothing ? b :
    b === nothing ? a : CC.tmerge(CC.fallback_lattice, widenucond(a), widenucond(b))

"""
    UCond

The port of `Core.Compiler.Conditional` (§10.3): a Bool-valued lattice
element that carries type refinements for a *subject* — a cell (slot analog)
or an SSA statement — on the true/false edges of a branch.
"""
struct UCond
    subject::Tuple{Symbol,Int32}     # (:cell, id) | (:stmt, id)
    thentype::Any
    elsetype::Any
end
"""
    UInterCond

The port of `Core.Compiler.InterConditional` (§10.3): a frame's Bool return
value that refines one of its *parameters* (by position in the root region's
args, 1 = the function itself). Context-free — safe to cache in `UResult`s —
and translated back to a caller-local `UCond` at each call site when the
caller passed a refinable subject in that position.
"""
struct UInterCond
    slot::Int
    thentype::Any
    elsetype::Any
end

widenucond(@nospecialize(t)) = (t isa UCond || t isa UInterCond) ? Bool : t
const RefMap = Dict{Tuple{Symbol,Int32},Any}

lat_eq(@nospecialize(a), @nospecialize(b)) =
    a === b || (CC.:⊑(CC.fallback_lattice, a, b) && CC.:⊑(CC.fallback_lattice, b, a))

"UInterCond-aware equality (⊑ is not defined on inter-conditionals)."
ulat_eq(@nospecialize(a), @nospecialize(b)) =
    (a isa UInterCond || b isa UInterCond) ?
        (a isa UInterCond && b isa UInterCond && a.slot == b.slot &&
         lat_eq(a.thentype, b.thentype) && lat_eq(a.elsetype, b.elsetype)) :
        lat_eq(a, b)

"UInterCond-aware tmerge for interprocedural result accumulation."
function umerge(@nospecialize(a), @nospecialize(b))
    a === Union{} && return b
    b === Union{} && return a
    if a isa UInterCond && b isa UInterCond && a.slot == b.slot
        return UInterCond(a.slot,
                          CC.tmerge(CC.fallback_lattice, a.thentype, b.thentype),
                          CC.tmerge(CC.fallback_lattice, a.elsetype, b.elsetype))
    end
    return CC.tmerge(CC.fallback_lattice, widenucond(a), widenucond(b))
end

# ---------------------------------------------------------------------------
# Frame inference over one IR body
# ---------------------------------------------------------------------------

mutable struct Frame
    ir::UnifiedIR.IR
    st::UInferState
    env::Vector{Any}                  # stmt id -> lattice element
    celltypes::Dict{Int32,Any}
    cells_changed::Bool
    rettype::Any                      # accumulated return-type join (nothing = none)
    continue_vals::Dict{Int32,Any}    # loop body region -> joined carried Vector{Any}
    break_vals::Dict{Int32,Any}       # loop body region -> joined result lattice
    reached::Set{Int32}               # blocks reached via cross-island gotos (§5.5)
    refinements::Vector{RefMap}       # active Conditional refinement scopes
    effects::CC.Effects               # frame ipo-effects accumulator (§5.1 rule 5,
                                      # merged via CC.merge_effects — the full stock currency)
    stmt_effects::Vector{UInt32}      # per-stmt effect masks (sentinel = untouched;
                                      # the 4 FLAG_* bits, the IR-pass/DCE channel)
    thrown::Vector{Any}               # exception-type collectors: [1] is the frame's
                                      # escape join; try bodies push/pop scopes so
                                      # caught exceptions type handler args instead
    override::Base.EffectsOverride    # the frame method's @assume_effects bits
    newed_cells::Set{Int32}           # cells with a cell_new (maybe-undef reads)
    pending_refine::Any               # (subject => lattice) from a typeassert, or nothing
    # §5.7 descent state (the late pipeline): monotone per-closure maps driven
    # by the same `cells_changed` flag and capped by the same widening
    # escalator as `celltypes` — one fixpoint, no second lattice
    closure_args::Dict{Int32,Vector{Any}} # closure stmt -> param joins over visible calls
    closure_rets::Dict{Int32,Any}         # closure stmt -> body return-type join
    closure_effs::Dict{Int32,CC.Effects}  # closure stmt -> body effects
    closure_escaped::Set{Int32}           # closures with a use besides call-callee
    closure_shifted::Set{Int32}           # closures with a world barrier before a call
    poisoned_cells::Set{Int32}            # shared cells whose reads must stay Any
    bc_guarded::Set{Int32}                # boundscheck stmts whose every use is the
                                          # boundscheck argument of a memory builtin
                                          # (their value cannot reach the result:
                                          # no frame consistency taint — the stock
                                          # post-opt boundscheck rule, at inference)
    exc_read::Set{Int32}                  # handler regions reading their exception
                                          # value (stock's :the_exception consistency taint)
    propagate_inbounds::Bool              # src.propagate_inbounds (meta)
    # per-statement `@inbounds`/`@assume_effects` context is entry-carried in
    # the flag column (codeinfo_entry.jl carry_ssaflags; accessors
    # stmt_inbounds / stmt_effects_override)
end

"""May the `latestworld` statement `L` execute after the creation of closure
`C`? The §5.8 world-split discipline for deferred bodies: a barrier between
creation and a call site means the body may run against a newer
method/binding table than the one inference consulted, so the closure is
unrefinable. Same-activation pairs use the capture criterion-(b) position
machinery (forward order + reach, shared loops always hazardous — no
fresh-cell cancellation applies to world state); a deferred-resident barrier
can run at any time relative to anything (v1 conservatism); a home barrier
against a nested closure applies to its outermost home creation site."""
function world_hazard(ir::UnifiedIR.IR, L::StmtId, C::StmtId)
    arL = UnifiedIR.activation_root(ir, UnifiedIR.stmt_region(ir, L))
    arC = UnifiedIR.activation_root(ir, UnifiedIR.stmt_region(ir, C))
    if arL != arC
        UnifiedIR.getregion(ir, arL).activation === UnifiedIR.ACT_DEFERRED && return true
        anchor = UnifiedIR._home_site(ir, C, arL)
        UnifiedIR.isnull(anchor) && return true
        C = anchor
    end
    UnifiedIR.isnull(UnifiedIR._innermost_shared_body(ir, L, C)) || return true
    return UnifiedIR.comes_before(ir, C, L) && UnifiedIR._may_reach(ir, C, L)
end

"The frame method's `@assume_effects` override bits (all-false when unknown)."
function frame_override(ir::UnifiedIR.IR)
    mi = get(ir.meta, :mi, nothing)
    if mi isa Core.MethodInstance && mi.def isa Method
        return try
            CC.decode_effects_override((mi.def::Method).purity)
        catch
            Base.EffectsOverride()
        end
    end
    return Base.EffectsOverride()
end

"Builtins that take a trailing boundscheck argument (`getfield_boundscheck`/
`memoryop_noub` subjects)."
function is_boundscheck_callee(@nospecialize(f))
    return f === Core.getfield || f === Core.memoryrefnew || f === Core.memoryrefget ||
           f === Core.memoryrefset! || f === Core.memoryrefunset! ||
           f === Core.memoryref_isassigned
end

"Frame constructor with empty analysis state (transfers.jl defines the masks)."
function Frame(ir::UnifiedIR.IR, st::UInferState, env::Vector{Any})
    fr = Frame(ir, st, env, Dict{Int32,Any}(), false, nothing, Dict{Int32,Any}(),
               Dict{Int32,Any}(), Set{Int32}(), RefMap[], CC.EFFECTS_TOTAL,
               fill(~UInt32(0), length(env)), Any[Union{}], frame_override(ir),
               Set{Int32}(), nothing,
               Dict{Int32,Vector{Any}}(), Dict{Int32,Any}(), Dict{Int32,CC.Effects}(),
               Set{Int32}(), Set{Int32}(), Set{Int32}(), Set{Int32}(), Set{Int32}(),
               get(ir.meta, :propagate_inbounds, false) === true)
    # One structural scan (positions do not change during inference):
    #   - escape discipline (§5.7): a closure value that flows anywhere but
    #     the callee position of a call escapes — unknown callers, and after
    #     materialization any holder can set the closure's untyped mutable
    #     fields, so params AND its captured cells' reads degrade;
    #   - capturing closures per shared cell (the deferred owners between
    #     each cell op and the cell's home activation);
    #   - world barriers (`latestworld`) for the shifted-closure rule.
    lws = StmtId[]
    closures = StmtId[]
    cellcaps = Dict{Int32,Set{Int32}}()   # shared cell -> capturing closure ids
    bcs = Set{Int32}()                    # K"boundscheck" stmt ids
    bcdirty = Set{Int32}()                # ...with a use outside bc-arg position
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        if k === K"cell_new"
            push!(fr.newed_cells, UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id)
        elseif k === K"latestworld"
            push!(lws, s)
        elseif k === K"closure"
            push!(closures, s)
        elseif k === K"boundscheck"
            push!(bcs, s.id)
        end
        iscellop = k === K"cell_get" || k === K"cell_set" ||
                   k === K"cell_new" || k === K"cell_isdefined"
        for j in 1:UnifiedIR.nops(ir, s)
            o = UnifiedIR.getop(ir, s, j)
            UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || continue
            d = UnifiedIR.asstmt(o)
            dk = UnifiedIR.stmt_kind(ir, d)
            if dk === K"boundscheck"
                # a use is clean when it is the trailing boundscheck argument
                # of a boundscheck-taking memory builtin; anything else lets
                # the inlining-context-dependent value flow
                (k === K"call" && j == UnifiedIR.nops(ir, s) && j >= 4 &&
                 is_boundscheck_callee(static_operand_value(ir, UnifiedIR.getop(ir, s, 1)))) ||
                    push!(bcdirty, d.id)
            elseif dk === K"region_arg"
                # a use of a handler's exception argument: the caught value's
                # identity is inconsistent (stock's :the_exception rule)
                rid = UnifiedIR.stmt_region(ir, d)
                reg = UnifiedIR.getregion(ir, rid)
                reg.kind === UnifiedIR.REGION_HANDLER && push!(fr.exc_read, rid.id)
            end
            if dk === K"closure"
                (k === K"call" && j == 1) || push!(fr.closure_escaped, d.id)
            elseif dk === K"cell_shared"
                if iscellop && j == 1
                    caps = get!(() -> Set{Int32}(), cellcaps, d.id)
                    home = UnifiedIR.activation_root(ir, UnifiedIR.stmt_region(ir, d))
                    r = UnifiedIR.stmt_region(ir, s)
                    while !UnifiedIR.isnull(r) && r != home
                        reg = UnifiedIR.getregion(ir, r)
                        if reg.activation === UnifiedIR.ACT_DEFERRED &&
                           !UnifiedIR.isnull(reg.owner)
                            push!(caps, reg.owner.id)
                        end
                        r = reg.parent
                    end
                else
                    # a shared cell used outside the cell-op vocabulary
                    # escapes as a value: its contents are unknowable
                    push!(fr.poisoned_cells, d.id)
                end
            end
        end
    end
    if !isempty(lws)
        for c in closures
            any(L -> world_hazard(ir, L, c), lws) && push!(fr.closure_shifted, c.id)
        end
    end
    # Content-join READ refinement is legal only when every capturing closure
    # is non-escaping and world-stable (Keno's rule): an escaping closure's
    # untyped mutable field is settable by any holder, and a shifted
    # closure's stores run against an unknown world — either poisons the
    # cell's reads to Any. The join itself is still computed (diagnostics,
    # the future await consumer); it is just never used for refinement.
    for (cid, caps) in cellcaps
        if any(x -> x in fr.closure_escaped || x in fr.closure_shifted, caps)
            push!(fr.poisoned_cells, cid)
        end
    end
    # a boundscheck used as a region guard condition steers control: dirty
    for reg in ir.regions
        (UnifiedIR.is_guard(reg) && !UnifiedIR.isnull(reg.cond)) || continue
        UnifiedIR.stmt_kind(ir, reg.cond) === K"boundscheck" && push!(bcdirty, reg.cond.id)
    end
    for id in bcs
        id in bcdirty || push!(fr.bc_guarded, id)
    end
    return fr
end

"""
    infer_ir!(ir, argtypes; state=UInferState()) -> rettype lattice

Run inference over a dense, sealed UnifiedIR body. Writes lattice elements
into the `type` column and `ir.meta[:rettype]`; returns the return type.
`argtypes` are lattice elements for region 1's args (position 1 = the
function itself).
"""
function infer_ir!(ir::UnifiedIR.IR, argtypes::Vector{Any};
                   state::UInferState = UInferState())
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_DENSE, "infer_ir!")
    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    length(argtypes) == length(root.args) ||
        error("infer_ir!: $(length(argtypes)) argtypes for $(length(root.args)) parameters")
    if isempty(state.active)
        # top-level query entry: fresh budget, fresh taint-scratch memos
        state.budget_mark = state.stats.frames
        empty!(state.scratch)
        empty!(state.cycle_scratch)
        empty!(state.scc_prev)
        state.stale_depth = typemax(Int)
    end
    state.stats.frames += 1
    fr = Frame(ir, state, Vector{Any}(nothing, UnifiedIR.nstmts(ir)))
    for (i, a) in enumerate(root.args)
        fr.env[a.id] = argtypes[i]
    end
    effmask = UnifiedIR.FLAG_CONSISTENT | UnifiedIR.FLAG_EFFECT_FREE |
              UnifiedIR.FLAG_NOTHROW | UnifiedIR.FLAG_TERMINATES
    # cell fixpoint: celltypes grow monotonically under tmerge; termination
    # comes from widening, never from stopping with a stale (unsound) state
    iter = 0
    while true
        iter += 1
        fr.cells_changed = false
        fr.rettype = nothing
        fr.effects = CC.EFFECTS_TOTAL
        fr.thrown = Any[Union{}]
        fill!(fr.stmt_effects, ~UInt32(0))
        infer_region!(fr, UnifiedIR.root_region(ir))
        fr.cells_changed || break
        if iter == 20
            # force-widen every cell to its widenconst to cap the ascent
            # (closure param/ret joins ride the same escalator — they are the
            # same kind of monotone accumulator)
            for (k, v) in fr.celltypes
                fr.celltypes[k] = CC.widenconst(widenucond(v))
            end
            for (k, v) in fr.closure_args
                fr.closure_args[k] = Any[CC.widenconst(widenucond(t)) for t in v]
            end
            for (k, v) in fr.closure_rets
                fr.closure_rets[k] = CC.widenconst(widenucond(v))
            end
        elseif iter > 40
            for (k, v) in fr.celltypes
                fr.celltypes[k] = Any
            end
            for (k, v) in fr.closure_args
                fr.closure_args[k] = Any[Any for _ in v]
            end
            for (k, v) in fr.closure_rets
                fr.closure_rets[k] = Any
            end
        end
    end
    # publish lattice elements into the type column and effect masks into the
    # flag column (only the 4 effect bits; other flag bits are preserved)
    for i in 1:UnifiedIR.nstmts(ir)
        s = StmtId(i)
        UnifiedIR.is_tombstone(ir, s) && continue
        m = fr.stmt_effects[i]
        if m != ~UInt32(0)
            old = UnifiedIR.stmt_flag(ir, s)
            new = (old & ~effmask) | (m & effmask)
            new == old || UnifiedIR.set_flag!(ir, s, new)
        end
        t = fr.env[i]
        t === nothing && continue
        UnifiedIR.set_type!(ir, s, widenucond(t))
    end
    # statement-position static-parameter reads elided by the entry converter:
    # a maybe-undefined one throws UndefVarError at runtime (stock
    # abstract_eval_static_parameter; operand-position reads are handled per
    # statement in note_effects!)
    let reads = get(ir.meta, :sparam_reads, nothing)
        if reads isa Vector{Int}
            for n in reads
                if sparam_maybe_undef(ir, n)
                    fr.effects = CC.Effects(fr.effects; nothrow = false)
                    note_thrown!(fr, UndefVarError)
                    break
                end
            end
        end
    end
    rt = fr.rettype === nothing ? Union{} : fr.rettype
    if rt isa UCond
        # InterConditional export: a conditional return whose subject is a
        # root parameter is context-free by position; anything else widens
        idx = 0
        if rt.subject[1] === :stmt
            for (i, a) in enumerate(root.args)
                a.id == rt.subject[2] && (idx = i; break)
            end
        end
        rt = idx == 0 ? widenucond(rt) : UInterCond(idx, rt.thentype, rt.elsetype)
    end
    ir.meta[:rettype] = widenucond(rt)
    eff, exct = finish_frame_effects(fr, rt, argtypes)
    ir.meta[:effects] = eff
    ir.meta[:exct] = exct
    ir.meta[:effects_mask] = effects_mask(eff)
    # late-pipeline channels (late.jl's query surfaces them): shared-cell
    # content joins, per-closure body result joins, and the refinement
    # classification (escape/world/poison discipline)
    ir.meta[:cell_content] = copy(fr.celltypes)
    ir.meta[:closure_rets] = copy(fr.closure_rets)
    ir.meta[:closure_escaped] = copy(fr.closure_escaped)
    ir.meta[:closure_shifted] = copy(fr.closure_shifted)
    ir.meta[:poisoned_cells] = copy(fr.poisoned_cells)
    return rt
end

"""Frame-finish effects adjustment (the stock `adjust_effects(sv)` port,
typeinfer.jl — minus the method-level override, which callers apply via
`apply_effects_override`): Bottom-rt consistency, exception-join nothrow
refinement, ARGMEM resolution over the root argument lattices, and the
conditional-bit resolutions (CONSISTENT_IF_NOTRETURNED against the return
type, *_IF_INACCESSIBLEMEMONLY against the final imo state). Returns
`(effects, exct)`."""
function finish_frame_effects(fr::Frame, @nospecialize(rt), argtypes::Vector{Any})
    eff = fr.effects
    exct = CC.widenconst(widenucond(fr.thrown[1]))
    if rt === Union{}
        # always throwing or never returning both count as consistent
        eff = CC.Effects(eff; consistent = CC.ALWAYS_TRUE)
    end
    if exct === Union{}
        # every raisable exception is caught (and no handler rethrows):
        # the per-statement nothrow taints do not escape this frame
        eff = CC.Effects(eff; nothrow = true)
    end
    CC.is_nothrow(eff) && (exct = Union{})
    if CC.is_inaccessiblemem_or_argmemonly(eff) &&
       Base.all(i -> CC.is_mutation_free_argtype(widenucond(argtypes[i])),
                1:length(argtypes))
        eff = CC.Effects(eff; inaccessiblememonly = CC.ALWAYS_TRUE)
    end
    if CC.is_consistent_if_notreturned(eff) &&
       CC.is_identity_free_argtype(widenucond(rt))
        # consistency tainted only by mutable allocations that provably do
        # not escape through the return value
        eff = CC.Effects(eff; consistent = eff.consistent & ~CC.CONSISTENT_IF_NOTRETURNED)
    end
    if CC.is_consistent_if_inaccessiblememonly(eff)
        if CC.is_inaccessiblememonly(eff)
            eff = CC.Effects(eff; consistent = eff.consistent & ~CC.CONSISTENT_IF_INACCESSIBLEMEMONLY)
        elseif CC.is_inaccessiblemem_or_argmemonly(eff)
        else # imo already tainted: no chance to refine later
            eff = CC.Effects(eff; consistent = CC.ALWAYS_FALSE)
        end
    end
    if CC.is_effect_free_if_inaccessiblememonly(eff)
        if CC.is_inaccessiblememonly(eff)
            eff = CC.Effects(eff; effect_free = eff.effect_free & ~CC.EFFECT_FREE_IF_INACCESSIBLEMEMONLY)
        elseif CC.is_inaccessiblemem_or_argmemonly(eff)
        else
            eff = CC.Effects(eff; effect_free = CC.ALWAYS_FALSE)
        end
    end
    return eff, exct
end

"Join a raisable exception type into the innermost active collector (a try
body's scope, or the frame's escape join)."
function note_thrown!(fr::Frame, @nospecialize(t))
    t === Union{} && return nothing
    t = CC.widenconst(widenucond(t))
    get(ENV, "UIR_DEBUG", "") == "2" && println("DBG thrown ", t, " in ",
        get(fr.ir.meta, :mi, fr.ir.meta))
    i = length(fr.thrown)
    fr.thrown[i] = CC.tmerge(CC.fallback_lattice, fr.thrown[i], t)
    return nothing
end

"Record a `return`: a single-operand UCond return stays conditional (the
InterConditional port; infer_ir! exports it by parameter position). Multiple
same-subject conditional returns join fieldwise; mixed returns widen."
function note_return!(fr::Frame, s::StmtId)
    if UnifiedIR.nops(fr.ir, s) == 1
        v = opl(fr, UnifiedIR.getop(fr.ir, s, 1))
        if v isa UCond
            old = fr.rettype
            if old === nothing
                fr.rettype = v
            elseif old isa UCond && old.subject == v.subject
                fr.rettype = UCond(v.subject,
                    CC.tmerge(CC.fallback_lattice, old.thentype, v.thentype),
                    CC.tmerge(CC.fallback_lattice, old.elsetype, v.elsetype))
            else
                fr.rettype = ⊔(fr.st, old, v)
            end
            return nothing
        end
    end
    fr.rettype = ⊔(fr.st, fr.rettype, joinvals(fr, opls(fr, s, 1)))
    return nothing
end

"A branch/continue condition that is not provably Bool throws a TypeError at
runtime (the stock GotoIfNot rule: merge EFFECTS_THROWS). Returns whether the
condition MUST throw (no intersection with Bool at all — stock types such a
GotoIfNot Bottom and the branch never completes, #41975)."
function taint_nonbool_cond!(fr::Frame, @nospecialize(condl))
    condl isa UCond && return false
    t = CC.widenconst(widenucond(condl))
    if !(t isa Type && t <: Bool)
        fr.effects = CC.merge_effects(fr.effects, CC.EFFECTS_THROWS)
        note_thrown!(fr, TypeError)
        return t isa Type && !CC.hasintersect(t, Bool)
    end
    return false
end

"Cyclic control (loop backedges, island back-gotos) drops `terminates` unless
the frame's method — or the backedge statement itself, via the entry-carried
statement override — declares `@assume_effects :terminates_locally` (the
`handle_control_backedge!` port; `s` is the looping terminator or loop op)."
function taint_backedge!(fr::Frame, s::StmtId)
    fr.override.terminates_locally && return nothing
    stmt_effects_override(fr.ir, s).terminates_locally && return nothing
    fr.effects = CC.Effects(fr.effects; terminates = false)
    return nothing
end

"Kill marker: a scope that masks any outer refinement of a subject (loop
backedges and throw edges invalidate path-sensitive facts)."
struct RefKill end
const REFINE_KILL = RefKill()

"Innermost active refinement for a subject, or nothing."
function refined(fr::Frame, key::Tuple{Symbol,Int32})
    for i in length(fr.refinements):-1:1
        v = get(fr.refinements[i], key, nothing)
        if v !== nothing
            v === REFINE_KILL && return nothing
            return v
        end
    end
    return nothing
end

"Cells stored anywhere within region `r`'s subtree (memoized per frame walk)."
function stored_cells_in(fr::Frame, r::RegionId)
    ir = fr.ir
    out = Set{Int32}()
    stack = RegionId[r]
    while !isempty(stack)
        cur = pop!(stack)
        for s in UnifiedIR.region_stmts(ir, cur)
            k = UnifiedIR.stmt_kind(ir, s)
            if k === K"cell_set" || k === K"cell_new"
                push!(out, UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id)
            end
            if UnifiedIR.owns_regions(k)
                for rid in UnifiedIR.live_owned_regions(ir, s)
                    push!(stack, rid)
                end
            end
        end
    end
    return out
end

"Push a scope masking cell refinements invalidated by `r`'s stores (backedge/
throw-edge rule); returns whether a scope was pushed."
function push_store_kills!(fr::Frame, r::RegionId)
    cells = stored_cells_in(fr, r)
    isempty(cells) && return false
    m = RefMap()
    for c in cells
        m[(:cell, c)] = REFINE_KILL
    end
    push!(fr.refinements, m)
    return true
end

"Read a cell's lattice element (refinements shadow the global celltype)."
function cell_lattice(fr::Frame, cellid::Int32)
    r = refined(fr, (:cell, cellid))
    r === nothing || return r
    return get(fr.celltypes, cellid, Union{})
end

# lattice element of a value operand
function opl(fr::Frame, o::UnifiedIR.Operand)
    ir = fr.ir
    t = UnifiedIR.optag(o)
    if t == UnifiedIR.TAG_STMT
        sid = UnifiedIR.payload(o) % Int32
        r = refined(fr, (:stmt, sid))
        r === nothing || return r
        v = fr.env[sid]
        return v === nothing ? Any : v   # not-yet-visited: conservative
    elseif t == UnifiedIR.TAG_INLINE
        return CC.Const(UnifiedIR.imm_value(o))
    elseif t == UnifiedIR.TAG_CONST
        return CC.Const(ir.body.constants[UnifiedIR.payload(o)])
    elseif t == UnifiedIR.TAG_GLOBAL
        g = ir.body.globals[UnifiedIR.payload(o)]
        col = fr.st.edges
        # driver mode: partition-based read (world-pinned, binding edge
        # recorded); otherwise the historical ambient read
        col === nothing || return global_partition_rte(col, g.mod, g.name).rt
        if isconst(g.mod, g.name) && isdefined(g.mod, g.name)
            return CC.Const(getglobal(g.mod, g.name))
        end
        return Any
    elseif t == UnifiedIR.TAG_SPARAM
        i = Int(UnifiedIR.payload(o))
        lat = get(ir.meta, :sptypes_lat, nothing)
        if lat isa Vector{Any} && i <= length(lat)
            return lat[i]     # stock-decoded lattice (sptypes_from_meth_instance)
        end
        if i <= length(ir.sptypes)
            # raw sparam values (transfers.jl decodes non-value markers)
            return raw_sparam_lattice(ir.sptypes[i])
        end
        return Any
    else
        return Any
    end
end

opls(fr::Frame, s::StmtId, from::Int) =
    Any[opl(fr, UnifiedIR.getop(fr.ir, s, i)) for i in from:UnifiedIR.nops(fr.ir, s)]

joinvals(fr::Frame, vals::Vector{Any}) =
    isempty(vals) ? CC.Const(nothing) :
    length(vals) == 1 ? widenucond(vals[1]) :
    CC.builtin_tfunction(fr.st.cfg.interp, Core.tuple, Any[widenucond(v) for v in vals], nothing)

# Region inference: returns the join of `result` values reaching the owner
# (nothing if no result terminator), accumulating return/continue/break joins on `fr`.
function infer_region!(fr::Frame, r::RegionId)
    ir = fr.ir
    results = nothing
    npush = 0
    for s in UnifiedIR.region_stmts(ir, r)
        k = UnifiedIR.stmt_kind(ir, s)
        if k !== K"region_arg" && (UnifiedIR.is_terminator(k) || UnifiedIR.owns_regions(k))
            # control statements bypass `transfer`; their operand evaluation
            # (global reads, mutable literals, sparams) still has effects
            note_operand_effects!(fr, s)
        end
        if k === K"region_arg"
            continue
        elseif k === K"result"
            results = ⊔(fr.st, results, joinvals(fr, opls(fr, s, 1)))
        elseif k === K"return"
            note_return!(fr, s)
        elseif k === K"continue"
            tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
            condl = opl(fr, UnifiedIR.getop(ir, s, 2))
            if taint_nonbool_cond!(fr, condl)
                # must-throw condition: neither the backedge nor the loop
                # exit is taken (stock's Bottom GotoIfNot rule)
                fr.env[s.id] = Union{}
                continue
            end
            vals = opls(fr, s, 3)
            prev = get(fr.continue_vals, tgt.id, nothing)
            joined = prev === nothing ? vals :
                Any[CC.tmerge(CC.fallback_lattice, prev[i], vals[i]) for i in 1:length(vals)]
            fr.continue_vals[tgt.id] = joined
            # cond not provably true ⇒ the loop can exit here with `vals`
            if !(condl isa CC.Const && condl.val === true)
                fr.break_vals[tgt.id] = ⊔(fr.st, get(fr.break_vals, tgt.id, nothing),
                                          joinvals(fr, vals))
            end
        elseif k === K"break"
            tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
            fr.break_vals[tgt.id] = ⊔(fr.st, get(fr.break_vals, tgt.id, nothing),
                                      joinvals(fr, opls(fr, s, 2)))
        elseif k === K"unreachable"
        elseif k === K"if"
            carry = infer_if!(fr, s)
            if carry !== nothing
                # one arm diverges: its complement's refinement holds for the
                # remainder of this region (§10.3(a) — the Pi/Conditional
                # machinery relocated)
                push!(fr.refinements, carry)
                npush += 1
            end
            fr.env[s.id] === Union{} && break   # no arm falls through: dead rest
        elseif k === K"loop"
            infer_loop!(fr, s)
            fr.env[s.id] === Union{} && break   # loop never exits: dead rest
        elseif k === K"try"
            infer_try!(fr, s)
            fr.env[s.id] === Union{} && break
        elseif k === K"cfg"
            infer_cfg!(fr, s)
            fr.env[s.id] === Union{} && break
        elseif k === K"closure"
            infer_closure!(fr, s)
        else
            fr.env[s.id] = transfer(fr, s, k)
            if fr.pending_refine !== nothing
                # typeassert/store back-propagation: the refinement holds for
                # the remainder of this region
                pr = fr.pending_refine::Pair
                push!(fr.refinements, RefMap(pr.first => pr.second))
                npush += 1
                fr.pending_refine = nothing
            end
            # a Bottom-typed statement never completes (guaranteed throw):
            # the rest of this region is unreachable (stock's dead-tail rule)
            fr.env[s.id] === Union{} && break
        end
    end
    for _ in 1:npush
        pop!(fr.refinements)
    end
    return results
end

"Does control ever reach the join after this region (false = diverges)?"
function region_falls_through(ir::UnifiedIR.IR, r::RegionId)
    t = UnifiedIR.region_terminator(ir, r)
    t === nothing && return true
    return UnifiedIR.stmt_kind(ir, t) === K"result"
end

function infer_if!(fr::Frame, s::StmtId)::Union{Nothing,RefMap}
    ir = fr.ir
    condl = opl(fr, UnifiedIR.getop(ir, s, 1))
    if taint_nonbool_cond!(fr, condl)
        # the condition cannot be Bool: the branch throws TypeError before
        # either arm runs (stock types the GotoIfNot Bottom; the region rest
        # is the caller's dead-tail rule)
        fr.env[s.id] = Union{}
        return nothing
    end
    rs = UnifiedIR.live_owned_regions(ir, s)
    local res
    carry = nothing
    # a Conditional with a Bottom arm decides the branch (stock's rule:
    # elsetype ⊥ ⇒ the condition is provably true on any live path)
    if condl isa UCond && (condl.thentype === Union{} || condl.elsetype === Union{})
        taken = condl.elsetype === Union{}
        reft = taken ? condl.thentype : condl.elsetype
        push!(fr.refinements, RefMap(condl.subject => reft))
        local rres
        if taken
            rres = infer_region!(fr, rs[1])
        elseif length(rs) >= 2
            rres = infer_region!(fr, rs[2])
        else
            rres = CC.Const(nothing)
        end
        pop!(fr.refinements)
        fr.env[s.id] = rres === nothing ? Union{} : rres
        # the surviving arm's refinement holds for the region rest
        return RefMap(condl.subject => reft)
    end
    if condl isa CC.Const && condl.val isa Bool
        # §10.3(b): a Const condition selects a single arm during inference
        if condl.val
            res = infer_region!(fr, rs[1])
        elseif length(rs) >= 2
            res = infer_region!(fr, rs[2])
        else
            res = CC.Const(nothing)
        end
    elseif condl isa UCond
        push!(fr.refinements, RefMap(condl.subject => condl.thentype))
        r1 = infer_region!(fr, rs[1])
        pop!(fr.refinements)
        local r2
        if length(rs) >= 2
            push!(fr.refinements, RefMap(condl.subject => condl.elsetype))
            r2 = infer_region!(fr, rs[2])
            pop!(fr.refinements)
        else
            r2 = CC.Const(nothing)
        end
        res = ⊔(fr.st, r1, r2)
        d1 = !region_falls_through(ir, rs[1])
        d2 = length(rs) >= 2 ? !region_falls_through(ir, rs[2]) : false
        if d1 && !d2
            carry = RefMap(condl.subject => condl.elsetype)
        elseif d2 && !d1
            carry = RefMap(condl.subject => condl.thentype)
        end
    else
        r1 = infer_region!(fr, rs[1])
        r2 = length(rs) >= 2 ? infer_region!(fr, rs[2]) : CC.Const(nothing)
        # §10.3(a): diverging arms (no result reached) contribute nothing
        res = ⊔(fr.st, r1, r2)
    end
    fr.env[s.id] = res === nothing ? Union{} : res
    return carry
end

function infer_loop!(fr::Frame, s::StmtId)
    ir = fr.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    bodyr = rs[1]
    breg = UnifiedIR.getregion(ir, bodyr)
    carried = Any[widenucond(v) for v in opls(fr, s, 1)]
    delete!(fr.break_vals, bodyr.id)
    # backedge rule: pre-loop refinements of cells the body stores are invalid
    # on iterations ≥ 2 — mask them for the whole body walk
    killed = push_store_kills!(fr, bodyr)
    iter = 0
    while true
        iter += 1
        for (i, a) in enumerate(breg.args)
            fr.env[a.id] = carried[i]
        end
        delete!(fr.continue_vals, bodyr.id)
        infer_region!(fr, bodyr)
        cont = get(fr.continue_vals, bodyr.id, nothing)
        cont === nothing && break   # body never continues: single trip
        newcarried = Any[CC.tmerge(CC.fallback_lattice, carried[i], widenucond(cont[i]))
                         for i in 1:length(carried)]
        if iter >= 4
            # §10.3(c) widening escalation: precision first, then widenconst,
            # then Any — the loop exits only at a (post-widening) fixpoint
            newcarried = Any[CC.widenconst(t) for t in newcarried]
        end
        iter > 24 && (newcarried = Any[Any for _ in newcarried])
        stable = all(i -> lat_eq(newcarried[i], carried[i]), 1:length(carried))
        carried = newcarried
        stable && break
    end
    killed && pop!(fr.refinements)
    result = get(fr.break_vals, bodyr.id, nothing)
    fr.env[s.id] = result === nothing ? Union{} : result   # never-exiting loop: ⊥
    # §5.1 rule 5: loops drop TERMINATES (bounded-trip proofs are future work)
    taint_backedge!(fr, s)
    return nothing
end

function infer_try!(fr::Frame, s::StmtId)
    ir = fr.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    # exceptions raised in the body are caught here: collect their join in a
    # fresh scope (it types the handler argument); the handler's own throws
    # land in the enclosing scope (post-pop), i.e. they escape this `try`
    push!(fr.thrown, Union{})
    r1 = infer_region!(fr, rs[1])
    thrown = pop!(fr.thrown)
    r2 = nothing
    if length(rs) >= 2
        if thrown === Union{}
            # the body provably raises nothing: the handler is dead code —
            # contributes neither values nor effects (the stock unreachable-
            # handler rule)
        else
            h = UnifiedIR.getregion(ir, rs[2])
            for (i, a) in enumerate(h.args)
                fr.env[a.id] = i == 1 ? thrown : Any   # %exc
            end
            if rs[2].id in fr.exc_read
                # the handler reads the caught value: its identity depends on
                # the dynamic environment (stock's :the_exception taint)
                fr.effects = CC.Effects(fr.effects; consistent = CC.ALWAYS_FALSE)
            end
            # throw-edge rule: the handler may run after any prefix of the
            # body, so refinements of cells the body stores are invalid there
            killed = push_store_kills!(fr, rs[1])
            r2 = infer_region!(fr, rs[2])
            killed && pop!(fr.refinements)
        end
    else
        # no handler region: the body's exceptions escape
        note_thrown!(fr, thrown)
    end
    res = ⊔(fr.st, r1, r2)
    fr.env[s.id] = res === nothing ? Union{} : res
    return nothing
end

"""
    infer_closure!(fr, s)

The §5.7 descent: type a `closure` op's deferred body "as if", inside the
enclosing frame's fixpoint. Sound regardless of when the body runs provided
the environment types are sound at every call time: value captures are SSA
(fixed values), cell captures read the monotone content join (`celltypes`,
which every `cell_set` — home or deferred — feeds), and params are the join
of argtypes over visible call sites (`closure_args`, fed by
`closure_callee_transfer`) or declared/`Any` for escapees. The body walk runs
behind three barriers:

  - refinement barrier: outer flow-sensitive facts are creation-time facts,
    invalid at call time (the body may run after any number of stores);
  - rettype isolation: body `return`s bind to the closure (its activation
    root), not the home frame — recorded as `closure_rets[s]`;
  - effects isolation (§3.3 mode-aware composition): deferred effects do not
    count at the creation site; the mask is recorded as `closure_effs[s]`
    and applied at call sites instead.

The op's own value stays `Any` (its runtime type does not exist until
materialization). Map changes set `cells_changed`, re-entering the same
frame fixpoint; the shared widening escalator caps the ascent.
"""
function infer_closure!(fr::Frame, s::StmtId)
    ir = fr.ir
    fr.env[s.id] = Any
    rs = UnifiedIR.live_owned_regions(ir, s)
    isempty(rs) && return nothing
    breg = UnifiedIR.getregion(ir, rs[1])
    np = length(breg.args)
    isva = false
    if UnifiedIR.nops(ir, s) >= 1
        flags = UnifiedIR.imm_value(UnifiedIR.getop(ir, s, 1))::Int64
        isva = (flags & UnifiedIR.CLOSURE_FLAG_ISVA) != 0
    end
    if s.id in fr.closure_escaped || s.id in fr.closure_shifted
        # unknown callers (escape) or unknown execution world (shifted):
        # declared types honored (region_arg type column; the current
        # producer writes Any), the trailing isva param is at least a tuple
        for (i, a) in enumerate(breg.args)
            t = UnifiedIR.stmt_type(ir, a)
            t === nothing && (t = Any)
            (isva && i == np && t === Any) && (t = Tuple)
            fr.env[a.id] = t
        end
    else
        joins = get(fr.closure_args, s.id, nothing)
        for (i, a) in enumerate(breg.args)
            fr.env[a.id] = joins === nothing ? Union{} : joins[i]
        end
    end
    saved_refs = fr.refinements
    saved_ret = fr.rettype
    saved_eff = fr.effects
    saved_thrown = fr.thrown
    fr.refinements = RefMap[]
    fr.rettype = nothing
    fr.effects = CC.EFFECTS_TOTAL
    fr.thrown = Any[Union{}]   # deferred throws surface at call sites, not here
    infer_region!(fr, rs[1])
    bodyret = fr.rettype === nothing ? Union{} : widenucond(fr.rettype)
    bodyeff = fr.effects
    if fr.thrown[1] === Union{}
        bodyeff = CC.Effects(bodyeff; nothrow = true)   # all body throws caught
    end
    fr.refinements = saved_refs
    fr.rettype = saved_ret
    fr.effects = saved_eff
    fr.thrown = saved_thrown
    old = get(fr.closure_rets, s.id, nothing)
    newret = old === nothing ? bodyret :
        CC.tmerge(CC.fallback_lattice, widenucond(old), bodyret)
    if old === nothing || !lat_eq(newret, old)
        fr.closure_rets[s.id] = newret
        fr.cells_changed = true
    end
    oldeff = get(fr.closure_effs, s.id, nothing)
    neweff = oldeff === nothing ? bodyeff : CC.merge_effects(oldeff, bodyeff)
    if oldeff === nothing || neweff != oldeff
        fr.closure_effs[s.id] = neweff
        fr.cells_changed = true
    end
    return nothing
end

function infer_cfg!(fr::Frame, s::StmtId)
    ir = fr.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    # classical per-block fixpoint, local to the island (§5.5): per-block
    # entry state = (block args, Conditional refinements) — the VarTable of
    # typeinf_local, with the unrefined base carried by the monotone
    # cell-type map
    blockargs = Dict{Int32,Vector{Any}}()
    blockrefs = Dict{Int32,RefMap}()
    blockargs[rs[1].id] = Any[widenucond(a) for a in opls(fr, s, 1)]
    blockrefs[rs[1].id] = RefMap()
    changed = true
    result = nothing
    guard = 0
    cursrc = Ref{Int32}(0)   # region id of the block being walked

    function merge_edge!(src_st::StmtId, dest::RegionId, vals::Vector{Any}, ref::RefMap)
        # backward edge (region ids are in creation = statement order): the
        # island may cycle — §5.1 rule 5 drops TERMINATES (unless the looping
        # terminator carries `:terminates_locally`). Applies to both
        # in-island backedges and backward cross-island gotos (catch→loop-head).
        dest.id <= cursrc[] && taint_backedge!(fr, src_st)
        if UnifiedIR.getregion(ir, dest).owner != s
            # sealed cross-island exit: mark reached; values cross scopes
            # through cells, not block args
            dest.id in fr.reached || (push!(fr.reached, dest.id); changed = true)
            return
        end
        old = get(blockargs, dest.id, nothing)
        if old === nothing
            blockargs[dest.id] = Any[widenucond(v) for v in vals]
            blockrefs[dest.id] = copy(ref)
            changed = true
        else
            for i in 1:length(vals)
                m = CC.tmerge(CC.fallback_lattice, old[i], widenucond(vals[i]))
                lat_eq(m, old[i]) || (old[i] = m; changed = true)
            end
            # refinement join: keep common subjects at their tmerge; drop others
            oldref = blockrefs[dest.id]
            for (k, v) in collect(oldref)
                nv = get(ref, k, nothing)
                if nv === nothing
                    delete!(oldref, k)
                    changed = true
                else
                    m = CC.tmerge(CC.fallback_lattice, widenucond(v), widenucond(nv))
                    lat_eq(m, v) || (oldref[k] = m; changed = true)
                end
            end
        end
    end

    while changed && (guard += 1) < 200
        changed = false
        result = nothing
        nreached0 = length(fr.reached)
        for rid in rs
            blk = UnifiedIR.getregion(ir, rid)
            args = get(blockargs, rid.id, nothing)
            if args === nothing
                # blocks entered only through cross-island gotos carry no args
                rid.id in fr.reached || continue
                args = Any[]
            end
            for (i, a) in enumerate(blk.args)
                i <= length(args) && (fr.env[a.id] = args[i])
            end
            cursrc[] = rid.id
            entryref = get(blockrefs, rid.id, nothing)
            get(ENV, "UIR_DEBUG", "") == "1" && println("DBG walk block ^", rid.id, " of cfg %", s.id, " entryref=", entryref)
            push!(fr.refinements, entryref === nothing ? RefMap() : copy(entryref))
            npush = 1
            curref() = begin   # currently active refinements, flattened
                m = RefMap()
                for k in (length(fr.refinements) - npush + 1):length(fr.refinements)
                    merge!(m, fr.refinements[k])
                end
                m
            end
            for st in UnifiedIR.region_stmts(ir, rid)
                k = UnifiedIR.stmt_kind(ir, st)
                k === K"region_arg" && continue
                if UnifiedIR.is_terminator(k) || UnifiedIR.owns_regions(k)
                    note_operand_effects!(fr, st)   # control-operand evaluation
                end
                if k === K"result"
                    result = ⊔(fr.st, result, joinvals(fr, opls(fr, st, 1)))
                elseif k === K"return"
                    note_return!(fr, st)
                elseif k === K"if"
                    carry = infer_if!(fr, st)
                    if carry !== nothing
                        push!(fr.refinements, carry)
                        npush += 1
                    end
                    fr.env[st.id] === Union{} && break
                elseif k === K"loop"
                    infer_loop!(fr, st)
                    fr.env[st.id] === Union{} && break
                elseif k === K"try"
                    infer_try!(fr, st)
                    fr.env[st.id] === Union{} && break
                elseif k === K"cfg"
                    infer_cfg!(fr, st)
                    fr.env[st.id] === Union{} && break
                elseif k === K"closure"
                    infer_closure!(fr, st)
                elseif k === K"break"
                    tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, st, 1))
                    fr.break_vals[tgt.id] = ⊔(fr.st, get(fr.break_vals, tgt.id, nothing),
                                              joinvals(fr, opls(fr, st, 2)))
                elseif k === K"continue"
                    tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, st, 1))
                    condl = opl(fr, UnifiedIR.getop(ir, st, 2))
                    if taint_nonbool_cond!(fr, condl)
                        fr.env[st.id] = Union{}
                        continue
                    end
                    vals = Any[widenucond(v) for v in opls(fr, st, 3)]
                    prev = get(fr.continue_vals, tgt.id, nothing)
                    joined = prev === nothing ? vals :
                        Any[CC.tmerge(CC.fallback_lattice, prev[i], vals[i]) for i in 1:length(vals)]
                    fr.continue_vals[tgt.id] = joined
                    if !(condl isa CC.Const && condl.val === true)
                        fr.break_vals[tgt.id] = ⊔(fr.st, get(fr.break_vals, tgt.id, nothing),
                                                  joinvals(fr, vals))
                    end
                elseif k === K"goto"
                    (dest, args_ops) = UnifiedIR.edge_bundles(ir, st)[1]
                    merge_edge!(st, dest, Any[opl(fr, o) for o in args_ops], curref())
                elseif k === K"br_if"
                    condl = opl(fr, UnifiedIR.getop(ir, st, 1))
                    if taint_nonbool_cond!(fr, condl)
                        # must-throw condition: no edge is taken
                        fr.env[st.id] = Union{}
                        continue
                    end
                    get(ENV, "UIR_DEBUG", "") == "1" && println("DBG br_if %", st.id, " condl=", condl)
                    bundles = UnifiedIR.edge_bundles(ir, st)
                    ref = curref()
                    if condl isa CC.Const && condl.val isa Bool
                        # §10.3(b) inside islands: Const conditions kill edges
                        (dest, args_ops) = bundles[condl.val ? 1 : 2]
                        merge_edge!(st, dest, Any[opl(fr, o) for o in args_ops], ref)
                    elseif condl isa UCond && (condl.thentype === Union{} ||
                                               condl.elsetype === Union{})
                        # Bottom-armed Conditional decides the edge (stock rule)
                        taken = condl.elsetype === Union{}
                        eref = copy(ref)
                        eref[condl.subject] = taken ? condl.thentype : condl.elsetype
                        (dest, args_ops) = bundles[taken ? 1 : 2]
                        merge_edge!(st, dest, Any[opl(fr, o) for o in args_ops], eref)
                    elseif condl isa UCond
                        thenref = copy(ref); thenref[condl.subject] = condl.thentype
                        elseref = copy(ref); elseref[condl.subject] = condl.elsetype
                        (d1, a1) = bundles[1]
                        merge_edge!(st, d1, Any[opl(fr, o) for o in a1], thenref)
                        (d2, a2) = bundles[2]
                        merge_edge!(st, d2, Any[opl(fr, o) for o in a2], elseref)
                    else
                        for (dest, args_ops) in bundles
                            merge_edge!(st, dest, Any[opl(fr, o) for o in args_ops], ref)
                        end
                    end
                elseif k === K"switch" || k === K"await"
                    ref = curref()
                    for (dest, args_ops) in UnifiedIR.edge_bundles(ir, st)
                        merge_edge!(st, dest, Any[opl(fr, o) for o in args_ops], ref)
                    end
                elseif k === K"unreachable"
                else
                    fr.env[st.id] = transfer(fr, st, k)
                    if fr.pending_refine !== nothing
                        pr = fr.pending_refine::Pair
                        push!(fr.refinements, RefMap(pr.first => pr.second))
                        npush += 1
                        fr.pending_refine = nothing
                    end
                    # dead-tail rule: a Bottom statement never completes
                    fr.env[st.id] === Union{} && break
                end
            end
            for _ in 1:npush
                pop!(fr.refinements)
            end
        end
        length(fr.reached) > nreached0 && (changed = true)
    end
    fr.env[s.id] = result === nothing ? Union{} : result
    return nothing
end
