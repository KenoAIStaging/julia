# Inlining parity upgrades (§10.4 / stock reference: ssair/inlining.jl — the
# CASES, not the mechanics):
#
#   * multi-return callees: `normalize_single_return!` rebuilds the callee
#     with its body wrapped in a single-iteration `loop`; every `return v`
#     becomes `break ^wrapper (v)`, and one `return %loop` is appended — the
#     splice matrix then always sees a single root-level return;
#   * invoke-site inlining (K"invoke" statements resolve through their
#     CodeInstance/MethodInstance);
#   * union-split inlining: a call with a Union-typed argument that resolves
#     to one method per component is rewritten into an isa-dispatch chain via
#     `wrap_in_if!`, with `refine` statements narrowing the argument in each
#     arm so the per-component calls become statically resolvable (and get
#     inlined on the next round);
#   * cost heuristic: FLAG_NOINLINE / `@noinline` callees are skipped;
#     statement-count budgets scale for `@inline`/FLAG_INLINE (the simple
#     analog of stock InliningParams cost thresholds).

struct InlineParams
    size_limit::Int          # non-arg stmt budget, default callees
    inline_size_limit::Int   # non-arg stmt budget under @inline/FLAG_INLINE
    max_union_split::Int     # maximum isa-dispatch components (stock: 4)
    split_budget::Int        # union splits per pass invocation
end
InlineParams(; size_limit = 32, inline_size_limit = 128, max_union_split = 3,
             split_budget = 4) =
    InlineParams(size_limit, inline_size_limit, max_union_split, split_budget)

# ---------------------------------------------------------------------------
# Stock-shaped inlining cost (policy parity)
# ---------------------------------------------------------------------------
#
# Stock admission consults the callee CodeInstance's `inlining_cost` — the
# `inline_cost_model` verdict over the OPTIMIZED callee body, where accessor
# and constructor wrappers have folded to a handful of cheap statements and
# residual dynamic calls carry the nonleaf penalty. A raw-statement count
# over the unoptimized lowered body mismeasures both directions (a 58-stmt
# ctor body that optimizes to 3 statements; a 2-stmt body around one
# dynamic call). This port: cached CodeInstance cost when available
# (stock's exact input), otherwise compute it — entry-convert, optimize
# through this pipeline, exit to IRCode, and run stock's
# `inline_cost_model`. Memoized per MethodInstance, world-stamped (any
# redefinition bumps the world counter), cycle/depth-guarded (the callee
# optimization recursively costs ITS callees).
const INLINE_COST_LOCK = Base.ReentrantLock()
const INLINE_COST_MEMO = IdDict{Core.MethodInstance,Any}()   # -> Int | nothing
const INLINE_COST_WORLD = Base.RefValue{UInt}(0)
const INLINE_COST_ACTIVE = Base.IdSet{Core.MethodInstance}()
const INLINE_COST_MAX_ACTIVE = 4
const INLINE_COST_MAX_SRC_STMTS = 1000
const CI_COST_ENABLED = Base.RefValue(true)

# CI-less pricing grade (wave 11): how a candidate WITHOUT a cached
# CodeInstance verdict is priced.
#   :optimize — full pipeline over the callee body, recursively (each level
#               prices ITS candidates the same way: the cold-landing cost
#               towers of scratchpad/wave10h — 3+ levels of
#               inline2_cost_uncached → optimize_ir! → inline_calls2!);
#   :depth1   — full pipeline only OUTSIDE a tower (TOWER_FRAME_CAP unset);
#               nested pricing (inside any cost/fx/ea tower walk) uses the
#               statement-cost walk, so towers terminate at depth 1;
#   :stmtwalk — statement-cost walk always (stock's own shape: the model
#               prices the callee's inferred body without optimizing it).
# Grades memoize separately (INLINE_COST_SW_MEMO): a nested stmt-walk
# verdict never masks the full-grade verdict a depth-0 query computes.
const COST_PRICING = Base.RefValue{Symbol}(:depth1)
const INLINE_COST_SW_MEMO = IdDict{Core.MethodInstance,Any}()  # -> Int | nothing

"""
    inline2_cost(st, mi, src) -> Union{Int,Nothing}

Stock `inlining_cost` for `mi`: the cached CodeInstance's value when the
interpreter's cache has one (stock's exact policy input, including
stock-produced entries when pipelines mix), else computed through the
unified pipeline + `Compiler.inline_cost_model`. `nothing` = no verdict
(cycle, budget, unconvertible body) — the caller falls back to the
statement-count heuristic.
"""
function inline2_cost(st::UInferState, mi::Core.MethodInstance, src::Core.CodeInfo)
    interp = st.cfg.interp
    interp isa Compiler.AbstractInterpreter || return nothing
    # Cached-CodeInstance fast path (wave 9): stock admission's EXACT policy
    # input — `inlining_cost(ci.inferred)`, the model verdict stored at the
    # callee's own publication (the driver measures its devirtualized exit
    # IRCode; stock entries carry stock's verdict). The wave-6 objection
    # (cache-warmth nondeterminism: the stored cost was measured over a
    # less-devirtualized body than this pipeline's recompute) predates the
    # driver's :invoke emission running BEFORE the cost model — stored and
    # recomputed verdicts now price the same body shape, and the recompute
    # towers were the cold-walk's dominant optimizer cost. Triage switch:
    # `CI_COST_ENABLED[] = false`.
    if CI_COST_ENABLED[]
        ci = get(Compiler.code_cache(interp), mi, nothing)
        if ci isa Core.CodeInstance && ci.max_world == typemax(UInt)
            inf = @atomic :monotonic ci.inferred
            # `nothing` = source discarded (const-ABI etc.): no verdict here —
            # unified inlines from the ORIGINAL source, so fall through to the
            # computed model rather than declining outright
            inf === nothing || return Int(Compiler.inlining_cost(inf))
        end
    end
    # narrow-budget states are the driver's reentrant/self-hosting passes:
    # optimizing callee bodies for cost there multiplies the burn-in
    # quadratically (the world advances between passes, restamping the
    # memo) — those passes keep the cheap statement-count fallback
    st.cfg.frame_budget >= 1000 || return nothing
    world = st.cfg.world
    # same-task reentry succeeds (ReentrantLock); a cross-thread race skips
    # the memo and yields no verdict rather than blocking a compile path
    # pricing grade for THIS query: nested tower walks (any cost/fx/ea
    # helper's bounded optimization sets TOWER_FRAME_CAP around itself)
    # demote to the statement-cost walk under :depth1
    mode = COST_PRICING[]
    sw = mode === :stmtwalk || (mode === :depth1 && TOWER_FRAME_CAP[] != 0)
    trylock(INLINE_COST_LOCK) || return nothing
    try
        if INLINE_COST_WORLD[] != world
            empty!(INLINE_COST_MEMO)
            empty!(INLINE_COST_SW_MEMO)
            INLINE_COST_WORLD[] = world
        end
        # a full-grade verdict serves every query; the stmt-walk memo only
        # serves stmt-walk-grade queries (depth-0 queries recompute at full
        # grade and store alongside)
        haskey(INLINE_COST_MEMO, mi) && return INLINE_COST_MEMO[mi]
        sw && haskey(INLINE_COST_SW_MEMO, mi) && return INLINE_COST_SW_MEMO[mi]
        (mi in INLINE_COST_ACTIVE || length(INLINE_COST_ACTIVE) >= INLINE_COST_MAX_ACTIVE) &&
            return nothing
        # the statement-walk grade is one bounded infer per mi (memoized) —
        # it does not draw down the per-body callee-optimization budget, or
        # the (much more numerous) cheap verdicts would starve the tail
        # refinements' EA summaries of their units (the f_EA_refine shape)
        sw || opt_work_take!() || return nothing
        push!(INLINE_COST_ACTIVE, mi)
        t0 = time_ns()
        r = try
            sw ? inline2_cost_stmtwalk(st, mi, src) :
                 inline2_cost_uncached(st, mi, src)
        catch
            nothing
        finally
            delete!(INLINE_COST_ACTIVE, mi)
            DRIVER_PHASES.cost_tower += Int(time_ns() - t0)
        end
        (sw ? INLINE_COST_SW_MEMO : INLINE_COST_MEMO)[mi] = r
        return r
    finally
        unlock(INLINE_COST_LOCK)
    end
end

# Shared pricing prologue: specTypes → per-parameter argtypes (packed va
# tail) + the entry-converted callee body, or nothing (unpriceable shape).
function cost_entry_convert(mi::Core.MethodInstance, src::Core.CodeInfo)
    m = mi.def
    m isa Method || return nothing
    length(src.code) <= INLINE_COST_MAX_SRC_STMTS || return nothing
    sig = mi.specTypes
    sig isa DataType || return nothing
    ps = collect(Any, sig.parameters)
    nargs = Int(m.nargs)
    if m.isva
        nargs >= 1 || return nothing
        length(ps) >= nargs - 1 || return nothing
        vat = try
            Tuple{ps[nargs:end]...}
        catch
            Tuple
        end
        ps = Any[ps[1:(nargs - 1)]; vat]
    end
    length(ps) == nargs || return nothing
    Base.any(p -> CC.isvarargtype(p), ps) && return nothing
    ir = codeinfo_to_ir(src; nargs, name = m.name)
    ir.meta[:method_instance] = mi
    ir.meta[:slotnames] = src.slotnames
    ir.sptypes = Any[t for t in mi.sparam_vals]
    ir.meta[:sptypes_lat] = sptypes_lattice(mi)
    return (ir, ps)
end

function inline2_cost_uncached(st::UInferState, mi::Core.MethodInstance,
                               src::Core.CodeInfo)
    conv = cost_entry_convert(mi, src)
    conv === nothing && return nothing
    ir, ps = conv
    # tower frame cap: price with a BOUNDED walk, refuse when it fires
    # (see TOWER_FRAME_CAP) — the statement-count fallback then applies,
    # which declines the same budget-busting bodies the capped walk would
    # have priced at MAX
    lim0 = st.limited
    prevcap = TOWER_FRAME_CAP[]
    ir = try
        TOWER_FRAME_CAP[] = TOWER_FRAME_BUDGET[]
        optimize_ir!(ir, ps; state = st, inline = true)
    finally
        TOWER_FRAME_CAP[] = prevcap
    end
    st.limited > lim0 && return nothing
    # statically-resolvable residual calls must be measured as `:invoke`
    # (stock's inliner has rewritten declined candidates before its cost
    # model sees them: UNKNOWN_CALL_COST, not the dynamic nonleaf penalty)
    UnifiedIR.editable(ir)
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        cmi = ea_resolve_residual_call(ir, st, s)
        cmi isa Core.MethodInstance || continue
        UnifiedIR.replace_stmt!(ir, s, K"invoke", UnifiedIR.vop(ir, cmi),
                                UnifiedIR.operands(ir, s)...;
                                type = UnifiedIR.stmt_type(ir, s))
    end
    ir = compact_carry_names!(ir)
    ircode = ir_to_ircode(ir)
    params = Compiler.OptimizationParams(st.cfg.interp)
    # cost capped at the widest threshold any caller applies (declared/
    # callsite @inline = 20x); beyond it the model returns MAX_INLINE_COST
    cap = 20 * params.inline_cost_threshold
    return Int(Compiler.inline_cost_model(ircode, params, Int(cap)))
end

# ---------------------------------------------------------------------------
# Statement-cost pricing over the inferred, UNOPTIMIZED body (wave 11)
# ---------------------------------------------------------------------------
#
# Stock's `statement_cost` vocabulary transplanted onto the typed region IR
# straight out of entry conversion + one `infer_ir!` pass — no optimizer, so
# pricing never recurses into `inline_calls2!` (the cold-landing cost
# towers). Divergence from the pipeline-optimized grade: wrapper chains that
# would fold away are priced at their pre-fold statement costs, and
# statically-resolvable residual calls are priced as the `:invoke` they
# would become (`UNKNOWN_CALL_COST`, the same rule `inline2_cost_uncached`
# applies by rewriting them before running stock's model) rather than at
# their post-inline expansion.

"""
    inline2_cost_stmtwalk(st, mi, src) -> Union{Int,Nothing}

The stock statement-cost model over `mi`'s inferred (pre-optimize) body.
`nothing` = no verdict (unpriceable shape, or the bounded inference walk
hit a cutoff).
"""
function inline2_cost_stmtwalk(st::UInferState, mi::Core.MethodInstance,
                               src::Core.CodeInfo)
    conv = cost_entry_convert(mi, src)
    conv === nothing && return nothing
    ir, ps = conv
    lim0 = st.limited
    prevcap = TOWER_FRAME_CAP[]
    try
        TOWER_FRAME_CAP[] = TOWER_FRAME_BUDGET[]
        infer_ir!(ir, ps; state = st)
    finally
        TOWER_FRAME_CAP[] = prevcap
    end
    st.limited > lim0 && return nothing
    params = Compiler.OptimizationParams(st.cfg.interp)
    cap = 20 * params.inline_cost_threshold
    return region_inline_cost(ir, st, params, Int(cap))
end

"""Blocks of the body's cfg islands unreachable once Const branch
conditions are honored (the walk prices an UNOPTIMIZED body — without
this, a wrapper's statically-dead generic arm charges its nonleaf penalty
while the optimized body the verdict stands in for would have folded it:
the `setproperty!` convert-arm shape). Returns a `BitSet` of DEAD region
ids (empty = everything live)."""
function sw_dead_blocks(ir::UnifiedIR.IR)
    dead = BitSet()
    branchkind(k::UnifiedIR.Kind) =
        k === K"goto" || k === K"br_if" || k === K"switch" || k === K"await"
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"cfg" || continue
        rs = UnifiedIR.live_owned_regions(ir, s)
        isempty(rs) && continue
        blocks = BitSet(Int(r.id) for r in rs)
        # island block containing a statement's region (parent-chain walk)
        function owning_block(t::StmtId)
            r = Int(UnifiedIR.stmt_region(ir, t).id)
            steps = 0
            while !(r in blocks)
                (steps += 1) <= UnifiedIR.nregions(ir) || return 0
                p = Int(UnifiedIR.getregion(ir, UnifiedIR.RegionId(Int32(r))).parent.id)
                p == 0 && return 0
                r = p
            end
            return r
        end
        succs = Dict{Int,Vector{Int}}()
        seeds = BitSet([Int(rs[1].id)])
        for t in UnifiedIR.each_stmt(ir)
            UnifiedIR.is_tombstone(ir, t) && continue
            k = UnifiedIR.stmt_kind(ir, t)
            branchkind(k) || continue
            dests = Int[]
            if k === K"br_if"
                condl = stmt_lattice(ir, UnifiedIR.getop(ir, t, 1))
                bs = UnifiedIR.edge_bundles(ir, t)
                if condl isa CC.Const && condl.val isa Bool && length(bs) >= 2
                    push!(dests, Int(bs[condl.val ? 1 : 2][1].id))
                else
                    for (d, _) in bs
                        push!(dests, Int(d.id))
                    end
                end
            else
                for (d, _) in UnifiedIR.edge_bundles(ir, t)
                    push!(dests, Int(d.id))
                end
            end
            bl = owning_block(t)
            if bl in blocks
                append!(get!(() -> Int[], succs, bl), dests)
            else
                # branch from outside this island: its in-island targets are
                # entries (conservatively live)
                for d in dests
                    d in blocks && push!(seeds, d)
                end
            end
        end
        reach = BitSet()
        wl = collect(seeds)
        while !isempty(wl)
            b = pop!(wl)
            (b in reach || !(b in blocks)) && continue
            push!(reach, b)
            for d in get(succs, b, Int[])
                d in reach || push!(wl, d)
            end
        end
        for b in blocks
            b in reach || push!(dead, b)
        end
    end
    return dead
end

"true when `s` sits (transitively) inside a dead island block"
function sw_in_dead_block(ir::UnifiedIR.IR, s::StmtId, dead::BitSet)
    r = Int(UnifiedIR.stmt_region(ir, s).id)
    steps = 0
    while r != 0
        r in dead && return true
        (steps += 1) <= UnifiedIR.nregions(ir) || return false
        r = Int(UnifiedIR.getregion(ir, UnifiedIR.RegionId(Int32(r))).parent.id)
    end
    return false
end

function region_inline_cost(ir::UnifiedIR.IR, st::UInferState,
                            params::Compiler.OptimizationParams, cap::Int)
    dead = sw_dead_blocks(ir)
    bodycost = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.is_tombstone(ir, s) && continue
        isempty(dead) || !sw_in_dead_block(ir, s, dead) || continue
        c = region_stmt_cost(ir, st, s, params)
        bodycost = Compiler.plus_saturate(bodycost, c)
        bodycost > cap && return Int(Compiler.MAX_INLINE_COST)
    end
    return Int(Compiler.inline_cost_clamp(bodycost))
end

# stock `statement_or_branch_cost` on region-IR vocabulary
function region_stmt_cost(ir::UnifiedIR.IR, st::UInferState, s::StmtId,
                          params::Compiler.OptimizationParams)
    k = UnifiedIR.stmt_kind(ir, s)
    if k === K"call"
        return region_call_cost(ir, st, s, params)
    elseif k === K"intrinsic"
        f = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        return f isa Core.IntrinsicFunction ? region_intrinsic_cost(ir, s, f) : 20
    elseif k === K"invoke"
        # non-returning invokes are error paths: free (stock's rule)
        t = UnifiedIR.stmt_type(ir, s)
        return (t isa Type && t === Union{}) ? 0 : 20
    elseif k === K"foreigncall"
        return 20
    elseif k === K"copyast"
        return 100
    elseif k === K"try"
        return typemax(Int)          # stock EnterNode: never inline
    elseif k === K"continue"
        return 40                    # loop backedge (stock backward goto)
    elseif k === K"goto" || k === K"br_if" || k === K"switch" || k === K"await"
        # cfg-island branch: backward target = a loop (dense spans: the
        # target block's first statement precedes the branch)
        for i in 1:UnifiedIR.nops(ir, s)
            o = UnifiedIR.getop(ir, s, i)
            UnifiedIR.optag(o) == UnifiedIR.TAG_BLOCK || continue
            tr = UnifiedIR.getregion(ir, UnifiedIR.asregion(o))
            tf = tr.first
            (tf.id != 0 && tf.id <= s.id) && return 40
        end
        return 0
    end
    return 0
end

# stock `statement_cost`'s `:call` arm on region-IR operands
function region_call_cost(ir::UnifiedIR.IR, st::UInferState, s::StmtId,
                          params::Compiler.OptimizationParams)
    #=const=# UNKNOWN_CALL_COST = 20
    flat = stmt_lattice(ir, UnifiedIR.getop(ir, s, 1))
    f = CC.singleton_type(flat)
    if f isa Core.IntrinsicFunction
        return region_intrinsic_cost(ir, s, f)
    end
    if f isa Core.Builtin && f !== Core.invoke
        nop = UnifiedIR.nops(ir, s)
        if f === Core.getfield || f === Core.tuple || f === Core.getglobal
            return 0
        elseif (f === Core.memoryrefget || f === Core.memoryref_isassigned) && nop >= 3
            atyp = stmt_lattice(ir, UnifiedIR.getop(ir, s, 2))
            return Compiler.isknowntype(atyp) ? 1 : params.inline_nonleaf_penalty
        elseif (f === Core.memoryrefset! || f === Core.memoryrefunset!) && nop >= 3
            atyp = stmt_lattice(ir, UnifiedIR.getop(ir, s, 2))
            return Compiler.isknowntype(atyp) ? 5 : params.inline_nonleaf_penalty
        elseif f === Core.typeassert && nop >= 3 &&
               CC.isconstType(CC.widenconst(stmt_lattice(ir, UnifiedIR.getop(ir, s, 3))))
            return 1
        end
        fidx = CC.find_tfunc(f)
        fidx === nothing && return UNKNOWN_CALL_COST
        return CC.T_FFUNC_COST[fidx]
    end
    t = UnifiedIR.stmt_type(ir, s)
    (t isa Type && t === Union{}) && return 0          # error path: free
    # a statically-resolvable residual call measures as the `:invoke` it
    # becomes (stock's inliner rewrites declined candidates before its
    # model sees them — the same rule the optimized-grade pricing applies)
    ea_resolve_residual_call(ir, st, s) isa Core.MethodInstance &&
        return UNKNOWN_CALL_COST
    return params.inline_nonleaf_penalty
end

# stock's IntrinsicFunction arm (const-arg halving heuristic included);
# operand 1 is the callee/which slot for both K"call" and K"intrinsic",
# so arities line up with stock's `length(ex.args)`
function region_intrinsic_cost(ir::UnifiedIR.IR, s::StmtId, f::Core.IntrinsicFunction)
    iidx = Int(reinterpret(Int32, f)) + 1
    nargs = UnifiedIR.nops(ir, s)
    isassigned(CC.T_IFUNC, iidx) || return 20
    minarg, maxarg, = CC.T_IFUNC[iidx]
    (minarg + 1 <= nargs <= maxarg + 1) || return 20
    cost = CC.T_IFUNC_COST[iidx]
    if cost == 0 || nargs < 3 || f === Core.Intrinsics.llvmcall
        return cost
    end
    aty2 = CC.widenconditional(stmt_lattice(ir, UnifiedIR.getop(ir, s, 2)))
    nconst = Int(aty2 isa CC.Const)
    for i in 3:nargs
        aty = CC.widenconditional(stmt_lattice(ir, UnifiedIR.getop(ir, s, i)))
        if CC.widenconst(aty) != CC.widenconst(aty2)
            nconst = 0
            break
        end
        nconst += aty isa CC.Const
    end
    nconst + 2 >= nargs && (cost = (cost - 1) ÷ 2)
    return cost
end

# A "single match" is only the dispatch outcome when it also FULLY COVERS
# the queried signature AND dispatch is unambiguous: a non-covering match
# means some argument tuples in `sig` dispatch to a MethodError, and baking
# the method's body in (or invoking it directly) would run it for those too.
# Ambiguity is the same soundness class — `_methods_by_ftype`/`ml_matches`
# can report ONE fully-covering match while dispatch is ambiguous on a
# subset of `sig` (two methods, neither more specific, applicable
# intersection); devirtualizing such a site drops the runtime MethodError.
# `CC.findall`'s `.ambig` flag is the authority — every resolution goes
# through it.
function resolve_single_match(@nospecialize(sig), world::UInt)
    result = try
        CC.findall(sig, CC.InternalMethodTable(world); limit = 1)
    catch
        nothing
    end
    result === nothing && return nothing        # >1 methods or failed query
    result.ambig && return nothing              # ambiguous dispatch: stay dynamic
    length(result.matches) == 1 || return nothing
    match = result.matches[1]::Core.MethodMatch
    match.fully_covers || return nothing
    return match
end

# State-threaded variant: inlining bakes callee bodies into the caller, so
# in driver mode (edge collector attached) the resolving lookup must be
# recorded and world-clamped like any inference lookup.
function resolve_single_match(st::UInferState, @nospecialize(sig))
    col = st.edges
    col === nothing && return resolve_single_match(sig, st.cfg.world)
    result = try
        CC.findall(sig, CC.InternalMethodTable(st.cfg.world); limit = 1)
    catch
        nothing
    end
    result === nothing && return nothing        # >1 methods or failed query
    record_call!(col, sig, result)
    result.ambig && return nothing              # ambiguous dispatch: stay dynamic
    length(result.matches) == 1 || return nothing
    match = result.matches[1]::Core.MethodMatch
    match.fully_covers || return nothing
    return match
end

# `invoke`'s method selection: the most specific method whose signature
# FULLY COVERS the declared type-tuple (jl_gf_invoke_lookup semantics) —
# non-covering intersections are irrelevant (invoke ignores runtime
# dispatch beyond the membership check). Edge-recorded: a callee-set
# change for `sig` invalidates the baked selection.
function resolve_invoke_lookup(st::UInferState, @nospecialize(sig))
    result = try
        CC.findall(sig, CC.InternalMethodTable(st.cfg.world); limit = 4)
    catch
        nothing
    end
    result === nothing && return nothing
    result.ambig && return nothing
    col = st.edges
    col === nothing || record_call!(col, sig, result)
    for m in result.matches
        m = m::Core.MethodMatch
        m.fully_covers && return m
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Multi-return normalization (deliverable 2a)
# ---------------------------------------------------------------------------

# Copy one region's contents into the builder's current open region,
# remapping statement/region references. Sibling owned regions are
# pre-created so cfg edge bundles between blocks resolve.
function _copy_normalized!(b::UnifiedIR.Builder, src::UnifiedIR.IR, cr::RegionId,
                           stmtmap::Dict{Int32,UnifiedIR.Operand},
                           regionmap::Dict{Int32,RegionId},
                           srcroot::RegionId, wrapper::RegionId)
    remap(o::UnifiedIR.Operand) = begin
        t = UnifiedIR.optag(o)
        if t == UnifiedIR.TAG_STMT
            r = get(stmtmap, UnifiedIR.asstmt(o).id, nothing)
            r === nothing && error("normalize_single_return!: forward reference %$(UnifiedIR.payload(o))")
            r
        elseif t == UnifiedIR.TAG_REGION || t == UnifiedIR.TAG_BLOCK
            nr = get(regionmap, Int32(UnifiedIR.payload(o)), nothing)
            nr === nothing && error("normalize_single_return!: unmapped region reference")
            UnifiedIR.mkoperand(t, nr.id)
        elseif t == UnifiedIR.TAG_CONST
            UnifiedIR.op_constidx(UnifiedIR.intern_const!(b.ir.body, src.body.constants[UnifiedIR.payload(o)]))
        elseif t == UnifiedIR.TAG_GLOBAL
            UnifiedIR.op_globalidx(UnifiedIR.intern_global!(b.ir.body, src.body.globals[UnifiedIR.payload(o)]))
        else
            o   # INLINE / SPARAM / NONE
        end
    end
    for s in UnifiedIR.region_stmts(src, cr)
        k = UnifiedIR.stmt_kind(src, s)
        if k === K"region_arg" && cr == srcroot
            continue   # function parameters were emitted in the new root
        end
        if k === K"return"
            vals = UnifiedIR.Operand[remap(UnifiedIR.getop(src, s, i))
                                     for i in 1:UnifiedIR.nops(src, s)]
            UnifiedIR.append_stmt!(b, K"break", UnifiedIR.op_region(wrapper), vals...)
            continue
        end
        ops = UnifiedIR.Operand[remap(UnifiedIR.getop(src, s, i))
                                for i in 1:UnifiedIR.nops(src, s)]
        ns = UnifiedIR.append_stmt!(b, k, ops...; type = UnifiedIR.stmt_type(src, s),
                                    flag = UnifiedIR.stmt_flag(src, s),
                                    debug = UnifiedIR.stmt_debug(src, s))
        stmtmap[s.id] = UnifiedIR.op_stmt(ns)
        if UnifiedIR.owns_regions(k)
            crids = UnifiedIR.live_owned_regions(src, s)
            for crid in crids   # pre-create all siblings (cfg edge targets)
                creg = UnifiedIR.getregion(src, crid)
                nr = UnifiedIR.Region(creg.kind, ns, UnifiedIR.current_region(b);
                                      activation = creg.activation)
                push!(b.ir.regions, nr)
                regionmap[crid.id] = RegionId(length(b.ir.regions))
            end
            for crid in crids
                nrid = regionmap[crid.id]
                nreg = UnifiedIR.getregion(b.ir, nrid)
                nreg.first = StmtId(Int(b.ir.body.len) + 1)
                push!(b.open, nrid)
                _copy_normalized!(b, src, crid, stmtmap, regionmap, srcroot, wrapper)
                nreg.last = StmtId(Int(b.ir.body.len))
                pop!(b.open)
            end
        end
    end
    return nothing
end

"""
    normalize_single_return!(callee::IR) -> IR

Pre-normalize a callee to the `splice_body!` matrix (§4.2): if the body has
more than one `return`, or its single return is not root-level, rebuild it
with the body wrapped in a single-iteration `loop` region — each `return v`
becomes `break ^wrapper (v)`, the loop's result is the returned value, and a
single `return %loop` follows. The result is verified at level 1.
"""
function normalize_single_return!(callee::UnifiedIR.IR)
    UnifiedIR.check_state(callee, UnifiedIR.LAYOUT_DENSE, "normalize_single_return!")
    root = UnifiedIR.root_region(callee)
    nret = 0
    rootret = true
    for i in 1:UnifiedIR.nstmts(callee)
        callee.body.kind[i] === K"return" || continue
        nret += 1
        UnifiedIR.stmt_region(callee, StmtId(Int32(i))) == root || (rootret = false)
    end
    (nret <= 1 && rootret) && return callee
    b = UnifiedIR.Builder(name = get(callee.meta, :name, :callee))
    append!(b.ir.argtypes, callee.argtypes)
    append!(b.ir.sptypes, callee.sptypes)
    b.ir.valid_worlds = callee.valid_worlds
    merge!(b.ir.meta, callee.meta)
    croot = UnifiedIR.getregion(callee, root)
    stmtmap = Dict{Int32,UnifiedIR.Operand}()
    regionmap = Dict{Int32,RegionId}()
    for a in croot.args
        na = UnifiedIR.append_stmt!(b, K"region_arg"; type = UnifiedIR.stmt_type(callee, a))
        stmtmap[a.id] = UnifiedIR.op_stmt(na)
    end
    rt = get(callee.meta, :rettype, Any)
    loop = UnifiedIR.append_stmt!(b, K"loop"; type = rt isa Type ? rt : Any)
    wrapper = UnifiedIR.open_region!(b, loop; kind = UnifiedIR.REGION_LOOP_BODY)
    regionmap[root.id] = wrapper
    _copy_normalized!(b, callee, root, stmtmap, regionmap, root, wrapper)
    UnifiedIR.close_region!(b)
    UnifiedIR.append_stmt!(b, K"return", loop)
    nir = UnifiedIR.finish!(b; verify = false)
    UnifiedIR.verify_ir(nir; level = 1)
    return nir
end

# ---------------------------------------------------------------------------
# Call/invoke inlining (deliverables 2a/2b/2d)
# ---------------------------------------------------------------------------

# Resolve an inlinable (method, method-instance, invoke_call) for a
# call/invoke statement, or nothing. Applies the dispatch-level legality
# checks only. `invoke_call` marks the `Core.invoke(f, types, args...)`
# call form, whose argument map skips the type-tuple operand.
function resolve_inline_target(ir::UnifiedIR.IR, s::StmtId, k::UnifiedIR.Kind, st::UInferState)
    if k === K"call"
        nop = UnifiedIR.nops(ir, s)
        args = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i)) for i in 1:nop]
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = args[1].val)
        if f === nothing
            # `singleton_type` returns nothing for `Type{X}` with
            # non-singleton `X` (TypeEq on this nightly) — a `T(args...)`
            # constructor call through a Type-valued argument still has a
            # unique callee (the B4 ctor-resolution finding)
            ft0 = CC.widenconst(args[1])
            if (ft0 isa DataType && CC.isType(ft0)) || CC.isTypeEq(ft0)
                p = CC.type_parameter(ft0)
                (p isa Type && !CC.has_free_typevars(p)) && (f = p)
            end
        end
        let po = args[1]
            if po isa CC.PartialOpaque
                # opaque-closure call: devirtualize to the closure's source
                # method (stock handle_opaque_closure_call!). Inlining drops
                # the implicit argument/return typeasserts, so it is only
                # legal when the static types already prove both.
                ocm = po.source
                (ocm isa Method && isdefined(ocm, :source) &&
                 !isdefined(ocm, :generator) && !ocm.isva) || return nothing
                local ocsig, ocrt, sig
                try
                    tt = po.typ
                    utt = Base.unwrap_unionall(tt)::DataType
                    ocargsig = Base.rewrap_unionall(utt.parameters[1], tt)
                    oa = Base.unwrap_unionall(ocargsig)
                    oa isa DataType || return nothing
                    ocsig = Base.rewrap_unionall(Tuple{Tuple, oa.parameters...}, ocargsig)
                    p2 = utt.parameters[2]
                    ocrt = Base.rewrap_unionall(p2 isa TypeVar ? p2.ub : p2, tt)
                    ocrt isa Type || return nothing
                    argts = Any[CC.widenconst(a) for a in args[2:end]]
                    any(t -> t === Union{}, argts) && return nothing
                    sig = Tuple{CC.widenconst(po.env), argts...}
                catch
                    return nothing
                end
                sig <: ocsig || return nothing
                sitet = UnifiedIR.stmt_type(ir, s)
                rok = try
                    CC.:⊑(CC.fallback_lattice, sitet === nothing ? Any : sitet, ocrt)
                catch
                    false
                end
                rok || return nothing
                match = Core.MethodMatch(sig, Core.svec(), ocm, true)
                return (ocm, CC.specialize_method(match), false)
            end
        end
        if f === Core.invoke && nop >= 3
            # Core.invoke(f2, types::Type{<:Tuple}, args...): the target
            # method is looked up on the DECLARED signature. Inlining drops
            # invoke's runtime argument check, so it is only legal when the
            # static argument types already prove membership.
            f2 = CC.singleton_type(args[2])
            f2 === nothing && args[2] isa CC.Const && (f2 = (args[2]::CC.Const).val)
            (f2 === nothing || f2 isa Core.Builtin || f2 isa Core.IntrinsicFunction) &&
                return nothing
            types = static_operand_value(ir, UnifiedIR.getop(ir, s, 3))
            (types isa Type && types <: Tuple && !CC.has_free_typevars(types)) ||
                return nothing
            types isa DataType || return nothing
            declared = types.parameters
            length(declared) == nop - 3 || return nothing
            for i in 4:nop
                at = CC.widenconst(args[i])
                d = declared[i - 3]
                (at isa Type && d isa Type && at <: d) || return nothing
            end
            sig = try
                Tuple{f2 isa Type ? Type{f2} : typeof(f2), declared...}
            catch
                return nothing
            end
            match = resolve_invoke_lookup(st, sig)
            match === nothing && return nothing
            return (match.method, CC.specialize_method(match), true)
        end
        f isa Core.Builtin && return nothing
        f isa Core.IntrinsicFunction && return nothing
        local ftt
        if f === nothing
            # non-singleton concrete callee: closure objects and other
            # callable structs dispatch on their concrete TYPE (the callee
            # value is callee parameter 1, so the ordinary argmap applies).
            # Values built by the unified closure machinery (K"closure")
            # are region activations, not callable structs — those stay
            # with their own machinery.
            fo2 = UnifiedIR.getop(ir, s, 1)
            if UnifiedIR.optag(fo2) == UnifiedIR.TAG_STMT &&
               UnifiedIR.stmt_kind(ir, skip_refines(ir, UnifiedIR.asstmt(fo2))) === K"closure"
                return nothing
            end
            ft0 = CC.widenconst(args[1])
            if ft0 isa DataType && isconcretetype(ft0) && !(ft0 <: Type) &&
               !(ft0 <: Core.Builtin) && !(ft0 <: Core.IntrinsicFunction) &&
               !(ft0 <: Core.OpaqueClosure)
                ftt = ft0
            else
                # non-const TYPE callee (a constructor through a computed
                # type — the `OldVal{i}()` shape): dispatch resolves on the
                # `Type{...}` lattice element itself. A single fully-covering
                # match then specializes with TypeVar sparams (stock's
                # allow_typevars=true single-match revisit); any sparam READS
                # in the body ride the pinned bake or the _compute_sparams
                # materialization below.
                tt = CC.unwrap_unionall(ft0)
                if tt isa DataType && CC.isType(tt)
                    ftt = ft0
                elseif CC.isTypeEq(tt)
                    # this nightly's exact-type lattice element: rebuild the
                    # dispatchable Type{...} form over the same environment
                    p = CC.type_parameter(tt)
                    ftt = try
                        Base.rewrap_unionall(Type{p}, ft0)
                    catch
                        return nothing
                    end
                else
                    return nothing
                end
            end
        else
            ftt = f isa Type ? Type{f} : typeof(f)
        end
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        any(t -> t === Union{}, argts) && return nothing
        sig = Tuple{ftt, argts...}
        match = resolve_single_match(st, sig)
        match === nothing && return nothing
        return (match.method, CC.specialize_method(match), false)
    else  # K"invoke"
        ci_op = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        mi = ci_op isa Core.CodeInstance ? ci_op.def : ci_op
        mi isa Core.MethodInstance || return nothing
        m = mi.def
        m isa Method || return nothing
        return (m, mi, false)
    end
end

"""A splice-able value for one `sparam_vals` entry. Plain values pass
through. An unresolved `TypeVar` or a constrained-TypeVar marker
(`svec(tv, flag)`) is only bakeable when the var's bounds PIN it
(`lb === ub` — the intersection admits exactly one binding, e.g. the
invariant `Ref{Any}` position that produces `svec(T>:Any, true)`); those
bake to the pinned bound. Anything else returns the `_unbakeable` sentinel
(stock handles the general case with a runtime `Core._compute_sparams`;
this port declines those splices instead)."""
struct _Unbakeable end
const _unbakeable = _Unbakeable()
function bakeable_sparam(@nospecialize(v))
    tv = v
    if v isa Core.SimpleVector
        (length(v) == 2 && v[1] isa TypeVar) || return _unbakeable
        tv = v[1]
    end
    if tv isa TypeVar
        tv.lb === tv.ub && return tv.ub
        return _unbakeable
    end
    return v
end

"""Collect the indices of callee-body `TAG_SPARAM` reads whose baked value
is `_unbakeable` (unique, insertion order). Unused parameters never block
inlining (their values are not materialized by `splice_body!`). Returns
`nothing` for an out-of-range read (env-depth mismatch — never inline)."""
function unbakeable_sparam_reads(callee::UnifiedIR.IR, spvals::Vector{Any})
    out = Int[]
    for s in UnifiedIR.each_stmt(callee)
        for j in 1:UnifiedIR.nops(callee, s)
            o = UnifiedIR.getop(callee, s, j)
            UnifiedIR.optag(o) == UnifiedIR.TAG_SPARAM || continue
            idx = Int(UnifiedIR.payload(o))
            idx <= length(spvals) || return nothing
            spvals[idx] === _unbakeable && !(idx in out) && push!(out, idx)
        end
    end
    return out
end

"""Substitute the callee method's static parameters into the STRUCTURAL
type slots of `cfunction`/`foreigncall` statements of a callee body about
to be spliced (stock `ssa_substitute_op!`'s cfunction/foreigncall arms).
Those slots are interned constants — `splice_body!`'s `TAG_SPARAM`
substitution never sees the TypeVars INSIDE them — so without this the
spliced body carries the CALLEE method's TypeVars into a caller whose
enclosing-method environment cannot resolve them, and codegen's
`verify_ref_type` hard-errors ("type Ref should have an element type, not
Ref{<:T}" — the libuv `@cfunction(_uv_hook_close, Cvoid, (Ref{T},))`
image-fatal class). Returns `false` (decline the splice) when a slot
references a typevar but `sparam_vals` is not fully static: stock
reconstructs VALUE reads through `spvals_ssa` in that regime but has no
runtime path for these structural slots either."""
function instantiate_foreign_type_slots!(callee::UnifiedIR.IR, m::Method,
                                         spvals::Core.SimpleVector)
    msig = m.sig
    msig isa UnionAll || return true
    static = !isempty(spvals) && CC.validate_sparams(spvals)
    inst(@nospecialize(t)) = ccall(:jl_instantiate_type_in_env, Any,
                                   (Any, Any, Ptr{Any}), t, msig, spvals)
    for s in UnifiedIR.each_stmt(callee)
        k = UnifiedIR.stmt_kind(callee, s)
        (k === K"cfunction" || k === K"foreigncall") || continue
        # operand layout mirrors the lowered Expr (codeinfo_entry stores the
        # pieces verbatim, in order):
        #   cfunction:   (output_type, fexpr, rt, argt, cconv) -> slots 3, 4
        #   foreigncall: (name, rt, argt, nreq, cconv, args...) -> slots 2, 3
        #   (the foreignglobal marker shifts the foreigncall layout by one)
        ofs = 0
        if k === K"foreigncall"
            o1 = UnifiedIR.getop(callee, s, 1)
            if UnifiedIR.optag(o1) == UnifiedIR.TAG_CONST &&
               UnifiedIR.getconst(callee, o1) === FOREIGNGLOBAL_MARKER
                ofs = 1
            end
        end
        for i in (k === K"cfunction" ? (3, 4) : (2 + ofs, 3 + ofs))
            i <= UnifiedIR.nops(callee, s) || return false
            o = UnifiedIR.getop(callee, s, i)
            UnifiedIR.optag(o) == UnifiedIR.TAG_CONST || continue
            t = UnifiedIR.getconst(callee, o)
            if t isa Core.SimpleVector
                any(x -> CC.has_free_typevars(x), Any[t...]) || continue
                static || return false
                t2 = Core.svec(Any[inst(x) for x in t]...)
            else
                CC.has_free_typevars(t) || continue
                static || return false
                t2 = inst(t)
            end
            UnifiedIR.setop!(callee, s, i, UnifiedIR.vop(callee, t2))
        end
    end
    return true
end

"""The effects-side half of stock `adjust_boundscheck!`: a callee inlined at
an `@inbounds`-flagged site enters an elided-boundscheck context, so every
spliced statement is marked FLAG_INBOUNDS — the post-optimization effects
recompute then keeps their boundscheck-guarded operations `noub`-tainted
(refine_bc_noub blocked, callee conditionals demoted at callsite_noub), and
the context stays transitive across inlining rounds (stock achieves this
batch-recursively through pre-inlined callee IR). The boundscheck VALUES
stay symbolic: the emitted code keeps its checks — `@inbounds` elision is a
separate (pure-performance) optimization this pipeline does not do yet.
Mutates the (dense, pre-splice) callee copy."""
function mark_inbounds_context!(callee::UnifiedIR.IR)
    for i in 1:UnifiedIR.nstmts(callee)
        s = StmtId(Int32(i))
        UnifiedIR.is_tombstone(callee, s) && continue
        UnifiedIR.add_flag!(callee, s, UnifiedIR.FLAG_INBOUNDS)
    end
    return callee
end

"""
    inline_calls2!(ir, state; params=InlineParams()) -> Int

Editable-session inlining via `splice_body!` for statically-resolved `call`
sites and `invoke` sites. Callees are entry-converted, normalized to single
return (multi-return supported through the loop wrapper), admitted by the
cost heuristic, and marked with the site's inbounds context
(`mark_inbounds_context!`). Returns the number of sites inlined.
"""
function inline_calls2!(ir::UnifiedIR.IR, state::UInferState;
                        params::InlineParams = InlineParams(),
                        materialize_sparams::Bool = false)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "inline_calls2!")
    caller_mi = get(ir.meta, :method_instance, nothing)
    caller_m = caller_mi isa Core.MethodInstance ? caller_mi.def : nothing
    inlined = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        k = UnifiedIR.stmt_kind(ir, s)
        (k === K"call" || k === K"invoke") || continue
        UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_NOINLINE != 0 && continue
        target = resolve_inline_target(ir, s, k, state)
        target === nothing && continue
        m, mi, invoke_call = target
        argofs = k === K"invoke" ? 1 : 0
        if invoke_call
            m.isva && continue
            Int(m.nargs) == UnifiedIR.nops(ir, s) - 2 || continue
        elseif m.isva
            # vararg callee: params 1..nargs-1 map positionally, the trailing
            # param receives a synthesized `Core.tuple` of the rest (the
            # body's packed-tuple convention — stock's va-handling)
            Int(m.nargs) >= 1 || continue
            UnifiedIR.nops(ir, s) - argofs >= Int(m.nargs) - 1 || continue
        else
            Int(m.nargs) == UnifiedIR.nops(ir, s) - argofs || continue
        end
        caller_m === m && continue                        # direct self-recursion
        src = if isdefined(m, :generator)
            # a staged method's runnable body is the generator's EXPANSION
            # for THIS specialization (stock retrieve_ir_for_inlining via
            # retrieve_code_info); expansion needs sufficiently concrete
            # specTypes — failures decline, so the f59018 abstract-specTypes
            # class keeps its dynamic call
            try
                CC.retrieve_code_info(mi, state.cfg.world)
            catch
                nothing
            end
        else
            try
                Base.uncompressed_ir(m)
            catch
                nothing
            end
        end
        src isa Core.CodeInfo || continue
        site_inline = UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_INLINE != 0
        # stock: a callsite `@inline` overrides the callee's declared
        # `@noinline` (the force_inline_explicit/f42078 family) and admits
        # without a cost check; the callsite `@noinline` (FLAG_NOINLINE,
        # checked above) symmetrically overrides a declared `@inline`
        src.inlining == 0x02 && !site_inline && continue  # @noinline callee
        declared_inline = src.inlining == 0x01
        callee_ir = try
            normalize_single_return!(codeinfo_to_ir(src; nargs = Int(m.nargs), name = m.name))
        catch e
            e isa Union{UnsupportedIR,UnifiedIR.VerifyError} || rethrow()
            nothing
        end
        callee_ir === nothing && continue
        # unresolved TypeVars and constrained-TypeVar markers (svec(tv, flag))
        # cannot be baked into the splice as plain sparam values — but that
        # only matters when the body actually READS the parameter (stock keys
        # the same decision off spvals_ssa/_compute_sparams; constructors of
        # diagonal-typevar methods are the common never-reads case), and
        # pinned markers (lb === ub) still have a unique bakeable value
        spvals = Any[bakeable_sparam(v) for v in mi.sparam_vals]
        spneeded = unbakeable_sparam_reads(callee_ir, spvals)
        spneeded === nothing && continue      # env-depth mismatch
        spstates = nothing
        if !isempty(spneeded)
            # settled-types phase only: stock inlines on FINAL inference
            # results, while these rounds iterate — materializing against a
            # round-1 under-refined specialization (marker sparams from a
            # still-Any argument) would burn the precise inline the next
            # rounds get for free (the UEA SafeRef corpus). The deferral is
            # recorded so the driver only pays the settled-phase inference
            # when a candidate actually exists.
            if !materialize_sparams
                ir.meta[:sparam_deferred] = true
                continue
            end
            # runtime sparam reconstruction (stock ir_prepare_inlining!'s
            # spvals_ssa + insert_spval!): `Core._compute_sparams` re-derives
            # the environment from the very arguments that dispatched here and
            # `Core._svec_ref` projects each needed entry — materialized at
            # the splice site below. Admitted only for plain-value reads of
            # guaranteed-defined parameters (stock's throw_undef_if_not
            # machinery for maybe-undef ones is not ported: decline).
            m.is_for_opaque_closure && continue
            # a site whose RESULT is unused keeps its call WHEN the callee's
            # interprocedural effects prove the site removable-if-unused:
            # stock carries the callee's optimized-IR nothrow flags through
            # the splice so the whole reconstruction chain DCEs; this
            # pipeline re-infers the spliced body and cannot re-prove
            # nothrow for the resulting apply_type/new chain (the
            # SparamUnused effects corpus) — for a removable callee the
            # un-inlined call's effects path is strictly better. A
            # NON-removable callee (finalizer-registering constructors, the
            # DoAllocNoEscapeSparam shape) gains nothing from the decline:
            # the call can never DCE, and only the inlined body lets the
            # finalizer machinery do its work (wave 11).
            if UnifiedIR.use_counts(ir)[s.id] == 0
                fx = opt_callee_effects(state, mi)
                (fx isa CC.Effects && CC.is_removable_if_unused(fx)) && continue
            end
            spstates = try
                CC.sptypes_from_meth_instance(mi)
            catch
                nothing
            end
            (spstates isa Vector{CC.VarState} &&
             length(spstates) >= maximum(spneeded) &&
             !any(i -> spstates[i].undef, spneeded)) || continue
        end
        instantiate_foreign_type_slots!(callee_ir, m, mi.sparam_vals) || continue
        if !site_inline
            # stock: never inline error paths — a Union{}-returning callee
            # keeps its (cold) call unless explicitly inline-annotated
            # (inline_cost_model's `!declared_inline && rt === Union{}` rule)
            if !declared_inline
                sitet = UnifiedIR.stmt_type(ir, s)
                (sitet isa Type && sitet === Union{}) && continue
            end
            # stock cost-model admission: the cached CodeInstance's
            # inlining_cost, or the same model computed over the optimized
            # callee body; the raw statement-count limits remain the
            # no-verdict fallback. OC callees skip the verdict: the
            # mi-generic body prices its capture-routed inner calls as
            # dynamic dispatch, but the SPLICED body devirtualizes them
            # through the captures forwarding (stock is deliberately
            # generous here — const_prop_methodinstance_heuristic's
            # is_for_opaque_closure arm — and inlines the const-prop
            # result, whose cost model never sees the dynamic shape)
            cost = m.is_for_opaque_closure ? nothing : inline2_cost(state, mi, src)
            if cost isa Int
                threshold = Compiler.OptimizationParams(state.cfg.interp).inline_cost_threshold
                declared_inline && (threshold += 19 * threshold)
                cost <= threshold || continue
            else
                limit = declared_inline ? params.inline_size_limit : params.size_limit
                UnifiedIR.nstmts(callee_ir) - Int(m.nargs) <= limit || continue
            end
        end
        # handler-bearing callees: stock declines these by default; admit them
        # only under an explicit @inline / FLAG_INLINE request (they exercise
        # the multi-return loop-wrapper normalization)
        if !(site_inline || declared_inline) &&
           any(i -> callee_ir.body.kind[i] === K"try", 1:UnifiedIR.nstmts(callee_ir))
            continue
        end
        # the site's boundscheck context (stock ir_inline_item!'s :off case):
        # an @inbounds-flagged site puts the spliced body in an
        # elided-checks context for the effects recompute
        if UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_INBOUNDS != 0
            mark_inbounds_context!(callee_ir)
        end
        argmap = if invoke_call
            # Core.invoke(f, types, args...): callee params map to (f, args...)
            ops = UnifiedIR.Operand[UnifiedIR.getop(ir, s, 2)]
            for i in 4:UnifiedIR.nops(ir, s)
                push!(ops, UnifiedIR.getop(ir, s, i))
            end
            ops
        elseif m.isva
            nfixed = Int(m.nargs) - 1
            ops = UnifiedIR.Operand[UnifiedIR.getop(ir, s, argofs + i)
                                    for i in 1:nfixed]
            extras = UnifiedIR.Operand[UnifiedIR.getop(ir, s, i)
                                       for i in (argofs + nfixed + 1):UnifiedIR.nops(ir, s)]
            vat = try
                Tuple{Any[CC.widenconst(stmt_lattice(ir, o)) for o in extras]...}
            catch
                Tuple
            end
            tup = UnifiedIR.insert_before!(ir, s, K"call",
                                           UnifiedIR.vop(ir, Core.tuple), extras...;
                                           type = vat, flag = UnifiedIR.FLAG_REMOVABLE |
                                                             UnifiedIR.FLAG_NOUB)
            push!(ops, UnifiedIR.op_stmt(tup))
            ops
        else
            UnifiedIR.Operand[UnifiedIR.getop(ir, s, i)
                              for i in (argofs + 1):UnifiedIR.nops(ir, s)]
        end
        if m.is_for_opaque_closure
            # the OC body's SELF slot is the capture environment: substitute a
            # captures load for argument 1 (stock ir_inline_spec_info!'s
            # `getfield(oc, :captures)` insertion); forward_extracts! then
            # forwards loads through it to the new_opaque_closure operands
            (invoke_call || m.isva || isempty(argmap)) && continue
            ocop = argmap[1]
            pol = stmt_lattice(ir, ocop)
            pol isa CC.PartialOpaque || continue
            capt = UnifiedIR.insert_before!(ir, s, K"call",
                UnifiedIR.vop(ir, Core.getfield), ocop, UnifiedIR.vop(ir, :captures);
                type = pol.env,
                flag = UnifiedIR.FLAG_REMOVABLE | UnifiedIR.FLAG_NOUB)
            argmap[1] = UnifiedIR.op_stmt(capt)
        end
        if !isempty(spneeded)
            spstates = spstates::Vector{CC.VarState}
            # stock's `Expr(:call, Core._compute_sparams, def, argexprs...)`:
            # built over the RAW caller-side value operands (pre va-fixup)
            spops = UnifiedIR.Operand[UnifiedIR.vop(ir, Core._compute_sparams),
                                      UnifiedIR.vop(ir, m)]
            if invoke_call
                push!(spops, UnifiedIR.getop(ir, s, 2))
                for i in 4:UnifiedIR.nops(ir, s)
                    push!(spops, UnifiedIR.getop(ir, s, i))
                end
            else
                for i in (argofs + 1):UnifiedIR.nops(ir, s)
                    push!(spops, UnifiedIR.getop(ir, s, i))
                end
            end
            spssa = UnifiedIR.insert_before!(ir, s, K"call", spops...;
                type = Core.SimpleVector,
                flag = UnifiedIR.FLAG_REMOVABLE | UnifiedIR.FLAG_NOUB)
            for idx in spneeded
                vs = UnifiedIR.insert_before!(ir, s, K"call",
                    UnifiedIR.vop(ir, Core._svec_ref), UnifiedIR.op_stmt(spssa),
                    UnifiedIR.vop(ir, idx);
                    type = spstates[idx].typ,
                    flag = UnifiedIR.FLAG_REMOVABLE | UnifiedIR.FLAG_NOUB)
                spvals[idx] = UnifiedIR.op_stmt(vs)
            end
        end
        UnifiedIR.splice_body!(ir, s, callee_ir; argmap, sparams = spvals)
        inlined += 1
    end
    return inlined
end

# ---------------------------------------------------------------------------
# _apply_iterate flattening (stock rewrite_apply_exprargs! — the CASES)
# ---------------------------------------------------------------------------

"""
    fold_apply_iterates!(ir) -> Int

`Core._apply_iterate(Base.iterate, f, containers...)` where every
container is a Const `Tuple`/`SimpleVector` or has a fixed-arity `Tuple`
type becomes the direct `f(elements...)` — Const containers contribute
constants, typed tuples contribute `extract` projections (pure loads,
inserted before the site). `Base.iterate` on those containers is the
identity protocol, so the rewrite preserves semantics exactly; anything
else (arrays, generators, unknown arity) is left alone. Editable state;
the direct call then resolves/inlines on later rounds.
"""
function fold_apply_iterates!(ir::UnifiedIR.IR)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "fold_apply_iterates!")
    n = 0
    unroll_ch = get(ir.meta, :apply_iter_unroll, nothing)
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        nop >= 3 || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, 1)) === Core._apply_iterate || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, 2)) === Base.iterate || continue
        unrolls = unroll_ch === nothing ? nothing :
                  get(unroll_ch::Dict{Int32,Vector{Tuple{Int,Int}}}, s.id, nothing)
        newops = UnifiedIR.Operand[UnifiedIR.getop(ir, s, 3)]
        # element plan: an existing operand, a tuple projection (op, idx,
        # type), or an iterate-protocol unroll (op, -count, nothing) resolved
        # at materialization time
        elems = Vector{Union{UnifiedIR.Operand,Tuple{UnifiedIR.Operand,Int,Any}}}()
        ok = true
        total = 0
        for i in 4:nop
            o = UnifiedIR.getop(ir, s, i)
            lat = stmt_lattice(ir, o)
            v = lat isa CC.Const ? lat.val : nothing
            if v isa Core.SimpleVector || v isa Tuple
                for j in 1:length(v)
                    push!(elems, UnifiedIR.vop(ir, v[j]))
                end
                total += length(v)
            else
                wt = CC.widenconst(lat)
                if wt isa DataType && wt <: Tuple && wt !== Tuple &&
                   !Base.any(p -> CC.isvarargtype(p), wt.parameters)
                    ps = wt.parameters
                    for j in 1:length(ps)
                        p = ps[j]
                        push!(elems, (o, j, p isa Type ? p : Any))
                    end
                    total += length(ps)
                else
                    # non-tuple container: inference recorded a provably
                    # exhausted fixed-length iterate unroll for this operand
                    # position, or the whole apply stays (stock
                    # rewrite_apply_exprargs' iterate materialization)
                    cnt = 0
                    if unrolls !== nothing
                        k = findfirst(u -> u[1] == i, unrolls)
                        k === nothing || (cnt = unrolls[k][2] + 1)
                    end
                    cnt == 0 && (ok = false; break)
                    push!(elems, (o, -(cnt - 1), nothing))
                    total += cnt - 1
                end
            end
            total <= 512 || (ok = false; break)
        end
        ok || continue
        itero = UnifiedIR.getop(ir, s, 2)
        for e in elems
            if e isa UnifiedIR.Operand
                push!(newops, e)
            else
                (o, j, pt) = e
                if j >= 1
                    ex = UnifiedIR.insert_before!(ir, s, K"extract", o,
                                                  UnifiedIR.op_inline(j); type = pt)
                    push!(newops, UnifiedIR.op_stmt(ex))
                else
                    # materialize the iterate chain: exactly the calls the
                    # runtime apply would make (the final exhausted call
                    # included — its effects are part of the semantics; DCE
                    # removes it when provably effect-free). Types come from
                    # the follow-up inference pass.
                    cnt = -j
                    stateo = nothing
                    for _ in 1:cnt
                        itc = stateo === nothing ?
                            UnifiedIR.insert_before!(ir, s, K"call", itero, o; type = Any) :
                            UnifiedIR.insert_before!(ir, s, K"call", itero, o, stateo; type = Any)
                        el = UnifiedIR.insert_before!(ir, s, K"extract",
                                UnifiedIR.op_stmt(itc), UnifiedIR.op_inline(1); type = Any)
                        st2 = UnifiedIR.insert_before!(ir, s, K"extract",
                                UnifiedIR.op_stmt(itc), UnifiedIR.op_inline(2); type = Any)
                        push!(newops, UnifiedIR.op_stmt(el))
                        stateo = UnifiedIR.op_stmt(st2)
                    end
                    # the final, nothing-returning iterate call
                    stateo === nothing ?
                        UnifiedIR.insert_before!(ir, s, K"call", itero, o; type = Any) :
                        UnifiedIR.insert_before!(ir, s, K"call", itero, o, stateo; type = Any)
                end
            end
        end
        # the apply site's inlining/inbounds context binds the rewritten
        # direct call (a callsite `@noinline f(args...)` must keep the
        # invoke — the Base.allocated measurement shape)
        sitebits = UnifiedIR.stmt_flag(ir, s) &
                   (UnifiedIR.FLAG_INLINE | UnifiedIR.FLAG_NOINLINE |
                    UnifiedIR.FLAG_INBOUNDS)
        UnifiedIR.replace_stmt!(ir, s, K"call", newops...;
                                type = UnifiedIR.stmt_type(ir, s),
                                flag = sitebits == 0 ? nothing : sitebits)
        n += 1
    end
    return n
end

"""
    svecify_apply_args!(ir) -> Int

Residual `Core._apply_iterate` statements reach codegen as dynamic apply
calls. Stock's `lift_apply_args!` (#59548) rewrites each fixed-shape
`Tuple`-typed container argument into a `Core.svec(...)` call — svec's
boxed layout matches codegen's apply ABI. Port: a container operand whose
def is a `Core.tuple` call reuses its element operands; otherwise a known
fixed-arity tuple type spreads through `extract` projections. Non-tuple
containers are left alone. Runs once after the final inference pass (the
svec type would only degrade the apply's flattening if re-inferred).
"""
function svecify_apply_args!(ir::UnifiedIR.IR)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "svecify_apply_args!")
    n = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        nop >= 4 || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, 1)) === Core._apply_iterate || continue
        newops = UnifiedIR.Operand[UnifiedIR.getop(ir, s, i) for i in 1:nop]
        changed = false
        for i in 4:nop
            o = newops[i]
            wt = CC.widenconst(stmt_lattice(ir, o))
            (wt isa DataType && wt.name === Tuple.name) || continue
            svecops = nothing
            if UnifiedIR.optag(o) == UnifiedIR.TAG_STMT
                d = UnifiedIR.asstmt(o)
                if !UnifiedIR.is_tombstone(ir, d) &&
                   UnifiedIR.stmt_kind(ir, d) === K"call" &&
                   static_operand_value(ir, UnifiedIR.getop(ir, d, 1)) === Core.tuple
                    svecops = UnifiedIR.Operand[UnifiedIR.getop(ir, d, j)
                                                for j in 2:UnifiedIR.nops(ir, d)]
                end
            end
            if svecops === nothing
                ps = wt.parameters
                (!isempty(ps) && !Base.any(p -> CC.isvarargtype(p), ps)) || continue
                svecops = UnifiedIR.Operand[]
                for j in 1:length(ps)
                    p = ps[j]
                    ex = UnifiedIR.insert_before!(ir, s, K"extract", o,
                                                  UnifiedIR.op_inline(j);
                                                  type = p isa Type ? p : Any)
                    push!(svecops, UnifiedIR.op_stmt(ex))
                end
            end
            sv = UnifiedIR.insert_before!(ir, s, K"call",
                                          UnifiedIR.vop(ir, Core.svec), svecops...;
                                          type = Core.SimpleVector)
            newops[i] = UnifiedIR.op_stmt(sv)
            changed = true
        end
        if changed
            UnifiedIR.replace_stmt!(ir, s, K"call", newops...;
                                    type = UnifiedIR.stmt_type(ir, s))
            n += 1
        end
    end
    # NB: no DCE here — the pipeline tail runs in the editable layout
    # (dce! is dense-only; calling it made every rewritten body error into
    # the stock fallback). The replaced tuple ctors stay as dead statements;
    # they are effect-free calls codegen ignores.
    return n
end

# ---------------------------------------------------------------------------
# Union-split inlining (deliverable 2c)
# ---------------------------------------------------------------------------

"""
    union_split_calls!(ir, state; params=InlineParams()) -> Int

For a `call` with a Union-typed SSA argument where inference leaves ≤
`params.max_union_split` applicable methods (exactly one per union
component), emit an isa-dispatch step via `wrap_in_if!`:

    %c = isa(x, T1)
    %r = if %c { <call with x refined to T1> ; result }        # then-arm
         else  { <call with x refined to T2|…> ; result }      # residual

with the result threaded through the if-result by `wrap_in_if!`. `refine`
statements carry the component types, so inference (whose `UCond` machinery
refines the isa subject inside each arm) makes the arm calls statically
resolvable — subsequent rounds inline them and peel the residual further.
"""
function union_split_calls!(ir::UnifiedIR.IR, state::UInferState;
                            params::InlineParams = InlineParams())
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "union_split_calls!")
    nsplit = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        nsplit >= params.split_budget && break
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_NOINLINE != 0 && continue
        nop = UnifiedIR.nops(ir, s)
        args = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i)) for i in 1:nop]
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = args[1].val)
        f === nothing && continue
        f === SPARAM_READ_MARKER && continue   # pseudo-call, not a dispatch site
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        any(t -> t === Union{}, argts) && continue
        j = 0
        comps = Any[]
        if f isa Core.Builtin || f isa Core.IntrinsicFunction
            # modify-op family only: split the Union-typed VALUE argument so
            # each arm's op signature resolves to a single match and
            # devirtualize_modifyops! marks both arms (stock's wrapper
            # union-split shape — two Expr(:invoke_modify) sites)
            f === Core.modifyfield! || continue
            pos = modifyop_positions(f)
            pos === nothing && continue
            (minargs, maxargs, op_argi, v_argi) = pos
            (minargs <= nop <= maxargs) || continue
            vt = argts[v_argi - 1]
            vt isa Union || continue
            o = UnifiedIR.getop(ir, s, v_argi)
            UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || continue
            UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(o)) === K"cell_get" && continue
            TFw = try
                CC.widenconst(CC.getfield_tfunc(CC.fallback_lattice, args[2], args[3]))
            catch
                continue
            end
            opft = CC.widenconst(args[op_argi])
            (opft isa Type && TFw isa Type && TFw !== Union{} &&
             !CC.has_free_typevars(TFw) && !CC.has_free_typevars(opft)) || continue
            cs = Base.uniontypes(vt)
            2 <= length(cs) <= params.max_union_split || continue
            all(c -> resolve_single_match(state, Tuple{opft, TFw, c}) !== nothing,
                cs) || continue
            j = v_argi
            comps = cs
        else
            ft = f isa Type ? Type{f} : typeof(f)
            # already statically resolvable: plain inlining handles it
            resolve_single_match(state, Tuple{ft, argts...}) !== nothing && continue
            # find a splittable argument: SSA (non-cell_get) Union with one
            # applicable method per component
            for i in 2:nop
                t = argts[i - 1]
                t isa Union || continue
                o = UnifiedIR.getop(ir, s, i)
                UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || continue
                UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(o)) === K"cell_get" && continue
                cs = Base.uniontypes(t)
                2 <= length(cs) <= params.max_union_split || continue
                all(c -> resolve_single_match(state, Tuple{ft, argts[1:i-2]..., c,
                                                           argts[i:end]...}) !== nothing, cs) || continue
                j = i
                comps = cs
                break
            end
            if j == 0
                # no splittable Union argument: try the match-based split
                # (abstract callsites — stock's union-split devirtualization
                # with the method-error fallback edge)
                match_split_call!(ir, s, state, ft, argts) && (nsplit += 1)
                continue
            end
        end
        xop = UnifiedIR.getop(ir, s, j)
        T1 = comps[1]
        residual = length(comps) == 2 ? comps[2] : Union{comps[2:end]...}
        rt = UnifiedIR.stmt_type(ir, s)
        callops = UnifiedIR.operands(ir, s)
        isacall = UnifiedIR.insert_before!(ir, s, K"call", UnifiedIR.vop(ir, isa), xop,
                                           UnifiedIR.vop(ir, T1);
                                           type = Bool, flag = UnifiedIR.FLAG_PURE)
        resused = UnifiedIR.use_counts(ir)[s.id] > 0
        UnifiedIR.wrap_in_if!(ir, s, s, isacall; else_arm = (ir2, er) -> begin
            rx = UnifiedIR.push_stmt!(ir2, er, K"refine", xop; type = residual)
            cops = copy(callops)
            cops[j] = UnifiedIR.op_stmt(rx)
            cc = UnifiedIR.push_stmt!(ir2, er, K"call", cops...; type = rt)
            if resused
                UnifiedIR.push_stmt!(ir2, er, K"result", UnifiedIR.op_stmt(cc))
            else
                UnifiedIR.push_stmt!(ir2, er, K"result")
            end
        end)
        # narrow the guarded call's argument inside the then-arm
        rx1 = UnifiedIR.insert_before!(ir, s, K"refine", xop; type = T1)
        UnifiedIR.setop!(ir, s, j, UnifiedIR.op_stmt(rx1))
        nsplit += 1
    end
    return nsplit
end

# The single tested argument position for a match: exactly one position
# whose static type is not already inside the match's signature slot (v1
# emits one isa test per arm). Returns (position j in operand numbering,
# narrowed type) or nothing.
function match_test_position(@nospecialize(specT), argts::Vector{Any})
    specT isa DataType || return nothing
    ps = specT.parameters
    length(ps) == length(argts) + 1 || return nothing
    j = 0
    local Tj
    for i in 1:length(argts)
        p = ps[i + 1]
        p isa Type || return nothing
        argts[i] <: p && continue
        j == 0 || return nothing   # more than one tested position
        j = i + 1
        Tj = p
    end
    j == 0 && return nothing
    return (j, Tj)
end

"""
    match_split_call!(ir, s, st, ft, argts) -> Bool

Union-split DEVIRTUALIZATION for abstract callsites (stock
`ssair/inlining.jl` union splitting, the non-`Union`-argument face): a
dynamic `call` whose full signature has ≤ 2 applicable methods (complete,
unambiguous, edge-recorded `findall`) becomes an isa-dispatch:

    %c = isa(x, T₁)                       # match₁'s tested position
    %r = if %c { call f(x::refine T₁) }   # resolves to match₁ next round
         else  { … }

with the else arm one of: a direct `invoke` of match₂ (both-matches, sig
fully covered — the else values provably dispatch there), a nested
isa-guarded call of match₂ plus a `Core.throw_methoderror` arm (not
covered), or the method-error call alone (single non-covering match).
Soundness: the recorded match edge caps this body whenever the callee set
changes; per-position `isa` tests decide tuple membership exactly because
match signatures with free typevars (cross-position constraints) are
refused; completeness of `findall` (not truncated) makes the residual
dispatch-exact.
"""
function match_split_call!(ir::UnifiedIR.IR, s::StmtId, st::UInferState,
                           @nospecialize(ft), argts::Vector{Any})
    nop = UnifiedIR.nops(ir, s)
    Base.any(t -> !(t isa Type) || CC.has_free_typevars(t), argts) && return false
    sig = try
        Tuple{ft, argts...}
    catch
        return false
    end
    world = st.cfg.world
    result = try
        CC.findall(sig, CC.InternalMethodTable(world); limit = 2)
    catch
        nothing
    end
    result === nothing && return false
    result.ambig && return false
    n = length(result.matches)
    n <= 2 || return false
    col = st.edges
    col === nothing || record_call!(col, sig, result)
    if n == 0
        callops0 = UnifiedIR.operands(ir, s)
        # no applicable method at all: the call is a guaranteed MethodError
        # (stock rewrites these to the throwing form outright)
        UnifiedIR.replace_stmt!(ir, s, K"call",
                                UnifiedIR.vop(ir, Core.throw_methoderror),
                                callops0...; type = Union{})
        return true
    end
    match1 = result.matches[1]::Core.MethodMatch
    spec1 = match1.spec_types
    CC.has_free_typevars(spec1) && return false
    tp1 = match_test_position(spec1, argts)
    tp1 === nothing && return false
    j1, T1 = tp1
    # the tested-arm call must become statically resolvable under the
    # narrowed signature (method shadowing gives a single covering match)
    sig1 = Tuple{ft, argts[1:j1-2]..., T1, argts[j1:end]...}
    resolve_single_match(st, sig1) === nothing && return false
    covered = false
    local match2, spec2
    if n == 2
        match2 = result.matches[2]::Core.MethodMatch
        spec2 = match2.spec_types
        CC.has_free_typevars(spec2) && return false
        covered = sig <: Union{spec1, spec2}
    end
    rt = UnifiedIR.stmt_type(ir, s)
    callops = UnifiedIR.operands(ir, s)
    xop1 = UnifiedIR.getop(ir, s, j1)
    resused = UnifiedIR.use_counts(ir)[s.id] > 0
    interp = st.cfg.interp
    if covered
        # else-arm values lie in spec2 minus spec1: dispatch-exact invoke
        mi2 = try
            CC.specialize_method(match2)
        catch
            return false
        end
        mi2 isa Core.MethodInstance || return false
        tgt = ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi2)
        tgt isa Core.MethodInstance || return false
        # stock's :invoke legality (compileable_specialization; the same rule
        # devirtualize_calls! applies): the target's static parameters must
        # be fully determined by the compilable signature — an
        # under-constrained environment makes the runtime's per-call sparam
        # re-derivation throw `UndefVarError: T ... in static parameter
        # matching` (the demo's zoo7/comprehension class). Such sites keep
        # the dynamic :call.
        sparams2 = tgt.sparam_vals
        (CC.unionall_depth((match2.method).sig) == length(sparams2) &&
         CC.validate_sparams(sparams2)) || return false
        (ci, _) = driver_ci_for_invoke(interp, tgt, true)
        if ci isa Core.CodeInstance && col isa UEdges
            clamp_world!(col, ci.min_world, ci.max_world) || (ci = nothing)
        end
        invtgt = ci === nothing ? tgt : ci
        isacall = UnifiedIR.insert_before!(ir, s, K"call", UnifiedIR.vop(ir, isa),
                                           xop1, UnifiedIR.vop(ir, T1);
                                           type = Bool, flag = UnifiedIR.FLAG_PURE)
        UnifiedIR.wrap_in_if!(ir, s, s, isacall; else_arm = (ir2, er) -> begin
            iops = UnifiedIR.Operand[UnifiedIR.vop(ir2, invtgt)]
            sp2 = spec2 isa DataType ? spec2.parameters : nothing
            for i in 1:length(callops)
                o = callops[i]
                if sp2 !== nothing && i >= 2 && sp2[i] isa Type && !(argts[i-1] <: sp2[i])
                    rx = UnifiedIR.push_stmt!(ir2, er, K"refine", o; type = sp2[i])
                    o = UnifiedIR.op_stmt(rx)
                end
                push!(iops, o)
            end
            iv = UnifiedIR.push_stmt!(ir2, er, K"invoke", iops...; type = rt)
            if resused
                UnifiedIR.push_stmt!(ir2, er, K"result", UnifiedIR.op_stmt(iv))
            else
                UnifiedIR.push_stmt!(ir2, er, K"result")
            end
        end)
    elseif n == 2
        # not covered: nested isa guard for match2, then the method error
        tp2 = match_test_position(spec2, argts)
        tp2 === nothing && return false
        j2, T2 = tp2
        sig2 = Tuple{ft, argts[1:j2-2]..., T2, argts[j2:end]...}
        resolve_single_match(st, sig2) === nothing && return false
        xop2 = UnifiedIR.getop(ir, s, j2)
        isacall = UnifiedIR.insert_before!(ir, s, K"call", UnifiedIR.vop(ir, isa),
                                           xop1, UnifiedIR.vop(ir, T1);
                                           type = Bool, flag = UnifiedIR.FLAG_PURE)
        UnifiedIR.wrap_in_if!(ir, s, s, isacall; else_arm = (ir2, er) -> begin
            c2 = UnifiedIR.push_stmt!(ir2, er, K"call", UnifiedIR.vop(ir2, isa),
                                      xop2, UnifiedIR.vop(ir2, T2);
                                      type = Bool, flag = UnifiedIR.FLAG_PURE)
            nif = UnifiedIR.push_stmt!(ir2, er, K"if", UnifiedIR.op_stmt(c2);
                                       type = rt)
            a1 = UnifiedIR.new_region!(ir2, nif, UnifiedIR.REGION_ARM)
            rx2 = UnifiedIR.push_stmt!(ir2, a1, K"refine", xop2; type = T2)
            cops = copy(callops)
            cops[j2] = UnifiedIR.op_stmt(rx2)
            cc = UnifiedIR.push_stmt!(ir2, a1, K"call", cops...; type = rt)
            if resused
                UnifiedIR.push_stmt!(ir2, a1, K"result", UnifiedIR.op_stmt(cc))
            else
                UnifiedIR.push_stmt!(ir2, a1, K"result")
            end
            a2 = UnifiedIR.new_region!(ir2, nif, UnifiedIR.REGION_ARM)
            th = UnifiedIR.push_stmt!(ir2, a2, K"call",
                                      UnifiedIR.vop(ir2, Core.throw_methoderror),
                                      callops...; type = Union{})
            if resused
                UnifiedIR.push_stmt!(ir2, a2, K"result", UnifiedIR.op_stmt(th))
            else
                UnifiedIR.push_stmt!(ir2, a2, K"result")
            end
            if resused
                UnifiedIR.push_stmt!(ir2, er, K"result", UnifiedIR.op_stmt(nif))
            else
                UnifiedIR.push_stmt!(ir2, er, K"result")
            end
        end)
    else
        # single non-covering match: guarded call + the method error
        isacall = UnifiedIR.insert_before!(ir, s, K"call", UnifiedIR.vop(ir, isa),
                                           xop1, UnifiedIR.vop(ir, T1);
                                           type = Bool, flag = UnifiedIR.FLAG_PURE)
        UnifiedIR.wrap_in_if!(ir, s, s, isacall; else_arm = (ir2, er) -> begin
            th = UnifiedIR.push_stmt!(ir2, er, K"call",
                                      UnifiedIR.vop(ir2, Core.throw_methoderror),
                                      callops...; type = Union{})
            if resused
                UnifiedIR.push_stmt!(ir2, er, K"result", UnifiedIR.op_stmt(th))
            else
                UnifiedIR.push_stmt!(ir2, er, K"result")
            end
        end)
    end
    # narrow the guarded call's argument inside the then-arm
    rx1 = UnifiedIR.insert_before!(ir, s, K"refine", xop1; type = T1)
    UnifiedIR.setop!(ir, s, j1, UnifiedIR.op_stmt(rx1))
    return true
end
