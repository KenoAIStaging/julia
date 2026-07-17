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

# A "single match" is only the dispatch outcome when it also FULLY COVERS
# the queried signature: a non-covering match means some argument tuples in
# `sig` dispatch to a MethodError, and baking the method's body in (or
# invoking it directly) would run it for those too.
function resolve_single_match(@nospecialize(sig), world::UInt)
    matches = try
        Base._methods_by_ftype(sig, 1, world)
    catch
        nothing
    end
    (matches === nothing || matches === false || length(matches) != 1) && return nothing
    match = matches[1]::Core.MethodMatch
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
    length(result.matches) == 1 || return nothing
    match = result.matches[1]::Core.MethodMatch
    match.fully_covers || return nothing
    return match
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

# Resolve an inlinable (method, method-instance) for a call/invoke statement,
# or nothing. Applies the dispatch-level legality checks only.
function resolve_inline_target(ir::UnifiedIR.IR, s::StmtId, k::UnifiedIR.Kind, st::UInferState)
    if k === K"call"
        nop = UnifiedIR.nops(ir, s)
        args = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i)) for i in 1:nop]
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = args[1].val)
        (f === nothing || f isa Core.Builtin || f isa Core.IntrinsicFunction) && return nothing
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        any(t -> t === Union{}, argts) && return nothing
        sig = Tuple{f isa Type ? Type{f} : typeof(f), argts...}
        match = resolve_single_match(st, sig)
        match === nothing && return nothing
        return (match.method, CC.specialize_method(match))
    else  # K"invoke"
        ci_op = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        mi = ci_op isa Core.CodeInstance ? ci_op.def : ci_op
        mi isa Core.MethodInstance || return nothing
        m = mi.def
        m isa Method || return nothing
        return (m, mi)
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

"""Does the callee body read (`TAG_SPARAM` operand) a static parameter whose
baked value is `_unbakeable`? Unused parameters do not block inlining
(their values are never materialized by `splice_body!`)."""
function reads_unbakeable_sparam(callee::UnifiedIR.IR, spvals::Vector{Any})
    for s in UnifiedIR.each_stmt(callee)
        for j in 1:UnifiedIR.nops(callee, s)
            o = UnifiedIR.getop(callee, s, j)
            UnifiedIR.optag(o) == UnifiedIR.TAG_SPARAM || continue
            idx = Int(UnifiedIR.payload(o))
            idx <= length(spvals) || return true
            spvals[idx] === _unbakeable && return true
        end
    end
    return false
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
                        params::InlineParams = InlineParams())
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
        m, mi = target
        # a staged method's runnable body is the generator's EXPANSION;
        # `uncompressed_ir(m)` below is not it — never inline those
        isdefined(m, :generator) && continue
        argofs = k === K"invoke" ? 1 : 0
        m.isva && continue
        Int(m.nargs) == UnifiedIR.nops(ir, s) - argofs || continue
        caller_m === m && continue                        # direct self-recursion
        src = try
            Base.uncompressed_ir(m)
        catch
            nothing
        end
        src === nothing && continue
        src.inlining == 0x02 && continue                  # @noinline callee
        limit = (src.inlining == 0x01 ||
                 UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_INLINE != 0) ?
                params.inline_size_limit : params.size_limit
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
        reads_unbakeable_sparam(callee_ir, spvals) && continue
        UnifiedIR.nstmts(callee_ir) - Int(m.nargs) <= limit || continue
        # handler-bearing callees: stock declines these by default; admit them
        # only under an explicit @inline / FLAG_INLINE request (they exercise
        # the multi-return loop-wrapper normalization)
        if limit == params.size_limit &&
           any(i -> callee_ir.body.kind[i] === K"try", 1:UnifiedIR.nstmts(callee_ir))
            continue
        end
        # the site's boundscheck context (stock ir_inline_item!'s :off case):
        # an @inbounds-flagged site puts the spliced body in an
        # elided-checks context for the effects recompute
        if UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_INBOUNDS != 0
            mark_inbounds_context!(callee_ir)
        end
        argmap = UnifiedIR.Operand[UnifiedIR.getop(ir, s, i)
                                   for i in (argofs + 1):UnifiedIR.nops(ir, s)]
        UnifiedIR.splice_body!(ir, s, callee_ir; argmap, sparams = spvals)
        inlined += 1
    end
    return inlined
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
        (f === nothing || f isa Core.Builtin || f isa Core.IntrinsicFunction) && continue
        ft = f isa Type ? Type{f} : typeof(f)
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        any(t -> t === Union{}, argts) && continue
        # already statically resolvable: plain inlining handles it
        resolve_single_match(state, Tuple{ft, argts...}) !== nothing && continue
        # find a splittable argument: SSA (non-cell_get) Union with one
        # applicable method per component
        j = 0
        comps = Any[]
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
