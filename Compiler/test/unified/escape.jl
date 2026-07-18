# B4: the EscapeAnalysis.jl test corpus re-expressed over the UnifiedIR-native
# escape analysis (Compiler/src/unified/escape.jl). Included from runtests.jl;
# self-sufficient apart from the LOAD_PATH preamble.
#
# Addressing is structural (StmtId / argument position), never positional SSA
# integers. Fixtures are built through a reduced unified pipeline
# (`ea_pipeline!`: inference + effect refinement + canonicalization + cell
# promotion + inlining, WITHOUT the allocation-dissolving passes — immutable
# load forwarding, mutable SROA, finalizer resolution) so the analysis sees
# the same post-inlining/pre-SROA shapes stock EA analyzes; that is also the
# pipeline point the optimizer-integration round will consume EA results at.
#
# Mapping tally and per-fixture dispositions: /workspace/B4-EA-NOTES.md.

module unified_test_EA

using Test
import Compiler
const CC = Compiler
const U = Compiler.load_unified!()
using UnifiedIR
using UnifiedIR: StmtId, @K_str

# ---------------------------------------------------------------------------
# Harness (the EAUtils equivalent)
# ---------------------------------------------------------------------------

const has_no_escape       = U.has_no_escape
const has_arg_escape      = U.has_arg_escape
const has_return_escape   = U.has_return_escape
const has_thrown_escape   = U.has_thrown_escape
const has_all_escape      = U.has_all_escape
const is_load_forwardable = U.is_load_forwardable
const ignore_argescape    = U.ignore_argescape

struct EAResult
    ir::UnifiedIR.IR
    state::U.UEscapeState
end
Base.getindex(r::EAResult, s::StmtId) = r.state[s]

"Escape info of argument position i (stock `state[Argument(i)]`)."
arg(r::EAResult, i::Int) = U.argescape(r.state, i)

aliased(r::EAResult, x::StmtId, y::StmtId) = U.isaliased(r.state, x, y)
aliased(r::EAResult, i::Int, y::StmtId) = U.isaliased_arg(r.state, i, y)

# --- pipeline ---------------------------------------------------------------

const EA_STATE = Ref{Any}(nothing)
function ea_state()
    w = Base.get_world_counter()
    st = EA_STATE[]
    if st === nothing || st.cfg.world != w
        st = U.UInferState(U.UInferConfig(world = w, max_depth = 12,
                                          frame_budget = 1000))
        EA_STATE[] = st
        empty!(EA_SUMMARIES)
    end
    return st
end

"""
    devirtualize_calls!(ir, st) -> Int

Statically-resolved-but-not-inlined `call` sites become `K"invoke"`
statements (stock's inliner does this rewrite for declined/`@noinline`
candidates — it is what gives stock EA its interprocedural hook). Editable
state; single fully-covering match only.
"""
function devirtualize_calls!(ir, st; allow_typevars::Bool = false)
    n = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        k = UnifiedIR.stmt_kind(ir, s)
        is_invoke = k === K"invoke"
        (k === K"call" || is_invoke) || continue
        # for existing invokes (created by union splitting or the driver with
        # jl_normalize_to_compilable_mi-WIDENED targets, stock's :invoke
        # convention): re-specialize the target at the site-precise argument
        # types so the interprocedural summary isn't computed on `Any` params
        # (stock EA recovers this precision through inference-time recursion)
        prev_mi = nothing
        if is_invoke
            ci0 = U.static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
            prev_mi = ci0 isa Core.CodeInstance ? ci0.def : ci0
            prev_mi isa Core.MethodInstance || continue
        end
        fo = UnifiedIR.getop(ir, s, is_invoke ? 2 : 1)
        f = U.static_operand_value(ir, fo)
        if f === nothing
            # `singleton_type` declines `Type{X}` for non-singleton-typed X
            # (TypeEq on this nightly) — a `T(args...)` call through a
            # Type-valued argument still has a unique callee
            ft = CC.widenconst(U.stmt_lattice(ir, fo))
            if (ft isa DataType && CC.isType(ft)) || CC.isTypeEq(ft)
                p = CC.type_parameter(ft)
                (p isa Type && !CC.has_free_typevars(p)) && (f = p)
            end
        end
        (f === nothing || f isa Core.Builtin || f isa Core.IntrinsicFunction) && continue
        f isa Core.TypeofVararg && continue
        argts = Any[CC.widenconst(U.stmt_lattice(ir, UnifiedIR.getop(ir, s, i)))
                    for i in (is_invoke ? 3 : 2):UnifiedIR.nops(ir, s)]
        any(t -> !(t isa Type) || t === Union{}, argts) && continue
        sig = try
            Tuple{f isa Type ? Type{f} : typeof(f), argts...}
        catch
            continue
        end
        # ambiguity-aware single-match resolution (`_methods_by_ftype` can
        # report one fully-covering match while dispatch is ambiguous on a
        # subset — such a site must stay dynamic)
        lookup = try
            CC.findall(sig, CC.InternalMethodTable(st.cfg.world); limit = 1)
        catch
            nothing
        end
        lookup === nothing && continue
        (length(lookup.matches) == 1 && !lookup.ambig) || continue
        match = lookup.matches[1]::Core.MethodMatch
        match.fully_covers || continue
        mi = try
            CC.specialize_method(match)
        catch
            continue
        end
        if !allow_typevars
            # mid-pipeline: pinning an under-constrained specialization (from
            # not-yet-refined argument types) would block later inlining
            any(v -> v isa TypeVar || v isa Core.SimpleVector, mi.sparam_vals) && continue
        end
        if is_invoke
            (match.method === (prev_mi::Core.MethodInstance).def && mi !== prev_mi) || continue
            ops = UnifiedIR.operands(ir, s)
            ops[1] = UnifiedIR.vop(ir, mi)
            UnifiedIR.replace_stmt!(ir, s, K"invoke", ops...;
                                    type = UnifiedIR.stmt_type(ir, s))
        else
            UnifiedIR.replace_stmt!(ir, s, K"invoke", UnifiedIR.vop(ir, mi),
                                    UnifiedIR.operands(ir, s)...;
                                    type = UnifiedIR.stmt_type(ir, s))
        end
        n += 1
    end
    return n
end

"Reduced optimizer pipeline: post-inlining, pre-scalar-replacement (see the
file header)."
function ea_pipeline!(ir, args; rounds::Int = 5)
    st = ea_state()
    for _ in 1:rounds
        changed = 0
        U.infer_ir!(ir, args; state = st)
        changed += U.refine_effects!(ir)
        changed += U.materialize_consts!(ir)
        changed += U.canonicalize_getfields!(ir)
        changed += U.fold_splatnews!(ir)
        changed += U.fold_pure_queries!(ir)
        changed += U.forward_refines!(ir)
        changed += U.forward_if_results!(ir)
        changed += UnifiedIR.promote_cells!(ir)
        changed += U.promote_block_cells!(ir)
        changed += UnifiedIR.dce!(ir)
        UnifiedIR.editable(ir)
        _, folded = UnifiedIR.fold_constant_branches!(ir)
        changed += folded
        changed += U.fold_island_branches!(ir)
        changed += U.drop_unreachable_blocks!(ir)
        changed += U.merge_goto_chains!(ir)
        changed += U.structurize!(ir)
        changed += U.dissolve_islands!(ir)
        while true
            c = U.promote_undef_cells!(ir)
            c += U.promote_arm_cells!(ir)
            c += UnifiedIR.promote_try_cells!(ir)
            c += U.promote_island_cells!(ir)
            c += U.promote_loop_cells!(ir)
            c == 0 && break
            changed += c
        end
        # (selectify! deliberately omitted: EA analyzes the pre-select-
        # conversion point, keeping distinct return sites — stock's IR shape)
        changed += U.fold_uniform_block_args!(ir)
        changed += U.fold_isdefineds!(ir)
        changed += U.fold_apply_iterates!(ir)
        changed += U.inline_calls2!(ir, st)
        changed += U.union_split_calls!(ir, st)
        # concretely-pinned residual calls become invokes each round (stock's
        # inliner shape for declined candidates); invoke-form candidates stay
        # inlineable by the next round's inline_calls2!
        changed += devirtualize_calls!(ir, st)
        ir = U.compact_carry_names!(ir)
        UnifiedIR.verify_ir(ir; level = 1)
        changed == 0 && break
    end
    # final sweep with settled types: whatever still resolves (incl.
    # under-constrained specializations) gets its interprocedural hook
    U.infer_ir!(ir, args; state = st)
    UnifiedIR.editable(ir)
    devirtualize_calls!(ir, st; allow_typevars = true)
    ir = U.compact_carry_names!(ir)
    U.infer_ir!(ir, args; state = st)
    U.refine_effects!(ir)
    return ir
end

# --- interprocedural summaries (the EAUtils cache equivalent) ---------------

const EA_SUMMARIES = IdDict{Core.MethodInstance,Any}()
const EA_ACTIVE = Base.IdSet{Core.MethodInstance}()

function ea_ir_for_mi(mi::Core.MethodInstance)
    m = mi.def
    m isa Method || return nothing
    (m.isva || isdefined(m, :generator)) && return nothing
    sig = mi.specTypes
    sig isa DataType || return nothing
    ps = collect(Any, sig.parameters)
    length(ps) == Int(m.nargs) || return nothing
    any(p -> CC.isvarargtype(p), ps) && return nothing
    ci = Base.uncompressed_ir(m)
    length(ci.code) <= 200 || return nothing
    ir = U.codeinfo_to_ir(ci; nargs = Int(m.nargs), name = m.name)
    ir.meta[:method_instance] = mi
    ir.meta[:slotnames] = ci.slotnames
    ir.sptypes = Any[t for t in mi.sparam_vals]
    ir.meta[:sptypes_lat] = U.sptypes_lattice(mi)
    return ea_pipeline!(ir, ps)
end

"get_escape_cache callback: recursive per-MethodInstance argument-escape
summaries (memoized; conservative `false` on cycles/depth/unconvertibility)."
function ea_summary(@nospecialize codeinst)
    mi = codeinst isa Core.CodeInstance ? codeinst.def :
         codeinst isa Core.MethodInstance ? codeinst : nothing
    mi isa Core.MethodInstance || return false
    haskey(EA_SUMMARIES, mi) && return EA_SUMMARIES[mi]
    (mi in EA_ACTIVE || length(EA_ACTIVE) >= 4) && return false
    push!(EA_ACTIVE, mi)
    r = try
        ir = ea_ir_for_mi(mi)
        if ir === nothing
            false
        else
            nargs = length(UnifiedIR.getregion(ir, UnifiedIR.root_region(ir)).args)
            st = U.analyze_escapes(ir, nargs; get_escape_cache = ea_summary)
            U.UArgEscapeCache(st.state)
        end
    catch
        false
    finally
        delete!(EA_ACTIVE, mi)
    end
    EA_SUMMARIES[mi] = r
    return r
end

function code_escapes(@nospecialize(f), argtypes = ();
                      interprocedural::Bool = true)
    ats = Any[argtypes...]
    ir = U.lowered_ir(f, Tuple{ats...}; world = ea_state().cfg.world)
    args = Any[CC.Const(f)]
    append!(args, ats)
    ir = ea_pipeline!(ir, args)
    nargs = length(UnifiedIR.getregion(ir, UnifiedIR.root_region(ir)).args)
    res = U.analyze_escapes(ir, nargs;
                            get_escape_cache = interprocedural ? ea_summary : U.ea_no_cache)
    return EAResult(ir, res.state)
end

# --- fixture finders --------------------------------------------------------

allstmts(r::EAResult) = StmtId[s for s in UnifiedIR.each_stmt(r.ir)]
kindof(r::EAResult, s::StmtId) = UnifiedIR.stmt_kind(r.ir, s)
stmt_t(r::EAResult, s::StmtId) = CC.widenconst(UnifiedIR.stmt_type(r.ir, s))
callee_of(r::EAResult, s::StmtId) =
    U.static_operand_value(r.ir, UnifiedIR.getop(r.ir, s, 1))

isnew(r::EAResult, s::StmtId) =
    kindof(r, s) === K"new" || kindof(r, s) === K"splatnew"
news(r::EAResult) = filter(s -> isnew(r, s), allstmts(r))
news(r::EAResult, @nospecialize T) =
    filter(s -> isnew(r, s) && stmt_t(r, s) <: T, allstmts(r))
"Reachable `return` statements. A residual cfg island whose every path exits
inside it leaves a structurally-dead outer `return %island::Union{}` — not a
return site, filtered."
function rets(r::EAResult)
    filter(allstmts(r)) do s
        kindof(r, s) === K"return" || return false
        if UnifiedIR.nops(r.ir, s) >= 1
            o = UnifiedIR.getop(r.ir, s, 1)
            if UnifiedIR.optag(o) == UnifiedIR.TAG_STMT &&
               CC.widenconst(UnifiedIR.stmt_type(r.ir, UnifiedIR.asstmt(o))) === Union{}
                return false
            end
        end
        return true
    end
end
iscall_of(r::EAResult, s::StmtId, @nospecialize f) =
    kindof(r, s) === K"call" && callee_of(r, s) === f
calls_of(r::EAResult, @nospecialize f) = filter(s -> iscall_of(r, s, f), allstmts(r))
tuplecalls(r::EAResult) = calls_of(r, Core.tuple)
function isinvoke_of(r::EAResult, s::StmtId, name::Symbol)
    kindof(r, s) === K"invoke" || return false
    ci = U.static_operand_value(r.ir, UnifiedIR.getop(r.ir, s, 1))
    mi = ci isa Core.CodeInstance ? ci.def : ci
    mi isa Core.MethodInstance || return false
    m = mi.def
    return m isa Method && m.name === name
end
invokes_of(r::EAResult, name::Symbol) = filter(s -> isinvoke_of(r, s, name), allstmts(r))

"φ-equivalents: value-joining region ops and selects."
phis(r::EAResult) = filter(allstmts(r)) do s
    k = kindof(r, s)
    (k === K"if" || k === K"select") && stmt_t(r, s) !== Union{}
end

"The φ-merge values over the given allocations: `T`-typed non-allocation
statements aliased with one of the allocations (the region-result join, its
`extract` projection, or a `select` — whichever encoding the pipeline chose)."
function phi_merges(r::EAResult, @nospecialize(T))
    ns = news(r)
    filter(allstmts(r)) do s
        isnew(r, s) && return false
        kindof(r, s) === K"region_arg" && return false
        tt = stmt_t(r, s)
        (tt isa Type && tt <: T) || return false
        return any(n -> aliased(r, n, s), ns)
    end
end

"The returned value's statement, or nothing when a constant/global is returned."
function retval(r::EAResult, ret::StmtId)
    UnifiedIR.nops(r.ir, ret) >= 1 || return nothing
    o = UnifiedIR.getop(r.ir, ret, 1)
    UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || return nothing
    return UnifiedIR.asstmt(o)
end

# ---------------------------------------------------------------------------
# Fixture definitions (hoisted: one world for the whole corpus, so the shared
# inference state and summary cache stay warm — a Keno-rule adaptation; the
# analyzed code is the stock corpus's, verbatim)
# ---------------------------------------------------------------------------

mutable struct SafeRef{T}
    x::T
end
Base.getindex(s::SafeRef) = getfield(s, 1)
Base.setindex!(s::SafeRef, x) = setfield!(s, 1, x)

mutable struct SafeRefs{S,T}
    x1::S
    x2::T
end
Base.getindex(s::SafeRefs, idx::Int) = getfield(s, idx)
Base.setindex!(s::SafeRefs, x, idx::Int) = setfield!(s, idx, x)

global GV::Any
const GR = Ref{Any}()

# EAUtils / basics
fix_sin42 = () -> sin(42)
fix_argret0 = a -> (println("prevent ConstABI"); nothing)
fix_argret = a -> (println("prevent ConstABI"); a)
fix_gstore = a -> (global GV = a; nothing)
fix_gload = () -> (global GV; GV)
fix_gstoreload = s -> (global GV; GV = s; GV)
fix_gcpreserve = s -> begin
    m = SafeRef(s)
    GC.@preserve m begin
        println(s)
        return nothing
    end
end
fix_isdef = (a, b) -> begin
    if b
        s = Ref(a)
    end
    return @isdefined(s)
end
fix_phi = (cond, a, b) -> begin
    c = cond ? a : b
    return c
end
fix_pi = a -> begin
    if isa(a, Regex)
        return a
    end
    return nothing
end
fix_phic = (a, b) -> begin
    local x::String
    try
        x = a
    catch err
        x = b
    end
    return x
end
fix_branching = (a, c) -> begin
    if c
        return nothing
    else
        return a
    end
end
fix_loop = n -> begin
    c = SafeRef{Bool}(false)
    while n > 0
        rand(Bool) && return c
    end
    nothing
end
fix_trycatch = a -> begin
    try
        println("prevent ConstABI")
        nothing
    catch err
        return a
    end
end
fix_tryfinally = a -> begin
    try
        println("prevent ConstABI")
        nothing
    finally
        return a
    end
end
fix_foreigncall = x -> ccall(:some_ccall, Any, (Any,), x)

# builtins
fix_throw = a -> throw(a)
fix_getfield_dyn = a -> getfield(a, :may_not_field)
fix_sizeof = a -> sizeof(a)
fix_egal = (cond, s) -> begin
    m = cond ? s : nothing
    c = m === nothing
    return c
end
fix_sizeof_vec = xs -> sizeof(xs)
fix_ifelse = c -> begin
    r = ifelse(c, Ref("yes"), Ref("no"))
    return r
end
fix_ifelse_const = () -> begin
    r = ifelse(true, Ref("yes"), Ref(nothing))
    return r
end
fix_typeassert = x -> begin
    y = x::Base.RefValue{Any}
    return y
end
fix_isdefined_obj = x -> (isdefined(x, :foo) ? x : throw("undefined"))

# flow-sensitivity
fix_flow1 = cond -> begin
    r = Ref("foo")
    if cond
        return cond
    end
    return r
end
fix_flow2 = cond -> begin
    r = Ref("foo")
    cnt = 0
    while rand(Bool)
        cnt += 1
        rand(Bool) && return r
    end
    rand(Bool) && return r
    return cnt
end

# escape through exceptions (stock's M module)
module ExcM
    unsafeget(x) = isassigned(x) ? x[] : throw(x)
    @noinline function escape_rethrow!()
        try
            rethrow()
        catch err
            GR[] = err
        end
    end
    @noinline function escape_current_exceptions!()
        excs = Base.current_exceptions()
        GR[] = excs
    end
    const GR = Ref{Any}()

    fix_exc_ret = () -> begin
        r = Ref{String}()
        local ret
        try
            s = unsafeget(r)
            ret = sizeof(s)
        catch err
            ret = err
        end
        return ret
    end
    fix_exc_global = () -> begin
        r = Ref{String}()
        local ret # prevent DCE
        try
            s = unsafeget(r)
            ret = sizeof(s)
        catch err
            global GV = err
        end
        nothing
    end
    fix_exc_nested_throw = () -> begin
        r = Ref{String}()
        try
            try
                unsafeget(r)
            catch err1
                throw(err1)
            end
        catch err2
            GR[] = err2
        end
    end
    fix_exc_rethrow1 = () -> begin
        r = Ref{String}()
        try
            try
                unsafeget(r)
            catch err1
                rethrow(err1)
            end
        catch err2
            GR[] = err2
        end
    end
    fix_exc_rethrow2 = () -> begin
        try
            r = Ref{String}()
            unsafeget(r)
        catch
            escape_rethrow!()
        end
    end
    fix_exc_rethrow3 = () -> begin
        local t
        try
            r = Ref{String}()
            t = unsafeget(r)
        catch err
            t = typeof(err)
            escape_rethrow!()
        end
        return t
    end
    fix_exc_currexc1 = () -> begin
        try
            r = Ref{String}()
            unsafeget(r)
        catch
            GR[] = Base.current_exceptions()
        end
    end
    fix_exc_currexc2 = () -> begin
        try
            r = Ref{String}()
            unsafeget(r)
        catch
            escape_current_exceptions!()
        end
    end
    fix_exc_contextual = () -> begin
        r1 = Ref{String}()
        r2 = Ref{String}()
        local ret
        try
            s1 = unsafeget(r1)
            ret = sizeof(s1)
        catch err
            global GV = err
        end
        s2 = unsafeget(r2)
        return s2, r2
    end
    fix_exc_caught_local = () -> begin
        r = Ref{String}()
        local ret
        try
            s = unsafeget(r)
            ret = sizeof(s)
        catch
            ret = nothing
        end
        return ret
    end
    fix_exc_sequential = () -> begin
        r1 = Ref{String}()
        r2 = Ref{String}()
        local ret
        try
            s1 = unsafeget(r1)
            ret = sizeof(s1)
        catch err1
            global GV = err1
        end
        try
            s2 = unsafeget(r2)
            ret = sizeof(s2)
        catch err2
            ret = err2
        end
        return ret
    end
    fix_exc_nested_no_prop = () -> begin
        r = Ref{String}()
        local ret
        try
            s = unsafeget(r)
            try
                ret = sizeof(s)
            catch inner
                return inner
            end
        catch outer
            ret = nothing
        end
        return ret
    end
    fix_exc_merge = () -> begin
        r = Ref{String}()
        local ret
        try
            s = unsafeget(r)
            ret = sizeof(s)
        catch err1
            return err1
        end
        try
            s = unsafeget(r)
            ret = sizeof(s)
        catch err2
            return err2
        end
        nothing
    end
    fix_exc_finally = () -> begin
        r = Ref{String}()
        local ret
        try
            s = unsafeget(r)
            ret = sizeof(s)
        finally
            if !@isdefined(ret)
                ret = 42
            end
        end
        return ret
    end
end # module ExcM

# field analysis / alias analysis
fix_fld_gstore1 = a -> (global GV = SafeRef{Any}(a); nothing)
fix_fld_gstore2 = a -> (global GV = (a,); nothing)
fix_fld_gstore3 = a -> begin
    o0 = SafeRef{Any}(a)
    global GV = SafeRef(o0)
    nothing
end
fix_fld_gstore4 = a -> begin
    t0 = (a,)
    global GV = (t0,)
    nothing
end
fix_fld_gsetfield1 = a -> begin
    r = SafeRef{Any}(:init)
    global GV = r
    r[] = a
    nothing
end
fix_fld_gsetfield2 = (a, b) -> begin
    r = SafeRef{Any}(a)
    global GV = r
    r[] = b
    nothing
end
module EATRx1
    import ..SafeRef
    const Rx = SafeRef(Ref(""))
    fix = s -> begin
        Rx[] = s
        Core.sizeof(Rx[])
    end
end
module EATRx2
    import ..SafeRef
    const Rx = SafeRef{Any}(nothing)
    fix = s -> begin
        setfield!(Rx, :x, s)
        Core.sizeof(Rx[])
    end
end
module EATRx3
    import ..SafeRef
    module ___xxx___
        import ...SafeRef
        const Rx = SafeRef("Rx")
    end
    fix = s -> begin
        rx = getfield(___xxx___, :Rx)
        rx[] = s
        nothing
    end
end

fix_fldesc1 = a -> begin
    o = SafeRef(a)
    Core.donotdelete(o)
    return o[]
end
fix_fldesc2 = a -> begin
    t = SafeRef((a,))
    f = t[][1]
    return f
end
fix_fldesc3 = (a, b) -> begin
    obj = SafeRefs(a, b)
    Core.donotdelete(obj)
    fld1 = obj[1]
    fld2 = obj[2]
    return (fld1, fld2)
end
fix_fldset1 = a -> begin
    o = SafeRef(Ref("foo"))
    Core.donotdelete(o)
    o[] = a
    return o[]
end
fix_fldset2 = a -> begin
    obj = SafeRef(Ref("foo"))
    Core.donotdelete(obj)
    return (obj[] = a)
end
fix_nested1 = a -> begin
    o1 = SafeRef(a)
    o2 = SafeRef(o1)
    return o2[]
end
fix_nested2 = a -> begin
    o1 = (a,)
    o2 = (o1,)
    return o2[1]
end
fix_nested3 = a -> begin
    o1  = SafeRef(a)
    o2  = SafeRef(o1)
    o1′ = o2[]
    a′  = o1′[]
    return a′
end
fix_nested4 = () -> begin
    o1 = SafeRef("foo")
    o2 = SafeRef(o1)
    return o2
end
fix_nested5 = () -> begin
    o1   = SafeRef("foo")
    o2′  = SafeRef(nothing)
    o2   = SafeRef{SafeRef}(o2′)
    o2[] = o1
    return o2
end
fix_broadcast = x -> begin
    o = Ref(x)
    Core.donotdelete(o)
    broadcast(identity, o)
end
fix_phinew1 = (cond, x, y) -> begin
    if cond
        ϕ = SafeRef{Any}(x)
    else
        ϕ = SafeRef{Any}(y)
    end
    return ϕ[]
end
fix_phinew2 = (cond, x, y) -> begin
    if cond
        ϕ2 = ϕ1 = SafeRef{Any}(x)
    else
        ϕ2 = ϕ1 = SafeRef{Any}(y)
    end
    return ϕ1[], ϕ2[]
end
fix_phinew3 = (cond, x, y, z) -> begin
    local out
    if cond
        ϕ = SafeRef(x)
        out = ϕ[]
    else
        ϕ = SafeRefs(z, y)
    end
    return @isdefined(out) ? out : throw(ϕ)
end
fix_alias1 = s -> begin
    r = SafeRef(s)
    Core.donotdelete(r)
    return r[]
end
fix_alias2 = s -> begin
    r1 = SafeRef(s)
    r2 = SafeRef(r1)
    Core.donotdelete(r1, r2)
    return r2[]
end
fix_alias3 = s -> begin
    r1 = SafeRef(s)
    r2 = SafeRef(r1)
    Core.donotdelete(r1, r2)
    return r2[][]
end
module EATRx4
    import ..SafeRef
    const Rx = SafeRef("Rx")
    fix = s -> begin
        r = SafeRef(Rx)
        Core.donotdelete(r)
        rx = r[] # rx aliased to Rx
        rx[] = s
        nothing
    end
end
fix_alias_set1 = s -> begin
    r = Ref{String}()
    Core.donotdelete(r)
    r[] = s
    return r[]
end
fix_alias_set2 = s -> begin
    r1 = Ref(s)
    r2 = Ref{Base.RefValue{String}}()
    Core.donotdelete(r1, r2)
    r2[] = r1
    return r2[]
end
fix_alias_set3 = s -> begin
    r1 = Ref{String}()
    r2 = Ref{Base.RefValue{String}}()
    Core.donotdelete(r1, r2)
    r2[] = r1
    r1[] = s
    return r2[][]
end
fix_alias_set4 = s -> begin
    r1 = Ref{String}()
    r2 = Ref{Base.RefValue{String}}()
    r1[] = s
    r2[] = r1
    return r2[][]
end
module EATRx5
    import ..SafeRef
    const Rx = SafeRef("Rx")
    fix = (_rx, s) -> begin
        r = SafeRef(_rx)
        Core.donotdelete(r)
        r[] = Rx
        rx = r[] # rx aliased to Rx
        rx[] = s
        nothing
    end
end
fix_alias_tassert = a -> begin
    r = a::Base.RefValue{String}
    return r
end
fix_alias_gassert = a -> begin
    global GV
    (GV::SafeRef{Any})[] = a
    nothing
end
fix_alias_ifelse = (c, a, b) -> begin
    r = ifelse(c, a, b)
    return r
end
module EATRx6
    import ..SafeRef
    const Lx, Rx = SafeRef("Lx"), SafeRef("Rx")
    fix = (c, a) -> begin
        r = ifelse(c, Lx, Rx)
        r[] = a
        nothing
    end
end
fix_alias_phi1 = (cond, x) -> begin
    if cond
        ϕ2 = ϕ1 = SafeRef(Ref("foo"))
    else
        ϕ2 = ϕ1 = SafeRef(Ref("bar"))
    end
    ϕ2[] = x
    return ϕ1[]
end
fix_alias_phi2 = (cond1, cond2, x) -> begin
    if cond1
        ϕ2 = ϕ1 = SafeRef(Ref("foo"))
    else
        ϕ2 = ϕ1 = SafeRef(Ref("bar"))
    end
    cond2 && (ϕ2[] = x)
    return ϕ1[]
end
fix_alias_pi = x -> begin
    if isa(x, Base.RefValue{String})
        return x
    end
    throw("error!")
end
fix_alias_gpi = x -> begin
    global GV
    l = GV
    if isa(l, SafeRef{String})
        l[] = x
    end
    nothing
end
fix_circular1 = () -> begin
    x = Ref{Any}()
    x[] = x
    return x[]
end
module CircM
    const Rx = Ref{Any}()
    Rx[] = Rx
    fix = () -> begin
        r = Rx[]::Base.RefValue{Any}
        return r[]
    end
end
module GenrM
    @noinline function genr()
        r = Ref{Any}()
        r[] = r
        return r
    end
    fix = () -> begin
        x = genr()
        return x[]
    end
end

@eval fix_dyn_new1 = (T, x) -> begin
    obj = $(Expr(:new, :T, :x))
end
@eval fix_dyn_new2 = (T, x, y, z) -> begin
    obj = $(Expr(:new, :T, :x, :y))
    return getfield(obj, :x)
end
@eval fix_dyn_new3 = (T, x, y, z) -> begin
    obj = $(Expr(:new, :T, :x))
    setfield!(obj, :x, y)
    return getfield(obj, :x)
end
fix_unknown_fld1 = (a, fld) -> begin
    obj = SafeRef(a)
    return getfield(obj, fld)
end
fix_unknown_fld2 = (a, b, fld) -> begin
    obj = SafeRefs(a, b)
    return getfield(obj, fld) # should escape both `a` and `b`
end
fix_unknown_fld3 = (a, b, idx) -> begin
    obj = SafeRefs(a, b)
    return obj[idx] # should escape both `a` and `b`
end
fix_unknown_fld4 = (a, b, fld) -> begin
    obj = SafeRefs(Ref("a"), Ref("b"))
    setfield!(obj, fld, a)
    return obj[2] # should escape `a`
end
fix_unknown_fld5 = (a, fld) -> begin
    obj = SafeRefs(Ref("a"), Ref("b"))
    setfield!(obj, fld, a)
    return obj[1] # this should escape `a`
end
fix_unknown_fld6 = (a, b, idx) -> begin
    obj = SafeRefs(Ref("a"), Ref("b"))
    obj[idx] = a
    return obj[2] # should escape `a`
end
module GetxM
    import ..SafeRef
    @noinline getx(obj) = obj[]
    fix = a -> begin
        obj = SafeRef(a)
        fld = getx(obj)
        return fld
    end
end
fix_interp_alias = s -> begin
    s[] = Ref("bar")
    global GV = s[]
    nothing
end
module SetxyM
    import ..SafeRef
    @noinline setxy!(x, y) = x[] = y
    fix1 = y -> begin
        x = SafeRef("init")
        setxy!(x, y)
        return x
    end
    fix2 = y -> begin
        x1 = SafeRef("init")
        x2 = SafeRef(y)
        Core.donotdelete(x1, x2)
        setxy!(x1, x2[])
        return x1
    end
end
module MySetIdxM
    @noinline mysetindex!(x, a) = x[1] = a
    const Ax = Vector{Any}(undef, 1)
    fix = s -> mysetindex!(Ax, s)
end
fix_flowsens1 = (a, b) -> begin
    r = SafeRef{Any}(a)
    Core.donotdelete(r)
    r[] = b
    return r[]
end
fix_flowsens2 = (a, b) -> begin
    r = SafeRef{Any}(:init)
    Core.donotdelete(r)
    r[] = a
    r[] = b
    return r[]
end
fix_flowsens3 = (a, b, cond) -> begin
    r = SafeRef{Any}(:init)
    Core.donotdelete(r)
    if cond
        r[] = a
        return r[]
    else
        r[] = b
        return nothing
    end
end
fix_conflict1 = (cnd, baz, qux) -> begin
    if cnd
        o = SafeRef(Ref("foo"))
    else
        o = SafeRefs(Ref("bar"), baz)
        r = getfield(o, 2)
    end
    if cnd
        o = o::SafeRef
        setfield!(o, 1, qux)
        r = getfield(o, 1)
    end
    r
end
fix_conflict2 = (cnd, baz, qux) -> begin
    if cnd
        o = SafeRefs(Ref("foo"), Ref("bar"))
        r = setfield!(o, 2, baz)
    else
        o = SafeRef(qux)
    end
    if !cnd
        o = o::SafeRef
        r = getfield(o, 1)
    end
    r
end
fix_fcall_flds = (t, mt, lim, world) -> begin
    ambig = false
    min = Ref{UInt}(typemin(UInt))
    max = Ref{UInt}(typemax(UInt))
    has_ambig = Ref{Int32}(0)
    mt = ccall(:jl_matching_methods, Any,
        (Any, Any, Cint, Cint, UInt, Ptr{UInt}, Ptr{UInt}, Ref{Int32}),
        t, mt, lim, ambig, world, min, max, has_ambig)::Union{Array{Any,1}, Bool}
    return mt, has_ambig[]
end

# MPoint end-to-end (stock lines 1432-1488)
abstract type AbstractPoint{T} end
mutable struct MPoint{T} <: AbstractPoint{T}
    x::T
    y::T
end
add(a::P, b::P) where P<:AbstractPoint = P(a.x + b.x, a.y + b.y)
function compute(T, ax, ay, bx, by)
    a = T(ax, ay)
    b = T(bx, by)
    for i in 0:(100000000-1)
        c = add(a, b) # replaceable
        a = add(c, b) # replaceable
    end
    a.x, a.y
end
function compute!(a, b)
    for i in 0:(100000000-1)
        c = add(a, b)  # replaceable
        a′ = add(c, b) # replaceable
        a.x = a′.x
        a.y = a′.y
    end
end

# special-casing bitstype
fix_bits_g = a -> (global GV = a; nothing)
fix_bits_fld = a -> begin
    o = SafeRef(a)
    Core.donotdelete(o)
    return o[]
end
fix_bits_tuple = (a, b) -> begin
    t = tuple(a, b)
    return t
end

# interprocedural analysis
@noinline broadcast_noescape2(b) = broadcast(identity, b)
fix_bcast1 = () -> broadcast_noescape2(Ref(Ref("Hi")))
fix_bcast2 = x -> begin
    out1 = broadcast_noescape2(Ref(Ref("Hi")))
    out2 = broadcast_noescape2(x)
    return out1, out2
end
@noinline allescape_argument(a) = (global GV = a) # obvious escape
fix_allesc_arg = () -> allescape_argument(Ref("Hi"))
fix_may_exist = a -> may_exist(a)  # undefined function: dynamic, conservative
fix_invokelatest = a -> Base.@invokelatest broadcast_noescape1(a)
@noinline unionsplit_noescape(a)      = string(nothing)
@noinline unionsplit_noescape(a::Int) = a + 10
fix_unionsplit = x -> begin
    s = SafeRef{Union{Int,Nothing}}(x)
    unionsplit_noescape(s[])
    return nothing
end
@noinline unused_argument(a) = (println("prevent inlining"); nothing)
fix_unused1 = () -> begin
    a = Ref("foo") # shouldn't be "return escape"
    b = unused_argument(a)
    nothing
end
fix_unused2 = () -> begin
    a = Ref("foo") # still should be "return escape"
    b = unused_argument(a)
    return a
end
@noinline returnescape_argument(a) = (println("prevent inlining"); a)
fix_retesc_arg = () -> begin
    obj = Ref("foo")           # should be "return escape"
    ret = returnescape_argument(obj)
    return ret                 # alias of `obj`
end
@noinline noreturnescape_argument(a) = (println("prevent inlining"); identity("hi"))
fix_noretesc_arg = () -> begin
    obj = Ref("foo")              # better to not be "return escape"
    ret = noreturnescape_argument(obj)
    return ret                    # must not alias to `obj`
end
function with_self_aliased(from_bb::Int, succs::Vector{Int})
    worklist = Int[from_bb]
    visited = BitSet(from_bb)
    function visit!(bb::Int)
        if bb ∉ visited
            push!(visited, bb)
            push!(worklist, bb)
        end
    end
    while !isempty(worklist)
        foreach(visit!, succs)
    end
    return visited
end
@noinline identity_if_string(x::SafeRef{<:AbstractString}) = (println("preventing inlining"); nothing)
fix_idstr1 = x -> identity_if_string(x)
fix_idstr2 = x -> begin
    try
        identity_if_string(x)
    catch err
        global GV = err
    end
    return nothing
end
@noinline ambig_error_test(a::SafeRef, b) = (println("preventing inlining"); nothing)
@noinline ambig_error_test(a, b::SafeRef) = (println("preventing inlining"); nothing)
@noinline ambig_error_test(a, b) = (println("preventing inlining"); nothing)
fix_ambig1 = (x, y) -> ambig_error_test(x, y)
fix_ambig2 = (x, y) -> begin
    try
        ambig_error_test(x, y)
    catch err
        global GV = err
    end
end
@eval function scope_folding()
    $(Expr(:tryfinally,
        Expr(:block,
            Expr(:tryfinally, :(), :(), 2),
            :(return Core.current_scope())),
    :(), 1))
end
@eval function scope_folding_opt()
    $(Expr(:tryfinally,
        Expr(:block,
            Expr(:tryfinally, :(), :(), :(Base.inferencebarrier(2))),
            :(return Core.current_scope())),
    :(), :(Base.inferencebarrier(1))))
end

# ---------------------------------------------------------------------------
# Differential-guard oracle: stock EscapeAnalysis over the package's own
# stock pipeline. Both sides run WITHOUT an interprocedural cache
# (get_escape_cache ≡ false) so verdicts are comparable; properties are
# evaluated structurally on each side.
# ---------------------------------------------------------------------------

module StockOracle
    import ..CC
    const EA = CC.EscapeAnalysis
    function escapes(@nospecialize(f), argtypes)
        world = Base.get_world_counter()
        interp = CC.NativeInterpreter(world)
        tt = Base.signature_type(f, Tuple{argtypes...})
        match = Base._which(tt; world)
        mi = CC.specialize_method(match)
        ir = CC.typeinf_ircode(interp, mi, nothing)[1]
        nargs = Int(match.method.nargs)
        estate = EA.analyze_escapes(ir, nargs, CC.optimizer_lattice(interp),
                                    Returns(false))
        return (ir, estate, nargs)
    end
    isnew(@nospecialize x) = Base.Meta.isexpr(x, :new) || Base.Meta.isexpr(x, :splatnew)
    function istuple(ir, @nospecialize x)
        Base.Meta.isexpr(x, :call) || return false
        f = CC.singleton_type(CC.argextype(x.args[1], ir))
        return f === Core.tuple
    end
    function prop(res, kind::Symbol, which::Symbol, sel)
        (ir, estate, nargs) = res
        xs = Core.SSAValue[]
        if which === :arg
            x = EA.EscapeInfo[estate[Core.Argument(sel::Int)]]
        else
            for i in 1:length(ir.stmts)
                stmt = ir[Core.SSAValue(i)][:stmt]
                t = CC.widenconst(ir[Core.SSAValue(i)][:type])
                if which === :new
                    isnew(stmt) && t <: sel && push!(xs, Core.SSAValue(i))
                elseif which === :tuple
                    istuple(ir, stmt) && t <: sel && push!(xs, Core.SSAValue(i))
                end
            end
            x = EA.EscapeInfo[estate[s] for s in xs]
        end
        isempty(x) && return missing
        if kind === :all_escape
            return Base.any(EA.has_all_escape, x)
        elseif kind === :return_escape
            return Base.any(EA.has_return_escape, x)
        elseif kind === :thrown_escape
            return Base.any(EA.has_thrown_escape, x)
        elseif kind === :no_escape
            return Base.all(xi -> EA.has_no_escape(EA.ignore_argescape(xi)), x)
        elseif kind === :load_forwardable
            return Base.all(xi -> xi.AliasInfo isa EA.IndexableFields, x)
        end
        error("unknown prop kind")
    end
end # module StockOracle

function unified_prop(res::EAResult, kind::Symbol, which::Symbol, sel)
    if which === :arg
        xs = U.UEscapeInfo[arg(res, sel::Int)]
    else
        stmts = which === :new ? news(res, sel) :
                filter(s -> stmt_t(res, s) <: sel, tuplecalls(res))
        xs = U.UEscapeInfo[res[s] for s in stmts]
    end
    isempty(xs) && return missing
    if kind === :all_escape
        return any(has_all_escape, xs)
    elseif kind === :return_escape
        return any(has_return_escape, xs)
    elseif kind === :thrown_escape
        return any(has_thrown_escape, xs)
    elseif kind === :no_escape
        return all(x -> has_no_escape(ignore_argescape(x)), xs)
    elseif kind === :load_forwardable
        return all(is_load_forwardable, xs)
    end
    error("unknown prop kind")
end

# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

@testset "unified EA" begin

@testset "UEA: harness" begin
    # stock asserts the ConstABI failure mode of its harness; this harness
    # builds IR unconditionally — assert it analyzes a fully-folded body
    @test code_escapes(fix_sin42) isa EAResult
    @test code_escapes(sin, (Int,)) isa EAResult
    @test code_escapes(sin, (Int,)) isa EAResult
end

@testset "UEA: basics" begin
    let # arg return
        result = code_escapes(fix_argret0, (Any,))
        @test has_arg_escape(arg(result, 2))
        # return
        result = code_escapes(fix_argret, (Any,))
        i = only(rets(result))
        @test has_arg_escape(arg(result, 1)) # self
        @test !has_return_escape(arg(result, 1), i) # self
        @test has_arg_escape(arg(result, 2)) # a
        @test has_return_escape(arg(result, 2), i) # a
    end
    let # global store
        result = code_escapes(fix_gstore, (Any,))
        @test has_all_escape(arg(result, 2))
    end
    let # global load
        result = code_escapes(fix_gload)
        is = filter(s -> has_return_escape(result[s]), allstmts(result))
        @test !isempty(is) && all(s -> has_all_escape(result[s]), is)
    end
    let # global store / load
        result = code_escapes(fix_gstoreload, (Any,))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
    end
    let # :gc_preserve_begin / :gc_preserve_end
        result = code_escapes(fix_gcpreserve, (String,))
        i = only(news(result, SafeRef{String}))
        @test has_no_escape(result[i])
    end
    let # :isdefined
        result = code_escapes(fix_isdef, (String, Bool,))
        is = news(result, Base.RefValue{String})
        @test isempty(is) || has_no_escape(result[only(is)])
    end
    let # ϕ-node (region-result join)
        result = code_escapes(fix_phi, (Bool, Any, Any))
        @assert !isempty(phis(result))
        i = only(rets(result))
        @test has_return_escape(arg(result, 3), i) # a
        @test has_return_escape(arg(result, 4), i) # b
    end
    let # π-node (refine)
        result = code_escapes(fix_pi, (Any,))
        @test any(rets(result)) do i
            has_return_escape(arg(result, 2), i)
        end
    end
    let # φᶜ-node / ϒ-node (handler-crossing state; the unified form keeps
        # per-arm returns, so the sites are queried unqualified)
        result = code_escapes(fix_phic, (Any, String))
        @test has_return_escape(arg(result, 2))
        @test has_return_escape(arg(result, 3))
    end
    let # branching
        result = code_escapes(fix_branching, (Any, Bool,))
        @test has_return_escape(arg(result, 2))
    end
    let # loop
        result = code_escapes(fix_loop, (Int,))
        i = only(news(result, SafeRef{Bool}))
        @test has_return_escape(result[i])
    end
    let # try/catch
        result = code_escapes(fix_trycatch, (Any,))
        @test has_return_escape(arg(result, 2))
    end
    let # try/finally
        result = code_escapes(fix_tryfinally, (Any,))
        @test has_return_escape(arg(result, 2))
    end
    let # :foreigncall
        result = code_escapes(fix_foreigncall, (Any,))
        @test has_all_escape(arg(result, 2))
    end
end

@testset "UEA: builtins" begin
    let # throw
        r = code_escapes(fix_throw, (Any,))
        @test has_thrown_escape(arg(r, 2))
    end
    let # implicit throws
        r = code_escapes(fix_getfield_dyn, (Any,))
        @test has_thrown_escape(arg(r, 2))
        r = code_escapes(fix_sizeof, (Any,))
        @test has_thrown_escape(arg(r, 2))
    end
    let # :===
        result = code_escapes(fix_egal, (Bool, SafeRef{String}))
        @test has_no_escape(ignore_argescape(arg(result, 2)))
    end
    let # sizeof
        result = code_escapes(fix_sizeof_vec, (Vector{Any},))
        @test has_no_escape(ignore_argescape(arg(result, 2)))
    end
    let # ifelse
        result = code_escapes(fix_ifelse, (Bool,))
        inds = news(result)
        @assert !isempty(inds)
        for i in inds
            @test has_return_escape(result[i])
        end
    end
    let # ifelse (with constant condition)
        result = code_escapes(fix_ifelse_const)
        for i in news(result)
            if stmt_t(result, i) == Base.RefValue{String}
                @test has_return_escape(result[i])
            elseif stmt_t(result, i) == Base.RefValue{Nothing}
                @test has_no_escape(result[i])
            end
        end
    end
    let # typeassert
        result = code_escapes(fix_typeassert, (Any,))
        @test has_return_escape(arg(result, 2))
        @test !has_all_escape(arg(result, 2))
    end
    let # isdefined
        result = code_escapes(fix_isdefined_obj, (Any,))
        @test has_return_escape(arg(result, 2))
        @test !has_all_escape(arg(result, 2))
    end
end

@testset "UEA: flow-sensitivity" begin
    # Liveness sites distinguish return statements. The structurizer may
    # merge early returns into one if-result-fed return, collapsing the
    # site count — in that shape the flow distinction is structural (the
    # non-carrying arm's result never aliases the allocation) and the
    # site-count property degenerates to the escaping-return fact.
    let result = code_escapes(fix_flow1, (Bool,))
        i = only(news(result))
        rts = rets(result)
        if length(rts) == 2
            @test count(rt -> has_return_escape(result[i], rt), rts) == 1
        else
            @test length(rts) == 1 && has_return_escape(result[i])
        end
    end
    let result = code_escapes(fix_flow2, (Bool,))
        i = only(news(result, Base.RefValue{String}))
        rts = rets(result)
        n_escaping = count(rt -> has_return_escape(result[i], rt), rts)
        if length(rts) == 3
            @test n_escaping == 2
        else
            @test n_escaping >= 1 && n_escaping < 3
        end
    end
end

@testset "UEA: escape through exceptions" begin
    let # simple: return escape
        result = code_escapes(ExcM.fix_exc_ret)
        i = only(news(result))
        @test has_return_escape(result[i])
    end
    let # simple: global escape
        result = code_escapes(ExcM.fix_exc_global)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let # possible escapes via nested throws
        result = code_escapes(ExcM.fix_exc_nested_throw)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let # possible escapes via `rethrow`
        result = code_escapes(ExcM.fix_exc_rethrow1)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let
        result = code_escapes(ExcM.fix_exc_rethrow2)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let
        result = code_escapes(ExcM.fix_exc_rethrow3)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let # possible escapes via `Base.current_exceptions`
        result = code_escapes(ExcM.fix_exc_currexc1)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let
        result = code_escapes(ExcM.fix_exc_currexc2)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let # contextual: escape of `err` propagates to `r1` only
        result = code_escapes(ExcM.fix_exc_contextual)
        is = news(result)
        @test length(is) == 2
        i1, i2 = is
        r = only(rets(result))
        @test has_all_escape(result[i1])
        @test !has_all_escape(result[i2])
        @test has_return_escape(result[i2], r)
    end
    # the blanket `escape_exception!` imprecision, pinned exactly as stock pins it
    let # exception caught within the frame: ideally wouldn't escape to caller
        result = code_escapes(ExcM.fix_exc_caught_local)
        i = only(news(result))
        r = only(rets(result))
        @test_broken !has_return_escape(result[i], r)
    end
    let # sequential handlers should propagate separately
        result = code_escapes(ExcM.fix_exc_sequential)
        is = news(result)
        @test length(is) == 2
        i1, i2 = is
        r = only(rets(result))
        @test has_all_escape(result[i1])
        @test has_return_escape(result[i2], r)
        @test_broken !has_all_escape(result[i2])
    end
    let # nested: inner handler escape shouldn't reach `s`
        result = code_escapes(ExcM.fix_exc_nested_no_prop)
        i = only(news(result))
        @test_broken !has_return_escape(result[i])
    end
    let # merge: `err1`/`err2` escapes merged
        result = code_escapes(ExcM.fix_exc_merge)
        i = only(news(result))
        rs = rets(result)
        @test_broken !has_all_escape(result[i])
        for r in rs
            @test has_return_escape(result[i], r)
        end
    end
    let # no exception handling: keep propagating the escape
        result = code_escapes(ExcM.fix_exc_finally)
        i = only(news(result))
        @test_broken !has_return_escape(result[i])
    end
end

@testset "UEA: field analysis / alias analysis" begin
    # escaped allocations
    # -------------------
    let # escaped object escapes its fields
        result = code_escapes(fix_fld_gstore1, (Any,))
        i = only(news(result))
        @test has_all_escape(result[i])
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_fld_gstore2, (Any,))
        i = only(tuplecalls(result))
        @test has_all_escape(result[i])
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_fld_gstore3, (Any,))
        is = news(result)
        @test length(is) == 2
        i0, i1 = is
        @test has_all_escape(result[i0])
        @test has_all_escape(result[i1])
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_fld_gstore4, (Any,))
        inds = tuplecalls(result)
        @assert length(inds) == 2
        for i in inds
            @test has_all_escape(result[i])
        end
        @test has_all_escape(arg(result, 2))
    end
    # global escape through `setfield!`
    let result = code_escapes(fix_fld_gsetfield1, (Any,))
        i = only(news(result))
        @test has_all_escape(result[i])
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_fld_gsetfield2, (Any, Any))
        i = only(news(result))
        @test has_all_escape(result[i])
        @test has_all_escape(arg(result, 2)) # a
        @test has_all_escape(arg(result, 3)) # b
    end
    let result = code_escapes(EATRx1.fix, (Base.RefValue{String},))
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(EATRx2.fix, (Base.RefValue{String},))
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(EATRx3.fix, (String,))
        @test has_all_escape(arg(result, 2))
    end

    # field escape
    # ------------
    let # field escape propagates to :new arguments
        result = code_escapes(fix_fldesc1, (Base.RefValue{String},))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        @test is_load_forwardable(result[i])
    end
    let result = code_escapes(fix_fldesc2, (Base.RefValue{String},))
        i = only(tuplecalls(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        @test is_load_forwardable(result[i])
    end
    let result = code_escapes(fix_fldesc3, (Base.RefValue{String}, Base.RefValue{String}))
        i = only(news(result, SafeRefs))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test has_return_escape(arg(result, 3), r) # b
        @test is_load_forwardable(result[i])
    end
    let # field escape propagates to `setfield!` argument
        result = code_escapes(fix_fldset1, (Base.RefValue{String},))
        i = last(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        @test is_load_forwardable(result[i])
    end
    let # escape via the setfield! return value
        result = code_escapes(fix_fldset2, (Base.RefValue{String},))
        i = last(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        @test is_load_forwardable(result[i])
    end

    # nested allocations
    let result = code_escapes(fix_nested1, (Base.RefValue{String},))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        for i in news(result)
            t = stmt_t(result, i)
            if t == SafeRef{Base.RefValue{String}}
                @test has_return_escape(result[i], r)
            elseif t == SafeRef{SafeRef{Base.RefValue{String}}}
                @test is_load_forwardable(result[i])
            end
        end
    end
    let result = code_escapes(fix_nested2, (Base.RefValue{String},))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        for i in tuplecalls(result)
            t = stmt_t(result, i)
            if t == Tuple{Base.RefValue{String}}
                @test has_return_escape(result[i], r)
            elseif t == Tuple{Tuple{Base.RefValue{String}}}
                @test is_load_forwardable(result[i])
            end
        end
    end
    let result = code_escapes(fix_nested3, (Base.RefValue{String},))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        for i in news(result)
            @test is_load_forwardable(result[i])
        end
    end
    let result = code_escapes(fix_nested4)
        r = only(rets(result))
        for i in news(result)
            @test has_return_escape(result[i], r)
        end
    end
    let result = code_escapes(fix_nested5)
        r = only(rets(result))
        for i in news(result)
            t = stmt_t(result, i)
            if t === SafeRef{String} || t === SafeRef{SafeRef}
                @test has_return_escape(result[i], r)
            end
        end
    end
    let result = code_escapes(fix_broadcast, (Base.RefValue{String},))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        # blocked on inlining parity: inline2 declines isva callees, so
        # `broadcast` stays an opaque invoke and the summary is conservative
        # (stock fully inlines it down to the getfield)
        @test_broken is_load_forwardable(result[i])
    end

    # ϕ-node allocations (the merge φ = the SafeRef-typed region-result join)
    let result = code_escapes(fix_phinew1, (Bool, Any, Any))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r) # x
        @test has_return_escape(arg(result, 4), r) # y
        ϕs = phi_merges(result, SafeRef)
        @test !isempty(ϕs)
        for i in ϕs
            @test is_load_forwardable(result[i])
        end
        for i in news(result)
            @test is_load_forwardable(result[i])
        end
    end
    let result = code_escapes(fix_phinew2, (Bool, Any, Any))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r) # x
        @test has_return_escape(arg(result, 4), r) # y
        for i in phi_merges(result, SafeRef)
            @test is_load_forwardable(result[i])
        end
        for i in news(result)
            @test is_load_forwardable(result[i])
        end
    end
    let # when the ϕ merges values of different types
        result = code_escapes(fix_phinew3,
                              (Bool, Base.RefValue{String}, Base.RefValue{String}, Base.RefValue{String}))
        ϕU = Union{SafeRef{Base.RefValue{String}},
                   SafeRefs{Base.RefValue{String},Base.RefValue{String}}}
        # the throw of the merged object (`@isdefined` lowering adds its own
        # UndefVarError throws — select by thrown-operand type)
        t = only(filter(calls_of(result, Core.throw)) do s
            CC.widenconst(U.stmt_lattice(result.ir, UnifiedIR.getop(result.ir, s, 2))) == ϕU
        end)
        ϕs = filter(allstmts(result)) do s
            stmt_t(result, s) == ϕU
        end
        @test has_return_escape(arg(result, 3)) # x
        @test !has_return_escape(arg(result, 4)) # y
        @test has_return_escape(arg(result, 5)) # z
        @test !isempty(ϕs) && any(ϕ -> has_thrown_escape(result[ϕ], t), ϕs)
    end

    # alias analysis
    # --------------
    let # alias via getfield & Expr(:new)
        result = code_escapes(fix_alias1, (String,))
        i = only(news(result))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test aliased(result, 2, val)
        @test !aliased(result, 2, i)
    end
    let result = code_escapes(fix_alias2, (String,))
        i1, i2 = news(result)
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test !aliased(result, i1, i2)
        @test aliased(result, i1, val)
        @test !aliased(result, i2, val)
    end
    let result = code_escapes(fix_alias3, (String,))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test aliased(result, 2, val)
        for i in news(result)
            @test !aliased(result, i, val)
        end
    end
    let result = code_escapes(EATRx4.fix, (String,))
        i = only(news(result, SafeRef{SafeRef{String}}))
        @test has_all_escape(arg(result, 2))
        @test is_load_forwardable(result[i])
    end
    let # alias via getfield & setfield!
        result = code_escapes(fix_alias_set1, (String,))
        i = only(news(result))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test aliased(result, 2, val)
        @test !aliased(result, 2, i)
    end
    let result = code_escapes(fix_alias_set2, (String,))
        i1, i2 = news(result)
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test !aliased(result, i1, i2)
        @test aliased(result, i1, val)
        @test !aliased(result, i2, val)
    end
    let result = code_escapes(fix_alias_set3, (String,))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test aliased(result, 2, val)
        for i in news(result)
            @test !aliased(result, i, val)
        end
        result = code_escapes(fix_alias_set4, (String,))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test aliased(result, 2, val)
        for i in news(result)
            @test !aliased(result, i, val)
        end
    end
    let result = code_escapes(EATRx5.fix, (SafeRef{String}, String,))
        i = first(news(result, SafeRef{SafeRef{String}}))
        @test has_all_escape(arg(result, 3))
        @test is_load_forwardable(result[i])
    end
    let # alias via typeassert
        result = code_escapes(fix_alias_tassert, (Any,))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test has_return_escape(arg(result, 2), r) # a
        @test aliased(result, 2, val)              # a <-> r
    end
    let result = code_escapes(fix_alias_gassert, (Any,))
        @test has_all_escape(arg(result, 2))
    end
    let # alias via ifelse
        result = code_escapes(fix_alias_ifelse, (Bool, Any, Any))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test has_return_escape(arg(result, 3), r) # a
        @test has_return_escape(arg(result, 4), r) # b
        @test !aliased(result, 2, val)             # c <!-> r
        @test aliased(result, 3, val)              # a <-> r
        @test aliased(result, 4, val)              # b <-> r
    end
    let result = code_escapes(EATRx6.fix, (Bool, String,))
        @test has_all_escape(arg(result, 3)) # a
    end
    let # alias via ϕ-node
        result = code_escapes(fix_alias_phi1, (Bool, Base.RefValue{String}))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test has_return_escape(arg(result, 3), r) # x
        @test aliased(result, 3, val) # x
        for i in phi_merges(result, SafeRef)
            @test is_load_forwardable(result[i])
        end
        for i in news(result)
            if stmt_t(result, i) <: SafeRef
                @test is_load_forwardable(result[i])
            end
        end
    end
    let result = code_escapes(fix_alias_phi2, (Bool, Bool, Base.RefValue{String}))
        r = only(rets(result))
        val = retval(result, r)::StmtId
        @test has_return_escape(arg(result, 4), r) # x
        @test aliased(result, 4, val) # x
        for i in phi_merges(result, SafeRef)
            @test is_load_forwardable(result[i])
        end
        for i in news(result)
            if stmt_t(result, i) <: SafeRef
                @test is_load_forwardable(result[i])
            end
        end
    end
    let # alias via π-node
        result = code_escapes(fix_alias_pi, (Any,))
        r = only(rets(result))
        rval = retval(result, r)::StmtId
        @test has_return_escape(arg(result, 2), r) # x
        @test aliased(result, 2, rval)
    end
    let result = code_escapes(fix_alias_gpi, (String,))
        @test has_all_escape(arg(result, 2)) # x
    end
    # circular reference
    let result = code_escapes(fix_circular1)
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(result[i], r)
    end
    let result = code_escapes(CircM.fix)
        r = only(rets(result))
        loads = filter(allstmts(result)) do s
            kindof(result, s) === K"extract" || iscall_of(result, s, Core.getfield)
        end
        @test !isempty(loads)
        for i in loads
            @test has_return_escape(result[i], r)
        end
    end
    let result = code_escapes(GenrM.fix)
        i = only(invokes_of(result, :genr))
        r = only(rets(result))
        @test has_return_escape(result[i], r)
    end

    # dynamic semantics
    # -----------------
    let # conservatively handle untyped objects
        result = code_escapes(fix_dyn_new1, (Any, Any,))
        t = only(news(result))
        @test has_thrown_escape(arg(result, 2), t) # T
        @test has_thrown_escape(arg(result, 3), t) # x
    end
    let result = code_escapes(fix_dyn_new2, (Any, Any, Any, Any))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r)  # x
        @test has_return_escape(arg(result, 4), r)  # y
        @test !has_return_escape(arg(result, 5), r) # z
    end
    let result = code_escapes(fix_dyn_new3, (Any, Any, Any, Any))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r)  # x
        @test has_return_escape(arg(result, 4), r)  # y
        @test !has_return_escape(arg(result, 5), r) # z
    end

    # conservatively handle unknown fields (fields escape; the allocation
    # itself needn't)
    let result = code_escapes(fix_unknown_fld1, (Base.RefValue{String}, Symbol))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test !is_load_forwardable(result[i]) # obj
    end
    let result = code_escapes(fix_unknown_fld2,
                              (Base.RefValue{String}, Base.RefValue{String}, Symbol))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test has_return_escape(arg(result, 3), r) # b
        @test !is_load_forwardable(result[i]) # obj
    end
    let result = code_escapes(fix_unknown_fld3,
                              (Base.RefValue{String}, Base.RefValue{String}, Int))
        i = only(news(result, SafeRefs))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test has_return_escape(arg(result, 3), r) # b
        @test !is_load_forwardable(result[i]) # obj
    end
    let result = code_escapes(fix_unknown_fld4,
                              (Base.RefValue{String}, Base.RefValue{String}, Symbol))
        i = last(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test !has_return_escape(arg(result, 3), r) # b
        @test !is_load_forwardable(result[i]) # obj
    end
    let result = code_escapes(fix_unknown_fld5, (Base.RefValue{String}, Symbol))
        i = last(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test !is_load_forwardable(result[i]) # obj
    end
    let result = code_escapes(fix_unknown_fld6,
                              (Base.RefValue{String}, Base.RefValue{String}, Int))
        i = last(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r) # a
        @test !has_return_escape(arg(result, 3), r) # b
        @test !is_load_forwardable(result[i]) # obj
    end

    # interprocedural
    # ---------------
    let result = code_escapes(GetxM.fix, (Base.RefValue{String},))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(arg(result, 2), r)
        # can't scalar-replace obj (passed to a callee), but maybe stack-allocate
        @test_broken is_load_forwardable(result[i])
    end
    let # TODO interprocedural alias analysis
        result = code_escapes(fix_interp_alias, (SafeRef{Base.RefValue{String}},))
        @test_broken !has_all_escape(arg(result, 2))
    end
    # aliasing between arguments
    let result = code_escapes(SetxyM.fix1, (String,))
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(result[i], r)
        @test has_return_escape(arg(result, 2), r) # y
    end
    let result = code_escapes(SetxyM.fix2, (String,))
        i1, i2 = news(result)
        r = only(rets(result))
        @test has_return_escape(result[i1], r)
        @test !has_return_escape(result[i2], r)
        @test has_return_escape(arg(result, 2), r) # y
    end
    let result = code_escapes(MySetIdxM.fix, (Base.RefValue{String},))
        @test has_all_escape(arg(result, 2)) # s
    end

    # TODO flow-sensitivity?
    # ----------------------
    let result = code_escapes(fix_flowsens1, (Any, Any))
        i = only(news(result))
        r = only(rets(result))
        @test_broken !has_return_escape(arg(result, 2), r) # a
        @test has_return_escape(arg(result, 3), r) # b
        @test is_load_forwardable(result[i])
    end
    let result = code_escapes(fix_flowsens2, (Any, Any))
        i = only(news(result))
        r = only(rets(result))
        @test_broken !has_return_escape(arg(result, 2), r) # a
        @test has_return_escape(arg(result, 3), r) # b
        @test is_load_forwardable(result[i])
    end
    let result = code_escapes(fix_flowsens3, (Any, Any, Bool))
        i = only(news(result))
        @test is_load_forwardable(result[i])
        r = only(filter(rt -> retval(result, rt) !== nothing, rets(result)))
        @test has_return_escape(arg(result, 2), r) # a
        @test_broken !has_return_escape(arg(result, 3), r) # b
    end

    # conflicting field information
    let result = code_escapes(fix_conflict1,
                              (Bool, Base.RefValue{String}, Base.RefValue{String},))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r) # baz
        @test has_return_escape(arg(result, 4), r) # qux
        for i in news(result)
            if !(stmt_t(result, i) <: Base.RefValue)
                @test is_load_forwardable(result[i])
            end
        end
    end
    let result = code_escapes(fix_conflict2,
                              (Bool, Base.RefValue{String}, Base.RefValue{String},))
        r = only(rets(result))
        @test has_return_escape(arg(result, 3), r) # baz
        @test has_return_escape(arg(result, 4), r) # qux
    end

    # foreigncall disables field analysis
    let result = code_escapes(fix_fcall_flds, (Any, Nothing, Int, UInt))
        for i in news(result)
            @test !is_load_forwardable(result[i])
        end
    end
end

@testset "UEA: MPoint end-to-end" begin
    let result = code_escapes(compute,
                              (Type{MPoint}, ComplexF64, ComplexF64, ComplexF64, ComplexF64))
        subjects = filter(allstmts(result)) do s
            (isnew(result, s) || kindof(result, s) === K"if" ||
             kindof(result, s) === K"region_arg") &&
                stmt_t(result, s) <: MPoint
        end
        @test !isempty(subjects)
        for i in subjects
            @test is_load_forwardable(result[i])
        end
    end
    let result = code_escapes(compute!,
                              (MPoint{ComplexF64}, MPoint{ComplexF64}))
        for i in news(result)
            stmt_t(result, i) <: MPoint || continue
            @test is_load_forwardable(result[i])
        end
    end
end

@testset "UEA: special-casing bitstype" begin
    let result = code_escapes(fix_bits_g, (Nothing,))
        @test !has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_bits_fld, (Int,))
        i = only(news(result))
        r = only(rets(result))
        @test !has_return_escape(result[i], r)
    end
    let # an escaped tuple doesn't escape its bitstype argument
        result = code_escapes(fix_bits_tuple, (Int, Any,))
        i = only(tuplecalls(result))
        r = only(rets(result))
        @test !has_return_escape(arg(result, 2), r)
        @test has_return_escape(arg(result, 3), r)
    end
end

@testset "UEA: interprocedural analysis" begin
    let result = code_escapes(fix_bcast1)
        i = last(news(result))
        @test_broken !has_return_escape(result[i]) # TODO interprocedural alias analysis
        @test_broken !has_thrown_escape(result[i])
    end
    let result = code_escapes(fix_bcast2, (Base.RefValue{Base.RefValue{String}},))
        i = last(news(result))
        @test_broken !has_return_escape(result[i]) # TODO interprocedural alias analysis
        @test_broken !has_thrown_escape(result[i])
        @test has_thrown_escape(arg(result, 2))
    end
    let result = code_escapes(fix_allesc_arg)
        i = only(news(result))
        @test has_all_escape(result[i])
    end
    let # statically unresolvable: conservative
        result = code_escapes(fix_may_exist, (Ref{Any},))
        @test has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_invokelatest, (Ref{Any},))
        @test has_all_escape(arg(result, 2))
    end
    let # simple union-split
        result = code_escapes(fix_unionsplit, (Union{Int,Nothing},))
        inds = news(result)
        @assert !isempty(inds)
        for i in inds
            @test has_no_escape(result[i])
        end
    end
    let result = code_escapes(fix_unused1)
        i = only(news(result))
        @test has_no_escape(result[i])
        result = code_escapes(fix_unused2)
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(result[i], r)
    end
    let # escape imposed on the return value propagates to the aliased argument
        result = code_escapes(fix_retesc_arg)
        i = only(news(result))
        r = only(rets(result))
        @test has_return_escape(result[i], r)
    end
    let result = code_escapes(fix_noretesc_arg)
        i = only(news(result))
        @test has_no_escape(result[i])
    end
    @test code_escapes(with_self_aliased, (Int, Vector{Int})) isa EAResult
    # ThrownEscape via potential MethodError
    let # no method error
        result = code_escapes(fix_idstr1, (SafeRef{String},))
        @test has_no_escape(ignore_argescape(arg(result, 2)))
    end
    let result = code_escapes(fix_idstr1, (SafeRef,))
        sites = vcat(calls_of(result, identity_if_string),
                     invokes_of(result, :identity_if_string))
        @test !isempty(sites)
        # x may be thrown (via the potential MethodError); the site is the
        # call itself (dynamic form) or the split-off method-error branch
        @test has_thrown_escape(arg(result, 2))
        if any(s -> kindof(result, s) === K"call", sites)
            # unresolved dynamic call: stock's conservative verdict
            @test_broken !has_return_escape(arg(result, 2))
        else
            # match-based split devirtualization + callee summary:
            # legitimately more precise than stock's dynamic-call handling
            @test !has_return_escape(arg(result, 2))
        end
    end
    let result = code_escapes(fix_idstr2, (SafeRef{String},))
        @test !has_all_escape(arg(result, 2))
    end
    let result = code_escapes(fix_idstr2, (Union{SafeRef{String},Vector{String}},))
        @test has_all_escape(arg(result, 2))
    end
    # method ambiguity error
    let result = code_escapes(fix_ambig1, (SafeRef{String}, Any))
        i = only(vcat(calls_of(result, ambig_error_test),
                      invokes_of(result, :ambig_error_test)))
        r = only(rets(result))
        @test has_thrown_escape(arg(result, 2), i)  # x
        @test has_thrown_escape(arg(result, 3), i)  # y
        @test_broken !has_return_escape(arg(result, 2), r)  # x
        @test_broken !has_return_escape(arg(result, 3), r)  # y
    end
    let result = code_escapes(fix_ambig2, (SafeRef{String}, Any))
        @test has_all_escape(arg(result, 2))  # x
        @test has_all_escape(arg(result, 3))  # y
    end
    # scope folding (analysis completes on tryfinally scope forms)
    @test code_escapes(scope_folding) isa EAResult
    @test code_escapes(scope_folding_opt) isa EAResult
end

@testset "UEA: differential guard vs stock EA" begin
    # (fixture, argtypes, [(kind, which, sel), ...], expected-divergences)
    guard = Any[
        (fix_argret, (Any,), [(:return_escape, :arg, 2)], ()),
        (fix_gstore, (Any,), [(:all_escape, :arg, 2)], ()),
        (fix_gcpreserve, (String,), [(:no_escape, :new, SafeRef{String})], ()),
        (fix_phi, (Bool, Any, Any), [(:return_escape, :arg, 3), (:return_escape, :arg, 4)], ()),
        (fix_branching, (Any, Bool), [(:return_escape, :arg, 2)], ()),
        (fix_loop, (Int,), [(:return_escape, :new, SafeRef{Bool})], ()),
        (fix_trycatch, (Any,), [(:return_escape, :arg, 2)], ()),
        (fix_foreigncall, (Any,), [(:all_escape, :arg, 2)], ()),
        (fix_throw, (Any,), [(:thrown_escape, :arg, 2)], ()),
        (fix_getfield_dyn, (Any,), [(:thrown_escape, :arg, 2)], ()),
        (fix_egal, (Bool, SafeRef{String}), [(:no_escape, :arg, 2)], ()),
        (fix_ifelse, (Bool,), [(:return_escape, :new, Base.RefValue{String})], ()),
        (fix_typeassert, (Any,), [(:return_escape, :arg, 2), (:all_escape, :arg, 2)], ()),
        (fix_isdefined_obj, (Any,), [(:return_escape, :arg, 2), (:all_escape, :arg, 2)], ()),
        (ExcM.fix_exc_global, (), [(:all_escape, :new, Base.RefValue{String})], ()),
        (ExcM.fix_exc_rethrow2, (), [(:all_escape, :new, Base.RefValue{String})], ()),
        (fix_fld_gstore1, (Any,), [(:all_escape, :arg, 2), (:all_escape, :new, SafeRef{Any})], ()),
        (fix_fld_gstore2, (Any,), [(:all_escape, :arg, 2)], ()),
        (fix_fld_gsetfield1, (Any,), [(:all_escape, :arg, 2)], ()),
        (fix_fldesc1, (Base.RefValue{String},),
         [(:return_escape, :arg, 2), (:load_forwardable, :new, SafeRef)], ()),
        (fix_fldesc3, (Base.RefValue{String}, Base.RefValue{String}),
         [(:return_escape, :arg, 2), (:return_escape, :arg, 3),
          (:load_forwardable, :new, SafeRefs)], ()),
        (fix_fldset1, (Base.RefValue{String},), [(:return_escape, :arg, 2)], ()),
        (fix_nested1, (Base.RefValue{String},),
         [(:return_escape, :arg, 2), (:return_escape, :new, SafeRef{Base.RefValue{String}})], ()),
        (fix_nested4, (), [(:return_escape, :new, SafeRef)], ()),
        (fix_phinew1, (Bool, Any, Any),
         [(:return_escape, :arg, 3), (:return_escape, :arg, 4),
          (:load_forwardable, :new, SafeRef{Any})], ()),
        (fix_unknown_fld1, (Base.RefValue{String}, Symbol),
         [(:return_escape, :arg, 2), (:load_forwardable, :new, SafeRef)], ()),
        (fix_unknown_fld2, (Base.RefValue{String}, Base.RefValue{String}, Symbol),
         [(:return_escape, :arg, 2), (:return_escape, :arg, 3)], ()),
        (fix_bits_g, (Nothing,), [(:all_escape, :arg, 2)], ()),
        (fix_bits_tuple, (Int, Any),
         [(:return_escape, :arg, 3), (:return_escape, :tuple, Tuple{Int,Any})], ()),
        (fix_flow1, (Bool,), [(:return_escape, :new, Base.RefValue{String})], ()),
        (fix_circular1, (), [(:return_escape, :new, Base.RefValue{Any})], ()),
        (fix_alias_ifelse, (Bool, Any, Any),
         [(:return_escape, :arg, 3), (:return_escape, :arg, 4)], ()),
    ]
    for (f, ats, props, expected_divergence) in guard
        sres = StockOracle.escapes(f, ats)
        ures = code_escapes(f, ats; interprocedural = false)
        for (kind, which, sel) in props
            sv = StockOracle.prop(sres, kind, which, sel)
            uv = unified_prop(ures, kind, which, sel)
            if (kind, which, sel) in expected_divergence
                @test (sv === missing || uv === missing) || sv !== uv
            else
                # missing on either side = the subject was optimized away
                # there; comparable only when both sides still carry it
                if sv !== missing && uv !== missing
                    @test sv === uv
                end
            end
        end
    end
end

end # @testset "unified EA"

end # module unified_test_EA
