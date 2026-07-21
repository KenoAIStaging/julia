# Transfer functions and interprocedural call inference for the UnifiedIR
# inference port. The frame/state walker lives in uinference.jl.

# ---------------------------------------------------------------------------
# Effects (§8.2 vocabulary; composition per §3.3/§5.1 rule 5)
#
# The currency is the full stock `Compiler.Effects` (all 9 axes, conditional
# bits included): per-statement effects merge into the frame accumulator via
# `CC.merge_effects`, interprocedural results carry them in `UResult`, and
# frame finish resolves the conditional bits (`finish_frame_effects`). The
# UInt32 `FLAG_*` column is a *projection* (`effects_mask`) kept for the IR
# passes (DCE removability, `materialize_consts!`) — stock's IR_FLAG split.
# ---------------------------------------------------------------------------

"""
    UResult

Interprocedural result: rettype lattice element + `Compiler.Effects` +
exception-type bestguess. This is the `st.cache` value.
"""
struct UResult
    rt::Any
    effects::CC.Effects
    exct::Any
    # invariant: nothrow ⟹ exct === Union{}
    UResult(@nospecialize(rt), effects::CC.Effects, @nospecialize(exct)) =
        new(rt, effects, effects.nothrow ? Union{} : exct)
end
UResult(@nospecialize(rt), effects::CC.Effects) =
    UResult(rt, effects, effects.nothrow ? Union{} : Any)

"CC.Effects -> UnifiedIR flag-column projection (the unconditional bits)."
function effects_mask(e::CC.Effects)
    m = UInt32(0)
    CC.is_consistent(e)  && (m |= UnifiedIR.FLAG_CONSISTENT)
    CC.is_effect_free(e) && (m |= UnifiedIR.FLAG_EFFECT_FREE)
    CC.is_nothrow(e)     && (m |= UnifiedIR.FLAG_NOTHROW)
    CC.is_terminates(e)  && (m |= UnifiedIR.FLAG_TERMINATES)
    return m
end

"""An under-initialized immutable `new` (the #52857 class) violates the
type-level "first min_ninitialized fields are defined" invariant that
`isdefined_tfunc`/`getfield_nothrow` trust BEFORE consulting PartialStruct
undef facts, so a load of such a field comes back nothrow and the optimizer
deletes its conditional UndefRefError throw — which stock's sroa preserves
in the conditional block (the "affinity" property). True when `f` is a
getfield of a field the PartialStruct does not prove defined inside the
min_ninitialized prefix (legitimately-constructed values always carry
`false` there, see `partialstruct_init_undefs`) — the load may throw (F11)."""
function getfield_maybe_undef(@nospecialize(f), argl::Vector{Any})
    f === Core.getfield || return false
    length(argl) >= 2 || return false
    obj = argl[1]
    obj isa CC.PartialStruct || return false
    ut = Base.unwrap_unionall(obj.typ)
    ut isa DataType || return false
    fld = argl[2]
    fld isa CC.Const || return false
    v = fld.val
    if v isa Symbol
        idx = Base.fieldindex(ut, v, false)
    elseif v isa Int
        idx = v
    else
        return false
    end
    und = obj.undefs
    (1 <= idx <= length(und) && idx <= CC.datatype_min_ninitialized(ut)) || return false
    return und[idx] !== false
end

"`CC.builtin_effects`/`CC.intrinsic_effects`, full-width, defensively.
`rt` must be the LATTICE element (Const-ness drives e.g. apply_type nothrow)."
function builtin_call_effects(@nospecialize(f), argl::Vector{Any}, @nospecialize(rt))
    f isa Core.Builtin || return CC.Effects()
    if f isa Core.IntrinsicFunction
        return try
            CC.intrinsic_effects(f, argl)
        catch
            CC.Effects()
        end
    end
    eff = try
        CC.builtin_effects(CC.fallback_lattice, f, argl, rt)
    catch
        CC.Effects()
    end
    if eff.nothrow && getfield_maybe_undef(f, argl)
        eff = CC.Effects(eff; nothrow = false)
    end
    return eff
end

"`CC.builtin_exct`/`CC.intrinsic_exct` for a builtin call, defensively."
function builtin_call_exct(@nospecialize(f), argl::Vector{Any}, @nospecialize(rt))
    f isa Core.Builtin || return Any
    if f isa Core.IntrinsicFunction
        return try
            CC.intrinsic_exct(CC.fallback_lattice, f, argl)
        catch
            Any
        end
    end
    exct = try
        CC.builtin_exct(CC.fallback_lattice, f, argl, rt)
    catch
        Any
    end
    if getfield_maybe_undef(f, argl) && exct isa Type && !(UndefRefError <: exct)
        exct = Union{exct, UndefRefError}   # the invariant-blind exct misses it (F11)
    end
    return exct
end

"""The abstract_eval_globalref port for cache-grade inference: resolve the
binding partition at the collector's world, record the binding edge and the
partition chain's world bounds, and return the partition-derived `RTEffects`
(`Const` rt for defined-const partitions, the declared type for typed
globals, `Any` for guards/declared — with stock's partition-load effects and
exception type). Unlike the ambient `isconst`/`getglobal` fold this is sound
under redefinition: the recorded worlds and the binding backedge bound the
answer's validity."""
function global_partition_rte(col::UEdges, mod::Module, name::Symbol)
    key = (mod, name)
    memo = get(col.globmemo, key, nothing)
    if memo !== nothing
        # re-consults must reach the fact trace even though the edge storage
        # and the clamp already happened (frame windows opened since the first
        # read still depend on this fact)
        trace!(col, (0x2, mod, name, memo))
        return memo::CC.RTEffects
    end
    local rte
    try
        b = convert(Core.Binding, GlobalRef(mod, name))
        partition = CC.lookup_binding_partition(col.world, b)
        clamp_world!(col, partition.min_world, partition.max_world)
        valid_worlds, (leaf_b, leaf_partition) = CC.walk_binding_partition(b, partition, col.world)
        clamp_world!(col, valid_worlds)
        record_binding!(col, b)
        rte = CC.abstract_eval_partition_load(nothing, leaf_b, leaf_partition)
    catch
        col.ok = false
        return CC.RTEffects(Any, Any, CC.Effects())
    end
    col.globmemo[key] = rte
    trace!(col, (0x2, mod, name, rte))
    return rte
end

"""Global-read model (rt + effects + exct): partition-based (world-pinned,
edge recorded) with a collector attached; the ambient equivalent of stock's
partition-load shapes otherwise (defined-const → total with mutation-free
imo; anything else → `generic_getglobal_effects`)."""
function global_rte(st::UInferState, mod::Module, name::Symbol)
    col = st.edges
    col === nothing || return global_partition_rte(col, mod, name)
    if isconst(mod, name) && isdefined(mod, name)
        rt = CC.Const(getglobal(mod, name))
        return CC.RTEffects(rt, Union{}, CC.Effects(CC.EFFECTS_TOTAL;
            inaccessiblememonly = CC.is_mutation_free_argtype(rt) ?
                CC.ALWAYS_TRUE : CC.ALWAYS_FALSE))
    end
    bt = try
        Core.get_binding_type(mod, name)
    catch
        Any
    end
    return CC.RTEffects(bt isa Type ? bt : Any, UndefVarError,
                        CC.Effects(CC.generic_getglobal_effects;
                                   effect_free = CC.ALWAYS_TRUE))
end

"const-and-defined test for a global: partition-based (world-pinned, edge
recorded) with a collector attached, ambient otherwise."
function global_is_const_defined(st::UInferState, mod::Module, name::Symbol)
    col = st.edges
    col === nothing || return global_partition_rte(col, mod, name).rt isa CC.Const
    return isconst(mod, name) && isdefined(mod, name)
end

"""The `global_assignment_rt_exct` port (frame-free): the stored-value rt and
exception type of assigning `vl` to the binding, from its partition kind at
the inference world (guard → ErrorException; const/import → error; typed
global → TypeError unless the value type fits). Records the binding edge in
driver mode."""
function global_assign_rt_exct(st::UInferState, M::Module, s::Symbol, @nospecialize(vl))
    col = st.edges
    try
        b = convert(Core.Binding, GlobalRef(M, s))
        partition = CC.lookup_binding_partition(st.cfg.world, b)
        if col !== nothing
            clamp_world!(col, partition.min_world, partition.max_world)
            record_binding!(col, b)
            memo_poison!(col)   # assignment-kind facts are not trace-encodable
        end
        kind = CC.binding_kind(partition)
        if CC.is_some_guard(kind)
            return (vl, ErrorException)
        elseif CC.is_some_const_binding(kind) || CC.is_some_imported(kind)
            # N.B.: backdating should not improve inference in an earlier world
            return (kind == CC.PARTITION_KIND_BACKDATED_CONST ? vl : Union{}, ErrorException)
        end
        ty = kind == CC.PARTITION_KIND_DECLARED ? Any : CC.partition_restriction(partition)
        wnew = CC.widenconst(widenucond(vl))
        if !CC.hasintersect(wnew, ty)
            return (Union{}, TypeError)
        elseif !(wnew <: ty)
            return (CC.tmeet(CC.fallback_lattice, widenucond(vl), ty), TypeError)
        end
        return (vl, Union{})
    catch
        col === nothing || (col.ok = false)
        return (vl, Any)
    end
end

"""The abstract_eval_get_binding_type port (frame-free): fold
`Core.get_binding_type(M, s)` from the leaf partition kind (typed global →
`Const(ty)`; const binding → `Const(Any)`; guard/declared → `Type`), with the
binding edge recorded in driver mode. The fold makes `global x = 1`'s
lowered convert-guard branch dead, like stock."""
function infer_get_binding_type(fr::Frame, args::Vector{Any})::UResult
    st = fr.st
    if length(args) != 3 || CC.isvarargtype(args[end])
        return UResult(Union{}, CC.EFFECTS_THROWS, ArgumentError)
    end
    ml = args[2]; sl = args[3]
    if ml isa CC.Const && sl isa CC.Const
        M = ml.val; s = sl.val
        (M isa Module && s isa Symbol) ||
            return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)
        col = st.edges
        rt = try
            b = convert(Core.Binding, GlobalRef(M, s))
            partition = CC.lookup_binding_partition(st.cfg.world, b)
            if col !== nothing
                clamp_world!(col, partition.min_world, partition.max_world)
            end
            valid_worlds, (leaf_b, leaf_partition) =
                CC.walk_binding_partition(b, partition, st.cfg.world)
            if col !== nothing
                clamp_world!(col, valid_worlds)
                record_binding!(col, b)
                memo_poison!(col)   # binding-type facts are not trace-encodable
            end
            kind = CC.binding_kind(leaf_partition)
            if CC.is_some_guard(kind) || kind == CC.PARTITION_KIND_DECLARED
                Type
            elseif CC.is_some_const_binding(kind)
                CC.Const(Any)
            else
                CC.Const(CC.partition_restriction(leaf_partition))
            end
        catch
            col === nothing || (col.ok = false)
            Type
        end
        return UResult(rt, CC.EFFECTS_TOTAL, Union{})
    end
    wM = CC.widenconst(widenucond(ml)); wS = CC.widenconst(widenucond(sl))
    if !(CC.hasintersect(wM, Module) && CC.hasintersect(wS, Symbol))
        return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)
    end
    nothrow = wM <: Module && wS <: Symbol
    return UResult(Type, CC.Effects(CC.EFFECTS_TOTAL; nothrow),
                   nothrow ? Union{} : TypeError)
end

"""The abstract_call_unionall port: `UnionAll(tv, body)` construction —
Const/Type-precise when the body is pinned, nothrow from the argument
lattices. (`Core.UnionAll` is a type callee, so without this it would
dispatch into the ccall-backed constructor method.)"""
function infer_unionall(fr::Frame, args::Vector{Any})::UResult
    na = length(args)
    lat = CC.fallback_lattice
    local a2, a3, nothrow::Bool
    if na >= 1 && CC.isvarargtype(args[end])
        na <= 2 && return UResult(Any, CC.EFFECTS_THROWS)
        na > 4 && return UResult(Union{}, CC.EFFECTS_THROWS)
        a2 = args[2]
        a3 = CC.unwrapva(args[3])
        nothrow = false
    elseif na == 3
        a2 = args[2]
        a3 = args[3]
        nothrow = CC.:⊑(lat, a2, TypeVar) &&
                  (CC.:⊑(lat, a3, Type) || CC.:⊑(lat, a3, TypeVar))
    else
        return UResult(Union{}, CC.EFFECTS_THROWS)
    end
    canconst = true
    local body
    if a3 isa CC.Const
        body = a3.val
    elseif CC.isconstType(a3)
        body = CC.type_parameter(a3)
    elseif CC.isType(a3)
        body = CC.type_parameter(a3)
        canconst = false
    else
        return UResult(Any, CC.Effects(CC.EFFECTS_TOTAL; nothrow))
    end
    (body isa Type || body isa TypeVar) || return UResult(Any, CC.EFFECTS_THROWS)
    if CC.has_free_typevars(body)
        local tv
        if a2 isa CC.Const
            tv = a2.val
        elseif a2 isa CC.PartialTypeVar
            tv = a2.tv
            canconst = false
        else
            return UResult(Any, CC.EFFECTS_THROWS)
        end
        tv isa TypeVar || return UResult(Any, CC.EFFECTS_THROWS)
        body = try
            UnionAll(tv, body)
        catch
            return UResult(Any, CC.EFFECTS_THROWS)
        end
    end
    rt = canconst ? CC.Const(body) : Type{body}
    return UResult(rt, CC.Effects(CC.EFFECTS_TOTAL; nothrow))
end

"The abstract_eval_setglobal! port: `args = [setglobal!, M, s, v(, order)]`."
function infer_setglobal(fr::Frame, args::Vector{Any})::UResult
    st = fr.st
    if !(4 <= length(args) <= 5) || CC.isvarargtype(args[end])
        return UResult(Union{}, CC.EFFECTS_THROWS, ArgumentError)
    end
    order_exct = Union{}
    if length(args) == 5
        order_exct = try
            CC.global_order_exct(args[5], #=loading=#false, #=storing=#true)
        catch
            Any
        end
    end
    ml = args[2]; sl = args[3]; vl = args[4]
    local rt, exct
    if ml isa CC.Const && sl isa CC.Const && ml.val isa Module && sl.val isa Symbol
        rt, exct = global_assign_rt_exct(st, ml.val::Module, sl.val::Symbol, vl)
    else
        wM = CC.widenconst(widenucond(ml)); wS = CC.widenconst(widenucond(sl))
        if !(CC.hasintersect(wM, Module) && CC.hasintersect(wS, Symbol))
            return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)
        elseif wM <: Module && wS <: Symbol
            rt, exct = vl, ErrorException
        else
            rt, exct = vl, Union{TypeError, ErrorException}
        end
    end
    exct = exct === Any ? Any : Union{exct, order_exct}
    eff = CC.Effects(CC.setglobal!_effects; nothrow = exct === Union{})
    return UResult(rt, eff, exct)
end

"""Effects and exception type of evaluating a statement's own operands
(the abstract_eval_value/abstract_eval_special_value port): non-const
global reads, mutable literals (#52531), maybe-undefined static parameters.
Folds `(e, exct)` through and returns the updated pair."""
function operand_effects(fr::Frame, s::StmtId, e::CC.Effects, @nospecialize(exct))
    ir = fr.ir
    for i in 1:UnifiedIR.nops(ir, s)
        o = UnifiedIR.getop(ir, s, i)
        t = UnifiedIR.optag(o)
        if t == UnifiedIR.TAG_GLOBAL
            g = ir.body.globals[UnifiedIR.payload(o)]
            rte = global_rte(fr.st, g.mod, g.name)
            e = CC.merge_effects(e, rte.effects)
            exct = exct === Any ? Any : CC.tmerge(CC.fallback_lattice, exct, rte.exct)
        elseif t == UnifiedIR.TAG_CONST
            # a literal referencing mutable memory (QuoteNode'd Ref, closure
            # box) makes that memory reachable without an argument: not
            # inaccessiblememonly (stock abstract_eval_special_value, #52531)
            v = ir.body.constants[UnifiedIR.payload(o)]
            CC.is_mutation_free_argtype(typeof(v)) ||
                (e = CC.merge_effects(e, MUTABLE_LITERAL_EFFECTS))
        elseif t == UnifiedIR.TAG_SPARAM
            # an undefined static parameter read throws UndefVarError
            # (stock abstract_eval_static_parameter)
            if sparam_maybe_undef(ir, Int(UnifiedIR.payload(o)))
                e = CC.Effects(e; nothrow = false)
                exct = exct === Any ? Any :
                       CC.tmerge(CC.fallback_lattice, exct, UndefVarError)
            end
        end
    end
    return (e, exct)
end

"Operand-evaluation effects for a control statement (no effects of its own —
the walkers handle those; `transfer`-dispatched kinds go through
`note_effects!` instead)."
function note_operand_effects!(fr::Frame, s::StmtId)
    e, exct = operand_effects(fr, s, CC.EFFECTS_TOTAL, Union{})
    e === CC.EFFECTS_TOTAL && exct === Union{} && return nothing
    e.nothrow ? (exct = Union{}) : (exct === Union{} && (exct = Any))
    fr.effects = CC.merge_effects(fr.effects, e)
    exct === Union{} || note_thrown!(fr, exct)
    return nothing
end

"""Record statement effects (and its raisable exception type) and fold them
into the frame accumulators, operand-evaluation effects included; the flag
column gets the UInt32 projection; the exception joins the innermost thrown
collector."""
function note_effects!(fr::Frame, s::StmtId, e::CC.Effects, @nospecialize(exct))
    e, exct = operand_effects(fr, s, e, exct)
    # a nothrow statement raises nothing; a throwing one raises at least *something*
    e.nothrow ? (exct = Union{}) : (exct === Union{} && (exct = Any))
    fr.stmt_effects[s.id] = effects_mask(e)
    fr.effects = CC.merge_effects(fr.effects, e)
    exct === Union{} || note_thrown!(fr, exct)
    return nothing
end

const MUTABLE_LITERAL_EFFECTS =
    CC.Effects(CC.EFFECTS_TOTAL; inaccessiblememonly = CC.ALWAYS_FALSE)

"Is static parameter `i` possibly undefined at runtime (`sptypes[i].undef`)?"
function sparam_maybe_undef(ir::UnifiedIR.IR, i::Int)
    und = get(ir.meta, :sptypes_undef, nothing)
    if und isa Vector{Bool}
        return 1 <= i <= length(und) ? und[i] : true
    end
    # no decoded undef information: sound only when assumed-undefined; but
    # a plain-value sparam (the common fully-specialized case) is defined
    if 1 <= i <= length(ir.sptypes)
        sp = ir.sptypes[i]
        return sp isa Core.SimpleVector || sp isa TypeVar
    end
    return true
end

# ---------------------------------------------------------------------------
# Plain-statement transfer functions
# ---------------------------------------------------------------------------

# Identify the refinement subject of a value operand: a cell (through a
# fresh cell_get) or the SSA statement itself. Returns nothing for
# non-refinable operands (constants, globals). Shared cells take no
# flow-sensitive refinements at all: a visible closure call between the test
# and the use may store (the same reason `cell_set` excludes them from the
# store-forwarding overlay), so an `isa`/typeassert fact about a
# `cell_shared` read does not survive to any later program point.
function cond_subject(fr::Frame, o::UnifiedIR.Operand)
    UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || return nothing
    sid = UnifiedIR.asstmt(o)
    if UnifiedIR.stmt_kind(fr.ir, sid) === K"cell_get"
        cellop = UnifiedIR.asstmt(UnifiedIR.getop(fr.ir, sid, 1))
        UnifiedIR.stmt_kind(fr.ir, cellop) === K"cell_shared" && return nothing
        return (:cell, cellop.id)
    end
    return (:stmt, sid.id)
end

function transfer(fr::Frame, s::StmtId, k::UnifiedIR.Kind)
    res = _transfer(fr, s, k)
    rt = res[1]
    eff = res[2]::CC.Effects
    exct = length(res) === 3 ? res[3] : (eff.nothrow ? Union{} : Any)
    # stock's per-statement epilogue: the statement-level `@assume_effects`
    # override (`merge_override_effects!`; covers the statement's own
    # effects — operand-evaluation effects merge separately in
    # `note_effects!`, stock's N.B.). The NOUB_IF_NOINBOUNDS callsite
    # resolution happens where interprocedural results are consumed
    # (`callsite_noub` in the call/invoke transfers), BEFORE the frame's own
    # boundscheck production (`refine_bc_noub`), whose conditional bit must
    # reach the frame result undemoted.
    ov = stmt_override_bits(UnifiedIR.stmt_flag(fr.ir, s))
    if ov != zero(UInt16)
        eff = CC.override_effects(eff, CC.decode_effects_override(ov))
    end
    note_effects!(fr, s, eff, exct)
    return rt
end

"Meet a caller argument lattice with a callee-side conditional type."
function meet_cond(@nospecialize(argt), @nospecialize(ct))
    lat = CC.fallback_lattice
    try
        if ct isa Type
            return CC.tmeet(lat, argt, ct)
        elseif CC.:⊑(lat, ct, argt)
            return ct
        end
    catch
    end
    return ct
end

"""Translate a callee's `UInterCond` return into a caller-local `UCond` (the
from_interconditional port): positional args start at operand `firstargop`,
so callee parameter `slot` maps to operand `firstargop + slot - 1`. Widens to
Bool when the caller's operand in that position is not a refinable subject."""
function apply_intercond(fr::Frame, s::StmtId, firstargop::Int, r::UResult)
    rt = r.rt
    rt isa UInterCond || return r
    ir = fr.ir
    opidx = firstargop + rt.slot - 1
    (firstargop <= opidx && opidx <= UnifiedIR.nops(ir, s)) ||
        return UResult(Bool, r.effects, r.exct)
    o = UnifiedIR.getop(ir, s, opidx)
    subj = cond_subject(fr, o)
    subj === nothing && return UResult(Bool, r.effects, r.exct)
    argt = widenucond(opl(fr, o))
    return UResult(UCond(subj, meet_cond(argt, rt.thentype),
                         meet_cond(argt, rt.elsetype)), r.effects, r.exct)
end

"""Call-site result refinement for visible closures (§5.7 piece 3): when the
callee operand's def is a `K"closure"` stmt in this frame, the call's result
type is the body's inferred return-type join and its effects are the body's
mask (`infer_closure!` records both). The site's argtypes feed the closure's
param join (`closure_args`) unless the closure is escaped/world-shifted
(whose params are already declared/`Any`). Arity-incompatible calls of a
visible closure always throw — the reference interpreter, the materialized
trampoline, and a native single-method closure all reject them, and methods
added to the closure type later are invisible without a world barrier
(shifted closures bypass this fast path entirely). Returns `nothing` when
the callee is not a visible closure op (generic `infer_call` applies)."""
function closure_callee_transfer(fr::Frame, s::StmtId)
    ir = fr.ir
    fo = UnifiedIR.getop(ir, s, 1)
    UnifiedIR.optag(fo) == UnifiedIR.TAG_STMT || return nothing
    cs = UnifiedIR.asstmt(fo)
    UnifiedIR.stmt_kind(ir, cs) === K"closure" || return nothing
    # a world barrier between creation and this call: the body may execute
    # against a newer method/binding table than inference consulted (§5.8) —
    # no refinement of any kind
    cs.id in fr.closure_shifted && return nothing
    rs = UnifiedIR.live_owned_regions(ir, cs)
    isempty(rs) && return nothing
    breg = UnifiedIR.getregion(ir, rs[1])
    np = length(breg.args)
    isva = false
    if UnifiedIR.nops(ir, cs) >= 1
        flags = UnifiedIR.imm_value(UnifiedIR.getop(ir, cs, 1))::Int64
        isva = (flags & UnifiedIR.CLOSURE_FLAG_ISVA) != 0
    end
    nargs = UnifiedIR.nops(ir, s) - 1
    if !(isva ? nargs >= np - 1 : nargs == np)
        return (Union{}, CC.EFFECTS_THROWS, Any)   # arity mismatch: guaranteed throw
    end
    if !(cs.id in fr.closure_escaped)
        args = Any[widenucond(opl(fr, UnifiedIR.getop(ir, s, 1 + i))) for i in 1:nargs]
        prm = Vector{Any}(undef, np)
        for i in 1:(isva ? np - 1 : np)
            prm[i] = args[i]
        end
        if isva
            rest = Any[args[i] for i in np:nargs]
            prm[np] = CC.builtin_tfunction(fr.st.cfg.interp, Core.tuple, rest, nothing)
        end
        old = get(fr.closure_args, cs.id, nothing)
        if old === nothing
            fr.closure_args[cs.id] = prm
            fr.cells_changed = true
        else
            for i in 1:np
                m = CC.tmerge(CC.fallback_lattice, old[i], prm[i])
                lat_eq(m, old[i]) || (old[i] = m; fr.cells_changed = true)
            end
        end
    end
    # def-before-use within a sweep makes these reads never stale at
    # convergence (a skipped def implies dead uses); missing entries are
    # unreadable, ⊥/no-guarantees only as a defensive default
    rt = get(fr.closure_rets, cs.id, Union{})
    eff = get(fr.closure_effs, cs.id, CC.Effects())
    return (rt, eff)
end

"""The stock NOUB_IF_NOINBOUNDS production rule, at inference over structured
IR: a boundscheck-taking memory builtin whose boundscheck argument is THIS
frame's own `K"boundscheck"` executes its bounds check unless that check is
elided by an `@inbounds` inlining context — so `noub = ALWAYS_FALSE` (from the
unprovable-inbounds argument type) refines to the conditional bit."""
function refine_bc_noub(fr::Frame, s::StmtId, e::CC.Effects)
    e.noub === CC.ALWAYS_FALSE || return e
    # an `@inbounds`-flagged statement (entry-carried IR_FLAG_INBOUNDS): its
    # boundscheck value may be pinned false — no conditional promise
    stmt_inbounds(fr.ir, s) && return e
    ir = fr.ir
    n = UnifiedIR.nops(ir, s)
    n >= 4 || return e
    is_boundscheck_callee(static_operand_value(ir, UnifiedIR.getop(ir, s, 1))) || return e
    o = UnifiedIR.getop(ir, s, n)
    UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || return e
    UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(o)) === K"boundscheck" || return e
    return CC.Effects(e; noub = CC.NOUB_IF_NOINBOUNDS)
end

"""The stock per-statement NOUB_IF_NOINBOUNDS resolution (the main-loop
branch after `abstract_eval_statement_expr`): a callee's conditional noub
refers to the callee's own `@boundscheck` blocks. When THIS statement is
`@inbounds`-flagged (entry-carried IR_FLAG_INBOUNDS), those blocks may be
elided by compilations of this body — including the stock-compiled code
concrete evaluation executes — so the conditional demotes to ALWAYS_FALSE.
Otherwise, when the frame does not propagate inbounds, the blocks are never
elided through us and the promise is unconditional."""
function callsite_noub(fr::Frame, s::StmtId, e::CC.Effects)
    if e.noub === CC.NOUB_IF_NOINBOUNDS
        stmt_inbounds(fr.ir, s) && return CC.Effects(e; noub = CC.ALWAYS_FALSE)
        fr.propagate_inbounds || return CC.Effects(e; noub = CC.ALWAYS_TRUE)
    end
    return e
end

"""Stock inlining marks its `Core._compute_sparams` / `Core._svec_ref`
sparam-reconstruction insertions removable-if-unused and never re-infers
them; this pipeline DOES re-infer spliced bodies every round (the flag
column republishes inference's effects projection), so the transfer must
re-derive those facts or the reconstruction could never be DCE'd once
`lift_svec_refs!` forwards its uses (stock `lift_svec_ref!`'s NamedTuple
constructor corpus). `_compute_sparams(::Method, args...)` is
`(+e,+n,+t)` with rt `SimpleVector` — the inliner emits it for the very
call that dispatched, so the runtime env intersection is non-empty; a
`_svec_ref(sp, idx)` whose vector IS such a `_compute_sparams` result is
in-bounds whenever `idx <= unionall_depth(method.sig)` (rt: the declared
statement type — the inliner seeded it from `sptypes_from_meth_instance`,
which this structural shape keeps valid; stock trusts the same
`insert_spval!` type unrevisited). Returns `nothing` for any other call."""
function sparam_reconstruction_transfer(fr::Frame, s::StmtId)
    ir = fr.ir
    n = UnifiedIR.nops(ir, s)
    n >= 3 || return nothing
    # the inliner interns the builtin itself (vop) — a cheap pool check
    # prunes every ordinary call before any lattice work
    fo = UnifiedIR.getop(ir, s, 1)
    UnifiedIR.optag(fo) == UnifiedIR.TAG_CONST || return nothing
    f = UnifiedIR.getconst(ir, fo)
    (f === Core._compute_sparams || f === Core._svec_ref) || return nothing
    eff() = CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE)
    if f === Core._compute_sparams
        ml = opl(fr, UnifiedIR.getop(ir, s, 2))
        (ml isa CC.Const && ml.val isa Method) || return nothing
        return (Core.SimpleVector, eff(), Union{})
    elseif f === Core._svec_ref && n == 3
        idxl = opl(fr, UnifiedIR.getop(ir, s, 3))
        (idxl isa CC.Const && idxl.val isa Int) || return nothing
        vecop = UnifiedIR.getop(ir, s, 2)
        UnifiedIR.optag(vecop) == UnifiedIR.TAG_STMT || return nothing
        def = UnifiedIR.asstmt(vecop)
        (UnifiedIR.stmt_kind(ir, def) === K"call" &&
         UnifiedIR.nops(ir, def) >= 3) || return nothing
        dfo = UnifiedIR.getop(ir, def, 1)
        (UnifiedIR.optag(dfo) == UnifiedIR.TAG_CONST &&
         UnifiedIR.getconst(ir, dfo) === Core._compute_sparams) || return nothing
        ml = opl(fr, UnifiedIR.getop(ir, def, 2))
        (ml isa CC.Const && ml.val isa Method) || return nothing
        1 <= (idxl.val::Int) <= CC.unionall_depth((ml.val::Method).sig) ||
            return nothing
        t0 = UnifiedIR.stmt_type(ir, s)
        return (t0 === nothing ? Any : t0, eff(), Union{})
    end
    return nothing
end

function _transfer(fr::Frame, s::StmtId, k::UnifiedIR.Kind)
    ir = fr.ir
    if k === K"call"
        cl = closure_callee_transfer(fr, s)
        # a closure body's conditional noub resolves at the call site like
        # any callee's (stock's per-statement rule)
        cl === nothing || return length(cl) === 3 ?
            (cl[1], callsite_noub(fr, s, cl[2]), cl[3]) :
            (cl[1], callsite_noub(fr, s, cl[2]))
        cond = conditional_call(fr, s)
        if cond !== nothing
            # the isdefined arm carries the builtin's own (nothrow) effects;
            # isa/===/! conditionals are total
            cond isa Tuple && return (cond[1], cond[2]::CC.Effects)
            return (cond, CC.EFFECTS_TOTAL)
        end
        spr = sparam_reconstruction_transfer(fr, s)
        spr === nothing || return spr
        args = Any[widenucond(a) for a in opls(fr, s, 1)]
        r = infer_call(fr, args; sid = s.id)
        r = apply_intercond(fr, s, 1, r)
        get(ENV, "UIR_DEBUG", "") == "1" && println("DBG call %", s.id, " args=", args, " -> ", r.rt, " exct=", r.exct)
        maybe_typeassert_refine!(fr, s, args, r.rt)
        fr.pending_refine === nothing && maybe_setfield_refine!(fr, s, args, r.rt)
        # order matters: resolve CALLEE conditional noub against this
        # statement's inbounds context (stock 4188), then produce THIS
        # frame's own-boundscheck conditional (which must survive)
        return (r.rt, refine_bc_noub(fr, s, callsite_noub(fr, s, r.effects)), r.exct)
    elseif k === K"invoke"
        tl = opl(fr, UnifiedIR.getop(ir, s, 1))
        args = Any[widenucond(a) for a in opls(fr, s, 2)]
        r = infer_invoke_target(fr, tl, args)
        r = apply_intercond(fr, s, 2, r)
        return (r.rt, callsite_noub(fr, s, r.effects), r.exct)
    elseif k === K"intrinsic"
        cond = conditional_call(fr, s)   # not_int Conditional inversion
        if cond !== nothing
            cond isa Tuple && return (cond[1], cond[2]::CC.Effects)
            return (cond, CC.EFFECTS_TOTAL)
        end
        args = opls(fr, s, 1)
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
        f === nothing && return (Any, CC.Effects())
        argl = Any[widenucond(a) for a in args[2:end]]
        rt = try
            CC.builtin_tfunction(fr.st.cfg.interp, f, argl, nothing)
        catch
            Any
        end
        eff = builtin_call_effects(f, argl, rt)
        return (rt, eff, eff.nothrow ? Union{} : builtin_call_exct(f, argl, rt))
    elseif k === K"extract"
        vl = widenucond(opl(fr, UnifiedIR.getop(ir, s, 1)))
        idx = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, s, 2))::Int64)
        argl = Any[vl, CC.Const(idx)]
        rt = CC.builtin_tfunction(fr.st.cfg.interp, Core.getfield, argl, nothing)
        eff = builtin_call_effects(Core.getfield, argl, rt)
        return (rt, eff, eff.nothrow ? Union{} :
                builtin_call_exct(Core.getfield, argl, rt))
    elseif k === K"select"
        c = opl(fr, UnifiedIR.getop(ir, s, 1))
        a = opl(fr, UnifiedIR.getop(ir, s, 2))
        b = opl(fr, UnifiedIR.getop(ir, s, 3))
        cw = CC.widenconst(widenucond(c))
        eff = (cw isa Type && cw <: Bool) ? CC.EFFECTS_TOTAL :
              CC.Effects(CC.EFFECTS_TOTAL; nothrow = false)   # TypeError on non-Bool
        c isa CC.Const && c.val === true && return (a, eff, TypeError)
        c isa CC.Const && c.val === false && return (b, eff, TypeError)
        return (CC.tmerge(CC.fallback_lattice, widenucond(a), widenucond(b)), eff, TypeError)
    elseif k === K"refine" || k === K"value"
        return (opl(fr, UnifiedIR.getop(ir, s, 1)), CC.EFFECTS_TOTAL)
    elseif k === K"globalref"
        return (opl(fr, UnifiedIR.getop(ir, s, 1)), CC.EFFECTS_TOTAL)  # operand effects cover it
    elseif k === K"new"
        return transfer_new(fr, s)
    elseif k === K"splatnew"
        return transfer_splatnew(fr, s)
    elseif k === K"foreigncall"
        if UnifiedIR.nops(ir, s) >= 1
            m1 = opl(fr, UnifiedIR.getop(ir, s, 1))
            if m1 isa CC.Const && m1.val === FOREIGNGLOBAL_MARKER
                # Expr(:foreignglobal, name): the cglobal lowering (stock's
                # abstract_eval_foreignglobal — always Ptr{Cvoid})
                return (Ptr{Cvoid}, CC.EFFECTS_UNKNOWN)
            end
        end
        UnifiedIR.nops(ir, s) >= 2 || return (Any, CC.EFFECTS_UNKNOWN)
        # `@ccall ... @assume_effects`-style overrides ride the cconv tuple
        # (operand 5, mirroring Expr(:foreigncall).args[5]; stock decode at
        # abstract_eval_foreigncall)
        eff = CC.EFFECTS_UNKNOWN
        if UnifiedIR.nops(ir, s) >= 5
            cconv = opl(fr, UnifiedIR.getop(ir, s, 5))
            if cconv isa CC.Const && (q = cconv.val; q isa QuoteNode) &&
               (v = q.value; v isa Tuple{Symbol, UInt16, Bool})
                eff = try
                    CC.override_effects(eff, CC.decode_effects_override(v[2]))
                catch
                    eff
                end
            end
        end
        rtl = opl(fr, UnifiedIR.getop(ir, s, 2))
        T = rtl isa CC.Const ? rtl.val : nothing
        mi = get(ir.meta, :mi, nothing)
        if T !== nothing && mi isa Core.MethodInstance
            # sparam-dependent ccall types instantiate in the mi environment
            # (the full sp_type_rewrap port): unsafe_wrap's Array{T,1} etc.
            rt = try
                CC.sp_type_rewrap(T, mi, true)
            catch
                nothing
            end
            rt === nothing || return (rt, eff)
        end
        return (foreigncall_rt(T), eff)
    elseif k === K"isdefined_global"
        # a binding can become defined later: not consistent; reads the
        # binding table: not inaccessiblememonly
        return (Bool, CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE,
                                 inaccessiblememonly = CC.ALWAYS_FALSE))
    elseif k === K"cell_isdefined"
        cellop = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1))
        cellid = cellop.id
        if UnifiedIR.stmt_kind(ir, cellop) === K"cell_shared" || cellid in fr.poisoned_cells
            # a visible closure may store between the test and any use: no
            # flow-sensitive definedness facts (the cond_subject discipline)
            return (Bool, CC.EFFECTS_TOTAL)
        end
        if !(cellid in fr.newed_cells) || refined(fr, (:cell, cellid)) !== nothing
            # provably assigned here: no cell_new at all, or an active
            # flow-sensitive witness (store/completed read on every path)
            return (CC.Const(true), CC.EFFECTS_TOTAL)
        end
        # conditional definedness (stock's @isdefined undef refinement): the
        # then path carries a witness typed by the join of the frame's stores
        # (any defined value is bounded by it); the else path carries an
        # explicit no-witness KILL
        return (UCond((:cell, cellid), cell_lattice(fr, cellid), REFINE_KILL),
                CC.EFFECTS_TOTAL)
    elseif k === K"boundscheck"
        # value depends on the inlining context: not consistent — unless its
        # every use is the boundscheck argument of a memory builtin, where the
        # value cannot reach the frame's result (the noub machinery models
        # that dependence instead; stock's post-opt boundscheck rule)
        s.id in fr.bc_guarded && return (Bool, CC.EFFECTS_TOTAL)
        return (Bool, CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE))
    elseif k === K"cell" || k === K"cell_shared"
        return (Any, CC.EFFECTS_TOTAL)      # the cell token
    elseif k === K"cell_get"
        cellid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
        # maybe-undef read can throw UndefVarError — unless a flow-sensitive
        # refinement is active for the cell: refinements arise only from a
        # store (`cell_set` overlay) or a completed read (branch/typeassert
        # subjects) on every path here, either of which proves definedness
        # (stock's VarState.undef precision; `cell_new` kills the witness)
        eff = (cellid in fr.newed_cells && refined(fr, (:cell, cellid)) === nothing) ?
              CC.Effects(CC.EFFECTS_TOTAL; nothrow = false) : CC.EFFECTS_TOTAL
        # escape/world discipline (§5.7): reads of a poisoned shared cell
        # (some capturing closure escapes or is world-shifted, or the cell
        # itself escapes as a value) are Any — the join is still accumulated
        # for diagnostics, but never used for refinement
        cellid in fr.poisoned_cells && return (Any, eff, UndefVarError)
        return (cell_lattice(fr, cellid), eff, UndefVarError)
    elseif k === K"cell_set"
        cellop = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1))
        cellid = cellop.id
        # a store kills any active Conditional/typeassert refinement of the cell
        for rm in fr.refinements
            delete!(rm, (:cell, cellid))
        end
        vl = widenucond(opl(fr, UnifiedIR.getop(ir, s, 2)))
        old = get(fr.celltypes, cellid, nothing)
        if old === nothing
            fr.celltypes[cellid] = vl
            fr.cells_changed = true
        else
            new = CC.tmerge(CC.fallback_lattice, old, vl)
            if !lat_eq(new, old)
                fr.celltypes[cellid] = new
                fr.cells_changed = true
            end
        end
        # flow-sensitive overlay (the VarTable port): later reads on this path
        # see the just-written value; joins fall back to the monotone celltypes.
        # The walker pushes it as a refinement scope; edge propagation through
        # `blockrefs` joins it across cfg edges, and the kill above removes it
        # on reassignment. Shared cells are excluded (closures may write).
        shared = UnifiedIR.stmt_kind(ir, cellop) === K"cell_shared"
        shared || (fr.pending_refine = (:cell, cellid) => vl)
        # writes to closure-shared cells are observable mutations (the
        # materialized closure's untyped field): setfield!-shaped effects
        eff = shared ? CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE,
                                  effect_free = CC.EFFECT_FREE_IF_INACCESSIBLEMEMONLY,
                                  inaccessiblememonly = CC.ALWAYS_FALSE) :
                       CC.EFFECTS_TOTAL
        return (nothing, eff)
    elseif k === K"cell_new"
        # re-declaration makes the binding fresh and unassigned: any active
        # refinement of the cell (a definedness witness above) must die on
        # this path. An innermost KILL shadows outer scopes without touching
        # them (deleting an outer entry could unmask a staler one).
        cellid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
        fr.pending_refine = (:cell, cellid) => REFINE_KILL
        return (nothing, CC.EFFECTS_TOTAL)
    elseif k === K"throw_undef_if_not"
        condl = opl(fr, UnifiedIR.getop(ir, s, 1))
        condl isa CC.Const && condl.val === true && return (nothing, CC.EFFECTS_TOTAL)
        # guaranteed throw poisons the tail (walker's dead-tail rule)
        condl isa CC.Const && condl.val === false &&
            return (Union{}, CC.EFFECTS_THROWS, UndefVarError)
        return (nothing, CC.EFFECTS_THROWS, UndefVarError)
    elseif k === K"latestworld" || k === K"coverage_effect"
        # not independently removable, but no observable effect of their own
        return (nothing, CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE,
                                    effect_free = CC.EFFECT_FREE_GLOBALLY,
                                    inaccessiblememonly = CC.ALWAYS_FALSE))
    elseif k === K"gc_preserve_end"
        return (nothing, CC.Effects(CC.EFFECTS_TOTAL; effect_free = CC.EFFECT_FREE_GLOBALLY))
    elseif k === K"gc_preserve_begin"
        return (Any, CC.Effects(CC.EFFECTS_TOTAL; effect_free = CC.EFFECT_FREE_GLOBALLY))
    elseif k === K"copyast"
        # fresh mutable copy each evaluation: not consistent
        return (Any, CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE))
    elseif k === K"new_opaque_closure"
        return transfer_new_opaque_closure(fr, s)
    elseif k === K"method_def" || k === K"cfunction"
        return (Any, CC.EFFECTS_UNKNOWN)
    else
        return (Any, CC.EFFECTS_UNKNOWN)   # unknown/external kind: opacity contract §8.2
    end
end

"ccall return-position semantics (the sp_type_rewrap port): Ref{T} means a
rooted T; Ref{Any} returns are invalid; free typevars degrade to Any."
function foreigncall_rt(@nospecialize(T))
    T isa Type || return Any
    T === Union{} && return Union{}
    if T isa DataType && T.name === Ref.body.name
        T = T.parameters[1]
        T === Any && return Union{}     # a return type of Ref{Any} is invalid
        T isa TypeVar && (T = T.ub)
    end
    T isa Type || return Any
    return CC.has_free_typevars(T) ? Any : T
end

"Back-propagate `typeassert(x, T)` as a refinement of x's subject (stock
SlotRefinement); the walker pushes `fr.pending_refine` for the region rest."
function maybe_typeassert_refine!(fr::Frame, s::StmtId, args::Vector{Any},
                                  @nospecialize(rt))
    length(args) == 3 || return nothing
    f = CC.singleton_type(args[1])
    f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
    f === typeassert || return nothing
    rt === Union{} && return nothing
    subj = cond_subject(fr, UnifiedIR.getop(fr.ir, s, 2))
    subj === nothing && return nothing
    fr.pending_refine = subj => rt
    return nothing
end

"""Back-propagate a successful `setfield!(x, name, v[, order])` as a
field-definedness refinement of x's subject (stock abstract_call_known's
form_partially_defined_struct site): later `isdefined(x, name)` folds and
later reads of that field are nothrow. Only with full argument type
information (stock's vararg gate)."""
function maybe_setfield_refine!(fr::Frame, s::StmtId, args::Vector{Any},
                                @nospecialize(rt))
    4 <= length(args) <= 5 || return nothing
    f = CC.singleton_type(args[1])
    f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
    f === setfield! || return nothing
    rt === Union{} && return nothing
    any(a -> CC.isvarargtype(a), args) && return nothing
    subj = cond_subject(fr, UnifiedIR.getop(fr.ir, s, 2))
    subj === nothing && return nothing
    refined = try
        CC.form_partially_defined_struct(CC.fallback_lattice, args[2], args[3])
    catch
        nothing
    end
    refined === nothing && return nothing
    fr.pending_refine = subj => refined
    return nothing
end

"Produce UCond lattice elements for conditional-shaped calls (§10.3)."
function conditional_call(fr::Frame, s::StmtId)
    ir = fr.ir
    n = UnifiedIR.nops(ir, s)
    fo = UnifiedIR.getop(ir, s, 1)
    fl = opl(fr, fo)
    f = CC.singleton_type(fl)
    f === nothing && fl isa CC.Const && (f = fl.val)
    f === nothing && return nothing
    lat = CC.fallback_lattice
    if f === isa && n == 3
        vo = UnifiedIR.getop(ir, s, 2)
        vt = widenucond(opl(fr, vo))
        tl = opl(fr, UnifiedIR.getop(ir, s, 3))
        tl isa CC.Const && tl.val isa Type || return nothing
        T = tl.val
        rt = CC.builtin_tfunction(fr.st.cfg.interp, isa, Any[vt, tl], nothing)
        rt isa CC.Const && return rt                     # statically decided
        subj = cond_subject(fr, vo)
        subj === nothing && return rt
        thent = CC.tmeet(lat, vt, T)
        elset = CC.typesubtract(CC.widenconst(vt), T,
                                CC.InferenceParams().max_union_splitting)
        return UCond(subj, thent, elset)
    elseif f === (===) && n == 3
        ao = UnifiedIR.getop(ir, s, 2)
        bo = UnifiedIR.getop(ir, s, 3)
        al = widenucond(opl(fr, ao))
        bl = widenucond(opl(fr, bo))
        rt = CC.builtin_tfunction(fr.st.cfg.interp, ===, Any[al, bl], nothing)
        rt isa CC.Const && return rt
        # refine against a singleton side (x === nothing and friends).
        # NB: the sentinel must be distinct from the VALUE nothing — the
        # `x === nothing` pattern is the single most important client.
        for (co, cl, vo, vl) in ((ao, al, bo, bl), (bo, bl, ao, al))
            has_c = false
            local cval
            if cl isa CC.Const
                cval = cl.val
                has_c = true
            else
                stype = CC.singleton_type(cl)
                if stype !== nothing
                    cval = stype
                    has_c = true
                end
            end
            has_c || continue
            Base.issingletontype(typeof(cval)) || continue
            subj = cond_subject(fr, vo)
            subj === nothing && continue
            vt = widenucond(vl)
            thent = CC.tmeet(lat, vt, typeof(cval))
            thent === Union{} && (thent = typeof(cval))
            elset = CC.typesubtract(CC.widenconst(vt), typeof(cval),
                                    CC.InferenceParams().max_union_splitting)
            return UCond(subj, thent, elset)
        end
        return rt
    elseif f === isdefined && n == 3
        # stock abstract_isdefined: refine the subject's field-definedness
        # (PartialStruct undefs) along the branch arms
        vo = UnifiedIR.getop(ir, s, 2)
        vl = opl(fr, vo)
        argtype2 = widenucond(vl)
        fldl = widenucond(opl(fr, UnifiedIR.getop(ir, s, 3)))
        argl = Any[argtype2, fldl]
        rt = try
            CC.builtin_tfunction(fr.st.cfg.interp, isdefined, argl, nothing)
        catch
            return nothing
        end
        # unlike isa/===, isdefined is not total: mutable/module subjects
        # taint consistency (the field can become defined later) — carry the
        # builtin's own effects with the refinement (nothrow-gated so the
        # exct channel stays empty)
        eff = builtin_call_effects(isdefined, argl, rt)
        eff.nothrow || return nothing
        rt isa CC.Const && return (rt, eff)
        subj = cond_subject(fr, vo)
        subj === nothing && return (rt, eff)
        wat = CC.widenconst(argtype2)
        if wat isa Union
            thent = Union{}
            elset = Union{}
            for ty in CC.uniontypes(wat)
                cnd = try
                    CC.isdefined_tfunc(lat, ty, fldl)
                catch
                    Bool
                end
                if cnd isa CC.Const
                    if cnd.val === true
                        thent = CC.tmerge(lat, thent, ty)
                    else
                        elset = CC.tmerge(lat, elset, ty)
                    end
                else
                    thent = CC.tmerge(lat, thent, ty)
                    elset = CC.tmerge(lat, elset, ty)
                end
            end
            return (UCond(subj, thent, elset), eff)
        end
        thent = try
            CC.form_partially_defined_struct(lat, argtype2, fldl)
        catch
            nothing
        end
        thent === nothing && return (rt, eff)
        return (UCond(subj, thent, argtype2), eff)
    elseif (f === (!) || f === Core.Intrinsics.not_int) && n == 2
        # stock's Conditional inversion for `!`/`not_int` (loop lowerings
        # negate the `=== nothing` exit test through not_int)
        cl = opl(fr, UnifiedIR.getop(ir, s, 2))
        cl isa UCond && return UCond(cl.subject, cl.elsetype, cl.thentype)
        return nothing
    elseif f === Core.ifelse && n == 4
        cl = opl(fr, UnifiedIR.getop(ir, s, 2))
        if cl isa CC.Const && cl.val isa Bool
            return opl(fr, UnifiedIR.getop(ir, s, cl.val ? 3 : 4))
        end
        return nothing
    end
    return nothing
end

# ---------------------------------------------------------------------------
# new / splatnew (the abstract_eval_new port)
# ---------------------------------------------------------------------------

"The stock `:new` consistency model (abstract_eval_new): any (pointer-carrying)
uninitialized field → never consistent; mutable → CONSISTENT_IF_NOTRETURNED
(frame finish may resolve it against the return type); immutable → consistent."
function new_consistency(ut::DataType, fcount::Union{Nothing,Int}, nargs::Int)
    has_any_uninitialized = fcount === nothing || (fcount > nargs &&
        Base.any(i -> CC.is_field_pointerfree(ut, i), (nargs + 1):fcount))
    if has_any_uninitialized
        return CC.ALWAYS_FALSE
    elseif ismutabletype(ut)
        return CC.CONSISTENT_IF_NOTRETURNED
    else
        return CC.ALWAYS_TRUE
    end
end

const NEW_EXCT = Union{ErrorException,TypeError}

"""Every VALUE of the type lattice `tl` is a fully-applied instantiation of
a (non-Tuple, non-abstract) struct wrapper — i.e. a concrete type — even
though no single concrete type is statically known. The `Type{C{_A}} where
_A` shape a materialized apply_type over reconstructed sparams produces:
its Type parameter is a DataType (fully applied by construction — a
partial application would be a UnionAll) with free typevars bound by the
enclosing `where`s."""
function concrete_shaped_type_lattice(@nospecialize tl)
    tw = tl
    while tw isa UnionAll
        tw = tw.body
    end
    # `Type{C{_A}}` here is a TypeEq (exact-type lattice), not a DataType —
    # `isType`/`type_parameter` cover both encodings
    CC.isType(tw) || return false
    p = try
        CC.type_parameter(tw)
    catch
        return false
    end
    p isa DataType || return false
    (isabstracttype(p) || p.name === Tuple.name) && return false
    w = Base.unwrap_unionall(p.name.wrapper)
    return length(p.parameters) == length((w::DataType).parameters)
end

function transfer_new(fr::Frame, s::StmtId)
    ir = fr.ir
    lat = CC.fallback_lattice
    tl = widenucond(opl(fr, UnifiedIR.getop(ir, s, 1)))
    local rt, isexact
    try
        rt, isexact = CC.instanceof_tfunc(tl, true)
    catch
        return (Any, CC.EFFECTS_UNKNOWN, NEW_EXCT)
    end
    rt === Union{} && return (Union{}, CC.EFFECTS_THROWS, NEW_EXCT)
    nargs = UnifiedIR.nops(ir, s) - 1
    ut = Base.unwrap_unionall(rt)
    (ut isa DataType && !isabstracttype(ut)) ||
        return (rt isa Type ? rt : Any, CC.EFFECTS_UNKNOWN, NEW_EXCT)
    try
        ismut = ismutabletype(ut)
        fcount = CC.datatype_fieldcount(ut)
        consistent = new_consistency(ut, fcount, nargs)
        (fcount === nothing || nargs > fcount) &&
            return (rt, CC.Effects(CC.EFFECTS_UNKNOWN; consistent), NEW_EXCT)
        nothrow = CC.isconcretedispatch(rt)
        tvfields = false
        if !nothrow && concrete_shaped_type_lattice(tl)
            # the type operand's lattice is `Type{C{_A}} where _A` for a
            # FULLY-APPLIED struct wrapper (a materialized `new{T}` through
            # apply_type, the DoAllocNoEscapeSparam shape): every VALUE of
            # that lattice is a full instantiation — a concrete type — so
            # the allocation itself cannot throw even though `instanceof`
            # is inexact (stock loses this precision, its @test_broken).
            # Field-assignment nothrow is only checkable against declared
            # field types carrying NO free typevars (`fieldtype` on the
            # UnionAll joins those loosely: an `x::T` field would accept
            # anything statically while throwing at runtime).
            nothrow = true
            tvfields = true
        end
        ats = Vector{Any}(undef, nargs)
        anyrefine = false
        allconst = CC.isconcretedispatch(rt)
        for i in 1:nargs
            at = widenucond(opl(fr, UnifiedIR.getop(ir, s, i + 1)))
            ft = fieldtype(rt, i)
            if nothrow && tvfields
                fti = fieldtype(ut, i)
                nothrow = !(fti isa TypeVar) && fti isa Type && !CC.has_free_typevars(fti)
            end
            nothrow && (nothrow = CC.:⊑(lat, at, ft))
            at = CC.tmeet(lat, at, ft)
            at === Union{} && return (Union{}, CC.EFFECTS_THROWS, TypeError)   # guaranteed TypeError
            if ismut && !isconst(rt, i)
                ats[i] = ft            # field may be mutated later
                allconst = false
                continue
            end
            allconst &= at isa CC.Const
            if !anyrefine
                anyrefine = CC.has_nontrivial_extended_info(lat, at) ||
                            CC.:⋤(lat, at, ft)
            end
            ats[i] = at
        end
        eff = CC.Effects(CC.EFFECTS_TOTAL; consistent, nothrow)
        if allconst && fcount == nargs && consistent === CC.ALWAYS_TRUE
            argvals = Vector{Any}(undef, nargs)
            for j in 1:nargs
                argvals[j] = (ats[j]::CC.Const).val
            end
            v = try
                CC.Const(ccall(:jl_new_structv, Any, (Any, Ptr{Cvoid}, UInt32),
                               rt, argvals, UInt32(nargs)))
            catch
                nothing
            end
            v === nothing || return (v, eff, NEW_EXCT)
        end
        # under-initialized news (nargs < min_ninitialized — the #52857
        # class) MUST keep the missing fields' undef facts: the plain
        # DataType would let getfield_nothrow trust the type-level
        # "first min_ninitialized fields are defined" invariant the
        # allocation just violated, and the optimizer would delete the
        # load's conditional UndefRefError throw stock preserves (F11)
        if anyrefine || nargs != CC.datatype_min_ninitialized(rt)
            undefs = Union{Nothing,Bool}[false for _ in 1:nargs]
            if nargs < fcount
                for i in (nargs + 1):fcount
                    ft = fieldtype(rt, i)
                    push!(ats, ft)
                    push!(undefs, ft === Union{} ? true :
                          (isconcretetype(ft) && CC.datatype_pointerfree(ft) ?
                           false : nothing))
                end
            end
            return (CC.PartialStruct(lat, rt, undefs, ats), eff, NEW_EXCT)
        end
        return (rt, eff, NEW_EXCT)
    catch
        return (rt isa Type ? rt : Any, CC.EFFECTS_UNKNOWN, NEW_EXCT)
    end
end

"""The abstract_eval_new_opaque_closure port: `opaque_closure_tfunc` builds a
`PartialOpaque` when the source Method is statically known (and the
allow-partial flag — operand 4 — is literally `true`), giving OC call sites
the source to devirtualize against. Stock's eager create-site child inference
(OpaqueClosureCreateInfo) is not needed here: `infer_opaque_call` re-derives
the callee from the lattice element at each use. Effects/exct parity with
stock: `Effects()`/`Any`."""
function transfer_new_opaque_closure(fr::Frame, s::StmtId)
    ir = fr.ir
    nop = UnifiedIR.nops(ir, s)
    nop >= 5 || return (Union{}, CC.Effects(), Any)
    mi = get(ir.meta, :mi, nothing)
    mi isa Core.MethodInstance || (mi = get(ir.meta, :method_instance, nothing))
    mi isa Core.MethodInstance || return (Any, CC.Effects(), Any)
    args = Any[widenucond(opl(fr, UnifiedIR.getop(ir, s, i))) for i in 1:5]
    env = Any[widenucond(opl(fr, UnifiedIR.getop(ir, s, i))) for i in 6:nop]
    rt = try
        CC.opaque_closure_tfunc(CC.fallback_lattice, args[1], args[2], args[3],
                                args[5], env, mi)
    catch
        Any
    end
    if rt isa CC.PartialOpaque
        a4 = args[4]
        # stock: `ea[4] !== true` disables PartialOpaque propagation
        (a4 isa CC.Const && a4.val === true) || (rt = CC.widenconst(rt))
    end
    # stock's stmt_effect_flags: a structurally well-formed new_opaque_closure
    # (exact Tuple argt, Type bounds, Method source) is nothrow and removable
    # (consistent stays false — each evaluation allocates a fresh identity)
    wellformed = try
        argt, isexact = CC.instanceof_tfunc(args[1], true)
        isexact && argt isa Type && argt <: Tuple &&
            CC.:⊑(CC.fallback_lattice, args[2], Type) &&
            CC.:⊑(CC.fallback_lattice, args[3], Type) &&
            CC.:⊑(CC.fallback_lattice, args[5], Method)
    catch
        false
    end
    wellformed || return (rt, CC.Effects(), Any)
    return (rt, CC.Effects(CC.EFFECTS_TOTAL; consistent = CC.ALWAYS_FALSE), Union{})
end

function transfer_splatnew(fr::Frame, s::StmtId)
    ir = fr.ir
    lat = CC.fallback_lattice
    tl = widenucond(opl(fr, UnifiedIR.getop(ir, s, 1)))
    local rt, isexact
    try
        rt, isexact = CC.instanceof_tfunc(tl, true)
    catch
        return (Any, CC.EFFECTS_UNKNOWN, NEW_EXCT)
    end
    rt === Union{} && return (Union{}, CC.EFFECTS_THROWS, NEW_EXCT)
    res = rt isa Type ? rt : Any
    try
        nothrow = false
        if UnifiedIR.nops(ir, s) == 2 && CC.isconcretedispatch(rt) && !ismutabletype(rt)
            at = widenucond(opl(fr, UnifiedIR.getop(ir, s, 2)))
            n = fieldcount(rt)
            if at isa CC.Const && at.val isa Tuple && n == length(at.val::Tuple) &&
               all(i -> getfield(at.val::Tuple, i) isa fieldtype(rt, i), 1:n)
                nothrow = isexact
                res = CC.Const(ccall(:jl_new_structt, Any, (Any, Any), rt, at.val))
            elseif at isa CC.PartialStruct && CC.:⊑(lat, at, Tuple) && n > 0 &&
                   n == length(at.fields) && !CC.isvarargtype(at.fields[end]) &&
                   all(i -> CC.:⊑(lat, at.fields[i], fieldtype(rt, i)), 1:n)
                nothrow = isexact
                res = CC.PartialStruct(lat, rt, Union{Nothing,Bool}[false for _ in 1:n],
                                       Any[f for f in at.fields])
            end
        end
        local consistent::UInt8
        u = Base.unwrap_unionall(rt)
        if u isa DataType && !isabstracttype(u)
            consistent = new_consistency(u, CC.datatype_fieldcount(u),
                                         typemax(Int) #= all fields supplied =#)
        else
            consistent = CC.ALWAYS_FALSE
        end
        return (res, CC.Effects(CC.EFFECTS_TOTAL; consistent, nothrow), NEW_EXCT)
    catch
        return (res, CC.EFFECTS_UNKNOWN, NEW_EXCT)
    end
end

# ---------------------------------------------------------------------------
# Interprocedural calls
# ---------------------------------------------------------------------------

const CONSTPROP_SRC_LIMIT = 250     # const_prop_entry_heuristic analog
# Frames per top-level query. Measured: the loading.jl giants
# (_include_from_serialized/compilecache/stale_cachefile) exhaust even 250k
# frames — a structural const-prop recompute cost, not a calibration issue —
# and deep IO chains (printstyled) become pathological beyond depth 128, so
# both knobs stay at the values the sweep timings were measured at; cutoffs
# resolve through native_fallback (or Any when the fallback is off).
const FRAME_BUDGET = 60_000

"""Method-table query for a call signature through `CC.findall` (which
reports the world range the answer is valid for and whether the match set is
ambiguous). With a collector attached (driver mode) the result is recorded so
the driver can emit stock-encoded method edges. Returns the
`MethodLookupResult`, or `nothing` when the query fails or exceeds
`max_methods` (the no-information answer, sound at every world without an
edge)."""
function lookup_call_matches(st::UInferState, @nospecialize(sig))
    result = try
        CC.findall(sig, CC.InternalMethodTable(st.cfg.world); limit = st.cfg.max_methods)
    catch
        nothing
    end
    result === nothing && return nothing
    col = st.edges
    if col !== nothing
        record_call!(col, sig, result)
        trace!(col, (0x1, sig, result, st.cfg.max_methods))
    end
    return result
end

"""The abstract_call_opaque_closure port. The callee frame's SELF slot is
seeded with the closure's capture ENVIRONMENT tuple (stock's
`newargtypes[1] = ft.env` — the runtime OC ABI passes captures as argument
1, and the body reads them via `getfield(_1, i)`), and the signature is
built from that same env element, so the mi-keyed generic frame
(`specialize_method` on `Tuple{env..., argts...}`) is exactly as precise as
stock's. The check block reproduces stock's implicit type asserts: a return
value or argument tuple outside the declared OC signature makes the call
!nothrow with a TypeError arm."""
function infer_opaque_call(fr::Frame, closure::CC.PartialOpaque, args::Vector{Any})::UResult
    tt = closure.typ
    envl = closure.env
    sigparts = Any[CC.widenconst(envl)]
    for i in 2:length(args)
        a = args[i]
        CC.isvarargtype(a) && return UResult(Any, CC.Effects())
        t = CC.widenconst(a)
        t === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)
        push!(sigparts, t)
    end
    local sig, ocsig, ocrt
    try
        sig = Tuple{sigparts...}
        utt = Base.unwrap_unionall(tt)::DataType
        ocargsig = Base.rewrap_unionall(utt.parameters[1], tt)
        oa = Base.unwrap_unionall(ocargsig)
        oa isa DataType || return UResult(Any, CC.Effects())
        ocsig = Base.rewrap_unionall(Tuple{Tuple, oa.parameters...}, ocargsig)
        p2 = utt.parameters[2]
        ocrt = Base.rewrap_unionall(p2 isa TypeVar ? p2.ub : p2, tt)
        ocrt isa Type || (ocrt = Any)
    catch
        return UResult(Any, CC.Effects())
    end
    if !Base.hasintersect(sig, ocsig)
        # arity/type mismatch is a guaranteed dispatch failure (stock's
        # EFFECTS_THROWS + MethodError∪TypeError)
        return UResult(Union{}, CC.EFFECTS_THROWS, Union{MethodError,TypeError})
    end
    ocmethod = closure.source
    ocmethod isa Method || return UResult(Any, CC.Effects())
    if !isdefined(ocmethod, :source)
        # created from optimized source: cannot infer further; the declared
        # return type still binds (stock's ocrt branch)
        return UResult(ocrt, CC.Effects(), Any)
    end
    match = Core.MethodMatch(sig, Core.svec(), ocmethod, sig <: ocsig)
    cargs = Vector{Any}(undef, length(args))
    cargs[1] = envl
    for i in 2:length(args)
        cargs[i] = args[i]
    end
    r = infer_method(fr, match, cargs)
    rt = r.rt
    eff = r.effects
    exct = r.exct
    ok = try
        CC.:⊑(CC.fallback_lattice, widenucond(rt), ocrt) && sig <: ocsig
    catch
        false
    end
    if !ok
        # implicit type asserts on the arguments and the return value
        eff = CC.Effects(eff; nothrow = false)
        exct = exct === Any ? Any :
            CC.tmerge(CC.fallback_lattice, exct, TypeError)
    end
    return UResult(rt, eff, exct)
end

function infer_call(fr::Frame, args::Vector{Any}; sid::Int32 = Int32(0))::UResult
    st = fr.st
    ftl = args[1]
    f = CC.singleton_type(ftl)
    if f === nothing && ftl isa CC.Const
        f = ftl.val
    end
    if f === SPARAM_READ_MARKER && length(args) == 2
        # statement-position static-parameter read (entry marker): the value
        # is the parameter's lattice element; the maybe-undef UndefVarError
        # arrives through the operand-effects channel (note_effects!)
        return UResult(args[2], CC.EFFECTS_TOTAL, Union{})
    end
    if f isa Core.Builtin
        if f === Core._apply_iterate
            return infer_apply(fr, args; sid)
        elseif f === Core.invoke
            return infer_invoke(fr, args)
        elseif f === Core.throw
            # the raised value is the exception: its type is the exct
            exct = length(args) == 2 ? CC.widenconst(widenucond(args[2])) : Any
            return UResult(Union{}, CC.EFFECTS_THROWS, exct)
        elseif f === Core.throw_methoderror
            # stock abstract_throw_methoderror: zero call args raises
            # ArgumentError; an imprecise (vararg) arity may be either
            exct = if length(args) == 1
                ArgumentError
            elseif !CC.isvarargtype(args[2])
                MethodError
            else
                Union{MethodError, ArgumentError}
            end
            return UResult(Union{}, CC.EFFECTS_THROWS, exct)
        elseif f === setglobal!
            return infer_setglobal(fr, args)
        elseif f === Core.get_binding_type
            return infer_get_binding_type(fr, args)
        end
        # module-global reads: builtin_tfunction(sv=nothing) cannot consult
        # bindings; fold here (the abstract_eval_globalref/getglobal port)
        if (f === getglobal && 3 <= length(args) <= 4) ||
           (f === getfield && length(args) == 3)
            ml = args[2]; sl = args[3]
            if ml isa CC.Const && ml.val isa Module && sl isa CC.Const && sl.val isa Symbol
                rte = global_rte(st, ml.val, sl.val)
                eff = rte.effects
                exct = rte.exct
                if length(args) == 4
                    # the memory-order argument may be invalid (stock's
                    # global_order_exct merge)
                    goe = try
                        CC.global_order_exct(args[4], #=loading=#true, #=storing=#false)
                    catch
                        Any
                    end
                    if goe !== Union{}
                        eff = CC.Effects(eff; nothrow = false)
                        exct = exct === Any ? Any : CC.tmerge(CC.fallback_lattice, exct, goe)
                    end
                end
                return UResult(rte.rt, eff, exct)
            end
        elseif f === getglobal && !(2 <= length(args) <= 4)
            return UResult(Union{}, CC.EFFECTS_THROWS, ArgumentError)
        end
        argl = args[2:end]
        rt = try
            CC.builtin_tfunction(st.cfg.interp, f, argl, nothing)
        catch
            Any
        end
        eff = builtin_call_effects(f, argl, rt)
        return UResult(rt, eff, eff.nothrow ? Union{} : builtin_call_exct(f, argl, rt))
    end
    if f === Core.UnionAll
        return infer_unionall(fr, args)
    end
    if f !== nothing && is_return_type_f(f)
        r = infer_return_type_call(fr, args)
        r === nothing || return r
        # stock's model: never descend into return_type's reflection body;
        # `nortcall=false` keeps callers out of concrete evaluation (a fold
        # there would re-enter inference — the RT_CALL_EFFECTS rule)
        return UResult(Type, CC.Effects(CC.EFFECTS_THROWS; nortcall = false))
    end
    # opaque closures: a PartialOpaque callee devirtualizes to its source
    # method (the abstract_call_opaque_closure port); a widened OpaqueClosure
    # callee still knows its declared return type (stock's hasintersect
    # fallback in abstract_call_unknown)
    if ftl isa CC.PartialOpaque
        return infer_opaque_call(fr, ftl, args)
    end
    let wft = f === nothing ? CC.widenconst(ftl) :
              (f isa Core.OpaqueClosure ? typeof(f) : nothing)
        if wft isa Type && wft !== Any && Base.hasintersect(wft, Core.OpaqueClosure)
            uft = Base.unwrap_unionall(wft)
            if uft isa DataType && uft.name === Core.OpaqueClosure.body.body.name &&
               length(uft.parameters) >= 2
                p2 = uft.parameters[2]
                rt = try
                    Base.rewrap_unionall(p2 isa TypeVar ? p2.ub : p2, wft)
                catch
                    Any
                end
                return UResult(rt isa Type ? rt : Any, CC.Effects())
            end
            return UResult(Any, CC.Effects())
        end
    end
    # union splitting (the abstract_call_gf_by_type port): small unions in
    # argument position dispatch per element and join — `<(::Union{Int32,
    # Int64}, 0)` must not fall into an abstract-signature match
    let r = maybe_union_split(fr, args)
        r === nothing || return r::UResult
    end
    # type callees (constructors) dispatch through Type{T}, not DataType
    ft = f === nothing ? CC.widenconst(ftl) : (f isa Type ? Type{f} : typeof(f))
    ft === Any && return UResult(Any, CC.Effects())
    ft === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)
    argts = Vector{Any}(undef, length(args) - 1)
    for i in 2:length(args)
        a = args[i]
        if CC.isvarargtype(a)
            i == length(args) || return UResult(Any, CC.Effects())  # malformed
            argts[i - 1] = a
        else
            t = CC.widenconst(a)
            t === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)  # unreachable call
            argts[i - 1] = t
        end
    end
    sig = try
        Tuple{ft, argts...}
    catch
        return UResult(Any, CC.Effects())
    end
    result = lookup_call_matches(st, sig)
    result === nothing && return UResult(Any, CC.Effects())
    matches = result.matches
    isempty(matches) && return UResult(Union{}, CC.EFFECTS_THROWS, MethodError)
    rt = nothing
    fx = CC.EFFECTS_TOTAL
    exct = Union{}
    fully = true
    for match in matches
        r = infer_method(fr, match::Core.MethodMatch, args)
        rt = ⊔(st, rt, r.rt)
        fx = CC.merge_effects(fx, r.effects)
        exct = exct === Any ? Any : CC.tmerge(CC.fallback_lattice, exct, r.exct)
        fully &= (match::Core.MethodMatch).fully_covers
    end
    if !fully || result.ambig
        # a MethodError with a non-covered or ambiguous signature remains
        fx = CC.Effects(fx; nothrow = false)
        exct = exct === Any ? Any : CC.tmerge(CC.fallback_lattice, exct, MethodError)
    end
    return UResult(rt === nothing ? Union{} : rt, fx, exct)
end

"""Split top-level Union argument types (bounded by max_union_splitting
signature combinations) into separate `infer_call`s and join the results.
Returns nothing when no split applies. Split elements are strictly narrower
non-Union types, so the recursion terminates."""
function maybe_union_split(fr::Frame, args::Vector{Any})
    total = 1
    splitat = 0
    for i in 1:length(args)
        a = args[i]
        a isa Union || continue
        total *= length(CC.uniontypes(a))
        splitat == 0 && (splitat = i)
    end
    (splitat == 0 || total < 2) && return nothing
    total > CC.InferenceParams().max_union_splitting && return nothing
    st = fr.st
    rt = nothing
    fx = CC.EFFECTS_TOTAL
    exct = Union{}
    for elt in CC.uniontypes(args[splitat])
        sub = copy(args)
        sub[splitat] = elt
        r = infer_call(fr, sub)   # recurses to split any further union args
        fx = CC.merge_effects(fx, r.effects)
        exct = exct === Any ? Any : CC.tmerge(CC.fallback_lattice, exct, r.exct)
        r.rt === Union{} && continue   # per-element guaranteed throw
        rt = ⊔(st, rt, r.rt)
    end
    rt === nothing && return UResult(Union{}, CC.Effects(fx; nothrow = false), exct)
    return UResult(rt, fx, exct)
end

# ---------------------------------------------------------------------------
# Core.Compiler.return_type (the return_type_tfunc port)
# ---------------------------------------------------------------------------

function is_return_type_f(@nospecialize(f))
    f === Core.Compiler.return_type && return true
    f === CC.return_type && return true
    isdefined(Base, :_return_type) && f === Base._return_type && return true
    return false
end

"The exactly-known type a Type-shaped native lattice pins, or nothing.
Covers `Type{T}` and this nightly's `TypeEgal{T}`/`TypeEq{T}` widenings."
function exact_type_param(@nospecialize(w))
    if CC.isType(w) || (isdefined(CC, :isTypeEgal) && CC.isTypeEgal(w)) ||
       (isdefined(CC, :isTypeEq) && CC.isTypeEq(w))
        p = w.parameters[1]
        (p isa Type && !CC.has_free_typevars(p)) && return p
    end
    return nothing
end

"""Fold `return_type(f, tt)` / `return_type(tt)` when the signature is known.
The runtime call *is* stock inference, so delegating the fold to
`Core.Compiler.return_type` reproduces the runtime answer exactly (stock's
own model of this call carries the same disclaimer)."""
function infer_return_type_call(fr::Frame, args::Vector{Any})
    # args[1] is return_type itself: `return_type(f, tt)` arrives as length-3
    # args, `return_type(tt)` as length-2
    (2 <= length(args) <= 3) || return nothing
    any(a -> CC.isvarargtype(a), args) && return nothing
    tt = args[end]
    local ttv
    if tt isa CC.Const
        ttv = tt.val
    else
        w = CC.widenconst(tt)
        ttv = exact_type_param(w)
        ttv === nothing && return nothing
    end
    (ttv isa DataType && ttv <: Tuple) || return nothing
    local sig
    if length(args) == 3
        aftl = args[2]
        aft = CC.singleton_type(aftl)
        aft === nothing && aftl isa CC.Const && (aft = (aftl::CC.Const).val)
        local ftt
        if aft !== nothing
            ftt = aft isa Type ? Type{aft} : typeof(aft)
        else
            w = CC.widenconst(aftl)
            p = exact_type_param(w)
            if p !== nothing
                ftt = Type{p}
            elseif isconcretetype(w) && !(w <: Core.Builtin)
                ftt = w
            else
                return nothing
            end
        end
        sig = try
            Tuple{ftt, ttv.parameters...}
        catch
            return nothing
        end
    else
        sig = ttv
    end
    st = fr.st
    let col = st.edges
        if col !== nothing
            # the folded answer bakes the method set for `sig` into a Const:
            # record it (world-clamped) so redefinitions invalidate the body
            result = try
                CC.findall(sig, CC.InternalMethodTable(st.cfg.world); limit = -1)
            catch
                nothing
            end
            result === nothing && return nothing   # unboundable: skip the fold
            record_call!(col, sig, result)
            trace!(col, (0x1, sig, result, -1))
        end
    end
    rt = try
        st.edges === nothing ? Core.Compiler.return_type(sig) :
                               Core.Compiler.return_type(sig, st.cfg.world)
    catch
        return nothing
    end
    # the folded answer depends on the oracle's transitive view of the method
    # tables, which the match edge alone cannot revalidate cross-request
    trace!(st.edges, (0x4, sig, rt))
    # `nortcall = false`: a `return_type` call must never become concrete-eval
    # eligible in its callers (that would re-enter inference at runtime); the
    # fold itself already happened, so everything else is total (stock's
    # RT_CALL_EFFECTS)
    return UResult(CC.Const(rt), CC.Effects(CC.EFFECTS_TOTAL; nortcall = false), Union{})
end

# ---------------------------------------------------------------------------
# Core._apply_iterate (the abstract_apply port)
# ---------------------------------------------------------------------------

"""
    container_elements(x) -> (elems::Vector{Any}, exact::Bool) | nothing

The `precise_container_type` port: element lattices of an iterated argument.
`exact` means the runtime performs no user `iterate` calls (tuple-shaped
containers). The last element may be a `Vararg`. `nothing` = unknown shape.
"""
function container_elements(fr::Frame, @nospecialize(x))
    if x isa CC.PartialStruct
        widet = Base.unwrap_unionall(x.typ)
        if widet isa DataType &&
           (widet.name === Tuple.name || widet.name === CC._NAMEDTUPLE_NAME)
            return (Any[fl for fl in x.fields], true)
        end
    end
    if x isa CC.Const
        v = x.val
        if v isa Core.SimpleVector || v isa Tuple
            return (Any[CC.Const(v[i]) for i in 1:length(v)], true)
        elseif v isa NamedTuple
            return (Any[CC.Const(getfield(v, i)) for i in 1:nfields(v)], true)
        end
    end
    tti0 = CC.widenconst(x)
    tti = Base.unwrap_unionall(tti0)
    if tti isa DataType && tti.name === CC._NAMEDTUPLE_NAME
        # NamedTuple iterates as its Tuple parameter
        tp = tti.parameters[2]
        tp isa Type || return nothing
        tti0 = Base.rewrap_unionall(tp, tti0)
        tti = Base.unwrap_unionall(tti0)
    end
    if tti isa Union
        utis = CC.uniontypes(tti)
        elts = nothing
        for t in utis
            (t isa DataType && t <: Tuple && CC.isknownlength(t)) || return nothing
            ps = Any[Base.rewrap_unionall(p, tti0) for p in t.parameters]
            if elts === nothing
                elts = ps
            else
                length(ps) == length(elts) || return nothing
                for j in 1:length(ps)
                    elts[j] = CC.tmerge(CC.fallback_lattice, elts[j], ps[j])
                end
            end
        end
        return elts === nothing ? nothing : (elts, true)
    end
    if tti0 <: Tuple
        if tti0 isa DataType
            return (Any[p for p in tti0.parameters], true)
        elseif !(tti isa DataType)
            return (Any[Vararg{Any}], true)
        else
            len = length(tti.parameters)
            elts = Any[Base.rewrap_unionall(p, tti0) for p in tti.parameters]
            if len > 0 && CC.isvarargtype(tti.parameters[len])
                elts[len] = tti.parameters[len]   # keep the Vararg tail as-is
            end
            return (elts, true)
        end
    elseif tti0 === Core.SimpleVector
        return (Any[Vararg{Any}], false)
    elseif tti0 <: Array || tti0 <: GenericMemory
        et = try
            eltype(tti0)
        catch
            Any
        end
        return (Any[Vararg{et === Union{} ? Any : et}], false)
    end
    return nothing     # unknown iterable: degrade to Vararg{Any} + unknown effects
end

"""The abstract_iteration port: enumerate an iterated argument's element
lattices by running the `iterate` protocol abstractly. Phase 1 unrolls finite
iterators precisely (guaranteed-present elements only); phase 2 folds the
remainder into a `Vararg` tail at the widened state fixpoint. Returns
`(elems, effects, exct)` — the iterate calls' joined effects (stock merges
them into the apply's, abstract_apply-style); `Any[Union{}]` elements mean
iteration provably throws or cannot terminate."""
function iterate_elements(fr::Frame, @nospecialize(x))
    lat = CC.fallback_lattice
    itf = CC.Const(Base.iterate)
    fx = CC.EFFECTS_TOTAL
    exct = Union{}
    join!(r::UResult) = begin
        fx = CC.merge_effects(fx, r.effects)
        exct = exct === Any ? Any : CC.tmerge(lat, exct, r.exct)
        r
    end
    r = join!(infer_call(fr, Any[itf, x]))
    sod = widenucond(r.rt)               # state-or-done, precise
    sodw = CC.widenconst(sod)
    sodw === Union{} && return (Any[Union{}], fx, exct)   # not an iterator: throws
    elems = Any[]
    statetype = Union{}
    # phase 1: precise unroll while termination is impossible
    while true
        sodw === Nothing && return (elems, fx, exct)   # provably exhausted (exact)
        (Nothing <: sodw || length(elems) >= 32) && break
        (sodw isa DataType && sodw <: Tuple && !CC.isvatuple(sodw) &&
         length(sodw.parameters) == 2) || break
        nst, vt = try
            (CC.getfield_tfunc(lat, sod, CC.Const(2)),
             CC.getfield_tfunc(lat, sod, CC.Const(1)))
        catch
            break
        end
        # no new state information: the iterator cannot be finite (stock's
        # infinite-iteration rule — the apply never completes)
        CC.:⊑(lat, nst, statetype) && return (Any[Union{}], fx, exct)
        push!(elems, vt)
        statetype = nst
        r = join!(infer_call(fr, Any[itf, x, statetype]))
        sod = widenucond(r.rt)
        sodw = CC.widenconst(sod)
    end
    # phase 2: widened tail to a state fixpoint
    valtype = Union{}
    statew = Union{}
    may_have_terminated = Nothing <: sodw
    guard = 0
    while valtype !== Any && (guard += 1) < 100
        nounion = try
            typeintersect(sodw, Tuple{Any,Any})
        catch
            Any
        end
        if nounion !== Union{} && !(nounion isa DataType)
            valtype = Any
            break
        end
        if nounion === Union{} || (nounion.parameters[1] <: valtype &&
                                   nounion.parameters[2] <: statew)
            # fixpoint (or the iterator failed / gave an invalid answer)
            if !CC.hasintersect(sodw, Nothing)
                # ...and cannot terminate during this loop
                may_have_terminated || return (Any[Union{}], fx, exct)
                valtype = Union{}   # only completes if it ended before here
            end
            break
        end
        valtype = CC.tmerge(lat, valtype, nounion.parameters[1])
        statew = CC.tmerge(lat, statew, nounion.parameters[2])
        r = join!(infer_call(fr, Any[itf, x, statew]))
        sod = widenucond(r.rt)
        sodw = CC.widenconst(sod)
    end
    valtype === Union{} || push!(elems, Vararg{CC.widenconst(valtype)})
    return (elems, fx, exct)
end

"Core._apply_iterate(iterate, f, iters...): flatten precisely when possible.
`sid != 0` names the apply's own statement: a precise iterate-protocol
unroll is then recorded in `ir.meta[:apply_iter_unroll]` (stmt id =>
[(operand index, element count)...]) for `fold_apply_iterates!` to
materialize (the stock ApplyCallInfo channel)."
function infer_apply(fr::Frame, args::Vector{Any}; sid::Int32 = Int32(0))::UResult
    length(args) >= 3 || return UResult(Any, CC.Effects())
    fl = args[3]
    fl === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)
    flat = Any[fl]
    exact = true
    precise = true
    iterfx = CC.EFFECTS_TOTAL      # the iterate protocol's own effects
    iterexct = Union{}
    unrolls = Tuple{Int,Int}[]     # (operand idx, element count) per iterate-container
    unrollok = true
    for i in 4:length(args)
        a = args[i]
        if CC.isvarargtype(a)
            precise = false
            break
        end
        ce = container_elements(fr, a)
        if ce === nothing
            # not a tuple-shaped container: run the iterate protocol
            # abstractly; its calls' effects merge into the apply's
            # (stock abstract_apply)
            elems, ifx, iexct = iterate_elements(fr, a)
            append!(flat, elems)
            iterfx = CC.merge_effects(iterfx, ifx)
            iterexct = iterexct === Any ? Any :
                       CC.tmerge(CC.fallback_lattice, iterexct, iexct)
            exact = false
            if unrollok && !Base.any(e -> CC.isvarargtype(e), elems) &&
               Base.all(e -> e !== Union{}, elems)
                # provably-exhausted fixed unroll: rewrite-eligible
                push!(unrolls, (i, length(elems)))
            else
                unrollok = false
            end
            continue
        end
        append!(flat, ce[1])
        exact &= ce[2]
    end
    if sid != 0 && precise && unrollok && !isempty(unrolls)
        ch = get!(() -> Dict{Int32,Vector{Tuple{Int,Int}}}(),
                  fr.ir.meta, :apply_iter_unroll)::Dict{Int32,Vector{Tuple{Int,Int}}}
        ch[sid] = unrolls
    end
    if !precise
        flat = Any[fl, Vararg{Any}]
        exact = false
        iterfx = CC.Effects()      # unknowable iteration
        iterexct = Any
    end
    if precise
        # fold a mid-list Vararg into a merged tail (stock's truncation rule)
        for k in 2:length(flat)
            if CC.isvarargtype(flat[k]) && k < length(flat)
                tail = CC.tuple_tail_elem(CC.fallback_lattice, CC.unwrapva(flat[k]),
                                          Any[flat[j] for j in (k + 1):length(flat)])
                resize!(flat, k)
                flat[k] = tail
                break
            end
        end
    end
    r = infer_call(fr, flat)
    # flattened positions do not map to caller operands: widen InterConditionals
    r.rt isa UInterCond && (r = UResult(Bool, r.effects, r.exct))
    exact && return r
    # non-tuple containers: the iterate calls' effects taint the apply
    return UResult(r.rt, CC.merge_effects(r.effects, iterfx),
                   iterexct === Any ? Any :
                   CC.tmerge(CC.fallback_lattice, r.exct, iterexct))
end

# ---------------------------------------------------------------------------
# invoke (the abstract_invoke port)
# ---------------------------------------------------------------------------

"Driver-mode record of a result read straight off a CodeInstance: the CI
edge plus its own world bounds (the stock InvokeCICallInfo shape)."
function record_ci_read!(st::UInferState, ci::Core.CodeInstance)
    col = st.edges
    col === nothing && return nothing
    clamp_world!(col, ci.min_world, ci.max_world)
    record_invoke!(col, nothing, ci)
    trace!(col, (0x3, ci))
    return nothing
end

"Full-width effects off a CodeInstance's stock-encoded ipo purity bits."
ci_result(ci::Core.CodeInstance) =
    UResult(ci.rettype, CC.decode_effects(ci.ipo_purity_bits),
            isdefined(ci, :exctype) ? ci.exctype : Any)

"K\"invoke\": the first operand is a CONST CodeInstance/MethodInstance."
function infer_invoke_target(fr::Frame, @nospecialize(tl), args::Vector{Any})::UResult
    target = tl isa CC.Const ? tl.val : CC.singleton_type(tl)
    if target isa Core.CodeInstance
        record_ci_read!(fr.st, target)
        return ci_result(target)
    elseif target isa Core.MethodInstance && target.def isa Method
        match = Core.MethodMatch(target.specTypes, target.sparam_vals,
                                 target.def::Method, true)
        let col = fr.st.edges
            if col !== nothing
                record_invoke!(col, target.specTypes, target)
                memo_poison!(col)   # mi-invoke facts are not trace-revalidatable
            end
        end
        return try
            infer_method(fr, match, args)
        catch
            UResult(Any, CC.Effects())
        end
    end
    isempty(args) && return UResult(Any, CC.Effects())
    return infer_call(fr, args)
end

widen_intercond(r::UResult) = r.rt isa UInterCond ? UResult(Bool, r.effects, r.exct) : r

"`Core.invoke(f, types_or_method_or_ci, args...)` as a call (argument
positions shift under the builtin: InterConditionals widen)."
function infer_invoke(fr::Frame, args::Vector{Any})::UResult
    length(args) >= 3 || return UResult(Union{}, CC.EFFECTS_THROWS)
    any(a -> CC.isvarargtype(a), args) && return UResult(Any, CC.Effects())
    ftl = args[2]
    ft = CC.widenconst(ftl)
    ft === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)
    types = args[3]
    callargs = Any[ftl]
    append!(callargs, args[4:end])
    argts = Any[CC.widenconst(a) for a in args[4:end]]
    if types isa CC.Const
        v = types.val
        if v isa Core.CodeInstance
            record_ci_read!(fr.st, v)
            return ci_result(v)
        elseif v isa Method
            argtype = Tuple{ft, argts...}
            return widen_intercond(invoke_match(fr, v, argtype, argtype, callargs))
        end
    end
    T, isexact = try
        CC.instanceof_tfunc(types, false)
    catch
        (Any, false)
    end
    isexact || return UResult(Any, CC.Effects())
    T === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS)
    unwrapped = Base.unwrap_unionall(T)
    (unwrapped isa DataType && unwrapped.name === Tuple.name) ||
        return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)   # TypeError
    Base.isdispatchelem(ft) || return UResult(Any, CC.Effects())
    argtype0 = Tuple{argts...}
    nargtype = typeintersect(T, argtype0)
    nargtype === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)
    nargtype isa DataType || return UResult(Any, CC.Effects())
    lookupsig = try
        Base.rewrap_unionall(Tuple{ft, unwrapped.parameters...}, T)
    catch
        return UResult(Any, CC.Effects())
    end
    matched, sup_worlds = try
        CC.findsup(lookupsig, CC.InternalMethodTable(fr.st.cfg.world))
    catch
        (nothing, nothing)
    end
    matched === nothing && return UResult(Any, CC.Effects())
    let col = fr.st.edges
        col === nothing || clamp_world!(col, sup_worlds)
    end
    return widen_intercond(invoke_match(fr, matched.method,
                                        Tuple{ft, nargtype.parameters...},
                                        Tuple{ft, argts...}, callargs))
end

function invoke_match(fr::Frame, method::Method, @nospecialize(nargtype),
                      @nospecialize(argtype), callargs::Vector{Any})::UResult
    local nt
    r = try
        nt = typeintersect(nargtype, method.sig)
        nt === Union{} && return UResult(Union{}, CC.EFFECTS_THROWS, TypeError)
        tienv = ccall(:jl_type_intersection_with_env, Any, (Any, Any),
                      nt, method.sig)::Core.SimpleVector
        ti = tienv[1]
        env = tienv[2]::Core.SimpleVector
        match = Core.MethodMatch(ti, env, method, argtype <: method.sig)
        let col = fr.st.edges
            # the stock invoke-edge shape: (invokesig, callee MethodInstance)
            if col !== nothing
                record_invoke!(col, argtype, CC.specialize_method(match))
                memo_poison!(col)   # invoke-edge facts are not trace-revalidatable
            end
        end
        infer_method(fr, match, callargs)
    catch
        return UResult(Any, CC.Effects())
    end
    # the runtime checks args against `types`: not provably passing → may throw
    passes = try
        argtype <: nt
    catch
        false
    end
    passes && return r
    exct = r.exct === Any ? Any : CC.tmerge(CC.fallback_lattice, r.exct, TypeError)
    return UResult(r.rt, CC.Effects(r.effects; nothrow = false), exct)
end

# ---------------------------------------------------------------------------
# Method frames: memoization, const-seeding, generated expansion
# ---------------------------------------------------------------------------

"Uncompressed or generator-expanded source for a method instance. With a
collector attached, staged expansions clamp the collector to the expansion's
world bounds (a generator's output is only valid for the worlds it reports)."
function method_src(m::Method, mi::Core.MethodInstance, world::UInt,
                    col::Union{Nothing,UEdges} = nothing)
    if isdefined(m, :generator)
        src = try
            CC.get_staged(mi, world)     # nothing when expansion fails
        catch
            nothing
        end
        if src !== nothing && col !== nothing
            clamp_world!(col, src.min_world, src.max_world)
            memo_poison!(col)   # staged-expansion windows are not trace-revalidatable
        end
        return src
    end
    return try
        Base.uncompressed_ir(m)
    catch
        nothing
    end
end

"Static-parameter indices read in statement position by lowered source (the
entry converter aliases such reads away, losing their maybe-undef throw)."
function sparam_statement_reads(ci::Core.CodeInfo)
    out = Int[]
    for st in ci.code
        if st isa Expr && st.head === :static_parameter
            n = st.args[1]
            n isa Int && push!(out, n)
        end
    end
    return out
end

function convert_src(srcci::Core.CodeInfo, m::Method, mi::Core.MethodInstance)
    try
        ir = codeinfo_to_ir(srcci; nargs = Int(m.nargs), name = m.name)
        ir.sptypes = Any[t for t in mi.sparam_vals]
        ir.meta[:sptypes_lat] = sptypes_lattice(mi)
        und = sptypes_undef(mi)
        und === nothing || (ir.meta[:sptypes_undef] = und)
        let reads = sparam_statement_reads(srcci)
            isempty(reads) || (ir.meta[:sparam_reads] = reads)
        end
        ir.meta[:mi] = mi     # sp_type_rewrap context for foreigncall rts
        ir.meta[:propagate_inbounds] = srcci.propagate_inbounds
        return ir
    catch e
        e isa UnsupportedIR || rethrow()
        return nothing
    end
end

"Per-sparam maybe-undefined-at-runtime bits (stock `sptypes[i].undef`)."
function sptypes_undef(mi::Core.MethodInstance)
    try
        return Bool[vs.undef for vs in CC.sptypes_from_meth_instance(mi)]
    catch
        return nothing
    end
end

"""Static-parameter lattice elements for inference. `mi.sparam_vals` entries
are not always plain values (constrained TypeVars arrive as `svec(tv, flag)`
markers); reuse stock's decoding."""
function sptypes_lattice(mi::Core.MethodInstance)
    try
        return Any[vs.typ for vs in CC.sptypes_from_meth_instance(mi)]
    catch
        return Any[raw_sparam_lattice(sp) for sp in mi.sparam_vals]
    end
end

raw_sparam_lattice(@nospecialize(sp)) =
    (sp isa Core.SimpleVector || sp isa TypeVar) ? Any : CC.Const(sp)

"Drop InterConditionals whose slot has no positional caller operand (the
vararg-packed parameter of an isva method, or out-of-range slots)."
sanitize_intercond(m::Method, @nospecialize(rt)) =
    (rt isa UInterCond &&
     (m.isva ? rt.slot >= Int(m.nargs) : rt.slot > Int(m.nargs))) ? Bool : rt

"A method's `@assume_effects` bits (all-false when undecodable)."
function effect_override(m::Method)
    return try
        CC.decode_effects_override(m.purity)
    catch
        Base.EffectsOverride()
    end
end

"""Apply a method's declared `@assume_effects` overrides — the stock
`adjust_effects(effects, def::Method)`, all 11 bits: e.g. `==(::Type, ::Type)`
is a total-declared foreigncall, and concrete evaluation keys off the
resulting effects."""
function apply_effects_override(m::Method, fx::CC.Effects)
    return try
        CC.adjust_effects(fx, m)
    catch
        fx
    end
end

"""Key-encode one argument lattice element for the const memo, or nothing
when the element carries information the key cannot capture. The key MUST
pin the seed exactly: the memoized result was computed at this precise
lattice element, so a key that widens (e.g. a PartialStruct keyed by its
widenconst) would replay one caller's field-precise answer for every other
caller of the same widened shape — a miscompile, not just imprecision
(`(x * 1.0) * 10` once folded the `10` to the first multiply's `1.0` through
exactly that collision in the promote/indexed_iterate chain)."""
function const_key_elem(@nospecialize(a))
    if a isa CC.Const
        v = a.val
        # mutable payloads key by IDENTITY: `:consistent`-cy (the concrete-eval
        # license) is an egal contract — two isequal-but-not-egal Dicts must
        # not share a memoized fold (objectid of immutables is content-based,
        # so the extra component is inert for them)
        return (0x0, v, ismutable(v) ? objectid(v) : nothing)
    end
    if a isa CC.PartialStruct
        enc = Vector{Any}(undef, length(a.fields))
        for (i, f) in enumerate(a.fields)
            e = const_key_elem(f)
            e === nothing && return nothing
            enc[i] = e
        end
        return (0x2, CC.widenconst(a), CC._getundefs(a), (enc...,))
    end
    if a isa CC.PartialOpaque
        # identity-keyed pieces (source Method, parent mi) plus the encoded
        # env element pin the seed exactly (the OC-capturing-OC corpus)
        enc = const_key_elem(a.env)
        enc === nothing && return nothing
        return (0x3, a.typ, a.source, a.parent, enc)
    end
    a isa Type && return (0x1, a)
    CC.isvarargtype(a) && return (0x1, a)
    # UCond/UInterCond/other extended elements: context-dependent seeds the
    # key cannot soundly identify — skip memoization for such frames
    return nothing
end

"Memo key for a const-seeded frame (`UConstKey`; see its docstring for the
dispatch-uniformity rationale), or nothing."
function const_key(mi::Core.MethodInstance, args::Vector{Any})
    parts = Vector{Any}(undef, length(args) + 1)
    parts[1] = mi
    h = objectid(mi)
    for (i, a) in enumerate(args)
        e = const_key_elem(a)
        e === nothing && return nothing
        parts[i + 1] = e
        # objectid is egal-consistent (structural over the immutable encoding
        # tuples) and total — no user `hash` method runs, nothing can throw
        h = hash(objectid(e), h)
    end
    return UConstKey(h, parts)
end

"""The concrete_eval_call port: when every argument is a Const
and the callee's own (widened, context-free) inferred effects satisfy stock's
`_concrete_eval_eligible` criteria — `is_foldable(effects, check_rtcall=true)`,
plus nothrow under `--check-bounds=no` — evaluate the call for real and
return Const of the result — both a precision and a speed lever (the abstract
const frame never runs). `nothrow` is NOT required in general: a foldable
callee that throws proves the call sites' rt is Bottom (stock's
ConcreteResult semantics). Returns nothing when ineligible.

Overlay accounting: the unified pipeline only ever consults the internal
method table (`lookup_call_matches`/`findsup` — no overlay tables), which is
stock's `is_nonoverlayed(interp)` fast-path condition, so the per-effects
nonoverlayed/consistent_overlay checks are not required here."""
function concrete_eval(fr::Frame, match::Core.MethodMatch, args::Vector{Any},
                       @nospecialize(ck))
    st = fr.st
    f = CC.singleton_type(args[1])
    f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
    f === nothing && return nothing
    argvals = Vector{Any}(undef, length(args) - 1)
    for i in 2:length(args)
        a = args[i]
        # any Const qualifies (stock's is_all_const_arg): the callee's proven
        # :consistent-cy is exactly the license to evaluate over the argument
        # objects, mutable ones included
        a isa CC.Const || return nothing
        argvals[i - 1] = (a::CC.Const).val
    end
    # the callee's effects come from its widened frame (memoized; computed
    # once per mi). Foldable implies a clean, converged, cutoff-free frame:
    # stale reads and resource cutoffs always pessimize the effects.
    wr = infer_method(fr, match, Any[])
    CC.is_foldable(wr.effects, #=check_rtcall=#true) || return nothing
    if CC.inbounds_option() === :off && !CC.is_nothrow(wr.effects)
        # under --check-bounds=no the callee may be compiled without the
        # bounds checks its :consistent-cy assumed: require nothrow
        return nothing
    end
    local v
    try
        v = Core._call_in_world_total(st.cfg.world, f, argvals...)
    catch
        # the evaluation threw: by :consistent-cy this happens at runtime too.
        # Stock keeps the ABSTRACT result's exception type here (:consistent-cy
        # does not mandate the exception type, so the concrete throw only
        # proves rt = Bottom — which the abstract const frame derives anyway):
        # decline, and let the const-seeded frame run.
        return nothing
    end
    r = UResult(CC.Const(v), CC.EFFECTS_TOTAL, Union{})
    ck === nothing || (st.constcache[ck] = r)
    return r
end

"""Would const-seeding add information over the widened signature? The
callee position counts too: a `Const` Type callee pins the instance where
its `Type{T}` widening does not (this nightly's #61323 semantics — a
constructor body's `fieldtype(self, ...)` only folds on the Const)."""
function const_args_profitable(args::Vector{Any})
    for i in 1:length(args)
        a = args[i]
        if a isa CC.Const
            Base.issingletontype(typeof(a.val)) || return true
        elseif a isa CC.PartialStruct || a isa CC.PartialOpaque
            return true
        end
    end
    return false
end

"""SCC membership/convergence bookkeeping for the outermost cycle root: fold
this pass's member results (the `cycle_scratch` memo) into the accumulated
`scc_prev` table with tmerge (bounded ascent), returning whether any member
appeared or moved. `scc_prev` also reseeds nested cycle roots on the next
pass, making the outer reruns a joint Gauss-Seidel iteration over the SCC."""
function scc_update!(st::UInferState, escalate::Bool)
    changed = false
    for (k, v) in st.cycle_scratch
        r = v isa UResult ? v : (v[1])::UResult   # unwrap epoch-tagged Bottoms
        old = get(st.scc_prev, k, nothing)
        if old === nothing
            st.scc_prev[k] = r
            changed = true
            continue
        end
        old = old::UResult
        merged = umerge(old.rt, r.rt)
        escalate && (merged = CC.widenconst(widenucond(merged)))
        fx = CC.merge_effects(old.effects, r.effects)
        exct = old.exct === Any ? Any : CC.tmerge(CC.fallback_lattice, old.exct, r.exct)
        if !ulat_eq(merged, old.rt) || fx != old.effects || !lat_eq(exct, old.exct)
            st.scc_prev[k] = UResult(merged, fx, exct)
            changed = true
        end
    end
    return changed
end

"""Runtime CodeInstance cache serving (wave 9, the durable memo home): stock
`typeinf_edge`'s cache-hit path (`return_cached_result`). A callee mi with a
current native-cache CodeInstance serves rettype/exct/effects straight from
the CI — the cost of a body inferred once (this session, a previous root, the
stock fallback, or a baked sysimage/pkgimage entry) is one cache read, never
a re-walk of its callee tree. Soundness mirrors stock exactly: the consuming
body records the CI as an edge (`store_backedges` registers the backedge, so
any transitive invalidation of the callee decays this body too) and clamps
its world window to the CI's. Precision equals stock's cache hits:
`cached_return_type` decodes Const/PartialStruct/PartialOpaque/
InterConditional from `rettype_const` (the driver publishes the same
encodings — see `driver_infer`'s tail). Only UNBOUNDED entries serve
(`max_world == typemax`): the driver never publishes bounded results, so
consuming a bounded fact would forfeit the whole request at finish.
Disable with `CI_SERVE_ENABLED[] = false` (triage switch)."""
const CI_SERVE_ENABLED = Base.RefValue(true)

function ci_cache_serve(st::UInferState, mi::Core.MethodInstance)
    CI_SERVE_ENABLED[] || return nothing
    interp = st.cfg.interp
    ci = get(CC.code_cache(interp), mi, nothing)
    # an InferenceResult is an IN-PROGRESS overlay entry (its .ci may be an
    # unfilled engine reservation): never serve those
    ci isa Core.CodeInstance || return nothing
    ci.max_world == typemax(UInt) || return nothing
    rt = CC.cached_return_type(ci)
    if rt isa CC.InterConditional
        rt = UInterCond(rt.slot, rt.thentype, rt.elsetype)
        let m = mi.def
            m isa Method && (rt = sanitize_intercond(m, rt))
        end
    elseif rt isa CC.InterMustAlias
        rt = CC.widenmustalias(rt)
    end
    effects = CC.decode_effects(ci.ipo_purity_bits)
    exct = ci.exctype
    col = st.edges
    if col isa UEdges
        clamp_world!(col, ci.min_world, ci.max_world) || return nothing
        record_invoke!(col, nothing, ci)
        trace!(col, (0x3, ci))
        # the single 0x3 fact is this frame's whole window: span it so
        # per-request cache hits stay fact-complete for enclosing frames
        col.spans[mi] = (length(col.trace), length(col.trace))
    end
    DRIVER_PHASES.ci_serves += 1
    return UResult(rt, effects, exct)
end

function infer_method(fr::Frame, match::Core.MethodMatch, args::Vector{Any})::UResult
    st = fr.st
    m = match.method
    mi = CC.specialize_method(match)
    if haskey(st.active, mi)
        st.stats.cycles += 1
        push!(st.cycle_hit, mi)
        # an outstanding stale read: frames whose window saw it and lie below
        # the target are cycle-tainted until the target completes
        st.stale_depth = min(st.stale_depth, st.active[mi])
        st.stale_events += 1
        # cycle: the target's current (optimistic, descending) approximation
        # with `terminates` tainted — landing here IS genuine MethodInstance
        # recursion, exactly stock's `is_edge_recursed` criterion (abstract
        # recursion over shrinking signatures creates distinct mi's and never
        # hits the active stack, #48983). The taint is suppressed when the
        # caller frame or the callee method declares :terminates_globally
        # (stock's MethodCallResult override order); the SCC joint fixpoint
        # makes the stale effects consistent at convergence.
        r = get(st.cache, mi, nothing)
        base = r === nothing ? UResult(Union{}, CC.EFFECTS_TOTAL, Union{}) : r::UResult
        eff = base.effects
        if fr.override.terminates_globally
            eff = CC.Effects(eff; terminates = true)
        elseif effect_override(m).terminates_globally
            eff = CC.Effects(eff; terminates = true)
        else
            eff = CC.Effects(eff; terminates = false)
        end
        return UResult(base.rt, eff, base.exct)
    end
    # cross-request memo bookkeeping (driver mode; see driver.jl's memo
    # section): this frame's fact-trace window starts here. `tco` snapshots
    # the world counter — facts queried after a mid-frame counter bump would
    # be certified fresher than the snapshot, so the stored entry always
    # revalidates on first consumption in that case (conservative).
    col = st.edges
    tlo = col === nothing ? 0 : length(col.trace)
    tpo = col === nothing ? 0 : col.poison
    tco = col === nothing ? UInt(0) : Base.get_world_counter()
    # const-seeded frames (interprocedural constant propagation) use their own
    # memo cache keyed by the const-extended signature. Vararg methods qualify
    # too (stock const-props them; method_arglattice builds the precise vararg
    # tuple via tuple_tfunc) — e.g. `_any_tuple(f, false, tt...)` needs the
    # Const(false) seed for `TupleOrBottom`/`promote_op` guards to fold.
    argsfit = !any(a -> CC.isvarargtype(a), args) &&
              (m.isva ? length(args) >= Int(m.nargs) - 1 :
                        Int(m.nargs) == length(args))
    constseeded = argsfit && const_args_profitable(args)
    ck = nothing
    if constseeded
        ck = const_key(mi, args)
        if ck === nothing
            constseeded = false
        else
            r = get(st.constcache, ck, nothing)
            r === nothing || (memo_note_hit!(st, ck); return r::UResult)
            r = get(st.scratch, ck, nothing)
            r === nothing || (memo_poison!(col); return r::UResult)
            r = get(st.cycle_scratch, ck, nothing)
            if r isa UResult
                st.cyscr_hits += 1
                memo_poison!(col)
                return r
            elseif r isa Tuple && r[2] == st.resolutions
                st.cyscr_hits += 1
                memo_poison!(col)
                return r[1]::UResult
            end
            # cross-request memo: a previous driver request's const frame,
            # with its recorded facts replayed into this request's collector
            r = global_memo_lookup(st, ck)
            if r !== nothing
                st.constcache[ck] = r::UResult
                return r::UResult
            end
            # all-Const call of a total callee: evaluate for real instead of
            # running the abstract const frame (the concrete_eval_call port)
            r = concrete_eval(fr, match, args, ck)
            if r !== nothing
                memo_frame_store!(st, ck, r::UResult, tlo, tpo, tco)
                return r::UResult
            end
        end
    end
    if !constseeded
        haskey(st.cache, mi) && (memo_note_hit!(st, mi); return st.cache[mi]::UResult)
        r = get(st.scratch, mi, nothing)
        r === nothing || (memo_poison!(col); return r::UResult)
        r = get(st.cycle_scratch, mi, nothing)
        if r isa UResult
            st.cyscr_hits += 1
            memo_poison!(col)
            return r
        elseif r isa Tuple && r[2] == st.resolutions
            st.cyscr_hits += 1
            memo_poison!(col)
            return r[1]::UResult
        end
        # cross-request memo: a previous driver request's widened frame
        r = global_memo_lookup(st, mi)
        if r !== nothing
            st.cache[mi] = r::UResult
            return r::UResult
        end
        # runtime CodeInstance cache: stock typeinf_edge's cache-hit path
        # (a body with a current CI never re-walks; see ci_cache_serve)
        r = ci_cache_serve(st, mi)
        if r !== nothing
            st.cache[mi] = r::UResult
            return r::UResult
        end
    end
    fbudget = st.cfg.frame_budget
    let cap = TOWER_FRAME_CAP[]
        # driver-grade callee towers (cost model / post-opt effects / EA
        # summaries) run under a hard frame cap: a tower whose candidate
        # roots a budget-busting graph (the print/string family) must not
        # re-walk thousands of frames per optimizer round to price one
        # inlining decision — the helpers refuse the verdict instead when
        # the cap fires (see inline2_cost_uncached)
        0 < cap < fbudget && (fbudget = cap)
    end
    if length(st.active) >= st.cfg.max_depth ||
       st.stats.frames - st.budget_mark >= fbudget
        # resource cutoff: the result is CONTEXT-dependent — callers must not
        # memoize anything computed on top of it (see `tainted` below)
        st.limited += 1
        return native_result(fr, match)
    end
    srcci = method_src(m, mi, st.cfg.world, st.edges)
    srcci === nothing && return native_result(fr, match)
    # a frame is tainted when its subtree (a) hit a resource cutoff, or
    # (b) depends on the stale approximation of a frame STILL active above us
    # (an outstanding stale read at a smaller depth). Such results are valid
    # transiently but must not enter the permanent caches: (a) goes to the
    # per-query scratch, (b) to the cycle scratch, which the cycle root clears
    # per fixpoint pass and flushes on completion
    mydepth = length(st.active) + 1
    lim0 = st.limited
    ev0 = st.stale_events
    h0 = st.cyscr_hits
    tainted_limit() = st.limited > lim0
    # tainted iff MY window saw a stale read whose target is still above us,
    # or consumed a stale-based scratch entry (independent clean subtrees
    # below an active cycle root stay permanently cacheable)
    tainted_cycle() =
        (st.stale_events > ev0 && st.stale_depth < mydepth) || st.cyscr_hits > h0
    function frame_done(flushable::Bool = false)
        # all outstanding stale reads targeted us or frames below us: resolved.
        # If we are the outermost stale target and the SCC's joint fixpoint
        # exited STABLE (our value and every member's per-pass result), the
        # members' last-pass results are jointly consistent with the final
        # values — flush them ALL into the permanent caches together (the
        # stock finish_cycle rule: the whole SCC commits when its outermost
        # frame converges). A genuinely non-converged pass discards instead.
        if st.stale_depth >= mydepth
            st.resolutions += 1
            if st.stale_depth == mydepth && flushable
                for (k, v) in st.cycle_scratch
                    v isa UResult || continue   # epoch-tagged Bottoms: no flush
                    if k isa Core.MethodInstance
                        st.cache[k] = v
                    else
                        st.constcache[k] = v
                    end
                    # jointly-converged SCC members commit with the ROOT's
                    # fact window (a member's own window misses the facts
                    # justifying the root's value it consumed): superset,
                    # sound for span references and the global memo alike
                    memo_frame_store!(st, k, v, tlo, tpo, tco)
                end
            end
            empty!(st.cycle_scratch)
            empty!(st.scc_prev)
            st.stale_depth = typemax(Int)
        end
        return nothing
    end
    if constseeded && length(srcci.code) <= CONSTPROP_SRC_LIMIT
        # caller-precise seed (Const/PartialStruct lattice), own memo cache;
        # recursion protection via `active`
        argl = method_arglattice(m, mi, args)
        argl === nothing && return native_result(fr, match)
        st.active[mi] = mydepth
        hit_cycle = false
        local rc
        try
            src_ir = convert_src(srcci, m, mi)
            src_ir === nothing && return native_result(fr, match)
            delete!(st.cycle_hit, mi)
            rt_const = sanitize_intercond(m, infer_ir!(src_ir, copy(argl); state = st))
            rc = UResult(rt_const,
                         apply_effects_override(m, frame_effects_meta(src_ir)),
                         get(src_ir.meta, :exct, Any))
            hit_cycle = mi in st.cycle_hit
        finally
            delete!(st.active, mi)
        end
        # a recursive const-seeded frame read its own stale approximation; the
        # const result is unsound — fall back to the widened fixpoint below
        # (bounded const-prop recursion loses const precision, keeps soundness)
        if !hit_cycle
            # cycle taint takes priority over limit taint: a stale-dependent
            # result must die with the pass, never enter the per-query scratch
            # (which outlives the cycle's resolution)
            if tainted_cycle()
                # a stale-collapsed Bottom expires as soon as any cycle root
                # resolves (the caches it depends on improve then); non-Bottom
                # entries live until the pass's scratch is cleared
                st.cycle_scratch[ck] = rc.rt === Union{} ? (rc, st.resolutions) : rc
            elseif tainted_limit()
                st.scratch[ck] = rc
            else
                st.constcache[ck] = rc
                memo_frame_store!(st, ck, rc, tlo, tpo, tco)
            end
            frame_done()
            return rc
        end
        # (frame_done(false): members computed against this aborted const frame
        # must not be flushed as converged)
        haskey(st.cache, mi) && (frame_done(false); memo_note_hit!(st, mi);
                                 return st.cache[mi]::UResult)
        let r = ci_cache_serve(st, mi)
            if r !== nothing
                st.cache[mi] = r::UResult
                frame_done(false)
                return r::UResult
            end
        end
    end
    # the memoized path is keyed by mi: it MUST be computed at the
    # mi.specTypes-derived lattice, never at one caller's lattice — a shared
    # cache entry computed from the first caller's (wider or narrower) args
    # would be imprecise or unsound for every other caller of the same mi
    argl = method_arglattice(m, mi, Any[])
    argl === nothing && return native_result(fr, match)
    st.active[mi] = mydepth
    # nested cycle roots reseed from the SCC's last-pass approximation (the
    # joint Gauss-Seidel iteration ascends instead of restarting at ⊥ — a ⊥
    # restart both loses the self-edge contribution and can never converge)
    prevapprox = get(st.scc_prev, mi, nothing)
    st.cache[mi] = prevapprox === nothing ? UResult(Union{}, CC.EFFECTS_TOTAL, Union{}) :
                                            prevapprox::UResult
    ok = false
    converged = false
    deferred = false
    nc0 = st.nonconverged
    try
        # SCC joint fixpoint: the OUTERMOST stale-read target reruns until
        # neither its own value nor any member's per-pass result moves.
        # Each pass recomputes every member frame (unlike stock's suspended
        # frames), so the cap is deliberately tight, with early widenconst
        # escalation forcing convergence on big print/show SCCs.
        for it in 1:12
            delete!(st.cycle_hit, mi)
            # on reruns, cycle members memoized against our previous
            # approximation must be recomputed against the updated one
            it > 1 && empty!(st.cycle_scratch)
            nc0 = st.nonconverged      # snapshot at final-pass start
            src_ir = convert_src(srcci, m, mi)
            if src_ir === nothing
                st.cache[mi] = native_result(fr, match)
                converged = true
                break
            end
            rt = sanitize_intercond(m, infer_ir!(src_ir, copy(argl); state = st))
            fx = apply_effects_override(m, frame_effects_meta(src_ir))
            exct = get(src_ir.meta, :exct, Any)
            old = st.cache[mi]::UResult
            widened = umerge(old.rt, rt)
            it >= 6 && (widened = CC.widenconst(widenucond(widened)))  # ascent escalation
            fx = CC.merge_effects(old.effects, fx)           # monotone descent
            exct = old.exct === Any ? Any :
                   CC.tmerge(CC.fallback_lattice, old.exct, exct)
            st.cache[mi] = UResult(widened, fx, exct)
            if !(mi in st.cycle_hit)
                converged = true
                break
            end
            if st.stale_depth < mydepth
                # nested root inside a larger active SCC: exactly one pass by
                # design — the outermost root's reruns recompute us (avoids
                # compounding nested fixpoints). NOT a failed fixpoint: the
                # outer root's joint convergence check owns our stability.
                deferred = true
                break
            end
            # outermost root: joint convergence over the whole SCC
            changed = !ulat_eq(widened, old.rt) || fx != old.effects ||
                      !lat_eq(exct, old.exct)
            changed |= scc_update!(st, it >= 6)
            if !changed
                converged = true
                break
            end
        end
        if (st.cache[mi]::UResult).rt === Union{} && mi in st.cycle_hit
            # Bottom "fixpoint" reached only through our own optimistic seed
            # (recursive reads returned the Union{} seed, whose dead-tail kills
            # then suppressed every real path). A self-supporting Bottom cannot
            # be trusted: settle at the sound over-approximation.
            st.cache[mi] = UResult(Any, CC.Effects(), Any)
            converged = true
        end
        ok = true
    finally
        delete!(st.active, mi)
        # an escaping exception must not leave the optimistic seed behind
        ok || delete!(st.cache, mi)
    end
    # a deferred nested root is not a failed fixpoint (the outer root owns it)
    converged || deferred || (st.nonconverged += 1)
    converged && mi in st.cycle_hit && (st.resolutions += 1)
    r = st.cache[mi]::UResult
    # context-dependent (tainted) results move to the appropriate scratch;
    # cycle taint takes priority (see the const-seeded epilogue)
    if tainted_cycle()
        delete!(st.cache, mi)
        st.cycle_scratch[mi] = r.rt === Union{} ? (r, st.resolutions) : r
    elseif tainted_limit()
        delete!(st.cache, mi)
        st.scratch[mi] = r
    elseif converged
        memo_frame_store!(st, mi, r, tlo, tpo, tco)
    end
    # flush only a clean, jointly-converged final pass (no non-converged
    # inner fixpoints, no resource cutoffs)
    frame_done(converged && st.nonconverged == nc0 && !tainted_limit())
    return r
end

"""Stock `most_general_argtypes`' per-parameter refinement for sig-derived
argument lattices: singleton types seed as their `Const` instance and
egality-pinned type arguments (`TypeEgal{T}`, `isconstType`) seed as
`Const(T)` — getfield/isdefined/fieldcount folds on such arguments need the
`Const` element (stock's cache-entry argtypes carry exactly this)."""
function seed_arglattice(@nospecialize(t))
    t isa Type || return t
    if t isa DataType && Base.issingletontype(t)
        return CC.Const(t.instance)
    elseif CC.isconstType(t)
        return CC.Const(CC.type_parameter(t))
    end
    return t
end

function method_arglattice(m::Method, mi::Core.MethodInstance, args::Vector{Any})
    nparams = Int(m.nargs)
    argl = Vector{Any}(undef, nparams)
    havecaller = !isempty(args) && !any(a -> CC.isvarargtype(a), args)
    if !m.isva && nparams == length(args) && havecaller
        for i in 1:nparams
            argl[i] = args[i]
        end
        return argl
    end
    if m.isva && havecaller && length(args) >= nparams - 1
        # precise vararg tuple from the caller's trailing argument lattices
        for i in 1:(nparams - 1)
            argl[i] = args[i]
        end
        rest = Any[widenucond(args[i]) for i in nparams:length(args)]
        argl[nparams] = try
            CC.tuple_tfunc(CC.fallback_lattice, rest)
        catch
            Tuple{Any[CC.widenconst(r) for r in rest]...}
        end
        return argl
    end
    spec = mi.specTypes
    sigts = Base.unwrap_unionall(spec)
    sigts isa DataType || return nothing
    ps = sigts.parameters
    if m.isva
        for i in 1:(nparams - 1)
            argl[i] = i <= length(ps) ? Base.rewrap_unionall(ps[i], spec) : Any
            argl[i] isa Type || (argl[i] = Any)
            argl[i] = seed_arglattice(argl[i])
        end
        # vararg tuple lattice: precise when the trailing sig is concrete
        rest = Any[Base.rewrap_unionall(ps[i], spec) for i in nparams:length(ps)]
        # degrade unusable entries elementwise; a Vararg tail must stay Vararg
        rest = Any[(t isa Type || CC.isvarargtype(t)) ? t : Any for t in rest]
        argl[nparams] = try
            Tuple{rest...}
        catch
            Tuple
        end
        return argl
    end
    for i in 1:nparams
        argl[i] = i <= length(ps) ? Base.rewrap_unionall(ps[i], spec) : Any
        argl[i] isa Type || (argl[i] = Any)
        argl[i] = seed_arglattice(argl[i])
    end
    return argl
end

"Frame-effects meta published by `infer_ir!` (all-false when missing)."
function frame_effects_meta(ir::UnifiedIR.IR)
    e = get(ir.meta, :effects, nothing)
    return e isa CC.Effects ? e : CC.Effects()
end

native_result(fr::Frame, match::Core.MethodMatch) =
    UResult(native_rt(fr, match), CC.Effects(), Any)

function native_rt(fr::Frame, match::Core.MethodMatch)
    fr.st.cfg.native_fallback || return Any
    fr.st.stats.native_fallbacks += 1
    try
        if fr.st.edges !== nothing
            # driver mode: the ambient world (jl_typeinf_world under global
            # activation) may lag the inference world — ask the stock oracle
            # at the collector's world explicitly. Edges: the caller's match
            # lookup is already recorded, and the oracle caches its own
            # CodeInstance chain in the global cache, so invalidation of
            # anything the answer depends on reaches us through the
            # match-edge backedge (mi-level invalidation is transitive).
            # For the cross-request memo the answer is a FACT (the oracle's
            # transitive view is not otherwise revalidatable): trace it.
            rt = Core.Compiler.return_type(match.spec_types, fr.st.cfg.world)
            trace!(fr.st.edges, (0x4, match.spec_types, rt))
            return rt
        end
        return Core.Compiler.return_type(match.spec_types)
    catch
        memo_poison!(fr.st.edges)
        return Any
    end
end
