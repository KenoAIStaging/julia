# The optimizer port (§10.4): passes over typed UnifiedIR. Simple passes on
# dense state; branch folding and inlining through editable sessions and
# splice_body! — the two-phase mutate-then-compact discipline, reformed.

"""
    refine_effects!(ir) -> Int

Set per-statement effect flags from inferred information: builtin calls get
`Compiler.builtin_effects`-derived flags; Const-typed statements of builtin
provenance become foldable; nothrow `new` constructions of concrete types
become REMOVABLE (dead-`new` elimination then falls out of `dce!` — SROA
deliverable 1c). Returns statements refined.
"""
function refine_effects!(ir::UnifiedIR.IR; interp = CC.NativeInterpreter())
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        if k === K"new"
            T = concrete_datatype(stmt_lattice(ir, UnifiedIR.getop(ir, s, 1)))
            T isa DataType || continue
            nsupplied = UnifiedIR.nops(ir, s) - 1
            nsupplied <= fieldcount(T) || continue    # over-arity `new` throws
            nothrow = true
            for i in 1:nsupplied
                at = CC.widenconst(stmt_lattice(ir, UnifiedIR.getop(ir, s, i + 1)))
                (at isa Type && at <: fieldtype(T, i)) || (nothrow = false; break)
            end
            nothrow || continue
            flags = UnifiedIR.FLAG_EFFECT_FREE | UnifiedIR.FLAG_NOTHROW |
                    UnifiedIR.FLAG_TERMINATES
            ismutabletype(T) || (flags |= UnifiedIR.FLAG_CONSISTENT)
            flags |= UnifiedIR.stmt_flag(ir, s)   # refinement is monotone upward
            if flags != UnifiedIR.stmt_flag(ir, s)
                UnifiedIR.set_flag!(ir, s, flags)
                n += 1
            end
            continue
        end
        if k === K"globalref"
            # reads of constant, defined bindings are foldable
            o = UnifiedIR.getop(ir, s, 1)
            UnifiedIR.optag(o) == UnifiedIR.TAG_GLOBAL || continue
            g = ir.body.globals[UnifiedIR.payload(o)]
            (isconst(g.mod, g.name) && isdefined(g.mod, g.name)) || continue
            flags = UnifiedIR.FLAG_CONSISTENT | UnifiedIR.FLAG_REMOVABLE
            flags |= UnifiedIR.stmt_flag(ir, s)   # refinement is monotone upward
            if flags != UnifiedIR.stmt_flag(ir, s)
                UnifiedIR.set_flag!(ir, s, flags)
                n += 1
            end
            continue
        end
        (k === K"call" || k === K"intrinsic") || continue
        fo = UnifiedIR.getop(ir, s, 1)
        fl = static_operand_value(ir, fo)
        fl isa Core.Builtin || continue
        argl = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i)) for i in 2:UnifiedIR.nops(ir, s)]
        rt = UnifiedIR.stmt_type(ir, s)
        effects = try
            CC.builtin_effects(CC.fallback_lattice, fl, argl, rt isa Type ? rt : Any)
        catch
            continue
        end
        flags = UInt32(0)
        CC.is_consistent(effects) && (flags |= UnifiedIR.FLAG_CONSISTENT)
        CC.is_effect_free(effects) && (flags |= UnifiedIR.FLAG_EFFECT_FREE)
        # builtin_effects is blind to min_ninitialized-violating news (F11,
        # #52857 class): a maybe-undef immutable field load must keep its
        # conditional UndefRefError throw
        CC.is_nothrow(effects) && !getfield_maybe_undef(fl, argl) &&
            (flags |= UnifiedIR.FLAG_NOTHROW)
        CC.is_terminates(effects) && (flags |= UnifiedIR.FLAG_TERMINATES)
        if flags & UnifiedIR.FLAG_NOTHROW == 0 &&
           (fl === Core.getfield || fl === Base.getfield) && length(argl) >= 2
            # the tfuncs refuse Const-of-mutable subjects, but definedness
            # is MONOTONE (a defined field never becomes undefined), so a
            # field observed defined now cannot throw later; the remaining
            # throw conditions are all statically checkable
            v = argl[1] isa CC.Const ? (argl[1]::CC.Const).val : nothing
            fld = argl[2] isa CC.Const ? (argl[2]::CC.Const).val : nothing
            extra_ok = true
            for k2 in 3:length(argl)
                e = argl[k2] isa CC.Const ? (argl[k2]::CC.Const).val : missing
                (e === true || e === false || e === :not_atomic) || (extra_ok = false; break)
            end
            if v !== nothing && extra_ok && length(argl) <= 4
                # note: not_atomic READS of atomic fields are legal (only
                # writes require an ordering), so no isfieldatomic guard
                fi = field_index_of(typeof(v), fld)
                if fi isa Int && isdefined(v, fi)
                    flags |= UnifiedIR.FLAG_NOTHROW
                end
            end
        end
        # inference's transfer results (just published into the flag column)
        # can be strictly more precise than the builtin recompute
        # (apply_type_nothrow with Const args, e.g.); never downgrade — the
        # refinement must be monotone upward or the round ledger flaps
        flags |= UnifiedIR.stmt_flag(ir, s)
        if flags != UnifiedIR.stmt_flag(ir, s)
            UnifiedIR.set_flag!(ir, s, flags)
            n += 1
        end
    end
    return n
end

"""Const-VALUE operand builder: like `vop`, but never reinterprets IR-typed
VALUES as IR references. `vop`'s `StmtId`/`Operand` pass-throughs exist for
callers holding actual references; a lattice-Const being materialized may
BE a `StmtId`/`Operand` value when the pipeline compiles UnifiedIR's own
code (self-hosting), and encoding it through `vop` silently rewires the
statement graph — `Const(StmtId(0))` produced the wrap_in_if! BoundsError
under activate!, any other id aliases an arbitrary statement (wave 11).
Likewise `vop`'s `GlobalRef` routing targets the globals table — the
binding-READ form — but a `Const(GlobalRef)` VALUE is data (e.g.
`invokelatest_gr`'s target, the TOML Printer miscompile, wave 12): it must
intern as a pool CONSTANT or downstream folding resolves the binding."""
function const_vop(ir::UnifiedIR.IR, @nospecialize(v))
    (v isa UnifiedIR.StmtId || v isa UnifiedIR.Operand || v isa GlobalRef) &&
        return UnifiedIR.op_constidx(UnifiedIR.intern_const!(ir.body, v))
    return UnifiedIR.vop(ir, v)
end

"Constant value of an operand, or nothing (statements consult the type column)."
function static_operand_value(ir::UnifiedIR.IR, o::UnifiedIR.Operand)
    t = UnifiedIR.optag(o)
    if t == UnifiedIR.TAG_INLINE
        return UnifiedIR.imm_value(o)
    elseif t == UnifiedIR.TAG_CONST
        return ir.body.constants[UnifiedIR.payload(o)]
    elseif t == UnifiedIR.TAG_GLOBAL
        g = ir.body.globals[UnifiedIR.payload(o)]
        (isconst(g.mod, g.name) && isdefined(g.mod, g.name)) && return getglobal(g.mod, g.name)
        return nothing
    elseif t == UnifiedIR.TAG_STMT
        tt = UnifiedIR.stmt_type(ir, UnifiedIR.asstmt(o))
        tt isa CC.Const && return tt.val
        st = CC.singleton_type(tt isa Type ? tt : Any)
        return st
    end
    return nothing
end

function stmt_lattice(ir::UnifiedIR.IR, o::UnifiedIR.Operand)
    t = UnifiedIR.optag(o)
    if t == UnifiedIR.TAG_STMT
        tt = UnifiedIR.stmt_type(ir, UnifiedIR.asstmt(o))
        return tt === nothing ? Any : tt
    elseif t == UnifiedIR.TAG_INLINE
        return CC.Const(UnifiedIR.imm_value(o))
    elseif t == UnifiedIR.TAG_CONST
        return CC.Const(ir.body.constants[UnifiedIR.payload(o)])
    elseif t == UnifiedIR.TAG_GLOBAL
        g = ir.body.globals[UnifiedIR.payload(o)]
        (isconst(g.mod, g.name) && isdefined(g.mod, g.name)) &&
            return CC.Const(getglobal(g.mod, g.name))
        return Any
    elseif t == UnifiedIR.TAG_SPARAM
        # the lattice channel inference publishes (Const for pinned values,
        # Type{_} bounds for constrained TypeVars); raw sparam_vals fallback
        idx = Int(UnifiedIR.payload(o))
        lat = get(ir.meta, :sptypes_lat, nothing)
        if lat isa Vector{Any} && 1 <= idx <= length(lat)
            return lat[idx]
        end
        sp = ir.sptypes
        if 1 <= idx <= length(sp)
            v = sp[idx]
            (v isa Core.SimpleVector || v isa TypeVar) || return CC.Const(v)
        end
        return Any
    end
    return Any
end

"""
    materialize_consts!(ir) -> Int

Replace statements whose inferred type is `Const(v)` and whose flags satisfy
the foldable mask with `K"value"` constants (footprint-preserving), and
forward the constant directly into the use sites so DCE can delete the whole
chain. Skips identity-bearing constants.
"""
function materialize_consts!(ir::UnifiedIR.IR)
    n = 0
    # stock's is_removable_if_unused mask (IR_FLAGS_REMOVABLE): a Const-typed
    # statement's uses may be forwarded on lattice soundness alone (any
    # completed value is egal to the Const), and the statement itself deleted
    # when it is effect-free, nothrow and terminating — :consistent-cy is an
    # egality contract for FRESH mutable values across executions, which the
    # identity-bearing-constant check below already excludes (e.g. an invoke
    # of a `(!c,+e,+n)` callee returning `nothing` must still fold away,
    # the broadcast_noescape corpus shape)
    foldable = UnifiedIR.FLAG_REMOVABLE
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        UnifiedIR.result_arity(k) == 1 || continue
        UnifiedIR.owns_regions(k) && continue
        (k === K"value" || k === K"region_arg" || k === K"cell" || k === K"cell_shared") && continue
        t = UnifiedIR.stmt_type(ir, s)
        t isa CC.Const || continue
        v = t.val
        ismutable(v) && !(v isa Union{Type,Function,Module,Symbol,String}) && continue
        UnifiedIR.stmt_flag(ir, s) & foldable == foldable || continue
        # K"value" requires a pool constant (its schema is OC_CONST)
        co = UnifiedIR.op_constidx(UnifiedIR.intern_const!(ir.body, v))
        UnifiedIR.replace_stmt!(ir, s, K"value", co; type = t)
        UnifiedIR.replace_uses!(ir, s => const_vop(ir, v))
        n += 1
    end
    n > 0 && UnifiedIR.flush_renames!(ir)
    return n
end

"""
    forward_refines!(ir) -> Int

Uses of a `refine` that adds no type information over its operand are
rewritten to the operand (the #54762 Pi-accumulation cleanup, per the
`refine` canonicalizability note in §5.8); constant-operand refines forward
unconditionally. Genuinely narrowing refines (union-split arms) are kept.
"""
function forward_refines!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"refine" || continue
        o = UnifiedIR.getop(ir, s, 1)
        t = UnifiedIR.optag(o)
        if t == UnifiedIR.TAG_STMT
            d = UnifiedIR.asstmt(o)
            rt = CC.widenconst(UnifiedIR.stmt_type(ir, s))
            dt = CC.widenconst(UnifiedIR.stmt_type(ir, d))
            (rt isa Type && dt isa Type && dt <: rt) || continue   # narrows: keep
            UnifiedIR.replace_uses!(ir, s => o)
            n += 1
        elseif t == UnifiedIR.TAG_CONST || t == UnifiedIR.TAG_INLINE
            UnifiedIR.replace_uses!(ir, s => o)
            n += 1
        end
    end
    n > 0 && UnifiedIR.flush_renames!(ir)
    return n
end

"""
    canonicalize_getfields!(ir) -> Int

Julia-dialect canonicalization (§3.2): `getfield(x, fld::Const)` calls —
integer index, or symbol on a value of known concrete type, optionally with a
trailing boundscheck operand — become the explicit `K"extract"` kind
(inline-encoded, index 0-based).
"""
function canonicalize_getfields!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        (nop == 3 || nop == 4) || continue
        callee = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        callee === Core.getfield || callee === Base.getfield || continue
        need_inbounds_proof = false
        if nop == 4
            # the trailing operand (boundscheck flag or memory order) is
            # dropped by the conversion: legal for a Bool literal or
            # :not_atomic (an INVALID order symbol makes the original
            # getfield throw). A DYNAMIC Bool (an `Expr(:boundscheck)`
            # carried through inlining) only selects whether an
            # out-of-bounds index raises BoundsError, so dropping it is
            # additionally legal when the index is provably in bounds —
            # then no execution can distinguish the two.
            extra = static_operand_value(ir, UnifiedIR.getop(ir, s, 4))
            if !(extra === true || extra === false || extra === :not_atomic)
                CC.widenconst(stmt_lattice(ir, UnifiedIR.getop(ir, s, 4))) === Bool || continue
                need_inbounds_proof = true
            end
        end
        vo = UnifiedIR.getop(ir, s, 2)
        UnifiedIR.optag(vo) == UnifiedIR.TAG_STMT || continue
        io = UnifiedIR.getop(ir, s, 3)
        idx = static_operand_value(ir, io)
        if idx isa Symbol
            xt = CC.widenconst(stmt_lattice(ir, vo))
            if xt isa DataType && isconcretetype(xt)
                idx = field_index_of(xt, idx)
                idx === nothing && continue
            elseif xt isa Union
                # every union component must agree on the field's index
                # (the union-split struct corpus: same field, same slot)
                i0 = nothing
                for c in Base.uniontypes(xt)
                    (c isa DataType && isconcretetype(c)) || (i0 = nothing; break)
                    fi = field_index_of(c, idx)
                    fi === nothing && (i0 = nothing; break)
                    i0 === nothing ? (i0 = fi) : (fi == i0 || (i0 = nothing; break))
                end
                i0 isa Int || continue
                idx = i0
            else
                continue
            end
        end
        idx isa Int || continue
        idx >= 1 || continue
        idx < (1 << 23) || continue
        if need_inbounds_proof
            xt = CC.widenconst(stmt_lattice(ir, vo))
            xt isa DataType || continue
            nf = CC.datatype_fieldcount(xt)
            (nf isa Int && idx <= nf) || continue
            # atomic fields reject a plain (non-order) access; dropping a
            # Bool boundscheck must not legalize an atomics violation
            Base.isfieldatomic(xt, idx) && continue
        end
        # the kind-default PURE flag claims nothrow, which a load of a field
        # the PartialStruct does not prove defined (under-initialized
        # immutable new, #52857/F11) must not: keep the conditional
        # UndefRefError throw observable
        flag = getfield_maybe_undef(Core.getfield,
                                    Any[stmt_lattice(ir, vo), CC.Const(idx)]) ?
            (UnifiedIR.FLAG_PURE & ~UnifiedIR.FLAG_NOTHROW) : nothing
        UnifiedIR.replace_stmt!(ir, s, K"extract", vo, UnifiedIR.op_inline(idx);
                                type = UnifiedIR.stmt_type(ir, s), flag)
        n += 1
    end
    return n
end

"Result-terminated live arms of an `if`/region-owning op: (arm region,
result terminator) pairs; diverging arms (throw/unreachable) are omitted.
Returns nothing when any live arm is a guard or lacks a terminator."
function result_arms(ir::UnifiedIR.IR, def::StmtId)
    arms = Tuple{RegionId,StmtId}[]
    for rid in UnifiedIR.live_owned_regions(ir, def)
        reg = UnifiedIR.getregion(ir, rid)
        reg.kind === UnifiedIR.REGION_ARM || return nothing
        t = UnifiedIR.region_terminator(ir, rid)
        t === nothing && return nothing
        tk = UnifiedIR.stmt_kind(ir, t)
        if tk === K"result"
            push!(arms, (rid, t))
        elseif !UnifiedIR.is_diverge_kind(tk)
            return nothing
        end
    end
    return arms
end

"A constant immutable struct value's field `fidx` as a fresh constant
operand, or nothing (mutable/undef/out-of-range values decline)."
function const_struct_field_op(ir::UnifiedIR.IR, @nospecialize(v), fidx::Int)
    v === nothing && return nothing
    ismutable(v) && return nothing
    (1 <= fidx <= nfields(v) && isdefined(v, fidx)) || return nothing
    return const_vop(ir, getfield(v, fidx))
end

"Rewrite the field-`fidx` load `s` over per-arm constructed values
(`armpairs` = (result stmt, value operand) per live arm of `ifop`) to
`select`/`refine` when every arm value is an arm-local immutable
new/`Core.tuple` (or a constant struct leaf) whose element operand is
visible at `s`. True on success."
function forward_arm_elements!(ir::UnifiedIR.IR, s::StmtId, ifop::StmtId,
                               armpairs::Vector{Tuple{StmtId,UnifiedIR.Operand}},
                               fidx::Int)
    els = UnifiedIR.Operand[]
    for (_, ro) in armpairs
        local el::UnifiedIR.Operand
        if UnifiedIR.optag(ro) != UnifiedIR.TAG_STMT
            # constant immutable struct leaf (a materialized arm result):
            # project the field as a fresh constant
            elc = const_struct_field_op(ir, static_operand_value(ir, ro), fidx)
            elc === nothing && return false
            push!(els, elc)
            continue
        end
        ad = skip_refines(ir, UnifiedIR.asstmt(ro))
        adk = UnifiedIR.stmt_kind(ir, ad)
        if adk === K"call" &&
           static_operand_value(ir, UnifiedIR.getop(ir, ad, 1)) === Core.tuple
            1 + fidx <= UnifiedIR.nops(ir, ad) || return false
            el = UnifiedIR.getop(ir, ad, fidx + 1)
        elseif adk === K"new"
            Ta = concrete_datatype(stmt_lattice(ir, UnifiedIR.getop(ir, ad, 1)))
            (Ta isa DataType && !ismutabletype(Ta)) || return false
            fidx <= UnifiedIR.nops(ir, ad) - 1 || return false
            el = UnifiedIR.getop(ir, ad, fidx + 1)
        else
            # Const-typed arm def of any kind: still a constant struct leaf
            lat = UnifiedIR.stmt_type(ir, ad)
            lat isa CC.Const || return false
            elc = const_struct_field_op(ir, lat.val, fidx)
            elc === nothing && return false
            push!(els, elc)
            continue
        end
        if UnifiedIR.optag(el) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(el), s) || return false
        end
        push!(els, el)
    end
    isempty(els) && return false
    if length(els) == 1
        UnifiedIR.replace_stmt!(ir, s, K"refine", els[1];
                                type = UnifiedIR.stmt_type(ir, s))
        return true
    end
    length(els) == 2 || return false
    co = UnifiedIR.getop(ir, ifop, 1)
    CC.widenconst(stmt_lattice(ir, co)) === Bool || return false
    if UnifiedIR.optag(co) == UnifiedIR.TAG_STMT
        UnifiedIR.visible(ir, UnifiedIR.asstmt(co), s) || return false
    end
    UnifiedIR.replace_stmt!(ir, s, K"select", co, els[1], els[2];
                            type = UnifiedIR.stmt_type(ir, s))
    return true
end

"""
    forward_extracts!(ir) -> Int

Immutable-struct SROA, load-forwarding case (deliverable 1a): `extract(x, i)`
of a locally-constructed `call Core.tuple(a...)` or `K"new"` of a concrete
*immutable* type — following `refine` chains — becomes `refine a[i]`.
Legality of the forwarded operand at the use site is checked with
`UnifiedIR.visible` (§5.1). Additionally, `extract` of a multi-result `if`
whose every result-terminated arm passes the SAME constant at that
position folds to the constant (the definedness-channel case the entry
lowering produces for `local`-scoped conditionals).
"""
function forward_extracts!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"extract" || continue
        vo = UnifiedIR.getop(ir, s, 1)
        UnifiedIR.optag(vo) == UnifiedIR.TAG_STMT || continue
        def = skip_refines(ir, UnifiedIR.asstmt(vo))
        idx = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, s, 2))::Int64)
        dk = UnifiedIR.stmt_kind(ir, def)
        local el::UnifiedIR.Operand
        if dk === K"if"
            arms = result_arms(ir, def)
            (arms === nothing || isempty(arms)) && continue
            # uniform-constant position: fold outright. POSITIONAL reads are
            # only meaningful for multi-operand (de-tupled) results — a
            # single-operand result IS the if's value, and an extract over
            # it projects INTO that value (the armpairs path below projects
            # correctly, incl. constant leaves via const_struct_field_op)
            v0 = nothing
            uniform = true
            for (_, t) in arms
                (2 <= UnifiedIR.nops(ir, t) && idx <= UnifiedIR.nops(ir, t)) ||
                    (uniform = false; break)
                v = static_operand_value(ir, UnifiedIR.getop(ir, t, idx))
                v === nothing && (uniform = false; break)
                ismutable(v) && !(v isa Union{Type,Function,Module,Symbol,String}) &&
                    (uniform = false; break)
                v0 === nothing ? (v0 = v) : (v === v0 || (uniform = false; break))
            end
            if uniform && v0 !== nothing
                UnifiedIR.replace_stmt!(ir, s, K"refine", const_vop(ir, v0);
                                        type = UnifiedIR.stmt_type(ir, s))
                n += 1
                continue
            end
            # per-arm construction (stock's phi-of-news load forwarding):
            # every result arm yields an arm-local immutable new/tuple whose
            # field `idx` operand is visible outside the arm — the load
            # becomes select(cond, el₁, el₂) (or the sole arm's element)
            armpairs = Tuple{StmtId,UnifiedIR.Operand}[]
            armok = true
            for (_, t) in arms
                UnifiedIR.nops(ir, t) == 1 || (armok = false; break)
                push!(armpairs, (t, UnifiedIR.getop(ir, t, 1)))
            end
            armok || continue
            forward_arm_elements!(ir, s, def, armpairs, idx) && (n += 1)
            continue
        elseif dk === K"extract"
            # extract-of-extract chain: the base projects result position k
            # of a multi-result if; this extract loads field `idx` of that
            # per-arm value
            bo = UnifiedIR.getop(ir, def, 1)
            UnifiedIR.optag(bo) == UnifiedIR.TAG_STMT || continue
            if2 = skip_refines(ir, UnifiedIR.asstmt(bo))
            UnifiedIR.stmt_kind(ir, if2) === K"if" || continue
            k = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, def, 2))::Int64)
            arms = result_arms(ir, if2)
            (arms === nothing || isempty(arms)) && continue
            armpairs = Tuple{StmtId,UnifiedIR.Operand}[]
            armok = true
            for (_, t) in arms
                # positional base extract: de-tupled multi-operand results
                # only (a single-operand result would make the base a
                # projection INTO the value, not a result selection)
                (2 <= UnifiedIR.nops(ir, t) && k <= UnifiedIR.nops(ir, t)) ||
                    (armok = false; break)
                push!(armpairs, (t, UnifiedIR.getop(ir, t, k)))
            end
            armok || continue
            forward_arm_elements!(ir, s, if2, armpairs, idx) && (n += 1)
            continue
        elseif dk === K"call"
            callee = static_operand_value(ir, UnifiedIR.getop(ir, def, 1))
            if (callee === Core.getfield || callee === Base.getfield) &&
               UnifiedIR.nops(ir, def) == 3 &&
               static_operand_value(ir, UnifiedIR.getop(ir, def, 3)) === :captures
                # captures-tuple load of an opaque closure (the OC-inlining
                # self substitution): element idx is new_opaque_closure env
                # operand 5+idx (stock sroa's is_getfield_captures walk)
                oo = UnifiedIR.getop(ir, def, 2)
                UnifiedIR.optag(oo) == UnifiedIR.TAG_STMT || continue
                ocdef = skip_refines(ir, UnifiedIR.asstmt(oo))
                UnifiedIR.stmt_kind(ir, ocdef) === K"new_opaque_closure" || continue
                5 + idx <= UnifiedIR.nops(ir, ocdef) || continue
                el = UnifiedIR.getop(ir, ocdef, 5 + idx)
            else
                callee === Core.tuple || continue
                1 + idx <= UnifiedIR.nops(ir, def) || continue
                el = UnifiedIR.getop(ir, def, idx + 1)
            end
        elseif dk === K"new"
            T = concrete_datatype(stmt_lattice(ir, UnifiedIR.getop(ir, def, 1)))
            (T isa DataType && !ismutabletype(T)) || continue
            idx <= UnifiedIR.nops(ir, def) - 1 || continue  # field must be supplied
            el = UnifiedIR.getop(ir, def, idx + 1)
        else
            continue
        end
        # the forwarded operand must be visible at the extract (§5.1 all three
        # clauses); constants/globals/immediates are always legal
        if UnifiedIR.optag(el) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(el), s) || continue
        end
        UnifiedIR.replace_stmt!(ir, s, K"refine", el; type = UnifiedIR.stmt_type(ir, s))
        n += 1
    end
    return n
end

"""
    dedup_selects!(ir) -> Int

CSE for `select`s: identical (cond, a, b) triples collapse to the first
occurrence when it is visible at the duplicate (stock SROA's lifting-cache
phi dedup — two loads of the same field of the same join produce ONE join).
The duplicate becomes a `refine` of the survivor; `forward_refines!` and DCE
finish the cleanup.
"""
function dedup_selects!(ir::UnifiedIR.IR)
    seen = Dict{NTuple{3,UnifiedIR.Operand},StmtId}()
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"select" || continue
        UnifiedIR.nops(ir, s) == 3 || continue
        key = (UnifiedIR.getop(ir, s, 1), UnifiedIR.getop(ir, s, 2),
               UnifiedIR.getop(ir, s, 3))
        first = get(seen, key, nothing)
        if first === nothing
            seen[key] = s
            continue
        end
        UnifiedIR.visible(ir, first, s) || continue
        UnifiedIR.replace_stmt!(ir, s, K"refine", UnifiedIR.op_stmt(first);
                                type = UnifiedIR.stmt_type(ir, s))
        n += 1
    end
    return n
end

"""
    lift_keyvalue_gets!(ir) -> Int

The stock `lift_keyvalue_get!` port (Core.OptimizedGenerics.KeyValue
protocol, the PersistentDict corpus): a `KeyValue.get(collection, key)`
whose collection chain walks through `KeyValue.set` calls resolves to the
value stored under an egal key — the get becomes `new Wrapper(val)` (the
`Some{V}` extracted from the get's `Union{Nothing, Wrapper}` return type).
Chains walk through refines, non-matching sets (their source collection),
and `select`s whose arms lift to one uniform operand (the stock
phi-lifting's uniform case; distinct-arm nests are left for a later
extension). The delete forms (shorter arglists) are unmodeled, as in stock.
"""
function lift_keyvalue_gets!(ir::UnifiedIR.IR)
    kvget = Core.OptimizedGenerics.KeyValue.get
    kvset = Core.OptimizedGenerics.KeyValue.set
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        (k === K"invoke" || k === K"call") || continue
        base = k === K"invoke" ? 2 : 1
        nop = UnifiedIR.nops(ir, s)
        nop - base == 2 || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, base)) === kvget || continue
        keyop = UnifiedIR.getop(ir, s, nop)
        collop = UnifiedIR.getop(ir, s, nop - 1)
        keyl = stmt_lattice(ir, keyop)
        # resolve the stored value operand for `keyop` through the chain
        function kv_walk(co::UnifiedIR.Operand, depth::Int)
            depth > 32 && return nothing
            UnifiedIR.optag(co) == UnifiedIR.TAG_STMT || return nothing
            def = skip_refines(ir, UnifiedIR.asstmt(co))
            dk = UnifiedIR.stmt_kind(ir, def)
            if dk === K"select"
                UnifiedIR.nops(ir, def) == 3 || return nothing
                v1 = kv_walk(UnifiedIR.getop(ir, def, 2), depth + 1)
                v1 === nothing && return nothing
                v2 = kv_walk(UnifiedIR.getop(ir, def, 3), depth + 1)
                (v2 === nothing || v2 != v1) && return nothing  # uniform arms only
                return v1
            end
            if dk === K"extract"
                # projection of a multi-result if: walk position k of each arm
                bo = UnifiedIR.getop(ir, def, 1)
                UnifiedIR.optag(bo) == UnifiedIR.TAG_STMT || return nothing
                ifdef = skip_refines(ir, UnifiedIR.asstmt(bo))
                UnifiedIR.stmt_kind(ir, ifdef) === K"if" || return nothing
                kidx = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, def, 2))::Int64)
                arms = result_arms(ir, ifdef)
                (arms === nothing || isempty(arms)) && return nothing
                v0 = nothing
                for (_, t) in arms
                    # positional walk: de-tupled multi-operand results only
                    # (a single-operand result is the if's VALUE; the extract
                    # projects into it — not a result-position selection)
                    (2 <= UnifiedIR.nops(ir, t) && kidx <= UnifiedIR.nops(ir, t)) ||
                        return nothing
                    v = kv_walk(UnifiedIR.getop(ir, t, kidx), depth + 1)
                    v === nothing && return nothing
                    v0 === nothing ? (v0 = v) : (v == v0 || return nothing)
                end
                return v0
            end
            if dk === K"if"
                arms = result_arms(ir, def)
                (arms === nothing || isempty(arms)) && return nothing
                v0 = nothing
                for (_, t) in arms
                    UnifiedIR.nops(ir, t) == 1 || return nothing
                    v = kv_walk(UnifiedIR.getop(ir, t, 1), depth + 1)
                    v === nothing && return nothing
                    v0 === nothing ? (v0 = v) : (v == v0 || return nothing)
                end
                return v0
            end
            (dk === K"invoke" || dk === K"call") || return nothing
            dbase = dk === K"invoke" ? 2 : 1
            static_operand_value(ir, UnifiedIR.getop(ir, def, dbase)) === kvset ||
                return nothing
            dnop = UnifiedIR.nops(ir, def)
            # set([T,] collection, key, val): three trailing value operands;
            # shorter forms are the unmodeled deletes
            dnop - dbase >= 3 || return nothing
            skop = UnifiedIR.getop(ir, def, dnop - 1)
            if skop == keyop
                return UnifiedIR.getop(ir, def, dnop)
            end
            skl = stmt_lattice(ir, skop)
            egal = try
                CC.egal_tfunc(CC.fallback_lattice, keyl, skl)
            catch
                Bool
            end
            egal isa CC.Const || return nothing
            egal.val === true && return UnifiedIR.getop(ir, def, dnop)
            egal.val === false || return nothing
            return kv_walk(UnifiedIR.getop(ir, def, dnop - 2), depth + 1)
        end
        valop = kv_walk(collop, 0)
        valop === nothing && continue
        if UnifiedIR.optag(valop) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(valop), s) || continue
        end
        # wrapper type: subtract Nothing from the get's declared return
        rt0 = UnifiedIR.stmt_type(ir, s)
        rt = CC.widenconst(rt0 === nothing ? Any : rt0)
        wrapper = try
            CC.typesubtract(rt, Nothing, 0)
        catch
            continue
        end
        (wrapper isa DataType && isconcretetype(wrapper) &&
         fieldcount(wrapper) == 1) || continue
        vt = CC.widenconst(stmt_lattice(ir, valop))
        ok = try
            vt isa Type && CC.:⊑(CC.fallback_lattice, vt, fieldtype(wrapper, 1))
        catch
            false
        end
        ok || continue
        UnifiedIR.replace_stmt!(ir, s, K"new", UnifiedIR.vop(ir, wrapper), valop;
                                type = wrapper)
        n += 1
    end
    return n
end

"""The structural arm of stock `_lift_svec_ref`: a
`Core._compute_sparams(m, args...)` whose method signature mentions its
(outermost) typevar in exactly one parameter, where the corresponding
argument was built by `new(apply_type(T′, x), ...)` (or IS an
`apply_type(T′, x)`-constructed type) with the typevar in the matching
parameter slot of the same type constructor — then sparam 1 is `x` itself
(the `NamedTuple{(name,)}(t)` constructor corpus). Returns the forwardable
operand or `nothing`."""
function _lift_compute_sparams(ir::UnifiedIR.IR, def::StmtId)
    UnifiedIR.nops(ir, def) >= 3 || return nothing   # (f, m, args...)
    m = static_operand_value(ir, UnifiedIR.getop(ir, def, 2))
    m isa Method || return nothing
    sig0 = m.sig
    sig0 isa UnionAll || return nothing
    tvar = sig0.var
    sig = sig0.body
    sig isa DataType || return nothing
    sig.name === Tuple.name || return nothing
    params = sig.parameters::Core.SimpleVector
    i = nothing
    for j in 1:length(params)
        if CC.has_typevar(params[j], tvar)
            i === nothing || return nothing   # exactly one mention
            i = j
        end
    end
    i === nothing && return nothing
    arg = params[i]
    2 + i <= UnifiedIR.nops(ir, def) || return nothing
    rarg = UnifiedIR.getop(ir, def, 2 + i)
    UnifiedIR.optag(rarg) == UnifiedIR.TAG_STMT || return nothing
    argdef = skip_refines(ir, UnifiedIR.asstmt(rarg))
    if UnifiedIR.stmt_kind(ir, argdef) === K"new"
        to = UnifiedIR.getop(ir, argdef, 1)
        UnifiedIR.optag(to) == UnifiedIR.TAG_STMT || return nothing
        argdef = skip_refines(ir, UnifiedIR.asstmt(to))
    else
        # N.B. `Type{X}` is a `TypeEq` instance on this nightly: `isType`
        # covers both kinds, `type_parameter` projects X
        au = CC.unwrap_unionall(arg)
        CC.isType(au) || return nothing
        arg = CC.type_parameter(au)
    end
    (UnifiedIR.stmt_kind(ir, argdef) === K"call" &&
     UnifiedIR.nops(ir, argdef) == 3) || return nothing
    static_operand_value(ir, UnifiedIR.getop(ir, argdef, 1)) === Core.apply_type ||
        return nothing
    applyTl = stmt_lattice(ir, UnifiedIR.getop(ir, argdef, 2))
    applyTl isa CC.Const || return nothing
    applyT = applyTl.val
    applyT isa UnionAll || return nothing
    # N.B. valI == 1 only (stock's TODO): the outermost tvar suffices
    applyTvar = applyT.var
    applyTbody = CC.unwrap_unionall(applyT.body)
    arg = CC.unwrap_unionall(arg)
    (arg isa DataType && applyTbody isa DataType) || return nothing
    applyTbody.name === arg.name || return nothing
    length(applyTbody.parameters) == length(arg.parameters) || return nothing
    for j in 1:length(applyTbody.parameters)
        if applyTbody.parameters[j] === applyTvar && arg.parameters[j] === tvar
            return UnifiedIR.getop(ir, argdef, 3)
        end
    end
    return nothing
end

"""
    lift_svec_refs!(ir) -> Int

Stock `lift_svec_ref!`: a `Core._svec_ref(vec, idx)` with a Const in-range
`idx` resolves through a Const `SimpleVector` (element folds), through a
`Core.svec` call (operand forwards), or — via `_lift_compute_sparams` —
through the sparam-reconstruction shape the inliner materializes (stock's
spvals_ssa), letting the reconstruction chain DCE away entirely.
"""
function lift_svec_refs!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        UnifiedIR.nops(ir, s) == 3 || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, 1)) === Core._svec_ref ||
            continue
        idxl = stmt_lattice(ir, UnifiedIR.getop(ir, s, 3))
        (idxl isa CC.Const && idxl.val isa Int) || continue
        valI = idxl.val::Int
        valI >= 1 || continue
        vecop = UnifiedIR.getop(ir, s, 2)
        vecl = stmt_lattice(ir, vecop)
        local repl::UnifiedIR.Operand
        if vecl isa CC.Const && vecl.val isa Core.SimpleVector
            v = vecl.val::Core.SimpleVector
            valI <= length(v) || continue
            repl = const_vop(ir, v[valI])
        elseif UnifiedIR.optag(vecop) == UnifiedIR.TAG_STMT
            def = skip_refines(ir, UnifiedIR.asstmt(vecop))
            UnifiedIR.stmt_kind(ir, def) === K"call" || continue
            df = static_operand_value(ir, UnifiedIR.getop(ir, def, 1))
            if df === Core.svec
                valI <= UnifiedIR.nops(ir, def) - 1 || continue
                repl = UnifiedIR.getop(ir, def, valI + 1)
            elseif df === Core._compute_sparams && valI == 1
                r = _lift_compute_sparams(ir, def)
                r === nothing && continue
                repl = r
            else
                continue
            end
        else
            continue
        end
        if UnifiedIR.optag(repl) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(repl), s) || continue
        end
        UnifiedIR.replace_stmt!(ir, s, K"refine", repl;
                                type = stmt_lattice(ir, repl))
        n += 1
    end
    return n
end

"""
    fold_pure_queries!(ir) -> Int

Query-call folding and comparison lifting (stock `lift_comparison!` /
`typeassert` elimination, over the structured encoding — the subjects
stock sees as `PhiNode`s arrive here as `select`s):

  * `typeassert(x, T)` whose subject's type already proves `<: T` becomes
    `refine x` (the post-SROA typeassert elimination corpus);
  * `Core.ifelse(c::Const, a, b)` forwards the chosen operand;
    `Core.ifelse(c::Bool, a, a)` forwards `a`;
  * `===` / `isa` / `isdefined` whose subject is a `select` (or a residual
    `Core.ifelse` call) with a per-arm Const answer becomes
    `select(c, ans₁, ans₂)` — the union-typed comparison disappears.
"""
function fold_pure_queries!(ir::UnifiedIR.IR)
    n = 0
    L = CC.fallback_lattice
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        callee = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        if callee === Core.typeassert && nop == 3
            xo = UnifiedIR.getop(ir, s, 2)
            T = static_operand_value(ir, UnifiedIR.getop(ir, s, 3))
            T isa Type || continue
            xt = CC.widenconst(stmt_lattice(ir, xo))
            (xt isa Type && xt <: T) || continue
            UnifiedIR.replace_stmt!(ir, s, K"refine", xo;
                                    type = UnifiedIR.stmt_type(ir, s))
            n += 1
            continue
        end
        if callee === Core.ifelse && nop == 4
            co = UnifiedIR.getop(ir, s, 2)
            cv = static_operand_value(ir, co)
            ao = UnifiedIR.getop(ir, s, 3)
            bo = UnifiedIR.getop(ir, s, 4)
            if cv isa Bool
                UnifiedIR.replace_stmt!(ir, s, K"refine", cv ? ao : bo;
                                        type = UnifiedIR.stmt_type(ir, s))
                n += 1
            elseif ao == bo && CC.widenconst(stmt_lattice(ir, co)) === Bool
                UnifiedIR.replace_stmt!(ir, s, K"refine", ao;
                                        type = UnifiedIR.stmt_type(ir, s))
                n += 1
            end
            continue
        end
        # comparison lifting: subject is a select, a residual Core.ifelse
        # call, an if-result, or an extract of a multi-result if
        (callee === (===) || callee === isa || callee === isdefined) || continue
        nop == 3 || continue
        subj = 0
        armlats = Any[]
        local co::UnifiedIR.Operand
        for i in (callee === (===) ? (2, 3) : (2,))
            o = UnifiedIR.getop(ir, s, i)
            UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || continue
            d = skip_refines(ir, UnifiedIR.asstmt(o))
            dk = UnifiedIR.stmt_kind(ir, d)
            # `elem == 0`: the subject is the if's VALUE itself; `elem >= 1`:
            # the subject is `extract(if, elem)`. An if whose arm results
            # carry ONE operand has that operand as its value (an extract
            # over it projects INTO the runtime value — tuple element
            # `elem`); only a MULTI-operand result if de-tuples positionally
            # (its value is the synthetic escape tuple). Confusing the two
            # flavors compared the whole tuple where an element was asked —
            # `isa(c, T)`/`c === nothing` over a destructured union-split
            # iterate result folded to Const(false)/Const(false), turning
            # the residual throw_methoderror arm into the taken path (the
            # StyledStrings termcolor manual-MethodError class).
            elem = -1
            if dk === K"extract" && begin
                   bo = UnifiedIR.getop(ir, d, 1)
                   UnifiedIR.optag(bo) == UnifiedIR.TAG_STMT &&
                       UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(bo)) === K"if"
               end
                elem = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, d, 2))::Int64)
                elem >= 1 || continue
                d = UnifiedIR.asstmt(UnifiedIR.getop(ir, d, 1))
                dk = K"if"
            elseif dk === K"if"
                elem = 0
            end
            if dk === K"select" ||
               (dk === K"call" && UnifiedIR.nops(ir, d) == 4 &&
                static_operand_value(ir, UnifiedIR.getop(ir, d, 1)) === Core.ifelse)
                ofs = dk === K"select" ? 0 : 1
                co = UnifiedIR.getop(ir, d, 1 + ofs)
                push!(armlats, stmt_lattice(ir, UnifiedIR.getop(ir, d, 2 + ofs)))
                push!(armlats, stmt_lattice(ir, UnifiedIR.getop(ir, d, 3 + ofs)))
                subj = i
                break
            elseif dk === K"if" && elem >= 0
                arms = result_arms(ir, d)
                (arms === nothing || isempty(arms)) && continue
                bad = false
                for (_, t) in arms
                    nres = UnifiedIR.nops(ir, t)
                    local al
                    if nres == 1
                        al = stmt_lattice(ir, UnifiedIR.getop(ir, t, 1))
                        if elem >= 1
                            # element projection into the arm's value
                            al = try
                                CC.getfield_tfunc(L, al, CC.Const(elem))
                            catch
                                nothing
                            end
                        end
                    elseif elem >= 1 && elem <= nres
                        al = stmt_lattice(ir, UnifiedIR.getop(ir, t, elem))
                    else
                        # elem == 0 over a multi-result if (its value is the
                        # synthetic escape tuple), or out-of-range: decline
                        al = nothing
                    end
                    (al === nothing || al === Union{}) && (bad = true; break)
                    push!(armlats, al)
                end
                bad && (empty!(armlats); continue)
                co = UnifiedIR.getop(ir, d, 1)
                subj = i
                break
            end
        end
        subj == 0 && continue
        CC.widenconst(stmt_lattice(ir, co)) === Bool || continue
        otherlat = stmt_lattice(ir, UnifiedIR.getop(ir, s, subj == 2 ? 3 : 2))
        answers = Bool[]
        ok = true
        for armlat in armlats
            r = callee === (===) ? CC.egal_tfunc(L, armlat, otherlat) :
                callee === isa ? CC.isa_tfunc(L, armlat, otherlat) :
                CC.isdefined_tfunc(L, armlat, otherlat)
            (r isa CC.Const && r.val isa Bool) || (ok = false; break)
            push!(answers, r.val::Bool)
        end
        (ok && !isempty(answers)) || continue
        if length(answers) == 1
            # the other arm diverges: the comparison's value is unconditional
            UnifiedIR.replace_stmt!(ir, s, K"refine", UnifiedIR.vop(ir, answers[1]);
                                    type = CC.Const(answers[1]))
            n += 1
            continue
        end
        length(answers) == 2 || continue
        # the subject's condition must be reusable at the comparison site
        if UnifiedIR.optag(co) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(co), s) || continue
        end
        UnifiedIR.replace_stmt!(ir, s, K"select", co,
                                UnifiedIR.vop(ir, answers[1]), UnifiedIR.vop(ir, answers[2]);
                                type = answers[1] == answers[2] ?
                                       CC.Const(answers[1]) : Bool)
        n += 1
    end
    return n
end

"""
    fold_retuples!(ir) -> Int

Tuple identity: `Core.tuple(extract(x, 1), …, extract(x, n))` over the
SAME `x` whose fixed-arity tuple type has exactly `n` components is `x`
itself (stock SROA's re-tupling case). The result forwards through a
`refine`.
"""
function fold_retuples!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        nop >= 2 || continue
        static_operand_value(ir, UnifiedIR.getop(ir, s, 1)) === Core.tuple || continue
        local xo::UnifiedIR.Operand
        ok = true
        for i in 2:nop
            o = UnifiedIR.getop(ir, s, i)
            UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || (ok = false; break)
            d = skip_refines(ir, UnifiedIR.asstmt(o))
            UnifiedIR.stmt_kind(ir, d) === K"extract" || (ok = false; break)
            Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, d, 2))::Int64) == i - 1 ||
                (ok = false; break)
            b = UnifiedIR.getop(ir, d, 1)
            if i == 2
                xo = b
            else
                b == xo || (ok = false; break)
            end
        end
        ok || continue
        xt = CC.widenconst(stmt_lattice(ir, xo))
        (xt isa DataType && xt <: Tuple && xt !== Tuple) || continue
        ps = xt.parameters
        length(ps) == nop - 1 || continue
        Base.any(p -> CC.isvarargtype(p), ps) && continue
        if UnifiedIR.optag(xo) == UnifiedIR.TAG_STMT
            UnifiedIR.visible(ir, UnifiedIR.asstmt(xo), s) || continue
        end
        UnifiedIR.replace_stmt!(ir, s, K"refine", xo;
                                type = UnifiedIR.stmt_type(ir, s))
        n += 1
    end
    return n
end

"""
    fold_splatnews!(ir) -> Int

Stock `inline_splatnew`: a `splatnew(T, t)` whose result type has a known
field count and whose splatted operand is a local `Core.tuple(a...)` of
exactly that arity becomes `new(T, a...)` (the abstract-`NamedTuple`
keyword-argument path). The type operand may stay dynamic — `new`
performs the same runtime field-type checks `splatnew` would.
"""
function fold_splatnews!(ir::UnifiedIR.IR)
    n = 0
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"splatnew" || continue
        UnifiedIR.nops(ir, s) == 2 || continue
        rt = UnifiedIR.stmt_type(ir, s)
        nf = CC.nfields_tfunc(CC.fallback_lattice, rt isa Type ? rt : CC.widenconst(rt))
        (nf isa CC.Const && nf.val isa Int) || continue
        tupop = UnifiedIR.getop(ir, s, 2)
        UnifiedIR.optag(tupop) == UnifiedIR.TAG_STMT || continue
        def = skip_refines(ir, UnifiedIR.asstmt(tupop))
        UnifiedIR.stmt_kind(ir, def) === K"call" || continue
        static_operand_value(ir, UnifiedIR.getop(ir, def, 1)) === Core.tuple || continue
        UnifiedIR.nops(ir, def) - 1 == nf.val || continue
        elems = UnifiedIR.Operand[UnifiedIR.getop(ir, def, i)
                                  for i in 2:UnifiedIR.nops(ir, def)]
        ok = true
        for el in elems
            UnifiedIR.optag(el) == UnifiedIR.TAG_STMT || continue
            UnifiedIR.visible(ir, UnifiedIR.asstmt(el), s) || (ok = false; break)
        end
        ok || continue
        UnifiedIR.replace_stmt!(ir, s, K"new", UnifiedIR.getop(ir, s, 1), elems...;
                                type = rt)
        n += 1
    end
    return n
end

# ---------------------------------------------------------------------------
# Post-optimization escape-analysis consumption (stock `ipo_dataflow_analysis!`
# / `refine_effects!`'s EA half, optimize.jl:699-803): `:effect_free`
# refinement for bodies whose only remaining taints are argmem-only writes
# (`EFFECT_FREE_IF_INACCESSIBLEMEMONLY` callees / setfield!) on provably
# non-escaping local allocations. Runs on the optimized dense IR, after the
# final inference pass (`FLAG_NOTHROW`/`FLAG_EFFECT_FREE` columns fresh),
# consuming `analyze_escapes` (escape.jl, the B4 API) with
#   * `EAOptSummarizer` — the `get_escape_cache` seam: recursive, memoized
#     per-MethodInstance argument-escape summaries (world-stamped module
#     memo; the driver-level CodeInstance-keyed cache is the A6 seam — see
#     the note at EA_OPT_SUMMARIES);
#   * a `resolve_call` hook — unified declined-inline candidates are still
#     `call`s at this point (stock's inliner has rewritten them to
#     `:invoke`), so statically-resolved residual calls get the same
#     interprocedural treatment without rewriting the IR.
# ---------------------------------------------------------------------------

# Per-MethodInstance argument-escape summaries for the optimizer's EA runs.
# World-stamped: any method (re)definition bumps the world counter, so a
# stamp mismatch empties the memo — conservative but sound (the summary of a
# body depends on facts that a redefinition can invalidate). ACTIVE guards
# cycles and recursion depth. A6 SEAM: the durable home for these summaries
# is the driver's CodeInstance cache (`stack_analysis_result!`-style, edges
# decay them precisely, cross-session reuse); this module-level memo is the
# self-contained interim, keyed the same way stock's protocol expects
# (`get_escape_cache(codeinst) -> Union{Bool,UArgEscapeCache}`).
const EA_OPT_LOCK = Base.ReentrantLock()
const EA_OPT_SUMMARIES = IdDict{Core.MethodInstance,Any}()
const EA_OPT_WORLD = Base.RefValue{UInt}(0)
const EA_OPT_ACTIVE = Base.IdSet{Core.MethodInstance}()
const EA_OPT_MAX_ACTIVE = 4
const EA_OPT_MAX_SRC_STMTS = 200
const EA_OPT_MAX_BODY_STMTS = 2048

"`get_escape_cache` for optimizer-integrated EA (stock GetNativeEscapeCache
shape): CodeInstance ipo-effects fast path, then the recursive summarizer."
struct EAOptSummarizer
    st::UInferState
end

function (S::EAOptSummarizer)(@nospecialize codeinst)
    if codeinst isa Core.CodeInstance
        effects = CC.decode_effects(codeinst.ipo_purity_bits)
        if CC.is_effect_free(effects) && CC.is_inaccessiblememonly(effects)
            # nothing escapes through a fully effect-free, memory-inaccessible
            # callee (stock's simple-frame fast path)
            return true
        end
    end
    mi = codeinst isa Core.CodeInstance ? codeinst.def :
         codeinst isa Core.MethodInstance ? codeinst : nothing
    mi isa Core.MethodInstance || return false
    return ea_opt_summary(S.st, mi)
end

function ea_opt_summary(st::UInferState, mi::Core.MethodInstance)
    st.cfg.frame_budget >= 1000 || return false
    world = st.cfg.world
    # trylock: same-task reentry succeeds (ReentrantLock); a cross-thread
    # race conservatively skips the cache AND the computation (false =
    # unknown callee) rather than risking a lock cycle on a compile path
    trylock(EA_OPT_LOCK) || return false
    try
        if EA_OPT_WORLD[] != world
            empty!(EA_OPT_SUMMARIES)
            EA_OPT_WORLD[] = world
        end
        haskey(EA_OPT_SUMMARIES, mi) && return EA_OPT_SUMMARIES[mi]
        (mi in EA_OPT_ACTIVE || length(EA_OPT_ACTIVE) >= EA_OPT_MAX_ACTIVE) &&
            return false
        opt_work_take!() || return false
        push!(EA_OPT_ACTIVE, mi)
        t0 = time_ns()
        r = try
            ea_opt_summary_uncached(st, mi)
        catch
            false
        finally
            delete!(EA_OPT_ACTIVE, mi)
            DRIVER_PHASES.ea_tower += Int(time_ns() - t0)
        end
        EA_OPT_SUMMARIES[mi] = r
        return r
    finally
        unlock(EA_OPT_LOCK)
    end
end

function ea_opt_summary_uncached(st::UInferState, mi::Core.MethodInstance)
    m = mi.def
    m isa Method || return false
    (m.isva || isdefined(m, :generator)) && return false
    sig = mi.specTypes
    sig isa DataType || return false
    ps = collect(Any, sig.parameters)
    length(ps) == Int(m.nargs) || return false
    Base.any(p -> CC.isvarargtype(p), ps) && return false
    src = Base.uncompressed_ir(m)
    length(src.code) <= EA_OPT_MAX_SRC_STMTS || return false
    ir = codeinfo_to_ir(src; nargs = Int(m.nargs), name = m.name)
    ir.meta[:method_instance] = mi
    ir.meta[:slotnames] = src.slotnames
    ir.sptypes = Any[t for t in mi.sparam_vals]
    ir.meta[:sptypes_lat] = sptypes_lattice(mi)
    # the full optimizer (inlining included): accessor wrappers
    # (setindex!/setproperty!/convert chains) must dissolve to raw
    # setfield!/getfield for the field-precise summary — a raw lowered body
    # keeps the field symbol behind a dynamic argument and the analysis
    # collapses to ⊤ (the same reason EAUtils analyzes post-inlining IR).
    # Tower frame cap (see TOWER_FRAME_CAP): bounded walk, no summary when
    # it fires (a cut walk's escape view would be optimistic)
    lim0 = st.limited
    prevcap = TOWER_FRAME_CAP[]
    ir = try
        TOWER_FRAME_CAP[] = TOWER_FRAME_BUDGET[]
        optimize_ir!(ir, ps; state = st, inline = true)
    finally
        TOWER_FRAME_CAP[] = prevcap
    end
    st.limited > lim0 && return false
    nargs = length(UnifiedIR.getregion(ir, UnifiedIR.root_region(ir)).args)
    res = analyze_escapes(ir, nargs; get_escape_cache = EAOptSummarizer(st),
                          resolve_call = ea_opt_resolver(ir, st))
    return UArgEscapeCache(res.state)
end

"""Resolve a residual generic `call` statement to its unique dispatch target
(single, unambiguous, fully-covering match — `resolve_single_match`'s
soundness gate; the lookup is edge-recorded through `st`), or nothing.
`Type`-valued callee operands whose `singleton_type` declines (TypeEq
`Type{X}` for non-singleton `X`) resolve through `CC.type_parameter` — the
constructor-through-Type-argument case."""
function ea_resolve_residual_call(ir::UnifiedIR.IR, st::UInferState, s::StmtId)
    UnifiedIR.stmt_kind(ir, s) === K"call" || return nothing
    nop = UnifiedIR.nops(ir, s)
    args = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i)) for i in 1:nop]
    f = CC.singleton_type(args[1])
    f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
    if f === nothing
        ft0 = CC.widenconst(args[1])
        if (ft0 isa DataType && CC.isType(ft0)) || CC.isTypeEq(ft0)
            p = CC.type_parameter(ft0)
            (p isa Type && !CC.has_free_typevars(p)) && (f = p)
        end
    end
    f isa Core.Builtin && return nothing
    f isa Core.IntrinsicFunction && return nothing
    f isa Core.TypeofVararg && return nothing
    local ftt
    if f === nothing
        # non-singleton concrete callee (closure objects): dispatch on the
        # concrete type; K"closure" activations stay with their machinery
        fo2 = UnifiedIR.getop(ir, s, 1)
        if UnifiedIR.optag(fo2) == UnifiedIR.TAG_STMT &&
           UnifiedIR.stmt_kind(ir, skip_refines(ir, UnifiedIR.asstmt(fo2))) === K"closure"
            return nothing
        end
        ft1 = CC.widenconst(args[1])
        (ft1 isa DataType && isconcretetype(ft1) && !(ft1 <: Type) &&
         !(ft1 <: Core.Builtin) && !(ft1 <: Core.IntrinsicFunction) &&
         !(ft1 <: Core.OpaqueClosure)) || return nothing
        ftt = ft1
    else
        ftt = f isa Type ? Type{f} : typeof(f)
    end
    argts = Any[CC.widenconst(a) for a in args[2:end]]
    Base.any(t -> !(t isa Type) || t === Union{}, argts) && return nothing
    fsig = try
        Tuple{ftt, argts...}
    catch
        return nothing
    end
    match = resolve_single_match(st, fsig)
    match === nothing && return nothing
    mi = try
        CC.specialize_method(match)
    catch
        return nothing
    end
    mi isa Core.MethodInstance || return nothing
    return mi
end

"Per-site-memoized `resolve_call` hook (the EA fixpoint revisits statements)."
function ea_opt_resolver(ir::UnifiedIR.IR, st::UInferState)
    memo = Dict{Int32,Any}()
    return function (s::StmtId)
        r = get!(memo, s.id) do
            ea_resolve_residual_call(ir, st, s)
        end
        return r isa Core.MethodInstance ? r : nothing
    end
end

# Kinds that never taint frame `:effect_free` on their own: values/reads,
# structure, terminators/control (region owners' observable work is their
# contained statements, scanned individually), frame-local cell machinery,
# and GC bookkeeping. `cell_set`/`cell_new` are handled separately (frame-
# local for plain `cell`s only), `try` bodies give the whole scan up
# (stock's EnterNode rule), everything else must carry FLAG_EFFECT_FREE or
# classify as an EA-refinable site.
function ea_scan_inert_kind(k::UnifiedIR.Kind)
    UnifiedIR.is_terminator(k) && return true
    return k === K"region_arg" || k === K"extract" || k === K"refine" ||
           k === K"value" || k === K"select" || k === K"globalref" ||
           k === K"isdefined_global" || k === K"copyast" || k === K"boundscheck" ||
           k === K"cell" || k === K"cell_shared" || k === K"cell_get" ||
           k === K"cell_isdefined" || k === K"throw_undef_if_not" ||
           k === K"gc_preserve_begin" || k === K"gc_preserve_end" ||
           k === K"if" || k === K"loop" || k === K"cfg" || k === K"closure"
end

"Is `s` a plain frame-local cell (not `cell_shared`)?"
function ea_plain_cell_target(ir::UnifiedIR.IR, s::StmtId)
    o = UnifiedIR.getop(ir, s, 1)
    UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || return false
    return UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(o)) === K"cell"
end

"""Classify one statement for the post-opt `:effect_free` scan. Returns
`true` (cannot taint), `false` (refuses the whole refinement), or pushes an
EA-validation site onto `pending` (an argmem-only-write site: an `invoke`
or a resolved residual `call` whose callee effects are
`EFFECT_FREE_IF_INACCESSIBLEMEMONLY`)."""
function ea_scan_stmt!(ir::UnifiedIR.IR, st::UInferState, s::StmtId,
                       k::UnifiedIR.Kind, pending::Vector{StmtId},
                       resolver)
    UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_EFFECT_FREE != 0 && return true
    ea_scan_inert_kind(k) && return true
    if k === K"cell_set" || k === K"cell_new"
        return ea_plain_cell_target(ir, s)
    end
    local effects::CC.Effects
    if k === K"invoke"
        tgt = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        if tgt isa Core.CodeInstance
            effects = CC.decode_effects(tgt.ipo_purity_bits)
        elseif tgt isa Core.MethodInstance
            # target effects through the inference cache (memoized per st)
            r = try
                fr = Frame(UnifiedIR.Builder().ir, st, Any[])
                argl = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i))
                           for i in 2:UnifiedIR.nops(ir, s)]
                infer_call(fr, argl)
            catch
                nothing
            end
            r isa UResult || return false
            effects = r.effects
        else
            return false
        end
    elseif k === K"call"
        resolver(s) isa Core.MethodInstance || return false
        r = try
            fr = Frame(UnifiedIR.Builder().ir, st, Any[])
            argl = Any[stmt_lattice(ir, UnifiedIR.getop(ir, s, i))
                       for i in 1:UnifiedIR.nops(ir, s)]
            infer_call(fr, argl)
        catch
            nothing
        end
        r isa UResult || return false
        effects = r.effects
    else
        return false
    end
    CC.is_effect_free(effects) && return true
    CC.is_effect_free_if_inaccessiblememonly(effects) || return false
    push!(pending, s)
    return true
end

"""Stock `check_all_args_noescape!` on unified IR: every mutable-typed value
operand of `s` must be (a) a caller argument with no escape — the refinement
then caps at `EFFECT_FREE_IF_INACCESSIBLEMEMONLY` (`:argmem`) — or (b) a
non-escaping local allocation chain (`new`/`splatnew`/`invoke`/resolved
`call` defs, recursively). Returns `:ok`, `:argmem`, or `:fail`."""
function ea_check_args_noescape(ir::UnifiedIR.IR, estate, # ::UEscapeState (escape.jl loads after this file)
                                argset::Base.IdSet{StmtId}, resolver,
                                s::StmtId, depth::Int)
    depth > 16 && return :fail
    k = UnifiedIR.stmt_kind(ir, s)
    first_idx = (k === K"invoke" || k === K"new" || k === K"splatnew") ? 2 : 1
    res = :ok
    for i in first_idx:UnifiedIR.nops(ir, s)
        o = UnifiedIR.getop(ir, s, i)
        lat = stmt_lattice(ir, o)
        CC.is_mutation_free_argtype(lat) && continue
        UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || return :fail
        d = skip_refines(ir, UnifiedIR.asstmt(o))
        info = ignore_argescape(estate[d])
        has_no_escape(info) || return :fail
        if d in argset
            # a caller argument: even with everything else effect-free the
            # best claim is effect-free-if-argmem-only (stock's rule)
            res = :argmem
            continue
        end
        dk = UnifiedIR.stmt_kind(ir, d)
        if dk === K"new" || dk === K"splatnew" || dk === K"invoke" ||
           (dk === K"call" && resolver(d) isa Core.MethodInstance)
            r = ea_check_args_noescape(ir, estate, argset, resolver, d, depth + 1)
            r === :fail && return :fail
            r === :argmem && (res = :argmem)
        else
            return :fail
        end
    end
    return res
end

# Post-optimization callee effects: the driver publishes refined bits only
# into CodeInstances, so mid-pipeline consumers (frame-nothrow scan, the
# finalizer gate) see inference-grade callee effects that miss post-opt
# statement facts. Memoized per MethodInstance, world-stamped, cycle-guarded
# (a callee's optimization recursively consults ITS callees through the
# same helper).
const OPT_FX_MEMO = IdDict{Core.MethodInstance,Any}()
const OPT_FX_WORLD = Base.RefValue{UInt}(0)
const OPT_FX_ACTIVE = Base.IdSet{Core.MethodInstance}()

"""
    opt_callee_effects(st, mi) -> Union{Nothing,Effects}

`mi`'s frame effects at driver grade: the body optimized through this
pipeline (whose tail applies the post-opt refinements, recursively through
this helper), with the method's `@assume_effects` override applied.
"""
function opt_callee_effects(st::UInferState, mi::Core.MethodInstance)
    # cached-CodeInstance fast path (wave 9): the callee CI's published ipo
    # effects ARE driver grade — the driver publishes `refine_post_opt`ed
    # bits, stock entries carry stock's refined bits. The consumption is
    # soundness-relevant (nothrow feeds statement flags and DCE), so the CI
    # is recorded as an edge and the window clamped — the `ci_cache_serve`
    # protocol. Unbounded entries only; a failed clamp falls through.
    if CI_SERVE_ENABLED[]
        ci = get(Compiler.code_cache(st.cfg.interp), mi, nothing)
        if ci isa Core.CodeInstance && ci.max_world == typemax(UInt)
            col = st.edges
            okcol = true
            if col isa UEdges
                okcol = clamp_world!(col, ci.min_world, ci.max_world)
                if okcol
                    record_invoke!(col, nothing, ci)
                    trace!(col, (0x3, ci))
                end
            end
            okcol && return CC.decode_effects(ci.ipo_purity_bits)
        end
    end
    # reentrant/self-hosting passes skip the driver-grade recompute (see
    # inline2_cost's budget gate)
    st.cfg.frame_budget >= 1000 || return nothing
    trylock(EA_OPT_LOCK) || return nothing
    try
        world = st.cfg.world
        if OPT_FX_WORLD[] != world
            empty!(OPT_FX_MEMO)
            OPT_FX_WORLD[] = world
        end
        haskey(OPT_FX_MEMO, mi) && return OPT_FX_MEMO[mi]
        (mi in OPT_FX_ACTIVE || length(OPT_FX_ACTIVE) >= EA_OPT_MAX_ACTIVE) &&
            return nothing
        opt_work_take!() || return nothing
        push!(OPT_FX_ACTIVE, mi)
        t0 = time_ns()
        r = try
            opt_callee_effects_uncached(st, mi)
        catch
            nothing
        finally
            delete!(OPT_FX_ACTIVE, mi)
            DRIVER_PHASES.fx_tower += Int(time_ns() - t0)
        end
        OPT_FX_MEMO[mi] = r
        return r
    finally
        unlock(EA_OPT_LOCK)
    end
end

function opt_callee_effects_uncached(st::UInferState, mi::Core.MethodInstance)
    m = mi.def
    m isa Method || return nothing
    isdefined(m, :generator) && return nothing
    src = Base.uncompressed_ir(m)
    length(src.code) <= EA_OPT_MAX_SRC_STMTS || return nothing
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
    # tower frame cap (see TOWER_FRAME_CAP): bounded walk, no verdict when
    # it fires — a cut walk's effects would be unsound to publish anyway
    lim0 = st.limited
    prevcap = TOWER_FRAME_CAP[]
    ir = try
        TOWER_FRAME_CAP[] = TOWER_FRAME_BUDGET[]
        optimize_ir!(ir, ps; state = st, inline = true)
    finally
        TOWER_FRAME_CAP[] = prevcap
    end
    st.limited > lim0 && return nothing
    return apply_effects_override(m, frame_effects_meta(ir))
end

"""
    refine_frame_nothrow!(ir) -> Int

Post-optimization `:nothrow` refinement from the statement flag column:
when every live statement is individually FLAG_NOTHROW (which includes
`refine_effects!`'s statement-level facts inference's transfer rules do
not carry, e.g. the definedness-monotonicity carve-out for `getfield` on
Const-of-mutable subjects) — or is a terminator/control statement whose
only throw condition (a non-Bool branch condition) is excluded by its
operand type — the frame cannot throw. Upgrades `ir.meta[:effects]` and
zeroes `ir.meta[:exct]`. Conservative: any unflagged computational
statement refuses (statements inside `try` bodies included, though their
throws would be caught — inference already models that channel).
"""
function refine_frame_nothrow!(ir::UnifiedIR.IR, st::Union{UInferState,Nothing} = nothing)
    eff = frame_effects_meta(ir)
    CC.is_nothrow(eff) && return 0
    # statement-position static-parameter reads are ELIDED by the entry
    # converter (meta[:sparam_reads]) — their UndefVarError potential is a
    # frame-level fact with no statement to carry a flag
    let reads = get(ir.meta, :sparam_reads, nothing)
        if reads isa Vector{Int}
            for n in reads
                sparam_maybe_undef(ir, n) && return 0
            end
        end
    end
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_NOTHROW != 0 && continue
        k = UnifiedIR.stmt_kind(ir, s)
        if k === K"br_if" || k === K"continue" || k === K"if"
            co = UnifiedIR.getop(ir, s, k === K"continue" ? 2 : 1)
            t = CC.widenconst(stmt_lattice(ir, co))
            (t isa Type && t <: Bool) || return 0
        elseif k === K"return" || k === K"result" || k === K"break" ||
               k === K"goto" || k === K"unreachable" || k === K"region_arg" ||
               k === K"refine" || k === K"loop" || k === K"cfg" ||
               k === K"cell" || k === K"cell_shared" ||
               k === K"cell_new" || k === K"cell_set"
            # terminators/structure never throw; cell allocation/writes are
            # frame-local and total (`cell_get` is NOT here: a read of a
            # maybe-undefined residual cell can throw, so it must carry the
            # flag column's proof)
            continue
        elseif st !== nothing && (k === K"call" || k === K"invoke")
            # unflagged interprocedural site: nothrow at driver grade — the
            # callee's own post-opt refinement — when the target is exact
            # (CI/mi invoke, or a single unambiguous fully-covering match,
            # which also excludes the MethodError channel)
            local fx
            if k === K"invoke"
                tgt = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
                if tgt isa Core.CodeInstance
                    fx = CC.decode_effects(tgt.ipo_purity_bits)
                elseif tgt isa Core.MethodInstance
                    fx = opt_callee_effects(st, tgt)
                else
                    return 0
                end
            else
                cmi = ea_resolve_residual_call(ir, st, s)
                cmi isa Core.MethodInstance || return 0
                fx = opt_callee_effects(st, cmi)
            end
            (fx isa CC.Effects && CC.is_nothrow(fx)) || return 0
        else
            return 0
        end
    end
    ir.meta[:effects] = eff = CC.Effects(eff; nothrow = true)
    ir.meta[:effects_mask] = effects_mask(eff)
    ir.meta[:exct] = Union{}
    return 1
end

"""
    ea_refine_effect_free!(ir, st) -> Int

The post-optimization EA consumer (stock `refine_effects!`'s
`validate_mutable_arg_escapes!` half): when the frame's remaining
`:effect_free` taints are exactly argmem-only-write sites on provably
non-escaping local allocations, upgrade `ir.meta[:effects]`'s
`effect_free` to `ALWAYS_TRUE` (or `EFFECT_FREE_IF_INACCESSIBLEMEMONLY`
when an argument's memory is written). Dense state, after the final
inference pass. Returns 1 when the meta effects were upgraded.

NOTE (driver seam): `refine_post_opt` currently forwards the post-opt
`effect_free` axis only when it is `ALWAYS_TRUE`; the `:argmem` outcome
(`EFFECT_FREE_IF_INACCESSIBLEMEMONLY` over an `ALWAYS_FALSE` base) needs
the same per-axis upgrade there to become IPO-visible.
"""
function ea_refine_effect_free!(ir::UnifiedIR.IR, st::UInferState)
    st.cfg.frame_budget >= 1000 || return 0
    eff = frame_effects_meta(ir)
    CC.is_effect_free(eff) && return 0
    UnifiedIR.nstmts(ir) <= EA_OPT_MAX_BODY_STMTS || return 0
    resolver = ea_opt_resolver(ir, st)
    pending = StmtId[]
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        k === K"try" && return 0     # exception paths not modeled (stock rule)
        ea_scan_stmt!(ir, st, s, k, pending, resolver) || return 0
    end
    isempty(pending) && return 0
    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    nargs = length(root.args)
    res = try
        analyze_escapes(ir, nargs; get_escape_cache = EAOptSummarizer(st),
                        resolve_call = resolver)
    catch
        return 0
    end
    estate = res.state
    argset = Base.IdSet{StmtId}()
    for a in root.args
        push!(argset, a)
    end
    argmem = false
    for s in pending
        r = ea_check_args_noescape(ir, estate, argset, resolver, s, 0)
        r === :fail && return 0
        r === :argmem && (argmem = true)
    end
    effect_free = argmem ? CC.EFFECT_FREE_IF_INACCESSIBLEMEMONLY : CC.ALWAYS_TRUE
    (argmem && eff.effect_free != CC.ALWAYS_FALSE) && return 0   # no upgrade
    ir.meta[:effects] = eff = CC.Effects(eff; effect_free)
    ir.meta[:effects_mask] = effects_mask(eff)
    return 1
end

"compact!, carrying the cell-name channel (meta[:cell_names], the undef-guard
variable names) across the statement renumbering."
function compact_carry_names!(ir::UnifiedIR.IR)
    names = get(ir.meta, :cell_names, nothing)
    fins = get(ir.meta, :finalizer_calls, nothing)
    ir, rs = UnifiedIR.compact!(ir)
    if names isa Dict{Int32,Symbol} && !isempty(names)
        newnames = Dict{Int32,Symbol}()
        for (id, nm) in names
            nid = 1 <= id <= length(rs.stmt) ? rs.stmt[id] : Int32(0)
            nid == 0 && continue                      # cell promoted away
            newnames[nid] = nm
        end
        ir.meta[:cell_names] = newnames
    end
    if fins isa Set{Int32} && !isempty(fins)
        # placed-finalizer ids (resolve_finalizers! → mutable-SROA load
        # forwarding) survive the renumbering the same way
        newfins = Set{Int32}()
        for id in fins
            nid = 1 <= id <= length(rs.stmt) ? rs.stmt[id] : Int32(0)
            nid == 0 || push!(newfins, nid)
        end
        ir.meta[:finalizer_calls] = newfins
    end
    return ir
end

"""
    optimize_ir!(ir, argtypes; state, inline=true, rounds=8, params) -> ir

The pipeline (§10.4), iterated to quiescence. Per round:

  dense:    inference → effects refinement (incl. `new` removability) →
            const materialization → getfield canonicalization → extract
            forwarding (immutable SROA) → refine forwarding → if-result
            forwarding → cell promotion (region-tree + single-region) → DCE
  editable: constant-branch folding → island branch folding → unreachable-
            block pruning → goto-chain merging → structurization (§10.5:
            if/loop recovery from islands) → island dissolution → loop-
            carried cell promotion → select conversion → mutable-struct
            SROA → region-op ADCE → inlining (calls + invokes) → union
            splitting
  compact! + verify (level 1)
"""
# Per-top-level-optimization work budget for the driver-grade callee
# machinery (cost model / post-opt effects / EA summaries): each memo MISS
# optimizes one callee body, and an unbounded transitive walk of a large
# callee graph (escape_string/print-family bodies) wedges a single query
# for minutes. The budget replenishes at every OUTERMOST optimize_ir! entry
# (nested entries are exactly those callee optimizations), so any one
# top-level body bounds its uncached exploration; memo hits are free, so
# warm sessions converge to full precision. Task-local soundness only —
# these are admission heuristics, a miscount under thread races just
# shifts where the fallback heuristic takes over.
const OPT_NEST_DEPTH = Base.RefValue(0)
const OPT_WORK_LEFT = Base.RefValue(0)
const OPT_WORK_BUDGET = Base.RefValue(64)

"Take one unit of callee-optimization budget (false = exhausted).
Unbudgeted outside a pipeline invocation (direct tool/test queries)."
function opt_work_take!()
    OPT_NEST_DEPTH[] == 0 && return true
    OPT_WORK_LEFT[] > 0 || return false
    OPT_WORK_LEFT[] -= 1
    return true
end

function optimize_ir!(ir::UnifiedIR.IR, argtypes::Vector{Any};
                      state::UInferState = UInferState(), inline::Bool = true,
                      rounds::Int = 8, params::InlineParams = InlineParams())
    OPT_NEST_DEPTH[] == 0 && (OPT_WORK_LEFT[] = OPT_WORK_BUDGET[])
    OPT_NEST_DEPTH[] += 1
    try
        return _optimize_ir!(ir, argtypes; state, inline, rounds, params)
    finally
        OPT_NEST_DEPTH[] -= 1
    end
end

function _optimize_ir!(ir::UnifiedIR.IR, argtypes::Vector{Any};
                       state::UInferState = UInferState(), inline::Bool = true,
                       rounds::Int = 8, params::InlineParams = InlineParams())
    lastspliced = 0
    for round in 1:rounds
        changed = 0
        infer_ir!(ir, argtypes; state)
        changed += refine_effects!(ir)
        changed += materialize_consts!(ir)
        changed += canonicalize_getfields!(ir)
        changed += forward_extracts!(ir)
        changed += fold_retuples!(ir)
        changed += fold_splatnews!(ir)
        changed += fold_pure_queries!(ir)
        changed += lift_keyvalue_gets!(ir)
        changed += lift_svec_refs!(ir)
        changed += dedup_selects!(ir)
        changed += forward_refines!(ir)
        changed += forward_if_results!(ir)
        changed += UnifiedIR.promote_cells!(ir)
        changed += promote_block_cells!(ir)
        # never-observed cells (no get/isdefined/escape) are dead wherever
        # they sit — including island blocks, where every promotion pass
        # refuses store-only/declaration-only cells and `dce!` structurally
        # cannot reach them (`cell_set`/`cell_new` have no result, and the
        # declaration keeps a use). Raw late-round callee splices strand
        # exactly this shape once folding deletes the slot's reads.
        changed += drop_dead_cells!(ir)
        changed += UnifiedIR.dce!(ir)
        UnifiedIR.editable(ir)
        _, folded = UnifiedIR.fold_constant_branches!(ir)
        changed += folded
        changed += fold_island_branches!(ir)
        changed += drop_unreachable_blocks!(ir)
        changed += merge_goto_chains!(ir)
        changed += structurize!(ir)
        changed += dissolve_islands!(ir)
        # joint cell-promotion fixpoint (§6 join completeness, docs
        # "Join-point completeness"): arm-join sinking turns conditional arm
        # stores into unconditional post-join stores, which loop promotion
        # consumes as carried values and (next round) promote_cells! as
        # dominating stores — and each can expose new cases for the others.
        while true
            c = promote_undef_cells!(ir)
            c += promote_arm_cells!(ir)
            c += UnifiedIR.promote_try_cells!(ir)
            c += promote_island_cells!(ir)
            c += promote_loop_cells!(ir)
            c == 0 && break
            changed += c
        end
        changed += fold_uniform_block_args!(ir)
        changed += selectify!(ir)
        changed += fold_isdefineds!(ir)
        changed += resolve_finalizers!(ir, state)
        changed += sroa_mutables!(ir)
        changed += adce_region_ops!(ir)
        lastspliced = 0
        if inline
            lastspliced += fold_apply_iterates!(ir)
            lastspliced += inline_calls2!(ir, state; params)
            lastspliced += union_split_calls!(ir, state; params)
            changed += lastspliced
        end
        ir = compact_carry_names!(ir)
        UnifiedIR.verify_ir(ir; level = 1)
        changed == 0 && break
    end
    settled_spliced = 0
    if inline && get(ir.meta, :sparam_deferred, false) === true
        # settled-types sparam materialization (stock ir_prepare_inlining!'s
        # spvals_ssa regime): sites whose specialization still carries
        # unbakeable static parameters after the iterative rounds have
        # refined every type get the `_compute_sparams`/`_svec_ref`
        # reconstruction splice — deferred to here so an early round's
        # under-refined match never burns the precise inline (the cleanup
        # rounds below run the lift/DCE over the spliced result)
        delete!(ir.meta, :sparam_deferred)
        infer_ir!(ir, argtypes; state)
        UnifiedIR.editable(ir)
        settled_spliced = inline_calls2!(ir, state; params, materialize_sparams = true)
        lastspliced += settled_spliced
        ir = compact_carry_names!(ir)
    end
    if settled_spliced > 0
        # a settled-phase splice can EXPOSE work only inlining discharges —
        # a finalizer registration inside the spliced ctor body: resolve
        # and place it now, load-forward against the placement, and give
        # the placed call one inline pass (the DoAllocNoEscapeSparam
        # shape). Scoped to actual placements so ordinary sparam splices
        # keep the inline-free cleanup below (an extra global inline round
        # here regrows settled bodies and flips admission shapes — the
        # wave-11 merge_fallback lesson).
        infer_ir!(ir, argtypes; state)
        UnifiedIR.editable(ir)
        if resolve_finalizers!(ir, state) > 0
            sroa_mutables!(ir)
            inline_calls2!(ir, state; params, materialize_sparams = true)
        end
        ir = compact_carry_names!(ir)
    end
    if lastspliced > 0
        # a final round that spliced callee bodies never saw the cleanup
        # passes (raw entry-converted bodies arrive as cfg islands; leaving
        # them undissolved strands island cells the promotion suite refuses)
        # — run the pipeline without inlining until the shapes settle
        ir = _optimize_ir!(ir, argtypes; state, inline = false, rounds = 3, params)
    end
    # the round budget is shared with inlining, so callee cells spliced by a
    # late round may never have seen the promotion passes: give promotion its
    # own fixpoint (cheap when there is nothing left to do), then one DCE for
    # the stores it strands. This is the substrate's shared driver — the same
    # entry lowering's closure-capture analysis runs.
    UnifiedIR.promote_fixpoint!(ir; stmt_value = _stmt_const_value)
    drop_dead_cells!(ir)
    UnifiedIR.dce!(ir)
    ir = compact_carry_names!(ir)
    infer_ir!(ir, argtypes; state)
    # post-final-inference constant sweep: when the round budget ends on a
    # still-changing body (large const-foldable chains), the last inference
    # can leave freshly-proven Consts unmaterialized; one more
    # refine+materialize+DCE round makes the fold visible to the exit
    if refine_effects!(ir) + materialize_consts!(ir) > 0
        UnifiedIR.dce!(ir)
        ir = compact_carry_names!(ir)
        infer_ir!(ir, argtypes; state)
    end
    # post-opt refinements (stock ipo_dataflow_analysis! analogs): frame
    # nothrow from the statement flag column, and the EA-backed
    # :effect_free upgrade for argmem-only writes on provably
    # non-escaping local allocations. The flag column is re-established
    # first: every infer_ir! pass republishes inference's projection,
    # dropping refine_effects!'s statement-level carve-outs (e.g. the
    # Const-of-mutable getfield definedness fact) that the frame-level
    # scans consume.
    try
        refine_effects!(ir)
        refine_frame_nothrow!(ir, state)
        ea_refine_effect_free!(ir, state)
    catch
    end
    # residual dynamic applies: convert fixed-shape tuple containers to
    # Core.svec for the codegen apply ABI (stock lift_apply_args!, #59548).
    # Deliberately after the last inference pass — a svec-typed container
    # would only degrade the apply's abstract flattening if re-inferred.
    # The rewrite needs an editable window (insert_before!/replace_stmt!);
    # the IR is dense here, so only open one when a residual apply exists.
    if any(s -> UnifiedIR.stmt_kind(ir, s) === K"call" &&
                UnifiedIR.nops(ir, s) >= 4 &&
                static_operand_value(ir, UnifiedIR.getop(ir, s, 1)) === Core._apply_iterate,
           UnifiedIR.each_stmt(ir))
        UnifiedIR.editable(ir)
        svecify_apply_args!(ir)
        ir, _ = UnifiedIR.compact!(ir)
        UnifiedIR.dce!(ir)   # the replaced tuple ctors are dead now (dense-legal here)
    end
    UnifiedIR.verify_ir(ir; level = 1)
    return ir
end
