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
        CC.is_nothrow(effects) && (flags |= UnifiedIR.FLAG_NOTHROW)
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
    foldable = UnifiedIR.FLAG_CONSISTENT | UnifiedIR.FLAG_EFFECT_FREE |
               UnifiedIR.FLAG_NOTHROW | UnifiedIR.FLAG_TERMINATES
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
        UnifiedIR.replace_uses!(ir, s => UnifiedIR.vop(ir, v))
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
        UnifiedIR.replace_stmt!(ir, s, K"extract", vo, UnifiedIR.op_inline(idx);
                                type = UnifiedIR.stmt_type(ir, s))
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

"Rewrite the field-`fidx` load `s` over per-arm constructed values
(`armpairs` = (result stmt, value operand) per live arm of `ifop`) to
`select`/`refine` when every arm value is an arm-local immutable
new/`Core.tuple` whose element operand is visible at `s`. True on success."
function forward_arm_elements!(ir::UnifiedIR.IR, s::StmtId, ifop::StmtId,
                               armpairs::Vector{Tuple{StmtId,UnifiedIR.Operand}},
                               fidx::Int)
    els = UnifiedIR.Operand[]
    for (_, ro) in armpairs
        UnifiedIR.optag(ro) == UnifiedIR.TAG_STMT || return false
        ad = skip_refines(ir, UnifiedIR.asstmt(ro))
        adk = UnifiedIR.stmt_kind(ir, ad)
        local el::UnifiedIR.Operand
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
            return false
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
            # uniform-constant position: fold outright
            v0 = nothing
            uniform = true
            for (_, t) in arms
                idx <= UnifiedIR.nops(ir, t) || (uniform = false; break)
                v = static_operand_value(ir, UnifiedIR.getop(ir, t, idx))
                v === nothing && (uniform = false; break)
                ismutable(v) && !(v isa Union{Type,Function,Module,Symbol,String}) &&
                    (uniform = false; break)
                v0 === nothing ? (v0 = v) : (v === v0 || (uniform = false; break))
            end
            if uniform && v0 !== nothing
                UnifiedIR.replace_stmt!(ir, s, K"refine", UnifiedIR.vop(ir, v0);
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
                k <= UnifiedIR.nops(ir, t) || (armok = false; break)
                push!(armpairs, (t, UnifiedIR.getop(ir, t, k)))
            end
            armok || continue
            forward_arm_elements!(ir, s, if2, armpairs, idx) && (n += 1)
            continue
        elseif dk === K"call"
            callee = static_operand_value(ir, UnifiedIR.getop(ir, def, 1))
            callee === Core.tuple || continue
            1 + idx <= UnifiedIR.nops(ir, def) || continue
            el = UnifiedIR.getop(ir, def, idx + 1)
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
            pos = 0
            if dk === K"extract" && begin
                   bo = UnifiedIR.getop(ir, d, 1)
                   UnifiedIR.optag(bo) == UnifiedIR.TAG_STMT &&
                       UnifiedIR.stmt_kind(ir, UnifiedIR.asstmt(bo)) === K"if"
               end
                pos = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, d, 2))::Int64)
                d = UnifiedIR.asstmt(UnifiedIR.getop(ir, d, 1))
                dk = K"if"
            elseif dk === K"if"
                pos = 1
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
            elseif dk === K"if" && pos >= 1
                arms = result_arms(ir, d)
                (arms === nothing || isempty(arms)) && continue
                bad = false
                for (_, t) in arms
                    pos <= UnifiedIR.nops(ir, t) || (bad = true; break)
                    push!(armlats, stmt_lattice(ir, UnifiedIR.getop(ir, t, pos)))
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

"compact!, carrying the cell-name channel (meta[:cell_names], the undef-guard
variable names) across the statement renumbering."
function compact_carry_names!(ir::UnifiedIR.IR)
    names = get(ir.meta, :cell_names, nothing)
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
function optimize_ir!(ir::UnifiedIR.IR, argtypes::Vector{Any};
                      state::UInferState = UInferState(), inline::Bool = true,
                      rounds::Int = 8, params::InlineParams = InlineParams())
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
        changed += forward_refines!(ir)
        changed += forward_if_results!(ir)
        changed += UnifiedIR.promote_cells!(ir)
        changed += promote_block_cells!(ir)
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
        if inline
            changed += fold_apply_iterates!(ir)
            changed += inline_calls2!(ir, state; params)
            changed += union_split_calls!(ir, state; params)
        end
        ir = compact_carry_names!(ir)
        UnifiedIR.verify_ir(ir; level = 1)
        changed == 0 && break
    end
    # the round budget is shared with inlining, so callee cells spliced by a
    # late round may never have seen the promotion passes: give promotion its
    # own fixpoint (cheap when there is nothing left to do), then one DCE for
    # the stores it strands. This is the substrate's shared driver — the same
    # entry lowering's closure-capture analysis runs.
    UnifiedIR.promote_fixpoint!(ir; stmt_value = _stmt_const_value)
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
    UnifiedIR.verify_ir(ir; level = 1)
    return ir
end
