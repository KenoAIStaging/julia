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
        push!(EA_OPT_ACTIVE, mi)
        r = try
            ea_opt_summary_uncached(st, mi)
        catch
            false
        finally
            delete!(EA_OPT_ACTIVE, mi)
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
    # collapses to ⊤ (the same reason EAUtils analyzes post-inlining IR)
    ir = optimize_ir!(ir, ps; state = st, inline = true)
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
    (f === nothing || f isa Core.Builtin || f isa Core.IntrinsicFunction) &&
        return nothing
    f isa Core.TypeofVararg && return nothing
    argts = Any[CC.widenconst(a) for a in args[2:end]]
    Base.any(t -> !(t isa Type) || t === Union{}, argts) && return nothing
    fsig = try
        Tuple{f isa Type ? Type{f} : typeof(f), argts...}
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
function refine_frame_nothrow!(ir::UnifiedIR.IR)
    eff = frame_effects_meta(ir)
    CC.is_nothrow(eff) && return 0
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
               k === K"cell" || k === K"cell_shared" || k === K"cell_get" ||
               k === K"cell_new" || k === K"cell_set" || k === K"cell_isdefined"
            # terminators/structure never throw; frame-local cell machinery
            # reads/writes defined slots (undef reads go through the guarded
            # `throw_undef_if_not` form, which is not in this list)
            continue
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
    # post-opt refinements (stock ipo_dataflow_analysis! analogs): frame
    # nothrow from the statement flag column, and the EA-backed
    # :effect_free upgrade for argmem-only writes on provably
    # non-escaping local allocations
    try
        refine_frame_nothrow!(ir)
        ea_refine_effect_free!(ir, state)
    catch
    end
    UnifiedIR.verify_ir(ir; level = 1)
    return ir
end
