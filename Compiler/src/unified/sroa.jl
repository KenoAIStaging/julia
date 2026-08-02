# SROA on UnifiedIR (§10.4; stock reference: Compiler/src/ssair/passes.jl
# `sroa_pass!`/`sroa_mutables!` — the CASES, not the mechanics):
#
#   1. Immutable-struct SROA: `extract` (canonicalized getfield) of a locally
#      constructed `Core.tuple`/`K"new"` of a concrete immutable type forwards
#      the field value, following `refine` chains; legality of the forwarded
#      operand at the use site is checked with `UnifiedIR.visible`.
#      (Extension of `forward_extracts!`, which lives in optimize.jl.)
#   2. If-result forwarding: an `if` whose live arms all produce the same
#      operand forwards that operand to the result's uses (the phi-of-one-
#      value case of stock SROA lifting).
#   3. Mutable-struct SROA: a `new` of a mutable struct that never escapes
#      (uses are only getfield/extract loads and setfield! stores with
#      constant fields) becomes per-field cells (K"cell" + cell_set/cell_get);
#      `UnifiedIR.promote_cells!` + `dce!` then clean up.
#   4. Dead `new` elimination falls out of `dce!` once `refine_effects!`
#      marks nothrow constructions REMOVABLE (optimize.jl).

"Field index (1-based) for a constant field designator (Int or Symbol), or nothing."
function field_index_of(@nospecialize(T), @nospecialize(fld))
    T isa DataType || return nothing
    if fld isa Int
        1 <= fld <= fieldcount(T) || return nothing
        return fld
    elseif fld isa Symbol
        fi = Base.fieldindex(T, fld, false)
        return fi == 0 ? nothing : fi
    end
    return nothing
end

"Concrete DataType of a lattice element/operand type, or nothing."
function concrete_datatype(@nospecialize(tl))
    T = tl isa CC.Const ? tl.val : CC.singleton_type(CC.widenconst(tl))
    if T === nothing
        wt = CC.widenconst(tl)
        wt isa DataType && isconcretetype(wt) && (T = wt)
    end
    T isa DataType && isconcretetype(T) || return nothing
    return T
end

"Skip through K\"refine\" chains to the underlying definition."
function skip_refines(ir::UnifiedIR.IR, def::StmtId)
    steps = 0
    while UnifiedIR.stmt_kind(ir, def) === K"refine" && (steps += 1) <= 32
        o = UnifiedIR.getop(ir, def, 1)
        UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || break
        def = UnifiedIR.asstmt(o)
    end
    return def
end
# ---------------------------------------------------------------------------
# Definite-initialization analysis for uninitialized-field SROA
# ---------------------------------------------------------------------------
#
# `getfield` of an uninitialized field throws `UndefRefError`; a promoted
# cell read must never reach that state, so a load of a field the `new` did
# not supply is only convertible when EVERY path from the allocation to the
# load passes a store. Two structural cases prove it (both piggyback on the
# promotion suite's editable-state dominance helpers, §6 throw-edge rules
# included — handler crossings refuse, throws exit the region so a path
# that continues past a store-bearing region prefix has executed it):
#
#   (a) a store dominates the load (`_cell_dominates_ed` + `comes_before`);
#   (b) an `if` op dominates the load and every one of its (two, non-guard)
#       live arms definitely stores the field — directly or through a
#       nested all-arms-store `if` (recursively).

function _store_dominates(ir::UnifiedIR.IR, stores::Vector{StmtId}, site::StmtId)
    for st in stores
        UnifiedIR.comes_before(ir, st, site) || continue
        UnifiedIR._cell_dominates_ed(ir, st, site) && return true
    end
    return false
end

function _arm_definitely_stores(ir::UnifiedIR.IR, stores::Vector{StmtId},
                                arm::RegionId, depth::Int)
    reg = UnifiedIR.getregion(ir, arm)
    reg.kind === UnifiedIR.REGION_ARM || return false
    UnifiedIR.is_guard(reg) && return false
    for m in UnifiedIR.region_stmts(ir, arm)
        UnifiedIR.is_tombstone(ir, m) && continue
        m in stores && return true
        if UnifiedIR.stmt_kind(ir, m) === K"if" &&
           _if_all_arms_store(ir, stores, m, depth + 1)
            return true
        end
    end
    return false
end

function _if_all_arms_store(ir::UnifiedIR.IR, stores::Vector{StmtId}, I::StmtId,
                            depth::Int)
    depth > 16 && return false
    arms = UnifiedIR.live_owned_regions(ir, I)
    length(arms) == 2 || return false
    return _arm_definitely_stores(ir, stores, arms[1], depth) &&
           _arm_definitely_stores(ir, stores, arms[2], depth)
end

"Every path from the allocation to `site` passes one of `stores` (see above)."
function definitely_initialized(ir::UnifiedIR.IR, stores::Vector{StmtId}, site::StmtId)
    isempty(stores) && return false
    _store_dominates(ir, stores, site) && return true
    for I in UnifiedIR.each_stmt(ir)
        UnifiedIR.is_tombstone(ir, I) && continue
        UnifiedIR.stmt_kind(ir, I) === K"if" || continue
        UnifiedIR.comes_before(ir, I, site) || continue
        UnifiedIR._cell_dominates_ed(ir, I, site) || continue
        _if_all_arms_store(ir, stores, I, 0) && return true
    end
    return false
end

"""
    sroa_mutables!(ir) -> Int

Mutable-struct SROA (§10.4 / stock `sroa_mutables!` cases): `new` of a
concrete mutable struct whose value never escapes — every use is
`extract`/`getfield(it, const fld)` or `setfield!(it, const fld, v)` — is
replaced by per-field frame cells. Loads become `cell_get`, stores become
`cell_set` (+ a `refine` carrying setfield!'s value result). Uninitialized
trailing fields are admitted when every load of such a field is definitely
initialized (see the analysis above); otherwise the allocation keeps memory
form (an unprovable load must keep `getfield`'s `UndefRefError`). Editable
state; `promote_cells!`/`dce!` finish the job on the next dense round.
"""
function sroa_mutables!(ir::UnifiedIR.IR)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "sroa_mutables!")
    promoted = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"new" || continue
        T = concrete_datatype(stmt_lattice(ir, UnifiedIR.getop(ir, s, 1)))
        (T isa DataType && ismutabletype(T)) || continue
        nf = fieldcount(T)
        nsupplied = UnifiedIR.nops(ir, s) - 1
        nsupplied <= nf || continue                   # over-arity `new` throws
        any(i -> Base.isfieldatomic(T, i), 1:nf) && continue
        # inside a cfg island the replacement cells could never promote
        # (promote_cells! §6 policy refuses island cells) — a pure
        # pessimization; leave the allocation in memory form there
        UnifiedIR.inside_island(ir, s) && continue
        # collect uses; any non-load/store use disqualifies (escape check).
        # One exception: a finalizer call resolve_finalizers! placed this
        # session (ir.meta[:finalizer_calls]) is a LIFETIME-END use, not an
        # escape — it runs only once the object is unreachable, so loads
        # may still forward; the allocation itself must stay (memory mode).
        fset = get(ir.meta, :finalizer_calls, nothing)
        loads = Tuple{StmtId,Int}[]
        stores = Tuple{StmtId,Int}[]
        ok = true
        finuse = false
        UnifiedIR.each_ssa_use(ir) do site, used
            (ok && used == s) || return
            site isa UnifiedIR.StmtOperand || (ok = false; return)
            u = site.user
            UnifiedIR.is_tombstone(ir, u) && return
            uk = UnifiedIR.stmt_kind(ir, u)
            if uk === K"extract" && site.opidx == 1
                idx = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, u, 2))::Int64)
                1 <= idx <= nf ? push!(loads, (u, idx)) : (ok = false)
            elseif uk === K"call"
                callee = static_operand_value(ir, UnifiedIR.getop(ir, u, 1))
                nopu = UnifiedIR.nops(ir, u)
                if (callee === Core.getfield || callee === Base.getfield) &&
                   site.opidx == 2 && (nopu == 3 || nopu == 4)
                    fld = field_index_of(T, static_operand_value(ir, UnifiedIR.getop(ir, u, 3)))
                    fld === nothing ? (ok = false) : push!(loads, (u, fld))
                elseif (callee === Core.setfield! || callee === Base.setfield!) &&
                       site.opidx == 2 && nopu == 4
                    fld = field_index_of(T, static_operand_value(ir, UnifiedIR.getop(ir, u, 3)))
                    fld === nothing ? (ok = false) : push!(stores, (u, fld))
                elseif fset isa Set{Int32} && u.id in fset && site.opidx == 2
                    finuse = true
                else
                    ok = false
                end
            else
                ok = false   # cell_set, return, result, phi-ish, nested call arg, …
            end
        end
        ok || continue
        if finuse
            promoted += forward_mutable_loads_only!(ir, s, loads, stores, nsupplied)
            continue
        end
        if nsupplied < nf
            # every load of an uninitialized field must be provably
            # initialized on all paths, or the whole allocation stays
            for (u, fld) in loads
                fld <= nsupplied && continue
                fstores = StmtId[st for (st, f2) in stores if f2 == fld]
                definitely_initialized(ir, fstores, u) || (ok = false; break)
            end
            ok || continue
        end
        # a promoted store must behave like the setfield!/new it replaces:
        # a value the field type does not admit would have thrown TypeError
        for (u, fld) in stores
            vt = CC.widenconst(stmt_lattice(ir, UnifiedIR.getop(ir, u, 4)))
            (vt isa Type && vt <: fieldtype(T, fld)) || (ok = false; break)
        end
        ok || continue
        for i in 1:nsupplied
            vt = CC.widenconst(stmt_lattice(ir, UnifiedIR.getop(ir, s, i + 1)))
            (vt isa Type && vt <: fieldtype(T, i)) || (ok = false; break)
        end
        ok || continue
        # rewrite: per-field cells (+ initial stores for supplied fields),
        # placed just before the new
        cells = StmtId[]
        for i in 1:nf
            ft = fieldtype(T, i)
            c = UnifiedIR.insert_before!(ir, s, K"cell", UnifiedIR.vop(ir, ft); type = ft)
            push!(cells, c)
            i <= nsupplied || continue
            UnifiedIR.insert_before!(ir, s, K"cell_set", UnifiedIR.op_stmt(c),
                                     UnifiedIR.getop(ir, s, i + 1))
        end
        for (u, fld) in loads
            UnifiedIR.replace_stmt!(ir, u, K"cell_get", UnifiedIR.op_stmt(cells[fld]);
                                    type = UnifiedIR.stmt_type(ir, u))
        end
        for (u, fld) in stores
            vo = UnifiedIR.getop(ir, u, 4)
            UnifiedIR.insert_before!(ir, u, K"cell_set", UnifiedIR.op_stmt(cells[fld]), vo)
            # setfield! evaluates to the stored value; keep that result shape
            UnifiedIR.replace_stmt!(ir, u, K"refine", vo; type = UnifiedIR.stmt_type(ir, u))
        end
        UnifiedIR.kill_stmt!(ir, s)
        promoted += 1
    end
    return promoted
end

"""Load forwarding WITHOUT elimination for a mutable `new` whose lifetime
ends in a placed finalizer call (stock's finalizer-elision load-forwarding
corpus): the allocation and its stores stay in memory form — the finalizer
body may read the fields at death — but program-order loads see the last
program-order store, because the finalizer runs only once the object is
unreachable (no load can observe it). v1: straight-line only — the `new`,
every load and every store must share one region; an ordered walk tracks
the current per-field value and rewrites each covered load to a `refine`
of it."""
function forward_mutable_loads_only!(ir::UnifiedIR.IR, s::StmtId,
                                     loads::Vector{Tuple{StmtId,Int}},
                                     stores::Vector{Tuple{StmtId,Int}},
                                     nsupplied::Int)
    homer = UnifiedIR.stmt_region(ir, s)
    for (u, _) in loads
        UnifiedIR.stmt_region(ir, u) == homer || return 0
    end
    for (u, _) in stores
        UnifiedIR.stmt_region(ir, u) == homer || return 0
    end
    cur = Dict{Int,UnifiedIR.Operand}()
    for i in 1:nsupplied
        cur[i] = UnifiedIR.getop(ir, s, i + 1)
    end
    loadmap = Dict{Int32,Int}(u.id => f for (u, f) in loads)
    storemap = Dict{Int32,Int}(u.id => f for (u, f) in stores)
    n = 0
    started = false
    for st in UnifiedIR.region_stmts(ir, homer)
        if st == s
            started = true
            continue
        end
        started || continue
        f = get(storemap, st.id, nothing)
        if f !== nothing
            cur[f] = UnifiedIR.getop(ir, st, 4)
            continue
        end
        f = get(loadmap, st.id, nothing)
        if f !== nothing
            v = get(cur, f, nothing)
            v === nothing && continue
            UnifiedIR.replace_stmt!(ir, st, K"refine", v;
                                    type = UnifiedIR.stmt_type(ir, st))
            n += 1
        end
    end
    return n
end

# ---------------------------------------------------------------------------
# isdefined folding over local allocations
# ---------------------------------------------------------------------------

"""
    fold_isdefineds!(ir) -> Int

`isdefined(x, fld)` where `x` traces to a local `new` folds to `true`
when the field was supplied at construction or SOME `setfield!` of that
field dominates the query (stock's isdefined elimination). Definedness is
MONOTONE — stores only add it — so the fold is sound even when the
object escapes (unknown code can only store more). The indeterminate and
never-stored cases are left for the full SROA analysis (which knows the
complete use set). Editable state.
"""
function fold_isdefineds!(ir::UnifiedIR.IR)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "fold_isdefineds!")
    n = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        nop = UnifiedIR.nops(ir, s)
        (nop == 3 || nop == 4) || continue
        callee = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        (callee === Core.isdefined || callee === Base.isdefined) || continue
        if nop == 4
            static_operand_value(ir, UnifiedIR.getop(ir, s, 4)) === :not_atomic || continue
        end
        xo = UnifiedIR.getop(ir, s, 2)
        UnifiedIR.optag(xo) == UnifiedIR.TAG_STMT || continue
        obj = skip_refines(ir, UnifiedIR.asstmt(xo))
        UnifiedIR.stmt_kind(ir, obj) === K"new" || continue
        T = concrete_datatype(stmt_lattice(ir, UnifiedIR.getop(ir, obj, 1)))
        T isa DataType || continue
        fld = field_index_of(T, static_operand_value(ir, UnifiedIR.getop(ir, s, 3)))
        fld === nothing && continue
        proven = fld <= UnifiedIR.nops(ir, obj) - 1
        if !proven
            for u in collect(UnifiedIR.each_stmt(ir))
                UnifiedIR.is_tombstone(ir, u) && continue
                UnifiedIR.stmt_kind(ir, u) === K"call" || continue
                UnifiedIR.nops(ir, u) == 4 || continue
                c2 = static_operand_value(ir, UnifiedIR.getop(ir, u, 1))
                (c2 === Core.setfield! || c2 === Base.setfield!) || continue
                so = UnifiedIR.getop(ir, u, 2)
                UnifiedIR.optag(so) == UnifiedIR.TAG_STMT || continue
                skip_refines(ir, UnifiedIR.asstmt(so)) == obj || continue
                field_index_of(T, static_operand_value(ir, UnifiedIR.getop(ir, u, 3))) == fld || continue
                (UnifiedIR.comes_before(ir, u, s) &&
                 UnifiedIR._cell_dominates_ed(ir, u, s)) || continue
                proven = true
                break
            end
        end
        proven || continue
        UnifiedIR.replace_stmt!(ir, s, K"refine", UnifiedIR.vop(ir, true);
                                type = CC.Const(true))
        n += 1
    end
    return n
end

# ---------------------------------------------------------------------------
# Finalizer resolution (stock `try_resolve_finalizer!` cases)
# ---------------------------------------------------------------------------

"""
    resolve_finalizers!(ir, st) -> Int

`Core.finalizer(f, obj)` registrations the optimizer can discharge
(stock ssair/passes.jl `try_resolve_finalizer!` — the CASES):

  * `f`'s call effects for `obj`'s type are removable-if-unused → the
    finalizer can never do observable work: the registration is erased
    (no escape analysis needed — this is legal for escaping objects too).
  * `f` is finalizer-inlineable (nothrow ∧ notaskstate) and `obj` is a
    non-escaping local `new` of a mutable type whose only other uses —
    counting the ones reached through the registration's own result, since
    `Base.finalizer(f, o)` returns `o` — are field loads/stores: the
    registration is erased and `f(obj)` is placed
    right after the last use's top-level container in the allocation's
    home region (the statically-known end of the object's lifetime; v1
    requires the registration itself to sit in that region, so the call
    runs exactly on the executions that registered it). The placed call
    then inlines on later rounds, exposing loads that mutable SROA
    scalarizes — the allocation disappears entirely.

Editable state. Effects queries run through the unified inference
machinery on `st` (edge-recorded in driver mode).
"""
function resolve_finalizers!(ir::UnifiedIR.IR, st::UInferState)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_EDITABLE, "resolve_finalizers!")
    n = 0
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.is_tombstone(ir, s) && continue
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        UnifiedIR.nops(ir, s) == 3 || continue
        callee = static_operand_value(ir, UnifiedIR.getop(ir, s, 1))
        (callee === Core.finalizer || callee === Base.finalizer) || continue
        fo = UnifiedIR.getop(ir, s, 2)
        f = static_operand_value(ir, fo)
        f === nothing && continue
        objo = UnifiedIR.getop(ir, s, 3)
        UnifiedIR.optag(objo) == UnifiedIR.TAG_STMT || continue
        obj = UnifiedIR.asstmt(objo)
        objT = CC.widenconst(stmt_lattice(ir, objo))
        # sparam-typed allocations (`new{T}` through a materialized
        # apply_type) carry a UnionAll: mutability is a property of the
        # wrapped datatype (the DoAllocNoEscapeSparam shape, wave 11)
        objTd = objT isa UnionAll ? Base.unwrap_unionall(objT) : objT
        (objTd isa DataType && ismutabletype(objTd)) || continue
        r = try
            fr = Frame(UnifiedIR.Builder().ir, st, Any[])
            infer_call(fr, Any[CC.Const(f), objT])
        catch
            nothing
        end
        r === nothing && continue
        fx = r.effects
        if !(CC.is_removable_if_unused(fx) || Compiler.is_finalizer_inlineable(fx))
            # inference-grade effects miss post-opt statement facts (the
            # driver publishes refined bits only into CodeInstances): retry
            # the gate at driver grade through the memoized post-opt helper
            fsig = try
                Tuple{f isa Type ? Type{f} : typeof(f), objT}
            catch
                nothing
            end
            match = fsig === nothing ? nothing : resolve_single_match(st, fsig)
            if match !== nothing
                mi2 = try
                    CC.specialize_method(match)
                catch
                    nothing
                end
                if mi2 isa Core.MethodInstance
                    fx2 = opt_callee_effects(st, mi2)
                    fx2 isa CC.Effects && (fx = fx2)
                end
            end
        end
        if CC.is_removable_if_unused(fx)
            UnifiedIR.replace_stmt!(ir, s, K"refine", objo;
                                    type = UnifiedIR.stmt_type(ir, s))
            n += 1
            continue
        end
        Compiler.is_finalizer_inlineable(fx) || continue
        UnifiedIR.stmt_kind(ir, obj) === K"new" || continue
        homer = UnifiedIR.stmt_region(ir, obj)
        UnifiedIR.stmt_region(ir, s) == homer || continue
        # escape check: every other use is a field load/store, or a GC
        # lifetime marker (`GC.@preserve` roots the object without leaking
        # it — stock EA's no-escape classification; the paired
        # `gc_preserve_end`s join the use set below so the placed call
        # lands after the preserved span)
        #
        # The scan runs over the object's ALIASES, not just the `new`:
        # `Base.finalizer(f, o)` RETURNS `o`, so the registration's own
        # result carries the object onward, and a `refine` over an alias is
        # the same value. Skipping the registration without following its
        # result would make an object that reaches the rest of the program
        # only through it look DEAD at the registration — exactly the
        # `finalizer(new(x)) do this … end` constructor, whose body is
        # optimized on its own before it is inlined: the anchor stays at
        # `s`, the call is placed right after the registration, and every
        # caller the constructor is inlined into then runs the finalizer
        # BEFORE its own `setfield!`s, observing the constructor's initial
        # field values (the cfg_finalization2/6/7 shapes).
        ok = true
        uses = StmtId[]
        aliases = StmtId[obj]
        # `Core.finalizer` evaluates to `nothing`; only the Base wrapper
        # forwards the object
        callee === Base.finalizer && push!(aliases, s)
        ai = 0
        while ai < length(aliases)
            ai += 1
            a = aliases[ai]
            UnifiedIR.each_ssa_use(ir) do site, used
                (ok && used == a) || return
                site isa UnifiedIR.StmtOperand || (ok = false; return)
                u = site.user
                UnifiedIR.is_tombstone(ir, u) && return
                u == s && return
                uk = UnifiedIR.stmt_kind(ir, u)
                if uk === K"refine" && site.opidx == 1
                    # bound the alias walk: one `each_ssa_use` sweep apiece,
                    # and declining is always sound
                    length(aliases) < 16 || (ok = false; return)
                    any(==(u), aliases) || push!(aliases, u)
                elseif uk === K"extract" && site.opidx == 1
                    push!(uses, u)
                elseif uk === K"gc_preserve_begin"
                    push!(uses, u)
                elseif uk === K"call"
                    callee2 = static_operand_value(ir, UnifiedIR.getop(ir, u, 1))
                    nopu = UnifiedIR.nops(ir, u)
                    if (callee2 === Core.getfield || callee2 === Base.getfield) &&
                       site.opidx == 2 && (nopu == 3 || nopu == 4)
                        push!(uses, u)
                    elseif (callee2 === Core.setfield! || callee2 === Base.setfield!) &&
                           site.opidx == 2 && nopu == 4
                        push!(uses, u)
                    else
                        ok = false
                    end
                else
                    ok = false
                end
            end
            ok || break
        end
        ok || continue
        # a preserved object's lifetime extends to the preserve END: add
        # each begin's paired end(s) so the anchor computation sees them
        for u in copy(uses)
            UnifiedIR.stmt_kind(ir, u) === K"gc_preserve_begin" || continue
            UnifiedIR.each_ssa_use(ir) do site2, used2
                used2 == u || return
                site2 isa UnifiedIR.StmtOperand || return
                e = site2.user
                UnifiedIR.is_tombstone(ir, e) && return
                UnifiedIR.stmt_kind(ir, e) === K"gc_preserve_end" && push!(uses, e)
            end
        end
        # anchor = the flat-last use's top-level container within the home
        # region (uses inside ifs/loops resolve to the owning op; handler
        # positions refuse — a throw path must keep the GC-time semantics)
        anchor = s
        for u in uses
            c = u
            rr = UnifiedIR.stmt_region(ir, c)
            steps = 0
            while rr != homer
                (steps += 1) <= UnifiedIR.nregions(ir) || (ok = false; break)
                reg = UnifiedIR.getregion(ir, rr)
                reg.kind === UnifiedIR.REGION_HANDLER && (ok = false; break)
                c = reg.owner
                c.id == 0 && (ok = false; break)
                rr = UnifiedIR.stmt_region(ir, c)
            end
            ok || break
            UnifiedIR.comes_before(ir, anchor, c) && (anchor = c)
        end
        ok || continue
        members = UnifiedIR.region_stmts(ir, homer)
        idx = findfirst(==(anchor), members)
        idx === nothing && continue
        idx < length(members) || continue
        placed = UnifiedIR.insert_before!(ir, members[idx + 1], K"call", fo, objo; type = Any)
        # record the placement for this editable session: mutable SROA
        # treats it as a lifetime-end use (load forwarding stays legal —
        # the finalizer only runs once the object is unreachable), not an
        # escape. The id set is session-transient (dropped at compact!).
        fset = get!(() -> Set{Int32}(), ir.meta, :finalizer_calls)::Set{Int32}
        push!(fset, placed.id)
        UnifiedIR.replace_stmt!(ir, s, K"refine", objo;
                                type = UnifiedIR.stmt_type(ir, s))
        n += 1
    end
    return n
end

# ---------------------------------------------------------------------------
# The cell-promotion mem2reg suite lives in the substrate now
# (UnifiedIR/src/promote.jl) so lowering's capture analysis runs the SAME
# machinery. Bind the names this module's passes, tests, and harnesses use;
# `promote_loop_cells!` gets the inferred-Const reader as its static-value
# hook (the substrate default cannot see the type lattice).
# ---------------------------------------------------------------------------

const forward_if_results!   = UnifiedIR.forward_if_results!
const promote_block_cells!  = UnifiedIR.promote_block_cells!
const promote_arm_cells!    = UnifiedIR.promote_arm_cells!
const promote_island_cells! = UnifiedIR.promote_island_cells!
const promote_undef_cells!  = UnifiedIR.promote_undef_cells!
const PROMOTION_TRACE       = UnifiedIR.PROMOTION_TRACE
const _in_handler           = UnifiedIR._in_handler
const is_diverge_kind       = UnifiedIR.is_diverge_kind

"Inferred-Const static value of a statement (the lattice-aware `stmt_value` hook)."
function _stmt_const_value(ir::UnifiedIR.IR, s::StmtId)
    tt = UnifiedIR.stmt_type(ir, s)
    tt isa CC.Const && return tt.val
    return CC.singleton_type(tt isa Type ? tt : Any)
end

promote_loop_cells!(ir::UnifiedIR.IR) =
    UnifiedIR.promote_loop_cells!(ir; stmt_value = _stmt_const_value)

# `promote_fixpoint!` is exported by UnifiedIR (visible here via `using`);
# call it with `stmt_value = _stmt_const_value` to give it the lattice.
