# This file is a part of Julia. License is MIT: https://julialang.org/license

# Bifurcation of functions on global-type speculation (#8870).
#
# A function that reads an untyped global whose binding carries a speculated type `S`
# (see `Speculated` and `SpeculatedGlobalAccessInfo`) is rewritten so that the *entire
# body* is versioned on a single entry guard, instead of guarding every use:
#
#     %1 = isdefinedglobal(M, :x)
#          goto GENERIC if not %1
#     %3 = M.x                              # the ONE read
#     %4 = %3 isa S
#          goto GENERIC if not %4
#          _virt = %3                       # fresh slot; FAST's view of the binding
#          enter CATCH                      # only when FAST contains stores
#     FAST:    copy of the body in which reads of the binding forward from `_virt`,
#              stores become `_virt = v` (write-after-write elided), and every return
#              materializes the pending value through the ordinary checked
#              `setglobal!` path
#     CATCH:   setglobal!(M, :x, _virt); rethrow    # materialize on unwind edges
#     GENERIC: verbatim copy of the body (the universal deoptimization target)
#
# Reads and writes of a non-atomic binding may assume the absence of concurrent
# accesses, which is what licenses collapsing all reads into one and eliding
# write-after-write: single-threaded observable behavior is preserved exactly (the
# unwind-edge and return materializations keep the binding's value equal to what
# original program order would have produced at every point a return or escaping
# exception can observe it), and the only cross-thread relaxation is that a racing
# reader may miss intermediate values of a racing writer. Re-materializing the value
# that was read (when no store has executed yet) is unobservable, so the
# materializations are unconditional. A store that has become erroneous (the binding
# was re-declared no longer writable) raises at the materialization point instead of
# at the original store's position; that reordering is likewise covered by the
# license, since only another thread can change the binding out from under us.
#
# Both guards fire before any side effect, and nothing else enters GENERIC, so no
# live state needs to be transferred at the side exit. Statements that could observe
# the deferral -- anything that may read or write global bindings -- make the pass
# bail out entirely; the per-call-site resolution of `SpeculatedCallInfo` in the
# inlining pass remains as the fallback strategy for those functions.
#
# The pass runs on the inferred, slotted `CodeInfo` before `convert_to_ircode!`, so
# that `slot2reg` constructs every φ (and the PhiC/Upsilon web carrying `_virt` into
# the catch block) for us. Since `type_annotate!` already widened all `Speculated`
# elements out of the statement types, the pass re-derives the fast copy's types with
# a small forward fixpoint seeded from the recorded speculations
# (`SpeculatedGlobalAccessInfo` for reads, `SpeculatedCallInfo.spec_rt` for calls),
# propagated through slot assignments. The results type the fast copy's statements
# and refine the per-block variable tables that `construct_ssa!` uses to type φs and
# π-narrow slot uses -- which is what keeps a loop-carried global unboxed.

const BIF_FLAGS_EFFECTS = IR_FLAG_CONSISTENT | IR_FLAG_EFFECT_FREE | IR_FLAG_NOTHROW |
    IR_FLAG_TERMINATES | IR_FLAG_NOUB | IR_FLAG_NORTCALL | IR_FLAG_EFIIMO |
    IR_FLAG_INACCESSIBLEMEM_OR_ARGMEM

const BIF_FLAGS_SIMPLE = IR_FLAG_EFFECT_FREE | IR_FLAG_NOTHROW | IR_FLAG_TERMINATES

# development aid: report why the pass refused a function
const BIF_DEBUG = RefValue{Bool}(false)
bif_debug_bail(i::Int, reason::String) = BIF_DEBUG[] && println("bifurcation bail at stmt ", i, ": ", reason)

# A weak `Core.declare_global` of a binding that is already declared (a runtime
# no-op) or not declared at all (backdated guard->DECLARED, no world bump) cannot
# touch any binding's value slot, so it cannot observe a deferred store; the
# `:latestworld` marker lowering pairs with it is inert for the same reason (see
# `latestworld_is_inert`). A weak declaration shadowing an implicit import is a real
# world event and fails this test, which bails the pass via its marker.
function bif_noop_weak_declare(world::UInt, @nospecialize(stmt))
    isexpr(stmt, :call) || return false
    args = (stmt::Expr).args
    length(args) == 4 || return false
    f = args[1]
    (isa(f, GlobalRef) && f.mod === Core && f.name === :declare_global) || return false
    (isa(args[2], Module) && isa(args[3], QuoteNode) && isa((args[3]::QuoteNode).value, Symbol) &&
     args[4] === false) || return false
    b = convert(Core.Binding, GlobalRef(args[2]::Module, (args[3]::QuoteNode).value::Symbol))
    kind = binding_kind(lookup_binding_partition(world, b))
    return kind == PARTITION_KIND_DECLARED || kind == PARTITION_KIND_GLOBAL ||
        kind == PARTITION_KIND_GUARD || is_some_const_binding(kind)
end

function bif_leaf_binding(world::UInt, b::Core.Binding)
    partition = lookup_binding_partition(world, b)
    return first(walk_to_leaf_partition(b, partition, world))
end

# Can a value of type `t` transitively reach a `Module` (and through it binding
# memory)? Arguments that cannot are safe under `IR_FLAG_INACCESSIBLEMEM_OR_ARGMEM`:
# a callee restricted to argument memory cannot observe or modify any binding through
# them.
function bif_module_free(@nospecialize t)
    t = widenconst(t)
    if isvarargtype(t)
        t = unwrapva(t)
        t === Union{} && return true
    end
    if isa(t, Union)
        return bif_module_free(t.a) && bif_module_free(t.b)
    end
    isa(t, DataType) || return false
    isbitstype(t) && return true
    (t === String || t === Symbol || t === Nothing) && return true
    if t <: Tuple
        isconcretetype(t) || return false
        for p in t.parameters
            bif_module_free(p) || return false
        end
        return true
    end
    if isconcretetype(t) && (t <: Array || t <: GenericMemory || t <: GenericMemoryRef || t <: Ref)
        for p in t.parameters
            (isa(p, Type) && !bif_module_free(p)) && return false
        end
        return true
    end
    # singleton (empty immutable) types, e.g. `typeof(sin)`
    isconcretetype(t) && !ismutabletype(t) && datatype_fieldcount(t) == 0 && return true
    return false
end

mutable struct BifurcationPlan
    gr::GlobalRef                     # the read spelling used for the guard load
    leaf::Core.Binding
    spec                              # guarded speculation; a Type
    store_gr::Union{Nothing,GlobalRef} # store spelling (identical across all stores)
    reads::BitSet                     # `GlobalRef` stmts of the binding (incl. `:(=)` rhs)
    stores::BitSet                    # own `setglobal!` call stmts
    returns::BitSet
    function BifurcationPlan(gr::GlobalRef, leaf::Core.Binding, @nospecialize(spec))
        return new(gr, leaf, spec, nothing, BitSet(), BitSet(), BitSet())
    end
end

# A cheap, sound type for a syntactic value position in slotted code (`nothing` when
# unknown). Constant bindings resolve to their value: replacing the constant bumps the
# world and invalidates this compilation through the ordinary binding edges.
function bif_syntactic_type(@nospecialize(x), ssavaluetypes::Vector{Any}, slottypes::Vector{Any},
                            world::UInt)
    if isa(x, SSAValue)
        return ssavaluetypes[x.id]
    elseif isa(x, SlotNumber) || isa(x, Argument)
        return slottypes[slot_id(x)]
    elseif isa(x, GlobalRef)
        b = convert(Core.Binding, x)
        leafb, partition = walk_to_leaf_partition(b, lookup_binding_partition(world, b), world)
        is_defined_const_binding(binding_kind(partition)) || return nothing
        return Const(partition_restriction(partition))
    elseif isa(x, Expr)
        return nothing
    elseif isa(x, QuoteNode)
        return Const(x.value)
    end
    return Const(x)
end

# Classify every statement; fills the plan, or returns `false` if any statement could
# observe the deferral (or is otherwise beyond this pass).
function bif_classify!(plan::BifurcationPlan, ci::CodeInfo, sv::OptimizationState, world::UInt)
    code = ci.code
    ssavaluetypes = ci.ssavaluetypes::Vector{Any}
    ssaflags = ci.ssaflags
    stmt_info = sv.stmt_info
    leaf = plan.leaf
    for i = 1:length(code)
        stmt = code[i]
        i in sv.unreachable && continue # copied verbatim, never executes
        if isa(stmt, ReturnNode)
            push!(plan.returns, i)
            continue
        end
        if stmt === nothing || isa(stmt, NewvarNode) || isa(stmt, GotoNode) ||
           isa(stmt, GotoIfNot) || isa(stmt, SlotNumber) || isa(stmt, Argument) ||
           isa(stmt, SSAValue) || isa(stmt, QuoteNode)
            continue
        end
        if isa(stmt, EnterNode) # user try/catch: leave to the fallback
            bif_debug_bail(i, "enter")
            return false
        end
        if isa(stmt, GlobalRef)
            if bif_leaf_binding(world, convert(Core.Binding, stmt)) === leaf
                info = stmt_info[i]
                # every speculated read must have seen the same speculation
                if isa(info, SpeculatedGlobalAccessInfo) && info.spec !== plan.spec
                    bif_debug_bail(i, "speculation mismatch")
                    return false
                end
                push!(plan.reads, i)
            end
            continue # reads of other bindings cannot observe a deferred store of ours
        end
        if !isa(stmt, Expr)
            bif_debug_bail(i, "unknown stmt kind")
            return false
        end
        head = stmt.head
        if head === :(=)
            rhs = stmt.args[2]
            if isa(rhs, GlobalRef)
                if bif_leaf_binding(world, convert(Core.Binding, rhs)) === leaf
                    info = stmt_info[i]
                    if isa(info, SpeculatedGlobalAccessInfo) && info.spec !== plan.spec
                        bif_debug_bail(i, "speculation mismatch (assign)")
                        return false
                    end
                    push!(plan.reads, i)
                end
                continue
            end
            isa(rhs, Expr) || continue # slot/ssa/literal copies
            # classify the rhs; an own-store never has its value used, so a
            # `:(=)`-wrapped `setglobal!` is deliberately not one
            stmt = rhs
            head = stmt.head
        end
        if head === :latestworld
            # inert markers (after no-op or backdated weak declarations) cannot
            # observe the deferral; a marker after a real world event must bail
            (i > 1 && bif_noop_weak_declare(world, code[i-1])) && continue
            return false
        end
        if head === :boundscheck || head === :meta || head === :loopinfo ||
           head === :code_coverage_effect || head === :inbounds || head === :isdefined ||
           head === :throw_undef_if_not || head === :copyast ||
           head === :gc_preserve_begin || head === :gc_preserve_end ||
           head === :new || head === :splatnew
            continue # cannot access binding memory
        end
        if head === :call || head === :invoke
            bif_noop_weak_declare(world, stmt) && continue # cannot touch any binding's value
            args = stmt.args
            firstarg = head === :invoke ? 2 : 1
            f = args[firstarg]
            if isa(f, GlobalRef) && f.mod === Core && !isexpr(code[i], :(=))
                # the direct-`Core` spellings lowering emits for global accesses; any
                # other spelling of these operations falls through to the effect test
                # below, which rejects it
                if f.name === :setglobal! && length(args) == 4 &&
                   isa(args[2], Module) && isa(args[3], QuoteNode) &&
                   isa((args[3]::QuoteNode).value, Symbol)
                    gr = GlobalRef(args[2]::Module, (args[3]::QuoteNode).value::Symbol)
                    if bif_leaf_binding(world, convert(Core.Binding, gr)) === leaf
                        # all stores must share one spelling, so a single
                        # materialization sequence reproduces their semantics
                        if plan.store_gr === nothing
                            plan.store_gr = gr
                        elseif !(plan.store_gr.mod === gr.mod && plan.store_gr.name === gr.name)
                            bif_debug_bail(i, "mixed store spellings")
                            return false
                        end
                        push!(plan.stores, i)
                        continue
                    end
                end
            end
            flags = ssaflags[i]
            info = stmt_info[i]
            spec_argtypes = nothing
            if isa(info, SpeculatedCallInfo)
                # in the fast copy, the speculated inference is what executes
                flags = (flags & ~BIF_FLAGS_EFFECTS) | flags_for_effects(info.spec_effects)
                spec_argtypes = info.spec_argtypes
            end
            if isa(ssavaluetypes[i], Const) && has_flag(flags, IR_FLAG_EFFECT_FREE) &&
               has_flag(flags, IR_FLAG_NOTHROW)
                continue # result is fixed and nothing is mutated: cannot observe us
            end
            if has_flag(flags, IR_FLAG_INACCESSIBLEMEM_OR_ARGMEM)
                argtypes_ok = true
                for k = firstarg:length(args)
                    at = spec_argtypes !== nothing && k - firstarg + 1 <= length(spec_argtypes) ?
                        spec_argtypes[k-firstarg+1] :
                        bif_syntactic_type(args[k], ssavaluetypes, sv.slottypes, world)
                    if at === nothing || !bif_module_free(at)
                        argtypes_ok = false
                        break
                    end
                end
                argtypes_ok && continue # binding memory unreachable from this call
                bif_debug_bail(i, "IAM call with module-reaching argument")
                return false
            end
            bif_debug_bail(i, "effectful call")
            return false
        end
        bif_debug_bail(i, "unknown head")
        return false # :foreigncall, :method, :latestworld, unknown heads, ...
    end
    return !isempty(plan.reads)
end

# Forward fixpoint computing the fast copy's statement types and per-block slot entry
# types, with the binding's virtual slot seeded to the guarded speculation and updated
# by (elided) stores. Returns `(fast_ssatypes, block_entries)` where each block entry
# is a `Vector{Any}` of length `nslots + 1` (the virtual slot last), or `nothing` for
# blocks inference found unreachable.
function bif_fast_types(ci::CodeInfo, sv::OptimizationState, plan::BifurcationPlan,
                        𝕃::AbstractLattice)
    code = ci.code
    ssavaluetypes = ci.ssavaluetypes::Vector{Any}
    stmt_info = sv.stmt_info
    cfg = sv.cfg
    nslots = length(sv.slottypes)
    nvirt = nslots + 1
    nbb = length(cfg.blocks)
    entries = Union{Nothing,Vector{Any}}[nothing for _ = 1:nbb]
    entry1 = Vector{Any}(undef, nvirt)
    bb1 = sv.bb_states[1]
    for k = 1:nslots
        entry1[k] = bb1 === nothing ? sv.slottypes[k] : bb1.vartable[k].typ
    end
    entry1[nvirt] = plan.spec
    entries[1] = entry1
    fast_ssatypes = copy(ssavaluetypes)
    virt_join = plan.spec # join of every value the virtual slot can hold
    state = Vector{Any}(undef, nvirt)
    worklist = Int[1]
    while !isempty(worklist)
        bb = pop!(worklist)
        blockentry = entries[bb]::Vector{Any}
        copyto!(state, blockentry)
        for i in cfg.blocks[bb].stmts
            i in sv.unreachable && continue
            stmt = code[i]
            local t
            if i in plan.reads
                t = state[nvirt]
            elseif i in plan.stores
                t = bif_fast_value_type((stmt::Expr).args[4], state, fast_ssatypes, sv.slottypes)
                state[nvirt] = t
                virt_join = tmerge(𝕃, virt_join, t)
            elseif isa(stmt, SlotNumber)
                t = state[slot_id(stmt)]
                t === Union{} && (t = ssavaluetypes[i]) # e.g. possibly-undef read
            elseif isexpr(stmt, :(=))
                t = bif_fast_rhs_type(stmt.args[2], i, state, fast_ssatypes, ssavaluetypes,
                                      sv.slottypes, stmt_info)
            else
                info = stmt_info[i]
                if isa(info, SpeculatedCallInfo)
                    t = info.spec_rt
                else
                    t = ssavaluetypes[i]
                end
            end
            t = bif_refine(𝕃, t, ssavaluetypes[i])
            if isexpr(stmt, :(=))
                state[slot_id(stmt.args[1]::SlotNumber)] = t
            end
            fast_ssatypes[i] = t
        end
        for succ in cfg.blocks[bb].succs
            succentry = entries[succ]
            if succentry === nothing
                entries[succ] = copy(state)
                push!(worklist, succ)
            else
                changed = false
                for k = 1:nvirt
                    tm = tmerge(𝕃, succentry[k], state[k])
                    if !(tm === succentry[k] || is_lattice_equal(𝕃, tm, succentry[k]))
                        succentry[k] = tm
                        changed = true
                    end
                end
                changed && push!(worklist, succ)
            end
        end
    end
    return fast_ssatypes, entries, virt_join
end

# The fixpoint's slot propagation is cruder than inference (no conditional
# refinement), so its answer is only used where it genuinely refines the inferred
# type; inference's per-statement/per-block-entry types remain authoritative
# otherwise. Both are sound for the fast copy: it executes the same statements under
# strictly more constraints.
function bif_refine(𝕃::AbstractLattice, @nospecialize(tfix), @nospecialize(torig))
    tfix === torig && return tfix
    return ⊑(𝕃, tfix, torig) ? tfix : torig
end

function bif_fast_value_type(@nospecialize(x), state::Vector{Any}, fast_ssatypes::Vector{Any},
                             slottypes::Vector{Any})
    if isa(x, SSAValue)
        return fast_ssatypes[x.id]
    elseif isa(x, SlotNumber) || isa(x, Argument)
        t = state[slot_id(x)]
        return t === Union{} ? slottypes[slot_id(x)] : t
    elseif isa(x, QuoteNode)
        return Const(x.value)
    elseif isa(x, GlobalRef) || isa(x, Expr)
        return Any
    end
    return Const(x)
end

function bif_fast_rhs_type(@nospecialize(rhs), i::Int, state::Vector{Any},
                           fast_ssatypes::Vector{Any}, ssavaluetypes::Vector{Any},
                           slottypes::Vector{Any}, stmt_info::Vector{CallInfo})
    if isa(rhs, SSAValue)
        return fast_ssatypes[rhs.id]
    elseif isa(rhs, SlotNumber) || isa(rhs, Argument)
        t = state[slot_id(rhs)]
        return t === Union{} ? ssavaluetypes[i] : t
    elseif isa(rhs, Expr)
        info = stmt_info[i]
        isa(info, SpeculatedCallInfo) && return info.spec_rt
        return ssavaluetypes[i]
    elseif isa(rhs, QuoteNode)
        return Const(rhs.value)
    elseif isa(rhs, GlobalRef)
        return ssavaluetypes[i] # non-plan binding read
    end
    return Const(rhs)
end

# Remap `SSAValue`s (and nested expression structure) of a copied statement through
# `map`. Branch targets are handled separately by the caller.
function bif_remap(@nospecialize(x), map::Vector{Int})
    if isa(x, SSAValue)
        return SSAValue(map[x.id])
    elseif isa(x, Expr)
        head = x.head
        nargs = length(x.args)
        e = Expr(head)
        resize!(e.args, nargs)
        for k = 1:nargs
            e.args[k] = bif_remap(x.args[k], map)
        end
        return e
    elseif isa(x, ReturnNode)
        return isdefined(x, :val) ? ReturnNode(bif_remap(x.val, map)) : x
    elseif isa(x, GotoIfNot)
        return GotoIfNot(bif_remap(x.cond, map), x.dest) # dest patched later
    end
    return x
end

struct BifEmit
    code::Vector{Any}
    types::Vector{Any}
    flags::Vector{UInt32}
    info::Vector{CallInfo}
    srcidx::Vector{Int}   # originating statement (0 = synthesized)
    oblock::Vector{Int}   # originating basic block (0 = prefix/catch)
    BifEmit() = new(Any[], Any[], UInt32[], CallInfo[], Int[], Int[])
end

function bif_push!(em::BifEmit, @nospecialize(stmt), @nospecialize(typ), flags::UInt32,
                   info::CallInfo, srcidx::Int, oblock::Int)
    push!(em.code, stmt)
    push!(em.types, typ)
    push!(em.flags, flags)
    push!(em.info, info)
    push!(em.srcidx, srcidx)
    push!(em.oblock, oblock)
    return length(em.code)
end

function bifurcate_speculated_globals!(ci::CodeInfo, sv::OptimizationState)
    sv.insert_coverage && return nothing
    stmt_info = sv.stmt_info
    # cheap scan for a candidate read before doing anything else
    cand = 0
    for i = 1:length(stmt_info)
        if isa(stmt_info[i], SpeculatedGlobalAccessInfo)
            i in sv.unreachable && continue
            cand = i
            break
        end
    end
    cand == 0 && return nothing
    code = ci.code
    stmt = code[cand]
    isexpr(stmt, :(=)) && (stmt = stmt.args[2])
    if !isa(stmt, GlobalRef)
        bif_debug_bail(cand, "candidate is not a direct GlobalRef read")
        return nothing
    end
    interp = sv.inlining.interp
    world = get_inference_world(interp)
    info = stmt_info[cand]::SpeculatedGlobalAccessInfo
    spec = info.spec
    if !(isa(spec, Type) && spec !== Any && spec !== Union{} && !has_free_typevars(spec))
        bif_debug_bail(cand, "unusable speculation")
        return nothing
    end
    plan = BifurcationPlan(stmt, bif_leaf_binding(world, info.b), spec)
    if !bif_classify!(plan, ci, sv, world)
        isempty(plan.reads) && bif_debug_bail(cand, "no reads matched the candidate binding")
        return nothing
    end
    BIF_DEBUG[] && println("bifurcating on ", plan.gr, " :: ", plan.spec, " (",
                           length(plan.reads), " reads, ", length(plan.stores), " stores)")

    𝕃 = optimizer_lattice(interp)
    fast_ssatypes, fast_entries, virt_join = bif_fast_types(ci, sv, plan, 𝕃)

    ssavaluetypes = ci.ssavaluetypes::Vector{Any}
    ssaflags = ci.ssaflags
    n = length(code)
    nslots = length(sv.slottypes)
    virt = SlotNumber(nslots + 1)
    has_stores = !isempty(plan.stores)
    oldcfg = sv.cfg
    em = BifEmit()
    fmap = zeros(Int, n)
    gmap = zeros(Int, n)
    patches = Tuple{Int,Int,Bool}[] # (new stmt idx, original target, in fast copy)

    # prefix: the one read and its guards
    grmod, grname = plan.gr.mod, plan.gr.name
    g0 = 0
    if !isa(sv.linfo.def, Method)
        # In a top-level thunk the speculated facts (in particular the fast copy's
        # invokes) are only known valid at the world this code was inferred in:
        # unlike a method, nothing re-validates a thunk against later worlds, and
        # execution proceeds at whatever world its (inert) prologue markers observe.
        # Guard on the world counter still being the inference world -- any
        # definition anywhere since then deopts to the generic copy, which subsumes
        # invalidation for this run-once code. The check is once per thunk entry;
        # loops contain no world markers, so the age cannot move inside them.
        wc = bif_push!(em, Expr(:foreigncall, Expr(:tuple, QuoteNode(:jl_get_world_counter)),
                                UInt, Core.svec(), 0, QuoteNode(:ccall)),
                       UInt, IR_FLAG_NOTHROW | IR_FLAG_EFFECT_FREE | IR_FLAG_TERMINATES,
                       NoCallInfo(), 0, 0)
        weq = bif_push!(em, Expr(:call, GlobalRef(Core, :(===)), SSAValue(wc), world),
                        Bool, BIF_FLAGS_SIMPLE, NoCallInfo(), 0, 0)
        g0 = bif_push!(em, GotoIfNot(SSAValue(weq), 0), Any, IR_FLAG_NOTHROW, NoCallInfo(), 0, 0)
    end
    defchk = bif_push!(em, Expr(:call, GlobalRef(Core, :isdefinedglobal), grmod, QuoteNode(grname)),
                       Bool, BIF_FLAGS_SIMPLE, NoCallInfo(), 0, 0)
    g1 = bif_push!(em, GotoIfNot(SSAValue(defchk), 0), Any, IR_FLAG_NOTHROW, NoCallInfo(), 0, 0)
    # nothrow: definedness was checked, and only another thread could revoke it
    readidx = bif_push!(em, plan.gr, Any, IR_FLAG_NOTHROW, GlobalAccessInfo(info.b), 0, 0)
    isachk = bif_push!(em, Expr(:call, GlobalRef(Core, :isa), SSAValue(readidx), spec),
                       Bool, BIF_FLAGS_SIMPLE, NoCallInfo(), 0, 0)
    g2 = bif_push!(em, GotoIfNot(SSAValue(isachk), 0), Any, IR_FLAG_NOTHROW, NoCallInfo(), 0, 0)
    # the typeassert (dominated by the guard, so it folds away) narrows the value's
    # IR type: everything downstream -- in particular the Upsilon/PhiC chain that
    # carries the pending value into the catch block -- must see the speculated
    # type, or the value gets boxed on every store into the PhiC slot
    narrowed = bif_push!(em, Expr(:call, GlobalRef(Core, :typeassert), SSAValue(readidx), spec),
                         spec, BIF_FLAGS_SIMPLE | IR_FLAG_CONSISTENT | IR_FLAG_NOUB, NoCallInfo(), 0, 0)
    bif_push!(em, Expr(:(=), virt, SSAValue(narrowed)), spec, BIF_FLAGS_SIMPLE, NoCallInfo(), 0, 0)
    enteridx = 0
    if has_stores
        enteridx = bif_push!(em, EnterNode(0), Any, IR_FLAG_NOTHROW, NoCallInfo(), 0, 0)
    end

    store_gr = plan.store_gr
    materialize() = Expr(:call, GlobalRef(Core, :setglobal!), (store_gr::GlobalRef).mod,
                         QuoteNode((store_gr::GlobalRef).name), virt)

    # FAST copy
    for i = 1:n
        stmt = code[i]
        ob = block_for_inst(oldcfg, i)
        fmap[i] = length(em.code) + 1
        if i in sv.unreachable
            bif_push!(em, bif_remap(stmt, fmap), ssavaluetypes[i], ssaflags[i], NoCallInfo(), i, ob)
            continue
        end
        if isa(stmt, ReturnNode) && has_stores
            bif_push!(em, materialize(), Any, IR_FLAG_NULL, GlobalAccessInfo(convert(Core.Binding, store_gr::GlobalRef)), i, ob)
            bif_push!(em, Expr(:leave, SSAValue(enteridx)), Nothing, IR_FLAG_NOTHROW, NoCallInfo(), i, ob)
            bif_push!(em, bif_remap(stmt, fmap), ssavaluetypes[i], ssaflags[i], NoCallInfo(), i, ob)
            continue
        end
        if i in plan.reads
            if isexpr(stmt, :(=))
                bif_push!(em, Expr(:(=), stmt.args[1], virt), fast_ssatypes[i], BIF_FLAGS_SIMPLE, NoCallInfo(), i, ob)
            else
                bif_push!(em, virt, fast_ssatypes[i], BIF_FLAGS_SIMPLE, NoCallInfo(), i, ob)
            end
            continue
        end
        if i in plan.stores
            val = bif_remap((stmt::Expr).args[4], fmap)
            bif_push!(em, Expr(:(=), virt, val), fast_ssatypes[i], BIF_FLAGS_SIMPLE, NoCallInfo(), i, ob)
            continue
        end
        newinfo = stmt_info[i]
        flags = ssaflags[i]
        if isa(newinfo, SpeculatedCallInfo)
            flags = (flags & ~BIF_FLAGS_EFFECTS) | flags_for_effects(newinfo.spec_effects)
            newinfo = newinfo.info
        elseif isa(newinfo, SpeculatedGlobalAccessInfo)
            newinfo = GlobalAccessInfo(newinfo.b)
        end
        pos = bif_push!(em, bif_remap(stmt, fmap), fast_ssatypes[i], flags, newinfo, i, ob)
        st = em.code[pos]
        if isa(st, GotoIfNot)
            push!(patches, (pos, st.dest, true))
        elseif isa(st, GotoNode)
            push!(patches, (pos, st.label, true))
        end
    end

    # CATCH: materialize the pending value, then rethrow
    catchstart = 0
    catchreturn = 0
    if has_stores
        catchstart = length(em.code) + 1
        bif_push!(em, materialize(), Any, IR_FLAG_NULL, GlobalAccessInfo(convert(Core.Binding, store_gr::GlobalRef)), 0, 0)
        bif_push!(em, Expr(:foreigncall, Expr(:tuple, QuoteNode(:jl_rethrow)), Union{}, Core.svec(), 0, QuoteNode(:ccall)),
                  Union{}, IR_FLAG_NULL, NoCallInfo(), 0, 0)
        # never reached (the rethrow always throws); `convert_to_ircode!` expects such
        # trailing terminators to be marked unreachable
        catchreturn = bif_push!(em, ReturnNode(), Union{}, IR_FLAG_NOTHROW, NoCallInfo(), 0, 0)
    end

    # GENERIC copy: the verbatim deoptimization target
    for i = 1:n
        stmt = code[i]
        ob = block_for_inst(oldcfg, i)
        gmap[i] = length(em.code) + 1
        newinfo = stmt_info[i]
        if isa(newinfo, SpeculatedCallInfo)
            # the sound answer was a failed generic resolution: a dynamic call
            newinfo = NoCallInfo()
        elseif isa(newinfo, SpeculatedGlobalAccessInfo)
            newinfo = GlobalAccessInfo(newinfo.b)
        end
        pos = bif_push!(em, bif_remap(stmt, gmap), ssavaluetypes[i], ssaflags[i], newinfo, i, ob)
        st = em.code[pos]
        if isa(st, GotoIfNot)
            push!(patches, (pos, st.dest, false))
        elseif isa(st, GotoNode)
            push!(patches, (pos, st.label, false))
        end
    end

    # patch branch targets
    g0 != 0 && (em.code[g0] = GotoIfNot((em.code[g0]::GotoIfNot).cond, gmap[1]))
    em.code[g1] = GotoIfNot((em.code[g1]::GotoIfNot).cond, gmap[1])
    em.code[g2] = GotoIfNot((em.code[g2]::GotoIfNot).cond, gmap[1])
    if has_stores
        em.code[enteridx] = EnterNode(catchstart)
    end
    for (pos, target, infast) in patches
        dest = infast ? fmap[target] : gmap[target]
        st = em.code[pos]
        if isa(st, GotoIfNot)
            em.code[pos] = GotoIfNot(st.cond, dest)
        else
            em.code[pos] = GotoNode(dest)
        end
    end

    # install the new code and re-derive all the state `convert_to_ircode!` and
    # `slot2reg` consume
    newn = length(em.code)
    dis = DebugInfoStream(sv.linfo, ci.debuginfo, newn)
    for k = 1:newn
        dis.codelocs[3k-2] = Int32(em.srcidx[k])
    end
    ci.code = em.code
    ci.ssavaluetypes = em.types
    ci.ssaflags = em.flags
    ci.debuginfo = Core.DebugInfo(dis, newn)
    push!(ci.slotnames, :var"#speculated_global")
    push!(ci.slotflags, 0x00)
    if ci.slottypes !== nothing && ci.slottypes !== sv.slottypes
        push!(ci.slottypes::Vector{Any}, Any)
    end
    push!(sv.slottypes, Any)
    sv.stmt_info = em.info

    newunreachable = BitSet()
    for i in sv.unreachable
        push!(newunreachable, fmap[i])
        push!(newunreachable, gmap[i])
    end
    catchreturn != 0 && push!(newunreachable, catchreturn)
    empty!(sv.unreachable)
    union!(sv.unreachable, newunreachable)

    newcfg = compute_basic_blocks(em.code)
    sv.cfg = newcfg
    nvirt = nslots + 1
    oldstates = sv.bb_states
    newstates = Union{Nothing,BBEntryState}[nothing for _ = 1:length(newcfg.blocks)]
    entrystate = oldstates[1]
    for (nb, block) in enumerate(newcfg.blocks)
        fs = first(block.stmts)
        ob = em.oblock[fs]
        local vartable::Vector{VarState}
        local aliases::Vector{Int}
        if ob == 0
            if has_stores && fs >= catchstart && fs < gmap[1]
                # the catch block: any point of FAST may transfer here; the virtual
                # slot is always assigned (the prefix dominates the enter), and its
                # type is the join of everything the fast copy can hold in it -- a
                # concrete join keeps the PhiC slot (and so the loop's Upsilon
                # stores) unboxed
                vartable = VarState[VarState(sv.slottypes[k], typemin(Int), true) for k = 1:nslots]
                push!(vartable, VarState(virt_join, typemin(Int), false))
            else
                # prefix blocks: function entry state
                entrystate === nothing && continue
                vartable = VarState[entrystate.vartable[k] for k = 1:nslots]
                push!(vartable, VarState(Any, typemin(Int), true))
            end
            aliases = zeros(Int, nvirt)
        else
            ostate = oldstates[ob]
            ostate === nothing && continue
            fastblock = fs < gmap[1]
            fentry = fastblock ? fast_entries[ob] : nothing
            vartable = Vector{VarState}(undef, nvirt)
            for k = 1:nslots
                vt = ostate.vartable[k]
                typ = fentry === nothing ? vt.typ : bif_refine(𝕃, fentry[k], vt.typ)
                vartable[k] = VarState(typ, vt.ssadef, vt.undef)
            end
            vartable[nvirt] = fastblock ?
                VarState(fentry === nothing ? Any : fentry[nvirt], typemin(Int), false) :
                VarState(Any, typemin(Int), true)
            aliases = copy(ostate.aliases)
            push!(aliases, 0)
        end
        newstates[nb] = BBEntryState(vartable, aliases)
    end
    sv.bb_states = newstates
    return nothing
end
