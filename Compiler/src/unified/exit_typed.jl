# Typed exit converter (§10.5): dense, typed, optimized UnifiedIR → IRCode,
# synthesizing header/merge PhiNodes from region args and results. This is
# the boundary that lets the UnifiedIR optimizer's output feed the stock
# backend (validated by the stock IR verifier; executed via OpaqueClosure).
#
# v2 feature matrix (§6 staged strategy, COMPILER-PORT-PLAN A4):
#   - `try` regions: EnterNode/`:leave`/`:pop_exception` synthesis. The enter
#     terminates its block (catch destination + fallthrough successors,
#     stock's compute_basic_blocks shape); exits that cross try scopes run
#     the structural leave/pop actions (§5.9) — `:leave` with the crossed
#     body tokens (innermost first), then `:pop_exception` per crossed
#     handler scope, mirroring stock lowering's order (leaves, then pops).
#   - residual frame cells: slot2ssa's algorithm relocated to the boundary
#     (§6 "boundary synthesis", P3): a flat mem2reg over the emitted block
#     graph — ordinary PhiNodes at the liveness-pruned iterated dominance
#     frontier of the store set, PhiCNode at each catch entry the cell is
#     live into fed by UpsilonNodes at every protected store (plus the
#     initial Upsilon at the `enter`, carrying the reaching value), and
#     stock's undef conventions (empty Upsilon / Bool definedness
#     companions / `:throw_undef_if_not` guards) on maybe-undef paths.
#   Declined precisely (per `classify_residual_cells` classes): `:gc_token`
#   cells (the pairing verifier tracks values), `:box_capture`
#   (`cell_shared`; no producer in this pipeline), and `:escape` (value
#   uses beyond the cell ops). `switch`, `await` and `closure` stay outside
#   the matrix.
#
# Layout discipline: blocks are objects, placed explicitly in final order;
# `GotoIfNot`/EnterNode/`:leave` fallthrough adjacency is guaranteed either
# by placing the continuation block next or by a trampoline block.

# marker for a cell value on a path where no store reached (stock's
# UNDEF_TOKEN equivalent): φ edges stay unassigned, Upsilons empty, guarded
# reads throw before any consumer runs (consumers collapse to `nothing`,
# stock's fixemup! convention)
struct CellUndef end
const CELL_UNDEF = CellUndef()

"the `:enter` token: resolved to the EnterNode's SSA position at assembly"
mutable struct TryTok
    ssaidx::Int
    scope::Any                # UnifiedIR.Operand | nothing
end

mutable struct TBB
    order::Int                # final position; 0 = not yet placed
    phis::Vector{Any}         # PhiSpec, in emission order (region + cell φs)
    phics::Vector{Any}        # PhiCSpec at a catch entry (value/flag pairs)
    items::Vector{Any}        # StmtId | cell ops (phase B rewrites) | synth items
    tuplemat::Vector{Any}     # SynthTuple, appended before the terminator
    term::Any                 # (:goto, TBB) | (:brifnot, cond, false::TBB, then::TBB)
                              # | (:return, val) | (:unreachable,)
                              # | (:enter, TryTok, catch::TBB, then::TBB)
                              # | (:leave, Vector{TryTok}, then::TBB) | nothing
end
TBB() = TBB(0, Any[], Any[], Any[], Any[], nothing)

mutable struct PhiSpec
    uirid::Int32
    typ::Any
    edges::Vector{Tuple{TBB,Any}}   # (pred block, value); CELL_UNDEF = unassigned
    ssaidx::Int
    iscell::Bool                    # cell φ: type is the fill-time edge join
end
PhiSpec(uirid, typ) = PhiSpec(uirid, typ, Tuple{TBB,Any}[], 0, false)

mutable struct SynthTuple
    vals::Vector{Any}
    ssaidx::Int
end

# ---- exception-SSA synthesis nodes (§6 boundary synthesis) -----------------

"UpsilonNode carrying `payload` (CELL_UNDEF = empty ϒ) into `phic`."
mutable struct SynthUps
    payload::Any
    phic::Any                 # PhiCSpec
    ssaidx::Int
    dropped::Bool
end

"PhiCNode at a catch entry for one cell (isflag: the Bool definedness leg)."
mutable struct PhiCSpec
    cellid::Int32
    ups::Vector{SynthUps}
    ssaidx::Int
    isflag::Bool
    dropped::Bool
    typ::Any
end
PhiCSpec(cellid, isflag) = PhiCSpec(cellid, SynthUps[], 0, isflag, false, isflag ? Bool : nothing)

"`Expr(:throw_undef_if_not, name, def)` guard before a maybe-undef read."
mutable struct SynthGuard
    name::Symbol
    defref::Any               # false | PhiSpec | PhiCSpec
    dropped::Bool
end

"`Expr(:pop_exception, tok)` on a normal exit out of a handler scope."
struct SynthPop
    tok::TryTok
end

"`Expr(:the_exception)` materializing a handler's %exc region arg."
mutable struct SynthExc
    typ::Any
    ssaidx::Int
end

"a raw expression item (e.g. the synthesized `rethrow()` of a handler-less try)"
struct SynthRaw
    ex::Any
    typ::Any
end

# ---- phase-A cell placeholders (rewritten by the phase-B mem2reg) ----------

struct CellDecl; cell::Int32; end
struct CellNewI; cell::Int32; prot::Vector{Any}; end            # prot: TryCtx list
struct CellStore; cell::Int32; val::Any; prot::Vector{Any}; end # innermost first
struct CellLoad; cell::Int32; uirid::Int32; end
struct CellIsdef; cell::Int32; uirid::Int32; end

"one active `try`: its token and catch block (nothing = synthesized rethrow)"
mutable struct TryCtx
    tok::TryTok
    catchbb::Union{TBB,Nothing}
end

mutable struct TCtx
    ir::UnifiedIR.IR
    placed::Vector{TBB}
    cur::TBB
    loopctx::Dict{Int32,Any}          # body region id => (header, exitctx, ehdepth)
    phi_of::Dict{Int32,PhiSpec}
    islands::Dict{Int32,Tuple{TBB,Int}}  # block region id => (tbb, ehdepth at cfg op)
    ehstack::Vector{Tuple{Symbol,TryCtx}}   # (:body|:handler, ctx), innermost last
    synth_of::Dict{Int32,Any}         # uirid => SynthExc (handler %exc args)
    cellload::Dict{Int32,Any}         # cell_get/cell_isdefined uirid => resolved payload
    names::Any                        # meta[:cell_names] or nothing
    catches::Vector{Tuple{TBB,TryCtx}}
end

"Create a block and place it at the end of the current layout."
function placebb!(cx::TCtx)
    bb = TBB()
    bb.order = length(cx.placed) + 1
    push!(cx.placed, bb)
    return bb
end
"Place a previously created (deferred) block now."
function place!(cx::TCtx, bb::TBB)
    @assert bb.order == 0
    bb.order = length(cx.placed) + 1
    push!(cx.placed, bb)
    return bb
end

setterm!(bb::TBB, t) = (bb.term === nothing && (bb.term = t); bb)

struct JoinCtx
    joinbb::TBB
    phis::Vector{PhiSpec}
    materialize::Bool
    unwind::Any               # nothing | (:leave, TryTok) | (:pop, TryTok)
end
JoinCtx(joinbb, phis, materialize) = JoinCtx(joinbb, phis, materialize, nothing)

"""
    ir_to_ircode(ir) -> Compiler.IRCode

Convert dense, typed UnifiedIR (including `try` regions and residual frame
cells) to stock IRCode with synthesized phis and exception SSA
(EnterNode/`:leave`/`:pop_exception`, PhiC/Upsilon). Throws `UnsupportedIR`
outside the matrix (see the header comment for the declined classes).
"""
function ir_to_ircode(ir::UnifiedIR.IR)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_DENSE, "ir_to_ircode")
    havecell = false
    for s in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, s)
        if k === K"cell" || k === K"cell_shared" || k === K"cell_get" ||
           k === K"cell_set" || k === K"cell_new" || k === K"cell_isdefined"
            havecell = true
        elseif k === K"await" || k === K"closure" || k === K"switch"
            throw(UnsupportedIR("$(UnifiedIR.kindname(k)) in typed exit"))
        end
    end
    if havecell
        # decline the residual classes the boundary synthesis does not model
        # (§6 taxonomy); every memory-shaped class goes through the flat
        # mem2reg below
        for (c, r) in classify_residual_cells(ir)
            if r === :gc_token || r === :box_capture || r === :escape
                throw(UnsupportedIR("residual cell %$(c.id) class $(r) in typed exit"))
            end
        end
    end

    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    argmap = Dict{Int32,Int}(a.id => i for (i, a) in enumerate(root.args))

    cx = TCtx(ir, TBB[], TBB(), Dict{Int32,Any}(), Dict{Int32,PhiSpec}(),
              Dict{Int32,Tuple{TBB,Int}}(), Tuple{Symbol,TryCtx}[],
              Dict{Int32,Any}(), Dict{Int32,Any}(),
              get(ir.meta, :cell_names, nothing), Tuple{TBB,TryCtx}[])
    cx.cur = placebb!(cx)
    emit_tregion!(cx, UnifiedIR.root_region(ir), nothing)
    for bb in cx.placed
        bb.term === nothing && (bb.term = (:unreachable,))
    end
    havecell && cell_mem2reg!(cx)
    return assemble_ircode(cx, ir, argmap, length(root.args))
end

# ---- use analysis ----------------------------------------------------------

function extract_only_uses(ir::UnifiedIR.IR, s::StmtId)
    extracts = StmtId[]
    others = 0
    UnifiedIR.each_ssa_use(ir) do site, used
        used == s || return
        if site isa UnifiedIR.StmtOperand &&
           UnifiedIR.stmt_kind(ir, site.user) === K"extract" &&
           UnifiedIR.asstmt(UnifiedIR.getop(ir, site.user, 1)) == s
            push!(extracts, site.user)
        else
            others += 1
        end
    end
    return (extracts, others)
end

function result_used(ir::UnifiedIR.IR, s::StmtId)
    used = false
    UnifiedIR.each_ssa_use(ir) do _, u
        u == s && (used = true)
    end
    return used
end

"Max value arity of the exits feeding owner `s` (results, breaks, continues).
A `continue` whose condition is literal `true` never exits its loop — it
contributes no result values (`emit_tcontinue!` emits only the back-edge for
it, and inference joins no exit values from it either)."
function owner_nvals(ir::UnifiedIR.IR, s::StmtId)
    rs = UnifiedIR.live_owned_regions(ir, s)
    rset = Set{Int32}(r.id for r in rs)
    n = 0
    for st in UnifiedIR.each_stmt(ir)
        k = UnifiedIR.stmt_kind(ir, st)
        if k === K"result"
            reg = UnifiedIR.stmt_region(ir, st)
            # a result terminator feeds its own region's owner
            UnifiedIR.getregion(ir, reg).owner == s && (n = max(n, UnifiedIR.nops(ir, st)))
        elseif k === K"break"
            tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, st, 1))
            tgt.id in rset && (n = max(n, UnifiedIR.nops(ir, st) - 1))
        elseif k === K"continue"
            tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, st, 1))
            if tgt.id in rset
                cond = UnifiedIR.getop(ir, st, 2)
                ctrue = UnifiedIR.optag(cond) == UnifiedIR.TAG_INLINE &&
                        UnifiedIR.imm_value(cond) === true
                ctrue || (n = max(n, UnifiedIR.nops(ir, st) - 2))
            end
        end
    end
    return n
end

function make_joinctx!(cx::TCtx, owner::StmtId, joinbb::TBB)
    ir = cx.ir
    nvals = owner_nvals(ir, owner)
    used = result_used(ir, owner)
    if nvals <= 1
        p = PhiSpec(owner.id, UnifiedIR.stmt_type(ir, owner))
        if used
            push!(joinbb.phis, p)
            cx.phi_of[owner.id] = p
        end
        return JoinCtx(joinbb, [p], false)
    end
    extracts, others = extract_only_uses(ir, owner)
    if others == 0
        phis = PhiSpec[PhiSpec(Int32(0), Any) for _ in 1:nvals]
        append!(joinbb.phis, phis)
        for ex in extracts
            idx = Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, ex, 2))::Int64)
            1 <= idx <= nvals || throw(UnsupportedIR("extract index out of range in typed exit"))
            cx.phi_of[ex.id] = phis[idx]
        end
        return JoinCtx(joinbb, phis, false)
    end
    p = PhiSpec(owner.id, UnifiedIR.stmt_type(ir, owner))
    push!(joinbb.phis, p)
    cx.phi_of[owner.id] = p
    return JoinCtx(joinbb, [p], true)
end

# Run the unwind action of a JoinCtx (a try body/handler exiting normally):
# `:leave` ends the current block (fallthrough continuation), `:pop_exception`
# is an ordinary statement item.
function apply_unwind!(cx::TCtx, j::JoinCtx)
    j.unwind === nothing && return nothing
    kind, tok = j.unwind
    if kind === :leave
        frombb = cx.cur
        contbb = TBB()
        setterm!(frombb, (:leave, TryTok[tok], contbb))
        place!(cx, contbb)
        cx.cur = contbb
    else # :pop
        push!(cx.cur.items, SynthPop(tok))
    end
    return nothing
end

function feed_join!(cx::TCtx, j::JoinCtx, vals::Vector{Any})
    apply_unwind!(cx, j)
    frombb = cx.cur
    if j.materialize
        st = SynthTuple(vals, 0)
        push!(frombb.tuplemat, st)
        push!(j.phis[1].edges, (frombb, st))
    else
        for (i, p) in enumerate(j.phis)
            push!(p.edges, (frombb, i <= length(vals) ? vals[i] : nothing))
        end
    end
    setterm!(frombb, (:goto, j.joinbb))
    return nothing
end

# Emit the structural leave/pop actions for an exit that unwinds the eh
# stack down to `depth` (§5.9): one `:leave` with the crossed body tokens
# (innermost first), then `:pop_exception` per crossed handler scope —
# stock lowering's order. Leaves cx.cur at the continuation block.
function emit_unwind!(cx::TCtx, depth::Int)
    length(cx.ehstack) > depth || return nothing
    toks = TryTok[]
    pops = TryTok[]
    for i in length(cx.ehstack):-1:(depth + 1)
        kind, ctx = cx.ehstack[i]
        kind === :body ? push!(toks, ctx.tok) : push!(pops, ctx.tok)
    end
    if !isempty(toks)
        frombb = cx.cur
        contbb = TBB()
        setterm!(frombb, (:leave, toks, contbb))
        place!(cx, contbb)
        cx.cur = contbb
    end
    for t in pops
        push!(cx.cur.items, SynthPop(t))
    end
    return nothing
end

"protecting try scopes of the current position: the handlers a store here
feeds (§6 — the `:body` entries of the eh stack), innermost first"
function protectors(cx::TCtx)
    prot = Any[]
    for i in length(cx.ehstack):-1:1
        kind, ctx = cx.ehstack[i]
        kind === :body && ctx.catchbb !== nothing && push!(prot, ctx)
    end
    return prot
end

# ---- region emission -------------------------------------------------------

function emit_tregion!(cx::TCtx, r::RegionId, jctx::Union{Nothing,JoinCtx})
    ir = cx.ir
    isblock = UnifiedIR.getregion(ir, r).kind === UnifiedIR.REGION_BLOCK
    for s in UnifiedIR.region_stmts(ir, r)
        k = UnifiedIR.stmt_kind(ir, s)
        if k === K"region_arg"
        elseif k === K"extract" && haskey(cx.phi_of, s.id)
            # de-tupled: the positional phi IS this value; no stmt emitted
        elseif k === K"if"
            emit_tif!(cx, s)
        elseif k === K"loop"
            emit_tloop!(cx, s)
        elseif k === K"cfg"
            emit_tcfg!(cx, s)
        elseif k === K"try"
            emit_ttry!(cx, s)
        elseif k === K"result"
            jctx === nothing && throw(UnsupportedIR("result terminator at root in typed exit"))
            feed_join!(cx, jctx, Any[UnifiedIR.getop(ir, s, i) for i in 1:UnifiedIR.nops(ir, s)])
        elseif k === K"return"
            emit_unwind!(cx, 0)
            setterm!(cx.cur, (:return, UnifiedIR.nops(ir, s) >= 1 ? UnifiedIR.getop(ir, s, 1) : nothing))
        elseif k === K"unreachable"
            setterm!(cx.cur, (:unreachable,))
        elseif k === K"break"
            tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
            (_, exitctx, depth) = cx.loopctx[tgt.id]
            emit_unwind!(cx, depth)
            feed_join!(cx, exitctx, Any[UnifiedIR.getop(ir, s, i) for i in 2:UnifiedIR.nops(ir, s)])
        elseif k === K"continue"
            emit_tcontinue!(cx, s)
        elseif k === K"goto"
            isblock || throw(UnsupportedIR("island terminator outside a cfg block"))
            emit_tgoto!(cx, s)
        elseif k === K"br_if"
            isblock || throw(UnsupportedIR("island terminator outside a cfg block"))
            emit_tbrif!(cx, s)
        elseif k === K"cell"
            push!(cx.cur.items, CellDecl(s.id))
        elseif k === K"cell_set"
            cid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
            push!(cx.cur.items, CellStore(cid, UnifiedIR.getop(ir, s, 2), protectors(cx)))
        elseif k === K"cell_get"
            cid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
            push!(cx.cur.items, CellLoad(cid, s.id))
        elseif k === K"cell_new"
            cid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
            push!(cx.cur.items, CellNewI(cid, protectors(cx)))
        elseif k === K"cell_isdefined"
            cid = UnifiedIR.asstmt(UnifiedIR.getop(ir, s, 1)).id
            push!(cx.cur.items, CellIsdef(cid, s.id))
        elseif k === K"cell_shared"
            throw(UnsupportedIR("cell_shared in typed exit"))   # gated above
        else
            push!(cx.cur.items, s)
            if UnifiedIR.stmt_type(ir, s) === Union{}
                # stock's unreachable-after rule: a Bottom-typed statement
                # never completes — the region tail is dead and contributes
                # no join edge (dead join edges after Union{} calls prune
                # here, like stock's convert_to_ircode `sv.unreachable`)
                setterm!(cx.cur, (:unreachable,))
                return nothing
            end
        end
    end
    return nothing
end

function emit_tif!(cx::TCtx, s::StmtId)
    ir = cx.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    cond = UnifiedIR.getop(ir, s, 1)
    frombb = cx.cur
    joinbb = TBB()                      # deferred: placed after the arms
    j = make_joinctx!(cx, s, joinbb)
    thenbb = placebb!(cx)               # fallthrough-adjacent to frombb
    if length(rs) >= 2
        elsebb = TBB()                  # deferred: placed after the then arm
        frombb.term = (:brifnot, cond, elsebb, thenbb)
        cx.cur = thenbb
        emit_tregion!(cx, rs[1], j)
        cx.cur.term === nothing && feed_join!(cx, j, Any[])
        place!(cx, elsebb)
        cx.cur = elsebb
        emit_tregion!(cx, rs[2], j)
        cx.cur.term === nothing && feed_join!(cx, j, Any[])
    else
        frombb.term = (:brifnot, cond, joinbb, thenbb)
        if result_used(ir, s)
            for p in j.phis
                push!(p.edges, (frombb, nothing))
            end
        end
        cx.cur = thenbb
        emit_tregion!(cx, rs[1], j)
        cx.cur.term === nothing && feed_join!(cx, j, Any[])
    end
    place!(cx, joinbb)
    cx.cur = joinbb
    return nothing
end

function emit_tloop!(cx::TCtx, s::StmtId)
    ir = cx.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    bodyr = rs[1]
    breg = UnifiedIR.getregion(ir, bodyr)
    frombb = cx.cur
    header = placebb!(cx)
    setterm!(frombb, (:goto, header))
    for (i, a) in enumerate(breg.args)
        p = PhiSpec(a.id, UnifiedIR.stmt_type(ir, a))
        push!(p.edges, (frombb, UnifiedIR.getop(ir, s, i)))
        push!(header.phis, p)
        cx.phi_of[a.id] = p
    end
    exitbb = TBB()                      # deferred: placed after the body
    j = make_joinctx!(cx, s, exitbb)
    cx.loopctx[bodyr.id] = (header, j, length(cx.ehstack))
    cx.cur = header
    emit_tregion!(cx, bodyr, nothing)
    place!(cx, exitbb)
    cx.cur = exitbb
    return nothing
end

function emit_tcontinue!(cx::TCtx, s::StmtId)
    ir = cx.ir
    tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
    (header, exitctx, depth) = cx.loopctx[tgt.id]
    breg = UnifiedIR.getregion(ir, tgt)
    cond = UnifiedIR.getop(ir, s, 2)
    vals = Any[UnifiedIR.getop(ir, s, i) for i in 3:UnifiedIR.nops(ir, s)]
    # the back edge re-enters the loop from outside any try opened inside it:
    # run the leave/pop actions BEFORE the branch (both paths exit the scopes)
    emit_unwind!(cx, depth)
    frombb = cx.cur
    ctrue = UnifiedIR.optag(cond) == UnifiedIR.TAG_INLINE && UnifiedIR.imm_value(cond) === true
    if ctrue
        for (i, a) in enumerate(breg.args)
            push!(cx.phi_of[a.id].edges, (frombb, vals[i]))
        end
        setterm!(frombb, (:goto, header))
    else
        backbb = placebb!(cx)           # fallthrough-adjacent (the true edge)
        exitfeed = TBB()
        frombb.term = (:brifnot, cond, exitfeed, backbb)
        for (i, a) in enumerate(breg.args)
            push!(cx.phi_of[a.id].edges, (backbb, vals[i]))
        end
        setterm!(backbb, (:goto, header))
        place!(cx, exitfeed)
        cx.cur = exitfeed
        feed_join!(cx, exitctx, vals)   # continue-false: results = carried vals (§5.3)
    end
    return nothing
end

# try { body } catch (%exc) { handler } (§6): EnterNode terminates the
# current block; body results run `:leave`, handler results `:pop_exception`
# before feeding the value join (stock's usetrydiv shape). Handler-less trys
# get the synthesized `rethrow()` catch block (exit_lowered's convention).
function emit_ttry!(cx::TCtx, s::StmtId)
    ir = cx.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    (length(rs) == 1 || length(rs) == 2) ||
        throw(UnsupportedIR("try with $(length(rs)) regions in typed exit"))
    hreg = length(rs) == 2 ? UnifiedIR.getregion(ir, rs[2]) : nothing
    hreg === nothing || hreg.kind === UnifiedIR.REGION_HANDLER ||
        throw(UnsupportedIR("try without a handler-kind second region in typed exit"))
    scope = UnifiedIR.nops(ir, s) >= 1 ? UnifiedIR.getop(ir, s, 1) : nothing
    tok = TryTok(0, scope)
    catchbb = TBB()
    tctx = TryCtx(tok, hreg === nothing ? nothing : catchbb)
    joinbb = TBB()
    j = make_joinctx!(cx, s, joinbb)

    frombb = cx.cur
    bodybb = TBB()
    setterm!(frombb, (:enter, tok, catchbb, bodybb))
    place!(cx, bodybb)                  # fallthrough-adjacent to the enter
    cx.cur = bodybb
    push!(cx.ehstack, (:body, tctx))
    emit_tregion!(cx, rs[1], JoinCtx(j.joinbb, j.phis, j.materialize, (:leave, tok)))
    cx.cur.term === nothing &&
        feed_join!(cx, JoinCtx(j.joinbb, j.phis, j.materialize, (:leave, tok)), Any[])
    pop!(cx.ehstack)

    place!(cx, catchbb)
    cx.cur = catchbb
    if hreg !== nothing
        push!(cx.catches, (catchbb, tctx))
        if !isempty(hreg.args)
            a = hreg.args[1]
            if result_used(ir, a)
                exc = SynthExc(UnifiedIR.stmt_type(ir, a), 0)
                push!(catchbb.items, exc)
                cx.synth_of[a.id] = exc
            end
        end
        push!(cx.ehstack, (:handler, tctx))
        emit_tregion!(cx, rs[2], JoinCtx(j.joinbb, j.phis, j.materialize, (:pop, tok)))
        cx.cur.term === nothing &&
            feed_join!(cx, JoinCtx(j.joinbb, j.phis, j.materialize, (:pop, tok)), Any[])
        pop!(cx.ehstack)
    else
        # no handler: re-raise (should not occur from our frontends)
        push!(catchbb.items, SynthRaw(Expr(:call, GlobalRef(Base, :rethrow)), Union{}))
        setterm!(catchbb, (:unreachable,))
    end
    place!(cx, joinbb)
    cx.cur = joinbb
    return nothing
end

# sealed cross-island goto (§5.5/§5.9): target block of an ancestor island —
# run the leave/pop actions of every try crossed, then edge
function emit_tgoto!(cx::TCtx, s::StmtId)
    ir = cx.ir
    (dest, args) = UnifiedIR.edge_bundles(ir, s)[1]
    ent = get(cx.islands, dest.id, nothing)
    ent === nothing && throw(UnsupportedIR("goto to an unregistered island block in typed exit"))
    (dbb, ddepth) = ent
    ddepth <= length(cx.ehstack) ||
        throw(UnsupportedIR("goto into a deeper eh scope in typed exit"))
    emit_unwind!(cx, ddepth)
    dblk = UnifiedIR.getregion(ir, dest)
    for (i, a) in enumerate(dblk.args)
        push!(cx.phi_of[a.id].edges, (cx.cur, args[i]))
    end
    setterm!(cx.cur, (:goto, dbb))
    return nothing
end

function emit_tbrif!(cx::TCtx, s::StmtId)
    ir = cx.ir
    bs = UnifiedIR.edge_bundles(ir, s)
    e1 = get(cx.islands, bs[1][1].id, nothing)
    e2 = get(cx.islands, bs[2][1].id, nothing)
    (e1 === nothing || e2 === nothing) &&
        throw(UnsupportedIR("br_if to an unregistered island block in typed exit"))
    (e1[2] == length(cx.ehstack) && e2[2] == length(cx.ehstack)) ||
        throw(UnsupportedIR("cross-scope br_if in typed exit"))
    cond = UnifiedIR.getop(ir, s, 1)
    # trampoline for the true edge keeps fallthrough adjacency
    srcbb = cx.cur
    tramp = placebb!(cx)
    for (edge, predbb) in ((bs[1], tramp), (bs[2], srcbb))
        dest, args = edge
        dblk = UnifiedIR.getregion(ir, dest)
        for (i, a) in enumerate(dblk.args)
            push!(cx.phi_of[a.id].edges, (predbb, args[i]))
        end
    end
    setterm!(tramp, (:goto, e1[1]))
    srcbb.term = (:brifnot, cond, e2[1], tramp)
    cx.cur = tramp   # the walk continues on the next region anyway
    return nothing
end

function emit_tcfg!(cx::TCtx, s::StmtId)
    ir = cx.ir
    rs = UnifiedIR.live_owned_regions(ir, s)
    frombb = cx.cur
    joinbb = TBB()
    j = make_joinctx!(cx, s, joinbb)
    depth = length(cx.ehstack)
    # blocks as deferred objects, registered for (possibly cross-island)
    # gotos; placed in region order as we walk
    for rid in rs
        bb = TBB()
        cx.islands[rid.id] = (bb, depth)
        blk = UnifiedIR.getregion(ir, rid)
        for a in blk.args
            p = PhiSpec(a.id, UnifiedIR.stmt_type(ir, a))
            push!(bb.phis, p)
            cx.phi_of[a.id] = p
        end
    end
    entryblk = UnifiedIR.getregion(ir, rs[1])
    for (i, a) in enumerate(entryblk.args)
        push!(cx.phi_of[a.id].edges, (frombb, UnifiedIR.getop(ir, s, i)))
    end
    setterm!(frombb, (:goto, cx.islands[rs[1].id][1]))
    for rid in rs
        bb = cx.islands[rid.id][1]
        place!(cx, bb)
        cx.cur = bb
        emit_tregion!(cx, rid, j)
    end
    place!(cx, joinbb)
    cx.cur = joinbb
    return nothing
end

# ---- phase B: flat mem2reg over the emitted block graph (§6 P3) ------------
#
# slot2ssa's algorithm relocated to the boundary: per residual cell, place
# value (+ Bool definedness) PhiNodes at the liveness-pruned iterated
# dominance frontier of the def set (stock `iterated_dominance_frontier`
# over the emitted CFG), a PhiCNode pair at every catch entry the cell is
# live into, Upsilons at each protected store / cell_new plus the initial
# Upsilon at the `enter`, and rename all reads by a single DFS walk. Reads
# on maybe-undef paths acquire `:throw_undef_if_not` guards; definitely-
# defined flag chains are simplified away afterwards.

function tbb_succs(bb::TBB)
    t = bb.term
    t[1] === :goto && return TBB[t[2]]
    t[1] === :brifnot && return TBB[t[4], t[3]]
    t[1] === :enter && return TBB[t[3], t[4]]
    t[1] === :leave && return TBB[t[3]]
    return TBB[]
end

function cell_mem2reg!(cx::TCtx)
    bbs = cx.placed
    n = length(bbs)
    idxof = Dict{TBB,Int}(bb => i for (i, bb) in enumerate(bbs))
    succs = [Int[idxof[sb] for sb in tbb_succs(bb)] for bb in bbs]
    preds = [Int[] for _ in 1:n]
    for (bi, ss) in enumerate(succs), si in ss
        push!(preds[si], bi)
    end
    reach = falses(n)
    let work = Int[1]
        while !isempty(work)
            b = pop!(work)
            reach[b] && continue
            reach[b] = true
            append!(work, succs[b])
        end
    end

    # per-cell op scan: defs (store/new/decl), gen (read before any def)
    cells = Int32[]
    defbbs = Dict{Int32,Vector{Int}}()
    gen = Dict{Int32,Set{Int}}()        # blocks with a read before a def
    for (bi, bb) in enumerate(bbs)
        seendef = Set{Int32}()
        for it in bb.items
            if it isa CellDecl
                it.cell in cells || push!(cells, it.cell)
                push!(get!(() -> Int[], defbbs, it.cell), bi)
                push!(seendef, it.cell)
            elseif it isa CellStore || it isa CellNewI
                push!(get!(() -> Int[], defbbs, it.cell), bi)
                push!(seendef, it.cell)
            elseif it isa CellLoad || it isa CellIsdef
                it.cell in seendef ||
                    push!(get!(() -> Set{Int}(), gen, it.cell), bi)
            end
        end
    end
    isempty(cells) && return nothing

    fblocks = CC.BasicBlock[CC.BasicBlock(CC.StmtRange(i, i), preds[i], succs[i]) for i in 1:n]
    fcfg = CC.CFG(fblocks, collect(2:n))
    domtree = CC.construct_domtree(fblocks)

    # liveness (block-level backward may-analysis) + φ/PhiC placement
    cellphis = Dict{Tuple{Int,Int32},Tuple{PhiSpec,PhiSpec}}()   # (bb, cell) => (val, flag)
    phicof = Dict{Tuple{TBB,Int32},Tuple{PhiCSpec,PhiCSpec}}()   # (catchbb, cell) => (val, flag)
    livein = Dict{Int32,BitSet}()
    for c in cells
        # live-in blocks: a use not preceded by an in-block def, propagated
        # backward through predecessors until a def block kills the path
        # (stock compute_live_ins' shape, block-granular)
        defs = Set{Int}(get(defbbs, c, Int[]))
        li = BitSet()
        work = collect(get(gen, c, Set{Int}()))
        while !isempty(work)
            b = pop!(work)
            (b in li || !reach[b]) && continue
            push!(li, b)
            for p in preds[b]
                p in defs || p in li || push!(work, p)
            end
        end
        livein[c] = li
        rdefs = Int[b for b in get(defbbs, c, Int[]) if reach[b]]
        isempty(rdefs) && continue
        idf = CC.iterated_dominance_frontier(fcfg, CC.BlockLiveness(rdefs, collect(li)), domtree)
        for b in idf
            reach[b] || continue
            vp = PhiSpec(Int32(0), nothing); vp.iscell = true
            fp = PhiSpec(Int32(0), Bool); fp.iscell = true
            push!(bbs[b].phis, vp)
            push!(bbs[b].phis, fp)
            cellphis[(b, c)] = (vp, fp)
        end
    end
    for (catchbb, _) in cx.catches
        bi = idxof[catchbb]
        reach[bi] || continue
        for c in cells
            bi in livein[c] || continue
            vpc = PhiCSpec(c, false)
            fpc = PhiCSpec(c, true)
            push!(catchbb.phics, vpc)
            push!(catchbb.phics, fpc)
            phicof[(catchbb, c)] = (vpc, fpc)
        end
    end

    # renaming walk (slot2ssa's worklist: (block, pred, incoming))
    init = Dict{Int32,Tuple{Any,Any}}(c => (CELL_UNDEF, false) for c in cells)
    worklist = Tuple{Int,Int,Dict{Int32,Tuple{Any,Any}}}[(1, 0, init)]
    visited = falses(n)
    guards = SynthGuard[]
    upsilons = SynthUps[]
    iscatch = Dict{TBB,TryCtx}(bb => t for (bb, t) in cx.catches)
    while !isempty(worklist)
        (bi, pred, state) = pop!(worklist)
        bb = bbs[bi]
        # fill cell φ edges from this pred (every edge, even when visited)
        for c in cells
            spec = get(cellphis, (bi, c), nothing)
            spec === nothing && continue
            (vp, fp) = spec
            (v, d) = state[c]
            push!(vp.edges, (pred == 0 ? bb : bbs[pred], v))
            push!(fp.edges, (pred == 0 ? bb : bbs[pred], d))
            state[c] = (vp, fp)
        end
        visited[bi] && continue
        visited[bi] = true
        # catch entry: the PhiC pair is the incoming state (stock's record)
        tc = get(iscatch, bb, nothing)
        if tc !== nothing
            for c in cells
                spec = get(phicof, (bb, c), nothing)
                spec === nothing && continue
                state[c] = spec
            end
        end
        newitems = Any[]
        for it in bb.items
            if it isa CellDecl
                state[it.cell] = (CELL_UNDEF, false)
            elseif it isa CellNewI
                state[it.cell] = (CELL_UNDEF, false)
                for ctx in it.prot
                    spec = get(phicof, (ctx.catchbb, it.cell), nothing)
                    spec === nothing && continue
                    push!(newitems, mkups!(upsilons, CELL_UNDEF, spec[1]))
                    push!(newitems, mkups!(upsilons, false, spec[2]))
                end
            elseif it isa CellStore
                state[it.cell] = (it.val, true)
                for ctx in it.prot
                    spec = get(phicof, (ctx.catchbb, it.cell), nothing)
                    spec === nothing && continue
                    push!(newitems, mkups!(upsilons, it.val, spec[1]))
                    push!(newitems, mkups!(upsilons, true, spec[2]))
                end
            elseif it isa CellLoad
                (v, d) = state[it.cell]
                if d !== true
                    g = SynthGuard(cellname(cx, it.cell), d, false)
                    push!(guards, g)
                    push!(newitems, g)
                end
                cx.cellload[it.uirid] = v
            elseif it isa CellIsdef
                (v, d) = state[it.cell]
                cx.cellload[it.uirid] = d
            else
                push!(newitems, it)
            end
        end
        bb.items = newitems
        # initial Upsilons: the value each protected cell holds on entry to
        # the try (stock's insert-before-the-:enter convention)
        t = bb.term
        if t !== nothing && t[1] === :enter
            cb = t[3]
            for c in cells
                spec = get(phicof, (cb, c), nothing)
                spec === nothing && continue
                (v, d) = state[c]
                push!(bb.items, mkups!(upsilons, v, spec[1]))
                push!(bb.items, mkups!(upsilons, d, spec[2]))
            end
        end
        for si in succs[bi]
            push!(worklist, (si, bi, copy(state)))
        end
    end
    # unreachable blocks: cell ops dissolve (stock deletes such statements);
    # reads resolve to the undef marker so consumers collapse to `nothing`
    for (bi, bb) in enumerate(bbs)
        visited[bi] && continue
        newitems = Any[]
        for it in bb.items
            if it isa CellLoad
                cx.cellload[it.uirid] = CELL_UNDEF
            elseif it isa CellIsdef
                cx.cellload[it.uirid] = false
            elseif !(it isa CellDecl || it isa CellStore || it isa CellNewI)
                push!(newitems, it)
            end
        end
        bb.items = newitems
        empty!(bb.phics)
    end

    # definedness simplification: flag legs provably `true` everywhere are
    # dropped (guards, flag φ/PhiCs and their upsilons) — the common
    # definitely-assigned cells keep a clean value-only form. Downward
    # fixpoint: start every flag node optimistically true, falsify from
    # literal-false/undef inputs until stable (a memoized recursion would
    # commit stale trues across cycles).
    flagnodes = Any[]
    for bb in bbs
        for p in bb.phis
            p isa PhiSpec && p.iscell && p.typ === Bool && push!(flagnodes, p)
        end
        for pc in bb.phics
            pc.isflag && push!(flagnodes, pc)
        end
    end
    truth = IdDict{Any,Bool}(x => true for x in flagnodes)
    flagval(@nospecialize(x)) = x === true ? true :
        (x isa PhiSpec || x isa PhiCSpec) ? get(truth, x, false) : false
    changed = true
    while changed
        changed = false
        for x in flagnodes
            truth[x] || continue
            ok = x isa PhiSpec ? all(e -> flagval(e[2]), x.edges) :
                                 all(u -> flagval(u.payload), x.ups)
            ok || (truth[x] = false; changed = true)
        end
    end
    for g in guards
        flagval(g.defref) && (g.dropped = true)
    end
    for (uirid, v) in cx.cellload
        (v isa PhiSpec || v isa PhiCSpec) && flagval(v) && (cx.cellload[uirid] = true)
    end
    # rewrite every remaining reference to an always-true (about-to-drop)
    # flag node with the literal — surviving flag φ edges and flag Upsilon
    # payloads may point at dropped nodes across joins
    droppedflag(@nospecialize(x)) = (x isa PhiSpec || x isa PhiCSpec) && flagval(x)
    for bb in bbs
        for p in bb.phis
            (p isa PhiSpec && p.iscell && p.typ === Bool) || continue
            for (i, (pred, v)) in enumerate(p.edges)
                droppedflag(v) && (p.edges[i] = (pred, true))
            end
        end
    end
    for u in upsilons
        droppedflag(u.payload) && (u.payload = true)
    end
    for bb in bbs
        filter!(p -> !(p isa PhiSpec && p.iscell && p.typ === Bool && flagval(p)), bb.phis)
        for pc in bb.phics
            pc.isflag && flagval(pc) && (pc.dropped = true)
        end
        filter!(pc -> !pc.dropped, bb.phics)
    end
    for u in upsilons
        u.phic.dropped && (u.dropped = true)
    end
    for bb in bbs
        filter!(it -> !(it isa SynthUps && it.dropped) && !(it isa SynthGuard && it.dropped),
                bb.items)
    end
    celltypes!(cx, bbs)
    return nothing
end

# Types for the synthesized cell φ/PhiC nodes: the join of what actually
# flows in (§5.1 rule 2 — a recomputed join, never a declared type). Upward
# fixpoint from Union{} over the node graph (payload chains may cycle
# through loop headers); capped, with Any as the safe overflow.
function celltypes!(cx::TCtx, bbs::Vector{TBB})
    ir = cx.ir
    nodes = Any[]
    for bb in bbs
        for p in bb.phis
            p isa PhiSpec && p.iscell && p.typ === nothing && push!(nodes, p)
        end
        for pc in bb.phics
            pc.isflag || push!(nodes, pc)
        end
    end
    isempty(nodes) && return nothing
    typof = IdDict{Any,Any}(x => Union{} for x in nodes)
    function ptyp(@nospecialize(p))
        p === CELL_UNDEF && return Union{}         # contributes nothing
        (p isa PhiSpec || p isa PhiCSpec) && return get(typof, p, p isa PhiSpec ? something_typ(p) : Bool)
        p isa UnifiedIR.Operand || return CC.Const(p)
        t = UnifiedIR.optag(p)
        if t == UnifiedIR.TAG_STMT
            sid = UnifiedIR.asstmt(p)
            haskey(cx.cellload, sid.id) && return ptyp(cx.cellload[sid.id])
            tt = UnifiedIR.stmt_type(ir, sid)
            return tt === nothing ? Any : tt
        end
        t == UnifiedIR.TAG_INLINE && return CC.Const(UnifiedIR.imm_value(p))
        t == UnifiedIR.TAG_CONST && return CC.Const(ir.body.constants[UnifiedIR.payload(p)])
        if t == UnifiedIR.TAG_GLOBAL
            g = ir.body.globals[UnifiedIR.payload(p)]
            (isconst(g.mod, g.name) && isdefined(g.mod, g.name)) &&
                return CC.Const(getglobal(g.mod, g.name))
        end
        return Any
    end
    something_typ(p::PhiSpec) = p.typ === nothing ? Any : p.typ
    rounds = 0
    changed = true
    while changed
        changed = false
        rounds += 1
        if rounds > 100                            # widen out of oscillation
            for x in nodes
                typof[x] = Any
            end
            break
        end
        for x in nodes
            acc = Union{}
            ins = x isa PhiSpec ? Any[v for (_, v) in x.edges] :
                                  Any[u.payload for u in x.ups]
            for p in ins
                acc = CC.tmerge(CC.fallback_lattice, acc, ptyp(p))
            end
            if !(acc === typof[x]) && !CC.:⊑(CC.fallback_lattice, acc, typof[x])
                typof[x] = CC.tmerge(CC.fallback_lattice, typof[x], acc)
                changed = true
            end
        end
    end
    for x in nodes
        x isa PhiSpec ? (x.typ = typof[x]) : (x.typ = typof[x])
    end
    return nothing
end

mkups!(upsilons::Vector{SynthUps}, @nospecialize(payload), phic::PhiCSpec) = begin
    u = SynthUps(payload, phic, 0, false)
    push!(phic.ups, u)
    push!(upsilons, u)
    u
end

function cellname(cx::TCtx, cell::Int32)
    names = cx.names
    names isa Dict{Int32,Symbol} && return get(names, cell, :cell)
    return :cell
end

# ---- assembly --------------------------------------------------------------

function assemble_ircode(cx::TCtx, ir::UnifiedIR.IR, argmap::Dict{Int32,Int}, nargs::Int)
    bbs = cx.placed
    ssaof = Dict{Int32,Int}()
    nst = 0
    for bb in bbs
        for p in bb.phis
            nst += 1
            p.ssaidx = nst
            p.uirid != 0 && (ssaof[p.uirid] = nst)
        end
        for pc in bb.phics
            nst += 1
            pc.ssaidx = nst
        end
        for it in bb.items
            nst += 1
            if it isa StmtId
                ssaof[it.id] = nst
            elseif it isa SynthUps || it isa SynthExc
                it.ssaidx = nst
            end
        end
        for st in bb.tuplemat
            nst += 1
            st.ssaidx = nst
        end
        nst += 1
        t = bb.term
        t[1] === :enter && (t[2].ssaidx = nst)
    end

    # resolve a renaming payload / synthesized reference to an IRCode value
    function pval(@nospecialize(p))
        p isa UnifiedIR.Operand && return tval(p)
        (p isa PhiSpec || p isa PhiCSpec || p isa SynthUps || p isa SynthExc) &&
            return Core.SSAValue(p.ssaidx)
        p isa SynthTuple && return Core.SSAValue(p.ssaidx)
        p === CELL_UNDEF && return CELL_UNDEF
        return p                        # literal (incl. true/false flags)
    end

    function tval(@nospecialize(o))
        o === nothing && return nothing
        if o isa StmtId
            haskey(argmap, o.id) && return Core.Argument(argmap[o.id])
            p = get(cx.phi_of, o.id, nothing)
            p !== nothing && return Core.SSAValue(p.ssaidx)
            haskey(cx.cellload, o.id) && return pval(cx.cellload[o.id])
            haskey(cx.synth_of, o.id) && return Core.SSAValue(cx.synth_of[o.id].ssaidx)
            haskey(ssaof, o.id) && return Core.SSAValue(ssaof[o.id])
            error("typed exit: unmapped value %$(o.id)")
        elseif o isa SynthTuple
            return Core.SSAValue(o.ssaidx)
        elseif o isa UnifiedIR.Operand
            t = UnifiedIR.optag(o)
            t == UnifiedIR.TAG_STMT && return tval(UnifiedIR.asstmt(o))
            t == UnifiedIR.TAG_INLINE && return UnifiedIR.imm_value(o)
            if t == UnifiedIR.TAG_CONST
                v = ir.body.constants[UnifiedIR.payload(o)]
                return v isa Union{Symbol,Expr} ? QuoteNode(v) : v
            end
            t == UnifiedIR.TAG_GLOBAL && return ir.body.globals[UnifiedIR.payload(o)]
            t == UnifiedIR.TAG_SPARAM && return Expr(:static_parameter, Int(UnifiedIR.payload(o)))
            error("typed exit: bad operand tag")
        else
            return o
        end
    end

    # lattice type of a renaming payload (for synthesized ϒ types; φ/φᶜ
    # types were computed by celltypes!'s fixpoint)
    function ptyp(@nospecialize(p))
        if p isa UnifiedIR.Operand
            t = UnifiedIR.optag(p)
            if t == UnifiedIR.TAG_STMT
                sid = UnifiedIR.asstmt(p)
                haskey(cx.cellload, sid.id) && return ptyp(cx.cellload[sid.id])
                tt = UnifiedIR.stmt_type(ir, sid)
                return tt === nothing ? Any : tt
            end
            t == UnifiedIR.TAG_INLINE && return CC.Const(UnifiedIR.imm_value(p))
            t == UnifiedIR.TAG_CONST && return CC.Const(ir.body.constants[UnifiedIR.payload(p)])
            if t == UnifiedIR.TAG_GLOBAL
                g = ir.body.globals[UnifiedIR.payload(p)]
                (isconst(g.mod, g.name) && isdefined(g.mod, g.name)) &&
                    return CC.Const(getglobal(g.mod, g.name))
                return Any
            end
            return Any
        end
        p isa PhiSpec && return p.typ === nothing ? Any : p.typ
        p isa PhiCSpec && return p.typ === nothing ? Any : p.typ
        p isa SynthTuple && return Any
        p === CELL_UNDEF && return Union{}
        p === nothing && return CC.Const(nothing)
        return CC.Const(p)
    end

    function tflags(f::UInt32)
        out = UInt32(0)
        (f & UnifiedIR.FLAG_CONSISTENT != 0) && (out |= CC.IR_FLAG_CONSISTENT)
        (f & UnifiedIR.FLAG_EFFECT_FREE != 0) && (out |= CC.IR_FLAG_EFFECT_FREE)
        (f & UnifiedIR.FLAG_NOTHROW != 0) && (out |= CC.IR_FLAG_NOTHROW)
        (f & UnifiedIR.FLAG_TERMINATES != 0) && (out |= CC.IR_FLAG_TERMINATES)
        (f & UnifiedIR.FLAG_INBOUNDS != 0) && (out |= CC.IR_FLAG_INBOUNDS)
        return out
    end

    # foreigncall/cfunction operands are STRUCTURAL syntax pieces (the
    # (name, lib) tuple Expr, sparam-dependent type Exprs): emit interned
    # Expr constants raw, not value-quoted
    function raw_structural(@nospecialize(o))
        o isa QuoteNode && o.value isa Expr && return o.value
        return o
    end

    # a statement whose operand reads a never-stored cell sits behind an
    # unconditional throw guard: collapse it to `nothing` (stock fixemup!)
    function undef_operand(s::StmtId)
        for i in 1:UnifiedIR.nops(ir, s)
            o = UnifiedIR.getop(ir, s, i)
            UnifiedIR.optag(o) == UnifiedIR.TAG_STMT || continue
            get(cx.cellload, UnifiedIR.asstmt(o).id, nothing) === CELL_UNDEF && return true
        end
        return false
    end

    function translate_stmt(s::StmtId)
        k = UnifiedIR.stmt_kind(ir, s)
        n = UnifiedIR.nops(ir, s)
        ops = Any[tval(UnifiedIR.getop(ir, s, i)) for i in 1:n]
        k === K"call" && return Expr(:call, ops...)
        k === K"invoke" && return Expr(:invoke, ops...)
        k === K"new" && return Expr(:new, ops...)
        k === K"splatnew" && return Expr(:splatnew, ops...)
        if k === K"foreigncall" || k === K"cfunction" || k === K"new_opaque_closure"
            rawops = Any[raw_structural(o) for o in ops]
            if k === K"foreigncall" && !isempty(rawops) &&
               rawops[1] isa QuoteNode && rawops[1].value === FOREIGNGLOBAL_MARKER
                # marker-encoded Expr(:foreignglobal, name) — see codeinfo_entry
                return Expr(:foreignglobal, rawops[2:end]...)
            end
            return Expr(k === K"cfunction" ? :cfunction :
                        k === K"foreigncall" ? :foreigncall : :new_opaque_closure, rawops...)
        end
        if k === K"extract"
            return Expr(:call, GlobalRef(Core, :getfield), ops[1],
                        Int(UnifiedIR.imm_value(UnifiedIR.getop(ir, s, 2))))
        end
        if k === K"refine"
            t = UnifiedIR.stmt_type(ir, s)
            return Core.PiNode(ops[1], t isa Type ? t : CC.widenconst(t))
        end
        (k === K"value" || k === K"globalref") && return ops[1]
        k === K"select" && return Expr(:call, GlobalRef(Core, :ifelse), ops...)
        k === K"isdefined_global" && return Expr(:isdefined, ops[1])
        k === K"boundscheck" && return Expr(:boundscheck)
        k === K"gc_preserve_begin" && return Expr(:gc_preserve_begin, ops...)
        k === K"gc_preserve_end" && return Expr(:gc_preserve_end, ops...)
        k === K"latestworld" && return Expr(:latestworld)
        k === K"coverage_effect" && return Expr(:code_coverage_effect)
        k === K"copyast" && return Expr(:copyast, ops...)
        if k === K"throw_undef_if_not"
            nm = ops[2] isa QuoteNode ? ops[2].value : ops[2]
            return Expr(:throw_undef_if_not, nm, ops[1])
        end
        throw(UnsupportedIR("kind $(UnifiedIR.kindname(k)) in typed exit"))
    end

    stmts = Vector{Any}(undef, nst)
    types = Vector{Any}(undef, nst)
    flags = fill(UInt32(0), nst)
    lines = fill(Int32(0), 3nst)
    infos = CC.CallInfo[CC.NoCallInfo() for _ in 1:nst]

    pos = 0
    blocks = CC.BasicBlock[]
    catchpred0 = Int[]                  # blocks that get the virtual 0 pred
    for (bi, bb) in enumerate(bbs)
        start = pos + 1
        for p in bb.phis
            pos += 1
            edges = Int32[Int32(pred.order) for (pred, _) in p.edges]
            vals = Vector{Any}(undef, length(p.edges))
            for (i, (_, v)) in enumerate(p.edges)
                rv = p.iscell ? pval(v) : tval(v)
                rv === CELL_UNDEF || (vals[i] = rv)   # undef edge: unassigned
            end
            t = p.typ
            if length(edges) == 1 && isassigned(vals, 1)
                # a join left with a single live edge (dead edges pruned by
                # the unreachable-after rule) is no φ at all — stock never
                # emits 1-edge φs. A bare value statement (renaming copy) is
                # legal IRCode and, unlike PiNode, also legal pre-inference
                # method source (the ssa_method engine legs).
                stmts[pos] = vals[1]
            else
                stmts[pos] = Core.PhiNode(edges, vals)
            end
            types[pos] = t === nothing ? Any : t
        end
        for pc in bb.phics
            pos += 1
            stmts[pos] = Core.PhiCNode(Any[Core.SSAValue(u.ssaidx) for u in pc.ups])
            types[pos] = pc.typ === nothing ? Any : pc.typ
        end
        for it in bb.items
            pos += 1
            if it isa StmtId
                if undef_operand(it)
                    stmts[pos] = nothing         # guarded-unreachable consumer
                    types[pos] = Nothing
                else
                    stmts[pos] = translate_stmt(it)
                    t = UnifiedIR.stmt_type(ir, it)
                    types[pos] = t === nothing ? Any : t
                    flags[pos] = tflags(UnifiedIR.stmt_flag(ir, it))
                end
            elseif it isa SynthUps
                v = pval(it.payload)
                stmts[pos] = v === CELL_UNDEF ? Core.UpsilonNode() : Core.UpsilonNode(v)
                types[pos] = v === CELL_UNDEF ? Union{} :
                             (it.phic.isflag ? Bool : ptyp(it.payload))
            elseif it isa SynthGuard
                stmts[pos] = Expr(:throw_undef_if_not, it.name, pval(it.defref))
                types[pos] = Any
            elseif it isa SynthPop
                stmts[pos] = Expr(:pop_exception, Core.SSAValue(it.tok.ssaidx))
                types[pos] = Nothing
            elseif it isa SynthExc
                stmts[pos] = Expr(:the_exception)
                types[pos] = it.typ === nothing ? Any : it.typ
            elseif it isa SynthRaw
                stmts[pos] = it.ex
                types[pos] = it.typ
            else
                error("typed exit: unknown item $(typeof(it))")
            end
        end
        for st in bb.tuplemat
            pos += 1
            vals = Any[]
            for v in st.vals
                rv = tval(v)
                push!(vals, rv === CELL_UNDEF ? nothing : rv)
            end
            stmts[pos] = Expr(:call, GlobalRef(Core, :tuple), vals...)
            types[pos] = Any
        end
        pos += 1
        term = bb.term
        succs = Int[]
        if term[1] === :goto
            stmts[pos] = Core.GotoNode(term[2].order)   # IRCode: block numbers
            types[pos] = Any
            push!(succs, term[2].order)
        elseif term[1] === :brifnot
            _, cond, falsebb, thenbb = term
            thenbb.order == bi + 1 ||
                throw(UnsupportedIR("typed exit: brifnot fallthrough not adjacent (layout bug)"))
            cnd = tval(cond)
            cnd === CELL_UNDEF && (cnd = false)          # guarded-unreachable branch
            stmts[pos] = Core.GotoIfNot(cnd, falsebb.order)
            types[pos] = Any
            push!(succs, thenbb.order)
            push!(succs, falsebb.order)
        elseif term[1] === :return
            v = tval(term[2])
            # a return of a never-stored cell sits behind a throw guard:
            # stock's convention keeps the CFG with a bare unreachable
            stmts[pos] = v === CELL_UNDEF ? Core.ReturnNode() : Core.ReturnNode(v)
            types[pos] = v === CELL_UNDEF ? Union{} : Any
        elseif term[1] === :enter
            _, tok, catchbb, thenbb = term
            thenbb.order == bi + 1 ||
                throw(UnsupportedIR("typed exit: enter fallthrough not adjacent (layout bug)"))
            @assert tok.ssaidx == pos
            stmts[pos] = tok.scope === nothing ? Core.EnterNode(catchbb.order) :
                         Core.EnterNode(catchbb.order, tval(tok.scope))
            types[pos] = Any
            push!(succs, catchbb.order)
            push!(succs, thenbb.order)
            push!(catchpred0, catchbb.order)
        elseif term[1] === :leave
            _, toks, thenbb = term
            thenbb.order == bi + 1 ||
                throw(UnsupportedIR("typed exit: leave fallthrough not adjacent (layout bug)"))
            stmts[pos] = Expr(:leave, Any[Core.SSAValue(t.ssaidx) for t in toks]...)
            types[pos] = Any
            push!(succs, thenbb.order)
        else
            stmts[pos] = Core.ReturnNode()
            types[pos] = Union{}
        end
        push!(blocks, CC.BasicBlock(CC.StmtRange(start, pos), Int[], succs))
    end
    for (bi, blk) in enumerate(blocks)
        for su in blk.succs
            push!(blocks[su].preds, bi)
        end
    end
    for cb in catchpred0
        # stock compute_basic_blocks gives catch blocks a virtual 0 pred
        # ("entered from outside"); verify and domtree skip it
        push!(blocks[cb].preds, 0)
    end
    index = Int[blocks[i].stmts.start for i in 2:length(blocks)]
    cfg = CC.CFG(blocks, index)

    is = CC.InstructionStream(stmts, types, infos, lines, flags)
    di = CC.DebugInfoStream(lines)
    argtypes = Any[t for t in ir.argtypes]
    length(argtypes) == nargs || (argtypes = Any[Any for _ in 1:nargs])
    splat = get(ir.meta, :sptypes_lat, nothing)
    sptypes = CC.VarState[]
    for (i, sp) in enumerate(ir.sptypes)
        lat = splat !== nothing && i <= length(splat) ? splat[i] : CC.Const(sp)
        push!(sptypes, CC.VarState(lat, #=ssadef=#typemin(Int), #=undef=#false))
    end
    # a finite world range: value-position GlobalRefs live in partitioned
    # bindings whose validity never spans all worlds (the driver overwrites
    # the emitted CodeInfo's bounds with its edge collector's window)
    worlds = CC.WorldRange(Base.get_world_counter(), typemax(UInt))
    return CC.IRCode(is, cfg, di, argtypes, Expr[], sptypes, worlds)
end
