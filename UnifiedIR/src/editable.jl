# Editable state (§4.2): per-region intrusive doubly-linked lists + a global
# order-key accelerator. Ids stay what they were; `compact!` returns to dense.

"""
    editable(ir) -> ir

Materialize the editable layout (O(n)). The same handle is returned; the
layout-state tag flips, so stale dense assumptions fail deterministically.
"""
function editable(ir::IR)
    check_state(ir, LAYOUT_DENSE, "editable")
    flush_renames!(ir)
    n = Int(ir.body.len)
    next = zeros(Int32, n)
    prev = zeros(Int32, n)
    okey = zeros(UInt64, n)
    # thread per-region lists from dense spans; okey = global position
    lastin = IdDict{Int32,Int32}()   # region id -> last member seen
    for i in 1:n
        okey[i] = UInt64(i) << 32
        r = ir.body.region[i].id
        p = get(lastin, r, Int32(0))
        prev[i] = p
        p != 0 && (next[p] = Int32(i))
        lastin[r] = Int32(i)
    end
    # region first/last become list head/tail (heads: first member in span)
    heads = IdDict{Int32,Int32}()
    for i in n:-1:1
        heads[ir.body.region[i].id] = Int32(i)
    end
    for (ri, reg) in enumerate(ir.regions)
        h = get(heads, Int32(ri), Int32(0))
        t = get(lastin, Int32(ri), Int32(0))
        reg.first = StmtId(h)
        reg.last = StmtId(t)
    end
    ir.edit = EditState(next, prev, okey)
    ir.owner.state = LAYOUT_EDITABLE
    ir.cache.layout_epoch += 1
    rebuild_region_links!(ir, ir.edit::EditState)
    return ir
end

# ---- owned-region links (the `flat_next` region step; §4.2) ----

# Ready the owner→regions links for a flattened-walk step. The O(1) check
# below is the whole validity contract: `alloc_stmt!` and `register_region!`
# extend the links in step with the statement/region tables, and anything that
# grows a table without them leaves a length mismatch that costs one rebuild.
@inline function region_links(ir::IR)
    e = ir.edit::EditState
    if length(e.nextsib) != length(ir.regions) || length(e.firstreg) != Int(ir.body.len)
        rebuild_region_links!(ir, e)
    end
    return e
end

function rebuild_region_links!(ir::IR, e::EditState)
    n = Int(ir.body.len)
    nr = length(ir.regions)
    resize!(e.firstreg, n); fill!(e.firstreg, Int32(0))
    resize!(e.lastreg, n);  fill!(e.lastreg, Int32(0))
    resize!(e.nextsib, nr); fill!(e.nextsib, Int32(0))
    resize!(e.prevsib, nr); fill!(e.prevsib, Int32(0))
    for i in 1:nr
        o = ir.regions[i].owner.id
        (o == 0 || o > n) && continue      # guard/root regions have no owner
        t = e.lastreg[o]
        if t == 0
            e.firstreg[o] = Int32(i)
        else
            e.nextsib[t] = Int32(i)
            e.prevsib[i] = t
        end
        e.lastreg[o] = Int32(i)
    end
    return nothing
end

"""
    register_region!(ir, rid)

Thread a region just appended to `ir.regions` onto its owner's sibling chain
(editable state; a no-op elsewhere, where the links are rebuilt on entry).
Regions are only ever appended, so the chain stays in table order.
"""
function register_region!(ir::IR, rid::RegionId)
    layout(ir) === LAYOUT_EDITABLE || return nothing
    e = ir.edit
    e === nothing && return nothing
    e = e::EditState
    # not in step (some other path grew a table): leave it to the rebuild
    (length(e.nextsib) == rid.id - 1 && length(e.firstreg) == Int(ir.body.len)) ||
        return nothing
    push!(e.nextsib, Int32(0))
    push!(e.prevsib, Int32(0))
    o = ir.regions[rid.id].owner.id
    o == 0 && return nothing
    t = e.lastreg[o]
    if t == 0
        e.firstreg[o] = Int32(rid.id)
    else
        e.nextsib[t] = Int32(rid.id)
        e.prevsib[rid.id] = t
    end
    e.lastreg[o] = Int32(rid.id)
    return nothing
end

# ---- linked-list primitives ----

function unlink!(ir::IR, s::StmtId)
    e = ir.edit::EditState
    r = getregion(ir, stmt_region(ir, s))
    p, n = e.prev[s.id], e.next[s.id]
    p != 0 ? (e.next[p] = n) : (r.first = StmtId(n))
    n != 0 ? (e.prev[n] = p) : (r.last = StmtId(p))
    e.next[s.id] = 0
    e.prev[s.id] = 0
    return nothing
end

# Insert row `s` into region `r`'s list between `p` and `n` (either may be 0).
function link_between!(ir::IR, s::StmtId, r::RegionId, p::Int32, n::Int32)
    e = ir.edit::EditState
    reg = getregion(ir, r)
    e.prev[s.id] = p
    e.next[s.id] = n
    p != 0 ? (e.next[p] = s.id) : (reg.first = s)
    n != 0 ? (e.prev[n] = s.id) : (reg.last = s)
    ir.body.region[s.id] = r
    return nothing
end

# Assign an order key strictly between flattened neighbors `lo` and `hi`
# (StmtId or nothing). Global relabel on gap exhaustion.
#
# The key STEPS UP from `lo` by a bounded amount instead of bisecting the whole
# free range. Bisection halves the gap on every insert, so appending a run of
# statements after an advancing `lo` towards a fixed `hi` — exactly what
# `copy_region_into!`/`copy_region_into_fresh!` do when splicing an inlinee —
# exhausts the gap after ~32 statements and triggers a full `relabel_okeys!`.
# That walk is O(body) with a `flat_next` that is itself linear in the region
# nest depth, so splicing into a deeply nested body (e.g. a 1000-block
# `&&`-chain) burned essentially all of its time relabeling. A bounded step
# keeps the same invariant (strictly between the neighbors) while making the
# post-relabel gap last ~2^24 inserts instead of ~32.
const OKEY_STEP = UInt64(1) << 8   # bounded insertion step (relabel spreads 2^32)

function assign_okey!(ir::IR, s::StmtId, lo::Union{Nothing,StmtId}, hi::Union{Nothing,StmtId})
    e = ir.edit::EditState
    for attempt in 1:2
        lokey = lo === nothing ? UInt64(0) : e.okey[lo.id]
        hikey = hi === nothing ? typemax(UInt64) : e.okey[hi.id]
        if hikey - lokey >= 2
            step = (hikey - lokey) >> 1
            step > OKEY_STEP && (step = OKEY_STEP)
            e.okey[s.id] = lokey + step
            return nothing
        end
        attempt == 2 && error("order-key space exhausted")
        relabel_okeys!(ir)
    end
    return nothing
end

"Re-spread all order keys by one flattened traversal."
function relabel_okeys!(ir::IR)
    e = ir.edit::EditState
    i = 1
    s = flat_first(ir)
    while s !== nothing
        e.okey[s.id] = UInt64(i) << 32
        i += 1
        s = flat_next(ir, s)
    end
    return nothing
end

# ---- row allocation ----

function alloc_stmt!(ir::IR, k::Kind, ops::Vector{Operand};
                     type = Any, flag::Union{Nothing,UInt32} = nothing,
                     debug::NTuple{3,Int32} = (Int32(0), Int32(0), Int32(0)))
    info = kindinfo(k)
    check_arity(info, length(ops))
    f = flag === nothing ? info.effects : flag
    info.result == 0 && (type = Nothing)
    w = encode_ops!(ir.body, k, ops)
    s = _append_row!(ir, k, w, type, f, debug, NULL_REGION)
    e = ir.edit::EditState
    push!(e.next, 0); push!(e.prev, 0); push!(e.okey, 0)
    if length(e.firstreg) == s.id - 1     # keep the owned-region links in step
        push!(e.firstreg, Int32(0)); push!(e.lastreg, Int32(0))
    end
    return s
end

mkops(ir::IR, args) = Operand[a isa Operand ? a : vop(ir, a) for a in args]

"""
    insert_before!(ir, at, kind, ops...; type, flag) -> StmtId

Insert a new statement immediately before `at`, in `at`'s region. O(1).
"""
function insert_before!(ir::IR, at::StmtId, k::Kind, args...; kws...)
    check_state(ir, LAYOUT_EDITABLE, "insert_before!")
    s = alloc_stmt!(ir, k, mkops(ir, args); kws...)
    e = ir.edit::EditState
    r = stmt_region(ir, at)
    link_between!(ir, s, r, e.prev[at.id], at.id)
    assign_okey!(ir, s, flat_prev_of_new(ir, s), at)
    ir.cache.stmt_epoch += 1
    return s
end

"""
    insert_after!(ir, at, kind, ops...; type, flag) -> StmtId

Insert a new statement immediately after `at` in `at`'s region list (i.e.
after `at`'s owned regions, if any, in flattened order).
"""
function insert_after!(ir::IR, at::StmtId, k::Kind, args...; kws...)
    check_state(ir, LAYOUT_EDITABLE, "insert_after!")
    s = alloc_stmt!(ir, k, mkops(ir, args); kws...)
    e = ir.edit::EditState
    r = stmt_region(ir, at)
    nxt = e.next[at.id]
    link_between!(ir, s, r, at.id, nxt)
    assign_okey!(ir, s, deep_last(ir, at), nxt == 0 ? flat_next(ir, s) : StmtId(nxt))
    ir.cache.stmt_epoch += 1
    return s
end

"""
    push_stmt!(ir, region, kind, ops...; type, flag) -> StmtId

Append a statement at the end of `region`'s list.
"""
function push_stmt!(ir::IR, r::RegionId, k::Kind, args...; kws...)
    check_state(ir, LAYOUT_EDITABLE, "push_stmt!")
    s = alloc_stmt!(ir, k, mkops(ir, args); kws...)
    reg = getregion(ir, r)
    tail = reg.last.id
    link_between!(ir, s, r, tail, Int32(0))
    lo = tail != 0 ? deep_last(ir, StmtId(tail)) : region_entry_anchor(ir, r)
    assign_okey!(ir, s, lo, flat_next(ir, s))
    if k === K"region_arg"
        push!(reg.args, s)
    end
    ir.cache.stmt_epoch += 1
    return s
end

# Flattened predecessor of an empty region's first insertion point: the
# owner's previous owned-region tail, or the owner itself.
function region_entry_anchor(ir::IR, r::RegionId)
    reg = getregion(ir, r)
    isnull(reg.owner) && return nothing
    e = region_links(ir)
    q = e.prevsib[r.id]
    while q != 0
        t = ir.regions[q].last
        t.id != 0 && return deep_last(ir, StmtId(t.id))
        q = e.prevsib[q]
    end
    return reg.owner
end

# For a freshly linked stmt: its flattened predecessor (walking the links that
# are already in place).
function flat_prev_of_new(ir::IR, s::StmtId)
    e = ir.edit::EditState
    p = e.prev[s.id]
    p != 0 && return deep_last(ir, StmtId(p))
    return region_entry_anchor(ir, stmt_region(ir, s))
end
