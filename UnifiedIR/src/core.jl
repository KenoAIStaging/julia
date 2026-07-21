# The core data structure (§3): statement table, region table, IR handle,
# layout states, analysis-cache epochs.

# ---------------------------------------------------------------------------
# Regions (§3.3)
# ---------------------------------------------------------------------------

# (Bootstrap dialect: UInt8-wrapper structs + named consts in place of @enum —
# see compat.jl.)
struct RegionKind
    x::UInt8
end
const REGION_BODY      = RegionKind(0x00)
const REGION_ARM       = RegionKind(0x01)
const REGION_GUARD     = RegionKind(0x02)
const REGION_LOOP_BODY = RegionKind(0x03)
const REGION_HANDLER   = RegionKind(0x04)
const REGION_BLOCK     = RegionKind(0x05)
const _REGION_KIND_NAMES = (:REGION_BODY, :REGION_ARM, :REGION_GUARD,
                            :REGION_LOOP_BODY, :REGION_HANDLER, :REGION_BLOCK)
Base.show(io::IO, k::RegionKind) = print(io, _REGION_KIND_NAMES[Int(k.x) + 1])

struct Activation
    x::UInt8
end
const ACT_IMMEDIATE = Activation(0x00)
const ACT_DEFERRED  = Activation(0x01)
const ACT_RESUME    = Activation(0x02)
const _ACTIVATION_NAMES = (:ACT_IMMEDIATE, :ACT_DEFERRED, :ACT_RESUME)
Base.show(io::IO, a::Activation) = print(io, _ACTIVATION_NAMES[Int(a.x) + 1])

mutable struct Region
    kind::RegionKind
    activation::Activation
    owner::StmtId            # 0 for region 1 (root) and `guard` regions
    parent::RegionId
    args::Vector{StmtId}     # region_arg statements (leading positions)
    cond::Value              # STORED ONLY for guard regions (owner == 0)
    negated::Bool
    first::StmtId            # dense: span endpoints; editable: list head/tail
    last::StmtId
    dead::Bool               # killed by surgery; dropped at compact!
end

Region(kind::RegionKind, owner::StmtId, parent::RegionId;
       activation::Activation = ACT_IMMEDIATE,
       cond::Value = NULL_STMT, negated::Bool = false) =
    Region(kind, activation, owner, parent, StmtId[], cond, negated, NULL_STMT, NULL_STMT, false)

is_guard(r::Region) = r.kind === REGION_GUARD
is_ordered(r::Region) = r.kind !== REGION_GUARD

# ---------------------------------------------------------------------------
# Statement body (§3.1)
# ---------------------------------------------------------------------------

# IRBody = AttrGraph + IR-only columns and pools (§3.7 Level 1: one shared
# storage core — SyntaxGraph wraps the same AttrGraph with tree conventions;
# UnifiedIR adds the type/flag/debug/region columns, constant/global pools,
# regions, and layout states).
#
# The historical row fields (:len/:kind/:ops/:operands/:cols) live in the
# AttrGraph and are exposed through property forwarding, so the rest of the
# codebase keeps addressing `body.kind` etc. unchanged.
mutable struct IRBody{Cols}
    # THE shared storage core (§3.7): len + kind column + packed two-mode ops
    # words + tagged operand pool + extension columns (§3.5)
    const graph::AttrGraph{Cols}
    # IR-only core columns, StmtId-indexed
    type::Vector{Any}
    flag::Vector{UInt32}
    debug::Vector{NTuple{3,Int32}}
    region::Vector{RegionId}
    # IR-only shared pools (append-only within a generation)
    constants::Vector{Any}
    constmap::IdDict{Any,Int}     # egal interning (§13.8)
    globals::Vector{GlobalRef}
    globalmap::IdDict{GlobalRef,Int}
end

IRBody(cols) = IRBody{typeof(cols)}(AttrGraph(cols), Any[], UInt32[],
                                    NTuple{3,Int32}[], RegionId[],
                                    Any[], IdDict{Any,Int}(),
                                    GlobalRef[], IdDict{GlobalRef,Int}())

@inline function Base.getproperty(b::IRBody, name::Symbol)
    if name === :len || name === :kind || name === :ops || name === :operands ||
       name === :cols
        return getproperty(getfield(b, :graph), name)
    end
    return getfield(b, name)
end

@inline function Base.setproperty!(b::IRBody, name::Symbol, v)
    if name === :len
        return setfield!(getfield(b, :graph), :len, convert(Int32, v))
    elseif name === :kind || name === :ops || name === :operands || name === :cols
        return setproperty!(getfield(b, :graph), name, v)
    end
    return setfield!(b, name, v)
end

Base.propertynames(b::IRBody) =
    (:graph, :len, :kind, :ops, :operands, :cols,
     :type, :flag, :debug, :region, :constants, :constmap, :globals, :globalmap)

# ---------------------------------------------------------------------------
# Analysis cache (§11.1)
# ---------------------------------------------------------------------------

mutable struct AnalysisCache
    # key = (analysis type, config). Egal-keyed (bootstrap dialect + compiler
    # hygiene): keys must be Symbols, Strings, or egal-comparable bits
    # values — a MUTABLE config object (e.g. a Vector) is keyed by identity,
    # not content.
    entries::IdDict{Any,Any}
    stmt_epoch::UInt64
    region_epoch::UInt64
    type_epoch::UInt64
    flag_epoch::UInt64
    layout_epoch::UInt64
end
AnalysisCache() = AnalysisCache(IdDict{Any,Any}(), 0, 0, 0, 0, 0)

# ---------------------------------------------------------------------------
# Layout states and the IR handle (§2.2, §3.1)
# ---------------------------------------------------------------------------

struct LayoutState
    x::UInt8
end
const LAYOUT_BUILDER  = LayoutState(0x00)
const LAYOUT_DENSE    = LayoutState(0x01)
const LAYOUT_EDITABLE = LayoutState(0x02)
const LAYOUT_FLOATING = LayoutState(0x03)
const _LAYOUT_STATE_NAMES = (:LAYOUT_BUILDER, :LAYOUT_DENSE, :LAYOUT_EDITABLE,
                             :LAYOUT_FLOATING)
Base.show(io::IO, st::LayoutState) = print(io, _LAYOUT_STATE_NAMES[Int(st.x) + 1])

mutable struct BodyOwner
    state::LayoutState
    generation::UInt32
end

mutable struct EditState
    next::Vector{Int32}
    prev::Vector{Int32}
    okey::Vector{UInt64}      # order-key accelerator (global flattened order)
end

"""
    IR{Cols}

The IR handle: one shared body owner (layout-state tag + generation), the
statement table, the region table, and function-level metadata. `Cols` is the
consumer's column universe (§3.5).
"""
mutable struct IR{Cols}
    owner::BodyOwner
    body::IRBody{Cols}
    regions::Vector{Region}
    argtypes::Vector{Any}
    sptypes::Vector{Any}
    valid_worlds::Tuple{UInt64,UInt64}
    edit::Union{Nothing,EditState}
    pending::Vector{Pair{StmtId,Operand}}   # queued replace_uses!
    cache::AnalysisCache
    meta::IdDict{Symbol,Any}  # :name, :module, source linetable, etc.
end

const NOCOLS = NamedTuple()

layout(ir::IR) = ir.owner.state
generation(ir::IR) = ir.owner.generation
nstmts(ir::IR) = Int(ir.body.len)
nregions(ir::IR) = length(ir.regions)

function check_state(ir::IR, want::LayoutState, what::String)
    layout(ir) === want || error(LazyString(what, " requires ", want, " layout state; IR is in ", layout(ir)))
    return nothing
end
function check_state(ir::IR, want::Tuple{Vararg{LayoutState}}, what::String)
    layout(ir) in want || error(LazyString(what, " requires one of ", want, "; IR is in ", layout(ir)))
    return nothing
end

getregion(ir::IR, r::RegionId) = ir.regions[r.id]
root_region(ir::IR) = RegionId(1)

stmt_region(ir::IR, s::StmtId) = ir.body.region[s.id]
stmt_kind(ir::IR, s::StmtId) = ir.body.kind[s.id]
stmt_type(ir::IR, s::StmtId) = ir.body.type[s.id]
stmt_flag(ir::IR, s::StmtId) = ir.body.flag[s.id]
stmt_debug(ir::IR, s::StmtId) = ir.body.debug[s.id]
is_tombstone(ir::IR, s::StmtId) = ir.body.kind[s.id] === KIND_DELETED

function set_type!(ir::IR, s::StmtId, @nospecialize(t))
    ir.body.type[s.id] = t
    ir.cache.type_epoch += 1
    return t
end
function set_flag!(ir::IR, s::StmtId, f::UInt32)
    ir.body.flag[s.id] = f
    ir.cache.flag_epoch += 1
    return f
end
add_flag!(ir::IR, s::StmtId, f::UInt32) = set_flag!(ir, s, stmt_flag(ir, s) | f)

# ---------------------------------------------------------------------------
# Operand access (mode-dispatching; §3.2)
# ---------------------------------------------------------------------------

function nops(ir::IR, s::StmtId)
    w = ir.body.ops[s.id]
    is_ops_inline(w) ? inline_arity(w) : Int(ops_len(w))
end

function getop(ir::IR, s::StmtId, i::Integer)::Operand
    w = ir.body.ops[s.id]
    if is_ops_inline(w)
        a = inline_arity(w)
        1 <= i <= a || throw(BoundsError(LazyString("operand ", i, " of ", a, "-ary inline stmt")))
        return i == 1 ? op_stmt(inline_stmt(w)) : op_inline(inline_imm(w))
    else
        len = ops_len(w)
        1 <= i <= len || throw(BoundsError(LazyString("operand ", i, " of ", len, "-ary stmt")))
        return ir.body.operands[ops_offset(w) + i]
    end
end

"Set operand `i` in place. Inline mode only permits replacing the STMT slot."
function setop!(ir::IR, s::StmtId, i::Integer, o::Operand)
    w = ir.body.ops[s.id]
    if is_ops_inline(w)
        if i == 1 && optag(o) == TAG_STMT
            ir.body.ops[s.id] = set_inline_stmt(w, asstmt(o))
        elseif i == 2 && optag(o) == TAG_INLINE
            imm = Int(imm_value(o))
            ir.body.ops[s.id] = ops_inline(inline_stmt(ir.body.ops[s.id]), imm, inline_arity(w))
        else
            error(LazyString("cannot store ", optag(o), "-tagged operand into inline slot ", i))
        end
    else
        len = ops_len(w)
        1 <= i <= len || throw(BoundsError(LazyString("operand ", i, " of ", len, "-ary stmt")))
        ir.body.operands[ops_offset(w) + i] = o
    end
    ir.cache.stmt_epoch += 1
    return o
end

"All operands of `s` as a (small) vector — convenience, allocates."
function operands(ir::IR, s::StmtId)
    n = nops(ir, s)
    [getop(ir, s, i) for i in 1:n]
end

# Store a fresh operand list for statement `s` (append to pool or inline-encode).
function store_ops!(ir::IR, s::StmtId, ops::Vector{Operand})
    k = ir.body.kind[s.id]
    w = encode_ops!(ir.body, k, ops)
    ir.body.ops[s.id] = w
    ir.cache.stmt_epoch += 1
    return nothing
end

function encode_ops!(body::IRBody, k::Kind, ops::Vector{Operand})::UInt64
    if has_inline_ops(k)
        # schema guarantees: exactly one STMT (first) + optional immediate
        if length(ops) == 1 && optag(ops[1]) == TAG_STMT
            return ops_inline(asstmt(ops[1]), nothing, 1)
        elseif length(ops) == 2 && optag(ops[1]) == TAG_STMT && optag(ops[2]) == TAG_INLINE
            imm = Int(imm_value(ops[2])::Int64)
            if -(1 << 23) <= imm < (1 << 23)
                return ops_inline(asstmt(ops[1]), imm, 2)
            end
        end
        # fall through to pool encoding for out-of-range immediates
    end
    # shared substrate pool append (same primitive setchildren! grows through)
    return pool_append!(body.graph, ops)
end

# ---------------------------------------------------------------------------
# Constant / global interning
# ---------------------------------------------------------------------------

"Intern a constant by egal identity (§13.8); returns pool index."
function intern_const!(body::IRBody, @nospecialize(v))
    get!(body.constmap, v) do
        push!(body.constants, v)
        length(body.constants)
    end
end

function intern_global!(body::IRBody, g::GlobalRef)
    get!(body.globalmap, g) do
        push!(body.globals, g)
        length(body.globals)
    end
end

getconst(ir::IR, o::Operand) = (@assert optag(o) == TAG_CONST; ir.body.constants[payload(o)])
getglobal_op(ir::IR, o::Operand) = (@assert optag(o) == TAG_GLOBAL; ir.body.globals[payload(o)])

"Build a value operand from a Julia value: statements pass through, small ints
inline, everything else interns into the constant pool."
function vop(ir::IR, @nospecialize(x))
    x isa StmtId && return op_stmt(x)
    x isa Operand && return x
    if x isa Int64 && -(Int64(1) << 55) <= x < (Int64(1) << 55)
        return op_inline(x)
    elseif x isa Bool || x isa UInt8
        return op_inline(x)
    elseif x isa GlobalRef
        return op_globalidx(intern_global!(ir.body, x))
    else
        return op_constidx(intern_const!(ir.body, x))
    end
end

"Decode a *value* operand to (kind, value) where kind ∈ (:stmt, :const, :sparam)."
function op_value(ir::IR, o::Operand)
    t = optag(o)
    t == TAG_STMT && return (:stmt, asstmt(o))
    t == TAG_INLINE && return (:const, imm_value(o))
    t == TAG_CONST && return (:const, ir.body.constants[payload(o)])
    t == TAG_GLOBAL && return (:global, ir.body.globals[payload(o)])
    t == TAG_SPARAM && return (:sparam, Int(payload(o)))
    error(LazyString("operand tag ", t, " is not a value operand"))
end

# ---------------------------------------------------------------------------
# Region tree queries
# ---------------------------------------------------------------------------

"Is `a` an ancestor of (or equal to) `r`?"
function is_ancestor(ir::IR, a::RegionId, r::RegionId)
    while !isnull(r)
        r == a && return true
        r = getregion(ir, r).parent
    end
    return false
end

"Regions owned by statement `s`, in table order (they are contiguous by construction)."
function owned_regions(ir::IR, s::StmtId)
    out = RegionId[]
    for (i, r) in enumerate(ir.regions)
        r.owner == s && push!(out, RegionId(i))
    end
    return out
end

"Innermost activation root region containing `r` (walks past immediate regions)."
function activation_root(ir::IR, r::RegionId)
    while !isnull(r)
        reg = getregion(ir, r)
        reg.activation !== ACT_IMMEDIATE && return r
        isnull(reg.parent) && return r
        r = reg.parent
    end
    return r
end

"Depth of region `r` in the region tree (root = 1)."
function region_depth(ir::IR, r::RegionId)
    d = 0
    while !isnull(r)
        d += 1
        r = getregion(ir, r).parent
    end
    return d
end

# ---------------------------------------------------------------------------
# Order and visibility (§5.1)
# ---------------------------------------------------------------------------

function comes_before(ir::IR, a::StmtId, b::StmtId)
    st = layout(ir)
    if st === LAYOUT_EDITABLE
        e = ir.edit::EditState
        return e.okey[a.id] < e.okey[b.id]
    elseif st === LAYOUT_FLOATING
        error("comes_before is undefined in floating state")
    else
        return a.id < b.id
    end
end

"""
    visible(ir, def, use_at) -> Bool

The visibility rule (§5.1): (1) region ancestry, (2) order, (3) not within a
region owned by the def. Clauses 1–3 apply within one activation.
"""
function visible(ir::IR, def::StmtId, use_at::StmtId)
    dr = stmt_region(ir, def)
    ur = stmt_region(ir, use_at)
    # clause 3: use not within any region owned by def (checked on the walk up)
    r = ur
    found_ancestor = false
    while !isnull(r)
        reg = getregion(ir, r)
        reg.owner == def && return false
        if r == dr
            found_ancestor = true
        end
        r = reg.parent
    end
    # clause 1 — with the §5.5 island refinement: inside a `cfg` island,
    # cross-block visibility is classical dominance, computed locally.
    found_ancestor || return island_visible(ir, def, use_at)
    if layout(ir) !== LAYOUT_FLOATING            # clause 2 (dropped when floating)
        comes_before(ir, def, use_at) || return false
    end
    return true
end

"Innermost `block`-kind region enclosing `s` (NULL_REGION if none)."
function enclosing_block(ir::IR, s::StmtId)
    r = stmt_region(ir, s)
    while !isnull(r)
        reg = getregion(ir, r)
        reg.kind === REGION_BLOCK && return r
        r = reg.parent
    end
    return NULL_REGION
end

# Cross-block visibility inside one cfg island: the def's block must strictly
# dominate the use's block (same-block cases are handled by the main rule).
# The dominator computation is local to the island (§5.5, §11.1). For a use
# nested inside deeper structure (an `if` arm, or a nested island spliced
# into a block — the inlining case), the relevant use block is the one on the
# use's region ancestry belonging to the DEF's island.
function island_visible(ir::IR, def::StmtId, use_at::StmtId)
    db = enclosing_block(ir, def)
    isnull(db) && return false
    downer = getregion(ir, db).owner
    isnull(downer) && return false
    # climb the use's region ancestry to a block of the same island
    r = stmt_region(ir, use_at)
    ub = NULL_REGION
    while !isnull(r)
        reg = getregion(ir, r)
        if reg.kind === REGION_BLOCK && reg.owner == downer
            ub = r
            break
        end
        r = reg.parent
    end
    (isnull(ub) || db == ub) && return false   # same-block: main rule (ancestry)
    dom = island_dominators(ir, downer)
    blocks = get(dom, ub, nothing)
    blocks === nothing && return true    # unreachable use site: vacuously visible
    return db in blocks
end

"""
    island_dominators(ir, cfgop) -> IdDict{RegionId,IdSet{RegionId}}

Dominator sets over one island's block graph (entry = first owned block;
edges from terminator edge bundles); each entry maps an entry-reachable block
to its dominator blocks. BitSet-based iteration over local block indices —
islands are small and the result is a candidate `AnalysisCache` entry.
"""
function island_dominators(ir::IR, cfgop::StmtId)
    blocks = RegionId[]
    for (i, reg) in enumerate(ir.regions)
        reg.owner == cfgop && reg.kind === REGION_BLOCK && !reg.dead &&
            push!(blocks, RegionId(i))
    end
    dom = IdDict{RegionId,IdSet{RegionId}}()
    isempty(blocks) && return dom
    nb = length(blocks)
    lidx = IdDict{Int32,Int}()          # region id -> local block index
    for (i, b) in enumerate(blocks)
        lidx[b.id] = i
    end
    # local successor lists; cross-island successors (sealed exits, §5.5) are
    # not part of the local graph
    succs = Vector{Vector{Int}}(undef, nb)
    for (bi, b) in enumerate(blocks)
        ss = Int[]
        t = region_terminator(ir, b)
        if t !== nothing && (stmt_kind(ir, t) === K"goto" || stmt_kind(ir, t) === K"br_if" ||
                             stmt_kind(ir, t) === K"switch" || stmt_kind(ir, t) === K"await")
            for (dest, _) in edge_bundles(ir, t)
                si = get(lidx, dest.id, 0)
                si == 0 || push!(ss, si)
            end
        end
        succs[bi] = ss
    end
    # dominators are defined over the entry-reachable subgraph of THIS island;
    # unreachable blocks get no entry (use sites vacuously visible).
    reach = falses(nb)
    reach[1] = true
    stack = Int[1]
    while !isempty(stack)
        bi = pop!(stack)
        for si in succs[bi]
            reach[si] || (reach[si] = true; push!(stack, si))
        end
    end
    all_set = BitSet()
    for bi in 1:nb
        reach[bi] && push!(all_set, bi)
    end
    ldom = Vector{BitSet}(undef, nb)
    ldom[1] = BitSet((1,))
    for bi in 2:nb
        reach[bi] && (ldom[bi] = copy(all_set))
    end
    preds = [Int[] for _ in 1:nb]
    for bi in 1:nb
        reach[bi] || continue
        for si in succs[bi]
            reach[si] && push!(preds[si], bi)
        end
    end
    changed = true
    while changed
        changed = false
        for bi in 2:nb
            reach[bi] || continue
            ps = preds[bi]
            newset = if isempty(ps)
                BitSet((bi,))
            else
                t = copy(ldom[ps[1]])
                for k in 2:length(ps)
                    intersect!(t, ldom[ps[k]])
                end
                push!(t, bi)
                t
            end
            if newset != ldom[bi]
                ldom[bi] = newset
                changed = true
            end
        end
    end
    for bi in 1:nb
        reach[bi] || continue
        s = IdSet{RegionId}()
        for li in ldom[bi]
            push!(s, blocks[li])
        end
        dom[blocks[bi]] = s
    end
    return dom
end

# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------

function Base.show(io::IO, ir::IR)
    print(io, "IR(", nstmts(ir), " stmts, ", nregions(ir), " regions, ",
          layout(ir), ", gen ", generation(ir), ")")
end
