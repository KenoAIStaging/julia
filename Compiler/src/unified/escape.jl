# UnifiedIR-native escape analysis (B4; stock reference:
# Compiler/src/ssair/EscapeAnalysis.jl — the LATTICE and the transfer CASES,
# re-expressed over the region tree instead of flat SSA).
#
# The stock analysis is a backward dataflow over IRCode: `EscapeInfo`
# (ReturnEscape / ThrownEscape sites / AliasInfo field sets / Liveness sites)
# per Argument/SSAValue, an alias union-find for φ/π/typeassert/ifelse
# equivalences, and a blanket AllEscape for values throwable into a `try`
# (see stock `escape_exception!` for why the blanket rule is the honest v1).
#
# On UnifiedIR every analyzable value IS a statement (`region_arg`s subsume
# `Argument`s), so the state is one vector indexed by StmtId. The flat-IR
# vocabulary maps structurally:
#
#   PhiNode          → region-result feeding: an owner op (`if`/`loop`/`try`/
#                      `cfg`) aliases the operands of its `result`/`break`/
#                      `continue` terminators (arity-1), or acts as a
#                      tuple-shaped object whose "fields" the terminators
#                      define and `extract` loads (arity-n results)
#   loop-carried φ   → loop-body `region_arg`s alias the loop op's init
#                      operands and each `continue`'s carried values
#   cfg block args   → alias the edge-bundle values of `goto`/`br_if`/`switch`
#   PiNode           → `refine` aliases its operand
#   PhiC/Upsilon     → residual `cell`s: a 1-field object (`cell_set` defines,
#                      `cell_get` loads through the same field machinery)
#   EnterNode regions→ the `try` op's body region; "thrown inside a try"
#                      is region ancestry, not a pc-range scan
#   Argument(n)      → root-region `region_arg` position n
#   SSA statement pc → StmtId (ThrownEscape/Liveness sites are stmt ids)
#
# Analysis addressing is structural: `analyze_escapes(ir, nargs) ->
# UEscapeResult`, queried by StmtId or argument position — never by
# positional integer indexing into a statement stream.

# ---------------------------------------------------------------------------
# Lattice (stock EscapeInfo, verbatim semantics)
# ---------------------------------------------------------------------------

const EAInfo = Base.IdSet{Any}

"A field-set def entry: the object's field is defined at statement `idx`."
struct EALocalDef
    idx::Int
end
"A field-set use entry: the field is loaded by statement `idx` (its result
aliases whatever the field holds)."
struct EALocalUse
    idx::Int
end

struct EAIndexableFields
    infos::Vector{EAInfo}
end
struct EAUnindexable
    info::EAInfo
end
EAIndexableFields(nflds::Int) = EAIndexableFields(EAInfo[EAInfo() for _ in 1:nflds])
EAUnindexable() = EAUnindexable(EAInfo())
Base.copy(a::EAIndexableFields) = EAIndexableFields(EAInfo[copy(i) for i in a.infos])
Base.copy(a::EAUnindexable) = EAUnindexable(copy(a.info))

ea_merge_to_unindexable(a::EAIndexableFields) = EAUnindexable(ea_merge_infos(EAInfo(), a.infos))
ea_merge_to_unindexable(a::EAUnindexable, b::EAIndexableFields) = EAUnindexable(ea_merge_infos(copy(a.info), b.infos))
function ea_merge_infos(info::EAInfo, infos::Vector{EAInfo})
    for i in 1:length(infos)
        info = info ∪ infos[i]
    end
    return info
end

"""
    x::UEscapeInfo

Stock `EscapeAnalysis.EscapeInfo` with structural site addressing:
`ThrownEscape`/`Liveness` hold *statement ids* (of the may-throw statement /
the use site). `0 ∈ Liveness` marks a call argument of the analyzed frame;
`-1 ∈ ThrownEscape`/`-1 ∈ Liveness` are the respective tops.
"""
struct UEscapeInfo
    Analyzed::Bool
    ReturnEscape::Bool
    ThrownEscape::Base.BitSet
    AliasInfo    #::Union{Bool,EAIndexableFields,EAUnindexable}
    Liveness::Base.BitSet
    function UEscapeInfo(Analyzed::Bool, ReturnEscape::Bool, ThrownEscape::Base.BitSet,
                         AliasInfo, Liveness::Base.BitSet)
        @nospecialize AliasInfo
        return new(Analyzed, ReturnEscape, ThrownEscape, AliasInfo, Liveness)
    end
    function UEscapeInfo(x::UEscapeInfo, AliasInfo = x.AliasInfo;
                         Analyzed::Bool = x.Analyzed,
                         ReturnEscape::Bool = x.ReturnEscape,
                         ThrownEscape::Base.BitSet = x.ThrownEscape,
                         Liveness::Base.BitSet = x.Liveness)
        @nospecialize AliasInfo
        return new(Analyzed, ReturnEscape, ThrownEscape, AliasInfo, Liveness)
    end
end

const EA_BOT_THROWN = Base.BitSet()
const EA_TOP_THROWN = Base.BitSet(-1)
const EA_BOT_LIVE = Base.BitSet()
const EA_TOP_LIVE = Base.BitSet(-1:0)
const EA_ARG_LIVE = Base.BitSet(0)

EANotAnalyzed() = UEscapeInfo(false, false, EA_BOT_THROWN, false, EA_BOT_LIVE)
EANoEscape() = UEscapeInfo(true, false, EA_BOT_THROWN, false, EA_BOT_LIVE)
EAArgEscape() = UEscapeInfo(true, false, EA_BOT_THROWN, true, EA_ARG_LIVE)
EAReturnEscape(pc::Int) = UEscapeInfo(true, true, EA_BOT_THROWN, false, Base.BitSet(pc))
EAThrownEscape(pc::Int) = UEscapeInfo(true, false, Base.BitSet(pc), false, EA_BOT_LIVE)
EAAllEscape() = UEscapeInfo(true, true, EA_TOP_THROWN, true, EA_TOP_LIVE)

const EA_⊥ = EANotAnalyzed()
const EA_⊤ = EAAllEscape()

# convenience queries (stock names, exported by the harness)
has_no_escape(x::UEscapeInfo) = !x.ReturnEscape && isempty(x.ThrownEscape) && 0 ∉ x.Liveness
has_arg_escape(x::UEscapeInfo) = 0 ∈ x.Liveness
has_return_escape(x::UEscapeInfo) = x.ReturnEscape
has_return_escape(x::UEscapeInfo, at::StmtId) =
    x.ReturnEscape && (-1 ∈ x.Liveness || at.id ∈ x.Liveness)
has_thrown_escape(x::UEscapeInfo) = !isempty(x.ThrownEscape)
has_thrown_escape(x::UEscapeInfo, at::StmtId) =
    -1 ∈ x.ThrownEscape || at.id ∈ x.ThrownEscape
has_all_escape(x::UEscapeInfo) = ea_issub(EA_⊤, x)
is_load_forwardable(x::UEscapeInfo) = x.AliasInfo isa EAIndexableFields
ignore_argescape(x::UEscapeInfo) = UEscapeInfo(x; Liveness = delete!(copy(x.Liveness), 0))
ignore_aliasinfo(x::UEscapeInfo) = UEscapeInfo(x, false)

"Lattice equality (stock `==`; convergence detection)."
function ea_eq(x::UEscapeInfo, y::UEscapeInfo)
    x === y && return true
    x.Analyzed === y.Analyzed || return false
    x.ReturnEscape === y.ReturnEscape || return false
    xt, yt = x.ThrownEscape, y.ThrownEscape
    if xt === EA_TOP_THROWN
        yt === EA_TOP_THROWN || return false
    elseif yt === EA_TOP_THROWN
        return false
    else
        xt == yt || return false
    end
    xa, ya = x.AliasInfo, y.AliasInfo
    if xa isa Bool
        xa === ya || return false
    elseif xa isa EAIndexableFields
        ya isa EAIndexableFields || return false
        xa.infos == ya.infos || return false
    else
        xa = xa::EAUnindexable
        ya isa EAUnindexable || return false
        xa.info == ya.info || return false
    end
    xl, yl = x.Liveness, y.Liveness
    if xl === EA_TOP_LIVE
        yl === EA_TOP_LIVE || return false
    elseif yl === EA_TOP_LIVE
        return false
    else
        xl == yl || return false
    end
    return true
end

"The non-strict partial order (stock ⊑ₑ)."
function ea_issub(x::UEscapeInfo, y::UEscapeInfo)
    if y === EA_⊤
        return true
    elseif x === EA_⊤
        return false
    elseif x === EA_⊥
        return true
    elseif y === EA_⊥
        return false
    end
    x.Analyzed ≤ y.Analyzed || return false
    x.ReturnEscape ≤ y.ReturnEscape || return false
    xt, yt = x.ThrownEscape, y.ThrownEscape
    if xt === EA_TOP_THROWN
        yt !== EA_TOP_THROWN && return false
    elseif yt !== EA_TOP_THROWN
        xt ⊆ yt || return false
    end
    xa, ya = x.AliasInfo, y.AliasInfo
    if xa isa Bool
        xa && ya !== true && return false
    elseif xa isa EAIndexableFields
        if ya isa EAIndexableFields
            xinfos, yinfos = xa.infos, ya.infos
            length(xinfos) > length(yinfos) && return false
            for i in 1:length(xinfos)
                xinfos[i] ⊆ yinfos[i] || return false
            end
        elseif ya isa EAUnindexable
            for i in 1:length(xa.infos)
                xa.infos[i] ⊆ ya.info || return false
            end
        else
            ya === true || return false
        end
    else
        xa = xa::EAUnindexable
        if ya isa EAUnindexable
            xa.info ⊆ ya.info || return false
        else
            ya === true || return false
        end
    end
    xl, yl = x.Liveness, y.Liveness
    if xl === EA_TOP_LIVE
        yl !== EA_TOP_LIVE && return false
    elseif yl !== EA_TOP_LIVE
        xl ⊆ yl || return false
    end
    return true
end

"The join (stock ⊔ₑ)."
function ea_join(x::UEscapeInfo, y::UEscapeInfo)
    if x === EA_⊤ || y === EA_⊤
        return EA_⊤
    elseif x === EA_⊥
        return y
    elseif y === EA_⊥
        return x
    end
    xt, yt = x.ThrownEscape, y.ThrownEscape
    if xt === EA_TOP_THROWN || yt === EA_TOP_THROWN
        ThrownEscape = EA_TOP_THROWN
    elseif xt === EA_BOT_THROWN
        ThrownEscape = yt
    elseif yt === EA_BOT_THROWN
        ThrownEscape = xt
    else
        ThrownEscape = xt ∪ yt
    end
    AliasInfo = ea_merge_alias_info(x.AliasInfo, y.AliasInfo)
    xl, yl = x.Liveness, y.Liveness
    if xl === EA_TOP_LIVE || yl === EA_TOP_LIVE
        Liveness = EA_TOP_LIVE
    elseif xl === EA_BOT_LIVE
        Liveness = yl
    elseif yl === EA_BOT_LIVE
        Liveness = xl
    else
        Liveness = xl ∪ yl
    end
    return UEscapeInfo(x.Analyzed | y.Analyzed, x.ReturnEscape | y.ReturnEscape,
                       ThrownEscape, AliasInfo, Liveness)
end

function ea_merge_alias_info(@nospecialize(xa), @nospecialize(ya))
    if xa === true || ya === true
        return true
    elseif xa === false
        return ya
    elseif ya === false
        return xa
    elseif xa isa EAIndexableFields
        if ya isa EAIndexableFields
            xinfos, yinfos = xa.infos, ya.infos
            xn, yn = length(xinfos), length(yinfos)
            nmax, nmin = max(xn, yn), min(xn, yn)
            infos = Vector{EAInfo}(undef, nmax)
            for i in 1:nmax
                if i > nmin
                    infos[i] = (xn > yn ? xinfos : yinfos)[i]
                else
                    infos[i] = xinfos[i] ∪ yinfos[i]
                end
            end
            return EAIndexableFields(infos)
        elseif ya isa EAUnindexable
            return ea_merge_to_unindexable(ya, xa)
        else
            return true
        end
    else
        xa = xa::EAUnindexable
        if ya isa EAIndexableFields
            return ea_merge_to_unindexable(xa, ya)
        elseif ya isa EAUnindexable
            return EAUnindexable(xa.info ∪ ya.info)
        else
            return true
        end
    end
end

# ---------------------------------------------------------------------------
# Alias set (union-find over statement ids; stock IntDisjointSet dies with
# ssair/ — a self-contained port lives here)
# ---------------------------------------------------------------------------

mutable struct EAAliasSet
    parents::Vector{Int}
    ranks::Vector{Int}
end
EAAliasSet(n::Int) = EAAliasSet(collect(1:n), zeros(Int, n))

function ea_find_root!(a::EAAliasSet, x::Int)
    p = a.parents[x]
    if p != x
        a.parents[x] = p = ea_find_root!(a, p)
    end
    return p
end

ea_in_same_set(a::EAAliasSet, x::Int, y::Int) = ea_find_root!(a, x) == ea_find_root!(a, y)

function ea_union!(a::EAAliasSet, x::Int, y::Int)
    xr, yr = ea_find_root!(a, x), ea_find_root!(a, y)
    xr == yr && return xr
    if a.ranks[xr] < a.ranks[yr]
        xr, yr = yr, xr
    elseif a.ranks[xr] == a.ranks[yr]
        a.ranks[xr] += 1
    end
    a.parents[yr] = xr
    return xr
end

"All members of `x`'s alias set (nothing when the set is a singleton)."
function ea_aliases(a::EAAliasSet, x::Int)
    root = ea_find_root!(a, x)
    if x != root || a.ranks[x] > 0
        out = Int[]
        for i in 1:length(a.parents)
            ea_find_root!(a, i) == root && push!(out, i)
        end
        return out
    end
    return nothing
end

# ---------------------------------------------------------------------------
# State and results
# ---------------------------------------------------------------------------

"""
    UEscapeState

Escape information per statement (StmtId-indexed) plus the alias set and the
argument table (`argids[i]` = the root-region `region_arg` for argument
position `i` — stock's `Argument(i)`).
"""
struct UEscapeState
    escapes::Vector{UEscapeInfo}
    aliasset::EAAliasSet
    nargs::Int
    argids::Vector{Int}
end

Base.getindex(state::UEscapeState, s::StmtId) = state.escapes[s.id]

"Escape info of argument position `i` (1 = the function itself)."
argescape(state::UEscapeState, i::Int) = state.escapes[state.argids[i]]
"StmtId of argument position `i`."
argstmt(state::UEscapeState, i::Int) = StmtId(Int32(state.argids[i]))

isaliased(state::UEscapeState, x::StmtId, y::StmtId) =
    ea_in_same_set(state.aliasset, Int(x.id), Int(y.id))
isaliased_arg(state::UEscapeState, i::Int, y::StmtId) =
    ea_in_same_set(state.aliasset, state.argids[i], Int(y.id))

"""
    UEscapeResult

`analyze_escapes` output: the analyzed IR and its `UEscapeState`.
"""
struct UEscapeResult
    ir::UnifiedIR.IR
    state::UEscapeState
end

# --- interprocedural summary (stock ArgEscapeInfo/ArgEscapeCache) ----------

const EA_ARG_ALL_ESCAPE = 0x01 << 0
const EA_ARG_RETURN_ESCAPE = 0x01 << 1
const EA_ARG_THROWN_ESCAPE = 0x01 << 2

struct UArgEscapeInfo
    escape_bits::UInt8
end
function UArgEscapeInfo(x::UEscapeInfo)
    has_all_escape(x) && return UArgEscapeInfo(EA_ARG_ALL_ESCAPE)
    bits = 0x00
    has_return_escape(x) && (bits |= EA_ARG_RETURN_ESCAPE)
    has_thrown_escape(x) && (bits |= EA_ARG_THROWN_ESCAPE)
    return UArgEscapeInfo(bits)
end
has_all_escape(x::UArgEscapeInfo) = x.escape_bits & EA_ARG_ALL_ESCAPE ≠ 0
has_return_escape(x::UArgEscapeInfo) = x.escape_bits & EA_ARG_RETURN_ESCAPE ≠ 0
has_thrown_escape(x::UArgEscapeInfo) = x.escape_bits & EA_ARG_THROWN_ESCAPE ≠ 0
has_no_escape(x::UArgEscapeInfo) =
    !has_all_escape(x) && !has_return_escape(x) && !has_thrown_escape(x)

struct UArgAliasing
    aidx::Int
    bidx::Int
end

struct UArgEscapeCache
    argescapes::Vector{UArgEscapeInfo}
    argaliases::Vector{UArgAliasing}
end
function UArgEscapeCache(state::UEscapeState)
    nargs = state.nargs
    argescapes = Vector{UArgEscapeInfo}(undef, nargs)
    argaliases = UArgAliasing[]
    for i in 1:nargs
        argescapes[i] = UArgEscapeInfo(argescape(state, i))
        for j in (i+1):nargs
            if ea_in_same_set(state.aliasset, state.argids[i], state.argids[j])
                push!(argaliases, UArgAliasing(i, j))
            end
        end
    end
    return UArgEscapeCache(argescapes, argaliases)
end

# ---------------------------------------------------------------------------
# Changes and propagation (stock machinery, id-indexed)
# ---------------------------------------------------------------------------

abstract type EAChange end
struct EAEscapeChange <: EAChange
    xidx::Int
    xinfo::UEscapeInfo
end
struct EAAliasChange <: EAChange
    xidx::Int
    yidx::Int
end
struct EALivenessChange <: EAChange
    xidx::Int
    livepc::Int
end

mutable struct EAAnalysisState{GetEscapeCache}
    const ir::UnifiedIR.IR
    const estate::UEscapeState
    const changes::Vector{EAChange}
    const get_escape_cache::GetEscapeCache
    const caught::Vector{Bool}       # region id -> lies inside some try BODY
    const actroot::Vector{Int32}     # region id -> activation root region id
    const hastry::Bool
    # optional consumer hook (the optimizer integration): resolve a
    # statically-devirtualizable residual `call` to its MethodInstance so
    # the interprocedural summary machinery applies without rewriting the
    # IR (stock reaches the same states because its inliner has already
    # rewritten declined candidates to `:invoke`). `nothing` = no hook.
    const resolve_call::Any
end

function ea_propagate_changes!(estate::UEscapeState, changes::Vector{EAChange})
    local anychanged = false
    for change in changes
        if change isa EAEscapeChange
            anychanged |= ea_propagate_escape_change!(estate, change)
        elseif change isa EALivenessChange
            anychanged |= ea_propagate_liveness_change!(estate, change)
        else
            change = change::EAAliasChange
            xroot = ea_find_root!(estate.aliasset, change.xidx)
            yroot = ea_find_root!(estate.aliasset, change.yidx)
            if xroot ≠ yroot
                ea_union!(estate.aliasset, xroot, yroot)
                anychanged = true
            end
        end
    end
    return anychanged
end

function ea_propagate_escape_change!(estate::UEscapeState, change::EAEscapeChange)
    (; xidx, xinfo) = change
    anychanged = ea_apply_escape!(estate, xidx, xinfo)
    aliases = ea_aliases(estate.aliasset, xidx)
    if aliases !== nothing
        for aidx in aliases
            anychanged |= ea_apply_escape!(estate, aidx, xinfo)
        end
    end
    return anychanged
end

function ea_apply_escape!(estate::UEscapeState, xidx::Int, info::UEscapeInfo)
    old = estate.escapes[xidx]
    new = ea_join(old, info)
    if !ea_eq(old, new)
        estate.escapes[xidx] = new
        return true
    end
    return false
end

function ea_propagate_liveness_change!(estate::UEscapeState, change::EALivenessChange)
    (; xidx, livepc) = change
    info = estate.escapes[xidx]
    Liveness = info.Liveness
    Liveness === EA_TOP_LIVE && return false
    livepc ∈ Liveness && return false
    if Liveness === EA_BOT_LIVE || Liveness === EA_ARG_LIVE
        Liveness = copy(Liveness)
        push!(Liveness, livepc)
        estate.escapes[xidx] = UEscapeInfo(info; Liveness)
        return true
    else
        push!(Liveness, livepc)
        return true
    end
end

# ---------------------------------------------------------------------------
# Change constructors (operand-aware)
# ---------------------------------------------------------------------------

"StmtId of a STMT-tagged operand, else nothing (constants/immediates/sparams
are identity-free at the IR level; globals get the ⊤-alias rule)."
ea_opstmt(o::UnifiedIR.Operand) =
    UnifiedIR.optag(o) == UnifiedIR.TAG_STMT ? UnifiedIR.asstmt(o) : nothing

ea_lat(ir::UnifiedIR.IR, o::UnifiedIR.Operand) = stmt_lattice(ir, o)

function ea_tracked(ir::UnifiedIR.IR, s::StmtId)
    t = UnifiedIR.stmt_type(ir, s)
    t === nothing && return true
    return !CC.is_identity_free_argtype(t)
end

function add_escape_change!(astate::EAAnalysisState, @nospecialize(x), info::UEscapeInfo,
                            force::Bool = false)
    info === EA_⊥ && return nothing
    if x isa UnifiedIR.Operand
        s = ea_opstmt(x)
        s === nothing && return nothing
        x = s
    end
    x = x::StmtId
    if force || ea_tracked(astate.ir, x)
        push!(astate.changes, EAEscapeChange(Int(x.id), info))
    end
    return nothing
end

function add_liveness_change!(astate::EAAnalysisState, @nospecialize(x), livepc::StmtId)
    if x isa UnifiedIR.Operand
        s = ea_opstmt(x)
        s === nothing && return nothing
        x = s
    end
    x = x::StmtId
    if ea_tracked(astate.ir, x)
        push!(astate.changes, EALivenessChange(Int(x.id), Int(livepc.id)))
    end
    return nothing
end

"Alias `x` and `y` (operands or StmtIds). A global on either side imposes ⊤
on the other (stock's GlobalRef rule); untracked constants are ignored."
function add_alias_change!(astate::EAAnalysisState, @nospecialize(x), @nospecialize(y))
    xg = x isa UnifiedIR.Operand && UnifiedIR.optag(x) == UnifiedIR.TAG_GLOBAL
    yg = y isa UnifiedIR.Operand && UnifiedIR.optag(y) == UnifiedIR.TAG_GLOBAL
    xg && return add_escape_change!(astate, y, EA_⊤)
    yg && return add_escape_change!(astate, x, EA_⊤)
    xs = x isa UnifiedIR.Operand ? ea_opstmt(x) : x::StmtId
    ys = y isa UnifiedIR.Operand ? ea_opstmt(y) : y::StmtId
    (xs === nothing || ys === nothing) && return nothing
    estate = astate.estate
    xidx, yidx = Int(xs.id), Int(ys.id)
    if !ea_in_same_set(estate.aliasset, xidx, yidx)
        pushfirst!(astate.changes, EAAliasChange(xidx, yidx))
    end
    xinfo = estate.escapes[xidx]
    yinfo = estate.escapes[yidx]
    add_escape_change!(astate, xs, ea_join(xinfo, yinfo), #=force=#true)
    return nothing
end

function add_alias_escapes!(astate::EAAnalysisState, @nospecialize(v), ainfo::EAInfo)
    for x in ainfo
        x isa EALocalUse || continue
        add_alias_change!(astate, v, StmtId(Int32(x.idx)))
    end
end

function add_thrown_escapes!(astate::EAAnalysisState, pc::StmtId,
                             ops, first_idx::Int = 1, last_idx::Int = length(ops))
    info = EAThrownEscape(Int(pc.id))
    for i in first_idx:last_idx
        add_escape_change!(astate, ops[i], info)
    end
end

function add_liveness_changes!(astate::EAAnalysisState, pc::StmtId,
                               ops, first_idx::Int = 1, last_idx::Int = length(ops))
    for i in first_idx:last_idx
        add_liveness_change!(astate, ops[i], pc)
    end
end

function add_fallback_changes!(astate::EAAnalysisState, pc::StmtId,
                               ops, first_idx::Int = 1, last_idx::Int = length(ops))
    info = EAThrownEscape(Int(pc.id))
    for i in first_idx:last_idx
        add_escape_change!(astate, ops[i], info)
        add_liveness_change!(astate, ops[i], pc)
    end
end

function add_conservative_changes!(astate::EAAnalysisState, pc::StmtId,
                                   ops, first_idx::Int = 1, last_idx::Int = length(ops))
    for i in first_idx:last_idx
        add_escape_change!(astate, ops[i], EA_⊤)
    end
    if UnifiedIR.result_arity(UnifiedIR.stmt_kind(astate.ir, pc)) == 1
        add_escape_change!(astate, pc, EA_⊤)
    end
    return nothing
end

function escape_unanalyzable_obj!(astate::EAAnalysisState, @nospecialize(obj), objinfo::UEscapeInfo)
    objinfo = UEscapeInfo(objinfo, true)
    add_escape_change!(astate, obj, objinfo)
    return objinfo
end

ea_is_nothrow(ir::UnifiedIR.IR, s::StmtId) =
    UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_NOTHROW != 0

# ---------------------------------------------------------------------------
# The analysis driver
# ---------------------------------------------------------------------------

ea_no_cache(@nospecialize codeinst) = false

"""
    analyze_escapes(ir::UnifiedIR.IR, nargs::Int;
                    get_escape_cache = ea_no_cache,
                    resolve_call = nothing) -> UEscapeResult

Analyze escape information in typed unified IR. `nargs` is the number of
parameters (leading `region_arg`s of the root region, position 1 = the
function itself). `get_escape_cache(codeinst) ->
Union{Bool,UArgEscapeCache}` supplies interprocedural argument-escape
summaries for `invoke` sites (stock protocol: `true` = effect-free callee,
only ret-arg aliasing; `false` = unknown, conservative). `resolve_call(s) ->
Union{Nothing,MethodInstance}` optionally resolves a residual generic
`call` statement to a single (unambiguous, fully-covering) target so the
same summary machinery applies to it; on unified IR statically-resolved
declined-inline candidates are still `call`s at this point (stock's inliner
has rewritten them to `:invoke`), and without the hook they are analyzed
conservatively.
"""
function analyze_escapes(ir::UnifiedIR.IR, nargs::Int;
                         get_escape_cache = ea_no_cache,
                         resolve_call = nothing)
    UnifiedIR.check_state(ir, UnifiedIR.LAYOUT_DENSE, "analyze_escapes")
    n = UnifiedIR.nstmts(ir)
    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    nargs <= length(root.args) ||
        error("analyze_escapes: $nargs args for $(length(root.args)) root region_args")
    argids = Int[Int(root.args[i].id) for i in 1:nargs]

    escapes = UEscapeInfo[EA_⊥ for _ in 1:n]
    estate = UEscapeState(escapes, EAAliasSet(n), nargs, argids)
    for id in argids
        escapes[id] = EAArgEscape()
    end

    # region precomputation: inside-try-body mask, activation roots, and the
    # handler `%exc` args (conservative ⊤: the exception object may alias
    # anything thrown, including callee-internal values)
    nreg = UnifiedIR.nregions(ir)
    caught = Vector{Bool}(undef, nreg)
    actroot = Vector{Int32}(undef, nreg)
    hastry = false
    for ri in 1:nreg
        reg = ir.regions[ri]
        pc = reg.parent
        pcaught = UnifiedIR.isnull(pc) ? false : caught[pc.id]
        selfcaught = reg.kind === UnifiedIR.REGION_BODY && !UnifiedIR.isnull(reg.owner) &&
                     UnifiedIR.stmt_kind(ir, reg.owner) === K"try"
        caught[ri] = pcaught | selfcaught
        selfcaught && (hastry = true)
        if reg.activation !== UnifiedIR.ACT_IMMEDIATE || UnifiedIR.isnull(pc)
            actroot[ri] = Int32(ri)
        else
            actroot[ri] = actroot[pc.id]
        end
        if reg.kind === UnifiedIR.REGION_HANDLER && !isempty(reg.args) && !reg.dead
            escapes[reg.args[1].id] = EA_⊤
        end
    end

    astate = EAAnalysisState(ir, estate, EAChange[], get_escape_cache,
                             caught, actroot, hastry, resolve_call)

    while true
        local anyupdate = false
        for id in n:-1:1
            s = StmtId(Int32(id))
            k = UnifiedIR.stmt_kind(ir, s)
            k === UnifiedIR.KIND_DELETED && continue
            escape_stmt!(astate, s, k)
            isempty(astate.changes) && continue
            anyupdate |= ea_propagate_changes!(estate, astate.changes)
            empty!(astate.changes)
        end
        if hastry
            escape_exception!(astate)
            if !isempty(astate.changes)
                anyupdate |= ea_propagate_changes!(estate, astate.changes)
                empty!(astate.changes)
            end
        end
        anyupdate || break
    end

    return UEscapeResult(ir, estate)
end

"""
    escape_exception!(astate)

The stock blanket rule, structurally: any value whose `ThrownEscape` holds a
site lying (by region ancestry) inside some `try` op's body region — or the
top site `-1` when any `try` exists — gets ⊤. See stock `escape_exception!`
for why the imprecision is deliberate (`rethrow`, `current_exceptions`).
"""
function escape_exception!(astate::EAAnalysisState)
    estate = astate.estate
    ir = astate.ir
    for i in 1:length(estate.escapes)
        x = estate.escapes[i]
        xt = x.ThrownEscape
        caught = false
        if xt === EA_TOP_THROWN || -1 ∈ xt
            caught = true
        else
            for pc in xt
                pc >= 1 || continue
                if astate.caught[UnifiedIR.stmt_region(ir, StmtId(Int32(pc))).id]
                    caught = true
                    break
                end
            end
        end
        caught || continue
        add_escape_change!(astate, StmtId(Int32(i)), EA_⊤)
    end
end

# ---------------------------------------------------------------------------
# Per-statement transfer
# ---------------------------------------------------------------------------

function escape_stmt!(astate::EAAnalysisState, s::StmtId, k::UnifiedIR.Kind)
    ir = astate.ir
    # activation boundaries (§5.7): a value captured by a deferred region
    # (closure body) escapes conservatively — the body may run anywhere,
    # any number of times
    myact = astate.actroot[UnifiedIR.stmt_region(ir, s).id]
    if myact != Int32(1)
        for i in 1:UnifiedIR.nops(ir, s)
            d = ea_opstmt(UnifiedIR.getop(ir, s, i))
            d === nothing && continue
            if astate.actroot[UnifiedIR.stmt_region(ir, d).id] != myact
                add_escape_change!(astate, d, EA_⊤)
            end
        end
    end

    if k === K"call"
        escape_call!(astate, s)
    elseif k === K"invoke"
        escape_invoke!(astate, s)
    elseif k === K"intrinsic"
        ops = UnifiedIR.operands(ir, s)
        if ea_is_nothrow(ir, s)
            add_liveness_changes!(astate, s, ops, 2)
        else
            add_fallback_changes!(astate, s, ops, 2)
        end
    elseif k === K"new" || k === K"splatnew"
        escape_new!(astate, s)
    elseif k === K"extract"
        escape_extract!(astate, s)
    elseif k === K"refine"
        add_alias_change!(astate, s, UnifiedIR.getop(ir, s, 1))
    elseif k === K"select"
        escape_select!(astate, s)
    elseif k === K"globalref"
        add_escape_change!(astate, s, EA_⊤)
    elseif k === K"return"
        info = EAReturnEscape(Int(s.id))
        for i in 1:UnifiedIR.nops(ir, s)
            add_escape_change!(astate, UnifiedIR.getop(ir, s, i), info)
        end
    elseif k === K"result"
        reg = UnifiedIR.getregion(ir, UnifiedIR.stmt_region(ir, s))
        UnifiedIR.isnull(reg.owner) && return nothing
        UnifiedIR.stmt_kind(ir, reg.owner) === K"closure" && return nothing
        escape_region_feed!(astate, s, reg.owner,
                            UnifiedIR.operands(ir, s), 1)
    elseif k === K"break"
        tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
        owner = UnifiedIR.getregion(ir, tgt).owner
        UnifiedIR.isnull(owner) && return nothing
        escape_region_feed!(astate, s, owner, UnifiedIR.operands(ir, s), 2)
    elseif k === K"continue"
        tgt = UnifiedIR.asregion(UnifiedIR.getop(ir, s, 1))
        treg = UnifiedIR.getregion(ir, tgt)
        owner = treg.owner
        ops = UnifiedIR.operands(ir, s)
        # carried values feed the next iteration's region args...
        for i in 3:length(ops)
            ai = i - 2
            ai <= length(treg.args) || break
            add_alias_change!(astate, ops[i], treg.args[ai])
        end
        # ...and the loop's own results on fall-out
        UnifiedIR.isnull(owner) || escape_region_feed!(astate, s, owner, ops, 3)
    elseif k === K"loop"
        # init values feed the first iteration's region args
        rs = UnifiedIR.live_owned_regions(ir, s)
        if !isempty(rs)
            args = UnifiedIR.getregion(ir, rs[1]).args
            for i in 1:UnifiedIR.nops(ir, s)
                o = UnifiedIR.getop(ir, s, i)
                i <= length(args) && add_alias_change!(astate, o, args[i])
                add_liveness_change!(astate, o, s)
            end
        end
    elseif k === K"if" || k === K"try"
        # results flow at the terminators; operands (condition / optional
        # dynscope) just stay live
        add_liveness_changes!(astate, s, UnifiedIR.operands(ir, s), 1)
    elseif k === K"cfg"
        # cfg operands feed the entry block's args
        rs = UnifiedIR.live_owned_regions(ir, s)
        if !isempty(rs)
            args = UnifiedIR.getregion(ir, rs[1]).args
            for i in 1:UnifiedIR.nops(ir, s)
                o = UnifiedIR.getop(ir, s, i)
                i <= length(args) && add_alias_change!(astate, o, args[i])
                add_liveness_change!(astate, o, s)
            end
        end
    elseif k === K"goto" || k === K"br_if" || k === K"switch"
        for (dest, argops) in UnifiedIR.edge_bundles(ir, s)
            dargs = UnifiedIR.getregion(ir, dest).args
            for i in 1:length(argops)
                i <= length(dargs) || break
                add_alias_change!(astate, argops[i], dargs[i])
                add_liveness_change!(astate, argops[i], s)
            end
        end
    elseif k === K"cell" || k === K"cell_shared"
        # the cell token: a 1-field object; `cell_shared` may be captured by
        # reference (closures) — conservative
        if k === K"cell_shared"
            objinfo = astate.estate[s]
            escape_unanalyzable_obj!(astate, s, objinfo)
        end
    elseif k === K"cell_get"
        escape_cell_get!(astate, s)
    elseif k === K"cell_set"
        escape_cell_set!(astate, s)
    elseif k === K"foreigncall"
        escape_foreigncall!(astate, s)
    elseif k === K"gc_preserve_begin"
        add_liveness_changes!(astate, s, UnifiedIR.operands(ir, s), 1)
    elseif k === K"gc_preserve_end"
        tok = ea_opstmt(UnifiedIR.getop(ir, s, 1))
        if tok !== nothing && UnifiedIR.stmt_kind(ir, tok) === K"gc_preserve_begin"
            add_liveness_changes!(astate, s, UnifiedIR.operands(ir, tok), 1)
        end
    elseif k === K"throw_undef_if_not"
        add_escape_change!(astate, UnifiedIR.getop(ir, s, 1), EAThrownEscape(Int(s.id)))
    elseif k === K"closure"
        # the closure value: unanalyzable fields (captures are handled by the
        # activation-boundary rule above)
        escape_unanalyzable_obj!(astate, s, astate.estate[s])
    elseif k === K"region_arg" || k === K"value" || k === K"boundscheck" ||
           k === K"latestworld" || k === K"coverage_effect" || k === K"copyast" ||
           k === K"cell_new" || k === K"cell_isdefined" || k === K"isdefined_global" ||
           k === K"unreachable" || k === K"deleted"
        return nothing
    else
        # new_opaque_closure, cfunction, await, method_def, toplevel forms,
        # unknown extensions: escape everything conservatively
        add_conservative_changes!(astate, s, UnifiedIR.operands(ir, s))
    end
    return nothing
end

"""
    escape_region_feed!(astate, t, owner, ops, first_idx)

An exit terminator `t` feeds `owner`'s results with `ops[first_idx:end]`.
Arity 1: direct aliasing (the φ case). Arity n: the owner acts as a
tuple-shaped object whose field `i` is defined by value `i` at `t` — the
same field machinery `extract` loads through (stock's phi-of-tuples).
"""
function escape_region_feed!(astate::EAAnalysisState, t::StmtId, owner::StmtId,
                             ops, first_idx::Int)
    nvals = length(ops) - first_idx + 1
    nvals <= 0 && return nothing
    if nvals == 1
        add_alias_change!(astate, ops[first_idx], owner)
        return nothing
    end
    # terminators never throw: the fed values take no ThrownEscape here
    escape_object_def!(astate, owner, Int(t.id), ops, first_idx, #=nothrow=#true)
    return nothing
end

# ---------------------------------------------------------------------------
# Object definitions (new/splatnew/tuple/region-fed results)
# ---------------------------------------------------------------------------

"Field-count of the object `obj` defines (nothing = unindexable)."
function ea_object_nflds(astate::EAAnalysisState, obj::StmtId)
    t = UnifiedIR.stmt_type(astate.ir, obj)
    t === nothing && return nothing
    return CC.fieldcount_noerror(CC.widenconst(t))
end

"""
    escape_object_def!(astate, obj, defsite, ops, first_idx, nothrow)

Stock `escape_new!` generalized: `ops[first_idx:end]` define the fields of
`obj` at `defsite`.
"""
function escape_object_def!(astate::EAAnalysisState, obj::StmtId, defsite::Int,
                            ops, first_idx::Int, nothrow::Bool;
                            force_unindexable::Bool = false)
    objinfo = astate.estate[obj]
    AliasInfo = objinfo.AliasInfo
    nargs = length(ops)
    if AliasInfo isa Bool
        if AliasInfo
            @goto conservative_propagation
        end
        nflds = force_unindexable ? nothing : ea_object_nflds(astate, obj)
        if nflds === nothing
            AliasInfo = EAUnindexable()
            @goto escape_unindexable_def
        else
            AliasInfo = EAIndexableFields(nflds)
            @goto escape_indexable_def
        end
    elseif AliasInfo isa EAIndexableFields
        AliasInfo = copy(AliasInfo)
        @label escape_indexable_def
        infos = AliasInfo.infos
        nf = length(infos)
        objinfo′ = ignore_aliasinfo(objinfo)
        for i in first_idx:nargs
            fidx = i - first_idx + 1
            fidx > nf && break
            arg = ops[i]
            add_alias_escapes!(astate, arg, infos[fidx])
            push!(infos[fidx], EALocalDef(defsite))
            add_escape_change!(astate, arg, objinfo′)
            add_liveness_change!(astate, arg, StmtId(Int32(defsite)))
        end
        add_escape_change!(astate, obj, UEscapeInfo(objinfo, AliasInfo))
    elseif AliasInfo isa EAUnindexable
        AliasInfo = copy(AliasInfo)
        @label escape_unindexable_def
        info = AliasInfo.info
        objinfo′ = ignore_aliasinfo(objinfo)
        for i in first_idx:nargs
            arg = ops[i]
            add_alias_escapes!(astate, arg, info)
            push!(info, EALocalDef(defsite))
            add_escape_change!(astate, arg, objinfo′)
            add_liveness_change!(astate, arg, StmtId(Int32(defsite)))
        end
        add_escape_change!(astate, obj, UEscapeInfo(objinfo, AliasInfo))
    else
        objinfo = escape_unanalyzable_obj!(astate, obj, objinfo)
        @label conservative_propagation
        for i in first_idx:nargs
            arg = ops[i]
            add_escape_change!(astate, arg, objinfo)
            add_liveness_change!(astate, arg, StmtId(Int32(defsite)))
        end
    end
    if !nothrow
        add_thrown_escapes!(astate, StmtId(Int32(defsite)), ops)
    end
    return nothing
end

function escape_new!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    ops = UnifiedIR.operands(ir, s)
    # splatnew's field values are the elements of the splatted tuple, not the
    # operand itself: couple through an unindexable set (conservative, sound)
    force_unindexable = UnifiedIR.stmt_kind(ir, s) === K"splatnew"
    escape_object_def!(astate, s, Int(s.id), ops, 2, ea_is_nothrow(ir, s);
                       force_unindexable)
    # the type operand (over-approximated as a thrown-escape participant when
    # the construction may throw — matches stock, which passes all args)
    if !ea_is_nothrow(ir, s)
        add_escape_change!(astate, ops[1], EAThrownEscape(Int(s.id)))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Field loads (extract / getfield / cell_get)
# ---------------------------------------------------------------------------

"Field index of `fldval` in `typ`; union-aware (the region-result tuple of an
`if` joining differently-typed arms is Union-typed — every component must
agree on the index; stock's flat φs never carry that shape)."
function ea_fieldidx(@nospecialize(typ), @nospecialize(fldval))
    fldval === nothing && return nothing
    if typ isa DataType
        return CC.try_compute_fieldidx(typ, fldval)
    elseif typ isa Union
        ia = ea_fieldidx(typ.a, fldval)
        ia === nothing && return nothing
        ib = ea_fieldidx(typ.b, fldval)
        return ia === ib ? ia : nothing
    end
    return nothing
end

function ea_analyze_fields(astate::EAAnalysisState, obj::StmtId, @nospecialize(fldval))
    nflds = ea_object_nflds(astate, obj)
    nflds === nothing && return EAUnindexable(), 0
    typ = CC.widenconst(UnifiedIR.stmt_type(astate.ir, obj))
    fidx = ea_fieldidx(typ, fldval)
    fidx === nothing && return EAUnindexable(), 0
    return EAIndexableFields(nflds), fidx
end

function ea_reanalyze_fields(astate::EAAnalysisState, AliasInfo::EAIndexableFields,
                             obj::StmtId, @nospecialize(fldval))
    nflds = ea_object_nflds(astate, obj)
    nflds === nothing && return ea_merge_to_unindexable(AliasInfo), 0
    typ = CC.widenconst(UnifiedIR.stmt_type(astate.ir, obj))
    fidx = ea_fieldidx(typ, fldval)
    fidx === nothing && return ea_merge_to_unindexable(AliasInfo), 0
    AliasInfo = copy(AliasInfo)
    infos = AliasInfo.infos
    for _ in 1:(nflds - length(infos))
        push!(infos, EAInfo())
    end
    return AliasInfo, fidx
end

"getfield-shaped load: `s` loads field `fldval` (nothing = unknown) of `objop`."
function escape_field_load!(astate::EAAnalysisState, s::StmtId,
                            objop::UnifiedIR.Operand, @nospecialize(fldval))
    ir, estate = astate.ir, astate.estate
    typ = CC.widenconst(ea_lat(ir, objop))
    if CC.hasintersect(typ, Module) # global load
        add_escape_change!(astate, s, EA_⊤)
    end
    obj = ea_opstmt(objop)
    if obj === nothing
        add_escape_change!(astate, s, EA_⊤)
        return false
    end
    objinfo = estate[obj]
    AliasInfo = objinfo.AliasInfo
    if AliasInfo isa Bool
        AliasInfo && @goto conservative_propagation
        AliasInfo, fidx = ea_analyze_fields(astate, obj, fldval)
        if AliasInfo isa EAIndexableFields
            @goto record_indexable_use
        else
            @goto record_unindexable_use
        end
    elseif AliasInfo isa EAIndexableFields
        AliasInfo, fidx = ea_reanalyze_fields(astate, AliasInfo, obj, fldval)
        AliasInfo isa EAUnindexable && @goto record_unindexable_use
        @label record_indexable_use
        push!(AliasInfo.infos[fidx], EALocalUse(Int(s.id)))
        add_escape_change!(astate, obj, UEscapeInfo(objinfo, AliasInfo))
    elseif AliasInfo isa EAUnindexable
        AliasInfo = copy(AliasInfo)
        @label record_unindexable_use
        push!(AliasInfo.info, EALocalUse(Int(s.id)))
        add_escape_change!(astate, obj, UEscapeInfo(objinfo, AliasInfo))
    else
        objinfo = escape_unanalyzable_obj!(astate, obj, objinfo)
        @label conservative_propagation
        add_alias_change!(astate, obj, s)
    end
    return false
end

function escape_extract!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    objop = UnifiedIR.getop(ir, s, 1)
    idx = UnifiedIR.imm_value(UnifiedIR.getop(ir, s, 2))
    escape_field_load!(astate, s, objop, idx isa Int64 ? Int(idx) : idx)
    ops = UnifiedIR.operands(ir, s)
    if ea_is_nothrow(ir, s)
        add_liveness_changes!(astate, s, ops, 1)
    else
        add_fallback_changes!(astate, s, ops, 1)
    end
    return nothing
end

function escape_cell_get!(astate::EAAnalysisState, s::StmtId)
    ir, estate = astate.ir, astate.estate
    cell = ea_opstmt(UnifiedIR.getop(ir, s, 1))
    cell === nothing && return nothing
    objinfo = estate[cell]
    AliasInfo = objinfo.AliasInfo
    if AliasInfo isa Bool
        if AliasInfo
            add_alias_change!(astate, cell, s)
            return nothing
        end
        AliasInfo = EAIndexableFields(1)
    elseif AliasInfo isa EAIndexableFields
        AliasInfo = copy(AliasInfo)
    else
        AliasInfo = copy(AliasInfo::EAUnindexable)
        push!(AliasInfo.info, EALocalUse(Int(s.id)))
        add_escape_change!(astate, cell, UEscapeInfo(objinfo, AliasInfo))
        return nothing
    end
    push!(AliasInfo.infos[1], EALocalUse(Int(s.id)))
    add_escape_change!(astate, cell, UEscapeInfo(objinfo, AliasInfo))
    return nothing
end

function escape_cell_set!(astate::EAAnalysisState, s::StmtId)
    ir, estate = astate.ir, astate.estate
    cell = ea_opstmt(UnifiedIR.getop(ir, s, 1))
    val = UnifiedIR.getop(ir, s, 2)
    if cell === nothing
        add_escape_change!(astate, val, EA_⊤, #=force=#true)
        return nothing
    end
    objinfo = estate[cell]
    AliasInfo = objinfo.AliasInfo
    if AliasInfo isa Bool
        if AliasInfo
            add_alias_change!(astate, val, cell)
            return nothing
        end
        AliasInfo = EAIndexableFields(1)
    elseif AliasInfo isa EAIndexableFields
        AliasInfo = copy(AliasInfo)
    else
        AliasInfo = copy(AliasInfo::EAUnindexable)
        add_alias_escapes!(astate, val, AliasInfo.info)
        push!(AliasInfo.info, EALocalDef(Int(s.id)))
        objinfo = UEscapeInfo(objinfo, AliasInfo)
        add_escape_change!(astate, cell, objinfo)
        add_escape_change!(astate, val, ignore_aliasinfo(objinfo))
        return nothing
    end
    add_alias_escapes!(astate, val, AliasInfo.infos[1])
    push!(AliasInfo.infos[1], EALocalDef(Int(s.id)))
    objinfo = UEscapeInfo(objinfo, AliasInfo)
    add_escape_change!(astate, cell, objinfo)
    add_escape_change!(astate, val, ignore_aliasinfo(objinfo))
    return nothing
end

# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

function escape_call!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    ops = UnifiedIR.operands(ir, s)
    f = static_operand_value(ir, ops[1])
    if f isa Core.IntrinsicFunction
        if ea_is_nothrow(ir, s)
            add_liveness_changes!(astate, s, ops, 2)
        else
            add_fallback_changes!(astate, s, ops, 2)
        end
        return nothing
    end
    if f isa Core.Builtin
        result = escape_builtin!(astate, s, f, ops)
        if result === missing
            add_conservative_changes!(astate, s, ops)
        elseif result === true
            add_liveness_changes!(astate, s, ops, 2)
        elseif ea_is_nothrow(ir, s)
            add_liveness_changes!(astate, s, ops, 2)
        else
            add_fallback_changes!(astate, s, ops, 2)
        end
        return nothing
    end
    # statically-resolvable residual call (consumer hook): interprocedural
    # treatment — callee param i is ops[i] (ops[1] = the function itself)
    if astate.resolve_call !== nothing
        mi = astate.resolve_call(s)
        if mi isa Core.MethodInstance
            return escape_invoke_target!(astate, s, ops, mi, 1)
        end
    end
    # unknown or dynamic callee: conservative
    add_conservative_changes!(astate, s, ops)
    return nothing
end

function escape_builtin!(astate::EAAnalysisState, s::StmtId, @nospecialize(f),
                         ops::Vector{UnifiedIR.Operand})
    ir = astate.ir
    # safe builtins: no escape beyond liveness/throw fallback
    if f === Core.isa || f === Core.typeof || f === Core.sizeof || f === Core.:(===) ||
       f === Core.donotdelete || f === Core.isdefined || f === Core.throw ||
       f === Core.throw_methoderror
        return false
    end
    if f === Core.ifelse && length(ops) == 4
        condl = ea_lat(ir, ops[2])
        if condl isa CC.Const && condl.val isa Bool
            add_alias_change!(astate, condl.val ? ops[3] : ops[4], s)
        else
            add_alias_change!(astate, ops[3], s)
            add_alias_change!(astate, ops[4], s)
        end
        return false
    end
    if f === Core.typeassert && length(ops) == 3
        add_alias_change!(astate, s, ops[2])
        return false
    end
    if f === Core.tuple
        escape_object_def!(astate, s, Int(s.id), ops, 2, ea_is_nothrow(ir, s))
        return false
    end
    if f === Core.getfield && length(ops) ≥ 3
        return escape_field_load!(astate, s, ops[2], static_operand_value(ir, ops[3]))
    end
    if f === Core.setfield! && length(ops) ≥ 4
        return escape_setfield!(astate, s, ops)
    end
    if f === Core.finalizer && length(ops) ≥ 3
        add_liveness_change!(astate, ops[3], s)
        return false
    end
    return missing
end

function escape_setfield!(astate::EAAnalysisState, s::StmtId,
                          ops::Vector{UnifiedIR.Operand})
    ir, estate = astate.ir, astate.estate
    objop, fldop, valop = ops[2], ops[3], ops[4]
    val = valop
    obj = ea_opstmt(objop)
    if obj === nothing
        # unanalyzable object (a direct global operand etc.): escape the
        # value. Forced: stock reaches this state through an alias join with
        # the ⊤ global load, which bypasses the identity-free gate
        add_escape_change!(astate, val, EA_⊤, #=force=#true)
        @goto add_thrown_escapes
    end
    objinfo = estate[obj]
    AliasInfo = objinfo.AliasInfo
    if AliasInfo isa Bool
        AliasInfo && @goto conservative_propagation
        AliasInfo, fidx = ea_analyze_fields(astate, obj, static_operand_value(ir, fldop))
        if AliasInfo isa EAIndexableFields
            @goto escape_indexable_def
        else
            @goto escape_unindexable_def
        end
    elseif AliasInfo isa EAIndexableFields
        AliasInfo, fidx = ea_reanalyze_fields(astate, AliasInfo, obj,
                                              static_operand_value(ir, fldop))
        AliasInfo isa EAUnindexable && @goto escape_unindexable_def
        @label escape_indexable_def
        add_alias_escapes!(astate, val, AliasInfo.infos[fidx])
        push!(AliasInfo.infos[fidx], EALocalDef(Int(s.id)))
        objinfo = UEscapeInfo(objinfo, AliasInfo)
        add_escape_change!(astate, obj, objinfo)
        add_escape_change!(astate, val, ignore_aliasinfo(objinfo))
    elseif AliasInfo isa EAUnindexable
        AliasInfo = copy(AliasInfo)
        @label escape_unindexable_def
        add_alias_escapes!(astate, val, AliasInfo.info)
        push!(AliasInfo.info, EALocalDef(Int(s.id)))
        objinfo = UEscapeInfo(objinfo, AliasInfo)
        add_escape_change!(astate, obj, objinfo)
        add_escape_change!(astate, val, ignore_aliasinfo(objinfo))
    else
        objinfo = escape_unanalyzable_obj!(astate, obj, objinfo)
        @label conservative_propagation
        add_alias_change!(astate, val, obj)
    end
    # escape information imposed on the return value of this setfield!
    ssainfo = estate[s]
    add_escape_change!(astate, val, ssainfo)
    @label add_thrown_escapes
    if length(ops) == 4 && CC.setfield!_nothrow(CC.fallback_lattice,
        ea_lat(ir, ops[2]), ea_lat(ir, ops[3]), ea_lat(ir, ops[4]))
        return true
    else
        add_thrown_escapes!(astate, s, ops, 2)
        return true
    end
end

function escape_select!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    condl = ea_lat(ir, UnifiedIR.getop(ir, s, 1))
    if condl isa CC.Const && condl.val isa Bool
        add_alias_change!(astate, UnifiedIR.getop(ir, s, condl.val ? 2 : 3), s)
    else
        add_alias_change!(astate, UnifiedIR.getop(ir, s, 2), s)
        add_alias_change!(astate, UnifiedIR.getop(ir, s, 3), s)
    end
    add_liveness_changes!(astate, s, UnifiedIR.operands(ir, s), 1)
    return nothing
end

# ---------------------------------------------------------------------------
# invoke (interprocedural)
# ---------------------------------------------------------------------------

function escape_invoke!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    ops = UnifiedIR.operands(ir, s)
    codeinst = static_operand_value(ir, ops[1])
    return escape_invoke_target!(astate, s, ops, codeinst, 2)
end

"Interprocedural site treatment shared by `invoke` statements (`first_idx =
2`: ops[1] is the CodeInstance/MethodInstance, callee param i = ops[i+1])
and hook-resolved residual `call`s (`first_idx = 1`: callee param i =
ops[i], the function itself included)."
function escape_invoke_target!(astate::EAAnalysisState, s::StmtId,
                               ops::Vector{UnifiedIR.Operand},
                               @nospecialize(codeinst), first_idx::Int)
    last_idx = length(ops)
    add_liveness_changes!(astate, s, ops, first_idx, last_idx)
    mi = codeinst isa Core.CodeInstance ? codeinst.def :
         codeinst isa Core.MethodInstance ? codeinst : nothing
    cache = mi === nothing ? false : astate.get_escape_cache(codeinst)
    if cache isa Bool
        if cache
            # effect-free callee: nothing escapes, but arguments may be
            # returned — account for ret-arg aliasing
            for i = first_idx:last_idx
                arg = ops[i]
                UnifiedIR.optag(arg) == UnifiedIR.TAG_GLOBAL && continue
                add_alias_change!(astate, s, arg)
            end
            return nothing
        else
            return add_conservative_changes!(astate, s, ops, first_idx)
        end
    end
    cache = cache::UArgEscapeCache
    retinfo = astate.estate[s]
    method = (mi::Core.MethodInstance).def::Method
    nargs = Int(method.nargs)
    for (i, argidx) in enumerate(first_idx:last_idx)
        arg = ops[argidx]
        if i > nargs
            i = nargs # isva
        end
        i <= length(cache.argescapes) || break
        argescape = cache.argescapes[i]
        info = ea_from_interprocedural(argescape, s)
        add_escape_change!(astate, arg, info)
        if has_return_escape(argescape)
            add_alias_change!(astate, s, arg)
        end
    end
    for (; aidx, bidx) in cache.argaliases
        (aidx + first_idx - 1 <= last_idx && bidx + first_idx - 1 <= last_idx) || continue
        add_alias_change!(astate, ops[aidx+(first_idx-1)], ops[bidx+(first_idx-1)])
    end
    # disable alias analysis on the newly introduced (callee-returned) object
    add_escape_change!(astate, s, UEscapeInfo(retinfo, true))
    return nothing
end

function ea_from_interprocedural(argescape::UArgEscapeInfo, pc::StmtId)
    has_all_escape(argescape) && return EA_⊤
    ThrownEscape = has_thrown_escape(argescape) ? Base.BitSet(Int(pc.id)) : EA_BOT_THROWN
    return UEscapeInfo(#=Analyzed=#true, #=ReturnEscape=#false, ThrownEscape,
                       #=AliasInfo=#true, #=Liveness=#Base.BitSet(Int(pc.id)))
end

# ---------------------------------------------------------------------------
# foreigncall (operand layout mirrors Expr(:foreigncall); codeinfo_entry.jl)
# ---------------------------------------------------------------------------

function escape_foreigncall!(astate::EAAnalysisState, s::StmtId)
    ir = astate.ir
    ops = UnifiedIR.operands(ir, s)
    nops = length(ops)
    if nops >= 1 && static_operand_value(ir, ops[1]) === FOREIGNGLOBAL_MARKER
        return nothing # cglobal lowering: name only, nothing escapes
    end
    if nops < 6
        add_conservative_changes!(astate, s, ops)
        return nothing
    end
    argtypes = static_operand_value(ir, ops[3])
    if !(argtypes isa Core.SimpleVector)
        add_conservative_changes!(astate, s, ops)
        return nothing
    end
    nccallargs = length(argtypes)
    nothrow = ea_is_nothrow(ir, s)
    # the callee name (op 1) is constant in this encoding; args are 6..5+n
    for i = 1:nccallargs
        5 + i <= nops || break
        arg = ops[5+i]
        if argtypes[i] === Any
            add_escape_change!(astate, arg, EA_⊤)
        elseif !nothrow
            add_escape_change!(astate, arg, EAThrownEscape(Int(s.id)))
        end
        add_liveness_change!(astate, arg, s)
    end
    for i = (5+nccallargs+1):nops
        add_liveness_change!(astate, ops[i], s)
    end
    return nothing
end
