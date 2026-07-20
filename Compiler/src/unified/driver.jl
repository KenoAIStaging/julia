# The real typeinf driver (COMPILER-PORT-PLAN A1): `unified_typeinf` runs the
# native unified pipeline — entry-convert → infer_ir! → optimize_ir! →
# ir_to_ircode → CodeInfo — behind the Compiler module's standard entry
# points, producing cache-grade CodeInstances with stock-encoded edges and
# sound world bounds. Every body the pipeline cannot (yet) handle falls back
# to the stock compiler, per body, with a counted reason (`pipeline_stats()`
# is the ratchet). Installed via `enable_pipeline!` (Compiler.UNIFIED_HOOKS);
# `activate!` additionally flips the runtime's jl_typeinf_func.
#
# Concurrency and reentrancy (A5): every request runs with a FRESH
# `UInferState`/`UEdges` pair, so there is no shared inference state between
# passes — each collector sees exactly the facts its own pass consumed. The
# per-mi serialization is the engine's (`engine_reserve`, whose C side
# resolves same-thread and cross-thread reservation cycles without
# deadlocking; a same-thread re-reservation returns a non-owning placeholder
# CodeInstance and `jl_engine_fulfill` ignores non-reservations). What the
# driver adds is a small per-TASK discipline:
#   - an `inflight` set declines requests for a MethodInstance this task is
#     already driving (`:reentrant_self`) — recursing on the same body can
#     only redo the same work against the engine placeholder;
#   - a depth counter bounds nested driver passes (`:reentrant_depth`).
#     Reentrant requests below the bound — the runtime compiling something
#     the driver's own execution needs, and the devirtualizer's callee
#     CodeInstance production — run the unified pipeline recursively; at the
#     bound they decline precisely and stock compiles the body (cached, so
#     each such body is compiled at most once per session). The runtime
#     itself additionally caps `jl_typeinf_func` reentrancy per task (gf.c
#     reentrant_timing), so runtime-initiated recursion is shallow by
#     construction; the driver bound mainly governs its own recursion.
#
# Soundness protocol (mirrors stock finish!/finish_nocycle):
#   - the collector starts at WorldRange(1, world_counter) and intersects the
#     validity window of every consulted fact (see UEdges in uinference.jl);
#   - at finish, if the intersection no longer reaches the CURRENT counter,
#     the result is NOT cached (reason :world_moved/:world_bounded) — unlike
#     stock we do not publish bounded CodeInstances, because the unified
#     OPTIMIZER still reads ambient global state (isconst/getglobal in
#     static_operand_value & co.), which is only provably world-consistent
#     when nothing moved during the pass. Lazy binding/partition
#     materialization bumps the counter once per binding per process, so the
#     driver retries once before giving up;
#   - otherwise: store_backedges (stock encoding via build_edges) →
#     jl_fill_codeinst → cache insert → engine fulfill → codegen-cache
#     insert → jl_promote_ci_to_current, exactly stock's sequence.

# ---------------------------------------------------------------------------
# Fallback ledger (the ratchet)
# ---------------------------------------------------------------------------

mutable struct PipelineLedger
    unified::Int                  # bodies fully through the unified pipeline
    fallbacks::Dict{Symbol,Int}   # reason -> count (stock handled the body)
    last_error::Any               # (reason, mi, exception) of the last error-class fallback
end
const PIPELINE_STATS = PipelineLedger(0, Dict{Symbol,Int}(), nothing)
# Ledger locking discipline. Once the driver IS the runtime's compiler, any
# call boundary inside a locked region can demand a first-time compile, and
# that compile's own serve/decline paths write this ledger. A plain
# non-reentrant lock therefore self-deadlocks: the nested writer spins
# forever on the lock its own parked outer frame holds (the demo's wedge —
# `reset_pipeline_stats!` executing `empty!` inside the locked region raised
# the compile of an inner target, whose pipeline pass hit a `count_fallback!`
# and spun on STATS_LOCK for the rest of the session). Rules:
#   - SAME-TASK reentry runs the update WITHOUT re-acquiring: the outer
#     frame is parked at a call boundary (its structures are between
#     mutations) and it already holds the lock, so cross-thread exclusion
#     still stands while the nested update runs;
#   - compile-path writers (`count_fallback!`/`note_unified!`) never spin
#     unboundedly on CROSS-task contention either — a holder can be parked
#     mid-compile for seconds, and an unbounded spin can deadlock against
#     an engine-reservation cycle. After a bounded spin the write is
#     dropped and counted (`STATS_DROPPED`, surfaced as `ledger_dropped`);
#   - user-context entries (`pipeline_stats`/`reset_pipeline_stats!`) block
#     normally (with the same same-task reentry escape).
const STATS_LOCK = Base.Threads.SpinLock()
const STATS_OWNER = Base.RefValue{Any}(nothing)     # task currently holding STATS_LOCK
const STATS_DROPPED = Base.Threads.Atomic{Int}(0)   # contended-away ledger writes

# acquire states: 0x0 locked here (must unlock), 0x1 same-task reentry
# (already held up-stack: proceed unlocked), 0x2 contended away (drop).
# Shared by every driver-global structure touched from compile paths (the
# ledger, the cross-request memo): the wave-5 P1 design rule.
function _guarded_acquire(lck::Base.Threads.SpinLock, owner::Base.RefValue{Any}, bounded::Bool)
    ct = ccall(:jl_get_current_task, Any, ())
    owner[] === ct && return 0x1
    if bounded
        spins = 0
        while !Base.trylock(lck)
            spins += 1
            spins >= 1_000_000 && return 0x2
            ccall(:jl_cpu_suspend, Cvoid, ())
            ccall(:jl_gc_safepoint, Cvoid, ())
        end
    else
        Base.lock(lck)
    end
    owner[] = ct
    return 0x0
end
function _guarded_release(lck::Base.Threads.SpinLock, owner::Base.RefValue{Any}, state::UInt8)
    if state == 0x0
        owner[] = nothing
        Base.unlock(lck)
    end
    return nothing
end
_stats_acquire(bounded::Bool) = _guarded_acquire(STATS_LOCK, STATS_OWNER, bounded)
_stats_release(state::UInt8) = _guarded_release(STATS_LOCK, STATS_OWNER, state)

function count_fallback!(reason::Symbol, @nospecialize(mi = nothing), @nospecialize(err = nothing))
    st = _stats_acquire(true)
    if st == 0x2
        Base.Threads.atomic_add!(STATS_DROPPED, 1)
        return nothing
    end
    try
        d = PIPELINE_STATS.fallbacks
        d[reason] = get(d, reason, 0) + 1
        err === nothing || (PIPELINE_STATS.last_error = (reason, mi, err))
    finally
        _stats_release(st)
    end
    return nothing
end

function note_unified!()
    st = _stats_acquire(true)
    if st == 0x2
        Base.Threads.atomic_add!(STATS_DROPPED, 1)
        return nothing
    end
    try
        PIPELINE_STATS.unified += 1
    finally
        _stats_release(st)
    end
    return nothing
end

"""
    pipeline_stats() -> NamedTuple

The driver's ledger: `unified` counts bodies compiled end-to-end by the
unified pipeline, `fallbacks` maps fallback reason to count (those bodies
were handled by the stock compiler), `last_error` retains the most recent
`(reason, mi, exception)` for error-class fallbacks, and `memo` reports the
cross-request memo's honesty counters — `hits` (frames served by replaying
a stored entry's facts), `misses` (no entry), `stale` (entries dropped at
world revalidation), `stores` (entries recorded), `replayed` (facts
replayed into collectors), `dropped` (memo operations contended away), and
`entries` (current table size).
"""
function pipeline_stats()
    st = _stats_acquire(false)
    try
        fallbacks = copy(PIPELINE_STATS.fallbacks)
        # keep the ledger honest: writes contended away by a parked holder
        # (see the locking discipline above) surface as their own reason
        STATS_DROPPED[] == 0 || (fallbacks[:ledger_dropped] = STATS_DROPPED[])
        return (; unified = PIPELINE_STATS.unified,
                  fallbacks,
                  last_error = PIPELINE_STATS.last_error,
                  memo = (; hits = MEMO_HITS[], misses = MEMO_MISSES[],
                            stale = MEMO_STALE[], stores = MEMO_STORES[],
                            replayed = MEMO_REPLAYED[], dropped = MEMO_DROPPED[],
                            entries = length(DRIVER_MEMO)))
    finally
        _stats_release(st)
    end
end

function reset_pipeline_stats!()
    st = _stats_acquire(false)
    try
        PIPELINE_STATS.unified = 0
        empty!(PIPELINE_STATS.fallbacks)
        PIPELINE_STATS.last_error = nothing
        STATS_DROPPED[] = 0
        MEMO_HITS[] = 0
        MEMO_MISSES[] = 0
        MEMO_STALE[] = 0
        MEMO_STORES[] = 0
        MEMO_REPLAYED[] = 0
        MEMO_DROPPED[] = 0
    finally
        _stats_release(st)
    end
    return nothing
end

"One-line ledger print (the demo/bench surface)."
function print_pipeline_stats(io::IO = Base.stdout)
    stats = pipeline_stats()
    total = stats.unified + sum(values(stats.fallbacks); init = 0)
    println(io, "unified pipeline: ", stats.unified, "/", total, " bodies")
    for (reason, n) in sort!(collect(stats.fallbacks); by = last, rev = true)
        println(io, "  fallback ", rpad(String(reason), 22), " ", n)
    end
    m = stats.memo
    println(io, "  memo hits ", m.hits, " misses ", m.misses, " stale ", m.stale,
            " stores ", m.stores, " entries ", m.entries)
    return nothing
end

# ---------------------------------------------------------------------------
# Cross-request memoization with per-result edge replay (A6)
# ---------------------------------------------------------------------------
#
# Every driver request runs a fresh UInferState/UEdges pair, so before this
# section each request re-inferred its whole callee tree from scratch (the
# single-collector soundness basis). The global memo removes the
# re-inference WITHOUT weakening that basis: a clean callee frame's result
# is stored TOGETHER WITH the fact set it consumed (its window of the
# request's fact trace, `UEdges.trace`, flattened across span references),
# and a consuming request REPLAYS those facts into its own collector — the
# entry CodeInstance's edges and world bounds come out exactly as if the
# callee had been re-inferred, minus the walk.
#
# Keying: per-MethodInstance for widened frames, per-const-key (`UConstKey`:
# precomputed egal-consistent hash, `===`-swept parts — one concrete key
# type, so the table never compiles per-shape hash/lookup specializations)
# for const-seeded frames and concrete evaluations. The table is a plain
# `Dict`: `UConstKey`s are freshly built per request, so equality (not
# identity) must key the hits; `hash(::UConstKey)` reads a field.
#
# World validation (invalidation correctness): an entry stores the world its
# facts were last validated at plus the world COUNTER observed at that
# frame's start. A lookup at the same (world, counter) replays the stored
# fact objects directly — nothing can have been redefined without bumping
# the counter. Any other (world, counter) re-executes every fact query at
# the consuming world and compares answers — method-match sets, binding-
# partition loads, CodeInstance world cover, oracle answers: all equal ⇒
# the entry revalidates (fact objects refresh, so replayed world ranges are
# current); any difference ⇒ the entry is stale and dropped. Facts the
# trace cannot encode (staged expansions, mi-invoke edges, assignment-kind
# partition facts) POISON their windows: such frames are never stored.
# Mid-request counter movement is backstopped by the driver's finish
# protocol exactly as for live facts (`valid_worlds` must reach the
# validation world, else the result is declined, never published).
#
# Reentry discipline (the wave-5 P1 design rule): memo-table accesses use
# the ledger's owner-tracked bounded acquisition — same-task reentry
# proceeds unlocked, cross-task contention drops the memo operation (the
# memo is only ever an optimization). Fact revalidation and replay, which
# can demand compiles, run OUTSIDE the lock.

mutable struct MemoEntry
    const result::UResult
    facts::Vector{Any}     # flattened fact events (tags 0x1..0x4; see UEdges)
    world::UInt            # the world `facts` were last validated at
    counter::UInt          # world counter observed at the recording frame's start
end

const DRIVER_MEMO_ENABLED = Base.RefValue(true)
const DRIVER_MEMO = Dict{Any,Any}()            # (mi | UConstKey) -> MemoEntry
const MEMO_LOCK = Base.Threads.SpinLock()
const MEMO_OWNER = Base.RefValue{Any}(nothing)
const MEMO_HITS = Base.Threads.Atomic{Int}(0)
const MEMO_MISSES = Base.Threads.Atomic{Int}(0)
const MEMO_STALE = Base.Threads.Atomic{Int}(0)
const MEMO_STORES = Base.Threads.Atomic{Int}(0)
const MEMO_REPLAYED = Base.Threads.Atomic{Int}(0)
const MEMO_DROPPED = Base.Threads.Atomic{Int}(0)

_memo_acquire() = _guarded_acquire(MEMO_LOCK, MEMO_OWNER, true)
_memo_release(state::UInt8) = _guarded_release(MEMO_LOCK, MEMO_OWNER, state)

"Drop every memo entry (tests/benchmarks; never required for correctness —
entries self-invalidate through world revalidation)."
function reset_driver_memo!()
    lk = _memo_acquire()
    lk == 0x2 && return nothing
    try
        empty!(DRIVER_MEMO)
    finally
        _memo_release(lk)
    end
    return nothing
end

"Match-set equality for revalidation: the same methods, coverage and
ambiguity answer the consuming world's query."
function same_lookup(a::Compiler.MethodLookupResult, b::Compiler.MethodLookupResult)
    a.ambig == b.ambig || return false
    length(a.matches) == length(b.matches) || return false
    for i in 1:length(a.matches)
        ma = a.matches[i]::Core.MethodMatch
        mb = b.matches[i]::Core.MethodMatch
        ma.method === mb.method || return false
        ma.fully_covers == mb.fully_covers || return false
        ma.spec_types === mb.spec_types || ma.spec_types == mb.spec_types || return false
    end
    return true
end

"Partition-load equality for revalidation (rt/exct/effects of the read)."
same_rte(a::CC.RTEffects, b::CC.RTEffects) =
    lat_eq(a.rt, b.rt) && a.exct == b.exct && a.effects == b.effects

"The binding-partition load at `world`, as a pure query (no collector)."
function memo_partition_probe(world::UInt, mod::Module, name::Symbol)
    return try
        b = convert(Core.Binding, GlobalRef(mod, name))
        partition = CC.lookup_binding_partition(world, b)
        _, (leaf_b, leaf_partition) = CC.walk_binding_partition(b, partition, world)
        CC.abstract_eval_partition_load(nothing, leaf_b, leaf_partition)
    catch
        nothing
    end
end

"""Re-execute every recorded fact query at `world` and compare answers.
Returns the refreshed fact vector (query results carry current world
ranges), or `nothing` when any fact no longer reproduces — the entry is
stale. Pure with respect to the consuming collector."""
function memo_revalidate(facts::Vector{Any}, world::UInt)
    out = Vector{Any}(undef, length(facts))
    for (i, f) in enumerate(facts)
        tag = f[1]::UInt8
        if tag == 0x1
            res = try
                CC.findall(f[2], CC.InternalMethodTable(world); limit = f[4]::Int)
            catch
                nothing
            end
            res isa Compiler.MethodLookupResult || return nothing
            same_lookup(f[3]::Compiler.MethodLookupResult, res) || return nothing
            out[i] = (0x1, f[2], res, f[4])
        elseif tag == 0x2
            rte = memo_partition_probe(world, f[2]::Module, f[3]::Symbol)
            rte isa CC.RTEffects || return nothing
            same_rte(f[4]::CC.RTEffects, rte) || return nothing
            out[i] = f    # replay re-derives the partition read at its world
        elseif tag == 0x3
            ci = f[2]::Core.CodeInstance
            (ci.min_world <= world <= ci.max_world) || return nothing
            out[i] = f
        elseif tag == 0x4
            rt = try
                Core.Compiler.return_type(f[2], world)
            catch
                nothing
            end
            rt === f[3] || return nothing
            out[i] = f
        else
            return nothing    # unknown fact class: never serve it
        end
    end
    return out
end

"""Replay validated facts into the consuming request's collector: the same
record/clamp calls the original inference performed, so the entry
CodeInstance's edges and world bounds stay complete. Every replayed fact
also re-enters the trace (enclosing frame windows depend on it)."""
function memo_replay!(st::UInferState, facts::Vector{Any})
    col = st.edges::UEdges
    for f in facts
        tag = f[1]::UInt8
        if tag == 0x1
            record_call!(col, f[2], f[3]::Compiler.MethodLookupResult)
            trace!(col, f)
        elseif tag == 0x2
            # partition re-read at this request's world (globmemo-deduped;
            # answers were compared at validation): records edge + clamps +
            # trace event itself
            global_partition_rte(col, f[2]::Module, f[3]::Symbol)
        elseif tag == 0x3
            ci = f[2]::Core.CodeInstance
            clamp_world!(col, ci.min_world, ci.max_world)
            record_invoke!(col, nothing, ci)
            trace!(col, f)
        else # 0x4: oracle answers carry no edge of their own (the enclosing
             # match edge is a separate fact); trace only
            trace!(col, f)
        end
    end
    Base.Threads.atomic_add!(MEMO_REPLAYED, length(facts))
    return nothing
end

"""
    global_memo_lookup(st, key) -> Union{Nothing,UResult}

Serve a cross-request memo entry for `key` (mi or const key) into the
consuming request: validate the entry's facts at this request's world,
replay them into the collector, and record the replayed span so enclosing
frame windows stay fact-complete. `nothing` on miss/stale/contention (the
caller infers fresh)."""
function global_memo_lookup(st::UInferState, @nospecialize(key))
    DRIVER_MEMO_ENABLED[] || return nothing
    col = st.edges
    col === nothing && return nothing
    world = st.cfg.world
    lk = _memo_acquire()
    if lk == 0x2
        Base.Threads.atomic_add!(MEMO_DROPPED, 1)
        return nothing
    end
    local entry
    try
        entry = get(DRIVER_MEMO, key, nothing)
    finally
        _memo_release(lk)
    end
    if entry === nothing
        Base.Threads.atomic_add!(MEMO_MISSES, 1)
        return nothing
    end
    entry = entry::MemoEntry
    facts = entry.facts
    if !(entry.world == world && entry.counter == Base.get_world_counter())
        # something may have been redefined since the facts were recorded:
        # re-execute every fact query at this world (outside the lock)
        newfacts = memo_revalidate(facts, world)
        if newfacts === nothing
            Base.Threads.atomic_add!(MEMO_STALE, 1)
            lk = _memo_acquire()
            if lk == 0x2
                Base.Threads.atomic_add!(MEMO_DROPPED, 1)
            else
                try
                    # drop only OUR generation: a concurrent revalidation may
                    # have already refreshed the entry
                    get(DRIVER_MEMO, key, nothing) === entry && delete!(DRIVER_MEMO, key)
                finally
                    _memo_release(lk)
                end
            end
            return nothing
        end
        facts = newfacts
        counter = Base.get_world_counter()
        lk = _memo_acquire()
        if lk == 0x2
            Base.Threads.atomic_add!(MEMO_DROPPED, 1)
        else
            try
                entry.facts = newfacts
                entry.world = world
                entry.counter = counter
            finally
                _memo_release(lk)
            end
        end
    end
    lo = length(col.trace)
    memo_replay!(st, facts)
    col.spans[key] = (lo + 1, length(col.trace))
    Base.Threads.atomic_add!(MEMO_HITS, 1)
    return entry.result
end

"Flatten a trace window into a self-contained fact vector: span references
resolve recursively (order is irrelevant — record/clamp calls commute), and
duplicate events dedup by identity."
function memo_collect_facts(col::UEdges, lo::Int, hi::Int)
    trace = col.trace
    out = Any[]
    seen_spans = Set{Tuple{Int,Int}}()
    seen = Base.IdSet{Any}()
    work = Tuple{Int,Int}[(lo, hi)]
    while !isempty(work)
        (l, h) = pop!(work)
        (l, h) in seen_spans && continue
        push!(seen_spans, (l, h))
        for i in l:h
            ev = trace[i]
            if (ev[1]::UInt8) == 0x5
                push!(work, (ev[2]::Int, ev[3]::Int))
            elseif !(ev in seen)
                push!(seen, ev)
                push!(out, ev)
            end
        end
    end
    return out
end

"""
    memo_frame_store!(st, key, r, tlo, tpo, tco)

Record a cleanly-completed frame's result in the cross-request memo: `tlo`/
`tpo`/`tco` are the frame-entry trace length, poison count and world
counter. Sets the per-request span for `key` (fact-completeness for
enclosing windows) and, when the window is clean, publishes the entry."""
function memo_frame_store!(st::UInferState, @nospecialize(key), r::UResult,
                           tlo::Int, tpo::Int, tco::UInt)
    col = st.edges
    col === nothing && return nothing
    (col.poison == tpo && col.ok) || return nothing
    hi = length(col.trace)
    col.spans[key] = (tlo + 1, hi)
    DRIVER_MEMO_ENABLED[] || return nothing
    facts = memo_collect_facts(col, tlo + 1, hi)
    entry = MemoEntry(r, facts, st.cfg.world, tco)
    lk = _memo_acquire()
    if lk == 0x2
        Base.Threads.atomic_add!(MEMO_DROPPED, 1)
        return nothing
    end
    try
        DRIVER_MEMO[key] = entry
    finally
        _memo_release(lk)
    end
    Base.Threads.atomic_add!(MEMO_STORES, 1)
    return nothing
end

# ---------------------------------------------------------------------------
# Reentrancy / concurrency guard (see the header comment)
# ---------------------------------------------------------------------------

"Per-task driver state: nesting depth, the MethodInstances this task is
currently driving (each holds an engine reservation up-stack), and whether a
devirtualization CodeInstance production is in progress (`devirt` — bounds
eager callee compilation to one level per root chain; see
`driver_ci_for_invoke`)."
mutable struct DriverTaskState
    depth::Int
    devirt::Int
    const inflight::Base.IdSet{Core.MethodInstance}
end

const DRIVER_TLS_KEY = :unified_compiler_driver_state

function driver_task_state()::DriverTaskState
    tls = Base.task_local_storage()
    v = get(tls, DRIVER_TLS_KEY, nothing)
    v isa DriverTaskState && return v
    st = DriverTaskState(0, 0, Base.IdSet{Core.MethodInstance}())
    tls[DRIVER_TLS_KEY] = st
    return st
end

"Nested driver passes this task admits before declining (`:reentrant_depth`).
Native recursion: each level stacks a full pipeline pass (which itself
recurses per DRIVER_MAX_DEPTH), so the bound stays small — declined bodies
are stock-compiled once and cached, and devirtualization targets degrade to
MethodInstance invokes whose CodeInstances materialize on first call."
const DRIVER_REENTRY_LIMIT = Base.RefValue(8)

"Per-session admission budget for REENTRANT passes (requests arriving while
this task is already inside the driver — the self-hosting burn-in). Every
pre-A6 pass re-inferred its callee tree with fresh state, so admissions
CASCADED (a tower of nested fresh-state passes per admitted body) and the
valve had to stay tiny (32; 431a7d4e82 measured a budget of 1000 spending
~30s in the cascade where 32 spent ~2s). The A6 cross-request memo
amortizes the inference share of each admission (callee trees replay from
recorded facts), which reopens the valve to its pre-wedge width — and
makes it a net WIN for `activate!`: at budget 1000 it measures 58.7s
against the budget-32 baseline's 70.6s, with 990 unified bodies against
35. But the RUNTIME demo showed the other side of that ledger (wave-7 P1):
the memo amortizes only the inference share — entry-convert + optimizer +
typed exit are per-body and un-memoized, and a workload pass whose
first-touch surface is the driver's own interior specializations (surgery
helpers, pipeline passes) spends SECONDS per admitted body compiling the
compiler through its own optimizer, wedging `unified_driver_demo` pass 1
from ~60s to timeout. Until the per-body optimizer/exit work is amortized,
reentrant admissions beyond the small sample must go to stock (fast,
cached, semantically identical), so the valve stays at the wave-5 width."
const DRIVER_REENTRANT_BUDGET = Base.RefValue(32)
const REENTRANT_ADMITTED = Base.Threads.Atomic{Int}(0)

":invoke emission switch (devirtualize_calls!)."
const DEVIRTUALIZE = Base.RefValue(true)

"Eager CodeInstance productions per root pass (devirtualize_calls!): the
first N uncached targets get a recursive driver pass; the rest keep
MethodInstance invokes (dispatch-free; compiled and cached by the runtime
on first call). Deterministic and small — a root whose optimized body has
dozens of resolved callees (collect/print chains) must not multiply its
own compile time by that fan-out."
const DEVIRT_PRODUCTION_BUDGET = Base.RefValue(4)

# Per-body inference budgets: each request runs a fresh state — the edge
# collector's soundness requires every consumed method-table/binding fact to
# be observed within (or replayed into, via the A6 memo) this body's pass —
# so the budgets stay deliberately tight. Depth/frame cutoffs resolve
# through the stock return_type oracle (fast, cached, and covered by the
# recorded match edge + the traced oracle fact), trading callee-type
# precision for bounded per-body cost; memo hits consume no frames, so a
# warm session sees the cutoffs progressively less.
const DRIVER_MAX_DEPTH = Base.RefValue(16)
const DRIVER_FRAME_BUDGET = Base.RefValue(3_000)
# Reentrant passes (nested driver work: the runtime compiling the driver's
# own code mid-pass, and devirtualization targets) run with narrower budgets:
# the same soundness protocol at lower callee-type precision, so the
# self-hosting burn-in costs a fraction of a root pass. Cutoffs stay sound
# (return_type oracle + recorded edges), and the memo restores precision as
# leaves complete cleanly across requests.
const DRIVER_REENTRANT_MAX_DEPTH = Base.RefValue(4)
const DRIVER_REENTRANT_FRAME_BUDGET = Base.RefValue(400)

# ---------------------------------------------------------------------------
# One pipeline pass over one body
# ---------------------------------------------------------------------------

"A per-body decline: `reason` keys the ledger; `err` is retained evidence."
struct Fallback
    reason::Symbol
    err::Any
    Fallback(reason::Symbol, @nospecialize(err = nothing)) = new(reason, err)
end

"Everything one pipeline pass proves about a body (driver_infer's result)."
struct DriverResult
    src::Any                        # optimized CodeInfo (nothing when optimize=false)
    rt::Any                         # return lattice element (Const-precise)
    exct::Any
    effects::Compiler.Effects
    edges::Core.SimpleVector        # stock encoding (build_edges)
    valid_worlds::Compiler.WorldRange
    start_counter::UInt             # world counter at pass start
    rettype_const::Any
    const_flags::UInt8              # stock encoding: 0x2 rettype_const set, 0x3 const ABI
end

"""Per-axis upward refinement of the inference-time ipo effects with the
post-optimization recompute (stock `refine_effects!` semantics: an axis only
improves when the optimized body PROVES the better value — both computations
are sound for the emitted body, so taking the better bit per axis is too)."""
function refine_post_opt(base::Compiler.Effects, post::Compiler.Effects)
    return Compiler.Effects(base;
        consistent = post.consistent === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.consistent,
        effect_free = post.effect_free === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.effect_free,
        nothrow = base.nothrow | post.nothrow,
        terminates = base.terminates | post.terminates,
        notaskstate = base.notaskstate | post.notaskstate,
        inaccessiblememonly = post.inaccessiblememonly === Compiler.ALWAYS_TRUE ?
            Compiler.ALWAYS_TRUE : base.inaccessiblememonly,
        noub = post.noub === Compiler.ALWAYS_TRUE ? Compiler.ALWAYS_TRUE :
            (post.noub === Compiler.NOUB_IF_NOINBOUNDS &&
             base.noub === Compiler.ALWAYS_FALSE ? Compiler.NOUB_IF_NOINBOUNDS :
             base.noub),
        nortcall = base.nortcall | post.nortcall)
end

"Encode the collector's records as a stock-format CodeInstance edges vector:
`user_edges` (a staged expansion's generator-declared edges, already in
stock encoding) first, then binding edges, per-lookup MethodMatchInfo
encodings (mi_edge=true, so match backedges land on MethodInstances), and
invoke edges (incl. the devirtualizer's CodeInstance targets)."
function build_edges(col::UEdges, @nospecialize(user_edges = nothing))
    edges = Any[]
    if user_edges !== nothing
        for e in user_edges
            push!(edges, e)
        end
    end
    for b in col.bindings
        push!(edges, b)
    end
    for (atype, result) in col.calls
        fullmatch = Base.any(m -> (m::Core.MethodMatch).fully_covers, result.matches)
        info = Compiler.MethodMatchInfo(result, Core.methodtable, atype, fullmatch)
        Compiler._add_edges_impl(edges, info, #=mi_edge=#true)
    end
    for (invokesig, target) in col.invokes
        if invokesig === nothing
            target isa Core.CodeInstance ? Compiler.add_one_edge!(edges, target) :
                                           Compiler.add_one_edge!(edges, target::Core.MethodInstance)
        else
            Compiler.add_invoke_edge!(edges, invokesig, target)
        end
    end
    return Core.svec(edges...)
end

"The stock inline_cost_model criteria over the driver's optimized IRCode
(a sane equivalent of compute_inlining_cost, so the stock inliner can
consume unified-produced CodeInstances when pipelines mix)."
function driver_inlining_cost(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                              src0::Core.CodeInfo, ircode, @nospecialize(rt))
    src0.inlining == 0x02 && return Compiler.MAX_INLINE_COST      # @noinline
    declared_inline = src0.inlining == 0x01
    sig = Base.unwrap_unionall(mi.specTypes)
    (sig isa DataType && sig.name === Tuple.name) || return Compiler.MAX_INLINE_COST
    !declared_inline && rt === Union{} && return Compiler.MAX_INLINE_COST
    if declared_inline && Base.isdispatchtuple(mi.specTypes)
        return Compiler.MIN_INLINE_COST
    end
    params = Compiler.OptimizationParams(interp)
    cost_threshold = params.inline_cost_threshold
    declared_inline && (cost_threshold += 19 * params.inline_cost_threshold)
    return try
        Compiler.inline_cost_model(ircode, params, Int(cost_threshold))
    catch
        Compiler.MAX_INLINE_COST
    end
end

"""
    driver_infer(interp, mi; optimize=true, emit_code=true)
        -> Union{DriverResult,Fallback}

One unified-pipeline pass over `mi`'s body. Pure with respect to the global
caches: nothing is cached or reserved here — the callers decide (the cache
entry wraps this with engine semantics; the reflection bridges use the
result directly). `optimize = false` stops after inference; `emit_code =
false` runs the optimizer (so rt/effects/exct see post-optimization
refinement, stock's `ipo_dataflow_analysis!` analog) but skips the
CodeInfo exit — for effects/exct queries, which need no code. `src` is
`nothing` in both reduced modes.
"""
function driver_infer(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance;
                      optimize::Bool = true, emit_code::Bool = true,
                      max_depth::Int = DRIVER_MAX_DEPTH[],
                      frame_budget::Int = DRIVER_FRAME_BUDGET[])
    world = Compiler.get_inference_world(interp)
    def = mi.def
    def isa Method || return Fallback(:toplevel)
    def.is_for_opaque_closure && return Fallback(:opaque_closure)
    Compiler.InferenceParams(interp).force_enable_inference && return Fallback(:trim)
    ccall(:jl_get_module_infer, Cint, (Any,), def.module) == 0 &&
        return Fallback(:inference_disabled)

    start_counter = Base.get_world_counter()
    col = UEdges(world)
    world <= start_counter || return Fallback(:world_unprovable)

    # Generated functions: `retrieve_code_info` expands the staged body
    # (jl_code_for_staged) — the expansion's validity window arrives as
    # `src.min_world/max_world` (clamped below, exactly stock InferenceState's
    # rule) and any generator-declared edges as `src.edges` (appended raw to
    # the CodeInstance edges, stock compute_edges!' user_edges rule; the
    # runtime registers them on the cached uninferred expansion as well).
    # The generator itself is ordinary user code: any compilation it needs
    # reenters the driver (bounded recursion) or stock. Its errors surface
    # as a per-body decline — the stock path reproduces stock's call-time
    # generator-error semantics.
    staged = isdefined(def, :generator)
    src0 = try
        Compiler.retrieve_code_info(mi, world)
    catch err
        return Fallback(staged ? :staged_source : :no_source, err)
    end
    src0 isa Core.CodeInfo || return Fallback(staged ? :staged_source : :no_source)
    clamp_world!(col, src0.min_world, src0.max_world)
    user_edges = src0.edges
    user_edges isa Core.SimpleVector && isempty(user_edges) && (user_edges = nothing)
    user_edges isa Vector{Any} && isempty(user_edges) && (user_edges = nothing)

    local uir
    try
        uir = codeinfo_to_ir(src0; nargs = Int(def.nargs), name = def.name)
    catch err
        err isa UnsupportedIR || return Fallback(:internal_error, err)
        return Fallback(:entry_convert, err)
    end
    uir.meta[:method_instance] = mi
    uir.meta[:mi] = mi
    uir.meta[:slotnames] = src0.slotnames
    uir.meta[:propagate_inbounds] = src0.propagate_inbounds
    uir.sptypes = Any[t for t in mi.sparam_vals]
    uir.meta[:sptypes_lat] = sptypes_lattice(mi)
    let und = sptypes_undef(mi)
        und === nothing || (uir.meta[:sptypes_undef] = und)
    end
    let reads = sparam_statement_reads(src0)
        isempty(reads) || (uir.meta[:sparam_reads] = reads)
    end

    st = UInferState(UInferConfig(; world,
        max_methods = Compiler.InferenceParams(interp).max_methods,
        max_depth, frame_budget))
    st.edges = col
    argl = method_arglattice(def, mi, Any[])
    argl === nothing && return Fallback(:arglattice)

    local rt, effects, exct
    try
        infer_ir!(uir, copy(argl); state = st)
        # the ipo effects baseline is the INFERENCE-time frame effects with
        # the method-level `@assume_effects` override (stock's finish order);
        # the optimizer's recompute below only REFINES it upward — a
        # recompute over the inlined body can lose callee-override precision
        # (inlining dissolves the callee frames the overrides applied to)
        effects = apply_effects_override(def, frame_effects_meta(uir))
        exct = get(uir.meta, :exct, Any)
        if optimize
            uir = optimize_ir!(uir, argl; state = st, inline = true)
            # stock's ipo_dataflow_analysis!/refine_effects! analog: the
            # post-optimization body (branches folded, dead throws gone)
            # re-inferred; upgrade any axis it proves
            effects = refine_post_opt(effects, frame_effects_meta(uir))
        end
        rt = get(uir.meta, :rettype, Any)
    catch err
        err isa UnsupportedIR || return Fallback(:inference_error, err)
        return Fallback(:inference_unsupported, err)
    end
    rt = sanitize_intercond(def, rt)
    rt isa UInterCond && (rt = Bool)
    Compiler.is_nothrow(effects) && (exct = Union{})

    src = nothing
    if optimize && emit_code
        # :invoke emission for residual statically-resolved calls (each
        # rewrite is individually sound, so a failure just leaves the
        # remaining sites as dynamic calls)
        if DEVIRTUALIZE[]
            try
                devirtualize_calls!(uir, st, interp)
            catch
            end
            # exit-shape adoption (wave 6): Expr(:invoke_modify) targets for
            # statically-resolved atomic modify builtins (exit_typed.jl)
            try
                devirtualize_modifyops!(uir, st, interp)
            catch
            end
        end
        local ircode
        try
            ircode = ir_to_ircode(uir)
        catch err
            err isa UnsupportedIR || return Fallback(:exit_error, err)
            return Fallback(:typed_exit, err)
        end
        try
            nargs = Int(def.nargs)
            src = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
            slotnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), def.slot_syms)
            length(slotnames) < nargs && append!(slotnames,
                Symbol[Symbol("#arg", i) for i in (length(slotnames)+1):nargs])
            src.slotnames = slotnames
            src.slotflags = Base.fill(0x00, length(slotnames))
            src.slottypes = copy(ircode.argtypes)
            src.isva = def.isva
            src.nargs = UInt(nargs)
            ircode.debuginfo.def = mi
            Compiler.ir_to_codeinf!(src, ircode)
            src.rettype = CC.widenconst(rt)
            src.parent = mi
            src.min_world = col.valid_worlds.min_world
            src.max_world = col.valid_worlds.max_world
            src.inlining_cost = driver_inlining_cost(interp, mi, src0, ircode, CC.widenconst(rt))
        catch err
            return Fallback(:exit_error, err)
        end
    end

    col.ok || return Fallback(:world_unprovable)
    (col.valid_worlds.min_world <= world <= col.valid_worlds.max_world) ||
        return Fallback(:world_unprovable)
    edges = try
        build_edges(col, user_edges)
    catch err
        return Fallback(:internal_error, err)
    end
    src isa Core.CodeInfo && (src.edges = edges)

    rettype_const = nothing
    const_flags = 0x00
    if rt isa CC.Const
        rettype_const = rt.val
        constabi = Compiler.is_foldable_nothrow(effects) &&
                   Compiler.is_inlineable_constant(rt.val)
        const_flags = constabi ? 0x03 : 0x02
    elseif Compiler.isconstType(rt)
        rettype_const = Compiler.type_parameter(rt)
        const_flags = 0x02
    end

    return DriverResult(src, rt, exct, effects, edges, col.valid_worlds,
                        start_counter, rettype_const, const_flags)
end

# ---------------------------------------------------------------------------
# Devirtualization: statically-resolved residual calls become `:invoke`
# ---------------------------------------------------------------------------

"""
    driver_ci_for_invoke(interp, mi, allow_production) -> (Union{Nothing,CodeInstance}, produced)

A CodeInstance suitable as an `:invoke` target for `mi`: the world-covering
cache entry when one exists, else — when `allow_production` — a recursive
unified pass (per-task depth bound and inflight set apply; mutual
recursion, over-deep chains, and exhausted budgets return `nothing`, and
the site degrades to a MethodInstance invoke, which the runtime compiles
on first call through the ordinary entry). `produced` reports whether a
recursive pass ran (the per-pass DEVIRT_PRODUCTION_BUDGET accounting). No
JIT work happens here: the caller's `add_codeinsts_to_jit!` walk collects
embedded CodeInstance targets via `collectinvokes!`.
"""
function driver_ci_for_invoke(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                              allow_production::Bool)
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance &&
           Compiler.ci_meets_requirement(interp, code, Compiler.SOURCE_MODE_ABI)
            return (code, false)
        end
    end
    allow_production || return (nothing, false)
    dts = driver_task_state()
    (mi in dts.inflight || dts.depth >= DRIVER_REENTRY_LIMIT[]) && return (nothing, false)
    # eager production is bounded to ONE level, from ROOT passes only: a
    # nested (reentrant or production) pass embeds cached CodeInstances or
    # MethodInstance invokes. Without the root restriction the burn-in
    # compiles the STATIC call graph — far beyond the runtime-demand set —
    # eagerly; targets left as mi-invokes materialize (and cache) when the
    # runtime first needs them, so coverage converges by execution.
    (dts.devirt > 0 || dts.depth > 1) && return (nothing, false)
    local ci
    dts.depth += 1
    dts.devirt += 1
    push!(dts.inflight, mi)
    try
        ci = _unified_typeinf(interp, mi, Compiler.SOURCE_MODE_ABI)
    finally
        dts.depth -= 1
        dts.devirt -= 1
        delete!(dts.inflight, mi)
    end
    ci isa Core.CodeInstance || return (nothing, true)
    return (ci, true)
end

"""
    devirtualize_calls!(uir, st, interp) -> Int

Post-optimization `:invoke` emission (stock's inliner leaves
`Expr(:invoke, ci, ...)` at statically-resolved sites it does not inline):
for each residual `K"call"` whose signature — built from the final inferred
operand types, exactly what the last `infer_ir!` pass looked up — resolves
to a SINGLE, FULLY-COVERING method match in a world-clamped, edge-recorded
query (`resolve_single_match(st, sig)`), rewrite the statement to
`K"invoke"` targeting the callee's CodeInstance (produced through the
driver, bounded recursion) or, when a CI cannot be soundly produced right
now, the compilable MethodInstance. Soundness: the recorded match edge caps
this body's CodeInstance whenever the callee set changes, and the emitted
world bounds are additionally intersected with the callee CI's; sparams of
non-dispatch-tuple targets are re-derived per call by the runtime's invoke
convention, so the rewrite is dispatch-exact. Types/effects columns are
unchanged (the rewrite preserves semantics per statement).
"""
function devirtualize_calls!(uir, st::UInferState, interp::Compiler.AbstractInterpreter)
    n = 0
    produced = 0
    col = st.edges
    for s in UnifiedIR.each_stmt(uir)
        UnifiedIR.is_tombstone(uir, s) && continue
        UnifiedIR.stmt_kind(uir, s) === K"call" || continue
        nop = UnifiedIR.nops(uir, s)
        nop >= 1 || continue
        args = Any[stmt_lattice(uir, UnifiedIR.getop(uir, s, i)) for i in 1:nop]
        f = CC.singleton_type(args[1])
        f === nothing && args[1] isa CC.Const && (f = (args[1]::CC.Const).val)
        f === nothing && continue
        (f isa Core.Builtin || f isa Core.IntrinsicFunction) && continue
        argts = Any[CC.widenconst(a) for a in args[2:end]]
        Base.any(t -> t === Union{} || !(t isa Type) || CC.has_free_typevars(t), argts) && continue
        ft = f isa Type ? Type{f} : typeof(f)
        sig = try
            Tuple{ft, argts...}
        catch
            continue
        end
        match = resolve_single_match(st, sig)   # records the match edge
        match === nothing && continue
        match.fully_covers || continue
        mi = try
            CC.specialize_method(match)
        catch
            continue
        end
        mi isa Core.MethodInstance || continue
        target = ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)
        target isa Core.MethodInstance || continue
        # stock's :invoke legality (compileable_specialization): the target's
        # static parameters must be fully determined. An under-constrained
        # match (free TypeVars/Varargs/SimpleVectors in the environment the
        # runtime re-derives per call) leaves the callee's sparam reads
        # unbound — the emitted code throws `UndefVarError: T` at the first
        # `static_parameter` use. Such sites keep the dynamic :call.
        sparams = target.sparam_vals
        (CC.unionall_depth((match.method).sig) == length(sparams) &&
         CC.validate_sparams(sparams)) || continue
        (ci, did_produce) = driver_ci_for_invoke(interp, target,
                                                 produced < DEVIRT_PRODUCTION_BUDGET[])
        did_produce && (produced += 1)
        tgt = ci === nothing ? target : ci
        if ci isa Core.CodeInstance && col isa UEdges
            # the embedded CI must cover every world this body claims
            clamp_world!(col, ci.min_world, ci.max_world) || continue
        end
        ops = UnifiedIR.Operand[UnifiedIR.vop(uir, tgt)]
        for i in 1:nop
            push!(ops, UnifiedIR.getop(uir, s, i))
        end
        UnifiedIR.replace_stmt!(uir, s, K"invoke", ops...;
                                type = UnifiedIR.stmt_type(uir, s),
                                flag = UnifiedIR.stmt_flag(uir, s))
        n += 1
    end
    return n
end

# ---------------------------------------------------------------------------
# The cache-grade entry (typeinf_ext_toplevel hook)
# ---------------------------------------------------------------------------

"Fill + publish a driver result following stock's finish!/promotecache!
sequence. Takes ownership of the engine-reserved `ci`."
function finish_unified!(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                         ci::Core.CodeInstance, result::DriverResult)
    valid_worlds = result.valid_worlds
    validation_world = Base.get_world_counter()
    if valid_worlds.max_world < validation_world
        # something moved (or was already bounded) during the pass: v0 never
        # publishes bounded CodeInstances (see the header comment)
        Compiler.engine_reject(interp, ci)
        count_fallback!(valid_worlds.max_world < result.start_counter ?
                        :world_bounded : :world_moved, mi)
        return nothing
    end
    src = result.src::Core.CodeInfo
    discard_src = result.const_flags == 0x03 && Compiler.may_discard_trees(interp)
    inferred = nothing
    debuginfo = nothing
    if !discard_src
        inferred = Compiler.maybe_compress_codeinfo(interp, mi, src)
        debuginfo = src.debuginfo
    end
    debuginfo === nothing && (debuginfo = Core.DebugInfo(mi))
    # all facts verified at validation_world: register invalidation edges
    Compiler.store_backedges(ci, result.edges)
    ipo = Compiler.encode_effects(result.effects)
    ccall(:jl_fill_codeinst, Cvoid,
          (Any, Any, Any, Any, Any, Int32, UInt, UInt, UInt32, Any,
           Float64, Float64, Float64, Any, Any),
          ci, CC.widenconst(result.rt), result.exct, result.rettype_const, inferred,
          Int32(result.const_flags), valid_worlds.min_world, valid_worlds.max_world,
          ipo, nothing, 0.0, 0.0, 0.0, debuginfo, result.edges)
    Compiler.code_cache(interp)[mi] = ci
    Compiler.engine_reject(interp, ci)          # fulfill: wake any waiters
    if !discard_src
        codegen = Compiler.codegen_cache(interp)
        codegen === nothing || (codegen[ci] = src)
    end
    ccall(:jl_promote_ci_to_current, Cvoid, (Any, UInt), ci, validation_world)
    note_unified!()
    return ci
end

function _unified_typeinf(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                          source_mode::UInt8)
    mi = ccall(:jl_normalize_to_compilable_mi, Any, (Any,), mi)::Core.MethodInstance
    # fast cache path (stock typeinf_ext's)
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance && Compiler.ci_meets_requirement(interp, code, source_mode)
            return code
        end
    end
    ci = Compiler.engine_reserve(interp, mi)
    # check cache again if it is still new after reserving in the engine
    let code = get(Compiler.code_cache(interp), mi, nothing)
        code isa Compiler.InferenceResult && (code = code.ci)
        if code isa Core.CodeInstance && Compiler.ci_meets_requirement(interp, code, source_mode)
            Compiler.engine_reject(interp, ci)
            return code
        end
    end
    local result
    # nested passes (depth > 1: reentrant driver-code compiles and
    # devirtualization targets — plus compiler-internal bodies at any
    # depth, see is_selfhost_module) run with the narrower budgets
    nested = driver_task_state().depth > 1 ||
             (mi.def isa Method && is_selfhost_module((mi.def::Method).module))
    max_depth = nested ? DRIVER_REENTRANT_MAX_DEPTH[] : DRIVER_MAX_DEPTH[]
    frame_budget = nested ? DRIVER_REENTRANT_FRAME_BUDGET[] : DRIVER_FRAME_BUDGET[]
    try
        result = driver_infer(interp, mi; max_depth, frame_budget)
        if result isa DriverResult && result.valid_worlds.max_world == result.start_counter &&
           Base.get_world_counter() > result.start_counter
            # the counter moved but no consulted fact was bounded below the
            # pass start: lazy binding/partition materialization (one bump
            # per binding per process). The bindings exist now — one retry
            # settles it.
            result = driver_infer(interp, mi; max_depth, frame_budget)
        end
        if result isa Fallback
            Compiler.engine_reject(interp, ci)
            count_fallback!(result.reason, mi, result.err)
            return nothing
        end
        return finish_unified!(interp, mi, ci, result::DriverResult)
    catch err
        # fallback discipline: NO unified-path error escapes the hook — the
        # reservation is released and stock compiles the body
        Compiler.engine_reject(interp, ci)
        count_fallback!(:internal_error, mi, err)
        return nothing
    end
end

"""Is `m` compiler-internal (a module named `Compiler` or `UnifiedIR`, or
nested inside one)? Such bodies are the compiler compiling ITSELF, whatever
call depth they arrive at: when the valve declines a body and the stock
path compiles it, the compiles that stock inference's own execution then
demands (its tfuncs, specialized on the full lattice stack — the wave-7
`replacefield!_tfunc` wedge) arrive at task depth 0, because the fallback
runs after `unified_typeinf` returned and popped its depth. Classifying
self-hosting by MODULE instead of only by depth routes them through the
same reentrant valve/budgets as the rest of the burn-in — a body like
`replacefield!_tfunc(::InferenceLattice{MustAliasesLattice{...}}, ...)`
must never receive a full-budget unified pass mid-workload."""
function is_selfhost_module(m::Module)
    while true
        (nameof(m) === :Compiler || nameof(m) === :UnifiedIR) && return true
        p = parentmodule(m)
        p === m && return false
        m = p
    end
end

"""
    unified_typeinf(interp::AbstractInterpreter, mi::MethodInstance, source_mode::UInt8)
        -> Union{Nothing,CodeInstance}

The `Compiler.UNIFIED_HOOKS.typeinf_ext_toplevel` implementation: the real
unified pipeline with stock cache/engine semantics. Returns `nothing` when
this body falls back (counted in `pipeline_stats()`); the caller —
`Compiler.typeinf_ext_toplevel` — then runs the stock path for it.
"""
function unified_typeinf(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                         source_mode::UInt8)
    dts = driver_task_state()
    if mi in dts.inflight
        # this task is already driving this exact body up-stack: recursing
        # can only redo the same pass against the engine placeholder
        count_fallback!(:reentrant_self)
        return nothing
    end
    if dts.depth >= DRIVER_REENTRY_LIMIT[]
        count_fallback!(:reentrant_depth)
        return nothing
    end
    selfhost = let d = mi.def
        d isa Method && is_selfhost_module(d.module)
    end
    if dts.depth > 0 || selfhost
        # reentrant request (the self-hosting burn-in, by depth or by
        # module): admit within the session budget, else decline precisely
        # — stock compiles + caches
        if REENTRANT_ADMITTED[] >= DRIVER_REENTRANT_BUDGET[]
            count_fallback!(:reentrant_budget)
            return nothing
        end
        Base.Threads.atomic_add!(REENTRANT_ADMITTED, 1)
    end
    local ci
    dts.depth += 1
    push!(dts.inflight, mi)
    try
        ci = _unified_typeinf(interp, mi, source_mode)
    finally
        dts.depth -= 1
        delete!(dts.inflight, mi)
    end
    ci isa Core.CodeInstance || return nothing
    # stock typeinf_ext_toplevel's JIT closure (needs no unified state; may
    # stock-infer un-JIT'd invoke targets)
    return Compiler.add_codeinsts_to_jit!(interp, ci, source_mode)
end

# ---------------------------------------------------------------------------
# Reflection bridges (typeinf_code / _infer_effects / _infer_exception_type)
# ---------------------------------------------------------------------------

# run `f(...)` under the driver's per-task depth accounting, declining
# (nothing) at the reentrancy bound or on any escaped unified-path error
# (the fallback discipline: the hook caller must always be able to continue
# on stock). Nested driver work stays bounded and every pass still builds
# its own fresh state.
function with_driver_guard(f)
    dts = driver_task_state()
    if dts.depth >= DRIVER_REENTRY_LIMIT[]
        count_fallback!(:reentrant_depth)
        return nothing
    end
    dts.depth += 1
    try
        return f()
    catch err
        count_fallback!(:internal_error, nothing, err)
        return nothing
    finally
        dts.depth -= 1
    end
end

"""
    unified_typeinf_code(interp, mi, run_optimizer) -> Union{Nothing,CodeInfo}

The `typeinf_code` bridge: `code_typed`/`@code_typed`/`code_warntype` show
the unified pipeline's optimized output. Like stock, a const-ABI result
renders as the synthetic `return <const>` CodeInfo. Unoptimized queries
(`optimize=false`) stay on stock (that view is representation-independent).
"""
function unified_typeinf_code(interp::Compiler.AbstractInterpreter, mi::Core.MethodInstance,
                              run_optimizer::Bool)
    run_optimizer || return nothing   # uninferred/unoptimized view: stock
    return with_driver_guard() do
        result = driver_infer(interp, mi)
        if result isa Fallback
            count_fallback!(result.reason, mi, result.err)
            return nothing
        end
        note_unified!()
        if result.const_flags == 0x03 && Compiler.may_discard_trees(interp)
            return Compiler.codeinfo_for_const(interp, mi, result.valid_worlds,
                                               result.edges, result.rettype_const)
        end
        return result.src::Core.CodeInfo
    end
end

"""
    unified_infer_effects(interp, tt, optimize) -> Union{Nothing,Effects}

The `_infer_effects` bridge (`Base.infer_effects`): per-match driver
inference merged with stock's MethodError accounting. `optimize` mirrors
stock's `typeinf_frame(...; run_optimizer)` semantics — the optimizer runs
(post-opt effects refinement) but no code is emitted. Declines whole-query
on any per-match fallback.
"""
function unified_infer_effects(interp::Compiler.AbstractInterpreter, @nospecialize(tt),
                               optimize::Bool)
    return with_driver_guard() do
        matches = Compiler.findall(tt, Compiler.method_table(interp))
        matches === nothing && return nothing
        effects = Compiler.EFFECTS_TOTAL
        if Compiler._may_throw_methoderror(matches)
            effects = Compiler.Effects(effects; nothrow = false)
        end
        for match in matches.matches
            match = match::Core.MethodMatch
            result = driver_infer(interp, Compiler.specialize_method(match);
                                  optimize, emit_code = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            note_unified!()
            effects = Compiler.merge_effects(effects, result.effects)
        end
        return effects
    end
end

"""
    unified_infer_exception_type(interp, tt, optimize) -> Union{Nothing,Type}

The `_infer_exception_type` bridge: the per-match frame exception-type
bestguess (the thrown-escape join tracked by inference, `Union{}` for
proven-nothrow bodies), plus stock's MethodError account.
"""
function unified_infer_exception_type(interp::Compiler.AbstractInterpreter, @nospecialize(tt),
                                      optimize::Bool)
    return with_driver_guard() do
        matches = Compiler.findall(tt, Compiler.method_table(interp))
        matches === nothing && return nothing
        exct = Union{}
        if Compiler._may_throw_methoderror(matches)
            exct = MethodError
        end
        for match in matches.matches
            match = match::Core.MethodMatch
            result = driver_infer(interp, Compiler.specialize_method(match);
                                  optimize, emit_code = false)
            if result isa Fallback
                count_fallback!(result.reason, nothing, result.err)
                return nothing
            end
            note_unified!()
            exct = CC.tmerge(CC.fallback_lattice, exct, result.exct)
        end
        return CC.widenconst(exct)
    end
end

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

"""
    enable_pipeline!() -> Nothing

Install the unified driver behind the Compiler module's standard entry
points (`Compiler.UNIFIED_HOOKS`): `typeinf_ext_toplevel`, `typeinf_code`,
`_infer_effects` and `_infer_exception_type` route NativeInterpreter
requests through the unified pipeline, falling back to stock per body
(`pipeline_stats()`). Combined with `@activate Compiler`-style reflection
activation, `code_typed`/`infer_effects` show unified results; combined
with [`activate!`](@ref)'s jl_set_typeinf_func flip, ALL runtime inference
routes here. Undo with [`disable_pipeline!`](@ref).
"""
function enable_pipeline!()
    Compiler.UNIFIED_HOOKS[] = Compiler.UnifiedHooks(
        unified_typeinf, unified_typeinf_code,
        unified_infer_effects, unified_infer_exception_type)
    return nothing
end

"Remove the driver from `Compiler.UNIFIED_HOOKS`: stock behavior, bit-identical."
function disable_pipeline!()
    Compiler.UNIFIED_HOOKS[] = nothing
    return nothing
end

pipeline_enabled() = Compiler.UNIFIED_HOOKS[] !== nothing
