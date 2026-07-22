# This file is a part of Julia. License is MIT: https://julialang.org/license

## Cancellation tokens
#
# Cancellation is organized around *cancellation token sources*
# (`Core.CancellationTokenSource`): level-triggered condition nodes arranged
# in a tree. Cancelling a source cancels its whole subtree; the cancelled
# state is monotonic (severities only escalate, never reset). Following the
# .NET split, a `CancellationToken` is the observe view handed to code that
# may only *react* to cancellation; the source is the capability to *request*
# it.
#
# The token governing a piece of code is carried dynamically as a scoped
# value (see `CANCEL_TOKEN`), which `@cancel_check` resolves at every check.
# Tasks inherit their creating task's scope, so cancellation scopes propagate
# to child tasks without explicit plumbing.
#
# This file provides the data model and the cooperative (polling) checks.
# Delivery — waking tasks parked in blocking operations, interrupting
# running computations, `cancel` keyword arguments on the blocking APIs, and
# the ^C machinery — builds on top of it separately.
#
# TODO(compiler): a scoped-value lookup is a `Core.current_scope()` read plus
# a persistent-dict (HAMT) lookup. The optimizer currently only folds
# `current_scope()` when the enclosing `@with` is visible in the same
# (post-inlining) frame; for inherited scopes the lookup stays in hot loops.
# Teaching the compiler to CSE/hoist `current_scope()` + `KeyValue.get` for
# inherited scopes (sound: the scope is enter/leave-balanced and
# task-private) would make per-iteration `@cancel_check` in tight loops
# cheap. Until then, hot loops can hoist manually via the
# `@cancel_check tok` form.

const CancellationTokenSource = Core.CancellationTokenSource

"""
    CancellationToken(src::CancellationTokenSource)

The observe side of a [`CancellationTokenSource`](@ref): code holding a
token can be interrupted by - and can query ([`iscancelled`](@ref)) -
cancellation of the associated source, but cannot request cancellation
itself. Only the holder of the *source* can call [`cancel!`](@ref).

A token takes effect in one of two ways: pass it as the `cancel` keyword
argument of a specific blocking operation, or scope it over a whole
computation via the [`CANCEL_TOKEN`](@ref) scoped value (spawned tasks
inherit it). Either way, once the source is cancelled the affected
operations throw a [`CancellationRequest`](@ref).

See the manual chapter on [Task Cancellation](@ref man-cancellation) for an
overview.
"""
struct CancellationToken
    source::CancellationTokenSource
end

"""
    CancellationRequest

The exception thrown by cancellation points whose governing cancellation
token has been cancelled. The `request` field records the severity
([`CANCEL_REQUEST_SAFE`](@ref), [`CANCEL_REQUEST_ABANDON_EXTERNAL`](@ref) or
[`CANCEL_REQUEST_ABANDON_ALL`](@ref)) as observed at delivery time; the
source may escalate afterwards.
"""
struct CancellationRequest
    request::UInt8
end

"""
    CANCEL_REQUEST_SAFE

Request safe cancellation. Code observing the cancellation will request safe
cancellation of any resources it is waiting for and wait for the cancellation
of such resources to be completed.

As a result, if either the cancelled code or any of its dependent resources
are currently unable to process cancellation, the request may hang and a more
aggressive cancellation severity may be required. However, in general _SAFE
should be tried first.
"""
const CANCEL_REQUEST_SAFE = CancellationRequest(0x0)

"""
    CANCEL_REQUEST_ACK

Set by the task itself to indicate that a (safe) cancellation request was
received and acknowledged, but that there are dependent tasks for whom
cancelation is still pending.
"""
const CANCEL_REQUEST_ACK = CancellationRequest(0x1)

"""
    CANCEL_REQUEST_QUERY

Request that the system create an asynchronous report of why the task is currently
not able to be canceled. The report will be provided in the ->cancelation_request
field of the current task (as long as this field is still CANCEL_REQUEST_QUERY).

N.B.: Transition to CANCEL_REQUEST_QUERY is only allowed from CANCEL_REQUEST_ACK.
      Once the waiting task has read the cancelation report, it may set the cancelation
      request back to CANCEL_REQUEST_ACK.
"""
const CANCEL_REQUEST_QUERY = CancellationRequest(0x2)

"""
    CANCEL_REQUEST_ABANDON_EXTERNAL

Request a cancellation that will cease waiting for any external resources
(e.g. I/O objects) without going through a safe cancellation procedure for
such resources. However, internal computational tasks are still awaited.

This is a middleground between CANCEL_REQUEST_SAFE and
CANCEL_REQUEST_ABANDON_ALL, as external I/O is often engineered for
robustness in case of sudden disappearance of peers.
"""
const CANCEL_REQUEST_ABANDON_EXTERNAL = CancellationRequest(0x3)

"""
    CANCEL_REQUEST_ABANDON_ALL

Request a cancellation that will cease waiting for all external resources,
and give up on tasks that have not responded to the cancellation: they are
frozen in place and never scheduled again.

!!! warning
    If any cancelled task has acquired locks or other resources that are
    contested, this method of cancellation may leak such resources and create
    deadlocks in future code. It is intended as a last-resort method to
    recover a system, but the necessity of this operation should in general
    be considered a bug (e.g. due to insufficient cancellation points in
    computationally-heavy code).
"""
const CANCEL_REQUEST_ABANDON_ALL = CancellationRequest(0x4)

"""
    CANCEL_REQUEST_YIELD

Request that the task yield to the scheduler at the next cancellation point to
allow another task to run its cancellation propagation logic. The cancelled task
itself will reset to ordinary operation before yielding, but may of course be
canceled by said other task before it resumes operation.
"""
const CANCEL_REQUEST_YIELD = CancellationRequest(0x5)

# The state byte of a cancelled source is (STATE_CANCELLED_BIT | severity).
# The 0x40 bit is reserved (status bytes of compiled cancellation points use
# it to report a pending cooperative-yield request).
const STATE_CANCELLED_BIT = 0x80
const STATUS_PREEMPT_BIT = 0x40
const SEVERITY_MASK = 0x3f

severity(cr::CancellationRequest) = cr.request & SEVERITY_MASK

"""
    cancel_severity(src::CancellationTokenSource) -> Union{Nothing, CancellationRequest}
    cancel_severity(tok::CancellationToken)

Return `nothing` if the source has not been cancelled, or a
`CancellationRequest` recording the current (monotonically escalating)
severity if it has.
"""
function cancel_severity(src::CancellationTokenSource)
    st = @atomic :acquire src.state
    st == 0x00 && return nothing
    return CancellationRequest(st & SEVERITY_MASK)
end
cancel_severity(tok::CancellationToken) = cancel_severity(tok.source)

"""
    iscancelled(src::CancellationTokenSource)::Bool
    iscancelled(tok::CancellationToken)::Bool

Whether the source has been cancelled (level-triggered: once cancelled, a
source stays cancelled).
"""
iscancelled(src::CancellationTokenSource) = (@atomic :monotonic src.state) != 0x00
iscancelled(tok::CancellationToken) = iscancelled(tok.source)

## Source construction and tree linkage
#
# A source is a variable-sized object: its fixed fields (`child_head`,
# `state`, `nparents`) are followed by `nparents` {parent,
# next, pprev} link entries (see `jl_cancel_source_t` in julia_threads.h).
# The `parent` slots are strong, const references - a child keeps its
# parents alive - while `child_head` and the `next`/`pprev` slots form
# intrusive per-parent sibling lists of *weak* references: when a source is
# collected, the sweep - which visits every dead object anyway - detects it
# and unlinks it from its parents' lists in O(1) via the `pprev`
# back-pointer, so there is no explicit detach operation, the collector's
# work is proportional to the number of sources that actually died, and, at
# any point the mutator can observe, the lists contain only live sources.
#
# The lists are lock-free: mutators only ever prepend (in the C constructor
# `jl_new_cancel_source`, via CAS on `child_head`); removal happens only
# inside the collector with the world stopped. Construction and cancellation
# synchronize with seq_cst operations so that attachment is level-triggered:
# either the canceller's walk observes the new child, or the constructor
# observes the cancelled state (and the child is born cancelled).

# Construction is the builtin `Core._new_cancel_source(parents...)`
# (jl_new_cancel_source): it allocates the object with one link entry per
# argument and performs the linking and state inheritance in C, where the
# absence of safepoints between publishing a link and reading the parent's
# state can be guaranteed.

"""
    CancellationTokenSource() -> CancellationTokenSource
    CancellationTokenSource(parents::CancellationToken...)

Create a new cancellation token source. With no arguments the source is a
standalone root; given one or more parent tokens, the new source is linked
underneath each of them, so that cancellation of *any* parent (or any of its
ancestors) also cancels the new source - at the highest severity requested
among them. Sources therefore form a directed acyclic graph; a source
created under an already-cancelled parent is born cancelled. Linking one
source under several parents is how an operation respects two independent
lifetimes at once (say, a request scope and the connection it arrived on).

A child source stays linked to its parents for exactly as long as it is
reachable - a held token, or work governed by it, keeps it alive. Once
nothing can observe it any more, it is garbage collected and thereby drops
out of the graph; there is no explicit detach operation.

Use [`CancellationToken`](@ref)`(src)` for the observe view, and
[`cancel!`](@ref)`(src)` to request cancellation.
"""
CancellationTokenSource(parent::CancellationToken) =
    Core._new_cancel_source(parent.source)::CancellationTokenSource
function CancellationTokenSource(parent::CancellationToken, rest::CancellationToken...)
    srcs = CancellationTokenSource[parent.source]
    for tok in rest
        any(s -> s === tok.source, srcs) || push!(srcs, tok.source)
    end
    return Core._new_cancel_source(srcs...)::CancellationTokenSource
end
CancellationTokenSource(::Nothing) = Core._new_cancel_source()::CancellationTokenSource
CancellationTokenSource() = Core._new_cancel_source()::CancellationTokenSource

# The i-th (1-based) parent of `src`. Parent links are strong and const, so
# these reads need no synchronization.
_cancel_parent(src::CancellationTokenSource, i::Int) =
    ccall(:jl_cancel_source_parent, Any, (Any, Csize_t), src, i - 1)::CancellationTokenSource

# The sibling after `child` on `parent`'s child list (`nothing` at its end).
# Weak, but safe to traverse from Julia: the returned reference is rooted the
# moment the ccall returns, and the GC's splice pass keeps the lists free of
# collected entries at every safepoint, so a traversal (re-)started from a
# rooted node only ever sees live sources.
_cancel_next_child(parent::CancellationTokenSource, child::CancellationTokenSource) =
    ccall(:jl_cancel_source_next_child, Any, (Any, Any), parent, child)::Union{Nothing, CancellationTokenSource}

# CAS-max the source's state to (STATE_CANCELLED_BIT | sev). Returns true if
# the state was raised, false if it was already at (or above) the severity.
# seq_cst: pairs with the (child-list publication; state read) sequence in
# `jl_new_cancel_source` - see the walk in `_cancel_walk_node!`.
function _raise_state!(src::CancellationTokenSource, sev::UInt8)
    old = @atomic :monotonic src.state
    while true
        if old != 0x00 && (old & SEVERITY_MASK) >= sev
            return false
        end
        old, success = @atomicreplace :sequentially_consistent :monotonic src.state old => (STATE_CANCELLED_BIT | sev)
        success && return true
    end
end

## Cancellation
#
# Cancellation is uniformly level-triggered: while the governing token is
# cancelled, every cancellation point throws the `CancellationRequest`.
# There is no per-task acknowledgement state; code that must keep running
# under a cancelled scope shields itself by scoping `CANCEL_TOKEN => nothing`
# over the block.


## Wait registrations (used by condition.jl and every parked wait)

# A task's registration on a wait queue. All fields are plain: `next` and
# `queue` are protected by the waitee's lock (`queue` holds the queue's
# identity - see `waitqueue` - while the entry is enqueued, acting as the
# "am I registered, and on what" witness, and `nothing` otherwise); `task`
# is written by the owning task before enqueueing, so it is ordered by the
# same lock for lock-holding readers.
#
# The wake-claim protocol: a parked task `t` points to its current
# registration through the atomic field `t.waiting_on`. Whoever wants to wake
# it must first claim the wake by atomically clearing that field:
#
#   - `notify` (holding the waitee's lock) pops an entry `w` and claims via
#     CAS(t.waiting_on, w => nothing). The expected-value CAS makes stale
#     entries harmless: if `t` was interrupted and has since registered
#     elsewhere, the CAS fails and the popped corpse is simply dropped.
#   - an interrupter (`schedule(t, exc, error=true)`) claims via an
#     unconditional swap: it is directed at the *task*, not at any particular
#     wait, so claiming whatever `t` is currently registered on is correct.
#     The claimed entry stays linked - the interrupter may not touch the queue
#     without its lock - and is unlinked lazily, either by the interrupted
#     task's own wait cleanup or by the `notify` that pops and drops it.
#   - wake sources directed at one *specific* wait (e.g. the timeout task of
#     `Experimental.wait_with_timeout`) must register the wait with a fresh,
#     single-use entry: single-use-ness is what guarantees their
#     expected-value CAS cannot mistakenly claim a later, unrelated wait.
#
# Entries are heap objects (rather than links folded into the Task) so that a
# task whose interrupted wait left a stale registration behind can immediately
# register anew - e.g. park on a lock during its cleanup - with a fresh entry.
# To keep the common park allocation-free, each task caches one entry
# (`t.cached_wait_entry`) and reuses it whenever it is free, i.e. not still
# linked into some queue (`w.queue === nothing`). Reuse requires the owning
# task to be synchronized with the unlinker: either the task unlinked the entry
# itself, or the unlinker subsequently scheduled it. Interrupted-wait cleanup
# temporarily removes its entry from the cache before relocking, since another
# task may unlink that stale entry without being the task that scheduled us.
mutable struct WaitEntry
    task::Union{Task, Nothing}
    next::Union{WaitEntry, Nothing}
    queue::Any
    # The cancellation half of the registration: while the wait runs under a
    # cancellation token, `token` holds its source and `tnext`/`tprev` link
    # this entry into that source's waiter list (guarded by the source's
    # `_lock`), where the cancellation walk finds and claims the parked
    # task. `min_severity` admits teardown waits that re-park until an
    # escalation. See `register_cancellation!`.
    token::Union{Nothing, Core.CancellationTokenSource}
    tnext::Union{Nothing, WaitEntry}
    tprev::Union{Nothing, WaitEntry}
    min_severity::UInt8
    # For waits on an in-flight libuv request (stream writes/shutdowns, UDP
    # sends, getaddrinfo): the uv request this registration represents, so a
    # cancelling claimer can `uv_cancel` it (see stream.jl). C_NULL
    # otherwise.
    uvreq::Ptr{Cvoid}
    WaitEntry(task::Union{Task, Nothing}) =
        new(task, nothing, nothing, nothing, nothing, nothing, 0x00, C_NULL)
end

# Return the cached entry of `waiter` if it is free, else a fresh (and newly
# cached) one.
function _cached_wait_entry(waiter::Task)
    w = waiter.cached_wait_entry
    if w isa WaitEntry && w.queue === nothing
        w.task = waiter
    else
        w = WaitEntry(waiter)
        waiter.cached_wait_entry = w
    end
    return w
end

## Delivery semantics
#
# Cancellation is uniformly level-triggered: while the governing token is
# cancelled, every cancellation point and every blocking-operation entry
# check throws the `CancellationRequest`. There is no per-task
# acknowledgement state; cleanup code that must block under a cancelled
# scope explicitly shields itself (`cancel = nothing`, or scoping
# `CANCEL_TOKEN => nothing` over a block), and the interactive machinery re-arms
# with a *fresh* episode source between epochs (see `sigint_new_episode!`),
# detaching any still-unwinding work from the ^C target.

# Record that a cancellation of `src` at severity `sev` was delivered to
# (observed by) some task: either thrown at one of its cancellation points,
# or handed to it by the cancellation walk waking its parked wait. Feeds the
# ^C episode state machine ("was the request ever seen?").
# TODO: propagate the delivered bits up the parent chain, so that a delivery
# against a nested scope's source is visible on the episode source too.
function _mark_delivered!(src::CancellationTokenSource, sev::UInt8)
    @atomic :monotonic src.delivered |= (0x01 << sev)
    return nothing
end

@noinline function _wait_registration_error()
    throw(ConcurrencyViolationError("Task is already registered on a wait queue"))
end

# Publish `w` as `waiter`'s only armed wait registration.
function _arm_wait(waiter::Task, w::WaitEntry)
    armed = @atomicreplace :release :monotonic waiter.waiting_on nothing => w
    armed.success || _wait_registration_error()
    return w
end

# Claim the wake of the wait that `w` was registered for (returns whether the
# claim succeeded). `w` must be an entry armed for `t` by `_wait2`.
function claim_wait(t::Task, w::WaitEntry)
    return (@atomicreplace t.waiting_on w => nothing).success
end

## Waiter registration

# Spinlock guarding a source's waiter list only; attachment and state stay
# lock-free (see the C-side concurrency notes on jl_cancel_source_t).
@inline function _lock_source(src::CancellationTokenSource)
    while !(@atomicreplace :acquire :monotonic src._lock 0x00 => 0x01).success
        ccall(:jl_cpu_suspend, Cvoid, ())
    end
    return nothing
end
@inline function _unlock_source(src::CancellationTokenSource)
    @atomic :release src._lock = 0x00
    return nothing
end

# Register the armed wait entry `w` (see `_wait2`/`_arm_wait`) on `src`'s
# waiter list, so the cancellation walk can find and claim the parked task.
# Refuses (returns `false`) when `src` is already cancelled at or above
# `min_severity`: the caller delivers the cancellation itself instead of
# parking. `min_severity` admits teardown waits that must survive lower
# severities and re-park until an escalation (see `sync_end`).
function register_cancellation!(src::CancellationTokenSource, w::WaitEntry;
                                min_severity::UInt8=0x00)
    _lock_source(src)
    st = @atomic :monotonic src.state
    if st != 0x00
        sev = st & SEVERITY_MASK
        if sev >= min_severity
            _unlock_source(src)
            return false
        end
    end
    w.token = src
    w.min_severity = min_severity
    tail = src.waiters_tail
    w.tprev = tail isa WaitEntry ? tail : nothing
    w.tnext = nothing
    if tail isa WaitEntry
        tail.tnext = w
    else
        src.waiters_head = w
    end
    src.waiters_tail = w
    _unlock_source(src)
    return true
end

# Remove `w` from `src`'s waiter list; a no-op if the cancellation walk
# already unlinked it.
function unregister_cancellation!(src::CancellationTokenSource, w::WaitEntry)
    _lock_source(src)
    if w.tprev !== nothing || w.tnext !== nothing || src.waiters_head === w
        _unlink_waiter!(src, w)
    end
    w.token = nothing
    _unlock_source(src)
    return nothing
end

# caller must hold src's lock and have checked membership
function _unlink_waiter!(src::CancellationTokenSource, w::WaitEntry)
    prev = w.tprev
    next = w.tnext
    if prev === nothing
        src.waiters_head = next === nothing ? nothing : next
    else
        prev.tnext = next
    end
    if next === nothing
        src.waiters_tail = prev === nothing ? nothing : prev
    else
        next.tprev = prev
    end
    w.tnext = nothing
    w.tprev = nothing
    return nothing
end

# Watcher (`wait(::CancellationToken)`) registration: park the registration
# entry `w` on `src`'s watcher list, whose parked tasks the cancellation walk
# *completes* - delivering the `CancellationRequest` as a value - in contrast
# to the waiter list above, whose parked tasks it interrupts with the request
# as an exception. Watcher entries link singly through their edge half
# (`next`, with `queue` identifying the source as the membership witness);
# the entry's level half (`token`/`tnext`/`tprev`) stays free for the
# ordinary registration on the wait's own governing token. Returns `false`
# (without parking) if `src` is already cancelled.
function _register_watcher!(src::CancellationTokenSource, w::WaitEntry)
    _lock_source(src)
    st = @atomic :monotonic src.state
    if st != 0x00
        _unlock_source(src)
        return false
    end
    w.queue = src
    watchers = src.watchers
    w.next = watchers isa WaitEntry ? watchers : nothing
    src.watchers = w
    _unlock_source(src)
    return true
end

# Remove `w` from `src`'s watcher list; a no-op if the cancellation walk
# already emptied it (`queue` no longer witnesses membership).
function _unregister_watcher!(src::CancellationTokenSource, w::WaitEntry)
    _lock_source(src)
    if w.queue === src
        p = src.watchers
        if p === w
            n = w.next
            src.watchers = n === nothing ? nothing : n
        else
            while p isa WaitEntry
                n = p.next
                if n === w
                    p.next = w.next
                    break
                end
                p = n
            end
        end
        w.next = nothing
        w.queue = nothing
    end
    _unlock_source(src)
    return nothing
end

## Cancellation

"""
    cancel!(src::CancellationTokenSource,
            request::CancellationRequest=CANCEL_REQUEST_SAFE)::Bool

Cancel `src` and its whole subtree at the given severity. Level-triggered
and monotonic: observers (including future registrants) see the cancellation
until the source is discarded, and repeated calls only have an effect when
they *escalate* the severity ([`CANCEL_REQUEST_SAFE`](@ref) ->
[`CANCEL_REQUEST_ABANDON_EXTERNAL`](@ref) ->
[`CANCEL_REQUEST_ABANDON_ALL`](@ref)). Returns whether the call changed the
state.

Computations governed by a token of the subtree observe the cancellation at
their cancellation points (see [`@cancel_check`](@ref)), which throw a
[`CancellationRequest`](@ref).
"""
function cancel!(src::CancellationTokenSource,
                 request::CancellationRequest=CANCEL_REQUEST_SAFE)
    sev = request.request
    if !(sev == 0x0 || sev == 0x3 || sev == 0x4)
        throw(ArgumentError("invalid cancellation severity $(repr(request.request))"))
    end
    _raise_state!(src, sev) || return false
    # Pairs with the compiler-order-only publication of per-task token
    # bindings at compiled cancellation points (upcoming): after this fence,
    # either the canceller observes the binding of a running task, or the
    # task's next cancellation point observes our state write.
    Threads.atomic_fence_heavy()
    # Mark the subtree (waking parked waiters): each node is marked before
    # its children so a concurrent construction of a child source is
    # level-triggered.
    _cancel_walk!(src, sev)
    # Interrupt computations currently running under the subtree.
    _cancel_running!(src, sev)
    return true
end

function _cancel_walk!(src::CancellationTokenSource, sev::UInt8)
    # Iterative worklist (no recursion): a deep source chain must not
    # overflow the canceller's stack, and a reconverging ("linked") graph
    # must visit each node once, not once per path.
    pending = CancellationTokenSource[src]
    while !isempty(pending)
        _cancel_walk_node!(pop!(pending), sev, pending)
    end
    return nothing
end

function _cancel_walk_node!(node::CancellationTokenSource, sev::UInt8,
                            pending::Vector{CancellationTokenSource})
    # Advance at least to the node's current severity: a concurrent higher-
    # severity cancel! may have raised the state after this walk's own
    # transition; its walk skips children this one already advanced, so this
    # walk must carry the escalated severity onward.
    st = @atomic :acquire node.state
    stsev = st & SEVERITY_MASK
    sev < stsev && (sev = stsev)
    creq = CancellationRequest(sev)
    # Claim and unlink waiters at this node. The actual wakes happen after
    # the lock is released: waking may take other locks (a frozen task's
    # donenotify), and waiters take their waitee's lock *before* this node's
    # lock, so waking under it could deadlock.
    towake = nothing
    tonotify = nothing
    _lock_source(node)
    w = node.waiters_head
    while w isa WaitEntry
        wnext = w.tnext
        if w.min_severity <= sev
            t = w.task
            claimed = t isa Task && (@atomicreplace t.waiting_on w => nothing).success
            _unlink_waiter!(node, w)
            w.token = nothing
            if claimed
                towake = (t::Task, towake)
            end
            # !claimed: a completion (or another interrupter) won the race;
            # the waiter resumes normally and unregisters its (now unlinked)
            # entry itself.
        end
        w = wnext
    end
    # Claim and unlink watchers (tasks in `wait(::CancellationToken)`). This
    # cancellation is the event they wait *for*, so they are woken with the
    # request as a value - and, unlike waiters, never frozen: a watcher
    # observes this source but does not run under it.
    w = node.watchers
    while w isa WaitEntry
        wnext = w.next
        t = w.task
        claimed = t isa Task && (@atomicreplace t.waiting_on w => nothing).success
        w.next = nothing
        w.queue = nothing
        if claimed
            tonotify = (t::Task, tonotify)
        end
        # !claimed: the wait's own governing token was cancelled first (or a
        # completion won the race); its epilogue unlinks nothing - we just
        # did.
        w = wnext
    end
    node.watchers = nothing
    _unlock_source(node)
    while towake !== nothing
        (t, towake) = towake::Tuple{Task, Any}
        if sev >= CANCEL_REQUEST_ABANDON_ALL.request
            # do not wake the task; freeze it in place
            freeze_task!(t, creq, node)
        else
            # The claimed wait-queue entry stays linked; the waiter's own
            # cleanup (or a later notify) lazily unlinks it.
            _mark_delivered!(node, sev)
            schedule(t, creq, error=true)
        end
    end
    # Watchers are woken with the request as a *value*: this cancellation is
    # the event their wait completes on.
    while tonotify !== nothing
        (t, tonotify) = tonotify::Tuple{Task, Any}
        _mark_delivered!(node, sev)
        schedule(t, creq)
    end
    # Walk the node's (weak, intrusive) child list, queueing the children
    # whose state this walk advanced; a child whose state was already at (or
    # above) this severity has been walked - or is being walked - by whoever
    # advanced it, so revisiting it would make a reconverging graph
    # exponential. The seq_cst `child_head` read below (paired with the
    # seq_cst state write that queued `node`) closes the race against a
    # concurrent attach: a child that this read misses was published after
    # our state write, so its constructor observes that write and the child
    # is born at (at least) this severity. Children attached concurrently
    # *during* the walk are prepended before the list positions already
    # traversed and are likewise born cancelled.
    c = @atomic node.child_head
    while c !== nothing
        c = c::CancellationTokenSource
        if _raise_state!(c, sev)
            push!(pending, c)
        end
        c = _cancel_next_child(node, c)
    end
    return nothing
end

# Re-run the delivery walk for an already-cancelled source at its current
# severity: wakes waiters that registered without observing the cancellation
# and re-sends the interruption signal to bound running computations (the
# signal-based delivery is best-effort and can be missed while a reset point
# is unpublished). Used by the ^C machinery when a repeat press arrives
# within the escalation grace period. Returns whether the source was
# cancelled at all.
function redeliver!(src::CancellationTokenSource)
    st = @atomic :acquire src.state
    st == 0x00 && return false
    sev = st & SEVERITY_MASK
    Threads.atomic_fence_heavy()
    _cancel_walk!(src, sev)
    _cancel_running!(src, sev)
    return true
end

# Interrupt computations currently running on some thread whose published
# bound token lies in the cancelled subtree: send the cancellation signal
# that unwinds compiled code to its most recent cancellation point.
function _cancel_running!(src::CancellationTokenSource, sev::UInt8)
    creq = CancellationRequest(sev)
    ct = current_task()
    self_bound = false
    tasks = ccall(:jl_cancel_collect_bound, Any, (Any,), src)::Vector{Any}
    for t in tasks
        t = t::Task
        istaskdone(t) && continue

        if t === ct
            # The canceller itself is governed by the cancelled subtree;
            # deliver to ourselves last (below), so that the remaining tasks
            # are still processed.
            self_bound = true
        elseif sev >= CANCEL_REQUEST_ABANDON_ALL.request
            freeze_task!(t, creq, src)
        else
            tid = ccall(:jl_get_task_tid, Int16, (Any,), t)
            if tid >= 0
                # Best-effort: the signal only unwinds published (reset-safe)
                # regions; a miss is recovered level-triggered at the task's
                # next cancellation point.
                ccall(:jl_send_cancellation_signal, Cvoid, (Int16,), tid)
            end
        end
    end
    if self_bound && sev >= CANCEL_REQUEST_ABANDON_ALL.request
        # Self-cancellation with ABANDON_ALL: unwind with the request.
        _mark_delivered!(src, sev)
        throw(creq)
    end
    # A SAFE/ABANDON_EXTERNAL self-cancellation is observed at the caller's
    # next cancellation point (level-triggered).
    return nothing
end

# Whether `src` has a cancellation the given task has not yet observed
# (level-triggered: any cancelled state counts).
function cancel_pending(src::CancellationTokenSource, t::Task=current_task())
    return (@atomic :monotonic src.state) != 0x00
end
cancel_pending(::Nothing, t::Task=current_task()) = false

# Called by the runtime when a task starts under a dynamic scope, before its
# body runs: a task spawned into an already-cancelled scope observes the
# cancellation immediately (and in particular a task spawned into an
# ABANDON_ALL-frozen scope never runs user code).
function start_task_cancel_check()
    s = default_cancel_source()
    s === nothing && return nothing
    st = Core.cancellation_point!(s)::UInt8
    st != 0x00 && handle_cancellation!(s, st)
    return nothing
end

## Cancellation points

# The slow path of `@cancel_check`: `st` is the (non-zero) state byte of the
# governing source.
@noinline function handle_cancellation!(src::Union{Nothing, CancellationTokenSource}, st::UInt8)
    ct = current_task()
    if st & STATUS_PREEMPT_BIT != 0x00
        # consume the cooperative-yield request
        @atomic :monotonic ct.preempt_request = 0x00
    end
    if st & STATE_CANCELLED_BIT == 0x00
        # preempt-only: let another task (e.g. a canceller sharing this
        # thread) run, then resume
        yield()
        return nothing
    end
    src = src::CancellationTokenSource
    # re-read: deliver the severity current at throw time, not the one the
    # fast path happened to observe
    st = @atomic :acquire src.state
    sev = st & SEVERITY_MASK
    _mark_delivered!(src, sev)
    throw(CancellationRequest(sev))
end

"""
    @cancel_check
    @cancel_check token

Explicit cancellation point: checks whether the cancellation token governing
the current computation has been cancelled and, if so, throws the
corresponding [`CancellationRequest`](@ref). Long-running computational code
should place these in its hot loops so that it can be cancelled.

The one-argument form checks an explicitly provided
`Union{Nothing, CancellationToken}` instead of resolving the scoped default
token; use it to hoist the token lookup out of a tight loop.
"""
macro cancel_check()
    quote
        local s = default_cancel_source()
        local st = Core.cancellation_point!(s)::UInt8
        st != 0x00 && handle_cancellation!(s, st)
        nothing
    end
end

macro cancel_check(tok)
    quote
        local t = $(esc(tok))
        local s = t === nothing ? nothing : (t::CancellationToken).source
        local st = Core.cancellation_point!(s)::UInt8
        st != 0x00 && handle_cancellation!(s, st)
        nothing
    end
end

# Throw the `CancellationRequest` if `src` is cancelled (level-triggered:
# no per-task state is consulted). This is the entry check of every blocking
# API taking a `cancel` keyword argument: it must run *before* the operation
# has any side effects. Unlike `@cancel_check` this is not a compiled
# cancellation point (it opens no async-interruptible region).
@inline function checkcancel(src::CancellationTokenSource)
    st = @atomic :monotonic src.state
    st == 0x00 && return nothing
    handle_cancellation!(src, st)
    return nothing
end
checkcancel(::Nothing) = nothing
checkcancel(tok::CancellationToken) = checkcancel(tok.source)


## The scoped default token

# The scoped-value key under which the governing cancellation token is
# carried. `AbstractScopedValue` so the ScopedValues API (`@with
# Base.CANCEL_TOKEN => tok ...`) works on it; the accessors below avoid the
# ScopedValues module so they are usable during early bootstrap.
struct CancelTokenKey <: AbstractScopedValue{Union{Nothing, CancellationToken}} end

"""
    CANCEL_TOKEN

The scoped value carrying the [`CancellationToken`](@ref) that governs the
current dynamic extent, or `nothing` if there is none. Blocking operations
default their `cancel` keyword argument to it, [`@cancel_check`](@ref)
checks it, and tasks spawned within a scope inherit it.

Establish a governing token with the standard scoped-value API
([`ScopedValues.@with`](@ref) / [`ScopedValues.with`](@ref)):

```julia
using Base.ScopedValues

src = Base.CancellationTokenSource()
with(Base.CANCEL_TOKEN => Base.CancellationToken(src)) do
    ...   # blocking operations in here are cancellable via `cancel!(src)`
end
```

Scoping `Base.CANCEL_TOKEN => nothing` instead *shields* the enclosed code
from an outer (possibly cancelled) token, making its blocking operations
non-cancellable; use this for cleanup that must complete while the
surrounding computation is being cancelled.

The current value can be read with `Base.CANCEL_TOKEN[]`, for example to
hand the governing token across a boundary that does not preserve dynamic
scope (a `ccall` callback, a queue consumed by unrelated tasks, another
process).
"""
const CANCEL_TOKEN = CancelTokenKey()

@inline function default_cancel_token()
    scope = Core.current_scope()::Union{Scope, Nothing}
    scope === nothing && return nothing
    v = KeyValue.get(scope.values, CANCEL_TOKEN)
    v === nothing && return nothing
    return something(v)::Union{Nothing, CancellationToken}
end

@inline function default_cancel_source()
    tok = default_cancel_token()
    tok === nothing && return nothing
    return (tok::CancellationToken).source
end

# The severity of the current dynamic scope's cancellation, or `nothing` if
# the scope is not cancelled (or there is no scoped token).
function ambient_cancel_severity()
    src = default_cancel_source()
    src === nothing && return nothing
    return cancel_severity(src)
end

# Whether the current dynamic scope was cancelled at a severity that directs
# it to abandon external (I/O) waits without safe teardown. External wait
# entry points consult this: when true, they must not park waiting for
# external resources (they issue their operation, if any, and return
# immediately).
function abandoning_external_waits(t::Task=current_task())
    sev = ambient_cancel_severity()
    return sev !== nothing && sev.request >= CANCEL_REQUEST_ABANDON_EXTERNAL.request
end

## `cancel` keyword-argument plumbing

# The sentinel default for `cancel` keyword arguments: "use the scoped
# default token". Resolution to a concrete token happens once, at the first
# potential-block point of an operation, so fast paths never pay for the
# scope lookup. `cancel = nothing` makes a wait explicitly non-cancellable.
#
# N.B.: a resolved token (`Union{Nothing, CancellationToken}`) is passed
# through *positional* arguments internally: passing the union as a keyword
# argument builds an abstractly-typed NamedTuple whose kwcall the optimizer
# cannot devirtualize (which, among other things, breaks `juliac --trim`).
struct UseDefaultToken end
const DEFAULT_CANCEL = UseDefaultToken()
const CancelTokenArg = Union{UseDefaultToken, CancellationToken, Nothing}
const MaybeToken = Union{Nothing, CancellationToken}

@inline resolve_cancel_token(::UseDefaultToken) = default_cancel_token()
@inline resolve_cancel_token(tok::Union{CancellationToken, Nothing}) = tok

# The entry check of a public API taking a `cancel` keyword argument:
# resolve the token and throw if it is already cancelled (uniformly
# level-triggered for the scoped default and explicit tokens alike).
@inline function check_cancel_arg(cancel::CancelTokenArg)
    tok = resolve_cancel_token(cancel)
    tok === nothing || checkcancel(tok.source)
    return tok
end

# Run `f()` in a dynamic scope governed by `tok` - the raw `Scope` form of
# `ScopedValues.@with(CANCEL_TOKEN => tok, f())`, which is not yet available
# at this point of bootstrap.
@eval function _run_with_cancel_token(f, tok::Union{Nothing, CancellationToken})
    $(Expr(:tryfinally, :(f()), nothing,
           :(Scope(Core.current_scope()::Union{Nothing, Scope}, CANCEL_TOKEN => tok))))
end

# Implementation of a `cancel` keyword argument as dynamic-scope sugar: with
# the default sentinel, run `f()` as-is (zero overhead; `f`'s blocking points
# resolve the scoped token themselves); with an explicit argument, check it
# (throwing before any side effect) and run `f()` in a scope governed by it.
# This composes with *any* implementation underneath `f` - including methods
# of user-defined types that know nothing about cancellation keywords - as
# long as its blocking points use the standard wait machinery. Passing
# `cancel = nothing` shadows an outer token, making `f`'s waits
# non-cancellable.
@inline function _with_cancel_arg(f, cancel::CancelTokenArg)
    cancel === DEFAULT_CANCEL && return f()
    tok = check_cancel_arg(cancel)
    return _run_with_cancel_token(f, tok)
end

## Waiting for cancellation as an event

"""
    wait(tok::CancellationToken; cancel=...)

Block until `tok`'s source is cancelled, and return the corresponding
[`CancellationRequest`](@ref) as an ordinary value; return immediately if it
already is. This inverts the usual delivery - cancellation of `tok` is the
event this operation waits *for*, not an interruption of it - and is the
building block of the watcher-task ("cancellation callback") pattern; see
the manual chapter on [Task Cancellation](@ref man-cancellation).

The wait itself accepts the standard `cancel` keyword argument (defaulting
to the scoped token) and is interrupted by that token like any other
blocking operation. Waiting on the token that also governs the wait is
refused with an `ArgumentError`, since completing and interrupting the wait
would be the same event; pass `cancel = nothing` to wait for `tok`
unconditionally.
"""
function wait(tok::CancellationToken; cancel::CancelTokenArg=DEFAULT_CANCEL)
    src = tok.source
    gov = resolve_cancel_token(cancel)
    govsrc = gov === nothing ? nothing : gov.source
    if govsrc === src
        throw(ArgumentError(
            "cannot wait for a token's cancellation under the same governing token; " *
            "pass `cancel = nothing` to wait for it unconditionally"))
    end
    govsrc === nothing || checkcancel(govsrc)
    st = @atomic :acquire src.state
    if st != 0x00
        sev = st & SEVERITY_MASK
        _mark_delivered!(src, sev)
        return CancellationRequest(sev)
    end
    ct = current_task()
    # One registration entry carries both halves: its edge half links it on
    # `src`'s watcher list, its level half registers on the governing source.
    w = _cached_wait_entry(ct)
    @atomic :release ct.waiting_on = w
    if !_register_watcher!(src, w)
        # cancelled between the fast path and registration
        @atomicreplace ct.waiting_on w => nothing
        st = @atomic :acquire src.state
        sev = st & SEVERITY_MASK
        _mark_delivered!(src, sev)
        return CancellationRequest(sev)
    end
    if govsrc !== nothing && !register_cancellation!(govsrc, w)
        @atomicreplace ct.waiting_on w => nothing
        _unregister_watcher!(src, w)
        checkcancel(govsrc) # delivers the cancellation (throws)
        error("cancellation registration refused, but the source is not cancelled")
    end
    ret = try
        wait()
    catch
        # Interrupted (governing token, throwto, ...): disarm before the
        # unregistrations below can register new waits, then unlink.
        @atomicreplace ct.waiting_on w => nothing
        govsrc === nothing || unregister_cancellation!(govsrc, w)
        _unregister_watcher!(src, w)
        rethrow()
    end
    # A value wake implies the walk claimed `waiting_on` and unlinked the
    # watcher registration; unregistering below is then a no-op.
    govsrc === nothing || unregister_cancellation!(govsrc, w)
    _unregister_watcher!(src, w)
    return ret::CancellationRequest
end
