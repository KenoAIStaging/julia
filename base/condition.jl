# This file is a part of Julia. License is MIT: https://julialang.org/license

## thread/task locking abstraction

@noinline function concurrency_violation()
    # can be useful for debugging
    #try; error(); catch; ccall(:jlbacktrace, Cvoid, ()); end
    throw(ConcurrencyViolationError("lock must be held"))
end

"""
    AbstractLock

Abstract supertype describing types that
implement the synchronization primitives:
[`lock`](@ref), [`trylock`](@ref), [`unlock`](@ref), and [`islocked`](@ref).
"""
abstract type AbstractLock end
function lock end
function unlock end
function trylock end
function islocked end
unlockall(l::AbstractLock) = unlock(l) # internal function for implementing `wait`
relockall(l::AbstractLock, token::Nothing) = lock(l) # internal function for implementing `wait`
assert_havelock(l::AbstractLock, tid::Integer) =
    (islocked(l) && tid == Threads.threadid()) ? nothing : concurrency_violation()
assert_havelock(l::AbstractLock, tid::Task) =
    (islocked(l) && tid === current_task()) ? nothing : concurrency_violation()
assert_havelock(l::AbstractLock, tid::Nothing) = concurrency_violation()

"""
    AlwaysLockedST

This struct does not implement a real lock, but instead
pretends to be always locked on the original thread it was allocated on,
and simply ignores all other interactions.
It also does not synchronize tasks; for that use a real lock such as [`ReentrantLock`](@ref).
This can be used in the place of a real lock to, instead, simply and cheaply assert
that the operation is only occurring on a single cooperatively-scheduled thread.
It is thus functionally equivalent to allocating a real, recursive, task-unaware lock
immediately calling `lock` on it, and then never calling a matching `unlock`,
except that calling `lock` from another thread will throw a concurrency violation exception.
"""
struct AlwaysLockedST <: AbstractLock
    ownertid::Int16
    AlwaysLockedST() = new(Threads.threadid())
end
assert_havelock(l::AlwaysLockedST) = assert_havelock(l, l.ownertid)
lock(l::AlwaysLockedST) = assert_havelock(l)
unlock(l::AlwaysLockedST) = assert_havelock(l)
trylock(l::AlwaysLockedST) = l.ownertid == Threads.threadid()
islocked(::AlwaysLockedST) = true


## condition variables

# (The WaitEntry registration type and the wake-claim protocol live in
# cancellation.jl, which is included earlier in bootstrap: registrations
# carry the cancellation half of a parked wait.)

"""
    GenericCondition

Abstract implementation of a condition object
for synchronizing task objects with a given lock.
"""
mutable struct GenericCondition{L<:AbstractLock}
    # mutable for identity only
    const waitq::IntrusiveLinkedList{WaitEntry}
    const lock::L

    GenericCondition{L}() where {L<:AbstractLock} = new{L}(IntrusiveLinkedList{WaitEntry}(), L())
    GenericCondition{L}(l::L) where {L<:AbstractLock} = new{L}(IntrusiveLinkedList{WaitEntry}(), l)
    GenericCondition(l::AbstractLock) = new{typeof(l)}(IntrusiveLinkedList{WaitEntry}(), l)
end

waitqueue(c::GenericCondition) = ILLRef(c.waitq, c)

"""
    try_unlink_claimed!(w::WaitEntry)

Opportunistically attempt to unlink a wait entry from its queue. This is a memory pressure
optimization. If the queue is locked by another task, the entry will remain linked and will
be unlinked upon the next wakeup attempt.
"""
function try_unlink_claimed!(w::WaitEntry)
    q = w.queue
    q === nothing && return true
    # Manual split for --trim
    if q isa Task
        dn = q.donenotify
        dn isa GenericCondition{Threads.SpinLock} || return false
        return _try_unlink_from!(dn, q, w)
    elseif q isa GenericCondition{Threads.SpinLock}
        return _try_unlink_from!(q, q, w)
    elseif q isa GenericCondition{ReentrantLock}
        return _try_unlink_from!(q, q, w)
    elseif q isa GenericCondition{AlwaysLockedST}
        return _try_unlink_from!(q, q, w)
    end
    return false
end

function _try_unlink_from!(c::GenericCondition, @nospecialize(waitee), w::WaitEntry)
    trylock(c.lock) || return false
    try
        list_deletefirst!(ILLRef(c.waitq, waitee), w)
    finally
        unlock(c.lock)
    end
    return true
end

show(io::IO, c::GenericCondition) = print(io, GenericCondition, "(", c.lock, ")")

assert_havelock(c::GenericCondition) = assert_havelock(c.lock)
lock(c::GenericCondition) = lock(c.lock)
unlock(c::GenericCondition) = unlock(c.lock)
trylock(c::GenericCondition) = trylock(c.lock)
islocked(c::GenericCondition) = islocked(c.lock)

lock(f, c::GenericCondition) = lock(f, c.lock)

# have waiter wait for c: register `waiter` on c's wait queue (with `waitee`
# recorded as the queue identity) and arm the registration for a wake.
# Returns the registration entry.
function _wait2(c::GenericCondition, waiter::Task, first::Bool=false;
                waitee=c, entry::Union{WaitEntry, Nothing}=nothing)
    ct = current_task()
    assert_havelock(c)
    w = entry === nothing ? _cached_wait_entry(waiter) : entry
    _arm_wait(waiter, w)
    if first
        pushfirst!(ILLRef(c.waitq, waitee), w)
    else
        push!(ILLRef(c.waitq, waitee), w)
    end
    # since _wait2 is similar to schedule, we should observe the sticky bit now
    if waiter.sticky && Threads.threadid(waiter) == 0 && !GC.in_finalizer()
        # Issue #41324
        # t.sticky && tid == 0 is a task that needs to be co-scheduled with
        # the parent task. If the parent (current_task) is not sticky we must
        # set it to be sticky.
        # XXX: Ideally we would be able to unset this
        ct.sticky = true
        tid = Threads.threadid()
        ccall(:jl_set_task_tid, Cint, (Any, Cint), waiter, tid-1)
    end
    return w
end

"""
    wait([x])

Block the current task until some event occurs.

* [`Channel`](@ref): Wait for a value to be appended to the channel.
* [`Condition`](@ref): Wait for [`notify`](@ref) on a condition and return the `val`
  parameter passed to `notify`. See the `Condition`-specific docstring of `wait` for
  the exact behavior.
* `Process`: Wait for a process or process chain to exit. The `exitcode` field of a process
  can be used to determine success or failure.
* [`Task`](@ref): Wait for a `Task` to finish. See the `Task`-specific docstring of `wait` for
  the exact behavior.
* [`RawFD`](@ref): Wait for changes on a file descriptor (see the `FileWatching` package).

If no argument is passed, the task blocks for an undefined period. A task can only be
restarted by an explicit call to [`schedule`](@ref) or [`yieldto`](@ref).

Often `wait` is called within a `while` loop to ensure a waited-for condition is met before
proceeding.
"""
function wait end

"""
    wait(c::GenericCondition; first::Bool=false, cancel=Base.DEFAULT_CANCEL)

Wait for [`notify`](@ref) on `c` and return the `val` parameter passed to `notify`.

If the keyword `first` is set to `true`, the waiter will be put _first_
in line to wake up on `notify`. Otherwise, `wait` has first-in-first-out (FIFO) behavior.

The `cancel` keyword argument controls which cancellation token may interrupt
the wait (throwing the [`CancellationRequest`](@ref) into the waiter): by
default the scoped token (see `Base.CANCEL_TOKEN`); pass a
[`CancellationToken`](@ref) to override it, or `nothing` to make the wait
non-cancellable.
"""
wait(c::GenericCondition; first::Bool=false, waitee=c,
     cancel::CancelTokenArg=DEFAULT_CANCEL) =
    wait(c, check_cancel_arg(cancel); first, waitee)

function wait(c::GenericCondition, tok::MaybeToken; first::Bool=false, waitee=c,
              min_severity::UInt8=0x00)
    ct = current_task()
    assert_havelock(c)
    src = tok === nothing ? nothing : tok.source
    # entry check: throw before enqueueing anything (skipped for teardown
    # waits that re-park after acknowledging a severity)
    src === nothing || min_severity != 0x00 || checkcancel(src)
    w = _wait2(c, ct, first; waitee)
    if src !== nothing && !register_cancellation!(src, w; min_severity=min_severity)
        # The governing source is already cancelled: withdraw the
        # registration we just armed and deliver the cancellation ourselves.
        @atomicreplace ct.waiting_on w => nothing
        list_deletefirst!(ILLRef(c.waitq, waitee), w)
        _deliver_refused_cancellation(src)
    end
    token = unlockall(c.lock)
    ret = try
        wait()
    catch
        # We were resumed without a wake having been delivered through our
        # registration: either an interrupter claimed it (leaving the entry
        # linked for us to clean up), or we got a raw `throwto`. Disarm the
        # registration first - before the relock below can register a new
        # wait - then unlink our entry (a no-op if a `notify` already popped
        # and dropped it).
        @atomicreplace ct.waiting_on w => nothing
        # Our wake may already have been claimed and turned into a workqueue
        # enqueue that this unwind will never consume - drop that too, so a
        # later `schedule` of this task does not find it spuriously queued.
        q = ct.queue
        q === nothing || list_deletefirst!(q::StickyWorkqueue, ct)
        # Do not reuse `w` while relocking: a notifier may have popped this
        # stale entry without scheduling us and may still retain its identity
        # for the wake-claim CAS. If relocking does not need to park and cache
        # a replacement, restore `w` once cleanup under the old lock makes it
        # safe to reuse again.
        was_cached = ct.cached_wait_entry === w
        was_cached && (ct.cached_wait_entry = nothing)
        relockall(c.lock, token)
        src === nothing || unregister_cancellation!(src, w)
        list_deletefirst!(ILLRef(c.waitq, waitee), w)
        if was_cached && ct.cached_wait_entry === nothing
            ct.cached_wait_entry = w
        end
        rethrow()
    end
    # a normal wake implies our claim was won and our waitee-queue entry was
    # already unlinked by the notifier; the source registration is ours to
    # drop (and a wake that arrived as a value from something other than
    # this waitee's notify leaves the entry to us - the delete below is a
    # no-op in the common case)
    relockall(c.lock, token)
    src === nothing || unregister_cancellation!(src, w)
    list_deletefirst!(ILLRef(c.waitq, waitee), w)
    return ret
end

"""
    notify(condition, val=nothing; all=true, error=false)

Wake up tasks waiting for a condition, passing them `val`. If `all` is `true` (the default),
all waiting tasks are woken, otherwise only one is. If `error` is `true`, the passed value
is raised as an exception in the woken tasks.

Return the count of tasks woken up. Return 0 if no tasks are waiting on `condition`.
"""
@constprop :none notify(c::GenericCondition, @nospecialize(arg = nothing); all=true, error=false) = notify(c, arg, all, error)
function notify(c::GenericCondition, @nospecialize(arg), all, error)
    assert_havelock(c)
    cnt = 0
    while !isempty(c.waitq)
        w = popfirst!(waitqueue(c))
        # An entry whose wake was already claimed by an interrupter does not
        # count as woken: drop it and continue to the next waiter (the
        # interrupted task resumes via whatever its claimer scheduled and
        # will find its entry already unlinked).
        t = w.task
        if !(t isa Task && claim_wait(t, w))
            continue
        end
        schedule(t, arg, error=error)
        cnt += 1
        all || break
    end
    return cnt
end

notify_error(c::GenericCondition, err) = notify(c, err, true, true)

"""
    isempty(condition)

Return `true` if no tasks are waiting on the condition, `false` otherwise.
"""
function isempty(c::GenericCondition)
    for w in c.waitq
        t = w.task
        t isa Task && (@atomic t.waiting_on) === w && return false
    end
    return true
end


# default (Julia v1.0) is currently single-threaded
# (although it uses MT-safe versions, when possible)
"""
    Condition()

Create an edge-triggered event source that tasks can wait for. Tasks that call [`wait`](@ref) on a
`Condition` are suspended and queued. Tasks are woken up when [`notify`](@ref) is later called on
the `Condition`. Waiting on a condition can return a value or raise an error if the optional arguments
of [`notify`](@ref) are used. Edge triggering means that only tasks waiting at the time [`notify`](@ref)
is called can be woken up. For level-triggered notifications, you must keep extra state to keep
track of whether a notification has happened. The [`Channel`](@ref) and [`Threads.Event`](@ref) types do
this, and can be used for level-triggered events.

This object is NOT thread-safe. See [`Threads.Condition`](@ref) for a thread-safe version.
"""
const Condition = GenericCondition{AlwaysLockedST}

show(io::IO, ::Condition) = print(io, Condition, "()")

lock(c::GenericCondition{AlwaysLockedST}) =
    throw(ArgumentError("`Condition` is not thread-safe. Please use `Threads.Condition` instead for multi-threaded code."))
unlock(c::GenericCondition{AlwaysLockedST}) =
    throw(ArgumentError("`Condition` is not thread-safe. Please use `Threads.Condition` instead for multi-threaded code."))
