# This file is a part of Julia. License is MIT: https://julialang.org/license

# Cancellation tests that require multiple threads (run from cancellation.jl
# in a subprocess with -t2, so they are exercised even when the test driver
# itself is single-threaded).

using Test
using Libdl
using Base: cancel!, CancellationRequest, CancellationToken, CancellationTokenSource

@assert Threads.nthreads() > 1

# Start `f` as a task governed by a fresh cancellation source (non-sticky,
# explicitly on the default pool - a compute-bound victim must not land on
# the interactive/io thread).
function cancellable_spawn(f)
    src = CancellationTokenSource()
    t = Base.with_cancel_token(() -> Threads.@spawn(f()), CancellationToken(src))
    return t, src
end

@noinline function find_collatz_counterexample_inner()
    collatz(n) = (n & 1) == 1 ? (3n + 1) : (n ÷ 2)
    i = 1
    while true
        j = i
        while true
            j = collatz(j)
            j == 1 && break
            j == i && return j
        end
        i += 1
    end
end
function find_collatz_counterexample2()
    # A single cancellation point at function entry; interrupting the inner
    # (checkless) loop requires the reset_ctx mechanism.
    Base.@cancel_check
    return find_collatz_counterexample_inner()
end

@testset "async interruption of checkless loops (reset_ctx)" begin
    t, src = cancellable_spawn(find_collatz_counterexample2)
    sleep(0.5)
    cancel!(src)
    sleep(0.5)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
end

@testset "task abandonment wakes waiters" begin
    started = Base.Event()
    victim = Threads.@spawn begin
        notify(started)
        x = Ref(1.0)
        while true
            x[] = x[] * 1.0000001 + 0.1
        end
    end
    wait(started)
    watcher = @async wait(victim)
    sleep(0.5) # make sure the victim is actually spinning on its thread
    rescue = Task(() -> (while true; wait(); end))
    rescue.sticky = false
    Base.unsafe_abandon!(victim, rescue)
    @test timedwait(() -> istaskdone(victim), 5.0) == :ok
    @test victim.state === :abandoned
    @test istaskfailed(victim)
    # the watcher must be woken (abandoned tasks skip the regular
    # completion path)
    @test timedwait(() -> istaskdone(watcher), 5.0) == :ok
    @test_throws TaskFailedException fetch(watcher)
end

@testset "ABANDON_ALL freezes a running task" begin
    started = Base.Event()
    # The victim has no cancellation points in its loop: only freezing can
    # stop it promptly. The task-start check published its binding to the
    # scope's token, which is how the cancellation walk finds it.
    victim, src = cancellable_spawn() do
        notify(started)
        x = Ref(1.0)
        while x[] > 0
            x[] = x[] * 1.0000001 + 0.1
        end
    end
    wait(started)
    sleep(0.5) # make sure it is spinning on its thread
    @test cancel!(src, Base.CANCEL_REQUEST_ABANDON_ALL)
    @test timedwait(() -> istaskdone(victim), 10.0) == :ok
    @test victim.state === :abandoned
    @test istaskfailed(victim)
end

@testset "cancel storm" begin
    # Hammer cancellation against tasks in every wait state, from a tight
    # loop with concurrent GC pressure. This is a regression test for the
    # class of races that only show up under load (e.g. an uninitialized or
    # stale reset_ctx being consumed by the cancellation signal).
    function polling_loop()
        x = 1
        while true
            Base.@cancel_check
            x += 1
        end
    end
    function checkless_loop()
        Base.@cancel_check
        # No cancellation points in the loop below: only asynchronous
        # delivery (reset_ctx) can interrupt it. Bound the iteration count:
        # a safepoint-free spin loop blocks GC's stop-the-world, and if GC
        # wins the race against cancel! (the canceller parks at a safepoint
        # before it manages to deliver), an unbounded loop would wedge the
        # process. With the bound, that race degrades to a transient stall.
        x = Ref(1.0)
        i = 0
        while x[] > 0 && i < 2_000_000_000
            x[] = x[] * 1.0000001 + 0.1
            i += 1
        end
    end
    stop = Ref(false)
    garbage = @async begin # allocation/GC pressure
        while !stop[]
            zeros(UInt8, 1 << 20)
            yield()
        end
    end
    held = ReentrantLock()
    lock(held)
    function make_victims()
        src = CancellationTokenSource()
        late = Ref{Task}()
        ts = Base.with_cancel_token(CancellationToken(src)) do
            late[] = Task(() -> nothing) # scheduled only after the cancellation
            Task[
                @async(sleep(1000)),            # timer wait
                @async(take!(Channel{Int}(0))), # condition wait
                @async(lock(held)),             # lock wait (the test holds `held`)
                Threads.@spawn(polling_loop()),
                Threads.@spawn(checkless_loop()),
                late[],
            ]
        end
        return ts, src, late[]
    end
    deadline = time() + 10
    rounds = 0
    failures = 0
    while time() < deadline
        ts, src, late = make_victims()
        # vary the delivery window: sometimes cancel immediately (tasks not
        # yet started/parked), sometimes after they have parked/started
        # spinning
        rounds % 2 == 0 && sleep(0.05)
        cancel!(src)
        cancel!(src) # double-cancellation must be harmless
        Base.redeliver!(src) # explicit redelivery must be harmless too
        # a task scheduled into an already-cancelled scope dies at start
        schedule(late)
        for t in ts
            if timedwait(() -> istaskdone(t), 20.0) !== :ok
                failures += 1
                @error "cancelled task failed to complete" t t.state
            end
        end
        rounds += 1
    end
    stop[] = true
    wait(garbage)
    unlock(held)
    @test failures == 0
    @test rounds > 0
end
