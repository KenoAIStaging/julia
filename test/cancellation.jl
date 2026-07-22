# This file is a part of Julia. License is MIT: https://julialang.org/license

using Base: cancel!, CancellationRequest, CancellationToken, CancellationTokenSource,
    CANCEL_REQUEST_SAFE, CANCEL_REQUEST_ABANDON_EXTERNAL, CANCEL_REQUEST_ABANDON_ALL

# Start `f` as an @async-style (sticky, co-scheduled) task governed by a
# fresh cancellation source; returns (task, source).
function cancellable(f)
    src = CancellationTokenSource()
    t = Base.with_cancel_token(() -> @async(f()), CancellationToken(src))
    return t, src
end

# Threads.@spawn-style variant (non-sticky, explicitly on the default pool -
# a compute-bound victim must not land on the interactive/io thread).
function cancellable_spawn(f)
    src = CancellationTokenSource()
    t = Base.with_cancel_token(() -> Threads.@spawn(f()), CancellationToken(src))
    return t, src
end

# whether `t` is parked (its wait registration is enqueued on some waitee)
is_parked(t::Task) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue !== nothing)
parked_on(t::Task, @nospecialize(x)) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue === x)

const collatz_code = quote
    collatz(n) = (n & 1) == 1 ? (3n + 1) : (n ÷ 2)
    function find_collatz_counterexample()
        i = 1
        while true
            j = i
            while true
                Base.@cancel_check
                j = collatz(j)
                j == 1 && break
                j == i && error("$j is a collatz counterexample")
            end
            i += 1
        end
    end
    @noinline function find_collatz_counterexample_inner()
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
end
eval(collatz_code)

# wait a little, so cancellation targets are (most likely) started and parked
spin(n=4) = for _ in 1:n; yield(); end

@testset "cancellation token tree semantics" begin
    # cancel! marks the whole subtree, level-triggered
    root = CancellationTokenSource()
    child = CancellationTokenSource(CancellationToken(root))
    grandchild = CancellationTokenSource(CancellationToken(child))
    @test !Base.iscancelled(grandchild)
    @test cancel!(root)
    @test Base.iscancelled(root) && Base.iscancelled(child) && Base.iscancelled(grandchild)
    @test !cancel!(root) # idempotent at the same severity

    # a source attached under an already-cancelled parent is born cancelled
    late = CancellationTokenSource(CancellationToken(child))
    @test Base.iscancelled(late)
    @test Base.cancel_severity(late) === CANCEL_REQUEST_SAFE

    # escalation is monotonic and propagates down
    @test cancel!(root, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test Base.cancel_severity(grandchild) === CANCEL_REQUEST_ABANDON_EXTERNAL
    @test !cancel!(grandchild, CANCEL_REQUEST_SAFE) # never de-escalates

    # cancellation is uniformly level-triggered: after catching the request,
    # unshielded waits under the cancelled scope keep throwing; shielded
    # cleanup proceeds
    src = CancellationTokenSource()
    phase = Ref{Any}(:init)
    t = Base.with_cancel_token(CancellationToken(src)) do
        @async try
            sleep(1000)
        catch e
            e isa CancellationRequest || rethrow()
            phase[] = :caught
            rethrew = try
                sleep(1000)
                false
            catch e2
                e2 isa CancellationRequest
            end
            sleep(0.01; cancel=nothing) # shielded cleanup is permitted
            phase[] = rethrew ? :done : :no_retrigger
        end
    end
    spin()
    cancel!(src)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test phase[] === :done

    # an internal teardown re-park (min_severity) is woken only by escalation
    srcm = CancellationTokenSource()
    cancel!(srcm)
    inner = @async sleep(5)
    tm = @async Base._wait(inner, CancellationToken(srcm); min_severity=0x01)
    spin()
    cancel!(srcm, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test_throws TaskFailedException wait(tm)
    @test tm.result isa CancellationRequest

    # the current scoped token is discoverable
    tok = CancellationToken(CancellationTokenSource())
    @test Base.with_cancel_token(Base.cancellation_token, tok) === tok
end

@testset "cancellation of waiting tasks" begin
    # A task spawned under an already-cancelled scope starts but observes the
    # cancellation before running any user code
    src = CancellationTokenSource()
    body_ran = Ref(false)
    t = Base.with_cancel_token(CancellationToken(src)) do
        @task (body_ran[] = true)
    end
    @test cancel!(src)
    schedule(t)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test istaskfailed(t)
    @test t.result isa CancellationRequest
    @test !body_ran[]
    @test_throws TaskFailedException wait(t)

    # Cancellation of `sleep`
    t, src = cancellable(() -> sleep(1000))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest

    # After catching (acknowledging) the request, cleanup code may still park
    t2, src2 = cancellable() do
        try
            sleep(1000)
        catch e
            e isa CancellationRequest || rethrow()
            sleep(0.01; cancel=nothing) # shielded: parking for cleanup is permitted
            return :cleanup_ok
        end
    end
    spin()
    cancel!(src2)
    @test fetch(t2) === :cleanup_ok

    # Cancellation of a task blocked on a Channel
    c = Channel{Int}(0)
    t, src = cancellable(() -> take!(c))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # The channel remains usable
    t2 = @async take!(c)
    put!(c, 7)
    @test fetch(t2) == 7

    # Cancelling a scope reaches a task waiting on another task; the waited-on
    # task (in the same scope) is cancelled through the same tree
    local t_in
    t, src = cancellable() do
        t_in = @async sleep(1000)
        wait(t_in)
    end
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test timedwait(() -> istaskdone(t_in), 10.0) == :ok
    @test istaskfailed(t_in)

    # ... but a task waited on from a *different* scope is unaffected by the
    # waiter's cancellation
    t_out = @async sleep(5)
    t, src = cancellable(() -> wait(t_out))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test !istaskdone(t_out)
    wait(t_out)
    @test istaskdone(t_out) && !istaskfailed(t_out)
end

@testset "cancellation of lock and condition waits" begin
    # Task blocked in lock(::ReentrantLock)
    lk = ReentrantLock()
    lock(lk)
    t, src = cancellable(() -> lock(lk))
    spin()
    # let it spin through the fast path and park
    @test timedwait(() -> is_parked(t), 5.0) == :ok
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # the lock remains functional
    unlock(lk)
    @test trylock(lk)
    unlock(lk)
    t2 = @async (lock(lk); unlock(lk); true)
    @test fetch(t2)

    # Task blocked in put! on a full channel
    c = Channel{Int}(1)
    put!(c, 1)
    t, src = cancellable(() -> put!(c, 2))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @test take!(c) == 1
    put!(c, 3) # channel remains functional
    @test take!(c) == 3

    # Task blocked in wait(::Threads.Condition)
    cond = Threads.Condition()
    t, src = cancellable(() -> @lock cond wait(cond))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @lock cond notify(cond) # still functional (no waiters)

    # Task blocked in wait(::Base.Process); the process itself keeps running
    p = run(`sleep 1000`; wait=false)
    t, src = cancellable(() -> wait(p))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @test process_running(p)
    kill(p); wait(p)

    # Task blocked in waitany; the awaited tasks live in different scopes and
    # remain unaffected by the waiter's cancellation
    t1, src1 = cancellable(() -> sleep(1000))
    t2, src2 = cancellable(() -> sleep(1000))
    t, src = cancellable(() -> waitany([t1, t2]))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @test !istaskdone(t1) && !istaskdone(t2)
    # their own scopes' cancellation reaches them
    cancel!(src1); cancel!(src2)
    @test_throws TaskFailedException wait(t1)
    @test_throws TaskFailedException wait(t2)
end

@testset "cancellation of stdlib waits (Sockets, FileWatching, Semaphore)" begin
    # Base.Semaphore: a cancelled acquire does not leak a permit
    sem = Base.Semaphore(1)
    Base.acquire(sem)
    t, src = cancellable(() -> Base.acquire(sem))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    Base.release(sem)
    Base.acquire(sem) # the permit is still available
    Base.release(sem)

    # Sockets.accept
    Sockets = Base.require(Base.PkgId(Base.UUID("6462fe0b-24de-5631-8697-dd941f90decc"), "Sockets"))
    port, server = Sockets.listenany(Sockets.localhost, 0)
    t, src = cancellable(() -> Sockets.accept(server))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # the server keeps accepting afterwards
    t2 = @async Sockets.accept(server)
    sock = Sockets.connect(Sockets.localhost, port)
    @test fetch(t2) isa Sockets.TCPSocket
    close(sock); close(server)

    # FileWatching: fd polling and file watching
    FileWatching = Base.require(Base.PkgId(Base.UUID("7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"), "FileWatching"))
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    fd = Base._fd(p.out)
    t, src = cancellable(() -> FileWatching.wait(fd; readable=true)) # nothing is ever written
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    close(p)

    path = tempname()
    touch(path)
    t, src = cancellable(() -> FileWatching.watch_file(path, 100.0)) # the file never changes
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    rm(path)

    # Distributed: a (local) never-fulfilled Future wait; remote waits go
    # through the same channel-based wait path on the caller side
    Distributed = Base.require(Base.PkgId(Base.UUID("8ba89e20-285c-5b6f-9357-94700520ee1b"), "Distributed"))
    fut = Distributed.Future()
    t, src = cancellable(() -> fetch(fut))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    put!(fut, 1) # the future remains usable
    @test fetch(fut) == 1
end

@testset "explicit cancel keyword arguments" begin
    Sockets = Base.require(Base.PkgId(Base.UUID("6462fe0b-24de-5631-8697-dd941f90decc"), "Sockets"))
    FileWatching = Base.require(Base.PkgId(Base.UUID("7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"), "FileWatching"))
    cancelled_src = CancellationTokenSource()
    cancel!(cancelled_src)
    ctok = CancellationToken(cancelled_src)

    # a pre-cancelled token throws at entry, before any side effect
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    @test_throws CancellationRequest read(p.out, 10; cancel=ctok)
    @test_throws CancellationRequest read(p.out; cancel=ctok)
    @test_throws CancellationRequest read(p.out, String; cancel=ctok)
    @test_throws CancellationRequest read(p.out, UInt8; cancel=ctok)
    @test_throws CancellationRequest read!(p.out, zeros(UInt8, 4); cancel=ctok)
    @test_throws CancellationRequest readbytes!(p.out, zeros(UInt8, 4); cancel=ctok)
    @test_throws CancellationRequest readline(p.out; cancel=ctok)
    @test_throws CancellationRequest readuntil(p.out, 0x0a; cancel=ctok)
    @test_throws CancellationRequest readavailable(p.out; cancel=ctok)
    @test_throws CancellationRequest eof(p.out; cancel=ctok)
    @test_throws CancellationRequest write(p.in, zeros(UInt8, 8); cancel=ctok)
    @test_throws CancellationRequest write(p.in, "hello"; cancel=ctok)
    @test_throws CancellationRequest write(p.in, "a", "b"; cancel=ctok)
    @test_throws CancellationRequest flush(p.in; cancel=ctok)
    @test_throws CancellationRequest sleep(10; cancel=ctok)
    @test_throws CancellationRequest wait(Timer(10); cancel=ctok)
    @test_throws CancellationRequest run(`sleep 5`; cancel=ctok)
    @test_throws CancellationRequest success(`sleep 5`; cancel=ctok)
    @test_throws CancellationRequest read(`sleep 5`; cancel=ctok)
    @test_throws CancellationRequest readchomp(`sleep 5`; cancel=ctok)
    @test_throws CancellationRequest Sockets.getalladdrinfo("localhost"; cancel=ctok)
    @test_throws CancellationRequest Sockets.getaddrinfo("localhost"; cancel=ctok)
    @test_throws CancellationRequest Sockets.getnameinfo(Sockets.localhost; cancel=ctok)
    @test_throws CancellationRequest FileWatching.watch_file(tempdir(), 5.0; cancel=ctok)
    @test_throws CancellationRequest FileWatching.poll_fd(Base._fd(p.out), 5.0; readable=true, cancel=ctok)

    # `cancel = nothing` shadows an (already cancelled) outer scope
    write(p.in, "ab\n")
    Base.with_cancel_token(ctok) do
        @test read(p.out, 2; cancel=nothing) == b"ab"
    end
    close(p)

    # live cancellation through an explicit token: blocked read
    p2 = Pipe()
    Base.link_pipe!(p2, reader_supports_async=true, writer_supports_async=true)
    src = CancellationTokenSource()
    t = @async read(p2.out, 10; cancel=CancellationToken(src))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    close(p2)

    # live cancellation: blocked write
    p3 = Pipe()
    Base.link_pipe!(p3, reader_supports_async=true, writer_supports_async=true)
    src3 = CancellationTokenSource()
    big = zeros(UInt8, 200_000_000)
    t3 = @async write(p3.in, big; cancel=CancellationToken(src3))
    sleep(0.5)
    cancel!(src3)
    @test_throws TaskFailedException wait(t3)
    @test t3.result isa CancellationRequest
    close(p3)

    # live cancellation: Sockets.accept and recv with explicit tokens
    port, server = Sockets.listenany(Sockets.localhost, 0)
    src4 = CancellationTokenSource()
    t4 = @async Sockets.accept(server; cancel=CancellationToken(src4))
    spin()
    cancel!(src4)
    @test_throws TaskFailedException wait(t4)
    @test t4.result isa CancellationRequest
    close(server)

    udp = Sockets.UDPSocket()
    Sockets.bind(udp, Sockets.localhost, 0)
    src5 = CancellationTokenSource()
    t5 = @async Sockets.recv(udp; cancel=CancellationToken(src5))
    spin()
    cancel!(src5)
    @test_throws TaskFailedException wait(t5)
    @test t5.result isa CancellationRequest
    close(udp)

    # live cancellation: FileWatching.watch_file with an explicit token
    path = tempname()
    touch(path)
    src6 = CancellationTokenSource()
    t6 = @async FileWatching.watch_file(path, 100.0; cancel=CancellationToken(src6))
    spin()
    cancel!(src6)
    @test_throws TaskFailedException wait(t6)
    @test t6.result isa CancellationRequest
    rm(path)

    # live cancellation: run with an explicit token; the child process is
    # not reaped by the cancelled wait
    src7 = CancellationTokenSource()
    t7 = @async run(`sleep 5`; cancel=CancellationToken(src7))
    sleep(0.5)
    cancel!(src7)
    @test_throws TaskFailedException wait(t7)
    @test t7.result isa CancellationRequest
end

@testset "cancellation of computing tasks" begin
    # The victims never yield, so a second thread must run the canceller.
    # (Signal-side delivery that also covers -t1 arrives with the ^C
    # machinery later in this series; the checkless reset_ctx variant runs
    # in the -t2 exec subprocess.)
    if Threads.nthreads() > 1
        # Polling cancellation via @cancel_check
        t, src = cancellable_spawn(find_collatz_counterexample)
        sleep(0.2)
        cancel!(src)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest
    end
end

@testset "structured cancellation of @sync" begin
    t, src = cancellable() do
        @sync begin
            @async sleep(1000)
            @async sleep(1000)
        end
    end
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CompositeException
    @test length(t.result.exceptions) == 2
end

@testset "structured cancellation of Experimental.@sync" begin
    t1 = Ref{Task}(); t2 = Ref{Task}()
    t, src = cancellable() do
        Base.Experimental.@sync begin
            t1[] = @async sleep(1000)
            t2[] = @async sleep(1000)
        end
    end
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # cancellation propagated to the children
    @test timedwait(() -> istaskdone(t1[]) && istaskdone(t2[]), 10.0) == :ok
    @test istaskfailed(t1[]) && istaskfailed(t2[])
end

@testset "cancellation of blocked stream writes" begin
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    try
        # A write far exceeding the OS pipe buffer blocks until cancelled
        big = zeros(UInt8, 200_000_000)
        t, src = cancellable(() -> write(p, big))
        sleep(0.5)
        @test parked_on(t, p.in)
        cancel!(src)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest
    finally
        close(p)
    end
end

@testset "cancellation of closewrite (shutdown) waits" begin
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    try
        # A blocked write keeps the shutdown request (which queues behind it)
        # from completing; the closewrite wait must still be interruptible.
        big = zeros(UInt8, 200_000_000)
        tw, srcw = cancellable(() -> write(p, big))
        sleep(0.5)
        @test parked_on(tw, p.in)
        ts, srcs = cancellable(() -> closewrite(p.in))
        @test timedwait(() -> parked_on(ts, p.in), 5.0) == :ok
        cancel!(srcs)
        @test_throws TaskFailedException wait(ts)
        @test ts.result isa CancellationRequest
        cancel!(srcw)
        @test_throws TaskFailedException wait(tw)
    finally
        close(p)
    end
end

@testset "unfriendly cancellation modes" begin
    # Acknowledgment preserves the request's severity.
    seen = Ref{Any}(nothing)
    t, src = cancellable() do
        try
            sleep(1000)
        catch e
            seen[] = (e, Base.ambient_cancel_severity(), Base.abandoning_external_waits())
            rethrow()
        end
    end
    spin()
    cancel!(src, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    e, sev, abandoning = seen[]
    @test e === CANCEL_REQUEST_ABANDON_EXTERNAL
    @test sev === CANCEL_REQUEST_ABANDON_EXTERNAL
    @test abandoning

    # SAFE acknowledgments report SAFE severity and permit external waits.
    seen2 = Ref{Any}(nothing)
    t2, src2 = cancellable() do
        try
            sleep(1000)
        catch
            seen2[] = (Base.ambient_cancel_severity(), Base.abandoning_external_waits())
            rethrow()
        end
    end
    spin()
    cancel!(src2)
    @test timedwait(() -> istaskdone(t2), 10.0) == :ok
    @test seen2[] === (CANCEL_REQUEST_SAFE, false)

    # ABANDON_ALL freezes a parked task immediately: no unwind, no cleanup.
    cleanup_ran = Ref(false)
    t3, src3 = cancellable() do
        try
            sleep(1000)
        finally
            cleanup_ran[] = true
        end
    end
    spin()
    @test cancel!(src3, CANCEL_REQUEST_ABANDON_ALL)
    @test istaskdone(t3)
    @test t3.state === :abandoned
    @test istaskfailed(t3)
    @test !cleanup_ran[]
    @test_throws TaskFailedException wait(t3)

    # A task spawned into an ABANDON_ALL-cancelled scope never runs its body.
    src4 = CancellationTokenSource()
    body_ran = Ref(false)
    t4 = Base.with_cancel_token(() -> @task(body_ran[] = true), CancellationToken(src4))
    @test cancel!(src4, CANCEL_REQUEST_ABANDON_ALL)
    schedule(t4)
    @test timedwait(() -> istaskdone(t4), 10.0) == :ok
    @test !body_ran[]

    # ABANDON_EXTERNAL interrupts a blocked stream write without waiting for
    # the write's cancellation to complete.
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    try
        big = zeros(UInt8, 200_000_000)
        tw, srcw = cancellable(() -> write(p, big))
        spin()
        cancel!(srcw, CANCEL_REQUEST_ABANDON_EXTERNAL)
        @test timedwait(() -> istaskdone(tw), 10.0) == :ok
        @test istaskfailed(tw)
        @test tw.result === CANCEL_REQUEST_ABANDON_EXTERNAL
    finally
        close(p)
    end
end

@testset "^C escalation severity ladder" begin
    # Episode classification for the ^C escalation ladder.
    src = CancellationTokenSource()
    @test Base.sigint_active_severity(src) === nothing
    @test cancel!(src)
    @test Base.sigint_active_severity(src) === CANCEL_REQUEST_SAFE
    @test cancel!(src, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test Base.sigint_active_severity(src) === CANCEL_REQUEST_ABANDON_EXTERNAL
    @test cancel!(src, CANCEL_REQUEST_ABANDON_ALL)
    @test Base.sigint_active_severity(src) === CANCEL_REQUEST_ABANDON_ALL
    # severities never de-escalate
    @test !cancel!(src, CANCEL_REQUEST_SAFE)
    @test Base.sigint_active_severity(src) === CANCEL_REQUEST_ABANDON_ALL
end

@testset "unfriendly cancellation of Experimental.@sync" begin
    # ABANDON_EXTERNAL propagates through the token tree to the children.
    t1 = Ref{Task}(); t2 = Ref{Task}()
    t, src = cancellable() do
        Base.Experimental.@sync begin
            t1[] = @async sleep(1000)
            t2[] = @async sleep(1000)
        end
    end
    spin()
    cancel!(src, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test_throws TaskFailedException wait(t)
    @test timedwait(() -> istaskdone(t1[]) && istaskdone(t2[]), 10.0) == :ok
    @test istaskfailed(t1[]) && istaskfailed(t2[])

    # ABANDON_ALL freezes the parent and the children alike (they are all
    # parked under the cancelled subtree).
    t3 = Ref{Task}()
    tp, srcp = cancellable() do
        Base.Experimental.@sync begin
            t3[] = @async sleep(1000)
        end
    end
    spin()
    @test cancel!(srcp, CANCEL_REQUEST_ABANDON_ALL)
    @test tp.state === :abandoned
    @test timedwait(() -> istaskdone(t3[]), 10.0) == :ok
    @test t3[].state === :abandoned
end

@testset "structured cancellation of Experimental.@sync" begin
    t1 = Ref{Task}(); t2 = Ref{Task}()
    t, src = cancellable() do
        Base.Experimental.@sync begin
            t1[] = @async sleep(1000)
            t2[] = @async sleep(1000)
        end
    end
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # cancellation propagated to the children
    @test timedwait(() -> istaskdone(t1[]) && istaskdone(t2[]), 10.0) == :ok
    @test istaskfailed(t1[]) && istaskfailed(t2[])
end

@testset "threaded cancellation (subprocess with -t2)" begin
    cmd = `$(Base.julia_cmd()) --depwarn=error --startup-file=no --threads=2 $(joinpath(@__DIR__, "cancellation_exec.jl"))`
    p = run(pipeline(cmd, stdout=stdout, stderr=stderr), wait=false)
    # A cancellation-delivery regression can wedge the child completely (a
    # surviving spin loop blocks GC's stop-the-world, which also blocks all
    # signal processing), in which case not even SIGTERM gets through.
    # SIGKILL it rather than hanging the test suite.
    if timedwait(() -> process_exited(p), 240.0) !== :ok
        kill(p, Base.SIGKILL)
    end
    wait(p)
    @test success(p)
end

@testset "^C" begin
    function run_with_sigint(code::String, delays; forcekill::Bool=false)
        out = Pipe()
        p = run(pipeline(`$(Base.julia_cmd()) --startup-file=no -e $code`, stdout=out, stderr=out), wait=false)
        close(out.in)
        reader = @async read(out, String)
        killer = @async begin
            for d in delays
                sleep(d)
                process_running(p) && kill(p, Base.SIGINT)
            end
            if forcekill
                # e.g. an abandoned script has nothing left to run and idles
                sleep(3)
                process_running(p) && kill(p, Base.SIGKILL)
            end
        end
        wait(p)
        wait(killer)
        return fetch(reader), p
    end

    # Catching ^C in a script: continuing requires re-arming a fresh ^C
    # epoch (the script's cancelled scope stays cancelled otherwise)
    output, p = run_with_sigint("""
        try
            sleep(100)
            println("FAIL: not cancelled")
        catch e
            Base.with_cancel_token(Base.sigint_new_episode!()) do
                println("caught: ", typeof(e))
                println("continued")
                sleep(0.1) # cancellable operations work again
            end
        end
    """, [1.0])
    @test occursin("caught: Base.CancellationRequest", output)
    @test occursin("continued", output)
    @test p.exitcode == 0

    # Uncaught ^C produces a proper error report
    output, p = run_with_sigint("sleep(100)", [1.0])
    @test occursin("CancellationRequest: Safe Cancellation (CANCEL_REQUEST_SAFE)", output)
    @test p.exitcode == 1

    # TODO(port): the @sync compute-spinner ^C test is deferred while the port
    # of #60281 proceeds: it needs signal-thread episode marking + bound-source
    # propagation (the -t1 child's listener task starves behind the compiled
    # spinner). It is restored by the commits that port that machinery.

    # Escalation: an unresponsive process warns after 1s, and a second ^C
    # abandons the stuck task; with the interactive evaluator gone, the
    # process exits like an uncaught ^C
    output, p = run_with_sigint("""
        x = Ref(1.0)
        while true
            x[] = x[] * 1.0000001 + 0.1
        end
        """, [1.0, 2.5]; forcekill=true)
    @test occursin("failed to acknowledge SIGINT", output)
    @test occursin("Abandoning current task", output)
    @test p.exitcode == 128 + 2
end

# TODO(port): the interactive pty ^C escalation-ladder testset is deferred:
# reliable rung escalation requires the standing-offer/generation semantics
# ported later in the series (a press must not invalidate the offer it
# accepts), and the abandonment announcement wording it expects arrives with
# the same arc. Restored by the commits that port that machinery; content
# preserved in the port notes.
