# This file is a part of Julia. License is MIT: https://julialang.org/license

using Base: cancel!, CancellationRequest, CancellationToken, CancellationTokenSource,
    CANCEL_REQUEST_SAFE, CANCEL_REQUEST_ABANDON_EXTERNAL, CANCEL_REQUEST_ABANDON_ALL,
    CANCEL_TOKEN
using Base.ScopedValues: with, ScopedValue

# Threads.@spawn-style cancellable task (non-sticky, explicitly on the
# default pool - a compute-bound victim must not land on the interactive/io
# thread); returns (task, source).
function cancellable_spawn(f)
    src = CancellationTokenSource()
    t = with(() -> Threads.@spawn(f()), CANCEL_TOKEN => CancellationToken(src))
    return t, src
end

@testset "cancellation token graph semantics" begin
    # cancel! marks all descendants, level-triggered
    root = CancellationTokenSource()
    child = CancellationTokenSource(CancellationToken(root))
    grandchild = CancellationTokenSource(CancellationToken(child))
    @test !Base.iscancelled(grandchild)
    @test Base.cancel_severity(grandchild) === nothing
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
    @test Base.cancel_severity(grandchild) === CANCEL_REQUEST_ABANDON_EXTERNAL

    # invalid severities are rejected
    @test_throws ArgumentError cancel!(CancellationTokenSource(), CancellationRequest(0x2))
    @test_throws ArgumentError cancel!(CancellationTokenSource(), CancellationRequest(0x7f))

    # walk a source's (weak, intrusive) child list
    function live_children(src::CancellationTokenSource)
        kids = CancellationTokenSource[]
        c = @atomic src.child_head
        while c !== nothing
            c = c::CancellationTokenSource
            push!(kids, c)
            c = Base._cancel_next_child(src, c)
        end
        return kids
    end

    # children are held weakly: a child that becomes unreachable is spliced
    # out of its parents' child lists by the GC, while an escaped token
    # keeps its source attached (cancellation still reaches whoever can
    # observe it)
    @noinline function make_children(root)
        CancellationTokenSource(CancellationToken(root)) # unreachable after return
        c = CancellationTokenSource(CancellationToken(root))
        return CancellationToken(c) # only the token escapes
    end
    root2 = CancellationTokenSource()
    kept = CancellationTokenSource(CancellationToken(root2))
    escaped_tok = make_children(root2)
    GC.gc() # splices the collected child out of root2's list
    cancel!(root2)
    @test Base.iscancelled(kept)
    @test Base.iscancelled(escaped_tok)
    kids = live_children(root2)
    @test length(kids) == 2 # kept + escaped; the dead child was spliced out
    @test kept in kids && escaped_tok.source in kids

    # linked sources: a source with several parents is cancelled by any of
    # them (the graph is a DAG, not just a tree)
    la = CancellationTokenSource()
    lb = CancellationTokenSource()
    linked = CancellationTokenSource(CancellationToken(la), CancellationToken(lb))
    @test linked.nparents == 2
    @test Base._cancel_parent(linked, 1) === la && Base._cancel_parent(linked, 2) === lb
    @test !Base.iscancelled(CancellationToken(linked))
    @test cancel!(lb)
    @test Base.iscancelled(CancellationToken(linked))
    @test !Base.iscancelled(CancellationToken(la))
    # escalation propagates through the other parent too
    @test cancel!(la, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test Base.cancel_severity(linked) === CANCEL_REQUEST_ABANDON_EXTERNAL

    # born cancelled at the highest severity among the parents
    lc = CancellationTokenSource()
    ld = CancellationTokenSource()
    cancel!(ld, CANCEL_REQUEST_ABANDON_EXTERNAL)
    born = CancellationTokenSource(CancellationToken(lc), CancellationToken(ld))
    @test Base.cancel_severity(born) === CANCEL_REQUEST_ABANDON_EXTERNAL

    # a diamond converges: the shared descendant is cancelled exactly once
    # from the root
    droot = CancellationTokenSource()
    dl = CancellationTokenSource(CancellationToken(droot))
    dr = CancellationTokenSource(CancellationToken(droot))
    dd = CancellationTokenSource(CancellationToken(dl), CancellationToken(dr))
    cancel!(droot)
    @test Base.iscancelled(dd)
    @test Base.cancel_severity(dd) === CANCEL_REQUEST_SAFE

    # duplicate parents collapse to the single-parent form
    dup = CancellationTokenSource(CancellationToken(droot), CancellationToken(droot))
    @test dup.nparents == 1
    @test Base._cancel_parent(dup, 1) === droot
    @test Base.iscancelled(CancellationToken(dup)) # born under the cancelled root

    # attachment and GC splicing keep the sibling lists consistent across
    # many children coming and going
    sroot = CancellationTokenSource()
    skeep = CancellationTokenSource[]
    for i in 1:1000
        c = CancellationTokenSource(CancellationToken(sroot))
        i % 7 == 0 && push!(skeep, c)
        i % 250 == 0 && GC.gc(false)
    end
    GC.gc()
    @test length(live_children(sroot)) >= length(skeep)
    cancel!(sroot)
    @test all(Base.iscancelled, skeep)

    # deep chains cancel without recursion depth issues
    deep_root = CancellationTokenSource()
    node = deep_root
    chain = CancellationTokenSource[]
    for _ in 1:50_000
        node = CancellationTokenSource(CancellationToken(node))
        push!(chain, node) # keep them alive
    end
    cancel!(deep_root)
    @test Base.iscancelled(chain[end])

    # the current scoped token is discoverable, and `=> nothing` scopes it out
    tok = CancellationToken(CancellationTokenSource())
    @test with(() -> CANCEL_TOKEN[], CANCEL_TOKEN => tok) === tok
    @test with(() -> CANCEL_TOKEN[], CANCEL_TOKEN => nothing) === nothing
    # an unrelated nested scope inherits the governing token
    inherited = with(CANCEL_TOKEN => tok) do
        with(() -> CANCEL_TOKEN[], ScopedValue(0) => 1)
    end
    @test inherited === tok
end

@testset "cancel! repairs partially-cancelled subgraphs" begin
    # Simulate a cancel! whose descendant walk never ran (e.g. the cancelling
    # task torn down mid-walk): the state is raised, but no child is.
    root = CancellationTokenSource()
    child = CancellationTokenSource(CancellationToken(root))
    @test Base._raise_state!(root, 0x1)
    @test !Base.iscancelled(child)
    # A repeated cancel! loses the state transition (returns false) but
    # must still perform the full walk itself.
    @test !cancel!(root)
    @test Base.iscancelled(child)
end

@testset "concurrent child construction is level-triggered" begin
    # A child constructed concurrently with cancel! must end up cancelled,
    # whichever side wins the race: either the walk sees it in the child
    # list, or its constructor observes the already-cancelled parent.
    nspawners = max(Threads.nthreads() - 1, 1)
    for trial in 1:20
        root = CancellationTokenSource()
        tok = CancellationToken(root)
        go = Threads.Event()
        tasks = map(1:nspawners) do _
            Threads.@spawn begin
                wait(go)
                kids = CancellationTokenSource[]
                for _ in 1:500
                    push!(kids, CancellationTokenSource(tok))
                end
                kids
            end
        end
        notify(go)
        cancel!(root)
        for t in tasks
            @test all(Base.iscancelled, fetch(t))
        end
    end
end

@testset "cancellation source GC with dying parents" begin
    # Parents dying in the same cycle as their children: the unlink pass
    # writes into the dead parents' memory, which the sweep must keep
    # valid through the cycle.
    for _ in 1:5
        for _ in 1:1000
            r = CancellationTokenSource()
            m = CancellationTokenSource(CancellationToken(r))
            CancellationTokenSource(CancellationToken(m))
        end
        GC.gc()
    end
    GC.gc()
    GC.gc() # pages now hold no sources: the sweep flag must clear, not pin them
    # big-object sources (many parents) take the deferred-free path
    let
        parents = [CancellationTokenSource() for _ in 1:100]
        cancel!(parents[1])
        big = CancellationTokenSource(map(CancellationToken, parents)...)
        @test Base.iscancelled(big)
        parents = nothing
        big = nothing
    end
    GC.gc()
    GC.gc()
    # a survivor amid heavy churn stays correctly linked throughout
    root = CancellationTokenSource()
    keep = CancellationTokenSource(CancellationToken(root))
    for _ in 1:10_000
        CancellationTokenSource(CancellationToken(root))
    end
    GC.gc()
    GC.gc()
    cancel!(root)
    @test Base.iscancelled(keep)
end

@testset "cancellation source memory accounting" begin
    a = CancellationTokenSource()
    b = CancellationTokenSource()
    # instances are variable-sized, so (like String or Memory) the type has
    # no definite size and inference must not fold an instance's sizeof
    @test_throws ErrorException Core.sizeof(CancellationTokenSource)
    @test Base.infer_return_type(Core.sizeof, Tuple{CancellationTokenSource}) == Int
    base = Core.sizeof(a)
    linksz = 3 * sizeof(Ptr{Cvoid})
    c2 = CancellationTokenSource(CancellationToken(a), CancellationToken(b))
    @test Core.sizeof(c2) == base + 2 * linksz
    # summarysize charges the link tail and the (strong) parents, but not
    # the (weak) children
    @test Base.summarysize(c2) == base + 2 * linksz + 2 * base
    @test Base.summarysize(c2; count=true) == 3
    @test Base.summarysize(a) == base # a's children are weak: c2 not charged
    # the hidden parent references go through the regular traversal policy
    one = CancellationTokenSource(CancellationToken(a))
    @test Base.summarysize(one; exclude=CancellationTokenSource) == base + linksz
    @test Base.summarysize(one; exclude=CancellationTokenSource, count=true) == 1
    # deep parent chains are traversed iteratively, not by recursion
    node = CancellationTokenSource()
    for _ in 1:100_000
        node = CancellationTokenSource(CancellationToken(node))
    end
    @test Base.summarysize(node) >= 100_001 * base + 100_000 * linksz
end

@testset "cancellation points" begin
    # @cancel_check with no scoped token is a no-op
    @test with(() -> (Base.@cancel_check; :ran), CANCEL_TOKEN => nothing) === :ran
    @test (Base.@cancel_check; :ran) === :ran

    # a cancellation point under a cancelled scope throws the request
    src = CancellationTokenSource()
    cancel!(src, CANCEL_REQUEST_ABANDON_EXTERNAL)
    err = with(CANCEL_TOKEN => CancellationToken(src)) do
        try
            Base.@cancel_check
            nothing
        catch e
            e
        end
    end
    @test err isa CancellationRequest
    @test err == CANCEL_REQUEST_ABANDON_EXTERNAL

    # the explicit-token form checks the given token, ignoring the scope
    live = CancellationToken(CancellationTokenSource())
    dead = CancellationToken(src)
    with(CANCEL_TOKEN => dead) do
        @test (Base.@cancel_check(live); :ran) === :ran
    end
    @test_throws CancellationRequest Base.@cancel_check(dead)
    @test (Base.@cancel_check(nothing); :ran) === :ran

    # level-triggered: after catching one request, the next point throws again
    with(CANCEL_TOKEN => dead) do
        caught = 0
        for _ in 1:2
            try
                Base.@cancel_check
            catch e
                e isa CancellationRequest || rethrow()
                caught += 1
            end
        end
        @test caught == 2
        # shielding scopes the token out
        with(CANCEL_TOKEN => nothing) do
            @test (Base.@cancel_check; :ran) === :ran
        end
    end

    # a cancellation against a nested source also throws from the nested
    # scope's cancellation points
    qroot = CancellationTokenSource()
    qchild = CancellationTokenSource(CancellationToken(qroot))
    cancel!(qroot)
    @test_throws CancellationRequest with(() -> Base.@cancel_check,
                                          CANCEL_TOKEN => CancellationToken(qchild))
end

@testset "cooperative cancellation of running tasks" begin
    # a @cancel_check polling loop is stopped cross-thread by cancel!
    started = Threads.Atomic{Bool}(false)
    t, src = cancellable_spawn() do
        started[] = true
        while true
            Base.@cancel_check
            yield() # let the canceller run when there is only one thread
        end
    end
    @test timedwait(() -> started[], 30.0) == :ok
    cancel!(src)
    @test timedwait(() -> istaskdone(t), 30.0) == :ok
    @test istaskfailed(t)
    @test t.result isa CancellationRequest

    # the scoped token is inherited through nested task spawns
    inner_result = Ref{Any}(nothing)
    t2, src2 = cancellable_spawn() do
        inner = Threads.@spawn begin
            while true
                Base.@cancel_check
                yield()
            end
        end
        # A cancellable wait would be interrupted by the delivery before
        # `inner` observes the cancellation at its own cancellation point;
        # the assertion is about `inner`'s own observation, so wait for its
        # completion shielded.
        Base._wait(inner, nothing)
        inner_result[] = inner.result
    end
    spin_started = timedwait(() -> istaskstarted(t2), 30.0)
    @test spin_started == :ok
    cancel!(src2)
    @test timedwait(() -> istaskdone(t2), 30.0) == :ok
    @test inner_result[] isa CancellationRequest

    # the hoisted-token form polls the explicit token
    src3 = CancellationTokenSource()
    tok3 = CancellationToken(src3)
    t3 = Threads.@spawn begin
        while true
            Base.@cancel_check tok3
            yield()
        end
    end
    @test timedwait(() -> istaskstarted(t3), 30.0) == :ok
    cancel!(src3, CANCEL_REQUEST_ABANDON_EXTERNAL)
    @test timedwait(() -> istaskdone(t3), 30.0) == :ok
    @test t3.result isa CancellationRequest
    @test t3.result == CANCEL_REQUEST_ABANDON_EXTERNAL
end

## Request-delivery tests (cancellation of waiting tasks)

# Start `f` as an @async-style (sticky, co-scheduled) task governed by a
# fresh cancellation source; returns (task, source).
function cancellable(f)
    src = CancellationTokenSource()
    t = with(() -> @async(f()), CANCEL_TOKEN => CancellationToken(src))
    return t, src
end

# wait a little, so cancellation targets are (most likely) started and parked
spin(n=4) = for _ in 1:n; yield(); end

# whether `t` is parked (its wait registration is enqueued on some waitee)
is_parked(t::Task) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue !== nothing)
parked_on(t::Task, @nospecialize(x)) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue === x)

@testset "level-triggered delivery and shielding" begin
    # cancellation is uniformly level-triggered: after catching the request,
    # unshielded waits under the cancelled scope keep throwing; shielded
    # cleanup proceeds, and the severity remains observable under the shield
    src = CancellationTokenSource()
    phase = Ref{Any}(:init)
    t = with(CANCEL_TOKEN => CancellationToken(src)) do
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
            rethrew &= Base.ambient_cancel_severity() === CANCEL_REQUEST_SAFE
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
end

@testset "cancellation of waiting tasks" begin
    # Cancellation of `sleep`
    t, src = cancellable(() -> sleep(1000))
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest

    # After catching the request, cleanup that must block shields itself
    t2, src2 = cancellable() do
        try
            sleep(1000)
        catch e
            e isa CancellationRequest || rethrow()
            sleep(0.01; cancel=nothing) # shielded: parking for cleanup
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
    if !Sys.iswindows() # fd polling requires a socket on Windows (ENOTSOCK)
        p = Pipe()
        Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
        fd = Base._fd(p.out)
        t, src = cancellable(() -> FileWatching.wait(fd; readable=true)) # nothing is ever written
        spin()
        cancel!(src)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest
        close(p)
    end

    path = tempname()
    touch(path)
    t, src = cancellable(() -> FileWatching.watch_file(path, 100.0)) # the file never changes
    spin()
    cancel!(src)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    rm(path)
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
    with(CANCEL_TOKEN => ctok) do
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

@testset "cancellation of blocked stream writes" begin
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    try
        # A write far exceeding the OS pipe buffer blocks until cancelled
        big = zeros(UInt8, 200_000_000)
        t, src = cancellable(() -> write(p, big))
        @test timedwait(() -> parked_on(t, p.in), 10.0) == :ok
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
        @test timedwait(() -> parked_on(tw, p.in), 10.0) == :ok
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

@testset "cancelled condition waiter reacquiring a contended lock" begin
    # A cancelled `wait(::Threads.Condition)` must rethrow the
    # CancellationRequest after reacquiring the condition lock, even when
    # the reacquire is contended: the waiter's stale (lazily collected)
    # condition-queue entry stays linked while it parks on the lock with a
    # fresh wait entry, and must not corrupt either queue.
    cond = Threads.Condition()
    src = Base.CancellationTokenSource()
    waiter_result = Channel{Any}(1)
    waiter = Threads.@spawn begin
        try
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.CancellationToken(src)) do
                lock(cond)
                try
                    wait(cond)
                finally
                    unlock(cond)
                end
            end
            put!(waiter_result, :completed)
        catch e
            put!(waiter_result, e)
        end
    end
    # wait until the waiter is parked on the condition
    @test timedwait(10) do
        lock(cond)
        parked = !isempty(cond)
        unlock(cond)
        parked
    end === :ok
    # a holder keeps the condition lock while the cancellation is delivered
    held = Base.Event()
    release = Base.Event()
    holder = Threads.@spawn begin
        lock(cond)
        notify(held)
        wait(release; cancel=nothing)
        unlock(cond)
    end
    wait(held)
    Base.cancel!(src)
    # give the woken waiter time to reach the contended reacquire and park
    sleep(0.5)
    notify(release)
    wait(waiter)
    result = take!(waiter_result)
    @test result isa Base.CancellationRequest
    # the condition lock must be intact and uncontended afterwards
    @test trylock(cond.lock)
    unlock(cond.lock)
    wait(holder)
end
