# This file is a part of Julia. License is MIT: https://julialang.org/license

using Base: cancel!, CancellationRequest, CancellationToken, CancellationTokenSource,
    CANCEL_REQUEST_SAFE, CANCEL_REQUEST_ACK, CANCEL_REQUEST_ABANDON_EXTERNAL,
    CANCEL_REQUEST_ABANDON_ALL, CANCEL_TOKEN
using Base.ScopedValues: with, ScopedValue

# Threads.@spawn-style cancellable task (non-sticky, explicitly on the
# default pool - a compute-bound victim must not land on the interactive/io
# thread); returns (task, source).
function cancellable_spawn(f)
    src = CancellationTokenSource()
    t = with(() -> Threads.@spawn(f()), CANCEL_TOKEN => CancellationToken(src))
    return t, src
end

@testset "cancellation token tree semantics" begin
    # cancel! marks the whole subtree, level-triggered
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
    @test_throws ArgumentError cancel!(CancellationTokenSource(), CancellationRequest(0x1))
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
    @test Base.severity(err) == Base.severity(CANCEL_REQUEST_ABANDON_EXTERNAL)

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
        try
            wait(inner)
        catch
        end
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
    @test Base.severity(t3.result) == Base.severity(CANCEL_REQUEST_ABANDON_EXTERNAL)
end

## Request-delivery tests (cancellation of running and waiting tasks)

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

@testset "cancellation of waiting tasks" begin
    # Cancellation of a task that was never started
    t = @task nothing
    @test cancel!(t)
    @test t.state === :cancelled
    @test istaskfailed(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # Scheduling a cancelled task transitions it to failed, without running it
    schedule(t)
    @test t.state === :failed

    # Cancellation of `sleep`
    t = @async sleep(1000)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest

    # Cancellation of a task blocked on a Channel
    c = Channel{Int}(0)
    t = @async take!(c)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # The channel remains usable
    t2 = @async take!(c)
    put!(c, 7)
    @test fetch(t2) == 7

    # Cancellation of a task waiting on another task propagates
    t_in = @async sleep(1000)
    t = @async wait(t_in)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test istaskdone(t_in) && istaskfailed(t_in)
end

@testset "cancellation of lock and condition waits" begin
    # Task blocked in lock(::ReentrantLock)
    lk = ReentrantLock()
    lock(lk)
    t = @async lock(lk)
    spin()
    # let it spin through the fast path and park
    @test timedwait(() -> t.queue !== nothing, 5.0) == :ok
    cancel!(t)
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
    t = @async put!(c, 2)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @test take!(c) == 1
    put!(c, 3) # channel remains functional
    @test take!(c) == 3

    # Task blocked in wait(::Threads.Condition)
    cond = Threads.Condition()
    t = @async @lock cond wait(cond)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @lock cond notify(cond) # still functional (no waiters)

    # Task blocked in wait(::Base.Process); the process itself keeps running
    p = run(`sleep 1000`; wait=false)
    t = @async wait(p)
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    @test process_running(p)
    kill(p); wait(p)

    # Task blocked in waitany
    t1 = @async sleep(1000)
    t2 = @async sleep(1000)
    t = @async waitany([t1, t2])
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CancellationRequest
    # the awaited tasks remain unaffected and cancellable
    cancel!(t1); cancel!(t2)
    @test_throws TaskFailedException wait(t1)
    @test_throws TaskFailedException wait(t2)
end

@testset "cancellation of computing tasks" begin
    # Both variants need a second thread to run the cancellation from: the
    # compute victims never yield, so with one thread the canceller would
    # never run. (Signal-side delivery that also covers -t1 arrives with the
    # ^C machinery later in this series.)
    if Threads.nthreads() > 1
        # Polling cancellation via @cancel_check
        t = Threads.@spawn find_collatz_counterexample()
        sleep(0.2)
        cancel!(t)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest

        # Asynchronous interruption of a checkless loop through the
        # reset_ctx mechanism
        t = Threads.@spawn find_collatz_counterexample2()
        sleep(0.5)
        cancel!(t)
        sleep(0.5)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest
    end
end

@testset "structured cancellation of @sync" begin
    t = @async begin
        @sync begin
            @async sleep(1000)
            @async sleep(1000)
        end
    end
    spin()
    cancel!(t)
    @test_throws TaskFailedException wait(t)
    @test t.result isa CompositeException
    @test length(t.result.exceptions) == 2
end

@testset "cancellation of blocked stream writes" begin
    p = Pipe()
    Base.link_pipe!(p, reader_supports_async=true, writer_supports_async=true)
    try
        # A write far exceeding the OS pipe buffer blocks until cancelled
        big = zeros(UInt8, 200_000_000)
        t = @async write(p, big)
        sleep(0.5)
        @test t.queue === p.in
        cancel!(t)
        @test_throws TaskFailedException wait(t)
        @test t.result isa CancellationRequest
    finally
        close(p)
    end
end

@testset "acknowledged requests do not re-trigger" begin
    t = @async begin
        try
            sleep(1000)
        catch e
            e isa CancellationRequest || rethrow()
        end
        # The request was delivered (and acknowledged); this task can still
        # perform IO and sleep.
        sleep(0.01)
        (@atomic :acquire current_task().cancellation_request) === CANCEL_REQUEST_ACK
    end
    spin()
    cancel!(t)
    @test fetch(t)
end

@testset "task abandonment wakes waiters" begin
    if Threads.nthreads() > 1
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
        spin()
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

    # Catching ^C in a script
    output, p = run_with_sigint("""
        try
            sleep(100)
            println("FAIL: not cancelled")
        catch e
            println("caught: ", typeof(e))
        end
        println("continued")
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
