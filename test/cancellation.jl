# This file is a part of Julia. License is MIT: https://julialang.org/license

using Base: cancel!, CancellationRequest, CancellationToken, CancellationTokenSource,
    CANCEL_REQUEST_SAFE, CANCEL_REQUEST_ABANDON_EXTERNAL, CANCEL_REQUEST_ABANDON_ALL,
    CANCEL_TOKEN
using Base.ScopedValues: with, ScopedValue

# Start `f` as an @async-style (sticky, co-scheduled) task governed by a
# fresh cancellation source; returns (task, source).
function cancellable(f)
    src = CancellationTokenSource()
    t = with(() -> @async(f()), CANCEL_TOKEN => CancellationToken(src))
    return t, src
end

# Threads.@spawn-style variant (non-sticky, explicitly on the default pool -
# a compute-bound victim must not land on the interactive/io thread).
function cancellable_spawn(f)
    src = CancellationTokenSource()
    t = with(() -> Threads.@spawn(f()), CANCEL_TOKEN => CancellationToken(src))
    return t, src
end

# whether `t` is parked (its wait registration is enqueued on some waitee)
is_parked(t::Task) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue !== nothing)
parked_on(t::Task, @nospecialize(x)) = (w = @atomic :acquire t.waiting_on; w isa Base.WaitEntry && w.queue === x)

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

# Park a shielded watcher on a fresh child of `parent`; the child source
# escapes this frame only through the parked watcher task.
@noinline function _spawn_watcher_on_child(parent)
    child = CancellationTokenSource(CancellationToken(parent))
    t = @async wait(CancellationToken(child); cancel=nothing)
    @assert timedwait(() -> parked_on(t, child), 10.0) == :ok
    return t
end

@testset "waiting for a token as an event" begin
    # an already-cancelled token: immediate value return, no throw
    src = CancellationTokenSource()
    cancel!(src, CANCEL_REQUEST_ABANDON_EXTERNAL)
    req = wait(CancellationToken(src); cancel=nothing)
    @test req isa CancellationRequest
    @test req.request == CANCEL_REQUEST_ABANDON_EXTERNAL.request

    # a parked watcher is completed (not interrupted) by the cancellation
    src = CancellationTokenSource()
    t = @async wait(CancellationToken(src); cancel=nothing)
    @test timedwait(() -> parked_on(t, src), 10.0) == :ok
    @test !istaskdone(t)
    cancel!(src)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test !istaskfailed(t)
    @test fetch(t) isa CancellationRequest
    @test src.watchers === nothing

    # ancestor cancellation reaches a watcher on a descendant source
    parent = CancellationTokenSource()
    child = CancellationTokenSource(CancellationToken(parent))
    t = @async wait(CancellationToken(child); cancel=nothing)
    @test timedwait(() -> parked_on(t, child), 10.0) == :ok
    cancel!(parent)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test fetch(t) isa CancellationRequest

    # ordinary cancel semantics: the governing token (inherited from the
    # scope) interrupts the wait, leaving the watched token untouched
    watched = CancellationTokenSource()
    gov = CancellationTokenSource()
    t = with(CANCEL_TOKEN => CancellationToken(gov)) do
        @async wait(CancellationToken(watched))
    end
    @test timedwait(() -> parked_on(t, watched), 10.0) == :ok
    cancel!(gov)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test istaskfailed(t)
    @test t.result isa CancellationRequest
    @test !Base.iscancelled(CancellationToken(watched))
    @test watched.watchers === nothing

    # a watcher governed by an *ancestor* of the watched source is
    # interrupted, not completed: the ancestor's own drain claims it before
    # descending to the watched child (this is why callback watchers shield)
    outer = CancellationTokenSource()
    inner = CancellationTokenSource(CancellationToken(outer))
    t = with(CANCEL_TOKEN => CancellationToken(outer)) do
        @async wait(CancellationToken(inner))
    end
    @test timedwait(() -> parked_on(t, inner), 10.0) == :ok
    cancel!(outer)
    @test timedwait(() -> istaskdone(t), 10.0) == :ok
    @test istaskfailed(t)

    # waiting under the same token is refused, explicitly and inherited
    srcs = CancellationTokenSource()
    toks = CancellationToken(srcs)
    @test_throws ArgumentError wait(toks; cancel=toks)
    @test_throws ArgumentError with(() -> wait(toks), CANCEL_TOKEN => toks)

    # the callback pattern: a shielded watcher performs its action during
    # the cancellation
    src = CancellationTokenSource()
    fired = Base.Event()
    watcher = @async begin
        wait(CancellationToken(src); cancel=nothing)
        notify(fired)
    end
    @test timedwait(() -> parked_on(watcher, src), 10.0) == :ok
    cancel!(src)
    wait(fired; cancel=nothing)
    wait(watcher)
    @test !istaskfailed(watcher)

    # a parked watcher keeps the watched source attached to the tree
    # (observability == reachability), so an ancestor cancellation still
    # reaches it after a GC
    parent2 = CancellationTokenSource()
    t2 = _spawn_watcher_on_child(parent2)
    GC.gc()
    cancel!(parent2)
    @test timedwait(() -> istaskdone(t2), 10.0) == :ok
    @test fetch(t2) isa CancellationRequest
end

@testset "cancellation of waiting tasks" begin
    # A task spawned under an already-cancelled scope starts but observes the
    # cancellation before running any user code
    src = CancellationTokenSource()
    body_ran = Ref(false)
    t = with(CANCEL_TOKEN => CancellationToken(src)) do
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
    t4 = with(() -> @task(body_ran[] = true), CANCEL_TOKEN => CancellationToken(src4))
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

@testset "cancelled scopes are level-triggered" begin
    t, src = cancellable() do
        try
            sleep(1000)
        catch e
            e isa CancellationRequest || rethrow()
        end
        # The scope stays cancelled: unshielded blocking operations keep
        # throwing until the task leaves the scope or shields.
        rethrew = try
            sleep(0.01)
            false
        catch e
            e isa CancellationRequest
        end
        # Shielded IO still works, and the severity remains observable.
        sleep(0.01; cancel=nothing)
        rethrew && Base.ambient_cancel_severity() === CANCEL_REQUEST_SAFE
    end
    spin()
    cancel!(src)
    @test fetch(t)
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
    function run_with_sigint(code::String, delays; forcekill::Bool=false,
                             open_stdin::Bool=false, threads::Int=0)
        out = Pipe()
        cmd = threads > 0 ?
            `$(Base.julia_cmd()) --startup-file=no --threads=$threads -e $code` :
            `$(Base.julia_cmd()) --startup-file=no -e $code`
        inpipe = open_stdin ? Pipe() : devnull
        p = run(pipeline(cmd, stdin=inpipe, stdout=out, stderr=out), wait=false)
        close(out.in)
        open_stdin && close(inpipe.out)
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
        open_stdin && close(inpipe.in)
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
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.sigint_new_episode!()) do
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

    # TODO(port): the @sync compute-spinner ^C sub-test is deferred until the
    # scoped-child delivery port lands (7a32ba2f40, "Deliver a pending ^C to
    # scoped tasks without the listener"): signal-side marking covers the
    # episode source, but the spinner here is bound to the @sync child source,
    # which at JULIA_NUM_THREADS=1 nothing marks without that commit's
    # per-thread bound-source propagation - the child spins forever.

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
    @test occursin(r"Abandoning (the )?current task", output)
    @test p.exitcode == 128 + 2

    # ^C with a stray @async task pending is catchable and the script exits
    # cleanly - historically a "fatal: error thrown and no exception handler
    # available" (issues #29369, #45055)
    output, p = run_with_sigint("""
        @async println("Hello!")
        try
            println("Hit ctrl-c!")
            sleep(10)
        catch err
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.sigint_new_episode!()) do
                showerror(stdout, err); println()
                println("done")
            end
        end
    """, [1.0])
    @test occursin("Hello!", output)
    @test occursin("CancellationRequest", output)
    @test occursin("done", output)
    @test !occursin("fatal", output)
    @test p.exitcode == 0

    # ^C during a blocked read from stdin reports and exits - historically a
    # fatal unhandled InterruptException on the second press (issue #43451)
    output, p = run_with_sigint("read(stdin)", [1.0]; open_stdin=true)
    @test occursin("CancellationRequest", output)
    @test !occursin("fatal", output)
    @test p.exitcode == 1

    # A rapid second press while the first cancellation is still unwinding
    # or reporting must not crash the process (issue #50045). The second
    # press may cancel the error-report epoch itself, in which case the
    # fallback note appears instead of the report.
    output, p = run_with_sigint("sleep(100)", [1.0, 0.1])
    @test occursin("CancellationRequest", output) ||
        occursin("displaying the error report failed", output)
    @test !occursin("fatal", output)
    @test p.exitcode == 1

    # A catch-all loop that swallows every CancellationRequest cannot hide
    # from ^C (issue #4037): while the scope stays cancelled the request is
    # re-thrown at every blocking operation (the warning shows the
    # delivered-but-not-completed flavor), and the escalation ladder still
    # progresses to the point of abandoning the task. The abandonment rung
    # itself is a hail mary that may leave the process inconsistent, so this
    # asserts only that it is reached and announced - not any process
    # behavior after the freeze (the watchdog reaps the process).
    output, p = run_with_sigint("""
        while true
            try
                sleep(10)
            catch
            end
        end
    """, [1.0, 2.5, 2.5]; forcekill=true)
    @test occursin("Cancellation is in progress, but has not completed", output)
    @test occursin(r"Abandoning (the )?current task", output)

    # ^C stops a swarm of print-flooding tasks and the script continues
    # (issue #47839)
    output, p = run_with_sigint("""
        ts = [@async (while true; println("hi"); end) for _ in 1:20]
        try
            sleep(100)
        catch e
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.sigint_new_episode!()) do
                for t in ts
                    try; wait(t); catch; end
                end
                println("ALL-STOPPED")
            end
        end
    """, [1.5])
    @test occursin("ALL-STOPPED", output)
    @test !occursin("fatal", output)
    @test p.exitcode == 0

    # ^C on a Threads.@threads loop raises a catchable CompositeException
    # instead of killing the process (issue #56462)
    output, p = run_with_sigint("""
        try
            Threads.@threads for i in 1:8
                sleep(100)
            end
        catch e
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.sigint_new_episode!()) do
                println("caught: ", typeof(e))
                println("session-alive")
            end
        end
    """, [1.5]; threads=4)
    @test occursin("caught: CompositeException", output)
    @test occursin("session-alive", output)
    @test !occursin("fatal", output)
    @test !occursin("attempt to switch to exited task", output)
    @test p.exitcode == 0

    # A watcher task on the ^C episode token is the supported shape for a
    # user-defined interrupt handler (superseding the design of #49541): the
    # ^C completes - rather than unwinds - its wait, and its reaction runs
    # under its own shielded scope
    output, p = run_with_sigint("""
        tok = Base.CANCEL_TOKEN[]
        w = Threads.@spawn Base.ScopedValues.with(Base.CANCEL_TOKEN => nothing) do
            req = wait(tok)
            println("HANDLER-RAN ", typeof(req))
        end
        try
            sleep(100)
        catch e
            Base.ScopedValues.with(Base.CANCEL_TOKEN => Base.sigint_new_episode!()) do
                wait(w)
                println("DONE")
            end
        end
    """, [1.0])
    @test occursin("HANDLER-RAN Base.CancellationRequest", output)
    @test occursin("DONE", output)
    @test p.exitcode == 0
end

if Sys.isunix()
    # TODO(port): the interactive pty ^C escalation-ladder testset is deferred:
# reliable rung escalation requires the standing-offer/generation semantics
# ported later in the series (a press must not invalidate the offer it
# accepts), and the abandonment announcement wording it expects arrives with
# the same arc. Restored by the commits that port that machinery; content
# preserved in the port notes (/workspace/.git/port-deferred-pty-ladder.jl).

@testset "^C in the REPL (pty)" begin
        isdefined(Main, :FakePTYs) || @eval Main include("testhelpers/FakePTYs.jl")
        pts, ptm = Main.FakePTYs.open_fake_pty()

        # Interactive julia on the pty; drive it like a user pressing ^C.
        env = copy(ENV)
        env["TERM"] = "dumb"
        env["JULIA_HISTORY"] = tempname()
        p = run(detach(setenv(`$(Base.julia_cmd()) -i -q --startup-file=no --color=no`, env)),
                pts, pts, pts; wait=false)
        ccall(:close, Cint, (Cint,), pts) # only the child owns the pts now

        transcript_lock = ReentrantLock()
        transcript = UInt8[]
        reader = @async try
            while true
                chunk = readavailable(ptm)
                isempty(chunk) && break
                @lock transcript_lock append!(transcript, chunk)
            end
        catch # pty closes when the child exits
        end
        cursor = Ref(1)
        snapshot() = @lock transcript_lock String(copy(transcript))
        function expect(needle::String; timeout::Real=30.0)
            status = timedwait(timeout; pollint=0.05) do
                idx = findnext(needle, snapshot(), cursor[])
                idx === nothing && return false
                cursor[] = last(idx) + 1
                return true
            end
            if status !== :ok
                @error "expect timed out" needle tail=snapshot()[max(1, cursor[]):end]
            end
            @test status == :ok
        end
        sendline(s) = write(ptm, s * "\n")

        expect("julia> ")

        # a SIGINT at an idle prompt (^C or an external `kill -INT`) must
        # not disturb the session (issue #42072)
        kill(p, Base.SIGINT)
        sleep(0.5)
        sendline("20 + 21")
        expect("41")
        expect("julia> ")

        # ^C interrupts a sleeping REPL evaluation and reports it
        sendline("println(\"EVAL-1\"); sleep(1000)")
        expect("EVAL-1") # the evaluation is running (robust under load)
        sleep(0.5)       # ... and parked in sleep(1000)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")

        # the REPL evaluates normally afterwards
        sendline("6 * 7")
        expect("42")
        expect("julia> ")

        # a spinning evaluation triggers the escalation warning; the second
        # ^C abandons it and the REPL is rescued with a fresh backend
        sendline("println(\"EVAL-2\"); xr = Ref(1.0); while true; xr[] = xr[] * 1.0000001 + 0.1; end")
        expect("EVAL-2") # the evaluation is running (robust under load)
        sleep(0.5)       # ... and spinning
        kill(p, Base.SIGINT)
        expect("failed to acknowledge SIGINT"; timeout=15.0)
        kill(p, Base.SIGINT)
        expect("Abandoning current task")
        expect("julia> ")

        # the rescued REPL still evaluates
        sendline("3 + 4")
        expect("7")
        expect("julia> ")

        # and ^C still works after the rescue
        sendline("println(\"EVAL-3\"); sleep(1000)")
        expect("EVAL-3")
        sleep(0.5)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")

        # a background task from an earlier evaluation belongs to an earlier
        # ^C epoch: interrupting the current evaluation leaves it running
        # (issue #25790)
        sendline("global bgc = Ref(0); global bg = @async while true; sleep(0.01); bgc[] += 1; end; println(\"BG-UP\")")
        expect("BG-UP")
        expect("julia> ")
        sendline("println(\"EVAL-4\"); sleep(1000)")
        expect("EVAL-4")
        sleep(0.5)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")
        sendline("print(\"bg-done=\", istaskdone(bg)); c0 = bgc[]; sleep(0.3); println(\"; bg-alive=\", bgc[] > c0)")
        expect("bg-done=false; bg-alive=true")
        expect("julia> ")

        # ^C during an in-evaluation terminal read recovers the prompt
        # (the class of issue #58105's "Install package?" prompt)
        sendline("println(\"EVAL-5\"); readline()")
        expect("EVAL-5")
        sleep(0.5)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")

        # ^C while parked in a server accept recovers, leaving the server
        # usable (the class of issue #58689)
        sendline("using Sockets; global srv = listen(Sockets.localhost, 0); println(\"LISTENING\"); accept(srv)")
        expect("LISTENING")
        sleep(0.5)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")
        sendline("println(\"srv-open=\", isopen(srv)); close(srv)")
        expect("srv-open=true")
        expect("julia> ")

        # cancelling a BigInt computation never yanks control out of libgmp
        # in an unsafe spot the way the old asynchronous InterruptException
        # delivery could (corrupting the heap - issue #56545): the loop is
        # deliberately checkless - delivery lands either on an MPZ entry
        # point's own cancellation point, asynchronously inside audited
        # libgmp compute (unwound via the reset region published across the
        # annotated call), or inside the allocation hooks (deferred and
        # chained into the reset on exit) - and BigInt arithmetic in the
        # session works correctly afterwards
        sendline("println(\"EVAL-6\"); let b = big(3); while true; b = b*b % (big(10)^200); end; end")
        expect("EVAL-6")
        sleep(0.5)
        kill(p, Base.SIGINT)
        expect("CancellationRequest")
        expect("julia> ")
        sendline("println(string(factorial(big(30))))")
        expect("265252859812191058636308480000000")
        expect("julia> ")

        sendline("exit()")
        @test success(p)
        close(ptm)
        wait(reader)
    end
end
