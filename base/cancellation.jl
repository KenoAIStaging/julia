# This file is a part of Julia. License is MIT: https://julialang.org/license

## Cancellation tokens
#
# Cancellation is organized around *cancellation token sources*
# (`Core.CancellationTokenSource`): level-triggered condition nodes arranged
# in a DAG (a source may have several parents). Cancelling a source cancels
# all of its descendants; the cancelled state is monotonic (severities only
# escalate, never reset). Following the
# .NET split, a `CancellationToken` is the observe view handed to code that
# may only *react* to cancellation; the source is the capability to *request*
# it.
#
# For convenience, the scoped value `CANCEL_TOKEN` carries the governing token
# carries the default cancellation token.

const CancellationTokenSource = Core.CancellationTokenSource

"""
    CancellationToken(src::CancellationTokenSource)

The observe side of a [`CancellationTokenSource`](@ref): code holding a
token can be interrupted by - and can query ([`iscancelled`](@ref)) -
cancellation of the associated source, but cannot request cancellation
itself. Only the holder of the *source* can call [`cancel!`](@ref).

A token takes effect by scoping it over a computation via the
[`CANCEL_TOKEN`](@ref) scoped value or by explicit token passing.
Once the associated source is cancelled, the computation's cancellation points (see
[`@cancel_check`](@ref)) throw a [`CancellationRequest`](@ref).
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
struct CancellationRequest <: Exception
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
const CANCEL_REQUEST_SAFE = CancellationRequest(0x1)

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

# The status byte reported by a compiled cancellation point is the governing
# source's state byte (0 while uncancelled, the severity once cancelled) with
# the 0x40 bit merged in when the task has a pending cooperative-yield
# request. The mask recovers the severity half of such a status byte; source
# state reads themselves need no masking.
const STATUS_PREEMPT_BIT = 0x40
const SEVERITY_MASK = 0x3f

# A request's payload is the plain severity under the simplified state
# encoding; kept as an accessor for the delivery layer's comparisons.
severity(cr::CancellationRequest) = cr.request

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
    return CancellationRequest(st)
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

## Source construction and graph linkage

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

# CAS-max the source's state to `sev` (a valid, nonzero severity). Returns
# true if the state was raised, false if it was already at (or above) the
# severity.
# seq_cst: pairs with the (child-list publication; state read) sequence in
# `jl_new_cancel_source` - see the walk in `_cancel_walk_node!`.
function _raise_state!(src::CancellationTokenSource, sev::UInt8)
    old = @atomic :monotonic src.state
    while true
        if old >= sev
            return false
        end
        old, success = @atomicreplace :sequentially_consistent :monotonic src.state old => sev
        success && return true
    end
end

## Cancellation
#
# Cancellation is uniformly level-triggered: while the governing token is
# cancelled, every cancellation point throws the `CancellationRequest`.

"""
    cancel!(src::CancellationTokenSource,
            request::CancellationRequest=CANCEL_REQUEST_SAFE)::Bool

Cancel `src` and all of its descendants at the given severity. Level-triggered
and monotonic: observers (including future registrants) see the cancellation
until the source is discarded, and repeated calls only have an effect when
they *escalate* the severity ([`CANCEL_REQUEST_SAFE`](@ref) ->
[`CANCEL_REQUEST_ABANDON_EXTERNAL`](@ref) ->
[`CANCEL_REQUEST_ABANDON_ALL`](@ref)). Returns whether the call changed the
state.

Computations governed by a token of `src` or a descendant observe the cancellation at
their cancellation points (see [`@cancel_check`](@ref)), which throw a
[`CancellationRequest`](@ref).

When `cancel!` returns, every source currently reachable from `src` has
been advanced to (at least) the requested severity by this very call.
"""
function cancel!(src::CancellationTokenSource,
                 request::CancellationRequest=CANCEL_REQUEST_SAFE)
    sev = request.request
    if !(sev == 0x1 || sev == 0x3 || sev == 0x4)
        throw(ArgumentError("invalid cancellation severity $(repr(request.request))"))
    end
    raised = _raise_state!(src, sev)
    _cancel_walk!(src, sev)
    Threads.atomic_fence_heavy()
    return raised
end

function _cancel_walk!(src::CancellationTokenSource, sev::UInt8)
    visited = IdSet{CancellationTokenSource}()
    push!(visited, src)
    pending = CancellationTokenSource[src]
    while !isempty(pending)
        _cancel_walk_node!(pop!(pending), sev, pending, visited)
    end
    return nothing
end

function _cancel_walk_node!(node::CancellationTokenSource, sev::UInt8,
                            pending::Vector{CancellationTokenSource},
                            visited::IdSet{CancellationTokenSource})
    st = @atomic :sequentially_consistent node.state
    sev < st && (sev = st)
    c = @atomic node.child_head
    while c !== nothing
        c = c::CancellationTokenSource
        _raise_state!(c, sev)
        if !(c in visited)
            push!(visited, c)
            push!(pending, c)
        end
        c = _cancel_next_child(node, c)
    end
    return nothing
end

## Cancellation points

# The slow path of `@cancel_check`: `st` is the (non-zero) state byte of the
# governing source.
@noinline function handle_cancellation!(src::CancellationTokenSource, st::UInt8)
    # re-read: deliver the severity current at throw time, not the one the
    # fast path happened to observe
    st = @atomic :acquire src.state
    throw(CancellationRequest(st))
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
        # also a GC safepoint, so that a tight polling loop cannot starve a
        # concurrent stop-the-world
        ccall(:jl_gc_safepoint, Cvoid, ())
        # the scoped cancellation token ...
        checkcancel(default_cancel_source())
        # ... and per-task requests (delivered through
        # `Task.cancellation_request`, see `Base.cancel!(::Task)`). The
        # builtin must come LAST: it establishes the reset point, and any
        # non-reset-safe call sequenced after it would force the lowering
        # pass to clear the just-published reset context again.
        local req = Core.cancellation_point!()
        req !== nothing && handle_cancellation!(req)
        nothing
    end
end

macro cancel_check(tok)
    quote
        local t = $(esc(tok))
        ccall(:jl_gc_safepoint, Cvoid, ())
        checkcancel(t === nothing ? nothing : (t::CancellationToken).source)
        # see above: the reset-point-establishing builtin must come last
        local req = Core.cancellation_point!()
        req !== nothing && handle_cancellation!(req)
        nothing
    end
end

# Throw the `CancellationRequest` if `src` is cancelled (level-triggered:
# no per-task state is consulted).
@inline function checkcancel(src::CancellationTokenSource)
    st = @atomic :monotonic src.state
    st == 0x00 && return nothing
    handle_cancellation!(src, st)
    return nothing
end
checkcancel(::Nothing) = nothing
checkcancel(tok::CancellationToken) = checkcancel(tok.source)

## CANCEL_TOKEN
struct CancelTokenKey <: AbstractScopedValue{Union{Nothing, CancellationToken}} end

"""
    CANCEL_TOKEN

The scoped value carrying the [`CancellationToken`](@ref) that governs the
current dynamic extent, or `nothing` if there is none. [`@cancel_check`](@ref)
checks it, and tasks spawned within a scope inherit it.

Establish a governing token with the standard scoped-value API
([`ScopedValues.@with`](@ref) / [`ScopedValues.with`](@ref)):

```julia
using Base.ScopedValues

src = Base.CancellationTokenSource()
with(Base.CANCEL_TOKEN => Base.CancellationToken(src)) do
    ...   # cancellation points in here observe `cancel!(src)`
end
```

Scoping `Base.CANCEL_TOKEN => nothing` instead *shields* the enclosed code
from an outer (possibly cancelled) token; use this for cleanup that must
complete while the surrounding computation is being cancelled.

The current value can be read with `Base.CANCEL_TOKEN[]`.
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
