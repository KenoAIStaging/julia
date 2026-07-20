# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    UnifiedCompiler

Pkgimage wrapper for the `Compiler.Unified` port (`Compiler/src/unified/*`).

`Compiler.load_unified!` normally evaluates the unified sources into a
Main-rooted carrier module (interpreted include + JIT of everything the
pipeline touches, ~30-60s of warmup tax per process). This package includes
the SAME source files against the SAME loader-registered Compiler package
instance, so the whole stack — plus the stdlib Compiler specializations its
precompile workload drives — lands in a precompiled pkgimage.
`Compiler.load_unified!` prefers this package when it is loadable and bound
to the calling Compiler instance, and falls back to the carrier include
otherwise (stdlib-vs-Base.Compiler mismatch, bootstrap, `--compiled-modules=no`,
or `JULIA_UNIFIED_PKGIMAGE=0`).

The carrier contract is preserved: `Unified.jl` resolves the Compiler
instance it runs against via `Base.parentmodule(@__MODULE__).CompilerModule`,
which this module binds below. This module is a normal module, so the
carrier's reason for existing (the Compiler baremodule rebinds
`getproperty = Core.getfield`, breaking property-forwarding types) does not
apply here either.
"""
module UnifiedCompiler

import Compiler

# The carrier contract: Compiler/src/unified/Unified.jl binds
# `const Compiler = Base.parentmodule(@__MODULE__).CompilerModule`.
const CompilerModule = Compiler

include(Base.joinpath(@__DIR__, "..", "..", "Compiler", "src", "unified", "Unified.jl"))

"""
    _reset_session_state!()

Empty every piece of session-scoped mutable state in `Unified` (memo tables
keyed by MethodInstance with world bounds, the pipeline ledger, reentrancy
guards). Run at the END of the precompile workload so none of it is baked
into the pkgimage, and again defensively from `__init__` (world numbers and
MethodInstance references from the precompile process are meaningless in a
new session).
"""
function _reset_session_state!()
    U = Unified
    # driver ledger + A6 memo
    U.reset_pipeline_stats!()
    empty!(U.DRIVER_MEMO)
    U.MEMO_OWNER[] = nothing
    U.STATS_OWNER[] = nothing
    U.STATS_DROPPED[] = 0
    # optimizer memos
    empty!(U.EA_OPT_SUMMARIES); empty!(U.EA_OPT_ACTIVE); U.EA_OPT_WORLD[] = 0
    empty!(U.OPT_FX_MEMO);      empty!(U.OPT_FX_ACTIVE); U.OPT_FX_WORLD[] = 0
    U.OPT_NEST_DEPTH[] = 0; U.OPT_WORK_LEFT[] = 0
    # inline cost memo
    empty!(U.INLINE_COST_MEMO); empty!(U.INLINE_COST_ACTIVE); U.INLINE_COST_WORLD[] = 0
    # queries surface
    empty!(U.QUERY_STATES)
    # activation state
    U.GLOBAL_MODE[] = false
    U.SHADOW_ACTIVE[] = false
    U.SHADOW_ENABLED[] = true
    let s = U.SHADOW
        s.seen = 0; s.converted = 0; s.verified = 0
        s.outside_matrix = 0; s.errors = 0; s.last_error = nothing
    end
    # inference tower budget
    U.TOWER_FRAME_CAP[] = 0
    return nothing
end

function __init__()
    _reset_session_state!()
    return nothing
end

if Base.generating_output()
    include("precompile.jl")
end

end # module UnifiedCompiler
