# The effects-parity scoreboard (COMPILER-PORT-PLAN A3 / EFFECTS-PARITY-PLAN):
# run the UNMODIFIED Compiler/test/effects.jl assert corpus with the unified
# pipeline enabled (Compiler.UNIFIED_HOOKS on), and report pass/fail/error/
# broken counts, the failing-assert list (with source locations), and the
# driver's fallback ledger. `Base.infer_effects`/`code_typed` sites route
# through the unified driver; every per-match fallback silently yields the
# STOCK answer, so the ledger is as much the instrument as the counts.
#
# Usage (from the julia checkout root):
#     ./usr/bin/julia --startup-file=no --project=Compiler \
#         Compiler/bench/effects_scoreboard.jl [logfile]
#
# stdout carries the summary; the full failing list goes to `logfile`
# (default: effects_scoreboard.log in the current directory).

using Test

import Compiler
const Unified = Compiler.load_unified!()
Unified.reset_pipeline_stats!()
Unified.enable_pipeline!()

const LOGFILE = isempty(ARGS) ? "effects_scoreboard.log" : ARGS[1]

# Run the corpus. setup_Compiler.jl (included by effects.jl) performs the
# `@activate Compiler` reflection routing; the pipeline hooks are already on.
# The wrapper testset absorbs the outermost `finish` (which would otherwise
# throw TestSetException on any failure, skipping the report below).
const wrapper = Test.DefaultTestSet("scoreboard-wrapper")
Base.ScopedValues.@with Test.CURRENT_TESTSET => wrapper Test.TESTSET_DEPTH => 1 begin
    @testset "effects (unified pipeline)" begin
        include(joinpath(@__DIR__, "..", "test", "effects.jl"))
    end
end
const ts = wrapper.results[1]::Test.DefaultTestSet

# ---------------------------------------------------------------------------
# Walk the testset tree: count leaves, collect non-passing results
# ---------------------------------------------------------------------------

struct Tally
    npass::Base.RefValue{Int}
    nfail::Base.RefValue{Int}
    nerror::Base.RefValue{Int}
    nbroken::Base.RefValue{Int}
    failing::Vector{Tuple{String,Any}}
end
Tally() = Tally(Ref(0), Ref(0), Ref(0), Ref(0), Tuple{String,Any}[])

function walk!(t::Tally, set::Test.DefaultTestSet, prefix::String)
    t.npass[] += set.n_passed
    for r in set.results
        if r isa Test.DefaultTestSet
            walk!(t, r, string(prefix, "/", r.description))
        elseif r isa Test.Fail
            t.nfail[] += 1
            push!(t.failing, (prefix, r))
        elseif r isa Test.Error
            t.nerror[] += 1
            push!(t.failing, (prefix, r))
        elseif r isa Test.Broken
            t.nbroken[] += 1
        elseif r isa Test.Pass
            t.npass[] += 1
        end
    end
    return t
end

tally = walk!(Tally(), ts, "effects")
stats = Unified.pipeline_stats()
nfallback = sum(values(stats.fallbacks); init = 0)

open(LOGFILE, "w") do io
    for out in (stdout, io)
        println(out, "SCOREBOARD pass=", tally.npass[], " fail=", tally.nfail[],
                " error=", tally.nerror[], " broken=", tally.nbroken[],
                " total=", tally.npass[] + tally.nfail[] + tally.nerror[] + tally.nbroken[])
        println(out, "PIPELINE unified=", stats.unified, " fallbacks=", nfallback)
        for (reason, n) in sort!(collect(stats.fallbacks); by = last, rev = true)
            println(out, "  fallback ", rpad(String(reason), 24), " ", n)
        end
    end
    println(io)
    println(io, "== failing asserts (", length(tally.failing), ") ==")
    for (prefix, r) in tally.failing
        src = r.source
        kind = r isa Test.Error ? "ERROR" : "FAIL"
        println(io, kind, " @ ", src.file, ":", src.line, "  [", prefix, "]")
        if r isa Test.Fail
            println(io, "    expr: ", r.orig_expr)
            r.data === nothing || println(io, "    got:  ", r.data)
        else
            firstline = split(string(r.value), '\n'; limit = 2)[1]
            println(io, "    expr: ", r.orig_expr)
            println(io, "    err:  ", firstline)
        end
    end
end
println("failing-assert detail: ", abspath(LOGFILE))
