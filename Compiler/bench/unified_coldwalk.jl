# Cold whole-workload walk benchmark (wave 9, B1): the unified driver over
# the Test-loading workload's compile set, in a cold-cache process.
#
#   ./usr/bin/julia --startup-file=no --compiled-modules=no \
#       Compiler/bench/unified_coldwalk.jl
#
# `--compiled-modules=no` loads Test (and deps) from SOURCE, so the stdlib
# methods below have no cached CodeInstances — the sys-unified image
# scenario, where Base bodies are baked but every stdlib body compiles
# through the pipeline on first load. The target list is the stock
# compiler's own account of the workload, harvested once via
#
#   ./usr/bin/julia --startup-file=no --pkgimages=no \
#       --trace-compile=trace.jl -e 'using Test; @testset "x" begin @test 1+1==2 end'
#
# filtered to signatures resolvable after `using Test` (Test/Base/Core
# rows). Each target gets one driver pass through the hook-shaped entry
# (`_unified_typeinf` under the task-state discipline `unified_typeinf`
# applies; the stock `add_codeinsts_to_jit!` closure is skipped so the
# numbers isolate PIPELINE compute from LLVM codegen). Driver-published
# CodeInstances stay in the cache across targets, exactly like the image
# walk. A wall-clock budget (COLDWALK_BUDGET seconds, default 240) stops
# the walk early and reports the partial table — cumulative time through
# target #k is comparable across runs either way.

using Test  # the workload subject: must be loaded (cold) before resolving sigs

pushfirst!(LOAD_PATH, joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))
import Compiler
const U = Compiler.load_unified!()
const CC = Compiler

include(joinpath(@__DIR__, "coldwalk_sigs.jl"))  # WORKLOAD_SIGS

function resolve_mi(sigstr::String)
    tt = try
        Core.eval(Main, Meta.parse(sigstr))
    catch
        return nothing
    end
    tt isa Type || return nothing
    world = Base.get_world_counter()
    match = try
        Base._which(tt; world, raise = false)
    catch
        nothing
    end
    match === nothing && return nothing
    return try
        CC.specialize_method(match)
    catch
        nothing
    end
end

# the hook-shaped driver entry, minus the JIT closure (see header)
function drive(mi::Core.MethodInstance)
    interp = CC.NativeInterpreter(Base.get_world_counter())
    dts = U.driver_task_state()
    dts.depth += 1
    push!(dts.inflight, mi)
    try
        return U._unified_typeinf(interp, mi, CC.SOURCE_MODE_ABI)
    finally
        dts.depth -= 1
        delete!(dts.inflight, mi)
    end
end

# warmup: compile the pipeline itself on small bodies (not measured)
let
    f_warm(x) = x + 1 > 2 ? string(x) : "no"
    g_warm(v) = sum(v) ÷ length(v)
    t = @elapsed for (f, a) in ((f_warm, (3,)), (g_warm, ([1, 2],)))
        m = U.lookup_method_instance(f, a...)
        m === nothing || drive(m)
    end
    println("warmup ", round(t; digits = 1), "s")
end

const BUDGET = parse(Float64, get(ENV, "COLDWALK_BUDGET", "240"))
# A/B switches: COLDWALK_CI_SERVE=0 turns the CodeInstance-cache serving
# (inference + cost/effects fast paths) off — the pre-wave-9 behavior
if get(ENV, "COLDWALK_CI_SERVE", "1") == "0"
    U.CI_SERVE_ENABLED[] = false
    U.CI_COST_ENABLED[] = false
end
println("ci_serve=", U.CI_SERVE_ENABLED[], " memo=", U.DRIVER_MEMO_ENABLED[])
U.reset_pipeline_stats!()
U.reset_driver_phases!()

resolved = Tuple{String,Core.MethodInstance}[]
skipped = 0
for s in WORKLOAD_SIGS
    m = resolve_mi(s)
    m === nothing ? (global skipped += 1) : push!(resolved, (s, m))
end
println("targets resolved ", length(resolved), " / skipped ", skipped)

t_walk0 = time()
completed = 0
total = 0.0
for (i, (s, m)) in enumerate(resolved)
    t = @elapsed r = drive(m)
    global total += t
    global completed += 1
    tag = r isa Core.CodeInstance ? "ci" : "--"
    println(rpad(string("#", i), 5), rpad(string(round(t * 1000; digits = 1), "ms"), 12),
            tag, "  ", first(s, 90))
    if time() - t_walk0 > BUDGET
        println("BUDGET EXCEEDED after ", completed, " targets")
        break
    end
end

println()
println("== coldwalk: ", completed, "/", length(resolved), " targets in ",
        round(total; digits = 2), "s ==")
U.print_pipeline_stats()
U.print_driver_phases()
