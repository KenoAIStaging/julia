# Cold ANSI-write walk benchmark (wave 11): the unified driver over the
# StyledStrings styled-print compile set, in a cold-cache process — the
# `_ansi_writer` first-compile chain whose CI-less callee graph drives the
# recursive inline-cost towers (scratchpad/wave10h/img_workload.log).
#
#   ./usr/bin/julia --startup-file=no --compiled-modules=no \
#       Compiler/bench/unified_ansiwalk.jl
#
# Same harness shape as unified_coldwalk.jl (see its header for the
# methodology); the target list comes from
#
#   ./usr/bin/julia --startup-file=no --pkgimages=no \
#       --trace-compile=trace.jl -e 'using StyledStrings;
#           io = IOContext(IOBuffer(), :color => true);
#           print(io, styled"{red:hello} {(foreground=blue):world $(1+2)}");
#           printstyled(io, "x"; color = :green, bold = true);
#           print(io, styled"{bold:{yellow:nested} tail}")'
#
# Knobs: ANSIWALK_BUDGET (seconds, default 480), COLDWALK_CI_SERVE=0,
# COLDWALK_COST_PRICING=optimize|depth1|stmtwalk (as in unified_coldwalk.jl).

using StyledStrings  # the workload subject: loaded (cold) before resolving sigs

pushfirst!(LOAD_PATH, joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))
# UNIFIED_LOAD_OVERLAY: a directory whose package entries (UnifiedIR, ...)
# take priority — pins a source snapshot when the working tree is in flux
haskey(ENV, "UNIFIED_LOAD_OVERLAY") && pushfirst!(LOAD_PATH, ENV["UNIFIED_LOAD_OVERLAY"])
import Compiler
const U = Compiler.load_unified!()
const CC = Compiler

include(joinpath(@__DIR__, "ansiwalk_sigs.jl"))  # ANSI_SIGS

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

# the hook-shaped driver entry, minus the JIT closure
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

const BUDGET = parse(Float64, get(ENV, "ANSIWALK_BUDGET", "480"))
if get(ENV, "COLDWALK_CI_SERVE", "1") == "0"
    U.CI_SERVE_ENABLED[] = false
    U.CI_COST_ENABLED[] = false
end
if haskey(ENV, "COLDWALK_COST_PRICING")
    U.COST_PRICING[] = Symbol(ENV["COLDWALK_COST_PRICING"])
end
println("ci_serve=", U.CI_SERVE_ENABLED[], " memo=", U.DRIVER_MEMO_ENABLED[],
        " pricing=", U.COST_PRICING[])
U.reset_pipeline_stats!()
U.reset_driver_phases!()

resolved = Tuple{String,Core.MethodInstance}[]
skipped = 0
for s in ANSI_SIGS
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
println("== ansiwalk: ", completed, "/", length(resolved), " targets in ",
        round(total; digits = 2), "s ==")
U.print_pipeline_stats()
U.print_driver_phases()
