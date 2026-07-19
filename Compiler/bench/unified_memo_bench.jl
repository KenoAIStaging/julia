# Repeat-inference microbench for the driver's cross-request memo (A6):
#
#   ./usr/bin/julia --startup-file=no Compiler/bench/unified_memo_bench.jl
#
# Protocol: for each target, one warmup driver pass (compiles the unified
# pipeline itself through stock; not measured), then N full driver passes
# (fresh UInferState/UEdges per pass — the driver's request shape) with the
# cross-request memo DISABLED, then N with it ENABLED. Each pass includes
# entry conversion, inference, optimization and the typed exit; the memo
# serves inference's callee tree by fact replay, so the delta isolates the
# cross-body re-inference cost. Memo counters print per mode.

pushfirst!(LOAD_PATH, joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))
import Compiler
const U = Compiler.load_unified!()
const CC = Compiler

interp() = CC.NativeInterpreter(Base.get_world_counter())
mi(f, args...) = U.lookup_method_instance(f, args...)

sortjoin(v) = join(sort(v), "-")
mixed(a) = (a + 1.0) / (a + 2) - a * 3.0 + a ÷ 2
sumto(n) = begin s = 0; i = 1; while i <= n; s += i; i += 1; end; s end
printer(io, x) = print(io, "x = ", x, "\n")

const TARGETS = [
    ("sortjoin", () -> mi(sortjoin, [3, 1, 2])),
    ("mixed",    () -> mi(mixed, 7)),
    ("sumto",    () -> mi(sumto, 100)),
    ("printer",  () -> mi(printer, IOBuffer(), 1)),
]
const N = 10

function run_mode(enabled::Bool)
    U.DRIVER_MEMO_ENABLED[] = enabled
    U.reset_driver_memo!()
    U.reset_pipeline_stats!()
    total = 0.0
    for (name, getmi) in TARGETS
        m = getmi()
        ts = Float64[]
        for _ in 1:N
            push!(ts, @elapsed(U.driver_infer(interp(), m)))
        end
        total += sum(ts)
        println(rpad(name, 10), " memo=", rpad(string(enabled), 6),
                " total=", rpad(round(sum(ts); digits = 3), 7),
                " min=", rpad(round(minimum(ts); digits = 4), 7),
                " last=", round(ts[end]; digits = 4))
    end
    println("mode memo=", enabled, " grand total ", round(total; digits = 3), "s  ",
            U.pipeline_stats().memo)
    return total
end

# warmup: the unified pipeline compiles (stock) on the first pass
U.DRIVER_MEMO_ENABLED[] = false
for (_, getmi) in TARGETS
    U.driver_infer(interp(), getmi())
end

println("== repeat inference, N=", N, " passes per target ==")
t_off = run_mode(false)
t_on = run_mode(true)
println("\nmemo off ", round(t_off; digits = 3), "s -> memo on ", round(t_on; digits = 3),
        "s  (", round(t_off / t_on; digits = 2), "x)")
U.DRIVER_MEMO_ENABLED[] = true
