# Fresh-process demo of the REAL unified pipeline as the runtime compiler
# (COMPILER-PORT-PLAN A1/A2 acceptance):
#
#   ./usr/bin/julia --startup-file=no Compiler/bench/unified_driver_demo.jl
#
# Protocol: the capture-zoo workload (UnifiedIR/demo/capture_zoo.jl's
# definitions) plus sum/sort/string ops run twice — once compiled by the
# STOCK compiler (before activation; the expected values), once as fresh
# methods compiled under `Unified.activate!()` (jl_typeinf_func routed
# through the unified driver, per-body stock fallback). Every outcome must
# match; the pipeline ledger (bodies through unified vs fallbacks by
# reason) prints at the end. Exits nonzero on any mismatch.

pushfirst!(LOAD_PATH, joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))
import Compiler
const U = Compiler.load_unified!()

# the capture-zoo definitions (see UnifiedIR/demo/capture_zoo.jl)
const ZOO = raw"""
function abmult(r::Int)                 # julia#15276: value capture after if-join
    if r < 0
        r = -r
    end
    f = x -> x * r
    return f
end
function zoo1(c)                        # assigned in both arms before capture
    local x
    if c; x = 1; else; x = 2; end
    cl = () -> x
    return cl()
end
function zoo2(a)                        # try/catch definite assignment
    local x
    try
        x = sqrt(a)
    catch
        x = -1.0
    end
    cl = () -> x
    return cl()
end
function zoo3()                         # store before first use: sunk creation
    x = 1
    f = () -> x
    x = 2
    return f()
end
function zoo3b()                        # use blocks the sink: shared capture
    x = 1
    f = () -> x
    a = f()
    x = 2
    return (a, f())
end
function zoo4(n)                        # loop creation: one shared location
    fs = Any[]
    local x::Int = 0
    for i in 1:n
        push!(fs, () -> x)
        x = i
    end
    return Any[f() for f in fs]
end
function zoo5()                         # closure writing its capture
    local x::Int = 0
    inc = () -> (x = x + 1)
    inc(); inc(); inc()
    return x
end
function zoo5b()
    x = 0
    inc = () -> (x = x + 1)
    inc(); inc(); inc()
    return x
end
function zoo6(c)                        # maybe-undef capture: error at USE
    local x
    if c; x = 1; end
    f = () -> x
    return f
end
function zoo7(c)                        # comprehension capture
    if c; x = 1; else; x = 2; end
    return [x + i for i in 1:3]
end
sumto(n) = begin s = 0; i = 1; while i <= n; s += i; i += 1; end; s end
sortjoin(v) = join(sort(v), "-")
banger(s, n) = s * "!" ^ n
revup(s) = uppercase(reverse(s))
mixed(a) = (a + 1.0) / (a + 2) - a * 3.0 + a ÷ 2
trycatch(x) = try; div(10, x); catch; -1; end
"""

const CASES = [
    ("abmult(-3)(2)", M -> M.abmult(-3)(2)),
    ("abmult(5)(7)",  M -> M.abmult(5)(7)),
    ("zoo1(true)",    M -> M.zoo1(true)),
    ("zoo1(false)",   M -> M.zoo1(false)),
    ("zoo2(4.0)",     M -> M.zoo2(4.0)),
    ("zoo2(-4.0)",    M -> M.zoo2(-4.0)),
    ("zoo3()",        M -> M.zoo3()),
    ("zoo3b()",       M -> M.zoo3b()),
    ("zoo4(3)",       M -> M.zoo4(3)),
    ("zoo5()",        M -> M.zoo5()),
    ("zoo5b()",       M -> M.zoo5b()),
    ("zoo6(false)()", M -> M.zoo6(false)()),
    ("zoo6(true)()",  M -> M.zoo6(true)()),
    ("zoo7(true)",    M -> M.zoo7(true)),
    ("zoo7(false)",   M -> M.zoo7(false)),
    ("sumto(100)",    M -> M.sumto(100)),
    ("sortjoin",      M -> M.sortjoin([3, 1, 2])),
    ("banger",        M -> M.banger("hey", 3)),
    ("revup",         M -> M.revup("abc")),
    ("mixed(7)",      M -> M.mixed(7)),
    ("trycatch(5)",   M -> M.trycatch(5)),
    ("trycatch(0)",   M -> M.trycatch(0)),
]

outcome(f) = try
    (:ok, f())
catch e
    (:err, sprint(showerror, e))
end

# 1. stock-computed expected values (before any activation)
module StockZoo end
Base.include_string(StockZoo, ZOO, "zoo.jl")
expected = Any[outcome(() -> run(StockZoo)) for (_, run) in CASES]

# 2. the real pipeline, globally
t0 = time()
U.activate!()
println("activate!(:native): ", round(time() - t0; digits = 1), "s")
U.reset_pipeline_stats!()

# fresh methods, compiled under the flipped runtime
module UnifiedZoo end
t0 = time()
Base.include_string(UnifiedZoo, ZOO, "zoo.jl")
got = Any[outcome(() -> run(UnifiedZoo)) for (_, run) in CASES]
println("workload under unified runtime: ", round(time() - t0; digits = 1), "s")

stats = U.pipeline_stats()
U.deactivate!()

# 3. the differential + the ledger
println("\n== execution differential (stock-compiled vs unified-compiled) ==")
ndiff = 0
for (i, (label, _)) in enumerate(CASES)
    ok = isequal(expected[i], got[i])
    ok || (global ndiff += 1)
    println(rpad(label, 16), ok ? "MATCH  " : "DIFF   ", repr(got[i]),
            ok ? "" : "   expected: " * repr(expected[i]))
end

println("\n== pipeline ledger (bodies through unified vs fallbacks) ==")
total = stats.unified + sum(values(stats.fallbacks); init = 0)
println("unified: ", stats.unified, " / ", total, " inference requests")
for (reason, n) in sort!(collect(stats.fallbacks); by = last, rev = true)
    println("  fallback ", rpad(String(reason), 24), " ", n)
end
if stats.last_error !== nothing
    reason, mi, err = stats.last_error
    println("  last error-class fallback: ", reason, " at ", mi)
    println("    ", sprint(showerror, err)[1:min(end, 200)])
end

if ndiff == 0 && stats.unified >= 1
    println("\nOK: all ", length(CASES), " outcomes match; ",
            stats.unified, " bodies compiled by the unified pipeline")
    exit(0)
else
    println("\nFAIL: ", ndiff, " mismatches (unified bodies: ", stats.unified, ")")
    exit(1)
end
