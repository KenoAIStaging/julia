# Fresh-process demo of the REAL unified pipeline as the runtime compiler
# (COMPILER-PORT-PLAN A1/A2 acceptance):
#
#   ./usr/bin/julia --startup-file=no Compiler/bench/unified_driver_demo.jl
#
# Protocol: the capture-zoo workload (UnifiedIR/demo/capture_zoo.jl's
# definitions) plus sum/sort/string ops run compiled by the STOCK compiler
# (before activation; the expected values), then TWICE as fresh methods
# compiled under `Unified.activate!()` (jl_typeinf_func routed through the
# unified driver, per-body stock fallback), with a per-pass ledger. Pass 1
# includes the driver compiling its own code (reentrant recursion burn-in);
# by pass 2 that code is cached, so its ledger must be reentrant-quiet.
# Every outcome must match; exits nonzero on any mismatch.

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

function print_ledger(stats)
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
end

# fresh methods, compiled under the flipped runtime — twice: pass 1 pays the
# driver's self-compilation (reentrant recursion), pass 2 must not
function run_pass(n)
    U.reset_pipeline_stats!()
    mod = Module(Symbol(:UnifiedZoo, n))
    t0 = time()
    Base.include_string(mod, ZOO, "zoo.jl")
    # the zoo methods are newer than this frame's world (include_string just
    # defined them): each case must run at the latest world
    got = Any[outcome(() -> Base.invokelatest(run, mod)) for (_, run) in CASES]
    println("workload pass ", n, " under unified runtime: ",
            round(time() - t0; digits = 1), "s")
    return got, U.pipeline_stats()
end
got1, stats1 = run_pass(1)
got2, stats2 = run_pass(2)
U.deactivate!()

# 3. the differential + the per-pass ledgers
ndiff = 0
for (passno, got) in ((1, got1), (2, got2))
    println("\n== execution differential, pass ", passno,
            " (stock-compiled vs unified-compiled) ==")
    for (i, (label, _)) in enumerate(CASES)
        ok = isequal(expected[i], got[i])
        ok || (global ndiff += 1)
        println(rpad(label, 16), ok ? "MATCH  " : "DIFF   ", repr(got[i]),
                ok ? "" : "   expected: " * repr(expected[i]))
    end
end

println("\n== pipeline ledger, pass 1 (includes driver self-compilation) ==")
print_ledger(stats1)
println("\n== pipeline ledger, pass 2 (driver code already compiled) ==")
print_ledger(stats2)

reentrant2 = get(stats2.fallbacks, :reentrant_self, 0) +
             get(stats2.fallbacks, :reentrant_depth, 0)
println("\nreentrant declines: pass 2 = ", reentrant2)

if ndiff == 0 && stats1.unified >= 1 && stats2.unified >= 1
    println("\nOK: all ", length(CASES), " outcomes match on both passes; unified ",
            stats1.unified, " (pass 1) / ", stats2.unified, " (pass 2)")
    exit(0)
else
    println("\nFAIL: ", ndiff, " mismatches (unified pass1=", stats1.unified,
            " pass2=", stats2.unified, ")")
    exit(1)
end
