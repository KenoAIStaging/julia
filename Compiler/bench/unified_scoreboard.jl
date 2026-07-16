# This file is a part of Julia. License is MIT: https://julialang.org/license
#
# The Phase-B0 all-groups scoreboard (COMPILER-PORT-PLAN B0): run every
# Compiler testgroup serially in a subprocess twice — hook-off (stock) and
# hook-on (`JULIA_UNIFIED_COMPILER=1`, the Compiler/test/setup_Compiler.jl
# knob that enables the unified pipeline after `@activate Compiler`) — parse
# each run's final testset tallies (pass/fail/error/broken) plus the unified
# driver's fallback ledger, and emit a per-group markdown table to stdout and
# to a results file.
#
# Usage (from the julia checkout root):
#     ./usr/bin/julia --startup-file=no Compiler/bench/unified_scoreboard.jl \
#         [--groups=effects,ssair,...] [--modes=stock,unified] \
#         [--timeout=1800] [--logdir=DIR] [--out=UNIFIED-SCOREBOARD.md]
#
# Each (group, mode) run writes <logdir>/<group>.<mode>.log (the full test
# output) and <logdir>/<group>.<mode>.result (parsed key=value summary,
# including the git sha the run executed at). The emitted table covers every
# group with .result files present, so partial (re)runs accumulate into one
# table. special_loading (Base.Compiler-only) and newinterp (helper file, no
# tests of its own) are not scoreboard groups. Anything below a
# `<!-- MANUAL ANALYSIS BELOW -->` marker in --out survives regeneration.

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const JULIA = joinpath(Sys.BINDIR, "julia")
const SKIPPED_GROUPS = ("special_loading", "newinterp")
const MANUAL_MARKER = "<!-- MANUAL ANALYSIS BELOW -->"

function parse_opts(args)
    opts = Dict{String,String}()
    for a in args
        m = match(r"^--(groups|modes|timeout|logdir|out)=(.+)$", a)
        if m === nothing
            println(stderr, "unrecognized argument: ", a)
            println(stderr, "usage: julia Compiler/bench/unified_scoreboard.jl ",
                    "[--groups=a,b] [--modes=stock,unified] [--timeout=SECS] ",
                    "[--logdir=DIR] [--out=FILE]")
            exit(2)
        end
        opts[m[1]] = m[2]
    end
    return opts
end

all_groups() = [g for g in readlines(joinpath(ROOT, "Compiler", "test", "testgroups"))
                if !isempty(g) && g ∉ SKIPPED_GROUPS]

# The per-group subprocess: the canonical serial include pattern, with the
# outermost TestSetException absorbed so the machine-readable SCOREBOARD
# epilogue always prints; the hook-on variant appends the pipeline ledger
# (never by editing test files — the knob lives in setup_Compiler.jl).
function subprocess_code(group::AbstractString, mode::AbstractString)
    io = IOBuffer()
    print(io, """
        using Test
        const counts = try
            ts = @testset "$group" begin
                include("Compiler/test/$group.jl")
            end
            tc = Test.get_test_counts(ts)
            (; pass = tc.passes + tc.cumulative_passes,
               fail = tc.fails + tc.cumulative_fails,
               error = tc.errors + tc.cumulative_errors,
               broken = tc.broken + tc.cumulative_broken)
        catch err
            err isa Test.TestSetException || rethrow()
            (; pass = err.pass, fail = err.fail, error = err.error, broken = err.broken)
        end
        println("SCOREBOARD pass=", counts.pass, " fail=", counts.fail,
                " error=", counts.error, " broken=", counts.broken)
        """)
    if mode == "unified"
        print(io, """
            let C = Base.REFLECTION_COMPILER[]
                if C !== nothing && isdefined(C, :Unified)
                    st = C.Unified.pipeline_stats()
                    println("PIPELINE unified=", st.unified,
                            " fallbacks=", sum(values(st.fallbacks); init = 0))
                    for (r, n) in sort!(collect(st.fallbacks); by = last, rev = true)
                        println("PIPELINE_FALLBACK ", r, " ", n)
                    end
                else
                    println("PIPELINE not-active (group never activated the stdlib Compiler)")
                end
            end
            """)
    end
    return String(take!(io))
end

function parse_log(logfile)
    text = isfile(logfile) ? read(logfile, String) : ""
    counts = nothing
    for m in eachmatch(r"^SCOREBOARD pass=(\d+) fail=(\d+) error=(\d+) broken=(\d+)"m, text)
        counts = (; pass = parse(Int, m[1]), fail = parse(Int, m[2]),
                    error = parse(Int, m[3]), broken = parse(Int, m[4]))
    end
    ledger = nothing
    for m in eachmatch(r"^PIPELINE unified=(\d+) fallbacks=(\d+)"m, text)
        ledger = (; unified = parse(Int, m[1]), fallbacks = parse(Int, m[2]))
    end
    fallbacks = Tuple{String,Int}[]
    for m in eachmatch(r"^PIPELINE_FALLBACK (\S+) (\d+)"m, text)
        push!(fallbacks, (String(m[1]), parse(Int, m[2])))
    end
    return counts, ledger, fallbacks
end

function write_result(path, group, mode, status, exitcode, termsignal, secs, sha,
                      counts, ledger, fallbacks)
    open(path, "w") do io
        println(io, "group=", group)
        println(io, "mode=", mode)
        println(io, "status=", status)
        println(io, "exit=", exitcode)
        println(io, "signal=", termsignal)
        println(io, "secs=", round(secs; digits = 1))
        println(io, "sha=", sha)
        if counts !== nothing
            println(io, "pass=", counts.pass)
            println(io, "fail=", counts.fail)
            println(io, "error=", counts.error)
            println(io, "broken=", counts.broken)
        end
        if ledger !== nothing
            println(io, "ledger_unified=", ledger.unified)
            println(io, "ledger_fallbacks=", ledger.fallbacks)
        end
        for (r, n) in fallbacks
            println(io, "fallback ", r, " ", n)
        end
    end
end

function read_result(path)
    isfile(path) || return nothing
    kv = Dict{String,String}()
    fallbacks = Tuple{String,Int}[]
    for line in eachline(path)
        if startswith(line, "fallback ")
            parts = split(line)
            length(parts) == 3 && push!(fallbacks, (String(parts[2]), parse(Int, parts[3])))
        else
            m = match(r"^([a-z_]+)=(.*)$", line)
            m === nothing || (kv[m[1]] = m[2])
        end
    end
    return (kv, fallbacks)
end

function run_one(group, mode, timeout_s, logdir)
    logfile = joinpath(logdir, "$group.$mode.log")
    sha = readchomp(Cmd(`git rev-parse HEAD`; dir = ROOT))
    code = subprocess_code(group, mode)
    envval = mode == "unified" ? "1" : "0"
    cmd = addenv(Cmd(`$JULIA --startup-file=no --project=Compiler -e $code`; dir = ROOT),
                 "JULIA_UNIFIED_COMPILER" => envval)
    println("[$group.$mode] start (timeout=$(timeout_s)s, sha=$(first(sha, 10))) → $logfile")
    flush(stdout)
    t0 = time()
    proc = run(pipeline(cmd; stdout = logfile, stderr = logfile); wait = false)
    status = "ok"
    while process_running(proc)
        if time() - t0 > timeout_s
            status = "timeout"
            kill(proc)                     # SIGTERM first
            deadline = time() + 15
            while process_running(proc) && time() < deadline
                sleep(0.5)
            end
            process_running(proc) && kill(proc, Base.SIGKILL)
            break
        end
        sleep(1.0)
    end
    wait(proc)
    secs = time() - t0
    counts, ledger, fallbacks = parse_log(logfile)
    if status == "ok" && counts === nothing
        status = "crash"
    end
    write_result(joinpath(logdir, "$group.$mode.result"), group, mode, status,
                 proc.exitcode, proc.termsignal, secs, sha, counts, ledger, fallbacks)
    println("[$group.$mode] $status exit=$(proc.exitcode) sig=$(proc.termsignal) ",
            "$(round(secs; digits = 1))s ",
            counts === nothing ? "(no testset summary)" :
                "pass=$(counts.pass) fail=$(counts.fail) error=$(counts.error) broken=$(counts.broken)",
            ledger === nothing ? "" : " | unified=$(ledger.unified) fallbacks=$(ledger.fallbacks)")
    flush(stdout)
    return nothing
end

# --------------------------------------------------------------------------
# Markdown emission (covers every group with accumulated .result files)
# --------------------------------------------------------------------------

function counts_cell(r)
    r === nothing && return "—"
    kv, _ = r
    status = get(kv, "status", "?")
    if haskey(kv, "pass")
        s = string(kv["pass"], "/", kv["fail"], "/", kv["error"], "/", kv["broken"])
        status == "ok" || (s = string(s, " [", status, "]"))
        return s
    end
    status == "timeout" && return string("TIMEOUT(", get(kv, "secs", "?"), "s)")
    return string("CRASH(exit=", get(kv, "exit", "?"), ",sig=", get(kv, "signal", "?"), ")")
end

function delta_cell(rs, ru)
    (rs === nothing || ru === nothing) && return "—"
    ks, _ = rs
    ku, _ = ru
    (haskey(ks, "pass") && haskey(ku, "pass")) || return "—"
    dp = parse(Int, ku["pass"]) - parse(Int, ks["pass"])
    dfe = (parse(Int, ku["fail"]) + parse(Int, ku["error"])) -
          (parse(Int, ks["fail"]) + parse(Int, ks["error"]))
    (dp == 0 && dfe == 0) && return "±0"
    return string(dp > 0 ? "+" : "", dp, "p ", dfe > 0 ? "+" : "", dfe, "f+e")
end

function ledger_cell(ru)
    ru === nothing && return "—"
    kv, fallbacks = ru
    haskey(kv, "ledger_unified") || return "n/a"
    s = string(kv["ledger_unified"], " / ", kv["ledger_fallbacks"])
    if !isempty(fallbacks)
        top = first(fallbacks, 3)
        s *= " (" * join([string(r, "=", n) for (r, n) in top], ", ")
        length(fallbacks) > 3 && (s *= ", …")
        s *= ")"
    end
    return s
end

function emit_markdown(groups, logdir, outfile, timeout_s)
    rows = Tuple{String,Any,Any}[]
    shas = String[]
    for g in groups
        rs = read_result(joinpath(logdir, "$g.stock.result"))
        ru = read_result(joinpath(logdir, "$g.unified.result"))
        for r in (rs, ru)
            r === nothing && continue
            haskey(r[1], "sha") && push!(shas, r[1]["sha"])
        end
        push!(rows, (g, rs, ru))
    end
    io = IOBuffer()
    println(io, "# UNIFIED-SCOREBOARD — Compiler testgroups, stock vs unified pipeline")
    println(io)
    println(io, "COMPILER-PORT-PLAN Phase B0: each Compiler testgroup runs serially in a")
    println(io, "subprocess twice — hook-off (stock) and hook-on (`JULIA_UNIFIED_COMPILER=1`,")
    println(io, "the `Compiler/test/setup_Compiler.jl` knob: after `@activate Compiler` it runs")
    println(io, "`Compiler.load_unified!()` + `enable_pipeline!()`). Generated by")
    println(io, "`Compiler/bench/unified_scoreboard.jl`; skipped groups: special_loading")
    println(io, "(Base.Compiler-only), newinterp (helper, no tests of its own).")
    println(io)
    println(io, "- generated: ", Libc.strftime("%Y-%m-%d %H:%M:%S %Z", time()))
    println(io, "- julia: ", VERSION, " (", JULIA, ")")
    println(io, "- HEAD at table generation: ", readchomp(Cmd(`git rev-parse HEAD`; dir = ROOT)))
    usha = unique(shas)
    if length(usha) == 1
        println(io, "- every row measured at: ", usha[1])
    elseif !isempty(usha)
        println(io, "- rows measured at MIXED shas (other work landed mid-sweep; per-group")
        println(io, "  shas below and in the `.result` files):")
        for (g, rs, ru) in rows
            gs = unique([r[1]["sha"] for r in (rs, ru) if r !== nothing && haskey(r[1], "sha")])
            isempty(gs) || println(io, "    - ", g, ": ", join(first.(gs, 12), " / "))
        end
    end
    println(io, "- per-run timeout: ", timeout_s, "s; full logs + parsed `.result` files: ", logdir)
    println(io, "- P/F/E/B = pass/fail/error/broken leaf tallies (same accounting as the")
    println(io, "  `Test` summary). Ledger = unified-compiled bodies / stock-fallback bodies")
    println(io, "  (by reason) from `pipeline_stats()`; `n/a` = the group never activates the")
    println(io, "  stdlib Compiler, so the knob is inert there.")
    println(io, "- machine caveat: tarjan's pass total is RANDOMIZED (invariant fuzz loops);")
    println(io, "  its baseline is \"invariants hold + exit 0\", not a stable count.")
    println(io)
    println(io, "| group | stock P/F/E/B | unified P/F/E/B | delta | ledger unified/fallbacks |")
    println(io, "|---|---|---|---|---|")
    for (g, rs, ru) in rows
        println(io, "| ", g, " | ", counts_cell(rs), " | ", counts_cell(ru), " | ",
                delta_cell(rs, ru), " | ", ledger_cell(ru), " |")
    end
    md = String(take!(io))
    tail = string(MANUAL_MARKER, "\n")
    if isfile(outfile)
        old = read(outfile, String)
        idx = findfirst(MANUAL_MARKER, old)
        idx === nothing || (tail = old[first(idx):end])
    end
    write(outfile, string(md, "\n", tail))
    print(md)
    println("\nresults doc: ", outfile)
    return nothing
end

function main()
    opts = parse_opts(ARGS)
    known = all_groups()
    groups = haskey(opts, "groups") ? String.(split(opts["groups"], ",")) : known
    modes = String.(split(get(opts, "modes", "stock,unified"), ","))
    timeout_s = parse(Float64, get(opts, "timeout", "1800"))
    logdir = abspath(get(opts, "logdir", joinpath(pwd(), "unified_scoreboard_logs")))
    outfile = abspath(get(opts, "out", joinpath(pwd(), "UNIFIED-SCOREBOARD.md")))
    mkpath(logdir)
    for g in groups
        g in known || error("unknown group: $g (see Compiler/test/testgroups; " *
                            "special_loading and newinterp are not scoreboard groups)")
    end
    for m in modes
        m in ("stock", "unified") || error("unknown mode: $m (stock|unified)")
    end
    for g in groups, m in modes
        run_one(g, m, timeout_s, logdir)
    end
    emit_markdown(known, logdir, outfile, timeout_s)
end

main()
