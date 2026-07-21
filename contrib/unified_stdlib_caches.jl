# This file is a part of Julia. License is MIT: https://julialang.org/license

# Build stdlib package-image caches under the *unified* sysimage
# (sys-unified.$(SHLIB_EXT)) so every cache is compiled by the UnifiedIR
# compiler port baked into that image (Compiler/src/unified/bootstrap_driver.jl).
#
# Invoked by pkgimage.mk's `release-unified`/`debug-unified` targets with the
# same environment as the stock cache build: JULIA_DEPOT_PATH pinned to the
# build tree's bundled depot ($(build_prefix)/share/julia), JULIA_LOAD_PATH
# '@stdlib', JULIA_CPU_TARGET "sysimage".  The caches land in
# usr/share/julia/compiled next to the stock ones and coexist with them:
# the cache filename slug hashes JLOptions().image_file (sys.so vs
# sys-unified.so pick different slots) and loading validates the recorded
# Base build_id, so each sysimage only ever accepts its own set.
#
# Differences from the stock `precompilepkgs` invocation — deliberate, the
# unified pipeline is still experimental:
#   * per-package timeout (JULIA_UNIFIED_PRECOMPILE_TIMEOUT seconds, default
#     3600): on expiry the precompile worker gets SIGTERM, 60s later SIGKILL
#   * failure tolerance: a failing stdlib is logged and its dependents are
#     marked skipped; everything else continues.  The final summary lists
#     every non-ok package; the exit status is nonzero if anything failed,
#     so pkgimage.mk only writes the stamp after a full pass
#   * ARGS may name root packages, restricting the build to their dependency
#     closure (subset validation; pkgimage.mk skips the stamp in that mode)
#   * only the primary cache config (debug_level=2, opt_level=3) is built;
#     the check_bounds=1 test-suite config is left to a follow-up
#   * package extensions are not precompiled (none exist in the Test closure;
#     Pkg/Statistics extensions fall back to runtime precompilation)
#
# Parallelism: JULIA_UNIFIED_PRECOMPILE_JOBS concurrent packages (default
# min(4, CPU_THREADS)), scheduled in dependency order so a package's deps
# are always cached before its worker starts (no nested precompile cascades).

const CACHEFLAGS = Base.CacheFlags(debug_level=2, opt_level=3)
const TIMEOUT = parse(Float64, get(ENV, "JULIA_UNIFIED_PRECOMPILE_TIMEOUT", "3600"))
const JOBS = parse(Int, get(ENV, "JULIA_UNIFIED_PRECOMPILE_JOBS",
                            string(min(4, Sys.CPU_THREADS))))

const t_start = time()
stamp() = string("[", lpad(round(time() - t_start, digits=1), 7), "s]")
say(args...) = (println(stderr, stamp(), " unified-caches: ", args...); flush(stderr))

let img = unsafe_string(Base.JLOptions().image_file)
    contains(img, "sys-unified") ||
        @warn "unified_stdlib_caches.jl expects to run under sys-unified" image=img
    say("image: ", img)
    say("depot: ", join(Base.DEPOT_PATH, ':'))
    say("jobs: ", JOBS, "  per-package timeout: ", TIMEOUT, "s")
end

# ---- dependency graph from the stdlib environment's manifest ----
manifest = Base.parsed_toml(joinpath(dirname(@__DIR__), "stdlib", "Manifest.toml"))
graph = Dict{String,Vector{String}}()     # name => manifest deps (names)
pkgids = Dict{String,Base.PkgId}()
for (name, entries) in manifest["deps"]::Dict{String,Any}
    entry = only(entries)::Dict{String,Any}
    pkgids[name] = Base.PkgId(Base.UUID(entry["uuid"]::String), name)
    graph[name] = String[d::String for d in get(entry, "deps", Any[])]
end

roots = isempty(ARGS) ? collect(keys(graph)) : copy(ARGS)
selected = Set{String}()
let stack = copy(roots)
    while !isempty(stack)
        n = pop!(stack)
        n in selected && continue
        haskey(graph, n) || error("package $n not found in stdlib/Manifest.toml")
        push!(selected, n)
        append!(stack, graph[n])
    end
end

insysimg = Set{String}(n for n in selected if Base.in_sysimage(pkgids[n]))
work = sort!(collect(setdiff(selected, insysimg)))
say(length(selected), " packages in closure; ", length(insysimg),
    " already in the sysimage; ", length(work), " to cache")

# ---- scheduler ----
status = Dict{String,Symbol}(n => :pending for n in work)   # :pending/:running/
elapsed = Dict{String,Float64}()                            # :cached/:ok/:failed/
failmsg = Dict{String,String}()                             # :timeout/:skipped
active = Ref(0)

satisfied(d) = !haskey(status, d) || status[d] in (:ok, :cached)
depfailed(d) = haskey(status, d) && status[d] in (:failed, :timeout, :skipped)

function compile_one(name::String, pkg::Base.PkgId)
    if Base.isprecompiled(pkg)
        return :cached, 0.0, ""
    end
    sigch = Channel{Int32}(4)
    t0 = time()
    ct = @async Base.compilecache(pkg; cacheflags=CACHEFLAGS, signal_channel=sigch)
    term_at = 0.0
    timedout = false
    while !istaskdone(ct)
        if !timedout && time() - t0 > TIMEOUT
            say(name, ": TIMEOUT after ", round(time() - t0, digits=1),
                "s, sending SIGTERM to worker")
            try put!(sigch, Int32(15)) catch end
            term_at = time()
            timedout = true
        elseif timedout && term_at > 0.0 && time() - term_at > 60
            say(name, ": worker still alive 60s after SIGTERM, sending SIGKILL")
            try put!(sigch, Int32(9)) catch end
            term_at = 0.0
        end
        sleep(1)
    end
    close(sigch)
    dt = time() - t0
    timedout && return :timeout, dt, "timed out after $(TIMEOUT)s"
    if istaskfailed(ct)
        msg = try
            sprint(showerror, ct.result)
        catch
            "unknown error"
        end
        length(msg) > 500 && (msg = msg[1:thisind(msg, 500)] * " …")
        return :failed, dt, msg
    end
    return :ok, dt, ""
end

while true
    launched = false
    for n in work
        status[n] === :pending || continue
        active[] >= JOBS && break
        deps = graph[n]
        if any(depfailed, deps)
            status[n] = :skipped
            failmsg[n] = "dependency failed: " *
                join((d for d in deps if depfailed(d)), ", ")
            say(n, ": SKIPPED (", failmsg[n], ")")
            launched = true
        elseif all(satisfied, deps)
            status[n] = :running
            active[] += 1
            launched = true
            let n = n
                @async begin
                    st, dt, msg = try
                        compile_one(n, pkgids[n])
                    catch err
                        :failed, 0.0, sprint(showerror, err)
                    end
                    status[n] = st
                    elapsed[n] = dt
                    isempty(msg) || (failmsg[n] = msg)
                    say(n, ": ", uppercase(string(st)),
                        st === :cached ? "" : string(" (", round(dt, digits=1), "s)"),
                        isempty(msg) ? "" : string(" — ", msg))
                    active[] -= 1
                end
            end
        end
    end
    if !any(st -> st in (:pending, :running), values(status))
        break
    end
    launched || sleep(0.5)
end

# ---- summary ----
counts = Dict{Symbol,Int}()
for st in values(status)
    counts[st] = get(counts, st, 0) + 1
end
say("summary: ", join((string(k, "=", v) for (k, v) in sort!(collect(counts))), " "))
bad = sort!([n for (n, st) in status if st in (:failed, :timeout, :skipped)])
if isempty(bad)
    say("all ", length(work), " stdlib caches present for the unified image")
    exit(0)
else
    for n in bad
        say("  ", n, " [", status[n], "] ", get(failmsg, n, ""))
    end
    say("FAILURES: ", join(bad, ", "))
    exit(1)
end
