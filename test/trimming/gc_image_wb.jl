# This file is a part of Julia. License is MIT: https://julialang.org/license

# Regression test for write barriers on image-resident objects in trimmed
# executables (see #61474 and the corresponding backport).
#
# TABLE below is a let-bound local captured by global methods -- the same shape
# as FileWatching's `let FDWatchers = Vector{Any}()`, re-created here so the
# test does not depend on FileWatching internals. The capture is serialized
# into the compiled image and, in a trimmed executable, is unreachable from the
# GC root set:
#   - there is no module binding for it (and `--trim` strips binding tables
#     anyway), so it is not reachable through any module;
#   - the capturing methods are `@noinline`, so they are compiled as standalone
#     specializations and their method `roots` array (which holds the capture)
#     suppresses global rooting of the value during native code emission
#     (`aot_optimize_roots`), and `--trim`/`--strip-ir` then strips method
#     roots -- leaving TABLE referenced only by native-code gvar slots, which
#     are not GC roots.
#
# Its liveness -- and that of everything it references at runtime -- therefore
# depends entirely on the image-object GC regime from #61474: image objects
# must be loaded pre-marked (`GC_OLD_MARKED | GC_IN_IMAGE`) so their write
# barrier is armed from birth and mutations enroll them in the persistent
# image remset. On builds where image objects load unmarked
# (`GC_OLD | GC_IN_IMAGE`, e.g. 1.12.x and 1.13.0-rc1) the barrier never
# fires for TABLE, the old->young edges created by `push!` (a freshly
# allocated backing Memory and young elements stored into the image-resident
# Vector) are recorded in no remset, the GC sweeps the still-referenced
# memory, the churn below recycles it, and the reads crash (UndefRefError or
# segfault) or return corrupted data instead of printing "survived".
module ImageTable

let TABLE = Vector{Any}()
    global @noinline function remember!(@nospecialize(x))
        push!(TABLE, x)
        return nothing        # TABLE holds the sole reference
    end
    global @noinline table_length() = length(TABLE)
    global @noinline read_table(i::Int) = TABLE[i]
end

end # module

# Keep the mutating and reading code in their own (non-inlined) frames: if it
# inlines into main(), stale GC-frame slots of main can keep the young backing
# Memory reachable across the GC.gc() calls below and mask the bug.
@noinline function fill_batch!()
    for i in 1:100
        ImageTable.remember!(Base.RefValue{Int}(i))
    end
    return nothing
end

@noinline function check_sum()
    s = 0
    for i in 1:ImageTable.table_length()
        r = ImageTable.read_table(i)
        s += (r::Base.RefValue{Int})[]
    end
    return s
end

# allocation churn in the same size classes as TABLE's backing Memory and
# elements, so swept (but still referenced) chunks get recycled and overwritten
@noinline function churn()
    x = Vector{Any}()
    for i in 1:20000
        push!(x, Base.RefValue{Int}(0))
    end
    return length(x)
end

function (@main)(args::Vector{String})::Cint
    for it in 1:20
        fill_batch!()
        GC.gc(it % 3 == 0)   # mix of minor and full collections
        churn()
        GC.gc(false)
        churn()
        s = check_sum()
        if s != it * 5050    # each iteration appends Ref.(1:100)
            print(Core.stderr, "corruption at iteration ")
            print(Core.stderr, it)
            print(Core.stderr, ": got sum ")
            print(Core.stderr, s)
            print(Core.stderr, ", expected ")
            println(Core.stderr, it * 5050)
            return 1
        end
    end
    println(Core.stdout, "survived")
    return 0
end
