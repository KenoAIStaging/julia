# Bootstrap-dialect support (COMPILER-PORT-PLAN C2; BOOTSTRAP-SUBSET-NOTES).
#
# UnifiedIR's compiler-needed core (everything except the tree/print/parse/
# interp/testdialect layer, which `load_syntax!` finishes under full Base)
# loads during the basecompiler bootstrap stage, where only the partial Base
# of COMPILER_SRCS exists — no Dict/Set, no @enum, no ReentrantLock, no Base
# sort/reduce keyword forms, and no runtime `string(...)`/interpolation. The
# core therefore:
#   * uses IdDict/IdSet/BitSet — egal keying, which is also stock-Compiler
#     hygiene (compiler data structures must not call user-extensible
#     hash/isequal mid-inference); every key type used (Int/Int32, Symbol,
#     StmtId/RegionId, GlobalRef, bits tuples) is egal-identical to its
#     isequal behavior,
#   * declares enum-like constants as UInt8-wrapper structs (`===`-comparable
#     bits types with name-printing `show`, replacing `@enum`),
#   * guards the kind registry through a lock shim (`_registry_locked`,
#     kinds.jl) that is upgraded to a real `ReentrantLock` post-Base,
#   * uses the private sort/reduce helpers below,
#   * interpolates error-path messages lazily (`LazyString("...")`).

"Stable merge sort with `by`/`rev`, matching `Base.sort!(v; by, rev)`
semantics (Base's keyword `sort!` is unavailable at basecompiler)."
function _sort!(v::Vector; by = identity, rev::Bool = false)
    n = length(v)
    n <= 1 && return v
    _msort!(v, similar(v), 1, n, by, rev)
    return v
end

@inline _sort_lt(by, rev::Bool, @nospecialize(a), @nospecialize(b)) =
    rev ? isless(by(b), by(a)) : isless(by(a), by(b))

function _msort!(v::Vector, t::Vector, lo::Int, hi::Int, by, rev::Bool)
    if hi - lo < 32
        # insertion sort for small runs (stable)
        for i in (lo + 1):hi
            x = v[i]
            j = i
            while j > lo && _sort_lt(by, rev, x, v[j - 1])
                v[j] = v[j - 1]
                j -= 1
            end
            v[j] = x
        end
        return nothing
    end
    mid = (lo + hi) >>> 1
    _msort!(v, t, lo, mid, by, rev)
    _msort!(v, t, mid + 1, hi, by, rev)
    _sort_lt(by, rev, v[mid + 1], v[mid]) || return nothing  # already in order
    for i in lo:mid
        t[i] = v[i]
    end
    i, j, k = lo, mid + 1, lo
    while i <= mid && j <= hi
        if _sort_lt(by, rev, v[j], t[i])
            v[k] = v[j]
            j += 1
        else
            v[k] = t[i]
            i += 1
        end
        k += 1
    end
    while i <= mid
        v[k] = t[i]
        i += 1
        k += 1
    end
    return nothing
end

"`minimum(f, xs)` over a nonempty collection (the reduce family is post-Base)."
function _minimum(f, xs)
    y = iterate(xs)
    y === nothing && throw(ArgumentError("_minimum over an empty collection"))
    v, st = y
    m = f(v)
    while true
        y = iterate(xs, st)
        y === nothing && return m
        v, st = y
        fv = f(v)
        isless(fv, m) && (m = fv)
    end
end

"`minimum(xs; init)` (identity form; returns `init` when empty)."
function _minimum_init(xs, init)
    m = init
    for x in xs
        isless(x, m) && (m = x)
    end
    return m
end

"`count(f, xs)`."
function _count(f, xs)
    n = 0
    for x in xs
        f(x) && (n += 1)
    end
    return n
end

"`searchsortedfirst(v, x; by)`: index of the first element whose `by`-key is
not less than `by(x)`; `length(v) + 1` when every key is."
function _searchsortedfirst_by(v::Vector, x, by)
    kx = by(x)
    lo, hi = 1, length(v) + 1
    while lo < hi
        m = (lo + hi) >>> 1
        if isless(by(v[m]), kx)
            lo = m + 1
        else
            hi = m
        end
    end
    return lo
end
