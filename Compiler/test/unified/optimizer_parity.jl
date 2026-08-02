# Optimizer-parity regressions (wave 4): pass-level fixes verified through
# the real reflection bridge (enable_pipeline! + Base.code_typed), matching
# the stock test-suite predicates. Included from runtests.jl.

module UnifiedOptimizerParityTests

using Test
import Compiler
const OPCC = Compiler
const OPUnified = Compiler.load_unified!()

_isnew(@nospecialize x) = Meta.isexpr(x, :new)
_isreturn(@nospecialize x) = x isa Core.ReturnNode && isdefined(x, :val)
function _iscall(src::Core.CodeInfo, @nospecialize(f), @nospecialize(x))
    Meta.isexpr(x, :call) || return false
    OPCC.singleton_type(OPCC.argextype(x.args[1], src, OPCC.VarState[])) === f
end
# :invoke targets are CodeInstances when the driver's per-pass production
# budget reaches them, MethodInstances otherwise (compiled lazily) — both
# are the same devirtualized shape
function _isinvoke(sym::Symbol, @nospecialize(x))
    Meta.isexpr(x, :invoke) || return false
    t = x.args[1]
    mi = t isa Core.CodeInstance ? t.def : t
    return mi isa Core.MethodInstance && mi.def.name === sym
end

function _code_typed1(f, at)
    (src, _) = only(Base.code_typed(f, at))
    return src::Core.CodeInfo
end

mutable struct OPMutXYZ; x; y; z; end
struct OPImm; x; y; end

op_tupidx(x) = begin a = (x, 2); a[1] end
op_refchain(x) = Base.RefValue{Any}(x)[]              # pinned-sparam ctor inlining
op_refwrap(x) = Ref{Any}(x)[]                         # two-level pinned sparams
op_uninit_line() = begin r = Ref{Any}(); r[] = 42; r[] end
op_uninit_arms(c) = begin
    r = Ref{Any}()
    if c; r[] = 42; else; r[] = 32; end
    r[]
end
op_uninit_nested(c1, c2, x, y, z) = begin
    r = Ref{Any}()
    if c1
        if c2; r[] = x; else; r[] = y; end
    else
        r[] = z
    end
    r[]
end
op_uninit_unsafe(c) = begin
    r = Ref{Any}()
    if c; r[] = 42; end
    r[]
end
op_mut_sroa(x, y, z) = begin m = OPMutXYZ(x, y, z); m.y = 42; (m.x, m.y, m.z) end

@Base.constprop :none @noinline op_split_cov(@nospecialize x::Any) = Base.inferencebarrier(:Any)
@Base.constprop :none @noinline op_split_cov(@nospecialize x::Number) = Base.inferencebarrier(:Number)
@Base.constprop :none @noinline op_split_fb(@nospecialize x::Type) = Base.inferencebarrier(:Type)
@Base.constprop :none @noinline op_split_fb(@nospecialize x::Number) = Base.inferencebarrier(:Number)
@noinline op_split_one(x::Int) = Base.inferencebarrier(x)

op_call_cov(x) = op_split_cov(x)
op_call_fb(x) = op_split_fb(x)
op_call_one(x) = op_split_one(x)
op_call_none() = op_split_one(nothing)

struct OPImmutArms
    x::Union{Nothing, Int, Float64}
end
function op_immut_arms(b, x, y)
    z = b ? OPImmutArms(x) : OPImmutArms(y)
    z.x::Union{Float64, Int}
end
function op_union_tuple_arms(c, x1, x2)
    t = c ? (x1,) : (x2,)
    getfield(t, 1)
end

op_isassigned_sroa(a) = begin
    r = Ref{Any}()
    r[] = a
    isassigned(r) ? r[] : nothing
end
function op_isdefined_dominated()
    a = Ref{Any}()
    setfield!(a, :x, 2)
    invokelatest(identity, a)
    isdefined(a, :x) && return 1.0
    a[]
end

op_identity_splat(t) = (t...,)
function op_apply_type_svec()
    A = (Tuple, Float32)
    B = Tuple{Float32, Float32}
    Core.apply_type(A..., B.types...)
end

struct OPAmbigSR{T}; x::T; end
op_ambf(a::OPAmbigSR, b) = 1
op_ambf(a, b::OPAmbigSR) = 2
op_call_ambig(a::OPAmbigSR{String}, @nospecialize(b)) = op_ambf(a, b)

@eval op_construct_splatnew(T, fields) = $(Expr(:splatnew, :T, :fields))
op_invoke34900(x::Int, y) = x
op_invoke34900(x, y::Int) = y
op_invoke34900(x::Int, y::Int) = invoke(op_invoke34900, Tuple{Int, Any}, x, y)

mutable struct OPTAFoo; x; end
function op_typeassert_elim(a)
    x1 = OPTAFoo(a)
    x2 = OPTAFoo(x1)
    typeassert(x2.x, OPTAFoo).x
end
op_typeassert_keep(a) = (a::Int) + 1

let b = Expr(:block, (:(y += sin($x)) for x in randn(300))...)
    @eval function op_sin_chain()
        y = 0.0
        $b
        y
    end
end

@noinline op_fin_effect(x) =
    Base.@assume_effects :total !:effect_free @ccall jl_(x::Any)::Cvoid
mutable struct OPAllocNoEscape
    function OPAllocNoEscape()
        finalizer(new()) do this
            op_fin_effect(nothing)
        end
    end
end
function op_useless_finalizer()
    x = Ref(1)
    finalizer(x) do x
        nothing
    end
    return x
end

@testset "optimizer parity: wave-4 pass fixes" begin
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        OPUnified.enable_pipeline!()

        @testset "getfield canonicalization tolerates a dynamic boundscheck" begin
            src = _code_typed1(op_tupidx, (Int,))
            @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                  src.code[1].val == Core.Argument(2)
        end

        @testset "pinned-sparam constructor inlining" begin
            for f in (op_refchain, op_refwrap)
                src = _code_typed1(f, (Any,))
                @test count(_isnew, src.code) == 0
                @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                      src.code[1].val == Core.Argument(2)
            end
        end

        @testset "uninitialized-field mutable SROA: safe cases promote" begin
            src = _code_typed1(op_uninit_line, ())
            @test count(_isnew, src.code) == 0
            src = _code_typed1(op_uninit_arms, (Bool,))
            @test count(_isnew, src.code) == 0
            src = _code_typed1(op_uninit_nested, (Bool, Bool, Any, Any, Any))
            @test count(_isnew, src.code) == 0
        end

        @testset "uninitialized-field mutable SROA: unsafe case keeps memory form" begin
            src = _code_typed1(op_uninit_unsafe, (Bool,))
            @test count(_isnew, src.code) == 1
            @test count(x -> _iscall(src, getfield, x), src.code) == 1
            # behavior: the UndefRefError path survives
            @test op_uninit_unsafe(true) === 42
            @test_throws UndefRefError op_uninit_unsafe(false)
        end

        @testset "mutable SROA with store stays scalar-replaced" begin
            src = _code_typed1(op_mut_sroa, (Any, Any, Any))
            @test count(_isnew, src.code) == 0
            @test !any(x -> _iscall(src, getfield, x), src.code)
            @test !any(x -> _iscall(src, setfield!, x), src.code)
        end

        @testset "large const-foldable chain folds completely" begin
            # the round budget must not strand folded-but-unswept const calls
            src = _code_typed1(op_sin_chain, ())
            @test length(src.code) == 1 && _isreturn(src.code[1])
        end

        @testset "splatnew folds to new; Core.invoke calls inline" begin
            for tt in Any[(Int, Int), (Any, Any)]
                src = _code_typed1((a, b) -> op_construct_splatnew(
                    NamedTuple{(:a, :b), typeof((a, b))}, (a, b)), tt)
                @test count(x -> Meta.isexpr(x, :splatnew), src.code) == 0
                @test count(_isnew, src.code) == 1
            end
            @test op_construct_splatnew(NamedTuple{(:a, :b), Tuple{Int, Int}}, (1, 2)) == (a = 1, b = 2)
            src = _code_typed1(op_invoke34900, (Int, Int))
            @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                  src.code[1].val == Core.Argument(2)
            @test op_invoke34900(3, 4) === 3
        end

        @testset "_apply_iterate flattening and tuple identity" begin
            src = _code_typed1(op_identity_splat, (Tuple{Int, Int},))
            @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                  src.code[1].val == Core.Argument(2)
            @test op_identity_splat((1, 2)) === (1, 2)
            src = _code_typed1(op_apply_type_svec, ())
            @test length(src.code) == 1 && _isreturn(src.code[1])
            @test op_apply_type_svec() === NTuple{3, Float32}
        end

        @testset "finalizer resolution" begin
            # non-escaping allocation with an inlineable finalizer: both go
            src = _code_typed1(() -> (for i = 1:100; OPAllocNoEscape(); end), ())
            @test count(_isnew, src.code) == 0
            # a finalizer that can do no observable work is erased even
            # though the object escapes (returned)
            src = _code_typed1(op_useless_finalizer, ())
            @test !any(x -> _iscall(src, Core.finalizer, x), src.code)
            @test length(src.code) == 2
        end

        @testset "comparison lifting and pure-query folds" begin
            src = _code_typed1((c, x) -> (y = c ? x : nothing; y === nothing), (Bool, Int))
            @test !any(x -> _iscall(src, ===, x), src.code)
            src = _code_typed1((c, x) -> (y = c ? x : nothing; isa(y, Int)), (Bool, Int))
            @test !any(x -> _iscall(src, isa, x), src.code)
            src = _code_typed1((c, x) -> (y = c ? x : nothing; isdefined(y, 1)), (Bool, Some{Int}))
            @test !any(x -> _iscall(src, isdefined, x), src.code)
            # behavior of the lifted forms
            @test ((c, x) -> (y = c ? x : nothing; y === nothing))(true, 1) === false
            @test ((c, x) -> (y = c ? x : nothing; y === nothing))(false, 1) === true
            # ifelse const/equal-arm forwarding
            src = _code_typed1((a, b) -> Core.ifelse(true, a, b), (Any, Any))
            @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                  src.code[1].val == Core.Argument(2)
            src = _code_typed1((c, x) -> Core.ifelse(c, x, x), (Bool, Float64))
            @test length(src.code) == 1 && _isreturn(src.code[1]) &&
                  src.code[1].val == Core.Argument(3)
            # typeassert elimination when the subject's type proves it
            src = _code_typed1(op_typeassert_elim, (Int,))
            @test !any(x -> _iscall(src, typeassert, x), src.code)
            # ...and preservation when it does not
            @test_throws TypeError op_typeassert_keep("nope")
        end

        @testset "load forwarding through if-arm constructions" begin
            src = _code_typed1(op_immut_arms, (Bool, Int, Float64))
            @test count(_isnew, src.code) == 0
            @test !any(x -> _iscall(src, typeassert, x), src.code)
            @test op_immut_arms(true, 1, 2.0) === 1
            @test op_immut_arms(false, 1, 2.0) === 2.0
            src = _code_typed1(op_union_tuple_arms, (Bool, Int, Float64))
            @test !any(x -> _iscall(src, getfield, x), src.code)
            @test op_union_tuple_arms(false, 1, 2.0) === 2.0
        end

        @testset "isdefined folding over local allocations" begin
            src = _code_typed1(op_isassigned_sroa, (Any,))
            @test count(_isnew, src.code) == 0
            @test !any(x -> _iscall(src, isdefined, x), src.code)
            @test op_isassigned_sroa(7) === 7
            # dominating setfield! decides the query even though the object
            # escapes (definedness is monotone)
            src = _code_typed1(op_isdefined_dominated, ())
            @test !any(x -> _iscall(src, isdefined, x), src.code)
            @test op_isdefined_dominated() === 1.0
        end

        @testset "match-based union split: covered pair" begin
            src = _code_typed1(op_call_cov, (Any,))
            @test count(x -> _isinvoke(:op_split_cov, x), src.code) == 2
            @test !any(x -> _iscall(src, op_split_cov, x), src.code)
            @test op_call_cov(1) === :Number
            @test op_call_cov("s") === :Any
        end

        @testset "match-based union split: method-error fallback pair" begin
            src = _code_typed1(op_call_fb, (Any,))
            @test count(x -> _isinvoke(:op_split_fb, x), src.code) == 2
            @test count(x -> _iscall(src, Core.throw_methoderror, x), src.code) == 1
            @test op_call_fb(1) === :Number
            @test op_call_fb(Int) === :Type
            @test_throws MethodError op_call_fb("s")
        end

        @testset "match-based union split: no applicable method" begin
            src = _code_typed1(op_call_none, ())
            @test count(x -> _iscall(src, Core.throw_methoderror, x), src.code) == 1
            @test !any(x -> _iscall(src, op_split_one, x), src.code)
            @test_throws MethodError op_call_none()
        end

        @testset "ambiguous single-match devirtualization refused" begin
            # `ml_matches` reports ONE fully-covering match for this
            # signature while dispatch is ambiguous on the
            # (OPAmbigSR, OPAmbigSR) argument intersection — inlining or
            # devirtualizing the site would drop the runtime MethodError
            src = _code_typed1(op_call_ambig, (OPAmbigSR{String}, Any))
            @test any(x -> _iscall(src, op_ambf, x), src.code)
            @test !any(x -> Meta.isexpr(x, :invoke), src.code)
            @test op_call_ambig(OPAmbigSR("x"), 1) === 1
            @test_throws MethodError op_call_ambig(OPAmbigSR("x"), OPAmbigSR(1))
        end

        @testset "match-based union split: single non-covering match" begin
            src = _code_typed1(op_call_one, (Any,))
            @test count(x -> _iscall(src, Core.throw_methoderror, x), src.code) == 1
            @test !any(x -> _iscall(src, op_split_one, x), src.code)
            @test op_call_one(2) === 2
            @test_throws MethodError op_call_one("s")
        end
    finally
        OPUnified.disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

# wave 9: static-parameter reconstruction family
op_cfun_hook(x) = nothing
function op_cfun_assoc(handle::Ptr{Cvoid}, jlobj::T) where T
    # the base/libuv.jl associate_julia_struct shape: the cfunction's Ref{T}
    # references the METHOD's static parameter
    _ = @cfunction(op_cfun_hook, Cvoid, (Ref{T},))
    handle == C_NULL && return nothing
    ccall(:jl_uv_associate_julia_struct, Cvoid, (Ptr{Cvoid}, Any), handle, jlobj)
end
mutable struct OPCFunObj; x::Int; end
op_cfun_caller(w::OPCFunObj) = op_cfun_assoc(C_NULL, w)

# stock inline.jl "inlining with unmatched type parameters" (issue class of
# inline:1747): the callee reads a sparam its specialization cannot bake —
# the inliner must materialize `_compute_sparams`/`_svec_ref`; the
# constructor callee resolves through a computed Type{...} lattice element
@eval struct OPOldVal{T}
    (OV::Type{OPOldVal{T}})() where T = $(Expr(:new, :OV))
end
op_f_oldval(x::OPOldVal{i}) where {i} = i
function op_unmatched_typeparam()
    r = 0
    for i = 1:100
        r += op_f_oldval(OPOldVal{i}())
    end
    return r
end

# stock irpasses.jl named_tuple_elim: the materialized reconstruction must
# lift away entirely (lift_svec_refs! + the reconstruction transfer facts)
op_named_tuple_elim(name::Symbol, result) = NamedTuple{(name,)}(result)

# stock inline.jl issue #58915: the staged `merge` callee inlines via its
# generator EXPANSION and the whole setindex chain folds
op_f58915(nt) = @inline Base.setindex(nt, 2, :next)

_op_nonbuiltin_call(src) = (@nospecialize(x),) -> Meta.isexpr(x, :call) &&
    !(OPCC.singleton_type(OPCC.argextype(x.args[1], src, OPCC.VarState[])) isa Core.Builtin)

@testset "optimizer parity: wave-9 sparam reconstruction" begin
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        OPUnified.enable_pipeline!()

        @testset "inlined cfunction type slots carry no method typevars" begin
            src = _code_typed1(op_cfun_caller, (OPCFunObj,))
            # the callee must actually inline (guards vacuity below): no
            # residual call/invoke of op_cfun_assoc
            @test !any(x -> _isinvoke(:op_cfun_assoc, x) ||
                            _iscall(src, op_cfun_assoc, x), src.code)
            ncfun = 0
            for x in src.code
                if Meta.isexpr(x, :cfunction)
                    ncfun += 1
                    # rt + argt slots must be fully instantiated: a free
                    # TypeVar here is codegen-fatal in the spliced-into
                    # method ("type Ref should have an element type")
                    @test !Compiler.has_free_typevars(x.args[3])
                    for t in x.args[4]::Core.SimpleVector
                        @test ccall(:jl_has_free_typevars, Cint, (Any,), t) == 0
                    end
                elseif Meta.isexpr(x, :foreigncall)
                    @test ccall(:jl_has_free_typevars, Cint, (Any,), x.args[2]) == 0
                    for t in x.args[3]::Core.SimpleVector
                        @test ccall(:jl_has_free_typevars, Cint, (Any,), t) == 0
                    end
                end
            end
            @test ncfun == 1
            # behavior: codegen accepts the emitted form
            @test op_cfun_caller(OPCFunObj(1)) === nothing
        end

        @testset "unmatched-typeparam callee inlines via _compute_sparams" begin
            src = _code_typed1(op_unmatched_typeparam, ())
            # stock inline.jl's predicate: no residual zero-arg dynamic call
            # (the OldVal{i}() constructor through the computed type)
            @test !any(x -> Meta.isexpr(x, :call) && length(x.args) == 1, src.code)
            @test op_unmatched_typeparam() == sum(1:100)
        end

        @testset "reconstruction lift: named-tuple ctor fully eliminates" begin
            src = _code_typed1(op_named_tuple_elim, (Symbol, Tuple))
            @test count(x -> _iscall(src, Core._compute_sparams, x), src.code) == 0
            @test count(x -> _iscall(src, Core._svec_ref, x), src.code) == 0
            @test count(_op_nonbuiltin_call(src), src.code) == 0
            @test op_named_tuple_elim(:x, (1,)) === (x = 1,)
        end

        @testset "staged callee inlines its expansion (issue #58915)" begin
            src = _code_typed1(op_f58915, (@NamedTuple{next::UInt32, prev::UInt32},))
            @test count(x -> Meta.isexpr(x, :invoke), src.code) == 0
            @test count(_op_nonbuiltin_call(src), src.code) == 0
        end
    finally
        OPUnified.disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

# wave 10: comparison lifting must not confuse the two K"extract" flavors —
# `extract(if, i)` over an if whose arms produce ONE result operand projects
# INTO that runtime value (tuple element i), it does not select a result
# operand. Confusing them compared the destructured iterate-result TUPLE
# (instead of its first element) against `nothing`/the guard type, folding
# both guards to Const(false) and leaving the union-split residual
# `throw_methoderror` arm as the unconditionally-taken path (the
# StyledStrings `termcolor(::IOBuffer, ::SimpleColor, ::Char)` manual
# MethodError during incremental precompile).
struct OPSCol
    v::Symbol
end
struct OPFaceUL
    ul::Union{Nothing, Bool, OPSCol, Tuple{Union{Nothing, OPSCol}, Symbol}}
end
@noinline op_termc(io::IO, c::OPSCol, cat::Char) = (write(io, 'S'); nothing)
@noinline op_termc(io::IO, ::Nothing, cat::Char) = (write(io, 'N'); nothing)
function op_destructure_guard(io::IO, f::OPFaceUL)
    if f.ul isa Tuple
        c, s = f.ul     # re-read: the destructure iterates the wide union
        isnothing(c) || op_termc(io, c, '5')
    end
    nothing
end

@testset "optimizer parity: wave-10 extract-flavor comparison lifting" begin
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        OPUnified.enable_pipeline!()

        @testset "destructured union element keeps its guards" begin
            src = _code_typed1(op_destructure_guard, (IOBuffer, OPFaceUL))
            # the guarded op_termc dispatch must survive (inlined write or
            # call/invoke); pre-fix every path in the tuple branch collapsed
            # into the residual throw_methoderror arm
            @test any(src.code) do x
                _isinvoke(:op_termc, x) || _iscall(src, op_termc, x) ||
                    _isinvoke(:write, x)
            end
            # and the isa-OPSCol dispatch guard must not fold away (pre-fix
            # it folded to Const(false), erasing the guarded arm)
            @test any(src.code) do x
                Meta.isexpr(x, :call) && length(x.args) == 3 &&
                    OPCC.singleton_type(OPCC.argextype(x.args[1], src, OPCC.VarState[])) === isa &&
                    OPCC.singleton_type(OPCC.argextype(x.args[3], src, OPCC.VarState[])) === OPSCol
            end
        end
    finally
        OPUnified.disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

# A Const whose VALUE is a StmtId/Operand (self-hosting: the pipeline
# compiling UnifiedIR's own code) must materialize as a POOL CONSTANT —
# `vop`'s StmtId pass-through encodes it as a statement REFERENCE, which
# for id 0 crashed use_counts (the wrap_in_if! BoundsError under
# activate!) and for any other id silently aliases an arbitrary statement.
const OPUIR = OPUnified.UnifiedIR
op_stmtid_null() = OPUIR.NULL_STMT
op_stmtid_pair() = (op_stmtid_null(), Int32(7))
op_stmtid_three() = OPUIR.StmtId(Int32(3))
op_stmtid_use3() = (op_stmtid_three(), op_stmtid_three())

@testset "optimizer parity: wave-11 Const-StmtId materialization" begin
    for (f, expect) in ((op_stmtid_pair, (OPUIR.NULL_STMT, Int32(7))),
                        (op_stmtid_use3, (OPUIR.StmtId(Int32(3)), OPUIR.StmtId(Int32(3)))))
        ir = OPUnified.typed_ir(f, Any[])
        # no statement operand may reference id 0, and the executed result
        # must be the VALUE tuple (pre-fix: BoundsError during optimize for
        # the null id; a stmt-graph alias for nonzero ids)
        for s in OPUIR.each_stmt(ir)
            OPUIR.is_tombstone(ir, s) && continue
            for i in 1:OPUIR.nops(ir, s)
                o = OPUIR.getop(ir, s, i)
                @test !(OPUIR.optag(o) == OPUIR.TAG_STMT && OPUIR.payload(o) == 0)
            end
        end
        @test f() === expect
    end
end

# A GlobalRef VALUE (QuoteNode(GlobalRef) literal at entry, or a lattice
# Const(GlobalRef) materialized by the optimizer) is DATA — e.g.
# `invokelatest_gr`'s world-latest call target, the exact TOML Printer
# precompile shape — and must never be conflated with a semantic binding
# READ (TAG_GLOBAL). Pre-fix the entry's QuoteNode unwrap and const_vop
# both routed it through `vop`'s read form, so the pipeline resolved the
# binding and passed the bound VALUE:
#   MethodError: no method matching invokelatest_gr(::typeof(f), ...)
module OPGRData
    is_even(x::Int) = x % 2 == 0
    # TOML Printer shape: bare-symbol callee -> QuoteNode(GlobalRef) literal
    call_latest(x) = Base.@invokelatest is_even(x)
end
@eval op_gr_lit(x) = Base.invokelatest_gr($(QuoteNode(GlobalRef(OPGRData, :is_even))), x)
const OP_GRC = GlobalRef(OPGRData, :is_even)
op_gr_const(x) = Base.invokelatest_gr(OP_GRC, x)          # Const-binding fold leg
struct OPHoldGR; g::GlobalRef; end
const OP_HGR = OPHoldGR(GlobalRef(OPGRData, :is_even))
op_gr_field(x) = Base.invokelatest_gr(OP_HGR.g, x)        # getfield-fold leg
op_gr_tuple() = (GlobalRef(OPGRData, :is_even), 1)        # plain data position

@testset "optimizer parity: wave-12 GlobalRef-as-data vs binding read" begin
    for (f, args, want) in ((op_gr_lit, (4,), true),
                            (OPGRData.call_latest, (3,), false),
                            (op_gr_const, (4,), true),
                            (op_gr_field, (3,), false),
                            (op_gr_tuple, (), (GlobalRef(OPGRData, :is_even), 1)))
        ir = OPUnified.typed_ir(f, Any[map(typeof, args)...])
        # no read-form (globals-table) operand may name the data target
        for s in OPUIR.each_stmt(ir)
            OPUIR.is_tombstone(ir, s) && continue
            for i in 1:OPUIR.nops(ir, s)
                o = OPUIR.getop(ir, s, i)
                if OPUIR.optag(o) == OPUIR.TAG_GLOBAL
                    g = ir.body.globals[OPUIR.payload(o)]
                    @test !(g.mod === OPGRData && g.name === :is_even)
                end
            end
        end
        # and the exited body must execute with the GlobalRef ARGUMENT
        # intact (pre-fix: the resolved function object -> MethodError)
        irc = OPUnified.ir_to_ircode(ir)
        irc.argtypes[1] = Tuple{}
        oc = Core.OpaqueClosure(irc)
        @test isequal(oc(args...), want)
    end
end

# wave 13: a cell promoted inside a cfg ISLAND that a loop carries across
# iterations threads its value through every `continue`. Block liveness was
# seeded only from island READS, so a latch block that merely falls through
# to the backedge counted as live-out dead: the store-join phi was pruned
# and the `continue`'s carried value resolved through the idom chain to the
# island ENTRY value. The carried cell then circulated its initial value
# forever — `joinpath("/a", "b", "c")` returned `"/a"`, and
# `test_path("Compiler")` the bare `@__DIR__`. The short-circuit `||` is
# what keeps the loop body an unstructured island (a plain if/else in a
# `for` structurizes and never reaches this pass).
function op_accum_join(xs)
    s = xs[1]
    for i in 2:length(xs)
        if isempty(s) || s[end] == '/'    # read-free latch after both arms
            s *= xs[i]
        else
            s *= "/" * xs[i]
        end
    end
    return s
end
# Base.joinpath's body verbatim (the original miscompile)
function op_joinpath_body(paths::Union{Tuple,AbstractVector})
    path = paths[1]
    for i in firstindex(paths)+1:lastindex(paths)
        p = paths[i]
        if isabspath(p)
            path = p
        elseif isempty(path) || path[end] == '/'
            path *= p
        else
            path *= "/" * p
        end
    end
    return path
end
# a union-typed carried cell: a stale entry value here would also reach
# downstream union splits, not just produce a wrong result
function op_union_carry(n)
    y = nothing
    for i in 1:n
        if isodd(i) || i > 100
            y = i
        else
            y = Float64(i)
        end
    end
    return y
end

@testset "optimizer parity: wave-13 island backedge liveness" begin
    for (f, args, want) in ((op_accum_join, (("/a", "b", "c"),), "/a/b/c"),
                            (op_joinpath_body, (("/x", "y", "z"),), "/x/y/z"),
                            (op_union_carry, (4,), 4.0))
        # the executed body must advance the carried cell (pre-fix the
        # joinpath-shaped cases returned their INITIAL value)
        ir = OPUnified.typed_ir(f, Any[map(typeof, args)...])
        irc = OPUnified.ir_to_ircode(ir)
        irc.argtypes[1] = Tuple{}
        oc = Core.OpaqueClosure(irc)
        @test isequal(oc(args...), want)
    end
end

# ---------------------------------------------------------------------------
# Finalizer elision: EXECUTION semantics (wave 14)
#
# The structural predicates (`Core.finalizer` gone, one inlined finalizer
# call) hold whether the placed call runs at the END of the object's
# lifetime or right after its registration — only counting the finalizer's
# work tells the two apart, and the whole point of the transform is WHEN it
# runs. These mirror `Compiler/test/inline.jl`'s cfg_finalization corpus with
# the counter it uses (the finalizer adds the object's field, so the sum
# pins down the field value observed at death).
# ---------------------------------------------------------------------------

const OP_FIN_COUNT = Ref(0)
const OP_FIN_SINK = Ref(0)
@noinline op_add_fin_count!(x) = OP_FIN_COUNT[] += x
@noinline op_fin_sink!(x::Int) = (OP_FIN_SINK[] = x; nothing)

mutable struct OPAllocField
    x::Int
    function OPAllocField(x::Int)
        finalizer(new(x)) do this
            op_add_fin_count!(this.x)
        end
    end
end
mutable struct OPAllocFieldInter
    x::Int
end
function op_register_finalizer!(o::OPAllocFieldInter)
    finalizer(o) do this
        op_add_fin_count!(this.x)
    end
end

op_fin_const(n) = (for i = 1:n; o = OPAllocField(1); op_fin_sink!(o.x); end)
function op_fin_ctor_arg(n)                      # cfg_finalization1
    for i = (1 - n):n
        o = OPAllocField(i)
        if i == n
            op_fin_sink!(o.x)
        elseif i > 0
            op_fin_sink!(o.x)
        end
    end
end
function op_fin_store(n)                         # cfg_finalization2
    for i = (1 - n):n
        o = OPAllocField(1)
        o.x = i
        if i == n
            op_fin_sink!(o.x)
        elseif i > 0
            op_fin_sink!(o.x)
        end
    end
end
function op_fin_interproc(n)                     # cfg_finalization3
    for i = (1 - n):n
        o = OPAllocFieldInter(i)
        op_register_finalizer!(o)
        if i == n
            op_fin_sink!(o.x)
        elseif i > 0
            op_fin_sink!(o.x)
        end
    end
end
function op_fin_store_in_arm(n)                  # cfg_finalization6
    for i = (1 - n):n
        o = OPAllocField(0)
        if i == n
            o.x = i
        elseif i > 0
            op_fin_sink!(o.x)
        end
    end
end
function op_fin_store_chain(n)                   # cfg_finalization7
    for i = (1 - n):n
        o = OPAllocField(0)
        o.x = 0
        if i == n
            o.x = i
        end
        o.x = i
        if i == n - 1
            o.x = i
        end
        o.x = 0
        if i == n
            o.x = i
        end
    end
end
function op_fin_store_use(n)                     # straight line: store, load
    for i = 1:n
        o = OPAllocField(1)
        o.x = i
        op_fin_sink!(o.x)
    end
end

@testset "optimizer parity: finalizer elision runs at the lifetime END" begin
    n = 20
    # the effects gate the whole transform gates on: bumping a `Ref` through
    # a const global must stay `:nothrow`. Stock never propagates a mutable
    # constant into a callee, so its `getfield` is always inferred against
    # the widened argument; this pipeline inlines the body and re-infers it
    # with the constant SUBJECT, where `isdefined_tfunc` gives up.
    let fx = OPUnified.frame_effects_meta(OPUnified.typed_ir(op_add_fin_count!, Any[Int]))
        @test OPCC.is_nothrow(fx)
        @test OPCC.is_finalizer_inlineable(fx)
    end
    # The shapes that STORE the field after the constructor registered the
    # finalizer (op_fin_store, the store-in-arm pair, op_fin_store_use) are
    # the ones that pin down WHERE the placed call landed: they used to
    # observe the `new`'s initial value because `resolve_finalizers!`
    # discharged the registration inside the CONSTRUCTOR, whose object
    # reaches the rest of the program only through `Base.finalizer`'s own
    # result — the use scan skipped it and the object looked dead at the
    # registration.
    for (f, want) in ((op_fin_const, n),
                      (op_fin_ctor_arg, n),
                      (op_fin_store, n),
                      (op_fin_interproc, n),
                      (op_fin_store_in_arm, n),
                      (op_fin_store_chain, n),
                      (op_fin_store_use, div(n * (n + 1), 2)))
        ir = OPUnified.typed_ir(f, Any[Int])
        irc = OPUnified.ir_to_ircode(ir)
        elided = !any(irc.stmts.stmt) do @nospecialize(x)
            Meta.isexpr(x, :call) && !isempty(x.args) &&
                (x.args[1] === Core.finalizer ||
                 x.args[1] === GlobalRef(Core, :finalizer))
        end
        irc.argtypes[1] = Tuple{}
        oc = Core.OpaqueClosure(irc)
        OP_FIN_COUNT[] = 0
        oc(n)
        if !elided
            # declining is a missed optimization, not a bug: the registration
            # survives and the finalizers run at GC time instead
            @test_skip OP_FIN_COUNT[] == want
        else
            @test OP_FIN_COUNT[] == want
        end
    end
end

end # module UnifiedOptimizerParityTests
