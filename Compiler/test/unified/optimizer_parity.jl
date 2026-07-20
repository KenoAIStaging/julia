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
    finally
        OPUnified.disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

end # module UnifiedOptimizerParityTests
