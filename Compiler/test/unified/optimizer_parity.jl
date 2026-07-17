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
_isinvoke(sym::Symbol, @nospecialize(x)) =
    Meta.isexpr(x, :invoke) && (x.args[1]::Core.CodeInstance).def.def.name === sym

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

end # module UnifiedOptimizerParityTests
