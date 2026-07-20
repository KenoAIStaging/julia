# Effects parity (COMPILER-PORT-PLAN A3 / EFFECTS-PARITY-PLAN E1+E2): the
# unified inference computes full-width `Compiler.Effects`. These are the
# regression anchors for the effects_scoreboard.jl corpus run — each testset
# pins a behavior class the scoreboard measured against stock.

const UP = UnifiedCompiler

"Frame-level Effects of `f(argtypes...)` through the unified driver bridge
(the `_infer_effects` hook implementation; no global hook state needed)."
function ueffects(@nospecialize(f), argtypes::Tuple = (); optimize::Bool = true)
    tt = Base.signature_type(f, argtypes)
    r = UP.unified_infer_effects(CC.NativeInterpreter(), tt, optimize)
    @assert r isa CC.Effects "unified_infer_effects declined for $tt"
    return r
end

uexct(@nospecialize(f), argtypes::Tuple = (); optimize::Bool = true) =
    UP.unified_infer_exception_type(CC.NativeInterpreter(),
                                    Base.signature_type(f, argtypes), optimize)

@testset "effects parity: full-width frame effects" begin
    @test CC.is_foldable_nothrow(ueffects(+, (Int, Int)))
    # always-throwing frames are consistent, not nothrow, exct-precise
    e = ueffects(() -> error("x"))
    @test CC.is_consistent(e) && !CC.is_nothrow(e)
    @test uexct(() -> error("x")) === ErrorException
    # loops drop terminates only
    e = ueffects(n -> (s = 0; for i in 1:n; s += i; end; s), (Int,))
    @test !CC.is_terminates(e) && CC.is_nothrow(e) && CC.is_consistent(e)
end

@testset "effects parity: CONSISTENT_IF_NOTRETURNED" begin
    # mutable allocation not escaping through the return: resolved at finish
    e = ueffects(x -> (r = Ref(x); r[] + 1), (Int,))
    @test CC.is_consistent(e)
    # returned allocation keeps the conditional bit observable
    e = ueffects(x -> Ref(x), (Int,))
    @test !CC.is_consistent(e)
    @test CC.is_consistent_if_notreturned(e)
end

@testset "effects parity: @assume_effects overrides + recursion" begin
    # method-level :terminates_globally beats the self-recursion taint
    @eval Base.@assume_effects :terminates_globally function par_recur1(x)
        x == 0 && return 1
        0 ≤ x < 20 || error("bad")
        return x * par_recur1(x - 1)
    end
    e = ueffects(par_recur1, (Int,))
    @test CC.is_terminates(e) && CC.is_foldable(e)
    # plain self-recursion taints terminates ONLY (the is_edge_recursed rule)
    @eval par_recur2(x) = x <= 0 ? 0 : par_recur2(x - 1)
    e = ueffects(par_recur2, (Int,))
    @test !CC.is_terminates(e)
    @test CC.is_consistent(e) && CC.is_effect_free(e)
    # abstract recursion over shrinking tuple signatures does not taint
    @eval function par_sumrecur(a, x)
        isempty(a) && return x
        return par_sumrecur(Base.tail(a), x + first(a))
    end
    @test CC.is_terminates(ueffects(par_sumrecur, (Tuple{Int,Int,Int}, Int)))
    @test !CC.is_terminates(ueffects(par_sumrecur, (Tuple{Int,Int,Int,Vararg{Int}}, Int)))
end

@testset "effects parity: exct-lite and caught exceptions" begin
    # swallowed exceptions upgrade frame nothrow at finish
    e = ueffects(x -> (try; sqrt(x); catch; 0.0; end), (Float64,))
    @test CC.is_nothrow(e)
    @test uexct(x -> (try; sqrt(x); catch; 0.0; end), (Float64,)) === Union{}
    # a handler reading the exception value taints consistency
    @eval par_catchread() = (try; error("x"); catch err; err; end)
    @test !CC.is_consistent(ueffects(par_catchread))
    @test CC.is_nothrow(ueffects(par_catchread))
end

@testset "effects parity: throw_methoderror exct arms" begin
    # stock abstract_throw_methoderror: zero call args ⇒ ArgumentError,
    # fixed nonzero arity ⇒ MethodError, imprecise (vararg) arity ⇒ Union
    @test uexct(() -> Core.throw_methoderror()) === ArgumentError
    @test uexct(x -> Core.throw_methoderror(x), (Int,)) === MethodError
    @test uexct(args -> Core.throw_methoderror(args...), (Vector{Any},)) ===
          Union{MethodError,ArgumentError}
end

@testset "effects parity: cell definedness witnesses" begin
    # a for-loop under a guard lowers to a cfg island whose iterate cell is
    # newed: the flow-sensitive store witness must keep the reads nothrow
    @eval par_guarded_loop(c) = (s = 0; if c; for i in 1:2; s += 1; end; end; s)
    e = ueffects(par_guarded_loop, (Bool,))
    @test CC.is_nothrow(e)
    @test uexct(par_guarded_loop, (Bool,)) === Union{}
    # @isdefined-guarded read of a conditionally-assigned local (stock's
    # VarState.undef refinement through the cell_isdefined conditional)
    @eval function par_isdef_guard(c, x)
        local val
        if c
            val = x
        end
        if @isdefined val
            return val
        end
        return zero(Int)
    end
    @test CC.is_nothrow(ueffects(par_isdef_guard, (Bool, Int)))
    # negation flips the witness to the else arm
    @eval function par_isdef_neg(c, x)
        local val
        if c
            val = x
        end
        if !(@isdefined val)
            return 0
        end
        return val
    end
    @test CC.is_nothrow(ueffects(par_isdef_neg, (Bool, Int)))
end

@testset "effects parity: isdefined field conditionals" begin
    # stock abstract_isdefined: the then arm refines the subject's field
    # definedness (PartialStruct undefs), making the guarded read nothrow
    e = ueffects((Base.RefValue{Any},)) do x
        if isdefined(x, :x)
            return getfield(x, :x)
        end
    end
    @test CC.is_nothrow(e)
    # setfield! back-propagates the definedness (stock's
    # form_partially_defined_struct site): the isdefined folds
    e = ueffects((Base.RefValue{String}, String)) do x, v
        setfield!(x, :x, v)
        getfield(x, :x)
    end
    @test CC.is_nothrow(e)
end

@testset "effects parity: noub matrix (array indexing)" begin
    # frame-own conditional (the method-level @_noub_if_noinbounds_meta)
    @test CC.is_noub_if_noinbounds(ueffects(getindex, (Vector{Int}, Int)))
    # callsite resolution in an inbounds-free, non-propagating frame
    @test CC.is_noub(ueffects((xs, i) -> xs[i], (Vector{Int}, Int)))
    # an @inbounds-marked body demotes the callee conditional
    @test !CC.is_noub(ueffects((xs, i) -> (@inbounds xs[i]), (Vector{Int}, Int)))
    # tuple getindex: guarded boundscheck does not taint frame consistency
    e = ueffects(getindex, (NTuple{6,Float64}, Int))
    @test CC.is_consistent(e)
end

@testset "effects parity: globals" begin
    @eval global par_typed_global::Int = 42
    @eval const par_const_global = 42
    e = ueffects(() -> Base.getglobal(@__MODULE__, :par_const_global))
    @test CC.is_foldable_nothrow(e) && CC.is_inaccessiblememonly(e)
    e = ueffects(() -> Base.getglobal(@__MODULE__, :par_typed_global))
    @test !CC.is_consistent(e) && !CC.is_inaccessiblememonly(e) && !CC.is_nothrow(e)
    # setglobal! of a fitting value into a typed global: consistent + nothrow,
    # never effect-free
    @eval par_setglobal!() = setglobal!(@__MODULE__, :par_typed_global, 1)
    e = ueffects(par_setglobal!)
    @test CC.is_consistent(e) && CC.is_nothrow(e) && !CC.is_effect_free(e)
    # assignment through the lowered convert-guard (get_binding_type fold)
    @eval par_assignglobal!() = global par_typed_global = 2
    e = ueffects(par_assignglobal!; optimize = false)
    @test CC.is_consistent(e) && CC.is_nothrow(e) && !CC.is_effect_free(e)
    # a mutable literal makes its memory reachable: not inaccessiblememonly
    ref = Ref(1)
    @eval par_qnref() = $(QuoteNode(ref)).x
    @test !CC.is_consistent(ueffects(par_qnref))
end

@testset "effects parity: concrete evaluation gating" begin
    # (definitions first: the inference state pins the world at construction)
    @eval Base.@assume_effects :foldable par_cefold(d, k) = d[k]
    @eval par_rtcall() = Core.Compiler.return_type(sin, Tuple{Float64})
    @eval par_spthrow(::Union{Nothing,Type{T}}) where {T} = (T; nothing)
    st = UP.UInferState()
    fr = UP.Frame(UnifiedIR.Builder().ir, st, Any[])
    # :foldable + all-const args (mutable Dict included) folds to Const
    dict = Dict{Any,Any}(:a => 1)
    r = UP.infer_call(fr, Any[CC.Const(par_cefold), CC.Const(dict), CC.Const(:a)])
    @test r.rt isa CC.Const && r.rt.val == 1
    # the return_type fold carries nortcall=false: callers never fold into
    # re-entering inference
    r = UP.infer_call(fr, Any[CC.Const(par_rtcall)])
    @test r.rt isa CC.Const && r.rt.val === Float64
    @test !CC.is_nortcall(r.effects)
    @test !CC.is_foldable(r.effects, #=check_rtcall=#true)
    # undefined static parameter reads taint nothrow
    @test CC.is_nothrow(ueffects(par_spthrow, (Type{Int},)))
    @test !CC.is_nothrow(ueffects(par_spthrow, (Nothing,)))
end

@testset "effects parity: ambiguity/MethodError accounting" begin
    @eval begin
        par_ambig(a::Int, b) = 1
        par_ambig(a, b::Int) = 1
        par_ambig(a, b) = 1
    end
    @test !CC.is_nothrow(ueffects(par_ambig, (Int, Any)))
    @test CC.is_nothrow(ueffects(par_ambig, (Int, Float64)))
end
