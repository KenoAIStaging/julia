# Exit-shape parity (wave 6): stock emitted-form conventions at the typed
# exit boundary —
#   * `Expr(:invoke_modify)` emission for statically-resolved atomic modify
#     builtins (stock's handle_modifyop!_call! relocated to the boundary:
#     `devirtualize_modifyops!` + `ir_to_ircode` consumption),
#   * IRCode argtypes/slottypes convention (inferred lattice per root arg,
#     singletons refined to `Const` — stock's matching_cache_argtypes shape,
#     including the isva packed-tuple form),
#   * value-position hoist of unbound/partitioned globals from IR-native
#     construction (B3a Findings F3 residual).
# Included from runtests.jl.

module UnifiedExitShapeTests

using Test
using UnifiedIR
using UnifiedIR: Builder, append_stmt!, finish!, @K_str, op_stmt, op_inline, op_region
import Compiler
const ES_CC = Compiler
const ES_U = Compiler.load_unified!()

es_state() = ES_U.UInferState(ES_U.UInferConfig(world = Base.get_world_counter()))

"emit `f(argtypes...)` through typed_ir + devirtualize_modifyops! +
ir_to_ircode; returns (rewrite count, stmt vector, ircode)"
function es_emit(f, ats)
    uir = ES_U.typed_ir(f, ats)
    n = ES_U.devirtualize_modifyops!(uir, es_state(), ES_CC.NativeInterpreter())
    irc = ES_U.ir_to_ircode(uir)
    stmts = Any[irc[Core.SSAValue(i)][:stmt] for i in 1:length(irc.stmts)]
    return n, stmts, irc
end

es_isinvokemodify(sym::Symbol) = (@nospecialize(x),) -> Meta.isexpr(x, :invoke_modify) &&
    (x.args[1] isa Core.CodeInstance ? (x.args[1]::Core.CodeInstance).def :
                                       x.args[1]::Core.MethodInstance).def.name === sym

mutable struct ESAtomic{T}
    @atomic x::T
end

es_plus(a)   = @atomic a.x + 1
es_pluseq(a) = @atomic a.x += 1
es_max(a)    = @atomic a.x max 10
es_ptr(a)    = unsafe_modify!(a, +, 1)
es_mem(a)    = Core.memoryrefmodify!(a, +, 1, :sequentially_consistent, true)
global es_glob::Int = 1
es_globinc() = @atomic (@__MODULE__).es_glob += 1
const es_const_glob = 1
es_constglobinc() = Core.modifyglobal!(@__MODULE__, :es_const_glob, +, 1, :sequentially_consistent)
es_mymax(x::T, y::T) where T<:Real = max(x, y)
es_mymax(x::T, y::Real) where T<:Real = convert(T, max(x, y))::T
es_union(a, b) = @atomic a.x es_mymax b

@testset "invoke_modify emission: single-match modify builtins" begin
    for (f, ats, op, builtin) in Any[
            (es_plus,   Any[ESAtomic{Int}], :+,   Core.modifyfield!),
            (es_pluseq, Any[ESAtomic{Int}], :+,   Core.modifyfield!),
            (es_max,    Any[ESAtomic{Int}], :max, Core.modifyfield!),
            (es_ptr,    Any[Ptr{Int}],      :+,   Core.Intrinsics.atomic_pointermodify),
            (es_mem,    Any[AtomicMemoryRef{Int}], :+, Core.memoryrefmodify!),
            (es_globinc, Any[],             :+,   Core.modifyglobal!)]
        n, stmts, _ = es_emit(f, ats)
        @test n == 1
        @test count(es_isinvokemodify(op), stmts) == 1
        im = stmts[findfirst(es_isinvokemodify(op), stmts)]
        # stock's handle_modifyop!_call! shape: resolved target prepended,
        # the original builtin call kept verbatim behind it
        @test im.args[1] isa Core.CodeInstance
        f2 = im.args[2]
        f2 isa GlobalRef && (f2 = getglobal(f2.mod, f2.name))
        @test f2 === builtin
    end
end

@testset "invoke_modify union split and non-modifiable declines" begin
    # op over a Union v: union_split_calls! (wave 8) splits the modifyfield!
    # on its value argument, so BOTH narrowed arms devirtualize — stock's
    # two-Expr(:invoke_modify) shape (the modifyproperty! wrapper split)
    n, stmts, _ = es_emit(es_union, Any[ESAtomic{Int}, Union{Int,Float64}])
    @test n == 2
    @test count(es_isinvokemodify(:es_mymax), stmts) == 2
    # defined-const binding cannot be modified: stays dynamic
    n, stmts, _ = es_emit(es_constglobinc, Any[])
    @test n == 0
end

@testset "invoke_modify executes through codegen" begin
    n, stmts, irc = es_emit(es_pluseq, Any[ESAtomic{Int}])
    @test n == 1
    irc.argtypes[1] = Tuple{}          # OC env convention
    oc = Core.OpaqueClosure(irc)
    a = ESAtomic{Int}(41)
    @test oc(a) == 42
    @test (@atomic a.x) == 42
end

es_va(x::Vararg{T}) where {T <: Number} = x[1]
es_nothing_arg(x::Nothing, y::Int) = y

@testset "exit argtypes: stock slottypes convention" begin
    uir = ES_U.typed_ir(es_va, Any[Rational])
    irc = ES_U.ir_to_ircode(uir)
    @test irc.argtypes[1] == ES_CC.Const(es_va)
    # the isva packed-tuple slot form comes through the driver's arglattice;
    # here the query seeds per-position types — the singleton function slot
    # is the convention under test
    uir = ES_U.typed_ir(es_nothing_arg, Any[Nothing, Int])
    irc = ES_U.ir_to_ircode(uir)
    @test irc.argtypes == Any[ES_CC.Const(es_nothing_arg), ES_CC.Const(nothing), Int]
end

module ESHoistM
global q                     # declared, never defined: unbound
global r::Int = 3            # typed non-const global
const c = 42                 # defined-const: legal in value position
end

@testset "value-position global hoist (F3 residual)" begin
    # call operand
    b = Builder(name = :es_hoist_call)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"call", GlobalRef(Base, :identity), GlobalRef(ESHoistM, :q); type = Any)
    append_stmt!(b, K"return", op_stmt(x))
    irc = ES_U.ir_to_ircode(finish!(b))
    ES_CC.verify_ir(irc)
    @test irc[Core.SSAValue(1)][:stmt] == GlobalRef(ESHoistM, :q)
    @test Meta.isexpr(irc[Core.SSAValue(2)][:stmt], :call)
    @test irc[Core.SSAValue(2)][:stmt].args[2] == Core.SSAValue(1)

    # return position
    b = Builder(name = :es_hoist_ret)
    append_stmt!(b, K"region_arg"; type = Any)
    append_stmt!(b, K"return", GlobalRef(ESHoistM, :r))
    irc = ES_U.ir_to_ircode(finish!(b))
    ES_CC.verify_ir(irc)
    @test irc[Core.SSAValue(1)][:stmt] == GlobalRef(ESHoistM, :r)
    @test irc[Core.SSAValue(1)][:type] === Int
    @test irc[Core.SSAValue(2)][:stmt] == Core.ReturnNode(Core.SSAValue(1))

    # φ-edge value (if-region result fed by a global)
    b = Builder(name = :es_hoist_phi)
    append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"region_arg"; type = Bool)
    fi = append_stmt!(b, K"if", op_stmt(c); type = Any)
    UnifiedIR.open_region!(b, fi)
    append_stmt!(b, K"result", GlobalRef(ESHoistM, :q))
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi)
    append_stmt!(b, K"result", GlobalRef(ESHoistM, :r))
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", op_stmt(fi))
    irc = ES_U.ir_to_ircode(finish!(b))
    ES_CC.verify_ir(irc)
    grefs = [i for i in 1:length(irc.stmts) if irc[Core.SSAValue(i)][:stmt] isa GlobalRef &&
             irc[Core.SSAValue(i)][:stmt].mod === ESHoistM]
    @test length(grefs) == 2

    # defined-const and Core/Base globals stay in value position (stock's
    # canonical inlined form; hoisting them would change emitted shape)
    b = Builder(name = :es_nohoist)
    append_stmt!(b, K"region_arg"; type = Any)
    y = append_stmt!(b, K"call", GlobalRef(Base, :identity), GlobalRef(ESHoistM, :c); type = Any)
    append_stmt!(b, K"return", op_stmt(y))
    irc = ES_U.ir_to_ircode(finish!(b))
    ES_CC.verify_ir(irc)
    @test irc[Core.SSAValue(1)][:stmt].args[2] == GlobalRef(ESHoistM, :c)
end

end # module UnifiedExitShapeTests
