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

# Quoted-AST constants at the exit boundary (C1 sys-unified regression): the
# entry converter unwraps EVERY QuoteNode into the constants pool, so both
# exits must re-quote with stock breadth (`Base.is_self_quoting`) — not just
# Symbol/Expr. A bare LineNumberNode in value position is a codegen error
# (the class that broke Test.parse_testset_args under the sys-unified image),
# a bare SSAValue literal a dangling-reference miscompile.
@eval es_lnn_lit(x) = ($(QuoteNode(LineNumberNode(7, :es_lit))), x)
@eval es_ssa_lit(x) = (x, $(QuoteNode(Core.SSAValue(7))))
@testset "quoted AST-node constants through the exits" begin
    # exit_lowered: redefine through IR and execute both exits' emissions
    let g = ES_U.redefine_through_ir(es_lnn_lit, Tuple{Int}; mod = @__MODULE__)
        @test Base.invokelatest(g, 3) == es_lnn_lit(3) == (LineNumberNode(7, :es_lit), 3)
    end
    let g = ES_U.redefine_through_ir(es_ssa_lit, Tuple{Int}; mod = @__MODULE__)
        @test Base.invokelatest(g, 2) == es_ssa_lit(2) == (2, Core.SSAValue(7))
    end
    # typed exit: no bare AST-node constant in any emitted value position
    let uir = ES_U.typed_ir(es_lnn_lit, Any[Int])
        irc = ES_U.ir_to_ircode(uir)
        bare = 0
        for i in 1:length(irc.stmts)
            stmt = irc[Core.SSAValue(i)][:stmt]
            stmt isa Expr || continue
            for a in stmt.args
                a isa LineNumberNode && (bare += 1)
            end
        end
        @test bare == 0
    end
    # end-to-end through the driver + codegen
    @test ES_U.with_unified_compiler(es_lnn_lit, 5) == (LineNumberNode(7, :es_lit), 5)
    @test ES_U.with_unified_compiler(es_ssa_lit, 5) == (5, Core.SSAValue(7))
end

# The flip side of the same boundary: foreigncall/cfunction STRUCTURAL slots
# are stored VERBATIM at entry (no QuoteNode unwrap), so the exits must strip
# exactly the one QuoteNode layer the value accessors add — over-quoting the
# cconv slot's own QuoteNode makes emit_ccall's convert_cconv segfault (the
# C1 build-6 crash), under-quoting was never sound either.
es_pid() = ccall(:getpid, Cint, ())
es_pid_wrap(x) = (es_pid() > 0) ? x + 1 : x
@testset "foreigncall structural slots stay verbatim through the exits" begin
    let g = ES_U.redefine_through_ir(es_pid, Tuple{}; mod = @__MODULE__)
        @test Base.invokelatest(g) == es_pid()
    end
    # typed exit + codegen (the segfault path): the emitted foreigncall's
    # cconv slot must satisfy emit_ccall's QuoteNode contract
    @test ES_U.with_unified_compiler(es_pid) == es_pid()
    @test ES_U.with_unified_compiler(es_pid_wrap, 41) == 42
    let uir = ES_U.typed_ir(es_pid, Any[])
        irc = ES_U.ir_to_ircode(uir)
        fcs = [irc[Core.SSAValue(i)][:stmt] for i in 1:length(irc.stmts)
               if Meta.isexpr(irc[Core.SSAValue(i)][:stmt], :foreigncall)]
        @test !isempty(fcs)
        for fc in fcs
            @test fc.args[5] isa QuoteNode  # emit_ccall asserts this shape
        end
    end
end

# Inlining substitutes pool constants (stored BARE, general convention) into
# foreigncall VALUE slots: they must come back value-QUOTED while the
# verbatim-stored structural slots are stripped — the build-7 regression,
# where `Module()`'s inlined ccall got a bare `:anonymous` Symbol in value
# position (evaluated as a binding read: UndefVarError).
@inline es_newmod(name::Symbol) = ccall(:jl_f_new_module, Ref{Module}, (Any, Bool, Bool), name, false, false)
es_newmod0() = es_newmod(:es_anon_mod)
@testset "substituted constants in foreigncall value slots stay quoted" begin
    let uir = ES_U.typed_ir(es_newmod0, Any[])
        irc = ES_U.ir_to_ircode(uir)
        fcs = [irc[Core.SSAValue(i)][:stmt] for i in 1:length(irc.stmts)
               if Meta.isexpr(irc[Core.SSAValue(i)][:stmt], :foreigncall)]
        @test !isempty(fcs)  # the ccall inlined into es_newmod0
        for fc in fcs
            @test fc.args[5] isa QuoteNode                       # cconv verbatim
            @test all(a -> !(a isa Symbol), fc.args[6:end])      # no bare value Symbol
        end
    end
    @test nameof(ES_U.with_unified_compiler(es_newmod0)) === :es_anon_mod
end

# `@aliasscope` brackets are NOT metadata: codegen walks the emitted statement
# array linearly, pushing an alias scope on `Expr(:aliasscope)` and popping on
# `Expr(:popaliasscope)`, and annotates every load in between. The entry
# converters used to fold both heads away with `:meta`/`:inbounds`, so the
# emitted body carried no `!alias.scope`/`!noalias` names at all (codegen.jl's
# `occursin("aliasscope", str)` check on foo31018!).
function es_aliasscope!(a, b)
    @Base.Experimental.aliasscope for i in eachindex(a, b)
        a[i] = Base.Experimental.Const(b)[i]
    end
end
es_aliasscope_straight!(a, b) = @Base.Experimental.aliasscope (a[1] = Base.Experimental.Const(b)[1])

@testset "aliasscope brackets survive to the typed exit, in order" begin
    for (f, ats) in Any[(es_aliasscope!, Any[Vector{Int}, Vector{Int}]),
                        (es_aliasscope_straight!, Any[Vector{Int}, Vector{Int}])]
        uir = ES_U.typed_ir(f, ats)
        # present in the unified IR as first-class statements (not dropped on entry)
        kinds = [UnifiedIR.stmt_kind(uir, s) for s in UnifiedIR.each_stmt(uir)]
        @test count(==(K"aliasscope"), kinds) == 1
        @test count(==(K"popaliasscope"), kinds) == 1
        # ...and re-emitted at the boundary, `:aliasscope` strictly first
        irc = ES_U.ir_to_ircode(uir)
        heads = [irc[Core.SSAValue(i)][:stmt].head for i in 1:length(irc.stmts)
                 if Meta.isexpr(irc[Core.SSAValue(i)][:stmt], :aliasscope) ||
                    Meta.isexpr(irc[Core.SSAValue(i)][:stmt], :popaliasscope)]
        @test heads == [:aliasscope, :popaliasscope]
    end
end

@testset "unbalanced aliasscope emission declines to stock" begin
    # the balance guard the exit runs over the linearized statement array
    @test ES_U.check_aliasscope_balance(Any[Expr(:aliasscope), Expr(:popaliasscope)]) === nothing
    @test_throws ES_U.UnsupportedIR ES_U.check_aliasscope_balance(Any[Expr(:popaliasscope)])
    @test_throws ES_U.UnsupportedIR ES_U.check_aliasscope_balance(Any[Expr(:aliasscope)])
    @test_throws ES_U.UnsupportedIR ES_U.check_aliasscope_balance(
        Any[Expr(:popaliasscope), Expr(:aliasscope)])
end

end # module UnifiedExitShapeTests
