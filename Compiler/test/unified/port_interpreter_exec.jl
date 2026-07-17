# Re-expression of Compiler/test/interpreter_exec.jl over UnifiedIR
# (COMPILER-PORT-PLAN B3a): the runtime interpreter must implement the IR
# node semantics — φ parallel moves, PhiC/Upsilon exceptional stores,
# EnterNode scopes, undef edges — identically to codegen. The original
# hand-writes SSA thunks and evals them under the default and `compile=min`
# modules; here the same adversarial programs are built as region IR, pushed
# through the typed exit (which synthesizes the φ/PhiC/Upsilon forms), and
# executed on every engine: the UnifiedIR reference interpreter, native
# codegen (OpaqueClosure + methods), the runtime AST interpreter (methods in
# a compile=min module, the original's forcing), and toplevel thunks in both
# module modes (the original's eval form). Mapping: /workspace/B3A-PORT-MAP.md.

module B3APortInterpExec

using Test
using UnifiedIR
using UnifiedIR: op_stmt, op_region, op_inline, StmtId
import ..UnifiedCompiler
import ..CC as Compiler

module B3AIGlob
    global test29262u::Bool = true
end
b3a_setflag!(v::Bool) = Core.eval(B3AIGlob, :(test29262u = $v))

# the original's interpreter forcing: a module under
# `@compiler_options compile=min` (methods defined here run on
# jl_fptr_interpret_call; toplevel thunks are always interpreted)
module B3AIMin
    Base.Experimental.@compiler_options compile=min
end
module B3AIDef end

"Define the (φ-bearing) IRCode as the *source* of a niladic method in `mod`,
so the executing engine is decided by the module's compile mode — the
runtime AST interpreter for B3AIMin, native codegen for B3AIDef."
function b3a_ssa_method(mod::Module, irc0::Compiler.IRCode)
    ircx = Compiler.copy(irc0)
    ircx.debuginfo.def = :b3a_ssa_method
    src = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    src.slotnames = Symbol[Symbol("#self#")]
    src.slotflags = fill(0x00, 1)
    src.slottypes = Any[Any]
    src.isva = false
    src.nargs = UInt(1)
    Compiler.ir_to_codeinf!(src, ircx)
    src.ssavaluetypes = length(src.code)     # uninferred marker
    src.rettype = Any
    src.edges = nothing
    src.parent = nothing
    name = gensym(:b3a_ssaexec)
    f = Core.eval(mod, :(function $name end))
    argdata = Core.svec(Core.svec(typeof(f)), Core.svec(),
                        LineNumberNode(0, :b3a_ssa_method))
    ccall(:jl_method_def, Any, (Any, Ptr{Cvoid}, Any, Any),
          argdata, C_NULL, src, mod)
    return f
end

"The original's eval form: the IRCode as an argument-less toplevel thunk."
function b3a_thunk(irc0::Compiler.IRCode)
    ircx = Compiler.copy(irc0)
    ircx.debuginfo.def = :b3a_thunk
    src = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    src.slotnames = Symbol[Symbol("#self#")]
    src.slotflags = fill(0x00, 1)
    src.slottypes = Any[Any]
    src.isva = false
    src.nargs = UInt(1)
    Compiler.ir_to_codeinf!(src, ircx)
    src.nargs = UInt(0)
    resize!(src.slotnames, 0)
    resize!(src.slotflags, 0)
    src.slottypes = Any[]
    src.ssavaluetypes = length(src.code)
    src.rettype = Any
    src.edges = nothing
    src.parent = nothing
    return Expr(:thunk, src)
end

# the module flag really forces the interpreter (JL_OPTIONS_COMPILE_MIN == 3)
@testset "compile=min forcing is in effect" begin
    @test ccall(:jl_get_module_compile, Cint, (Any,), B3AIMin) == 3
    @test ccall(:jl_get_module_compile, Cint, (Any,), B3AIDef) != 3
end

# ---------------------------------------------------------------------------
# (#29262 program 1) φ selection by executed edge: t ? :a : :b through a join
# ---------------------------------------------------------------------------

@testset "interpreter matches codegen: φ edge selection (#29262)" begin
    b = Builder(name = :p1)
    append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"globalref", GlobalRef(B3AIGlob, :test29262u); type = Bool)
    fi = append_stmt!(b, K"if", op_stmt(t); type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", :a)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", :b)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", fi)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)      # fixture legality
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing     # stock legality of the exit
    # the join reached codegen/interpreter as a real φ
    @test count(s -> s isa Core.PhiNode, irc.stmts.stmt) >= 1
    oc = Core.OpaqueClosure(irc)
    gmin = b3a_ssa_method(B3AIMin, irc)
    gdef = b3a_ssa_method(B3AIDef, irc)
    for (flag, want) in ((true, :a), (false, :b))
        b3a_setflag!(flag)
        @test UnifiedIR.interpret(ir, nothing) === want
        @test oc() === want
        @test Base.invokelatest(gmin) === want    # runtime AST interpreter
        @test Base.invokelatest(gdef) === want    # native codegen
        @test Core.eval(B3AIMin, b3a_thunk(irc)) === want  # interpreted thunk
        @test Core.eval(B3AIDef, b3a_thunk(irc)) === want  # default-mode thunk
    end
end

# ---------------------------------------------------------------------------
# (#29262 program 2) the parallel-move monster: a φ block whose phis read
# each other's PRE-update values across a backedge, nested inside a second
# loop level. Region form: the carried-arg lists of two nested loops with a
# multi-level continue — the typed exit synthesizes exactly the original's
# adversarial φ chain (φ6=:b, φ7=φ6_old, φ8=φ7_old), and every engine must
# implement the simultaneous-assignment semantics.
# ---------------------------------------------------------------------------

function b3a_build_parallel_move()
    b = Builder(name = :p2)
    append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"globalref", GlobalRef(B3AIGlob, :test29262u); type = Bool)
    # outer loop carries (o5, o6, o7, o8) — inits are the original's entry
    # edge values; %8's NULL init is unobservable in the original (never
    # read before the first backedge assigns it), so any placeholder is
    # semantics-preserving — :x here.
    outer = append_stmt!(b, K"loop", false, :a, :c, :x; type = Any)
    obody = UnifiedIR.open_region!(b, outer; kind = UnifiedIR.REGION_LOOP_BODY)
    o5 = append_stmt!(b, K"region_arg"; type = Any)
    o6 = append_stmt!(b, K"region_arg"; type = Any)
    o7 = append_stmt!(b, K"region_arg"; type = Any)
    o8 = append_stmt!(b, K"region_arg"; type = Any)
    # inner loop carries (i10, i14): entered from the outer body with
    # (t, o8) — the original's B2→B3 edge values
    inner = append_stmt!(b, K"loop", op_stmt(t), op_stmt(o8); type = Any)
    ibody = UnifiedIR.open_region!(b, inner; kind = UnifiedIR.REGION_LOOP_BODY)
    i10 = append_stmt!(b, K"region_arg"; type = Any)
    i14 = append_stmt!(b, K"region_arg"; type = Any)
    # !o5 → the original's backedge 16→B2: continue the OUTER loop with
    # (true, :b, o6_old, o7_old) — the parallel move under test
    no5 = append_stmt!(b, K"call", GlobalRef(Base, :!), o5; type = Any)
    fi1 = append_stmt!(b, K"if", no5; type = Nothing)
    UnifiedIR.open_region!(b, fi1; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"continue", op_region(obody), true, true, :b,
                 op_stmt(o6), op_stmt(o7))
    UnifiedIR.close_region!(b)
    # !i10 → the original's backedge 17→B3: continue the INNER loop
    ni10 = append_stmt!(b, K"call", GlobalRef(Base, :!), i10; type = Any)
    fi2 = append_stmt!(b, K"if", ni10; type = Nothing)
    UnifiedIR.open_region!(b, fi2; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"continue", op_region(ibody), true, true, :b)
    UnifiedIR.close_region!(b)
    # both conditions true → B5: leave with the observed tuple
    append_stmt!(b, K"break", op_region(obody), op_stmt(o6), op_stmt(o7),
                 op_stmt(o8), op_stmt(i14))
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"unreachable")
    UnifiedIR.close_region!(b)
    e1 = append_stmt!(b, K"extract", op_stmt(outer), op_inline(1); type = Any)
    e2 = append_stmt!(b, K"extract", op_stmt(outer), op_inline(2); type = Any)
    e3 = append_stmt!(b, K"extract", op_stmt(outer), op_inline(3); type = Any)
    e4 = append_stmt!(b, K"extract", op_stmt(outer), op_inline(4); type = Any)
    tup = append_stmt!(b, K"call", GlobalRef(Core, :tuple), e1, e2, e3, e4; type = Any)
    append_stmt!(b, K"return", tup)
    return finish!(b)
end

@testset "interpreter matches codegen: φ parallel moves (#29262)" begin
    ir = b3a_build_parallel_move()
    @test UnifiedIR.verify_ir(ir; level = 1)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    # the loop headers became real φ blocks (the adversarial chain included)
    @test count(s -> s isa Core.PhiNode, irc.stmts.stmt) >= 6
    oc = Core.OpaqueClosure(irc)
    gmin = b3a_ssa_method(B3AIMin, irc)
    gdef = b3a_ssa_method(B3AIDef, irc)
    for (flag, want) in ((true, (:b, :a, :c, :c)), (false, (:b, :a, :c, :b)))
        b3a_setflag!(flag)
        @test UnifiedIR.interpret(ir, nothing) === want
        @test oc() === want
        @test Base.invokelatest(gmin) === want    # runtime AST interpreter
        @test Base.invokelatest(gdef) === want    # native codegen
        @test Core.eval(B3AIMin, b3a_thunk(irc)) === want
        @test Core.eval(B3AIDef, b3a_thunk(irc)) === want
    end
    # exit_lowered leg: the `continue` lowering performs a real parallel
    # move (F1 fixed — sources naming carried slots of the same loop are
    # snapshotted before any slot is written; exit_lowered.jl
    # `emit_parallel_binds!`), so the swap/rotation chain executes as φ
    # semantics on the slot-form path too.
    b3a_setflag!(true)
    glow = UnifiedCompiler.define_ir_method!(B3AIDef, gensym(:p2low), 1, ir)
    @test Base.invokelatest(glow) === (:b, :a, :c, :c)
    glowmin = UnifiedCompiler.define_ir_method!(B3AIMin, gensym(:p2low), 1, ir)
    @test Base.invokelatest(glowmin) === (:b, :a, :c, :c)
end

# ---------------------------------------------------------------------------
# (#29262 program 3) exceptional stores: the handler observes the last
# Upsilon executed on the throwing prefix; conditional stores select the
# value; the exception stack is restored afterwards. Region form: a
# handler-crossing cell — the typed exit synthesizes the EnterNode/
# PhiC/Upsilon forms (A4), executed by codegen; the slot-form exit runs the
# same program on the runtime interpreter's enter/leave machinery (which is
# how compile=min methods execute exceptional code).
# ---------------------------------------------------------------------------

@testset "interpreter matches codegen: PhiC/Upsilon stores (#29262)" begin
    b = Builder(name = :p3)
    append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"globalref", GlobalRef(B3AIGlob, :test29262u); type = Bool)
    c = append_stmt!(b, K"cell", Any; type = Any)
    tr = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, tr; kind = UnifiedIR.REGION_BODY)
    append_stmt!(b, K"cell_set", c, :b)               # υ(:b) — always runs
    fi = append_stmt!(b, K"if", op_stmt(t); type = Nothing)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"cell_set", c, :a)               # υ(:a) — when t
    append_stmt!(b, K"result")
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"throw_undef_if_not", false, :expected)  # always throws
    append_stmt!(b, K"result", :never)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, tr; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    g = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)  # φᶜ read
    append_stmt!(b, K"result", g)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", tr)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    # the handler-crossing cell became real exception SSA
    @test count(s -> s isa Core.PhiCNode, irc.stmts.stmt) >= 1
    @test count(s -> s isa Core.UpsilonNode, irc.stmts.stmt) >= 2
    @test count(s -> s isa Core.EnterNode, irc.stmts.stmt) == 1
    oc = Core.OpaqueClosure(irc)
    glow = UnifiedCompiler.define_ir_method!(B3AIDef, gensym(:p3low), 1, ir)
    glowmin = UnifiedCompiler.define_ir_method!(B3AIMin, gensym(:p3low), 1, ir)
    # the original asserts `isempty(current_exceptions())` after the handler
    # paths — the :pop_exception restore. Asserted here as a depth delta so
    # the check is self-contained: earlier suite tests execute unified-
    # compiled catch handlers that LEAK the exception stack (exit_lowered's
    # break/continue emission runs :leave for crossed trys but never
    # :pop_exception for crossed handler scopes — its return path does both;
    # runtests.jl's inlined-usetrydiv handler exits its multi-return loop
    # wrapper via such a break), so the absolute stack is not necessarily
    # empty when this testset runs. See the B3a port map, finding F7.
    depth0 = length(Base.current_exceptions())
    for (flag, want) in ((true, :a), (false, :b))
        b3a_setflag!(flag)
        @test UnifiedIR.interpret(ir, nothing) === want
        @test oc() === want                            # codegen over φᶜ/υ
        @test Base.invokelatest(glow) === want         # compiled slot form
        @test Base.invokelatest(glowmin) === want      # AST-interpreted EH
    end
    @test length(Base.current_exceptions()) == depth0
    # NOTE (engine contract, documented for Stage D): the typed exit places
    # the initial Upsilon of each handler-crossing cell BEFORE the
    # EnterNode — stock slot2ssa's own convention — but the runtime AST
    # interpreter binds Upsilon→PhiC slots only when the EnterNode executes,
    # so it cannot run this (or stock's equivalent) optimized form directly;
    # b3a_ssa_method over `irc` would fault in the Upsilon handler. Engine-
    # level φᶜ interpretation therefore stays covered by the hand-SSA
    # fixtures of Compiler/test/interpreter_exec.jl, which satisfy the
    # contract (all Upsilons after the enter) and survive Stage D minus
    # their stock verify_ir line.
end

# ---------------------------------------------------------------------------
# (#29262 program 2's NULL legs) undef φ edges: a value undefined on one
# path is fine while unread, and the guarded read throws UndefVarError —
# on every engine. Region form: a maybe-undef cell (conditional store,
# unconditional read) — the typed exit synthesizes the undef-edge φ, the
# Bool definedness φ, and the :throw_undef_if_not guard.
# ---------------------------------------------------------------------------

@testset "interpreter matches codegen: undef φ edges guarded" begin
    b = Builder(name = :p4)
    append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"globalref", GlobalRef(B3AIGlob, :test29262u); type = Bool)
    c = append_stmt!(b, K"cell", Any; type = Any)
    fi = append_stmt!(b, K"if", op_stmt(t); type = Nothing)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"cell_set", c, :stored)
    append_stmt!(b, K"result")
    UnifiedIR.close_region!(b)
    g = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)
    append_stmt!(b, K"return", g)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    # undef edge + guard synthesized
    @test any(s -> s isa Core.PhiNode && any(i -> !isassigned(s.values, i),
                                             1:length(s.values)), irc.stmts.stmt)
    @test any(s -> Meta.isexpr(s, :throw_undef_if_not), irc.stmts.stmt)
    oc = Core.OpaqueClosure(irc)
    gmin = b3a_ssa_method(B3AIMin, irc)
    gdef = b3a_ssa_method(B3AIDef, irc)
    b3a_setflag!(true)
    @test UnifiedIR.interpret(ir, nothing) === :stored
    @test oc() === :stored
    @test Base.invokelatest(gmin) === :stored
    @test Base.invokelatest(gdef) === :stored
    b3a_setflag!(false)
    @test_throws UndefVarError UnifiedIR.interpret(ir, nothing)
    @test_throws UndefVarError oc()
    @test_throws UndefVarError Base.invokelatest(gmin)
    @test_throws UndefVarError Base.invokelatest(gdef)
    b3a_setflag!(true)
end

end # module B3APortInterpExec
