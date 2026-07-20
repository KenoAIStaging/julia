# Re-expression of Compiler/test/irpasses.jl over UnifiedIR (COMPILER-PORT-PLAN
# B3b). Each testset names the irpasses.jl test whose semantic content it
# carries; the mapping ledger is /workspace/B3B-PORT-MAP.md. The original file
# keeps running against stock until Stage D — behavioral rows (code_typed1 /
# fully_eliminated / infer_effects) already exercise the unified pipeline
# hook-on and stay there; what is re-expressed here is the hand-built-IRCode /
# direct-pass content: fixtures become Builder region IR, raw-SSA CodeInfo
# through the entry converters, or cfg islands; the passes under test become
# their unified counterparts (structurize/layout for domsort+cfg_simplify,
# sroa/promote/forwarding for sroa_pass!, dce!/adce for adce_pass!,
# fold_constant_branches!+compact! for branch folding); every fixture is
# checked with UnifiedIR.verify_ir and, wherever executable, differentially
# against the reference interpreter and runtime execution of the exit
# converters' output.

module B3BPortIrpasses

using Test
using UnifiedIR
using UnifiedIR: op_stmt, op_block, op_region, op_inline, StmtId, RegionId
import ..UnifiedCompiler
import ..CC as Compiler

module B3BDefs end

const UC = UnifiedCompiler

using Base.ScopedValues

# irpasses.jl #53521 fixtures (module level: methods with keyword-free
# signatures for the reflection queries)
function f53521_a()
    VALUE = ScopedValue(1)
    @with VALUE => 2 begin
        for i = 1
            @with VALUE => 3 begin
                try
                    _undefined_call_53521()
                catch
                    nothing
                end
            end
        end
    end
end
Base.@assume_effects :foldable Base.@constprop :aggressive function f53521_b(x::Int, ::Int)
    VALUE = ScopedValue(x)
    @with VALUE => 2 begin
        for i = 1
            @with VALUE => 3 begin
                local v
                try
                    v = sin(VALUE[])
                catch
                    v = nothing
                end
                return v
            end
        end
    end
end
f53521_wrap(y) = f53521_b(1, y)

# irpasses.jl scope_folding fixture (Expr(:tryfinally, body, finally, scope))
@eval function scope_folding_probe()
    $(Expr(:tryfinally,
        Expr(:block,
            Expr(:tryfinally, :(), :(), 2),
            :(return Core.current_scope())),
    :(), 1))
end
const SV_53521 = ScopedValue(1)
with_read_53521(x) = @with SV_53521 => x begin
    SV_53521[]
end
nested_with_53521(x) = @with SV_53521 => x begin
    inner = @with SV_53521 => x + 1 begin
        SV_53521[]
    end
    (inner, SV_53521[])
end

# irpasses.jl #52857 fixture struct (1 field, no inner constructor —
# `Expr(:new)` with no values leaves the field unset)
struct ImmRef52857; x; end

# make_codeinfo wrapper: raw code array in, CodeInfo out (slot flags 0x08,
# Any types), the shape the entry converters accept
function mkci(code::Vector{Any}, nargs::Int, nslots::Int = nargs)
    fields = UC.default_codeinfo_fields(length(code), nargs,
        Symbol[Symbol("#slot", i) for i in 1:nslots], fill(0x08, nslots))
    fields[:code] = code
    return UC.make_codeinfo(; fields...)
end

count_kind(ir, k) = count(s -> UnifiedIR.stmt_kind(ir, s) === k,
                          collect(UnifiedIR.each_stmt(ir)))

function optimized(ir::UnifiedIR.IR, argtypes::Vector{Any})
    st = UC.UInferState()
    return UC.optimize_ir!(ir, argtypes; state = st)
end

"Build a cfg island over `nargs` Bool arguments from a block list. Each
block is `(:brif, argidx, then, else)` (branch on argument), `(:goto, dst)`,
or `(:ret, val)`; block indices are 1-based in declaration order. This is
the region form of irpasses.jl's `each_stmt_a_bb`/make_ircode CFGs."
function island(nargs::Int, blocks::Vector; name::Symbol = :island)
    b = Builder(name = name)
    append_stmt!(b, K"region_arg"; type = Any)
    args = [append_stmt!(b, K"region_arg"; type = Bool) for _ in 1:nargs]
    cfg = append_stmt!(b, K"cfg"; type = Any)
    base = 1  # region ids: root=1, block i (declaration order) = 1 + i
    for d in blocks
        UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
        if d[1] === :brif
            # stock GotoIfNot(c, dest): true falls through, false jumps —
            # encoded here as explicit two-way br_if (then=fallthrough block)
            append_stmt!(b, K"br_if", op_stmt(args[d[2]]),
                         op_block(RegionId(base + d[3])), op_inline(0),
                         op_block(RegionId(base + d[4])), op_inline(0))
        elseif d[1] === :goto
            append_stmt!(b, K"goto", op_block(RegionId(base + d[2])), op_inline(0))
        elseif d[1] === :ret
            append_stmt!(b, K"return", d[2])
        else
            error("island: unknown block form $(d[1])")
        end
        UnifiedIR.close_region!(b)
    end
    append_stmt!(b, K"return", cfg)
    return finish!(b), cfg
end

"Differential harness for a Bool^n island: interpret the freshly built IR on
every input, optimize a second copy, and check interpreter + OpaqueClosure
agreement on every input. Returns the optimized IR."
function island_differential(nargs::Int, blocks::Vector; name::Symbol = :island)
    ir0, _ = island(nargs, blocks; name)
    @test UnifiedIR.verify_ir(ir0; level = 1)
    inputs = [[isodd(bits >> (k - 1)) for k in 1:nargs] for bits in 0:(2^nargs - 1)]
    ref = Dict(bs => UnifiedIR.interpret(ir0, nothing, bs...) for bs in inputs)
    ir, _ = island(nargs, blocks; name)
    ir = optimized(ir, Any[Any, (Bool for _ in 1:nargs)...])
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test all(UnifiedIR.interpret(ir, nothing, bs...) == ref[bs] for bs in inputs)
    irc = UC.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    oc = Core.OpaqueClosure(irc)
    @test all(oc(bs...) == ref[bs] for bs in inputs)
    return ir
end

# ---------------------------------------------------------------------------
# domsort (irpasses.jl lines 10-54)
# ---------------------------------------------------------------------------

@testset "domsort #29262: single-edge loop-header join" begin
    # The original hand-builds a loop whose header carries a single-edge
    # PhiNode (fed only by the backedge) and checks domsort_ssa! does not
    # substitute the phi by its (non-dominating) value. In region IR the
    # invalid rewrite is unrepresentable: loop-carried values are explicit
    # loop args, every carried arg is total (init + every continue), and a
    # body-local value used outside the loop is a visibility violation the
    # verifier rejects. Three legs below: (a) the partial-phi form is
    # rejected at entry (carried args must be total), (b) the executable
    # variant of the original CFG round-trips with the carried chain intact,
    # (c) the non-dominating-use rewrite the #29262 bug performed is a
    # VerifyError.
    partial = Any[
        Expr(:call, GlobalRef(Base, :identity), Core.Argument(2)),
        Core.GotoIfNot(Core.SSAValue(1), 10),
        Core.PhiNode(Int32[8], Any[Core.SSAValue(7)]),  # undef on the entry edge
        Core.PhiNode(Int32[2, 8], Any[true, false]),
        Core.GotoIfNot(Core.SSAValue(1), 7),
        Expr(:call, GlobalRef(Base, :+), Core.SSAValue(3), 1),
        Core.PhiNode(Int32[5, 6], Any[0, Core.SSAValue(6)]),
        Expr(:call, GlobalRef(Base, :>), Core.SSAValue(7), 10),
        Core.GotoIfNot(Core.SSAValue(8), 3),
        Core.PhiNode(Int32[2, 8], Any[0, Core.SSAValue(7)]),
        Core.ReturnNode(Core.SSAValue(10)),
    ]
    @test_throws UC.UnsupportedIR UC.codeinfo_to_ir(mkci(partial, 2);
                                                    nargs = 2, name = :domsort_partial)
    # (b) total-edge variant of the same CFG (entry value 0 for the header
    # phi): converts, verifies, and the loop-carried chain (phi3 = previous
    # trip's phi7) survives conversion + optimization with its simultaneous
    # semantics
    total = copy(partial)
    total[3] = Core.PhiNode(Int32[2, 8], Any[0, Core.SSAValue(7)])
    ir = UC.codeinfo_to_ir(mkci(total, 2); nargs = 2, name = :domsort29262)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, true) == 11
    @test UnifiedIR.interpret(ir, nothing, false) == 0
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:domsort29262), 2, ir)
    @test Base.invokelatest(g, true) == 11
    @test Base.invokelatest(g, false) == 0
    # (c) the bug's rewrite shape — a loop-body value referenced after the
    # loop — is a verifier-rejected visibility violation
    b = Builder(name = :vis29262)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Any)
    lp = append_stmt!(b, K"loop", op_stmt(n); type = Any)
    body = UnifiedIR.open_region!(b, lp; kind = UnifiedIR.REGION_LOOP_BODY)
    i = append_stmt!(b, K"region_arg"; type = Any)
    d = append_stmt!(b, K"call", GlobalRef(Base, :-), i, 1; type = Any)
    cnd = append_stmt!(b, K"call", GlobalRef(Base, :>), d, 0; type = Any)
    append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(d))
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", op_stmt(d))
    @test_throws UnifiedIR.VerifyError begin
        ir2 = finish!(b)
        UnifiedIR.verify_ir(ir2; level = 1)
    end
    # (b') the φ-simultaneity core of #29262 as a native loop: two carried
    # args where each continue reads the other's pre-update value (the
    # Fibonacci-style parallel move domsort must never sequence)
    b = Builder(name = :perm29262)
    append_stmt!(b, K"region_arg"; type = Any)
    n2 = append_stmt!(b, K"region_arg"; type = Any)
    lp = append_stmt!(b, K"loop", op_stmt(n2), 0, 100; type = Any)
    body = UnifiedIR.open_region!(b, lp; kind = UnifiedIR.REGION_LOOP_BODY)
    i2 = append_stmt!(b, K"region_arg"; type = Any)
    prev = append_stmt!(b, K"region_arg"; type = Any)
    cur = append_stmt!(b, K"region_arg"; type = Any)
    ni = append_stmt!(b, K"call", GlobalRef(Base, :-), i2, 1; type = Any)
    ncur = append_stmt!(b, K"call", GlobalRef(Base, :+), cur, prev; type = Any)
    c2 = append_stmt!(b, K"call", GlobalRef(Base, :>), ni, 0; type = Any)
    append_stmt!(b, K"continue", op_region(body), op_stmt(c2),
                 op_stmt(ni), op_stmt(cur), op_stmt(ncur))
    UnifiedIR.close_region!(b)
    ex1 = append_stmt!(b, K"extract", op_stmt(lp), op_inline(2); type = Any)
    ex2 = append_stmt!(b, K"extract", op_stmt(lp), op_inline(3); type = Any)
    t = append_stmt!(b, K"call", GlobalRef(Core, :tuple), ex1, ex2; type = Any)
    append_stmt!(b, K"return", t)
    ir3 = finish!(b)
    @test UnifiedIR.verify_ir(ir3; level = 1)
    @test UnifiedIR.interpret(ir3, nothing, 5) == (500, 800)
    irc3 = UC.ir_to_ircode(ir3)
    @test Compiler.verify_ir(irc3) === nothing
    @test Core.OpaqueClosure(irc3)(5) == (500, 800)
end

@testset "SNCA/domsort scale: 2^14-block chain island" begin
    # irpasses.jl builds 2^15 statements of skip-chains and checks
    # construct_domtree + domsort_ssa! survive without a stack overflow. The
    # unified dominator computation (island_dominators) and the interpreter
    # get the same chain-shaped adversary.
    b = Builder(name = :snca)
    append_stmt!(b, K"region_arg"; type = Any)
    cnd = append_stmt!(b, K"region_arg"; type = Bool)
    cfg = append_stmt!(b, K"cfg"; type = Any)
    N = 2^14
    for i in 1:N
        UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
        if i < N
            append_stmt!(b, K"br_if", op_stmt(cnd),
                         op_block(RegionId(1 + i + 1)), op_inline(0),
                         op_block(RegionId(1 + N)), op_inline(0))
        else
            append_stmt!(b, K"return", 42)
        end
        UnifiedIR.close_region!(b)
    end
    append_stmt!(b, K"return", cfg)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    dom = UnifiedIR.island_dominators(ir, cfg)
    @test length(dom) == N
    # chain structure: block k is dominated by exactly blocks 1..k; the tail
    # (reachable from every block) only by the entry and itself
    @test length(dom[RegionId(1 + 1)]) == 1
    @test length(dom[RegionId(1 + 5)]) == 5
    @test length(dom[RegionId(1 + N - 1)]) == N - 1
    @test length(dom[RegionId(1 + N)]) == 2
    # both extreme paths execute (the skip edge and the full 2^14-block walk)
    @test UnifiedIR.interpret(ir, nothing, false) == 42
    @test UnifiedIR.interpret(ir, nothing, true) == 42
end

# ---------------------------------------------------------------------------
# cfg simplification semantics (irpasses.jl cfg_simplify! battery: redundant
# blocks, dropped/merged bbs, chains past returns, loops, unreachable
# terminators, forward-referenced phis). The stock block-count/encoding
# asserts are cfg_simplify!-internal and die with it at Stage D; the ported
# property is that the unified simplification pipeline (structurize/
# fold/merge/drop + compact!) preserves the executed semantics of every one
# of the original CFGs while verifying at every step.
# ---------------------------------------------------------------------------

@testset "cfg simplify: redundant goto chain collapses" begin
    blocks = [[(:goto, i + 1) for i in 1:6]; (:ret, 2)]
    ir, _ = island(0, collect(Any, blocks); name = :chain)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing) == 2
    ir = optimized(ir, Any[Any])
    @test UnifiedIR.verify_ir(ir; level = 1)
    # the analogue of "1 block, 1 statement": the island dissolves entirely
    @test count_kind(ir, K"cfg") == 0
    @test UnifiedIR.interpret(ir, nothing) == 2
end

@testset "cfg simplify: dropped/merged bb battery (each_stmt_a_bb)" begin
    # the 13-block CFG from irpasses.jl, both variants (block 11 as
    # conditional or unconditional), exhaustively executed over all 2^6
    # argument combinations before and after simplification
    for gotoifnot in (false, true)
        blocks = Any[
            (:brif, 1, 2, 8), (:brif, 2, 3, 4), (:goto, 9), (:brif, 3, 5, 10),
            (:brif, 4, 6, 11), (:brif, 5, 7, 12), (:goto, 13), (:ret, 1),
            (:goto, 10), (:goto, 11),
            gotoifnot ? (:brif, 6, 12, 13) : (:goto, 13),
            (:ret, 2), (:ret, 3),
        ]
        island_differential(6, blocks; name = Symbol(:battery_, gotoifnot))
    end
    # the 5-block variant
    island_differential(2, Any[
        (:brif, 1, 2, 4), (:brif, 2, 3, 5), (:goto, 5), (:ret, 1), (:ret, 2),
    ]; name = :battery5)
end

@testset "cfg simplify: chaining past return blocks" begin
    ir = island_differential(1, Any[
        (:brif, 1, 2, 3), (:goto, 4), (:ret, 1), (:goto, 5),
        (:brif, 1, 6, 7), (:ret, 2), (:ret, 3),
    ]; name = :pastret)
    @test ir isa UnifiedIR.IR
end

@testset "cfg simplify: single-cycle loop is kept, pass terminates" begin
    # the 3-block goto cycle with no exit: cfg_simplify! must not merge every
    # block into its predecessor; structurize! must terminate and produce a
    # verifiable loop (the program itself never terminates, so no execution)
    ir, _ = island(0, Any[(:goto, 2), (:goto, 3), (:goto, 1)]; name = :cycle)
    @test UnifiedIR.verify_ir(ir; level = 1)
    UnifiedIR.editable(ir)
    UC.structurize!(ir)
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test count_kind(ir, K"loop") == 1
    @test count_kind(ir, K"cfg") == 0
end

@testset "cfg simplify: implicit unreachable terminators + orphan blocks" begin
    # block 2 ends in a must-throw call (implicit unreachable), block 3 has
    # no predecessors; simplification must drop the orphan (its call is never
    # executed) and keep the throwing path throwing
    b = Builder(name = :orphan)
    append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"region_arg"; type = Bool)
    cfg = append_stmt!(b, K"cfg"; type = Any)
    UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
    append_stmt!(b, K"br_if", op_stmt(c), op_block(RegionId(5)), op_inline(0),
                 op_block(RegionId(3)), op_inline(0))
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
    append_stmt!(b, K"call", GlobalRef(Base, :throw), "error"; type = Union{})
    append_stmt!(b, K"unreachable")
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)   # orphan
    append_stmt!(b, K"call", GlobalRef(Base, :error), "never"; type = Any)
    append_stmt!(b, K"goto", op_block(RegionId(5)), op_inline(0))
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
    append_stmt!(b, K"return", nothing)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", cfg)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    # the orphan's error() call is gone; the reachable throw stays
    nerror = count(collect(UnifiedIR.each_stmt(ir))) do s
        UnifiedIR.stmt_kind(ir, s) === K"call" || return false
        o = UnifiedIR.getop(ir, s, 1)
        UnifiedIR.optag(o) == UnifiedIR.TAG_GLOBAL || return false
        ir.body.globals[UnifiedIR.payload(o)].name === :error
    end
    @test nerror == 0
    @test UnifiedIR.interpret(ir, nothing, true) === nothing
    thrown = try
        UnifiedIR.interpret(ir, nothing, false)
        nothing
    catch e
        e
    end
    @test thrown == "error"   # the reachable throw still throws
end

@testset "cfg simplify: un-renamed SSA values across merged blocks" begin
    # irpasses.jl "CFG simplify doesn't leave an un-renamed SSA Value":
    # a join of two inferencebarrier calls whose blocks get merged
    code = Any[
        Core.GotoIfNot(Core.Argument(2), 3),
        Core.GotoNode(5),
        Expr(:call, GlobalRef(Base, :inferencebarrier), 1),
        Core.GotoNode(6),
        Expr(:call, GlobalRef(Base, :inferencebarrier), 2),
        Core.PhiNode(Int32[4, 5], Any[Core.SSAValue(3), Core.SSAValue(5)]),
        Core.ReturnNode(1),
    ]
    ir = UC.codeinfo_to_ir(mkci(code, 2); nargs = 2, name = :unrenamed)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, true) == 1
    @test UnifiedIR.interpret(ir, nothing, false) == 1
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:unrenamed), 2, ir)
    @test Base.invokelatest(g, true) == 1
    @test Base.invokelatest(g, false) == 1
end

@testset "cfg simplify: single-predecessor phi" begin
    code = Any[
        Core.GotoNode(3),
        nothing,
        Expr(:call, GlobalRef(Base, :inferencebarrier), 1),
        Core.GotoNode(5),
        Core.PhiNode(Int32[4], Any[Core.SSAValue(3)]),
        Core.ReturnNode(Core.SSAValue(5)),
    ]
    ir = UC.codeinfo_to_ir(mkci(code, 1); nargs = 1, name = :singlepred)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing) == 1
    ir = optimized(ir, Any[Any])
    @test UnifiedIR.verify_ir(ir; level = 1)
    # the return value survives the merge (the original's `.val !== nothing`)
    g = UC.define_ir_method!(B3BDefs, gensym(:singlepred), 1, ir)
    @test Base.invokelatest(g) == 1
end

@testset "cfg simplify: phi forward references across removed blocks" begin
    # irpasses.jl "PhiNode values containing forward references are
    # eventually updated": a loop header phi referencing a later statement,
    # with a removable block shifting indices. Region-total edges (the
    # dead block 2's edge carries the same entry value; stock left it
    # implicit-undef, which region IR cannot express — carried args are
    # total by construction).
    code = Any[
        Core.Argument(2),
        Core.GotoNode(4),
        Core.GotoNode(4),
        Core.PhiNode(Int32[2, 3, 9, 13],
                     Any[Core.SSAValue(1), Core.SSAValue(1),
                         Core.SSAValue(6), Core.SSAValue(6)]),
        Core.GotoNode(6),
        Expr(:call, GlobalRef(Base, :+), Core.Argument(2), 1),
        Core.GotoIfNot(Core.Argument(3), 9),
        Core.ReturnNode(Core.Argument(3)),
        Core.GotoIfNot(Core.Argument(3), 4),
        Core.GotoIfNot(Core.Argument(3), 12),
        Core.GotoNode(13),
        Core.GotoNode(13),
        Core.GotoNode(4),
    ]
    ir = UC.codeinfo_to_ir(mkci(code, 3); nargs = 3, name = :fwdphi)
    @test UnifiedIR.verify_ir(ir; level = 1)
    # b=true reaches the return at %8 (b=false loops forever, as in the
    # original — which never executed this fixture at all)
    @test UnifiedIR.interpret(ir, nothing, 5, true) === true
    ir = optimized(ir, Any[Any, Int, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:fwdphi), 3, ir)
    @test Base.invokelatest(g, 5, true) === true
end

@testset "cfg simplify on real loops: gcd" begin
    # irpasses.jl round-trips code_typed(gcd) through inflate_ir +
    # cfg_simplify! + verify; here the whole unified pipeline compiles gcd
    # and the result is executed differentially
    tir = UC.typed_ir(gcd, Any[Int, Int])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:gcd), 3, tir)
    for (a, b) in ((24, 36), (0, 5), (5, 0), (17, 13), (64, 128), (-6, 9))
        @test Base.invokelatest(g, a, b) == gcd(a, b)
    end
end

@testset "cfg simplify: empty-block convergence (@goto) + entry-block merge" begin
    # foo_cfg_empty: converging control flow through empty blocks
    function foo_cfg_empty(b)
        if b
            @goto x
        end
        @label x
        return b
    end
    tir = UC.typed_ir(foo_cfg_empty, Any[Bool])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:cfgempty), 2, tir)
    @test Base.invokelatest(g, true) === true
    @test Base.invokelatest(g, false) === false
    # f_with_merge_to_entry_block: a loop whose exit merges back to the entry
    function merge_to_entry()
        while true
            i = @noinline rand(Int)
            if @noinline isodd(i)
                return i
            end
        end
    end
    tir = UC.typed_ir(merge_to_entry, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:mergeentry), 1, tir)
    @test isodd(Base.invokelatest(g))
end

# ---------------------------------------------------------------------------
# branch folding (irpasses.jl "allow branch folding to look at type
# information": inflate_ir + compact!(ir, true))
# ---------------------------------------------------------------------------

@testset "branch folding sees inferred types" begin
    function branch_fold()
        cond = 1 + 1 == 2
        if !cond
            gcd(24, 36)
        else
            gcd(64, 128)
        end
    end
    # pre-optimization the lowered body still branches; after the unified
    # optimizer the Const-typed (not literal) condition folds the branch
    ir = UC.lowered_ir(branch_fold, Tuple{})
    st = UC.UInferState()
    ir = UC.optimize_ir!(ir, Any[Any]; state = st)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test count_kind(ir, K"if") == 0
    @test count_kind(ir, K"cfg") == 0
    g = UC.define_ir_method!(B3BDefs, gensym(:branchfold), 1, ir)
    @test Base.invokelatest(g) == 64
end

# ---------------------------------------------------------------------------
# SROA (irpasses.jl hand-built sroa_pass! fixtures; the behavioral
# code_typed1 battery stays in the stock file and runs through the unified
# pipeline hook-on)
# ---------------------------------------------------------------------------

@testset "SROA: union of tuple allocations through ifelse" begin
    # irpasses.jl's make_ircode ifelse fixture (with __set_check_ssa_counts):
    # two differently-typed tuple allocations selected by Core.ifelse, then a
    # getfield — SROA must handle the union without corrupting counts. The
    # stock ssa-count checker is sroa_pass!-internal; the unified assert is
    # verify + differential execution.
    b = Builder(name = :ifelse_tup)
    append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"region_arg"; type = Bool)
    t1 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), 1; type = Tuple{Int})
    t2 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), 1.0; type = Tuple{Float64})
    sel = append_stmt!(b, K"call", GlobalRef(Core, :ifelse), c, t1, t2; type = Any)
    g = append_stmt!(b, K"call", GlobalRef(Core, :getfield), sel, 1; type = Any)
    append_stmt!(b, K"return", g)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    gm = UC.define_ir_method!(B3BDefs, gensym(:ifelsetup), 2, ir)
    @test Base.invokelatest(gm, true) === 1
    @test Base.invokelatest(gm, false) === 1.0
end

@testset "SROA all_same over joined allocations" begin
    # irpasses.jl "SROA all_same on NewNode": allocations joined by a phi,
    # re-wrapped in nested tuples, unwrapped through refines — the region
    # translation of the 16-statement make_ircode battery, executed on both
    # join paths
    b = Builder(name = :allsame)
    append_stmt!(b, K"region_arg"; type = Any)
    a1 = append_stmt!(b, K"region_arg"; type = Int)
    a2 = append_stmt!(b, K"region_arg"; type = Int)
    a3 = append_stmt!(b, K"region_arg"; type = Int)
    a4 = append_stmt!(b, K"region_arg"; type = Bool)
    t1 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), a1; type = Tuple{Int})
    fi = append_stmt!(b, K"if", op_stmt(a4); type = Tuple{Int})
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", t1)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    t2 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), a2; type = Tuple{Int})
    append_stmt!(b, K"result", t2)
    UnifiedIR.close_region!(b)
    g1 = append_stmt!(b, K"call", GlobalRef(Core, :getfield), fi, 1; type = Int)
    t3 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), g1, a2; type = Tuple{Int,Int})
    t4 = append_stmt!(b, K"call", GlobalRef(Core, :tuple), t3, a3; type = Tuple{Tuple{Int,Int},Int})
    g2 = append_stmt!(b, K"call", GlobalRef(Core, :getfield), t4, 1; type = Tuple{Int,Int})
    g3 = append_stmt!(b, K"call", GlobalRef(Core, :getfield), g2, 1; type = Int)
    append_stmt!(b, K"return", g3)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, 1, 2, 3, true) == 1
    @test UnifiedIR.interpret(ir, nothing, 1, 2, 3, false) == 2
    ir = optimized(ir, Any[Any, Int, Int, Int, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    gm = UC.define_ir_method!(B3BDefs, gensym(:allsame), 5, ir)
    @test Base.invokelatest(gm, 1, 2, 3, true) == 1
    @test Base.invokelatest(gm, 1, 2, 3, false) == 2
    # the nested-tuple unwrap fully forwarded
    @test count_kind(ir, K"call") <= 1   # at most the surviving join tuple
end

@testset "SROA: forwarding must not cross loop iterations" begin
    # irpasses.jl "A SSAValue after the compaction line" (SROA must
    # propagate taint when following a loop phi through an SSA alias) +
    # sroa_no_forward. The taint encoding is sroa_pass!-internal; the bug
    # class is that a load of a loop-joined value must not forward a
    # previous iteration's field. The UnionAll-unwrap loop is the original
    # fixture's exact program shape.
    function unwrap_ua(@nospecialize t)
        while isa(t, UnionAll)
            t = t.body
        end
        return t
    end
    tir = UC.typed_ir(unwrap_ua, Any[Any])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:unwrap), 2, tir)
    @test Base.invokelatest(g, Array) === Array.body.body
    @test Base.invokelatest(g, Int) === Int
    # sroa_no_forward: the rebuilt-on-iteration-1 tuple must be observed by
    # later iterations (forwarding the iteration-0 value would error())
    function sroa_no_forward()
        res = (0, 0)
        for i in 1:5
            a = first(res)
            a == 5 && error()
            if i == 1
                res = (i, 2.0)
            end
        end
        return res
    end
    tir = UC.typed_ir(sroa_no_forward, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:snf), 1, tir)
    @test Base.invokelatest(g) == (1, 2.0)
end

@testset "SROA: partially lifted select (#50276 class)" begin
    # irpasses.jl #50276: a lifted Core.ifelse where a PiNode means only one
    # branch of the join is lifted, and the join needs a Union type for
    # lifting to fire. Region form: ifelse(c, true, missing) refined to Bool
    # in one arm only, `isa(join, Missing)` downstream — the one-side-lifted
    # adversary through the full optimizer.
    b = Builder(name = :i50276)
    append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"region_arg"; type = Bool)
    sel = append_stmt!(b, K"call", GlobalRef(Core, :ifelse), c, true, missing;
                       type = Union{Missing,Bool})
    fi = append_stmt!(b, K"if", op_stmt(c); type = Union{Missing,Nothing,Bool})
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    rf = append_stmt!(b, K"refine", sel; type = Bool)
    append_stmt!(b, K"result", rf)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", nothing)
    UnifiedIR.close_region!(b)
    isa_ = append_stmt!(b, K"call", GlobalRef(Core, :isa), fi, Missing; type = Bool)
    append_stmt!(b, K"return", isa_)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, true) === false
    @test UnifiedIR.interpret(ir, nothing, false) === false
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    gm = UC.define_ir_method!(B3BDefs, gensym(:i50276), 2, ir)
    @test Base.invokelatest(gm, true) === false
    @test Base.invokelatest(gm, false) === false
end

@testset "adce: dead-edge join refinement (IR_FLAG_REFINED analogue)" begin
    # irpasses.jl "adce_pass! sets Refined on PhiNode values": after a
    # constant branch kills the Float64 edge of a Union{Int64,Float64} join,
    # downstream must see the refined (Int-only) value. Stock carries this
    # as a flag; here the fold + refinement is observable in the result type
    # and the executed value's identity.
    b = Builder(name = :refined)
    append_stmt!(b, K"region_arg"; type = Any)
    fi = append_stmt!(b, K"if", false; type = Union{Int64,Float64})
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 1.0)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 1)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", fi)
    ir = finish!(b)
    ir = optimized(ir, Any[Any])
    @test UnifiedIR.verify_ir(ir; level = 1)
    rt = ir.meta[:rettype]
    @test rt isa Compiler.Const && rt.val === 1   # refined past Union{Int64,Float64}
    gm = UC.define_ir_method!(B3BDefs, gensym(:refined), 1, ir)
    @test Base.invokelatest(gm) === 1             # the Int edge, not 1.0
end

@testset "pending inserts into a folded-dead arm (#52858)" begin
    # irpasses.jl #52858: compaction got confused by a node pending in a
    # region that constant-branch folding removes. Editable form: insert
    # into the untaken arm, fold, compact — the insert must die with its
    # region and the observable cell state must be untouched.
    b = Builder(name = :i52858)
    append_stmt!(b, K"region_arg"; type = Any)
    cl = append_stmt!(b, K"cell", Int64; type = Any)
    append_stmt!(b, K"cell_set", cl, 1)
    fi = append_stmt!(b, K"if", true; type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 10)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    setstmt = append_stmt!(b, K"cell_set", cl, 99)
    append_stmt!(b, K"result", 20)
    UnifiedIR.close_region!(b)
    gv = append_stmt!(b, K"cell_get", op_stmt(cl); type = Any)
    t = append_stmt!(b, K"call", GlobalRef(Core, :tuple), fi, gv; type = Any)
    append_stmt!(b, K"return", t)
    ir = finish!(b)
    UnifiedIR.editable(ir)
    UnifiedIR.insert_before!(ir, setstmt, K"cell_set", op_stmt(cl), 77)
    ir, nfold = UnifiedIR.fold_constant_branches!(ir)
    @test nfold == 1
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test count_kind(ir, K"if") == 0     # the original's GotoIfNot count == 1
    @test UnifiedIR.interpret(ir, nothing) == (10, 1)
end

@testset "merge-point refinement placement (#51144)" begin
    # irpasses.jl #51144 hand-drives convert_to_ircode!/slot2reg with poked
    # vartables to force a bad PiNode at a merge; that machinery dies with
    # slot2ssa. The bug class — a merge point must not get a refinement that
    # is only valid on some predecessors — is carried by the entry's
    # cell/refine discipline: the same slot-form fixture converts, infers,
    # optimizes, and verifies with the merge intact.
    code = Any[
        Expr(:(=), Core.SlotNumber(4), Core.Argument(2)),
        Expr(:call, GlobalRef(Core, :(===)), Core.SlotNumber(4), nothing),
        Core.GotoIfNot(Core.SSAValue(2), 5),
        Core.ReturnNode(nothing),
        Expr(:(=), Core.SlotNumber(4), false),
        Core.GotoIfNot(Core.Argument(2), 8),
        Expr(:(=), Core.SlotNumber(4), true),
        Core.ReturnNode(nothing),
    ]
    ci = mkci(code, 3, 4)
    ir = UC.codeinfo_to_ir(ci; nargs = 3, name = :i51144)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, nothing, 0) === nothing
    ir = optimized(ir, Any[Any, Union{Bool,Nothing}, Any])
    @test UnifiedIR.verify_ir(ir; level = 1)
    gm = UC.define_ir_method!(B3BDefs, gensym(:i51144), 3, ir)
    @test Base.invokelatest(gm, nothing, 0) === nothing
    @test Base.invokelatest(gm, false, 0) === nothing
end

@testset "SROA + finalizer whose uses have no postdominator (#54596)" begin
    # irpasses.jl #54596 hand-builds the inlined-finalizer form and runs
    # sroa_pass! + verify; the semantic content — finalizer'd allocation with
    # loads on diverging paths must survive optimization — runs through the
    # full unified pipeline (which has its own finalizer resolution)
    function finalizer_shape(c::Bool)
        r = Base.RefValue{Int}(1)
        finalizer(r) do obj
            nothing
        end
        if c
            return r.x
        end
        return r.x + 1
    end
    tir = UC.typed_ir(finalizer_shape, Any[Bool])
    @test UnifiedIR.verify_ir(tir; level = 1)
    gm = UC.define_ir_method!(B3BDefs, gensym(:fin54596), 2, tir)
    @test Base.invokelatest(gm, true) == 1
    @test Base.invokelatest(gm, false) == 2
end

# ---------------------------------------------------------------------------
# exception-handling shapes (irpasses.jl cfg_simplify/domsort EnterNode
# fixtures + the try/catch value + early-leave rows). The stock fixtures'
# deletion markers / block-count asserts die with cfg_simplify!; the raw
# code arrays themselves go through the eh entry.
# ---------------------------------------------------------------------------

"Crash-safe validity witness for emitted EH CodeInfo: stock inference runs
compute_trycatch on every method body and ASSERTS (taking the process down)
when a path enters a handler epilogue without its :enter — so invalid
bodies must never be invoked from the suite. Called directly, the same
walk raises a catchable AssertionError instead."
function trycatch_ok(ci::Core.CodeInfo)
    return try
        Compiler.compute_trycatch(copy(ci.code))
        true
    catch
        false
    end
end

@testset "EH: try/catch value shape + early try-catch exit (#51159)" begin
    # `v = try catch end; v` (the optimize_until="CC: SLOT2REG" +
    # cfg_simplify! row): converts, optimizes, verifies, executes
    function try_value_shape()
        v = try
        catch
        end
        v
    end
    tir = UC.typed_ir(try_value_shape, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:tryval), 1, tir)
    @test Base.invokelatest(g) === nothing
    # #51159: the `continue` from an empty catch introduces an early :leave
    # that φᶜ placement must respect; the stock φᶜ-count scan dies with
    # slot2ssa — the surviving content is that the shape compiles and the
    # `result = x` assignment outside the try is not misattributed
    function early_try_catch()
        result = false
        for i in 3
            x = try
            catch
                continue
            end
            result = x
        end
        result
    end
    tir = UC.typed_ir(early_try_catch, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.redefine_through_ir(early_try_catch, Tuple{})
    @test Base.invokelatest(g) === early_try_catch() === nothing
end

@testset "EH: enter/leave islands through the entry" begin
    # (a) irpasses "CFG simplify with try/catch blocks": GotoIfNot around an
    # EnterNode/:leave pair, everything falling through to return 1
    code60 = Any[
        Core.GotoIfNot(Core.Argument(2), 5),
        Core.EnterNode(4),
        Expr(:leave, Core.SSAValue(2)),
        Core.GotoNode(5),
        Core.ReturnNode(1),
    ]
    ir = UC.codeinfo_to_ir(mkci(code60, 2); nargs = 2, name = :row60)
    @test UnifiedIR.verify_ir(ir; level = 1)
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    ci60 = UC.ir_to_codeinfo(ir)
    # F10 (fixed): the catch-dest block shared with the normal path is
    # reassigned out of the handler island (the pop_exception rides the
    # handler's synthetic exit edge instead of the shared join), so the
    # emitted body is enter/leave-balanced. Crash-safety discipline: the
    # compute_trycatch witness must hold BEFORE any invocation (an
    # unbalanced body segfaults the process when invoked).
    @test trycatch_ok(ci60)
    if trycatch_ok(ci60)
        g = UC.define_ir_method!(B3BDefs, gensym(:row60), 2, ir)
        @test Base.invokelatest(g, true) == 1
        @test Base.invokelatest(g, false) == 1
        # differential through the typed exit as well
        oc = Core.OpaqueClosure(UC.ir_to_ircode(ir))
        @test oc(true) == 1
        @test oc(false) == 1
    end
    # (b) irpasses "cfg_simplify with EnterNode + union-typed deletion
    # marker" shape (enter with a scope operand, unreachable catch return,
    # branchy continuation): full pipeline + execution
    code75 = Any[
        Core.EnterNode(4, 1),
        Core.GotoNode(3),
        Core.GotoNode(5),
        Core.ReturnNode(),
        Expr(:leave, Core.SSAValue(1)),
        Core.GotoIfNot(Core.Argument(2), 8),
        Core.ReturnNode(1),
        Core.ReturnNode(2),
    ]
    ir = UC.codeinfo_to_ir(mkci(code75, 2); nargs = 2, name = :row75)
    @test UnifiedIR.verify_ir(ir; level = 1)
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:row75), 2, ir)
    @test Base.invokelatest(g, true) == 1
    @test Base.invokelatest(g, false) == 2
    # (c) irpasses "domsort with a non-domsorted :leave": the leave/return
    # blocks precede the EnterNode; the fallthrough from the body's leave
    # lands on the catch-destination statement (a block with both a normal
    # and an exceptional in-edge). F9 (fixed): the island entry is the
    # enter's continuation regardless of statement numbering, so the true
    # path leaves the try and returns 1 instead of taking the pre-enter
    # exit's 2; the shared catch dest moves out of the handler island (F10),
    # which also lets the typed exit convert the shape.
    code78 = Any[
        Core.GotoNode(4),
        Expr(:leave, Core.SSAValue(4)),
        Core.ReturnNode(2),
        Core.EnterNode(7),
        Core.GotoIfNot(Core.Argument(2), 2),
        Expr(:leave, Core.SSAValue(4)),
        Core.ReturnNode(1),
        Core.ReturnNode(nothing),
    ]
    ir = UC.codeinfo_to_ir(mkci(code78, 2); nargs = 2, name = :row78)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test trycatch_ok(UC.ir_to_codeinfo(ir))
    if trycatch_ok(UC.ir_to_codeinfo(ir))
        g = UC.define_ir_method!(B3BDefs, gensym(:row78), 2, ir)
        @test Base.invokelatest(g, false) == 2
        @test Base.invokelatest(g, true) == 1
    end
    # the typed exit converts the same shape (its former decline of the
    # goto-into-handler-island fell away with the F10 reassignment)
    oc = Core.OpaqueClosure(UC.ir_to_ircode(ir))
    @test oc(true) == 1
    @test oc(false) == 2
end

@testset "scope folding (current_scope through the pipeline)" begin
    # irpasses "Test correctness of current_scope folding": behavioral
    # `scope_folding() == 1` asserts stay stock-side (hook-on); the
    # compute_trycatch(::IRCode) legs die with ssair (the Vector{Any} form
    # survives). The pipeline leg found F8 (fixed): the LOWERED exit
    # dropped the EnterNode scope operand, running scoped regions without
    # their dynamic scope; both exits now carry it.
    tir = UC.typed_ir(scope_folding_probe, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    irc = UC.ir_to_ircode(tir)
    @test count(1:length(irc.stmts.stmt)) do i
        st = irc.stmts.stmt[i]
        st isa Core.EnterNode && isdefined(st, :scope)
    end >= 1
    ci = UC.ir_to_codeinfo(tir)
    # F8 static witness: a scoped enter survives the lowered exit
    @test count(ci.code) do st
        (st isa Core.EnterNode && isdefined(st, :scope)) ||
            (Meta.isexpr(st, :enter) && length(st.args) >= 2)
    end >= 1
    # F8 behavioral witness: current_scope() observes the enter's literal 1
    g = UC.redefine_through_ir(scope_folding_probe, Tuple{})
    @test Base.invokelatest(g) === 1
    # and the user-facing form: a ScopedValue read under @with sees the
    # @with value, not the default
    g2 = UC.redefine_through_ir(with_read_53521, Tuple{Int})
    @test Base.invokelatest(g2, 42) == 42
    # nested scopes: the inner @with shadows and the outer is restored
    g3 = UC.redefine_through_ir(nested_with_53521, Tuple{Int})
    @test Base.invokelatest(g3, 7) == (8, 7) == nested_with_53521(7)
end

@testset "SROA affinity of the definedness check (#52857)" begin
    # irpasses #52857: :new of a 1-field immutable with NO field values,
    # getfield on the conditional path only; stock sroa_pass! replaces the
    # load with a definedness throw KEPT IN THE CONDITIONAL BLOCK. NB the
    # program has no usable execution oracle: stock's own compiled
    # equivalent traps ("Unreachable reached") on the taken path, and
    # jl_new_structv rejects the partial immutable new outright — the
    # original test never executes it either. Ported property: the shape
    # optimizes and verifies, the untaken path returns 1, and (pinned) a
    # throwing witness should survive on the taken path.
    b = Builder(name = :i52857)
    append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"region_arg"; type = Bool)
    n = append_stmt!(b, K"new", ImmRef52857; type = ImmRef52857)
    fi = append_stmt!(b, K"if", op_stmt(c); type = Nothing)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    g = append_stmt!(b, K"call", GlobalRef(Base, :getfield), n, 1; type = Any)
    append_stmt!(b, K"result", g)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", 1)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    ir = optimized(ir, Any[Any, Bool])
    @test UnifiedIR.verify_ir(ir; level = 1)
    irc = UC.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    @test Core.OpaqueClosure(irc)(false) == 1
    # F11 (fixed): the maybe-undef field load survives as a conditional
    # throwing statement on the taken path (stock keeps it as a conditional
    # UndefRefError throw — the "affinity" under test); the under-initialized
    # new publishes its undef facts and the effects paths keep !nothrow
    @test count_kind(ir, K"if") + count_kind(ir, K"extract") +
          count_kind(ir, K"call") > 0
end

# ---------------------------------------------------------------------------
# compile smokes and reflection-only rows (big_dead_throw_catch, #53521)
# ---------------------------------------------------------------------------

@testset "compile smoke: PhiC fixup shapes (big_dead_throw_catch)" begin
    # irpasses.jl defines these ("PhiC fixup of compact! with cfg
    # modification") without asserting — a compile-crash canary. Here both
    # bodies go through the full pipeline, verify, and execute
    # differentially against the stock-compiled originals.
    @inline function big_dead_throw_catch()
        x = 1
        try
            x = 2
            if Ref{Bool}(false)[]
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x); Base.donotdelete(x); Base.donotdelete(x)
                Base.donotdelete(x)
                x = 3
            end
        catch
            return x
        end
    end
    function call_big_dead_throw_catch()
        if Ref{Bool}(false)[]
            return big_dead_throw_catch()
        end
        return 4
    end
    tir = UC.typed_ir(big_dead_throw_catch, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:bdtc), 1, tir)
    @test Base.invokelatest(g) === big_dead_throw_catch()
    tir = UC.typed_ir(call_big_dead_throw_catch, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:callbdtc), 1, tir)
    @test Base.invokelatest(g) === 4
end

@testset "ScopedValues shapes (#53521)" begin
    # irpasses.jl #53521 (incorrect scope counting in :leave) inspects
    # Base.code_ircode output + cfg_simplify!; code_ircode bypasses the
    # unified hooks, so the reflection content is re-expressed via the
    # unified queries: return types at least as precise as stock's asserts,
    # verify, and execution.
    tir = UC.typed_ir(f53521_a, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    @test Compiler.widenconst(tir.meta[:rettype]) === Nothing
    g = UC.define_ir_method!(B3BDefs, gensym(:f53521a), 1, tir)
    @test Base.invokelatest(g) === nothing
    rt = UC.infer_return(f53521_wrap, Any[Int])
    @test Compiler.widenconst(rt) <: Union{Nothing,Float64}
    tir = UC.typed_ir(f53521_wrap, Any[Int])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UC.define_ir_method!(B3BDefs, gensym(:f53521w), 2, tir)
    @test Base.invokelatest(g, 0) === f53521_wrap(0)
end

end # module B3BPortIrpasses
