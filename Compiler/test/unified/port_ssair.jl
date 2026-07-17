# Re-expression of Compiler/test/ssair.jl over UnifiedIR (COMPILER-PORT-PLAN
# B3a). Each testset names the ssair.jl test whose semantic content it
# carries; machinery-specific originals (IncrementalCompact/DFS/domtree
# internals) keep running against stock until Stage D — the mapping ledger is
# /workspace/B3A-PORT-MAP.md. Fixtures are Builder-made region IR or entry-
# converted real bodies; dominance is checked against an independent
# by-definition oracle, semantics against the reference interpreter and
# runtime execution of the exit converters' output.

module B3APortSsair

using Test
using UnifiedIR
using UnifiedIR: op_stmt, op_block, op_region, op_inline, StmtId, RegionId
import ..UnifiedCompiler
import ..CC as Compiler

module B3AConstGlobs
    const global_error_switch_const1::Bool = false
    const global_error_switch_const2::Bool = true
    global global_error_switch::Bool = true
end
module B3ADefs end

# ---------------------------------------------------------------------------
# island dominators (ssair.jl #31121 + the dynamic-domtree battery)
# ---------------------------------------------------------------------------

"Build a cfg island from an adjacency list (entry = block 1). `declorder`
permutes block declaration order (entry stays first) — the analogue of
ssair.jl's pred/succ-order reversals, since block iteration order is what a
numbering-sensitive dominator algorithm would key on. Returns (ir, cfgop,
pos) with `pos[block] = declaration position` (region id = 1 + position)."
function island_ir(succs::Vector{Vector{Int}}; declorder = collect(1:length(succs)))
    b = Builder(name = :domtest)
    cnd = append_stmt!(b, K"region_arg"; type = Bool)
    cfg = append_stmt!(b, K"cfg"; type = Any)
    pos = Dict(bi => j for (j, bi) in enumerate(declorder))
    rid(bi) = RegionId(1 + pos[bi])
    for bi in declorder
        ss = succs[bi]
        UnifiedIR.open_region!(b, cfg; kind = UnifiedIR.REGION_BLOCK)
        if length(ss) == 0
            append_stmt!(b, K"result", 0)
        elseif length(ss) == 1
            append_stmt!(b, K"goto", op_block(rid(ss[1])), op_inline(0))
        elseif length(ss) == 2
            append_stmt!(b, K"br_if", op_stmt(cnd),
                         op_block(rid(ss[1])), op_inline(0),
                         op_block(rid(ss[2])), op_inline(0))
        else
            error("island_ir: 3+ successors not needed here")
        end
        UnifiedIR.close_region!(b)
    end
    append_stmt!(b, K"return", cfg)
    return finish!(b), cfg, pos
end

"Dominator sets straight from the definition (independent oracle): d
dominates v iff v is unreachable from the entry once d is removed.
Unreachable blocks get no entry, matching `island_dominators`."
function naive_doms(succs::Vector{Vector{Int}})
    n = length(succs)
    function reach(skip::Int)
        seen = falses(n)
        skip == 1 && return seen
        seen[1] = true
        stack = [1]
        while !isempty(stack)
            u = pop!(stack)
            for v in succs[u]
                (v == skip || seen[v]) && continue
                seen[v] = true
                push!(stack, v)
            end
        end
        return seen
    end
    base = reach(0)
    doms = Dict{Int,Set{Int}}()
    for v in 1:n
        (v == 1 || base[v]) || continue
        ds = Set{Int}([v])
        for d in 1:n
            d == v && continue
            (d == 1 || base[d]) || continue
            reach(d)[v] || push!(ds, d)
        end
        doms[v] = ds
    end
    return doms
end

"island_dominators result re-keyed to original block numbers."
function block_doms(succs; declorder = collect(1:length(succs)))
    ir, cfgop, pos = island_ir(succs; declorder)
    UnifiedIR.verify_ir(ir; level = 1)
    inv = Dict(v => k for (k, v) in pos)
    dom = UnifiedIR.island_dominators(ir, cfgop)
    return Dict(inv[Int(r.id) - 1] => Set(inv[Int(d.id) - 1] for d in ds)
                for (r, ds) in dom)
end

@testset "island dominators: issue #31121 shape" begin
    # ssair.jl's DFS-numbering regression CFG (A→{B,C}, B→{D,E}, C→D, D→E):
    # the bug gave E the dominator B instead of A. The invariant the bug
    # violated: dom(E) = {A, E} — no other block dominates E.
    succs = [[2, 3], [4, 5], [4], [5], Int[]]
    want = Dict(1 => Set([1]), 2 => Set([1, 2]), 3 => Set([1, 3]),
                4 => Set([1, 4]), 5 => Set([1, 5]))
    @test naive_doms(succs) == want
    @test block_doms(succs) == want
    # ssair.jl reversed pred/succ orders 16 ways to shake numbering
    # sensitivity out of the algorithm; here the corresponding enumeration
    # orders are br_if bundle order (successors) and block declaration order
    # (which drives block iteration and hence pred discovery order).
    swaps = [(false, false), (true, false), (false, true), (true, true)]
    perms = [[1, 2, 3, 4, 5], [1, 3, 2, 4, 5], [1, 2, 3, 5, 4], [1, 5, 4, 3, 2],
             [1, 4, 5, 2, 3], [1, 3, 4, 5, 2], [1, 5, 3, 4, 2], [1, 2, 4, 3, 5]]
    ok = true
    for (s1, s2) in swaps, p in perms
        sv = [copy(v) for v in succs]
        s1 && reverse!(sv[1])
        s2 && reverse!(sv[2])
        ok &= block_doms(sv; declorder = p) == want
    end
    @test ok
end

@testset "island dominators: 6-block edge-set battery" begin
    # ssair.jl's dynamic domtree test mutates edges of one 6-block CFG and
    # checks idoms against naive_idoms after every step. UnifiedIR has no
    # incremental domtree (island dominators are recomputed per island), so
    # the ported property is: for every edge-set the original stepped
    # through, the dominator sets are correct. Expected sets derived from
    # the idoms ssair.jl asserts ([0,1,1,3,1,4] etc.).
    base = [[3, 2], [5], [4], [6], Int[], [5, 3]]
    variants = [
        # (mutation of `base`, expected dominator sets)
        (succs -> succs,                                     # as built
         Dict(1 => Set([1]), 2 => Set([1, 2]), 3 => Set([1, 3]),
              4 => Set([1, 3, 4]), 5 => Set([1, 5]), 6 => Set([1, 3, 4, 6]))),
        (succs -> (succs[2] = Int[]; succs),                 # delete 2→5
         Dict(1 => Set([1]), 2 => Set([1, 2]), 3 => Set([1, 3]),
              4 => Set([1, 3, 4]), 5 => Set([1, 3, 4, 6, 5]), 6 => Set([1, 3, 4, 6]))),
        (succs -> (succs[6] = [3]; succs),                   # delete 6→5
         Dict(1 => Set([1]), 2 => Set([1, 2]), 3 => Set([1, 3]),
              4 => Set([1, 3, 4]), 5 => Set([1, 2, 5]), 6 => Set([1, 3, 4, 6]))),
        (succs -> (succs[6] = [5]; succs),                   # delete 6→3
         Dict(1 => Set([1]), 2 => Set([1, 2]), 3 => Set([1, 3]),
              4 => Set([1, 3, 4]), 5 => Set([1, 5]), 6 => Set([1, 3, 4, 6]))),
        (succs -> (succs[1] = [3]; succs),                   # delete 1→2
         Dict(1 => Set([1]), 3 => Set([1, 3]), 4 => Set([1, 3, 4]),
              5 => Set([1, 3, 4, 6, 5]), 6 => Set([1, 3, 4, 6]))),
        (succs -> (succs[1] = Int[]; succs),                 # delete 1→{2,3}
         Dict(1 => Set([1]))),
        (succs -> (succs[1] = [2]; succs),                   # re-add only 1→2
         Dict(1 => Set([1]), 2 => Set([1, 2]), 5 => Set([1, 2, 5]))),
    ]
    for (mut, want) in variants
        succs = mut([copy(v) for v in base])
        @test naive_doms(succs) == want
        @test block_doms(succs) == want
    end
end

@testset "region tree as domtree: structured diamond" begin
    # ssair.jl's construct_domtree/dominates + postdominates battery on the
    # 4-block println diamond. For structured code dominance is read off the
    # region tree (design §5.1): `visible` is the dominance oracle, arm
    # membership the "belongs to block i" oracle, and result-terminated arms
    # + root membership after the op the postdominance witness. Execution
    # traces prove the reads: the join statement runs on both paths.
    diam = (c, x) -> begin
        println(x, 1)
        if c
            println(x, 2)
        else
            println(x, 3)
        end
        println(x, 4)
    end
    ir = UnifiedCompiler.lowered_ir(diam, Tuple{Bool,IOBuffer})
    UnifiedIR.editable(ir)
    UnifiedCompiler.structurize!(ir)
    UnifiedCompiler.promote_loop_cells!(ir)
    UnifiedCompiler.selectify!(ir)
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    # shape: the island fully structurized into one two-armed if
    @test !any(s -> UnifiedIR.stmt_kind(ir, s) === K"cfg", UnifiedIR.each_stmt(ir))
    ifops = [s for s in UnifiedIR.each_stmt(ir) if UnifiedIR.stmt_kind(ir, s) === K"if"]
    @test length(ifops) == 1
    fiop = only(ifops)
    arms = [r for r in UnifiedIR.owned_regions(ir, fiop) if !UnifiedIR.getregion(ir, r).dead]
    @test length(arms) == 2
    # locate the four println calls by their literal argument
    callof = Dict{Int,StmtId}()
    for s in UnifiedIR.each_stmt(ir)
        UnifiedIR.stmt_kind(ir, s) === K"call" || continue
        for j in 1:UnifiedIR.nops(ir, s)
            o = UnifiedIR.getop(ir, s, j)
            if UnifiedIR.optag(o) == UnifiedIR.TAG_INLINE &&
               UnifiedIR.imm_value(o) isa Int64 && 1 <= UnifiedIR.imm_value(o) <= 4
                callof[Int(UnifiedIR.imm_value(o))] = s
            end
        end
    end
    @test sort(collect(keys(callof))) == [1, 2, 3, 4]
    c1, c2, c3, c4 = callof[1], callof[2], callof[3], callof[4]
    # block membership: 1 and 4 in the root around the if; 2 and 3 in the arms
    @test UnifiedIR.stmt_region(ir, c1) == UnifiedIR.stmt_region(ir, fiop)
    @test UnifiedIR.stmt_region(ir, c4) == UnifiedIR.stmt_region(ir, fiop)
    @test UnifiedIR.stmt_region(ir, c2) in arms
    @test UnifiedIR.stmt_region(ir, c3) in arms
    @test UnifiedIR.stmt_region(ir, c2) != UnifiedIR.stmt_region(ir, c3)
    # dominance battery (ssair.jl: dominates(1, i) for i = 2:4; nothing else)
    @test UnifiedIR.visible(ir, c1, c2)
    @test UnifiedIR.visible(ir, c1, c3)
    @test UnifiedIR.visible(ir, c1, c4)
    for i in (c2, c3), j in (c1, c2, c3, c4)
        i === j && continue
        @test !UnifiedIR.visible(ir, i, j)
    end
    for j in (c1, c2, c3)
        @test !UnifiedIR.visible(ir, c4, j)
    end
    # postdominance battery (postdominates(4, i) for i = 1:3; nothing else):
    # both arms feed the join (`result` terminators — no early exit), and c4
    # sits after the op in the same region, so every path through the if
    # reaches it; c1..c3 precede other statements, so they postdominate none.
    for r in arms
        t = UnifiedIR.region_terminator(ir, r)
        @test t !== nothing && UnifiedIR.stmt_kind(ir, t) === K"result"
    end
    @test UnifiedIR.comes_before(ir, fiop, c4)
    # executed evidence: the join call runs on both paths, arms exclusively
    buf = IOBuffer()
    UnifiedIR.interpret(ir, diam, true, buf)
    @test String(take!(buf)) == "1\n2\n4\n"
    UnifiedIR.interpret(ir, diam, false, buf)
    @test String(take!(buf)) == "1\n3\n4\n"
end

# ---------------------------------------------------------------------------
# compaction/renaming semantics (ssair.jl PR #32145, #29107, dead-block
# tests; the machinery-specific IncrementalCompact originals die with their
# subject at Stage D — this is the representation-level semantic content)
# ---------------------------------------------------------------------------

@testset "constant-branch fold + compact!: dead regions dropped, refs intact" begin
    # ssair.jl folds hand-built CFGs with dead blocks through
    # IncrementalCompact and checks verify_ir + that live code never
    # references removed blocks. Region form: fold_constant_branches! kills
    # the untaken arm, compact! drops dead regions and renumbers, RemapSet
    # proves the dead statements are gone and every surviving reference is
    # rewritten (a live reference to a dropped statement is a hard error).
    b = Builder(name = :fold)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    fi = append_stmt!(b, K"if", true; type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    v = append_stmt!(b, K"call", GlobalRef(Base, :+), x, 1; type = Any)
    append_stmt!(b, K"result", v)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    w = append_stmt!(b, K"call", GlobalRef(Base, :-), x, 1; type = Any)
    append_stmt!(b, K"result", w)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", fi)
    ir = finish!(b)
    @test UnifiedIR.interpret(ir, nothing, 10) == 11
    nregions_before = count(r -> !r.dead, ir.regions)
    UnifiedIR.editable(ir)
    ir, nfold = UnifiedIR.fold_constant_branches!(ir)
    @test nfold == 1
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.remap(remap, w).id == 0          # dead-arm stmt dropped
    @test UnifiedIR.remap(remap, v).id != 0          # live arm renumbered
    @test all(r -> !r.dead, ir.regions)              # dead regions gone
    @test count(r -> true, ir.regions) < nregions_before
    @test count(s -> UnifiedIR.stmt_kind(ir, s) === K"if", UnifiedIR.each_stmt(ir)) == 0
    @test UnifiedIR.interpret(ir, nothing, 10) == 11 # semantics preserved
end

@testset "dead backedge: literal-false continue (self-edge analogue)" begin
    # ssair.jl checks compaction survives removing a dead self-edge of a phi
    # (GotoIfNot(true, 2)). Loops have no phis here; the dead backedge is a
    # `continue` whose condition is literal false — a single-trip loop that
    # must verify, execute, and exit through both converters.
    b = Builder(name = :deadback)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    lp = append_stmt!(b, K"loop", op_stmt(x); type = Any)
    body = UnifiedIR.open_region!(b, lp; kind = UnifiedIR.REGION_LOOP_BODY)
    a = append_stmt!(b, K"region_arg"; type = Any)
    v = append_stmt!(b, K"call", GlobalRef(Base, :+), a, 1; type = Any)
    append_stmt!(b, K"continue", op_region(body), false, op_stmt(v))
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", lp)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, 5) == 6
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    @test Core.OpaqueClosure(irc)(5) == 6
    g = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:deadback), 2, ir)
    @test Base.invokelatest(g, 5) == 6
end

# ---------------------------------------------------------------------------
# editable-layout insertion semantics (ssair.jl #46967, pending-node
# insert_node!, #50379 end-of-block insert, flag-dependent DCE of inserts)
# ---------------------------------------------------------------------------

@testset "issue #46967 analogue: replace a stmt via insertion, compact once" begin
    # generate some IR, swap the add's constant via an inserted replacement,
    # compact, and check nothing pending leaks and the replaced stmt count is
    # exact (the original guarded undef stmts introduced by compaction);
    # print_ir is the `show(devnull, ir)` smoke.
    b = Builder(name = :i46967)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    v = append_stmt!(b, K"call", GlobalRef(Base, :+), x, 42; type = Any)
    w = append_stmt!(b, K"call", GlobalRef(Base, :*), v, 2; type = Any)
    append_stmt!(b, K"return", w)
    ir = finish!(b)
    n0 = UnifiedIR.nstmts(ir)
    @test UnifiedIR.interpret(ir, nothing, 1) == 86
    UnifiedIR.editable(ir)
    n1 = UnifiedIR.insert_before!(ir, v, K"call",
        UnifiedIR.getop(ir, v, 1), UnifiedIR.getop(ir, v, 2), 999; type = Any)
    UnifiedIR.replace_uses!(ir, v => n1)
    UnifiedIR.flush_renames!(ir)
    # the old stmt is now unused; make it collectable and collect it
    UnifiedIR.set_flag!(ir, v, UnifiedIR.FLAG_REMOVABLE)
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.dce!(ir) == 1
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.nstmts(ir) == n0                 # exact replacement
    @test UnifiedIR.interpret(ir, nothing, 1) == 2000
    @test !isempty(UnifiedIR.print_ir(ir))
end

@testset "insert_node! for pending node analogue: chained insert_after!" begin
    # ssair.jl inserts a node attached after an invoke, then another attached
    # after the *pending* node, and checks order and operands after
    # compaction. Editable form: chained insert_after! with the second
    # insertion anchored on the first (not yet compacted) statement.
    b = Builder(name = :chain)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    v = append_stmt!(b, K"call", GlobalRef(Base, :+), x, 1; type = Any)
    r = append_stmt!(b, K"return", v)
    ir = finish!(b)
    nstmts0 = UnifiedIR.nstmts(ir)
    UnifiedIR.editable(ir)
    n1 = UnifiedIR.insert_after!(ir, v, K"call", GlobalRef(Base, :*), op_stmt(v), 10; type = Any)
    n2 = UnifiedIR.insert_after!(ir, n1, K"call", GlobalRef(Base, :+), op_stmt(n1), 7; type = Any)
    UnifiedIR.setop!(ir, r, 1, op_stmt(n2))
    @test UnifiedIR.comes_before(ir, v, n1)
    @test UnifiedIR.comes_before(ir, n1, n2)
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.nstmts(ir) == nstmts0 + 2
    n1n = UnifiedIR.remap(remap, n1)
    n2n = UnifiedIR.remap(remap, n2)
    @test UnifiedIR.comes_before(ir, n1n, n2n)
    # the chained node still references its (renumbered) anchor
    @test UnifiedIR.asstmt(UnifiedIR.getop(ir, n2n, 2)) == n1n
    @test UnifiedIR.interpret(ir, nothing, 5) == 67
end

@testset "issue #50379 analogue: insert at the end of a region" begin
    # insertion at the end of a basic block during compaction; region form:
    # insert_before! the arm's terminator, i.e. at the region's end.
    b = Builder(name = :i50379)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Bool)
    acc = append_stmt!(b, K"cell", Any; type = Any)
    append_stmt!(b, K"cell_set", acc, 0)
    fi = append_stmt!(b, K"if", op_stmt(x); type = Nothing)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"cell_set", acc, 1)
    term = append_stmt!(b, K"result")
    UnifiedIR.close_region!(b)
    g2 = append_stmt!(b, K"cell_get", op_stmt(acc); type = Any)
    append_stmt!(b, K"return", g2)
    ir = finish!(b)
    @test UnifiedIR.interpret(ir, nothing, true) == 1
    UnifiedIR.editable(ir)
    UnifiedIR.insert_before!(ir, term, K"cell_set", op_stmt(acc), 42)
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, nothing, true) == 42
    @test UnifiedIR.interpret(ir, nothing, false) == 0
end

@testset "inserted-node flags decide DCE (effect-ful kept, effect-free dead)" begin
    # ssair.jl: an effectful inserted call survives compaction, an unused
    # effect-free one is deleted. Here that is exactly the REMOVABLE mask
    # contract of dce! on inserted statements.
    function insert_unused(flagged::Bool)
        b = Builder(name = :flags)
        append_stmt!(b, K"region_arg"; type = Any)
        x = append_stmt!(b, K"region_arg"; type = Any)
        v = append_stmt!(b, K"call", GlobalRef(Base, :+), x, 1; type = Any)
        append_stmt!(b, K"return", v)
        ir = finish!(b)
        UnifiedIR.editable(ir)
        n = UnifiedIR.insert_after!(ir, v, K"call", GlobalRef(Base, :*), op_stmt(v), 2;
                                    type = Any)
        flagged && UnifiedIR.set_flag!(ir, n, UnifiedIR.FLAG_REMOVABLE)
        ir, _ = UnifiedIR.compact!(ir)
        removed = UnifiedIR.dce!(ir)
        ir, _ = UnifiedIR.compact!(ir)
        UnifiedIR.verify_ir(ir; level = 1)
        return removed, UnifiedIR.nstmts(ir), UnifiedIR.interpret(ir, nothing, 5)
    end
    removed, n, val = insert_unused(false)   # effectful default call flags
    @test removed == 0 && n == 5 && val == 6
    removed, n, val = insert_unused(true)    # marked effect-free/removable
    @test removed == 1 && n == 4 && val == 6
end

# ---------------------------------------------------------------------------
# operand protocol (ssair.jl userefs battery) and refine (PiNode) folding
# ---------------------------------------------------------------------------

@testset "operand read/write roundtrip across the kind zoo (userefs)" begin
    # ssair.jl walks UseRefs over every stmt flavor, setindex!s a dummy and
    # restores. The uniform operand protocol replaces per-node-type UseRef
    # traversal: getop/setop! roundtrip on every operand of every statement,
    # with the inline-encoding discipline (STMT slot 1, IMM slot 2) upheld.
    b = Builder(name = :zoo)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"cell", Int64; type = Any)
    append_stmt!(b, K"cell_set", c, x)
    gv = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)
    isd = append_stmt!(b, K"cell_isdefined", op_stmt(c); type = Any)
    cnd = append_stmt!(b, K"call", GlobalRef(Base, :>), gv, 0; type = Any)
    fi = append_stmt!(b, K"if", cnd; type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", gv)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 0)
    UnifiedIR.close_region!(b)
    lp = append_stmt!(b, K"loop", fi; type = Any)
    body = UnifiedIR.open_region!(b, lp; kind = UnifiedIR.REGION_LOOP_BODY)
    a1 = append_stmt!(b, K"region_arg"; type = Any)
    dec = append_stmt!(b, K"call", GlobalRef(Base, :-), a1, 1; type = Any)
    cn2 = append_stmt!(b, K"call", GlobalRef(Base, :>), dec, 0; type = Any)
    append_stmt!(b, K"continue", op_region(body), op_stmt(cn2), op_stmt(dec))
    UnifiedIR.close_region!(b)
    ex = append_stmt!(b, K"extract", op_stmt(lp), op_inline(1); type = Any)
    rf = append_stmt!(b, K"refine", ex; type = Any)
    tup = append_stmt!(b, K"call", GlobalRef(Core, :tuple), rf, isd; type = Any)
    append_stmt!(b, K"return", tup)
    ir = finish!(b)
    @test UnifiedIR.verify_ir(ir; level = 1)
    kinds_seen = Set{Symbol}()
    ok = true
    for s in UnifiedIR.each_stmt(ir)
        push!(kinds_seen, UnifiedIR.kindname(UnifiedIR.stmt_kind(ir, s)))
        for j in 1:UnifiedIR.nops(ir, s)
            v1 = UnifiedIR.getop(ir, s, j)
            dummy = UnifiedIR.optag(v1) == UnifiedIR.TAG_STMT ?
                op_stmt(StmtId(Int32(1))) : op_inline(7)
            UnifiedIR.setop!(ir, s, j, dummy)
            ok &= UnifiedIR.getop(ir, s, j).bits == dummy.bits
            UnifiedIR.setop!(ir, s, j, v1)
            ok &= UnifiedIR.getop(ir, s, j).bits == v1.bits
        end
    end
    @test ok
    for k in (:call, :cell, :cell_get, :cell_isdefined, :cell_set, :continue,
              :extract, :if, :loop, :refine, :region_arg, :result, :return)
        @test k in kinds_seen
    end
    @test UnifiedIR.verify_ir(ir; level = 1)     # roundtrip left the body intact
    @test UnifiedIR.interpret(ir, nothing, 5) == (0, true)
end

@testset "refine of a constant folds away (constant PiNode compaction)" begin
    b = Builder(name = :rfc)
    append_stmt!(b, K"region_arg"; type = Any)
    rf = append_stmt!(b, K"refine", 0.0; type = Compiler.Const(0.0))
    append_stmt!(b, K"return", rf)
    ir = finish!(b)
    UnifiedCompiler.forward_refines!(ir)
    UnifiedIR.dce!(ir)
    ir, _ = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    # fully eliminated: only the argument and the constant return remain
    @test UnifiedIR.nstmts(ir) == 2
    @test UnifiedIR.interpret(ir, nothing) === 0.0
end

# ---------------------------------------------------------------------------
# verifier canonical-form battery (ssair.jl GlobalRef/static_parameter
# non-canonical + #29107 use-after-def-in-dead-code). Globals are first-class
# operands here so the stock non-canonicality has no analogue; the class —
# the verifier rejects representationally-invalid references — is covered by
# the corresponding UnifiedIR rules. Use-before-def is impossible by
# construction (appends cannot reference the future; L0/L1 catch the escape
# hatches), which retires the #29107 bug class structurally.
# ---------------------------------------------------------------------------

@testset "verifier rejects invalid references" begin
    # reference to a zero-result statement
    let b = Builder(name = :vz)
        append_stmt!(b, K"region_arg"; type = Any)
        c = append_stmt!(b, K"cell", Int64; type = Any)
        stz = append_stmt!(b, K"cell_set", c, 1)
        append_stmt!(b, K"return", op_stmt(stz))
        @test_throws UnifiedIR.VerifyError finish!(b)
    end
    # out-of-range (forward) reference
    let b = Builder(name = :vf)
        append_stmt!(b, K"region_arg"; type = Any)
        append_stmt!(b, K"return", op_stmt(StmtId(Int32(99))))
        @test_throws UnifiedIR.VerifyError finish!(b)
    end
    # use of a tombstoned statement
    let b = Builder(name = :vt)
        append_stmt!(b, K"region_arg"; type = Any)
        v = append_stmt!(b, K"call", GlobalRef(Base, :+), 1, 2; type = Any)
        append_stmt!(b, K"return", v)
        ir = finish!(b)
        UnifiedIR.delete_stmt!(ir, v)
        @test_throws UnifiedIR.VerifyError UnifiedIR.verify_ir(ir; level = 1)
    end
    # visibility violation: arm-local value used after the if (#29107's
    # "value used where its definition need not have executed", now a
    # verifier-rejected shape instead of a latent compaction hazard)
    let b = Builder(name = :vv)
        append_stmt!(b, K"region_arg"; type = Any)
        c = append_stmt!(b, K"region_arg"; type = Bool)
        fi = append_stmt!(b, K"if", op_stmt(c); type = Any)
        UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
        v = append_stmt!(b, K"call", GlobalRef(Base, :+), 1, 2; type = Any)
        append_stmt!(b, K"result", v)
        UnifiedIR.close_region!(b)
        append_stmt!(b, K"return", op_stmt(v))
        ir = finish!(b)   # L0 passes (structure fine)
        @test_throws UnifiedIR.VerifyError UnifiedIR.verify_ir(ir; level = 1)
    end
end

# ---------------------------------------------------------------------------
# branch folding and dead join edges (ssair.jl "GotoIfNot folding" +
# the unreachable-frontend-PhiNode-edge generated-function trio)
# ---------------------------------------------------------------------------

@testset "GotoIfNot folding: no const-condition branches after optimize" begin
    function f_with_maybe_nonbool_cond(a::Int, r::Bool)
        a = r ? true : a
        if a
            x = a ? 1 : 2.
        else
            x = a ? 1 : 2.
        end
        return x
    end
    tir = UnifiedCompiler.typed_ir(f_with_maybe_nonbool_cond, Any[Int, Bool])
    @test UnifiedIR.verify_ir(tir; level = 1)
    # the original: after conversion no GotoIfNot may target statically
    # unreachable code — i.e. statically-decided branches must have been
    # folded. Region form: no `if` with a constant-Bool condition survives.
    nconst = 0
    for s in UnifiedIR.each_stmt(tir)
        UnifiedIR.stmt_kind(tir, s) === K"if" || continue
        o = UnifiedIR.getop(tir, s, 1)
        cv = UnifiedIR.optag(o) == UnifiedIR.TAG_INLINE ? UnifiedIR.imm_value(o) :
             UnifiedIR.optag(o) == UnifiedIR.TAG_CONST ? tir.body.constants[UnifiedIR.payload(o)] :
             nothing
        cv isa Bool && (nconst += 1)
        if UnifiedIR.optag(o) == UnifiedIR.TAG_STMT
            ct = UnifiedIR.stmt_type(tir, UnifiedIR.asstmt(o))
            ct isa Compiler.Const && ct.val isa Bool && (nconst += 1)
        end
    end
    @test nconst == 0
    # and the compiled body implements the source semantics, including the
    # non-Bool TypeError path (a::Int used as a condition when r is false)
    g = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:fmnb), 3, tir)
    @test Base.invokelatest(g, 1, true) === 1
    @test Base.invokelatest(g, 0, true) === 1
    @test_throws TypeError Base.invokelatest(g, 1, false)
end

@testset "join edges from folded and must-throw arms" begin
    # ssair.jl feeds generated-function CodeInfo whose PhiNode edge becomes
    # unreachable (const-global branch) or must-throw (error() edge). The
    # raw phi-bearing CodeInfo cannot enter the converter (documented gap
    # below); the semantic content — joins with statically-dead incoming
    # edges must fold away cleanly and execution must match — is expressed
    # over region IR.
    function build_join_on_global(gname)
        b = Builder(name = :uedge)
        append_stmt!(b, K"region_arg"; type = Any)
        x = append_stmt!(b, K"region_arg"; type = Any)
        y = append_stmt!(b, K"region_arg"; type = Any)
        t = append_stmt!(b, K"globalref", GlobalRef(B3AConstGlobs, gname); type = Bool)
        fi = append_stmt!(b, K"if", op_stmt(t); type = Any)
        UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
        v = append_stmt!(b, K"call", GlobalRef(Base, :identity), y; type = Any)
        append_stmt!(b, K"result", v)
        UnifiedIR.close_region!(b)
        UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
        append_stmt!(b, K"result", x)
        UnifiedIR.close_region!(b)
        append_stmt!(b, K"return", fi)
        return finish!(b)
    end
    for (gname, expect) in ((:global_error_switch_const1, 1),
                            (:global_error_switch_const2, 2))
        ir = build_join_on_global(gname)
        @test UnifiedIR.verify_ir(ir; level = 1)
        st = UnifiedCompiler.UInferState()
        ir = UnifiedCompiler.optimize_ir!(ir, Any[Any, Int, Int]; state = st)
        @test UnifiedIR.verify_ir(ir; level = 1)
        # the const-global branch folded away: no join left at all
        @test count(s -> UnifiedIR.stmt_kind(ir, s) === K"if",
                    UnifiedIR.each_stmt(ir)) == 0
        g = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:uedge), 3, ir)
        @test Base.invokelatest(g, 1, 2) == expect
    end
    # must-throw edge: one join input comes from an arm that always throws
    b = Builder(name = :mte)
    append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"globalref", GlobalRef(B3AConstGlobs, :global_error_switch); type = Bool)
    fi = append_stmt!(b, K"if", op_stmt(t); type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"call", GlobalRef(Base, :error), "This error is expected"; type = Any)
    append_stmt!(b, K"result", 2)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 1)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", fi)
    ir3 = finish!(b)
    st = UnifiedCompiler.UInferState()
    ir3 = UnifiedCompiler.optimize_ir!(ir3, Any[Any]; state = st)
    @test UnifiedIR.verify_ir(ir3; level = 1)
    irc3 = UnifiedCompiler.ir_to_ircode(ir3)
    @test Compiler.verify_ir(irc3) === nothing
    # stock prunes the dead (post-Union{}) join edge so no PhiNode remains;
    # the typed exit's unreachable-after rule (F5 fixed) prunes it the same
    # way — the must-throw arm contributes no edge and the single-edge join
    # collapses to a plain value
    @test count(s -> s isa Core.PhiNode, irc3.stmts.stmt) == 0
    g3 = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:mte), 1, ir3)
    @test_throws ErrorException Base.invokelatest(g3)
    Core.eval(B3AConstGlobs, :(global_error_switch = false))
    @test Base.invokelatest(g3) == 1
    Core.eval(B3AConstGlobs, :(global_error_switch = true))
    # the raw-CodeInfo entry path for phi-bearing sources (F6 fixed): a
    # block's leading φs become its region args, the values travel on the
    # in-edges; an edgeless φ is undef and legal while unused. Generated
    # functions returning pre-SSA'd CodeInfo can now enter the pipeline.
    fields = UnifiedCompiler.default_codeinfo_fields(2, 1, Symbol[Symbol("#self#")],
                                                     fill(0x08, 1))
    fields[:code] = Any[Core.PhiNode(Int32[], Any[]), Core.ReturnNode(1)]
    ci = UnifiedCompiler.make_codeinfo(; fields...)
    ir_phi = UnifiedCompiler.codeinfo_to_ir(ci; nargs = 1, name = :phi_entry)
    @test UnifiedIR.verify_ir(ir_phi; level = 1)
    @test UnifiedIR.interpret(ir_phi, nothing) === 1
    # a φ join over a runtime-global branch — the original
    # gen_must_throw_phinode_edge/unreachable_phinode_edge input shape
    # (statement-position global read, GotoIfNot on its SSA value, φ join
    # with one edge from the fallthrough arm)
    fields2 = UnifiedCompiler.default_codeinfo_fields(7, 2,
        Symbol[Symbol("#self#"), :x], fill(0x08, 2))
    fields2[:code] = Any[
        GlobalRef(B3AConstGlobs, :global_error_switch),
        Core.GotoIfNot(Core.SSAValue(1), 5),
        Expr(:call, GlobalRef(Base, :+), 10, 1),
        Core.GotoNode(6),
        Expr(:call, GlobalRef(Base, :*), 4, 5),
        Core.PhiNode(Int32[4, 5], Any[Core.SSAValue(3), Core.SSAValue(5)]),
        Core.ReturnNode(Core.SSAValue(6)),
    ]
    ci2 = UnifiedCompiler.make_codeinfo(; fields2...)
    ir_join = UnifiedCompiler.codeinfo_to_ir(ci2; nargs = 2, name = :phi_join)
    @test UnifiedIR.verify_ir(ir_join; level = 1)
    @test UnifiedIR.interpret(ir_join, nothing, 100) == 11   # switch true: then-arm
    stj = UnifiedCompiler.UInferState()
    ir_join = UnifiedCompiler.optimize_ir!(ir_join, Any[Any, Int]; state = stj)
    @test UnifiedIR.verify_ir(ir_join; level = 1)
    @test Compiler.verify_ir(UnifiedCompiler.ir_to_ircode(ir_join)) === nothing
    gj = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:phijoin), 2, ir_join)
    @test Base.invokelatest(gj, 100) == 11
    Core.eval(B3AConstGlobs, :(global_error_switch = false))
    @test Base.invokelatest(gj, 100) == 20
    Core.eval(B3AConstGlobs, :(global_error_switch = true))
    # backedge φs that permute each other (the #29262 parallel-move class,
    # entering as raw SSA): φa/φb swap on every trip; three trips land back
    # on the initial assignment
    fields3 = UnifiedCompiler.default_codeinfo_fields(10, 1,
        Symbol[Symbol("#self#")], fill(0x08, 1))
    fields3[:code] = Any[
        Core.GotoNode(2),
        Core.PhiNode(Int32[1, 8], Any[0, Core.SSAValue(5)]),
        Core.PhiNode(Int32[1, 8], Any[QuoteNode(:a), Core.SSAValue(4)]),
        Core.PhiNode(Int32[1, 8], Any[QuoteNode(:b), Core.SSAValue(3)]),
        Expr(:call, GlobalRef(Base, :+), Core.SSAValue(2), 1),
        Expr(:call, GlobalRef(Base, :<), Core.SSAValue(5), 3),
        Core.GotoIfNot(Core.SSAValue(6), 9),
        Core.GotoNode(2),
        Expr(:call, GlobalRef(Core, :tuple), Core.SSAValue(3), Core.SSAValue(4)),
        Core.ReturnNode(Core.SSAValue(9)),
    ]
    ci3 = UnifiedCompiler.make_codeinfo(; fields3...)
    ir_swap = UnifiedCompiler.codeinfo_to_ir(ci3; nargs = 1, name = :phi_swap)
    @test UnifiedIR.verify_ir(ir_swap; level = 1)
    @test UnifiedIR.interpret(ir_swap, nothing) === (:a, :b)
    sts = UnifiedCompiler.UInferState()
    ir_swap = UnifiedCompiler.optimize_ir!(ir_swap, Any[Any]; state = sts)
    @test UnifiedIR.verify_ir(ir_swap; level = 1)
    gs = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:phiswap), 1, ir_swap)
    @test Base.invokelatest(gs) === (:a, :b)
    # exceptional pre-SSA'd forms still take the declared eh-path gap
    fields4 = UnifiedCompiler.default_codeinfo_fields(2, 1, Symbol[Symbol("#self#")],
                                                      fill(0x08, 1))
    fields4[:code] = Any[Core.UpsilonNode(1), Core.ReturnNode(1)]
    ci4 = UnifiedCompiler.make_codeinfo(; fields4...)
    @test_throws UnifiedCompiler.UnsupportedIR UnifiedCompiler.codeinfo_to_ir(
        ci4; nargs = 1, name = :ups_entry)
end

# ---------------------------------------------------------------------------
# behavioral regressions re-run through the unified pipeline (ssair.jl
# #32579, #41975, #57153, #60660, #37919, code_ircode/slots/IRShow)
# ---------------------------------------------------------------------------

@testset "issue #32579: optimizer type constraints (behavior)" begin
    function f32579(x::Int, b::Bool)
        if b
            x = nothing
        end
        if isa(x, Int)
            y = x
        else
            y = x
        end
        if isa(y, Nothing)
            z = y
        else
            z = y
        end
        return z === nothing
    end
    tir = UnifiedCompiler.typed_ir(f32579, Any[Int, Bool])
    @test UnifiedIR.verify_ir(tir; level = 1)
    g = UnifiedCompiler.define_ir_method!(B3ADefs, gensym(:f32579), 3, tir)
    @test Base.invokelatest(g, 0, true) === true
    @test Base.invokelatest(g, 0, false) === false
end

@testset "issue #41975: conversion must not drop the non-Bool branch check" begin
    f_if_typecheck() = (if nothing; end; unsafe_load(Ptr{Int}(0)))
    # entry → exit round trip preserves the check: the call throws TypeError
    # (were the check dropped, this would be a null pointer load)
    g = UnifiedCompiler.redefine_through_ir(f_if_typecheck, Tuple{})
    @test_throws TypeError Base.invokelatest(g)
    # F2 fixed: the degenerate-branch collapse emits the branch's mandatory
    # Bool typecheck (`typeassert(cond, Bool)` — nothrow/removable for
    # provably-Bool conditions, throwing otherwise), and inference types a
    # cannot-be-Bool branch condition Bottom (stock's must-throw GotoIfNot
    # rule), so `if nothing` leaves a Union{} witness and the TypeError
    # survives to runtime. (Executing the once-miscompiled body would have
    # dereferenced Ptr(0); the static witness stays the assert.)
    tir = UnifiedCompiler.typed_ir(f_if_typecheck, Any[])
    hascheck = any(UnifiedIR.each_stmt(tir)) do s
        k = UnifiedIR.stmt_kind(tir, s)
        k === K"if" || k === K"unreachable" ||
            UnifiedIR.stmt_type(tir, s) === Union{}
    end
    @test hascheck
end

@testset "issue #57153 shape: loop + try/finally + return crossing finally" begin
    # the original inspects stock's block-0 entry-edge encoding; that
    # encoding dies with the stock CFG. The surviving semantic content is
    # that this shape converts, types, verifies, and exits cleanly.
    function _worker_task57153()
        while true
            r = let
            try
                if @noinline rand(Bool)
                    return nothing
                end
                q, m
            finally
                missing
            end
            end
            r[1]::Bool
        end
    end
    tir = UnifiedCompiler.typed_ir(_worker_task57153, Any[])
    @test UnifiedIR.verify_ir(tir; level = 1)
    irc = UnifiedCompiler.ir_to_ircode(tir)
    # F3 fixed: the body references unbound globals (`q`, `m` — undefined
    # here, as in the original). Lowering emits such reads as statements;
    # the entry converters now PRESERVE statement-position global loads as
    # `globalref` statements (instead of dissolving them into operands), so
    # the typed exit keeps them out of value position and stock's
    # canonicality rule ("Unbound or partitioned GlobalRef not allowed in
    # value position") is satisfied.
    stockok = try
        Compiler.verify_ir(irc, false)
        true
    catch
        false
    end
    @test stockok
end

@testset "issue #60660: nested-iterator comprehension through the converters" begin
    trips_60660() = let Ts = (Float64, Float32)
        [(Ta, Tb, Tc) for Ta in Ts for Tb in Ts for Tc in Ts]
    end
    g = UnifiedCompiler.redefine_through_ir(trips_60660, Tuple{})
    @test Base.invokelatest(g) == [
        (Float64, Float64, Float64),
        (Float64, Float64, Float32),
        (Float64, Float32, Float64),
        (Float64, Float32, Float32),
        (Float32, Float64, Float64),
        (Float32, Float64, Float32),
        (Float32, Float32, Float64),
        (Float32, Float32, Float32),
    ]
end

@testset "issue #37919: @isdefined converts and verifies" begin
    f37919() = @isdefined(_not_def_37919_)
    ir = UnifiedCompiler.lowered_ir(f37919, Tuple{})
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.interpret(ir, f37919) === false
end

@testset "typed-IR reflection (code_ircode battery)" begin
    # Base.code_ircode(...) isa IRCode → typed_ir(...) isa UnifiedIR.IR,
    # including the staged (optimize_until) forms
    tir = UnifiedCompiler.typed_ir(+, Any[Float64, Float64])
    @test tir isa UnifiedIR.IR
    tir = UnifiedCompiler.typed_ir(+, Any[Float64, Float64]; optimize_until = "inference")
    @test tir isa UnifiedIR.IR
    demo(f) = (f(); f(); f())
    @test UnifiedCompiler.typed_ir(demo, Any[typeof(sin)]) isa UnifiedIR.IR
    @test UnifiedCompiler.typed_ir(demo, Any[typeof(sin)];
                                   optimize_until = "inference") isa UnifiedIR.IR
end

@testset "slots after conversion (locals are cells; args are region args)" begin
    function f_with_slots(a, b)
        c = a + b
        d = c > 0
        return (c, d)
    end
    ir = UnifiedCompiler.lowered_ir(f_with_slots, Tuple{Int,Int})
    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    # #self#, a, b as region args; the locals c, d arrive as cells
    @test length(root.args) == 3
    @test count(s -> UnifiedIR.stmt_kind(ir, s) === K"cell",
                UnifiedIR.each_stmt(ir)) == 2
    st = UnifiedCompiler.UInferState()
    oir = UnifiedCompiler.optimize_ir!(ir, Any[Compiler.Const(f_with_slots), Int, Int];
                                       state = st)
    root2 = UnifiedIR.getregion(oir, UnifiedIR.root_region(oir))
    @test length(root2.args) == 3
    # after optimization the locals are SSA: no cells remain
    @test count(s -> UnifiedIR.stmt_kind(oir, s) === K"cell",
                UnifiedIR.each_stmt(oir)) == 0
end

@testset "IR printing smoke + text round trip (IRShow analogue)" begin
    irshow_smoke(x) = (y = x + 1; y)
    tir = UnifiedCompiler.typed_ir(irshow_smoke, Any[Int])
    out = UnifiedIR.print_ir(tir)
    @test occursin("return", out)
    # the portable-subset round trip (print → parse → structural equality)
    b = Builder(name = :rt)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    v = append_stmt!(b, K"call", GlobalRef(Base, :add_int), x, 1; type = Any)
    append_stmt!(b, K"return", v)
    ir = finish!(b)
    ir2 = UnifiedIR.parse_ir(UnifiedIR.print_ir(ir))
    @test UnifiedIR.struct_eq(ir, ir2)
end

# ---------------------------------------------------------------------------
# conditional-successor execution semantics (ssair.jl
# visit_conditional_successors trio: the traversal API is irinterp-internal
# and dies with it; the executed semantics — every conditionally-reachable
# arm really runs/throws on its path — port through the region form)
# ---------------------------------------------------------------------------

@testset "conditional successors: throwing arm vs value path" begin
    mkthrowarm = function (throw_on_true::Bool)
        b = Builder(name = :vcs)
        append_stmt!(b, K"region_arg"; type = Any)
        cnd = append_stmt!(b, K"region_arg"; type = Bool)
        x = append_stmt!(b, K"region_arg"; type = Any)
        cop = throw_on_true ? cnd :
            append_stmt!(b, K"call", GlobalRef(Base, :!), cnd; type = Any)
        fi = append_stmt!(b, K"if", op_stmt(cop); type = Nothing)
        UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
        append_stmt!(b, K"call", GlobalRef(Base, :throw), "potential throw"; type = Any)
        append_stmt!(b, K"unreachable")
        UnifiedIR.close_region!(b)
        append_stmt!(b, K"return", x)
        return finish!(b)
    end
    for throw_on_true in (true, false)
        ir = mkthrowarm(throw_on_true)
        @test UnifiedIR.verify_ir(ir; level = 1)
        # both conditional successors are visible in the region tree: the
        # diverging arm and the fall-through continuation
        fiop = only(s for s in UnifiedIR.each_stmt(ir)
                    if UnifiedIR.stmt_kind(ir, s) === K"if")
        arms = [r for r in UnifiedIR.owned_regions(ir, fiop)
                if !UnifiedIR.getregion(ir, r).dead]
        @test length(arms) == 1
        t = UnifiedIR.region_terminator(ir, arms[1])
        @test t !== nothing && UnifiedIR.stmt_kind(ir, t) === K"unreachable"
        irc = UnifiedCompiler.ir_to_ircode(ir)
        @test Compiler.verify_ir(irc) === nothing
        oc = Core.OpaqueClosure(irc)
        goodcond = !throw_on_true
        @test oc(goodcond, 1) == 1
        @test_throws "potential throw" oc(!goodcond, 1)
        @test UnifiedIR.interpret(ir, nothing, goodcond, 1) == 1
    end
    # the two-armed variant: add + return vs throw (ssair.jl's third case)
    b = Builder(name = :vcs3)
    append_stmt!(b, K"region_arg"; type = Any)
    cnd = append_stmt!(b, K"region_arg"; type = Bool)
    xx = append_stmt!(b, K"region_arg"; type = Any)
    yy = append_stmt!(b, K"region_arg"; type = Any)
    fi = append_stmt!(b, K"if", op_stmt(cnd); type = Nothing)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"call", GlobalRef(Base, :throw), "potential throw"; type = Any)
    append_stmt!(b, K"unreachable")
    UnifiedIR.close_region!(b)
    v = append_stmt!(b, K"call", GlobalRef(Core.Intrinsics, :add_int), xx, yy; type = Any)
    append_stmt!(b, K"return", v)
    ir = finish!(b)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    oc = Core.OpaqueClosure(irc)
    @test oc(false, 1, 1) == 2
    @test_throws "potential throw" oc(true, 1, 1)
end

end # module B3APortSsair
