# Re-expression of Compiler/test/compact.jl over UnifiedIR (COMPILER-PORT-
# PLAN B3a). compact.jl tests IncrementalCompact's statefulness contract:
# mutation interleaved with traversal must produce verifiable IR with the
# edited semantics. The editable layout is that contract's successor —
# stable ids during edits, one renaming point (compact!) returning a
# RemapSet — so each testset carries one original property onto it. The
# stock-internal introspection (did_just_finish_bb) has no analogue and its
# original keeps running against stock until Stage D (see
# /workspace/B3A-PORT-MAP.md).

module B3APortCompact

using Test
using UnifiedIR
using UnifiedIR: op_stmt, op_region, StmtId
import ..UnifiedCompiler
import ..CC as Compiler

# the original's fixture: foo_test_function(i) = i == 1 ? 1 : 2, entry-
# converted and structurized so the edits run over the same shape
function b3a_foo_ir()
    foo_test_function(i) = i == 1 ? 1 : 2
    ir = UnifiedCompiler.lowered_ir(foo_test_function, Tuple{Int})
    UnifiedIR.editable(ir)
    UnifiedCompiler.structurize!(ir)
    ir, _ = UnifiedIR.compact!(ir)
    UnifiedIR.verify_ir(ir; level = 1)
    return ir
end

@testset "editable statefulness: interleaved traversal and mutation" begin
    # IncrementalCompact statefulness: two iterations are set up over one
    # compaction, with mutation in between, and the result must verify.
    # Editable form: traverse, insert mid-walk, traverse again (ids stable),
    # then compact once and check verification and semantics.
    ir = b3a_foo_ir()
    @test UnifiedIR.interpret(ir, nothing, 1) == 1
    @test UnifiedIR.interpret(ir, nothing, 2) == 2
    UnifiedIR.editable(ir)
    # first walk: record every statement, insert an (unused, effect-free)
    # probe after the first call while the walk is live
    seen1 = StmtId[]
    probe = nothing
    for s in collect(UnifiedIR.each_stmt(ir))
        push!(seen1, s)
        if probe === nothing && UnifiedIR.stmt_kind(ir, s) === K"call"
            probe = UnifiedIR.insert_after!(ir, s, K"call",
                GlobalRef(Base, :identity), op_stmt(s); type = Any)
        end
    end
    @test probe isa StmtId
    # second walk over the same (edited) body: original ids are all still
    # there, plus the probe, in a consistent order
    seen2 = collect(UnifiedIR.each_stmt(ir))
    @test issubset(seen1, seen2)
    @test probe in seen2
    @test length(seen2) == length(seen1) + 1
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.remap(remap, probe).id != 0
    @test UnifiedIR.interpret(ir, nothing, 1) == 1
    @test UnifiedIR.interpret(ir, nothing, 2) == 2
end

@testset "early finish: truncate the body to a constant return" begin
    # compact.jl inserts `ReturnNode(1)` at the start and finishes early;
    # the block count collapses to 1. Editable form: rewrite the return to
    # the constant and kill the now-unneeded branch op; after compact! a
    # single region remains (the block-count-1 analogue), it verifies, and
    # the program is the constant function.
    b = Builder(name = :early)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    cnd = append_stmt!(b, K"call", GlobalRef(Base, :(==)), x, 1; type = Any)
    fi = append_stmt!(b, K"if", op_stmt(cnd); type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 1)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result", 2)
    UnifiedIR.close_region!(b)
    retstmt = append_stmt!(b, K"return", fi)
    ir = finish!(b)
    @test UnifiedIR.interpret(ir, nothing, 1) == 1
    @test UnifiedIR.interpret(ir, nothing, 2) == 2
    UnifiedIR.editable(ir)
    UnifiedIR.setop!(ir, retstmt, 1, UnifiedIR.op_inline(1))
    UnifiedIR.kill_stmt!(ir, fi)                      # cut the body short
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test UnifiedIR.remap(remap, fi).id == 0          # branch gone
    @test length(ir.regions) == 1                     # one region left
    @test UnifiedIR.interpret(ir, nothing, 1) == 1
    @test UnifiedIR.interpret(ir, nothing, 2) == 1    # the truncated program
    # and it still exits + executes through the boundary
    irc = UnifiedCompiler.ir_to_ircode(ir)
    @test Compiler.verify_ir(irc) === nothing
    @test Core.OpaqueClosure(irc)(2) == 1
end

@testset "reverse-affinity insert: entry-of-region insertion" begin
    # compact.jl's reverse-affinity insert lands a statement at the start of
    # the current block rather than after the processed position. Editable
    # form: insert_before! the first non-arg statement of a region places
    # the new statement at region entry, order and verification intact.
    ir = b3a_foo_ir()
    UnifiedIR.editable(ir)
    first_nonarg = nothing
    for s in collect(UnifiedIR.each_stmt(ir))
        UnifiedIR.stmt_region(ir, s) == UnifiedIR.root_region(ir) || continue
        UnifiedIR.stmt_kind(ir, s) === K"region_arg" && continue
        first_nonarg = s
        break
    end
    @test first_nonarg isa StmtId
    entry = UnifiedIR.insert_before!(ir, first_nonarg, K"call",
        GlobalRef(Base, :identity), 0; type = Any)
    @test UnifiedIR.comes_before(ir, entry, first_nonarg)
    ir, remap = UnifiedIR.compact!(ir)
    @test UnifiedIR.verify_ir(ir; level = 1)
    e2 = UnifiedIR.remap(remap, entry)
    @test e2.id != 0
    # after renumbering the probe is the first non-arg statement of the root
    root = UnifiedIR.getregion(ir, UnifiedIR.root_region(ir))
    firstreal = StmtId(Int32(length(root.args) + 1))
    @test e2 == firstreal
    @test UnifiedIR.interpret(ir, nothing, 1) == 1
    @test UnifiedIR.interpret(ir, nothing, 2) == 2
end

end # module B3APortCompact
