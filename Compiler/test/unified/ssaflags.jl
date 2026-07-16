# Statement-level ssaflags carriage (A5/E3): the entry converters copy
# per-statement `@inbounds`/`@inline`/`@noinline`/`@assume_effects` context
# from stock `ssaflags` into the UnifiedIR flag column; inference applies the
# statement overrides (stock merge_override_effects!) and resolves
# NOUB_IF_NOINBOUNDS per statement. Included from runtests.jl.

ssf_override(x) = Base.@assume_effects :nothrow sin(x)
function ssf_nested(xs)
    isempty(xs) && return 0.0
    Base.@assume_effects :nothrow begin
        x = Base.@assume_effects :noub @inbounds xs[1]
        isinf(x) && return 0.0
        return sin(x)
    end
end
function ssf_termloc(x)
    res = 1
    0 <= x < 20 || error("bad fact")
    Base.@assume_effects :terminates_locally while x > 1
        res *= x
        x -= 1
    end
    return res
end
ssf_inb(xs, i) = @inbounds xs[i]
ssf_plain(xs, i) = xs[i]
ssf_noinl_callee(x) = x + 1
ssf_noinl_site(x) = @noinline ssf_noinl_callee(x)

@testset "ssaflags: entry carriage into the flag column" begin
    # the callsite-@assume_effects flag lands on the marked call statement
    src = only(Base.code_lowered(ssf_override, (Float64,)))
    @test any(!iszero, src.ssaflags)   # lowering did flag something
    ir = UnifiedCompiler.codeinfo_to_ir(src; nargs = 2, name = :ssf_override)
    found = 0
    for i in 1:UnifiedIR.nstmts(ir)
        s = UnifiedIR.StmtId(Int32(i))
        ov = UnifiedCompiler.stmt_effects_override(ir, s)
        ov.nothrow && (found += 1)
    end
    @test found >= 1
    # @inbounds marks IR_FLAG_INBOUNDS; the converter carries FLAG_INBOUNDS
    src = only(Base.code_lowered(ssf_inb, (Vector{Int}, Int)))
    ir = UnifiedCompiler.codeinfo_to_ir(src; nargs = 3, name = :ssf_inb)
    @test any(i -> UnifiedCompiler.stmt_inbounds(ir, UnifiedIR.StmtId(Int32(i))),
              1:UnifiedIR.nstmts(ir))
    # an unflagged body carries nothing
    src = only(Base.code_lowered(ssf_plain, (Vector{Int}, Int)))
    ir = UnifiedCompiler.codeinfo_to_ir(src; nargs = 3, name = :ssf_plain)
    @test !any(i -> UnifiedCompiler.stmt_inbounds(ir, UnifiedIR.StmtId(Int32(i))),
               1:UnifiedIR.nstmts(ir))
    # round trip through carry_ssaflags/stmt_override_bits is lossless
    for raw in (UInt32(0), CC.IR_FLAG_INBOUNDS, CC.IR_FLAG_INLINE, CC.IR_FLAG_NOINLINE,
                UInt32(0x7ff) << CC.NUM_IR_FLAGS,
                CC.IR_FLAG_INBOUNDS | (UInt32(0b10000) << CC.NUM_IR_FLAGS))
        carried = UnifiedCompiler.carry_ssaflags(raw)
        @test UnifiedCompiler.stmt_override_bits(carried) ==
              UInt16((raw >> CC.NUM_IR_FLAGS) & 0x7ff)
        @test (carried & UnifiedIR.FLAG_INBOUNDS != 0) == (raw & CC.IR_FLAG_INBOUNDS != 0)
    end
end

@testset "ssaflags: statement overrides drive frame effects (effects.jl:1329-1353)" begin
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        enable_pipeline!()
        @test Compiler.is_nothrow(Base.infer_effects(ssf_override, (Float64,)))
        effects = Base.infer_effects(ssf_nested, (Vector{Float64},))
        @test Compiler.is_nothrow(effects)          # nested :nothrow block applies
        @test Compiler.is_noub(effects)             # nested :noub applies through @inbounds
        @test Compiler.is_terminates(Base.infer_effects(ssf_termloc, (Int,)))
    finally
        disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

@testset "ssaflags: execution unchanged under the pipeline (differential)" begin
    xs = [1.0, 2.0, 3.0]
    @test ssf_nested(xs) == sin(1.0)
    @test ssf_nested(Float64[]) == 0.0
    @test ssf_termloc(5) == 120
    @test ssf_inb([10, 20, 30], 2) == 20
    ci = unified_typeinf(CC.NativeInterpreter(Base.get_world_counter()),
                         UnifiedCompiler.lookup_method_instance(ssf_termloc, 5),
                         CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test invoke(ssf_termloc, ci, 5) == 120
    @test invoke(ssf_termloc, ci, 0) == 1
    ci = unified_typeinf(CC.NativeInterpreter(Base.get_world_counter()),
                         UnifiedCompiler.lookup_method_instance(ssf_inb, [10, 20], 2),
                         CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test invoke(ssf_inb, ci, [10, 20], 2) == 20
end

@testset "ssaflags: callsite @noinline reaches the unified inliner" begin
    # the carried FLAG_NOINLINE keeps inline_calls2! away from the site
    ir = typed_ir(ssf_noinl_site, Any[Int])
    calls = [s for s in UnifiedIR.each_stmt(ir)
             if UnifiedIR.stmt_kind(ir, s) in (UnifiedIR.@K_str("call"), UnifiedIR.@K_str("invoke"))]
    @test !isempty(calls)   # the callee was not inlined
end
