# The real typeinf driver (driver.jl): cache-grade CodeInstances, the
# UNIFIED_HOOKS reflection bridge, edges/invalidation, the fallback ledger,
# and global activation (subprocess). Included from runtests.jl.

drv_add(a, b) = a + b
drv_mysum(n) = begin s = 0; i = 1; while i <= n; s += i; i += 1; end; s end
drv_const() = 42
drv_try(x) = try; div(10, x); catch; -1; end
@generated drv_gen(x) = :(x + 1)
drv_callee(x) = x + 1
drv_caller(x) = drv_callee(x) * 10
drv_tailwhile(r) = while true; r[] && break; end

drv_interp() = CC.NativeInterpreter(Base.get_world_counter())
drv_mi(f, args...) = UnifiedCompiler.lookup_method_instance(f, args...)

@testset "driver: cache-grade CodeInstances" begin
    UnifiedCompiler.reset_pipeline_stats!()
    ci = unified_typeinf(drv_interp(), drv_mi(drv_add, 1, 2), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int
    @test ci.min_world >= 1 && ci.max_world == typemax(UInt)   # promoted to current
    @test invoke(drv_add, ci, 20, 22) == 42                    # the ABI is real
    # the loop body: inferred Int, executes right
    ci2 = unified_typeinf(drv_interp(), drv_mi(drv_mysum, 10), CC.SOURCE_MODE_ABI)
    @test ci2 isa Core.CodeInstance
    @test ci2.rettype === Int
    @test invoke(drv_mysum, ci2, 100) == 5050
    @test invoke(drv_mysum, ci2, 0) == 0
    @test pipeline_stats().unified >= 2
    # a second request is a cache hit (same CodeInstance, no new pass)
    n0 = pipeline_stats().unified
    ci3 = unified_typeinf(drv_interp(), drv_mi(drv_mysum, 10), CC.SOURCE_MODE_ABI)
    @test ci3 === ci2
    @test pipeline_stats().unified == n0
end

@testset "driver: const-ABI results match stock criteria" begin
    ci = unified_typeinf(drv_interp(), drv_mi(drv_const), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int
    @test isdefined(ci, :rettype_const) && ci.rettype_const == 42
    @test invoke(drv_const, ci) == 42
end

@testset "driver: maybe-undef break-block result slot (wave 13)" begin
    # A tail/value-position `while` lowers to a break-block whose `loop-exit`
    # result slot is maybe-undef (the taken `break` skips the store; an
    # `Expr(:isdefined, ...)` guard backfills `nothing`) but carries NO
    # NewvarNode — lowering materializes the slot after the newvar machinery.
    # The entry converters must declare the undefined-at-entry state with a
    # `cell_new`, or the definedness transfer folds the guard to Const(true),
    # the undef-path store dies, and the compiled body throws
    # `UndefVarError(:loop-exit)` the moment the break is taken (seen as the
    # REPL stdlib precompile deadlock: LineEdit `prompt!`'s spawned input
    # loop dying on its first key dispatch).
    ci = unified_typeinf(drv_interp(), drv_mi(drv_tailwhile, Ref(true)), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test invoke(drv_tailwhile, ci, Ref(true)) === nothing
end

@testset "driver: try/catch compiles through unified (A4)" begin
    UnifiedCompiler.reset_pipeline_stats!()
    # try/catch bodies go through the typed exit's exception-SSA synthesis:
    # cache-grade CodeInstance, correct on the normal AND the handler path
    ci = unified_typeinf(drv_interp(), drv_mi(drv_try, 4), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int
    @test invoke(drv_try, ci, 5) == 2          # normal path through the real ABI
    @test invoke(drv_try, ci, 0) == -1         # handler path through the real ABI
    @test get(pipeline_stats().fallbacks, :typed_exit, 0) == 0
    @test pipeline_stats().unified >= 1
    @test drv_try(5) == 2 && drv_try(0) == -1
end

@testset "driver: generated functions expand and compile (A5)" begin
    UnifiedCompiler.reset_pipeline_stats!()
    # simple staged body
    ci = unified_typeinf(drv_interp(), drv_mi(drv_gen, 1), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int
    @test invoke(drv_gen, ci, 41) == 42
    @test pipeline_stats().unified >= 1
    @test drv_gen(41) == 42
    # a generator whose own execution needs fresh inference
    @gensym helper genf
    @eval $helper(T) = string(nameof(T))
    @eval @generated function $genf(x)
        nm = $helper(x)
        return :(($nm, x))
    end
    gf = Base.invokelatest(getglobal, @__MODULE__, genf)
    ci2 = unified_typeinf(drv_interp(), drv_mi(gf, 7), CC.SOURCE_MODE_ABI)
    @test ci2 isa Core.CodeInstance
    @test invoke(gf, ci2, 7) == ("Int64", 7)
    # a staged body with control flow (loop over a type-computed count)
    @gensym genloop
    @eval @generated function $genloop(x, n)
        quote
            s = zero(x)
            for i in 1:n
                s += x
            end
            s
        end
    end
    gl = Base.invokelatest(getglobal, @__MODULE__, genloop)
    ci3 = unified_typeinf(drv_interp(), drv_mi(gl, 2.5, 4), CC.SOURCE_MODE_ABI)
    @test ci3 isa Core.CodeInstance
    @test invoke(gl, ci3, 2.5, 4) == 10.0
end

@noinline drv_bigcallee(x::Int) = begin s = 0; for i in 1:x; s += i * i; end; s end
drv_devcaller(x::Int) = drv_bigcallee(x) + 1

@testset "driver: :invoke emission for statically-resolved residual calls (A5)" begin
    interp = drv_interp()
    ci = unified_typeinf(interp, drv_mi(drv_devcaller, 3), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    src = CC.ci_get_source(interp, ci)
    @test src isa Core.CodeInfo
    invokes = [st for st in src.code if Meta.isexpr(st, :invoke)]
    @test length(invokes) == 1
    tgt = invokes[1].args[1]
    @test tgt isa Core.CodeInstance
    @test CC.get_ci_mi(tgt).def.name === :drv_bigcallee
    @test invoke(drv_devcaller, ci, 10) == drv_devcaller(10)
    # redefining the devirtualized callee invalidates the caller
    @eval @noinline drv_bigcallee(x::Int) = -1
    @test ci.max_world != typemax(UInt)
    @test drv_devcaller(10) == 0
end

@noinline drv_probe_str(s::String) = s * "!"
@noinline drv_probe_sum(v::Vector{Int}) = sum(v)
@noinline drv_probe_two(a::Int, b::Int) = a === b ? a : a - b
drv_probe_c1(s::String) = drv_probe_str(s)
drv_probe_c2(v::Vector{Int}) = drv_probe_sum(v) + 1
drv_probe_c3(x::Int) = drv_probe_two(x, 2x)

@testset "driver: :invoke class parity with stock on a probe set" begin
    saved = Base.REFLECTION_COMPILER[]
    count_invokes(src) = count(st -> Meta.isexpr(st, :invoke), src.code)
    try
        for (f, at) in ((drv_probe_c1, (String,)), (drv_probe_c2, (Vector{Int},)),
                        (drv_probe_c3, (Int,)))
            Base.REFLECTION_COMPILER[] = nothing
            disable_pipeline!()
            (stock_src, _) = only(Base.code_typed(f, at))
            Base.REFLECTION_COMPILER[] = Compiler
            enable_pipeline!()
            (uni_src, _) = only(Base.code_typed(f, at))
            # the statically-resolved @noinline callee is an :invoke under
            # both pipelines, and the unified target is a CodeInstance
            @test count_invokes(stock_src) >= 1
            @test count_invokes(uni_src) >= count_invokes(stock_src)
            tgt = [st for st in uni_src.code if Meta.isexpr(st, :invoke)][1].args[1]
            @test tgt isa Union{Core.CodeInstance,Core.MethodInstance}
        end
    finally
        disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

# a non-fully-covering single match must neither inline nor devirtualize
# (dispatch still throws MethodError for the uncovered part)
drv_cov_callee(x::Int) = 1
drv_cov_caller(x::Integer) = drv_cov_callee(x)

@testset "driver: non-covering matches keep dynamic dispatch (soundness)" begin
    mi = CC.specialize_method(Base._which(Tuple{typeof(drv_cov_caller), Integer};
                                          world = Base.get_world_counter()))
    ci = unified_typeinf(drv_interp(), mi, CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test invoke(drv_cov_caller, ci, 1) == 1
    @test_throws MethodError invoke(drv_cov_caller, ci, Int8(1))
end

# an under-constrained match (free TypeVar in the re-derived sparams) must
# not devirtualize: the emitted :invoke would compile the callee with an
# unbound static parameter (`UndefVarError: T` at the first sparam use);
# the site must keep the dynamic :call (stock's validate_sparams rule)
@noinline drv_sp_callee(x::Vector{T}) where {T} = T
drv_sp_caller(v) = drv_sp_callee(v)

@testset "driver: under-constrained sparams keep dynamic dispatch (soundness)" begin
    mi = CC.specialize_method(Base._which(Tuple{typeof(drv_sp_caller), Vector};
                                          world = Base.get_world_counter()))
    interp = drv_interp()
    ci = unified_typeinf(interp, mi, CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    src = CC.ci_get_source(interp, ci)
    @test src isa Core.CodeInfo
    # no :invoke to the under-constrained callee survives
    @test !any(st -> Meta.isexpr(st, :invoke), src.code)
    # the dynamic call re-derives T per concrete argument type
    @test invoke(drv_sp_caller, ci, [1, 2, 3]) === Int
    @test invoke(drv_sp_caller, ci, Any["x"]) === Any
end

@testset "driver: reentrant requests run unified (per-task bound)" begin
    # a nested direct request (same task, different mi) is admitted, not
    # blanket-declined: the depth guard only rejects at the bound
    UnifiedCompiler.reset_pipeline_stats!()
    dts = UnifiedCompiler.driver_task_state()
    @test dts.depth == 0
    @eval drv_nested_probe(x) = x + 2
    f = Base.invokelatest(getglobal, @__MODULE__, :drv_nested_probe)
    dts.depth = 1   # simulate arriving mid-driver-pass
    ci = try
        unified_typeinf(drv_interp(), drv_mi(f, 1), CC.SOURCE_MODE_ABI)
    finally
        dts.depth = 0
    end
    @test ci isa Core.CodeInstance
    @test pipeline_stats().unified >= 1
    # at the bound: precise decline, counted
    dts.depth = UnifiedCompiler.DRIVER_REENTRY_LIMIT[]
    @eval drv_depth_probe(x) = x + 3
    f2 = Base.invokelatest(getglobal, @__MODULE__, :drv_depth_probe)
    ci2 = try
        unified_typeinf(drv_interp(), drv_mi(f2, 1), CC.SOURCE_MODE_ABI)
    finally
        dts.depth = 0
    end
    @test ci2 === nothing
    @test get(pipeline_stats().fallbacks, :reentrant_depth, 0) >= 1
    # an mi already being driven by this task declines precisely
    @eval drv_self_probe(x) = x + 4
    f3 = Base.invokelatest(getglobal, @__MODULE__, :drv_self_probe)
    mi3 = drv_mi(f3, 1)
    push!(dts.inflight, mi3)
    ci3 = try
        unified_typeinf(drv_interp(), mi3, CC.SOURCE_MODE_ABI)
    finally
        delete!(dts.inflight, mi3)
    end
    @test ci3 === nothing
    @test get(pipeline_stats().fallbacks, :reentrant_self, 0) >= 1
end

@testset "driver: redefinition invalidation (edges/world bounds)" begin
    interp = drv_interp()
    mi = drv_mi(drv_caller, 5)
    ci = unified_typeinf(interp, mi, CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int
    @test !isempty(ci.edges)                   # the callee dependency is recorded
    @test ci.max_world == typemax(UInt)
    @test invoke(drv_caller, ci, 5) == 60
    # redefine the callee: the recorded edges must invalidate the caller CI
    @eval drv_callee(x) = x * 1.0
    @test ci.max_world != typemax(UInt)
    # fresh inference at the new world sees the new callee and executes right
    ci2 = unified_typeinf(drv_interp(), drv_mi(drv_caller, 5), CC.SOURCE_MODE_ABI)
    if ci2 isa Core.CodeInstance               # per-body fallback is legal, staleness is not
        @test ci2.rettype === Float64
        @test invoke(drv_caller, ci2, 5) == 50.0
    end
    @test drv_caller(5) == 50.0
    # binding edge: a const the body folded gets a backedge too
    @eval const DRV_CONST = 7
    @eval drv_readc(x) = x + DRV_CONST
    ci3 = unified_typeinf(drv_interp(), drv_mi(Base.invokelatest(getglobal, @__MODULE__, :drv_readc), 1),
                          CC.SOURCE_MODE_ABI)
    @test ci3 isa Core.CodeInstance
    @test any(e -> e isa Core.Binding, ci3.edges)
end

@testset "driver: reflection bridge (code_typed / infer_effects / exct)" begin
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        enable_pipeline!()
        UnifiedCompiler.reset_pipeline_stats!()
        # code_typed through the hook: unified SSA shape
        (src, rt) = only(Base.code_typed(drv_add, (Int, Int)))
        @test rt === Int
        @test src isa Core.CodeInfo
        @test src.code[end] isa Core.ReturnNode
        @test count(st -> Meta.isexpr(st, :call), src.code) == 1  # one add_int
        @test pipeline_stats().unified >= 1
        # const-abi shape: code_typed of a trivial const is the constant return
        (csrc, crt) = only(Base.code_typed(drv_const, ()))
        @test crt === Int
        @test length(csrc.code) == 1 && csrc.code[1] isa Core.ReturnNode
        # infer_effects: + on Int is total-ish under the unified flags
        effects = Base.infer_effects(drv_add, (Int, Int))
        @test Compiler.is_consistent(effects)
        @test Compiler.is_effect_free(effects)
        @test Compiler.is_nothrow(effects)
        @test Compiler.is_terminates(effects)
        effects = Base.infer_effects(drv_try, (Int,))   # falls back to stock, still sane
        @test effects isa Compiler.Effects
        # exception types: nothrow body -> Union{}, unknown -> Any
        @test Base.infer_exception_type(drv_add, (Int, Int)) === Union{}
        # a body the pipeline handles but cannot prove nothrow
        @test Base.infer_exception_type(drv_mysum, (Int,)) isa Type
    finally
        disable_pipeline!()
        Base.REFLECTION_COMPILER[] = saved
    end
end

@testset "driver: disable restores stock exactly" begin
    @test Compiler.UNIFIED_HOOKS[] === nothing
    saved = Base.REFLECTION_COMPILER[]
    try
        Base.REFLECTION_COMPILER[] = Compiler
        (src_off, rt_off) = only(Base.code_typed(drv_add, (Int, Int)))
        enable_pipeline!()
        @test Compiler.UNIFIED_HOOKS[] !== nothing
        disable_pipeline!()
        @test Compiler.UNIFIED_HOOKS[] === nothing
        (src_off2, rt_off2) = only(Base.code_typed(drv_add, (Int, Int)))
        @test rt_off === rt_off2
        @test length(src_off.code) == length(src_off2.code)
        n0 = pipeline_stats().unified
        ci = Compiler.typeinf_ext_toplevel(drv_interp(), drv_mi(drv_add, 1, 2), Compiler.SOURCE_MODE_ABI)
        @test ci isa Core.CodeInstance
        @test pipeline_stats().unified == n0   # the driver did not run
    finally
        Base.REFLECTION_COMPILER[] = saved
    end
end

@testset "driver: pipeline_stats ledger" begin
    UnifiedCompiler.reset_pipeline_stats!()
    st0 = pipeline_stats()
    @test st0.unified == 0 && isempty(st0.fallbacks) && st0.last_error === nothing
    @eval drv_fresh_ledger(x) = x + 3
    ci = unified_typeinf(drv_interp(), drv_mi(Base.invokelatest(getglobal, @__MODULE__, :drv_fresh_ledger), 1),
                         CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    # a generated body whose generator throws: the expansion is unavailable,
    # counted as a precise :staged_source decline (stock then reproduces the
    # call-time generator error)
    @eval @generated drv_fresh_gen(x) = error("no expansion for you")
    unified_typeinf(drv_interp(), drv_mi(Base.invokelatest(getglobal, @__MODULE__, :drv_fresh_gen), 4),
                    CC.SOURCE_MODE_ABI)
    st1 = pipeline_stats()
    @test st1.unified >= 1
    @test get(st1.fallbacks, :staged_source, 0) >= 1
    io = IOBuffer()
    UnifiedCompiler.print_pipeline_stats(io)
    out = String(take!(io))
    @test occursin("unified pipeline:", out)
    @test occursin("memo hits", out)
end

@testset "driver: memo counters in the ledger" begin
    UnifiedCompiler.reset_pipeline_stats!()
    m0 = pipeline_stats().memo
    @test m0.hits == 0 && m0.misses == 0 && m0.stale == 0 && m0.stores == 0
    @eval drv_memo_leaf(x) = x + 5
    @eval drv_memo_top(x) = drv_memo_leaf(x) * 2
    f = Base.invokelatest(getglobal, @__MODULE__, :drv_memo_top)
    interp = drv_interp()
    r1 = Base.invokelatest(UnifiedCompiler.driver_infer, interp, drv_mi(f, 1))
    @test r1 isa UnifiedCompiler.DriverResult
    m1 = pipeline_stats().memo
    @test m1.stores >= 1                     # the leaf frame entered the memo
    r2 = Base.invokelatest(UnifiedCompiler.driver_infer, interp, drv_mi(f, 1))
    @test r2 isa UnifiedCompiler.DriverResult
    m2 = pipeline_stats().memo
    @test m2.hits > m1.hits                  # the second request replayed it
    @test m2.replayed > m1.replayed
    @test m2.entries >= 1
    @test r1.rt == r2.rt && r1.effects == r2.effects
end

# invalidation.jl-style deep parity: edges/worlds/cache chains/memo staleness
include("driver_invalidation.jl")

@testset "driver: global activation executes correctly (subprocess)" begin
    # jl_set_typeinf_func flips are process-global: exercise them in a child
    script = """
    pushfirst!(LOAD_PATH, joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia"))
    import Compiler
    const U = Compiler.load_unified!()
    U.activate!()
    # fresh bodies compiled under the flipped runtime
    gsum(n) = begin s = 0; i = 1; while i <= n; s += i; i += 1; end; s end
    gsum(100) == 5050 || exit(1)
    gbranch(x) = x > 10 ? "big" : x > 0 ? "small" : "neg"
    (gbranch(11) == "big" && gbranch(5) == "small" && gbranch(-1) == "neg") || exit(2)
    gtry(x) = try; div(10, x); catch; -1; end      # try/catch: unified EH exit (A4)
    (gtry(5) == 2 && gtry(0) == -1) || exit(3)
    join(sort([3, 1, 2]), "-") == "1-2-3" || exit(4)
    stats = U.pipeline_stats()
    stats.unified >= 1 || exit(5)
    U.deactivate!()
    Compiler.UNIFIED_HOOKS[] === nothing || exit(6)
    gafter(n) = n + 1
    gafter(41) == 42 || exit(7)
    print("SUBPROCESS_OK unified=", stats.unified)
    """
    jl = joinpath(Sys.BINDIR, "julia")
    out = read(pipeline(`$jl --startup-file=no -e $script`; stderr = devnull), String)
    @test occursin("SUBPROCESS_OK", out)
end
