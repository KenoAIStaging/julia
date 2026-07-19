# Deep invalidation-parity checks for driver-produced CodeInstances
# (invalidation.jl's semantic content under the unified pipeline): method
# redefinition decays max_world through direct edges, :invoke edges and
# INLINED callees; binding redefinition invalidates readers while ordinary
# assignment does not; mi.cache chains carry the expected owner/world
# structure; and the cross-request memo (A6) respects all of it — a
# redefinition mid-session must never replay stale facts. Included from
# driver.jl (same helper namespace: drv_interp/drv_mi/CC/UnifiedCompiler).

dinv_mi(f, args...) = Base.invokelatest(drv_mi, f, args...)
dinv_getf(name) = Base.invokelatest(getglobal, @__MODULE__, name)

# --- direct (dynamic residual) call edges -----------------------------------

@eval dinv_dyn_callee(x::Int) = 1
@eval dinv_dyn_caller(x::Integer) = dinv_dyn_callee(x)

@testset "driver invalidation: method-table edge on a residual dynamic call" begin
    # the single match does not fully cover Integer: the site stays a dynamic
    # call carrying a method-match edge; ADDING an intersecting method must
    # decay the caller's CodeInstance
    mi = CC.specialize_method(Base._which(Tuple{typeof(dinv_dyn_caller), Integer};
                                          world = Base.get_world_counter()))
    ci = unified_typeinf(drv_interp(), mi, CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.max_world == typemax(UInt)
    @eval dinv_dyn_callee(x::Int8) = 2
    @test ci.max_world != typemax(UInt)
    @test Base.invokelatest(dinv_dyn_caller, Int8(1)) == 2
    @test Base.invokelatest(dinv_dyn_caller, 1) == 1
end

# --- the inlining edge must invalidate the inliner --------------------------

@eval dinv_inl_callee(x::Int) = x + 1
@eval dinv_inl_caller(x::Int) = dinv_inl_callee(x) * 2

@testset "driver invalidation: redefinition reaches through INLINED callees" begin
    interp = drv_interp()
    ci = unified_typeinf(interp, dinv_mi(dinv_getf(:dinv_inl_caller), 1), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    src = CC.ci_get_source(interp, ci)
    @test src isa Core.CodeInfo
    # the callee really was inlined: no residual call/invoke of it remains
    mentions_callee(st) = (Meta.isexpr(st, :call) || Meta.isexpr(st, :invoke)) &&
        Base.any(a -> (a isa GlobalRef && a.name === :dinv_inl_callee) ||
                      (a isa Core.CodeInstance &&
                       CC.get_ci_mi(a).def.name === :dinv_inl_callee), st.args)
    @test !Base.any(mentions_callee, src.code)
    @test ci.max_world == typemax(UInt)
    @test Base.invoke(dinv_inl_caller, ci, 5) == 12
    @eval dinv_inl_callee(x::Int) = x - 1
    @test ci.max_world != typemax(UInt)     # the inliner decayed
    @test Base.invokelatest(dinv_inl_caller, 5) == 8
end

# --- :invoke edges + cache-chain structure ----------------------------------

@eval @noinline dinv_inv_callee(x::Int) = begin s = 0; for i in 1:x; s += i * i; end; s end
@eval dinv_inv_caller(x::Int) = dinv_inv_callee(x) + 1

@testset "driver invalidation: :invoke edge decay and mi.cache chain shape" begin
    interp = drv_interp()
    mi = dinv_mi(dinv_getf(:dinv_inv_caller), 3)
    ci = unified_typeinf(interp, mi, CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    src = CC.ci_get_source(interp, ci)
    @test Base.any(st -> Meta.isexpr(st, :invoke), src.code)
    @test ci.max_world == typemax(UInt)
    world_before = Base.get_world_counter()
    @eval @noinline dinv_inv_callee(x::Int) = -x
    @test ci.max_world != typemax(UInt)
    # decayed to a bounded world in [pre-redefinition, current)
    @test world_before <= ci.max_world < Base.get_world_counter()
    # recompile at the new world: the SAME mi's cache chains both generations
    ci2 = unified_typeinf(drv_interp(), dinv_mi(dinv_getf(:dinv_inv_caller), 3),
                          CC.SOURCE_MODE_ABI)
    @test ci2 isa Core.CodeInstance
    @test ci2.max_world == typemax(UInt)
    @test ci2.min_world > ci.max_world      # non-overlapping validity windows
    @test Base.invoke(dinv_inv_caller, ci2, 3) == -2
    chain = Core.CodeInstance[]
    walk = mi.cache
    while walk isa Core.CodeInstance
        walk.owner === nothing && push!(chain, walk)
        walk = isdefined(walk, :next) ? walk.next : nothing
    end
    @test ci in chain && ci2 in chain
    # exactly one live entry; every superseded generation is world-bounded
    live = [c for c in chain if c.max_world == typemax(UInt)]
    @test length(live) == 1 && live[1] === ci2
    for c in chain
        c === ci2 && continue
        @test c.max_world < ci2.min_world
    end
end

# --- backedge transitivity: leaf redefinition reaches the top through the
# --- middle's CodeInstance (whose body INLINED the leaf) --------------------

@eval dinv_t_leaf(x::Int) = x + 1
@eval @noinline dinv_t_mid(x::Int) = dinv_t_leaf(x) * 2
@eval dinv_t_top(x::Int) = dinv_t_mid(x) + 3

@testset "driver invalidation: transitivity through an inlined leaf" begin
    interp = drv_interp()
    ci_top = unified_typeinf(interp, dinv_mi(dinv_getf(:dinv_t_top), 1), CC.SOURCE_MODE_ABI)
    @test ci_top isa Core.CodeInstance
    @test ci_top.max_world == typemax(UInt)
    @test Base.invoke(dinv_t_top, ci_top, 5) == 15
    @eval dinv_t_leaf(x::Int) = x - 1
    # the leaf edge lives on mid's CI; mid's mi backedges carry the decay up
    @test ci_top.max_world != typemax(UInt)
    @test Base.invokelatest(dinv_t_top, 5) == 11
end

# --- binding redefinition invalidates readers; assignment does not ----------

@eval const DINV_CONST = 10
@eval dinv_cread(x::Int) = x + DINV_CONST

@testset "driver invalidation: const redefinition invalidates the reader" begin
    ci = unified_typeinf(drv_interp(), dinv_mi(dinv_getf(:dinv_cread), 1), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test Base.any(e -> e isa Core.Binding, ci.edges)   # the binding edge is real
    @test ci.max_world == typemax(UInt)
    @test Base.invoke(dinv_cread, ci, 1) == 11
    @eval const DINV_CONST = 100
    @test ci.max_world != typemax(UInt)
    @test Base.invokelatest(dinv_cread, 1) == 101
    ci2 = unified_typeinf(drv_interp(), dinv_mi(dinv_getf(:dinv_cread), 1), CC.SOURCE_MODE_ABI)
    @test ci2 isa Core.CodeInstance
    @test Base.invoke(dinv_cread, ci2, 1) == 101
end

@eval global dinv_tg::Int = 1
@eval dinv_gread(x::Int) = x + dinv_tg

@testset "driver invalidation: typed-global assignment does NOT invalidate" begin
    ci = unified_typeinf(drv_interp(), dinv_mi(dinv_getf(:dinv_gread), 1), CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test ci.rettype === Int                     # the declared type, not the value
    @test ci.max_world == typemax(UInt)
    @test Base.invoke(dinv_gread, ci, 1) == 2
    Base.invokelatest(setglobal!, @__MODULE__, :dinv_tg, 41)
    @test ci.max_world == typemax(UInt)          # a value write moves no world
    @test Base.invoke(dinv_gread, ci, 1) == 42   # and the same code reads it
end

# --- the memo layer respects invalidation -----------------------------------

@eval dinv_m_leaf(x::Int) = 1.0
@eval @noinline dinv_m_mid(x::Int) = dinv_m_leaf(x)
@eval dinv_m_top(x::Int) = dinv_m_mid(x)

@testset "driver memo: replay serves equal results at a stable world" begin
    UnifiedCompiler.reset_pipeline_stats!()
    m = dinv_mi(dinv_getf(:dinv_m_top), 1)
    r1 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(), m)
    @test r1 isa UnifiedCompiler.DriverResult
    h1 = pipeline_stats().memo.hits
    r2 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(), m)
    @test r2 isa UnifiedCompiler.DriverResult
    @test pipeline_stats().memo.hits > h1      # the callee tree replayed
    @test r1.rt == r2.rt
    @test r1.effects == r2.effects
    @test r1.exct == r2.exct
    # the replayed pass carries the same edge set (sound worlds either way)
    @test Base.length(r1.edges) == Base.length(r2.edges)
end

@testset "driver memo: a mid-session redefinition must not replay stale facts" begin
    UnifiedCompiler.reset_pipeline_stats!()
    r1 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_m_top), 1))
    @test r1 isa UnifiedCompiler.DriverResult
    @test CC.widenconst(r1.rt) === Float64
    # redefine the LEAF: the memo entry for the UNCHANGED mid must not serve
    # its recorded Float64 facts at the new world
    @eval dinv_m_leaf(x::Int) = "changed"
    r2 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_m_top), 1))
    @test r2 isa UnifiedCompiler.DriverResult
    @test CC.widenconst(r2.rt) === String       # no stale replay
    @test pipeline_stats().memo.stale >= 1      # the entry was dropped, honestly
    # and the refreshed memo serves the NEW result thereafter
    r3 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_m_top), 1))
    @test r3 isa UnifiedCompiler.DriverResult
    @test CC.widenconst(r3.rt) === String
end

@eval const DINV_MB = 1
@eval dinv_mb_read() = DINV_MB + 1
@eval dinv_mb_top(x::Int) = dinv_mb_read()

@testset "driver memo: binding redefinition invalidates memoized readers" begin
    UnifiedCompiler.reset_pipeline_stats!()
    r1 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_mb_top), 1))
    @test r1 isa UnifiedCompiler.DriverResult
    @test r1.rettype_const == 2                 # the folded partition value
    @eval const DINV_MB = 10
    r2 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_mb_top), 1))
    @test r2 isa UnifiedCompiler.DriverResult
    @test r2.rettype_const == 11                # the new partition, not a replay
end

# --- memo-produced CodeInstances still carry complete edges -----------------

@eval dinv_e_leaf(x::Int) = x * 3
@eval dinv_e_top(x::Int) = dinv_e_leaf(x) + 1

@testset "driver memo: replayed edges keep CodeInstances invalidation-complete" begin
    # first request populates the memo; a SECOND caller of the same callee is
    # then built from replayed facts — its CI must still decay on redefinition
    r0 = Base.invokelatest(UnifiedCompiler.driver_infer, drv_interp(),
                           dinv_mi(dinv_getf(:dinv_e_top), 1))
    @test r0 isa UnifiedCompiler.DriverResult
    @eval dinv_e_top2(x::Int) = dinv_e_leaf(x) + 2
    h0 = pipeline_stats().memo.hits
    ci = unified_typeinf(drv_interp(), dinv_mi(dinv_getf(:dinv_e_top2), 1),
                         CC.SOURCE_MODE_ABI)
    @test ci isa Core.CodeInstance
    @test pipeline_stats().memo.hits > h0       # the leaf came from the memo
    @test ci.max_world == typemax(UInt)
    @test Base.invoke(dinv_e_top2, ci, 5) == 17
    @eval dinv_e_leaf(x::Int) = 0
    @test ci.max_world != typemax(UInt)         # replayed edges invalidate too
    @test Base.invokelatest(dinv_e_top2, 5) == 2
end
