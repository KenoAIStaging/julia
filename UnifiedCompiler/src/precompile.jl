# This file is a part of Julia. License is MIT: https://julialang.org/license
#
# Precompile workload for the UnifiedCompiler pkgimage. Runs only under
# `Base.generating_output()` (see UnifiedCompiler.jl). Three phases:
#
#   1. the unified pipeline over a representative body set (loop, branch,
#      EH, closures, strings, tuples, kwargs, recursion — the demo-zoo /
#      fresh-corpus flavor), through every consumer entry: the driver
#      (`unified_typeinf`), the `Compiler.UNIFIED_HOOKS` reflection entries
#      (what `JULIA_UNIFIED_COMPILER=1` test subprocesses hit), and the
#      Queries API (what the unified suite and fresh corpus hit);
#   2. the stock `Compiler.bootstrap!` inference sweep (WITHOUT its
#      jl_set_typeinf_func flip), so `activate!`'s
#      `Compiler.activate!(codegen = true)` step resolves from cache
#      instead of "Compiling the compiler" in every process
#      (skippable via JULIA_UNIFIED_PKGIMAGE_SWEEP=0);
#   3. `_reset_session_state!` — no session state is baked.
#
# Everything here runs with the process's stock sysimage compiler as
# jl_typeinf_func; nothing flips runtime hooks beyond Compiler.UNIFIED_HOOKS,
# which lives in the (non-serialized) Compiler dependency and is restored
# before the workload ends.

module PrecompileWorkload

# representative bodies (kept pure: they are executed below)
wl_sum(n) = begin s = 0; i = 1; while i <= n; s += i; i += 1; end; s end
wl_branchy(x) = x > 10 ? "big" : x > 0 ? "small" : "neg"
wl_pow(x, n) = begin r = 1; for _ in 1:n; r *= x; end; r end
wl_tup(a, b) = begin t = (a + b, a - b); t[1] + t[2] end
wl_str(s, n) = s * "!" ^ n
wl_eh(x) = try; div(10, x); catch; -1; end
wl_ehfin(x) = begin local r = 0; try; r = div(10, x); finally; r += 1; end; r end
wl_undef(c) = begin local y; if c; y = 1; end; c ? y : 0 end
wl_fact(n) = n <= 1 ? 1 : n * wl_fact(n - 1)
wl_clo(r::Int) = begin (r < 0) && (r = -r); f = x -> x * r; f(3) end
wl_closet() = begin local x::Int = 0; inc = () -> (x = x + 1); inc(); inc(); x end
wl_loopclo(n) = begin fs = Any[]; local x::Int = 0
    for i in 1:n; push!(fs, () -> x); x = i; end
    s = 0; for f in fs; s += f()::Int; end; s end
wl_kw(x; a = 1, b = 2) = x + a + b
wl_kwcall(x) = wl_kw(x; b = 7)
wl_vararg(args...) = length(args)
wl_dict(n) = begin d = Dict{Int,Int}(); for i in 1:n; d[i] = i * i; end; length(d) end
wl_arr(n) = begin v = zeros(Int, n); for i in 1:n; v[i] = i; end; sum(v) end
wl_union(c) = begin x = c ? 1 : nothing; x === nothing ? 0 : x + 1 end

const BODIES = Any[
    (wl_sum, (25,)),
    (wl_branchy, (11,)), (wl_branchy, (5,)), (wl_branchy, (-1,)),
    (wl_pow, (2, 10)),
    (wl_tup, (3, 4)),
    (wl_str, ("hey", 3)),
    (wl_eh, (5,)), (wl_eh, (0,)),
    (wl_ehfin, (5,)),
    (wl_undef, (true,)),
    (wl_fact, (10,)),
    (wl_clo, (-3,)),
    (wl_closet, ()),
    (wl_loopclo, (4,)),
    (wl_kwcall, (1,)),
    (wl_vararg, (1, 2, 3)),
    (wl_dict, (8,)),
    (wl_arr, (8,)),
    (wl_union, (true,)), (wl_union, (false,)),
]

end # module PrecompileWorkload

let U = Unified, CC = Compiler, WL = PrecompileWorkload
    # ---- phase 1: the unified pipeline over the body set -------------------
    world = Base.get_world_counter()
    for (f, args) in WL.BODIES
        # the runtime-driver entry (what activate!/enable_pipeline! route to)
        mi = U.lookup_method_instance(f, args...)
        interp = CC.NativeInterpreter(world)
        U.unified_typeinf(interp, mi, CC.SOURCE_MODE_ABI)
        # the reflection hook entries (JULIA_UNIFIED_COMPILER=1 subprocesses)
        U.unified_typeinf_code(CC.NativeInterpreter(world), mi, true)
        tt = Tuple{Core.Typeof(f), map(Core.Typeof, args)...}
        U.unified_infer_effects(CC.NativeInterpreter(world), tt, false)
        U.unified_infer_exception_type(CC.NativeInterpreter(world), tt, false)
    end
    # the with_unified_compiler execution path (native entry + invoke)
    U.with_unified_compiler(WL.wl_sum, 25)
    U.with_unified_compiler(WL.wl_eh, 5)
    # the Queries/converter surface (unified suite + fresh corpus entries)
    U.typed_ir(WL.wl_sum, Any[Int])
    U.typed_ir(WL.wl_eh, Any[Int])
    U.typed_ir(WL.wl_loopclo, Any[Int])
    U.infer_return(WL.wl_branchy, Any[Int])
    U.effects_of(WL.wl_tup, Any[Int, Int])
    U.lowered_ir(WL.wl_pow, Tuple{Int, Int})
    U.roundtrip_codeinfo(WL.wl_sum, Tuple{Int})
    # the hooked stdlib-Compiler reflection entries, hooks ON (the exact
    # path a `JULIA_UNIFIED_COMPILER=1` test subprocess compiles first)
    U.enable_pipeline!()
    try
        for (f, args) in Any[(WL.wl_pow, (3, 5)), (WL.wl_ehfin, (2,))]
            mi = U.lookup_method_instance(f, args...)
            CC.typeinf_code(CC.NativeInterpreter(Base.get_world_counter()), mi, true)
        end
    finally
        U.disable_pipeline!()
    end

    # ---- phase 2: the stock bootstrap! inference sweep (no codegen flip) ---
    # `activate!` runs `Compiler.activate!(codegen = true)` → `bootstrap!()`,
    # whose typeinf_ext_toplevel sweep is the dominant per-process cost.
    # Precompiling the same sweep here parks those CodeInstances (owner ==
    # nothing, native code) in this pkgimage, so the in-session sweep is a
    # cache walk. Body replicated from Compiler/src/bootstrap.jl:21 minus
    # activate_codegen! (never flip jl_typeinf_func inside precompile).
    if Base.get(Base.ENV, "JULIA_UNIFIED_PKGIMAGE_SWEEP", "1") == "1"
        ssa_inlining_pass!_tt = Tuple{typeof(CC.ssa_inlining_pass!), CC.IRCode,
                                      CC.InliningState{CC.NativeInterpreter}, Bool}
        optimize_tt = Tuple{typeof(CC.optimize), CC.NativeInterpreter,
                            CC.OptimizationState{CC.NativeInterpreter}, CC.InferenceResult}
        typeinf_ext_tt = Tuple{typeof(CC.typeinf_ext), CC.NativeInterpreter,
                               Core.MethodInstance, UInt8}
        typeinf_tt = Tuple{typeof(CC.typeinf), CC.NativeInterpreter,
                           CC.InferenceState{CC.NativeInterpreter}}
        typeinf_edge_tt = Tuple{typeof(CC.typeinf_edge), CC.NativeInterpreter,
                                Method, Any, Core.SimpleVector,
                                CC.InferenceState{CC.NativeInterpreter}, Bool, Bool}
        fs = Any[
            CC.compact!, ssa_inlining_pass!_tt, optimize_tt,
            typeinf_ext_tt, typeinf_tt, typeinf_edge_tt,
        ]
        for x in CC.T_FFUNC_VAL
            push!(fs, x[3])
        end
        for i = 1:length(CC.T_IFUNC)
            isassigned(CC.T_IFUNC, i) && push!(fs, CC.T_IFUNC[i][3])
        end
        world = Base.get_world_counter()
        for f in fs
            if isa(f, DataType) && f.name === Base.typename(Tuple)
                tt = f
            else
                tt = Tuple{typeof(f), Vararg{Any}}
            end
            for m in CC._methods_by_ftype(tt, 10, world)::Vector
                m = m::Core.MethodMatch
                params = Any[m.spec_types.parameters...]
                for i = 1:length(params)
                    params[i] = CC.unwraptv(params[i])
                end
                mi = CC.specialize_method(m.method, Tuple{params...}, m.sparams)
                CC.typeinf_ext_toplevel(mi, world,
                    CC.isa_compileable_sig(mi) ? CC.SOURCE_MODE_ABI : CC.SOURCE_MODE_NOT_REQUIRED,
                    CC.TRIM_NO)
            end
        end
    end

    # ---- phase 3: nothing session-scoped survives into the image -----------
    _reset_session_state!()
end
