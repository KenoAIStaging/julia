# This file is a part of Julia. License is MIT: https://julialang.org/license

using InteractiveUtils: @activate

if Base.identify_package("Compiler") !== nothing && !isdefined(Main, :__custom_compiler_active)
    Base.eval(Main, :(__custom_compiler_active=true))
    @activate Compiler
end

if !@isdefined(Compiler)
    if Base.REFLECTION_COMPILER[] === nothing
        using Base.Compiler: Compiler
    else
        const Compiler = Base.REFLECTION_COMPILER[]
    end
end

# COMPILER-PORT-PLAN Phase B0: `JULIA_UNIFIED_COMPILER=1` additionally routes the
# activated stdlib Compiler through the unified pipeline (per-body stock fallback,
# `Compiler.Unified.pipeline_stats()` is the ledger). Default (env unset): stock.
if get(ENV, "JULIA_UNIFIED_COMPILER", "0") == "1" &&
        Base.REFLECTION_COMPILER[] !== nothing &&
        !isdefined(Main, :__unified_pipeline_active)
    Base.eval(Main, :(__unified_pipeline_active=true))
    let U = Base.REFLECTION_COMPILER[].load_unified!()
        # enable_pipeline! was defined within this top-level expression (newer world)
        Base.invokelatest(Base.invokelatest(Core.getglobal, U, :enable_pipeline!))
    end
end
