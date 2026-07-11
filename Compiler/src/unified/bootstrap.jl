# This file is a part of Julia. License is MIT: https://julialang.org/license

# Bake the UnifiedIR-native compiler port into the sysimage as
# `Base.UnifiedCompiler`: evaluated via `Core.include(Base, ...)` by the
# sys-unified sysimage stage (sysimage.mk), the bootstrap sibling of the
# package-mode `Compiler.Unified` (Compiler/src/unified/Unified.jl) and of
# the UnifiedCompiler pkgimage (UnifiedCompiler/src/UnifiedCompiler.jl).
# The include list below mirrors Unified.jl and must be kept in sync.
#
# Under Base this is a normal nested module — the package-mode carrier
# workarounds do not apply — and the port references the baked
# `Base.Compiler` and `Base.UnifiedIR` directly (the same instances the
# stage's runtime uses, so `enable_pipeline!` reaches the
# `Compiler.UNIFIED_HOOKS` that the already-active `jl_typeinf_func` —
# `Base.Compiler.typeinf_ext_toplevel` — consults).
module UnifiedCompiler

const Compiler = Base.Compiler
const UnifiedIR = Base.UnifiedIR
using .UnifiedIR
using .UnifiedIR: StmtId, RegionId, NULL_STMT, NULL_REGION, @K_str

# `Core.include` does not track a source path, so relative includes would
# resolve against the cwd; locate the port sources explicitly (the driver
# records the dir; the fallback is the installed stdlib path).
const _SRCDIR = isdefined(Base, :_UNIFIED_BOOT_DIR) ? Base._UNIFIED_BOOT_DIR::String :
                joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "Compiler", "src", "unified")

export codeinfo_to_ir, ir_to_codeinfo, UnsupportedIR,
    lowered_ir, define_ir_method!, roundtrip_codeinfo,
    infer_ir!, UInferConfig, optimize_ir!,
    infer_return, typed_ir, effects_of, InferenceConfig,
    typed_region_ir!,
    with_unified_compiler, UnifiedCacheOwner, @code_unified,
    unified_typeinf, enable_pipeline!, disable_pipeline!,
    pipeline_stats, reset_pipeline_stats!

include(joinpath(_SRCDIR, "codeinfo_entry.jl"))
include(joinpath(_SRCDIR, "eh_entry.jl"))
include(joinpath(_SRCDIR, "exit_lowered.jl"))
include(joinpath(_SRCDIR, "exit_typed.jl"))
include(joinpath(_SRCDIR, "methods.jl"))
include(joinpath(_SRCDIR, "uinference.jl"))
include(joinpath(_SRCDIR, "transfers.jl"))
include(joinpath(_SRCDIR, "sroa.jl"))
include(joinpath(_SRCDIR, "adce.jl"))
include(joinpath(_SRCDIR, "structurize.jl"))
include(joinpath(_SRCDIR, "inline2.jl"))
include(joinpath(_SRCDIR, "optimize.jl"))
include(joinpath(_SRCDIR, "escape.jl"))
include(joinpath(_SRCDIR, "completeness.jl"))
include(joinpath(_SRCDIR, "queries.jl"))
include(joinpath(_SRCDIR, "late.jl"))
include(joinpath(_SRCDIR, "driver.jl"))
include(joinpath(_SRCDIR, "activate.jl"))

# Runs at every boot of an image this module is baked into (sysimage module
# initializers): reset the per-session driver state the dumping process
# could not clear after its own output phase — most importantly the
# reentrant valve's admission counter, which the output phase re-consumes
# after the stage script's pre-exit reset (a baked exhausted counter would
# make the booted image decline every reentrant unified pass forever).
# Baked memo entries stay: they revalidate per (world, counter) stamps.
function __init__()
    REENTRANT_ADMITTED[] = 0
    MEMO_OWNER[] = nothing
    STATS_OWNER[] = nothing
    STATS_DROPPED[] = 0
    OPT_NEST_DEPTH[] = 0; OPT_WORK_LEFT[] = 0
    TOWER_FRAME_CAP[] = 0
    return nothing
end

end # module UnifiedCompiler
