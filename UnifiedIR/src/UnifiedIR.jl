"""
    UnifiedIR

One IR data structure for the Julia compiler and external compilers: a flat
statement table with hybrid regions, layout states (dense / editable /
floating), a namespaced kind registry, extension-column universes, and
exactly two renaming points (`compact!`, `schedule!`).

Zero dependencies; testable without Julia semantics via the `test` dialect
and the textual format. See `unifiedir-design.md` for the specification.
"""
module UnifiedIR

export Kind, StmtId, RegionId, Value, Operand, IR, Builder, RemapSet,
    # kinds
    @K_str, register_dialect!, register_kind!, kindname, kindinfo,
    is_terminator, owns_regions, result_arity,
    OC_VALUE, OC_STMT, OC_REGION, OC_BLOCK, OC_CONST, OC_IMM, OC_ANY,
    FLAG_CONSISTENT, FLAG_EFFECT_FREE, FLAG_NOTHROW, FLAG_TERMINATES,
    FLAG_REMOVABLE, FLAG_PURE, FLAG_INLINE, FLAG_NOINLINE,
    CLOSURE_FLAG_ISVA,
    # operands
    op_stmt, op_block, op_region, op_inline, vop, optag, op_value,
    # regions
    Region, RegionKind, Activation,
    REGION_BODY, REGION_ARM, REGION_GUARD, REGION_LOOP_BODY, REGION_HANDLER, REGION_BLOCK,
    ACT_IMMEDIATE, ACT_DEFERRED, ACT_RESUME,
    # builder
    append_stmt!, open_region!, open_guard_region!, close_region!, finish!,
    build_if!, build_loop!,
    # core API
    layout, generation, nstmts, nregions, getregion, root_region,
    stmt_kind, stmt_type, stmt_flag, stmt_region, set_type!, set_flag!, add_flag!,
    nops, getop, setop!, operands, comes_before, visible,
    each_stmt, region_stmts, region_terminator,
    # dense mutation
    replace_stmt!, delete_stmt!, replace_uses!, flush_renames!,
    # editable
    editable, insert_before!, insert_after!, push_stmt!,
    wrap_in_if!, wrap_in_loop!, inline_region!, splice_body!, compact!,
    # floating
    float!, schedule!, CausalityError,
    # columns
    DenseCol, SparseCol, DictColumns, ProvenanceCol, Semantic, Annotation, Derived,
    hasrefs, remap_refs!, semclass, convert_universe,
    # AttrGraph substrate + generic tree porcelain (§3.7 Level 1)
    AttrGraph, compact_graph!, collect_syntax!, Tree, NodeList,
    # verification / analyses
    verify_ir, VerifyError, use_counts, AnalysisCache, closure_environment,
    # passes
    dce!, promote_cells!, fold_constant_branches!,
    # cell promotion (the mem2reg suite, promote.jl). The individual join
    # passes stay unexported (qualified access) so providers may bind
    # same-named lattice-aware wrappers; the driver is the public entry.
    promote_fixpoint!,
    # text
    print_ir, parse_ir, struct_eq, display_maxlines!,
    # test dialect interpreter
    interpret, UClosure

# During the Base bootstrap, includes resolve against the build CWD, not
# this file; route the module's own includes through the DATAROOT path scheme
# Base_compiler.jl itself uses (the Compiler.jl bootstrap-include pattern).
const _BOOTSTRAPPING = !Base.isdefined(Base, :end_base_include)
_include_src(x::String) =
    Base.include(@__MODULE__, _BOOTSTRAPPING ?
        Base.strcat(Base.strcat(Base.DATAROOT, "julia/UnifiedIR/src/"), x) : x)

# The compiler-needed core, in the bootstrap dialect (see compat.jl): these
# files load during the basecompiler bootstrap stage, under the partial Base
# of COMPILER_SRCS.
_include_src("compat.jl")
_include_src("kinds.jl")
_include_src("operands.jl")
_include_src("columns.jl")
_include_src("attrgraph.jl")
_include_src("core.jl")
_include_src("refs.jl")
_include_src("builder.jl")
_include_src("stmts.jl")
_include_src("verify.jl")
_include_src("dense.jl")
_include_src("editable.jl")
_include_src("surgery.jl")
_include_src("compact.jl")
_include_src("floating.jl")
_include_src("analysis.jl")
_include_src("passes.jl")
_include_src("promote.jl")

# The debug/syntax layer needs the full Base vocabulary (IO, `view`, sort
# keywords, runtime string interpolation). During the Base bootstrap only the
# core above loads at the basecompiler stage; base/Base.jl finishes the
# module with `load_syntax!()` just before JuliaSyntax bootstraps on the
# substrate (the Compiler `load_irshow!` staging pattern). Every other load
# context (the LOAD_PATH package instance) loads eagerly below.
const _SYNTAX_SRCS = ("tree.jl", "testdialect.jl", "print.jl", "parse.jl",
                      "interp.jl")
const _syntax_loaded = Base.RefValue(false)

function load_syntax!()
    _syntax_loaded[] && return nothing
    _syntax_loaded[] = true
    for f in _SYNTAX_SRCS
        _include_src(f)
    end
    # upgrade the kind-registry lock shim now that ReentrantLock exists
    # (post-Base, external dialects may register at runtime from any thread)
    REGISTRY.lock isa Base.ReentrantLock || (REGISTRY.lock = Base.ReentrantLock())
    return nothing
end

if !(parentmodule(@__MODULE__) === Base && !Base.isdefined(Base, :end_base_include))
    load_syntax!()
end

function __init__()
    # Session-local kind numbering: re-register the test dialect on load.
    # The Base-baked instance (bootstrap substrate for JuliaSyntax) keeps its
    # registry clean of the test dialect; the loadable package registers it
    # (its own test suite and the textual-format tests use it).
    if parentmodule(@__MODULE__) !== Base
        register_test_dialect!()
    end
end

end # module UnifiedIR
