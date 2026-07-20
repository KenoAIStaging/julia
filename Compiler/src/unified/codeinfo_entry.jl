# Entry converter, CodeInfo → UnifiedIR (§10.5), cfg-wrap mode: the whole
# body becomes one `cfg` island; slots become cells. Always available for the
# supported feature matrix; structurization is a separate mode (P1).
#
# Feature matrix (v1): no exception handlers (EnterNode/:leave/:pop_exception),
# no PhiNode/PhiCNode/UpsilonNode (uninferred slot-form code has none).

struct UnsupportedIR <: Exception
    what::String
end
Base.showerror(io::IO, e::UnsupportedIR) = print(io, "UnsupportedIR: ", e.what)

"First-operand marker distinguishing an `Expr(:foreignglobal, name)` (the
cglobal lowering, rt Ptr{Cvoid}) encoded on the K\"foreigncall\" kind."
const FOREIGNGLOBAL_MARKER = Symbol("unified.foreignglobal")

"First-operand marker for a statement-position `Expr(:static_parameter, n)`
read carried as a K\"call\" (operand 2 = the sparam). The read of a
maybe-undefined parameter throws UndefVarError, so it must survive as a
statement (issue45490); the exits re-emit the raw form."
const SPARAM_READ_MARKER = Symbol("unified.sparam_read")

# ---------------------------------------------------------------------------
# Statement-level ssaflags carriage (A5/E3)
# ---------------------------------------------------------------------------
# Lowered sources carry per-statement context in `ssaflags`: IR_FLAG_INBOUNDS
# (`@inbounds`), IR_FLAG_INLINE/IR_FLAG_NOINLINE (callsite `@inline` etc.),
# and — shifted above stock's NUM_IR_FLAGS — the `@assume_effects` statement
# override payload (NUM_EFFECTS_OVERRIDES bits). The entry converters copy
# them onto every UnifiedIR statement emitted for the source statement (one
# lowered statement's cell reads/writes are part of its evaluation, exactly
# the span stock's per-statement flag covers); inference reads them back
# through the accessors below. The low FLAG_* effect bits stay the analysis
# channel and are recomputed each pass — the publish sites preserve
# `FLAGS_CARRIED`.

"Flag-column bit position of the carried `@assume_effects` override payload."
const STMT_OVERRIDE_SHIFT = 16
const STMT_OVERRIDE_MASK = (UInt32(1) << Compiler.NUM_EFFECTS_OVERRIDES) - UInt32(1)

"Flag-column bits owned by entry carriage (not analysis passes)."
const FLAGS_CARRIED = UnifiedIR.FLAG_INBOUNDS | UnifiedIR.FLAG_INLINE |
                      UnifiedIR.FLAG_NOINLINE | (STMT_OVERRIDE_MASK << STMT_OVERRIDE_SHIFT)

"Project one stock `ssaflags` word onto the carried UnifiedIR flag bits."
function carry_ssaflags(f::UInt32)
    out = zero(UInt32)
    (f & Compiler.IR_FLAG_INBOUNDS) != zero(UInt32) && (out |= UnifiedIR.FLAG_INBOUNDS)
    (f & Compiler.IR_FLAG_INLINE) != zero(UInt32) && (out |= UnifiedIR.FLAG_INLINE)
    (f & Compiler.IR_FLAG_NOINLINE) != zero(UInt32) && (out |= UnifiedIR.FLAG_NOINLINE)
    ov = (UInt32(f >> Compiler.NUM_IR_FLAGS)) & STMT_OVERRIDE_MASK
    return out | (ov << STMT_OVERRIDE_SHIFT)
end

"The raw carried `@assume_effects` override payload of a flag word."
stmt_override_bits(f::UInt32) = UInt16((f >> STMT_OVERRIDE_SHIFT) & STMT_OVERRIDE_MASK)

"The `@assume_effects` statement override carried on `s` (all-false when none)."
stmt_effects_override(ir::UnifiedIR.IR, s::StmtId) =
    Compiler.decode_effects_override(stmt_override_bits(UnifiedIR.stmt_flag(ir, s)))

"Was `s` inside an `@inbounds` in the entry source (stock IR_FLAG_INBOUNDS)?"
stmt_inbounds(ir::UnifiedIR.IR, s::StmtId) =
    (UnifiedIR.stmt_flag(ir, s) & UnifiedIR.FLAG_INBOUNDS) != zero(UInt32)

"""
    codeinfo_to_ir(ci::Core.CodeInfo; nargs, name=:f) -> UnifiedIR.IR

cfg-wrap conversion of slot-form lowered code. `nargs` counts the function
slots including slot 1 (`#self#`).
"""
function codeinfo_to_ir(ci::Core.CodeInfo; nargs::Int, name::Symbol = :f)
    code = ci.code
    # exception handlers take the scope-recovering path (eh_entry.jl)
    if any(st -> st isa Core.EnterNode || Meta.isexpr(st, :enter) ||
                 Meta.isexpr(st, :leave) || Meta.isexpr(st, :pop_exception) ||
                 Meta.isexpr(st, :the_exception), code)
        return codeinfo_to_ir_eh(ci; nargs, name)
    end
    n = length(code)
    nslots = length(ci.slotnames)

    # -- block structure ----------------------------------------------------
    isleader = falses(n + 1)
    isleader[1] = true
    for (i, st) in enumerate(code)
        if st isa Core.GotoNode
            isleader[st.label] = true
            i < n && (isleader[i + 1] = true)
        elseif st isa Core.GotoIfNot
            isleader[st.dest] = true
            i < n && (isleader[i + 1] = true)
        elseif st isa Core.ReturnNode
            i < n && (isleader[i + 1] = true)
        elseif st isa Core.EnterNode || Meta.isexpr(st, :enter) ||
               Meta.isexpr(st, :leave) || Meta.isexpr(st, :pop_exception)
            throw(UnsupportedIR("exception handler IR (EnterNode/:leave) — outside the cfg-wrap v1 feature matrix"))
        elseif st isa Core.PhiCNode || st isa Core.UpsilonNode
            # pre-SSA'd exceptional stores need the eh path's scope recovery
            throw(UnsupportedIR("$(typeof(st)) in slot-form input"))
        end
        # PhiNode input is accepted (F6): a block's leading φs become its
        # region args; the values travel on the in-edges (see blockphis)
    end
    leaders = [i for i in 1:n if isleader[i]]
    blockof = zeros(Int, n)                 # stmt -> block index
    for (bi, l) in enumerate(leaders)
        hi = bi < length(leaders) ? leaders[bi + 1] - 1 : n
        for i in l:hi
            blockof[i] = bi
        end
    end
    nblocks = length(leaders)

    # -- φ input (F6): a block's leading φs become its region args ----------
    # φ edges name the predecessor's terminator statement; the edge-emission
    # sites look the values up by predecessor block. φs outside a block's
    # leading run are malformed SSA; a φ in the entry block has no value for
    # the function-entry edge; an edgeless φ is undef and must be unused.
    blockphis = Dict{Int,Vector{Pair{Int,Core.PhiNode}}}()
    for (i, st) in enumerate(code)
        st isa Core.PhiNode || continue
        bi = blockof[i]
        i == leaders[bi] || code[i - 1] isa Core.PhiNode ||
            throw(UnsupportedIR("PhiNode outside its block's leading positions"))
        isempty(st.edges) && continue           # undef φ: poisoned at the walk
        bi == 1 && throw(UnsupportedIR("PhiNode in the entry block"))
        push!(get!(() -> Pair{Int,Core.PhiNode}[], blockphis, bi), i => st)
    end

    # -- builder ------------------------------------------------------------
    b = UnifiedIR.Builder(; name)
    argmap = Vector{StmtId}(undef, nargs)
    for i in 1:nargs
        t = ci.slottypes === nothing ? Any : something(ci.slottypes[i], Any)
        argmap[i] = append_stmt!(b, K"region_arg"; type = t isa Type ? t : Any)
        push!(b.ir.argtypes, Any)
    end
    # cells for non-argument slots; their variable names travel in
    # meta[:cell_names] (optimize_ir! remaps it across compact!) so
    # synthesized undef guards can name the variable like stock does
    cellmap = Dict{Int,StmtId}()
    cellnames = Dict{Int32,Symbol}()
    b.ir.meta[:cell_names] = cellnames
    for sl in (nargs+1):nslots
        c = append_stmt!(b, K"cell", Any; type = Any)
        cellmap[sl] = c
        cellnames[c.id] = ci.slotnames[sl]
    end

    single = nblocks == 1 && !any(st -> st isa Core.GotoNode || st isa Core.GotoIfNot, code)

    cfgop = NULL_STMT
    blockregions = RegionId[]
    if !single
        cfgop = append_stmt!(b, K"cfg"; type = Any)
    end

    ssamap = Vector{Any}(undef, n)          # CodeInfo ssa idx -> Operand
    debugtriple(i) = (Int32(0), Int32(0), Int32(0))

    function convert_value(@nospecialize(v))::UnifiedIR.Operand
        if v isa Core.SSAValue
            isassigned(ssamap, v.id) || throw(UnsupportedIR("forward SSA reference"))
            o = ssamap[v.id]
            o === :undef_phi && throw(UnsupportedIR("use of an edgeless φ (undef value)"))
            o isa UnifiedIR.Operand || throw(UnsupportedIR("forward SSA reference"))
            return o
        elseif v isa Core.SlotNumber
            if v.id <= nargs
                return UnifiedIR.op_stmt(argmap[v.id])
            else
                g = append_stmt!(b, K"cell_get", UnifiedIR.op_stmt(cellmap[v.id]); type = Any)
                return UnifiedIR.op_stmt(g)
            end
        elseif v isa Core.Argument
            return UnifiedIR.op_stmt(argmap[v.n])
        elseif v isa GlobalRef
            return UnifiedIR.vop(b.ir, v)
        elseif v isa QuoteNode
            return UnifiedIR.vop(b.ir, v.value)
        elseif v isa Expr
            v.head === :static_parameter && return UnifiedIR.op_sparam(v.args[1]::Int)
            throw(UnsupportedIR("nested Expr operand $(v.head)"))
        else
            return UnifiedIR.vop(b.ir, v)
        end
    end

    # the values a `frombi → tobi` edge carries for `tobi`'s φ block args
    # (in φ-edge order; region-IR edge args are parallel by construction). A
    # φ lacking an assigned value on a taken edge is the undef-φ class —
    # region IR carried args are total by construction (B3A map row 2).
    function edge_args(frombi::Int, tobi::Int)
        phis = get(blockphis, tobi, nothing)
        phis === nothing && return UnifiedIR.Operand[]
        ops = UnifiedIR.Operand[]
        for (_, phi) in phis
            ki = 0
            for (k2, e) in enumerate(phi.edges)
                (1 <= Int(e) <= n && blockof[Int(e)] == frombi) || continue
                ki = k2
                break
            end
            (ki != 0 && isassigned(phi.values, ki)) ||
                throw(UnsupportedIR("undef φ edge (region IR carried args are total)"))
            push!(ops, convert_value(phi.values[ki]))
        end
        return ops
    end

    returns = 0
    # per-statement ssaflags carriage: everything emitted for source
    # statement `i` gets its carried flag bits (see carry_ssaflags)
    ssaflags = ci.ssaflags
    function convert_stmt_carry!(i::Int, st)
        from = Int(b.ir.body.len) + 1
        convert_stmt!(i, st)
        carry = i <= length(ssaflags) ? carry_ssaflags(ssaflags[i]) : zero(UInt32)
        carry == zero(UInt32) && return
        for j in from:Int(b.ir.body.len)
            UnifiedIR.add_flag!(b.ir, StmtId(Int32(j)), carry)
        end
        return
    end
    function convert_stmt!(i::Int, st)
        if st isa Core.ReturnNode
            isdefined(st, :val) || begin
                append_stmt!(b, K"unreachable")
                return
            end
            v = convert_value(st.val)
            if single
                append_stmt!(b, K"return", v)
            else
                append_stmt!(b, K"result", v)
            end
            returns += 1
        elseif st isa Core.GotoNode
            dbi = blockof[st.label]
            dargs = edge_args(blockof[i], dbi)
            append_stmt!(b, K"goto", UnifiedIR.op_block(blockregions[dbi]),
                         UnifiedIR.op_inline(length(dargs)), dargs...)
        elseif st isa Core.GotoIfNot
            cond = convert_value(st.cond)
            fbi = blockof[i] + 1
            dbi = blockof[st.dest]
            fargs = edge_args(blockof[i], fbi)
            dargs = edge_args(blockof[i], dbi)
            append_stmt!(b, K"br_if", cond,
                         UnifiedIR.op_block(blockregions[fbi]),
                         UnifiedIR.op_inline(length(fargs)), fargs...,
                         UnifiedIR.op_block(blockregions[dbi]),
                         UnifiedIR.op_inline(length(dargs)), dargs...)
        elseif st isa Core.NewvarNode
            sl = st.slot.id
            haskey(cellmap, sl) && append_stmt!(b, K"cell_new", UnifiedIR.op_stmt(cellmap[sl]))
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)
        elseif st isa Expr
            convert_expr!(i, st)
        elseif st isa GlobalRef
            if isconst(st.mod, st.name) && isdefined(st.mod, st.name)
                # a defined-const binding is legal (and stable) in value
                # position — dissolve into an operand like any literal
                ssamap[i] = convert_value(st)
            else
                # otherwise the statement-position read must SURVIVE: both
                # the load's ordering (its UndefVarError/world-sensitivity
                # point) and stock IRCode canonicality (unbound/partitioned
                # GlobalRefs are not allowed in value position) depend on
                # the placement (F3)
                s = append_stmt!(b, K"globalref", UnifiedIR.vop(b.ir, st); type = Any)
                ssamap[i] = UnifiedIR.op_stmt(s)
            end
        elseif st isa Core.SlotNumber || st isa Core.SSAValue ||
               st isa QuoteNode || !(st isa Union{Core.GotoNode,Core.GotoIfNot})
            # bare value statement: its SSA value is the value itself
            ssamap[i] = convert_value(st)
        end
        return
    end

    function convert_expr!(i::Int, st::Expr)
        h = st.head
        if h === :(=)
            lhs = st.args[1]
            rhs = st.args[2]
            rhsop = if rhs isa Expr
                convert_expr!(i, rhs)
                ssamap[i]
            else
                convert_value(rhs)
            end
            lhs isa Core.SlotNumber || throw(UnsupportedIR("assignment to $(typeof(lhs))"))
            if lhs.id <= nargs
                throw(UnsupportedIR("assignment to argument slot"))
            end
            append_stmt!(b, K"cell_set", UnifiedIR.op_stmt(cellmap[lhs.id]), rhsop)
            ssamap[i] = rhsop
        elseif h === :call
            ops = UnifiedIR.Operand[convert_value(a) for a in st.args]
            s = append_stmt!(b, K"call", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :invoke
            tgt = st.args[1]
            tgt isa Union{Core.MethodInstance,Core.CodeInstance} ||
                throw(UnsupportedIR("invoke with non-instance target"))
            ops = UnifiedIR.Operand[UnifiedIR.vop(b.ir, tgt)]
            for a in st.args[2:end]
                push!(ops, convert_value(a))
            end
            s = append_stmt!(b, K"invoke", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :new
            ops = UnifiedIR.Operand[convert_value(a) for a in st.args]
            s = append_stmt!(b, K"new", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :splatnew
            ops = UnifiedIR.Operand[convert_value(a) for a in st.args]
            s = append_stmt!(b, K"splatnew", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :isdefined
            a = st.args[1]
            if a isa Core.SlotNumber && a.id > nargs
                s = append_stmt!(b, K"cell_isdefined", UnifiedIR.op_stmt(cellmap[a.id]); type = Bool)
                ssamap[i] = UnifiedIR.op_stmt(s)
            elseif a isa GlobalRef
                s = append_stmt!(b, K"isdefined_global", UnifiedIR.vop(b.ir, a); type = Bool)
                ssamap[i] = UnifiedIR.op_stmt(s)
            else
                ssamap[i] = UnifiedIR.op_inline(true)
            end
        elseif h === :throw_undef_if_not
            name_, cond = st.args
            append_stmt!(b, K"throw_undef_if_not", convert_value(cond),
                         UnifiedIR.vop(b.ir, name_ isa Symbol ? name_ : Symbol(name_)))
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)
        elseif h === :boundscheck
            s = append_stmt!(b, K"boundscheck"; type = Bool)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :static_parameter
            # a real statement (marker call), not an operand alias: the read
            # of a maybe-undefined parameter throws UndefVarError at exactly
            # this position and must not disappear (issue45490)
            s = append_stmt!(b, K"call",
                             UnifiedIR.vop(b.ir, SPARAM_READ_MARKER),
                             UnifiedIR.op_sparam(st.args[1]::Int); type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :meta || h === :inbounds || h === :loopinfo || h === :aliasscope ||
               h === :popaliasscope || h === :inline || h === :noinline || h === :purity
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)  # carried as flags/columns later
        elseif h === :code_coverage_effect
            append_stmt!(b, K"coverage_effect")
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)
        elseif h === :gc_preserve_begin
            ops = UnifiedIR.Operand[convert_value(a) for a in st.args]
            s = append_stmt!(b, K"gc_preserve_begin", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :gc_preserve_end
            append_stmt!(b, K"gc_preserve_end", convert_value(st.args[1]))
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)
        elseif h === :latestworld
            append_stmt!(b, K"latestworld")
            ssamap[i] = UnifiedIR.vop(b.ir, nothing)
        elseif h === :foreigncall || h === :cfunction || h === :foreignglobal
            # operands: all pieces; non-value pieces interned as constants.
            # :foreignglobal (the cglobal lowering; rt Ptr{Cvoid}) rides the
            # foreigncall kind behind a marker first operand — the transfer
            # and the exit converter both recognize FOREIGNGLOBAL_MARKER.
            ops = UnifiedIR.Operand[]
            h === :foreignglobal &&
                push!(ops, UnifiedIR.vop(b.ir, FOREIGNGLOBAL_MARKER))
            for a in st.args
                push!(ops, a isa Union{Core.SSAValue,Core.SlotNumber,Core.Argument} ?
                      convert_value(a) : UnifiedIR.vop(b.ir, a))
            end
            s = append_stmt!(b, h === :cfunction ? K"cfunction" : K"foreigncall",
                             ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :new_opaque_closure
            # value pieces (captures, computed types) converted; structural
            # pieces (the :opaque_closure_method Expr, literals) interned raw
            # — the exit converter re-emits them via raw_structural. The
            # transfer models the statement as (Any, EFFECTS_UNKNOWN); the
            # closure BODY compiles through the runtime on first call.
            ops = UnifiedIR.Operand[]
            for a in st.args
                push!(ops, a isa Union{Core.SSAValue,Core.SlotNumber,Core.Argument} ?
                      convert_value(a) : UnifiedIR.vop(b.ir, a))
            end
            s = append_stmt!(b, K"new_opaque_closure", ops...; type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :the_exception || h === :enter || h === :leave || h === :pop_exception
            throw(UnsupportedIR("exception IR ($h) — outside the v1 feature matrix"))
        elseif h === :method
            throw(UnsupportedIR("nested :method definition"))
        elseif h === :copyast
            s = append_stmt!(b, K"copyast", convert_value(st.args[1]); type = Any)
            ssamap[i] = UnifiedIR.op_stmt(s)
        elseif h === :globaldecl || h === :const
            throw(UnsupportedIR("toplevel form :$h"))
        else
            throw(UnsupportedIR("Expr head :$h"))
        end
        return
    end

    if single
        for (i, st) in enumerate(code)
            if st isa Core.PhiNode
                # no branches exist, so a φ can carry no edge value: undef
                isempty(st.edges) ||
                    throw(UnsupportedIR("PhiNode with edges in a single-block body"))
                ssamap[i] = :undef_phi
                continue
            end
            convert_stmt_carry!(i, st)
        end
    else
        # pre-create block regions so edges can reference them
        for bi in 1:nblocks
            r = UnifiedIR.Region(UnifiedIR.REGION_BLOCK, cfgop, UnifiedIR.stmt_region(b.ir, cfgop))
            push!(b.ir.regions, r)
            push!(blockregions, RegionId(length(b.ir.regions)))
        end
        for bi in 1:nblocks
            rid = blockregions[bi]
            reg = UnifiedIR.getregion(b.ir, rid)
            reg.first = StmtId(Int(b.ir.body.len) + 1)
            push!(b.open, rid)
            lo = leaders[bi]
            hi = bi < nblocks ? leaders[bi + 1] - 1 : n
            # the block's φs become its region args (F6); edgeless φs stay
            # undef-poisoned (legal only while unused)
            for (j, phi) in get(() -> Pair{Int,Core.PhiNode}[], blockphis, bi)
                a = append_stmt!(b, K"region_arg"; type = Any)
                ssamap[j] = UnifiedIR.op_stmt(a)
            end
            for i in lo:hi
                st = code[i]
                if st isa Core.PhiNode
                    isempty(st.edges) && (ssamap[i] = :undef_phi)
                    continue                     # bound above (or poisoned)
                end
                convert_stmt_carry!(i, st)
            end
            # implicit fallthrough becomes explicit goto (§5.5)
            lastst = code[hi]
            if !(lastst isa Core.GotoNode || lastst isa Core.GotoIfNot ||
                 lastst isa Core.ReturnNode)
                bi == nblocks && throw(UnsupportedIR("function falls off the end"))
                fargs = edge_args(bi, bi + 1)
                append_stmt!(b, K"goto", UnifiedIR.op_block(blockregions[bi + 1]),
                             UnifiedIR.op_inline(length(fargs)), fargs...)
            end
            reg.last = StmtId(Int(b.ir.body.len))
            pop!(b.open)
        end
        r = append_stmt!(b, K"return", UnifiedIR.op_stmt(cfgop))
    end

    ir = UnifiedIR.finish!(b; verify = false)
    UnifiedIR.verify_ir(ir; level = 0)
    return ir
end
