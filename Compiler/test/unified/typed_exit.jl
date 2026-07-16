# Typed exit converter (§10.5): UnifiedIR → IRCode with phi synthesis and
# exception SSA (EnterNode/:leave/:pop_exception, PhiC/Upsilon — A4).
# Acceptance: the STOCK IR verifier passes, and Core.OpaqueClosure execution
# matches the UnifiedIR interpreter on the same inputs — including the
# throwing paths (outcomes compare thrown errors too).

using UnifiedIR: op_stmt, op_inline, op_region

function oc_vs_interp(ir, inputs...)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    Compiler.verify_ir(irc)
    oc = Core.OpaqueClosure(irc)
    for inp in inputs
        want = UnifiedIR.interpret(ir, nothing, inp...)
        got = oc(inp...)
        isequal(got, want) || return false
    end
    return true
end

"(:ok, value) | (:err, message) — interpreter statement ids stripped, as in
the CellFuzz differential."
function tx_outcome(f, args...)
    try
        (:ok, f(args...))
    catch e
        (:err, replace(sprint(showerror, e), r"( \(%\d+\)| at %\d+)" => ""))
    end
end

"Exit `ir`, verify with the STOCK verifier, and differential-execute the
OpaqueClosure against the UnifiedIR interpreter over `inputs` (self arg
passed as `nothing` to the interpreter), including thrown-error outcomes.
Returns the produced IRCode for shape assertions."
function exit_differential(ir, inputs...)
    irc = UnifiedCompiler.ir_to_ircode(ir)
    Compiler.verify_ir(irc)
    oc = Core.OpaqueClosure(irc)
    for inp in inputs
        want = tx_outcome((a...) -> UnifiedIR.interpret(ir, nothing, a...), inp...)
        got = tx_outcome(oc, inp...)
        if !isequal(want, got)
            @error "typed-exit differential" inp want got
            error("typed exit differential mismatch")
        end
    end
    return irc
end

count_nodes(irc, T::Type) = count(s -> s isa T, irc.stmts.stmt)
count_exprs(irc, h::Symbol) = count(s -> Meta.isexpr(s, h), irc.stmts.stmt)

@testset "typed exit: if" begin
    b = Builder(name = :tif)
    append_stmt!(b, K"region_arg"; type = Any)
    a = append_stmt!(b, K"region_arg"; type = Int64)
    c = append_stmt!(b, K"call", GlobalRef(Base, :slt_int), 0, a; type = Bool)
    z = build_if!(b, c; type = Int64) do b
        append_stmt!(b, K"result", 1)
    end
    UnifiedIR.open_region!(b, z)
    y = append_stmt!(b, K"call", GlobalRef(Base, :mul_int), a, a; type = Int64)
    append_stmt!(b, K"result", y)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", z)
    ir = finish!(b)
    @test oc_vs_interp(ir, (5,), (-3,), (0,))
end

@testset "typed exit: loop with carried args + de-tupled extracts" begin
    b = Builder(name = :tloop)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Int64)
    r = build_loop!(b, 0, 1; type = Tuple{Int64,Int64}, argtypes = Any[Int64, Int64]) do b, args
        s, j = args
        s2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), s, j; type = Int64)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), j, 1; type = Int64)
        cnd = append_stmt!(b, K"call", GlobalRef(Base, :sle_int), j2, n; type = Bool)
        body = UnifiedIR.current_region(b)
        append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(s2), op_stmt(j2))
    end
    tot = append_stmt!(b, K"extract", op_stmt(r), op_inline(1); type = Int64)
    lst = append_stmt!(b, K"extract", op_stmt(r), op_inline(2); type = Int64)
    fin = append_stmt!(b, K"call", GlobalRef(Base, :add_int), tot, lst; type = Int64)
    append_stmt!(b, K"return", fin)
    ir = finish!(b)
    @test oc_vs_interp(ir, (10,), (1,), (3,))
end

@testset "typed exit: break with value out of loop + nested if" begin
    # find first j with j*j > n, else n itself
    b = Builder(name = :tbreak)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Int64)
    r = build_loop!(b, 1; type = Int64, argtypes = Any[Int64]) do b, args
        j, = args
        sq = append_stmt!(b, K"call", GlobalRef(Base, :mul_int), j, j; type = Int64)
        c = append_stmt!(b, K"call", GlobalRef(Base, :slt_int), n, sq; type = Bool)
        body = UnifiedIR.current_region(b)
        fi = append_stmt!(b, K"if", c; type = Nothing)
        UnifiedIR.open_region!(b, fi)
        append_stmt!(b, K"break", op_region(body), op_stmt(j))
        UnifiedIR.close_region!(b)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), j, 1; type = Int64)
        cnd = append_stmt!(b, K"call", GlobalRef(Base, :sle_int), j2, n; type = Bool)
        append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(j2))
    end
    append_stmt!(b, K"return", r)
    ir = finish!(b)
    @test oc_vs_interp(ir, (10,), (2,), (100,))
end

@testset "typed exit: escaping tuple result materializes" begin
    # loop result used as a first-class tuple (not extract-only)
    b = Builder(name = :ttup)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Int64)
    r = build_loop!(b, 0, 1; type = Tuple{Int64,Int64}, argtypes = Any[Int64, Int64]) do b, args
        s, j = args
        s2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), s, j; type = Int64)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), j, 1; type = Int64)
        cnd = append_stmt!(b, K"call", GlobalRef(Base, :sle_int), j2, n; type = Bool)
        body = UnifiedIR.current_region(b)
        append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(s2), op_stmt(j2))
    end
    append_stmt!(b, K"return", r)      # whole tuple escapes
    ir = finish!(b)
    @test oc_vs_interp(ir, (10,), (1,))
end

@testset "typed exit: multi-carried loop whose only exit is a break" begin
    # regression: a `continue` with literal-true condition never exits the
    # loop, so it must not count toward the exit-value arity — a 2-carried
    # loop whose only exit is a 1-value break used to materialize a spurious
    # 1-tuple around the scalar result (returned as junk)
    b = Builder(name = :tbreakonly)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Int64)
    r = build_loop!(b, 0, 1; type = Int64, argtypes = Any[Int64, Int64]) do b, args
        s, j = args
        body = UnifiedIR.current_region(b)
        s2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), s, j; type = Int64)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :add_int), j, 1; type = Int64)
        c = append_stmt!(b, K"call", GlobalRef(Base, :slt_int), n, j2; type = Bool)
        fi = append_stmt!(b, K"if", c; type = Nothing)
        UnifiedIR.open_region!(b, fi)
        append_stmt!(b, K"break", op_region(body), op_stmt(s2))
        UnifiedIR.close_region!(b)
        append_stmt!(b, K"continue", op_region(body), true, op_stmt(s2), op_stmt(j2))
    end
    append_stmt!(b, K"return", r)
    ir = finish!(b)
    @test oc_vs_interp(ir, (10,), (1,), (0,))
end

@testset "typed exit: cfg island with block args" begin
    src = """
    func @absmax(%1::Any, %2::Int64, %3::Int64) -> Int64 {
      %4 = cfg (%2) {
      ^bb2(%5::Int64):
        %6 = call global Base.slt_int, %5, %3 :: Bool
        br_if %6 (^bb3: %3) (^bb4: %5)
      ^bb3(%8::Int64):
        result %8
      ^bb4(%10::Int64):
        result %10
      } :: Int64
      return %4
    }
    """
    ir = parse_ir(src)
    @test UnifiedIR.verify_ir(ir; level = 1)
    @test oc_vs_interp(ir, (3, 7), (9, 2))
end

@testset "typed exit: residual cells convert; precise decline classes" begin
    # a plain frame cell goes through the boundary mem2reg (A4)
    b = Builder(name = :tcell)
    append_stmt!(b, K"region_arg"; type = Any)
    a = append_stmt!(b, K"region_arg"; type = Int64)
    cell = append_stmt!(b, K"cell", Int64; type = Any)
    append_stmt!(b, K"cell_set", cell, a)
    g = append_stmt!(b, K"cell_get", op_stmt(cell); type = Int64)
    append_stmt!(b, K"return", g)
    ir = finish!(b)
    irc = exit_differential(ir, (5,), (-3,))
    @test count_nodes(irc, Core.PhiCNode) == 0     # no handler: plain rename

    # cell_shared (box_capture) declines precisely
    b = Builder(name = :tshared)
    append_stmt!(b, K"region_arg"; type = Any)
    a = append_stmt!(b, K"region_arg"; type = Int64)
    cell = append_stmt!(b, K"cell_shared", Int64; type = Any)
    append_stmt!(b, K"cell_set", cell, a)
    g = append_stmt!(b, K"cell_get", op_stmt(cell); type = Int64)
    append_stmt!(b, K"return", g)
    ir = finish!(b)
    err = try; UnifiedCompiler.ir_to_ircode(ir); nothing; catch e; e; end
    @test err isa UnifiedCompiler.UnsupportedIR && occursin("box_capture", err.what)

    # an escaping cell (value use beyond the cell ops) declines precisely
    b = Builder(name = :tescape)
    append_stmt!(b, K"region_arg"; type = Any)
    append_stmt!(b, K"region_arg"; type = Int64)
    cell = append_stmt!(b, K"cell", Int64; type = Any)
    append_stmt!(b, K"cell_set", cell, 1)
    v = append_stmt!(b, K"call", GlobalRef(Core, :tuple), cell; type = Any)
    append_stmt!(b, K"return", v)
    ir = finish!(b)
    err = try; UnifiedCompiler.ir_to_ircode(ir); nothing; catch e; e; end
    @test err isa UnifiedCompiler.UnsupportedIR && occursin("escape", err.what)

    # gc-token cells decline precisely (the pairing verifier tracks values)
    b = Builder(name = :tgctok)
    append_stmt!(b, K"region_arg"; type = Any)
    a = append_stmt!(b, K"region_arg"; type = Any)
    cell = append_stmt!(b, K"cell", Any; type = Any)
    tok = append_stmt!(b, K"gc_preserve_begin", a; type = Any)
    append_stmt!(b, K"cell_set", cell, tok)
    g = append_stmt!(b, K"cell_get", op_stmt(cell); type = Any)
    append_stmt!(b, K"gc_preserve_end", g)
    append_stmt!(b, K"return", a)
    ir = finish!(b)
    err = try; UnifiedCompiler.ir_to_ircode(ir); nothing; catch e; e; end
    @test err isa UnifiedCompiler.UnsupportedIR && occursin("gc_token", err.what)
end

# ---------------------------------------------------------------------------
# try regions (A4): EnterNode/:leave/:pop_exception + PhiC/Upsilon synthesis
# ---------------------------------------------------------------------------

# the throw trigger for builder-made bodies (arg == 7 throws)
tx_throw(x) = x == 7 ? error("txboom") : x

# scope-carrying enter (`@with`): top level so the function captures nothing
# (an OpaqueClosure's self slot replaces the fuzzed #self# argument)
const txsv = Base.ScopedValues.ScopedValue(1)
txwith(x) = Base.ScopedValues.@with(txsv => x, txsv[] + 1)

@testset "typed exit: try/catch value join" begin
    # Stock oracle (code_ircode of the inlined `try; div(a,b); catch; 0; end`):
    #   2 ─ %2 = enter #5
    #   3 ─ %3 = intrinsic Base.checked_sdiv_int(_2, _3)::Int64
    #   └──      $(Expr(:leave, :(%2)))
    #   4 ─      goto #6
    #   5 ┄      $(Expr(:pop_exception, :(%2)))::Nothing
    #   └──      goto #6
    #   6 ┄ %8 = φ (#4 => %3, #5 => 0)::Int64
    # — the value-producing try is an ordinary φ join of the post-:leave
    # body edge and the post-:pop_exception handler edge.
    b = Builder(name = :tvaljoin)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    t = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
    v = append_stmt!(b, K"call", tx_throw, x; type = Any)
    v2 = append_stmt!(b, K"call", GlobalRef(Base, :+), v, 1; type = Any)
    append_stmt!(b, K"result", v2)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    append_stmt!(b, K"result", -1)
    UnifiedIR.close_region!(b)
    r = append_stmt!(b, K"call", GlobalRef(Base, :*), t, 2; type = Any)
    append_stmt!(b, K"return", r)
    ir = finish!(b)
    irc = exit_differential(ir, (1,), (7,))     # ok path and handler path
    @test count_nodes(irc, Core.EnterNode) == 1
    @test count_exprs(irc, :leave) == 1
    @test count_exprs(irc, :pop_exception) == 1
    @test count_nodes(irc, Core.PhiNode) >= 1   # the value join
end

@testset "typed exit: handler-crossing single store" begin
    # store before the try + one store in the body, read in the handler:
    # the handler observes the LAST store on the throwing prefix — a
    # PhiCNode fed by the initial Upsilon (pre-try value, inserted at the
    # `enter`) and the per-store Upsilon (stock slot2ssa's convention).
    b = Builder(name = :tphic1)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"cell", Any; type = Any)
    append_stmt!(b, K"cell_set", c, 0)
    t = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
    v = append_stmt!(b, K"call", tx_throw, x; type = Any)          # may throw (reads 0)
    append_stmt!(b, K"cell_set", c, v)
    x1 = append_stmt!(b, K"call", GlobalRef(Base, :+), x, 1; type = Any)
    w = append_stmt!(b, K"call", tx_throw, x1; type = Any)         # may throw (reads v)
    append_stmt!(b, K"result", w)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    g = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)
    append_stmt!(b, K"result", g)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", t)
    ir = finish!(b)
    irc = exit_differential(ir, (1,), (7,), (6,))   # ok / throw-at-1st / throw-at-2nd
    @test count_nodes(irc, Core.PhiCNode) == 1
    @test count_nodes(irc, Core.UpsilonNode) == 2   # initial + the body store
end

@testset "typed exit: handler-crossing multi-store (PhiC with several Upsilons)" begin
    # Stock oracle (code_ircode of multistore: y stored 4x, read in catch):
    #   2 ─ %2  = ϒ (1)      └── %3 = enter #6
    #   3 ─ %4  = ϒ (2) ... %6 = ϒ (3) ... %9 = ϒ (4)
    #   6 ┄ %13 = φᶜ (%2, %4, %6, %9)::Int64
    b = Builder(name = :tphicN)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"cell", Any; type = Any)
    append_stmt!(b, K"cell_set", c, 0)
    t = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
    append_stmt!(b, K"cell_set", c, 10)
    append_stmt!(b, K"call", tx_throw, x; type = Any)
    append_stmt!(b, K"cell_set", c, 20)
    x1 = append_stmt!(b, K"call", GlobalRef(Base, :-), x, 1; type = Any)
    append_stmt!(b, K"call", tx_throw, x1; type = Any)
    append_stmt!(b, K"cell_set", c, 30)
    append_stmt!(b, K"result", 99)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    g = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)
    append_stmt!(b, K"result", g)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", t)
    ir = finish!(b)
    # x=1: ok → 99; x=7: throws after store 10; x=8: x-1==7 throws after 20
    irc = exit_differential(ir, (1,), (7,), (8,))
    @test count_nodes(irc, Core.PhiCNode) == 1
    @test count_nodes(irc, Core.UpsilonNode) == 4   # initial + 3 stores
end

@testset "typed exit: maybe-undef handler read (empty Upsilon + Bool flag)" begin
    # Stock oracle (code_ircode of maybeundef — conditional store, catch read):
    #   2 ─ %2  = ϒ (#undef)::Union{}       %3 = ϒ (false)::Bool
    #   4 ─ %8  = ϒ (%7)::Int64             %9 = ϒ (true)::Bool
    #   6 ┄ %12 = φᶜ (%2, %8)::Int64        %13 = φᶜ (%3, %9)::Bool
    #   │         $(Expr(:throw_undef_if_not, :y, :(%13)))
    b = Builder(name = :tundef)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"cell", Any; type = Any)      # never stored on one path
    t = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
    cnd = append_stmt!(b, K"call", GlobalRef(Base, :<), x, 5; type = Any)
    fi = append_stmt!(b, K"if", cnd; type = Any)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"cell_set", c, 99)
    append_stmt!(b, K"result")
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
    append_stmt!(b, K"result")
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"call", tx_throw, 7; type = Any)  # always throws
    append_stmt!(b, K"result", 0)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    g = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)   # UndefVarError when x >= 5
    append_stmt!(b, K"result", g)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", t)
    ir = finish!(b)
    irc = exit_differential(ir, (1,), (9,))     # defined path / undef path
    @test count_nodes(irc, Core.PhiCNode) == 2  # value + Bool definedness
    @test count_exprs(irc, :throw_undef_if_not) == 1
    @test any(s -> s isa Core.UpsilonNode && !isdefined(s, :val), irc.stmts.stmt)
end

@testset "typed exit: nested try (Upsilon chains into both handlers)" begin
    b = Builder(name = :tnest)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    c = append_stmt!(b, K"cell", Any; type = Any)
    append_stmt!(b, K"cell_set", c, 1)
    touter = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, touter; kind = UnifiedIR.REGION_BODY)
    tinner = append_stmt!(b, K"try"; type = Any)
    UnifiedIR.open_region!(b, tinner; kind = UnifiedIR.REGION_BODY)
    append_stmt!(b, K"cell_set", c, 2)
    append_stmt!(b, K"call", tx_throw, x; type = Any)
    append_stmt!(b, K"cell_set", c, 3)
    x1 = append_stmt!(b, K"call", GlobalRef(Base, :-), x, 1; type = Any)
    append_stmt!(b, K"call", tx_throw, x1; type = Any)
    append_stmt!(b, K"result", 0)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, tinner; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    gi = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)  # inner observation
    hundred = append_stmt!(b, K"call", GlobalRef(Base, :*), gi, 100; type = Any)
    append_stmt!(b, K"cell_set", c, hundred)
    append_stmt!(b, K"call", tx_throw, 7; type = Any)          # re-raise to outer
    append_stmt!(b, K"result", 0)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"result", tinner)
    UnifiedIR.close_region!(b)
    UnifiedIR.open_region!(b, touter; kind = UnifiedIR.REGION_HANDLER)
    append_stmt!(b, K"region_arg"; type = Any)
    go = append_stmt!(b, K"cell_get", op_stmt(c); type = Any)  # outer observation
    append_stmt!(b, K"result", go)
    UnifiedIR.close_region!(b)
    append_stmt!(b, K"return", touter)
    ir = finish!(b)
    # x=1: ok → 0; x=7: inner sees 2 → outer sees 200; x=8: inner 3 → outer 300
    irc = exit_differential(ir, (1,), (7,), (8,))
    @test count_nodes(irc, Core.EnterNode) == 2
    @test count_nodes(irc, Core.PhiCNode) == 2
    # inner PhiC: initial + 2 body stores = 3; outer: initial + the 2 body
    # stores (each protected by BOTH trys) + the inner-handler store
    # (protected by the outer try only) = 4
    @test count_nodes(irc, Core.UpsilonNode) == 7
end

@testset "typed exit: try in loop (per-iteration enter + handler)" begin
    b = Builder(name = :tinloop)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Any)
    r = build_loop!(b, 0, 1; type = Any, argtypes = Any[Any, Any]) do b, args
        s, j = args
        body = UnifiedIR.current_region(b)
        t = append_stmt!(b, K"try"; type = Any)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
        v = append_stmt!(b, K"call", tx_throw, j; type = Any)   # throws when j == 7
        append_stmt!(b, K"result", v)
        UnifiedIR.close_region!(b)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
        append_stmt!(b, K"region_arg"; type = Any)
        append_stmt!(b, K"result", 100)
        UnifiedIR.close_region!(b)
        s2 = append_stmt!(b, K"call", GlobalRef(Base, :+), s, t; type = Any)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :+), j, 1; type = Any)
        cnd = append_stmt!(b, K"call", GlobalRef(Base, :<=), j2, n; type = Any)
        append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(s2), op_stmt(j2))
    end
    tot = append_stmt!(b, K"extract", op_stmt(r), op_inline(1); type = Any)
    append_stmt!(b, K"return", tot)
    ir = finish!(b)
    # n=5: no throw (sum 1..5); n=9: iteration j==7 contributes 100
    irc = exit_differential(ir, (5,), (9,))
    @test count_nodes(irc, Core.EnterNode) == 1
end

@testset "typed exit: break crossing the try (leave synthesis)" begin
    # loop-in-try inverted: the loop is OUTSIDE, the `try` inside the body,
    # and a `break` from inside the try body exits the loop — the exit runs
    # the try's structural leave action (§5.9): an extra `:leave` beyond the
    # body-result one.
    b = Builder(name = :tbreakleave)
    append_stmt!(b, K"region_arg"; type = Any)
    n = append_stmt!(b, K"region_arg"; type = Any)
    r = build_loop!(b, 0, 1; type = Any, argtypes = Any[Any, Any]) do b, args
        s, j = args
        body = UnifiedIR.current_region(b)
        t = append_stmt!(b, K"try"; type = Any)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
        v = append_stmt!(b, K"call", tx_throw, j; type = Any)   # throws when j == 7
        big = append_stmt!(b, K"call", GlobalRef(Base, :>), v, n; type = Any)
        fi = append_stmt!(b, K"if", big; type = Any)
        UnifiedIR.open_region!(b, fi; kind = UnifiedIR.REGION_ARM)
        append_stmt!(b, K"break", op_region(body), op_stmt(s), op_stmt(j))  # crosses the try
        UnifiedIR.close_region!(b)
        append_stmt!(b, K"result", v)
        UnifiedIR.close_region!(b)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
        append_stmt!(b, K"region_arg"; type = Any)
        append_stmt!(b, K"result", 1000)
        UnifiedIR.close_region!(b)
        s2 = append_stmt!(b, K"call", GlobalRef(Base, :+), s, t; type = Any)
        j2 = append_stmt!(b, K"call", GlobalRef(Base, :+), j, 1; type = Any)
        append_stmt!(b, K"continue", op_region(body), true, op_stmt(s2), op_stmt(j2))
    end
    tot = append_stmt!(b, K"extract", op_stmt(r), op_inline(1); type = Any)
    append_stmt!(b, K"return", tot)
    ir = finish!(b)
    # n=3: breaks when j==4 (sum 1+2+3); n=9: handler fires at j==7 first
    irc = exit_differential(ir, (3,), (9,))
    @test count_exprs(irc, :leave) >= 2         # body result + the crossing break
end

@testset "typed exit: continue crossing the try (leave on both edges)" begin
    # conditional `continue` inside a try body: the back edge re-enters the
    # loop (and the next iteration's `enter` re-executes), the false edge
    # exits the loop — both run the try's leave action first
    b = Builder(name = :tcontleave)
    append_stmt!(b, K"region_arg"; type = Any)
    x = append_stmt!(b, K"region_arg"; type = Any)
    r = build_loop!(b, x; type = Any, argtypes = Any[Any]) do b, args
        s, = args
        body = UnifiedIR.current_region(b)
        t = append_stmt!(b, K"try"; type = Any)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_BODY)
        s1 = append_stmt!(b, K"call", GlobalRef(Base, :+), s, 1; type = Any)
        v = append_stmt!(b, K"call", tx_throw, s1; type = Any)  # throws when s+1 == 7
        cnd = append_stmt!(b, K"call", GlobalRef(Base, :<), v, 3; type = Any)
        append_stmt!(b, K"continue", op_region(body), op_stmt(cnd), op_stmt(v))
        UnifiedIR.close_region!(b)
        UnifiedIR.open_region!(b, t; kind = UnifiedIR.REGION_HANDLER)
        append_stmt!(b, K"region_arg"; type = Any)
        append_stmt!(b, K"result", -1)
        UnifiedIR.close_region!(b)
        # the try completes normally only through the handler (-1): stop
        append_stmt!(b, K"break", op_region(body), op_stmt(t))
    end
    append_stmt!(b, K"return", r)
    ir = finish!(b)
    # x=0: 1,2,3 → exits with 3; x=6: s+1 == 7 throws → handler → -1
    irc = exit_differential(ir, (0,), (6,))
    @test count_exprs(irc, :leave) >= 1
    @test count_nodes(irc, Core.EnterNode) == 1
end

@testset "typed exit: real bodies through the pipeline (finally/rethrow/nesting)" begin
    # real lowered forms — `finally` arrives frontend-lowered (try/catch +
    # rethrow on the exceptional path), `rethrow()` is a plain call in the
    # handler; differential = plain call vs OpaqueClosure of the exit,
    # including the throwing inputs
    txcatch(x) = try; div(10, x); catch; -1; end
    function txfinally(x)
        s = 0
        try
            s = div(10, x)
        finally
            s += 1
        end
        s
    end
    function txrethrow(x)
        try
            div(10, x)
        catch e
            e isa DivideError ? rethrow() : -1
        end
    end
    function txnested(x)
        try
            try
                div(10, x)
            catch
                -1
            end
        catch
            -2
        end
    end
    function txhandlerread(x)
        y = 0
        try
            y = div(10, x)
            error("no")
        catch
            return y
        end
    end
    function txmultistore(x)
        y = 1
        try
            y = 2
            tx_throw(x)
            y = 3
            tx_throw(x + 1)
            y = 4
        catch
            return y
        end
        return y
    end
    function txexcuse(x)                        # `catch e` binding (the_exception)
        try
            div(10, x)
        catch e
            e isa DivideError ? -5 : -6
        end
    end
    function txloopintry(n)
        s = 0
        try
            i = 1
            while true
                s += div(100, i)
                i += 1
                i > n && break
            end
        catch
            s = -1
        end
        s
    end
    for (f, inputs) in Any[
        (txcatch, [(5,), (0,)]),
        (txfinally, [(5,), (0,)]),
        (txrethrow, [(5,), (0,)]),
        (txnested, [(5,), (0,)]),
        (txhandlerread, [(5,), (0,)]),
        (txmultistore, [(1,), (7,), (6,)]),
        (txexcuse, [(5,), (0,)]),
        (txloopintry, [(5,), (0,), (-1,)]),
        (txwith, [(41,), (7,)]),
    ]
        ir = UnifiedCompiler.typed_ir(f, Any[Int])
        irc = UnifiedCompiler.ir_to_ircode(ir)
        Compiler.verify_ir(irc)
        oc = Core.OpaqueClosure(irc)
        for inp in inputs
            @test isequal(tx_outcome(f, inp...), tx_outcome(oc, inp...))
        end
    end
end

# ---------------------------------------------------------------------------
# seeded fuzz: try-bearing region IR → typed exit → differential vs the
# reference interpreter (CellFuzz harness patterns, §6 leg (b) — here the
# object under test is the BOUNDARY, so the raw un-promoted body goes
# through the exit's flat mem2reg; every 5th case also exits the promoted
# copy, the exact residual-classes-only shape the driver emits)
# ---------------------------------------------------------------------------

isdefined(@__MODULE__, :CellFuzz) || include("cellfuzz.jl")

module TryFuzz

using Random
using UnifiedIR
using UnifiedIR: op_stmt, op_inline, op_region, StmtId, RegionId
using ..CellFuzz

"One random try-bearing body: CellFuzz's generators over a root with a
dedicated #self# arg (so the OpaqueClosure's closure slot never collides
with a fuzzed value), at least one `try` region guaranteed."
function randir(rng::AbstractRNG)
    b = Builder(name = :tryfz)
    append_stmt!(b, K"region_arg"; type = Any)          # #self#
    a1 = append_stmt!(b, K"region_arg"; type = Any)
    a2 = append_stmt!(b, K"region_arg"; type = Any)
    ints = StmtId[a1, a2]
    cx = CellFuzz.Ctx(b, rng, StmtId[], RegionId[])
    for _ in 1:rand(rng, 1:3)
        c = append_stmt!(b, K"cell", Any; type = Any)
        rand(rng) < 0.75 && append_stmt!(b, K"cell_set", c, CellFuzz.pick(cx, ints))
        push!(cx.cells, c)
    end
    CellFuzz.gentry!(cx, ints, 0)                       # the guaranteed try
    CellFuzz.body!(cx, ints, 0)
    ret = if !isempty(cx.cells) && rand(rng) < 0.5
        append_stmt!(b, K"cell_get", op_stmt(rand(rng, cx.cells)); type = Any)
    else
        CellFuzz.pick(cx, ints)
    end
    append_stmt!(b, K"return", ret isa StmtId ? op_stmt(ret) : ret)
    return finish!(b)
end

"""
    run_exit_cases(U, CC, n; seed, promoted_every = 5) -> stats

For each seeded case: build a try-bearing body, take interpreter outcomes
(values AND thrown errors) on 4 input pairs, exit to IRCode, STOCK
verify_ir, execute the OpaqueClosure, compare outcomes. Every
`promoted_every`th case repeats the exit on a promotion_fixpoint!-ed copy.
Returns counters + the outcome histogram (asserted by the caller: seeded,
deterministic).
"""
function run_exit_cases(U, CC, n::Int; seed::Int = 0x7e57, promoted_every::Int = 5)
    stats = (; cases = Ref(0), diffs = Ref(0), verifyfails = Ref(0), declines = Ref(0),
             promoted = Ref(0), hist = Dict{Symbol,Int}(),
             failures = Tuple{Int,Int,Symbol}[])
    for case in 1:n
        rng = Xoshiro(seed + case)
        ir = randir(copy(rng))
        UnifiedIR.verify_ir(ir; level = 1)
        inputs = [(rand(rng, -2:9), rand(rng, -2:9)) for _ in 1:3]
        push!(inputs, (7, 7))                       # the fzthrow trigger
        ref = [CellFuzz.outcome(ir, nothing, a...) for a in inputs]
        for r in ref
            stats.hist[r[1]] = get(stats.hist, r[1], 0) + 1
        end
        function check(irx, tag)
            irc = try
                U.ir_to_ircode(irx)
            catch
                push!(stats.failures, (seed, case, Symbol(tag, :_decline)))
                stats.declines[] += 1
                return
            end
            try
                CC.verify_ir(irc)
            catch
                push!(stats.failures, (seed, case, Symbol(tag, :_verify)))
                stats.verifyfails[] += 1
                return
            end
            oc = Core.OpaqueClosure(irc)
            got = [(try
                        (:ok, oc(a...))
                    catch e
                        (:err, replace(sprint(showerror, e), r"( \(%\d+\)| at %\d+)" => ""))
                    end) for a in inputs]
            if got != ref
                push!(stats.failures, (seed, case, Symbol(tag, :_differential)))
                stats.diffs[] += 1
            end
        end
        check(ir, :raw)
        if promoted_every > 0 && case % promoted_every == 0
            ir2 = U.promotion_fixpoint!(randir(Xoshiro(seed + case)))
            UnifiedIR.verify_ir(ir2; level = 1)
            check(ir2, :promoted)
            stats.promoted[] += 1
        end
        stats.cases[] += 1
    end
    return stats
end

end # module TryFuzz

@testset "typed exit: try fuzz battery (120 cases, seeded)" begin
    s = TryFuzz.run_exit_cases(UnifiedCompiler, Compiler, 120; seed = 20260716)
    @test s.cases[] == 120
    @test s.promoted[] == 24
    @test s.diffs[] == 0
    @test s.verifyfails[] == 0
    @test s.declines[] == 0
    @test isempty(s.failures)
    # seeded, deterministic outcome histogram: both classes exercised
    @test s.hist == Dict(:ok => 344, :err => 136)
    println("try fuzz battery: ", s.cases[], " cases (+", s.promoted[],
            " promoted variants); outcomes ", s.hist)
end
