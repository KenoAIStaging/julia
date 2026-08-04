# Upstream flags from the UnifiedIR compiler-port campaign

Issues found during the port that belong to stock Julia (runtime, stdlib,
tests, or lowering), not to the unified pipeline. Each was hit while
running the full build/test cycle under the unified-by-default sysimage;
where the unified pipeline was implicated it merely widened a window or
raised the trigger frequency — the defect itself is stock.

## 1. LMDB objcache skips the optimizer on warm JIT hits

`Compiler/test/codegen.jl` `test_jl_dump_llvm_opt`: with the LMDB object
cache enabled, a warm cache hit bypasses the optimizer so the dump hook
never fires and the test fails. Reproduces on stock and on nightly;
`JULIA_OBJCACHE=0` makes it pass. The fix belongs in the objcache path
(fire the hook or bypass the cache when a dump hook is installed); the
test was left untouched in this tree.

## 2. SIGTERM-injection hazards in the runtime exit path (two)

Observed as kill-time `val already in a list` corruption and SpinLock
spins when a process receives SIGTERM: the exit callback is injected
into (a) tasks currently holding SpinLocks and (b) tasks parked in wait
queues. Any pipeline stretches the window (longer compile times make it
easier to hit); the hazard is stock runtime exit machinery. Base was
left untouched here.

## 3. Stock traps on the maybe-undef immutable-field-load class

While auditing an optimizer difference (unified deleting a maybe-undef
immutable field load that stock preserves as a conditional throw), we
found stock itself TRAPS on this class of partially-initialized
immutable getfield. Worth an audit of the stock behavior — the
conditional-throw preservation and the runtime trap disagree.

## 4. REPL precompile workload hangs forever on subtask death

`stdlib/REPL/src/precompile.jl` (`repl_workload`): if any spawned
REPL/LineEdit subtask dies (we hit this via a compiler bug, but any
exception works), the workload's `wait` on its lock never completes and
`Base.Precompilation.precompilepkgs` has no timeout — the build hangs
silently and indefinitely inside the REPL precompile with zero CPU.
Suggested hardening: propagate subtask failure to the waiter (or a
watchdog around the workload), and/or a per-package timeout in
`precompilepkgs`. Diagnosis recipe that worked: gdb attach +
`call jl_print_task_backtraces(0)` through the worker's captured stderr.

## 5. `drop_all_caches` test deadlocks on a full stderr pipe

`Compiler/test/invalidation.jl` (`drop_all_caches` testset): the
subprocess writes `--trace-compile` output to a pipe that the test only
reads AFTER `run()` returns. When the trace volume exceeds the 64 KiB
pipe buffer the child blocks flushing stderr inside its atexit uv loop
and `run()` never returns. Under stock the volume usually stays below
the buffer; any configuration that raises trace output (more
invalidation, bigger compiler) deadlocks. Fixed in this tree by
draining concurrently (commit e5fb3689e4) — the same pattern should be
applied upstream.

## 6. JuliaLowering under-reports SLOT_USEDUNDEF on late-materialized slots

JuliaLowering's linear IR leaves `is_used_undef`/`is_read` unset on the
break-block result slot it materializes during linearization (the
`loop-exit_result` slot of a value-position `while` with a conditional
`break`). The emitted CodeInfo is correct (an `Expr(:isdefined)` guard
backfills), but slotflags under-report SLOT_USEDUNDEF, which
stock-codegen consumers may rely on. Flisp lowering has the same shape
via `src/method.c` name simplification. Found while root-causing a
consumer bug in this tree; the producer-side flag gap is upstream
JuliaLowering material.

## 7. `Type{X}`-bound static parameters are pinned by `==`, not `===`

JuliaLang/julia#61323. A static parameter bound through a `Type{T}` slot
is chosen by subtyping (`==` on the type), not object identity, so
`T === Int` may be false for a type that is `== Int`. This tree carries
a consumer-side fix for its own inliner (commit 794b1b1b0b — the working
distinction is `Type{tv}` bound by `==` vs `Type{Foo{tv}}` deduplicated
by the type cache and therefore identity-pinned); the upstream issue
remains open for the general rule.

## 8. (Resolved upstream) exception-stack leak on break/continue across handlers

Found independently during the port as F7; already fixed upstream in
30346cd089. Listed for completeness — no action needed.
