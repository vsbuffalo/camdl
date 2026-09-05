---
date: 2026-06-07
status: implemented
related: ../../rust/crates/sim/src/resolved_expr.rs, ../../rust/crates/sim/src/propensity.rs
evidence: ../notes/2026-06-07-runtime-binding-cache.md (before/after timings + profiles)
gate: ../../rust/crates/sim/tests/gate_binding_cache_ab.rs
---

# Runtime binding cache: evaluate each model binding once per propensity step

This is a **Rust** runtime change (the propensity evaluator). The OCaml compiler
already emits the shared bindings (`N[l]`, `I_agg[l]`, …) via Fix-B hoisting;
nothing OCaml-side changes. The win is purely in how the runtime _re-uses_ them.

## Problem

`BindingRef` is evaluated on demand, with no memoization:

```rust
// resolved_expr.rs, today:
ResolvedExpr::BindingRef(slot) => eval_resolved(&ctx.model.resolved.bindings[*slot], ctx),
```

In a spatially-coupled model the per-source aggregates `N[q]` (population) and
`I_agg[q]` (infectious) appear once per destination stratum in the FOI
`sum(q, W[l,q] * I_agg[q] / N[q])`. On a dense P=44, A=21 model that is **945
references each**, and every reference re-runs the binding's PopSum from scratch
_within a single propensity-vector evaluation_:

```
cost report (gen_spatial P=44 dense):
  N_p0      state  size=1  refs=945  ~saved=944    ← "saved" is gated on caching
  I_agg_p0  state  size=1  refs=945  ~saved=944
```

The profile lands the cost exactly there: on the simulation thread,
`sim::resolved_expr::eval_resolved` is **46–54% of compute** (46% on this P=44
model; 54% on a national-scale model), dominated by these redundant `BindingRef`
re-evaluations. Hoisting more bindings does **not** help on its own — a
`BindingRef` is recomputed on every reference, so the saving only exists once
the cache does.

## Design

Memoize binding values for the lifetime of one propensity-vector evaluation (one
state snapshot of one cell/particle). All rates for that state share the cache;
the next state bumps a generation counter to invalidate in O(1).

```rust
// EvalCtx gains a per-evaluation binding cache. Interior-mutable (Cell) so the
// existing `eval_resolved(expr, &EvalCtx)` signature is unchanged:
pub struct EvalCtx<'a> {
    // … existing fields (model, int_s, real_s, params, t, dt, …) …
    pub bind_val: &'a [Cell<f64>],   // one slot per binding (allocated once, reused)
    pub bind_gen: &'a [Cell<u32>],   // per-slot generation stamp
    pub gen:      &'a Cell<u32>,     // current generation; bump = invalidate all
}

// resolved_expr.rs:
ResolvedExpr::BindingRef(slot) => {
    let g = ctx.gen.get();
    if ctx.bind_gen[*slot].get() == g {
        ctx.bind_val[*slot].get()                              // hit
    } else {
        let v = eval_resolved(&ctx.model.resolved.bindings[*slot], ctx);
        ctx.bind_val[*slot].set(v);
        ctx.bind_gen[*slot].set(g);
        v                                                      // miss → fill
    }
}
```

Invalidation is one increment, where the state the rates read changes:

```rust
// propensity / backend step, before computing all transition rates for a state:
ctx.gen.set(ctx.gen.get().wrapping_add(1));
```

Notes that keep it correct:

- Bindings are topologically ordered (a `BindingRef` only references earlier
  ones), so a miss that recursively evals a body hits the earlier slots' caches.
- The cache is **per cell/particle propensity eval** and `Cell` is `!Sync`, so
  parallelism must stay _across_ cells/particles (each owning its own
  `bind_val`/`bind_gen`), never _within_ one propensity vector — which is
  already how the backends batch. Confirm this before wiring the construction
  sites.
- A generation stamp (not `clear()`) makes invalidation O(1) regardless of
  binding count.

### As built (deviation from the sketch above)

The sketch threads `bind_val`/`bind_gen`/`gen` through `EvalCtx`. As built, the
cache is a **thread-local** (`BindingCache` in `resolved_expr.rs`) entered via
an RAII `CacheScope` in `eval_propensities`, _not_ `EvalCtx` fields.

Reason: the `EvalCtx`-fields form changes the `eval_resolved(expr, &EvalCtx)`
signature transitively and forces every `EvalCtx` construction site (propensity,
gradient, obs-likelihood) to allocate and pass the cache buffers, even the ones
that don't want caching. The thread-local keeps the signature untouched and
scopes the cache to exactly the one site that benefits (`eval_propensities`),
falling through to on-demand eval everywhere else — which is _why_ gradient and
obs-likelihood evals stay byte-identical to the pre-cache path. Correctness
properties (topological order, per-step generation stamp, cross-particle
isolation via thread-locality) are unchanged.

Cost of the deviation: a thread-local access (`LocalKey::with` +
`_tlv_get_addr`) on every `BindingRef`, ~12% of the after-profile busy thread.
The `EvalCtx`-by- reference form was subsequently built and benchmarked to
reclaim it — and a controlled binary A/B measured **no speedup** (the eliminated
thread-local samples re-attribute to `eval_resolved`; total work is unchanged).
The thread-local form is therefore the final design. See the negative-result
writeup in `../notes/2026-06-07-runtime-binding-cache.md`.

## Expected speedup

`eval_resolved` is the dominant cost; within it the redundant work is
`N[q]`/`I_agg[q]` recomputed ~945× each (a PopSum of ~105 / ~21 terms). Caching
collapses 945 evals → 1 per binding per step. The irreducible FOI sum (P²
multiply-adds) and the non-rate fraction (RNG, output, alloc) remain.

The conservative pre-implementation estimate (~1.5×) was anchored on a **short
365-day run**, where setup/IO dilutes `eval_resolved` to ~46% of the busy
thread:

```
eval_resolved 3× faster → 1/(0.46 + 0.54/3) = 1.56× overall   ← short-run estimate
```

But the headline workload is a long horizon where per-step eval dominates and
`eval_resolved` is ~79%. Amdahl on that share, with the cache cutting eval work
~6×, predicts ~3×:

```
1/( (1-0.79) + 0.79/6.2 ) = 2.96×                              ← long-run prediction
```

**Estimate: ~1.5× (short run) to ~3× (long run).** The realized number lands at
the long-run end (below).

## Before / after — estimate vs realized

Benchmark: `gen_spatial P=44 A=21` dense coupling, chain_binomial, dt=1, seed 1,
**horizon 3650 d** (steady-state per-step eval dominates; reproducible via
`scripts/gen_scaling_models.py`; the colleague's national model is private and
not used here). Same release binary, cache toggled by `CAMDL_NO_BINDING_CACHE`;
wall = best-of-3, profile = samply busy-thread leaf attribution.

```
                       wall (s)   speedup   eval_resolved (% busy)   eval_resolved samples
  before (no cache)      9.06       1.0×          78.9%                   7443
  after  (cache)         3.31       2.74×         36.3%                   1194  (6.2× fewer)

  estimate (proposal)     —        ~1.5×           —                       —     (short-run anchor)
```

**Realized 2.74× — at the long-run end of the estimate, not the short-run
anchor.** Busy-sample ratio (2.87×) corroborates the wall-clock. `eval_resolved`
dropped 6.2× in absolute samples; the residual gap to the 2.96× Amdahl
prediction is the cache's own cost, visible in the after profile:
`thread::local::LocalKey::with` 12.4% + `_tlv_get_addr` — the thread-local
indirection on every `BindingRef`. Passing the cache by reference through
`EvalCtx` (the original sketch above) was tried to reclaim that and a controlled
A/B measured **no speedup** — the eliminated thread-local samples re-attribute
to `eval_resolved` rather than disappearing (negative-result writeup in the
note). The thread-local form stands. Profile artifacts:
`docs/dev/notes/assets/2026-06-07-binding-cache-{before,after}.json.gz` (+
`.syms.json` sidecars).

## Lift / risk / gate

~50–100 lines (`resolved_expr.rs` cache arm + `EvalCtx` fields + the propensity
/ backend construction sites + invalidation). Medium lift; hot-path,
inference-adjacent. Gate:

1. **Byte-identical A/B** (`gate_constant_fold_ab.rs` pattern): same model,
   cache on vs off, assert identical trajectories under every backend, with a
   non-vacuity check that the cache actually serves hits (else it proves
   nothing).
2. Re-run the profile above; record realized wall + `eval_resolved %` in the
   table.

No OCaml change. Deferred follow-up: a per-dependency-class generation (Time /
Param bindings cacheable across more steps) — only if the profile after step 1
still shows binding re-eval; the per-step cache already captures the FOI win.
