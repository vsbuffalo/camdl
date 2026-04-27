---
status: complete
date: 2026-04-24
scope: full codebase — OCaml frontend, Rust backend, IR schema, tests
reviewer: Claude (claude-sonnet-4-6)
prior-reviews:
  - docs/dev/reviews/2026-04-19-review-inference.md
  - docs/dev/reviews/2026-04-20-review-ocaml-design.md
  - docs/dev/reviews/2026-04-20-review-rust-design.md
  - docs/dev/reviews/2026-04-21-spec-claims-vs-tests.md
---

# Full Codebase Review — 2026-04-24

## Prior Finding Resolution Status

| Code   | Description                                             | Status       |
|--------|---------------------------------------------------------|--------------|
| RdM1   | EstimatedParam scattered across if2.rs                  | Resolved ✅  |
| RdM2   | rate_grads linear search by name at eval time           | Resolved ✅  |
| RdM3   | restore_z_values documented/moved to types.rs           | Resolved ✅  |
| RdM4   | log_transition_density_substep 162-line monolith        | Resolved ✅  |
| Rdm1   | InferenceConfig trait missing                           | Resolved ✅  |
| Rdm2   | LOG_PROB_FLOOR unnamed                                  | Resolved ✅  |
| Rdm3   | init_particle_rngs unnamed                              | Resolved ✅  |
| Rdm4   | RESAMPLE_RNG_STREAM unnamed                             | Resolved ✅  |
| Rdn2   | Duplicate lower/upper bounds on EstimatedParam          | Deferred     |
| OcM1   | ir.mli phantom interface file                           | Resolved ✅  |
| OcM2   | dimcheck/validate diagnostics carry no source location  | Deferred     |
| OcM3   | differentiate_rate emitted Const 0.0 entries            | Resolved ✅  |
| OcM4   | autodiff Mod failwith uncaught                          | Resolved ✅  |
| Ocm1   | No .mli files for compiler modules                      | Open         |
| Ocm2   | Hashtbl.create 16 ad-hoc sizes                          | Resolved ✅  |
| Ocm3   | Diagnostics.has_any missing                             | Resolved ✅  |
| Ocn1   | 73 no_loc sites in expander.ml                          | Open         |
| IC1/2  | BetaBinomial/Normal obs models                          | Resolved ✅  |
| IC3    | TransformedNormal double-Jacobian                       | Resolved ✅  |
| IC4    | Prior × transform compatibility validator               | Resolved ✅  |
| IM1    | Per-particle RNG streams                                | Resolved ✅  |
| IM6    | CSMC-AS ancestor weights post-resample non-uniform      | Resolved ✅  |
| IM7    | gamma_idx per-transition in gradient                    | Resolved ✅  |
| IM8    | CPM multi-overdispersion preflight missing              | Resolved ✅  |
| IM9    | RATE_EPSILON alignment source-group path                | Partial ✅*  |
| IM10   | Config hash missing                                     | Resolved ✅  |
| IM11   | Geyer pair-sum ESS                                      | Resolved ✅  |
| IM12   | ESS NaN on non-convergence                              | Resolved ✅  |
| Im18   | Heated rung re-warmup on resume                         | Documented   |
| Im1-4  | CLI dead code / unused terminal colors                  | Open         |
| P1.1   | scenario `set` not tested at runtime                    | Resolved ✅  |
| P1.2   | scenario `scale` not tested at runtime                  | Resolved ✅  |
| P2     | 84+ error codes without golden error fixtures           | Open         |

\* IM9 fixed the source-group (multinomial) path but not the ungrouped (Poisson) path — see **InM1** below.

---

## Summary

**What the codebase does well.** The inference architecture is sound: the
complete-data log-likelihood, the CSMC-AS ancestor-sampling sweep, and the
symbolic-differentiation gradient pipeline are all mathematically faithful to
the algorithms they implement. The constant and type discipline introduced since
the April 19 review (LOG_PROB_FLOOR, RESAMPLE_RNG_STREAM, init_particle_rngs,
InferenceConfig trait, per-particle RNG isolation, config hash) makes the
high-stakes inference paths substantially easier to audit. OCaml and Rust IR
types are in tight correspondence; the golden-file test harness gives broad
regression coverage; error-code diagnostics are thorough and actionable for
users.

**Where work is still needed.** One residual correctness gap from the IM9 fix
remains: the ungrouped-transition (Poisson inflow) path in pgas_grad.rs still
uses `rate <= 0.0` rather than `RATE_EPSILON`, causing a gradient/density
mismatch for near-zero-rate inflows that will produce HMC divergences with no
obvious diagnostic. A secondary performance concern applies to
overdispersed models: complete_data_loglik re-allocates state vectors inside
every source-group loop iteration per substep, called thousands of times per
PGAS sweep. Several bare magic literals (seed-mixing multipliers, probability
clamps, Cholesky regularization, mass-adaptation phase split) lack named
constants or citations; these are low-priority individually but add up to
audit friction. The OCaml compiler continues to use `failwith` for a user-
reachable date-parse error and has no `.mli` interface files for its internal
modules.

---

## Findings

### Major

#### InM1 — Ungrouped-transition gradient uses `rate <= 0.0`, not `RATE_EPSILON`

**File:** `rust/crates/sim/src/inference/pgas_grad.rs:221`

The IM9 fix (April 19 review) corrected the source-group (multinomial)
path to use `RATE_EPSILON` at line 117, matching chain_binomial's `step_one`
and pgas.rs's density. The ungrouped/inflow Poisson path was not updated:

```rust
// pgas_grad.rs:220-221  (ungrouped / inflow transitions — Poisson)
for (tr_idx, &rate) in propensities.iter().enumerate() {
    if handled[tr_idx] || rate <= 0.0 { continue; }   // ← should be RATE_EPSILON
```

`pgas.rs:log_transition_density_substep` and `chain_binomial::step_one` both
gate on `RATE_EPSILON = 1e-15`. If a model has an inflow transition with rate
in (0, RATE_EPSILON], the density includes a Poisson(k; λ≈0) term but the
gradient skips it. The mismatch makes the HMC trajectory non-conservative: the
sampler walks in a direction inconsistent with the log-posterior geometry.
Practically this manifests as elevated divergences and stalled chains for any
model with near-zero immigration or seasonal forcing that dips to zero — the
user sees bad mixing rather than an explicit error.

**Fix:** Replace `rate <= 0.0` with `rate <= crate::chain_binomial::RATE_EPSILON`
and also gate on `DrawMethod::Deterministic` to match the grouped path:

```rust
if handled[tr_idx]
    || rate <= crate::chain_binomial::RATE_EPSILON
    || matches!(model.model.transitions[tr_idx].draw_method,
        ir::transition::DrawMethod::Deterministic)
{ continue; }
```

The comment block at lines 98-112 in pgas_grad.rs already describes this exact
invariant — the ungrouped path was inadvertently excluded from the fix.

---

### Minor

#### SiM1 — complete_data_loglik allocates inside the source-group loop

**File:** `rust/crates/sim/src/inference/pgas.rs:540-549`

For overdispersed models, the gamma-density block inside
`complete_data_loglik` allocates three heap objects per source group per
substep:

```rust
// pgas.rs:540-549  (inside: for s in 0..n_substeps, for &(src_local, ref group) in &model.source_groups)
let mut local_props = vec![0.0; n_tr];                         // ← allocated per group
let _ = eval_propensities(model, &{
    let mut s = IntState::new(n_int_local);                     // ← allocated per group
    s.counts.copy_from_slice(&rec.counts_before);
    s
}, &real_s_local, params, ctx.t, &mut local_props);
```

`IntState::new` and `RealState::new` are called once each outside the group
loop (lines 528-529) and are unused — the state actually needed for rate
evaluation is constructed fresh inside the closure. During a typical PGAS run
with 1000 particles × 1000 sweeps × 10 substeps × 4 source groups, this is
~40 million allocations for the three objects combined. The fix is to hoist the
`local_props` Vec and the evaluation state above the source-group loop,
resetting `local_props` to zero between groups if needed (or computing
propensities once per substep and reading from it inside the loop).

#### OcN2 — `parse_iso_date` uses `failwith` for a user-reachable error

**File:** `ocaml/lib/compiler/expander.ml:108-109`

```ocaml
with _ -> failwith (Printf.sprintf
    "invalid date literal '%s': components must be integers" s))
| _ -> failwith (Printf.sprintf
    "date literal must be YYYY-MM-DD, got '%s'" s)
```

`parse_iso_date` is called from `date_to_substep` which is called when
processing `events` date fields — direct user input. Both `failwith` calls
produce a bare OCaml exception with no error code, no source location, and no
hint text. This violates the "error messages are a feature" principle in
CLAUDE.md and the rule against `failwith` for user-facing errors. The fix is
to emit a structured `Diagnostics` error (a new code in the E4xx or E5xx
range for date-literal parse failures) with the date string and format hint.

---

### Nit

#### Inn1 — CSMC seed-mixing multipliers are undocumented magic constants

**File:** `rust/crates/sim/src/inference/pgas.rs:1410-1411, 1652-1654`

```rust
// warmup sweep seed (pgas.rs:1410-1411):
let csmc_seed = seed ^ ((warmup_sweep as u64).wrapping_mul(0x517cc1b727220a95))
    ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142);

// production sweep seed (pgas.rs:1652-1654):
let csmc_seed = seed ^ ((sweep as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
    ^ (rung as u64).wrapping_mul(0x6c62272e07bb0142)
    ^ (csmc_rep as u64).wrapping_mul(0xa2ce44bbfe0cf6d5);
```

`0x9e3779b97f4a7c15` is the Knuth/Fibonacci multiplicative hash constant
(golden-ratio multiple of 2^64); the others appear to be Fowler-Noll-Vo or
similar constants. None is named or cited. Add named constants with comments
explaining the choice (e.g., "Fibonacci/Knuth multiplicative hash — ensures
each (sweep, rung, rep) triple maps to a distinct seed with good avalanche
properties").

#### Inn2 — 0.7 mass-adaptation phase split is a bare literal

**File:** `rust/crates/sim/src/inference/pgas.rs:1526`

```rust
let mass_adapt_end = (adapt_end as f64 * 0.7) as usize;
```

No named constant, no citation. Stan's dual-averaging schedule uses 0.75 for
the equivalent split. Introduce `MASS_ADAPT_FRAC: f64 = 0.7` with a comment
explaining why 70% rather than the Stan default.

#### Inn3 — Probability clamp 1e-15 is distinct from LOG_PROB_FLOOR but unnamed

**File:** `rust/crates/sim/src/inference/pgas.rs:301, 319`

```rust
let p_total = (1.0 - (-total_rate * dt).exp()).clamp(1e-15, 1.0 - 1e-15);
let p_split = (eff_rate / rate_remaining).clamp(1e-15, 1.0 - 1e-15);
```

`LOG_PROB_FLOOR = 1e-300` is the log-domain floor; these are probability-domain
clamps with a completely different magnitude. The distinction matters: using the
wrong constant for either purpose would be silently incorrect. Name this
`PROB_CLAMP_EPS: f64 = 1e-15` and use it at all four sites (pgas.rs:301, 319
and any corresponding sites in pgas_grad.rs).

#### Inn4 — Cholesky regularization 1e-6 unnamed in nuts.rs

**File:** `rust/crates/sim/src/inference/nuts.rs:53`

```rust
reg[i * d + i] += 1e-6; // regularize for numerical stability
```

Add `CHOLESKY_REG: f64 = 1e-6` with a note on why this magnitude was chosen
(typical Tikhonov regularization for numerically marginal SPD matrices at
double precision).

#### Sin1 — substep_ancestors identity Vec allocated per non-observation substep

**File:** `rust/crates/sim/src/inference/pgas.rs:797`

```rust
substep_ancestors = (0..n_particles).collect();
```

This `Vec<usize>` of length n_particles is allocated once per substep that
does not trigger resampling. For a model with 1000 particles and 100 substeps
per observation interval, this is 100 allocations of a 8 KB Vec per sweep per
particle. The identity case could be represented as an `Option<Vec<usize>>`
(None = identity) to eliminate these allocations.

#### Irn1 — `always_active` serde default silently changes semantics for old IR

**File:** `rust/crates/ir/src/intervention.rs:74-75`

```rust
#[serde(default)]
pub always_active: bool,
```

`#[serde(default)]` means IR written before `always_active` existed
deserializes with `always_active = false`, making previously unconditional
interventions into toggleable ones. Given CLAUDE.md's "backwards compatibility
is a non-goal" policy this field should require an explicit value (remove the
`default`) or the schema version should be bumped. At minimum, document the
field's introduction version in a comment.

---

## Cross-Cutting Themes

**1. Partial fixes require explicit checklists.** IM9 (RATE_EPSILON alignment)
was marked resolved after fixing the source-group path but the ungrouped path
was missed. Multi-site invariants — any case where the same threshold, formula,
or ordering must be consistent across more than one function — should be
accompanied by a code comment naming all the sites, so future reviewers and
authors can verify completeness. The pgas_grad.rs comment at lines 98-112
already does this for the source-group path; it should have listed the
ungrouped path explicitly.

**2. Named constants for every magic number in inference-critical code.** The
batch introduced since the April 19 review (LOG_PROB_FLOOR, RATE_EPSILON,
RESAMPLE_RNG_STREAM) demonstrates the value of the pattern. Inn1–Inn4 represent
the remaining unnamed literals in the same files. Each unnamed constant is a
future audit burden: the reader cannot tell whether 1e-15 in one place and
1e-15 in another are intentionally the same quantity or coincidentally the same
number.

**3. Hot-path allocation discipline needs a single ownership pass.** SiM1 and
Sin1 are independent discoveries that share a root cause: no systematic review
has been done of allocations inside the innermost loops of the PGAS sweep.
Given that PGAS is called millions of times in production inference runs
(particles × sweeps × substeps), a single allocation audit of
`csmc_as` and `complete_data_loglik` — profiling with `heaptrack` or
`cargo flamegraph` — would surface the highest-impact opportunities together
rather than catching them one at a time in reviews.

**4. OCaml compiler error infrastructure is uneven.** The diagnostic system is
excellent where it is applied: error codes, source locations, hint text,
structured formatting. But `failwith` survives at two user-reachable sites in
expander.ml (OcN2), the dimcheck/validate passes emit all diagnostics with
`no_loc` (OcM2, deferred), and 73 expander sites still use `no_loc` (Ocn1).
These are all instances of the same problem: the infrastructure exists, the
discipline to apply it everywhere has not been enforced. A single pass through
expander.ml and dimcheck.ml converting `failwith` to `Diagnostics.emit` and
threading source locations through would close OcN2, chip away at Ocn1, and
potentially close OcM2.
