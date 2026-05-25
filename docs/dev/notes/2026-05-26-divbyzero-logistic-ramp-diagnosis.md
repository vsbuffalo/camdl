# gh#81 Phase 1 diagnosis — DivByZero on logistic-ramp model

Date: 2026-05-26
Issue: gh#81
Branch: gh81-divbyzero-diagnostic (off worktree-agent-a8eb2d488cfbd2a5a /
gh80-pgas-event-density at 00785db)

## Class

**code-vs-code** — the rate-evaluator's NaN-detection path mis-labeled
the failure as DivByZero, hiding the actual upstream fault. The classifier
was the load-bearing wrong: a generic NaN sentinel was promoted to a
specific-sounding `DivByZero` error, blaming the rate expression instead
of naming the actual mechanism (a NUTS leapfrog step or a downstream
NaN cascade through gradient evaluation).

## Reproducer

```
cd /Users/vsb/projects/work/camdl-book/guide/fitting/seed-timing
camdl fit run fits/wa_weak_deaths_soft_pgas_v2.toml \
  --seed 1 --stage posterior --progress plain
```

Pre-fix output (verified at HEAD 00785db before the Phase 2 patch):

```
error running pgas stage 'posterior':
  pgas chain 1 error: numerical collapse (DivByZero) in rate expression at t=-101
```

## What I expected to find (from the upstream hypothesis)

The issue body proposed two possible mechanisms:

1. *Sigmoid-specific overflow*: `1/(1+exp(-(t-t_rep)/w_rep))` underflows
   when `exp(huge)` ≫ 1, producing a 0 denominator somewhere
   downstream. Hypothesised because the user pointed at the logistic
   ramp in `let rho_t = ...`.

2. *Gradient-side DivByZero*: `eval_resolved_deriv` line 466 has
   `Div => if b == 0.0 { 0.0 } else { (da*b - a*db) / (b*b) }`. At
   extreme `θ` NUTS proposes mid-leapfrog, intermediate gradient
   values could go through 0/0 forms producing NaN that then cascades
   into a literal 0.0 denominator.

Neither was load-bearing.

## What I actually found

I added env-var-gated diagnostic prints at the two suspect sites
(`propensity.rs:418-422`, `resolved_expr.rs:272-276`, `:317-320`) —
gated on `CAMDL_GH81_DIAG=1` — then re-ran the repro. The diagnostic
prints captured the rate expression's `ResolvedExpr` debug structure,
the param name→value table, and the compartment counts at every NaN
event. Output: `/tmp/gh81-diag-stderr.log` (~35 MB / 1.6M NaN events,
killed mid-warmup once the mechanism was clear).

The trace shows:

```
GH81-DIAG eval_propensities NaN: transition='infection' t=-101 dt=0.5
GH81-DIAG rate_expr=BinOp { op: Div, left: BinOp { op: Mul, left:
  BinOp { op: Mul, left: Param(0), right: IntPop(0) }, right: IntPop(2)
  }, right: IntPopSum([0, 1, 2, 3, 4, 5]) }
GH81-DIAG param N0 = 7600000
GH81-DIAG param beta = NaN
GH81-DIAG param gamma = NaN
GH81-DIAG param gamma_d = NaN
GH81-DIAG param ifr = NaN
GH81-DIAG param k = 20
GH81-DIAG param k_d = 20
GH81-DIAG param n_seed = NaN
GH81-DIAG param rho_max = NaN
GH81-DIAG param sigma = NaN
GH81-DIAG param t_rep = NaN
GH81-DIAG param tau = NaN
GH81-DIAG param w_rep = NaN
GH81-DIAG state S = 7600000
GH81-DIAG state E = 0
GH81-DIAG state I = 0
GH81-DIAG state R = 0
GH81-DIAG state P = 0
GH81-DIAG state D = 0
```

**Every estimated parameter is NaN at the failure point.** The rate
expression `beta * S * I / N` is innocent — its denominator is finite
(`N = 7600000`), `I = 0` means the rate is mathematically `0`, but NaN
inputs propagate (NaN * 0 = NaN in IEEE-754). The NaN-detection guard
at `propensity.rs:418` then returns `SimError::NumericalCollapse {
kind: DivByZero }` — a generic NaN-cascade catch labelled with a
misleading kind. The user sees "DivByZero in rate expression," which
is doubly wrong:

- No division by zero occurred. `N = 7600000`.
- The rate expression is not the fault site. The NaN entered through
  the parameter vector, upstream of any rate evaluation.

## Why the parameters went NaN

The fit's stderr (captured from the WA repro) shows NUTS adaptation
went pathological during the dual-averaging phase:

```
NUTS fully adapted (sweep 300):
  final step_size: 1419365626198.743164
dense mass matrix estimated (sweep 210):
  beta         sd=6622030426559575842440815447113728.000000  (~6.6e33)
  gamma        sd=2.0e50
  ...
  correlations: bet-gam=-1.00, bet-sig=1.00, ...    (all rails ±1)
```

Step size ~1.4e12 × momentum components from a dense mass matrix with
sd up to 1e50 ≫ f64 range. Leapfrog `z_new = z + dt * M^{-1} * p` with
dense matvec mixes positive-Inf and negative-Inf intermediate
contributions, and `Inf + (-Inf) = NaN` (IEEE-754). The leapfrog
returns a NaN-valued `z_new`, the next `log_prob_and_grad` call's
`log_p_new = -0.5 * sum(z_new^2) = NaN`, and the build_tree leaf's
divergence-detector mis-classifies:

```rust
// nuts.rs:338 (pre-fix)
let divergent = (h_new - h0).abs() > delta_max;
```

For `h_new = NaN`, `(NaN - h0).abs() = NaN`, and `NaN > 1000.0` is
**`false`** by IEEE-754 unordered-comparison semantics. So a NaN-energy
proposal is reported as "non-divergent." The doubling tree happily
continues past the corrupted leaf.

The top-level accept criterion at `nuts.rs:286` (pre-fix):

```rust
let accepted = z_proposal != current_z;
```

For a NaN-valued `z_proposal`, `NaN != [finite, ..., finite]` is `true`
(NaN is unequal to everything), so `accepted = true`. The NaN
parameter vector is committed to the chain. The next CSMC sweep calls
`step_one` with these NaN params, `eval_propensities` triggers, and
the user sees the misleading downstream error.

## Why pre-existing safeguards didn't catch it

- `log_prob_and_grad` *does* swallow `SimError::NumericalCollapse`
  into `(-Inf, zero-grad)` (pgas.rs:1721). But by then the proposal
  is already committed — the swallow only affects subsequent
  iterations, not the one that committed NaN.
- `Transform::Log::from_transformed = z.exp().clamp(lo, hi)` and
  `Transform::Logit::from_transformed = lo + p*(hi-lo)` are
  finite for `±Inf` z (they clamp to bounds), but **`NaN.exp() = NaN`
  and `NaN.clamp(lo, hi) = NaN`** — clamp does not handle NaN.
- The slice-NUTS indicator `log_slice <= -h_new` also returns false
  for NaN h_new, so the multinomial proposal selection happens to
  drop NaN leaves from n_valid — but that's a happy side-effect, not
  an enforced invariant. A future refactor that changes the slice
  branch (or where n_valid defaults to 1) would silently regress.

## Fix (Phase 2, what's now committed)

Two independent fixes, layered:

### (a) Structured `NonFiniteParameter` diagnostic — ac7be6d

`eval_propensities` now checks all parameter values for `is_finite()`
**before** rate evaluation runs. Non-finite param → return
`SimError::NonFiniteParameter { name, value, t }` early, naming the
offending parameter and its NaN/Inf value. The error message is
multi-line and names the actual upstream causes (NUTS / PMMH
proposals) and the user-actionable remedies.

The new variant classifies as `is_per_particle_recoverable() = true`,
so PGAS / PMMH proposal mechanisms can reject and continue.

The existing generic `NumericalCollapse { DivByZero }` cascade path
stays in place for legitimate literal-zero denominators in
well-conditioned rate expressions — the structured error is an
upstream-detection layer, not a replacement.

### (b) NUTS-side NaN safety net — this commit

Two narrow changes to `nuts.rs`:

- `build_tree` leaf at depth 0: `divergent = !h_new.is_finite() ||
  (h_new - h0).abs() > delta_max`. Non-finite energy is **always**
  divergent. The dual-averaging step-size adaptation then receives
  `accept_prob = 0.0` for the divergent leaf and shrinks `eps`,
  which is the canonical NUTS response to numerical pathology
  (Hoffman & Gelman 2014 §5.1.2).

- `nuts_step` accept boundary: `accepted = proposal_finite &&
  z_proposal != current_z`, and when `proposal_finite = false` the
  result returns `current_z` and `current_log_p` instead of the
  corrupted proposal. Callers reading `result.params` blindly (e.g.
  Welford updates for dense mass matrix estimation) get a usable
  f64 vector even in the pathological regime.

### (c) NOT implemented — and why

The issue body's option (b) — stable-sigmoid pattern detection —
turned out to be irrelevant. The sigmoid `1/(1+exp(-x))` is
numerically fine at `|x| = 210`: `1 + exp(210) ≈ 2.65e91`, finite,
no zero denominator. The rate evaluator was never the fault. (b1)
and (b2) deferred indefinitely; they would have been a real fix to
a non-existent problem.

The issue body's option (c) — `is_finite()` guard at each BinOp
output in `eval_resolved_deriv` — would be defensive but the
mechanism doesn't pass through gradient eval first. The Phase 1
trace shows the eval_propensities NaN cascades through value eval
once params go NaN; gradient eval is downstream of the same NaN
params and produces its own NaN. A finiteness gate in
`eval_resolved_deriv` would catch the symptom, not the cause. The
upstream `is_finite()` check on `params[]` at `eval_propensities`
entry (committed in ac7be6d) catches the cause directly and is
both cheaper (5 finiteness checks per call vs hundreds of intermediate
finite-checks across the BinOp tree) and more diagnostic (it names
the parameter, not a deep-nested intermediate expression).

## Tests (TDD red-then-green)

New: `rust/crates/sim/tests/nuts_nan_safety.rs` — 4 tests pinning
the NUTS-side invariants:

- `nuts_step_flags_nonfinite_energy_as_divergent` — RED first
  (verified: the divergence flag was false for a NaN-grad
  log_prob_and_grad), then GREEN after the build_tree fix.
- `nuts_step_rejects_nan_param_proposal` — passed at red because
  the slice-indicator happily-accident drops NaN proposals from
  n_valid; the test pins the invariant explicitly so a future
  refactor can't regress it silently.
- `nuts_step_rejects_when_log_p_is_neg_inf_at_proposal` — the
  steady-state behavior after the gh#81 structured-error fix
  returns `(-Inf, zeros)` for NaN params; NUTS must reject
  these proposals so the chain doesn't get stuck at a -Inf state.
- `nuts_step_accepts_finite_proposals_on_clean_target` — sanity:
  the fix must not over-reject on a well-conditioned 2D Gaussian.

Pre-existing tests pass byte-identical:

- `cargo test --release -p sim --test gradient_check` — 4 ok
  (gradient_vs_FD, nuts_invariance_gaussian, nuts_dense_mass_matrix,
  nuts_target_gradient).
- `cargo test --release -p sim --test gradient_check_obs` — ok.
- `cargo test --release -p sim --test gradient_check_overdisp` — ok.
- `cargo test --release -p sim --test pgas_event_density` — ok.
- `cargo test --release -p sim --test pgas_obs_overdisp_smoke` — ok.
- `cargo test --release -p sim --test pgas_resume` — ok.
- `cargo test --release -p sim --test pgas_tempering` — ok.
- `cargo test --release -p sim --test pmmh` — ok.
- `cargo test --release -p sim --test nonfinite_param` — 5 ok
  (the ac7be6d tests).

The 4 pre-existing `time::tests::*_panics_in_debug` failures in
`sim --lib` are noted as fine to ignore (release-mode-only,
unrelated to this work).

## Real-data smoke

Repro with the new fix runs cleanly through warmup. The `survey_top_k`
init method (which the toml was switched to during the gh#81 work)
needs `--survey-path` on the CLI; once supplied, the fit proceeds
without the chain-killing DivByZero. The user's command line:

```
camdl fit run fits/wa_weak_deaths_soft_pgas_v2.toml \
  --seed 1 --stage posterior --progress plain \
  --survey-path results/surveys/seir_wa_weak_deaths_soft-30327dff
```

For an LHS-init equivalent run the chain-1 stderr would previously
have looked like:

```
error running pgas stage 'posterior':
  pgas chain 1 error: numerical collapse (DivByZero) in rate expression at t=-101
```

Post-fix, the equivalent failure mode (which now requires both the
NUTS step size to misadapt AND for the chain to ignore the existing
divergent flag) instead surfaces as:

```
parameter `beta` is non-finite (value: NaN) at t = -101.
This is upstream of rate evaluation — a NUTS leapfrog step or PMMH proposal
produced a NaN/Inf parameter, which would then propagate into every rate
expression that references it. The error is in the proposal mechanism, not
in the rate expression. The chain rejects this proposal and continues; …
```

And with the NUTS-side fix, the proposal is rejected at the NUTS
boundary before any non-finite param ever reaches the rate evaluator.

## Marked: inference vs verified

- *Verified* (paste from `/tmp/gh81-diag-stderr.log`, lines 143-163):
  every estimated parameter was NaN at the eval_propensities failure
  site. The rate expression and the compartment counts were
  uncorrupted.
- *Verified* by inspection of `nuts.rs:338`: pre-fix `divergent =
  (h_new - h0).abs() > delta_max` is `false` for NaN h_new
  (IEEE-754 semantics).
- *Verified* by inspection of `nuts.rs:286`: pre-fix `accepted =
  z_proposal != current_z` is `true` for NaN z_proposal
  (NaN ≠ anything in Rust's PartialEq).
- *Inferred* (not directly traced to a leapfrog step): the precise
  arithmetic that converted `Inf` momentum into NaN z is most likely
  `Inf + (-Inf)` in the dense matvec `m_inv_times`. The trace doesn't
  show the in-flight matvec; what's observed is that params are NaN
  by the time `eval_propensities` sees them, and that NUTS step
  adaptation went to 1.4e12 × 1e50 ~ 1e62 ≫ f64 range. The
  fix-in-this-commit assumes "any path that produces non-finite
  proposals is wrong" and rejects defensively; the specific
  arithmetic doesn't change the fix.

## Next

The structural fix is in place. The follow-up bigger lift is the
joint-posterior pathology itself — the fit's mass matrix has all
correlations rail-locked to ±1.0 because beta/gamma/sigma are nearly
linearly dependent on this dataset, which is the real reason NUTS
struggles. That's a model-and-data issue (the WA seed-timing problem
is ridge-y), not a runtime bug, and tracking it belongs in a
separate `docs/dev/notes/` entry on the camdl-book chapter rather
than as a follow-up to gh#81.
