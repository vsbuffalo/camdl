# BTRS binomial sampler review

Date: 2026-08-31 Scope: the gh#747 arc on `main` at `ba3c3aa8` —
`rust/crates/sim/src/rng.rs` (+1036 lines: `BtrsHat`, `btrs_binomial`, the
`BinomialAlgorithm` seam, and both test modules),
`rust/crates/sim/benches/binomial_ab.rs`, and the two accompanying docs. Method:
four independent review agents, one per lens — rejection-sampler exactness,
dispatch/boundary conditions, test mutation-resistance, and
containment/run-identity — each given the code cold and asked to demonstrate
rather than assert.

Verification status: every finding below marked CONFIRMED was independently
re-reproduced before this document was written, not taken from the reviewing
agent. Specifically re-verified here: the `binv_inverse_cdf` overflow (with the
function extracted verbatim from `ba3c3aa8` and driven directly), the BTPE
`n >= 2^63` abort (against `rand_distr` 0.4.3 from the lockfile), the Stirling
index shift (against `math.lgamma`), the squeeze predicate duplication (by
reading both call sites), the `DOMAIN` upper bound, the containment scan root
(by `git ls-tree`), and the `n·eps` growth law the proportionality figures rest
on. Items marked UNVERIFIED are the reviewing agents' measurements, reported
because they are reproducible from the stated method, not because they were
re-run.

One reviewer claim was checked and **rejected**: a reported sign error at
`rng.rs:21`. The term `(n+1)·ln((n−m+1)/(n−k+1))` is positive for `k > m`, so
dropping it lowers `log_bound` and the comment's `e^−(k−m)` is correct.

Prior context: gh#510 (unbounded BINV loop), gh#525 (BTPE panic at
`n > i32::MAX` with small `n·p`), gh#530 (BINV doc/tail repair), gh#548 (the
`n = u64::MAX` overflow, open). Proposal this reviews against:
`docs/dev/proposals/2026-08-24-faster-binomial-sampler.md`. Profiling note:
`docs/dev/notes/2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`.

## Standing

The BTRS algorithm itself is correct. Three reviewers converged on that from
different directions, including a from-first-principles derivation of the
acceptance test checked against a 40-digit binomial pmf. What is weak is the
**fence around it** and the **tests that pin it**: mutation testing found six
mutations the 24-test suite does not catch, two of which produce multi-percent
density errors while leaving every test green.

Nothing here is reachable today. `DEFAULT_BINOMIAL = BinomialAlgorithm::Btpe`
(`rng.rs:383`), and containment holds — every caller of `set_binomial_algorithm`
is in `#[cfg(test)] mod btrs_tests` or in `benches/`. All of it should land
before the typed stage field makes BTRS selectable.

Two findings are **pre-existing bugs in a different code path**, surfaced by
this review rather than introduced by it. They are numbered separately below and
say so.

---

## Findings — the BTRS arc

**1. The squeeze comparison the sampler runs is not the expression the
domination proof certifies.**

- Severity: **high**
- Where: `rng.rs:332` (sampler) against `rng.rs:1105`–`1121` (`worst_ratios`)
- Evidence: `worst_ratios` certifies the squeeze by
  `if us >= SQUEEZE_US_MIN { worst_squeeze.max(h.v_r - v) }`, where
  `v = h.accept_ratio(us, k)`. The sampler compares `v <= h.v_r` at its own call
  site. Both read the `v_r` field; nothing ties the two comparisons. Mutating
  the call site to `v <= h.v_r * 1.10` leaves the **entire 24-test suite green**
  at a measured total variation of `4.6e-3`, max relative pmf error 2.5% at
  `(1000, 0.5)` and **4.3% at the province regime** `(6.3e6, 3.05e-5)`.
  Detection threshold lies between ×1.10 (green) and ×1.20 (red, and only via
  chi-square/moments — never via the sweep). CONFIRMED as to mechanism
  (duplication read directly at both sites); the damage figures are UNVERIFIED.
- Independent: yes. Lands alone.
- Disposition: filed. `SQUEEZE_US_MIN` is already a shared named constant
  precisely so this class of drift cannot happen — its own docstring
  (`rng.rs:41`–`46`) says so — and `BtrsHat` exists so the proof evaluates the
  shipped expressions. The discipline was applied to every part of the hat
  except the comparison itself. Fix: give `BtrsHat` a
  `squeeze_accepts(v, us) -> bool` and have the sampler and `worst_ratios` both
  call it.

**2. `BTRS_MAX_N` is a correctness fence that no test pins.**

- Severity: **high**
- Where: `rng.rs:39`
- Evidence: raising `BTRS_MAX_N` from `1e12` to `1e15` — or to `1e18` — leaves
  all 24 tests green. The constant's own docstring records `sup V = 1.06` at
  `1e15` and `2.7e46` at `1e18`, i.e. the hat stops dominating and the sampler
  silently returns a wrong distribution.
  `btrs_de_selects_itself_above_its_max_n` (`rng.rs:1380`) uses the constant
  symbolically, so it follows the mutation. The lower fence has two guards
  (`rng.rs:550` and the inverted
  `the_hat_stops_dominating_below_the_routing_threshold` at `rng.rs:1594`); the
  upper fence has neither. UNVERIFIED (the mutation was run by the reviewing
  agent in a detached worktree, with a pattern-count assertion so a
  silently-non-matching edit could not masquerade as a miss).
- Independent: entangled with #3 — both are about this constant and its
  supporting sweep, and a single change should address them together.
- Disposition: filed with #3. Fix: the mirror of `rng.rs:1594` — assert the hat
  fails to dominate above the fence — plus a `DOMAIN` cell at `BTRS_MAX_N`
  itself.

**3. `BTRS_MAX_N`'s derivation measures the criterion that does not bind.**

- Severity: **high**
- Where: `rng.rs:17`–`38`
- Evidence: BTRS exactness needs three conditions — `V <= 1` (domination),
  `V >= v_r` where the squeeze fires, and `exp(log_bound(k)) ∝ pmf(k)`
  (proportionality). The fence is derived from the first: "the worst domination
  margin anywhere in the routed domain is 0.22%, so requiring `n·eps` to sit a
  decade inside it gives `n <= 2.2e-4/eps ≈ 1e12`." The third binds ~2000×
  lower. Measured spread of `log_bound(k) − ln pmf(k)` against a 40-digit
  reference: `1.93e-9` at `n = 8.75e6`, `2.22e-7` at `1e9`, `2.21e-4` at `1e12`.
  The suite's own bar — `assert!(spread < 1e-7)` in
  `log_bound_is_proportional_to_the_exact_pmf` (`rng.rs:1485`, bar at line
  `1524`) — is crossed at **n ≈ 4.6e8**. CONFIRMED that the bar is `1e-7` and
  that no `DOMAIN` cell exceeds `n = 8.75e6`; the spread figures are UNVERIFIED
  but lie on the stated `n·eps` growth law to within 1% when extrapolated from
  the reviewer's own `8.75e6` anchor (ratios 1.01, 0.99, 1.00), which is the
  internal consistency check that makes them credible.
- Severity qualifier, stated plainly: at `n = 4.6e8` — reachable, a national
  susceptible compartment or an `n_examined` column — the implied total
  variation is `~1e-8`, harmless. The `2e-5` figure needs `n ≈ 1e12`, which is
  not an epidemiological population. **The defect is that the fence's stated
  justification does not support the fence**, not that a fit is wrong today.
- Independent: entangled with #2.
- Disposition: needs a decision — see Design calls, D1.

**4. `the_two_acceptance_forms_agree` has a ±10% dead band around the acceptance
boundary.**

- Severity: **medium**
- Where: `rng.rs:1437`
- Evidence: it probes `v` at `ratio × {0.5, 0.9, 1.1, 2.0}`, so it certifies
  agreement only outside `f ∈ (0.909, 1.111)`. Injecting a 9% bias into
  `slow_accepts` alone passes the suite; 12% is caught. Measured damage at 1.09:
  TV `3.5e-3`, max relative pmf error 3.5% at `(6.3e6, 3.05e-5)`. Below ×1.007
  the mutation is genuinely null (the squeeze clamp makes small boundary shifts
  a uniform rescaling), so the real blind window is roughly
  `f ∈ [1.007, 1.111]`. UNVERIFIED.
- Independent: yes.
- Disposition: filed with #1. Fix: add scales `0.999, 1.001` around the boundary
  rather than skipping only `|scale − 1| < 1e-9`.

**5. `in_support`'s edge comparisons are unpinned in both directions.**

- Severity: **medium**
- Where: `rng.rs:210`
- Evidence: `k > self.count` → `k >= self.count` (the sampler can then never
  return `k = n`) and `k < 0.0` → `k <= 0.0` (never returns `k = 0`) both pass
  the whole suite. `draws_stay_in_support_including_the_squeeze_return`
  (`rng.rs:1176`) asserts only `d <= n`, and the chi-square pools those cells
  away (expected < 5 at 200k draws). Under the `p > 0.5` reflection a lost
  `k = n` becomes a lost `k = 0`, and at `(200, 0.05)`-shaped cells
  `P(K = 0) ≈ 3.5e-5` is real mass. Nothing asserts the reachable support is
  `[0, n]` inclusive. UNVERIFIED.
- Independent: yes.
- Disposition: filed with #1.

**6. The gate never runs this module under `--release`, so every `debug_assert!`
in the sampler is unexercised in the build configuration fits use.**

- Severity: **medium**
- Where: `rng.rs:301`–`302`, `rng.rs:328`, `rng.rs:131`
- Evidence: deleting the hoisted `in_support` guard at `rng.rs:324` fails 5
  tests in debug, but only through `debug_assert!`s — no test _assertion_ sees
  it. Under `cargo test --release -p sim --lib rng` all 24 pass in 0.17 s. The
  candidates are in fact rejected downstream (`log_bound` takes `ln` of a
  negative ratio and returns NaN), so the deviation is near-null in release —
  but that means the docstring's claim at `rng.rs:1176`, that the guarantee is
  "tested rather than trusted", describes something the test does not do.
  UNVERIFIED.
- Independent: yes, though the cheapest fix overlaps #5 (an assertion on the
  reachable support would catch the guard's removal in either profile).
- Disposition: filed with #1.

**7. The domination sweep is a 100,000-point lattice cited as a proof, and it
systematically understates `sup V`.**

- Severity: **low**
- Where: `rng.rs:1105`–`1106`, and the doc references to it at
  `rng.rs:164`–`168` and `rng.rs:236`–`238`
- Evidence: `sup V` over each `k`-plateau sits at that plateau's smallest `us`,
  so a finite lattice always lands short. Refining to `STEPS = 2e8`:
  `(23, 0.4583)` 0.99749621 → 0.99777508 (understatement `2.79e-4`);
  `(100, 0.1)` `2.72e-4`; `(20, 0.5)` `1.54e-4`. The certified margin at the
  tightest cell is `2.225e-3`, so the lattice error is 12.5% of it — eight-fold
  headroom, no violation hidden today. UNVERIFIED.
- Independent: yes. Documentation change only.
- Disposition: filed with #1. The sweep should describe itself as a sample with
  a known bias direction, not as a proof.

**8. `DOMAIN` has no cell between `n = 8.75e6` and `BTRS_MAX_N = 1e12`.**

- Severity: **low**
- Where: `rng.rs:1094`–`1101`
- Evidence: five decades of routed `n` are swept nowhere. CONFIRMED — the
  largest cell is `(8_750_000, 2.2e-5)`. Two reviewers independently scanned the
  gap for domination and found none: worst `sup V = 0.9955` at `(7.9e11, 0.325)`
  and `0.9955` at `(1e12, 0.5)`. So the low-end adversarial cells were well
  chosen and the gap is entirely at the high end — where, per #3, the criterion
  that actually binds is not the one the sweep measures.
- Independent: entangled with #2 and #3.
- Disposition: filed with #2/#3.

**9. `critical(df)` is labelled "≈6σ" but delivers 4.1–5.0σ.**

- Severity: **low**
- Where: `rng.rs:1019`
- Evidence: `df = 18` → `critical = 54.00`, true chi-square 6σ quantile 79.65,
  `P(chi2 > 54.00) = 1.84e-5`, implied 4.13σ; `df = 54` → 4.64σ; `df = 130` →
  4.98σ. The values are correct for the stated normal approximation; only the
  label overstates. Direction is benign (stricter than advertised for the fit
  test, more permissive than a true 6σ for the canary) and seeds are fixed, so
  nothing flakes. UNVERIFIED.
- Independent: yes. Documentation change only.
- Disposition: filed with #1.

**10. Four documentation defects in the new code.**

- Severity: **low**, with one exception noted below
- Where and evidence:
  - `rng.rs:130` — `stirling_approx_tail(k)` returns `delta(k+1)`, not
    `delta(k)`; the docstring describes `delta(k)`. CONFIRMED: every `TAIL[k]`
    matches `delta(k+1)` to 1e-14, `TAIL[0] = 1 − ln(2*pi)/2 = delta(1)`, and
    `delta(0)` is undefined because `sqrt(2*pi*0) = 0`. **This one is a trap,
    not a nit**: `log_bound` is derived under the shifted convention, and
    "correcting" the code to match the doc blows the density error from `6e-11`
    to `0.038` at `(20, 0.5)`. The fix is to correct the sentence _and_ say the
    shift is deliberate.
  - `rng.rs:287`–`288` — claims flipping first keeps `k > n` unreachable at
    `n > 2^53`. It is `BTRS_MAX_N` that fences this; `in_support` compares
    against `n as f64`, which rounds up above `2^53`. Directly contradicts
    `rng.rs:318`–`326`. One of the two should go.
  - `rng.rs:634`–`636` — says the BTPE fallback preserves "its own huge-`n`
    fallback." `rand_distr` has no such fallback; it has
    `assert!(x < i64::MAX as f64)` (`binomial.rs:80`), and `Err(_)` catches
    `Result`, not `panic!`. See finding 12.
  - `rng.rs:249`–`252` — a clause was lost in an edit; the sentence does not
    parse.
- Independent: yes, all four.
- Disposition: filed with #1.

**11. `active_binomial_algorithm()` is effectively dead, and does not apply the
`BTRS_MAX_N` de-selection.**

- Severity: **low**
- Where: `rng.rs:392`
- Evidence: CONFIRMED by `git grep` — exactly one caller, `rng.rs:1336`, in the
  same file's own test module, asserting the value of `DEFAULT_BINOMIAL`. That
  says nothing about "the sampler that ran equals the one that was hashed,"
  because nothing is hashed yet. Separately, it returns the thread-local
  selection without the `n > BTRS_MAX_N` de-selection applied at `rng.rs:640`,
  so above the fence it reports `Btrs` while `Btpe` ran.
- Independent: yes.
- Disposition: keep only if the step-2 PR adds the real caller — an assertion,
  at the point the hashed `binomial` value is resolved, that it equals what the
  sampler will use. If step 2 lands without wiring it, delete it.

---

## Findings — pre-existing, different code path

These two are **not defects in the gh#747 arc**. `binv_inverse_cdf` and the
`((n + 1) as f64)` are present at `21b47ce6` — the commit before the BTRS work —
and the arc did not touch those lines. The BTPE arm is likewise unchanged. They
are recorded here because this review is what surfaced them.

**12. Nothing bounds `n` from above in `StatefulRng::binomial`, and both
downstream branches fail differently.**

- Severity: **high**
- Where: `rng.rs:539`–`541` (the guard block), with the two failures at
  `rng.rs:87` (BINV) and `rng.rs:653` (BTPE)
- Evidence: CONFIRMED, both reproduced with the function extracted verbatim from
  `ba3c3aa8`. The guard block bounds `p` from both sides and `n` from below;
  nothing bounds `n` from above.
  - BINV, `n = u64::MAX`, `p = 5e-19` (`n·p = 9.22`, routes to BINV): `n + 1`
    wraps to 0, so `a = 0`, so `r *= 0/x − s` is negative and the walk returns
    on its first iteration.

    ```
    n=u64::MAX    n*p=9.2234  draws=[1,1,1,1,1,1,1,1,1,1,1,1]  mean=1.0000
    n=u64::MAX-1  n*p=9.2234  draws=[4,6,7,7,8,9,9,10,11,12,13,15]  mean=9.2500
    ```

    The return is 0 with probability `exp(−n·p)` and **1** otherwise — `9.87e-5`
    versus `0.99990` at this cell.
  - BTPE, `n = u64::MAX`, `p = 1e-17` (`n·p = 184.5`, routes to BTPE): aborts
    the process from inside the dependency, and probabilistically — three draws
    returned before it fired.

    ```
    draw 0 = 186    draw 1 = 200    draw 2 = 166
    thread 'main' panicked at rand_distr-0.4.3/src/binomial.rs:80:
    assertion failed: x < (core::i64::MAX as f64)
    ```
- Reachability: `f64 -> u64` casts saturate, so
  `f64::INFINITY.round().max(0.0)
  as u64` is exactly `u64::MAX`.
  `obs_args_nan` (`obs_model.rs:535`) screens NaN only; `+inf` passes. The three
  cast sites are `obs_model.rs:614` (Binomial denominator, an arbitrary resolved
  expression — a data column, a compartment sum, a ratio), `obs_model.rs:632`
  (BetaBinomial), and `compiled_model.rs:2265` (`InitCountLaw::Binomial`).
- Independent: yes — independent of every BTRS finding above, and of #13 only in
  that one guard closes both.
- Disposition: the BINV half is gh#548, open, `effort/S`. **Its symptom
  description is wrong** — it says "the walk returns 0"; it returns 1 with
  probability `1 − exp(−n·p)`. It also cites `obs_model.rs:499`, now
  `obs_model.rs:614`. The BTPE half appears unfiled; gh#525 covered the
  large-`n`/small-`n·p` complement and closed it by routing to BINV.

**13. The diagnostic that would surface finding 12 can never fire.**

- Severity: **medium**
- Where: `rng.rs:653`–`658`, with the dead report at `eval_stats.rs:151`
- Evidence: after the guards, `p ∈ (0, 1)` and finite, so `p.clamp(0.0, 1.0)` is
  a no-op on every reachable input and `Binomial::new` errs only on conditions
  already excluded. `Err(_)` is therefore unreachable, `inc_binomial_fallback()`
  never increments, and `eval_stats.rs:151`'s `if self.binomial_fallback > 0`
  gate never opens. The degeneracy counter reads clean while the draws are
  wrong. UNVERIFIED as to unreachability across every input; CONFIRMED by
  inspection of the guard block.
- Independent: yes, but the natural fix is the same guard as #12.
- Disposition: filed with #12.

---

## Findings — containment and run identity

**14. The containment scan is rooted one directory too deep.**

- Severity: **medium**
- Where: `rng.rs:1300`–`1304`
- Evidence: CONFIRMED. `CARGO_MANIFEST_DIR.parent()` is `rust/crates`, but the
  workspace root package owns three `.rs` files outside it — `rust/src/lib.rs`,
  `rust/tests/external_validation.rs`, `rust/tests/golden_deser.rs`
  (`git ls-tree -r --name-only main -- rust/ |
  grep '\.rs$' | grep -v '^rust/crates/'`).
  A `set_binomial_algorithm` call added to `rust/src/lib.rs` — a `pub fn` in a
  lib target, non-test by the scan's own criterion — leaves the test green; the
  same call in `rust/crates/cli` turns it red, so the mechanism works and only
  the root is wrong. Secondary: `ALLOWED` whitelists whole files, so a
  production call inside `rng.rs` itself (1615 lines) is permitted.
- Independent: yes.
- Disposition: filed with #1. Fix: root at `crates.parent()` and drop the
  `crates/` prefix from `ALLOWED`.

**15. The obvious thread-local repair would run two samplers inside one PGAS
sweep.**

- Severity: **high** (as a trap for the step-2 PR; nothing is wrong today)
- Where: `rng.rs:376`–`382` (the comment), against `pgas.rs:1587` and
  `pgas.rs:2453`
- Evidence: CONFIRMED by reading both paths. The comment says a per-chain
  thread-local would reach "almost none" of the draws. It is worse than that:
  `simulate_reference_on_grid` (`pgas.rs:1587`) is a plain sequential `for` loop
  calling `step_one` at `pgas.rs:1633`, plus `initial_state_draw` at
  `pgas.rs:1600` — both on the chain thread — while the particle ensemble draws
  on rayon workers (`pgas.rs:2453`, `step_one` at `:2474`). So the reference
  trajectory would use the selected sampler and the ensemble mostly the default,
  split by work-stealing and therefore not reproducible run to run. That is a
  worse failure than uniformly-wrong, and it is the case a reviewer would most
  plausibly wave through.
- Independent: yes, and it constrains the step-2 design rather than requiring a
  change now.
- Disposition: carry into the step-2 review. `gate_pgas_thread_invariance` would
  catch it, but only because `RAYON_NUM_THREADS=1` makes the override uniform —
  name that explicitly in the step-2 test plan rather than relying on it.
  UNVERIFIED (the gate was not run).

**16. The proposal's compile-enforcement argument does not apply to
`Stage::identity_payload`.**

- Severity: **medium**
- Where: `docs/dev/proposals/2026-08-24-faster-binomial-sampler.md`, the "The
  split already exists, compile-enforced" bullet, against `config_v2.rs:1043`
- Evidence: CONFIRMED. The proposal cites `runid-derive`'s guarantee that "a
  field whose type is not `ContentAddressed` is a compile error — you cannot
  forget to make an input hashable." But `Stage` derives
  `Debug, Clone, Deserialize, Serialize` — **not** `RunInput` — and
  `identity_payload` routes through `cas::serialize_minus`. Include-by-default
  here is a serde property with no compile enforcement: a `#[serde(skip)]`, or a
  type whose `Serialize` collapses both variants to the same JSON, drops the
  field from the key silently. The proposal's conclusion is right; the reason it
  gives for trusting it is not.
- Independent: yes.
- Disposition: correct the proposal before step 2 is implemented against it.

**17. `BinomialAlgorithm` carries no serde derives, and the re-key inventory
understates the blast radius.**

- Severity: **medium**
- Where: `rng.rs:349`, and the sequencing table in the proposal
- Evidence: CONFIRMED that the enum derives only
  `Debug, Clone, Copy, PartialEq, Eq`, so the typed field cannot be added as-is.
  UNVERIFIED but well-argued: the proposal's table says step 2 re-keys only the
  two `identity_payload` tests and gives `btrs` runs their own addresses — which
  holds only if the field carries `skip_serializing_if` on the default. The
  house style throughout the PGAS/PMMH variants is bare `#[serde(default)]`,
  which affects deserialization only, so an always-serialized field emits
  `"binomial":"btpe"` on every stage, re-keying every existing PGAS leaf and
  invalidating every `resume_state.bin`.
- Independent: entangled with #16 — same document, same step.
- Disposition: needs a decision — see Design calls, D2.

**18. `step_one` is not the only binomial draw site under a PGAS stage.**

- Severity: **medium**
- Where: `chain_binomial.rs:717`, `:734`; `compiled_model.rs:812`, `:2265`;
  `obs_model.rs:614`, `:632`
- Evidence: CONFIRMED by `git grep '\.binomial('`. A stage addressed
  `binomial = "btrs"` whose resolved value reaches only `step_one` would still
  draw its initial state (`compiled_model.rs:812`, called from `pgas.rs:1600`)
  and its posterior-predictive observations (`obs_model.rs:614`, `:632`) from
  BTPE. Separately, `step_one`'s signature change fans out to three call sites —
  `chain_binomial.rs:438` (simulate), `pgas.rs:1633` (reference producer),
  `pgas.rs:2474` (particle propagation) — and the first is shared with
  PMMH/IF2/pfilter/`camdl run`, which have no such stage field.
- Independent: yes; constrains step 2.
- Disposition: carry into the step-2 review. Either thread the value to every
  site under the stage's control, or document the others as deliberately BTPE.

---

## What is sound

Recorded so it is not re-reviewed.

- **The hat and the acceptance test.** Every constant in `BtrsHat::new`
  (`rng.rs:171`–`183`) matches TensorFlow core's `random_binomial_op.cc` `btrs`
  and TFP's `_btrs` character-for-character; all ten `TAIL` entries match TF's
  `kTailValues` digit-for-digit, including index 7 where a known extra-zero typo
  lives. The acceptance test was re-derived from first principles: the hat
  density is `1/(a/us^2 + b)`, the Jacobian cancels exactly
  (`T'(u) = b + a/us^2` on both branches, so each cell integrates to 1), and the
  four Stirling terms carry the right signs and arguments.
  `log_bound(k) −
  ln pmf(k)` is constant in `k` to `3.3e-13` at `(1000, 0.5)`.
- **The squeeze**, with `v` correctly _not_ rescaled — both references compare
  raw `v` against `v_r`, and the squeeze is a sufficient condition for the slow
  test rather than an alternative comparison. Checked exhaustively per `k`-cell
  via closed-form endpoints rather than on a lattice; worst margin `+0.0079` at
  `n ≈ 492, p ≈ 0.0234`.
- **The hoisted `in_support` check** is distributionally correct: out-of-support
  `k` has pmf 0, so a correct rejection sampler must accept it with probability
  0, and redrawing leaves the conditional law unchanged. It is also a no-op on
  the routed domain. TFP already applies the support check to both accept paths,
  so this deviates from the TF core kernel only, not from the reference family.
- **BINV/BTPE routing parity with `rand_distr` 0.4.3.** The flip predicate
  (`binomial.rs:100`) is the identical expression to `rng.rs:624`; both use
  strict `<` on `BINV_THRESHOLD`; at `p == 0.5` both take the no-flip branch.
  Over 2000 draws x 7 cells, zero value mismatches against upstream BINV and the
  _next_ `gen::<f64>()` from each stream is bit-identical — stream position
  aligned, so no golden moves.
- **`binv_inverse_cdf`'s recurrence** is the exact pmf ratio
  (`a/x − s = ((n+1−x)/x)(p/q)`) to `1.3e-16` relative, and the `r <= 0.0`
  branch can only be underflow — `x > n` is tested before the multiply, so the
  zero-factor case at `x = n+1` is unreachable.
- **`chi_square_rejects_a_one_percent_bias` does what its name says.** Realized
  chi-square over critical under a 1% bias: 3.50 in the weakest cell
  `(100, 0.1)`, up to 305. The smallest uniform bias in `p` the fit test would
  catch is 0.043%; the canary, requiring all cells to reject, is limited by
  `(100, 0.1)` at 0.43%.
- **The benchmark measured what it claims.** `binomial_ab.rs:118` sets the
  override and `:123`/`:128` draw in the same function body; `main` at `:211` is
  a sequential loop; the file imports no rayon and spawns no threads. The rayon
  hazard of finding 15 does not touch it, so the 1.48x sampler ratio stands.
- **Containment holds at this commit.** Every caller of `set_binomial_algorithm`
  is in `#[cfg(test)] mod btrs_tests` or in `benches/binomial_ab.rs`; nothing in
  `cli`, `io`, `ir`, `runid`, `numerics`, or `external-harness` names it or
  `BinomialAlgorithm::Btrs`; no feature flag gates it and no `examples/`
  directory exists.
- **Domination in the unswept band.** Two reviewers independently scanned
  `n ∈ (8.75e6, 1e12]` and found no violation — worst `sup V = 0.9955`. It is
  proportionality (#3), not domination, that fails there.

One limit on all of the above: Hörmann (1993) itself is paywalled and was not
obtained, so the transcription comparison is against TensorFlow and TFP, which
are two transcriptions of one algorithm rather than independent sources. That
establishes camdl transcribed faithfully; it does not independently establish
the algorithm. The from-first-principles derivation against a 40-digit pmf
reference is what carries that, and it holds.

---

## Design calls

**D1. How to repair `BTRS_MAX_N` (finding 3).**

The fence is currently safe by accident: it was derived from domination, which
holds to `1e12`, while proportionality — the criterion that actually binds —
crosses the suite's own bar at `n ≈ 4.6e8`.

- **Apply the `ln_1p` repair** the constant's own docstring already names, for
  `log_bound`'s second term. This removes the `n·eps` growth entirely, so the
  fence can legitimately be set by domination and the existing derivation
  becomes correct rather than accidentally-safe. Cost: it changes arithmetic
  that `hat_dominates_and_squeeze_is_valid` and
  `log_bound_is_proportional_to_the_exact_pmf` currently certify, so both must
  be re-run and the `sup V` figures in the docstring re-measured.
- **Lower the fence to `~4.6e8`.** Cheap, but picks a bound off a test threshold
  rather than from the analysis — the same category of mistake as the current
  constant, in the other direction.
- **Accept and document.** State that proportionality degrades to `~2e-4` at the
  fence and that this is tolerated. Cheapest; leaves a documented gap between
  the suite's bar and the routed domain.

Recommendation: the `ln_1p` repair. The docstring already identifies it and says
"fencing first is the conservative order" — that order has now been served, and
the measurement that would have justified deferring it turns out to measure the
wrong quantity. Confidence: **leaning**. What would flip me: if re-running the
domination sweep after the change moves `sup V` anywhere near 1, the repair has
bought a proportionality gain at a domination cost and the fence should simply
drop to `4.6e8` instead.

**D2. Whether the typed `binomial` field is always serialized (finding 17).**

- **Bare `#[serde(default)]`**, matching house style. Emits `"binomial":"btpe"`
  on every PGAS stage, re-keying every existing PGAS leaf and invalidating every
  `resume_state.bin`. The payload then always records which sampler produced the
  run.
- **`skip_serializing_if` on the default.** No invalidation; `btrs` runs get
  their own addresses. But a `btpe` run's stored payload no longer names its
  sampler, so the record cannot distinguish "ran BTPE" from "predates the
  field."

Recommendation: take the full re-key. Re-keying is already authorised for this
arc, pre-1.0 invalidation is cheap, and a stored posterior whose address does
not name its sampler is precisely what the design exists to prevent. Confidence:
**leaning**. What would flip me: an unfinished long PGAS run whose
`resume_state.bin` you actually intend to resume.

**D3. Whether an out-of-range `n` should be a diagnostic rather than a value
(finding 12).**

gh#548 raises this and leaves it open. The denominator usually came from a file
the user can fix, so a silent clamp hides a data error that a named diagnostic
would surface — but `binomial` is called per particle per substep, where an
error path is expensive and a panic is unacceptable.

Recommendation: clamp `n` in the guard block and count it through the existing
`eval_stats` fallback counter (which finding 13 shows is currently dead), so the
fit summary names it once rather than the hot path branching on it. Confidence:
**need you** on whether the fit should additionally _refuse_ — that is a call
about what modellers should be allowed to run, not one the code answers.
