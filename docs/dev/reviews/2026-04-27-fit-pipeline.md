---
status: open
date: 2026-04-27
scope: fit pipeline (clean_eval, gating, fit_summary, evidence, config_v2, runner, refine, scout, state, compare, if2)
reviewer: Claude (claude-sonnet-4-6)
prior-reviews:
  - docs/dev/reviews/2026-04-24-full-codebase.md
---

# Fit Pipeline Review — 2026-04-27

Scope: all new and changed files added in the 24 commits since the 2026-04-24
review. Primary focus is the clean-evaluation pipeline
(`clean_eval.rs`, `gating.rs`, `fit_summary.rs`, `evidence.rs`, `config_v2.rs`,
`runner.rs`), plus the affected stage drivers (`scout.rs`, `refine.rs`),
inter-stage handoff (`state.rs`), and standalone comparison command
(`compare.rs`, `if2.rs`). Re-verification of all six open findings from the
prior review (InM1, SiM1, OcN2, Inn1–Inn4, Sin1, Irn1) is included.

---

## Resolution Status

| Code  | Description                                                     | Status          |
|-------|-----------------------------------------------------------------|-----------------|
| InM1  | pgas_grad.rs ungrouped-transition uses `rate <= 0.0`            | Still open      |
| SiM1  | complete_data_loglik allocates inside source-group loop         | Still open      |
| OcN2  | parse_iso_date uses `failwith` for user-reachable error         | Still open      |
| Inn1  | CSMC seed-mixing multipliers undocumented magic constants       | Still open      |
| Inn2  | 0.7 mass-adaptation phase split bare literal                    | Still open      |
| Inn3  | Probability clamp 1e-15 unnamed                                 | Still open      |
| Inn4  | Cholesky regularization 1e-6 unnamed in nuts.rs                 | Still open      |
| Sin1  | substep_ancestors identity Vec allocated per non-obs substep    | Still open      |
| Irn1  | `always_active` serde default silently changes semantics        | Still open      |

---

## Summary

**What the new code does well.** The clean-evaluation pipeline (Proposal 1) is
mathematically correct: `logmeanexp` for combining PF replicates is unbiased on
the likelihood scale; the three-candidate design (FinalIter, TailMeanLastK,
BestInRunIter) adequately covers the IF2 tail; the seed scheme
`seed + chain_id*10_000 + cand_ix*1000 + rep_k` is deterministic and
collision-free within the parameter ranges used. The compound gate (Proposal 3)
correctly implements the SE-aware decibans floor
`max(decibans_thresh, 8 * max(SE) * NATS_TO_DB)`. `ChainResults::winner_theta`
correctly returns the clean-eval winner's theta rather than IF2's noisy argmax,
closing the GH-#16 silent-wrong-answer. `FitState` Phase-3 fields
(`resolved_gate`, `resolved_clean_eval`) and their round-trip tests are correct;
the legacy-file test confirms `None` deserialization. `camdl fit summary`
covers all four formats with a clean test suite.

**Where work is still needed.** Four correctness and design concerns were found
in the new code. The most important is the legacy bridge: `FitConfigV2::to_legacy_toml`
silently discards `clean_eval` and `gate` sub-configs from every `Stage::IF2`
block — users who write per-stage `clean_eval`/`gate` overrides in v2 TOML and
rely on the bridge will get the defaults silently. Second, standalone `camdl if2`
was not updated: it still uses the old `n_eval_particles = min(n_particles, 500)`
argmax, so the 40-nat extraction bias documented in the proposal is not fixed for
that entry point. Third, `refine.rs` hard-codes `1.1` for its own convergence
check rather than reading `GateConfig::a_thresh`, creating a gap between what
the gate reports and what `all_converged` records. Fourth, the `8.0` SE-floor
multiplier in `gating.rs` is duplicated across two functions without a named
constant, which means the proposal's "k=8 recommended default" cannot be changed
in one place.

None of the nine prior open findings were closed in this batch.

---

## Findings

### Major

#### ClM1 — Legacy bridge silently drops `clean_eval` and `gate` from IF2 stages

**File:** `rust/crates/cli/src/fit/config_v2.rs:662`

```rust
Stage::IF2 { chains, particles, iterations, cooling, .. } => {
    let sc = StageConfig {
        chains: Some(*chains),
        particles: Some(*particles),
        iterations: Some(*iterations),
        cooling: Some(*cooling),
        rw_sd_scale: None,
        start_chains: None,
    };
```

The `..` pattern discards `clean_eval` and `gate`. `StageConfig` (the legacy
type) has no fields for either, so they are gone without warning.

A user who writes a v2 TOML with a non-default `clean_eval` or `gate` and whose
tooling calls `to_legacy_toml` (e.g., `camdl fit scout --config fit_v2.toml`)
will silently run with `CleanEvalConfig::default()` (4000 particles, 8
replicates) and `GateConfig::default()` (a_thresh=1.01, decibans_thresh=30.0)
regardless of what they configured. No error, no warning, no log line.

The bridge's purpose is to support the migration window; the risk is real because
the runner still accepts either config format. Fix: detect non-default
`clean_eval` or `gate` values in the `Stage::IF2` arm and return an
`Err(...)` explaining that the v2-only fields are not supported through the
legacy bridge, or add explicit forwarding if the legacy type can be extended.

---

#### ClM2 — Standalone `camdl if2` not updated with clean-eval

**File:** `rust/crates/cli/src/if2.rs`

`camdl if2` is the direct-invocation entry point for IF2. It still computes
the output parameter estimate using `n_eval_particles = n_particles.min(500)`
and picks the argmax over the in-run noisy `if2_perturbed_loglik`. This is
exactly the 40-nat extraction bias that the clean-eval pipeline was introduced
to fix (Proposal 1, GH #16).

Users who run `camdl if2` directly — rather than `camdl fit run` — get the
old biased estimator with no warning. The remediation proposal addresses
`camdl fit scout`/`refine` only; `camdl if2` is structurally the same algorithm
and should either call `run_clean_eval_with_scorer` with a default
`CleanEvalConfig`, or display a prominent `[WARNING] Use 'camdl fit run' for
production estimates; 'camdl if2' does not apply clean-eval de-biasing.`

**Minimum fix:** add an `eprintln!` warning at the top of the `camdl if2` run
function noting that it uses the legacy estimator. **Full fix:** wire in the
clean-eval pipeline with a default `CleanEvalConfig`.

---

### Minor

#### ClM3 — Gate verdict logic duplicated between text and structured output paths

**File:** `rust/crates/cli/src/fit/fit_summary.rs`

The gate verdict rendering (pass/warn/fail classification, threshold comparison,
SE-floor computation) is implemented twice: once in `Formatter::gate_verdict_block`
for the text/md/latex path and once in `stage_report` for the JSON path. These
are coupled to the same `FitState` fields and `GateConfig` thresholds. Any
future change to the gate semantics (e.g., adding a new verdict class) must be
updated in both places. Factor the verdict logic into a pure function returning
a `GateVerdictSummary` struct; both paths render from that.

---

#### ClM4 — `refine.rs` convergence check uses hardcoded `1.1`, not `GateConfig::a_thresh`

**File:** `rust/crates/cli/src/fit/refine.rs:142, 152`

```rust
let all_converged = chain_results.chain_agreement.values().all(|&r| r < 1.1);
// ...
let n_unconverged = chain_results.chain_agreement.values().filter(|&&r| r > 1.1).count();
```

The compound gate uses `gate.a_thresh` (default 1.01). Refine's own
`all_converged` flag — written to `fit_state.toml`, logged, and used in
`write_summary` — uses `1.1`. If a user sets `a_thresh = 1.05` in their v2
TOML, the compound gate will pass at Â = 1.03 but `all_converged` will still
report `true` even at Â = 1.08, which conflicts with what the gate would say.
Conversely if `a_thresh = 1.2`, `all_converged` will fire false positives at Â
between 1.1 and 1.2.

Fix: read the `a_thresh` from the resolved `GateConfig` and use it for both the
`all_converged` flag and the `ConvergenceIncomplete` diagnostic's threshold
field, which currently hardcodes `1.1` implicitly by emitting the raw Â values
with no threshold context.

---

#### ClN1 — SE-floor multiplier `8.0` is a bare literal duplicated across two functions

**File:** `rust/crates/cli/src/fit/gating.rs:162, 236`

```rust
// check_scout_convergence (line 162):
let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;

// format_decibans_spread_verdict (line 236):
let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
```

The proposal (`2026-04-24-if2-scout-findings-remediation.md`) names this
`k=8` and calls it the "recommended default." With it spelled as `8.0` in two
places there is no guarantee they stay in sync. Add:

```rust
/// Multiplier for the SE-aware decibans floor: threshold = max(decibans_thresh,
/// k * max(SE) * NATS_TO_DB). k=8 is the proposal's recommended default —
/// see 2026-04-24-if2-scout-findings-remediation.md §Proposal 3.
const SE_FLOOR_K: f64 = 8.0;
```

and replace both occurrences. This also makes it trivially configurable if a
future `GateConfig` field exposes `se_floor_k`.

---

#### ClN2 — `params_agree` tolerance inconsistent between provenance and stage-report paths

**File:** `rust/crates/cli/src/fit/fit_summary.rs`

`Formatter::provenance_block` uses:
```rust
let rel_tol = v.abs().max(other.abs()).max(1.0);
```

`stage_report::state_matches_final` uses:
```rust
let rel_tol = fv.abs().max(1.0);
```

The first normalizes by the larger of the two values; the second normalizes only
by the state value. For parameters where the state and final values are both
large but differ, the two paths can disagree about whether they "match." The
correct formula is the first (symmetric). Fix: consolidate into a single
`fn params_close(a: f64, b: f64, rel_eps: f64) -> bool` helper and use it
everywhere.

---

#### ClN3 — `render_json` in compare.rs always emits all metric fields, ignoring the `metrics_chosen` filter

**File:** `rust/crates/cli/src/compare.rs`

`render_table` and `render_md` respect `metrics_chosen`:
```rust
if metrics_chosen.contains(&m) { /* emit column */ }
```

`render_json` emits all metric fields unconditionally. A user who passes
`--metrics elpd` to restrict the table output to ELPD will still see all other
fields in `--format json` output, making JSON a different information surface
from the human-readable formats. Fix: filter the emitted JSON fields by
`metrics_chosen`, or document the divergence explicitly with a comment
explaining why JSON always emits all fields (e.g., "JSON is lossless; filtering
is the consumer's responsibility").

---

### Nit

#### TsN1 — No test for `render_json` in compare.rs

**File:** `rust/crates/cli/src/compare.rs`

`render_json` has no unit test. In particular, the `delta_elpd_db` and
`evidence_label` fields it emits are not round-tripped against an expected
fixture. Add a test that constructs a minimal `MatchedHorizon` table,
calls `render_json`, and asserts the expected JSON keys and evidence-label
value.

---

#### TsN2 — No golden test for LaTeX tabular output in fit_summary

**File:** `rust/crates/cli/src/fit/fit_summary.rs`

The LaTeX path (`render_latex`) is exercised by a test that checks column
headers and a few `\\hline` markers, but does not snapshot a complete tabular
block against an expected fixture. Since LaTeX output is consumed by downstream
documents, silent regressions (e.g., an extra `\\` or a missing `\texttt`
wrapper) are not caught. Add a golden-string test using `insta` or an inline
`assert_eq!` against the complete rendered block for a minimal state fixture.

---

#### TsN3 — Gate integration: no end-to-end test covering compound-gate Hard verdict path

**File:** `rust/crates/cli/src/fit/gating.rs`

`check_scout_convergence` has good unit tests for each individual branch, but
there is no test that constructs a `FitState` with both `tail_chain_agreement`
and `chain_clean_logliks` populated, invokes the full gate, and asserts the
correct verdict for each quadrant of (Â pass/fail) × (decibans pass/fail). The
`Hard` verdict branch in particular is only reachable via the Â check, while
`DecibansSpread` is only reachable via the spread check; a test covering both
simultaneously (Â OK but spread fails → DecibansSpread; Â fails → Hard) would
protect against refactors that merge or reorder the two checks.

---

#### ClN4 — Misleading comment about LaTeX `_` escaping in fit_summary

**File:** `rust/crates/cli/src/fit/fit_summary.rs`

The `render_latex` function escapes underscores with `replace("_", "\\_")` and
has a comment stating that `_` is allowed in `\texttt`. The claim is only true
inside a `\texttt{}` argument at math mode boundaries — in tabular cell text
without explicit `\texttt`, `_` is a LaTeX subscript command and must be
escaped. The current code (escaping) is correct; the comment is wrong.
Remove or correct the comment.

---

#### ClN5 — Legacy scout path writes `resolved_gate: None` without a log warning

**File:** `rust/crates/cli/src/fit/scout.rs`

The legacy `camdl fit scout` driver intentionally writes `resolved_gate: None`
to `fit_state.toml` because it predates per-stage `GateConfig`. `camdl fit
summary` will render a "(thresholds unknown)" caveat when it encounters `None`,
which is correct per the proposal. However, there is no `eprintln!` at the point
where `None` is written, so operators running `camdl fit scout` directly have no
indication that the gate config will be absent in the handoff. Add a single
`eprintln!("[note] fit_state: resolved_gate not recorded (legacy scout path — upgrade to 'camdl fit run' for full Phase-3 reporting)")`.

---

## Cross-Cutting Themes

**1. The legacy bridge is a correctness hazard while it exists.** ClM1 is the
third time a "compatibility bridge" path has silently discarded semantically
significant configuration (after similar issues with the v1→v2 estimate spec
migration). The pattern is: a new `..` in a match arm, a `None` default in a
struct constructor, or a missing `else` branch — all legal Rust, all invisible
to `cargo test`, all capable of causing a production fit to run with wrong
parameters. A rule worth adopting: *any code path that accepts configuration
and then ignores some of it must either reject the input with an error or
log a structured warning naming the discarded fields.* The bridge should be
treated as a temporary shim with a removal date, not a permanent API surface.

**2. Standalone entry points must be kept consistent with the pipeline.**
ClM2 identifies `camdl if2` as a diverged entry point that exposes a known-bad
estimator without warning. The same risk applies to `camdl pfilter` and
`camdl simulate` whenever the pipeline adds new correctness invariants: if there
is no mechanism to ensure that all entry points receive updates atomically, the
pipeline gains correctness that standalone commands don't reflect. A solution
is to maintain a one-line `ENTRY_POINTS.md` listing all commands that expose
inference math, with a checkbox for each new pipeline invariant.

**3. Named constants and single-location decisions.** ClN1 (the `8.0`
SE-floor multiplier), InM1 (the `0.0` vs `RATE_EPSILON` threshold), and
Inn2–Inn4 (mass adaptation phase split, NUTS step size, Cholesky regularization)
are all the same class of defect: a magic literal that embodies a design decision
but lives at multiple sites, making the decision implicit and the code harder to
audit. The prior review noted that the batch of constants introduced in April
(LOG_PROB_FLOOR, RATE_EPSILON, RESAMPLE_RNG_STREAM) demonstrates the value of
the pattern. The remaining unlabeled constants in these files represent continued
technical debt at the inference-critical level.

**4. Verdict rendering should have a single source of truth.** ClM3 and ClN2
are both symptoms of the same pattern: rendering logic for a derived quantity
(gate verdict, parameter agreement) is written independently for each output
format. This is fragile by construction. The right abstraction is a pure
intermediate type (`GateVerdictSummary`, `ParamAgreement`) that is computed once
and rendered by each format. Any format-specific logic should be limited to
serialization, not computation.
