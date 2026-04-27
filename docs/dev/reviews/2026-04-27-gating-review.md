---
date: 2026-04-27
scope: gating subsystem — type map, UX audit, ClM3 fix proposal, code review
reviewer: agent
---

# Gating subsystem review — 2026-04-27

## 1. Type map and data flow

### Types

#### `GateConfig` (`config_v2.rs:533–556`)

```rust
pub struct GateConfig {
    pub a_thresh: f64,       // default 1.01 — max tolerated Â across non-IVP params
    pub decibans_thresh: f64,// default 30.0 — floor on the SE-aware decibans threshold
}
```

`a_thresh` controls Gate 1's hard cutoff. The compound threshold the code
actually uses is `max(decibans_thresh, SE_FLOOR_K * sigma_max * NATS_TO_DB)`:
`decibans_thresh` is a floor, not the raw threshold. Neither field name nor the
TOML comment clarifies that `decibans_thresh` is a floor rather than the
effective threshold. A user who sets `decibans_thresh = 60.0` may not realise
that a problem with large SE could push the effective threshold much higher.

`GateConfig` is `#[derive(PartialEq)]` which enables the equality check in
`to_legacy_toml` (config_v2.rs:654), but is otherwise used only as a plain
value type.

#### `ScoutGateVerdict` (`gating.rs:56–83`)

```rust
pub enum ScoutGateVerdict {
    Ok,
    SoftWarn { param_agreement: Vec<(String, f64)> },
    Hard {
        failing:         Vec<(String, f64)>,  // non-IVP params with Â ≥ a_thresh
        all_structural:  Vec<(String, f64)>,  // all non-IVP Â values
        ivp:             Vec<(String, f64)>,  // IVP Â values (not gated)
        loglik_spread:   f64,                 // hi − lo of chain_logliks
    },
    DecibansSpread {
        delta_db:     f64,      // observed spread in decibans
        threshold_db: f64,      // effective threshold (max of config + SE floor)
        sigma_max:    f64,      // max SE across chains (used for floor)
        chain_logliks: Vec<f64>,// per-chain clean logliks (nats)
    },
}
```

**Variants are mutually exclusive and ordered.** `check_scout_convergence`
returns `Hard` if Â fails before checking the decibans leg; thus a run where
BOTH Â and decibans fail returns only `Hard`. The `DecibansSpread` variant is
only reachable when Â passes. This means a single call to
`check_scout_convergence` can never surface both problems simultaneously; the
user must fix Â first, re-run, then potentially see the decibans failure. This
ordering is intentional per the proposal (the Â gate is cheaper to diagnose)
but is not documented on the enum.

#### `CleanEvalConfig` (`config_v2.rs:504–528`)

```rust
pub struct CleanEvalConfig {
    pub n_particles: usize,   // default 4000 — particles per replicate PF
    pub n_replicates: usize,  // default 8 — independent PF replicates per candidate
    pub combine: CombineMode, // default LogMeanExp
}
```

#### `FitState` gate-related fields (`state.rs:38–103`)

| Field | Type | Serialized | Written by | Read by |
|---|---|---|---|---|
| `tail_chain_agreement` | `HashMap<String, f64>` | TOML | `mod.rs:799` | `gating.rs:113` |
| `ivp_params` | `Vec<String>` | TOML | `mod.rs:800` | `gating.rs:109` |
| `chain_logliks` | `Vec<f64>` | TOML | `mod.rs:803` | `gating.rs:131`, Gate 2 epsilon |
| `chain_clean_logliks` | `Vec<f64>` | TOML | `mod.rs:804` | `gating.rs:157`, `fit_summary.rs` |
| `chain_clean_ses` | `Vec<f64>` | TOML | `mod.rs:805` | `gating.rs:158`, `fit_summary.rs` |
| `resolved_gate` | `Option<GateConfig>` | TOML | `mod.rs:814` | `fit_summary.rs:261–264` |
| `resolved_clean_eval` | `Option<CleanEvalConfig>` | TOML | `mod.rs:815` | `fit_summary.rs` (GateReport only) |

All seven fields are `#[serde(default)]`, so legacy files load without
migration. The `skip_serializing_if` predicates differ by field:
`HashMap::is_empty`, `Vec::is_empty`, and `Option::is_none` — which means an
empty `chain_clean_logliks: []` is omitted from TOML (correct), but the field
is always populated by a modern run.

**`fit_state.toml` written at `mod.rs:817`**, after Gate 2 passes. If Gate 2
fires, `fit_state.toml` is **not written** for the failing stage — so the
filesystem tells the truth (no "stage completed" artefact when the stage failed
the loglik-regression check). Gate 1 fires before any computation, so there is
no `fit_state.toml` at all for a stage that is hard-blocked by Gate 1.

### Data flow

```
IF2 runs complete
       ↓
clean_eval::run_clean_eval()   [clean_eval.rs:259]
  → CleanEvalOutcome {
       all_scores: Vec<CandidateScore>,     ← (chain × candidate) table
       per_chain_winners: Vec<ChainWinner>, ← best candidate per chain
       overall_winner_idx: usize,           ← index of overall best
    }
       ↓
runner::run_chains_with_per_chain_params()
  → ChainResults {
       results: Vec<(usize, IF2Result)>,
       best_chain, best_loglik,
       chain_agreement: HashMap<String, f64>,
       clean_eval: CleanEvalOutcome,
    }
       ↓
Gate 2 check [mod.rs:718]
  gating::check_loglik_regression(
      scout_best, chain_results.best_loglik,
      &scout_chain_logliks_for_gate2)
  → Ok(()) or Err(msg) → process::exit(1)
       ↓
FitState constructed [mod.rs:784]
  chain_clean_logliks = chain_results.chain_clean_logliks()  ← per-chain winner logliks
  chain_clean_ses     = chain_results.chain_clean_ses()      ← per-chain winner SEs
  resolved_gate       = Some(effective_gate.clone())
  resolved_clean_eval = Some(effective_clean_eval.clone())
       ↓
fit_state.toml written [mod.rs:817]

── Gate 1 fires at the START of the NEXT stage (pre-run) ──

prior FitState loaded from scout's fit_state.toml
       ↓
gating::check_scout_convergence(prior_state, &effective_gate)   [mod.rs:612]
  reads: tail_chain_agreement, ivp_params,
         chain_clean_logliks, chain_clean_ses,
         chain_logliks (for Gate 2 epsilon)
  → ScoutGateVerdict::{Ok, SoftWarn, Hard, DecibansSpread}
       ↓
ScoutGateVerdict::Hard or DecibansSpread
  → error printed to stderr [mod.rs:643, 665]
  → process::exit(1)   OR   sweep_failures.push(...)
  → NO FitState written for this (blocked) stage

── camdl fit summary ──

FitState::load() reads fit_state.toml
       ↓
TEXT PATH:  Formatter::gate_verdict_block(&state)   [fit_summary.rs:254]
  → recomputes Â leg and decibans leg from raw FitState fields
  → renders with ANSI colour

JSON/MD/LATEX PATH: build_summary_doc() → stage_report()   [fit_summary.rs:647]
  → recomputes SAME Â leg and decibans leg from raw FitState fields
  → builds GateReport struct
  → MD/LaTeX renderers read GateReport
```

### Computation duplication (ClM3)

The gate-verdict computation (Â pass/fail, decibans leg pass/fail, overall
verdict) is implemented **twice** in `fit_summary.rs`:

**Text path** (`Formatter::gate_verdict_block`, lines 254–319):
```rust
let max_a = state.tail_chain_agreement.values().cloned()
    .fold(0.0_f64, f64::max);
let a_passes = max_a < gate.a_thresh;
// ...
let se_floor_db = crate::fit::gating::SE_FLOOR_K * sigma_max * NATS_TO_DB;
let threshold_db = gate.decibans_thresh.max(se_floor_db);
let db_passes = delta_db < threshold_db;
let overall_pass = a_passes && db_passes;
```

**JSON/structured path** (`stage_report`, lines 660–684):
```rust
let max_a = state.tail_chain_agreement.values().cloned()
    .fold(0.0_f64, f64::max);
let a_passes = max_a < gate_cfg.a_thresh;
// ...
let se_floor_db = crate::fit::gating::SE_FLOOR_K * sm * NATS_TO_DB;
let td = gate_cfg.decibans_thresh.max(se_floor_db);
// db_passes = Some(dd < td)
// overall_pass = db_passes.map(|p| p && a_passes)
```

These two blocks implement the same mathematical computation. There are two
structural differences:

1. The text path uses a combined `a_passes && db_passes` boolean; the JSON path
   uses `Option<bool>` throughout, propagating the `None` (no clean-eval data)
   case into `overall_pass`. This means the text path renders "overall: ✓ PASS"
   only when the decibans data is present (the `else` branch prints a dash), but
   the JSON path encodes `overall_pass: None` when data is absent. The semantics
   align, but the code paths are independently maintained.

2. The text path's `max_a_param` lookup (`state.tail_chain_agreement.iter().max_by(...)`)
   is identical to the JSON path's — line-for-line. Neither references the
   other.

SE_FLOOR_K is a named constant (`gating::SE_FLOOR_K = 8.0`), so its value is
shared, but the formula `SE_FLOOR_K * sigma_max * NATS_TO_DB` is duplicated
verbatim at `fit_summary.rs:289` and `fit_summary.rs:678`.

There is a **third** location: `gating::format_decibans_spread_verdict`
(line 242) recomputes `se_floor_db = SE_FLOOR_K * sigma_max * NATS_TO_DB` to
format the diagnostic message. This path correctly uses the formula for display
(determining which limb of `max(...)` is binding), but it is a third site where
the same arithmetic appears.

---

## 2. UX audit

### Gate verdict outcomes — what the user sees and where

#### Gate 1, verdict: `SoftWarn`

**When:** `A_SOFT (1.05) ≤ max(Â) < gate.a_thresh (1.01 default)`.
Note: with the default `a_thresh = 1.01`, the SoftWarn band `[A_SOFT, a_thresh)
= [1.05, 1.01)` is **empty**. SoftWarn is only reachable when the user has
configured `a_thresh > A_SOFT` (e.g. `a_thresh = 1.10` in the legacy gate
config). This is documented in `gating.rs:146` but is a latent UX confusion —
a user running with a custom `a_thresh = 1.10` sees SoftWarn for Â in
[1.05, 1.10); a user with the default `a_thresh = 1.01` never sees SoftWarn.

**Message shown during `camdl fit run`** (`mod.rs:614–619`):
```
warning: prior stage tail Â in SoftWarn band ([1.05, 1.10)) for: beta (Â=1.07)
```
- Shows the soft bound, the hard threshold, and names each failing param with
  its value. **Actionable.** No suggested remediation beyond the values.
- **No hard block.** Run proceeds.

**In `fit_state.toml`:** Not recorded. SoftWarn leaves no trace in the
persistent state. A user who inspects `fit_state.toml` after the fact cannot
determine that a SoftWarn fired during the run.

**In `camdl fit summary`:** The text output renders the Â leg with `~` markers
on params above `A_SOFT` but below `a_thresh`
(`parameter_table`, `fit_summary.rs:349`). The gate verdict block shows the Â
leg with an arrow to the param, threshold shown. **No direct indication** that
a SoftWarn fired at run time — the user would have to compare the Â value
against both thresholds themselves.

#### Gate 1, verdict: `Hard`

**When:** `max(Â) ≥ gate.a_thresh` on any non-IVP estimated parameter.

**Message during `camdl fit run`** (`mod.rs:622–644`, rendered via
`gating::format_hard_verdict`). Full example from the code:
```
refine stage requires scout convergence.

  Scout tail Â (last half of iterations):
    ✗ beta        Â =  3.502   (> 1.10)
    ~ gamma       Â =  1.194
      I0          Â = 16.527   (ivp — not gated)

  Scout loglik spread: 794.4 (best chain loglik -60.2)
  -> likelihood surface is almost certainly multi-modal.

  Failing: beta (Â=3.50)

  Pick one:
    - re-run scout with more chains or iterations
    - narrow bounds to the basin scout's best chain found
    - mark weakly-identified params as `ivp = true`
      (reported but not gated)

  To run refine anyway (results may launder multi-modality):
    camdl fit run fit.toml --allow-nonconverged-scout
```
- **Shows actual values, threshold, per-param table, loglik spread.** Highly
  actionable.
- The `scout_best_chain_values` parameter to `format_hard_verdict` is called
  with `None` in `mod.rs:623`. The branch that would print per-param best-chain
  values ("copy into [estimate.*] bounds / start values") is therefore **never
  reached** in the current code — the optional tightening hint is dead at the
  call site.
- **Hard block:** `process::exit(1)` unless `--allow-nonconverged-scout` or
  `has_sweep`.
- **In sweep mode:** Gate failure is recorded in `sweep_failures` but the stage
  is skipped, not the entire sweep. The error message is prefixed with
  `sweep-skip:` rather than `error:`.
- **In `fit_state.toml`:** No `fit_state.toml` is written for the blocked stage
  (Gate 1 fires pre-run).
- **In `camdl fit status`:** When `fit_state.toml` exists but `run.json` is
  absent (gate failed after IF2 but before run.json write — the scout stage
  itself), status shows:
  ```
  scout    ✗ gate failed — see `camdl fit summary <parent>`
  ```
  This is correct (`mod.rs:174–178`). But for a stage that was Gate-1-blocked
  *before it ran* (refine blocked on scout failure), the refine directory
  doesn't exist at all, so `camdl fit status` simply omits refine. The user
  must infer from the absence of `refine/` that refine was blocked.

**In `camdl fit summary` (text):** The Â leg shows `✗` next to `max Â = X.XXX
(param)`. The `overall: ✗ FAIL` line appears only when the decibans data is
present. If the run failed Gate 1 *before running*, there is no `fit_state.toml`
for the blocked stage, so `fit summary` has nothing to render for that stage.
**There is no explicit Gate 1 failure line in the summary for blocked stages.**

#### Gate 1, verdict: `DecibansSpread`

**When:** Â passes but inter-chain clean-eval loglik spread (in decibans)
exceeds `max(gate.decibans_thresh, SE_FLOOR_K * sigma_max * NATS_TO_DB)`.

**Message during `camdl fit run`** (`mod.rs:648–666`, rendered via
`gating::format_decibans_spread_verdict`):
```
scout chains landed in different basins.

  clean-eval log-likelihood spread:
    Δℓ = 78234.5 dB > threshold = 30.0 dB (user-configured floor decibans_thresh = 30.0 dB)

  Per-chain clean logliks (nats / dB from worst):
    chain 1   ℓ = -5982.7  (+78234.5 dB from worst)
    chain 2   ℓ = -6340.7  (+0.0 dB from worst)

  Pick one:
    - re-run scout with more chains (the wider the spread, ...)
    - inspect chain_evaluations.tsv to see which candidate label dominated ...
    - if the spread is genuinely Monte-Carlo noise, raise [stages.scout.clean_eval] n_particles ...
    - relax the gate via [stages.scout.gate].decibans_thresh ...

  To proceed anyway:  camdl fit run fit.toml --allow-nonconverged-scout
```
- **Shows actual Δ, effective threshold, which limb of max() is binding, and
  per-chain logliks.** Highly actionable. The `floor_source` string in
  `format_decibans_spread_verdict` correctly identifies whether the SE floor or
  the user-configured floor is binding (`gating.rs:243–248`).
- **Hard block** (or sweep-skip or warn under `--allow-nonconverged-scout`).

**In `fit_state.toml` of the scout (which did complete):** `chain_clean_logliks`
and `chain_clean_ses` are written. `resolved_gate` captures the thresholds used.
So the scout's own `fit_state.toml` is complete. The refine stage never runs, so
there is no refine `fit_state.toml`.

**In `camdl fit summary` (text):** The scout stage renders with `✗ FAIL` on the
decibans leg. The user sees the failing values and thresholds in the gate verdict
block. This is where `fit_summary.rs:gate_verdict_block` is useful — the scout
ran, wrote state, so summary can render it. The threshold is shown.

#### Gate 2, loglik regression

**When:** After a stage that `starts_from` a prior stage, `refine.best_loglik <
scout.best_loglik - ε`.

**Message during `camdl fit run`** (`gating::check_loglik_regression`,
`gating.rs:213–229`):
```
refine regressed below scout.

  scout  best_loglik = -60.1
  refine best_loglik = -76.3   delta = -16.2, threshold ε = 4.2

  Refine landed in a worse basin than scout found. ...
  [5 bullet causes listed]

  scout/fit_state.toml is authoritative for "what scout's best looked like."
  Investigate before re-running.
```
- **Shows both logliks, delta, ε, and root-cause diagnoses.**
- **Hard block, no override.** `process::exit(1)` unconditionally (sweep mode
  uses `break` + `sweep_failures.push(...)`).
- **Not overridable.** This is intentional and correct per the design doc.
- **No `fit_state.toml` written** for the regressed stage (`fit_state.save`
  happens *after* Gate 2 at `mod.rs:817`). This is correct — the gate fires
  at `mod.rs:719`, before the FitState construction at `mod.rs:784`.

**In `camdl fit summary`:** No `fit_state.toml` for the regressed stage. The
refine directory may have been created (`std::fs::create_dir_all` at `mod.rs:685`
runs before Gate 2), so `camdl fit status` may show refine as `✗ gate failed`
if the directory exists with no `run.json`. The Gate 2 message itself is only
visible in the `camdl fit run` stderr output — there is no way to recover it
from the fit directory after the fact.

### Gates pass — what does the user see?

**During `camdl fit run`:** No gate-pass line. The user sees:
```
── stage: refine (method=if2) ──
```
followed by the run, and at the end:
```
refine complete in 123.4s: path/to/refine/
  best loglik: -61.2 (chain 3)
```
There is no explicit "Gate 1 PASSED" or "Gate 2 PASSED" confirmation in the
live output. The user must infer pass from the absence of an error.

**In `camdl fit summary` (text):** The gate verdict block renders `✓ PASS` when
both legs pass. This is the only place where a gate pass is explicitly confirmed.

### Gaps and problems

**G1 — Best-chain values hint is dead code.**
`format_hard_verdict` accepts `scout_best_chain_values: Option<&[(String, f64)]>`
and emits per-param best-chain values when `Some`. The call site at `mod.rs:623`
always passes `None`. The actionable hint "narrow bounds to the basin scout's
best chain found: β ≈ 1.834" is never shown. This is the most specific
remediation advice and it never fires.

**G2 — Gate 1 hard block before IF2 runs is invisible to `fit summary`.**
When Gate 1 blocks refine (refine never runs), the refine stage directory either
does not exist or contains no `fit_state.toml`. `camdl fit summary` skips
stages with no `fit_state.toml`. The user who runs `camdl fit summary` after a
Gate 1 failure sees only the scout stage and must infer from the absence of
refine that it was blocked. There is no "refine was blocked by Gate 1 —
re-run `camdl fit run` to see why" message.

**G3 — SoftWarn leaves no trace in persistent state.**
A SoftWarn during `camdl fit run` is printed to stderr and then lost. There is
no field in `fit_state.toml` that records whether a SoftWarn fired. A user who
re-runs `camdl fit summary` hours after a run that soft-warned has no way to
know the warning occurred.

**G4 — Gate 2 failure reason is not recoverable from the fit directory.**
The Gate 2 error message appears only in `camdl fit run` stderr. The fit
directory contains a created-but-empty refine dir (or a partial one). No file in
the refine dir records "gate 2 fired: scout_best=-60.1, refine_best=-76.3,
delta=-16.2, ε=4.2". This means forensic investigation of gate failures
requires the user to have kept the terminal output.

**G5 — Default a_thresh (1.01) makes SoftWarn band permanently empty.**
`A_SOFT = 1.05`, default `a_thresh = 1.01`. The SoftWarn band `[1.05, 1.01)` is
empty. Any Â ≥ 1.01 goes directly to `Hard`. For users to get SoftWarn they
must explicitly configure `a_thresh > 1.05`. This is intentional (the default is
deliberately strict), but the code comment at `gating.rs:144` — "SoftWarn band
only exists when gate.a_thresh > A_SOFT" — does not surface to the user; they
discover it by noticing SoftWarn never fires on the default. The design note
about the empty band should appear in TOML documentation.

**G6 — Gate 1 threshold not shown in live `camdl fit run` output at startup.**
When a stage starts, the effective `a_thresh` and `decibans_thresh` are not
printed. The user does not see "refine will gate on: Â < 1.01, spread < 30 dB"
before the run begins. They only see the threshold at gate-failure time (in the
error message) or at `camdl fit summary` time. A brief "gate thresholds: Â <
1.01, decibans < 30.0 (SE-adaptive)" line at stage startup would let users know
what they're being held to before seeing the result.

**G7 — `resolved_clean_eval` is stored but never surfaced in text summary.**
`resolved_clean_eval` is written to `fit_state.toml` and appears in the JSON
`GateReport.resolved_clean_eval` field, but `Formatter::gate_verdict_block` and
`Formatter::chain_clean_eval_table` do not show n_particles or n_replicates to
the user in text mode. A user reading `camdl fit summary` cannot tell how many
particles were used for the clean-eval scores that drove the gate decision.

---

## 3. ClM3 fix proposal

### Current duplication

**Path A — text output** (`fit_summary.rs:254–319`, `Formatter::gate_verdict_block`):

```rust
// Â leg
let max_a = state.tail_chain_agreement.values().cloned()
    .fold(0.0_f64, f64::max);
let max_a_param = state.tail_chain_agreement.iter()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(k, _)| k.clone()).unwrap_or_else(|| "—".into());
let a_passes = max_a < gate.a_thresh;
// ...
// Decibans leg
let hi = state.chain_clean_logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let lo = state.chain_clean_logliks.iter().cloned().fold(f64::INFINITY, f64::min);
let delta_db = (hi - lo) * NATS_TO_DB;
let sigma_max = state.chain_clean_ses.iter().cloned().fold(0.0_f64, f64::max);
let se_floor_db = crate::fit::gating::SE_FLOOR_K * sigma_max * NATS_TO_DB;
let threshold_db = gate.decibans_thresh.max(se_floor_db);
let db_passes = delta_db < threshold_db;

let overall_pass = a_passes && db_passes;
```

**Path B — structured output** (`fit_summary.rs:647–787`, `stage_report`):

```rust
// Â leg
let max_a = state.tail_chain_agreement.values().cloned().fold(0.0_f64, f64::max);
let max_a_param = state.tail_chain_agreement.iter()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    .map(|(k, _)| k.clone());
let a_passes = max_a < gate_cfg.a_thresh;
// ...
// Decibans leg
let hi = state.chain_clean_logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
let lo = state.chain_clean_logliks.iter().cloned().fold(f64::INFINITY, f64::min);
let dd = (hi - lo) * NATS_TO_DB;
let sm = state.chain_clean_ses.iter().cloned().fold(0.0_f64, f64::max);
let se_floor_db = crate::fit::gating::SE_FLOOR_K * sm * NATS_TO_DB;
let td = gate_cfg.decibans_thresh.max(se_floor_db);
// db_passes = Some(dd < td)
// overall_pass = db_passes.map(|p| p && a_passes)
```

The two paths differ in:
- Variable names (`max_a_param` is `String` in path A, `Option<String>` in path B).
- `overall_pass` type: `bool` in path A (elided when clean-eval data is absent
  by the surrounding `if`), `Option<bool>` in path B.
- Path A calls `state.chain_clean_logliks.len() >= 2` guard before computing the
  decibans leg; path B uses the same guard.

**Divergence risk:** The two paths can produce different verdicts if their inputs
differ — but since both read from the same `FitState`, the only way they can
diverge is if the formula is applied differently. Currently they are consistent,
but there is no test that verifies consistency between the two paths given the
same `FitState`. A future edit to one path that misses the other would go
undetected until a user reports inconsistency between text and JSON output.

**Third duplication site:** `gating::format_decibans_spread_verdict`
(`gating.rs:236–275`) recomputes `se_floor_db` at line 242 for display
purposes. This is a rendering helper, not a gate-decision computation, but it
uses the same formula and is another maintenance point.

### Proposed `GateVerdictSummary` type

Define this in `gating.rs` (near `ScoutGateVerdict`, which it complements):

```rust
/// Pre-computed gate verdict for a completed stage. Computed once from
/// a `FitState` and consumed by all rendering paths (text, JSON, MD, LaTeX).
/// Eliminates the ClM3 duplication where `gate_verdict_block` (text) and
/// `stage_report` (JSON) independently recompute the same arithmetic.
///
/// All `Option` fields are `None` when the decibans leg cannot be evaluated
/// (missing `chain_clean_logliks` / `chain_clean_ses`). The `overall_pass`
/// field uses the same three-valued semantics: `Some(true)` = both legs pass,
/// `Some(false)` = at least one leg fails, `None` = incomplete data (decibans
/// leg indeterminate).
#[derive(Debug, Clone)]
pub struct GateVerdictSummary {
    // Â leg
    pub max_a: f64,
    pub max_a_param: Option<String>,
    pub a_thresh: f64,
    pub a_passes: bool,

    // Decibans leg (None when chain_clean_logliks / chain_clean_ses absent)
    pub delta_db: Option<f64>,
    pub threshold_db: Option<f64>,
    pub sigma_max: Option<f64>,
    pub db_passes: Option<bool>,

    // Compound verdict
    pub overall_pass: Option<bool>,

    // Provenance: where did the thresholds come from?
    pub threshold_source: GateThresholdSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GateThresholdSource {
    /// `state.resolved_gate` was present (written by Phase 3+ runs).
    /// Thresholds reflect what the run was actually judged against.
    Resolved,
    /// Legacy `fit_state.toml` — no `resolved_gate`. Showing
    /// `GateConfig::default()` with a caveat.
    DefaultFallback,
}

/// Compute the gate verdict summary once from a `FitState`. All rendering
/// paths (text, JSON, MD, LaTeX) should call this and consume the result
/// rather than re-implementing the arithmetic.
pub fn compute_gate_verdict(state: &FitState) -> GateVerdictSummary {
    use crate::evidence::NATS_TO_DB;

    let (gate, threshold_source) = match &state.resolved_gate {
        Some(g) => (g.clone(), GateThresholdSource::Resolved),
        None    => (GateConfig::default(), GateThresholdSource::DefaultFallback),
    };

    let max_a = state.tail_chain_agreement.values().cloned()
        .fold(0.0_f64, f64::max);
    let max_a_param = state.tail_chain_agreement.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone());
    let a_passes = max_a < gate.a_thresh;

    let (delta_db, threshold_db, sigma_max, db_passes) =
        if state.chain_clean_logliks.len() >= 2
            && state.chain_clean_ses.len() == state.chain_clean_logliks.len()
        {
            let hi = state.chain_clean_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_clean_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let dd   = (hi - lo) * NATS_TO_DB;
            let sm   = state.chain_clean_ses.iter().cloned().fold(0.0_f64, f64::max);
            let floor = SE_FLOOR_K * sm * NATS_TO_DB;
            let td   = gate.decibans_thresh.max(floor);
            (Some(dd), Some(td), Some(sm), Some(dd < td))
        } else {
            (None, None, None, None)
        };

    let overall_pass = db_passes.map(|p| p && a_passes);

    GateVerdictSummary {
        max_a, max_a_param, a_thresh: gate.a_thresh, a_passes,
        delta_db, threshold_db, sigma_max, db_passes, overall_pass,
        threshold_source,
    }
}
```

### Migration path

1. Move `GateVerdictSummary`, `GateThresholdSource`, and `compute_gate_verdict`
   into `gating.rs`. (The `GateThresholdSource` enum is currently defined
   privately in `fit_summary.rs:466`; move it and make it `pub`.)

2. In `fit_summary.rs::Formatter::gate_verdict_block`, replace the inline
   computation with:
   ```rust
   let verdict = crate::fit::gating::compute_gate_verdict(state);
   // render verdict fields
   ```

3. In `fit_summary.rs::stage_report`, replace the inline computation with:
   ```rust
   let verdict = crate::fit::gating::compute_gate_verdict(&state);
   let gate = GateReport {
       max_a_hat: verdict.max_a,
       max_a_param: verdict.max_a_param,
       a_thresh: verdict.a_thresh,
       a_passes: verdict.a_passes,
       delta_db: verdict.delta_db,
       threshold_db: verdict.threshold_db,
       sigma_max: verdict.sigma_max,
       db_passes: verdict.db_passes,
       overall_pass: verdict.overall_pass,
       threshold_source: match verdict.threshold_source {
           GateThresholdSource::Resolved => "resolved".to_string(),
           GateThresholdSource::DefaultFallback => "default_fallback".to_string(),
       },
       resolved_gate: state.resolved_gate.clone(),
       resolved_clean_eval: state.resolved_clean_eval.clone(),
   };
   ```

4. Add a unit test in `gating.rs` that verifies `compute_gate_verdict` produces
   the same result when called twice on the same `FitState` (trivially passes
   since it is pure), and a cross-format consistency test in `fit_summary.rs`
   that builds a `FitState`, renders both text and JSON, then parses the JSON
   to verify `gate.a_passes`, `gate.db_passes`, and `gate.overall_pass` match
   what the text output shows.

The `GateReport` struct in `fit_summary.rs` can be kept as-is (it is the JSON
output contract); the migration only removes the computation from the two render
paths, replacing it with a single call to `compute_gate_verdict`.

---

## 4. Code review findings

### Major

**M1 — `scout_best_chain_values` is always `None` at the call site (dead hint).**
File: `mod.rs:623`.
```rust
let msg = gating::format_hard_verdict(
    &failing, &all_structural, &ivp,
    loglik_spread, ps.best_loglik, None);  // ← always None
```
`format_hard_verdict`'s `scout_best_chain_values` parameter exists for the
actionable "copy into bounds" hint, which is the most operationally useful
guidance on a Gate 1 Hard failure. It is never populated. The values needed are
in `ps.start_values` (the scout's winning θ̂), ordered to match the params in
`all_structural`. Fix: build the list from `ps.start_values` filtered to
structural params, pass as `Some(&values)`.

**M2 — Gate 1 Hard failure is invisible to `camdl fit summary` for the blocked stage.**
When Gate 1 blocks refine, refine's stage directory may not exist (or may exist
empty). `cmd_fit_summary` skips stages with no `fit_state.toml`
(`fit_summary.rs:111`). There is no mechanism for `fit summary` to show "refine
was blocked by Gate 1 on scout's Â." The user's only recovery path is the
terminal output from `camdl fit run`, which may not be available.

Proposal: write a minimal `gate_failure.toml` (or a JSON sidecar) into the
blocked stage directory when Gate 1 fires, recording the verdict type and
values. `fit summary` checks for this file when `fit_state.toml` is absent and
renders a "gate blocked" block.

**M3 — `overall_pass` in text path is silently `false` when clean-eval data is
absent.** In `Formatter::gate_verdict_block`, the `overall_pass` variable is
computed as `a_passes && db_passes` at `fit_summary.rs:297`, but `db_passes` is
only set inside the `if state.chain_clean_logliks.len() >= 2` block. If the
`else` branch fires (no clean-eval data), there is no `overall:` line at all
— the code paths diverge:
```rust
if state.chain_clean_logliks.len() >= 2 && ... {
    // ...
    s.push_str(&format!("    overall:         {}\n", overall));  // shown
} else {
    s.push_str(&format!("    decibans leg:    {} ...\n", self.dim("—")));  // no overall line
}
```
This is correct behaviour (no overall verdict without both legs), but it means
the text format omits the `overall:` line while the JSON path emits
`"overall_pass": null`. These are semantically equivalent but structurally
different — a downstream parser that looks for `overall:` in text output would
fail on legacy states. Minor structural inconsistency, noted as M3 rather than a
nit because a downstream script could fail on it.

### Minor

**Mi1 — `check_scout_convergence` returns `Ok` for legacy states
(no `tail_chain_agreement`).**
`gating.rs:105–107`:
```rust
if scout.tail_chain_agreement.is_empty() {
    return ScoutGateVerdict::Ok;
}
```
The caller at `mod.rs:612` is expected to warn-and-proceed for the legacy case,
per the comment at `gating.rs:99`. But `mod.rs:612` does not check whether `Ok`
came from a legacy state or a genuine pass — it treats both as "proceed
silently." A user running a legacy scout feed through a modern refine gets no
warning that the gate was skipped entirely due to missing data.

The fix is to add a `LegacySkipped` variant to `ScoutGateVerdict` and handle it
explicitly in `mod.rs` with a "(gate skipped — legacy scout state)" warning. Or,
simpler: have the caller check `ps.tail_chain_agreement.is_empty()` before
calling `check_scout_convergence` and emit the caveat there.

**Mi2 — Compound gate is sequential, not truly compound.**
Gate 1's Â check runs before the decibans check. If Â fails, `DecibansSpread`
is never evaluated. The comment at `gating.rs:86–101` documents this correctly,
but the enum's doc comment says "Compound gate" without mentioning that only one
failure mode can be returned per call. When a user has both Â > a_thresh AND
decibans spread > threshold, they only see the Â failure. After they fix Â and
re-run, they may be surprised by a decibans failure.

This is arguably correct design (fix the harder diagnostic first), but the
compound gate error message for `Hard` should note "decibans-spread check
skipped — fix Â first" so the user knows to expect a possible second gate
failure. Currently the Hard message does not mention the decibans leg at all.

**Mi3 — `loglik_spread` in `Hard` uses `chain_logliks` (in-run), not
`chain_clean_logliks`.**
`gating.rs:131–137`:
```rust
let loglik_spread = if scout.chain_logliks.len() >= 2 {
    let hi = scout.chain_logliks.iter()...;
    let lo = scout.chain_logliks.iter()...;
    hi - lo
} else { 0.0 };
```
The `chain_logliks` field contains the **in-run (noisy) final-iteration PF
logliks**, which have ~30 nat SD at 500 particles. The displayed "Scout loglik
spread" is therefore noisy and may overstate or understate the true cross-basin
spread. The decibans gate correctly uses `chain_clean_logliks` (de-biased
clean-eval scores) when computing its threshold test. But the diagnostic in the
`Hard` message shows the noisier number. This is not an error in the gate
logic — the Â gate fires before the clean-eval data is available in the code
path — but it means the "loglik spread" in the Hard error message is not
comparable to the decibans threshold in the `DecibansSpread` verdict.

**Mi4 — `a_thresh` is applied to `max(Â)` without excluding IVP params
(but they were already split).**
`gating.rs:127`: `if worst >= gate.a_thresh` where `worst` is
`structural.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max)` and `structural`
has already had IVP params removed (`gating.rs:111–119`). Correct. But the
`SoftWarn` branch at `gating.rs:145` filters `structural.into_iter()` for the
warnable params after the sort — if `structural` was consumed by the sort, the
`into_iter()` on line 149 works because `structural` was sorted in-place. This
is correct Rust but easy to misread; a comment that `structural` is sorted
in-place before the `SoftWarn` branch would help.

**Mi5 — `GateReport.threshold_source` is a raw `String` in the JSON output.**
`fit_summary.rs:569`:
```rust
pub threshold_source: String,
```
With values `"resolved"` or `"default_fallback"`. This is not an enum in the
serialized output, so consumers must string-match against undocumented values.
It should be a serialized enum or documented as a versioned vocabulary in the
schema comment.

### Nit

**N1 — `format_decibans_spread_verdict` re-computes `se_floor_db` for display.**
`gating.rs:242`:
```rust
let se_floor_db = SE_FLOOR_K * sigma_max * NATS_TO_DB;
```
This is a third invocation of the same formula (ClM3). When
`GateVerdictSummary` is introduced, `threshold_db` should carry the effective
threshold. `format_decibans_spread_verdict` should accept the pre-computed
`se_floor_db` and `threshold_db` rather than recomputing them from
`sigma_max`. Currently the inputs are redundant: `threshold_db =
max(decibans_thresh, se_floor_db)`, and `sigma_max` is only used to recompute
`se_floor_db`. This creates a subtle inconsistency risk: if the constant
`SE_FLOOR_K` or `NATS_TO_DB` is changed, the display formatter would need to be
updated separately from the gate logic.

**N2 — `GateConfig` has no validation that `a_thresh > 1.0`.**
A user who sets `a_thresh = 0.5` would produce a gate that always fails (since
Â ≥ 1.0 by construction). `GateConfig::default()` is fine, but there is no
validation at config-load time in `FitConfigV2::validate`. This should be added
alongside bounds validation.

**N3 — `loglik_regression_epsilon` uses population-SD formula comment but
sample-SD implementation.**
`gating.rs:192`:
```rust
let var = scout_chain_logliks.iter()
    .map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
```
This is the **sample variance** (Bessel-corrected), not population variance. The
doc comment at `gating.rs:184` says "2 · σ(scout.chain_logliks)" without
specifying population vs. sample. For two chains, the sample SD is
$\sqrt{(x_1 - x_2)^2 / 1} = |x_1 - x_2|$, which gives `two_sigma = 2|x_1 - x_2|`.
This is mathematically sensible (the natural scale for a two-chain spread), but
should be documented.

**N4 — `A_HARD` is "retained as a documented constant" but is now dead in the
default gate.**
`gating.rs:29`: `pub const A_HARD: f64 = 1.10`. Per the comment it informs the
SoftWarn band when `a_thresh` is configured loosely, but with the default
`a_thresh = 1.01`, `A_HARD` only appears in test fixtures (`legacy_gate()`).
The constant is not used in production code paths when running under defaults.
It is not wrong to keep it, but its pub visibility means it is part of the
public API surface of the gating module; a future reader may add a dependency
on it expecting it to have runtime significance.

**N5 — `chain_clean_ses.len() == chain_clean_logliks.len()` guard is checked in
three places with no shared helper.**
`gating.rs:157`, `fit_summary.rs:279`, `fit_summary.rs:668`. The check
`chain_clean_ses.len() == chain_clean_logliks.len()` is a precondition for
safe zip iteration. After the `GateVerdictSummary` refactor this would be
reduced to one check in `compute_gate_verdict`, resolving the repetition.
