---
date: 2026-10-29
status: proposed — implements maintainer decisions D3 (assessment thresholds leave run identity) and D4 (refuse unsupported old records as inputs, keep them inspectable) of 2026-10-29; §11 records the calls made under them; §12 names the one value left to the maintainer
area: fit result records / convergence verdicts / run identity / fit table, fit summary, fit predict
issues: none filed; the follow-ups are named in §13
related:
  - ../inference-architecture.md (the invariants this implements: I6, I7, I8)
  - 2026-10-26-parameter-roles.md (accepted; untouched by this proposal)
  - 2026-10-29-method-scoped-settings.md (branch docs/method-scoped-settings; §10 here states the order)
---

# A fit stores what it measured; whether it converged is computed when someone asks, under a policy that is named

Code references are against `origin/main` at `a9a89178` unless marked with a
branch. A claim marked _verified_ was read at the cited line or produced by the
command shown; _inferred_ means reasoned from the code and not executed. Nothing
in this document was measured by running a fit. The concepts and invariants this
builds on are stated in `docs/dev/inference-architecture.md`; this document uses
its terms (`ResolvedProblem`, `ExecutionPlan`, `RunRecord`, result record,
`Assessment`).

## 1. Every method writes one flat record, and its verdict is computed four ways

A fit leaf (one method, one seed, under one problem) stores its outcome in
`fit_state.toml`, deserialized as `FitState` (`cli/src/fit/state.rs:19-182`,
verified). IF2, the three NLopt optimizers, PMMH, MH and PGAS all write that one
struct, each leaving the others' fields at an empty or `None` sentinel, and
NLopt writes its per-parameter relative range across chains into
`tail_chain_agreement`, the field named for IF2's chain-agreement statistic Â
(`nlopt_stage.rs:297`, verified).

Whether a leaf converged is then decided in several places, by different rules,
against thresholds of which some are part of the run's identity and some are
not. Raising `--decibans-thresh` re-keys and re-runs an IF2 fit that has already
been computed, because the threshold is written into `[method]` before its
identity is taken (`config_v2.rs:1597`, `:1652-1655`, verified); lowering the R̂
bar in the source would silently re-judge every stored Bayesian fit, because the
bar is a constant applied at read time and recorded nowhere
(`method_result.rs:585`, verified).

This proposal makes three changes, in independently landable stages (§9):

- **One rule per measurement type.**
  `assess(measurements, policy) -> Assessment` is the only function that turns
  measurements into a verdict. `fit table`, `fit summary`, the warm-start source
  check and `fit predict` call it. Optimizer agreement (IF2, NLopt) and
  posterior mixing (PGAS, PMMH, MH) keep distinct measurement types and distinct
  statistics; what is unified is the signature, the verdict type and the rule
  that every verdict names its policy.
- **Thresholds leave run identity (D3).** A `DiagnosticPolicy` holds every
  post-run threshold, lives in a new `[assessment]` table and in read-time
  flags, and is in no hash. Execution controls stay where they are.
- **Typed result records with a schema tag (D4).** `FitState` is replaced by a
  per-method `ResultRecord` carrying `schema` and `producer`. A record of any
  other schema is refused as an input to a new computation with a versioned
  error, and is still shown, read-only, by `fit summary` and `fit table`.

## 2. Terms

A **measurement** is a number the run computed and stored: a tail Â per
parameter, a clean-evaluation log-likelihood and its standard error per chain, a
rank-normalized R̂ (the between-chain potential scale reduction, Vehtari et al.
2021, _Bayesian Analysis_ 16(2):667–718) or a bulk ESS (effective sample size)
per parameter, an optimizer's per-chain θ̂ and exit status, a dt-check ladder.

**Optimizer agreement** asks whether independent optimizer chains (IF2 or NLopt)
reached the same maximum. **Posterior mixing** asks whether sampler chains
explore the same distribution. They are different questions with different
statistics; an Â of 1.02 and an R̂ of 1.02 are not comparable numbers (the
distinction `If2StageResult.max_chain_agreement`'s doc already draws,
`method_result.rs:79-84`).

An **assessment policy** is the set of thresholds that turn measurements into a
verdict. An **execution control** changes what the engine computes or when it
stops. `docs/dev/inference-architecture.md` ("What counts as which") is the
normative definition; §5 here classifies every current value.

A **verdict** is `Pass`, `Warn` (passes, with a named parameter near the
threshold), `Fail`, or `NotAssessed` with a reason (one chain; a point start, so
between-chain statistics are uninformative; no clean evaluation).

## 3. What the code does today

### 3.1 The IF2 verdict: four implementations, three rules

| site                                                        | initial-value parameters (`perturb_only_at_t0`) | decibans leg missing | warning band (`A_SOFT`)                   |
| ----------------------------------------------------------- | ----------------------------------------------- | -------------------- | ----------------------------------------- |
| `check_scout_convergence`, `gating.rs:182-260`              | exempt from the Â leg                           | judged on Â alone    | yes, `SoftWarn`                           |
| `compute_if2_gate_verdict`, `method_result.rs:1082-1123`    | included                                        | judged on Â alone    | no                                        |
| `Formatter::gate_verdict_block`, `fit_summary.rs:1520-1560` | included                                        | prints the Â leg     | glyphs only, through `GateConfig::a_band` |
| `if2_stage_report`, `fit_summary.rs:3028-3056`              | included                                        | `overall_pass: None` | no                                        |

All verified by reading. The first is what the warm-start source check applies
(`check_source_converged`, `gating.rs:439-480`); the second is `fit table`'s
`converged` and `gate_verdict` (`table_row.rs:320-337`); the third and fourth
are `fit summary`'s text and its JSON, Markdown and LaTeX outputs. The JSON
document embeds the `fit table` row (`FitSummaryDoc.table_row`,
`fit_summary.rs:2622-2629`) beside the per-method report, so one
`fit summary --format json` can carry two IF2 verdicts computed by different
rules. An IF2 fit whose only Â failure is on an initial-value parameter is
therefore accepted as a `from_mle` source and reported as failed by `fit table`
and `fit summary`. Branch `fix/settings-silent-noops` adds a test pinning that
the three agree
(`if2_verdict_is_the_same_for_the_table_the_summary_and_a_starts_source`,
uncommitted in its worktree on `e9358984`); no shared function exists there yet.

### 3.2 The NLopt verdict: two rules, and a source check that cannot fail

At the end of a run, `check_convergence` (`nlopt_stage.rs:738-815`, verified)
fails the agreement leg on a parameter only when its relative range across
chains exceeds `DET_REL_RANGE_THRESH = 0.05` _and_ its absolute range exceeds
`DET_ABS_RANGE_FACTOR · xtol_rel · max(|θ̂₀|, 1)` = `2 · xtol_rel · …`
(`:735-736`, `:782-786`), and fails a decibans leg when the converged chains'
log-likelihood spread exceeds `gate.decibans_thresh` (`:803`). It prints this
verdict and stores neither leg's outcome. `fit table` instead declares an NLopt
leaf converged when every chain converged and `max_rel_range < 0.05`
(`table_row.rs:293-294`): no absolute floor and no decibans leg.

As a warm-start source, an NLopt leaf has a non-empty `tail_chain_agreement`, so
`check_source_converged` judges it by IF2's rule: the relative ranges against
`a_thresh = 1.01`, and a decibans leg that is skipped because NLopt writes no
clean-evaluation log-likelihoods (`nlopt_stage.rs:301-303`). A relative range is
an absolute range divided by the search-box width, so for θ̂ inside the box it is
at most 1, and the source check cannot refuse an NLopt leaf (_inferred_; Stage 1
pins it with a red test).

### 3.3 The posterior verdict: one rule, an unrecorded bar

PGAS, PMMH and MH store R̂ and ESS in `<sampler>_summary.json`, tagged
`camdl.fit-summary/v2` (`run_meta.rs:44`, checked by `check_fit_summary_schema`,
`:50-71`). `fit table` (`table_row.rs:340-372`), `fit summary`
(`fit_summary.rs:1819-1825`) and the source check (`gating.rs:490-512`) all call
`PosteriorDiagnostics::converged_at(RHAT_CONVERGED_THRESHOLD)` or the band built
on it (`method_result.rs:477-479`, `:600-625`). The rule is consistent; the bar
is a constant that no output records. No minimum ESS is applied anywhere
(verified: `grep -n "min_ess" cli/src/fit/method_result.rs` shows only the
accessor at `:513-536`).

### 3.4 Thresholds in identity

`GateConfig { a_thresh, decibans_thresh }` (`config_v2.rs:1851-1873`) is a field
of `Algorithm::IF2` (`:1021`) and of the NLopt config (`:1268`). The method's
identity payload subtracts nothing for those methods (`config_v2.rs:1396-1400`),
and `--decibans-thresh` is written into the gate before the payload is taken
(`:1597`, `:1609`). The applied gate is also stored in the leaf as
`resolved_gate` (`state.rs:107-125`), which is why the override was made
identity: under the run-identity rule a field that changes stored bytes must
re-key. The dt-check tolerance follows the same path: `threshold_nats`
(`config_v2.rs:1920`), its backend defaults and `--dt-check-strict` halving
(`dt_check.rs:115-120`) resolve into a value stored in
`fit_state.toml [dt_check]` beside a stored `verdict` (`dt_check.rs:68-94`).

### 3.5 Precedents this proposal follows

- **The applied policy is already recorded for IF2** (`resolved_gate`, with
  `fit summary` reporting `default_fallback` when absent,
  `fit_summary.rs:3028-3031`). The proposal keeps that idea and moves the record
  out of the stored bytes.
- **A schema tag that refuses another vintage** exists for
  `<sampler>_summary.json` (gh#728, `run_meta.rs:44-71`).
- **A record of an old layout is refused by its loader**:
  `chain_index::refuse_one_based_layout` in `FitState::load`
  (`state.rs:189-190`, gh#990).
- **A read-side typed result already exists**: `MethodResult`
  (`method_result.rs:33-39`) with `If2StageResult`, `NloptStageResult`,
  `PmmhStageResult`, `PgasStageResult`, built from `FitState` and the summary
  JSON. It computes a verdict while loading (`If2StageResult.gate_verdict`,
  `:84`), which this proposal removes: a result type holds measurements, and
  verdicts are computed beside it.

## 4. The types

### 4.1 Result records

```rust
// rust/crates/cli/src/fit/record.rs (new; replaces state.rs)

pub const RESULT_SCHEMA: &str = "camdl.fit-result/v1";

/// One leaf's `fit_state.toml`. The method is the variant, so a record cannot
/// carry another method's fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultRecord {
    /// Always `RESULT_SCHEMA`; `load` refuses anything else (§6).
    pub schema: String,
    /// `crate::version::VERSION` of the camdl that wrote the record.
    pub producer: String,
    pub seed: u64,
    pub timestamp: String,
    pub chains: ChainCount,
    pub starts: StartsRecord,
    /// The dt-check ladder at the point estimate; no verdict and no tolerance
    /// (those are assessment, §4.3). `None` when the check was disabled.
    pub dt_check: Option<DtCheckLadder>,
    pub method: MethodRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChainCount { pub requested: usize, pub completed: usize }

/// Today's `chain_init_source` and `chain_starts_kind`, both required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartsRecord { pub rule: String, pub kind: ChainStartsKind }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MethodRecord {
    If2(If2Record),
    Nlopt { algorithm: NloptAlgorithm, record: NloptRecord },   // sbplx | bobyqa | lbfgs
    Sampler { sampler: Sampler, record: SamplerRecord },        // pgas | pmmh | mh
}

/// The point a method reports and a `from_mle` start reads. Replaces
/// `best_loglik`, `best_chain`, `loglik_type` and `start_values`, whose last
/// name described a use, not a quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointEstimate {
    pub chain: ChainIndex,
    pub loglik: f64,
    pub loglik_kind: LoglikType,                 // fit::loglik, unchanged
    pub theta: BTreeMap<String, f64>,            // estimated parameters only
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct If2Record {
    pub point: PointEstimate,                    // the clean-evaluation winner
    pub chain_final_logliks: Vec<(ChainIndex, f64)>,
    pub agreement: If2Agreement,
}

/// What the IF2 rule reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct If2Agreement {
    /// Tail chain-agreement Â per estimated parameter; NaN where the
    /// within-chain variance collapsed. Empty with one chain.
    pub tail_a: BTreeMap<String, f64>,
    /// Parameters perturbed only at t = 0. The rule judges them separately
    /// (§4.4). Named as method-scoped-settings D10 names it.
    pub initial_value_params: Vec<String>,
    /// `None` when clean evaluation did not run.
    pub clean_eval: Option<CleanEval>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanEval {
    /// One entry per evaluated chain, so a value cannot be attributed to the
    /// wrong chain (the gh#978 defect the parallel vectors allowed).
    pub chains: Vec<CleanEvalChain>,
    /// What the evaluation ran at: execution controls, recorded.
    pub n_particles: usize,
    pub n_replicates: usize,
    pub combine: CombineMode,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CleanEvalChain { pub chain: ChainIndex, pub loglik: f64, pub se: f64 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NloptRecord {
    pub point: PointEstimate,
    pub agreement: NloptAgreement,
}

/// What the NLopt rule reads. `chain_results.tsv` stays as the tabular view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NloptAgreement {
    pub chains: Vec<NloptChain>,
    /// The search box per estimated parameter: the relative range's denominator.
    pub search_box: BTreeMap<String, (f64, f64)>,
    /// `xtol_rel` as run: an execution control the absolute floor is scaled by.
    pub xtol_rel: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NloptChain {
    pub chain: ChainIndex,
    pub status: OptStatus,          // converged(reason) | max_eval | max_time | failed
    pub loglik: f64,
    pub theta: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerRecord {
    /// PMMH and MH: the MAP draw (marginal or ODE log-likelihood). PGAS: the
    /// best sweep (complete-data log-likelihood), as today; `loglik_kind` says
    /// which.
    pub point: PointEstimate,
    pub acceptance_rate: f64,
    /// PMMH only (gh#764); `None` for MH and PGAS.
    pub pf_noise: Option<PfNoiseCheck>,
    // The mixing measurements stay in `<sampler>_summary.json`
    // (camdl.fit-summary/v2), loaded as the existing `PosteriorDiagnostics`.
}

/// Today's `DtCheckResult` without `verdict`, `threshold_nats` and
/// `threshold_se_aware_nats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtCheckLadder {
    pub ladder: Vec<LadderEntry>,
    pub leg1_delta_nats: f64,
    pub leg2_delta_nats: f64,
    pub sigma_max_nats: f64,
    pub pf_se_inflation: bool,
}

impl ResultRecord {
    /// Refuses a file whose `schema` is not `RESULT_SCHEMA` (§6), a one-based
    /// chain layout (gh#990), and a parse failure. Every input path calls this.
    pub fn load(dir: &Path) -> Result<Self, RecordError>;
    pub fn save(&self, dir: &Path) -> Result<(), String>;
    /// The point a `from_mle` start reads.
    pub fn point(&self) -> &PointEstimate;
}
```

Fields of today's `FitState` that do not appear: `rw_sd` (deleted on branch
`fix/settings-silent-noops`, `e9358984`), `input_hash` (never set: every writer
passes `None`, `mod.rs:1477`, `nlopt_stage.rs:281`, `pmmh.rs:910`,
`pgas.rs:1637`; no reader), `resolved_gate` (policy, §4.3), `initial_loglik`
(written as −∞ or NaN by IF2, PGAS and NLopt, and as the pre-run mean only by
PMMH; it stays in PMMH's `SamplerRecord` only if a reader is found during Stage
3, and is otherwise deleted with the struct), and `n_good_chains` (now
`chains.completed`, always written). `resolved_loglik_eval` becomes
`CleanEval`'s three recorded fields.

### 4.2 Measurements

```rust
// rust/crates/cli/src/fit/assess.rs (new)

/// What `assess` reads: one variant per question, each with its own
/// statistics.
pub enum Measurements<'a> {
    If2(&'a If2Agreement),
    Nlopt(&'a NloptAgreement),
    Posterior { diag: &'a PosteriorDiagnostics, starts: ChainStartsKind },
    DtCheck { ladder: &'a DtCheckLadder, backend: InferenceBackend },
}

impl ResultRecord {
    /// The convergence measurement and, when the check ran, the dt-check one.
    /// A sampler's convergence measurement needs its summary JSON.
    pub fn measurements<'a>(&'a self, posterior: Option<&'a PosteriorDiagnostics>)
        -> Result<LeafMeasurements<'a>, AssessError>;
}
pub struct LeafMeasurements<'a> {
    pub convergence: Measurements<'a>,
    pub dt_check: Option<Measurements<'a>>,
}
```

### 4.3 The policy

```rust
// rust/crates/cli/src/fit/assess.rs

/// Every post-run threshold. Defaults are today's values (§5), so a fit judged
/// before and after this proposal under the default policy gets the same verdict
/// except where §8 says otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DiagnosticPolicy {
    pub if2: If2Policy,
    pub nlopt: NloptPolicy,
    pub posterior: PosteriorPolicy,
    pub dt_check: DtCheckPolicy,
}

pub struct If2Policy {
    pub a_thresh: f64,            // 1.01   (config_v2.rs:1863)
    pub a_warn: f64,              // 1.05   (A_SOFT, gating.rs:33)
    pub decibans_thresh: f64,     // 30.0   (config_v2.rs:1864)
    pub se_floor_sigmas: f64,     // 8.0    (gating.rs:256)
}
pub struct NloptPolicy {
    pub rel_range: f64,           // 0.05   (nlopt_stage.rs:735)
    pub abs_range_xtol: f64,      // 2.0    (nlopt_stage.rs:736)
    pub decibans_thresh: f64,     // 30.0   (shares GateConfig today)
}
pub struct PosteriorPolicy {
    pub max_rhat: f64,            // 1.05   (method_result.rs:585)
    pub severe_rhat: f64,         // 1.5    (method_result.rs:590), display band only
    pub min_ess: Option<f64>,     // None: not applied today (§3.3)
}
pub struct DtCheckPolicy {
    pub tolerance: DtTolerance,
    pub se_floor_sigmas: f64,     // 4.0    (dt_check.rs:128-130)
}
pub enum DtTolerance {
    /// 2.0 nats chain-binomial, 0.5 ODE (dt_check.rs:115-120).
    BackendDefault,
    /// Today's `--dt-check-strict`: 0.5 and 0.1.
    Strict,
    Nats(f64),
}

/// A policy as applied: every value, where each non-default one came from, and
/// which camdl's defaults filled the rest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppliedPolicy {
    pub values: DiagnosticPolicy,
    /// Dotted key (`posterior.max_rhat`) → source, for non-default values only.
    pub sources: BTreeMap<String, PolicySource>,
    pub defaults_from: String,    // crate::version::VERSION
    /// First 8 hex of the canonical-JSON digest of `values`: the short name a
    /// table column shows.
    pub digest: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicySource { FitConfig, Cli }

impl AppliedPolicy {
    /// CLI flags over the fit config's `[assessment]` over the defaults.
    pub fn resolve(fit_config: Option<&DiagnosticPolicySpec>, cli: &PolicyArgs)
        -> Result<Self, PolicyError>;
}
```

`DiagnosticPolicySpec` is the `[assessment]` table as written, every field
optional; `FitConfig` gains it as a third part beside `problem` and `inference`
(`config_v2.rs:183-186`), so the problem digest never sees it.
`config_identity_hash` (`cas.rs:467-471`), which a `fit.toml` handle is looked
up by, subtracts it beside `provenance`, so a config that differs only in its
policy finds the fit already stored.

```toml
# fit.toml
[assessment.if2]
a_thresh = 1.01
decibans_thresh = 30

[assessment.posterior]
max_rhat = 1.01
min_ess = 400
```

### 4.4 The assessment

```rust
// rust/crates/cli/src/fit/assess.rs

/// The one place measurements become a verdict.
pub fn assess(m: &Measurements<'_>, policy: &AppliedPolicy) -> Assessment;

/// A verdict cannot exist without the policy it was computed under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assessment {
    pub question: Question,
    pub verdict: Verdict,
    pub legs: Vec<Leg>,
    pub policy: AppliedPolicy,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Question { OptimizersAgree, ChainsMix, DtStable }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict { Pass, Warn, Fail, NotAssessed(NotAssessed) }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotAssessed { OneChain, PointStart, NoMeasurement, SamplerFailure(Vec<String>) }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Leg {
    pub kind: LegKind,
    pub verdict: Verdict,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
    /// Parameters (or chains) past the threshold, worst first.
    pub offenders: Vec<(String, f64)>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegKind {
    TailA, TailAInitialValue, DecibansSpread,          // IF2
    OptimizerStatus, ParameterRange,                   // NLopt (+ DecibansSpread)
    MaxRhat, MinEss,                                   // samplers
    DtHalving, DtPlateau,                              // dt-check
}

impl Assessment {
    /// `Pass` and `Warn` count as converged: `fit table`'s boolean.
    pub fn converged(&self) -> bool;
    /// `fit table`'s `gate_verdict` string (§8).
    pub fn code(&self) -> &'static str;
}
```

The rules, one per variant, each the rule the code already applies at its most
careful site:

- **IF2** (`check_scout_convergence`'s rule). The `TailA` leg fails when Â ≥
  `a_thresh` on a parameter not in `initial_value_params`, and warns in
  `[a_warn, a_thresh)`. The `TailAInitialValue` leg reports those parameters' Â
  and never fails the verdict. The `DecibansSpread` leg fails when the
  clean-evaluation spread in decibans is ≥
  `max(decibans_thresh,
  se_floor_sigmas · σ_max · NATS_TO_DB)`, and is
  `NotAssessed(NoMeasurement)` without clean evaluation, in which case the
  verdict is the Â leg's. With one chain the verdict is `NotAssessed(OneChain)`.
- **NLopt** (`check_convergence`'s rule, `nlopt_stage.rs:738-815`).
  `OptimizerStatus` fails unless every chain converged; `ParameterRange` fails
  on a parameter whose relative range exceeds `rel_range` and whose absolute
  range exceeds `abs_range_xtol · xtol_rel · max(|θ̂₀|, 1)`; `DecibansSpread`
  fails when the converged chains' spread exceeds `decibans_thresh`. With one
  chain, `ParameterRange` and `DecibansSpread` are `NotAssessed(OneChain)` and
  the verdict is the status leg's, as `fit table` does today.
- **Posterior** (`PosteriorDiagnostics::max_rhat_status`'s rule). `MaxRhat`
  fails at or above `max_rhat`, and an unassessable R̂ is
  `NotAssessed(SamplerFailure)` and sinks the verdict to `Fail`, as
  `sampler_source_verdict` does (`gating.rs:508-511`). `MinEss` fails below
  `min_ess` and is absent when `min_ess` is `None`. A point start makes the
  verdict `NotAssessed(PointStart)` (`demote_point_start`,
  `method_result.rs:397-415`, applied from `point_start`, `:935`).
- **dt-check** (today's `dt_check.rs` rule): `DtHalving` and `DtPlateau` against
  `max(tolerance, se_floor_sigmas · σ_max)`, `Warn` for a plateau leg in
  `[τ, 2τ)` as today's `Marginal`.

### 4.5 Where a verdict and its policy are recorded

Three kinds of surface report a verdict, and each carries its `AppliedPolicy`:

1. **At the end of a run**, the method's runner calls `assess` under the run's
   policy (CLI flags, then the fit config's `[assessment]`, then the defaults),
   prints it, and records it in `run.json` as
   `provenance.assessment = { convergence, dt_check }`. `provenance` is where
   recorded-only facts about one execution already live (`argv`, `created_at`,
   `host`; `runid/src/record.rs:144-161`), so the run's identity and its
   artifact set are untouched. `runid::Provenance` gains one optional
   `assessment: serde_json::Value`, keeping the `runid` crate free of fit types.
   A cache hit does not rewrite it.
2. **At read time**, `fit table`, `fit summary` and `fit predict` call `assess`
   under the reader's policy (CLI flags, then the `[assessment]` of the fit
   config the handle named, if it named one, then the running camdl's defaults)
   and print the policy beside the verdict. When the verdict recorded at run
   time differs, the row is marked and the recorded verdict and its policy
   digest are shown (§8). `--as-run` shows the recorded verdicts instead.
3. **The warm-start source check** (`starts = { from_mle = "@h" }`) assesses the
   source under the consuming run's policy and records the outcome in the
   consuming run's `run.json` `inputs.starts_source.assessment` (the field
   gh#896 adds, branch `fix/gh896-handles`). It is not hashed: the check decides
   whether the run proceeds, never what it computes.

### 4.6 Overriding the policy

```rust
// rust/crates/cli/src/args/mod.rs: flattened into FitRunArgs, FitTableArgs,
// FitSummaryArgs and FitPredictArgs.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct PolicyArgs {
    #[arg(long)] pub a_thresh: Option<f64>,
    #[arg(long)] pub decibans_thresh: Option<f64>,   // IF2 and NLopt, as today
    #[arg(long)] pub agreement_range: Option<f64>,   // NLopt rel_range
    #[arg(long)] pub max_rhat: Option<f64>,
    #[arg(long)] pub min_ess: Option<f64>,
    #[arg(long)] pub dt_check_tolerance: Option<f64>,
    #[arg(long, conflicts_with = "dt_check_tolerance")] pub dt_check_strict: bool,
    /// A TOML file holding an `[assessment]` table.
    #[arg(long)] pub policy: Option<PathBuf>,
}
```

Each flag has an observable effect or is refused (invariant I3): on `fit run`
and on `fit summary` or `fit predict` of one leaf, a flag for a question the
method does not answer (`--max-rhat` on an IF2 fit) is refused naming the
method; on `fit table`, which spans methods, a flag applies to the rows of its
question and is refused only when no row answers it. `--as-run` conflicts with
every policy flag.

## 5. Every threshold and tolerance, classified (D3)

"In identity today" was verified by reading the method's identity payload
(`config_v2.rs:1365-1400`) and `CliStageOverrides` (`:1635-1680`); a code
constant is covered by the engine version, which the fit level folds.

| value                                                                                                                                                  | where                                                                                                   | default                           | class                                               | in identity today                                               | after                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| IF2 `iterations`, `particles`, `chains`, `cooling`                                                                                                     | `Algorithm::IF2`, `config_v2.rs:995-1030`                                                               | per config                        | execution control                                   | yes                                                             | unchanged                                                            |
| IF2 `cooling_target_iters`, `--cooling-target-iters`                                                                                                   | `config_v2.rs:1011`, `:1596`                                                                            | 50 (`:1752`)                      | execution control (cooling)                         | yes                                                             | unchanged                                                            |
| IF2 clean evaluation `n_particles`, `n_replicates`, `combine`                                                                                          | `LoglikEvalConfig`, `config_v2.rs:1736-1750`                                                            | 4000 × 8, logmeanexp              | execution control (what is measured)                | yes                                                             | unchanged; recorded in `CleanEval`                                   |
| dt-check `enabled`, `n_halvings`, `n_particles`, `n_replicates`                                                                                        | `DtCheckConfig`, `config_v2.rs:1886-1925`                                                               | on, 2                             | execution control (what is measured)                | yes                                                             | unchanged                                                            |
| NLopt `tolerance` (`xtol_rel`), `max_evals`                                                                                                            | `config_v2.rs:1256-1263`                                                                                | 1e-6, 5000 (`:1276-1277`)         | execution control (stopping)                        | yes                                                             | unchanged; `xtol_rel` recorded                                       |
| PMMH/MH `iterations`, `burn_in`, `thin`, `adapt`, `adapt_start`                                                                                        | `Algorithm::PMMH`/`Mh`; defaults `pmmh.rs:37-38`                                                        | 5000 burn-in, thin 10             | execution control                                   | yes (`iterations` as target length)                             | unchanged                                                            |
| PMMH adaptation target 0.234 + 0.206/d                                                                                                                 | `sim/src/inference/pmmh.rs:320-322`                                                                     | —                                 | execution control (adaptation)                      | code                                                            | unchanged                                                            |
| PGAS `sweeps`, `burn_in`, `thin`, NUTS settings                                                                                                        | `Algorithm::PGAS`; defaults `cli/src/fit/pgas.rs:217-218`                                               | 2000, 5                           | execution control                                   | yes                                                             | unchanged                                                            |
| PGAS MH-within-Gibbs target 0.44; NUTS dual-averaging target 0.80                                                                                      | `sim/src/inference/pgas.rs:3953`, `:4110`, `:4416`                                                      | —                                 | execution control (adaptation)                      | code                                                            | unchanged                                                            |
| particle-filter substep budget                                                                                                                         | `ITER_BUDGET`, `sim/src/inference/degeneracy.rs:81`                                                     | 1e10                              | execution control (termination)                     | code (fit); `profile --pf-max-substeps` keyed, `profile.rs:160` | unchanged                                                            |
| IF2 `gate.a_thresh`                                                                                                                                    | `GateConfig`, `config_v2.rs:1855`                                                                       | 1.01 (`:1863`)                    | **assessment policy**                               | **yes**                                                         | `[assessment.if2] a_thresh`, `--a-thresh`                            |
| IF2 and NLopt `gate.decibans_thresh`, `--decibans-thresh`                                                                                              | `config_v2.rs:1860`, `:1597`, `:1609`                                                                   | 30 (`:1864`)                      | **assessment policy**                               | **yes**                                                         | `[assessment.if2]`, `[assessment.nlopt]`, `--decibans-thresh`        |
| SE floor `8 · σ_max` on the decibans leg                                                                                                               | `gating.rs:256`, copies `:290`, `method_result.rs:1109`, `fit_summary.rs:1549`, `:3050`                 | 8                                 | assessment policy                                   | code                                                            | `if2.se_floor_sigmas`                                                |
| Â warning band `A_SOFT`                                                                                                                                | `gating.rs:33`                                                                                          | 1.05                              | assessment policy                                   | code                                                            | `if2.a_warn`                                                         |
| NLopt relative-range and absolute-floor thresholds                                                                                                     | `nlopt_stage.rs:735-736`; literal `table_row.rs:294`                                                    | 0.05, 2.0                         | assessment policy                                   | code                                                            | `nlopt.rel_range`, `nlopt.abs_range_xtol`                            |
| R̂ certification bar                                                                                                                                    | `RHAT_CONVERGED_THRESHOLD`, `method_result.rs:585`                                                      | 1.05                              | assessment policy                                   | no (read time, unrecorded)                                      | `posterior.max_rhat`, recorded                                       |
| R̂ severe band                                                                                                                                          | `RHAT_SEVERE_THRESHOLD`, `method_result.rs:590`                                                         | 1.5                               | assessment policy (display)                         | no                                                              | `posterior.severe_rhat`                                              |
| minimum ESS                                                                                                                                            | —                                                                                                       | none                              | assessment policy                                   | —                                                               | `posterior.min_ess`, default `None`                                  |
| dt-check `threshold_nats`, `--dt-check-strict`                                                                                                         | `config_v2.rs:1920`; `dt_check.rs:115-120`; stored `:83-86`                                             | 2.0 / 0.5 nats (strict 0.5 / 0.1) | **assessment policy**                               | **yes** (gh#730)                                                | `[assessment.dt_check]`, `--dt-check-tolerance`, `--dt-check-strict` |
| dt-check SE floor `4 · σ_max`                                                                                                                          | `dt_check.rs:128-130`                                                                                   | 4                                 | assessment policy                                   | code                                                            | `dt_check.se_floor_sigmas`                                           |
| run-time findings: Â > 1.1, Â > 1.5 with spread > 50, ESS at θ̂ < 5 %, R̂ report 1.1, loglik spread 3·3 nats, chain outlier \|z\| > 3.5, starved ESS 1 % | `runner.rs:2565-2569`, `:2587`, `:2923`; `gating.rs:106`; `chain_diagnostics.rs:99`; `filter_ess.rs:60` | as listed                         | assessment (findings written to `diagnostics.json`) | code                                                            | unchanged; out of scope (§13)                                        |

The three bold rows are the ones that move a value out of run identity. Every
other assessment value is a code constant today and becomes a policy field with
the same default, so no stored fit's verdict moves under the default policy
except as §8 states.

## 6. Old records are refused as inputs and shown read-only (D4)

**The tag.** A `fit_state.toml` written after Stage 3 begins
`schema = "camdl.fit-result/v1"` and `producer = "<camdl version>"`.
`ResultRecord::load` refuses any other file:

```rust
pub enum RecordError {
    /// Any file without `schema = RESULT_SCHEMA`.
    Unsupported {
        path: PathBuf,
        /// The file's `schema`, or `None` for every record before v1.
        schema: Option<String>,
        /// The file's `producer`, else its `camdl_version`, else the
        /// `# Generated by <version>` line `FitState::save` writes
        /// (`state.rs:202`), else `None`.
        producer: Option<String>,
    },
    OneBasedLayout(String),     // gh#990, unchanged
    Parse { path: PathBuf, msg: String },
}
```

The message names both versions and both ways forward:

```
error: fits/sir-198ada3a/if2-607db55a/seed_1-06cbd6b3/fit_state.toml was
  written by camdl 0.1.0+fc2039b2 with no result schema (before
  camdl.fit-result/v1). This camdl reads camdl.fit-result/v1 and will not
  start a new computation from it. Re-run the fit, or read it with
  `camdl fit summary <leaf>`, which shows an old record without judging it.
```

**Input paths**, which call `ResultRecord::load` and so refuse: the `from_mle`
start (`chain_starts.rs:1031`), `profile`'s `from_mle` start
(`profile.rs:1953`), the runner reading back the record it just wrote
(`mod.rs:2101`), and the warm-start source check.

**The read-only path.** `fit summary` and `fit table` read a leaf through
`LeafView`:

```rust
pub enum LeafView {
    Current { record: ResultRecord, posterior: Option<PosteriorDiagnostics> },
    /// A record of another schema: shown, never assessed, never an input.
    Legacy(LegacySummary),
}
/// Read from the raw TOML value, with no typed compatibility struct: the keys
/// present are shown, absent ones are omitted.
pub struct LegacySummary {
    pub producer: Option<String>,
    pub schema: Option<String>,
    pub method: Option<String>,
    pub seed: Option<u64>,
    pub best_loglik: Option<f64>,
    pub loglik_type: Option<String>,
    pub best_chain: Option<i64>,
    pub estimates: BTreeMap<String, f64>,   // the old `start_values`
}
```

`fit summary` prints a legacy leaf as one block (producer, method, seed, the
point estimate and its log-likelihood) with the line "record predates
camdl.fit-result/v1: not assessed, not usable as a start"; `fit table` shows its
row with `converged` empty and `gate_verdict = "legacy"`. Neither exits non-zero
for a legacy leaf. That is the whole compatibility surface: no field of an old
record is interpreted beyond display, so an old record can never be judged by a
rule it was not written for. The release that ships Stage 3 re-keys every fit
anyway (the fit level folds the engine version, `cas.rs:508-520`), so a re-run
produces a current record under a new `run_id` beside the old one.

## 7. Call sites

| caller                              | today                                                                      | after                                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| warm-start source check             | `check_scout_convergence` or `sampler_source_verdict`, `gating.rs:439-512` | `assess` of the source under the consumer's policy; refuses on `Fail` unless `--allow-nonconverged-source` |
| `fit table`                         | `table_row.rs:289-372`, one rule per `from_*`                              | `TableRow::from(&LeafView, &AppliedPolicy)` through `assess`                                               |
| `fit summary` text                  | `gate_verdict_block`, `fit_summary.rs:1520-1600`; R̂ at `:1819-1825`        | renders `Assessment.legs`                                                                                  |
| `fit summary` JSON, Markdown, LaTeX | `if2_stage_report`, `fit_summary.rs:3028-3070`                             | serializes `Assessment`                                                                                    |
| end of an IF2 run                   | Â table against `config.gate`, `runner.rs:2530-2552`                       | `assess`, printed, recorded in `provenance.assessment`                                                     |
| end of an NLopt run                 | `check_convergence` + `print_verdict`, `nlopt_stage.rs:218`, `:818`        | `assess`, printed, recorded                                                                                |
| `fit predict`                       | prints `fit_rhat_max` / `fit_ess_min`, no verdict                          | prints the fit's assessment once in its launch block; TSV columns unchanged                                |

`check_scout_convergence`, `compute_if2_gate_verdict`, `GateVerdict`,
`If2StageResult.gate_verdict`, `NloptStageResult.max_rel_range`'s verdict use,
`check_convergence`'s verdict fields, and `GateConfig` are deleted.
`format_hard_verdict` and `format_decibans_spread_verdict` (`gating.rs:284-424`)
stay as renderers of an `Assessment`'s legs.

## 8. What users see change

Under the default policy:

1. An IF2 fit whose only Â failure is on an initial-value parameter is reported
   converged by `fit table` and `fit summary`, as the source check already
   reports it (§3.1).
2. `fit summary --format json` gives an IF2 fit without clean evaluation an
   overall verdict from its Â leg, where it gave `null`.
3. `fit table` judges an NLopt fit by the rule its run printed: the absolute
   floor can turn a failure into a pass, and a decibans spread past 30 dB turns
   a pass into `fail_db`.
4. An NLopt leaf used as a `from_mle` source is refused when its chains
   disagree, where today nothing refuses it (§3.2).
5. A single-chain IF2 fit's row reads `not_assessed` with `converged` false,
   where it read `pass`, as a single-chain sampler already does.
6. `gate_verdict` for sampler rows reads `pass`, `fail_rhat`, `fail_ess` or
   `not_assessed`, where it read `n/a` beside a `converged` of false. IF2 rows
   keep `pass`, `fail_a`, `fail_db`, `fail_both`; NLopt rows keep
   `fail_chain_agreement` and `fail_status` and gain `fail_db`.
7. `fit table` and `fit summary` print the policy they judged under: one line
   naming its digest and every non-default value. The JSON row gains
   `assessment` and `recorded_assessment` and its `TableRowSchema` version
   (`table_row.rs:39-42`) is bumped.
8. A row whose read-time verdict differs from the one recorded at run time is
   marked `*`, with a footer giving the recorded verdict and its policy digest.

Configuration and flags:

9. `[method.gate]` and `[method.dt_check] threshold_nats` are refused with the
   new spelling (`[assessment.if2]`, `[assessment.dt_check] tolerance`) and the
   value converted. `--decibans-thresh` and `--dt-check-strict` keep their names
   and stop re-keying.
10. Changing a threshold no longer re-runs a fit: `fit run` with a new
    `--decibans-thresh` is a cache hit that prints the new verdict.
11. The source check judges a source under the consuming run's policy, not the
    gate the source was run with.

## 9. Stages

Each stage is green under `make test` alone. None edits a high-risk engine file
(`sim/src/inference/{pgas,pgas_grad,if2,particle_filter}.rs`, `obs_loglik.rs`,
`obs_model.rs`); Stage 3 edits the record construction in the CLI drivers
`cli/src/fit/pgas.rs`, `pmmh.rs`, `nlopt_stage.rs` and the IF2 arm of
`cli/src/fit/mod.rs`, and the implementer reads each full function first.
`runner.rs` is not split by any stage; its split comes after the stages of this
proposal and of method-scoped-settings have landed.

The fixed-seed check named in Stages 1–4 runs `tests/recovery/cases/sir` under
IF2, `nl-sbplx`, PMMH and PGAS at seed 1 before and after the stage and
compares: `draws.tsv`, every `chain_*` TSV, `chain_results.tsv`,
`chain_evaluations.tsv` and `mle_params.toml` byte for byte, and every numeric
field of `fit_state.toml` by value (its layout changes in Stages 2 and 3).

**Stage 0: dead field.** Delete `FitState.input_hash` (§4.1). Its own commit.

**Stage 1: one rule per measurement type, on today's storage.** Add
`fit/assess.rs` with `Measurements`, `DiagnosticPolicy` (defaults as §5),
`AppliedPolicy`, `Assessment` and `assess`. Today's records are adapted into
`Measurements` by a loader in the same module (IF2 from `FitState`; NLopt from
`chain_results.tsv`, the fit sidecar's search box and the method's `tolerance`;
samplers from `PosteriorDiagnostics`). Every caller of §7 except the end-of-run
reports moves to `assess`; the policy is `resolved_gate` where present, else the
defaults, exactly as today. If branch `fix/settings-silent-noops` has landed a
shared `assess_if2`, it becomes the IF2 arm unchanged. Tests, each red on
`a9a89178`:

- the settings-fixes agreement test (§3.1), extended to
  `fit summary --format
  json`;
- an NLopt leaf whose chains disagree in relative but not absolute range is
  `converged` in `fit table` (item 3);
- an NLopt leaf with chains at opposite ends of the box is refused as a
  `from_mle` source (item 4);
- a single-chain IF2 row is `not_assessed` (item 5).

Each is mutation-checked: restoring the old rule at its site turns its test red.
Fixed-seed check: no writer changes, so all outputs are byte-identical.

**Stage 2: thresholds leave run identity (D3).** `FitConfig` gains
`assessment: DiagnosticPolicySpec`; `GateConfig` leaves `Algorithm::IF2` and the
NLopt config; `DtCheckConfig.threshold_nats` leaves `[method]`;
`CliStageOverrides` loses `decibans_thresh` and `dt_check_strict`; `PolicyArgs`
is flattened into `fit run`, `fit table` and `fit summary`;
`config_identity_hash` subtracts `assessment`; `FitState.resolved_gate` and
`DtCheckResult`'s `verdict`, `threshold_nats` and `threshold_se_aware_nats` are
removed; the IF2 and NLopt runners call `assess` at the end and record the
result in `provenance.assessment`; `fit table --as-run` and the `*` marker land.
This re-keys every IF2, NLopt and dt-checked MH fit once (their method payload
loses the gate and the tolerance); the release re-keys every fit regardless.
Tests:

```rust
/// D3: a threshold changes the verdict a reader sees, not the run.
#[test]
fn a_policy_threshold_moves_the_table_verdict_not_the_run_id() {
    let root = fit_run(&sir_pmmh_toml(), &[]);                  // `camdl fit run`
    let leaf = only_leaf(&root);
    let id = run_id(&leaf);
    let state = std::fs::read(leaf.join("fit_state.toml")).unwrap();

    let before = fit_table_json(&root, &[]);
    let rhat = before.row.max_rhat.unwrap();
    // A bar just past the measured R̂, on the side that flips the verdict.
    let bar = if before.row.converged { rhat - 1e-6 } else { rhat + 1e-6 };
    let after = fit_table_json(&root, &["--max-rhat", &bar.to_string()]);
    assert_ne!(before.row.converged, after.row.converged);
    assert_eq!(after.row.assessment.policy.values.posterior.max_rhat, bar);
    assert_eq!(before.row.max_rhat, after.row.max_rhat, "measurements unchanged");

    // The same threshold given to `fit run` is a cache hit on the same leaf.
    let root2 = fit_run(&sir_pmmh_toml(), &["--max-rhat", &bar.to_string()]);
    assert_eq!(only_leaf(&root2), leaf);
    assert_eq!(run_id(&leaf), id);
    assert_eq!(std::fs::read(leaf.join("fit_state.toml")).unwrap(), state);
}
```

The same test is repeated for IF2 with `--a-thresh`, and for
`--decibans-thresh`, which today fails it (red on `a9a89178`, _inferred_ from
the identity payload: the second `fit run` lands in a new leaf). Also:
`[method.gate]` and `[method.dt_check]
threshold_nats` refused with the
converted spelling; `--max-rhat` refused on an IF2 `fit run` and on
`fit summary` of an IF2 leaf; `--as-run` refused beside a policy flag; a leaf
whose recorded verdict differs from the read-time one is marked. Fixed-seed
check.

**Stage 3: per-method result records (D4).** `ResultRecord` replaces `FitState`;
the four writers construct their `MethodRecord` variant; the input paths of §6
call `ResultRecord::load`; `fit table` and `fit summary` read `LeafView`;
`MethodResult` is rebuilt on `ResultRecord` without verdict fields. Lands after
method-scoped-settings commit 3 (§10). Tests: each variant round-trips; a
committed pre-v1 `fit_state.toml` fixture is refused as a `from_mle` source with
both versions in the message, and is shown by `fit
summary` and `fit table` with
exit 0; an `if2` record whose body is `[nlopt]` fails to parse. Fixed-seed
check.

**Stage 4: `fit predict` reports the assessment.** `PolicyArgs` on
`fit
predict`; its launch block prints the fit's assessment and policy. Every
TSV it writes is byte-identical at a fixed seed.

## 10. Dependencies on work in flight

- **`fix/settings-silent-noops`** (`e9358984` plus an uncommitted test in its
  worktree): deletes `FitState.rw_sd` and refuses inert flags. Stage 1 builds on
  its IF2 agreement test and on any `assess_if2` it lands. Its refusal of inert
  flags is the model for `PolicyArgs`' refusals (§4.6).
- **`docs/method-scoped-settings`** (being revised): its D10 renames
  `perturb_only_at_t0_params` to `initial_value_params` and refuses the old key
  in `FitState::load`. Stage 3 lands after its commit 3, takes the name, and
  replaces the key-specific refusal with the schema refusal of §6. Its
  `ExecutionPlan` carries no policy: `MethodPlan` holds execution controls only,
  and `GateConfig` leaving `[method]` in Stage 2 removes the one assessment
  field its `If2Plan` would otherwise inherit. Either order of Stage 2 and its
  commit 3 works; both edit `Algorithm::IF2`, so the second rebases.
- **`2026-10-26-parameter-roles.md`** (accepted) and its Stage 2: no type here
  depends on `ResolvedProblem`; `NloptAgreement.search_box` is filled from the
  search box the partition resolves once it exists, and from the fit sidecar
  until then.
- **`feat/support-transforms`**: changes which estimates are reported near a
  bound (`877c517a`), a run-time finding this proposal leaves alone (§13), and
  edits `if2.rs` and the NLopt start code; no stage here touches those lines.
- **`fix/gh896-handles`**: a handle resolves to one leaf, and a sourced run
  records `inputs.starts_source`. §4.5 item 3 adds `assessment` to that object;
  if this lands first, the source check records nothing until gh#896 lands. Rows
  of `fit table` are per leaf on that branch, which is what a per-leaf verdict
  needs.

## 11. Decisions made in this proposal

**Policy lives in a top-level `[assessment]` table, not in `[method]`.**
`[method]` is what the engine runs (method-scoped-settings), and the problem
digest must not see a threshold either. A third part of `FitConfig` is outside
both by construction, so no hash needs a subtraction except the config lookup.

**The run-time verdict is recorded in `run.json` provenance, not in a leaf
file.** A leaf file is in the artifact set, and a policy-dependent byte there
would make the policy identity under the run-identity rule, which is how
`--decibans-thresh` came to re-key. Provenance already varies between executions
of one `run_id`.

**The reader's policy is the default view, with a marker where it disagrees with
the recorded one.** The verdict a user acts on should be the standard they hold
now; the marker and `--as-run` keep the historical verdict visible, so nothing
flips silently. The alternative, showing recorded verdicts by default, would let
a tightened default leave old fits reading as converged.

**The source check uses the consumer's policy.** The question is whether this
run should start from that fit, which is the consumer's judgement. The outcome
is recorded in the consumer's `run.json`.

**Each rule is the most careful one the code already has.** IF2 takes
`check_scout_convergence`'s, NLopt takes the stage's own, samplers keep theirs.
No statistic is new.

**Legacy records are read as raw TOML for display only.** A typed compatibility
struct would be a second schema to maintain; displaying the keys present is
enough to inspect an old fit, and refusing it as an input is what D4 asks.

**`initial_loglik` is deleted unless Stage 3 finds a reader.** Three of four
writers fill it with a sentinel.

## 12. Pending the maintainer

**The default R̂ and ESS values.** `PosteriorPolicy` ships with `max_rhat = 1.05`
and `min_ess = None`, today's behaviour. Vehtari et al. (2021) recommend R̂ <
1.01 for the rank-normalized statistic and a bulk ESS above 400 before R̂ is
read; adopting them is the gh#84 threshold decision, already with the
maintainer. Under this proposal it is a change of two defaults, and every stored
Bayesian fit whose verdict it changes is marked in `fit table` (§8 item 8)
rather than silently re-judged.

## 13. Out of scope, each with its reason

- **The run-time findings in `diagnostics.json`** (§5, last row): their
  thresholds are code constants, so they cannot differ within one `run_id`, and
  they describe a run rather than certify it. Moving them to read time is a
  separate change if a finding ever needs a user-settable threshold.
- **`profile` and `survey` verdicts**: they report likelihood surfaces, not
  convergence; neither calls any verdict function listed in §3.
- **`[estimate].start` in the problem digest**
  (`docs/dev/inference-architecture.md` I1): a problem-identity question, owned
  by the `ResolvedProblem` work, not by result records.
- **Splitting `runner.rs`**: after these stages, as decided.
