---
date: 2026-10-29
status: proposed — implements maintainer decisions D3 (assessment thresholds leave run identity) and D4 (refuse unsupported old records as inputs, keep them inspectable; method-scoped-settings §1 decision 6, recorded there as its D16) of 2026-10-29, builds on D2 (method-scoped-settings §1 decision 4), and adopts the external statistician's review the maintainer accepted; §11 records the calls made under them; §12 records the maintainer's decision on the default thresholds (gh#84); nothing is pending
area: fit result records / convergence verdicts / run identity / fit table, fit summary, fit predict
issues: none filed; §13 lists what is out of scope and why, and no follow-up is open
related:
  - ../inference-architecture.md (the invariants this implements: I5, I6, I7, I8)
  - 2026-10-26-parameter-roles.md (accepted; untouched by this proposal)
  - 2026-10-29-method-scoped-settings.md (branch docs/method-scoped-settings at 2df329b0; §10 here states the order: Stage 3 after its D16 loader-split commit)
---

# A fit stores what it measured; whether it converged is computed when someone asks, under a policy that is named

Code references are against `origin/main` at `a9a89178` unless marked with a
branch. A claim marked _verified_ was read at the cited line or produced by the
command shown; _inferred_ means reasoned from the code and not executed. Nothing
in this document was measured by running a fit. The concepts and invariants this
builds on are stated in `docs/dev/inference-architecture.md`; this document uses
its terms (`ResolvedProblem`, `ExecutionPlan`, `RunRecord`, result record,
`Assessment`, assessment record).

## 1. Every method writes one flat record, and its verdict is computed four ways

A fit leaf (one method, one seed, under one problem) stores its outcome in
`fit_state.toml`, deserialized as `FitState` (`cli/src/fit/state.rs:19-182`,
verified). IF2, the three NLopt optimizers, PMMH, MH and PGAS all write that one
struct, each leaving the others' fields at an empty or `None` sentinel. NLopt
writes its per-parameter relative range across chains into
`tail_chain_agreement`, the field named for IF2's chain-agreement statistic Â
(`nlopt_stage.rs:297`, verified). The samplers write their selected draw into
`start_values`, the field `from_mle` reads as a maximum-likelihood estimate
(§3.6).

Whether a leaf converged is then decided in several places, by different rules,
against thresholds of which some are part of the run's identity and some are
not. Raising `--decibans-thresh` re-keys and re-runs an IF2 fit that has already
been computed, because the threshold is written into `[method]` before its
identity is taken (`config_v2.rs:1597`, `:1652-1655`, verified); lowering the R̂
bar in the source would silently re-judge every stored Bayesian fit, because the
bar is a constant applied at read time and recorded nowhere
(`method_result.rs:585`, verified).

This proposal makes four changes, in independently landable stages (§9):

- **One rule per measurement type.**
  `assess(measurements, policy) -> Assessment` is the only function that turns
  measurements into a verdict. `fit table`, `fit summary`, the warm-start source
  check and `fit predict` call it. Optimizer agreement (IF2, NLopt) and
  posterior mixing (PGAS, PMMH, MH) keep distinct measurement types and distinct
  statistics; what is unified is the signature, the verdict type and the rule
  that every verdict names its policy. A diagnostic that cannot be computed is
  reported as unavailable and never yields a pass.
- **Thresholds leave run identity (D3).** A `DiagnosticPolicy` holds every
  post-run threshold, lives in a new `[assessment]` table and in read-time
  flags, and is in no hash and in no run record. The verdict a run reached when
  it finished is kept in a separate, write-once assessment record beside the
  leaf.
- **Typed result records with a schema tag (D4).** `FitState` is replaced by a
  per-method `ResultRecord` carrying `schema` and `producer`. A record of any
  other schema is refused as an input to a new computation with a versioned
  error, and is still shown, read-only, by `fit summary`, `fit table` and
  `fit predict`.
- **An optimum is not a draw.** Only IF2 and NLopt records carry a point
  estimate; a sampler records a selected draw with its selection criterion, and
  `from_mle` accepts only an optimizer record.

## 2. Terms

A **measurement** is a number the run computed and stored: a tail Â per
parameter, a clean-evaluation log-likelihood and its standard error per chain, a
rank-normalized R̂ (the between-chain potential scale reduction, Vehtari et al.
2021, _Bayesian Analysis_ 16(2):667–718) or a bulk or tail ESS (effective sample
size) per parameter, an optimizer's per-chain θ̂ and exit status, a dt-check
ladder.

**Optimizer agreement** asks whether independent optimizer chains (IF2 or NLopt)
reached the same maximum. **Posterior mixing** asks whether sampler chains
explore the same distribution. They are different questions with different
statistics; an Â of 1.02 and an R̂ of 1.02 are not comparable numbers (the
distinction `If2StageResult.max_chain_agreement`'s doc already draws,
`method_result.rs:79-84`). Both are **statistical-estimation diagnostics**. The
dt-check, particle-filter degeneracy and sampler divergences are
**numerical-execution diagnostics**: they ask whether the computation was done
accurately, not whether the estimate is settled. An assessment keeps the two
kinds apart.

An **assessment policy** is the set of thresholds that turn measurements into a
verdict. An **execution control** changes what the engine computes or when it
stops. `docs/dev/inference-architecture.md` ("What counts as which") is the
normative definition; §5 here classifies every current value.

A **leg** is one diagnostic within an assessment: one statistic, its value, its
threshold and its outcome. A leg is **required** when the overall verdict
depends on it. Each leg ends in one of four outcomes: it passes; it passes with
a named qualification; it fails against its threshold; or its measurement is
**unavailable**, with the reason. The overall verdict is defined in §4.4.

An **assessment record** is a persisted assessment: written once, beside the
leaf, outside `run.json` and outside every addressed artifact (§4.5).

## 3. What the code does today

### 3.1 The IF2 verdict: four implementations, three rules

| site                                                        | initial-state parameters (`perturb_only_at_t0`) | decibans leg missing | warning band (`A_SOFT`)                                                                              |
| ----------------------------------------------------------- | ----------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `check_scout_convergence`, `gating.rs:182-260`              | exempt from the Â leg                           | judged on Â alone    | yes, `SoftWarn`, empty at the default `a_thresh` (`gating.rs:27-30`)                                 |
| `compute_if2_gate_verdict`, `method_result.rs:1082-1123`    | included                                        | judged on Â alone    | no                                                                                                   |
| `Formatter::gate_verdict_block`, `fit_summary.rs:1520-1560` | included                                        | prints the Â leg     | no: the leg is `max_a < a_thresh` (`:1532`); per-parameter glyphs use `GateConfig::a_band` (`:1512`) |
| `if2_stage_report`, `fit_summary.rs:3028-3056`              | included                                        | `overall_pass: None` | no                                                                                                   |

All verified by reading. The first is what the warm-start source check applies
(`check_source_converged`, `gating.rs:439-480`); the second is `fit table`'s
`converged` and `gate_verdict` (`table_row.rs:320-337`); the third and fourth
are `fit summary`'s text and its JSON, Markdown and LaTeX outputs. The JSON
document embeds the `fit table` row (`FitSummaryDoc.table_row`,
`fit_summary.rs:2622-2629`) beside the per-method report, so one
`fit summary --format json` can carry two IF2 verdicts computed by different
rules. An IF2 fit whose only Â failure is on an initial-state parameter is
therefore accepted as a `from_mle` source and reported as failed by `fit table`
and `fit summary`.

In all four, a NaN Â (within-chain variance collapsed) is skipped by
`fold(0.0, f64::max)` and so passes silently (`gating.rs:202`,
`method_result.rs:1087-1091`). The end-of-run report says to "rely on Δ_dB leg"
when every Â is NaN (`runner.rs:2525-2528`).

Branch `fix/settings-silent-noops` adds a test pinning that the sites agree
(`if2_verdict_is_the_same_for_the_table_the_summary_and_a_starts_source`,
uncommitted in its worktree on `e9358984`), and is adding the shared IF2 verdict
function `assess_if2`, folding in `if2_stage_report` and the NLopt source-check
defect of §3.2. Its test asserts the initial-state exemption, which §4.4 here
removes; §10 states the consequence.

### 3.2 The NLopt verdict: two rules, and a source check that cannot fail

At the end of a run, `check_convergence` (`nlopt_stage.rs:738-815`, verified)
fails the agreement leg on a parameter only when its relative range across
chains exceeds `DET_REL_RANGE_THRESH = 0.05` _and_ its absolute range exceeds
`DET_ABS_RANGE_FACTOR · xtol_rel · max(|θ̂₀|, 1)` (`:735-736`, `:782-786`), and
fails a decibans leg when the converged chains' log-likelihood spread exceeds
`gate.decibans_thresh` (`:803`). It prints this verdict and stores neither leg's
outcome. `fit table` instead declares an NLopt leaf converged when every chain
converged and `max_rel_range < 0.05` (`table_row.rs:293-294`): no absolute floor
and no decibans leg.

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
accessor at `:513-536`). Bulk and tail ESS are stored per parameter
(`ParamConvergence::Scored.ess_bulk`, `.ess_tail`, `method_result.rs:195-197`).

When the chains began at one point, `demote_point_start`
(`method_result.rs:397-415`) overwrites every R̂ with `Undefined`, and
`fit predict` reports the fit as not assessed through `point_start`
(`method_result.rs:935`, `predict.rs:789`, `:865`). The R̂ was computed and is
discarded.

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

### 3.5 A warm start's source is already execution identity

A run that starts from a stored fit names its source in the method level's
`deps`: the source leaf's `run_id` and the consumed file's digest
(`fit_state.toml` for `from_mle`, `draws.tsv` for `from_posterior`;
`chain_starts.rs:424-443`, `cas::cas_leaf_file_dep`, `cas.rs:658-670`, folded
into `deps` at `mod.rs:1049-1050`; verified by reading). The starts rule itself
is in the method's identity payload (`config_v2.rs:1390-1392`), and the drawn
starts are a function of the rule, the seed and the consumed file
(`chain_starts.rs:999-1013` for `from_posterior` rows). So changing the source
run changes the `run_id`. One coupling remains: the consumed `fit_state.toml`
today contains `resolved_gate`, so the source's policy enters the consumer's
identity through that digest. Stage 2 removes it (§9).

### 3.6 A sampler's selected draw is read as an MLE

`draw_from_mle` puts every chain at `FitState.start_values`, documented as "the
clean-eval winner for IF2, the MAP for a sampler, the best chain for an
optimizer" (`chain_starts.rs:1017-1041`, verified). What the samplers write
there:

- **PGAS** writes the parameters of the sweep with the highest complete-data
  log-likelihood, `best_sweep.log_complete_data_ll` (`fit/pgas.rs:1621-1630`,
  `:1639`, verified). That is an argmax over (X, θ) of p(Y, X | θ) among visited
  sweeps, not an estimate of the marginal MLE.
- **PMMH and MH** write `map_result.map_params` (`fit/pmmh.rs:851-873`,
  verified): the visited state with the largest log-likelihood plus
  natural-scale log prior, without the transform's Jacobian, tracked at every
  step including warm-up (`sim/src/inference/pmmh.rs:747-751`, verified). For
  PMMH the log-likelihood is a particle-filter estimate, so the maximum is
  biased upward by selection on noise (_inferred_). It is not a reliable MAP:
  the mode depends on the coordinate measure, which is not stated.

Both are accepted as `from_mle` sources: `resolve_starts` does not look at the
method (`chain_starts.rs:412-443`), and `profile`'s `--starts from_mle` reads
the same field (`profile.rs:1949-1957`). The chains' starts are then recorded as
`mle_point` in `chain_starts.tsv` (`chain_starts.rs:80`). `fit summary` labels
the PMMH and MH value "MAP loglik" (`fit_summary.rs:1764-1765`). These are the
user-facing places where a sampler's selected draw is presented as an MLE or a
MAP. Samplers write no `final_params.toml` at the leaf root (only IF2 does,
`mod.rs:1416`; NLopt writes `mle_params.toml`, `nlopt_stage.rs:320-336`), so
`fit summary --params-only` and `compare`'s `winner_params_toml`
(`fit_summary.rs:3551-3590`) do not read a sampler draw as a winner (_inferred_:
they fail to find the file).

### 3.7 Precedents this proposal follows

- **The applied policy is already recorded for IF2** (`resolved_gate`, with
  `fit summary` reporting `default_fallback` when absent,
  `fit_summary.rs:3028-3031`). The proposal keeps that idea and moves the record
  out of the measurements into an assessment record of its own.
- **A schema tag that refuses another vintage** exists for
  `<sampler>_summary.json` (gh#728, `run_meta.rs:44-71`).
- **A record of an old layout is refused by its loader**:
  `chain_index::refuse_one_based_layout` in `FitState::load`
  (`state.rs:189-190`, gh#990).
- **The store already skips files by name**: `is_reserved`
  (`runid/src/store.rs:951-953`) exempts `run.json`, its tmp and the locks from
  the exact-set check.
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
    /// Always `RESULT_SCHEMA`; the loaders refuse anything else (§6).
    pub schema: String,
    /// `crate::version::VERSION` of the camdl that wrote the record.
    pub producer: String,
    pub seed: u64,
    pub timestamp: String,
    pub chains: ChainCount,
    pub starts: StartsRecord,
    /// The dt-check ladder at the reported point; no verdict and no tolerance
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
    Optimizer(OptimizerRecord),
    Sampler(SamplerRecord),
}

/// The methods whose output is a maximum of the likelihood.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerRecord {
    If2(If2Record),
    Nlopt { algorithm: NloptAlgorithm, record: NloptRecord },   // sbplx | bobyqa | lbfgs
}

impl OptimizerRecord {
    /// The point a `from_mle` start reads. There is no such method on
    /// `ResultRecord` or `SamplerRecord`.
    pub fn point(&self) -> &PointEstimate;
}

/// An optimization result: the maximum one optimizer chain reached.
/// Replaces `best_loglik`, `best_chain`, `loglik_type` and `start_values` for
/// IF2 and NLopt.
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
    /// Parameters declared `initial_state` (perturbed at t = 0 only). Read only
    /// to report the dynamics-only sub-assessment beside the overall one
    /// (§4.4). Named as method-scoped-settings §7.5 names the list.
    pub initial_state_params: Vec<String>,
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

/// A sampler's record. The mixing measurements stay in
/// `<sampler>_summary.json` (camdl.fit-summary/v2), loaded as the existing
/// `PosteriorDiagnostics`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplerRecord {
    pub selected: SelectedDraw,
    pub acceptance_rate: f64,
    pub sampler: SamplerKind,
}

/// One visited state, chosen by a stated criterion. Not an estimate of the
/// MLE or the MAP; it is reported as what it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedDraw {
    pub chain: ChainIndex,
    pub criterion: SelectionCriterion,
    /// The value the criterion maximized, in the criterion's own terms.
    pub value: f64,
    /// The log-likelihood at the draw, of the kind `loglik_kind` names.
    pub loglik: f64,
    pub loglik_kind: LoglikType,
    pub theta: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionCriterion {
    /// PGAS: the visited sweep maximizing log p(Y, X | θ), the complete-data
    /// log-likelihood. Not the marginal MLE.
    MaxCompleteDataLoglik,
    /// PMMH: the visited state maximizing a particle-filter estimate of
    /// log p(Y | θ) plus the natural-scale log prior, warm-up included.
    MaxPfLogPosterior,
    /// MH on the ODE likelihood: the same, with the exact likelihood.
    MaxLogPosterior,
}

/// The sampler-specific fields, so each exists only where it has a meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerKind {
    Pgas,
    Pmmh {
        /// The pre-run mean log-likelihood (`pmmh.rs:913`), which
        /// `fit summary` shows (`fit_summary.rs:3184-3185`, `:3234`).
        initial_loglik: f64,
        /// gh#764. `None` when the filter ruled out the base θ or the
        /// proposed θ′, so the spread could not be measured.
        pf_noise: Option<PfNoiseCheck>,
    },
    Mh { initial_loglik: f64 },
}

/// Today's `DtCheckResult` (`dt_check.rs:68-94`) without the fields that are
/// judgements: `verdict`, `threshold_nats`, `threshold_se_aware_nats`, and
/// `notes`, which is built from the verdict and τ (`build_notes`,
/// `dt_check.rs:191`) and is rendered from the `Assessment` instead.
/// `pf_se_inflation`, today a boolean against a hard-coded 2×
/// (`pf_se_inflation_fires`, `dt_check.rs:142-147`), is stored as the ratio it
/// compares, and the 2× becomes `DtCheckPolicy::se_inflation_ratio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DtCheckLadder {
    pub ladder: Vec<LadderEntry>,
    pub leg1_delta_nats: f64,
    pub leg2_delta_nats: f64,
    pub sigma_max_nats: f64,
    /// Max over the halved rungs of `se / se(fit rung)`; `None` when the fit
    /// rung's SE is zero or not finite (where today's check returns false).
    pub se_inflation_ratio: Option<f64>,
}

impl ResultRecord {
    /// `load_input`, `load_optimizer_input` and `load_for_inspection`, split
    /// by use as method-scoped-settings D16 splits `FitState::load`, are in §6.
    pub fn save(&self, dir: &Path) -> Result<(), String>;
}
```

Fields of today's `FitState` that do not appear: `rw_sd` (deleted on branch
`fix/settings-silent-noops`, `e9358984`), `input_hash` (never set: every writer
passes `None`, `mod.rs:1477`, `nlopt_stage.rs:281`, `pmmh.rs:910`,
`pgas.rs:1637`; no reader), `resolved_gate` (policy, §4.3), and `n_good_chains`
(now `chains.completed`, always written). `resolved_loglik_eval` becomes
`CleanEval`'s three recorded fields. `initial_loglik` is kept only in
`SamplerKind::Pmmh` and `::Mh`, the two writers of a finite value; IF2, NLopt
and PGAS write −∞ or NaN, which `fit summary` already discards.

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
    /// Statistical estimation: optimizer agreement or posterior mixing.
    pub convergence: Measurements<'a>,
    /// Numerical execution: the dt-check, when it ran.
    pub dt_check: Option<Measurements<'a>>,
}
```

`PosteriorDiagnostics` no longer has `demote_point_start`: the R̂ values are
kept, and the point start is a qualification (§4.4).

### 4.3 The policy

```rust
// rust/crates/cli/src/fit/assess.rs

/// Every post-run threshold. Defaults are today's values (§5) except the ESS
/// minimums, which are new (§12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DiagnosticPolicy {
    pub if2: If2Policy,
    pub nlopt: NloptPolicy,
    pub posterior: PosteriorPolicy,
    pub dt_check: DtCheckPolicy,
}

// Every nested policy struct carries the same attributes as the outer one:
// #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
// #[serde(deny_unknown_fields, default)]
// so `[assessment.nlopt] a_thresh` is refused, not ignored.
pub struct If2Policy {
    pub a_thresh: f64,            // 1.01   (config_v2.rs:1863)
    pub decibans_thresh: f64,     // 30.0   (config_v2.rs:1864)
    pub se_floor_sigmas: f64,     // 8.0    (gating.rs:256)
}
pub struct NloptPolicy {
    pub rel_range: f64,           // 0.05   (nlopt_stage.rs:735)
    pub abs_range_xtol: f64,      // 2.0    (nlopt_stage.rs:736)
    pub decibans_thresh: f64,     // 30.0   (shares GateConfig today)
}
pub struct PosteriorPolicy {
    pub max_rhat: f64,            // 1.05   (method_result.rs:585; decided, §12)
    pub severe_rhat: f64,         // 1.5    (method_result.rs:590), display band only
    pub min_ess_bulk: f64,        // 100    (new; decided, §12)
    pub min_ess_tail: f64,        // 100    (new; decided, §12)
}
pub struct DtCheckPolicy {
    pub tolerance: DtTolerance,
    pub se_floor_sigmas: f64,     // 4.0    (dt_check.rs:128-130)
    /// Above this SE ratio the report notes filter-noise inflation; it never
    /// changes the verdict, as today.
    pub se_inflation_ratio: f64,  // 2.0    (dt_check.rs:146)
}
/// TOML: `tolerance = "backend"` (the default), `tolerance = "strict"`, or a
/// number of nats, `tolerance = 0.5`. A hand-written `Deserialize` takes a
/// string or a float and refuses any other string, naming the three forms.
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
    /// Hex of the canonical-JSON digest of `values`. Tables show its first 8
    /// characters; the assessment record's file name uses all of it.
    pub digest: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicySource { FitConfig, Cli }

impl AppliedPolicy {
    /// CLI flags over the fit config's `[assessment]` over the defaults, then
    /// `DiagnosticPolicy::validate`.
    pub fn resolve(fit_config: Option<&DiagnosticPolicySpec>, cli: &PolicyArgs)
        -> Result<Self, PolicyError>;
}

impl DiagnosticPolicy {
    /// Every value finite and in its admissible range; the first violation is
    /// refused naming its dotted key, the value and the range.
    pub fn validate(&self) -> Result<(), PolicyError>;
}
```

The admissible ranges, checked by `validate` wherever a policy is built (a fit
config's `[assessment]` at load, a `--policy` file, a flag):

| key                                                | admissible           |
| -------------------------------------------------- | -------------------- |
| `if2.a_thresh`                                     | finite, > 1          |
| `if2.decibans_thresh`, `nlopt.decibans_thresh`     | finite, > 0          |
| `if2.se_floor_sigmas`, `dt_check.se_floor_sigmas`  | finite, ≥ 0          |
| `nlopt.rel_range`                                  | finite, in (0, 1]    |
| `nlopt.abs_range_xtol`                             | finite, ≥ 0          |
| `posterior.max_rhat`                               | finite, > 1          |
| `posterior.severe_rhat`                            | finite, > `max_rhat` |
| `posterior.min_ess_bulk`, `posterior.min_ess_tail` | finite, ≥ 0          |
| `dt_check.tolerance` as `Nats(τ)`                  | finite, > 0          |
| `dt_check.se_inflation_ratio`                      | finite, > 1          |

Â and R̂ are at least 1 for chains that agree, so a threshold of 1 or less fails
every fit and is refused rather than accepted.

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

[assessment.posterior] # a project stricter than the defaults
max_rhat = 1.01
min_ess_bulk = 400
min_ess_tail = 400

[assessment.dt_check]
tolerance = "strict"
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
    /// IF2 only: the same rule over the parameters not declared
    /// `initial_state`. Reported beside `verdict`, never in place of it.
    pub dynamics_only: Option<Box<Assessment>>,
    pub policy: AppliedPolicy,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Question { OptimizersAgree, ChainsMix, DtStable }

/// One leg's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegOutcome {
    Pass,
    /// Passes the threshold, with a named qualification the reader must see.
    Qualified(Qualification),
    /// Past its threshold.
    Fail { threshold: f64 },
    /// The measurement could not be computed; never a pass.
    Unavailable { measurement: String, reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qualification {
    /// "chains share a start: R̂ does not test exploration".
    SharedStart { n_chains: usize, starts: String },
    /// A dt-check plateau leg in [τ, 2τ), today's `Marginal`.
    DtMarginal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Leg {
    pub kind: LegKind,
    pub required: bool,
    pub outcome: LegOutcome,
    pub value: Option<f64>,
    /// Parameters (or chains) past the threshold or unavailable, worst first.
    pub offenders: Vec<(String, Option<f64>)>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegKind {
    TailA, DecibansSpread,                             // IF2
    OptimizerStatus, ParameterRange,                   // NLopt (+ DecibansSpread)
    MaxRhat, MinEssBulk, MinEssTail,                   // samplers
    DtHalving, DtPlateau,                              // dt-check
}

/// The overall verdict, from the required legs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every required leg passes with no qualification.
    Pass,
    /// Every required leg passes; at least one carries a qualification.
    QualifiedPass(Vec<Qualification>),
    /// No required leg fails, but at least one is unavailable.
    Incomplete { unavailable: Vec<String> },
    /// At least one required leg fails. Unavailable legs are still listed.
    Fail { failing: Vec<LegKind>, unavailable: Vec<String> },
    /// Nothing the question needs exists: one chain.
    NotAssessed(NotAssessed),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotAssessed { OneChain }

impl Assessment {
    /// `Pass` and `QualifiedPass` count as converged: `fit table`'s boolean.
    pub fn converged(&self) -> bool;
    /// `fit table`'s `gate_verdict` string (§8).
    pub fn code(&self) -> &'static str;
}
```

**The overall rule is conservative and the same for every question.** Any
required `Fail` makes the verdict `Fail`. Otherwise any required `Unavailable`
makes it `Incomplete`, naming the measurements. Otherwise any `Qualified` makes
it `QualifiedPass`, carrying the qualifications. Only when every required leg is
`Pass` is the verdict `Pass`. So a NaN Â, an uncomputable R̂, an undefined bulk
or tail ESS, or a missing clean evaluation never yields `Pass` or
`QualifiedPass`.

The legs, one rule per question, each the most specific one the code already
applies for that question:

- **IF2** (`check_scout_convergence`'s statistics, `gating.rs:182-260`).
  - `TailA`, required: fails when Â ≥ `a_thresh` on any estimated parameter,
    initial-state parameters included. A NaN Â is `Unavailable` for that
    parameter ("within-chain variance collapsed").
  - `DecibansSpread`, required: fails when the clean-evaluation spread in
    decibans is ≥ `max(decibans_thresh, se_floor_sigmas · σ_max · NATS_TO_DB)`;
    `Unavailable` without clean evaluation.
  - The decibans leg does not stand in for an unavailable Â. It tests whether
    the chains reached the same log-likelihood, and chains spread along a ridge
    agree in log-likelihood while disagreeing in θ, so an IF2 fit whose every Â
    is NaN is `Incomplete` (or `Fail`, if the decibans leg fails), not a pass.
    This is stricter than the end-of-run advice to "rely on Δ_dB"
    (`runner.rs:2525-2528`); §8 states the cost.
  - `dynamics_only` applies the same two legs with `TailA` restricted to the
    parameters not in `initial_state_params`, and is reported beside the overall
    verdict.
  - With one chain: `NotAssessed(OneChain)`.
- **NLopt** (`check_convergence`'s rule, `nlopt_stage.rs:738-815`).
  `OptimizerStatus`, required, fails unless every chain converged.
  `ParameterRange`, required, fails on a parameter whose relative range exceeds
  `rel_range` and whose absolute range exceeds
  `abs_range_xtol · xtol_rel · max(|θ̂₀|, 1)`. `DecibansSpread`, required, fails
  when the converged chains' spread exceeds `decibans_thresh`. With one chain:
  `NotAssessed(OneChain)`, as for IF2, with the status leg still reported.
- **Posterior** (`PosteriorDiagnostics::max_rhat_status`'s statistics).
  - `MaxRhat`, required: fails at or above `max_rhat`. An R̂ that cannot be
    computed because the sampler failed (a frozen chain, a non-finite draw;
    today `MaxRhat::Unassessable`, `gating.rs:508-511`) is `Fail`, naming the
    parameter: the measurement exists and says the sampler did not move.
  - `MinEssBulk` and `MinEssTail`, required: the minimum over estimated
    parameters of the bulk and tail ESS, failing below `min_ess_bulk` or
    `min_ess_tail`. An undefined bulk or tail ESS is `Unavailable`, naming the
    parameter.
  - A point start does not remove R̂. The R̂ values are computed and judged as
    usual, and every leg that passes carries `Qualified(SharedStart)`: "chains
    share a start: R̂ does not test exploration". Such a fit is at best
    `QualifiedPass`, never `Pass`.
  - With one chain: `NotAssessed(OneChain)`.
- **dt-check** (today's `dt_check.rs` rule): `DtHalving` and `DtPlateau`,
  required, against `max(tolerance, se_floor_sigmas · σ_max)`; a plateau leg in
  `[τ, 2τ)` is `Qualified(DtMarginal)`, today's `Marginal`.

**Why an undefined tail ESS is `Unavailable` and not a warning.** A tail ESS is
undefined when the tail-quantile indicator is constant across draws, typically
because a parameter's draws pile on a bound of its box. Then the 5 % and 95 %
quantiles of that parameter, and any interval reported for it, have no
Monte-Carlo error estimate. A `Warn` that still counts as converged would let
`fit table --converged` select a fit whose intervals are unsupported, which is a
pass with the problem stated only in a column nobody filters on. `Incomplete`
names the parameter and keeps the fit out of the converged set; the user who
accepts it says so with `--allow-nonconverged-source` or reads it under
`--as-run`.

**Why initial-state parameters are not exempt.** The exemption today rests on
IF2's perturbation timing: an initial-state parameter is perturbed once per
iteration, at t = 0, rather than at every observation (`EstimatedParam` doc,
`types.rs:67-73`). Timing explains why its Â may be larger; it does not show
that disagreement in it is harmless. An initial-state parameter is estimated
because it is part of the scientific question (an initial infected count, an
initial susceptible fraction), and chains that disagree about it have not found
one maximum of the likelihood in θ. No argument from the IF2 theory (Ionides et
al. 2015, _PNAS_ 112(3):719–724) makes the maximum in the other coordinates
valid conditional on an unsettled initial state, since the dynamics parameters
are generally correlated with it. So the overall verdict judges every estimated
parameter, and the dynamics-only sub-assessment is reported beside it for the
reader who wants to see whether the disagreement is confined to the initial
state.

### 4.5 Where a verdict and its policy are recorded

A run record never depends on a policy (architecture I5). Assessments are
computed on read; the two that are persisted go in the leaf's assessment record.

```rust
// rust/crates/cli/src/fit/assessment_record.rs (new)

pub const ASSESSMENT_SCHEMA: &str = "camdl.fit-assessment/v1";

/// `<leaf>/assessment/<policy digest>.toml`. Written once, when the run is
/// finalized; never rewritten.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentRecord {
    pub schema: String,            // ASSESSMENT_SCHEMA
    pub producer: String,
    /// The run this record judges.
    pub run_id: ContentHash,
    /// The policy every assessment below was computed under.
    pub policy: AppliedPolicy,
    /// The as-run verdict: what the run's own measurements gave under the
    /// run's policy, when it finished.
    pub as_run: AsRun,
    /// Present when the run started from a stored fit.
    pub admission: Option<Admission>,
}

pub struct AsRun {
    pub convergence: Assessment,
    pub dt_check: Option<Assessment>,
}

/// Why the run was allowed to start from its source.
pub struct Admission {
    /// The source's run id, as in the run's `deps`.
    pub source_run_id: ContentHash,
    pub source_artifact: String,                 // fit_state.toml | draws.tsv
    pub assessment: Assessment,
    /// `--allow-nonconverged-source` admitted a source that did not pass.
    pub overridden: bool,
}
```

**Where it lives.** `assessment/<policy digest>.toml` inside the leaf directory.
The store's exact-set check skips it by name: `is_reserved`
(`runid/src/store.rs:951-953`) gains the `assessment` directory beside
`run.json` and the locks. It is outside `run.json` and outside every addressed
artifact, so it enters no `run_id`, no artifact checksum and no dependency
digest. `runid::Provenance` is not changed.

**When it is written.** By the runner's finalize step, before `run.json` turns
`Completed`, with a create-new open that refuses to overwrite. A later reader
never writes it, whatever policy the reader holds. A cache hit runs nothing, so
it writes nothing: `run.json` and the assessment record are byte-identical
before and after a cache hit under a different policy (Stage 2's test).

The three kinds of surface that report a verdict:

1. **At the end of a run**, every method's runner (IF2, the three NLopt
   optimizers, PMMH, MH and PGAS) calls `assess` under the run's policy (CLI
   flags, then the fit config's `[assessment]`, then the defaults) for its
   convergence question and, where the dt-check ran (IF2, NLopt, MH), for that
   too; prints the result; and writes it as the record's `as_run`.
2. **At read time**, `fit table`, `fit summary` and `fit predict` call `assess`
   under the reader's policy (CLI flags, then the `[assessment]` of the fit
   config the handle named, if it named one, then the running camdl's defaults)
   and print the policy beside the verdict. When the record's `as_run` verdict
   differs, the row is marked (§8). `--as-run` shows the record's verdicts
   instead.
3. **The warm-start source check** (`starts = { from_mle = "@h" }` or
   `from_posterior`) assesses the source under the consuming run's policy at
   launch and refuses unless the verdict is `Pass` or `QualifiedPass` or
   `--allow-nonconverged-source` is given. The outcome is the consuming run's
   `admission`. The selection itself (the source `run_id` and the file consumed)
   is already in the run's `deps` (§3.5); `run.json`'s `inputs.starts_source`
   carries only gh#896's handle and `run_id`. A policy change can therefore
   change whether a new run may start, never what an existing `run_id` computed.

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
    #[arg(long)] pub min_ess_bulk: Option<f64>,
    #[arg(long)] pub min_ess_tail: Option<f64>,
    #[arg(long)] pub dt_check_tolerance: Option<f64>,
    #[arg(long, conflicts_with = "dt_check_tolerance")] pub dt_check_strict: bool,
    /// A TOML file holding an `[assessment]` table.
    #[arg(long)] pub policy: Option<PathBuf>,
}
```

Each flag has an observable effect or is refused (invariant I3), and its value
passes `validate`. On `fit run` and on `fit summary` or `fit predict` of one
leaf, a flag for a question the method does not answer (`--max-rhat` on an IF2
fit) is refused naming the method; on `fit table`, which spans methods, a flag
applies to the rows of its question and is refused only when no row answers it.
`--as-run` reads the assessment records and conflicts with every policy flag.

## 5. Every threshold and tolerance, classified (D3)

"In identity today" was verified by reading the method's identity payload
(`config_v2.rs:1365-1400`) and `CliStageOverrides` (`:1635-1680`); a code
constant is covered by the engine version, which the fit level folds.

| value                                                                                                                                                  | where                                                                                                   | default                           | class                                               | in identity today                                               | after                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| IF2 `iterations`, `particles`, `chains`, `cooling`                                                                                                     | `Algorithm::IF2`, `config_v2.rs:995-1030`                                                               | per config                        | execution control                                   | yes                                                             | unchanged                                                                        |
| IF2 `cooling_target_iters`, `--cooling-target-iters`                                                                                                   | `config_v2.rs:1011`, `:1596`                                                                            | 50 (`:1752`)                      | execution control (cooling)                         | yes                                                             | unchanged                                                                        |
| IF2 clean evaluation `n_particles`, `n_replicates`, `combine`                                                                                          | `LoglikEvalConfig`, `config_v2.rs:1736-1750`                                                            | 4000 × 8, logmeanexp              | execution control (what is measured)                | yes                                                             | unchanged; recorded in `CleanEval`                                               |
| dt-check `enabled`, `n_halvings`, `n_particles`, `n_replicates`                                                                                        | `DtCheckConfig`, `config_v2.rs:1886-1925`                                                               | on, 2                             | execution control (what is measured)                | yes                                                             | unchanged                                                                        |
| NLopt `tolerance` (`xtol_rel`), `max_evals`                                                                                                            | `config_v2.rs:1256-1263`                                                                                | 1e-6, 5000 (`:1276-1277`)         | execution control (stopping)                        | yes                                                             | unchanged; `xtol_rel` recorded                                                   |
| PMMH/MH `iterations`, `burn_in`, `thin`, `adapt`, `adapt_start`                                                                                        | `Algorithm::PMMH`/`Mh`; defaults `pmmh.rs:37-38`                                                        | 5000 burn-in, thin 10             | execution control                                   | yes (`iterations` as target length)                             | unchanged                                                                        |
| PMMH adaptation target 0.234 + 0.206/d                                                                                                                 | `sim/src/inference/pmmh.rs:320-322`                                                                     | —                                 | execution control (adaptation)                      | code                                                            | unchanged                                                                        |
| PGAS `sweeps`, `burn_in`, `thin`, NUTS settings                                                                                                        | `Algorithm::PGAS`; defaults `cli/src/fit/pgas.rs:217-218`                                               | 2000, 5                           | execution control                                   | yes                                                             | unchanged                                                                        |
| PGAS MH-within-Gibbs target 0.44; NUTS dual-averaging target 0.80                                                                                      | `sim/src/inference/pgas.rs:3953`, `:4110`, `:4416`                                                      | —                                 | execution control (adaptation)                      | code                                                            | unchanged                                                                        |
| particle-filter substep budget                                                                                                                         | `ITER_BUDGET`, `sim/src/inference/degeneracy.rs:81`                                                     | 1e10                              | execution control (termination)                     | code (fit); `profile --pf-max-substeps` keyed, `profile.rs:160` | unchanged                                                                        |
| IF2 `gate.a_thresh`                                                                                                                                    | `GateConfig`, `config_v2.rs:1855`                                                                       | 1.01 (`:1863`)                    | **assessment policy**                               | **yes**                                                         | `[assessment.if2] a_thresh`, `--a-thresh`                                        |
| IF2 and NLopt `gate.decibans_thresh`, `--decibans-thresh`                                                                                              | `config_v2.rs:1860`, `:1597`, `:1609`                                                                   | 30 (`:1864`)                      | **assessment policy**                               | **yes**                                                         | `[assessment.if2]`, `[assessment.nlopt]`, `--decibans-thresh`                    |
| SE floor `8 · σ_max` on the decibans leg                                                                                                               | `gating.rs:256`, copies `:290`, `method_result.rs:1109`, `fit_summary.rs:1549`, `:3050`                 | 8                                 | assessment policy                                   | code                                                            | `if2.se_floor_sigmas`                                                            |
| Â warning band `A_SOFT`                                                                                                                                | `gating.rs:33`                                                                                          | 1.05                              | assessment policy                                   | code                                                            | dropped: above the default `a_thresh` 1.01 the band is empty (`gating.rs:27-30`) |
| NLopt relative-range and absolute-floor thresholds                                                                                                     | `nlopt_stage.rs:735-736`; literal `table_row.rs:294`                                                    | 0.05, 2.0                         | assessment policy                                   | code                                                            | `nlopt.rel_range`, `nlopt.abs_range_xtol`                                        |
| R̂ certification bar                                                                                                                                    | `RHAT_CONVERGED_THRESHOLD`, `method_result.rs:585`                                                      | 1.05                              | assessment policy                                   | no (read time, unrecorded)                                      | `posterior.max_rhat`, recorded                                                   |
| R̂ severe band                                                                                                                                          | `RHAT_SEVERE_THRESHOLD`, `method_result.rs:590`                                                         | 1.5                               | assessment policy (display)                         | no                                                              | `posterior.severe_rhat`                                                          |
| minimum ESS                                                                                                                                            | —                                                                                                       | none                              | assessment policy                                   | —                                                               | `posterior.min_ess_bulk`, `min_ess_tail`, default 100 each                       |
| dt-check `threshold_nats`, `--dt-check-strict`                                                                                                         | `config_v2.rs:1920`; `dt_check.rs:115-120`; stored `:83-86`                                             | 2.0 / 0.5 nats (strict 0.5 / 0.1) | **assessment policy**                               | **yes** (gh#730)                                                | `[assessment.dt_check] tolerance`, `--dt-check-tolerance`, `--dt-check-strict`   |
| dt-check SE floor `4 · σ_max`                                                                                                                          | `dt_check.rs:128-130`                                                                                   | 4                                 | assessment policy                                   | code                                                            | `dt_check.se_floor_sigmas`                                                       |
| dt-check SE-inflation note `2×`                                                                                                                        | `dt_check.rs:146`                                                                                       | 2                                 | assessment (note only)                              | code                                                            | `dt_check.se_inflation_ratio`; the ratio is stored                               |
| run-time findings: Â > 1.1, Â > 1.5 with spread > 50, ESS at θ̂ < 5 %, R̂ report 1.1, loglik spread 3·3 nats, chain outlier \|z\| > 3.5, starved ESS 1 % | `runner.rs:2565-2569`, `:2587`, `:2923`; `gating.rs:106`; `chain_diagnostics.rs:99`; `filter_ess.rs:60` | as listed                         | assessment (findings written to `diagnostics.json`) | code                                                            | unchanged; out of scope (§13)                                                    |

The three bold rows are the ones that move a value out of run identity. Every
other assessment value is a code constant today and becomes a policy field with
the same default, except the two ESS minimums, which are new (§12).

## 6. Old records are refused as inputs and read for inspection (D4)

This section implements maintainer decision D4, which method-scoped-settings
lists as its §1 decision 6 and records as its D16 (its §7.5): the loader is
split by use, and Stage 3 carries that split onto the new type.

**The tag.** A `fit_state.toml` written after Stage 3 begins
`schema = "camdl.fit-result/v1"` and `producer = "<camdl version>"`.

**Three loaders, by use.**

```rust
impl ResultRecord {
    /// For a new computation that reads a record of any method (the
    /// `from_posterior` source check, the runner's read-back). Refuses any
    /// file whose `schema` is not `RESULT_SCHEMA`, a one-based chain layout
    /// (gh#990), and a parse failure.
    pub fn load_input(dir: &Path) -> Result<ResultRecord, RecordError>;
    /// For a `from_mle` start. As `load_input`, and refuses a sampler record
    /// with `RecordError::NotAnOptimizer`.
    pub fn load_optimizer_input(dir: &Path) -> Result<OptimizerRecord, RecordError>;
    /// For `fit summary`, `fit table` and `fit predict`. A current record loads
    /// as itself; an older one loads as the frozen `FitState` of
    /// method-scoped-settings D16, with one notice naming its producer, its
    /// schema and the keys this camdl does not support.
    pub fn load_for_inspection(dir: &Path) -> Result<Inspected, RecordError>;
}

pub enum Inspected {
    Current(ResultRecord),
    /// Read-only. Never passed to a function that starts a computation.
    Legacy { state: legacy::FitState, notice: LegacyNotice },
}

pub enum RecordError {
    Unsupported {
        path: PathBuf,
        /// The file's `schema`, or `None` for every record before v1.
        schema: Option<String>,
        /// The file's `producer`, else its `camdl_version` (`state.rs:30-32`),
        /// else the `# Generated by <version>` line `FitState::save` writes
        /// (`state.rs:202`), else `None`.
        producer: Option<String>,
    },
    /// A `from_mle` source whose record is a sampler's. The message names
    /// `from_posterior` and the record's selection criterion.
    NotAnOptimizer { path: PathBuf, sampler: &'static str },
    OneBasedLayout(String),     // gh#990, unchanged
    Parse { path: PathBuf, msg: String },
}
```

`draw_from_mle` (`chain_starts.rs:1020-1041`) and `profile`'s `from_mle` start
(`profile.rs:1949-1957`) take `&OptimizerRecord` and read
`OptimizerRecord::point()`; with no `point()` on a sampler record, the
restriction is in the types. The refusal for a sampler source reads:

```
error: starts = { from_mle = "@pgas_fit" }: the source is a pgas fit, whose
  record holds a selected draw (max complete-data log-likelihood), not a
  maximum-likelihood estimate. Start from its posterior with
  `starts = { from_posterior = "@pgas_fit" }`.
```

The input error for an old record names both versions and both ways forward:

```
error: fits/sir-198ada3a/if2-607db55a/seed_1-06cbd6b3/fit_state.toml was
  written by camdl 0.1.0+fc2039b2 with no result schema (before
  camdl.fit-result/v1). This camdl reads camdl.fit-result/v1 and will not
  start a new computation from it. Re-run the fit, or inspect it with
  `camdl fit summary <leaf>`.
```

**Input paths** call `load_input` or `load_optimizer_input` and so refuse. This
is the one list, and method-scoped-settings §7.5 carries the same one: the
`from_mle` start (`chain_starts.rs:1031`) and `profile`'s `from_mle` start
(`profile.rs:1953`), through `load_optimizer_input`; the warm-start source check
for `from_mle` and `from_posterior` (`check_source_converged`, its load at
`gating.rs:441` and the sampler arm below) and the runner reading back the
record it just wrote (`mod.rs:2101`), through `load_input`. Three paths are not
on it because they do not read `fit_state.toml`: `fit run --resume` copies only
`resume_state.bin` and `parameter_traces.tsv` from the base's chain directories
(`copy_resume_carryover`, `mod.rs:2131-2149`), and `profile --fit` and
`--fixed-file` read the fit config and `mle_params.toml`.

**The source check's sampler arm loads through `load_input` too, and propagates
every error.** Today a source whose `fit_state.toml` has no chain-agreement
table falls through to `sampler_source_verdict` (`gating.rs:491-512`), which
reads `run.json` with `.ok()` and the result with
`MethodResult::load_from(..).ok()?` (`gating.rs:497`). A load error there
becomes `None`, which the caller treats as converged, so a `from_posterior`
source with an unreadable or pre-v1 record is accepted without any check
(verified by reading; branch `fix/settings-silent-noops` is fixing it on main).
After: the source check calls `ResultRecord::load_input` once, for every method,
and passes `RecordError` up as the refusal; `sampler_source_verdict` takes the
loaded record and its `PosteriorDiagnostics` and reads no file itself.

**The inspection readers are `fit summary`, `fit table` and `fit predict`, and
nothing else.** Each calls `load_for_inspection`: a legacy record gives the
notice, and a corrupt or one-based one is an error (a row error in `fit table`,
which lists the rest; it already matches on the load result,
`fit_summary.rs:618`, `:2857`). The reads that today swallow a load error, and
what each becomes:

- `fit_summary.rs:1746`, `FitState::load(..).ok()` for the identity block:
  propagates.
- `method_result::point_start`, `.ok()?` (`method_result.rs:936`), reached from
  `fit predict` (`predict.rs:789`, `:865`) and `fit summary`: becomes
  `point_start(dir) -> Result<Option<(usize, String)>, RecordError>`, and every
  caller propagates. A completed leaf always has a record
  (`load_stage_result_or_exit`, `mod.rs:2096-2110`), so a missing file is an
  error too. Its result now feeds the `SharedStart` qualification rather than a
  demotion to not assessed (§4.4).
- `gating.rs:497`: an input path, above.

`NloptStageResult::load` and `If2StageResult::load` (`method_result.rs:972`,
`:1017`) already map a load error to `MethodResultError::Io` and propagate it;
they move to `load_for_inspection` unchanged in that respect.

**`fit predict` is an inspection reader.** Its numbers come from `draws.tsv` and
`mle_params.toml`; it reads the record only to label convergence (and, from
Stage 4, to print the assessment), so a pre-v1 fit can still be predicted from,
with the notice.

**Inspecting a legacy record.** A legacy state is adapted into `Measurements` by
the Stage 1 adapter, which already reads `FitState`, and assessed under the
reader's policy by the same rule as a current one: every estimated parameter's Â
counts. Its old initial-state key is read, per D16, only to build the
dynamics-only sub-assessment. A legacy sampler's `start_values` are shown as the
selected draw, with the criterion its method implies (§3.6). That adapter is the
whole compatibility surface: the frozen `legacy::FitState` is read-only and is
not written by anything. The release that ships Stage 3 re-keys every fit anyway
(the fit level folds the engine version, `cas.rs:508-520`), so a re-run produces
a current record under a new `run_id` beside the old one.

## 7. Call sites

| caller                              | today                                                                                           | after                                                                                                                                                                     |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| warm-start source check             | `check_scout_convergence` or `sampler_source_verdict`, `gating.rs:439-512`                      | `assess` of the source under the consumer's policy; refuses unless `Pass` or `QualifiedPass`, or `--allow-nonconverged-source`; the outcome is the consumer's `admission` |
| `from_mle` start                    | `draw_from_mle` reads `start_values` of any method, `chain_starts.rs:1020-1041`                 | `load_optimizer_input`, then `OptimizerRecord::point()`; a sampler source is refused                                                                                      |
| `fit table`                         | `table_row.rs:289-372`, one rule per `from_*`                                                   | `TableRow::from(&Inspected, &AppliedPolicy)` through `assess`; `--as-run` reads the assessment records                                                                    |
| `fit summary` text                  | `gate_verdict_block`, `fit_summary.rs:1520-1600`; R̂ at `:1819-1825`; "MAP loglik" at `:1765`    | renders `Assessment.legs` and `dynamics_only`; a sampler's selected draw is labelled by its criterion                                                                     |
| `fit summary` JSON, Markdown, LaTeX | `if2_stage_report`, `fit_summary.rs:3028-3070`                                                  | serializes `Assessment`                                                                                                                                                   |
| end of an IF2 run                   | Â table against `config.gate`, `runner.rs:2530-2552`                                            | `assess`, printed, written as the assessment record's `as_run`                                                                                                            |
| end of an NLopt run                 | `check_convergence` + `print_verdict`, `nlopt_stage.rs:218`, `:818`                             | `assess`, printed, written as `as_run`                                                                                                                                    |
| end of a PMMH or MH run             | R̂/ESS block against `RHAT_CONVERGED_THRESHOLD`, `pmmh.rs:792-795` through `runner.rs:3100-3113` | `assess` (chains mix; dt-check for MH), printed, written as `as_run`                                                                                                      |
| end of a PGAS run                   | the same block, `pgas.rs:1371-1374`                                                             | `assess`, printed, written as `as_run`                                                                                                                                    |
| `fit predict`                       | prints `fit_rhat_max` / `fit_ess_min`, no verdict                                               | prints the fit's assessment once in its launch block; TSV columns unchanged                                                                                               |

`check_scout_convergence`, `compute_if2_gate_verdict`, `GateVerdict`,
`If2StageResult.gate_verdict`, `NloptStageResult.max_rel_range`'s verdict use,
`check_convergence`'s verdict fields, `PosteriorDiagnostics::demote_point_start`
and `GateConfig` are deleted. `format_hard_verdict` and
`format_decibans_spread_verdict` (`gating.rs:284-424`) stay as renderers of an
`Assessment`'s legs. `InitSource::MlePoint` (`chain_starts.rs:65`, `:80`) can
now name only an optimizer source.

## 8. What users see change

Under the default policy:

1. **An initial-state parameter's Â counts.** An IF2 fit whose only Â failure is
   on an initial-state parameter fails everywhere. `fit table` and `fit summary`
   already fail it, so their verdict does not move; the warm-start source check,
   which today exempts it (`gating.rs:190-199`), becomes stricter and refuses
   such a source unless `--allow-nonconverged-source` is given. `fit summary`
   shows the dynamics-only sub-assessment beside the verdict.
2. **No unavailable diagnostic passes.** An IF2 fit with a NaN Â reads
   `incomplete` and names the parameter, where today it passes silently. This
   includes the common case of a cold-cooled IF2 refinement whose every Â is NaN
   (`runner.rs:2525-2528`): it is `incomplete` even when the decibans leg
   passes, and as a `from_mle` source it is refused unless
   `--allow-nonconverged-source` is given. A sampler fit with an undefined bulk
   or tail ESS for some parameter reads `incomplete`, naming it. An IF2 fit
   without clean evaluation reads `incomplete`, where
   `fit summary --format
   json` gave `null` and the other sites judged it on Â
   alone.
3. `fit table` judges an NLopt fit by the rule its run printed: the absolute
   floor can turn a failure into a pass, and a decibans spread past 30 dB turns
   a pass into `fail_db`.
4. An NLopt leaf used as a `from_mle` source is refused when its chains
   disagree, where today nothing refuses it (§3.2).
5. A single-chain IF2 or NLopt fit's row reads `not_assessed` with `converged`
   false, where it read `pass`, as a single-chain sampler already does.
6. **A shared start is a qualification, not an absence.** A multi-chain sampler
   fit whose chains began at one point shows its R̂ and ESS, judged as usual, and
   reads at best `qualified_pass` with "chains share a start: R̂ does not test
   exploration", where today its R̂ is discarded and it reads not assessed
   (`method_result.rs:397-415`). `fit predict`'s `fit_rhat_max` column carries
   the value for such fits instead of being empty.
7. `gate_verdict` takes one code per verdict: `pass`, `qualified_pass`,
   `incomplete`, `not_assessed`, and per method the failing codes. IF2:
   `fail_a`, `fail_db`, `fail_both`, as today (`method_result.rs:41-60`). NLopt:
   `fail_status`, `fail_chain_agreement`, and the new `fail_db`
   (`table_row.rs:296-303`). Samplers: `fail_rhat` and `fail_ess`, where they
   read `n/a` (`table_row.rs:345`, `:363`). `converged` is true for `pass` and
   `qualified_pass` only.
8. `fit table` and `fit summary` print the policy they judged under: one line
   naming its digest and every non-default value. The JSON row gains
   `assessment` and `as_run_assessment` and its `TableRowSchema` version
   (`table_row.rs:39-42`) is bumped.
9. **Changed verdicts are marked.** A row whose read-time verdict differs from
   the `as_run` verdict in its assessment record is marked `*`, with a footer
   giving the as-run verdict and its policy digest. A row with no assessment
   record is marked `†`, with the footer "judged under policy <digest>; no
   verdict recorded at run time". That is every leaf written before Stage 2, and
   the mark lands in Stage 1, with the first rule change, so no verdict moves
   unmarked between stages. For a sampler leaf written before Stage 2 the as-run
   verdict can be reconstructed faithfully, because the posterior rule's
   statistics are unchanged and its policy then was fixed (R̂ below 1.05, no ESS
   minimum, a point start demoted to not assessed); such a row also gets `*`
   when the reconstruction differs. For an IF2 or NLopt leaf the rule itself
   changes (items 1–5), so no as-run verdict is claimed and the row carries `†`
   only. `fit table --converged` and `--gate-failed` filter on the read-time
   verdict (`fit_table.rs:378-382`); when they exclude rows whose as-run verdict
   differs, a footer counts them ("3 rows excluded whose as-run verdict was
   converged; `--as-run` to filter on it").
10. A sampler fit (PGAS, PMMH, MH) with any estimated parameter's bulk or tail
    ESS below 100 now fails, with `gate_verdict = "fail_ess"`. A stored fit that
    passed when it ran and fails only on ESS is shown with the `*` mark of item
    9, as-run `pass` beside read-time `fail_ess`, whether its as-run verdict is
    in an assessment record (Stage 2 onward) or reconstructed (earlier sampler
    leaves); it is never silently re-judged.
11. **A sampler draw is not an MLE.** `starts = { from_mle = "@h" }` and
    `profile --starts from_mle` refuse a PGAS, PMMH or MH source, naming
    `from_posterior`. `fit summary` labels the sampler's reported point by its
    criterion ("max complete-data log-likelihood draw", "max PF log-posterior
    draw", "max log-posterior draw") where it said "MAP loglik".

Configuration and flags:

12. `[method.gate]` and `[method.dt_check] threshold_nats` are refused with the
    new spelling (`[assessment.if2]`, `[assessment.nlopt]`,
    `[assessment.dt_check] tolerance`) and the value converted. Under an `nl-*`
    method, `[method.gate] a_thresh` is refused as having no meaning for `nl-*`,
    with no converted spelling: NLopt agreement is judged by relative and
    absolute range (§4.4), and `[assessment.nlopt]` has no `a_thresh` field
    (`deny_unknown_fields`). `--decibans-thresh` and `--dt-check-strict` keep
    their names and stop re-keying.
13. A threshold outside its admissible range (§4.3) is refused when the policy
    is built, naming the key, the value and the range.
14. Changing a threshold no longer re-runs a fit: `fit run` with a new
    `--decibans-thresh` is a cache hit that prints the new verdict, and writes
    nothing.
15. The source check judges a source under the consuming run's policy, not the
    gate the source was run with, and the consuming run's assessment record says
    which policy admitted it.

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
`fit/assess.rs` with `Measurements`, `DiagnosticPolicy` with `validate`
(defaults as §5, the ESS minimums at zero until Stage 2), `AppliedPolicy`,
`Assessment`, the overall rule of §4.4 and `assess`. Today's records are adapted
into `Measurements` by a loader in the same module (IF2 from `FitState`; NLopt
from `chain_results.tsv`, the fit sidecar's search box and the method's
`tolerance`; samplers from `PosteriorDiagnostics`, with `demote_point_start`
removed). Every read-time caller of §7 moves to `assess`; the policy is
`resolved_gate` where present, else the defaults. The `†` mark of §8 item 9
lands here. Branch `fix/settings-silent-noops` is adding the shared
`assess_if2`; Stage 1 takes its statistics as the IF2 arm of `assess` and
removes its initial-state exemption (§10). Tests, each red on `a9a89178`:

- the settings-fixes agreement test (§3.1), extended to
  `fit summary --format json`, asserting that the three sites agree on `fail_a`
  for an IF2 leaf whose only Â failure is on an initial-state parameter, and
  that the source check refuses it (item 1);
- an IF2 leaf with one NaN Â and a passing decibans leg reads `incomplete`,
  naming the parameter (item 2);
- an NLopt leaf whose chains disagree in relative but not absolute range is
  `converged` in `fit table` (item 3);
- an NLopt leaf with chains at opposite ends of the box is refused as a
  `from_mle` source (item 4);
- a single-chain IF2 row is `not_assessed` (item 5);
- a point-start PMMH leaf with R̂ below 1.05 reads `qualified_pass` and shows its
  R̂ (item 6);
- `validate` refuses `max_rhat = 1.0`, `decibans_thresh = 0`,
  `min_ess_tail =
  -1` and a NaN, each naming the key (item 13).

Each is mutation-checked: restoring the old rule at its site turns its test red.
Fixed-seed check: no writer changes, so all outputs are byte-identical.

**Stage 2: thresholds leave run identity (D3), and the assessment record.**
`FitConfig` gains `assessment: DiagnosticPolicySpec`; `GateConfig` leaves
`Algorithm::IF2` and the NLopt config; `DtCheckConfig.threshold_nats` leaves
`[method]`; `CliStageOverrides` loses `decibans_thresh` and `dt_check_strict`;
`PolicyArgs` is flattened into `fit run`, `fit table` and `fit summary`;
`config_identity_hash` subtracts `assessment`; `FitState.resolved_gate` and
`DtCheckResult`'s `verdict`, `threshold_nats`, `threshold_se_aware_nats` and
`notes` are removed, so a source's `fit_state.toml`, which the consumer's deps
digest, no longer carries any policy (§3.5). `AssessmentRecord` lands with
`is_reserved` skipping `assessment/`; every runner (IF2, NLopt, PMMH, MH, PGAS)
calls `assess` at finalize, for its convergence question and its dt-check, and
writes `as_run`; the source check writes `admission`. `fit table --as-run`, the
`*` marker and the filter footer of §8 item 9 land. The ESS minimums of §12 take
their decided defaults of 100 here, together with sampler recording, with the
§12 test. This re-keys every IF2, NLopt and dt-checked MH fit once (their method
payload loses the gate and the tolerance); the release re-keys every fit
regardless. Tests:

```rust
/// D3: a threshold changes the verdict a reader sees, not the run, and never a
/// finalized assessment record.
#[test]
fn a_policy_threshold_moves_the_table_verdict_not_the_run_or_its_records() {
    let root = fit_run(&sir_pmmh_toml(), &[]);                  // `camdl fit run`
    let leaf = only_leaf(&root);
    let id = run_id(&leaf);
    let state = std::fs::read(leaf.join("fit_state.toml")).unwrap();
    let run_json = std::fs::read(leaf.join("run.json")).unwrap();
    let records = read_dir_bytes(&leaf.join("assessment"));     // one file

    let before = fit_table_json(&root, &[]);
    let rhat = before.row.max_rhat.unwrap();
    // A bar just past the measured R̂, on the side that flips the verdict.
    let bar = if before.row.converged { rhat - 1e-6 } else { rhat + 1e-6 };
    let after = fit_table_json(&root, &["--max-rhat", &bar.to_string()]);
    assert_ne!(before.row.converged, after.row.converged);
    assert_eq!(after.row.assessment.policy.values.posterior.max_rhat, bar);
    assert_eq!(before.row.max_rhat, after.row.max_rhat, "measurements unchanged");
    assert_eq!(read_dir_bytes(&leaf.join("assessment")), records, "a read writes nothing");

    // The same threshold given to `fit run` is a cache hit on the same leaf
    // that rewrites nothing.
    let root2 = fit_run(&sir_pmmh_toml(), &["--max-rhat", &bar.to_string()]);
    assert_eq!(only_leaf(&root2), leaf);
    assert_eq!(run_id(&leaf), id);
    assert_eq!(std::fs::read(leaf.join("fit_state.toml")).unwrap(), state);
    assert_eq!(std::fs::read(leaf.join("run.json")).unwrap(), run_json);
    assert_eq!(read_dir_bytes(&leaf.join("assessment")), records);
}

/// A warm start's identity follows its source run, not the policy that
/// admitted it.
#[test]
fn the_source_run_keys_a_warm_start_and_the_admitting_policy_does_not() {
    let a = only_leaf(&fit_run(&sir_if2_toml_seed(1), &[]));
    let b = only_leaf(&fit_run(&sir_if2_toml_seed(2), &[]));
    let from_a = only_leaf(&fit_run(&from_mle(&a), &[]));
    let from_a_loose = only_leaf(&fit_run(&from_mle(&a), &["--a-thresh", "2.0"]));
    let from_b = only_leaf(&fit_run(&from_mle(&b), &[]));
    assert_eq!(run_id(&from_a), run_id(&from_a_loose), "policy is not identity");
    assert_ne!(run_id(&from_a), run_id(&from_b), "the source run is identity");
    // The second invocation was a cache hit: the record of the first stands.
    assert_eq!(admission(&from_a).policy.values.if2.a_thresh, 1.01);
}
```

The first test is repeated for IF2 with `--a-thresh`, and for
`--decibans-thresh`, which today fails it (red on `a9a89178`, _inferred_ from
the identity payload: the second `fit run` lands in a new leaf). Also:
`[method.gate]` and `[method.dt_check] threshold_nats` refused with the
converted spelling, and `[method.gate] a_thresh` under `nl-sbplx` refused with
no conversion; `--max-rhat` refused on an IF2 `fit run` and on `fit summary` of
an IF2 leaf; `--as-run` refused beside a policy flag; a second write of an
assessment record refused; a leaf whose `as_run` verdict differs from the
read-time one is marked. Fixed-seed check.

Stage 2 also extends method-scoped-settings' `FitConfig::load_archived` (its
§7.2), the loader for a stored fit's `fit.toml.original`. `fit table`
(`load_archived_fit_toml`, `fit_table.rs:297-309`) and the `@handle` lookup
(`handle.rs:174-199`) reload that copy, so a stored fit whose copy carries
`[method.gate]` or `[method.dt_check] threshold_nats` would otherwise become
unlistable and unaddressable once Stage 2 refuses those keys. `load_archived`
sets them aside with the same one notice, listing them; anything that would run
from the stored config still refuses them. Test: `fit table` over a store
holding a fit archived with `[method.gate] decibans_thresh = 60` lists it with
the notice once, and `@handle` resolves it; red when `load_archived` refuses the
key.

**Stage 3: per-method result records (D4).** `ResultRecord` replaces `FitState`;
the four writers construct their `MethodRecord` variant, the samplers a
`SelectedDraw` with its criterion; the input paths of §6 call `load_input` or
`load_optimizer_input` and `fit table`, `fit summary` and `fit predict` call
`load_for_inspection`, carrying method-scoped-settings D16's split onto the new
type; `MethodResult` is rebuilt on `ResultRecord` without verdict fields;
`fit summary` labels a selected draw by its criterion. Lands after the
method-scoped-settings commit that splits `FitState::load` into `load_input` and
`load_for_inspection` and adds a `schema` tag to `FitState` (its §11 names it).
Tests: each variant round-trips; a committed pre-v1 `fit_state.toml` fixture is
refused as a `from_mle` source and as a `from_posterior` source, each with both
versions in the message (the second red on `a9a89178`, where `gating.rs:497`
accepts it), and is shown by `fit summary` and `fit table` with exit 0; a PGAS
leaf named as a `from_mle` source, and as `profile --starts from_mle`, is
refused naming `from_posterior` (red on `a9a89178`, where it is accepted as
`mle_point`); an `if2` record whose body is a sampler's fails to parse.
Fixed-seed check.

**Stage 4: `fit predict` reports the assessment.** `PolicyArgs` on
`fit predict`; its launch block prints the fit's assessment and policy. Every
TSV it writes is byte-identical at a fixed seed, except `fit_rhat_max` and
`fit_ess_min` for a point-start fit, which carry values where they were empty
(§8 item 6, from Stage 1).

## 10. Dependencies on work in flight

**Landing order.** The method-scoped-settings and parameter-roles work lands as
step series (on `feat/support-transforms`) → plan series → parameter-roles Stage
2 → parameter-roles Stage 3 → release. Stages 0, 1 and 2 here are independent of
that sequence: Stage 1 needs only `fix/settings-silent-noops`'s `assess_if2`,
and Stage 2 needs Stage 1. Stage 3 lands after the D16 commit of the step
series. Stage 4 needs Stage 3.

- **`fix/settings-silent-noops`** (`e9358984` plus an uncommitted test in its
  worktree): deletes `FitState.rw_sd`, refuses inert flags, and is adding the
  shared `assess_if2`, which folds in `if2_stage_report` (§3.1) and judges an
  NLopt source by NLopt's rule (§3.2). Its agreement test asserts that an
  initial-state Â failure passes; under §4.4 it fails. Whichever lands second
  changes that assertion: if the branch lands first, Stage 1 flips it; if Stage
  1 lands first, the branch takes the rule from here. Stage 1 wraps `assess_if2`
  rather than writing a second one. The branch's refusal of inert flags is the
  model for `PolicyArgs`' refusals (§4.6), and it is fixing the `gating.rs:497`
  defect of §6 on main.
- **`docs/method-scoped-settings`** (`2df329b0`): its D16 (maintainer decision 6
  of its §1, its §7.5) splits `FitState::load` by use into `load_input`, which
  refuses an unsupported old `fit_state.toml` naming the producer and schema
  versions, and `load_for_inspection`, which returns the state with one notice.
  Stage 3 keeps that split on the new type and adds `load_optimizer_input`;
  there is no key-specific refusal. Stage 3 takes its flag name `initial_state`
  and list name `initial_state_params` (its §7.5). The two documents carry one
  list of input paths, §6 here. Its D16 test asserts that `fit summary` applies
  the old initial-state exemption to a legacy IF2 record; under §4.4 the legacy
  record's initial-state list builds only the dynamics-only sub-assessment, so
  that assertion moves to the sub-assessment. Its §5.2 moves `[estimate].start`
  out of the fit-level digest (`Problem::identity_view()`) and hashes, at the
  method level, the written `[estimate].start` values (`starts_base`, from
  `Problem::start_values`) beside the starts rule already in the method payload;
  that settles the problem-identity half of D2 (its §1 decisions 4 and 5), which
  this proposal does not touch. Its `ExecutionPlan` (its §6.1) carries no
  policy: `MethodPlan` holds execution controls only. Its `If2Plan` (as of
  `2df329b0`) holds no `gate` and keeps only the dt-check measurement controls
  (`enabled`, `n_halvings`, particles, replicates); the tolerance is
  `DtCheckPolicy` here. The same holds for the dt-check measurement controls
  that `MhOdePlan` and `NloptPlan` carry (MH and the NLopt optimizers also run
  the dt-check). Either order of Stage 2 and the step-series commit that adds
  the `[method]` tables works; both edit `Algorithm::IF2`, so the second
  rebases.
- **`2026-10-26-parameter-roles.md`** (accepted) and its Stage 2: no type here
  depends on `ResolvedProblem`; `NloptAgreement.search_box` is filled from the
  search box the partition resolves once it exists, and from the fit sidecar
  until then.
- **`feat/support-transforms`**: carries method-scoped-settings' step series,
  including the D16 commit Stage 3 depends on. It also changes which estimates
  are reported near a bound (`877c517a`), a run-time finding this proposal
  leaves alone (§13). Its step series edits `state.rs`, `config_v2.rs`
  (`Algorithm::IF2`, `EstimateSpecV2`), `nlopt_stage.rs` and the IF2 arm of
  `mod.rs`, which Stages 0–3 here also edit, so whichever lands second rebases
  onto the other.
- **`fix/gh896-handles`**: a handle resolves to one leaf, and a sourced run
  records `inputs.starts_source` with the handle and the source `run_id`. This
  proposal adds nothing to that object; the admission lives in the consuming
  run's assessment record (§4.5). Rows of `fit table` are per leaf on that
  branch, which is what a per-leaf verdict needs.

## 11. Decisions made in this proposal

**Policy lives in a top-level `[assessment]` table, not in `[method]`.**
`[method]` is what the engine runs (method-scoped-settings), and the problem
digest must not see a threshold either. A third part of `FitConfig` is outside
both by construction, so no hash needs a subtraction except the config lookup.

**A persisted verdict lives in a write-once assessment record, not in
`run.json`.** A run record describes what happened (architecture I5); two
executions of one `run_id` under different policies would otherwise write
different `run.json` files. The record sits at `assessment/<policy digest>.toml`
inside the leaf, skipped by name in the exact-set check, so it enters no
identity and no dependency digest, and is never rewritten by a read.

**The overall verdict is conservative.** Any required unavailable diagnostic
makes it `Incomplete`; only a fit whose every required leg passes outright is
`Pass`. The cost is that cold-cooled IF2 refinements with collapsed Â read
`incomplete` (§8 item 2); the alternative, letting the decibans leg stand in,
would pass chains spread along a likelihood ridge.

**Initial-state parameters count.** §4.4 gives the argument; the dynamics-only
sub-assessment keeps the distinction visible without letting it decide the
verdict.

**A shared start qualifies a pass; it does not erase R̂.** R̂ remains computable
and informative about within-run mixing; what it cannot show is global
exploration, which the qualification says.

**An optimum and a draw are different types.** `from_mle` reads only an
`OptimizerRecord`, so a sampler's selected draw cannot be consumed as an MLE by
any code path that compiles.

**Sampler-specific fields go in `SamplerKind`, without further abstraction.**
`initial_loglik` exists for PMMH and MH and `pf_noise` for PMMH, so each is a
field of its variant; `pf_noise` stays an `Option` inside `Pmmh` because "not
measurable" is a real outcome there. Nothing more is introduced to remove those
two.

**The reader's policy is the default view, with a marker where it disagrees with
the as-run verdict, and another where none was recorded.** The verdict a user
acts on should be the standard they hold now; the markers and `--as-run` keep
the historical verdict visible, so nothing flips silently.

**The source check uses the consumer's policy.** The question is whether this
run should start from that fit, which is the consumer's judgement; the
consumer's assessment record says which policy admitted it.

**Each rule is the most specific one the code already has.** IF2 takes
`check_scout_convergence`'s statistics, NLopt the stage's own, samplers theirs.
No statistic is new.

**Legacy records follow method-scoped-settings D16.** That maintainer decision
already splits loading by use, and an old record is inspected through the Stage
1 adapter that reads `FitState`, so no second compatibility layer is written.

**`initial_loglik` is kept, for PMMH and MH.** `fit summary` reads it
(`fit_summary.rs:3184-3185`, `:3234`), and only those two write a finite value.

## 12. Default thresholds: decided by the maintainer (gh#84)

**Posterior samplers default to max R̂ 1.05, bulk ESS ≥ 100 and tail ESS ≥ 100
per estimated parameter** (maintainer decision, 2026-10-29). The R̂ bar is
today's; the two ESS minimums are new, since no ESS minimum is applied today
(§3.3). Each is a field of `PosteriorPolicy` and is overridable per project in
`[assessment.posterior]` and per command with `--max-rhat`, `--min-ess-bulk` and
`--min-ess-tail`. For reference, Vehtari et al. (2021) recommend R̂ < 1.01 for
the rank-normalized statistic and an ESS above 400 for four chains; a project
that wants those writes them in `[assessment.posterior]` (§4.3).

Because the ESS minimums are new, some stored sampler fits that were reported
converged now fail. They are shown with the `*` mark, with the as-run verdict
beside the new one (§8 items 9 and 10), and are never silently re-judged: a
sampler fit run from Stage 2 onward has its as-run verdict in its assessment
record (§4.5), and an earlier one has it reconstructed under the fixed
pre-proposal posterior policy. The ESS minimums therefore take effect in Stage
2, together with sampler recording and the `*` mark; Stage 1 runs them at zero.
Stage 2's test: a stored PMMH fixture with R̂ below 1.05 and one parameter's tail
ESS below 100 reads `fail_ess` with `*` in `fit table`, and `pass` under
`--as-run`.

## 13. Out of scope, each with its reason

- **The run-time findings in `diagnostics.json`** (§5, last row): their
  thresholds are code constants, so they cannot differ within one `run_id`, and
  they describe a run rather than certify it. Moving them to read time is a
  separate change if a finding ever needs a user-settable threshold.
- **`profile` and `survey` verdicts**: they report likelihood surfaces, not
  convergence; neither calls any verdict function listed in §3.
- **A better agreement measure when IF2's within-chain variance collapses**:
  §4.4 reports such an Â as unavailable rather than inventing a substitute
  statistic; choosing one is a measurement change, which needs its own note
  under the rule that a new inference feature lands with its measurement.
- **Splitting `runner.rs`**: after these stages, as decided.
