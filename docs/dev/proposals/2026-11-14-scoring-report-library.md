# Proposal: the scoring report as a library

**Status:** Proposal **Date:** 2026-11-14 **Motivation:** `camdl compare`
decides whether fitted models may be compared and by how much they differ, and
today that logic lives in a binary, exits at the first failed check, has no
convergence check, and produces its three output formats from three separate
computations. A published comparison needs one typed, serializable result that
records every check and every label, computed by code that can be called without
a file system.

This is proposal 3 in the set defined by
`2026-11-13-published-comparison-principles.md` (the principles). It reads the
score artifact defined in `2026-11-14-score-artifact.md` (proposal A) and
creates the crate both proposals put their types in.

Code references are to `main` at `e8e7c24a0` (2026-11-14). Every statement under
[What exists today](#what-exists-today) was read in the code at that commit.
Nothing was built or run for this proposal; anything that rests on more than
reading the cited lines is marked _not verified_ with the reason.

## Terms

- **Member.** One fit taking part in a comparison: one row of the table.
- **Preflight check.** A test that the members may be compared, run before any
  difference is formed (same observations, same scored steps, and so on).
- **Waiver.** A command-line flag that lets a comparison render although a named
  check failed. A waiver that was given and needed is an **override**, and is
  recorded.
- **Refusal.** A failed check that was not waived. A refused comparison forms no
  difference.
- **Evidence label.** The word attached to a score difference: "within noise",
  "within filter noise", or a tier of Jeffreys' scale.
- **Verdict.** camdl's statement of whether a fit's sampler converged.
- **R̂.** The between-chain agreement statistic; near 1 when chains agree.
- **elpd, CRPS, PIT, replicate, filter-noise standard error, leaf.** As defined
  in proposal A.

## What exists today

### The logic is split across two crates, and half of it cannot be imported

The `cli` crate builds one binary and has no library target
(`rust/crates/cli/Cargo.toml:24-26`).

| What                                                              | Where                                                         |
| ----------------------------------------------------------------- | ------------------------------------------------------------- |
| Trace types, the log-score, CRPS and PIT kernels                  | `rust/crates/sim/src/inference/prequential.rs`                |
| Projection of a trace onto one stream                             | `prequential.rs:497`                                          |
| `log_sum_exp`, `logmeanexp_with_se`                               | `rust/crates/sim/src/inference/types.rs:736`, `:775`          |
| The `Coverage` enum (what an observed value was accumulated over) | `rust/crates/sim/src/inference/multi_stream_obs.rs:1596`      |
| Combining traces across draws and replicates                      | `rust/crates/cli/src/compare.rs:2653`, `:2682`                |
| Paired difference and its standard error                          | `compare.rs:3048-3069`                                        |
| Newey–West standard error, the noise checks, the evidence label   | `rust/crates/cli/src/evidence.rs`                             |
| Every preflight check                                             | `preflight`, `compare.rs:1057-1255`                           |
| The three renderers                                               | `compare.rs:3286` (table), `:3514` (markdown), `:3602` (JSON) |
| Verdict and the rule that decides it                              | `rust/crates/cli/src/fit/report/facts.rs:198-243`             |

`sim` depends on `rayon`, `bincode`, `rand`, `rand_chacha`, and optionally
`nlopt`, `arrow` and `parquet` (`rust/crates/sim/Cargo.toml`). The trace module
uses three things from the rest of `sim`: `log_sum_exp` and
`logmeanexp_with_se`, `Coverage`, and in `build_trace` only, the filter's
recorded output and the seeded random stream (`prequential.rs:743-764`).

### The first failed check exits

`preflight` returns `Result<Preflight, String>` and returns `Err` at the first
unwaived failure (`compare.rs:1057`, for example `:1081`, `:1091`, `:1173`).
Later checks are never evaluated, so their results do not exist. Its order is:

1. mixed conditioning (in-sample against held-out): no waiver;
2. differing training boundaries: no waiver;
3. in-sample stream ablation: `--allow-in-sample-ablation`;
4. mixed provenance (plug-in against posterior mixture):
   `--allow-mixed-provenance`;
5. differing numbers of scored steps: `--allow-mismatched-horizon`;
6. pairing: the same times, the same leaves at each time, the same windows: no
   waiver;
7. differing bound data: `--allow-data-mismatch`;
8. data changed since the fit: `--allow-changed-data`;
9. differing particle or replicate counts: `--allow-mismatched-filter-settings`.

Before `preflight`, three checks run per member while its trace is derived, and
each exits the command: the fit's backend must be chain-binomial
(`check_derivable_backend`, `compare.rs:1951`); a declared holdout must have a
recorded training window (`detect_sealed_holdout`, `compare.rs:1997`); and no
training row may lie in the window scored as held-out
(`check_training_ends_by_train_end`, `compare.rs:2039`).

The outcome type records overrides as typed values and everything else as
optional strings (`Preflight`, `compare.rs:936-953`; `Override`,
`compare.rs:909-930`). A refusal is a formatted `String`.

### Each renderer computes the scores again

`paired_delta` is called in the table renderer (`compare.rs:3321`), the markdown
renderer (`:3552`), the JSON builder (`:3633`) and the row ordering (`:3181`).
The JSON document is assembled field by field with `serde_json::json!`
(`json_document`, `compare.rs:3623-3770`); there is no type for it, no schema
version, and no engine version in it.

### Labels are strings

`evidence_label` returns `&'static str` (`evidence.rs:201-209`): `within_noise`,
`within_filter_noise`, or a tier name from `jeffreys_label`
(`evidence.rs:69-77`), one of which, `very strong`, contains a space. The
principles cite this function at `evidence.rs:259`; on `main` it is at line 201.

### There is no convergence check, and no stored verdict

`grep -n -i 'converge\|rhat\|verdict' rust/crates/cli/src/compare.rs` matches
nothing. gh#716 (open) records a chain that never moved changing a comparison by
13 units of elpd.

A verdict is one of six kinds: converged, not converged, chains frozen,
unassessable, not assessed, not applicable (`Verdict`, `facts.rs:198-214`). It
is not stored. `report::verdict` rebuilds it on each read from the method's
stored diagnostics (`facts.rs:218-243`), and the rule is a constant in the
running binary, `RHAT_CONVERGED_THRESHOLD = 1.05`
(`rust/crates/cli/src/fit/method_result.rs:787`), applied through `RhatBand::of`
(`method_result.rs:838`). `Converged` carries the largest R̂ and its parameter
and not the threshold. `Verdict` derives `Serialize` only, so it cannot be read
back. gh#725 (open) proposes a threshold of 1.01 with a minimum effective sample
size.

The diagnostics the verdict is built from are stored and typed:
`PosteriorDiagnostics` (`method_result.rs:529`) holds, per estimated parameter,
R̂ in three forms and bulk and tail effective sample sizes (`ParamConvergence`,
`method_result.rs:341-365`), and the frozen chains (`method_result.rs:567`). The
version of their computation is recorded in the method's summary file as
`camdl.fit-summary/v2` (`FIT_SUMMARY_SCHEMA`,
`rust/crates/cli/src/run_meta.rs:44`). Methods without chains have a separate
result with six values (`GateVerdict`, `method_result.rs:152-166`).

## The gap

1. Nothing outside the `camdl` binary can compute or read a comparison as a
   typed value.
2. A refused comparison says only its first reason.
3. A member whose sampler did not converge is compared without remark.
4. A verdict of "converged" does not say under which rule.
5. The three formats can disagree, because each computes; and a label is a
   string a consumer must match by spelling.
6. A comparison's output does not say which camdl computed it or which artifact
   versions it read.

## The design

### The crate

A new workspace crate, `score`, at `rust/crates/score`.

```
cli   → io, ir, runid, sim, score
sim   → ir, numerics, score
score → numerics
```

`score` depends on `serde`, `serde_json` and `numerics`, and on nothing else.
`numerics` depends on `libm` only (`rust/crates/numerics/Cargo.toml`). `score`
has no access to files, the store, processes, threads, the clock, environment
variables or a random number generator. Three things enforce that:

- its dependency list, asserted by a test that reads `score/Cargo.toml`;
- a `clippy.toml` in the crate that disallows `std::fs`, `std::process`,
  `std::thread`, `std::env`, `std::net` and `std::time::SystemTime` (the
  workspace has no `clippy.toml` today, so this adds the first);
- `cargo check -p score --target wasm32-unknown-unknown` in CI.

What lives in it:

| Module        | Contents                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- |
| `trace`       | `PrequentialTrace` and its parts, `Coverage`, the kernels, `project_onto_stream`             |
| `combine`     | `combine_mixture`, `combine_scored_traces`, `ReplicateTotals`                                |
| `artifact`    | `ScoreArtifact` and its parts, the profile, `ScoredLeadTime`, `LeadTimeStatus` (proposal A)  |
| `evidence`    | the paired difference, its standard errors, the noise checks, `Evidence`                     |
| `convergence` | `Verdict`, `ConvergenceRule`, `ConvergenceDiagnostics`, `ConvergenceRecord`, `judge`         |
| `preflight`   | `CheckId`, `CheckResult`, `Finding`, `Waiver`, `Override`, `Refusal`, one function per check |
| `report`      | `Member`, `Request`, `Report`, `compare`                                                     |
| `render`      | `table`, `markdown`, `json`                                                                  |

`log_sum_exp` and `logmeanexp_with_se` move to `numerics`, which `sim` already
depends on. `build_trace` stays in `sim`: it needs the filter's recorded output.
It constructs `score::trace` types. Everything that touches a file, the store or
a subprocess stays in `cli`: resolving a handle, reading a leaf, hashing a data
file, running a filter pass, reading a fit's diagnostics.

### The entry point

```rust
pub fn compare(members: &[Member], request: &Request) -> Report
```

It is total: it returns a `Report` for every input, including a refused one. It
does not return `Result`, print, or exit.

```rust
pub struct Member {
    pub name: String,
    pub source: MemberSource,            // ScoreArtifact, or BareTrace
    pub artifact: Option<ScoreArtifact>,
    pub trace: PrequentialTrace,         // the artifact's, or the bare one
    pub assimilated: Recorded<Vec<String>>, // the leaves the fit bound
    pub data_now: Vec<FileFinding>,      // each scored file against its record
    pub diagnostics: Recorded<ConvergenceDiagnostics>, // read from the fit, if at hand
    pub unusable: Vec<Refusal>,          // refusals found while loading it
}
pub struct Request {
    pub engine: EngineStamp,             // supplied by the caller
    pub waivers: Vec<Waiver>,
    pub baseline: Option<String>,
    pub streams: Option<Vec<String>>,    // --streams, already resolved to leaves
    pub metrics: Vec<Metric>,
}
```

`cli` builds each `Member` by doing the reading: it looks up or derives the
score artifact (proposal A), re-hashes the data files to fill `data_now`, and
reads the fit's stored diagnostics when the fit is in a store it can reach. The
three per-member checks that exit during derivation today become entries in
`unusable`, so the report lists them with every other result.

### The Report

```rust
pub struct Report {
    pub schema: SchemaVersion,           // { name: "camdl.report", major: 1, minor: 0 }
    pub engine: EngineStamp,             // camdl version and git hash
    pub inputs: Vec<InputRecord>,        // one per member
    pub convergence_rule: ConvergenceRule,
    pub waivers_given: Vec<Waiver>,
    pub tables: Vec<Table>,              // one; or one per leaf under --streams
    pub outcome: Outcome,                // Compared, or Refused
}
pub struct InputRecord {
    pub name: String,
    pub source: MemberSource,
    pub score_run_id: Recorded<String>,
    pub fit_run_id: Recorded<String>,
    pub score_schema: Recorded<SchemaVersion>,
    pub score_engine: Recorded<EngineStamp>,
    pub trace_schema: u32,
    pub diagnostics_schema: Recorded<String>,
    pub profile: Recorded<ProfileRef>,
}
pub struct Table {
    pub scope: Scope,                    // Joint, or Stream(leaf)
    pub checks: Vec<CheckResult>,        // every check, in a fixed order
    pub overrides: Vec<Override>,
    pub refusals: Vec<Refusal>,
    pub baseline: Recorded<String>,
    pub scored_steps: Recorded<usize>,
    pub conditioning: Recorded<Conditioning>,
    pub lead: Recorded<ScoredLeadTime>,
    pub rows: Vec<Row>,
    pub notes: Vec<Note>,                // typed caveats: small sample, optimism, the scale
}
pub struct Row {
    pub name: String,
    pub scored_steps: usize,
    pub elpd: f64,
    pub mc_se_elpd: Recorded<f64>,
    pub sample: SampleStatus,
    pub difference: Difference,
    pub accuracy: Accuracy,              // per stream, or Unavailable(reason)
    pub assimilated: Recorded<Vec<String>>,
    pub provenance: Provenance,
    pub warnings: Vec<PrequentialWarning>,
}
pub enum Difference {
    Baseline,
    NotFormed { because: Vec<CheckId> },
    Formed {
        delta_elpd: f64, se: f64, mc_se: Recorded<f64>,
        delta_db: f64, likelihood_ratio: f64, lag1_autocorrelation: Recorded<f64>,
        evidence: Evidence, direction: Direction,
        noise_check: NoiseCheck,         // Checked, or OneSided { has, lacks }, or NoReplicates
    },
}
```

`Recorded<T>` is proposal A's explicit "value or not recorded". `Report` and
everything in it derive `Serialize` and `Deserialize`. A `Report` contains no
trace and no timestamp, so the same members and request give the same bytes.

The schema `camdl.report` is versioned on its own. A reader refuses a major
version it does not know and reports an enum value it does not know as unknown
(principles, R4 and R5). The report records the engine that computed it and, per
member, the schema versions of the score artifact, the trace and the diagnostics
it read, and the engine that computed the score.

The comparison-level `overrides` list `compare` writes today is
`Table.overrides`. Under `--streams` there is one table per leaf and an override
belongs to the table whose check it waived.

### Renderers are functions of the Report

```rust
pub fn table(report: &Report) -> String
pub fn markdown(report: &Report) -> String
pub fn json(report: &Report) -> String      // serde of the Report, pretty-printed
```

They take a `Report` and nothing else, and they compute nothing: every number
and label they print is a field. Fixed sentences (a waiver's consequence, a
refusal's remedy, the scale note) come from one lookup keyed by the typed value,
in `render`. Colour and terminal width stay with the caller.

The JSON output becomes the serialized `Report`. That changes its shape once.
The change is the subject of its own commit, the document is listed in
`docs/camdl-run-spec.md`, and camdl-scope's reader changes with it
(camdl-scope#13 already tracks the reader's update for PR #1048).

### Every check is evaluated and recorded

```rust
pub enum CheckId {
    MemberUsable, Convergence, ConvergenceRule, Conditioning, TrainingBoundary,
    InSampleAblation, Provenance, ScoredSteps, Pairing, SharedData, ChangedData,
    FilterSettings, FilterSeed, Profile,
}
pub struct CheckResult {
    pub check: CheckId,
    pub status: CheckStatus,
    pub waiver: Option<Waiver>,          // the flag that can waive it, if any
}
pub enum CheckStatus {
    Passed,
    Failed { finding: Finding, members: Vec<String>, waived: bool },
    Noted { finding: Finding, members: Vec<String> },  // reported, never refuses
    NotChecked { members: Vec<String>, reason: NotCheckedReason },
    NotEvaluated { because: CheckId },   // a check it depends on failed
}
```

`compare` runs every check in the order of `CheckId` and records each result.
None is skipped silently. Two checks read what an earlier one established:
`Pairing` compares only shared times when `ScoredSteps` failed and was waived,
as today (`compare.rs:1168-1172`), and `SharedData` under `--streams` assumes
`Pairing` passed. When the earlier check failed unwaived, the later one is
`NotEvaluated { because }`. That is a recorded result, not an absence.

A `Finding` is an enum with one variant per way a check can fail, carrying the
values that differed
(`TrainingBoundaryDiffers { per_member: Vec<(String, f64)> }`,
`ParticlesDiffer { .. }`, `DrawsDiffer { .. }`, `NotConverged { verdict }`, and
so on). The sentence is rendered from it. A `Refusal` is a failed, unwaived
check: `{ check, finding, members }`. `Outcome::Refused` holds when any table
has a refusal; the command then exits with status 2, as it does today
(`compare.rs:798-800`).

`NotCheckedReason` is an enum (`NoDerivationBlock`, `BareTrace`,
`NoDataHashRecorded`, `HashPrefixOnly`, `NoDiagnostics`, ...). A member that
lacks the fact a check needs is `NotChecked` for that check and named. It never
passes by default.

Three checks are new or changed relative to `main`:

- `FilterSettings` also fails on a differing number of posterior draws used.
- `FilterSeed` is `Noted` when seeds differ. It never refuses.
- `Profile` is `Noted` when a member was scored under no named profile, or the
  members' profile names differ while their values agree.

gh#1054 asks for the first two on `main` ahead of this proposal.

### The convergence refusal

`Convergence` judges each member and fails when any member's verdict is not
converged, chains frozen, or unassessable. Its waiver is a new flag,
`--allow-nonconverged`, following `fit run`'s existing
`--allow-nonconverged-source` (`rust/crates/cli/src/args/mod.rs:706`).

| Verdict                                    | `Convergence` status for that member         |
| ------------------------------------------ | -------------------------------------------- |
| Converged                                  | passes                                       |
| Not converged, chains frozen, unassessable | fails; waivable                              |
| Not assessed                               | passes, `Noted` with camdl's recorded reason |
| Not applicable (a method without chains)   | passes, `Noted` with its `GateVerdict`       |
| No verdict and no diagnostics (bare trace) | `NotChecked { NoDiagnostics }`               |

The last row is this proposal's call: a score with no fit behind it is treated
as every other missing fact is, reported as not checked and never assumed
converged. The verdict judged is that of the method leaf that supplied the
parameters the score was computed at (`ScoredFit.run_id`, proposal A).

When the check fails and is waived, the override is recorded in the table's
`overrides`, and each affected row carries it:

```rust
pub enum SampleStatus {
    Converged,
    NotConvergedOverridden { verdict: Verdict },
    NotAssessed { reason: String },
    NotApplicable { gate: Recorded<GateVerdict> },
    NotChecked,
}
```

Every renderer prints, on a `NotConvergedOverridden` row, that its scores were
computed from a sample that did not converge. The label is on the row, so it
travels with the row into any excerpt, and it is computed, never authored
(principles, R2).

### The verdict records its rule, and is decided on read

```rust
pub struct ConvergenceRule {
    pub id: String,                      // "rhat-max-below"
    pub version: u32,                    // 1
    pub statistic: RhatStatistic,        // MaxOfRankNormalizedAndFolded
    pub threshold: f64,                  // 1.05
    pub min_ess: Recorded<f64>,          // NotRecorded: rule 1 has no floor
}
pub struct ConvergenceDiagnostics {
    pub schema: String,                  // "camdl.fit-summary/v2"
    pub per_param: BTreeMap<String, ParamConvergence>,
    pub frozen_chains: Recorded<Vec<usize>>,
}
pub struct ConvergenceRecord {
    pub verdict: Verdict,
    pub rule: ConvergenceRule,
    pub diagnostics: Recorded<ConvergenceDiagnostics>,
    pub gate: Recorded<GateVerdict>,
}
pub fn judge(d: &ConvergenceDiagnostics, rule: &ConvergenceRule) -> Verdict
```

A verdict is a function of stored diagnostics and a rule, and is computed when
it is read. It is not stored in the fit leaf. `judge` is today's
`report::verdict` and `max_rhat_status` (`method_result.rs:603`) with the
threshold passed in; `RHAT_CONVERGED_THRESHOLD` becomes the `threshold` of
`score::convergence::CURRENT_RULE`. `Verdict`, `ParamConvergence`, `Stat`,
`RhatBand` and `GateVerdict` move to `score::convergence` and `Verdict` gains
`Deserialize`. `fit summary` calls `judge` with `CURRENT_RULE` and prints the
rule beside the verdict.

Where each thing is written:

- **The score artifact** (`ScoredFit.convergence`, proposal A) stores a
  `ConvergenceRecord`: the verdict under the rule current when the score was
  computed, that rule, and the diagnostics with the version of their
  computation.
- **The bundle record**, written by `camdl publish` (proposal 1), stores the
  same record at publish time. So a bundle always carries the diagnostics needed
  to judge it again.
- **The report** stores the one rule it judged every member by
  (`Report.convergence_rule`) and, per row, the verdict under that rule.

How `compare` judges a member, in order:

1. If diagnostics are available (from the fit, in `Member.diagnostics`, or in
   the member's `ConvergenceRecord`), it calls `judge` with `CURRENT_RULE`.
   Every such member is judged by one rule.
2. If the record has a verdict and no diagnostics, it uses the recorded verdict.
   If the recorded rule differs from `CURRENT_RULE`, the `ConvergenceRule` check
   is `Noted` for that member, naming both rules.
3. If the diagnostics' `schema` is not one `judge` understands, the member is
   `NotChecked` and the schema is named.
4. Otherwise it is `NotChecked { NoDiagnostics }`.

When gh#725 changes the rule, it adds `CURRENT_RULE` version 2 with a threshold
and a `min_ess`. The diagnostics already hold bulk and tail effective sample
sizes, so a bundle published under rule 1 can be judged under rule 2 without its
fit. A bundle may then read "converged (rule 1)" beside a later report's "not
converged (rule 2)"; each is shown with its rule.

The viewer shows recorded verdicts with their rules. It never calls `judge` and
holds no threshold (principles, R3).

### Computed labels are enums

```rust
pub enum Evidence { WithinNoise, WithinFilterNoise, Tier(JeffreysTier) }
pub enum JeffreysTier { Indeterminate, Substantial, Strong, VeryStrong, Decisive, Overwhelming }
pub enum Direction { For, Against, Undirected }
```

`evidence::label(delta, se, mc_se) -> Evidence` replaces `evidence_label`. The
thresholds and their order of precedence do not change. The display cell
(`evidence_cell`, `evidence.rs:230`) becomes a rendering of `Evidence` and
`Direction`. Serialized names are snake case, so `very strong` becomes
`very_strong` in the JSON; that is part of the one JSON change.

The other computed labels become enums in the same way: `Waiver`, `CheckId`,
`Finding`, `NotCheckedReason`, `SampleStatus`, `NoiseCheck`, `Scope`,
`MemberSource` and `Note`. No field of a `Report` is a sentence.

### What `compare` guarantees

Each guarantee is a property of `score::compare` and is tested as stated.

1. **No difference without a clean preflight.** If a table has a refusal, every
   row's `difference` is `Baseline` or `NotFormed`, and no renderer prints a
   difference, a ratio or an evidence label for it.
2. **Every check has a result.** `checks` has exactly one entry per `CheckId`,
   in declaration order, for every input.
3. **Unknown is never a pass.** For each check and each fact it reads, a member
   lacking that fact makes the status `NotChecked` naming the member.
4. **Overrides are exact.** `overrides` holds a waiver if and only if its check
   failed and the waiver was given. A waiver given and not needed is absent.
5. **An unconverged member is refused or labelled.** A member whose verdict is
   not converged, chains frozen or unassessable either refuses the table, or
   appears with `NotConvergedOverridden` and an override, never otherwise.
6. **One rule.** Every member judged from diagnostics is judged by
   `Report.convergence_rule`; any member judged otherwise has a `Noted` or
   `NotChecked` result saying so.
7. **Renderers add nothing.** For any `Report` `r`,
   `render(deserialize(serialize(r)))` equals `render(r)`, for each renderer.
8. **Deterministic.** Equal inputs give byte-identical serialized reports.
9. **The label follows from the row.** For every `Formed` row,
   `evidence::label(delta_elpd, se, mc_se)` equals its `evidence`.
10. **Antisymmetric.** Swapping the baseline between two members negates
    `delta_elpd`, keeps `se`, and keeps the evidence tier.
11. **Order-free.** Permuting the members changes no number and no check result.
12. **No figure combines streams.** No field holds a sum, mean or total across
    leaves; under `--streams` there is one table per leaf and nothing else.
13. **Self-identifying.** `engine` and every `InputRecord` are filled, with
    `NotRecorded` where the input does not say.

It does not guarantee that a comparison is adequate. A fair comparison of poor
models is fair (principles, R6).

## Invariants, as tests

Beyond the thirteen above, each of which is one test over generated or fixture
inputs:

14. **Dependency list.** `score/Cargo.toml` names exactly `serde`, `serde_json`
    and `numerics` as dependencies.
15. **No ambient access.** `cargo clippy -p score` passes with the disallowed
    list, and `cargo check -p score --target wasm32-unknown-unknown` passes.
16. **Moves change no output.** See [Steps](#steps), phase 1.
17. **All refusals at once.** A fixture with three independent failures yields a
    report with three refusals, where `main` prints one.
18. **Dependent checks.** A fixture that fails `Pairing` records `SharedData` as
    `NotEvaluated { because: Pairing }` under `--streams`.
19. **The draw count.** Two members with 64 and 16 draws used fail
    `FilterSettings`; with differing seeds only, `FilterSeed` is `Noted` and the
    outcome is `Compared`.
20. **Rule in the verdict.** A fixture with a largest R̂ of 1.03 is `Converged`
    under a rule with threshold 1.05 and `NotConverged` under one with 1.01, and
    each serialized record names its rule.
21. **Re-judging without the fit.** A `ConvergenceRecord` with diagnostics,
    judged under a second rule, gives the verdict the fit's own diagnostics give
    under that rule.
22. **Unknown values.** A report with `schema.major = 2` is refused by name; one
    with an unknown `CheckId` is read and the value reported as unknown.
23. **Mutation checks.** For tests 1, 5 and 19, removing the property in the
    source makes the test fail; the commit message records the run.

## What this proposal does not do

- It does not design a browser build. What one would need is below.
- It does not choose the Rust-to-TypeScript generator (proposal 7).
- It does not change any threshold: the R̂ bar stays 1.05 (gh#725), and the
  evidence tiers and noise checks are as on `main`.
- It does not change how a filter pass is run or how a score is derived.
- It does not add a combined table over several streams, or a measure of whether
  a quantity is driven by its prior (proposal 5).
- It does not define the bundle record or `camdl publish`; it states what they
  must store for convergence.
- It does not make an ODE fit comparable (gh#312).

## What a later browser build would need

Stated so that nothing here forecloses it; none of it is designed.

- The `score` crate as specified, which already compiles for
  `wasm32-unknown-unknown` (test 15) and takes its inputs as values.
- A decision on floating-point agreement. `exp`, `ln` and `powf` may differ in
  the last bits between a native build and a browser; `numerics` already uses
  `libm`, and routing `score`'s transcendental calls through it would make the
  two agree. Whether agreement is exact or within a stated tolerance has to be
  decided and tested with reports computed both ways. _Not verified:_ that any
  difference occurs on the functions used.
- Integer fields that exceed 2⁵³ (a `u64` seed) serialized as text, since a
  browser's JSON reader loses them.
- The generated TypeScript types of proposal 7.
- A scope statement. The filter is not in `score`, so a browser could recompute
  a report from score artifacts and could not score a fit.

## Steps

Each step is one commit with a `Proposal:` footer.

**Phase 1: moves, with no logic change.** Each commit moves code and edits `use`
lines and nothing else. The check is the same for every one: the outputs below
are byte-identical before and after.

0. Pin the outputs. Add tests that hold, as committed text, the table, markdown
   and JSON renderings of the existing in-memory fixtures (`single_stream_rows`,
   `two_stream_rows`, the ablation table; `compare.rs:4825`, `:4915`, `:5783`)
   and the serialized trace of the two-stream fixture. Where `main` already pins
   one (the JSON document of a two-stream comparison), keep it. Separately,
   record `camdl compare` in three formats and
   `camdl pfilter
   --save-prequential` on two committed model configurations
   with the binary built before step 1; the same commands after each step must
   produce identical bytes. The commit message of each step states the
   comparison and its result.
1. Create `score` with its dependency test and `clippy.toml`. Move `log_sum_exp`
   and `logmeanexp_with_se` to `numerics`.
2. Move `Coverage` and the trace module (types, kernels, `project_onto_stream`,
   their tests) from `sim` to `score::trace`. `build_trace` stays.
3. Move `evidence.rs` to `score::evidence`.
4. Move `combine_mixture`, `combine_scored_traces`, `ReplicateTotals`,
   `paired_delta` and `paired_diffs` to `score`.
5. Move `Verdict`, `ParamConvergence`, `Stat`, `RhatBand`, `GateVerdict` and the
   body of `report::verdict` to `score::convergence`.

**Phase 2: the report.** Table and markdown output stay byte-identical through
steps 6 to 8 (the step 0 pins); the JSON changes in step 9 only.

6. `CheckId`, `CheckResult`, `Finding`, `Refusal`, with each existing check
   rewritten as a function returning a `CheckResult`; `preflight` evaluates all
   of them. Tests 2, 3, 4, 17, 18.
7. `Member`, `Request`, `Report`, `score::compare`; `cmd_compare` builds members
   and calls it. `Evidence` and the other label enums. Tests 1, 8, 9, 10, 11,
   12, 13.
8. `render::table` and `render::markdown` as functions of the `Report`; the old
   renderers deleted. Test 7.
9. `render::json` as the serialized `Report`; `docs/camdl-run-spec.md` documents
   it. This commit's subject is the JSON change.
10. `ConvergenceRule`, `ConvergenceDiagnostics`, `ConvergenceRecord`, `judge`
    with the rule passed in; `fit summary` prints the rule. Tests 20, 21.
11. The `Convergence` and `ConvergenceRule` checks, `--allow-nonconverged`,
    `SampleStatus` and its label in every format. Tests 5, 6. Addresses the
    `compare` half of gh#716.
12. `FilterSettings` with draws, `FilterSeed`, `Profile`. Test 19. Closes
    gh#1054 if it has not landed already.
13. Schema versions and unknown-value handling for `camdl.report`. Test 22.
14. `docs/commands.md`, `docs/methods/model-comparison.md`, and the dependency
    table in `.claude/rules/rust-conventions.md`.

Proposal A's steps follow step 5 and may interleave with phase 2: its artifact
types land in `score::artifact`, and step 7 here reads them.

## Verification

Commands run on `main` at `e8e7c24a0` for the claims the design rests on.

- `sed -n 24,26p rust/crates/cli/Cargo.toml` → a `[[bin]]` named `camdl`; the
  manifest has no `[lib]`.
- `grep -n 'paired_delta(' rust/crates/cli/src/compare.rs` → the definition at
  3048 and calls at 3181, 3182, 3321, 3552, 3633, 3728.
- `grep -n 'pub fn evidence_label' rust/crates/cli/src/evidence.rs` → line 201,
  returning `&'static str`.
- `grep -n -i 'converge\|rhat\|verdict' rust/crates/cli/src/compare.rs` → no
  match.
- `grep -n 'RHAT_CONVERGED_THRESHOLD: f64' rust/crates/cli/src/fit/method_result.rs`
  → line 787, value 1.05.
- `grep -rn 'report.json\|"verdict"' rust/crates/cli/src --include='*.rs'` → no
  site that writes a verdict to a leaf; `fit_summary.rs:377` holds it in a
  document built on read.
- `sed -n '/\[dependencies\]/,$p' rust/crates/numerics/Cargo.toml` → `libm`
  only.
- `ls rust/clippy.toml rust/crates/clippy.toml` → neither exists.
- `gh issue view 725 --json state`, `716`, `312` → each `OPEN`.
