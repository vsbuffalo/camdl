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

Code references are to `main` at `e8e7c24a0` (2026-11-14). Since then phase 1 of
this proposal has moved the trace and evidence modules into `rust/crates/score`,
and `compare`'s chain selection has gained `--chains` beside `--exclude-chains`
(`982eca439`, `770013f76`); "excluded chains" below means the resolved set under
either spelling. Fits on the ODE backend can now be compared (gh#312): a trace
records its `basis`, and `preflight` refuses to difference a filtered trace
against an ODE one, with no override. File paths and line numbers are as they
were at `e8e7c24a0`. Every statement under
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
recorded output and the seeded random stream (`prequential.rs:743-764`). The
combiner in `cli` also draws from that stream (`compare.rs:2711`, `:2735`).

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
two comment lines and no code. gh#716 (open) records a chain that never moved
changing a comparison by 13 units of elpd.

A verdict is one of six kinds: converged, not converged, chains frozen,
unassessable, not assessed, not applicable (`Verdict`, `facts.rs:198-214`). It
is not stored. `report::verdict` rebuilds it on each read (`facts.rs:218-243`),
and the rule is a constant in the running binary,
`RHAT_CONVERGED_THRESHOLD =
1.05`
(`rust/crates/cli/src/fit/method_result.rs:787`), applied through `RhatBand::of`
(`method_result.rs:833`) together with a second constant,
`RHAT_SEVERE_THRESHOLD = 1.5` (`:792`). The first constant is also read directly
in `rust/crates/cli/src/fit/table_row.rs:393` and `:411` and in
`rust/crates/cli/src/fit/gating.rs:842-844`. `Converged` carries the largest R̂
and its parameter and not the threshold. `Verdict` derives `Serialize` only, so
it cannot be read back. gh#725 (open) proposes a threshold of 1.01 with a
minimum effective sample size.

The diagnostics the verdict is built from are typed but are not a stored file.
`PosteriorDiagnostics` (`method_result.rs:529`) holds, per estimated parameter,
R̂ in four forms, bulk and tail effective sample sizes and the per-chain
effective sample sizes (`ParamConvergence`, `method_result.rs:341`), and the
frozen chains (`method_result.rs:567`). `cli` builds it from the keys of the
method's summary file (`ConvergenceMaps`, `method_result.rs:1655`), whose
version is `camdl.fit-summary/v2` (`FIT_SUMMARY_SCHEMA`,
`rust/crates/cli/src/run_meta.rs:44`), and then, for a fit whose chains all
started at one point, replaces each R̂ with that refusal (`demote_point_start`,
`method_result.rs:584`), reading the fit's `fit_state.toml` to know
(`point_start`, `method_result.rs:1204`). `ParamConvergence` embeds error types
defined in `sim` (`ConvergenceError`, `RhatRefusal`,
`rust/crates/sim/src/inference/convergence.rs`), and `Verdict::ChainsFrozen`
holds `cli`'s `ChainIndex`. Methods without chains have a separate result with
six values (`GateVerdict`, `method_result.rs:152-166`).

`fit summary --exclude-chains` recomputes the diagnostics over the retained
chains (`recompute_subset_diagnostics`,
`rust/crates/cli/src/chain_selection.rs:298`). `compare --exclude-chains` scores
without the excluded chains (`compare.rs:644-657`) and judges nothing.

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
cli   → io, ir, runid, sim, score, numerics
sim   → ir, numerics, score
score → numerics
```

`score` depends on `serde`, `serde_json` and `numerics`, and on nothing else.
`numerics` depends on `libm` only (`rust/crates/numerics/Cargo.toml`). `score`
has no access to files, the store, processes, threads, the clock, environment
variables or a random number generator. Two tests enforce that:

- its dependency list, asserted by a test that reads `score/Cargo.toml`;
- a test that reads every source file under `score/src` and fails if one names
  `std::fs`, `std::process`, `std::thread`, `std::env`, `std::net`, `std::time`
  or `std::io`, or imports from the root of `std` with braces (`use std::{`),
  which would let a module be named without its `std::` prefix.

A lint cannot do the second job: clippy's `disallowed_methods` and
`disallowed_types` take paths to single functions and types, and reject a module
path with "expected a function, found a module" (tried on the pinned toolchain,
clippy 0.1.94). A hand-kept list of every item in five modules would miss what
it did not name.

`cli` gains `numerics` as a direct dependency: it calls the three helpers that
move there (`logmeanexp_with_se` in `compare.rs` and `loglik_eval.rs`,
`sample_sd` in `loglik_eval.rs`, `log_sum_exp` in `survey.rs`).

A third check, `cargo check -p score --target wasm32-unknown-unknown`, is a make
target run in CI. It shows that the crate and its dependencies build for a
browser. It is not a check on ambient access: the standard library compiles for
that target with the file and thread calls present and failing at run time (_not
verified_; the target is not installed on the machine this was written on).

What lives in it:

| Module        | Contents                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- |
| `num`         | `json_f64` (non-finite floats in JSON), moved from `sim::inference`                          |
| `trace`       | `PrequentialTrace` and its parts, `Coverage`, the kernels, `project_onto_stream`             |
| `combine`     | `ReplicateTotals`; later `finish`, which replaces the two combiners (see below)              |
| `artifact`    | `ScoreArtifact` and its parts, the profile, `ScoredLeadTime`, `LeadTimeStatus` (proposal A)  |
| `evidence`    | the paired difference, its standard errors, the noise checks, `Evidence`                     |
| `convergence` | the diagnostics types, `Verdict`, `ConvergenceRule`, `ConvergenceRecord`, `judge`            |
| `preflight`   | `CheckId`, `CheckResult`, `Finding`, `Waiver`, `Override`, `Refusal`, one function per check |
| `report`      | `Member`, `Request`, `Report`, `compare`                                                     |
| `render`      | `table`, `markdown`, `json`                                                                  |

`log_sum_exp`, `sample_sd` and `logmeanexp_with_se` (which calls `sample_sd`,
`rust/crates/sim/src/inference/types.rs:744`, `:775`) move to `numerics`, which
`sim` already depends on.

What does not move:

- `build_trace` stays in `sim`: it needs the filter's recorded output and draws
  the PIT uniforms. It constructs `score::trace` types. The trace module's tests
  that use `sim`'s random stream or the filter's output stay in `sim` with it.
- `combine_scored_traces` draws a PIT uniform per pooled value
  (`rust/crates/cli/src/compare.rs:2711`, `:2735`), and `combine_mixture` calls
  it (`:2659`). They stay in `cli` through phases 1 and 2. Proposal A's step A1
  replaces both with one function, `score::combine::finish`, which pools the
  passes' draws and draws no random number.
- Everything that touches a file, the store or a subprocess stays in `cli`:
  resolving a handle, reading a leaf, hashing a data file, running a filter
  pass, reading and recomputing a fit's diagnostics, writing `--pointwise`
  files.

### The entry point

```rust
pub fn compare(members: &[Member], request: &Request) -> Report
```

It is total over its inputs: it returns a `Report` for every list of members and
every valid `Request`, including a refused comparison. It does not return
`Result`, print, or exit.

```rust
pub enum Member {
    Usable(UsableMember),
    /// Loading found a reason this member cannot be scored. No trace exists.
    Unusable { name: String, path: String, refusals: Vec<Refusal> },
}
pub struct UsableMember {
    pub name: String,
    pub path: String,                       // the handle or file as given
    pub source: MemberSource,               // ScoreArtifact, or BareTrace
    pub artifact: Option<ScoreArtifact>,
    pub trace: PrequentialTrace,            // the artifact's, or the bare one
    pub assimilated: Recorded<Vec<String>>, // the leaves the fit bound
    pub data: DataIdentity,                 // the recorded digests the fit was bound to
    pub data_now: Vec<FileFinding>,         // each scored file against its record
    pub convergence: Recorded<ConvergenceRecord>,
    pub chains_excluded: Vec<usize>,
}
pub struct Request {                        // built by Request::new, which validates
    pub engine: EngineStamp,                // supplied by the caller
    pub waivers: Vec<Waiver>,
    pub baseline: Option<String>,
    pub streams: Option<Vec<String>>,       // --streams, already resolved to leaves
    pub metrics: Vec<Metric>,
}
```

`cli` builds each `Member` by doing the reading: it looks up or derives the
score, re-hashes the data files to fill `data_now`, and reads the fit's
diagnostics.

**What stays an input error.** Four conditions are mistakes in the command, not
findings about the members. `cli` reports them and exits with status 1 before
`compare` is called, as today: fewer than two members (`compare.rs:634`), an
unknown format, a `--streams` name that resolves in no member, and a baseline
name that is not a member (`compare.rs:766-771`), which `Request::new` refuses.

**What becomes an unusable member.** Every refusal raised while a member's trace
is derived becomes a `Refusal` in `Member::Unusable`, so the report lists it
with every other result and the remaining members are still checked against each
other. On `main` these are: no `[data]` block or no bound stream (`:2333`,
`:2341`); a declared holdout with no recorded training window (`:2351`); a
training row inside the window scored as held-out (`:2387`); a posterior draw
the filter cannot score (`:2512`); an empty held-out window (`:2582`); and a
scored time at or before the training boundary (`:2589`).

**`--pointwise`** stays in `cli`. It writes its files from the usable members'
traces and the report's baseline, and only when the report's outcome is
`Compared`.

**`DataIdentity`** is today's type (`compare.rs:286`), moved: the digests a fit
recorded, by source, or the reason they are unknown. When proposal A's score
artifact exists, it is filled from the artifact's `rows_read` digest per leaf
(every row the filter read), and the `SharedData` check compares those: two
members were scored on the same observations when, for each scored leaf, their
`rows_read` digests are equal. Under `--streams` only the scored leaf's digest
is compared, as today.

**`data_now`** is today's per-file finding (each scored file's bytes against the
fit record). When proposal A's artifact exists it is replaced by the artifact's
per-leaf prefix result (`Matches`, `Differs`, `NotRecorded`): the `ChangedData`
check fails on `Differs`, lists `NotRecorded` under `not_checked`, and is waived
by `--allow-changed-data` given on this `compare` command. Proposal A stores no
waiver in an artifact, so an artifact computed under a waiver is refused here
unless the waiver is given again.

### The Report

```rust
pub struct Report {
    pub schema: SchemaVersion,           // { name: "camdl.report", major: 1, minor: 0 }
    pub engine: EngineStamp,             // camdl version and git hash
    pub inputs: Vec<InputRecord>,        // one per member, in the order given
    pub convergence_rule: ConvergenceRule,
    pub waivers_given: Vec<Waiver>,
    pub metrics: Vec<Metric>,
    pub tables: Vec<Table>,              // one; or one per leaf under --streams
    pub outcome: Outcome,                // Compared, or Refused
}
pub struct InputRecord {
    pub name: String,
    pub path: String,
    pub source: MemberSource,
    pub usable: bool,
    pub score_run_id: Recorded<String>,
    pub fit_run_id: Recorded<String>,
    pub score_schema: Recorded<SchemaVersion>,
    pub score_engine: Recorded<EngineStamp>,
    pub trace_schema: Recorded<u32>,
    pub diagnostics_schema: Recorded<String>,
    pub profile: Recorded<ProfileRef>,
    pub derivation: Recorded<Derivation>, // particles, replicates, seed, step, where the
                                          // parameters came from, excluded chains, overrides
}
pub struct Table {
    pub scope: Scope,                    // Joint, or Stream(leaf)
    pub checks: Vec<CheckResult>,        // every check, in a fixed order
    pub overrides: Vec<Override>,
    pub refusals: Vec<Refusal>,
    pub baseline: Recorded<String>,
    pub scored_steps: Recorded<usize>,
    pub first_scored_index: Recorded<usize>, // the trace's `t0`; joint tables only
    pub conditioning: Recorded<Conditioning>,
    pub lead: Recorded<ScoredLeadTime>,
    pub rows: Vec<Row>,                  // usable members, in the order given
    pub display_order: Vec<usize>,       // indices into `rows`
    pub notes: Vec<Note>,                // typed caveats: small sample, optimism, the scale
}
pub struct Row {
    pub name: String,
    pub scored_steps: usize,
    pub elpd: Num,
    pub mc_se_elpd: Recorded<Num>,
    pub sample: SampleStatus,
    pub difference: Difference,
    pub accuracy: Accuracy,              // per stream, or Unavailable(reason)
    pub assimilated: Recorded<Vec<String>>,
    pub provenance: Provenance,
    pub conditioning: Conditioning,
    pub warnings: Vec<PrequentialWarning>,
}
pub enum Difference {
    Baseline,
    NotFormed { because: Vec<CheckId> },
    /// The steps pair, but a score on one side is not finite.
    NotFinite { delta_elpd: Num },
    Formed {
        delta_elpd: f64, se: f64, mc_se: Recorded<f64>,
        delta_db: f64, likelihood_ratio: Num, lag1_autocorrelation: Recorded<Num>,
        evidence: Evidence, direction: Direction,
        noise_check: NoiseCheck,         // Checked, or OneSided { has, lacks }, or NoReplicates
    },
}
```

`Recorded<T>` is an explicit "value or not recorded": `Value(T)` or
`NotRecorded`, serialized as a named variant and never as an absent key or
`null`. It is defined in `score` in step 7, and proposal A's artifact uses it.
`Report` and everything in it derive `Serialize` and `Deserialize`. A `Report`
contains no trace and no timestamp.

**Numbers that may not be finite.** A step's log score can be negative infinity
(no particle could produce the observed value) or not a number
(`rust/crates/sim/src/inference/prequential.rs:460-462`), so a member's elpd can
be too. `serde_json` writes such a value as `null`, which does not read back as
a number. `Num` is an `f64` serialized through `score::num::json_f64`, which
writes a finite value as a number and the others as the text `"inf"`, `"-inf"`
and `"NaN"` and reads all four back exactly
(`rust/crates/sim/src/inference/mod.rs:36`). Fields typed `f64` in a `Formed`
difference are finite by construction: a pair with a non-finite difference is
`NotFinite` and carries no standard error and no evidence label.

This changes what is printed. On `main` a row with a step scored at negative
infinity prints its difference as `-inf`, its likelihood ratio as `0` and its
standard error as `0.00`, with "—" in the evidence column only and no line
naming the non-finite score; when that row is the baseline, the other row prints
`+inf` and a standard error of `0.00` (pinned as it is by
`a_row_with_a_non_finite_score_renders_as_pinned`, step 0). The standard error
is not zero; it is undefined. A `NotFinite` row renders its own elpd as the
value it is (`-inf`), renders "—" for the difference, the ratio, the standard
error and the evidence, and adds a note naming the row and the number of its
steps that scored non-finite.

**Row order.** `rows` is in the order the members were given, which is the JSON
order today. `display_order` is the order the table and markdown print: by
difference ascending, the baseline placed as today's `render_order` places it
(`compare.rs:3174`), or the given order when no difference is formed.

**The baseline** is the requested member, else the member with the largest elpd;
among members tied for the largest, the first by name. On `main` the tie goes to
the last such row in the given order (`compare.rs:760-764`), which depends on
the order of the arguments.

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
and label they print is a field. Which columns appear is decided by
`Report.metrics`. Fixed sentences (a waiver's consequence, a refusal's remedy,
the scale note) come from one lookup keyed by the typed value, in `render`.
Colour and terminal width stay with the caller.

The JSON output becomes the serialized `Report`. That changes its shape once.
The change is the subject of its own commit, the document is listed in
`docs/camdl-run-spec.md`, and camdl-scope's reader changes with it
(camdl-scope#13 already tracks the reader's update for PR #1048).

### Every check is evaluated and recorded

```rust
pub enum CheckId {    MemberUsable, Convergence, ConvergenceRule, Basis, Conditioning,
    TrainingBoundary, InSampleAblation, Provenance, StreamScored, ScoredSteps, Pairing, SharedData,
    ChangedData, FilterSettings, FilterSeed, Profile,
}
pub struct CheckResult {
    pub check: CheckId,
    pub outcome: CheckOutcome,           // over the members that could be compared
    pub not_checked: Vec<Unchecked>,     // members the check could not cover, and why
    pub qualified: Vec<Qualified>,       // members covered on weaker evidence, and how
    pub waiver: Option<Waiver>,          // the flag that can waive it, if any
}
pub enum CheckOutcome {
    Passed,
    Failed { finding: Finding, waived: bool },
    Noted { finding: Finding },          // reported, never refuses
    NothingToCompare,                    // fewer than two members carry the fact
    NotEvaluated { because: CheckId },   // a check it depends on failed unwaived
}
pub struct Unchecked { pub member: String, pub reason: NotCheckedReason }
pub struct Qualified { pub member: String, pub how: Qualification }
```

A check's result has three parts because one check can do three things at once.
On `main` the bound-data check can find a mismatch among the members that
recorded hashes while other members recorded none (`compare.rs:1225-1227`), and
the changed-data and filter-settings checks have the same shape (`:1237-1239`,
`:1251-1253`); every format prints both statements (`warning_lines`,
`compare.rs:3409-3416`). `outcome` is the result over the members that carry the
fact, `not_checked` names each member that does not, and `qualified` names each
member compared on weaker evidence. The one `Qualification` today is
`HashPrefixOnly`: an older fit record holds 8-character hashes, and the check
ran on that prefix (`compare.rs:1240`).

`compare` runs every check in the order of `CheckId`, on every table, and
records each result. Checks other than `MemberUsable` run over the usable
members.

A `Finding` is an enum with one variant per way a check can fail. It carries the
members partitioned by the value that differed, each group sorted by member name
and the groups sorted by value
(`TrainingBoundaryDiffers { groups: Vec<(f64, Vec<String>)> }`,
`ParticlesDiffer { .. }`, `DrawsDiffer { .. }`, `NotConverged { .. }`, and so
on). On `main` several findings name the first differing pair against the first
row (`compare.rs:1528`, `:2114`), which depends on the order of the arguments;
the partition does not. The sentence is rendered from the finding, so the
wording of findings changes in the step that introduces them.

A `Refusal` is a failed, unwaived check: `{ check, finding }`.
`Outcome::Refused` holds when any table has a refusal; the command then exits
with status 2, as it does today (`compare.rs:798-800`).

**Dependencies between checks.** A check that reads what an earlier check
established is `NotEvaluated { because }` when that check failed unwaived. That
is a recorded result, not an absence.

| Check                                                                             | Depends on                    | Why                                                                                      |
| --------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- |
| `TrainingBoundary`                                                                | `Conditioning`                | boundaries are compared among held-out members only                                      |
| `InSampleAblation`                                                                | `Conditioning`                | `FilterSettings`                                                                         |
| it reads the one conditioning the members share (`compare.rs:1038`, `:1100-1101`) |                               |                                                                                          |
| `Pairing`                                                                         | `StreamScored`, `ScoredSteps` | with step counts differing under the waiver it compares shared times only (`:1168-1172`) |
| `SharedData`                                                                      | `Pairing`, under `--streams`  | it assumes every member is scored on the same leaf                                       |

`Basis` fails when the members' traces are on different bases (a particle
filter's estimate against an ODE trajectory's exact density). It has no waiver,
as on `main`.

`StreamScored` exists under `--streams` only: it fails for a table when a member
cannot be projected onto that table's leaf (today's exit at `compare.rs:704`).
It has no waiver.

`NotCheckedReason` is an enum (`NoDerivationBlock`, `BareTrace`,
`NoDataHashRecorded`, `NoDiagnostics`, `UnknownDiagnosticsSchema`, ...). A
member that lacks the fact a check needs is listed in `not_checked`. It never
passes by default.

Three checks are new or changed relative to `main`:

- `FilterSettings` also fails on a differing number of posterior draws used.
- `FilterSeed` is `Noted` when seeds differ. It never refuses.
- `Profile` is `Noted` when a member was scored under no named profile, or the
  members' profile names differ while their values agree.

gh#1054 asks for the first two on `main` ahead of this proposal.

### The convergence refusal

`Convergence` judges each usable member and fails when any member's verdict is
not converged, chains frozen, or unassessable. Its waiver is a new flag,
`--allow-nonconverged`, following `fit run`'s existing
`--allow-nonconverged-source` (`rust/crates/cli/src/args/mod.rs:706`).

| Verdict                                    | For that member                            |
| ------------------------------------------ | ------------------------------------------ |
| Converged                                  | passes                                     |
| Not converged, chains frozen, unassessable | fails; waivable                            |
| Not assessed                               | passes, noted with camdl's recorded reason |
| Not applicable (a method without chains)   | passes, noted with its `GateVerdict`       |
| No convergence record (a bare trace)       | `not_checked`, `NoDiagnostics`             |

The last row is this proposal's call: a score with no fit behind it is treated
as every other missing fact is, reported as not checked and never assumed
converged. The verdict judged is that of the method leaf that supplied the
parameters the score was computed at (`ScoredFit.run_id`, proposal A).

**The sample judged is the sample scored.** `compare --exclude-chains` scores a
fit with some chains removed (`compare.rs:644-657`; recorded in the trace at
`:2554-2557`). The diagnostics a fit stores cover every chain. Judging those
would refuse a fit whose one frozen chain the user had just excluded, and under
the waiver would label its scores as from an unconverged sample when the scored
sample left that chain out; the reverse error is equally possible. So the
diagnostics in a member's `ConvergenceRecord` are computed over the chains that
were scored: `cli` calls the recomputation `fit summary --exclude-chains`
already uses (`recompute_subset_diagnostics`,
`rust/crates/cli/src/chain_selection.rs:298`) when the selection is not empty.
`UsableMember.chains_excluded` and `InputRecord.derivation` record the
selection. Proposal A's `ScoredFit.convergence` is filled the same way, so a
score artifact's convergence record describes the sample it was scored from.

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
    pub threshold: f64,                  // 1.05: below it, converged
    pub severe_threshold: f64,           // 1.5: at or above it, the band is Severe
    pub min_ess: Recorded<f64>,          // NotRecorded: rule 1 has no floor
}
pub struct ConvergenceDiagnostics {
    pub schema: String,                  // "camdl.fit-summary/v2"
    pub per_param: BTreeMap<String, ParamConvergence>,
    pub frozen_chains: Recorded<Vec<ChainIndex>>,
    pub chains_excluded: Vec<usize>,
}
pub struct ConvergenceRecord {
    pub verdict: Verdict,
    pub rule: ConvergenceRule,
    pub diagnostics: Recorded<ConvergenceDiagnostics>,
    pub gate: Recorded<GateVerdict>,
}
pub fn judge(d: &ConvergenceDiagnostics, rule: &ConvergenceRule) -> Verdict
```

A verdict is a function of diagnostics and a rule, and is computed when it is
read. It is not stored in the fit leaf.

**What the diagnostics are.** `PosteriorDiagnostics` is not a stored file. `cli`
builds it from the method's summary file (`ConvergenceMaps::read`,
`rust/crates/cli/src/fit/method_result.rs:1655`, `:1680`) and then, when the
fit's `fit_state.toml` records that every chain started at one point, replaces
each R̂ with that refusal (`demote_point_start`, `method_result.rs:584`;
`point_start`, `:1204`), because chains that began together agree by
construction. The value handed to `score`, and the value a `ConvergenceRecord`
stores, is the one after that step. `judge` therefore needs no file: a
point-started fit's stored diagnostics already say its R̂ was not assessable,
under any rule. A record built from the summary file alone would read such a fit
as converged; test 21 covers it.

**What moves, and what does not.** The verdict reads two things from a fit's
diagnostics: the per-parameter statistics and the frozen chains
(`max_rhat_status`, `method_result.rs:603-640`). `ConvergenceDiagnostics` holds
exactly those. `PosteriorDiagnostics` itself stays in `cli`: it also carries the
agreement of the chains' log-likelihoods (`chain_logliks`, a `cli` type with its
own family of types and tests) and a `findings` method that returns a `sim`
diagnostic type, neither of which the verdict reads and neither of which `score`
may depend on. `cli` builds a `ConvergenceDiagnostics` from a
`PosteriorDiagnostics` by copying the two fields.

Step 5 moves to `score::convergence`, verbatim, the types that move cleanly:

- from `cli`: `ParamConvergence`, `Stat`, `RhatBand` (with the two threshold
  constants it reads), `MaxRhat`, `MinEss`, `GateVerdict`, the serde default
  helpers only these use (`default_thin` stays with `PosteriorDiagnostics`;
  `undefined_stat` moves), and the `Verdict` type;
- from `rust/crates/sim/src/inference/convergence.rs`: the error types
  `ParamConvergence` embeds, `ConvergenceError`, `RhatRefusal` and `RhatDriver`.
  The estimator in that file stays in `sim`;
- from `rust/crates/cli/src/chain_index.rs`: the `ChainIndex` newtype with its
  directory-name methods (`dir_name`, `dir_in`, `parse_dir_name`), which use
  only path types and read no file. `refuse_one_based_layout`, which reads the
  file system, stays in `cli` as a function.

`demote_point_start` (a method of `PosteriorDiagnostics`, reading no file),
`point_start` (which reads `fit_state.toml`), `max_rhat_status` and
`report::verdict` do not move in step 5.

Step 10 then adds `judge`. Its body is `max_rhat_status` followed by
`report::verdict`'s mapping from `MaxRhat` to `Verdict` (`facts.rs:231-242`),
reading `d.per_param` and `d.frozen_chains` where the originals read `self`, and
the rule's thresholds where they read the constants.
`PosteriorDiagnostics::
max_rhat_status` and `report::verdict` become calls to
it, so there is one implementation. `Verdict` gains `Deserialize` in step 10,
and `Verdict::ChainsFrozen` keeps `Vec<ChainIndex>`.

**One place for the thresholds.** `RHAT_CONVERGED_THRESHOLD` and
`RHAT_SEVERE_THRESHOLD` (`method_result.rs:787`, `:792`) become the two
thresholds of `score::convergence::CURRENT_RULE`. Every site that reads either
constant reads the rule instead: `RhatBand::of`, `converged_at` as called in
`rust/crates/cli/src/fit/table_row.rs:393` and `:411`,
`rust/crates/cli/src/fit/gating.rs:842-844`, and the renderers in
`rust/crates/cli/src/fit/report/render.rs`. `fit summary` prints the rule beside
the verdict.

Where each thing is written:

- **The score artifact** (`ScoredFit.convergence`, proposal A) stores a
  `ConvergenceRecord`: the verdict under the rule current when the score was
  computed, that rule, and the diagnostics over the scored chains with the
  version of their computation.
- **The bundle record**, written by `camdl publish` (proposal 1), stores the
  same record at publish time. So a bundle always carries the diagnostics needed
  to judge it again.
- **The report** stores the one rule it judged every member by
  (`Report.convergence_rule`) and, per row, the verdict under that rule.

How `compare` judges a member, in order:

1. If the member's record holds diagnostics whose `schema` this build reads, it
   calls `judge` with `CURRENT_RULE`. Every such member is judged by one rule.
2. If the record holds a verdict and no diagnostics, it uses the recorded
   verdict. If the recorded rule differs from `CURRENT_RULE`, the
   `ConvergenceRule` check is `Noted`, naming the member and both rules.
3. If the diagnostics' `schema` is not one this build reads, the member is
   `not_checked` with `UnknownDiagnosticsSchema`, and the schema is named.
4. With no record it is `not_checked` with `NoDiagnostics`.

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
`Finding`, `NotCheckedReason`, `Qualification`, `SampleStatus`, `NoiseCheck`,
`Scope`, `MemberSource`, `Metric` and `Note`. No field of a `Report` is a
sentence.

### What `compare` guarantees

Each guarantee is a property of `score::compare` and is tested as stated.

1. **No difference without a clean preflight.** If a table has a refusal, every
   row's `difference` is `Baseline` or `NotFormed`, and no renderer prints a
   difference, a ratio or an evidence label for it.
2. **Every check has a result.** `checks` has exactly one entry per `CheckId`
   that applies to the table (`StreamScored` under `--streams` only), in
   declaration order, for every input.
3. **Unknown is never a pass.** For each check and each fact it reads, a member
   lacking that fact appears in that check's `not_checked`, whatever the outcome
   over the other members.
4. **Overrides are exact.** `overrides` holds a waiver if and only if its check
   failed and the waiver was given. A waiver given and not needed is absent.
5. **An unconverged member is refused or labelled.** A member whose verdict is
   not converged, chains frozen or unassessable either refuses the table, or
   appears with `NotConvergedOverridden` and an override, never otherwise.
6. **One rule.** Every member judged from diagnostics is judged by
   `Report.convergence_rule`; any member judged otherwise is named in a `Noted`
   outcome or in `not_checked`.
7. **Renderers add nothing.** For any `Report` `r`,
   `render(deserialize(serialize(r)))` equals `render(r)`, for each renderer,
   including a report holding non-finite scores.
8. **Deterministic.** Equal inputs give byte-identical serialized reports.
9. **The label follows from the row.** For every `Formed` row,
   `evidence::label(delta_elpd, se, mc_se)` equals its `evidence`.
10. **Antisymmetric.** Swapping the baseline between two members negates
    `delta_elpd`, keeps `se`, and keeps the evidence tier.
11. **Order-free.** Permuting the members permutes `inputs` and `rows` and
    changes nothing else: no number, no check result, no finding, no baseline.
12. **No figure combines streams.** No field holds a sum, mean or total across
    leaves; under `--streams` there is one table per leaf and nothing else.
13. **Self-identifying.** `engine` and every `InputRecord` are filled, with
    `NotRecorded` where the input does not say.
14. **An unusable member does not hide the rest.** With one `Unusable` member
    and two usable ones, every check other than `MemberUsable` has a result over
    the two.

It does not guarantee that a comparison is adequate. A fair comparison of poor
models is fair (principles, R6).

## Invariants, as tests

Beyond the fourteen above, each of which is one test over generated or fixture
inputs:

15. **Dependency list.** `score/Cargo.toml` names exactly `serde`, `serde_json`
    and `numerics` as dependencies.
16. **No ambient access.** The source test above passes, and fails when a file
    under `score/src` is given a line naming `std::fs`. `make check-score-wasm`
    passes in CI.
17. **Moves change no output.** See [Steps](#steps), phase 1.
18. **All refusals at once.** A fixture with three independent failures yields a
    report with three refusals, where `main` prints one.
19. **Dependent checks.** A fixture that fails `Conditioning` records
    `InSampleAblation` and `TrainingBoundary` as `NotEvaluated`; one that fails
    `Pairing` under `--streams` records `SharedData` as `NotEvaluated`.
20. **Failed and unchecked together.** Three members, two with differing
    recorded data and one with none, give `SharedData` a `Failed` outcome and
    one `not_checked` entry.
21. **The draw count.** Two members with 64 and 16 draws used fail
    `FilterSettings`; with differing seeds only, `FilterSeed` is `Noted` and the
    outcome is `Compared`.
22. **Rule in the verdict.** A fixture with a largest R̂ of 1.03 is `Converged`
    under a rule with threshold 1.05 and `NotConverged` under one with 1.01, and
    each serialized record names its rule.
23. **Re-judging without the fit.** A `ConvergenceRecord` with diagnostics,
    judged under a second rule, gives the verdict the fit's own diagnostics give
    under that rule. A point-started fit's record gives `NotAssessed` under
    every rule.
24. **Excluded chains.** A fit with one frozen chain is refused; the same fit
    with that chain excluded passes `Convergence`, and its record lists the
    excluded chain.
25. **Non-finite scores.** A member whose elpd is negative infinity yields a
    report that serializes, reads back equal, and renders "—" for its
    difference.
26. **Unknown values.** A report with `schema.major = 2` is refused by name; one
    with an unknown `CheckId` is read and the value reported as unknown.
27. **Mutation checks.** For tests 1, 5, 21 and 24, removing the property in the
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

## What a later browser build would need

Stated so that nothing here forecloses it; none of it is designed.

- The `score` crate as specified, which already compiles for
  `wasm32-unknown-unknown` (test 16) and takes its inputs as values.
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

Each step is one commit with a `Proposal:` footer. This is the one order for
both proposals; proposal A's steps are named `A1` to `A8` and placed here.

**How a move is checked.** A committed script, `scripts/check_score_moves.sh`,
fits two small committed models with a posterior method at fixed seeds, then
writes to a directory: `fit summary` in each of its formats;
`pfilter
--save-prequential` for one; and `compare` in table, markdown and JSON,
once at the defaults and once with `--draws 2 --replicates 2` so that both
combiners run, and once under `--streams`. It is run with the binary built at
the commit before step 1, and again after each move; the two directories must be
byte-identical (`diff -r`). The commit message states the comparison and its
result. This is the check PR #1048 used for its change to the filter. What it
does not cover, and the unit tests moved with the code therefore must: waived
comparisons, non-finite scores, and traces older than the current schema.

**Phase 1: moves, with no change to any function body.** Each commit moves items
verbatim, splits a file where only part of it moves, and edits `use` lines.

0. Add `scripts/check_score_moves.sh` and its two model configurations. Add unit
   tests that hold, as committed text, the table and markdown renderings of the
   existing in-memory fixtures (`single_stream_rows`, `two_stream_rows`, the
   ablation table; `compare.rs:4825`, `:4915`, `:5783`), one waived comparison
   per waiver, and one row with a non-finite score.
1. Create `score` with its dependency test, its source test and the
   `check-score-wasm` make target; update the dependency table in
   `.claude/rules/rust-conventions.md`. Move `log_sum_exp`, `sample_sd` and
   `logmeanexp_with_se` to `numerics`, and `json_f64` to `score::num`.
2. Move `Coverage` and the trace module (types, kernels, `project_onto_stream`)
   from `sim` to `score::trace`. `build_trace` and the tests that need `sim`'s
   random stream or the filter's output stay in `sim`.
3. Move `evidence.rs` to `score::evidence`.
4. Move `ReplicateTotals`, `paired_delta` and `paired_diffs` to `score`.
5. Move the convergence types listed under "What moves, and what does not" to
   `score::convergence`.

**Phase 2: the report.** Through steps 6 to 8 the step 0 renderings of
comparisons in which no check failed stay byte-identical. Renderings that state
a finding change in step 6, because findings are restated as partitions; that
commit's diff of the pinned text is the record of each change. The fixtures are
rebuilt as `Member` values in step 7 with their pinned text unchanged, except
the non-finite pin, which step 7 changes as described under "Numbers that may
not be finite". The JSON changes in step 9 only. No step of proposal A lands
before step 9.

6. `CheckId`, `CheckResult`, `Finding`, `Refusal` and `Waiver` in
   `score::preflight`. Each existing check in `compare.rs` is rewritten as a
   function returning a `CheckResult`, still over `cli`'s rows; `preflight`
   evaluates all of them and returns every result, and a refused comparison
   prints every refusal, where `main` prints the first. The check functions move
   into `score` in step 7, when `Member` exists for them to take. Tests 2, 3, 4,
   18, 19, 20.
7. `Member`, `Request`, `Report`, `Num`, `score::compare`; `cmd_compare` builds
   members and calls it. `Evidence` and the other label enums. The baseline tie
   rule. Tests 1, 8, 9, 10, 11, 12, 13, 14, 25.
8. `render::table` and `render::markdown` as functions of the `Report`; the old
   renderers deleted. Test 7.
9. `render::json` as the serialized `Report`; `docs/camdl-run-spec.md` documents
   it. This commit's subject is the JSON change.
10. `ConvergenceRule`, `ConvergenceDiagnostics`, `ConvergenceRecord` and `judge`
    with the rule passed in; `max_rhat_status` and `report::verdict` call it;
    every threshold site reads the rule; `fit summary` prints it. Tests 22, 23,
    and a test that `judge` gives, for every existing verdict fixture, the
    verdict `report::verdict` gave before.
11. The `Convergence` and `ConvergenceRule` checks, `--allow-nonconverged`,
    diagnostics over the scored chains, `SampleStatus` and its label in every
    format. Tests 5, 6, 24. Addresses the `compare` half of gh#716.
12. `FilterSettings` with draws, `FilterSeed`. Test 21. Closes gh#1054 if it has
    not landed already.
13. Schema versions and unknown-value handling for `camdl.report`. Test 26.

**Phase 3: the score artifact** (proposal A's steps, in its numbering).

14. A1: the pass record, `score::combine::finish` and trace schema 6. It deletes
    `combine_mixture` and `combine_scored_traces`, so nothing that combines
    traces remains in `cli`. It changes the printed coverage figures; the commit
    re-pins them and that is its subject.
15. A2, the row digest; A3, what a fit records and the prefix check; A4, the
    identity change, which needs the maintainer's confirmation and shares the
    hash-version bump with the docket's D25; A5, the `.meta.toml` reader.
16. A6 to A8: the artifact types and the `Profile` check, the `Score` store kind
    and `camdl score`, and `compare` reading through the store with `SharedData`
    switched to `rows_read` digests.
17. This proposal's documentation: `docs/commands.md`,
    `docs/methods/model-comparison.md`.

## Verification

Commands run on `main` at `e8e7c24a0` for the claims the design rests on.

- `sed -n 24,26p rust/crates/cli/Cargo.toml` → a `[[bin]]` named `camdl`; the
  manifest has no `[lib]`.
- `grep -n 'paired_delta(' rust/crates/cli/src/compare.rs` → the definition at
  3048 and calls at 3181, 3182, 3321, 3552, 3633, 3728.
- `grep -n 'pub fn evidence_label' rust/crates/cli/src/evidence.rs` → line 201,
  returning `&'static str`.
- `grep -n -i 'converge\|rhat\|verdict' rust/crates/cli/src/compare.rs` → two
  comment lines (1158, 4138) and no code.
- `grep -n 'RHAT_CONVERGED_THRESHOLD: f64' rust/crates/cli/src/fit/method_result.rs`
  → line 787, value 1.05.
- `grep -rn 'RHAT_CONVERGED_THRESHOLD' rust/crates/cli/src/fit/table_row.rs rust/crates/cli/src/fit/gating.rs`
  → `table_row.rs:393`, `:411`; `gating.rs:825`, `:842`, `:844`.
- `sed -n 2700,2740p rust/crates/cli/src/compare.rs | grep -n 'rng\|uniform'` →
  `StatefulRng::new_stream(pit_seed, PIT_RNG_STREAM)` and `pit_rng.uniform()`
  inside `combine_scored_traces`.
- `grep -n 'fn demote_point_start\|pub fn point_start' rust/crates/cli/src/fit/method_result.rs`
  → lines 584 and 1204.
- `grep -n 'pub fn recompute_subset_diagnostics' rust/crates/cli/src/chain_selection.rs`
  → line 298.
- `grep -rn 'report.json\|"verdict"' rust/crates/cli/src --include='*.rs'` → no
  site that writes a verdict to a leaf; `fit_summary.rs:377` holds it in a
  document built on read.
- `sed -n '/\[dependencies\]/,$p' rust/crates/numerics/Cargo.toml` → `libm`
  only.
- `gh issue view 725 --json state`, `716`, `312` → each `OPEN`.

The design was reviewed against the code by an independent agent on 2026-11-14;
its thirteen findings are resolved in the text above.
