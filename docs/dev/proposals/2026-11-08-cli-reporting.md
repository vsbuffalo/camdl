# Proposal: one reporting path for command output

**Status:** Proposal **Date:** 2026-11-08 **Motivation:** `camdl fit run`
repeats warnings once per chain, prints per-chain values without saying which
chain they belong to, writes colour codes into files, prints facts that are
recorded nowhere, and draws its end-of-run block with code separate from
`fit summary`'s. The cause is that output is printed wherever a fact is noticed,
through five mechanisms, instead of being built into a typed report and rendered
once.

**Prerequisite:** the stage → method rename (branch `refactor/stage-to-method`):
the names below (`method_dir`, `run_method`, `PmmhRun`, …) are the renamed ones.

## The rule this proposal sets

**Every fact a command shows is a value in a typed report before it is text.**
The same builder makes the report whether its inputs come from memory (at the
end of `fit run`) or from the files a run wrote (in `fit summary`), and the same
renderer draws it. Consumers that are not the terminal —
`fit summary --format
json`, camdl-scope, a future terminal UI — read the
report, not text, and never recompute what the report already decides.

Three consequences shape the design:

1. One builder and one renderer per fact. The end-of-run block and `fit summary`
   cannot disagree about a value, because there is one function that computes
   it.
2. The simulation library (`sim`) never prints. It returns values and records
   findings.
3. Colour is applied when text is written, by one policy per language.

## Scope

**Presentation, and the records presentation needs.** No statistic, threshold,
diagnostic trigger or diagnostic severity changes. These do change, each in its
own commit, with the change as its subject:

- `diagnostics.json` gains two finding kinds (`ScaleAtFloor`,
  `ScaleFarFromOptimum`) and a chain id on `AcceptanceRateUnhealthy`, so it
  holds more records and the printed warning count rises for fits that trigger
  them.
- `pmmh_summary.json` gains a per-chain `warmup` record, and `PfNoiseCheck` in
  `fit_state.toml` gains the mean `log L̂` it prints today.
- `fit summary --format json` gains `checks`, `warmup`, `findings` and a derived
  `verdict`, under a schema version bump.
- A failed write of `diagnostics.json` becomes an error (exit 1). Today it is
  discarded (`let _ = collector.write_json(…)`, five sites), and a run reports
  success with its findings unrecorded.

New fields on the structs that read stored files carry `#[serde(default)]`, so
files written before the change still load. None of these files is hashed into a
run's identity.

**Streams do not change.** stdout carries data, or a report command's report;
everything else goes to stderr. `status.rs` states the rule, the earlier
proposal `2026-05-30-cli-io-and-progress-ux.md` states it for banners, and
`cli/tests/cas_integration.rs` and `cli/tests/lineage_e2e.rs` assert that
`simulate` and `lineage` write nothing to stdout. `fit run` is not a report
command, so its end-of-run block stays on stderr; `fit summary` stays on stdout.
`camdl fit summary @label > summary.txt` gives a user the block in a file.

**Sentences do not change.** Messages move, merge and gain labels; their wording
does not. Where a claim is wrong, that is its own issue: the PMMH noise-check
conclusion is gh#1010. Text this proposal adds is listed under
[Text that is new](#text-that-is-new).

**Out of scope, with issues:** the outlier-chain flag (gh#1008); `compare`'s
data-path resolution and error ordering (gh#1009); threshold values (gh#84).

## What is wrong today

The example is the first-fit tutorial's failing fit (`docs/guide/first-fit/` on
the `docs/website` branch), `camdl fit run fit-beta.toml --label beta --seed
4`:
PMMH (particle marginal Metropolis–Hastings), four chains, two of which stick.
Its output is 84 lines, all on stderr.

### Five mechanisms print, and none sees the others

| Mechanism                                    | Where                                                                                                                                                  | Records anything?                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| print macros                                 | about 1,080 lines of `eprintln!`/`eprint!`/`println!`/`print!` in 40 files of `cli/src/`; 59 lines in `sim/src/inference/`                             | no                                           |
| `log::warn!` / `log::info!` via `env_logger` | progress off a terminal (`progress.rs:247`); warnings in `sim` (`lifecycle.rs:76`, `pgas.rs:1904,3374,3399`, `effect_plan.rs:374`, `obs_model.rs:138`) | no; printed with a timestamp and module path |
| `DiagnosticCollector`                        | `sim/src/inference/diagnostic.rs`                                                                                                                      | `diagnostics.json`, when its write succeeds  |
| `Notice` / `Notices`                         | `cli/src/notice.rs`; used by the parameter resolver, `obs_anchor.rs`, `cas_sink.rs`, `profile`, `survey`                                               | `run.json`'s display inputs                  |
| `status.rs` milestones                       | `cached`, `stored`, `compiled`, …                                                                                                                      | no                                           |

The progress bars (`progress::Reporter`, indicatif) are transient by design and
stay so.

### Two renderers draw the same facts

`fit run`'s end-of-run block comes from `ConvergenceDiagnostics::report`
(`cli/src/fit/runner.rs:3165`), which both prints the R̂ table and pushes each
`RhatHigh` finding (`:3253`); from each method's acceptance loop
(`cli/src/fit/pmmh.rs:778`); and from `DiagnosticCollector::render_to_stderr`
(`sim/src/inference/diagnostic.rs:797`). `fit summary`'s comes from
`bayesian_block` and `verdict_block` (`cli/src/fit/fit_summary.rs:1769`,
`:1854`), which read the method directory themselves. The `RhatBand` doc comment
(`method_result.rs`) records one disagreement between the two already: a
parameter in [1.05, 1.1) printed ✓ when the run finished and ✗ in `fit summary`.

`fit summary`'s builder, `build_summary_doc`, also ends its match on the method
with `_ => continue` (`fit_summary.rs:2964`): an NLopt run (`nl-sbplx`,
`nl-bobyqa`, `nl-lbfgs`) gets no `MethodReport`, so `--format json` omits it.

### What the reader sees

- The 8-line "the adaptive proposal scale sat at its lower bound" warning prints
  twice, word for word, once per stuck chain, naming neither. A closure inside
  the library's per-chain sampler prints it (`sim/src/inference/pmmh.rs:788`),
  and that closure sees one chain at a time.
- Four `adaptive proposal scale at end of warm-up: lambda = …` lines print in
  the order the chains' threads finish, unlabelled. The values are in
  `PMMHResult` (`final_scale`, `steps_at_scale_floor`) and recorded nowhere. The
  warm-up length (`adapt_steps`, a local at `sim/src/inference/pmmh.rs:641`) and
  the target acceptance (private to the adaptive proposal, `:320`) are not even
  returned.
- The checks before sampling (observation streams, particle-filter noise check,
  the method's caveat, the note on how chains start) are text only, except the
  noise check's `sigma`, `s`, particles and pairs, which `fit_state.toml`
  records as `pf_noise` (`PfNoiseCheck`, `pf_noise.rs:109`; `state.rs:180`).
- Acceptance and R̂ appear in tables, then again as sentences in
  `── diagnostics ──`.
- R̂ has four bars. The output names two of them, without roles:

  | Constant                     | Value | Where                  | Question                                                   |
  | ---------------------------- | ----- | ---------------------- | ---------------------------------------------------------- |
  | `RHAT_CONVERGED_THRESHOLD`   | 1.05  | `method_result.rs:600` | may the fit be reported converged (R̂ < 1.05)               |
  | `RHAT_REPORT_THRESHOLD`      | 1.1   | `runner.rs:2986`       | is an `RhatHigh` finding drawn (R̂ > 1.1, `runner.rs:3191`) |
  | `CONVERGENCE_ERROR_SEVERITY` | 1.5   | `diagnostic.rs:406`    | is that finding an error, not a warning (R̂ > 1.5)          |
  | `RHAT_SEVERE_THRESHOLD`      | 1.5   | `method_result.rs:604` | ✗ rather than ~: "the chains are in different places"      |

  The beta run prints "R̂ threshold 1.05" above its table and "(threshold 1.100)"
  in its diagnostics, which reads as a contradiction.
- The PMMH caveat from the method registry (`status_note`,
  `cli/src/fit/methods.rs`: "Acceptance rates degrade for long observation
  series (T > 500)") prints on every PMMH run without the series length; here T
  = 14.
- The `output:` line prints an absolute path.
- Off a terminal, progress reads
  `[2026-11-08T05:21:57Z INFO  camdl::progress] chain 1 1/3000 …`. GH #14 named
  the fix: "route progress through a dedicated non-`log::*` channel" (comment at
  `cli/src/main.rs:520`).
- camdl-scope recomputes the verdict in Python
  (`camdl_scope/diagnostics.py:258-278`) and labels R̂ > 1.1 an error, which is
  the mislabel above. It has no camdl value to read instead.

### Colour

Four policies, and places with none:

- `style::enabled` (`cli/src/style.rs:42`): off under `NO_COLOR` or when stderr
  is not a terminal.
- `should_use_color` (`cli/src/fit/fit_summary.rs:2637`): `--no-color`, then
  `NO_COLOR`, then `CLICOLOR_FORCE` (any value), then whether stdout is a
  terminal. This one is right.
- Another in `cli/src/fit/config_v2.rs:2917`.
- None: about 70 lines with literal `\x1b[` codes in 15 files; direct
  `owo_colors` calls in `status.rs:26,39`, `browse.rs` and `methods.rs:268-270`.
- The OCaml compiler sets `Ansi_tty` unconditionally in `run_check`
  (`ocaml/lib/compiler/inspect.ml:1437`) and `Diagnostics.render`
  (`diagnostics.ml:232`); `inspect` honours `--no-color` and otherwise also
  forces `Ansi_tty` (`inspect.ml:1380`). So `camdl check` writes colour into a
  pipe.

## Design

### The report

`fit summary` already has the report type: `FitSummaryDoc` and `MethodReport`
(`cli/src/fit/fit_summary.rs:2686`, `:2719`), serialized with a schema version,
behind `--format json`, `md` and `latex`. `MethodReport` holds the typed
`MethodResult`, which holds the posterior diagnostics and `MaxRhat`. There is no
second report type. What changes is that `MethodReport` is built by one builder
from one input type, and gains what the run knows and the record lacks.

```rust
/// Everything the report is built from. Two producers fill it: the end of
/// `run_method` from memory, and `fit summary`'s loader from a method
/// directory. Nothing else constructs a MethodReport.
pub struct MethodFacts {
    pub header: RunHeader,              // label, config, method, model, IR hash, estimate,
                                        // fixed, starts, output (relative)
    pub result: MethodResult,           // existing typed result, every method incl. Nlopt
    pub checks: Vec<Check>,
    pub warmup: Vec<ChainWarmup>,       // empty when adapt = false
    pub findings: Vec<Diagnostic>,      // one record per chain, as in diagnostics.json
    pub chain_detail: Option<ChainDetail>, // outlier scores, efficiency, forkable ensemble
    pub stored: Option<Stored>,         // leaf path, sampling and total elapsed, filter
                                        // evaluations and ms each; None until finalize
}

pub fn build_method_report(facts: MethodFacts) -> MethodReport;   // exhaustive over MethodResult

// MethodResult gains a Pfilter arm (log-likelihood mean ± sd across replicates,
// the replicate count and particles), so `fit run` with `algorithm = "pfilter"`
// has a report like every other method.

pub enum Check {
    ObsStream { name: String, carrier: String, family: String },
    FilterNoise { check: PfNoiseCheck, ll_mean: f64 },   // the existing record, plus its mean
    MethodCaveat { method: String, note: String, series_len: usize },
    Starts { mode: String, params: Vec<String>, note: String },
}

pub struct ChainWarmup {
    pub chain: usize,
    pub warmup_steps: usize,          // new PMMHResult field (today a local)
    pub steps_at_scale_floor: usize,  // PMMHResult
    pub final_scale: f64,             // PMMHResult
    pub target_accept: f64,           // new PMMHResult field (today private)
}
```

`Check` values are derived, not stored twice: `FilterNoise` from
`fit_state.toml`'s `pf_noise` (which gains `ll_mean`), `ObsStream` and
`MethodCaveat` from the config and the method registry, `Starts` from the starts
rule already in `fit_state.toml`. The ceiling verdict is computed from `s` and
the target, as it is today. `ChainWarmup` is recorded in `pmmh_summary.json`. A
resumed leaf skips warm-up; it copies its base leaf's `warmup` record with the
resume state, so its findings describe the kernel its draws were taken under.

**Every fact the run-end view prints is recorded**, so the loader can rebuild
it. Today's files lack several, and each gets a home:

| Fact                                                       | Recorded in (new unless noted)                                       |
| ---------------------------------------------------------- | -------------------------------------------------------------------- |
| config path, model path, estimate, fixed                   | `run.json` display inputs, beside `Notice`s (`RunHeader`)            |
| starts rule and its reason ("default: every estimated…")   | `RunHeader`; the reason is the same function the run calls today     |
| sampling plan (`burn_in`, `thin`, `adapt`, `proposal_sd`)  | `<method>_summary.json` `plan`                                       |
| per-chain acceptance                                       | `pmmh_summary.json` (already), carried into `MethodResult` per chain |
| sampling elapsed, filter evaluations, ms per evaluation    | `<method>_summary.json` `cost`                                       |
| total elapsed                                              | `run.json.inputs.wall_time_seconds` (already)                        |
| warm-up per chain                                          | `pmmh_summary.json` `warmup`                                         |
| outlier count and scores, forkable draws (verdict section) | derived from `chain_*/trace.tsv` and `draws.tsv` (already written)   |

Two facts are about the invocation rather than the run and are not recorded:
whether the IR came from the cache (`cached IR …`), and the live progress lines.
`RunHeader` marks the first as run-only.

The functions that derive values from a run's outputs take the data, not a path,
and both producers give them the values as written. `trace.tsv` stores
`log_likelihood` to four decimals, so the run passes each value through the
writer's own format and parse (`trace_value(x)`) before computing the outlier
scores; computing from full precision would let a score near its threshold flag
in one command and not the other. The same holds for every derived value whose
input file rounds. The functions: `posterior_summaries` takes the draws, the
outlier scores take the per-chain log-likelihood series, the efficiency figures
take the ESS and the elapsed time. The run calls them on what it holds in
memory; the loader reads the files (`draws.tsv`, `chain_*/trace.tsv`) and calls
the same functions. `MethodResult` carries per-chain acceptance alongside the
mean it carries today. `--exclude-chains` is a `fit summary` option: the loader
applies it to the data before calling those functions, so it is not part of
`MethodFacts`.

The run-equals-loader test compares `serde_json::to_value` of the two
`MethodFacts`, after dropping the run-only fields. Plain `f64` fields serialize
NaN and ±∞ all as `null`, which would make NaN equal ∞, so the test first passes
both values through a normalizer that writes non-finite numbers as the strings
`"NaN"`, `"inf"` and `"-inf"`. Equality is exact; there is no tolerance. No
`PartialEq` derive is needed on `MethodResult` or `Diagnostic`; their
`timestamp` field is run-only.

`MethodReport` derives only `Serialize`, so the `#[serde(default)]` attributes
belong on the stored-file structs the loader reads (`PmmhSummary`, `FitState`,
…), not on the report.

The verdict is computed by one function and also serialized, so a consumer reads
it rather than recomputing it:

```rust
/// Decided by MaxRhat alone. The findings count is reported beside it.
pub enum Verdict {
    Converged    { max_rhat: f64, param: String },
    NotConverged { max_rhat: f64, param: String, band: RhatBand },
    NotAssessed  { reason: String },  // MaxRhat's not-assessed arms, incl. gh#890's point start
    NotApplicable,                    // IF2, NLopt and pfilter: no chains to compare
}
pub fn verdict(r: &MethodReport) -> Verdict;   // also written as MethodReport.verdict
```

A run that fails has no `MethodResult`. It still has a report:

```rust
pub enum MethodOutcome {
    Completed(MethodReport),
    /// Every chain failed to start, no chain reached a finite log-likelihood,
    /// or the method returned an error. Built from the header, checks and
    /// findings, which exist before sampling ends.
    Failed { header: RunHeader, checks: Vec<Check>, findings: Vec<Diagnostic>, error: CommandError },
}
```

### Findings

Construction stays as it is, `collector.push(kind)`. Two variants are new, both
`Severity::Warning`, which is what they print as today:

```rust
DiagnosticKind::ScaleAtFloor { chain_id: usize, steps_at_floor: usize,
                               warmup_steps: usize, target_accept: f64 }
DiagnosticKind::ScaleFarFromOptimum { chain_id: usize, final_scale: f64 }
```

The command-line crate pushes them from `PMMHResult` after the chains return,
with today's rule kept: `ScaleFarFromOptimum` is pushed only when the chain
never reached the floor (`sim/src/inference/pmmh.rs:829`). The library prints
neither. `AcceptanceRateUnhealthy` gains `chain_id: Option<usize>`. The
`RhatHigh` push moves out of `ConvergenceDiagnostics::report` into the method's
run, before the printer is deleted.

Findings merge only by an explicit rule. The chain id stays inside each variant
(`BadInit { chain_id, .. }` already does this), so there is one source for it:

```rust
impl DiagnosticKind {
    pub fn chain(&self) -> Option<usize>;
    /// Exhaustive: a new variant does not compile until it decides.
    pub fn merge_key(&self) -> MergeKey;
}
pub enum MergeKey {
    Never,   // BadInit, and every variant whose detail is per-chain prose
    Group { variant: &'static str, identity: Vec<(&'static str, String)> },
}
```

Findings with equal keys print as one entry naming every chain, with each
chain's own value beside its id; per-chain values are never part of the key and
never hidden. The two acceptance findings in the beta run (`rate` 0.0005 and
0.001) have the key
`(AcceptanceRateUnhealthy, side=below, kernel=random_walk,
band=[0.15,0.50])`.

Findings print once, merged, in the end-of-run block, and are pushed on the
command's thread after the chains join, in chain order. While the run is going,
a chain that finishes prints one live line if its result would raise a per-chain
finding, such as
`chain 0: proposal scale at its floor for 241 of 1000 warm-up steps (details at
the end)`,
so a long run does not hide a problem until it ends. The chain workers reach
`Out` through `Out::live()`, a `LiveHandle` that is `Clone + Send + Sync` and
serializes its writes behind a lock; it offers `progress` and `chain_finished`
and nothing that records. The live line and the finding are decided by the same
predicate on the chain's result, so a line is printed exactly when the finding
is later pushed: one free function,
`scale_at_floor(chain, &PMMHResult) -> Option<DiagnosticKind>`, decides both;
the live line prints when it returns `Some`, and the command pushes that same
value after the join.

Every per-chain acceptance rate prints through one formatter, `fmt_rate`: one
decimal at 1% and above, two significant digits below (`39.6%`, `0.10%`,
`0.050%`). The acceptance table and the acceptance findings use it, so a chain's
rate never appears at two precisions in one block. Nothing else changes format:
the progress lines keep `acc={:.0}%`, the healthy band keeps `[15%, 50%]`, and
the ESS share keeps `(0.7% of 800 draws)`.

Checks are not findings. The count line counts findings; checks show their own ⚠
in the checks section.

### The reporting interface

Commands get one handle and print through nothing else:

```rust
pub struct Out { /* painters, sinks, collector, notices, progress, heartbeat */ }

impl Out {
    // stderr
    pub fn milestone(&self, verb: &str, detail: impl Display);  // `   stored …`
    pub fn check(&mut self, c: Check);
    pub fn progress(&self, p: Progress);                         // throttled; bar on a terminal, plain line off it
    pub fn finding(&mut self, kind: DiagnosticKind);             // collector.push; live line if per-chain
    pub fn notice(&mut self, n: Notice);
    pub fn method_report(&mut self, o: &MethodOutcome);          // end-of-run block, rendered once
    pub fn error(&mut self, e: CommandError) -> ExitCode;        // marks the heartbeat failed if a method is running
    // stdout
    pub fn report(&mut self, lines: &[Line]);                    // a report command's report (fit summary, browse, fit diff)
    pub fn data(&mut self) -> impl Write + '_;                   // data commands (simulate's TSV, cat)
}

pub struct Progress { pub chain: usize, pub step: u64, pub of: u64, pub loglik: f64, pub accept: f64 }
pub struct CommandError { pub code: ExitCode, pub message: String, pub hint: Option<String> }
```

`CommandError` carries the exit code each site uses today (1 at 382 sites, 2 at
8), so no exit code changes. Panics and clap's argument errors do not pass
through `Out`; they keep their current behaviour.

A call site, before. The acceptance report in `cli/src/fit/pmmh.rs:778`:

```rust
eprintln!("\nacceptance rates:");
for (chain_id, result) in &results {
    let status = if result.acceptance_rate < lo {
        "\x1b[31m✗ too low\x1b[0m"
    } else if result.acceptance_rate > hi {
        "\x1b[33m~ high\x1b[0m"
    } else {
        "\x1b[32m✓\x1b[0m"
    };
    eprintln!("  chain {}: {:.1}% {}", ChainIndex(*chain_id), result.acceptance_rate * 100.0, status);
    if let Some(d) = acceptance_diagnostic(result.acceptance_rate, None, kernel) {
        collector.push(d);
    }
}
```

After. The run records; the renderer draws the table from the report:

```rust
for (chain_id, r) in &results {
    if let Some(kind) = acceptance_diagnostic(Some(*chain_id), r.acceptance_rate, None, kernel) {
        out.finding(kind);
    }
    if r.steps_at_scale_floor > 0 {
        out.finding(DiagnosticKind::ScaleAtFloor {
            chain_id: *chain_id,
            steps_at_floor: r.steps_at_scale_floor,
            warmup_steps: r.warmup_steps,
            target_accept: r.target_accept,
        });
    } else if scale_is_far_from_optimum(r.final_scale) {
        out.finding(DiagnosticKind::ScaleFarFromOptimum { chain_id: *chain_id, final_scale: r.final_scale });
    }
}
```

The noise check, today a block of `eprintln!` in `pf_noise.rs`:

```rust
out.check(Check::FilterNoise { check: noise, ll_mean });
```

Progress, today `log::info!` off a terminal and a bar on one:

```rust
out.progress(Progress { chain, step, of: n_steps, loglik, accept });
```

Two tests keep it that way. In `cli/`, no source file other than `out.rs` and
`style.rs` contains a print macro, a literal escape code or an `owo_colors`
call; data writers get their writer only from `Out::data`, so the test has no
per-file exemptions. In `sim/src/`, no non-test code contains a print macro,
`log::warn!` or `log::info!`, outside one module, `sim::trace`. The seven
`log::warn!` sites (`lifecycle.rs:76`, `pgas.rs:1904,3374,3399`,
`effect_plan.rs:374`, `obs_model.rs:138`, `inference/linalg.rs:18`) become
findings or returned values, and `warn_zero_firings`
(`transition_diagnostics.rs:66-84`) returns the transitions that never fired,
which the command reports as a `Notice`. `log::debug!` and `log::trace!` stay
for `RUST_LOG` debugging, as `CLAUDE.md` documents.

`sim::trace` holds the developer traces that are switched on by an environment
variable and must reach stderr from inside the engine: `CAMDL_TRACE_STEPS`
(`chain_binomial.rs:880-902`, and the intervention firings in `effects.rs:153`),
which `CLAUDE.md` lists under "Diagnostics that already exist", and
`CAMDL_EVAL_UNRESOLVED` (`eval_stats.rs:77`). Each becomes one named function in
that module, which reads its variable once and writes to stderr. Their output
does not change.

### Rendering and colour

```rust
pub struct Span { pub text: String, pub style: Option<Style> }
pub struct Line(pub Vec<Span>);
pub enum Style { Ok, Warn, Err, Info, Bold, Dim, Code }

/// Which sections a view shows. Each section is one function; a view is a list
/// of sections, so the run-end block and fit summary share every section they
/// both show.
pub enum View { RunEnd, Summary }

pub fn render_report(o: &MethodOutcome, view: View) -> Vec<Line>;   // pure

pub struct Painter { color: bool }
impl Painter {
    /// should_use_color's policy, moved to style.rs and parameterized by
    /// stream: --no-color, else NO_COLOR, else CLICOLOR_FORCE (any value),
    /// else whether this stream is a terminal.
    pub fn for_stream(stream: Stream, no_color_flag: bool) -> Self;
    pub fn write(&self, w: &mut impl Write, lines: &[Line]) -> io::Result<()>;
}
```

`View::RunEnd` shows what `fit run` shows today; `View::Summary` shows what
`fit summary` shows today, including the outlier table, efficiency and forkable
ensemble from `ChainDetail`. Neither view gains or loses content. Colour is
applied only in `Painter::write`, so no string carries another stream's colour
decision, and `style.rs`'s string helpers go once their callers build spans.

The OCaml compiler gets the same policy in one function used by every
`set_style_renderer` call: `--no-color`, `NO_COLOR`, `CLICOLOR_FORCE`, then
`Unix.isatty` on that stream.

### How the types flow

```
fit.toml ─▶ run_method(&mut out)
             ├─ checks ───────────── out.check(Check) ─────────▶ stderr now; records as above
             ├─ sim::…::run_chain(…, on_step) ─▶ PMMHResult per chain
             │      on_step ──────── out.progress(Progress) ───▶ stderr
             ├─ from the results ─── out.finding(kind) ────────▶ collector ─▶ diagnostics.json (write failure = error)
             ├─ warm-up ──────────────────────────────────────▶ pmmh_summary.json: warmup
             └─ finalize ─▶ MethodFacts (from memory) ─▶ build_method_report ─▶ MethodOutcome
                                                                 │
                                       render_report(RunEnd) ─▶ Painter ─▶ stderr

method dir ─▶ fit summary's loader ─▶ MethodFacts (from files) ─▶ build_method_report ─▶ MethodOutcome
                                                                 ├─ render_report(Summary) ─▶ stdout
                                                                 └─ --format json ─▶ camdl-scope, a TUI
```

The run builds from memory, so a write failure cannot make the screen misreport,
and a failed run, whose summary file was never written, still has a report. The
loader produces the same `MethodFacts` from files, which is the point where the
two can be tested equal: for each fixture, the facts the run built and the facts
the loader reads back from that run's directory must be equal.

The report is built after `finalize`, because `MethodResult::load_from` reads
`run.json.inputs`, which `finalize` writes (`resolve.rs:490`), and the `stored`
line needs the leaf path. Its last line is the `stored` milestone.

### Live consumers

camdl-scope reads the run directory, and `fit summary --format json` gives it
camdl's verdict, checks, warm-up and findings. It follows a running fit through
the growing `trace.tsv` files and `progress.json`'s heartbeat. A terminal UI
needs the live events as well, so `Out` appends each `check`, `progress` (at the
throttled rate), `finding` and `method_report` as one JSON line to
`events.jsonl` in the method directory. `Out` owns the file and the chain
threads reach it only through `Out`, so there is one writer. The file is listed
among the leaf's known non-tabular outputs, so `fit_output_schema` does not
classify it. It is not hashed into the run's identity. This is step 9, last,
because nothing needs it yet; it is the seam a terminal UI attaches to.

### Ordering

`fit run` writes only to stderr, which is unbuffered, so its output order is the
order of the calls. `Out` clears the progress bars before writing any other
line. For commands that write to both streams, `Out` flushes stdout before each
stderr write. A `BrokenPipe` on stdout is a normal exit (status 0), so
`… | head` does not panic.

## Before and after

Before: the beta run (excerpts of the 84 lines of stderr; `…` marks cuts).

```
   cached IR for sir-beta.camdl (0c033f4c)
fit: fit-beta.toml (method=pmmh)
  model:    sir-beta.camdl
  estimate: beta, gamma
  fixed:    N0, I0
  starts:   from_prior (one independent draw per chain) — default: every estimated parameter declares a prior
  output:   /private/tmp/…/streams/results/fits/fit_beta-e0b51b25

── method: pmmh ──
⚠ Acceptance rates degrade for long observation series (T > 500); prefer PGAS — the default Bayesian path — on long chain-binomial series.

  note: `starts = "from_prior"` draws every chain's start, so … (one 300-character line)

pfilter noise check (1000 particles, 20 evaluation pairs)...
  … 14 lines …
pmmh: 4 chains × 3000 steps × 1000 particles, burn_in=1000, thin=10, adapt=true
  proposal_sd (transformed): [beta=0.2500, gamma=0.2500]
[2026-11-09T03:37:56Z INFO  camdl::progress] chain 1 1/3000  ll=-319.5  acc=0%
  … 3 more timestamped lines …
  warning: the adaptive proposal scale sat at its lower bound (lambda = 4.5e-5) … (8 lines)
  adaptive proposal scale at end of warm-up: lambda = 4.540e-5
  warning: the adaptive proposal scale sat at its lower bound … (the same 8 lines)
  adaptive proposal scale at end of warm-up: lambda = 4.540e-5
  … 2 timestamped lines, 2 more unlabelled lambda lines …
acceptance rates:
  chain 0: 0.1% ✗ too low
  …
R̂ (rank-normalized split, Vehtari et al. 2021) / ESS — R̂ threshold 1.05:
  beta         Rhat=2.250 ✗ ESS bulk=5 tail=4 (0.7% of 800 draws)
  …
── diagnostics ──────────────────────────────────────
  x chain acceptance rate 0.1% is outside healthy range [15%, 50%] (random-walk MH).
  x chain acceptance rate 0.1% is outside healthy range [15%, 50%] (random-walk MH).
  x rank-normalized split R̂ for 'beta' is 2.250 (threshold 1.100). Chain estimates have not converged.
  x rank-normalized split R̂ for 'gamma' is 3.325 (threshold 1.100). Chain estimates have not converged.
  4 error(s), 0 warning(s), 0 info
pmmh complete in 30.7s (12000 PF evaluations, 10.2ms/eval): results/fits/…/seed_4-8b49cfaa/
  loglik at the highest log-posterior draw (natural-scale prior, no Jacobian): -58.7 (chain 3)
   stored pmmh · results/fits/…/seed_4-8b49cfaa · 31.1s
```

After, all on stderr. The chain-to-λ assignment is illustrative: today's output
does not record it.

```
   cached IR for sir-beta.camdl (0c033f4c)
fit: fit-beta.toml (method=pmmh)
  model:    sir-beta.camdl
  estimate: beta, gamma
  fixed:    N0, I0
  starts:   from_prior (one independent draw per chain) — default: every estimated parameter declares a prior
  output:   results/fits/fit_beta-e0b51b25

── method: pmmh ──
⚠ Acceptance rates degrade for long observation series (T > 500); prefer PGAS — the default Bayesian path — on long chain-binomial series. This series has T = 14.

  note: `starts = "from_prior"` draws every chain's start, so … (the same line)

pfilter noise check (1000 particles, 20 evaluation pairs)...
  … the same 14 lines, observations (1 stream) included …
pmmh: 4 chains × 3000 steps × 1000 particles, burn_in=1000, thin=10, adapt=true
  proposal_sd (transformed): [beta=0.2500, gamma=0.2500]
chain 1 1/3000  ll=-319.5  acc=0%
  …
  chain 0: proposal scale at its floor for 241 of 1000 warm-up steps (details at the end)
  chain 1: proposal scale at its floor for 365 of 1000 warm-up steps (details at the end)
chain 3 2924/3000  ll=-59.1  acc=37%

✗ NOT converged — the chains are in different places
  … fit summary's verdict lines, as it prints them today …
  R̂: below 1.05 counts as converged; above 1.1 draws a finding, a warning up to 1.5
  and an error above it.

acceptance rates:
  chain  acceptance        λ end of warm-up   steps at floor
  0        0.050% ✗ too low        4.540e-5       241 / 1000
  1         0.10% ✗ too low        4.540e-5       365 / 1000
  2         39.6% ✓                7.175e-1         0 / 1000
  3         40.2% ✓                3.763e-1         0 / 1000

R̂ (rank-normalized split, Vehtari et al. 2021) / ESS — R̂ threshold 1.05:
  beta         Rhat=2.250 ✗ ESS bulk=5 tail=4 (0.7% of 800 draws)
                 R̂ = max(bulk 2.250, folded 1.662), classic 9.361; the bulk half is larger — the chains disagree about where the posterior sits
                 per-chain ESS [100, 3, 200, 134]
  gamma        Rhat=3.325 ✗ ESS bulk=6 tail=6 (0.8% of 800 draws)
                 R̂ = max(bulk 1.734, folded 3.325), classic 5.143; the folded half is larger — the chains agree on location and disagree on spread
                 per-chain ESS [100, 3, 124, 149]
  draws.tsv: 800 posterior samples (all 4 params)

── diagnostics ──────────────────────────────────────
  x chains 0, 1: chain acceptance rates 0.050% and 0.10% are outside healthy range [15%, 50%] (random-walk MH).
  x rank-normalized split R̂ for 'beta' is 2.250 (threshold 1.100). Chain estimates have not converged.
  x rank-normalized split R̂ for 'gamma' is 3.325 (threshold 1.100). Chain estimates have not converged.
  ! chains 0, 1: the adaptive proposal scale sat at its lower bound (lambda = 4.5e-5)
           for 241 and 365 of 1000 warm-up steps: no scale reached the
           33.7% target acceptance, so the proposal is far narrower
           than the posterior and the draws taken under the frozen kernel do
           not cover it. Under a particle-filter likelihood the attainable
           acceptance is capped by the spread of log L-hat, so raise the
           particle count (aim for a log-likelihood standard deviation near 1
           at the mode). A longer chain will not help.
  4 error(s), 2 warning(s), 0 info

pmmh complete in 30.7s (12000 PF evaluations, 10.2ms/eval)
  loglik at the highest log-posterior draw (natural-scale prior, no Jacobian): -58.7 (chain 3)
   stored pmmh · results/fits/fit_beta-e0b51b25/pmmh-83841c1b/seed_4-8b49cfaa · 31.1s
```

What changed, and nothing else did:

- the order of the sections is today's; the per-chain values are labelled and in
  chain order, in one table under today's `acceptance rates:` heading;
- the two identical scale-floor warnings are one, naming both chains and both
  step counts, and are now counted (`2 warning(s)`);
- the two acceptance findings are one, naming both chains;
- every rate prints through `fmt_rate`, so chain 0's 0.0005 reads `0.050%`
  everywhere instead of `0.1%`;
- `fit summary`'s verdict section, as it prints today, comes before the tables,
  followed by the R̂ legend line;
- progress lines lose their timestamps and module path, and the output path is
  relative.

### Text that is new

Every template below is new or changed; everything else prints as today.

| Template                                                                                                                     | Where                            | From                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `R̂: below 1.05 counts as converged; above 1.1 draws a finding, …`                                                            | after the verdict                | the four constants                                                      |
| `This series has T = {n}.`                                                                                                   | after the PMMH caveat            | the data                                                                |
| `chain {k}: proposal scale at its floor for {n} of {m} warm-up steps (details at the end)`                                   | live, as a chain finishes        | `ScaleAtFloor::applies`                                                 |
| `chain  acceptance  λ end of warm-up  steps at floor`                                                                        | the acceptance table's header    | replaces `chain {k}: {rate} {glyph}`                                    |
| `chains {list}: chain acceptance rates {r} and {r} are outside …`                                                            | a merged acceptance finding      | the one-chain sentence, made plural                                     |
| `chains {list}: the adaptive proposal scale … for {n} and {n} of {m} …`                                                      | a merged floor finding           | the one-chain sentence with values joined                               |
| `chain {k} {step}/{n}  ll={x}  acc={r}`                                                                                      | plain progress                   | today's line without the log prefix                                     |
| `pmmh complete in {t}s ({n} PF evaluations, {m}ms/eval)`                                                                     | after the diagnostics            | today's line without its trailing path, which the `stored` line carries |
| `fit summary`'s verdict lines (`max R̂ … threshold 1.05`, `N of M parameters above threshold`, outlier count, forkable draws) | after the headline, in `fit run` | `fit summary`'s verdict section, new to `fit run`'s output              |

A merged finding joins its chains with commas and its values with "and"; a
finding about one chain prints its one-chain sentence, which is today's.

The verdict lines are new to `fit run` but not new text: `View::RunEnd` shows
the same section function `View::Summary` does, so they read as `fit summary`
prints them.

## Rollout

Each step lands green with `make test`. Steps 1 and 2 are independent of each
other and of the rest.

1. **One colour policy, Rust.** `Style`, `Span`, `Line` and
   `Painter::for_stream` (from `should_use_color`) land here, with
   `Painter::paint(&str, Style) -> String` for call sites that still build
   strings. Route the literal escape codes, the direct `owo_colors` calls and
   `config_v2.rs`'s policy through it. `sim`'s `render_to_stderr` takes a
   `color: bool` from its caller until step 5 moves it, since `sim` cannot
   depend on `cli`. Add the no-escape-code test for `cli/`. Test helpers that
   read stderr clear `CLICOLOR_FORCE` (CI may set it,
   `cli/tests/design_preserving_simulate.rs:645`).
2. **One colour policy, OCaml.** One function decides the `Fmt` renderer for
   every call site.
3. **Write failures are errors.** The five `let _ = collector.write_json(…)`
   sites return the error.
4. **Record what the run knows.** `PMMHResult` gains `warmup_steps` and
   `target_accept`; `pmmh_summary.json` gains `warmup`, `plan` and `cost`;
   `PfNoiseCheck` gains `ll_mean`; `run.json` gains `RunHeader`; the resume path
   carries `warmup` over. Output unchanged.
5. **Findings.** `ScaleAtFloor`, `ScaleFarFromOptimum`, `scale_at_floor`,
   `chain_id` on `AcceptanceRateUnhealthy`, `chain()` and `merge_key()`. The
   `RhatHigh` push moves into each method's run and is removed from
   `ConvergenceDiagnostics::report` in the same commit, so it is pushed once.
   `render_to_stderr` moves to the CLI and merges; `fmt_rate`.
   `sim/src/inference/pmmh.rs` stops printing. This step changes
   `diagnostics.json` and the warning count, which is its commit subject.
6. **`MethodFacts`, the builder and `MethodOutcome`, summary side.**
   `MethodResult` gains its Pfilter arm and per-chain acceptance;
   `build_summary_doc`'s match becomes exhaustive; the derivation functions take
   data, not paths; `fit summary`'s loader produces `MethodFacts`.
   `fit summary --format json` gains `checks`, `warmup`, `findings` and
   `verdict` under a schema bump. `fit summary`'s output is otherwise unchanged.
7. **`Out`, views and the end-of-run block, run side.** `Out` and `LiveHandle`;
   `fit run` builds `MethodFacts` from memory and calls
   `render_report(_, View::RunEnd)`; `fit summary` calls
   `render_report(_, View::Summary)`. `ConvergenceDiagnostics::report`'s
   printing and the per-method acceptance printers are deleted. The live
   per-chain lines. The run-equals-loader test, which needs both producers,
   lands here.
8. **Progress and the smaller items.** Plain progress lines from `Out` without
   `env_logger` (GH #14's option 2, which also removes the `Info` auto-bump at
   `main.rs:530`); relative paths; the series length after the PMMH caveat.
9. **The other commands, PGAS and the library.** One commit per command
   (`pfilter`, `profile`, `survey`, `compare`, `predict`, `simulate`, `lineage`,
   `browse`, `fit diff`, `fit table`), each moving its prints and its exit sites
   onto `Out` with their exit codes; then `pgas.rs`'s prints, `sim::trace`, the
   seven `log::warn!` sites and `warn_zero_firings`; then the two no-print
   tests.
10. **`events.jsonl` and documentation colour.** `Out`'s event sink; then
    `docs/_web/camdl-run.lua` (on `docs/website` until it merges) sets
    `CLICOLOR_FORCE=1` and maps every code `Painter` and the OCaml policy emit,
    the compiler's magenta included, to `<span>`s with theme colours. An unknown
    code fails the render.

## How we know nothing was dropped

Before step 4, for each fixture, record the facts its current output shows as
(label, value, stream) triples, including facts that are sentences with no
number (`("noise check", "ceiling below target", stderr)`). The recording is a
committed file, reviewed once by a person. After each step a test runs the
fixture and checks that every triple is present: the value under the same label,
in the same stream. Hashes, timings and absolute paths are not facts and are not
recorded. A fact whose format changes (`0.1%` → `0.05%`) is changed in the
recording in the same commit, with the reason, so every change to what is shown
is a reviewed edit to one file.

The fixtures are their own runs, with the smallest settings that still produce
each kind of fact, not the tutorial's fits: their values differ from the
tutorial's, and the recording describes them. They cover a PMMH fit with stuck
chains, a converged PMMH fit, a PGAS fit, an IF2 fit, an NLopt fit, a fit whose
chains all fail to start, a resumed fit, and the `compare` and `pfilter` outputs
of one fit.

## What this removes and adds

Removed: `ConvergenceDiagnostics::report`'s printing, the per-method acceptance
and R̂ printers, `render_to_stderr` in `sim`, the library's `warn` closure,
`style.rs`'s string helpers, three of the four Rust colour policies, the
`env_logger` progress path, and every literal escape code. Added: `Out`,
`MethodFacts` and the builder, `MethodOutcome`, `Line`/`Span`/`View`/`Painter`,
`Check`, `ChainWarmup`, `merge_key`, two finding kinds and `events.jsonl`. The
net change in lines is small. The change is in where printing lives: from 40
files in `cli/src/` and the library to `out.rs`, `style.rs` and the renderers,
with `sim` printing nothing.

## Implementation notes

Where the implementation of steps 1–5 differs from the text above, and why.

- **Eight discarded writes, not five.** `let _ = collector.write_json(…)`
  appeared at eight sites: three in `pmmh.rs`, three in `pgas.rs`, two in IF2's
  `runner.rs`. All eight return the error. On a path where the run is already
  failing, the run's own error stays first and the unwritten record is appended
  to it (`fit::write_diagnostics_then_fail`), so a write failure never hides why
  the run stopped. IF2's successful path writes no `diagnostics.json` at all;
  step 6's loader has to account for that.
- **The acceptance merge key carries the severity.** Two below-band rates, one
  at 0.1% (an error) and one at 12% (a warning), would otherwise share
  `(side, kernel, band)` and print under one icon. `merge_key` adds `severity`
  to the identity, so each entry's icon is true of every chain it names.
- **One-chain entries carry their chain.** A finding about one chain prints its
  one-chain sentence with `chain k:` in front (`BadInit`, whose sentence already
  names its chain, excepted), so no per-chain value prints unlabelled. Three or
  more merged values read `a, b and c`.
- **The library's resume line goes.** `run_pmmh` printed
  `resuming from step N...` once per chain beside the CLI's own
  `chain k: resuming from step N`; the CLI line carries the fact. The
  already-completed warning moved to the CLI's resume loader, word for word.
- **Until step 7 draws the acceptance table, the λ lines print after the chains
  join**, in chain order, as
  `chain k: adaptive proposal scale at end of
  warm-up: lambda = …`.
- **Records.** `PfNoiseCheck.ll_mean` reads as NaN from a `fit_state.toml`
  written before it existed, so an old record cannot show a mean it never had.
  `PMMHResult.target_accept` is `pmmh::target_acceptance(d)`, the formula the
  adaptive proposal already used, now a free function; `warmup` is recorded only
  when `adapt = true`. A resumed chain whose base has no `warmup` record gets
  none, rather than its own call's partial count.
- **The OCaml policy reaches compile errors camdl relays.** camdl captures
  camdlc's stderr when compiling, so camdlc sees a pipe; camdl passes its own
  stderr decision on as `CLICOLOR_FORCE=1` or `NO_COLOR=1`
  (`util::pass_stderr_color_policy`).

Decided during step 6:

- **`fit summary`'s text output comes from the report too.** `format_text`
  rendered from `MethodResult` and the leaf's files and never touched
  `MethodReport`; only json, md and latex went through `build_summary_doc`. Step
  7 rebuilds it on `MethodReport` through `render_report`, so all four formats
  come from one report, and the facts fixture is the proof that nothing was
  dropped: a changed fact is a named edit to `facts.tsv`.
- **IF2 goes through the builder like every other method.** `If2Run::load` kept
  a lossy summary and `if2_method_report` read `fit_state.toml` a second time.
  `If2Run` now carries `record: If2Record`, everything the IF2 report and text
  read, from the one load; `gating::assess_if2` takes the record.
- **The summary's private per-chain read-back is `ChainDetail`.** It was named
  `MethodFacts`; it is the proposal's `ChainDetail`, renamed before the real
  `MethodFacts` was built around it.
- **A successful IF2 run writes `diagnostics.json`.** It did not, so the loader
  had no IF2 findings to read.
- **Records the loader derives checks from.** `run.json` `inputs` gains
  `unused_starts` (`{rule, params}`), from which `Check::Starts` and its note
  are rebuilt; `pmmh_summary.json` gains `acceptance_chains`, the chain each
  `acceptance_rate` belongs to (refused chains leave gaps), from which
  `PmmhRun::acceptance_per_chain` is read. `Check::FilterNoise` carries the
  `PfNoiseCheck` alone, since the record holds its own `ll_mean`.
- **`Verdict` has an `Unassessable { params }` arm**, for `MaxRhat`'s "R̂
  unavailable for a reason that is a problem": not converged, with no number to
  report. `NotConverged` needs a `max_rhat` it does not have.
- **The JSON report also carries `header` and `stored`**, beside `checks`,
  `warmup`, `findings` and `verdict` (schema version 4). The calendar context a
  report's date annotations need is passed to `build_method_report` beside the
  facts, since it belongs to the fit, not the method.

The layout the reporting code lands in:

| Path                                        | Holds                                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `cli/src/out/`                              | the stream layer: how anything reaches the terminal, and the running method's record       |
| `out/mod.rs`                                | `Out`, `LiveHandle`, `CommandError`, `ExitCode`                                            |
| `out/paint.rs`                              | `Style`, `Span`, `Line`, `Painter`, `Stream`; `style.rs` keeps only the `--help` colouring |
| `out/notice.rs`, `progress.rs`, `status.rs` | moved from `cli/src/`                                                                      |
| `cli/src/fit/report/`                       | the fit report: what a method run shows; never writes, returns `Line`s                     |
| `report/facts.rs`                           | `MethodFacts`, `RunHeader`, `Check`, `ChainWarmup`, `ChainDetail`                          |
| `report/build.rs`                           | `build_method_report` → `MethodOutcome`                                                    |
| `report/render.rs`                          | `render_report`, `View`                                                                    |
| `report/findings.rs`                        | the diagnostics block                                                                      |
| `report/export.rs`                          | `fit summary`'s md and LaTeX projections                                                   |
| `report/exact.rs`                           | the report as JSON with exact non-finite numbers, and `CAMDL_REPORT_CAPTURE`               |

`fit_summary.rs` ends as loading plus choosing a view, with no rendering of its
own. `DiagnosticKind`, `merge_key`, `render_entry` and `fmt_rate` stay in `sim`.
Files move in commits of moves only.

Steps 7 and 8 as landed:

- **Step 8 landed before the rest of step 7.** Plain progress lines are written
  whole through `out::LiveHandle`, not `log::info!`, so they carry no timestamp
  or module path, and `--progress` no longer raises the log filter (GH #14's
  option 2). The `output:` line is relative to the working directory when the
  fit lands under it, else absolute (`report::display_path`). PMMH's caveat
  carries `This series has T = n.`; because the series is known only once its
  data are bound, PMMH says its caveat at the start of `pmmh::run_method`, after
  the starts note rather than before it, and records `plan.series_len` for the
  loader's `Check::MethodCaveat`.
- **Step 7, PMMH's run end.** A chain that finishes with its scale at the floor
  says so at once
  (`chain k: proposal scale at its floor for n of m warm-up steps
  (details at the end)`),
  decided by the predicate that pushes the finding after the join; on a terminal
  the line prints above the bars. The acceptance section is one table in chain
  order with the scale and floor count beside each rate
  (`report::render::acceptance_table`), replacing the per-chain λ lines. The
  completion line drops the leaf path, which the `stored` line carries.
- **Step 7, `fit summary` from one report.** `fit summary`'s text loads each
  leaf into its `MethodReport` through `load_method_facts` and
  `build_method_report`, as json, md and latex already did, and draws it with
  `render_report(&MethodOutcome::Completed(report), View::Summary(..))`;
  `fit_summary.rs` keeps the loading and the choice of format, and no rendering.
  The md and LaTeX renderers moved to `report/export.rs`. Text output is
  byte-identical on every fixture; the facts fixture gained `fit summary` runs
  for PGAS (plain, and under `--explain --parameters`), IF2, nl-sbplx, pfilter,
  `--exclude-chains` and the all-chains-fail fit before the port, and lost
  nothing.
- **`View` has one arm, `Summary(SummaryView)`.** `SummaryView` carries
  `--explain` and whether the camdl version leads the method's identity line (a
  fit-level fact said once, on the first method), which a bare `Summary` arm had
  nowhere to hold. `RunEnd` lands with its renderer on the run side; an arm with
  no renderer would have had to panic.
- **`MethodOutcome::Failed` carries `error: String`**, since `CommandError`
  belongs to `Out`, and nothing constructs it yet: `fit run`'s failure path
  builds it from memory on the run side, so the variant carries an
  `expect(dead_code)` outside tests until then. Its summary rendering (the run's
  opening block, the findings, the error) is tested. `fit summary` of a failed
  leaf still refuses it as no completed method, as today. `build_method_report`
  still returns a `MethodReport`, which the JSON document holds;
  `MethodOutcome::Completed` wraps it.
- **`ChainDetail` is everything the summary's text reads beside the typed
  result**: the per-chain trace means and outlier scores, the degeneracy
  screens, the filter ESS, the latent paths, the chain counts and starts in
  `fit_state.toml`, the stored-fit source run, the forkable counts (counts, not
  the whole `JointEnsemble`) and the `--exclude-chains` subset. It is
  `MethodFacts::chain_detail: Option<_>`, filled only for the text view, and
  `MethodReport` carries it unserialized. The subset sits there, not in
  `MethodFacts` proper, because it is the summary's read-back and the run never
  fills it. A leaf that predates `latent_convergence.tsv` has it written by that
  read, never by the renderer.
- **`diagnostics.json` is read entry by entry for every format**, as the text's
  refused-chain reader already did: an entry that does not parse costs that
  entry and is named. The JSON report gains `findings_unreadable` (omitted when
  empty, schema version unchanged), so a consumer is told what was not read
  instead of a leaf failing to load. `ProvenanceReport` gains two unserialized
  flags saying which params file was absent, which the text names.

Step 7's run side, as landed:

- **`Out` is the stream layer and the running method's record.** It holds the
  two painters, the notices, and, from `begin_method` to `end_method`, the
  running method's opening block, the checks it showed, its collector and its
  heartbeat. The record names two fit types (`RunHeader`, `Check`), so `out/` is
  no longer free of the fit layer; the alternative, a second handle for the
  method's record beside `Out`, would have split one call site's state across
  two objects. `Out::error` marks the heartbeat failed, as the proposal says.
- **`ExitCode` is camdl's own `ExitCode(i32)`**, with `exit()`, since
  `std::process::ExitCode` cannot be handed to `std::process::exit` from deep in
  a command. `CommandError::message` is the whole first line as each site
  printed it (`error: …`, `error running pmmh: …`): the sites word their lead
  differently, and the type moves them without rewording them.
- **`Out::check` records; the check printers still write.** The noise check
  (`pf_noise::report`), the observation streams
  (`util::print_observations_summary`, shared with `pfilter`), the caveat banner
  and the starts note keep writing their own lines, and the run records each
  check on `Out` beside them. Their text depends on context the `Check` does not
  carry (the parameter count and series length the noise check reads, the
  scheduled-actions block printed between them), and `pfilter` prints the same
  observation block without `Out`; moving them is step 9's.
- **Progress stays with `out::progress::Reporter`/`Task`**, which own the bar,
  the plain line and the throttle and write through `LiveHandle`; an
  `Out::progress` beside them would have been a second answer to the same
  question. Chain workers reach the terminal through `out::live()`.
- **`Out::notice` and `Out::data`.** `data` carries `fit summary`'s json, md,
  latex and `--params-only` output, where a closed pipe now exits 0. `notice`
  lands as the foundation for step 9, with an `expect(dead_code)` until the
  commands that collect notices move onto `Out`.
- **The report `fit run` builds.** After `finalize`, `fit run` builds
  `MethodFacts` from the header, checks, findings and warm-up it holds, the wall
  time it measured, and the typed result `MethodResult::load_from` reads from
  the leaf. `ChainDetail` (the verdict's outlier and forkable counts) is
  gathered from the files the run wrote, by the function `fit summary` uses,
  rather than from memory with the writer's rounding applied: those files are
  the record the proposal names for these facts, and reading them is the
  rounding.
- **`View::RunEnd` draws** the verdict section `fit summary` draws, the R̂
  legend, the acceptance table, the R̂ table (`rhat_run_table`, from the report's
  `PosteriorDiagnostics`), the findings, and for PMMH and `mh` the completion
  and highest-posterior lines; `Out::milestone` prints `stored` after it. The
  completion line and PGAS's acceptance table needed records:
  `pmmh_summary.json`'s `cost` and `fit_state.toml`'s `best_chain` are now
  carried by `PmmhRun` (`cost`, `map_chain`), and `pgas_summary.json` gains
  `acceptance_chains` and `acceptance_kernel`, carried by `PgasRun`
  (`acceptance_per_chain`, `acceptance_kernel`). All are additive and omitted
  when absent; the schema version is unchanged.
- **Two sections move.** The runner writes `draws.tsv` and prints its line
  before the report exists, so `draws.tsv: N posterior samples` now precedes the
  verdict rather than following the R̂ table. PGAS's other run-end blocks (path
  renewal, filter ESS, latent paths, the cross-chain matrix, the completion
  line) are still printed by `pgas::run_method`, so its verdict, acceptance and
  R̂ tables follow them; step 9 moves PGAS's prints.
- **A frozen R̂ prints `Rhat=— ✗`.** The summary the table is drawn from records
  a non-finite R̂ as the refusal `NonFiniteRhat`, not as `+∞`, so the table no
  longer has the number it printed as `inf`; it keeps the worst band.
- **A failed run** builds `MethodOutcome::Failed` in `abandon_method`, which
  draws its findings (`View::RunEnd`) and reports the error through
  `Out::error`. The runners no longer print findings on their way out. `Failed`
  carries no `checks`: nothing reads them, since the run's end has shown them
  and `fit summary` does not render a failed leaf, which it still refuses as no
  completed method (the proposal does not ask it to render one). A failed run's
  hints are rendered with no completed-chain count, so the BadInit hint "Other
  chains in this run completed normally; treat the surviving chains as the
  result" no longer prints on a run that produced no result. The failed fixture
  lost its empty acceptance table and its R̂ table over zero chains.
- **The IF2 runner and NLopt keep a heartbeat-only `abandon_method_on`.** One
  IF2 site is a chain worker's thread, and neither runner holds `Out`.
- **IF2 printed `stored` twice**, once from its dispatch arm before the leaf was
  finalized; it prints it once, after `finalize`, with its `best ll=` hint under
  it.
- **The run-equals-loader test** is part of `reporting_facts`: both commands
  append the report they built to the file `CAMDL_REPORT_CAPTURE` names,
  serialized by `report::exact::exact_json`, which is the proposal's normalizer
  built into the serializer (a non-finite float becomes `"NaN"`, `"inf"` or
  `"-inf"` as it is written, since `serde_json` has already collapsed them to
  `null` by the time a value exists). It compares every completed fixture
  method; the failed fit has no report on either side. It found that the loader
  ordered PMMH's checks as the run did before step 8 (caveat before the starts
  note); the loader now follows the run. The variable is the test's seam until
  `events.jsonl` carries `method_report`.
- **Two small fixes rode with the step.** `fit summary`'s point-mass and `-inf`
  chain notes advised `--exclude-chains <method>:k`, which `fit summary`'s
  parser refuses; they now advise `--exclude-chains k`, the form the parser and
  the run spec document (the `@fit:ids` form is `compare`'s). And the IF2
  section and its refusals no longer speak of a "scout": the gate is the
  "compound convergence gate".
- **What of `fit run` is on `Out` and what is not.** The method's record, its
  end-of-run block, its `stored` milestone, and every exit inside the method's
  dispatch (`abandon_method`, the finalize error) go through `Out`. The exits
  before a method starts (config resolution, the cache-hit and label checks,
  about a hundred `eprintln!` and `process::exit` pairs) and the runners' own
  progress and preflight prints do not yet; they move with step 9's no-print
  test, which is what makes "commands print through nothing else" checkable.

Step 9's library half, as landed:

- **PGAS's chain events are values handed over live, not returned.** `run_pgas`
  takes an `on_event` callback and hands it a `PgasEvent` for each thing it used
  to print (the reference trajectory's draw and density check, the starting
  log-likelihood, NUTS's mass matrix and step size, the end of burn-in, a
  recovered start, and CSMC's degenerate-ancestor and weight-collapse readings).
  The CLI prints each word for word with `chain k:` on its first line, through
  the chain's progress task. They are not fields of `PGASResult` because they
  arise during runs that can last hours, and a chain refused at its start has no
  result to carry them. As for PMMH, the library's resume line goes and the
  already-completed warning moves to the CLI's resume loader.
- **PGAS's run end is drawn from the report, in PMMH's order**: the verdict, the
  R̂ legend, the acceptance and R̂ tables, then the path-renewal, filter-ESS,
  latent-path and cross-chain blocks, the findings, and the completion line
  (without the leaf path, as PMMH's) and the best sweep's log-likelihood. Each
  block is read back from the record the run wrote, by `ChainDetail::gather`;
  `pgas_summary.json` gains `cost.sampling_seconds` and `PgasRun` carries it
  with `fit_state.toml`'s `best_chain` and `best_loglik`. Reading back exposed
  that `serde_json`'s default float parser is not exact: it reads
  `0.9724999999999999` as `0.9725`, so the fixture's renewal prefix, a tie at
  the third decimal, now prints `0.973` where the run computed `0.972`
  (facts.tsv edited). Enabling `serde_json`'s `float_roundtrip` feature would
  fix it, but features unify across the workspace, so it would also change how
  every other JSON file parses, the model IR among them; it is left as its own
  issue. A cross-chain number the file cannot hold (non-finite, which JSON
  writes as `null`) prints `—`.
- **What a run meets deep in the engine is recorded once, in `sim::run_notes`,
  and said by the command.** The effect-snapped note, the non-integer Normal
  observation, a failed Cholesky factorisation and the gamma-index divergence
  detector arise where nothing can be returned (inside a likelihood evaluation
  repeated thousands of times, or a plan built once per evaluation), so the
  engine records a `RunNote`; its first occurrence in the process goes to the
  reporter `main` installs, which prints it as a warning notice, and repeats are
  dropped as before. `cholesky_lower` returns `Result<_, NotPositiveDefinite>`,
  so the condition is a value its callers hold. The sentences are unchanged;
  they lose `env_logger`'s timestamp and module path.
- **A negative balance compartment is counted, not logged.** It fired once per
  particle per substep under inference; it is now `balance_negative` in the
  eval-stats summary the command prints at its end.
- **`eval_stats::report_if_nonzero` became `nonzero_since`**, which returns the
  counters that moved; the CLI's `EvalStatsReportGuard` prints them.
- **`restore_z_values` is silent**; `z_restore_gaps` says what it will
  recompute, and the PGAS and PMMH resume loaders print it, word for word, under
  each chain's resume line.
- **`warn_zero_firings` became `never_fired`**, and `simulate` reports the
  transitions as one warning `Notice` whose message keeps today's lines.
- **The no-print test is `sim/tests/sim_prints_nothing.rs`.** It blanks comments
  and every `#[cfg(test)]` item before searching, so test modules and test-only
  helpers may print; `src/trace.rs` is the one exempt module.
- **What of PGAS still prints from the CLI.** `pgas::run_method`'s own lines
  (the priors, the sampling plan, `chain k done`, the live `BadInit` line, the
  `draws.tsv` line) and the chain-event renderer write directly; they move onto
  `Out` with the CLI's no-print test.

Step 9's command half, as landed:

- **Every command writes through one `Out`.** `pfilter`, `profile`, `survey`,
  `compare`, `fit predict`, `simulate`, the four `lineage` verbs, `list`,
  `show`, `cat`, `fit diff`, `fit table`, `fit run` before its method starts,
  the PMMH, PGAS, IF2 and NLopt runners, and the remaining commands (`docs`,
  `data split`, `dev eval`, `mre`, `check-update`, the removed `if2`, `fit new`,
  `label`, `fit methods`, `dev reindex`, and the camdlc-delegated `check`,
  `inspect`, `render`, `dev compile`, `dev doctest`) create one at their entry
  point and pass it to the helpers that print. Each command's text, streams and
  exit codes are unchanged; the evidence per commit is a 55-invocation snapshot
  of the reporting fixture, byte-identical on both streams with and without
  `CLICOLOR_FORCE` after masking hashes and timings, and, for the paths it does
  not reach, each command's refusals run against the previous binary.
- **`Out` gained what the sites needed, and nothing speculative.** `say` and
  `say_raw` (a stderr line, after flushing stdout and under the live lock, so a
  chain worker's line never splits one), `fail` (`error` then `exit 1`, the pair
  every error site was written as), `hint`, `report_line` and `report_raw` (a
  report command's report as text already painted for stdout), `data_line` and
  `data_raw`, and `stdout_painter`. `error` takes `&self`: reporting an error
  changes nothing `Out` holds, so a closure passed to `unwrap_or_else` can
  borrow the handle beside the command's other uses. The proposal's
  `report(&[Line])` and `data()` stay; most reports are still built as strings,
  which `report_line` carries without rebuilding them as spans.
- **`out::detached()` is an `Out` for code below every command's handle.** It
  writes what the command's own `Out` would. It serves helpers that warn or exit
  where they stand and are reached from several commands or off the command's
  thread: the camdlc handshake and compile milestones in `util`, the observation
  projection's invariant exits, `FitRunConfig::build` and `build_obs_model`, the
  run-store readers, `abandon_method_on` (IF2 and NLopt hold the heartbeat, not
  the handle), simulate's stream sink, the engine's grid notices, and the unit
  tests that print measurements. Threading the handle through the engine's
  `RunSink` trait and every method's setup would have touched six modules for
  messages that are invariant failures or one-off warnings. A function that can
  be handed the command's `Out` takes it.
- **`Out::notice` writes and does not keep.** The run's record of its notices is
  `run.json`'s display inputs, which `cas_sink` builds from the same resolution
  (`params_resolver::notices`, `obs_anchor::spline_hold_notices`), so the list
  step 7 kept on `Out` had no reader and went with its `expect(dead_code)`.
  `params_resolver::emit_notices`, `obs_anchor`'s resolvers and anchor report,
  profile and fit predict hand their notices to the command's handle;
  `Notices::emit_stderr` is gone.
- **`Out::check` still only records.** The check printers write through `Out`
  now (`util::print_observations_summary` and `print_scheduled_actions_summary`
  take it, the noise check's `pf_noise::report` and the dt-convergence ladder
  are rendered to text and written with `say_raw`), but each check's text is
  still drawn where its context is (the parameter count and series length the
  noise check reads, the scheduled-actions block between the checks), which the
  `Check` value does not carry.
- **Chain workers write through the same `&Out`.** `Out` is `Sync`, and every
  write takes the live lock, so the IF2 chain loop's `chain k: ✗ skipped` line
  is whole beside the progress lines without a separate `LiveHandle` call.
  PGAS's chain-event renderer writes through its progress task's `live_line`,
  which is the stream layer's own.
- **A closed stdout is a normal exit** for every command, as "Ordering" sets:
  `camdl list --format json | head -1` panicked (status 101) and now exits 0;
  `simulate --stdout` into a closed pipe printed an error and exited 1, and now
  exits 0.
- **PGAS's own lines moved.** `pgas::run_method`'s priors, plan, `chain k done`
  and `draws.tsv` lines and the live `BadInit` line go through `Out` like every
  other method's; simulate's never-fired-transition notice goes through
  `out::detached().notice` from inside its stream sink.
- **The cli no-print test is `cli/tests/no_print_outside_out.rs`.** It fails on
  `println!`, `print!`, `eprintln!`, `eprint!`, `dbg!`, `io::stdout()` and
  `io::stderr()` anywhere under `cli/src/` outside `out/`, test modules included
  (stricter than `sim`'s, which blanks them); `style.rs` needs no entry, since
  clap prints the help it colours. Two `process::exit` calls stay outside `out/`
  by design: the hidden `--check-args` mode's status and camdlc's own status
  when camdl hands it a command; neither prints.
- **The facts fixture covers the other commands** (pfilter's replicates and
  files, profile and survey fresh and cached, simulate's four shapes and event
  log, the lineage verbs, list, show, cat, fit diff, fit table, and one refusal
  each) and runs every command with its own `CAMDL_IR_CACHE_DIR`, so whether a
  run prints `compiling` or `cached IR` no longer depends on `~/.cache`.
