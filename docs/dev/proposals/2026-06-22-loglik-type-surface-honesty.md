---
date: 2026-06-22
status: implemented (gh#280) — core Changes 1–3 + Amendment A2.2/A3/A5; A1/A2.1/A4 deferred
related:
  - 2026-06-20-model-criticism-outputs.md # prequential elpd is the comparison-correct quantity
area: inference output / progress feed / machine-readable artifacts / fit summary + table
issue: gh#280
---

# Carry the log-likelihood's type onto every surface that shows or scrapes it

## Problem

A log-likelihood reported by camdl is one of two **classes**:

- **marginal** `log p(y | θ)` — a function of θ alone, the quantity model/chain
  comparison needs. Produced by IF2 (a clean high-particle PF re-eval at θ̂,
  `loglik_eval.rs:1`), PMMH (`map_loglik`), and NLopt/ODE.
- **complete-data (joint)** `log p(y, x | θ) = transition_ll + obs_ll` — a
  function of the _sampled trajectory_ `x`, not θ alone. Produced by PGAS, whose
  Gibbs target is `(θ, x)`. It is **not** comparable to a marginal and is
  gameable: a chain can raise it by finding a smoother trajectory
  (`transition_ll ↑`) at the cost of data fit (`obs_ll ↓`).

camdl tags this. `FitState.loglik_type` records the kind — `"complete_data"`
(PGAS), `"marginal"` (PMMH), `"if2"` (IF2), `"ode_marginal"` (NLopt)
(`fit/state.rs:29`) — and the PGAS trace column is named `log_complete_data_ll`
with a test forbidding a bare `log_likelihood` "mistaken for the marginal"
(`tests/pgas_resume.rs:246`). Good hygiene where it exists.

The tag is honored on the two surfaces the team hardened (the trace column;
`camdl compare`, which ranks by Δelpd, `compare.rs:234`). It is **dropped
everywhere else** — and the highest-stakes "everywhere else" is the set of
machine-readable artifacts an agent scrapes to make a decision:

1. **`loglik_type` rides in no JSON/TSV artifact at all.** It is serialized only
   into `fit_state.toml` (via `FitState`). Every machine-readable loglik is
   emitted bare: `run.json` for survey (`survey.rs:570`), profile
   (`profile.rs:1639`), pfilter (`pfilter.rs:498`); `browse --format json`
   (`browse.rs:1414`); the `StageReport` JSON struct (`fit_summary.rs:1008`,
   `pub best_loglik` with no type field); the survey landscape TSV
   (`survey.rs:1119`). An agent reading any of these cannot tell a joint value
   from a marginal one.
2. **The live progress feed strips the type.** PGAS feeds its complete-data
   value through `progress::ll(x)` → `format!("ll={:.1}", x)`
   (`progress.rs:286`, `fit/pgas.rs:649`) — the same helper that carries
   _marginals_ for IF2/PMMH — so the live `ll=` means joint for PGAS and
   marginal for everyone else. This is the bare form the trace test exists to
   prevent, leaking into the feed a human or agent watches mid-run.
3. **Human display headlines are untyped** — the IF2 (`fit_summary.rs:470`) and
   NLopt (`:363`) stage headlines, the markdown/LaTeX exports (`:1465`,
   `:1618`), and the `fit table` column (`fit_table.rs:375`) print the number
   with no type.

### What is _not_ a problem (corrected scope)

The joint value never reaches a comparison surface, because PGAS reports no
scalar `best_loglik`: `MethodView::from_pgas` sets `best_loglik: None`
(`table_row.rs:300`), so PGAS is excluded from `fit table`'s max/delta
(`fit_table.rs:108`), and `bayesian_block` prints no loglik headline for PGAS
(`fit_summary.rs:759`). So there is **no** "complete-data differenced against
marginal" misrank in `fit table` or the summary delta — the only values that get
differenced there are the three **marginal-class** types, which are
commensurable in kind. This RFC therefore does **not** touch the delta/ranking
logic; the fix is to _carry the type_, not to regroup comparisons. (The subtler
question of comparing marginals _across backends_ — ODE-deterministic vs
chain-binomial-PF — is real but out of scope; it is a backend caveat, not a
type-class error.)

## The invariant

Every surface that shows or serializes a log-likelihood **carries its type**, so
no human or agent reads a bare number whose class (joint vs marginal) they
cannot recover. We change _labels and a serialized field_, not what any method
computes or how anything is ranked.

## Design — one typed source, read everywhere

The root cause is that `loglik_type` is a free `String` set independently at
each method's result site (`pgas.rs:940` `"complete_data"`, `pmmh.rs:945`
`"marginal"`, `gating.rs:347`/`method_result.rs:785` `"if2"`,
`nlopt_stage.rs:281` `"ode_marginal"`) and then dropped on the way to display
and serialization. Consolidate onto a single typed source:

```rust
// fit/state.rs (or a small fit/loglik.rs)
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoglikType { If2, Marginal, OdeMarginal, CompleteData }

impl LoglikType {
    pub fn tag(self) -> &'static str { /* "if2" | "marginal" | … */ }
    /// The comparison class. Only `CompleteData` is non-marginal.
    pub fn is_marginal(self) -> bool { !matches!(self, LoglikType::CompleteData) }
    /// Progress-feed metric prefix: `cdll=` for the joint, `ll=` for a marginal.
    pub fn metric_prefix(self) -> &'static str {
        if self.is_marginal() { "ll" } else { "cdll" }
    }
}
```

- **Derive it once from the method**, not by hand at each site: a
  `MethodResult → LoglikType` mapping that `FitState`, `MethodView`, the
  progress call, and the artifact writers all read. This kills the
  stringly-typed fork (the type is currently spelled in ≥4 places and absent
  from `MethodView` entirely — `fit table` would otherwise need a _fifth_
  hand-written copy).
- `FitState.loglik_type` becomes `Option<LoglikType>` (legacy/None tolerated).

### Change 1 — typed progress label

Route the PGAS per-sweep metric through the type:

```rust
// fit/pgas.rs:649 — was progress::ll(result.log_complete_data_ll)
task.set(crate::progress::ll_kind(result.log_complete_data_ll, LoglikType::CompleteData));
```

`progress::ll_kind(x, kind)` formats `{prefix}={:.1}` using
`kind.metric_prefix()` (`cdll=` for PGAS, `ll=` for the marginal callers). The
existing `ll`/`mcmc` helpers become thin marginal-default wrappers so the
IF2/PMMH/survey sites are untouched. This is the live-feed analogue of the
trace-column guard, driven by the same enum rather than a one-off `cdll`
formatter (avoids a 5th label helper alongside `best_ll`/`ll`/`mcmc`).

_Note:_ the type tag must render **after** the number on every line — a survey
log scraper (`tests/fit_survey_denominator.rs:161`) reads `loglik=` and stops at
the first non-numeric char, so a trailing `(marginal)` is safe and a leading one
is not.

### Change 2 — emit `loglik_type` in every machine-readable artifact (the core fix)

Add the tag next to each serialized loglik, so an agent can tell the class:

- `StageReport` JSON: add `loglik_type` beside `best_loglik`
  (`fit_summary.rs:1008`).
- `run.json` inputs for survey / profile / pfilter (`survey.rs:570`,
  `profile.rs:1639`, `pfilter.rs:498`).
- `browse --format json` per-stage object (`browse.rs:1414`).
- survey landscape TSV: a `loglik_type` column (`survey.rs:1119`).
- `fit table` JSON `TableRow` + an **appended** CSV column (`fit_table.rs:375`).
  (`TableRow` is shared by `fit table` and `fit summary --format json`, so this
  lands on both consistently — one field, both surfaces. CSV is positional:
  append, never insert.)

All read `LoglikType` from the single derived source, so the tag can never
disagree with `fit_state.toml`.

### Change 3 — label human-facing headlines (clarity, not correctness)

Append `({loglik_type})` to the IF2 / NLopt / markdown / LaTeX headlines
(`fit_summary.rs:363,470,1465,1618`) and the `fit table` display column, so the
number is self-describing. These are all marginal-class today (PGAS shows no
headline), so this is clarity, not a misrank fix — but it closes the "bare
number" reading and is near-free once the enum exists.

### Optionally surfaced — elpd in summary when present

When a fit has a PFilter stage that wrote `prequential.json`, show its elpd in
`fit summary` (reusing `compare.rs`'s reader, made `pub(crate)`; the file lives
in the PFilter _stage_ leaf, so this needs stage discovery, not a bare
`load_trace(dir)`). **Caveat, stated plainly:** a default PGAS fit has no
PFilter stage, so no `prequential.json` — this does _not_ give PGAS a comparable
number. It only surfaces elpd for pipelines that already opted into a PFilter
pass. Making PGAS itself report a marginal is a separate, heavier change
(below).

## Out of scope (separate, heavier follow-ups)

- **A PF-marginal for PGAS by default** — an extra particle-filter pass at the
  posterior so PGAS reports a comparable `log p(y|θ)` like PMMH. The right
  eventual ergonomic, but it is new computation, not a relabel; its own RFC.
- **Cross-backend marginal comparability** (ODE vs chain-binomial PF) — a real
  caveat for `fit table` / `compare`, but a distinct problem from carrying the
  type.

## Test plan (honest red → green)

- `progress`: unit test `ll_kind(x, CompleteData)` → `cdll=…`,
  `ll_kind(x, Marginal)` → `ll=…`, with `-inf` handling; a PGAS-feed assertion
  that the plain line carries `cdll=`, not bare `ll=`.
- **Artifacts**: a PGAS fit's `fit_state.toml`-derived `StageReport`/`run.json`
  (where applicable) and an IF2/PMMH fit's `run.json` / survey TSV each carry
  `loglik_type`. These fail on current code (the tag is in _no_ JSON/TSV today)
  and pass after — the load-bearing red→green.
- Headlines: IF2 summary headline carries `(marginal)`/`(if2)`, NLopt carries
  `(ode_marginal)`. (No "PGAS headline carries complete_data" test — PGAS shows
  no headline loglik; asserting one would be a test that cannot pass.)
- `fit table`: CSV gains an appended `loglik_type` column; a single-method
  cohort is byte-stable except for the new column (guards against the column
  change perturbing deltas).

## Open questions

1. **Enum home** — `fit/state.rs` vs a small `fit/loglik.rs` that both `state`
   and `MethodView` import (avoids `table_row` depending on `state`).
2. **Display vocabulary** — `cdll=` for the joint is the recommendation; confirm
   it reads clearly against `ll=`. A one-line legend in `--help`/docs may be
   worth it once two prefixes exist.
3. **Markdown/LaTeX exports** — label inline (`… (marginal)`) or as a separate
   "likelihood type" row? Inline is lighter.
4. **Legacy `loglik_type = None`** — render as `unknown` in artifacts and
   headlines; never inferred.

## Amendment A — machine-consumed surfaces and comparability (second round)

A review from the interactive-watcher-UX side found that the core change, as
above, hardens the human text feed and headlines but leaves the two surfaces a
_machine_ actually consumes under-typed. Fold these in as a second round, after
the core change lands. Each is verified against code.

### A1 — type the structured progress feed, and disambiguate which `progress.rs`

There are **two** `progress.rs`: `cli/src/progress.rs` (the human/log text feed,
`ll()` at `:286`) and `io/src/progress.rs` (the gh#278 heartbeat written to
`progress.json`, which a dashboard/agent parses for liveness). The core change's
Change 1 cites "`progress.rs:286`" unqualified — it means **`cli::progress`**.
State that, so an implementer doesn't harden the wrong file.

The heartbeat (`io::progress`) carries
`RunState::Running { phase, step, total }` and **no loglik**
(`io/src/progress.rs:87`). So the one _structured_ live feed a machine consumes
has no loglik to type — the proposal's "machine honesty" framing skips it.
Resolve explicitly, one of:

- **(recommended) add a typed loglik to the heartbeat.** Extend
  `RunState::Running` with an optional
  `loglik: Option<{ value: f64, kind:
  LoglikType }>`, written best-effort on
  the same monotonic-bump path. This is the structured twin of the `cdll=` text
  line and is what the watcher actually reads. Liveness stays the heartbeat's
  primary job; the loglik is advisory.
- **or narrow the framing** — state that the heartbeat is liveness-only and the
  typed-loglik artifacts are run.json / StageReport / survey TSV / fit table.

Do not leave it implicit. Adding the field is a deliberate scope expansion of a
liveness file; the alternative is honest scoping. (Maintainer decides.)

### A2 — type is necessary but not sufficient: carry `backend` for comparability

`LoglikType::is_marginal()` answers "same _kind_," not "safe to subtract." Two
marginals from different **process models** — ODE-deterministic vs
chain-binomial-PF — are not on the same scale, so a consumer reading
`is_marginal() == true` as "subtractable" reproduces the bare-number misread one
level up. The artifacts do not currently co-locate the backend with the loglik:
the `fit table` CSV header carries `method` but **not `backend`**
(`fit_table.rs:375`), and `method` is the _algorithm_ (if2/pgas/pmmh), not the
process model that governs comparability.

- **Emit `backend` next to `loglik_type`** in every artifact Change 2 touches,
  so a consumer can gate on `(loglik_type, backend)`.
- **Do not expose a bare `is_marginal()` boolean** that reads as "safe to
  compare." If a helper is wanted, name it for what it is (`comparison_class()`
  returning the class), and document that comparability also requires equal
  backend. Cross-backend marginal comparison stays the consumer's explicit
  decision, not an implied one.

### A3 — reserve a slot for the observation-conditional likelihood

`obs_ll = log p(y | x, θ)` — the data-fit half of the complete-data density — is
already a per-sweep PGAS trace column (`fit/pgas.rs:565`), and it is what
diagnostics actually want. It is **neither** marginal nor complete-data. If/when
it is surfaced as a typed metric (the watcher intends to), it gets a distinct
`LoglikType::ObsConditional`, not a fifth ad-hoc string. Design the enum so
`is_marginal()`/`comparison_class()` treats `ObsConditional` as its own
non-marginal class. Add the variant **when obs_ll is first surfaced**, not
speculatively (no producer/consumer ⇒ no variant yet).

### A4 — legacy traces violate the invariant the proposal scopes out

The core change calls the trace column "already clean" and scopes it out. That
holds only **after** `587df38d` (the column rename). Legacy PGAS traces on disk
predate it: a sampled store showed **54 of 155** PGAS traces carrying a bare
`log_likelihood` column — complete-data under the marginal's name — with **no
`loglik_type` in trace metadata**. So the invariant ("no bare number whose class
you can't recover") is violated for legacy data, precisely on the surface the
proposal declared clean.

- **Do not migrate the 54 files** — alpha; backwards-compat is a non-goal.
- **Correct the claim**: the trace is clean for new runs only.
- **Robust consumer key**: prefer `obs_ll` (present in _every_ PGAS trace) over
  `log_likelihood` for column-name-keyed readers.
- **Optional, forward-looking**: write `loglik_type` into _new_ trace metadata
  (a header/sidecar) so column-name-keying stops being load-bearing.

### A5 — label legibility and the live-feed contract change

- `cdll=` is machine-honest but human-opaque. Prefer `ll(joint)=` (or keep
  `cdll=` plus a one-line legend in `--help`). Resolves open question 2 toward
  the more legible form.
- `ll=` → (`cdll=`/`ll(joint)=`) for PGAS is a **deliberate contract change** to
  the live-feed key: anything grepping `ll=` on a PGAS run was already reading
  the non-comparable joint, so the break is the fix. State it as a contract
  change, not a cosmetic tweak.

## Amendment A — resolution (gh#280 implementation)

- **A2.2 — folded in.** `is_marginal()` documents that it answers _kind_, not
  comparability (equal backend also required); no `comparison_class()` helper
  was added — there is no consumer for one yet, and a speculative public API
  would be dead code.
- **A3 — folded in (design only).** `is_marginal()` is now defined by inclusion
  (`If2 | Marginal | OdeMarginal`), so a future non-marginal kind defaults to
  non-marginal instead of silently joining the marginals. The `ObsConditional`
  variant is **not** added — there is no producer/consumer yet, per A3's own
  instruction.
- **A5 — folded in, with a wording correction.** The PGAS live-feed prefix is
  `ll(complete)=` (not `cdll=`, and not A5's suggested `ll(joint)=`). "joint" is
  ambiguous — joint over _what_? — and a reader can misread it as the joint over
  the observation vector, i.e. the marginal, inverting the meaning. "complete"
  echoes the codebase's established term (`log_complete_data_ll` column,
  `complete_data` tag), is legible without a legend, and is still a distinct key
  from `ll=`, so a marginal scraper never picks it up.
- **A1 — deferred (narrow framing).** The `io::progress` heartbeat stays
  liveness-only. `io` is a dependency of `cli`, so it cannot reference
  `cli::fit::loglik::LoglikType`; a typed heartbeat loglik would force
  relocating the enum into `io` (a layering inversion) or re-stringifying it
  (the fork this RFC removes). The typed loglik lives in the result artifacts
  (run.json / StageReport / survey TSV / fit table). A typed heartbeat field is
  a clean follow-up if a live watcher needs it.
- **A2.1 — deferred.** Co-locating `backend` is a distinct comparability concern
  the core RFC already scopes out, and `backend` is not in scope at
  `StageReport` / survey / profile / pfilter `run.json` without threading it
  through. A focused "carry backend for comparability" change is the right home
  (`browse --format json` already emits `backend`).
- **A4 — deferred.** The trace surface is untouched here; writing `loglik_type`
  into _new_ trace metadata is a separate `trace_writer` change. The claim is
  corrected: the PGAS trace column is clean for **new** runs only (post
  `587df38d`); legacy traces on disk are not migrated (alpha).
