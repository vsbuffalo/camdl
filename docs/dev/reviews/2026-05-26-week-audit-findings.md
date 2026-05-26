---
status: open
date: 2026-05-26
scope: last-week diff — six thematic clusters (lineage, typed-time, events/sim, profile-inference, prior precedence, batch from_csv)
reviewer: internal (HEAD = d3ebe965, branch features/from-csv-and-forcing-discoverability)
triggered-by: code review per docs/dev/code-review.md
counts: 7 Critical / 19 High / 22 Medium / 9 Low
surface-map: 2026-05-26-week-audit-summary.md
---

# Week-of-2026-05-19 → 2026-05-26 audit — findings

Driven from `docs/dev/code-review.md`. Surface map in
`2026-05-26-week-audit-summary.md`; six parallel reviewers (lineage,
typed-time, events/sim, profile-inference, prior precedence, batch
`from_csv`) plus a cross-cutting sweep.

Findings are severity-ordered (Critical → High → Medium → Low), then
by likelihood of triggering. Each finding cites file:line and includes
the command output that demonstrates the defect, per the CLAUDE.md
"paste-the-receipt" rule. Cross-cluster correlations are noted.

**Headline.** Three distinct Critical correctness items, one Critical
retrospective acknowledgement, and one Critical test-coverage gap.
The most common pattern this week is **silent-numeric-bug**: a
construct that compiles and runs but produces a wrong number — most
acutely in `survey_top_k` (independently flagged by two reviewers),
the Gillespie inhomogeneous-Poisson sampler, and the OCaml↔Rust date
parser disagreement.

For wiring/style smells the surface holds up well: IR version
atomicity is enforced by `include_str!("../../../../ir/VERSION")`,
all 22 new CLI fields are wired to use-sites, no `unwrap`/`expect`
appears in non-test code in the new modules, and no `Hash`/`Eq`
derives on float-bearing types. The `#[lineage]` linearity check
(`ocaml/lib/ir/lineage.ml`) correctly handles the `β·S·I/N`
normalizer exemption.

---

## CRITICAL

### C1 — `survey_top_k` ranks by likelihood, not posterior; PGAS/PMMH chains target the posterior

*Independently flagged by the prior-precedence and profile-inference
reviewers (E-F2, D-F2).*

- **Location:** `rust/crates/cli/src/fit/init.rs:521–526`;
  `rust/crates/cli/src/survey.rs:565–571,1099–1115` (writer emits
  only `loglik`).
- **Category:** statistical correctness (§4); user footgun (§6).
- **Defect:** `build_chain_starts_from_survey` sorts survey rows by
  `loglik` descending and seeds K chains at the top-K. PGAS and
  PMMH then target `p(θ|y) ∝ likelihood × prior`. For any
  non-flat prior the init points are systematically biased toward
  likelihood maxima irrespective of prior mass. The survey writer
  never computes `log_prior`, so even an opt-in posterior-ranked
  variant is impossible at the consumer.
- **Why it matters:** The whole reason to go Bayesian on cVDPV2
  decision support is that the prior encodes structural epi
  knowledge (R₀ band, generation interval, vaccine efficacy) that
  the data alone doesn't constrain. Seeding from likelihood-only
  ranks defeats that, and the bias is silent: no warning surfaces
  that the init distribution differs from the chain target. A user
  who runs `camdl survey` then `camdl fit run --init_method
  survey_top_k` expecting "good starts for my posterior" gets
  MLE-seeded starts that may sit in a region the prior excludes;
  chains then waste burn-in walking back into the prior's support
  — or fail to mix when the surveyed MLE region sits in a tail
  of the prior.
- **Fix:** Two-step.
  (a) Survey writer must emit `log_prior` (and/or
  `log_posterior`) alongside `loglik`; the `[estimate]` priors are
  already in scope. Document the new column.
  (b) `build_chain_starts_from_survey` ranks by `log_posterior =
  loglik + log_prior` when the survey/fit prior set agrees
  (validate via a `prior_hash` analogous to the existing
  `model_hash` cross-check). If priors disagree, refuse the init
  method with a named error. *Fallback if (a)+(b) is too
  ambitious for v2:* refuse `init_method = survey_top_k` whenever
  any estimated parameter has a non-flat prior at fit time; that
  refusal is correct by construction, flat-prior runs still
  benefit.
- **Severity:** **Critical**.

---

### C2 — Profile-PMMH was silently MLE-with-flat-priors before 2026-05-24 (`5f658a16`); no incident report, no changelog

- **Location:** pre-`5f658a16` `profile.rs:1013–1021` hard-coded
  `Prior::Flat` for every estimated parameter on the `--algorithm
  pmmh` path regardless of model IR / `--fit`.
- **Category:** statistical correctness (retrospective).
- **Defect:** Every "Bayesian" profile-PMMH posterior produced
  before 5f658a16 silently targeted the unconditioned likelihood
  (scaled-likelihood). The commit message itself acknowledges this
  ("Net effect: PMMH-via-profile was silently MLE-with-flat-priors")
  and gives a concrete reproduction in the camdl-book seed-timing
  chapter (`t_rep = −40` at a `Normal(4, 5)` prior; `n_seed = 1000`
  pinned at bound).
- **Why it matters:** Any saved profile run from before 2026-05-24
  that the user is interpreting as a posterior sweep is wrong.
  With camdl now public alpha (per `9481135b`, the alpha-announce
  commit) and external consumers possibly running gh#73-affected
  versions, a single commit-message acknowledgement does not
  reach the right audience.
- **Fix:** Add
  `docs/dev/incidents/2026-05-24-profile-pmmh-flat-priors.md`
  with (i) the seed-timing reproduction, (ii) the affected version
  range (whenever profile-PMMH first shipped → `5f658a16^`),
  (iii) the user remediation: "any saved `profile.tsv` / `run.json`
  from that range is a scaled-likelihood profile, not a posterior;
  re-run with the fix." Add a one-line note to the release-notes
  /changelog if one exists. Per CLAUDE.md "Incident reports
  require a reproduction" — the commit body has one; lift it.
- **Severity:** **Critical** (retrospective).

---

### C3 — Gillespie inhomogeneous-Poisson sampling is still wrong; the bare-`t` fix (`424b6a9a`) only narrows the failure mode

- **Location:** `rust/crates/sim/src/gillespie.rs:177–225`;
  classifier at `compiled_model.rs:170–189`
  (`expr_is_time_dependent`). Incident:
  `docs/dev/incidents/2026-05-20-gillespie-bare-time-frozen-propensity.md:75–83`
  ("piecewise-constant approximation on the output grid").
- **Category:** numerical / statistical correctness (§4).
- **Defect:** Commit `424b6a9a` recognizes bare-`t` rates and
  recomputes them *at output and intervention boundaries*. But the
  next-event draw at line 180 still uses
  `dt = −ln(u₁)/λ_total` with `λ_total` *frozen* at the value at
  `t`. For an inhomogeneous Poisson process this is biased — the
  correct sampler is thinning with a per-segment upper bound, or
  piecewise re-draws on a fine grid. The TODO at
  `gillespie.rs:195` ("PDMP thinning for real compartments")
  acknowledges this; the incident report is candid about the
  residual.
- **Why it matters:** The seed-timing chapter shipped on the back
  of this fix; the test only checks that Gillespie / chain-binomial
  agree to *within 30%* at `τ=30`
  (`rust/crates/cli/tests/seed_timing_e2e.rs:142–143`). For polio
  cVDPV2 seed-timing inference, 30% bias in inflow propagates
  directly to a τ-posterior shift. The chain-binomial backend
  re-evaluates every substep; users who pick Gillespie because it
  is "exact" get *biased* dynamics under any time-varying rate
  (seasonal forcing, importation pulses, vaccination ramps,
  time-varying contact).
- **Fix:** Three parts.
  (i) Document the residual prominently in the backend-picker UX
  — not only in the incident.
  (ii) Hard-error (or W-warn) when `time_dep_transitions` is
  non-empty, backend is gillespie, *and* the output grid is
  coarser than e.g. `generation_interval / 10`.
  (iii) Schedule the thinning / modified-next-reaction fix
  tracked in the `TODO(v0.2)`. Today Gillespie silently produces
  wrong numbers for the most common forced/seeded model.
- **Severity:** **Critical**.

---

### C4 — gh#69 (`07394aff`) parametric `at [param]` schedules — no committed regression test; the red→green proof is in the commit message only

- **Location:** OCaml: no test exercises `at_times_expr`. Rust:
  no test passes a parametric `at [param]` schedule end-to-end.
  Verified by
  `git grep -l 'AtTimesExpr\|at_times_expr' ocaml/test rust/crates/*/tests`
  → nothing.
- **Category:** tests (§9); TDD discipline (CLAUDE.md "Fix bugs
  via TDD").
- **Defect:** The commit reports "End-to-end verification (paste
  from a fresh run on `main` after this commit)" with manual
  numbers for `t_seed ∈ {−25, −12, −5, 5, 10, 15}`. No test
  compiles a model with `add(I, n) at [t_seed]`, runs it under
  each backend, and asserts the event fires at `t_seed` rather
  than `t=0`. CLAUDE.md is explicit: "paste the red-then-green
  test output in the commit message *as the proof*"; manual
  verification is not a regression net.
- **Why it matters:** The pre-fix bug was *silent* — the OCaml
  expander substituted `0.0` for any non-constant `at [...]`
  expression. Without a regression test, the same shape of bug
  re-lands at the next refactor of `expand_scheduled_actions`,
  and the failure mode is again silent. This is exactly the
  pattern that produced the Gillespie bare-`t` bug in the first
  place ("no committed model used bare `t` in a rate, so no test
  exercised the gap").
- **Fix:** Three tests. (i) OCaml `test_compiler.ml`: compile
  `events { booster : add(I, 1) at [t_seed] }`, assert IR is
  `AtTimesExpr` and contains `Param "t_seed"`. (ii) Rust unit
  in `rust/crates/sim/tests/`: load such a model,
  `resolve_fire_times(&[10.0])` for `t_seed=10`, assert
  `vec![10.0]`. (iii) Rust e2e via CLI like
  `events_backend_parity.rs`: pass `--param t_seed=5` vs
  `--param t_seed=10`, assert `I` jumps at different times.
- **Severity:** **Critical** (no regression net for a silent-bug
  class).

---

### C5 — Profile-PMMH reports a `loglik` that may not match the saved `mle.toml` parameters

- **Location:** `rust/crates/cli/src/profile.rs:1389–1406`.
- **Category:** numerical correctness; not-wired-through. Same
  class as `f52d1ecd` (the IF2 fix).
- **Defect:** The PMMH per-cell branch sets
  `mle_params = result.map_params` (the MAP θ) but reports
  `final_ll = result.map_loglik.max(best_ll)` where
  `best_ll = max(s.log_likelihood)` over all chain steps. When
  `best_ll > result.map_loglik` (legal under any non-flat prior
  — the MLE step is not the MAP step), the reported loglik comes
  from a step whose parameters are *not* `result.map_params`.
  The cell's `mle.toml` claims "loglik at MLE" using a loglik
  from one θ and parameters from another. The code comment at
  `profile.rs:1392–1397` explicitly acknowledges the mismatch
  and treats "still finite" as the goal.
- **Why it matters:** Exactly the bug class `f52d1ecd` just fixed
  on the IF2 path. The fix there ran a clean-PF re-pass at
  `r.mle`. Profile-PMMH has the same hazard under any informative
  prior — and after gh#73 (this same week!) model-IR priors are
  honored on the profile path, so this path is now reachable by
  ordinary users.
- **Fix:** Mirror `f52d1ecd`. Report `result.map_loglik` (coherent
  with `result.map_params` by construction —
  `pmmh.rs:478–483` sets them together). If `map_loglik` is
  non-finite, write `f64::NEG_INFINITY` honestly. Drop the
  `.max(best_ll)` clause. Add an integration test that runs
  PMMH-profile with a strongly informative prior, parses the
  saved params and loglik from `mle.toml`, and pfilters them;
  the two must agree to within PF SE.
- **Severity:** **Critical**.

---

### C6 — OCaml and Rust `parse_iso_date` accept *different* sets of strings — date()-literal corrupts IR; no cross-language golden table

- **Location:** OCaml `ocaml/lib/compiler/expander.ml:104–109`;
  Rust `rust/crates/ir/src/caltime.rs:101–143`. Promised golden
  table `ir/golden/caltime.tsv` does not exist (`find … -name
  'caltime.tsv'` → no results).
- **Category:** FFI / cross-language boundary (§8); user footgun
  (§6).
- **Defect:** Three concrete divergences:
  (i) **Zoned strings.** Rust accepts `2020-03-15Z` /
  `2020-03-15+06:00`. OCaml `int_of_string "15Z"` raises
  `Failure`.
  (ii) **Out-of-range months/days.** Rust rejects `2020-13-01` /
  `2020-02-30` (`caltime.rs:139–141`). OCaml accepts:
  `parse_iso_date "2020-13-01"` returns `(2020, 13, 1)`,
  `days_of_date` then returns garbage. The IR is emitted with a
  garbage `origin_rata_die` and garbage internal-time conversion
  for any `date()` literal.
  (iii) **Whitespace.** Rust trims; OCaml does not.

  Worse, the only cross-language equivalence test
  (`ocaml/test/test_compiler.ml:4106`) compares OCaml against
  OCaml — there is no test exercising `caltime::rata_die` on the
  Rust side against the same inputs. The doc-comment at
  `caltime.rs:14` *promises* an
  `ir/golden/caltime.tsv` fixture; it does not exist.
- **Why it matters:** Exactly the §5.6 failure mode the
  typed-time proposal flagged: "the design becomes a net negative
  if there are two un-pinned date parsers." A single date string
  traversing the OCaml `date()` path and the Rust `--data` path
  can produce two different internal times, or compile in one
  place and reject in the other. A user typing `date("2020-02-30")`
  by accident (a calendar mistake) on the OCaml side gets a
  corrupted IR with no diagnostic — and a confidently wrong fit.
- **Fix:** Two parts.
  (i) Port the Rust grammar and validation (trim, optional zone,
  leap-aware day range) to OCaml — *or* pick OCaml as canonical
  and tighten Rust accordingly. The proposal explicitly mandates
  "one grammar, one golden table" (§6.4).
  (ii) Commit `ir/golden/caltime.tsv` with
  `(origin, date, time_unit, expected_delta_days, expected_t)`
  covering leap-year rules, month boundaries, dates ≤ 1583 CE,
  negative deltas, and (post-fix) zoned strings. Add
  `rust/crates/ir/tests/caltime_golden.rs` and
  `ocaml/test/test_caltime_golden.ml`, both reading the same
  TSV.
- **Severity:** **Critical** (silent IR corruption under foreseeable
  typo).

---

### C7 — `origin_rata_die` is computed, serialized, deserialized — and never read by the Rust runtime

- **Location:** `rust/crates/ir/src/model.rs:145`;
  `ocaml/lib/compiler/expander.ml:4803–4806` writes;
  `rust/crates/cli/src/caltime_load.rs:214` and
  `rust/crates/cli/src/main.rs:901` use
  `ir::caltime::{date_to_internal,internal_to_date}` — which
  re-parse `origin` via `parse_iso_date`. Receipt:
  `rg -n 'origin_rata_die' rust --include='*.rs' | rg -v
  'origin_rata_die:.*(None|Option<i64>)'` → only the type
  declaration matches.
- **Category:** not wired through (§5); CLAUDE.md "delete dead
  code on sight."
- **Defect:** The proposal §6.2 / `caltime.rs:9–14`'s normative
  doc-comment require the runtime to *use* the compiler-derived
  `origin_rata_die` and never re-parse the `origin` string. The
  Rust runtime in fact never reads the field; every consumer
  passes `None` or ignores it. The two date parsers (C6) genuinely
  disagree — and the IR field that would have made the
  disagreement irrelevant is dead code.
- **Why it matters:** Either the contract is real (then C6 is
  fixable by routing through the field), or it isn't (then the
  field is a lie at the OCaml↔Rust boundary). Carrying the field
  unread gives false confidence to anyone reading the proposal.
- **Fix:** Either
  (a) thread `origin_rata_die` into
  `date_to_internal`/`internal_to_date` as the primary path
  (`origin` string becomes display-only), add a `validate.rs`
  check that `origin_rata_die.unwrap() == rata_die(parse_iso_date(origin))`
  on load; or
  (b) delete the field, the OCaml computation, and the schema
  entry; document that the runtime always re-parses. (a) is the
  cleaner choice; it composes with C6's fix.
- **Severity:** **Critical** (an IR field is a load-bearing
  contract; an unread one is documentation that lies).

---

## HIGH

### H1 — Event-action path silently allows negative compartment counts; gh#67 multiplies blast radius to ode/tau_leap/gillespie

- **Location:** `rust/crates/sim/src/intervention.rs:117–194`
  (`inject_event_deltas`). Compare with `apply_intervention`
  (same file, 250–353): `AbsoluteTransfer` is clamped to
  `int_s.counts[s_local]` there but raw-pushed in the event
  path; `Action::Add` hard-errors on negative `count` there but
  raw-pushed in the event path.
- **Category:** numerical correctness; not wired through.
- **Defect:** Three Actions miss the validation the
  intervention path has had: `Add` (no negative guard),
  `AbsoluteTransfer` (no clamp), `Set` (no `≥0` check —
  pre-existing, but now exercised by three more backends). After
  gh#67's `apply_events_at`, the post-step `first_negative()`
  check in ode/tau_leap/gillespie runs *before*
  `apply_events_at` (e.g. `tau_leap.rs:269` vs `:293`), so it
  cannot catch event-induced negatives.
- **Why it matters:** A user writing `events { booster : add(I,
  n_seed) at [tau] }` with a negative `n_seed` coming from a CLI
  fit/profile sweep silently corrupts state; the negative-count
  surface fires inside the next substep with the wrong
  diagnostic location. Different backends produce different
  errors at different times for the same model. For polio
  cVDPV2 work, where event-driven importation and booster
  campaigns are routine, this is exactly the silent-misspec
  class CLAUDE.md flags.
- **Fix:** Move the validation from `apply_intervention` into
  `inject_event_deltas`: hard-error on `Add` with negative
  resolved value, clamp `AbsoluteTransfer`, hard-error on `Set`
  with negative new value. Add a regression test per backend
  that exercises `add(I, -1) at [10]`.
- **Severity:** **High**.

### H2 — `apply_events_at` evaluates state-reading event actions against *post-step* state on ode/tau_leap/gillespie but *start-of-step snapshot* on chain_binomial

- **Location:** `rust/crates/sim/src/chain_binomial.rs:417–420`
  (snapshot = `scratch.int_s`, start-of-step) vs
  `rust/crates/sim/src/intervention.rs:222–224`
  (`apply_events_at` passes live post-step `int_s`).
- **Category:** numerical correctness; SOLID-LSP (events aren't
  substitutable across backends).
- **Defect:** A `FractionTransfer(I, R, 0.5)` event composed with
  a normal `infection` transition in the same substep computes
  `0.5 * I(t)` on chain_binomial but `0.5 * I(t+dt)` on the
  other three backends. The IR is meant to be backend-agnostic
  — this divergence violates the contract.
- **Why it matters:** Users running cross-backend sanity checks
  will see disagreement and not know whether it's their model
  or the backend semantics. `events_backend_parity.rs` only
  exercises `add(I, 100)` (state-independent), so it can't
  detect this. For real models with fractional events this is a
  silent numerical drift.
- **Fix:** Pick a canonical evaluation point in the spec
  (`docs/compartmental-ir-spec.md` §2.3 currently formalizes
  only interventions, not events) and align all backends.
  Start-of-step is the cleaner semantics (matches chain_binomial
  today). Add a parity test using a state-reading event action.
- **Severity:** **High**.

### H3 — `[source.from_csv].file` is TOML-anchored; `params`, `config.model`, `config.geo` are CWD-anchored — same TOML, two anchors

- **Location:** `rust/crates/cli/src/batch.rs:207` (TOML-anchored
  via `resolve_relative_to_toml`); `:521, :534–573, :664–668,
  :1057` (CWD-anchored).
- **Category:** user footgun (§6, "CLI ergonomics").
- **Defect:** Running `camdl batch run experiments/posterior.toml`
  from repo root finds `draws.csv` (TOML-relative) but fails on
  `params = "params.toml"` in the same file (CWD-relative). The
  user "fixes" it by moving/symlinking until both happen to
  work, with no error message guiding them.
- **Fix:** Route all four user-supplied paths through
  `resolve_relative_to_toml` — already tested and in scope.
  TOML-relative is the right behavior because the TOML is the
  manifest of related artifacts.
- **Severity:** **High**.

### H4 — `[source.from_csv]` accepts `"inf"` / `"nan"` / `"infinity"` as `f64` parameters

- **Location:** `rust/crates/cli/src/batch.rs:241`
  (`raw.parse::<f64>()`). Verified: `"inf".parse::<f64>() →
  Ok(inf)`, `"nan".parse::<f64>() → Ok(NaN)`.
- **Category:** numerical correctness (§4); user footgun (§6).
- **Defect:** A row with `R0 = "inf"` (Stan divergent draws, a
  typo, R producing `"Inf"` in NA-fill) becomes a parameter
  override of `f64::INFINITY`; rate expressions emit NaN; the
  chain-binomial backend handles NaN-as-rate inconsistently
  across compartments; the run lands in the manifest as
  "completed." `"NA"` errors loudly (good), but `"NaN"` doesn't.
  The commit advertises Stan-output compatibility, which makes
  encountering inf/nan rows more likely.
- **Fix:** After `parse()`, reject non-finite with a row+column
  message. Add tests for `"inf"`, `"nan"`, `""`.
- **Severity:** **High**.

### H5 — UTF-8 BOM in CSV header survives `.trim()`; first column silently named `"\u{feff}R0"`

- **Location:** `rust/crates/cli/src/batch.rs:209–219`. Verified:
  `"\u{feff}R0".trim()` is `"\u{feff}R0"`.
- **Category:** user footgun (§6).
- **Defect:** CSVs exported from Excel / many Windows tools /
  `write.csv` carry a UTF-8 BOM. Without explicit BOM stripping,
  every column passes through verbatim and the downstream
  "unknown parameter" error names a byte sequence the user
  cannot type.
- **Fix:** `content.strip_prefix('\u{feff}').unwrap_or(&content)`
  after `read_to_string`. Add a BOM fixture test.
- **Severity:** **High**.

### H6 — `[source.from_csv].delimiter` silently falls back to comma for empty / multi-character values

- **Location:** `rust/crates/cli/src/batch.rs:175–185`
  (`from_csv_separator`). `delimiter = ""` → `chars().next()` is
  `None` → falls back to `','`. `delimiter = "abc"` → silently
  picks `'a'`.
- **Category:** user footgun (§6); error-message design.
- **Defect:** Auto-generated TOML or user-fat-fingering of
  `delimiter` accepts garbage and produces a wrong-by-comma
  parse of a non-comma file.
- **Fix:** Return `Result<char, String>` with explicit errors
  ("delimiter must be a single character, got empty string" /
  "got 'abc'"). Tests for both.
- **Severity:** **High**.

### H7 — IF2 vs PMMH per-cell `loglik_rhat_starts` column shares a name but measures different quantities

- **Location:** `rust/crates/cli/src/profile.rs:1231–1233` (IF2:
  `if2_perturbed_loglik`) vs `:1386–1388` (PMMH:
  `s.log_likelihood`); aggregated at
  `profile_diagnostics.rs:268–278`. Docs at
  `docs/inference.md:713` describe one quantity without
  mentioning the IF2 caveat.
- **Category:** SOLID-LSP; user footgun.
- **Defect:** Same column header, two distinct quantities. R̂ on
  the IF2 trace measures whether K starts wandered the same
  basin — *not* posterior convergence. The IF2 source itself
  documents the perturbed-loglik as "NOT useful for model
  assessment" (`profile.rs:1227`). A user comparing R̂ across
  two profile runs (one IF2, one PMMH) draws wrong conclusions.
- **Fix:** Split into `rhat_perturbed_loglik` (IF2) vs
  `rhat_loglik` (PMMH). Each algorithm populates only its own;
  the other writes NaN. Document the distinction in
  `inference.md` §"Per-cell diagnostics" with the source
  caveat verbatim.
- **Severity:** **High**.

### H8 — `fit summary` / `fit tree` / `mle.toml` don't surface resolved-prior provenance

- **Location:** `run.json.FitMeta.resolved_priors` is populated
  (`fit/mod.rs:1651–1675`) but never rendered by `fit_summary.rs`,
  `fit_tree.rs`, or `provenance.rs:115–135`
  (`MleProvenance` omits priors).
- **Category:** not wired through (§5).
- **Defect:** Classic "computed but not surfaced" — a reviewer
  reading the human-facing summary cannot tell whether the
  posterior was driven by the model-IR prior, a fit-toml
  override, or `flat_explicit`. The whole provenance system
  exists so a downstream consumer can audit "what priors
  actually drove the chain that produced this credible
  interval."
- **Fix:** Add a `Priors` section to `fit summary` text/markdown
  output (one row per estimated parameter: `name | distribution
  | source`). Same for `fit tree`. For `mle.toml`, add
  `resolved_priors = [{param=…, source=…}, …]` under
  `[provenance]` — even though IF2 doesn't *use* priors, the
  precedence the downstream Bayesian stage *would* consume is
  worth pinning.
- **Severity:** **High**.

### H9 — `camdl simulate --draws prior --fit fit.toml` ignores the three-tier chain and contradicts `fit run`

- **Location:** `rust/crates/cli/src/main.rs:1439–1469`
  (`generate_prior_draws`). Checks only `spec.prior` on the
  fit-toml side; never falls through to model-IR `~` prior.
- **Category:** DRY (§2); user footgun (§6).
- **Defect:** A model that declares `beta ~ log_normal(…)` plus a
  fit toml omitting `prior =` succeeds under `camdl fit run`
  (tier 2 applies) but errors under `camdl simulate --draws
  prior --fit fit.toml` ("Missing or flat priors: beta"). The
  user-facing claim that the model file is the "single source
  of truth for stable priors" (`docs/inference.md:585–589`) is
  broken at this entry.
- **Fix:** Reuse `priors_precedence::resolve_priors_with_precedence`
  instead of reinventing a third precedence chain in `main.rs`.
- **Severity:** **High**.

### H10 — Profile-PMMH hardcodes `burn_in = 100`; `--pmmh-steps ≤ 100` silently emits empty diagnostics

- **Location:** `rust/crates/cli/src/profile.rs:1294–1305`
  (`burn_in: 100` hardcoded); test fixture
  `tests/profile_diagnostics.rs:172–175` acknowledges the trap
  with a "use 200 because of burn_in" comment.
- **Category:** user footgun; not wired through.
- **Defect:** `--pmmh-steps` (default 500) is exposed;
  `--pmmh-burn-in` is not. The engine's `post_burn_steps =
  n_steps - burn_in` (`sim/src/inference/pmmh.rs:513–518`),
  so `--pmmh-steps 100` yields zero post-burn-in samples,
  empty loglik trace, R̂ = NaN — but the cell still writes
  `mle.toml`, so the failure looks like a convergence problem.
- **Fix:** Add `--pmmh-burn-in`, `--pmmh-thin`,
  `--pmmh-adapt-start`, `--pmmh-proposal-scale`. Reject
  `pmmh_steps ≤ burn_in` at dispatch with a named error. Or
  accept a `--pmmh-config fit.toml` that reads the `[pmmh]`
  block.
- **Severity:** **High**.

### H11 — `DemeId` / `CompartmentId` / `TransitionId` are `pub type` aliases, not newtypes

- **Location:** `rust/crates/sim/src/lineage/mod.rs:83`:
  `pub type DemeId = u32; pub type CompartmentId = usize; pub
  type TransitionId = usize;`.
- **Category:** type/trait design (§1 newtype hygiene).
- **Defect:** The "real `DemeId`" promised by `6f848bcd` is a
  transparent alias. Any `u32` is assignable to a `DemeId`; a
  `CompartmentId` is interchangeable with a `TransitionId`.
  Pool keys `(DemeId, CompartmentId)` are `(u32, usize)`
  tuples in disguise. Two helper functions (`writer.rs:110–118`
  `deme_column` vs `comp_column`) exist *only* to disambiguate
  sentinels the type system doesn't.
- **Why it matters:** Rubric §1 is explicit: `usize` is not a
  compartment, stratum, transition, or particle index. The
  domain ("which patch did the infector come from") demands
  that swapping `(deme, comp)` is a compile error.
- **Fix:** Make all three newtype structs
  (`pub struct DemeId(pub u32)`, etc.) with `Copy + Eq + Hash
  + Ord + Debug`. Do *not* implement `From<u32>` — force
  callers to write `DemeId(n)` so Cartesian-index computations
  can't leak into a deme slot.
- **Severity:** **High** (design debt with a concrete
  correctness hazard).

### H12 — Streaming `realize` and the lineage projection helpers trust line lists to be time-ordered with no guard

- **Location:** `rust/crates/sim/src/lineage/realize.rs:162–172`
  (docstring says "MUST be in recorded time order"; no check);
  `event_log_io.rs:230–260` (reads file order, no
  monotonicity assert); `tree.rs:450–453`, `:546–562`.
- **Category:** not wired through; user footgun.
- **Defect:** Three correctness-load-bearing functions
  (`RealizeState::process`, `summarize`, `migration_event_count`)
  document a precondition the IO layer never checks. A
  user-edited TSV, a Parquet file with shuffled row groups, or
  a future writer regression silently miscomputes — at-step
  snapshots, parent draws, migration-vs-shuffle classification.
- **Fix:** Track `last_time: f64` and `last_step: u64` in
  `RealizeState::process`; hard-error
  `SimError::Validation` on regression. Assert monotonicity in
  `summarize` and `migration_event_count` too.
- **Severity:** **High**.

### H13 — `events_backend_parity.rs` (and four sibling tests) silently skip when `target/release/camdl` is absent

- **Location:** `rust/crates/cli/tests/events_backend_parity.rs:91–94`
  and four siblings: `seed_timing_e2e.rs`, `lineage_e2e.rs`,
  `dated_data_loader.rs`, `lineage_migration_e2e.rs`. Receipt:
  `grep -l "release camdl binary not built" rust/crates/cli/tests/`
  → five files.
- **Category:** tests (§9).
- **Defect:** All five tests silently pass when the binary
  doesn't exist. `cargo test --workspace` without a prior
  `cargo build --release` looks green even though gh#67's
  regression net is no-op. Per rubric §9: "Tests that actually
  fail when the code is wrong. A test that passes whether or
  not the algorithm is correct is worse than no test."
- **Fix:** Either build from the test (`escargot` or a single
  `Command::new("cargo")` per test process) or fail loudly with
  `panic!("release camdl binary required; run `make build`
  first")`. Pick fail-loud unless build time dominates.
- **Severity:** **High** (a class-of-test gate that is currently
  theatrical).

### H14 — `CAMDL_TRACE_STEPS` documented in CLAUDE.md but only wired for chain_binomial `Action::Add` events

- **Location:** `CLAUDE.md` "Debugging a diverging simulation"
  promises intervention tracing. Receipt:
  `grep -n "CAMDL_TRACE\|eprint" rust/crates/sim/src/intervention.rs`
  → one hit, inside `inject_event_deltas`'s `Action::Add`
  branch only (line 144).
- **Category:** not wired through; doc-vs-code drift.
- **Defect:** The primary debugging tool CLAUDE.md tells users
  to reach for doesn't fire for scheduled interventions, for
  Transfer/Set events, or for any of the new gh#67/gh#69
  surfaces. The diagnostic gap aligns precisely with the
  surfaces just modified.
- **Fix:** Wire trace prints in all four `Action` arms of
  `inject_event_deltas`; in `apply_intervention`; and in the
  `AtTimesExpr` resolver so gh#69's parametric resolution is
  visible. Document the format.
- **Severity:** **High**.

### H15 — `always_active` IR flag is overloaded as "is-event" but spec defines it as "fires regardless of scenario enable/disable"

- **Location:** `ocaml/lib/compiler/expander.ml:3623–3624`
  (interventions: `always_active=false`, events:
  `always_active=true`). Spec `docs/compartmental-ir-spec.md:141`
  defines the flag as scenario-participation. Rust runtime
  uses it as a discriminant (`intervention.rs:99,133`).
- **Category:** type/trait design (§1); user footgun.
- **Defect:** One boolean encodes two orthogonal axes (scenario
  participation; fire-every-substep semantics). The DSL
  already distinguishes `interventions {}` and `events {}` at
  parse time, so the IR could carry an explicit `kind:
  Intervention | Event` variant. The flag-overload breaks the
  moment anyone adds a third axis the spec already names
  ("always-active intervention" — gh#69 is exactly that
  shape).
- **Fix:** Replace `bool always_active` with `enum Kind {
  Intervention, Event }`, or split into two top-level lists
  `model.interventions` / `model.events`. Atomic IR change
  (back-compat is non-goal per CLAUDE.md).
- **Severity:** **High**.

### H16 — `param_kind` is a free-form `Option<String>` in IR; no enum, no schema constraint, no validator

- **Location:** `rust/crates/ir/src/parameter.rs:130`;
  `ocaml/lib/ir/ir.ml:318`; `ir/schema.json:768`. `validate.rs`
  performs no check.
- **Category:** type/trait design (§1); not wired through (§5).
- **Defect:** A typo (`"isntant"`, `"Instant"`, `"duration "`)
  silently disables date-rendering for the parameter; the
  doc-comments listing valid kinds are out of date (omit
  `instant`/`duration`).
- **Fix:** Replace `param_kind: Option<String>` with `Option<ParamKind>`
  enum, `serde(rename_all = "snake_case")`. Add `enum`
  constraint to the schema. Mirror the OCaml side (parser
  already has the tokens at `parser.mly:285–286`).
- **Severity:** **High**.

### H17 — Rendering an `instant`-kind estimand with no `origin` silently degrades to numeric; proposal §9.9 mandates a warning

- **Location:** `rust/crates/cli/src/fit/fit_summary.rs:121–130`;
  acceptance criterion in
  `docs/dev/proposals/2026-05-22-typed-time-and-dsl-ergonomics.md`
  §9.9: "rendering an `instant` with no origin **falls back to
  numeric with a note**." The implementation falls back without
  a note (docstring at `:102–107` explicitly contradicts the
  proposal: "no formatter changes shape").
- **Category:** user footgun (§6); proposal compliance.
- **Defect:** The most likely user state when an `instant`-kind
  parameter has no origin is "I forgot to declare `origin`."
  Silent fallback lets the user ship a fit whose seed-time
  interpretation is *in unanchored time*, then read the number
  as anchored.
- **Fix:** When the model has instant-kind parameters but no
  `origin`, emit a one-line warning at the top of the summary:
  `"warning: model declares instant-kind parameter(s) [tau, …]
  but no origin; rendering as numeric. Add origin =
  date(\"YYYY-MM-DD\") for date-rendered estimates."`
- **Severity:** **High**.

### H18 — `internal_to_date` rounds to whole days; `--dates` column silently coalesces sub-day `dt` rows onto the same date

- **Location:** `rust/crates/ir/src/caltime.rs:178`
  (`(t * days_per_unit(time_unit)?).round() as i64`).
- **Category:** numerical correctness; user footgun.
- **Defect:** With `dt = 0.5 'days`, snapshots at
  `t = 0.0, 0.5, 1.0, 1.5, 2.0` render dates `origin+0,
  origin+1, origin+1, origin+2, origin+2`. The TSV is lossy
  for sub-day integration. `docs/dates.md:230` advertises
  `--dates` as a non-lossy "additive column."
- **Fix:** Document the rounding in the `--dates` TSV header,
  *and* warn (or hard-error) when `dt < days_per_unit(time_unit)`.
- **Severity:** **High**.

### H19 — Profile-PMMH exposes only three of eight PMMH knobs; the rest are hardcoded with a poor default proposal scale

- **Location:** CLI `args/mod.rs:1175–1194` exposes
  `pmmh_steps`, `pmmh_particles`, `pmmh_rho` only. Per-cell
  config at `profile.rs:1294–1305` hardcodes
  `adapt: true, adapt_start: 50, thin: 1, burn_in: 100,
  proposal_sd = rw_sd * 5.0`.
- **Category:** not wired through; user footgun.
- **Defect:** The hardcoded `proposal_sd = rw_sd * 5.0` is
  load-bearing: `--rw-sd auto` tunes for IF2 cooling, *not*
  MH proposal scale, so the 5× multiplier is a heuristic that
  yields low acceptance rates on tight posteriors — and gh#74
  diagnostics surface the problem with no CLI knob to address
  it.
- **Fix:** Add `--pmmh-burn-in`, `--pmmh-thin`,
  `--pmmh-adapt-start`, `--pmmh-proposal-scale`. Or accept a
  `--pmmh-config fit.toml` that reads the fit-side `[pmmh]`
  block, sharing the parser.
- **Severity:** High (paired with H10).

---

## MEDIUM

### M1 — `survey_top_k` ranking has unstable NaN tiebreak
- `rust/crates/cli/src/fit/init.rs:522–525`: `partial_cmp(...).unwrap_or(Equal)`
  makes NaN compare equal to everything. Today the survey
  writer puts NaN-as-`NEG_INFINITY` at the bottom, but the
  consumer doesn't replicate. A user-edited TSV with NaN
  rows near the top can be selected as a chain start. Fix:
  filter or treat NaN as `NEG_INFINITY` at the consumer.

### M2 — `clean-eval` re-pass uses `wrapping_add(1)` for seed derivation, not domain-separated streams
- `rust/crates/cli/src/profile.rs:1203–1220`. `seed + 1` may
  collide with `seed_for_cell(start_idx)` of an adjacent cell
  whose XOR mask is one apart. Low likelihood; fragile audit
  story. Fix: `StatefulRng::new_stream(seed,
  STREAM_CLEAN_EVAL)` per `rng.rs:26–31`.

### M3 — `--suppress-warnings` is a bool; future profile warnings will be silently swallowed
- `rust/crates/cli/src/profile.rs:781–800`. The flag is
  documented as silencing one specific warning but the gate
  is broad. The next warning added gets silenced too without
  the user opting in. Fix: typed warning ids
  (`--suppress profile_flat_prior_fallback`).

### M4 — Cumulative-mass parent draw has a `cumulative ≥ u` edge case when `u == 0` and first pool mass is 0
- `rust/crates/sim/src/lineage/realize.rs:421–435`.
  `ChaCha8Rng::gen()` returns `[0.0, 1.0)`; `0.0` is reachable.
  Selecting a zero-mass pool surfaces as "weights diverged"
  instead of the correct "skip zero-mass pools." Fix: strict
  `cumulative > u`, or add a `chosen = last` fallback at
  `u → total`.

### M5 — `summarize` drops non-transmitting seed individuals; sampling under-counts seeds with empty `parent_deme`
- `rust/crates/sim/src/lineage/tree.rs:486–493`. Seed
  individuals with `parent_deme = None` are silently assigned
  deme 0 (`unwrap_or(0)`); seeds that never transmit have no
  entry in `acc` at all. Thread `EventLog.initial_pools`
  through to `summarize`; mint seed `IndividualSummary` from
  the initial pools.

### M6 — `TransitionObserver::on_fired`'s `DemeId` argument is always passed `0` and discarded
- `rust/crates/sim/src/lineage/mod.rs:259–273` (trait);
  `gillespie.rs:272`, `tau_leap.rs:254`,
  `chain_binomial.rs:200` (all pass `0`); `event_log.rs:262`
  (`_deme`). Dead parameter. Either thread the real firing-deme
  or delete the parameter.

### M7 — `[source.from_csv]` does not validate against the model's parameter set; per-run errors instead of one fail-fast
- `rust/crates/cli/src/batch.rs:247–259`. The doc-comment is
  honest that "downstream errors on unknown parameters," but
  the IR is loaded by then; validate column → parameter mapping
  up front, fail once.

### M8 — Dry-run printer labels `from_csv` as "Sweep" and abbreviates "varying keys" (every column varies by design)
- `rust/crates/cli/src/batch.rs:1144–1215`. `cmd_batch_run`
  knows the difference (line 677) but `--dry-run` returns
  earlier with the wrong label. Fix: pass a `PointSource` enum
  in.

### M9 — `cmd_batch_status` swallows from_csv parse errors and synthesizes a phantom grid (`Vec::new()` → one null point)
- `rust/crates/cli/src/batch.rs:1073–1075` →
  `plan_runs:411`. Status lies about the grid. Fix: print
  "Live count: unavailable (CSV parse failed)" instead.

### M10 — PGAS coverage gap in `fit_priors.rs` (all six tests stage PMMH only)
- `rust/crates/cli/tests/fit_priors.rs`. PGAS shares the
  validator but no end-to-end test pins it. PGAS has its own
  runtime "refuses to run with implicit improper-uniform
  priors" error that no test fires. Add one PGAS variant of
  each PMMH test.

### M11 — Profile-PMMH inline comment ("Not reachable on the profile path today") contradicts the code path; `flat_explicit` *is* reachable via `--fit fit.toml`
- `rust/crates/cli/src/profile.rs:938–943`. The comment misleads
  future readers. No test pins explicit-flat-via-`--fit` on the
  profile path. Fix the comment; add a regression test.

### M12 — `--dates` argument validation is duplicated and exits with non-Diagnostics text
- `rust/crates/cli/src/main.rs:863–879, 916–925`. Same
  "no `origin`" check, same `eprintln + exit(1)`. Lift to
  helper, route through the standard CLI error pathway.

### M13 — Out-of-range / malformed `date()` literals `failwith` (then silently absorbed to `0.0` inside `date_range`)
- `ocaml/lib/compiler/expander.ml:108` (failwith),
  `:1246` (`with Failure _ -> 0.0`). CLAUDE.md forbids
  `failwith` for user-facing errors; the silent absorb is a
  worse footgun. Fix: rewrite `parse_iso_date` to return
  `result`, emit a Diagnostics E-code, remove the absorb.

### M14 — OCaml-side IR emission doesn't validate `origin` before computing `origin_rata_die`
- `ocaml/lib/compiler/expander.ml:4805–4806`. A malformed
  origin produces nonsense `origin_rata_die`. Fix lands
  automatically with C6 / M13.

### M15 — `time_typing.ml` `lub_dur` has dead `TInstant` branches admitting silent fall-throughs
- `ocaml/lib/compiler/time_typing.ml:61, :163`. Comments
  acknowledge "shouldn't normally reach here." Future expander
  changes that produce `instant + instant` arithmetic silently
  succeed. Fix: split `tclass`; reject via `assert false` or a
  diagnostic at the offending caller.

### M16 — gh#69 IR schema change ships without an `ir/VERSION` bump (additive, justified in commit body); policy needs to be documented
- `07394aff` is "purely additive." Within one build it's
  fine, but a stale Rust binary deserializing a new OCaml IR
  would fail without the version check catching it. Document
  the policy in the spec — bump on every schema change, or
  only on breaking? The current ambiguity is itself a smell.

### M17 — `apply_interventions_at`'s `_tolerance: f64` parameter is unused; five call sites pass `1e-10`
- `rust/crates/sim/src/intervention.rs:85`. Per CLAUDE.md "No
  loose semantics" — a parameter that looks meaningful is dead.
  Delete the parameter; drop the constant from call sites.

### M18 — `apply_intervention` `NegativeCount` error hard-codes `t: 0.0` despite `t` being in scope
- `rust/crates/sim/src/intervention.rs:336–341`. User sees
  `NegativeCount at t=0` for an intervention firing at, say,
  t=180, and investigates initial conditions instead of the
  intervention. One-character fix: `t: t`. Add a regression
  test.

### M19 — Absorbing-state output-flush in Gillespie (gh#70) is acknowledged in gh#67 commit message, unfixed
- `gillespie.rs:152–158`. A user writing `init { I = 0 }`
  plus `add(I, 1) at [tau]` (the natural seed-timing pattern)
  trips the absorbing-state branch. Headline use case;
  unblocked work waiting.

### M20 — `event_log::wrapping_add(1)` for step counter is inconsistent with `mint`'s `checked_add` on the ID counter
- `rust/crates/sim/src/lineage/event_log.rs:251, 309`. Both
  in the same module. The wrapping variant silently corrupts
  on overflow; checked variant panics. Make both `checked_add`.

### M21 — IF2 trace-doc claim of posterior R̂ semantics has no caveat in the user-facing reference
- `docs/inference.md:693–734`. Caveat lives in the code
  docstring at `profile.rs:1226–1230`. Surface it in
  `inference.md` alongside the column description.

### M22 — Profile loads `--fit fit.toml` via `FitConfigV2::load` without calling `.validate()`
- A malformed fit-toml (empty bounds, singleton simplex group,
  etc.) compiles on the profile path while `camdl fit run`
  rejects it. Parity gap; file separately if substantive.

---

## LOW

### L1 — `LineageRng::below` is a hand-rolled multiply-and-floor; biased for `n ≥ 2⁵³`
- `rust/crates/sim/src/lineage/mod.rs:131–137`. Comment
  acknowledges "with `n` bounded by population size this has
  negligible modulo bias." Type doesn't enforce the bound.
  Fix: `self.0.gen_range(0..n)`.

### L2 — `LineageRng::from_sim_seed` mixes via `^ LINEAGE_RNG_OFFSET` *and* `wrapping_add(0xdeadbeef_cafebabe)`; one constant suffices
- `rust/crates/sim/src/lineage/mod.rs:117–121`. Comment
  justifies the XOR; nothing justifies the wrapping-add.

### L3 — Stale dead-reference: docstrings cite a `LineageObserver` type that no longer exists
- `writer.rs:81`, `event_log.rs:137`, `realize.rs:14, :391`.
  The Layer-1/2 split refactor (`ab79ad2d`) deleted the inline
  observer; the prose survived. `--deny
  rustdoc::broken_intra_doc_links` would catch the first.

### L4 — CLI exit on error in `lineage.rs` uses `eprintln + exit(1)` in 14 places, bypassing the shared CLI error infrastructure
- `rust/crates/cli/src/lineage.rs` (14 hits). Cluster D/E work
  already does Result-bubbling elsewhere. Make lineage CLI
  follow.

### L5 — Test coverage gap for `[source.from_csv]`: no test pins the error messages for `NA`, empty cell, or scientific notation
- `rust/crates/cli/src/batch.rs:1597–1606` covers only
  `"not-a-number"`. Future refactors that silently accept
  `"NA"` as zero (a real footgun some tools have) wouldn't be
  caught.

### L6 — `[source.from_csv]` allocates two HashMaps per row
- `rust/crates/cli/src/batch.rs:239–259`. Hoist a
  `col_index` `&str → usize` map from the header; build one
  HashMap per row in the un-mapped path. One-alloc-per-row.
  Cheap.

### L7 — Commit `0f39d787` uses "Rules 1, 2, 4, 5, 7" numbering that doesn't match the proposal — leaves phantom "rules 3 and 6" looking unimplemented
- Verify: the proposal §3 numbers Rules 1 and 2; the implementation
  numbering is internal. Source comments in `time_typing.ml` /
  `expander.ml:4028` use "Rule 7" for what the proposal calls
  "Rule 1 extended to recurring schedules." Use the E-code
  names (E320–E323, W324–W326) in source and future commits.

### L8 — Language-spec examples use Julian `365.25` while the engine uses Gregorian `365.2425`
- `docs/camdl-language-spec.md:936, 971, 1075, 1594`. Doc-vs-doc.
  Edit the four spec example lines. (Code is correct.)

### L9 — Cross-cutting: there is no OCaml↔Rust calendar-constants equivalence test; only header comments saying "MUST match"
- See C6 / C7. Header comments at `caltime.rs:9–14` and
  `expander.ml:115` agree numerically — but per CLAUDE.md
  "single source of truth, mirror only with an equivalence
  test." The `ir/golden/caltime.tsv` fixture is the right
  place for it; C6's fix lands this too.

---

## Followups (low confidence; worth confirming before acting)

- **Cond-branch lineage tests.** `8a6dc8b9` is titled "deferred
  Cond-branch handling for `#[lineage]`." No test exercises
  `#[lineage] infection : S → I @ if t > 30 then β·S·I/N else 0`.
  A model author's intuition will silently disagree with the
  current implementation; pin via a regression test.
- **IF2 + priors at provenance.** `fit/mod.rs:1650` only emits
  `resolved_priors` when at least one Bayesian stage is
  present. A `survey + if2` workflow that produces an
  `mle.toml` consumed by a later `pgas` stage records prior
  provenance only at the PGAS stage — not the IF2/survey
  stages. Intent or gap?
- **`PriorSource::ModelIrHierarchical`.** Today both `~`
  prior and `parameter.hierarchical` collapse into
  `PriorSource::ModelIr`. A reviewer asking "was this prior
  conditional on a hyperparameter?" can't tell from `run.json`.
- **`AtTimesExpr` permission of `Expr::Time` / `Expr::Dt`.**
  The validation at `compiled_model.rs:830–862` allows
  `Expr::Time` / `Expr::Dt` "so future use-cases can be
  supported." Today `intervention_fire_times` passes
  `t: 0.0, dt: 0.0` (`intervention.rs:39–40`), so any current
  model that *does* reference `t`/`dt` in a schedule
  expression silently resolves to 0/0. Either reject these
  references explicitly, or substitute the intended value
  (`t_start`).
- **OCaml-side `parse_iso_date` adopting a `result`-returning
  shape (M13) also enables a single source of truth at C7.**
  These three (C6, C7, M13, M14) are coupled and should land
  in one commit.

---

## What lands first

The remediation order that minimizes regression-risk and
unblocks the most users:

1. **C2** — file the incident (no code change).
2. **C6 + C7 + M13 + M14** — one OCaml-side commit unifying
   `parse_iso_date`, lifting `origin_rata_die` to the
   load-time validator, and adding the `caltime.tsv` golden
   table. Single atomic IR change.
3. **C1** — fix `survey_top_k` ranking; gate on a `prior_hash`
   match between survey and fit. Add a refusal path for the
   conservative variant first.
4. **C5** — drop `.max(best_ll)` in profile-PMMH; add the
   integration test.
5. **C4** — add the three regression tests for gh#69; this is
   a pure-test commit.
6. **C3** — guard Gillespie with a hard error when a
   time-dependent rate composes with a coarse output grid.
   The thinning fix itself is a v0.2 lift.
7. **H1, H2** — atomic event-action validation pass (the two
   pair cleanly: tighten `inject_event_deltas` *and* fix the
   evaluation-point divergence in the same commit).
8. **H3–H6** — the `from_csv` cleanup pass.
9. **H7, H8, H10, H17, H18** — diagnostic-and-rendering
   correctness on the user-facing summary surface.
10. **H11, H15, H16** — three small newtype/enum lifts. Atomic
    IR changes; do these once together.
11. Everything below H is best handled as the surfaces are
    touched for other reasons.
