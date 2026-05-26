---
status: open
date: 2026-05-26
kind: surface map
scope: last-week diff — commit inventory + per-cluster review focal points
reviewer: internal (HEAD = d3ebe965)
findings: 2026-05-26-week-audit-findings.md
---

# Week-of-2026-05-19 → 2026-05-26 audit — surface map

Working notes for the review pass driven from `docs/dev/code-review.md`.
Built from `git log --since="7 days ago"` (98 commits, 428 files,
+36 945 / −3 447). The clusters below are review surfaces, not the
git order.

## Cluster A — Lineage tracking subsystem (May 19–21, 25 commits)

Largest new surface this week. New directory `rust/crates/sim/src/lineage/`
(event_log, event_log_io, deme, project, realize, tree, writer, mod) plus
CLI surface (`rust/crates/cli/src/lineage.rs`) and a brand-new
DSL/IR attribute `#[lineage]`.

**Architecture (per the proposal series):** three layers —
1. Layer-1 event log (every transition writes a row, streamed Parquet)
2. Layer-2 `realize` (deterministic projection of event log into
   per-individual trajectories, sojourn, cohort)
3. Layer-3 trees / sampling tiers (all-individuals, stratified)

**Commits, in order:**
- `8d4685a2` docs(proposal): simulate_with_lineages — line lists and trees
- `9d4e3eb6` docs(proposal): individual sampling layer (supersedes v1)
- `dc703694` feat(ir): `#[lineage]` foundation — DSL attribute, linearity check, IR fields (76 files +882 −131)
- `756b119e` feat(ir): bump seir_spatial_5 fixture to ir_version 0.5
- `8a6dc8b9` docs(proposal): deferred Cond-branch handling for `#[lineage]`
- `40024485` feat(sim): lineage runtime — observer seam, identity tracking, line-list writers, offline tree (9 files +2038 −13)
- `3c97a16b` feat(cli): `camdl simulate --lineages` + offline `camdl lineage tree`
- `640a747f` test(sim,cli): lineage acceptance tiers + end-to-end pipeline (+1267)
- `6f848bcd` feat(sim): stratified parent attribution — real `DemeId` and `parent_deme`
- `b5df4ae4` test(sim): Tier 2b — stratified contact-weighted parent attribution (+957)
- `c5a58c36` test(sim): Tier 5 external-oracle scaffold for stratified attribution
- `8ee1ac35` docs(proposal): park Tier-5 oracle choice
- `d7fc79ed` feat(sim): lineage tracking on tau-leap + chain-binomial backends (+633)
- `c2da18d0` feat(cli): offline lineage sojourn + cohort projections (+468)
- `570afbb2` fix(sim): count Gillespie lineage edges in sub-dt diagnostic
- `abcdde4e` test(sim): Tier 4 — coalescent-interval validation of transmission tree (+484)
- `dd297ed4` test(sim): offspring-distribution check vs realized effective-R
- `6ea3cb5d` docs(proposal): fix Tier-4 coalescent rate (was wrong by I²/N) — flagged as a docs fix; verify the code matched the original or the corrected rate
- `0047d52f` docs: add docs/lineages.md
- `e3f98b75` docs(proposal): sampling milestone
- `1d8caf09` docs(proposal): three-layer architecture
- `940970ca` docs(proposal): apply review fixes to three-layer RFC
- `56d43e10` docs(proposal): resolve VGsim oracle question; pin MASTER/nosoi
- `ab79ad2d` feat(sim): split lineage tracking into Layer-1 event log + Layer-2 realize (+1116 −513)
- `32cb5e97` feat(cli): wire event-log recording + `lineage realize`; rework tests (+641 −532)
- `de6a0b33` fix(sim): realize batched events against frozen start-of-step pools
- `8c4bcf68` docs: rewrite lineages.md for three-layer event-log workflow
- `8dd7474b` fix(cli): infer `--event-log` format from output extension
- `a485c139` feat(lineage): all-individuals sampling with pendant tips at sampling time
- `a42eecc5` feat(cli): all-individuals + stratified sampling for lineage tree
- `0ce593bc` docs(lineages): document all-individuals + stratified sampling
- `ea8572a1` docs(notes): scaling assessment of the lineage workflow
- `782cecc5` perf(lineage): chunk event-log Parquet writer into row groups
- `23febf9e` perf(lineage): don't retain IDs in write-only identity pools
- `55457473` perf(lineage): stream the event log into realize (bounded memory)
- `8cb4766e` docs(proposal): human vs pathogen migration + deme trajectory
- `b2fa11dc` feat(lineage): per-individual deme trajectory; fix migrant sampling
- `ddab05f4` feat(lineage): cross-deme/migration statistics + paired migration goldens (+996)

**Review focal points:**
- `#[lineage]` linearity check in `ocaml/lib/compiler` (autodiff or
  expander) — what counts as "linear" and where is this enforced
- Layer-1 → Layer-2 boundary correctness: do the event log writer
  and the realize reader agree on schema / ordering / dt semantics
- Identity pool semantics: when are IDs retained vs dropped?
- Migrant sampling fix (`b2fa11dc`) — what was wrong, what is right,
  is there a regression test pinning the agreement
- Coalescent-rate validation: docs flagged a previous I²/N error in
  Tier-4; check that the **code** uses the post-fix rate, not the
  pre-fix one
- Streaming realize (`55457473`): bounded memory only if the event
  log is sorted; verify the writer guarantees this
- Parquet row-group chunk size (`782cecc5`) — is the chunk size
  configurable or a magic constant
- DemeId is now "real" (`6f848bcd`) — confirm no `usize` confusion
  with stratum index or compartment index

## Cluster B — Typed time / calendar time (May 22, 18 commits)

Major DSL + IR + CLI lift. Phase 1 (rules), Phase 2 (date_range +
primitives), Phase 3 (IR: instant/duration kinds), Phase 4 (CLI dates).

**Commits, in order:**
- `fd09bacb` docs(proposal): calendar time — dated I/O boundary translator
- `f546c57d` docs(proposal): calendar time — verify engine origin-invariance
- `0afc988d` feat(ir): caltime — calendar-date ↔ internal-time boundary conversion (+281)
- `e21c557e` feat(cli): dated data loader — calendar-time boundary translator (phase 2)
- `6b9a9d96` feat(ir): time parameter-kinds (instant/duration) + numeric origin in IR (phase 3) — touches **85 files**, mostly golden JSON
- `57f4def0` feat(cli): `--dates` calendar column in simulate output (phase 4)
- `d2308f2f` feat(cli): render instant-kind estimands as calendar dates in fit summary (+1226)
- `3525dc77` fix(compiler): preserve negative lower bounds in parameter declarations
- `2db6dd4a` docs: add docs/dates.md
- `e69516dd` fix(inference): thread observation time into obs likelihood/sample/mean (+197 −105)
- `9481135b` docs(claude.md): alpha-status + DSL design principle
- `5acd5450` docs(spec): correct §2.1 month/year conversion constants to Gregorian
- `cc491c98` docs(proposal): typed time RFC — anchored vs unanchored
- `8cc96688` docs(cheatsheet): one-page DSL surface orientation
- `e878effa` docs(claude.md): require reading normative docs first
- `53b8fded` docs(cheatsheet): drop reference to deleted incident report
- `24717228` docs(claude.md): require evidence inline with claims
- `deb42dc8` docs(typed-time): address upstream review (A/B/C)
- `9e88d579` docs(typed-time): add date_range generator to Phase 2
- `bb284673` docs(typed-time): finalize proposal
- `0f39d787` feat(compiler): typed-time Phase 1 — Rules 1, 2, 4, 5, 7 + W326 (+1169)
- `959b0d16` feat(compiler): typed-time Phase 2 — calendar primitives + date_range (+1091)
- `f5af2786` docs(typed-time): fix date_range quarterly example
- `76fc6ea4` docs(typed-time): Phase 3 — integrate vocabulary
- `0b2bec84` test(golden): first anchored fixture exercising Phase 1+2 typed-time (+1372)

**Review focal points:**
- `caltime.rs` — single source of truth for date↔t conversion; check
  the rata-die invariant and round-trip
- IR `ir_version` bumped to 0.6; check OCaml `ir_version_generated.ml`
  / Rust `ir/lib.rs` agree atomically
- "Numeric origin" stored in IR — how is it validated when present /
  required when absent (e.g. anchored-only operations like `date_range`
  vs unanchored sims)
- Phase 3 commit touches 85 files but adds only +312 −138 — overwhelmingly
  golden-JSON updates; spot-check a handful for hand-edited drift
- `e69516dd` "thread observation time into obs likelihood/sample/mean"
  changed signatures across `multi_stream_obs.rs`, `obs_model.rs` —
  any inference algorithm still calling the old shape?
- `3525dc77` negative-lower-bound fix — does the test actually exercise
  a previously-failing case (TDD red)?
- `--dates` column behavior — what when origin isn't set?
- "instant" vs "duration" parameter kinds: any path that treats both
  the same way (e.g. constant-folding, dimcheck)?
- `time_typing.ml` Rules 1,2,4,5,7 + W326 — what about Rules 3 and 6,
  are they unreached or deferred?

## Cluster C — Events + simulator fixes (May 20, 23, 5 commits)

- `424b6a9a` fix(sim): Gillespie re-evaluates rates that depend on bare `t`
  (+159 −11) — propensity-cache invalidation
- `e9ea277c` fix(events): fire always_active events under ode/tau_leap/gillespie (gh#67)
- `07394aff` fix(events): parametric `at [param]` schedules honor the parameter (gh#69) — touches IR schema + every backend (20 files)
- `768423a1` fix(expander): default output schedule covers negative t_start (snap_at obs-only bug)
- `1fd135ee` feat(sim): smooth-importation seed mechanism + identifiability test (+972)
- `71ef2678` docs(proposal): record seed-timing mechanism-B implementation + bug fix
- `dc225cb2` docs(proposal): seed-timing inference for early-outbreak models

**Review focal points:**
- gh#67: "always_active" events under three backends — was the bug
  silent, and does the test (events_backend_parity) actually drive
  divergent traces?
- gh#69: parametric `at [param]` — IR schema change implies an
  ir_version bump; confirm one happened (this commit predates the 0.6
  bump in `6b9a9d96`)
- Gillespie bare-`t` fix — Gillespie samples next-event from a
  homogeneous Poisson; if a rate depends on `t` you can't reuse the
  draw across a discontinuity. Check the re-evaluation policy
- Smooth-importation seed (`1fd135ee`) — what is the proposal's
  "mechanism B"? Does the identifiability test actually fail without
  the mechanism?

## Cluster D — Profile + per-cell diagnostics + survey_top_k (May 23–24, 12 commits)

- `26657cd3` feat(profile): add PMMH as a per-cell algorithm (+412)
- `f52d1ecd` fix(profile): clean-eval re-pass on IF2 path so mle.toml reports loglik at saved MLE
- `4c1d791b` gh#51 v2 follow-up: align gh#51 comments with v2/v3 split
- `460d27d3` gh#51 v2: PMMH supports `init_method = "survey_top_k"` (+461)
- `f1d61d58` gh#51 v2: PGAS supports `init_method = "survey_top_k"` (+415)
- `713c3c80` refactor(fit): extract `resolve_per_chain_starts_from_method` for survey_top_k v2 (+287 −46)
- `ff12499a` gh#74 Option B: failing integration tests for per-cell diagnostics (+469)
- `b6ef23e8` gh#74 Option B: per-cell diagnostics infra + PMMH wiring (+830 −56)
- `edc89690` gh#74 Option B: wire IF2 per-cell diagnostic capture
- `1de0b301` gh#74 Option B: docs

**Review focal points:**
- `clean-eval re-pass` (`f52d1ecd`) — fixing a real bug where mle.toml
  reported the loglik *as inflated by parameter perturbations*; verify
  the fix doesn't accidentally re-run inference (just re-evaluates)
- PMMH-as-per-cell — does its config schema match IF2's per-cell schema?
- `resolve_per_chain_starts_from_method` — extracted from `pmmh.rs` and
  shared with `pgas.rs`; confirm the two callers actually pass the same
  shape (no quiet behavioral difference)
- `survey_top_k` initialization: what defines "top"? log-posterior?
  loglik only? prior-blind? Document explicitly
- Per-cell diagnostics (`b6ef23e8`) — 830-line addition; check the
  ESS / acceptance / log-evidence diagnostics are *surfaced* (per
  code-review.md §5)

## Cluster E — Prior precedence (May 24, 7 commits)

- `5f658a16` gh#73: honor priors in `profile --algorithm pmmh` (+696)
- `2c813abe` gh#73: integration tests + warning-text polish
- `94775bbd` gh#75: extend gh#73 prior-precedence chain to `fit run` / `fit where`
- `c4848c8f` gh#75: integration tests for fit-run prior precedence (TDD red) (+490)
- `d6847e71` gh#75: lift `profile_priors` → `fit/priors_precedence` for sharing
- `dd016f87` gh#75: wire three-tier prior precedence + explicit-flat opt-in for fit run (+382 −51)
- `24aef1a5` gh#75: docs — Priors and precedence + `camdl fit run --help`

**Review focal points:**
- Three-tier precedence — what are the three tiers exactly? CLI > config > model? Or model defaults < config < CLI? Document.
- "Explicit-flat opt-in" — a flat prior must be opt-in per code-review.md
  §6 ("No prior, no run"); confirm the default refuses to run without
  an explicit prior
- The lift (`d6847e71`) — pre-existing duplication is exactly the DRY
  smell the code-review prompt flags; check no remaining duplicate
  copies of the priors logic in profile.rs

## Cluster F — `[source.from_csv]` batch source (May 26, 1 commit)

- `d3ebe965` feat(batch): `[source.from_csv]` — one run per row of an external CSV/TSV
  - `rust/crates/cli/src/batch.rs` +373
  - `docs/camdl-run-spec.md` +18

**Review focal points:**
- Path handling — relative to config file or CWD?
- Type coercion of CSV cells into TOML-like values — what happens
  with strings that look like numbers? With empty cells?
- Header validation
- Interaction with parameter sweep (does each row override CLI
  arguments? batch table?)

## Cluster G — Smaller items
- `e6f52e26` gh#77: install.sh PREFIX env var
- `0ba5b15c` gh#71 stuck-chain LHS warning (docs only — proposal)
- `2c4f2d27` docs(forcing): piecewise-linear pattern (golden + table)
- Various proposal-only commits (no code impact)

## How findings will be reported

Per `docs/dev/code-review.md` §"How to report findings":
each finding gets Location / Category / Defect / Why-it-matters / Fix /
Severity. Critical+High first, by likelihood not by file.

The audit prompt is explicit that brevity beats exhaustiveness; the
final report will pick correctness over style.
