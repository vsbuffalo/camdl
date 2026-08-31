# Particle-filter wrapper audit: validating an external performance review

An external review of the particle-filter code
(`rust/crates/sim/src/inference/particle_filter.rs` and neighbors) proposed
thirteen changes, mostly allocation and layout fixes around — not inside — the
per-particle propagation step. This audit checks each claim against the code
and, more importantly, against where camdl's fitting time actually goes, using
existing profiling notes (`docs/dev/notes/2026-06-01-pf-eval-profiling.md`,
`docs/dev/notes/2026-06-13-pgas-parallelism-and-scaling.md`) rather than
re-deriving that from scratch.

**Bottom line.** Three of the review's changes are verified, low-risk, and —
this is the part the review didn't establish — confirmed to sit on the actual
per-iteration hot loop of default (uncorrelated) PMMH, IF2, and PGAS, not just
on standalone diagnostics. One of the review's proposed simplifications
(deriving the prequential joint score by summing the per-stream scores, dropping
the joint storage) is unsafe as written and would silently miscompute the
log-likelihood for multi-cadence models at a partial-hole observation step; it
must not be implemented as described. The single highest-expected-value item —
gating the predictive-diagnostics block behind an explicit request — turns out
to already be filed as gh#520, with the exact call chain that makes it matter
for a real fit already worked out below.

## What "the particle filter" means here — four separate engines, one shared substrate

The review reads as though `bootstrap_filter` (`particle_filter.rs`) is _the_
particle filter camdl uses for fitting. It is not. Four inference engines each
carry their own particle-propagation loop:

| engine                                                                    | file                                                                                            | own `par_iter_mut` loop |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------- |
| bootstrap PF (`camdl pfilter`, `survey`, `fit predict`, preflight checks) | `particle_filter.rs`                                                                            | line 339                |
| PMMH (uncorrelated, ρ = 0 — the CLI default)                              | `pmmh.rs` → `fit/runner.rs::run_quick_pfilter_with_dt` → `particle_filter.rs::bootstrap_filter` | (reuses bootstrap PF)   |
| PMMH (correlated, ρ > 0)                                                  | `correlated_pf.rs`                                                                              | line 922                |
| IF2 (iterated filtering)                                                  | `if2.rs`                                                                                        | line 562                |
| PGAS (particle Gibbs with ancestor sampling, the default Bayesian method) | `pgas.rs::csmc_as`                                                                              | lines 2453, 2652        |

This matters because the review's headline claim ("PMMH and PGAS discard the
prediction block") is only true for _correlated_ PMMH and for PGAS —
`correlated_pf.rs` and `pgas.rs::csmc_as` are separate implementations that
never compute a prediction block in the first place. **Default, uncorrelated
PMMH is different**: `pmmh.rs`'s per-step likelihood evaluation (`eval_loglik`,
used whenever `rho` is unset — the JSON config default is `"rho": null`) calls
`run_quick_pfilter_with_dt`, which calls `bootstrap_filter` directly. So the
prediction-block cost the review flags is real, and it lands on the single most
commonly used fitting path in the codebase, on every proposed θ, every
Metropolis-Hastings step, every chain.

## Ranked findings

1. **Gate the prediction-diagnostics block behind an explicit request, not the
   observation model's shape.** High. `particle_filter.rs:194-208, 382-407`.
   `has_predictions` is true whenever
   `obs_model.n_streams() > 0 && !obs_model.mean(...).is_empty()` — and the only
   production `ObservationModel` implementation, `MultiStreamObsModel`
   (`multi_stream_obs.rs:1574`), always overrides both `mean()` and `sample()`,
   so `has_predictions` is true for essentially every fitted model, not a corner
   case. When true, the block does one `obs_model.mean()` + one
   `obs_model.sample()` (a real RNG-consuming draw — for BetaBinomial/Beta
   observation distributions this is a rejection-sampled Gamma draw, not a cheap
   closed-form one, per `obs_model.rs:575`) per particle, plus two O(N log N)
   sorts, at every observation. `pmmh.rs`'s uncorrelated path (`eval_loglik` →
   `run_quick_pfilter_full` → `run_quick_pfilter_with_dt`, `fit/runner.rs:777`)
   discards everything but `.log_likelihood`. This is filed as **gh#520**
   ("perf: bootstrap_filter computes the full PredictionDiag block on every
   eval; pmmh/pgas discard all of it," effort/S) with a concrete count for one
   real model: 1,200 particles × 11 observations × 5 streams ≈ 13,200 predictive
   draws per filter evaluation, ~8×10⁷ over a full fit. gh#520's title
   overstates PGAS's exposure (PGAS never calls `bootstrap_filter` at all — see
   the table above) but understates that this is _default_ PMMH's main loop, not
   a side path. Independent: lands alone. Disposition: **fixed 2026-08-31**
   (`SMCConfig.record_predictions`, off by default; commit `22b7348f`).
   Benchmarked on a real national-scale fit (uncorrelated PMMH, 19,200
   particles, 4 streams): **~3.0× on `bootstrap_filter`'s per-evaluation cost**
   (2118–2160 ms/eval before → 696–723 ms/eval after, two seeds), MAP
   log-likelihood unchanged — see "How much faster" below for the full
   measurement. gh#520 itself flagged the missing profile ("I have not profiled
   it... that measurement should be part of the fix"); this is it.

2. **Reuse allocation-free weight/resampling workspaces across all four
   engines.** High. `types.rs:444-490` (`logw_variance`,
   `normalize_log_weights`) and `resampling.rs:18-119` (`systematic_resample`,
   `systematic_resample_core`, `conditional_multinomial_resample`) each allocate
   one or more fresh `Vec<f64>`/`Vec<usize>` per call, sized N. Confirmed
   shared, not duplicated, across every engine: `correlated_pf.rs:1148-1150`
   calls the same `normalize_log_weights` / `systematic_resample_core` (via a
   thin `systematic_resample_fixed_u` wrapper for common-random-numbers, CRN,
   coupling); `if2.rs:615,665,676` calls `normalize_log_weights`,
   `ess_from_log_weights`, `systematic_resample` directly; PGAS's
   `conditional_multinomial_resample` (`pgas.rs:2418`, via
   `resampling.rs:86-119`) itself calls `normalize_log_weights` and allocates
   two more N-sized buffers (`cdf`, `out`) internally. A fix here is a single
   surgical change in `types.rs`/`resampling.rs` that benefits all four engines'
   actual per-step/per-sweep loop, not a wrapper-only fix. Independent: lands
   alone. Disposition: **not yet filed — worth a `gh issue create`.**

3. **Stop copying `flow_accumulators` into the resample buffer only to zero it
   immediately after, and delete the now-dead log-weight reset.** High, but be
   careful in one file. `particle_filter.rs:541-547` copies `flow_accumulators`
   during the resample buffer-swap; `particle_filter.rs:585-588` unconditionally
   zeros it ~40 lines later, and nothing reads it in between (directly verified:
   the fold into `acc`, line 378, and the ancestry projection capture, line 500,
   both read the _pre_-resample `flow_accumulators`, before the copy). The
   destination can be zeroed directly, skipping the copy. Confirmed the
   identical pattern in `correlated_pf.rs:1102-1106` (copy) / `:1112-1115`
   (zero) and `if2.rs:679` (copy) / `:691` (zero). **Caveat, checked and
   cleared**: `correlated_pf.rs:1077-1082` reads `flow_accumulators` as a
   resampling sort key for CRN coupling — but that read happens on the pre-copy
   _source_ array (`swarm.states`), before the resample swap, never on the
   post-copy destination, so the copy-elision is still safe there. Also
   confirmed independently: `particle_filter.rs:591`
   (`for lw in &mut swarm.log_weights { *lw = 0.0; }`) writes a value that is
   unconditionally overwritten before it is ever read — every weight is
   rewritten at line 415-417 of the next iteration regardless of death status,
   and nothing after the loop reads `swarm.log_weights` — so this loop is dead
   and safe to delete (verified for `particle_filter.rs`; not separately checked
   in the other three engines). Independent: lands alone, byte-identical to the
   current behavior. Disposition: **not yet filed — worth a `gh issue create`**,
   and low-risk enough to bundle with #2.

4. **Worker-local, not particle-local, process scratch.** Medium-high value,
   larger effort. `particle_filter.rs:171-173`, `correlated_pf.rs:632-634`,
   `if2.rs:377-379`, and `pgas.rs:2340-2341` all allocate one `P::Scratch`
   (`StepScratch` for chain-binomial) per particle, held for the whole run —
   confirmed in all four engines. `StepScratch` (`chain_binomial.rs:52-91`)
   carries several `Vec`s sized by transition count (`propensities`, `draws`,
   `pending_deltas` at 2×, `handled`, `probs`); for a large coupled model this
   is not small — the polio-scale reconstruction in
   `docs/dev/notes/2026-06-13-pgas-parallelism-and-scaling.md` used 11,713
   transitions, at which point back-of-envelope arithmetic on those five `Vec`s
   alone (not a measured RSS) is roughly 800 KB–1 MB per particle; at a few
   hundred to a few thousand particles that is hundreds of MB to low GB of
   scratch footprint that a worker-local design (one scratch per rayon thread,
   not per particle) would collapse toward (thread count) × (that same figure) —
   a potential order-of-magnitude reduction, **but this is an estimate from
   struct field sizes, not a measurement, and should be checked with an actual
   allocator/RSS profile before it goes into a proposal doc.** The fix itself is
   a real refactor: it changes `ProcessModel::step`'s calling contract from
   "your own scratch" to "a scratch shared with other particles in your chunk,"
   which requires auditing that no scratch field silently carries
   particle-scoped state across that boundary (`gamma_override`,
   `binomial_z_values`, `gamma_used` on `StepScratch` look purpose-built as
   _cross-call_ correlated-MCMC/CPM channels, not particle state, but that needs
   confirming, not assuming, before the change lands). Independent: can land
   after or alongside #2/#3, but touches the `ProcessModel` trait contract, so
   it is the one item here that crosses `chain_binomial_process.rs`'s and
   `particle_filter.rs`'s "read the full function before editing"
   high-risk-surface line. Disposition: **needs a design decision — see below.**

5. **Replace the per-particle `Vec<Result<bool, SimError>>` (or
   `Vec<Result<(), SimError>>`) collection with a direct death-mask mutation.**
   Medium, needs care. `SimError` measures **72 bytes** (reconstructed via a
   standalone compile mirroring its field shapes — not a `size_of::<SimError>()`
   printed directly from the crate, so treat as a strong estimate rather than a
   pasted command output), driven by the `NonFiniteChainStart` variant's nested
   `InitSource`. So a `Result<bool, SimError>` costs 72 bytes per particle per
   propagation window on the success path, instead of 1. Confirmed present in
   **five** sites, not the three a prior dev note counted:
   `particle_filter.rs:339-366`, `correlated_pf.rs:922`, `if2.rs:562`, and
   `pgas.rs:2453` _and_ `pgas.rs:2174` (the dev note undercounted PGAS by one).
   **Caution**: `DeathMask::absorb` (`degeneracy.rs:143-154`) folds `outcomes`
   in index order and propagates the first index-order `Err`, and a test
   (`degeneracy.rs:352-373`) pins that. A naive `rayon::try_for_each` rewrite
   does not guarantee the reported error is the lowest-index one when multiple
   particles fail in the same window — plausible but not documented as the
   reason this pattern was chosen (no commit found stating it explicitly), so
   any rewrite needs an index-order-preserving fold (e.g. a `try_fold` per chunk
   plus a serial reduce), not a bare short-circuit, to keep the current
   deterministic-error-reporting guarantee this codebase's reproducibility
   discipline depends on. Independent: lands alone per site; the five sites can
   be fixed in any order but should share one pattern. Disposition: **not yet
   filed — worth a `gh issue create`** with the determinism caveat written into
   the issue.

6. **Ancestor-trace flattening is a real but smaller win than described, and
   only fires when `record_ancestry`/`--save-paths` is requested.** Low-medium.
   `AncestorTrace` (`ancestor_trace.rs:35-65`) has `states: Vec<Vec<Vec<f64>>>`
   and `projections: Vec<Vec<Vec<f64>>>` — genuinely one allocation per particle
   per observation, as the review says. But `log_weights: Vec<Vec<f64>>` and
   `ancestors: Vec<Vec<usize>>` are **already** flat, one allocation per
   observation (`particle_filter.rs:504,561`) — the review's proposed struct
   "fixes" two fields that were never broken. Flattening `states`/`projections`
   to a row-per-observation layout is a real rewrite, not a type substitution:
   at least five consumer call sites index
   `[obs][particle]`/`[obs][particle][k]` directly
   (`ancestor_trace.rs::sample_paths`,
   `cli/pfilter.rs::write_filtering_tsv`/`write_paths_tsv`, and the module's own
   test helpers) and would need slice-arithmetic rewrites. One genuine, narrower
   correction the review gets right: `states` (integer compartment counts, cast
   to `f64` "for downstream real-valued arithmetic,"
   `particle_filter.rs:472-476`) could become `i64` with no consumer relying on
   its being float — but on a 64-bit target `i64` and `f64` are the same width,
   so this buys type-correctness, not memory. `projections`, by contrast, must
   stay `f64`: it carries a live `f64::NAN` sentinel for "stream not scheduled
   at this union index" (`particle_filter.rs:491-498`), emitted literally as
   `"NaN"` in TSV output (`cli/pfilter.rs:1508-1516`). Independent: lands alone;
   only matters for the opt-in ancestry/path-sampling path. Disposition: **needs
   a decision on whether the ~5-site rewrite is worth it for an opt-in feature —
   see below.**

7. **`_into` buffer-writing variants for
   `sample`/`mean`/`log_likelihood_per_stream` are low-risk but should be
   bundled, not done standalone.** Low. The trait methods (`traits.rs:151-179`)
   always return a `Vec` of exactly `n_streams()` length regardless of
   `obs_idx`; an unscheduled stream writes `f64::NAN`/`0.0` _in place_ rather
   than shrinking the vector (`multi_stream_obs.rs:1624-1629,1653-1656`), so a
   caller-supplied fixed-size buffer works without special-casing. This is a
   real but modest allocation reduction and is best implemented alongside #4/#5
   rather than as its own change.

8. **Do not implement the review's proposed lazy joint-from-per-stream
   derivation for the prequential trace — it is unsafe for multi-cadence
   models.** Correction, not a finding to act on. The review suggests dropping
   the joint `log_liks`/`y_pred_samples` fields on `PrequentialRecorded` and
   deriving them by summing the per-stream fields. This is unsafe as stated:
   `prequential.rs`'s own `build_trace` only reuses the recorded joint verbatim
   at _hole-free_ steps and explicitly **recomputes** it by filtered-summing the
   per-stream tensors at _partial-hole_ steps (`prequential.rs:345-358`) —
   because the per-stream and joint absence conventions differ
   (`multi_stream_obs.rs:1199-1210`: per-stream marks an unobserved stream as
   `NaN`; the joint sum treats an absent stream as contributing 0). A blind
   "derive joint from per-stream" implementation would silently double-count or
   miscompute the joint log-likelihood at any observation step where streams are
   on different cadences — exactly the multi-cadence models this project is
   actively building
   (`docs/dev/proposals/2026-06-10-multi-stream-multi-cadence-union-axis.md`).
   Per this repo's own stakes framing, a plausible-looking wrong likelihood is
   the worst outcome available; this item should be dropped from consideration
   as described. If deduplication is still wanted, it has to reimplement
   `build_trace`'s existing hole-aware reconstruction, which is real complexity
   that costs one extra `score_streams` pass per particle when
   `record_prequential` is on — not a pure win. Also worth noting while here:
   `sample()` genuinely _is_ drawn twice per particle per observation when
   `has_predictions && config.record_prequential` are both true (once at
   `particle_filter.rs:397` for predictions, again at `:441` for prequential —
   two separate calls to the same `diag_rngs[i]`, not one draw split two ways,
   despite an adjacent comment ("EXACTLY ONCE") that is accurate only about the
   prequential block's _internal_ joint/per-stream relationship, not this
   cross-block duplication). Fixing that changes diagnostic RNG-consumption
   order — a golden-affecting change under this repo's own discipline, not a
   free mechanical fix — and only fires when a user explicitly requests
   predictions and prequential together (the `fit ... pfilter` diagnostic stage,
   not the PMMH/IF2/PGAS fitting loop). Low priority; needs a decision, not
   urgent.

9. **`RealState::new` inside `ChainBinomialProcess::step`'s hot loop is real but
   currently near-zero impact.** Low. Confirmed structurally
   (`chain_binomial_process.rs:96`): one `RealState::new(n_real)` per particle
   per substep. For `n_real == 0` — a `Vec` constructed with zero elements —
   this is allocation-free in Rust (a dangling-pointer `RawVec`, no heap call),
   so it costs nothing for the common case. For `n_real > 0`,
   `docs/dev/incidents/2026-06-07-chain-binomial-stale-real-state.md` and
   `.claude/rules/sim-and-inference.md`'s capabilities table both confirm
   chain-binomial inference deliberately withholds the `REAL_COMPARTMENTS`
   capability for the general case (gh#191) — so real-compartment models mostly
   cannot reach this code path via chain-binomial inference today. Worth folding
   into the scratch bundle opportunistically alongside #4, not worth
   prioritizing on its own.

## Design calls

**Is the worker-local scratch refactor (#4) worth doing now, and does it need an
RSS measurement first?** Options: (a) measure current scratch RSS on a large
coupled model before committing to the refactor, to confirm the
order-of-magnitude estimate above and decide whether memory or wall-clock is the
actual constraint on the national-scale fits this project is scaling toward; (b)
do the refactor now on the strength of the structural argument alone. My
recommendation: (a) — this repo's own culture is "measure it, don't assume" (the
τ² example in the PGAS-scaling note is the model to follow), and #4 is the one
item here that touches the `ProcessModel` trait contract across four call sites,
so getting the number right first is cheap insurance against a refactor that
turns out not to matter. Confidence: **leaning** — the structural argument (one
scratch per particle, sized by transition count, held for the whole run) is
solid, but whether it's actually the binding constraint anywhere today is an
empirical question this audit didn't answer.

**Is the ~5-site ancestor-trace rewrite (#6) worth doing given it only benefits
the opt-in `--save-paths`/`record_ancestry` path?** Options: (a) do it as part
of a broader pass whenever `ancestor_trace.rs` or its consumers are next touched
for a feature reason; (b) file it and leave it, since it's
memory/allocation-count, not correctness, and the feature is opt-in. My
recommendation: (b) — it's real but the lowest-leverage item here relative to
its rewrite cost, and the codebase's own convention is not to land refactors
without a wired reason. Confidence: **leaning**.

**Should items #2/#3/#5 be filed as one issue or three?** They're independent
and separately landable, but small enough that one issue covering all three
("particle-filter shared-substrate allocation cleanup: weight workspace reuse,
resample copy-elision, death-mask sizing") with three sub-checkboxes may be more
useful than three effort/S issues. My recommendation: one issue, since they'll
likely be implemented and reviewed together (same files, same risk profile).
Confidence: **need you** — this is a workflow preference, not something the
evidence settles.

## How much faster, how much memory — with the hedges the evidence actually supports

Two different currencies are being conflated in "how much faster can the PFs
get," and the evidence separates them:

**Default (uncorrelated) PMMH's per-step wall-clock — finding #1, gh#520 — now
measured, not just argued.** Fixed 2026-08-31 (`record_predictions`, off by
default) and benchmarked before/after on a real national-scale fit:
`bvd_national_delay_od_lab_direct_sum.camdl` against the vendored ebola-bdbv
case/death/lab streams, uncorrelated PMMH, 19,200 particles, 4 observation
streams, 85+ observations, 30 PF evaluations at fixed proposal (no adaptation in
play), single chain, two seeds. Same binary build, same machine, only the git
commit differs (`ba3c3aa8` before vs `22b7348f` after); camdl's own per-run
summary line reports the mean cost directly:

| seed | before (ms/eval) | after (ms/eval) | speedup |
| ---- | ---------------- | --------------- | ------- |
| 1    | 2117.6           | 695.5           | 3.05×   |
| 2    | 2160.0           | 722.7           | 2.99×   |

MAP log-likelihood matched exactly between before/after at each seed (−1348.7
for seed 1, −1348.5 for seed 2) — the fix changes only wall-clock, not the
statistical output, confirming the diagnostic-RNG separation held. **~3.0× on
default PMMH's per-evaluation cost for this model class**, not the 1.2–3× guess
this section originally hedged — `obs_model.sample()`'s per-particle draw
(BetaBinomial/NegBinomial rejection sampling across 4 streams) turned out to be
a larger share of `bootstrap_filter`'s own loop than the process substep
propagation for this model. Reproduction: `docs/dev/notes/` entry pending;
commands and scratch fit.toml available on request.

**Large, coupled, national-scale models — the regime IF2/PMMH/PGAS actually
struggle with today.** Here the evidence argues against a large wall-clock win
from this review's changes. The one real profile that exists for a
densely-coupled model (P=16 patches × A=7 age groups,
`docs/dev/notes/2026-06-13-pgas-parallelism-and-scaling.md`) measured 62.7% of
thread-samples in `eval_resolved` (rate-tree evaluation) and 24.4% in the
binding-cache thread-local protocol — 87% of the loop, in a category this review
does not touch at all — leaving well under 13% for everything else combined,
including resampling, weight normalization, and the death mask. That fraction is
expected to grow, not shrink, as coupling width increases
(`docs/dev/notes/2026-06-01-pf-eval-profiling.md`: "climbing toward 100% as P
grows"). So for exactly the models this project's own memory notes flag as the
current frontier (national-scale inference, spatial basin-finding), expect items
#2/#3/#5 to buy a modest, single-digit-to-low-teens percentage off end-to-end
wall-clock, not a multiplier — Amdahl's law bounds it, given how little of the
loop budget they occupy in that regime.

**Memory is the more promising currency for the coupled-model regime.** Item #4
(worker-local scratch) is a footprint reduction, not primarily a speed one — the
arithmetic above suggests a potential order-of-magnitude cut in per-particle
scratch RSS for large-transition-count models at typical production particle
counts, collapsing (N particles) × (scratch size) toward (rayon thread count) ×
(scratch size). That doesn't make a given fit run faster; it raises the particle
count or model size that fits in memory at all before the process is killed — a
different, and for this project's stated national-scale ambitions probably more
valuable, kind of win. It needs the RSS measurement flagged in the design-calls
section before it's sized precisely.

**Net recommendation, in order:** #1 (gh#520) is now shipped and measured at
~3.0×; next #2/#3/#5 together as one shared-substrate cleanup (safe,
cross-engine, but expect modest end-to-end percentages on coupled models); then
decide #4 on the strength of an actual RSS measurement rather than the estimate
here. Do not implement #8 (lazy joint derivation) as the external review
describes it.
