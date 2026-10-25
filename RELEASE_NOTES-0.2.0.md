<!--
MAINTAINER NOTES (delete before publishing)

(a) RECOMMENDED SEMVER BUMP: 0.2.0 (MINOR), per VERSIONING.md.
    The range v0.1.0-alpha..HEAD contains 155 `feat` commits AND several
    breaking changes (`feat(ir)!`, `refactor(ir)!`, `refactor(cli)!`,
    `refactor(fit)!`, plus the DSL observation-block reshape and the
    `--params`/`--starts-from` CLI removal). Pre-1.0, any breaking change OR any
    feat drives a MINOR bump — both apply, so 0.2.0. NOTE the tooling caveat:
    the only existing tag (v0.1.0-alpha) is a PRERELEASE, so
    `git-cliff --bumped-version` will continue it to v0.1.0-alpha.1, NOT pick
    0.2.0 — you must finalize the line manually (tag 0.2.0 directly, or first
    drop the -alpha). This is the FIRST published release: v0.1.0-alpha was
    tagged 2026-05-15 but never published, so these notes fold in everything
    since alpha and read as "here is camdl 0.2.0," not a thin delta.

(b) OPEN QUESTIONS / JUDGMENT CALLS FOR THE MAINTAINER:
    1. LEAD ORDER. I led Highlights with inference (the methods + backends),
       since that's the differentiating capability for an epidemic-modelling
       toolchain. If you'd rather the first thing a new reader sees is the DSL
       (the authoring experience), reorder the first two highlight bullets.
    2. [RESOLVED — maintainer: promote] `fit predict` is now a Highlight
       ("Posterior-predictive checks and forecasts"). It shipped the
       FreeForward × Posterior slice (5def3b11); broader param treatments /
       horizons are in flight, so the wording stays modest.
    3. `mh on ode` (deterministic Metropolis-Hastings on the ODE backend,
       590e80da) is "Phase 1." I listed it under Inference & engine, not as a
       headline. Confirm it's stable enough to advertise at all, or cut it.
    4. REACTIVE INTERVENTIONS are PARSED + VALIDATED but RUNTIME-REJECTED outside
       forward chain-binomial (Gillespie/ODE/all inference reject with a
       capability error; gh#204). I described it honestly as "forward-only,
       chain-binomial-only, experimental." Decide whether to mention it at all in
       a first release or hold it for when inference support lands.
    5. CALENDAR TIME / typed-time (dated I/O, `--dates`, instant/duration kinds):
       this shipped across several feats and is genuinely useful — but it's a
       large surface. Confirm the one-line framing is accurate to what a user can
       actually do end-to-end today (I describe the dated-data loader + calendar
       output columns + date literals in `at` schedules, all of which have feats).
    6. CHECK-UPDATE (`camdl check-update`, 0b6278af) queries GitHub for a newer
       release. Since this is the FIRST published release, the update check has
       nothing to find until a second release exists — fine to ship, just noting
       it's inert until v0.2.1+.

(c) COULD-NOT-CONFIDENTLY-CLASSIFY / PROPOSAL-NOT-SHIPPED watch list:
    - LINEAGES (line lists + transmission trees): there are MANY feats
      (`simulate --lineages`, `lineage tree/realize`, event-log layer, sampling
      layers). It's clearly implemented as a forward-simulation feature. BUT the
      backends it tracked on include tau-leap, which was later REMOVED in this
      same range — so verify the lineage path still works on the surviving
      backends (chain_binomial / gillespie / ode) before headlining it. I placed
      it under "Language (DSL)" + "Engine" conservatively, not in Highlights.
    - HIERARCHICAL PRIORS: NOT shipped as a usable feature — the only commit is
      `fix(sim): refuse hierarchical priors in PGAS+NUTS` (b58f3275). This is
      v0.3 in-progress (per the phase table). I deliberately EXCLUDED it.
    - `camdl if2` SUBCOMMAND [RESOLVED]: it is an INTENTIONAL deprecation stub
      (gh#147 M3.3, 85250cc3) — `crates/cli/src/if2.rs` accepts/ignores args and
      prints a migration message pointing to `camdl fit run`, exit 2. This is the
      "signpost the migration" pattern, not dead code. Keep it for 0.2.0; retiring
      the stub (→ bare clap error) is a post-alpha call. Notes don't reference the
      subcommand, so no notes change.
    - GENERATED QUANTITIES / counterfactual contrasts / cloud fit-dispatch /
      grouping-dimensions: these appear ONLY as `docs(proposals)` in the range
      (no implementing `feat`). EXCLUDED — proposals, not shipped.
-->

# camdl 0.2.0 — YYYY-MM-DD

This is the first published release of camdl: a toolchain for stochastic
compartmental epidemic modelling, from a human-readable model DSL through
simulation to full Bayesian and maximum-likelihood inference on real
surveillance data. It is **alpha** software — the surface is documented but
breaking changes are still expected before 1.0.

## Highlights

- **Production inference on real data:** PGAS+NUTS for Bayesian posterior
  sampling (gradient-based, using compiler-emitted analytic derivatives), IF2
  for maximum-likelihood estimation, and PMMH — plus a standalone bootstrap
  particle filter for likelihood evaluation.
- **Posterior-predictive checks and forecasts:** `camdl fit predict` replays a
  fit forward from the posterior and writes predicted-vs-observed tidy
  artifacts, for calibration checks and projection beyond the data.
- **Three simulation backends — chain-binomial, Gillespie, and ODE — with an
  adaptive RK4(5) integrator** for long-horizon ODE fits; every backend × method
  combination either works or fails loudly through the capability system.
- **A readable, dimensionally-typed DSL:** units and dimensions are checked at
  compile time (a missing `/N` is a hard error, not a silent bug), with clear
  diagnostics that name the parameter, the location, and the fix.
- **Calendar time end-to-end:** load dated surveillance data, write calendar
  columns in output, render estimands as dates, and use date literals in
  scheduling — the engine works in internal time and translates at the I/O
  boundary.
- **Sparse, irregular, and conditioned observations:** fit data with missing
  values (`NA`) and irregular sampling, with a conditioning window to discard a
  burn-in span — validated against pomp's particle filter to within Monte-Carlo
  error.
- **Content-addressed runs by default:** every simulate/fit/profile run is keyed
  by its inputs and stored in a cache, so identical re-runs are instant and
  every artifact is traceable; browse them with `camdl list` / `show` / `cat`.
- **Self-documenting binary:** `camdl docs <topic>` ships version-matched usage
  guides offline, and `camdl mre` packages a minimal reproducible example for
  bug reports.

## Breaking changes

These are relative to `v0.1.0-alpha`. Each carries an old → new migration.

- **Observation block reshaped (gh#171).** The measurement model is now written
  with `~` (as priors are), data binds **by name** via a required `columns {}`
  block, and the emission cadence is renamed `emit_schedule` (simulate-only).

  ```camdl
  # OLD
  cases : {
    projected  = incidence(infection)
    every      = 7 'days
    likelihood = neg_binomial(mean = rho * projected, r = k)
  }
  # NEW
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days        # simulate-only; omit for a fit-only model
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
  ```

  Diagnostics name each migration: `likelihood = …` → `<col> ~ …` is E273; the
  stream-header colon removal is E270; `every` → `emit_schedule` is E272.

- **`--params` and `--starts-from` removed from inference commands
  (gh#83/gh#85).** Chain start and fixed parameters are now unified under
  `--init <mode>` and `--fixed NAME=VALUE`. Migration: replace `--params p.toml`
  with the appropriate `--init from_params` / `--init from_mle` mode, and a
  held-fixed parameter with `--fixed NAME=VALUE`.

- **`scope` key removed from `reactive_interventions {}` (gh#204).** The
  `scope = exogenous | particle` key is gone; a reactive trigger always reads
  reported surveillance. Migration: delete the `scope = …` line (`exogenous` is
  now implicit). Old form → E106.

- **`output {}` phantom sub-blocks removed.** `summary {}`, `flows {}`,
  `synthetic {}`, and the experiment/compare sub-blocks never did anything and
  are now errors (E106). Migration: delete them; trajectory cadence and format
  are set on `output {}` directly.

- **Strict dimensions on likelihood arguments (gh#116).** A count where a
  probability is required (`binomial(p = projected)` with a count `projected`)
  is now rejected (E304). Migration: pass a proportion — `p = projected / N`.

- **Synthetic-data backend key moved (gh#241).** `[config].backend` →
  `[synthetic].backend` in `fit.toml` (it only ever governed synthetic-data
  generation).

- **Run-output layout changes (CAS, gh#147/gh#241).** Simulate now writes to the
  content-addressed store by default instead of streaming TSV to stdout (use
  `--stdout` to opt out); synthetic-fit and design-experiment output paths
  relocate under the runid digest. Scripts that globbed the old `<output>/…`
  trees should switch to `camdl list` / `cat`, or `--stdout`.

- **The tau-leap backend was removed.** Models are simulated and fit on
  chain-binomial, Gillespie, or ODE. Migration: pick one of the three surviving
  backends (chain-binomial is the closest stochastic analogue).

- **PGAS parallel tempering was removed (gh#789).** The `tempering = [...]` key
  under `[method]` for `algorithm = "pgas"` and the `--tempering` flag on
  `fit run` are gone; every PGAS chain samples the posterior directly, as the
  ladder's cold rung did. Migration: delete the `tempering` line and drop the
  flag. A `fit.toml` that still carries the key is refused with that
  instruction. The `LowSwapRate` diagnostic goes with it.

- **Correlated pseudo-marginal PMMH was removed.** The `rho` key under
  `[method]` for `algorithm = "pmmh"`, the `--rho` flag on `fit run` and the
  `--pmmh-rho` flag on `profile` are gone; `pmmh` runs the bootstrap particle
  filter with independent likelihood evaluations, as it did with `rho` unset.
  Migration: delete the `rho` line and drop the flags. A `fit.toml` that still
  carries the key, or a script that still passes either flag, is refused with
  that instruction.

## Run identity

- **Every stored PGAS fit, every stored PMMH fit and every stored
  `camdl profile` run re-keys.** `tempering` was part of the PGAS method
  identity and `rho` of the PMMH method identity, and `camdl profile` hashed its
  PMMH knobs, `rho` included, into the method level of every run whatever its
  algorithm. Removing the two fields is one identity change: a new PGAS or PMMH
  fit, or a new profile run of any algorithm, does not deduplicate against one
  stored before this change, and an in-flight PGAS or PMMH fit cannot `--resume`
  across it (its saved config hash no longer matches). Stored results stay
  readable; the fits and profiles run again.
- **Every stored multi-cell `simulate` ensemble re-keys once.** The ensemble
  leaf (the combined `ensemble.tsv` of a `--replicates` / `--seeds` /
  multi-`--scenario` / `--draws` run) is now keyed on its cells' identities
  alone — per cell the parameter-point index, the process seed and the cell's
  own `run_id` — plus the run's resolved design; the arm's display label has
  left the key. Every stored multi-cell ensemble is therefore a miss once and is
  rebuilt from the per-cell `Sim` leaves the store already holds, so the cost is
  aggregation, not simulation. Two ensembles of identical cells now share one
  identity whether an arm was named by a preset or derived from
  `--enable`/`--disable`/`--param`, and renaming a preset no longer re-keys the
  ensembles that used it.

## Language (DSL)

- **Restricted sums for sparse/spatial coupling (gh#185):**
  `sum(... where P, ...)` lets a coupling term range over only the pairs a
  predicate selects — the basis for fittable spatial-coupling kernels.
- **`positive` / `real` parameter kinds accept a unit literal (gh#60):**
  `tau : positive 'ratio`, `iota : positive 'count` — resolves the "dimension
  could not be determined" info and makes dimensional misuse a hard error.
- **Calendar / typed time:** instant-vs-duration parameter kinds with a numeric
  origin, calendar primitives and `date_range`, ISO dates accepted in
  instant-kind tables and `at` schedules.
- **`dt` in the `simulate {}` block (gh#161)** and an optional tagged
  `integrator` (`rk4` default, or `rk45 { atol rtol }` for adaptive stepping).
- **`min(a, b)` / `max(a, b)`**, `incidence()` over a stratified transition
  family summing strata, and let-bound identifiers resolving inside `projected`.
- **Forcing/table coefficients are live, estimable parameters (gh#119):** a
  parameter inside a sinusoidal/Fourier coefficient or an inline table is now
  evaluated live during inference (previously frozen — a silent flat
  likelihood), with analytic gradients so they're estimable under NUTS.
  Structural data (interpolation knots, step grids) as a parameter is now a
  clear compile error.
- **Lineage tracking (`#[lineage]`):** forward simulation can emit line lists
  and reconstruct transmission trees offline (`camdl lineage tree`), with
  per-individual and stratified sampling.
- **Reactive interventions (`reactive_interventions {}`, gh#204) — experimental,
  forward chain-binomial only.** State/observation-triggered policies
  (`when sum_observed(stream, window = D) >= threshold`) fire at
  run-time-discovered timing. Gillespie/ODE and all inference paths reject an
  active reactive policy with a capability error.
- **New prior distributions (gh#155):** `log_uniform(lower, upper)`,
  `truncated_normal(mean, sd)`, and `prior = { uniform = {} }` (uniform over the
  parameter's declared bounds).
- **`#'` declaration doc-comments** annotate a parameter through to the fit
  report; a model linter pass flags dead compartments (L402).

## CLI

- **Content-addressed run management:** `simulate` / `batch` / `fit` / `profile`
  / `survey` write to a keyed store; browse with `list`, `show`, `cat`,
  `reindex`, and `compare` (prequential elpd / CRPS / PIT).
- **`camdl docs <topic>`** — embedded, version-matched usage guides (workflow,
  fit-toml, concepts, diagnosing-fits, language, language-changes) served
  offline.
- **`camdl mre simulate` / `mre fit`** package a minimal reproducible example
  (model + data + config) for sharing; `camdlc --emit-deps` emits the compile
  read-closure (gh#212).
- **`camdl check-update`** queries GitHub for a newer release (gh#205 also ships
  a no-sudo, user-native installer).
- **`fit predict`** writes predicted-vs-observed posterior-predictive artifacts.
- **`camdl eval`** evaluates time-dependent expressions against a model;
  `simulate --draws`, `simulate --integrator rk4|rk45`, and `--dates` calendar
  columns in output.
- **Trajectory output view:** `--output-every` / `--no-flows` / `--columns`
  (gh#156); `--parallel N` caps the thread pool on the inference commands.
- **Unified chain-start surface:** `--init <mode>` and `--fixed NAME=VALUE`
  across `fit` / `pfilter` / `profile` (see Breaking changes).
- Better diagnostics throughout: structured `--json-errors`, a method-caveat
  banner driven by the registry, and warnings for single-init multi-chain fits
  (gh#71) and origin-less instant parameters (gh#103).

## Inference & engine

- **PGAS+NUTS** Bayesian sampling with compiler-emitted analytic gradients
  (`autodiff.ml` source-to-source differentiation — no runtime autodiff, no
  finite differences), including obs-model and overdispersion-parameter
  gradients; per-sweep NUTS diagnostics in the trace (gh#294).
- **IF2** maximum-likelihood estimation; **PMMH** with structural-error
  surfacing; **bootstrap particle filter** with a `--pf-health` ESS/τ²
  diagnostic and a deterministic substep-cap watchdog.
- **Deterministic Metropolis-Hastings on the ODE backend** (`mh on ode`,
  experimental Phase 1) with a deterministic dt-check.
- **Sparse / irregular / multi-cadence observations and a conditioning window**
  (#218): `NA` holes are skipped in the likelihood (incidence bins still reset
  on cadence); validated against pomp's particle filter on a holed measles
  series to within Monte-Carlo error.
- **Real-valued ODE flow** recorded directly (not rounded counts); ODE incidence
  unified onto an augmented-flow formulation (gh#166); adaptive RK4(5)
  integrator for long horizons.
- **Parallelism:** PGAS CSMC particle loops parallelize across cores (gh#209,
  ~3.4× on 8 / 4.7× on 16 end-to-end); buffered trajectory writes (~2.6×).
- Numerous correctness fixes to the inference math (see Fixed).

> Not yet supported: hierarchical / pooled priors are rejected under PGAS+NUTS
> (gh#175) and remain in-progress for a later release.

## Formats & compatibility

- **IR schema `ir/VERSION` advanced 0.4 → 0.19** over this range. An IR-schema
  bump is a compatibility event independent of the release version: previously
  serialized `.ir.json` files may not load against this binary — recompile your
  `.camdl` source. Notable bumps include the compact serialization format (one
  IR element per line — 4.6× faster compile, ~5× smaller IR), forcing/table
  coefficients stored as expressions (gh#119), the tagged integrator (gh#166,
  0.14→0.15), the reactive `fire` source (gh#204), and live binding/reduction
  nodes.
- **`fit.toml`** now rejects unknown keys (typo'd keys are an error, gh#173),
  and `[config].backend` moved to `[synthetic].backend` (see Breaking changes).
- **Output:** content-addressed store layout is the default (see Breaking
  changes); fit output gains a `fit.meta.json` sidecar, a doc dictionary in the
  IR envelope, per-run progress/liveness heartbeat artifacts, and a
  round-trippable parameter format in MCMC traces.

## Fixed

User-visible correctness fixes (selected from ~138):

- **Inference math:** missing terms in the PGAS+NUTS energy for gamma
  multipliers (gh#197) and deterministic source-less inflow (gh#200); corrected
  `log_normal` NUTS gradient bias (gh#155); coherent counts across CSMC-AS joins
  in saved trajectories (gh#264); prequential `y_obs` now records the real
  observed value, not 0 (gh#268).
- **Dimensions/compiler:** prevalence-as-proportion observation models now
  type-check (the `projected` keyword carried a spurious population dimension);
  negative parameter lower bounds preserved; structural errors surfaced instead
  of a misleading divide-by-zero (gh#81).
- **Backends:** Gillespie re-evaluates rates that depend on bare `t`; absorbing-
  state and sparse-propensity edge cases hardened (gh#70, gh#208); real-
  compartment models gated out of inference to prevent a frozen-reservoir
  mis-fit (gh#191); negative / non-finite intervention values rejected instead
  of a silent cast.
- **Forcing:** forcing/table coefficients evaluated live during inference, not
  frozen at construction (gh#119) — a previously silent flat-likelihood bug.
- **CLI / identity:** the IR cache invalidates when a `read()`-loaded data file
  changes (gh#260); `--table` content and `init_method`/survey folded into run
  identity (gh#147); robust lock-reclaim for interrupted fits.
- **Robustness:** clean errors (not panics) for empty observation data and
  degenerate IF2 chains; non-increasing / sub-dt-colliding observation times
  rejected at construction (gh#188).

## Internal / docs / CI

- Major engine consolidation: a single merged-timeline `Schedule` spine that all
  backends and the inference filters route through, a single boundary authority
  (gh#233), unified effect/event/intervention application, and consolidated
  Tier-1 inference forks — the substrate that keeps the backend × method matrix
  from silently diverging.
- Front-end pipeline unified into one core with structured non-raising compile
  outcomes and source-located diagnostics (gh#181, gh#170).
- Compiler-verified documentation (a doctest gate compiles the spec's code
  blocks) and a CLI-doc drift gate; an embedded `camdl docs` topic system.
- Per-area CI workflows, a Conventional-Commit changelog spine
  (`make
  changelog`), a `RELEASING.md` runbook, and a versioning policy.

Full changelog: `v0.1.0-alpha..v0.2.0`
