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

- **`perturb_only_at_t0` is refused under every method but `if2`.** It is IF2's
  perturbation schedule; `pgas`, `pmmh`, `mh` and the `nl-*` optimizers perturb
  nothing, and used to load the flag and read it nowhere. Migration: keep the
  flag in the `if2` config only and delete it from the others. `rw_sd` is still
  accepted everywhere: `pmmh` and `mh` read it as their initial proposal scale.
- **A Bayesian fit that estimates a hyperparameter over held hierarchical leaves
  is refused.** Under `pmmh`, `mh` and `pgas`, a leaf in `[fixed]` (e.g.
  `gamma ~ log_normal(mu = mu_g, sigma = 0.1)` with `gamma` held and `mu_g`
  estimated) contributed no prior term, so `mu_g`'s posterior was its prior.
  Migration: estimate the leaves too, or hold the hyperparameter.
- **`profile --sweep` values outside the parameter's bounds are refused**,
  naming the parameter, the values and the bounds (the model's, narrowed by
  `--fit` `[estimate]`). Migration: sweep inside the bounds.
- **`profile --rw-sd auto` estimates fewer parameters**: never a name from
  `--fixed-file`, and never a model constant (`let omega : rate = 0.01`). Such
  profiles, and profiles whose fit config declares `[[simplex_groups]]`, re-key
  and recompute. Migration: none; list a name in `--rw-sd` to estimate it.
- **`survey --fit` honours `--fixed`.** A pinned `[estimate]` name is held and
  no longer sampled, so the landscape loses that column and re-keys. Migration:
  drop `--fixed` to sample it again.

- **Coincident event drains that could overdraw a compartment are refused.**
  Among the `events {}` due at one instant, two or more `transfer`s from one
  compartment are an error when at least one moves a `count`, and fraction-only
  `transfer`s are an error when their fractions sum to more than
  1. Two `transfer(count = 6, ...)` from `X = 10` used to leave `X = -2`; the
     ODE reported that as an intervention error, but chain-binomial reported it
     as a transition overshoot, which inference turns into a silent −∞ particle
     weight, biasing the posterior toward trajectories where the drains never
     collide. The count case is refused whatever the compartment holds, so it
     fires the first time the events are due together, on every backend and
     every particle. Migration: give the drains different times, or merge them
     into one transfer. Fractions summing to at most 1 still run, unchanged.
- **An event that leaves a compartment negative is `InterventionNegative` on
  chain-binomial.** A `set` to a negative value (and before the refusal above,
  an overdrawing pair of drains) was reported by chain-binomial as
  `BinomialOvershoot`, so `pfilter`, PMMH, IF2 and PGAS gave the particle −∞
  weight and carried on. It is now a configuration error that stops the run, as
  on the ODE and Gillespie. Migration: fix the `set` value.
- **A transition that consumes several units of one compartment (`S + S --> D`)
  is refused on chain-binomial, and Gillespie fires a transition only when its
  sources can pay.** Chain-binomial draws how many individuals leave a
  compartment and applied a two-unit transition to that count twice over: on
  `pair : S + S --> D @ k*S*S/(S+D+R)` from `S = 5` it took `S` to −3 to −5 on
  every seed, which `pfilter`, PMMH, IF2 and PGAS turned into a −∞ particle
  weight. Such a transition is now refused when the model is loaded, on
  `simulate`, `pfilter` and every `fit` method on chain-binomial, naming the
  transition, the compartment and the units. Gillespie fired the same transition
  at `S = 1` (and a constant-rate exit `S --> R @ k` at `S = 0`); it now fires a
  transition only while every compartment it consumes holds what one firing
  removes, at the declared rate, unchanged. Single-unit models whose rates are
  proportional to their source give byte-identical Gillespie trajectories. The
  ODE is unchanged. Migration: run the model on `--backend gillespie` (or fit it
  on `ode`), or restructure the transition.
- **`NegativeCountCause::BinomialOvershoot` is gone.** No transition can leave a
  compartment negative any more, so a negative left by the transition stage is
  `SimError::SimulatorInvariant`: it stops the run, under inference too, and its
  message says it is a camdl bug. A negative count that comes from the starting
  state (`init { S = N - I0 }` with `I0 > N`) is
  `NegativeCount { cause: InitialState }`, classified as `BinomialOvershoot` was
  (under inference the particle gets −∞ weight). Also fixed on the way: an
  overdispersed exit group whose gamma multipliers were all drawn as 0 fell
  through to an unbounded draw and could take its source far below zero (`S = 5`
  to −3305); it now draws no exits.
- **A random initial law that makes a starting count negative stops the run,
  under inference too.** With `init { I ~ poisson(rate = I0)  R = 5 - I }`, a
  particle that draws `I > 5` starts with `R < 0`. `pfilter`, PMMH, IF2 and PGAS
  gave that particle −∞ weight and kept the others, which silently conditioned
  the posterior on the draw being feasible — a restriction the model never
  declared. It is now `NegativeCount { cause: InitialStateDrawn }`, a structural
  error raised when the initial state is drawn, before any rate reads it; the
  message names the compartment and its value and says to constrain the init
  expression or the law. A starting count that is negative as a function of the
  parameters alone (`R = N0 - S0 - I0` with `I0` too large) is unchanged: it is
  still `InitialState`, and under inference the parameters score −∞ and the fit
  continues. Which case applies is read off the `init {}` block, per
  compartment. Migration: write the init so it cannot go negative, e.g.
  `I ~ binomial(n = N0, p = ..)` in place of a `poisson` draw subtracted from
  `N0`.
- **A scenario that disagrees with a fit.toml's `[fixed]` on a parameter is
  refused.** `fit run` on such a file used to run at the `[fixed]` value while
  its notice said the scenario's value had replaced it ("scenario fastrec
  replaced gamma = 0.2 (from fit.toml [fixed]) with 0.3"), so its estimates were
  those of a fit without the scenario. It is now refused before anything is
  written, naming the parameter, both values, the scenario and the fit.toml.
  `survey --fit` and `profile --fit` refuse the same collision, whether the
  scenario comes from the fit.toml or from `--scenario`; there the scenario's
  value had silently won. A scenario `scale` that moves a `[fixed]` value is
  refused the same way; an equal value is not a collision. `fit run --sweep`
  over a parameter the scenario sets or scales is refused, as `simulate --sweep`
  already was (gh#572). Migration: to run under the scenario, give `[fixed]` the
  scenario's value; otherwise remove the parameter from the scenario or name one
  that does not set it. A refit of such a file under the scenario gives
  different estimates from the old run.

- **A parameter cannot be both estimated and given a value: a fit.toml whose
  scenario sets or scales an estimated parameter is refused, and so is a name in
  both `[fixed]` and `[estimate]`.** `fit run` on a fit.toml whose `scenario`
  sets a parameter `[estimate]` lists used to estimate it anyway, discard the
  scenario's value and print nothing. `fit run` and `survey --fit` now refuse a
  scenario, from the fit.toml or (on `survey`) `--scenario`, that sets or scales
  an estimated parameter, composed presets included; the error names the
  parameter and the scenario's value. A name in both `[fixed]` and `[estimate]`
  of one fit.toml, which `profile --fit` and `survey --fit` used to accept with
  a warning (`[fixed]` won), is refused by every command; `fit run` already
  refused it. Unchanged: `--fixed` over `[estimate]` (see CLI); a scenario that
  sets a `profile` nuisance parameter; and `fit predict --scenario`. Migration:
  to fit under the scenario's value, move the parameter from `[estimate]` to
  `[fixed]` with that value; otherwise name a scenario that does not touch it.
  For an overlap, remove the name from whichever block is wrong.

- **`[fixed] from_scenario` naming a scenario that sets or scales an estimated
  parameter is refused.** The import used to skip every parameter `[estimate]`
  lists (gh#37), so `from_scenario = "baseline"` with `beta` estimated fitted
  beta and dropped baseline's value without a word. A parameter cannot be both
  fixed (imported from the scenario) and estimated, so `fit run`,
  `profile --fit` and `survey --fit` now refuse it, composed presets included;
  the error names the parameter, the scenario and the scenario's value.
  Migration: define a scenario that does not set the estimated parameters (for
  example one `baseline` composes from, minus them) and name it in
  `from_scenario`, or drop them from `[estimate]` to hold them at the scenario's
  value. A scenario that only scales an estimated parameter has the first fix
  only: the import would have no value for the scale to multiply.

- **`simulate`'s implicit arm is labelled `(none)`, not `baseline`.** A run
  without `--scenario` shows `(none)` (path segment `_none_-<h8>`) in its leaf
  path, `run.json` and `camdl list`, and its derived labels read
  `(none)+vaccination`, `(none);beta=0.3`, so it can no longer be mistaken for a
  scenario the model names `baseline`. Stored runs are still found: the store
  looks a leaf up by hash, so a leaf stored under `baseline-<h8>` is reused.
  Migration: a script that matches the label text matches `(none)`.

- **`simulate --params` refuses a fit-written `mle_params.toml` whose values no
  longer match its declared `content_hash`, and every file written by an earlier
  fit is refused.** The fit writer hashed the unrounded estimates while the file
  holds ten decimals, so no earlier file's hash was ever verifiable; the writer
  now hashes the values as written, and `simulate --params` checks them. A
  mismatch is refused with the two remedies in the message. Migration: re-run
  the fit and use the `mle_params.toml` it writes, or remove the `[provenance]`
  block from the old file to run its values knowingly (as a plain params file:
  no backend or dt is matched from it and no fit lineage is recorded). A
  `[provenance]` block that fails to parse is likewise an error with the same
  remedies, not a missing block.

- **Every chain index a fit shows counts from 0 (gh#990).** The `chain_N/`
  directories, the progress and refusal lines, `chain_starts.tsv`'s `chain_id`,
  `fit_state.toml`'s `best_chain` and `chain_eval_ids`, `run.json`'s
  `inputs.best_chain`, `mle_params.toml`'s `[provenance] chain`, every
  `fit summary` table, `--exclude-chains`, `fit predict --by-chain`'s `chain`
  column, `progress.json`'s per-chain rows and `cross_chain_compat.json`'s
  `chains` are numbered `0 … N-1`, the same numbers as the `chain` column of
  `draws.tsv` and `trajectories.tsv`, which already counted from 0 and do not
  change. Migration: every consumer that joins on a chain number shifts by one
  (`--exclude-chains 3,5` now names the chains that were `4,6`; a script that
  opened `chain_<c+1>/` for a `draws.tsv` row with `chain = c` opens
  `chain_<c>/`). A fit written before this release (`chain_1/` with no
  `chain_0/`) is refused with a message naming the change by `fit summary`
  (every `--format`), `fit predict`, `fit table --quantity`, `compare` when it
  derives the fit's prequential, `simulate --draws posterior`, and
  `fit run --resume` (before any chain state is copied); re-run the fit. Also
  closes gh#978, gh#979, gh#981 and gh#913, which were the same two-base
  conversion at four more sites.

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
  `[method]` for `algorithm = "pmmh"` and the `--rho` flag on `fit run` are
  gone; `pmmh` runs the bootstrap particle filter with independent likelihood
  evaluations, as it did with `rho` unset. Migration: delete the `rho` line and
  drop the flag. A `fit.toml` that still carries the key, or a script that still
  passes the flag, is refused with that instruction. (`--pmmh-rho` on `profile`
  went with per-cell PMMH, below.)

- **`camdl profile --algorithm pmmh` was removed.** It ran a short MCMC chain in
  each grid cell with the swept parameter ψ pinned, and reported the
  log-likelihood at the highest-posterior sample the chain visited as that
  cell's profile value. That number is not the profile likelihood max_λ L(ψ, λ):
  under an informative prior the highest-posterior sample sits below the
  maximum, by an amount that depends on how well a short chain explored. Nor is
  it the marginal posterior p(ψ | y), which needs each cell's marginal
  likelihood p(y | ψ), and conditional samples do not supply it. The chain
  itself sampled p(λ | ψ, y) correctly; the summary was wrong.
  `--algorithm pmmh` on `profile` is now refused, as are `--pmmh-steps`,
  `--pmmh-particles` and `--pmmh-rho`, with the alternatives.
  `--suppress-warnings` and the `profile_flat_prior_fallback` warning it
  silenced are gone with it, and leaf `mle.toml` files no longer carry
  `final_log_posterior`, `acc_rate` or `log_posterior_trace` (only per-cell PMMH
  wrote them). Migration: for a profile likelihood use `--algorithm if2`, or an
  `nl-*` optimizer on `--backend ode`; for a posterior of ψ, fit once with
  `camdl fit run` (`pmmh` or `pgas`) and read ψ's marginal from the draws. A
  command that runs conditional chains over a grid with a correctly defined
  summary may be added later under its own name.

- **NUTS on the ODE backend was removed.** `algorithm = "nuts"` (with
  `backend = "ode"`) and its keys `warmup`, `samples`, `target_accept`,
  `max_tree_depth`, `dense_mass` and `burnin_dt` under it are gone, on every
  backend. Migration: sample an ODE model with `algorithm = "mh"` (gradient-free
  adaptive Metropolis-Hastings, which also takes `burnin_dt`), or take its
  gradient-based maximum likelihood with `algorithm = "nl-lbfgs"`; on
  `chain_binomial`, NUTS is PGAS's parameter move (`use_nuts`, on by default). A
  `fit.toml` or `--algorithm` that still names `nuts` is refused with that
  instruction. Stored NUTS fits are no longer read: their leaves record `nuts`
  as their method, which this camdl does not recognise, so `fit summary`,
  `fit table` and `fit predict` skip them, printing one line per skipped leaf
  that names it and the reason; read one with a camdl built at the
  `archive/ode-nuts` tag. The forward-sensitivity gradient, `nl-lbfgs` and
  PGAS's NUTS move are unchanged.

- **`camdl batch` was removed.** `batch run MANIFEST.toml` and `batch status`
  are gone. The one capability behind them — a deterministic grid of a
  parameter, run with common random numbers into the store and resumable — is
  `camdl simulate --sweep PARAM=GRID` (repeatable; a comma list,
  `lin(min,max,n)` or `log10(min,max,n)`) and `--sweep-file FILE`, composed with
  `--scenario`, `--replicates` / `--seeds`, `--parallel`, `--force` and
  `--dry-run` (which now reports every cell as a store hit or miss). Migration:
  `[config]` becomes the `simulate` flags, each `[[scenario]]` a
  `--scenario NAME` (an ad-hoc arm is `--param` / `--enable` / `--disable`),
  `[sweep] beta = [0.2, 0.3]` becomes `--sweep beta=0.2,0.3`,
  `[config.seeds] n = 50` becomes `--seeds 1:50`, `[obs] enabled = true` becomes
  `--obs-dir DIR`, and a `[design.*]` block is generated externally and passed
  as `--sweep-file`.

- **Events due at the same instant follow one rule on every backend (gh#985).**
  Inflow `add` events resolve against the state before the instant; draining
  `transfer` events and `set` then resolve against the post-inflow state;
  interventions follow in declaration order (language spec §13.10). This was
  already chain-binomial's rule. On Gillespie and the ODE, an `add(X, 10)` and
  an `{ X = 5 ... }` event at one instant gave `X = 15` (Gillespie, and the ODE
  when the `set` was declared first) and now give `X = 5`; a draining `transfer`
  beside an inflow into its source moved nothing and now moves its share of the
  post-inflow count. Within that stage the changes add on every backend: from
  `X = 10`, two half drains of `X` at one instant leave `X = 0` in either order
  (a `set` beside a drain is refused, below). On the ODE, any compartment that
  receives two or more changes in one stage — two `add` events into it, two
  `transfer`s into it, a drain from it beside a `transfer` into it — is now its
  value plus those changes added in ascending order, so such runs can differ
  from earlier builds in the last bits even where no rule changed. Migration:
  none needed; Gillespie and ODE runs whose events coincide this way give
  different, now backend-consistent, results. Chain-binomial and every inference
  method are unchanged.

- **Two `set`s on one compartment at one instant are refused (gh#985), and so is
  a `set` beside a `transfer` from or into it.** Their changes would add to a
  value neither names (from `X = 5`, two `X = 7` events gave 9 on every backend;
  from `X = 10`, `{ X = 5 ... }` beside a half drain of `X` gave `X = 0`). An
  inflow `add` beside a `set` still runs: the `set` overwrites the post-inflow
  value. Every backend now stops with an error naming the compartment, the
  events and the time. The ODE likewise now refuses events that leave an integer
  compartment negative, as chain-binomial and Gillespie already did, instead of
  printing it as 0 — including a single `{ X = -1 ... }` event on an integer
  compartment. Migration: give the `set` and the other event different times or
  merge them into one.

- **An effect at `t_start` now applies to the starting state, on every backend
  and every fit (gh#958).** `at [0]` with `from = 0`, and the first fire of a
  recurring schedule whose `from` defaults to `t_start`, used to be dropped
  silently by chain-binomial simulation, the particle filter, PMMH, IF2 and
  PGAS, and applied by the ODE and Gillespie only after the `t_start` row was
  recorded. It now applies to the `init {}` state before that row is recorded or
  an observation there is scored, in the §13.10 order (inflow events, then
  drains and `set` on the post-inflow state, then interventions, then
  `balance`). `every = 2 'days` over `[0, 6]` fires four times on every backend
  (chain-binomial fired three). A forecast or contrast arm resumed from a saved
  state does not apply it again. Migration: an `at [t_start]` effect that was
  meant to be inert must be removed; a schedule that relied on chain-binomial
  skipping its first fire should set `from`.

- **The particle filter, PMMH, IF2 and PGAS fire every effect where
  chain-binomial simulation does (gh#957).** Each backend fires effects in its
  own natural time: chain-binomial and every fit that steps it fire at the end
  of the nearest step of the run's own grid, `k = ⌊(t − t_start)/dt + ½⌋`; the
  ODE and Gillespie fire at the declared time. An off-grid event used to fire at
  its exact time in the filter while `simulate` snapped it (−0.355 nats against
  the exact likelihood on the oracle model), and an off-grid scheduled
  intervention was refused under the filter; both now snap. The key is now
  relative to `t_start` (`from = 0.5`, `dt = 1`: an event at 1 fires at 1.5,
  where it used to be dropped, and one at 2.4 at 2.5, not 1.5), a tie goes to
  the later step whatever its sign, and only effects inside `[t_start, t_end]`
  fire (an effect in `(t_end, t_end + dt/2]` used to fire on chain-binomial; one
  in `(t_start − dt/2, t_start)` on the ODE and Gillespie). When `t_end` is off
  the grid, an effect in the shortened last step fires at its nearer end, with a
  warning. The ODE and Gillespie now fire an intervention once per declared time
  (`at [2.3, 2.4]` fires twice) and no longer fire an effect early at another
  effect's landing in the same step. Migration: none needed; the likelihood of a
  model with such effects changes, so refit it.

- **PGAS and IF2 refuse a parameter they cannot fit (gh#986, gh#984).** Under
  PGAS an estimated parameter that sets an effect's amount, an effect's
  `at [...]` time, or a plain `init {}` assignment (`init { I = I0 }`) could not
  be moved by the sampler through that channel — one entering only that way
  sampled its prior and reported it as the posterior; under IF2 an estimated
  parameter in an effect's schedule stayed at its start value. Both are now
  refused, naming the parameter and the effect. Migration: fit seed timing and
  size with PMMH, fix the parameter, or, for an initial size under PGAS, declare
  a law (`I ~ poisson(rate = I0)`).

- **ODE and Gillespie runs whose recurring period is shorter than `dt` now
  run.** They were refused at start ("one fire per period would be silently
  dropped"); the continuous backends fire every recurring target at its own
  instant, so the check now applies to the chain-binomial clock only (and a
  fit's pre-flight checks only the particle-filter methods). Migration: none.

- **A fit whose data run past the model's simulation end is refused.** The
  particle filter, PMMH, IF2 and PGAS (and `profile`, `survey`, `compare` and
  `fit predict` through them) used to step on to the last observation past
  `simulate { to }`, scoring states `simulate` never produces, with the effects
  due there unplanned. They now refuse, naming both times: "the data run to
  t=7672 days but the model's simulation ends at t=7670.0925 days …". Migration:
  extend `to` to cover the data, e.g. `to = last_obs` or
  `to = last_obs + 8 'weeks`.

- **A `fit predict` contrast forks before the instant its toggled intervention
  fires, not its declared time.** The arms run chain-binomial, which fires an
  intervention declared at 2.4 with `dt = 1` at 2; the fork used to land on the
  post-effect snapshot at 2, so both arms carried the effect the contrast
  isolates. Migration: none; such contrasts now measure the effect.

## Run identity

- **Every stored PGAS fit, every stored PMMH fit and every stored
  `camdl profile` run re-keys.** `tempering` was part of the PGAS method
  identity and `rho` of the PMMH method identity, and `camdl profile` hashed its
  per-cell PMMH knobs (`steps`, `particles`, `rho`) into the method level of
  every run whatever its algorithm, so an IF2 or `nl-*` profile was keyed on
  settings it never read. Removing them is one identity change: a new PGAS or
  PMMH fit, or a new profile run of any algorithm, does not deduplicate against
  one stored before this change, and an in-flight PGAS or PMMH fit cannot
  `--resume` across it (its saved config hash no longer matches). Stored results
  stay readable; the fits and profiles run again.
- **Every stored multi-cell `simulate` ensemble re-keys once.** The ensemble
  leaf (the combined `ensemble.tsv` of a `--replicates` / `--seeds` /
  multi-`--scenario` / `--draws` run) is now keyed on its cells' identities
  alone — per cell the parameter-point index, the process seed and the cell's
  own `run_id` — plus the run's resolved design; the arm's display label has
  left the key. Every stored multi-cell ensemble is therefore a miss once and is
  rebuilt from the per-cell `Sim` leaves the store already holds, so the cost is
  aggregation, not simulation. Two ensembles of identical cells now share one
  identity whether an arm was named by a preset or derived from
  `--enable`/`--disable`/`--param`. Renaming a preset still re-keys, because a
  preset's name is part of the model's content digest.
- **Every stored `--sweep-file` ensemble re-keys once.** The resolved design of
  a sweep-file job now records the written `point_id` of every row (the file's
  own column when present, else the 0-based row), and the grid level keys each
  cell on that written id, so the identity names what the ensemble's rows say.
  Cells are reused; the ensemble is rebuilt from them.

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

- **Content-addressed run management:** `simulate` / `fit` / `profile` /
  `survey` write to a keyed store; browse with `list`, `show`, `cat`, `reindex`,
  and `compare` (prequential elpd / CRPS / PIT).
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
- **`--fixed` / `--fixed-file` over a name in the fit config's `[estimate]` is
  reported at `info:`, not `warning:`.** It stays a layered override — the
  parameter is pinned and not estimated — and the line now names the value:
  ``info: beta: `--fixed beta=0.4` pins it, so it is not estimated (the fit
  config lists it in [estimate])``.
  Its `run.json` notice code is `fixed-over-estimate` (was
  `kicked-from-estimate`).
- **`profile` says when its scenario sets the focal parameter.** The grid
  replaces the scenario's value at every point, as before; one line per run now
  says so:
  ``info: scenario `baseline` sets beta = 0.4; beta is profiled, so
  the grid values replace it``.
  Recorded in `run.json`'s notices.
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

- **`profile` estimated parameters the invocation pinned.** `--rw-sd auto` built
  its estimated set from `--fixed` and `[fixed]` only, so a `--fixed-file`
  holding `gamma = 0.15` profiled gamma as a nuisance parameter and moved it (to
  0.177); it also estimated model constants. Both are now excluded.
- **`profile --fit` ignored `[[simplex_groups]]`.** Each IF2 cell perturbed the
  members independently, so two cells' optima summed to 1.85 and 1.44. The
  groups now reach every IF2 cell; a member the profile does not estimate is
  refused by name.
- **`profile` recorded every parameter as `fixed`.** A cell's
  `parameters_provenance` now records the swept parameter as `focal` and the
  parameters the cell moves as `estimated`, and the profile base `fit.meta.json`
  carries the same map instead of an empty one.
- **`survey --fit … --fixed beta=0.3` ignored `--fixed`.** No notice was printed
  and the landscape sampled beta across its bounds. It now pins beta, with the
  info notice `profile` prints for `--fixed` over `[estimate]`.
- **`fit run` recorded an `[estimate].start` as `model_default`.** The
  provenance now says `estimate_start` (or `estimate_start_drawn` for an
  in-bounds draw); the same holds for `survey --fit` and `profile --fit`.
- **`pmmh`/`mh`/`pgas` reported a hyperparameter's prior as its posterior** when
  its hierarchical leaves were held; such a fit is now refused (see Breaking
  changes).

- **`[fixed] from_scenario` applies the scenario's `scale`.** The import copied
  the scenario's `set` values and ignored its `scale` factors, so a fit under a
  scenario that scales a parameter it sets held that parameter at the unscaled
  value, with nothing said: with
  `s { set = { gamma = 0.15 } scale = { gamma =
  2.0 } }`,
  `simulate --scenario s` ran at gamma = 0.3 and a `from_scenario = "s"` fit
  held it at 0.15. The import now gives exactly the values `simulate --scenario`
  uses (the `set` values with `scale` applied, through composed presets) on
  `fit run`, `profile --fit` and `survey --fit`. A scenario that scales a
  parameter it does not set is refused, since the import has no value for the
  factor to multiply; the error names the parameter and the factor, and the fix
  is to set the parameter in the scenario as well. Fits under a scaling
  `from_scenario` change value and re-key (their `fit_hash` covers the fixed
  values).

- **PGAS scores an observation at `t_start`.** It kept an observation only at a
  substep after the first, so a `t_start` row was silently dropped from every
  PGAS likelihood — the draws were byte-identical with and without it, and the
  summary counted one observation fewer than the file held — and under exact
  alignment every later observation was dropped with it. It is now scored on the
  state after the effects at `t_start`, as the particle filter scores it, in the
  initialization filter, the CSMC sweep, the complete-data likelihood and its
  gradient. Fits whose data start at `t_start` change.

- **The particle filter, IF2 and PGAS fire effects on the model's own horizon.**
  Their effect timing was built over the window ending at the last observation,
  so the data could move an effect: with `dt = 1` an effect at 4.3 fired at 4.4
  when the data ended at 4.4 and at 4 when they ended later, and an effect at
  10.3 that `simulate` fires at the step ending at 10 (`to = 20`) was never seen
  by a filter whose data ended at 10. They now use the plan `simulate` builds,
  over `[t_start, to]`, so the data never move an effect.

- **`profile --fit` applies the fit config's scenario.** `profile` read
  `[estimate]`, `[fixed]` and `[data]` from the fit config but never its
  top-level `scenario`, `enable` or `disable`, so earlier `profile --fit` runs
  whose config named a scenario profiled the unmodified model, and their
  log-likelihoods did not include the scenario. The config's selection now
  applies, combined with `--scenario`/`--enable`/`--disable` by the same rule as
  `survey --fit`: either alone selects, both must name the same scenario, and
  anything else is refused, naming both.
- **`survey --fit` applies `--scenario`.** The fit-aware form of `survey` read
  only the fit config's own `scenario` key, so the `--scenario` flag never
  reached parameter resolution: earlier `survey --fit … --scenario NAME` runs
  used the unmodified model, so their landscapes did not include the scenario
  (they were byte-identical to, and stored in the same leaf as, the run with no
  scenario), and a misspelled name ran at exit status 0. The flag is now
  resolved like every other verb's: a declared scenario is applied and keys its
  own leaf, an undeclared one is refused with the declared names listed. When
  the fit config also names a scenario, the two must agree or the run is
  refused; `--scenario` beside the config's `enable`/`disable` lists is refused
  too. Separately, a scenario that set a parameter the config also pins in
  `[fixed]` was overridden by the `[fixed]` value when the landscape was scored;
  the scenario's value now applies, as the precedence order states. Surveys
  whose config named such a scenario should be re-run.
- **`baseline` is an ordinary scenario name on every verb.**
  `--scenario baseline` means the scenario the model declares under that name,
  on `simulate`, `fit predict`, `pfilter`, `profile` and inline-mode `survey`; a
  model that declares no `baseline` gets the same refusal, listing its declared
  scenarios, as any other undeclared name. `fit predict` used to accept an
  undeclared `baseline` in its up-front check, run, and then fail with exit
  status 1 after writing partial output, and its message for a misspelled
  scenario advised `use 'baseline' for the unmodified model`, which is that
  failing form. It now refuses before running anything, and the message says to
  drop `--scenario` to run the model with no scenario applied. A declared
  `baseline` scenario was applied correctly before this change, so earlier
  predictions made with `--scenario baseline` are unaffected.
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
