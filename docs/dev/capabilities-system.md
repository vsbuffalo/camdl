# The capability / compatibility systems

camdl decides "can this _model_ run with this _backend_ under this _fitting
algorithm_?" along **three distinct axes**, enforced by **two structured
mechanisms plus one family of ad-hoc checks**. They are not one system, and they
are not fully orthogonal. This document is the source of truth for what the axes
are, how each is enforced, and where. CLAUDE.md and the review charter point
here rather than re-describing it (an earlier review draft conflated the first
two axes and missed the third — that is the failure mode this doc exists to
prevent).

Scope: the **core triple** that answers "will my fit/sim run, and is the answer
trustworthy." Parameter-level, CLI-flag, and version-handshake gates are real
but out of scope here; they are listed under [Other axes](#other-axes) with
pointers.

Verification basis: citations below were read against the tree on 2026-06-09;
the `mh`/`ode` row and the axis-2 note were re-verified and corrected
2026-06-18. Line numbers drift; the cited **symbol names** are the stable
anchor. When in doubt, grep the symbol, don't trust the line number.

## The three axes at a glance

| Axis                             | Relates                                                                                       | Mechanism                             | Kind                                  | Enforced at                                    |
| -------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------------- |
| 1. **model-feature × backend**   | what a model _needs_ vs what a backend _provides_                                             | `Capabilities` bitflags               | **structured** (auto-derived from IR) | dispatch — simulate _and_ fit paths            |
| 2. **algorithm × backend**       | which fitting algorithm is statistically valid on which backend                               | `METHODS` registry + `validate_combo` | **structured** (static table)         | config-load / pre-flight                       |
| 3. **model-feature × algorithm** | model properties a _specific algorithm_ can't handle, on a backend that otherwise supports it | scattered ad-hoc checks               | **NOT a system** — convention         | mixed: config-load _and_ deep in the algorithm |

Key consequences to hold onto:

- The **DSL/OCaml side declares nothing** about backends, algorithms, or
  capabilities. Capabilities are 100% _derived_ from model structure on the Rust
  side (`CompiledModel::required_capabilities`). The IR has no `capability` or
  `target_backend` field. So "DSL × backend" is really "model-structural-feature
  × backend", computed in Rust.
- Axes 1 and 2 **both ultimately constrain "which backend"**, through two
  unconnected mechanisms — and two divergent backend-capability tables (see
  [Known gaps](#known-gaps)). They are not orthogonal.
- Axis 3 is the one most likely to surprise: the same model on the same backend
  can be **accepted by one algorithm and rejected by another** (PGAS vs PMMH on
  `chain_binomial`).

## Axis 1 — model-feature × backend (the `Capabilities` bitflags)

The only axis enforced by a uniform, auto-derived mechanism. Defined in
`rust/crates/sim/src/lib.rs` (`bitflags! { pub struct Capabilities }`). A "model
feature" here means a **DSL primitive** — `overdispersed(...)`, a `balance {}`
block, a real-valued compartment, `dt` in a rate expression — that survives
compilation into the IR, where `required_capabilities` detects it. (`LINEAGES`
is the exception: it is request-driven, not feature-derived — see its row.)

| Flag                     | Model feature that requires it                                                                                                              | Backends that provide it                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `OVERDISPERSION`         | a transition uses `overdispersed(...)` (NegBinomial draws)                                                                                  | chain-binomial only                                                |
| `REAL_COMPARTMENTS`      | real-valued compartments with ODE equations (PDMP)                                                                                          | all three (simulate); see fork below                               |
| `BALANCE`                | a `balance { ... }` block                                                                                                                   | chain-binomial only                                                |
| `LINEAGES`               | the `#[lineage]` DSL annotation exists, but the requirement is **not derived from it** — raised only on explicit `--lineages`/`--event-log` | gillespie + chain-binomial (not ODE)                               |
| `RUNTIME_DT`             | a rate **or its `rate_grad`** references `Expr::Dt`                                                                                         | ODE + chain-binomial (not gillespie)                               |
| `REACTIVE_INTERVENTIONS` | an intervention has a reactive fire source (`fire = Reactive(..)`, gh#204)                                                                  | chain-binomial forward only (not gillespie/ode; not inference yet) |

**Derivation (model side).** `CompiledModel::required_capabilities`
(`compiled_model.rs`) scans the IR and ORs in a flag per feature present. It is
purely _structural_ — independent of parameter values (the fit path fills
placeholder values just to run the scan). `LINEAGES` is deliberately **not**
derived here (a `#[lineage]`-annotated model runs identically with or without
tracking); it is raised at the `--event-log` request site instead.

**Declaration (backend side) — and the fork.** This is the subtle part: there
are **two** backend-capability tables, and they disagree on purpose.

- **Simulate path:** each backend's `Simulate::capabilities()` impl
  (`gillespie.rs`, `ode.rs`, `chain_binomial.rs`). The gate is a direct subset
  check in `util.rs::simulate_compiled`:

  ```rust
  let caps = backend.capabilities();
  let required = compiled.required_capabilities();
  if !caps.contains(required) { /* hard error */ }
  ```

- **Fit/inference path:** a **separate, hardcoded** table inside
  `fit/methods.rs::check_model_capabilities` — _not_ the trait. It intentionally
  diverges: `chain_binomial` **inference withholds `REAL_COMPARTMENTS`** even
  though the `Simulate` impl grants it, because the filter loops carry no real
  state and would silently mis-fit a real-coupled model (gh#191):

  ```rust
  "chain_binomial" => Capabilities::OVERDISPERSION | Capabilities::BALANCE | Capabilities::RUNTIME_DT,
  "ode"            => Capabilities::REAL_COMPARTMENTS | Capabilities::RUNTIME_DT,
  ```

  Gillespie is absent here — it is not an inference backend.

So the effective relation on this axis is **(model-feature × backend ×
execution-mode{simulate | infer})**. Same flags, same `required_capabilities`,
but the "what the backend provides" side forks by mode.

**Enforcement sites** (all hard errors):

- `util.rs::simulate_compiled` — the simulate gate (subset check vs the trait).
- `util.rs::run_simulation_event_log` — the `LINEAGES` gate, raised at
  `--event-log` request time (ODE is rejected by a hand-written arm just above,
  so the bitflag check there only ever fires as an internal-error assertion).
- `fit/methods.rs::check_model_capabilities` — the inference gate (hardcoded
  table). Called per-stage on `camdl fit run` (`fit/mod.rs`, via
  `gate_run_stages_against_model`), in `profile.rs`, in `survey.rs` (for
  `--eval simulate`), and inside `nlopt_stage.rs`.

The inference math itself contains **zero** capability references — the gate is
the only thing standing between a real-compartment model and a silent mis-fit
(PGAS just passes a zeroed real reservoir: see the `KNOWN LIMITATION` comment in
`pgas.rs`). That coupling — load-bearing gate far from the math it protects — is
flagged under [Known gaps](#known-gaps).

## Axis 2 — algorithm × backend (the `METHODS` registry)

A static table of supported `(algorithm, backend)` pairs in
`rust/crates/cli/src/fit/methods.rs` (`pub const METHODS`). It is
**structural-only**: it consults the two strings and never looks at the model or
the `Capabilities` bitflags (the doc comment on `validate_combo` says so
explicitly).

The registered (valid) matrix is **block-diagonal by backend** — particle-filter
methods live on the stochastic backend; the deterministic backend carries the
gradient-free optimizers and a direct-likelihood sampler (`mh`):

| algorithm              | `chain_binomial` | `ode`   |
| ---------------------- | ---------------- | ------- |
| `if2` (MLE)            | ✅ Stable        | ❌      |
| `pgas` (Bayesian)      | ✅ Stable        | ❌      |
| `pmmh` (Bayesian)      | ✅ Stable        | ❌      |
| `pfilter` (diagnostic) | ✅ Stable        | ❌      |
| `nl-sbplx` (MLE)       | ❌               | ✅ Beta |
| `nl-bobyqa` (MLE)      | ❌               | ✅ Beta |
| `mh` (Bayesian)        | ❌               | ✅ Beta |

`gillespie` never appears — it is not a fitting backend, and the typed `Backend`
enum (`run_meta.rs`) has only `ChainBinomial` / `Ode`, so serde rejects anything
else at TOML-parse time.

**Why each off-diagonal cell is rejected** is the statistically interesting
part, and `rejection_reason` (`fit/methods.rs`) hand-writes the explanation per
pair, each pointing at the right alternative. The recurring theme: a PF method
on ODE is degenerate (all particles identical → no variance to exploit/filter),
and a deterministic optimizer on chain-binomial sees ranking noise that defeats
convergence. Examples:

- `(pgas, ode)` → "the CSMC step needs stochastic process variance to refresh
  the trajectory between θ updates. Under ODE all particles produce identical
  trajectories per θ, so the CSMC step is degenerate." Suggests `mh`/`nuts`.
- `(if2, ode)` → "collapses to a noisy gradient-free hill-climber." Suggests
  `nl-sbplx`/`nl-bobyqa`.
- `(nl-sbplx, chain_binomial)` → "the single-trajectory loglik is a noisy
  estimator … the optimizer sees ranking noise that defeats convergence."
  Suggests `if2`.

Note `mh` is now **registered** as the ODE-Bayesian method (Beta — direct
Metropolis-Hastings on the deterministic ODE marginal likelihood; landed
`590e80da`, dispatched in `fit/mod.rs`). `nuts` (the planned gradient-based
ODE-Bayesian method) is still only a _suggested_ answer in `rejection_reason`
and **not yet registered** — asking for it today hits a bespoke rejection.

**Enforcement.** `validate_combo` is called at config-load / pre-flight, both
hard aborts:

- `config_v2.rs` (inside `FitConfigV2::validate`) — **per stage** on
  `camdl fit run` (a config can mix an ODE scout + a chain-binomial refine).
- `profile.rs` — before any expensive setup, on the raw
  `--algorithm`/`--backend` flags.

Not bypassable on either real entry point.

## Axis 3 — model-feature × algorithm (ad-hoc, not a system)

This axis is real but has **no unifying mechanism** — it is a scatter of checks,
some at config-load, some deep inside the algorithm at runtime. The defining
property: a model accepted on a backend by one algorithm is **rejected on the
same backend by another**. Representative confirmed cases:

- **Priors required for Bayesian, not MLE.** `Stage::requires_priors` returns
  true only for `PGAS`/`PMMH` (`config_v2.rs`); enforced pre-flight by
  `validate_priors_present`. PGAS additionally refuses _implicit flat_ priors at
  runtime (`pgas.rs`). IF2/NLopt ignore priors (and warn if you declare unused
  ones).
- **PGAS rejects hierarchical priors; PMMH accepts them.** Same backend, same
  model — decided purely by algorithm. PGAS errors with "does not support
  hierarchical priors … the NUTS gradient for hierarchical leaves is not yet
  implemented (Gate 3b) … Use `algorithm = pmmh`" (`pgas.rs`).
- **PGAS+NUTS rejects parameters reachable through a parametric `DerivedExpr`
  observation projection** (`pgas.rs`) — the projection chain-rule term is
  omitted so the gradient is identically zero on that coordinate, which would
  give "silently biased posteriors." Blocked with a hint to fix the parameter or
  switch to IF2/PMMH. This is a top-severity _silent-wrong-answer_ guard, caught
  correctly.
- **The correlated-PMMH family** (`correlated_pf.rs`) rejects unequal obs-window
  substep counts, state-dependent overdispersion σ², >1 overdispersed transition
  per source group, etc. — all keyed on the correlated-PMMH _algorithm variant_
  (rho ≠ None), not the backend.
- **`ic_free` requires `if2`** (`methods.rs::validate_ic_free`, called per stage
  from `FitConfigV2::validate`). Two properties are needed, not one: the
  algorithm must drop y₁ from the accumulated log-likelihood, _and_ its
  particles must differ in x₀ — the reweight at y₁ is what pins the initial
  state, and with one shared x₀ it scores every particle identically. Only IF2
  has both (it perturbs θ per particle at t=0 and each particle draws its own x₀
  from its own θ, gh#364). `pgas` / the ODE algorithms / correlated `pmmh` fail
  the first; `pfilter` and plain `pmmh` fail the second, because the bootstrap
  particle filter copies one deterministic x₀ to every particle (gh#732). This
  is the axis's second silent-wrong-answer guard, and it was wrong until gh#732:
  it tested a _proxy_ (is some parameter flagged?) for a _property_ (does spread
  exist?), and the proxy did not imply the property.
- **`perturb_only_at_t0` requires the fit to have an `if2` stage**
  (`methods.rs::validate_perturb_only_at_t0`, called once from
  `FitConfigV2::validate`). The flag is an IF2 schedule — "perturb at t=0 only"
  — and IF2 is the only algorithm that perturbs at all, so under `pgas`, `pmmh`,
  `mh`, `nuts` or the nlopt family it is parsed, hashed, and read nowhere.
  **This one is checked against the whole fit, not per stage**, and it is the
  exception worth remembering on this axis: `[estimate]` is global to the fit
  while the algorithm is per stage, so the flag is a property of the fit. An
  if2-scout → pgas-posterior pipeline is accepted (one stage reads it, the other
  ignores it); a fit with no `if2` stage anywhere is refused, because there the
  declaration genuinely does nothing. A `pfilter`-only fit is tolerated —
  `pfilter` estimates nothing, so the flag is no more inert than the `rw_sd`
  beside it.

  Checking it per stage instead was a real regression, caught in review before
  it shipped: it refused the ordinary scout-then-refine shape and left the user
  only worse escapes — drop the flag, and the IF2 scout perturbs an
  initial-value parameter at every observation, which is the thing the flag
  exists to prevent. The general lesson for this axis: when a config key is
  fit-scoped and the algorithm is stage-scoped, the check belongs at the fit
  level.

Because there is no registry for this axis, the only way to know whether an
algorithm accepts a model feature is to read the algorithm. That is the cost of
it not being systematized; see [Known gaps](#known-gaps).

**Two common mis-classifications** (corrected here so they don't recur):

- **Missing `rate_grad` is _not_ a hard gate on this axis.** PGAS falls back to
  MH-within-Gibbs when no gradients are present (`pgas.rs`); it degrades, it
  does not reject. The only hard gradient-related reject is the
  parametric-projection case above.
- **`--flow` × multi-stream is _not_ model-feature × algorithm.** It is CLI-flag
  × observation-structure (`pfilter.rs`, `profile.rs`) and fires regardless of
  algorithm. It belongs under [Other axes](#other-axes).

## How the axes compose

A fit run is gated by **all three**, in this order:

1. **Axis 2** (`validate_combo`): is `(algorithm, backend)` a registered pair?
2. **Axis 1** (`check_model_capabilities`): does the _backend_ (inference table)
   provide what the _model_ requires?
3. **Axis 3** (scattered): does the _algorithm_ accept this model's features?

Axes 1+2 together gate the `(algorithm, backend, model)` triple at the
backend-choice level; axis 3 adds the per-algorithm model constraints that
neither backend-keyed mechanism can express. No single function is the whole
triple.

## Other axes

Out of scope for this doc, listed so the map is complete:

- **parameter-attribute × {attribute, config-flag}** — prior↔transform and
  prior↔bounds (`fit/runner.rs::validate_prior_transform_compat`),
  `perturb_only_at_t0`↔`ic_free` (`runner.rs` — the residual parameter-level
  half; the algorithm-level half moved to axis 3, above),
  `perturb_only_at_t0`↔simplex-membership (`config_v2.rs`). Purely
  parameter-level; no algorithm or backend involved. Partly typed already:
  `docs/dev/proposals/2026-06-08-typed-parameter-surface.md` (landed, IR 0.11)
  made prior-on-a-fixed-value and prior+hierarchical unrepresentable via the
  `ParamValue`/`PriorSpec` ADTs; the prior↔transform and `perturb_only_at_t0`
  checks above stay runtime predicates by design (`perturb_only_at_t0` is a
  fit-layer perturbation schedule, not an IR property).
- **CLI-flag × {algorithm, obs-structure}** — `--flow`×multi-stream
  (`pfilter.rs`/`profile.rs`), `--resume`×algorithm (`fit/mod.rs`),
  `--init`×companion-path (`args/mod.rs`), `obs_alignment`×algorithm
  (`fit/methods.rs`).
- **version / format handshakes** — IR version match (`ir/src/envelope.rs`),
  camdlc↔camdl git-hash guard (`util.rs`; see CLAUDE.md "camdlc version
  mismatch").
- **model-validity (OCaml frontend)** — dimensional/structural/calendar checks
  in `dimcheck.ml`/`expander.ml`. Backend-agnostic: OCaml answers "is this a
  well-formed model?", Rust answers "can this backend/algorithm run it?". The
  OCaml↔Rust boundary on compatibility is clean — the frontend knows nothing
  about backends or algorithms.

## Known gaps

Descriptive. Each names the in-flight work that owns the fix; none should be
closed ad-hoc ahead of its owning proposal — that churns the same code those
proposals rewrite.

1. **Two sources of truth for "what a backend supports."**
   `Simulate::capabilities()` (the trait, simulate path) and the hardcoded match
   in `check_model_capabilities` (inference path) are independent tables. They
   diverge _on purpose_ (chain-binomial inference withholds
   `REAL_COMPARTMENTS`), and the divergence is correct and commented — but
   nothing links them, so a capability added to one will not propagate to the
   other. _Owned by_
   `docs/dev/proposals/2026-06-08-capability-gate-consolidation.md` (RFC,
   unimplemented): one `check_capabilities` gate fed by an
   `inference_capabilities(backend, algorithm)` profile, replacing the hardcoded
   table.

2. **The inference capability gate is the _only_ guard against a silent
   real-compartment mis-fit.** The inference math (`pgas.rs`,
   `particle_filter.rs`) does not check any capability — it passes a zeroed real
   reservoir by convention. The `REAL_COMPARTMENTS`-withholding in
   `check_model_capabilities` is load-bearing and lives far from the math it
   protects; any future caller that reaches the algorithm without going through
   that gate would mis-fit silently. _Owned by_
   `docs/dev/proposals/2026-06-09-real-compartments-inference-stack.md` (RFC
   v3): it makes inference actually advance the reservoir at every site, then
   §2.7 re-grants chain-binomial `REAL_COMPARTMENTS` for inference — which also
   collapses gap 1 for that flag. Sequencing constraint: the re-grant must be
   the _last_ step (after all four math sites land), or the silent `W=0` seam
   reopens.

3. **Error-quality asymmetry on axis 1.** The simulate-path failure
   (`util.rs::simulate_compiled`) is `{:?}`-Debug-formatted with no fix hint;
   the inference-path failure (`check_model_capabilities` → `capability_hint`)
   has rich per-flag remediation. Same relation, two UX bars — the simulate
   message is below the repo's "error messages are a feature" standard. _Owned
   by_ the same capability-gate-consolidation proposal, which upgrades the
   simulate message to the shared hint builder.

4. **Gillespie + time-varying rates is a piecewise-constant approximation, not a
   capability error.** Gillespie's next-event draw holds the total propensity
   constant over each exponential wait, so a time-varying rate (seasonal
   forcing, a bare `t`, importation forcing) is frozen within the wait and only
   refreshed at grid boundaries — biasing the inhomogeneous Poisson process. It
   is _allowed_ (a fine output grid shrinks the bias), so it is surfaced as a
   WARNING (`warn_if_gillespie_time_dep`, gh#95) at the gillespie forward
   dispatch, not a hard gate. Mitigation: use a fine output grid, or prefer
   `chain_binomial` (re-evaluates the rate every substep). Mirrors
   `warn_if_ode_euler_flow` (the `dt`-in-rate ODE-Euler caveat).

## Two traps to internalise

- **"It ran" ≠ "it's valid for this algorithm."** Axis 3 is where the
  silent-wrong-answer risk concentrates and it is the least systematized — a
  combination that passes axes 1 and 2 can still be wrong for the chosen
  algorithm. Don't infer correctness from a clean run.
- **Nothing is declared in the DSL/IR.** Compatibility is derived (axis 1),
  tabled in Rust (axis 2), or convention (axis 3) — never a field you can read
  off the model.
