---
status: approved (narrowed scope)
date: 2026-05-02
authors: camdl-side, prompted by typhoid-vignette agent's gh#40 finding
target: profile-only deterministic inference via NLopt; ship in ~1 week
supersedes: original two-phase proposal (phase 1 dropped — slice-only
  doesn't replace IF2's per-cell inner reoptimisation that profile
  performs today)
---

# ODE-Backend Deterministic Inference via NLopt (profile-only)

## TL;DR

`camdl profile` currently runs IF2 over nuisance parameters at each grid
cell. On large-population stratified equilibrium models the per-cell IF2
takes hours and the resulting profile is noise-jagged. Replace IF2's
inner-cell role with **deterministic ODE forward sims + NLopt
optimisation** (default algorithm: **Sbplx**, robust to the boundary
non-smoothness typical of compartmental likelihoods). Wire only into
`camdl profile`. Leave `camdl fit` chain-binomial-only for now —
multi-stage backend selection (scout/refine, fit.toml `[stages.X]
backend`) is deferred until the architectural picture is clearer.

Closes the typhoid agent's gh#40 self-consistency-surface gap (their
Python wrapper computing Poisson loglik in numpy disappears).

## Two likelihoods, not one optimiser for one likelihood

This is the load-bearing point and it must be in the user-facing docs.

When a model is fit with `--backend chain_binomial`, the model's
likelihood is *defined by* its stochastic forward kernel: the PF gives
an unbiased estimator of `p(y | θ)` where the process noise is part of
the generative model. When the same `.camdl` file is fit with
`--backend ode`, we are computing a *different* likelihood —
`p(y | θ, deterministic skeleton)` — for the same model. These are
not the same statistical object. They will give:

- Different MLEs (the stochastic likelihood penalises high-variance
  regions of parameter space; ODE doesn't see them).
- Different identifiability properties (process noise can break
  identifiability that ODE preserves, and vice versa).
- Different uncertainty (Wald CIs from a Hessian under-cover relative
  to PMMH for stochastic models).

In low-noise regimes (large per-cell populations, near-deterministic
trajectories) the two likelihoods converge empirically. In high-noise
regimes they don't. The right user-facing rule is therefore not
*"`--backend ode` is faster when populations are large"* but rather
*"`--backend ode` answers a different scientific question; in low-noise
regimes the answers converge — verify, don't assume."*

### Diagnostic experiment as a ship gate

Before merge: take the typhoid model at the small end of "stratified
equilibrium" populations (smallest cell ~5,000 — choose the actual
boundary from the typhoid data). Fit ω with both backends. Compare
MLEs ± per-method within-method variance. If they agree to within the
within-method spread, the rule "use `--backend ode` for stratified
equilibrium models" holds and that's what the docs say. If they
diverge meaningfully, the docs guidance becomes more nuanced (a
specific population threshold, or a population-dependent caveat).

This experiment must run before phase 2 ships, not after. Half a
session of work.

## Why NLopt

`nlopt` Rust crate (≥ v0.7, wrapping Steven Johnson's well-tested C
library) gives:

- Many algorithms behind one trait (Sbplx, BOBYQA, COBYLA, ISRES,
  CRS2, etc.).
- Native bounds + linear/nonlinear constraints — no hand-rolled
  log/logit transforms.
- Standard stopping criteria (`xtol_rel`, `ftol_rel`, `maxeval`).
- Algorithm swap is a CLI string — no re-implementation when one
  algorithm struggles on a given likelihood landscape.

### Cargo dependency cost

The `nlopt` crate is a C-FFI wrapper around libnlopt. Implications to
audit before the implementation lands:

- macOS / Linux: cargo build picks up libnlopt via `pkg-config` or
  builds from vendored source. Confirm both modes work in our CI.
- Windows: NLopt builds on MSVC — confirm Windows CI green before
  merge. If it isn't, gate the dependency behind `--features nlopt`
  so the default build remains dependency-free; document the feature
  flag in install instructions.

## Algorithm choice

**Default: Sbplx** (NLopt's `LN_SBPLX`, a robust Nelder-Mead variant).
Compartmental likelihoods are smooth in the interior of the parameter
box but non-smooth at boundaries (degenerate states) and where event
timing depends on parameter values. BOBYQA's quadratic trust region
fails badly when smoothness assumptions break — exactly the typical
compartmental failure mode at a parameter boundary. Sbplx is slower on
truly smooth problems but doesn't fail catastrophically. NLopt's own
documentation recommends Sbplx for "noisy or otherwise non-smooth"
objectives.

### Algorithms exposed in v1

| `--optimizer` | NLopt name | Use case |
|---|---|---|
| `sbplx` (default) | `LN_SBPLX` | Robust to boundary non-smoothness. Default. |
| `bobyqa` | `LN_BOBYQA` | When you know the objective is smooth. Faster. |
| `cobyla` | `LN_COBYLA` | Active linear-inequality constraints. |
| `isres` | `GN_ISRES` | Global, multi-modal "is this the basin?" pass. Slow. |
| `crs2` | `GN_CRS2_LM` | Global, controlled random search. Faster than ISRES. |

### NLopt success-state semantics

`nlopt::SuccessState` distinguishes `Success`, `XtolReached`,
`FtolReached`, `MaxEvalReached`. Our diagnostic must treat
`MaxEvalReached` as a **soft failure** (likely didn't converge —
report and surface) and the others as success. Spell this out at the
dispatch boundary, don't lump them under a single `status` string.

## Integration sketch

```rust
// crates/sim/src/inference/deterministic.rs (new module)

use nlopt::{Algorithm, Nlopt, Target, SuccessState};

pub struct DetOptResult {
    pub params: Vec<f64>,
    pub neg_loglik: f64,
    pub n_evals: usize,
    pub status: SuccessState,
    pub converged: bool,  // true iff status ∈ {Success, XtolReached, FtolReached}
}

pub fn optimize_det(
    compiled: &CompiledModel,
    obs_streams: &[ObservationModel],
    data: &ObsData,
    base_params: &[f64],
    estimated: &[EstimatedParam],
    config: &DetOptConfig,
) -> Result<DetOptResult, String> {
    let objective = |theta: &[f64], _grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let mut params = base_params.to_vec();
        for (i, e) in estimated.iter().enumerate() { params[e.index] = theta[i]; }
        match OdeSim.run(compiled, &params, /* seed irrelevant */ 0,
                         &SimConfig::Ode(config.ode.clone())) {
            Ok(traj) => -compute_obs_loglik(obs_streams, &traj, data),
            Err(_)   => f64::INFINITY,
        }
    };

    let mut opt = Nlopt::new(
        config.algorithm.into(), estimated.len(), objective,
        Target::Minimize, ()
    );
    opt.set_lower_bounds(&estimated.iter().map(|e| e.lower).collect::<Vec<_>>())?;
    opt.set_upper_bounds(&estimated.iter().map(|e| e.upper).collect::<Vec<_>>())?;
    opt.set_xtol_rel(config.xtol_rel)?;
    opt.set_maxeval(config.max_evals as u32)?;

    let mut x: Vec<f64> = estimated.iter().map(|e| base_params[e.index]).collect();
    let (status, neg_loglik) = opt.optimize(&mut x)?;

    let converged = matches!(status, SuccessState::Success
        | SuccessState::XtolReached | SuccessState::FtolReached);
    Ok(DetOptResult { params: x, neg_loglik, n_evals: opt.get_numevals() as usize, status, converged })
}
```

The objective is the only camdl-specific piece; everything else is
NLopt.

## Scope

### What ships

- `crates/sim/src/inference/deterministic.rs` — new module wrapping
  `optimize_det` (above).
- `camdl profile --backend ode --optimizer <algo>` — at each grid
  cell, the inner reoptimisation over nuisance parameters becomes a
  call to `optimize_det` instead of `run_if2`. The grid loop, the
  multi-start logic, the per-cell parallelism all stay.
- `--particles` is silently ignored (or a warning is emitted) when
  `--backend ode`. With ODE the marginal likelihood is exact at
  N=1 — the PF degenerates to a single deterministic forward sim,
  and that's not "a degraded approximation," it's the analytical
  limit as process noise → 0.
- New convergence diagnostics for deterministic chains (below).
- DSL-compatibility check at dispatch: overdispersed models error
  cleanly (already structural via `Capabilities::OVERDISPERSION` —
  no new code, just confirm the dispatch hits this path before
  optimiser construction).
- Diagnostic experiment from §"Two likelihoods, not one optimiser":
  worked, with results in the docs.
- One calibrated example: `vignettes/typhoid` ω profile with
  `--backend ode --optimizer sbplx`, comparing the result to the
  Python wrapper's slice and to chain_binomial profile (small
  population sanity check).

### What does NOT ship in v1

- `camdl fit run --backend ode`. Multi-stage scout/refine, per-stage
  `[stages.X] backend` in fit.toml — the architectural picture
  isn't clear yet (mixed-backend stages: scout=ODE for speed,
  refine=chain_binomial for proper stochastic likelihood — is that
  a feature we want?). Defer.
- Bayesian inference under ODE. MLE-only for v1. The natural future
  path is "fit with NLopt, Laplace-approximate around the optimum"
  rather than a separate Bayesian backend; flag in docs.
- Reactive interventions interacting with deterministic inference.
  Reactive interventions in a stochastic model fire at different
  times across particles; in ODE they fire at one deterministic
  time. The likelihood difference there is non-trivial. The typhoid
  model has no reactive interventions, so this doesn't bind v1, but
  it is a known caveat for future malaria/reactive work — flag in
  docs and "future work."
- Hierarchical priors via deterministic inference (PGAS remains the
  Bayesian path).
- Gradient-based optimisation (would need autodiff over the ODE
  solver).

## Convergence diagnostics for deterministic chains

The existing IF2 gate is compound: per-parameter chain-agreement Â
on iteration trajectories + loglik decibans-spread across chains.
Â on per-iteration trajectories doesn't carry the same meaning when
chains are deterministic NLopt instances. Replacement:

**Leg 1: did chains agree on the basin?** Compare final parameter
vectors across N starts. Report **two numbers**, refuse only if
**both** exceed thresholds:

- *Relative range vs bound width* — catches "starts found different
  basins entirely." Sensitive to bound choice (wide bounds make
  small relative spread look fine; tight bounds make trivial spread
  look bad), so it can't be the only signal.
- *Absolute range in natural-scale space* — catches "starts agreed
  on the basin but the basin is wider than the user's tolerance."

**Leg 2: was the agreed basin actually good?** Loglik spread across
the N converged chains, in decibans, with the same SE-aware floor
as today. SE comes from a single deterministic eval (typically zero,
so the threshold reduces to `decibans_thresh` directly).

UX surface (verdict line) stays the same shape:

```
chain-agreement: rel range = X% bound | abs range = Y nat. units   ✓/✗
loglik-eval:     Δ = X dB / threshold Y dB                         ✓/✗
```

## Estimated cost

5–8 sessions:

- ~400 LOC across `inference::deterministic`, profile dispatch, gate
  replacement, NLopt Cargo plumbing, dispatch capability check,
  integration tests.
- Diagnostic experiment (the "two likelihoods" cross-backend
  comparison): half a session.
- Gate threshold validation against a real worked example: half a
  session.
- Docs: chapter-grade prose explaining the two-likelihoods framing,
  when ODE is the right choice, and the stochasticity-as-noise
  caveat. One session.

(Original estimate of 600–800 LOC / 3–5 sessions was tight. This
is the realistic number once gate validation and the diagnostic
experiment are included.)

## Acknowledged tradeoffs / future cleanup

The user is shipping this on the explicit understanding that
follow-up architectural cleanup may be required. Specifically:

- Multi-stage fit integration is deferred. When phase 2 (fit-side
  ODE) is later picked up, the question of CLI-vs-fit.toml
  precedence and mixed-backend stages will need its own design pass.
  The current proposal does not commit to either answer.
- The convergence-gate replacement is novel for camdl. Threshold
  values may need re-tuning once we have downstream usage.
- Revert window: if downstream feedback reveals a worse-than-expected
  architectural fit, this lands as a small enough surface (~400 LOC,
  one CLI subcommand) that revert-and-redesign is cheap.
