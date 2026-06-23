# One-step-ahead posterior predictive — design sketch

Date: 2026-06-23 Project: camdl (pred-ergo) Tags: predictive, particle-filter,
one-step, prequential, capability-gating

## Context / what we're building

The one-step-ahead posterior predictive distribution `p(y_t | y_{1:t-1})` for an
observation stream, folded into `camdl fit predict` as
`horizon=one_step,
treatment=posterior` rows in the existing
`predictive/<stream>.tsv`. The honest short-horizon forecast object — "given
everything observed through `t-1`, what does the model say `t` will be?" — the
operational complement to the shipped free-forward generative check.

Scope: chain-binomial only (the particle-filter backend). ODE and Gillespie are
**typed out**, not silently mishandled. PGAS is the workhorse and runs on
chain-binomial, so the supported path is exactly the one PGAS produces.

## The quantity, and why it differs from free-forward

For each posterior draw `θ` from the fit, run a bootstrap particle filter
forward over the data. At each observation time `t`, **before** assimilating
`y_t`, the propagated particles `{x_t^{(j)}}` are distributed as
`p(x_t | y_{1:t-1}, θ)`; drawing `ỹ ~ p(y | x_t^{(j)}, θ)` from them is a sample
from the one-step predictive `p(y_t | y_{1:t-1}, θ)`. Pool `ỹ` across particles
and across the posterior `θ` draws → quantile per `(time, stratum)` → the
posterior one-step band.

The contrast with free-forward is the diagnostic, not bookkeeping. Free-forward
never re-anchors to data, so a model whose dynamics drift produces a band that
blows up over time; one-step re-conditions every step, so it stays tight **iff**
the filter can track the data. A model can pass the free-forward check (wide
enough to contain the data) yet fail one-step (can't predict next week) and vice
versa. Hence both, with a typed `horizon` column so neither is read as the
other.

## Key finding: the recorder exists; we re-run it per draw (gh#269)

**We re-calculate — there is no saved file to read.** The fit does not produce
this quantity: PGAS fits by conditional SMC (conditioned on a reference path),
not a bootstrap forward filter, and persists only the posterior θ-draws
(`draws.tsv`) and the _smoothed_ trajectories `X|θ,y`. The one-step predictive
is a forward-filter, past-only quantity (`p(y_t | y_{1:t-1})`) — a different
object from anything the fit saves, and not derivable from the smoothed path.
gh#269's prequential is also not it (a separate single-θ `camdl pfilter` run,
not stored per draw). So `fit predict --horizon one_step` runs a bootstrap
filter **per posterior draw** over the data and pools — the statistically
correct estimate of `(1/D) Σ_d p(y_t | y_{1:t-1}, θ_d)`, each term one filter
pass at `θ_d`.

What already exists is the **recorder**, not the samples: the filter knows how
to emit the one-step samples without perturbing its loglik, so we write no new
filter-loop code — but we **execute** it `D` times. From `particle_filter.rs`:

```rust
/// Captured BEFORE obs-reweight and BEFORE resampling, so particles
/// are distributed as the one-step-ahead predictive p(x_t | y_{1:t-1}).
pub struct PrequentialRecorded {
    pub obs_times: Vec<f64>,
    /// [obs_idx][stream][particle] = per-stream predictive draw ỹ^stream.
    pub per_stream_samples: Vec<Vec<Vec<f64>>>,
    // … log_liks, per_stream_log_liks (the scores) …
}
```

Populated when `SMCConfig.record_prequential = true`. gh#269 already established
the load-bearing property: the predictive draw **shares the one RNG-consuming
`obs_model.sample(...)` call** with the score, so recording it does **not**
perturb the filter's weights/resampling — the loglik is byte-identical with
recording on vs off. So the inference-math part is done and validated; we are
not adding a side-output to the filter loop.

What gh#269 ships is the **plug-in** prequential (a single `θ`, via
`camdl
pfilter`). The one-step _posterior_ predictive is the same machinery run
**per posterior draw** and pooled — so it carries parameter uncertainty
(`treatment=posterior`), the honest version.

## Backends — supported via types, others typed out

The filter propagates the stochastic discrete-time process, so the one-step
predictive is only meaningful on **chain-binomial**:

- **chain_binomial** — ✅ filterable. PGAS/PMMH/Mh fits run here. The whole
  point.
- **ODE** — ✗ deterministic given `θ`: there is no process noise to integrate
  out, so `p(x_t | y_{1:t-1}, θ) = δ(x_t(θ))` and `p(y_t | y_{1:t-1}, θ)`
  reduces to the observation model at the deterministic state, **identical to
  free-forward**. Offering a separate `one_step` band for ODE would be a
  relabelled duplicate — refuse and point at `free_forward`.
- **Gillespie** — ✗ not an inference backend (no `InferenceBackend::Gillespie`),
  so a fit never has Gillespie draws; handled for exhaustiveness only.

The gate is **in the type**, not a runtime `if` the caller can forget. The
one-step entry point takes a witness that can only be built from a filterable
fit:

```rust
/// Proof that a fit's draws can drive a particle filter (the one-step / filtered
/// horizons). The one-step entry point takes this BY VALUE, so it cannot be
/// called for a non-filterable fit. The only constructor is the backend gate.
pub struct FilterableFit {
    draws: PosteriorDraws,   // private — see `from_posterior`
}

pub enum NotFilterable {
    /// ODE: deterministic → one-step ≡ free-forward.
    Deterministic,
    /// Gillespie: not an inference backend.
    NotAnInferenceBackend,
}

impl FilterableFit {
    pub fn from_posterior(d: PosteriorDraws) -> Result<Self, NotFilterable> {
        match d.backend {
            ForwardBackend::ChainBinomial => Ok(FilterableFit { draws: d }),
            ForwardBackend::Ode           => Err(NotFilterable::Deterministic),
            ForwardBackend::Gillespie     => Err(NotFilterable::NotAnInferenceBackend),
        }
    }
}

// Free-forward takes any cloud:
fn free_forward(draws: &PosteriorDraws, …) -> Result<Vec<StreamBands>, String>;
// One-step REQUIRES the witness — non-chain-binomial can't reach it:
fn one_step(fit: &FilterableFit, …)       -> Result<Vec<StreamBands>, String>;
```

So `--horizon one_step` is _requestable_ for any fit, but the run is gated: a
chain-binomial fit constructs the witness and proceeds; an ODE fit fails to
construct it and gets the actionable redirect ("this fit is ODE; its one-step
predictive is identical to free-forward — use `--horizon free_forward`"). There
is no code path that produces a silently-degenerate one-step band. (`backend`
already rides on `PosteriorDraws` — threaded in the review-fix commit.)

PGAS path, end to end: PGAS → chain_binomial → `FilterableFit::from_posterior`
succeeds → one-step runs. The workhorse is the supported case by construction.

## Where it lives — the existing `fit predict` surface

No new subcommand. `fit predict` already resolves the posterior draws, compiles
the model, loads the observed data + cadence, and writes the tidy artifact. The
horizon selects the producer:

```
fit predict --fit <run>                    # DEFAULT: all applicable horizons, stacked
fit predict --fit <run> --horizon one_step # just one-step
fit predict --fit <run> --horizon free_forward  # just free-forward
```

**Default emits all _applicable_ horizons, stacked** in one file: a
chain-binomial fit → `free_forward` + `one_step`; an ODE fit → `free_forward`
only (its one-step is identical, so it is simply not applicable — no error). The
ODE hard-error fires only on an **explicit** `--horizon one_step` request (you
asked for something specific that doesn't apply; don't silently no-op). So the
default does the sensible thing per backend; an explicit impossible request
errors.

`one_step` rows land in the **same** `predictive/<stream>.tsv`, at the
observation times, with `horizon=one_step, treatment=posterior` — they join the
`observed/` series on `(time, <dims>)` exactly like the free-forward rows. The
two axes were made columns precisely so this is "more rows," not new consumer
code.

## Orchestration & cost

```
resolve draws  →  FilterableFit gate  →  for each draw θ:
    bootstrap_filter(process, obs, θ, SMCConfig{ record_prequential: true, … }, seed_θ)
        → PrequentialRecorded.per_stream_samples : [obs_idx][stream][particle]
  →  pool ỹ over (particle × draw) per (obs_time, stream-leaf)
  →  band() quantiles  →  one_step rows
```

Reuses the same process+obs setup `camdl pfilter` / the fit's pfilter stage
already build (touched in the loader-unification commit), so the per-draw call
is just `bootstrap_filter` with the draw's params.

Cost is the real consideration: `draws × N_particles × T`, i.e. `N_particles×`
the free-forward cost. Mitigations, all already-available knobs:

- subsample the posterior cloud (`--n-draws`), since the band pools
  `draws × N_particles` samples per cell — a few hundred draws is plenty for
  `q05…q95`;
- a modest particle count for prediction (it need not match the fit's `N`);
- emit from all `N` particles per step (the recorder already keeps them), so
  even a small draw subsample yields a dense band.

Document the cost and default to a sensible subsample; never silently run the
full cloud at fit-grade `N`.

## Reuse vs new

| piece                                                                          | status               |
| ------------------------------------------------------------------------------ | -------------------- |
| posterior draws resolver, `backend` on the cloud                               | done                 |
| `bootstrap_filter` + `record_prequential` + `per_stream_samples` (the samples) | **exists (gh#269)**  |
| byte-identical-loglik property of recording                                    | **exists (gh#269)**  |
| process + obs setup (per-draw filter input)                                    | reuse `pfilter` path |
| `band()` quantile reduction + `predictive/<stream>.tsv` writer                 | done                 |
| `FilterableFit` type gate                                                      | new (small)          |
| per-draw filter orchestration + pooling                                        | new (the bulk)       |
| `Horizon::OneStepAhead` + `--horizon` flag wiring                              | new (small)          |

So the inference-math surface is **not** touched — this is orchestration over an
existing, validated predictive recorder, behind a type gate.

## Scores — band first, scores as the immediate fast-follow

One-step is genuinely out-of-sample, so the proper scores (`log_score`, `crps`,
randomized PIT for counts) are meaningful here (unlike free-forward, where they
are optimistic). **Decided:** ship the band (`q05…q95`) first; scores follow
immediately (the maintainer wants them soon) as extra columns on the `one_step`
rows — left blank on `free_forward` rows, where they would mislead. Not a big
lift: the ingredients are already recorded in `PrequentialRecorded` (`log_liks`
/ `per_stream_log_liks` for the log score; `per_stream_samples` for CRPS and
PIT), and the observed value per `(time, stratum)` is already loaded
(`observed/`), so it is three standard sample-based estimators + columns + one
fixed-seed `U` for the PIT randomization (~1–2 days):

- **log score** `log p(y_t | y_{1:t-1})` = `logsumexp` of the per-particle
  `log_liks` pooled over (particles × draws) − `log(N·D)`.
- **CRPS** — empirical CRPS from the pooled predictive samples.
- **randomized PIT** — `F(y−1) + U·(F(y) − F(y−1))` from the pooled samples.

## Decisions (resolved)

1. **Particle weighting for `ỹ`** — pool the pre-reweight (uniform) particles
   straight; that is `p(x_t | y_{1:t-1})`.
2. **Cost defaults** — default a posterior-cloud subsample (`--n-draws`) and a
   modest prediction `N_particles`; never silently run the full cloud at
   fit-grade `N`.
3. **ODE behaviour** — **hard-error** on an explicit `--horizon one_step` for an
   ODE fit, with the redirect to `free_forward` (a relabelled-identical band is
   the mislabeling the typed columns exist to prevent). The default skips it for
   ODE (not applicable), no error.
4. **`--horizon` surface** — **all by default** (all applicable horizons,
   stacked in one file); `--horizon free_forward|one_step` narrows.
5. **Scores** — fast-follow, on `one_step` rows only (above).
6. **Naming (parked)** — `horizon=one_step` value and `FilterableFit` name
   follow the horizon/conditioning naming decision when it lands.

## Next

Implement behind `FilterableFit`; first test is the byte-identical-loglik gate
(run a fit-stage filter with `record_prequential` off vs on, assert the loglik
is unchanged) so the predictive path provably can't perturb inference — then the
per-draw orchestration, then the `one_step` rows, with an e2e that asserts a
chain-binomial fit yields one-step rows and an ODE fit is refused with the
redirect.
