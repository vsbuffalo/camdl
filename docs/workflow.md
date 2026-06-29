# The camdl fit workflow

The canonical path from a model to a calibrated, diagnosed fit. Every command
here is verified against the current CLI; a run-gate that executes the whole
sequence against a fixture — so this doc can't drift — is the companion next
step.

This is the procedural runbook. The _why_ behind it — identifiability, the
necessary-not-sufficient logic of synthetic recovery, the worked WA-State case
study — lives in `camdl docs concepts`. The full `fit.toml` schema lives in
`camdl docs fit-toml`. Writing the model itself is `camdl docs getting-started`
and `camdl docs language`.

Calibration is the half of compartmental modelling that is _identifiability and
prior judgement_, not engineering — see "When to stop and ask a human."

## At a glance

```
write model → check → simulate (sanity) → synthetic recovery → survey
            → write fit.toml → fit run → fit summary → read diagnostics
            → (refine priors, repeat) → validate
```

## 1. Compile and sanity-check the dynamics

```bash
camdl check model.camdl
camdl simulate model.camdl --params p.toml --obs sim.tsv --seed 1
```

`simulate` defaults to the `chain_binomial` backend — the same default `fit`
uses, so a forward sim of an MLE reproduces the fit's dynamics. Pass
`--backend gillespie` (exact SSA) or `--backend ode` (deterministic) to switch.
The same `observations {}` block that _scores_ real data in the fit also
_samples_ synthetic data here via `--obs` — that duality is why the next step is
a valid test. Look at `sim.tsv`: is the curve epidemiologically reasonable
(timing, peak, final size)?

## 2. Validate the pipeline on synthetic data — necessary, not sufficient

Generate data from _known_ truth, fit it back, confirm you recover the truth. Do
this **before** real data: it separates "my pipeline works" from "my model fits
the world."

```bash
camdl simulate model.camdl --params truth.toml --backend chain_binomial \
    --dt 0.5 --seed 7 --obs synth.tsv
camdl survey model.camdl --fit fit_synth.toml --render
camdl fit run fit_synth.toml --label synth --seed 0
camdl fit summary @synth
```

If θ̂ doesn't recover the truth within its CI, stop — the problem is the pipeline
(structure, `dt`, observation model), not the data. Choose `dt` small relative
to the fastest rate (a few steps per mean dwell time); when unsure, halve it and
watch the dt-convergence verdict. A coarse `dt` silently biases estimates in a
way recovery at the _same_ `dt` won't reveal (why: `camdl docs concepts`).

## 3. Map the landscape before the real fit

```bash
camdl survey model.camdl --fit fit.toml --render
```

`survey` does Latin-hypercube landscape sampling — the cheapest hour in the
pipeline. It exposes basins, ridges, and bound-pinning _before_ you commit a
fit. **Agent note:** trust the numerical top-k, not the scatter geometry — your
read of "one clear basin" vs "a ridge" from the rendered HTML is often wrong.
Surface `survey.html` and let a human confirm before you seed a fit on it.

## 4. Write `fit.toml` and run the fit

A `fit.toml` names the model, the data, what to estimate, what to fix, and the
inference stages. Minimal, complete:

```toml
[model]
camdl = "model.camdl"

[data.observations]
cases = "data/cases.tsv" # one key per observation stream in the model

[estimate] # bounds always; a posterior stage needs a prior (here, or in the model via ~)
beta = { bounds = [0.001, 0.5], start = 0.04, prior = { log_normal = { mu = -2.0, sigma = 1.0 } } }
gamma = { bounds = [0.01, 1.0], start = 0.12, prior = { log_normal = { mu = -1.2, sigma = 0.5 } } }

[fixed]
rho = 0.6
k = 10.0

[stages.scout] # stages are USER-NAMED; `algorithm` picks the method
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 150
cooling = 0.7

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 600
sweeps = 300
```

- **Stages are user-named** `[stages.<name>]` blocks; `algorithm` (`if2` |
  `pgas` | `pmmh` | `pfilter`) picks the method. The conventional pipeline is
  **scout** (`if2`, find the basin) → optionally **refine** (`if2`, sharpen) →
  **posterior** (`pgas`, sample) → **validate** (`pfilter`, score).
- **Priors** for a `pgas`/`pmmh` stage must be explicit — declared here in
  `[estimate].prior` or in the model (a `~` declaration); it refuses implicit
  flat. Menu: `log_normal {mu,sigma}` · `normal {mean,sd}` · `beta {alpha,beta}`
  · `uniform` · `half_normal`.
- Fits run `chain_binomial` (needed for chain-binomial process noise and
  `balance`).
- Full schema — every section, every stage field, transforms, holdout:
  `camdl docs fit-toml`.

```bash
camdl fit run fit.toml --label baseline --seed 1
```

`--label baseline` names the fit so every downstream verb can refer to it as
`@baseline` instead of its run directory (see §5). While tuning, run one stage
at a time: `camdl fit run fit.toml --stage scout`.

## 5. Read the diagnostics

```bash
camdl fit summary @baseline      # or the run directory `fit run` printed, or a hash prefix
```

Every fit verb — `fit summary`, `fit predict`, `compare` — takes the same
**handle**: an `@label`, a fit hash-prefix, the run directory `camdl fit run`
prints, or the `fit.toml` itself (resolved to its unique run; an ambiguous match
is listed rather than guessed). A handle beats hunting for a path, and because
each run archives its own compiled model, the fit stays resolvable even if the
`.camdl` moved. `camdl list` enumerates every run if you need to look one up.
The summary prints a fixed set of blocks:

- **best loglik (loglik-eval)** — the MLE _re-scored_ at a high particle count.
  The clean number; IF2's running loglik during optimization is perturbation-
  biased.
- **The scout-convergence gate — two legs, both must pass:**
  - **Â (chain-agreement):** `< 1.05` ✓, `1.05–1.10` marginal, `≥ 1.10` ✗. "Did
    the independent optimizer chains climb to the _same place_?"
  - **Δ_dB (decibans spread):** best-vs-worst chain loglik spread, vs a ~30 dB
    SE-aware threshold. "Was where-they-agreed any _good_?" Chains can pass Â
    while sitting in basins thousands of dB apart — this leg is the catch.
- **per-chain loglik-eval** — re-scored loglik ± SE per chain; `← selected`
  marks the MLE.
- **ESS at θ̂** — particle-filter effective sample size (`min`/`mean` over
  observation steps). A single-digit `min` at the most informative observation
  is tolerable; `min ≈ 1` _everywhere_ means the loglik at θ̂ is unreliable —
  raise the particle count.
- **dt-convergence (Richardson)** — loglik at θ̂ re-scored at `dt`, `dt/2`,
  `dt/4`. `PASS` = the MLE survived finer discretization.

### Â vs R̂ — do not conflate them

`Â` is for the IF2 **optimizer** (chain agreement — "did the optimizers converge
to the same point?"). `R̂` (`rhat`) is the genuine MCMC **mixing** diagnostic
camdl reports — _by that name_ — for the PGAS / PMMH **posterior**, where the
chains are real posterior draws. `camdl fit summary` reports e.g.
`max R̂ = 1.216 ✗` when the posterior hasn't mixed. **Posterior mixing is read
off R̂, NUTS divergences, and trajectory renewal — never Â.** Importing R̂'s
"above 1.01 keep sampling" reflex onto Â is the wrong mental model.

### Diagnostics reference

| Stage            | Diagnostic           | Healthy         | Warning           | Action                                         |
| ---------------- | -------------------- | --------------- | ----------------- | ---------------------------------------------- |
| Particle filter  | ESS per-obs          | > 50% of N      | 10–50%            | more particles or looser obs model             |
| Particle filter  | ESS at MLE           | mean > 50%      | mean < 30%        | estimate `σ²` or `k`                           |
| Particle filter  | log-likelihood       | finite          | `−∞`              | check starts / model structure                 |
| IF2 (MLE)        | Â                    | < 1.05          | 1.1–1.5           | more iterations                                |
| IF2 (MLE)        | Â                    | —               | > 1.5 + LL spread | multimodal surface, more chains                |
| IF2 (MLE)        | logit position \|z\| | < 2             | > 3               | widen bounds or change transform               |
| PGAS (posterior) | R̂                    | < 1.05          | > 1.1             | more sweeps; check multimodality with `survey` |
| PGAS (posterior) | trajectory renewal   | > 30%           | < 10%             | more CSMC particles or tempering               |
| PGAS (posterior) | NUTS divergences     | 0               | any               | reduce step size, reparameterize               |
| PGAS (posterior) | NUTS max tree depth  | < 20% of sweeps | > 50%             | increase `max_treedepth`                       |
| PGAS (posterior) | tempering swap rate  | 20–50%          | < 10%             | denser temperature ladder                      |

## 6. When it doesn't converge: a failing gate is information

Don't tune a failing diagnostic away. The canonical illustration is the WA-State
cryptic-introduction fit (estimating the early-COVID introduction time τ): under
uninformative priors the `n_seed` posterior pins to its upper bound and
`fit summary` reports `max R̂ = 1.216 ✗` — because the data alone can't separate
a small-late from a large-early introduction (the `(τ, n_seed)` ridge). The fix
isn't more sweeps; it's **weakly-informative structural priors** (a `log_normal`
on seed size, a `beta` on reporting) that regularize the non-identified
direction — after which R̂ passes, the posterior is unimodal, and τ agrees with
the independent Bedford genomic estimate. The failing R̂ was _correct_: it said
"the data don't identify this," not "run longer."

Full case study — prior-predictive checks, the ridge, PSIS-LOO, prequential
elpd, mechanism and prior sensitivity: `camdl docs concepts`.

## 7. Validate the fit

### Predicted-vs-observed in one verb: `camdl fit predict`

The workhorse posterior check — overlay the observed data on the fitted model's
predictive band — is a single verb that reads the fit and writes a tidy,
plot-ready artifact:

```bash
camdl fit predict @baseline --stream onset
# wrote results/fits/sle-8a3f12b4/predictive/onset.tsv
# wrote results/fits/sle-8a3f12b4/observed/onset.tsv
```

`fit predict` resolves the fit's canonical post-warm-up posterior draws,
forward-simulates each draw through the **real** observation model (sampling
`y_rep`, not the projected mean, so observation noise is in the band),
integrates over the cloud by pooling and quantiling, and writes two files under
the run directory:

- `predictive/<stream>.tsv` —
  `time | <dims…> | horizon | treatment | rhat_max | ess_min | n_draws | q05 q25 q50 q75 q95`.
  The `q05…q95` columns are the ribbon; the `horizon` and `treatment` columns
  make the two predictive axes explicit (so an honestly-wide posterior band is
  never confused with a narrow plug-in one), and `rhat_max`/`ess_min` carry the
  fit's own convergence numbers alongside every band.
- `observed/<stream>.tsv` — `time | <dims…> | value`, the recorded series in the
  same tidy keys.

A consumer reads both files, joins on `(time, <dims>)`, and plots `observed`
over the `predictive` ribbon — one facet per stratum, with no run-store, DSL, or
likelihood knowledge:

```python
import polars as pl
pred = pl.read_csv(".../predictive/onset.tsv", separator="\t")
obs  = pl.read_csv(".../observed/onset.tsv",   separator="\t")
```

**Two horizons answer two questions** (see
[`camdl docs diagnosing-fits`](diagnosing-fits.md) §5). Omit `--horizon` to emit
all applicable for the fit's backend:

- `--horizon free_forward` — the generative check: replay the fitted model from
  the start, never re-anchored to data. The harshest test, and what exposes
  generative misspecification. Available on any backend.
- `--horizon one_step` — the honest short-horizon forecast `p(y_t | y_{1:t-1})`:
  re-condition on the data each step. Chain-binomial only (an ODE fit's one-step
  is identical to its free-forward and is refused with a redirect). Pools over a
  posterior subsample (`--n-draws`, default 200); both horizons stack in the
  same file, distinguished by the `horizon` column.

`fit predict` refuses an optimizer fit (IF2 / NLopt) with an actionable message
— such a fit returns one best-fit point, not a distribution, so there is no
posterior band to draw; get its parameters with
`camdl fit summary <run>
--params-only` and run a plug-in
`camdl simulate --params …` instead.

If the model declares a `quantities {}` block, `fit predict` also bands each
derived quantity (peak size, attack rate, time-to-peak, …) over the same
posterior draws into `quantities/<name>.tsv` with a `quantities.json` manifest —
the posterior of a reported summary, not just of the fitted series. See
[`camdl docs user-features`](user-features.md) ("Reporting derived quantities").

### Other validation steps

```bash
# Prior-predictive — do the priors imply plausible epidemics?
camdl simulate model.camdl --draws prior --fit fit.toml -n 200 --obs prior_ppc.tsv

# Identifiability — profile a suspect parameter (1D or 2D)
camdl profile model.camdl --particles 1500 \
    --fixed gamma=0.1 --sweep "tau=lin(-60,5,12)" --fit fit.toml

# Model comparison — prequential predictive scoring, straight from two fits
camdl compare @baseline @candidate --particles 2000 --seed 7
```

`camdl compare` ranks models by prequential elpd / CRPS / PIT. Passed two fit
handles, it **auto-derives** each model's prequential at θ̂ via `pfilter` — the
same particle count and seed for both, so the scores stay commensurable — so you
no longer hand-run `pfilter --save-prequential` first. (You still can: an
explicit `prequential.json` path is read as-is, for a custom filter
configuration.) Its scores are **plug-in and in-sample-optimistic** — computed
at a single θ that was fit to the whole series — so they are useful for
_relative_ comparison but are not a leave-future-out forecast score; `compare`
prints this caveat on every run.

## When to stop and ask a human

Agents are good at _running_ the workflow; humans own the _modeling decisions_.
Pause and surface, don't decide unilaterally, when:

- **Choosing a prior shape** for a parameter you lack domain context for.
  Picking `Normal(0,1)` "to make PGAS run" puts that choice straight into the
  posterior.
- **A convergence gate fails.** Diagnose _why_ (bounds? more chains? more
  iterations? multimodality?) — never lower the threshold to make it pass.
- **`survey` geometry.** Your read of the scatter is unreliable; surface the
  HTML.
- **An external oracle disagrees** (a genomic estimate, a published result) —
  raise it rather than tuning until the numbers match.
