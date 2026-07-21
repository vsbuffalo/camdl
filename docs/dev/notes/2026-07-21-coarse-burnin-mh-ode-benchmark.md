# Coarse `burnin_dt` on `mh`-on-`ode` — measured speedup on Garki `ctl_bb`

Date: 2026-07-21 Project: camdl Tags: ode, mh, burn-in, coarse-burnin,
performance, garki, gh#396

## Headline

**Coarse `burnin_dt` cuts the `ctl_bb` mh+ode per-eval cost by ~2.5–3.5× with a
negligible likelihood change.** At `burnin_dt = 7` (fit `dt = 1`): **3.49×**
per-eval speedup, base-θ loglik shifted **0.08 nats** — an order of magnitude
under the ODE dt-check threshold (~0.5 nats). This is the measured fit-path
confirmation of the fix for the burn-in problem in
`2026-07-12-stan-ctl-bb-benchmark.md` (a ~60.9-year transient re-integrated on
every likelihood eval). Feature:
`docs/dev/proposals/2026-07-08-coarse-burnin-dt.md`.

## Measurements

`ctl_bb` (Kwaru + Ajura × 5 DMT bands, 70 continuous compartments), `mh` on
`ode`, 1 chain × 40 iterations = 41 evals, `init = single` (identical θ-path
across configs, so per-eval time is the clean comparison). Origin 1910, first
obs 1970 → ~60-year unscored warm-up; ~3-year scored window.

| `burnin_dt` | wall (s) | per-eval (s) | speedup | base-θ loglik |
| ----------- | -------: | -----------: | ------: | ------------: |
| off (= dt)  |    32.31 |        0.788 |   1.00× |      −621.314 |
| 3.5         |    13.17 |        0.321 |   2.45× |      −621.328 |
| 7.0         |     9.25 |        0.226 |   3.49× |      −621.233 |

Loglik bias vs `off` at the shared start θ: **0.014 nats** (dt=3.5), **0.081
nats** (dt=7). Both negligible — the coarse warm-up lands on essentially the
same seasonal state at the data-window start.

## Interpretation

- The speedup is real and the bias is small enough to ignore for a fit. `dt = 7`
  is the sweet spot on this model: 3.49× at 0.08 nats.
- **The measured end-to-end speedup (~3.5×) is lower than the ~5× that had been
  quoted from the proposal's forward-solve numbers, and 3.5× is the honest
  figure.** The forward-solve estimate (~5.4× at dt=7) is _burn-in only_; the
  fit additionally integrates the ~3-year scored window at the fine `dt` (not
  coarsened) and carries per-iteration MH overhead. So end-to-end ≈
  `1 / (burnin_frac/c + scored_frac + overhead)`, which lands below the
  burn-in-only ratio.
- Phase 0 has **no forcing-knot snapping**. `ctl_bb`'s vectorial-capacity
  forcing is piecewise-constant with biannual jumps (~180-day spacing), so 7-day
  coarse steps rarely straddle a knot — hence the tiny bias. Pushing to
  `dt = 14+` would invite the knot-straddle bias the proposal's Phase 1
  (`ForcingTimes`) addresses; at `dt = 7` it is clean.

## Reproduction

`scripts/`-free ad-hoc benchmark (scratch): writes three `ctl_bb.toml` variants
(`burnin_dt` unset / 3.5 / 7 on the `[stages.posterior]` mh stage), warms the IR
cache, times 41 evals each, reads the base-θ `log_likelihood` from row 1 of the
chain trace. Binary: the coarse-burnin build (gh#396 branch). Env:
`CAMDL_SKIP_VERSION_CHECK=1`.

## Next

- The same win applies unchanged to any Garki host fit with a long warm-up
  (`ctl_bb*`, `mosq_dyn_pool*`, the compound/immladder models, and the ordered
  density-class fit `ctl_prev_density_ladder_dclass`) — they all pay the
  ~60-year transient. Handoff to the garki agent covers applying it.
- Phase 1 knot-snapping if a model needs `burnin_dt` past the forcing-knot
  spacing; not needed at `dt = 7` on `ctl_bb`.
