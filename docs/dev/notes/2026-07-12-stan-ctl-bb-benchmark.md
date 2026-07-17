# Stan re-implementation of Garki `ctl_bb` — ODE-fit benchmark vs camdl

Date: 2026-07-12 Project: camdl Tags: ode, stan, benchmark, performance, garki,
burn-in

## Headline

**NUTS-through-a-60-year ODE is impractical for `ctl_bb`.** A faithful Stan
model compiles and samples in the right region, but a full fit extrapolates to
**~7 hours (optimistic floor) to ~3–4 days (realistic) per chain**, vs camdl's
**~28 minutes** for the whole 6-chain fit. The cost is the ~60.9-year burn-in:
every NUTS gradient differentiates through a ~23,355-step RK4 tape, and NUTS
pays tens-to-thousands of gradients per iteration where camdl's Metropolis–
Hastings pays one forward solve. This independently reproduces garki's own
finding (friction-log F21).

A **separately actionable sub-finding** is in §5: Stan's tight compiled RK4
forward solve (0.055 s) is faster than camdl's resolved-expression evaluator per
solve. The wall-clock loss is entirely NUTS × burn-in, not Stan's engine — and
the per-solve gap is a standalone lever for camdl's own ODE fits.

## 1. Model and faithfulness

Translated verbatim from `models/ctl_bb.camdl`:

- **70 ODE states** = 2 villages (kwaru, ajura) × 5 age bands × 7 compartments
  (S_naive, E_naive, I_infectious, I_recovering, I_immune, S_immune, E_immune).
- Saturating FOI `h_v = g·(1 − exp(−Cscale·C_v(t)·(I_infectious_vil/N_vil)))`.
- Superinfection-corrected recovery `R1 = h/(exp(h/r1)−1)`, `R2` with
  `r2 = 10·r1` (implemented with `expm1`; algebraically identical).
- Incubation `nu`, loss-of-infectivity `a1`, immunity acquisition `a2`.
- Aging across 5 bands at `1/age_dur`, widths [4,5,5,10,20] yr; birth = death
  `delta` (population conserved). **Year = 365.2425 days**, camdl's
  Gregorian-average constant (`caltime.rs` / `expander.ml`), so aging rate/day =
  `1/(age_dur_yr·365.2425)`.
- **Forcing** `C_v(t)`: piecewise-constant, **left-continuous step** (value at
  the largest knot ≤ t, clamped at ends), matching
  `rust/crates/sim/src/propensity.rs::constant_value`. 129 shared knot dates,
  per-village values, from `data/processed/mv_village_C.tsv`.
- **Observation**: beta-binomial over patent-prevalence cells. Patent
  `p = (q1·I_infectious + q2·I_recovering + q3·I_immune)/N` (q1=q2=1, q3=0.7),
  concentration `φ = 1/phi_inv`, `α = p·φ`, `β = (1−p)·φ`, `beta_binomial_lpmf`.
- **Fitted (4)**: g, r1, a2, phi_inv (matches `fits/ctl_bb.toml [estimate]` and
  the recorded fit's `fit.meta.json`; the model file gives a1 a prior but the
  fit **fixes** a1=0.002). Priors/bounds exactly as ctl_bb.toml:
  g~lognormal(−2.50,0.8)∈[0.01,0.5]; r1~lognormal(−6.50,0.7)∈[5e-4,0.02];
  a2~lognormal(−8.50,1.0)∈[2e-5,2e-3]; phi_inv~half_normal(1.0)∈[1e-4,3.0].
- **Solver**: hand-rolled **fixed-step RK4, dt = 1 day** (camdl
  `[config]
  dt=1.0`, `backend = ode`). Deliberately NOT `ode_rk45`: camdl
  friction-log **F17** records that an adaptive solver restarts and takes tiny
  steps at each of the ~130 forcing discontinuities, making it slower than
  fixed-step RK4. Hand-rolled RK4 also gives a clean, predictable per-gradient
  cost and the closest apples-to-apples with camdl.

**Numeric faithfulness check.** Stan forward solve at camdl's recorded MAP
(g=0.0481, r1=0.001154, a2=0.000245, phi_inv=0.0900) reproduces the age
gradient:

| age band | Stan pred p | observed p |
| -------- | ----------- | ---------- |
| a1_4     | 0.699       | 0.824      |
| a5_9     | 0.692       | 0.812      |
| a10_14   | 0.598       | 0.627      |
| a15_24   | 0.473       | 0.369      |
| a25_44   | 0.307       | 0.238      |
| overall  | 0.554       | 0.574      |

The declining-with-age patent prevalence (the model's signature mechanism) is
reproduced, and the _young-under / old-over_ residual matches the pattern
camdl's own report documents. This is a **structural-faithfulness** check, not
golden bit-parity (different sampler/RNG; standard Stan Jacobian on the bounded
transforms).

**Simplifications**: none in the dynamics — full 60.9-yr burn-in, dt=1, all 5
bands, both villages, no equilibrium shortcut.

## 2. Environment / versions

- **cmdstan 2.39.0** (built from source via `cmdstanpy.install_cmdstan`),
  **cmdstanpy 1.3.0**, Python 3.12.11, Apple clang 17.0.0, macOS (M4 Max).
- **camdl 0.1.0+61551324** on PATH; the recorded `ctl_bb` fit used **camdl
  0.1.0+5c527817** (2026-07-10) — cited with that provenance below.

## 3. Burn-in handling and its cost

Origin **1910-01-01**; first observation **1970-11-17** = day 22235 = **60.88
yr** burn-in; last obs 1973-12-11 = day 23355 = 63.94 yr total. At dt=1 the
faithful solve is **~23,355 RK4 steps** (×4 deriv evals/step ×70 states). The
Stan model integrates from 1910 with the seeded IC (S_naive=900,
I_infectious=100 in the youngest band per village), exactly as camdl does — no
equilibrium shortcut. The system reaches an annual limit cycle within a few
years (pre-1971 forcing is a repeating seasonal climatology), so the long
burn-in tail is wasted work in both frameworks; it is precisely the cost that
camdl's gh#396 "coarse burn-in dt" / equilibrium-warm-start work targets.

## 4. Timing (measured, then extrapolated)

All on M4 Max, cmdstan 2.39.0, dt=1 fixed-step RK4, single thread.

### Measured anchors (real numbers)

| quantity                                    | value                                      | how measured                          |
| ------------------------------------------- | ------------------------------------------ | ------------------------------------- |
| **forward solve** (double, no grad)         | **0.055 s**                                | 50 `fixed_param` GQ evals (2.77 s)    |
| **single reverse-mode gradient**            | **1.0–1.36 s**                             | Stan "Gradient evaluation took …", ×2 |
| gradient / forward ratio                    | **~18–25×**                                | derived (memory-bound tape)           |
| 1 warmup iter, max_depth=2 (≤4 leapfrog)    | **2.2 s**                                  | tiny run (real lower bound)           |
| sampling iter, max_depth=5 (~10 leapfrog)   | **9.1 s/iter**                             | completed 6-iter run                  |
| diagnostic (8 warmup + 6 sampling, depth≤5) | **163.3 s** (108.6 warmup + 54.7 sampling) | full short run                        |
| adaptation run (40 warmup, depth≤6)         | killed at **222 s**, barely past warmup    | ~60–90 s/iter with adapted step       |

The sampler moved to the right region even in the tiny run (g=0.047, r1=0.00133,
a2=0.000259, phi_inv=0.107 — all near camdl's MAP), so this is a cost problem,
not a wiring problem.

### Extrapolation to a full fit (order-of-magnitude)

Per-iteration cost ≈ (avg leapfrog steps) × (per-gradient cost). Gradient ≈ 1.2
s.

- **Optimistic floor — 10 leapfrog/iter**: 2000 × 10 × 1.2 s = **~6.7
  hr/chain**. (Stan's own printed estimate: ~5.6–7.6 hr for 2000.)
- **Realistic — ~128 leapfrog/iter** (depth ~7; the g–r1 ridge forces a small
  adapted step ⇒ deep trees; camdl's own R̂≈2.2 documents the ridge): 2000 × 128
  × 1.2 s = **~3.6 days/chain**.

Chains are independent, so with enough cores wall ≈ per-chain: **~7 hr (floor)
to ~3–4 days (realistic).** The killed 40-warmup run (≈60–90 s/iter) sits in the
realistic regime.

## 5. Comparison to camdl (garki's recorded ctl_bb fit)

Source: `results/fits/ctl_bb-b35f911a/.../run.json` (`wall_time_seconds`) +
`pmmh_summary.json`, produced by **camdl 0.1.0+5c527817**, 2026-07-10.

|                     | camdl `ctl_bb` (mh-ode)                            | Stan `ctl_bb` (NUTS), extrapolated    |
| ------------------- | -------------------------------------------------- | ------------------------------------- |
| sampler             | adaptive MH, 1 forward solve/iter                  | NUTS, ~10–1000 gradient solves/iter   |
| config              | 6 chains × 2000 iters (burn_in 1000)               | ~4 chains × ~2000 iters               |
| **wall (full fit)** | **1682.8 s ≈ 28.0 min**                            | **~7 hr (floor) – ~3–4 days (ridge)** |
| slowdown vs camdl   | 1×                                                 | **~15× (floor) – ~150–200× (ridge)**  |
| convergence         | R̂ g=2.22, r1=2.38, a2=1.97; phi_inv=1.06 (ESS 360) | not reached                           |
| ESS/sec (phi_inv)   | 360 / 1682.8 = **0.21 ESS/s**                      | —                                     |

**camdl's 28 min ≠ converged.** Even camdl does not converge the ridge params in
28 min (R̂ ≈ 2 for g, r1, a2; only phi_inv, R̂=1.06, ESS=360, is usable). The
`ctl_bb` posterior is a genuinely hard g–r1 identifiability ridge with one stuck
chain. So the target is hard for _both_ samplers; NUTS's theoretical edge
(better ESS/iter on hard geometry) is swamped by the per-gradient burn-in cost.

**Sub-finding — the solver gap (standalone camdl lever).** Stan's hand-rolled
forward solve (0.055 s) is _faster_ than camdl's implied per-mh-iteration
(1682.8 s / 2000 iters ≈ 0.84 s wall/iter across 6 parallel chains — mostly one
forward solve + MH overhead). Same algorithm (RK4 dt=1), same span: a tight
**compiled** RK4 beats camdl's **resolved-expression evaluator** per solve. If
camdl's per-solve cost closed toward Stan's, camdl's own ODE fits would speed up
roughly an order of magnitude — a lever independent of the NUTS-vs-MH question.
Needs its own measurement + profile (interpreted AST eval vs native codegen; cf.
the known `eval_resolved`-dominated sim profile).

## 6. Honest caveats (not apples-to-apples)

1. **Sampler.** camdl = random-walk adaptive MH (one forward solve/iter); Stan =
   NUTS (many gradient solves/iter). The dominant, structural difference.
2. **Autodiff cost is the burn-in tape, not the model.** The 1.2 s gradient is
   ~20× the forward solve because reverse-mode over a 23,355-step tape is
   memory-bound. Shrink the tape and this collapses.
3. **Solver.** Both fixed-step RK4 dt=1, but camdl integrates via its
   resolved-expression evaluator and Stan via compiled RK4; trajectories are not
   bit-identical (dynamics verified at the MAP, not golden parity). No
   adaptive-tolerance solver was used (rejected per F17).
4. **Convergence.** No full Stan fit ran, so no Stan R̂/ESS/divergence numbers;
   the extrapolation is order-of-magnitude.
5. **The real lever is the burn-in, for both.** A fair Stan version would need
   the same fix camdl is building: solve the **endemic / periodic steady state**
   as the IC (a stroboscopic fixed point) instead of integrating 60 yr through
   every gradient, or a coarse burn-in dt on the unscored warm-up (gh#396). With
   the burn-in tape cut ~14× the gradient drops toward ~0.1 s and the full-fit
   extrapolation falls into the ~0.5–4 hr range — competitive, but still an
   unproven equilibrium-IC engineering exercise, and still fighting the g–r1
   ridge.

## 7. Bottom line

A faithful Stan re-implementation of `ctl_bb` **works** (dynamics validated at
camdl's MAP), but **NUTS through the 60.9-year burn-in is not viable as-is**:
0.055 s forward solve, ~1.2 s per gradient, full fit extrapolating to **~7 hr to
a few days** — vs camdl's mh-ode **~28 min** (which itself doesn't converge the
g–r1 ridge). Stan buys nothing here until the burn-in is killed by an
equilibrium / periodic steady-state IC (or a coarse warm-up dt) — exactly the
lever camdl's gh#396 is already pulling. Stan's per-solve engine is actually
fast; the wall-clock loss is entirely NUTS × burn-in, reproducing garki's F21
conclusion from an independent toolchain.

## Next

- **Chase the §5 solver gap** — measure camdl's own `ctl_bb` forward solve
  directly, profile where the per-step cost goes (resolved-expression eval vs
  native), and scope whether a ~10× is reachable for camdl's ODE fits.
- (Optional) build the equilibrium-IC Stan variant to test the ~0.5–4 hr
  hypothesis from the Stan side — a cross-check on gh#396's payoff.

The Stan model + drivers (`ctl_bb.stan`, `ctl_bb_functions.stan`,
`forward_only.stan`, `prep_data.py`, `run_forward.py`, `run_fit.py`) were built
in a scratch working dir; the data reference is external
(`camdl-garki/data/processed/mv_village_C.tsv`), so this note is the durable
record rather than a committed reproduction.
