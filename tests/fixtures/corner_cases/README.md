# Corner-case fixtures

Small models that exercise the timeline edge cases the all-on-grid golden corpus
does **not**: where the chain-binomial grid and the exact clock of the ODE and
Gillespie backends disagree, and where short substeps, coincident boundaries and
fractional endpoints occur. Without an off-grid model in the corpus, a check
that two timelines agree byte for byte passes trivially.

This page describes five of the fixtures, each a closed SIR model (`beta`,
`gamma`) plus one edge case. The others (`dt_rate`, `event_drain_fusion`,
`event_intervention_agree`, `gh70_absorbing_importation`, `inflow_drain_origin`,
`multi_effect_same_time`, `offset_origin`, `origin_effects`,
`recurring_default_from`, `seasonal_drift`) describe themselves in their file
headers.

Interventions are **off by default** in `camdl simulate`; pass
`--enable <name>`. Runs below are `chain_binomial`, `dt=1`, `seed=1`,
`beta=1.0`, `gamma=0.2` unless noted.

## `off_grid_intervention.camdl`: cross-backend fire-time divergence

A pulse cull (50% of S → R) is scheduled at **t = 2.5**, off the `dt=1` grid.

| t | S (chain_binomial) | S (gillespie) |
| - | ------------------ | ------------- |
| 2 | 971                | 958           |
| 3 | 475                | 463           |

`chain_binomial` fires an effect at the end of the step nearest its declared
time, ties going to the later step (`effect_plan::fire_step`): 2.5 is a tie
between 2 and 3, so it culls at **t = 3**, half a day late. `gillespie` and
`ode` fire it at **t = 2.5** exactly. The integer-grid output mostly hides it
(both have culled by t=3), but the infection between 2.5 and 3 acts on the
culled S under `gillespie` and on the un-culled S under `chain_binomial`. The
snap depends on `dt`: at `dt=0.5`, 2.5 is on the grid and every backend fires
there. Run: `--enable pulse`.

## `coincident_obs_intervention.camdl`: coincident-boundary lifecycle order

An observation of `I` and a cull (50% of I → R) both fall at **t = 10**.

```
t9  I = 507
t10 I = 268      <- cull fired, I ~halved, BEFORE the obs read it
t11 I = 266
```

The observation at t=10 reads the **post-intervention** I (268, about half of
the pre-cull value), which confirms the substep order
`… → intervention → observe` over coincident boundaries. Run: `--enable cull10`.

## `fractional_output_end.camdl`: fractional endpoint

`simulate { to = 80.5 }` is off the `dt=1` grid. The output's last row is **t =
80**: the 0.5-day tail is not recorded (the `seir_vaccine_seasonal`
`output.end = 1095.7275` case in miniature).

## `off_grid_obs.camdl`: off-grid observation cadence

`emit_schedule = every 2.5 'days` places observations at **2.5, 5, 7.5, 10,
12.5, 15, 17.5, 20**, alternating off and on the `dt=1` grid. A simulation emits
an observation only at a recorded output time, and `chain_binomial` records only
grid times, so at `dt=1` `--obs` is refused at t = 2.5. To emit them, run
`--dt 0.5 --output-every 0.5 --obs <file>` (or use `gillespie` or `ode` with
`--output-every 0.5`).

Under inference the methods differ. The bootstrap particle filter (pfilter, IF2,
PMMH) shortens a step to land on each observation time. PGAS steps only on its
grid and refuses data whose period boundaries lie off it
(`runner::refuse_periods_off_the_grid`); this stream's periods,
`closing_at(time, 2.5 'days)`, start and end at multiples of 2.5, so PGAS
refuses it at `dt=1`.

## `all_lifecycle.camdl`: all five substep stages at one boundary

At **t = 10** the full substep lifecycle fires together: transitions → event
(importation, `add(I, 5)`) → intervention (cull S→R) → balance
(`R = N0 − S − I`) → observe (`projected = I`). An every-5-day importation and
the every-substep balance run throughout. Observed (`--enable cull10`,
`cull=0.5`, `N0=1000`):

```
t    S     I     R     total
9    244   530   226   1000
10   72    534   394   1000    <- cull fired (S ~halved); balance holds total
```

The **total stays exactly 1000** at every step: the balance constraint absorbs
the importation-driven growth (R falls as I rises). S halves at t=10 (the cull,
after the fused transition+event advance).

## Reproduce

```bash
camdl simulate tests/fixtures/corner_cases/off_grid_intervention.camdl \
    --params p.toml --backend chain-binomial --seed 1 --enable pulse --stdout
# p.toml: beta=1.0, gamma=0.2 (+ cull=0.5 for the intervention fixtures, N0=1000
# for all_lifecycle). A parameter the model does not declare is refused, so
# each fixture needs its own file.
```

## Wiring into the gates

- **Forward:** `gate_corner_case_baseline.rs` loads every model's committed IR
  (`ir/*.ir.json`, params baked via `--set`; regenerate with
  `make update-corner-golden`) and pins an FNV hash of the compartment
  trajectory under each capability-supported backend (`gillespie`,
  `chain_binomial`, `ode`) at seed=42. `off_grid_intervention` hashes to
  different values under `chain_binomial` (fires at 3) and `gillespie` / `ode`
  (fire at 2.5); `all_lifecycle` pins the fused transition+event advance →
  intervention → balance → observe order.
- **Inference:** `gate_inference_baseline.rs` pins the particle-filter
  log-likelihood of an SIR model whose observations are off the grid
  (`sir_incidence_offgrid`, observations at 7.3, 14.6, …: −59.79, against −59.45
  for the on-grid `sir_incidence_truth`). It builds that model in Rust and does
  not read these fixtures. Of these fixtures, `seasonal_drift`
  (`gate_pgas_density_baseline.rs`, `pgas_ancestor_weight.rs`),
  `event_intervention_agree` (`gh187_pgas_scheduled_intervention.rs`) and
  `dt_rate` (`gate_dt_rate_exact_clip.rs`) are read by inference tests.
- **Not yet covered:** an inference reference over a model whose intervention
  fires off the observation grid.
