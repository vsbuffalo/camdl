# Corner-case fixtures

Small models that exercise the timeline edge cases the all-on-grid golden corpus
does **not** — where snap-vs-exact, short substeps, coincident boundaries, and
fractional endpoints diverge. They are the stress tests that make the
unified-timeline refactor's "byte-identical" parity claim non-vacuous: without
an off-grid model in the corpus, a Stage-1 parity gate passes _vacuously_ (the
round-2 review's #1 finding). Each is a closed SIR base (`beta`, `gamma`) plus
one edge case; the inference references that pin them belong in
`gate_inference_baseline.rs` / `gate_trajectory_baseline.rs` (follow-up).

Interventions are **off by default** — pass `--enable <name>`. Runs below are
`chain_binomial`, `dt=1`, `seed=1`, `beta=1.0`, `gamma=0.2` unless noted.

## `off_grid_intervention.camdl` — cross-backend fire-time divergence (#9)

A pulse cull (50% of S → R) is scheduled at **t = 2.5**, off the `dt=1` grid.

| t | S (chain_binomial) | S (tau_leap) |
| - | ------------------ | ------------ |
| 2 | 971                | 971          |
| 3 | 475                | 477          |

`chain_binomial` rounds the fire time to a step: `round(2.5/1) = 3`, so it culls
at **t = 3** — a half-day late. `tau_leap` truncates a step to land on **t =
2.5** exactly. The integer-grid output mostly hides it (both have culled by
t=3), but the fire time genuinely differs, and the sub-step dynamics (the
infection between 2.5 and 3 acts on the culled S under tau_leap, the un-culled S
under chain_binomial) differ — exactly what the snap-vs-exact policy must make
explicit. The snap is `dt`-dependent: at `dt=0.5`, 2.5 is on-grid and both fire
exactly. Run: `--enable pulse`.

## `coincident_obs_intervention.camdl` — coincident-boundary lifecycle order

An observation of `I` and a cull (50% of I → R) both fall at **t = 10**.

```
t9  I = 507
t10 I = 268      <- cull fired, I ~halved, BEFORE the obs read it
t11 I = 266
```

The observation at t=10 reads the **post-intervention** I (268, ≈ half of the
pre-cull value) — confirming the substep order `… → intervention → observe`.
This is the coincident-boundary ordering the refactor must reproduce bit-for-bit
(impl-review gap #1: the order over coincident `Boundary` kinds). Run:
`--enable cull10`.

## `fractional_output_end.camdl` — fractional endpoint snap

`simulate { to = 80.5 }` is off the `dt=1` grid. The output ends at **t = 80** —
the 0.5-day tail is dropped/snapped (the `seir_vaccine_seasonal`
`output.end = 1095.7275` case in miniature). Pins the tail-snap behavior.

## `off_grid_obs.camdl` — off-grid observation cadence

`every = 2.5 'days` places observations at **0, 2.5, 5, 7.5, 10, 12.5, 15,
17.5** — alternating off and on the `dt=1` grid (the forward sim emits them at
the exact times):

```
t=2.5  19      t=7.5  239     t=12.5 118
t=5.0  122     t=10.0 433     t=15.0 43
```

Under inference the alignment forks: the bootstrap PF steps exactly to each obs;
PGAS snaps them to the grid (and two sub-`dt` obs could collide on one step).
Pins the obs-alignment behavior. Run: `--obs <file>`.

## `all_lifecycle.camdl` — all five substep stages at one boundary

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

The **total stays exactly 1000** at every step — the balance constraint absorbs
the importation-driven growth (R falls as I rises). S halves at t=10 (the cull,
after the fused transition+event ADVANCE). Pins the full substep order the
unified-timeline refactor must reproduce bit-for-bit. Run: `--enable cull10`.

## Reproduce

```bash
camdl simulate tests/fixtures/corner_cases/off_grid_intervention.camdl \
    --params p.toml --backend chain_binomial --seed 1 --enable pulse --stdout
# p.toml: beta=1.0, gamma=0.2 (+ cull=0.5 for the intervention fixtures, N0=1000
# for all_lifecycle)
```

## Wiring into the gates

- **Forward (enforced now):** `gate_corner_case_baseline.rs` loads each model's
  committed IR (`ir/*.ir.json`, params baked via `--set`; regenerate with
  `make
  update-corner-golden`) and pins the FNV trajectory hash under every
  capability-supported backend at seed=42 — 17 baselines. The snap-vs-exact
  divergence is now a regression surface: `off_grid_intervention` hashes to
  _different_ values under `chain_binomial` (snap to step 3) and `tau_leap`
  (exact at 2.5); its `gillespie` baseline pins the `iv_resolution_dt`
  phantom-grid behaviour; `all_lifecycle` pins the fused transition+event
  ADVANCE → intervention → balance → observe order.
- **Inference (enforced now):** `gate_inference_baseline.rs` pins the off-grid
  PF loglik via the `sir_incidence_offgrid` reference (obs at 7.3, 14.6, … — a
  _distinct_ loglik, −59.79, vs the on-grid −59.45). The off-grid likelihood
  path the on-grid corpus cannot reach is a committed baseline.
- **Still to wire:** an off-grid-_intervention_ inference reference (PF/PGAS
  over a model whose intervention fires off the obs grid), and the
  PGAS/IF2/PMMH + RNG-draw-count harnesses (Stage-0 oracle remainder).
