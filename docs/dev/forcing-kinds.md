# Forcing-function kinds

camdl's `forcing {}` block lets a model author declare named time-dependent
quantities — `pop(t)`, `school(t)`, `beta_seas(t)` — that get evaluated at
runtime against `t` (and never against compartment state). They're the
principled escape hatch for "things that change over time but aren't part of the
dynamics."

## The 2×2

The kinds shipped fall into a natural 2×2:

|           | Estimable parametric                       | Fixed / data-driven           |
| --------- | ------------------------------------------ | ----------------------------- |
| Periodic  | `sinusoidal`, `fourier`, `periodic_spline` | `periodic` (step over period) |
| Aperiodic | _(none currently — no clear need)_         | `interpolated`, `piecewise`   |

- **Estimable parametric**: the forcing's shape parameters (amplitudes, phases,
  basis coefficients) are themselves `Param` references that IF2/PGAS/NUTS
  estimate from data. The functional form is fixed at compile time; the values
  aren't.
- **Fixed / data-driven**: the forcing's values are pinned at compile time, from
  either an inline literal schedule or an external TSV. The runtime reads them;
  the inference pipeline never touches them.

## Choosing a kind

### Estimable, periodic

| Kind              | Free params | When                                                   |
| ----------------- | ----------- | ------------------------------------------------------ |
| `sinusoidal`      | 4 (a,T,φ,b) | Single-harmonic seasonality with explicit baseline     |
| `fourier`         | 2N          | N-harmonic Fourier (bimodal, multi-modal seasonality)  |
| `periodic_spline` | K           | Flexible periodic shape; King 2008-style 6-coef spline |

Pick by parsimony: prefer the smallest basis that captures the expected shape.
`sinusoidal` for one cycle/year, `fourier` with 2 harmonics for bimodal,
`periodic_spline` when the shape is genuinely non-Fourier (e.g., sharp
boundaries at school terms — though `periodic` with daily values is also a fit
there).

### Fixed, periodic

`periodic`: step-function over period, defined by `values[]` at implicit equal
spacing. Common use: school-term forcing where the schedule is known and not
estimable.

### Fixed, aperiodic

`interpolated`: load `times[]` and `values[]` from a TSV (or inline);
interpolate with `method = linear | spline | constant` (a bare enum). Common
use: demographic covariates (pop(t), birthrate(t)) from census data.

`piecewise`: inline step-function with explicit `breakpoints[]` and `values[]`.
Use when the schedule is short enough to fit in the model file and a covariate
file would be overkill.

## Dimensional units

Every forcing carries a declared dim from its tier-3 unit literal, matching the
rest of the IR's dim machinery:

```camdl
forcing {
  pop : interpolated 'count { ... }
  beta_seas : fourier 'ratio { ... }   # multiplier on a baseline rate
}
```

The dim-checker uses the declared dim authoritatively; expander applies any
scale factor at compile time so runtime evaluation returns values in the model's
`time_unit`.

## Common pitfalls

- **Rate non-negativity invariant**: A forcing used as a _direct_ rate
  multiplier should be non-negative under all parameter settings.
  `sinusoidal { baseline = β₀; amplitude = a }` with `a > β₀` will go negative
  when sin oscillates to −1. The runtime clamps `rate < 0 → propensity = 0`, but
  the modeller is responsible for keeping the analytic form sane.
- **Periodic vs `Interpolated{method=step}`**: similar at first glance, but
  `periodic` wraps at `t mod period` (annual schedules that repeat decades),
  while `interpolated` is one-shot in absolute time. Pick by whether the
  schedule recurs.
- **`fourier` cos/sin convention**: harmonics are paired as `(a_k, b_k)` for cos
  and sin respectively; harmonic index `k` starts at 1 (so the first pair is the
  fundamental cycle, not the constant term — that's the baseline modulated by
  `1 + Σ`).

## Adding a new kind

If a model needs a forcing the current kinds can't express:

1. Decide which cell of the 2×2 it belongs in.
2. Add the variant to `time_func_kind` (both languages atomically).
3. Update serde, dimcheck, expander parser, runtime evaluator.
4. Add an entry to this doc with the "when to use" rule.
5. Bump the IR schema version (`ir/VERSION`) — additions are minor; renames or
   removals are major.

For one-off shapes, prefer `interpolated` with a precomputed value table over a
new variant. The 2×2 is meant to stay small.
