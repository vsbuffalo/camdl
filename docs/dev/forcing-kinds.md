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
| Aperiodic | `bspline`                                  | `interpolated`, `piecewise`   |

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

### Estimable, aperiodic

`bspline`: a clamped B-spline over a window — uniform knots from
`window = [a, b]` and `n_basis`, or explicit `breakpoints` — optionally
`link = log` for a positive rate. A run outside the window is an error unless
overridden ([below](#outside-the-range-a-forcing-is-defined-on)). Use for a
transmission rate that moves smoothly through an outbreak on no known date.
Window ends may be observation anchors (`last_obs - 2 'weeks`). Coefficients may
be parameters and carry an emitted gradient (a `forcing_basis` node), so NUTS
can fit them; the knots, window and period are structural (E600). Or
(implemented, refused by every `camdl` command until released)
`coefs = smooth(level, slope, scale, modes, per)`: a line plus whitened
curvature modes under a penalized-complexity prior on the scale, the penalty Ω
and its whitening built by the runtime (`sim/src/spline_penalty.rs`); its
gradients use the `whitened` and `linear` basis spaces. Proposal
`2026-10-27-estimable-bsplines.md`.

### Fixed, periodic

`periodic`: step-function over period, defined by `values[]` at implicit equal
spacing. Common use: school-term forcing where the schedule is known and not
estimable.

### Fixed, aperiodic

`interpolated`: load `times[]` and `values[]` from a TSV (or inline);
interpolate with `method = linear | spline | constant` (a bare enum). Common
use: demographic covariates (pop(t), birthrate(t)) from census data. The data
must cover the run ([below](#outside-the-range-a-forcing-is-defined-on)).

`piecewise`: inline step-function with explicit `breakpoints[]` and `values[]`.
Use when the schedule is short enough to fit in the model file and a covariate
file would be overkill.

## Outside the range a forcing is defined on

Two kinds are defined only on a range: an `interpolated` forcing, read linearly,
as a cubic spline or piecewise-constant, from its first data point to its last,
and a `bspline` over its window. The others — `sinusoidal`, `fourier`,
`periodic`, `periodic_spline` and `piecewise`, whose first and last values
extend before its first and after its last breakpoint — are defined for all
time.

A run that evaluates a ranged forcing outside its range is an error. The error
names the forcing, its range and the first time outside it:

```
error: forcing `mult` was evaluated at t = 51, after its last data point (it is defined on t = 0 to t = 50, its data).
```

The run gives three ways out: extend the data (or the window) to cover the run,
end the run within the range, or pass `--allow-forcing-extrapolation` (in a fit
config, `[config] allow_forcing_extrapolation = true`). Under the override the
forcing holds its end value there, exactly as it did before the check existed,
the run prints one warning per forcing naming its range and the span held, and
each `run.json` the command writes records the override and the forcings held
(`inputs.forcing_extrapolation`). In a fit config the setting is part of the
fit's identity, so a fit stored under it is not reused by a run without it.

The check reads the times the run actually evaluated, with the forcing's `lag`
applied, not the run's span. A forcing read only under a condition that is false
past its range is never evaluated there and raises nothing. The engine records
the first time outside the range in a set-once slot per forcing and side
(`sim::forcing_range`), so a particle filter's threads do not contend on it, and
the command decides whether that is an error or a warning
(`cli::forcing_extrapolation`). The design is in
`docs/dev/temporal-architecture.md` §4.6.

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
