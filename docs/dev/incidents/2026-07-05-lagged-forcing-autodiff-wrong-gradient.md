# Lagged forcing: autodiff emits the gradient at `t`, not `t − lag`

Date: 2026-07-05

Status: **fixed** — `autodiff.ml` threads `lag` into the sinusoidal/Fourier
closed forms; a finite-difference gradient gate
(`test_gradient_vs_finite_differences_lagged_forcing`) pins it. Red→green
verified below; full `make test` green; all existing goldens byte-identical.

Class: **code-vs-code** — the OCaml autodiff closed forms disagree with the Rust
runtime's forward evaluation. Fix the OCaml side; pin the agreement with a
finite-difference gradient gate.

## What happened

For a forcing declared with a `lag` (gh#314), the symbolic autodiff of a
transition rate that differentiates _through_ the forcing — i.e. with respect to
a forcing **coefficient** (a sinusoidal `amplitude`/`phase`/`baseline`, or a
Fourier harmonic coefficient) — emits a gradient expression built over **bare
`Time`**. The forward value of that same forcing is evaluated at `t − lag`. The
runtime evaluates the emitted bare-`Time` gradient at `t`. So the value uses
`t − lag` and its gradient uses `t`; they disagree by the lag offset whenever
`lag ≠ 0`.

The result is a **silently wrong `∂rate/∂θ`** consumed by PGAS+NUTS (the default
Bayesian method) for any fit that estimates a lagged sinusoidal/Fourier forcing
coefficient. Forward simulation and the gradient-free methods (IF2, bootstrap
PF, PMMH) never touch `rate_grad` and are unaffected.

## Detection

Compiler review, 2026-07-05 (a reviewer reading `autodiff.ml` against the Rust
evaluators). Not caught by any existing gate.

## Reproduction

Model (`lag_repro.camdl`): a sinusoidal forcing with an estimated `amplitude`
and a 90-day `lag`, used in a rate.

```
parameters { beta : rate  alpha : probability  phi : real }
forcing {
  seasonal : sinusoidal 'ratio {
    amplitude = alpha
    period    = 365.0
    phase     = phi
    baseline  = 1.0
    lag       = 90 'days
  }
}
transitions { infection : S --> I @ beta * seasonal(t) * S }
```

Compile and read the emitted gradient:

```
$ camdlc check lag_repro.camdl        # ✓ no errors, 0 warnings  (exit 0)
$ camdlc lag_repro.camdl              # inspect transition "infection"
```

The forward rate uses a `time_func` node (runtime applies `t − lag`); the
emitted `rate_grad["alpha"]` uses a **bare `Time`** node:

```
rate_grad["alpha"] = beta * sin( 2π·(Time − phi) / 365 ) * S
                                    ^^^^  {"time": null}  — evaluated at t, NOT t − 90
```

`grep` on the gradient JSON: `"time_func"` absent, bare `"time"` present, the
lag value `90` absent.

Verified runtime semantics (why the bare `Time` is evaluated at `t`):

- `rust/crates/sim/src/resolved_expr.rs:525` — `ResolvedExpr::Time => ctx.t` (no
  lag shift).
- `rust/crates/sim/src/resolved_expr.rs:603–614` — `t_eff = ctx.t − lag` is
  applied **only** inside the `TimeFunc` arm.

Numeric error (correct `∂R/∂alpha = sin(2π(t−lag−phi)/period)` vs the emitted
`sin(2π(t−phi)/period)`, with `period=365, lag=90, phi=0`):

| t     | correct ∂R/∂α | emitted (bug) | note             |
| ----- | ------------: | ------------: | ---------------- |
| 0     |       −0.9998 |        0.0000 |                  |
| 45    |       −0.6995 |        0.6995 | **sign flipped** |
| 90    |        0.0000 |        0.9998 | invented from 0  |
| 182.5 |        0.9998 |       −0.0000 |                  |
| 227.5 |        0.6995 |       −0.6995 | **sign flipped** |
| 272.5 |       −0.0000 |       −0.9998 |                  |

Evidence tier: mechanism confirmed on both sides from source; structural
reproduction from a real compile (the emitted bare-`Time` gradient); numeric
error computed from the verified runtime eval semantics. The runtime
finite-difference gradient check is the **red test** that lands with the fix
(below) and is the numeric-runtime reproduction.

## Root cause

`ocaml/lib/ir/autodiff.ml`:

- `sinusoidal_closed` (`:120–129`) and `fourier_closed` (`:135–149`) build the
  forcing's closed form over **bare `Time`**, not `Time − lag`.
- The `TimeFunc` differentiation arm (`:206–219`) guards with `lag_mentions`
  (`:170–172, :208`), which fires only when the differentiation **parameter is
  inside `tf.lag`** (→ `Omitted`). When the parameter drives a **coefficient**
  and the forcing merely _has_ a lag, `lag_mentions` is false and the arm falls
  through to `d (sinusoidal_closed s)` / `d (fourier_closed f)` — the
  bare-`Time` closed form.

The runtime carries the lag correctly for the forward value (via the `TimeFunc`
arm) but has no way to re-apply it to a bare `Time` node the autodiff emitted.

## Blast radius

- **Hit:** PGAS+NUTS (and any gradient-consuming path) fitting a **sinusoidal or
  Fourier** forcing with a **`lag`** and an **estimated coefficient**
  (amplitude/phase/baseline/harmonic). The gradient is wrong, sign-flipping over
  the period, so the sampler is actively misled.
- **Not hit:** forward simulation; IF2; bootstrap PF; PMMH (all gradient-free);
  any forcing without a `lag`; `Periodic` forcings (already `Omitted` when a
  coefficient is estimated); a parameter that _is_ the lag (already `Omitted`).

## Fix

Thread `tf.lag` into `sinusoidal_closed` / `fourier_closed`, substituting
`Time → BinOp(Sub, Time, lag)` when `tf.lag = Some lag`. This is correct because
the `lag_mentions` guard runs **first**: in the closed-form branch the parameter
is not inside the lag, so `d(Time − lag)/dparam = 0 − 0`, and the chain rule is
unchanged apart from evaluating the sinusoid at `t − lag`. When `tf.lag = None`,
no substitution — byte-identical to today.

- **Red test (verified):** `test_gradient_vs_finite_differences_lagged_forcing`
  over `tests/fixtures/gradient/seir_seasonal_lagged.camdl` (`lag = 60 'days`,
  estimated `alpha`/`phi_season`). On the buggy compiler `∂ll/∂alpha = −1022.99`
  (analytic) vs `+344.01` (finite difference) — **wrong sign**, rel_err 3.97;
  after the fix, `344.0111` vs `344.0110`, rel_err 4.3e-7. The no-lag
  seasonal/SIR/spatial FD tests stay green.
- **Golden impact:** no existing golden uses a forcing `lag`, so the fix is
  golden-neutral — `make update-golden` moved zero existing files; the new
  fixture golden is the only addition.

## What it suggests

gh#314 (`lag`) shipped without a fixture that exercises the **gradient** path of
a lagged forcing — the forward path was tested, the differentiated path was not.
The durable fix is a gradient gate that walks the forcing-kind ×
estimated-coefficient matrix, so a new forcing feature cannot land with a
value-only test again. The autodiff closed forms should be checked against the
runtime forward evaluation (FD), not only against each other.
