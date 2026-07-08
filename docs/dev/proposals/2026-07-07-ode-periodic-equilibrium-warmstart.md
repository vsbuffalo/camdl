# ODE periodic-equilibrium warm-start — skip the spin-up transient in `nuts`/`mh` on `ode`

Status: spec — implement against this. Resolves: gh#396. Builds on:
`2026-06-26-ode-nuts-gradient-spine.md` (the augmented forward-sensitivity
integrator this reuses). Sibling: the general forward checkpoint-fork (gh#213)
reuses **Phase A** below but is otherwise independent. High-risk: inference-math
change on the ODE gradient path — read the full `ode.rs` sensitivity core and
`ode_grad.rs` before editing any part, and gate every change on the oracles in
"Correctness". Rust-only: warm-start is a fit-time runtime option, not a model
construct — no IR/OCaml/golden changes.

## Problem

A forced (seasonal) compartmental model with a long spin-up to periodic
equilibrium pays the **entire transient cost on every likelihood/gradient
evaluation**. The DMT/Garki `mv_age` family is the exemplar: `origin = 1910`
gives a ~63-year burn-in before the data window (196 transition-driven
compartments, ~8 estimated params). Measured on the current build by the garki
testing agent: one augmented-ODE gradient ≈ ~1 min (~9× the 6.5 s plain solve);
a single `nuts` warm-up iteration does not complete in >7 min ⇒ a full fit =
days–weeks, **zero samples**. The cost is the transient re-integrated per eval;
it is **sampler-independent** (gates `nuts` and the cheaper `mh` on `ode`
equally). Measured relaxation time ≈ **56 yr ≫ 5-yr data window** — the data
cannot "forget" a wrong initial state within the window.

## Why the cheap options don't work (measured)

- **Frozen checkpoint** (burn in once, reuse the state — what
  `dmt_garki_mv_age_warmstart.camdl` does with a hand-dumped 1961 init): wrong
  for a fit. The equilibrium `X*(θ)` moves with the estimated parameters; a
  frozen state has `∂X*/∂θ = 0`, biasing the gradient. And because relaxation
  (56 yr) ≫ window (5 yr), a short residual burn-in does **not** wash the error
  out. Correct only for forward sim or fits of non-dynamical params.
- **Coarse global `dt`** (garki dt-sweep, 2026-07-07): **hard-blocked**. Garki
  survey dates have coprime day-offsets, so the score path rejects any `dt > 1`
  (`observation at t=… is not a multiple of dt`). Unmeasurable, unusable.
- **Per-phase `dt`** (coarse burn-in only): the burn-in _does_ tolerate coarse
  `dt` (~1% state drift at weekly steps, ~5–8× faster), but leaves a ~1%
  θ-dependent bias that perturbs the gradient. A legitimate approximate interim,
  not an exact default under this project's discipline.

The exact fix is to **solve for the equilibrium at the current `θ`, every eval,
with its exact sensitivity** — sidestepping both the integration cost and the
obs-alignment wall.

## Approach: solve the stroboscopic fixed point

Let the seasonal forcing have fundamental period `P` (typically 1 yr). The limit
cycle sampled at the warm-start time `T_eq` is the fixed point of the period-`P`
flow map anchored at `T_eq`:

```
X* = Φ_P(X*; θ),   Φ_P(·; θ) = flow of the ODE from T_eq to T_eq+P at θ.
```

Newton uses the **monodromy** `M = ∂Φ_P/∂X₀` (state-transition Jacobian over one
period): `(I − M) ΔX = Φ_P(X) − X`. The **equilibrium sensitivity** follows from
the implicit function theorem on `X* − Φ_P(X*; θ) = 0`:
`∂X*/∂θ = (I − M)⁻¹ ∂Φ_P/∂θ`. `M` is needed for the sensitivity regardless, and
once you have it Newton is free — one matrix serves both. The 60-yr transient
collapses to a handful of one-year period-solves plus dense linear solves.

## Conservation — the load-bearing correction

For a **closed** model `(I − M)` is **singular**, endemic or not: a conserved
linear quantity `cᵀx` (e.g. per-village total population) makes `cᵀM = cᵀ`, so
`c` is a left eigenvector of `M` with eigenvalue 1. **Garki is conserved, k =
4** (per village): `Nvil[v] = Σ_a N[v,a]` is dynamic, and `birth = δ·Nvil`
exactly balances `death = δ·c` summed → `dN/dt ≡ 0`. This conservation is
_dynamical_ (a rate-balance), not stoichiometric (`1ᵀν ≠ 0`), so it cannot be
read off the stoichiometry — it must be detected **numerically**.

**Detection is robust, not fragile.** Genuine conservation is _exact_
(`dN/dt ≡ 0`), so the corresponding singular values of `(I − M)` sit at machine
precision (~1e-12), while the slowest _genuine_ mode of a 56-yr relaxation is at
`1 − |μ| ≈ 1/56 ≈ 0.018` — a 10-order-of-magnitude gap. Any SVD threshold in
`[1e-8, 1e-4]` cleanly separates the `k` conserved directions. (The fragile case
— a multiplier genuinely near 1 — is the _non-endemic_ boundary, which the
endemic gate refuses anyway.)

**Solve on the conservation manifold via a bordered system.** With `C`
(`n_int × k`) the conserved directions (left null space of `(I − M)`, from the
SVD) and `b = CᵀX_init` the totals pinned by the model's initial condition,
Newton solves the nonsingular KKT/bordered system each iteration:

```
[ I − M   C ] [ΔX] = [ Φ_P(X) − X ]
[ Cᵀ      0 ] [λ ]   [ b − CᵀX    ]
```

which enforces `CᵀX = b` exactly and is nonsingular whenever the _transverse_
cycle is hyperbolic. The sensitivity reuses the **same LU factorization**:

```
[ I − M   C ] [∂X*/∂θ] = [ ∂Φ_P/∂θ ]
[ Cᵀ      0 ] [  μ   ]   [ ∂b/∂θ   ]
```

where `∂b/∂θ = Cᵀ · (ic_grad seed)` — so the conserved-direction sensitivity
**does** reuse `ic_grad`, while the transverse part comes from the equilibrium.
`∂Φ_P/∂θ` is the _partial_ (see the seed trap below).

**Endemic gate.** Refuse if the transverse system is ill-conditioned (reciprocal
condition of the bordered matrix below `RCOND_MIN`, i.e. a non-conservation
multiplier → 1). Under NUTS this surfaces as a `-inf` rejection, not an abort
(`det_grad` `Err` → `(-inf, 0)`, `ode_nuts.rs:147-152`), so a non-endemic
proposal mid-run is contained; only a non-endemic _initial_ point aborts (with a
clear message). Report a rejected-for-non-endemic count so the boundary isn't a
silent posterior wall.

## Design in types

### Phase A — ODE start-from-state seam (also the gh#213 ODE arm)

One new argument on `integrate_obs_sensitivity` (`ode.rs:885`, `pub(crate)`);
every existing caller passes `None` and is byte-identical to today.

```rust
pub(crate) fn integrate_obs_sensitivity(
    model: &CompiledModel, params: &[f64], param_model_idx: &[usize],
    x0: Option<&[f64]>,          // NEW: overrides initial_state_continuous (ode.rs:925)
    state_sens_0: &[f64], cfg: &OdeConfig, obs_times: &[f64],
    seed: SensSeed,              // NEW: see B1
) -> Result<Vec<ObsSensitivity>, SimError>;
```

The forward value path (`OdeSim::run` / `run_ode`) gets the analogous
`x0: Option<&[f64]>` injection for the `mh` wiring. Oracle: the splice
invariant, bit-exact because fixed-step RK4 is deterministic (no RNG) —
`solve(t0→t_end) ≡ solve(t0→T*) ++ solve_from(x(T*), S(T*), T*→t_end)`.

### Phase B — the equilibrium solver

**B1. Sensitivity seed mode.** The `(param_model_idx, state_sens_0)` pair is
replaced by an explicit seed, decoupling the sensitivity column count from the
parameter set and making the `∂rate/∂θ` forcing switchable. It is a real branch,
not a `param_model_idx` trick (a compartment index colliding with a param's
model index would inject a spurious `∂rate/∂θ`):

```rust
enum SensSeed<'a> {
    Params { model_idx: &'a [usize], s0: &'a [f64] },  // d = params, forcing ON  (today)
    Zero   { model_idx: &'a [usize] },                 // d = params, S(0)=0, forcing ON (∂Φ/∂θ partial)
    Monodromy,                                         // d = n_int, S(0)=I, forcing OFF (M)
}
```

Threaded through `sensitivity_derivs` (`ode.rs:160`) and `rk4_step`; the
Monodromy path skips the wasted flow-sensitivity block. Everything sizes off the
seed's column count, so an identity `n_int`-column seed allocates uniformly.

**B2. The solver** (`solve_equilibrium`, new module `ode_equilibrium.rs`):

1. `integrate_period(X_k)` in `SensSeed::Monodromy` over `[T_eq, T_eq+P]` →
   `(Φ, M)`. (Value clamps `max(0.0)`; assert the clamp is inactive on the cycle
   — a clamped iterate corrupts `M`.)
2. On the first `M`: SVD `(I − M)` → `C` = left-null directions (σ <
   `CONS_EPS`).
3. Bordered Newton to `‖Φ_P(X) − X‖∞ < TOL_EQ`; cap iterations; non-convergence
   → endemic-gate error.
4. Sensitivity: one `SensSeed::Zero` period solve → `∂Φ_P/∂θ`; reuse the
   bordered LU with RHS `(∂Φ_P/∂θ, Cᵀ·ic_grad)` → `∂X*/∂θ`.

Returns
`Equilibrium { x_star: Vec<f64> /* n_int */, x_star_sens: Vec<f64> /* n_int×d */ }`.

**B3. Wiring.** `det_grad` (nuts, `ode_grad.rs:62-70`): when warm-start is on,
`solve_equilibrium(…, T_eq, P, dt)` → set `cfg.t_start = T_eq`, inject
`x0 = X*`, `state_sens_0 = ∂X*/∂θ` (via `SensSeed::Params`).
`compute_ode_loglik` (mh/MLE, `ode_loglik.rs:36`): solve for `X*` only (state
path), inject `x0 = X*` at `T_eq` into `OdeSim::run`.

### Validity gate (pre-obs periodicity)

Runs at fit setup once `T_eq` is resolved. Refuse (hard error, naming the
feature) if the warm-start assumptions don't hold:

- any `event`/`intervention` fires in `[origin, T_eq]` (non-periodic effect in
  the transient window);
- the forcing is not `P`-periodic over the burn-in neighborhood — a
  **numerical** check `‖C(t) − C(t+P)‖ < FORCING_PERIOD_EPS` sampled across
  `[T_eq − P, T_eq]` (handles `interpolated`/table forcings, which carry no
  declared period — the garki case);
- `P` is not supplied, is non-positive, or the model has no forcing.

Refusal points at the alternatives (checkpoint for fixed-transient-param fits;
pay the transient otherwise).

## Surface (CLI + TOML, via the run input system)

Warm-start is a fit-time option, expressed identically as a CLI flag and a
`fit.toml` field, and folded into the fit config-identity hash (it changes the
scored trajectory → it re-keys the run).

```
camdl fit model.camdl --data d.tsv --method nuts --backend ode \
      --warm-start equilibrium --warm-start-period 365 [--warm-start-at DATE]
```

```toml
[fit]
warm_start = "equilibrium" # default "none"
warm_start_period = 365 # days (model time_unit); required for equilibrium
warm_start_at = "1970-01-01" # optional; default = earliest first_obs across streams
```

`warm_start_at` (`T_eq`) is a **single** anchor `≤ min(first_obs)` — decoupled
from the per-stream `cond_from` (which stays per-stream and runs `[T_eq, t_end]`
untouched). Enum `WarmStart = None | Equilibrium`; a future `Checkpoint(path)`
variant is where gh#213 attaches. A `--dump-gradient` eval flag (fixed-θ, emits
the autodiff gradient) supports external gradient verification (gh#78).

## Correctness / tests (red-first where a bug would hide)

1. **Phase-A splice invariant** (bit-exact): inject a continuous solve's exact
   `x(T*)`, `S(T*)`; assert the resumed tail equals the continuous tail
   bit-for-bit, state and sensitivity.
2. **Monodromy vs finite difference**: `M` (Monodromy-mode) matches a central FD
   of the one-period map `∂Φ_P/∂X₀` column by column.
3. **Equilibrium == long burn-in** (headline): `X*` from the solver matches
   `X(T_eq)` from integrating the full transient at the same `θ` (grids
   matched).
4. **Sensitivity three ways**: `∂X*/∂θ = (I−M)⁻¹∂Φ_P/∂θ` matches (a) a central
   FD of the equilibrium re-solve and (b) the augmented `S(T_eq)` from the long
   burn-in. Negative control: a param not entering the transient → zero column.
   (Only the burn-in check distinguishes the partial `∂Φ_P/∂θ` seed from a total
   one — the seed trap.)
5. **Conservation**: on a closed seasonal model (`k ≥ 1`), the bordered solve
   converges, preserves `CᵀX = b` to machine precision, and the endemic gate
   does **not** fire on a genuinely endemic model. The CI reference model is
   closed-with-waning so this path is exercised.
6. **Endemic gate fires** on a parameterization driven non-hyperbolic
   (multiplier → 1), with the endemicity message, not a NaN.
7. **End-to-end**: a `nuts`-on-`ode` fit with `--warm-start equilibrium`
   recovers the same posterior as a short-burn-in reference, in a fraction of
   wall-clock.

## Scope (v1)

- Transition-driven (integer) compartments only (real-compartment forward
  sensitivity is refused at `gradient_capability.rs:298`; garki is in scope).
- rk4 fixed-step; no `dt`-in-rate (existing `ode_grad` guards).
- Single constant fundamental `P`; multi-period/LCM and `θ`-dependent `P`
  refused, are follow-ups.
- Dense bordered solve (`n_int + k`), fine to a few hundred compartments (garki
  196); national-scale block-sparse is a follow-up.

## Resolved decisions

- **`nalgebra`** (pure-Rust, no system BLAS) for the SVD (conservation
  detection) and the bordered LU + reciprocal-condition estimate. A hand-rolled
  solve is rejected: the matrix is deliberately near-singular for slow cycles.
- **`T_eq`** is a single declared anchor, not the per-stream `cond_from`.
- **`P`** is user-declared and numerically verified (interpolated forcings carry
  no period).
- Flow/incidence accumulators reset at `T_eq`; the conserved-direction
  sensitivity reuses `ic_grad`.

## Follow-ups (named, not blocking)

Real-compartment forward sensitivity → warm-start for explicit `dx/dt` models;
multi-period/LCM `P`; block-sparse bordered solve for national scale; Newton
warm-start across evals (seed `X₀` from the previous gradient's `X*`; deferred —
couples evals, land the stateless version + oracles first).
