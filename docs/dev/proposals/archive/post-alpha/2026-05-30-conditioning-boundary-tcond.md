# The conditioning boundary `t_cond`: decoupling burn-in from the start of filtering

Status: **SUPERSEDED (archived 2026-06-09)** by the time-interval model —
`2026-06-09-time-interval-model.md` (which carries this inference math forward
as the conditioning window's mechanism). Start from
`2026-06-09-time-and-observation-overview.md`. Retained for history only — do
not implement from this. (Inbound references — the W329 message,
`warning-catalog.md` — still point here until the implementation repoints them.)

Date: 2026-05-30

**This is inference math** (`particle_filter.rs` / `pgas.rs` / `if2.rs`
filter-start logic + an IR/config field). Per CLAUDE.md it is high-risk
regardless of how mechanical it looks; treat accordingly when unpinned.

---

## 1. The problem it solves

The particle filter conditions on observation _k_ over `[t_{k-1}, t_k]`. The
**first** window is `[t_start, first_obs_time]`, with `t_start` from the model
(`simulate.from` → `Ir.t_start`, `expander.ml:3082`) and `first_obs_time` from
the data. When `t_start` sits far behind the first observation — e.g. a measles
model with `origin = 2011` and case data from 2014 — the first window spans the
whole gap. Reproduced (gh#134): same data, fixed parameters, only
`simulate.from` moved:

| `simulate.from`                              | first window       | loglik   |
| -------------------------------------------- | ------------------ | -------- |
| `date("2011-12-26")` (origin)                | `[0, 980]` = 980 d | −3416.31 |
| `date("2014-08-25")` (1 wk before first obs) | `[973, 980]` = 7 d | −3202.18 |

The legitimate use case is a **covariate-informed burn-in**: over 2011–2014
births accumulate susceptibles and MCV/SIA deplete them, so the demographic
covariates are informative about `S(2014)`. The user _wanted_ this but the bug
forced them to abandon it and estimate a free `initial_susceptible_fraction`
instead — a real loss of rigor.

## 2. The design

Decouple where _dynamics begin_ from where _conditioning begins_:

- `simulate.from` (→ `t_start`) — where dynamics begin (e.g. 2011).
- `t_cond` — where the particle filter starts weighting / resampling /
  accumulating. **Default: `first_obs`** (so models with data from `t_start` are
  unchanged); set explicitly to enable a burn-up window.
- `[simulate.from, t_cond)` is a **warm-up**; `[t_cond, …]` is the conditioned
  fit.

**Comparison to He et al.** (book vignette `he2010_london.camdl`): that model
uses `from = 0 'days`, is _unanchored_ (no `origin`), and its cases +
`pop(t)`/`birthrate(t)` covariates all start at `t = 0`. So
`t_start == first_obs == 0`: no gap, no bug, no separate burn-in window — the
entire 15-year fit window is the data window and the filter conditions
throughout; early transients are conditioned away by data that exists there. He
can do that because he _has_ data spanning the whole period. Kano cannot (no
case data 2011–2014), which is exactly why `t_cond` is needed: you cannot
condition on data you do not have, but you _can_ let the model relax over a
covariate-only span.

## 3. The crux: what is the warm-up? (open)

The original draft proposed a **deterministic ODE skeleton** warm-up. The
adversarial review found this **incoherent** for the chain_binomial,
integer-compartment models this targets:

- there is no ODE skeleton beside the stochastic process
  (`ChainBinomialProcess::initial_state` returns integer `ParticleState.counts`;
  no real-valued state in the PF path);
- a single deterministic trajectory seeds every particle with the same state →
  **zero process variance**, an overconfident cloud at `t_cond`, near-degenerate
  ESS at the first weight — a silent statistical error;
- running an ODE mean would require a _different backend_ (`OdeSim`,
  `REAL_COMPARTMENTS`) the PF process does not own, then a real→integer
  round-trip to seed particles.

**Current best design — an unweighted stochastic ensemble.** Run the _same_
chain_binomial dynamics forward from `simulate.from` to `t_cond` as a particle
ensemble **with weighting and resampling disabled** (the existing propagate step
at `particle_filter.rs:200–227`, minus the weight/resample steps). This
accumulates the _correct_ process variance. For a seasonal-endemic model the
ensemble relaxes to the model's quasi-stationary distribution under the
covariate forcing — exactly the right Bayesian prior
`p(x_{t_cond} | θ, covariates)` when no case data exists yet. At `t_cond` the
filter begins weighting/resampling with a **freshly reset** flow accumulator, so
the first scored window is one cadence.

Key realization: free-running an ensemble is **not** itself the bug. gh#134 was
a bug only because the giant accumulated window got _scored_ against a single
datum. An unscored warm-up has no such problem — the ensemble spread at `t_cond`
is genuine prior uncertainty. Because it reuses the existing propagation path
and adds no new dynamics implementation, it is lower risk than the ODE framing —
but it still changes filter-start logic.

**Open correctness questions (must resolve before code):**

1. **Decoherence on non-stationary spans.** A long warm-up over a span with
   directional covariate drift and no restoring dynamics can let the ensemble
   spread without bound. The seasonal-endemic attractor bounds it; the general
   case does not. Is a cap / diagnostic needed?
2. **Watchdog interaction.** The warm-up is one long propagation with no
   resample → collides with the gh#133 degeneracy watchdog: the wall-clock
   budget (`degeneracy.rs:67`) scales with particle count but not interval
   length (false `PFWallclockTimeout`), and folded steps that push no ESS entry
   desync the `ESS_COLLAPSE_WINDOWS = 3` window count from wall-clock time. The
   warm-up must pause the wall-clock timer and decide ESS-entry semantics.
3. **PGAS/IF2 parity.** `t_cond` must thread through `SMCConfig`, `IF2Config`,
   and the PGAS config plus each algorithm's first-window logic — not a single
   field.

## 4. Guardrail (decided: error + opt-out)

After the `t_cond` default, if the first conditioning window
(`first_obs − t_cond`) is still ≫ the modal observation spacing (or there is an
all-missing leading stretch), **hard error** naming the gap, the modal cadence,
and the opt-out (an explicit flag/key to allow a genuine leading gap). Decided
2026-05-30: error + opt-out, per the stakes doc ("silent wrong answers are
critical"). The cheap immediate safety net independent of the full `t_cond`
machinery — could ship _before_ `t_cond` as a standalone guard.

## 5. IR / config surface (when unpinned)

- `t_cond: Option<f64>` threaded through `SMCConfig` / `IF2Config` / PGAS config
  (not the IR observation model). OCaml serde + Rust + schema if it lands in the
  IR config; bump `ir/VERSION`.
- Filter-start logic in `particle_filter.rs` (and PGAS/IF2 mirrors): warm-up
  loop (propagate-only) then conditioned loop from `t_cond`.
- Tests: red→green proving (a) `t_cond` default reproduces current behavior
  bit-for-bit on a data-from-`t_start` model; (b) a burn-up model gives a
  one-cadence first window; (c) the warm-up ensemble has nonzero variance at
  `t_cond` (negative control against the degenerate-seed failure mode).

## 6. References

- gh#134; the Kano measles repro (§1).
- He et al. (2010) _J. R. Soc. Interface_ 7:271–283.
- gh#133 (PF degeneracy / wall-clock watchdog split).
- Parent proposal: `2026-05-30-unified-observation-data.md`.
