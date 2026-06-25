# Counterfactual contrasts (cases averted)

Status: **Design record — deferred.** The design is converged (three adversarial
review rounds), but the conditioned fork it requires is a real fit-output +
engine build, not buildable from today's artifacts. Gated on the prerequisites
below. It depends on `2026-06-25-generated-quantities.md` (ships first,
standalone).

Splits the counterfactual half out of
`2026-06-24-generated-quantities-and-counterfactuals.md` (superseded).
Supersedes `2026-06-04-experiment-compare.md`.

## What this delivers

The headline policy number — **cases averted**: how many cases an intervention
(an SIA) prevented, with full posterior uncertainty, as a banded
`compare/<name>.tsv`.

## Why not a forward-only version (the deferral rationale)

A counterfactual can be posed two ways, and they answer different questions:

- **Free-forward (prospective):** replay the posterior _parameters_ from `t0`
  under both arms. Answers "in outbreaks like the ones we inferred, how many
  cases would the SIA avert _on average_." A population-design object.
- **Conditioned (retrospective):** condition on the _observed_ outbreak — branch
  both arms from the fit's inferred latent state at the intervention time.
  Answers "for _this_ outbreak we saw, how many did the SIA avert."

"Cases averted" almost always means the **retrospective** question, and a user
who runs a forward-only `compare` and reads the prospective number as the
realized one has a silent-wrong on a policy headline. So this surface ships
**only** with the conditioned fork; a forward-only `compare` block is not an
acceptable interim. (The prospective object is recoverable via plain
`simulate --draws` under two scenarios for users who genuinely want it; it is
not dressed up as "cases averted".)

## The design (converged)

`compare` is **forward simulation that reads a fit's output** — it never invokes
the inference kernels. Per posterior draw `i`:

```
read θ_i  and  the fit's inferred latent state X_i(T*) at the fork T*
arm A (factual):        forward-sim from X_i(T*), no intervention,  seed s_i
arm B (counterfactual): forward-sim from X_i(T*), with intervention, seed s_i   ← CRN
averted_i = quantity(arm A) − quantity(arm B)
band averted over draws
```

Both arms branch from the **same** `X_i(T*)`, so there is **zero pre-fork
variance**; the band over `averted_i` carries the **joint (θ, X) posterior**
(parameter _and_ latent uncertainty), since a stochastic-fit draw is the joint
`(θ_i, X_i)`.

**CRN, stated honestly.** Sharing the seed makes the firing substep
byte-identical across arms (the SIA `transfer` is RNG-free). After the transfer
changes the compartment counts, the two streams desynchronize (the backend
consumes a rate-dependent number of RNG words), so the post-fork coupling is
correlated and **decays** — CRN does _not_ cancel the post-fork forward noise.
The real variance-reduction win is the shared `X_i(T*)` (eliminating latent
variance at the fork), not forward-noise cancellation.

### Validity per inference method — extend `FilterableFit`, don't fork it

The conditioned fork is valid only for methods that produced a conditionable
latent state, and that differs by method. camdl already has the witness pattern
— `FilterableFit` (`predict.rs:197`), whose `NotFilterable::Deterministic` arm
already encodes "ODE: deterministic given θ → conditioned ≡ free-forward."
**Extend it** with a conditioned-fork capability rather than standing up a
parallel `ForkableFit` (the v3 mistake — reach for the existing seam). The
classification keys on the **artifact** ("does the stage have a `draws.tsv`
cloud? aligned latent paths?"), _not_ the method name (`predict.rs:234`), so:

| method    | backend        | latent state                 | conditioned fork               |
| --------- | -------------- | ---------------------------- | ------------------------------ |
| PGAS      | chain_binomial | smoothed path per draw       | valid (CRN)                    |
| PMMH / PF | chain_binomial | path _if saved_ (today: not) | valid once saved (CRN)         |
| IF2       | chain_binomial | none (MLE point)             | rejected — `PointEstimate`     |
| MH        | ODE            | θ-determined                 | valid, deterministic (no CRN)  |
| **NLopt** | ODE            | θ-determined **point**       | **rejected — `PointEstimate`** |

NLopt on ODE is a point estimate (no posterior), so it rejects exactly like IF2
— the witness must key on posterior-vs-point (the artifact), not on
`backend == Ode`.

### Surface

```
interventions { sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 20 'weeks] }
scenarios     { no_sia { disable = [sia] }   with_sia { enable = [sia] } }
quantities    { deaths = final(D) }

compare {
  averted = no_sia.deaths - with_sia.deaths   over [origin + 20 'weeks, origin + 52 'weeks]
}
```

- A **dot** member-access (`no_sia.deaths` = quantity under scenario) — one
  general namespace operator (verified non-breaking: `.5`/`1.5` stay floats,
  `ident.ident` takes a new `DOT` token). v1 restricts to `IDENT DOT IDENT`;
  stratified contrasts (`no_sia.deaths[p]`) and the `patch` output column are a
  follow-up — **v1 contrasts are whole-population only**.
- The window's `from` instant is the fork; endpoints are **instants** in the
  existing typed-time system (`origin + 20 'weeks`, `date(...)`; verified to
  parse). A new **endpoint type check** rejects a bare duration (today
  `at [20 'weeks]` compiles silently — this is new code, reconciled with the
  same `at` loophole).
- `compare {}` is a block; each entry is
  `name = <contrast_expr> over [<instant>, <instant>]`, with `over` a new
  keyword binding looser than arithmetic (a real production with explicit
  precedence, not asserted). **Name clash to resolve:** `camdl compare` is
  already a CLI subcommand (model Δelpd comparison, `compare.rs`); the block
  name overloads it — pick a non-colliding block name or justify the overload.

## Prerequisites (the real build — why this is deferred)

1. **Joint, keyed `(θ, X)` fit output.** Today `draws.tsv` is keyless (params
   only) and `trajectories.tsv` is sweep-keyed and strided by `traj_stride`
   _independently_ of the `thin` used for `draws.tsv` (`fit/pgas.rs:588` vs
   `:1031`), so the two are not joinable — there is no `(θ_i, X_i)` pair to
   read. This needs a common stride + a join key (a fit-pipeline output change).
   PMMH/PF save **no** latent path today (`fit/pmmh.rs`) — a net-new path
   writer. _(Also flagged: `draws.tsv` may double-apply burn-in/thin —
   `fit/pgas.rs:1029` re-thins an already-thinned index; verify with a TDD test,
   independent of this work.)_
2. **A start-from-state engine seam.** The forward engine always builds initial
   state from the model at `t_start` (`chain_binomial.rs:172`; `SimConfig` has
   no state field). Injecting `X(T*)` and resuming the substep loop at `T*` is
   **high-risk** — it must re-seat the schedule/cursor (gh#216 firing), the
   flow-accumulator resets (`chain_binomial.rs:328`), and `t`. Not "moderate";
   treat as inference-adjacent.
3. **Multi-scenario two-arm replay in `fit predict`.** Today it builds one
   inline baseline (`predict.rs:860`); the paired two-arm replay + the
   differencing reducer + the per-draw contrast band are net-new.
4. **Extend `FilterableFit`** with the conditioned-fork capability + the
   `NotForkable::{PointEstimate, PathsNotSaved}` outcomes, keyed on artifact.
5. **A stored quantity dimension** for the contrast binop-agreement check
   (`no_sia.deaths - with_sia.deaths` requires equal dims) — `dimcheck` does not
   persist computed dimensions today (`dimcheck.ml`), so this is a small IR-side
   add.

## Open follow-ups within this proposal

- Stratified contrasts + `a.b[p]` dot chains (v1 is whole-pop).
- Decoupling the conditioning instant from the accumulation window.
- `last_obs`/`first_obs` as named instants — define the resolver's time source
  for multi-stream (ragged) models.
- Reconcile the `free_forward` naming with the existing `Horizon::FreeForward`
  predict axis if a prospective surface is ever added.

## Decisions recorded

- Ship only the conditioned fork; no forward-only `compare` (misread risk).
- `compare` is forward sim that _reads_ the fit's latent `X(T*)`; no re-filter.
- Validity gated by extending `FilterableFit`, keyed on artifact
  (posterior-vs-point), not method name or backend axis.
- CRN's win is the shared `X(T*)`, not post-fork noise cancellation (stated
  honestly).
- Conditioned-fork prerequisites (joint keyed `(θ, X)` output, start-from-state
  seam) are named, not buried — this is why the proposal is deferred behind
  quantities.
