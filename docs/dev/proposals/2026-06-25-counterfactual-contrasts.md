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

### Validity per inference method — classified by `LatentPath`

The conditioned fork is valid only for methods that produced a conditionable
latent state, and that differs by method. Fork-validity is classified by the
3-state `LatentPath` ADT from
[`2026-06-28-keyed-joint-param-trajectory-output.md`](2026-06-28-keyed-joint-param-trajectory-output.md)
§3 — `Deterministic` (ODE, valid: recompute X from θ) / `Sampled` (a stored
path, valid) / `NotSaved` (a stochastic fit with no saved path, rejected) — and
a point-estimate fit (no posterior cloud) rejects outright. This keys on the
latent **artifact**, not the method name. It does **not** extend the
particle-filter-drive `FilterableFit` witness (`predict.rs:197`): that witness
_rejects_ ODE as `NotFilterable::Deterministic`, the exact opposite of the
fork's verdict (ODE is the easy, valid case), so reusing it would overload one
enum arm with two opposite meanings. The validity table:

| method    | backend        | latent state                 | conditioned fork               |
| --------- | -------------- | ---------------------------- | ------------------------------ |
| PGAS      | chain_binomial | smoothed path per draw       | valid (CRN)                    |
| PMMH / PF | chain_binomial | path _if saved_ (today: not) | valid once saved (CRN)         |
| IF2       | chain_binomial | none (MLE point)             | rejected — `PointEstimate`     |
| MH        | ODE            | θ-determined                 | valid, deterministic (no CRN)  |
| **NLopt** | ODE            | θ-determined **point**       | **rejected — `PointEstimate`** |

NLopt on ODE is a point estimate (no posterior), so it rejects exactly like IF2
— the classifier keys on posterior-vs-point (the artifact), not on
`backend == Ode`. PMMH/PF are `Sampled` once their fits save a latent path, and
`NotSaved` (rejected) until then.

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

1. **Joint, keyed `(θ, X)` fit output** — specced in
   [`2026-06-28-keyed-joint-param-trajectory-output.md`](2026-06-28-keyed-joint-param-trajectory-output.md).
   `draws.tsv` is keyless and pooled across chains; `trajectories.tsv` is
   `(chain, draw)`-keyed but strided independently of `draws.tsv`'s `thin`. v1
   adds a `(chain, draw)` key to `draws.tsv` and inner-joins the **path-saved
   subset** (a partial join, with the joined count surfaced); a PMMH/PF
   latent-path writer and full-coverage join are deferred follow-ups there.
   _(The `draws.tsv` double-apply of burn-in/thin flagged in the original draft
   was fixed — `docs/dev/incidents/2026-06-28-pgas-draws-double-thinning.md`.)_
2. **A start-from-state engine seam.** The forward engine always builds initial
   state from the model at `t_start` (`chain_binomial.rs:172`; `SimConfig` has
   no state field). Injecting `X(T*)` and resuming the substep loop at `T*` is
   **high-risk** — it must re-seat the schedule/cursor (gh#216 firing), the
   flow-accumulator resets (`chain_binomial.rs:328`), and `t`. Not "moderate";
   treat as inference-adjacent.
3. **Multi-scenario two-arm replay in `fit predict`.** Today it builds one
   inline baseline (`predict.rs:860`); the paired two-arm replay + the
   differencing reducer + the per-draw contrast band are net-new.
4. **Fork-validity classifier** — the `LatentPath` ADT from prerequisite #1
   (`Deterministic | Sampled | NotSaved`) + a point-estimate rejection. NOT an
   extension of `FilterableFit` (the PF-drive witness, which rejects ODE — see
   "Validity per inference method" above).
5. **A stored quantity dimension** for the contrast binop-agreement check
   (`no_sia.deaths - with_sia.deaths` requires equal dims) — `dimcheck` does not
   persist computed dimensions today (`dimcheck.ml`). Owned here; see the
   "IR-side stored quantity dimension" sketch below.

### IR-side stored quantity dimension (prerequisite #5)

The `compare {}` binop `no_sia.deaths - with_sia.deaths` is an arithmetic
combination of two quantity values, so the dimensional checker must verify the
two operands agree (both `deaths`, a count) — otherwise a `deaths - rate`
contrast either silently produces a meaningless number or fails opaquely.
`dimcheck.ml` checks dimensions during expansion but does not **persist** the
computed dimension of a declared `quantities {}` entry into the IR. This is a
small OCaml/IR-side add: carry each quantity's resolved dimension on its IR node
(an `ir/schema.json` field on the quantity, mirrored OCaml↔Rust), so the Rust
`compare {}` reducer can check operand-dimension agreement (E-code on mismatch,
naming both quantities and their dimensions) before differencing. No new unit
literals or DSL surface — purely persisting a dimension `dimcheck` already
computes. (Sized as a follow-up alongside the contrast reducer, not a blocker
for prerequisites #1–#2.)

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
