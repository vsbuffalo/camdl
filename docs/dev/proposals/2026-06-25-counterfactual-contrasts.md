# Counterfactual contrasts (cases averted)

Status: **spec — implement.** The design is converged (three adversarial review
rounds) and its infrastructure prerequisites are now built: the joint keyed
`(θ, X)` output (#1), the start-from-state engine seam (#2), and the
`LatentPath` fork-validity classifier (#4) have landed (gh#322), and generated
quantities (`2026-06-25-generated-quantities.md`) shipped. What remains to build
is the DSL `contrasts {}` surface + the two-arm replay reducer (#3) and a stored
quantity dimension for the contrast binop check (#5).

Splits the counterfactual half out of
`2026-06-24-generated-quantities-and-counterfactuals.md` (superseded).
Supersedes `2026-06-04-experiment-compare.md`.

## What this delivers

The headline policy number — **cases averted**: how many cases an intervention
(an SIA) prevented, with full posterior uncertainty, as a banded
`contrasts/<name>.tsv`.

## Why not a forward-only version (the deferral rationale)

A counterfactual can be posed two ways, and they answer different questions:

- **Free-forward (prospective):** replay the posterior _parameters_ from `t0`
  under both arms. Answers "in outbreaks like the ones we inferred, how many
  cases would the SIA avert _on average_." A population-design object.
- **Conditioned (retrospective):** condition on the _observed_ outbreak — branch
  both arms from the fit's inferred latent state at the intervention time.
  Answers "for _this_ outbreak we saw, how many did the SIA avert."

"Cases averted" almost always means the **retrospective** question, and a user
who runs a forward-only `contrasts` and reads the prospective number as the
realized one has a silent-wrong on a policy headline. So this surface ships
**only** with the conditioned fork; a forward-only `contrasts` block is not an
acceptable interim. (The prospective object is recoverable via plain
`simulate --draws` under two scenarios for users who genuinely want it; it is
not dressed up as "cases averted".)

## The design (converged)

The `contrasts {}` block is **forward simulation that reads a fit's output** —
it never invokes the inference kernels. Per posterior draw `i`:

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

### Why the smoothed `X_i(T*)`, not a filtered one

`X_i(T*)` is the fit's **smoothed** latent state — conditioned on the _whole_
observed series `y_{1:T}`, including data _after_ the fork (PGAS samples the
joint smoothing distribution `p(θ, X | y_{1:T})`). Using post-`T*` data to set
the state we fork from looks suspect, so it is worth saying why it is correct —
and when it would not be.

This surface ships the **retrospective** estimand: "for _this_ outbreak we
actually observed, how many cases did the SIA avert?" That is a
**counterfactual** query, and a counterfactual is _defined_ by conditioning on
the factual evidence (Pearl's abduction → action → prediction: condition the
latent variables on _all_ the evidence, then `do(no SIA)`, then predict).
Conditioning the latent state on all of `y_{1:T}` **is** smoothing. The
post-`T*` data is not the future leaking into the counterfactual: `X_i(T*)` is a
**pre-fork, shared** quantity — both arms are identical up to `T*` and diverge
only when the SIA is applied _after_ the fork — so the later observations
sharpen our estimate of the one true state that existed at `T*`, _through the
known factual forward model_ (`X(T*) → SIA →
dynamics → y_{>T*}`). A fast
post-`T*` decline tells us how many were really infected at `T*`; the smoother
uses it, a filter (state given `y_{1:T*}` only) throws it away.

Filtering would be correct for a **different** estimand — the _prospective_,
decision-time question "with only what was knowable at `T*`, how much would an
SIA be expected to avert?" — which deliberately ignores hindsight. We do not
ship that one (a user reading the prospective number as the realized one is the
silent-wrong this proposal exists to avoid). So: retrospective ⇒ smoothing (read
`X` from the fit's smoothed trajectory, never a re-run filter).

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

contrasts {
  averted = no_sia.deaths - with_sia.deaths   over [origin + 20 'weeks, origin + 52 'weeks]
}
```

Each contrast bands over the forkable posterior subset and is emitted as
`contrasts/<name>.tsv` (the joined/forkable count surfaced alongside, per the
`(θ, X)` partial-join contract).

- The block is named **`contrasts {}`**, not `compare {}` — `camdl compare` is
  already a CLI subcommand (model Δelpd comparison, `compare.rs`); a model block
  and a CLI verb do not collide at parse time, but `contrasts` is the precise
  word and avoids the conceptual overload.
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
- Each entry is `name = <contrast_expr> over [<instant>, <instant>]`, with
  `over` a new keyword binding looser than arithmetic — a real grammar
  production with explicit precedence (`over` below the additive/subtractive
  operators, so `a - b over [..]` parses as `(a - b) over [..]`), not an
  asserted precedence.

## Prerequisites

Three of the five are built (gh#322); the remaining two — the `contrasts {}`
surface + reducer (#3) and the stored quantity dimension (#5) — are this
proposal's build.

1. **[done] Joint, keyed `(θ, X)` fit output** — specced in
   [`2026-06-28-keyed-joint-param-trajectory-output.md`](2026-06-28-keyed-joint-param-trajectory-output.md),
   landed. `draws.tsv` carries a leading `(chain, draw)` key inner-joined to the
   **path-saved subset** of `trajectories.tsv` (a partial join, with the
   forkable count surfaced via `fit::joint::classify_joint`); the shared loader
   strips the key so every param-only reader is unchanged. A PMMH/PF latent-path
   writer and full-coverage join are deferred follow-ups there.
2. **[done] A start-from-state engine seam** — landed. `chain_binomial`'s
   `run_chain_binomial_with_observer` resumes from an injected `X(T*)` at
   `cfg.t_start = T*` via `Resume{ start: Some(StartState{..}) }`: state inject,
   output-cursor re-seat, fresh/restored RNG, with reactive/off-grid rejections.
   Validated by the splice invariant. Reads `io::trajectories::read_state_at`
   for a `Sampled` path; an ODE arm recomputes `X(T*)` from θ.
3. **[build] The `contrasts {}` surface + two-arm replay reducer.** The DSL
   block (DOT member-access, `over` keyword, endpoint type check), then the Rust
   reducer: for each forkable draw, replay arm A and arm B from `X_i(T*)` via
   the engine seam, difference the quantities, and band over the forkable
   subset. Today `fit predict` builds one inline baseline (`predict.rs:860`);
   the paired two-arm replay + differencing + per-draw contrast band are
   net-new.
4. **[done] Fork-validity classifier** — the `LatentPath` ADT
   (`Deterministic | Sampled | NotSaved`) landed in prerequisite #1's
   `fit::joint`, with a point-estimate (no-posterior) rejection. NOT an
   extension of `FilterableFit` (the PF-drive witness, which rejects ODE — see
   "Validity per inference method" above).
5. **[build] A stored quantity dimension** for the contrast binop-agreement
   check (`no_sia.deaths - with_sia.deaths` requires equal dims) — `dimcheck`
   does not persist computed dimensions today (`dimcheck.ml`). Owned here; see
   the "IR-side stored quantity dimension" sketch below.

### IR-side stored quantity dimension (prerequisite #5)

The `contrasts {}` binop `no_sia.deaths - with_sia.deaths` is an arithmetic
combination of two quantity values, so the dimensional checker must verify the
two operands agree (both `deaths`, a count) — otherwise a `deaths - rate`
contrast either silently produces a meaningless number or fails opaquely.
`dimcheck.ml` checks dimensions during expansion but does not **persist** the
computed dimension of a declared `quantities {}` entry into the IR. This is a
small OCaml/IR-side add: carry each quantity's resolved dimension on its IR node
(an `ir/schema.json` field on the quantity, mirrored OCaml↔Rust), so the Rust
`contrasts {}` reducer can check operand-dimension agreement (E-code on
mismatch, naming both quantities and their dimensions) before differencing. No
new unit literals or DSL surface — purely persisting a dimension `dimcheck`
already computes. (Sized as a follow-up alongside the contrast reducer, not a
blocker for prerequisites #1–#2.)

## Deferred to a follow-up (explicitly out of v1 scope)

These are named non-goals for v1, not unresolved design questions — v1 ships
whole-population, single-instant contrasts:

- Stratified contrasts + `a.b[p]` dot chains (v1 is whole-pop).
- Decoupling the conditioning instant from the accumulation window.
- `last_obs`/`first_obs` as named instants — define the resolver's time source
  for multi-stream (ragged) models.
- Reconcile the `free_forward` naming with the existing `Horizon::FreeForward`
  predict axis if a prospective surface is ever added.

## Decisions recorded

- Ship only the conditioned fork; no forward-only `contrasts` (misread risk).
- The `contrasts {}` block is forward sim that _reads_ the fit's latent `X(T*)`;
  no re-filter.
- Validity gated by the `LatentPath` ADT (`Deterministic | Sampled | NotSaved`),
  keyed on the latent **artifact** (posterior-vs-point), not the method name or
  the backend axis — explicitly NOT an extension of `FilterableFit` (which
  rejects ODE, the opposite verdict; see "Validity per inference method").
- CRN's win is the shared `X(T*)`, not post-fork noise cancellation (stated
  honestly).
- The infrastructure prerequisites (joint keyed `(θ, X)` output #1,
  start-from-state seam #2, `LatentPath` classifier #4) are named, not buried,
  and are now built (gh#322); this proposal builds the surface + reducer (#3)
  and the stored quantity dimension (#5).
