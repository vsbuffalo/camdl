# Keyed-joint `(θ, X)` fit output — pairing parameters with latent trajectories

Status: draft (a captured design, not yet a spec to implement against — open
questions are intentional). Relates-to:
`2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` (Phase 4, the
typed `Ensemble` + can't-drop-uncertainty invariant — whose **invariant half is
already enforced**, see §1), `2026-06-25-counterfactual-contrasts.md` (the
conditioned-fork feature this output unblocks, its prerequisite #1).

## Problem

A completed Bayesian fit produces, for each posterior draw _i_, two things that
describe that draw:

- **θ_i — the parameters.** R₀, the recovery rate, the reporting fraction. The
  numbers estimated. Stored today, one row per draw, in the stage's `draws.tsv`.
- **X_i — the latent trajectory.** The unobserved true epidemic curve that draw
  implies: the hidden state at each time behind the noisy data. Stored today
  (for PGAS) in `trajectories.tsv`.

Today these two are **not joinable**. `draws.tsv` is thinned by the fit's
`thin`; `trajectories.tsv` is strided by an independent `traj_stride`
(`fit/pgas.rs:588` vs `:1031`), and neither carries a shared draw key — so there
is no way to read "the parameters θ_417 _and_ the specific curve X_417 they
produced." PMMH and the bootstrap particle filter save **no** latent path at all
(`fit/pmmh.rs`).

That missing pairing is the single prerequisite blocking two things:

1. **Counterfactual contrasts** (`2026-06-25-counterfactual-contrasts.md`): "how
   many deaths would an earlier campaign have averted?" forks each draw's
   _trajectory_ X at a conditioning time T\*, replays it two ways, and
   differences the arms — carrying the **joint (θ, X) posterior** so the
   contrast band is honest. Without paired `(θ, X)` there is nothing to fork.
2. **The latent dimension of the sealed-fit `Ensemble`** (Phase 4 of the
   sealed-fit proposal). See §1: the _invariant_ half of Phase 4 is already
   shipped; this output is what the _materialization_ half needs.

## 1. Phase 4's can't-drop-uncertainty invariant is already enforced

Recording this so the sealed-fit proposal's Phase 4 is not mistaken for
unstarted work. The proposal's safety-critical deliverable — _a predictive or
quantity band can only be built from the posterior cloud, never from a collapsed
point_ — is **already enforced by illegal-states-unrepresentable typing**,
today:

- `PosteriorDraws` (`fit/predict.rs`) is the typed cloud. Its **only**
  constructor rejects an empty draw set, the `draws` field is private, and the
  band producers (free-forward, the one-step `FilterableFit`, the quantity
  evaluator) all take it by type. A band over a hand-collapsed point is
  unconstructible.
- `ParamTreatment` (`fit/predict.rs`, "the safety-critical axis") splits
  `Posterior(PosteriorDraws)` from `PlugIn { method, stage }`; the predict path
  **refuses** a `PlugIn` (point-estimate) fit (`plugin_refusal`), so an
  optimizer fit can never be silently banded as if it carried posterior spread.
- The ODE deterministic-X case is already typed: `NotFilterable::Deterministic`
  gates the one-step horizon out for ODE, because given θ the trajectory is a
  point mass — exactly the `LatentPath::Deterministic` distinction §3
  formalizes.
- Pinned by `fit_predict_refuses_an_optimizer_fit` (an IF2 fit is refused and
  writes no band).

So `PosteriorDraws` **is** the proposal's `Ensemble`, minus two fields the
proposal also lists — per-parameter provenance and the latent trajectory. Both
of those are what this output supplies; neither is part of the already-shipped
invariant. The remaining Phase-4 work is therefore _additive_ onto a type whose
guarantee already holds — there is no reshape, which is why building the latent
dimension blind was the only real risk, and this proposal removes it.

## 2. Goal

A fit emits a **keyed, joinable `(θ, X)`** output: for every retained draw, its
parameter vector and its latent trajectory share one draw key and one cadence,
so a consumer reads paired `(θ_i, X_i)`. The pairing is uniform across inference
methods (PGAS, PMMH, PF) modulo the deterministic/stochastic distinction below.

## 3. The latent-path representation (an ADT, not a flag)

X is not the same kind of object across backends, so the join carries a typed
latent source rather than always-a-stored-path:

```rust
enum LatentPath {
    /// Deterministic backend (ODE): X = integrate(θ) exactly. Nothing is
    /// stored — the trajectory is recomputed from θ on demand. The pairing is
    /// trivial (θ alone is the joint).
    Deterministic,
    /// Stochastic backend (chain-binomial via PGAS/PMMH/PF): X is one sampled
    /// realization given θ. Two draws with the SAME θ can differ, so the path
    /// must be stored, keyed to its draw.
    Sampled(Trajectory),
}
```

This makes the ODE case free (no storage, no join problem) and isolates the real
work to the stochastic backends. It also matches the
`NotFilterable::Deterministic` distinction the runtime already encodes (§1).

The sealed-fit `Ensemble` grows this additively:

```rust
struct Ensemble {
    draws:      Vec<Map<Param, f64>>,     // θ cloud — already PosteriorDraws today
    provenance: Map<Param, ValueSource>,  // per-parameter source (reuses ValueSource)
    latent:     LatentSource,             // NEW: Deterministic | Sampled(paths) — this output
}
```

— so when this lands, `PosteriorDraws` gains `provenance` + `latent` and is
renamed (or aliased) to `Ensemble`; the band invariant (§1) is untouched.

## 4. The work (and why it is inference-adjacent)

1. **A shared draw key + common cadence across `draws.tsv` and the stored
   paths.** Today `thin` and `traj_stride` diverge (`fit/pgas.rs:588` vs
   `:1031`). Reconcile to one retained-draw index that keys both.
   _(Independently flagged in the contrasts proposal: `draws.tsv` may
   double-apply burn-in/thin at `fit/pgas.rs:1029` — verify with a TDD test
   while here.)_
2. **A latent-path writer for PMMH and the bootstrap PF.** They save no path
   today; the conditioned fork needs `X(T\*)` from these methods too. Net-new,
   touching `fit/pmmh.rs` / the PF path. High-risk (inference writers) — treat
   as such.
3. **The ODE path: nothing to store.** `LatentPath::Deterministic`; a reader/
   forker recomputes X by integrating θ. The only code is the enum + the
   recompute-on-read seam.
4. **(Downstream, the contrasts feature itself.)** Two-arm conditioned replay +
   the differencing reducer + the per-draw contrast band — specced in
   `2026-06-25-counterfactual-contrasts.md`, built on top of this output.

## Open questions (intentional — this is a draft)

- **Cadence policy.** Does the paired output store X at the data cadence, the
  integrator cadence, or a declared snapshot grid? The contrast only needs X at
  T\* (the conditioning instant) plus the post-fork forward grid — storing the
  full path may be wasteful at national scale (the 774-LGA cVDPV2 case).
- **Storage format + identity.** Is the paired output a new artifact (a re-keyed
  `joint.tsv`/parquet) or a join-key column added to the existing two files?
  Either way it is a fit-pipeline _output_ change — confirm it is
  identity-neutral (a presentation/derived artifact, not a hashed level) the way
  the sealed-fit annotations are.
- **PF path memory.** Saving per-draw latent paths from the bootstrap PF is
  O(draws × T × state) — bound it (or store only X(T\*) + a forward seed) before
  the 774-LGA case.
- **`Ensemble` rename vs alias.** When the latent dimension lands, rename
  `PosteriorDraws → Ensemble` (clearer-name churn vs proposal-vocabulary
  alignment — decide then, with the consumer in hand).

## Decisions recorded

- The latent source is an ADT (`Deterministic | Sampled`), not an always-stored
  path — ODE pays nothing, and the distinction already exists in the runtime
  (`NotFilterable::Deterministic`).
- Phase 4's _invariant_ (can't-drop-uncertainty) is **already shipped** (§1);
  this output supplies only the _materialization_ (provenance + latent), so the
  `Ensemble` completion is additive, not a reshape.
- This output is the gating prerequisite for counterfactual contrasts; it is
  scoped and proposed separately rather than folded into the sealed-fit work,
  because it is a fit-pipeline / inference-writer change, not an ergonomics one.
