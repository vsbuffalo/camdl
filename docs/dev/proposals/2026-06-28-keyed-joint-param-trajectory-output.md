# Keyed-joint `(θ, X)` fit output — pairing parameters with latent trajectories

Status: proposed Supersedes: the initial draft of this file (which left the join
scope, the latent ADT, and the join key as open questions). Relates-to:
`2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` (Phase 4 — its
can't-drop-uncertainty invariant is **already enforced**, see §1),
`2026-06-25-counterfactual-contrasts.md` (the conditioned fork this unblocks;
its prerequisite #1), `2026-06-28-start-from-state-engine-seam.md` (prerequisite
#2, which consumes `X_i(T*)`).

## Problem

A completed Bayesian fit produces, per posterior draw _i_, two things:

- **θ_i — the parameters.** Stored today one row per draw in the stage's
  `draws.tsv`, **pooled across chains with no key column** (`fit/pgas.rs` writes
  every estimated+fixed column, no `chain`/`draw`).
- **X_i — the latent trajectory.** The reconstructed unobserved epidemic curve.
  Stored today (PGAS only) in per-chain `chain_N/trajectories.tsv`, keyed
  `(chain, draw)` — but **strided by `traj_stride` independently** of the `thin`
  applied to `draws.tsv`, so the two retain _different_ subsets of sweeps.

So the two cannot be joined: `draws.tsv` carries no key, and even with one the
saved sets differ. PMMH/PF save **no** latent path. The conditioned
counterfactual fork (`2026-06-25-counterfactual-contrasts.md`) must read
`X_i(T*)` paired with `θ_i`; today there is no such pair.

## 1. Phase 4's can't-drop-uncertainty invariant is already enforced

Recorded so Phase 4 of the sealed-fit proposal is not mistaken for unstarted
work. _A predictive/quantity band can only be built from the posterior cloud,
never a collapsed point_ is **already enforced** by
illegal-states-unrepresentable typing: `PosteriorDraws` (`fit/predict.rs`) has a
sole non-empty constructor with private `draws`; every band producer takes it by
type; `ParamTreatment` **refuses** a `PlugIn` (optimizer) fit; pinned by
`fit_predict_refuses_an_optimizer_fit`. The ODE deterministic-X case is already
typed (`NotFilterable::Deterministic`). So `PosteriorDraws` **is** the
proposal's `Ensemble` minus the latent dimension this output supplies;
per-parameter `provenance` is a separate sealed-fit concern and is **out of
scope here**.

## 2. Goal — a partial, keyed `(θ_i, X_i)` join

For every draw that has a saved latent path, pair its parameters and its
trajectory under one `(chain, draw)` key at one cadence, so a consumer reads
`(θ_i, X_i)`. **The join is partial by design** (decided): paths are
substep-resolution and large, so storing one per draw at the `thin` cadence
blows up at national scale (the 774-LGA cVDPV2 case). v1 pairs only the
**path-saved subset** — and **surfaces the joined count**, so "cases averted"
bands honestly over that subset, never silently over fewer draws than the
parameter posterior. (Full-coverage join + a PMMH/PF latent-path writer are
named follow-ups in §8.)

## 3. The latent classifier (`LatentPath`) — a 3-state ADT, the fork-validity axis

X is not the same object across fits, and "is this draw forkable?" is decided by
the latent **artifact**, not the method name. This ADT — not an extension of the
particle-filter-drive `FilterableFit` witness — is the fork-validity axis
(decided; `FilterableFit` rejects ODE as `Deterministic`, the opposite verdict a
fork needs):

```rust
enum LatentPath {
    /// Deterministic backend (ODE): X = integrate(θ) exactly. Nothing stored;
    /// recomputed from θ on demand. Forkable (θ alone is the joint).
    Deterministic,
    /// Stochastic backend with a stored realization for this draw. Forkable.
    Sampled(Trajectory),   // the projected output `Trajectory` (carries int_state.counts at any t)
    /// Stochastic backend with NO saved path for this draw — every PMMH/PF fit
    /// today, and any PGAS draw outside the saved (traj_stride) subset. NOT
    /// forkable; the contrast must skip it (and the skipped count is surfaced).
    NotSaved,
}
```

The third state is load-bearing: forcing a path-less stochastic draw into
`Deterministic` is silent-wrong (it asserts θ determines X for a stochastic
backend). This reconciles with the contrasts doc's `NotForkable::PathsNotSaved`:
that doc is updated to classify fork-validity via `LatentPath`, and to retract
its "extend `FilterableFit`" directive.

The sealed-fit `Ensemble` grows the latent dimension additively (the §1
invariant untouched): `PosteriorDraws` gains a `latent: Vec<LatentPath>` aligned
to `draws` when this lands; `provenance` is the sealed-fit proposal's separate
concern.

## 4. The join key — `(chain, draw)`, not a flat index

`draws.tsv` pools all chains and sweep/`draw` numbers are unique only _within_ a
chain, so the key must be `(chain, draw)`:

- `draws.tsv` gains leading `chain` and `draw` columns (today it has neither).
- `chain_N/trajectories.tsv` already carries `(chain, draw)`
  (`PosteriorDraw {
  chain, draw, .. }`).
- The join is the inner join on `(chain, draw)`: the path-saved subset. Its size
  (joined-draw count vs total draws) is surfaced in the fit's diagnostics and
  carried to the contrast band.

## 5. Loader / validator changes (the "additive" caveat)

Adding key columns to `draws.tsv` is **not** transparent — name these in the
build:

- `fit predict` validates that _every_ `draws.tsv` column is a declared model
  parameter and hard-errors otherwise (`predict.rs:990-1008`). The
  `chain`/`draw` key columns trip this — the loader must strip/whitelist them
  (or carry the keys in a sidecar that the existing param-only `draws.tsv`
  references).
- `FitResult::resolve` loads only `draws.tsv` (`predict.rs:551-578`) — it gains
  a join step against `trajectories.tsv` to assemble `LatentPath` per draw.
- A **row-count / join-count invariant test** (the
  `2026-06-28-pgas-draws-double-thinning.md` incident's process ask): the paired
  output pins the joined count, so a truncated/duplicated join is caught — band
  shape alone hides it.

## 6. Cadence contract — stored-X must contain every legal fork instant T\*

Hard invariant (couples this output to the engine seam, which requires on-grid
T\*): **the stored-X cadence must include every fork instant T\* a contrast may
choose, and T\* must lie on the output grid.** Contrasts pick T\* _after_ the
fit (`at [origin + 20 'weeks]`) and need the integer compartment state `X_i(T*)`
at that instant — integer counts cannot be interpolated, so the path must
already carry that snapshot. The `Trajectory × T* → (IntState, RealState)`
extraction is the seam the engine consumes (§8). Contrasts re-simulate _forward_
from `X_i(T*)` with a fresh per-arm seed, so the stored path _after_ T\* and any
RNG state are **not** needed here (no RNG is persisted).

## 7. Identity — re-encode is neutral; a new what-gets-stored knob re-keys

Two halves (the draft conflated them):

- **Re-encoding** `draws.tsv` (adding `chain`/`draw` columns) is
  **identity-neutral**: `fit/cas.rs` hashes the fit's _inputs_ (model IR + data
  digests + config + engine), never the `draws.tsv`/`trajectories.tsv`
  _content_.
- **Any new knob that changes _which_ draws/paths are stored** (a cadence
  policy, a "save PMMH paths" toggle, full-vs-partial coverage) is
  output-shaping and **must re-key**, exactly as `n_trajectories` already does
  (`cas.rs:18-24, :328` — "a count that changes stored output has to change the
  run_id"). Re-keys here are deliberate and version-bumped, never collateral.

## 8. The work (partial-join v1)

1. **`(chain, draw)` key on `draws.tsv`** + the loader/validator whitelist (§5).
   Identity-neutral (§7).
2. **The `LatentPath` classifier** (§3) and the inner join on `(chain, draw)` in
   `FitResult::resolve`; surface the joined-subset count.
3. **The `Trajectory × T* → (IntState, RealState)` extraction seam** the engine
   seam consumes (§6).
4. **Reconcile the contrasts doc** (`2026-06-25`): fork-validity via
   `LatentPath` (retract "extend `FilterableFit`"); the dimcheck
   stored-dimension prerequisite (its #5) is folded into that doc.

**Named follow-ups (deferred, not v1):**

- A **latent-path writer for PMMH and the bootstrap PF** — they save none today,
  so their fits are `NotSaved` (not forkable) until this lands. Net-new, in the
  inference writers (`fit/pmmh.rs`, the PF path) — high-risk; its own change.
- **Full-coverage join** (every draw gets an X) — needs a path-memory bound
  (store only `X(T*)` + the forward grid, not the full path) before national
  scale; re-keys (§7).

## Decisions recorded

- **Partial join (path-saved subset), with the joined count surfaced** — full
  coverage and a PMMH/PF path writer are deferred follow-ups.
- **Fork-validity is classified by the 3-state `LatentPath` ADT**
  (`Deterministic | Sampled | NotSaved`), NOT by extending `FilterableFit`
  (which gives ODE the opposite verdict); the contrasts doc is updated to match.
- **Join key is `(chain, draw)`** — `draws.tsv` is keyless and pooled today.
- **Cadence contract: stored-X must contain every legal fork instant T\*** (and
  T\* on the output grid); no RNG is stored (contrasts use a fresh per-arm
  seed).
- **Identity split**: re-encoding `draws.tsv` is neutral; a new what-gets-stored
  knob re-keys like `n_trajectories`.
- **`provenance` is out of scope** (a sealed-fit concern); the `Ensemble` latent
  dimension is additive onto `PosteriorDraws`.
