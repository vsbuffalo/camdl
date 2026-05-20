# Proposal: individual sampling layer for compartmental simulation

Date: 2026-05-19
Status: Draft (RFC) — replaces `simulate_with_lineages` v1
Branch: `feature/lineages`
Scope: forward simulation only. Orthogonal to the inference stack —
nothing in `pgas.rs` / `if2.rs` / `particle_filter.rs` changes.

## Problem

camdl's backends track compartment counts, not individuals. A
transmission event decrements a susceptible count and increments an
infected count; *which* infectious individual was the infector is
never recorded, so a transmission tree is not a byproduct of
count-level simulation. More generally, individual-level event
histories — sojourn times, recovery times, cohort trajectories — are
not recoverable from count-level state even though the model fully
determines them in distribution.

For benchmark generation, phylodynamic-inference validation, and
individual-level diagnostic outputs, we need an *identity layer* on
top of the existing simulator. This proposal adds it as a strictly
additive feature: count-level dynamics are unchanged byte-for-byte
when the feature is off, and the runtime cost is zero when not in use.

## Key reframings

Three reframings carry through the rest of the design.

**1. The artifact is a line list. Trees are a projection.** The
primary object the augmented simulator emits is a per-event line list:
one record per identity-tracked event (infection, progression,
recovery, death, …) with timestamp, deme/stratum, individual ID, and
— at events with parent-child structure — the parent ID. The
transmission *tree* is that line list filtered to `#[lineage]` events
and pruned to sampled tips. Survival curves, sojourn distributions,
cohort analyses are other projections. Same primary artifact, many
consumers.

**2. The math is a Markovian refinement of the existing CTMC, with
uniqueness derived for linear rates.** Treat each compartment as
carrying a multiset of distinguishable IDs. When the CTMC fires an
event, sample the specific individual(s) involved according to the
rate's pair-decomposition. For linear mass-action rates and their
stratified analogues, the pair-decomposition is uniquely determined
by the count-level rate — uniform within-pool sampling is *derived*
from the structure of the rate, not chosen. Aggregating IDs recovers
the original count-level CTMC by theorem. v1 restricts to linear
rates precisely so this derivation holds without modeling assumptions;
nonlinear rates (He alpha-mixing, log-saturation, Michaelis–Menten
saturation in the infector) are deferred to v2, where the
augmentation becomes a documented modeling choice rather than a
derivation.

**3. The annotation is a semantic claim, not an inference target.**
Mathematical structure `β·S·I/N` is identical across disease
transmission, predator-prey reproduction, information diffusion, and
chemistry. The rate AST cannot tell which interpretation applies. The
user must declare that a transition has parent-child semantics; once
declared, the rate AST's linear structure tells us *who the parent is*.
The declaration is in-model; everything else follows from it.

## DSL change

One new annotation form, one compile-time check, one lexer change.
That is the entire surface-level addition for v1.

### `#[lineage]` annotation on transitions

A transition is marked as a lineage event by attaching `#[lineage]`
above or inline:

```camdl
transitions {
  #[lineage]
  infection : S --> E  @ beta * S * I / N

  progression : E --> I  @ sigma * E
  recovery    : I --> R  @ gamma * I
  vaccination : S --> V  @ nu * S
}
```

Reading the annotation: this transition represents a parent-child
lineage event. At firing time, a parent is sampled from the parent
pool (identified by linear-decomposition analysis on the rate AST,
see below) and a new tracked individual is minted in the destination
compartment.

Inline form for terse cases:

```camdl
#[lineage] infection : S --> E  @ beta * S * I / N
```

Both forms produce identical IR.

### Linear-in-parents requirement

For a `#[lineage]` transition to compile in v1, its rate expression
must be linear in every compartment it references that is not the
transition's source. Formally:

> A rate expression `r` is *linear in parents* iff `r` can be written,
> after expansion of let-bindings and forcing-function references, as
> a sum of products `(factors_without_parents) · L`, where:
>
> - `factors_without_parents` may reference parameters, time, source
>   compartments, and non-source compartments appearing only inside
>   normalizing denominators (e.g., `S+E+I+R` in `β·S·I/(S+E+I+R)`);
> - `L` is either a single non-source compartment reference (e.g., `I`,
>   `I[b]`), or a sum of such single-reference terms (e.g.,
>   `β_I·I + β_A·A`), with each non-source compartment appearing at
>   most once and linearly.

This is decidable by AST inspection. The compiler classifies every
Pop reference in the rate as one of:

- **Source.** The transition's source compartment. Not a parent.
- **Denominator-only.** Appears only inside `/N` or analogous
  normalizers. Not a parent.
- **Linear parent candidate.** Appears as a top-level multiplicative
  factor (or summand of multiplicative factors) outside any nonlinear
  function. Parent.
- **Nonlinear use.** Appears inside a non-unit power, `log`, `exp`,
  `sqrt`, `Cond`, `min`, `max`, or any other nonlinear function
  applied to a Pop reference. *Rejected* in v1.

**Precedence — normalizing denominators win.** A parent compartment
may appear in *both* the numerator (as a linear parent) and a
normalizing denominator. Division by a normalizer is formally a
nonlinear function, so a naive classifier would bucket the
denominator appearance as "nonlinear use" and wrongly reject
`β·S·I/N` — frequency-dependent transmission, the *most common* form.
The rule: **denominator/normalizer appearances are classified
Denominator-only and are exempt from the Nonlinear-use bucket;
the exemption is applied before the nonlinear-use check.** Linearity
is required only of the *numerator* dependence on parent counts; a
normalizer is a frozen coefficient at the event instant, even when it
references the parent compartment. Thus `β·S·I/N` compiles (`I` is a
linear parent in the numerator; its `N` appearance is a frozen
normalizer), while `β·S·(I+ι)^α/N` does not (the power is a genuine
nonlinear use of `I` in the numerator).

If any non-source Pop reference falls in the "nonlinear use" bucket,
the compiler emits:

```
E601: lineage tracking on transition 'infection' requires linear
dependence on parent compartments. Found nonlinear use of 'I' in the
rate expression (inside (I + iota)^alpha at position …).

Options:
  1. Rewrite the rate so 'I' appears as a top-level linear factor,
     potentially absorbing the nonlinearity into other parameters.
  2. Remove the #[lineage] annotation; v1 lineage tracking does not
     support nonlinear parent dependence.
  3. Wait for Phase 4 lineage support, which will accept nonlinear
     rates with explicit attribution semantics.
```

**No `infector(...)` wrapper in v1.** The wrapper was needed for
nonlinear cases where the rate AST does not unambiguously identify
the parent pool; v1's linearity restriction removes those cases by
construction. The wrapper returns in Phase 4 alongside nonlinear rate
support and its accompanying modeling-choice documentation.

### Lexer rule

The `#` character begins a comment that extends to end of line,
*unless* immediately followed by `[`, in which case it begins an
attribute. To start a comment with `[`, add a space:
`# [this is a comment]`. One-character lookahead; documented in §1.1
of the language spec.

Acceptance check before locking in: grep the existing codebase,
golden files, and example models for any pre-existing `#[` patterns
to confirm zero migration cost. Expected result: no hits (camdl uses
`#` exclusively for line comments today). One-line CI check.

## Mathematical structure

The augmented process is a Markovian refinement of the count-level
CTMC. If the count-level state is `X(t) = (N_S(t), N_E(t), …)`, the
augmented state is `X̃(t) = (𝓘_S(t), 𝓘_E(t), …)` where each `𝓘_k(t)`
is a multiset of individual IDs with `|𝓘_k(t)| = N_k(t)`.

**Derivation of within-pool uniformity for linear rates.** For a
linear-in-parents rate `r = (factors not involving non-source
compartments) · L`, the linearity guarantees that `r` can be written
as a sum over individual contributors:

- **Single parent**, `r = α · I` where `α` is the rate's coefficient
  on the parent count, *evaluated at the current state* (normalizers
  such as `1/N` are frozen at their instantaneous value, even when `N`
  references `I` — see the denominator-precedence rule above): then
  `r = Σ_{i ∈ I-pool} α`. Each I-individual contributes `α` to the
  rate at that instant. When an event fires, the specific parent is
  sampled uniformly from the I-pool — this is *forced* by the
  instantaneous decomposition, not chosen. (For density-dependent
  `β·S·I`, `α = β·S` is literally independent of `I`; for
  frequency-dependent `β·S·I/N`, `α = β·S/N` is the frozen value of
  the normalizer at the event time. Both decompose to equal
  per-individual contributions.)

- **Multi-parent**, `r = α_1 · I_1 + α_2 · I_2`: then
  `r = Σ_{i ∈ I_1-pool} α_1 + Σ_{i ∈ I_2-pool} α_2`. The probability
  that the parent comes from pool `k` is `α_k · I_k / r`. Within the
  chosen pool, sampling is uniform.

- **Stratified**, `r = S · Σ_b C[a,b] · I[b] / N[b]`: same structure,
  with per-class weights `C[a,b]/N[b]`. Class `b` is selected with
  probability `C[a,b]·I[b]/N[b] / Σ_{b'} C[a,b']·I[b']/N[b']`; within
  the class, uniform.

In all three cases, the count-level rate aggregates to `r` exactly:
this is a theorem about the linear structure, not a modeling
assumption. Colloquially: for linear rates, each pair really does
contribute equally, so uniform within-pool sampling is the only
refinement consistent with the count-level dynamics.

**Attribution rules** the compiler emits to the IR:

| Event type | Attribution rule |
|---|---|
| Simple transition on identity-tracked source (no `#[lineage]`) | One ID sampled uniformly from source pool; transferred to destination pool. |
| Linear lineage event (`r = factors · Σ_b w_b · X_b` after expansion) | Parent pool `b` sampled with `P(b) ∝ w_b · X_b`; within pool, uniform; new child ID minted in destination, parent ID recorded. |
| Inflow (no source, e.g., importation) | New ID minted, parent = `ImportSentinel`. |
| Outflow (no destination, e.g., death) | One ID sampled uniformly from source; removed. |

These rules are *derived* from the rate AST given the `#[lineage]`
annotation and the linearity constraint. The compiler computes the
per-class weight expressions at compile time and emits them as
evaluable IR sub-expressions.

## Identity-tracked subgraph (inferred)

Given the set of `#[lineage]` transitions, the compiler computes the
**identity-tracked subgraph** by forward reachability:

1. Seed: destinations of `#[lineage]` events ∪ parent-pool
   compartments of `#[lineage]` events.
2. Close under: for every transition `c_1 → c_2`, if `c_1` is in the
   tracked set, add `c_2`.
3. Result: every compartment whose individuals should carry IDs.

Every transition involving identity-tracked compartments produces a
line list entry. Transitions purely within the untracked set produce
nothing (no overhead).

For SEIR with `#[lineage]` on `S→E`:
- Seed: `{E}` (lineage destination) ∪ `{I}` (parent pool). Set: `{E, I}`.
- Forward closure: `I → R` reachable, add `R`. Set: `{E, I, R}`.
- Untracked: `{S, V}`.

**Cyclic models and reachability cost.** Models with global cycles
(SIRS with `R → S` waning, demographic turnover with `R → death → birth → S`)
propagate the tracked set through the cycle: once `R` is tracked, the
`R → S` transition pulls `S` into the tracked set, which means IDs
must be minted for the entire initial susceptible population.

For a typical 774-LGA polio model with millions of initial
susceptibles, this is millions of IDs minted at `t=0`. Cost is
acceptable (IDs are `u64`, mint is a one-time O(N) operation, line
list growth is dominated by event firings not initial state) but it
needs to be reported by `camdl inspect --lineage` so users see the
memory and disk footprint before running.

The full inspection output:

```
$ camdl inspect model.camdl --lineage
Lineage events:
  infection : S --> E   parent pool: I

Identity-tracked compartments (forward-reachable):
  E, I, R
  (cycle detected: R → S adds S to tracked set; +N_S initial IDs)

Tracked event types per individual:
  infection (lineage event, parent recorded)
  progression, recovery, death_I, death_R, waning

Estimated initial IDs at t=0: 1,000,000 (from S init)
Estimated event firings during simulation: ~5,000,000
Estimated line list size: ~400 MB (Parquet, compressed)
```

## Type design

New `lineage` module in the `sim` crate:

```rust
struct IndividualId(u64);                  // monotone per-run counter

enum ParentRef {
    Individual(IndividualId),
    Import,                                // exogenous import
    Seed,                                  // initial population
    None,                                  // non-lineage event
    // Environment(...) reserved for Phase 5
}

struct LineListEntry {
    time:        f64,
    transition:  TransitionId,
    individual:  IndividualId,             // the focal individual
    source:      Option<CompartmentId>,    // pre-event compartment
    destination: Option<CompartmentId>,    // post-event compartment
    deme:        DemeId,
    parent:      ParentRef,                // populated at #[lineage] events
}

// Append-only; streamed to disk in Parquet, never held whole in RAM.
struct LineListWriter { /* writer handle */ }

// Per (deme, identity-tracked compartment) live identity pools.
struct IdentityState {
    pools: HashMap<(DemeId, CompartmentId), Vec<IndividualId>>,
    next:  u64,
}
```

Dynamics-unchanged seam:

```rust
trait TransitionObserver {
    // Called by the core loop AFTER it has drawn its own RNG and
    // decided how many of each transition fired. Cannot reorder the
    // simulation RNG.
    fn on_fired(&mut self,
                transition:   TransitionId,
                deme:         DemeId,
                multiplicity: u64,
                pre_state:    &State,
                rate_terms:   &EvaluatedRate,
                rng:          &mut LineageRng);
}
```

The simulator takes `Option<&mut dyn TransitionObserver>`. `None` →
today's behavior, byte-for-byte, zero overhead. `Some(...)` →
identity bookkeeping runs strictly downstream of the dynamics.

## RNG: separate stream is an invariant

Paired-seed CRN holds only while the simulation RNG is consumed in
the same order on both sides. If identity-attribution draws (parent
sampling, source sampling) came from the simulation's ChaCha8 stream,
a run with `--lineages` would diverge from the same run without it,
breaking determinism and every paired-scenario guarantee.

The identity layer owns an **independent RNG stream**, seeded
`main_seed ⊕ fixed_offset` (reproducible, disjoint). The
byte-identical-trajectory invariant is enforced as Tier 2a of the
validation suite.

## Backend matrix

| Backend | Lineage support | Parent attribution | Bias |
|---|---|---|---|
| Gillespie (SSA) | exact | One firing = one event; sample at the event time using the rate decomposition. | None within the model. |
| tau-leap | approximate | *k* firings against frozen start-of-step rates; sample from start-of-step pool. | Propensity-freezing **systematically loses parent–child edges with time difference shorter than `dt`** — intra-step events that should be sequential are collapsed against frozen state. For benchmark trees this biases the distribution of short-generation-time edges and the deepest coalescent intervals. |
| chain-binomial | approximate | As tau-leap. | Same as tau-leap. |
| ODE | **incompatible** | No individuals — hard error via `Capabilities::LINEAGES` not declared. | N/A. |

Documentation must state plainly: trustworthy benchmark trees want
Gillespie. The validation suite includes a diagnostic that measures,
for tau-leap and chain-binomial runs, the expected fraction of
would-be-sub-`dt` edges and reports it alongside the run.

## Online vs offline

**Online (during simulation):**

- Mint IDs at lineage events; resolve parent via the per-class linear
  weight decomposition at the event instant.
- Maintain identity pools for tracked compartments.
- Stream line list entries to an append-only Parquet writer; do not
  hold the full log in RAM.

**Offline (`camdl lineage <command>` post-processing):**

- Apply a sampling scheme to select observed tips/events.
- Project: extract a tree (parent–child edges from lineage events,
  pruned to sampled tips), compute survival statistics, or write a
  stratified line list. All pure functions of `(line list, projection
  spec)` — independently testable, re-runnable, cacheable.

This matches camdl's content-addressable ethos: the line list is a
deterministic function of `(model, params, seed)`; downstream
projections are deterministic functions of `(line list, projection
spec)`. Two cache keys, one expensive step.

## CLI

Lineage projections namespaced under `camdl lineage`:

```bash
# Forward simulation with lineage tracking
camdl simulate model.camdl --params P.toml --seed 42 --lineages

# Inspect the inferred identity-tracked subgraph and cost estimate
camdl inspect model.camdl --lineage

# Offline projections (all under 'lineage' namespace)
camdl lineage tree     line_list.parquet --scheme sampling.toml --output out.newick
camdl lineage sojourn  line_list.parquet --compartment I        --output sojourn.tsv
camdl lineage cohort   line_list.parquet --event infection      --output cohort.tsv
```

Without `--lineages`, annotations are parsed and stored in the IR
but the runtime tracking subsystem is not activated. Models with no
`#[lineage]` annotations run identically with or without the flag.

**Line list format: Parquet for production, TSV for debug.** Parquet
is the production format: columnar, streaming-friendly, scales to
millions of rows, native consumption by Polars / Pandas / DuckDB /
Arrow. Adds the `parquet` Rust crate to the dependency set. TSV is
supported via `--format tsv` for debugging small runs only; the docs
explicitly do not recommend it for production simulations.

## Validation

The code is moderate; *proving the attribution is correct* is the
load-bearing work. Tiers in increasing strength:

**Tier 1 — Structural invariants.** Every lineage-event child has
exactly one parent; parent is in the named pool at child's event
time; pruned tips equal sampled set; no unary nodes after pruning.

**Tier 2a — Trajectory invariance.** Count trajectory under
`--lineages` equals count trajectory without, byte-for-byte, for the
same seed. **Catches RNG-stream leakage. Does NOT test attribution
correctness** — a bug in parent attribution (e.g., sampling from `S`
instead of `I`) would still pass this test because count RNG draws
are untouched. Tier 2a is necessary but trivial.

**Tier 2b — Empirical attribution frequencies (load-bearing).**
Simulate `n = 10⁴` independent runs of a model with multiple parent
classes. For each lineage event, record which class provided the
parent. Compare observed class frequencies to the
linear-decomposition-predicted weights
`P(class b) = w_b · X_b / Σ w_{b'} · X_{b'}` evaluated at the event-time
state. Assert agreement within 3σ Monte Carlo error. **This is the
test that actually catches a wrong decomposition rule.** Tier 2a
without Tier 2b would let a wrong-pool-sampling bug ship.

**Tier 3 — Analytic.** Linear birth–death / Yule tree statistics
(Sackin imbalance, expected TMRCA, branch-length sums) have closed
forms — assert against them.

**Tier 4 — Large-N coalescent limit.** Under homogeneous mixing the
SIR transmission tree converges to the structured-coalescent
prediction (Volz 2009 / Rasmussen–Volz line). Specific testable
statistic: distribution of coalescent intervals at time `t` matches
`Exp(C(k,2) · 2 β S(t) I(t) / N(t)²)` within 2σ over 10⁴ replicates,
**for population sizes N ≥ 10⁴** — the diffusion approximation has
O(1/N) bias that makes the test flaky at smaller populations.

**Tier 5 — External oracle.** Cross-validate a *stratified* scenario
against an independent lineage-aware simulator (VGsim or MASTER,
pinned versions), run as a CI gate. Same pattern as the existing
pomp / scipy / numpy oracle tests.

Tier 2b and Tier 5 on the stratified case are the real deliverables:
anyone can get the well-mixed tree right; the value (and the
silent-wrong-answer risk) is contact-structured, time-varying parent
attribution.

## Phasing

**Phase 1.** Observer seam + separate RNG stream + Gillespie +
single-population linear rates + streamed Parquet line list +
`#[lineage]` parsing + identity-tracked subgraph inference + offline
tree pruner + Newick output + Tier 1 / 2a / 2b / 3 tests. Proves
the architecture and the attribution-correctness guarantees.

**Phase 2.** Stratified / spatial parent attribution + multi-class
linear decomposition + external-oracle CI gate (Tier 5). The
scientifically hard part, deliberately isolated.

**Phase 3.** tau-leap / chain-binomial backends (with documented
`dt`-bias diagnostic) + additional offline projections (sojourn
analysis, cohort summaries).

**Phase 4.** Nonlinear-in-parents rate support: `infector(...)`
wrapper for explicit attribution semantics, partial-derivative-based
between-pool weighting with symbolic sign-check, documented
modeling-choice semantics for within-pool sampling (principle of
insufficient reason / maximum entropy among permutation-symmetric
refinements). Until shipped, any `#[lineage]` transition with
nonlinear parent dependence errors with `E601`.

**Phase 5.** Environmental transmission via tagged-contribution
semantics on real-valued compartments. Until shipped, `#[lineage]`
whose parent pool is a real-valued compartment errors with `E602`.

## IR change

Additive fields. One on the transition record:

```
lineage: { is_lineage_event: bool,
           parent_pool_weights: [(CompId, AST)] } | null
```

`parent_pool_weights` is the linear decomposition of the rate over
parent pools — a list of `(compartment, weight_expression)` pairs.
For `β·S·I/N` with parent `I`, this is `[("I", β·S/N)]`. For
multi-class linear, it's the per-class list. For stratified
`S[a]·Σ_b C[a,b]·I[b]/N[b]`, expanded over `a` and listing each
`I[b]` with its weight `C[a,b]·S[a]/N[b]`.

One at the top level:

```
identity_tracked_compartments: [CompId]
```

Computed from forward reachability. Cached so the runtime doesn't
recompute. Empty when no `#[lineage]` annotations exist; in that
case the lineage subsystem is statically inert.

Atomic OCaml + Rust + golden-file update per `CLAUDE.md`'s "Changing
the IR schema" procedure.

## Open questions

1. **`#[transmission]` alias.** Ship with just `#[lineage]` and add
   the alias if users ask. The generality framing argues for
   `#[lineage]`; the epi-specific readability case is real but
   probably not load-bearing.

2. **Sampling scheme interface.** Phase 1 ships with `Flat(rate)`.
   Realistic uses need richer schemes (per-deme rates, time-varying
   rates, conditional-on-removal sampling, AFP-style surveillance for
   polio). Draft the `SamplingScheme` trait in Phase 1 even if only
   `Flat` is implemented, so the shape is locked.

3. **Documentation discipline for the linear restriction.** v1's
   restriction will surprise users with He et al. style models. Worth
   a prominent doc page explaining the restriction, the Phase 4
   roadmap, and how to linearly approximate common nonlinear models
   in the meantime.

4. **Within-pool heterogeneity hook.** v1 assumes individuals within
   a pool are exchangeable. Heterogeneous infectiousness (Lloyd-Smith
   superspreading, individual-level risk) would require stratifying
   the parent pool — the lineage feature does not provide a
   sub-stratum weighting hook, by design, to keep the linear
   derivation clean.

## Non-goals

- No change to inference, fitting, or any existing backend dynamics.
- **No nonlinear-in-parents rates in v1.** Hard error at compile time
  with a clear path forward (Phase 4).
- No within-host or sequence evolution in v1.
- No non-Markovian waiting-time distributions. The line list reports
  realized samples honestly from whatever distribution the
  compartmental structure implies (exponential, or Erlang via
  sub-staging). Arbitrary distributions require a non-Markovian
  extension to the language that is out of scope.
- No environmental transmission in v1 (Phase 5).
- No ODE lineage support (incoherent — hard error by capability).

## Future possible features

Forward-looking design sketches. Not v1 scope; recorded so the v1
architecture is built to admit them without rework.

### Nonlinear-in-parents mixing: the `infector(...)` wrapper (Phase 4)

v1 rejects nonlinear parent dependence (`E601`) because uniform
within-pool sampling can only be *derived* for linear rates. Many
real models are nonlinear in the infector count: He et al.
alpha-mixing `(I+ι)^α`, Michaelis–Menten / saturating infectiousness
`I/(K+I)`, log-saturation. Phase 4 admits them through an explicit
attribution annotation that mirrors `unchecked_dim`'s shape — the
`reason` field is required, and the wrapper is transparent at runtime
(identity over `expr`):

```camdl
#[lineage]
infection : S --> E  @ overdispersed(
    beta_base * seas * S
    * infector(
        (I + iota)^alpha,
        from = I,
        reason = "He et al. alpha-mixing: nonlinear infector contribution"
      )
    / pop(t),
    sigma_se
  )
```

Stratified, multi-class form:

```camdl
#[lineage]
infection[a in age] : S[a] --> E[a]  @ S[a]
  * infector(
      (sum(b in age, C[a,b] * I[b]))^alpha,
      from = I,
      reason = "Age-stratified He alpha-mixing"
    )
  / pop(t)
```

**Semantics.** The wrapper does two things the linear classifier
cannot do automatically:

1. *Pool identification.* `from = I` names the parent pool when the
   nonlinear shape prevents the classifier from reading it off the
   AST structure.
2. *Between-pool weighting via partial derivatives.* For a multi-pool
   nonlinear rate, the marginal contribution of pool `k` is
   `weight_k = (∂rate / ∂count(X_k)) · count(X_k)` — the local
   sensitivity of the rate to that pool's size, times its size. This
   reuses the compiler's existing symbolic differentiation
   (`autodiff.ml`, already emitting `rate_grad` for inference
   gradients). The compiler **sign-checks** the partial derivative:
   if it cannot prove non-negativity over the feasible region
   (non-monotonic infector dependence), it rejects the model rather
   than producing ill-posed (possibly negative) weights.

**The key semantic difference from v1.** Within-pool sampling is
still uniform, but here it is an explicit *modeling choice*, not a
derivation. A nonlinear aggregate rate like `(I+ι)^α` implies no
per-individual contribution mechanism — `α` is phenomenological — so
no refinement is forced by the count-level dynamics. Uniform
within-pool is justified as the maximum-entropy / principle-of-
insufficient-reason refinement among permutation-symmetric
augmentations, and the required `reason` field documents the
assumption at the call site. This is exactly the distinction the v1
linearity restriction exists to avoid having to make silently; Phase 4
makes it, explicitly and locally annotated.

The IR `parent_pool_weights` field generalizes unchanged: in v1 each
weight is a linear sub-expression of the rate; in Phase 4 it is the
partial-derivative expression emitted by `autodiff.ml`. The runtime
evaluation path (sample pool `∝ weight_k · count_k`, then uniform
within pool) is identical, so the runtime built for v1 needs no
change to support Phase 4 — only the compiler's weight-extraction and
the `infector(...)` parse path are added.

### Other directions (flagged, not designed)

- **Environmental transmission** (Phase 5): tagged-contribution
  semantics on real-valued reservoir compartments (each shedding
  event deposits a token tagged with the shedder ID; environmental
  infections sample from the live token pool weighted by recency and
  amount). Hard-errors as `E602` until shipped.
- **Sequence-evolution layer:** mutation accumulation along tree
  branches → simulated alignments → input to phylodynamic inference.
  The natural layer above the line list / tree; out of scope here.

## Why this proposal supersedes v1

The original `simulate_with_lineages` proposal framed the feature as
"add transmission line lists and trees." This version:

- Treats identity tracking as the primary capability, with lineage
  events as a special case carrying parent IDs.
- Uses `#[lineage]` Rust-style attributes, visually separated from
  rate content; the lexer change is one-character lookahead.
- **Restricts to linear-in-parents rates in v1**, which makes the
  Markovian-refinement guarantee a *derivation* rather than a
  modeling choice. Nonlinear support is explicit Phase 4 territory
  with documented attribution semantics. The colleague's critique of
  the v2 draft — that uniform-within-pool sampling for nonlinear
  rates is a modeling choice not a derivation — is eliminated by
  construction.
- **Splits validation into trajectory-invariance (Tier 2a) and
  empirical-attribution-frequency (Tier 2b)** tests, the latter being
  the actually load-bearing correctness check. The previous draft
  presented trajectory-invariance as if it tested attribution; it
  doesn't.
- Names tau-leap's `dt`-bias explicitly with a diagnostic.
- Pins Parquet as the production output format.
- Namespaces CLI projections under `camdl lineage`.

**On implementation cost.** This proposal is more ambitious than v1,
not comparable: identity-tracked-subgraph inference,
linear-decomposition analysis with explicit error messages, multiple
offline projections, attribute-syntax parser changes, Parquet writer,
empirical-attribution-frequency testing, and the cyclic-model cost
reporting all add real surface area. The additional capability — line
list as primary artifact, derived-not-chosen attribution, multiple
projections — justifies the larger investment.
