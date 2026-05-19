# Proposal: `simulate_with_lineages` — transmission line lists and trees

Date: 2026-05-19
Status: Draft (RFC)
Branch: `feature/lineages`
Scope: forward simulation only. Orthogonal to the inference stack —
nothing in `pgas.rs`/`if2.rs`/`particle_filter.rs` changes.

## Problem

camdl's backends track compartment counts, not individuals. A
transmission event decrements a susceptible count and increments an
infected count; *which* infectious individual was the infector is
never recorded, so a transmission tree is not a byproduct of
count-level simulation.

For inference we do not need it. For **benchmark / synthetic-data
generation** we do: a known true transmission tree is required to
validate phylodynamic inference, to test tree-based estimators, and
to study where epidemic structure and genealogy interact. This
proposal adds a `simulate_with_lineages` mode that runs the existing
model unchanged and additionally maintains an identity layer.

## Key reframing: the artifact is a perfect-ascertainment line list

The intermediate object the simulation emits is best understood as a
**transmission line list under perfect ascertainment**: one record
per case with infection time, deme/stratum, and — the field a
real-world line list almost never has — the *true* infector. A real
line list has `infected_by` mostly empty or contact-traced; ours has
it exactly, for every case.

The transmission **tree** is that line list rendered as a genealogy
and then projected through an observation/sampling model (pruning to
sampled tips, suppressing unary nodes). The tree is a *view*; the
line list is the primary artifact.

Three consequences used throughout this proposal:

1. The line list is event-indexed (append-only log); the classic
   individual-indexed line list is a trivial pivot of it.
2. A log *without* resolved parents is exactly a line list *missing
   the `infected_by` column* — useful (who/when/where) but not a
   tree, and the genealogy cannot be recovered post hoc.
3. Environmental infections (e.g. SIWR `S·W`) appear exactly as a
   real line list encodes an unknown/environmental-source case:
   `infected_by = environment`, no individual parent.

## What must be declared vs inferred

Two facts are needed; they have very different inferability.

**Inferred — per-infector-class contribution weights.** Standard
forces of infection are *additive over infector classes*: `β·S·I/N`,
`S·Σ_b C[a,b]·I[b]/N[b]`, `S·(β_I·I + β_A·A)/N`. The transmission
rate expression *is* the contribution structure. At an event, the
probability the infector is class *k* is class *k*'s share of the
**evaluated** force of infection. Nothing about the weights is
declared — they are decomposed from the rate AST evaluated at event
time (current counts, resolved `TableLookup` contact matrix, resolved
`TimeFunc` forcing, `Cond` guards applied).

**Declared — the transition's role and its infector pool.** Two
inference failures make this non-negotiable:

- *Role ambiguity.* `ν·S` (vaccination, S leaves, no parent),
  `ξ·R` (waning), and `β·S·I/N` (infection) are structurally
  similar — a compartment decreasing at some rate. AST shape cannot
  distinguish "this S→V is vaccination" from "this S→E is
  transmission." The heuristic "rate mentions an infectious
  compartment ⇒ transmission" presupposes knowing which compartments
  are infectious — exactly the model knowledge the count-level IR
  never needed.
- *Infector-vs-normalizer ambiguity.* In `β·S·I/N` with
  `N = S+E+I+R`, the symbol `I` appears as the infector *and* inside
  the `N` normalizer (`PopSum`). Structurally both are references to
  `I`. One declared token (`infector = I`) disambiguates; inferring
  it means reconstructing epidemiological intent from `N`'s
  definition — brittle and silently wrong when wrong.

**Resolution: declare role + infector pool (one light annotation per
transmission); infer the weights.** This keeps the declaration burden
to ~one annotation per transmission transition and makes the
quantitatively delicate part (structured, time-varying weighting) a
derived quantity rather than a hand-maintained one.

## Type design

New `lineage` module in the `sim` crate.

```rust
struct IndividualId(u64);                 // monotone per-run counter

struct TransmissionEvent {
    parent:      Option<IndividualId>,    // None = seed / import / environmental
    child:       IndividualId,
    time:        f64,
    transition:  TransitionId,
    child_deme:  DemeId,
    parent_deme: Option<DemeId>,          // infector stratum (cross-patch trees)
}

// Append-only; streamed, never held whole in RAM (see "Online vs offline").
struct TransmissionLog { /* writer handle, not a Vec in the hot path */ }

// Per (deme, infectious-compartment) live identity pools.
// swap-remove Vec / slab: O(1) amortized insert and remove-by-index.
struct IdentityState {
    pools: HashMap<(DemeId, CompartmentId), Vec<IndividualId>>,
    next:  u64,
}
```

The seam that makes "dynamics unchanged" structural, not asserted:

```rust
trait TransitionObserver {
    // Called by the core loop AFTER it has drawn its own RNG and
    // decided how many of each transition fired. Cannot reorder the
    // simulation RNG.
    fn on_fired(&mut self,
                transition:   TransitionId,
                deme:         DemeId,
                multiplicity: u64,            // k firings (tau-leap / chain-binomial)
                pre_state:    &State,         // counts before applying — the sampling pool
                rate_terms:   &EvaluatedFoi,  // per-infector-class evaluated weights
                rng:          &mut LineageRng);
}
```

The core simulator takes `Option<&mut dyn TransitionObserver>`.
`None` → today's behavior, byte-for-byte, zero overhead. `Some(...)`
→ identity bookkeeping runs strictly downstream of the dynamics.

Pure, simulation-independent output seam (offline, unit-testable):

```rust
fn prune_to_samples(log: &Path, scheme: &SamplingScheme) -> Tree;
fn write_newick(tree: &Tree, w: &mut impl Write);
```

## RNG: a separate stream is an invariant, not an optimization

`CLAUDE.md` is explicit that paired-seed CRN holds *only while the
RNG is consumed in the same order on both sides*. If parent sampling
draws from the simulation's ChaCha8 stream, a run *with* lineages
diverges from the same run *without* it, breaking determinism and
every paired-scenario guarantee.

Therefore the identity layer owns an **independent RNG stream**,
seeded `main_seed ⊕ fixed_offset` (reproducible, disjoint). This
makes the central guarantee architectural: trajectories are
byte-identical with and without `--lineages`. This must be an
explicit, tested invariant (a golden trajectory run both ways and
diffed).

## Backend matrix

| backend         | lineage support | parent attribution |
|-----------------|-----------------|--------------------|
| Gillespie (SSA) | exact           | one firing = one infectee; decompose the evaluated rate at the exact event time |
| tau-leap        | approximate     | *k* firings vs frozen start-of-step rates; parents drawn from the start-of-step pool — matches tau-leap's own propensity-freezing |
| chain-binomial  | approximate     | as tau-leap; inherits chain-binomial discretization error (note: this is the fitting workhorse) |
| ODE             | **incompatible**| no individuals — hard error |

ODE rejection reuses the existing `Capabilities` mechanism: add
`Capabilities::LINEAGES`; ODE does not declare it;
`required_capabilities()` mismatch → error before simulation starts,
identical machinery to `OVERDISPERSION`. Documentation must state
that trustworthy benchmark trees want Gillespie (or small `dt`); the
tau-leap / chain-binomial trees are only as accurate as the
backend's propensity approximation.

## Online vs offline split

Parent attribution **must be online**: it needs the live infectious
pool *and* the evaluated-rate decomposition *at the event instant*
(contact matrix, seasonal forcing, current counts all enter the
weights). It cannot be reconstructed from counts post hoc.

Everything after is **offline**:

- *Online (during simulation):* mint IDs; resolve parents by
  evaluated-FOI decomposition; **stream** each event to an
  append-only writer; never hold the full log in RAM (a large
  epidemic is millions of events).
- *Offline (`camdl` post-process over the file):* apply a sampling
  scheme, postorder-prune to sampled tips, suppress unary nodes,
  emit Newick. Pure function of `(log file, sampling scheme)` — no
  simulation dependency, independently testable, and re-runnable:
  one simulation log → many sampled trees under different
  observation models without re-simulating.

This matches the project's content-addressable ethos: the log is a
deterministic function of `(model, params, seed)`; the tree is a
deterministic function of `(log, sampling scheme)` — two cache keys,
one expensive step.

## DSL annotation

Light, matching camdl's existing transition syntax and
function-style annotations (`overdispersed(...)`,
`unchecked_dim(...)`):

```camdl
transitions {
  # direct: infector pool is I; weights inferred from the rate
  infection : S --> E  @ beta * S * I / N        transmission(infector = I)

  # multiple infectious classes; per-class weight from the rate
  infection : S --> E  @ S * (beta_I*I + beta_A*A) / N
                                                 transmission(infector = [I, A])

  # stratified: infector compartment is I; the C_age[a,b] weighting
  # is read structurally from the rate, NOT restated here
  infection[a in age] : S[a] --> E[a]
      @ S[a] * sum(b in age, C_age[a,b] * I[b] / N[b])
                                                 transmission(infector = I)

  # environmental: parent is a reservoir, not an individual
  infection : S --> I  @ S * W * beta_w
                                  transmission(source = environmental, via = W)

  # non-transmission: unannotated. Identity still shadows through
  # these (E->I moves an ID into the infector pool; I->R removes it)
  progression : E --> I  @ sigma * E
  recovery    : I --> R  @ gamma * I
  vaccination : S --> V  @ nu * S
}
```

The annotation declares only: *this is a transmission*, the *infector
pool*, and for the indirect case that it is *environmental and
through which compartment*. All quantitative splitting is derived
from the evaluated rate.

## IR / schema impact

One optional field on the transition record:

```
transmission: { infector: [CompId], source: direct|environmental,
                via: Option<CompId> } | null
```

Additive and optional. Atomic OCaml + Rust + golden-file update per
`CLAUDE.md`'s "Changing the IR schema" procedure. A model with no
`transmission(...)` annotation simply cannot run in `--lineages`
mode (clean capability error) — zero cost for every existing model.
This is the only part of the change that touches the IR contract;
everything else is additive `sim`/`io`/`cli` code.

## Validation strategy (the load-bearing work)

The code is moderate; *proving the trees are correct* is where the
effort and the scientific risk are. The decomposition must be tested
on the **evaluated** rate, not the symbolic structure with nominal
values — a test that validates symbolic splitting under constant
weights would pass while being silently wrong under seasonality or
contact structure. Tiers, increasing strength:

1. **Structural invariants.** Every non-seed child has exactly one
   parent; the parent is in an infectious pool at the child's
   infection time; pruned tips = sampled set; no unary nodes;
   trajectory byte-identical with/without `--lineages`.
2. **Analytic.** Linear birth–death / Yule tree statistics (Sackin
   imbalance, expected TMRCA, branch-length sums) have closed forms
   — assert against them.
3. **Large-N limit.** Under homogeneous mixing the SIR transmission
   tree converges to the structured-coalescent prediction (Volz
   2009 / Rasmussen–Volz line — *cite-check before publication;
   lineage flagged, not a section asserted*).
4. **External oracle.** Cross-validate a *stratified* scenario
   against an independent lineage-aware simulator (MASTER, nosoi,
   VGsim, or TiPS — versions to be pinned), run as a CI gate, same
   pattern as the existing pomp / scipy / numpy oracle tests.

Tiers 3–4 on the **stratified** case are the real deliverable:
anyone can get the well-mixed tree right; the value (and the risk of
a silent wrong answer feeding a benchmark) is the
contact-structured, time-varying parent attribution.

## Phasing

- **Phase 1.** Observer seam + separate RNG stream + Gillespie +
  single-population + streamed log + offline prune + Newick/TSV +
  structural & analytic tests. Proves the architecture and the
  byte-identical-trajectory guarantee.
- **Phase 2.** Stratified / spatial parent attribution (evaluated-FOI
  decomposition) + external-oracle CI gate. The scientifically hard
  part, deliberately isolated so its risk does not contaminate
  Phase 1.
- **Phase 3.** tau-leap / chain-binomial (approximate, documented) +
  environmental-source semantics (SIWR pseudo-nodes).

## Open design questions

1. **Environmental transmission interacts with offline pruning.**
   Environmental "parents" are not individuals and cannot be
   sampled tips. Options: a single shared environmental pseudo-root
   per deme; per-event environmental sentinels; or treat
   environmental edges as polytomy attachments to the local
   infected set. This needs resolving before Phase 3 and affects
   the Newick schema (environmental nodes have no tip semantics).
2. **Sampling model location.** Is the benchmark target purely
   phylodynamic-inference validation (true tree + tip-sampling +
   sequence-free Newick), or will a sequence-generating layer
   (mutations down the tree) come later? The latter means
   `SamplingScheme` needs a per-deme, time-varying sampling-rate
   model now rather than a flat tip set.
3. **Infer-vs-declare escape hatch.** Should the common pure
   mass-action case be inferable to reduce annotation burden, with
   declaration as the override? Current lean: always declare —
   explicit beats a clever guess that is silently wrong for
   SIWR / vaccination / reinfection.

## Non-goals

- No change to inference, fitting, or any existing backend dynamics.
- No within-host or sequence evolution in v1 (see open question 2).
- ODE lineage support (incoherent — hard error by capability).
