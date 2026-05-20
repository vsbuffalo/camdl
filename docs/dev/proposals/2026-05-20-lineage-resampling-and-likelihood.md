# Proposal: three-layer lineage architecture — resampling, tree likelihood, and the inference boundary

Date: 2026-05-20
Status: Draft (RFC) — for review before implementation
Branch: `feature/lineages` (refactor of the shipped two-layer design)
Supersedes the *architecture* of `2026-05-19-individual-sampling-layer.md`;
that document's implementation (count-level lineage tracking, stratified
attribution, three backends, projections, validation) is the foundation this
refactors, not work to discard.

> Citations verified against primary sources (2026-05-20); volume/page pinned
> inline.

---

## 1. The insight: a line list is a conditional sample, not a realization

The shipped design produces one line list per simulation and treats it as *the*
genealogy. That is a category error. The augmented process factorizes:

$$P(\text{augmented trajectory}) = P(\text{count trajectory}) \times P(\text{identity attribution} \mid \text{count trajectory}).$$

The compartmental simulation draws the **first** factor: it fixes the ordered
event sequence — at $t_1$ a transmission fired, at $t_2$ a recovery fired, … —
and the counts evolve deterministically given those events. It does **not** fix
the second factor. Given that event sequence, *which specific individual* was
the infector at each transmission, and *which* individual underwent each
recovery, are a separate stochastic layer: many identity attributions are
equally consistent with the same count trajectory.

So a single compartmental run defines a **distribution over line lists**, not
one line list. For benchmark generation this is decisive: the phylodynamic
method under validation (MASCOT, BASTA, BDMM) assumes the observed tree is *one
draw* from $P(\text{tree} \mid \text{trajectory})$. Validating it requires the
*ensemble* of trees consistent with an epidemic, not a single tree.

The shipped code already half-embodies this: identity randomness is drawn from
a **separate RNG stream** from the count dynamics (the verified byte-identity
invariant). That separation is the empirical proof of the factorization. This
refactor makes the factorization *structural*: persist the first factor once,
resample the second cheaply.

---

## 2. Three layers, three artifacts, three cache keys

| Layer | Map | Cost | Cache key |
|---|---|---|---|
| **1. Event log** | $(\text{model}, \text{params}, \text{seed}) \to$ event log | expensive (one epidemic) | `f(model, params, dynamics_seed)` |
| **2. Line list** | $(\text{event log}, \text{identity\_seed}) \to$ line list | cheap (replay) | `f(event_log, identity_seed)` |
| **3. Tree** | $(\text{line list}, \text{scheme}, \text{sample\_seed}) \to$ tree | cheap (prune) | `f(line_list, scheme, sample_seed)` |

One expensive epidemic → many cheap identity realizations → many cheap trees. A
Monte Carlo benchmark sweeps all three independently (`dynamics_seed`,
`identity_seed`, `sample_seed`). Caching means tweaking the sampling scheme or
the identity seed never re-runs the epidemic.

The count **dynamics** become lineage-free: the simulation draws no identities.
(The event *recorder* still evaluates the per-pool weights at lineage events —
the rate and state are available there — but it records them rather than
sampling a parent.) The byte-identity invariant becomes trivially true: the
simulation is literally unchanged; `--event-log` records what it already
computes.

### CLI

```bash
camdl simulate model.camdl --params p.toml --seed 42 --event-log epi.parquet
camdl lineage realize epi.parquet --identity-seed 7 -o line_list.parquet
camdl lineage tree    line_list.parquet --scheme stratified --sample-seed 3 -o tree.nwk

# fused (common case) + replicate sweeps:
camdl lineage tree epi.parquet --identity-seed 1:100 --scheme stratified --sample-seed 1:10 -o trees/
```

"One line list" is just "realize once"; the event log is the canonical source.
No single-realization shortcut that tempts treating a line list as canonical.

---

## 3. The event log

Minimal sufficient statistic for resampling: **initial state + ordered
$(\text{time}, \text{transition\_id})$**, plus `multiplicity` for batched
(tau-leap / chain-binomial) steps. From that, the model, and the parameters,
one can replay the counts and recompute the per-pool weights at every lineage
event.

We additionally record the **evaluated per-pool FOI weights**
$\{(b, w_b X_b)\}$ at each `#[lineage]` event, so the event log is
**self-contained**: Layer-2 replay needs only the event log, not the model or
the rate AST. Size cost is modest (weights only at lineage events). The event
log is **identity-free** — IDs are minted during replay; the log is the
epidemic, not the genealogy. It is finer-grained than the existing `flows`
output (cumulative counts at output times): the event log is the raw event
sequence. New output type, gated by `--event-log`.

---

## 4. Likelihoods — what is and is not a clean product

The recorded weights make the **line-list** likelihood fall out of the Layer-2
replay for free. The **tree** likelihoods do not; this section states precisely
which is which, because the boundary is a correctness trap.

Notation: at a transmission event the parent pools have per-individual weight
coefficients $w_b$ and sizes $X_b$; the realized FOI total is
$\Lambda = \sum_{b'} w_{b'} X_{b'}$.

### 4a. Line-list likelihood — exact, cheap, a clean product

The line list specifies **every** attribution (transmission parents *and*
recovery/progression identities). Given the event log, the attributions are
conditionally independent across events (Markov), so

$$\log P(\text{line list} \mid \text{event log}) = \sum_{\text{events}} \log P(\text{attribution at that event}).$$

Per event:
- **Transmission**, parent = individual $i$ in pool $b$:
  $P = \frac{w_b X_b}{\Lambda} \cdot \frac{1}{X_b} = \frac{w_b}{\Lambda}$
  (pool choice × within-pool uniform; the $X_b$ cancels).
- **Recovery / removal**: uniform within the **relevant pool** — $1/|I_b|$ for
  the deme/stratum $b$ the removal fires in ($1/|I|$ in the unstratified case).

A product of per-event terms, accumulated during replay. Each resampled line
list carries its own log-probability. **This is the only clean exact likelihood
the architecture provides.**

### 4b. There is no separate cheap "full tree" likelihood

A natural-looking but **incorrect** claim is that the full labeled tree
likelihood is the product over transmission events of the parent-assignment
probabilities, with recovery identities "summing to 1." This is wrong: recovery
attributions are **not** independent of the tree. They determine *which
specific individuals remain in the infectious pool*, which constrains who can be
a parent at later transmission events.

**Counterexample.** SIR. Event log: (t₁) transmission, pool `{A}` (seed),
mints child `B`, pool → `{A,B}`; (t₂) recovery, one of `{A,B}` leaves, each
w.p. ½; (t₃) transmission, pool size now 1, mints child `C`. Two labeled trees
are possible: `{A→B, A→C}` and `{A→B, B→C}`, each with true probability ½ (the
½ comes from *which* individual recovered at t₂). The naive product gives, at
t₁ `P=1` (pool size 1) and at t₃ `w_b/Λ = 1/1 = 1` (recorded count `I=1`), so
`1·1 = 1` for **both** trees — summing to 2, not 1. The per-event term
`1/pool_size` conditions on pool membership, and pool membership is set by the
recovery attribution the naive formula drops.

**Consequence:** the only clean product is the line list (4a). Any tree
likelihood — full or sampled — requires marginalizing over the
non-tree-determining attributions (recovery/progression identities), which
couple to the transmission structure through pool membership and is therefore
**not** a per-event product. Do not implement a `--full-tree` likelihood as a
transmission-event product; it returns plausible-looking numbers that are
systematically wrong and fail normalization. (Keep this counterexample as a
code comment / test so the trap stays documented.)

### 4c. Sampled-tree likelihood — intractable in general

A sampled tree (tips = a subset of individuals) requires marginalizing over the
genealogical placement of all unsampled individuals:

$$P(\text{sampled tree} \mid \text{event log}) = \sum_{\substack{\text{line lists consistent}\\\text{with the sampled tree}}} P(\text{line list} \mid \text{event log}).$$

This sum is combinatorial. **The structured coalescent is exactly the analytic
approximation to this quantity in the large-$N$ diffusion limit** (Volz et al.
2009, *Genetics* 183(4):1421; structured-coalescent approximation theory:
Müller, Rasmussen & Stadler 2017, *MBE* 34(11):2970; its marginal
implementation MASCOT: Müller, Rasmussen & Stadler 2018, *Bioinformatics*
34(22):3843). A forward Monte Carlo
estimator ("resample line lists, count the fraction consistent with the sampled
tree") is exact in expectation but has **catastrophic variance**: the consistent
set is an exponentially small fraction of identity realizations, so the hit rate
is ≈ 0 for any non-trivial tree. This is precisely why phylodynamics uses
analytic approximations rather than forward MC.

**Honest scope:** the forward-MC sampled-tree likelihood is tractable only for
**small trees** (few tips — direct enumeration or importance-sampled MC). In
that regime it is a genuinely novel tool: an *exact reference* against which to
measure where the coalescent approximation deviates, for a *specific dynamical
regime*. It does **not** provide an exact likelihood for arbitrary large trees.

---

## 5. The type architecture

This is the part to get exactly right; it is the "deep architecture" the rest
of the system hangs on. Types are Rust (the simulation/inference backend).

### 5.1 Forward path — types in sequence

```rust
// ── Layer 1: epidemic realization. (simulate --event-log) ──────────────────
struct EventLog {
    initial_state: State,
    events: Vec<EventRecord>,
}
struct EventRecord {
    time: f64,
    transition: TransitionId,
    multiplicity: u64,                               // batched backends; 1 for Gillespie
    lineage_weights: Option<Vec<(CompartmentId, f64)>>,  // {(b, w_b·X_b)} at #[lineage] events
}

// ── Layer 2: identity realization. (lineage realize, + identity_seed) ───────
struct LineListEntry {
    time: f64,
    transition: TransitionId,
    individual: IndividualId,
    source: Option<CompartmentId>,
    destination: Option<CompartmentId>,
    deme: DemeId,
    parent: ParentRef,
    attribution_logprob: f64,                        // §4a accumulation, per event
}
enum ParentRef {
    Individual(IndividualId),
    Import,
    Seed,
    None,                                            // non-lineage event
    // Environment(CompartmentId) reserved for the environmental phase
}

// derived view: the FULL transmission forest (who infected whom; all individuals)
struct TransmissionForest {
    nodes: BTreeMap<IndividualId, TransmissionNode>,
    roots: Vec<IndividualId>,                        // seeds / imports
}
struct TransmissionNode {
    individual: IndividualId,
    parent: Option<IndividualId>,
    infection_time: f64,
    removal_time: Option<f64>,
    deme_path: SmallVec<[(f64, DemeId); 1]>,         // deme over time; len 1 unless host mobility
    children: Vec<IndividualId>,
}
```

### 5.2 The shared boundary type: `Tree`

`Tree` is the observable, sampled, time-calibrated, tip-labeled tree. It is the
**output of Layer 3** of the forward path **and the input to the inference
path**. Same type both directions — that closes the validation loop without a
serialization round-trip. For real data, `Tree` is parsed from time-calibrated
Newick + a tip-metadata sidecar (dates, demes).

```rust
struct Tree {
    topology: TreeTopology,                  // nodes, edges, node times, branch lengths
    tip_demes: BTreeMap<NodeId, DemeId>,     // sampling location — OBSERVED, always known
    tip_times: BTreeMap<NodeId, f64>,        // sampling time (pendant tips) — OBSERVED
    tip_labels: BTreeMap<NodeId, String>,    // opaque labels (accession / individual id); inference ignores
}

impl Tree {
    fn summaries(&self) -> TreeSummaries;          // Sackin, Colless, LTT features, γ, offspring k, ...
    fn coalescent_timeline(&self) -> CoalescentTimeline;  // preprocess for the PF (§7)
    fn to_newick(&self, w: &mut impl Write);
    fn from_newick(src: &str, meta: &TipMetadata) -> Result<Tree>;
}
```

**INVARIANT (the no-cheating guarantee).** `Tree` carries only quantities an
observer can see: topology, branch lengths/times, tip demes, tip times, opaque
labels. It carries **no** internal-lineage demes, **no** individual back-map,
**no** true infection times. The inference path takes `&Tree` (or `TreeData`,
§7) and therefore *structurally cannot* read ground truth. This is enforced by
the type, not by reviewer discipline.

### 5.3 The synthetic tree: `Tree` + ground truth (composition)

```rust
struct SyntheticTree {
    tree: Tree,                                       // ← the observable part, embedded
    true_lineage_demes: BTreeMap<EdgeId, DemePath>,   // deme trajectory per branch (latent in reality)
    individual_map: BTreeMap<NodeId, IndividualId>,   // back-ref to the line list
    true_infection_times: BTreeMap<NodeId, f64>,      // distinct from sampling times
}
impl SyntheticTree {
    fn observe(&self) -> &Tree { &self.tree }         // the projection: a borrow, not a rebuild
}
```

`SyntheticTree` **contains** `Tree`. The projection `observe()` is a field
borrow — no second traversal, no conversion cost. All traversal/summary/Newick
code lives once, on `Tree`, and `SyntheticTree` reuses it through `.observe()`.

**Why composition, not optional truth fields on one type, and not a newtype
around a truth-bearing tree:** putting `Option<true_deme>` on a single `Tree`
encodes a sum type (synthetic | observed) as a product type with optionals
("always Some here, always None there") — a code smell — and worse, leaves the
truth *reachable from inside inference code*, so the no-cheating guarantee
degrades to discipline. A newtype around a `Tree` that *has* truth fields has
the same defect (truth reachable through the wrapper). Composition puts truth
**only** on the wrapper and keeps it **absent** from the embedded `Tree`, so the
guarantee is structural. Direction matters: wrap-and-add-truth is safe;
wrap-a-truth-bearing-tree is not.

**Three tree-ish types, one morphism chain:**
`TransmissionForest` (full, all individuals, many roots) → *sample + prune +
pendant tips* → `Tree` (sampled, observable). `SyntheticTree` = `Tree` + the
truth recorded during that morphism. The sample-and-prune step is the explicit
observation model.

---

## 6. The forward → inference data flow

```
FORWARD (generative, exact):
  (model, params, dynamics_seed)
    │ simulate --event-log
    ▼  EventLog
    │ lineage realize  (+ identity_seed)         → attribution_logprob accumulates §4a
    ▼  LineList ──derive──► TransmissionForest
    │ lineage tree     (+ scheme, sample_seed)   → sample + prune + pendant tips,
    ▼  SyntheticTree                                recording truth alongside
    │ .observe()
    ▼  Tree   ──────────────────────────────────  ← SHARED INTERFACE
    │ [FUTURE] mutations (+ substitution model)  → sequence evolution / within-host coalescent
    ▼  Phylogeny  (= Tree under transmission-tree≈phylogeny assumption)

INFERENCE (scoring, approximate):
     Tree  (synthetic via .observe(), OR real data via from_newick)
    │ wrap as TreeData::Fixed
    ▼  CoalescentTimeline   (preprocess once)
    │ in PGAS/PMMH: each particle carries trajectory {I_k(t), f_k(t)} + latent lineage-deme state
    │ per interval: coalescent_loglik(interval | trajectory, lineage_demes)
    ▼  += particle weight  (alongside case-data loglik, sero loglik, …)
    ▼  posterior over params

VALIDATION (closing the loop):
     SyntheticTree.observe() → inference → posterior / reconstructed lineage demes
     compare reconstructed lineage demes against SyntheticTree.true_lineage_demes
     compare recovered params against the dynamics_seed's known params
```

The inference reads the trajectory off the **particle**, not off the event log
or line list — it does not replay identities. So the forward three-layer
machinery and the inference path are cleanly decoupled; they meet only at
`Tree`. Adding the coalescent observation channel later touches none of the
forward code.

---

## 7. Inference path (reserved native future work)

This proposal does not implement inference, but it must not architecturally
block it. The native joint-tree-inference path is the **coalescent / birth-death
tree likelihood as a particle-filter observation channel**, using the existing
PGAS/PMMH stack — the Rasmussen–Volz–Koelle approach.

The types and the per-interval formula below are **illustrative of what the
`Tree` boundary must permit — not settled inference design.** The
marginalize-vs-sample fork (open question 4) and the channel's exact shape are
deferred; the per-interval expression shows only the *coalescent* term — the
migration / structured contribution to the structured-coalescent likelihood is
omitted for brevity. Do not read §7 as committed; the load-bearing commitment
is the `Tree` boundary (§5.2), not these internal types.

```rust
enum TreeData {
    Fixed(Tree),
    // reserved: Posterior(Vec<Tree>) / Sample(TreePosterior) for tree uncertainty
}

struct CoalescentTimeline { intervals: Vec<CoalescentInterval> }
struct CoalescentInterval {
    t_start: f64,                            // backward time
    t_end: f64,
    extant: Vec<LineageId>,
    boundary: CoalescentEvent,
}
enum CoalescentEvent {
    Coalescence { a: LineageId, b: LineageId, parent: LineageId, deme: Option<DemeId> },
    Sampling    { lineage: LineageId, deme: DemeId },   // tip enters going backward
}
```

The structured-coalescent likelihood per interval, given a particle's trajectory
$\{I_k(t), f_k(t)\}$ per deme $k$ (force of infection $f_k = \beta_k S_k I_k / N_k$,
pairwise coalescence rate $\lambda_k = 2 f_k / I_k^2$, $k_k$ lineages currently
in deme $k$):

$$\log P(\text{interval}) = -\int_{t_{\text{start}}}^{t_{\text{end}}} \sum_k \binom{k_k}{2}\, \lambda_k(t)\, dt \;+\; \mathbb{1}[\text{coalescence in deme } k]\,\log \lambda_k(t_{\text{end}}).$$

Lineage demes are latent → **marginalize** (MASCOT-style ODEs on per-lineage
deme probabilities) or **sample** (PGAS lineage-deme paths as additional
particle state). The marginalize-vs-sample fork is itself a future decision;
the architecture must permit both. The contribution is *additive* in the
particle log-weight, alongside the case-data likelihood, which is exactly how
joint case + tree inference composes.

**Scope of the first inference cut (when built):** condition on a fixed
point-estimate phylogeny (`TreeData::Fixed`), not integrating over phylogenetic
uncertainty. `TreeData` is an enum so the `Posterior` variant can be added later
with no rework of the fixed-tree path.

**Decision (was open question):** the coalescent/BD-likelihood-in-PF inference
path is **native** camdl work (extends the existing inference stack). General
likelihood-free inference engines (synthetic likelihood, BSL, ABC; §8) are
**external** downstream consumers — but `Tree::summaries()` lives on the shared
type so a *future* native SL consumer is not blocked either.

---

## 8. Summary-statistic synthetic likelihood (tractable path; external fit)

For trees beyond the small-tree exact regime, the tractable route to a
likelihood is a composite / synthetic likelihood over tree summary statistics.
Pick $S(\text{tree}) \in \mathbb{R}^d$ via `Tree::summaries()`. At parameter
$\theta$, the cheap forward ensemble (one event log → many identity draws → many
sampled trees) yields $\{S^{(1)}, \dots, S^{(M)}\}$. Under the synthetic-
likelihood normality assumption (Wood 2010, *Nature* 466(7310):1102):

$$\hat\mu(\theta) = \tfrac{1}{M}\sum_m S^{(m)},\quad \hat\Sigma(\theta) = \widehat{\mathrm{Cov}}(S),\quad \ell_{\mathrm{SL}}(\theta; S_{\mathrm{obs}}) = \log \mathcal{N}\!\big(S_{\mathrm{obs}};\, \hat\mu(\theta),\, \hat\Sigma(\theta)\big).$$

BSL (Price, Drovandi, Lee & Nott 2018, *JCGS* 27(1):1) is the Bayesian variant;
ABC the simulation-only alternative.

**Scope decision:** camdl **emits the ensemble and computes the summaries**
(`Tree::summaries()`); the SL/BSL/ABC **fit** is a downstream consumer
(notebook), *for now*. Because `summaries()` is a method on the shared `Tree`
type, promoting the SL fit to native later requires no retrofit — only a new
internal consumer. This keeps the "orthogonal to the inference stack" claim
honest while reserving the room you flagged.

**Candidate summaries** (selection is the central open scientific choice, §11):
- **Offspring dispersion** — NB dispersion $k$ of the per-infector offspring
  distribution (Lloyd-Smith et al. 2005, *Nature* 438:355); *already partly
  computed* by the shipped offspring check.
- **Imbalance** — Sackin / Colless.
- **Temporal** — LTT-curve features; the $\gamma$ statistic (Pybus & Harvey
  2000, *Proc. R. Soc. B* 267:2267).
- **Structured** — tip-stratum proportions; cross-stratum transition counts.

**Normality caveat:** several of these (Sackin especially) are strongly skewed;
the multivariate-normal SL assumption can be poor. Summary choice must consider
the normality assumption or use a transform, not just informativeness.

---

## 9. Reserved future layers (do not block; do not build now)

1. **Sequence evolution** (`lineage { mutations { … } }`): mutations down tree
   branches, optionally a within-host coalescent pushing the MRCA before
   transmission. This is a layer **above** `Tree` producing `Phylogeny`; it does
   **not** alter `Tree`. Under the transmission-tree≈phylogeny assumption (fine
   for low within-host diversity, e.g. polio) `Phylogeny = Tree` and inference
   takes `Tree` directly. State the assumption as one explicit, swappable layer
   so it never becomes buried debt.
2. **Tree uncertainty / joint tree inference**: `TreeData::Posterior(Vec<Tree>)`
   (BEAST-style posterior sample), consumed by iterating the PF observation
   channel over the variant. The enum reserves this; the fixed-tree path is
   unchanged.
3. **Native synthetic-likelihood fit**: a future internal consumer of
   `Tree::summaries()`. Reserved by putting `summaries()` on the shared type.
4. **Environmental / nonlinear-parent** rates: unchanged deferrals from the
   prior proposal (tagged-contribution semantics; `infector(...)` wrapper).

---

## 10. Refactor + validation plan (on `feature/lineages`, reuse-heavy)

Reused unchanged: the IR schema (`parent_pool_weights`,
`identity_tracked_compartments`), the attribution math (incl. stratified
contact-weighting, Tier-2b validated), the three backends, the projections
(`tree`/`sojourn`/`cohort`), the corrected coalescent diagnostic.

Changes:
1. **Event-log writer** (`--event-log`): the current observer becomes an *event
   recorder* — evaluates per-pool weights at lineage events but **records** them
   rather than sampling a parent. Count dynamics no longer draw identities.
2. **`camdl lineage realize`** (Layer 2): event log → line list, with
   `--identity-seed`; samples pool-then-individual from recorded weights;
   accumulates the line-list log-likelihood (§4a).
3. **`camdl lineage tree`** consumes a line list **or** an event log (fusing
   realize + sample); produces `SyntheticTree`; serializes `.observe()` to
   Newick. `--keep-truth` optionally writes the truth sidecar for validation.
4. **`camdl lineage loglik`**: line-list log-likelihood (§4a, exact, general).
   **Do not** expose a full-tree product (§4b). Sampled-tree estimator (§4c)
   only for small trees, with the tip-count bound enforced and an explicit
   variance warning.
5. **Provenance**: three hashes (§2); scheme/identity/sample choices flow into
   the line-list and tree hashes, never the model hash.
6. **Type split**: introduce `SyntheticTree` (composition over the existing
   `Tree`); hold the §5.2 invariant (no truth on `Tree`).

The attribution-sampling code moves from the inline observer to the `realize`
replay; the logic is identical, only its location changes.

### Validation tiers

- **Tier 1 — Structural invariants.** Single parent per lineage child; parent in
  pool at child's event time; pruned tips = sampled set; no unary nodes;
  pendant tips at sampling time.
- **Tier 2a — Trajectory invariance.** Count trajectory with/without
  `--event-log` byte-identical for the same seed. Catches RNG-stream leakage;
  does **not** test attribution.
- **Tier 2b — Empirical attribution frequencies.** Over many identity
  realizations from one event log, observed parent-class frequencies match
  $w_b X_b / \Lambda$ within 3σ MC error. The load-bearing attribution test.
- **Tier 3 — Analytic.** Yule / linear-BD tree statistics against closed forms.
- **Tier 4 — Large-N coalescent limit.** Coalescent-interval distribution at $t$
  matches $\mathrm{Exp}\!\big(\binom{k}{2}\,\lambda(t)\big)$ with the per-pair
  coalescent rate $\lambda = 2f/I^2 = 2\beta S(t)/(N(t)\,I(t))$ **as defined once
  in §7** (single source of truth — *not* $2\beta S I/N^2$, a stale form off by
  $I^2/N$ that earlier drafts carried; cf. correction in `5d8e2c0`), within 2σ
  over 10⁴ replicates, **for $N \ge 10^4$** (O(1/N) diffusion bias makes smaller
  N flaky).
- **Tier 5 — External oracle (validates the simulator).** Cross-validate a
  stratified scenario against an independent **exact-forward** lineage-aware
  simulator. **MASTER** (Vaughan & Drummond 2013, *MBE* 30(6):1480; exact
  Gillespie for compartmental models — VGsim itself validates against it) is
  the primary, best-matched oracle: you specify the exact stratified reactions,
  so the contact-matrix semantic match is provable. **VGsim** (Shchur et al.
  2022) is also exact-forward — it runs an exact (hierarchical-Gillespie) event
  chain and samples the genealogy *backward conditioned on that realized chain*
  (exact-conditional, **not** the structured-coalescent diffusion approximation,
  which replaces the stochastic trajectory with a deterministic ODE — something
  VGsim does not do) — so it too tests the *forward model*, keeping Tier 5
  distinct from Tier 6. VGsim's advantage is scale (millions of tips); its
  limitation for this check is that its migration-based population structure may
  not express camdl's arbitrary asymmetric contact matrix cleanly. Gated behind
  realistic sampling (leaf-only Flat trees are not comparable to all-case
  samplers). See the oracle-landscape survey in
  `2026-05-19-individual-sampling-layer.md`.
- **Tier 6 — Forward reference vs analytic approximation (validates the
  approximation).** On small trees (4c exact regime) or via summaries (§8),
  compare the forward reference against the structured-coalescent analytic
  likelihood (MASCOT). Reports regime-dependent divergence.
  **Tier 6 builds on Tier 5, it does not replace it:** Tier 5 establishes the
  forward model is correct; Tier 6 then measures the approximation against the
  now-trusted forward model. A buggy forward model would make Tier 6
  meaningless. Both, in order.

---

## 11. Open questions

1. **Summary-statistic selection** (§8) — which summaries, and how to choose
   them objectively (sensitivity of $\hat\mu(\theta)$ to target parameters; the
   normality caveat). The single most consequential scientific choice.
2. **Event-log format/size** — Parquet schema for the event sequence + sparse
   lineage-event weights; compression.
3. **`realize` over batched backends** — within a tau-leap / chain-binomial
   step, sample all $k$ attributions against the **frozen start-of-step** pools
   recorded in the event log (mirroring the backend's own propensity-freezing).
   Pin the within-step ordering convention: "all attributions in a batched step
   use start-of-step pool membership." A consequence is that one individual can
   be sampled as both a parent and the recoverer within a step — this is part of
   the documented tau-leap bias, not a new one.
4. **Marginalize vs sample lineage demes** in the future structured-coalescent
   channel (§7) — MASCOT-style ODEs vs PGAS lineage-deme paths. Architecture
   permits both; the choice is deferred to the inference milestone.

---

## 12. Non-goals / deferred

- Inference itself (this proposal reserves the §7 path; does not build it).
- Nonlinear-in-parents rates (`infector(...)`), environmental transmission —
  unchanged deferrals.
- Sequence evolution / phylogeny / tree uncertainty — reserved as layers above
  `Tree` (§9), not built.
- General large-tree exact sampled-tree likelihood — out of reach; §8
  (summaries) is the answer.
- A full-tree product likelihood — explicitly **not** built; it is incorrect
  (§4b).
- Native SL/BSL/ABC fit — external for now; reserved via `Tree::summaries()`.
