# Proposal: three-layer lineage architecture — resampling and tree likelihood

Date: 2026-05-20
Status: Draft (RFC) — for review before implementation
Branch: `feature/lineages` (refactor of the shipped two-layer design)
Supersedes: the *architecture* of
`docs/dev/proposals/2026-05-19-individual-sampling-layer.md`; that
document's implementation (Phases 1–3: count-level lineage tracking,
stratified attribution, three backends, projections, validation) is the
foundation this refactors, not work to discard.

> Citations marked **[verify]** are from background knowledge and must
> be checked against primary sources before this circulates for review.

## 1. The insight: a line list is a conditional sample, not a realization

The shipped design produces one line list per simulation and treats it
as *the* genealogy. That is a category error. The augmented process
factorizes:

$$P(\text{augmented trajectory}) \;=\; P(\text{count trajectory}) \;\times\; P(\text{identity attribution} \mid \text{count trajectory}).$$

The compartmental simulation draws the **first** factor: it fixes the
ordered event sequence — at $t_1$ a transmission fired, at $t_2$ a
recovery fired, … — and the counts evolve deterministically given those
events. It does **not** fix the second factor. Given that event
sequence, *which specific individual* was the infector at each
transmission, and *which* individual underwent each recovery, are a
separate stochastic layer: many identity attributions are equally
consistent with the same count trajectory.

So a single compartmental run defines a **distribution over line
lists**, not one line list. For benchmark generation this is decisive:
the phylodynamic method under validation (MASCOT, BASTA, BDMM) assumes
the observed tree is *one draw* from $P(\text{tree} \mid \text{trajectory})$.
Validating it requires the *ensemble* of trees consistent with an
epidemic, not a single tree. The current two-layer design can only
reach the ensemble by re-running the (expensive) epidemic per identity
draw — and it tempts the user into treating one tree as canonical.

This already half-holds in the shipped code: identity randomness is
drawn from a **separate RNG stream** from the count dynamics (the
verified byte-identity invariant). That separation is the empirical
proof of the factorization. The refactor makes the factorization
*structural*: persist the first factor once, resample the second cheaply.

## 2. Three layers, three artifacts, three cache keys

| Layer | Map | Cost | Cache key |
|---|---|---|---|
| **1. Event log** | $(\text{model}, \text{params}, \text{seed}) \to$ event log | expensive (one epidemic) | `f(model, params, dynamics_seed)` |
| **2. Line list** | $(\text{event log}, \text{identity\_seed}) \to$ line list | cheap (replay) | `f(event_log, identity_seed)` |
| **3. Tree** | $(\text{line list}, \text{scheme}, \text{sample\_seed}) \to$ tree | cheap (prune) | `f(line_list, scheme, sample_seed)` |

One expensive epidemic → many cheap identity realizations → many cheap
trees. A Monte Carlo benchmark sweeps all three: many epidemics
(`dynamics_seed`), each with many identity draws (`identity_seed`), each
with many sampled trees (`sample_seed`). Caching means tweaking the
sampling scheme never re-runs the epidemic, and tweaking the identity
seed never re-runs it either.

The count simulation becomes **completely lineage-free**: it draws no
identities, it only records the event log. The byte-identity invariant
becomes trivially true (the simulation is literally unchanged;
`--event-log` records what it already computes). The refactor
*simplifies* the simulation-side coupling we worked to establish.

### CLI

```bash
camdl simulate model.camdl --params p.toml --seed 42 --event-log epi.parquet
camdl lineage realize epi.parquet --identity-seed 7 -o line_list.parquet
camdl lineage tree    line_list.parquet --scheme stratified --sample-seed 3 -o tree.nwk

# fused (common case) + replicate sweeps:
camdl lineage tree epi.parquet --identity-seed 1:100 --scheme stratified --sample-seed 1:10 -o trees/
```

(No single-realization shortcut: "one line list" is just "realize once."
The event log is the canonical source.)

## 3. The event log

Minimal sufficient statistic for resampling: **initial state +
ordered $(\text{time}, \text{transition\_id})$**, plus `multiplicity`
for batched (tau-leap / chain-binomial) steps. From that, the model,
and the parameters, one can replay the counts and recompute the
per-pool weights at every lineage event.

We additionally record the **evaluated per-pool FOI weights**
$\{(b, w_b X_b)\}$ at each `#[lineage]` event, so the event log is
**self-contained**: the Layer-2 replay needs only the event log, not
the model or the rate AST. Size cost is modest (weights only at lineage
events, not every event); the decoupling is worth it. The event log is
**identity-free** — individual IDs are minted during replay; the log is
the epidemic, not the genealogy. It is finer-grained than the existing
`flows` output (cumulative counts at output times): the event log is
the raw event sequence. New output type, gated by `--event-log`.

## 4. Likelihoods

The recorded weights make several likelihoods fall out of the Layer-2
replay. Notation: at a transmission event, the infector pools have
weights $w_b$ and sizes $X_b$; the realized FOI total is
$\Lambda = \sum_{b'} w_{b'} X_{b'}$.

### 4a. Line-list likelihood (exact, cheap, general)

Identity choices are conditionally independent across events given the
event log (Markov), so

$$\log P(\text{line list} \mid \text{event log}) = \sum_{\text{events}} \log P(\text{attribution at that event}).$$

Per event:
- **Transmission**, parent = individual $i$ in pool $b$:
  $P = \dfrac{w_b X_b}{\Lambda} \cdot \dfrac{1}{X_b} = \dfrac{w_b}{\Lambda}$ (pool choice × within-pool uniform).
- **Recovery / single-pool removal**: $P = 1/|I|$.

A product of per-event terms, accumulated for free during replay. Each
resampled line list carries its own log-probability.

### 4b. Full labeled tree likelihood (exact, cheap)

The tree topology and node times are determined entirely by the parent
assignments at transmission events; recovery/progression identities are
marginal and independent of the tree (they sum to 1). Hence

$$P(\text{full labeled tree} \mid \text{event log}) = \prod_{\text{transmission events}} P(\text{parent assignment}).$$

This is the *full* genealogy (every individual a tip), exact and cheap.

### 4c. Sampled-tree likelihood (intractable in general — be honest)

A sampled tree (tips = a subset of individuals) requires marginalizing
over the genealogical placement of all unsampled individuals:

$$P(\text{sampled tree} \mid \text{event log}) = \sum_{\substack{\text{line lists consistent}\\\text{with the sampled tree}}} P(\text{line list} \mid \text{event log}).$$

This sum is combinatorial. **The structured coalescent is exactly the
analytic approximation to this quantity in the large-$N$ diffusion
limit** [verify: Volz 2009; MASCOT — Müller et al. 2017]. A forward
Monte Carlo estimator ("resample line lists, count the fraction
consistent with the sampled tree") is *exact in expectation* but has
**catastrophic variance**: the consistent set is an exponentially small
fraction of identity realizations, so the hit rate is ≈ 0 for any
non-trivial tree. This is precisely why phylodynamics uses analytic
approximations rather than forward MC.

**Honest scope:** the forward-MC sampled-tree likelihood is tractable
only for **small trees** (few tips — direct enumeration or
importance-sampled MC). In that regime it is a genuinely powerful,
novel tool: an *exact reference* against which to measure where and how
badly the coalescent approximation (MASCOT et al.) deviates, for a
*specific dynamical regime*. It does **not** provide an exact
likelihood for arbitrary large trees, and we should not claim it does.

## 5. Summary-statistic synthetic likelihood (the tractable path)

For trees beyond the small-$N$ enumeration regime, abandon the exact
sampled-tree likelihood and use a **composite / synthetic likelihood**
over tree summary statistics — the approach the three-layer
architecture is built to feed.

Pick a summary vector $S(\text{tree}) \in \mathbb{R}^d$. At parameter
$\theta$, the cheap forward ensemble (one event log → many resampled
identity draws → many sampled trees) yields empirical summaries
$\{S^{(1)}, \dots, S^{(M)}\}$. Under the **synthetic likelihood**
assumption [verify: Wood 2010, *Nature* 466:1102] that $S$ is
approximately multivariate normal,

$$\hat\mu(\theta) = \tfrac{1}{M}\sum_m S^{(m)}, \quad \hat\Sigma(\theta) = \widehat{\mathrm{Cov}}(S), \quad \ell_{\mathrm{SL}}(\theta; S_{\mathrm{obs}}) = \log \mathcal{N}\!\big(S_{\mathrm{obs}};\, \hat\mu(\theta),\, \hat\Sigma(\theta)\big).$$

The Bayesian variant is BSL [verify: Price et al. 2018, *JCGS*]; ABC
with the same summaries is the simulation-only alternative. All three
are **likelihood-free inference on the forward model** — no exact tree
likelihood required.

### Candidate summary statistics

The summary set is the central scientific design choice: informative
for the target parameters, low-dimensional enough for a stable
$\hat\Sigma$, and capturing the tree features inference depends on.

- **Superspreading / offspring dispersion** — the negative-binomial
  dispersion $k$ of the per-infector offspring distribution [verify:
  Lloyd-Smith et al. 2005, *Nature* 438:355]. Epidemiologically
  meaningful, and **already partly computed** by the shipped offspring
  check.
- **Tree imbalance** — Sackin / Colless indices.
- **Temporal structure** — lineages-through-time (LTT) curve features;
  the $\gamma$ statistic [verify: Pybus & Harvey 2000].
- **Structured-tree summaries** — tip-stratum proportions; cross-stratum
  transition counts implied by the tree (the contact-structure
  signal).

### Validation payoff

Compare the forward-summary likelihood/posterior surface to MASCOT's
coalescent-analytic surface: where they peak together vs. diverge,
across the dynamical regime, is the **tractable Level-0
approximation-error test** — the deepest validation the lineage feature
enables, and one no existing tool provides (none keeps the event log to
resample from). This turns the feature from "generates benchmark trees"
into "generates benchmark trees *and* provides a forward reference for
validating approximate phylodynamic inference."

## 6. Refactor plan (on `feature/lineages`, reuse-heavy)

This is a refactor + extension, not a rewrite. Reused unchanged: the IR
schema (`parent_pool_weights`, `identity_tracked_compartments`), the
attribution math (incl. stratified contact-weighting, Tier-2b
validated), the three backends, the projections (`tree`/`sojourn`/
`cohort`), the corrected coalescent (Tier 4), the docs/proposal.

Changes:
1. **Event-log writer** (`--event-log`): the current observer becomes an
   *event recorder* — it still evaluates the per-pool weights at lineage
   events (the rate/state are available there) but **records** them
   rather than sampling a parent. The count simulation no longer draws
   identities.
2. **`camdl lineage realize`** (Layer 2): event log → line list, with
   `--identity-seed`; samples pool-then-individual using the recorded
   weights and accumulates the line-list log-likelihood (§4a).
3. **`camdl lineage tree`** consumes either a line list or an event log
   (fusing realize + sample).
4. **`camdl lineage loglik`**: full-tree log-likelihood (§4b, general);
   sampled-tree estimator (§4c, small-tree only, clearly bounded);
   summary-statistic synthetic-likelihood scoring (§5).
5. **Provenance**: three hashes (§2); the override/scheme choices flow
   into the tree hash, never the model hash.

The attribution-sampling code moves from the inline observer to the
`realize` replay; the logic is identical, only its location changes.

## 7. New validation tier

**Tier 6 — forward reference vs analytic approximation.** On
small-to-moderate stratified trees, compare (a) the forward-MC /
synthetic-likelihood surface from the resampled ensemble against (b)
the structured-coalescent analytic likelihood (MASCOT). Report the
regime-dependent divergence. This subsumes the parked Tier-5
external-oracle goal in a more informative form — it is not just
"do we agree with another simulator," but "where does the analytic
approximation the field relies on break, measured against an exact-ish
forward reference."

## 8. Non-goals / deferred

- Nonlinear-in-parents rates (`infector(...)`), environmental
  transmission — unchanged deferrals.
- General large-tree exact sampled-tree likelihood — explicitly out of
  reach; §5 (summaries) is the answer instead.
- Importance-sampling / particle schemes to extend the exact small-tree
  regime — research, not v1.

## 9. Open questions

1. **Summary-statistic selection** (§5) — which summaries, and how to
   choose them objectively (e.g. sensitivity of $\hat\mu(\theta)$ to the
   parameters of interest). The single most consequential scientific
   choice here.
2. **Event-log format/size** — Parquet schema for the event sequence +
   sparse lineage-event weights; whether to compress.
3. **`realize` over batched backends** — replaying a tau-leap /
   chain-binomial step samples $k$ parents against the frozen
   start-of-step pools recorded in the event log; confirm the recorded
   weights suffice (they should — they are the frozen per-pool values).
4. **Whether the synthetic-likelihood machinery lives in camdl or is
   an external consumer** of the tree ensemble (camdl emits trees +
   summaries; the SL/BSL fit could be downstream). Leaning: camdl emits
   the ensemble and the summaries; the inference comparison is a
   notebook/consumer, not core.
