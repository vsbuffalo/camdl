# Lineages: event logs, line lists, and transmission trees

camdl's backends track compartment **counts**. The lineage layer adds an
optional **identity** layer that records *which* individual infected *which* —
but it does so in three cleanly separated stages, because the genealogy is a
*conditional sample*, not a deterministic function of the epidemic.

This is for **benchmark and synthetic-data generation** (e.g. validating
phylodynamic inference against a known true tree). Enabling it never changes
the count dynamics.

## Why three layers: the line list is a conditional sample

The augmented process factorizes:

```
P(augmented trajectory) = P(count trajectory) × P(identity attribution | count trajectory)
```

The compartmental simulation draws the **first** factor — it fixes the ordered
event sequence (a transmission fired at t₁, a recovery at t₂, …). It does *not*
fix the second: given that event sequence, *which* individual was the infector
at each transmission, and *which* recovered at each recovery, are a separate
stochastic layer. Many identity attributions are equally consistent with one
count trajectory. So **one epidemic defines a distribution over genealogies**,
not a single tree — and benchmark validation needs the ensemble.

camdl therefore separates three stages, each independently cacheable:

| Layer | Command | Produces |
|---|---|---|
| **1. Event log** | `simulate --event-log` | the epidemic: ordered events, identity-free |
| **2. Line list** | `lineage realize --identity-seed` | one genealogy realization + its log-probability |
| **3. Tree** | `lineage tree --scheme --sample-seed` | a sampled, pruned, Newick transmission tree |

```bash
camdl simulate model.camdl --params p.toml --seed 42 --event-log epi.parquet
camdl lineage realize epi.parquet --identity-seed 7 -o line_list.parquet
camdl lineage tree    line_list.parquet --scheme flat:0.1 --seed 3 -o tree.nwk
```

One expensive epidemic → many cheap identity realizations → many cheap trees.
Changing the identity seed or the sampling scheme never re-runs the epidemic.

## Marking transmissions: `#[lineage]`

A transition becomes a parent→child event with the `#[lineage]` attribute:

```camdl
transitions {
  #[lineage]
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
```

At record time the recorder evaluates the per-infector-pool weights from the
rate; at `realize` time a parent is sampled from those weights.

**Linear-in-parents requirement.** The rate must be linear in the infector
compartments. `β·S·I/N` compiles (`I` is a linear parent in the numerator; its
appearance in the `N` denominator is a frozen normalizer — denominator
precedence). `β·S·(I+ι)^α/N` does **not** (the power is a genuine nonlinear use
of `I`) and errors with `E601`. Nonlinear mixing is future work.

**Overdispersion is transparent.** `overdispersed(β·S·I/N, σ²)` on a
`#[lineage]` transition compiles: the classifier sees through the noise wrapper
and extracts the same linear weight `β·S/N`. The σ² environmental noise affects
the *count dynamics*, not the *attribution* — so overdispersed processes work,
on the chain-binomial / tau-leap backends that `overdispersed()` requires.

## Backends and tree accuracy

| Backend | Lineage support | Tree accuracy |
|---|---|---|
| `gillespie` | exact | one event at a time — exact attribution; sub-`dt` bias = 0 |
| `tau_leap`, `chain_binomial` | approximate | *k* events per step against frozen start-of-step pools; **systematically loses parent→child edges shorter than `dt`**. `realize` reports the sub-`dt` edge-loss fraction. |
| `ode` | rejected | no individuals — hard error |

`overdispressed()` models require chain-binomial / tau-leap, so their trees are
approximate; shrink `--dt` for trustworthy trees, and read the reported sub-`dt`
bias as the accuracy bound. The exact backend (Gillespie) cannot run
overdispersed models.

## Layer 1 — the event log

Identity-free. Records, per event: `time`, `transition`, `multiplicity` (for
batched backends), the batched-step index, and — at `#[lineage]` events — the
**evaluated per-pool masses** `{w_b · X_b}`. With the per-transition route table
and the t=0 initial pools (stored as metadata), the event log is
**self-contained**: replay needs only the log, not the model or the rate AST.
The simulation draws no identity randomness, so `--event-log` trajectories are
byte-identical to plain runs at the same seed.

## Layer 2 — `realize`, and the line-list likelihood

`realize` replays the event log under an `--identity-seed` (an independent RNG
stream), maintaining identity pools and sampling, at each event, **pool then
individual**: pool `b` with probability `w_b·X_b / Λ` (Λ = Σ_b w_b·X_b), then a
uniform individual within that pool. Different identity seeds give i.i.d. draws
from `P(identities | event log)`.

The line list specifies **every** attribution, so its likelihood is a clean
product over events (conditional independence given the log):

```
log P(line list | event log) = Σ_events log P(attribution)
  transmission, parent in pool b:   log( w_b / Λ )         (pool choice × uniform; X_b cancels)
  recovery / removal in pool b:     log( 1 / |I_b| )       (uniform within the relevant pool)
```

`realize` accumulates this and reports it (and stores it per line list). **This
is the only clean exact likelihood the architecture provides.** There is *no*
cheap full-tree product: recovery attributions are not independent of the tree
(they set which individuals remain available as parents), so any tree
likelihood requires a marginalization that is combinatorial — the sampled-tree
likelihood is exactly what structured-coalescent methods (MASCOT et al.)
approximate. See the design proposal for the counterexample and the
summary-statistic synthetic-likelihood route.

### Line-list columns

`time`, `transition`, `individual`, `source`/`destination` (compartment indices,
`-1` if none), `deme`, `parent_kind` (`individual`/`import`/`seed`/`none`),
`parent_id`, `parent_deme`, `attribution_logprob`.

## Layer 3 — projections

`lineage tree`, `lineage sojourn`, `lineage cohort` are pure functions of the
line list.

```bash
camdl lineage tree    line_list.parquet --scheme flat:0.1 --seed 3 -o tree.nwk
camdl lineage sojourn line_list.parquet --compartment 1            # I dwell-time dist
camdl lineage cohort  line_list.parquet --event infection --window 7
```

`lineage tree` builds the parent→child forest, samples observed tips, prunes to
the minimal subtree spanning them (suppressing unary nodes), and emits Newick
with **time-calibrated branch lengths**.

### Forest vs. tree

The structure recovered from a line list is in general a **forest** — one tree
per *independent introduction*. A root is any individual with no in-simulation
parent: each seed infective (`I₀ > 1`) and each importation founds its own
tree. Only a single introduction yields a single tree. `lineage tree` reports
how many forest components survived sampling and emits one Newick per surviving
root.

## Sampling (current and planned)

**Today:** only `flat:RATE`, drawing candidates from the forest **leaves**
(individuals who infected nobody), tips placed at *infection* time. So a
`flat:RATE` tree is a constant-probability sample of transmission-chain
*endpoints* — **not** realistic surveillance (which samples *any* case,
including infectors, at its *sampling* time). Tree-shape statistics from
`flat:RATE` are biased relative to a realistically-sampled tree.

**Planned (sampling milestone).** A `SamplingScheme` over *all* individuals with
pendant tips at sampling time; the scheme declared structurally in the model
(`lineage { sampling { scheme, condition, by, rate } }`) with rates as ordinary
parameters. See the design proposal.

## Caching / provenance

```
event_log_hash = f(model, params, dynamics_seed)
line_list_hash = f(event_log, identity_seed)
tree_hash      = f(line_list, sampling_scheme, sample_seed)
```

Three keys, one expensive step.

## Status and roadmap

**Shipped:** the three-layer pipeline (event log → realize → tree/sojourn/
cohort) across all three stochastic backends; the exact line-list likelihood;
stratified contact-weighted attribution; overdispersed processes; validation
against Yule statistics and the SIR structured-coalescent rate
(`λ = 2f/I² = 2βS/(NI)`).

**Coming next / not yet built** (don't rely on these):
- **Sampling realism** — all-individuals sampling, pendant tips at sampling
  time, the `lineage { sampling }` block. Replaces leaf-only `flat:RATE`.
- **`Tree`/`SyntheticTree` no-cheating split** — an observable `Tree` boundary
  type for the inference-validation loop.
- **`lineage loglik`** — tree-likelihood scoring; only the line-list logprob
  exists today.
- **Nonlinear mixing** (`infector(...)`), **environmental transmission**,
  **sequence evolution**, **native tree inference** — all deferred.
