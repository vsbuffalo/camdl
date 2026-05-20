# Lineages: transmission line lists and trees

camdl's backends track compartment **counts**. The lineage layer adds an
optional **identity** layer on top: it records *which* individual infected
*which*, producing a per-event **line list** from which transmission trees,
sojourn-time distributions, and cohort summaries are derived offline.

This is for **benchmark and synthetic-data generation** (e.g. validating
phylodynamic inference against a known true tree). It is orthogonal to the
inference stack — enabling it never changes the count dynamics.

> **Status.** Forward simulation, the line list, and the tree/sojourn/cohort
> projections ship today. The sampling model is currently a placeholder
> (`flat:RATE` over transmission-chain endpoints); a realistic sampling
> milestone is in design (see [Roadmap](#roadmap-and-current-limitations)
> and `docs/dev/proposals/2026-05-19-individual-sampling-layer.md`).

## Marking transmissions: the `#[lineage]` annotation

A transition becomes a lineage (parent→child) event with the `#[lineage]`
attribute:

```camdl
transitions {
  #[lineage]
  infection : S --> E  @ beta * S * I / N

  progression : E --> I  @ sigma * E
  recovery    : I --> R  @ gamma * I
}
```

At each firing the runtime samples a parent from the infector pool (read from
the rate's structure) and mints a new tracked individual in the destination.
Non-lineage transitions still appear in the line list (with no parent), which
is what powers sojourn/cohort projections.

**Linear-in-parents requirement (v1).** The rate of a `#[lineage]` transition
must be linear in the infector compartments. `β·S·I/N` compiles (frequency-
dependent transmission: `I` is a linear parent in the numerator; its
appearance in the `N` denominator is a frozen normalizer). `β·S·(I+ι)^α/N`
does **not** — the power is a genuine nonlinear use of `I` — and errors with
`E601`. Nonlinear mixing (He α-mixing, saturating infectiousness) is future
work (the `infector(...)` wrapper, deferred).

## Running a simulation with lineage tracking

```bash
# Parquet line list (production)
camdl simulate model.camdl --params p.toml --seed 42 \
      --lineages --backend gillespie --lineage-out line_list.parquet

# TSV line list (debug / small runs)
camdl simulate model.camdl --params p.toml --seed 42 \
      --lineages --tsv --lineage-out line_list.tsv
```

`--lineages` is single-run only (conflicts with `--seeds` / `--replicates`).

### Backends and accuracy

| Backend | Lineage support | Tree accuracy |
|---|---|---|
| `gillespie` | exact | One event at a time — exact attribution. |
| `tau_leap` | approximate | *k* events per step against frozen start-of-step pools; **systematically loses parent→child edges shorter than `dt`**. A sub-`dt` bias fraction is reported alongside the run. |
| `chain_binomial` | approximate | As tau-leap. |
| `ode` | **rejected** | No individuals — hard error. |

Trustworthy benchmark trees want **Gillespie** (or a small `dt`). The
approximate backends report their sub-`dt` edge-loss fraction so the accuracy
bound is explicit (exactly `0.000` for Gillespie).

## Line-list format

One row per identity-tracked event. Columns:

| column | meaning |
|---|---|
| `time` | event time (model time unit) |
| `transition` | transition index (0-based, in model order) |
| `individual` | focal individual id |
| `source` / `destination` | compartment indices (`-1` if none, e.g. inflow/outflow) |
| `deme` | the focal individual's stratum/patch index (`0` if unstratified) |
| `parent_kind` | `individual` (lineage event), `import`, `seed`, or `none` |
| `parent_id` | infector's individual id (`-1` when `parent_kind ≠ individual`) |
| `parent_deme` | infector's stratum (`-1` for non-lineage events) |

The infection rows (`parent_kind = individual`) are the transmission-tree
edges; the line list *is* the tree, latently.

## Offline projections (`camdl lineage`)

All projections are pure functions of the line list — re-runnable and
cacheable without re-simulating. Input format (`.tsv` / `.parquet`) is
auto-detected.

### Transmission tree → Newick

```bash
camdl lineage tree line_list.parquet --scheme flat:0.1 --seed 1 -o tree.newick
```

Builds the parent→child forest, applies the sampling scheme to choose observed
tips, prunes to the minimal subtree spanning them (suppressing unary internal
nodes), and emits Newick with **branch lengths in time units**.

`--scheme flat:RATE` — each candidate tip kept i.i.d. with probability `RATE`.
`flat:1.0` keeps every candidate (the full tree); `--seed` makes the draw
deterministic.

### Sojourn-time distribution

```bash
camdl lineage sojourn line_list.tsv --compartment 1   # e.g. the I compartment
```

Per-individual dwell time in a compartment (entry→exit), with a summary
(count, censored, mean, quantiles) to stderr. `--compartment` takes the
integer compartment index (matching the `source`/`destination` columns).

### Cohort / incidence summary

```bash
camdl lineage cohort line_list.tsv --event infection --window 7
```

Per-time-window event counts (incidence + cumulative). `--event infection`
counts all lineage events; an integer counts a specific transition.
`--window` sets the bin width; `--align-zero` aligns bins to t=0.

## Forest vs. tree

The structure recovered from a line list is in general a **forest** — a
disjoint union of trees, one per *independent introduction*. A root is any
individual with no in-simulation parent: every seed infective (`I₀ > 1`) and
every importation founds its own tree. Only with a **single introduction**
(`I₀ = 1`, no imports) is the result a single tree. `camdl lineage tree`
reports how many forest components survived sampling, and emits one tree per
surviving root.

## Sampling (current and planned)

**Today:** only `flat:RATE`, and it draws candidates from the **leaves** of
the forest — individuals who infected nobody. So a `flat:RATE` tree is a
constant-probability sample of *transmission-chain endpoints*, with tips
placed at *infection* time.

This is a Phase-1 placeholder and is **not realistic surveillance**: real
sequencing samples *any* case (including infectors, often especially
super-spreaders) at its *sampling* time. Tree-shape statistics from
`flat:RATE` are biased relative to a realistically-sampled tree.

**Planned (sampling milestone, in design).** A `SamplingScheme` that samples
over *all* individuals and places a pendant tip at the sampling time, with the
scheme declared structurally in the model and its rates supplied as ordinary
parameters:

```camdl
# PLANNED — not yet implemented
lineage {
  sampling {
    scheme    = stratified
    condition = at_removal
    by        = [patch, age]
    rate      = surveillance_rate     # a regular parameter (priors, bounds, fittable)
  }
}
```

with `Flat`-over-all-individuals, `Stratified`, and `ConditionalOnRemoval`
implementations. Sampling-relevant state that the dynamics don't use (e.g. a
detection sub-process) is expressible *today* as a stratification dimension no
rate references (`stratify(by = detection, only = [I])`) — but conditioning
sampling on it awaits this milestone. See the proposal for the full design.

## Roadmap and current limitations

Shipped: line list (TSV/Parquet), `#[lineage]` (linear rates), all three
stochastic backends, `tree`/`sojourn`/`cohort` projections, validation against
Yule statistics and the SIR structured-coalescent rate.

Known limitations / coming next:
- **Sampling realism** (next milestone): all-individuals sampling, pendant
  tips at sampling time, the `lineage { sampling { } }` block. Replaces
  leaf-only `flat:RATE`.
- **Nonlinear mixing** (`infector(...)`, deferred): He α-mixing etc.
- **Environmental transmission** (deferred): SIWR-style reservoir sources.
- **External-oracle validation** (gated behind realistic sampling): an
  independent lineage simulator cross-check.
