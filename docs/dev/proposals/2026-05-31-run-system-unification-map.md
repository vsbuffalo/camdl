# Run-system unification map: simulate / batch / fit / profile, the CAS, and what's diverged

**Date:** 2026-05-31
**Status:** Living map / planning doc — NOT a single proposal. The thing to
read first when touching `simulate`, `batch`, `fit`, `profile`, the CAS
layout, `run.json`, or ensemble output.
**Why this exists:** the CLI-UX rev-3 work (CAS-default output, progress)
kept colliding with half-finished unifications — each discovered by stumbling
into it mid-change (output sink, ensemble layout, `ReplicateSet` adoption,
fit's ad-hoc grouping). This doc inventories the moving parts up front: what
is one cohesive thing, what has diverged, what is stale, and which seams a
change must respect. It is descriptive (the territory), with a recommended
target; individual fixes get their own proposals/issues that cite this map.

> **Verification note.** Every "today" claim below was checked against source
> on 2026-05-31 (HEAD `fbedb5a`, rebased on origin/main). `file:line` cited
> where load-bearing. Where I have NOT re-verified since, it's marked
> *(unverified)*. Treat this as a map to re-walk, not gospel — re-grep before
> relying on any single line.

---

## 0. The intended design (run-spec §3.1)

The run-spec already states the north star: **`SimulateJob` is THE type —
"CLI and file both produce this."** One job type, one engine, one CAS layout,
read uniformly by `camdl list/show/cat` and camdl-viewer. The four run
producers (`simulate`, `batch`, `fit`, `profile`) should differ only at the
*front* (how the job is specified) and converge on shared *middle* (the
engine) and *back* (the CAS) layers.

The reality: the **engine** layer largely converged (2026-05-28); the
**output/sink** and **ensemble-grouping** layers did not. The divergences
below are all "the convergence stopped partway."

---

## 1. The producers and what they're for

| command | front-end | what it produces | does fitting? |
|---|---|---|---|
| `simulate` | CLI flags (ergonomic) | forward trajectories; 1 or N seeds | no |
| `batch run` | TOML config | forward trajectories over scenario × sweep × seed grid | no |
| `fit run` | fit.toml | MLE / posterior draws over stages × chains × seeds | yes |
| `profile` | CLI flags + grid | profile-likelihood: grid × start × seed of mini-fits | yes (per cell) |

`simulate` (ergonomic) vs `batch` (config-driven) is a *sound* split — same
underlying job, two front doors. Not a smell. The smell is downstream.

---

## 2. Layer-by-layer: unified vs diverged

### 2a. Engine (cell expansion + RNG seed mix) — ✅ UNIFIED
`simulate` and `batch run` both build a `SimulateJob` and run it through
`engine::run_job` over the `RunSink` trait (commit `0b09d2f`,
"unify simulate and batch under one run_job engine"). Cell loop, seed
arithmetic, scenario resolution (`ScenarioRef`) are shared. Determinism is
pinned by `cli/tests/determinism_pin.rs` (CRN coupling + seed coherence).
**This is the model for what "unified" should look like everywhere.**
`fit`/`profile` have their own runners (not `run_job`) — expected, they're a
different job shape (estimation, not forward sim).

### 2b. Output sink (where a run's bytes go) — ❌ DIVERGED
The engine is shared but each caller passes a different `RunSink`:
- `simulate`: `StreamSink` (`main.rs:958`) — concatenates all cells to one
  wide-format TSV (`replicate`/`scenario`/`draw` columns) to **stdout** (or
  `-o`). Plus a *separate* single-run-only `--cas` path (`cas_ctx` /
  `RunBuffer`, `main.rs:662,917`) that writes one `sims/.../seed_N/`.
- `batch`: `CasSink` (`batch.rs:785`) — writes **every** cell to
  `sims/.../seed_N/` + a batch `manifest.json`.

Consequences (the bugs this caused): `simulate` defaults to stdout, `batch`
to CAS; `simulate --cas` is **rejected for ensembles** (`main.rs:512`) so you
can't get a simulate ensemble into the CAS at all; multi-seed simulate to
stdout has only a `replicate` column, not the actual seed. **This is the
divergence the CAS-default work (rev-3 §4.4) must close** — by routing
simulate's default through a shared CAS sink.

### 2c. Ensemble grouping (how N seeds are grouped on disk) — ❌ DIVERGED (3 ways)
The deepest divergence. "A group of runs that vary only in seed" has **three
different on-disk shapes** today:

| producer | grouping shape | umbrella object? | cross-replicate file | `camdl cat <group>`? |
|---|---|---|---|---|
| `profile` | **`ReplicateSet`** (`typed.rs:196`) — `replicates/seed_N/` under an umbrella dir | ✅ `run.json` kind=`ReplicateSet` + content hash | `summary.tsv` (per umbrella) | ✅ emits summary |
| `batch` | `CasSink` grid — `<scen>-<scen8>/seed_N/` siblings | ❌ implicit (siblings) | `manifest.json` (batch-wide) | ✗ no group handle |
| `fit` | **ad-hoc**, ReplicateSet-shaped but not wrapped | ❌ (explicit TODO) | — | ✗ |
| `simulate` | **doesn't exist** (ensembles only stream to stdout) | — | — | — |

- `ReplicateSet` is **real and the best of the three**: the group is a
  first-class addressable object (one hash) with a `summary.tsv` and working
  `cat`. Used only by `profile` today (`profile.rs:1224`), with 6 unit tests
  (`typed.rs`). It was *designed* for simulate too — `sim_inputs.rs:7`:
  *"if [multi-seed simulate] added, the runner would build N SimulateInputs
  and group them via cas::typed::ReplicateSet — same pattern as profile."*
- `fit` has a literal `TODO(typed-cas): formalize fit_seeds as
  cas::typed::ReplicateSet` (`fit/mod.rs:502-506`) — *"mirrors a ReplicateSet
  but isn't wrapped … no RunKind::ReplicateSet umbrella, no replicates/seed_S/
  path."*
- `batch`'s grid is legitimately a *different* shape (scenario × sweep × seed,
  not a single replicate dimension) — a batch is conceptually a *set of*
  ReplicateSets (one per grid cell), so unifying batch onto ReplicateSet is a
  real structural change, not a rename.

**Decision recorded (2026-05-31):** `simulate` ensembles will adopt
**`ReplicateSet`** (reusing profile's machinery), per the stated design intent.
This makes simulate consistent with `profile` (the other `--seeds` producer)
and gives the addressable-umbrella + `summary.tsv` + `cat` UX for free.
Single-seed simulate stays the plain `sims/.../seed_N/` path. Batch and fit
are NOT changed by the rev-3 UX work — their convergence onto ReplicateSet is
deferred (§4).

### 2d. The `run.json` metadata contract — ✅ MOSTLY UNIFIED
One tagged `Run { hash, version, created_at, argv, status, label, kind }`
with `RunKind ∈ {Simulate, Fit, FitStage, Profile, Survey, Batch,
ReplicateSet}` (`run_meta.rs`). All readers (`browse.rs`: `list`/`show`/`cat`)
dispatch on `kind`. This is the consumer contract (run-spec §9.5). Good shape;
the gap is only that the producers above don't all *emit* the umbrella kinds
they should (2c).

### 2e. The compiled IR — ❌ NOT CACHED (orthogonal)
Recompiled every run (~21s/8.4GB on Kano), no CAS caching, double-compiled on
`--obs` (gh#141). Not a unification issue per se, but it's the same "content-
addressable artifact that isn't content-addressed" pattern. Tracked: gh#141.

### 2f. Observation data binding — 🔶 IN-FLIGHT (separate proposal)
Unified observation-data surface (`--data NAME=PATH`, `data.<col>`) is its own
landed-proposal (`2026-05-30-unified-observation-data.md`, commit `bca0360`)
+ a `t_cond` conditioning-boundary proposal. The rev-3 CLI work defers all
`--obs`/`--data` flag questions to it. Noted here only so the map is complete.

---

## 3. What's stale vs deeply integrated (the "don't trip on it" inventory)

- **`ReplicateSet`** — deeply integrated (profile + browse + 6 tests). NOT
  stale. Safe to build on. Under-adopted (only profile), not rotting.
- **`StreamSink`** — live, but its role shrinks to "the `--stdout` path" once
  CAS-default lands. Not stale yet; will become a narrow special-case.
- **single-run `--cas` path in `simulate`** (`cas_ctx`/`RunBuffer`/
  `prepare_cas_ctx`) — live, but **superseded** once simulate routes through a
  shared sink; candidate for deletion after the CAS-default commit (don't
  leave both paths — CLAUDE.md "v1 alongside v2 is dead code").
- **batch `manifest.json`** — live; the only cross-cell index batch has. If
  batch ever moves to ReplicateSet, manifest's role is reconsidered.
- **`fit` ad-hoc seed grouping** — live but explicitly marked for
  formalization (`fit/mod.rs:502`). The TODO is the breadcrumb.
- **`model_hash`** — fixed (gh#135, envelope descent). The two test-mirrors
  that lagged are also fixed (`aabf54a`, `7966a62`). Not stale.

---

## 4. Recommended target + sequencing (so point-fixes stop colliding)

**Target state:** one ensemble-grouping abstraction (`ReplicateSet`), one CAS
sink shared by all forward-sim producers, `run.json` umbrella kinds emitted by
everyone, read uniformly. Concretely:

1. **(rev-3, now) simulate CAS-default.**
   - Single-run: flip default stdout→CAS + `--stdout` opt-out + banner.
     Delete the superseded single-run `--cas` path once the new one works.
   - Ensemble: `--seeds`/`--replicates` → a `ReplicateSet` umbrella;
     `camdl cat <umbrella>` re-emits the convenient `replicate`-column merged
     TSV (read-side merge — storage stays per-seed, dedupable). `--stdout`
     preserves today's merged-stream behavior byte-for-byte.
   - **Gate:** `ir/expected/*.tsv` + `determinism_pin` byte-identical.
2. **(later, tracked) batch ↔ ReplicateSet.** Decide whether a batch sweep
   becomes a set of ReplicateSets (one per grid cell) or keeps its grid
   layout with an added umbrella handle. Structural; its own proposal.
3. **(later, tracked) fit ↔ ReplicateSet.** Resolve `fit/mod.rs:502`'s TODO —
   wrap fit-seed groups in the formal umbrella. Inference-adjacent → careful,
   own proposal, own determinism gate.
4. **(orthogonal) compiled-IR cache** — gh#141.
5. **(in-flight) observation-data** — `2026-05-30-unified-observation-data.md`.

**Rule going forward:** before adding another per-producer special case to the
CAS/output path, check this map. If the change touches a 2b/2c divergence,
either close it via the shared abstraction or extend this map with the new
seam — don't add a fourth shape.

---

## 5. Open questions for the maintainer

- **Batch's grid vs ReplicateSet:** is a batch sweep best modeled as N
  ReplicateSets (one per scenario×sweep cell), or does the grid warrant its
  own umbrella kind? (Affects §4.2.)
- **summary.tsv contents for a *simulate* ReplicateSet:** profile's summary is
  fit-oriented (MLE/loglik per seed). A forward-sim ensemble summary wants
  something else (per-time quantile bands? final-state distribution?). What's
  the right cross-seed aggregate for a simulate ensemble — or is `cat`-merge
  to the raw replicate TSV enough and summary.tsv stays empty/N=1-trivial?
- **Scope appetite:** do batch+fit unification (§4.2, §4.3) belong in this
  push, or are they a deliberately separate, later effort?
