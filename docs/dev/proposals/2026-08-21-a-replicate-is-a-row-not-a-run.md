# A replicate is a row, not a run

Date: 2026-08-21 Status: proposed Issue: gh#704 Related: gh#699 (`list` walks
the fan-out), gh#698 (compression, the weaker lever)

## The decision

A multi-cell `simulate` writes its trajectories **twice**: once as a combined
`ensemble.tsv` with a `replicate` column, and once as one content-addressed
`Sim` leaf per cell. This proposal removes the second, and states the semantic
commitment behind it: **a stochastic replicate is a row inside an artifact, not
an artifact with its own identity.**

## Measured

On the ebola-bdbv store,
`camdl simulate … --replicates 1200 --seed 1 --scenario
control_10`, run once
per scenario:

| kind      |      dirs | leaves (`run.json`) |
| --------- | --------: | ------------------: |
| sims      | 2,735,850 |             550,647 |
| ensembles |       262 |                  88 |
| fits      |       654 |                  71 |

Sims are 99.96% of a 28 GB, 4.1-million-file store. **461,282 of 550,647 sim
leaves (83.8%) are ensemble members** — their trajectory bytes already sit in an
`ensemble.tsv`.

For one such run (1,200 replicates × 5 scenarios = 6,000 cells):

|                     |  files | bytes | counterpart in the ensemble |
| ------------------- | -----: | ----: | --------------------------- |
| 5 × `ensemble.tsv`  |      5 | 29 MB | —                           |
| 6,000 leaves        | 12,000 | 50 MB |                             |
| — of which traj.tsv |  6,000 | 29 MB | yes, 1:1                    |
| — of which run.json |  6,000 | 22 MB | **none**                    |

So 43% of the fan-out is metadata with no counterpart in the compact form, and
that metadata is redundant with itself: **90 keys per record, 8 of which differ
between cells** (`run_id`, two level hashes, the traj digest/bytes/mtime, and
`created_at`). `du` reports 70 MB against 50 MB of bytes; the rest is 4 KB block
padding on 12,000 small files.

## Why compression is not the answer

Measured on that one run (50.5 MB raw): per-file gzip -9 gives **2.50×**, a
solid tar.zst -19 gives **3.90×**. Three reasons it does not solve this:

1. **It does not reduce file count.** The measured pain — `list` and `find` both
   exceeding 120 s, and every backup, `rsync` and Time Machine pass — is a
   function of 4.1M files, not of bytes. Compression leaves every one in place.
2. **Per-file compression cannot see the redundancy that is actually there.**
   The 2.50× / 3.90× gap _is_ the 6,000 near-identical `run.json` files, which
   only a solid archive can dedupe. A live store must stay per-file readable, so
   only the 2.50× is available on disk.
3. **It is the smaller win.** Per-file gzip: 28 GB → ~11 GB, 4.1M files.
   Dropping the fan-out: 28 GB → ~11 GB **and** 550k leaves → ~90k. The two
   compose (~4 GB together), but only one of them fixes browsing.

## Design

`main.rs:1986` already draws the line this proposal needs, and states it:

> A multi-cell run (`total_runs > 1`) keeps its N per-cell `Sim` leaves AND
> additionally writes the combined wide-format TSV as a content-addressed
> ensemble that REFERENCES them (deps). **A single-run simulate writes NO
> ensemble (the one leaf is the whole thing).**

guarded by `if total_runs > 1 && !suppress_trajectory`.

**The change lives entirely inside that branch.** A one-off
`camdl simulate
model.camdl` is untouched: it writes one `Sim` leaf with its
`traj.tsv`, exactly as today. This is a hard requirement, not an incidental
property — see Verification.

### What changes

`ensemble.tsv` is today a _derived view_: `sim_ensemble_cas.rs:9` calls it "a
derived view over the N `Sim` leaves", and `batch.rs:397` describes a cell's
`traj.tsv` as the artifact the combined TSV is built from. So this is not "stop
writing leaves" — the per-cell files are the **source** the ensemble is read
back from. The fan-out path must instead append each replicate's rows to the
combined buffer as it completes, and never materialise a per-cell leaf.

### What does not change: the ensemble's identity

The ensemble's `grid` level folds in each cell's `sim_run_id`
(`sim_ensemble_cas.rs:16-26`), and a `sim_run_id` is a pure function of that
cell's model/config/params/scenario/seed. **It can still be computed without a
leaf existing.** So the ensemble's `run_id` stays byte-identical, every existing
ensemble keeps resolving, and there is no `ir/VERSION` implication.

### What is deliberately lost

- `camdl show <cell-hash>` and `--kind sim` over a fan-out. A cell stops being
  addressable.
- A CAS cache hit on a single replicate. Re-running a fan-out re-runs all cells
  rather than reusing a subset.

Both are acceptable **for replicates**, which is what the fan-out on this store
actually is: 1,200 realisations of one θ differing only in RNG seed. A replicate
is exchangeable — nothing asks for realisation #847 by name.

**This proposal does not extend that judgement to posterior draws.** A draw
carries a specific θ, which is a meaningful thing to address, and the `--draws`
path has not been measured here. If it fans out the same way, it needs its own
decision, not this one by analogy.

`camdl cat` is already correct: `browse.rs:856` defaults to `ensemble.tsv` for
new-format ensembles.

## Existing stores

Old fan-outs stay on disk and stay readable; new runs stop creating them. No
migration, and per the alpha posture no compatibility shim. But the 461k
existing leaves and ~17 GB do **not** go away on their own — removing them needs
the archive/prune surface, which is why that work should land first or
alongside. That is the only real coupling between the two.

## The downstream consumer, and the sequencing rule

`camdl 'scope` reads the fan-out directly and has **no code for `ensembles/`**:
`camdl_watch/sims.py:71` discovers runs with `rglob("run.json")` under `sims/`
and `sims.py:129` reads each leaf's `traj.tsv`; the only two matches for
"ensemble" in that repository are prose in comments. It reconstructs the
ensemble view itself, computing quantiles across the per-cell members
(`camdl_watch/api/models.py:559`).

So the viewer's change is a simplification — read one `ensemble.tsv` instead of
1,200 leaves — but it is a change, in a separately installed and separately
versioned repository.

**Sequencing: file the camdl-scope issue only once this work is complete and on
`main`.** Filing it earlier invites the viewer to change against a camdl that
has not shipped, which is the one ordering that can break `'scope` for a user
who upgrades one side. The issue should state the exact artifact
(`ensembles/**/ensemble.tsv`, with its `replicate` column), the fact that
one-off sims are unaffected, and the camdl version the change lands in.

## Verification

- **A one-off `camdl simulate` is byte-identical** — same leaf, same `traj.tsv`,
  same `run_id`. This is the sharpest available oracle for the scoping claim and
  must be a committed test, not an inspection.
- **A multi-cell run's ensemble `run_id` is byte-identical to today's** for the
  same inputs. This pins the hash-neutrality claim; without it the change
  silently re-keys every ensemble.
- **`ensemble.tsv` is byte-identical** before and after, for the same seed — the
  rows are the same rows, produced by a different route.
- The fan-out writes **no** `sims/` leaves, verified by leaf count on a fixture
  store rather than by absence of an error.
- `camdl cat <ensemble>` and the `-o` combined TSV still agree, which
  `sim_ensemble_cas.rs` states as an invariant today.

## Not this proposal

Compressing artifacts on disk (gh#698); bounding `list`'s discovery (gh#699),
which helps every store including ones already full of fan-outs; the
archive/prune surface that removes the existing leaves; and the `--draws` path,
which is unmeasured here.
