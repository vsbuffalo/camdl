# CAS lifecycle: labels as refs, fits you can retire, and a bundle you can send

Date: 2026-08-21 Status: proposed Related: gh#594 (store-root model archive is
last-writer-wins), gh#698 (gzip benchmark for `trajectories.tsv`), gh#699
(`camdl list` does not complete on a 64-fit store)

## Three problems, one surface

A results store accumulates. The ebola project's holds **64 fit directories**.
Nothing in camdl retires a run, nothing addresses a run by a name a human chose,
and nothing packages a run to send. Today those three are `rm -rf`, a hash
prefix, and `tar` — with the store's own guarantees left at the door.

Measured, not asserted:

- `camdl list --root results` on that store **did not complete within 120
  seconds**. Browsing already does not scale to the number of fits a real
  project produces, which is the strongest argument that retiring runs is a
  lifecycle need and not tidiness.
- The store API (`runid::store`) is `new`, `lookup`, `commit_atomic`,
  `claim_streaming`, `dir`, `write`, `finalize`. **There is no delete, archive,
  prune or gc.** `camdl dev` offers `reindex`, `eval`, `compile`, `doctest`.
- A fit directory is **37 MB**, of which `trajectories.tsv` is **15 MB** and
  `resume_state.bin` a further 0.2 MB. `camdl 'scope` reads **neither** — zero
  references in `camdl_watch/`.
- Fit directories carry **no absolute paths** in `run.json`, `fit.toml.original`
  or `fit.meta.json`. A fit directory is relocatable as-is.

## 1. Labels are git tags, and today they are not stored like one

The design target is exactly git's: **a tag is a name kept beside the object
store that points at an immutable object; it is not part of the object, and it
is unique within the repository.** camdl gets the first half right and the
second two wrong.

Right: a label is already outside identity. `RunProvenance` is documented
"always skipped from the hash … must never be hashed"
(`runid/src/inputs.rs:352`). Labelling cannot change a `run_id`, and this
proposal does not change that.

Wrong, first: **a label is stored inside the object it names.** `cmd_label`
(`cli/src/fit/mod.rs:2488`) rewrites the committed leaf — `write_fit_sidecar`
for a fit segment, or a read-modify-`rename` of `run.json` for everything else.
A content-addressed artifact is mutated after commit, every time someone names
it. Git does not put the tag in the commit.

Wrong, second: **nothing enforces uniqueness.** `cmd_label` resolves its target
by hash prefix and writes; it never scans other runs. Two runs can hold one
label. That is inert while a label is display text and becomes a
wrong-fit-packaged bug the moment `pack` accepts one.

A third fact is worth stating because the directory names invite the opposite
belief. `results/fits/fit_national_base-1c52e37a` is **not** a label: it is
`runid::layout::path_label(stem)` — the fit.toml's _filename stem_, lowercased
with non-`[a-z0-9_.-]` replaced by `_` — plus eight hex of the fit hash
(`fit/cas.rs:419`). Stems are not unique; that store has three
`fit_national_base_rho_high-*` directories. So the visible prefix cannot be an
address, and no run in that store carries a label at all.

### Decision: move labels to a store-level ref map

`<root>/labels.json`, a map from label to `{run_id, kind, created, message?}`,
written atomically (tmp + rename), sitting beside `index.json`.

- **Uniqueness is structural**, not a check someone can forget: it is a map key.
- **Leaves become immutable again after commit.** Naming a run stops rewriting
  it.
- `--list`, `--delete` and move-with-`--force` become one-file operations rather
  than leaf rewrites, which also makes them safe to run against a store while a
  fit is streaming into it.

The dedicated `label` field in the leaf goes away. It is redundant for a
run-time `--label` (argv is already recorded in provenance) and wrong for a
post-hoc one. Per the alpha posture there is no compatibility shim:
`camdl dev
reindex` harvests any in-leaf labels into the map once. **This does
not change a single `run_id`**, because the field was never hashed.

`camdl 'scope` is unaffected: it reads `levels[].label` — the _factored level_
label of `record.rs:34`, a different thing — and does not read
`provenance.label` at all (`camdl_watch/sims.py:77`).

### Decision: the surface, and the naming rules git needed too

```
camdl label <name> <selector>     # create; refuses if taken
camdl label <name> <selector> -f  # move (clears the previous holder)
camdl label --list [glob]
camdl label -d <name>
```

- **Resolution is uniform wherever a run is named** — label, then stem, then
  hash prefix. An ambiguous token errors and **lists the candidates** with hash,
  kind and creation time; it never picks one.
- **A label that is pure hex of four or more characters is refused.** It would
  make that resolution order load-bearing rather than a convenience. Git guards
  the same hazard for branch names.
- **`/` is allowed and useful** — `review/national-v1` namespaces the way git
  tags do. Labels never appear in filesystem paths under this design, so the
  path-sanitising constraint that governs `path_label` does not apply to them.
  `..` and leading/trailing `/` are refused.
- Labels stay optional. The fallback is what people use now — the stem+hash
  directory name — and it stays addressable, so nothing that works today stops.

## 2. Archive and prune

**Archive is a reversible state, not a deletion.** An archived run stays on
disk, stops appearing in `camdl list` and in `'scope`, and becomes the unit
`prune` operates on. It is what lets someone clear a browsing surface without
deciding, in that same moment, that a run is worthless.

```
camdl archive <selector>...          # hide from list/'scope, keep on disk
camdl archive --undo <selector>...
camdl list --archived
camdl prune --archived [--older-than 30d] [--dry-run]
```

### Decisions

- **Archive state is a store-level set** (`<root>/archived.json`), for the same
  reason labels are: a leaf that has been committed is not rewritten again. A
  marker file inside the run would reintroduce exactly the mutation this
  proposal removes from labelling.
- **`prune` refuses anything not archived.** Two steps, deliberately: archive is
  reversible and cheap, prune is neither. `--dry-run` prints what would go and
  its total size.
- **Prune is whole-leaf only.** Removing part of a leaf leaves a directory that
  passes a path check and fails a content check — the store's worst state. If a
  leaf cannot be removed entirely it is not removed.
- **Pruning clears the run's labels and archive entry**, and then reindexes. A
  dangling ref and a stale `index.json` are precisely the confusion this feature
  exists to reduce.
- **Archived runs stay readable by hash.** `camdl show <hash>` works and says
  the run is archived. Hiding is a browsing concern, not an access one.

## 3. Pack and unpack

```
camdl pack <selector>... -o review.camdl-fits.tar.zst [--with-paths] [--no-data]
camdl unpack review.camdl-fits.tar.zst [--into DIR]
```

`camdl mre fit` already bundles a fit's **input closure** so a recipient can
re-run it; that is a bug-report tool and stays. This is its complement — the
**outputs**, so a recipient can look without re-running.

### What travels, and the one real trade

**Default is view-complete**: posterior draws, per-chain traces, quantities,
predictive, observed data, model source and IR, and the metadata sidecars —
about 22 MB raw for that ebola fit, roughly 7 MB compressed.

**`--with-paths` adds `trajectories.tsv`** (15 MB). Without it the recipient can
view everything but cannot fork counterfactuals (`contrasts {}` needs saved
paths) or forecast from the fitted state (`simulate --init-state fit` needs the
terminal states). That is the trade, and the bundle manifest states it, so the
recipient learns the limitation from the artifact rather than from an error.

`resume_state.bin` never travels: per-machine continuation state, not a result.

**Labels travel as refs, and apply only if free** — git's `--tags` behaviour.
The bundle carries its own `labels` section; on unpack, a free label is applied,
a taken one is reported by name with both run ids and left unapplied. The
recipient's names are theirs.

### Where it lands, and why not the results tree by default

**`--into` defaults to `./camdl-inbox/<bundle-stem>/`, created if absent.**

A temporary directory is rejected outright. The users here are epidemiologists,
and an artifact under `/var/folders/…` is one they will not find again — nor
will they think to move it before it is cleaned up. A visible directory in the
working directory, named after the bundle, is `ls`-able, greppable, and
survives.

**Folding into the local `results/` is opt-in, never the default.** The store's
promise is that a `run_id` is a function of its inputs. A foreign leaf sitting
in your store asserts an identity your inputs did not produce and that you
cannot verify locally — you do not have the sender's data to recompute against.
Two stores with the same layout are not one store.

Consequences, decided:

- **`unpack` recomputes every content digest the bundle's own `run.json`
  records, and refuses on mismatch.** That verifies the bundle arrived intact.
  It does **not** claim to verify the run was correctly produced, and the
  manifest says so in those words.
- **Clash on `run_id` in the target: refuse and report.** The same id from a
  different sender is either the same run (nothing to do) or something worth
  seeing. `--force` overwrites.
- **Clash on directory name with a _different_ id: suffix the directory.** Never
  merge two leaves.
- The manifest records sender, camdl version, `ir/VERSION`, and the selector
  used. A bundle from a different `ir/VERSION` unpacks with a loud note: the
  artifacts are readable, but re-running against them may not reproduce.

## Verification

- Labelling a run leaves every byte of its leaf unchanged, and its `run_id`
  unchanged — the sharpest available oracle for the ref move, and cheap.
- A second `camdl label` with a taken name exits non-zero, names the holder, and
  writes nothing.
- A pure-hex label is refused; an ambiguous selector lists candidates and exits
  non-zero rather than resolving.
- `prune` on an unarchived run refuses; on an archived one it removes the leaf,
  clears its refs, and leaves `index.json` consistent (`camdl list` then
  `camdl
  show` on the removed hash both behave).
- A pack/unpack round-trip on a store copy reproduces every packed file
  byte-identically, and `unpack` of a bundle with one corrupted byte fails.
- Unpacking into a store that already holds one of the run ids refuses by
  default and overwrites under `--force`.

## Not this proposal

Compressing artifacts on disk (gh#698 — needs the benchmark first); the
store-root model archive being last-writer-wins (gh#594); and making listing
fast on a large store, which archiving mitigates but does not fix, and filed as
gh#699 with a profile named as the first step.
