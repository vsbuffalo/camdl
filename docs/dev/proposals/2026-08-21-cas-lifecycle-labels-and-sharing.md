# CAS lifecycle: a `store` namespace, labels that move, and a bundle you can send

Date: 2026-08-21 Status: **UNDER REVISION — do not implement as written.** Two
adversarial reviews returned two KILLs and six severe gaps against the code. The
measured motivation below stands; the label mechanism does not. Related: gh#594,
gh#698, gh#699, gh#701, gh#704. Prior art this document failed to cite and must
be rewritten against:
`docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md`
(`Status: proposed`, not archived) — origin of the `@label` sigil, implemented
at `cli/src/fit/handle.rs:44` and normative in `docs/camdl-run-spec.md`.

> ### What review killed
>
> **The single `<root>/labels.json` widens the blast radius of a lost update
> from one run to every name in the store.** tmp+rename is crash-atomicity, not
> mutual exclusion; two concurrent `label` calls silently drop one binding.
> Today's per-leaf write loses one run and says so (`fit/mod.rs:2483-2486`). The
> `index.json` analogy is wrong because that file is _derived_ — `rebuild()`
> reconstructs it from a walk — and `labels.json` would have no rebuild source
> once the in-leaf field is deleted. Git itself does not do this:
> `refs/heads/<name>` is one file per ref under a per-ref `.lock`. Per-name
> files under `<root>/labels/` fix this and the "one `rm` destroys every name in
> a gitignored tree" hazard together.
>
> **"One run per label" cannot represent what `--label` already does.**
> `batch.rs:1394` calls `ensure_provenance_label` per cell, and `main.rs:2001`
> labels the ensemble too, so `simulate --label X --draws 1200` over 5 scenarios
> stamps one label onto **6,001 objects**. The claim that "argv is already in
> provenance" is also false on a CAS **cache hit**, where the leaf's argv
> belongs to the original producing invocation — which is precisely why
> `ensure_provenance_label` exists, with red-green tests at
> `tests/cas_integration.rs:1528` and `tests/pfilter_cas.rs:237`.
>
> ### Claims in the body below that are wrong
>
> - **`camdl 'scope` is NOT unaffected.** `camdl_watch/ingest.py:440` shells out
>   to `camdl list --root … --kind fit --format json` and reads each row's
>   `label`, returning `{}` on any non-zero exit — so the `camdl store` rename
>   degrades the viewer to derived labels with **no error**.
> - **`@label` already exists and already refuses ambiguity**, listing
>   candidates via `ResolveError::Ambiguous` (`handle.rs:264-292`). So
>   non-uniqueness today produces a refusal, not the "wrong-fit-packaged bug"
>   claimed below, and the proposed `label:` / `hash:` escapes reinvent a
>   shipped sigil — with `:` colliding with `--exclude-chains @a:4`.
> - **"No precedence" resolution is O(store).** Proving a token has no _second_
>   match requires enumerating, and `cas_index::resolve_prefix` returns `None`
>   for any prefix under 64 hex by design (`cas_index.rs:145`), so every label
>   lookup would take the >120 s walk. `@name` is one file read.
> - **The label is not stored inside a content-addressed leaf for fits.**
>   `fit.meta.json` sits at the _segment_, is not a CAS leaf, and is
>   deliberately mutable with sticky-label semantics normative at
>   `docs/camdl-run-spec.md:360-361`. The real defect there is that
>   `write_fit_sidecar` ends in a bare `std::fs::write` (`run_meta.rs:645`), so
>   `camdl label` on a fit can leave a torn file today.
> - **`/` is gated by `validate_label` (`fit/mod.rs:2451`), not `path_label`.**
>   The current charset is `^[a-zA-Z0-9 ,._-]{1,64}$` — spaces and commas legal,
>   `/` rejected — and `--help` actively teaches sentence-shaped labels.
> - **Stem resolution is ambiguous by construction**, since the body itself
>   notes three `fit_national_base_rho_high-*` directories in one store.
> - **A fit directory is not relocatable across machines.**
>   `load_config_for_segment` (`handle.rs:175-198`) anchors the archived
>   `fit.toml.original`'s relative paths at `FitSidecar.fit_toml_path`, recorded
>   verbatim as typed (`fit/mod.rs:2200`). `pack` must re-point that anchor or
>   `fit summary` / `fit predict` / `compare` fail on the recipient's machine.
>
> ### Gaps that must be decided before any rewrite ships
>
> - **`prune` is the first destructive verb outside the store's lock protocol**
>   (`store.rs:378-392`, `:842-890`). An archived leaf can be re-claimed by a
>   resumed fit at any time, so prune can delete a directory being streamed
>   into.
> - **Archiving must prune the walk, not filter it.** Keyed on `run_id`,
>   `archived.json` requires reading all 550k `run.json` files to apply — so
>   archiving 6,000 leaves would leave `list` exactly as slow. It has to key on
>   the store-relative directory prefix.
> - **An archived run that is re-run stays hidden**, because
>   `FsCasStore::lookup` (`store.rs:202-228`) consults only the leaf's own
>   `run.json` and cannot see a store-level set.
> - **Whole-leaf-only prune leaves ~4 empty ancestors per leaf** (2,735,850 dirs
>   for 550,647 leaves), which `cas_read::walk_records` still descends — so it
>   does not deliver the browsing relief it exists for. Ancestor removal races
>   `fs::create_dir_all` at `store.rs:376`.
> - **`batch run` writes no ensemble leaf at all** (`resolve_sim_ensemble` is
>   reached only from `simulate`, `main.rs:2001`), and even `simulate`'s
>   ensemble write is best-effort (`main.rs:2556-2561`). So "one selector
>   archives the whole fan-out" has no guaranteed object behind it.
> - **Reference edges prune would dangle**: ensembles hash member `sim_run_id`s
>   into their own identity (`sim_ensemble_cas.rs:16-26`), leaves declare child
>   artifacts (`batch.rs:1296`), and `lineage realize` takes a raw path into a
>   leaf.
> - **The migration harvest cannot run after the field is deleted** —
>   `RunRecord` has no `deny_unknown_fields` (`record.rs:212`), so the label is
>   silently dropped on read from that moment.
> - **`list --format json` publishes `provenance.label`** for four run kinds
>   (`browse.rs:1497`, `:1504`, `:1588`, `:333`) — a published contract, not an
>   internal field.
>
> ### What the rename actually costs
>
> 191 references across 34 `docs/**.md` files (51 in `docs/camdl-run-spec.md`
> alone, whose §4.5 is a **named normative section**), 24 integration-test
> shell-outs across 21 files, four docs baked into the binary via `include_str!`
> (so `camdl docs agents` ships stale text until rebuilt), 35 references in
> `../camdl-book`, and one cross-repo consumer that cannot be fixed atomically.
>
> ### Sound as written, and to be carried forward
>
> The two-step archive-then-prune shape; refusing to fold foreign leaves into
> the local store; a visible `./camdl-inbox/` rather than a temp directory;
> recomputing digests on unpack while explicitly not claiming to verify the run
> was correctly produced; printing the hash beside the label in every artifact
> camdl writes; and the measured motivation in the next section.

## What a real store looks like

The ebola-bdbv project's `results/` tree, measured:

| kind      |      dirs | leaves (`run.json`) |
| --------- | --------: | ------------------: |
| sims      | 2,735,850 |             550,647 |
| ensembles |       262 |                  88 |
| pfilters  |       227 |                  73 |
| fits      |       654 |                  71 |
| surveys   |         7 |                   2 |

**4.1 million files, 28 GB, and no run of any kind carries a label.** A single
`simulate` over a posterior writes 6,000 leaves — 1,200 draws × 5 scenarios, one
leaf per cell — so sims are 99.96% of the store and everything a person actually
browses is the other 1,156 directories.

Three things follow, and they are what this proposal is for. Nothing retires a
run, so the tree only grows. Nothing addresses a run by a name a human chose, so
every reference is an eight-hex prefix. Nothing packages a run to send, so
sharing is `tar` with the store's guarantees left at the door.

Two more measurements that shape the decisions below:

- A fit directory is **37 MB**, of which `trajectories.tsv` is **15 MB** and
  `resume_state.bin` a further 0.2 MB. `camdl 'scope` reads **neither** — zero
  references in `camdl_watch/`.
- Fit directories carry **no absolute paths** in `run.json`, `fit.toml.original`
  or `fit.meta.json`. A fit directory is relocatable as-is.

The store API (`runid::store`) is `new`, `lookup`, `commit_atomic`,
`claim_streaming`, `dir`, `write`, `finalize`. **There is no delete, archive,
prune or gc.**

## 0. One namespace: `camdl store`

Today the store verbs are scattered across the top level (`list`, `show`, `cat`,
`label`) and one is filed under `dev` (`reindex`), while the top level also
holds the modelling verbs. `camdl list` does not say list _what_.

**The seam: `camdl store` holds verbs whose subject is the store as a container
of runs — find them, name them, retire them, move them in and out. Verbs that
_produce_ or _analyse_ runs stay top level, even when they address a run by
selector.**

| moves to `camdl store`                    | stays top level                       |
| ----------------------------------------- | ------------------------------------- |
| `list`, `show`, `cat`, `label`            | `simulate`, `batch`, `fit`, `pfilter` |
| `dev reindex` → `store reindex`           | `profile`, `survey`, `compare`        |
| new: `archive`, `prune`, `pack`, `unpack` | `check`, `inspect`, `render`, `data`  |

`compare` stays out because its subject is fits' predictive scores, not the
store; `fit` takes a path without being a "path" command, and `compare` takes
selectors without being a store command. `mre` stays out for the reason it was
built: its subject is a bug report, and its output is not a store artifact.
`reindex` moves out of `dev` because once `prune` exists, reindexing is part of
the ordinary lifecycle rather than a maintenance escape hatch.

Per the alpha posture there is no alias for the old spellings; the rename is
atomic and the docs move with it.

## 1. Labels move, and that is the point

The motivating cases are a _front-runner_ model that keeps being the
front-runner as it changes, and a _best-elpd_ that moves when a better fit
lands. Both are **pointers that move**. In git terms that is branch behaviour,
not tag behaviour: a tag is conventionally pinned, and moving one is an
antipattern precisely because tags get published and cited. So camdl should take
git's _storage_ model and reject its _immutability_ convention.

What git gets right and camdl should copy is where the name lives. A tag sits in
`refs/tags/`, beside the object store, never inside the object. camdl gets one
of the three properties right and two wrong:

**Right — a label is never hashed.** `RunProvenance` is documented "always
skipped from the hash … must never be hashed" (`runid/src/inputs.rs:352`).
Labelling cannot change a `run_id`, and this proposal does not change that.

**Wrong — the label is stored inside the object it names.** `cmd_label`
(`cli/src/fit/mod.rs:2488`) rewrites the committed leaf: `write_fit_sidecar` for
a fit segment, a read-modify-`rename` of `run.json` for everything else. Naming
a run mutates a content-addressed artifact after commit.

**Wrong — nothing enforces uniqueness.** `cmd_label` resolves its _target_ by
hash prefix and writes; it never scans other runs. Two runs can hold one name.
Inert while a label is display text; a wrong-fit-packaged bug the moment `pack`
accepts one.

One more fact, because the directory names invite the opposite belief.
`results/fits/fit_national_base-1c52e37a` is **not** a label: it is
`runid::layout::path_label(stem)` — the fit.toml's _filename stem_, lowercased
with non-`[a-z0-9_.-]` replaced by `_` — plus eight hex of the fit hash
(`fit/cas.rs:419`). Stems are not unique; that store has three
`fit_national_base_rho_high-*` directories. The visible prefix is not an
address.

### Decision: a store-level ref map

`<root>/labels.json`, a map from label to
`{run_id, kind, created, message?,
history[]}`, written atomically (tmp +
rename), beside `index.json`.

- **Uniqueness is structural** — a map key, not a check someone forgets.
- **Leaves become immutable again after commit.** Naming stops rewriting.
- `--list`, `-d` and moves become one-file operations, safe to run against a
  store while a fit streams into it.

The leaf's `label` field goes away: redundant for a run-time `--label` (argv is
already in provenance) and wrong for a post-hoc one. `camdl store reindex`
harvests any in-leaf labels once. **No `run_id` changes**, because the field was
never hashed. `camdl 'scope` is unaffected — it reads `levels[].label`, the
_factored level_ label of `record.rs:34`, and does not read `provenance.label`
at all (`camdl_watch/sims.py:77`).

### Decision: moving is ordinary, and recorded

`camdl store label front-runner <selector>` on a name already in use **moves
it**, with no flag, and prints what it moved:

    front-runner: 1c52e37a → 8f67d9fb  (previous binding kept in history)

A `--force` gate was considered and rejected. The motivating use _is_ moving, so
a gate would be typed on nearly every invocation, and a flag typed reflexively
stops guarding the case it exists for — the accidental reuse of a name you
forgot was taken. The printout is the guard, and
`camdl store label --history
front-runner` makes the accident recoverable rather
than merely forbidden. For a local research store, recoverable beats prevented.

**The reproducibility hazard this creates is real and is closed elsewhere.** If
a report says "see fit `front-runner`" and the label later moves, that citation
silently means something else. The fix is not friction at assignment; it is that
**every durable artifact records the hash alongside the label** — pack
manifests, `store show`, exported summaries all print `front-runner (1c52e37a)`.
The rule, in one line for the docs: **name runs to find them; cite runs by
hash.**

### Decision: many labels per run, one run per label

The map direction makes many-to-one free, and it is what git allows too — a
commit can carry `v1.0` and `stable` at once. `best-elpd` and `front-runner`
pointing at the same fit is a normal and informative state. The constraint runs
the other way: one name resolves to one run. `store list` shows labels in a
column, truncating past two with `+N`.

### Decision: no hex ban — no precedence instead

An earlier draft refused pure-hex labels so they could not be confused with hash
prefixes. That is the wrong shape: it bans legitimate names (`cafe`, `face`,
`beef`) to pre-empt a collision that has not happened.

**Resolution has no precedence order.** A token is looked up as a label, a stem,
and a hash prefix; if it matches exactly one run it resolves, and if it matches
more than one it **errors and lists the candidates** with hash, kind and
creation time. It never picks.

This is simpler to explain than an order, safer than either order, and makes the
hex question disappear: a hex-shaped label is fine until an actual hash prefix
collides with it, at which point the error names the escape —

    error: 'cafe' is ambiguous
      label  cafe        → fit 8f67d9fb  bvd_national_base       2026-08-19
      hash   cafe1d20…   → sim           bvd_national_delay      2026-08-20
    disambiguate with 'label:cafe' or 'hash:cafe1d20'

`/` is allowed and encouraged for namespacing (`review/national-v1`). Labels
never appear in filesystem paths under this design, so the sanitising that
governs `path_label` does not apply to them. `..` and leading/trailing `/` are
refused.

## 2. Archive and prune

**Archive is a reversible state, not a deletion.** An archived run stays on
disk, stops appearing in `store list` and in `'scope`, and becomes the unit
`prune` operates on. It is what lets someone clear a browsing surface without
deciding, in that same moment, that a run is worthless.

```
camdl store archive <selector>...          # hide from list/'scope, keep on disk
camdl store archive --undo <selector>...
camdl store list --archived
camdl store prune --archived [--older-than 30d] [--dry-run]
```

### Decisions

- **Archive state is a store-level set** (`<root>/archived.json`), for the same
  reason labels are: a committed leaf is not rewritten again. A marker file
  inside the run would reintroduce the exact mutation this removes from
  labelling.
- **`prune` refuses anything not archived.** Two steps, deliberately: archive is
  reversible and cheap, prune is neither. `--dry-run` prints what would go and
  its total size.
- **Prune is whole-leaf only.** Removing part of a leaf leaves a directory that
  passes a path check and fails a content check — the store's worst state. If a
  leaf cannot be removed entirely, it is not removed.
- **Pruning clears the run's labels and archive entry, then reindexes.** A
  dangling ref and a stale `index.json` are precisely the confusion this feature
  exists to reduce.
- **Archived runs stay readable by hash.** `store show <hash>` works and says
  the run is archived. Hiding is a browsing concern, not an access one.
- **Archiving a sim archives its whole ensemble fan-out**, all 6,000 leaves, as
  one operation against one selector. Anything that made a user archive cells
  individually would be unusable on this store.

## 3. Pack and unpack

```
camdl store pack <selector>... -o review.camdl-fits.tar.zst [--with-paths] [--no-data]
camdl store unpack review.camdl-fits.tar.zst [--into DIR]
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
terminal states). That is the trade, and the manifest states it, so the
recipient learns the limitation from the artifact rather than from an error.

`resume_state.bin` never travels: per-machine continuation state, not a result.

### Labels travel, namespaced by the bundle

Git tags do not auto-push because a repository is a **shared, long-lived
namespace** that other people pin to; withholding is the safe default there. A
camdl bundle is the opposite: a **point-to-point handoff**, where the label is
often the entire content of the message — "look at `front-runner`". A recipient
who gets `fit_national_base-1c52e37a` and no name has lost the one piece of
information the sender most wanted to convey. So camdl departs from git here:
**labels travel by default.**

Collisions are avoided rather than resolved. **Incoming labels land under the
bundle stem** — `front-runner` arrives as `review-2026-08/front-runner`. Nothing
can clash with a name the recipient already uses, the provenance is legible in
the name itself, `store label --list 'review-2026-08/*'` shows exactly what
arrived, and two colleagues' bundles never collide with each other. `--flat`
opts out and applies at top level, refusing-and-reporting on a real collision.

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
- Moving a label succeeds without a flag, prints both bindings, and `--history`
  recovers the previous one.
- Two labels on one run both resolve; one label on two runs is unrepresentable.
- An ambiguous token exits non-zero listing candidates, and the `label:` /
  `hash:` escapes each resolve to exactly one.
- `prune` on an unarchived run refuses; on an archived one it removes the leaf,
  clears its refs, and leaves `index.json` consistent (`store list`, then
  `store show` on the removed hash, both behave).
- Archiving a 6,000-leaf sim by one selector hides all of it from `store list`
  and from `'scope`, and `--undo` restores it.
- A pack/unpack round-trip on a store copy reproduces every packed file
  byte-identically; a bundle with one corrupted byte fails.
- An incoming label lands namespaced, and a bundle unpacked twice into one store
  does not produce two conflicting refs.

## Not this proposal

Compressing artifacts on disk (gh#698 — needs the benchmark first); the
store-root model archive being last-writer-wins (gh#594); and the two questions
in gh#699 — bounding `list`'s discovery by `--limit`/`--since` rather than
filtering after it, and whether a posterior ensemble should be 6,000 store
leaves at all. Archiving makes that fan-out survivable; it does not make it
right, and the compact `ensembles/` kind already suggests the other answer.
