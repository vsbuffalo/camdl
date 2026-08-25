# `camdl store`: a lifecycle for a store whose steady state is debris

Date: 2026-08-24 Status: proposed Supersedes:
`docs/dev/proposals/2026-08-21-cas-lifecycle-labels-and-sharing.md` (which
carried two review KILLs and was never implemented; its measurements are carried
forward here, its label mechanism is not). Related: gh#594, gh#699, gh#701,
gh#704.

## What this document assumes you have read

Nothing, but two existing designs constrain it and are cited where they bind:

- **`docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md`**
  (`Status: proposed`) introduced the `@` handle sigil and the rule that
  resolution is a fallible boundary with ambiguity as a typed outcome. It is
  implemented at `rust/crates/cli/src/fit/handle.rs:43` (`FitRef::classify`) and
  normative in `docs/camdl-run-spec.md` §4.5. **This proposal keeps the sigil
  and the classifier and changes only what the `@` branch reads.**
- **`docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md`** (landed
  in part) gave the store an _augment door_ — a `Completed` leaf may gain an
  artifact under its own lock, identity-checked — and made "a hash that may
  become an identity level" a type (`LevelHash`,
  `rust/crates/cli/src/fit/cas.rs:106`). Those two facts change what a lifecycle
  surface may assume, and both are used below.

Three terms, since they are used throughout. A **leaf** is one content-addressed
run directory: the thing holding a `run.json`. A **segment** is a fit's
top-level directory (`results/fits/<stem>-<hash8>/`), which holds
`fit.meta.json` and the per-stage leaves beneath it but is _not_ itself a leaf.
A **selector** is whatever a user types to name a run — a path, a hash prefix,
or an `@name`.

## The store's steady state

Measured on the ebola-bdbv project's `results/` tree:

| kind      |      dirs | leaves (`run.json`) |
| --------- | --------: | ------------------: |
| sims      | 2,735,850 |             550,647 |
| ensembles |       262 |                  88 |
| pfilters  |       227 |                  73 |
| fits      |       654 |                  71 |
| surveys   |         7 |                   2 |

4.1 million files, 28 GB, and **no run of any kind carries a label**. Sims are
99.96% of the store; everything a person browses is the other 1,156 directories.
A fit directory is 37 MB, of which `trajectories.tsv` is 15 MB.

Three facts follow and they are what this proposal is for. Nothing retires a
run, so the tree only grows. Nothing addresses a run by a name a human chose, so
every reference is an eight-hex prefix. Nothing packages a run to send.

### Why the debris is not a failure of tidiness

This matters for the shape of the verbs, so it is worth grounding rather than
asserting. **Verified against the primary text** — Gelman, Vehtari, McElreath,
with Simpson, Margossian, Yao, Kennedy, Gabry, Bürkner, Modrák and Leos Barajas,
_Bayesian Workflow_ (CRC Press, 2026; corrected edition of 20 July 2026, the
book successor to arXiv:2011.01808). I read §§9.1–9.3, 10.3, 12.1 and 15.2–15.4
of that edition; the quotations below are transcribed from it, not relayed.

§9.3 states the premise plainly: "The key aspect of Bayesian workflow, which
takes it beyond Bayesian data analysis, is that we are fitting many models while
working on a single problem. We are not talking here about model selection or
model averaging, but rather of the use of a series of fitted models to better
understand each one." Its list of reasons includes "When constructing models, we
make a lot of mistakes: typos, coding errors, conceptual errors," and — the line
that decides `archive` versus delete — "We'll check a model, find problems, and
then expand or replace it. This is part of 'Bayesian data analysis'; the extra
'workflow' part is that **we still keep the old model**, not for the purpose of
averaging but for the purpose of understanding what we are doing."

§9.2 gives the navigation frame: "Our interest here is not in averaging over
models but in **navigating among them**," and, on the size of the space, "To
focus on the most relevant models, we can **filter out models with bad
performance or serious computational issues**." Filter out, not destroy.

§12.1 ("Fit fast, fail fast") supplies the production rate: the workflow is
explicitly organised to "waste as little time as possible on the models that we
will ultimately abandon," which is a design for generating abandoned fits
quickly.

And §15.2 supplies the argument for the immutable name, in words that are almost
a specification: version control "is particularly useful for its ability to
**package up and label 'release candidate' versions of models and data that
correspond to milestone reports and publications** and to store them in the same
directory without resorting to the dreaded `_final_final_161020.pdf`-style
naming conventions."

One further passage is load-bearing for the destructive verb and is easy to
miss. §9.3, on researcher degrees of freedom: "if we are not careful, we can
consider our inferences from a set of fitted models to bracket some total
uncertainty, **without recognizing that there are other models we could have
fit**." Deleting the failures is not neutral bookkeeping — it makes the
surviving set look more consensual than it was. That is a reason to make the
reversible step the default one and the irreversible step explicit, rather than
the other way round.

## Three defects, verified

### D-a. Naming a run rewrites a content-addressed artifact after commit

`cmd_label` (`rust/crates/cli/src/fit/mod.rs:2548`) writes the name **into the
object it names**. For a fit segment it rewrites `fit.meta.json` via
`write_fit_sidecar`; for a sim, pfilter or survey leaf it read-modify-renames
the leaf's `run.json` (`fit/mod.rs:2660`), under a doc comment that concedes
"Concurrent invocations are last-write-wins; we don't lock the file"
(`fit/mod.rs:2544`). `batch.rs:1515` (`ensure_provenance_label`) does the same
at write time, at both the commit site (`batch.rs:1480`) and the **cache-hit**
site (`batch.rs:1273`), so `simulate --label X --draws 1200` over 5 scenarios
rewrites 6,000 committed leaves plus the ensemble.

Neither changes a `run_id` — `Provenance` is recorded-not-hashed
(`runid/src/record.rs:135-154`, and `runid/src/inputs.rs:352` says it "must
never be hashed") — so this is not an identity bug. It is a durability and
contract bug: `write_fit_sidecar` ends in a bare `std::fs::write`
(`run_meta.rs:645`), so a `camdl label` on a fit can leave a torn
`fit.meta.json`; and the store's whole premise is that a `Completed` leaf's byte
set is fixed, which is why the augment door had to be designed as a locked,
identity-checked, divergence-detecting operation rather than a plain write.

### D-b. The name camdl asks for is not the name camdl resolves

This is the mechanism behind "zero labels in a 28 GB store," and it is visible
in the argument definitions alone.

`--label`'s help (`args/mod.rs:963-969`) reads: "User-supplied **display label**
for this fit (1–64 chars after trim; allowed: letters, digits, spaces, commas,
dot, underscore, hyphen). … Examples: `--label "narrow R0, take 1"`,
`--label "iota free"`, `--label "log_normal R0 prior"`." `validate_label`
(`fit/mod.rs:2451`) enforces exactly that charset — spaces and commas legal, `/`
rejected. So the tool teaches a **sentence**.

`handle.rs:133-141` then resolves `@name` by scanning fit sidecars for a `label`
equal to `name`. The handle over a sentence is `@narrow R0, take 1`, which is
not a usable shell token, and the hint camdl prints to the very users who were
about to label something names a command that does not exist (gh#701:
`fit_table.rs:203` prints `camdl fit label`, but the command is top-level
`camdl label`).

**One word is doing two jobs.** A description ("what was I trying, that day")
and an address ("what do I call this from now on") have different charsets,
different uniqueness requirements, different lifetimes, and different homes.
Conflating them produced a feature that is impossible to adopt, and no amount of
ergonomic polish on `camdl label` fixes it.

### D-c. The store has no way to retire anything

```
$ rg -n "fn delete|fn prune|fn archive|fn gc\b" rust/crates/runid/src/store.rs
(no matches)
```

The public store API on `main` (HEAD `b3666ba6`) is `new`, `lookup`,
`displace_completed`, `augment`, `commit_atomic`, `claim_streaming`, `dir`,
`write`, `finalize`. `displace_completed` is the closest thing to a removal and
it **quarantines** rather than deletes (`store.rs:263-273`) — so `.quarantine/`
also grows without bound, and nothing collects it either.

## The design, in one paragraph

Split the overloaded word into three things that do not overlap. `label` stays
what it is — free display prose captured at run time, in the record, never
addressable. A **pin** is an immutable name bound once to one run, stored as one
small file outside every leaf, created with `O_EXCL` so no lost update is
representable; it is the citable handle and the thing `pack` accepts. A **mark**
is an explicitly moving pointer, stored the same way but append-only, so a move
never loses the binding it replaced. Separately, `archive` **relocates** a
subtree into `<root>/.archive/`, which the store walk already skips for free, so
archiving actually makes `list` faster instead of merely filtering its output;
`prune` is the only destructive verb, it refuses anything not archived and
anything a pin or mark currently points at, and there is no `rm`.

## 1. Two kinds of name

### 1.1 The concepts

An **immutable** name answers "what do I cite this as." It is bound once and
never rebound, so creating it is an atomic create-or-fail and the lost-update
problem does not exist — the kernel arbitrates. This is the `_final_final` cure
§15.2 describes.

A **moving** name answers "which one am I working with right now." Its
motivating case is a `front-runner` that stays the front-runner as it changes.

### 1.2 The naming call: `pin` and `mark`

The hazard is real and it cuts both ways, so here is the full comparison rather
than a bare choice.

| immutable / moving | why not                                                                                                                                                                                                                                                                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label` / `tag`    | Inverts git for **both** words at once. Anyone with git habits reaches for the wrong verb twice, and `label` is exactly the word that is already broken (D-b).                                                                                                                                                                                        |
| `tag` / `branch`   | Keeps git's spelling but imports two collisions. `Tags:` is already a **free-keyword** field in this project's own lab-note frontmatter, and every camdl model file carries a `# Base:` / `# Adds:` header describing a **branch of models** — so `camdl store branch` would name a fit pointer in the vocabulary already reserved for model lineage. |
| `tag` / `bookmark` | Mercurial precedent for a moving `bookmark` is correct, but `tag` still carries the free-keyword collision above.                                                                                                                                                                                                                                     |
| **`pin` / `mark`** | **Chosen.**                                                                                                                                                                                                                                                                                                                                           |

**Chosen: `pin` (immutable) and `mark` (moving), and this deliberately chooses
"what a modeller means" over "matches git's spelling."** The justification is
not that git is wrong but that both of git's words are already taken _inside
this project_ — `tag` by the lab-note keyword field, `branch` by model lineage —
and borrowing a word that already means something else locally is worse than not
borrowing.

What is borrowed is git's _meaning_, without its spelling. "Pinned" is the
ordinary English for a reference that does not move; it is how git's own
community describes what a tag is, and it is the same sense as a pinned
dependency version. So a git user's intuition transfers correctly and their
muscle memory has nothing to misfire on.

`mark` is chosen for the moving one because it has an exact, favourable
precedent in the maintainer's daily toolchain: a **Vim mark** is a named
position you jump to (`'a`) and re-set at will (`ma` overwrites it). Nobody
expects a Vim mark to be stable, which is precisely the property that must be
intuitive here.

There is one deeper reason to avoid `branch` beyond the collision. A git branch
moves as a **side effect of doing work** — you commit, and the head advances;
nobody types "move the branch." Nothing in camdl advances a pointer as a side
effect. Every rebind is a manual assertion, which makes the camdl operation
behaviourally `git tag -f`, not `git branch`. Naming it `branch` would promise a
safety property (the move is automatic and therefore honest) that the
implementation cannot provide.

### 1.3 Storage: one file per name

```
<root>/refs/pin/<name>       # exactly one line, written once
<root>/refs/mark/<name>      # append-only; the last line is the binding
```

Each line is five tab-separated fields:

```
<run_id 64-hex>	<kind>	<ISO-8601 UTC>	<store-relative leaf path>	<note>
```

The path is a **verified hint, never authoritative** — the same contract
`index.json` already carries (`cas_index.rs:1-25`): on read, the leaf's
`run.json` is re-read and its `run_id` re-checked, and a hint that does not
verify falls through to `cas_read::resolve_prefix_indexed` on the full id. It
exists only so the common resolution is two small file reads instead of an index
lookup.

### 1.4 Atomicity, and why the blast radius is zero names

The killed proposal put every name in one `<root>/labels.json`. Review's
objection was decisive and is worth restating because it is the constraint the
whole storage decision turns on: **tmp+rename is crash-atomicity, not mutual
exclusion**, so two concurrent writers silently drop one binding, and with a
shared file that costs _every_ name in the store rather than one.

Per-file refs make each operation atomic at the kernel, not by convention:

- **Create a pin** — `OpenOptions::new().write(true).create_new(true)`, i.e.
  `O_EXCL`. Two concurrent creates: one wins, the other gets `AlreadyExists` and
  reports which run already holds the name. **A lost update is
  unrepresentable**, which is the property the maintainer identified and it is
  the reason the immutable concept is worth having as its own thing rather than
  as a flag on a general one.
- **Move a mark** — open `O_APPEND|O_CREAT`, write one line, `sync_all`. POSIX
  makes the seek-and-write atomic with respect to other appenders. Two
  concurrent moves: **both lines land**; one is last; neither binding is lost,
  and the displaced one is recoverable by reading the file.
- **Remove a name** — `fs::remove_file`. One name.

So no operation in this design can lose a name at all — strictly better than
today's per-leaf write, which is last-write-wins and loses one run
(`fit/mod.rs:2544`), and categorically better than a shared map. Nothing here is
shared, so there is nothing to justify against a rebuild source.

A mark file whose **last** line is malformed is a hard error naming the file and
the line number, not a silent fall-back to the previous line. Falling back would
resolve the name to a stale run without saying so, which is the silent-wrong
class this store exists to prevent. Wedged-and-loud is the correct failure.

### 1.5 Charset, namespacing, and why no hex ban is needed

```
^[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*$    max 128 bytes
```

`/` namespaces, exactly as `refs/heads/feature/x` does in git, and nests as real
directories so a namespace is `ls`-able. Rejected: uppercase, `:`, `..`, empty
components, leading/trailing `/`, and any name ending `.lock`.

Two of those are not obvious. **Lowercase-only** is required because APFS is
case-insensitive by default and ext4 is not: allowing `Foo` would make `pin Foo`
after `pin foo` refuse on macOS and succeed on Linux, so create-or-fail would
mean different things on different machines. **`:` is rejected** because
`compare --exclude-chains @fit:ids` is shipped syntax
(`chain_selection.rs:136`), and a `:` inside a ref name would make that selector
ambiguous.

**No hex ban, and no ambiguity machinery.** The killed proposal proposed
precedence-free resolution — look a token up as a label, a stem and a hash
prefix and error if more than one matches — which review showed is O(store),
because proving there is no _second_ match requires enumerating. That whole
problem is an artefact of letting a bare token mean a name. It does not arise
here: `@` is mandatory for a ref, so `@cafe` is a ref and `cafe` is a hash
prefix, and neither can be the other. `cafe`, `face` and `beef` are legal ref
names.

### 1.6 Resolution

`FitRef::classify` (`handle.rs:43`) is kept **exactly as it is** — `@` → ref;
`*.toml` → config; an existing directory → run dir; else → hash prefix. Only
what the `@` branch _does_ changes: instead of `read_dir`ing `fits/` and reading
a sidecar per segment (O(#fits), and fits-only), it reads
`<root>/refs/pin/<name>`, then `<root>/refs/mark/<name>`. One file read, and it
works for every artifact kind rather than only fits.

A name may not exist as both a pin and a mark; creation checks the sibling
directory first. Two concurrent creates of the same name in different kinds can
still both succeed — an unavoidable two-file race with no kernel primitive
available. The blast radius is one ambiguous name and it is detected loudly at
resolution (both files present → error naming both and the runs they hold),
never resolved by a rule.

A **dangling** ref — the leaf was removed out of band — errors with the name,
the hash, and the remedy (`camdl store unpin <name>`). It never silently
resolves to nothing.

### 1.7 What `label` becomes, and what is deleted

`label` keeps exactly one job: **free display prose, captured at run time,
written into the record at commit, never addressable.** `--label` on
`fit`/`simulate`/`profile`/`survey` stays, unchanged, help text unchanged. It is
provenance, and provenance written as part of a commit is not a rewrite.

Deleted:

- **`cmd_label` and the top-level `camdl label` command** (`fit/mod.rs:2488`,
  `main.rs:577`). Post-hoc naming is what pins are for, and this is the site
  that rewrites a committed leaf. Its `validate_label` charset check stays,
  still serving the `--label` flags.
- **`ensure_provenance_label` and both call sites** (`batch.rs:1273`, `:1480`).
  Rewriting 6,000 committed leaves on a cache hit is the same defect at scale.

The consequence has to be stated rather than glossed, because it is a real
behaviour change: on a **cache hit**, a cell's `provenance.label` will name the
invocation that originally produced it, not the one that re-requested it. That
is correct — the leaf's provenance describes its production — and the user's
actual intent ("call this batch X") is served by pinning the ensemble, which is
one object rather than 6,001.

`simulate` writes exactly one `SimEnsemble` leaf for a multi-cell run
(`main.rs:2670`), so that object exists to pin. `batch run` writes **none** —
verified: `rg -n "ArtifactKind::SimEnsemble" rust/crates/cli/src/*.rs` matches
`cas_read.rs`, `browse.rs`, `main.rs` and `sim_ensemble_cas.rs`, and never
`batch.rs`. That gap is real and is named as a follow-up (F3) rather than
papered over; until it closes, a `batch` fan-out is archived and pinned by
subtree, which §2 supports natively.

**Cross-repo impact, checked.** `../camdl-watcher/camdl_watch/ingest.py:573`
(`_native_labels`) shells out to `camdl list --kind fit --format json` and reads
each row's `label`, returning `{}` on any non-zero exit — a silent degradation.
Deleting `cmd_label` does **not** break it: the `label` field stays in the JSON
row and stays populated by run-time `--label`. Post-hoc labels are the only
thing lost, and there are zero of those in the measured store. A `pins` field is
added to the same row so the viewer can adopt pins when it chooses to.

## 2. `archive`, `restore`, `prune`

### 2.1 Archive relocates; it does not mark

**Decision: an archived subtree is moved to
`<root>/.archive/<same relative
path>`.** There is no `archived.json`, no marker
file, and no per-leaf flag.

The reason is that a manifest cannot deliver the thing archiving exists for. The
walk's cost is not parsing — that was already fixed. `walk_gated`
(`cas_read.rs:176-215`) reads every `run.json`'s **bytes** and only then
decides, via the cheap `run_header`, whether to do the full parse. So an
`archived.json` keyed on `run_id` would still require opening 550k files to
apply, and archiving 6,000 leaves would leave `list` exactly as slow as before.
Review said this and it is correct.

Relocation prunes the walk instead of filtering it, and it does so through
machinery that already exists and costs nothing:

```
$ rg -n "starts_with\('\.'\)" -B2 -A2 rust/crates/cli/src/cas_read.rs
243-        if is_dir {
244:            if name.to_string_lossy().starts_with('.') {
245-                continue; // .staging / .quarantine
246-            }
```

`scan_dir` already skips every dot-prefixed directory from the directory
listing, before any `stat` and before any read. `.archive` joins `.staging` and
`.quarantine` as store machinery, `<root>/.archive` is documented alongside them
in run-spec §2.6, and archived leaves disappear from every walk — `list`,
`cas_index::rebuild`, and the per-kind discovery — for free.

The three properties that make this the smallest possible design fall out of it.
**A leaf's bytes never change**, so an archived run keeps its `run_id`, its
manifest and its exact-set integrity: a move is not a rewrite, which satisfies
the constraint that a committed leaf must not be rewritten. **There is no shared
state to lose**, so the blast-radius question does not arise — the archive's
layout _is_ the state. And **`list --archived` needs no new reader**: it is the
same walker pointed at `.archive`.

### 2.2 Re-running an archived run

Review flagged the sharpest consequence: `FsCasStore::lookup`
(`store.rs:217-242`) consults only the leaf's own `run.json` at the canonical
path, so a relocated leaf reads as `Miss`, and the run would be recomputed and
committed alongside the archived copy — two copies, growing disk, silently.

**Decision: `lookup` gains a fifth outcome, `Lookup::Archived(Box<RunRecord>)`,
and the restore happens on the write path, never inside `lookup`.** `FsCasStore`
already holds `root` (`store.rs:196-198`) and every caller derives the leaf path
from `layout::store_path(root, …)`, so `lookup` can strip the root and `stat`
`<root>/.archive/<rel>`. That is one extra `stat` on a `Miss` — the branch that
was about to do real work anyway.

`lookup` stays a pure read. The caller that is about to write matches on
`Archived` and restores (moves the subtree back, then re-runs `lookup` to
confirm a `Hit`). Putting the mutation there rather than in `lookup` keeps the
read honest and puts the policy where skip/force policy is already headed, per
S2 of the run-identity proposal.

### 2.3 Emptied ancestors, and the race

Archiving a leaf leaves its now-empty ancestors in the live tree — about four
per leaf, 2.7M directories for the measured store — which the walk still
descends. So archiving removes emptied ancestors up to the kind partition
(`sims/`, `fits/`, … ), which is never removed.

This races `fs::create_dir_all` in the commit path, and the race is benign in
both directions **because `fs::remove_dir` is used, never `remove_dir_all`**:
`remove_dir` only succeeds on an empty directory, so a concurrent writer that
has just populated it makes the removal fail with `ENOTEMPTY`, which is ignored;
and a writer that recreates a directory we just removed simply gets it back. The
worst outcome is a leftover or recreated empty directory. A leaf can never be
lost.

### 2.4 `prune` — and the answer on `rm` and `gc`

```
camdl store prune <selector>...            # only archived subtrees
camdl store prune --all-archived
camdl store prune ... --dry-run
camdl store prune --created-before 2026-06-01
```

**Is there a place for a direct `rm`? No.** Not because a two-step is
pedagogically nicer, but because two paths to one destruction drift. This
codebase has the case study: `--force` "has four different behaviors at five
call sites, none of them 'overwrite'" (run-identity proposal §S2), which is what
motivated collapsing every skip/force policy into a single store door. Adding
`rm` beside `archive`+`prune` would recreate exactly that shape for the one
operation where drift is unrecoverable. `archive` is cheap and reversible, so
the two-step costs nothing; `archive --undo` is the escape hatch a hasty `rm`
would otherwise need.

**Is `gc` the better name? No, and the reason is a promise the implementation
cannot keep.** `git gc` collects what is provably **unreachable** — you never
mark an object for deletion; you delete the ref and the object becomes garbage
by derivation. camdl cannot use that rule as its collection criterion, because a
leaf with no ref pointing at it is the _normal_ state (zero named runs in a 28
GB store). Calling the operation `gc` would tell a user "this only removes what
nothing points at," which is false and dangerous here. `prune` carries the right
sense — `git remote prune` removes refs that are already dead upstream,
`git
prune` removes what is already unreachable — i.e. **collect what has
already been declared dead**, which is exactly archive-then-prune.

**`prune` refuses anything not archived**, and refuses any leaf a pin or a mark
_currently_ binds. That second rule gives camdl an honest partial version of
git's reachability model: a name is a guard against deletion. It also gives
pinning a second reason to exist beyond naming — pinning protects — which is
worth something against the zero-adoption problem in D-b.

Only a mark's **current** binding protects. Its history does not; otherwise a
mark moved twenty times would immortalise twenty runs.

**Prune is whole-subtree only.** Removing part of a leaf leaves a directory that
passes a path check and fails a content check, which is the store's worst state.
If a subtree cannot be removed entirely, it is not removed.

**Prune takes each leaf's lock.** This is the first destructive verb outside the
store's lock protocol, and an archived leaf can be re-claimed by a resumed fit
at any moment. Prune routes through the same `.lock` / `reclaim_or_refuse`
(`store.rs:1092`) machinery `augment` uses (`store.rs:352-362`), refusing a leaf
whose holder is live and taking over one whose holder is provably dead. A live
holder is reported, not skipped silently.

**Prune also collects `<root>/.quarantine/` and orphaned `<root>/.staging/`**,
which today grow without bound and which nothing else removes. Both are inert
debris by construction (run-spec §2.6, `store.rs:441-460`), and `--dry-run`
lists them separately from archived runs so the two are never confused.

**Prune clears nothing else and reindexes at the end.** A pruned run cannot have
had a ref (refused above), so there is no ref to clear; `index.json` is
refreshed via `cas_index::rebuild` so a stale entry never survives the
operation.

### 2.5 `--dry-run`, and whether a size total changes behaviour

`--dry-run` prints, and does nothing: the subtrees, the leaf count, and the
total size on disk, with archived runs, quarantine and staging as three separate
lines.

**Decision: a size total never changes behaviour.** No threshold, no interactive
confirmation, no `--yes`. Three reasons. A threshold is a magic number that is
wrong for someone. Consent was already given once, per leaf, at archive time —
that is what makes the two-step meaningful. And an interactive prompt breaks
scripted and cluster use, which produces a reflexive `--yes` in every script,
which is a guard that no longer guards. What the size does is inform: prune
prints the totals before acting and after, always, `--dry-run` or not.

`--all-archived` is deliberately a long flag with no short form, so it cannot be
typed by accident.

### 2.6 What is dropped from the killed proposal, and why

`--older-than 30d` (meaning "archived more than 30 days ago") is **dropped**. It
requires storing an archive timestamp, and the only place to put it is a file —
reintroducing the shared mutable state this design eliminated, for a
convenience. `--created-before <date>` replaces it, reads the run's own
`provenance.created_at` from the record that is already being read, needs no new
state, and answers the more meaningful question.

A small audit trail survives: each archive operation writes one file
`<root>/.archive/log/<ISO-8601>-<nonce>.json` listing the paths it moved,
created with `O_EXCL` and never appended to. It is what `archive --undo` reads
to reverse "whatever I did yesterday." **The `.archive` layout is the truth and
the log is an audit trail**; losing every log file costs the convenience of
undoing a whole operation by name and costs no data, since the archived subtrees
are still there and still restorable individually.

## 3. `pack` and `unpack`

`camdl mre fit` already bundles a fit's **input closure** so a recipient can
re-run it (`mre.rs:1-13`); it is a bug-report tool and stays. This is its
complement: the **outputs**, so a recipient can look without re-running.

```
camdl store pack <selector>... -o review.camdl-fits.tar.zst [--with-paths] [--no-data]
camdl store unpack review.camdl-fits.tar.zst [--into DIR]
```

### 3.1 Yes — `pack` refuses a mark

**Decision: `pack` accepts a pin or a hash. It refuses a mark, and it refuses a
run that has no pin only by pointing out that a hash works.**

The argument is the one §15.2 makes: a bundle is the "release candidate … that
corresponds to milestone reports and publications," read after it is sent and
cited after it is read. A moving name in a manifest is a citation that silently
re-points, and "silently means something else later" is the failure this store's
entire design is organised against. This is precisely the situation git's
immutability convention exists for, and here the convention can be _enforced_
rather than merely recommended, because the two kinds of name are
distinguishable by type.

The refusal teaches the concept rather than just blocking:

```
error: '@front-runner' is a mark, and a mark moves.
  A bundle is read after it is sent, so what it names must not move.
  It currently points at fit 8f67d9fb.
  Pin it:   camdl store pin national-v1 8f67d9fb
  Or pack the hash:   camdl store pack 8f67d9fb
```

This is the one place the pin/mark distinction is enforced rather than merely
displayed, and it is deliberately the only one. Reading through a mark
(`fit predict @front-runner`) stays allowed, because a read is not durable.

**No second sigil.** A syntactically distinct token (`~front-runner`) would
carry the distinction into prose for free, and it was considered. It is rejected
because `~name` is home-directory expansion in every shell camdl's users run,
every other candidate character is either taken (`:`) or unmemorable, and the
prose hazard is closed more directly: **every artifact camdl writes prints the
hash beside the name** — `national-v1 (8f67d9fb)` for a pin,
`front-runner → 8f67d9fb (moves)` for a mark. The text a user copies into a note
therefore already carries the distinction, without a shell-quoting hazard.

### 3.2 What travels

**Default is view-complete**: posterior draws, per-chain traces, quantities,
predictive output, observed data, the model source and IR, and the metadata
sidecars — about 22 MB raw for the measured ebola fit, roughly 7 MB compressed.

`--with-paths` adds `trajectories.tsv` (15 MB). Without it the recipient can
view everything but cannot fork counterfactuals (`contrasts {}` needs saved
paths) or forecast from the fitted state (`simulate --init-state fit` needs
terminal states). The manifest states the limitation in those words, so the
recipient learns it from the artifact rather than from an error.

`resume_state.bin` never travels: per-machine continuation state, not a result.

Pins travel and land namespaced under the bundle stem — `national-v1` arrives as
`review-2026-08/national-v1` — so nothing can collide with a name the recipient
already holds, two colleagues' bundles cannot collide with each other, and
`camdl store refs 'review-2026-08/*'` shows exactly what arrived. `/` is already
the namespace separator (§1.5), so this needs no new mechanism. `--flat` opts
out at top level and refuses-and-reports on a real collision. Marks never
travel; `pack` refused them.

### 3.3 Where it lands

`--into` defaults to `./camdl-inbox/<bundle-stem>/`, created if absent. A
temporary directory is rejected: an artifact under `/var/folders/…` is one an
epidemiologist will not find again and will not think to move before it is
cleaned up.

**Folding a foreign bundle into the local `results/` is opt-in, never the
default.** A `run_id` asserts "these bytes are a function of these inputs"; a
foreign leaf asserts an identity the recipient's inputs did not produce and
cannot verify, because they do not have the sender's data to recompute against.
Two stores with the same layout are not one store.

Consequences, decided:

- `unpack` recomputes every content digest the bundle's own `run.json` records
  and refuses on mismatch. That verifies the bundle **arrived intact**. It does
  not claim to verify the run was correctly produced, and the manifest says so
  in those words.
- Clash on `run_id` in the target: refuse and report. `--force` overwrites.
- Clash on directory name with a **different** id: suffix the directory,
  mirroring the store's own `~{disambiguator}` escalation (`layout.rs:114-121`).
  Never merge two leaves.
- The manifest records sender, camdl version, `ir/VERSION`, and the selector
  used. A bundle from a different `ir/VERSION` unpacks with a loud note:
  readable, but re-running against it may not reproduce.

### 3.4 The prerequisite that must land first

**A fit segment is not relocatable across machines today, and `pack` is blocked
on that.** `load_config_for_segment` (`handle.rs:174-199`) resolves the archived
`fit.toml.original`'s relative data and model paths against
`FitSidecar.fit_toml_path` — the producing config's directory, recorded verbatim
as typed at `fit/mod.rs:2260`. On the recipient's machine that directory does
not exist, so `fit summary`, `fit predict` and `compare` all fail on an unpacked
fit. This is gh#652's fix working exactly as designed and is not a bug; it
simply means `pack` must re-anchor that path to the unpacked location and record
the original in the manifest. That is Phase 3's first task, and Phase 3 does not
start until it is done.

## 4. The `camdl store` namespace, narrowed

**Decision: `camdl store` holds only the new verbs. `list`, `show` and `cat`
stay at top level. `camdl label` is deleted (§1.7) and `camdl dev reindex` stays
in `dev`.**

The killed proposal moved the whole reading surface into `camdl store`. Review
measured what that costs: 191 references across 34 `docs/**.md` files (51 in
`docs/camdl-run-spec.md`, whose §4.5 is a **named normative section**), 24
integration-test shell-outs across 21 files, four documents baked into the
binary via `include_str!` — so `camdl docs agents` ships stale text until
rebuilt — 35 references in `../camdl-book`, and one cross-repo consumer
(`../camdl-watcher/camdl_watch/ingest.py:591`) that cannot be fixed atomically
and that **degrades silently** on a non-zero exit.

That is the largest single body of work in the proposal and it delivers **zero
lifecycle capability**. The new verbs have no existing spelling, so grouping
_them_ costs nothing and gains the coherence immediately. If `camdl store list`
later proves wanted, it can land as its own change, where the doc sweep and the
cross-repo coordination are the commit's subject rather than a footnote. Alpha
posture permits the rename; it does not require paying for it now, and
`.claude/rules/proposals.md` asks for the call to be made rather than deferred —
this is the call.

`dev reindex` stays in `dev` because `prune` reindexes itself, so an ordinary
user never types it.

## CLI surface

| command                                     | effect                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| `camdl store pin <name> <selector> [-m …]`  | Bind `name` to one run, once. `O_EXCL`; fails if the name exists.        |
| `camdl store unpin <name>`                  | Remove the pin. The run is untouched.                                    |
| `camdl store mark <name> <selector> [-m …]` | Point `name` at a run, creating or moving it; prints both bindings.      |
| `camdl store unmark <name>`                 | Remove the mark and its history.                                         |
| `camdl store refs [GLOB] [--history NAME]`  | List pins and marks with their runs; `--history` prints a mark's file.   |
| `camdl store archive <selector>...`         | Move subtrees into `.archive/`; hide from every walk; keep on disk.      |
| `camdl store restore <selector>...`         | Move them back. `--undo <op>` reverses one logged archive operation.     |
| `camdl store prune <selector>...`           | Remove archived subtrees. Refuses unarchived, pinned, marked, or locked. |
| `camdl store prune --all-archived`          | The same, over everything archived, plus quarantine and staging debris.  |
| `camdl store pack <selector>... -o FILE`    | Bundle outputs. Accepts a pin or a hash; refuses a mark.                 |
| `camdl store unpack FILE [--into DIR]`      | Unpack to `./camdl-inbox/<stem>/`, verifying digests.                    |
| `camdl list --archived`                     | The same table, over `.archive/`.                                        |

`list` gains a `REFS` column, populated by one `read_dir` over `refs/pin` and
`refs/mark` (dozens of files, not thousands) inverted into `run_id → names`, and
a `pins` field in `--format json`. The gh#701 hint is rewritten to name
`camdl store pin`, which exists.

## Decisions

1. **Three concepts, not one word.** `label` = run-time display prose, in the
   record, not addressable. `pin` = immutable name, one run, citable. `mark` =
   explicitly moving pointer. The conflation of the first two is the verified
   cause of zero adoption (D-b), and no ergonomic fix to `camdl label` addresses
   it.
2. **`pin` / `mark`, choosing local meaning over git's spelling.** Both git
   words are already taken inside this project (`Tags:` in lab notes; `# Base:`
   model branching), and `branch` would additionally promise automatic
   advancement that camdl cannot provide. `pin` inherits git's _meaning_ without
   its word.
3. **One file per name; pins `O_EXCL`, marks `O_APPEND`.** No operation can lose
   a name. Nothing is shared, so no rebuild source is needed.
4. **A mark is for a judgment, never for a derived ranking.** `front-runner` is
   a legitimate mark. **`best-elpd` is not, and will not be supported as one** —
   it is the argmax of a computation `compare` already performs, and a
   hand-moved pointer at it is a manually-maintained cache of a derived fact
   that goes stale silently the first time a better fit lands and nobody
   re-points it. The supported spelling is a fresh query over a namespace:
   `camdl compare '@national/*'`. This is a deliberate refusal of one of the two
   motivating cases the killed proposal cited, and it is the right refusal.
5. **Archive relocates into `<root>/.archive/`.** A manifest keyed on `run_id`
   cannot make `list` faster, because the walk reads every `run.json`'s bytes to
   decide. Relocation reuses the dot-directory skip that already exists at
   `cas_read.rs:244`. A move is not a rewrite, so the committed-leaf constraint
   holds.
6. **`lookup` gains `Lookup::Archived`; the restore happens on the write path.**
   One extra `stat` on a `Miss`. `lookup` stays a pure read; without this,
   re-running an archived run silently produces a second copy.
7. **No `rm`.** Two paths to one destruction drift, and this codebase has the
   `--force` case study. `archive` is cheap and reversible; the two-step costs
   nothing.
8. **`prune`, not `gc`.** `gc` promises collection-by-unreachability, which
   camdl cannot offer as its criterion because an unreferenced leaf is the
   normal state. `prune` carries "collect what is already declared dead," which
   is what this is.
9. **A pin or a current mark refuses `prune`.** Naming protects, which restores
   an honest partial reachability model and gives pinning a second reason to
   exist. A mark's history does not protect.
10. **Prune takes each leaf's lock**, via the existing `reclaim_or_refuse`
    machinery, and collects `.quarantine/` and orphaned `.staging/` as well.
11. **A size total never changes behaviour.** No threshold, no prompt, no
    `--yes`. Consent was given per leaf at archive time; a reflexive `--yes` is
    a guard that does not guard.
12. **`pack` accepts a pin or a hash and refuses a mark**, with a message that
    names the pin command. It is the only place the distinction is enforced
    rather than displayed; reads through a mark stay allowed.
13. **One sigil.** `@` for both kinds. A second sigil buys prose legibility at
    the cost of a shell-expansion hazard; printing `name (hash)` everywhere buys
    the same legibility with none.
14. **`camdl store` holds only new verbs.** `list`/`show`/`cat` stay top level;
    the 191-reference doc sweep and the silently-degrading cross-repo consumer
    are not paid for zero capability.
15. **`--older-than` is dropped; `--created-before` replaces it.** The former
    needs a stored archive timestamp, i.e. the shared state this design removed.
16. **`cmd_label` and `ensure_provenance_label` are deleted**, per the alpha
    posture: no alias, no shim. On a cache hit a cell's label names its
    producing invocation, which is correct.

## Sequencing

| phase | contents                                                                                                                                                                                         | risk                                                                                                                |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| 1     | `refs/` storage; `pin`/`unpin`/`mark`/`unmark`/`refs`; re-point `handle.rs`'s `@` branch; delete `cmd_label` + `ensure_provenance_label`; fix gh#701's hint; `REFS` column and `pins` JSON field | No identity change. `run_id` byte-stability test is the check.                                                      |
| 2     | `.archive/` relocation; `archive`/`restore`; `Lookup::Archived` + write-path restore; ancestor cleanup; `list --archived`; `prune` with locks, ref protection, and quarantine/staging collection | Touches `store.rs`. The `Lookup` enum gains a variant, so every match site is a compile error — which is the point. |
| 3     | Re-anchor `fit_toml_path` on unpack (**prerequisite**); `pack`/`unpack`; namespaced incoming pins                                                                                                | Largest surface; blocked on the prerequisite.                                                                       |

Nothing here re-keys anything. `Provenance` is recorded-not-hashed
(`inputs.rs:352`), refs live outside every leaf, and archiving moves bytes
without altering them. `ir/VERSION` is untouched and no golden moves.

## Verification

Each item names the oracle, not just the behaviour.

- **Naming leaves the leaf untouched.** Pin a run; assert every byte of its
  leaf, its `run.json` mtime, and its `run_id` are unchanged. This is the
  sharpest available oracle for the whole ref design and it is cheap.
- **A lost update is unrepresentable.** Two processes `pin` the same name
  concurrently; exactly one succeeds, the other reports the incumbent's hash,
  and the file holds one line. Two processes `mark` the same name concurrently;
  the file holds **two** well-formed lines and one of the two bindings resolves.
- **A malformed last line in a mark errors** naming the file and line, and does
  **not** resolve to the previous line. Negative control: the previous line is a
  valid, different run, so a silent fall-back would pass a naive test.
- **`@name` is O(1).** On a store with 550k leaves, `@name` resolution performs
  a bounded number of file reads — assert on the read count, not on wall time.
- **Archiving makes the walk shorter, not just the table.** Archive a fan-out
  and assert the number of `run.json` reads performed by `list` drops by the
  archived count. Counting parses is already instrumented (`FULL_PARSES`,
  `cas_read.rs:207`); this needs the same for reads. A test that only asserts
  the rows disappeared would pass against the design this replaces.
- **An archived run that is re-run is restored, not recomputed.** Archive a
  leaf, re-run the identical command, assert one leaf exists (not two) and that
  no recompute occurred.
- **Archiving is byte-neutral.** Archive then restore; assert the leaf's
  manifest digests all still verify and `lookup` returns `Hit`.
- **`prune` refuses**, separately and with distinct messages: an unarchived
  subtree; an archived subtree a pin binds; an archived subtree a **current**
  mark binds; and an archived leaf held by a live `.lock`. Negative control: an
  archived subtree bound only by a mark's **historical** line prunes
  successfully.
- **`prune --dry-run` removes nothing** — assert the byte count on disk is
  unchanged — and its reported total equals what a real prune then removes.
- **`pack @mark` exits non-zero** naming the pin remedy; `pack @pin` and
  `pack <hash>` succeed. Round-trip: unpack reproduces every packed file
  byte-identically; a bundle with one flipped byte fails.
- **An unpacked fit answers `fit summary`** on a machine where the original
  `fit_toml_path` directory does not exist — the prerequisite in §3.4, tested by
  deleting that directory before the call.
- **`run_id` byte-stability** across the whole change: a pinned, archived,
  restored run hashes identically to the same run before any of it.

## Named follow-ups

- **F1 — `write_fit_sidecar` ends in a bare `std::fs::write`**
  (`run_meta.rs:645`), so any writer of `fit.meta.json` can leave a torn file.
  It should use the store's own tmp + fsync + rename ordering. Deleting
  `cmd_label` removes the _post-hoc_ writer but not the fit-time one. Small,
  independent, and worth filing now.
- **F2 — gh#704: a posterior ensemble is stored twice.** 84% of sim leaves are a
  second copy of data already held compactly in `ensemble.tsv`, ~17 GB and 461k
  leaves. Archiving makes that fan-out survivable; it does not make it right,
  and it is a larger reclamation than anything here.
- **F3 — `batch run` writes no `SimEnsemble` leaf**, so a batch fan-out has no
  single object to pin. Verified by the absence of `ArtifactKind::SimEnsemble`
  in `batch.rs`. Until it lands, `archive` and `pin` address a batch fan-out by
  subtree.
- **F4 — gh#594: the store-root model archive is last-writer-wins across
  models.** `batch.rs:614` writes `model.ir.json` at the root of a store that
  holds many models. Independent of this proposal but in the same directory.

## Not this proposal

Compressing artifacts on disk (gh#698 — needs the benchmark first). Whether a
posterior ensemble should be 6,000 leaves at all (gh#704 / F2). Bounding
`list`'s discovery further (gh#699's first half already landed: `--since` bounds
discovery and ensemble members are skipped without a full parse — commits
`4a0b5cd0`, `db0c9008`). And any change to what is hashed: nothing here touches
identity.
