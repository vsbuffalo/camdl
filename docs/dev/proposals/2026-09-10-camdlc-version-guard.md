# Key the camdlc↔camdl guard on IR schema version, not git hash

Status: Accepted — the two-signal design (hard schema gate, soft hash warning)
is settled; what remains is implementation. A workaround landed
(`build(make): resolve fresh camdlc
via PATH-prepend in test-rust`, 458c8cb);
this is the root-cause fix that lets the dev workaround be removed.

## Problem

`camdl` (runtime) refuses to run a `camdlc` (compiler) whose **git hash**
differs from its own (`rust/crates/cli/src/util.rs`: `find_camdlc` →
`check_camdlc_version_once` → `eval_version_output`, comparing
`camdlc --camdl-version` against `crate::version::GIT_HASH`). The guard exists
to stop a runtime reading IR a mismatched compiler emitted.

Keying on exact git hash is **stricter than the actual compatibility contract**,
which is the IR schema (`ir/schema.json` / `ir/VERSION`). Two binaries built at
different commits with an _identical_ schema are fully compatible, but the guard
rejects them on the commit label. So it false-reds on essentially every dev
iteration where the two sides were built at different commits — which is the
common case, because:

- a parallel checkout running `make install` clobbers the shared
  `~/.local/bin/camdlc` to a different commit; and
- cargo does not rebuild the `camdl` binary when only OCaml/docs change, so its
  embedded `GIT_HASH` lags HEAD while a freshly-built `camdlc` does not.

The reproduction that motivated this: a session of OCaml-only changes (no schema
change) produced `error: camdlc version mismatch` in the cargo acceptance tests,
against a `~/.local/bin/camdlc` left at a prior commit by other work — a pure
false-red.

## Current workaround (shipped)

`make test`'s `test-rust` prepends the freshly-built `camdlc` to PATH and sets
`CAMDL_SKIP_VERSION_CHECK=1`. The PATH-prepend pins the compiler under test (no
divergence — `camdl` never falls back to a stale PATH camdlc); the skip mutes
the commit-label mismatch against a cargo-cached stale `camdl` binary, which is
safe because **`camdl` only goes stale when no Rust changed → the IR schema is
unchanged → a stale `camdl` is schema-compatible with a fresh `camdlc` by
construction**. It works, but it is a workaround: the guard is muted in the
harness rather than passing on its merits.

### The skip has a downstream cost (gh#888)

Because the skip is universal — every test harness and every ad-hoc worktree run
sets it — a second defect becomes reachable through it. The compiled-IR cache
(`~/.cache/camdl/ir`) keys an entry on (model, compiler, the schema the runtime
expects) and does **not** check the `ir_version` the compiler actually emitted
before publishing. With the handshake skipped and a stale `camdlc` on PATH, a
0.39 document is written under a 0.40 key; every later read of that model then
hard-errors on the version mismatch rather than missing the cache and
recompiling. Three such poisoned entries were observed on 2026-09-09, cleared
only by hand.

gh#888 fixes the cache's own missing check and does not depend on this proposal.
But the reason the skip is on everywhere is this guard, so landing the
two-signal gate removes the precondition rather than only the symptom.

## Proposed fix

Gate on **IR schema compatibility**, not git hash. The load-bearing design
choice is _what_ represents the schema:

- **Not the hand-maintained `ir/VERSION` integer alone.** `ir/VERSION` is bumped
  by hand (CLAUDE.md "Changing the IR schema"). An OCaml-only change to the
  emitted IR _shape_ in `ocaml/lib/ir/serde.ml` (a new field, a renamed tag, a
  reordered variant) that forgets the bump would leave `ir/VERSION` matching
  while the emitted shape diverges from what `camdl`'s deserializer expects — a
  real incompatibility a VERSION-keyed gate would wave through, defeating the
  guard's purpose. (This is also why the "staleness ⟹ schema-compatible"
  argument below is _not_ airtight for the permanent gate — only for the
  same-tree harness workaround, where both sides share one `serde.ml` regardless
  of `ir/VERSION`.)
- **A content hash of the schema surface.** Derive the gate value from a hash of
  `ir/schema.json` (or the serde surface itself), embedded in both binaries;
  `camdlc --camdl-version` reports it and `camdl` compares against its own. An
  emit-shape change then moves the hash even if `ir/VERSION` is forgotten, so
  the gate trips. (Alternatively/additionally, a CI check that fails any commit
  touching `serde.ml`/`schema.json` without an `ir/VERSION` bump — belt and
  braces.)

With a schema-content gate, a matched-schema pair passes _on its own merits_ —
no skip in dev or prod, no PATH-prepend needed — and a genuine schema change
still trips it.

## The gate is coarser than a git hash, and that is why there are two of them

Schema version is **coarser** than git hash: it will not catch an
expander/codegen behavior change that alters emitted IR _without_ bumping the
schema — e.g. an autodiff/`rate_grad` fix, or a dimensional-rescale correction.
Those change the _values_ in the IR, not its shape, so a schema-version guard
alone would treat a fixed and an unfixed `camdlc` as interchangeable.

That is not a reason to keep the hash as the gate; it is a reason to carry both
signals, because they answer different questions. For the _deserialization_ risk
the guard nominally protects against (runtime cannot read the compiler's IR),
schema version is exactly right. For "am I running the camdlc I think I am," it
is not. So:

1. **Schema version — hard gate** (refuse on mismatch). This is the real
   incompatibility, and the only one worth refusing over.
2. **Git hash — soft warning** (print, do not exit) when it drifts, so an
   operator running a knowingly-mismatched pair is told, without blocking dev
   iteration.

This also lets the harness drop both the PATH-prepend and the skip: a
freshly-built pair shares a schema version and passes the hard gate, and the
soft hash warning is harmless noise (or suppressed under
`CAMDL_SKIP_VERSION_CHECK`).

## Recommendation

Implement the two-signal guard (hard schema gate + soft hash warning) in
`util.rs::eval_version_output`, embed a **content hash of the schema surface**
(not the `ir/VERSION` integer) in both binaries, and once it lands, simplify
`test-rust` back to a bare `cargo test --workspace` (removing the PATH-prepend +
skip workaround). Add a unit test for `eval_version_output` covering: schema
match + hash match (pass, no warn), schema match + hash drift (pass + warn),
schema mismatch (hard fail).
