# Aggregation-semantics arc — working tracker

Scratch coordination file for the remaining increments of
`docs/dev/proposals/2026-07-31-aggregation-semantics.md`. **Delete when the arc
lands.** Not a design document — the proposal is. This only records where we
are, so a fresh session can resume without re-deriving the plan.

Branch: `feat/aggregation-remaining` (off `main` @ `9643c246`).

## Decisions taken (2026-08-11)

- **Typed `dim_name`** — `User of string | Generated of { kind; source }`,
  landing inside B's `ir/VERSION` bump. gh#568. Supersedes C5 (lexer reservation
  is unnecessary once the discriminant is a type) and subsumes C6
  (`generated : bool` becomes redundant).
- **Re-keying is not a design constraint** (`4833858f`). Land pending re-keying
  changes in one bump rather than dropping any.
- **`ir/VERSION` 0.30 → 0.31** approved for B.
- **Order** below approved.

## Steps

- [x] **1. Proposal edits** — `edb67728`. All items below done, plus two
      decisions the sweep forced: the interim pooling warning folds into E (it
      was specified to ship with A, A shipped without it, and its predicate
      requires an indexed stream while both known E hits are un-indexed), and
      §8.1 pins the wire format (`name` + absent-means-`User` discriminant)
      because `ir/golden/` is frozen and loaded directly by Rust tests.
- [x] **2. gh#566 audit** — matrix posted to gh#566. Findings that change the
      plan:
  - **Four mechanisms, not three.** The reactive whitelist (`E279`/`E273`) is
    its own, and its message invents a stream named `'?'`.
  - **The table is small** — only three constructs are genuinely scoped, so
    building the scope table is now the _cheaper_ path for B9, not the more
    expensive one. Build it rather than adding a fourth ad-hoc arm.
  - **The `E100` hint is actively harmful**: `incidence` in a rate says "add a
    declaration in forcing { }", which an agent will try.
- [ ] **3. C4 + C4a** — `lowering` metadata replaces the `__` sniff; apply the
      `via` rewrite to `quantities` / `interventions` / `events` / `reactive`.
      No bump.
- [ ] **4. Increment B** — `ir/VERSION` 0.30 → 0.31. Land as a stacked sequence
      behind one bump, each piece green:
  - [ ] 4a. typed `dim_name` + goldens regenerated
  - [ ] 4b. `WeightedFlowSum` lowering (B1), per-reference accumulator (B2)
  - [ ] 4c. weight restrictions (B3) + deferral diagnostics (B4) + extend the
        `gradient_capability.rs:442` gate to the new variant
  - [ ] 4d. delete `explicit_incidence_sum` (B5); scope check from step 2
  - [ ] 4e. `inc_<stream>` doc correction (B8); release-notes line (B7)
- [ ] **5. Increment D** — `prevalence(of =, among =)`, checks D2–D5, removals
      D6, dimension-change migration D7, language-changes entry.
- [ ] **6. Increment E** — observation-boundary rule. Closes gh#478.

## Out of scope, tracked elsewhere

- gh#565 — `quantities {}` cannot read a flow (cumulative incidence). `blocker`,
  separate design. B does **not** close it.
- gh#567 — `observed(s, window =, reduce =)` naming. Cosmetic.
- gh#111 — the dimension-aware resolver. gh#568 is one slice; if the typed
  `dim_name` design starts demanding the full `resolve_indexed_ref`, stop and
  scope gh#111 rather than growing gh#568.
- gh#559 / gh#560 / gh#504 — Increment A follow-ups (E334/E283 skip
  `quantities{}`; A6 leaves indexed parameters on the mangled-name E100; the
  spec's E263 positional-index claim; duplicate E263). Small; fold in wherever
  they touch the same code.

## Gate

`make test` before anything lands. `make test-ocaml` in the inner loop for
compiler-only steps. Goldens are staged explicitly, never with `git add -A`.
