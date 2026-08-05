---
paths:
  - "rust/**/*.rs"
description: Rust code conventions — dead code, existing seams, wiring primitives, named tolerances, parse-at-the-boundary
---

# Rust conventions

## Delete dead code on sight

Unused functions, unused modules, "v1" paths kept around after a "v2" rewrite,
prototype code kept around "in case we need it" — all delete-on-sight. There is
no consumer to placate, no migration to stage, no contract to honour. Code that
comes back can come back from `git log -S '<symbol>'`.

- **`#[allow(dead_code)]` is a smell, not a fix.** At a definition site it tells
  a future reader "I know this is dead but didn't delete it." At a module level
  (`#![allow(dead_code)]` or `#[allow(dead_code)] mod foo;`) it hides _which
  specific items_ are dead, blocking the compiler from reporting individual rot.
  Either prove the item is reachable from a live entry point, or delete it.
- **"v1" alongside "v2" is dead code.** When a rewrite lands, the old path is
  deleted in the same commit. Carrying both is the number-one source of context
  tax.
- **Comments saying "kept in case X" are dead code with extra steps.** If X
  happens, `git log` recovers the file in seconds.
- **Ruthlessness is collegial.** Smaller surface = humans review faster, agents
  edit faster and read less context.

When you encounter dead code while doing other work, delete it in a separate
commit before the substantive change — review is easier when each commit is one
thing.

## Reach for the existing seam before adding a parallel one

Before adding a primitive, helper, method, or constant, search for one that
already answers the question and extend it. A second function that answers the
same question a hair differently is not a convenience — it is a fork the two
sides drift across.

The boundary loop is the cautionary tale: "where does the integrator stop next"
is now answered **four** incompatible ways — `Schedule::substep`,
`Schedule::clip`, `Schedule::next_boundary`, and the unused
`Schedule::next_stop` — because successive changes each reached for a fresh
accessor instead of the one-carrying-the-reasons that already existed (gh#233).

Before you add `foo_v2` / `next_thing` / a sibling accessor: `rg` the type for
its existing methods, read them, and either call one or extend one. If you
genuinely need a new one, the commit must say _why the existing seam could not
serve_ — that one sentence is the review bar, and its absence is the smell.

## A shared primitive ships wired into a consumer, or as a named step of a committed arc

Do not _create_ dead code by landing a primitive that nothing calls **and that
nothing is committed to call**. The failure mode is the _speculative_ primitive
— landed "in case we need it," with no consumer and no plan, often advertised as
if the consolidation it promises were already done. `Schedule::next_stop`
shipped exactly this way — advertised as the "single boundary authority,"
unit-tested, never called, with no tracked plan to finish the centralization —
so the consolidation stayed half-done, which is the soil the gh#70 / gh#208
silent-wrong bugs grew from (gh#233).

A unit-tested function with zero callers is unexercised on every path that
matters, and its tests are thin evidence (written to the same mental model the
code was — they confirm the author's intent, not the system's behaviour).

Two honest ways to land a primitive:

1. **Wired now** — the change routes at least one real consumer through it, in
   the same PR. The default.
2. **A named step of a committed arc** — a foundation landed _ahead_ of its
   consumer is sound engineering when it is an explicit prerequisite of a
   feature we are committed to shipping: a tracked proposal/issue names the
   consumer that will wire it, the commit says
   `foundation for <arc>; wired by <next step>`, and the primitive is exercised
   by its own tests meanwhile.

What stays prohibited is the orphan: a primitive with no committed consumer, or
one dressed up as if the work it enables were already complete. If you cannot
name the consumer or the arc, you do not have one yet — do not land it.

## Name tolerances and magic numbers once; never inline a bare epsilon

A bare numeric literal in control flow (`if dt <= 1e-15`,
`(iv - t).abs() < 1e-10`) is unreadable and un-greppable: the next reader cannot
tell a step floor from a due-tolerance from a rate floor, and the same concept
silently drifts in value across call sites.

Define each threshold **once**, as a named `const` at the module that owns the
concept (time tolerances belong in `schedule.rs`), with a doc comment saying
what the check _means_, and reference it everywhere. Distinct concepts that
share a value keep **distinct names** — a time `MIN_STEP_EPS` and a
`RATE_EPSILON` are not the same thing even at `1e-15`.

The cost of inlining is concrete: the "effectively-zero step" threshold was
spelled `1e-15` at four sites while PGAS's equivalent floor `GRID_STEP_EPS`
silently used `1e-12` — a three-orders-of-magnitude disagreement that surfaced
only when someone tried to give it a name (gh#233).

## Parse at the boundary; don't pass raw and validate

We want **illegal states unrepresentable** — ideally a wrong wiring won't
compile, rather than being caught by a comment, a `debug_assert!`, or a test.
Hold this in a **careful, pragmatic balance**: the aim is to delete a class of
silent-wrong bug, not to turn the code into a type exercise.

The high-leverage move: at a trust boundary — where raw/loosely-typed data
enters the typed core (`Vec<f64>`, `String`, `&CompiledModel`, CLI args, JSON) —
_parse_ it once into a type whose constructor is the only way to make it and
whose existence proves the invariant ("parse, don't validate", Alexis King
2019). Downstream receives the parsed type and never re-checks. Prefer a
fallible smart constructor that folds _produce + validate + role-tag_ into one
seam: `OutputTimes::from_model(model)?` is the producer, the sort/finite check,
and the "this is the output axis, not the effect axis" tag, all in one place.

Tells you're validating instead of parsing — each is a cue to promote to a
parsed type:

- a `debug_assert!` of an invariant on a _public_ constructor (checked only in
  debug; the type still permits the illegal value — e.g. `Schedule::new`'s
  `debug_assert!(sorted)`);
- a comment carrying an invariant ("must be sorted", "caller guarantees finite")
  instead of a type;
- the same check repeated at several call sites;
- a primitive-heavy signature where the primitives have distinct semantics and
  the same underlying type (`fn(…, Vec<f64>, Vec<f64>)` — adjacent, swappable, a
  swap compiles → silent-wrong).

**The pragmatic line.** Wrap a value when its instances are genuinely different
_and_ swappable into the same slot — different semantics, same underlying type,
so a swap type-checks and silently corrupts. Do **not** wrap values that are
usually the same number or already validated elsewhere — that is
over-engineering, and it is a real cost (noisier signatures and tests, tiny
types the maintainer must mentally unwrap).

gh#233 shows both sides: `OutputTimes` / `EffectTimes` / `ObsTimes` over a
checked `SortedFiniteTimes` earn their keep (three `Vec<f64>` axes with distinct
meaning — record / fire / score+reset — so a swap is silent-wrong), while
`NominalStep` / `SnapGrid` scalar newtypes were dropped (`dt == grid` at six of
seven sites — ceremony). Keep wrappers at the construction boundary and unwrap
to the primitive for the hot path so nothing threads through the inner loop.
