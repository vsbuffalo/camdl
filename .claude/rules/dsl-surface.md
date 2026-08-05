---
paths:
  - "ocaml/lib/compiler/**"
  - "docs/camdl-language-spec.md"
  - "docs/dsl-cheatsheet.md"
  - "docs/user-features.md"
  - "docs/language-changes.md"
description: DSL surface rules — required reading, no-loose-semantics, error quality, breaking-change signposting
---

# DSL surface

## Required reading before changing the DSL

For lexer, parser, expander, dimcheck, new unit literals, or new functions in
DSL constant positions:

- [`docs/camdl-language-spec.md`](../../docs/camdl-language-spec.md) end-to-end,
  especially §2 (units and dimensions), §4 (parameter kinds), §6 (tables), §7
  (forcings).
- [`docs/user-features.md`](../../docs/user-features.md) for example patterns.
- [`docs/dsl-cheatsheet.md`](../../docs/dsl-cheatsheet.md) for a fast
  orientation.
- The grammar itself: `ocaml/lib/compiler/lexer.mll` (unit literals + tokens),
  `ocaml/lib/compiler/parser.mly` (the rule for whatever you're changing),
  `ocaml/lib/compiler/dimcheck.ml` (dimensional behaviour).

## No loose semantics

Never silently accept invalid input. If a construct looks like it means
something, it must either mean exactly that or produce a clear error. Examples:
`_args` patterns that discard function arguments, optional fields that default
to "works but wrong." If the compiler accepts it, the behavior must be fully
specified and intentional.

## Error messages are a feature, not polish

Error quality is a first-class design goal. A bad error message is a bug — it
means the compiler detected a problem but failed to help the user fix it.

Every diagnostic should:

- Show what went wrong (the mismatch, the constraint violation)
- Show where (source location, transition name, parameter name)
- Show why (the expected vs actual value, with domain-specific names)
- Suggest a fix when possible (hint text, corrected code)

When two possible error codes could fire for the same root cause, prefer the one
that points closest to the actual mistake. E.g., a parameter used inconsistently
across transitions should produce E303 ("conflicting dimensions in transition A
vs B") not E302 ("dimension mismatch in addition") — even though E302 is
technically correct, E303 gives the user the cross-transition context they need.

**Never use `failwith` or `assert false` for user-facing errors.** These produce
stack traces instead of diagnostics. Use the Diagnostics module with error
codes, source locations, and hint text.

Every emit site gets a one-line entry in
[`docs/dev/warning-catalog.md`](../../docs/dev/warning-catalog.md); reviewers
should reject any diagnostic emit-site that isn't in the catalog.

## Design the DSL for humans first; agents follow

A meaningful fraction of `.camdl` files now come from coding agents, and that
share will grow. The temptation is to optimize the surface for agents directly —
explicit verbosity, machine-friendly tags, lots of "obvious" guardrails. Resist
it.

The DSL's value to agents comes from the _same_ property that makes it value to
humans: that a sharp non-software-engineer epidemiologist (a health-ministry
modeler in an under-resourced setting, the recurring target user) can read a
model and have a chance of being right about what it does. Agents do well on
this DSL because it is human-readable, not in spite of it.

When a syntax choice is in tension between "what an agent would tolerate" and
"what a model author would understand at a glance," the model author's gut is
the tiebreaker — that is the choice that serves both audiences, because it is
the one that doesn't ask either of them to carry hidden calendar arithmetic,
ambiguous units, or implicit conventions in their head.

Concretely: prefer explicitly named functions over polymorphic operators where
the semantics differ (`add_calendar_months(d, 1)` beats `d + 1.month` when the
operation is non-affine), prefer hard errors with hint text over warnings
(warnings are noise an agent will suppress and a non-specialist will skim), and
keep the surface small enough that the entire grammar fits in a head.

## Breaking language changes must signpost the migration

Backwards compatibility is a non-goal, but a _silent_ break is a bug. When you
change the DSL surface in a breaking way — rename or remove a keyword, require
new syntax, tighten a semantic rule — the compiler must reject the old form with
a diagnostic that **names the replacement (old → new)**, not a bare `E001`
syntax error. A model written against last month's grammar should fail with a
migration, not a mystery. The diagnostic is the migration tool.

And every breaking language change gets an entry — newest first, with the old →
new migration — in [`docs/language-changes.md`](../../docs/language-changes.md),
which is embedded into `camdl docs language-changes` so an agent on any binary
can see what changed. The diagnostic should point there (`… see \`camdl docs
language-changes\``) until the targeted hint exists. Backfilling old changes
into that log is welcome; not adding new ones is a regression.
