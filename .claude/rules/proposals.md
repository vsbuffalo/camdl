---
paths:
  - "docs/dev/proposals/**"
description: Writing a proposal — read the area's normative docs first, be self-contained, and ship with no open questions
---

# Writing a proposal

Before drafting, read the normative docs for the area you're touching. The
relevant path-scoped rule (`dsl-surface`, `ir-schema`, `sim-and-inference`,
`run-identity`, `calendar-time`) lists them and loads when you open a matching
file — open one first if you haven't.

Working from a mental model rather than from the spec has produced proposals
that reinvent existing surface badly. Once is bad luck; twice is a pattern; the
pattern is fixed by reading first, not by trying harder to remember.

## Self-containment

When a proposal is the _first_ thing someone would read about a topic, it must
either be self-contained — citing all the existing surface relevant to its
claims — or state explicitly what background the reader is assumed to bring. The
"read the spec first" rule is for the author, not just the reviewer.

## A shipped proposal has no open questions

A proposal in `docs/dev/proposals/` is the spec an implementer follows. By the
time it ships — committed as the decision record and implemented against — every
design question it raises must be **resolved**: make the call and record it
inline.

An `## Open questions` section with undecided items is the tell that the
proposal is still a draft. It punts the design onto the implementer, who then
either guesses (a silent-wrong risk) or stalls. Decide each question before
shipping, or convert any that genuinely can't be settled into a named follow-up
(`gh#NN` or a separate RFC) with the reason. Never leave a bare list of
undecided questions in a proposal you're treating as done.

Drafting with open questions is fine. _Shipping_ with them is not.

## Implementation

Implementation commits cite the proposal via a `Proposal:` footer and follow it
exactly unless a deviation is documented inline. Don't improvise design
mid-implementation.

## Archived proposals are not pending work

`docs/dev/proposals/archive/` holds decided or superseded proposals. Read them
for rationale; do not cite them as in-flight design. Check the path before
describing a proposal's status — CLAUDE.md carried two proposals as "in-flight"
for months after they were archived.
