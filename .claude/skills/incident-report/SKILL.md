---
name: incident-report
description: Write a docs/dev/incidents/ entry, or decide whether something qualifies as one at all. Enforces the reproduction bar and the doc-vs-doc / doc-vs-code / code-vs-code classification. Use when a serious bug or misbehavior warrants an engineering response beyond a one-line fix.
---

# Incident reports

An incident report records a serious bug, outage, or misbehavior that warrants
an engineering response beyond a one-line fix: what happened, how it was
detected, root cause, remediation, and what architectural or process change (if
any) it suggests. These are the artifacts that justify later refactors.

Path: `docs/dev/incidents/<YYYY-MM-DD>-<slug>.md`.

## The reproduction bar

**An incident report requires a reproduction.** A concrete input → wrong output,
with the command that produced it.

"Would be off by ~0.4 days" is a **hypothesis**, not an incident. If you can't
produce a reproduction, the artifact is a _question_ filed under
`docs/dev/notes/`, not a `docs/dev/incidents/` entry.

The reproduction bar is what keeps phantoms out of the incident archive.

## Classify the discrepancy before proposing a fix

Three classes, three different fixes. **State the class explicitly at the top**
of any incident or proposal that depends on the answer.

- **doc-vs-doc** — edit a doc.
- **doc-vs-code** — verify which side is right, then sync the loser.
- **code-vs-code** — fix the code and add a test pinning the agreement.

Misclassifying inflates a typo into an engineering project (or, the other
direction, hides a real bug behind a doc edit).

## Don't document the broken interim

When a bug fix is straightforward and the fixed state is the right state, apply
the fix and update the user-facing doc to describe the _fixed_ reality. Long
descriptions of the broken interim state belong here, in the incident report —
not in the spec, cheatsheet, or user-features.

## Related

- Verification discipline — paste the command and its output alongside every
  normative claim about current behaviour:
  [`docs/dev/agent-verification-conventions.md`](../../../docs/dev/agent-verification-conventions.md)
- Reporting findings as a numbered list: the `review-report` skill.
- Red → green as the proof a fix landed where intended: same conventions doc.
