---
name: review-report
description: Format the results of any review — a proposal reviewed against the code, a PR review, an audit, a subagent fan-out you are relaying — as a numbered issue list, plus the Design-calls block for anything blocked on the maintainer. Use when reporting review findings, relaying an audit, or ending a turn where work is blocked on a judgment call.
---

# Reporting a review

Any time you report the results of a review, **the last thing on screen must be
a numbered list of concrete issues.** Not a narrative, not a recommendation, not
a summary of what you did. The list is what gets acted on, so it must be the
thing in front of the reader when you stop talking.

Prose framing before the list is fine and often useful. It does not replace the
list.

## Each entry carries, in this order

1. **A one-line claim** — what is wrong, stated as a fact.
2. **Severity** — blocker / high / medium / low. A blocker is something that
   makes the reviewed artifact unimplementable or unsafe as written.
3. **Where** — `file:line` or the doc section.
4. **Evidence** — the command and its output, or the input → wrong output. A
   finding without a reproduction is a _question_, and must be labelled as one.
5. **Independent or entangled** — does fixing this depend on any other item, or
   can it land alone? Say which, explicitly. This is the field that determines
   what can be parallelized or knocked out immediately, and it is the one most
   often omitted.
6. **Disposition** — fixed already (with the commit), needs a decision from the
   maintainer (with the specific question), or filed as `gh#NN`.

## Rules that make the list usable

- **Order by severity, not by discovery order or by file.**
- **Separate what is verified from what is inferred.** A reviewer's claim you
  did not re-check is not a finding — either verify it or mark it unverified.
- **Independent bugs found adjacent to the reviewed work get their own entry and
  their own issue**, never a footnote inside the reviewed artifact's findings. A
  bug in a different code path with a different trigger is a different bug even
  if you found it while looking at this one; say so in as many words, because
  the reader cannot tell from the narrative.
- **State plainly what is blocked on the maintainer and what is not.** If some
  findings are actionable now and others need a design call, split the list so
  the actionable ones can proceed while the decision is pending.

## Design calls: say what you need, and how sure you are

When work is blocked on the maintainer's judgment, end with a **Design calls**
block — separate from the issue list, because these are questions, not defects.
Each entry:

1. **The question**, in one sentence, answerable without re-reading the thread.
2. **The options**, with the concrete consequence of each — not a survey, two or
   three real candidates.
3. **My recommendation** — always give one, even when uncertain. "You decide" is
   not an answer; it pushes the work back without adding anything.
4. **Confidence**, as one of exactly three labels:

- **Solid** — the evidence is decisive and I would act on it if you did not
  reply. Say so: "proceeding with X unless you object." Do not manufacture a
  decision point for something the code already answers.
- **Leaning** — I have a real preference and a real reason, but the tradeoff is
  genuine and one sentence from you settles it. State what would flip me.
- **Need you** — outside my judgment: a scientific call, a product call, a
  tolerance for breakage, a question about what modellers actually do. Do
  **not** guess these, and do not bury them in a recommendation dressed up as
  confident. Say plainly what I cannot determine and why.

Be honest about which label applies. Marking a genuine unknown "solid" to seem
decisive is worse than the delay it saves; marking a clear call "need you"
wastes a decision the evidence already made.

## Related

- Verification discipline (paste the command, mark inference vs verified):
  [`docs/dev/agent-verification-conventions.md`](../../../docs/dev/agent-verification-conventions.md)
- Filing an incident rather than a review: the `incident-report` skill.
- Where reviews are archived: `docs/dev/reviews/`. Audit-fix commits cite these
  via an `Audit ref:` footer.
