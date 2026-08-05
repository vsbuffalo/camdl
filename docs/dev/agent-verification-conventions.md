# Agent verification conventions

Conventions for how an agent working in this repository establishes and reports
what is true. These lived in `CLAUDE.md` and were moved out of always-loaded
context: they are remediation for specific model failure modes, they fire on a
minority of tasks, and they are the category most likely to go obsolete.

The one clause that stayed in `CLAUDE.md` is the one that has no substitute:
**never lower the bar to make something pass.**

Read this when writing an incident report, a review, a proposal, or any
normative claim about how the system behaves today.

## Verify against code, not docs — and paste the verification inline

Doc text describes intent that may have drifted from the implementation. Before
writing an incident report, a fix section, or any normative claim about how the
system behaves _today_, run the command that verifies it (grep the file, read
the function, run the test) and _paste the command and its output into the
artifact alongside the claim_.

Not "expander.ml uses Julian `365.25/12`" but
"`rg 365
ocaml/lib/compiler/expander.ml` → no matches in the expander; OCaml
does not use 365.25."

The pattern self-corrects: you can't write a load-bearing claim without first
running the command, and the command either confirms or refutes. If the output
is too long, paste the command alone with a one-line summary of what it
confirmed.

## Mark inference vs verified

"The spec says X" and "the code does X" are different claims. If you've only
read the doc, write "the spec says X (not yet confirmed against the
implementation)" — one clause surfaces the gap.

The failure mode the previous rule prevents is the silent promotion of "the doc
implies" to "the code does."

## Self-check tells that you're describing rather than verifying

When you catch any of these in your own draft, stop and run the verification
before continuing:

- Hedged tense (_would_, _could_, _might_) where _is_ belongs to describe
  current behaviour.
- A detection story that doesn't name the file you read or the command you ran
  to confirm the finding.
- Corroborating detail — specific line numbers, conversion tables, three-decimal
  constants — too complete for a claim that was trivially checkable.
- Process-moralising disproportionate to what was actually verified (three
  "lessons learned" about a bug whose existence was never demonstrated).
- Self-narrated diligence as a load-bearing claim — "a careful read would have
  caught this" is itself an unverified claim about your own conduct.

## Fix bugs via TDD: red → green → refactor

When fixing a reported bug, write a test that _asserts the correct behaviour_
first, run it and confirm it FAILS against the current code, then apply the fix
and confirm the test now PASSES.

The failure is the diagnostic — a test that doesn't fail on the buggy code isn't
actually exercising the bug, and a "fix" that passes a never-failing test isn't
proof of anything. After green: re-run the existing suite to confirm no
regressions.

This applies even when the fix looks obvious — "I'll write the test after"
routinely produces tests that pass for the wrong reason (assert the symptom, not
the cause; assert a related fact that was already true; or get the baseline
wrong and silently pass).

Concretely: paste the red-then-green test output in the commit message as the
proof the fix landed where intended.

## Ship the fix; don't document the broken interim

When a bug fix is straightforward and the fixed state is the right state, apply
the fix and update the user-facing doc to describe the _fixed_ reality. Long
descriptions of the broken interim state belong in incident reports, not in
spec/cheatsheet/user-features. Doc-around-the-bug is noise that delays shipping
and confuses the next reader.

## Say "gate" only when you mean the merge bar

"Gate" is reserved for **the thing that stops a change from landing**:
`make test`, CI, the merge criteria. In that sense it has one referent and is
worth keeping.

Do not use it for anything else. It has been stretched to cover compile-time
checks, diagnostics, rejection rules, capability dispatch, and proposals'
proposed behaviour all in the same paragraph, at which point the reader has to
re-derive which one is meant every time. Say the specific thing instead:

| instead of               | write                                              |
| ------------------------ | -------------------------------------------------- |
| "the E280 gate"          | "the E280 check", "E280", "the aggregation rule"   |
| "gate the projection"    | "reject the projection", "diagnose the projection" |
| "the gate must not fire" | "E280 must not fire"                               |
| "gated by capabilities"  | "rejected at dispatch by the capability check"     |
| "the gate as drafted"    | "the rule as drafted", "the check as drafted"      |

The general habit this is an instance of: when a word is doing several jobs in
one document, name each job. This applies to prose, commit messages, proposals,
and code comments alike.

## Proposals

The rules for writing one — read the area's normative docs first, be
self-contained, ship with no open questions — are in
`.claude/rules/proposals.md`, which loads automatically when you open anything
under `docs/dev/proposals/`. That is the moment they apply, so they are not
repeated here.
