# Methods notes

Deep methodological dives on specific algorithmic or statistical choices in
camdl. Sits between the narrative `docs/inference.md` (which gives a tour for
modellers) and the proposals in `docs/dev/proposals/` (which are design
decisions at a particular point in time). A methods note is for the long-lived
"here is how this actually works, with the formula and a worked example" writeup
that every developer or reviewer will want once.

Scope:

- Each note stays in a single file under `docs/methods/<slug>.md`
- Each cites the authoritative source file + line numbers for the
  implementation, so the note and the code can be cross-checked
- Each includes at least one worked numerical example
- Each owns a specific methodological question, not a feature or a subcommand
  (use `docs/inference.md` or the specs for those)

Current notes:

- [`cooling.md`](cooling.md) — IF2 cooling schedule: the formula, its relation
  to pomp's `cooling.fraction.50`, and worked numbers.
- [`particle-methods.md`](particle-methods.md) — the four particle- method
  implementations (bootstrap PF, IF2's parameter-augmented loop, CSMC-AS for
  PGAS, correlated PF for correlated-MH PMMH); algorithm equations, file:line,
  when-to-use-which, full citations.

Add new notes by convention: create `docs/methods/<topic>.md`, add a line here
pointing at it, commit both together.
