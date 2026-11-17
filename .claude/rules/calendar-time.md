---
paths:
  - "rust/crates/ir/src/caltime.rs"
  - "docs/dates.md"
description: Calendar / time / date changes — the policy document, the unit table, the conversion code, and in-flight design
---

# Calendar and time

## Required reading

- [`docs/dates.md`](../../docs/dates.md) — the policy document.
- [`docs/camdl-language-spec.md`](../../docs/camdl-language-spec.md) §2.1 — the
  unit table.
- `rust/crates/ir/src/caltime.rs` — the conversion code.
- [`docs/dev/temporal-architecture.md`](../../docs/dev/temporal-architecture.md)
  — observation periods, training membership, forecast origins, targets and
  horizons; which layer owns each rule.

Background design, **archived** (decided; read for rationale, not for pending
work):

- `docs/dev/proposals/archive/post-alpha/2026-05-22-calendar-time.md`
- `docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`

## Cross-language constants

`caltime.rs::rata_die` is the pattern: single source of truth, mirror only with
an equivalence test. Never two independent copies of a constant.

## A calendar step is resolved by type once, and named when composed

A single whole-number calendar step from a date the compiler knows
(`origin + 6 'months`, `date("2020-01-31") + 1 'months`, `to = 2 'years` in an
instant position, `every = 1 'months` from a known start) is resolved by the
position's type to `add_calendar_months` / `add_calendar_years`, because it has
one meaning (the one-step calendar rule). Composed calendar arithmetic (two
steps, a step mixed with days) is written with the named functions, because
there the order of operations changes the answer and the reader needs to see it;
the compiler refuses the operator form with a hint. See
`.claude/rules/dsl-surface.md`, and "The one-step calendar rule" in
`docs/dates.md`. (Cadences, `every = 1 'months`, resolve from Stage 2 of
`docs/dev/proposals/2026-11-16-calendar-durations-by-position.md`; until then
they are refused, E322.)
