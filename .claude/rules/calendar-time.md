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

Background design, **archived** (decided; read for rationale, not for pending
work):

- `docs/dev/proposals/archive/post-alpha/2026-05-22-calendar-time.md`
- `docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`

## Cross-language constants

`caltime.rs::rata_die` is the pattern: single source of truth, mirror only with
an equivalence test.
