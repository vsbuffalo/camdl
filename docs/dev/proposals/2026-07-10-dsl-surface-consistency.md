---
status: decided (2026-07-10) — adversarially reviewed, GO on Phase 0 (gh#425) and the gh#423/schedule plan
date: 2026-07-10
authors: camdl core
related: gh#414 (block separators), gh#423 (forcing selectors), gh#424 (reserved words — deferred)
note: file:line citations are approximate (some drifted after the gh#425 diff); the claims verify against code.
---

# DSL surface consistency — findings and proposal

## Summary

Four investigations were run against the grammar (`ocaml/lib/compiler/lexer.mll`,
`parser.mly`) and spec: two deep dives on the known issues (block-member
separators, gh#414; forcing-block quoting, gh#423) and two sweeps for *other*
surface inconsistencies (lexical/value axis and structural/block axis). Every
finding below is grounded in `file:line`; the highest-impact ones were confirmed
by compiling snippets with `camdlc`.

**One root cause runs through most of it:** new blocks were added by copying a
nearby production rather than routing through a shared seam, so the surface
forked along several axes — cadence, header separators, the classifier colon,
string handling. The result is a language where a correct-looking model is often
a bare `E001 syntax error`, and where the reader must memorise which shape each
block uses. This is exactly the "keep the grammar in a head" property the DSL
philosophy targets.

The strongest signal: **both sweeps independently identified schedule/cadence
fragmentation as the #1 structural problem** — the same "when does this fire"
concept is spelled five incompatible ways. That convergence, not either known
issue, is the largest single win.

Recommended shape: a **phased** plan — cheap error-quality wins first (no grammar
change), then two shared-seam consolidations, then the two known-issue grammar
decisions, with a formatter as a later lever. Nothing here is committed; the
open decisions are collected in §6 for us to settle together.

---

## 1. The two headline issues

### 1.1 gh#414 — block-member separators

**Finding (verified).** `compartments` is the *sole* top-level `{}` block whose
members are comma-separated (`separated_list(COMMA, …)`, `parser.mly:266`); every
other block is `list(…)` — whitespace/newline. The same token is **required** in
`compartments { S, I, R }` and **rejected** in `parameters { a:rate, b:rate }`,
both as a bare `E001`. Newlines are pure whitespace (`lexer.mll:142-143`, no
`NEWLINE` token), so the parser never sees line structure.

The de-facto rule the grammar *almost* follows: **commas separate items inside a
delimiter pair (`[...]`, `(...)`); `{}`-block members are whitespace-separated.**
`compartments` breaks it. Two principled exceptions exist and should be named,
not erased: `columns { }` already allows *either* (an optional trailing comma,
`parser.mly:782-784`), and a probabilistic dest-branch `--> { D:w, … }` uses
required commas because it is an inline weight-*map* literal, not a statement
block.

**Feasibility of the maintainer's "comma OR newline" rule.** As a single uniform
grammar rule it is one of two things: either newline-significance (reject `S I R`
run-ons, enforce comma-or-linebreak) — which requires a `NEWLINE` token, a
bracket-depth lexer mode, and line-continuation machinery (Python/Haskell layout)
that is a rupture with camdl's one-page-grammar goal — or it degrades to
"comma optional everywhere," which *permits* comma-in-`transitions`, the thing we
don't want. **Recommend against newline-significance.**

**Recommendation: B + E (keep the set/statement distinction; make it principled
and signposted).**

- **B — forgive the trailing/newline comma in `compartments`** (`separated_nonempty_list(COMMA, …) ioption(COMMA)`). Purely additive, zero breakage; fixes the real multiline footgun (today a trailing comma after the last compartment is `E001`).
- **E — turn every bare separator `E001` into a directional diagnostic**, reusing the existing template at `parser.mly:1339-1352` (the scenario `set/scale` block already does this): "`parameters` separates members with whitespace/newlines, not commas — put each on its own line," and the converse for `compartments`.
- **Document the rule in prose** (spec + cheatsheet currently teach it only by example, which is *why* it's invisible until you hit `E001`): "a `{}` block is either a comma-separated **set** (`compartments`) or a whitespace-separated **statement list** (everything else); commas are required inside `[...]` and `(...)`."

**Why B+E over "comma optional everywhere" (C).** The set-vs-statement split is a
real semantic distinction (a *set of names* vs a *sequence of typed statements*),
and the mainstream precedents encode exactly this rather than one permissive
rule: Rust (comma fields + trailing comma, `;` statements), Stan (`,` args, `;`
statements), Nix (whitespace list elements, `;` bindings), odin (newline-separated
equations, commas only in `c(...)`/indices). No mainstream tool makes commas
*optional-and-equivalent*. C remains the runner-up if we prefer one maximally
simple permissive rule and accept a future lint to discourage comma-in-statements
(we have no `.camdl` formatter to enforce it today).

**Blast radius (B is additive, so this is only if we ever chose A/whitespace-only):**
117 `.camdl` files, 17 goldens, all spec examples use `compartments { S, I, R }`.

### 1.2 gh#423 — forcing-block quoted-vs-bare selectors

**Finding (verified).** In a `forcing` block the kwarg *values* mix quoted strings
and bare identifiers for the same category. The grammar collapses both to the
same AST node — `STRING` becomes `EIdent(s, dummy_loc)` (`parser.mly:1196`), a
bare word becomes `EIdent(name, real_loc)` — so `key_col = "village"` is *already
legal today* and produces identical IR to `key_col = village`. The only
discriminator is `dummy_loc`, which is the implicit-convention trap the codebase
elsewhere warns against.

The clean semantic line, from the kwarg classification (full table in the agent
report): **(a) foreign-file strings** — `data`, `method`, `time_col`,
`value_col`, `key_col`; **(b) model expressions** — `amplitude`, `period`,
`phase`, `values`, `on`, `harmonics`, `lag`, …; **(c) integer literals** —
`n_basis`, `degree`; **(d) model-name-as-string** — `table`, `time_dim` (bare,
but they name model constructs, so they must *stay* bare). Only the three column
selectors in (a) are the offenders; a naive "quote everything bare" would wrongly
sweep in (d).

`data`/`value_col`/`key_col`/`time_col` are consumed as **OCaml-expand-time
strings** and never cross the IR seam (the IR `Interpolated` stores only baked
`{times, values, method}` — no column fields). So the collision with model names
(`village`, `C`) is purely a *reader*-facing ambiguity, not a compiler name
clash.

**Recommendation: Option A — quote the foreign-file selectors** (`value_col = "C"`,
`time_col = "time"`, `key_col = "village"`), parallel to `data`/`method`. Give
forcing args a typed value (`farg_value = FStr of string | FExpr of expr`) so the
string-vs-expr distinction is *carried in the type* (parse-don't-validate),
confined to the forcing surface — no change to the global `expr` AST, dimcheck,
autodiff, or IR. Consumers then require `FStr` for (a), `FExpr(EIdent _)` for (d),
`FExpr` for (b). Rule for the reader: **quoted = outside world (file), bare =
inside the model.** Precedent: pomp (King et al. 2016) names the covariate time
column with a quoted string at the data boundary; odin/Stan handle files entirely
in the host and so never face this — camdl *does* read the file from inside the
DSL, which is exactly why it must name columns, and a quoted string is the
readable marker when a model language reaches out to an external column.

**Ship alongside:** the forcing kwarg handler is the *one* kwarg surface in the
compiler that does **not** reject unknown keys, so `value_column`/typos are
silently ignored and the selector falls back to its default. That is a
silent-wrong hole; add the unknown-kwarg check (mirror `expander.ml:854`).

---

## 2. Schedule/cadence surface (DOWNGRADED after review)

Both sweeps flagged this as the largest structural inconsistency. On closer
inspection (prompted by maintainer review), that framing is **overstated**: there
are genuinely **two** schedule types serving **different needs**, and they should
NOT be merged.

- **`schedule_core` (`every`/`at`) — "record at a cadence."** Used by `output`
  (trajectory snapshots) and `observations` emit. A simple regular cadence.
- **`schedule_decl` — "fire a state change."** Used by `interventions`/`events`:
  `SAtTimes` (fire at listed times), `SRecurring` (every P in a from–to window),
  `SEveryAtDay` (every P aligned to a calendar day — cohort entry on the 1st).
  The IR reflects the difference: interventions carry `end_` + `at_day`, output
  carries neither.

Firing a windowed, calendar-aligned cohort entry is legitimately richer than
"record every 7 days," so the two types earn their keep.

**What is genuinely accidental is narrower, and only one piece is a clean win:**

- **`until` vs `to`** for the recurring window end — the *same* `SRecurring`,
  two keywords (`parser.mly:987` builds `SRecurring(every, from?, until?)`;
  the set-block path spells the end `to`). **Verified: `until` appears in zero
  `.camdl` models and zero goldens as syntax — only in prose comments** — so
  retiring it (always `to`, matching `simulate { from … to … }`) breaks nothing
  and removes a real coin-flip. **Recommend: do this.**
- **`every =` vs bare `every`, `at = [...]` vs `at [...]`** — these *look*
  cosmetic but are entangled with a real structural difference: obs is
  `emit_schedule = every 7 'days` (a `key = value` whose value is the cadence),
  output is `trajectories { every = 7 'days }` (a field in a block). "Always
  `every =`" would produce `emit_schedule = every = 7 'days` (double `=`), so
  unifying is restructuring, not a keyword swap. **Recommend: park it.**

The rest of this section (the original "five spellings" framing) is retained
below for the record, but the recommendation is the narrow one above.

### (retained) the surface fork the sweeps found

| Surface | `every` | `at` | window end | site |
|---|---|---|---|---|
| `output.trajectories` | `every = E` | `at = [...]` | — | `parser.mly:1025-1029` |
| `observations` emit | `every E` (no `=`) | `at [...]` (no `=`) | — | `748-752` |
| intervention `set` block | `every = E` | `at = [...]` | **`to = E`** | `938-942` |
| intervention `transfer/add` recurring body | `every = E` | — | **`until = E`** | `919-922` |
| intervention `transfer/add(...) at` | — | `at [...]` (no `=`) | — | `804, 812` |
| event `add(...) every P at_day D` | `every P` (no `=`) | — | `at_day D` | `820` |

Two verified sub-inconsistencies: the recurring window end is `until` for
`transfer/add` bodies but `to` for `set` blocks (and `to` in `simulate`), both
building the identical `SRecurring` — a coin-flip the author memorises per
action; and `=` appears on `every`/`at` iff the schedule sits inside a `{}` body,
vanishing when it trails a `(...)` action (the split is *documented as
intentional* at `parser.mly:695-697`, which makes it more corrosive — it can only
be looked up, not reasoned out).

**Recommendation:** route every cadence-bearing block through one
`schedule_core`-style production (`every`, `at`, `from`/`to`), delete
`emit_schedule_spec`, the `until` spelling, and the bare `at`/`every … at_day`
tails (express `at_day` as a field in the shared block). This is the
reach-for-the-existing-seam rule applied to schedules and is the highest-leverage
consolidation in the whole survey.

---

## 3. Ranked catalog of the other findings

Severity = frequency × misleadingness. "✓" = reproduced against `camdlc`.

| # | Finding | Axis | Sev | ✓ | Direction |
|---|---|---|---|---|---|
| S1 | **Schedule cadence spelled 5 ways** (`until`/`to`, `every =`/bare, `at`/`at =`) | struct | HIGH→**narrow** | ✓ | §2 — **retire `until`→`to` only**; two types kept, rest parked |
| L2 | **Reserved words unusable as identifiers, bare `E001` no hint** — `count` (the natural case-count column), `rate`, `to`, `by`… ; rescued only in kwarg position (`kw_arg_name`) | lex | HIGH | ✓ | "`X` is a reserved word — rename" diagnostic, or soft-keyword in name slots |
| C2 | **`:` overloaded** — classifier (`beta : rate`) vs body-introducer (`inf : S-->I`, `vacc : transfer(...)`) | struct | ~~HIGH~~ | | **DESCOPED** — maintainer declined the churn |
| L7 | **Typo'd unit → raw apostrophe lex error; friendly `E102` "unknown unit" is unreachable dead code** (`'per_capita`) | lex | MED-HIGH | ✓ | lex general `'alnum+` so `E102` fires with its "expected one of…" list |
| L3 | **`STRING` collapses to `EIdent` + `dummy_loc`** — quoting inert in name slots, mandatory in path/date slots, no signal which; string args error with no location | lex | MED-HIGH | | typed `EStr of string*loc`; underlies gh#423 |
| C3 | **Params index positionally `[dim]` (single dim only); everything else binds `[v in dim]`** (`beta[a in age]` → `E001`) | struct | MED-HIGH | ✓ | accept the `[v in dim]` binder in `param_decl` |
| C4 | **Declaration header separator `:` vs `=` vs nothing** — quantities (`=`) vs observations (nothing), the "twins" disagree | struct | ~~MED-HIGH~~ | | **DESCOPED** — maintainer declined (bundled with C2) |
| L4 | **`read(...)` has two incompatible grammars** — rigid `STRING` args in `dimensions`, generic funcall (`EIdent` path, arbitrary kwargs) in `tables` | lex | MED | | parse `read` once as a funcall; validate shape in the expander |
| L5 | **External-data reference has three surfaces** — obs `from data` (bare, CLI-bound), forcing `data = "…"` (in-model string), table `read("…")` (call) | lex | MED | | converge on one in-model mechanism |
| C5 | **`#'` doc comments attach to only 7 of 15 declaration kinds** — documenting an intervention/forcing is `E001` | struct | MED | ✓ | thread `doc_opt` through the rest (mechanical) |
| L6/C? | **Enum-choice quoting split** — `method = "linear"` (quoted) is the sole quoted closed-enum; `integrator = rk45`, `format = tsv`, `~ poisson(...)` are bare | lex | MED | | make `method` bare to match; intersects gh#423 |
| C6 | **Inline-vs-block asymmetry** — transitions have full `@rate`/`via`/`{}` duality; interventions can't block-form `transfer` nor inline `set`; obs/forcing are block-only | struct | MED | | let the `{}` form carry any intervention action |
| L8 | **`'unit` scale-active on values/tables/forcings, scale-inert on param kinds** (a pure `[dim]` alias there); two spellings for one param dimension | lex | MED | | forbid tier-3 `'unit` on param kinds; keep `[dim]` |
| L9 | **`#` = comment / `#[attr]` / `#'` doc, disambiguated by one-char lookahead** — the two load-bearing forms *look* commented-out | lex | MED | ✓ | (low urgency) distinct markers if revisited |
| C7 | **`stratify(...)` is the only top-level paren-call** among `kw = value` / `kw { }` / `let` | struct | MED | | `stratify { by = …  only = […] }` |
| L11 | **No boolean literal** — `once = true` is a bare `EIdent("true")` interpreted late; `once = maybe` parses | lex | LOW-MED | | real `true`/`false` literals |
| L12 | **`null` → `EConst 0.0`** — reserved keyword, no documented DSL use, silent coercion | lex | LOW | | remove from keyword table, or document + check |
| L10/L13 | **`@` two meanings** (rate + doc-tag); **`:` range constructor** `lo:hi` inside `[...]` | lex | LOW | | note only |
| C8 | **`~` pooling suffix `\| dim` on priors, rejected on obs likelihoods** (deliberate, well-diagnosed) | struct | LOW | | leave |

---

## 4. Incidental bugs found (file separately — not part of the surface design)

These are concrete defects the sweeps turned up; each is independently
actionable:

1. **`method` enum doc-vs-code — AND a live silent-wrong hole (adversarial-confirmed).**
   Spec §7 advertises `method = "cubic_spline"` / `"pchip"`, but the Rust
   `InterpMethod` accepts only `linear`/`constant`/`spline` (`time_func.rs:25-31`),
   and OCaml stores the string raw with NO validation (`ir.ml:191`) — so
   `method = "cubic_spline"` fails to deserialize at the sim boundary, **and
   `method = "banana"` is accepted with zero validation.** The doctest exercises
   only camdlc (OCaml→IR), never the sim path. **This makes the #423 "bare
   method" step load-bearing:** making `method` bare is feasible (`linear` /
   `constant` / `spline` all parse bare, none are keywords), but it MUST also
   validate the value against `{linear, constant, spline}` in the OCaml
   expander — otherwise `method = banana` stays silently wrong. Purge
   `cubic_spline` / `pchip` from the spec (`camdl-language-spec.md:1170, 1325,
   1326`) in the same change, and add a sim-path golden.
2b. **`fourier` `harmonics` parens doc-vs-code (adversarial-found, new).** Spec
   `camdl-language-spec.md:1124` shows `harmonics = [(a1, b1), (a2, b2), …]`
   (parens), but the grammar rejects parens and requires 2-element *lists*:
   `harmonics = [[0.1,0.2],[0.05,0.0]]` compiles; the paren form is `E001`. Fold
   into the doc-sync (spec fix).
2. **`emit_schedule = at [...] 'unit` doc-vs-code.** Spec line 2665 documents a
   trailing unit after the bracket; the grammar (`parser.mly:751`) has no such
   slot (the unit must ride on the list elements). `at [7,14] 'days` → `E001`.
   Fix the spec or add the slot.
3. **Dead `E102` (= L7 above).** The friendly "unknown unit" diagnostic never
   fires; users get a raw apostrophe lex error.
4. **Forcing unknown-kwarg silent-ignore (= gh#423 companion).** Typo'd forcing
   kwargs are dropped silently.

---

## 5. Prioritized roadmap

**Phase 0 — error-quality, no grammar change (ship independently, high value).**
These make the *existing* surface honest without changing what parses: the
directional separator diagnostics (E), the reserved-word "X is reserved"
diagnostic (L2), resurrecting `E102` (L7), and the forcing unknown-kwarg check
(gh#423 companion). All are pure wins, no migration.

**Phase 1 — shared-seam consolidations (the structural payoff).**
(1) the schedule `schedule_core` unification (S1) — the biggest single
consolidation; (2) the typed `EStr` string node (L3), which also unlocks the
gh#423 fix cleanly.

**Phase 2 — the two known-issue grammar decisions.**
gh#414 (B: forgive trailing comma + keep the set/statement split) and gh#423
(Option A: quote the file-column selectors). Both are breaking-ish and
signpostable; both want a `docs/language-changes.md` entry.

**Phase 3 — bigger or optional.**
The `[v in dim]` param binder (C3), threading `#'` docs everywhere (C5), the `:`
classifier/body de-overload (C2) and the header-separator rule (C4) — these are
larger and interact, so batch them behind a decision. A `.camdl` formatter
(`camdl fmt`) is the cross-cutting lever that makes the whole surface uniform
regardless of which grammar decisions we make; it is gated on comment
preservation (the lexer currently discards `#` comments).

---

## 6. Decisions — RESOLVED (2026-07-10 maintainer review)

1. **gh#414 → B+E.** Keep the set-vs-statement distinction (`compartments` =
   comma set; everything else = whitespace statements), forgive the trailing
   comma in `compartments`, and teach the boundary with directional diagnostics.
   Not C ("comma optional everywhere" would admit comma-in-`transitions`).
2. **gh#423 + enums → quote external strings, bare enums.** Quote the file-column
   selectors (`value_col = "C"`); make **all closed enums bare** DSL-wide
   (`method = linear`, matching `integrator = rk45`, `format = tsv`). Net forcing
   rule: **quoted = external file** (`data`, column selectors), **bare =
   enum-or-model** (`method`, `table`, `time_dim`, all model exprs).
   **REQUIRED (adversarial review):** the bare-`method` step must ALSO validate
   the value against `{linear, constant, spline}` in the expander and purge
   `cubic_spline`/`pchip` from the spec — else `method = banana` stays silently
   accepted (the E409 kwarg-*name* check does not cover the *value*).
   **Migration touches real sources** (IR is byte-identical, but `.camdl`/docs
   are not): `ocaml/golden/flu_data_forcing.camdl:34-36` (`time_col`/`value_col`
   bare → quoted; `method` quoted → bare) + spec examples at
   `camdl-language-spec.md:1168-1170, 1253-1254, 1325-1326` — land atomically
   with a `docs/language-changes.md` entry and `make update-golden` if the golden
   source regenerates.
3. **Schedules → narrow.** Keep the two types (they serve different needs — record
   cadence vs fire-timing; §2). Retire **`until` → `to`** only (verified unused in
   all models/goldens). **Park** the `every=`/`at=` unification (entangled with a
   real structural difference, not worth the restructuring).
4. **Reserved words (L2) → better error only.** "`count` is reserved — rename it."
   Soft-keywords (letting `count`/`rate` name things) deferred to a follow-up
   issue.
5. **`:`/header cleanup (C2/C4) → descoped.** No strong case for the churn.

Immediate action: **Phase 0** (error-quality, no migration) is in flight — the
directional separator diagnostics (decision 1's E), the resurrected `E102`, the
forcing unknown-kwarg check, and the reserved-word hint (decision 4).

---

## Appendix — before / after syntax at a glance

The point of the recommended plan is that **the everyday syntax you already
write barely changes.** Most of the work is better *errors* and one new
consolidation (schedules). Here is every proposed surface change side by side.

### gh#414 — separators (recommended: B+E)

Everyday code is **unchanged**. Two things get better:

```camdl
# UNCHANGED — the two conventions stay exactly as they are today:
compartments { S, I, R }          # a SET of names → commas (as now)
parameters {                       # a STATEMENT list → whitespace/newlines (as now)
  beta  : rate
  gamma : rate
}

# NEW (B): a trailing comma in a multiline compartments list no longer errors.
compartments {
  S,
  I,
  R,        # <- today this trailing comma is a bare E001; after B it's fine
}

# NEW (E): using the wrong separator gives a real message instead of `E001`:
#   parameters { a : rate, b : rate }
#   before:  error[E001]: syntax error
#   after:   error: `parameters` separates members with whitespace/newlines,
#            not commas — put each on its own line.
#
#   compartments { S I R }
#   before:  error[E001]: syntax error
#   after:   error: `compartments` members are comma-separated — write
#            `compartments { S, I, R }`
```

**Plain-language version of the whole gh#414 debate:** today `compartments`
demands commas and every other block forbids them, and when you get it wrong the
compiler just says "syntax error." The disagreement is *why*: is that a bug to
unify, or a real distinction (a *set of names* vs a *list of statements*) to keep
and just explain? The recommendation keeps the distinction (it matches how Rust,
Stan, Nix, and odin all work) but makes it learnable — a helpful error either
way, plus a trailing-comma fix so multiline lists stop biting. The alternative
(C) is "let a comma be optional in *every* block" — one simpler rule, but it
would also start accepting `transitions { infect : … , recover : … }`, which you
said you don't want.

### gh#423 — forcing column selectors (recommended: quote them)

```camdl
# before                              # after
forcing {                             forcing {
  C[v in village] : interpolated 'ratio {   C[v in village] : interpolated 'ratio {
    data      = "series.csv"                  data      = "series.csv"
    method    = "constant"                    method    = constant        # enum → BARE
    key_col   = village          →            key_col   = "village"       # file column → quoted
    time_col  = time                          time_col  = "time"
    value_col = C                             value_col = "C"
  }                                         }
}                                     }
# Rule the reader learns: quoted = names something OUTSIDE (a FILE or a file
# COLUMN); bare = a closed enum OR something INSIDE the model (param, compartment,
# dimension, table). So `method = constant` (enum) and `table = mymatrix` /
# `time_dim = age` (model names) are bare; only the file path + column names quote.

# Also new: a typo is now caught instead of silently ignored.
#   value_column = C   →   error: unknown forcing argument `value_column`
#                          (expected: data, method, time_col, value_col, key_col)
```

### Schedules (recommended: narrow — retire `until`, keep everything else)

The two schedule types stay (they serve different needs; §2). The ONE change is
retiring the redundant `until` keyword for the recurring window end:

```camdl
# before — the recurring window end spelled two ways (same underlying SRecurring):
vacc  : transfer(from=S, to=V, fraction=0.1) { every = 30 'days  until = 90 'days }   # `until`
pulse : { S = S*0.5  every = 30 'days  to = 90 'days }                                # `to`

# after — always `to` (matching `simulate { from … to … }`); `until` retired
# (verified: it appears in zero models/goldens as syntax):
vacc  : transfer(from=S, to=V, fraction=0.1) { every = 30 'days  to = 90 'days }
```

PARKED (not doing): the `every =` vs bare `every` / `at =` vs `at [...]` spelling
split — it is entangled with a real structural difference (`emit_schedule =
<cadence>` field vs a `{ }`-block field), so unifying it is restructuring, not a
keyword swap, and not worth it.

### Other sweep items that change syntax (lower priority — Phase 3)

```camdl
# C3 — parameters could take the same `[v in dim]` binder as everything else:
parameters { beta[age] : rate }        # before: positional, single dim only
parameters { beta[a in age] : rate }   # after: same binder as transitions/obs/…

# L2 — reserved words as identifiers (Phase 0 improves the ERROR only, per decision 4):
observations { … columns { time : time, count : count } }   # before: bare E001 on `count`
                                                            # after Phase 0: "`count` is reserved — rename"
                                                            # (allowing it as a name = deferred follow-up)
```
(The `:`/header cleanup that was sketched here is **descoped** — see §6 decision 5.)

