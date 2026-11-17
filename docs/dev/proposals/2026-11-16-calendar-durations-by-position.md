# Proposal: calendar durations resolved by the position they are written in

**Status:** Accepted 2026-11-16 in stages. **Stage 1a** (typed positions, the
census, `Lower` per kind, and the refusal of every calendar duration in an
instant or cadence position) is approved for implementation now. **Stage 1b**
(the one-step calendar rule, §8) is approved on the conditions stated there and
lands only after 1a. Stage 2 follows 1b. Every decision of §9 is recorded there.
Built on decisions D62 (a), D65 (a), D68 and D70 (a). PR #1157 (gh#1148) has
landed: `on` endpoints of a periodic forcing are model time, binned by `step`,
with misaligned ranges refused as E406.

**Verified against:** `main` at `842d3b66a`, and PR #1157's branch at
`e035d97d3` for the periodic-forcing claims. _Verified (run)_ means compiled
with that commit's `ocaml/_build/default/bin/camdlc.exe` on a scratch model
under `rust/target/agent-scratch/cal/`; _verified (read)_ means read in the code
at that commit; _inferred_ means neither. The commands are in the
[appendix](#appendix-verification-transcript).

## Summary

camdl has two months, by design. A unit literal `1 'months` is a fixed average
length, 365.2425/12 = 30.436875 days. `add_calendar_months(d, 1)` steps a date
along the calendar to the same day of the next month, clamping at month ends.
Under `origin = date(...)` the compiler refuses a month literal wherever it
would move a date (E321) or set a repeating schedule (E322), and its hint tells
the modeller which function to write instead. The compiler already knows the
answer it is asking the modeller to type.

This proposal gives every expression that holds a time a _position kind_ (an
instant, a recurring cadence, an offset from a run-time instant, a span, an
interval, a length) carried in the AST's types, and gives each kind one lowering
function with one rule. Under an origin, a single whole-number calendar step
from a date the compiler knows is resolved to the calendar function
automatically: `simulate { to = 2 'years }` means
`add_calendar_years(origin, 2)`, and `every = 1 'months` from a known start
means the dates `add_calendar_months(start, i)`. Everything whose meaning is not
settled by the position stays a hard error with a hint: fractional months, a
month mixed with days, two calendar steps in a row, and any calendar step from
an instant known only at run time. There is no override flag; the explicit
spellings (`730 'days`, `add_calendar_years(origin, 2)`) remain the escape.

The change lands in two steps so that no accepted model changes value silently.
Stage 1a gives every position its kind and refuses, under an origin, every
calendar duration in an instant or cadence position, including the spellings
that compile today with the average month (`simulate { to = 2 'years }`,
`at [6 'months]`); each refusal's hint names the explicit form. Stage 1b then
gives those spellings the calendar meaning by the _one-step calendar rule_: a
model that compiles under 1a contains none of them, so 1b changes the value of
no model that has been compiled since 1a.

Because the checks move from a hand-written list into the types, the class of
bug in which a time position is missed by the checker (five fixed issues, three
open ones, and seven more positions found while drafting this) cannot recur: a
new block that takes a time does not compile until its field has a kind.

## 1. What is wrong today

### 1.1 The modeller translates by hand what the types already determine

`add_calendar_months` exists because adding a month to a date is not a fixed
offset, and composing such additions does not behave like arithmetic. Two
verified examples, origin 2020-01-01:

| expression                                                                  | date       | t   |
| --------------------------------------------------------------------------- | ---------- | --- |
| `add_calendar_months(add_calendar_months(date("2020-08-31"), 6), 6)`        | 2021-08-28 | 605 |
| `add_calendar_months(date("2020-08-31"), 12)`                               | 2021-08-31 | 608 |
| `add_calendar_years(date("2020-02-29"), 1)` (26 Feb + 3 days, then a year)  | 2021-02-28 | 424 |
| `add_calendar_years(date("2020-02-26"), 1) + 3 'days` (a year, then 3 days) | 2021-03-01 | 425 |

Two six-month steps are not one twelve-month step, and a year-then-three-days is
not three-days-then-a-year. That is the reason for the named functions (§2.3 of
the language spec, `.claude/rules/calendar-time.md`), and it remains true.

But a single step from a known date has exactly one calendar meaning. When a
modeller writes `simulate { to = origin + 2 'years }` they get (verified, run):

```
error[E321]: calendar duration `2 'years` cannot translate an instant in simulate.to

  = hint: calendar months/years aren't invertible spans (e.g. date("2021-01-31") + 1 month = date("2021-02-28") because day-31 clamps to day-28 in Feb 2021). For a calendar-exact date use add_calendar_years(d, N). For an explicit affine span use 730 'days (≈ 2 'years).
```

followed by a spurious E401 and E602, because `simulate { to }` does not accept
instant arithmetic at all (`docs/dates.md`, "`origin` as a referenceable
identifier"). The hint already names the function and the number. The language
asks a human to perform a translation that the compiler has fully determined.

### 1.2 Dropping `origin +` silently changes the answer (gh#847)

Writing the duration alone in the same position is accepted and folded with the
average month (verified, run; origin 2020-01-01):

| `simulate { to = ... }`           | `t_end` today | date             |
| --------------------------------- | ------------- | ---------------- |
| `2 'years`                        | 730.485       | 2021-12-31 11:38 |
| `add_calendar_years(origin, 2)`   | 731           | 2022-01-01       |
| `18 'months`                      | 547.86375     | 2021-07-01 20:44 |
| `add_calendar_months(origin, 18)` | 547           | 2021-07-01       |
| `origin + 2 'years`               | E321 + E401   | —                |

So E321 guards one spelling of "two years after the origin" and the shorter
spelling walks past it to a horizon half a day before New Year. `docs/dates.md`
lists this under "Known implementation deviations".

### 1.3 Every bug in this area was a position the checker did not visit

The calendar checks run in `check_surface_time_typing`
(`ocaml/lib/compiler/expander.ml`), which walks a hand-written list of
declaration fields. Each fix in this area was a field added to that list:
`covers` widths (gh#1069), output `every` (gh#1088), reactive `window`, `after`
and `cooldown` (gh#1101), `quantities` (gh#1096), `at_day` (gh#1120). Still
open: gh#1128 (`value_at(I, 60)` draws no W325, verified: exit 0, no warning,
while `at [60]` draws W325), gh#1129 (output `at` and scenario `to` refuse the
legal `date(a) + 7 'days` with E401, verified), gh#1133, gh#1140, gh#1097,
gh#1102 and gh#847.

Drafting this proposal found seven more positions in which a month literal is
folded with the average and nothing is said (all verified, run, origin
2020-01-01; `t = 182` is 2020-07-01, six calendar months after the origin):

| what the modeller writes                                      | compiles to today                            |
| ------------------------------------------------------------- | -------------------------------------------- |
| `scenarios { late { simulate { to = 30 'months } } }`         | `t_end = 913.10625` (calendar: 912)          |
| `interventions { b : ... at [6 'months] }`                    | fires at 182.62125                           |
| `quantities { q = value_at(I, 6 'months) }`                   | reads t = 182.62125                          |
| `forcing { f : piecewise ... { breakpoints = [6 'months] } }` | breakpoint 182.62125                         |
| `parameters { tau : instant in [0 'days, 6 'months] }`        | upper bound 182.62125                        |
| a rate `if t >= 6 'months then 0.5 else 1.0`                  | switch at 182.62125                          |
| `timepoints { peak = origin + 6 'months }`                    | accepted; the block is parsed and dropped (` |

Scenario `simulate { to }` is not in the walk at all; `origin + 30 'months`
there is refused only because E401 refuses all instant arithmetic in that
position.

The list is the defect. Every new position is another place to forget, and the
sibling sweep that follows each fix is drawn from the checker's own list, so it
cannot find a position the list never contained (the review of #1094 says so).

### 1.4 Periodic `on` ranges are resolved by a loop of their own (gh#1140, gh#1155, gh#1156)

On PR #1157's branch a range-form periodic forcing reads `on = [lo:hi]` as model
time and converts it to bins of width `step`. Two defects remain there, both
verified (run) on `e035d97d3`:

- **gh#1155.** Each endpoint is truncated to a whole model-time unit with
  `int_of_float`. With `time_unit = 'weeks`, `period = 364 'days`,
  `step = 1 'weeks`, `on = [10 'days : 20 'days]` compiles and switches on bins
  1 and 2, days 7 to 20: the window opens three days early. The same truncation
  turns `on = [1 'months : 2 'months]` (daily bins) into bins 30 to 60, the
  average month floored.
- **gh#1156.** The compiler builds `round(period / step)` bins of width `step`;
  the runtime (`periodic_bin`, `rust/crates/sim/src/propensity.rs`) reads them
  as bins of width `period / len(values)`. With `period = 365 'days`,
  `step = 7 'days`, `on = [350 'days : 356 'days]` the compiler switches on bin
  50 of 52, which it means as days 350 to 356; the runtime's bin 50 is days
  350.96 to 357.98. With the spec's own `period = 365.25 'days, step = 1 'days`
  the runtime bin is 1.000685 days wide, so a window late in the year is moved
  by up to a quarter of a day.

`date_range(start, end)` and a `bspline` `window = [a, b]` are the other two
user-written intervals; each validates its own pair (gh#1140).

## 2. Position kinds

Every expression that holds a time is one of seven kinds. The kind is a property
of the position, not of the expression: `6 'months` is an instant in
`simulate { to }`, a cadence in `every`, and a length in `dt`.

| kind               | what the value is                                                     | positions                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instant**        | a point on the axis; a bare duration `D` means `origin + D`           | `simulate { from, to }`; scenario `simulate { to }`; intervention and event `at [...]` entries; recurring `from` and `to`; output `trajectories { at }`; `emit_schedule = at [...]`; `value_at`'s time; piecewise and bspline `breakpoints`; bounds of an `instant`-kind parameter; the operand compared with model time `t`; `timepoints` entries |
| **Cadence**        | the step of a recurring schedule whose start is known at compile time | recurring `every`; `every … at_day …` period; `emit_schedule = every`; output `trajectories { every }`                                                                                                                                                                                                                                             |
| **Runtime offset** | an offset from an instant known only during the run                   | reactive `after`, `cooldown`; the offset on `first_obs` / `last_obs`; arithmetic on model time `t`                                                                                                                                                                                                                                                 |
| **Span**           | the width of a period that tiles or trails the axis                   | `covers` width; `sum_observed(window = …)`                                                                                                                                                                                                                                                                                                         |
| **Interval**       | a user-written pair whose order and bounds are validated together     | periodic `on = [lo:hi]` (pair of phases); `bspline` `window = [a, b]` and `date_range(start, end)` (pairs of instants)                                                                                                                                                                                                                             |
| **Phase**          | an offset within a repeating period                                   | periodic `on` endpoints (D70 a); sinusoidal `phase`; `at_day`                                                                                                                                                                                                                                                                                      |
| **Length**         | a fixed length, average by design (D65 a)                             | `simulate { dt }`; periodic and sinusoidal `period`, periodic `step`; forcing `lag`; `via` delay `mean`; bounds of a `duration`-kind parameter; table values; prior arguments; rate literals (`'per_month`)                                                                                                                                        |

### 2.1 The rule for each kind under an origin

Without an origin nothing changes: there is no calendar, every month literal is
the average, and the rules below are vacuous (the archived typed-time proposal
§1.1, decision of record).

The table states the rule once Stage 1b and Stage 2 have landed. Under Stage 1a
the Instant and Cadence rows read **refused, always**, with a hint naming the
explicit form, and every other row is as below (§8, Stage 1a).

| kind           | a calendar duration (`'months`, `'years`) is …                                                                                                                                                    | refused, with a hint                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Instant        | **resolved**: `[B ±] k 'months` with `B` `origin` or a `date(...)` literal (or a `let` of one; default `origin`) and a whole month count `k` is `add_calendar_months(B, ±k)`; years are 12 months | fractional months; months mixed with days; a second calendar step; a base known only at run time |
| Cadence        | **resolved**: `every = k 'months` from a whole-day start `s` gives entries `add_calendar_months(s, i·k)`, counted from `s` (the `date_range` rule, gh#1068)                                       | fractional months; months mixed with days; a start that is not the start of a day                |
| Runtime offset | **refused** (the base is known only at run time; §4.3)                                                                                                                                            | always (E321)                                                                                    |
| Span           | **refused** (a calendar month ending at `t` has a length that depends on `t`; planned calendar `covers` forms, D45)                                                                               | always (E321)                                                                                    |
| Interval       | each endpoint by its own kind; the pair is validated once (§4.5)                                                                                                                                  | as the endpoint's kind, plus order and bounds                                                    |
| Phase          | **refused** where the phase is one fixed offset (`at_day`, gh#1120; a periodic `on` endpoint, which must be a whole unit); average for sinusoidal `phase` (decision G)                            | as stated                                                                                        |
| Length         | **average** (D65 a)                                                                                                                                                                               | never on calendar grounds                                                                        |

## 3. Types

### 3.1 A time position is a typed value, not an expression

A new module, `ocaml/lib/compiler/time_pos.ml`, owns the kinds. A position is a
GADT-tagged wrapper around the expression the modeller wrote, so that its kind
is known statically (a lowering function cannot be handed the wrong kind) and at
run time (generic walkers such as `inspect`, the declared-model builder and the
LaTeX renderer can still read it):

```ocaml
(* time_pos.ml *)
type instant
type cadence
type runtime_offset
type span
type phase
type length

type _ kind =
  | Instant        : instant kind
  | Cadence        : cadence kind
  | Runtime_offset : runtime_offset kind
  | Span           : span kind
  | Phase          : phase kind
  | Length         : length kind

(* Private: the parser and the forcing/reduction schemas (§3.2) construct one;
   only [Lower] folds one to a number. *)
type 'k t = private { kind : 'k kind; expr : Ast.expr; loc : Ast.loc }

val make : 'k kind -> Ast.expr -> Ast.loc -> 'k t
val expr : _ t -> Ast.expr          (* read-only view for printers and walkers *)

(* An interval is a pair of positions of one kind. *)
type 'k interval = { lo : 'k t; hi : 'k t; iloc : Ast.loc }
```

The AST fields that hold a time change type. Before and after, for the fields
that §1.3's bugs were in:

```ocaml
(* before *)
type simulate_decl = { sim_from : expr; sim_to : expr; sim_dt : expr option; ... }
type schedule_decl =
  | SAtTimes    of expr list
  | SRecurring  of expr * expr option * expr option
  | SEveryAtDay of expr * expr
type schedule_core = SchedEvery of expr | SchedAt of expr list
type reactive_decl = { ...; rxafter : expr option; rxcooldown : expr option; ... }
type obs_covers = { ...; ocv_span : expr option; ... }
type scenario_field = ... | ScTEnd of expr

(* after *)
open Time_pos
type simulate_decl = {
  sim_from : instant t; sim_to : instant t; sim_dt : length t option; ... }
type schedule_decl =
  | SAtTimes    of instant t list
  | SRecurring  of { every : cadence t; from : instant t option; until : instant t option }
  | SEveryAtDay of { every : cadence t; at_day : phase t }
type schedule_core = SchedEvery of cadence t | SchedAt of instant t list
type reactive_decl = {
  ...; rxafter : runtime_offset t option; rxcooldown : runtime_offset t option; ... }
type obs_covers = { ...; ocv_span : span t option; ... }
type scenario_field = ... | ScTEnd of instant t
type timepoint_decl = { tpname : string; tptime : instant t; tpspan : loc }

(* A parameter's bounds take the kind of the parameter, which the same
   production parses. *)
type bounds =
  | Instant_bounds of instant t * instant t     (* pkind = PInstant *)
  | Length_bounds  of length t * length t       (* pkind = PDuration or [T] *)
  | Value_bounds   of expr * expr               (* every other kind *)
```

The parser builds each wrapper in the production that already knows which field
it is parsing, so no table of positions exists anywhere.

**Where the modules live.** The declaration records that hold a `'k Time_pos.t`
are in `ast.ml`, and `Time_pos.t` holds an `Ast.expr`, so a separate
`time_pos.ml` would be a module cycle. The kinds, the private wrapper, `make`,
`expr` and `interval` are therefore a submodule `Ast.Time_pos`, as `Ast.Pos_int`
is. The census (§3.3), the forcing keyword roles and the reduction table (§3.2)
are in `ocaml/lib/compiler/time_census.ml`, which reads only `Ast`. `Lower`
(§3.5) is a submodule of `expander.ml`, because every lowering function needs
the expander's `ctx`, `resolve_expr` and diagnostics.

**Which lengths are typed.** A length is a typed field where a declaration holds
one: `simulate { dt }`, the bounds of a `duration`-kind or `[T]`-annotated
parameter, and the forcing keywords whose role is a length (`period`, `step`,
`lag`). A unit literal inside an ordinary expression (a rate, a prior's
arguments, a table value, a `via` mean) is not a position: the average is the
only rule that can apply to it (D65 a), dimcheck already judges it, and the
census does not list it.

### 3.2 Positions inside expressions

Three families of time position sit inside ordinary expressions rather than in a
declaration field. Each gets a closed variant, so the OCaml exhaustiveness check
covers it:

- **Reduction arguments.** `temporal_reduction_names` becomes a variant
  `reduction = Final | Mean | Integral | Value_at | Time_of_max | ...` with
  `reduction_args : reduction -> arg_role list`, where
  `arg_role = Series | Threshold | Time of instant kind`. `value_at`'s second
  argument is `Time Instant`; the quantity classifier obtains it as an
  `instant t` from that table, never as a bare `expr`.
- **Forcing keyword arguments.** `fkind` becomes a variant
  (`Periodic | Sinusoidal | Piecewise | Bspline | Interpolated | ...`) and
  `Forcing_schema.role : fkind -> string -> kwarg_role option` gives every
  keyword its role (`breakpoints` and interpolated `times` → instants, `window`
  → an instant interval, `on` → a phase interval, `period`, `step`, `lag` →
  lengths, `phase` → a phase, `data`, `time_col` → file). An unknown keyword is
  already an error; a known one can only be folded through its role.
- **Comparisons with model time.** In any expression, `t <op> e` or `e <op> t`
  with `<op>` a comparison makes `e` an instant position, and `t ± d` makes `d`
  a runtime offset. This is syntactic (`EIdent ("t", _)`), so it needs no change
  to `Time_typing.classify` (whose redesign, gh#1097 part 1, is out of scope).

### 3.3 How a new block cannot be missed

Two mechanisms, one per family:

1. **Fields.** A time held in a declaration field has type `'k Time_pos.t`. The
   only functions that turn one into a number are in `Lower` (§3.5);
   `resolve_time_expr`, the untyped fold all these positions share today, is
   deleted. A new block whose field is typed `expr` and folded with
   `resolve_float_expr` would still compile, which is why there is a second
   mechanism.
2. **The census.** `Time_pos.census : Ast.declaration list -> any_position list`
   enumerates every time position in a model, by an exhaustive match over
   `Ast.declaration` and every record it contains, compiled with warning 4
   (fragile match) and warning 9 (missing record fields) as errors. Adding
   `DFoo` to `declaration`, or a field to a record the census reads, fails the
   OCaml build until the census says what kind each new time field is. The
   census drives one property test (§8, test 1.1): every `Lower` function
   records the position it lowered (its kind, location and expression) in the
   compilation context, and for every position the census lists, the record
   holds that position with the same kind. A site that folds a time without
   going through its kind's function leaves its position unrecorded, and the
   test fails. Because `resolve_time_expr` is deleted, `Lower` is the only path
   from a time field to a number, so the recorded value is the IR's value.

```ocaml
type any_position = Any : 'k t -> any_position
val census : Ast.declaration list -> any_position list
```

### 3.4 The resolver (Stage 1b)

The rule is a pure function of a reading of the expression. The reading is taken
from the leaves before any unit is folded, which is where provenance still
exists (the reason `time_typing.ml` runs before `resolve_expr`):

```ocaml
(* calendar_reading.ml *)
type ymd = int * int * int

type base =
  | Origin                     (* `origin`, or the implicit base of a bare duration *)
  | Date of ymd                (* a `date(...)` literal, or a `let` of one *)
  | Stepped of ymd             (* the result of an `add_calendar_*` call *)
  | Runtime                    (* `t`, an instant-kind parameter, an observation anchor *)

type duration = {
  days   : float;              (* exact part: 'days, 'weeks, Instant − Instant *)
  months : float;              (* calendar part: 'months + 12 · 'years *)
}

type reading =
  | Moved    of { base : base; moves : duration list }  (* in source order *)
  | Duration of duration
  | Bare     of float                                    (* a unitless number *)
  | Not_constant                                         (* parameters, compartments *)

val read : Time_typing.env -> Ast.expr -> reading

(* Evidence that the model is anchored. Constructed only from
   `origin = date(...)`; every calendar conversion requires one, which closes
   gh#1133 by type: no caller can obtain a model time from a date without it. *)
type anchor = private { origin : ymd; origin_rd : int; unit : Ast.unit_lit }
val anchor_of_ctx : ctx -> anchor option

type refusal =
  | Fractional_months   of { months : float; shown : string }
  | Mixed               of { shown : string; base : ymd;
                             calendar_first : ymd; days_first : ymd }
  | Two_calendar_steps  of { shown : string; base : ymd; stepwise : ymd; at_once : ymd }
  | Calendar_from_runtime of { shown : string; base : string }
  | Calendar_span       of { shown : string; position : string }
  | Not_whole_day_start of { t : float; date : string }
  | Not_whole_unit      of { value : float; unit : Ast.unit_lit }
  | Out_of_order        of { lo : float; hi : float }
  | Out_of_period       of { lo : int; hi : int; last : int }

(* The instant rule. *)
val resolve_instant : anchor -> reading -> (float, refusal) result
```

`resolve_instant` is, in full:

- `Duration d` is read as `Moved { base = Origin; moves = [d] }`.
- `Moved { base = Origin | Date b; moves = [{ days = 0; months = k }] }` with
  `k` a whole number: `rd(add_calendar_months_ymd b k) − origin_rd`, divided by
  `days_per_unit unit`.
- `Moved { base = Origin | Date _ | Stepped _; moves }` with every move
  calendar-free: the base plus the sum of the days, as today.
- `months` not a whole number: `Fractional_months`.
- a move with both parts, or a calendar move beside an exact one: `Mixed`,
  computed both ways for the note.
- a calendar move on a `Stepped` base, or two calendar moves:
  `Two_calendar_steps`, computed both ways for the note.
- a calendar move on a `Runtime` base: `Calendar_from_runtime`.

Years are folded into months because `add_calendar_years_ymd (y, m, d) n` and
`add_calendar_months_ymd (y, m, d) (12 · n)` are the same function (verified
(read): both return `(y + n, m, min d (days_in_month (y + n) m))`), so
`1 'years + 6 'months` is one step of 18 months and is not mixed.

### 3.5 One lowering function per kind

```ocaml
module Lower : sig
  type instant_value =
    | At        of float               (* a compile-time model time *)
    | At_expr   of Ir.expr             (* an instant-kind parameter, `tau + 7 'days` *)
    | At_anchor of Ir.anchored_time    (* `last_obs ± d`, d exact *)

  val instant : ctx -> instant Time_pos.t -> instant_value   (* emits W324/W325/E321/E360 *)

  type cadence_value =
    | Every    of float                (* a fixed step, in model time units *)
    | Calendar of { months : int }     (* lowered to the IR calendar schedules, §5.2 *)

  val cadence :
    ctx -> start:instant_value -> cadence Time_pos.t -> cadence_value  (* E322 *)

  val runtime_offset : ctx -> runtime_offset Time_pos.t -> float      (* E321 *)
  val span           : ctx -> span Time_pos.t -> Ir.span               (* E321 *)
  val phase          : ctx -> phase Time_pos.t -> float                (* gh#1120 rule *)
  val length         : ctx -> length Time_pos.t -> float               (* never calendar-refuses *)

  (* Intervals (gh#1140): one resolver per interval kind, returning a value
     whose existence proves its order and bounds. *)
  type bins = private { period : float; step : float; n : int }
  val bins : ctx -> period:length Time_pos.t -> step:length Time_pos.t -> bins
  type bin_range = private { first : int; last : int }  (* 0 ≤ first ≤ last < n *)
  val phase_range : ctx -> bins -> phase Time_pos.interval -> bin_range option
  type instant_range = private { start : instant_value; stop : instant_value }
  val instant_range : ctx -> instant Time_pos.interval -> instant_range option
end
```

Each function emits its diagnostics itself and returns a placeholder on refusal
so that a model with several mistakes reports them all, as the expander does
today. A position lowered more than once (the model's `simulate { from }` is
read by the simulate block and by the output schedule's default start) reports
once: the record that test 1.1 reads is also the set of positions already
judged. The checks now in `check_surface_time_typing` move into these functions
and the walk over declaration fields is deleted; the E320 check (a property of
`time_unit`, not of a position) stays where it is.

Two checks in that function are not about positions and stay as an expression
check over every ordinary expression (transition rates, ODE derivatives, `let`
bodies, `init` values, prior arguments, intervention actions, `balance`,
quantity bodies): E321 on `Instant ± calendar duration` wherever it is written,
and E360 on the arithmetic an instant does not have (gh#1070). Every `Lower`
function runs the same two checks on its own expression, except that
`Lower.instant` replaces the generic E321 with its own refusal, so a position
reports one E321, not two.

In Stage 1a each function implements its kind's rule from §8, Stage 1a: today's
value, with every calendar duration in an instant or cadence position refused.
Stage 1b replaces the instant refusal with `resolve_instant` (§3.4); Stage 2
adds the cadence and interval resolvers.

## 4. The rules, by kind, with what they compile to

All examples use `time_unit = 'days` and `origin = date("2020-01-01")` unless
stated. "Today" is `main` at `842d3b66a`, verified (run). "Proposed" is the
value of the explicit equivalent compiled on the same commit, also verified
(run), since the proposal makes the two spellings equal.

### 4.1 Instants

| the modeller writes                                           | today                   | proposed (= the explicit form, compiled)                        |
| ------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------- |
| `simulate { to = 2 'years }`                                  | `t_end = 730.485`       | 731 (`add_calendar_years(origin, 2)`; 2022-01-01)               |
| `simulate { to = 18 'months }`                                | 547.86375               | 547 (2021-07-01)                                                |
| `simulate { to = origin + 2 'years }`                         | E321 + E401 + E602      | 731                                                             |
| `simulate { to = origin + 90 'days }`                         | E401 + E602             | 90 (as an `at [...]` entry compiles today)                      |
| `simulate { from = 6 'months }`                               | `t_start = 182.62125`   | 182 (2020-07-01)                                                |
| `simulate { to = date("2020-03-01") + 7 'days }`              | E401 + E602             | 67 (as an `at [...]` entry compiles today)                      |
| scenario `simulate { to = 30 'months }`                       | 913.10625               | 912 (`add_calendar_months(origin, 30)`)                         |
| `at [6 'months]`                                              | 182.62125               | 182                                                             |
| `at [date("2020-03-01") - 1 'months]`                         | E321                    | 31 (`add_calendar_months(date("2020-03-01"), -1)` = 2020-02-01) |
| output `trajectories { at = [date("2020-03-01") + 7 'days] }` | E401                    | 67                                                              |
| `value_at(I, 6 'months)`                                      | reads t = 182.62125     | 182                                                             |
| `value_at(I, 60)`                                             | no diagnostic           | W325, as `at [60]` (gh#1128)                                    |
| `breakpoints = [6 'months]`                                   | 182.62125               | 182                                                             |
| `tau : instant in [0 'days, 6 'months]`                       | bounds `[0, 182.62125]` | `[0, 182]`                                                      |
| rate `if t >= 6 'months then … else …`                        | switches at 182.62125   | 182                                                             |
| `simulate { to = 1.5 'years }`                                | 547.86375               | 547: 18 whole months (decision A)                               |
| `simulate { to = 600 'months }`, origin 1891-01-01            | 18262.125               | 18262 (1941-01-01)                                              |

The refusals, each with the exact text and the correction:

**Fractional months.**

```
error[E321]: `0.5 'months` in simulate.to is not a whole number of calendar months

  = hint: a month counted from a date is a calendar step, and a calendar step is a whole number of months. Write a whole number of months, or state the offset in days: `15 'days`.
```

The modeller writes `15 'days` (or `1 'months`).

**Months mixed with days.**

```
error[E321]: `date("2020-02-26") + 1 'years + 3 'days` in intervention at-time mixes a calendar step with a number of days, and which comes first changes the date

  = note: from 2020-02-26, a year and then 3 days is 2021-03-01; 3 days and then a year is 2021-02-28
  = hint: write the order you mean: `add_calendar_years(date("2020-02-26"), 1) + 3 'days` takes the year first; for the days first, write the shifted date: `add_calendar_years(date("2020-02-29"), 1)`.
```

The two dates are §1.1's (verified, 425 and 424). The note is computed from the
modeller's own base. Where the two orders happen to agree (from 2020-01-01 both
give 2021-01-04) the refusal still stands, because whether they agree depends on
the base, and a model whose `origin` or `date` is edited must not change meaning
silently. `simulate { to = 1 'years + 3 'days }` (368.2425 today) is refused the
same way, with `origin` as the base.

**Two calendar steps.**

```
error[E321]: `origin + 6 'months + 6 'months` in simulate.to takes two calendar steps, and two steps of 6 months are not one step of 12

  = note: from 2020-01-01 both give 2021-01-01, but from 2020-08-31 two steps give 2021-08-28 and one step gives 2021-08-31
  = hint: write one step, `origin + 12 'months`, or nest the steps you mean: `add_calendar_months(add_calendar_months(origin, 6), 6)`.
```

`add_calendar_months(origin, 6) + 1 'months` is refused the same way, since its
base is already a calendar step. `origin + (6 'months + 6 'months)` is one step
of twelve months and resolves.

**A base known only at run time.**

```
error[E321]: calendar duration `1 'months` cannot move `tau`, an instant parameter whose date is known only when the model runs

  = hint: a calendar month from a date depends on which date it is. State the offset in days: `30 'days` (≈ 1 'months).
```

The same message names `t` or `last_obs` when that is the base. The observation
anchor form already exists (`classify_anchor`, verified (run)); its message is
replaced by this one so that the three bases read alike.

### 4.2 Cadences

A cadence with a compile-time start resolves to a calendar cadence counted from
the start, exactly `date_range`'s calendar semantics (gh#1068, `docs/dates.md`
"Generating a list of dates"): entry `i` is `add_calendar_months(start, i · k)`.

| the modeller writes                                                                                              | today | proposed fire times (verified via the equal `date_range`)                                                                                |
| ---------------------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `b : transfer(...) { every = 1 'months  from = date("2020-01-31")  to = date("2020-06-30") }`                    | E322  | `[30, 59, 90, 120, 151, 181]`: 31 Jan, 29 Feb, 31 Mar, 30 Apr, 31 May, 30 Jun                                                            |
| the hint's suggestion today, `every = 30 'days` with the same `from`, `to`                                       | —     | `[30, 60, 90, 120, 150, 180]` (recurring start 30, period 30, end 181): the 1st of March, the 30th of May                                |
| `output { trajectories { every = 1 'months } }`, `simulate { from = origin to = add_calendar_years(origin, 2) }` | E322  | `[0, 31, 60, 91, …, 700, 731]`, the 1st of each month, 25 rows                                                                           |
| `emit_schedule = every 1 'months`, same horizon                                                                  | E322  | the same 25 times                                                                                                                        |
| `b : add(S, 10) every 1 'years at_day 30 'days`                                                                  | E322  | `add_calendar_years(origin, k) + 30` for each `k` with the target in the run window (the rule; no explicit equivalent exists to compile) |

The second row is the cost of the current hint: an "affine monthly" cadence
drifts off the month within a quarter.

The refusals:

- `every = 0.5 'months` and `every = 1 'months + 1 'days`: E322 with the same
  text as the instant refusals, naming `every`.
- A start that is not the start of a day (`from = 10.5`, or a horizon
  `from = 0.5 'weeks`):

  ```
  error[E322]: a calendar cadence counts whole months from its start, and `from` is t = 10.5 (2020-01-11 12:00), which is not the start of a day

    = hint: start the schedule on a date: `from = date("2020-01-11")`.
  ```

- A calendar `at_day` keeps gh#1120's E322 and its hint unchanged.

Why the cadence is lowered to a calendar schedule in the IR, and not to an
explicit list of times, is §5.2 and decision C.

### 4.3 Offsets from a run-time instant: refused, with the lift stated

`after = 1 'months` on a reactive intervention means "the same day of the next
month after the trigger fires", but the firing time is known only during the
run. Calendar resolution there needs three things the runtime lacks:
`add_calendar_months` in Rust (`rust/crates/ir/src/caltime.rs` has `rata_die`,
`date_from_rata_die` and a private `days_in_month`, verified (read), but no
month stepping); a representation of a calendar offset in the IR (`after` is an
`f64` today); and a rule for an instant that is not the start of a day, since a
trigger evaluated at an observation time in a `'weeks` model can fire at t = 1.5
weeks (10.5 days), and "one month after noon on 11 January" needs a definition.
Stage 2 supplies the first of these. The other two are a proposal of their own.

So runtime offsets stay refused (E321, the existing messages, verified (run)):

```
error[E321]: reactive intervention 'mop': calendar duration `1 'months` as `after`

  = hint: a trigger's `window`, `after` and `cooldown` are each folded to one fixed span, but a calendar month or year is 28-31 or 365-366 days depending on which one it is. State it in 'days or 'weeks: 30 'days (≈ 1 'months).
```

The modeller writes `30 'days`. Arithmetic on `t` in a rate
(`if t - 1 'months > tau`) gets the §4.1 runtime-base message.

### 4.4 Spans: refused, pointing at the calendar `covers` forms

A month-wide `covers` window ending at a row's time is 28 to 31 days depending
on the row, so no single number represents it. That is the planned calendar
`covers` forms (`month(time)`, `iso_week(time)`, `mmwr_week(time)`; D45,
`docs/dev/temporal-architecture.md` §9, "the immediate follow-up"). Until they
exist the refusal stays (verified (run)):

```
error[E321]: observation 'cases': calendar duration `1 'months` as the width of `covers = ending_on(...)`

  = hint: a `covers` width is one fixed span for every row, but a calendar month or year is 28-31 or 365-366 days depending on which one it is. For calendar periods give the file `window_start` and `window_stop` columns, which state each row's own boundaries; for a fixed span write the width in 'days or 'weeks.
```

When D45 lands, the hint gains "or `covers = month(time)`", and that form is the
Span kind's calendar branch (gh#1102's "the helper's calendar branch").
`sum_observed(window = 1 'months)` keeps its E321 and hint (§4.3), since a
trailing calendar window is also a runtime offset.

### 4.5 Intervals: one resolver each (gh#1140, gh#1155, gh#1156)

An interval is a pair the modeller wrote, validated as a pair once, returning a
value whose existence proves the validation. Three resolvers:

**Periodic `on` ranges** (`Lower.phase_range`). With #1157's semantics, an
endpoint is model time, whole units, inclusive. The resolver takes the forcing's
`bins` (period, step, count, built once by `Lower.bins`) and returns
`bin_range = { first; last }` with `0 ≤ first ≤ last < n`, or refuses:

- an endpoint that is not a whole number of model-time units (gh#1155):

  ```
  error[E406]: periodic time function 'school': `on` endpoint `10 'days` is 1.43 weeks, and a range is written in whole weeks (the model's time unit)

    = hint: bins are 1 weeks wide; write the range in weeks, e.g. `1 'weeks : 2 'weeks`, or use `time_unit = 'days` for day-level ranges.
  ```

  This also refuses a month endpoint with daily bins (`1 'months` is 30.44
  days), which gives D70 (a) its full statement: endpoints with units are days.
- reversed (gh#1135), past the period (gh#1139) and off a bin edge (#1157's
  E406): the existing messages, moved into the one function.

The fill loop reads only `bin_range`, so an empty or out-of-period range is
unrepresentable past the resolver.

The runtime must read the bins the compiler built (gh#1156). Decision D sets out
the two options; the recommendation is to carry the bin width in the IR:

```ocaml
(* ir.ml, periodic forcing *)
type periodic_bins =
  | Equal_division                 (* the values form: bin i is [i·P/n, (i+1)·P/n) *)
  | Fixed_width of float           (* the `on` form: bin i is [i·step, (i+1)·step);
                                      the last bin ends at the period *)
```

and `periodic_bin` uses `floor(phase / step)`, clamped to the last bin, under
`Fixed_width`. Under that rule `period = 365 'days, step = 7 'days` has 52 bins
of 7 days and a last bin of 8 days (357 to 365), and `on = [350 : 356]` is days
350 to 357 at run time as it is at compile time.

**Instant pairs** (`Lower.instant_range`): `date_range(start, end)` and a
`bspline` `window = [a, b]`. Each endpoint is lowered by the Instant rule (so
`window = [0 'days, 6 'months]` resolves the calendar step), then the pair is
checked for order where both ends are compile-time values (gh#1118's
`date_range` check and the bspline check become this one). An endpoint on an
observation anchor is ordered at load time, as today.

### 4.6 Lengths and phases

`dt = 1 'months` is 30.436875 (verified (run)), `period = 12 'months` is
365.2425 (verified (run)), and so are a forcing `lag`, a `via` mean and a
`duration` parameter's bounds. These are lengths and keep the average by design
(D65 a): a time step, the length of a seasonal cycle and a mean delay are
lengths, and nothing is stepped along the calendar from them. A sinusoidal
`phase = 2 'months` (60.87375 days, verified (run)) is a phase within a period
of average length and is average for the same reason (decision G).

### 4.7 `time_unit = 'months` under an origin (E320)

E320 is unchanged: an axis whose unit is an average month would drift against
the calendar under repeated conversion. Its hint changes in one clause, because
the unit-literal annotation it recommends now means the calendar date. Today
(verified (run)):

```
… Annotate each with a unit literal (e.g. to = 600 'months) or a date literal (e.g. to = date("1940-12-01")) to preserve intent. …
```

Proposed:

```
… Annotate each with a unit literal (`to = 600 'months` is the date 600 calendar months after the origin) or a date literal (`to = date("1940-12-01")`) to preserve intent. …
```

## 5. What changes in the IR, the goldens and the runtime

### 5.1 Stage 1 changes no IR type

Instants resolve to constants the IR already carries (`t_end`, `AtTimes`,
`breakpoints`, `value_at`'s `ATime`). No golden changes: the only anchored
goldens are `bspline_explicit_dated`, `sia_anchored_dates`,
`sirv_anchored_calendar` (in `ocaml/golden/`) and `sirv_anchored_calendar` (in
`ir/golden/`), and none contains a `'months` or `'years` value (verified, by
grep of `"origin_rata_die": <n>` and then `"months"|"years"` in each). No
`.camdl` file in the repository that declares an origin uses a month or year
literal (verified, grep over all nine). The IR changes value, and so its hash,
for any outside model that does; see decision B.

### 5.2 Stage 2 adds calendar schedules to the IR

Each cadence position has an explicit-list IR form already: `Ir.AtTimes`,
`Ir.ObsAtTimes`, `Ir.OutAtTimes` (verified (read), `ocaml/lib/ir/ir.ml`). A
compile-time list is not a faithful lowering of the last two, and not a safe one
of the first:

- An output `every` and an `emit_schedule = every` follow the run's horizon. A
  regular schedule is enumerated to the run's `t_end` at run time
  (`obs_schedule_times`, `rust/crates/cli/src/main.rs`; the W106 comment in the
  scenario resolver: a regular schedule "enumerates all the way to whatever
  `t_end` the cell has"), and the horizon moves with a scenario's
  `simulate { to }`, with `--to`, and with `to = last_obs + …`, which is unknown
  at compile time. A compile-time list stops at the model horizon, so a scenario
  that extends the run would silently lose every row past it.
- A recurring intervention whose end defaults to the model horizon is refused
  when the horizon is extended (`check_baked_recurring_ends`,
  `rust/crates/cli/src/util.rs`), but that check matches only
  `Recurring { end = old_end }`. An `AtTimes` list would pass it, and the
  campaign would stop at the old horizon with no error.

So Stage 2 adds one calendar form to each of the three schedule types, expanded
by the runtime against the run's horizon:

```ocaml
(* ir.ml *)
type calendar_recurring = {
  cal_start_rd : int;          (* rata die of the start; a whole day by construction *)
  cal_months   : int;          (* > 0 *)
  cal_at_day   : float option; (* the `every … at_day …` phase, in days, added to each step *)
  cal_end      : float;        (* model time; the baked horizon, or NaN when anchored, as Recurring *)
}
type intervention_schedule = ... | CalendarRecurring of calendar_recurring
type observation_schedule  = ... | ObsCalendar of { start_rd : int; months : int; end_ : float }
type output_schedule       = ... | OutCalendar of { start_rd : int; months : int }
```

The runtime converts entry `i` with the model's `origin_rata_die` (already in
the IR, verified (run)) and `days_per_unit`:
`t_i = (rd(add_calendar_months(start, i · months)) − origin_rd) / D(time_unit)`.
`add_calendar_months` joins `caltime.rs` beside `rata_die`, pinned to the OCaml
`add_calendar_months_ymd` by a shared fixture, `ir/golden/calendar_steps.tsv`,
read by both test suites (the pattern of `ir/golden/caltime.tsv`, gh#98). The
Rust consumers are the intervention schedule builder, `obs_schedule_times`,
`sim::output::output_times`, `check_baked_recurring_ends` (which must match
`CalendarRecurring` too), the NaN-horizon rewrite in `obs_anchor.rs`,
`emit_every.rs`, and `runid::ir_hash`.

This is an `ir/VERSION` bump (0.44 to 0.45). It changes the `ir_version` field
of every golden and no golden's schedule, since no golden has a calendar cadence
(§5.1). It is the atomic OCaml, Rust and golden update of CLAUDE.md's "Goldens
and `ir/VERSION`", and it is the bump that gh#1156's `periodic_bins` should ride
(decision D). A new anchored golden fixture covers the three calendar schedules
and the fixed-width periodic forcing.

## 6. What this reverses and amends

**The archived typed-time proposal §1.2**
(`docs/dev/proposals/archive/post-alpha/2026-05-22-typed-time-and-dsl-ergonomics.md`)
ruled that `simulate.from` and `simulate.to` are duration-typed positions and
that `to = 600 'months` is "a one-shot affine conversion". This proposal
reverses that in two steps. Stage 1a makes them instant positions in which a
bare duration denotes `origin + D`, and refuses a calendar duration there
(`to = 600 'months` is E321, whose hint names
`add_calendar_months(origin, 600)`). Stage 1b gives the same spelling the
calendar meaning. The archived reasoning that a single conversion is unambiguous
is right; the conclusion drew the wrong unambiguous reading, the one no modeller
means.

**The named-functions rule.** Under Stage 1a, `.claude/rules/calendar-time.md`,
"Non-affine operations get named functions", stands as written, and is now
enforced at every instant and cadence position. With Stage 1b it becomes:

> ## A calendar step is resolved by type once, and named when composed
>
> A single whole-number calendar step from a date the compiler knows
> (`origin + 6 'months`, `date("2020-01-31") + 1 'months`, `to = 2 'years` in an
> instant position, `every = 1 'months` from a known start) is resolved by the
> position's type to `add_calendar_months` / `add_calendar_years`, because it
> has one meaning (the one-step calendar rule). Composed calendar arithmetic
> (two steps, a step mixed with days) is written with the named functions,
> because there the order of operations changes the answer and the reader needs
> to see it; the compiler refuses the operator form with a hint.

and `.claude/rules/dsl-surface.md` line 83 becomes "prefer explicitly named
functions over polymorphic operators where the semantics differ and the position
cannot settle them (composed calendar steps are written with
`add_calendar_months`; a single step from a known date is resolved by type)".

**E321 and E322** keep their codes. Under Stage 1a, E321 fires on every calendar
duration in an instant position (the seven positions of §1.3, `simulate` `from`
and `to`, scenario `to`, `at` lists, recurring `from` and `to`, output and emit
`at`, `value_at`'s time, `breakpoints`, `instant` bounds, the operand compared
with `t`, `timepoints`), on a calendar offset from model time `t`, on spans and
on runtime offsets; E322 fires on every calendar cadence, as today, with a hint
that names `date_range(..., calendar_months = k)`. Under Stage 1b E321 no longer
fires on a single whole-number step from a known date; it fires on the four
instant refusals of §4.1, on spans and on runtime offsets. Stage 2 moves E322 to
the cadence refusals of §4.2. The `docs/dev/warning-catalog.md` rows are
rewritten at each stage.

**`docs/dates.md`**, Stage 1a: "Exact vs Calendar duration kinds" and "Stepping
a date by calendar months/years" state that under an origin a calendar duration
in a date or schedule position is refused, with the explicit forms; the
`simulate
{ to }` paragraph drops the E401 sentence and its reference to gh#844
(instant arithmetic is accepted there); the known-deviation row for gh#847 is
deleted, since the silent spelling is now refused; option 1 of "Migrating an
unanchored monthly model" writes `to = add_calendar_months(origin, 600)`. Stage
1b adds the section on the one-step calendar rule (§8, Stage 1b).
`docs/camdl-language-spec.md` §2.1 and §14.2 change to match at each stage, by
hand (the spec is never formatted).

**`docs/language-changes.md`** gets an entry at each stage. Stage 1a: each newly
refused spelling, the value it compiled to before, and its replacement; the
instant arithmetic newly accepted in `simulate`, scenario `to` and output `at`.
Stage 1b: what now resolves, and what is still refused.

**The test `test_typed_time_pos_simulate_to_months_oneshot`**
(`ocaml/test/test_compiler.ml`) asserts only that the model compiles. Stage 1a
inverts it: `to = 600 'months` from 1891-01-01 is E321, and the hint names
`add_calendar_months(origin, 600)`. Stage 1b rewrites it to assert
`t_end = 18262.0`, the value of `date("1941-01-01")`.

## 7. Issues this closes

| issue   | stage | how                                                                                                                                              |
| ------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| gh#847  | 1a    | the silent spelling `to = 2 'years` is refused with the explicit form named; 1b then gives it the calendar meaning                               |
| gh#1097 | 1a    | part 2 (one enumeration of positions, derived from the AST): the census and the typed fields. Part 1 (the classifier) stays open                 |
| gh#1102 | 1a    | one lowering function per kind; Span's calendar branch is D45's                                                                                  |
| gh#1128 | 1a    | `Lower.instant` emits W325 for a bare number at `value_at`'s time, as at an `at [...]` entry                                                     |
| gh#1129 | 1a    | `simulate { from, to }`, scenario `to`, output and emit `at` accept instant arithmetic (`date(a) + 7 'days`); also the deferred change of gh#844 |
| gh#1133 | 1b    | every calendar conversion takes an `anchor`, which exists only under an origin                                                                   |
| gh#1140 | 2     | three interval resolvers                                                                                                                         |
| gh#1155 | 2     | the whole-unit endpoint check                                                                                                                    |
| gh#1156 | 2     | `periodic_bins` in the IR (decision D)                                                                                                           |

## 8. Implementation, in stages

Each stage lands on a branch with `make test` green, follows this document
exactly, and cites it with a `Proposal:` footer. The lift is a rough count of
the code each stage touches.

### Stage 1a: typed positions and the census; every calendar instant and cadence refused

The AST types of §3.1 and §3.2, the parser productions that build them, `Lower`
with one function per kind, the census, and the deletion of
`check_surface_time_typing`'s walk over declaration fields and of
`resolve_time_expr`. No IR type changes and every golden stays byte-identical:
no anchored golden contains a month or year literal (§5.1), so every refusal
below falls outside them, and every value `Lower` returns is the value the
deleted fold returned.

**The rule in Stage 1a.** There is no resolution. Under an origin:

| kind           | a calendar duration (`'months`, `'years`, directly or through a `let`) is …                     | code |
| -------------- | ----------------------------------------------------------------------------------------------- | ---- |
| Instant        | refused, always, including a bare duration (`to = 2 'years`, which denotes `origin + 2 'years`) | E321 |
| Cadence        | refused, always (as today)                                                                      | E322 |
| Runtime offset | refused (as today); also a calendar offset from model time `t`, which is new                    | E321 |
| Span           | refused (as today)                                                                              | E321 |
| Phase          | `at_day` refused (as today, E322); periodic `on` endpoints as today; sinusoidal `phase` average | —    |
| Length         | the average (D65 a), never refused on calendar grounds                                          | —    |

Without an origin nothing changes. Instant positions additionally accept the
instant arithmetic that `at [...]` lists accept today (`origin + 90 'days`,
`date("2020-03-01") + 7 'days`) in `simulate { from, to }`, scenario `to` and
output and emit `at` (gh#1129), where today it is E401; and every `at` list
(intervention, event, output, emit) splices `date_range(...)`, which output and
emit `at` refuse today with E270 (verified (run) on `main` at `19c1b64a3`:
`output { trajectories { at = [date_range(origin, count = 3, calendar_months = 1)] } }`
→ `error[E270]`). The second is needed so that the E322 hint below names a form
that compiles at every cadence position.

**Positions newly refused.** Each compiles today with the average month and no
diagnostic (verified (run) on `main` at `19c1b64a3`, origin 2020-01-01; §1.2 and
§1.3 give the values). Stage 1a refuses each with E321:

| position                                | spelling refused                           | compiles to today       |
| --------------------------------------- | ------------------------------------------ | ----------------------- |
| `simulate { to }`                       | `to = 2 'years`                            | `t_end = 730.485`       |
| `simulate { from }`                     | `from = 6 'months`                         | `t_start = 182.62125`   |
| scenario `simulate { to }`              | `to = 30 'months`                          | `t_end = 913.10625`     |
| intervention and event `at [...]`       | `at [6 'months]`                           | fires at 182.62125      |
| output `trajectories { at }`, emit `at` | `at = [6 'months]`                         | `at_times [182.62125]`  |
| `value_at`'s time                       | `value_at(I, 6 'months)`                   | reads t = 182.62125     |
| piecewise and bspline `breakpoints`     | `breakpoints = [6 'months]`                | breakpoint 182.62125    |
| bspline `window`                        | `window = [0 'days, 6 'months]`            | upper knot 182.62125    |
| interpolated `times`                    | `times = [0 'days, 6 'months]`             | sample time 182.62125   |
| bounds of an `instant` parameter        | `tau : instant in [0 'days, 6 'months]`    | upper bound 182.62125   |
| the operand compared with `t`           | `if t >= 6 'months then … else …`          | switches at 182.62125   |
| `timepoints` entries                    | `timepoints { peak = origin + 6 'months }` | accepted, block dropped |
| an offset from `t` (runtime offset)     | `if t - 1 'months > tau then … else …`     | 30.436875 subtracted    |

The bspline `window` and interpolated `times` rows were found while amending
this proposal (verified (run): `window = [0 'days, 6 'months]` lowers to an
upper knot of `182.62125 'months`; `times = [0 'days, 6 'months]` to a sample
time of `182.62125 'months`). Both are lists of instants, so they are Instant
positions; the forcing role table of §3.2 gives `times` the role of a list of
instants and `window` an instant interval.

A recurring schedule's `from` and `to` are instants, so a calendar duration
there moves from E322 with the cadence hint (§10) to E321 with the instant hint.

**The refusals, with the exact text.** The instant refusal from a base known at
compile time (`origin`, implicit for a bare duration, or a `date(...)` literal,
directly or through a `let`):

```
error[E321]: calendar duration `2 'years` in simulate.to: under `origin` this position is a date, and a date moves by calendar months or years only through `add_calendar_months` or `add_calendar_years`

  = hint: for the date 2 calendar years after `origin` write `add_calendar_years(origin, 2)`; for a fixed span write `730 'days` (≈ 2 'years).
```

The first clause of the hint is present when the expression is one calendar step
of a literal count from that base: `add_calendar_years(B, k)` for a whole number
of years, otherwise `add_calendar_months(B, m)` for a whole number of months `m`
(`1.5 'years` names `add_calendar_months(origin, 18)`), with "before" and a
negative count for a subtraction (`date("2020-03-01") - 1 'months` names
`add_calendar_months(date("2020-03-01"), -1)`). Otherwise (fractional months,
months mixed with days, two calendar steps) the hint is

```
= hint: write each calendar step with `add_calendar_months(d, n)` or `add_calendar_years(d, n)` from a date `d`, or state the offset as a fixed span: `15 'days` (≈ 0.5 'months).
```

and the fixed-span clause appears only when the calendar part folds to a whole
number of days of at least one; otherwise it reads "or state the offset as a
fixed span in 'days or 'weeks."

From a base known only at run time (model time `t`, an `instant` parameter, an
observation anchor):

```
error[E321]: calendar duration `1 'months` in a comparison with `t` cannot move `tau`, an instant parameter, whose date is known only when the model runs

  = hint: a calendar month from a date depends on which date it is. State the offset in days: `30 'days` (≈ 1 'months).
```

with `` `t`, the model time `` and `` `last_obs`, an observation anchor `` as
the other two bases, and no "in …" clause for an offset from `t`, whose position
the base already names. This replaces the observation-anchor message of
`classify_anchor`, so the three bases read alike.

The cadence refusal keeps E322 and its message; its hint becomes

```
= hint: a calendar cadence counts whole months from its start, which one fixed step cannot represent. List the calendar dates instead, with `date_range(<start>, <end>, calendar_months = 1)` in an `at` list, or write a fixed cadence: `every = 30 'days` (≈ 1 'months).
```

with `calendar_months` the literal's whole month count where it has one.

E320's hint replaces "Annotate each with a unit literal (e.g. to = 600 'months)
or a date literal (e.g. to = date("1940-12-01")) to preserve intent." by
"Annotate each with a date: `to = add_calendar_months(origin, 600)` or
`to = date("1940-12-01")`; a calendar duration such as `600 'months` in one of
these positions is refused (E321)."

**Bare numbers.** W324 (simulate `from` and `to`) and W325 (`at` entries and a
recurring `from` and `to`) stay where they are, and W325 is added at
`value_at`'s time (gh#1128). No other instant position gains a bare-number
warning in this stage: an `instant` parameter's bounds, a `breakpoints` entry
and a comparison with `t` are routinely bare numbers, and warning there is a
change of its own.

**Expression checks that stay.** E321 on `Instant ± calendar duration` and E360
inside ordinary expressions stay as one expression check (§3.5), unchanged.

Lift: medium to large. The AST fields are read at roughly 120 sites in
`ocaml/lib` and `ocaml/bin` (verified, by `grep -c` per field name: `pbounds`
26, `sim_to` 16, `SchedEvery` 11, `sim_from` 11, `fkind` 13, …), most of them
mechanical (`Time_pos.expr p` where a printer reads the expression).

Tests, each red on `main` where it asserts a change:

1. (1.1) Census coverage: for each fixture model in `ocaml/golden/` and the
   scratch models of these tests that compile, every position the census lists
   is in the record of lowered positions with the same kind.
2. (1.2) Each newly refused spelling of the table above, under an origin, draws
   E321 with the exact message and hint; `simulate { to = 2 'years }` is one of
   them. `test_typed_time_pos_simulate_to_months_oneshot` is inverted:
   `to = 600 'months` from 1891-01-01 is E321 and its hint names
   `add_calendar_months(origin, 600)`.
3. (1.3) `value_at(I, 60)` under an origin draws W325.
4. (1.4) `simulate { to = origin + 90 'days }`, scenario
   `to = date("2020-03-01") + 7 'days` and output
   `at = [date("2020-03-01") + 7 'days]` compile to 90, 67 and 67, and output
   `at = [date_range(origin, count = 3, calendar_months = 1)]` to `[0, 31, 60]`.
5. (1.5) The same spellings without an origin compile to today's average values
   (the rules are vacuous).
6. (1.6) Compile-only: the census's guard is checked by hand: a scratch
   `declaration` constructor, or a field added to a record the census reads,
   fails `dune build` with warning 4 or 9 reported as an error in
   `time_census.ml`. It cannot be a test that runs; the census's header comment
   says how to repeat the check, and the PR states that it was done.

Each test is mutation-checked: the refusal or acceptance it asserts is reverted
in `Lower` and the test is seen to fail.

### Stage 1b: the one-step calendar rule

**The name.** _The one-step calendar rule_: under an origin, one calendar step
of a whole number of months (or years, as twelve months) from a date the
compiler knows is that calendar date, as `add_calendar_months` computes it; any
other use of a calendar duration in a date or schedule position is refused. The
name is used in the language spec, in `docs/dates.md`, in the E321 and E322
messages that refuse what the rule does not cover, and in
`docs/language-changes.md`.

`Calendar_reading.read`, `resolve_instant`, its four refusals and their texts
(§4.1), the CLI refusal of decision F, the docs of §6 and the language-changes
entry.

**Conditions of approval.** Stage 1b lands only after Stage 1a has landed, and
only with:

1. **A section in `docs/dates.md`**, "The one-step calendar rule", stating the
   rule for each position kind of §2 with an example of each: what resolves
   (`to = 2 'years` is 2022-01-01 from 2020-01-01), what is refused and its
   replacement (fractional months, months mixed with days, two steps, a step
   from `t`, a parameter or an anchor), and the kinds that keep the average
   (`dt`, `period`, `lag`, sinusoidal `phase`).
2. **A request-for-feedback box in that section**, of the `.feedback` kind of
   `docs/documentation-style.md` (§5, "Feedback: …"), asking modellers whether
   the rule matches what they mean by a month in each position, and where to say
   so.
3. **The name** above, used consistently in those places.

Lift: small to medium, about 300 lines of OCaml and tests.

Tests, each red on Stage 1a:

1. Every row of the §4.1 table resolves to the value in its "proposed" column,
   asserted against the IR value of the explicit form compiled in the same test.
2. Each refusal of §4.1 emits its code and text; the two-step and mixed notes
   carry the dates computed from the base (`2020-08-31`: 2021-08-28 and
   2021-08-31; `2020-02-26`: 2021-03-01 and 2021-02-28).
3. Unanchored: every row of the §4.1 table that names neither `origin` nor a
   date, compiled without the origin, gives today's average value (the rules are
   vacuous).
4. `test_typed_time_pos_simulate_to_months_oneshot` asserts 18262.0.
5. W327 still fires on the literal nested round trip.
6. Decision F: `--to last_obs+6months` on an anchored fit is refused with the
   DSL's hint.

### Stage 2: calendar cadences and the periodic bin width

`Lower.cadence`, the IR forms of §5.2 and `periodic_bins` of §4.5 in OCaml, Rust
and serde; `caltime::add_calendar_months` and the shared
`ir/golden/calendar_steps.tsv`; the seven Rust consumers; `Lower.phase_range`,
`Lower.instant_range` and the gh#1155 check; `ir/VERSION` 0.45 with the golden
update; the new anchored golden fixture.

Lift: medium. About 250 lines of OCaml, 300 of Rust, the fixture, and the golden
version update (one line per golden).

Tests:

1. `calendar_steps.tsv` rows, read by an OCaml test and a Rust test, including
   31 January monthly, 29 February yearly, negative steps and a start before the
   origin.
2. The §4.2 intervention row fires at `[30, 59, 90, 120, 151, 181]` on every
   backend (the cross-backend lifecycle test, extended).
3. Output and emit `every 1 'months`: 25 rows at the 1st of each month for the
   two-year horizon; with a scenario `to = add_calendar_years(origin, 3)`, 37
   rows (red against a compile-time-list lowering, which gives 25).
4. A recurring calendar intervention with a defaulted end is refused under
   `--to` extension, as a `Recurring` one is.
5. gh#1155: `on = [10 'days : 20 'days]` in a `'weeks` model is E406.
6. gh#1156: `period = 365 'days, step = 7 'days, on = [350 'days : 356 'days]`
   is on at t = 350.5 and t = 356.5 and off at t = 357.5 at run time (today: off
   at 350.5).
7. The cadence refusals of §4.2.

### Stage 3 (not in this proposal): runtime offsets

The IR representation of a calendar offset on a runtime instant and the
fractional-day rule (§4.3). A proposal of its own when wanted; this one commits
only to the refusal.

## 9. Decisions (recorded 2026-11-16)

Each decision below was put to the maintainer with the recommendation that
follows it, and is recorded at its end as **Decided**.

**A. Is `1.5 'years` a calendar step?** The brief that motivated this proposal
refuses fractional counts. A fractional year that is a whole number of months
(`1.5 'years`, `0.25 'years`) is one calendar step of 18 or 3 months, and
`add_calendar_years` is `add_calendar_months` times twelve, so there is one
meaning to resolve it to; `0.5 'months` has none. Resolving it means a modeller
who writes `to = 1.5 'years` gets 2021-07-01 (547) rather than a refusal;
refusing it means writing `18 'months`. The risk of resolving is a reader who
expects 1.5 × 365.2425 = 547.86 days; the refusal costs one rewrite.
Recommendation: resolve when the month count is whole. Leaning.

**Decided: resolve when the month count is whole** (Stage 1b). Under Stage 1a
`to = 1.5 'years` is refused like every calendar instant, and its hint names
`add_calendar_months(origin, 18)`, the form 1b will make it equal to.

**B. An accepted model changes value with no diagnostic.** Stage 1b changes what
already-compiling anchored models mean: `to = 600 'months` moves from 18262.125
to 18262, `at [6 'months]` from 182.62125 to 182. The shift is bounded: over 1
to 1200 months, `n` calendar months from the origin differ from `n` average
months by between −2.8 and +3.0 days for the four origins tried (2020-01-01,
2020-01-31, 1891-01-01, 2026-07-01; computed by a script that re-implements
`add_calendar_months_ymd`, not by the compiler). The repository's rule is that a
silent break is a bug (`.claude/rules/dsl-surface.md`). The options: (i) accept
the change with a `language-changes.md` entry, since the new value is the one
the documentation says these models mean, no model in the tree is affected
(§5.1), and the IR hash changes, so no cached run or fit of an affected model is
reused under the new meaning; (ii) a warning for one release naming the old and
new value at each resolved site, which the repository's preference for errors
over warnings argues against; (iii) refuse the bare calendar duration for one
release with a hint, which is the friction this proposal removes.
Recommendation: (i). Need you, since it is a call about the rule itself.

**Decided: settled by the sequencing, so none of (i) to (iii) is needed.** Stage
1a refuses every old spelling under an origin before Stage 1b gives it the
calendar meaning. A model that compiles under 1a contains no calendar duration
in an instant or cadence position, so 1b changes the value of no model compiled
since 1a. The one path to a silent change is an upgrade that skips every release
carrying 1a and none carrying 1b; Stage 1b's `language-changes.md` entry
therefore repeats each spelling's pre-1a value beside its new one, and 1a and 1b
are not released together.

**C. Lower cadences to a calendar schedule in the IR, not to a list.** §5.2: a
compile-time list stops at the model horizon, so output and emit rows past a
scenario's or `--to`'s longer horizon would be silently absent, and a campaign
would stop at the old horizon past the check that refuses that for `Recurring`.
The list is golden-neutral and needs no Rust; the IR form needs a version bump,
Rust calendar stepping and seven consumers. A middle course, lists for
interventions with an explicit `to` and the IR form elsewhere, leaves two
lowerings of one concept. Recommendation: the IR form for all three. Solid.

**Decided: the IR form for all three** (Stage 2).

**D. gh#1156: carry the bin width, or refuse a period that is not a whole number
of steps.** (a) Add `periodic_bins` to the IR (§4.5): the runtime uses the
compiler's bins, the last bin absorbs the remainder (365.25 days of daily bins
has a last bin of 1.25 days), and the spec's own
`period = 365.25 'days, step = 1 'days` example stays legal. It costs an
`ir/VERSION` bump, which Stage 2 needs anyway, and the periodic goldens change
shape. (b) Refuse `period` not a multiple of `step` with a hint: no IR change,
but it refuses the spec's daily-year example and every 365-day year with weekly
bins, and asks the modeller to choose between a 364- and a 371-day year.
Recommendation: (a), in Stage 2's bump. Solid.

**Decided: (a), in Stage 2's `ir/VERSION` bump.**

**E. Runtime offsets stay refused.** §4.3. Resolving `after = 1 'months` needs
an IR offset form and a rule for an instant that is not the start of a day,
neither of which this proposal settles, and the refusal names the fix.
Recommendation: refuse now; a separate proposal if a model needs it. Solid.

**Decided: refused**, with the messages of §4.3 unchanged in every stage.

**F. The CLI time grammar reads `last_obs + 6 months` as average months.**
`parse_time_spec` (`rust/crates/cli/src/fit/runner.rs`) accepts `month(s)` and
`year(s)` on an anchored `--to` or `holdout_after` and documents them as "fixed
spans (`days_per_unit`), not calendar arithmetic" (verified (read)), while the
DSL refuses `to = last_obs + 6 'months` with E321 (verified (run)). The same
value means different things on the two surfaces. Options: refuse month and year
units on an anchored CLI offset with the DSL's hint (a CLI breaking change), or
leave the CLI as is and document the difference. Recommendation: refuse, in
Stage 1b. Leaning.

**Decided: refuse month and year units on an anchored CLI offset, in Stage 1b**,
with the DSL's hint (Stage 1b test 6).

**G. Forcing `lag` and sinusoidal `phase` keep the average.** Neither is in
D65's list (`dt`, `period`). A lag is a delay on a continuous signal and a phase
is an offset within a period of average length; neither is stepped along the
calendar, so they fall under D65's reasoning. The alternative is to refuse month
units there, as `at_day` does. Recommendation: average, documented in the spec's
unit table. Leaning.

**Decided: average**, in every stage, documented in the spec's unit table.

**H. Comparisons with `t` and instant-parameter bounds are instant positions.**
Found while drafting (§1.3): `if t >= 6 'months` in a rate and
`tau : instant in [0 'days, 6 'months]` fold the average today. Making them
instant positions changes `t >= 6 'months` from 182.62125 to 182 and refuses
`t - 1 'months`. Recommendation: include in Stage 1. Solid.

**Decided: included, in Stage 1a.** Under 1a both are refused (E321):
`t >= 6 'months` with the instant hint, `t - 1 'months` with the run-time base
hint. Stage 1b resolves the first to 182.

## 10. Defects found while drafting, outside this proposal

Each verified (run) on `842d3b66a`. The first two are not fixed here; the last
two are fixed by Stage 1a, because `Lower.instant` judges an anchored `to` once
and returns a placeholder on refusal, and because a recurring `from` and `to`
are instant positions.

- `timepoints { }` is parsed and dropped (`| DTimepoints _ -> ()` in
  `expander.ml`); the spec marks the block "partially implemented". A model
  using it compiles and the block does nothing. Under this proposal its entries
  are instant positions; whether the block should be refused until it is
  implemented is a separate issue.
- `t >= date("2020-07-01")` in a rate is refused with E302, "`182` is a bare
  number here": the date folds to a dimensionless constant before the dimension
  check.
- `simulate { to = last_obs + 6 'months }` draws E321 and then E401, E100
  ("undeclared name 'last_obs'") and E602, three errors that follow from the
  first.
- E322's message on a recurring `from` or `to` ("calendar duration `6 'months`
  in recurring schedule `from = ...`") gives the cadence hint ("Use every = 30
  'days …"), though `from` is an instant. Stage 1a replaces both with the
  instant refusal (E321, the instant hint); Stage 1b resolves a single step.

## Appendix: verification transcript

Scratch models are written by `rust/target/agent-scratch/cal/run.sh`, which
fills a three-compartment SIR model (`time_unit`, `origin = date("2020-01-01")`,
an `EXTRA` block and a `simulate` line from the environment) and compiles it
with `ocaml/_build/default/bin/camdlc.exe`. Selected runs:

```
$ SIM="2 'years" ./run.sh to2y; jq -c '[.model.simulation.t_start,.model.simulation.t_end]' to2y.ir.json
[to2y] exit=0
[0.0,730.485]
$ SIM="add_calendar_years(origin, 2)" ./run.sh to2yc        → [0.0,731.0]
$ SIM="18 'months"                                          → [0.0,547.86375]
$ SIM="add_calendar_months(origin, 18)"                     → [0.0,547.0]
$ SIM="1.5 'years"                                          → [0.0,547.86375]
$ SIM="1 'years + 3 'days"                                  → [0.0,368.2425]
$ SIM="2 'years" FROM="6 'months"                           → [182.62125,730.485]
$ SIM="origin + 90 'days"                                   → error[E401] … error[E602]
$ SIM="origin + 6 'months + 6 'months"                      → error[E321] … simulate.to
# origin 1891-01-01:
$ SIM="600 'months"                                         → t_end 18262.125
$ SIM="add_calendar_months(origin, 600)"                    → t_end 18262.0
$ SIM="date(\"1941-01-01\")"                                → t_end 18262.0

$ EXTRA="scenarios { late { simulate { to = 30 'months } } }"            → t_end 913.10625
$ EXTRA="scenarios { late { simulate { to = add_calendar_months(origin, 30) } } }" → t_end 912.0
$ EXTRA="… at [6 'months, add_calendar_months(origin, 6), date(\"2020-03-01\") + 7 'days]"
  → at_times_expr: unit_literal 182.62125 'months, const 182.0, 60 + 7 'days
$ EXTRA="… at [add_calendar_months(add_calendar_months(date(\"2020-08-31\"), 6), 6),
              add_calendar_months(date(\"2020-08-31\"), 12), add_calendar_months(date(\"2020-08-31\"), 6), …]"
  → at_times [605.0, 608.0, 424.0, …]
$ EXTRA="… at [add_calendar_years(date(\"2020-02-29\"), 1), add_calendar_years(date(\"2020-02-26\"), 1) + 3 'days]"
  → [424.0, 422 + 3 'days]
$ EXTRA="… at [date_range(date(\"2020-01-31\"), date(\"2020-06-30\"), calendar_months = 1)]"
  → at_times [30.0,59.0,90.0,120.0,151.0,181.0]
$ EXTRA="… { every = 30 'days  from = date(\"2020-01-31\")  to = date(\"2020-06-30\") }"
  → recurring {"start":30.0,"period":30.0,"end":181.0}
$ EXTRA="… at [date_range(origin, count = 24, calendar_months = 1)]"
  → [0,31,60,91,121,152,182,213,244,274,305,335,366,397,425,456,486,517,547,578,609,639,670,700,731]
$ EXTRA="… { every = 1 'months … }"                          → error[E322]
$ EXTRA="output { trajectories { every = 1 'months } }"     → error[E322]
$ EXTRA="… at [date(\"2020-03-01\") - 1 'months]"            → error[E321]
$ EXTRA="… at [add_calendar_months(date(\"2020-03-01\"), -1)]" → at_times [31.0]
$ EXTRA="output { trajectories { at = [date(\"2020-03-01\") + 7 'days] } }" → error[E401]
$ EXTRA="output { trajectories { at = [6 'months] } }"      → {"at_times":[182.62125]}
$ EXTRA="quantities { q = value_at(I, 6 'months) }"         → value_at time unit_literal 182.62125
$ EXTRA="quantities { q = value_at(I, origin + 6 'months) }"→ error[E321]
$ EXTRA="quantities { q = value_at(I, 60) }"                → exit=0, no diagnostic
$ EXTRA="interventions { b : … at [60] }"                   → warning[W325]
$ EXTRA="… breakpoints = [6 'months, add_calendar_months(origin, 9)]" → [182.62125 'months, 274.0]
$ EXTRA="timepoints { peak = origin + 6 'months }"          → exit=0
$ (model with tau : instant in [0 'days, 6 'months] and rate `if t >= 6 'months`)
  → tau bounds [0.0,182.62125]; rate compares t with unit_literal 182.62125
$ SIM="add_calendar_years(origin, 2)  dt = 1 'months"       → dt 30.436875
$ (sinusoidal period = 1 'years, phase = 2 'months, lag = 1 'months)
  → period 365.2425, phase 60.87375, lag 30.436875
$ TU="'months" SIM="24"                                     → error[E320] (hint quoted in §4.7)

# PR #1157 branch (e035d97d3), run with its own camdlc:
$ period = 365 'days, step = 7 'days, on = [350 'days : 356 'days]   → 52 bins, on [50]
$ time_unit = 'weeks, period = 364 'days, step = 1 'weeks, on = [10 'days : 20 'days] → 52 bins, on [1, 2]
$ period = 365.25 'days, step = 1 'days, on = [7 'days : 100 'days]  → 365 bins, on 7..100
$ period = 365 'days, step = 1 'days, on = [1 'months : 2 'months]   → 365 bins, on 30..60
```

```
$ grep -lE '"origin_rata_die": *[0-9]' ocaml/golden/*.ir.json ir/golden/*.json \
    | while read f; do echo "$f $(grep -c '"months"\|"years"' $f)"; done
ocaml/golden/bspline_explicit_dated.ir.json 0
ocaml/golden/sia_anchored_dates.ir.json 0
ocaml/golden/sirv_anchored_calendar.ir.json 0
ir/golden/sirv_anchored_calendar.ir.json 0
```
