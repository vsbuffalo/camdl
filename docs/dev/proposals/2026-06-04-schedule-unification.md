# Unifying the scheduling surfaces

Status: **Option A landed** — `schedule_core` (`SchedEvery | SchedAt`) shared by
**output** (A.1, 99806f6) and **observations** (A.2, 546635a), collapsing the
two structurally-identical frontend schedule types into one, byte-identical IR.
**A.3 (interventions/events) deliberately NOT done as a frontend reuse:** their
`schedule_decl` is genuinely richer (windowed `every` with `from`/`to`,
`at_day`, parametric `at`), overlapping `schedule_core` only in the bare `at`
arm; forcing the full core onto them makes a windowless `SchedEvery`
representable (an illegal state), for a one-line grammar saving. Interventions
keep their own type — that's honest, and their unification belongs in Option B's
factored IR design (`source = Specified(schedule) | External`), not a frontend
contortion. Option B (the _factored_ IR-type merge) remains the long-term
design, deferred.

## Problem

"When does this happen?" is asked in four places, each with its own grammar rule
and IR type:

| surface       | IR type                 | shared core                        | surface-specific extension                                |
| ------------- | ----------------------- | ---------------------------------- | --------------------------------------------------------- |
| observations  | `observation_schedule`  | `every = E`, `at = [...]`          | `ObsFromData` (read times from the data file)             |
| output        | `output_schedule`       | `every = E`, `at = [...]`          | `OutMatchObservations` (align to obs times)               |
| interventions | `intervention_schedule` | `at = [...]`, `every = E from..to` | `AtTimesExpr` (parametric `at [t_seed]`, gh#69), `at_day` |
| events        | (recurring/`SAtTimes`)  | `at [...]`, `every … at_day`       | `at_day`                                                  |

Plus two byte-identical records: `regular_obs_schedule` and
`regular_output_schedule` are both `{ start; step; end_ }`.

The vocabulary is _already_ consistent at the token level (`EVERY`/`AT_KW` are
shared), but the parsing rules and IR types are duplicated. The goal: keep the
UX consistent by construction and cut duplication, without introducing bugs in
inference-critical code.

## The catch: a shared core wrapped in divergent extensions

The four surfaces share a genuine common core — `every = E` and `at = [...]`
mean the same thing everywhere — but each has an extension that does **not**
generalize:

- `from_data` is meaningful only for observations (the data file _is_ the
  source); an intervention has no data file.
- `match_observations` is meaningful only for output (align output to obs
  times); for observations it would be circular.
- `at_day` (fire on a day-of-period) and parametric `at [t_seed]`
  (`AtTimesExpr`, gh#69, resolved against the param vector at sim start) are
  intervention/event concepts.

So a single unified type is not "the same thing three times." It is `{ every, at
} + (from_data | match_observations | at_day | parametric)

- a validity matrix` stating which extension is legal in which context. That
  matrix is itself complexity and a bug surface; it may not be a net
  simplification over three small, honest types.

And the _semantics_ around the times differ, not just the times: observations
are **sampled**, output is **snapshotted**, interventions **fire
deterministically mid-substep and perturb propensities** — and the paired-seed
CRN coupling depends on RNG-consumption order at intervention times. The
schedule is a thin shared notion sitting on top of three different evaluation
models.

**The types differ because the domain objects differ — and that is the design
principle, not an accident.** There are two genuinely distinct objects here:

- A **sampling schedule** (observations, output): "at which times do I _read_
  the state — observe or snapshot?" Its natural shape is a bare cadence
  (`every = E` over the run window) or an explicit list (`at = [...]`). That is
  exactly `schedule_core`, and it is why obs + output collapse cleanly (A).
- A **firing schedule** (interventions, events): "at which times does this
  recurring _action_ fire and mutate state?" Its natural shape carries a
  **window** (`from`/`to` — a campaign recurs only within a period), a **phase**
  (`at_day` — fire on day 90 of each year), and **parametric** times (gh#69 — a
  fire time you are inferring). None of that is meaningful for a read.

Forcing them into one type is bad consolidation: it either drops the
window/phase (information loss) or makes obs/output carry always-`None`
window/phase fields and a windowless `SchedEvery` they don't support. Good
consolidation joins same-concept things (obs+output) and keeps distinct-concept
things distinct. The lone genuine overlap is the explicit-times arm
(`at = [...]`), which is one grammar line — not worth coupling two domain
objects over.

(Note the inference _join_ this enables: under `--data`, each observation block
binds to a data stream by **name**, and the data's **time column** supplies the
observation times — the declared sampling schedule is used only for forward
synthetic-data generation, not consulted when fitting. The schedule is a
forward-direction notion; real data supplies the times in the inverse
direction.)

## Option A — unify the frontend surface (recommended)

One shared frontend type + grammar rule for the common `every`/`at`, lowered
per-surface in the expander to each construct's _existing_ IR variant. Today the
observation and output schedule ASTs are literally the same two-constructor type
under different names:

```ocaml
(* before — duplicated *)
type obs_schedule         = ObsEvery of expr | ObsTimes of expr list
type output_schedule_spec = OtEvery  of expr | OtAt    of expr list

(* after — one shared type *)
type schedule_core =
  | SchedEvery of expr        (* every = E      *)
  | SchedAt    of expr list   (* at = [t1, ...] *)
```

Parsed once:

```
schedule_core:
  | EVERY EQ e = expr                                          { SchedEvery e }
  | AT_KW EQ LBRACKET ts = separated_list(COMMA, expr) RBRACKET { SchedAt ts }
```

Each construct reuses it and keeps its own extension rule; the expander lowers
`schedule_core` to the existing IR variant:

- **observations** use the full core (`SchedEvery`→`ObsRegular`,
  `SchedAt`→`ObsAtTimes`). (`ObsFromData` has no frontend producer today — it is
  an IR-only stub, not a live obs `from_data` rule — so the obs surface is
  exactly the two-arm core.)

  > **Superseded (gh#143):** the obs/output lowering **divergence** described
  > below was a bug — output lowered `start = min 0.0 t_start`, which clamped
  > `t_start > 0` to 0 and emitted the trajectory over `[0, t_end]` instead of
  > `[from, to]`. Output now lowers `start = t_start`, so **both surfaces
  > agree** in all regimes, and a shared `lower_schedule_core` helper would be
  > safe (indeed correct) for a future Option B. The historical reasoning is
  > kept below as the decision record; do not act on the "keep the sites
  > distinct" instruction.

  **Critical — the obs and output lowerings are NOT identical:** obs `every`
  lowers `start = t_start` (expander.ml:3922); output lowers
  `start = min 0.0 t_start` (expander.ml:3217). They agree at `t_start = 0` (the
  common case) and for negative `t_start` (where `min(0, t_start) = t_start`),
  but **diverge for `t_start > 0`** — e.g. `simulate { from = 10 }` or an
  anchored `from` after `origin`: obs.start = 10 while output.start = 0. Share
  only the grammar rule + AST type; keep the two expander lowering sites
  distinct with their existing `start` expressions. Do **not** factor a shared
  `lower_schedule_core` helper — that would silently shift observation times
  (which PGAS conditions on).
- **output** use the full core (`→ OutRegular` / `OutAtTimes`); `format` stays
  its own field.
- **interventions / events** reuse only the `SchedAt` arm (explicit times).
  Their `every` is _windowed_ (`every = E from F to T`) or day-of-period
  (`at_day`) — richer than the bare core, so it stays a separate variant. Do
  **not** let interventions carry the whole `schedule_core`, or `SchedEvery` (a
  windowless cadence they don't support) becomes a representable illegal state —
  the very smell Option B's naive form has.

Why no Rust/schema change: the expander emits the same IR variants it does
today, so the serialized IR is byte-identical. The Rust enums
(`ObservationSchedule` / `OutputSchedule` / `InterventionSchedule`) and
`schema.json` are untouched — their triplication is what _Option B_ would
collapse, not A.

Migration order, asserting byte-identical golden IR at each step: output
(already `OtEvery`/`OtAt`) → observations (delete `obs_schedule`) → the `at` arm
of interventions/events. Behavior-preserving; gated by the existing golden +
integration suite — **plus, for the obs step, a NEW test** with `t_start > 0`
(e.g. `simulate { from = 10 }`) carrying an obs `every`, pinning
`obs.start = t_start` while `output.start = 0`. Existing goldens use `from = 0`
(where the two lowerings agree), so the suite would not otherwise catch a
regressed shared lowering (the trap above).

Wins: `every`/`at` parse identically by construction (a fifth surface can't
drift), and two identical AST types plus their parsing collapse to one. Cost: a
frontend refactor, no contract exposure.

The adjacent `regular_obs_schedule` / `regular_output_schedule` record merge is
_not_ part of A — it is an IR-type change, so it rides with B.

## Option B — merge the IR types (defer; the _factored_ design, not a fat type)

The naive merge — one `Schedule` type carrying every variant with per-context
validation rejecting the illegal ones — is a worse design: it widens the type so
`output { from_data }` is representable, then leans on a runtime validity
matrix, violating "make illegal states unrepresentable." Don't build that.

The _right_ merge factors two axes the current types conflate — **specified
times** (`every`/`at`/parametric, genuinely shared) vs **derived source**
(`from_data`, `match_observations`, which are not schedules at all; note both —
plus the intervention-only `External` — are currently IR-only stubs with no
frontend producer, so a clean B must first decide whether they are roadmap or
delete-on-sight):

```ocaml
type schedule =
  | Every of { step; from?; until?; at_day? }   (* windowing/at_day live HERE *)
  | At    of expr list
  | AtExpr of expr list                          (* parametric; intervention-only today *)

(* observations *)  source = Specified of schedule | FromData
(* output *)        source = Specified of schedule | MatchObservations
(* interventions *) source = Specified of schedule | External   (* 4th variant, also a stub *)
```

This factors the **derived-source** axis cleanly — that part is real and good.
But it does **not** eliminate the validity matrix, only relocate it:
`at_day`/windowing are fields of `Every` that are always-`None` for obs/output,
and `AtExpr` (parametric `at`, gh#69) is intervention-only today, so a shared
`schedule` makes it _representable_ where it is currently unrepresentable. The
within-`schedule` per-surface legality (which of `at_day` / windowing / `AtExpr`
is valid where) is an open decision to settle **before** building B — it
determines whether `schedule` is genuinely shared or carries per-surface phantom
fields. (`at_day` is also not a peer variant: in the IR it is an `Option` field
of the recurring case — `RecurringSchedule
{ start; period; end; at_day }` — so
the proposal's earlier "peer extension" framing was wrong.)

The boundary that keeps it honest: **share the _times_, never the
_evaluation_.** The schedule answers "when"; the surfaces differ in "what
happens then" — sampled vs snapshotted vs fired-with-propensity-effects, where
the paired-seed CRN coupling and PGAS conditioning live. This boundary holds
**structurally, not by discipline**: PGAS reads obs times as a plain `Vec<f64>`
from the _data_ path (zero `ObservationSchedule` references in
`crates/sim/src/inference/`), and `output_times` / `intervention_fire_times` are
pure functions of the payload — so a faithful merge cannot perturb CRN ordering.
The load-bearing B equivalence tests are therefore `output_times(old) == new`
and `intervention_fire_times(old) == new` over all goldens, plus golden-IR
byte-identity — not anything inside `pgas.rs`.

Why defer it anyway — none of these is "it is a bad abstraction":

- **Execution risk.** A cross-language IR schema change (OCaml types + the three
  Rust enums + `schema.json` + bump + full golden regen) reaching
  `intervention.rs` / `pgas.rs` / `particle_filter.rs` — surfaces CLAUDE.md
  flags high-risk regardless of how mechanical the edit looks.
- **The factoring needs care.** The naive fat-type version is a real trap;
  separating the two axes is design work best done deliberately, not under
  refactor momentum.
- **No forcing function.** No incident on record is _caused_ by the current
  duplication; nothing breaks if we wait. Electing this risk now buys long-term
  tidiness, not a fixed bug.

**Sizing (from the Rust consumer survey).** The type change is small; the lift
is breadth. Real _logic_ consumers are few — `output_times` (`output.rs`),
intervention fire-times (`intervention.rs`), `AtTimesExpr` resolution
(`compiled_model.rs`), `resolve_output` (`resolve.rs`), and the obs
schedule→times path — a couple inference-adjacent (fire-times feed PGAS event
density; obs times set what PGAS conditions on). The bulk is mechanical: dozens
of inline construction sites (`OutputConfig { times: …
}` / `schedule: …`,
mostly in tests) re-shape under the factored `source` wrapper. The sharp edge is
**run identity**: `runid/src/ir_hash.rs` hand-hashes each schedule type into the
`run_id` via a `ContentAddressed` impl that emits the **fully-qualified
type-path string** (e.g. `"ir::observation::ObservationSchedule"`) plus
**positional `u32` variant indices**. Renaming the three types into a shared
`Schedule` + `source` necessarily changes those — so byte-identical `run_id` is
**not** the default; achieving it would mean hand-forging the legacy tag
strings + flat indices in the new impls (ugly, and a perpetual trap for the next
`ir_hash` editor who "tidies" a tag to match the new type name). The honest plan
is the opposite: **accept a one-time re-key** — bump the schedule types' `SV`
and the committed `GOLDEN` pin in `runid/src/ir_hash/tests.rs`, with a test
pinning the NEW hash. The churn is the test pin + any operator's live store,
**not** golden files (CAS dirs `results/` / `output/` are gitignored).

And it is a full **"Changing the IR schema"** procedure, not a Rust-side
refactor: it also restructures the six hand-written OCaml serde
`to_json`/`of_json` pairs (`ocaml/lib/ir/serde.ml`), with byte-identity of the
emitted JSON through OCaml as its own obligation, and regenerates all golden IR
atomically (schema.json + VERSION + OCaml serde + Rust types + Rust ir_hash +
goldens).

If taken, B should also settle: do _all_ surfaces gain parametric `at` (gh#69)
and `at_day`, or does the type carry per-context legality? Its own review, not a
default.

## Recommendation

1. Land **Option A** (frontend `schedule_core`) per the plan above — low risk,
   byte-identical IR, no Rust/schema/inference exposure, most of the UX win. It
   is a strict stepping-stone toward B: the shared frontend core is what B later
   promotes into the IR.
2. Hold **Option B** (the _factored_ design, not the fat type) until a concrete
   problem justifies it — a real inconsistency bug, or a feature that needs one
   IR type. Then do it as its own schema change with byte-equivalence golden
   coverage, settling the parametric-`at` / `at_day` question first.
3. Treat "reduce bugs" as the test, not the slogan: the change that most reduces
   bug risk _right now_ is the one that doesn't touch the live, tested,
   inference-critical IR schedule types.
