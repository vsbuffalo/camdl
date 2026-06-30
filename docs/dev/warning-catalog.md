# Diagnostic catalog

Central index of every diagnostic the camdl compiler can emit, plus its
severity, category, and rationale. Severities are `Error | Warning | Info` per
`ocaml/lib/compiler/diagnostics.ml`.

When you add a new emit site (`Diagnostics.error`, `.warning`, `.info`, or
future `.lint`), add a one-line entry here. Reviewers should reject any
diagnostic emit-site that isn't in the catalog.

## Code namespaces

- **`E0xx` — meta / internal** (compiler bug-class; should be rare)
- **`E1xx` — parse / lex** (file-level syntax issues)
- **`E2xx` — semantic / scoping** (resolution, redeclarations, missing names)
- **`E3xx` — dimensional analysis** (rate vs flux, P/T mismatch)
- **`E4xx` — schedule / forcing / intervention** (wrong-shape recurring blocks,
  range parse errors)
- **`E6xx` — simulation config** (rejected before runtime)
- **`W1xx` — model-file warnings** (questionable but valid declarations)
- **`W2xx` — IR / compiler warnings** (suspicious but legal expressions)
- **`W3xx` — covariate / forcing warnings** (alignment, interpolation)
- **`I3xx` — dimensional-analysis info** (undetermined dimensions, etc.)
- **`L4xx` — lints** (semantically valid but discouraged patterns)

## Errors

(Errors block compilation. The list below is the current state; specifics are
documented at each emit site in `ocaml/lib/compiler/`.)

| Code      | Category       | Summary                                                                                                                                                                                                                                                                                                                           |
| --------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E001      | meta           | internal compiler error / unreachable                                                                                                                                                                                                                                                                                             |
| E100      | parse          | undeclared name                                                                                                                                                                                                                                                                                                                   |
| E101      | parse          | duplicate compartment                                                                                                                                                                                                                                                                                                             |
| E102      | parse          | duplicate parameter                                                                                                                                                                                                                                                                                                               |
| E103      | parse          | duplicate let binding                                                                                                                                                                                                                                                                                                             |
| E104      | parse          | reserved name used as identifier                                                                                                                                                                                                                                                                                                  |
| E105      | parse          | unknown unit suffix                                                                                                                                                                                                                                                                                                               |
| E106      | parse          | unknown / conflicting key in a config block (`output {}` section, `simulate {}` key, schedule `every`/`at` conflict)                                                                                                                                                                                                              |
| E107      | parse          | ambiguous unit literal after `/`                                                                                                                                                                                                                                                                                                  |
| E108      | parse          | malformed initial-condition expression                                                                                                                                                                                                                                                                                            |
| E109      | parse          | unknown forcing function shape                                                                                                                                                                                                                                                                                                    |
| E110      | parse          | unknown transition attribute `#[...]` (only `#[lineage]` is supported)                                                                                                                                                                                                                                                            |
| E111      | parse          | unknown `#'` doc tag (only `@symbol` / `@ref` are recognized; values/bounds/priors belong in the --params TOML, not the model)                                                                                                                                                                                                    |
| E200–E221 | semantic       | scoping / declaration / resolution errors (multiple variants); E221 = read() data-file header has too few columns for the table's index dimensions                                                                                                                                                                                |
| E222      | semantic       | table uses `read(...)` but declares no index dimensions                                                                                                                                                                                                                                                                           |
| E223      | semantic       | `date(...)` / `origin` ISO date is malformed or out of range (month 1–12, day valid for the month, leap-aware); mirrors Rust `caltime::parse_iso_date`                                                                                                                                                                            |
| E230–E276 | semantic       | observation, balance, simulation-block validation                                                                                                                                                                                                                                                                                 |
| E277      | semantic       | initial condition does not name an expanded compartment cell (bare stratified, or unknown compartment)                                                                                                                                                                                                                            |
| E278      | semantic       | declaration name is duplicated within a namespace or ambiguous across namespaces (compartments / parameters / lets / forcings / tables), including after stratification expansion                                                                                                                                                 |
| E279      | semantic       | observation aux data column collides with a model name (compartment / parameter / let / forcing of the same name); the likelihood reference would be ambiguous (2026-06-10 obs data-entry §3.1)                                                                                                                                   |
| E280      | semantic       | un-indexed observation projection on a stratified model would silently sum across strata; state the aggregation explicitly (`rho * sum(p in dim, incidence(tr[p]))`) (2026-06-10 obs data-entry §5.2)                                                                                                                             |
| E281      | semantic       | tier-3 unit literal on a parameter kind that already fixes its dimension; only `positive`/`real` accept a unit literal (gh#60)                                                                                                                                                                                                    |
| E282      | semantic       | parameter declared with both a tier-3 unit literal and a `[dim]` bracket annotation; use one or the other (gh#60)                                                                                                                                                                                                                 |
| E283      | semantic       | a `sum` variable shadows an enclosing index/bound variable, which first-match-wins resolution would silently rebind; rename the sum variable                                                                                                                                                                                      |
| E284      | semantic       | a restricted-sum `where` predicate is not compile-time decidable (non-constant table, parameterized cell, or a parameter/fitted threshold); the support must be constant                                                                                                                                                          |
| E285      | semantic       | `truncated_normal` prior on a parameter with no `in [lo, hi]` bounds — there is nothing to truncate to; the truncation support is the declared range (gh#155)                                                                                                                                                                     |
| E286      | semantic       | `log_uniform` / `truncated_normal` prior used hierarchically (param-reference argument or `\| dim` pooling clause); these are constant-only distributions and cannot be pooled (gh#155)                                                                                                                                           |
| E287      | semantic       | partial dimension omission in a rate read: a compartment stratified over 2+ dimensions is indexed with some but not all dimensions (`E[a]` when `E` has `[age, latent_stage]`) — a partial index has no defined cell; index all dims or marginalize explicitly with `sum(s in dim, X[a, s])` (bare name still sums over all dims) |
| E288      | semantic       | a forbidden leaf (`dt`) in a quantity body — a quantity is read at output cadence where the integrator step has no value (2026-06-25 generated quantities)                                                                                                                                                                        |
| E289      | semantic       | malformed quantity body — wrong reduction arity, `total`/`sum` (deferred to the flow source), a forward / series / cross-stratum `QRef`, a compartment-vs-scalar mix, or a name colliding with a compartment/param/let/observation/earlier quantity (2026-06-25 generated quantities)                                             |
| E290      | semantic       | a temporal reduction (`final`/`mean`/`integral`/`time_of_max`/`first_above`/…) used outside a `quantities { }` block — it folds a whole trajectory and has no meaning in a rate/binding (2026-06-25 generated quantities)                                                                                                         |
| E291      | semantic       | a `scenarios { }` preset named `fitted` — reserved for the no-overlay row (the fitted model, no scenario applied) in the `scenario` column of `camdl fit predict` output; rename the scenario (2026-06-27 scenario-aware fit predict)                                                                                             |
| E292      | semantic       | a run-rooted reference (`<run>.quantities.<q>` / `<run>.observations.<stream>`) used outside a `contrasts { }` block — it is a contrast operand, with no value in a rate/binding/per-instant expression (2026-06-25 counterfactual contrasts)                                                                                     |
| E293      | semantic       | a run-rooted reference written inside a `quantities { }` recipe — in a recipe the run is implicit, so drop the `<run>.` prefix; run-prefixed operands belong in `contrasts { }` (2026-06-25 counterfactual contrasts)                                                                                                             |
| E294      | semantic       | a `contrasts { }` operand names an undeclared run (not a `scenarios { }` preset nor reserved `fitted`) or an undeclared quantity / observation stream (2026-06-25 counterfactual contrasts)                                                                                                                                       |
| E295      | semantic       | a `contrasts { }` body is not run-rooted arithmetic — a bare const, comparison, inline reducer, or other form appeared where `<run>.<ns>.<member>` operands combined with `+ - * /` are required (2026-06-25 counterfactual contrasts)                                                                                            |
| E296      | semantic       | a `contrasts { }` window endpoint is a bare duration, not an instant — use `origin + N 'weeks` or `date(...)` (the `from` instant is the counterfactual fork point) (2026-06-25 counterfactual contrasts)                                                                                                                         |
| E297      | dimensional    | the two arms of a contrast difference have incompatible dimensions (e.g. `deaths - rate`) — both operands must report the same quantity/dimension (2026-06-25 counterfactual contrasts)                                                                                                                                           |
| E300      | dimensional    | transition rate has wrong dimension (e.g. per-capita where total propensity expected)                                                                                                                                                                                                                                             |
| E301      | dimensional    | exponent has non-dimensionless dimension                                                                                                                                                                                                                                                                                          |
| E302      | dimensional    | dimension mismatch (e.g. adding a count and a rate)                                                                                                                                                                                                                                                                               |
| E303      | dimensional    | parameter used with conflicting dimensions across transitions                                                                                                                                                                                                                                                                     |
| E304      | dimensional    | `sqrt` requires even dimension exponents / distribution parameter has wrong dimension (e.g. binomial `p` is a count)                                                                                                                                                                                                              |
| E305      | dimensional    | balance expression has wrong dimension                                                                                                                                                                                                                                                                                            |
| E306      | dimensional    | ODE derivative has wrong dimension                                                                                                                                                                                                                                                                                                |
| E307      | dimensional    | observation dispersion parameter must be dimensionless                                                                                                                                                                                                                                                                            |
| E308      | dimensional    | overdispersion `sigma^2` must be dimensionless                                                                                                                                                                                                                                                                                    |
| E310      | dimensional    | misc dimensional mismatch                                                                                                                                                                                                                                                                                                         |
| E320      | calendar       | integer `time_unit` cannot be combined with `origin = date("...")`                                                                                                                                                                                                                                                                |
| E321      | calendar       | calendar duration cannot translate an instant in the model's time unit                                                                                                                                                                                                                                                            |
| E322      | calendar       | calendar duration used in a recurring schedule field                                                                                                                                                                                                                                                                              |
| E323      | calendar       | periodic forcing has bare-numeric entries in `on=[...]` under a calendar origin                                                                                                                                                                                                                                                   |
| E327      | calendar       | `date_range` with `start = origin` requires an anchored model                                                                                                                                                                                                                                                                     |
| E328      | calendar       | `date_range` missing required `start` argument                                                                                                                                                                                                                                                                                    |
| E329      | calendar       | `date_range` `count`/`every` out of range (must be ≥ 1 / positive)                                                                                                                                                                                                                                                                |
| E401      | schedule       | recurring block missing required field                                                                                                                                                                                                                                                                                            |
| E402–E408 | schedule       | recurring/periodic block validation (period, on-list, alignment)                                                                                                                                                                                                                                                                  |
| E500      | validate       | duplicate compartment after expansion                                                                                                                                                                                                                                                                                             |
| E501      | validate       | duplicate transition after expansion                                                                                                                                                                                                                                                                                              |
| E502      | validate       | duplicate parameter                                                                                                                                                                                                                                                                                                               |
| E503      | validate       | unknown compartment referenced                                                                                                                                                                                                                                                                                                    |
| E504      | validate       | unknown parameter referenced                                                                                                                                                                                                                                                                                                      |
| E505      | validate       | unknown table referenced                                                                                                                                                                                                                                                                                                          |
| E506      | validate       | unknown time_function referenced                                                                                                                                                                                                                                                                                                  |
| E507      | validate       | unknown transition referenced in observation                                                                                                                                                                                                                                                                                      |
| E508      | validate       | real-valued compartment in transition stoichiometry                                                                                                                                                                                                                                                                               |
| E509      | validate       | real-valued compartment has no ODE equation                                                                                                                                                                                                                                                                                       |
| E510      | validate       | ODE equation for a non-real compartment                                                                                                                                                                                                                                                                                           |
| E511      | validate       | transition has zero delta for a compartment                                                                                                                                                                                                                                                                                       |
| E512      | validate       | hoisted binding references a parameter (gradient would be silently zeroed)                                                                                                                                                                                                                                                        |
| E513      | validate       | initial condition names a compartment absent from the expanded model (contract-boundary net; frontend reports the located E277)                                                                                                                                                                                                   |
| E600      | runtime config | rejected before backend dispatch                                                                                                                                                                                                                                                                                                  |
| E601      | semantic       | lineage tracking requires linear dependence on parent compartments                                                                                                                                                                                                                                                                |

## Warnings

| Code | Severity | Category   | Summary                                                                                       |
| ---- | -------- | ---------- | --------------------------------------------------------------------------------------------- |
| W100 | Warning  | model-file | inconsistent digit grouping in a numeric literal (drained from the lexer)                     |
| W103 | Warning  | model-file | questionable model-file construct                                                             |
| W104 | Warning  | model-file | `read(...)` uses an absolute path — non-portable model (gh#211)                               |
| W105 | Warning  | model-file | per-(p,q) coupling antipattern (O(P²) transitions); use a summed rate `sum(q in dim where …)` |
| W200 | Warning  | IR         | suspicious IR shape                                                                           |
| W201 | Warning  | IR         | suspicious IR shape                                                                           |
| W301 | Warning  | covariate  | periodic range not aligned to step size                                                       |
| W310 | Warning  | covariate  | covariate / interpolation issue                                                               |
| W311 | Warning  | covariate  | covariate / interpolation issue                                                               |
| W324 | Warning  | calendar   | bare number in `simulate.from`/`.to` with a calendar origin declared                          |
| W325 | Warning  | calendar   | bare number in a recurring/at time position with a calendar origin declared                   |
| W327 | Warning  | calendar   | calendar `add_*`/`subtract_*` round-trip is not in general the identity (month-end clamping)  |
| W328 | Warning  | calendar   | `date_range` `end` does not land on a cadence boundary                                        |

(Each row should eventually be expanded with a one-paragraph rationale
documenting the failure mode the warning catches. Future emit-site additions
must update this table in the same commit — the catalog-consistency meta-test in
`ocaml/test/test_diagnostics.ml` fails the build if an emit-site code is missing
here.)

**Rust-side (runtime/fit) warnings are documented in prose, not as table rows.**
The catalog-consistency meta-test scans only the OCaml compiler sources
(`ocaml/lib`) for emit sites and requires every _single-code table row_ to have
a matching OCaml emit. A `[warn Wxxx]` `eprintln!` in the Rust CLI (e.g.
**W326**, **W329**) has no OCaml emit site, so it gets a prose `### Wxxx`
section below instead of a first-cell table row — otherwise the meta-test flags
it as a stale catalog row. Do not "fix" the apparent table omission by adding a
row.

### W104 — absolute path in `read(...)` (non-portable model)

**Fires when:** a `read("...")` data-load (table or any other `read()`-loaded
file) is given an _absolute_ path — one for which `Filename.is_relative` is
false (e.g. `read("/home/alice/data/contact.tsv")`). A relative path
(`read("data/contact.tsv")`, even a `../`-escaping `read("../shared/x.tsv")`)
does NOT fire — those resolve against the `.camdl` source directory and travel
with the model.

**Why:** an absolute path bakes one machine's filesystem layout into the model.
It compiles fine on the author's machine and is silently non-portable — it
breaks for anyone else, breaks model-repo sharing, and breaks the `camdl mre`
bundle (which has to detect and rewrite such paths). For software whose outputs
inform public-health decisions, a model that only runs on one filesystem is a
latent reproducibility bug. The diagnostic is a _warning_, not an error: an
absolute path still works locally, so a hard error would block legitimate
exploratory work. `-Werror` / `--deny` (gh#56) promote it to a hard failure for
CI; per-site suppression (gh#55) silences the rare deliberate case.

**Where it fires:** at expander time, not at IR-lint time. By IR time the
`read()` path is gone (inline tables become `{values}`, external becomes
`{external: name}` — no path field survives serialization), so an `ir/lint.ml`
pass cannot see it. The check has to live where the path string still exists:
the single `read()` chokepoint `read_csv_rows` in
`ocaml/lib/compiler/expander.ml`, beside the existing E200 (file-not-found)
raise. The warning is checked on the path STRING _before_ the file-existence
check, so it fires whether or not the absolute file happens to exist on the
compiling machine — non-portability is a property of the path, not of local
presence. (Consequently a missing absolute file emits both W104 and E200.)

**Scope:** absolute paths are the clear win and the only thing W104 flags. A
`../`-escaping _relative_ path is a legitimate multi-model-repo pattern and is
deliberately left un-flagged. The portability check uses the OCaml stdlib's
`Filename.is_relative` (the platform-portable predicate), so a Windows-style
drive-rooted path (`C:\...`) is also classed as non-relative on a Windows host;
a leading-`~` path is _not_ expanded by camdl and is treated as a relative path
beginning with the literal `~` directory (it is not flagged — the file simply
won't resolve and E200 fires).

**Fix / silence:** rewrite the path relative to the `.camdl` source file
(`read("data/contact.tsv")`), so the model runs on any machine. Pack-time
counterpart: `docs/dev/proposals/2026-06-09-mre-bundle.md` surfaces the same
smell at bundle time; W104 is the upstream fix that helps every author.

### W324 / W325 — bare numeric in an absolute-time position under a date origin

**Fires when:** the model declares `origin = date(...)` (anchored) and a _bare
numeric_ literal appears in an absolute-time position: `simulate.from` /
`simulate.to` (**W324**), or an `at = [...]` schedule entry / a recurring
`from`/`until` bound on an `interventions {}` or `events {}` entry (**W325**). A
unit-annotated literal (`730 'days`) or a `date(...)` literal does NOT fire —
those state their intent. `simulate.dt` is a step _length_ (a duration), so a
bare numeric `dt` is correct and never warns.

**Why:** under a date origin, `from = 730` silently means "730 internal-time
units (here days) after origin" — the reader has to compute the calendar date in
their head, and an off-by-a-window mistake (e.g. `from = 0` against a data
window that begins years after origin) is invisible. `from = date("1952-01-01")`
reads in calendar terms and makes the offset auditable. This is the model-side
mirror of the data loader's **W326** (numeric `--data` time column under a date
origin): gh#134 closed the asymmetry so the calendar-vs-raw choice is surfaced
the same way on both the model surface and the data surface.

**Fix / silence:** write `date("YYYY-MM-DD")` for a calendar instant, or — if
the offset really is an intentional internal-time count — annotate it with a
unit literal (`<n> 'days`) to state that and suppress the warning. The emit site
and hint text live in `ocaml/lib/compiler/expander.ml` (`warn_bare_numeric`) and
`ocaml/lib/compiler/time_typing.ml` (`hint_bare_numeric_simulate` /
`hint_bare_numeric_at_schedule`).

**Related — gh#134 first-interval sanity (shipped as W329):** a separate
first-interval guard — flag when the leading gap (`simulate.from` → first bound
data time) is `≫` the modal observation spacing — lives on the Rust fit side,
not here. It is a soft warn for prevalence and a hard error for incidence (§6.8
of the burn-in / conditioning proposal). See **W329** below.

### W329 — oversized first observation interval (`simulate.from` far behind the data)

**Fires when:** a `fit run` binds observation data, `condition_from` is
**unset**, and the leading gap `first_obs − t_start` exceeds `K = 5 ×` the
**modal** spacing of the bound observation times (with at least 3 observations,
so the mode is meaningful). `t_start` is the model origin (`simulate.from` in
internal time).

**Severity depends on the canonical stream's `TemporalKind` (§6.8 of the burn-in
/ conditioning proposal):**

- **Incidence (`Interval`)** — _hard error_. The first bin would accumulate the
  entire leading gap and score it against one datum, the gh#134 wrong-number
  (loglik −3416 on the Kano repro). The fit is **rejected**, naming the fix.
- **Prevalence (`Instant`)** — _soft warn_. A prevalence datum reads the
  instantaneous state, so window length does not enter the score; the wide gap
  is only free-running drift the first datum corrects, not a wrong number.

Raised once on the canonical stream in `rust/crates/cli/src/fit/runner.rs`
`FitRunConfig::build`, routed through `crate::util::first_window_guard` (the
severity policy) → `check_first_interval_window` (the detector). Setting
`condition_from` (to run a warm-up, or explicitly to the model start to score
the whole gap) **suppresses the guard** — the modeler has engaged with the
boundary, and `resolve_condition_from` validates the value.

**Why:** `simulate { from = 0 }` (or any origin well before the data window)
against data that begins much later makes the first window enormous relative to
the cadence — e.g. a ~1000-day first window against a 7-day weekly cadence
(~143×). Two silent consequences wreck the fit start: (1) the model **free-runs
unconditioned** over that whole span — no observation pulls the particle filter
back toward the data, so the cloud drifts wherever the uncalibrated
initial-guess dynamics take it before the first likelihood term fires; (2) for
incidence projections the **first incidence window accumulates a giant flow**
(cumulative over ~1000 days instead of ~7), so the opening one-step-ahead
prediction is off-scale and the first prequential / log-likelihood terms are
dominated by that one window. Nothing else in the pipeline points at the cause;
the fit just starts badly.

**Modal vs median, and `K`:** the warning uses the **mode** of the consecutive
inter-obs gaps, not the median, because the median is itself distorted by the
very anomaly being detected — with few observations a single oversized first gap
drags the median up and masks the signal. The mode is the cadence the data
actually settles into (the recurring "every 7 days") and is robust to one or
several outlier gaps as long as the regular cadence is the plurality; gaps are
binned to a ~1% relative tolerance first so 28/30/31-day months and dt rounding
don't shatter the mode. `K = 5` (the conservative end of the design note's 5–10
range): a legitimately missed observation or two gives a 2–4× first window,
which is normal and must not warn; `K = 5` clears that band with margin while
still firing decisively on the pathological case.

**Per-stream (multi-cadence).** The check runs **per observation stream**
against that stream's own first window + modal cadence, so it fires only on the
offending stream — and the message names the per-stream fix
(`condition_from.<label> = first_obs - 1 'week`). The boundary is **explicit**:
camdl does not infer it (an inferred boundary would fail silently on irregular
data), so a wide-window incidence stream with no conditioning is a hard error,
not an auto-corrected default.

**Fix / silence:** set `condition_from = "first_obs - 1 week"` (all streams) or
`[condition_from] <label> = "first_obs - 1 week"` (the offending stream) to run
a covariate-informed warm-up and score the first datum against one cadence (the
principled fix when the early origin is intentional); or move `simulate.from`
closer to the first observation (when it was accidental). To deliberately score
the whole leading window, set `condition_from` to the model start explicitly.
See `docs/dev/proposals/2026-06-09-burnin-conditioning-window.md`,
`docs/dev/proposals/2026-06-10-multi-stream-multi-cadence-union-axis.md` (§3.1),
and `camdl docs fit-toml`.

## Info

| Code | Severity | Category    | Summary                                                                          |
| ---- | -------- | ----------- | -------------------------------------------------------------------------------- |
| I300 | Info     | dimensional | parameter dimension could not be determined (annotate with a more specific kind) |

## Lints

Lints are warnings that catch _semantically valid but discouraged_ patterns —
code that compiles and runs but is likely a bug. They share the diagnostic
infrastructure with `Wxxx` warnings; the `Lxxx` prefix marks them as lints
rather than compiler-internal warnings, which clarifies their intent for users
(a lint is asking "did you mean this?", not "this is suspicious internally").

| Code | Severity | Category       | Summary                                                                               |
| ---- | -------- | -------------- | ------------------------------------------------------------------------------------- |
| L401 | Warning  | discretization | discretization-correction pattern uses fixed time literal — likely meant `dt` (gh#54) |
| L402 | Warning  | dead-code      | compartment declared but referenced nowhere — likely a leftover (gh#168)              |

### L401 — fixed-time-literal in Euler-correction pattern

**Fires when:** the AST contains the shape `(1 - exp(-RATE * TIME_LITERAL))` or
`(1 - exp(-RATE * TIME_LITERAL)) / TIME_LITERAL`, where `RATE` has dimension
`T^-1` and `TIME_LITERAL` is a constant time-typed expression (e.g. `1 'days`,
`0.5 'years`) rather than the `dt` primitive.

**Why:** This is the Euler-multinomial per-step transition-probability template
(pomp's csnippet uses it via `(1 - exp(-(γ+μ)*dt))/dt`). Pinning the `τ` factor
to a fixed time literal produces a model correct only when the runtime
integrator step (`config.dt`) equals that literal. Any other dt produces a
discretization-pinned bias — gh#53 / gh#54 are the canonical real-world example:
He et al. 2010 measles fit at sub-day dt diverged from pomp by 5862 + 12-22 nats
(cohort fire-step bug + this discretization pinning, respectively).

**Fix:** use the `dt` primitive — `(1 - exp(-RATE * dt)) / dt` is dt-invariant
in effective R0 and matches pomp's standard formulation.

**False positives:**

- Pure unit conversions like `mu_per_day = mu_per_year / 1 'years` do NOT match
  (no `exp(...)` wrapping).
- Half-life computations like `t_half = ln(2) / lambda` do NOT match (no time
  literal inside `exp`).

If the fixed time literal IS intentional (a model where the dt-1-day
discretization is the calibrated form, not a bug), v2's per-site suppression
syntax (gh#55) will let users silence the lint explicitly. Until then, the lint
fires; users can suppress at the CLI level via gh#56's `--allow=L401` flag.

### L402 — dead compartment

**Fires when:** a compartment is declared in the `compartments` block but its
name is referenced _nowhere_ in the rest of the model — not in any transition
(stoichiometry, `source`/`dest`, or rate expression), ODE equation, intervention
action, observation projection or likelihood, model-level `let` binding, initial
condition, the balance constraint, the identity-tracked (lineage) set, or a
time-function definition.

**Why:** A compartment touched by none of these contributes nothing to the
dynamics, the observation model, or the initial state. It is almost always a
leftover from editing (a removed transition, a renamed state) rather than an
intentional inert pool. The model still compiles and runs, so this is a lint
(Warning), not an error.

**Fix:** remove the compartment, or wire it into a transition / init /
observation as intended.

**False positives (explicitly NOT flagged):** the reference scan is
comprehensive precisely to keep the false-positive rate at zero. A compartment
is live if it appears in _any_ position above. In particular:

- a compartment used only inside a `let` binding body (`let N = S + I + R`, with
  `R` nowhere else) is live;
- a compartment used only in an observation (`CurrentPop`, `CurrentPopSum`, or
  inside a `DerivedExpr` / likelihood expression) is live;
- a compartment used only as an initial-condition target is live.

`CumulativeFlow`'s string argument names a _flow / transition_, not a
compartment, and is deliberately excluded from the reference set — it never
keeps a compartment alive.

The lint lives in `ocaml/lib/ir/lint.ml` (`Lint.check_model`), mirroring the
Dimcheck pass, and is routed to a non-blocking `Diagnostics.warning` by
`compiler.ml`'s `run_lint` (run by both `camdlc compile` and `camdlc check`).

## Future work

- **gh#55**: per-site lint suppression syntax (e.g. `#[allow(L401)]` attribute
  or `// camdl-allow: L401` comment). Lets model authors silence a lint at a
  specific source location with documented rationale.
- **gh#56**: CLI lint-policy knobs (`--allow=L401`, `--deny=L401`, `-Werror`).
  Depends on gh#55 for `--allow` semantics.

Both deferred from gh#54's v1 scope. The bare minimum here is the catalog (this
file) plus the L401 inline emit; structured lint infrastructure follows when ≥ 3
lints have customers asking for suppression.
