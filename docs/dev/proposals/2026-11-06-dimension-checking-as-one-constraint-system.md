# Dimension checking as one constraint system over every model position

Date: 2026-11-06

Status: implemented, with the unit-literal IR node in `ir/VERSION` 0.43;
deviations from the plan are recorded inline where they occur

Builds on: `docs/dev/proposals/2026-11-05-size-brackets-and-dim-clause.md` (§4,
§5) and `docs/dev/notes/2026-11-05-dimension-inference-measurement.md`

## 1. Summary

The dimension checker (`ocaml/lib/ir/dimcheck.ml`, 1,882 lines) has an inference
core for single expressions that is exhaustive, but it decides _which_
expressions to check, and against what, from a hand-written list spread across
about 650 lines of `check_model`. That list skips whole regions of the model:
every intervention, event and reactive-trigger expression, every prior argument,
quantity thresholds and times, smooth-spline modes, and (under a blanket
permissive mode) the inside of every likelihood argument and every derived
projection. Twenty-one minimal models written for this review compile cleanly on
both `main` (`6bcffa45`) and this branch although each contains a dimension
error, and four more valid models are refused (§2). Separately, the checker
mixes two inference mechanisms, two passes, three rounds and a fixpoint, a
hand-written cross-transition analysis and four global mode flags, and it counts
solver contradictions without reporting them.

This proposal replaces all of it with one design:

1. **One list of positions.** A single function, `Dim_sites.sites`, names every
   expression-bearing position of the expanded IR (plus the prior arguments the
   expander folds away) together with its dimensional obligation. It
   destructures every IR record with a closed pattern and every variant without
   a catch-all, under warnings 4, 8 and 9 as errors, so a new IR field or
   constructor does not compile until someone writes its rule.
2. **One equation system.** Each expression contributes linear equations over
   integer exponent vectors (Kennedy, "Dimension types", ESOP 1994), each tagged
   with the position it came from. Each obligation of a model position
   contributes one more equation. Prior arguments are covered too, but are
   checked after the solve and contribute none (§5.9).
3. **One solver.** Exact integer elimination on unit pivots, then a Hermite
   normal form on the small remainder, distinguishes determined,
   underdetermined, contradictory and integer-infeasible systems. Every
   contradiction and every integer infeasibility is a located error.

It also fixes the semantic errors the review found: the overdispersion variance
has dimension T, not 1 (E308 is wrong); `'count` literals lose their P during
lowering; a typed constant silently overrides a unit literal; a unit literal
passed directly as a log-scale prior argument is converted multiplicatively;
hierarchical prior arguments get no name checks and can produce a chain that
never moves and exits 0.

**Return on investment: high, and I recommend doing it before families Stage
A.** Families multiplies the expression-bearing positions (indexed priors,
non-centred `mu + sd * z[p]` hierarchies, coefficient families), and each one
added under the current structure is another place to forget. The silent
failures found here are of the kind `CLAUDE.md` names as the worst outcome: a
model that runs and fits and is wrong. The parts with the highest return are the
position list and the solver's reporting; the deletion of the old machinery is
lower return but is what makes the result maintainable, and leaving the old
checker alongside the new one is the outcome to avoid.

## 2. What the review established

Every result below is from compiling a minimal model with this branch's `camdlc`
and with `main`'s; both give the same result in every case, so none is a
regression from this branch. "Clean" means exit 0 and no diagnostic.

### 2.1 Errors that compile clean

| Position                                 | Example                                                          | Required                                          |
| ---------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| intervention fraction                    | `transfer(fraction = gamma, ...)`                                | 1                                                 |
| intervention fraction, unit literal      | `transfer(fraction = 0.1 'per_day, ...)`                         | 1                                                 |
| intervention count                       | `transfer(count = gamma, ...)`                                   | P                                                 |
| compartment assignment                   | `{ I = gamma at = [10] }`                                        | P                                                 |
| event addition                           | `add(I, gamma) every 365 'days at_day 10`                        | P                                                 |
| schedule time expression                 | `... at [gamma]`                                                 | T                                                 |
| reactive threshold                       | `when sum_observed(cases, ...) >= thr`, `thr : rate`             | the stream's dimension, P                         |
| reactive action fraction                 | `sia_cov : rate` as `fraction`                                   | 1                                                 |
| log-scale prior argument with a unit     | `log_normal(mu = -2 'per_day, ...)` under weeks → IR μ = −14     | refused                                           |
| dimensionless prior argument with a unit | `log_normal(..., sigma = 0.5 'days)` under weeks → IR σ = 0.0714 | refused                                           |
| Normal likelihood mean                   | `normal(mean = gamma * projected, ...)` on a `count` column      | P                                                 |
| Normal likelihood sd                     | `normal(..., sd = gamma)` on a `count` column                    | T exponent 0                                      |
| sum inside a strict argument             | `binomial(p = projected / N0 + gamma)`                           | operands agree                                    |
| derived projection under a strict family | `projected = I / N + gamma`, `beta(mean = projected)`            | operands agree                                    |
| quantity threshold                       | `first_above(I, gamma)`, `count_above(I, gamma)`                 | P                                                 |
| quantity series                          | `final(I + gamma)`                                               | operands agree                                    |
| `value_at` time                          | `value_at(I, N0)`                                                | T                                                 |
| identity-link smooth level and scale     | `link = identity` on a `'per_day` spline, `a_b : real dim 1`     | D = T⁻¹                                           |
| smooth modes                             | `z[mode] : real dim T`                                           | 1                                                 |
| typed constant over a unit literal       | `let tau : rate = 30 'days` → constant 30.0, kind rate           | refused                                           |
| two `positive` factors, inconsistent use | `kappa * w * S * I / N` and `kappa * w * V * I`                  | contradiction (solver counts it, reports nothing) |

### 2.2 Valid models refused

| Model                                                          | Refusal        | Why it is valid                                                                                                                                 |
| -------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `sigma_se : positive dim T` in `overdispersed(rate, sigma_se)` | E308           | the runtime draws Gamma(shape = dt/σ², scale = σ²/dt) (`rust/crates/sim/src/rng.rs:555`); the shape is dimensionless only if σ² has dimension T |
| `gamma * I^k / N^(k - 1)`, `let k : real = 2`                  | E301           | `k` is a compile-time integer                                                                                                                   |
| `gamma * 5 'count`                                             | E300 (got T⁻¹) | `'count` lowers to a bare constant and loses P (`ocaml/lib/compiler/expander.ml:3804`)                                                          |
| `recovery @ I / (5 'days)`                                     | E107           | the unit literal is parenthesized; E107's ambiguity does not arise                                                                              |

And one that is refused for want of syntax: a literal per-person-per-day
coefficient. `0.0001 * S * I` is E300 (the literal is dimensionless),
`let b : positive dim P^-1*T^-1 = 0.0001` is a syntax error (typed `let` takes
no `dim` clause), while `(1/10000) * S * I` compiles, because a ratio of
literals takes whatever dimension its context needs. The same rule lets
`(1/7) * I * I` compile as a transition rate.

### 2.3 Priors are on θ, and the Jacobian is right

Traced by reading (a subagent's trace, spot-checked):

- One density implementation, `rust/crates/sim/src/inference/prior.rs`,
  normalized, evaluated on θ in model units. Log-normal μ and σ are the mean and
  sd of ln θ (`:224`).
- PMMH and MH-on-ODE (`cli/src/fit/pmmh.rs:603-605, 695-718`), the PGAS NUTS
  move (`pgas.rs:4208-4216`) and the PGAS MH-within-Gibbs move
  (`pgas.rs:4360-4369`) all use log-likelihood + log p(θ) + log|J|, with the
  sampler's change of variables chosen from the parameter's domain
  (`types.rs:129-140`). IF2 and the NLopt optimizers use no prior.
- The IR's `transform` field was always `Identity` and no sampler read it; only
  run identity hashed it. It is removed in the same `ir/VERSION` 0.43 bump that
  adds the unit-literal node (decision 3).

Three defects, outside the checker:

- Hierarchical prior arguments skip the name and positivity checks that constant
  ones get (`expander.ml:5478-5530` run only on the constant path).
  `log_normal(mean = m_h, sd = s_h)` compiles; Rust looks up `mu` and `sigma`,
  gets NaN, and the prior is −∞ everywhere. Reproduced to the IR by me; the
  subagent ran 20 PMMH iterations, which never accepted a move and exited 0.
- A hierarchical argument may mention time or state (`expander.ml:5677`), which
  is NaN at evaluation (`hierarchical.rs:130`). Compiled by the subagent, not
  run.
- PMMH starts from a −∞ log-posterior without refusing; PGAS refuses one. Shown
  by the run above, not independently.

## 3. The external directive, evaluated

An external reviewer read the review and `dimcheck.ml` and wrote an
implementation directive. My position on each part:

**Agree, adopted as written.** The distinction between T (a physical dimension
whose unit can change) and P (a semantic dimension for counts, which are
cardinal and whose variance formulas need not be homogeneous in P); removing the
blanket permissive mode in favour of rules local to each construct; one
traversal with closed patterns and an independent path-set coverage test, with
the three mutation checks the directive lists; one constraint generator and one
solver; integer infeasibility as an error, with the observation that rational
elimination alone misses it; underdetermination reported but not refused; every
equation carrying its origin; the list of literal and unit defects (each of
which I reproduced, §2.2); static integer and rational exponents; one set of
prior validation rules for constant and hierarchical arguments; promoting every
model from the review to a permanent test; keeping unrelated IR cleanups (the
dead `transform` field) out of this work.

**Agree, with a narrower scope.**

- _The count-noise exception._ Its scope should be exactly the `sd` of a
  `normal` likelihood whose scored column is declared `count`, not every Normal,
  and not a `real` column that happens to hold a count-valued mean (§5.6). The
  `normal` initial-value law stays strict. Priors are not model positions at all
  (§5.9).
- _Deleting union-find._ Plain equalities are equations too, so the separate
  mechanism goes; whether a union-find fast path survives inside the solver is
  decided by the compile-time benchmark, not in advance (§6.5).
- _Unit literals as `UncheckedDim`._ Right: a unit literal is a typed constant,
  not an escape. It gets its own IR node, `unit_literal`, in `ir/VERSION` 0.43
  (decision 3).

**Disagree on ordering.** The directive's first commit "captures existing bugs
as committed regression tests". This repository does not land red tests ahead of
their fix (an ignored or expected-to-fail test on `main` is refused by project
practice). Each negative model therefore lands in the commit that makes it pass;
the first commit carries the positive models, which pass today, and the fixes
that are independent of the new architecture (§8).

**Log-normal: the directive's position, and the maintainer's, is right, and I
withdraw the median proposal.** My argument for a median was that μ is not
dimensionally meaningful. But the hazard that argument identified is narrower
than I made it: every plain-number prior argument is already a number in
`time_unit` (a `normal(mu = 0.2, ...)` on a rate is per day under days and per
week under weeks), so μ is not uniquely unit-dependent; it differs only in that
a change of unit shifts it additively (ln 7 for days to weeks) instead of
scaling it. The concrete defect is that unit literals on μ and σ are converted
multiplicatively, and refusing unit literals on log-scale and dimensionless
arguments fixes it without changing the interface. The `mu`/`sigma`
parameterization is the one in Stan's `lognormal(mu, sigma)` and R's
`dlnorm(meanlog, sdlog)` (R Core Team, `stats` package documentation; Stan
Development Team, _Stan Functions Reference_, "Lognormal distribution"; both
from memory, sections not re-checked), and departing from it would cost every
user to fix a problem only unit literals cause. The rule becomes: a unit literal
passed directly as μ or σ must be dimensionless, and one that is not is refused
with a hint that names the additive conversion; `log(0.2 'per_day)` and a
hierarchical μ of any declared dimension are accepted (§5.9).

**Does the directive go too far anywhere?** In one place it risks it: "every
expression-bearing position" includes the derivative maps autodiff writes into
the IR (`rate_grad`, `diffable.grad`, `ic_grad`, ...). Checking them would be a
test of autodiff, not of the user's model. They are exempt with a stated reason
(§5.1), and the coverage test treats them as exempt paths.

## 4. Design

Three modules, each depending only on the one before it:

```
Dim_solve   integer linear algebra; knows nothing about the IR
Dim_sites   the list of positions; knows the IR, nothing about solving
Dimcheck    turns sites into equations, solves them, writes diagnostics
```

### 4.1 The public interface

```ocaml
(* ocaml/lib/ir/dimcheck.mli *)
type prior_source = {
  param  : string;
  family : string;                  (* "normal", "log_normal", ... *)
  args   : (string * Ir.expr) list; (* resolved, unit literals intact *)
}

val check_model :
  ?prior_args:prior_source list ->            (* every prior, before folding *)
  ?constant_dims:(string * (int * int)) list ->  (* typed constants' `dim` clauses *)
  Ir.model -> result                          (* result type unchanged *)
```

Two inputs come from the expander beside the IR. `prior_args` exist because a
constant prior's arguments are folded to floats before the IR is built, so the
unit literal in `mu = -2 'per_day` is gone from it; the expander resolves each
argument with `resolve_expr` and records the expression. `constant_dims` carry
the `dim` clause of a typed constant, which the IR's constant record does not.
Nothing new is serialized and the IR does not change. The compiler and
`camdlc inspect` pass both. A model checked without them (an IR reloaded from
JSON, a model built in a test) gets weaker results, never contradictory ones: a
constant with a `dim` clause is inferred from its use rather than declared, and
constant priors' unit literals go unchecked, since the IR no longer holds them;
hierarchical priors are read from the IR. The next IR schema bump puts the
clause on the constant record and closes the first gap.

The checker runs where it always has: after validation and before autodiff,
constant folding and LICM, so a table lookup still carries its declared cell
dimension.

### 4.2 `Dim_sites`: one function names every position and its obligation

```ocaml
(* ocaml/lib/ir/dim_sites.ml *)
[@@@warning "@4@8@9"]
(* 4: a `_` catch-all over a variant is an error, so a new constructor cannot
      fall into an existing arm.
   8: a non-exhaustive match is an error.
   9: a record pattern that omits a field is an error, so a new field cannot
      be skipped by not naming it. *)

type key =                       (* things whose dimension other positions refer to *)
  | K_param    of string         (* a parameter or a typed constant *)
  | K_table    of string         (* a table's cell dimension *)
  | K_binding  of string
  | K_stream   of string         (* an observation stream's projected dimension *)
  | K_column   of string * string   (* (stream, column) *)
  | K_quantity of string         (* a quantity's reported dimension *)
  | K_series   of string         (* the series a quantity reduces *)

type obligation =
  | Has      of dim_vec          (* a fixed dimension: P·T⁻¹, P, 1, T, ... *)
  | Equals   of key              (* the dimension of [key] *)
  | Equals_t of key              (* [key]'s T exponent; P free (count noise, §5.6) *)
  | Defines  of key              (* this expression's dimension is [key]'s *)
  | Free                         (* operator rules only (contrasts, prior args) *)
  | Exempt   of exemption
and exemption = Constant_argument   (* a literal forcing argument, read in the
                                       forcing's own unit *)

type literals = Literals_dimensionless | Whole_terms   (* §5.2 *)

type strength =
  | Binding   (* a model position: its equations and obligation enter the solve *)
  | Checked   (* a prior argument: read after the solve, never an equation (§5.9) *)

type site = {
  subject    : subject option;   (* where diagnostics point *)
  role       : string;           (* "transition 'infection' rate", ... *)
  expr       : site_expr;        (* E of Ir.expr | Scalar | Contrast *)
  obligation : obligation;
  literals   : literals;
  code       : string;           (* the code a violated obligation reports *)
  strength   : strength;
  stream     : string option;    (* the observation a Projected leaf reads *)
  note       : string option;    (* guidance the diagnostic carries *)
}

val sites : Ir.model -> site list
```

The top of `sites`, to show what the closed pattern buys:

```ocaml
let sites (m : model) : site list =
  let { name = _; version = _; time_unit = _; description = _; origin = _;
        origin_rata_die = _; compartments = _; doc_index = _;
        model_structure = _; identity_tracked_compartments = _;
        output = _; simulation = _; presets = _;  (* no expressions *)
        constants = _;            (* declarations, read as K_param *)
        per_eval_bindings = _;    (* created by LICM, after dimcheck *)
        ic_grad = _;              (* derivative map *)
        transitions; ode_equations; time_functions; tables; interventions;
        observations; parameters; bindings; initial_conditions; balance;
        quantities; contrasts } = m in
  ...
  List.iter (binding_site acc) bindings;          (* first: see below *)
  List.iter (transition_sites acc) transitions;
  ...
```

and one nested function, to show the rules reading like the language spec:

```ocaml
let action_sites acc ?subject ~role:what (a : action) =
  let role field = Printf.sprintf "%s `%s`" what field in
  let count field e =
    add acc ?subject ~role:(role field) ~code:"E314" ~literals:Whole_terms
      (Has population) (E e) in
  match a with
  | FractionTransfer { src = _; dst = _; fraction } ->
    add acc ?subject ~role:(role "fraction") ~code:"E314" (Has dimensionless) (E fraction)
  | AbsoluteTransfer { src = _; dst = _; count = c } -> count "count" c
  | Set { compartment = _; value } -> count "value" value
  | AddAction { add_compartment = _; add_count } -> count "count" add_count
```

Derivative maps written by autodiff (`rate_grad`, `sigma_sq_grad`, `grad`,
`proj_grad`, `ic_grad`, `projection_state_grad`), lineage weights and LICM's
per-eval bindings are bound to `_` with a comment: they are not the user's
expressions. Binding sites come first, because the solver blames the later of
two disagreeing equations, and a definition must not be blamed for a mistake at
one of its uses.

**What the compiler guarantees.** Adding a field to `Ir.model`, `transition`,
`action`, a likelihood record or any other record `sites` destructures is a
compile error in `dim_sites.ml` (warning 9). Adding a constructor to `action`,
`likelihood`, `init_spec`, `time_func_kind`, `spline_coefs`, `fire_source`,
`trigger_expr`, `quantity_body`, `temporal_reduce` or any other variant it
matches is a compile error (warnings 4 and 8). The author of the new field
writes its rule or binds it to `_` with a reason.

**What the compiler cannot guarantee, and the test that does.** A field can
still be bound to `_`. `ocaml/test/test_dim_sites.ml` closes this without
depending on `Dim_sites`: it serializes every IR golden with `Serde`, collects
every outermost expression node of the JSON (an object with one key naming an
expression constructor), skips the exempt keys above, and requires the multiset
of those nodes to be covered by the multiset of site expressions serialized the
same way. An uncovered node is reported with its JSON path. The mutation checks:
a new field `extra : expr` on `fraction_transfer` and a new constructor
`Scale of expr` on `action` each fail to compile in `dim_sites.ml`; binding
`fraction = _` makes the test name the uncovered
`interventions[0].actions[0].fraction_transfer.fraction` of three goldens.

### 4.3 Expressions are linear forms; equations carry their origin

```ocaml
(* ocaml/lib/ir/dimcheck.ml *)
type form  = { vars : (int * int) list; k : dim_vec }   (* Σ cᵥ·xᵥ + k *)
type value = Any | F of form   (* Any: the literal 0, or a whole-term literal *)
```

`gen` turns an expression into a linear form over dimension variables, one per
key. A product is the sum of its factors' forms, a quotient their difference,
and a static power `a^n` is `n` times `a`'s form, so these need no equation. An
equation is emitted wherever two forms must agree: the operands of `+`, `min` or
a comparison, and the branches of a conditional (`l = r`); the argument of
`exp`, `log` or a trigonometric function (`a = 0`); a square root or a static
fractional power (`q·x = p·a` for a fresh `x`). Each site then adds one equation
for its obligation. A known dimension (a `rate` parameter, a column declared
`count`) is a `Declared` equation on its key's variable. The rule for an
operator is the same at every site.

```ocaml
(* ocaml/lib/ir/dim_solve.mli *)
type stage  = Declared | Structure | Obligation
type origin = { site : int; stage : stage; what : string }  (* site: an equation id *)
type eqn    = { terms : (var * int) list; rhs : int array; origin : origin }
```

The checker keeps, for each equation id, the site it came from, its code, and
what kind of agreement it states, so a conflict can be reported in the site's
own terms.

### 4.4 The solver distinguishes four outcomes

The system decouples by base: an equation over exponent vectors holds separately
for its P part and its T part, so the solver works on one coefficient matrix
with one right-hand side per base.

```ocaml
type conflict_kind = Contradiction | Not_integral
type conflict = { culprit : eqn; kind : conflict_kind; context : origin list }

val solve     : nbases:int -> eqn list -> outcome
val conflicts : outcome -> conflict list            (* every one, in order *)
val value     : outcome -> var -> int array option  (* Some when determined *)
val relations : outcome -> var list -> relation list
(* the integer relations among free variables: kappa + w = T⁻¹ *)
```

1. **Order.** Equations are added `Declared`, then `Structure`, then
   `Obligation`, so a conflict is usually found when an obligation is added and
   reported at that site with its code: the transition rate is blamed, not a
   product inside it.
2. **Exact elimination on unit pivots.** An equation is reduced against the
   accepted rows. If it reduces to 0 = 0 it is redundant. If it reduces to 0 = r
   with r ≠ 0 it is a contradiction: reported, with the origins it was reduced
   against, then dropped so it does not cascade. Otherwise, if some coefficient
   is ±1, it is pivoted on the newest such variable, which keeps elimination
   local. A ±1 pivot is unimodular, so it preserves the integer solutions and
   keeps every coefficient an integer.
3. **The residual.** A row can reach the end of reduction with no ±1
   coefficient: directly from a power or a root (`2x = a`), or by substitution
   from products and quotients alone, as a·b = T⁻¹ with a/b = 1 reduces to 2a =
   T⁻¹. Such rows are set aside and checked together by a column Hermite normal
   form: the system has an integer solution exactly when forward substitution
   divides evenly, and a rank comparison of [A] with [A | b] separates a
   rational contradiction from integer infeasibility. This is what catches
   `(kappa * w)^2 * I` in a rate: 2·κ + 2·w = −T has rational solutions and no
   integer one.
4. **Arithmetic.** Native integers, every addition and multiplication checked,
   in the solver and in equation generation alike (forms, exponent vectors,
   static powers); an overflow is E310, never a wrapped value.
5. **Underdetermination.** A variable left free is not an error. Each
   undetermined parameter that some model position reads gets I300 (Info) with
   the relations that are determined among the undetermined parameters ("only
   this is determined: kappa · w is T⁻¹"). No dimension is invented. A parameter
   no model position reads (one used only in priors, or not at all) is not
   reported.

**One base at a time when needed.** A count-noise sd (§5.6) holds on T alone.
When any such equation exists the checker solves P without them and T with every
equation, as two one-base systems, which is exact because the bases decouple;
otherwise it makes one two-base solve.

### 4.5 Diagnostics

A conflict at an obligation reports the site's code, its role, the expected
dimension and the dimension the rest of the system implies, and the uses it
conflicts with:

```
error[E300]: transition 'infection' rate has wrong dimension
  = note: rate = (((alpha * S) * I) / N)
  expected dimension: P*T^-1 (population-level rate)
  got dimension: P^2 (P^2*T^0)
  the conflicting requirement comes from: transition 'waning' rate
```

A conflict among `Structure` equations (an inconsistent sum inside an
expression) reports E302 at the site, naming both operands' dimensions. When one
operand is a bare number, the diagnostic says so and its hint gives both fixes
(§5.2). The notes name only uses that require a dimension (sums, function
arguments, obligations), not definitions the value flows through. No minimal
unsatisfiable core is computed: the equations the culprit was reduced against
are the context, which is deterministic and enough to act on.

## 5. The rules

### 5.1 Every position

Dimensions: P a population count, T time, D the enclosing forcing's declared
unit, θ the parameter a prior is on. "Lit" marks positions where a literal that
is an entire additive term takes the obligation (§5.2).

| Position                                                                                                                                 | Obligation                                         | Code |
| ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ---- |
| transition rate                                                                                                                          | P·T⁻¹ (Lit)                                        | E300 |
| overdispersion σ²                                                                                                                        | T (§5.7)                                           | E308 |
| ODE derivative                                                                                                                           | P·T⁻¹ (Lit)                                        | E306 |
| balance expression                                                                                                                       | P (Lit)                                            | E305 |
| binding body                                                                                                                             | defines the binding's dimension                    | E302 |
| forcing `lag`                                                                                                                            | T                                                  | E309 |
| sinusoid baseline, amplitude; piecewise, interpolated, periodic values; Fourier coefficients; identity-link explicit spline coefficients | D (constant arguments exempt, as now)              | E312 |
| periods, phases, breakpoints, interpolation times, knots                                                                                 | T                                                  | E312 |
| log-link spline coefficients                                                                                                             | 1                                                  | E301 |
| smooth `per`                                                                                                                             | T                                                  | E312 |
| smooth level, slope, scale, modes: log link                                                                                              | 1, T⁻¹, 1, 1 (§5.8)                                | E312 |
| smooth level, slope, scale, modes: identity link                                                                                         | D, D·T⁻¹, D, 1 (§5.8)                              | E312 |
| inline table values                                                                                                                      | the table's cell dimension                         | E302 |
| table lookup index                                                                                                                       | 1                                                  | E311 |
| initial value; Poisson rate; Binomial n; NegBinomial mean                                                                                | P (Lit)                                            | E313 |
| initial Binomial p; NegBinomial dispersion                                                                                               | 1                                                  | E313 |
| initial Normal mean, sd                                                                                                                  | P, P (Lit)                                         | E313 |
| transfer fraction                                                                                                                        | 1                                                  | E314 |
| transfer count; assigned value; added count                                                                                              | P (Lit)                                            | E314 |
| schedule time expression                                                                                                                 | T                                                  | E315 |
| reactive threshold parameter                                                                                                             | the stream's dimension                             | E315 |
| projection `DerivedExpr`                                                                                                                 | defines the stream's dimension                     | E302 |
| Poisson rate; NegBinomial and zero-inflated mean                                                                                         | P (Lit)                                            | E304 |
| Binomial and BetaBinomial n                                                                                                              | P (Lit)                                            | E304 |
| Binomial and Bernoulli p; Beta mean; zero-inflation π                                                                                    | 1                                                  | E304 |
| dispersions; Beta concentration; BetaBinomial shapes                                                                                     | 1                                                  | E304 |
| Normal mean                                                                                                                              | the scored column's dimension (§5.6)               | E304 |
| Normal sd                                                                                                                                | §5.6                                               | E304 |
| prior arguments                                                                                                                          | `Checked`, no equation: explicit units only (§5.9) | E316 |
| quantity series                                                                                                                          | defines the series dimension                       | E302 |
| quantity thresholds (`count_above`, `first_above`, ...)                                                                                  | the series dimension (Lit)                         | E317 |
| `value_at` time                                                                                                                          | T                                                  | E317 |
| derived-quantity arithmetic; contrasts                                                                                                   | operator rules of §4.3                             | E297 |
| derivative maps (`rate_grad`, `rate_state_grad`, `sigma_sq_grad`, `diffable.grad`, `proj_grad`, `ic_grad`, `projection_state_grad`)      | exempt: produced by autodiff                       | —    |
| lineage parent-pool weights                                                                                                              | exempt: derived from the rate by the expander      | —    |
| per-eval bindings                                                                                                                        | exempt: created after dimcheck                     | —    |

E311 and E314–E317 are unused codes; E303 is retired (§6.4).

### 5.2 Literals follow one documented rule

Today a literal is dimensionless, except zero (any dimension), a ratio of two
literals (any dimension), and any literal in an initial condition (any
dimension). The ratio rule is the inconsistency: `0.0001 * S * I` is refused and
`(1/10000) * S * I` is accepted, and `(1/7) * I * I` passes as a rate.

The rule this proposal adopts (decision 1):

> A numeric literal is dimensionless, with two exceptions. Zero takes any
> dimension, since 0·u = 0 for every unit u. A literal that is an entire
> additive term of a position marked Lit in §5.1 takes that position's
> dimension: `I = 10`, `S = N0 - 50`, `add(I, 10)`, `rate = 100`, and a constant
> importation term `beta * S * I / N + 0.5`.

As implemented, arithmetic of literals counts as a literal: `S = 0.95 * 10000`
is a count, since a number written as arithmetic is still a number, while as a
factor it is dimensionless, so `(1/730) * R` and `0.0001 * S * I` in a rate are
still refused.

A literal inside a product, a ratio of literals included, is dimensionless.
`(1/730) * R` in a rate becomes E300, with a fix-it to `R / (730 'days)`.
Measured on the repo's tracked models: one occurrence
(`ocaml/golden/seir_spatial_5_inference.camdl:69`). The Ebola and other
downstream corpora must be measured the same way in the implementation commit
before the change lands.

The initial-condition rule becomes this rule. The expander's split of `I = 10`
into `10 * cfr` across a hyper-Erlang chain must emit the 10 as a `'count`
literal, since the split is the expander's, not the user's.

For literal coefficients that carry a dimension, typed constants accept the
`dim` clause this branch introduced (decision 2):
`let b : positive dim P^-1*T^-1 = 0.0001`. No new syntax concept is added; it is
the parameter grammar reused.

### 5.3 Unit literals keep their dimension, and conflicts are refused

- `'count` lowers with dimension P (today a bare constant, `expander.ml:3804`).
  `'ratio` lowers dimensionless, as now.
- A typed constant whose unit-literal initializer has a different dimension is
  refused: `let tau : rate = 30 'days` (today accepted as the number 30.0 of
  kind `rate`).
- E107 does not fire when the unit literal is parenthesized: `I / (5 'days)`.
- A unit literal whose dimension disagrees with its prior argument's
  conventional dimension is refused, including one passed directly as a
  log-normal argument (§5.9).

### 5.4 Powers

- `a^n` with n a compile-time integer (a literal, a named constant, or any
  expression `Constant_fold` reduces to an integer): x = n·a.
- `a^(p/q)` with a compile-time rational exponent: q·x = p·a, an integer
  equation that is infeasible unless the exponents divide (the solver's residual
  handles it, as for `sqrt`).
- Otherwise: the exponent and the base are both dimensionless, and so is the
  result. When the base's dimension is unknown, this _adds_ the equation base =
  1 rather than leaving the base unconstrained, as `infer_pow` does today. For a
  base with dimension d ≠ 1 the result would be d^y with y real and changing
  during a fit: not a point of the integer lattice, so no equation can represent
  it, and its value would scale by c^y under a change of unit.
  `unchecked_dim(...)` remains the explicit escape (the He et al. 2010 (I + ι)^α
  term).

### 5.5 `unchecked_dim` stays the only escape

The wrapped expression is opaque: the checker reads only the asserted dimension,
and does not infer the inside. The escape's motivating case, He et al.'s (I +
ι)^α, is a count under a parameter exponent, which the rule of §5.4 refuses, so
inferring the inside would refuse the escape's own example. A unit literal is
not an escape: it lowers to its own IR node,
`unit_literal { value, dim, unit }`, whose dimension the checker reads as a
fact. The one compiler-generated `UncheckedDim` left is the log-link spline
shift.

### 5.6 Count-noise Normal likelihoods: P is free in the sd, T is not

For a `normal` likelihood:

- **mean** has the scored column's dimension. A `count` column is P. A `real`
  column has a dimension variable shared by the column and the mean, so the
  dimension of continuous data is inferred, not ignored.
- **sd**, when the scored column is `count`: the equations generated inside the
  sd expression, and the obligation itself, keep only their T components. T must
  match the mean's; P is unconstrained.
- **sd**, otherwise: the mean's dimension, exactly.

The P exemption is justified, and limited, by what counts are. Variance formulas
for count data add a term in the count to a term in its square: He et al.
(2010)'s reporting model has variance ρ(1−ρ)C + ρ²ψ²C² (_J. R. Soc. Interface_
7:271–283; from memory), and the reviewer's `sd = projected^2` is a legitimate
if odd noise model. Such formulas are consistent only because a count is a
number of individuals, not a quantity whose unit can be rescaled; a Poisson's
variance equals its mean only in units of one individual. Nothing similar
excuses a T mismatch. An sd with a T exponent different from the mean's means a
rate was written where a count was meant, which is the daily versus weekly
reporting error.

Keeping only the T components is exact because the system decouples by base. A
sum inside the sd whose terms differ in T, such as `projected + gamma`, is still
an error. The P exemption covers the sd's sums, products, powers and roots, the
operations a count variance formula is built from; the argument of `exp`, `log`
or a trigonometric function stays dimensionless in every base, so
`sd = exp(projected)` is E301. Nothing about counts being cardinal makes a
transcendental function of a count meaningful, and it is almost always a
mistake. The exemption does not extend to the initial `normal` law or to a
`real` column: there, sd and mean have the same dimension. Whether some existing
model uses a `real` column for count data with a P-inhomogeneous sd is measured
across the corpora before this rule lands; if one does, the right fix is to
declare the column `count`.

### 5.7 Overdispersion σ² has dimension T

The runtime draws a unit-mean Gamma multiplier with shape dt/σ² and scale σ²/dt
(`rust/crates/sim/src/rng.rs:555, 608`); the documented variance is mean +
mean²·σ²/dt. A shape is dimensionless, so σ² has the dimension of dt. This is
the gamma white noise of Bretó et al. (2009), _Ann. Appl. Stat._ 3:319–348 (from
memory), and pomp's `rgammawn`. E308's rule changes from 1 to T.

The three he2010 external cases declare `sigma_se : positive dim 1`; they change
to `dim T`. `param_dim` is serialized, so their IR changes and their manifests
are regenerated (only `case_sha` moves; the numeric comparison against pomp is
untouched, since the runtime was already right).

### 5.8 Smooth splines: the whitening carries T^{3/2}, and κ cancels it

From the estimable-bsplines proposal §6.3: Ω is the curvature penalty matrix,
whose quadratic form wᵀΩw approximates ∫ g''² dt, so for a dimensionless g (log
link) Ω has dimension T⁻³. M is built with MᵀΩM = I, so M has dimension T^{3/2}.
κ = √(K − 2) · T_ref^{3/2}, with T_ref the declared `per`, has dimension
T^{3/2}. Hence M/κ is dimensionless, and in

w = a·1 + b·(ξ − t_m) + (τ/κ)·M·z

each term has the dimension of w when a has it, b has it per unit time, τ has
it, and z is dimensionless. Under the log link w is dimensionless (1, T⁻¹, 1,
1); under the identity link w has D (D, D·T⁻¹, D, 1). This is a derivation from
the proposal's definitions, not a reading of the runtime; the implementation
commit checks it against the Rust construction of M and κ.

### 5.9 Priors: numerical coordinates, with explicit units honoured

Dimensional analysis has two jobs in camdl, and they are different in kind. In
model dynamics it is a type system: a transition rate that is not P·T⁻¹, or a
duration added to a rate, is wrong epidemic mathematics, and §5.1 refuses it. In
prior specification it is closer to a modelling linter. A prior is a probability
measure on the number θ takes in model units; the parameter's dimension comes
from its declaration and its use in the model, never from the family chosen for
its prior. `gamma : rate ~ beta(alpha = 2, beta = 2)` is a valid prior on a
recovery rate whose numerical value lies in (0, 1) per `time_unit`. That a Beta
family is not closed under a change of unit does not make it an invalid choice:
the measure transforms with the Jacobian of the change of coordinates, and its
density, which has units of 1/[θ], changes form.

Three rules follow, and nothing else is checked.

**Validity is a hard error.** One function in the expander, applied to both
branches of `classify_and_resolve_prior_spec`, checks argument names, required
arguments, the family, finite constant arguments and their domains (a negative
Normal sigma), and the permitted expression forms: a hierarchical argument may
mention parameters, constants and literals only; time, `dt`, compartments,
projections and observation columns are refused. An argument whose domain
depends on a parameter's value cannot be checked statically, and the runtime
already handles it: `prior.rs:196–272` collapses a non-finite or out-of-domain
argument to −∞, never NaN. PMMH refuses to start where the log prior is
non-finite, naming the parameter, as PGAS refuses a non-finite start
(`sim/src/error.rs:159`); this is a guard in `cli/src/fit/pmmh.rs`, not a change
to any acceptance ratio. It is narrower than refusing a non-finite
log-posterior: a −∞ log-likelihood at the start stays the warning of gh#334,
since a chain can still reach a finite-likelihood region from there, while a
start of zero prior density under a prior that is −∞ near it (a hierarchical σ
whose hyperparameter is negative) cannot, and the likelihood-only backstop of
gh#226 does not see it. A start outside a constant prior's support is already
refused earlier, by the bounds-versus-support check.

**Explicit units are honoured, and only they are checked.** Inside a prior
argument a parameter reference is a number in model units, and arithmetic on
such numbers is permitted whatever the parameters' declared dimensions:
`log_normal(mu = log(m_h), ...)` with `m_h : rate` takes the logarithm of the
number `m_h` is, and is valid. A unit literal is the one thing in a prior
argument that asserts a physical quantity, and it is converted to model units as
everywhere else (`normal(mu = 0.2 'per_day)` under weeks gives 1.4;
`log(0.2 'per_day)` under weeks gives ln 1.4). Its assertion is checked by a
small evaluation over three values, not by equations and not by the solver:

```ocaml
(* in ocaml/lib/ir/dimcheck.ml *)
type value =
  | Bare                     (* only bare numbers: a coordinate, no claim *)
  | Units of Dimcheck.dim_vec  (* determined by unit literals alone *)
  | Coord                    (* depends on a parameter or a function value *)
```

| Expression                      | Value                                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------------- |
| bare literal                    | `Bare`                                                                                  |
| unit literal of dimension d     | `Units d` (`'ratio` is `Units 1`; `'count` is `Units P`)                                |
| parameter or constant reference | `Coord`                                                                                 |
| `a + b`, `a - b`                | `Units a` with `Units b`: equal, or E316; `Bare` adopts the other; `Coord` absorbs      |
| `a * b`, `a / b`                | `Units` with `Units`: the product or quotient; `Bare` is a factor of 1; `Coord` absorbs |
| `-a`                            | `a`                                                                                     |
| `a ^ n`, n a literal            | `Units (n·d)` from `Units d`; otherwise as the base                                     |
| `log`, `exp`, any function      | the argument is evaluated (so a conflict inside it is found); the result is `Coord`     |

Each whole additive term of an argument is evaluated on its own. A term that is
`Bare` or `Coord` is accepted without comment; a term that is `Units d` is
compared with the argument's conventional dimension, whatever the other terms
are, so `normal(mu = m_h + 1 'days)` on a rate is E316: the literal asserts a
duration where the argument is in θ's units, and `m_h` being a coordinate does
not change what the literal says. Two `Units` terms that disagree are E316
whatever θ is. The conventional dimensions:

| Family           | Argument     | Conventional dimension |
| ---------------- | ------------ | ---------------------- |
| uniform          | lower, upper | dim θ                  |
| normal           | mu, sigma    | dim θ                  |
| truncated normal | mean, sd     | dim θ                  |
| half-normal      | sigma        | dim θ                  |
| log-uniform      | lower, upper | dim θ                  |
| exponential      | rate         | (dim θ)⁻¹              |
| gamma            | shape        | 1                      |
| gamma            | rate         | (dim θ)⁻¹              |
| beta             | alpha, beta  | 1                      |
| log-normal       | mu, sigma    | 1                      |

This table is exactly the condition under which converting a literal gives the
right number: `normal(mu = 0.2 'per_day)` is converted correctly because μ is in
θ's units, and `exponential(rate = 2 'days)` on a rate because that argument is
in θ's inverse units. Where the literal's dimension differs from the convention,
no conversion is right, so the mismatch is E316. Concretely:
`log_normal(mu = -2 'per_day, ...)` is E316, since a rate is passed directly as
a log-coordinate (under weeks it would silently become −14), while
`log_normal(mu = log(0.2 'per_day), ...)` is valid, since the literal is
converted and then its logarithm taken. `normal(mu = 0.2 'days, ...)` on a rate
is E316. `0.5 'ratio` is accepted on a log-normal σ.

θ's dimension is read from the solved model, never written. Where the solved
model leaves θ undetermined, an argument whose convention depends on θ is
converted by its literal's own unit and accepted without a check: a known
mismatch is an error, an unknown dimension is not. Arguments whose convention is
fixed (log-normal, gamma shape, beta) are checked regardless of θ.

**Prior arguments never constrain the model.** Every prior argument is a site in
`Dim_sites`, so the coverage test of §4.2 covers it, but its strength is
`Checked`: it is evaluated after the solve and contributes no equation. A Beta
prior does not make θ dimensionless, and a hierarchical Normal does not equate
the dimension of its sigma with θ's. The same family names in a likelihood
(`normal(mean = projected, sd = ...)`) and in an initial-value law
(`I ~ poisson(rate = I0)`) are model positions and stay hard (§5.1, §5.6): their
arguments are expressions over the model's state.

**One informational lint.** I316 (Info) fires when a log-normal `mu` or `sigma`
is exactly a reference to a parameter whose solved dimension is known and not 1,
and only then:

```
info[I316]: log-normal `mu` of 'r1' is the parameter 'mu_r1', declared `rate`
  = note: `mu` is the mean of ln θ, a log-coordinate in model units; a rate's
    value used there sets the prior's median to exp(mu_r1) per day. If mu_r1 is
    meant as a median rate, write `mu = log(mu_r1)`; if it is meant as a
    log-coordinate, declare it `real dim 1`.
```

It does not fire on `log(m_h)`, on a Beta prior on a rate, on a Normal sigma
taken from a probability, or on any other unconventional choice; one argument
gives at most one I316. The model it was written for is
`docs/dev/proposals/fixtures/garki_post_proposal.camdl`, whose
`mu_r1 : rate ~ half_normal(sigma = 0.1)` supplies the μ of a log-normal prior
on a per-day recovery rate; the half-normal forces μ ≥ 0, which puts that rate's
prior median at 1 per day or more. The fixture is left as it is and gets the
I316. If I316 fires on a model in the corpora where the use is clearly intended,
the lint is removed rather than widened or tuned.

**Hyperparameters need no dimension.** A parameter that appears only in prior
arguments, or nowhere, has no dimension that matters to anything, so I300 does
not report it (§4.4). I300 still reports a parameter that appears in model
dynamics with an undetermined dimension, with the relation the solver did
establish (`dim(kappa) + dim(w) = T⁻¹`).

**Changing `time_unit` does not preserve an arbitrary prior.** A prior written
with unit literals in conventional positions is converted and describes the same
physical belief under either unit (§7.4 tests this). A bare number is a
coordinate and is not converted: `normal(mu = 0.2, sigma = 0.1)` on a rate means
0.2 per day under days and 0.2 per week under weeks. Moving a model to a new
unit means rewriting such priors, and for a log-normal shifting μ by the log of
the conversion factor (+ln 7 from days to weeks; σ is unchanged). The spec
states this (§8, commit 5).

## 6. What is deleted

| Removed (line references in today's `dimcheck.ml`)                         | Replaced by                                                                  |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| union-find `find_root`, `bind`, `unify` (257–316)                          | equalities as equations (§4.3); a fast path only if §6.5 shows one is needed |
| `propagate`, `propagate_product`, `collect_product_factors` (738–847)      | equations solve in both directions                                           |
| `read_dim` and its helpers (903–997)                                       | the solver's `value`                                                         |
| three rounds and the fixpoint (1329, 1640–1650), `rate_cache`, `recording` | one solve                                                                    |
| `permissive_dim`, `contract_arg`, `literal_free`                           | `Equals_t` on one site; `literals` per site                                  |
| the check phase (1660–1730)                                                | conflicts reported by the solver at their sites                              |
| `implied_param_dim`, `params_in`, E303 (1729–1840)                         | contradiction reports, which name both transitions                           |
| `scalar_dim` and `infer_ce` (1083–1205)                                    | derived-quantity and contrast expressions generate the same equations        |
| module `Q` (rationals, 377–399)                                            | integer elimination with checked arithmetic                                  |

The old `dimcheck.ml` was 1,882 lines. The replacement is 1,749 lines:
`dim_sites.ml` 468 (mostly mechanical patterns), `dim_solve.ml` 381 and
`dimcheck.ml` 900, which also holds the prior checks and the diagnostics. The
more meaningful comparison is what a reader must hold: today, the set of checked
positions is implicit in 650 lines of control flow, four mode flags change what
an operator means, and two mechanisms decide each product. After, the positions
are one list, an operator means the same thing everywhere, and there is one
solver.

### 6.4 E303 is retired

E303 ("parameter has conflicting dimensions across transitions") is a
hand-written approximation of what the solver now reports as a contradiction,
with both transitions named. Its golden
(`ocaml/golden/errors/e303_param_inconsistent.camdl`) moves to the new code,
with the message reviewed.

### 6.5 Compile time is a gate, not an afterthought

The generated n-patch model from `952e9cf2` (n² − n coupling transitions) is
measured at n = 30, 60 and 100 against `main`, best of three. The bound is 1.5×
`main` at n = 100. If removing union-find breaks it, a union-find layer for
`x = y` equations returns, inside the solver and invisible to rules.

## 7. Tests

### 7.1 An adversarial corpus, committed

**How `ocaml/golden/` is used today.** Its root holds 62 IR goldens, each a
`.camdl` with its committed `.ir.json`, and each serves several tests at once:
the byte comparison in `ocaml/test/test_compiler.ml` (`test_golden`); the
OCaml-to-Rust run in `tests/test_ocaml_to_rust.sh`, which simulates every golden
with its `*.params.toml`; the schema check in
`rust/crates/ir/tests/schema_validates_goldens.rs`; about 40 Rust source and
test files that load 26 of them as sample models (`sir_basic` 32 times); and
`test_corpus_clean` in `ocaml/test/test_diagnostics.ml`, which compiles every
`.camdl` under `ocaml/golden/` (pruning `errors/` and `data/`),
`tests/fixtures/` and `tests/recovery/` and fails on any Error or Warning
outside `corpus_allowlist`. All 62 pass it on this branch; six emit only I300
Info or allowlisted W105. Every IR golden is therefore already a positive test.
Diagnostic models live in two other conventions: 11 error models in
`ocaml/golden/errors/`, each registered by hand in `ocaml/test/test_dimcheck.ml`
(`negative_golden`, from line 1631) and checking one code; and 20 lint fixtures
in `ocaml/test/lints/`, read by the `# expect:` driver in
`ocaml/test/test_diagnostics.ml`.

**One tree for diagnostic models; IR goldens stay where they are.** The split
that matters is what a test asserts, because that sets its maintenance cost: an
IR golden asserts bytes and is regenerated and reviewed on every IR change; a
diagnostic model asserts only what the compiler reports and changes only when a
verdict does. The layout becomes:

```
ocaml/golden/
  *.camdl, *.ir.json     IR goldens (unchanged, at the root)
  README.md              new: the root holds IR goldens; each subdirectory
                         is named for its role
  data/                  input files (unchanged)
  diagnostics/           models whose expected diagnostics are the claim
    dimcheck/
      interventions/  observations/  priors/  quantities/
      forcings/  literals/  powers/  solver/  transitions/
    lints/               the W and L fixtures of today's ocaml/test/lints/
    semantic/            the other E-code fixtures of both, and their controls
```

The IR goldens are not moved. Moving them would change 147 path references,
mostly in the Rust tests that use goldens as sample models, which is churn
unrelated to dimension checking; and IR goldens also live in `ir/golden/` (the
frozen set, gh#384) and seven `tests/fixtures/*/ir/` directories, so moving one
set would not make IR goldens consistent. Consolidating all eight is a separate
cleanup, and that is when these would move.

Within `diagnostics/`, the directory names the area of the language under test,
and the expected outcome is stated inside the file. A model whose correct
verdict changes (a false error fixed) changes one header line and stays beside
its neighbours; an agent assigned an area owns one directory and sees what it
already covers; valid and invalid models of one rule sit side by side. Errors
and lints are not separate kinds of test: both assert "exactly these
diagnostics", and differ only in severity.

Each file is one minimal model testing one invariant, with three required header
lines:

```camdl
# invariant: a transfer `fraction` is dimensionless
# spec: §5.1, position "transfer fraction"
# expect: E314 Error
time_unit = 'days
...
interventions { campaign : transfer(fraction = gamma, from = S, to = V) at [10] }
```

A valid model is the same with `# expect: (none)`, which is stronger than the
clean-corpus test because it also asserts the absence of I300 Info. The
`# expect:` driver is extended to discover files recursively under
`ocaml/golden/diagnostics/`, to refuse a file missing any of the three header
lines, and to print the invariants per area on request: the list an author, or
an agent, reads before adding a model, so the corpus does not fill with
near-duplicates. `test_corpus_clean` prunes `diagnostics/` (in place of
`errors/`), and the hand-written `negative_golden` registrations are deleted
once their models carry `# expect:` lines. The expectation is the code and
severity, plus the subject (the construct the diagnostic points at), not the
full message, so that wording can improve without churning the corpus.

**The structure is built for a later adversarial round.** After this lands, the
intended next step is to assign agents one area each, with that area's rules
from §5.1 as targets: break a rule, or get a valid model refused. Two rules make
that safe at scale. A finding lands with its fix, never as a failing or ignored
test on `main`; until then it lives in a GitHub issue with the model attached,
reproduced by a run before it is filed. And hand-written models are paired with
property tests (§7.4's unit invariance, and mutations of valid models that must
be refused), which find a different class of defect.

The models from this review go to these area directories by file-name prefix:
`iv_`, `ev_`, `rx_` to `interventions/`; `pr_` to `priors/`; `ob_` to
`observations/`; `q_` to `quantities/`; `sm_` to `forcings/`; `pw_` to
`powers/`; `solve_` to `solver/`; `lit_`, `tc_`, `count_`, `unit_` to
`literals/`; the rest to `transitions/`.

Negative models (each lands with its fix):

| File                         | Invariant                                                                 | Expect                          |
| ---------------------------- | ------------------------------------------------------------------------- | ------------------------------- |
| `iv_fraction_rate`           | transfer fraction is 1                                                    | E314                            |
| `iv_fraction_unit_literal`   | `0.1 'per_day` is not a fraction                                          | E314                            |
| `iv_count_rate`              | transfer count is P                                                       | E314                            |
| `iv_set_rate`                | an assigned compartment value is P                                        | E314                            |
| `ev_add_rate`                | an event's added count is P                                               | E314                            |
| `iv_at_rate`                 | a schedule time is T                                                      | E315                            |
| `rx_threshold_rate`          | a reactive threshold has the stream's dimension                           | E315                            |
| `rx_fraction_rate`           | a reactive transfer fraction is 1                                         | E314                            |
| `pr_hier_wrong_names`        | hierarchical arguments are named as the family's                          | E2xx (the constant path's code) |
| `pr_hier_time`               | a hierarchical argument cannot mention time                               | E2xx                            |
| `pr_hier_negative_sigma`     | a constant argument outside its domain, on the hierarchical path          | E2xx (the constant path's code) |
| `pr_lognormal_mu_unit`       | `mu = -2 'per_day`: a rate passed directly as a log-coordinate            | E316                            |
| `pr_lognormal_sigma_unit`    | `sigma = 0.5 'days`: a duration passed as a log-scale spread              | E316                            |
| `pr_normal_duration_on_rate` | `normal(mu = 0.2 'days, ...)` on a rate                                   | E316                            |
| `pr_literal_sum_conflict`    | `mu = 0.2 'per_day + 1 'days`: explicit units disagree inside an argument | E316                            |
| `ob_normal_mean_rate`        | a Normal mean has the column's dimension                                  | E304                            |
| `ob_normal_sd_rate`          | a count-noise sd has the mean's T exponent                                | E304                            |
| `ob_binomial_p_sum`          | sums inside a strict argument agree                                       | E302                            |
| `ob_projection_sum`          | sums inside a derived projection agree                                    | E302                            |
| `q_threshold_rate`           | a `first_above` threshold has the series dimension                        | E317                            |
| `q_count_above_rate`         | likewise for `count_above`                                                | E317                            |
| `q_series_sum`               | sums inside a quantity series agree                                       | E302                            |
| `q_value_at_count`           | a `value_at` time is T                                                    | E317                            |
| `sm_identity_level`          | an identity-link smooth level has D                                       | E312                            |
| `sm_modes_time`              | smooth modes are 1                                                        | E312                            |
| `od_sigma_dimensionless`     | overdispersion σ² is T                                                    | E308                            |
| `solve_contradiction`        | `kappa*w*S*I/N` with `kappa*w*V*I`                                        | E300                            |
| `solve_not_integral`         | `(kappa*w)^2 * I` in a rate                                               | E300                            |
| `lit_ratio_in_product`       | `(1/7) * I * I` is not a rate                                             | E300                            |
| `tc_unit_conflict`           | `let tau : rate = 30 'days`                                               | E2xx                            |
| `pw_variable_exponent_base`  | `kappa^alpha` makes kappa dimensionless; `kappa` used elsewhere as a rate | E301                            |

Kept as they are (already errors, and correct): `pw_var_exp_count_base` (E301),
`bad_typed_const` (E300), `count_plus_rate` (`gamma * I + 5 'count`, E302), the
B-spline fixture inconsistency and the `'ratio` forcing with a rate baseline
(E312, `smoke/ratio_rate`).

Positive models (each `# expect: (none)`, or only the stated I300):

| File                               | What it protects                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `od_sigma_time`                    | `sigma_se : positive dim T` in `overdispersed`                               |
| `pw_named_integer_exponent`        | `I^k`, `let k : real = 2`                                                    |
| `pw_static_rational_dimensionless` | `(I / N)^0.5`                                                                |
| `count_literal_rate`               | `gamma * 5 'count` is P·T⁻¹                                                  |
| `unit_literal_parenthesized`       | `I / (5 'days)`                                                              |
| `unit_conversion_rate`             | `0.7 'per_week` under days                                                   |
| `density_dependent_typed_const`    | `let b : positive dim P^-1*T^-1 = 0.0001`                                    |
| `density_dependent_inferred`       | `beta : positive`, `beta * S * I`                                            |
| `density_dependent_declared`       | `beta : positive dim P^-1*T^-1`                                              |
| `frequency_and_density_mixed`      | one model with both forms, separate parameters                               |
| `normal_count_noise_he2010`        | `sd = sqrt(rho*(1-rho)*projected + (rho*psi*projected)^2)`                   |
| `normal_count_noise_squared`       | `sd = projected^2`                                                           |
| `normal_real_column`               | a `real` column, mean and sd of one dimension                                |
| `jointly_undetermined`             | only `kappa * w` pinned; I300 Info names the relation, no error              |
| `pr_beta_on_rate`                  | a Beta prior on a rate                                                       |
| `pr_normal_bare_on_rate`           | `normal(mu = 0.2, sigma = 0.1)` on a rate: bare numbers are coordinates      |
| `pr_normal_units_on_rate`          | `normal(mu = 0.2 'per_day, sigma = 0.1 'per_day)`, converted                 |
| `pr_gamma_exponential_on_rate`     | `gamma(shape = 2, rate = 10)` and `exponential(rate = 2 'days)` on rates     |
| `pr_hier_sigma_probability`        | `normal(mu = m, sigma = s)` on a rate, `s : probability`                     |
| `pr_hier_mixed_arithmetic`         | `sigma = s_rate * s_prob + 0.1`: arithmetic over differently declared values |
| `pr_hier_log_of_rate`              | `log_normal(mu = log(m_h), ...)`, `m_h : rate`                               |
| `pr_hier_mu_rate`                  | `log_normal(mu = m_h, ...)`, `m_h : rate`: compiles, `# expect: I316 Info`   |
| `pr_lognormal_hier_dimensionless`  | log-normal `mu`, `sigma` from `real dim 1` hyperparameters                   |
| `pr_lognormal_log_of_unit`         | `mu = log(0.2 'per_day)` under weeks: IR μ = ln 1.4                          |
| `pr_lognormal_ratio_sigma`         | `sigma = 0.5 'ratio`                                                         |
| `pr_units_theta_undetermined`      | `normal(mu = 0.2 'per_day, ...)` on a `positive` θ of undetermined dimension |
| `pr_hyperparameter_no_i300`        | prior-only hyperparameters of undeclared dimension: no I300                  |
| `unchecked_dim_alpha_mixing`       | the He et al. (I + ι)^α term                                                 |
| `importation_constant`             | `beta * S * I / N + 0.5` in a rate                                           |
| `init_literals`                    | `S = N0 - 50`, `I = 10`, a hyper-Erlang split                                |

Every negative model is mutation-checked once in its commit: revert the fix,
keep the model, confirm the test fails.

### 7.2 Solver tests

- Unique solution; underdetermined and consistent (with the relation reported);
  rational contradiction; integer infeasibility (2x + 2y = −T); a contradiction
  among non-unit rows.
- Permutation invariance: a consistent system (right-hand sides computed from a
  planted solution) in shuffled order gives the same determined values, each
  equal to the planted one. Which equation of an inconsistent system is named as
  the culprit depends on order by design, so the conflicting set is not
  compared.
- An oracle: random small systems (4 variables, coefficients in [−3, 3], one
  base), solved by exhaustive search over [−4, 4]⁴. A box search can only
  certify implications, and the test checks each one that holds for every
  system: a solution in the box means the solver reports no conflict; every box
  solution agrees with a value the solver calls determined; two box solutions
  that differ in a variable mean the solver calls it free. It cannot certify
  infeasibility, which the hand-written cases cover.
- Overflow: a constructed system that would overflow native arithmetic raises
  the internal error, not a wrong answer.

### 7.3 Coverage tests

The path-set test of §4.2 over every committed model, and the three mutation
checks recorded in the commit that adds it.

### 7.4 End-to-end

- **Unit invariance.** A pair of models identical except for `time_unit = 'days`
  versus `'weeks`, with every rate, duration and prior argument written with
  unit literals in conventional positions; a log-normal μ is written
  `log(0.2 'per_day)`, which converts itself. Their IRs differ only by the
  conversion, the fixed-θ `pfilter` log-likelihoods agree to round-off once θ is
  mapped between units, and the log-priors differ by exactly the log Jacobian of
  the map (−ln 7 per rate parameter, from days to weeks; +ln 7 per duration).
  The test lists the priors it covers: a prior that is not closed under
  rescaling (Beta on a rate, a bare-number Normal) is correctly not invariant
  and is not in it. This test, and the prior validity tests, are kept apart from
  the solver tests of §7.2.
- **No stationary chains from invalid priors.** `pr_hier_wrong_names` fails
  compilation; a model whose start is −∞ makes PMMH exit non-zero with the
  offending term named.
- **Corpora.** Every tracked model in the repo, `../ebola-bdbv-camdl`, garki and
  the other downstream projects listed in the branch's migration notes: exit
  codes and IR bytes compared with the parent commit, with every change listed
  and explained in the commit that causes it.
- **Compile time.** §6.5.

## 8. Commit sequence

Each commit passes `make test` on its own; old and new checkers never coexist
across a commit boundary that lands.

0. **The diagnostics tree.** `ocaml/golden/diagnostics/` with its header rules
   and recursive driver; `ocaml/golden/README.md`; the 11 error models and the
   20 lint fixtures moved into it with `# expect:` lines; the hand-written
   `negative_golden` registrations deleted. No verdict changes and no IR golden
   moves, so every moved model passes as before.
1. **Positive models and the fixes outside the checker.** The positive corpus
   that passes today; hierarchical prior name, form and time checks (one
   validation function for both branches); the PMMH non-finite start guard;
   `'count` lowering with P; the typed-constant unit conflict; E107 on
   parenthesized literals. Each with its negative model.
2. **The solver.** `dim_solve.ml` with its unit, permutation, oracle and
   overflow tests, unused by the checker yet. It is a self-contained module with
   its own tests, not a second checker.
3. **The replacement.** `dim_sites.ml`, the equation generator and the rewritten
   `dimcheck.ml`, switched over in one commit, with the old machinery deleted in
   the same commit. Every existing golden and error golden keeps its result or
   has its change listed. The coverage test and its three mutation checks.

   As implemented, three points differ from the text above, each because the old
   behaviour has no counterpart in one equation system. An expression's
   dimension is a linear form over dimension variables rather than a fresh
   variable per node, so products, quotients and integer powers need no equation
   at all and only sums, function arguments, roots and obligations emit one (the
   same mathematics as §4.3, with far fewer equations). Sums inside likelihood
   arguments, derived projections and quantity series are strict from this
   commit, since the old permissive and read-only modes cannot be expressed
   without discarding a contradiction; across the repo and downstream corpora no
   verdict changed. And the positions whose rule lands in step 4 are already
   sites, marked `Exempt (Pending _)` with the step named, so the coverage test
   holds from this commit on.
4. **The rules that change verdicts.** One commit per rule family, each with its
   negative and positive models and its corpus measurement: interventions,
   events and reactive triggers; observations (§5.6 and strict sums); priors
   (§5.9: the prior unit evaluation, I316 with the garki fixture as its case,
   the I300 exemption for parameters in no `Binding` site); quantities and
   contrasts; smooth splines; powers; literals (§5.2); E308 (with the he2010
   manifests).
5. **The spec.** `docs/camdl-language-spec.md` §4 rewritten to state the
   positions and their rules, P versus T, literals, unit conversion, the
   count-noise rule, underdetermination and `unchecked_dim`, and what a
   successful compile does and does not guarantee. A short subsection states the
   boundary of §5.9 in the spec's own words: in model dynamics dimensional
   analysis is a type system, in prior specification it is closer to a modelling
   linter; dimensions come from declarations and use, priors are measures on
   numerical coordinates and do not establish or restrict a parameter's
   dimension; bare numbers are model-unit coordinates; unit literals are real
   conversions; the conventional argument dimensions support diagnostics and
   explicit-unit checks only; changing `time_unit` does not preserve an
   arbitrary prior, and the user transforms it (for a log-normal, μ shifts by
   the log of the factor). One paragraph on densities: the measure is
   dimensionless and the density has units of 1/[θ], which is why a family's
   form need not survive a change of unit; the Jeffreys prior for a scale
   parameter, p(θ) ∝ 1/θ, is the form that does (Jeffreys, H. 1946, "An
   invariant form for the prior probability in estimation problems", _Proc. R.
   Soc. Lond. A_ 186:453–461), given as context, not a requirement. Edited by
   hand: this file is never formatted by dprint.

Commits 3 and 4 can be one series on a branch; splitting 4 by family keeps each
verdict change reviewable on its own.

## 9. Decisions

All settled; recorded here as the decision record.

1. **Literal rule.** A literal is dimensionless except zero and a literal that
   is a whole additive term of a dimensioned position (§5.2). `0.0001 * S * I`
   and `(1/10000) * S * I` behave the same, and `(1/7) * I * I` no longer passes
   as a rate. One repo model is rewritten; downstream corpora are measured
   before the rule lands.
2. **`dim` on typed constants.** `let b : positive dim P^-1*T^-1 = 0.0001`, the
   parameter grammar reused.
3. **A unit-literal IR node.** `unit_literal { value, dim, unit }`, in
   `ir/VERSION` 0.43, together with removing the dead `transform` field of an
   estimated parameter.
4. **Order relative to families.** This work lands first.
5. **Scope of the count-noise rule.** Only the sd of a `normal` likelihood on a
   `count` column (§5.6).
6. **Prior arguments are numerical coordinates.** Arithmetic over parameters of
   any declared dimension, `log(m_h)` included, is accepted; only explicit unit
   literals are checked, after the solve, and nothing a prior says enters the
   solve (§5.9).
7. **One prior lint.** I316 (Info) for a log-normal argument that is directly a
   dimensioned parameter; removed, not widened, if it misfires on real models.
8. **I300 exemption.** Parameters in no `Binding` site (prior-only or unused)
   get no I300. Whether I300 becomes an error (size-brackets proposal §5) is
   decided with this distinction, and not by making every undetermined dimension
   fatal.
9. **A unit literal on a θ of undetermined dimension** is converted by its own
   unit and accepted; θ gets no dimension from it.
