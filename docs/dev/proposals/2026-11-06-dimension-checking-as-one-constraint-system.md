# Dimension checking as one constraint system over every model position

Date: 2026-11-06

Status: proposed, for review before implementation

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
projection. Twenty-two minimal models written for this review compile cleanly on
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
   with the position it came from. Each obligation contributes one more
   equation.
3. **One solver.** Exact integer elimination on unit pivots, then a Hermite
   normal form on the small remainder, distinguishes determined,
   underdetermined, contradictory and integer-infeasible systems. Every
   contradiction and every integer infeasibility is a located error.

It also fixes the semantic errors the review found: the overdispersion variance
has dimension T, not 1 (E308 is wrong); `'count` literals lose their P during
lowering; a typed constant silently overrides a unit literal; unit literals on
log-scale prior arguments are converted multiplicatively; hierarchical prior
arguments get no name checks and can produce a chain that never moves and exits
0.

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
| θ under a `beta` prior                   | `w : rate ~ beta(alpha = 2, beta = 2)`                           | θ = 1                                             |
| hierarchical prior spread                | `w : rate ~ normal(mu = m, sigma = s)`, `s : probability`        | T⁻¹                                               |
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
- The IR's `transform` field is always `Identity` (`expander.ml:5714`) and no
  sampler reads it; only run identity hashes it. It is dead, and its removal
  re-keys runs, so it is tracked separately and not part of this work.

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
  and not a `real` column that happens to hold a count-valued mean (§5.6).
  Normal priors and the `normal` initial-value law stay strict.
- _Deleting union-find._ Plain equalities are equations too, so the separate
  mechanism goes; whether a union-find fast path survives inside the solver is
  decided by the compile-time benchmark, not in advance (§6.5).
- _Unit literals as `UncheckedDim`._ Conceptually right: a unit literal is a
  typed constant, not an escape. But the fix is a new IR node, which bumps
  `ir/VERSION` and touches the Rust evaluator. The checker already treats
  `UncheckedDim { inner = Const _ }` as a known dimension, so correctness does
  not depend on it. I propose to make this change inside the IR bump families
  Stage A already requires (decision 3), not as a separate bump.

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
user to fix a problem only unit literals cause. The rule becomes: μ and σ are
dimensionless; a hierarchical μ must be a dimensionless parameter; a unit
literal in either is refused with a hint that names the additive conversion.

**Does the directive go too far anywhere?** In one place it risks it: "every
expression-bearing position" includes the derivative maps autodiff writes into
the IR (`rate_grad`, `diffable.grad`, `ic_grad`, ...). Checking them would be a
test of autodiff, not of the user's model. They are exempt with a stated reason
(§5.1), and the coverage test treats them as exempt paths.

## 4. Design

### 4.1 The public interface does not change shape

```ocaml
(* ocaml/lib/ir/dimcheck.mli *)
type input = {
  model      : Ir.model;
  prior_args : prior_source list;   (* every prior, before the expander folds it *)
}
and prior_source = {
  param  : string;
  family : string;                  (* "normal", "log_normal", ... *)
  args   : (string * Ir.expr) list; (* resolved, unit literals intact *)
}

val check_model : input -> result   (* result type unchanged *)
```

`prior_args` exists because the expander folds a constant prior's arguments to
floats before the IR is built, and the unit literal in `mu = -2 'per_day` is
gone by then. The expander already resolves each argument with `resolve_expr`
before folding; it hands those expressions to the checker instead of discarding
them. Nothing new is serialized, and the IR does not change.

### 4.2 `Dim_sites`: one function names every position and its obligation

```ocaml
(* ocaml/lib/ir/dim_sites.ml *)
[@@@warning "@4@8@9"]
(* 4: a `_` catch-all over a variant is an error, so a new constructor cannot
      fall into an existing arm.
   8: a non-exhaustive match is an error (already `-w @8` in this library).
   9: a record pattern that omits a field is an error, so a new field cannot
      be skipped by not naming it. *)

type key =                       (* things whose dimension other positions refer to *)
  | K_param   of string
  | K_forcing of string          (* the declared unit, D *)
  | K_column  of { stream : string; column : string }
  | K_stream  of string          (* an observation stream's projected dimension *)
  | K_series  of string * (string * string) list   (* a quantity's source series *)
  | K_site    of int             (* another site's expression *)

type obligation =
  | Has      of Dimcheck.dim_vec   (* a fixed dimension: P·T⁻¹, P, 1, T, ... *)
  | Equals   of key                (* the same dimension as [key] *)
  | Equals_t of key                (* the same T exponent as [key]; P unconstrained *)
  | Scaled   of key * Dimcheck.dim_vec   (* dim([key]) · v, e.g. an integral is series · T *)
  | Defines  of key                (* this expression's dimension is [key]'s *)
  | Exempt   of exemption

and exemption =
  | Derivative_map        (* written by autodiff; dim = dim(f)/dim(θ) by construction *)
  | Lineage_weight        (* derived from the rate by the expander *)
  | After_dimcheck        (* per-eval bindings: created by LICM, which runs later *)

type literals =
  | Literals_dimensionless       (* the default (§5.2) *)
  | Whole_terms_take_obligation  (* a literal that is an entire additive term
                                    of this position takes its dimension *)

type site = {
  id         : int;
  subject    : Dimcheck.subject;   (* where diagnostics point *)
  role       : string;             (* "transfer `fraction`", "Binomial `p`", ... *)
  path       : string;             (* IR JSON path, for the coverage test *)
  expr       : Ir.expr;
  obligation : obligation;
  literals   : literals;
  code       : string;             (* the error code a violation reports *)
}

val sites : Dimcheck.input -> site list
```

The top of `sites`, to show what the closed pattern buys:

```ocaml
let sites (input : Dimcheck.input) : site list =
  let { model; prior_args } = input in
  let { name = _; version = _; time_unit = _; description = _; origin = _;
        origin_rata_die = _; compartments = _; doc_index = _;
        model_structure = _; identity_tracked_compartments = _;
        output = _; simulation = _; presets = _;   (* no expressions: numbers
                                                      already in time_unit *)
        constants = _;                             (* declarations, read by
                                                      [Dimcheck] as K_param *)
        per_eval_bindings = _;                     (* After_dimcheck *)
        ic_grad = _;                               (* Derivative_map *)
        transitions; ode_equations; time_functions; tables; interventions;
        observations; parameters; bindings; initial_conditions; balance;
        quantities; contrasts } = model in
  List.concat [
    List.concat_map transition_sites transitions;
    List.map ode_site ode_equations;
    List.concat_map forcing_sites time_functions;
    List.concat_map table_sites tables;
    List.concat_map intervention_sites interventions;
    List.concat_map observation_sites observations;
    List.concat_map binding_site bindings;
    List.concat_map init_sites initial_conditions;
    Option.to_list (Option.map balance_site balance);
    List.concat_map quantity_sites quantities;
    List.concat_map contrast_sites contrasts;
    List.concat_map prior_sites prior_args;
  ]
```

and one nested function, to show the rules reading like the language spec:

```ocaml
let action_sites ~subject (a : action) : site list =
  match a with
  | FractionTransfer { src = _; dst = _; fraction } ->
    [ site ~subject ~role:"transfer `fraction`" fraction
        ~obligation:(Has dimensionless) ~code:"E314" ]
  | AbsoluteTransfer { src = _; dst = _; count } ->
    [ site ~subject ~role:"transfer `count`" count
        ~obligation:(Has population) ~literals:Whole_terms_take_obligation
        ~code:"E314" ]
  | Set { compartment = _; value } ->
    [ site ~subject ~role:"assigned value" value
        ~obligation:(Has population) ~literals:Whole_terms_take_obligation
        ~code:"E314" ]
  | AddAction { add_compartment = _; add_count } ->
    [ site ~subject ~role:"added count" add_count
        ~obligation:(Has population) ~literals:Whole_terms_take_obligation
        ~code:"E314" ]
```

Compare today's version of this function: it does not exist, and nothing fails
to compile when it is missing. That is the change.

**What the compiler guarantees.** Adding a field to `Ir.model`, `transition`,
`action`, a likelihood record or any other record `sites` destructures is a
compile error in `dim_sites.ml` (warning 9). Adding a constructor to `action`,
`likelihood`, `init_spec`, `time_func_kind`, `spline_coefs`, `fire_source`,
`trigger_expr`, `quantity_body`, `temporal_reduce` or any other variant it
matches is a compile error (warnings 4 and 8). The decision is forced; the
author of the new field writes its obligation or an exemption.

**What the compiler cannot guarantee, and the test that does.** A field can
still be bound to `_`. Warning 9 is satisfied by naming it, and the compiler
cannot tell an expression-bearing field from a string. The coverage test closes
this:

- Serialize each corpus model with `Serde` (the IR writer, an independent code
  path that must cover every field for the round-trip tests to pass).
- Collect the JSON path of every expression node.
- Assert that every path equals some site's `path` or lies under an exempt path
  (`*.rate_grad`, `per_eval_bindings`, ...).

This differs from counting visited expressions: a count says how many, not
which, and cannot tell a skipped `fraction` from an extra visit elsewhere. The
path-set comparison names the missing position. The serializer's own
completeness is checked by the existing round-trip tests (parse ∘ serialize is
the identity on every golden); the coverage test relies on that, and says so in
its comment.

The three mutation checks, each run once and recorded in the commit that adds
the mechanism: add a field `extra : expr` to `fraction_transfer` (the build of
`dim_sites.ml` fails); add a constructor `Scale of expr` to `action` (fails);
bind `fraction = _` in `action_sites` (the coverage test fails on
`interventions[0].actions[0].fraction`).

### 4.3 Equations carry their origin

```ocaml
(* ocaml/lib/ir/dim_solve.ml *)
type var = int
type origin = {
  site  : int;                      (* the site whose expression produced it *)
  stage : stage;
  what  : string;                   (* "operands of +", "obligation", "declared kind" *)
}
and stage = Declared | Structure | Obligation

type eqn = {
  terms  : (var * int) list;        (* Σ cᵢ·xᵢ, integer coefficients *)
  rhs    : int array;               (* = an exponent vector, one entry per base *)
  origin : origin;
}
```

Expression inference generates equations and returns the expression's variable.
Today's `Mul` arm:

```ocaml
| Mul ->
  let dl = resolve st dl in
  let dr = resolve st dr in
  (match dl, dr with
   | Any, d | d, Any -> d
   | Known v1, Known v2 -> Known (dim_mul v1 v2)
   | _ ->
     let r = fresh_var st in
     record_equation st r [ (1, 1, dl); (1, 1, dr) ];
     r)
```

has three cases because two mechanisms meet in it: known dimensions are computed
directly, unknown ones become equations, and `record_equation` is silently a
no-op in permissive mode or after round 1. The replacement has one:

```ocaml
| Mul ->
  let l = gen b.left and r = gen b.right and x = fresh () in
  emit { terms = [ (x, 1); (l, -1); (r, -1) ]; rhs = zero; origin = here "product" };
  x
```

A known dimension (a `rate` parameter, a compartment) is a `Declared` equation
on its variable, so the product rule needs no special case. Addition is `x = l`
and `x = r`; `exp` is `l = 0` and `x = 0`; a static power `a^n` is `x = n·a`;
`sqrt a` is `2x = a`. Every rule is one line, and the rule for an operator is
the same at every site.

### 4.4 The solver distinguishes four outcomes

The system decouples by base: an equation over exponent vectors holds separately
for its P part and its T part, so the solver works on one coefficient matrix
with two right-hand sides.

```ocaml
type outcome = {
  value        : var -> Dimcheck.dim_vec option;   (* Some when determined *)
  conflicts    : conflict list;
  undetermined : relation list;   (* among parameters, after intermediates are eliminated *)
}
and conflict = {
  culprit : eqn;                  (* the equation that could not be added *)
  kind    : [ `Contradiction of int array  (* the residual 0 = r, r ≠ 0 *)
            | `Not_integral ];
  context : origin list;          (* the earlier equations it conflicts with *)
}
and relation = { params : (string * int) list; equals : int array }
(* e.g. kappa + w = T⁻¹: "only kappa·w is determined, as T⁻¹" *)

val solve : eqn list -> outcome
```

1. **Order.** Equations are added `Declared`, then `Structure`, then
   `Obligation`. A conflict is therefore found when an obligation is added, and
   reported at that site with that site's code: the transition rate is blamed,
   not the product node inside it.
2. **Exact elimination on unit pivots.** An equation is reduced against the
   accepted rows. If it reduces to 0 = 0 it is redundant. If it reduces to 0 = r
   with r ≠ 0 it is a contradiction: reported, then dropped so it does not
   cascade. Otherwise, if some coefficient is ±1, pivot on it (preferring the
   newest variable, which keeps elimination local, as `952e9cf2` measured). A ±1
   pivot is a unimodular operation, so it preserves the set of integer solutions
   and keeps every coefficient an integer. Almost every equation qualifies:
   products, quotients and sums have only ±1 coefficients.
3. **The residual.** Rows with no ±1 coefficient come only from integer powers
   and square roots. They are set aside and, at the end, put in Hermite normal
   form; the system has an integer solution exactly when back-substitution in
   that form divides evenly (Cohen, _A Course in Computational Algebraic Number
   Theory_, 1993, §2.4; cited from memory). The residual is small (its size is
   the number of non-unit powers in the model), so a dense method is adequate.
   This is what catches `(kappa * w)^2 * I` in a rate: 2·κ + 2·w = −T has
   rational solutions and no integer one.
4. **Arithmetic.** Integers are native, with checked multiplication and
   addition; an overflow is an internal compiler error with a message, never a
   wrapped value. With ±1 pivots the coefficients grow additively, so this is a
   guard, not an expected path. The rational module `Q` is deleted.
5. **Underdetermination.** A variable that remains free is not an error. For
   each undetermined parameter, I300 (Info) reports the relations that _are_
   determined, eliminated down to parameter names: "only `kappa · w` is
   determined: T⁻¹". No dimension is invented.

### 4.5 Diagnostics

A conflict at an obligation reports the site's code, its role, the expected
dimension and the dimension the rest of the system implies (the residual gives
it), and the origins of the equations it conflicts with, as source locations:

```
error[E300]: transition 'infection_v' rate has wrong dimension
  expected P·T⁻¹ (population-level rate), got T⁻¹ (per-capita rate)
  = note: `kappa * w` is T⁻¹ because of transition 'infection' (line 12)
```

A conflict found among `Structure` equations (an inconsistent sum inside an
expression) reports E302 at the site, naming both operands. No minimal
unsatisfiable core is computed: the earlier equations in the same row are the
context, which is deterministic and enough to act on.

## 5. The rules

### 5.1 Every position

Dimensions: P a population count, T time, D the enclosing forcing's declared
unit, θ the parameter a prior is on. "Lit" marks positions where a literal that
is an entire additive term takes the obligation (§5.2).

| Position                                                                                                                                 | Obligation                                    | Code |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ---- |
| transition rate                                                                                                                          | P·T⁻¹ (Lit)                                   | E300 |
| overdispersion σ²                                                                                                                        | T (§5.7)                                      | E308 |
| ODE derivative                                                                                                                           | P·T⁻¹ (Lit)                                   | E306 |
| balance expression                                                                                                                       | P (Lit)                                       | E305 |
| binding body                                                                                                                             | defines the binding's dimension               | E302 |
| forcing `lag`                                                                                                                            | T                                             | E309 |
| sinusoid baseline, amplitude; piecewise, interpolated, periodic values; Fourier coefficients; identity-link explicit spline coefficients | D (constant arguments exempt, as now)         | E312 |
| periods, phases, breakpoints, interpolation times, knots                                                                                 | T                                             | E312 |
| log-link spline coefficients                                                                                                             | 1                                             | E301 |
| smooth `per`                                                                                                                             | T                                             | E312 |
| smooth level, slope, scale, modes: log link                                                                                              | 1, T⁻¹, 1, 1 (§5.8)                           | E312 |
| smooth level, slope, scale, modes: identity link                                                                                         | D, D·T⁻¹, D, 1 (§5.8)                         | E312 |
| inline table values                                                                                                                      | the table's cell dimension                    | E302 |
| table lookup index                                                                                                                       | 1                                             | E311 |
| initial value; Poisson rate; Binomial n; NegBinomial mean                                                                                | P (Lit)                                       | E313 |
| initial Binomial p; NegBinomial dispersion                                                                                               | 1                                             | E313 |
| initial Normal mean, sd                                                                                                                  | P, P (Lit)                                    | E313 |
| transfer fraction                                                                                                                        | 1                                             | E314 |
| transfer count; assigned value; added count                                                                                              | P (Lit)                                       | E314 |
| schedule time expression                                                                                                                 | T                                             | E315 |
| reactive threshold parameter                                                                                                             | the stream's dimension                        | E315 |
| projection `DerivedExpr`                                                                                                                 | defines the stream's dimension                | E302 |
| Poisson rate; NegBinomial and zero-inflated mean                                                                                         | P (Lit)                                       | E304 |
| Binomial and BetaBinomial n                                                                                                              | P (Lit)                                       | E304 |
| Binomial and Bernoulli p; Beta mean; zero-inflation π                                                                                    | 1                                             | E304 |
| dispersions; Beta concentration; BetaBinomial shapes                                                                                     | 1                                             | E304 |
| Normal mean                                                                                                                              | the scored column's dimension (§5.6)          | E304 |
| Normal sd                                                                                                                                | §5.6                                          | E304 |
| prior arguments                                                                                                                          | per family (§5.9)                             | E316 |
| quantity series                                                                                                                          | defines the series dimension                  | E302 |
| quantity thresholds (`count_above`, `first_above`, ...)                                                                                  | the series dimension (Lit)                    | E317 |
| `value_at` time                                                                                                                          | T                                             | E317 |
| derived-quantity arithmetic; contrasts                                                                                                   | operator rules of §4.3                        | E297 |
| derivative maps (`rate_grad`, `rate_state_grad`, `sigma_sq_grad`, `diffable.grad`, `proj_grad`, `ic_grad`, `projection_state_grad`)      | exempt: produced by autodiff                  | —    |
| lineage parent-pool weights                                                                                                              | exempt: derived from the rate by the expander | —    |
| per-eval bindings                                                                                                                        | exempt: created after dimcheck                | —    |

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
- Unit literals are refused on log-scale and dimensionless prior arguments
  (§5.9).

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

Its inner expression is still inferred (so a mistake inside it that is unrelated
to the asserted dimension, such as `exp` of a count, is still found), but the
inner result is not equated with the asserted dimension. Unit literals stop
being represented as `UncheckedDim` when the IR bump of decision 3 lands; until
then the checker treats `UncheckedDim { inner = Const _ }` as a known dimension,
which is what it does now.

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
an error. The exemption does not extend to Normal priors, to the initial
`normal` law, or to a `real` column: there, sd and mean have the same dimension.
Whether some existing model uses a `real` column for count data with a
P-inhomogeneous sd is measured across the corpora before this rule lands; if one
does, the right fix is to declare the column `count`.

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

### 5.9 Priors: one set of rules for constant and hierarchical arguments

Names, required arguments, domains and permitted expression forms are checked by
one function in the expander, applied to both branches of
`classify_and_resolve_prior_spec`. Hierarchical arguments may mention parameters
and constants only: time, `dt`, compartments, projections and observation
columns are refused. Dimensions are checked by the sites of §5.1 from
`prior_args`, with θ of dimension d:

| Family           | Arguments           | θ     |
| ---------------- | ------------------- | ----- |
| uniform          | lower, upper: d     |       |
| normal           | mu, sigma: d        |       |
| truncated normal | mean, sd: d         |       |
| half-normal      | sigma: d            |       |
| exponential      | rate: d⁻¹           |       |
| gamma            | shape: 1; rate: d⁻¹ |       |
| beta             | alpha, beta: 1      | θ = 1 |
| log-uniform      | lower, upper: d     |       |
| log-normal       | mu, sigma: 1        |       |

A unit literal is accepted where the argument has dimension d or d⁻¹, and is
converted as now (`normal(mu = 0.2 'per_day)` under weeks gives 1.4, which is
right). On a dimensionless argument it is refused. On log-normal μ the hint
names the conversion: "μ is the mean of ln θ in model units; under
`time_unit = 'weeks` a median of 0.135 per day is μ = ln(0.135 × 7) = −0.054".

One fixture changes meaning under these rules:
`docs/dev/proposals/fixtures/garki_post_proposal.camdl` declares
`mu_r1 : rate ~ half_normal(sigma = 0.1)` and uses it as the μ of a log-normal
prior on a per-day recovery rate. The half-normal forces μ ≥ 0, so the prior's
median for that rate is at least 1 per day; I read that as unintended. It
becomes E316, and the fixture is corrected to `mu_r1 : real dim 1` with a normal
prior centred on the log of the intended median.

PMMH refuses to start from a non-finite log-posterior, with the parameter whose
prior or likelihood term is non-finite, as PGAS already does
(`sim/src/error.rs:159`). This is a guard in `cli/src/fit/pmmh.rs`, not a change
to any acceptance ratio.

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

The current file is 1,882 lines. I estimate the replacement at roughly 900 lines
across `dim_sites.ml` (mostly mechanical patterns), `dim_solve.ml` and a smaller
`dimcheck.ml`; that is a projection from the parts removed, not a measurement.
The more meaningful comparison is what a reader must hold: today, the set of
checked positions is implicit in 650 lines of control flow, four mode flags
change what an operator means, and two mechanisms decide each product. After,
the positions are one list, an operator means the same thing everywhere, and
there is one solver.

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

A new directory, `ocaml/test/dimcheck/`, scanned by the existing `# expect:`
driver in `ocaml/test/test_diagnostics.ml`. Each file is one minimal model with
one defect, a header stating the invariant it tests, and its expected
diagnostics:

```camdl
# Invariant: a transfer `fraction` is dimensionless (§5.1).
# Violation: the fraction is a per-day rate.
# expect: E314 Error
time_unit = 'days
...
interventions { campaign : transfer(fraction = gamma, from = S, to = V) at [10] }
```

The expectation is the code and severity, plus the subject (the construct the
diagnostic points at), not the full message, so that wording can improve without
churning the corpus. Locations are asserted by the subject.

Negative models (each lands with its fix):

| File                        | Invariant                                                                 | Expect                          |
| --------------------------- | ------------------------------------------------------------------------- | ------------------------------- |
| `iv_fraction_rate`          | transfer fraction is 1                                                    | E314                            |
| `iv_fraction_unit_literal`  | `0.1 'per_day` is not a fraction                                          | E314                            |
| `iv_count_rate`             | transfer count is P                                                       | E314                            |
| `iv_set_rate`               | an assigned compartment value is P                                        | E314                            |
| `ev_add_rate`               | an event's added count is P                                               | E314                            |
| `iv_at_rate`                | a schedule time is T                                                      | E315                            |
| `rx_threshold_rate`         | a reactive threshold has the stream's dimension                           | E315                            |
| `rx_fraction_rate`          | a reactive transfer fraction is 1                                         | E314                            |
| `pr_beta_on_rate`           | θ under a beta prior is 1                                                 | E316                            |
| `pr_hier_sigma_probability` | a normal prior's sigma has θ's dimension                                  | E316                            |
| `pr_hier_mu_rate`           | a hierarchical log-normal μ is dimensionless                              | E316                            |
| `pr_hier_wrong_names`       | hierarchical arguments are named as the family's                          | E2xx (the constant path's code) |
| `pr_hier_time`              | a hierarchical argument cannot mention time                               | E2xx                            |
| `pr_lognormal_mu_unit`      | no unit literal on log-normal μ                                           | E316                            |
| `pr_lognormal_sigma_unit`   | no unit literal on log-normal σ                                           | E316                            |
| `ob_normal_mean_rate`       | a Normal mean has the column's dimension                                  | E304                            |
| `ob_normal_sd_rate`         | a count-noise sd has the mean's T exponent                                | E304                            |
| `ob_binomial_p_sum`         | sums inside a strict argument agree                                       | E302                            |
| `ob_projection_sum`         | sums inside a derived projection agree                                    | E302                            |
| `q_threshold_rate`          | a `first_above` threshold has the series dimension                        | E317                            |
| `q_count_above_rate`        | likewise for `count_above`                                                | E317                            |
| `q_series_sum`              | sums inside a quantity series agree                                       | E302                            |
| `q_value_at_count`          | a `value_at` time is T                                                    | E317                            |
| `sm_identity_level`         | an identity-link smooth level has D                                       | E312                            |
| `sm_modes_time`             | smooth modes are 1                                                        | E312                            |
| `od_sigma_dimensionless`    | overdispersion σ² is T                                                    | E308                            |
| `solve_contradiction`       | `kappa*w*S*I/N` with `kappa*w*V*I`                                        | E300                            |
| `solve_not_integral`        | `(kappa*w)^2 * I` in a rate                                               | E300                            |
| `lit_ratio_in_product`      | `(1/7) * I * I` is not a rate                                             | E300                            |
| `tc_unit_conflict`          | `let tau : rate = 30 'days`                                               | E2xx                            |
| `pw_variable_exponent_base` | `kappa^alpha` makes kappa dimensionless; `kappa` used elsewhere as a rate | E301                            |

Kept as they are (already errors, and correct): `pw_var_exp_count_base` (E301),
`bad_typed_const` (E300), `count_plus_rate` (`gamma * I + 5 'count`, E302), the
B-spline fixture inconsistency and the `'ratio` forcing with a rate baseline
(E312, `smoke/ratio_rate`).

Positive models (each `# expect: (none)`, or only the stated I300):

| File                               | What it protects                                                                 |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| `od_sigma_time`                    | `sigma_se : positive dim T` in `overdispersed`                                   |
| `pw_named_integer_exponent`        | `I^k`, `let k : real = 2`                                                        |
| `pw_static_rational_dimensionless` | `(I / N)^0.5`                                                                    |
| `count_literal_rate`               | `gamma * 5 'count` is P·T⁻¹                                                      |
| `unit_literal_parenthesized`       | `I / (5 'days)`                                                                  |
| `unit_conversion_rate`             | `0.7 'per_week` under days                                                       |
| `density_dependent_typed_const`    | `let b : positive dim P^-1*T^-1 = 0.0001`                                        |
| `density_dependent_inferred`       | `beta : positive`, `beta * S * I`                                                |
| `density_dependent_declared`       | `beta : positive dim P^-1*T^-1`                                                  |
| `frequency_and_density_mixed`      | one model with both forms, separate parameters                                   |
| `normal_count_noise_he2010`        | `sd = sqrt(rho*(1-rho)*projected + (rho*psi*projected)^2)`                       |
| `normal_count_noise_squared`       | `sd = projected^2`                                                               |
| `normal_real_column`               | a `real` column, mean and sd of one dimension                                    |
| `jointly_undetermined`             | only `kappa * w` pinned; I300 Info names the relation, no error                  |
| `hierarchical_prior_valid`         | normal and log-normal leaves under dimensionless and dimensioned hyperparameters |
| `unchecked_dim_alpha_mixing`       | the He et al. (I + ι)^α term                                                     |
| `importation_constant`             | `beta * S * I / N + 0.5` in a rate                                               |
| `init_literals`                    | `S = N0 - 50`, `I = 10`, a hyper-Erlang split                                    |

Every negative model is mutation-checked once in its commit: revert the fix,
keep the model, confirm the test fails.

### 7.2 Solver tests

- Unique solution; underdetermined and consistent (with the relation reported);
  rational contradiction; integer infeasibility (2x + 2y = −T); a contradiction
  among non-unit rows.
- Permutation invariance: the same system in shuffled order gives the same
  determined values and the same set of conflicting equations.
- An oracle: random small systems (up to 6 variables, coefficients in [−3, 3],
  right-hand sides in [−3, 3]²), solved by exhaustive search over exponent
  vectors in [−6, 6]². The oracle can only certify feasibility inside its box,
  so generated systems are restricted to those whose solutions, when they exist,
  lie in it (generated from a planted solution, plus planted infeasible ones).
- Overflow: a constructed system that would overflow native arithmetic raises
  the internal error, not a wrong answer.

### 7.3 Coverage tests

The path-set test of §4.2 over every committed model, and the three mutation
checks recorded in the commit that adds it.

### 7.4 End-to-end

- **Unit invariance.** A pair of models identical except for `time_unit = 'days`
  versus `'weeks`, with every rate, duration and prior argument written with
  unit literals (log-normal μ written per §5.9 with the additive conversion):
  their IRs differ only by the conversion, and a fixed-θ `pfilter`
  log-likelihood plus log-prior agree to round-off once θ is mapped between
  units.
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

1. **Positive models and the fixes outside the checker.** The positive corpus
   that passes today; hierarchical prior name, form and time checks (one
   validation function for both branches); the PMMH non-finite start guard;
   `'count` lowering with P; the typed-constant unit conflict; E107 on
   parenthesized literals; unit literals refused on dimensionless and log-scale
   prior arguments (applied in the expander, where the literal is still
   visible). Each with its negative model.
2. **The solver.** `dim_solve.ml` with its unit, permutation, oracle and
   overflow tests, unused by the checker yet. It is a self-contained module with
   its own tests, not a second checker.
3. **The replacement.** `dim_sites.ml`, the equation generator and the rewritten
   `dimcheck.ml`, switched over in one commit, with the old machinery deleted in
   the same commit. Every existing golden and error golden keeps its result or
   has its change listed. The coverage test and its three mutation checks.
4. **The rules that change verdicts.** One commit per rule family, each with its
   negative and positive models and its corpus measurement: interventions,
   events and reactive triggers; observations (§5.6 and strict sums); priors
   (§5.9, including the garki fixture); quantities and contrasts; smooth
   splines; powers; literals (§5.2); E308 (with the he2010 manifests).
5. **The spec.** `docs/camdl-language-spec.md` §4 rewritten to state the
   positions and their rules, P versus T, literals, unit conversion, prior
   argument dimensions and the log-normal convention, the count-noise rule,
   underdetermination and `unchecked_dim`, and what a successful compile does
   and does not guarantee. Edited by hand: this file is never formatted by
   dprint.

Commits 3 and 4 can be one series on a branch; splitting 4 by family keeps each
verdict change reviewable on its own.

## 9. Decisions

1. **Literal rule.** A literal is dimensionless except zero and a literal that
   is a whole additive term of a dimensioned position (§5.2). The alternative is
   today's rule, where a ratio of literals takes any dimension. The new rule
   makes `0.0001 * S * I` and `(1/10000) * S * I` behave the same (both refused,
   with a fix-it), stops `(1/7) * I * I` passing as a rate, and keeps every
   initial-condition and importation idiom I found. It costs one repo model a
   rewrite (`(1.0 / 730.0) * R[p]`) and an unknown number downstream, to be
   measured before it lands. Recommendation: adopt it. Leaning.
2. **`dim` on typed constants.** `let b : positive dim P^-1*T^-1 = 0.0001`. It
   is the only way to write a literal coefficient with a compound dimension once
   ratios stop being contextual, and it reuses the parameter syntax this branch
   introduced rather than adding a unit vocabulary for compound units. The cost
   is a grammar production and a tree-sitter regeneration. Recommendation:
   adopt. Solid.
3. **A unit-literal IR node.** Replace `UncheckedDim { inner = Const }` for unit
   literals with a dedicated node, so `UncheckedDim` means only a user's escape.
   It changes `ir/VERSION`, every golden containing a unit literal, and the Rust
   evaluator, and correctness does not depend on it. Recommendation: do it
   inside the IR bump families Stage A requires, not separately. Leaning.
4. **Order relative to families.** Land this before families Stage A, so that
   families' new positions are added to `Dim_sites` from the start rather than
   retrofitted. The cost is that families waits for this work. Recommendation:
   this first. Leaning.
5. **Scope of the count-noise rule.** Only the sd of a `normal` likelihood on a
   `count` column (§5.6). The wider alternative, any Normal sd whose mean has a
   P component, would admit `real` columns holding counts but would also exempt
   continuous data with a population dimension (a person-time density, say) from
   a check that is right for it. Recommendation: the narrow scope, with the
   corpus measurement deciding whether any model needs `count` declared.
   Leaning.
