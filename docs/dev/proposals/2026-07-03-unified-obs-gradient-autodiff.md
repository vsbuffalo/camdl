---
status: proposal
date: 2026-07-03
area: inference (PGAS+NUTS gradient), IR schema, OCaml autodiff
implements: gh#180 (obs parametric-DerivedExpr projection ∂projected/∂θ + the C1 fence)
parent: gh#76 (obs-density gradient — closed; gh#180 is its residual)
follow_up: gh#342 (3b — derive the traversal + unify the rate path, §11)
related: docs/dev/proposals/2026-06-26-ode-nuts-gradient-spine.md
ir_version: 0.23 → 0.24
---

# Unified observation-gradient autodiff — one differentiation authority

## 1. Problem

PGAS+NUTS needs `∂ℓ(θ)/∂θ` for the complete-data log-likelihood. Two of its
terms — the observation density and the σ² overdispersion density — compute the
θ-derivative of their **argument expressions** with a _runtime_ forward-mode
differentiator, `eval_resolved_deriv` (`resolved_expr.rs:867`). That
differentiator returns **0** for `TimeFunc`/`TableLookup`
(`resolved_expr.rs:880-881`) and has a `_ => 0.0` catch-all — so a parameter
reaching an observation through a forcing, or through any future `Expr` variant,
gets a silently-zero gradient.

The transition (rate) term does not have this problem: it rides the
**compiler-emitted** symbolic gradient `rate_grad`, and the OCaml autodiff
(`autodiff.ml`) differentiates forcings analytically for the kinds that admit it
(§3). The autodiff `d` match is **exhaustive and wildcard-free** over `Expr` and
its operator sub-ADTs (`autodiff.ml:143-326`), returning a two-case ADT
`deriv =
Known of expr | Unsupported of {node; reason}` — so a new `Expr`
variant is a compile error until classified, and a non-differentiable node is a
loud `Unsupported{reason}`, never a silent zero. The observation path forfeits
this seal by using the Rust runtime differentiator.

Two gaps fall out of the one root cause:

1. **Parametric `DerivedExpr` projection** (the live bug). A projection like
   `qgam * prevalence` makes `projected` a function of θ; the chain-rule term
   `∂L/∂(projected)·∂(projected)/∂θ` is dropped. `run_pgas` refuses such a fit
   with a hard error (`pgas.rs:2285-2317`) rather than bias the posterior. A
   user hit this estimating `qgam`.
2. **Forcing coefficient inside any observation expression** (projection _or_
   likelihood argument). A sinusoidal amplitude in
   `rate = seasonal(t)·projected` or a `DerivedExpr` projection is
   differentiable in a rate, zero in an observation.

Both are "the compiler already knows how to differentiate this expression; the
observation path just isn't asking it to." **This proposal routes the
observation and σ² gradients through the compiler autodiff and carries its
`Known | Unsupported{reason}` classification all the way to the fit-time gate,
so the seal that protects rates protects observations too.**

## 2. Background: why there are two differentiation paths

Load-bearing for the decision — the fork is _debt_, not a needed design, and the
debt was consciously recorded.

- `2026-04-06` `f70dd291` — compiler symbolic autodiff + `rate_grad` in the IR.
- `2026-04-08` `edb91d84` — `resolved_expr.rs` lands with `eval_resolved_deriv`,
  "infallible forward-mode AD on resolved trees." General-purpose; no callers
  yet. Its unresolved-`Expr` twin `eval_expr_deriv` (`propensity.rs:317`) dates
  here too.
- `2026-05-25` `cb46b408` / `00b1a2df` — the σ² (gh#20) and observation (gh#76)
  gradient terms are wired, reusing `eval_resolved_deriv`. Locally reasonable:
  the compiler could not differentiate forcings on _any_ path until gh#119, two
  weeks later.
- `2026-06-09` `1e6a55a6` — gh#119 teaches the compiler to differentiate
  forcing/table coefficients. Its proposal (line 326) **explicitly posed** _"fix
  `eval_expr_deriv`'s `TimeFunc→0` (secondary path) or document it value-only"_
  — and chose **document**, producing the "stay at 0 on purpose" comment now in
  `resolved_expr.rs`.

A recorded, gated, conscious deferral — protected throughout by the C1
preflight. This proposal retires it.

## 3. The three-tier forcing boundary — what "everywhere" can mean

`autodiff.ml:167-208` classifies every forcing/table derivative; any unification
inherits this boundary, corrected against code:

| Tier                      | Kinds                                                                           | `autodiff.ml` today                                             | Estimable via NUTS  |
| ------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------- |
| **1 — differentiable**    | Sinusoidal, Fourier; inline param-table cell (constant index)                   | analytic derivative                                             | **yes (the fix)**   |
| **2a — genuine zero**     | param does not drive the coefficient                                            | `Known (Const 0.0)`                                             | yes (deriv truly 0) |
| **2b — live-but-omitted** | Periodic step/period, `lag`, inline-table value via non-const index             | `Known (Const 0.0)` **but coefficient is live** (gh#215/gh#314) | **no — refused**    |
| **3 — structural**        | Piecewise / Interpolated / PeriodicSpline coefficient; non-constant table index | `Unsupported`                                                   | **no — refused**    |

Two corrections this encodes: **Periodic/`lag` are refused, not "genuinely 0"**
(their coefficients are live; the derivative is deliberately un-emitted), and
**tier 3 is refused everywhere** (a spline knot is precomputed at construction —
not a live coefficient). "Everywhere" tops out at tiers 1–2a. See §4.2 for how
the `deriv` ADT is refined so tier-2b stops masquerading as tier-2a.

## 4. Design (types first) — Level 3a

The observation log-likelihood for one stream is
`L = logpmf(y; args(projected,
θ))`, `projected = g(X, θ)` for a `DerivedExpr`
projection and `g(X)` otherwise. Given fixed trajectory `X` (the PGAS θ|X step):

```
∂L/∂θ = Σ_arg  (∂logpmf/∂arg)  ·  (∂arg/∂θ)         [total derivative, projection inlined into arg]
```

- **`∂logpmf/∂arg`** — the per-distribution factor (`negbin_logpmf_grad`, …,
  `obs_loglik.rs`). Needs the observed `y` + evaluated arg — **inherently
  runtime; stays**. The irreducible runtime factor, not the workaround.
  (Discretized-Normal: the `sd` arg pairs with `d_var·2·sd`; the `2·sd` Jacobian
  stays in this factor — do not fold it into `sd_grad`.)
- **`∂arg/∂θ`** — the θ-derivative of the argument _expression_. Moves to the
  compiler.

The decomposition is verified complete for all six likelihood arms (cross-terms,
`projected` in multiple args, aux constancy, multi-stream holes) — §8 pins it.

### 4.1 The `Diffable` seam and the `DerivEntry` ADT

The load-bearing type decision: **carry the compiler's `Known | Unsupported`
classification across the IR** instead of collapsing it to a bare
`HashMap<param, Expr>` (which conflates "absent" and "undifferentiable" and
forces the fit gate to re-triangulate coverage — the mechanism the review found
three bugs in).

```rust
// rust/crates/ir/src/expr.rs (or a new deriv.rs) — mirrors OCaml `deriv`
enum DerivEntry {
    Grad(Expr),                          // a real ∂arg/∂param
    Unsupported { node: String, code: UnsupportedReason },  // loud; stable code, not free text
}
// UnsupportedReason is an enum (Lag, PeriodicCoeff, StructuralForcing, NonConstTableIndex,
// Mod, ParametricN, …); the human message is DERIVED from it at display time.
```

**run_id must hash the stable `code`, not a free-text string.** Because obs
tier-2b/3 is _omit-and-refuse_ (not compile-error), the `Unsupported` entry is
**serialized into the IR** and hashed into run_id identity (`rate_grad`, by
contrast, never serializes a string — a rate `Unsupported` is an E600 at compile
time). A free-text `reason` would re-key every affected golden on a copy-edit,
so the entry carries an enum `code` (hashed) and the message is derived for
display.

Every _differentiable position_ is a `Diffable` — the (expression, its per-param
gradient) pair travelling as one value, so a derivative can never be written
without (a slot for) its expression, and the differentiation pass visits each
one uniformly:

```rust
struct Diffable {
    expr: Expr,
    grad: HashMap<String, DerivEntry>,   // param → entry; absent key = genuine 0
}
```

`Diffable` is the abstraction for **every** obs likelihood argument (`mean`,
`sd`, `rate`, `p`, `alpha`, `beta`) and the σ² expression. `n` is **not** a
`Diffable` (§4.5). To keep rate goldens byte-stable, the _serialized_ form of a
`Diffable` is the existing adjacent-field convention (`mean` + `mean_grad`,
mirroring `rate` + `rate_grad`), realized via `#[serde(flatten)]`/custom serde;
`Diffable` is the in-memory + resolved-runtime shape. **The rate path keeps its
current `rate_grad: HashMap<String, Expr>` unchanged** in 3a (rates hard-fail on
`Unsupported`, so they never carry one — §4.4); folding rate into the literal
`Diffable`/`DerivEntry` type is 3b (§11).

Exhaustive `match Likelihood` (no `_`) in the pass → a new likelihood variant is
a **compile error** until each of its args is wired as a `Diffable`.

### 4.2 Refine `deriv` so tier-2b stops masquerading as a genuine zero

`autodiff.ml` returns `Known (Const 0.0)` for a **live-but-omitted** coefficient
(Periodic step/period `:171`, inline-table value via non-const index `:220`). On
the rate path that is harmless (a separate `coeff_guard` set refuses those). But
the obs driver would read `Known (Const 0.0)` as _"differentiable, derivative
zero"_ and **admit a silent-zero NUTS fit**. Fix at the root: refine `deriv`
from two cases to three so the two "can't" reasons are distinct from a genuine
zero:

```ocaml
type deriv =
  | Known of expr                              (* real derivative (incl. genuine Const 0.0) *)
  | Omitted of { node : string; reason : string }      (* live coefficient, derivative not emitted (tier 2b) *)
  | Unsupported of { node : string; reason : string }  (* structural, param cannot drive it (tier 3) *)
```

The two `Known (Const 0.0)`-for-a-live-coefficient sites (`:171`, `:220`) become
`Omitted{reason}`. **Plus a third conversion that is not a rewrite of an
existing site — the `lag` guard.** `autodiff.ml` never inspects `tf.lag`
(gh#314): the closed forms differentiate against bare `Time`, not `Time − lag`
(`autodiff.ml:100,116`), so a param driving _only_ a forcing's evaluation-time
shift produces `Known (Const 0.0)` as a **genuine zero** — which the obs driver
would admit as `DerivEntry::Grad(0)` (a silent bias; the true
`∂forcing/∂lag ≠ 0`). Today only `coeff_guard`'s global `periodic_coeff` scan
catches this (`coeff_guard.rs:202-216`). So the `TimeFunc` arm must gain an
explicit guard — `param ∈ tf.lag ⇒ Omitted{reason=lag}`, **before** the
closed-form/`:184`/`:171` branches — classifying `lag` correctly at the source
on every surface (no wrong `Grad(0)` rescued by a separate backstop). This is
the one place the reviewers found the seal leaking; it is a new guard, not part
of the "two sites."

**Rate behaviour is unchanged**: the rate driver maps `Known→emit`,
`Omitted→drop-and-compile` (today's behaviour — a `lag`-in-rate param stays
`coeff_guard`'s concern), `Unsupported→E600`. The obs/σ² driver maps
`Known→DerivEntry::Grad`, and **both** `Omitted` and
`Unsupported →
DerivEntry::Unsupported{reason}` (for an observation, both mean
"refuse NUTS with this reason"). The differentiation logic stays one exhaustive
function; the compile-vs-defer policy is a thin per-driver adapter (the "natural
seam").

### 4.3 Rust evaluation — one shared seam

`eval_likelihood_resolved_grad` (`obs_model.rs:159`) stops calling
`eval_resolved_deriv` and evaluates each `DerivEntry::Grad` with the value
evaluator `eval_resolved` — how `pgas_grad.rs` already eats `rate_grad`. Per
arg, per estimated param `i`:
`grad[i] += (∂logpmf/∂arg) · eval_resolved(grad[i], ctx)`. A
`DerivEntry::Unsupported` never reaches runtime — the preflight (§4.4) refused
first. The **9** obs `eval_resolved_deriv` call sites
(`obs_model.rs:184,185,200,
201,209,224,245,246,259`) are deleted.

**Expose this as a shared seam** (an `obs_arg_grad(...)` on the resolved obs
model) so the in-flight ODE-NUTS `det_grad` reuses it rather than re-deriving —
otherwise this trades one fork for another (§10).

### 4.4 The fit-time gate becomes a consumer of `DerivEntry`, not a set-triangulator

The single invariant:

> A NUTS fit runs only if every estimated parameter that reaches an observation
> — through a projection or any likelihood argument, after projection inlining —
> is covered by a `DerivEntry::Grad`. Any estimated parameter with a
> `DerivEntry::Unsupported` in a `Diffable` it reaches is **refused, at the
> `run_pgas` boundary, with that entry's `reason`.**

This _deletes_ the bespoke `pgas.rs` gate **and** the observation half of
`coeff_guard`'s 3-set triangulation (`coeff ∧ ¬body ∧ ¬has_grad`,
`coeff_guard.rs:160-237`). The `DerivEntry::Unsupported` records _are_ the
coverage-and-reason ledger, so the three review-found bugs become
**unrepresentable** rather than patched:

- _spurious refusal_ (has_grad was `rate_grad`-only) — gone: a tier-1
  obs-forcing param carries `DerivEntry::Grad`, so it is admitted by
  construction.
- _silent-zero via projection / table-index_ (coeff_guard never scanned
  projections) — gone: the projection is inlined and differentiated, so its
  params get `Grad` or `Unsupported` entries directly.
- _tier-2b masquerade_ — gone via §4.2.

Two explicit rules the ledger does **not** subsume, stated so nothing is silent:

- **`n` (Binomial/BetaBinomial) must be θ-independent.** `n` is rounded to an
  integer (`obs_model.rs:219,241`) — genuinely non-differentiable. Refuse any
  estimated param reaching `n` (after inlining) with a located message ("`n`
  must be θ-independent — a constant or an observed data column"). Dropping the
  bespoke gate without this turns the current _refusal_ of `n = projected` into
  a **silent bias** (a regression) — and a param directly in `n` is a
  pre-existing silent hole this closes.
- **Keep the gate at the `run_pgas` boundary** (sim crate), invoked for every
  caller — not only the one CLI `if use_nuts` site (`fit/pgas.rs:177`). Direct
  `run_pgas` callers (tests, API, ODE-NUTS) must be protected at the layer that
  produces the gradient.

**`coeff_guard` disposition (must be explicit — it is not "unchanged").**
`coeff_guard` runs at the CLI layer (`fit/pgas.rs:177`) and its `coeff` /
`periodic_coeff` sets scan **all** `model.time_functions`/`tables` _globally_,
with no rate-vs-obs seam (`coeff_guard.rs:174-234`). Left as-is it would still
refuse the exact tier-1 obs-forcing param the fix enables (it is in a global
forcing, not in `body`, not in `rate_grad`). So 3a makes two changes and asserts
one partition:

- **Union obs `DerivEntry::Grad` keys into `has_grad`**
  (`coeff_guard.rs:220-225`) so a tier-1 obs-forcing param with a real emitted
  obs gradient is admitted.
- **Scope the forcing/table refusal to rate/IC-referenced coefficients** — a
  param whose only forcing/table reference is through an observation is the obs
  preflight's domain (refused there with its specific `reason`), not
  double-refused by `coeff_guard`'s global scan. The exact obs-only-vs-shared
  scoping is a P5 deliverable.
- **The partition invariant**: every forcing/table coefficient param is
  classified **once** — by `coeff_guard` if referenced by a rate/IC, by the obs
  preflight if observation-only — and both verdicts derive from the _same_
  autodiff `Grad|Omitted|Unsupported` classification (the `lag` guard of §4.2 is
  what makes `lag` part of it). Neither path can silently admit. A
  `lag`-in-observation test (§8) pins the case both quick-reviews flagged.

Full unification (rate onto `DerivEntry`, retiring `coeff_guard` into the one
preflight) is 3b (§11).

### 4.5 Retire the runtime differentiators; do not touch LICM

`eval_resolved_deriv` has exactly two consumer sites (9 obs + 1 σ²,
`pgas_grad.rs:393`); both move, so it is deleted. Its twin **`eval_expr_deriv`
(`propensity.rs:317`)** is a functionally identical runtime forward-mode
differentiator with **zero production callers**, kept alive only by the
equivalence test `tests/resolved_expr.rs:328-340` (also `eval_resolved_deriv`'s
only test caller). **Delete all three together** — only then is "one
differentiation authority (`autodiff.ml`), zero runtime forward-mode AD" true;
Rust is left with only the irreducible `∂logpmf/∂arg` helpers and pure
_evaluation_ (`eval_resolved`, exhaustive over `ResolvedExpr`).

**Do not extend LICM.** `licm.ml:14-18` documents a hard contract: it hoists
only the three dynamics surfaces and deliberately **excludes** observation/
overdispersion expressions (their consumers must never see a `PerEvalRef`).
Observations are evaluated per-observation (sparse) — hoisting buys nothing.
Keep them PerEvalRef-free. `constant_fold` _does_ fold the new grads (add an obs
arm; it early-returns unless the model has inline tables — cheap).

### 4.6 Pin the seal in the build

The "new `Expr` variant ⇒ compile error" guarantee currently rides dune's
default dev profile (warning 8 fatal); the `ir`/compiler `dune` files carry no
explicit `(flags ...)`. Pin it: add `(flags (:standard -w @8))` (partial-match
fatal) to the OCaml libraries that own differentiation, so the exhaustiveness
seal is explicit and robust to a profile change — cheap insurance for a
load-bearing invariant.

## 5. User-facing surface

- **Fixes the bug**: `[estimate.qgam]` on a parametric `DerivedExpr` projection
  works; a param scaling a projection, or a Sinusoidal/Fourier amplitude in a
  projection or likelihood argument, gets a correct NUTS gradient.
- **No DSL change.** No new syntax/keyword; existing `.camdl` files unchanged.
- **Better refusals**: a tier-2b/tier-3 coefficient or a parametric `n` reaching
  an observation is refused with the **compiler's own `reason` string**, carried
  end-to-end — not a generic gate message. The bespoke "does not cover
  parametric DerivedExpr…" error disappears.
- **IR format bump** `ir/VERSION 0.23 → 0.24`. Alpha: no compat shim; recompile.
  The obs `*_grad`/`sigma_sq_grad` (now `DerivEntry` values) are **hashed into
  run_id identity** (like `rate_grad`, which `normalize_for_hash` does _not_
  strip — `resolve.rs:89-94`, `ir_hash.rs:260`); a deliberate re-key coincident
  with the VERSION bump. **Rate goldens are byte-stable** (§4.1).
- Not a DSL breaking change → no `docs/language-changes.md` entry;
  `docs/user-features.md`'s obs-gradient note is updated. `ir/schema.json` edit
  is optional for validity (`rate_grad` is absent from the schema and validates
  as an extra); if touched, fix the pre-existing
  `bernoulli`-missing-from-`oneOf` drift (`schema.json:706-779`).

## 6. Implementation — six independently-landable phases

Each phase compiles + is `make test`-green on its own (a subagent can knock out
one, I review the full diff + run `make test` before it lands). Ordered so risk
rises as the foundation sets.

- **P1 — `deriv` 3-way refinement + `lag` guard (OCaml only, foundational).**
  Add `Omitted` to `autodiff.ml`'s `deriv`; convert the two live-coefficient
  `Known (Const 0.0)` sites (`:171,:220`); add the **`param ∈ tf.lag ⇒ Omitted`
  guard** in the `TimeFunc` arm (§4.2 — the seal-leak both quick-reviews found);
  add the rate-driver adapter (`Known→emit`, `Omitted→drop`, `Unsupported→E600`)
  preserving current behaviour. **Rate goldens unchanged**; existing autodiff
  tests green. Gate: `make test-ocaml`.
- **P2 — IR schema: `DerivEntry` + obs/σ² grad surfaces.**
  `ir/VERSION 0.23→0.24`; `DerivEntry` type (Rust `ir` + OCaml `ir.ml`) with
  `serde.ml` round-trip and an enum `UnsupportedReason` **code** (hashed, not
  free text — §4.1); obs likelihood `*_grad` fields + the `Overdispersed` grad
  surface; `ir_hash` `Likelihood`/`DrawMethod` arms; `ResolvedLikelihood` grad
  structure; the ~56 construction-site edits. Fields land **present-but-empty**
  here (P3 populates; safe because the runtime consumer is still
  `eval_resolved_deriv` until P4). Golden regen for obs + overdispersion (≈15–20
  files, human-reviewed) — note the **double regen** (empty here, populated in
  P3; two benign reviewed diffs). Gate: `make test` + golden review.
- **P3 — OCaml obs/σ² autodiff driver.** Exhaustive `match Likelihood`;
  differentiate each arg (projection inlined) and the σ² expr; the obs-driver
  adapter (`Known→Grad`, `Omitted|Unsupported→DerivEntry::Unsupported`);
  `constant_fold` obs arm; `n`-validation emission. Gate: `make test-ocaml` +
  the golden `*_grad` content review.
- **P4 — Rust evaluation (inference math — highest risk).** Evaluate emitted
  obs/ σ² grads via the shared seam (§4.3); move the σ² term; **delete**
  `eval_resolved_deriv`, `eval_expr_deriv`, the equivalence test. Gate:
  `make test-inference` + the FD matrix (§8). _I review this phase's diff line
  by line._
- **P5 — Fit-time gate (under-rated risk — can go green while silently wrong).**
  The `DerivEntry::Unsupported` preflight at the `run_pgas` boundary; delete the
  bespoke `pgas.rs` gate; the `coeff_guard` partition per §4.4 (union obs `Grad`
  into `has_grad`; scope its forcing/table refusal to rate/IC-referenced
  coefficients); `n`-refusal; invert `pgas_gate_betabinomial.rs:311-332`. The
  `lag`-in-obs and tier-1-obs-forcing-acceptance tests (§8) are **required in
  this phase** — a green suite without them would hide the exact seam both
  reviews flagged. Gate: `make test-inference` + those tests.
- **P6 — Seal + full matrix.** Dune `-w @8` pin (§4.6); the full §8 test matrix;
  `docs/user-features.md` update; `make test` all phases.

## 7. Size and why

**~10–15 focused days** (Level 3a). Over the plain-unification (~6–10 days) the
`DerivEntry`/`Diffable`/`deriv`-refinement + exhaustive-`Likelihood` +
gate-rewrite add ~4–6 days, and they are the difference between _fixing this
gap_ and _making this class of gap unrepresentable_. The differentiation _logic_
remains reuse (`autodiff.ml` already handles forcings; projection-inlining is a
trivial AST substitution); the cost is the schema dance (P2), the σ² second
surface, and the FD test matrix (the real time sink, and the point). Rejected:
the ~2-day "port the forcing closed-forms into Rust `eval_resolved_deriv`"
shortcut forks the derivative math — the opposite of the goal.

## 8. Test plan (TDD)

- **Red-first FD** (`gradient_check_obs.rs`, ≤1e-4 rel):
  parametric-`DerivedExpr` projection (`qgam·prevalence`); Sinusoidal amplitude
  (a) in a projection and (b) in a likelihood argument — zero today, FD-matching
  after.
- **Acceptance (headline)**: a Sinusoidal amplitude used _only_ in an
  observation is **admitted** and NUTS-estimated (pins the spurious-refusal
  fix).
- **`n`-gating**: a parametric Binomial `n` (and `n = projected`) is **refused**
  with the located message.
- **Compile-success**: a spline-coefficient-in-observation model still
  **compiles** and runs forward-sim/IF2/PF (pins §4.2 Omitted→refuse, not E600).
- **tier-2b**: a Periodic coefficient in an observation is **refused** (pins the
  masquerade fix), and its reason surfaces.
- **`lag`-in-observation** (both quick-reviews' finding): a param driving a
  forcing's `lag` and reaching an observation is **refused**, not admitted as a
  `Grad(0)`. Pins the `lag` guard (§4.2); the current test suite has _no_
  lag-in-observation coverage, so this is net-new.
- **Regression/safety**: existing `FlowSum`/`IntCompSum` obs grad checks match —
  **`==` for finite / ≤ a few ULP for compound args** (not `to_bits()`;
  `simplify`'s `Neg 0.0` and dropped-`x*0` make strict byte-identity
  falsifiable).
- **σ² regression**: existing gh#20 gamma FD checks unchanged after the σ² move.
- **Seal test**: a deliberately-unhandled synthetic `Expr`/`Likelihood` variant
  fails to compile (documents the seal; may be a `dune`-level negative test).

## 9. Decisions (resolved — no open questions)

- **Carry the `DerivEntry` ADT across the IR** (§4.1) — reasons ride to
  fit-time; the gate consumes them; the 3-set triangulation is deleted.
- **Refine `deriv` to 3 cases** (§4.2) so tier-2b ≠ genuine zero; rate behaviour
  unchanged.
- **Add the `lag` guard** (§4.2) — `param ∈ tf.lag ⇒ Omitted`; the one seal-leak
  both quick-reviews found (`autodiff.ml` never inspects `tf.lag`).
- **`coeff_guard` partition, not "unchanged"** (§4.4): union obs `Grad` into
  `has_grad`; scope forcing/table refusal to rate/IC-referenced coefficients;
  obs is the preflight's domain. Every coefficient param classified exactly
  once.
- **`Unsupported` carries a hashed `code`, not free text** (§4.1) — a reason
  copy-edit must not re-key goldens.
- **`Diffable` seam for obs/σ²; rate keeps `rate_grad: HashMap<_, Expr>`** in 3a
  (rate goldens stable); literal-type + rate unification is **3b** (§11).
- **Obs tier-3/2b ⇒ omit-and-refuse (not E600)**; forward-sim/IF2/PF preserved.
- **Gate `n`** rather than emit `n_grad` — `n` is rounding-discontinuous.
- **Delete both runtime differentiators** (`eval_resolved_deriv` +
  `eval_expr_deriv`) + the equivalence test.
- **Do not extend LICM** (deliberately-excluded surface).
- **Pin `-w @8`** so the exhaustiveness seal is explicit (§4.6).
- **run_id: obs grads hashed** (mirror `rate_grad`); the VERSION bump is the
  re-key. **Regression tolerance is ULP/FD, not byte-identity.**
- **This proposal implements gh#180** (obs parametric-`DerivedExpr` projection);
  gh#76 (the parent obs-density gradient) is already closed. gh#180 is the live
  tracking issue (§10).

## 10. Cross-proposal coordination (ODE-NUTS)

`docs/dev/proposals/2026-06-26-ode-nuts-gradient-spine.md` (lines 131-137)
assumes parametric-`DerivedExpr` "stays rejected by `coeff_guard` … the C1
preflight gate at `pgas.rs` fences it off." **That assumption is now void**:
this proposal (gh#180) _removes_ the bespoke C1 fence and _replaces_ it with the
`DerivEntry::Unsupported` preflight, which admits the now-differentiable
parametric projection (`qgam`) and refuses only the genuinely-uncovered cases.
Two actions: (1) the ODE-NUTS proposal's §131-137 must be updated to reference
the new preflight rather than the deleted fence; (2) ODE-NUTS's `det_grad`
**reuses the shared obs-gradient seam** (§4.3), not a re-derivation — else this
replaces the `eval_resolved_deriv` fork with a PGAS/ODE fork.

## 11. Deferred: 3b — derive the traversal + unify the rate path

3a makes every differentiable _expression_ sealed (exhaustive `Expr`/operator
matches, `DerivEntry` carrying reasons, exhaustive `Likelihood` match). **3b
makes every differentiable _position_ sealed** and is the intended end-state;
deferred to keep 3a landable and to avoid re-keying every rate golden in one
lift. Tracked in **gh#342**. 3b scope:

- **Derive the traversal** — a `#[derive(Differentiate)]` (the run_id
  `#[derive(RunInput)]` technique from
  `docs/dev/notes/2026-06-08-static-typing-as-bug-prevention.md`) that folds
  every `Diffable` field, so adding a differentiable _position_ can't be
  forgotten — coverage becomes a property of the type, not a hand-written pass.
- **Fold the rate path onto the literal `Diffable`/`DerivEntry` type**, retiring
  `rate_grad: HashMap<_, Expr>`. This re-keys all rate goldens (why it is
  deferred).
- **Retire rate E600 into the same fit-time refusal** — rates then compile even
  with a structural-coefficient param and refuse only the NUTS fit, removing the
  rate/obs asymmetry and the latent "E600 blocks forward-sim" question (itself
  worth confirming: is a spline-coefficient param IF2-estimable at all, given
  the basis is frozen at construction?). This subsumes `coeff_guard` entirely
  into the `DerivEntry` preflight.
