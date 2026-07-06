---
status: proposal
date: 2026-07-06
area: inference autodiff, IR schema, OCaml+Rust differentiation traversal
implements: gh#342 (3b — seal every differentiable position; derive the traversal + fold rate onto DerivEntry)
parent: gh#180, docs/dev/proposals/2026-07-03-unified-obs-gradient-autodiff.md (3a — seal every differentiable expression)
precedes: gh#275 (ODE-NUTS state gradients — the reason 3b lands first)
ir_version: 0.24 → 0.25
---

# Seal differentiation coverage (3b) — one derived traversal, one refusal ledger

## 1. Problem

3a (gh#180) made every differentiable **expression** sealed: the OCaml autodiff
`differentiate` is exhaustive and wildcard-free over `Expr` and its operator
sub-ADTs, it returns a classified `Known | Omitted | Unsupported` instead of a
silent `Const 0.0`, that classification rides the IR as `DerivEntry`, and the
fit-time preflight consumes it. A new `Expr` variant is a compile error until
classified. That closed the silent-zero-gradient class **for expressions**.

It did not close it **for positions**. Two residual holes remain, both of the
same shape 3a killed one level up — a coverage obligation carried by hand
instead of by a type.

**Hole A — the traversal is hand-written ~6 times.** The set of differentiable
positions per likelihood (`Poisson.rate`, `NegBinomial.mean`/`dispersion`,
`Normal.mean`/`sd`, `Binomial.p`, `BetaBinomial.alpha`/`beta`, `Bernoulli.p`,
the σ² of an overdispersed transition, and — post-3b — the transition rate) is
enumerated by hand in every pass that touches gradients:

| pass                | site                                           | what it does per position   |
| ------------------- | ---------------------------------------------- | --------------------------- |
| OCaml produce       | `autodiff.ml:622` `differentiate_likelihood`   | fill each `*_grad`          |
| OCaml serialize     | `serde.ml:808+` per-arm `grad_field`           | emit each `*_grad`          |
| Rust resolve        | `resolved_expr.rs:931+` `ResolvedLikelihood`   | resolve each grad map       |
| Rust eval           | `obs_model.rs` `eval_likelihood_resolved_grad` | evaluate each grad          |
| Rust preflight scan | `pgas.rs:2394+` `note_unsupported` loop        | scan each grad for refusals |
| Rust run_id hash    | `ir_hash.rs:521-559` `write_str_map` per grad  | hash each grad              |

The exhaustive `match Likelihood` (no `_`) in each of these makes a new
**variant** a compile error. It does **not** catch a new differentiable
**argument** added to an existing variant. Add `NegBinomial.zero_infl: Expr`
with its `zero_infl_grad` and forget to wire that slot into one of the six
passes, and you get a silent-zero gradient on that argument — the exact failure
mode 3a made unrepresentable for expressions, still fully representable for
positions.

**Hole B — the rate path is a second, unclassified representation.** The
transition rate still carries `rate_grad: HashMap<String, Expr>`
(`transition.rs:131`), not the `GradMap = HashMap<String, DerivEntry>` the
obs/σ² paths use (`observation.rs:79`). It carries no `Known | Unsupported`
classification, so its refusals cannot ride the preflight; they live in a
separate mechanism, `coeff_guard` (`cli/src/fit/coeff_guard.rs` — ~400 lines of
logic, 795 with tests), that **re-derives coverage** from a three-set
`coeff ∧ ¬body ∧ ¬has_grad` triangulation — precisely the mechanism 3a's obs
preflight was built to replace. Rate and obs refuse the _same_ tier-2b
coefficient (a Periodic step value, a `lag`, an inline-table value via a
non-constant index) through _two_ code paths that can drift.

**Why now — before the state-gradient work (gh#275).** ODE-NUTS state gradients
add **new differentiable positions** (deterministic-flow / ODE-RHS derivatives).
Landing them into an unsealed frame is another hand-wired pass that can forget a
slot — Hole A, one more time, on the highest-risk surface (inference math). Seal
the frame first; then the new positions are forced-complete by construction, and
the ODE-NUTS `det_grad` joins the one sealed refusal-ledger enumeration (§4.4)
rather than introducing a fourth hand-rolled path.

## 2. Background — the two paths after 3a, and where each tier is actually rejected

The three-tier forcing boundary from 3a §3, corrected to show **every** gate a
coefficient parameter passes through, not just the autodiff:

| Tier                        | Example                                            | `CompiledModel::new` value-half | Rate autodiff (OCaml)    | Obs autodiff (OCaml)          |
| --------------------------- | -------------------------------------------------- | ------------------------------- | ------------------------ | ----------------------------- |
| **1 — differentiable**      | Sinusoidal/Fourier amplitude; const-idx cell       | builds live                     | `Known` → `Grad`         | `Known` → `Grad`              |
| **2a — genuine zero**       | param does not drive the coefficient               | builds live                     | absent (proven 0)        | absent (proven 0)             |
| **2b — live, grad omitted** | Periodic step/period; `lag`; non-const idx         | **builds live**                 | `Omitted` → **dropped**  | `Omitted` → `Unsupported`     |
| **3 — structural**          | spline / interp / periodic-spline / piecewise knot | **REJECTED at load**            | `Unsupported` → **E600** | `Unsupported` → `Unsupported` |

The row that matters for 3b is the **value-half** column — the gate 3a's §3
table omitted. It is what reshapes the issue's third sub-goal (§3 below).

## 3. The load-bearing finding — the value-half dominates tier-3

The issue's third bullet asks to "retire rate E600 into the same fit-time
refusal, so rates compile even with a structural-coefficient param and refuse
only the NUTS fit," and flags an open question: "is a spline-coefficient param
IF2-estimable at all, given the basis is frozen at construction?" That question
is load-bearing, and it resolves to **no** — verified against code:

```
$ sed -n '463,482p' rust/crates/sim/src/compiled_model.rs   # eval_structural
    let mut names = Vec::new();
    collect_param_names(expr, &mut names);
    if !names.is_empty() { return Err(SimError::Validation(format!(
        "forcing '{forcing}': {what} references parameter {plist}, but it is
         structural data — {what}s are fixed at construction and cannot be
         estimated. ..."))); }
```

`eval_structural` runs over `model.time_functions` at `CompiledModel::new`
(`compiled_model.rs:1106+`) — the interpolation-knot, spline-basis,
periodic-spline-coef and piecewise arrays — and **rejects any parameter-bearing
entry, reference-independent, even for a `Fixed` parameter.** The test pins it:
`structural_coefficient_rejection.rs` declares `v` as
`ParamValue::Fixed { value:
1.0 }`, references it from no rate and no
observation, and asserts `CompiledModel::new` still errors.

Two consequences:

1. **A tier-3 coefficient parameter is estimable by no method** — not IF2, not
   PF, not NUTS — because the model does not build. The spline basis is a
   construction-time Thomas solve; a live knot would need a per-proposal
   re-solve, which is a real feature (out of scope here), not a gate to relax.
   Retiring rate E600 does not make tier-3 estimable; it moves the rejection
   from `camdlc` (early, source-located, with a reparameterize hint) to runtime
   `CompiledModel::new` (later, no source location) — an **error-quality
   regression for zero capability gain**. So **3b keeps rate E600 for tier-3.**

2. **Obs `Unsupported{StructuralForcing}` is itself unreachable in practice** —
   any model that would emit it is rejected by the value-half first. The 3a test
   `pgas_gate_obs_unsupported.rs::preflight_fires_before_eval_emitted_grad`
   passes only because it hand-builds the
   `DerivEntry::Unsupported{Structural...}` entry with a fictional
   `node: "time_func:spline"` and **no real structural forcing in the model**.
   This is harmless (defense-in-depth for hand-built or future-emitted IR) and
   3b leaves it, but notes it so the seal's own coverage claim stays honest.

The clean consequence: the only tier where a "compiles, but NUTS refuses" state
is **real** is tier-2b — the live-but-grad-omitted coefficients. That is exactly
the tier where rate and obs diverge (rate drops, obs refuses), so that is where
3b's unification pays off. Tier-3 stays E600; tier-1/2a are already symmetric.

## 4. Design (types first)

### 4.1 `Diffable` — one struct per differentiable **likelihood argument**

```rust
// rust/crates/ir/src/deriv.rs
pub struct Diffable {
    pub expr: Expr,
    pub grad: GradMap,          // HashMap<String, DerivEntry>; absent key = genuine 0
}
```

```ocaml
(* ocaml/lib/ir/ir.ml *)
type diffable = { expr : expr; grad : grad_map }   (* grad_map = (string * deriv_entry) list *)
```

The `(expression, its per-param classified gradient)` pair travels as **one
value**, so a derivative can never be written without a slot for its expression,
and a new argument can never be added as a bare `Expr` that some passes
differentiate and others miss. Each likelihood stores its differentiable
arguments as `Diffable` fields:

```rust
pub struct NegBinomialLikelihood { pub mean: Diffable, pub dispersion: Diffable }
pub struct BinomialLikelihood    { pub n: Expr, pub p: Diffable }   // n is NOT a Diffable
```

`n` (Binomial/BetaBinomial) stays a plain `Expr` — it must be θ-independent and
carries no gradient (`obs_model.rs` rounds it to an integer). Its type says so:
it is not a `Diffable`, so no traversal differentiates it, and the "n must be
θ-independent" obligation stays an **explicit** gate (the D-n scan, §4.4), not
an absent grad map.

**Scope: `Diffable` is for the multi-argument surface (likelihoods), not for
single-position owners.** The forget-a-slot hazard of §1 Hole A lives where a
variant has **several** differentiable arguments that must each be wired — the
likelihoods. A transition has exactly **one** differentiable expression (its
rate); an overdispersed draw has exactly one (its σ²). There is no "added a
second argument, forgot its grad" hazard for a single-position owner, so
bundling its `expr` and grad into a `Diffable` buys nothing — and, for the rate,
would leak into the propensity hot path (§4.4). So rate and σ² keep their
existing adjacent `expr` + grad-map shape; only their grad-map **value type** is
unified to `DerivEntry` (§4.4). This is the "consolidate to the natural seam,
not past it" line: bundle where multi-argument forgetting is real, leave the
single positions simple.

**Wire format (decided: let the obs goldens move).** Making `mean` a single
`Diffable` field changes its serialization from two adjacent parent keys
(`{"mean": …, "mean_grad": …}`) to one nested key
(`{"mean": {"expr": …, "grad": …}}`). Keeping the old adjacent shape would
require **hand-written `Serialize`/`Deserialize` on all six likelihood structs**
(serde's `flatten` emits the inner names `expr`/`grad` and _collides_ when two
`Diffable` fields flatten; it cannot rename to `mean`/`mean_grad`), following
the `DrawMethod` precedent (`transition.rs:57-116`) — ~6 × ~45 lines purely to
avoid golden churn we are already taking on the rate side. **Not worth it:**
accept the nested `Diffable` shape on the wire and let the obs goldens move to
it. The move is mechanical (every obs likelihood argument gains the
`{"expr":…,"grad":…}` nesting) and is reviewed under the standing golden rule —
the diff must be _only_ that nesting, nothing else. σ² is already
`{"overdispersed": <expr>, "overdispersed_grad": <GradMap>}` (a single-position
adjacent shape, **not** a `Diffable`) and is unaffected. The **rate** grad
values also re-key (§4.5): bare `Expr` (`{"beta": <expr>}`) → `DerivEntry`
(`{"beta": {"grad": <expr>}}`).

### 4.2 The derived traversal (Rust) — coverage as a property of the type

Collapse the hand-written per-likelihood enumerations to a derived traversal:

```rust
pub trait Differentiable {
    /// Every differentiable position, declaration order, as (label, &Diffable).
    fn diffables(&self) -> Vec<(&'static str, &Diffable)>;
    fn diffables_mut(&mut self) -> Vec<(&'static str, &mut Diffable)>;
}

#[derive(Differentiate)]
pub struct NegBinomialLikelihood { mean: Diffable, dispersion: Diffable }
pub struct BinomialLikelihood { #[differentiate(skip)] n: Expr, p: Diffable }
```

**Decided: include-all + explicit skip, exactly `runid-derive`'s technique — not
skip-by-type.** `#[derive(RunInput)]`
(`rust/crates/runid-derive/src/lib.rs:74-77`) folds **every** field by default
and skips only those tagged `#[run_input(provenance)]`; it never inspects a
field's type — safety comes from the emitted code calling a trait method, so a
field whose type does not implement the trait is a **compile error**.
`#[derive(Differentiate)]` mirrors this: it folds every field into the
`diffables()` collection and skips only `#[differentiate(skip)]` fields (the two
`n` fields). The enforcement is the `Vec<(_, &Diffable)>` element type — an
unskipped non-`Diffable` field (e.g. a new argument mistyped as `Expr`)
generates `&self.field: &Expr` into a `&Diffable` slot and **fails to compile**.
`Expr` therefore must **not** `impl Differentiable` — an empty impl would
silently swallow a mistyped argument and reopen Hole A. (Skip-by-type — "fold
only fields whose written type token is `Diffable`" — is **rejected**: a
proc-macro sees tokens, not resolved types, and a new argument accidentally
typed `Expr` would be _silently dropped_ rather than rejected — the exact
silent-miss 3b exists to kill. Include-all makes the mistake loud; skip-by-type
makes it silent.)

Three of the passes are pure enumerations and collapse onto `diffables()`:

- **run_id hash** (`ir_hash.rs`) — one loop; the per-arm `write_str_map`
  deletes.
- **preflight scan** (`pgas.rs`) — one loop over `diffables()`; the per-variant
  `match` deletes.
- **serialize** — routed through the enumeration (with the caveat in §4.1 that
  the obs wire shape moves).

The other three stay per-variant **by necessity** and the seal reaches them a
different way: **produce** (OCaml `differentiate_likelihood`) and **resolve**
(`resolved_expr.rs`, which must reconstruct named per-variant fields for eval to
`match` on) and **eval** (`obs_model.rs`, a distinct score form per argument)
each have a different closed form per position, so a flat fold cannot replace
them — the OCaml _produce_ seal is full reconstruction (§4.3), and resolve/eval
inherit coverage because they consume the produced `Diffable`s. So the accurate
claim is **"3 pure-enumeration passes fold through `diffables()`; produce stays
per-variant, sealed by reconstruction; resolve/eval stay per-variant by
necessity."**

A new differentiable field is a `Diffable`, so it is in `diffables()`, so every
pure-enumeration pass sees it for free. `diffables()` allocates a small `Vec`,
so it is used at **build / scan / hash time** (each once per fit), never in the
hot eval path: the per-observation evaluator keeps consuming the pre-built
`ResolvedLikelihood` (constructed once), exactly as 3a already does.

The `Likelihood` **enum** also needs a `diffables()` that delegates to the
active variant (`match self { Likelihood::NegBinomial(l) => l.diffables(), … }`)
— a shape `runid-derive` does not generate, so it is a small addition to the
derive (or a hand-written enum impl, which an exhaustive `match` already seals
against a new variant).

### 4.3 The OCaml seal — full reconstruction, not functional update

OCaml has no proc-macro, but it has record-construction exhaustiveness, which
gives the same guarantee for the **producing** pass (the one pass where a
forgotten position is silent-wrong, not merely inert). Today
`differentiate_likelihood` uses a functional update:

```ocaml
| NegBinomial nb -> NegBinomial { nb with mean_grad = g nb.mean;
                                          dispersion_grad = g nb.dispersion }
```

`{ nb with ... }` does **not** force you to name a new field — add
`zero_infl_grad` and forget it here and it silently keeps its prior (empty)
value. Change the producing pass to **full record reconstruction** over
`diffable` fields:

```ocaml
| NegBinomial nb -> NegBinomial { mean       = diff proj nb.mean;
                                  dispersion = diff proj nb.dispersion }
```

Now a new `diffable` field is a compile error ("field `zero_infl` not defined")
until it is differentiated — OCaml's equivalent of the Rust derive. Route
`differentiate` **and** `serialize` through one
`map_diffables : (diffable ->
diffable) -> likelihood -> likelihood` and one
`fold_diffables`, so the position enumeration lives in exactly one OCaml
function, mirroring the one Rust `diffables()`.

### 4.4 Rate grad → `DerivEntry`, and subsume `coeff_guard`

- **Representation (narrow — value type only).**
  `transition.rate_grad:
  HashMap<String, Expr>` →
  `HashMap<String, DerivEntry>` (`GradMap`). The rate **expression** stays
  `transition.rate: Expr`, untouched. This is the minimal change that achieves
  the actual goal (subsume `coeff_guard`): the preflight refuses a tier-2b rate
  coefficient because its `rate_grad` entry is a serialized `Unsupported`, and
  that needs only the grad-map's value type, not the rate bundled into a
  `Diffable`. σ² is **already** `sigma_sq_grad: HashMap<String,
  DerivEntry>`
  (`transition.rs:43`), so it needs no change at all — rate is the lone outlier.

  **Resolve: reuse `resolve_grad_map` (decided, seam over fork).** The runtime
  already resolves a `DerivEntry` grad map to its fast form via
  `resolve_grad_map` (`resolved_expr.rs:849`, used by the obs path). The rate
  path today has its own resolver producing `Vec<(usize, ResolvedExpr)>`
  (`rate_grads_indexed`). Route the rate grad through the **same**
  `resolve_grad_map` — its `ResolvedGradMap` (= `ResolvedDerivEntry` entries)
  cascades through `pgas_grad.rs` (`rate_grads_for_run`,
  `complete_data_loglik_grad`) and ~10 tests that iterate `(idx, ResolvedExpr)`.
  The alternative — hand-unwrap `Grad`/drop `Unsupported` to keep `ResolvedExpr`
  — would fork `resolve_grad_map` (re-copying its unknown-key check), the exact
  parallel-mechanism 3b exists to delete, and it is what the ODE `state_grad`
  will also need shared (§ prereq for gh#275). The cascade is mechanical; take
  it.

  **Interim `coeff_guard` (P3, do not defer to P4).** `coeff_guard` reads
  `t.rate_grad` in two ways that break the moment its value type changes: it
  builds `Transition { rate_grad: … }` with `Expr` values in three test fixtures
  (`coeff_guard.rs:533-541,583,614`) — type errors — and it unions **every**
  `t.rate_grad.keys()` into `has_grad` **without a `Grad` filter**
  (`coeff_guard.rs:382`, unlike the σ² branch at `:385-388`). After the type
  change the new `Unsupported` keys would leak into `has_grad` and coeff_guard
  would _silently stop flagging_ a tier-2b rate coefficient in the P3→P4 window
  — and worse, corrupt the P4 parity oracle (the "old coeff_guard" it compares
  against is no longer the old behavior). So P3 **must** migrate those fixtures
  to `DerivEntry` and add the `Grad` filter at `:382`, atomically with the type
  change — this is not a P4 concern.

  **Rejected: fold rate into a `Diffable` (`transition.rate.expr`).** It would
  put rate under the unified `diffables()`, but rate is a **single** position
  (no forget-hazard, §4.1) and `transition.rate` is on the **propensity hot
  path** — every reader across the sim core (`propensity`, `resolved_expr`,
  `pgas_grad`, the backends) becomes `transition.rate.expr`. That is a sim-wide,
  perf-neutral but wide mechanical sweep for **zero correctness gain** over the
  value-type change, and it drags the hot path through a wrapper the eval path
  must unwrap. It is the over-reach "consolidate to the seam, not past it" warns
  against; if a future arc genuinely wants rate in `diffables()`, it is a
  separate, named consolidation, not 3b.

- **Rate driver policy** (`autodiff.ml`, `differentiate_rate` /
  `differentiate_transitions`):
  - `Known → Grad` (unchanged).
  - `Omitted → DEUnsupported{code}` — **the change**. Today `Omitted` is
    _dropped_ (`autodiff.ml:528`), and `coeff_guard` re-discovers the omission.
    Serialising it as `Unsupported{code}` lets the preflight refuse it directly.
  - `Unsupported → E600` (unchanged — tier-3 stays a compile error, §3).

  The `map2` precedence in `differentiate` (`Unsupported` dominates `Omitted`
  dominates `Known`, `autodiff.ml:67-73`) means a param whose derivative through
  a rate is incomplete anywhere in that rate serializes an `Unsupported` for the
  whole rate. For the cited case — `rate = wpeak·S + weekly(t)` with `wpeak` a
  Periodic step value — the `weekly` term's `Omitted` dominates the `wpeak·S`
  term's `Known`, so `∂rate/∂wpeak` is `Unsupported` even though `wpeak` also
  appears in a body, matching coeff_guard's "flag Periodic even in a rate body."

  **The preflight refuses a _superset_ of coeff_guard, not the identical set —
  and the extra refusals are correct.** This is a deliberate, verified
  divergence, not a refactor bug (do not "fix" a mismatch by weakening the
  preflight). coeff_guard has two unsound escapes that admit a param whose rate
  gradient is genuinely incomplete; `map2` domination refuses those cases:
  - _body escape:_ `rate = k·S + tbl[I]`, `tbl` an inline table with value `k`
    read at a non-constant index. `∂rate/∂k = map2(Known(S), Omitted)` →
    `Unsupported` → preflight refuses. coeff_guard admits it: `k ∈ body` (from
    `k·S`) makes its `¬body` clause false (`coeff_guard.rs:399-403`) — yet the
    emitted gradient dropped the whole term, so NUTS would sample `k` against an
    incomplete gradient today (a latent bias this fixes).
  - _has_grad escape:_ the same table value read at a **constant** index in
    transition A (`Grad`) and a **non-constant** index in transition B
    (`Unsupported`). The preflight scans all transitions and refuses on B;
    coeff_guard's `has_grad` union (`coeff_guard.rs:382`) sees A's key and
    admits (same latent bias).

  So the safety-relevant invariant is directional: **every param coeff_guard
  refuses for a rate/IC-reachable live-omitted coefficient is still refused, and
  any _additional_ refusal is a case coeff_guard unsoundly admitted.** The one
  case where they diverge in the _unsafe_ direction (coeff_guard refuses, new
  admits) is an unreferenced "dead" forcing coefficient — a param that reaches
  no rate/obs/IC, so no grad map carries its `Unsupported`. That admission is
  benign (a param driving nothing has posterior = prior, which NUTS returns
  correctly), and it is not the IC case (§ IC bullet below covers IC-reachable
  params). §6's parity gate encodes this as a _directional_ assertion, not
  equality.

- **Preflight** (`pgas.rs`) gains a scan of each transition's `rate_grad` and
  `sigma_sq_grad` maps, alongside the obs `diffables()` scan — all one refusal
  loop with the same `note_unsupported` closure. This shared "scan every grad
  map in the model for `Unsupported`" enumeration is the **single refusal-ledger
  seam**: `diffables()` covers the multi-argument likelihood surface, and this
  loop is where the single-position owners (rate, σ²) join it — so a new
  single-position grad map (e.g. the ODE `state_grad` from gh#275) is sealed by
  being added to this one enumeration, not by wrapping its hot-path expression
  in a `Diffable`. A tier-2b rate coefficient is now refused **at the `run_pgas`
  boundary** with the compiler's own `code`, protecting every caller (tests,
  API, ODE-NUTS), not just the CLI `if use_nuts` site.

- **`coeff_guard`'s rate/obs half deletes.** Its `periodic_coeff`/`lag` sets,
  its `has_grad` union, its `coeff ∧ ¬body ∧ ¬has_grad` triangulation, and its
  rate-vs-obs partition all collapse into "scan every grad map (obs
  `Diffable.grad`, rate `rate_grad`, σ² `sigma_sq_grad`) for `Unsupported`."
  Only its **IC scan survives** (bullet below), extracted to a small standalone
  guard. The migrated rate/obs tests (§6) move from
  `cli::fit::coeff_guard::tests` to the preflight test module.

- **Residual: initial-condition-referenced coefficients — keep coeff_guard's IC
  half verbatim (decided).** `coeff_guard` scans
  `InitialConditions::Parameterized` and folds IC-referenced forcings/tables
  into its `body`/`coeff`/`has_grad` logic (`coeff_guard.rs:288-292, 315-319`),
  refusing a param that reaches a forcing/table coefficient _only_ through an
  initial-condition expression. This is a **genuine bias hazard, not a benign
  over-refusal**: such a param moves the whole trajectory (it sets the starting
  state), yet no gradient is emitted for it — the compiler emits no forcing
  gradient there **and** camdl computes no gradient for IC expressions at all
  (there is no `ic_grad`; IC/state sensitivity is the separate gh#275 surface).
  The path is reachable — the DSL permits a forcing reference or an inline-table
  lookup in an `init` RHS (spec §15.3) — but no committed golden or fixture
  currently exercises it (verified: `python3` scan of `model.initial_conditions`
  across `ir/golden/*.ir.json` → zero forcing/table refs; the spatial fixtures'
  `N0[patch]` in `init` is an _indexed parameter_, which expands to a plain
  `Param`, not a `TableLookup`). So the IC guard is a defensive net for a real
  but unexercised state, covered by hand-built unit tests rather than an
  end-to-end golden. Rather than **reimplement** this as a "minimal named guard"
  — which risks getting the `body`/`has_grad`/periodic-vs- structural
  distinctions wrong and would over- or under-refuse relative to coeff_guard —
  **retain coeff_guard's IC scan verbatim**, as a small standalone guard renamed
  to say what it is ("initial-condition gradients are not computed; NUTS is
  refused for a param reaching a coefficient only through an IC"). We delete the
  rate/obs parts of `coeff_guard` (subsumed by the preflight) and keep its IC
  part unchanged until IC/state gradients exist (gh#275). Reimplementing a
  working refusal check to shave lines is the wrong trade when a bias hole is
  the downside.

### 4.5 run_id and goldens

- `ir/VERSION 0.24 → 0.25`. Two overlapping golden sets move:
  - **The rate re-key: ~91 files carrying a populated `rate_grad`** (not 74 —
    that count omitted `tests/fixtures/*/ir/`; the full
    `grep -rl '"rate_grad"' ir/golden ocaml/golden tests/fixtures
    rust/crates/sim/tests/fixtures`
    set is ~91). The value type `Expr → DerivEntry` is a genuine byte change
    (`{"beta": <expr>}` → `{"beta": {"grad": <expr>}}`); `normalize_for_hash`
    does **not** strip `rate_grad` (`resolve.rs:89-94`), so run_id moves with it
    — deliberate, coincident with the VERSION bump. `ir_hash` already hashes
    `DerivEntry` by its stable `code` (`ir_hash.rs:98-114`), so the re-key is
    copy-edit-stable, and the Rust hash/serde re-key **for free** (both
    derived).
  - **The obs move: every golden with a populated obs likelihood grad**, now
    nested (`{"mean":{"expr":…,"grad":…}}`, §4.1) rather than adjacent. σ²
    goldens are unaffected (already a `DerivEntry` map, single-position shape).
- **Regeneration is not one command.** `make update-golden` covers
  `ocaml/golden` and `tests/fixtures/*/ir` only. It does **not** touch
  `ir/golden/` (11 files) or `rust/crates/sim/tests/fixtures/` (13), which have
  no regen rule — and ~7 sim fixtures have **no `.camdl` source** (hand-authored
  IR: `licm_ab_*`, `seir_spatial_5`, `sir_coalescent`, `sir_lineage`,
  `two_pool_lineage`, `yule_lineage`) and must be **hand-edited** to wrap each
  `rate_grad` value in `{"grad": …}`. Also the
  `reactive`/`quantities`/`contrasts` fixtures sit behind byte-for-byte
  `check-*-golden` drift gates in `make test` (they go red if not regenerated).
  And the runid representative-model constructor + the gh#128 tests read
  `rate_grad` as `Expr` and must migrate. The P3 commit must spell out the exact
  regen path per directory; this is the load-bearing part of P3, not a footnote
  (see gh#382 — the CLAUDE.md golden doc is itself stale on this).
- Every moved golden is **human-reviewed** — each diff must be _only_ the
  intended wrapping/nesting, nothing else; the re-key is the commit's subject
  (the atomic IR-schema-change procedure), never collateral.

## 5. Implementation — phases

Each phase is `make test`-green and I review the full diff before it lands.
Ordered so the representation and the seal set before the behavioural change
that consumes them. **P1 and P2 land as ONE atomic commit** (the obs likelihood
wire is a cross-language contract: with the nested `Diffable` shape decided in
§4.1, Rust cannot switch without OCaml, or the OCaml-emitted goldens stop
round-tripping — the same atomicity the IR-schema rule requires, and the same
shape P3 uses for rate). They are described as two logical steps for clarity;
they are not separately landable. P3, P4, P5 are independently landable after.

- **P1 (Rust half of the atomic obs phase) — `Diffable` +
  `#[derive(Differentiate)]`.** Introduce `Diffable`, the `Differentiable`
  trait, the `differentiate-derive` crate (include-all +
  `#[differentiate(skip)]`, §4.2), the enum-delegate `diffables()`; convert the
  six likelihood structs to `Diffable` fields (nested wire, §4.1); route the
  pure-enumeration obs consumers (preflight scan, hash) through `diffables()`
  and the per-variant consumers (resolve/eval) onto the `.expr`/`.grad`
  accessors. σ² and rate are untouched.

- **P2 (OCaml half of the atomic obs phase) — `diffable` + full-reconstruction
  seal.** Add `diffable`; convert the producing pass and `serde.ml` to
  `map_diffables`/`fold_diffables` emitting the nested shape; make the producing
  pass a **full record reconstruction** so a new position is a compile error.
  The OCaml wire must match P1's nested shape. **Land P1+P2 as one commit**,
  with the moved obs goldens and the cross-language byte round-trip green. Gate:
  `make test` + obs-golden review.

- **P3 — Rate grad → `DerivEntry` (schema, both languages; atomic).**
  `ir/VERSION
  0.24→0.25`; `rate_grad` value type `Expr → DerivEntry`; rate
  driver `Omitted → DEUnsupported` (`autodiff.ml`). **Blast radius (the honest
  list — the rate _expression_ is untouched, but the rate-_grad_ readers are
  not):**
  - Rust type + serde (`transition.rs`); `ir_hash` re-keys for free (derived).
  - Rust `compiled_model.rs:1429` (`resolve_expr` on the grad value) and `:1602`
    (`expr_contains_dt` in `required_capabilities()`) — two sim-core compile
    breaks.
  - Resolve via the shared `resolve_grad_map` (§4.4) → cascades through
    `pgas_grad.rs` (`rate_grads_for_run`, `complete_data_loglik_grad`) and ~10
    tests iterating `(idx, ResolvedExpr)`.
  - OCaml `ir.ml`/`serde.ml`/`autodiff.ml` **+ `constant_fold.ml:159` (reuse
    `fold_grad_map`) + `licm.ml:191` (needs a NEW `deriv_entry` rewrite helper —
    LICM is a gradient-perf surface, high-risk, gated by `gate_licm_ab`)**.
  - **`coeff_guard.rs`**: migrate its three `rate_grad` test fixtures to
    `DerivEntry` and add the `Grad` filter at `:382` (§4.4 — atomic here, not
    P4).
  - The ~91 re-keyed goldens per the per-directory regen path (§4.5). Gate:
    `make test` + golden review.

- **P4 — Preflight subsumes `coeff_guard`'s rate/obs half (inference — highest
  risk).** Add the `rate_grad`/`sigma_sq_grad` scan to the `run_pgas` preflight;
  **extract `coeff_guard`'s IC scan verbatim** to a small standalone guard
  (§4.4); then delete `coeff_guard`'s rate/obs half. The correctness crux is the
  **directional corpus-parity assertion** (§6), gated on **first adding
  fixtures** that actually exercise the divergence (a param periodic-step, a
  param `lag`, a non-const-indexed table value — each rate-only / obs-only /
  IC-only): without them the assertion compares empty-to-empty and proves
  nothing. Add: rate Periodic/`lag` refused at the boundary; tier-1 rate-forcing
  admitted; tier-3 rate still E600; the IC-reachable param still refused. Gate:
  `make test-inference` + the parity assertion green **before** the deletion
  commit. _I review this phase line by line._

- **P5 — Seal test + docs.** Forced-completeness tests: OCaml — a synthetic
  added `diffable` field fails to compile until differentiated (negative test);
  Rust — an unskipped non-`Diffable` likelihood field fails to compile (the
  include-all guarantee, §4.2). Update `docs/user-features.md`. Gate:
  `make test`.

## 6. Test plan

3b is a **pure refactor of the traversal and the rate representation — no
gradient math changes.** The FD gradient checks (`gradient_check_obs.rs`, the
gh#180 `qgam` proof, the gh#20 σ² gamma checks) must pass **unchanged**; a moved
FD value is a bug in the refactor, not an expected diff.

- **Seal (the point).** OCaml: a synthetic added `diffable` field fails to
  compile until differentiated (negative test). Rust: the derive auto-includes a
  new `Diffable` field in `diffables()`, so a test that adds one and asserts it
  appears in the preflight scan / hash without touching those sites passes —
  documenting that forgetting is unrepresentable.
- **Corpus parity — directional, and the fixtures come first (the P4 crux).**
  Two reviewer findings reshape this from the naive form. (i) The existing
  corpus has **zero** models that exercise the divergence: every param-bearing
  forcing coefficient in the tree is a tier-1 Sinusoidal amplitude (both
  mechanisms admit), every `periodic`/`lag` is constant, no inline table has a
  param cell. So a parity assertion over today's fixtures compares
  empty-to-empty and proves nothing — **P4 must first add fixtures** that
  actually emit an `Omitted` rate derivative for an estimated param: a param
  periodic-step, a param `lag`, a non-const-indexed table value, **each** in a
  rate-only, an obs-only, and an IC-only variant. (ii) The assertion is
  **directional, not equality**: the preflight refuses a superset (§4.4), so
  assert **`new_refusals ⊇ coeff_guard_refusals`** on the rate/IC domain, and
  that every _extra_ refusal is a justified case (the `body`/`has_grad` escapes
  coeff_guard unsoundly admitted) — an equality gate would be RED on the correct
  behavior and tempt a "fix" that reintroduces the bug. Scope the compare to the
  rate + σ² + IC-guard portion (the obs-only preflight already existed in 3a, so
  it is not part of what coeff_guard covered). Both gated green **before** the
  deletion commit, with `coeff_guard` and the preflight temporarily coexisting.
- **Rate tier-2b via preflight** (migrated from `coeff_guard::tests`): a
  Periodic step value, a `lag`, and a non-const-index inline-table value in a
  **rate** are refused at the `run_pgas` boundary with their `code` — including
  the "also-in-a-rate-body" cases
  (`flags_periodic_coeff_param_even_when_in_a_rate_body`,
  `flags_lag_param_even_when_in_a_rate_body`).
- **Rate tier-1 admitted:** a Sinusoidal/Fourier amplitude in a rate carries a
  `Grad` and is NUTS-estimated (parity with the deleted `has_grad` escape).
- **Rate tier-3 still E600:** a spline/interp/periodic-spline coefficient in a
  rate is a compile-time E600 (not a later refusal) — pins §3.
- **IC residual:** a param reaching a forcing coefficient only through an IC
  expression is still refused (coeff_guard's IC scan, retained verbatim).
- **Dead-forcing admission:** a param in an unreferenced forcing coefficient is
  admitted (coeff_guard over-refused it; posterior = prior, correct) — asserted
  so the directional gate's one intentional relaxation is pinned, not
  accidental.
- **Un-inlined expr:** the `Diffable`/`diffable` constructor stores the **raw**
  argument expression (not the projection-inlined form); only the _gradient_
  sees the inlined projection, matching today's `differentiate_obs_arg`. Storing
  the inlined expr would double-apply the projection at eval — a one-line test
  pins it.
- **Golden movement:** σ² byte-identical across P1–P4; obs likelihood goldens
  move once (P1, nested `Diffable` shape); the ~91 `rate_grad` goldens move once
  (P3); each reviewed to be _only_ the intended change.

## 7. Decisions (resolved — no open questions)

- **New proposal, not a 3a revision.** 3a is a landed decision record for work
  on `main`; 3b is a distinct refactor with its own decisions and a correction
  to 3a §11's premise. It references 3a as background.
- **Keep rate E600 for tier-3 (structural).** Verified (§3): a structural
  coefficient param is estimable by no method (the value-half rejects it at
  `CompiledModel::new`, reference-independent, even when `Fixed`). Retiring E600
  is an error-quality regression for zero capability gain. The issue's open
  question resolves to "not estimable."
- **Subsume `coeff_guard`'s rate/obs half; refuse tier-2b via the preflight** —
  carry live-omitted rate coefficients as `DerivEntry::Unsupported` in
  `rate_grad`; the preflight refuses them; delete the rate/obs half, gated on
  the **directional** parity assertion + new fixtures (§6).
- **Rate change is narrow: `rate_grad` value type only, `rate: Expr`
  untouched.** σ² is already a `DerivEntry` map, so rate is the lone outlier.
  Folding rate into a `Diffable` is rejected — a single position on the
  propensity hot path (§4.4). `Diffable` is scoped to the multi-argument
  likelihood surface; `state_grad` (gh#275) enters as a `rate_grad`-sibling
  `GradMap`, not a `Diffable`.
- **Reuse `resolve_grad_map` for the rate path** (§4.4) — the shared resolver,
  not a fork; the cascade through `pgas_grad` + ~10 tests is mechanical and is
  what `state_grad` will also need.
- **Keep `coeff_guard`'s IC scan verbatim** (§4.4) — a param reaching a
  coefficient only through an initial-condition expression is a genuine bias
  hazard (no `ic_grad`); retain the working check unchanged rather than
  reimplement it, until IC/state gradients exist (gh#275).
- **Let the obs likelihood goldens move to the nested `Diffable` wire shape**
  (§4.1) rather than hand-write six per-struct serdes to preserve the adjacent
  shape; σ² unaffected; the **~91 `rate_grad`-carrying goldens re-key**
  (`0.24 →
  0.25`); each move reviewed to be _only_ the intended change.
- **Two-language seal:** Rust `#[derive(Differentiate)]` — **include-all +
  explicit `#[differentiate(skip)]`** on the two `n` fields (the `runid-derive`
  technique; skip-by-type is rejected because it silently drops a mistyped
  argument, §4.2); `Expr` must **not** `impl Differentiable`. OCaml full record
  reconstruction for the producer.
- **Pure refactor of representation + refusal path:** no gradient formula
  changes; the FD test matrix is the invariant, not a diff. (The one behavioral
  change is intentional and safe-direction: the preflight refuses cases
  coeff_guard's `body`/`has_grad` escapes unsoundly admitted — a fixed latent
  bias, §4.4.)
- **Lands before gh#275 (state gradients)** so ODE-NUTS `det_grad` joins the one
  sealed refusal-ledger enumeration (§4.4) rather than adding a fourth
  hand-wired pass.
