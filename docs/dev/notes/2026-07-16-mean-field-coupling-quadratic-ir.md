# Dense mean-field coupling compiled to an O(H²) IR

Date: 2026-07-16 Project: camdl Tags: expander, autodiff, hoisting, scaling,
metapopulation

## Context / question

A TB household-transmission probe (household-as-stratum, H households, two-scale
force of infection) hit a hard ceiling around H≈500: IR size, compile time, and
per-simulation cost all grew as O(H²). The between-household mean-field term was
the cause — a local-FoI-only variant of the same model scaled O(H).

The model shape is the textbook metapopulation mean-field, not anything exotic:

```camdl
let N[h in hh] = S[h] + Lf[h] + Ls[h] + I[h] + T[h] + R[h]
let I_glob = sum(h in hh, I[h])
let N_glob = sum(h in hh, N[h])
let foi[h in hh] = beta_w * I[h] / N[h] + beta_b * I_glob / N_glob
```

## Measurements

`camdlc --no-state-grad`, IR bytes, H households (2·H FoI-bearing transitions,
14 parameters). `x/dbl` is the growth factor per doubling of H — 4× is O(H²), 2×
is O(H).

| H   | baseline | x/dbl | + Σ-fold | x/dbl | + Σ-fold + hoist | x/dbl | total |
| --- | -------- | ----- | -------- | ----- | ---------------- | ----- | ----- |
| 20  | 0.86 MB  | —     | 0.24 MB  | —     | 0.16 MB          | —     | 5.4×  |
| 40  | 2.88 MB  | 3.36× | 0.65 MB  | 2.70× | 0.32 MB          | 1.98× | 9.1×  |
| 80  | 10.45 MB | 3.62× | 2.00 MB  | 3.05× | 0.63 MB          | 1.99× | 16.5× |
| 160 | 40.23 MB | 3.85× | 6.85 MB  | 3.43× | 1.27 MB          | 2.01× | 31.6× |
| 320 | —        | —     | —        | —     | 2.55 MB          | 2.01× | —     |

Baseline trends to 4×/doubling (O(H²)); with both fixes it sits at
2.01×/doubling (O(H)) across four doublings. The win compounds with H: 31.6× at
H=160, and the extrapolated H=400 figure goes from ~236 MB to ~3.2 MB.

## Observations

Two independent defects, both hit by the same model shape. Neither is the one
the first-pass diagnosis named.

**A false lead worth recording.** The obvious probe —
`rg -c BindingRef model.ir.json` → 0 — reads as "nothing is hoisted at all". It
is a measurement artifact: the serialized JSON tag is `binding_ref`
(`ocaml/lib/ir/serde.ml:113`), not the OCaml constructor name. The same file at
H=20 in fact carries 21 bindings and 24,720 `binding_ref` uses. Grepping IR JSON
for a constructor name always returns 0; grep the serde tag.

Where the bytes actually were, at H=20 (`model.transitions` = 97.7% of the IR):

| field           | bytes   |
| --------------- | ------- |
| `rate_grad`     | 788,790 |
| `rate`          | 61,690  |
| `metadata`      | 29,140  |
| `stoichiometry` | 7,520   |

**Defect 1 — `simplify` cannot fold a `Reduce` of zeros.** `Autodiff.simplify`
matched `BinOp`, `UnOp`, `Cond`, and fell through on everything else, so
`Reduce` (the n-ary sum) was an opaque leaf. Differentiating `Σ_h N[h]` w.r.t.
any parameter yields `Reduce [Const 0.0; ...]` — one zero per term, _not_
`Const 0.0`. Unfolded, the quotient rule's `-f·g'` term never meets
`Mul, _, Const 0.0`, so `Div, Const 0.0, _` cannot fire, and the whole O(H)
denominator survives — for every parameter. `infection_h0` emitted a 1,254-byte
gradient for all 14 parameters, including `rho` and `k`, which are
observation-only and appear nowhere in any rate:

```
d(infection_h0)/dk  =  (neg(0.0) / (Σ N_h · Σ N_h)) · S_h0
```

The `neg(const 0.0)` is the tell: `Sub, Const 0.0, x -> Neg x` fired because the
right operand was an unfoldable `Mul(u, Reduce[0,…])`, and a later pass
collapsed its interior without revisiting the `Neg`. After the fix
`infection_h0` carries exactly `beta_w` and `beta_b`.

**Defect 2 — the hoist predicate rejected an aggregate-of-an-aggregate.**
`let_is_hoistable` (`expander.ml`) requires `not (body_refs_param_or_let …)`,
and `body_refs_param_or_let` returned true — rejecting the hoist — for _any_
body naming another `let`, commented "conservatively excluded; it may
transitively carry a parameter."

The contract it guards is narrower than that: `validate.ml` requires a hoisted
binding body to be **param-free**, because `autodiff` differentiates
`BindingRef` to 0 under `WrtParam` (a param leaking into a binding is a silent
zero gradient). Param-freeness is closed under let-reference, so "names a let"
is strictly coarser than the property. `N_glob` names `N`; `N` is
compartments-only, so `N_glob` is param-free and hoistable — but was rejected,
and its O(H) sum was inlined into all H rates.

`N[h]` and `I_glob` hoisted correctly all along, which is why the bindings list
was non-empty and why only `N_glob` was missing:

```
'N_glob' is param-free and must hoist into model.bindings; got [N_h0; I_glob; N_h1; N_h2]
```

The two defects compose: defect 2 puts an O(H) sum in each rate, and defect 1
copies it into all 14 gradient columns of each of the 2·H FoI transitions.

## Interpretation

Both fixes are value-preserving: neither changes a rate's or a gradient's value.
They are not size-only — see "Runtime effect" below.

- The Σ-fold removes provably-zero summands (`Const 0.0` is the additive
  identity of `Reduce`, exactly as the existing `Add, x, Const 0.0 -> x` rule
  assumes) and drops derivative keys that fold to `Const 0.0` — where an absent
  key already means "genuine zero" by the documented `differentiate_rate`
  convention.

  `simplify_fixpoint`'s callers are `differentiate_rate` (:790),
  `differentiate_rate_state` (:814), `obs_deriv_entry` (:893), and
  `lineage.ml:230–231`'s `weight_of_parent`. The first three are derivative
  expressions. The fourth is **not** — a lineage parent-pool weight is evaluated
  during forward simulation — so "the fold only touches gradients" would be
  wrong. It is still value-preserving there: `deriv_num_wrt_pop` maps a `Reduce`
  termwise, so `Reduce [0; 0; β·S; 0]` folds to `β·S`, the same sum minus exact
  zeros. Lineage transitions also set `suppress_hoist` (`expander.ml`), so the
  hoist change cannot reach them at all.
- The hoist replaces an inlined expression with a `BindingRef` to that same
  expression, evaluated on demand from an identical body.

Verified rather than argued. With the Σ-fold alone, every forward `rate` and the
whole `bindings` list are byte-identical to baseline (compared the serialized
`rate` of all 340 transitions at H=20). The hoist _does_ change the forward
`rate` — an inlined sum becomes a `BindingRef` — so that was checked end-to-end
rather than assumed:

```
camdl simulate {base,m}.ir.json --scenario baseline --backend chain_binomial --dt 1 --seed $s -o …
  seed=1   BYTE-IDENTICAL  rows=3653  cksum=1646313648
  seed=7   BYTE-IDENTICAL  rows=3653  cksum=2890757652
  seed=99  BYTE-IDENTICAL  rows=3653  cksum=735862416
  seed=42  both fail identically: NumericalCollapse { kind: DivByZero, t: 2532.0 }
```

H=80, baseline IR 10.4 MB vs fixed IR 0.63 MB. Seed 42 is not a regression: a
household empties (`N_h → 0`) at t=2532 and the runtime hard-errors on the
divisor, before and after, identically.

That last one also kills a tempting overclaim. The unfolded
`∂rate/∂k =
-0.0/(Σ N_h)²` would be `NaN` at zero population, so the fold looks
like it removes a NaN-gradient hazard. It does not, by default: the forward rate
divides by the same zero and `NumericalCollapse` fires first. The claim is only
live under `--allow-degenerate-rates` (which substitutes the legacy zero-rate
behaviour), and that path was not exercised here — so it is a hypothesis, not a
demonstrated fix.

Scope of who is affected: the fixes fire on any model with a param-free `let`
naming another param-free `let`, or a `Reduce` under a nonlinear operator — that
is, every mean-field/metapopulation model with a global denominator. The win is
nil at H=1 and compounds with the number of coupled strata.

## Runtime effect — not just compile time

The IR shrink is not the point; it tracks a real per-step cost. `BindingRef` is
memoized within one propensity-vector evaluation (`BINDING_CACHE`,
`rust/crates/sim/src/resolved_expr.rs`), so a hoisted `N_glob` is evaluated once
per step and reused by all 2·H rates. Inlined, each rate re-walked its own O(H)
sum — O(H²) work per propensity vector.

Forward sim, 10y daily chain-binomial, seed 1, best of 3:

| H   | baseline | fixed | speedup |
| --- | -------- | ----- | ------- |
| 20  | 0.11s    | 0.09s | 1.29×   |
| 40  | 0.25s    | 0.16s | 1.53×   |
| 80  | 0.64s    | 0.32s | 2.03×   |
| 160 | 1.93s    | 0.59s | 3.27×   |

Baseline grows ~2.3–3.0× per doubling (superlinear); fixed grows ~1.8–2.0×
(linear). Forward simulation goes from superlinear to linear, so the speedup
keeps growing with H.

Attribution: this forward-sim win is **defect 2** (the hoist) alone. Defect 1
touches only gradient expressions and lineage weights, so it is invisible to
forward sim. Its payoff is on gradient-based inference (NUTS/PGAS), where the
dropped parameter columns are gradient evaluations skipped per transition per
step — mechanically the larger of the two (O(H²)→O(H) per gradient evaluation).
**That is unmeasured**: it needs a fit run, and no number is claimed here.

## Why defect 1 escaped the exhaustiveness seal

Worth recording, because the guard rail for exactly this class of bug exists and
still missed it.

`Autodiff.simplify` was written 2026-04-06 (`f70dd291`), when `Reduce` did not
exist. `Reduce` was added seven weeks later (`8d990a0a`, 2026-05-29). The `ir`
library carries a deliberate seal in `ocaml/lib/ir/dune`:

```
; -w @8: promote non-exhaustive-match (warning 8) to a hard error in every
; profile. autodiff.ml's `differentiate` is exhaustive and wildcard-free …
```

The seal worked where it was aimed. `8d990a0a` **did** add a `Reduce` arm to
`differentiate` — omitting it would not have compiled. It did **not** add one to
`simplify`, in the same file, because `simplify` ends in `| _ -> e`: a catch-all
wildcard makes warning 8 unreachable, so the compiler stayed silent and the new
variant became an opaque leaf.

The generalizable bit: `-w @8` protects a function only if that function is
wildcard-free. A `| _ ->` in a pass that must handle every variant is not a
convenience — it opts that pass out of the seal. `simplify`'s remaining wildcard
still covers `TableLookup` args and `UncheckedDim.inner` (see below).

## Golden impact

Both defects are live across the committed corpus, not just the probe. 12
goldens change — every stratified/mean-field model in `ocaml/golden`. Gradient
bytes (`rate_grad` + `rate_state_grad`) fall 2,389,443 → 651,307 (3.67×).
Audited structurally rather than eyeballed: no transition set changed, no
binding was removed, and

- **11 of 12** have a **byte-identical forward `rate`** and **no new bindings**
  — pure Σ-fold, gradients only.
- **1** (`seir_cross_dim`, the only aggregate-of-an-aggregate in the corpus)
  gains 4 bindings and changes 16 of 120 rates. The rate edit is exactly the
  substitution and nothing else:

  ```
  old:  … / reduce[binding_ref N_north_child, N_north_adult, N_north_elder]
  new:  … / binding_ref N_total_north
  with: N_total_north = reduce[N_north_child, N_north_adult, N_north_elder]
  ```

  Its gradient bytes drop 1,845,427 → 416,667 (4.4×) — the largest in the
  corpus, as expected for the one model with the shape.

What the committed goldens were carrying is worth seeing directly. `seir_age`'s
infection transition had a gradient for `sigma` (E→I progression) and `gamma`
(recovery) — parameters that appear nowhere in an infection rate:

```
rate_grad.sigma = beta·S_child · reduce[ 0/(N_child·N_child), 0/(N_adult·N_adult) ]
```

Structurally present, evaluating to zero, differentiated w.r.t. a parameter the
expression does not contain. Those keys are now absent, which is what the
`differentiate_rate` convention already means by "genuine zero".

## What this does not fix

The remaining O(H) per-model cost is irreducible: `beta_b`'s gradient genuinely
contains `I_glob/N_glob`, and `N_glob` is a sum over H strata — now referenced
once rather than copied. The ceiling this lifts is IR size / compile / memory,
and the per-eval work that tracked it. Whether large-H global-coupling
_inference_ is practical is a separate, untested question (particle degeneracy
at household resolution; and `beta_w` is weakly identified from aggregate
notifications — a finding from the same probe, independent of this).

`simplify` still does not recurse into `TableLookup` index expressions or
`UncheckedDim.inner`. Both are real gaps in the same function, neither has a
measured consequence, and folding them would churn goldens for no demonstrated
win — left alone deliberately.

## Next

- gh#439 (unconditional state-Jacobian on mean-field coupling) is a different
  axis of the same model's cost and still open.
- The probe's aggregate-vs-household identifiability question is unaffected.
- Measure the gradient-evaluation win (NUTS/PGAS) on a mean-field model; it is
  the larger claim and is currently unmeasured.
- Unrelated, found while gating: `camdlc render ocaml/golden/sir_basic.ir.json`
  fails with `parse error`, so the CAS archive degrades to
  `warning: could not render model for archive` (21 occurrences in one
  `make test`). Pre-existing — the pre-change `camdlc` built from HEAD fails
  identically, and `sir_basic.ir.json` is untouched by this work. Needs its own
  issue.
