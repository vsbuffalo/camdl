# Dimension checking: the algebra is sound, the boundary is not

Date: 2026-08-20 Scope: `ocaml/lib/ir/dimcheck.ml` and every slot that does or
does not reach it Trigger: `NegBinomial.mean` was inferred and discarded, so a
rate-dimensioned projection compiled clean and produced a wrong likelihood
(fixed, `4c44b0e7`)

## The finding that reframes the rest

The four unchecked likelihood arguments were not four independent oversights.
The checker can express only **constant** dimensional expectations —
`require_count` and `require_dimensionless` (`dimcheck.ml:945-951`) assert
`population` or `dimensionless` and nothing else. Sort the 16 observation
likelihood arguments by whether their correct expectation is a constant:

- **12 whose expectation is constant** (probabilities, shape parameters,
  external denominators): **all 12 are checked.**
- **4 whose expectation is a function of the model** — it is the dimension of
  the scored value column — `NegBinomial.mean`, `Normal.mean`, `Normal.sd`,
  `ZeroInflatedNegBinomial.mean`: **all 4 are unchecked.**

The apparent exception confirms it. `Poisson.rate` _is_ checked, against a
hardcoded `population`, and that constant is correct only because a Poisson
never scores anything but a count.

Nobody forgot four times. The idiom ran out, and the arguments it could not
express were dropped rather than modelled. The information needed is already in
hand: `observation_model` carries `scored : string` naming the value column
(`ir.ml:545`), whose declared kind is converted to a dimension into
`st.obs_col_dims` at `dimcheck.ml:1071-1075` — and then never consulted.
`grep -n scored ocaml/lib/ir/dimcheck.ml` returns nothing.

The discard itself is not the bug. For transitions, ODE derivatives, `balance`
and overdispersion, `ignore (infer …)` is always paired with a `propagate` and a
read-phase emit (`:1008`, `:1013`, `:1023`, `:1031` → E300/E308/E305/E306). For
the four likelihood location/scale arguments it is paired with nothing. **The
missing companion is the bug.**

## Confirmed by probe

Each compiled with `camdlc` at `4c44b0e7`; exit status as stated.

### The two remaining holes of the trigger's own class — exit 0

`dimcheck.ml:1112`:
`| Normal n -> ignore (infer st ~ctx n.mean.expr); ignore (infer st ~ctx n.sd.expr)`

```camdl
cases ~ normal(mean = rho * gamma * projected, sd = gamma)   # gamma : rate
```

`mean` is `P·T⁻¹`, `sd` is `T⁻¹`, scored column declared `real`. Accepted. This
is the same wrong-by-the-window-length error as the trigger, in the family
`ocaml/golden/surveillance_likelihoods.camdl:52` itself uses. Normal also has a
contract needing no scored-column knowledge at all: a location and scale must
share a dimension, so `dim(sd) = dim(mean)` is checkable today.

### `permissive_dim` blinds the whole observation block — exit 0

```camdl
y ~ binomial(n = ncount + gamma, p = rho + gamma)   # count + rate, probability + rate
```

`st.permissive_dim <- true` covers the entire observation loop (`:1059-1060`,
restored `:1185`). `unify` then swallows E302 (`:260`) and returns the **left**
operand's dimension (`:267`), which is the dimension `constrain_known` goes on
to inspect — so the mismatch is both unreported and manufactures a value that
satisfies the check. The identical mismatch in a transition rate is E302.

The gh#116 argument checks (`:1036-1055`) are therefore skin-deep: they
constrain the resolved top-level dimension of an expression whose interior has
been made unfalsifiable. One `+` defeats them.

The same window covers the `DerivedExpr` projection (`:1082-1086`), so
`projected = I + gamma` — population plus rate — also compiles clean. That is
how the trigger bug became reachable in the first place.

### `quantities {}` stamps a wrong dimension into IR that Rust trusts — exit 0

```camdl
quantities {
  peak_t   = time_of_max(I)     # [0,1] — a time
  total_d  = final(D)           # [1,0] — a count
  nonsense = total_d - peak_t   # a count minus a time
}
```

Emitted IR: `nonsense  dimension=[1,0]`. dimcheck wrote **a count** onto a
count-minus-a-time. `scalar_dim` handles `Add | Sub` by returning `dl` on
mismatch, silently (`:857-858`); `check_contrasts`'s `infer_ce` handles the
structurally identical case 50 lines later by emitting **E297** (`:911-921`).

The field is not decorative: `rust/crates/cli/src/fit/contrasts.rs:883` reads
`model.quantities[…].dimension` and `combine_dim` (`:982-996`) uses it as the
authority for the runtime contrast dimension check. A contrast differencing
`nonsense` against a genuine count passes. This is a silent-wrong channel that
crosses the language boundary.

### Diagnostics point one declaration too early

```
error[E300]: transition 'death' rate has wrong dimension
 14│    recovery  : I --> R @ gamma * I      <- caret on the WRONG transition
```

`transition_decl` (`parser.mly:656-659`) begins with two nullable prefixes
(`doc_opt`, `lineage_attr_opt`); when both are empty menhir's `$startpos` is the
end position of the previous token, and `trloc` inherits it. For the first
transition in a block the caret lands on `transitions {`. Affects E300, E302,
E303, E304, E308 — essentially every dimensional diagnostic, since almost no
transition carries a `#'` doc or `#[lineage]`.

## Unprobed but read

- **Interventions, events and initial conditions get no dimensional checking at
  all.** `grep` for them in `dimcheck.ml` returns nothing.
  `transfer(fraction = <a count>)` and `init { I = <a rate> }` compile.
  `fraction` has a hard `[0,1]` contract that dimension analysis would catch for
  free.
- **Forcing sub-expressions other than `lag` are unchecked**, and the comment at
  `:970-971` claims `period`/`phase` are validated "the same way" — they are
  not. A `period` in `T⁻¹` inverts a seasonal forcing's timescale.
- **`is_bare_const` is syntactic, not semantic** (`:383-386`). `n = 1000`
  passes, `n = 1000 / 2` passes, `n = 1000 * 5` is E304 — and the error tells
  the user to fix a column that does not exist.
- **Seven live constraints have no test** (`BetaBinomial.n`/`beta`, `Beta.mean`/
  `concentration`, `ZINB.dispersion`/`pi`, E308). All verified firing today;
  each is one refactor from vanishing the way `NegBinomial.mean` did. Note the
  shape: the _tested_ arguments survived and the _untested_ one turned out never
  to have existed.

No test was found that would pass with its constraint removed. The negative
goldens are genuine red-if-reverted; the gap is absence, not vacuity.

## `permissive_dim` is a leak, not load-bearing

Its sole documented justification is the He et al. (2010) variance formula
(cited `:161-167`). It landed 2026-04-20 (`17039cec`, gh#4). `unchecked_dim`
landed 2026-04-22 (`b3f4a993`) **for the same formula**, with a required
`reason` argument and an explicit "never ship a silent escape" rationale (spec
§314). The blanket flag is the un-audited twin of a facility the language
already has, and was never retired.

## Recommendation

**1. Retire `permissive_dim`; migrate its users to `unchecked_dim`.** The
structural fix. Closes the observation-block blind spot, un-blinds the
`DerivedExpr` projection, and converts an invisible block-wide suppression into
a per-expression assertion carrying a reason. Migration cost is a mechanical
one-line wrap in `surveillance_likelihoods.camdl` and any non-homogeneous
variance model.

**2. Replace the per-arm calls with an exhaustive expectation table — where the
table's TYPE is the fix, not its contents.** A table of `arg -> dim_vec` would
not have caught the trigger, because the four unchecked expectations are not
constants. It has to be:

```ocaml
type arg_expectation =
  | ExpCount                 (* external denominator: Binomial.n, BetaBinomial.n *)
  | ExpDimensionless         (* p, pi, alpha, beta, dispersion, concentration *)
  | ExpScored                (* the scored column's dim: NB.mean, Normal.mean, ZINB.mean, Poisson.rate *)
  | ExpSameAs of string      (* Normal.sd = Normal.mean *)

val likelihood_arg_dims : likelihood -> (string * expr * arg_expectation) list
```

as a wildcard-free match under the library's existing `-w @8`, so a ninth
likelihood family fails to compile until every argument declares an expectation.
`ExpScored` is what makes this a fix rather than a tidier checklist — it
consumes `obs.scored` and `obs.columns`, both already resolved into
`st.obs_col_dims`.

Roughly 40 lines added, ~90 deleted. One wrinkle: `ZINB` arguments are bare
`expr` while the rest are `diffable` (`ir.ml:448-468`), so the table returns
`expr` and each arm projects `.expr`.

This is the discipline the repo already chose twice — for `diffable` ("a new
differentiable argument is a compile error until it is differentiated",
`ir.ml:433-441`) and for `autodiff.ml`'s sealed match. Dimensional expectations
are the one part of the likelihood contract left as a checklist while its two
siblings were made type-enforced, and it is the part that produced a −5.5
log-likelihood error on a real fit.

**3. File the rest separately rather than folding them in.** The `quantities`
dimension-stamping bug deserves its own fix: "what does `+` do to dimensions" is
written four times in OCaml (`infer_binop`, `read_dim_binop`, `scalar_dim`,
`infer_ce` — three silently, one emitting E297) and a fifth time in Rust
`combine_dim`. That is the seam.

## What is NOT wrong

No evidence the expression-level dimensional algebra is unsound. Union-find
resolution, the `Any`-as-zero identity, the `Const/Const` ambiguity carve-out,
`Pow`'s read-phase `Unknown (-1)` guard, and the `Cond`-predicate exemption are
all sound, reasoned and tested. The rot is entirely at the boundary — in which
_slots_ connect to that algebra, and in one flag that disconnects a whole block
from it.
