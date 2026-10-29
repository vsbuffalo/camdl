# An automatic random-walk step of 1e-149 froze every positive parameter boxed at zero

Date: 2026-10-29

Status: **refused, not yet fixed.** Since `0abe59f8` (2026-10-28) camdl refuses
the configuration instead of running it. The architectural fix, a step defined
in the transformed coordinate, is decided in
`docs/dev/proposals/2026-10-29-support-keyed-transforms.md` (D8) and is being
implemented on the branch `feat/support-transforms`; it is not on main.

Affected: every camdl build from `4a90daac` (2026-04-02) to `3c480d2c`, the
parent of `0abe59f8` (2026-10-28), including the `v0.1.0-alpha` tag and every
installed `0.1.0+<hash>` in that range. That is about seven months and 2,729
commits.

Class: **code-vs-code**, with a **doc-vs-code** component. The function that
derives an automatic step on the natural scale disagrees with the code that
converts that step to the scale the samplers move on: the conversion turns it
into a step that cannot change the parameter in double precision. Separately,
`docs/fit-toml.md` said the step was read only by IF2, while PMMH and MH also
read it.

## Terms

- **IF2** (iterated filtering, Ionides et al. 2015) estimates a maximum
  likelihood by perturbing each particle's parameters with a random walk and
  shrinking the perturbation over iterations.
- **PMMH** (particle marginal Metropolis–Hastings) samples the posterior with a
  random-walk Metropolis–Hastings proposal, scoring each proposal with a
  particle-filter likelihood. **MH** is the same proposal scored with the
  deterministic ODE likelihood (`algorithm = "mh"`).
- **`rw_sd`** is the per-parameter random-walk standard deviation, set in a fit
  config's `[estimate]` entry. IF2 perturbs by it. PMMH and MH start their
  proposal at five times it, converted to the transformed scale (`fit/pmmh.rs`,
  `build_proposal_sd`), and an adaptive scheme rescales from there. When no
  `rw_sd` is given, camdl derives one from the parameter's search box: the
  **automatic step**.
- **Transformed scale.** A parameter of kind `rate`, `positive`, `count` or
  `duration` with a lower bound of zero or more is searched on the log scale, z
  = ln θ (`Transform::Log`). A natural-scale step s at θ becomes s/θ in z.

PGAS (particle Gibbs with ancestor sampling, with NUTS or MH-within-Gibbs for
the parameters) and the `nl-*` optimisers did not read `rw_sd` on main and were
not affected.

## What happened

`auto_rw_sd_from_value` (`rust/crates/cli/src/fit/runner.rs`, at `3c480d2c`
line 1474) derived the log-scale automatic step from the box [lo, hi] as one
twentieth of the log range, mapped to the natural scale at the geometric
midpoint:

```rust
Transform::Log { lo, hi } => {
    let lo = lo.max(1e-300);
    let hi_val = if hi.is_finite() { *hi } else { lo * 1000.0 };
    let log_range = (hi_val / lo).ln();
    let log_sd = log_range / 20.0;
    // Convert to natural scale at geometric midpoint
    let midpoint = (lo * hi_val).sqrt();
    midpoint * log_sd
}
```

For a box starting at 0 the floor makes lo = 10⁻³⁰⁰. The log range is then about
691, which is large but finite; the geometric midpoint √(10⁻³⁰⁰·hi) is about
10⁻¹⁵⁰. Their product is the step: 3.45·10⁻¹⁴⁹ for [0, 1] and 4.89·10⁻¹⁴⁹ for
[0, 2]. On the transformed scale that step is s/θ ≈ 10⁻¹⁴⁸ for any θ of order
one. Adding it to z = ln θ leaves z unchanged in double precision, whose
relative resolution is 2.2·10⁻¹⁶.

The parameter therefore never moved. Each chain stayed at its starting value,
and chains start at dispersed draws (by default one prior draw per chain). The
pooled result for the parameter was the spread of the chain starts. That spread
looks like a plausible, somewhat wide posterior. Parameters correlated with the
frozen one compensated for it chain by chain, so they disagreed across chains
and their between-chain R̂ (the potential scale reduction factor) was inflated. A
reader sees a convergence problem on the correlated parameter rather than a
parameter that was never estimated.

A box of the form [0, hi] is the ordinary way to declare a rate:
`beta : rate in
[0.0, 2.0]`. Before the fix a regex scan found 309 positive-kind
declarations with lower bound 0 and a finite upper bound across this repository
and its sibling model repositories (proposal §4); whether each was estimated
with an automatic step is a separate question, answered for the sibling
repositories in "Downstream impact" below.

## Timeline

Found with `git log -S 'auto_rw_sd_from_value'` and
`git log -G 'lo\.max\(1e-300\)'` on main.

| Date       | Commit     | Event                                                                                                                                                                |
| ---------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-03-31 | `1acdabd7` | The automatic log-scale step is 0.1 × the current value, which is a 10% move on z at any θ. Not degenerate.                                                          |
| 2026-04-02 | `4a90daac` | **Introduced.** `Transform` carries its bounds, and the log-scale step becomes `√(lo·hi)·ln(hi/lo)/20` with lo floored at 10⁻³⁰⁰. IF2 under `camdl fit` is affected. |
| 2026-04-02 | `ba84cd04` | `profile --rw-sd auto` uses the same derivation; profile's IF2 cells are affected.                                                                                   |
| 2026-04-04 | `979f78da` | PMMH added; its initial proposal is `transformed_sd(rw_sd, initial) × 5`. PMMH is affected.                                                                          |
| 2026-05-31 | tag        | `v0.1.0-alpha` contains the defect.                                                                                                                                  |
| 2026-06-15 | `590e80da` | MH on the ODE backend added, sharing PMMH's proposal construction. MH is affected.                                                                                   |
| 2026-10-26 | `ab0f9758` | `docs/fit-toml.md` corrected to say PMMH and MH read `rw_sd`; until then it said "Ignored by every other method".                                                    |
| 2026-10-27 | `8d5d521c` | Sibling case refused: a box with an infinite end, where the same floor gave a step of about 10⁻²⁹⁹. Finite [0, hi] boxes still ran.                                  |
| 2026-10-28 | `0abe59f8` | **Refused.** A log box with lower bound ≤ 0, or any automatic step that is non-finite, ≤ 0 or below 10⁻⁹ of the box width, is refused for IF2, PMMH, MH and profile. |

The defect was live for 209 days, from `4a90daac` to `0abe59f8`.

## Detection

The defect was found while writing the support-keyed transforms proposal
(`docs/dev/proposals/2026-10-29-support-keyed-transforms.md`, §4), by evaluating
the automatic-step expression for a [0, 2] box while auditing how each sampler
converts `rw_sd` to the transformed scale. It was not found by a test, by a
diagnostic, or by a user. The proposal recorded it as inferred from the code;
`0abe59f8` and this report reproduce it.

## Reproduction

The case is `tests/recovery/cases/sir` (closed SIR, weekly negative-binomial
cases, truth β = 0.4, γ = 0.15) with one edit to the model: β's box becomes
[0.0, 1.0] instead of [0.05, 1.0]. The fit configs estimate β with no `rw_sd`
and γ boxed to [0.01, 0.5]; N0, I0, ρ and k are fixed at truth.

```
# model.camdl: the one changed line
  beta  : rate         in [0.0, 1.0]   ~ log_normal(mu = -1.0, sigma = 0.5)
```

```toml
# if2.toml
output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "data/weekly_cases.tsv"
[estimate]
beta = { start = 0.25 }
gamma = { bounds = [0.01, 0.5], start = 0.10 }
[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 30
cooling = 0.7
```

`pmmh.toml` is the same with `algorithm = "pmmh"`, `particles = 200`,
`iterations = 400`, `burn_in = 100` and no `cooling`.

Builds: two detached worktrees of this repository, `$PRE` at `3c480d2c` (the
parent of `0abe59f8`) and `$FIX` at `0abe59f8`, each built with
`make build-ocaml` and `cargo build --release -p cli --bin camdl`. Every command
runs with `CAMDLC=$X/ocaml/_build/default/bin/camdlc.exe` and
`CAMDL_SKIP_VERSION_CHECK=1` for the matching build `$X`.

```
$ $PRE/rust/target/release/camdl --version
camdl 0.1.0+3c480d2c (2026-10-29)
$ $PRE/rust/target/release/camdl simulate model.camdl --backend chain_binomial \
    --seed 1 --params truth.toml --obs-only-dir data --output-dir simres
observations written to data/weekly_cases.tsv
```

### IF2 on the pre-fix build: β never leaves its start

```
$ $PRE/rust/target/release/camdl fit run if2.toml --seed 1 --no-progress
...
transforms (chain 1 of 4; chains start at different points — see chain_starts.tsv):
  beta         log     [0, 1]  log(0.2759) = -1.29  rw_sd=0.0000 (0.000/step, auto)
  gamma        log     [0.01, 0.5]  log(0.1602) = -1.83  rw_sd=0.0138 (0.086/step, auto)
  ⚠ 2/2 parameters using auto rw_sd. Check traces and set explicit values.
...
best chain: 3 (loglik=-58.58 ± 0.01)
Â (IF2 chain agreement), threshold 1.01:
  beta         Â=n/a (W ≈ 0; rely on Δ_dB)
  gamma        Â=10.846 ✗
── diagnostics ──────────────────────────────────────
  i Auto rw_sd for 'beta': 0.000000.
  i Auto rw_sd for 'gamma': 0.013831.
```

Exit status 0. Two of the four chains were skipped because their starts made the
filter degenerate, which is unrelated to this defect. The traces of the two
surviving chains (`chain_<i>/parameter_traces.tsv`, columns `iteration`,
`loglik`, `if2_perturbed_loglik`, `beta`, `gamma`):

```
== chain_1      start beta = 0.4252950066706779
0   NA  -61.08  0.4252950066706741   0.2062163055524114
29  NA  -59.22  0.4252950066706741   0.24131701434889713
== chain_3      start beta = 0.4768413416314137
0   NA  -80.21  0.47684134163141983  0.16159612209350308
29  NA  -59.31  0.47684134163141983  0.28962722788557876
```

Every row of chain 3 carries the same β to 17 significant figures, which differs
from the start only by one round trip through ln and exp. The reported maximum
likelihood estimate is that chain's start, β = 0.4768, with γ = 0.2896 moved to
compensate (truth 0.4 and 0.15). The R̂ of 10.8 on γ is the compensation, chain
by chain, for a β that was never searched.

### PMMH on the pre-fix build: the pooled posterior is the spread of the starts

```
$ $PRE/rust/target/release/camdl fit run pmmh.toml --seed 1 --no-progress
...
pmmh: 4 chains × 400 steps × 200 particles, burn_in=100, thin=10, adapt=true
  proposal_sd (transformed): [beta=0.0000, gamma=0.6916]
...
  beta         Rhat=3.993 ✗ ESS bulk=6 tail=6 (5.4% of 120 draws)
  gamma        Rhat=2.891 ✗ ESS bulk=7 tail=15 (5.7% of 120 draws)
```

Exit status 0. β in the 120 retained draws (`draws.tsv`), per chain, against
each chain's start (`chain_starts.tsv`, the `accepted` row):

| chain | start β | β range in draws  | distinct values |
| ----- | ------- | ----------------- | --------------- |
| 0     | 0.2759  | 0.27569 – 0.27676 | 10              |
| 1     | 0.4253  | 0.42456 – 0.42794 | 11              |
| 2     | 0.1679  | 0.16792 – 0.16792 | 1               |
| 3     | 0.4752  | 0.47370 – 0.47697 | 11              |

After warm-up three chains moved within 0.4% of their starts; chain 2 did not
move at all. The mechanism of the small post-warm-up moves was not investigated;
they do not change the conclusion. Pooled, β has mean 0.336 and sd 0.122.

The control is the same config with `beta = { start = 0.25, rw_sd = 0.02 }`, run
on the same pre-fix build with the same seed, so the chains start at the same
four points:

```
$ $PRE/rust/target/release/camdl fit run pmmh_rwsd.toml --seed 1 --no-progress
  proposal_sd (transformed): [beta=0.4000, gamma=0.6916]
  beta         Rhat=1.147 ~ ESS bulk=22 tail=33 (18.3% of 120 draws)
  gamma        Rhat=1.165 ~ ESS bulk=21 tail=30 (17.7% of 120 draws)
```

Pooled β has mean 0.415 and sd 0.040, against truth 0.4. The frozen run's sd is
three times the posterior's and its mean is off by two posterior sds: a
plausible-looking wrong posterior. (The control run on `$FIX` gives the same
draws, as expected for a commit that only adds a refusal.)

### On `0abe59f8` both fits are refused

```
$ $FIX/rust/target/release/camdl --version
camdl 0.1.0+0abe59f8 (2026-10-29)
$ $FIX/rust/target/release/camdl fit run if2.toml --seed 1 --no-progress
error building run config: `beta` has search box [0, 1] on the log scale, and
a log-scale random walk has no automatic step (rw_sd) when the lower bound is
0: the derived step would be 3.4538776394910683e-149, which never moves beta.
Give an explicit rw_sd (`rw_sd = …` in its [estimate] entry), or a positive
lower bound (`in [lo, hi]` in the model, or `bounds = [lo, hi]` in
[estimate]).
$ echo $?
1
$ $FIX/rust/target/release/camdl fit run pmmh.toml --seed 1 --no-progress
error running pmmh: `beta` has search box [0, 1] on the log scale, ...
$ echo $?
1
```

The printed step, 3.4538776394910683·10⁻¹⁴⁹, is √(10⁻³⁰⁰·1)·ln(10³⁰⁰)/20, as the
derivation above predicts.

## Root cause

The automatic step is defined on the natural scale from the box, and then
converted to the transformed scale at the parameter's value. Nothing checks that
the converted step can move the parameter. On the log scale the natural step is
anchored at the geometric midpoint of the box. When the lower bound is 0, the
floor that keeps ln finite puts that midpoint 150 orders of magnitude below any
value the parameter takes, and the conversion carries the discrepancy straight
into z.

## The class: this is the third degenerate automatic step

| Issue / commit        | Box and transform                                   | Natural-scale step          | Effect                                                                                                                          |
| --------------------- | --------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| gh#763 (closed 09-01) | `probability` with no `in [...]`: Logit over [0, ∞] | ∞                           | IF2's first perturbation is NaN; the chain errors with `NonFiniteParameter`. Loud.                                              |
| gh#967 (closed 10-27) | `duration` boxed [−120, 120]: Log, start −20        | explicit 2.0 → 2·10³⁰⁰ in z | `ln(max(−20, 10⁻³⁰⁰))` = −690.8, so the step in z is 2/10⁻³⁰⁰; the first perturbation throws every particle to a bound. Exit 0. |
| this incident         | positive kind boxed [0, hi]: Log                    | ≈ 5·10⁻¹⁴⁹                  | The parameter never moves. Exit 0.                                                                                              |

The infinite-end case refused by `8d5d521c` (a step of about 10⁻²⁹⁹ for an
unbounded `rate`) is the same mechanism as this incident on a box with no upper
end.

The shared root cause: a step is derived on the natural scale, from the box, and
converted to the transformed scale through a derivative evaluated at a floored
or clamped value; no step checks that the result is a usable move in the
coordinate the sampler walks in. Each defect was a different way for the floor
or clamp (10⁻³⁰⁰ in the log transform, an infinite end in the logit) to reach
the conversion. Each fix so far has closed one entrance: gh#763 gave
`probability` its [0, 1] support, gh#967 (`6bbc0a9b`) gave a box below zero the
scaled logit, `8d5d521c` refused an infinite end, and `0abe59f8` refuses a zero
lower bound on the log scale plus any automatic step below 10⁻⁹ of the box
width. The last condition is the first check on the step's size itself rather
than on the box that produced it.

## Downstream impact

An audit of the sibling model repositories' stored fits, made from the stored
chains, found seven runs in which a parameter was not estimated:

- **ebola-bdbv-camdl**, four runs (one correlated-PMMH, three PMMH), all on the
  process-noise standard deviation `sigma_se` boxed at 0. In the correlated-PMMH
  run each chain stayed within about 2% of its start and R̂ was 3.38.
- **camdl-garki**, three ODE MH runs: one on a random-effect standard deviation
  (`sigma_c`), with one value per chain in all six chains; two on a waning rate
  (`w`), frozen for about 300 steps until the adaptive proposal moved some
  chains.

Seven further run directories with the same configuration never sampled and
carry no estimate. The Ebola project's main PGAS fits do not read `rw_sd` and
are unaffected.

Three downstream conclusions rest on the affected runs and are withdrawn there:

1. An Ebola-project comparison of PGAS against correlated PMMH, which found
   correlated PMMH worse, is confounded: each correlated-PMMH chain sampled with
   `sigma_se` fixed at a different value.
2. The Ebola project attributed the frozen `sigma_se` to collapse of the
   proposal scale, and rejected the zero-lower-bound explanation because a
   second parameter moved. That parameter is a `probability`, which gets a logit
   transform and a working step, so the test could not detect this defect. A
   probe run used to rule out a half-normal prior is likewise unsupported.
3. A Garki-project reading that `sigma_c` "wanders its prior along the
   non-centred neck", a funnel, and the request for a slice or Gibbs update on
   `sigma_c` built on it, rest on a run in which `sigma_c` never moved. A funnel
   may exist; that run cannot show it.

No downstream figures built from affected fits were found other than prior-only
figures. In this repository, `tests/fixtures/polio_afp_es/fit.toml` estimated
`kappa` boxed [0, 0.3] under IF2 with an automatic step; `0abe59f8` moved its
lower bound to 0.001. Its recovery test (`synthetic_fit_recovers_params`) does
not assert on `kappa`, on the stated ground that it is weakly identified, so a
frozen `kappa` could not have failed it.

## Remediation

1. **Done: refusal (`0abe59f8`).** `refuse_degenerate_auto_rw_sd` refuses, under
   `fit run` with IF2, PMMH or MH and in profile's IF2 cells, an automatic step
   for a log box with lower bound ≤ 0, and any automatic step that is
   non-finite, ≤ 0, or below 10⁻⁹ of the box width. The message names the
   parameter, the box, the step it would have used, and the two remedies (an
   explicit `rw_sd`, or a positive lower bound). No step is invented. Tests in
   `rust/crates/cli/tests/auto_rw_sd_needs_a_finite_box.rs` were red on the
   parent commit and green after, with mutation checks on each branch of the
   refusal.
2. **Pending: the step is defined in the transformed coordinate.** The
   support-keyed transforms proposal makes each transform the map onto its
   parameter's effective domain, and defines the default step in z per map
   family (D8), not on the natural scale at a midpoint. An explicit `rw_sd` is
   converted to z once rather than at each particle's value. This removes the
   conversion through a floored value that every member of the class went
   through. A start exactly on a bound, which has no finite z, is refused with
   an interior value to use instead (D3). The decisions were made on 2026-10-29;
   the implementation is in progress on `feat/support-transforms` and is not on
   main. Until it lands, the refusal is the only protection.

## Process lessons

**No test pinned the value of the automatic step for a [0, hi] box.** The
existing tests ran fits with interior boxes, where the derivation works, or gave
an explicit `rw_sd`. A unit test over representative boxes per transform — for
each kind and each box the language allows, the step at a typical value changes
the value when applied in z — would have failed on `4a90daac`. The general form
is a property test on the transform and step together:
`from_z(to_z(θ) + step_z) ≠ θ` for θ across the box, the same "test at the edge
of the domain, not the interior" gap recorded in
`2026-10-02-pow-autodiff-nan-at-zero-base.md`.

**The evidence was printed, but in a form that hid it.** The transform table
showed `rw_sd=0.0000 (0.000/step, auto)` and the diagnostics
`Auto rw_sd for
'beta': 0.000000` as an information line; four and six decimal
places make 10⁻¹⁴⁹ read as "small". IF2's chain-agreement line saw zero
within-chain variance for β and printed it as a dimmed
`Â=n/a (W ≈ 0; rely on Δ_dB)`, a rendering chosen (gh#45) so that an unassessed
parameter would not be reported as failing. A parameter whose trace has zero
variance across every chain is the signature of this defect, and should be an
error. A post-run check that every estimated parameter moved from its start, in
IF2 and in the samplers, would have caught every affected run, including the
downstream ones.

**A recovery test that excludes a parameter cannot see it freeze.** The polio
recovery test excluded `kappa` from its assertions as weakly identified. A
frozen parameter and a weakly identified one look alike to a test that does not
check whether the parameter moved.

**The documentation said `rw_sd` was IF2-only.** Until `ab0f9758` (2026-10-26),
`docs/fit-toml.md` described `rw_sd` as "IF2 per-parameter random-walk SD …
Ignored by every other method." PMMH and MH read it from their introduction. A
reader of a frozen PMMH chain, following the documentation, would rule out the
automatic step as a cause. Each config key's documented consumers should be
checked against the code that reads it; a test that greps the dispatch for each
`[estimate]` key and compares against the documented method list would have
caught this.
