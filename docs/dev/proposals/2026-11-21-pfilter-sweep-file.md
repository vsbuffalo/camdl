# `pfilter --sweep-file`: one log-likelihood per design point

- Date: 2026-11-21
- Status: accepted; every decision in §8 is made, and the commits that implement
  it cite this proposal
- Relates-to: `2026-09-27-coordinate-kind-design-vs-sample.md` (accepted; D-B,
  D-C, D-E, the `--sweep-file` reader), `2026-11-20-run-store-api.md` (the
  `Pfilter` run kind, §"By-products of a filter run"), `docs/camdl-run-spec.md`
  §1.3

Code references are against `46f6a6b66`.

## Summary

`camdl pfilter` scores one parameter set per call: it runs a bootstrap particle
filter and reports the estimated marginal log-likelihood log p(y | θ), as a mean
over `--replicates` independent filter runs. This proposal gives it the design
axis `simulate` already has. With `--sweep-file design.tsv`, `pfilter` scores
every row of the file and writes one table row per design point.

```
camdl design lhs sir.camdl --fit fit.toml --n 200 --seed 1 -o design.tsv
camdl pfilter    sir.camdl --fit fit.toml --sweep-file design.tsv \
                 --particles 1000 --replicates 5 --seed 1 -o landscape.tsv
```

Each design point is scored exactly as the single call
`camdl pfilter sir.camdl --fit fit.toml --param beta=… --param gamma=… --particles 1000 --replicates 5 --seed 1`
would score it, and is stored as that call's ordinary `Pfilter` leaf. The design
adds no run kind, no identity field and no seed rule of its own.

Together with `camdl design lhs`, this covers the scoring half of what
`camdl survey` did.

## 1. What exists today

**The design file.** `simulate --sweep-file FILE` reads a TSV whose header names
model parameters and whose rows are design points, through
`sweep::read_sweep_file` (`rust/crates/cli/src/sweep.rs:118`). An optional
`point_id` column names each row; without it, a row's id is its 0-based position
(decision D-C of the coordinate-kind proposal). The reader refuses an unknown
parameter column, an empty file, a duplicate row (D-E), a non-numeric cell, and
a malformed or repeated `point_id`. The file enters the run by its content
digest. Its result is a `SweepFile { points, design: ResolvedDesign }`.

**How a design value reaches the model.** A design coordinate is resolver tier
3.5 (`docs/camdl-run-spec.md` §1.3), passed as `point_overrides` to
`params_resolver::resolve_parameters`. A scenario that sets or scales a swept
parameter, or a `--param` naming one, is refused rather than allowed to override
every point (coordinate-kind proposal, workflows W10 and W12).

**One pfilter call.** `pfilter::run_pfilter` (`pfilter.rs:163`) resolves the
parameters once (`pfilter.rs:393`), builds the process and observation model,
runs `bootstrap_filter` once or `--replicates` times, and writes one `Pfilter`
leaf through `write_cas_leaf` (`pfilter.rs:725`). The leaf's identity
(`pfilter_cas::PfilterCtx`) hashes the model, the data digests, the particle and
replicate counts, `dt`, the objective, the engine switches, the binomial
sampler, the resolved `(name, value)` parameters and the seed. Replicate `r`
runs on seed `seed + r · 0x9e3779b97f4a7c15` (`pfilter.rs:910-922`). When every
replicate collapses (every particle's weight is zero at some observation), the
call exits with status `FILTER_FAILED` and writes no leaf (`pfilter.rs:951`).

**`pfilter --fit` does not apply the fit's `[fixed]` table.** The resolver is
called with an empty `fit_toml_fixed` (`pfilter.rs:388`). Run spec §1.3 says
tier 2, "fit.toml [fixed]", applies to "inference commands, and --fit on
others". Verified on a two-parameter SIR fixture whose fit config has
`[estimate] beta` and `[fixed] gamma = 1.5`:

```
$ camdl pfilter model.camdl --fit fit.toml --param beta=0.8 --particles 200 --seed 1
error: parameter 'gamma' has no value: no scenario sets it and nothing supplies it.
```

The failure is loud, not a wrong number: a `.camdl` model compiles every
parameter without a value (§1.3, tier 1), so a parameter that only `[fixed]`
sets has no value at all. For a design workflow, though, it means every
`pfilter --sweep-file --fit` call would be refused unless the user restated the
fit's `[fixed]` table by hand. The implementation comment at
`pfilter.rs:213-217`, which says the fit's parameter effect "is already the
fit's `[fixed]` value", is also false.

## 2. Behaviour

`--sweep-file FILE` is read with `sweep::read_sweep_file` against the loaded
model. For each design point, in file order:

1. The point's values are passed to `resolve_parameters` as `point_overrides`
   (tier 3.5). All other tiers resolve as they do for a single call: the fit's
   `[fixed]` (§3), `--params`, the scenario, `--param`.
2. The filter runs `--replicates` times, on the same seeds a single call uses
   (§8, D1).
3. The point is stored as an ordinary `Pfilter` leaf. When that leaf already
   exists, the point is reported as cached and not recomputed (§8, D2).

Points run concurrently on the `--parallel` thread pool. Each row of the output
depends only on its point's inputs and seeds, so the output does not depend on
the schedule.

**Output table.** It goes to `-o PATH` or to stdout, one row per design point,
in file order:

| column          | meaning                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `point_id`      | the row's id (D-C)                                                                                |
| `sweep:<param>` | one per design column, the value scored; the same naming as `simulate`'s combined tables          |
| `loglik`        | mean over replicates of the estimated log-likelihood; `-inf` when any replicate failed            |
| `loglik_sd`     | standard deviation over replicates; empty when `--replicates 1` or when any replicate failed      |
| `n_failed`      | how many replicates failed (below)                                                                |
| `failure`       | empty when every replicate scored; else the first failed replicate's message, on one line         |
| `run_id`        | the stored leaf (`camdl cat <run_id>`); empty when every replicate failed and no leaf was written |

`loglik` is printed with shortest round-trip formatting, so a value read back
equals the stored one.

**Failed points.** A replicate fails at a design point in either of two ways: it
returns a log-likelihood of `-inf` (every particle's weight is zero at some
observation), or `bootstrap_filter` returns an error, such as
`PFDegenerate { EssCollapsed }` when the effective sample size stays collapsed
over several observations. Both are properties of the parameter values, since
every input the points share is read and checked before the first point runs, so
both are recorded in the row and neither stops the run. In the `failure` column,
tabs and newlines in the message are replaced by spaces, so the table stays one
row per point. A point whose every replicate fails has `loglik = -inf`,
`n_failed = R` and an empty `run_id`. After the last point the run prints how
many points failed entirely and the first failure message. It exits with
`FILTER_FAILED` only when every point failed, and with 0 otherwise.

A single call without `--sweep-file` keeps its present behaviour: a filter error
exits with `FILTER_FAILED` at once.

**Refused beside `--sweep-file`**, each with a message naming both flags:

- `--param` or a scenario that sets or scales a design column (the `simulate`
  rule, W10 and W12);
- every output that describes one filter run: `--trace`, `--pf-health`,
  `--save-final-state`, `--save-paths`, `--save-filtering`,
  `--save-filtered-streams` and `--save-prequential`. They are the outputs
  `check_per_run_outputs` (`pfilter.rs:32`) already refuses beside
  `--replicates N`; the same function gains the design case.

A parameter that no tier gives a value, such as an `[estimate]` parameter the
design file has no column for, is refused by the resolver's existing "has no
value" error before any filter runs. No new check is needed; a test pins it.

## 3. `pfilter --fit` applies the fit's `[fixed]` table

This is a prerequisite, landed as its own commit before the design axis.
`pfilter` passes the fit config's `[fixed]` table as `fit_toml_fixed`, as the
run spec states. Lower tiers still win over it: `--params`, the scenario and
`--param` override a `[fixed]` value, and the resolver reports each override.

What changes for users: `pfilter --fit` with a parameter set only in `[fixed]`
is scored instead of refused. A call that passed every value explicitly scores
the same parameters, and its leaf keeps its `run_id`, since the identity hashes
resolved values, not where they came from. The false comment at
`pfilter.rs:213-217` is corrected in the same commit.

## 4. Types and seams

There is no new type in `runid` and no new run kind. In `cli`:

```rust
// args/mod.rs, PfilterArgs
/// A TSV of design points (header: parameter names, optional `point_id`);
/// one log-likelihood row per point.
#[arg(long, value_name = "FILE")]
pub sweep_file: Option<PathBuf>,
```

`run_pfilter` currently resolves parameters, runs the filter and writes the leaf
in one body. It splits at the parameter point:

```
before:  run_pfilter(config) = resolve θ → build process → filter × R → leaf

after:   run_pfilter(config)
           ├─ no design:  score_point(&shared, point_overrides = [])     → print, leaf
           └─ design:     for each row (parallel):
                            score_point(&shared, point_overrides = row)  → table row, leaf

         shared = everything resolved once: model load, data binding,
                  dt, objective, observation model, engine, seed
         score_point(shared, overrides) -> PointScore {
             loglik: f64, loglik_sd: Option<f64>, n_failed: usize,
             failure: Option<String>, leaf: Option<ContentHash>,
         }
```

`score_point` runs the parameter resolution, the filter replicates and the leaf
write, so the single-call path and the design path share one implementation, and
the equality the tests check (§7) holds by construction. The parameters that
`point_overrides` can change, and the compiled model built from them, are per
point. Everything else is built once.

## 5. Run identity and the store

Each design point's leaf has exactly the identity of the single call at that
point: `PfilterCtx` hashes the resolved `(name, value)` list, and a value that
came from the design file is the same number as one passed with `--param`. The
design file's digest is not part of any leaf's identity. Consequences:

- a design run interrupted after k points, and run again, computes only the
  remaining points;
- a point already scored by a single `pfilter` call is reused by a design that
  contains it, and the reverse;
- two design files that share rows share those rows' leaves.

The output table itself is not stored. It is a deterministic function of the
design file, the flags and the stored leaves, so a `make` rule that names
`-o landscape.tsv` regenerates it byte-identically.

For the store migration, this adds nothing: the design is a loop over
`PfilterIdentity` leaves that the migration already plans
(`2026-11-20-run-store-api.md`, `PfilterIdentity`). A thousand-point design puts
a thousand directories in one parameters level, which that proposal already
notes for `simulate` sweeps.

## 6. Documentation and release note

`docs/commands.md` and `docs/camdl-run-spec.md` (beside `simulate
--sweep-file`)
describe the flag, the table and the refusals. The workflow page that the survey
removal points at `camdl design lhs` gains the scoring step. Release note, under
the survey removal's entry:

> `camdl pfilter --sweep-file FILE` scores every row of a design file and writes
> one log-likelihood row per point; with `camdl design lhs` it replaces the
> landscape `camdl survey` produced. `pfilter --fit` now applies the fit
> config's `[fixed]` values; a parameter set only there was refused before.

## 7. Tests

Each red first, then green, and mutation-checked.

1. `pfilter --fit` scores a parameter set only in `[fixed]`, and the result
   equals the call with that value passed by `--param` (bit-equal `loglik`,
   equal `run_id`). Red on `46f6a6b66` with the error quoted in §1.
2. For a three-row design, each row's `loglik`, `loglik_sd` and `run_id` equal
   the single call at that row's values, with `--replicates 1` and
   `--replicates 3`.
3. A second run of the same design reports every point cached; a design whose
   first row was scored by a single call computes only the other rows.
4. `-o` output is byte-identical between `--parallel 1` and `--parallel 4`.
5. Failed points, one test per way of failing. A row placed where every
   replicate scores `-inf`, and a row placed where the filter returns
   `PFDegenerate`, each read `-inf`, `n_failed = R`, a non-empty one-line
   `failure` and an empty `run_id`, while the other rows are scored and the run
   exits 0. A design where every row fails exits `FILTER_FAILED`.
6. The refusals of §2: `--param` on a design column, a scenario setting one,
   each per-run output, and an `[estimate]` parameter with no column.
7. End to end: `camdl design lhs` then `camdl pfilter --sweep-file` on the same
   fixture.

## 8. Decisions

**D1. A design point does not enter the seed.** Every point runs on the seeds a
single call with the same `--seed` uses, so the filters at different points
share random numbers. Survey did the opposite and mixed the point index into
each seed (`survey.rs:1627`, `util::mix_cell_seed`). The choice here follows D-B
of the coordinate-kind proposal, which `simulate --sweep-file` already obeys,
and it is what makes each row reproducible by one plain `pfilter` call and
reusable from the store (§5). Shared random numbers should also make the
difference in `loglik` between neighbouring points less noisy than independent
seeds would. That last part is inferred, not measured: resampling in a bootstrap
filter is discontinuous in θ, so how much shared numbers help is an empirical
question, and test 2 does not settle it. The cost is that the errors at
different points are correlated, so the table is not a set of independent
estimates. Recommendation: solid.

**D2. Each point is an ordinary `Pfilter` leaf; the table is not stored.** The
alternative is a stored design-level record, like `simulate`'s `SimEnsemble`
grid level, holding the table and the design digest. That would make the table
addressable by one run id, at the price of a new identity type now and another
kind for the store migration to carry. Per-point leaves give resume and reuse
for free and add nothing to the migration. Recommendation: solid.

**D3. A failed point is a row, not a failed run.** In a space-filling design
some points are expected to be infeasible, and survey's feasibility tally
existed to count them. At most infeasible points the filter does not return
`-inf`; it stops with `PFDegenerate { EssCollapsed }` before the last
observation. Verified on the implementation branch before this rule was widened:
a two-point design whose second point was infeasible exited with status 3 after
storing point 0 and wrote no table. Ending the run there would make large
designs unusable, so both ways of failing become rows. The run fails only when
no point was scored, which keeps the single call's meaning of `FILTER_FAILED`.
The `failure` column keeps the first message per point, which restores survey's
split by cause in readable form without a fixed list of cause codes.
Recommendation: solid.

**D4. `--sweep-file` only, not `--sweep`.** `--sweep beta=0.2,0.3,0.4` would
reuse the same expansion (`sweep::expand_sweep_specs`) and give a likelihood
slice along one axis at no extra design cost. It is left out to keep this change
to what replaces survey, and because `profile --sweep` already exists with a
different meaning (it maximises over the other parameters at each point, where a
`pfilter` slice holds them fixed), so the two would need documenting side by
side. Adding it later changes no identity. Recommendation: leaning.

## 9. Not in scope

**How replicates are combined.** `loglik` is the arithmetic mean of the
replicates' log-likelihoods (`pfilter.rs:957`). A bootstrap particle filter's
likelihood estimate is unbiased on the natural scale, not the log scale, so the
mean of logs is biased below log p(y | θ) by Jensen's inequality, and the bias
grows with the variance between replicates. pomp combines replicates with
`logmeanexp`, the log of the mean of the likelihoods (King, Nguyen and Ionides
2016, _Journal of Statistical Software_ 69(12), doi:10.18637/jss.v069.i12;
pomp's `logmeanexp` reference page). The design path reuses the single call's
combination so that rows equal single calls; whether to change both is a
separate issue, to be filed with this proposal. Which combination survey used
was not checked.
