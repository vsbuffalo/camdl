# PGAS `draws.tsv` double-applied burn-in/thin — half (or all) of the posterior silently dropped

Date: 2026-06-28 Severity: high (silent-wrong posterior output; misleads every
downstream read of `draws.tsv`) Status: fixed Fix: `fit/pgas.rs` draws.tsv
writer iterates the already-filtered sweeps; red→green regression
`tests/pgas_draws_count.rs`.

## What happened

`camdl fit run` with a PGAS stage persisted only a fraction of the posterior to
`draws.tsv`. The fit's own stderr contradicted itself: it announced the correct
retained count and then wrote half of it.

```
$ camdl fit run fit.toml --seed 1       # chains=1, sweeps=60, burn_in=20, thin=1
  estimated output: 40 posterior samples per chain     # fit/pgas.rs:450 (correct)
  draws.tsv: 20 posterior samples (all 6 params)        # fit/pgas.rs:1045 (actual)
$ tail -n +2 results/fits/*/01-posterior-*/seed_1-*/draws.tsv | wc -l
20                                                       # want 40
```

At `thin = 2` it is worse — the writer drops _every_ retained draw and emits an
**empty** `draws.tsv`:

```
$ camdl fit run fit.toml --seed 1       # sweeps=30, burn_in=10, thin=2  → 10 retained
  draws.tsv: 0 posterior samples                         # want 10  (downstream `fit predict` then fails)
```

The dropped draws are the _earliest_ post-burn-in draws (it behaves like a
second burn-in), so the persisted cloud is a biased tail of the chain, not a
random subsample.

## How it was detected

While scoping the keyed-joint `(θ, X)` proposal
(`docs/dev/proposals/2026-06-28-keyed-joint-param-trajectory-output.md`), the
draft flagged a _suspected_ double-thin at `fit/pgas.rs:1029`. Reading the code
plus a reproduction (the runs above) confirmed it: the fit's announced count
(`(n_sweeps − burn_in)/thin`) and the written count disagreed by exactly the
double-filter.

It had survived because no test asserted the posterior draw **count** — the
`fit predict` / `fit table --quantity` tests band over "whatever is in
`draws.tsv`," and a truncated-but-non-empty cloud still produces well-formed,
ordered bands. The truncation was invisible to every shape/ordering assertion.

## Root cause

The sim-side recorder (`sim/inference/pgas.rs:2702`) already applies burn-in and
thinning when it builds the sweeps it returns:

```rust
// Record (respecting burn-in and thinning)
if sweep >= config.burn_in && (sweep - config.burn_in).is_multiple_of(config.thin) {
    sweeps.push(sweep_result);
}
```

So `result.sweeps` is the _already-retained_ set. The `draws.tsv` writer
(`fit/pgas.rs`) then applied burn-in/thin a **second** time — but indexing the
already-retained list by position, not by sweep number:

```rust
for (i, sweep) in sweeps.iter().enumerate() {
    if i < burn_in { continue; }                       // drops the first burn_in RETAINED draws
    if !(i - burn_in).is_multiple_of(thin) { continue; } // thins the retained draws again
    ...
}
```

Every _other_ consumer of `result.sweeps` treats it as already-filtered and
iterates all of it — `compute_diagnostics` (`fit/pgas.rs:1084`) and the CSMC
diagnostics aggregation (`fit/pgas.rs:892`). So R̂/ESS were computed over the
full 40 retained draws while `draws.tsv` held 20: the persisted posterior was
desynced from its own convergence diagnostics. The writer was the lone
double-applier.

PMMH is unaffected: its `draws.tsv` writer reads each chain's already-filtered
`trace.tsv` and writes every row (verified: a `sweeps=60, burn_in=20, thin=1`
PMMH fit writes 40). The bug is PGAS-only.

## Remediation

The writer now iterates all of `sweeps` (they are already post-burn-in and
thinned); it no longer references `burn_in`/`thin`. One-block change in
`fit/pgas.rs`.

Blast radius of the bug (now closed): every PGAS fit's `draws.tsv` was
truncated, so everything that reads it operated on the smaller cloud —
`fit predict`'s predictive/quantity bands, `fit table --quantity` (which derives
via predict), posterior summaries, and the draws loader. R̂/ESS were _not_
affected (computed pre-write over the full set), so convergence reporting was
correct even while the saved cloud was wrong — which is what made it silent.

## Reproduction / regression

`tests/pgas_draws_count.rs` asserts `draws.tsv` row count equals the sim-side
retained count
`(burn_in..n_sweeps).filter(|s| (s − burn_in) % thin == 0).count()
× n_chains`,
for `thin = 1` (20 vs the buggy 10) and `thin = 2` (10 vs the buggy 0 — the
strong discriminator). Red against the pre-fix writer, green after.

## Process change

The blind spot was structural: no inference-output test asserted a **count**,
only shapes/orderings/values that a truncated cloud still satisfies. Going
forward, a persisted posterior artifact (`draws.tsv`, and the `(θ, X)` paired
output when it lands) should carry a count invariant pinned by a test —
truncation/duplication of draws is otherwise invisible to band-shape assertions.
