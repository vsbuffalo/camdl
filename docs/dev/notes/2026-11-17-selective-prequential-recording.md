# Selective prequential recording in the forecast driver

- Date: 2026-11-17
- Proposal: `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`,
  D84

## What changed

The bootstrap filter's prequential record holds, at every stop of the
observation axis, each particle's joint log-likelihood, its per-stream
log-likelihoods and its per-stream predictive draws: `(2·S + 1)·N` values for
`S` streams and `N` particles. The fixed-origin forecast
(`rust/crates/cli/src/fit/forecast_driver.rs`) asked for that record at every
stop and read it only after the origin: the predictive draws at the targets, and
the log-weights at each post-origin stop, where a non-finite value is a particle
that died. The driver now calls `bootstrap_filter_recording_at` with
`PrequentialStops::only(<stops after the origin> ∪ <target stops>)`, so the
training period, which is most of the axis, keeps no record.

The filter still makes the predictive draw at every stop and discards it at a
stop it does not keep, so the diagnostic random-number streams advance as under
a full recording and the records kept are bit-identical to it.
`rust/crates/sim/tests/prequential_record_at_stops.rs` pins that, and fails when
the draw is skipped at unselected stops.

## Measurement

**Fixture.** An endemic two-patch SIRS (100 000 per patch; infection at β·S·(I +
ι)/N with a constant import ι, recovery γ·I, waning ω·R), on the chain-binomial
backend at one-day steps, with one daily negative-binomial incidence stream per
patch. Eight years (2920 days) were simulated at β = 0.3, γ = 0.2, ω = 0.005, ι
= 2, ρ = 0.5, k = 10 (seed 3), and fitted by a short PMMH run on β with training
rows through day 2890. The fit's quality is irrelevant here; it supplies a
posterior cloud for `fit predict`. The forecast is
`--horizon "30 days" --n-draws 4`: 60 targets on 2 streams, 4 draws × 500
particles, over a 2920-stop axis.

**Result.** One run of each build on an M4 Max, read from `/usr/bin/time -l`:

| build                         | maximum resident set size | peak memory footprint |
| ----------------------------- | ------------------------: | --------------------: |
| before (records at all stops) |                   89.7 MB |               74.1 MB |
| after (records at read stops) |                   29.8 MB |               16.9 MB |

The 57.2 MB drop in peak footprint is the record the training period no longer
keeps: `(2·2 + 1) · 500 · 8 B · 2890 stops = 57.8 MB`. The draws run one after
another, so this is the peak per draw. The saving grows with streams, particles
and the length of the training period: for 25 patches it is about 0.6 GB per
draw at 500 particles over the same eight years.

Every file the forecast wrote is byte-identical between the two builds (`cmp`):
`forecast/cases.tsv` (sha256 `d102644b…`), `forecast-samples.tsv` (sha256
`669e5fc2…`), `forecast.json` and `report.json`.

## Reproduce

In an empty directory, with `model.camdl` and `fit.toml` below and a release
build (`make build-rust`) as `$CAMDL`:

```sh
export CAMDL_SKIP_VERSION_CHECK=1
$CAMDL simulate model.camdl --param beta=0.3 --param gamma=0.2 --param omega=0.005 \
  --param iota=2 --param rho=0.5 --param k=10 --seed 3 --obs-only cases.tsv
awk -F'\t' 'BEGIN{OFS="\t"; print "time","patch","cases"} NR>1{print $1,"a",$2; print $1,"b",$3}' \
  cases.tsv > cases_long.tsv
$CAMDL fit run fit.toml --seed 1
/usr/bin/time -l $CAMDL fit predict --fit fit.toml --horizon "30 days" --n-draws 4
```

For the comparison, copy the directory after `fit run` and run the last line in
each copy with the build before and after the driver change, then `cmp` the
files under `results/fits/*/predict/*/`.

`model.camdl`:

```
time_unit = 'days

dimensions { patch = [a, b] }

compartments { S, I, R }

stratify(by = patch)

parameters {
  beta  : rate        in [0.05, 2.0]
  gamma : rate        in [0.01, 1.0]
  omega : rate        in [0.0001, 0.1]
  iota  : positive    in [0.0, 100.0]
  rho   : probability in [0.0, 1.0]
  k     : positive    in [0.1, 100.0]
}

let N[p in patch] = S[p] + I[p] + R[p]

transitions {
  infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * (I[p] + iota) / N[p]
  recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
  waning[p in patch]    : R[p] --> S[p]  @ omega * R[p]
}

observations {
  cases[p in patch] {
    columns       { time : time, patch : dim, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection[p])
    emit_schedule = every 1 'days
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
}

init {
  S[a] = 60000
  I[a] = 200
  R[a] = 39800
  S[b] = 70000
  I[b] = 50
  R[b] = 29950
}

simulate {
  from = 0 'days
  to   = 2920 'days
}
```

`fit.toml`:

```toml
output_dir = "results"

[model]
camdl = "model.camdl"

[data]
holdout_after = 2890.0

[data.observations]
cases = "cases_long.tsv"

[estimate]
beta = { bounds = [0.1, 0.6], start = 0.3, prior = { uniform = { lower = 0.1, upper = 0.6 } } }

[fixed]
gamma = 0.2
omega = 0.005
iota  = 2.0
rho   = 0.5
k     = 10.0

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 20
burn_in = 10
thin = 1
```
