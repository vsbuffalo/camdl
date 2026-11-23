# Synthetic-data parameter-recovery harness

Self-contained recovery cases: **plant θ, simulate, fit, recover.** A case
declares ground-truth parameters, simulates observation streams at those values,
then fits with IF2 / PGAS / PMMH so you can compare θ̂ against the planted truth.
The comparison is manual today (see Status below) — the harness runs the fits
and stores the results; it does not yet assert recovery.

Sibling to [`tests/external/`](../external/README.md). The external harness
compares camdl against _external_ references (pomp, Stan, NumPyro, analytical);
this one needs no external tooling — the ground truth is camdl's own synthetic
data. The two are complementary slices of "is the inference right."

**Status: manual.** Today you run a target and eyeball θ̂ against `truth.toml`.
The planned successor is a `cargo test` harness that asserts recovery within a
Monte-Carlo tolerance (the same `expected.toml` + power-rationale discipline
`tests/external` uses). Until then this also serves as the standing place to
**watch the progress bars** across every fitter.

## Layout

```
tests/recovery/
  Makefile                 # synth / if2 / pgas / pmmh / ensemble / all / clean
  cases/
    seir_age/
      model.camdl          # the model, priors inline
      truth.toml           # planted ground-truth parameters (the recovery target)
      if2.toml             # IF2 fit config
      pgas.toml            # PGAS fit config
      pmmh.toml            # PMMH fit config
      data/                # generated obs streams (gitignored)
      results/             # CAS run output (gitignored)
```

Committed: the **recipe** (model + truth + fit configs). Gitignored: everything
generated (`data/`, `results/`) — deterministic from the recipe + seed.

## Running

```bash
make -f tests/recovery/Makefile synth      # simulate at truth.toml → obs streams
make -f tests/recovery/Makefile if2        # IF2 fit     — per-chain ll bars
make -f tests/recovery/Makefile pgas       # PGAS fit    — per-chain ll bars
make -f tests/recovery/Makefile pmmh       # PMMH fit    — per-chain ll + acc bars
make -f tests/recovery/Makefile ensemble   # 30-cell forward sim — the cells bar
make -f tests/recovery/Makefile all        # synth + the three fits
make -f tests/recovery/Makefile clean
```

Append `--progress plain` to any target for the plain/CI bar form, or
`--no-progress` to silence. `make -f tests/recovery/Makefile help` lists the
targets. The fits depend on `synth`, so a bare `make … if2` generates the data
first if it is missing.

Uses `cargo run` (dev build), so it works from a fresh checkout with no install
step. Inspect the fits afterward with the normal reader:

```bash
camdl list --store tests/recovery/cases/seir_age/results
camdl fit table tests/recovery/cases/seir_age/results
```

## The seir_age case

Two-age-group SEIR (`child`, `adult`) mixing through a `2×2` contact matrix,
weekly per-age case streams via a negative-binomial observation model. `beta`,
`sigma`, `gamma` are estimated against the three planted rates in `truth.toml`;
`rho` and `k` are held fixed. Small enough to fit in seconds, structured enough
to exercise stratified expansion, contact-matrix `TableLookup`, and per-stratum
observations.

## Adding a case

Create `cases/<name>/` with `model.camdl`, `truth.toml`, and a fit config per
algorithm you want to exercise, then run
`make -f tests/recovery/Makefile
CASE=tests/recovery/cases/<name> all`.
