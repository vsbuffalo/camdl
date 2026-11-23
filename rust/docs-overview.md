# Overview

The runtime is the Rust half of camdl: the `camdl` command, which reads the JSON
a compiled model is written in, simulates it, fits it to data and stores every
result under the hash of its inputs. It is one cargo workspace, `rust/`, of
eight crates documented here. The compiler, `camdlc`, is a separate OCaml
program that `camdl` runs as a subprocess.

A command runs these steps:

<!-- figure: internals-runtime-pipeline -->

```
model.camdl
    |   camdlc (OCaml), run as a subprocess
    v
IR envelope (JSON) ------> ir::envelope, ir::model    the compiled model
    |                      declared                   the model as written
    |   cli: parse the command line, resolve parameters and data
    v
sim::compiled_model        names resolved to indices, expressions compiled
    |   sim backends (Gillespie, chain-binomial, ODE)
    |   sim::inference (particle filter, PGAS, PMMH, IF2, NUTS)
    v
results ------------------> store   each result, keyed by a hash of its inputs
                            io      trajectories as TSV
                            score   model comparison
```

<!-- /figure -->

## The compiled model {#model}

`ir` holds the types of the compiled model and of the envelope it arrives in,
with their serde codecs and no simulation logic. `declared` holds the model as
its author wrote it, before expansion; it depends on nothing in the workspace,
so a reader of a published declared-model document needs only it.

<!-- modules: ir declared -->

## Run identity and output files {#identity}

The `store` crate, not documented here, keys every stored result by a hash of
its complete, typed input set and names the result's folder from it.
`ir::model_hash` is the model's structural hash, by which `camdl` compares and
prints models. `io` writes the trajectories that simulation and inference
produce, in one long TSV format.

<!-- modules: io -->

## Numerics and scoring {#scoring}

`numerics` holds special functions shared by the engine and the external
validation harness. `score` computes a model comparison from values, reading no
files and starting no processes.

<!-- modules: numerics score -->

## Simulation and inference {#engine}

`sim` compiles a model for evaluation, runs it under each backend, and holds
every inference method and observation model. Its crate root has no
documentation yet; the table of modules on its page is the way in.

<!-- modules: sim -->

## Comparing model structure {#diff}

`diff` decides what differs between two models from their declared sections.

<!-- modules: diff -->

## The camdl command {#cli}

`cli` is the `camdl` binary: argument parsing, reading inputs, choosing a
backend or method, and writing results to the store. Its pages are listed under
the package name, `cli`; inside the code the crate is `crate`.

<!-- modules: cli -->
