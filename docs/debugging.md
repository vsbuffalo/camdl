# Debugging and Inspection

Tools for inspecting what the simulator computes without guessing.

---

## `camdl dev eval` — Evaluate Expressions at a Time Grid

Evaluate time-dependent expressions without running a simulation. No compartment
state, no RNG, no trajectories. Useful for inspecting forcing curves,
covariates, and parameter-derived quantities.

### Usage

```bash
# Forcing function over one year
camdl dev eval model.ir.json --params p.toml --expr "school" --from 0 --to 365 --every 1

# Multiple expressions
camdl dev eval model.ir.json --params p.toml --expr "school,R0,gamma" --from 0 --to 730 --every 7

# Specific time points
camdl dev eval model.ir.json --params p.toml --expr "school" --at 0,100,200,300,365

# Parameter override
camdl dev eval model.ir.json --params p.toml --expr "school" --from 0 --to 365 --every 1 --param amplitude=0.8
```

### Output

TSV to stdout. First column is `t`, remaining columns are the requested
expressions:

```
t       school
0       0.000000
7       0.000000
14      1.000000
21      1.000000
...
```

### What's Evaluable

Anything that depends only on `t`, parameters, and forcing functions:

- **Forcing functions**: `school`, `seasonal`, `pop_trend`
- **Parameters**: `R0`, `gamma`, `sigma_se`
- **Math on time**: `exp(-mu * t)` (via inline expressions, future)

### What's NOT Evaluable

Expressions referencing compartment populations:

```bash
camdl dev eval model.ir.json --params p.toml --expr "S"
# error: expression 'S' references compartment state.
#   Compartment values require a running simulation.
#   Run 'camdl simulate <model> -o traj.tsv' instead (writes compartment and flow_* columns per step).
```

### Workflow: Comparing Covariates

To validate that camdl's cubic spline matches pomp's `smooth.spline()`:

```bash
# Dump camdl's interpolated population at weekly points
camdl dev eval model.ir.json --params p.toml --expr "pop" --from 0 --to 7665 --every 7 > camdl_pop.tsv

# Compare against pomp output in R/Python
```

---

## `--trace` — Named Quantities During Simulation _(planned)_

A planned `--trace` flag would emit forcing function values and let binding
evaluations as additional TSV columns alongside trajectory output. Useful for
debugging unexpected dynamics by seeing what the simulator computed at each
step.

A normal simulation already writes the cumulative `flow_*` columns next to the
compartment columns; pass `-o` to mirror the trajectory to a file:

```bash
camdl simulate model.ir.json --params p.toml --backend chain_binomial --dt 1 --seed 42 -o traj.tsv
```

```
t   t_start  t_stop  S       E     I     R     flow_infection  flow_progression  flow_recovery
0   0        0       73151   127   127   2.4M  0               0                 0
1   0        1       73080   198   127   2.4M  71              0                 0
```

`t_start`/`t_stop` is the period each row's `flow_*` columns were accumulated
over; the compartment columns are read at `t`.

The planned `--trace` flag would add the remaining traced columns (forcing
functions and let bindings, e.g. `school`, `beta_base`). It is **not yet
implemented**. Until then, use `camdl dev eval` for time-dependent quantities
and post-hoc trajectory analysis for state-dependent quantities.

## A model that is slow to compile: `CAMDL_TIME_PASSES`

`CAMDL_TIME_PASSES=1` makes the compiler print how long each pass took, to
stderr, at the end of a compile:

```
$ CAMDL_TIME_PASSES=1 camdlc model.camdl -o /dev/null --no-state-grad
[camdl pass timing]  (processor seconds, CAMDL_TIME_PASSES)
  expand           1.131 s   31.6%
  autodiff         0.374 s   10.5%
  ...
  serialize        1.990 s   55.6%
  TOTAL            3.578 s  100.0%
```

Compare a compile with and without `--no-state-grad`. Without it, `camdlc` also
builds the state Jacobian (`rate_state_grad`, each rate's derivative with
respect to each compartment). Only a gradient fit on the ODE backend
(`nl-lbfgs`) reads it, and its cost grows with transitions × compartments, so on
a large stratified model it can take minutes where everything else takes
seconds. `camdl simulate` and `fit` with every other method compile without it,
and so does `camdl check`.
