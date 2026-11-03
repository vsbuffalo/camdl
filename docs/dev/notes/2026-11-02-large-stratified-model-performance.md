# Large stratified models: an annual-event bottleneck before the tensor IR

Date: 2026-11-02 Project: camdl (compiler / runtime) Tags: performance, ODE,
stratification, events, compilation, inference Status: investigation with
implementation follow-up below

## Context

The target is fitting compartmental models with 10,000–100,000 cells, dense
coupling, and 10,000–100,000 likelihood evaluations. At that scale, both the
representation of the model and the work repeated inside each solve matter. A
faster compiler helps iteration on the model; a faster likelihood determines
whether fitting it is practical.

This investigation started from a diagnosis of large JSON IR, repeated CLI
loading, tree-walking rate evaluation, and expensive state-Jacobian generation.
Those are real concerns, but the larger benchmark exposed another one:
**checking simultaneous annual events for conflicting assignments is quadratic
in the number of actions.** A short profile caught that check occupying about
two-thirds of the main thread's samples.

The highest-ROI next change is therefore a bounded Rust optimization: make that
conflict check linear while preserving its decisions and diagnostics. It needs
neither a new solver nor an IR schema change. Its actual whole-run benefit still
needs a controlled before/after benchmark.

## Models and experimental setup

The real-model benchmark is a TB model, examined in three configurations:

- A baseline with 738 cells.
- A larger, single-arm configuration with 4,428 cells and 11,316 transitions.
- A three-arm configuration with 13,284 cells and 33,948 transitions.

The model combines time-varying dense contact coupling with annual indexed
assignment events. Those events are important to the performance result. Private
model inputs and scientific details are deliberately omitted; the synthetic
control below is independently reproducible.

The machine was an M4 Max. The installed runtime reported `0.1.0+2a3933de`; the
source examined was `2a3933dee41eec2cc738a865ea843dbc5f79992f`. All experiments
used existing binaries, with one heavy job at a time. No builds, gates, or
implementation changes were made during the investigation. In particular, no
large-model state-Jacobian compile was attempted.

Models and benchmark artifacts were kept in a session scratch directory.
Relative data references in the copied models were adjusted to preserve their
meaning. Private inputs are not included in this note.

### Commands

From the scratch directory, for each of `baseline`, `one`, and `three`:

```bash
CAMDL_NO_IR_CACHE=1 /usr/bin/time -l env CAMDL_TIME_PASSES=1 \
  camdlc baseline.camdl -o baseline.json --no-state-grad

/usr/bin/time -l camdl simulate baseline.json \
  --params params.toml \
  --backend ode --to 2023 --stdout > /dev/null
```

Each configuration used the same fixed parameter point, with its additional
parameters supplied where applicable. The private-model commands above are
schematic: `baseline.camdl` and `params.toml` stand for local inputs.

The copied models specify RK45 with `atol = 1`, `rtol = 1e-7`. Timing from the
already compiled JSON excludes compilation but includes CLI startup, loading,
integration and formatting TSV to `/dev/null`. `--stdout` avoids store writes.
These are single observations on a shared machine, not interleaved repeated
benchmarks under controlled load.

An initial baseline invocation incorrectly included configuration-specific
parameters and failed. Its 2.88 s is excluded. An initial sandboxed `time -l`
could not read `kern.clockrate`; the successful resource measurements were rerun
with the necessary permission.

## Compile and simulate results

| Model      |  Cells | Compile wall s | Compile max RSS GB | JSON MB | Simulate wall s | Simulate max RSS GB |
| ---------- | -----: | -------------: | -----------------: | ------: | --------------: | ------------------: |
| Baseline   |    738 |           4.03 |              1.217 | 113.935 |            7.33 |               1.681 |
| One arm    |  4,428 |           7.50 |              1.088 | 115.601 |           17.05 |               1.697 |
| Three arms | 13,284 |          19.43 |              1.542 | 154.206 |           82.40 |               2.139 |

MB and GB are decimal. RSS is `/usr/bin/time -l`'s maximum resident set size,
not its separate peak-footprint statistic. The three-arm run includes the
sampling perturbation described below.

The baseline compiler reported 3.951 processor seconds:

| Pass               | Processor s | Share |
| ------------------ | ----------: | ----: |
| Expansion          |       1.234 | 31.2% |
| Parameter autodiff |       0.425 | 10.8% |
| Serialization      |       2.196 | 55.6% |

The remaining passes account for the rest. This reproduces the earlier finding
that serialization dominates the lean baseline compile. It also shows why cells
alone are a poor predictor: going from 738 to 4,428 cells barely changes JSON
size because the common forcing payload is already large.

The runtime grows 4.83× from one arm to three arms. It would be premature to
attribute that to rate interpretation or solver steps alone.

## The surprise: annual-event conflict detection

While the three-arm solve was running:

```bash
sample 97783 3 -f three.sample.txt
```

The sample recorded 2,269 main-thread samples. Of those, 1,500 were under
`sim::effects::refuse_coincident_sets`, called by `apply_event_phases`: **66.1%
of this sampling window**. This is an inclusive stack attribution, not leaf-only
self-time and not a whole-run profile.

The source explains why that function can become expensive. In
`rust/crates/sim/src/effects.rs:409`, it first constructs a list of every
compartment touched by a `set` or a transfer. For each first `set` target it
then:

1. Scans the preceding list to determine whether that target was already seen.
2. Scans the whole list to collect the target's set-event names.
3. Scans the whole list again to find a transfer touching the target.

With E simultaneous disjoint sets, the legal, conflict-free case is O(E²).
Annual per-cell assignment families make E proportional to the number of cells.
This work is unrelated to the contact matrix or the number of ODE stages.

There is a second issue nearby. `eval_amount_f64` at `effects.rs:740` allocates
an integer-state-sized placeholder and clones the real state for each action.
The continuous event phase calls it inside its action loop (`effects.rs:920`).
That introduces O(E·C) buffer initialization/copy work for C cells. The sample
also contains allocation underneath this path, but does not establish its
whole-run share.

The conflict check itself is necessary. Simultaneous sets and transfers read a
shared phase snapshot; combining a set with another drain or set on the same
target can produce a value neither event requested. The optimization must keep
that refusal. Removing validation would be a correctness regression, not a
performance fix.

### Proposed first fix: index touches once

Replace the repeated scans with a per-target summary built in one pass over the
due actions. A borrowed-key hash map is sufficient for an initial
implementation; pre-resolved compartment indices are another option if the
existing types make that straightforward. Do not require a new serialized
representation.

Keep a separate list of targets in first-set order. For each target, retain its
ordered set-event names and the first transfer touching it. Validate targets in
that list's order, checking multiple sets before set/transfer conflicts, as the
current implementation does. Preserve the current consecutive-name deduplication
behavior rather than silently substituting a different definition of duplicate.
Hash-map iteration order must not determine which error is reported.

This makes the normal valid batch expected O(E), with O(E) temporary storage.
The no-set path should remain cheap, and there is no reason to allocate a tiny
vector for every individual touch. The change should be limited to conflict
checking first. Buffer reuse in action evaluation is a separate follow-up so its
benefit and correctness can be assessed independently.

This is a small-to-medium implementation with a much narrower correctness
surface than code generation or tensor IR. It changes no rate arithmetic, event
schedule, state update, RNG consumption, or OCaml↔Rust schema. It should require
no golden regeneration.

### Expected benefit and acceptance evidence

There is no measured speedup yet. If the 66% share were representative of the
whole run, eliminating it would give an upper bound of roughly
`1 / (1 - 0.66) = 2.9×`. The short sample can overrepresent annual-event work,
so that is a conditional ceiling, not a promised result. The case for
prioritizing the change is the combination of a hot profile and a clear
quadratic algorithm.

Before claiming success:

- Benchmark disjoint simultaneous sets at increasing E, independently of RHS
  size. Doubling E should approach 2× validation time rather than 4×.
- Compare the existing checker and replacement on ordered batches containing
  sets, adds, fraction transfers and absolute transfers, including repeated
  targets, repeated names and several possible conflicts. Compare errors as well
  as success, including diagnostic order.
- Exercise the existing collision coverage in `effects.rs` and
  `tests/cross_backend_lifecycle_agreement.rs`, across continuous/discrete and
  integer/real state handling. Adds must remain allowed beside sets because they
  belong to an earlier phase.
- Run interleaved repeated before/after TB simulations, compare trajectories
  byte-for-byte, and profile the full solve. Follow the repository's normal gate
  before landing an implementation; none was run for this note.

For the allocation follow-up, reuse a context against the immutable phase
snapshot. Its lifetime must not cross a state-changing phase. Introducing a
binding cache there would be an additional semantic lifetime decision and should
not be smuggled into buffer reuse.

## A separate control: increasing the dense coupling dimension

The real-model experiment increases additional strata and arms while holding the
coupled dimension fixed. To vary the coupled dimension instead, the repository's
synthetic SEIR generator was run with P patches and four age groups:

```bash
python3 scripts/gen_scaling_models.py -P 32 -A 4 \
  --coupling on --grad full -o syn32.camdl

# From the scratch directory; repeat for P = 8, 16, 32, 64, 128.
/usr/bin/time -p env CAMDL_NO_IR_CACHE=1 camdlc syn32.camdl \
  --no-state-grad -o syn32.json
/usr/bin/time -p camdl simulate syn32.json --scenario baseline \
  --backend ode --integrator rk4 --dt 1 --stdout > /dev/null
```

| P   | Cells | JSON MB | Compile s | Simulate s |
| --- | ----: | ------: | --------: | ---------: |
| 8   |   128 |   0.348 |      0.01 |       0.05 |
| 16  |   256 |   1.079 |      0.02 |       0.06 |
| 32  |   512 |   3.709 |      0.09 |       0.21 |
| 64  | 1,024 |  13.641 |      0.43 |       0.73 |
| 128 | 2,048 |  55.750 |      4.21 |       2.79 |

These are successful single runs, with 0.01 s timing resolution. The generator's
minimal-gradient variant failed dimensional checking, so the retained runs use
full parameter gradients and its declared baseline scenario.

From P=32 to 128, cells increase 4×, JSON grows 15.0× and whole-run time grows
13.3×. That supports approximately quadratic expanded dense-coupling cost over
this range. Compile time grows 46.8×; these data do not justify a simple
quadratic compile-time fit. Allocation, garbage collection and derivative work
need separate profiling.

The generator's uniform off-diagonal contact matrix is algebraically special.
These measurements characterize its current expanded execution, not a lower
bound for a general matrix operation. They also include loading and output, so
dividing by the nominal RK4 stage count would not measure isolated RHS latency.

## What the findings change about the broader roadmap

There are several distinct scaling axes: cells, transitions, contact dimensions,
forcing knots, solver stages, output times and simultaneous event actions.
Optimizing one does not automatically improve the others.

For the larger model, shared age aggregates and contact reductions suggest a
forward-evaluation cost of roughly O(AVM + A²M + T), where A is ages, V
additional strata, M arms and T transitions. Growing arms at fixed A differs
from growing A. Arbitrary dense coupling across 100,000 cells still entails a
100,000² matrix: 80 GB for one f64 matrix alone. A compact IR does not eliminate
that data bound.

Adaptive steps further confound cell-based comparisons. `ode.rs:705–743` forms
an RMS error across compartments. Adding empty compartments changes the
averaging denominator and can change step counts. This investigation did not
measure RHS counts for the larger runs; their time per RHS remains unknown.

### Shared expressions before native code generation

Parameter-free bindings are already cached per ODE stage (`ode.rs:124`).
Parameter/table-only loop-invariant code motion is also already implemented and
on by default (`ocaml/lib/ir/licm.ml`). Neither is a new proposed win.

The remaining distinction is parameter-and-state-dependent sharing.
`expander.ml:3403` excludes parameter-bearing lets from ordinary binding
hoisting, because the parameter-autodiff handling treats a binding reference as
constant. Changing that hoist predicate alone would silently corrupt gradients.
A safe implementation needs either differentiation through shared expressions or
common-subexpression elimination after differentiation, with correct stage-local
cache lifetimes. Post-differentiation sharing helps runtime but cannot undo all
earlier expansion costs.

This is already covered by the
[ODE per-iteration performance draft](../proposals/2026-07-12-ode-per-iteration-performance.md).
The existing [LICM benchmark](2026-06-22-licm-kernel-benchmark.md) and
[binding-cache investigation](2026-06-07-runtime-binding-cache.md) document
landed work. The latter also records a failed cache-access optimization:
apparently removable profile attribution did not translate into a speedup. That
is a useful warning against treating the event sample as a completed
before/after result.

A flat bytecode evaluator already exists (`flat_eval.rs`), but the ODE RHS calls
`eval_resolved` directly (`ode.rs:148`). Simply setting `CAMDL_EVAL_FLAT` is not
an ODE evaluator benchmark. Wiring and measuring the existing evaluator should
precede a new JIT. Closures or bytecode are not guaranteed to beat the tree
walk; repeated expression work may matter more than dispatch.

### Loading versus fitting throughput

The earlier baseline investigation reported about 3 s of startup and repeated
parsing of the same large IR. This session's 7.33 s baseline whole-run timing is
consistent with that report but does not independently count the parses.
`util.rs` contains several deserialization paths, including `load_model`,
`load_run_model_and_anchors` and `rematerialize_with_output_every`.

Parse-once loading is worthwhile for startup and memory. It is not a repeated
MCMC speedup: `compute_ode_loglik` receives a prebuilt model and observation
model. More directly relevant there, `ode_loglik.rs:39–49` integrates to the
simulation horizon and builds a trajectory before scoring it. Stopping at the
last required observation, streaming scoring and reusing scratch space deserve
separate work, with particular care for interval-flow accumulation and resets.

For scale, the earlier reported 4.4 s baseline integration time would imply
about 5.1 days for 100,000 serial solves. Saving startup seconds barely affects
that total. Effective posterior samples per second should accompany likelihood
milliseconds when assessing fitting performance.

### Tensor IR and derivatives remain the structural work

Keeping indexed families and contact operations in the IR would enable batched
matrix/vector evaluation and avoid repeating expression structure per cell.
Sharing forcing payloads would also reduce serialization and loading costs. A
binary encoding alone cannot remove duplicated data. These are larger,
cross-language changes requiring schema/version and golden coordination.

The Jacobian problem is separate again. Method-based state-gradient gating has
landed (#439); #921 remains open for factored aggregate derivatives. Improving
dependency lookup can reduce compiler work but cannot make an explicitly dense
Jacobian small.

Stratified contact models need not resemble the scalar mean-field example from
#921. With stratum-specific denominators, aggregate coupling rank can grow with
the coupled dimension and arms. Sparse local dynamics plus factored aggregate
coupling, or Jacobian-vector products, is the relevant direction—not an
assumption that the whole contact Jacobian is rank one.

An implicit solver is only justified if measured stiffness makes its linear
solves cheaper than the explicit stages it replaces. Cell count alone does not
show stiffness, and dense factorization is not a viable solution at 100,000
states. No implicit-solver speedup or 100,000-cell feasibility was established
by these experiments.

## Recommended sequence

Start with indexed event-conflict validation and measure it. Follow with event
context allocation reuse as a separate change. These address demonstrated or
source-visible superlinear work without changing the IR or numerical method.

Next, measure shared parameter-and-state expression evaluation and the
observation-limited likelihood path. Improve parse-once loading for startup,
without counting its benefit toward every likelihood iteration. Benchmark the
existing bytecode evaluator through the actual ODE call site.

Use those profiles to guide the larger family/tensor IR and factored-derivative
design. Before that redesign, collect whole-run profiles and
RHS/accepted/rejected step counts, then vary ages, additional strata, arms and
event counts independently. The immediate event fix has a clear acceptance
experiment; the longer-term architecture still needs those measurements.

## Implementation follow-up

The indexed conflict checker is now implemented in `effects.rs`. This follow-up
changes only validation: event evaluation and buffer allocation remain
unchanged. The initial investigation above used existing binaries; this
follow-up builds release binaries before and after the change with the same
Cargo configuration.

The checker builds borrowed-string target summaries and explicitly preserves
first-set order, consecutive duplicate-name handling, first-transfer selection,
and multiple-set error precedence. No IR/schema or golden changes are involved.

A regression oracle retains the previous scan algorithm in test-only code. It
compares exact results and error strings across 19,440 ordered cases: all
four-action combinations from six action choices, three event-name patterns, and
five due-event lists, including reordered and repeated event indices. The
existing effect tests additionally cover state updates and phase semantics.

The end-to-end benchmark uses three repetitions per configuration, alternating
before/after order. Both binaries read the same compiled JSON and parameter
point, and write complete trajectory TSV output for SHA-256 comparison. Loading,
integration and file output are included; compilation is excluded. The timings
are separate from the earlier single-run measurements to `/dev/null`.

### Controlled before/after results

| Configuration | Before seconds (three runs) | After seconds (three runs) | Median speedup |
| ------------- | --------------------------- | -------------------------- | -------------: |
| Baseline      | 8.169, 8.015, 7.790         | 8.043, 7.930, 7.754        |          1.01× |
| One arm       | 14.749, 14.901, 14.509      | 9.006, 9.115, 9.100        |          1.62× |
| Three arms    | 75.152, 74.266, 75.252      | 25.710, 25.706, 25.548     |          2.92× |

All six complete trajectory TSVs per configuration have the same SHA-256 hash.
The three-arm median falls from 75.152 s to 25.706 s, removing about 66% of
whole-run time. This independently supports the earlier short profile: here the
predicted opportunity survives an actual implementation and controlled A/B. The
baseline changes by only about 1%; that is not a persuasive speedup on its own.
The benefit grows sharply with simultaneous event-batch size.

The residual 25.7 s still includes loading, integration, output and event amount
evaluation. This experiment does not attribute that remainder or establish the
benefit of buffer reuse. It also does not measure fitting throughput directly,
although fits using the same event path can benefit from the same optimization.

Both binaries were built with `cargo build --release -p cli --offline` from the
Rust workspace directory, preserving its warning configuration. The before
binary was saved before changing production code. The only production source
change in the after binary is this checker; the additional test code is excluded
from release builds. No compiler, schema, rate, event-update or solver changes
were made.

### Validation

The following checks passed from the Rust workspace directory:

```bash
CAMDL_SKIP_VERSION_CHECK=1 cargo test -p sim --offline --lib \
  --test cross_backend_lifecycle_agreement \
  --test gate_corner_case_baseline --test event_transition_codrain
cargo clippy -p sim --all-targets --offline -- -D warnings
```

That is 529 simulation unit tests, 27 cross-backend lifecycle tests, three
event/transition co-drain tests and the trajectory-baseline test: 560 tests
total. The new scan-comparison oracle is included in the unit tests. Clippy
passed with warnings denied. At this stage the full `make test` gate had not yet
been run; the later attempt and final review are recorded below.

A subsequent `make test CARGO_WRAP=` attempt passed workspace Clippy, compiler
checks and golden synchronization, then stopped in the non-simulation Rust
suite: `cli --test store_leaf_found_by_hash`,
`renaming_the_model_file_keeps_every_leaf_a_hit`, created distinct store leaves
after a model-file rename. The full gate is therefore not green. The failure is
outside the modified event-validation path; its baseline status was not checked
at that point (see final review below). The optional sccache wrapper was
disabled because it failed with an environment permission error before invoking
the compiler.

## Second profiling pass: schedule construction

A full-process Samply recording of the conflict-check-optimized three-arm run
captured 25,368 main-thread samples over approximately 26.1 seconds. Symbols
were resolved using Samply's presymbolication sidecar. Inclusive attribution
(recursive frames counted once per sample) was:

| Path                       | Main-thread samples |
| -------------------------- | ------------------: |
| Rate expression evaluation |               53.9% |
| Effect-plan construction   |               13.5% |
| JSON deserialization       |               12.1% |
| Forcing evaluation         |                8.6% |

These are inclusive categories and can overlap; forcing evaluation is part of
expression evaluation. This supersedes an earlier five-second sample beginning
six seconds after launch, in which plan construction occupied about 53% of the
window. That window identified the hotspot but substantially overstated its
whole-run share. The full profile bounds the benefit of eliminating plan
construction at roughly 1.16x on this run.

The exact-time effect planner runs once at the start of every ODE solve,
including every ODE likelihood evaluation. At integration boundaries consumers
read the prepared batches. Construction resolves firing times, sorts time/event
pairs, groups instants using EFFECT_EPS, and produces event indices in
declaration order. The old collection loop tested membership and sorted the
growing batch after each append. Large simultaneous event families repeatedly
scanned and sorted the same growing vector.

The second optimization appends during collection, then sorts and deduplicates
each completed batch once. A vector remains appropriate for ordered consumption;
a new hash-set representation is unnecessary. Time grouping, origin handling and
per-effect time deduplication are unchanged. The old algorithm is retained as a
test-only oracle: 2,000 generated schedules compare exact origin indices, firing
times and batches, including unsorted inputs, duplicates, outside-window times,
and values around the grouping tolerance.

Three alternating before/after pairs, comparing against the first optimization:

| Variant                      | Wall seconds           | Median |
| ---------------------------- | ---------------------- | -----: |
| Indexed conflict checker     | 24.783, 25.474, 25.164 | 25.164 |
| Plus completed-batch sorting | 22.217, 22.520, 22.410 | 22.410 |

That is a further 1.12x speedup, or about 11% less total time. All trajectory
headers and values match byte-for-byte. Whole-file hashes differ because the new
binary embeds a newer commit hash in the first comment line; direct file
comparison finds that version banner is the only difference. Do not describe
these particular files as having identical whole-file hashes.

The recording used `samply record --save-only --unstable-presymbolicate` around
the complete simulation command, with output directed to the null device. Timing
pairs were run without the profiler and wrote the complete TSV files. Raw
profiles, symbol sidecars and timing logs remain local artifacts; private inputs
and identifying paths are omitted here.

The second change passes the same focused validation command as the first: 530
unit tests plus 31 lifecycle/co-drain/trajectory tests (561 total), and sim
all-target Clippy with warnings denied. No schema or golden changes are needed.
The previously recorded full-gate CLI store-identity failure remains unresolved;
the second change was left pending review at that stage.

The next profile-driven priority is repeated expression evaluation, which now
has the largest measured share. Event buffer reuse remains a bounded candidate,
but its whole-run contribution has not been isolated. Shared forcing values per
stage, shared parameter-and-state expressions and evaluation dispatch should be
measured independently rather than assumed to provide multiplicative gains.

## Evaluator contender measurements

The next investigation compares evaluation strategies before choosing a
production implementation. The measurements use a scratch Rust executable linked
directly to the existing release `sim` and `ir` libraries, with the same Rust
toolchain. No production evaluator source was changed for these probes.

### Method

The timed unit is one complete transition-rate vector, with the binding cache
invalidated for each evaluation as it is at an ODE stage. It excludes parsing,
model construction, expression rewriting, bytecode construction, rate guards,
stoichiometric accumulation and integration. It is therefore a rate-vector
benchmark, not a complete RHS or likelihood benchmark.

Two input sets were used: twelve synthetic perturbations of the initial state,
and twelve states sampled evenly from saved trajectories. The latter use the
recorded population values and evaluate at a small offset from the snapshot
time. These are rounded output snapshots, not exact internal Runge–Kutta stage
states. The fixed parameter point is identical across variants. A broader
parameter cloud and post-intervention trajectories remain untested.

Each process checks every transition rate bit-for-bit against the original tree
interpreter at all twelve input states before timing. Each timed batch performs
1,200 rate-vector evaluations; seven batches run per process. Contenders run in
forward then reverse order, producing fourteen batch observations per contender
and model. These are correlated observations from two processes, not fourteen
independent machine trials. The shared machine produces some timing variation;
small differences and overlapping ranges should not be treated as established
wins. Other benchmark/build work was kept sequential; a scratch source copy
coincided with part of the later specialization experiment, further weakening
small specialization claims.

### Contenders

- **Tree:** the current evaluator with its existing per-stage binding cache.
- **Bytecode:** the existing flat VM, with its own cache and reusable scratch.
- **Forcing memo:** replace each referenced forcing with a shared, on-demand
  binding. This measures one concrete memoization mechanism; it does not prove
  that every possible forcing-cache design has the same cost.
- **CSE:** share repeated structural subexpressions of at least four nodes via
  on-demand binding slots. Structural matching and rewriting happen outside the
  timer. No algebraic reassociation is performed.
- **CSE + bytecode:** execute that shared representation with the flat VM.
- **Parameter specialization:** evaluate parameter/constant-only arithmetic once
  outside the timer, substituting constants for the fixed parameter point. Its
  bytecode combination is also measured separately.

The CSE and specialization probes establish finite-value equality on these input
sets, not production readiness. They do not validate gradient handling,
error-attribution side effects, cache lifetimes across changed parameters, or
construction-cost amortization. Runtime lowering could keep the wire IR intact;
compiler-emitted shared derivative representations would need a separate design.

### Rate-vector results

Medians in microseconds on trajectory-derived inputs:

| Model    |   Tree | Bytecode | Forcing memo |    CSE | CSE + bytecode |
| -------- | -----: | -------: | -----------: | -----: | -------------: |
| baseline |  299.6 |    177.0 |        324.7 |  285.5 |          175.1 |
| one      |  456.6 |    285.8 |        461.6 |  374.7 |          248.0 |
| three    | 1373.9 |    848.2 |       1235.6 | 1090.9 |          722.6 |

The synthetic-perturbation control shows the same broad ordering. Bytecode is
consistently faster on all sizes (about 1.6–1.7x here). CSE adds a useful
benefit on the larger configurations, and its combination with bytecode reaches
about 1.9x on the largest rate vector. These gains overlap and must not be
multiplied. Forcing memoization alone regresses on the baseline, is
approximately flat on one arm, and helps somewhat on three arms. It is not the
best first candidate. The small baseline CSE difference is within observed
timing variation.

The structural probe adds 180, 344 and 1,015 sharing slots respectively; the
forcing memo adds 6,728 slots in each model. A logical traversal of one
large-model input, honoring existing binding memoization and conditional
branches, counts 429,798 expression-node visits, 1,230 binding misses, 91,266
binding hits and 49,200 forcing references to 6,728 distinct forcings. These
counts exclude work inside forcing implementations. Many repeated forcing
references do not imply that the proposed extra cache is cheap enough to win.

Parameter specialization is a separate experiment with its own tree controls.
The existing compiler LICM pass deliberately leaves cheap arithmetic inline
(`licm.ml`, `contains_expensive`), so this probes a broader policy rather than
claiming LICM does not exist. Its results and full-solve bytecode check follow.

### Parameter specialization probe

Medians in microseconds, with separate contemporaneous tree controls:

| Model    |   Tree | Specialization | Specialization + bytecode |
| -------- | -----: | -------------: | ------------------------: |
| baseline |  287.0 |          281.7 |                     172.2 |
| one      |  595.8 |          442.1 |                     280.8 |
| three    | 1350.6 |         1184.8 |                     800.3 |

The one-arm tree control is noisy (471–727 microseconds across batches), so its
apparent specialization gain is not reliable enough to rank. The large-model
specialization result is modest compared with bytecode; baseline specialization
is effectively flat within timing spread. Specialization plus bytecode does not
beat the earlier sharing-plus-bytecode result on the large model. These separate
runs are not a precise head-to-head ranking of the combinations. A production
specialization strategy would also pay rebuilding costs at every changed
parameter point, which this timed region excludes.

### Full-solve bytecode validation

A scratch runtime copy routes the ODE rate loop through the existing flat VM
when enabled. The same executable runs both paths, toggled by `CAMDL_EVAL_FLAT`.
Both retain per-rate table-error clearing and rate guards; this prototype
creates a flat cache/scratch buffer per stage rather than implementing
production scratch reuse. It includes the completed-batch schedule optimization.
Real-valued explicit ODE equations still use the tree path.

Three alternating timing pairs per model include JSON loading, VM construction,
integration, guards and complete TSV output, with compilation excluded:

| Model    | Tree seconds (three runs) | Bytecode seconds (three runs) | Median speedup |
| -------- | ------------------------- | ----------------------------- | -------------: |
| baseline | 7.223, 6.835, 6.783       | 5.360, 5.671, 5.470           |          1.25x |
| one      | 8.871, 8.762, 8.842       | 6.786, 6.744, 6.772           |          1.31x |
| three    | 22.123, 21.809, 21.853    | 16.450, 16.558, 16.487        |          1.33x |

All six complete trajectory files per model have identical SHA-256 hashes,
including the version banner because each pair uses the same executable. The
flat path actually executes in the ODE loop; merely setting the existing toggle
on an unmodified runtime would not establish this result.

The initial scratch build omitted files embedded by the CLI. A prematurely
launched set of comparisons consequently used the old copied binary; that entire
set was discarded. The table above was rerun only after the scratch build
completed successfully and its executable timestamp was checked.

### Decision supported by these measurements

The strongest next implementation candidate is wiring the existing bytecode VM
into ODE rate evaluation with appropriate scratch ownership. It wins across all
three model sizes in both isolated and full-solve tests, with exact outputs at
the measured parameter point. General-purpose CSE is the next candidate, with
additional gradient and error-semantics design work. Forcing memoization and
broader parameter specialization should not precede those based on this data.

These experiments do not certify a production bytecode change: error cases,
rejected stages, explicit real-valued equations, sensitivity paths, stochastic
backends and a parameter cloud still need their appropriate validation. No JIT,
matrix-kernel or tensor-IR implementation was benchmarked; no measured gain is
claimed for those larger designs. MCMC throughput and effective samples per
second were not measured. The production evaluator source remains unchanged.

## Final review of completed-batch sorting (2026-11-03)

The production change remains confined to `EffectPlan::build_exact`: collect
indices using the existing time-group anchors, then sort and deduplicate once
per batch. The old scan/sort implementation and generated comparison test live
in the test module. No event values, grouping tolerance, origin behavior,
declaration-order semantics, grid planner, schema or goldens change.

The final focused run passes 562 tests: 530 simulation unit tests, 27 lifecycle
tests, three co-drain tests, one binding-cache A/B test and one
trajectory-baseline test. Sim all-target Clippy also passes with warnings
denied. Tests were rerun after moving the oracle into the test module; the
production algorithm is the same one used for the 25.164-to-22.410-second
measurement above.

The full-gate store-identity failure is now reproduced independently in a
scratch source copy with the committed `effect_plan.rs` and the unmodified
production ODE evaluator, excluding this schedule optimization and all bytecode
probes. `cargo test -p cli --test store_leaf_found_by_hash --offline` gives the
same model-file-rename failure (two tests pass, one fails). The local compiler
was pinned for that baseline comparison and version checking bypassed as
documented. This establishes that the schedule fix does not introduce that
failure. The store-identity defect remains unresolved and the full gate is still
not green; no assertion or gate was weakened to conceal it.

The completed-batch optimization and this investigation update are suitable for
check-in as a bounded performance change with that baseline failure documented.
Experimental ODE bytecode, workspace reuse, CSE and specialization remain
scratch prototypes and are not part of the production change.
