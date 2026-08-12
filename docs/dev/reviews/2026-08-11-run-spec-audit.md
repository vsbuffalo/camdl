# Run-spec audit: the doc rebuilt against the code, and what the code showed

Date: 2026-08-11\
Scope: `docs/camdl-run-spec.md` §1–§14 + appendices, audited against the
implementation at `3e2b2888`\
Spec rewrite: `daddc6f2` (branch `docs/run-spec-rewrite`)

Seven agents each took a contiguous slice of the spec and checked every claim
against the source, running the binary rather than trusting the doc's
transcripts. The doc is now rebuilt from what they found. This file records what
the audit turned up about the _code_, which is the more valuable half.

Two findings are silent-wrong-answer bugs and are reproduced below in full. Both
were re-verified independently of the agent that found them.

## Verification status

Findings marked **[verified]** were reproduced by the author of this document
from a clean state, with the commands shown. Findings marked **[agent]** were
reproduced by the auditing subagent and are recorded with their citation but
were not independently re-run; they should be re-checked before a fix lands.

Three findings were reported independently by more than one agent working from
different slices and different evidence; those are marked **[corroborated]** and
carry unusually high confidence.

## Where each finding was filed

| Finding                        | Filed as                                                   |
| ------------------------------ | ---------------------------------------------------------- |
| 1, 3, 4, 5, 6, 7, 8, 9, 10, 19 | gh#583 (cluster; 9 folded into 1). gh#573 is a sibling.    |
| 2                              | gh#584                                                     |
| 11                             | gh#572 — pre-existing, broadened to cover generated draws  |
| 12                             | gh#587                                                     |
| 13                             | gh#588                                                     |
| 14                             | gh#585                                                     |
| 15, 16, 17                     | gh#586                                                     |
| 18                             | gh#591                                                     |
| 20                             | gh#590                                                     |
| 21                             | split: gh#592, gh#593, gh#594, gh#595; label case → gh#577 |
| 22                             | gh#589                                                     |
| 23                             | no issue — subsumed by the spec rewrite                    |

One defect found while verifying rather than auditing — the CLI-docs gate cannot
express an intentional negative example — is gh#596.

---

## Correctness — silent wrong answers

### 1. `--integrator` changes the trajectory but not the `run_id`

**Severity:** blocker. **[verified]**\
`main.rs:1717`, `util.rs:2677`, `runid/src/inputs.rs:148`

`camdl simulate --integrator rk4` and `--integrator rk45` produce different
trajectories, receive the same `run_id`, and land in the same store leaf. The
store keeps whichever ran first and serves it for the other.

```
$ camdl simulate sir_basic.camdl --backend ode --integrator rk4 \
    --param beta=0.3 --param gamma=0.1 --param N0=1000 --param I0=5 -o rk4.tsv
   stored ./results/sims/sir_basic-cd37d79d/ode-dt1-c3f7aa2c/base-7cd0eaec/baseline-33233b72/seed_1-06cbd6b3
          camdl cat e2dee5c9…

$ camdl simulate … --integrator rk45 -o rk45.tsv
   stored ./results/sims/sir_basic-cd37d79d/ode-dt1-c3f7aa2c/base-7cd0eaec/baseline-33233b72/seed_1-06cbd6b3
          camdl cat e2dee5c9…          # same id, same leaf

$ grep '^17	' rk4.tsv   →  17  824  113  63  26.4417  10.5345
$ grep '^17	' rk45.tsv  →  17  824  113  63  26.4422  10.5348
$ camdl cat e2dee5c9… | grep '^17	'
                            17  824  113  63  26.4417  10.5345   # rk4's answer
```

Cause: `--integrator` is applied to the model only on the execution path
(`util.rs:2677`, `apply_integrator_override`). `build_simulate_cas_sink`
(`main.rs:1717`) performs a second, independent IR load for the identity path
and never applies it; `SimConfig` has no `integrator` field.

The hasher is not at fault — `ir_hash.rs:1054` folds in the integrator, and
`integrator_choice_changes_run_id` (`ir_hash/tests.rs:714`) passes. That test
mutates `m.simulation.integrator` directly and never crosses the CLI, so it
exercises the hasher while the wiring is broken. Test on an extracted pure
function, bug at the callsite.

`--dates` is a second instance of the same class **[agent]**: it adds a `date`
column to the `SimEnsemble` artifact but appears in none of the four identity
levels.

`--force` does not repair either. `commit_atomic` returns `AlreadyCompleted` and
`remove_dir_all`s the freshly computed correct bytes (`store.rs:305`)
**[agent]**; only removing the leaf by hand recovers.

**Independent.** Fixable alone. The structural fix (item 3) is separable.

### 2. A `derived_expr` projection over a real-valued compartment reads zero

**Severity:** blocker. **[verified]**\
`main.rs:2356`, `sim/src/inference/multi_stream_obs.rs:864`

Two observation streams over the same ODE compartment `W`, same run. The
restricted spelling reads the compartment; the general spelling reads zero, with
no warning and no error.

```
observations {
  safe_w    { projected = prevalence(W)  … }
  derived_w { projected = W + 0.0        … }
}

$ camdl simulate res_obs.camdl --scenario baseline --backend ode --seed 1 --obs-dir obs/
$ cat obs/safe_w.tsv     →  0  20  126  233  298  384  782
$ cat obs/derived_w.tsv  →  0   0    0    0    0    0    0
$ grep '^30	' res_traj.tsv  →  30  246  290  464  592.4623  …   # W is real and nonzero
```

Cause: `main.rs:2356` allocates `RealState::new(...)` — permanently zero, never
populated from the trajectory snapshot — and passes it to
`eval_stream_projection`. The inference-side twin at `multi_stream_obs.rs:864`
holds the same zero state behind the comment _"likelihood eval never reads real
compartments"_, which is an assumption no invariant enforces. Per the auditing
agent it holds today only because an unrelated capability gate (gh#191) blocks
the chain-binomial path, leaving the ODE backend exposed: an `nl-sbplx` fit
against such a stream runs to completion with `ll = -inf` at every θ and
surfaces only as a dt-convergence FAIL **[agent]**.

The asymmetry is the dangerous part. The _safe_ spelling hard-errors cleanly;
the _general_ one zeroes silently.

The `compiled.default_params` at `main.rs:2357` is **not** a second bug, though
it reads like one. The model handed to `CompiledModel::new` has already had
`--params` and `--param` applied, so `default_params` carries the resolved
values. Checked: **[verified]**

```
$ camdl simulate rho_test.camdl --scenario baseline --seed 1 --obs a.tsv
                                              # projected = rho * I, rho = 1.0
  0  14   5  19   10  37   15  88   20 155   25 214   30 271
$ camdl simulate rho_test.camdl --scenario baseline --param rho=0.1 --seed 1 --obs b.tsv
  0   0   5   1   10  10   15  11   20  17   25  20   30  27
```

The override reaches the projection. Only the _real compartment_ state is
unpopulated.

**Independent.** The interim refusal (reject a real-compartment reference at
`StreamProjection::from_ir`) is a small change and can land immediately, ahead
of threading the real state through.

---

## Identity and caching

### 3. The model is loaded twice — for identity and for execution — with nothing linking them

**Severity:** high. **[agent]** `main.rs:1717`, `util.rs:2669`

The direct cause of item 1, and it will cause the next one. One flag already has
the right pattern: `rematerialize_with_output_every` (`util.rs:3161`) lowers
`--output-every` into a rematerialized IR file precisely so that _"both the
engine and the CAS identity load `base_model` from this path."_ The sibling flag
did not get it, and nothing in the types prevents that.

Proposed guard: a test enumerating `SimulateJob` / `OutputView` fields that
asserts each is either present in `SimConfig`, or in an explicit `NOT_IDENTITY`
allow-list with a one-line reason. That test catches items 1 and the `--dates`
variant at authoring time.

**Entangled** with items 1 and 4 — this is their common fix.

### 4. `commit_atomic` discards freshly computed bytes on a same-identity hit without comparing them

**Severity:** high. **[agent]** `store.rs:305`

Given correct identity this is right. Given an identity gap it converts a loud
disagreement into a silent wrong answer, and it is why `--force` cannot repair
item 1. The staged digests are already computed at `store.rs:271` and the
incumbent's are already in its `run.json`; comparing them and warning — or
erroring under `--force` — costs nothing and turns every gap of this class into
an immediately visible defect.

The single highest-leverage change in the subsystem.

**Independent.** Lands alone and makes the whole class visible.

### 5. `batch run` labels the store's model level from the compiled-IR path

**Severity:** high. **[agent] [corroborated — three agents, three methods]**
`batch.rs:529`

The same model, params, scenario and seed lands in a different directory
depending on how it was launched:

```
sims/sir_basic-cd37d79d/…              # simulate
sims/6a541b38…fda-cd37d79d/…           # batch run, IR cache on
sims/camdl_64692-cd37d79d/…            # batch run --no-ir-cache  (PID in the name)
sims/camdl_every_0km80w-9be7ab1b/…     # [output] every           (random per invocation)
```

Three directories, one `run_id`. `camdl show <id>` then reports an ambiguous
match and advises passing a longer hash prefix, which cannot help — the
ambiguity is not in the hash. Because `store.lookup` is directory-scoped, the
last two forms are a cache miss on every invocation: one agent ran the same
18-cell manifest twice and got two parallel trees with byte-identical content
and a full re-simulation. `simulate` gets this right by threading the original
path (`main.rs:1774`), with a comment at `main.rs:1698` warning about exactly
this failure mode. `batch status` computes the label a third way
(`batch.rs:1819`).

Fix: thread `exp.config.model` through as the display path, mirroring
`build_simulate_cas_sink`'s `display_path`. One line plus a test asserting
`simulate` and `batch run` of the same cell land on the same path.

**Independent.**

### 6. Two parallel, hand-maintained fit-identity hash functions, whose comments claim they are one

**Severity:** high. **[agent] [corroborated — two agents]** `fit/cas.rs:339`,
`fit/provenance.rs:302`

`fit_level_digest` + `stage_config_hash` is the CAS identity; `fit_stage_hash`
is the `--resume` guard. They enumerate different input sets in different
encodings (typed `ContentAddressed` vs manual `Sha256::update` with `\x00`
separators) and neither references the other. Only the CAS one covers
`[data.holdout]` bytes, `n_trajectories`, and the resolved `obs_alignment`.
Meanwhile `pgas.rs:291` and `pmmh.rs:501` both claim in comments that
`fit_stage_hash` is _"the same hash the v2 dispatch site uses for cache-hit
checks"_ — dispatch actually uses `cas::resolve_fit_stage` + `store.lookup`.

Two independent notions of "same statistical problem," one of them outside the
run-identity rules. The resume guard should be derived from the CAS stage
identity minus the extension dimension, not re-listed.

**Independent.**

### 7. Flags applied after the CAS claim are silently ignored on a cache hit

**Severity:** high. **[agent] [corroborated — two agents]** `fit/`
`CliStageOverrides`

`--record-ancestry`, `--record-prequential`, `--decibans-thresh` and the three
dt-check flags are not in `CliStageOverrides` and are applied after the claim.
Run once without the flag, once with: the second gets a cache hit, never writes
the requested artifact, and exits 0 with no warning. The gh#514 / gh#540 shape.
`--n-trajectories` re-keys correctly and is the working contrast.

**Independent** per flag, but all six want the same fix.

### 8. The integrity gate checks size and mtime; the recorded SHA-256 is never verified

**Severity:** medium. **[agent]** `store.rs:624`, `record.rs:47`

`check_exact_set` compares `bytes + mtime` only. `FileChecksum.digest` has two
consumers — an ensemble `deps` edge and an 8-char display — and `record.rs:47`
names `camdl verify` as the verifier, which does not exist. A store copied
without `-p` reports every leaf stale; a file edited in place to the same length
with mtime restored reads as a hit. For a store whose outputs inform
public-health decisions, either wire an opt-in digest check on read, add
`camdl verify`, or drop the field.

**Independent.**

### 9. `SimulationConfig::hash_into` folds in the integrator only when non-default

**Severity:** medium. **[agent]** `ir_hash.rs:1059`

Folds `"rk45"` + tolerances only for the `Rk45` variant, explicitly _"so a
default-Rk4 model keeps its pre-gh#166 run-id (no cache churn)."_ Making the
hash a function of _whether a value is the default_ rather than of the value
breaks the encoding's own framing rule — every other enum writes its variant
index unconditionally — and is the kind of exception that made item 1 easy to
miss on a code read. The re-key it avoided is the cheap thing; the invariant it
broke is the expensive one.

**Entangled** with item 1: fix together, accept the re-key.

### 10. `[provenance] reason` — free-text human annotation — is part of the fit identity

**Severity:** medium. **[agent] [corroborated — two agents]**

Three different `reason` strings produce three fit hashes. Editing a comment
discards the cache and re-runs the fit.

**Independent.**

---

## Silent degeneracy: variation that is requested and then discarded

### 11. A scenario preset silently voids a sweep or a draw, and the path labels advertise the values that were not used

**Severity:** high. **[agent] [corroborated — two agents]** `engine.rs:357`,
`sim_job.rs:110`

```
$ camdl simulate sir_basic.camdl --scenario baseline --draws uniform -n 3 --backend ode
```

Three leaves, three distinct `run_id`s, three byte-identical `traj.tsv`. The
sweep case is the same: six leaves labelled `beta_0.2_gamma_0.1` …
`beta_0.4_gamma_0.2`, all byte-identical.

The tier order (scenario `set` is tier 4, draw/sweep tier 3) is deliberate and
documented. The defect is the silence. The collision guard already exists —
`check_explicit_draws_scenario_collision` (`engine.rs:357`) — but is gated on
`ParamSource::Draws { explicit_file: Some(..) }`, i.e. only
`--draws <file.tsv>`, precisely the case where the user was _most_ explicit. It
never fires for `prior` / `uniform` / `posterior`, which is where
prior-predictive and space-filling workflows live, and never for
`ParamSource::Sweep`.

Better shape: move the check out of the `explicit_file` guard and grade it by
how much of the draw is shadowed — hard error when a scenario pins _every_ drawn
column (the run is provably degenerate), warning naming the shadowed columns
otherwise. One `HashSet` intersection per job.

Second-order: run identity keys on the discarded draw values, so the store holds
N distinct ids for one physical computation.

**Independent**, and it should land before any `simulate --sweep` work (item
22), or that ships the same trap.

### 12. A swept fit's `fit.meta.json` records the wrong `fixed` values

**Severity:** high. **[agent]** `fit/mod.rs:557`

The sidecar is built once from the base config and written into every swept
segment. Both segments of `--sweep rho=0.5,0.7` record `rho: 0.6`. The sidecar
is the artifact a reader consults to learn what a fit held fixed.

**Independent.**

### 13. `camdl simulate` never reports a cache hit and `--force` is inert

**Severity:** medium. **[agent]** `main.rs:1661`

`SimSink` never overrides `should_run`, so every cell re-simulates on every
invocation and `--force` changes nothing. This is why both runs in item 1 print
`stored` while the store keeps only the first bytes — the divergence is between
the `-o` mirror (freshly computed, correct) and the CAS artifact (written once,
never updated). `batch run` already has the behavior the doc and `--force`'s own
help text promise.

**Entangled** with item 4 — decide the compare-on-hit policy first.

---

## Inert configuration presented as working

### 14. `[data] holdout` / `holdout_after` do nothing

**Severity:** high. **[agent] [corroborated — two agents]**

A/B on identical configs differing only in `holdout_after`: same 12 observations
scored, same θ̂, same per-chain logliks. The keys parse, validate against
`[data.holdout]`, and feed the fit hash (gh#190), so editing a holdout file
forces a re-fit for a bit-identical artifact — but nothing withholds or scores
them. `git log -S holdout_after` shows the field landed with the original
`FitConfigV2` and was only ever touched for hashing.

Three documents describe the feature as working, one with sample output. A
modeler currently believes their holdout data was withheld when it was never
loaded, and reports an in-sample score as held-out.

**Independent.** The doc half is a same-day change and is the higher-stakes
half.

### 15. `fit_starts` is dead, and setting it silences a correctness warning

**Severity:** high. **[agent]**

No runner reads `fit_starts`. Setting `fit_starts = "prior"` silences the
dangling-priors warning — and camdl's own warning text tells the user to set it.
The advice actively makes the situation worse.

**Independent.**

### 16. `prior = { fixed = 0.3 }` parses and silently resolves to a flat prior

**Severity:** high. **[agent]** `prior.rs:378`

Passes `validate_priors_present`, fires no warning. Precisely what gh#75 was
built to prevent.

**Independent.**

### 17. `simulate --parallel` is declared with an env binding and never read

**Severity:** medium. **[agent] [corroborated — two agents]** `args/mod.rs:629`,
`main.rs:1183`

`main.rs:1183` hardcodes `parallel: 1`. A user who exports `CAMDL_PARALLEL` for
their fits gets a silent no-op here. Either honour it (the engine already
branches on `grid.parallel > 1`) or reject it with "use `camdl batch run`".
Accepting and discarding violates the no-silent-defaults posture.

**Independent.**

### 18. Dead `RunInput` leaf-shape structs, two of which contradict the code that replaced them

**Severity:** medium. **[agent]** `runid/src/inputs.rs:289`

`TrajectoryInput`, `PfilterEvalInput`, `SurveyInput` and siblings are documented
as _"the identity contract, expressed as types"_ and are constructed by nothing.
`PfilterEvalInput` and `SurveyInput` do not describe what pfilter and survey
actually hash: `pfilter_cas.rs:82` digests
`{particles, replicates, dt, obs_block, flow_indices, data}` as ad-hoc JSON,
while the typed struct carries a full `SimConfig` and no `replicates`. A reader
auditing "what is hashed" from `inputs.rs` gets a wrong answer for two artifact
kinds.

Per the repo's own rust-conventions rule, delete or wire them.

**Independent.**

### 19. `run.json.inputs` is null for every `Sim` leaf, and a comment claims otherwise

**Severity:** medium. **[agent]** `batch.rs:1225`, `batch.rs:1077`

`camdl show` prints no parameter values for a sim leaf, while `batch.rs:1077`
justifies collapsing a long params label to the tag `draws` on the grounds that
_"the full drawn values live in `run.json`."_ They do not.

**Independent.**

---

## Diagnostics and CLI surface

### 20. A `pfilter` stage removes the whole fit from `camdl fit table`

**Severity:** medium. **[agent]**

Proved by deleting the stage directory and watching the fit reappear. Also
blanks `fit summary --format json`'s `table_row`.

**Independent.**

### 21. Assorted user-facing surface defects

**Severity:** low–medium. **[agent]** Each independent; each small.

- `camdl batch status` cannot parse a `.camdl` manifest — the shape of every
  example in its own `--help` — and prints "Run … to start" for a completed
  sweep (`batch.rs:1762`). Also emits `results/sims/sims/…` (`batch.rs:1840`).
- `camdl fit diff` re-implements config diffing inline and collapses everything
  but `algorithm`/`chains` to the string `settings changed`, while
  `config_diff.rs` already has the typed per-key engine that
  `fit table --format json` uses.
- `camdl fit new` prints a hint telling users to write `starts_from`, a key the
  loader hard-rejects with a migration error.
- The stage-DAG validator's errors name `starts_from` — the internal Rust field
  (`config_v2.rs:2956`) — not the TOML key `init_mle`. **[verified]** A user who
  follows the error verbatim hits the legacy-key rejection.
- The binary's help advertises three subcommands that do not exist (`fit list`,
  `fit where`, `fit label`).
- The gate's error hardcodes both "refine stage" and "(> 1.10)"; the real
  threshold is 1.01.
- `camdl list --format json` is not one parseable document — five printers emit
  NDJSON, `print_survey_json` emits a pretty array, so `jq -s .` fails.
- The multi-cadence `--obs` error dumps Rust `Debug` at the user
  (`RegularSchedule { start: 0.0, step: 7.0, … }`).
- `--dates` validation fires _after_ the simulation completes (`main.rs:2027`),
  throwing the work away; the fact is knowable at model load.
- The store's `model.ir.json` / `model.render.json` / `model.graph.json` are
  written at the root of a multi-model store, last-writer-wins (`batch.rs:596`).
  They belong at the model level.
- An ad-hoc `--enable` run is labelled `baseline` in `camdl list`, identical to
  a run with nothing enabled (`main.rs:1743`).
- `[fixed] from_file` and `[synthetic] true_params` anchor at the CWD, not the
  fit.toml, contradicting `fit-toml.md:92`.
- `nuts` numbers chain directories from 0; everything else from 1.
- Batch `[sweep] range` has an unbounded push loop for `step <= 0` — the gh#257
  class. Flagged from code, deliberately not executed.

---

## Observation semantics

### 22. `simulate --obs` quantizes synthetic data to the trajectory output grid

**Severity:** high. **[agent]**

Weekly `output { trajectories { every = 7 } }` with a daily `emit_schedule`
yields incidence `0,0,0,0,0,0,0,77` and step-function prevalence, with no
warning. `--output-every 7` reproduces it — a presentational-looking flag
changes the data you then fit.

**Independent.**

### 23. The spec's statement of observation semantics was wrong in ways that change interpretation

**Severity:** high as a doc defect; fixed in `daddc6f2`. **[agent]**

Recorded because each was a live misreading risk, not merely stale:

- §12's precedence chain ended in `Prior::Flat`. `fit run` hard-errors; only
  `profile` falls through with a warning. The doc documented neither half.
- §14.2's three per-backend snapshot-timing rules were each wrong on at least
  one path. Inference lands _exactly_ on `t`; ODE matches a recorded snapshot
  within 1e-9 and errors otherwise, with the value path rounded and the gradient
  path not.
- §14.4 omitted that `normal(...)` is a discretized _count_ likelihood, that all
  count families round the observation, that `bernoulli` thresholds at 0.5, and
  that `beta` returns −∞ at an observed 0 or 1.
- §14.1's "two projection modes" is five IR variants. Bare `incidence` over
  strata is error E280; bare `prevalence` pools silently.

One rounding decision has two answers inside one backend (ODE value path rounds,
gradient path does not) — that is a code finding, own entry, needs its own
issue.

---

## Worth building

Each checked against `git log` and `docs/dev/` to distinguish a gap from an
abandoned path.

1. **Make `holdout_after` work, or delete it — M.** The pieces exist: a
   `pfilter` stage already evaluates a likelihood at fixed θ and writes a
   prequential trace, and `camdl compare` already consumes it. Missing: load the
   holdout streams as a second `BoundObs` set, run the post-fit `pfilter` twice,
   report train and holdout loglik separately. gh#277 is the closest tracker.
2. **Real-compartment support in `derived_expr` projections — M**, with an **S**
   interim refusal at `StreamProjection::from_ir`. Item 2.
3. **`camdl verify` — S.** Gives the recorded per-artifact digests their first
   consumer (item 8).
4. **`simulate --sweep NAME=SPEC` — S.** The parser (`args/types.rs:243`) and
   `ParamSource::Sweep` both already exist and are already driven by the shared
   engine; `fit run` and `fit predict` already use the parser. Explicitly
   deferred in the archived 2026-05-28 coherence proposal, never done. Land
   _after_ item 11.
5. **Cache-hit reporting and a working `--force` on `simulate` — S.** Item 13.
6. **`[draws]` in a batch manifest — M.** Posterior-predictive across scenarios
   is CLI-only today and must be scripted as N invocations. Should fold into the
   batch schema-v2 alignment already flagged at `batch.rs:9`.
7. **Structured failure status — S/M.** `RunStatus` is payload-free, so a failed
   leaf records no reason and a crashed national-scale fit leaves a `running`
   record with no diagnosis. Needs a `FORMAT_VERSION` bump.
8. **`output_schema` producers for `fit predict` / survey / profile — S–M.** Six
   vocabulary members are currently unreachable. `predict.rs:1505` already
   computes the coordinate list, band labels and index dims, so the predictive
   case is nearly mechanical.
9. **Populate `parameters_provenance` — S.** The resolver exists; the field
   exists on `fit.meta.json` and was empty in a real fit. "Why is beta 0.4 here"
   currently requires re-deriving the precedence chain by hand.
10. **A `FitLeaf` newtype over the resolved leaf directory — S.**
    `pgas.rs:1030`, `posterior_draws.rs:89` and `compare.rs:911` each spell
    `dir.join("draws.tsv")` by hand, and `mle_params.toml` vs
    `final_params.toml` vs `<algorithm>_summary.json` is per-algorithm knowledge
    duplicated across readers. The spec's fictional §6.8 accidentally described
    a real seam.
11. **A sweep-aware exit status and grid roll-up — S/M.** A ten-value `rho`
    sweep produces ten unrelated directories and no artifact answering "how did
    the fit vary with rho". Build as a derived sidecar, not a layout change;
    `fit/mod.rs:1930` marks it deferred.
12. **`fit run --stage` on a stage with `init_mle` — S/M.** `stage_identities`
    holds only stages run in the same invocation, so the on-disk upstream is
    ignored. Because `--resume` requires `--stage`, resume is currently
    unreachable for the standard scout→posterior pipeline.
13. **Classical SBC — M.** Genuinely absent and distinct from the replicate-fit
    workflow.

Explicitly **not** worth rebuilding, verified as superseded rather than undone:
the `real/` vs `synthetic/` partition, the fit-level `run.json`, and the
`swept_stage_dir` layout (all retired by gh#147); `skip_chains` (rejected under
gh#419 in favour of explicit `--exclude-chains`); `cooling = "auto"` (never
existed, and conflicts with the no-silent-defaults stance); a serialized
`SimulateJob` (rejected on purpose by the 2026-06-17 proposal, to avoid a second
drifting wire schema); and the `#camdl traj v1` marker (superseded by
`run.json.output_schema`).
