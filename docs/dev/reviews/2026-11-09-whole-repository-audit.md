# Whole-repository audit

Date: 2026-11-09 Audited at: `7dd210ba` (main) Status: 14 fixes committed; the
rest listed for review

## Scope and method

The whole tree was read for things that are broken, inconsistent, inefficient or
no longer needed: the OCaml compiler, the Rust runtime (simulation engine,
inference stack, CLI, run identity), the tests and build infrastructure, and the
documentation that claims to describe current behaviour.

Seven read-only agents each read one subsystem and reported findings with a
`file:line`, a scenario and a proposed check. None of them built or ran
anything. Their reports are hypotheses. Each entry below therefore carries one
of four evidence labels, and the label is the thing to read first:

- **reproduced** — I ran it and saw the wrong output; the command or test is
  given.
- **read** — I read the cited code myself and it does what the entry says; not
  executed.
- **agent** — an agent read the code and reported it; I have not re-checked it.
  Treat as a question until reproduced.
- **arithmetic** — the claim follows from a calculation shown in the entry.

Coverage was not complete. Each agent stated what it did not read; the largest
unaudited areas are `expander.ml` lines 6650–12800 (forcings, observations,
interventions, quantities, scenarios), the lineage and spline code in `sim`,
`fit/contrasts.rs`, `fit/path_renewal.rs`, `fit/report/`, the IF2 chain loop in
`fit/runner.rs`, the `fit summary` read path, and most of `util.rs`. Absence of
a finding in those places means they were not examined.

## What was fixed

Each is one commit on local `main`. The compiler fixes are golden-neutral
(`make update-golden` changes no file). Tests marked red→green were run with the
source change reverted and failed.

| Commit     | Defect                                                                                                                                                            | Evidence before the fix                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `404da59d` | Chain-binomial and Gillespie advanced each real compartment by another compartment's derivative when the `ode {}` block order differed from the declaration order | `kw = 1, kv = 0`, 10 days: ODE gives W = 0.0045, V = 100; chain binomial and Gillespie gave W = 100, V = 0 |
| `22453c75` | A `let` reading a parameter through a table cell or a forcing coefficient was hoisted and lost that parameter's rate gradient                                     | `rate_grad` keys `[beta]` with the `let`, `[beta, sigma]` with the same sum written inline                 |
| `c039ebf9` | A name in a `where` guard that is neither an index variable nor a level was compared as a string                                                                  | `where b != adlut` kept 4 instances instead of 2, no diagnostic; `where a == chld` kept none, W200 only    |
| `57b38ae7` | The state derivative of a repeated `PopSum` member was 1, not its multiplicity                                                                                    | `d(I + I)/dI` emitted as `Const 1`; the runtime evaluates `I + I` as 2·I                                   |
| `dee6c5b7` | `simplify_fixpoint` did not terminate when folding produced NaN                                                                                                   | a rate containing `(0 - 8) ^ 0.5` hung `camdlc` (killed after 10 s)                                        |
| `fa7f0e9f` | `camdlc --set beta=abc` raised an uncaught exception                                                                                                              | `Fatal error: exception Failure("float_of_string")`, exit 2                                                |
| `86bb7953` | `[method] thin = 0` passed validation; PGAS then divided by it after claiming the leaf                                                                            | read: `fit/pgas.rs:520`, no check in `FitConfig::validate`                                                 |
| `5da32b41` | The external-validation harness passed any value against a zero reference under a relative tolerance                                                              | `check_scalar(5.0, 0.0, None, Some(0.01))` returned pass                                                   |
| `1c53bc64` | `parse_iso_date` panicked on a date with non-ASCII dashes                                                                                                         | `parse_iso_date("2020–01–01")` panicked in `split_at`                                                      |
| `8d96c111` | `axum`, `tokio`, `tower-http` (cli) and `approx` (sim) were declared and unused                                                                                   | zero references in any `.rs`; lockfile shrinks by 419 lines                                                |
| `368c547e` | A tracked test used an absolute path under a developer's home directory                                                                                           | `browse.rs:1537,1963`                                                                                      |
| `5aad6239` | `inference.yml` did not run for changes to `numerics`, `Cargo.lock`, the toolchain pin or `tests/fixtures`; `cli-docs.yml` omitted `docs/diagnosing-fits.md`      | path filters compared with `sim`'s dependencies and the Makefile's `CLI_DOCS`                              |
| `de79594b` | Four doc statements the code contradicts (`from_posterior` fallback, `dimcheck.ml` location ×2, `dmeasure.rs`, `normalize_for_hash` location)                     | grep                                                                                                       |
| `c531b39f` | The language spec's kind table said `positive` is dimensionless; §2 and §4.1.1 say its dimension is inferred, and the compiler infers it                          | `beta : positive` in `beta * S * I` compiles; `beta : rate` is E300                                        |

`404da59d` affects forward simulation only: chain-binomial inference withholds
the real-compartment capability (gh#191), so no fit ran through that path.

`c039ebf9` is a breaking language change and has an entry in
`docs/language-changes.md`.

`dee6c5b7` leaves a follow-up: the NaN model now compiles with a NaN constant in
its rate (issue 18 below).

Repository housekeeping, also done: 62 worktrees inside `.claude/worktrees/`
removed (unlocked, no uncommitted or untracked files, every commit already on
`origin/main`); 244 local branches deleted on the same criterion, each logged
with its commit hash in `.git/audit-deleted-branches-2026-11-09.tsv`; five
landed worktrees' `rust/target` cleared. Free disk went from 3.5 GiB to 193 GiB.
Local `main` was 123 commits behind `origin/main` and was fast-forwarded before
the audit.

## Open issues

Ordered by severity. "Independent" means the fix can land alone.

### High

1. **The PGAS transition density scores a source group with a zero
   overdispersion multiplier as a Poisson on the un-multiplied rate.** Where:
   `sim/src/inference/pgas.rs:1623`, mirrored at `pgas_grad.rs:174`. The group
   is skipped with `continue` before its members are marked handled, so they
   fall into the ungrouped loop at `:1660`. The sampler
   (`chain_binomial.rs:749`) treats the same case as "no member leaves".
   Evidence: **reproduced** — probe test below; S = 1000, `loss` rate μ·S = 50,
   recorded flows 0, recorded multiplier 0 → density −50, where 0 is correct.
   The spurious term is θ-dependent and enters the MH/NUTS target and the
   ancestor weights. Value and gradient paths agree with each other, so the
   existing spine oracle cannot see it. How often it occurs depends on the gamma
   shape dt/σ²: rare at shape 0.35, frequent at small dt/σ² (the agent's
   estimate of ~0.25 per draw at shape 0.035 is arithmetic, not measured).
   Independent. Disposition: design call D1.

2. **A resumed PGAS chain restarts its parameter-move random stream from the
   beginning.** Where: `sim/src/inference/pgas.rs:3819` builds
   `StatefulRng::new(seed)` before the resume branch, and the resume branch
   (`:3834–3912`) does not replace it; `ChainResumeState` (`:1139`) holds no RNG
   state. PMMH mixes the completed step count in (`pmmh.rs:604`). Evidence:
   **read**. Sweeps K, K+1, … reuse the momentum and accept draws of sweeps 0,
   1, …. Entangled with issue 3. Disposition: design call D3.

3. **A resumed PGAS or PMMH leaf holds only the extension's draws.** Where:
   `cli/src/fit/mod.rs:2551` copies `resume_state.bin` and
   `parameter_traces.tsv`; PGAS and PMMH write `trace.tsv`, and `draws.tsv` is
   rebuilt from this call's sweeps (`fit/pgas.rs:1688`, `fit/pmmh.rs:1082`).
   Evidence: **read** for the copy list; **agent** for the writers. Base
   `sweeps = 3000, burn_in = 2000`, resume to 4000: the leaf keyed as a
   4000-sweep chain would hold 1000 sweeps, and its posterior summaries, R-hat
   and ESS would cover the extension alone, while `docs/camdl-run-spec.md:3572`
   says the leaf is the longer chain. `tests/pgas_resume.rs` asserts identity
   only, never the leaf's contents. Entangled with issue 2. Disposition: D3.

4. **Editing a `[fixed] from_file` parameter file does not change the fit's
   identity.** Where: `cli/src/fit/cas.rs:429` hashes `Problem`, whose
   `FixedParams` (`config_v2.rs:717`) serialises the path string; the contents
   are read at `config_v2.rs:884`. Evidence: **read**. Edit a value in the file
   and re-run: same `run_id`, "cache hit — reusing", and the fit made at the old
   value is served. The path is also read relative to the working directory, so
   one fit.toml run from two directories can read two files under one id (gh#594
   item 2, closed without this part). No test covers `from_file` identity.
   Independent. Disposition: design call D2 (it re-keys).

5. **`profile` reuses random streams across `--seeds` and starts.** Where:
   `cli/src/profile.rs:1711`, `job_seed = seed ^ (grid_idx*1000 + start_idx)`.
   Evidence: **arithmetic** — with 3 starts and seeds 1–3 the nine jobs at a
   grid point get seeds {1,0,3}, {2,3,0}, {3,2,1}: four distinct streams.
   `--seeds` is the documented check for Monte Carlo instability, and it is
   understated. Every other command uses `util::mix_cell_seed`. The clean
   evaluation seed `job_seed + 1` (`:1785`) also equals a neighbouring start's
   seed. Independent. Disposition: design call D4.

6. **A `positive` or `real` transmission parameter absorbs a wrong rate
   dimension with no diagnostic.** Where: `ocaml/lib/ir/dimcheck.ml`. Evidence:
   **reproduced** at `7dd210ba` — `beta * S * I` (density-dependent, where `/ N`
   was intended) is E300 with `beta : rate` and compiles silently with
   `beta : positive`, `positive in [...]` or `real`, with `beta` inferred as
   1/(P·T). This is the textbook error dimensional analysis exists to catch. It
   is documented behaviour (spec §4.1.1). Independent. Disposition: design call
   D5.

7. **Gillespie does not re-evaluate a propensity that reads a real compartment,
   and does not advance real compartments when every propensity is zero.**
   Where: `sim/src/gillespie.rs:536–585` and `:276–345`. Evidence: **agent**.
   Claimed scenario: waterborne-only infection `beta*S*W/(W+k)` with W(0) = 0
   stays at propensity 0 while W grows, until a 10,000-event full recompute.
   `docs/compartmental-ir-spec.md` says Gillespie re-evaluates after each jump.
   Independent. Disposition: reproduce next (an SIWR model under
   `--backend
   gillespie` against chain binomial at small dt); the fix is
   mechanical if it reproduces.

8. **A completed store leaf that reads as stale can never be re-claimed.**
   Where: `runid/src/store.rs:617–621`. Evidence: **agent**. Claimed
   self-inflicted trigger: `camdl survey … --render` writes `landscape.html`
   into a completed leaf (`survey.rs:392`), the file is not in the manifest, the
   leaf becomes stale, and the next run fails with "refusing to clear a
   Completed leaf (concurrent finalize)" with or without `--force`. Copying a
   store without preserving mtimes, or a `.DS_Store`, would do the same.
   Independent. Disposition: reproduce (three survey commands), then design call
   on reclaim versus quarantine.

9. **An unexplained 0.9-nat log-likelihood offset between `camdl pfilter` and an
   independent bootstrap filter.** Where: recorded in the untracked root file
   `bugs.tmp` (2026-11-05): chain-binomial SIR, weekly negative-binomial cases,
   `pfilter` mean −61.2 over 100 replicates against −62.14 ± 0.24. Evidence:
   **not reproduced, not investigated in this audit**; it may sit in the other
   implementation. The separating test is in that file: compare per-observation
   increments on the ODE backend, then check the weekly accumulator boundaries.
   Never filed. Disposition: file it and run that test.

### Medium — inference

10. **A source-less `overdispersed` transition is sampled as negative binomial
    and scored as Poisson.** `pgas.rs:1668`, `pgas_grad.rs:288`; the comment
    says "approximate as Poisson since ungrouped overdispersed transitions are
    rare". σ² of such a transition would get no likelihood term. **read** (the
    comment and the Poisson call are in the function I read for issue 1);
    whether such a model reaches PGAS without a refusal is **agent**. Design
    call: score the marginal negative binomial, or refuse through the capability
    system as `.claude/rules/sim-and-inference.md` requires.
11. **Under `ic_free`, a first observation every particle scores −∞ is dropped
    and the filter returns a finite likelihood.** `particle_filter.rs:539–578`.
    **agent**. Affects `pfilter --fit`, PMMH, profile and survey.
12. **The NUTS energy lacks two −∞ guards the value path has** (rate ≤ ε with a
    recorded flow; a state consuming a different number of gamma multipliers).
    `pgas_grad.rs:141` against `pgas.rs:1428, 1650`. **agent**.
13. **`StatefulRng::new(seed)` is the same stream as `new_stream(seed, 0)`**, so
    a chain's proposal stream coincides with particle 0's.
    `sim/src/rng.rs:506,
    518`. **read**: both constructors expand the same
    key and ChaCha's default stream is 0.
14. **IF2 has no per-particle death mask**: one particle's recoverable error
    ends the run. `if2.rs:457–570`. **agent**.
15. **PMMH's reported MAP log-likelihood is the arithmetic mean of 8
    log-estimates with non-finite ones dropped**; IF2 uses `log_mean_exp`.
    `pmmh.rs:854`. **agent**.
16. Smaller, all **agent**: dual-averaging step size 1.0 when `burn_in ≤ 3`
    (`pgas.rs:4375`); truncated-normal normaliser underflows in the upper tail
    (`prior.rs:315`); discretized-normal value and gradient floor the sd
    differently (`obs_loglik.rs:139` vs `:441`); `mcmc_ess` pairs lags (1,2)
    rather than (0,1) (`pmmh.rs:920`).

### Medium — compiler

17. **`consecutive(dim)` over a one-level dimension is an error instead of no
    transitions.** `expander.ml:4805`. **reproduced**: `age = [all]` with an
    `aging[(a, a_next) in consecutive(age)]` transition gives
    `error[E100]: undeclared name 'S_a'`.
18. **A rate that folds to NaN compiles.** Follow-up to `dee6c5b7`.
    **reproduced**: `(0 - 8) ^ 0.5` in a rate now exits 0. The runtime's rate
    guard refuses NaN rates, so this fails late rather than silently.
19. **State derivatives of a conditional on state are emitted as if smooth**,
    while `min`/`max`/`abs`/`floor` of state are refused.
    `autodiff.ml:825,
    749`. **agent**. Design call (the guard
    `if N > 0 then I/N else 0` must keep working).
20. **`#[lineage]` linearity check accepts a product of parent counts.**
    `lineage.ml:118`; the same multiplicity error fixed in `57b38ae7` is also at
    `lineage.ml:188`. **agent**.
21. **The loop-binding form of `init` builds the cell name from binder order**,
    not the compartment's dimension order. `expander.ml:6626`. **agent**; with
    two dimensions sharing level names the wrong cell would be seeded silently.
22. **`read()` table cells accept `nan`, `inf`, hex and `1_000`; a non-literal
    `default =` is ignored.** `expander.ml:744, 6331`. **agent**.
23. **`read()` value columns are mapped by position with no header check**
    (documented behaviour; a permuted header silently swaps two tables).
    `expander.ml:650`. **agent**. Design call.
24. **Error codes carry unrelated meanings**: E310 (two), E270/E271/E273/E274
    (three each); catalog rows for E200–E221, E230–E236, E250–E254, E260–E262,
    E266–E276 are placeholders rather than per-site entries. **agent**.

### Medium — simulation engine (all **agent** unless marked)

25. **Reactive triggers compare against the accumulated clock with absolute 1e-9
    tolerances** (`reactive.rs:107, 210`): at `dt = 0.1` weekly emits are
    claimed to be missed from t ≈ 2304, and a 28-day window to hold 5
    observations instead of 4 at t = 70. The drift numbers are from a Python
    reproduction of `t += dt`, not from camdl.
26. **The chain-binomial loop takes an extra step for some fractional dt, and
    emits output one substep late once drift exceeds the output tolerance**
    (`chain_binomial.rs:409–486`). Claimed: `dt = 0.1, t_end = 10` runs 101
    steps; late output from t = 197 at `dt = 1/365`. A fix changes bytes for
    fractional-dt runs.
27. **`value_at` compares exactly against accumulated output times**
    (`quantity.rs:717`): `value_at(X, 0.3)` with `every = 0.1` would return the
    value at 0.2.
28. **The `at_day` recurring schedule clamps its first index to k ≥ 0**,
    contradicting spec §13.7 for negative `t_start` (`intervention.rs:93`).
29. **A negative `transfer(count = …)` is a silent reverse transfer**
    (`effects.rs:1106`); `add` refuses negatives.
30. **The ODE backend clamps negative states the discrete backends refuse**
    (`ode.rs:837`), so an ODE likelihood at such θ is finite.
31. **`Piecewise` forcings are not validated** for increasing breakpoints or
    matching lengths (`compiled_model.rs:2040`).
32. **`deterministic(rate)` rounds half away from zero; the spec says ties to
    even** (`chain_binomial.rs:729`). Any deterministic flow with rate·dt < 0.5
    is identically zero, so refining dt removes the flow.
33. **Real compartments are written with six fixed decimals and contrast forks
    read them back as state** (`io/src/trajectories.rs:~338`): a reservoir at
    3.2e-7 restarts at 0.
34. Smaller: initial conditions evaluated at `t = 0` rather than `t_start`
    (`compiled_model.rs:2686`); duplicate table/forcing names not rejected by
    the Rust validator (`ir/src/validate.rs:182`); `Fourier`/`Periodic` return 0
    for a non-positive period while `Sinusoidal` gives NaN; `poisson(NaN)` draws
    from Poisson(1e15) (`rng.rs:527`); the default evaluator does not count
    `log(x ≤ 0)` coercions under `--allow-degenerate-rates`
    (`resolved_expr.rs:634`); the coarse burn-in refusal for `events {}` gives a
    reason that no longer holds (`ode.rs:877`).

### Medium — fit CLI and run identity (all **agent** unless marked)

35. **For PGAS, the stuck-chain screen reads the last K trace rows where K is
    the thinned draw count**, so it covers the last 1/thin of the post-burn-in
    sweeps (`fit/chain_diagnostics.rs:576`).
36. **A `from_mle`/`from_params` source missing an estimated parameter falls
    back to a uniform draw over the model's bounds with a constant seed**,
    ignoring fit.toml bounds (`fit/chain_starts.rs:1142`).
37. **`[config] allow_degenerate_rates` is applied by `fit run` only**; predict,
    profile and survey replay with it off (`fit/mod.rs:424`).
38. **A failed profile cell is stored as completed and cached**, its error
    discarded (`profile.rs:1828`).
39. **`compare --allow-mismatched-horizon --format json` emits Δelpd for pairs
    that skipped every commensurability check** (`compare.rs:420, 2174`).
40. **`compare` checks recorded data hashes but scores the files as they are
    now** (`compare.rs:1003`).
41. **The ensemble run id ignores cell order and arm labels, both of which are
    in the stored `ensemble.tsv`** (`sim_ensemble_cas.rs:150`).
42. **`compare` writes pfilter leaves into `./results`**: 320 per fit at the
    defaults (`compare.rs:1207`).
43. **`survey` integrates at a constant dt = 1.0**, ignoring the model's and the
    fit's dt (`survey.rs:54`).
44. **`camdl cat <sim> --stream NAME` returns the first match in directory
    order** when a leaf holds several cadences (`browse.rs:1002`).
45. Smaller: `compare.toml` precedence contradicts its comment and silently
    drops positional paths (`compare.rs:283`); the default start hash uses
    `DefaultHasher`, which std does not pin across releases
    (`fit/runner.rs:214`); a no-op resume writes an empty summary then fails
    (`fit/pgas.rs:1571`); `mh` is plumbed for resume and refused by the gate
    (`fit/mod.rs:767`); `cmd_cat` ignores write errors (`browse.rs:905–995`);
    `fixture_sha` is written and never compared
    (`external-harness/src/runner.rs:137`); the store's Mode A reclaim removes a
    leaf without checking for a live lock (`store.rs:533`).

### Medium — tests and build infrastructure (all **agent**)

46. **Six statistical tests are documented as "nightly CI" and no nightly job
    runs them.** The only scheduled workflow is `install-e2e.yml`; 32 of 38
    `#[ignore]` tests are run by nothing.
47. **CI's "golden files unchanged" step regenerates nine golden sets and diffs
    four** (`ci.yml:98`); `ir/golden` is in the diff list but nothing
    regenerates it, so that diff is a no-op.
48. **`gradient_check.rs:911` asserts `< 0.3` against a standard error near
    0.02**, so it would have passed with the NUTS U-turn defect of gh#956.
49. **`ocaml/test/errors/*.camdl` (six fixtures for E001 and E100) is read by
    nothing**, and `docs/dev/testing.md:107, 436` instructs developers to add
    fixtures there. No fixture under `ocaml/golden/diagnostics/` expects E001 or
    E100. Verified by grep (**read**). Recommended: move the six into
    `ocaml/golden/diagnostics/` with the three header lines and rewrite the two
    doc passages.
50. Smaller: the pre-push hook is not installed in this checkout and runs raw
    `cargo test` without the camdlc guard handling; the commit-trailer rule has
    no CI enforcement; `docs/language-changes.md`, `agents.md` and `commands.md`
    have code blocks gated by no doctest; doc-only changes to files embedded by
    `include_str!` skip the test job; nothing builds with
    `--no-default-features`; several Makefile recipes write fixed `/tmp` paths.
    The two infrastructure agents disagree on whether CLI tests skip silently or
    fail loudly when a binary is missing (`binary_guard.rs`); unresolved.

### Low — documentation that contradicts the code (all **agent**)

51. **`docs/camdl-inference-spec.md` §4–§7 describes a removed stage workflow**:
    a `camdl fit status` subcommand, IF2 defaults that do not exist (the four
    keys are required), removed `--loglik-eval-*` flags, a `fit_record.json`
    nothing writes, and chains numbered from 1.
52. **`docs/project-structure.md`** gives `ir/VERSION` as 0.20 (it is 0.43),
    names `serialize.ml`/`deserialize.ml` and an `observe` crate that do not
    exist.
53. **`docs/agents.md`**: `degenerate_step_count` does not exist; the
    acceptance-rate bands are wrong; `[estimate].transform` is listed and is
    refused; `camdl docs user-features` should be `camdl docs features`.
54. **`docs/fit-toml.md:37` claims every field is verified against the parser;
    about 25 accepted keys are absent**, and the `burn_in`/`thin` defaults are
    never stated.
55. Smaller: `docs/workflow.md:237` (`max_treedepth`, 20%); `README.md` (~800
    tests, the envelope's missing `docs` field); `docs/dev/testing.md` (silent
    skip, clippy without `--workspace`, a three-platform release);
    `docs/commands.md` omits six subcommands and has an example that fails;
    eight references to `camdl reindex`/`camdl verify`/`camdl fit where`;
    `CHANGELOG.md`'s unreleased section is 1742 commits old and is embedded as
    `camdl docs changelog`; `docs/dev/lifecycle-consolidation-todo.md` and
    `docs/dev/features-roadmap.md` are superseded trackers.

### Low — inefficiency (all **agent**, none profiled)

56. `eval_propensities` iterates the parameter `HashMap` on every call
    (`propensity.rs:~616`). The PGAS density allocates six vectors per call and
    re-evaluates all rates once per source group (`pgas.rs:1554, 1730`).
    `csmc_as` clones particle state every substep (`pgas.rs:3043`). Each ODE RK4
    step allocates about 20 vectors (`ode.rs:103`). The run-identity model level
    is re-hashed two to three times per cell (`cas_sink.rs:255`).
    `read_state_at` re-parses `trajectories.tsv` per (arm, draw). LICM rescans a
    table's cells at every lookup (`licm.ml:63`). Autodiff walks every
    (transition, parameter) pair including unreachable ones (`autodiff.ml:980`).

### Low — no longer needed

57. **Unused code**: `numerics::gammp`, `gammp_inv`, `gamma_pdf_unit` (no caller
    outside their own tests; **read**, by grep); `DensityProcess` trait,
    `ParticleState::clamp_nonneg`, `StatefulRng::exp`,
    `ReactiveAgenda::firings`, `random_starts` parameter always false,
    `FitState.input_hash` always `None`, three OCaml functions
    (`Ir.param_initial_value`, `Time_typing.hint_bare_numeric_data_column`,
    `Latex.derived_ode`) (**agent**). See design call D7.
58. **Deliberate leftovers against the no-shim posture** (**agent**):
    `simulate --cas` no-op, a hidden `camdl if2` stub, eight hidden `_removed_*`
    flags.
59. **Six plotting scripts referenced only by one dev note each**
    (`scripts/plot_*.py`, `blog_index_color_figure.py`). **agent**.
60. **Repository state** (**read**, by command):
    - 73 worktrees remain: 15 locked, 37 with unlanded commits, the rest holding
      uncommitted files or belonging to the active `camdl-docs` session.
    - 115 local branches remain, 84 with commits not on `origin/main`.
    - 226 branches on `origin`, 16 of them ancestors of `main`.
    - Four open PRs untouched since August–September: #744, #741, #603, #581.
    - 20 documents under `docs/dev/notes/` and `docs/dev/proposals/` have never
      been committed (oldest dated 2026-04-03); three tracked proposals cite
      them.
    - Root clutter, untracked: `bugs.tmp` (the source of issues 6 and 9),
      `HANDOFF.md`, five `*-NOTE.md`/`.tsv` review files,
      `Bayesian-Workflow.pdf` (50 MB), `camdl-talk.pptx`, `camdl-claude.zip`,
      two images, `tmp-bench`, `tb-household-probe/` (5.5 GiB). Tracked:
      `AGGREGATION-ARC.md`, which says "delete when the arc lands" (gh#568 is
      still open). `todo.md` is locally excluded and last updated 2026-08-04.
    - Three stashes, the oldest from June.
    - Ten idle `rust-analyzer-proc-macro-srv` processes from sessions back to
      2026-10-23.

## Probe for issue 1

Saved as a `sim` integration test; it fails on `7dd210ba` with "got -50".

```rust
use sim::compiled_model::CompiledModel;
use sim::inference::pgas::log_transition_density_substep;

#[test]
fn zero_gamma_group_with_zero_flows_scores_no_rate_term() {
    let json = std::fs::read_to_string(
        "../../../ocaml/golden/sir_two_overdispersed.ir.json").unwrap();
    let mut model: ir::Model = ir::from_str(&json).unwrap();
    let defaults = [("beta", 0.3), ("gamma", 0.1), ("mu", 0.05), ("sigma_inf", 0.1),
                    ("sigma_loss", 0.1), ("N0", 1000.0), ("I0", 0.0)];
    for p in &mut model.parameters {
        let v = defaults.iter().find(|(n, _)| *n == p.name).map(|x| x.1).unwrap();
        p.value = p.value.with_value(v);
    }
    let compiled = CompiledModel::new(model).unwrap();
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
    }
    // S, I, R, V. I = 0, so `infection` and `recovery` have rate 0; only `loss`
    // (rate mu*S = 50) is live, and it draws one gamma multiplier.
    let counts = [1000_i64, 0, 0, 0];
    let flows = [0_u64, 0, 0];
    let lp = log_transition_density_substep(
        &compiled, &counts, &flows, &[0.0], &params, 0.0, 1.0, None).unwrap();
    // Multiplier 0 ⇒ hazard 0 ⇒ zero exits is certain ⇒ log density 0.
    assert!(lp.abs() < 1e-9, "got {lp}");
}
```

## Design calls

**D1. Fix the PGAS zero-multiplier density now, in the two high-risk files?**
The defect (issue 1) is reproduced and the correct value is not in doubt: the
sampler itself says that a group whose effective hazard is zero has no exits, so
a record of zero flows has probability one and a record of positive flows has
probability zero. The fix is to mark the group's members handled before the
`continue` (returning −∞ if any recorded flow is positive), identically in
`pgas.rs` and `pgas_grad.rs`. What it changes for users: PGAS posteriors on
models with `overdispersed` exits, by an amount that grows as dt/σ² shrinks; no
stored run is re-keyed, but a fit re-run after the fix can differ from a stored
one. The alternative is to leave it until someone measures the posterior impact,
which keeps a known wrong term in the target. Recommendation: fix it, on a
branch, with the probe as its red→green test and a `docs/dev/notes/` measurement
of the posterior shift on `sir_two_overdispersed` at two values of dt/σ², since
the repository rule requires inference changes to land with a measurement.
Confidence: **solid** on the fix; I am asking only because both files are on the
high-risk list.

**D2. Fold the resolved `[fixed] from_file` values into the fit identity?**
Today the identity carries the path, so editing the file serves a stale fit
(issue 4). Option A: hash the resolved name→value map, as `from_scenario`
already is. An edit then re-keys, a rename does not, and running from another
directory no longer matters to the identity. Option B: hash the file's bytes;
simpler, but a comment or whitespace edit re-keys. Either option re-keys every
existing fit that uses `from_file`: those fits stay on disk under their old ids
and would be re-run on next use. Recommendation: A, plus resolving the path
relative to the fit.toml (the unfinished half of gh#594), with a test that edits
a value and asserts the id changes. Confidence: **solid**; needs your
confirmation because it re-keys.

**D3. What is a resumed leaf: the whole chain, or the extension?** The run spec
says the whole chain; the code produces the extension (issue 3), and for PGAS
the extension reuses the first segment's random numbers (issue 2). Option A:
make the code match the spec — carry `trace.tsv` over, rebuild `draws.tsv` and
the summaries over base plus extension, and seed the resumed stream from the
completed sweep count. Summaries on resumed fits then cover all post-burn-in
draws. Option B: define the leaf as extension-only and say so in the spec, the
summary and the leaf name; cheaper, but a resumed fit then reports fewer draws
than its key says. Under either option the RNG reuse must be fixed.
Recommendation: A. Confidence: **leaning** — what would flip me is a reason you
wanted leaves to be independent segments. I have not executed a resume; the
first step is a test asserting the resumed leaf's row count.

**D4. Replace `profile`'s seed derivation with the shared mixer?** The current
XOR collides (issue 5). Using `util::mix_cell_seed(seed, grid_idx, start_idx)`
and a distinct stream for the clean evaluation removes the collisions. It
changes every profile result, and because the run id keys on (seed, point,
start) and not on the derivation, stale leaves would be served unless the
profile method level is bumped in the same change. Recommendation: do both in
one commit. Confidence: **solid** on the defect; **leaning** on bumping the
level rather than adding the derivation to the key.

**D5. Should a `positive` or `real` parameter whose inferred dimension is not
dimensionless be reported?** Inference is the documented design (§4.1.1), and it
is what lets `beta * S * I` pass when `/ N` was meant (issue 6). Option A: an
info line stating each inferred dimension; nothing breaks, and it is easy to
miss. Option B: a warning when the inferred dimension contains a population
power (`1/(P·T)` is the signature of a density-dependent rate written by
mistake), silenced by a `dim` clause; existing models that use `positive` for a
per-capita rate are untouched, and deliberate density dependence costs one
clause. Option C: a hard error whenever inference yields a non-dimensionless
result, which amounts to removing inference for rates and breaks many existing
models. `.claude/rules/dsl-surface.md` prefers hard errors to warnings, which
argues for C over B. Recommendation: B, as an error rather than a warning if you
hold to that rule, restricted to inferred dimensions with a population power.
Confidence: **need you** — this is a language-design call on a checker that
landed three days before this audit, and it depends on how often modellers write
density-dependent transmission on purpose.

**D6. Commit or drop the 20 never-committed documents under `docs/dev/`?** They
date from April to October, three tracked proposals cite them, and they exist
only in this checkout. I did not read them for staleness. Options: commit them
all as they are (preserves them; some are probably superseded), or triage each
as live or superseded first. Recommendation: let me triage them against the code
and the issue tracker and bring you a keep/drop list. Confidence: **need you**
for the drop decisions; they are your notes.

**D7. Delete the unused incomplete-gamma functions in `numerics`?** `gammp` and
`gammp_inv` are validated against SciPy and have had no caller since correlated
PMMH was removed. `.claude/rules/rust-conventions.md` says delete on sight.
Recommendation: delete them and the other items in issue 57, one commit each.
Confidence: **leaning** — I held back only because they are tested numerical
code you may want for a later sampler; say so and I leave them.

**D8. Merge PR #581 (incremental compilation off for dev builds)?** Per-worktree
`rust/target` directories were 17–42 GiB each and filled the disk to 3.5 GiB
free during this audit. The PR's stated measurement is a debug `cargo test` at
6m52s with incremental on and 19s with it off on this machine. Recommendation:
rebase and merge it. Confidence: **leaning**; I did not re-measure.

**D9. Remove the remaining worktrees and remote branches?** I removed only what
was provably lossless. Left: 15 locked worktrees (some from 2026-11-05/06
sessions that may be finished), about 20 landed worktrees with uncommitted
files, 16 remote branches that are ancestors of `main`, and four stale PRs.
Recommendation: I list the uncommitted files in each landed worktree for you to
glance at, then remove; delete the 16 landed remote branches; close #603 and
#744 if their content has landed. Confidence: **need you** — deleting remote
branches and others' uncommitted files is not mine to decide.
