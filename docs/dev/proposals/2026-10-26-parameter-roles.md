---
date: 2026-10-26
status: accepted — D1–D8 decided by the maintainer on 2026-10-26 (§12); D9 (typed-constant entries in existing fit configs) is open; implementation follows §11
area: CLI parameter resolution / inference setup / run records
issues: gh#965 (typed constants); §13 lists the follow-ups
related:
  - 2026-09-27-coordinate-kind-design-vs-sample.md (accepted, partly implemented; §9 states what this amends)
  - archive/post-alpha/2026-05-25-cli-init-and-params-ux.md (the value resolver this builds on)
---

# Every parameter has exactly one role, decided before any value

All code references are against `origin/main` at `1a059920` unless a later
commit is named. A claim marked _verified_ was read at the cited line;
_measured_ means the command shown was run with the `camdl` and `camdlc` built
at `1a059920`, on the SIR recovery case `tests/recovery/cases/sir/` (`beta`,
`gamma`, `N0`, `I0`, `rho`, `k`) with synthetic weekly cases simulated from its
`truth.toml`; _reproduced by an independent check_ means another agent ran it at
`1a059920` and it was not rerun here; _inferred_ means reasoned from the code
and not executed.

## 1. One small partition, resolved first, replaces four selectors and an override system

A camdl run does one of four things with each model parameter. It holds it at
one value (**fixed**), searches over it (**estimated**), sets it per cell from a
grid the user wrote (**design**), or sets it per cell from a draw of a stated
probability measure (**sampled**). That assignment is the run's **partition**:
the statistical question the run answers. A profile likelihood over R₀ with γ
held at 0.1 and ρ re-maximized at every grid point is a different estimand from
one that re-maximizes both.

Today no command computes the partition as an object. `fit run` reads it from
`[estimate]`, `survey` from `[estimate]` or `--estimate`, and `profile` from
`--rw-sd`, the IF2 step-size flag. It is interleaved with value resolution —
`--fixed` removes a name from the estimated set inside the value resolver — and
with special-case refusals that each re-derive part of it. Runs record only
fragments of it, and `profile` records it wrongly (§3).

This proposal adds one typed, exhaustive `Partition` over the model's
`parameters {}` entries, resolved first from role declarations only:
`[estimate]`, `--estimate`, `[fixed]`, `--fixed`, `--fixed-file`,
`--sweep`/`--sweep-file` and `--draws`. Each parameter has at most one
declaration; two are an error, except for three legitimate special cases —
profiling, sweeping a held parameter, and design-vs-sample's overrides of a
measure's coverage (§4). Values are then resolved per role through the existing
value tiers. Typed constants (`let omega : rate = 0.01`) are not configurable.
Algorithm settings never decide a role; a scenario supplies values and
interventions, never roles; starts come only from `--starts` or
`[estimate].start`. Model support, search bounds and priors are separate types.
The run prints and stores the partition whole. The aim is a smaller,
statistically coherent architecture, not a general override system; §8 lists
what the simplification removes.

## 2. The terms: parameters, constants, roles, declarations and value tiers

A **parameter** is a `parameters {}` declaration. It compiles to
`ParamValue::Estimated` (when it carries bounds or a prior) or
`ParamValue::Required` (when it carries neither), never with a value — verified,
`ocaml/lib/compiler/expander.ml:5711-5727`. An indexed declaration
`beta[region]` compiles to one scalar parameter per cell, named `beta_north`,
`beta_south` (`expander.ml:5805-5865`). The IR's `ParamValue::Estimated` names a
parameter that _carries inference configuration_; it is not a role:
`beta : rate in [0.05, 1.0]` is held in `simulate` and estimated in `fit run`.

A **typed constant** is a `let` with a type and a constant body, such as
`let omega : rate = 0.01`. It compiles to an IR parameter with
`ParamValue::Fixed { value }` (`expander.ml:5871-5881`; measured: `omega` gets
`{"mode": "fixed", "value": 0.01}`), so the dimension checker sees its kind
(language spec §8.4, `docs/camdl-language-spec.md:1807-1830`). It is part of the
model, not a configurable parameter (D2): no command may fix it at another
value, estimate, sweep or sample it. The spec's sentence that it is
"indistinguishable, once compiled, from the same value declared in
`parameters { }`" (`:218-220`) overstates the equivalence: until `fe412028`
another `let` over it stopped the compiler with E512, and `fit run` still
demands a `[fixed]` entry for it (gh#965). Stage 4 rewrites that sentence.

A **role declaration** is an input that names a parameter and says what the run
does with it. A **value source** gives a parameter a number. A scenario is a
value source (its `set` and `scale`) plus an **intervention set** (its `enable`
and `disable`); it never declares a role. The **varying set** of a measure is
the parameters it actually varies: for a posterior the estimated parameters, for
`prior` those with a prior, for `uniform` those with declared bounds
(design-vs-sample §2). A measure's coverage is _implicit_: the user did not name
those parameters.

The **value tiers** order the sources of a value, later winning. They are those
of `docs/camdl-run-spec.md` §1.3, implemented in
`params_resolver::resolve_parameters_with_starts` (verified,
`params_resolver.rs:1422-1824`), with the split design-vs-sample §7.4 makes and
without today's tier 1.5 (a `--starts` file's values) and model-default tier:

| Tier | Source                                                                |
| ---- | --------------------------------------------------------------------- |
| 2    | the fit config's `[fixed]`                                            |
| 3    | `--fixed-file` (inference commands) / `--params` (`simulate` and kin) |
| 3.5a | the cell's sample row                                                 |
| 3.5b | the cell's design point                                               |
| 4    | the scenario's `set` and `scale`                                      |
| 5    | `--fixed` / `--param` / `--param-vec`                                 |

Two terms come from design-vs-sample §2 and §6.2. An **imposition** is a value a
scenario or `--param` puts on a parameter the sample varies: every draw then
runs at that value while its other parameters keep their sampled values. A
**design shadow** is a design value on a parameter the sample varies: the design
value wins in each cell.

Three properties of an estimated parameter are kept apart because they mean
different things. Its **support** is the set of values the model admits — the
model's declared `in [lo, hi]`, else its kind's range; a value outside it is not
a value of this model. Its **search box** is where the method looks — the fit
config's or `--estimate`'s bounds, within the support; it is part of a
likelihood estimand, because a narrower box can change a maximum. Its **prior**
is a belief, read only by the Bayesian methods.

The profile-likelihood notation is θ = (ψ, λ, φ): ψ the focal (design)
parameters, λ the nuisance (estimated) parameters, φ the fixed ones. The profile
is ℓ_p(ψ; φ₀) = max over λ ∈ Λ of ℓ(ψ, λ, φ₀), where Λ is the search box.

## 3. Each command decides roles differently today, and several decide wrongly

### 3.1 Four commands, four selectors

| Command                       | What decides "estimated" (verified)                                                                                                                       | Undeclared parameters                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `fit run`                     | `[estimate]` keys; every model parameter must be in `[estimate]` or `[fixed]` (`fit/config_v2.rs:2906-2914`)                                              | refused                                                     |
| `fit run --sweep`             | as above; a swept name must be in `[fixed]` and not in `[estimate]` (`fit/mod.rs:519-535`)                                                                | refused                                                     |
| `survey --fit`                | `[estimate]` keys (`survey.rs:763`)                                                                                                                       | resolved from the tiers                                     |
| `survey` inline               | `--estimate NAME=LO:HI` (`args/mod.rs:2026`, `survey.rs:1026-1032`)                                                                                       | resolved from the tiers                                     |
| `profile`                     | `--rw-sd`: an explicit list estimates exactly its names; `auto` estimates every parameter not focal and not in `[fixed]`/`--fixed` (`profile.rs:750-797`) | held at whatever the tiers give, a `--starts` file included |
| `simulate`, `pfilter`, `eval` | nothing is estimated (`fit_toml_estimate` empty: `util.rs:3391`, `pfilter.rs:208`, `eval.rs:103`)                                                         | resolved from the tiers                                     |
| `fit predict`                 | the fit's posterior draws carry every parameter (`fit/predict.rs:1604-1636`)                                                                              | refused (every parameter must be a draw column)             |

Inside the resolver, `[estimate]` membership is narrowed by `--fixed` and
`--fixed-file` (`params_resolver.rs:1675-1717`, verified), and each removal is
reported as a `PinnedOverEstimate` notice (`params_resolver.rs:353-377`).

### 3.2 The defects follow from deciding the role in several places

Items marked _interim_ are being fixed now, in changes separate from this
proposal; the design does not depend on them staying broken, and each interim
fix is superseded by the partition.

1. **`profile` takes the estimated set from an algorithm setting.** Measured: a
   fit config listing `beta`, `gamma` and `k` in `[estimate]`, `--sweep k=…`,
   `--rw-sd beta=0.02` estimates only `beta`; `gamma` is held at its
   `[estimate].start` in every cell, with no notice. `--rw-sd` is also required
   by the NLopt per-cell algorithms, which never read it (`profile.rs:317-320`).
2. **`--rw-sd auto` estimates parameters the user fixed in a file** (interim).
   It excludes `--fixed` and `[fixed]` names only (`profile.rs:750-755`), not
   `--fixed-file` keys; reproduced by an independent check, `gamma`, `rho` and
   `k` given by a `--fixed-file` came out estimated.
3. **`--rw-sd auto` estimates typed constants** (interim). Measured: on the SIR
   model with `let omega : rate = 0.01` added,
   `camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10 --fixed gamma=0.15 --rw-sd auto --starts from_params=b.toml`
   writes `[mle] omega = 0.010000000000000007` and records
   `"estimated": ["omega"]` (`profile.rs:786-791` filters `model.parameters`,
   which includes typed constants).
4. **`profile` records the wrong roles and drops simplex groups** (interim).
   Same run: the base segment's `fit.meta.json` has `"fixed": {}` and
   `"parameters_provenance": {}`; each leaf's `run.json` marks all seven
   parameters `"role": "fixed"`, including the focal `beta` and the estimated
   `omega` (`params_resolver.rs:1767-1775`). The per-cell `IF2Config` is built
   with `simplex_groups: vec![]` (`profile.rs:1490`, verified).
5. **A starting point is recorded as a model default** (interim). Measured with
   `fit run` (`beta = { …, start = 0.25 }`): `fit.meta.json` records
   `"beta": {"value": 0.25, "source": "model_default", "role": "estimated"}`,
   and under the default `starts = "from_prior"` 0.25 is not the start used. The
   start is written into the model before the resolver runs
   (`fit/runner.rs:223-246`), and the resolver labels any pre-existing value
   `ModelDefault` (`params_resolver.rs:1438-1443`). The same pre-seeding loop is
   in `profile.rs:453-461` and `survey.rs:705-712`, `:805-826`, `:960-966`.
6. **`survey --fit` ignores `--fixed`** (interim). Measured:
   `camdl survey sir.camdl --fit fit.toml --fixed beta=0.3 --n-points 4 --eval-particles 20`
   exits 0 and its `landscape.tsv` samples `beta` at 0.226 and 0.545
   (`survey.rs:785-787` passes an empty `fixed_cli`).
7. **`fit run` refuses a typed constant it does not need** (gh#965, interim).
   Measured: `camdl fit run fit_let.toml` exits with
   `parameters neither estimated nor fixed: omega`.
8. **PMMH silently ignores the IF2 settings** (interim). Reproduced by an
   independent check: `perturb_only_at_t0` and `rw_sd` in `[estimate]` change
   nothing under PMMH, and the loader says nothing
   (`fit/config_v2.rs:2403-2407`, verified; the field's doc comment at
   `:606-612` calls it a config-load error, which is stale).
9. **A grid value outside the bounds runs** (interim; the fix refuses it).
   Reproduced by an independent check: `profile --sweep beta=2.0` runs with
   `beta in [0.05, 1.0]`, while a start of 2.0 is refused.
10. **A swept parameter still needs a value from elsewhere.** Measured: sweeping
    `beta` with no other source fails with
    `parameter 'beta' has no value: no model default, …  Fix: declare a default in the .camdl model, or pin via --fixed beta=<value>`.
    Both fixes are wrong.
11. **The same scenario means two things.** In `fit run` a scenario setting an
    estimated parameter is refused (`params_resolver.rs:1219-1272`, `2b872798`);
    in `profile --rw-sd auto` its value becomes the starting point: it is in
    `base_params`, which becomes each estimated parameter's `initial`
    (`fit/runner.rs:1288`, called from `profile.rs:818-820`; verified by
    reading). Ten test files run `profile --scenario baseline` this way (§10.2).
12. **An estimated hyperparameter over held leaves has the wrong posterior** (to
    be reproduced; interim refusal in progress). The Bayesian methods sum the
    log-prior over the estimated parameters only (PMMH:
    `sim/src/inference/pmmh.rs:599-602`, verified; PGAS inferred), so a held
    leaf's density p(leaf | hyper) is missing (§6.4, D8).

## 4. One declaration per parameter; three special cases; everything else is an error

### 4.1 Who declares a role

| Role      | Declared by                                                                                                                                       | Its value in a cell comes from                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| fixed     | `[fixed]` (inline, `from_file`, `from_scenario`), `--fixed`, `--fixed-file`, `--param`/`--params`/`--param-vec`, or nothing (D3)                  | tiers 2–5                                                   |
| estimated | `[estimate]`, `--estimate`                                                                                                                        | the method; its start from `--starts` or `[estimate].start` |
| design    | `--sweep`, `--sweep-file`, `fit run --sweep`, `fit predict --sweep`, `profile --sweep`                                                            | tier 3.5b, the design point                                 |
| sampled   | the varying set of `--draws prior\|uniform\|posterior`, `--draws FILE` (`--draws-file` after design-vs-sample Stage 5), `fit predict`'s posterior | tier 3.5a, the draw row                                     |

Nothing else declares a role. `--rw-sd`, `[estimate].rw_sd`,
`perturb_only_at_t0`, `--starts` and a scenario do not. A role declaration
naming a typed constant is refused: "`omega` is a typed constant in the model;
move it into `parameters {}` to configure it" (D2).

### 4.2 The rule

**Each parameter is named by at most one role declaration. A parameter named by
two declarations of different roles is refused**, naming both and the fix. Two
fixed declarations are not a conflict: they are value tiers (a `--fixed` over a
`[fixed]` entry is tier 5 over tier 2, with an override record, as today).

Three cases are the operations camdl exists to run, and are not conflicts:

1. **Profiling a fit config's estimated parameter.** In `profile`, a `--sweep`
   name that the fit config lists in `[estimate]` is the focal parameter; the
   rest of `[estimate]` is the nuisance set. The `[estimate]` entry's search
   box, start, prior and IF2 settings are not used for it; one info line says
   so, and the grid must lie inside that entry's search box (§5.2). Outside
   `profile`, a `--sweep` over `[estimate]` is refused (`fit run --sweep` over
   `[estimate]` is a grid of conditional fits, §13).
2. **Sweeping a held parameter** (D1). A `--sweep` over a `[fixed]` entry: at
   job level the parameter is a design coordinate; in each cell it is fixed at
   the grid value. The `[fixed]` value is not used, and one info line says so.
   This is what `fit run --sweep` requires today (`fit/mod.rs:519-535`).
3. **A named declaration over a measure's coverage.** A parameter that is
   sampled only because a measure covers it takes a `--sweep` value (a design
   shadow) or a `--param`/`--fixed` value (an imposition by `--param`), recorded
   as design-vs-sample §6.2 and §8 record them. These are its accepted
   semantics.

Everything else that names one parameter twice is an error, including the two
command-line overrides of a fit config that resolve silently or with a notice
today:

- **`--fixed` or `--fixed-file` over `[estimate]` is refused.** Today `--fixed`
  removes the parameter from the estimated set with an info notice (`21d4ed47`)
  and `--fixed-file` does the same (`params_resolver.rs:1675-1717`). Both turn
  the fit config's statistical problem into a conditional one while the fit
  config, which is the record of the problem, still says the parameter is
  estimated. It is not needed: holding a parameter is one line moved from
  `[estimate]` to `[fixed]` in a copy of the config, and that copy then names
  the problem it runs. The error says exactly that.
- **A `--fixed-file` entry naming a swept or `--estimate` parameter is
  refused.** A file written for another purpose — a fit's `mle.toml` — names
  every parameter; using it to hold "the rest" while also sweeping or estimating
  some of them was a silent precedence. The error names the entry and offers
  `--starts from_params=FILE` when the file was meant as starting values.

Examples:

| Input                                                                                  | Result                                              |
| -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `[estimate]` + `[fixed]` naming one parameter, one file                                | refused (already refused, `6c94825f`)               |
| `profile --fit F --sweep beta`, `beta` in `[estimate]`                                 | valid: `beta` focal (case 1)                        |
| `fit run --sweep beta`, `beta` in `[estimate]`                                         | refused (§13)                                       |
| `--sweep N0`, `N0` in `[fixed]`                                                        | valid: design at job level, fixed per cell (case 2) |
| `--fixed X` + `--sweep X`; `--estimate X` + `--fixed X`                                | refused                                             |
| `--fixed rho`, `rho` in `[estimate]`                                                   | refused (reverses `21d4ed47`)                       |
| `--fixed-file mle.toml --sweep beta`, `mle.toml` naming `beta`                         | refused                                             |
| `--draws uniform --sweep beta`                                                         | valid: design shadow (case 3)                       |
| `--draws uniform --param beta=0.5`                                                     | valid: imposition by `--param` (case 3)             |
| `--estimate omega` or `--fixed omega=…`, `omega` a typed constant                      | refused (D2)                                        |
| `--rw-sd NAME=SD` or `perturb_only_at_t0` on the command line for a non-estimated name | refused                                             |

`--estimate` and `--fit` are mutually exclusive, as in `survey` today
(`args/mod.rs:2020`): with a fit config, `[estimate]` alone declares the
estimated set.

### 4.3 A scenario is checked against the role of each parameter it touches

| Role of the parameter the scenario sets or scales | Result                                                                                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| fixed                                             | the scenario's value, at tier 4; refused when the same fit config's `[fixed]` gives another value (`scenario_fixed_collision`, unchanged) |
| estimated                                         | refused                                                                                                                                   |
| design                                            | refused (gh#572; in `profile` this replaces today's info notice, `09ccbed1`)                                                              |
| sampled                                           | valid: an imposition by the scenario, tier 4 over tier 3.5a, recorded per arm                                                             |

A scenario is per arm (`simulate --scenario a,b`), so it changes no role: an
imposition by a scenario is a value override inside the sampled role, and the
partition is the same for every arm.

### 4.4 Each command accepts a declared set of roles

| Command                   | fixed | estimated                            | design                       | sampled                  |
| ------------------------- | ----- | ------------------------------------ | ---------------------------- | ------------------------ |
| `simulate`                | yes   | no                                   | optional                     | optional                 |
| `pfilter`, `eval`         | yes   | no                                   | no                           | no                       |
| `fit run`, `fit recovery` | yes   | yes                                  | optional (one fit per point) | no                       |
| `fit predict`             | yes   | no                                   | optional                     | required (the posterior) |
| `survey`                  | yes   | at least one                         | no                           | no                       |
| `profile`                 | yes   | at least one, or none with `--slice` | at least one                 | no                       |

A `profile` with nothing estimated is a likelihood slice: every non-focal
parameter held. It is refused unless the user says so with `--slice`: dropping
`--rw-sd auto` without adding `--estimate` would otherwise turn a full profile
into a slice, every nuisance parameter fixed by D3, with nothing said. The
refusal names `--estimate NAME,…`, `[estimate]` in a `--fit` config, and
`--slice`.

A declaration of a role the command does not accept is refused naming the
command, for example `--estimate` on `simulate`. `lineage` reads an event log
and takes no parameter flags (`args/mod.rs:2250-2288`, verified); `if2` is a
removed-command trap (`if2.rs:16`). Neither changes.

### 4.5 Names are scalar IR names

The partition is keyed by the IR's scalar parameter names, so an indexed
parameter has one role per cell (`beta_north`). A declaration naming the family
`beta` when no scalar `beta` exists is refused as an unknown name; today
`--sweep` lists only the unknown name (`sweep.rs:65-73`, verified), and the new
message also lists the family's cells. `simulate --param-vec beta=FILE` expands
to `beta_<key>` names before roles are resolved (`util.rs:3360-3362`, verified).

## 5. Values are resolved per role, and checked against the constraints

### 5.1 A fixed parameter takes tiers 2–5

A fixed parameter's value comes from the tiers of §2, with the same override
records and notices as today (`b262fde0`). Under D3, an undeclared parameter is
fixed: if a source gives it a value (a scenario `set`, a `--fixed-file` or
`--params` key, a sample row's constant column), it is held at that value and
the record names the source; if none does, the run is refused with "parameter
`k` has no value and no role: give it one with `--fixed k=…`, `--fixed-file`,
`[fixed]` or a scenario that sets it, or estimate it with `--estimate k`". A
typed constant holds its declared value and is recorded under `constants`, not
in the partition.

### 5.2 Design and sampled parameters take their own tier; estimated take none

A design parameter's value in a cell is tier 3.5b and a sampled one's tier 3.5a.
Tiers 2–3 cannot supply them: a `[fixed]`, `--fixed-file` or `--starts` value
for one is §4.2 case 2, a refusal, or (for `--starts`) unused. Over a sampled
parameter, tiers 4 and 5 are impositions; over a design parameter both are
refused.

An estimated parameter takes no tier value. Its start comes from the starts rule
(§5.3); the slot the compiled model needs is filled from that start. This
removes the four pre-resolver seeding loops (§3.2 item 5) and the "a swept
parameter needs a value" error (item 10).

### 5.3 Starts come only from `--starts` or `[estimate].start`

The starts rule is the existing `ChainStarts` (`fit/starts.rs:48-85`),
unchanged. A starts source (`from_params`, `from_mle`, `from_posterior`)
supplies values for estimated parameters only. A name it carries that has
another role is not used, and the launch block lists such names once; a starts
source is not a role declaration, so this is not a conflict. An estimated
parameter the source does not name starts at its `[estimate].start`, else at a
transform-aware draw within its search box (`fit::init::draw_start_in_bounds`,
now owned by the starts resolver). `--starts from_scenario` is not part of this
proposal (D6).

### 5.4 Constraints are checked at resolution and at every cell

Each value is checked against what constrains it, once structurally when the
partition is resolved and again for every cell that gives it a value:

| Constraint                             | At resolution                                                               | At each cell / point                           |
| -------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| support (`in [lo, hi]`, else the kind) | search box ⊆ support; `--estimate` bounds ⊆ support                         | every fixed, design and sample value ∈ support |
| search box (estimated parameters)      | start ∈ box                                                                 | —                                              |
| a profile's grid                       | every grid value ∈ the focal's search box (§4.2 case 1), else ∈ its support | —                                              |
| simplex group (§6.4)                   | membership: all members one role                                            | fully held group: Σ members = 1 within 1e-9    |
| hierarchical prior (§6.4, D8)          | no estimated hyperparameter over a held leaf (PMMH, PGAS)                   | —                                              |

A profile's grid respects the fitted search box unless the user changes the box
itself, in the fit config's `[estimate]` entry or with `--estimate NAME=LO:HI`;
there is no flag that admits grid values outside it. The interim fix of §3.2
item 9 refuses grid values outside the bounds, which is this rule.

### 5.5 The intervention set is resolved separately

The intervention filter already runs apart from value precedence
(`params_resolver.rs:1505-1526`, verified). It becomes its own result,
`InterventionSet`, so a scenario's `enable`/`disable` apply whatever roles its
`set`/`scale` keys have. `--scenario` still conflicts with
`--enable`/`--disable` (`args/mod.rs:147-157`).

## 6. The types: a partition, separate support, box and prior, and the problem

### 6.1 What the partition does and does not hold

A role says _that_ a parameter is a design coordinate or a sample column; the
axes say _where_ the points are. A 2-D profile's grid is a product
(`profile.rs:1015-1027`), a `--sweep-file` row sets several parameters jointly
(`sim_job.rs:166-178`), and a measure is over the joint draw, so
design-vs-sample's `DesignAxis` and `SampleAxis` own the points. Fixed values
are per scenario arm, so they are not in the partition either. Typed constants
are not in it, because they are not configurable.

### 6.2 The partition

```rust
// rust/crates/cli/src/roles.rs (new). The partition holds no values.

/// What a run does with one `parameters {}` entry.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// One value per scenario arm, from tiers 2–5.
    Fixed,
    /// Searched by the method.
    Estimated(Estimated),
    /// Set per cell from the design axis (tier 3.5b); fixed inside each cell.
    Design,
    /// Set per cell from the sample axis (tier 3.5a); fixed inside each cell.
    Sampled,
}

/// The model's admissible values: `in [lo, hi]`, else the kind's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Support { pub lower: f64, pub upper: f64, pub from: SupportFrom }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportFrom { Declared, Kind }

/// Where the method searches: within the support. Part of a likelihood estimand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchBox { pub lower: f64, pub upper: f64, pub from: BoxFrom }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxFrom { FitConfig, Cli, Support }

/// An estimated parameter: its support, search box and prior are distinct.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimated {
    pub support: Support,
    pub search: SearchBox,
    pub transform: sim::inference::types::Transform,   // clamps to `search`
    pub transform_from: TransformFrom,
    /// Read by the Bayesian methods only; always recorded.
    pub prior: ResolvedPrior,                          // fit::priors_precedence
    /// `[estimate].start`, if written.
    pub start: Option<f64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformFrom { FitConfig, Kind }

/// What declared a parameter's role.
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    /// Named by nothing: fixed (D3).
    Default,
    FitEstimate,
    FitFixed(FitFixedVia),
    CliEstimate,
    CliFixed,                     // --fixed, --param, --param-vec
    CliFixedFile(PathBuf),        // --fixed-file, --params
    Design(DesignOrigin),         // design-vs-sample §7.1
    Sample(MeasureTag),
    /// `fit predict`: the stored fit's own `[fixed]` block.
    FitRecord,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FitFixedVia { Inline, FromFile(PathBuf), FromScenario(String) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleKind { Fixed, Estimated, Design, Sampled }

/// One of §4.2's three special cases, when it applied.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecialCase {
    ProfileFocal,                 // case 1: the fit config's [estimate] entry is not used
    SweepOverFixed,               // case 2: the [fixed] value is not used
    DesignShadow,                 // case 3
    ImpositionByParam,            // case 3
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoleEntry {
    pub role: Role,
    pub declared_by: Declaration,
    pub special: Option<SpecialCase>,
}

/// Exhaustive: exactly one entry per `parameters {}` name, in declaration
/// order. Built only by `resolve_roles`.
#[derive(Debug, Clone)]
pub struct Partition {
    entries: IndexMap<String, RoleEntry>,
    constants: IndexMap<String, f64>,      // typed constants, recorded only
    simplex_groups: Vec<ResolvedSimplex>,  // §6.4
}

impl Partition {
    pub fn role(&self, name: &str) -> &Role;
    pub fn estimated(&self) -> impl Iterator<Item = (&str, &Estimated)>;
    pub fn names_with(&self, kind: RoleKind) -> impl Iterator<Item = &str>;
    /// One info line per special case that applied.
    pub fn notices(&self) -> Notices;
    /// The partition one cell runs: design and sampled parameters fixed at that
    /// cell's values, tagged with their tier (§7.2).
    pub fn cell(&self, design: &DesignPointRef, sample: &SampleRowRef) -> CellPartition;
    pub fn record(&self) -> PartitionRecord;
}

impl PartitionRecord {
    /// The legacy `estimated` and `fixed` sidecar fields, generated here and
    /// nowhere else (D4).
    pub fn legacy_fields(&self) -> (Vec<String>, BTreeMap<String, f64>);
}
```

### 6.3 The resolved problem and the resolver signatures

```rust
pub struct CommandRoles {
    pub command: &'static str,
    pub estimated: Allowed,
    pub design: Allowed,
    pub sampled: Allowed,
}
#[derive(Clone, Copy)]
pub enum Allowed { No, Optional, AtLeastOne, AtLeastOneUnlessSlice }
pub const SIMULATE: CommandRoles;    pub const PFILTER: CommandRoles;
pub const EVAL: CommandRoles;        pub const FIT_RUN: CommandRoles;
pub const FIT_PREDICT: CommandRoles; pub const SURVEY: CommandRoles;
pub const PROFILE: CommandRoles;

/// The role declarations. Values are not here.
pub struct RoleInputs<'a> {
    pub model: &'a ir::Model,
    pub fit_estimate: &'a IndexMap<String, EstimateSpecV2>,   // empty without --fit
    pub fit_fixed: &'a [(String, FitFixedVia)],
    pub simplex_groups: &'a [config_v2::SimplexGroup],
    pub cli_estimate: &'a [EstimateDecl],      // --estimate NAME[=LO:HI]
    pub cli_fixed: &'a [String],               // --fixed / --param / --param-vec names
    pub cli_fixed_files: &'a [(PathBuf, Vec<String>)],
    pub design: &'a [(String, DesignOrigin)],
    pub sample: Option<(&'a [String], MeasureTag)>,   // the sample axis's varying set
    pub slice: bool,                           // profile --slice
    pub method: FitAlgorithm,                  // for the D8 refusal
    pub scenarios: &'a [ScenarioRef],          // each arm, for §4.3
}

pub fn resolve_roles(inputs: &RoleInputs<'_>, command: &CommandRoles)
    -> Result<Partition, RoleError>;

#[derive(Debug, Clone)]
pub enum RoleError {
    TwoDeclarations { name: String, first: Declaration, second: Declaration },
    TypedConstant { name: String, by: Declaration },
    UnknownName { name: String, by: Declaration, candidates: Vec<String> },
    NotAccepted { name: String, role: RoleKind, command: &'static str },
    Missing { role: RoleKind, command: &'static str },
    ScenarioOnRole { scenario: String, name: String, role: RoleKind,
                     actions: Vec<ScenarioParamAction> },   // params_resolver.rs:797-806
    OutsideSupport { name: String, value: f64, support: Support },
    BoxOutsideSupport { name: String, given: (f64, f64), support: Support },
    GridOutsideBox { name: String, value: f64, search: SearchBox },
    UnboundedForMethod { name: String, method: &'static str },
    SimplexPartlyHeld { group: Vec<String>, held: Vec<String> },
    SimplexSum { group: Vec<String>, cell: Option<u64>, sum: f64 },
    HyperOverHeldLeaf { hyper: String, held_leaves: Vec<String>, method: &'static str },
    SettingForNonEstimated { name: String, setting: &'static str, role: RoleKind },
}

pub struct If2ParamSettings {
    pub rw_sd: f64,
    pub rw_sd_from: RwSdFrom,                  // Cli | FitConfig | Auto
    pub perturb_only_at_t0: bool,
}

pub enum AlgorithmSettings {
    If2 { per_param: IndexMap<String, If2ParamSettings>, /* particles, iterations, cooling */ },
    /// PMMH, PGAS, NLopt, pfilter: no per-parameter settings; `[estimate]`
    /// entries' IF2 settings are listed as inert in the launch block.
    Other { name: &'static str, inert: Vec<(String, &'static str)> },
}

impl AlgorithmSettings {
    /// Refuses `--rw-sd NAME=SD` for a non-estimated `NAME`, and any `--rw-sd`
    /// under a method other than IF2.
    pub fn resolve(partition: &Partition, method: FitAlgorithm,
                   cli_rw_sd: &IndexMap<String, f64>,
                   fit_estimate: &IndexMap<String, EstimateSpecV2>)
        -> Result<Self, RoleError>;
}

pub struct InterventionSet { pub scenario: Option<String>, pub enable: Vec<String>,
                             pub disable: Vec<String> }

/// What an inference command runs.
pub struct ResolvedProblem {
    pub partition: Partition,
    pub fixed: ResolvedParameters,             // fixed parameters only, with provenance
    pub interventions: InterventionSet,
    pub starts: fit::chain_starts::DrawnStarts,   // §6.5
    pub algorithm: AlgorithmSettings,
}

pub fn resolve_problem(partition: Partition, values: ParameterInputs<'_>,
                       starts: &fit::chain_starts::ResolvedStarts,
                       algorithm: AlgorithmSettings, n_chains: usize, seed: u64)
    -> Result<ResolvedProblem, ResolveError>;
```

The value resolver is renamed, so the change cannot be confused with the wrapper
it replaces:

```rust
// params_resolver.rs: before
pub fn resolve_parameters(inputs: ParameterInputs<'_>)                        // :1413
    -> Result<ResolvedParameters, ResolveError>;
pub fn resolve_parameters_with_starts(inputs: ParameterInputs<'_>,
    starts: Option<StartsValues<'_>>) -> Result<ResolvedParameters, ResolveError>;

// after
pub fn resolve_values(partition: &Partition, inputs: ParameterInputs<'_>)
    -> Result<ResolvedParameters, ResolveError>;
// ParameterInputs without `fit_toml_estimate`; `point_overrides` splits into
// sample and design overrides with design-vs-sample Stage 1 (its §7.4).
```

`resolve_parameters` has eight production callers (`util.rs:3393`,
`survey.rs:776`, `:992`, `pfilter.rs:210`, `eval.rs:105`, `contrasts.rs:667`,
`runner.rs:289`, and `profile.rs:589` through the `_with_starts` form) and about
40 test callers in `params_resolver.rs`'s own module; each moves to
`resolve_values` in the stage that moves its command, and the old name is
deleted in Stage 4. `ResolvedParameter.role: ParameterRole` and `FixReason`
(`params_resolver.rs:255-275`), `ResolvedParameters.estimate_set` and
`pinned_over_estimate` go.

### 6.4 Domains, simplex groups, hierarchical priors and IF2 settings

**Support, search box and transform.** The support is the model's `in [lo, hi]`,
else the kind's range: `probability` `[0, 1]`; `rate`, `positive`, `count`,
`duration` `[0, ∞)`; `real`, `instant` `(−∞, ∞)` — the classification
`kind_support`'s doc gives (`params_resolver.rs:108-117`). Today an undeclared
range falls back to `[0, ∞)` for every kind (`fit/runner.rs:1130`, `:1273`,
verified), a box that excludes the negative values a `real` or `instant`
parameter admits; the partition corrects it. The search box is the fit config's
or `--estimate`'s bounds, refused if it leaves the support (today's
tighten-not-loosen rule, `fit/runner.rs:1250-1275`), else the support. The
transform comes from `[estimate].transform`, else the kind, and clamps to the
search box (`fit/runner.rs:1240-1250`). A method that needs a finite box —
`survey`'s Latin hypercube, the `uniform` and `lhs` starts rules — refuses an
infinite box naming the parameter. The prior is resolved by
`fit::priors_precedence` (fit config, then the model's `~`, then flat). Because
the transform clamps to the box, a Bayesian fit's effective prior is the stated
prior restricted to the box (inferred for PMMH and PGAS); the record stores
both.

**Simplex groups** are sets of estimated parameters constrained to sum to 1,
which IF2 perturbs jointly through a log-ratio transform whose softmax inverse
fixes the sum at 1 (`sim/src/inference/if2.rs:58-67`,
`fit/config_v2.rs:90-106`):

- every member estimated: the group runs as today, and `profile` passes it to
  its per-cell `IF2Config`;
- every member fixed or design: the group becomes a check, refused at any cell
  whose member values do not sum to 1 within 1e-9;
- some members held and some estimated: refused. The free members would have to
  sum to 1 − Σ(held), a scaled simplex `SimplexGroup` cannot express (it has no
  total, `if2.rs:62-67`); renormalizing them to 1 would silently fit a different
  model.

**Hierarchical priors.** A leaf's prior refers to its hyperparameters as
ordinary parameters (`HierarchicalPrior.args`, `ir/src/parameter.rs:98-108`),
"whose arguments are expressions over hyperparameters re-evaluated at each MCMC
step" (`docs/camdl-run-spec.md:5485-5487`). Neither the language spec nor the
run spec states what a held leaf means (verified by searching both for the
hierarchical prior; the IR spec's §6.2 and §15.3 are design sketches). The model
the prior states is p(hyper) · Π_i p(leaf_i | hyper), so a held leaf is a
conditioning value, and the posterior of an estimated hyperparameter must
include each held leaf's density p(leaf | hyper). The Bayesian methods omit it
(§3.2 item 12). Under D8 the defect is being reproduced; if it is confirmed, an
interim change refuses an estimated hyperparameter with any held leaf under PMMH
and PGAS (`HyperOverHeldLeaf`), and the full treatment is a follow-up (§13). A
held or design hyperparameter over estimated leaves is a defined conditional
prior and is accepted. IF2 and NLopt ignore priors, so the case does not arise
there.

**IF2 settings.** `perturb_only_at_t0` and `rw_sd` are `If2ParamSettings` fields
keyed by estimated names. `EstimatedParam` (`sim/src/inference/types.rs:49-74`),
which today carries the box, the start and the IF2 settings in one struct, is
built from the three:
`EstimatedParam::from_parts(name, index, &Estimated, start, Option<&If2ParamSettings>)`.
PMMH reads `rw_sd` for its initial proposal (`fit/pmmh.rs:1046-1052`); until it
has its own tuning (§13), `AlgorithmSettings::Other` fills it with the auto
value.

### 6.5 Starts are built on the existing chain-starts seam

`fit/chain_starts.rs` already separates the rule (`ResolvedStarts`, `:353`), the
context it draws in (`StartContext`, `:492`) and the drawn result
(`DrawnStarts`, `:150`). It is also the largest consumer of the resolver's role
fields: 41 references to `estimate_set`, `ParameterRole` and `FixReason`, and
`build_resolved_view_for_init` (`:505`) builds a `ResolvedParameters` only to
carry the estimated set to `draw_chain_starts` (`:559`, called from
`fit/runner.rs:760` and `profile.rs:968`). After: `StartContext` takes
`&Partition` and the search boxes instead of that view, and `DrawnStarts`
records each start's source (`[estimate].start`, params file, fit estimate, or
drawn with its seed). Stage 2 does this rework; `run_meta.rs:260-300`
(`ParameterProvenance.role`, `kicked_from_estimate`) and `eval.rs:127-136` move
with it.

### 6.6 Where each command calls the resolvers

| Command           | `resolve_roles` (once per job)                                                                                     | value resolution                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `fit run`         | in `resolve_fit_parameters` (`fit/runner.rs:171`), replacing the sweep and scenario checks at `fit/mod.rs:519-583` | `resolve_problem`, same site; `FitRunConfig::build` (`runner.rs:302`) reads it      |
| `survey`          | in `resolve_survey_inputs` (`survey.rs:640`); its two branches become one                                          | `resolve_problem`                                                                   |
| `profile`         | in `cmd_profile`, replacing `profile.rs:317-325`, `:410-422`, `:435-461`, `:750-820`                               | `resolve_problem` at `profile.rs:589`                                               |
| `simulate`        | in the front end once the plan is built (Stage 5; §11 gives the interim)                                           | `resolve_values` per cell, in `util::resolve_run_parameters_typed` (`util.rs:3346`) |
| `fit predict`     | before the sweep check (`fit/predict.rs:1584`) (Stage 5)                                                           | `resolve_values` per arm, `fit/contrasts.rs:647`                                    |
| `pfilter`, `eval` | at `pfilter.rs:210`, `eval.rs:105`                                                                                 | `resolve_values`                                                                    |

## 7. Every run prints the partition at launch and stores it whole

### 7.1 The launch block

For a profile whose fit config lists `beta`, `gamma` and `k` in `[estimate]` and
`N0`, `I0`, `rho` in `[fixed]`, run with `--sweep beta=lin(0.2, 0.8, 13)` on the
`omega` model:

```
roles (6 parameters, 1 typed constant):
  focal      beta     lin(0.2, 0.8, 13)            --sweep (in [estimate] in fit.toml)
  estimated  gamma    box [0.01, 0.5] log   support [0, ∞)   prior log_normal(-2, 0.5) (model)
             k        box [1, 100] log      support [1, 100]  prior half_normal(10) (model)
  fixed      N0 = 10000                            [fixed] in fit.toml
             I0 = 10                               [fixed] in fit.toml
             rho = 0.6                             [fixed] in fit.toml
  constant   omega = 0.01                          model
  starts     uniform, 3 per cell; gamma from [estimate].start (chain 1), k drawn
  method     if2: 2000 particles × 50 iterations, cooling 0.95
             rw_sd gamma = 0.0138 (auto), k = 2.1 (auto)
info: beta is the focal parameter (--sweep); its [estimate] entry in fit.toml is
      not used, and the grid lies inside its search box [0.05, 1]
note: inference is conditional on the fixed values; profile-likelihood intervals
      do not account for uncertainty in them.
```

The note prints whenever a fixed parameter is held; `fit run` and `survey` print
the same block without the focal row, with "fit results" in place of
"profile-likelihood intervals". `simulate` prints the block only under
`--dry-run` (from Stage 5) and prints special-case notices always.

### 7.2 The stored record: the job's partition, and each cell's

`partition` is the authoritative record (D4). A job's sidecar (`fit.meta.json`,
`run_meta.rs:644-696`) stores the job's partition; its `estimated` and `fixed`
fields are generated from it by `PartitionRecord::legacy_fields` and never
written any other way. Each leaf's `run.json` (`inputs.provenance`,
`profile.rs:1142-1155`) stores the **cell's** partition: inside a profile cell
or one fit of `fit run --sweep`, the swept parameter is fixed at the grid value,
and the leaf says so (`Partition::cell`).

A profile leaf:

```json
"partition": {
  "status": "resolved",
  "parameters": {
    "beta":  { "role": "fixed", "job_role": "design", "special": "profile_focal",
               "declared_by": "cli_sweep", "value": 0.4,
               "value_from": { "source": "design_point", "point_id": 4 } },
    "gamma": { "role": "estimated", "declared_by": "fit_estimate",
               "support": [0.0, "inf"], "support_from": "kind",
               "search": [0.01, 0.5], "search_from": "fit_config",
               "transform": "log", "transform_from": "kind",
               "prior": { "log_normal": { "mu": -2.0, "sigma": 0.5 }, "from": "model_ir" } },
    "N0":    { "role": "fixed", "declared_by": "fit_fixed", "value": 10000,
               "value_from": { "source": "fit_toml_fixed" } }
  },
  "constants": { "omega": 0.01 },
  "simplex_groups": [],
  "interventions": { "scenario": null, "enable": [], "disable": [] },
  "starts": { "rule": "uniform", "per_cell": 3,
              "base": { "gamma": { "value": 0.10, "from": "estimate_start" },
                        "k": { "value": 7.31, "from": { "drawn": { "seed": 1 } } } },
              "ignored": [] },
  "algorithm": { "name": "if2", "particles": 2000, "iterations": 50, "cooling": 0.95,
                 "per_param": { "gamma": { "rw_sd": 0.0138, "rw_sd_from": "auto",
                                           "perturb_only_at_t0": false } } }
}
```

In `simulate` (from Stage 5) the cell record carries impositions and design
shadows as design-vs-sample's `summary.impositions` and `summary.design_shadows`
define them. `parameters_provenance` keeps its per-value records for fixed
parameters; estimated parameters leave it, because they have no value (their
starts are under `starts`), which ends §3.2 item 5.

### 7.3 Before and after, command by command

**`profile`.** Before (measured, §3.2 items 3-4): the `omega` command prints
`profile: 1 grid (beta=1) × 1 starts × 1 seeds = 1 IF2 runs` and no roles,
estimates `omega`, and stores `estimated: ["omega"]`, `fixed: {}` and every leaf
role `fixed`. After:

```
camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --estimate gamma \
  --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10
```

resolves `beta` design, `gamma` estimated over its support `[0, ∞)` (no declared
bounds: refused if the starts rule needs a finite box, as the default `uniform`
does — add `--estimate gamma=0.01:0.5`), `N0`, `I0`, `rho`, `k` fixed, `omega` a
constant; prints §7.1's block; stores §7.2's records. The old command is refused
for `--rw-sd auto` (D5); without that flag it has nothing estimated and is
refused unless `--slice` is added.

With a fit config listing `beta`, `gamma`, `k` in `[estimate]`,
`--sweep k=… --rw-sd beta=0.02` before estimates only `beta` (§3.2 item 1);
after, it estimates `beta` (rw_sd 0.02) and `gamma` (auto), `k` is focal with an
info line, and the grid is refused if it leaves `k`'s search box.

**`fit run`.** Before (measured):

```
fit: fit.toml (method=if2)
  model:    sir.camdl
  estimate: beta, gamma
  fixed:    N0, I0, rho, k
  starts:   from_prior (one independent draw per chain) — default: …
```

and `fit.meta.json` labelling `beta`'s start `model_default`. After: §7.1's
block, §7.2's record, no `parameters_provenance` entry for `beta`/`gamma`. Every
valid fit config gives the same partition as today, so results do not change.
`fit run --sweep N0=…` over `[fixed]` runs as today (D1) with an info line, and
each point's leaf records `N0` fixed at its grid value.

**`survey`.** Before (measured): `survey --fit fit.toml --fixed beta=0.3`
samples `beta`. After: refused — `beta` is in `[estimate]` — naming the fix
(move it to `[fixed]` in a copy of the config). Inline:
`survey sir.camdl --data cases.tsv --estimate beta=0.05:1,gamma=0.01:0.5 --fixed …`;
a parameter whose box is infinite is refused.

**`simulate`, `fit predict`, `pfilter`, `eval`.** No change to values or output.
Until Stage 5 they write no `partition` (D7); from Stage 5 the record and the
`--dry-run` block.

## 8. The simplification removes an override system, and the code roughly breaks even

### 8.1 What is not built, or is deleted

- **Precedence clauses.** No layer order, no "a later layer may move a role", no
  "bulk files yield", no "`[estimate]` yields to `--fixed`/`--fixed-file`": one
  declaration per parameter, three named special cases, errors otherwise.
- **The constant-override path** (D2). No `--fixed`, `--param`, `[fixed]` or
  scenario `set` on a typed constant; `ValueSource::ModelDefault`, the
  resolver's tier 1 and its exemption from override records
  (`params_resolver.rs:1438-1443`, `Layers::set`'s `prev != ModelDefault` test)
  go.
- **`--starts from_scenario`** and the scenario-as-starts exception (D6): no
  `Point::FromScenario`, no `StartFrom::Scenario`, no per-scenario exemption in
  the scenario rule.
- **The kick-out and its notice.** `PinnedOverEstimate`, `FixReason`,
  `ParameterRole`, `ResolvedParameters.estimate_set` and the `21d4ed47`
  fixed-over-estimate notice.
- **The starts-as-values tier.** `StartsOrigin`, `StartsValues`,
  `resolve_parameters_with_starts` and tier 1.5.
- **Selector flags.** `--rw-sd` as a selector and `RwSd::Auto` with its per-name
  `auto` entries (D5).
- **Role history.** No displacement history in the record: one optional
  `special` tag.
- **Five refusal families**, replaced by `RoleError`'s two conflict shapes
  (`TwoDeclarations`, `ScenarioOnRole`): `scenario_estimate_collision`,
  `from_scenario_estimate_collision`, `scenario_focal_notices`, the
  focal-vs-fixed check in `profile`, and `fit run`'s sweep-must-be-fixed check.

### 8.2 Counts

Ranges were read at `1a059920`; counts are of those ranges. The added count is
an estimate.

| Removed                                                                                                                                           | Where                                                           | Lines |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----- |
| `profile`'s selection, `--rw-sd` requirement, focal-vs-fixed check, `auto` filter, spec assembly                                                  | `profile.rs:317-325`, `:750-820`                                | ~80   |
| `profile`'s start seeding, starts tier, `[fixed]` scenario check, focal notices                                                                   | `profile.rs:410-422`, `:435-461`, `:550-563` (part), `:610-622` | ~60   |
| kick-out, overlap check, `PinnedOverEstimate`, `ParameterRole`/`FixReason`                                                                        | `params_resolver.rs:255-286`, `:337-377`, `:1663-1717`          | ~130  |
| the starts tier and the model-default tier                                                                                                        | `params_resolver.rs:207-231`, `:1413-1443`, `:1527-1540`        | ~70   |
| scenario × estimate refusals and focal notices (`describe_scenario_actions`, `:1097-1124`, stays: `scenario_fixed_collision` calls it at `:1075`) | `params_resolver.rs:1126-1322`, `:1362-1411`                    | ~245  |
| `fit run`'s sweep check and scenario refusal block                                                                                                | `fit/mod.rs:519-583`                                            | ~65   |
| `survey`'s seeding loops, inline midpoint fall-back, collision block                                                                              | `survey.rs:705-712`, `:740-756`, `:800-826`, `:958-966`         | ~60   |
| `fit run`'s start seeding                                                                                                                         | `fit/runner.rs:223-246`                                         | ~25   |
| `[fixed]`∩`[estimate]` and `from_scenario` checks                                                                                                 | `fit/config_v2.rs:850-865`, `:2899-2903`                        | ~20   |
| `build_resolved_view_for_init` and the role reads                                                                                                 | `fit/chain_starts.rs:505-560` and 41 references                 | ~70   |

About 825 lines removed. Added (estimated): `roles.rs` about 470 (types 170,
`resolve_roles` with the constraint checks 180, launch block and the two
conflict formatters 120), the `chain_starts.rs` rework about 60, call sites
about 90, `PartitionRecord` and the cell projection about 90 — about 710, plus
roughly 550 lines of tests (§11). The production code shrinks by roughly 100
lines; the larger gain is that role logic lives in one module instead of five,
and the number of rules a user must learn falls from about a dozen precedence
and refusal cases to one rule and three cases.

## 9. This composes with design-vs-sample: roles per parameter, axes per cell

`2026-09-27-coordinate-kind-design-vs-sample.md` structures per-cell values of
the grid axes (`ParamPlan { design: DesignAxis, sample: SampleAxis }`). This
proposal structures per-parameter roles. In `simulate` and `fit predict`, where
the axes exist, they compose by two invariants checked in `resolve_roles`: the
design parameters are exactly `DesignAxis.names`; the sampled parameters are
exactly `SampleAxis.varying`, less those §4.2 case 3 gave to `--sweep` or
`--param`. `profile` and `fit run --sweep` build no `DesignAxis` (their grids
are `profile.rs:1015-1027` and `fit/mod.rs:588-604`); there the design
parameters are exactly the sweep's names.

### 9.1 What this subsumes

- **The collision policy keyed on the producing flag** (its §4.4 and §8):
  rekeyed to the role and extended to every command (§4.3). `GuardFindings`
  survives as the partition's special cases plus per-arm scenario impositions.
  Until its Stage 5 makes `--draws-file` assert a measure, a bare `--draws FILE`
  builds `MeasureTag::UndeclaredFile`, whose sampled parameters keep today's
  refusal under a scenario (its Stage 1 interim rule).
- **`SampleProvenance.from_outside` and `Origin`** (its §7.1): the parameters a
  sample does not vary are the partition's fixed parameters, with `ValueSource`
  provenance; `Origin::ModelDefault` has no counterpart, because typed constants
  are not configurable.
- **W12** (`--param` on a swept parameter): a `TwoDeclarations` refusal.

### 9.2 What it leaves untouched

The two axes and their constructors, `point_id` and the `sweep:<param>` columns,
D-B seeds, the tier split 3.5a/3.5b of its §7.4, the output schema (§9), banding
(§10), the draws sidecar (D-D) and duplicate design rows (D-E). The records go
into `run.json` provenance, not into the quantities manifest.

### 9.3 The inference half lands first (D7)

Its Stage 1 (`ParamPlan`) has not landed: `sim_job.rs` still defines
`ParamSource` (verified, `sim_job.rs:142-178`). `fit run`, `survey` and
`profile` resolve with `point_overrides: &[]` (its §6.4, verified), so §11
Stages 1-4 land independently. Stage 5 lands after its Stage 1 and takes the
place of its Stage 2.

### 9.4 Amendments it needs

Recorded here; that proposal is not edited by this one.

1. §6.4: `fit run --sweep`'s names are design parameters; its sweep check
   becomes §4.2 case 2.
2. §7.1: `SampleProvenance.from_outside` and `Origin` are deleted.
3. §7.4: the tier split stands; tiers 2–3 cannot supply a design or sampled
   parameter (§5.2), and the model-default tier is gone.
4. §8: `check_scenario_coordinate_collision` reads the `Partition` and each
   arm's footprint. An imposition by `--param` is also recorded in the partition
   (§4.2 case 3), because it removes the parameter from what the band pools over
   for every arm; an imposition by a scenario stays a per-arm value override.
5. §14 Stage 2 is implemented as §11 Stage 5 below; its tests carry over.
6. §5 W12's refusal text is the shared `TwoDeclarations` wording.

## 10. What changes for users

### 10.1 Refusals and flags

- `profile`: `--rw-sd` is optional, IF2-only, and sets step sizes only;
  `--rw-sd auto` and `NAME=auto` are refused (D5).
- `profile` gains `--estimate NAME[=LO:HI][,…]`, the only selector of estimated
  parameters without a fit config, and `--slice`. With `--fit`, `[estimate]` is
  the estimated set and `--estimate` is refused.
- `survey --estimate` takes the same grammar; `=LO:HI` becomes optional.
- `--fixed` or `--fixed-file` over a fit config's `[estimate]` is refused in
  `profile` and `survey` (today a notice, `21d4ed47`, or silently ignored in
  `survey --fit`).
- A `--fixed-file` entry naming a swept or `--estimate` parameter is refused
  (today it runs with no notice: `profile.rs:750-781` checks only `--fixed` and
  `[fixed]`).
- A scenario setting or scaling an estimated or design parameter is refused in
  every command (today only `fit run` and `survey --fit` refuse the first, and
  `profile` notices the second).
- A parameter whose only value was a `--starts` file is refused under D3.
- Typed constants cannot be named by any role declaration or value flag (D2).
- Grid values outside the search box are refused (§5.4).
- A split simplex is refused; a fully held one is checked to sum to 1.
- A `real` or `instant` parameter with no declared bounds has support `(−∞, ∞)`,
  not `[0, ∞)`.
- Under PMMH and PGAS, an estimated hyperparameter with a held leaf is refused,
  if D8's reproduction confirms the defect.

### 10.2 Existing workflows

- **`profile --scenario baseline --rw-sd auto`.** 10 test files
  (`rg -l '"profile"' rust/crates/cli/tests | xargs rg -l '"baseline"'`:
  `cas_integration.rs` with 36 occurrences, `scenario_baseline_is_ordinary.rs`,
  `profile_pmmh.rs`, `profile_diagnostics.rs`, `profile_indexed_long_obs.rs`,
  `prior_posterior_invariants.rs`, `profile_grid_identity.rs`,
  `profile_greedy_order_invariant.rs`, `profile_priors.rs`,
  `scenario_horizon.rs`) use the scenario as both held values and starting
  points. Each migrates to the existing `--starts from_params=FILE`: the
  scenario's values are written to `baseline.toml`; the held parameters go to
  `--fixed` flags or a `--fixed-file` listing only them; the estimated ones are
  named by `--estimate`; `--starts from_params=baseline.toml` starts them (its
  entries for held and focal names are listed as unused); any intervention the
  scenario enables becomes `--enable`. `profile_pmmh.rs` was removed with
  per-cell PMMH (`2b76f8f9`).
- **Every `profile --rw-sd …`.** 25 occurrences of `"--rw-sd"` in 15 test files
  (`rg -c '"--rw-sd"' rust/crates/cli/tests`) and the `camdl profile` help
  examples (`args/mod.rs:1663-1690`): an explicit list keeps working where its
  names are estimated and gains an `--estimate` with the same names; `auto` is
  dropped. (`Makefile:559` went with `aeca342b`.)
- **`profile --fit F` with `--rw-sd` narrower than `[estimate]`, or `--fixed`
  over `[estimate]`.** Copy the config and move the held parameters to
  `[fixed]`.
- **`profile --starts from_params=FILE` holding parameters.** Those parameters
  need `--fixed` or a `--fixed-file` that names only them.
- **Typed constants.** D9.
- **Docs.** Run spec §1.3 (`docs/camdl-run-spec.md:82-183`) rewritten around
  roles then values; `docs/camdl-language-spec.md:5201-5210` (the `--rw-sd`
  modes) and `:218-220` (typed constants) rewritten by hand, never through
  `dprint`; `docs/inference.md` and `camdl profile --help` gain the
  full-versus-held paragraph; `FIXED_LONG_ABOUT` corrected; the stale doc
  comment on `perturb_only_at_t0` (`fit/config_v2.rs:606-612`) corrected.

### 10.3 Run identity

Every run re-keys at the release that ships this regardless, because the model
digest folds the engine version (`runid/src/inputs.rs:163-168`, `:285-294`,
verified). What matters is that no two runs whose partitions differ share a key,
and that no key moves for a reason that does not change results:

- **`profile`.** Its base level (`ProfileBaseLevel`, `profile.rs:143-149`)
  hashes all parameter values, the fixed names, the fit config's bytes and the
  prior sources; its method level hashes `rw_sd` per estimated name
  (`profile.rs:1206-1219`). `--estimate` bounds would enter neither. The
  partition's canonical form — each role, each fixed value, each search box and
  support, each prior, the simplex groups — is added to the base level beside
  the prior sources, which stay. This re-keys every profile.
- **`survey`.** Its model level hashes the compiled model (`survey.rs:313`),
  which today carries the seeded starts; reproduced by an independent check,
  changing one `[estimate].start` from 0.25 to 0.26 moves the model segment from
  `sir-75a30421` to `sir-ec8867cf`, though a survey never reads the start. That
  is over-keying today. Stage 2 removes the seeding, so every `survey --fit`
  re-keys once, then only for changes that move the landscape.
- **`fit run`.** Its fit level hashes the `Problem` (`fit/cas.rs`); a valid fit
  config's partition is a function of the file, so nothing re-keys beyond the
  engine version.
- **`simulate`, `fit predict`.** Values are unchanged, so nothing re-keys beyond
  the engine version.

No IR change: no `ir/VERSION` bump, no golden moves.

## 11. Five stages, each green alone, with known likelihood profiles as the oracle

Each stage lands on a green `make test`, and each named test is mutation-checked
(revert the source change, keep the test, confirm it fails).

### 11.1 Known profiles

CLI tests check refusals and records; these check that the partition computes
the estimand it claims. Each is a small committed fixture whose exact answer is
computed independently of camdl's likelihood code.

- **K1 — iid normal, location and scale.** A model whose one compartment `X` has
  no transitions (`init { X = mu }`), observed
  `y_t ~ normal(mu = projected,
  sigma = s)` at n = 20 times, with fixed
  synthetic data. The likelihood is that of an iid normal sample. Profiling `mu`
  with `s` estimated must give ℓ_p(μ) = −(n/2)·log(2π·σ̂²(μ)) − n/2, with σ̂²(μ) =
  (1/n)·Σ(y_t − μ)²; holding `s = s₀` must give ℓ(μ; s₀) = −n·log s₀ − (n/2)·log
  2π − Σ(y_t − μ)²/(2s₀²). Run under the ODE backend with `nl-bobyqa`, whose
  likelihood is exact. Tolerance: 1e-6 nats at every grid point (the optimizer's
  tolerance set to 1e-10; inferred sufficient), and the held curve must not
  equal the profiled one except at μ where σ̂(μ) = s₀. The fixture's feasibility
  (a compartment with no transitions observed through `normal`) is inferred and
  is the first thing Stage 1 checks.
- **K2 — a non-identified product.** The same model observed through
  `y_t ~ poisson(rate = rho * projected)` with `X0` and `rho` parameters. The
  likelihood depends on `rho · X0` only. Profiling `rho` with `X0` estimated
  must be flat at the saturated maximum ℓ* = Σ[y_t·log ȳ − ȳ − log y_t!] (ȳ the
  sample mean) wherever the search box of `X0` contains ȳ/ρ, and fall off
  exactly at the box edge; holding `X0` gives the closed-form Poisson curve in
  `rho`. Tolerance 1e-6 nats. This checks that the nuisance parameter is really
  re-maximized per cell and that the search box is part of the estimand.
- **K3 — the exact-marginal particle-filter oracle.** The forward-algorithm
  oracle of `rust/crates/sim/tests/pf_exact_marginal_oracle.rs` (gh#201)
  computes log p(y | θ) exactly for a tiny binomial SIR (N₀ ≈ 6). A
  held-nuisance slice over `beta` with `gamma` fixed is checked pointwise: the
  particle-filter estimate averaged over R replicates must lie within 4 Monte
  Carlo standard errors of the exact value at every grid point, as that oracle
  already tests at single points. The profiled-out cell maximum (IF2 over
  `gamma`) must lie within 4 standard errors plus 0.5 nats of the exact maximum
  over a fine `gamma` grid, and never above it by more than 4 standard errors.

A simplex fixture (three members, one profiled) checks that a split group is
refused and that a fully held grid point not summing to 1 is refused naming the
cell.

### 11.2 Stages

**Stage 1 — the types, computed and recorded, no behaviour change.** Add
`roles.rs`. `fit run`, `survey` and `profile` build the partition from today's
rules (`profile`'s `--rw-sd` selection expressed as declarations) and write the
records. _Tests:_ unit tests of `resolve_roles` for every row of §4.2 and §4.3;
K1 and K2 fixtures compile and their exact answers are committed beside them.

**Stage 2 — `fit run` and `survey` resolve roles first.** Starts move out of the
model (§5.2) and onto the reworked chain-starts seam (§6.5); the launch block;
D1 and D2 for `fit run`; `survey`'s single branch; support, box and prior as
separate types. _Tests:_ `fit run`'s record has no `parameters_provenance` entry
for an estimated parameter; a typed constant runs without a `[fixed]` entry and
is refused in `[estimate]`; two `survey --fit` runs differing only in an
`[estimate].start` share a key; a `real` parameter with no bounds has support
`(−∞, ∞)` and `survey` refuses it; `--fixed` over `[estimate]` is refused.

**Stage 3 — `profile` resolves roles first.** `--estimate`, `--slice`;
`[estimate]` authoritative; `--rw-sd` optional and IF2-only; `auto` removed;
§5.4's checks; §4.3's scenario rules; §10.3's identity change; §10.2's test
migration. _Tests:_ K1, K2 and K3 profiled-out and held; the
`[estimate] beta, gamma, k` + `--sweep k` + `--rw-sd beta=0.02` profile
estimates `beta` and `gamma`; nothing estimated without `--slice` is refused
naming `--estimate`; a grid outside the search box is refused; two profiles
differing only in an `--estimate` bound have different base digests.

**Stage 4 — delete and document.** Everything in §8.2 the earlier stages left
unreferenced, `resolve_parameters` included; §10.2's documents.

During and after Stages 1-4, `simulate`, `fit predict`, `pfilter` and `eval`
compute values through `resolve_values` with an internal all-fixed partition and
keep today's `engine::check_scenario_coordinate_collision` and `ParamSource`.
They write no `partition` field at all until Stage 5 (D7), so no stored record
claims a partition that was not resolved.

**Stage 5 — `simulate`, `fit predict`, `pfilter`, `eval` on the full
partition.** After design-vs-sample Stage 1. The guard reads the partition
(§9.4); they write `partition`. _Tests:_ design-vs-sample Stage 2's tests,
unchanged in substance; `simulate --estimate` is refused naming `simulate`.

## 12. Decisions

**D1 — `--sweep` over a `[fixed]` parameter. Decided (2026-10-26): option (c).**
At job level the swept parameter is a design coordinate; in each cell it is
fixed at the grid value. It is not a role conflict but one of the three cases of
§4.2 (case 2). Rejected: (a) a role change with its own notice class; (b) a
refusal, which breaks every `fit run --sweep` config.

**D2 — Typed constants. Decided (2026-10-26): not configurable.** Only
`parameters {}` entries may be fixed, estimated, swept or sampled from outside.
A typed constant holds its declared value; naming it in any role declaration,
`--fixed`, `--param`, `[fixed]` or a scenario `set` is refused with "move it
into `parameters {}`". Rejected: allowing another fixed value; allowing
estimation. The search for anything requiring an override found two (D9) and
nothing in the language semantics: §8.4 of the language spec describes typed
constants as a dimensional device, and no golden, book model or vignette
overrides one (searched: `ocaml/golden`, `rust/crates/*/tests`, `docs/`,
`camdl-book`, `camdl-vignettes`, `camdl-garki`, `camdl-india-polio`,
`camdl-nigeria-polio`, `ebola-camdl`, `ebola-bdbv-camdl`, `camdl-paper`).

**D3 — Undeclared parameters. Decided (2026-10-26): fixed by default.** An
undeclared parameter with a value from a scenario, a `--fixed-file`/`--params`
key or a sample constant is fixed at that value, and the record names the
source; one with no value is refused, naming the parameter and the ways to
supply one (§5.1). `--slice` guards the profile case (§4.4). The fit config
keeps its completeness check.

**D4 — The stored record. Decided (2026-10-26): `partition` is authoritative.**
The legacy `estimated` and `fixed` fields are generated from it by
`PartitionRecord::legacy_fields` and maintained nowhere else.

**D5 — `--rw-sd auto` and `NAME=auto`. Decided (2026-10-26): refused**, with
"`--rw-sd` no longer selects parameters: declare them with `--estimate NAME,…`
or `[estimate]` in a `--fit` config; steps not given are derived from the
bounds".

**D6 — A scenario as the starts source. Decided (2026-10-26): deferred.**
`--starts from_scenario` is not part of this proposal (§13). The existing
`--starts from_params=FILE` covers the migration (§10.2).

**D7 — Order and interim records. Decided (2026-10-26): the inference half
first.** §11 Stages 1-4 now, Stage 5 after design-vs-sample Stage 1, replacing
its Stage 2. In the interim `simulate` and `fit predict` write no `partition`
field, the simpler of the two permitted forms, so no record claims an unresolved
partition.

**D8 — Estimated hyperparameters over held leaves. Decided (2026-10-26):
reproduce, then refuse.** The intended meaning is the stated model p(hyper)·Π
p(leaf | hyper), under which a held leaf's density belongs in the posterior
(§6.4). The defect is being reproduced separately; if it is a correctness bug, a
targeted refusal lands now as an interim fix, under PMMH and PGAS only, and the
full Bayesian treatment is deferred (§13).

**D9 — Typed-constant entries in existing configs and tests. Open.** Two users
of a typed-constant override exist, and D2 refuses both:

- nine `camdl-garki` fit configs (`fits/mosq_dyn_pool_*.toml` and kin) list
  `abund_floor = 0.000001` in `[fixed]`, its declared value, because `fit run`
  demands it today (gh#965);
- `rust/crates/cli/tests/generated_draws_varying_only_gh949.rs` test 4 sets the
  typed constant `c` (`let c : probability = 0.0`) to 0.9 with `--params` and
  `--param`, to have a parameter `--draws uniform` does not draw.

Options: (a) refuse every naming of a typed constant, with the message naming
the `let`; the Garki configs drop one line each, and the test declares `c` in
`parameters {}` without bounds, which `--draws uniform` also leaves alone
(`main.rs:4581-4587` draws declared bounds only, verified). (b) Accept a
`[fixed]`/`--fixed` entry naming a typed constant at exactly its declared value,
with an info line, and refuse any other value; the Garki configs keep working
unchanged, and the test still migrates. (a) is the rule D2 states without an
exception; (b) costs one equality check and spares the nine configs a no-op
edit. Recommendation: (a), leaning; it is the maintainer's call because it
breaks configs outside this repository.

## 13. Out of scope, each with its reason

- **`--starts from_scenario`** (D6), deferred; `--starts from_params` covers the
  migration.
- **Sweeping an `[estimate]` parameter in `fit run`.** Under IF2 that is
  `profile`; under PGAS or PMMH it is a grid of conditional posteriors, a
  command the maintainer has deferred. It stays refused.
- **Held-leaf densities in the prior sum** (D8): add log p(leaf | hyper) for
  each held leaf of an estimated hyperparameter in PMMH and PGAS, then lift the
  refusal. It touches `pmmh.rs` and `pgas.rs`, high-risk files.
- **A scaled simplex** (free members summing to 1 − Σ held), needed before a
  partly held simplex can run; it changes `if2.rs`.
- **The `rw_sd` scale convention** (natural-scale `rw_sd` converted at the
  current value, `sim/src/inference/types.rs:197-207`, `if2.rs:428`) goes to its
  own issue.
- **PMMH's own proposal tuning** (replacing `rw_sd × 5`).
- **Multi-unit transitions** are separate work and do not touch parameter roles.
- **Warm-starting a cell from its neighbour** would make a cell's identity
  depend on grid position and serialize the grid.
- **A profile interval against a trustworthy global maximum, with Monte Carlo
  error**, is a change to how profile results are summarized.
- **`profile --fit` ignores `ic_free`** (`skip_first_obs_from_loglik: false`,
  `profile.rs:1492`, verified), to be filed as an issue.
