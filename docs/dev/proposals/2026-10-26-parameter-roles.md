---
date: 2026-10-26
status: proposed — the design is complete; D1–D8 (§12) are the maintainer's calls, each with a recommendation
area: CLI parameter resolution / inference setup / run records
issues: none filed yet; §13 lists the follow-ups
related:
  - 2026-09-27-coordinate-kind-design-vs-sample.md (accepted, partly implemented; §9 states what this amends)
  - archive/post-alpha/2026-05-25-cli-init-and-params-ux.md (the value resolver this builds on)
---

# Every parameter has exactly one role, decided before any value

All code references are against `origin/main` at `1a059920`. A claim marked
_verified_ was read at the cited line; _measured_ means the command shown was
run with the `camdl` and `camdlc` built at `1a059920` (the clean build in the
`coincident-drains` worktree, `camdl 0.1.0+1a059920`), on the SIR recovery case
`tests/recovery/cases/sir/` (`β`, `γ`, `N0`, `I0`, `rho`, `k`) with synthetic
weekly cases simulated from its `truth.toml`; _inferred_ means reasoned from the
code and not executed.

## 1. Summary

A camdl run does one of four things with each model parameter. It holds it at
one value (**fixed**), searches over it (**estimated**), sets it per cell from a
grid the user wrote (**design**), or sets it per cell from a draw of a stated
probability measure (**sampled**). That assignment is the run's **partition**:
the statistical question the run answers. A profile likelihood over R₀ with γ
held at 0.1 and ρ re-maximized at every grid point is a different estimand from
one that re-maximizes both; a fit that holds `N0` answers a question conditional
on `N0`.

Today no command computes the partition as an object. Each command decides it
from a different input: `fit run` from `[estimate]`, `survey` from `[estimate]`
or `--estimate`, and `profile` from `--rw-sd`, the IF2 step-size flag. The
decision is interleaved with value resolution — `--fixed` "kicks" a name out of
the estimated set inside the value resolver — and with a set of special-case
refusals that each re-derive part of it. The run records only fragments of it,
and in `profile` records it wrongly (§3).

This proposal adds one typed, exhaustive `Partition`, resolved first, from role
declarations only: `[estimate]`, `--estimate`, `[fixed]`, `--fixed`,
`--fixed-file`, `--sweep`/`--sweep-file` and `--draws`. Values are then resolved
per role through the existing value tiers. Algorithm settings (`rw_sd`,
`perturb_only_at_t0`) are keyed by estimated names and never decide a role. A
scenario supplies values and interventions, never roles. Starting points come
only from `--starts` or `[estimate].start`, with a new
`--starts from_scenario=NAME`. Each command declares the roles it accepts, every
conflict is a rule about roles, and the run prints and stores the whole
partition with every fixed value's source, every domain, the starts and the
algorithm settings.

## 2. Terms

A **parameter** is an entry of the IR's `model.parameters`. Two kinds of model
declaration produce one:

- a `parameters {}` declaration, compiled to `ParamValue::Estimated` (when it
  carries bounds or a prior) or `ParamValue::Required` (when it carries
  neither), never with a value — verified,
  `ocaml/lib/compiler/expander.ml:5711-5727`; an indexed declaration
  `beta[region]` compiles to one scalar parameter per cell, named `beta_north`,
  `beta_south` (`expander.ml:5805-5865`);
- a **model constant**, a typed constant `let` such as
  `let omega : rate = 0.01`, compiled to `ParamValue::Fixed { value }`
  (`expander.ml:5871-5881`). Measured: adding that line to the SIR model and
  compiling gives a parameter `omega` with `{"mode": "fixed", "value": 0.01}`.

The IR's `ParamValue::Estimated` names a parameter that _carries inference
configuration_ (bounds, prior); it is not a role. A parameter declared
`beta : rate in [0.05, 1.0]` is held in `simulate` and estimated in `fit run`.

A **role declaration** is an input that says what a run does with a parameter. A
**value source** is an input that gives a parameter a number. A scenario is a
value source (its `set` and `scale`) plus an **intervention set** (its `enable`
and `disable`); it is never a role declaration. The value tiers are those of
`docs/camdl-run-spec.md` §1.3, implemented in
`params_resolver::resolve_parameters_with_starts` (verified,
`params_resolver.rs:1422-1824`).

A **layer** groups role declarations by where they were written: the model
(model constants), the fit config (`[estimate]`, `[fixed]` and its `from_file` /
`from_scenario`), and the command line (`--estimate`, `--fixed`, `--fixed-file`,
`--sweep`, `--sweep-file`, `--draws`). The command line is the later layer.

The profile-likelihood notation is θ = (ψ, λ, φ): ψ the focal (design)
parameters, λ the nuisance (estimated) parameters, φ the fixed ones. The profile
is ℓ_p(ψ; φ₀) = max over λ ∈ Λ of ℓ(ψ, λ, φ₀), where Λ is the estimated
parameters' search box. Both φ₀ and Λ are part of the estimand: a narrower box
can change the curve.

## 3. What each command does today, and where it goes wrong

### 3.1 Four commands, four selectors

| Command                       | What decides "estimated" (verified)                                                                                                                       | Undeclared parameters                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `fit run`                     | `[estimate]` keys; every model parameter must be in `[estimate]` or `[fixed]` (`fit/config_v2.rs:2906-2914`)                                              | refused                                                     |
| `fit run --sweep`             | as above; a swept name must be in `[fixed]` and not in `[estimate]` (`fit/mod.rs:519-535`)                                                                | refused                                                     |
| `survey --fit`                | `[estimate]` keys (`survey.rs:763`)                                                                                                                       | resolved from the tiers                                     |
| `survey` inline               | `--estimate NAME=LO:HI` (`args/mod.rs:2026`, `survey.rs:1026-1032`)                                                                                       | resolved from the tiers                                     |
| `profile`                     | `--rw-sd`: an explicit list estimates exactly its names; `auto` estimates every parameter not focal and not in `[fixed]`/`--fixed` (`profile.rs:783-797`) | held at whatever the tiers give, a `--starts` file included |
| `simulate`, `pfilter`, `eval` | nothing is estimated (`fit_toml_estimate` empty: `util.rs:3391`, `pfilter.rs:208`, `eval.rs:103`)                                                         | resolved from the tiers                                     |
| `fit predict`                 | the fit's posterior draws carry every parameter (`fit/predict.rs:1604-1636`)                                                                              | refused (every parameter must be a draw column)             |

Inside the resolver, `[estimate]` membership is narrowed by `--fixed` and
`--fixed-file` (the "kick-out", `params_resolver.rs:1675-1717`, verified), and
each removal is reported as a `PinnedOverEstimate` notice
(`params_resolver.rs:353-377`).

### 3.2 Defects that follow from it

Each is a consequence of the role being decided in several places, or not at
all.

1. **`profile` takes the estimated set from an algorithm setting.** Measured: a
   fit config listing `beta`, `gamma` and `k` in `[estimate]`, `--sweep k=…`,
   `--rw-sd beta=0.02` estimates only `beta`; `gamma` is held at its
   `[estimate].start` in every cell, with no notice. `--rw-sd` is also required
   by the NLopt per-cell algorithms, which never read it (`profile.rs:317-320`).
2. **`--rw-sd auto` estimates model constants.** Measured: on the SIR model with
   `let omega : rate = 0.01` added,
   `camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10 --fixed gamma=0.15 --rw-sd auto --starts from_params=b.toml`
   writes `[mle] omega = 0.010000000000000007` and records
   `"estimated": ["omega"]`. The modeller's stated constant became a nuisance
   parameter, because `auto` filters `model.parameters`, which includes model
   constants (`profile.rs:786-791`).
3. **`profile` records the wrong roles.** Same run: the base segment's
   `fit.meta.json` has `"fixed": {}` and `"parameters_provenance": {}`; each
   leaf's `run.json` `inputs.provenance.parameters_provenance` marks all seven
   parameters `"role": "fixed"`, including the focal `beta` and the estimated
   `omega`. The leaf resolves with an empty `fit_toml_estimate` (no `--fit`), so
   every role defaults to `Fixed { NotInEstimate }`
   (`params_resolver.rs:1767-1775`).
4. **A starting point is recorded as a model default.** Measured with `fit run`
   on the SIR fit config (`beta = { …, start = 0.25 }`): `fit.meta.json` records
   `"beta": {"value": 0.25, "source": "model_default", "role": "estimated"}`.
   The start is written into the model before the resolver runs
   (`fit/runner.rs:223-246`), and the resolver labels any pre-existing value
   `ModelDefault` (`params_resolver.rs:1438-1443`). The same pre-seeding loop
   exists in `profile.rs:453-461` and `survey.rs:705-712`, `:805-826`,
   `:960-966`.
5. **`survey --fit` ignores `--fixed`.** Measured:
   `camdl survey sir.camdl --fit fit.toml --fixed beta=0.3 --n-points 4 --eval-particles 20`
   exits 0 and its `landscape.tsv` samples `beta` at 0.226 and 0.545. The fit
   branch passes an empty `fixed_cli` (`survey.rs:785-787`); `a.fixed` is read
   only in the inline branch (`survey.rs:980`). Run spec §1.3 states the
   opposite (`docs/camdl-run-spec.md:170-174`).
6. **`fit run` refuses a model constant it does not need.** Measured:
   `camdl fit run fit_let.toml` on the `omega` model exits with
   `parameters neither estimated nor fixed: omega`. `simulate` runs the same
   model at `omega = 0.01`.
7. **`profile --fit` drops the fit config's simplex groups.** Verified: the
   per-cell `IF2Config` is built with `simplex_groups: vec![]`
   (`profile.rs:1490`), so IF2 perturbs a simplex's members independently and
   nothing keeps them summing to 1.
8. **A swept parameter still needs a value from elsewhere.** Measured: sweeping
   `beta` with no other source fails with
   `parameter 'beta' has no value: no model default, …  Fix: declare a default in the .camdl model, or pin via --fixed beta=<value>`.
   Both fixes are wrong: the language has no `parameters {}` defaults, and
   `--fixed beta` beside `--sweep beta` is refused (`profile.rs:761-781`).
9. **The same scenario means two things.** In `fit run` a scenario setting an
   estimated parameter is refused (`params_resolver.rs:1219-1272`, landed
   `2b872798`); in `profile --rw-sd auto` the same scenario value becomes the
   starting point: the scenario's value is in `base_params`, which becomes each
   estimated parameter's `initial` (`fit/runner.rs:1288`, called from
   `profile.rs:818-820`; verified by reading). Ten test files run
   `profile --scenario baseline` in this second reading (§10.2).

## 4. The partition is decided by declarations, layer by layer

### 4.1 Who declares a role

| Role      | Declared by                                                                                                                                                                        | Its value comes from                                        |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| fixed     | `[fixed]` (inline, `from_file`, `from_scenario`), `--fixed`, `--fixed-file`, `--param`/`--params` (`simulate`, `pfilter`, `eval`), a model constant, or no declaration at all (D3) | the value tiers, scenario included                          |
| estimated | `[estimate]`, `--estimate`                                                                                                                                                         | the method; its start from `--starts` or `[estimate].start` |
| design    | `--sweep`, `--sweep-file`, `fit run --sweep`, `fit predict --sweep`, `profile --sweep`                                                                                             | the design point of each cell                               |
| sampled   | `--draws prior\|uniform\|posterior`, `--draws FILE` (`--draws-file` after design-vs-sample Stage 5), `fit predict`                                                                 | the draw row of each cell                                   |

Nothing else declares a role. In particular `--rw-sd`, `[estimate].rw_sd`,
`perturb_only_at_t0`, `--starts` and a scenario do not.

### 4.2 Two roles in one layer is refused; a later layer may move a role, with a notice

The rule that replaces every special case:

- **Within one layer, a parameter given two roles is refused**, naming both
  declarations. `[estimate]` and `[fixed]` in one fit config; `--fixed beta`
  with `--sweep beta`; `--estimate beta` with `--fixed beta`.
- **A later layer may change the role an earlier layer gave.** Each change is
  printed once as an info notice and recorded (§7). The changes that exist are:
  estimated → design (`--sweep` over `[estimate]`: a profile's focal parameter),
  estimated → fixed (`--fixed` over `[estimate]`), fixed → design (`--sweep`
  over `[fixed]`, D1), fixed → estimated (`--estimate` over `[fixed]`), sampled
  → design (a design value over a varying sampled column — design-vs-sample's
  "design shadow"), sampled → fixed (`--param` over a varying sampled column —
  its "imposition by `--param`"), and any role over a model constant (D2).
- **An entry's settings travel with its role.** When `--sweep k` moves `k` out
  of `[estimate]`, the `[estimate].k` entry's `bounds`, `start`, `rw_sd` and
  `perturb_only_at_t0` are not used, and the notice says so.

This is the reviewer's conflict table expressed as one rule, with one
divergence, D1:

| Input combination                                                            | Result                                                                                                                                    |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `[estimate]` + `[estimate].start`                                            | valid: the start is the start                                                                                                             |
| `[estimate]` + `[fixed]`, one file                                           | refused (same layer; already refused, `6c94825f`)                                                                                         |
| `[estimate]` + `--sweep`                                                     | valid; estimated → design, info notice                                                                                                    |
| `[estimate]` + `--fixed`                                                     | valid; estimated → fixed, info notice (already a notice, `21d4ed47`)                                                                      |
| `[fixed]` + `--sweep`                                                        | valid; fixed → design, info notice (D1; the reviewer's table refuses it)                                                                  |
| `--fixed X` + `--sweep X`                                                    | refused (same layer)                                                                                                                      |
| `[estimate]` + `--starts`                                                    | valid                                                                                                                                     |
| `[estimate]` or `--estimate` + a scenario setting or scaling it              | refused, unless `--starts from_scenario` names that scenario (§5.3)                                                                       |
| design + a scenario setting or scaling it                                    | refused (gh#572), unless `--starts from_scenario` names that scenario in `profile` (§5.3)                                                 |
| sampled + a scenario setting or scaling it                                   | valid: an imposition, recorded (design-vs-sample §6.2)                                                                                    |
| fixed + a scenario setting it                                                | the scenario's value, at tier 4; refused when the same fit config's `[fixed]` gives another value (`scenario_fixed_collision`, unchanged) |
| `rw_sd` or `perturb_only_at_t0` on the command line for a non-estimated name | refused                                                                                                                                   |

### 4.3 Each command accepts a declared set of roles

| Command                   | fixed | estimated    | design                       | sampled                  |
| ------------------------- | ----- | ------------ | ---------------------------- | ------------------------ |
| `simulate`                | yes   | no           | optional                     | optional                 |
| `pfilter`, `eval`         | yes   | no           | no                           | no                       |
| `fit run`, `fit recovery` | yes   | yes          | optional (one fit per point) | no                       |
| `fit predict`             | yes   | no           | optional                     | required (the posterior) |
| `survey`                  | yes   | at least one | no                           | no                       |
| `profile`                 | yes   | zero or more | at least one                 | no                       |

`profile` with nothing estimated is a likelihood slice and is accepted. A
declaration of a role the command does not accept is refused naming the command,
for example `--estimate` on `simulate`. `lineage` reads an event log and takes
no parameter flags (`LineageRealizeArgs`, `args/mod.rs:2250-2288`, verified);
`if2` is a removed-command trap (`if2.rs:16`). Neither changes.

### 4.4 Names are scalar IR names

The partition is keyed by the IR's scalar parameter names, so an indexed
parameter has one role per cell (`beta_north`). A declaration naming the family
`beta` when no scalar `beta` exists is refused as an unknown name with the list
of its cells, as `--sweep` already refuses it (`sweep.rs:65-73`, verified). A
`simulate --param-vec beta=FILE` expands to `beta_<key>` names before roles are
resolved (`util.rs:3360-3362`, verified) and declares each fixed.

## 5. Values are resolved per role

### 5.1 Fixed parameters take the value tiers, unchanged

A fixed parameter's value comes from the run spec §1.3 tiers exactly as today
(`[fixed]` < `--fixed-file`/`--params` < draw constant < scenario < `--fixed`/
`--param`), with the same override records and notices (`b262fde0`). A model
constant's own value is the lowest tier and is labelled `model_constant`
(today's `ValueSource::ModelDefault`, renamed: it is reached only by model
constants, §2). A fixed parameter with no value is refused, naming the flags
that give one; the message no longer suggests a model default.

### 5.2 Estimated, design and sampled parameters take no value from the tiers

An estimated parameter's start comes from the starts rule (§5.3). A design
parameter's value comes from the design point, a sampled parameter's from the
draw row. None of the three needs, or reads, a tier value; a tier value for one
of them is either a role change (§4.2) or a refusal. This removes the four
pre-resolver `[estimate].start` seeding loops (§3.2 item 4) and the "a swept
parameter needs a value" error (item 8). Design values are validated once, at
plan time, against `resolved_bounds` (`params_resolver.rs:141-143`); today
`profile` writes each grid value into the parameter vector per cell without that
check (inferred from `profile.rs:1015-1027` and the per-cell assignment).

### 5.3 Starts come only from `--starts` or `[estimate].start`

The starts rule is the existing `ChainStarts` (`fit/starts.rs:48-85`) with one
new point rule, `--starts from_scenario=NAME`: every chain starts at the values
scenario `NAME` gives (its composed `set` with its `scale` applied, through the
existing `params_resolver::scenario_values`, `params_resolver.rs:1324-1368`,
which refuses a scale with no `set` beneath it).

A starts source supplies values for estimated parameters only. A name it carries
that has another role is not used, and the launch block lists such names once (a
`from_mle` estimate carries every parameter, so refusing them would make the
rule unusable). An estimated parameter the source does not name starts at its
`[estimate].start`, else at a transform-aware draw within its domain
(`fit::init::draw_start_in_bounds`, the gh#34 fall-back, now owned by the starts
resolver instead of seeded into the model).

**The scenario-as-starts exception.** When `--starts from_scenario=NAME` names
the run's own scenario, that scenario's values for estimated parameters are
starts, and in `profile` its values for focal parameters are replaced by the
grid; each is one info line. Without the opt-in, a scenario touching an
estimated or design parameter is refused. This is the reviewer's "unless the
scenario is used explicitly for initialisation", and it is what makes the
`--scenario baseline` profile tests a one-flag migration (§10.2, D7).

### 5.4 The intervention set is resolved separately

The intervention filter already runs apart from value precedence
(`params_resolver.rs:1505-1526`, verified: it calls `apply_scenario_filter`
unconditionally, so toggleable interventions default off). It becomes its own
result, `InterventionSet`, so a scenario's `enable`/`disable` apply whatever
roles its `set`/`scale` keys have. The CLI coupling (`--scenario` conflicts with
`--enable`/`--disable`, `args/mod.rs:147-157`) is unchanged.

## 6. Types

### 6.1 The sketch, checked against the code

The starting sketch was
`enum Role { Fixed, Estimated(Domain), Design(Grid), Sampled(DrawSource) }` with
`ResolvedProblem { partition, fixed, starts, interventions }`. Three parts do
not survive the code:

- **`Design(Grid)` cannot hold a design.** A 2-D profile's grid is a product
  (`profile.rs:1015-1027`), a `--sweep-file` row sets several parameters jointly
  (`sim_job.rs:166-178`, `SweepPoint.overrides`), and design-vs-sample's
  `DesignAxis` already owns the joint points. The role says _that_ a parameter
  is a design coordinate; the axis says _where_ the points are.
- **`Sampled(DrawSource)` has the same problem.** A measure is over the joint
  draw (`SampleAxis.measure`), and its varying set is a property of the axis.
- **`fixed` is per scenario arm, not per run.** `simulate --scenario a,b` runs
  each arm at its own scenario values; only the inference commands have one.

The sketch also lacked simplex groups, the IF2 per-parameter settings and model
constants, and named the estimated parameter's start as part of its domain while
the code reads it through the starts rule (`fit/runner.rs:352-358`).

### 6.2 The partition

```rust
// rust/crates/cli/src/roles.rs (new). The partition is pure data: no values.

/// What a run does with one parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// Held at one value per scenario arm, from the value tiers.
    Fixed,
    /// Searched by the method over `Domain`.
    Estimated(Domain),
    /// Set per cell from the run's design axis.
    Design,
    /// Set per cell from the run's sample axis.
    Sampled,
}

/// An estimated parameter's search domain: part of the estimand.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub lower: f64,
    pub upper: f64,
    pub bounds_from: BoundsFrom,
    pub transform: sim::inference::types::Transform,
    pub transform_from: TransformFrom,
    /// `[estimate].start`, if written. The starts rule decides whether it is read.
    pub start: Option<f64>,
}

/// The fit config may tighten the model's bounds and never loosen them
/// (`fit/runner.rs:1250-1275`); `--estimate NAME=LO:HI` obeys the same rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsFrom { FitConfig, Cli, Model, KindSupport, HalfLine }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformFrom { FitConfig, Kind }

/// What declared a parameter's role.
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    ModelConstant,
    /// Named by nothing: fixed (D3).
    Default,
    FitEstimate,
    FitFixed(FitFixedVia),
    CliEstimate,
    CliFixed,
    CliFixedFile(PathBuf),
    Design(DesignOrigin),          // design-vs-sample §7.1; today built from ParamSource::Sweep
    Sample(MeasureTag),
    /// `fit predict`: the stored fit's own `[fixed]` block.
    FitRecord,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FitFixedVia { Inline, FromFile(PathBuf), FromScenario(String) }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer { Model, FitConfig, Command }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleKind { Fixed, Estimated, Design, Sampled }

#[derive(Debug, Clone, PartialEq)]
pub struct RoleEntry {
    pub role: Role,
    pub declared_by: Declaration,
    /// The earlier-layer roles this one replaced, oldest first (§4.2).
    pub replaced: Vec<(RoleKind, Declaration)>,
}

/// Exhaustive: exactly one entry per `model.parameters` name, in declaration
/// order. Built only by `resolve_roles`, so an incomplete or contradictory
/// partition is unrepresentable outside this module.
#[derive(Debug, Clone)]
pub struct Partition {
    entries: IndexMap<String, RoleEntry>,
    /// Every member of every group is `Estimated` (§6.4).
    simplex_groups: Vec<SimplexGroup>,
}

impl Partition {
    pub fn role(&self, name: &str) -> &Role;
    pub fn estimated(&self) -> impl Iterator<Item = (&str, &Domain)>;
    pub fn names_with(&self, kind: RoleKind) -> impl Iterator<Item = &str>;
    pub fn simplex_groups(&self) -> &[SimplexGroup];
    /// One info notice per role change (§4.2), for the launch block.
    pub fn change_notices(&self) -> Notices;
    /// The stored form (§7.2).
    pub fn record(&self) -> PartitionRecord;
}
```

### 6.3 The resolved problem

```rust
/// Which roles a command accepts (§4.3), one constant per command.
pub struct CommandRoles {
    pub command: &'static str,
    pub estimated: Allowed,
    pub design: Allowed,
    pub sampled: Allowed,
}
#[derive(Clone, Copy)]
pub enum Allowed { No, Optional, AtLeastOne }

pub const SIMULATE: CommandRoles;   pub const PFILTER: CommandRoles;
pub const EVAL: CommandRoles;       pub const FIT_RUN: CommandRoles;
pub const FIT_PREDICT: CommandRoles; pub const SURVEY: CommandRoles;
pub const PROFILE: CommandRoles;

/// The role declarations, by layer. Values are not here.
pub struct RoleInputs<'a> {
    pub model: &'a ir::Model,
    pub fit_estimate: &'a IndexMap<String, EstimateSpecV2>,   // empty without --fit
    pub fit_fixed: &'a [(String, FitFixedVia)],
    pub simplex_groups: &'a [config_v2::SimplexGroup],
    pub cli_estimate: &'a [EstimateDecl],      // --estimate NAME[=LO:HI]
    pub cli_fixed: &'a [String],               // --fixed / --param names
    pub cli_fixed_files: &'a [(PathBuf, Vec<String>)],
    pub design: &'a [(String, DesignOrigin)],
    pub sample: Option<(&'a [String], MeasureTag)>,   // the sample axis's varying set
    /// Each arm's scenario, for the scenario-by-role rule (§4.2).
    pub scenarios: &'a [ScenarioRef],
    pub starts_scenario: Option<&'a str>,      // --starts from_scenario=NAME
}

pub fn resolve_roles(inputs: &RoleInputs<'_>, command: &CommandRoles)
    -> Result<Partition, RoleError>;

#[derive(Debug, Clone)]
pub enum RoleError {
    TwoRolesOneLayer { name: String, layer: Layer, first: Declaration, second: Declaration },
    UnknownName { name: String, by: Declaration, candidates: Vec<String> },
    NotAccepted { name: String, role: RoleKind, command: &'static str },
    Missing { role: RoleKind, command: &'static str },          // profile without --sweep
    ScenarioOnRole { scenario: String, name: String, role: RoleKind,
                     actions: Vec<ScenarioParamAction> },       // params_resolver.rs:797-806
    BoundsLoosened { name: String, given: (f64, f64), model: (f64, f64) },
    SimplexSplit { group: Vec<String>, moved: Vec<(String, RoleKind)> },
    SettingForNonEstimated { name: String, setting: &'static str, role: RoleKind },
}

/// Per-parameter algorithm settings, keyed by estimated names only.
pub struct If2ParamSettings {
    pub rw_sd: f64,
    pub rw_sd_from: RwSdFrom,                   // Cli | FitConfig | Auto
    pub perturb_only_at_t0: bool,
}
#[derive(Clone, Copy)]
pub enum RwSdFrom { Cli, FitConfig, Auto }

pub enum AlgorithmSettings {
    If2 { per_param: IndexMap<String, If2ParamSettings>, /* particles, iterations, cooling */ },
    /// PMMH, PGAS, NLopt, pfilter: no per-parameter settings yet.
    Other { name: &'static str },
}

impl AlgorithmSettings {
    /// Refuses a `--rw-sd NAME=SD` whose `NAME` is not estimated, and any
    /// `--rw-sd` under a method other than IF2. `[estimate]` entries' IF2
    /// settings under another method are listed once at info, not refused:
    /// the fit config was written for its own `[method]`.
    pub fn resolve(partition: &Partition, method: FitAlgorithm,
                   cli_rw_sd: &IndexMap<String, f64>,
                   fit_estimate: &IndexMap<String, EstimateSpecV2>)
        -> Result<Self, RoleError>;
}

/// Where each estimated parameter's start came from.
pub enum StartFrom { EstimateStart, Scenario(String), Params(PathBuf),
                     Mle(String), Drawn { seed: u64 } }

pub struct Starts {
    pub rule: ChainStarts,                            // fit/starts.rs:48
    pub base: IndexMap<String, (f64, StartFrom)>,     // keys = estimated names exactly
    pub ignored: Vec<String>,                         // source names with another role
}

pub struct InterventionSet {
    pub scenario: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
}

/// What an inference command runs: one scenario arm.
pub struct ResolvedProblem {
    pub partition: Partition,
    /// Fixed parameters only: the value resolver's output, with provenance.
    pub fixed: ResolvedParameters,
    pub interventions: InterventionSet,
    pub starts: Starts,
    pub algorithm: AlgorithmSettings,
}

pub fn resolve_problem(partition: Partition, values: ParameterInputs<'_>,
                       starts: &ChainStarts, algorithm: AlgorithmSettings, seed: u64)
    -> Result<ResolvedProblem, ResolveError>;
```

The value resolver keeps its name and tier walk. Its signature gains the
partition and loses the estimate set:

```rust
// params_resolver.rs: before
pub fn resolve_parameters_with_starts(inputs: ParameterInputs<'_>,
    starts: Option<StartsValues<'_>>) -> Result<ResolvedParameters, ResolveError>;
// ParameterInputs { …, fit_toml_estimate: &IndexSet<String>, … }

// after
pub fn resolve_parameters(partition: &Partition, inputs: ParameterInputs<'_>)
    -> Result<ResolvedParameters, ResolveError>;
// ParameterInputs without `fit_toml_estimate`; `point_overrides` splits into
// design and sample overrides when design-vs-sample Stage 1 lands (its §7.4).
```

`ResolvedParameter.role: ParameterRole` and `FixReason`
(`params_resolver.rs:255-275`) are replaced by a reference into the partition;
`ResolvedParameters.estimate_set` and `pinned_over_estimate` go (§8).

### 6.4 Simplex groups, transforms and `perturb_only_at_t0`

A **simplex group** is a set of estimated parameters constrained to sum to 1,
which IF2 perturbs jointly on the log-ratio scale (`fit/config_v2.rs:90-106`,
`fit/runner.rs:629-680`). It is a joint domain, so it lives on the partition.
`resolve_roles` refuses a group whose members are not all estimated after role
changes (`SimplexSplit`): sweeping one member of a three-member simplex leaves
two that cannot sum to 1 on their own. Moving every member out dissolves the
group, recorded with the role changes. `profile` passes the groups to its
per-cell `IF2Config`, fixing §3.2 item 7.

**Transforms and bounds** are resolved into `Domain` by the logic
`build_if2_params_from_specs` has today (`fit/runner.rs:1224-1301`): the fit
config or `--estimate` may tighten the model's bounds, never loosen them; with
neither, the kind's support (`params_resolver.rs:118-143`), then `[0, ∞)`; the
transform from `[estimate].transform`, else the kind. `EstimatedParam`
(`sim/src/inference/types.rs:49-74`), which today carries the domain, the start
and the IF2 settings in one struct, is built from the three:
`EstimatedParam::from_parts(name, index, &Domain, start, Option<&If2ParamSettings>)`.
PMMH reads `rw_sd` for its initial proposal (`fit/pmmh.rs:1046-1052`); until
PMMH has its own tuning (out of scope, §13), the `Other` arm fills it with the
auto value, as today.

**`perturb_only_at_t0`** is an IF2 setting, so it is an `If2ParamSettings` field
keyed by an estimated name. Under another method the loader treats it and
`rw_sd` as inert and says nothing (`fit/config_v2.rs:2403-2407`, verified; the
field's own doc comment at `:606-612` calls it a config-load error, which is
stale and is corrected in Stage 4). After this change the launch block lists
inert IF2 settings once, on its method line, in every command.

### 6.5 Where each command calls the resolvers

| Command           | `resolve_roles` (once per job)                                                                                     | value resolution                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `fit run`         | in `resolve_fit_parameters` (`fit/runner.rs:171`), replacing the sweep and scenario checks at `fit/mod.rs:519-583` | `resolve_problem`, same site; `FitRunConfig::build` (`runner.rs:302`) reads `ResolvedProblem` |
| `survey`          | in `resolve_survey_inputs` (`survey.rs:640`); the two branches become one                                          | `resolve_problem`                                                                             |
| `profile`         | in `cmd_profile`, replacing `profile.rs:317-325`, `:410-422`, `:435-461`, `:750-820`                               | `resolve_problem` at `profile.rs:589`                                                         |
| `simulate`        | in the front end once the plan is built, replacing `engine::check_scenario_coordinate_collision` (`engine.rs:390`) | `resolve_parameters` per cell, in `util::resolve_run_parameters_typed` (`util.rs:3346`)       |
| `fit predict`     | before the sweep check (`fit/predict.rs:1584`); estimated → sampled, the fit's `[fixed]` → fixed (`FitRecord`)     | `resolve_parameters` per arm in `fit/contrasts.rs:647`                                        |
| `pfilter`, `eval` | at `pfilter.rs:210`, `eval.rs:105`                                                                                 | `resolve_parameters`                                                                          |

## 7. What each command prints and stores

### 7.1 The launch block

Every inference command prints the partition before any work. For a profile
whose fit config lists `beta`, `gamma`, `k` and `rho` in `[estimate]` and `N0`,
`I0` in `[fixed]`, run with `--sweep beta=… --fixed rho=0.6` on the `omega`
model:

```
roles (7 parameters):
  focal      beta     lin(0.2, 0.8, 13)            --sweep; [estimate] in fit.toml (moved)
  estimated  gamma    [0.01, 0.5] log              [estimate] in fit.toml
             k        [1, 100] log                 [estimate] in fit.toml
  fixed      N0 = 10000                            [fixed] in fit.toml
             I0 = 10                               [fixed] in fit.toml
             rho = 0.6                             --fixed (replaces [estimate] in fit.toml)
             omega = 0.01                          model constant
  starts     uniform, 3 per cell; gamma from [estimate].start (chain 1), k drawn
  method     if2: 2000 particles × 50 iterations, cooling 0.95
             rw_sd gamma = 0.0138 (auto), k = 2.1 (auto)
info: rho is held at 0.6 by --fixed; the fit config lists it in [estimate]
info: beta is the focal parameter (--sweep); the fit config lists it in [estimate],
      so its bounds, start and rw_sd there are not used
note: inference is conditional on the fixed values; profile-likelihood intervals
      do not account for uncertainty in them.
```

The note prints whenever a fixed parameter other than a model constant is held;
`fit run` and `survey` print the same block without the focal row, with "fit
results" in place of "profile-likelihood intervals". `simulate` prints the block
only under `--dry-run` (an ordinary simulate holds every parameter, and printing
it on every run is noise) and prints role-change notices always.

### 7.2 The stored record

Every inference run writes a `partition` object beside `estimated` and `fixed`
in its `fit.meta.json` sidecar (`run_meta.rs:644-696`) and in each leaf's
`run.json` `inputs.provenance` (`profile.rs:1142-1155`), so a stored profile
says whether it is full or held, at which values, over which box:

```json
"partition": {
  "parameters": {
    "beta":  { "role": "design",    "declared_by": "cli_sweep",
               "replaced": [{ "role": "estimated", "declared_by": "fit_estimate" }] },
    "gamma": { "role": "estimated", "declared_by": "fit_estimate",
               "bounds": [0.01, 0.5], "bounds_from": "fit_config",
               "transform": "log", "transform_from": "kind" },
    "rho":   { "role": "fixed", "declared_by": "cli_fixed", "value": 0.6,
               "value_from": { "source": "fixed_cli" },
               "replaced": [{ "role": "estimated", "declared_by": "fit_estimate" }] },
    "omega": { "role": "fixed", "declared_by": "model_constant", "value": 0.01,
               "value_from": { "source": "model_constant" } }
  },
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

`estimated` and `fixed` stay, written from the partition by one function (D5),
so the four readers of `estimated` (`chain_selection.rs:295-306` among them) are
unchanged. `parameters_provenance` keeps its per-value records for fixed
parameters; estimated parameters leave it, because they have no value (their
starts are under `starts`), which ends §3.2 item 4.

### 7.3 Before and after, command by command

**`profile`.** Before (measured, §3.2 items 2-3):

```
camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 \
  --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10 --fixed gamma=0.15 \
  --rw-sd auto --starts from_params=b.toml
```

prints `profile: 1 grid (beta=1) × 1 starts × 1 seeds = 1 IF2 runs` and no
roles; estimates `omega`; stores `estimated: ["omega"]`, `fixed: {}` and every
leaf role `fixed`. After:

```
camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --estimate gamma \
  --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10
```

resolves β design, γ estimated over `[0.01, 0.5]` log (model bounds), `N0`,
`I0`, `rho`, `k` fixed (`--fixed`), `omega` fixed (model constant); prints the
§7.1 block; stores the §7.2 record. The old command is refused for
`--rw-sd auto` (D6); without that flag it is a valid likelihood slice (nothing
estimated, `gamma` held at 0.15), and its `--starts from_params=b.toml`, which
names only the focal `beta`, is listed as ignored.

With a fit config listing `beta`, `gamma`, `k` in `[estimate]`,
`--sweep k=… --rw-sd beta=0.02` before estimates only `beta` (§3.2 item 1);
after, it estimates `beta` (rw_sd 0.02, from the command line) and `gamma`
(auto), `k` is focal with an info line, and nothing is held that the file did
not hold.

**`fit run`.** Before (measured):

```
fit: fit.toml (method=if2)
  model:    sir.camdl
  estimate: beta, gamma
  fixed:    N0, I0, rho, k
  starts:   from_prior (one independent draw per chain) — default: …
```

and `fit.meta.json` with `estimated`, `fixed` (values) and
`parameters_provenance` labelling `beta`'s start `model_default`. After: the
§7.1 block (fixed values with sources, domains, starts, rw_sd), the §7.2 record,
no `parameters_provenance` entry for `beta`/`gamma`. The partition is the same
for every valid fit config, so results do not change. `fit run --sweep N0=…`
over `[fixed]` prints one info line per swept name (D1); `--sweep beta` over
`[estimate]`, refused today, runs one fit per point with `beta` held there. A
model constant no longer needs a `[fixed]` entry (D2).

**`survey`.** Before (measured): `survey --fit fit.toml --fixed beta=0.3`
samples `beta`. After: `beta` moves estimated → fixed with an info line, the box
is over `gamma` alone, and the record says so. Inline:
`survey sir.camdl --data cases.tsv --estimate beta=0.05:1,gamma --fixed …` takes
`gamma`'s bounds from the model; a parameter with no finite bound anywhere is
refused (LHS needs a box). `--fit` no longer conflicts with `--estimate`: it
adds estimated names as a later layer (D4).

**`simulate`.** No change to values or output for any run that resolves today.
The partition holds every parameter fixed except swept (design) and drawn
(sampled) ones. `--dry-run` prints the block; each leaf's `run.json` carries the
record. `--param beta=0.5 --draws uniform` is recorded as a role change sampled
→ fixed, which is design-vs-sample's imposition by `--param`.

**`fit predict`.** No change to values or output. Estimated parameters are
sampled (`MeasureTag::Posterior`), the fit's `[fixed]` parameters fixed
(`FitRecord`), `--sweep` names design; a sweep over a sampled name is the role
change sampled → design (the design shadow).

**`pfilter`, `eval`.** No change except the record and the refusal of a role
declaration they do not accept.

## 8. What goes away

Line ranges were read at `1a059920`; the counts are of those ranges. The added
count is an estimate.

| Removed                                                                                                                                                                                                     | Where                                                           | Lines |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----- |
| `profile`'s own selection: `--rw-sd` required, focal-vs-fixed check, `exclude`, `auto` filter, spec assembly                                                                                                | `profile.rs:317-325`, `:750-820`                                | ~80   |
| `profile`'s `[estimate].start` seeding, starts-as-values tier, `[fixed]` scenario check, focal notices                                                                                                      | `profile.rs:410-422`, `:435-461`, `:550-563` (part), `:610-622` | ~60   |
| the resolver's kick-out, overlap check, `PinnedOverEstimate`, `ParameterRole`/`FixReason`                                                                                                                   | `params_resolver.rs:255-286`, `:337-377`, `:1663-1717`          | ~130  |
| the starts tier: `StartsOrigin`, `StartsValues`, `resolve_parameters_with_starts`, tier 1.5                                                                                                                 | `params_resolver.rs:207-231`, `:1413-1424`, `:1527-1540`        | ~50   |
| scenario × estimate refusals (`estimate_clashes`, `ClashWords`, `scenario_estimate_collision`, `from_scenario_estimate_collision`) and `scenario_focal_notices`, replaced by one `ScenarioOnRole` formatter | `params_resolver.rs:1097-1322`, `:1362-1411`                    | ~275  |
| `fit run`'s sweep-must-be-fixed check and its scenario refusal block                                                                                                                                        | `fit/mod.rs:519-583`                                            | ~65   |
| `survey`'s two start-seeding loops, its inline midpoint fall-back, its collision block                                                                                                                      | `survey.rs:705-712`, `:740-756`, `:800-826`, `:958-966`         | ~60   |
| `fit run`'s `[estimate].start` seeding                                                                                                                                                                      | `fit/runner.rs:223-246`                                         | ~25   |
| the `[fixed]`∩`[estimate]` and `from_scenario` checks, now partition rules                                                                                                                                  | `fit/config_v2.rs:850-865`, `:2899-2903`                        | ~20   |

About 765 lines removed. Added (estimated): `roles.rs` about 550 (types 170,
`resolve_roles` 220, the launch block and the one conflict formatter 160), call
sites about 90, `PartitionRecord` about 80 — about 720, plus roughly 450 lines
of tests. `--rw-sd` stops being required; `RwSd::Auto` is deleted (D6).

## 9. Relationship to the design-vs-sample proposal

`2026-09-27-coordinate-kind-design-vs-sample.md` structures _per-cell values_ of
the grid axes: `ParamPlan { design: DesignAxis, sample: SampleAxis }`. This
proposal structures _per-parameter roles_. They compose by two invariants, both
checked in `resolve_roles`:

- the parameters with role design are exactly `DesignAxis.names`;
- the parameters with role sampled are exactly `SampleAxis.varying`, less those
  a later layer moved (to design: a design shadow; to fixed: `--param`).

Equality, not inclusion: a design name with another role would have two value
sources per cell.

### 9.1 What this subsumes

- **The collision policy keyed on the producing flag** (its §4.4 and §8). Its §8
  already rekeys the gh#572 guard from `ParamSource` to the axis kind; this
  proposal rekeys it once more, to the role, and extends it past `simulate` and
  `fit predict` to every command: a scenario over a design or estimated
  parameter is refused, over a sampled one is an imposition. `GuardFindings`
  (impositions, design shadows) survives as the partition's scenario findings
  plus its role changes. `UserCoordinate` and its `DrawsFile` arm go, as that §8
  already says. Until its Stage 5 makes `--draws-file` assert a measure, a bare
  `--draws FILE` builds `MeasureTag::UndeclaredFile`, whose sampled parameters
  keep today's refusal under a scenario (its Stage 1 interim rule).
- **`SampleProvenance.from_outside: Vec<(String, Origin)>`** and the `Origin`
  enum (its §7.1). The parameters a sample does not vary are the partition's
  fixed parameters, and their sources are `ValueSource` with override records.
  `Origin { Params, FitFixed, Scenario, Param, ModelDefault }` duplicates
  `ValueSource` (`params_resolver.rs:188-206`), and its `ModelDefault` is
  reachable only by a model constant (§2), not by a `parameters {}` default as
  its name implies.
- **W12** (`--param` on a swept parameter, refused at `simulate`'s front end): a
  same-layer role conflict.

### 9.2 What it leaves untouched

The two axes and their constructors, `point_id` and the `sweep:<param>` columns,
D-B seeds, the tier split of its §7.4 (`ValueSource::{Sample, DesignPoint}`),
the output schema (§9), banding (§10), the draws sidecar (D-D) and duplicate
design rows (D-E). The record of §7.2 goes into `run.json` provenance, not into
the quantities manifest, so its `camdl.quantities/v2` schema is unchanged.

### 9.3 Order: the inference half first

Its Stage 1 (`ParamPlan`) has not landed: `sim_job.rs` still defines
`ParamSource` (verified, `sim_job.rs:142-178`). The inference commands
(`fit
run`, `survey`, `profile`) do not touch `ParamSource` at all — they
resolve with `point_overrides: &[]` (its §6.4, verified) — so §11 Stages 1-4
land first and independently. Stage 5 (`simulate`, `fit predict`) lands after
its Stage 1 and takes the place of its Stage 2: the guard that "reads the plan"
reads the partition instead. Landing this first means the scenario-by-role rule
is written once, for every command, before its Stage 2 writes it for two (D8).

### 9.4 Amendments it would need

Recorded here; that proposal is not edited by this one.

1. §6.4: "`fit run --sweep` … is untouched" becomes: its sweep names are design
   parameters, and the sweep-must-be-fixed check is replaced by the §4.2 rule.
2. §7.1: `SampleProvenance.from_outside` and `Origin` are deleted; the fixed
   parameters' sources come from the partition and `ResolvedParameters`.
3. §8: `check_scenario_coordinate_collision` reads the `Partition` and the
   scenarios' footprints; the "Sources" list gains `--fixed`/`--fixed-file` for
   commands that have them.
4. §14 Stage 2 is implemented as §11 Stage 5 below; its tests carry over.
5. §5 W12's refusal text is the shared `TwoRolesOneLayer` wording.

## 10. What changes for users

### 10.1 Refusals and flags

- `profile`: `--rw-sd` is optional, IF2-only, and sets step sizes only;
  `--rw-sd NAME=SD` naming a non-estimated parameter is refused, and so is any
  `--rw-sd` under another `--algorithm`. `--rw-sd auto` is refused with "omit
  `--rw-sd`; steps not given are derived from the bounds" (D6).
- `profile` gains `--estimate NAME[=LO:HI][,…]`, the only selector of estimated
  parameters without a fit config. With `--fit`, `[estimate]` is the estimated
  set and `--estimate` adds to it (D4).
- `survey --estimate` takes the same grammar; `=LO:HI` becomes optional.
  `survey
  --fit` honours `--fixed` (today ignored) and accepts `--estimate`.
- `--starts from_scenario=NAME` on `fit run`, `profile`.
- A scenario setting or scaling an estimated parameter is refused in `profile`
  and inline `survey` too (today only `fit run` and `survey --fit`), unless
  `--starts from_scenario` names it.
- A scenario setting a profile's focal parameter is refused unless
  `--starts from_scenario` names it; today it is an info notice (`09ccbed1`).
- A parameter whose only value is a `--starts` file is refused as a fixed
  parameter with no value, naming `--fixed-file FILE` (to hold it) and
  `--estimate` (to estimate it).
- A simplex group split by a role change is refused; `profile --fit` honours
  simplex groups.
- `--sweep` over `[estimate]` in `fit run`, refused today, runs (D1).
- A model constant needs no `[fixed]` entry in a fit config (D2).
- `profile` and `survey` no longer accept, silently, a declaration that
  contradicts the fit config; each such case is a refusal or a printed role
  change.

### 10.2 Existing workflows

- **`profile --scenario baseline --rw-sd auto`.** Becomes
  `profile --scenario baseline --starts from_scenario=baseline --estimate <the nuisance names>`.
  The input notes count 28 such tests across 16 files;
  `rg -l '"profile"'
  rust/crates/cli/tests | xargs rg -l '"baseline"'` finds
  10 files (`cas_integration.rs` with 36 occurrences,
  `scenario_baseline_is_ordinary.rs`, `profile_pmmh.rs`,
  `profile_diagnostics.rs`, `profile_indexed_long_obs.rs`,
  `prior_posterior_invariants.rs`, `profile_grid_identity.rs`,
  `profile_greedy_order_invariant.rs`, `profile_priors.rs`,
  `scenario_horizon.rs`); Stage 3 lists every invocation it migrates. The
  `profile_pmmh.rs` cases go with PMMH's removal from `profile` (§13).
- **Every `profile --rw-sd …` invocation.** 25 occurrences of `"--rw-sd"` in 15
  test files (`rg -c '"--rw-sd"' rust/crates/cli/tests`): an explicit list keeps
  working where its names are estimated and gains an `--estimate` with the same
  names; `auto` is dropped.
- **`profile --fit F` with `--rw-sd` narrower than `[estimate]`.** Now estimates
  the whole `[estimate]` set (minus focal and `--fixed`). A user who relied on
  the narrowing writes `--fixed NAME=VALUE` for each parameter to hold, and the
  record shows it.
- **`profile --starts from_params=FILE` holding parameters.** Those parameters
  need `--fixed-file FILE` as well.
- **Docs.** Run spec §1.3 (`docs/camdl-run-spec.md:82-183`) is rewritten around
  roles-then-values; `docs/inference.md` and `camdl profile --help` gain the
  full-versus-held profile paragraph and lose "Random-walk SDs" as the only
  description of `--rw-sd`; `FIXED_LONG_ABOUT` stops saying `--fixed` applies to
  `fit run`.

### 10.3 Run identity

Every run re-keys at the release that ships this regardless, because the model
digest folds the engine version (`runid/src/inputs.rs:163-168`, `:285-294`,
verified). What matters is that no two runs whose partitions differ share a key:

- **`profile`:** its base level (`ProfileBaseLevel`, `profile.rs:143-149`)
  hashes all parameter values, the fixed names, the fit config's bytes and the
  prior sources; the estimated set enters only through the prior list, and
  `--estimate` bounds would not enter at all. It is replaced by the partition's
  canonical form: each parameter's role, each fixed value, each domain (bounds,
  transform), the simplex groups. The method level keeps `rw_sd` per estimated
  name and the starts rule. This re-keys every profile, as a content change.
- **`survey`:** its config level hashes `fixed` and `scenario`, its box level
  the bounds (`survey_cas.rs:64-79`); `fixed` becomes the partition's fixed
  values, which re-keys a survey whose fixed values came from a scenario or
  `--fixed`.
- **`fit run`:** its fit level hashes the `Problem` (`fit/cas.rs`); a valid fit
  config's partition is a function of the file, so nothing re-keys beyond the
  engine version. `--starts from_scenario` is keyed with the rule, as every
  `--starts` already is.
- **`simulate`, `fit predict`:** the params level hashes resolved values
  (`resolve.rs`); values are unchanged, so nothing re-keys beyond the engine
  version.

No IR change: no `ir/VERSION` bump, no golden moves.

## 11. Staged implementation

Each stage lands alone on a green `make test`, and each named test is
mutation-checked (revert the source change, keep the test, confirm it fails).

**Stage 1 — the types, computed and recorded, no behaviour change.** Add
`roles.rs` with `Partition`, `resolve_roles`, `CommandRoles`, `PartitionRecord`.
`fit run`, `survey` and `profile` build the partition from today's rules
(`profile`'s `--rw-sd` selection expressed as declarations) and write the
record. _Tests:_ unit tests of `resolve_roles` for every row of §4.2's table;
the `omega` profile of §3.2 item 2 records `omega` as estimated (today's
behaviour, now visible) and every leaf's role matches the partition.

**Stage 2 — `fit run` and `survey` resolve roles first.** Starts move out of the
model (§5.2); `--fixed` is honoured by `survey --fit`; the launch block; D1 and
D2 for `fit run`. _Tests:_ red-then-green: `survey --fit F --fixed beta=0.3`
holds `beta` (today it samples it); `fit run`'s record has no
`parameters_provenance` entry for an estimated parameter (today
`model_default`); a model constant runs without a `[fixed]` entry;
`fit run
--sweep` over `[estimate]` runs one fit per point.

**Stage 3 — `profile` resolves roles first.** `--estimate`; `[estimate]`
authoritative; `--rw-sd` optional and IF2-only; `auto` removed; simplex groups
honoured; `--starts from_scenario`; the scenario-by-role rule; the identity
change of §10.3; the test migration of §10.2. _Tests:_ red-then-green: the
`[estimate] beta, gamma, k` + `--sweep k` + `--rw-sd beta=0.02` profile
estimates `beta` and `gamma`; `--rw-sd auto` on the `omega` model is refused; a
simplex group reaches the cell's `IF2Config`; two profiles differing only in an
`--estimate` bound have different base digests.

**Stage 4 — delete and document.** Everything in §8 that Stages 2-3 left
unreferenced; run spec §1.3, `docs/inference.md`, the help texts.

**Stage 5 — `simulate`, `fit predict`, `pfilter`, `eval`.** After
design-vs-sample Stage 1. The guard reads the partition (§9.4); impositions and
design shadows are role changes plus scenario findings. _Tests:_
design-vs-sample Stage 2's tests, unchanged in substance; `simulate --estimate`
is refused naming `simulate`.

## 12. Decisions

**D1 — `--sweep` over a `[fixed]` parameter.** The question is whether a later
`--sweep` may move a parameter the fit config holds. The reviewer's table
refuses it. But `fit run --sweep` today _requires_ the swept name to be in
`[fixed]` (`fit/mod.rs:519-535`), so refusing it breaks every `fit run --sweep`
workflow, and it would make `profile --fit F --sweep N0` refuse a parameter the
file holds although `--sweep` over `[estimate]` is accepted. Options: (a) a role
change fixed → design with an info notice, everywhere, which also lets
`fit run --sweep` vary an `[estimate]` name; (b) refuse everywhere and change
`fit run --sweep` to require the name in neither block, breaking its configs;
(c) refuse in `profile` only. (a) gives one rule — two roles in one layer is
refused, a later layer moves — that covers every row of the table. The cost is a
notice on every `fit run --sweep`. **Recommendation: (a). Leaning.**

**D2 — Model constants in the partition.** A typed constant `let` is an IR
parameter with a value (§2). Today `fit run` refuses a fit config that does not
list it, and `profile --rw-sd auto` estimates it (both measured). Options: (a) a
model constant is fixed at its declared value unless a role declaration names
it, and needs no `[fixed]` entry; (b) keep requiring it in the fit config; (c)
forbid role declarations on model constants. (a) matches `simulate`, where the
constant simply holds; (c) would stop a modeller from profiling over a constant
without editing the model. **Recommendation: (a). Solid.**

**D3 — Undeclared parameters are fixed.** In a command that accepts fixed
parameters, a parameter no declaration names gets role fixed and must get a
value from the tiers. Options: (a) default fixed, printed in the launch block as
`declared_by: default`; (b) require a declaration for every parameter in the
inference commands, as `fit run`'s config check does. (a) is what `simulate`,
`pfilter` and `profile` do today and lets `--scenario` hold the rest of a
profile's parameters; (b) would make an inline `profile` list every parameter
twice (as `--fixed` and in a file). The fit config keeps its own completeness
check, because a fit config is the self-describing record of a fit. The
reviewer's "the optimizer never infers the partition" holds under (a): the
default is a stated rule, printed and stored. **Recommendation: (a). Leaning.**

**D4 — `--estimate` with `--fit`.** Options: (a) accepted, a later layer adding
estimated names (fixed → estimated is a printed role change), taking an existing
`[estimate]` entry's domain or the model's; (b) refused with `--fit` (as
`survey --fit` refuses it today, `args/mod.rs:2020`). (a) lets a user release
one `[fixed]` parameter without editing the file, symmetric with `--fixed` over
`[estimate]`. **Recommendation: (a). Leaning.**

**D5 — The stored record.** Options: (a) add `partition`, keep `estimated` and
`fixed` as projections written by one function; (b) replace them, and have
readers of older fits refuse. (b) is tidier but would make every stored fit
unreadable by `fit predict` and `fit summary`; (a) costs two derived fields.
**Recommendation: (a). Leaning.**

**D6 — `--rw-sd auto`.** It meant "select every unfixed parameter and size their
steps from the bounds". Once `--rw-sd` is optional, omitting it means auto
steps. Options: (a) refuse `auto` with the one-line replacement (alpha posture:
no shims); (b) accept it as a no-op. **Recommendation: (a). Solid.**

**D7 — The scenario-as-starts exception.** Options: (a) a scenario that
`--starts
from_scenario` also names may set estimated (and, in `profile`, focal)
parameters, whose values become starts (and are replaced by the grid); (b) no
exception: such a scenario is always refused, and a user wanting its
interventions passes `--enable`/`--disable` and its fixed values through a
`--fixed-file`. (b) is stricter but leaves the `--scenario baseline` tests and
workflows with nothing to run until they restructure; (a) is the reviewer's own
exception and makes the migration one flag, and the opt-in is visible on the
command line and in the record. **Recommendation: (a). Leaning.**

**D8 — Land the inference half before design-vs-sample Stage 2.** Options: (a)
§11 Stages 1-4 now, Stage 5 after design-vs-sample Stage 1, replacing its Stage
2; (b) design-vs-sample Stages 1-2 first, then this. (b) would implement the
scenario guard twice, once keyed on the axis and again on the role. Neither
proposal's earlier stages block the other's. **Recommendation: (a). Solid.**

## 13. Out of scope

- **PMMH in `profile`** is being removed separately; a cell's highest-posterior
  sample is not a profile maximum. Stage 3 does not migrate `profile_pmmh.rs`.
- **The `rw_sd` scale convention** (natural-scale `rw_sd` converted at the
  current value, `sim/src/inference/types.rs:197-207`, `if2.rs:428`) goes to its
  own issue.
- **Multi-unit transitions** are separate work and do not touch parameter roles.
- **PMMH's own proposal tuning** (replacing `rw_sd × 5`) belongs to PMMH.
- **Warm-starting a cell from its neighbour** would make a cell's identity
  depend on grid position and serialize the grid; not proposed.
- **A profile interval against a trustworthy global maximum, with Monte Carlo
  error**, is a separate change to how profile results are summarized.
- **`profile --fit` ignores `ic_free`** (`skip_first_obs_from_loglik: false`,
  `profile.rs:1492`, verified): an estimand setting the fit config states and
  `profile` drops, to be filed as an issue.
