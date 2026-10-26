---
date: 2026-10-26
status: proposed — the design is complete; D1–D7 (§12) are the maintainer's calls, each with a recommendation
area: CLI parameter resolution / inference setup / run records
issues: gh#965 (typed constants, open); §13 lists the follow-ups
related:
  - 2026-09-27-coordinate-kind-design-vs-sample.md (accepted, partly implemented; §9 states what this amends)
  - archive/post-alpha/2026-05-25-cli-init-and-params-ux.md (the value resolver this builds on)
---

# Every parameter has exactly one role, decided before any value

All code references are against `origin/main` at `1a059920`. A claim marked
_verified_ was read at the cited line; _measured_ means the command shown was
run with the `camdl` and `camdlc` built at `1a059920` (`camdl 0.1.0+1a059920`),
on the SIR recovery case `tests/recovery/cases/sir/` (`beta`, `gamma`, `N0`,
`I0`, `rho`, `k`) with synthetic weekly cases simulated from its `truth.toml`;
_inferred_ means reasoned from the code and not executed.

## 1. One partition, resolved first, replaces four selectors

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

This proposal adds one typed, exhaustive `Partition`, resolved first from role
declarations only: `[estimate]`, `--estimate`, `[fixed]`, `--fixed`,
`--fixed-file`, `--sweep`/`--sweep-file` and `--draws`. Values are then resolved
per role through the existing value tiers. Algorithm settings (`rw_sd`,
`perturb_only_at_t0`) are keyed by estimated names and never decide a role. A
scenario supplies values and interventions, never roles. Starting points come
only from `--starts` or `[estimate].start`, with a new
`--starts from_scenario=NAME`. Each command declares the roles it accepts, every
conflict is a rule about roles, and the run prints and stores the partition with
every fixed value's source, every domain and prior, the starts and the algorithm
settings.

## 2. The terms: parameters, roles, declarations and value tiers

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
  (`expander.ml:5871-5881`). Measured: adding that line to the SIR model gives a
  parameter `omega` with `{"mode": "fixed", "value": 0.01}`.

A model constant is not equivalent to a `parameters {}` entry in the compiler,
although the language spec says it is "indistinguishable, once compiled"
(`docs/camdl-language-spec.md:218-220`): another `let` that refers to it stops
the compiler with E512, a compiler-invariant error, where the same `let` over a
`parameters {}` entry compiles. That and `fit run`'s refusal of an unlisted
constant are gh#965, being fixed separately. Stage 4 corrects the spec sentence.

The IR's `ParamValue::Estimated` names a parameter that _carries inference
configuration_ (bounds, prior); it is not a role. `beta : rate in [0.05, 1.0]`
is held in `simulate` and estimated in `fit run`.

A **role declaration** is an input that says what a run does with a parameter. A
**value source** is an input that gives a parameter a number. A scenario is a
value source (its `set` and `scale`) plus an **intervention set** (its `enable`
and `disable`); it is never a role declaration.

A declaration is **named** when the user wrote the parameter's name in it: an
`[estimate]` or `[fixed]` key, a `[fixed] from_file` or `--fixed-file` key, a
`[fixed] from_scenario` import of a name the scenario sets, `--estimate`,
`--fixed`, `--param`, `--params` keys, a `--sweep` name, a `--sweep-file`
column. It is **implicit** when nobody wrote the name: a measure's coverage (the
**varying set** of `--draws prior|uniform|posterior` or a fit's posterior — the
parameters the measure actually varies), a model constant, or no declaration at
all.

The **value tiers** order the sources of a value, later winning. They are those
of `docs/camdl-run-spec.md` §1.3, implemented in
`params_resolver::resolve_parameters_with_starts` (verified,
`params_resolver.rs:1422-1824`), with the split design-vs-sample §7.4 makes:

| Tier | Source                                                                |
| ---- | --------------------------------------------------------------------- |
| 1    | a model constant's declared value                                     |
| 2    | the fit config's `[fixed]`                                            |
| 3    | `--fixed-file` (inference commands) / `--params` (`simulate` and kin) |
| 3.5a | the cell's sample row                                                 |
| 3.5b | the cell's design point                                               |
| 4    | the scenario's `set` and `scale`                                      |
| 5    | `--fixed` / `--param` / `--param-vec`                                 |

Today's tier 1.5 (a `--starts` file's values) is removed (§5.3).

Two terms come from design-vs-sample §2 and §6.2. An **imposition** is a value a
scenario or `--param` puts on a parameter the sample varies: every draw then
runs at that value while its other parameters keep their sampled values. A
**design shadow** is a design value on a parameter the sample varies: the design
value wins in each cell.

The profile-likelihood notation is θ = (ψ, λ, φ): ψ the focal (design)
parameters, λ the nuisance (estimated) parameters, φ the fixed ones. The profile
is ℓ_p(ψ; φ₀) = max over λ ∈ Λ of ℓ(ψ, λ, φ₀), where Λ is the estimated
parameters' search box. Both φ₀ and Λ are part of the estimand.

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
proposal; the design here does not depend on them staying broken, and each
interim fix is superseded by the partition.

1. **`profile` takes the estimated set from an algorithm setting.** Measured: a
   fit config listing `beta`, `gamma` and `k` in `[estimate]`, `--sweep k=…`,
   `--rw-sd beta=0.02` estimates only `beta`; `gamma` is held at its
   `[estimate].start` in every cell, with no notice. `--rw-sd` is also required
   by the NLopt per-cell algorithms, which never read it (`profile.rs:317-320`).
2. **`--rw-sd auto` estimates parameters the user fixed in a file** (interim).
   It excludes `--fixed` and `[fixed]` names only (`profile.rs:750-755`), not
   `--fixed-file` keys; reproduced by an independent check at `1a059920`,
   `gamma`, `rho` and `k` given by a `--fixed-file` came out estimated.
3. **`--rw-sd auto` estimates model constants** (interim). Measured: on the SIR
   model with `let omega : rate = 0.01` added,
   `camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10 --fixed gamma=0.15 --rw-sd auto --starts from_params=b.toml`
   writes `[mle] omega = 0.010000000000000007` and records
   `"estimated": ["omega"]`: `auto` filters `model.parameters`, which includes
   model constants (`profile.rs:786-791`).
4. **`profile` records the wrong roles and drops simplex groups** (interim).
   Same run: the base segment's `fit.meta.json` has `"fixed": {}` and
   `"parameters_provenance": {}`; each leaf's `run.json` marks all seven
   parameters `"role": "fixed"`, including the focal `beta` and the estimated
   `omega` (`params_resolver.rs:1767-1775`: with no `--fit`, every role defaults
   to `Fixed { NotInEstimate }`). The per-cell `IF2Config` is built with
   `simplex_groups: vec![]` (`profile.rs:1490`, verified), so IF2 perturbs a
   simplex's members independently.
5. **A starting point is recorded as a model default** (interim). Measured with
   `fit run` on the SIR fit config (`beta = { …, start = 0.25 }`):
   `fit.meta.json` records
   `"beta": {"value": 0.25, "source": "model_default", "role": "estimated"}`,
   and under the default `starts = "from_prior"` 0.25 is not even the start
   used. The start is written into the model before the resolver runs
   (`fit/runner.rs:223-246`), and the resolver labels any pre-existing value
   `ModelDefault` (`params_resolver.rs:1438-1443`). The same pre-seeding loop is
   in `profile.rs:453-461` and `survey.rs:705-712`, `:805-826`, `:960-966`.
6. **`survey --fit` ignores `--fixed`** (interim). Measured:
   `camdl survey sir.camdl --fit fit.toml --fixed beta=0.3 --n-points 4 --eval-particles 20`
   exits 0 and its `landscape.tsv` samples `beta` at 0.226 and 0.545. The fit
   branch passes an empty `fixed_cli` (`survey.rs:785-787`); `a.fixed` is read
   only inline (`survey.rs:980`). Run spec §1.3 says the opposite
   (`docs/camdl-run-spec.md:170-174`).
7. **`fit run` refuses a model constant it does not need** (gh#965, interim).
   Measured: `camdl fit run fit_let.toml` on the `omega` model exits with
   `parameters neither estimated nor fixed: omega`.
8. **PMMH silently ignores the IF2 settings** (interim). Reproduced by an
   independent check at `1a059920`: `perturb_only_at_t0` and `rw_sd` in
   `[estimate]` change nothing under PMMH, and the loader says nothing
   (`fit/config_v2.rs:2403-2407`, verified; the field's doc comment at
   `:606-612` calls it a config-load error, which is stale).
9. **A grid value outside the bounds runs** (interim). Reproduced by an
   independent check at `1a059920`: `profile --sweep beta=2.0` runs with
   `beta in [0.05, 1.0]`, while a start of 2.0 is refused.
10. **A swept parameter still needs a value from elsewhere.** Measured: sweeping
    `beta` with no other source fails with
    `parameter 'beta' has no value: no model default, …  Fix: declare a default in the .camdl model, or pin via --fixed beta=<value>`.
    Both fixes are wrong: `parameters {}` entries have no defaults, and
    `--fixed beta` beside `--sweep beta` is refused (`profile.rs:761-781`).
11. **The same scenario means two things.** In `fit run` a scenario setting an
    estimated parameter is refused (`params_resolver.rs:1219-1272`, `2b872798`);
    in `profile --rw-sd auto` its value becomes the starting point: it is in
    `base_params`, which becomes each estimated parameter's `initial`
    (`fit/runner.rs:1288`, called from `profile.rs:818-820`; verified by
    reading). Ten test files run `profile --scenario baseline` this way (§10.2).

## 4. Named declarations decide the partition; implicit coverage yields to them

### 4.1 Who declares a role

| Role      | Declared by                                                                                                                                                          | Its value in a cell comes from                              |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| fixed     | `[fixed]` (inline, `from_file`, `from_scenario`), `--fixed`, `--fixed-file`, `--param`/`--params` (`simulate`, `pfilter`, `eval`), a model constant, or nothing (D3) | tiers 1–5                                                   |
| estimated | `[estimate]`, `--estimate`                                                                                                                                           | the method; its start from `--starts` or `[estimate].start` |
| design    | `--sweep`, `--sweep-file`, `fit run --sweep`, `fit predict --sweep`, `profile --sweep`                                                                               | tier 3.5b, the design point                                 |
| sampled   | the varying set of `--draws prior\|uniform\|posterior`, `--draws FILE` (`--draws-file` after design-vs-sample Stage 5), `fit predict`'s posterior                    | tier 3.5a, the draw row                                     |

Nothing else declares a role. `--rw-sd`, `[estimate].rw_sd`,
`perturb_only_at_t0`, `--starts` and a scenario do not.

### 4.2 The rule

Roles are decided once per job by three clauses.

1. **An implicit role yields to a named declaration.** A parameter that is
   sampled only because a measure covers it, fixed only because it is a model
   constant, or fixed only because nothing names it, takes the role a named
   declaration gives it. Over a sampled parameter this is design-vs-sample's
   design shadow (`--sweep`) or imposition by `--param` (`--param`, `--fixed`),
   recorded as such (§7.2); over a model constant or the default it is silent.
2. **The fit config's `[estimate]` yields to the command line's `--sweep` and
   `--fixed`, with an info notice.** A fit config is reused across `fit run`,
   `survey` and `profile`; holding or profiling one of its estimated parameters
   is the command narrowing the estimand, not a contradiction. The `[estimate]`
   entry's `bounds`, `start`, `prior`, `rw_sd` and `perturb_only_at_t0` are then
   not used, and the notice says so. This applies in `profile` and `survey`;
   `fit run --sweep` naming an `[estimate]` parameter stays refused, as today
   (§13).
3. **Any other two named declarations giving one parameter different roles are
   refused**, naming both: `[estimate]` and `[fixed]` in one file;
   `--fixed beta` with `--sweep beta`; `--estimate beta` with `--fixed beta`.
   `--estimate` and `--fit` are mutually exclusive, as in `survey` today
   (`args/mod.rs:2020`): with a fit config, `[estimate]` alone declares the
   estimated set.

A `--sweep` over a `[fixed]` parameter is D1. Scenarios never enter this rule;
they are a value source, checked by role in §4.3.

The resulting conflicts:

| Input combination                                                            | Result                                                                         |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `[estimate]` + `[estimate].start`                                            | valid: the start is the start                                                  |
| `[estimate]` + `[fixed]`, one file                                           | refused (clause 3; already refused, `6c94825f`)                                |
| `[estimate]` + `--sweep` (`profile`)                                         | valid; estimated → design, info notice (clause 2)                              |
| `[estimate]` + `--fixed`                                                     | valid; estimated → fixed, info notice (clause 2; already a notice, `21d4ed47`) |
| `[fixed]` + `--sweep`                                                        | D1                                                                             |
| `--fixed X` + `--sweep X`                                                    | refused (clause 3)                                                             |
| `--draws uniform` + `--sweep beta`                                           | valid; design shadow (clause 1)                                                |
| `--draws uniform` + `--param beta=0.5`                                       | valid; imposition by `--param` (clause 1)                                      |
| `[estimate]` + `--starts`                                                    | valid                                                                          |
| `rw_sd` or `perturb_only_at_t0` on the command line for a non-estimated name | refused                                                                        |

### 4.3 A scenario is checked against the role of each parameter it touches

| Role of the parameter the scenario sets or scales | Result                                                                                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| fixed                                             | the scenario's value, at tier 4; refused when the same fit config's `[fixed]` gives another value (`scenario_fixed_collision`, unchanged) |
| estimated                                         | refused, unless `--starts from_scenario` names that scenario (§5.3)                                                                       |
| design                                            | refused (gh#572), unless, in `profile`, `--starts from_scenario` names that scenario (§5.3)                                               |
| sampled                                           | valid: an imposition by the scenario, at tier 4 over tier 3.5a, recorded per arm                                                          |

A scenario is per arm (`simulate --scenario a,b`), so it changes no role: an
imposition by a scenario is a value override inside the sampled role, and the
partition is the same for every arm. This is why clause 1 covers `--param` and
`--sweep` but not scenarios.

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
parameter held. It is refused unless the user says so with `--slice`, because
the easiest way to reach it is by accident — dropping `--rw-sd auto` without
adding `--estimate` would otherwise turn a full profile into a slice, every
nuisance parameter fixed by D3, with nothing said. The refusal names
`--estimate NAME,…`, `[estimate]` in a `--fit` config, and `--slice`.

A declaration of a role the command does not accept is refused naming the
command, for example `--estimate` on `simulate`. `lineage` reads an event log
and takes no parameter flags (`args/mod.rs:2250-2288`, verified); `if2` is a
removed-command trap (`if2.rs:16`). Neither changes.

### 4.5 Names are scalar IR names

The partition is keyed by the IR's scalar parameter names, so an indexed
parameter has one role per cell (`beta_north`). A declaration naming the family
`beta` when no scalar `beta` exists is refused as an unknown name. Today
`--sweep` lists only the unknown name (`sweep.rs:65-73`, verified); the new
message also lists the family's cells. `simulate --param-vec beta=FILE` expands
to `beta_<key>` names before roles are resolved (`util.rs:3360-3362`, verified).

## 5. Values are resolved per role

### 5.1 A fixed parameter takes tiers 1–5, unchanged

A fixed parameter's value comes from the tiers of §2 exactly as today, with the
same override records and notices (`b262fde0`). Tier 1 is labelled
`model_constant` (today's `ValueSource::ModelDefault`, renamed: only a model
constant reaches it). A fixed parameter with no value is refused, naming the
flags that give one; the message no longer suggests a model default.

### 5.2 Design and sampled parameters take their own tier; estimated take none

A design parameter's value in a cell is tier 3.5b and a sampled one's tier 3.5a.
Tiers 1–3 cannot supply them: the job-level resolution needs no value for them,
and a `[fixed]`, `--fixed-file` or `--starts` value for one is either a role
change under §4.2 or a refusal. Later tiers apply as §4.3 and §4.2 say: over a
sampled parameter, tier 4 and tier 5 are impositions; over a design parameter
both are refused.

An estimated parameter takes no tier value. Its start comes from the starts rule
(§5.3); the slot the compiled model needs is filled from that start. This
removes the four pre-resolver seeding loops (§3.2 item 5) and the "a swept
parameter needs a value" error (item 10).

Design values are validated once, at plan time, against `resolved_bounds`
(`params_resolver.rs:141-143`) for every command (§3.2 item 9).

### 5.3 Starts come only from `--starts` or `[estimate].start`

The starts rule is the existing `ChainStarts` (`fit/starts.rs:48-85`) with one
new point rule, `--starts from_scenario=NAME`: every chain starts at the values
scenario `NAME` gives (its composed `set` with its `scale` applied, through
`params_resolver::scenario_values`, `params_resolver.rs:1324-1368`, which
refuses a scale with no `set` beneath it).

A starts source supplies values for estimated parameters only. A name it carries
that has another role is not used, and the launch block lists such names once (a
`from_mle` estimate carries every parameter, so refusing them would make the
rule unusable). An estimated parameter the source does not name starts at its
`[estimate].start`, else at a transform-aware draw within its domain
(`fit::init::draw_start_in_bounds`, the gh#34 fall-back, now owned by the starts
resolver).

**A scenario named as the starts source may also set estimated and focal
parameters.** When `--starts from_scenario=NAME` names the run's own scenario,
that scenario's values for estimated parameters are starts, and in `profile` its
values for focal parameters are replaced by the grid; each is one info line.
Without the opt-in, §4.3 refuses both. This is what makes the
`--scenario baseline` profile tests a one-flag migration (§10.2, D6).

### 5.4 The intervention set is resolved separately

The intervention filter already runs apart from value precedence
(`params_resolver.rs:1505-1526`, verified: it calls `apply_scenario_filter`
unconditionally, so toggleable interventions default off). It becomes its own
result, `InterventionSet`, so a scenario's `enable`/`disable` apply whatever
roles its `set`/`scale` keys have. `--scenario` still conflicts with
`--enable`/`--disable` (`args/mod.rs:147-157`).

## 6. The types: a partition, its domains, and the problem built on it

### 6.1 What the partition does and does not hold

A role says _that_ a parameter is a design coordinate or a sample column; the
axes say _where_ the points are. A 2-D profile's grid is a product
(`profile.rs:1015-1027`), a `--sweep-file` row sets several parameters jointly
(`sim_job.rs:166-178`), and a measure is over the joint draw, so design-vs-
sample's `DesignAxis` and `SampleAxis` own the points and the partition carries
no grid and no measure. Fixed values are per scenario arm, so they are not in
the partition either.

### 6.2 The partition

```rust
// rust/crates/cli/src/roles.rs (new). The partition holds no values.

/// What a run does with one parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    /// One value per scenario arm, from tiers 1–5.
    Fixed,
    /// Searched by the method over `Domain`.
    Estimated(Domain),
    /// Set per cell from the design axis (tier 3.5b).
    Design,
    /// Set per cell from the sample axis (tier 3.5a).
    Sampled,
}

/// An estimated parameter's search domain and prior: part of the estimand.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub lower: f64,                       // may be -∞
    pub upper: f64,                       // may be +∞
    pub bounds_from: BoundsFrom,
    pub transform: sim::inference::types::Transform,
    pub transform_from: TransformFrom,
    /// The resolved prior and its source (`fit::priors_precedence`): read by
    /// the Bayesian methods, ignored by IF2 and NLopt, always recorded.
    pub prior: ResolvedPrior,
    /// `[estimate].start`, if written. The starts rule decides whether it is read.
    pub start: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsFrom { FitConfig, Cli, Model, Kind }

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

#[derive(Debug, Clone, PartialEq)]
pub struct RoleEntry {
    pub role: Role,
    pub declared_by: Declaration,
    /// The implicit or `[estimate]` role this one displaced (§4.2 clauses 1
    /// and 2). At most one: no other move exists.
    pub displaced: Option<(RoleKind, Declaration)>,
}

/// Exhaustive: exactly one entry per `model.parameters` name, in declaration
/// order. Built only by `resolve_roles`, so an incomplete or contradictory
/// partition cannot be constructed outside this module.
#[derive(Debug, Clone)]
pub struct Partition {
    entries: IndexMap<String, RoleEntry>,
    simplex_groups: Vec<ResolvedSimplex>,   // §6.4
}

impl Partition {
    pub fn role(&self, name: &str) -> &Role;
    pub fn estimated(&self) -> impl Iterator<Item = (&str, &Domain)>;
    pub fn names_with(&self, kind: RoleKind) -> impl Iterator<Item = &str>;
    pub fn simplex_groups(&self) -> &[ResolvedSimplex];
    /// One info notice per displaced role.
    pub fn change_notices(&self) -> Notices;
    /// The partition one cell runs: every design and sampled parameter fixed at
    /// that cell's value, tagged with its tier (§7.2).
    pub fn cell(&self, design: &DesignPointRef, sample: &SampleRowRef) -> CellPartition;
    pub fn record(&self) -> PartitionRecord;
    /// Every parameter fixed: the interim for commands not yet on roles (§11).
    pub fn all_fixed(model: &ir::Model) -> Partition;
}
```

### 6.3 The resolved problem and the resolver signatures

```rust
/// Which roles a command accepts (§4.4), one constant per command.
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
    pub scenarios: &'a [ScenarioRef],          // each arm, for §4.3
    pub starts_scenario: Option<&'a str>,      // --starts from_scenario=NAME
}

pub fn resolve_roles(inputs: &RoleInputs<'_>, command: &CommandRoles)
    -> Result<Partition, RoleError>;

#[derive(Debug, Clone)]
pub enum RoleError {
    TwoNamedRoles { name: String, first: Declaration, second: Declaration },
    UnknownName { name: String, by: Declaration, candidates: Vec<String> },
    NotAccepted { name: String, role: RoleKind, command: &'static str },
    Missing { role: RoleKind, command: &'static str },     // profile: no --sweep, or nothing estimated without --slice
    ScenarioOnRole { scenario: String, name: String, role: RoleKind,
                     actions: Vec<ScenarioParamAction> },   // params_resolver.rs:797-806
    BoundsLoosened { name: String, given: (f64, f64), model: (f64, f64) },
    UnboundedForMethod { name: String, method: &'static str },
    SimplexPartlyHeld { group: Vec<String>, held: Vec<String> },
    SimplexSum { group: Vec<String>, cell: Option<u64>, sum: f64 },
    SettingForNonEstimated { name: String, setting: &'static str, role: RoleKind },
}

/// Per-parameter algorithm settings, keyed by estimated names only.
pub struct If2ParamSettings {
    pub rw_sd: f64,
    pub rw_sd_from: RwSdFrom,
    pub perturb_only_at_t0: bool,
}
#[derive(Clone, Copy)]
pub enum RwSdFrom { Cli, FitConfig, Auto }

pub enum AlgorithmSettings {
    If2 { per_param: IndexMap<String, If2ParamSettings>, /* particles, iterations, cooling */ },
    /// PMMH, PGAS, NLopt, pfilter: no per-parameter settings.
    Other { name: &'static str, inert: Vec<(String, &'static str)> },
}

impl AlgorithmSettings {
    /// Refuses `--rw-sd NAME=SD` for a non-estimated `NAME`, and any `--rw-sd`
    /// under a method other than IF2. `[estimate]` entries' IF2 settings under
    /// another method are kept in `inert` and listed once in the launch block.
    pub fn resolve(partition: &Partition, method: FitAlgorithm,
                   cli_rw_sd: &IndexMap<String, f64>,
                   fit_estimate: &IndexMap<String, EstimateSpecV2>)
        -> Result<Self, RoleError>;
}

pub struct InterventionSet {
    pub scenario: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
}

/// What an inference command runs.
pub struct ResolvedProblem {
    pub partition: Partition,
    /// Fixed parameters only: the value resolver's output, with provenance.
    pub fixed: ResolvedParameters,
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
(`params_resolver.rs:255-275`) are replaced by a reference into the partition;
`ResolvedParameters.estimate_set` and `pinned_over_estimate` go.

### 6.4 Domains, simplex groups, hierarchical priors and IF2 settings

**Bounds and transforms** are resolved into `Domain` by the logic
`build_if2_params_from_specs` has today (`fit/runner.rs:1224-1301`): the fit
config or `--estimate` may tighten the model's bounds, never loosen them; the
transform comes from `[estimate].transform`, else the kind. With no bounds
anywhere, the range is the kind's: `probability` `[0, 1]`; `rate`, `positive`,
`count`, `duration` `[0, ∞)`; `real`, `instant` `(−∞, ∞)` — the classification
`kind_support`'s doc gives (`params_resolver.rs:108-117`), with
`BoundsFrom::Kind`. Today the fallback is `[0, ∞)` for every kind
(`fit/runner.rs:1130`, `:1273`, verified), which gives a `real` or `instant`
parameter a box that excludes negative values; that is a wrong estimand the
partition corrects. A method that needs a finite box — `survey`'s Latin
hypercube, the `uniform` and `lhs` starts rules — refuses an infinite domain
naming the parameter (`UnboundedForMethod`).

**Priors** are part of the estimand for the Bayesian methods, so they live in
`Domain` and in the stored record, resolved by the existing
`fit::priors_precedence` chain (fit config, then the model's `~`, then flat).

**A simplex group** is a set of estimated parameters constrained to sum to 1,
which IF2 perturbs jointly through a log-ratio transform whose softmax inverse
fixes the sum at 1 (`sim/src/inference/if2.rs:58-67`,
`fit/config_v2.rs:90-106`). After the partition:

- every member estimated: the group runs as today, and `profile` now passes it
  to its per-cell `IF2Config`;
- every member fixed or design: the group dissolves into a check, and
  `resolve_roles` refuses a cell whose member values do not sum to 1 within 1e-9
  (`SimplexSum`), naming the cell;
- some members fixed or design and some estimated: a conditional simplex whose
  free members sum to 1 − Σ(held). That is a legitimate model, but the runtime
  cannot express it — `SimplexGroup` has no total (`if2.rs:62-67`) and the
  softmax always returns 1 — so it is refused (`SimplexPartlyHeld`) until a
  scaled simplex is added (§13).

**Hierarchical priors.** A leaf's hierarchical prior refers to its
hyperparameters as ordinary parameters (`HierarchicalPrior.args` are expressions
over `Param` names, `ir/src/parameter.rs:98-108`). Every combination is a
defined model and none is refused: a fixed or design hyperparameter makes its
estimated leaves' prior conditional on that value (the expression is evaluated
at it); an estimated hyperparameter over fixed leaves is estimated from those
leaves' prior terms. The launch block prints each hierarchical family's members
with their roles, so a fixed hyperparameter is visible.

**IF2 settings.** `perturb_only_at_t0` and `rw_sd` are `If2ParamSettings` fields
keyed by estimated names. `EstimatedParam` (`sim/src/inference/types.rs:49-74`),
which today carries the domain, the start and the IF2 settings in one struct, is
built from the three:
`EstimatedParam::from_parts(name, index, &Domain, start, Option<&If2ParamSettings>)`.
PMMH reads `rw_sd` for its initial proposal (`fit/pmmh.rs:1046-1052`); until it
has its own tuning (§13), `AlgorithmSettings::Other` fills it with the auto
value, as today, and lists the `[estimate]` IF2 settings as inert.

### 6.5 Starts are built on the existing chain-starts seam

`fit/chain_starts.rs` already separates the rule (`ResolvedStarts`, `:353`), the
context it draws in (`StartContext`, `:492`) and the drawn result
(`DrawnStarts`, `:150`). It is also the largest consumer of the resolver's role
fields: 41 references to `estimate_set`, `ParameterRole` and `FixReason`, and
`build_resolved_view_for_init` (`:505`) builds a `ResolvedParameters` only to
carry the estimated set to `draw_chain_starts` (`:559`, called from
`fit/runner.rs:760` and `profile.rs:968`). After: `StartContext` takes
`&Partition` and the domains instead of that view, `resolve_starts` gains
`Point::FromScenario`, and `DrawnStarts` records each start's source
(`[estimate].start`, scenario, params file, fit estimate, or drawn with its
seed). Stage 2 does this rework; `run_meta.rs:260-300`
(`ParameterProvenance.role`, `kicked_from_estimate`) and `eval.rs:127-136` (a
debug assertion on `ParameterRole::Fixed`) move with it.

### 6.6 Where each command calls the resolvers

| Command           | `resolve_roles` (once per job)                                                                                     | value resolution                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `fit run`         | in `resolve_fit_parameters` (`fit/runner.rs:171`), replacing the sweep and scenario checks at `fit/mod.rs:519-583` | `resolve_problem`, same site; `FitRunConfig::build` (`runner.rs:302`) reads it      |
| `survey`          | in `resolve_survey_inputs` (`survey.rs:640`); its two branches become one                                          | `resolve_problem`                                                                   |
| `profile`         | in `cmd_profile`, replacing `profile.rs:317-325`, `:410-422`, `:435-461`, `:750-820`                               | `resolve_problem` at `profile.rs:589`                                               |
| `simulate`        | in the front end once the plan is built (Stage 5; until then §11 says how it runs)                                 | `resolve_values` per cell, in `util::resolve_run_parameters_typed` (`util.rs:3346`) |
| `fit predict`     | before the sweep check (`fit/predict.rs:1584`); estimated → sampled, the fit's `[fixed]` → fixed (`FitRecord`)     | `resolve_values` per arm, `fit/contrasts.rs:647`                                    |
| `pfilter`, `eval` | at `pfilter.rs:210`, `eval.rs:105`                                                                                 | `resolve_values`                                                                    |

## 7. Every run prints the partition at launch and stores it whole

### 7.1 The launch block

For a profile whose fit config lists `beta`, `gamma`, `k` and `rho` in
`[estimate]` and `N0`, `I0` in `[fixed]`, run with
`--sweep beta=… --fixed rho=0.6` on the `omega` model:

```
roles (7 parameters):
  focal      beta     lin(0.2, 0.8, 13)            --sweep (in [estimate] in fit.toml)
  estimated  gamma    [0.01, 0.5] log  log_normal(-2, 0.5) (model)   [estimate] in fit.toml
             k        [1, 100] log     half_normal(10) (model)       [estimate] in fit.toml
  fixed      N0 = 10000                            [fixed] in fit.toml
             I0 = 10                               [fixed] in fit.toml
             rho = 0.6                             --fixed (in [estimate] in fit.toml)
             omega = 0.01                          model constant
  starts     uniform, 3 per cell; gamma from [estimate].start (chain 1), k drawn
  method     if2: 2000 particles × 50 iterations, cooling 0.95
             rw_sd gamma = 0.0138 (auto), k = 2.1 (auto)
info: rho is held at 0.6 by --fixed; the fit config lists it in [estimate]
info: beta is the focal parameter (--sweep); the fit config lists it in [estimate],
      so its bounds, start, prior and rw_sd there are not used
note: inference is conditional on the fixed values; profile-likelihood intervals
      do not account for uncertainty in them.
```

The note prints whenever a fixed parameter other than a model constant is held;
`fit run` and `survey` print the same block without the focal row, with "fit
results" in place of "profile-likelihood intervals". `simulate` prints the block
only under `--dry-run` and prints displacement notices always.

### 7.2 The stored record: the job's partition, and each cell's

A job's sidecar (`fit.meta.json`, `run_meta.rs:644-696`) stores the job's
partition. Each leaf's `run.json` (`inputs.provenance`, `profile.rs:1142-1155`)
stores **the cell's**: inside a profile cell or one fit of `fit run --sweep`,
the swept parameter is held at the grid value, and the leaf must say so, or a
reader such as `fit predict` cannot tell at what value the cell's fit holds it.
`Partition::cell` produces it: each design parameter becomes fixed with
`"value_from": "design_point"`, each sampled one fixed with `"sample_row"`.

A profile leaf:

```json
"partition": {
  "parameters": {
    "beta":  { "role": "fixed", "declared_by": "cli_sweep", "value": 0.4,
               "value_from": { "source": "design_point", "point_id": 4 },
               "job_role": "design",
               "displaced": { "role": "estimated", "declared_by": "fit_estimate" } },
    "gamma": { "role": "estimated", "declared_by": "fit_estimate",
               "bounds": [0.01, 0.5], "bounds_from": "fit_config",
               "transform": "log", "transform_from": "kind",
               "prior": { "log_normal": { "mu": -2.0, "sigma": 0.5 }, "from": "model_ir" } },
    "rho":   { "role": "fixed", "declared_by": "cli_fixed", "value": 0.6,
               "value_from": { "source": "fixed_cli" },
               "displaced": { "role": "estimated", "declared_by": "fit_estimate" } },
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

In `simulate` the cell record carries impositions and design shadows as
design-vs-sample's `summary.impositions` and `summary.design_shadows` define
them; the job record lists the partition once.

`estimated` and `fixed` stay in the sidecar, written from the partition by one
function (D4), so their readers (`chain_selection.rs:295-306` among them) are
unchanged. `parameters_provenance` keeps its per-value records for fixed
parameters; estimated parameters leave it, because they have no value (their
starts are under `starts`), which ends §3.2 item 5.

### 7.3 Before and after, command by command

**`profile`.** Before (measured, §3.2 items 3-4): the `omega` command prints
`profile: 1 grid (beta=1) × 1 starts × 1 seeds = 1 IF2 runs` and no roles;
estimates `omega`; stores `estimated: ["omega"]`, `fixed: {}` and every leaf
role `fixed`. After:

```
camdl profile sir_let.camdl --data cases.tsv --sweep beta=0.4 --estimate gamma \
  --fixed N0=10000 --fixed I0=10 --fixed rho=0.6 --fixed k=10
```

resolves `beta` design, `gamma` estimated over `[0.01, 0.5]` log (model bounds),
`N0`, `I0`, `rho`, `k` fixed (`--fixed`), `omega` fixed (model constant); prints
§7.1's block; stores §7.2's records. The old command is refused for
`--rw-sd auto` (D5). Without that flag it has nothing estimated and is refused
unless `--slice` is added, which then lists `--starts from_params=b.toml`'s
`beta` (focal) as ignored.

With a fit config listing `beta`, `gamma`, `k` in `[estimate]`,
`--sweep k=… --rw-sd beta=0.02` before estimates only `beta` (§3.2 item 1);
after, it estimates `beta` (rw_sd 0.02, command line) and `gamma` (auto), `k` is
focal with an info line, and nothing is held that the file did not hold.

**`fit run`.** Before (measured):

```
fit: fit.toml (method=if2)
  model:    sir.camdl
  estimate: beta, gamma
  fixed:    N0, I0, rho, k
  starts:   from_prior (one independent draw per chain) — default: …
```

and `fit.meta.json` with `estimated`, `fixed` (values) and a
`parameters_provenance` labelling `beta`'s start `model_default`. After: §7.1's
block, §7.2's record, no `parameters_provenance` entry for `beta`/`gamma`. Every
valid fit config gives the same partition as today, so results do not change.
`fit run --sweep N0=…` over `[fixed]` follows D1; each sweep point's leaf
records `N0` fixed at its grid value. A model constant needs no `[fixed]` entry
(D2).

**`survey`.** Before (measured): `survey --fit fit.toml --fixed beta=0.3`
samples `beta`. After: `beta` moves estimated → fixed with an info line, the box
is over `gamma` alone, and the record says so. Inline:
`survey sir.camdl --data cases.tsv --estimate beta=0.05:1,gamma --fixed …` takes
`gamma`'s bounds from the model; a parameter whose domain is infinite is
refused.

**`simulate`.** No change to values or output for any run that resolves today.
Every parameter is fixed except swept (design) and drawn (sampled) ones.
`--dry-run` prints the block; each leaf's `run.json` carries the cell record.

**`fit predict`.** No change to values or output. Estimated parameters are
sampled (`MeasureTag::Posterior`), the fit's `[fixed]` parameters fixed
(`FitRecord`), `--sweep` names design; a sweep over a sampled name is a design
shadow.

**`pfilter`, `eval`.** No change except the record and the refusal of a role
declaration they do not accept.

## 8. About 750 lines go, and about as many come back

Ranges were read at `1a059920`; counts are of those ranges. The added count is
an estimate. The change does not shrink the code: it moves role logic from five
places into one module and adds the record, the launch block and refusals that
do not exist today.

| Removed                                                                                                                                                                                                                                                                                                                               | Where                                                           | Lines |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----- |
| `profile`'s selection: `--rw-sd` required, focal-vs-fixed check, `exclude`, `auto` filter, spec assembly                                                                                                                                                                                                                              | `profile.rs:317-325`, `:750-820`                                | ~80   |
| `profile`'s `[estimate].start` seeding, starts-as-values tier, `[fixed]` scenario check, focal notices                                                                                                                                                                                                                                | `profile.rs:410-422`, `:435-461`, `:550-563` (part), `:610-622` | ~60   |
| the resolver's kick-out, overlap check, `PinnedOverEstimate`, `ParameterRole`/`FixReason`                                                                                                                                                                                                                                             | `params_resolver.rs:255-286`, `:337-377`, `:1663-1717`          | ~130  |
| the starts tier: `StartsOrigin`, `StartsValues`, `resolve_parameters_with_starts`, tier 1.5                                                                                                                                                                                                                                           | `params_resolver.rs:207-231`, `:1413-1424`, `:1527-1540`        | ~50   |
| scenario × estimate refusals (`EstimateClash`, `estimate_clashes`, `ClashWords`, `scenario_estimate_collision`, `from_scenario_estimate_collision`) and `scenario_focal_notices`, replaced by one `ScenarioOnRole` formatter; `describe_scenario_actions` (`:1097-1124`) stays, because `scenario_fixed_collision` calls it (`:1075`) | `params_resolver.rs:1126-1322`, `:1362-1411`                    | ~245  |
| `fit run`'s sweep-must-be-fixed check and its scenario refusal block                                                                                                                                                                                                                                                                  | `fit/mod.rs:519-583`                                            | ~65   |
| `survey`'s start-seeding loops, its inline midpoint fall-back, its collision block                                                                                                                                                                                                                                                    | `survey.rs:705-712`, `:740-756`, `:800-826`, `:958-966`         | ~60   |
| `fit run`'s `[estimate].start` seeding                                                                                                                                                                                                                                                                                                | `fit/runner.rs:223-246`                                         | ~25   |
| the `[fixed]`∩`[estimate]` and `from_scenario` checks, now partition rules                                                                                                                                                                                                                                                            | `fit/config_v2.rs:850-865`, `:2899-2903`                        | ~20   |
| `build_resolved_view_for_init` and the role reads in `chain_starts.rs`                                                                                                                                                                                                                                                                | `fit/chain_starts.rs:505-560` and 41 references                 | ~70   |

About 805 lines removed. Added (estimated): `roles.rs` about 600 (types 180,
`resolve_roles` 240, launch block and the conflict formatter 180), the
`chain_starts.rs` rework about 60, call sites about 90, `PartitionRecord` and
the cell projection about 100 — about 850, plus roughly 450 lines of tests.
`RwSd` loses its `Auto` variant and its per-name `auto` entries (D5).

## 9. This composes with design-vs-sample: roles per parameter, axes per cell

`2026-09-27-coordinate-kind-design-vs-sample.md` structures per-cell values of
the grid axes (`ParamPlan { design: DesignAxis, sample: SampleAxis }`). This
proposal structures per-parameter roles. In `simulate` and `fit predict`, where
the axes exist, they compose by two invariants checked in `resolve_roles`:

- the design parameters are exactly `DesignAxis.names`;
- the sampled parameters are exactly `SampleAxis.varying`, less those clause 1
  of §4.2 gave to a named declaration (design shadows and impositions by
  `--param`).

`profile` and `fit run --sweep` build no `DesignAxis` (their grids are
`profile.rs:1015-1027` and `fit/mod.rs:588-604`); there the design parameters
are exactly the sweep's names.

### 9.1 What this subsumes

- **The collision policy keyed on the producing flag** (its §4.4 and §8). Its §8
  rekeys the gh#572 guard from `ParamSource` to the axis kind; this proposal
  rekeys it to the role and extends it to every command (§4.3). `GuardFindings`
  (impositions, design shadows) survives as the partition's displacements plus
  the per-arm scenario impositions. `UserCoordinate` and its `DrawsFile` arm go,
  as its §8 already says. Until its Stage 5 makes `--draws-file` assert a
  measure, a bare `--draws FILE` builds `MeasureTag::UndeclaredFile`, whose
  sampled parameters keep today's refusal under a scenario (its Stage 1 interim
  rule).
- **`SampleProvenance.from_outside: Vec<(String, Origin)>` and `Origin`** (its
  §7.1). The parameters a sample does not vary are the partition's fixed
  parameters, and their sources are `ValueSource` with override records.
  `Origin { Params, FitFixed, Scenario, Param, ModelDefault }` duplicates
  `ValueSource` (`params_resolver.rs:188-206`), and its `ModelDefault` is
  reached only by a model constant.
- **W12** (`--param` on a swept parameter): a refusal under clause 3.

### 9.2 What it leaves untouched

The two axes and their constructors, `point_id` and the `sweep:<param>` columns,
D-B seeds, the tier split 3.5a/3.5b of its §7.4 (kept here as §2's tiers), the
output schema (§9), banding (§10), the draws sidecar (D-D) and duplicate design
rows (D-E). The records of §7.2 go into `run.json` provenance, not into the
quantities manifest, so `camdl.quantities/v2` is unchanged.

### 9.3 The inference half lands first

Its Stage 1 (`ParamPlan`) has not landed: `sim_job.rs` still defines
`ParamSource` (verified, `sim_job.rs:142-178`). `fit run`, `survey` and
`profile` resolve with `point_overrides: &[]` (its §6.4, verified), so §11
Stages 1-4 land independently of it. Stage 5 (`simulate`, `fit predict`) lands
after its Stage 1 and takes the place of its Stage 2 (D7).

### 9.4 Amendments it would need

Recorded here; that proposal is not edited by this one.

1. §6.4: "`fit run --sweep` … is untouched" becomes: its sweep names are design
   parameters; the sweep-must-be-fixed check becomes D1's rule.
2. §7.1: `SampleProvenance.from_outside` and `Origin` are deleted; fixed
   parameters' sources come from the partition and `ResolvedParameters`.
3. §7.4: the tier split stands; tiers 1–3 cannot supply a design or sampled
   parameter (§5.2), and the earlier tiers are numbered as §2 numbers them.
4. §8: `check_scenario_coordinate_collision` reads the `Partition` and each
   arm's footprint. An imposition by `--param` is also a displacement in the
   partition (sampled → fixed under clause 1), because it removes the parameter
   from what the band pools over for every arm; an imposition by a scenario
   stays a per-arm value override with no role change. `Imposition { by }` is
   kept for both.
5. §14 Stage 2 is implemented as §11 Stage 5 below; its tests carry over.
6. §5 W12's refusal text is the shared `TwoNamedRoles` wording.

## 10. What changes for users

### 10.1 Refusals and flags

- `profile`: `--rw-sd` is optional, IF2-only, and sets step sizes only;
  `--rw-sd NAME=SD` naming a non-estimated parameter is refused, and so is any
  `--rw-sd` under another `--algorithm`. `--rw-sd auto` and a per-name
  `NAME=auto` entry are refused (D5).
- `profile` gains `--estimate NAME[=LO:HI][,…]`, the only selector of estimated
  parameters without a fit config, and `--slice` for a profile with nothing
  estimated. With `--fit`, `[estimate]` is the estimated set and `--estimate` is
  refused.
- `survey --estimate` takes the same grammar; `=LO:HI` becomes optional.
  `survey
  --fit` honours `--fixed`.
- `--starts from_scenario=NAME` on `fit run` and `profile`.
- A scenario setting or scaling an estimated parameter is refused in `profile`
  and inline `survey` too, unless `--starts from_scenario` names it.
- A scenario setting a profile's focal parameter is refused unless
  `--starts from_scenario` names it; today it is an info notice (`09ccbed1`).
- A parameter whose only value is a `--starts` file is refused as a fixed
  parameter with no value, naming `--fixed-file FILE` and `--estimate`.
- A simplex group split between estimated and held members is refused; a fully
  held one is checked to sum to 1.
- A `real` or `instant` parameter with no declared bounds is searched over
  `(−∞, ∞)`, not `[0, ∞)`; a method needing a finite box refuses it.
- A model constant needs no `[fixed]` entry and cannot be estimated (D2).

### 10.2 Existing workflows

- **`profile --scenario baseline --rw-sd auto`.** Becomes
  `profile --scenario baseline --starts from_scenario=baseline --estimate <the nuisance names>`.
  `rg -l '"profile"' rust/crates/cli/tests | xargs rg -l '"baseline"'` finds 10
  files (`cas_integration.rs` with 36 occurrences,
  `scenario_baseline_is_ordinary.rs`, `profile_pmmh.rs`,
  `profile_diagnostics.rs`, `profile_indexed_long_obs.rs`,
  `prior_posterior_invariants.rs`, `profile_grid_identity.rs`,
  `profile_greedy_order_invariant.rs`, `profile_priors.rs`,
  `scenario_horizon.rs`); Stage 3 lists every invocation it migrates. The
  `profile_pmmh.rs` cases go with PMMH's removal (§13).
- **Every `profile --rw-sd …`.** 25 occurrences of `"--rw-sd"` in 15 test files
  (`rg -c '"--rw-sd"' rust/crates/cli/tests`), `Makefile:559` (a profiling
  target) and the `camdl profile` help examples (`args/mod.rs:1663-1690`): an
  explicit list keeps working where its names are estimated and gains an
  `--estimate` with the same names; `auto` is dropped.
- **`profile --fit F` with `--rw-sd` narrower than `[estimate]`.** Now estimates
  the whole `[estimate]` set, minus focal and `--fixed` names. A user who relied
  on the narrowing writes `--fixed NAME=VALUE` for each parameter to hold.
- **`profile --starts from_params=FILE` holding parameters.** Those parameters
  need `--fixed-file FILE` as well.
- **Docs.** Run spec §1.3 (`docs/camdl-run-spec.md:82-183`) is rewritten around
  roles then values; `docs/camdl-language-spec.md:5201-5210` ("the list IS the
  partition", the three `--rw-sd` modes) and `:218-220` (model constants) are
  rewritten by hand, never through `dprint`; `docs/inference.md` and
  `camdl profile --help` gain the full-versus-held paragraph; `FIXED_LONG_ABOUT`
  stops saying `--fixed` applies to `fit run`; the stale doc comment on
  `perturb_only_at_t0` (`fit/config_v2.rs:606-612`) is corrected.

### 10.3 Run identity

Every run re-keys at the release that ships this regardless, because the model
digest folds the engine version (`runid/src/inputs.rs:163-168`, `:285-294`,
verified). What matters is that no two runs whose partitions differ share a key,
and that no key moves for a reason that does not change results:

- **`profile`.** Its base level (`ProfileBaseLevel`, `profile.rs:143-149`)
  hashes all parameter values, the fixed names, the fit config's bytes and the
  prior sources; its method level hashes `rw_sd` per estimated name
  (`profile.rs:1206-1219`), so the estimated set enters through both.
  `--estimate` bounds would enter neither. The partition's canonical form — each
  role, each fixed value, each domain with its prior, the simplex groups — is
  added to the base level beside the prior sources, which stay. This re-keys
  every profile.
- **`survey`.** Its config level hashes `fixed` and `scenario`, its box level
  the bounds (`survey_cas.rs:64-79`), and its model level hashes the compiled
  model (`survey.rs:313`), which today carries the seeded starts. Reproduced by
  an independent check at `1a059920`: changing one `[estimate].start` from 0.25
  to 0.26 moves the model segment from `sir-75a30421` to `sir-ec8867cf`,
  although a survey never reads the start. That is over-keying today. Stage 2
  removes the seeding, so every `survey --fit` re-keys once, and after that only
  for changes that move the landscape.
- **`fit run`.** Its fit level hashes the `Problem` (`fit/cas.rs`); a valid fit
  config's partition is a function of the file, so nothing re-keys beyond the
  engine version. `--starts from_scenario` is keyed with the rule, as every
  `--starts` is.
- **`simulate`, `fit predict`.** The params level hashes resolved values
  (`resolve.rs`); values are unchanged, so nothing re-keys beyond the engine
  version.

No IR change: no `ir/VERSION` bump, no golden moves.

## 11. Five stages, each green alone

Each stage lands on a green `make test`, and each named test is mutation-checked
(revert the source change, keep the test, confirm it fails).

**Stage 1 — the types, computed and recorded, no behaviour change.** Add
`roles.rs`. `fit run`, `survey` and `profile` build the partition from today's
rules (`profile`'s `--rw-sd` selection expressed as declarations) and write the
records. _Tests:_ unit tests of `resolve_roles` for every row of §4.2 and §4.3;
each leaf's record matches the cell projection.

**Stage 2 — `fit run` and `survey` resolve roles first.** Starts move out of the
model (§5.2) and onto the reworked chain-starts seam (§6.5); the launch block;
D1 and D2 for `fit run`; `survey`'s single branch. _Tests:_ `fit run`'s record
has no `parameters_provenance` entry for an estimated parameter; a model
constant runs without a `[fixed]` entry; two `survey --fit` runs differing only
in an `[estimate].start` share a key; a `real` parameter with no bounds gets
`(−∞, ∞)` and `survey` refuses it.

**Stage 3 — `profile` resolves roles first.** `--estimate`, `--slice`;
`[estimate]` authoritative; `--rw-sd` optional and IF2-only; `auto` removed;
simplex rules; `--starts from_scenario`; §4.3's scenario rules; §10.3's identity
change; §10.2's test migration. _Tests:_ the `[estimate] beta, gamma, k` +
`--sweep k` + `--rw-sd beta=0.02` profile estimates `beta` and `gamma`; a
profile with nothing estimated and no `--slice` is refused naming `--estimate`;
two profiles differing only in an `--estimate` bound have different base
digests; a split simplex is refused.

**Stage 4 — delete and document.** Everything in §8 the earlier stages left
unreferenced, `resolve_parameters` included; §10.2's documents.

During and after Stages 1-4, `simulate`, `fit predict`, `pfilter` and `eval`
call `resolve_values` with `Partition::all_fixed(model)` — every parameter
fixed, declared by default or by `--param`/`--params` — and keep today's
`engine::check_scenario_coordinate_collision` and `ParamSource`. Their values,
refusals and output are unchanged; their records list every parameter fixed,
with swept and drawn values tagged by tier, which is what they are in each cell.

**Stage 5 — `simulate`, `fit predict`, `pfilter`, `eval` on the full
partition.** After design-vs-sample Stage 1. The guard reads the partition
(§9.4); design shadows and impositions by `--param` are displacements, scenario
impositions per-arm findings. _Tests:_ design-vs-sample Stage 2's tests,
unchanged in substance; `simulate --estimate` is refused naming `simulate`.

## 12. Seven calls are the maintainer's

**D1 — `--sweep` over a `[fixed]` parameter.** Today `fit run --sweep`
_requires_ the swept name to be in `[fixed]` (`fit/mod.rs:519-535`), and every
`fit run --sweep` config is written that way; `profile --fit F --sweep N0` would
meet the same case. Options: (a) a displacement fixed → design with an info
notice, in every command; (b) refuse everywhere and require the name in neither
block, breaking every `fit run --sweep` config; (c) treat the design point as a
**value source for a parameter that is fixed in each cell**: `[fixed]` says the
parameter is not estimated, the sweep says which value each cell holds, and the
two do not conflict — the sweep's value replaces the `[fixed]` value at tier
3.5b with an info notice, the job records the role as design, and each cell
records it fixed at its point (§7.2). (c) matches what a cell actually runs,
needs no role change, and keeps every `fit run --sweep` config valid; (a) gives
the same output with one more kind of displacement to explain. **Recommendation:
(c). Leaning.**

**D2 — Model constants in the partition.** A typed constant `let` is an IR
parameter with a value (§2), but not equivalent to a `parameters {}` entry
(E512, gh#965). Options: (a) a model constant is fixed at its declared value,
needs no `[fixed]` entry, may be given another value by `--fixed`/`--param`, and
cannot be estimated, designed or sampled — to vary it, move it into
`parameters {}`; (b) also allow estimating it; (c) keep requiring a `[fixed]`
entry. (b) would make a role available that the compiler does not support for
every use of the constant; (c) is §3.2 item 7. **Recommendation: (a). Solid.**

**D3 — Undeclared parameters are fixed.** In a command that accepts fixed
parameters, a parameter no declaration names gets role fixed and must get a
value from the tiers. Options: (a) default fixed, printed and stored as
`declared_by: default`; (b) require a declaration for every parameter in the
inference commands, as `fit run`'s config check does. (a) is what `simulate`,
`pfilter` and `profile` do today and lets `--scenario` hold the rest of a
profile's parameters; (b) would make an inline `profile` list every parameter.
`--slice` (§4.4) closes the path by which (a) could silently turn a profile into
a slice. The fit config keeps its own completeness check, because a fit config
is the self-describing record of a fit. **Recommendation: (a). Leaning.**

**D4 — The stored record.** Options: (a) add `partition`, keep `estimated` and
`fixed` as projections written by one function; (b) replace them, and have
readers of older fits refuse. (b) would make every stored fit unreadable by
`fit predict` and `fit summary`; (a) costs two derived fields. **Recommendation:
(a). Leaning.**

**D5 — `--rw-sd auto` and `NAME=auto`.** `auto` meant "select every unfixed
parameter and size their steps from the bounds"; `NAME=auto` (the mixed form,
`RwSd::Map` of `Option<f64>`, `args/types.rs:392-395`; language spec ~5208)
meant "estimate `NAME` with an automatic step". Once `--rw-sd` is optional and
selects nothing, both mean what omitting them means. Options: (a) refuse both
with the message "`--rw-sd` no longer selects parameters: declare them with
`--estimate NAME,…` or `[estimate]` in a `--fit` config; steps not given are
derived from the bounds"; (b) accept both as no-ops. (b) would let
`--rw-sd auto` alone reach `--slice`'s refusal with a message that does not name
the cause. **Recommendation: (a). Solid.**

**D6 — A scenario named as the starts source.** Options: (a) a scenario that
`--starts from_scenario` also names may set estimated (and, in `profile`, focal)
parameters, whose values become starts (and are replaced by the grid); (b) no
exception: such a scenario is always refused, and a user wanting its
interventions passes `--enable`/`--disable` and its fixed values through a
`--fixed-file`. (b) leaves the `--scenario baseline` tests and workflows with
nothing to run until they restructure; (a) makes the migration one flag, visible
on the command line and in the record. **Recommendation: (a). Leaning.**

**D7 — Land the inference half before design-vs-sample Stage 2.** Options: (a)
§11 Stages 1-4 now, Stage 5 after design-vs-sample Stage 1, replacing its Stage
2; (b) design-vs-sample Stages 1-2 first, then this. (b) would implement the
scenario guard twice, once keyed on the axis and once on the role.
**Recommendation: (a). Solid.**

## 13. Out of scope, each with its reason

- **PMMH in `profile`** is being removed separately: a cell's highest-posterior
  sample is not a profile maximum. Stage 3 does not migrate `profile_pmmh.rs`.
- **Sweeping an `[estimate]` parameter in `fit run`.** Under IF2 that is
  `profile`; under PGAS or PMMH it is a grid of conditional posteriors, a
  command the maintainer has deferred. `fit run --sweep` over `[estimate]` stays
  refused.
- **The `rw_sd` scale convention** (natural-scale `rw_sd` converted at the
  current value, `sim/src/inference/types.rs:197-207`, `if2.rs:428`) goes to its
  own issue.
- **Multi-unit transitions** are separate work and do not touch parameter roles.
- **PMMH's own proposal tuning** (replacing `rw_sd × 5`) belongs to PMMH.
- **A scaled simplex** (free members summing to 1 − Σ held), needed before a
  partly held simplex can run; it changes `if2.rs`, a high-risk file.
- **Warm-starting a cell from its neighbour** would make a cell's identity
  depend on grid position and serialize the grid.
- **A profile interval against a trustworthy global maximum, with Monte Carlo
  error**, is a change to how profile results are summarized.
- **`profile --fit` ignores `ic_free`** (`skip_first_obs_from_loglik: false`,
  `profile.rs:1492`, verified): an estimand setting the fit config states and
  `profile` drops, to be filed as an issue.
