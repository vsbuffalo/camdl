# Proposal: a typed structural diff between models (`camdl diff`)

**Status:** Proposal **Date:** 2026-11-13 **Motivation:** camdl can say that two
models differ, as a boolean or a pair of hashes, but not how; the published
comparison viewer has to show what each model assumes that the others do not,
and what they all assume alike, from a typed report and not from a text diff or
a hand-written sentence.

**Drafted with Claude** from the design conversation of 2026-11-13. Code
references are to `c1f27d9d` (2026-11-13). Claims marked _verified_ were read in
that tree, and hashes marked _measured_ were computed there; the evidence is
collected under [Settled facts](#settled-facts). Statements marked _inferred_
were not run and are each a test the implementation must write first.

Proposal 4 of the published-viewer set; see
`2026-11-13-published-comparison-principles.md` for vocabulary and rules. A
"bundle" here is that document's published form of one fit, not the `camdl mre`
reproduction bundle of `2026-06-09-mre-bundle.md`.

## The rule this proposal sets

**A model difference is a typed report over the named things an author wrote,
computed by camdl, before it is a sentence or a highlighted arrow.** The viewer
draws the report. It does not parse model source, compare expressions or decide
what counts as the same.

And one limit, stated on every rendering: **the diff compares what is written,
not how the models behave.** Two parameterizations of one model differ in the
diff. The predictive score is the behavioural comparison.

## What exists today

All _verified_.

- **A yes/no identity check.** `compare_to_fitted` returns `ModelAgreement`:
  `Same`, `ReportingDiffers` (only `quantities`/`contrasts` differ), or
  `Differs { fitted, given }` carrying two hashes
  (`rust/crates/cli/src/fit/fitted_model.rs:36-60`).
- **One model identity.** `model_ir_hash` is "the single answer to which model
  is this" (`rust/crates/runid/src/inputs.rs:309-319`). It strips presentation
  fields before hashing (`normalize_for_hash`, `inputs.rs:332-337`); the hash
  itself is implemented in `rust/crates/runid/src/ir_hash.rs:1490-1548`.
- **A typed diff of fit configuration.** `ConfigDiff` covers estimated and fixed
  parameter sets, fixed values, bounds, priors, data hashes and method settings.
  Its only statement about the model is `model_changed: bool`
  (`rust/crates/cli/src/fit/config_diff.rs:30-61`).
- **A command that prints it.** `camdl fit diff A.toml B.toml` takes two fit
  configuration files, compiles each one's model, and prints four sections:
  Model (changed or not, by hash), Data (per-stream content hash), Parameters
  and Method (`rust/crates/cli/src/fit/mod.rs:2840-2925`). Text only.
- **A pre-expansion flow graph, in the compiler.** The `camdlc render` command,
  with its graph format, builds `model.graph.json` from the declarations: base
  compartments as nodes, dimensions as plates, one edge per source and
  destination pair of each declared transition, with a KaTeX rate string and the
  aggregates each edge reads (`ocaml/lib/compiler/latex.ml:533-560`,
  `build_graph` at `:704`). It works on parsed declarations and never expands
  the model.
- **An envelope around the compiled model.** `IrEnvelope` holds the IR version,
  the model, and `docs`, the doc-comment dictionary, which "sits outside
  `Model`" and is "never part of `run_id`"
  (`rust/crates/ir/src/envelope.rs:84-100`).
- **Typed parameter state.** `ParamValue` is `Fixed { value }`,
  `Estimated { init, bounds, prior }` or `Required`
  (`rust/crates/ir/src/parameter.rs:189-208`).

## The gap

`model_changed: true` is where a reader's real question starts. Which
compartment, which rate, which prior, which observation model? Today the answer
is to read two `.camdl` files side by side. That does not scale past two small
models, cannot be shown to a non-modeller, and cannot be checked.

Three things rule out the obvious shortcuts:

- **A text diff of source** breaks on reordering and formatting, and says
  nothing typed.
- **A diff of the expanded model** turns one changed stratified transition into
  thousands of rows, and reports every stratified transition as changed when a
  dimension gains one level, although no declared rate was touched.
- **A diff of the display graph** would compare KaTeX strings, which are a
  rendering and not an identity.

## Design

### The diff runs on declarations

Row status is decided on what the author declared. The compiled model is a
cross-check on the result and never the source of a row's status.

Inputs, for each side:

1. The **declared model**: a new typed section of the IR envelope, written by
   the compiler. Specified next.
2. The compiled model's identity, `model_ir_hash`, for the cross-check.
3. When the side is a fit, its **config facts** record (see
   [Where the code lives](#where-the-code-lives)).

The diff is a pure function of these. It reads no files and consults no store,
so it can run in a browser on two bundles.

### The declared model

A new section of the IR envelope, beside `docs`: written by the same compile
that writes the model, outside `Model`, and outside every model identity and run
id. One compile writing both is what guarantees that the declarations and the
compiled model describe the same model; a second compiler invocation at archive
time could see different `--set` values or data files.

It holds the declared model as typed data, not a summary derived for display.

```rust
pub struct IrEnvelope {
    pub ir_version: String,
    pub validated_by: Option<String>,
    pub model: Model,
    pub docs: ModelDocs,
    pub declared: Option<DeclaredModel>,   // None: not recorded
}

pub struct DeclaredModel {
    pub schema_version: u32,               // its own; starts at 1
    pub compile_inputs: CompileInputs,     // --set values; digests of files read
    pub dimensions:    Vec<DimensionDecl>,   // name, resolved levels, level source
    pub compartments:  Vec<CompartmentDecl>, // name, dims, kind
    pub transitions:   Vec<TransitionDecl>,  // name, from, to, rate or via, draw method
    pub definitions:   Vec<DefinitionDecl>,  // every `let`: name, indices, body
    pub parameters:    Vec<ParameterDecl>,   // name, state, prior, bounds, kind, dim
    pub constants:     Vec<ConstantDecl>,
    pub time_inputs:   Vec<TimeInputDecl>,   // time functions, tables, splines
    pub interventions: Vec<InterventionDecl>, // interventions, events, reactive
    pub scenarios:     Vec<ScenarioDecl>,
    pub initial:       Vec<InitialDecl>,     // compartment, expression or law
    pub observations:  Vec<ObservationDecl>, // stream, family, arguments, schedule,
                                             // source, columns, covers, projection
    pub simulation:    SimulationDecl,       // dt, time unit, origin, start, end,
                                             // seed, integrator
    pub output:        OutputDecl,
    pub balance:       Option<BalanceDecl>,
    pub ode_equations: Vec<OdeDecl>,
    pub quantities:    Vec<QuantityDecl>,
    pub contrasts:     Vec<ContrastDecl>,
}

pub struct DeclaredExpr {
    pub tree:  ExprTree,     // the parsed form; what sameness is decided on
    pub katex: String,       // what is shown
    pub reads: Vec<NameRef>, // the names it reads, each with its kind
}
```

Each list is in declaration order. `ExprTree` is the compiler's parsed
expression type (`expr`, `ocaml/lib/compiler/ast.ml:77-105`) with source
locations removed and each name tagged with what it resolved to (compartment,
parameter, definition, constant, time input, index variable).

Each declared transition, compartment and observation stream also carries
`expands_to`, the expanded names it produced. Nothing else links the expanded
model back to a declaration (see [Settled facts](#settled-facts)), and the
viewer and the assumption ledger (proposal 5) need the link to map per-stratum
outputs back to declarations. Row status never depends on it.

**The section is readable without the runtime's reader for compiled models.**
That reader refuses any envelope whose IR version is not its own
(`envelope.rs:126-133`), and a published bundle is read for as long as it is
cited. So `DeclaredModel` is a self-contained document: it deserializes through
its own types in the diff crate, is versioned by its own `schema_version` and
never by `ir_version`, refers to nothing in `Model`, and every published version
of its schema stays readable. `camdl publish` copies it out of the envelope into
the bundle as its own file, beside the model identity as a string, so neither
the viewer nor `camdl diff` on two bundles ever loads a compiled model. A test
reads a `declared` document from an envelope whose `ir_version` is not the
current one.

`declared` absent means not recorded, and the diff then reports itself
unavailable for that side. It is never backfilled by recompiling at archive
time. `model.graph.json` stays for its existing consumers and becomes a
projection of this section.

### When two expressions are the same

**Two declared expressions are the same when they parse to the same tree.**
Nothing is normalized beyond what parsing already does.

Ignored, because the parsed tree does not hold it:

- whitespace, line breaks and comments;
- redundant parentheses (the tree has no parenthesis node; _inferred_ from the
  constructor list, `ast.ml:77-105`);
- the spelling of a number: `1` and `1.0` are one constant (_inferred_:
  `EConst of float`);
- source locations, which are stripped on emission.

Not ignored:

- operand order: `beta * S * I` and `S * beta * I` differ;
- grouping that changes the tree: `beta * S * (I / N)` and `beta * S * I / N`
  differ;
- units: `7 'days` and `1 'weeks` differ;
- the name of a bound index variable: `sum(a in age) …` and `sum(b in age) …`
  differ;
- any algebraic rearrangement, and any reparameterization (`β` against `R0·γ`).

| Left                 | Right                       | Same? |
| -------------------- | --------------------------- | ----- |
| `beta * S * I / N`   | `beta*S*I/N  # mass action` | yes   |
| `(beta * S) * I`     | `beta * S * I`              | yes   |
| `gamma * 1 * I`      | `gamma * 1.0 * I`           | yes   |
| `beta * S * I`       | `S * beta * I`              | no    |
| `beta * S * (I / N)` | `beta * S * I / N`          | no    |
| `gamma * I`          | `(1 / D) * I`               | no    |

Those that differ here but compile to the same model are reported as such; see
[Three outcomes](#three-outcomes).

**A `let` is compared by name, and its body by tree.** Every `let` is a
Definition row. An expression that reads a `let` holds a reference to its name,
so a transition whose rate reads `N` is `Same` as written when `let N` changes,
and the Definition row for `N` is `Changed`. That is the correct statement about
what is written, and on its own it would leave the infection arrow unlit when
the force of infection changed. So each row also carries `affected_by`: the
changed definitions, constants and time inputs it reads, directly or through
other definitions, computed from `reads`. A row that is `Same` with a non-empty
`affected_by` is shown as "unchanged as written; reads a changed definition",
and the flow diagram highlights it.

Renaming a `let` shows as one definition only-left and one only-right, and every
expression reading it as `Changed`. Whether the compiled model agrees is the
cross-check's business: the compiler inlines a `let` that reaches a parameter
and keeps a parameter-free one under its name
(`ocaml/lib/compiler/expander.ml:3511-3539`), so the first rename compiles to
the same model and the second does not (measured).

### Matching entities

Entities are matched **by category and name, exactly.** No fuzzy matching and no
structural guessing.

- Compartments, dimensions, levels, definitions, parameters, constants,
  observation streams, scenarios, quantities: by name.
- Transitions: by declared name. A branching or multi-source transition fans out
  to several flow-graph edges with ids `name__src__dst` (`latex.ml:676-680`), so
  each edge carries the declared name it belongs to and the viewer highlights by
  that.
- Unmatched entities are reported as only-left or only-right.

For models from different groups, names will differ. A published comparison may
declare a **rename map** per member:

```toml
[members.seir_simple.rename.compartments]
Inf = "I"
[members.seir_simple.rename.streams]
cases_weekly = "confirmed_cases"
```

The map must be a bijection onto names that exist; otherwise the diff refuses.
It is applied by rewriting the tagged names in the declared trees of the renamed
side before comparison. A match made through a map is a claim by the
comparison's author, and is shown as one in two places:
`MatchedBy::DeclaredRename` on every such row, so the mark survives a row or
figure being exported alone, and one list of all declared renames at the top of
the report.

Suggested matches ("`Inf` and `I` have identical stoichiometry; possibly the
same compartment") are not in this proposal: no cross-group comparison exists
yet to design them against. `DiffReport` schema version 1 has no field for them.

### Classification

```rust
pub enum Status<T> {
    Same,
    Changed { fields: Vec<FieldChange<T>> },
    OnlyLeft,
    OnlyRight,
}
pub enum MatchedBy { Name, DeclaredRename }
pub enum Layer { Model, Reporting, Fit }
```

Every row is tagged with its layer.

**Model layer**: what the model file declares and `model_ir_hash` covers.

| Category             | Changed when                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------------- |
| Dimension            | Level set differs                                                                             |
| Compartment          | Dimensions or kind differ                                                                     |
| Transition           | From, to, rate tree, staging law or draw method differ                                        |
| Definition           | Indices or body tree differ                                                                   |
| Constant             | Value differs                                                                                 |
| Parameter (declared) | State, fixed value, prior, bounds, kind or dimension, as the model file declares them, differ |
| Initial condition    | Expression tree or law differs                                                                |
| Observation stream   | Family, argument trees, schedule, source label, columns, `covers` or projection differ        |
| Time input, table    | Definition differs                                                                            |
| Intervention, event  | Definition differs                                                                            |
| Scenario             | A declared `scenarios {}` entry differs                                                       |
| Simulation           | Step, time unit, origin, start, end, seed or integrator differ                                |
| Output               | Output times or the trajectory and observation flags differ                                   |
| Balance, ODE         | Definition differs                                                                            |
| Declaration order    | The same entities of one category appear in a different order                                 |
| Compile input        | A compile-time `--set` value or the digest of a file read at compile time differs             |

Declaration order is a category, and not ignored, because the hash covers it:
reordering transitions, parameters, compartments or `init` entries changes
`model_ir_hash` (measured), and the order of transitions changes random-number
consumption. Moving a `let` does not change the hash and is not an order
difference.

**Reporting layer**: `quantities` and `contrasts`, which the hash excludes.
Shown separately.

**Fit layer**: facts that belong to a fit and not to a model.

| Category              | Changed when                                                                      |
| --------------------- | --------------------------------------------------------------------------------- |
| Parameter (effective) | State, fixed value, prior or bounds differ after the fit configuration is applied |
| Applied scenario      | The fits ran under different scenarios, or enable and disable lists               |
| Backend               | The fits ran on different backends                                                |
| Data                  | A data file's content hash differs                                                |
| Method settings       | Per the existing `ConfigDiff`                                                     |

**Parameter state is reported as actually fitted, each fact attributed to the
layer that set it.** Where each lives (_verified_):

| Fact               | Model file                                               | Fit configuration                                                                                  | Which wins                                                                                         |
| ------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Fixed or estimated | `ParamValue` (`parameter.rs:189-208`)                    | `[estimate]` and `[fixed]`, which together must cover every parameter (`fit/config_v2.rs:115-119`) | The fit configuration always                                                                       |
| Fixed value        | `ParamValue::Fixed { value }`, or a compile-time `--set` | `[fixed]`                                                                                          | The fit configuration                                                                              |
| Prior              | `~` in the model (`PriorSpec` inside `Estimated`)        | `[estimate.<p>.prior]` (`config_v2.rs:585-600`)                                                    | Fit configuration, then model, then flat only by explicit opt-in (`fit/priors_precedence.rs:4-42`) |
| Bounds             | `in [lo, hi]`                                            | `[estimate.<p>].bounds`, which may tighten but not loosen (`config_v2.rs:575-583`)                 | Fit configuration when present, else model                                                         |
| Scenario           | `scenarios {}` definitions                               | `scenario`, `enable`, `disable` (`config_v2.rs:128-134`)                                           | Definitions are model; the choice is fit                                                           |

The model identity a fit stores is the model as compiled, without the fit
configuration applied. So the declared row and the effective row can disagree in
both directions: a parameter one fit fixes through `[fixed]` is `Same` as
declared, and a prior edited in the model file but overridden identically in
both configurations is `Changed` as declared although both fits used the same
prior.

The effective row is the one given prominence, and it names the source of each
fact ("prior: log-normal(−1, 0.5), set in fit configuration"). "Model A
estimates the incubation period; model B fixes it at 9 days" is an assumption
difference a reader must not miss, it is invisible in a flow diagram, and it is
always a fit-layer fact. The "same prior" label beside two posteriors reads the
effective row. When only model files are compared there are no fit rows.

### Three outcomes

The model layer has two independent facts: whether the declarations differ, and
whether the compiled models differ. The report states both.

| Declared rows   | Compiled hashes | Outcome                                                          |
| --------------- | --------------- | ---------------------------------------------------------------- |
| All `Same`      | Equal           | **Identical.**                                                   |
| Some not `Same` | Different       | **Different.** The rows say how.                                 |
| Some not `Same` | Equal           | **Written differently; compiles to the same model.** Both shown. |
| All `Same`      | Different       | **Refused.** A defect, in camdl or in this proposal's coverage.  |

The third outcome is expected: a `let` that reads a parameter is renamed, and
the compiler inlines it either way. The rows are shown, and so is the statement
that the two files compile to one model.

The fourth must not occur between two models compiled by one compiler version,
because the declared model then covers every input to compilation, including
compile inputs and declaration order. If it does occur the diff refuses with
both hashes, and does not report "identical". Between two compiler versions it
can occur legitimately, and is reported as "declared the same; compiled by
different camdl versions (`x`, `y`)".

### Roll-ups

Because rows are decided on declarations, adding a level to a stratifying
dimension is one Dimension row and no Transition rows: the declared rates are
untouched. Most of what an expanded diff would need to roll up never arises.

One roll-up remains. Adding or removing a whole stratifying dimension changes
the declared dimensions of every compartment it applies to and the index lists
in every rate that reads them. The report then carries "dimension `age` added; 6
compartments and 9 transitions restratified, otherwise unchanged", emitted only
when every affected row differs in exactly that respect: its tree equals the
other side's after the dimension's index is removed. The rows beneath stay
available.

### More than two models

- **Pairwise against a baseline**, matching how scores are shown.
- **Shared by all**: every entity whose status is `Same` across all members.
  This list is the "assumptions a comparison cannot test" required by principle
  R9, and it is an output of the diff, not a separate computation.
- **A matrix** of counts by category for N members, for the overview.

All three are projections of the same pairwise reports. Which of them a screen
leads with is a layout decision for the viewer proposal.

### The report

```rust
pub struct DiffReport {
    pub schema_version: u32,
    pub left:  ModelRef,   // name, model identity, compiler version, bundle hash
    pub right: ModelRef,
    pub model_outcome: ModelOutcome,     // Identical | Different | WrittenDifferently
    pub model_identical: bool,           // model_ir_hash(left) == model_ir_hash(right)
    pub reporting_identical: bool,       // quantities and contrasts, order included
    pub fit_identical: Option<bool>,     // None when a side is a bare model file
    pub renames: Vec<DeclaredRename>,
    pub rows:    Vec<Row>,               // layer, category, name(s), matched_by,
                                         // status, affected_by
    pub rollups: Vec<Rollup>,
    pub config:  Option<ConfigDiff>,
    pub counts:  BTreeMap<(Layer, Category), Counts>,
}
```

`model_identical` is the hash comparison itself, never derived from rows.
`reporting_identical` agrees with `compare_to_fitted` (`fitted_model.rs:49-60`),
which compares `quantities` and `contrasts` as ordered lists.

When one side is a model file and the other a fit, the file is first resolved
with `as_fitted` (observation anchors resolved to numbers, as the fit archived
its model), so the two hashes are of the same form.

Everything the mockup shows is a projection of this:

| Mockup element                        | Source                                                                                  |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| Summary tiles ("1 changed of 6")      | `counts`                                                                                |
| Highlighted arrow in the flow diagram | Transition rows that are not `Same` or have a non-empty `affected_by`, by declared name |
| Side-by-side detail table             | `rows`, with `DeclaredExpr.katex` for display                                           |
| "Same prior" beside two posteriors    | effective parameter row; posteriors come from the bundles, not the diff                 |
| "View source diff"                    | the two source files; a convenience, labelled as text                                   |
| "Shared by all models"                | the N-way shared list                                                                   |

The page's headline sentence is fixed wording over `model_outcome`, `counts` and
`rollups` ("B is A with one changed rate and one added parameter"). An author
may annotate it; the viewer does not generate interpretation.

### Where the code lives

- The declared model is emitted by the compiler on the compile path, after
  expansion, so that names are resolved and `expands_to` is known. Types in
  `ocaml/lib/ir/` and `rust/crates/ir/src/`.
- The diff is a Rust crate depending on the declared-model types, `serde` and
  `serde_json`. It hashes nothing: sameness is tree equality, and the two model
  hashes arrive as inputs. It reads no files and starts no threads, so it is
  meant to compile to WASM beside the scoring core (proposal 3). That build has
  not been attempted; step 8 settles it. The CLI and the viewer call the same
  function.
- `ConfigDiff` today derives `Serialize` only, holds `serde_json::Value`, and is
  built from `FitConfig`, `Method` and `PriorDist` in the `cli` crate, which
  parse TOML (`config_diff.rs:18-29`, `:152-185`), so it cannot move as it is.
  Each fit instead writes a serializable **config facts** record: for every
  parameter its effective state, value, prior and bounds with the source of
  each; the applied scenario and enable and disable lists; the backend; data
  hashes; method settings. `ConfigDiff` becomes a pure function of two such
  records and lives in the diff crate; the `cli` crate builds the record.

### Command

```
camdl diff A B [--rename FILE] [--format table|md|json]
```

`A` and `B` are each a model file, a fit configuration file, a fit handle or a
bundle path. Two model files give the model and reporting layers; anything else
adds the fit layer. Exit status 0 when identical in every layer present, 1 when
different, 2 on refusal (invalid rename map, unreadable input, declared model
not recorded, the fourth outcome above).

`camdl diff` replaces `camdl fit diff`, which is removed. Everything that
command prints is kept as the fit layer: the Data, Parameters and Method
sections, and its handling of a side whose model does not load (that side is
reported, not compared). Its Model section, which says only "compiled model
differs", is what this proposal replaces. `fit table`'s `model_changed` column
gains a pointer to `camdl diff`.

## The IR change

Adding `declared` to the envelope is an IR schema change: `ir/VERSION` goes from
0.43 to 0.44, the OCaml and Rust types change together, and every golden is
regenerated and reviewed.

- **One bump for all declaration kinds.** The section carries its own
  `schema_version`, so later additions to it need no further IR bump.
- **Identity is untouched, and a test proves it.** Before the change, record
  `model_ir_hash` and the run ids for every golden; after it, they are
  byte-identical. The test is committed with the change.
- **Old compiled models.** The runtime refuses any envelope whose `ir_version`
  is not its own (`envelope.rs:126-133`), so a model compiled before the bump is
  not read at all, with or without this section. `declared: None` therefore
  arises only for an envelope written at the current version without the
  compiler (hand-edited, or serialized from Rust). Both cases report the diff as
  unavailable for that side.
- **Landing.** The golden regeneration conflicts with any branch that touches
  `ocaml/lib/ir/`, `rust/crates/ir/src/` or a golden. It lands directly after
  the fix for gh#1046, which also regenerates goldens (and re-keys models with
  hoisted subtrees), so the two reviews of the golden set happen back to back,
  as separate commits.

## Invariants

Each is a test.

1. `diff(a, a)` has no rows other than `Same`, and the outcome is Identical.
2. Swapping the arguments swaps `OnlyLeft` and `OnlyRight` and nothing else.
3. For two models compiled by one compiler version, model layer only:
   - 3a. `model_identical` is `model_ir_hash(a) == model_ir_hash(b)`.
   - 3b. If every model-layer row is `Same`, the hashes are equal. Tested over
     every pair in the fixture corpus and by a property test that perturbs
     formatting, comments, parentheses and `let` position.
   - 3c. If the hashes are equal and some model-layer row is not `Same`, the
     outcome is WrittenDifferently, and the rows are still reported.
4. `reporting_identical` is false exactly when `compare_to_fitted` on models
   with equal hashes returns `ReportingDiffers`.
5. Every declared entity of each model appears in exactly one row per layer.
6. Applying a rename map and then diffing equals diffing the model recompiled
   from renamed source.
7. A roll-up never hides a row that differs in more than the rolled-up respect.
8. The report is byte-identical across runs and thread counts.
9. Compiling any golden yields the same `model_ir_hash` and run ids with and
   without the `declared` section.

## What it does not do

- Decide whether two differently written models are mathematically equivalent.
  It reports the one case it can observe: they compiled to the same model.
- Detect reparameterizations.
- Match entities across naming schemes without a declared map.
- Say which model is better. That is the score.
- Diff posteriors or predictions. Those are shown beside the diff, from the
  bundles.

## Tests and fixtures

A corpus of model pairs with known answers, drawn where possible from
`ocaml/golden/`: add a parameter; drop a compartment; change one rate; change a
prior family; fix a previously estimated parameter in the model file, and
separately through `[fixed]`; change an observation family from Poisson to
negative binomial; add a level to a dimension; add a stratifying dimension;
rename a compartment with and without a map.

Pairs whose expected outcome follows from how the compiler and the hash behave
(hash behaviour measured; see [Settled facts](#settled-facts)):

| Pair                                                    | Rows                                                    | Outcome            |
| ------------------------------------------------------- | ------------------------------------------------------- | ------------------ |
| Reformat; add a `#'` doc comment; move a `let`          | All `Same`                                              | Identical          |
| Rename a `let` that reads a parameter                   | Definition only-left and only-right; readers `Changed`  | WrittenDifferently |
| Rename a parameter-free `let`                           | The same rows                                           | Different          |
| Change the body of a `let`                              | Definition `Changed`; readers `Same` with `affected_by` | Different          |
| Reorder transitions, parameters, compartments or `init` | One Declaration order row                               | Different          |
| Add a level to a stratifying dimension                  | One Dimension row; no Transition row                    | Different          |
| Add a `scenarios {}` entry                              | Scenario only-right                                     | Different          |
| Change `simulate { to = … }`                            | Simulation `Changed`                                    | Different          |
| Compile one side with `--set beta=0.3`                  | Compile input `Changed`                                 | Different          |

The compiler's `declared` output is covered by the regenerated goldens, since it
is part of each `*.ir.json`. The diff's own golden reports use a named subset of
about ten model pairs covering stratification, `via` staging, branching, indexed
observations and definitions.

The Bundibugyo variants are the first real fixture set, and the source of the
viewer's first published comparison.

## Steps

1. Fix gh#1046: `model_ir_hash` differs between a full and a `--no-state-grad`
   compile of one source when a subtree is hoisted. Until it is fixed, a model
   file compared with its own fit can land in the refused fourth outcome.
2. The `declared` envelope section: OCaml and Rust types, emission, `ir/VERSION`
   0.44, goldens, invariant 9. Tests for the three _inferred_ parse facts.
3. The config facts record, and `ConfigDiff` as a function of two of them.
4. The diff crate: tree sameness, matching, the three layers, `affected_by`, the
   three outcomes, invariants 1 to 5 and 8.
5. The dimension roll-up and the N-way shared list, invariant 7.
6. Rename maps, invariant 6.
7. `camdl diff` and its three renderers, all from `DiffReport`; remove
   `camdl fit diff`.
8. WASM build; the viewer's structure screen reads `DiffReport` from fixtures.

Steps 1 to 7 need no published bundle: their inputs are model files, fit
configuration files and fit directories. Step 8, and copying anything into a
bundle, wait on the bundle proposal (1 of the set) and the scoring-core proposal
(3), neither of which is written. None of the steps depends on the compare
cleanup branch (`worktree-peppy-jingling-kite`), which does not touch the IR.

## Settled facts

What the design rests on, each established at `c1f27d9d`.

**An expanded transition does not record its declaration.** `TransitionMetadata`
holds only `origin_kind`, `source_compartment` and `dest_compartment`
(`rust/crates/ir/src/transition.rs:14-21`). The link exists as a naming
convention (`<declared>_<level>…`, `_<branch destination>`, `_stage_sN` for
`via`) that cannot be parsed backwards: `onset` under `via erlang(stages = 3)`
compiles to `onset_stage_s1`, `onset_stage_s2`, `onset`, and level names may
contain underscores (measured on `ocaml/golden/seir_erlang_via`). The compiler
has the forward map, `expand_transition_name` (`expander.ml:4901`).

**What `model_ir_hash` excludes** (`ir_hash.rs:146`, `:371`, `:400`, `:798`,
`:975`, `:1494-1508`; `inputs.rs:332-337`): the model name; `quantities`;
`contrasts`; `ic_grad` and every gradient map; a forcing's `data_source` path
and file digest; and, by normalization, `output.format` and
`simulation.time_semantics`. Doc comments sit outside the model. Everything else
is hashed, order included.

**Measured hashes** (first 16 hex digits) for variants of a three-compartment
SIR model whose base hashes to `47796106ad9e035d`:

| Variant                                              | Hash               | Same as base       |
| ---------------------------------------------------- | ------------------ | ------------------ |
| Move `let N` below the transitions block             | `47796106ad9e035d` | yes                |
| Add a `#'` doc comment                               | `47796106ad9e035d` | yes                |
| Rename `let N` to `let Npop` (parameter-free)        | `2401a1e122823566` | no                 |
| Change the let body to `S + I`                       | `4822bdfd1f803d25` | no                 |
| Reorder the two transitions                          | `647bdf297cd36a1e` | no                 |
| Reorder two parameters                               | `6082e8974b86acfb` | no                 |
| Reorder compartments                                 | `1efa5e76053e34a7` | no                 |
| Reorder `init` entries                               | `f5cb65109ee15f63` | no                 |
| Add a `scenarios {}` entry                           | `98dac2ff3d286134` | no                 |
| Change `simulate { to = 90 }`                        | `d2094837c034d278` | no                 |
| Compile with `--set beta=0.3`                        | `d390aca39c4279f8` | no                 |
| Parameter-bearing let `foi`, then renamed `lambda_t` | `5d1b71108f21631d` | same as each other |

**The expanded model is not a usable source of row status.** A compiled rate
refers to a kept `let` and to a subtree hoisted by loop-invariant code motion by
name, and the hoist names are numbered over the whole model
(`ocaml/lib/ir/licm.ml:207-222`). Measured: changing `let N = S + I + R` to
`S + I` leaves the stored hash of `infection @ beta * S * (I / N)` at
`aa9bfb2ff527`; adding an unrelated earlier transition moves the stored hash of
`recovery @ gamma * exp(-kappa) * I` from `eb8a0ad27024` to `cd69a0d7f98f`.

**No canonical form exists for pre-expansion expressions**, and none is
introduced. On expanded expressions there are `Licm.canon` (`licm.ml:114`),
`normalize_expr` (`expander.ml:2576`) and `constant_fold.ml`; none
alpha-renames.

**`model.graph.json` has consumers**: the archive writer
(`rust/crates/cli/src/fit/mod.rs:1448-1453`), an integration test
(`rust/crates/cli/tests/model_graph_render.rs`), `ocaml/test/test_latex.ml`, the
run spec (`docs/camdl-run-spec.md:409`) and the camdl-scope viewer.

**Observation streams** are one `obs_decl` per stream family before expansion,
with an index list, a measurement, a projection, an optional emit schedule and
`covers` (`ocaml/lib/compiler/ast.ml:403-453`). After expansion each stratum is
its own observation model with a `stratum` key.
