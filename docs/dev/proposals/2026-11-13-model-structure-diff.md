# Proposal: a typed structural diff between models (`camdl diff`)

**Status:** Proposal **Date:** 2026-11-13 **Motivation:** camdl can say that two
models differ, as a boolean or a pair of hashes, but not how; the published
comparison viewer has to show what each model assumes that the others do not,
and what they all assume alike, from a typed report and not from a text diff or
a hand-written sentence.

**Drafted with Claude** from the design conversation of 2026-11-13. Code
references are to `main` at `2e2f663` (2026-11-09). Claims marked _verified_
were read in that tree. Claims marked _to verify_ were not, and the agent
implementing this should settle them first (see
[To verify before starting](#to-verify-before-starting)).

Proposal 4 of the published-viewer set; see
`2026-11-13-published-comparison-principles.md` for vocabulary and rules.

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
  fields before hashing (`normalize_for_hash`, `inputs.rs:332-337`); the module
  doc of `fitted_model.rs` says the identity also excludes `quantities`,
  `contrasts`, the model's file-stem name and the gradient maps.
- **A typed diff of fit configuration.** `ConfigDiff` covers estimated and fixed
  parameter sets, fixed values, bounds, priors, data hashes and method settings.
  Its only statement about the model is `model_changed: bool`
  (`rust/crates/cli/src/fit/config_diff.rs:30-61`).
- **A pre-expansion flow graph, in the compiler.**
  `camdlc render --format
  graph` builds `model.graph.json` from the
  declarations: base compartments as nodes, dimensions as plates, one edge per
  transition family, with a KaTeX rate string and the aggregates each edge reads
  (`ocaml/lib/compiler/latex.ml:533-560`, `build_graph` at `:704`).
- **An expanded model, in the runtime.** `ir::Model` holds the transitions after
  strata are expanded, plus `ModelStructure` recording dimensions, base
  compartments and each compartment's dimensions
  (`rust/crates/ir/src/model.rs:380-386`, `:406`).
- **Typed parameter state.** `ParamValue` is `Fixed { value }`,
  `Estimated { init, bounds, prior }` or `Required`
  (`rust/crates/ir/src/parameter.rs:189-208`).

So the pieces are: a sound notion of "same model", a config diff with a hole
where the model should be, and a compact structural view that today carries only
display strings.

## The gap

`model_changed: true` is where a reader's real question starts. Which
compartment, which rate, which prior, which observation model? Today the answer
is to read two `.camdl` files side by side. That does not scale past two small
models, cannot be shown to a non-modeller, and cannot be checked.

Three things rule out the obvious shortcuts:

- **A text diff of source** breaks on reordering, renaming of local bindings and
  formatting, and says nothing typed.
- **A diff of the expanded model** turns one changed stratified transition into
  thousands of rows.
- **A diff of the display graph** would compare KaTeX strings, which are a
  rendering and not an identity.

## Design

### Inputs

For each model:

1. A **structure summary**: a new typed artifact, the model as the author wrote
   it, before expansion. Specified below.
2. The compiled model (`ir::Model`), for sound equality of expressions.
3. The fit configuration and sidecar, through the existing `ConfigDiff`.

The diff is a pure function of these. It reads no files and consults no store,
so it can run in a browser on two bundles.

### The structure summary

A pre-expansion projection, emitted by the compiler alongside
`model.graph.json`, which it generalizes. Sketch of the types (names are
proposals):

```rust
pub struct StructureSummary {
    pub schema_version: u32,
    pub dimensions:   Vec<Dimension>,        // name, levels
    pub compartments: Vec<CompartmentDecl>,  // name, dims, kind
    pub transitions:  Vec<TransitionDecl>,   // id, from, to, rate, reads
    pub parameters:   Vec<ParameterDecl>,    // name, state, prior, bounds, kind, dim
    pub constants:    Vec<ConstantDecl>,
    pub time_inputs:  Vec<TimeInputDecl>,    // time functions, tables, splines
    pub interventions: Vec<InterventionDecl>,
    pub initial:      Vec<InitialDecl>,      // compartment, expression
    pub observations: Vec<ObservationDecl>,  // stream, family, mean, params, schedule
    pub simulation:   SimulationDecl,        // backend, dt, time unit, origin
    pub quantities:   Vec<QuantityDecl>,
}

pub struct ExprView {
    pub identity: ContentHash, // what equality is decided on
    pub katex:    String,      // what is shown
    pub reads:    Vec<Name>,   // parameters, compartments, aggregates it uses
}
```

Every expression appears as an `ExprView`: an identity for comparison, a
rendering for display, and the names it reads (which the assumption ledger,
proposal 5, also needs).

### Deciding that two expressions are the same

This is the part most likely to go wrong, so it gets one rule: **expression
equality must agree with model identity.** If `model_ir_hash` says two models
are the same, the diff must be empty, and the reverse.

Two ways to compute `ExprView.identity`:

- **A. Canonical form in the compiler.** Normalize the pre-expansion expression
  (order commutative operands, fold constants, inline or alpha-rename local
  bindings) and hash it. Compact, but it is a second definition of sameness that
  can drift from the runtime's.
- **B. Hash of the expanded leaves.** Take the expanded transitions that one
  declared transition produced, hash each leaf's rate as the runtime sees it,
  and hash the sorted set. Sameness is then exactly the runtime's.

**Recommendation: B for identity, the compiler's rendering for display.** It
needs each expanded transition to record which declared transition it came from
(_to verify_: whether `TransitionMetadata` or the transition name already
carries this). If that provenance is missing, add it.

Whichever is chosen, the proposal fixes what is deliberately _not_ recognized as
the same: algebraic rearrangement beyond the normalization, and
reparameterization (`β` against `R0·γ`). Those show as changed, and the page
says the diff is about what is written.

### Matching entities

Entities are matched **by category and name, exactly.** No fuzzy matching and no
structural guessing.

- Compartments, dimensions, levels, parameters, constants, observation streams,
  quantities: by name.
- Transitions: by declared id.
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
Matches made through a map are labelled as declared by the comparison's author,
so a reader can see they are a claim.

The diff may also emit **hints**: "`Inf` and `I` have identical stoichiometry
and rate identity; possibly the same compartment". Hints are a separate list,
typed as non-authoritative, never applied automatically.

### Classification

```rust
pub enum Status<T> {
    Same,
    Changed { fields: Vec<FieldChange<T>> },
    OnlyLeft,
    OnlyRight,
}
pub enum MatchedBy { Name, DeclaredRename }
```

What "changed" means per category:

| Category                 | Changed when                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| Dimension                | Level set differs                                                                                 |
| Compartment              | Dimensions or kind differ                                                                         |
| Transition               | From, to, rate identity, or draw method differ                                                    |
| Parameter                | State (fixed, estimated, required), fixed value, prior, bounds, kind or declared dimension differ |
| Initial condition        | Expression identity differs                                                                       |
| Observation stream       | Family, mean identity, family parameters or schedule differ                                       |
| Time input, intervention | Definition identity differs                                                                       |
| Simulation               | Backend, step, time unit or origin differ                                                         |
| Quantity                 | Definition identity differs (reporting layer; shown separately)                                   |
| Fit configuration        | Per the existing `ConfigDiff`                                                                     |

Fixed-versus-estimated gets its own prominence. "Model A estimates the
incubation period; model B fixes it at 9 days" is an assumption difference a
reader must not miss, and it is invisible in a flow diagram.

### Roll-ups

Adding a stratifying dimension changes every compartment and most transitions.
Reported row by row, that buries the one fact that matters. The report therefore
carries roll-ups: "dimension `age` added; 6 compartments and 9 transitions
restratified, otherwise unchanged". A roll-up is emitted only when every
affected entity differs in that one respect and nothing else; the underlying
rows stay available.

### More than two models

- **Pairwise against a baseline**, matching how scores are shown.
- **Shared by all**: every entity whose status is `Same` across all members.
  This list is the "assumptions a comparison cannot test" required by principle
  R9, and it is an output of the diff, not a separate computation.
- **A matrix** of counts by category for N members, for the overview.

### The report

```rust
pub struct DiffReport {
    pub schema_version: u32,
    pub left:  ModelRef,   // name, model identity, bundle hash
    pub right: ModelRef,
    pub identical: bool,                 // agrees with model_ir_hash
    pub reporting_only: bool,            // ModelAgreement::ReportingDiffers
    pub rows:    Vec<Row>,               // category, name(s), matched_by, status
    pub rollups: Vec<Rollup>,
    pub hints:   Vec<Hint>,
    pub config:  ConfigDiff,             // existing type
    pub counts:  BTreeMap<Category, Counts>,
}
```

Everything the mockup shows is a projection of this:

| Mockup element                        | Source                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Summary tiles ("1 changed of 6")      | `counts`                                                                       |
| Highlighted arrow in the flow diagram | `rows` where category is transition and status is not `Same`, by transition id |
| Side-by-side detail table             | `rows`, with `ExprView.katex` for display                                      |
| "Same prior" beside two posteriors    | parameter row status; posteriors come from the bundles, not the diff           |
| "View source diff"                    | the two source files; a convenience, labelled as text                          |
| "Shared by all models"                | the N-way shared list                                                          |

The page's headline sentence is fixed wording over `counts` and `rollups` ("B is
A with one changed rate and one added parameter"). An author may annotate it;
the viewer does not generate interpretation.

### Where the code lives

- The structure summary is emitted by the compiler, because that is where the
  pre-expansion model exists. It is written into the run directory beside
  `model.graph.json` and copied into the bundle.
- The diff is a small Rust crate with no dependency beyond the summary types and
  serde, so it compiles to WASM beside the scoring core (proposal 3). The CLI
  and the viewer call the same function.
- `ConfigDiff` moves to, or is re-exported from, that crate so the report can
  contain it. (_To verify_: it currently takes `FitConfig` and `FitView` from
  the `cli` crate.)

### Command

```
camdl diff A B [--rename FILE] [--format table|md|json]
```

`A` and `B` are fit handles, bundle paths or model files. Exit status 0 when
identical, 1 when different, 2 on refusal (invalid rename map, unreadable
input). `fit table`'s `model_changed` column gains a pointer to this command.

## Invariants

Each is a test.

1. `diff(a, a)` has no rows other than `Same`, and `identical` is true.
2. Swapping the arguments swaps `OnlyLeft` and `OnlyRight` and nothing else.
3. `identical` is true exactly when `model_ir_hash(a) == model_ir_hash(b)`.
4. `reporting_only` is true exactly when `compare_to_fitted` returns
   `ReportingDiffers`.
5. Every declared entity of each model appears in exactly one row.
6. Applying a rename map and then diffing equals diffing the renamed model.
7. A roll-up never hides a row that differs in more than the rolled-up respect.
8. The report is byte-identical across runs and thread counts.

## What it does not do

- Decide whether two differently written models are mathematically equivalent.
- Detect reparameterizations.
- Match entities across naming schemes without a declared map.
- Say which model is better. That is the score.
- Diff posteriors or predictions. Those are shown beside the diff, from the
  bundles.

## Tests and fixtures

A corpus of model pairs with known answers, drawn where possible from
`ocaml/golden/`: add a parameter; drop a compartment; change one rate; change a
prior family; fix a previously estimated parameter; change an observation family
from Poisson to negative binomial; add a stratifying dimension; reorder
declarations (expect empty); rename a local binding (expect empty); rename a
compartment with and without a map.

The Bundibugyo variants are the first real fixture set, and the source of the
viewer's first published comparison.

## Steps

1. Settle the items under
   [To verify before starting](#to-verify-before-starting).
2. Add leaf-to-declaration provenance to expanded transitions if absent.
3. Emit the structure summary from the compiler, with `ExprView` identities.
   Golden files for every model in `ocaml/golden/`.
4. The diff crate: matching, classification, invariants 1 to 5 and 8.
5. Roll-ups and the N-way shared list, invariant 7.
6. Rename maps and hints, invariant 6.
7. `camdl diff` and the three renderers, all from `DiffReport`.
8. WASM build; the viewer's structure screen reads `DiffReport` from fixtures.

Steps 1 to 4 do not depend on the compare cleanup branch.

## To verify before starting

1. Does an expanded transition record the declared transition it came from
   (`TransitionMetadata`, name convention, or neither)?
2. What exactly does `Model`'s hashing exclude? The `fitted_model.rs` module doc
   lists `quantities`, `contrasts`, the name and gradient maps;
   `normalize_for_hash` clears two more fields. Confirm the full list, because
   invariant 3 depends on it.
3. Are local `let` bindings inlined, kept as `bindings`, or both in the compiled
   model, and does renaming one change `model_ir_hash`?
4. Is there a canonical expression form or simplifier in the compiler today (for
   autodiff or common-subexpression work) that option A could reuse?
5. Can `ConfigDiff` be built without the `cli` crate's `FitView`?
6. Does `model.graph.json` already have consumers that would break if the
   structure summary replaced it, or should the graph become a projection of the
   summary?
7. How are observation families and their parameters represented before
   expansion, and do indexed streams share one declaration?

## Open questions

1. **Initial conditions and data bindings.** Are they part of "structure" for
   the reader, or a separate section of the page?
2. **Interventions and scenarios.** A fit under scenario A and one under
   scenario B share a model. Is that a structure difference or a fit difference?
3. **How loud is a declared rename?** Marked on every matched row, or once at
   the top with a list?
4. **Hints.** Worth shipping in the first version, or deferred until real
   cross-group comparisons exist?
5. **Three or more models on one screen.** Pairwise against the baseline with a
   picker, as mocked, or a matrix view first?
