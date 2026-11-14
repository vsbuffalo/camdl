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
    pub declared: DeclaredSection,
}

pub enum DeclaredSection {
    NotRecorded,                       // the member is absent
    Recorded(Box<DeclaredModel>),
    UnsupportedVersion {               // a schema_version this build cannot read
        schema_version: SchemaVersion,
        raw: Box<RawValue>,            // the member's bytes, kept as read
    },
}

pub struct DeclaredModel {
    pub schema_version: u32,               // its own; starts at 1
    pub compile_inputs: CompileInputs,     // --set values; digests of files read
    pub time_unit: Option<UnitLit>,
    pub description: Option<String>,
    pub origin: Option<String>,
    // Every declaration below also carries `source_text`, its text as written.
    pub dimensions:    Vec<DimensionDecl>,   // name, level source, resolved levels
    pub compartments:  Vec<CompartmentDecl>, // name, kind, dims, expands_to
    pub stratify:      Vec<StratifyDecl>,
    pub parameters:    Vec<ParameterDecl>,   // name, dims, kind, dim, bounds, prior
    pub definitions:   Vec<DefinitionDecl>,  // every `let`, constants included
    pub tables:        Vec<TableDecl>,
    pub time_inputs:   Vec<TimeInputDecl>,
    pub timepoints:    Vec<TimepointDecl>,
    pub transitions:   Vec<TransitionDecl>,  // name, from, to, rate or via, guard,
                                             // expands_to
    pub ode_equations: Vec<OdeDecl>,
    pub balance:       Vec<BalanceDecl>,
    pub initial:       Vec<InitialDecl>,     // compartment, expression or law
    pub observations:  Vec<ObservationDecl>, // source, columns, measurement,
                                             // projection, schedule, covers,
                                             // expands_to
    pub interventions: Vec<InterventionDecl>,
    pub events:        Vec<InterventionDecl>,
    pub reactive_interventions: Vec<ReactiveDecl>,
    pub scenarios:     Vec<ScenarioDecl>,
    pub simulation:    Option<SimulationDecl>, // from, to, dt, integrator, tolerances
    pub output:        Option<OutputDecl>,
    pub quantities:    Vec<QuantityDecl>,
    pub contrasts:     Vec<ContrastDecl>,
}

pub struct DeclaredExpr {
    pub tree:  ExprTree,     // the parsed form; what sameness is decided on
    pub katex: String,       // the formula rendering
    pub katex_loss: Vec<KatexLoss>, // why `katex` does not show the whole tree
    pub reads: Vec<NameRef>, // the names it reads, each with its kind
}
```

The types are the `declared` crate, `rust/crates/declared/src/lib.rs`, which is
the authority; the listing above is its top level. The crate is a leaf: types
and serde only, with no dependency on `ir`, `runid`, `sim` or `cli`. `ir`
depends on it for the envelope field and for `ParamKind`, which is defined there
once and re-exported. Every type refuses unknown fields, because a reader that
ignored a field it did not know could report two models as the same when they
differ in it; adding a field is a `schema_version` step.

Each list is in declaration order. `ExprTree` is the compiler's parsed
expression type (`expr`, `ocaml/lib/compiler/ast.ml:77-105`) with source
locations removed and each name tagged with the kind it resolved to. The types
follow the parsed declarations, which differ from the compiled model in four
ways that shape the rows below:

- A parameter declaration holds name, kind, dimension, bounds and prior
  (`ast.ml:153-155`). Whether it is fixed or estimated is not declared; the
  compiled model's `ParamValue` state is derived by the compiler.
- There is no constant declaration. A constant is a `let` with a kind annotation
  (`ast.ml:292-305`), so it is a definition.
- A transition has no draw-method field (`ast.ml:263-280`). Overdispersion is a
  function call inside the rate and is part of the rate tree.
- The `simulate` block holds from, to, step, integrator and tolerances
  (`ast.ml:601-609`). Time unit, origin and description are top-level
  declarations. There is no seed: the compiler writes `rng_seed = None` at both
  places it builds the simulation settings
  (`ocaml/lib/compiler/expander.ml:6686`, `:6784`) and no compiler flag sets it,
  so the seed is a run input and never a model difference.

**`reads` and every name's kind come from the compiler's own name resolution**,
not from a second classification written for this section: the emitter
(`ocaml/lib/compiler/declared_emit.ml`) asks `Expander.classify_name` and
`Expander.classify_indexed_name`, the chain `resolve_expr` itself matches on.
What that chain cannot answer is settled by position, and each rule names the
compiler code that reads the name that way:

| Kind                                                                                      | A name is this when                                                                                                                                                                                |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compartment`, `parameter`, `definition`, `table`, `time_input`, `builtin`, `data_column` | the classifier says so; plus a compartment a `via hyper_erlang` lowering removed from the working context, a parameter family named without its index, and `t_start`/`t_end`                       |
| `index_var`                                                                               | it is bound by the declaration's index binders or an enclosing `sum`; checked first, as the compiler substitutes binders before resolving                                                          |
| `level`                                                                                   | it sits in an index position and is a level of a declared dimension                                                                                                                                |
| `quantity`, `stream`, `transition`                                                        | it is named inside a quantity or contrast body, a projection or a reactive trigger, and is declared as one (a transition also by one expanded cell, `recovery_p1`)                                 |
| `dimension`                                                                               | it is a forcing argument that names a declared dimension                                                                                                                                           |
| `obs_anchor`                                                                              | it is `first_obs` or `last_obs`                                                                                                                                                                    |
| `keyword`                                                                                 | the enclosing form reads the word as text: the `dim` of `unchecked_dim(...)`, the `label` of a `hyper_erlang` branch, `once = true`, a forcing argument selecting from a closed set (`link = log`) |
| `unresolved`                                                                              | no rule applies                                                                                                                                                                                    |

A quoted string (`date("2020-01-01")`, a file path) is a `Str` node and not a
name: the parser gives it no source location, which is how the emitter tells it
from an identifier. `reads` is every name in the tree, once each, in order of
first appearance, index variables and levels included.

Tests check, over every compiled golden, that no name is `unresolved`, that each
name's kind names something the section itself declares, that no name has two
declaring kinds, and that `reads` is the tree's names
(`ocaml/test/test_declared_emit.ml`,
`rust/crates/declared/tests/golden_sections.rs`). A name outside the goldens
that no rule covers is tagged `unresolved` in the section and does not fail the
compile. It is never silent:

- the compiler emits warning `W203`, naming the declaration and the name, and
  saying that `camdl diff` will report that declaration's dependencies as
  unknown. A valid model can trigger it today: a `timepoints` entry that refers
  to another timepoint, because the compiler discards that block (gh#1047);
- the diff reports any row whose tree reads an `unresolved` name as
  "dependencies unknown". It never gives such a row an empty `affected_by`,
  which would say that nothing it reads changed;
- `camdl publish --check` refuses or marks a model whose declared section holds
  one (a requirement on the bundle proposal).

`affected_by` follows only the names that can be a dependency. An `index_var`, a
`level` and a `keyword` are not: the kind on each name is what lets the diff
leave them out.

KaTeX strings come from the existing renderer (`Latex.tex`) and so use a
declaration's `@symbol` doc tag where one is given. A changed `@symbol` changes
`katex` and `katex_loss` and nothing else (tested: two sources differing only in
doc comments give byte-equal compiled-model JSON). Sameness is never decided on
`katex`.

**The formula rendering is not a faithful picture of the tree, and the section
says where.** `Latex.tex` was written to draw rates for a document, and it drops
or regroups parts of an expression. Each declared expression therefore carries
`katex_loss`, a list of the ways its rendering falls short, empty when the
rendering can be read back as the tree:

| `katex_loss` entry        | What the rendering does                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------- |
| `unit_dropped`            | `0.5 'per_day` is rendered `0.5`                                                               |
| `number_rounded`          | the rendered digits do not read back as the number (six significant digits are kept)           |
| `placeholder`             | a list, a range, `observations.x` or a run member is rendered as a dot                         |
| `grouping_changed`        | a product or quotient is regrouped into one fraction, or an operand lacks parentheses it needs |
| `sum_range_dropped`       | a `sum` is rendered without its dimension and its `where` guard                                |
| `argument_names_dropped`  | `f(rate = x)` is rendered `f(x)`                                                               |
| `index_dimension_dropped` | `S[age = a]` is rendered as `S[a]` is                                                          |
| `string_as_name`          | a quoted string is rendered as a name is                                                       |
| `symbol_override`         | a name is rendered as its `@symbol`                                                            |

The list is computed by `Latex.tex_loss`, which sits beside the renderer and
walks an expression the same way. It errs towards reporting:
`beta * S * (I /
N)` is listed as regrouped although a careful reader would read
its fraction correctly. On the 99 compiled goldens, 456 of 2577 expressions have
an entry (314 for a dropped unit), and 69 of 296 transition rates.

**Each declaration carries its text as written**, `source_text`: the bytes of
the declaration from its first token to its last, read back from the source by a
span the parser records on every declaration. A `#'` doc block above it and a
trailing comment are outside it; `let`, a block keyword and a `#[lineage]`
attribute are inside it. A quantity declared in a separate `--quantities` file
takes its text from that file. The text is per declaration, not per expression:
a branch weight or a prior argument has no text of its own, and the diff shows
the whole declaration. `null` means the compiler had no text; no compiled golden
holds one.

The text is for display only and is never compared: the parser reads different
spellings as one tree (`×` and `*`), so equal declarations can have different
text. Three rules for the diff follow:

- Where a shown expression has a non-empty `katex_loss`, the diff shows the
  declaration's `source_text` beside the formula. It never shows a placeholder
  or a number stripped of its unit as the whole account of a row.
- A row that is `Changed` while both sides render to the same `katex` string is
  flagged, and shown by `source_text`. This also covers the one collision
  `katex_loss` does not record: the styling of names, where `Iv` and `I_v` are
  both rendered `I_v`.
- Reformatting a model's source changes `source_text`, and so the compiled file,
  and no row: the row status and the model hash do not read it.

**Agreement with what already describes the model.** Two older partial
descriptions stay as they are: `ModelStructure` inside the compiled model
(`rust/crates/ir/src/model.rs:379-386`), which is hashed, and the compiler's
flow-graph types (`ocaml/lib/compiler/latex.ml:541-553`), which build
`model.graph.json`. A test checks, on every golden, that each agrees with the
declared section: the same dimensions and levels, the same base compartments and
their dimensions, and one graph edge per source and destination pair of each
declared transition. The doc-comment dictionary `docs` also stays where it is;
folding it into this section is a later change. A changed doc comment is a
reporting-layer row and never a model difference.

Each declared transition, compartment and observation stream also carries
`expands_to`, the expanded names it produced. Nothing else links the expanded
model back to a declaration (see [Settled facts](#settled-facts)), and the
viewer and the assumption ledger (proposal 5) need the link to map per-stratum
outputs back to declarations. Row status never depends on it.

The names are recorded where the expander makes them and are never recovered
from the spelling of a name: the staged-residence (`via`) lowering records each
transition it synthesises and each stage compartment it creates against the
declaration it stands for, and transition and observation expansion record the
compiled names each declaration produced. A test checks, on every compiled
golden and in both languages, that the lists partition the compiled model: the
`expands_to` of all declared compartments together are exactly the compiled
compartments, each claimed once, and the same for transitions and observation
streams. In `ocaml/golden/ebola_outcome_hyper`, compartment `I` expands to six
stage compartments (`I__fatal__1` to `I__recover__3`) and transition `outcome`
to six stage and exit transitions.

The field stays optional on the wire. `null` means not recorded; an empty list
means the declaration produced nothing (a transition whose guard excludes every
combination).

**The section is readable without the runtime's reader for compiled models.**
That reader refuses any envelope whose IR version is not its own
(`envelope.rs:126-133`), and a published bundle is read for as long as it is
cited. So `DeclaredModel` is a self-contained document: it deserializes through
the `declared` crate alone, is versioned by its own `schema_version` and never
by `ir_version`, refers to nothing in `Model`, and every published version of
its schema stays readable. `camdl publish` copies it out of the envelope into
the bundle as its own file, beside the model identity as a string, so neither
the viewer nor `camdl diff` on two bundles ever loads a compiled model. A test
reads a `declared` document from an envelope whose `ir_version` is not the
current one.

The section has three states to a reader, never two. Absent means not recorded.
Present at a `schema_version` the reader knows means recorded. Present at one it
does not know means "unsupported version N": the model still loads and runs, and
the section is not read. In the first and third states the diff reports itself
unavailable for that side, and says which. The section is never backfilled by
recompiling at archive time.

**A compiled model is copied as bytes with a digest, never re-serialized.** A
tool that rewrites an envelope writes the `declared` member back as the text it
read, in every state (`ir::reemit_with_model`). The general serializer for a
whole envelope was removed for this reason: a round trip through a generic JSON
value sorts object keys, and that changes `model_ir_hash` (pinned by
`rust/crates/runid/tests/value_round_trip.rs`).

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
changed definitions and time inputs it reads, directly or through other
definitions, computed from `reads`. A row that is `Same` with a non-empty
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

- Compartments, dimensions, levels, definitions, parameters, observation
  streams, scenarios, quantities: by name.
- Transitions: by declared name. A branching or multi-source transition fans out
  to several flow-graph edges with ids `name__src__dst` (`latex.ml:676-680`), so
  the viewer highlights by the declared name recovered from the id. The edge has
  no declared-name field of its own; adding one changes `model.graph.json`,
  which camdl-scope reads, and is left to the viewer proposal.
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

| Category             | Changed when                                                                           |
| -------------------- | -------------------------------------------------------------------------------------- |
| Dimension            | Level set differs                                                                      |
| Compartment          | Dimensions or kind differ                                                              |
| Transition           | From, to, rate tree, staging law or guard differ                                       |
| Definition           | Indices, kind or body tree differ (constants are definitions)                          |
| Parameter (declared) | Kind, dimension, bounds or prior, as the model file declares them, differ              |
| Initial condition    | Expression tree or law differs                                                         |
| Observation stream   | Family, argument trees, schedule, source label, columns, `covers` or projection differ |
| Time input, table    | Definition differs                                                                     |
| Intervention, event  | Definition differs                                                                     |
| Scenario             | A declared `scenarios {}` entry differs                                                |
| Simulation           | Start, end, step, integrator, tolerances, time unit or origin differ                   |
| Output               | Output times or the trajectory and observation flags differ                            |
| Balance, ODE         | Definition differs                                                                     |
| Declaration order    | The same entities of one category appear in a different order                          |
| Compile input        | A compile-time `--set` value or the digest of a file read at compile time differs      |

A file whose digest is absent (the compiler could not re-open it to hash it) is
reported as "not recorded", and the diff then states that compile inputs were
not checked for that file. Such a row is never `Same`.

`--set` values are applied by `camdlc` after the compile
(`ocaml/bin/camdlc.ml:257-284`), so they are added to the section there and are
not known to the library entry points.

Declaration order is a category, and not ignored, because the hash covers it:
reordering transitions, parameters, compartments or `init` entries changes
`model_ir_hash` (measured), and the order of transitions changes random-number
consumption. Moving a `let` does not change the hash and is not an order
difference.

**Reporting layer**: `quantities` and `contrasts`, which the hash excludes, and
doc comments, which sit outside the model. Shown separately.

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

| Fact               | Model file                                                     | Fit configuration                                                                                  | Which wins                                                                                         |
| ------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Fixed or estimated | Not declared; `ParamValue` (`parameter.rs:189-208`) is derived | `[estimate]` and `[fixed]`, which together must cover every parameter (`fit/config_v2.rs:115-119`) | The fit configuration always                                                                       |
| Fixed value        | A compile-time `--set` only                                    | `[fixed]`                                                                                          | The fit configuration                                                                              |
| Prior              | `~` in the model (`PriorSpec` inside `Estimated`)              | `[estimate.<p>.prior]` (`config_v2.rs:585-600`)                                                    | Fit configuration, then model, then flat only by explicit opt-in (`fit/priors_precedence.rs:4-42`) |
| Bounds             | `in [lo, hi]`                                                  | `[estimate.<p>].bounds`, which may tighten but not loosen (`config_v2.rs:575-583`)                 | Fit configuration when present, else model                                                         |
| Scenario           | `scenarios {}` definitions                                     | `scenario`, `enable`, `disable` (`config_v2.rs:128-134`)                                           | Definitions are model; the choice is fit                                                           |

The model identity a fit stores is the model as compiled, without the fit
configuration applied. So the declared row and the effective row can disagree: a
prior edited in the model file but overridden identically in both configurations
is `Changed` as declared although both fits used the same prior.

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
  expansion, so that names are resolved and `expands_to` is known. The Rust
  types are the `declared` crate; the OCaml types and the serializer are in
  `ocaml/lib/ir/`.
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
- **The section does not enter identity, and tests prove it.** `model_ir_hash`
  of every compiled golden was recorded before the change
  (`rust/crates/runid/tests/fixtures/golden_model_hashes.tsv`) and is unchanged
  after it, and after each later addition to the section.
- **Every stored run and fit gets a new address at 0.44, as at any IR bump.** A
  run id is derived from a model digest that folds in the IR version string and
  the engine string (`rust/crates/runid/src/inputs.rs:287-293`). So the bump
  re-keys every stored run, with or without this section. What the section
  itself changes is nothing: 30 run ids (five goldens, six kinds of run) were
  pinned before the change and are equal after it with those two strings held
  fixed (`rust/crates/cli/tests/fixtures/golden_run_ids.tsv`).
- **Old compiled models.** The runtime refuses any envelope whose `ir_version`
  is not its own (`envelope.rs:126-133`), so a model compiled before the bump is
  not read at all, with or without this section. `declared: None` therefore
  arises only for an envelope written at the current version without the
  compiler (hand-edited, or serialized from Rust). Both cases report the diff as
  unavailable for that side.
- **Landing.** The golden regeneration conflicts with any branch that touches
  `ocaml/lib/ir/`, `rust/crates/ir/src/` or a golden. The unmerged branches that
  do are the b-spline ones (`feat/bsplines`, `feat/bsplines-explicit`,
  `feat/bsplines-s3-local`, and `origin/feat/bsplines-s1`, which carries its own
  `ir/VERSION` change) and `nuts-divergence-tests`. Their last commits are dated
  2026-10-27 to 2026-10-29, none has a pull request, and the b-spline proposal
  keeps `feat/bsplines` off main until its measurement criteria are met. So this
  change lands first, and the b-spline work, when it lands, rebases, regenerates
  goldens, and adds its declaration kinds to the declared section as a
  `schema_version` step.
- **gh#1046 is a separate change and moves no golden.** None of the 62 files in
  `ocaml/golden/` contains a hoisted subtree (no `__licm_` name appears in any),
  so its fix leaves the golden set untouched. It also means no golden exercises
  the hoisting pass; the fix adds one. The fix is made in the hash, not in the
  compiler: in the reproduction, two of the three hoisted pieces are referenced
  only by gradient maps, which the hash excludes, so renaming in the compiler
  could not have fixed it. It will raise `HASH_VERSION` from 2 to 3, which
  re-keys every stored fit once; it is to land in the same window as 0.44 so
  that stored fits re-key once and not twice.

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
9. Compiling any golden yields the same `model_ir_hash` with and without the
   `declared` section, and the same run ids once the IR version and engine
   strings, which a run id folds in, are held fixed.
10. On every golden, each name in every declared tree resolves to exactly one
    kind, and `reads` is the set the compiler's name resolution reports.
11. On every golden, `ModelStructure` and the flow graph agree with the declared
    section.
12. On every golden, the `expands_to` lists of the declared compartments,
    transitions and observation streams each partition the compiled model's
    names.
13. On every golden, every declaration has a `source_text`, and a doc comment, a
    comment or a change of layout changes no tree.

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
| Reformat; add a `#'` doc comment; move a `let`          | All `Same` (`source_text` and `katex` may differ)       | Identical          |
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
   file compared with its own fit can land in the refused fourth outcome. Not
   done; failing tests are on the branch `wip/gh1046-red-tests`.
2. The `declared` envelope section: the `declared` crate, the OCaml types and
   emission, the envelope field, `ir/VERSION` 0.44, goldens, `source_text`,
   `katex_loss`, `expands_to`, warning `W203`, invariants 9 to 13, and
   compiler-side tests for the parse facts. Done.
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
contain underscores (measured on `ocaml/golden/seir_erlang_via`). The declared
section's `expands_to` is the recorded link.

**The flow graph draws a staged transition with branches as an outflow to
nowhere.** In `ocaml/golden/ebola_outcome_hyper` the transition out of `I` names
its two destinations inside the `branch(...)` arguments of its
`via hyper_erlang` law. `Latex.transition_graph` does not read those, so
`model.graph.json` holds one arrow from `I` to nothing, drawn as a death is.
This is a defect of the existing diagram and not of the declared section, which
holds the law's arguments; the agreement test pins the current drawing and fails
when it is fixed.

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
