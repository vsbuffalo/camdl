# Proposal: a typed structural diff between models (`camdl diff`)

**Status:** Proposal **Date:** 2026-11-13 **Motivation:** camdl can say that two
models differ, as a boolean or a pair of hashes, but not how; the published
comparison viewer has to show what each model assumes that the others do not,
and what they all assume alike, from a typed report and not from a text diff or
a hand-written sentence.

**Drafted with Claude** from the design conversation of 2026-11-13. Code
references are to `c1f27d9d` (2026-11-13). Claims marked _verified_ were read in
that tree, and hashes marked _measured_ were computed there; the evidence is
collected under [Settled facts](#settled-facts). Two questions remain, under
[Decisions pending](#decisions-pending).

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
  fields before hashing (`normalize_for_hash`, `inputs.rs:332-337`); the module
  doc of `fitted_model.rs` says the identity also excludes `quantities`,
  `contrasts`, the model's file-stem name and the gradient maps. The hash itself
  is implemented in `rust/crates/runid/src/ir_hash.rs:1490-1548`.
- **A typed diff of fit configuration.** `ConfigDiff` covers estimated and fixed
  parameter sets, fixed values, bounds, priors, data hashes and method settings.
  Its only statement about the model is `model_changed: bool`
  (`rust/crates/cli/src/fit/config_diff.rs:30-61`). `camdl fit diff` prints it
  (`rust/crates/cli/src/fit/mod.rs:2879-2890`).
- **A pre-expansion flow graph, in the compiler.** The `camdlc render` command,
  with its graph format, builds `model.graph.json` from the declarations: base
  compartments as nodes, dimensions as plates, one edge per source and
  destination pair of each declared transition, with a KaTeX rate string and the
  aggregates each edge reads (`ocaml/lib/compiler/latex.ml:533-560`,
  `build_graph` at `:704`). It works on parsed declarations and never expands
  the model.
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
3. When the model belongs to a fit, the fit's config facts record (see
   [Where the code lives](#where-the-code-lives)).

The diff is a pure function of these. It reads no files and consults no store,
so it can run in a browser on two bundles.

### The structure summary

A pre-expansion projection, emitted by the compiler from the same compile that
produces the model. `model.graph.json` becomes a projection of it. Every
declared transition, compartment, observation stream and definition carries
`expands_to`, the list of expanded names it produced; this is the only link from
the expanded model back to what the author wrote. Sketch of the types (names are
proposals):

```rust
pub struct StructureSummary {
    pub schema_version: u32,
    pub dimensions:   Vec<Dimension>,        // name, levels
    pub compartments: Vec<CompartmentDecl>,  // name, dims, kind
    pub transitions:  Vec<TransitionDecl>,   // name, from, to, rate, reads, expands_to
    pub definitions:  Vec<DefinitionDecl>,   // hoisted lets: name, body
    pub parameters:   Vec<ParameterDecl>,    // name, state, prior, bounds, kind, dim
    pub constants:    Vec<ConstantDecl>,
    pub time_inputs:  Vec<TimeInputDecl>,    // time functions, tables, splines
    pub interventions: Vec<InterventionDecl>, // interventions and events
    pub scenarios:    Vec<ScenarioDecl>,
    pub initial:      Vec<InitialDecl>,      // compartment, expression
    pub observations: Vec<ObservationDecl>,  // stream, family, mean, params, schedule,
                                             // source, columns, covers, projection
    pub simulation:   SimulationDecl,        // dt, time unit, origin, start, end,
                                             // seed, integrator
    pub output:       OutputDecl,
    pub quantities:   Vec<QuantityDecl>,
    pub contrasts:    Vec<ContrastDecl>,
}
```

Every expression appears as an `ExprView` (defined in the next section): hashes
for comparison, a rendering for display, and the names it reads (which the
assumption ledger, proposal 5, also needs). The compiler writes the rendering
and the names; the hashes are filled in afterwards, in Rust.

### Deciding that two expressions are the same

This is the part most likely to go wrong. The rule: **a row's status never
contradicts model identity** (invariants 3a to 3c). Whether two models are
identical is decided by `model_ir_hash` and by nothing else; rows explain a
difference, they do not decide it.

`ExprView` carries two hashes with different authority:

```rust
pub struct ExprView {
    pub identity: ContentHash, // authoritative: decides Same / Changed
    pub written:  ContentHash, // non-authoritative: roll-ups only
    pub katex:    String,
    pub reads:    Vec<Name>,
}
```

**`identity` is the hash of the expanded, reference-resolved leaves.** Take the
expanded entities one declaration produced, and hash each leaf expression with
the runtime's expression hash (`rust/crates/runid/src/ir_hash.rs:202`) after
replacing every `BindingRef` (a reference to a hoisted `let`) and every
`PerEvalRef` (a reference to a subtree hoisted by loop-invariant code motion)
with the body it names, recursively. Hash the sorted set of leaf hashes.

The substitution is required, not a refinement. A compiled rate holds these
references by name, so hashing the rate as stored is wrong in both directions.
Measured on a three-compartment SIR model at `c1f27d9d`:

- Changing `let N = S + I + R` to `let N = S + I` leaves the stored rate hash of
  `infection @ beta * S * (I / N)` at `aa9bfb2ff527` in both models. Only the
  binding body moves (`9d26d57535ed` to `7da4c392144f`). Without substitution
  the transition row reads `Same` although the force of infection changed.
- With `recovery @ gamma * exp(-kappa) * I`, adding an unrelated transition
  earlier in the file moves the stored `recovery` rate hash from `eb8a0ad27024`
  to `cd69a0d7f98f`. The hoisted subtree is unchanged (`abfa2b6a3fe4`) but is
  renumbered from `__licm_0` to `__licm_4`, because the hoist counter runs over
  the whole model (`ocaml/lib/ir/licm.ml:207-222`). Without substitution an
  unrelated edit marks the row `Changed`.

Identities are computed in Rust from `ir::Model`, with the existing hash code.
The compiler does not hash: its only hash is a file digest
(`ocaml/lib/compiler/sha256.ml`), and a second implementation of the model hash
in OCaml would be a second definition of sameness.

**`written` is a hash of the pre-expansion expression as the author wrote it.**
It exists because leaf hashes cannot answer "restratified, otherwise unchanged":
adding a dimension replaces every leaf. It is used only by roll-ups, is typed as
non-authoritative, and never sets a row's status.

A hoisted `let` is something the author wrote, so it is also its own row
(category Definition), with its body's identity.

What is deliberately _not_ recognized as the same: algebraic rearrangement, and
reparameterization (`β` against `R0·γ`). Those show as changed, and the page
says the diff is about what is written.

### Matching entities

Entities are matched **by category and name, exactly.** No fuzzy matching and no
structural guessing.

- Compartments, dimensions, levels, parameters, constants, observation streams,
  quantities: by name.
- Transitions: by declared name. A branching or multi-source transition fans out
  to several flow-graph edges with ids `name__src__dst`
  (`ocaml/lib/compiler/latex.ml:676-680`), so each edge in the summary carries
  the declared name it belongs to, and the viewer highlights by that.
- Unmatched entities are reported as only-left or only-right.

For models from different groups, names will differ. A published comparison may
declare a **rename map** per member:

```toml
[members.seir_simple.rename.compartments]
Inf = "I"
[members.seir_simple.rename.streams]
cases_weekly = "confirmed_cases"
```

The map must be a bijection onto names that exist; otherwise the diff refuses. A
match made through a map is a claim by the comparison's author, and is shown as
one in two places: `MatchedBy::DeclaredRename` on every such row, so the mark
survives a row or figure being exported alone, and one list of all declared
renames at the top of the report.

Compartment and parameter names are hashed inside the expressions that read them
(`ir_hash.rs:221-229`), so a precomputed identity cannot be compared across a
rename. Under a rename map the diff re-hashes the renamed side from its
expression trees with the substitution applied. The diff therefore takes the
compiled models as input, not only their summaries.

Suggested matches ("`Inf` and `I` have identical stoichiometry; possibly the
same compartment") are not in this proposal. They need the same re-hashing under
a candidate substitution, and no cross-group comparison exists yet to design
them against. `DiffReport` schema version 1 has no field for them.

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

Rows are in three layers, and a row says which it belongs to.

**Identity rows** describe content `model_ir_hash` covers. They take part in
invariants 3a to 3c.

| Category             | Changed when                                                                                                               |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Dimension            | Level set differs                                                                                                          |
| Compartment          | Dimensions or kind differ                                                                                                  |
| Transition           | From, to, rate identity, or draw method differ                                                                             |
| Definition           | Body identity of a hoisted `let` differs                                                                                   |
| Constant             | Value differs                                                                                                              |
| Parameter (declared) | State (fixed, estimated, required), fixed value, prior, bounds, kind or dimension, as the model file declares them, differ |
| Initial condition    | Expression identity differs                                                                                                |
| Observation stream   | Family, mean identity, family parameters, schedule, source label, columns, `covers` or projection differ                   |
| Time input, table    | Definition identity differs                                                                                                |
| Intervention, event  | Definition identity differs                                                                                                |
| Scenario             | A declared `scenarios {}` entry differs                                                                                    |
| Simulation           | Step, time unit, origin, start, end, seed or integrator differ                                                             |
| Output               | Output times or the trajectory and observation flags differ                                                                |
| Declaration order    | The same entities appear in a different order                                                                              |
| Other                | Any other hashed field differs; the row names the field                                                                    |

Declaration order is its own category because the hash covers it: reordering
transitions, parameters, compartments or `init` entries changes `model_ir_hash`
(measured; see [Settled facts](#settled-facts)), and the order of transitions
changes random-number consumption.

The Other row is what makes invariant 3b hold. It is produced by an exhaustive
destructure of `ir::Model`, in the manner of the hash implementation at
`ir_hash.rs:1494`, so a field added to the model fails the build until someone
assigns it a category or leaves it to Other. Fields it covers today include
`ode_equations`, `balance`, `description`, `version`, transition `lineage` and
`identity_tracked_compartments`.

**Reporting rows** describe `quantities` and `contrasts`, which the hash
excludes. They are shown separately and are what `reporting_only` summarizes.

**Fit rows** describe facts that belong to a fit and not to a model. They are
outside invariants 3a to 3c.

| Category              | Changed when                                                                            |
| --------------------- | --------------------------------------------------------------------------------------- |
| Parameter (effective) | State, fixed value, prior or bounds differ after the fit configuration is applied       |
| Backend               | The fits ran on different backends (the model has no backend field, `model.rs:290-319`) |
| Applied scenario      | The fits ran under different scenarios                                                  |
| Data                  | A data file's content hash differs                                                      |
| Method settings       | Per the existing `ConfigDiff`                                                           |

A parameter's state is recorded twice: in the compiled model, from the model
file (`rust/crates/ir/src/parameter.rs:189-208`), and in the fit configuration's
`[estimate]` and `[fixed]` tables, which override it
(`rust/crates/cli/src/fit/config_diff.rs:184-262`). The model identity a fit
stores is the model as compiled, without `[fixed]` applied. So the two can
disagree in both directions: a parameter one fit fixes through `[fixed]` reads
`Same` in the declared row, and a prior edited in the model file but overridden
identically in both configurations reads `Changed` there although both fits used
the same prior.

The effective row is the one given prominence. "Model A estimates the incubation
period; model B fixes it at 9 days" is an assumption difference a reader must
not miss, it is invisible in a flow diagram, and it is usually a `[fixed]`
entry. The "same prior" label beside two posteriors reads the effective row.
When only model files are compared there is no fit configuration, and the
effective rows are absent, not copied from the declared ones.

### Roll-ups

Adding a stratifying dimension changes every compartment and most transitions.
Reported row by row, that buries the one fact that matters. The report therefore
carries roll-ups: "dimension `age` added; 6 compartments and 9 transitions
restratified, otherwise unchanged". A roll-up is emitted only when every
affected entity differs in that one respect and nothing else; the underlying
rows stay available.

"Otherwise unchanged" is decided on `ExprView.written`, the hash of the
expression before expansion, because the expanded leaves of a restratified
transition are all different. A roll-up is therefore a statement about what is
written and is labelled so; the rows beneath it keep their authoritative status.

### More than two models

- **Pairwise against a baseline**, matching how scores are shown.
- **Shared by all**: every entity whose status is `Same` across all members.
  This list is the "assumptions a comparison cannot test" required by principle
  R9, and it is an output of the diff, not a separate computation.
- **A matrix** of counts by category for N members, for the overview.

All three are projections of the same pairwise reports. Which of them a screen
leads with is a layout decision for the viewer proposal, not this one.

### The report

```rust
pub struct DiffReport {
    pub schema_version: u32,
    pub left:  ModelRef,   // name, model identity, bundle hash
    pub right: ModelRef,
    pub identical: bool,                 // model_ir_hash(left) == model_ir_hash(right)
    pub reporting_only: bool,            // compare_to_fitted == ReportingDiffers
    pub renames: Vec<DeclaredRename>,    // every rename-map entry applied
    pub rows:    Vec<Row>,               // layer, category, name(s), matched_by, status
    pub rollups: Vec<Rollup>,
    pub config:  Option<ConfigDiff>,     // None when a side is a bare model file
    pub counts:  BTreeMap<Category, Counts>,
}
```

`identical` and `reporting_only` are computed from the two compiled models, by
`model_ir_hash` and `compare_to_fitted` (`fitted_model.rs:49-60`), and are never
derived from rows. `compare_to_fitted` compares `quantities` and `contrasts` as
ordered lists, so two models whose quantities are reordered are `reporting_only`
with every reporting row `Same`; the Declaration order category has a
reporting-layer counterpart for that case.

When one side is a model file and the other a fit, the file is first resolved
with `as_fitted` (observation anchors resolved to numbers, as the fit archived
its model), so the two hashes are of the same form. A model compiled with a
compile-time `--set` hashes differently from the same file without it
(measured), and the report says so through the Parameter or Constant row.

Everything the mockup shows is a projection of this:

| Mockup element                        | Source                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Summary tiles ("1 changed of 6")      | `counts`                                                                       |
| Highlighted arrow in the flow diagram | `rows` where category is transition and status is not `Same`, by declared name |
| Side-by-side detail table             | `rows`, with `ExprView.katex` for display                                      |
| "Same prior" beside two posteriors    | effective parameter row status; posteriors come from the bundles, not the diff |
| "View source diff"                    | the two source files; a convenience, labelled as text                          |
| "Shared by all models"                | the N-way shared list                                                          |

The page's headline sentence is fixed wording over `counts` and `rollups` ("B is
A with one changed rate and one added parameter"). An author may annotate it;
the viewer does not generate interpretation.

### Where the code lives

- The structure summary is emitted by the compiler, because that is where the
  pre-expansion model exists, from the compile path and after expansion, so that
  it can record `expands_to`. Failing to produce it is a hard error for anything
  published. Where it is written is the first of the
  [decisions pending](#decisions-pending).
- The diff is a Rust crate depending on the summary types, `serde`,
  `serde_json`, `ir` and `runid` (for the expression hash, and for re-hashing
  under a rename map). It reads no files and starts no threads, so it is meant
  to compile to WASM beside the scoring core (proposal 3); `runid` uses `libc`
  only under `cfg(unix)`. That build has not been attempted, and step 9 is where
  it is settled. The CLI and the viewer call the same function.
- `ConfigDiff` today derives `Serialize` only, holds `serde_json::Value`, and is
  built from `FitConfig`, `Method` and `PriorDist` in the `cli` crate, which
  parse TOML (`config_diff.rs:18-29`, `:152-185`), so it cannot move as it is.
  Each fit instead writes a serializable **config facts** record: the estimated
  set, the fixed map, bounds, canonical prior strings, data hashes and method
  settings. `ConfigDiff` becomes a pure function of two such records and lives
  in the diff crate; the `cli` crate builds the record.

### Command

```
camdl diff A B [--rename FILE] [--format table|md|json]
```

`A` and `B` are fit handles, bundle paths or model files. Exit status 0 when
identical, 1 when different, 2 on refusal (invalid rename map, unreadable
input). `fit table`'s `model_changed` column gains a pointer to this command.
The command's name and argument kinds are the second of the
[decisions pending](#decisions-pending).

## Invariants

Each is a test.

1. `diff(a, a)` has no rows other than `Same`, and `identical` is true.
2. Swapping the arguments swaps `OnlyLeft` and `OnlyRight` and nothing else.
3. Rows never contradict model identity:
   - 3a. `identical` is `model_ir_hash(a) == model_ir_hash(b)`, computed from
     the compiled models and not from rows.
   - 3b. If `identical` is false, at least one identity row is not `Same`.
   - 3c. If `identical` is true, every identity row is `Same`.
4. `reporting_only` is the result of `compare_to_fitted` being
   `ReportingDiffers`, computed from the compiled models and not from rows.
5. Every declared entity of each model appears in exactly one row per layer.
6. Applying a rename map and then diffing equals diffing the renamed model,
   where the renamed model is recompiled from renamed source.
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
prior family; fix a previously estimated parameter in the model file, and
separately through `[fixed]`; change an observation family from Poisson to
negative binomial; add a stratifying dimension; rename a compartment with and
without a map.

Pairs whose expected answer follows from how the hash behaves, each measured:

| Pair                                                          | Expected                                                                   |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Move a `let` above or below the transitions block             | Identical, no rows                                                         |
| Add a `#'` doc comment                                        | Identical, no rows                                                         |
| Rename a `let` that reads a parameter (inlined)               | Identical, no rows                                                         |
| Rename a parameter-free `let` (kept under its name)           | Different; Definition only-left and only-right                             |
| Change the body of a parameter-free `let`                     | Different; Definition `Changed`, and every transition reading it `Changed` |
| Reorder transitions, parameters, compartments or `init`       | Different; one Declaration order row, all else `Same`                      |
| Add an unrelated transition before one with a hoisted subtree | The later transition's row is `Same`                                       |
| Add a `scenarios {}` entry                                    | Different; Scenario only-right                                             |
| Change `simulate { to = … }`                                  | Different; Simulation `Changed`                                            |

The compiler's summary is tested by golden files for a named subset of about ten
models covering stratification, `via` staging, branching, indexed observations
and hoisted lets, not for all 62 models in `ocaml/golden/`: that set's target
writes only `*.ir.json` from compiler output (`Makefile:278-284`), and each file
is reviewed by a person. Golden summaries hold no hashes. Identities are tested
in Rust.

The Bundibugyo variants are the first real fixture set, and the source of the
viewer's first published comparison.

## Steps

1. Fix gh#1046 first: `model_ir_hash` differs between a full and a
   `--no-state-grad` compile of one source when a subtree is hoisted. Until it
   is fixed, a model file compared with its own fit can be reported different.
2. The compiler emits the structure summary after expansion, with no hashes.
   Each declared transition, compartment, observation stream and definition
   carries the list of expanded names it produced. Golden files for the named
   subset.
3. Rust fills `ExprView.identity` and `written` from `ir::Model` and the
   summary. Tests for the reference-resolved identity.
4. The config facts record, and `ConfigDiff` as a function of two of them.
5. The diff crate: matching, the three row layers, invariants 1 to 5 and 8.
6. Roll-ups and the N-way shared list, invariant 7.
7. Rename maps, invariant 6.
8. The command and the three renderers, all from `DiffReport`.
9. WASM build; the viewer's structure screen reads `DiffReport` from fixtures.

Steps 1 to 8 need no published bundle: their inputs are model files and fit
directories. Step 9, and copying the summary into a bundle, wait on the bundle
proposal (1 of the set) and the scoring-core proposal (3), neither of which is
written. None of the steps depends on the compare cleanup branch
(`worktree-peppy-jingling-kite`).

## Settled facts

What the design above rests on, each established at `c1f27d9d`.

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

**Lets are both inlined and kept.** A parameter-free `let` with no free index is
kept in `bindings` under its own name; a `let` that reaches a parameter is
inlined (`expander.ml:3511-3539`).

**No canonical form exists for pre-expansion expressions.** On expanded
expressions there are `Licm.canon` (bitwise key, no commutativity,
`licm.ml:114`), `normalize_expr` (population-sum collapse only,
`expander.ml:2576`) and `constant_fold.ml`. None alpha-renames. `written` is
therefore a hash of the expression tree as parsed, with no normalization.

**`model.graph.json` has consumers**: the archive writer
(`rust/crates/cli/src/fit/mod.rs:1448-1453`), an integration test
(`rust/crates/cli/tests/model_graph_render.rs`), `ocaml/test/test_latex.ml`, the
run spec (`docs/camdl-run-spec.md:409`) and the camdl-scope viewer. It stays,
and becomes a projection of the summary.

**Observation streams** are one `obs_decl` per stream family before expansion,
with an index list, a measurement, a projection, an optional emit schedule and
`covers` (`ocaml/lib/compiler/ast.ml:403-453`). After expansion each stratum is
its own observation model with a `stratum` key.

## Decisions pending

Two questions are not settled, and steps 2 and 8 wait on them.

1. **Where the structure summary is written.** Either a new section of the IR
   envelope beside `docs` (`rust/crates/ir/src/envelope.rs:84-100`), written by
   the same compile as the model, at the cost of an `ir/VERSION` bump and
   regenerated goldens; or a separate `model.structure.json` from a second
   compiler invocation at archive time, made a hard error for anything
   published. The file beside `model.graph.json` described under
   [Where the code lives](#where-the-code-lives) cannot serve as it is: it comes
   from `camdlc render`, which never expands the model, is skipped for a fit
   given a compiled `.ir.json` (`fit/mod.rs:1436`), and treats failure as "skip
   the archive" (`util.rs:1453`).
2. **How the command relates to `camdl fit diff`**, which exists
   (`fit/mod.rs:2879-2890`) and prints the configuration diff with a "Model:"
   section that says only changed or unchanged. Either `camdl diff` replaces it,
   or `camdl diff` takes model files and `camdl fit diff` takes fits and prints
   the full report.
