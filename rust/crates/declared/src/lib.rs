//! The declared model: a camdl model as its author wrote it, before expansion.
//!
//! These are the types of the `declared` section of the IR envelope
//! (`docs/dev/proposals/2026-11-13-model-structure-diff.md`). The compiler
//! writes the section from the same compile that writes the model; a model
//! diff reads two of them and decides, declaration by declaration, what
//! differs.
//!
//! The types mirror the compiler's parsed tree (`ocaml/lib/compiler/ast.ml`)
//! with two things removed and three added:
//!
//! - removed: source locations and `#'` doc comments. Neither is part of what
//!   a model says, so two models that differ only in them have equal values
//!   here. There is no parenthesis node either, because the parsed tree has
//!   none.
//! - added: on every name in an expression, what it resolved to
//!   ([`NameKind`]); on every top-level expression, its KaTeX rendering and
//!   the names it reads ([`DeclaredExpr`]); and, on a few declarations, facts
//!   the compiler resolves that the declaration alone does not state
//!   ([`DimensionDecl::levels`], [`CompartmentDecl::dims`], and `expands_to`).
//!
//! Sameness of two expressions is `==` on [`ExprTree`]. Nothing is normalized:
//! `beta * S * I` and `S * beta * I` are different trees.
//!
//! This crate depends on nothing in the workspace. A document of these types
//! is versioned by [`DeclaredModel::schema_version`] alone and is readable
//! without the reader for compiled models.

use serde::{Deserialize, Serialize};

/// The schema version this crate writes.
pub const SCHEMA_VERSION: u32 = 1;

// ── The document ────────────────────────────────────────────────────────────

/// One model, as declared. Every list is in declaration order; a block that
/// appears more than once in the source contributes its entries in order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredModel {
    pub schema_version: u32,
    pub compile_inputs: CompileInputs,
    /// `time_unit = 'days`.
    pub time_unit: Option<UnitLit>,
    /// `description = "..."`.
    pub description: Option<String>,
    /// `origin = "2024-01-01"`, as written.
    pub origin: Option<String>,
    pub dimensions: Vec<DimensionDecl>,
    pub compartments: Vec<CompartmentDecl>,
    pub stratify: Vec<StratifyDecl>,
    pub parameters: Vec<ParameterDecl>,
    pub definitions: Vec<DefinitionDecl>,
    pub tables: Vec<TableDecl>,
    pub time_inputs: Vec<TimeInputDecl>,
    pub timepoints: Vec<TimepointDecl>,
    pub transitions: Vec<TransitionDecl>,
    pub ode_equations: Vec<OdeDecl>,
    pub balance: Vec<BalanceDecl>,
    pub initial: Vec<InitialDecl>,
    pub observations: Vec<ObservationDecl>,
    pub interventions: Vec<InterventionDecl>,
    pub events: Vec<InterventionDecl>,
    pub reactive_interventions: Vec<ReactiveDecl>,
    pub scenarios: Vec<ScenarioDecl>,
    pub simulation: Option<SimulationDecl>,
    pub output: Option<OutputDecl>,
    pub quantities: Vec<QuantityDecl>,
    pub contrasts: Vec<ContrastDecl>,
}

/// What entered the compile from outside the model file. Two compiles of one
/// file that differ here can produce different models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileInputs {
    /// Compile-time `--set name=value` overrides, in the order given.
    pub set_values: Vec<SetValue>,
    /// Every file the compiler read (dimension levels, tables, time inputs),
    /// by the path the model names and the digest of its bytes.
    pub files: Vec<FileInput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetValue {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileInput {
    /// As written in the model, relative to the model file.
    pub path: String,
    /// Hex SHA-256 of the file's bytes.
    pub sha256: String,
}

// ── Expressions ─────────────────────────────────────────────────────────────

/// A top-level expression of a declaration: the parsed tree sameness is
/// decided on, the rendering that is shown, and the names it reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredExpr {
    pub tree: ExprTree,
    pub katex: String,
    /// Every name the tree reads, once each, in order of first appearance.
    pub reads: Vec<NameRef>,
}

/// The parsed expression (`expr` in `ast.ml`), without source locations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ExprTree {
    /// A bare number. `1` and `1.0` are the same constant.
    Const { value: f64 },
    /// A number with a unit literal: `7 'days`.
    Unit { value: f64, unit: UnitLit },
    /// A bare name.
    Ident { name: NameRef },
    /// An indexed name: `S[child]`, `S[age = child]`.
    Index { name: NameRef, indices: Vec<IndexItem> },
    BinOp { op: BinOp, lhs: Box<ExprTree>, rhs: Box<ExprTree> },
    UnOp { op: UnOp, arg: Box<ExprTree> },
    /// `sum(var in dim [where guard], body)`. The flat form with several
    /// binders is nested `Sum` nodes, as the parser produces it.
    Sum { var: String, dim: String, guard: Option<Guard>, body: Box<ExprTree> },
    /// `if pred then then_ else else_`.
    Cond {
        pred: Box<ExprTree>,
        #[serde(rename = "then")]
        then_: Box<ExprTree>,
        #[serde(rename = "else")]
        else_: Box<ExprTree>,
    },
    /// `fname(kw = v, ...)`.
    Call { function: String, args: Vec<Arg> },
    /// `[1.0, 2.0]`, possibly nested.
    List { items: Vec<ExprTree> },
    /// `7:100`, only inside a list.
    Range { lo: Box<ExprTree>, hi: Box<ExprTree> },
    /// `observations.<stream>`, only inside a quantity body.
    ObsAccess { stream: String },
    /// `<run>.<quantities|observations>.<member>`, only inside a contrast body.
    RunMember { run: String, namespace: RunNamespace, member: String },
}

/// One keyword argument of a call, a distribution or a staging law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arg {
    pub name: String,
    pub value: ExprTree,
}

/// A name in an expression, tagged with what the compiler resolved it to. The
/// tag is what lets a rename map rewrite compartment names without touching a
/// parameter of the same spelling.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NameRef {
    pub name: String,
    pub kind: NameKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NameKind {
    Compartment,
    Parameter,
    /// A `let`, whether the compiler later inlines it, keeps it as a binding
    /// or folds it to a constant.
    Definition,
    Table,
    /// A `forcing` declaration.
    TimeInput,
    Timepoint,
    /// A variable bound by an index binding or a `sum`.
    IndexVar,
    /// A level of a dimension, written as a literal index: `S[child]`.
    Level,
    /// A name the language defines, such as the time variable.
    Builtin,
    /// A column of an observation stream's data file, read by its likelihood.
    DataColumn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitLit {
    Days,
    Weeks,
    Months,
    Years,
    PerDay,
    PerWeek,
    PerMonth,
    PerYear,
    Count,
    Ratio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Sin,
    Cos,
    Tanh,
    /// `sum(e)` with no binder: the value of `e`, stated as a total.
    Total,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunNamespace {
    Quantities,
    Observations,
}

/// A positional or named index: `S[child]` or `S[age = child]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum IndexItem {
    Positional { value: ExprTree },
    Named { dim: String, value: ExprTree },
}

/// A binder on a declaration: `[a in age]`,
/// `[(a, a_next) in consecutive(age)]`, `[c in compartments]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum IndexBinding {
    In { var: String, dim: String },
    Consecutive { var: String, var_next: String, dim: String },
    Compartments { var: String },
}

/// A `where` predicate: index comparisons and constant table-cell
/// comparisons only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Guard {
    Eq { lhs: String, rhs: String },
    Neq { lhs: String, rhs: String },
    Table { table: String, indices: Vec<String>, op: RelOp, operand: GuardOperand },
    And { lhs: Box<Guard>, rhs: Box<Guard> },
    Or { lhs: Box<Guard>, rhs: Box<Guard> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum GuardOperand {
    Num { value: f64 },
    Name { name: String },
}

// ── Dimensions, compartments, stratification ────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimensionDecl {
    pub name: String,
    pub source: DimSource,
    /// The levels the compile used. For an inline source these are the levels
    /// written; for a file source, the levels read.
    pub levels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DimSource {
    Inline,
    /// `read("path", column = "col")`.
    Read { path: String, column: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompartmentDecl {
    pub name: String,
    pub kind: CompartmentKind,
    /// The dimensions this compartment is stratified by, resolved from the
    /// `stratify` declarations.
    pub dims: Vec<String>,
    /// The expanded compartment names this declaration produced.
    pub expands_to: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompartmentKind {
    Integer,
    Real,
}

/// `stratify by dim [only A, B]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StratifyDecl {
    pub dim: String,
    pub only: Option<Vec<String>>,
}

// ── Parameters, definitions, tables, time inputs ────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterDecl {
    pub name: String,
    /// Empty for a scalar parameter.
    pub dims: Vec<String>,
    pub kind: ParamKind,
    /// An explicit `dim` clause.
    pub dim: Option<DimAnnotation>,
    pub bounds: Option<Bounds>,
    pub prior: Option<PriorDecl>,
}

/// DSL parameter-type keyword (the `param_kind` production in
/// `parser.mly`). Was `Option<String>`; the typed enum is rejected at IR
/// deserialisation rather than re-parsed at every consumer (the gh#191
/// stringly-typed surface). Each kind's dimensional meaning lives in the
/// OCaml `Dimcheck.param_dim_of_kind`; `Instant`/`Duration` are time-typed
/// (`[T]`) per the 2026-05-22 calendar-time proposal. Serialises to the same
/// snake_case strings the field has always used (`"rate"`, …), so the type
/// swap is byte-compatible with existing IR JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    Rate,
    Probability,
    Count,
    Positive,
    Real,
    Instant,
    Duration,
}

impl ParamKind {
    /// The snake_case string used in IR JSON (mirrors the `serde` rename).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rate        => "rate",
            Self::Probability => "probability",
            Self::Count       => "count",
            Self::Positive    => "positive",
            Self::Real        => "real",
            Self::Instant     => "instant",
            Self::Duration    => "duration",
        }
    }
}

impl std::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An explicit dimension: exponents of population and of time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimAnnotation {
    pub population: i32,
    pub time: i32,
}

/// `in [lo, hi]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub lo: DeclaredExpr,
    pub hi: DeclaredExpr,
}

/// `~ family(kw = v, ...) [| dim]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorDecl {
    pub family: String,
    pub args: Vec<Arg>,
    /// The `| dim` clause of a partially pooled prior.
    pub pool_over: Option<String>,
}

/// A `let`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DefinitionDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    /// The dimensions of a shaped literal; `None` for a scalar or indexed let.
    pub shape: Option<Vec<String>>,
    /// An optional kind annotation.
    pub kind: Option<ParamKind>,
    pub dim: Option<DimAnnotation>,
    pub body: DeclaredExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableDecl {
    /// One or more names, for multi-value columns.
    pub names: Vec<String>,
    pub dims: Vec<TableDim>,
    pub cell_kind: Option<ParamKind>,
    /// An inline literal, or the call that reads the table from a file.
    pub value: DeclaredExpr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableDim {
    pub dim: String,
    pub unit: Option<UnitLit>,
}

/// A `forcing` declaration: a function of time supplied to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeInputDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    /// The forcing kind as written: `interpolated`, `sinusoidal`, ...
    pub kind: String,
    pub unit: UnitLit,
    pub args: Vec<TimeInputArg>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeInputArg {
    pub name: String,
    pub value: TimeInputValue,
}

/// A quoted string names something outside the model (a file path, a file
/// column); a bare expression names something inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeInputValue {
    Str { value: String },
    Expr { value: DeclaredExpr },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimepointDecl {
    pub name: String,
    pub time: DeclaredExpr,
}

// ── Transitions ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    pub from: Vec<StoichRef>,
    pub to: Destination,
    pub dynamics: Dynamics,
    pub guard: Option<Guard>,
    /// The `#[lineage]` attribute.
    pub lineage: bool,
    /// The expanded transition names this declaration produced.
    pub expands_to: Vec<String>,
}

/// A compartment in a transition's source or destination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoichRef {
    pub compartment: String,
    pub indices: Vec<IndexItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Destination {
    /// `A + B`: each compartment gains one.
    Sum { compartments: Vec<StoichRef> },
    /// `{ A : w_A, B : w_B }`: a probabilistic branch.
    Branch { arms: Vec<BranchArm> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BranchArm {
    pub to: StoichRef,
    pub weight: DeclaredExpr,
}

/// `@ rate`, or `via law(...)`. Never both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Dynamics {
    Rate { rate: DeclaredExpr },
    /// A staged residence, with the law's keyword arguments as written.
    Via { law: String, args: Vec<Arg> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OdeDecl {
    pub compartment: String,
    pub derivative: DeclaredExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BalanceDecl {
    pub compartment: String,
    pub expr: DeclaredExpr,
}

// ── Initial conditions ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialDecl {
    pub compartment: String,
    /// Positional indices: `S[child]`.
    pub indices: Vec<IndexItem>,
    /// Loop binders: `[p in patch]`.
    pub bindings: Vec<IndexBinding>,
    pub value: InitialValue,
}

/// Computed (`S = N0 - I`) or drawn once per particle (`I ~ poisson(...)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum InitialValue {
    Expr { value: DeclaredExpr },
    Law { law: Distribution },
}

/// A distribution call: an observation likelihood or an initial-value law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Distribution {
    pub family: Family,
    pub args: Vec<Arg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    NegBinomial,
    Poisson,
    Normal,
    Binomial,
    BetaBinomial,
    Beta,
    Bernoulli,
    ZeroInflatedNegBinomial,
}

// ── Observations ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    /// `from <label>`: the data source key. `None` means the stream name.
    pub source: Option<String>,
    pub columns: Vec<ObsColumn>,
    pub measurement: Measurement,
    pub projection: Projection,
    /// `emit_schedule`: when a simulation writes synthetic rows. Never read
    /// by a fit.
    pub emit_schedule: Option<ScheduleCore>,
    pub covers: Option<Covers>,
    /// The expanded observation-model names this declaration produced.
    pub expands_to: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObsColumn {
    pub name: String,
    pub role: ColumnRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ColumnRole {
    Time,
    Dim { dim: String },
    Value { kind: ParamKind },
    WindowStart,
    WindowStop,
}

/// `<scored column> ~ family(kw = v, ...)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub scored: String,
    pub likelihood: Distribution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Projection {
    /// The accumulated flow of a transition.
    Incidence { transition: String, indices: Vec<IndexItem> },
    Derived { value: DeclaredExpr },
}

/// `every = E` or `at = [...]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ScheduleCore {
    Every { every: DeclaredExpr },
    At { times: Vec<DeclaredExpr> },
}

/// `covers = day(t) | starting_on(t, d) | ending_on(t, d)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Covers {
    /// The form name as written.
    pub form: String,
    /// The time column named in the call.
    pub column: String,
    /// Absent for `day(t)`.
    pub span: Option<DeclaredExpr>,
}

// ── Interventions, events, reactive interventions ───────────────────────────

/// An entry of an `interventions` or an `events` block; the two share a form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    pub actions: Vec<Action>,
    pub schedule: Schedule,
    pub guard: Option<Guard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    /// `transfer(fraction =, count =, from =, to =)`, as written.
    Transfer { args: Vec<Arg> },
    Set { compartment: String, indices: Vec<IndexItem>, value: DeclaredExpr },
    Add { compartment: String, indices: Vec<IndexItem>, count: DeclaredExpr },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Schedule {
    AtTimes { times: Vec<DeclaredExpr> },
    Recurring { every: DeclaredExpr, from: Option<DeclaredExpr>, to: Option<DeclaredExpr> },
    EveryAtDay { period: DeclaredExpr, at_day: DeclaredExpr },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReactiveDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    pub when: TriggerPred,
    pub after: Option<DeclaredExpr>,
    pub once: Option<DeclaredExpr>,
    pub cooldown: Option<DeclaredExpr>,
    pub action: Action,
    pub guard: Option<Guard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TriggerPred {
    Atom { value: DeclaredExpr },
    And { lhs: Box<TriggerPred>, rhs: Box<TriggerPred> },
    Or { lhs: Box<TriggerPred>, rhs: Box<TriggerPred> },
    Not { arg: Box<TriggerPred> },
}

// ── Scenarios, simulation, output ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDecl {
    pub name: String,
    pub fields: Vec<ScenarioField>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ScenarioField {
    Label { text: String },
    Enable { names: Vec<String> },
    Disable { names: Vec<String> },
    Set { values: Vec<Assign> },
    Scale { values: Vec<Assign> },
    Compose { names: Vec<String> },
    TEnd { value: DeclaredExpr },
    Extends { parent: String },
}

/// `name = value` inside a scenario's `set` or `scale`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assign {
    pub name: String,
    pub value: DeclaredExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationDecl {
    pub from: DeclaredExpr,
    pub to: DeclaredExpr,
    pub dt: Option<DeclaredExpr>,
    /// `rk4` or `rk45`, as written.
    pub integrator: Option<String>,
    pub atol: Option<DeclaredExpr>,
    pub rtol: Option<DeclaredExpr>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputDecl {
    pub trajectories: Option<TrajectoryOutput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryOutput {
    pub schedule: ScheduleCore,
    pub format: String,
}

// ── Reporting ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantityDecl {
    pub name: String,
    pub indices: Vec<IndexBinding>,
    pub body: DeclaredExpr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContrastDecl {
    pub name: String,
    pub body: DeclaredExpr,
}
