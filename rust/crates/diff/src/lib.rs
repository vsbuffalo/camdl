//! The model structure diff: what differs between two models, decided on what
//! their authors declared.
//!
//! Design: `docs/dev/proposals/2026-11-13-model-structure-diff.md`.
//!
//! [`diff`] is a pure function of two [`Side`]s. Each side is a model's
//! declared section (the model as written, from the `declared` crate) and the
//! hash of the model it compiled to. The function reads no files and hashes
//! nothing: the two hashes arrive as inputs and are used only as a cross-check
//! on the rows.
//!
//! A row's status is decided on declarations. Two declarations are the same
//! when their documents are equal with the display-only members left out
//! ([`document::same`]); for an expression that is equality of its parsed tree.
//! Nothing is normalized: `beta * S * I` and `S * beta * I` differ.
//!
//! The model layer then has two independent facts, whether the declarations
//! differ and whether the compiled models differ, and [`ModelOutcome`] states
//! both. Declarations that are all the same over compiled models that differ is
//! the one combination that must not happen for one compiler; [`diff`] refuses
//! it ([`Refusal::SameDeclarationsDifferentModels`]) and never reports it as
//! "identical".
//!
//! Built so far: the model and reporting layers for two models. Not yet built
//! (later steps of the proposal): the fit layer, roll-ups, rename maps, and
//! rows for doc comments, which live outside the declared section.

use std::collections::{BTreeMap, BTreeSet};

use declared::{DeclaredError, DeclaredSection, KatexLoss};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod document;
mod entities;
pub mod render;

use entities::Entity;

/// The schema version of a [`DiffReport`].
pub const SCHEMA_VERSION: u32 = 1;

// ── Inputs ──────────────────────────────────────────────────────────────────

/// One model to compare.
#[derive(Debug, Clone, Copy)]
pub struct Side<'a> {
    /// What to call it in the report: a file name, a fit label.
    pub name: &'a str,
    /// The declared section of its compiled model.
    pub declared: &'a DeclaredSection,
    /// The hash of the model it compiled to (`model_ir_hash`), as hex.
    pub model_hash: &'a str,
    /// The compiler that wrote it (the envelope's `validated_by`), when known.
    pub compiler: Option<&'a str>,
}

/// Which of the two models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Which {
    Left,
    Right,
}

impl std::fmt::Display for Which {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
        })
    }
}

/// Why no report could be made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A side has no declared model this build can read: none was recorded, or
    /// it is of a schema version this build does not understand.
    DeclaredUnavailable { which: Which, name: String, why: DeclaredError },
    /// Every declaration is the same and the compiled models differ, for two
    /// models not known to come from different compilers. This is a defect, in
    /// camdl or in what the declared model covers, and is never reported as
    /// "identical".
    SameDeclarationsDifferentModels { left_hash: String, right_hash: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeclaredUnavailable { which, name, why } => {
                write!(f, "the diff is unavailable for the {which} model ({name}): {why}")
            }
            Self::SameDeclarationsDifferentModels { left_hash, right_hash } => write!(
                f,
                "the two models declare the same things but compile to different models \
                 ({left_hash} and {right_hash}). The declared model does not cover whatever \
                 differs; this is a defect in camdl, and the models are not reported as identical"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

// ── The report ──────────────────────────────────────────────────────────────

/// What a row belongs to. The model layer is what the model file declares and
/// the model hash covers; the reporting layer is `quantities` and `contrasts`,
/// which the hash leaves out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Model,
    Reporting,
}

/// The kind of thing a row is about, in the order rows are reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// A compile-time `--set` value, or a file the compiler read.
    CompileInput,
    /// The model's `description`, which the model hash covers.
    Description,
    Dimension,
    Compartment,
    /// A `stratify by` declaration.
    Stratification,
    Parameter,
    /// A `let`, constants included.
    Definition,
    Table,
    /// A `forcing` declaration.
    TimeInput,
    Timepoint,
    Transition,
    OdeEquation,
    Balance,
    InitialCondition,
    Observation,
    Intervention,
    Event,
    ReactiveIntervention,
    Scenario,
    /// The `simulate` block with the time unit and the origin.
    Simulation,
    Output,
    Quantity,
    Contrast,
    /// The entities two models share in one category appear in a different
    /// order. The row is named after that category.
    DeclarationOrder,
}

impl Category {
    /// The categories that hold declarations, in report order.
    const DECLARED: [Category; 23] = [
        Category::CompileInput,
        Category::Description,
        Category::Dimension,
        Category::Compartment,
        Category::Stratification,
        Category::Parameter,
        Category::Definition,
        Category::Table,
        Category::TimeInput,
        Category::Timepoint,
        Category::Transition,
        Category::OdeEquation,
        Category::Balance,
        Category::InitialCondition,
        Category::Observation,
        Category::Intervention,
        Category::Event,
        Category::ReactiveIntervention,
        Category::Scenario,
        Category::Simulation,
        Category::Output,
        Category::Quantity,
        Category::Contrast,
    ];

    pub fn layer(self) -> Layer {
        match self {
            Category::Quantity | Category::Contrast => Layer::Reporting,
            _ => Layer::Model,
        }
    }

    /// The name used in a report, singular.
    pub fn label(self) -> &'static str {
        match self {
            Category::CompileInput => "compile input",
            Category::Description => "description",
            Category::Dimension => "dimension",
            Category::Compartment => "compartment",
            Category::Stratification => "stratification",
            Category::Parameter => "parameter",
            Category::Definition => "definition",
            Category::Table => "table",
            Category::TimeInput => "time input",
            Category::Timepoint => "timepoint",
            Category::Transition => "transition",
            Category::OdeEquation => "ODE equation",
            Category::Balance => "balance",
            Category::InitialCondition => "initial condition",
            Category::Observation => "observation stream",
            Category::Intervention => "intervention",
            Category::Event => "event",
            Category::ReactiveIntervention => "reactive intervention",
            Category::Scenario => "scenario",
            Category::Simulation => "simulation",
            Category::Output => "output",
            Category::Quantity => "quantity",
            Category::Contrast => "contrast",
            Category::DeclarationOrder => "declaration order",
        }
    }
}

/// One side of a changed member of a declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FieldValue {
    /// The member is not there on this side.
    Absent,
    /// An expression, by its rendering. `katex_loss` says where the rendering
    /// does not show the whole expression; a reader then shows the
    /// declaration's text as written beside it.
    Expr { katex: String, katex_loss: Vec<KatexLoss> },
    /// Anything else, in its document form without display-only members.
    Other { value: Value },
}

/// One member of a declaration that differs. `field` is its path in the
/// declaration's document (`dynamics.rate.rate`, `from[0].compartment`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldChange<T> {
    pub field: String,
    pub left: T,
    pub right: T,
}

impl FieldChange<FieldValue> {
    /// Two different expressions that render to the same formula. A reader
    /// must show such a change by the text as written, never by the formula.
    pub fn renders_alike(&self) -> bool {
        match (&self.left, &self.right) {
            (FieldValue::Expr { katex: l, .. }, FieldValue::Expr { katex: r, .. }) => l == r,
            _ => false,
        }
    }
}

/// How the two sides of a row compare.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum Status<T> {
    Same,
    Changed { fields: Vec<FieldChange<T>> },
    OnlyLeft,
    OnlyRight,
    /// Present on both sides and not comparable: a file the compiler read
    /// whose digest was not recorded. Never counted as the same.
    NotCompared { why: String },
}

impl<T> Status<T> {
    pub fn is_same(&self) -> bool {
        matches!(self, Status::Same)
    }
}

/// A row, named.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityRef {
    pub category: Category,
    pub name: String,
}

/// The changed things a row reads, directly or through definitions: the
/// definitions, tables, time inputs and quantities that are not the same on
/// both sides. A transition whose rate is written the same while a `let` it
/// reads has changed is `Same` with a non-empty list here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "dependencies")]
pub enum AffectedBy {
    /// Every name the row reads was resolved by the compiler, and these are the
    /// changed things among them. Empty: nothing it reads changed.
    Known { changed: Vec<EntityRef> },
    /// The row reads, directly or through a definition, a name the compiler
    /// could not classify (`unresolved`), so what it depends on is not known.
    /// `changed` holds what could be followed and is not the whole list.
    Unknown { unresolved: Vec<String>, changed: Vec<EntityRef> },
}

impl AffectedBy {
    /// Whether a reader should treat the row as possibly affected.
    pub fn is_clear(&self) -> bool {
        matches!(self, AffectedBy::Known { changed } if changed.is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    pub layer: Layer,
    pub category: Category,
    pub name: String,
    #[serde(flatten)]
    pub status: Status<FieldValue>,
    pub affected_by: AffectedBy,
    /// The declaration's text as written on each side, for display. `None`
    /// when the side has no such declaration or the compiler had no text.
    pub left_text: Option<String>,
    pub right_text: Option<String>,
}

/// The two facts of the model layer, together.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ModelOutcome {
    /// The declarations are the same and so are the compiled models.
    Identical,
    /// The declarations differ and so do the compiled models.
    Different,
    /// The declarations differ and the two files compile to the same model.
    WrittenDifferently,
    /// The declarations are the same and the compiled models differ, for two
    /// models written by different compilers.
    SameDeclarationsDifferentCompilers { left: String, right: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub name: String,
    pub model_hash: String,
    pub compiler: Option<String>,
}

/// How many rows of one category have each status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub layer: Layer,
    pub category: Category,
    pub same: usize,
    pub changed: usize,
    pub only_left: usize,
    pub only_right: usize,
    pub not_compared: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffReport {
    pub schema_version: u32,
    pub left: ModelRef,
    pub right: ModelRef,
    pub model_outcome: ModelOutcome,
    /// The two model hashes are equal. Read from the hashes, never from rows.
    pub model_identical: bool,
    /// Every row of the reporting layer is the same, order included.
    pub reporting_identical: bool,
    pub rows: Vec<Row>,
    /// One entry for each category with a row, in report order.
    pub counts: Vec<Counts>,
}

impl DiffReport {
    /// The two models are the same in every layer present.
    pub fn identical(&self) -> bool {
        self.model_outcome == ModelOutcome::Identical && self.reporting_identical
    }
}

// ── The diff ────────────────────────────────────────────────────────────────

fn compare(category: Category, left: &Entity, right: &Entity) -> Status<FieldValue> {
    // A file the compiler read is compared by the digest of its bytes. Without
    // a digest on both sides there is nothing to compare, and saying "same"
    // on the path alone would be a claim about bytes nobody looked at.
    if category == Category::CompileInput && left.doc.get("path").is_some() {
        let missing = |e: &Entity| e.doc.get("sha256").is_none_or(Value::is_null);
        let why = match (missing(left), missing(right)) {
            (false, false) => None,
            (true, false) => Some("the left model's compile did not record the file's digest"),
            (false, true) => Some("the right model's compile did not record the file's digest"),
            (true, true) => Some("neither compile recorded the file's digest"),
        };
        if let Some(why) = why {
            return Status::NotCompared { why: why.to_string() };
        }
    }
    let fields = document::changes(&left.doc, &right.doc);
    if fields.is_empty() {
        Status::Same
    } else {
        Status::Changed { fields }
    }
}

/// The names of `mine` that `other` also has, in `mine`'s order.
fn shared_in_order<'a>(mine: &[&'a Entity], other: &[&Entity]) -> Vec<&'a str> {
    let theirs: BTreeSet<&str> = other.iter().map(|e| e.name.as_str()).collect();
    mine.iter().map(|e| e.name.as_str()).filter(|n| theirs.contains(n)).collect()
}

fn names_value(names: &[&str]) -> FieldValue {
    FieldValue::Other { value: Value::from(names.to_vec()) }
}

/// The rows of one category, then an order row if the shared entities are in a
/// different order.
fn category_rows(category: Category, left: &[&Entity], right: &[&Entity], rows: &mut Vec<Row>) {
    let layer = category.layer();
    let right_by_name: BTreeMap<&str, &Entity> = right.iter().map(|e| (e.name.as_str(), *e)).collect();
    let left_names: BTreeSet<&str> = left.iter().map(|e| e.name.as_str()).collect();
    let pending = AffectedBy::Known { changed: Vec::new() };

    for l in left {
        let (status, right_text) = match right_by_name.get(l.name.as_str()) {
            Some(r) => (compare(category, l, r), r.source_text.clone()),
            None => (Status::OnlyLeft, None),
        };
        rows.push(Row {
            layer,
            category,
            name: l.name.clone(),
            status,
            affected_by: pending.clone(),
            left_text: l.source_text.clone(),
            right_text,
        });
    }
    for r in right.iter().filter(|r| !left_names.contains(r.name.as_str())) {
        rows.push(Row {
            layer,
            category,
            name: r.name.clone(),
            status: Status::OnlyRight,
            affected_by: pending.clone(),
            left_text: None,
            right_text: r.source_text.clone(),
        });
    }

    // The order of `let`s is not a difference: where a definition sits does not
    // reach the compiled model (measured: two files that differ only in the
    // order of two definitions compile to the same model). For every other
    // category order is reported, whether or not the hash is known to cover
    // it; where it does not, the outcome reads "written differently".
    if category == Category::Definition {
        return;
    }
    let left_order = shared_in_order(left, right);
    let right_order = shared_in_order(right, left);
    if left_order != right_order {
        rows.push(Row {
            layer,
            category: Category::DeclarationOrder,
            name: category.label().to_string(),
            status: Status::Changed {
                fields: vec![FieldChange {
                    field: "order".to_string(),
                    left: names_value(&left_order),
                    right: names_value(&right_order),
                }],
            },
            affected_by: pending,
            left_text: None,
            right_text: None,
        });
    }
}

/// For each row, the changed things it reads.
///
/// From a row's own names the walk follows every name that refers to a
/// definition, a table, a time input or a quantity, on either side, and on
/// through what those read. An index variable, a level or a keyword refers to
/// none of them and is not followed; the kind on each name is what tells them
/// apart.
fn affected_by(rows: &mut [Row], left: &[Entity], right: &[Entity]) {
    type Key = (Category, String);
    let mut by_key: BTreeMap<Key, Vec<&Entity>> = BTreeMap::new();
    let mut providers: BTreeMap<&(String, String), BTreeSet<Key>> = BTreeMap::new();
    for e in left.iter().chain(right) {
        let key = (e.category, e.name.clone());
        by_key.entry(key.clone()).or_default().push(e);
        for name in &e.provides {
            providers.entry(name).or_default().insert(key.clone());
        }
    }
    let changed: BTreeSet<Key> = rows
        .iter()
        .filter(|r| !r.status.is_same() && r.category != Category::DeclarationOrder)
        .map(|r| (r.category, r.name.clone()))
        .collect();

    for row in rows.iter_mut().filter(|r| r.category != Category::DeclarationOrder) {
        let own: Key = (row.category, row.name.clone());
        let mut stack: Vec<&(String, String)> =
            by_key.get(&own).into_iter().flatten().flat_map(|e| e.reads.iter()).collect();
        let mut visited: BTreeSet<&Key> = BTreeSet::new();
        let mut unresolved: BTreeSet<String> = BTreeSet::new();
        let mut hit: BTreeSet<EntityRef> = BTreeSet::new();

        while let Some(name) = stack.pop() {
            if name.1 == "unresolved" {
                unresolved.insert(name.0.clone());
                continue;
            }
            for key in providers.get(&name).into_iter().flatten() {
                if *key == own || !visited.insert(key) {
                    continue;
                }
                if changed.contains(key) {
                    hit.insert(EntityRef { category: key.0, name: key.1.clone() });
                }
                stack.extend(by_key.get(key).into_iter().flatten().flat_map(|e| e.reads.iter()));
            }
        }

        let changed_list: Vec<EntityRef> = hit.into_iter().collect();
        row.affected_by = if unresolved.is_empty() {
            AffectedBy::Known { changed: changed_list }
        } else {
            AffectedBy::Unknown { unresolved: unresolved.into_iter().collect(), changed: changed_list }
        };
    }
}

fn count(rows: &[Row]) -> Vec<Counts> {
    let mut counts: BTreeMap<(Layer, Category), Counts> = BTreeMap::new();
    for row in rows {
        let entry = counts.entry((row.layer, row.category)).or_insert(Counts {
            layer: row.layer,
            category: row.category,
            same: 0,
            changed: 0,
            only_left: 0,
            only_right: 0,
            not_compared: 0,
        });
        match row.status {
            Status::Same => entry.same += 1,
            Status::Changed { .. } => entry.changed += 1,
            Status::OnlyLeft => entry.only_left += 1,
            Status::OnlyRight => entry.only_right += 1,
            Status::NotCompared { .. } => entry.not_compared += 1,
        }
    }
    counts.into_values().collect()
}

/// Compare two models.
///
/// Refuses when a side has no readable declared model, and when every
/// declaration is the same while the compiled models differ (see
/// [`Refusal`]).
pub fn diff(left: &Side<'_>, right: &Side<'_>) -> Result<DiffReport, Refusal> {
    let unavailable = |which, side: &Side<'_>, why| Refusal::DeclaredUnavailable {
        which,
        name: side.name.to_string(),
        why,
    };
    let left_model = left.declared.recorded().map_err(|why| unavailable(Which::Left, left, why))?;
    let right_model = right.declared.recorded().map_err(|why| unavailable(Which::Right, right, why))?;

    let left_entities = entities::entities(left_model);
    let right_entities = entities::entities(right_model);

    let mut rows: Vec<Row> = Vec::new();
    for category in Category::DECLARED {
        let l: Vec<&Entity> = left_entities.iter().filter(|e| e.category == category).collect();
        let r: Vec<&Entity> = right_entities.iter().filter(|e| e.category == category).collect();
        category_rows(category, &l, &r, &mut rows);
    }
    affected_by(&mut rows, &left_entities, &right_entities);

    let layer_same = |layer: Layer| rows.iter().filter(|r| r.layer == layer).all(|r| r.status.is_same());
    let declared_same = layer_same(Layer::Model);
    let model_identical = left.model_hash == right.model_hash;

    let model_outcome = match (declared_same, model_identical) {
        (true, true) => ModelOutcome::Identical,
        (false, false) => ModelOutcome::Different,
        (false, true) => ModelOutcome::WrittenDifferently,
        (true, false) => match (left.compiler, right.compiler) {
            (Some(l), Some(r)) if l != r => ModelOutcome::SameDeclarationsDifferentCompilers {
                left: l.to_string(),
                right: r.to_string(),
            },
            _ => {
                return Err(Refusal::SameDeclarationsDifferentModels {
                    left_hash: left.model_hash.to_string(),
                    right_hash: right.model_hash.to_string(),
                })
            }
        },
    };

    let model_ref = |side: &Side<'_>| ModelRef {
        name: side.name.to_string(),
        model_hash: side.model_hash.to_string(),
        compiler: side.compiler.map(str::to_string),
    };
    Ok(DiffReport {
        schema_version: SCHEMA_VERSION,
        left: model_ref(left),
        right: model_ref(right),
        model_outcome,
        model_identical,
        reporting_identical: layer_same(Layer::Reporting),
        counts: count(&rows),
        rows,
    })
}
