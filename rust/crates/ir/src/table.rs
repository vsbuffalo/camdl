use serde::{Deserialize, Serialize};
use crate::expr::Expr;

/// Table out-of-range lookups fail loud (a model bug). `Clamp`/`Wrap`
/// (silent flat-extrapolation / wrapping) were removed: the evaluator was
/// built but unselectable, and silent extrapolation masks modeling errors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OobPolicy {
    Error,
}

/// Source of table values.
///
/// - `Inline`: values resolved at compile time and embedded in the IR.
/// - `External`: values must be supplied at runtime via `--table name=file`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TableSource {
    /// `{"values": [...]}` — compile-time values
    Inline { values: Vec<Expr> },
    /// `{"external": "name", "shape": [{"dim": "patch", "levels": 3}]}` —
    /// values supplied when the model is run (`--table name=FILE`, or
    /// `[data.tables]` in a fit config). `shape` is the declaration's axes in
    /// order: the bound file must hold exactly one number per cell, the
    /// product of the levels (`cli::external_inputs::bind`).
    External { external: String, shape: Vec<TableAxis> },
}

/// One axis of a table declaration: its dimension and that dimension's
/// number of levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableAxis {
    pub dim:    String,
    pub levels: usize,
}

impl TableSource {
    pub fn values(&self) -> Option<&[Expr]> {
        match self {
            TableSource::Inline { values } => Some(values),
            TableSource::External { .. } => None,
        }
    }

    pub fn external_name(&self) -> Option<&str> {
        match self {
            TableSource::External { external, .. } => Some(external),
            TableSource::Inline { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Table {
    pub name:          String,
    #[serde(flatten)]
    pub source:        TableSource,
    pub out_of_bounds: OobPolicy,
    /// Optional cell-type annotation (gh#32). When present, every cell in
    /// the table is treated as having this dimensional kind (same vocabulary
    /// as `Parameter::param_kind`: `"rate"`, `"probability"`, `"positive"`,
    /// `"count"`, `"real"`). Absent = dimensionless cells (legacy
    /// behaviour). The OCaml dim-checker treats this as authoritative; the
    /// Rust backend currently passes the field through serde for round-trip
    /// fidelity but does not consult it during simulation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_kind:     Option<String>,
}
