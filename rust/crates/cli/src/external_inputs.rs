//! Binding a model's external inputs: the files that supply its
//! `external("NAME")` tables and `data = external("NAME")` forcings.
//!
//! A model declares an external input by name only; the file is chosen when
//! the model is run, with `--table NAME=FILE` on the command line or
//! `[data.tables] NAME = "FILE"` in a fit config. [`bind`] is the one place a
//! bound file is read into the model: it replaces each
//! [`TableSource::External`] with the file's values and each
//! [`InterpSource::External`] with the knots read from it, so the model that
//! reaches `sim::CompiledModel::new` carries no external source. Every command
//! reaches it through `params_resolver::resolve_parameters`.
//!
//! The file's bytes are a run input, held where each command holds its
//! inputs (a simulation cell's parameters, a fit problem's
//! `FitProblemIdentity::tables`); the model's own hash covers only the
//! declaration.
//!
//! An external forcing's file is read exactly as `camdlc` reads a
//! `data = "path"` forcing (`read_csv_rows` and `load_interpolated_for_level`
//! in `ocaml/lib/compiler/expander.ml`): separator from the extension (`.csv`
//! or `.tsv`), lines trimmed, blank and `#` lines skipped, the first remaining
//! line the header, cells trimmed, a time cell a number in model time or else
//! an ISO date resolved through the model's origin, a value scaled by the
//! declaration's `value_scale`, the rows kept in file order. The test
//! `external_forcing_matches_file_forcing` pins the two readers to one result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ir::expr::Expr;
use ir::table::{TableAxis, TableSource};
use ir::time_func::{ExternalSeries, InterpSource, TimeFuncKind};

/// Where a binding was written, so a refusal can quote it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingOrigin {
    /// `--table NAME=FILE`.
    Cli,
    /// `[data.tables] NAME = "FILE"` in a fit config.
    FitConfig,
}

impl BindingOrigin {
    /// The binding as the user wrote it.
    fn spell(self, name: &str, path: &Path) -> String {
        match self {
            BindingOrigin::Cli => format!("--table {name}={}", path.display()),
            BindingOrigin::FitConfig => format!("[data.tables] {name} = \"{}\"", path.display()),
        }
    }
}

/// One bound file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub path:   PathBuf,
    pub origin: BindingOrigin,
}

/// External name → the file bound to it. The command line is laid over a
/// fit config ([`ExternalBindings::overlay`]), as `--data` is over
/// `[data.observations]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalBindings {
    entries: BTreeMap<String, Binding>,
}

impl ExternalBindings {
    /// No bindings: a model with no external input binds against this.
    pub fn none() -> Self {
        Self::default()
    }

    /// `--table NAME=FILE` pairs.
    pub fn from_cli<I, S, P>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, P)>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        Self::with_origin(pairs, BindingOrigin::Cli)
    }

    /// A fit config's `[data.tables]`, its paths already anchored at the
    /// config's directory.
    pub fn from_fit_config<I, S, P>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, P)>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        Self::with_origin(pairs, BindingOrigin::FitConfig)
    }

    /// `self` with every binding in `top` replacing the one of the same name.
    pub fn overlay(mut self, top: ExternalBindings) -> Self {
        self.entries.extend(top.entries);
        self
    }

    fn with_origin<I, S, P>(pairs: I, origin: BindingOrigin) -> Self
    where
        I: IntoIterator<Item = (S, P)>,
        S: Into<String>,
        P: Into<PathBuf>,
    {
        let entries = pairs
            .into_iter()
            .map(|(name, path)| (name.into(), Binding { path: path.into(), origin }))
            .collect();
        ExternalBindings { entries }
    }

    pub fn get(&self, name: &str) -> Option<&Binding> {
        self.entries.get(name)
    }

    /// (name, binding), sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Binding)> {
        self.entries.iter().map(|(n, b)| (n.as_str(), b))
    }
}

/// One bound file as [`bind`] read it: the path, the bytes, their digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundFile {
    pub path:   PathBuf,
    pub bytes:  Vec<u8>,
    pub digest: store::Key,
}

/// Every file [`bind`] read, by external name. The values bound into the model,
/// the digests a run's identity folds and the copy a fit stores all come from
/// this one read, so they cannot disagree about which bytes were used.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundFiles {
    files: BTreeMap<String, BoundFile>,
}

impl BoundFiles {
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// (name, file), sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &BoundFile)> {
        self.files.iter().map(|(n, f)| (n.as_str(), f))
    }
}

/// A declaration in the model that reads an external input.
#[derive(Debug, Clone, PartialEq)]
enum Reader {
    Table { table: String },
    Forcing { forcing: String },
}

impl Reader {
    fn describe(&self) -> String {
        match self {
            Reader::Table { table } => format!("table '{table}'"),
            Reader::Forcing { forcing } => format!("forcing '{forcing}'"),
        }
    }
}

/// Why a model's external inputs could not be bound. Every variant names the
/// external input; the ones about a file name the file and the binding that
/// chose it.
#[derive(Debug, Clone, PartialEq)]
pub enum BindError {
    /// The model reads `name` and nothing binds it.
    Unbound { name: String, readers: Vec<String> },
    /// A binding names an input the model does not declare.
    Undeclared { name: String, binding: String, declared: Vec<String> },
    /// The bound file could not be read.
    Unreadable { name: String, binding: String, error: String },
    /// The bound file does not hold what a declaration reads from it.
    Mismatch { name: String, binding: String, reader: String, problem: String },
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindError::Unbound { name, readers } => write!(
                f,
                "external input '{name}' has no file: {} read{} it.\n  \
                 Fix: bind one with `--table {name}=FILE`, or in the fit config \
                 `[data.tables] {name} = \"FILE\"`.",
                readers.join(" and "),
                if readers.len() == 1 { "s" } else { "" },
            ),
            BindError::Undeclared { name, binding, declared } => {
                let declared = if declared.is_empty() {
                    "the model declares no external input".to_string()
                } else {
                    format!("the model's external inputs are: {}", declared.join(", "))
                };
                write!(
                    f,
                    "external input '{name}' is bound ({binding}) but the model does \
                     not declare it; {declared}.\n  \
                     Fix: correct the name, or remove the binding."
                )
            }
            BindError::Unreadable { name, binding, error } => {
                write!(f, "external input '{name}': cannot read the file ({binding}): {error}")
            }
            BindError::Mismatch { name, binding, reader, problem } => write!(
                f,
                "external input '{name}' ({binding}) does not match what {reader} \
                 reads from it: {problem}"
            ),
        }
    }
}

impl std::error::Error for BindError {}

/// The model's external inputs: each name and the declarations that read it.
fn declared(model: &ir::Model) -> BTreeMap<String, Vec<Reader>> {
    let mut out: BTreeMap<String, Vec<Reader>> = BTreeMap::new();
    for t in &model.tables {
        if let TableSource::External { external, .. } = &t.source {
            out.entry(external.clone()).or_default().push(Reader::Table { table: t.name.clone() });
        }
    }
    for tf in &model.time_functions {
        if let TimeFuncKind::Interpolated(i) = &tf.kind {
            if let InterpSource::External(e) = &i.source {
                out.entry(e.external.clone())
                    .or_default()
                    .push(Reader::Forcing { forcing: tf.name.clone() });
            }
        }
    }
    out
}

/// Whether `model` declares any external input still to be bound.
pub fn declares_any(model: &ir::Model) -> bool {
    !declared(model).is_empty()
}

/// Bind every external input of `model` from `bindings`, in place.
///
/// Refuses, naming the input: a declared input with no binding; a binding for
/// a name the model does not declare (a misspelt override would otherwise be
/// ignored while the input it meant ran on another file); a file that cannot
/// be read; a table file without exactly one number per declared cell; a
/// forcing file without the declared columns, with an unparseable cell, or
/// with no row for the forcing's stratum. Each bound file is read once, however
/// many declarations read it; what was read is returned.
#[expect(clippy::disallowed_methods, reason = "reads each file bound to an external input (`--table`, declared `#[input(identity)]` on the model overrides, or a fit config's `[data.tables]`) and returns its bytes and digest for the caller's identity; `camdl eval`, which stores nothing, drops them")]
pub fn bind(model: &mut ir::Model, bindings: &ExternalBindings) -> Result<BoundFiles, BindError> {
    let declared = declared(model);
    for (name, binding) in bindings.iter() {
        if !declared.contains_key(name) {
            return Err(BindError::Undeclared {
                name: name.to_string(),
                binding: binding.origin.spell(name, &binding.path),
                declared: declared.keys().cloned().collect(),
            });
        }
    }
    if declared.is_empty() {
        return Ok(BoundFiles::default());
    }

    let mut read = BoundFiles::default();
    let mut files: BTreeMap<&str, (String, String)> = BTreeMap::new();
    for (name, readers) in &declared {
        let Some(binding) = bindings.get(name) else {
            return Err(BindError::Unbound {
                name: name.clone(),
                readers: readers.iter().map(Reader::describe).collect(),
            });
        };
        let spelled = binding.origin.spell(name, &binding.path);
        let unreadable = |error: String| BindError::Unreadable {
            name: name.clone(),
            binding: spelled.clone(),
            error,
        };
        let bytes = std::fs::read(&binding.path).map_err(|e| unreadable(e.to_string()))?;
        let text = String::from_utf8(bytes.clone())
            .map_err(|_| unreadable("the file is not UTF-8 text".to_string()))?;
        read.files.insert(name.clone(), BoundFile {
            path: binding.path.clone(),
            digest: store::Key::digest_bytes(&bytes),
            bytes,
        });
        files.insert(name.as_str(), (spelled, text));
    }

    let mismatch = |name: &str, binding: &str, reader: &Reader, problem: String| BindError::Mismatch {
        name: name.to_string(),
        binding: binding.to_string(),
        reader: reader.describe(),
        problem,
    };

    for table in &mut model.tables {
        let TableSource::External { external, shape } = &table.source else { continue };
        let (binding, text) = &files[external.as_str()];
        let reader = Reader::Table { table: table.name.clone() };
        let path = &bindings.get(external).expect("bound above").path;
        let values = crate::util::parse_table_text(text, path)
            .map_err(|problem| mismatch(external, binding, &reader, problem))?;
        let cells: usize = shape.iter().map(|a| a.levels).product();
        if values.len() != cells {
            return Err(mismatch(external, binding, &reader, format!(
                "the table is declared over {} — {cells} cell{} — but the file holds {} number{}",
                describe_shape(shape),
                if cells == 1 { "" } else { "s" },
                values.len(),
                if values.len() == 1 { "" } else { "s" },
            )));
        }
        table.source = TableSource::Inline { values };
    }

    let origin = model.origin.clone();
    let time_unit = model.time_unit.clone();
    for tf in &mut model.time_functions {
        let TimeFuncKind::Interpolated(interp) = &mut tf.kind else { continue };
        let InterpSource::External(series) = &interp.source else { continue };
        let (binding, text) = &files[series.external.as_str()];
        let reader = Reader::Forcing { forcing: tf.name.clone() };
        let path = &bindings.get(&series.external).expect("bound above").path;
        let (times, values) = read_series(text, path, series, origin.as_deref(), &time_unit)
            .map_err(|problem| mismatch(&series.external, binding, &reader, problem))?;
        interp.source = InterpSource::Knots { times, values };
    }
    Ok(read)
}

fn describe_shape(shape: &[TableAxis]) -> String {
    if shape.is_empty() {
        return "no dimension".to_string();
    }
    shape
        .iter()
        .map(|a| format!("{} ({} level{})", a.dim, a.levels, if a.levels == 1 { "" } else { "s" }))
        .collect::<Vec<_>>()
        .join(" × ")
}

/// The knots an external forcing reads from `text`, as `camdlc` would write
/// them for `data = "path"`: `Const(t)` times and `Const(v)` values, each value
/// `Const(v) × Const(value_scale)` when the scale is not 1.
fn read_series(
    text: &str,
    path: &Path,
    series: &ExternalSeries,
    origin: Option<&str>,
    time_unit: &str,
) -> Result<(Vec<Expr>, Vec<Expr>), String> {
    let sep = match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("csv") => ',',
        Some("tsv") => '\t',
        other => {
            return Err(format!(
                "a forcing's data file must end in .csv or .tsv (got {})",
                other.map(|e| format!("'.{e}'")).unwrap_or_else(|| "no extension".to_string())
            ))
        }
    };
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));
    let Some((_, header)) = lines.next() else {
        return Err("the file is empty (no header row)".to_string());
    };
    let header: Vec<&str> = header.split(sep).map(str::trim).collect();
    let column = |name: &str| -> Result<usize, String> {
        header.iter().position(|h| *h == name).ok_or_else(|| {
            format!("no column '{name}' in its header (the columns are: {})", header.join(", "))
        })
    };
    let time_i = column(&series.time_col)?;
    let value_i = column(&series.value_col)?;
    let key = match &series.key {
        Some(k) => Some((column(&k.column)?, k.level.as_str())),
        None => None,
    };

    let mut times = Vec::new();
    let mut values = Vec::new();
    for (row, line) in lines {
        let cells: Vec<&str> = line.split(sep).collect();
        let cell = |i: usize| cells.get(i).map(|c| c.trim()).unwrap_or("");
        if let Some((key_i, level)) = key {
            if cell(key_i) != level {
                continue;
            }
        }
        let t_cell = cell(time_i);
        let t = match t_cell.parse::<f64>() {
            Ok(t) => t,
            Err(_) => match origin {
                Some(o) if ir::caltime::parse_iso_date(t_cell).is_ok() => {
                    ir::caltime::date_to_internal(o, t_cell, time_unit).map_err(|e| {
                        format!("row {row}: time cell '{t_cell}' is not a resolvable date ({e:?})")
                    })?
                }
                None if ir::caltime::parse_iso_date(t_cell).is_ok() => {
                    return Err(format!(
                        "row {row}: time cell '{t_cell}' is a date but the model declares no \
                         origin; add `origin = date(\"YYYY-MM-DD\")` to the model, or use \
                         numbers in model time"
                    ))
                }
                _ => {
                    return Err(format!(
                        "row {row}: time cell '{t_cell}' in column '{}' is neither a number \
                         nor an ISO date",
                        series.time_col
                    ))
                }
            },
        };
        let v_cell = cell(value_i);
        let v: f64 = v_cell.parse().map_err(|_| {
            format!("row {row}: value cell '{v_cell}' in column '{}' is not a number", series.value_col)
        })?;
        times.push(Expr::const_(t));
        values.push(if series.value_scale == 1.0 {
            Expr::const_(v)
        } else {
            Expr::bin_op(ir::expr::BinOp::Mul, Expr::const_(v), Expr::const_(series.value_scale))
        });
    }
    if times.is_empty() {
        return Err(match &series.key {
            Some(k) => format!("no rows where column '{}' = '{}'", k.column, k.level),
            None => "the file has a header but no data rows".to_string(),
        });
    }
    Ok((times, values))
}
