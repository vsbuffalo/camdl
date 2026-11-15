//! A [`DiffReport`] as text for a terminal.
//!
//! Everything printed is read from the report. A changed declaration is shown
//! by its text as written on each side, which is never lossy, with the paths of
//! the members that differ; the formula rendering is for the viewer.

use std::fmt::Write;

use crate::{AffectedBy, Category, DiffReport, FieldValue, Layer, ModelOutcome, Row, Status};

fn outcome_sentence(outcome: &ModelOutcome) -> String {
    match outcome {
        ModelOutcome::Identical => "identical".to_string(),
        ModelOutcome::Different => "different".to_string(),
        ModelOutcome::WrittenDifferently => {
            "written differently; the two compile to the same model".to_string()
        }
        ModelOutcome::SameDeclarationsDifferentCompilers { left, right } => {
            format!("declared the same; compiled by different camdl versions ({left}, {right})")
        }
    }
}

fn short(hash: &str) -> &str {
    hash.get(..16).unwrap_or(hash)
}

fn value_text(value: &FieldValue) -> String {
    match value {
        FieldValue::Absent => "(absent)".to_string(),
        FieldValue::Expr { katex, .. } => katex.clone(),
        FieldValue::Other { value } => match value {
            serde_json::Value::String(s) => s.clone(),
            // A list of names (an order) reads as one.
            serde_json::Value::Array(items) if items.iter().all(serde_json::Value::is_string) => {
                items.iter().filter_map(serde_json::Value::as_str).collect::<Vec<_>>().join(", ")
            }
            other => other.to_string(),
        },
    }
}

fn affected_text(affected: &AffectedBy) -> Option<String> {
    let list = |changed: &[crate::EntityRef]| -> String {
        changed
            .iter()
            .map(|e| format!("{} {}", e.category.label(), e.name))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match affected {
        AffectedBy::Known { changed } if changed.is_empty() => None,
        AffectedBy::Known { changed } => Some(format!("reads changed: {}", list(changed))),
        AffectedBy::Unknown { unresolved, changed } => {
            let mut text =
                format!("dependencies unknown (reads the unresolved name {})", unresolved.join(", "));
            if !changed.is_empty() {
                write!(text, "; reads changed: {}", list(changed)).expect("write to a string");
            }
            Some(text)
        }
    }
}

fn status_text(row: &Row) -> String {
    let affected = affected_text(&row.affected_by);
    let base = match &row.status {
        Status::Same => match &affected {
            // Printed only when the row is affected; see `shown`.
            Some(_) => "unchanged as written".to_string(),
            None => "same".to_string(),
        },
        Status::Changed { fields } => {
            let paths: Vec<&str> = fields.iter().map(|f| f.field.as_str()).collect();
            format!("changed: {}", paths.join(", "))
        }
        Status::OnlyLeft => "only in left".to_string(),
        Status::OnlyRight => "only in right".to_string(),
        Status::NotCompared { why } => format!("not compared: {why}"),
    };
    match affected {
        Some(a) => format!("{base}; {a}"),
        None => base,
    }
}

/// A row is printed when it is not the same, or reads something that changed.
fn shown(row: &Row) -> bool {
    !row.status.is_same() || !row.affected_by.is_clear()
}

/// A declaration's text, labelled with its side. The text starts at the
/// declaration's first token, so its first line has no indentation and the
/// later ones keep the file's; the indentation they share is removed.
fn write_text(out: &mut String, side: &str, text: &str) {
    let indent = |line: &str| line.len() - line.trim_start().len();
    let shared = text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(indent).min().unwrap_or(0);
    for (i, line) in text.lines().enumerate() {
        let label = if i == 0 { side } else { "" };
        let line = if i == 0 { line } else { line.get(shared..).unwrap_or(line.trim_start()) };
        writeln!(out, "      {label:<6} {line}").expect("write to a string");
    }
}

/// `3 compartments`, `1 quantity`.
fn counted(n: usize, label: &str) -> String {
    match (n, label.strip_suffix('y')) {
        (1, _) => format!("1 {label}"),
        (_, Some(stem)) => format!("{n} {stem}ies"),
        (_, None) => format!("{n} {label}s"),
    }
}

fn write_row(out: &mut String, row: &Row) {
    let kind = match row.category {
        Category::DeclarationOrder => format!("order of {}s", row.name),
        other => format!("{} {}", other.label(), row.name),
    };
    writeln!(out, "  {kind}: {}", status_text(row)).expect("write to a string");

    let has_text = row.left_text.is_some() || row.right_text.is_some();
    if has_text && row.left_text == row.right_text {
        if let Some(text) = &row.left_text {
            write_text(out, "both", text);
        }
    } else if has_text {
        if let Some(text) = &row.left_text {
            write_text(out, "left", text);
        }
        if let Some(text) = &row.right_text {
            write_text(out, "right", text);
        }
    } else if let Status::Changed { fields } = &row.status {
        // No text as written on either side (a compile input, the
        // description, an order): show the values themselves.
        for f in fields {
            write_text(out, "left", &value_text(&f.left));
            write_text(out, "right", &value_text(&f.right));
        }
    }
    if let Status::Changed { fields } = &row.status {
        if has_text && fields.iter().any(|f| f.renders_alike()) {
            writeln!(out, "      note   the two render to the same formula; compare the text as written")
                .expect("write to a string");
        }
    }
}

fn write_layer(out: &mut String, report: &DiffReport, layer: Layer, title: &str) {
    let rows: Vec<&Row> = report.rows.iter().filter(|r| r.layer == layer).collect();
    if rows.is_empty() {
        return;
    }
    writeln!(out, "\n{title}").expect("write to a string");
    let printed: Vec<&&Row> = rows.iter().filter(|r| shown(r)).collect();
    for row in &printed {
        write_row(out, row);
    }
    let unchanged: Vec<String> = report
        .counts
        .iter()
        .filter(|c| c.layer == layer && c.same > 0)
        .map(|c| counted(c.same, c.category.label()))
        .collect();
    if !unchanged.is_empty() {
        let lead = if printed.is_empty() { "  all the same" } else { "  the same on both sides" };
        writeln!(out, "{lead}: {}", unchanged.join(", ")).expect("write to a string");
    }
}

/// The report as plain text.
pub fn table(report: &DiffReport) -> String {
    let mut out = String::new();
    writeln!(out, "left   {}  model {}", report.left.name, short(&report.left.model_hash))
        .expect("write to a string");
    writeln!(out, "right  {}  model {}", report.right.name, short(&report.right.model_hash))
        .expect("write to a string");
    writeln!(out, "\nModel: {}.", outcome_sentence(&report.model_outcome)).expect("write to a string");
    let has_reporting = report.rows.iter().any(|r| r.layer == Layer::Reporting);
    if has_reporting {
        let reporting = if report.reporting_identical { "identical" } else { "different" };
        writeln!(out, "Reported quantities and contrasts: {reporting}.").expect("write to a string");
    }
    write_layer(&mut out, report, Layer::Model, "Model");
    write_layer(&mut out, report, Layer::Reporting, "Reporting (not part of the model's identity)");
    out
}
