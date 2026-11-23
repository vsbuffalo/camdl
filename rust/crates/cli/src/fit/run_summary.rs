//! A fit run's stored summary, `summary.json`: the run's entry in
//! `fit summary --format json` with every field that is not a function of the
//! run's identity left out, so that it can be a result file.
//!
//! The entry is the method's report and, for a method with a tabulated result,
//! its `fit table` row, built by the functions `fit summary` renders with. The
//! fields left out are of four kinds: times (when the run started, its age,
//! and the wall-clock durations of its parts), paths (the store folder, the
//! leaf, the configuration and model files as given), labels and the build
//! (the invocation's label, the configuration file's stem, the camdl version,
//! none of which is in a result's record), and comparisons with other runs
//! chosen when the summary is rendered (the configuration difference from a
//! baseline, the log-likelihood difference from the best row, the stale
//! flag). `fit summary` reads the stored summary and adds those fields back
//! from its own rendering, through [`with_rendered`].
//!
//! The format is `camdl.fit-run-summary/v1`, in the result-file contract of
//! `docs/dev/proposals/2026-11-23-workflow-bindings.md`.

use serde_json::Value;

/// The stored summary's name in a fit run.
pub(crate) const FILE: &str = "summary.json";

/// The stored summary's `schema` value.
pub(crate) const SCHEMA: &str = "camdl.fit-run-summary/v1";

/// The fields of a method's report that are added when it is rendered, as
/// JSON pointers into the report.
const RENDERED_REPORT: &[&str] = &[
    // The build label, of the report and of the run's IF2 record.
    "/camdl_version",
    "/method_result/record/camdl_version",
    // A comparison of the build that made the run with the one reading it.
    "/provenance/stale_camdl_version",
    // Paths: the configuration and model as given, and the run's folder,
    // which the first writer's labels name.
    "/header/config",
    "/header/model",
    "/header/output",
    // The leaf's path and the run's wall time.
    "/stored",
    // Wall-clock durations.
    "/method_result/cost/ms_per_evaluation",
    "/method_result/cost/sampling_seconds",
    "/method_result/sampling_seconds",
    "/method_result/diagnostics/wall_time_secs",
];

/// The fields of a run's `fit table` row that are added when it is rendered,
/// as JSON pointers into the row.
const RENDERED_ROW: &[&str] = &[
    "/created_at",
    "/age_seconds",
    "/ess_per_sec",
    "/label",
    "/stem",
    "/config_diff_from_baseline",
    "/delta_ll_vs_best",
    "/stale",
    "/stale_reason",
];

/// The stored summary of a run whose report and table row are `report` and
/// `row`, each as `fit summary --format json` writes it.
pub(crate) fn stored(report: &Value, row: Option<&Value>) -> Value {
    let mut report = report.clone();
    remove_all(&mut report, RENDERED_REPORT);
    let row = row.map(|r| {
        let mut r = r.clone();
        remove_all(&mut r, RENDERED_ROW);
        r
    });
    serde_json::json!({ "schema": SCHEMA, "report": report, "table_row": row })
}

/// The bytes of [`stored`] as the run writes them.
pub(crate) fn to_bytes(summary: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(summary).expect("a JSON value encodes");
    bytes.push(b'\n');
    bytes
}

/// The stored summary read back from its bytes. Refused when the bytes are
/// not JSON or name another schema.
pub(crate) fn parse(bytes: &[u8]) -> Result<Value, String> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("{FILE} is not JSON: {e}"))?;
    match v.get("schema").and_then(Value::as_str) {
        Some(SCHEMA) => Ok(v),
        other => Err(format!("{FILE} has schema {other:?}, where {SCHEMA:?} is read")),
    }
}

/// The report and row of a stored summary with the rendered fields of
/// `report` and `row` added: the entry `fit summary --format json` writes for
/// the run.
pub(crate) fn with_rendered(summary: &Value, report: &Value, row: Option<&Value>) -> (Value, Option<Value>) {
    let mut out = summary.get("report").cloned().unwrap_or(Value::Null);
    copy_all(&mut out, report, RENDERED_REPORT);
    let out_row = match (summary.get("table_row").filter(|v| !v.is_null()), row) {
        (Some(stored), Some(fresh)) => {
            let mut r = stored.clone();
            copy_all(&mut r, fresh, RENDERED_ROW);
            Some(r)
        }
        (Some(stored), None) => Some(stored.clone()),
        (None, _) => None,
    };
    (out, out_row)
}

fn split(pointer: &str) -> (&str, &str) {
    pointer.rsplit_once('/').expect("a pointer starts with '/'")
}

fn remove_all(v: &mut Value, pointers: &[&str]) {
    for p in pointers {
        let (parent, key) = split(p);
        let target = if parent.is_empty() { Some(&mut *v) } else { v.pointer_mut(parent) };
        if let Some(Value::Object(m)) = target {
            m.remove(key);
        }
    }
}

/// Sets each of `pointers` in `to` to its value in `from`, where `from` holds
/// it and `to` holds its parent.
fn copy_all(to: &mut Value, from: &Value, pointers: &[&str]) {
    for p in pointers {
        let Some(value) = from.pointer(p) else { continue };
        let (parent, key) = split(p);
        let target = if parent.is_empty() { Some(&mut *to) } else { to.pointer_mut(parent) };
        if let Some(Value::Object(m)) = target {
            m.insert(key.to_string(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_stored_summary_leaves_out_rendered_fields_and_rendering_adds_them_back() {
        let report = json!({
            "method": "mh", "best_loglik": -48.1, "camdl_version": "0.1.0+abc",
            "header": { "config": "/a/fit.toml", "model": "/a/sir.ir.json", "output": "x/y", "estimate": ["beta"] },
            "stored": { "leaf": "/a/b/", "total_seconds": 0.3 },
            "method_result": { "cost": { "evaluations": 120, "ms_per_evaluation": 0.3, "sampling_seconds": 0.02 },
                               "diagnostics": { "n_chains": 2, "wall_time_secs": 0.26 } },
        });
        let row = json!({ "best_loglik": -48.1, "created_at": "2026-11-23T00:00:00Z", "age_seconds": 4,
                          "label": "mh", "stem": "fit_mh", "config_diff_from_baseline": {}, "delta_ll_vs_best": 0.0,
                          "ess_per_sec": null, "stale": false, "stale_reason": null, "run_id": "f5" });
        let s = stored(&report, Some(&row));
        assert_eq!(s, json!({
            "schema": SCHEMA,
            "report": {
                "method": "mh", "best_loglik": -48.1,
                "header": { "estimate": ["beta"] },
                "method_result": { "cost": { "evaluations": 120 }, "diagnostics": { "n_chains": 2 } },
            },
            "table_row": { "best_loglik": -48.1, "run_id": "f5" },
        }));
        assert_eq!(parse(&to_bytes(&s)).unwrap(), s);
        let (r, w) = with_rendered(&s, &report, Some(&row));
        assert_eq!(r, report);
        assert_eq!(w, Some(row));
    }

    #[test]
    fn a_summary_of_another_schema_is_refused() {
        let e = parse(br#"{"schema": "camdl.fit-run-summary/v9"}"#).unwrap_err();
        assert!(e.contains("v9") && e.contains(SCHEMA), "{e}");
    }
}
