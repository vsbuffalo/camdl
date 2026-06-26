//! Generated-quantities output: banding + tidy-TSV rendering, shared by
//! `fit predict` (banded posterior/draws) and `simulate` (banded for `--draws`,
//! point for a single fixed-params run). The pure evaluator lives in
//! `sim::quantity`; this module turns its per-draw [`sim::quantity::QuantityResult`]s
//! into `quantities/<name>.tsv` sidecars + a `quantities.json` manifest.
//!
//! Two rendering modes:
//!   - [`Mode::Banded`] — one column per draw is reduced to a quantile band
//!     (`n_draws` / `q05…q95`, plus the censoring trio for a `Time` scalar). The
//!     predict path and `simulate --draws` use this.
//!   - [`Mode::Point`] — a single fixed-params run is ONE realization, so a leaf
//!     writes a bare `value` (a series writes `time … value`; a scalar writes
//!     `… value`; a censored `Time` scalar writes `value = NA`). No banding.
//!
//! The manifest is mode-independent (it describes each quantity's shape / source
//! / reduction / unit / censorability, not the rendering).

use indexmap::IndexMap;

use crate::fit::predict::{band, fmt_time, fmt_value, QUANTILE_LEVELS};

/// Banded (predict, `simulate --draws`) vs point (a single fixed-params
/// `simulate` run). Keyed by the param-source kind, never the cell count alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Banded,
    Point,
}

/// A scalar quantity leaf's banding result: either a real band over the finite
/// per-draw values (carrying the censored count), or every draw censored (no
/// band — `q*` rendered empty). Series quantities never reach here.
#[derive(Debug, Clone, PartialEq)]
enum BandResult {
    Banded { bands: Vec<f64>, n_value: usize, n_censored: usize },
    AllCensored { n_draws: usize },
}

/// Partition a scalar quantity's per-draw values into the finite set and the
/// censored count, band the finite set (reusing [`band`], which rejects a
/// non-finite value so a `NaN`/±∞ arithmetic result surfaces as an error rather
/// than a published band), and report the counts. The all-censored case returns
/// [`BandResult::AllCensored`] instead of calling `band(&[])`.
fn band_with_censoring(vals: &[sim::quantity::QuantityDrawValue]) -> Result<BandResult, String> {
    use sim::quantity::QuantityDrawValue;
    let mut finite: Vec<f64> = Vec::new();
    let mut n_censored = 0usize;
    for v in vals {
        match v {
            QuantityDrawValue::Value(x) => finite.push(*x),
            QuantityDrawValue::Censored => n_censored += 1,
        }
    }
    if finite.is_empty() {
        Ok(BandResult::AllCensored { n_draws: n_censored })
    } else {
        let bands = band(&finite)?;
        Ok(BandResult::Banded { bands, n_value: finite.len(), n_censored })
    }
}

/// A quantity leaf's output shape — a function of its IR body, so the TSV header
/// is deterministic (never data-dependent). A reduction-less `Reduced` is a
/// series; a `Time` reduction is a censorable scalar; any other reduction or a
/// `Derived` is a plain scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QShape {
    Series,
    ScalarPlain,
    ScalarCensorable,
}

impl QShape {
    fn of(body: &ir::quantity::QuantityBody) -> QShape {
        use ir::quantity::{QuantityBody, TemporalReduce};
        match body {
            QuantityBody::Reduced { reduce: None, .. } => QShape::Series,
            QuantityBody::Reduced { reduce: Some(TemporalReduce::Time(_)), .. } => {
                QShape::ScalarCensorable
            }
            QuantityBody::Reduced { reduce: Some(_), .. } => QShape::ScalarPlain,
            QuantityBody::Derived(_) => QShape::ScalarPlain,
        }
    }
    fn is_series(self) -> bool {
        matches!(self, QShape::Series)
    }
    fn manifest_shape(self) -> &'static str {
        if self.is_series() { "series" } else { "scalar" }
    }
}

/// The reduction's manifest name (the IR temporal reduction → its DSL spelling).
fn reduce_name(r: &ir::quantity::TemporalReduce) -> &'static str {
    use ir::quantity::{TemporalReduce, TimeReduce, ValueReduce};
    match r {
        TemporalReduce::Value(ValueReduce::Final) => "final",
        TemporalReduce::Value(ValueReduce::Max) => "max",
        TemporalReduce::Value(ValueReduce::Min) => "min",
        TemporalReduce::Value(ValueReduce::Mean) => "mean",
        TemporalReduce::Value(ValueReduce::CountAbove(_)) => "count_above",
        TemporalReduce::Value(ValueReduce::CountBelow(_)) => "count_below",
        TemporalReduce::Time(TimeReduce::TimeOfMax) => "time_of_max",
        TemporalReduce::Time(TimeReduce::TimeOfMin) => "time_of_min",
        TemporalReduce::Time(TimeReduce::FirstAbove(_)) => "first_above",
        TemporalReduce::Time(TimeReduce::FirstBelow(_)) => "first_below",
        TemporalReduce::Time(TimeReduce::LastAbove(_)) => "last_above",
        TemporalReduce::Time(TimeReduce::LastBelow(_)) => "last_below",
        TemporalReduce::Integral => "integral",
    }
}

/// The banded TSV header — a deterministic function of `(shape, stratified)`.
/// Every shape carries `n_draws` + the quantile columns; a series prepends
/// `time`; a stratified leaf inserts its `<dims…>`; a censorable scalar inserts
/// the censoring trio.
fn quantity_header(shape: QShape, dims: &[String]) -> String {
    let mut cols: Vec<String> = Vec::new();
    if shape.is_series() {
        cols.push("time".to_string());
    }
    cols.extend(dims.iter().cloned());
    cols.push("n_draws".to_string());
    if shape == QShape::ScalarCensorable {
        cols.push("n_value".to_string());
        cols.push("n_censored".to_string());
        cols.push("p_censored".to_string());
    }
    for (_, label) in QUANTILE_LEVELS {
        cols.push((*label).to_string());
    }
    cols.join("\t")
}

/// The point TSV header — a single realization, so a bare `value` column. A
/// series prepends `time`; a stratified leaf inserts its `<dims…>`. No `n_draws`,
/// no quantiles, no censoring trio (a censored `Time` scalar writes `value = NA`).
fn point_header(shape: QShape, dims: &[String]) -> String {
    let mut cols: Vec<String> = Vec::new();
    if shape.is_series() {
        cols.push("time".to_string());
    }
    cols.extend(dims.iter().cloned());
    cols.push("value".to_string());
    cols.join("\t")
}

/// Names of scalar quantities a `ScalarExpr` references via `QRef`.
fn collect_qrefs(se: &ir::quantity::ScalarExpr, out: &mut Vec<String>) {
    use ir::quantity::ScalarExpr::*;
    match se {
        Const(_) | Param(_) => {}
        QRef(q) => out.push(q.name.clone()),
        UnOp { arg, .. } => collect_qrefs(arg, out),
        BinOp { left, right, .. } => {
            collect_qrefs(left, out);
            collect_qrefs(right, out);
        }
        Cond { pred, then, else_ } => {
            collect_qrefs(pred, out);
            collect_qrefs(then, out);
            collect_qrefs(else_, out);
        }
    }
}

/// Render the accumulated per-draw quantity values into one tidy TSV per logical
/// quantity plus a `quantities.json` manifest. Leaves are grouped by base `name`
/// (a stratified quantity expands to one leaf per cell); within a group each
/// scalar leaf is a row and each series leaf a rowset (one row per snapshot time).
/// A series' time axis is the trajectory snapshot grid (every draw shares it).
///
/// `mode` selects banded (one column per draw → a quantile band) or point (one
/// realization → a bare `value`).
pub(crate) fn render_quantities(
    quantities: &[ir::quantity::Quantity],
    quant_draws: &[Vec<sim::quantity::QuantityResult>],
    snapshot_times: &[f64],
    mode: Mode,
) -> Result<(Vec<(String, String)>, String), String> {
    use ir::quantity::{QuantityBody, TemporalReduce};

    let n_draws = quant_draws.len();
    if mode == Mode::Point && n_draws != 1 {
        return Err(format!(
            "point-mode quantities require exactly one realization, got {n_draws} \
             (a multi-cell run must band)"
        ));
    }

    // Group leaf indices by base name, first-appearance order.
    let mut order: Vec<String> = Vec::new();
    let mut groups: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (gi, q) in quantities.iter().enumerate() {
        groups
            .entry(q.name.clone())
            .or_insert_with(|| {
                order.push(q.name.clone());
                Vec::new()
            })
            .push(gi);
    }

    // Transitive censorability: a quantity is censorable if it is a `Time`
    // reduction, OR a `Derived` whose reduction arithmetic references (via QRef)
    // a censorable quantity. So `outbreak_dur = fadeout - takeoff` (a Derived
    // over two `first_above`/`last_above` scalars) reports n_censored when an
    // operand never fired, rather than silently dropping those draws under a
    // plain-scalar header. QRefs are backward, so one ordered pass suffices.
    let mut censorable: std::collections::HashSet<String> = std::collections::HashSet::new();
    for q in quantities {
        let is_cens = match &q.body {
            QuantityBody::Reduced { reduce: Some(TemporalReduce::Time(_)), .. } => true,
            QuantityBody::Derived(se) => {
                let mut refs = Vec::new();
                collect_qrefs(se, &mut refs);
                refs.iter().any(|n| censorable.contains(n))
            }
            _ => false,
        };
        if is_cens {
            censorable.insert(q.name.clone());
        }
    }

    let mut outputs: Vec<(String, String)> = Vec::new();
    let mut manifest_entries: Vec<serde_json::Value> = Vec::new();

    for name in &order {
        let leaf_idxs = &groups[name];
        let first = &quantities[leaf_idxs[0]];
        // A Derived that transitively references a Time scalar is censorable.
        let shape = match QShape::of(&first.body) {
            QShape::ScalarPlain
                if matches!(first.body, QuantityBody::Derived(_)) && censorable.contains(name) =>
            {
                QShape::ScalarCensorable
            }
            s => s,
        };
        let dims: Vec<String> = first.stratum.iter().map(|k| k.dim.clone()).collect();

        let mut out = String::new();
        match mode {
            Mode::Banded => out.push_str(&quantity_header(shape, &dims)),
            Mode::Point => out.push_str(&point_header(shape, &dims)),
        }
        out.push('\n');

        for &gi in leaf_idxs {
            let levels: Vec<String> =
                quantities[gi].stratum.iter().map(|k| k.level.clone()).collect();
            match mode {
                Mode::Banded => render_banded_leaf(
                    name, gi, shape, &levels, n_draws, quant_draws, snapshot_times, &mut out,
                )?,
                Mode::Point => render_point_leaf(
                    name, gi, shape, &levels, quant_draws, snapshot_times, &mut out,
                )?,
            }
        }

        // Manifest entry for this logical quantity (one per group) — mode-independent.
        let source = match &first.body {
            QuantityBody::Reduced { .. } => "state",
            QuantityBody::Derived(_) => "derived",
        };
        let reduce_val: serde_json::Value = match &first.body {
            QuantityBody::Reduced { reduce: Some(r), .. } => {
                serde_json::Value::String(reduce_name(r).to_string())
            }
            _ => serde_json::Value::Null,
        };
        let censoring: serde_json::Value = if shape == QShape::ScalarCensorable {
            serde_json::json!({ "kind": "right", "conditional_quantiles": true })
        } else {
            serde_json::Value::Null
        };
        manifest_entries.push(serde_json::json!({
            "name": name,
            "shape": shape.manifest_shape(),
            "source": source,
            "index_dims": dims,
            "reduce": reduce_val,
            // The dim → unit renderer is a later phase; the field is present now.
            "unit": serde_json::Value::Null,
            "censoring": censoring,
        }));

        outputs.push((name.clone(), out));
    }

    let manifest = serde_json::json!({
        "schema": "camdl.quantities/v1",
        "quantities": manifest_entries,
    });
    let manifest_str = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("serializing quantities manifest: {e}"))?;
    Ok((outputs, manifest_str))
}

/// Banded rendering of one leaf (one column per draw → a quantile band).
fn render_banded_leaf(
    name: &str,
    gi: usize,
    shape: QShape,
    levels: &[String],
    n_draws: usize,
    quant_draws: &[Vec<sim::quantity::QuantityResult>],
    snapshot_times: &[f64],
    out: &mut String,
) -> Result<(), String> {
    use sim::quantity::{QuantityDrawValue, QuantityResult};
    if shape.is_series() {
        // Validate the series shape/length once for this leaf.
        for (di, draw) in quant_draws.iter().enumerate() {
            match draw.get(gi) {
                Some(QuantityResult::Series(v)) if v.len() == snapshot_times.len() => {}
                Some(QuantityResult::Series(v)) => {
                    return Err(format!(
                        "quantity '{name}': draw {di} has a series of length {} but the \
                         trajectory has {} snapshots",
                        v.len(),
                        snapshot_times.len()
                    ))
                }
                _ => return Err(format!("quantity '{name}': expected a series value")),
            }
        }
        for (ti, &t) in snapshot_times.iter().enumerate() {
            let col: Vec<f64> = quant_draws
                .iter()
                .map(|draw| match &draw[gi] {
                    QuantityResult::Series(v) => v[ti],
                    QuantityResult::Scalar(_) => f64::NAN,
                })
                .collect();
            let bands = band(&col).map_err(|e| format!("quantity '{name}' at t={}: {e}", fmt_time(t)))?;
            let mut cells: Vec<String> = Vec::with_capacity(2 + levels.len() + bands.len());
            cells.push(fmt_time(t));
            cells.extend(levels.iter().cloned());
            cells.push(n_draws.to_string());
            cells.extend(bands.iter().map(|b| fmt_value(*b)));
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
    } else {
        let vals: Vec<QuantityDrawValue> = quant_draws
            .iter()
            .map(|draw| match &draw[gi] {
                QuantityResult::Scalar(v) => *v,
                QuantityResult::Series(_) => QuantityDrawValue::Value(f64::NAN),
            })
            .collect();
        let (n_value, n_censored, bands_opt) =
            match band_with_censoring(&vals).map_err(|e| format!("quantity '{name}': {e}"))? {
                BandResult::Banded { bands, n_value, n_censored } => (n_value, n_censored, Some(bands)),
                BandResult::AllCensored { n_draws: nd } => (0usize, nd, None),
            };
        let total = n_value + n_censored;
        let mut cells: Vec<String> = Vec::new();
        cells.extend(levels.iter().cloned());
        cells.push(total.to_string());
        if shape == QShape::ScalarCensorable {
            cells.push(n_value.to_string());
            cells.push(n_censored.to_string());
            let p = if total > 0 { n_censored as f64 / total as f64 } else { 0.0 };
            cells.push(fmt_value(p));
        }
        match &bands_opt {
            Some(bands) => cells.extend(bands.iter().map(|b| fmt_value(*b))),
            None => cells.extend(QUANTILE_LEVELS.iter().map(|_| String::new())),
        }
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    Ok(())
}

/// Point rendering of one leaf — a single realization writes a bare `value`. A
/// censored `Time` scalar writes `value = NA`; a non-finite arithmetic value is
/// an error (a `NaN`/±∞ is an upstream bug, mirroring the banded path's `band`).
fn render_point_leaf(
    name: &str,
    gi: usize,
    shape: QShape,
    levels: &[String],
    quant_draws: &[Vec<sim::quantity::QuantityResult>],
    snapshot_times: &[f64],
    out: &mut String,
) -> Result<(), String> {
    use sim::quantity::{QuantityDrawValue, QuantityResult};
    // Point mode is guarded to exactly one realization in `render_quantities`.
    let draw = quant_draws
        .first()
        .ok_or_else(|| format!("quantity '{name}': point mode requires one realization, got none"))?;
    if shape.is_series() {
        let series = match draw.get(gi) {
            Some(QuantityResult::Series(v)) if v.len() == snapshot_times.len() => v,
            Some(QuantityResult::Series(v)) => {
                return Err(format!(
                    "quantity '{name}': series of length {} but the trajectory has {} snapshots",
                    v.len(),
                    snapshot_times.len()
                ))
            }
            _ => return Err(format!("quantity '{name}': expected a series value")),
        };
        for (ti, &t) in snapshot_times.iter().enumerate() {
            let v = series[ti];
            if !v.is_finite() {
                return Err(format!(
                    "quantity '{name}' at t={}: non-finite series value (a NaN/±∞ is an upstream bug)",
                    fmt_time(t)
                ));
            }
            let mut cells: Vec<String> = Vec::with_capacity(1 + levels.len() + 1);
            cells.push(fmt_time(t));
            cells.extend(levels.iter().cloned());
            cells.push(fmt_value(v));
            out.push_str(&cells.join("\t"));
            out.push('\n');
        }
    } else {
        let v = match draw.get(gi) {
            Some(QuantityResult::Scalar(v)) => *v,
            _ => return Err(format!("quantity '{name}': expected a scalar value")),
        };
        let value_cell = match v {
            QuantityDrawValue::Value(x) if x.is_finite() => fmt_value(x),
            QuantityDrawValue::Value(_) => {
                return Err(format!(
                    "quantity '{name}': non-finite scalar value (a NaN/±∞ arithmetic result is a \
                     bug, not censoring)"
                ))
            }
            QuantityDrawValue::Censored => "NA".to_string(),
        };
        let mut cells: Vec<String> = Vec::with_capacity(levels.len() + 1);
        cells.extend(levels.iter().cloned());
        cells.push(value_cell);
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_with_censoring_partitions_finite_and_censored() {
        use sim::quantity::QuantityDrawValue::{Censored, Value};
        // Mixed finite + censored: band the finite set, report both counts.
        let vals = vec![Value(1.0), Value(2.0), Censored, Value(3.0), Censored];
        match band_with_censoring(&vals).unwrap() {
            BandResult::Banded { bands, n_value, n_censored } => {
                assert_eq!(n_value, 3, "three finite draws banded");
                assert_eq!(n_censored, 2, "two censored draws counted, not banded");
                assert_eq!(bands.len(), 5, "five quantile levels");
                assert_eq!(bands[2], 2.0, "median of {{1,2,3}} is 2");
            }
            other => panic!("expected Banded, got {other:?}"),
        }
        // All censored: no band (never band(&[])), the draw count carried.
        match band_with_censoring(&[Censored, Censored, Censored]).unwrap() {
            BandResult::AllCensored { n_draws } => assert_eq!(n_draws, 3),
            other => panic!("expected AllCensored, got {other:?}"),
        }
        // A NaN `Value` is a non-finite arithmetic result, not censoring → a hard
        // error (the reused `band` rejects it).
        assert!(band_with_censoring(&[Value(1.0), Value(f64::NAN)]).is_err());
    }

    #[test]
    fn quantity_header_is_a_function_of_shape_and_dims() {
        assert_eq!(
            quantity_header(QShape::Series, &[]),
            "time\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        assert_eq!(
            quantity_header(QShape::Series, &["patch".to_string()]),
            "time\tpatch\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        assert_eq!(
            quantity_header(QShape::ScalarPlain, &[]),
            "n_draws\tq05\tq25\tq50\tq75\tq95"
        );
        // The censoring trio sits between n_draws and the quantiles, after the dims.
        assert_eq!(
            quantity_header(QShape::ScalarCensorable, &["patch".to_string()]),
            "patch\tn_draws\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95"
        );
    }

    #[test]
    fn point_header_is_a_bare_value_column() {
        // Series: time + value (no n_draws / quantiles).
        assert_eq!(point_header(QShape::Series, &[]), "time\tvalue");
        assert_eq!(point_header(QShape::Series, &["patch".to_string()]), "time\tpatch\tvalue");
        // Scalar: just value; a censorable scalar gets NO censoring trio (it
        // writes `value = NA` instead).
        assert_eq!(point_header(QShape::ScalarPlain, &[]), "value");
        assert_eq!(point_header(QShape::ScalarCensorable, &["patch".to_string()]), "patch\tvalue");
    }

    #[test]
    fn point_render_writes_bare_values_and_na_for_censored() {
        use ir::observation::StratumKey;
        use ir::quantity::{Quantity, QuantityBody, QuantitySource, TemporalReduce, TimeReduce};
        use sim::quantity::{QuantityDrawValue, QuantityResult};

        // Three quantities: a series, a value scalar, a censored time scalar.
        let q = |name: &str, body: QuantityBody| Quantity {
            name: name.to_string(),
            stratum: Vec::<StratumKey>::new(),
            body,
        };
        let quantities = vec![
            q("prevalence", QuantityBody::Reduced {
                source: QuantitySource::State(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.0 })),
                reduce: None,
            }),
            q("onset", QuantityBody::Reduced {
                source: QuantitySource::State(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.0 })),
                reduce: Some(TemporalReduce::Time(TimeReduce::TimeOfMax)),
            }),
        ];
        // One realization: prevalence is a 2-point series; onset is censored.
        let draws = vec![vec![
            QuantityResult::Series(vec![0.1, 0.4]),
            QuantityResult::Scalar(QuantityDrawValue::Censored),
        ]];
        let times = vec![0.0, 7.0];
        let (outs, _manifest) =
            render_quantities(&quantities, &draws, &times, Mode::Point).unwrap();

        let prev = &outs.iter().find(|(n, _)| n == "prevalence").unwrap().1;
        let plines: Vec<&str> = prev.trim_end().lines().collect();
        assert_eq!(plines[0], "time\tvalue", "series point header");
        assert_eq!(plines[1], "0\t0.1");
        assert_eq!(plines[2], "7\t0.4");

        let onset = &outs.iter().find(|(n, _)| n == "onset").unwrap().1;
        let olines: Vec<&str> = onset.trim_end().lines().collect();
        assert_eq!(olines[0], "value", "scalar point header");
        assert_eq!(olines[1], "NA", "a censored time scalar writes NA, not a fabricated value");
    }

    #[test]
    fn point_mode_rejects_multiple_realizations() {
        let draws: Vec<Vec<sim::quantity::QuantityResult>> = vec![vec![], vec![]];
        let err = render_quantities(&[], &draws, &[], Mode::Point).unwrap_err();
        assert!(err.contains("exactly one realization"), "got: {err}");
    }
}
