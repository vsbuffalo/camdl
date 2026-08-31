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

use crate::quantile::{band, fmt_time, fmt_value, QUANTILE_LEVELS};

/// Banded (predict, `simulate --draws`) vs point (a single fixed-params
/// `simulate` run). Keyed by the param-source kind, never the cell count alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Banded,
    Point,
}

/// The `fit predict` design-overlay coordinates a quantity output is tagged
/// with: the scenario overlay axis ([`DesignCoords::scenario`]) and the sweep
/// grid coordinate ([`DesignCoords::sweep`], one `(param, value)` per swept
/// parameter, in sorted-name order). A leading `scenario` column precedes the
/// `sweep:<param>` columns, which precede everything else.
///
/// `simulate` passes [`DesignCoords::none`] (no scenario column, no sweep
/// columns), keeping its output byte-identical.
#[derive(Clone, Copy)]
pub(crate) struct DesignCoords<'a> {
    /// The scenario this design cell belongs to. `None` means the run has **no
    /// scenario axis** — not "do not label these cells". The distinction is the
    /// whole of gh#562: the old `none()` constructor meant the latter, and was
    /// passed by a caller that had pooled several scenarios, so the renderer
    /// labelled a mixture as if it were one arm. Every caller now derives this
    /// from the cell it is rendering.
    pub scenario: Option<&'a str>,
    pub sweep: &'a [(String, f64)],
}

/// Which object a design cell's quantities were folded over, per leaf, plus how
/// many of its draws had a conditioned path to fold (gh#722).
///
/// `fit predict` always supplies one so EVERY manifest entry says which object
/// its numbers came from — the fitted arm's `value_at(N0 - S, last_obs)` is a
/// conditioned read and a scenario arm's is a free-forward replay, and the two
/// sit in one file under a `scenario` column. `simulate` supplies `None` (it
/// has no fit behind it), keeping its manifest byte-identical.
#[derive(Clone, Copy)]
pub(crate) struct EvaluatedOn<'a> {
    /// Per leaf, in `quantities` order.
    pub paths: &'a [sim::quantity::QuantityPath],
    /// Draws in this design cell whose saved smoothing path was opened. The
    /// honest denominator for a `smoothed` leaf: the rest are censored, and the
    /// forkable subset is REPORTED, never silently substituted.
    pub n_conditioned: usize,
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
        use ir::quantity::{QuantityBody, TemporalReduce, ValueReduce};
        match body {
            QuantityBody::Reduced { reduce: None, .. } => QShape::Series,
            QuantityBody::Reduced { reduce: Some(TemporalReduce::Time(_)), .. } => {
                QShape::ScalarCensorable
            }
            // value_at censors out-of-window (proposal 2026-08-17) — it needs
            // the censoring trio like a Time scalar, not the plain path.
            QuantityBody::Reduced {
                reduce: Some(TemporalReduce::Value(ValueReduce::ValueAt(_))), ..
            } => QShape::ScalarCensorable,
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
        TemporalReduce::Value(ValueReduce::ValueAt(_)) => "value_at",
        TemporalReduce::Time(TimeReduce::TimeOfMax) => "time_of_max",
        TemporalReduce::Time(TimeReduce::TimeOfMin) => "time_of_min",
        TemporalReduce::Time(TimeReduce::FirstAbove(_)) => "first_above",
        TemporalReduce::Time(TimeReduce::FirstBelow(_)) => "first_below",
        TemporalReduce::Time(TimeReduce::LastAbove(_)) => "last_above",
        TemporalReduce::Time(TimeReduce::LastBelow(_)) => "last_below",
        TemporalReduce::Integral => "integral",
    }
}

/// The manifest `source` tag — what a quantity folds over. Preserves the IR's
/// State-vs-Observation distinction (gh#317): a `Reduced` quantity can reduce
/// latent state (`"state"`) OR a simulated observation stream (`"observations"`,
/// the v1.1 `observations.<stream>` source), and a downstream consumer must be
/// able to tell them apart. Collapsing both to `"state"` was the bug.
fn manifest_source(body: &ir::quantity::QuantityBody) -> &'static str {
    use ir::quantity::{QuantityBody, QuantitySource};
    match body {
        QuantityBody::Reduced { source: QuantitySource::State(_), .. } => "state",
        QuantityBody::Reduced { source: QuantitySource::Observation { .. }, .. } => "observations",
        QuantityBody::Derived(_) => "derived",
    }
}

/// The leading design-overlay columns (`scenario`, then `sweep:<param>` per
/// swept parameter) shared by the banded and point headers. Empty `coords`
/// (the `simulate` path) push nothing → a byte-identical header.
fn push_design_header_cols(cols: &mut Vec<String>, coords: DesignCoords) {
    if coords.scenario.is_some() {
        cols.push("scenario".to_string());
    }
    for (name, _) in coords.sweep {
        cols.push(format!("sweep:{name}"));
    }
}

/// The leading design-overlay cells (the scenario name, then this cell's swept
/// values) shared by every banded/point row. Empty `coords` push nothing.
fn push_design_row_cells(cells: &mut Vec<String>, coords: DesignCoords) {
    if let Some(s) = coords.scenario {
        cells.push(s.to_string());
    }
    for (_, v) in coords.sweep {
        cells.push(fmt_value(*v));
    }
}

/// The banded TSV header — a deterministic function of
/// `(shape, stratified, convergence)`. Every shape carries `n_draws` + the
/// quantile columns; a series prepends `time`; a stratified leaf inserts its
/// `<dims…>`; a censorable scalar inserts the censoring trio; a caller that
/// supplied a chain partition gets `rhat`/`ess` after `n_draws` (gh#794). The
/// `fit predict` design overlay (`scenario`, then `sweep:<param>`) leads
/// everything else.
fn quantity_header(
    shape: QShape,
    dims: &[String],
    coords: DesignCoords,
    convergence: bool,
) -> String {
    let mut cols: Vec<String> = Vec::new();
    push_design_header_cols(&mut cols, coords);
    if shape.is_series() {
        cols.push("time".to_string());
    }
    cols.extend(dims.iter().cloned());
    cols.push("n_draws".to_string());
    if convergence {
        cols.push("rhat".to_string());
        cols.push("ess".to_string());
    }
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
fn point_header(shape: QShape, dims: &[String], coords: DesignCoords) -> String {
    let mut cols: Vec<String> = Vec::new();
    push_design_header_cols(&mut cols, coords);
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
///
/// `coords` is the `fit predict` design overlay: a `Some` scenario prepends a
/// leading `scenario` column to every TSV header + row and a `scenario` field to
/// every manifest entry; a non-empty sweep prepends one `sweep:<param>` column
/// per swept parameter (after the scenario column) and a `sweep` object to the
/// manifest entry (the same way the predictive TSV tags its rows). The
/// `simulate --quantities-out` caller passes [`DesignCoords::none`] (pools all
/// cells into one band) and omits both — today's behaviour, unchanged.
///
/// `chains` is the chain each draw came from, in `quant_draws` order. `Some`
/// adds a per-row `rhat` / `ess` pair to a banded header (gh#794): the R-hat
/// and bulk-ESS of THIS row's reported value, reduced over the same draws the
/// band pools but grouped by chain. `None` — the `simulate` path, which has no
/// chains behind it — adds no columns at all, so that output is byte-identical.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_quantities(
    quantities: &[ir::quantity::Quantity],
    quant_draws: &[Vec<sim::quantity::QuantityResult>],
    snapshot_times: &[f64],
    mode: Mode,
    coords: DesignCoords,
    evaluated_on: Option<EvaluatedOn>,
    chains: Option<&crate::fit::row_convergence::ChainOfDraw>,
    calendar: &io::CalendarMeta,
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
            // value_at censors out-of-window, so a Derived over it must too.
            QuantityBody::Reduced {
                reduce:
                    Some(TemporalReduce::Value(ir::quantity::ValueReduce::ValueAt(_))),
                ..
            } => true,
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
            Mode::Banded => out.push_str(&quantity_header(shape, &dims, coords, chains.is_some())),
            Mode::Point => out.push_str(&point_header(shape, &dims, coords)),
        }
        out.push('\n');

        for &gi in leaf_idxs {
            let levels: Vec<String> =
                quantities[gi].stratum.iter().map(|k| k.level.clone()).collect();
            match mode {
                Mode::Banded => render_banded_leaf(
                    name, gi, shape, &levels, n_draws, quant_draws, snapshot_times, coords,
                    chains, &mut out,
                )?,
                Mode::Point => render_point_leaf(
                    name, gi, shape, &levels, quant_draws, snapshot_times, coords, &mut out,
                )?,
            }
        }

        // Manifest entry for this logical quantity (one per group) — mode-independent.
        let source = manifest_source(&first.body);
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
        let mut entry = serde_json::json!({
            "name": name,
            "shape": shape.manifest_shape(),
            "source": source,
            "index_dims": dims,
            "reduce": reduce_val,
            // The dim → unit renderer is a later phase; the field is present now.
            "unit": serde_json::Value::Null,
            "censoring": censoring,
        });
        // The `fit predict` design overlay: one manifest entry per (quantity,
        // scenario, sweep-cell), tagged so a consumer can group/join by scenario
        // and sweep coordinate. Omitted for the simulate path (`DesignCoords::none`),
        // keeping its manifest byte-identical.
        if let Some(s) = coords.scenario {
            entry["scenario"] = serde_json::Value::String(s.to_string());
        }
        if !coords.sweep.is_empty() {
            let mut sweep_obj = serde_json::Map::new();
            for (param, value) in coords.sweep {
                sweep_obj.insert(
                    param.clone(),
                    serde_json::Value::from(*value),
                );
            }
            entry["sweep"] = serde_json::Value::Object(sweep_obj);
        }
        // gh#722: which object this cell's numbers came from. Every leaf in a
        // group shares a body shape, so the first leaf's routing is the group's
        // — the same rule `shape` above is read by. `n_conditioned` rides only
        // on a `smoothed` entry, where it is the band's real denominator.
        if let Some(ev) = evaluated_on {
            let path = ev.paths.get(leaf_idxs[0]).copied().unwrap_or(
                sim::quantity::QuantityPath::Replay,
            );
            entry["evaluated_on"] = serde_json::Value::String(path.as_str().to_string());
            if path == sim::quantity::QuantityPath::Smoothed {
                entry["n_conditioned_draws"] = serde_json::Value::from(ev.n_conditioned);
            }
        }
        manifest_entries.push(entry);

        outputs.push((name.clone(), out));
    }

    let manifest = serde_json::json!({
        "schema": "camdl.quantities/v1",
        "calendar": calendar.to_json(),
        "quantities": manifest_entries,
    });
    let manifest_str = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("serializing quantities manifest: {e}"))?;
    Ok((outputs, manifest_str))
}

// ── Stacking design cells into one file set ────────────────────────────────

/// Accumulates rendered design cells into one file set.
///
/// A quantity artifact holds one header per quantity and the rows of every
/// design cell stacked beneath it. Both verbs need that stacking — `simulate`
/// over its scenarios, `fit predict` over (sweep point × scenario) — and it is
/// the bug-prone part: drop the wrong line and a header lands mid-file; merge
/// the manifests wrongly and a consumer cannot tell the cells apart.
///
/// Fed incrementally rather than taking all cells at once, because `predict`
/// builds its sink inside the sweep loop and cannot hold every cell live at the
/// same time. That is also why a one-shot function would not do: it would
/// absorb only predict's inner (per-scenario) stacking and leave the outer
/// per-sweep-point pass duplicated, moving the logic up a frame rather than
/// removing it.
///
/// Proposal: `docs/dev/proposals/2026-08-11-scenario-banding-in-simulate.md` §3.5.
pub(crate) struct StackedQuantities {
    /// Quantity name → its accumulated file text (header + every cell's rows).
    bodies: IndexMap<String, String>,
    /// One manifest entry per (quantity, design cell), each tagged with its
    /// coordinates so a consumer can group by them.
    manifest_entries: Vec<serde_json::Value>,
    mode: Mode,
}

impl StackedQuantities {
    pub(crate) fn new(mode: Mode) -> Self {
        StackedQuantities { bodies: IndexMap::new(), manifest_entries: Vec::new(), mode }
    }

    /// Render one design cell and stack it. The first cell for a quantity
    /// contributes its header and rows; later cells contribute rows only.
    pub(crate) fn push_group(
        &mut self,
        quantities: &[ir::quantity::Quantity],
        coords: DesignCoords<'_>,
        draws: &[Vec<sim::quantity::QuantityResult>],
        times: &[f64],
        evaluated_on: Option<EvaluatedOn<'_>>,
        chains: Option<&crate::fit::row_convergence::ChainOfDraw>,
        calendar: &io::CalendarMeta,
    ) -> Result<(), String> {
        let (outs, manifest) = render_quantities(
            quantities, draws, times, self.mode, coords, evaluated_on, chains, calendar,
        )?;
        for (name, content) in outs {
            match self.bodies.entry(name) {
                indexmap::map::Entry::Vacant(e) => {
                    e.insert(content);
                }
                indexmap::map::Entry::Occupied(mut e) => {
                    // Drop the repeated header line so every cell stacks under
                    // the first one.
                    let body: String = content.split_inclusive('\n').skip(1).collect();
                    e.get_mut().push_str(&body);
                }
            }
        }
        let m: serde_json::Value = serde_json::from_str(&manifest)
            .map_err(|e| format!("parsing quantities manifest: {e}"))?;
        if let Some(arr) = m["quantities"].as_array() {
            self.manifest_entries.extend(arr.iter().cloned());
        }
        Ok(())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// The stacked files plus one merged manifest.
    ///
    /// The manifest carries the calendar block. `fit predict` previously built
    /// its own merged document with `schema` + `quantities` only, so a consumer
    /// of a predict quantities manifest could not map the numeric `time` column
    /// to dates without re-parsing the model — the one thing the block exists to
    /// prevent (proposal §6).
    pub(crate) fn finish(
        self,
        calendar: &io::CalendarMeta,
    ) -> Result<(Vec<(String, String)>, String), String> {
        let merged = serde_json::json!({
            "schema": "camdl.quantities/v1",
            "calendar": calendar.to_json(),
            "quantities": self.manifest_entries,
        });
        let manifest = serde_json::to_string_pretty(&merged)
            .map_err(|e| format!("serializing quantities manifest: {e}"))?;
        Ok((self.bodies.into_iter().collect(), manifest))
    }
}

/// Append this row's `rhat` / `ess` cells when a chain partition was supplied
/// (gh#794), and nothing at all when it was not — so a caller with no chains
/// behind it (`simulate`) renders a byte-identical row.
///
/// A quantity has ONE value per draw, so unlike a predictive row there is only
/// one reduction to report. What that value carries depends on the quantity: a
/// `state` or `derived` quantity is a deterministic function of the trajectory
/// and the parameters, so its R-hat is the undiluted kind; a quantity whose
/// manifest `source` is `observations` reduces sampled `y_rep` and therefore
/// carries observation noise, which makes its R-hat the diluted kind (the
/// `rhat_pred` flavour, see `fit predict --help`).
fn push_row_convergence(
    cells: &mut Vec<String>,
    chains: Option<&crate::fit::row_convergence::ChainOfDraw>,
    values: &[f64],
) {
    use crate::fit::row_convergence::{row_convergence, RowConvergence};
    let Some(chains) = chains else { return };
    let c = row_convergence(values, chains);
    cells.push(RowConvergence::rhat_cell(c.as_ref()));
    cells.push(RowConvergence::ess_cell(c.as_ref()));
}

/// The finite values of a scalar quantity's draws, with the chain ids of the
/// SAME draws — the two filtered in one pass so they cannot drift apart.
fn finite_with_chains(
    vals: &[sim::quantity::QuantityDrawValue],
    chains: Option<&crate::fit::row_convergence::ChainOfDraw>,
) -> (Vec<f64>, Option<crate::fit::row_convergence::ChainOfDraw>) {
    use sim::quantity::QuantityDrawValue;
    let mut finite = Vec::new();
    let mut ids = Vec::new();
    for (i, v) in vals.iter().enumerate() {
        if let QuantityDrawValue::Value(x) = v {
            finite.push(*x);
            ids.push(chains.and_then(|c| c.0.get(i).copied()).flatten());
        }
    }
    (finite, chains.map(|_| crate::fit::row_convergence::ChainOfDraw(ids)))
}

/// Banded rendering of one leaf (one column per draw → a quantile band). Every
/// row is prefixed with the design overlay cells (`scenario`, then this cell's
/// swept values) — empty `coords` prefix nothing.
fn render_banded_leaf(
    name: &str,
    gi: usize,
    shape: QShape,
    levels: &[String],
    n_draws: usize,
    quant_draws: &[Vec<sim::quantity::QuantityResult>],
    snapshot_times: &[f64],
    coords: DesignCoords,
    chains: Option<&crate::fit::row_convergence::ChainOfDraw>,
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
            let mut cells: Vec<String> = Vec::with_capacity(3 + levels.len() + bands.len());
            push_design_row_cells(&mut cells, coords);
            cells.push(fmt_time(t));
            cells.extend(levels.iter().cloned());
            cells.push(n_draws.to_string());
            push_row_convergence(&mut cells, chains, &col);
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
        push_design_row_cells(&mut cells, coords);
        cells.extend(levels.iter().cloned());
        cells.push(total.to_string());
        // The reduction runs over the SAME draws the band was built from — the
        // finite ones — with each draw's chain carried alongside, so a censored
        // draw drops out of the value list and its chain id with it. Dropping
        // one side only would pair a value with another draw's chain.
        if chains.is_some() {
            let (finite, finite_chains) = finite_with_chains(&vals, chains);
            push_row_convergence(&mut cells, finite_chains.as_ref(), &finite);
        }
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
    coords: DesignCoords,
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
            let mut cells: Vec<String> = Vec::with_capacity(2 + levels.len() + 1);
            push_design_row_cells(&mut cells, coords);
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
        let mut cells: Vec<String> = Vec::with_capacity(levels.len() + 2);
        push_design_row_cells(&mut cells, coords);
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

    /// An anchored calendar for the render tests. The manifest carries these
    /// verbatim; the render output itself is calendar-independent.
    fn test_cal() -> io::CalendarMeta {
        io::CalendarMeta {
            time_unit: "days".into(),
            origin: Some("2020-01-01".into()),
            days_per_unit: 1.0,
        }
    }

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
        let none = DesignCoords { scenario: None, sweep: &[] };
        assert_eq!(
            quantity_header(QShape::Series, &[], none, false),
            "time\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        assert_eq!(
            quantity_header(QShape::Series, &["patch".to_string()], none, false),
            "time\tpatch\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        assert_eq!(
            quantity_header(QShape::ScalarPlain, &[], none, false),
            "n_draws\tq05\tq25\tq50\tq75\tq95"
        );
        // The censoring trio sits between n_draws and the quantiles, after the dims.
        assert_eq!(
            quantity_header(QShape::ScalarCensorable, &["patch".to_string()], none, false),
            "patch\tn_draws\tn_value\tn_censored\tp_censored\tq05\tq25\tq50\tq75\tq95"
        );
        // With the scenario overlay column: it leads everything else.
        let scen = DesignCoords { scenario: Some("with_sia"), sweep: &[] };
        assert_eq!(
            quantity_header(QShape::Series, &["patch".to_string()], scen, false),
            "scenario\ttime\tpatch\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        // With a scenario AND a sweep: scenario leads, then one sweep:<param>
        // column per swept parameter, then the rest.
        let sweep = [("k".to_string(), 8.0)];
        let scen_sweep = DesignCoords { scenario: Some("with_sia"), sweep: &sweep };
        assert_eq!(
            quantity_header(QShape::Series, &["patch".to_string()], scen_sweep, false),
            "scenario\tsweep:k\ttime\tpatch\tn_draws\tq05\tq25\tq50\tq75\tq95"
        );
        // gh#794: a caller with a chain partition gets `rhat`/`ess` right after
        // `n_draws`, before the censoring trio and the band.
        assert_eq!(
            quantity_header(QShape::Series, &[], none, true),
            "time\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95"
        );
        assert_eq!(
            quantity_header(QShape::ScalarCensorable, &["patch".to_string()], none, true),
            "patch\tn_draws\trhat\tess\tn_value\tn_censored\tp_censored\
             \tq05\tq25\tq50\tq75\tq95"
        );
    }

    #[test]
    fn point_header_is_a_bare_value_column() {
        let none = DesignCoords { scenario: None, sweep: &[] };
        // Series: time + value (no n_draws / quantiles).
        assert_eq!(point_header(QShape::Series, &[], none), "time\tvalue");
        assert_eq!(point_header(QShape::Series, &["patch".to_string()], none), "time\tpatch\tvalue");
        // Scalar: just value; a censorable scalar gets NO censoring trio (it
        // writes `value = NA` instead).
        assert_eq!(point_header(QShape::ScalarPlain, &[], none), "value");
        assert_eq!(point_header(QShape::ScalarCensorable, &["patch".to_string()], none), "patch\tvalue");
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
            dimension: None,
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
            render_quantities(&quantities, &draws, &times, Mode::Point, DesignCoords { scenario: None, sweep: &[] }, None, None, &test_cal()).unwrap();

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
    fn manifest_source_distinguishes_state_observation_and_derived() {
        // gh#317: the manifest `source` must preserve the IR's State-vs-Observation
        // distinction. Before the fix, every `Reduced` body collapsed to "state",
        // so an `observations.<stream>` reduction was mislabeled — this asserts the
        // observation arm yields "observations", which fails on the old code.
        use ir::expr::{ConstExpr, Expr};
        use ir::quantity::{QuantityBody, QuantitySource, ScalarExpr};

        let state = QuantityBody::Reduced {
            source: QuantitySource::State(Expr::Const(ConstExpr { value: 0.0 })),
            reduce: None,
        };
        let observation = QuantityBody::Reduced {
            source: QuantitySource::Observation { stream: "cases".to_string() },
            reduce: None,
        };
        let derived = QuantityBody::Derived(ScalarExpr::Const(1.0));

        assert_eq!(manifest_source(&state), "state");
        assert_eq!(manifest_source(&observation), "observations");
        assert_eq!(manifest_source(&derived), "derived");
    }

    #[test]
    fn point_mode_rejects_multiple_realizations() {
        let draws: Vec<Vec<sim::quantity::QuantityResult>> = vec![vec![], vec![]];
        let err = render_quantities(&[], &draws, &[], Mode::Point, DesignCoords { scenario: None, sweep: &[] }, None, None, &test_cal()).unwrap_err();
        assert!(err.contains("exactly one realization"), "got: {err}");
    }

    #[test]
    fn banded_render_with_scenario_tags_header_rows_and_manifest() {
        // The `fit predict` overlay axis: a leading `scenario` column on the TSV
        // header + every row, and a `scenario` field on the manifest entry. `None`
        // (simulate) must omit both — guarded by the byte-identical tests above.
        use ir::observation::StratumKey;
        use ir::quantity::{Quantity, QuantityBody, QuantitySource, TemporalReduce, ValueReduce};
        use sim::quantity::{QuantityDrawValue, QuantityResult};

        let quantities = vec![Quantity {
            name: "peak".to_string(),
            stratum: Vec::<StratumKey>::new(),
            body: QuantityBody::Reduced {
                source: QuantitySource::State(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.0 })),
                reduce: Some(TemporalReduce::Value(ValueReduce::Max)),
            },
            dimension: None,
        }];
        // Two draws of a plain value scalar.
        let draws = vec![
            vec![QuantityResult::Scalar(QuantityDrawValue::Value(0.3))],
            vec![QuantityResult::Scalar(QuantityDrawValue::Value(0.5))],
        ];

        let (outs, manifest) =
            render_quantities(&quantities, &draws, &[], Mode::Banded,
                DesignCoords { scenario: Some("with_sia"), sweep: &[] }, None, None, &test_cal()).unwrap();
        let peak = &outs.iter().find(|(n, _)| n == "peak").unwrap().1;
        let lines: Vec<&str> = peak.trim_end().lines().collect();
        assert_eq!(
            lines[0],
            "scenario\tn_draws\tq05\tq25\tq50\tq75\tq95",
            "scenario leads the banded value-scalar header"
        );
        let row: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(row[0], "with_sia", "every row is tagged with the scenario");
        assert_eq!(row[1], "2", "n_draws after the scenario column");

        let mjson: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        let entry = mjson["quantities"].as_array().unwrap()[0].clone();
        assert_eq!(entry["scenario"], "with_sia", "manifest entry carries the scenario");
        // Calendar semantics travel at the manifest top level.
        assert_eq!(mjson["calendar"]["time_unit"], "days", "manifest carries the time_unit");
        assert_eq!(mjson["calendar"]["origin"], "2020-01-01", "manifest carries the origin");

        // Scenario + sweep: the sweep:<param> column follows the scenario column,
        // every row carries this cell's swept value, and the manifest entry gains
        // a `sweep` object.
        let sweep = [("k".to_string(), 8.0)];
        let (outs_sw, manifest_sw) = render_quantities(
            &quantities, &draws, &[], Mode::Banded,
            DesignCoords { scenario: Some("with_sia"), sweep: &sweep },
            None,
            None,
            &test_cal(),
        )
        .unwrap();
        let peak_sw = &outs_sw.iter().find(|(n, _)| n == "peak").unwrap().1;
        let lines_sw: Vec<&str> = peak_sw.trim_end().lines().collect();
        assert_eq!(
            lines_sw[0],
            "scenario\tsweep:k\tn_draws\tq05\tq25\tq50\tq75\tq95",
            "sweep:<param> column follows the scenario column"
        );
        let row_sw: Vec<&str> = lines_sw[1].split('\t').collect();
        assert_eq!(row_sw[0], "with_sia", "scenario cell");
        assert_eq!(row_sw[1], "8", "swept value cell");
        let mjson_sw: serde_json::Value = serde_json::from_str(&manifest_sw).unwrap();
        assert_eq!(
            mjson_sw["quantities"].as_array().unwrap()[0]["sweep"]["k"], 8.0,
            "manifest entry carries the sweep coordinate"
        );

        // None → no scenario column or field (simulate's byte-identical path).
        let (outs2, manifest2) =
            render_quantities(&quantities, &draws, &[], Mode::Banded, DesignCoords { scenario: None, sweep: &[] }, None, None, &test_cal()).unwrap();
        let peak2 = &outs2.iter().find(|(n, _)| n == "peak").unwrap().1;
        assert_eq!(
            peak2.lines().next().unwrap(),
            "n_draws\tq05\tq25\tq50\tq75\tq95",
            "scenario=None omits the column"
        );
        let mjson2: serde_json::Value = serde_json::from_str(&manifest2).unwrap();
        assert!(
            mjson2["quantities"].as_array().unwrap()[0].get("scenario").is_none(),
            "scenario=None omits the manifest field"
        );
    }

    // ── gh#794: per-row convergence on a reported estimand ──────────────────

    /// A quantity is the number that gets published, so its own R-hat is the
    /// one a report should rest on — and it can be far better determined than
    /// the parameters behind it (`f_cfr` at 1.31 while `p_fatal` and
    /// `h_care_surv` sit at 2.85 and 2.56).
    ///
    /// Two chains of eight draws that disagree about a series quantity: the
    /// `rhat` cell must flag it, and it must move from one time point to the
    /// next (a column that does not is a stamp, not a per-row statistic).
    #[test]
    fn banded_quantities_carry_the_row_s_own_rhat_and_ess() {
        use crate::fit::row_convergence::ChainOfDraw;
        use ir::observation::StratumKey;
        use ir::quantity::{Quantity, QuantityBody, QuantitySource};
        use sim::quantity::QuantityResult;

        let quantities = vec![Quantity {
            name: "prevalence".to_string(),
            stratum: Vec::<StratumKey>::new(),
            body: QuantityBody::Reduced {
                source: QuantitySource::State(ir::expr::Expr::Const(ir::expr::ConstExpr {
                    value: 0.0,
                })),
                reduce: None,
            },
            dimension: None,
        }];
        // t = 0: the chains agree (both near 0.10). t = 7: chain 0 sits near
        // 0.30 and chain 1 near 0.90 — a disagreement about the reported curve
        // that no parameter R-hat is being consulted for here.
        //
        // The within-chain jitter repeats across the two halves of the chain,
        // so split-R-hat sees no half-to-half difference within a chain: a
        // sequence that drifted (or narrowed) between the halves would inflate
        // every row, which is a property of the fixture and not of the chains.
        const JITTER: [f64; 8] = [-1.0, 1.0, -0.3, 0.3, -1.0, 1.0, -0.3, 0.3];
        let mut draws: Vec<Vec<QuantityResult>> = Vec::new();
        let mut ids: Vec<Option<usize>> = Vec::new();
        for chain in 0..2usize {
            for j in JITTER.iter() {
                let jitter = j * 0.002;
                let late = if chain == 0 { 0.30 } else { 0.90 };
                draws.push(vec![QuantityResult::Series(vec![0.10 + jitter, late + jitter])]);
                ids.push(Some(chain));
            }
        }
        let chains = ChainOfDraw(ids);
        let times = vec![0.0, 7.0];
        let (outs, _m) = render_quantities(
            &quantities,
            &draws,
            &times,
            Mode::Banded,
            DesignCoords { scenario: None, sweep: &[] },
            None,
            Some(&chains),
            &test_cal(),
        )
        .unwrap();

        let text = &outs.iter().find(|(n, _)| n == "prevalence").unwrap().1;
        let lines: Vec<&str> = text.trim_end().lines().collect();
        assert_eq!(
            lines[0], "time\tn_draws\trhat\tess\tq05\tq25\tq50\tq75\tq95",
            "rhat/ess follow n_draws"
        );
        let cell = |row: usize, col: usize| lines[row].split('\t').nth(col).unwrap();
        let rhat_at = |row: usize| cell(row, 2).parse::<f64>().unwrap();
        // Measured on this fixture: 0.8660 at t=0, 1.7419 at t=7. (A
        // rank-normalized R-hat can sit just below 1 on a finite sample where
        // the chains agree exactly — that is the estimator, not a bug.)
        assert!(
            rhat_at(1) < 1.05,
            "the chains agree at t=0, so its rhat is at or below 1: {}",
            cell(1, 2)
        );
        assert!(
            rhat_at(2) > 1.5,
            "the chains disagree threefold at t=7, so its rhat must flag it: {}",
            cell(2, 2)
        );
        assert_ne!(cell(1, 2), cell(2, 2), "a per-row statistic moves with the row");
        // ESS is reported alongside, and is a positive count.
        assert!(cell(1, 3).parse::<f64>().unwrap() > 0.0);
    }

    /// Without a chain partition — the `simulate --draws` path, which has no
    /// chains behind it — no columns are added at all, so that output is
    /// byte-identical.
    #[test]
    fn quantities_render_byte_identically_without_a_chain_partition() {
        use crate::fit::row_convergence::ChainOfDraw;
        use ir::observation::StratumKey;
        use ir::quantity::{Quantity, QuantityBody, QuantitySource};
        use sim::quantity::QuantityResult;

        let quantities = vec![Quantity {
            name: "prevalence".to_string(),
            stratum: Vec::<StratumKey>::new(),
            body: QuantityBody::Reduced {
                source: QuantitySource::State(ir::expr::Expr::Const(ir::expr::ConstExpr {
                    value: 0.0,
                })),
                reduce: None,
            },
            dimension: None,
        }];
        let draws: Vec<Vec<QuantityResult>> = (0..8)
            .map(|i| vec![QuantityResult::Series(vec![0.1 + i as f64 * 0.01])])
            .collect();
        let times = vec![0.0];
        let render = |chains: Option<&ChainOfDraw>| {
            render_quantities(
                &quantities,
                &draws,
                &times,
                Mode::Banded,
                DesignCoords { scenario: None, sweep: &[] },
                None,
                chains,
                &test_cal(),
            )
            .unwrap()
            .0
            .into_iter()
            .find(|(n, _)| n == "prevalence")
            .unwrap()
            .1
        };
        let without = render(None);
        assert!(!without.lines().next().unwrap().contains("rhat"), "{without}");

        let ids = ChainOfDraw((0..8).map(|i| Some(i / 4)).collect());
        let with = render(Some(&ids));
        assert!(with.lines().next().unwrap().contains("\trhat\tess\t"), "{with}");
        // Strip the two added cells and the two files agree.
        let stripped: String = with
            .lines()
            .map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                let mut keep: Vec<&str> = c[..2].to_vec();
                keep.extend_from_slice(&c[4..]);
                format!("{}\n", keep.join("\t"))
            })
            .collect();
        assert_eq!(stripped, without, "the flag adds two cells and changes nothing else");
    }

    /// A censored draw drops out of the band, and its chain id must drop with
    /// it. Filtering one side only pairs a value with another draw's chain and
    /// still produces a plausible number.
    #[test]
    fn a_censored_scalar_draw_drops_its_chain_id_too() {
        use crate::fit::row_convergence::ChainOfDraw;
        use sim::quantity::{QuantityDrawValue, QuantityResult};

        let vals: Vec<QuantityDrawValue> = vec![
            QuantityDrawValue::Value(1.0),
            QuantityDrawValue::Censored,
            QuantityDrawValue::Value(2.0),
            QuantityDrawValue::Value(3.0),
        ];
        let chains = ChainOfDraw(vec![Some(0), Some(0), Some(1), Some(1)]);
        let (finite, finite_chains) = finite_with_chains(&vals, Some(&chains));
        assert_eq!(finite, vec![1.0, 2.0, 3.0]);
        assert_eq!(
            finite_chains.unwrap().0,
            vec![Some(0), Some(1), Some(1)],
            "the censored draw's chain id is dropped with its value, not left \
             behind to shift every later pairing"
        );
        // And the untyped `None` case carries no partition at all.
        let (_, none) = finite_with_chains(&vals, None);
        assert!(none.is_none());
        let _ = QuantityResult::Scalar(QuantityDrawValue::Censored);
    }
}
