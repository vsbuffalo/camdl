//! The shared posterior-trajectory writer + manifest.
//!
//! Implements §4b of
//! `docs/dev/proposals/2026-06-09-latent-trajectory-output-consolidation.md`:
//! the tidy/long `trajectories.tsv` (all chains × all draws stacked, leading
//! `chain  draw  time [date]` id columns) plus a small `trajectories.json`
//! manifest so tooling can interpret a run without scraping the header.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use sim::{Flows, IntState, RealState, Snapshot, Trajectory};

use crate::calendar::CalendarMeta;

/// Tolerance for matching a requested fork time T* to a saved snapshot time.
/// A trajectory is recorded at discrete snapshot times; the conditioned
/// counterfactual fork requires T* to coincide with one of them — the "saved
/// cadence contains T*" contract. The match is therefore exact-within-float-
/// noise; a T* strictly between snapshots is an error, never silently snapped to
/// a neighbour (which would seed the fork from the wrong latent state).
pub const SNAPSHOT_TIME_TOL: f64 = 1e-9;

/// Time resolution of a posterior path. PGAS paths are substep resolution;
/// PF/PMMH paths (a later consolidation step) are observation-step resolution.
/// Carried into the file header + manifest so a downstream union of mixed
/// outputs cannot silently blend substep PGAS paths with obs-resolution
/// PF/PMMH paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    Substep,
    Observation,
}

impl Granularity {
    pub fn as_str(self) -> &'static str {
        match self {
            Granularity::Substep => "substep",
            Granularity::Observation => "observation",
        }
    }
}

/// One posterior draw: the latent path it implies, identified by `(chain,
/// draw)`. The path *is* a [`Trajectory`] (the `simulate` output type); the
/// model-predicted per-stream incidence rides as a sidecar
/// (`incidence[snapshot][stream]`, parallel to `path.snapshots`) so the writer
/// can emit `inc_<stream>` columns without finite-differencing counts.
pub struct PosteriorDraw {
    pub chain: usize,
    pub draw: usize,
    pub path: Trajectory,
    /// `incidence[s][k]` — the k-th incidence stream's model-predicted value at
    /// snapshot `s`. Either empty (no incidence streams) or
    /// `incidence.len() == path.snapshots.len()`, each inner row of length
    /// `stream_names.len()` (see [`write_trajectories_tsv`]). The producer (the
    /// PGAS adapter) computes this from the observation model's `FlowSum`
    /// projection — never from `−ΔS` / `diff(flow)`.
    pub incidence: Vec<Vec<f64>>,
}

/// Column-layout description shared by the writer and the manifest. Built once
/// from the model so the header and the `trajectories.json` `columns` list
/// cannot drift.
pub struct TrajColumnSpec {
    /// Integer compartment names, in model order (index into
    /// `Snapshot::int_state.counts`).
    pub int_comps: Vec<String>,
    /// Real compartment names, in model order (index into
    /// `Snapshot::real_state.values`).
    pub real_comps: Vec<String>,
    /// `flow_<transition>` names, in model order (index into the snapshot flow
    /// vector).
    pub flows: Vec<String>,
    /// `inc_<stream>` names, in incidence-stream order (index into a
    /// [`PosteriorDraw::incidence`] row).
    pub incidence: Vec<String>,
}

impl TrajColumnSpec {
    /// Build from a model + the incidence-stream names. Integer/real
    /// compartments split by `CompartmentKind`, matching the `simulate`
    /// trajectory writer's column order.
    pub fn from_model(model: &ir::Model, incidence_stream_names: &[String]) -> Self {
        let mut int_comps = Vec::new();
        let mut real_comps = Vec::new();
        for c in &model.compartments {
            match c.kind {
                ir::model::CompartmentKind::Integer => int_comps.push(c.name.clone()),
                ir::model::CompartmentKind::Real => real_comps.push(c.name.clone()),
            }
        }
        let flows = model
            .transitions
            .iter()
            .map(|t| format!("flow_{}", t.name))
            .collect();
        let incidence = incidence_stream_names
            .iter()
            .map(|n| format!("inc_{}", n))
            .collect();
        TrajColumnSpec {
            int_comps,
            real_comps,
            flows,
            incidence,
        }
    }

    /// Every data-column name (excluding the leading `chain/draw/time[/date]`
    /// id columns), in emit order. Used for the manifest `columns` list.
    pub fn data_column_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(self.int_comps.iter().cloned());
        out.extend(self.real_comps.iter().cloned());
        out.extend(self.flows.iter().cloned());
        out.extend(self.incidence.iter().cloned());
        out
    }
}

/// The `trajectories.json` manifest. Records how to interpret the sibling
/// `trajectories.tsv` without scraping its header, and surfaces the
/// conditioned-vs-forward distinction (this file's `inc_<stream>` is the
/// *conditioned* smoother `p(x|y)`; a `simulate --obs` file is the *forward*
/// posterior-predictive `p(y|θ)`).
pub struct TrajManifest {
    pub method: String,
    pub granularity: Granularity,
    pub n_chains: usize,
    pub n_draws: usize,
    /// Every TSV column name in emit order: the id columns
    /// (`chain`, `draw`, `time`, optional `date`) then the data columns.
    pub columns: Vec<String>,
    pub model_hash: String,
    /// `true` for a smoother path conditioned on the data (PGAS `X|θ,y`); the
    /// `inc_<stream>` columns are conditioned incidence, NOT the free-forward
    /// posterior-predictive a `simulate --obs` run produces.
    pub conditioned: bool,
    /// `true` if the paths come from an ancestral filter-smoother prone to
    /// early-time degeneracy (PF / PMMH). PGAS ancestor sampling mitigates this,
    /// so PGAS paths set `false`.
    pub degeneracy_caveat: bool,
    /// The requested number of saved trajectories (`n_trajectories` /
    /// `--save-paths N`) — the source count this file was produced from.
    pub n_trajectories: usize,
    /// Best-effort pointer to the run's free-forward posterior-predictive
    /// observation file (from `simulate --obs`), so a researcher can compare
    /// the conditioned smoother incidence here against the forward predictive.
    /// `None` when no such file is discoverable for this run.
    pub predictive_obs_file: Option<String>,
    /// Calendar semantics for the `time` axis — so a consumer maps `time →
    /// Date` without re-deriving `origin`/`time_unit` from the header.
    pub calendar: CalendarMeta,
}

impl TrajManifest {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "format": "camdl-trajectories",
            "version": 1,
            "method": self.method,
            "granularity": self.granularity.as_str(),
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "columns": self.columns,
            "model_hash": self.model_hash,
            // Conditioned (smoother p(x|y)) vs forward predictive (p(y|θ)). The
            // inc_<stream> columns here are conditioned; a simulate --obs file is
            // forward. Surfaced so a smoother is never mistaken for the
            // predictive.
            "conditioned": self.conditioned,
            "degeneracy_caveat": self.degeneracy_caveat,
            "n_trajectories": self.n_trajectories,
            "predictive_obs_file": self.predictive_obs_file,
            "calendar": self.calendar.to_json(),
        })
    }

    /// Write the manifest as pretty JSON to `path`.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.to_json())
            .map_err(|e| format!("trajectories manifest: json error: {e}"))?;
        std::fs::write(path, json)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

/// The `# camdl-trajectories v1 ...` header line for a `trajectories.tsv`.
fn header_comment(model_hash: &str, method: &str, granularity: Granularity) -> String {
    format!(
        "# camdl-trajectories v1\tmodel={}\tmethod={}\tgranularity={}",
        model_hash,
        method,
        granularity.as_str()
    )
}

/// Optional calendar-date origin for a `date` column.
///
/// `(origin, time_unit)` — the model's declared `origin =
/// date("...")` and `time_unit`. When `Some`, the writer emits a `date` column
/// rendered via [`ir::caltime::internal_to_date_hires`] (the same path
/// `simulate --dates` uses); when `None`, no date column.
pub type DateOrigin<'a> = Option<(&'a str, &'a str)>;

/// Write the tidy/long `trajectories.tsv`: a `# camdl-trajectories v1` header,
/// the column header (`chain  draw  time [date]  <int> <real> flow_* inc_*`),
/// then one row per snapshot per draw — all chains × all draws stacked into one
/// file, disambiguated by the leading id columns.
///
/// `columns` describes the data-column layout (built once from the model);
/// every draw's `path` must agree with it (same int/real/flow vector lengths)
/// and each draw's `incidence` (when non-empty) must be parallel to its
/// `path.snapshots` with rows of length `columns.incidence.len()`. A mismatch
/// is a hard error — the writer never silently truncates or zero-fills.
#[allow(clippy::too_many_arguments)]
pub fn write_trajectories_tsv(
    path: &Path,
    draws: &[PosteriorDraw],
    columns: &TrajColumnSpec,
    date_origin: DateOrigin,
    model_hash: &str,
    method: &str,
    granularity: Granularity,
) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    // BufWriter: a substep-resolution national-scale path is thousands of rows ×
    // hundreds of fields; unbuffered, each `write!` is a syscall. The simulate
    // and (old) PGAS writers both buffer for exactly this reason.
    let mut w = std::io::BufWriter::new(f);

    writeln!(w, "{}", header_comment(model_hash, method, granularity))
        .map_err(|e| e.to_string())?;

    // Column header.
    write!(w, "chain\tdraw\ttime").map_err(|e| e.to_string())?;
    if date_origin.is_some() {
        write!(w, "\tdate").map_err(|e| e.to_string())?;
    }
    for n in &columns.int_comps {
        write!(w, "\t{}", n).map_err(|e| e.to_string())?;
    }
    for n in &columns.real_comps {
        write!(w, "\t{}", n).map_err(|e| e.to_string())?;
    }
    for n in &columns.flows {
        write!(w, "\t{}", n).map_err(|e| e.to_string())?;
    }
    for n in &columns.incidence {
        write!(w, "\t{}", n).map_err(|e| e.to_string())?;
    }
    writeln!(w).map_err(|e| e.to_string())?;

    let n_int = columns.int_comps.len();
    let n_real = columns.real_comps.len();
    let n_flow = columns.flows.len();
    let n_inc = columns.incidence.len();

    for d in draws {
        // Validate the incidence sidecar shape once per draw. When the header
        // declares inc_<stream> columns (n_inc > 0) every snapshot needs a
        // matching incidence row — otherwise the per-row index below panics on
        // an empty sidecar; when no inc columns are declared, a provided sidecar
        // must still match the snapshot count. (n_inc == 0 + empty is fine.)
        if (n_inc > 0 || !d.incidence.is_empty()) && d.incidence.len() != d.path.snapshots.len() {
            return Err(format!(
                "trajectories: chain {} draw {}: incidence has {} rows but path \
                 has {} snapshots ({} inc columns declared)",
                d.chain, d.draw, d.incidence.len(), d.path.snapshots.len(), n_inc
            ));
        }
        for (s, snap) in d.path.snapshots.iter().enumerate() {
            if snap.int_state.counts.len() != n_int {
                return Err(format!(
                    "trajectories: chain {} draw {}: snapshot has {} integer \
                     compartments, header declares {}",
                    d.chain, d.draw, snap.int_state.counts.len(), n_int
                ));
            }
            if snap.real_state.values.len() != n_real {
                return Err(format!(
                    "trajectories: chain {} draw {}: snapshot has {} real \
                     compartments, header declares {}",
                    d.chain, d.draw, snap.real_state.values.len(), n_real
                ));
            }
            if snap.flows.len() != n_flow {
                return Err(format!(
                    "trajectories: chain {} draw {}: snapshot has {} flows, \
                     header declares {}",
                    d.chain, d.draw, snap.flows.len(), n_flow
                ));
            }

            write!(w, "{}\t{}\t{}", d.chain, d.draw, snap.t).map_err(|e| e.to_string())?;
            if let Some((origin, time_unit)) = date_origin {
                let date = ir::caltime::internal_to_date_hires(origin, snap.t, time_unit)
                    .map_err(|e| format!("trajectories: error rendering date: {e}"))?;
                write!(w, "\t{}", date).map_err(|e| e.to_string())?;
            }
            for &c in &snap.int_state.counts {
                write!(w, "\t{}", c).map_err(|e| e.to_string())?;
            }
            for &v in &snap.real_state.values {
                write!(w, "\t{:.6}", v).map_err(|e| e.to_string())?;
            }
            match &snap.flows {
                Flows::Int(fs) => {
                    for &fl in fs {
                        write!(w, "\t{}", fl).map_err(|e| e.to_string())?;
                    }
                }
                Flows::Real(fs) => {
                    for &fl in fs {
                        write!(w, "\t{:.6}", fl).map_err(|e| e.to_string())?;
                    }
                }
            }
            if n_inc > 0 {
                let row = &d.incidence[s];
                if row.len() != n_inc {
                    return Err(format!(
                        "trajectories: chain {} draw {}: incidence row {} has {} \
                         entries, header declares {}",
                        d.chain, d.draw, s, row.len(), n_inc
                    ));
                }
                for &inc in row {
                    write!(w, "\t{}", inc).map_err(|e| e.to_string())?;
                }
            }
            writeln!(w).map_err(|e| e.to_string())?;
        }
    }

    // Explicit flush: BufWriter swallows write errors on drop, which would
    // silently truncate the file if the disk filled during the final drain.
    w.flush()
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(())
}

/// Read the latent state `(IntState, RealState)` at fork time `t_star` for one
/// `(chain, draw)` from a `trajectories.tsv`. The inverse of
/// [`write_trajectories_tsv`] for a single snapshot — the step that hands a
/// saved smoothed path X(T*) to the conditioned counterfactual fork (the
/// engine seam; gh#322).
///
/// Compartment columns are resolved BY NAME from `columns`, in model order, so
/// the (integer, real) split cannot drift from the writer's layout and the
/// optional `date` column plus the `flow_*` / `inc_*` columns are skipped
/// without positional assumptions.
///
/// Errors if the file has no row for `(chain, draw)` at a snapshot time within
/// [`SNAPSHOT_TIME_TOL`] of `t_star` — i.e. the saved cadence does not contain
/// T*. (`Sampled` paths only; an ODE fit recomputes X(T*) from θ instead.)
pub fn read_state_at(
    path: &Path,
    columns: &TrajColumnSpec,
    chain: usize,
    draw: usize,
    t_star: f64,
) -> Result<(IntState, RealState), String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("empty trajectories file: {}", path.display()))?
        .split('\t')
        .collect();
    let col = |name: &str| -> Result<usize, String> {
        header
            .iter()
            .position(|c| *c == name)
            .ok_or_else(|| format!("trajectories file {} has no `{name}` column", path.display()))
    };
    let (ci, di, ti) = (col("chain")?, col("draw")?, col("time")?);
    let int_idx: Vec<usize> =
        columns.int_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?;
    let real_idx: Vec<usize> =
        columns.real_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?;

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let bad = |what: &str| format!("trajectories file {}: bad {what} field", path.display());
        let row_chain: usize =
            f.get(ci).and_then(|s| s.parse().ok()).ok_or_else(|| bad("chain"))?;
        let row_draw: usize =
            f.get(di).and_then(|s| s.parse().ok()).ok_or_else(|| bad("draw"))?;
        if row_chain != chain || row_draw != draw {
            continue;
        }
        let t: f64 = f.get(ti).and_then(|s| s.parse().ok()).ok_or_else(|| bad("time"))?;
        if (t - t_star).abs() <= SNAPSHOT_TIME_TOL {
            let counts = int_idx
                .iter()
                .map(|&i| {
                    f.get(i)
                        .and_then(|s| s.parse::<i64>().ok())
                        .ok_or_else(|| format!("trajectories file {}: bad integer compartment at column {i}", path.display()))
                })
                .collect::<Result<Vec<i64>, _>>()?;
            let values = real_idx
                .iter()
                .map(|&i| {
                    f.get(i)
                        .and_then(|s| s.parse::<f64>().ok())
                        .ok_or_else(|| format!("trajectories file {}: bad real compartment at column {i}", path.display()))
                })
                .collect::<Result<Vec<f64>, _>>()?;
            return Ok((IntState::from_vec(counts), RealState::from_vec(values)));
        }
    }
    Err(format!(
        "no saved snapshot at t={t_star} for (chain={chain}, draw={draw}) in {} — the fork \
         time T* must coincide with a saved snapshot time (the saved cadence must contain T*)",
        path.display()
    ))
}

/// The sorted, distinct snapshot times saved for `(chain, draw)` in a
/// `trajectories.tsv`. The counterfactual fork (gh#322) reads this to pick the
/// latest saved snapshot strictly *before* a toggled intervention's fire time:
/// the derived fork must coincide with a saved snapshot, since [`read_state_at`]
/// reads the forked state from exactly that snapshot.
pub fn snapshot_times(path: &Path, chain: usize, draw: usize) -> Result<Vec<f64>, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("empty trajectories file: {}", path.display()))?
        .split('\t')
        .collect();
    let col = |name: &str| -> Result<usize, String> {
        header
            .iter()
            .position(|c| *c == name)
            .ok_or_else(|| format!("trajectories file {} has no `{name}` column", path.display()))
    };
    let (ci, di, ti) = (col("chain")?, col("draw")?, col("time")?);

    let mut times: Vec<f64> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let bad = |what: &str| format!("trajectories file {}: bad {what} field", path.display());
        let row_chain: usize =
            f.get(ci).and_then(|s| s.parse().ok()).ok_or_else(|| bad("chain"))?;
        let row_draw: usize =
            f.get(di).and_then(|s| s.parse().ok()).ok_or_else(|| bad("draw"))?;
        if row_chain != chain || row_draw != draw {
            continue;
        }
        let t: f64 = f.get(ti).and_then(|s| s.parse().ok()).ok_or_else(|| bad("time"))?;
        times.push(t);
    }
    if times.is_empty() {
        return Err(format!(
            "no saved snapshots for (chain={chain}, draw={draw}) in {}",
            path.display()
        ));
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("snapshot times are finite"));
    times.dedup();
    Ok(times)
}

/// The last saved snapshot of one posterior draw's latent path: where the path
/// ends, and the state it ends at.
///
/// At the terminal observation time the smoothing distribution equals the
/// filtering distribution — no future data remains to condition on — so this
/// state is a draw from `p(x_T | y_{1:T})` paired with its own θ. That is the
/// forecast origin `simulate --init-state fit` runs forward from (gh#697).
#[derive(Debug, Clone, PartialEq)]
pub struct TerminalState {
    pub chain: usize,
    pub draw: usize,
    /// The model time of the path's last saved snapshot.
    pub t: f64,
    pub int_state: IntState,
    pub real_state: RealState,
}

/// Every saved path's TERMINAL snapshot, in one pass over the file.
///
/// [`read_state_at`] answers "the state of one draw at time T"; this answers
/// "the terminal state of every draw", which a forecast needs for the whole
/// ensemble at once. It is a separate function rather than a loop over
/// `read_state_at` because that loop re-reads and re-scans the whole file per
/// draw — quadratic on a national-scale run, where the file is hundreds of
/// megabytes and the forkable subset is in the hundreds.
///
/// The per-draw terminal time is returned rather than assumed equal across
/// draws: whether one saved cadence covers every draw is the caller's check to
/// make explicitly, not this reader's to paper over.
///
/// Returns one entry per `(chain, draw)` present, sorted by that key. An empty
/// file (header only) yields an empty vector — "no saved paths" is a state the
/// caller reports, not an error here.
pub fn read_terminal_states(
    path: &Path,
    columns: &TrajColumnSpec,
) -> Result<Vec<TerminalState>, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("empty trajectories file: {}", path.display()))?
        .split('\t')
        .collect();
    let col = |name: &str| -> Result<usize, String> {
        header
            .iter()
            .position(|c| *c == name)
            .ok_or_else(|| format!("trajectories file {} has no `{name}` column", path.display()))
    };
    let (ci, di, ti) = (col("chain")?, col("draw")?, col("time")?);
    // Compartments resolved BY NAME in model order — the same discipline
    // `read_state_at` uses, so the (integer, real) split cannot drift from the
    // writer's layout and the `flow_*` / `inc_*` / `date` columns are skipped
    // without positional assumptions.
    let int_idx: Vec<usize> = columns.int_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?;
    let real_idx: Vec<usize> =
        columns.real_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?;

    let mut latest: std::collections::BTreeMap<(usize, usize), TerminalState> =
        std::collections::BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let bad = |what: &str| format!("trajectories file {}: bad {what} field", path.display());
        let chain: usize = f.get(ci).and_then(|s| s.parse().ok()).ok_or_else(|| bad("chain"))?;
        let draw: usize = f.get(di).and_then(|s| s.parse().ok()).ok_or_else(|| bad("draw"))?;
        let t: f64 = f.get(ti).and_then(|s| s.parse().ok()).ok_or_else(|| bad("time"))?;
        if latest.get(&(chain, draw)).is_some_and(|prev| prev.t >= t) {
            continue;
        }
        let counts = int_idx
            .iter()
            .map(|&i| {
                f.get(i).and_then(|s| s.parse::<i64>().ok()).ok_or_else(|| {
                    format!(
                        "trajectories file {}: bad integer compartment at column {i}",
                        path.display()
                    )
                })
            })
            .collect::<Result<Vec<i64>, _>>()?;
        let values = real_idx
            .iter()
            .map(|&i| {
                f.get(i).and_then(|s| s.parse::<f64>().ok()).ok_or_else(|| {
                    format!(
                        "trajectories file {}: bad real compartment at column {i}",
                        path.display()
                    )
                })
            })
            .collect::<Result<Vec<f64>, _>>()?;
        latest.insert(
            (chain, draw),
            TerminalState {
                chain,
                draw,
                t,
                int_state: IntState::from_vec(counts),
                real_state: RealState::from_vec(values),
            },
        );
    }
    Ok(latest.into_values().collect())
}

/// One posterior draw's saved path as read back off disk: the latent
/// [`Trajectory`] and, parallel to its snapshots, the `inc_<stream>` row per
/// snapshot in `TrajColumnSpec::incidence` order. The read-side counterpart of
/// [`PosteriorDraw`]'s `(path, incidence)` pair.
pub type SavedPath = (Trajectory, Vec<Vec<f64>>);

/// Reconstruct the whole saved path for `(chain, draw)` as a [`Trajectory`].
///
/// [`read_state_at`] answers "the state at one instant", which is what a
/// counterfactual fork needs. A quantity anchored inside the data window needs
/// the *series*: `value_at(N0 - S, last_obs)` has to be evaluated against the
/// conditioned smoothing path, not against a fresh unconditioned replay that
/// ignores every observation it is anchored inside (gh#722).
///
/// The `inc_<stream>` columns are returned alongside, one row per snapshot, in
/// `columns.incidence` order. Per this file's own manifest they are the
/// **conditioned** smoother `p(x | y)` — which is precisely why they are the
/// right operand for an in-window quantity and the wrong one for a projection.
///
/// One key costs one whole scan of the file. A caller that wants many keys out
/// of one file must use [`read_trajectories`], which is also the reason this
/// one is kept: it is the deliberately simple reference that
/// `read_trajectories_matches_the_single_key_reader_on_every_key` differences
/// the one-pass reader against, so a bucketing bug in the fast path shows up as
/// a red test rather than as a plausible band.
///
/// # Chain-binomial only, loudly
///
/// [`Flows`] keeps integer and real flows as distinct variants on purpose: a
/// real (ODE) flow quantized through `u64` silently zeroes sub-unit flows. This
/// reader therefore refuses a real-flow file rather than rounding it. PGAS and
/// PMMH write integer flows, which is the whole of the smoothing-path use case
/// today; an ODE smoother would need this to grow a variant, not a cast.
pub fn read_trajectory(
    path: &Path,
    columns: &TrajColumnSpec,
    chain: usize,
    draw: usize,
) -> Result<SavedPath, String> {
    // Delegates rather than carrying a second row-selection loop. A duplicate
    // implementation kept alive only to be a test's reference oracle is worse
    // than no oracle: the two drift, and a differential test between them
    // agrees happily on a shared wrong answer. The tests that carry weight
    // here compare against what was WRITTEN.
    //
    // `read_trajectories` is total over its `keys` — an absent one is already
    // the `no_saved_path` refusal — so the `ok_or_else` below is unreachable
    // by that contract. It stays because a library returns an error rather
    // than panicking if the contract is ever weakened.
    read_trajectories(path, columns, &[(chain, draw)])?
        .remove(&(chain, draw))
        .ok_or_else(|| no_saved_path(path, chain, draw))
}

/// Reconstruct the saved paths for SEVERAL `(chain, draw)` keys in ONE pass
/// over the file.
///
/// [`read_trajectory`] reads one key and scans the whole file to do it, so a
/// caller that wants many keys out of one file pays `n_keys × file_size`. That
/// is the reporting configuration for `fit predict`: a `value_at(…, last_obs)`
/// quantity is read on every posterior draw's smoothing path (gh#722), and at
/// national scale that is hundreds of draws against a substep-resolution
/// `trajectories.tsv`. One pass instead of hundreds is the whole of the
/// difference — the decode of a kept row is the shared [`RowLayout`] both
/// readers use, so the two cannot disagree on what a row means.
///
/// The returned map is TOTAL over `keys`: a requested key with no rows in the
/// file is an error naming that key (the same refusal [`read_trajectory`]
/// gives), never a silently-absent entry or an empty path. A caller can
/// therefore look a requested key up and treat a miss as an internal
/// inconsistency rather than as data. Duplicate keys collapse; `keys` empty
/// reads nothing and yields an empty map, matching zero calls of the
/// single-key reader.
///
/// Peak memory is every requested path resident at once, which is what a
/// caller holding the whole posterior ensemble wants; a caller that needs only
/// one path at a time should ask for one key at a time.
///
/// Real flows are refused exactly as [`read_trajectory`] refuses them — see
/// its "Chain-binomial only, loudly" note.
pub fn read_trajectories(
    path: &Path,
    columns: &TrajColumnSpec,
    keys: &[(usize, usize)],
) -> Result<HashMap<(usize, usize), SavedPath>, String> {
    let mut out: HashMap<(usize, usize), SavedPath> = HashMap::new();
    if keys.is_empty() {
        return Ok(out);
    }
    let wanted: HashSet<(usize, usize)> = keys.iter().copied().collect();

    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let layout = RowLayout::resolve(path, header_of(path, lines.next())?, columns)?;

    let mut fields: Vec<&str> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let key = layout.ids(line)?;
        if !wanted.contains(&key) {
            continue;
        }
        fields.clear();
        fields.extend(line.split('\t'));
        let (snap, inc_row) = layout.decode(&fields)?;
        let entry = out.entry(key).or_insert_with(|| (Trajectory::new(), Vec::new()));
        entry.0.push(snap);
        entry.1.push(inc_row);
    }

    // Total over `keys`, in `keys` order so the refusal is deterministic.
    for &(chain, draw) in keys {
        if !out.contains_key(&(chain, draw)) {
            return Err(no_saved_path(path, chain, draw));
        }
    }
    Ok(out)
}

/// The refusal both path readers give for a `(chain, draw)` the file holds no
/// rows for. One function so the two cannot word it differently — the caller
/// surfaces this text to the researcher.
fn no_saved_path(path: &Path, chain: usize, draw: usize) -> String {
    format!(
        "no saved path for (chain={chain}, draw={draw}) in {} — that draw is not in the \
         forkable subset (the trajectory save stride and `thin` need not agree; gh#727)",
        path.display()
    )
}

/// The column-header line of a `trajectories.tsv`, split on tabs. `None` (no
/// non-comment line at all) is the empty-file refusal.
fn header_of<'a>(path: &Path, line: Option<&'a str>) -> Result<Vec<&'a str>, String> {
    Ok(line
        .ok_or_else(|| format!("empty trajectories file: {}", path.display()))?
        .split('\t')
        .collect())
}

/// The column indices one `trajectories.tsv` header implies for a given
/// [`TrajColumnSpec`], resolved once per file.
///
/// Both path readers decode a kept row through this one layout, so the
/// (integer, real, flow, incidence) split — and the real-flow refusal that
/// rides with it — cannot drift between the single-key and the multi-key scan.
/// What differs between them is only which rows they keep, which is the point:
/// the shared substrate is the bug-prone part, the row-selection loop is the
/// part that should stay distinct.
struct RowLayout<'a> {
    path: &'a Path,
    chain: usize,
    draw: usize,
    time: usize,
    /// One past the last id column. A scan splits only this prefix to read a
    /// row's `(chain, draw)` key, and splits the whole row only for the rows it
    /// keeps — on a national-scale file nearly every row belongs to some other
    /// draw, and splitting its few hundred fields to throw them away is the
    /// cost that made the per-draw read quadratic.
    id_prefix: usize,
    int_idx: Vec<usize>,
    real_idx: Vec<usize>,
    flow_idx: Vec<usize>,
    inc_idx: Vec<usize>,
}

impl<'a> RowLayout<'a> {
    /// Resolve every column BY NAME, in model order — the same discipline
    /// [`read_state_at`] uses, so the optional `date` column and any column the
    /// spec does not name are skipped without positional assumptions.
    fn resolve(
        path: &'a Path,
        header: Vec<&str>,
        columns: &TrajColumnSpec,
    ) -> Result<Self, String> {
        let col = |name: &str| -> Result<usize, String> {
            header
                .iter()
                .position(|c| *c == name)
                .ok_or_else(|| format!("trajectories file {} has no `{name}` column", path.display()))
        };
        let (chain, draw, time) = (col("chain")?, col("draw")?, col("time")?);
        Ok(RowLayout {
            path,
            chain,
            draw,
            time,
            id_prefix: chain.max(draw) + 1,
            int_idx: columns.int_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?,
            real_idx: columns.real_comps.iter().map(|n| col(n)).collect::<Result<_, _>>()?,
            flow_idx: columns.flows.iter().map(|n| col(n)).collect::<Result<_, _>>()?,
            inc_idx: columns.incidence.iter().map(|n| col(n)).collect::<Result<_, _>>()?,
        })
    }

    fn bad(&self, what: &str) -> String {
        format!("trajectories file {}: bad {what} field", self.path.display())
    }

    /// The `(chain, draw)` key of one data row, reading only the id prefix.
    /// A malformed key is an error on EVERY row, including rows the caller is
    /// about to skip — a file whose id columns do not parse is not a file we
    /// read a posterior off.
    fn ids(&self, line: &str) -> Result<(usize, usize), String> {
        let (mut chain, mut draw) = (None, None);
        for (i, fld) in line.split('\t').enumerate().take(self.id_prefix) {
            if i == self.chain {
                chain = Some(fld);
            }
            if i == self.draw {
                draw = Some(fld);
            }
        }
        Ok((
            chain.and_then(|s| s.parse().ok()).ok_or_else(|| self.bad("chain"))?,
            draw.and_then(|s| s.parse().ok()).ok_or_else(|| self.bad("draw"))?,
        ))
    }

    /// Decode one kept row into its snapshot and its `inc_<stream>` row.
    fn decode(&self, f: &[&str]) -> Result<(Snapshot, Vec<f64>), String> {
        let t: f64 = f.get(self.time).and_then(|s| s.parse().ok()).ok_or_else(|| self.bad("time"))?;
        let counts = self
            .int_idx
            .iter()
            .map(|&i| {
                f.get(i).and_then(|s| s.parse::<i64>().ok()).ok_or_else(|| {
                    format!("trajectories file {}: bad integer compartment at column {i}",
                        self.path.display())
                })
            })
            .collect::<Result<Vec<i64>, _>>()?;
        let values = self
            .real_idx
            .iter()
            .map(|&i| {
                f.get(i).and_then(|s| s.parse::<f64>().ok()).ok_or_else(|| {
                    format!("trajectories file {}: bad real compartment at column {i}",
                        self.path.display())
                })
            })
            .collect::<Result<Vec<f64>, _>>()?;
        let flows = self
            .flow_idx
            .iter()
            .map(|&i| {
                let raw = f.get(i).ok_or_else(|| self.bad("flow"))?;
                if raw.contains('.') {
                    return Err(format!(
                        "trajectories file {}: flow column {i} holds `{raw}`, a real-valued \
                         flow. This reader is chain-binomial only — rounding a real flow to \
                         an integer silently zeroes sub-unit flows, so it refuses rather \
                         than quantizing.",
                        self.path.display()
                    ));
                }
                raw.parse::<u64>().map_err(|_| {
                    format!("trajectories file {}: bad integer flow at column {i}",
                        self.path.display())
                })
            })
            .collect::<Result<Vec<u64>, String>>()?;
        let inc_row = self
            .inc_idx
            .iter()
            .map(|&i| {
                f.get(i).and_then(|s| s.parse::<f64>().ok()).ok_or_else(|| {
                    format!("trajectories file {}: bad incidence at column {i}",
                        self.path.display())
                })
            })
            .collect::<Result<Vec<f64>, _>>()?;

        Ok((
            Snapshot {
                t,
                int_state: IntState::from_vec(counts),
                real_state: RealState::from_vec(values),
                flows: Flows::Int(flows),
            },
            inc_row,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::{IntState, RealState, Snapshot};

    fn snap(t: f64, ints: Vec<i64>, flows: Vec<u64>) -> Snapshot {
        Snapshot {
            t,
            int_state: IntState::from_vec(ints),
            real_state: RealState::from_vec(Vec::new()),
            flows: Flows::Int(flows),
        }
    }

    fn cols() -> TrajColumnSpec {
        TrajColumnSpec {
            int_comps: vec!["S".into(), "I".into(), "R".into()],
            real_comps: vec![],
            flows: vec!["flow_infection".into(), "flow_recovery".into()],
            incidence: vec!["inc_cases".into()],
        }
    }

    /// gh#722: the whole saved path must come back exactly as written, because
    /// a quantity anchored inside the data window is evaluated against it.
    /// A per-instant reader (`read_state_at`) cannot answer that question.
    #[test]
    fn read_trajectory_round_trips_the_written_path() {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_io_traj_rt_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("trajectories.tsv");

        let mut t0 = Trajectory::new();
        t0.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t0.push(snap(1.0, vec![97, 2, 1], vec![2, 1]));
        t0.push(snap(2.0, vec![94, 4, 2], vec![3, 1]));
        // A second (chain, draw) that must NOT bleed into the first's path.
        let mut t1 = Trajectory::new();
        t1.push(snap(0.0, vec![50, 50, 0], vec![0, 0]));
        t1.push(snap(1.0, vec![40, 55, 5], vec![10, 5]));

        let draws = vec![
            PosteriorDraw {
                chain: 0, draw: 5, path: t0.clone(),
                incidence: vec![vec![0.0], vec![2.0], vec![3.0]],
            },
            PosteriorDraw {
                chain: 1, draw: 5, path: t1,
                incidence: vec![vec![0.0], vec![10.0]],
            },
        ];
        write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap();

        let (got, inc) = read_trajectory(&path, &cols(), 0, 5).expect("round trip");

        assert_eq!(got.snapshots.len(), 3, "must read only (chain 0, draw 5)'s rows");
        assert_eq!(inc, vec![vec![0.0], vec![2.0], vec![3.0]]);
        for (g, w) in got.snapshots.iter().zip(t0.snapshots.iter()) {
            assert_eq!(g.t, w.t);
            assert_eq!(g.int_state.counts, w.int_state.counts);
            assert_eq!(g.flows, w.flows, "flows must survive, not be dropped to zero");
        }
    }

    /// The forkable-subset boundary is an error with a name, not an empty path
    /// that would silently band over nothing (gh#727).
    #[test]
    fn read_trajectory_names_a_draw_with_no_saved_path() {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_io_traj_miss_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("trajectories.tsv");
        let mut t0 = Trajectory::new();
        t0.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        let draws = vec![PosteriorDraw {
            chain: 0, draw: 5, path: t0, incidence: vec![vec![0.0]],
        }];
        write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap();

        let err = read_trajectory(&path, &cols(), 0, 6).unwrap_err();
        assert!(
            err.contains("no saved path") && err.contains("forkable"),
            "the error must say the draw is outside the forkable subset, got: {err}"
        );
    }

    // ── The multi-key reader against the single-key one ────────────────────

    /// Column spec exercising every decoded kind at once: integer AND real
    /// compartments, flows, incidence. A reader that mixed up the (int, real)
    /// split or skipped the wrong trailing columns fails here.
    fn wide_cols() -> TrajColumnSpec {
        TrajColumnSpec {
            int_comps: vec!["S".into(), "I".into(), "R".into()],
            real_comps: vec!["P".into()],
            flows: vec!["flow_infection".into(), "flow_recovery".into()],
            incidence: vec!["inc_cases".into()],
        }
    }

    /// Three chains × four draws, every `(chain, draw)` carrying values unique
    /// to it at every snapshot — so a reader that bucketed a row under the
    /// wrong key, dropped a row, or duplicated one cannot pass by coincidence.
    /// Draw `d` gets `d + 2` snapshots, so the per-key row COUNT differs too.
    /// Returns the file plus the draws that went into it, so a test can check
    /// the reader against the WRITTEN values and not only against the other
    /// reader — the two share their row decode, so a differential alone would
    /// not see a decode bug.
    fn multi_key_fixture(
        dir: &Path,
        columns: &TrajColumnSpec,
    ) -> (std::path::PathBuf, Vec<PosteriorDraw>) {
        let mut draws = Vec::new();
        for chain in 0..3usize {
            for draw in 0..4usize {
                let n = draw + 2;
                let mut traj = Trajectory::new();
                let mut inc = Vec::new();
                for s in 0..n {
                    let tag = (chain * 100 + draw * 10 + s) as i64;
                    traj.push(Snapshot {
                        t: s as f64 * 0.5,
                        int_state: IntState::from_vec(vec![1000 - tag, tag, tag * 2]),
                        real_state: RealState::from_vec(vec![tag as f64 + 0.25]),
                        flows: Flows::Int(vec![tag as u64, (tag * 3) as u64]),
                    });
                    inc.push(vec![tag as f64 * 1.5]);
                }
                draws.push(PosteriorDraw { chain, draw, path: traj, incidence: inc });
            }
        }
        let path = dir.join("trajectories.tsv");
        write_trajectories_tsv(&path, &draws, columns, None, "h", "pgas", Granularity::Substep)
            .unwrap();
        (path, draws)
    }

    /// The one-pass multi-key reader must return EXACTLY what N calls of the
    /// single-key reader return, for every key: same snapshot count, same
    /// times, same integer and real compartments, same flows, same incidence
    /// rows, in the same order.
    ///
    /// This is the standing drift check between the two row-selection loops
    /// (`fit predict` reads the whole posterior's smoothing paths through the
    /// multi-key one; the number it prints is a published estimand). Both are
    /// also checked against the WRITTEN draws, because the two share their row
    /// decode — a differential alone would agree happily on a wrong value.
    #[test]
    fn read_trajectories_returns_exactly_what_was_written() {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_io_traj_multi_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let columns = wide_cols();
        let (path, written) = multi_key_fixture(&tmp, &columns);
        let keys: Vec<(usize, usize)> = written.iter().map(|d| (d.chain, d.draw)).collect();

        let many = read_trajectories(&path, &columns, &keys).expect("one-pass read");
        assert_eq!(many.len(), keys.len(), "one entry per requested key");

        for w in &written {
            let (chain, draw) = (w.chain, w.draw);
            let (many_traj, many_inc) =
                many.get(&(chain, draw)).expect("the map is total over the requested keys");

            assert_eq!(
                many_inc, &w.incidence,
                "incidence rows differ from what was WRITTEN for (chain {chain}, draw {draw})"
            );
            assert_eq!(
                many_traj.snapshots.len(),
                w.path.snapshots.len(),
                "snapshot count differs from what was WRITTEN for (chain {chain}, draw {draw})"
            );
            for (i, (m, w)) in many_traj
                .snapshots
                .iter()
                .zip(w.path.snapshots.iter())
                .enumerate()
            {
                // Ground truth: what the writer was handed.
                assert_eq!(m.t, w.t, "time differs from WRITTEN at snapshot {i}");
                assert_eq!(
                    m.int_state.counts, w.int_state.counts,
                    "integer compartments differ from WRITTEN at snapshot {i}"
                );
                assert_eq!(
                    m.real_state.values, w.real_state.values,
                    "real compartments differ from WRITTEN at snapshot {i}"
                );
                assert_eq!(m.flows, w.flows, "flows differ from WRITTEN at snapshot {i}");
            }
        }

        // A SUBSET request reads only what it asked for — the map is not the
        // whole file dressed up as a lookup.
        let subset = vec![(1usize, 2usize), (2, 0)];
        let got = read_trajectories(&path, &columns, &subset).expect("subset read");
        assert_eq!(got.len(), 2);
        assert!(got.contains_key(&(1, 2)) && got.contains_key(&(2, 0)));

        // A repeated key collapses rather than duplicating the path.
        let dup = read_trajectories(&path, &columns, &[(1, 2), (1, 2)]).expect("dup read");
        assert_eq!(dup.len(), 1);
        assert_eq!(
            dup[&(1, 2)].0.snapshots.len(),
            got[&(1, 2)].0.snapshots.len(),
            "a repeated key must not concatenate the path onto itself"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// An absent key is the single-key reader's refusal, verbatim — not an
    /// empty path standing in for a draw that has none, and not a quietly
    /// missing map entry the caller might read as "censored". The whole read
    /// fails, so a present key requested alongside it is not returned either.
    #[test]
    fn read_trajectories_refuses_an_absent_key_exactly_as_the_single_reader_does() {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_io_traj_multi_miss_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let columns = wide_cols();
        let (path, _written) = multi_key_fixture(&tmp, &columns);

        // (0,9) is not in the file; (0,1) is.
        let one_err = read_trajectory(&path, &columns, 0, 9).unwrap_err();
        let many_err = read_trajectories(&path, &columns, &[(0, 1), (0, 9)]).unwrap_err();
        assert_eq!(many_err, one_err, "the refusal must be the same text, naming the same key");
        assert!(many_err.contains("no saved path") && many_err.contains("forkable"));

        // Deterministic when several are absent: the FIRST absent key in the
        // requested order is the one named.
        let two = read_trajectories(&path, &columns, &[(0, 9), (7, 7)]).unwrap_err();
        assert_eq!(two, read_trajectory(&path, &columns, 0, 9).unwrap_err());

        // No keys ⇒ nothing read, empty map (the zero-call case).
        let none = read_trajectories(&path, &columns, &[]).expect("empty request");
        assert!(none.is_empty());
        let missing_file = read_trajectories(
            &tmp.join("does-not-exist.tsv"), &columns, &[],
        )
        .expect("an empty request reads no file at all");
        assert!(missing_file.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// [`Flows`] keeps `Int` and `Real` distinct because quantizing a real flow
    /// through `u64` silently zeroes sub-unit flows. BOTH readers refuse a
    /// real-flow file by name rather than rounding it.
    #[test]
    fn both_path_readers_refuse_a_real_flow_file() {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_io_traj_realflow_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let columns = TrajColumnSpec {
            int_comps: vec!["S".into()],
            real_comps: vec![],
            flows: vec!["flow_x".into()],
            incidence: vec![],
        };
        let mut t = Trajectory::new();
        // 0.4 of a case per step: rounding to u64 would report zero flow.
        t.push(Snapshot {
            t: 0.0,
            int_state: IntState::from_vec(vec![900]),
            real_state: RealState::from_vec(vec![]),
            flows: Flows::Real(vec![0.4]),
        });
        let draws = vec![PosteriorDraw { chain: 0, draw: 0, path: t, incidence: vec![] }];
        let path = tmp.join("trajectories.tsv");
        write_trajectories_tsv(&path, &draws, &columns, None, "h", "pgas", Granularity::Substep)
            .unwrap();

        let one = read_trajectory(&path, &columns, 0, 0).unwrap_err();
        let many = read_trajectories(&path, &columns, &[(0, 0)]).unwrap_err();
        assert!(one.contains("real-valued"), "got: {one}");
        assert_eq!(many, one, "the multi-key reader must refuse identically, not quantize");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn writes_tidy_long_header_and_stacked_rows() {
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("trajectories.tsv");

        let mut t0 = Trajectory::new();
        t0.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t0.push(snap(1.0, vec![97, 2, 1], vec![2, 1]));
        let mut t1 = Trajectory::new();
        t1.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t1.push(snap(1.0, vec![98, 2, 0], vec![1, 0]));

        let draws = vec![
            PosteriorDraw {
                chain: 0,
                draw: 5,
                path: t0,
                incidence: vec![vec![0.0], vec![2.0]],
            },
            PosteriorDraw {
                chain: 1,
                draw: 5,
                path: t1,
                incidence: vec![vec![0.0], vec![1.0]],
            },
        ];

        write_trajectories_tsv(&path, &draws, &cols(), None, "abc123", "pgas", Granularity::Substep)
            .unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("# camdl-trajectories v1"), "got: {}", lines[0]);
        assert!(lines[0].contains("method=pgas"));
        assert!(lines[0].contains("granularity=substep"));
        assert_eq!(
            lines[1],
            "chain\tdraw\ttime\tS\tI\tR\tflow_infection\tflow_recovery\tinc_cases"
        );
        // 2 draws × 2 snapshots = 4 data rows + header-comment + col-header.
        assert_eq!(lines.len(), 6);
        // First data row.
        assert_eq!(lines[2], "0\t5\t0\t99\t1\t0\t0\t0\t0");
        // The inc_cases column == the FlowSum value the producer computed.
        assert_eq!(lines[3], "0\t5\t1\t97\t2\t1\t2\t1\t2");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_state_at_round_trips_a_keyed_snapshot() {
        // Two draws, distinct states at the same times — so a wrong (chain,draw)
        // or wrong-time match would return the wrong vector, not silently pass.
        let mut t0 = Trajectory::new();
        t0.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t0.push(snap(7.0, vec![80, 15, 5], vec![19, 5]));
        t0.push(snap(14.0, vec![50, 30, 20], vec![30, 15]));
        let mut t1 = Trajectory::new();
        t1.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t1.push(snap(7.0, vec![70, 20, 10], vec![29, 10]));
        t1.push(snap(14.0, vec![40, 25, 35], vec![45, 20]));
        // cols() declares an inc_cases column, so each draw carries a matching
        // incidence row per snapshot — this also lets the test prove the reader
        // SKIPS the trailing flow_*/inc_* columns (mapping compartments by name).
        let draws = vec![
            PosteriorDraw {
                chain: 0,
                draw: 20,
                path: t0,
                incidence: vec![vec![0.0], vec![19.0], vec![30.0]],
            },
            PosteriorDraw {
                chain: 1,
                draw: 21,
                path: t1,
                incidence: vec![vec![0.0], vec![29.0], vec![45.0]],
            },
        ];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_read_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("trajectories.tsv");
        write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap();

        // (chain=0, draw=20) at T*=7 → that draw's second snapshot.
        let (int_s, real_s) = read_state_at(&path, &cols(), 0, 20, 7.0).unwrap();
        assert_eq!(int_s.counts, vec![80, 15, 5]);
        assert!(real_s.values.is_empty(), "this model has no real compartments");

        // The OTHER draw at the same T* → its OWN state (keying, not first-row).
        let (int_s2, _) = read_state_at(&path, &cols(), 1, 21, 7.0).unwrap();
        assert_eq!(int_s2.counts, vec![70, 20, 10]);

        // A T* strictly between snapshots → error (cadence does not contain T*).
        let err = read_state_at(&path, &cols(), 0, 20, 10.0).unwrap_err();
        assert!(err.contains("must coincide with a saved snapshot"), "got: {err}");

        // An unknown (chain, draw) → the same not-found error.
        let err2 = read_state_at(&path, &cols(), 9, 9, 7.0).unwrap_err();
        assert!(err2.contains("no saved snapshot"), "got: {err2}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// gh#697: the terminal snapshot of EVERY saved path, in one pass — each
    /// draw's own last row, and each draw's own terminal time.
    #[test]
    fn read_terminal_states_returns_each_draws_own_last_row() {
        // Two draws with DIFFERENT last times: draw 20 ends at 14, draw 21 at 7.
        // A reader that assumed one shared cadence — or that took the file's
        // last row for everyone — would return the same state twice.
        let mut t0 = Trajectory::new();
        t0.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t0.push(snap(7.0, vec![80, 15, 5], vec![19, 5]));
        t0.push(snap(14.0, vec![50, 30, 20], vec![30, 15]));
        let mut t1 = Trajectory::new();
        t1.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        t1.push(snap(7.0, vec![70, 20, 10], vec![29, 10]));
        let draws = vec![
            PosteriorDraw {
                chain: 0,
                draw: 20,
                path: t0,
                incidence: vec![vec![0.0], vec![19.0], vec![30.0]],
            },
            PosteriorDraw {
                chain: 1,
                draw: 21,
                path: t1,
                incidence: vec![vec![0.0], vec![29.0]],
            },
        ];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_term_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("trajectories.tsv");
        write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap();

        let got = read_terminal_states(&path, &cols()).unwrap();
        assert_eq!(got.len(), 2, "one entry per (chain, draw)");
        assert_eq!((got[0].chain, got[0].draw), (0, 20));
        assert_eq!(got[0].t, 14.0);
        assert_eq!(got[0].int_state.counts, vec![50, 30, 20]);
        assert_eq!((got[1].chain, got[1].draw), (1, 21));
        assert_eq!(got[1].t, 7.0, "a draw's own terminal time, not the file's max");
        assert_eq!(got[1].int_state.counts, vec![70, 20, 10]);
        assert!(got[0].real_state.values.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_state_at_recovers_real_compartments() {
        // A real-compartment column round-trips through the {:.6} format.
        let c = TrajColumnSpec {
            int_comps: vec!["S".into()],
            real_comps: vec!["P".into()],
            flows: vec!["flow_x".into()],
            incidence: vec![],
        };
        let mut t = Trajectory::new();
        t.push(Snapshot {
            t: 3.0,
            int_state: IntState::from_vec(vec![900]),
            real_state: RealState::from_vec(vec![1.5]),
            flows: Flows::Int(vec![0]),
        });
        let draws = vec![PosteriorDraw { chain: 0, draw: 0, path: t, incidence: vec![] }];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_real_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("t.tsv");
        write_trajectories_tsv(&path, &draws, &c, None, "h", "pgas", Granularity::Substep).unwrap();

        let (int_s, real_s) = read_state_at(&path, &c, 0, 0, 3.0).unwrap();
        assert_eq!(int_s.counts, vec![900]);
        assert_eq!(real_s.values, vec![1.5]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn no_incidence_streams_omits_inc_columns() {
        let c = TrajColumnSpec {
            int_comps: vec!["N".into()],
            real_comps: vec![],
            flows: vec!["flow_death".into()],
            incidence: vec![],
        };
        let mut t = Trajectory::new();
        t.push(snap(1.0, vec![950], vec![50]));
        let draws = vec![PosteriorDraw {
            chain: 0,
            draw: 0,
            path: t,
            incidence: vec![],
        }];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_noinc_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("t.tsv");
        write_trajectories_tsv(&path, &draws, &c, None, "h", "pgas", Granularity::Substep).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let header = text.lines().nth(1).unwrap();
        assert_eq!(header, "chain\tdraw\ttime\tN\tflow_death");
        assert!(!header.contains("inc_"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn declared_inc_columns_with_empty_incidence_is_a_hard_error() {
        // cols() declares an inc_cases column (n_inc=1). A draw that provides NO
        // incidence rows must error cleanly, not panic on an out-of-bounds index
        // at the per-row write — same contract as the other shape checks.
        let mut t = Trajectory::new();
        t.push(snap(0.0, vec![99, 1, 0], vec![0, 0]));
        let draws = vec![PosteriorDraw { chain: 0, draw: 0, path: t, incidence: vec![] }];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_inc0_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("t.tsv");
        let err = write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap_err();
        assert!(err.contains("incidence"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn shape_mismatch_is_a_hard_error() {
        // A snapshot with the wrong number of int compartments must error, not
        // silently truncate / pad.
        let mut t = Trajectory::new();
        t.push(snap(0.0, vec![1, 2], vec![0, 0])); // 2 ints, header wants 3
        let draws = vec![PosteriorDraw {
            chain: 0,
            draw: 0,
            path: t,
            incidence: vec![vec![0.0]],
        }];
        let tmp = std::env::temp_dir().join(format!("camdl_io_traj_err_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.join("t.tsv");
        let err = write_trajectories_tsv(&path, &draws, &cols(), None, "h", "pgas", Granularity::Substep)
            .unwrap_err();
        assert!(err.contains("integer compartments"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn manifest_records_conditioned_and_columns() {
        let m = TrajManifest {
            method: "pgas".into(),
            granularity: Granularity::Substep,
            n_chains: 2,
            n_draws: 8,
            columns: vec!["chain".into(), "draw".into(), "time".into(), "S".into(), "inc_cases".into()],
            model_hash: "abc".into(),
            conditioned: true,
            degeneracy_caveat: false,
            n_trajectories: 4,
            predictive_obs_file: None,
            calendar: CalendarMeta {
                time_unit: "days".into(),
                origin: Some("1910-01-01".into()),
                days_per_unit: 1.0,
            },
        };
        let json = m.to_json();
        assert_eq!(json["conditioned"], serde_json::json!(true));
        assert_eq!(json["granularity"], serde_json::json!("substep"));
        assert_eq!(json["n_trajectories"], serde_json::json!(4));
        assert_eq!(json["columns"][4], serde_json::json!("inc_cases"));
        assert_eq!(json["predictive_obs_file"], serde_json::Value::Null);
        // Calendar semantics travel with the manifest.
        assert_eq!(json["calendar"]["time_unit"], serde_json::json!("days"));
        assert_eq!(json["calendar"]["origin"], serde_json::json!("1910-01-01"));
        assert_eq!(json["calendar"]["days_per_unit"], serde_json::json!(1.0));
    }

    #[test]
    fn manifest_calendar_null_when_unanchored() {
        let cal = CalendarMeta { time_unit: "days".into(), origin: None, days_per_unit: 1.0 };
        let json = cal.to_json();
        assert_eq!(json["time_unit"], serde_json::json!("days"));
        assert_eq!(json["origin"], serde_json::Value::Null);
        assert_eq!(json["days_per_unit"], serde_json::json!(1.0));
    }
}
