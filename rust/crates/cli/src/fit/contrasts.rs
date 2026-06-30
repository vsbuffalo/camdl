//! Stage C of counterfactual contrasts (proposal 2026-06-25): the two-arm replay
//! reducer that CONSUMES the IR [`ir::contrast::Contrast`] node. Auto-emitted by
//! `fit predict` when the model declares any `contrasts {}`.
//!
//! Per forkable posterior draw `i`, per `run` referenced in a contrast (a scenario
//! or the reserved `fitted` no-overlay run):
//!   1. resolve that arm's θ via the 5-tier resolver
//!      ([`crate::params_resolver::resolve_parameters`]) — the fitted draw at tier
//!      3.5, the scenario `set`/`scale` at tier 4 (`fitted` ⇒ no overlay);
//!   2. fork from the smoothed `X_i(T*)` — chain_binomial reads the saved path with
//!      [`io::trajectories::read_state_at`] and resumes via
//!      [`sim::chain_binomial::Resume`]`{ start: Some(..) }`. CRN: both arms share
//!      `X_i(T*)` AND the per-draw seed, so the firing substep is byte-identical at
//!      the fork; post-fork noise desyncs by design;
//!   3. run `[T*, to]` (the contrast window), evaluate the operand quantities with
//!      the shared [`sim::quantity::QuantityEvaluator`];
//!   4. difference elementwise, shape- and dimension-preserving;
//!   5. band over the forkable subset → `contrasts/<name>.tsv`.
//!
//! `T*` is `window.from`; the accumulation window is `[from, to]`. The fork runs
//! exactly that span, so a series operand's time axis and a reduced operand's
//! reduction window both scope to `[from, to]` for free.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use indexmap::{IndexMap, IndexSet};

use ir::contrast::{Contrast, ContrastExpr, RunNamespace};
use ir::expr::BinOp;
use sim::quantity::{QuantityDrawValue, QuantityEvaluator, QuantityResult};

use crate::args::types::ForwardBackend;
use crate::fit::joint::{resolve_joint, LatentPath};
use crate::fit::predict::{band, fmt_time, fmt_value, write_tsv, QUANTILE_LEVELS};
use crate::params_resolver::{resolve_parameters, ParameterInputs};

// ── Shape model (the per-draw value of a contrast operand / body) ───────────────

/// The shape axis a contrast operand carries, inherited from the quantity:
/// `Series` (one value per output snapshot) or `Scalar` (time collapsed). A
/// BinOp's two sides must agree on this axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Series,
    Scalar,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Series => "series",
            Shape::Scalar => "scalar",
        }
    }
}

/// One stratum leaf's value for one draw — either an arm's evaluated operand or a
/// differenced result. `key` is the canonical (sorted) stratum, the match axis
/// across operands; `dims`/`levels` are the declaration-order header/row cells.
#[derive(Clone, Debug)]
struct LeafValue {
    key: Vec<(String, String)>,
    dims: Vec<String>,
    levels: Vec<String>,
    payload: LeafPayload,
}

#[derive(Clone, Debug)]
enum LeafPayload {
    /// One f64 per snapshot time (parallel to [`ShapedValue::times`]).
    Series(Vec<f64>),
    /// A single scalar (possibly right-censored — e.g. a `time_of_max` that never
    /// fired). Censoring propagates through arithmetic.
    Scalar(QuantityDrawValue),
}

/// A whole contrast (sub)expression's value for ONE draw: a shape-tagged family of
/// stratum leaves, the series time axis, and the resolved dimension `(P, T)`.
#[derive(Clone, Debug)]
struct ShapedValue {
    shape: Shape,
    leaves: Vec<LeafValue>,
    /// Series time axis (the arm's `[from, to]` snapshot times); empty for scalar.
    times: Vec<f64>,
    /// Resolved dimension `(P exponent, T exponent)`, or `None` when undetermined.
    dim: Option<(i32, i32)>,
}

// ── One arm's evaluated quantities for one draw ─────────────────────────────────

/// The result of replaying one arm (run) for one draw: every model quantity
/// evaluated (in `model.quantities` order) over the arm's `[from, to]` trajectory,
/// plus that trajectory's snapshot times.
struct ArmDrawResult {
    quant: Vec<QuantityResult>,
    times: Vec<f64>,
}

// ── Entry point ─────────────────────────────────────────────────────────────────

/// Compute and write every `contrasts/<name>.tsv` under `segment`. Called by
/// `fit predict` only when `model.contrasts` is non-empty, so a model without a
/// `contrasts {}` block is byte-identical to before.
///
/// Returns the written file paths. A non-forkable fit (no saved paths) or an ODE
/// fit emits NO file and a located stderr note (not a crash). A shape or dimension
/// mismatch in a contrast body is a hard error (it fails `fit predict`).
pub fn emit_contrasts(
    segment: &Path,
    stage: Option<&str>,
    model: &ir::Model,
    backend: ForwardBackend,
    seed: u64,
) -> Result<Vec<PathBuf>, String> {
    // The forkable subset, classified by LatentPath. A point-estimate fit is
    // already refused upstream by `fit predict` (PlugIn treatment), so here we
    // only see a posterior cloud.
    let joint = resolve_joint(&segment.to_string_lossy(), stage)?;
    if joint.n_forkable == 0 {
        eprintln!(
            "fit predict: skipping {} contrast(s) — this fit has no forkable posterior \
             draws (0/{} have a usable latent state X(T*)). A conditioned contrast needs \
             a stochastic fit that saved smoothed latent paths (PGAS), or a deterministic \
             (ODE) fit. PMMH/PF fits do not yet save paths.",
            model.contrasts.len(),
            joint.n_total,
        );
        return Ok(Vec::new());
    }

    // v1 forks only the chain_binomial conditioned path (the headline CRN case).
    // ODE (Deterministic) is a named, valid case in the proposal but its post-fork
    // re-integration seam is not wired here — surface it loudly, never silently.
    if backend != ForwardBackend::ChainBinomial {
        eprintln!(
            "fit predict: skipping {} contrast(s) — counterfactual contrasts fork only \
             chain_binomial posterior fits in this build; this fit ran on {}. (ODE \
             deterministic forking is a named gh#322 follow-up.)",
            model.contrasts.len(),
            backend.as_str(),
        );
        return Ok(Vec::new());
    }

    // Honest denominator: a contrast bands over the forkable subset only.
    if joint.n_forkable < joint.n_total {
        eprintln!(
            "fit predict: contrasts band over the forkable subset — {}/{} draws have a \
             saved latent path X(T*); the remaining {} are skipped.",
            joint.n_forkable,
            joint.n_total,
            joint.n_total - joint.n_forkable,
        );
    }

    // The forkable draws, in file order: (params θ_i, the saved-path locator).
    let forkable: Vec<(HashMap<String, f64>, usize, usize)> = joint
        .draws
        .iter()
        .filter_map(|d| match &d.latent {
            LatentPath::Sampled { chain, draw } => Some((d.params.clone(), *chain, *draw)),
            // Deterministic/NotSaved cannot occur here: NotSaved is filtered by
            // n_forkable, Deterministic only on ODE (rejected above).
            _ => None,
        })
        .collect();
    if forkable.is_empty() {
        // chain_binomial with n_forkable>0 must yield Sampled paths; a mismatch is
        // an upstream classifier bug, surfaced rather than silently emitting nothing.
        return Err(
            "internal: forkable count > 0 on a chain_binomial fit but no Sampled latent \
             paths were classified — the (θ,X) join is inconsistent"
                .to_string(),
        );
    }

    // Validate each contrast up front; an observation-namespace operand (or a
    // quantity that reduces an observation stream) is a named v1 deferral — skip
    // it with a loud note rather than fork an undefined obs-time axis.
    let mut to_emit: Vec<&Contrast> = Vec::new();
    for c in &model.contrasts {
        match validate_contrast(c, model) {
            Ok(()) => to_emit.push(c),
            Err(reason) => eprintln!("fit predict: skipping contrast '{}' — {reason}", c.name),
        }
    }
    if to_emit.is_empty() {
        return Ok(Vec::new());
    }

    // Compile each arm's model ONCE (the scenario filter is draw-independent: it
    // toggles interventions, never compartments/structure, so the structure is
    // window-independent too). Per draw we re-resolve only the parameter VALUES and
    // rebuild the param vector; the fork window is applied at replay time.
    let mut runs_union: Vec<String> = Vec::new();
    for c in &to_emit {
        collect_runs(&c.body, &mut runs_union);
    }
    // A complete parameter set (the first forkable draw) to satisfy each arm's
    // compilation — estimated parameters have no model default.
    let placeholder: Vec<(String, f64)> =
        forkable[0].0.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let mut arms: HashMap<String, Arm> = HashMap::new();
    for run in &runs_union {
        if !arms.contains_key(run) {
            arms.insert(run.clone(), Arm::build(model, run, &placeholder)?);
        }
    }

    let stage_dir = resolve_stage_dir(segment, stage)?;
    let col_spec = io::trajectories::TrajColumnSpec::from_model(model, &[]);
    let dt = model.simulation.dt.unwrap_or(1.0);

    // Per contrast: replay its runs over ITS window for every forkable draw, walk
    // the body into a per-draw ShapedValue, band over the forkable subset, write.
    let mut written = Vec::new();
    for c in &to_emit {
        let mut runs_c: Vec<String> = Vec::new();
        collect_runs(&c.body, &mut runs_c);
        let mut shaped: Vec<ShapedValue> = Vec::with_capacity(forkable.len());

        for (draw_pos, (params_i, chain, draw)) in forkable.iter().enumerate() {
            // CRN: both arms of THIS draw share one seed (run name is NOT mixed in),
            // so the firing substep is byte-identical at the fork.
            let arm_seed = crate::util::derive_chain_seed(seed, draw_pos);
            let mut draw_results: HashMap<String, ArmDrawResult> = HashMap::new();
            for run in &runs_c {
                let res = arms[run].replay(
                    model, run, params_i, &col_spec, &stage_dir, *chain, *draw, dt, arm_seed,
                    c.window,
                )?;
                draw_results.insert(run.clone(), res);
            }
            shaped.push(eval_body(&c.body, model, &draw_results, &c.name)?);
        }

        let content = band_and_render(&c.name, &shaped)?;
        written.push(write_tsv(segment, "contrasts", &c.name, &content)?);
    }
    Ok(written)
}

/// `<stage_dir>` = the directory holding `draws.tsv` and the `chain_*/` trajectory
/// subdirs, for [`io::trajectories::read_state_at`].
fn resolve_stage_dir(segment: &Path, stage: Option<&str>) -> Result<PathBuf, String> {
    let pref = crate::posterior_draws::resolve_posterior_draws(&segment.to_string_lossy(), stage)?;
    pref.draws_path
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or_else(|| format!("draws path has no parent: {}", pref.draws_path.display()))
}

// ── One arm: a per-run compiled model + quantity evaluator ──────────────────────

struct Arm {
    compiled: std::sync::Arc<sim::CompiledModel>,
    quant_eval: QuantityEvaluator,
}

impl Arm {
    /// Build the arm for `run` (a scenario preset, or `fitted` for no overlay):
    /// resolve the scenario filter + a placeholder parameter set (draw values are
    /// re-resolved per draw), compile once, and build the quantity evaluator.
    ///
    /// `placeholder` is a complete parameter set (a real draw) used only to satisfy
    /// compilation — an estimated parameter has no model default, so the structure
    /// must be materialized against concrete values. The model STRUCTURE (compartments,
    /// transitions, interventions after the scenario filter) is value-independent, so
    /// any draw serves; per-draw we rebuild the parameter vector below.
    fn build(model: &ir::Model, run: &str, placeholder: &[(String, f64)]) -> Result<Arm, String> {
        let resolved = resolve_arm(model, run, placeholder)?;
        let compiled = std::sync::Arc::new(
            sim::CompiledModel::new(resolved.model)
                .map_err(|e| format!("contrast arm '{run}': compiling model: {e:?}"))?,
        );
        let quant_eval = QuantityEvaluator::new(&model.quantities, compiled.as_ref())
            .map_err(|e| format!("contrast arm '{run}': building quantity evaluator: {e}"))?;
        Ok(Arm { compiled, quant_eval })
    }

    /// Replay this arm for one draw over the contrast window: resolve the draw's θ
    /// (+ scenario overlay), fork from the smoothed `X_i(T*)` (`T* = window.from`),
    /// run `[from, to]`, and evaluate every quantity over that span. The reduction
    /// window and a series' time axis both scope to `[from, to]` because the fork
    /// runs exactly that span.
    #[allow(clippy::too_many_arguments)]
    fn replay(
        &self,
        model: &ir::Model,
        run: &str,
        params_i: &HashMap<String, f64>,
        col_spec: &io::trajectories::TrajColumnSpec,
        stage_dir: &Path,
        chain: usize,
        draw: usize,
        dt: f64,
        arm_seed: u64,
        window: ir::contrast::ContrastWindow,
    ) -> Result<ArmDrawResult, String> {
        // θ_i at tier 3.5 (draw row), scenario `set`/`scale` at tier 4.
        let overrides: Vec<(String, f64)> =
            params_i.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let resolved = resolve_arm(model, run, &overrides)?;
        let mut pvec = self.compiled.default_params.clone();
        for rp in &resolved.params {
            if let Some(&idx) = self.compiled.param_index.get(rp.name.as_str()) {
                pvec[idx] = rp.value;
            }
        }

        // Read the smoothed latent state X_i(T*) from the saved path. The on-disk
        // chain dir is 1-based (`chain_{N+1}`); the in-file `chain` column is the
        // 0-based key `read_state_at` matches.
        let traj_path = stage_dir.join(format!("chain_{}", chain + 1)).join("trajectories.tsv");
        let (int_s, real_s) =
            io::trajectories::read_state_at(&traj_path, col_spec, chain, draw, window.from)
                .map_err(|e| format!("contrast arm '{run}': reading X(T*={}): {e}", window.from))?;

        // Fork from X_i(T*): inject the state at cfg.t_start = T*, fresh RNG from the
        // shared per-draw seed (CRN at the fork; post-fork noise desyncs by design).
        let cfg = sim::ChainBinomialConfig { t_start: window.from, t_end: window.to, dt };
        let ss = sim::chain_binomial::StartState { int_s, real_s, rng: None };
        let traj = sim::chain_binomial::run_chain_binomial_with_observer(
            self.compiled.as_ref(),
            &pvec,
            arm_seed,
            &cfg,
            None,
            None,
            sim::chain_binomial::Resume { start: Some(&ss), capture_final_rng: None },
        )
        .map_err(|e| format!("contrast arm '{run}': forking at T*={}: {e:?}", window.from))?;

        // Clip to [from, to] (a no-op for the fork, which already starts at T*;
        // a guard for any tail snapshot past `to`).
        let clipped = clip_trajectory(traj, window.from, window.to);
        let times: Vec<f64> = clipped.snapshots.iter().map(|s| s.t).collect();
        let quant = self.quant_eval.eval_draw(&pvec, &clipped, self.compiled.as_ref(), None);
        Ok(ArmDrawResult { quant, times })
    }
}

/// Keep only the snapshots whose time lies in `[from, to]` (with a small tolerance
/// at each end). The fork already starts at `from`; this guards an off-grid `to`.
fn clip_trajectory(mut traj: sim::Trajectory, from: f64, to: f64) -> sim::Trajectory {
    const EPS: f64 = 1e-9;
    traj.snapshots.retain(|s| s.t >= from - EPS && s.t <= to + EPS);
    traj
}

/// Resolve one arm's parameters: the draw overlay at tier 3.5, the scenario
/// `set`/`scale`/`enable`/`disable` at tier 4 (`fitted` ⇒ no scenario). Returns the
/// filtered, valued model + the per-parameter resolution.
fn resolve_arm(
    model: &ir::Model,
    run: &str,
    overrides: &[(String, f64)],
) -> Result<crate::params_resolver::ResolvedParameters, String> {
    let empty_tables: HashMap<String, PathBuf> = HashMap::new();
    let empty_fixed: IndexMap<String, f64> = IndexMap::new();
    let empty_estimate: IndexSet<String> = IndexSet::new();
    let scenario = if run == crate::args::FITTED { None } else { Some(run) };
    let inputs = ParameterInputs {
        model,
        scenario,
        adhoc_enable: &[],
        adhoc_disable: &[],
        scenario_inline_name: None,
        scenario_inline_set: &[],
        scenario_inline_scale: &[],
        point_overrides: overrides,
        fixed_cli: &[],
        fixed_files: &[],
        fit_toml_fixed: &empty_fixed,
        fit_toml_estimate: &empty_estimate,
        table_files: &empty_tables,
    };
    resolve_parameters(inputs).map_err(|e| format!("contrast arm '{run}': {e}"))
}

// ── Contrast body walk: per-draw evaluation ─────────────────────────────────────

/// Collect every distinct run named by a `RunMember` in `expr` (preserving first
/// appearance).
fn collect_runs(expr: &ContrastExpr, out: &mut Vec<String>) {
    match expr {
        ContrastExpr::RunMember { run, .. } => {
            if !out.iter().any(|r| r == run) {
                out.push(run.clone());
            }
        }
        ContrastExpr::BinOp { left, right, .. } => {
            collect_runs(left, out);
            collect_runs(right, out);
        }
    }
}

/// Validate a contrast for the v1 reducer: reject the `observations` namespace and
/// any quantity that reduces an observation stream (the obs-time axis over a fork
/// window is a named deferral). Returns `Err(reason)` to skip-with-note.
fn validate_contrast(c: &Contrast, model: &ir::Model) -> Result<(), String> {
    fn walk(expr: &ContrastExpr, model: &ir::Model) -> Result<(), String> {
        match expr {
            ContrastExpr::RunMember { ns: RunNamespace::Observations, member, run } => Err(format!(
                "the observations namespace (`{run}.observations.{member}`) is deferred in \
                 this build; reduce the stream inside a `quantities {{}}` entry and contrast \
                 the named quantity instead"
            )),
            ContrastExpr::RunMember { ns: RunNamespace::Quantities, member, .. } => {
                if quantity_reduces_observations(model, member) {
                    Err(format!(
                        "quantity '{member}' reduces an `observations.<stream>` source; \
                         observation-sourced contrasts are deferred in this build (the \
                         obs-time axis over a counterfactual window is unspecified)"
                    ))
                } else {
                    Ok(())
                }
            }
            ContrastExpr::BinOp { left, right, .. } => {
                walk(left, model)?;
                walk(right, model)
            }
        }
    }
    walk(&c.body, model)
}

/// Whether the named quantity (any leaf, transitively through `Derived` QRefs)
/// reduces an `observations.<stream>` source.
fn quantity_reduces_observations(model: &ir::Model, name: &str) -> bool {
    use ir::quantity::{QuantityBody, QuantitySource, ScalarExpr};
    fn qrefs(se: &ScalarExpr, out: &mut Vec<String>) {
        match se {
            ScalarExpr::Const(_) | ScalarExpr::Param(_) => {}
            ScalarExpr::QRef(q) => out.push(q.name.clone()),
            ScalarExpr::UnOp { arg, .. } => qrefs(arg, out),
            ScalarExpr::BinOp { left, right, .. } => {
                qrefs(left, out);
                qrefs(right, out);
            }
            ScalarExpr::Cond { pred, then, else_ } => {
                qrefs(pred, out);
                qrefs(then, out);
                qrefs(else_, out);
            }
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack = vec![name.to_string()];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.clone()) {
            continue;
        }
        for q in model.quantities.iter().filter(|q| q.name == n) {
            match &q.body {
                QuantityBody::Reduced { source: QuantitySource::Observation { .. }, .. } => {
                    return true
                }
                QuantityBody::Reduced { .. } => {}
                QuantityBody::Derived(se) => {
                    let mut refs = Vec::new();
                    qrefs(se, &mut refs);
                    stack.extend(refs);
                }
            }
        }
    }
    false
}

/// Evaluate a contrast body for ONE draw against the per-run arm results.
fn eval_body(
    expr: &ContrastExpr,
    model: &ir::Model,
    results: &HashMap<String, ArmDrawResult>,
    cname: &str,
) -> Result<ShapedValue, String> {
    match expr {
        ContrastExpr::RunMember { run, ns, member } => {
            if *ns != RunNamespace::Quantities {
                // validate_contrast already excluded these; defensive.
                return Err(format!(
                    "contrast '{cname}': the observations namespace is unsupported here"
                ));
            }
            let dr = results.get(run).ok_or_else(|| {
                format!("contrast '{cname}': no replayed arm for run '{run}'")
            })?;
            run_member_value(model, member, dr, run, cname)
        }
        ContrastExpr::BinOp { op, left, right } => {
            let lv = eval_body(left, model, results, cname)?;
            let rv = eval_body(right, model, results, cname)?;
            combine(op, lv, rv, cname)
        }
    }
}

/// Build a `RunMember`'s ShapedValue from an arm's evaluated quantities: select the
/// leaves named `member` (one per stratum cell), in `model.quantities` order.
fn run_member_value(
    model: &ir::Model,
    member: &str,
    dr: &ArmDrawResult,
    run: &str,
    cname: &str,
) -> Result<ShapedValue, String> {
    let idxs: Vec<usize> = model
        .quantities
        .iter()
        .enumerate()
        .filter(|(_, q)| q.name == member)
        .map(|(i, _)| i)
        .collect();
    if idxs.is_empty() {
        return Err(format!(
            "contrast '{cname}': run '{run}' has no quantity named '{member}'"
        ));
    }
    let shape = match &dr.quant[idxs[0]] {
        QuantityResult::Series(_) => Shape::Series,
        QuantityResult::Scalar(_) => Shape::Scalar,
    };
    let dim = model.quantities[idxs[0]].dimension;
    let mut leaves = Vec::with_capacity(idxs.len());
    for &i in &idxs {
        let q = &model.quantities[i];
        let key: Vec<(String, String)> = {
            let mut v: Vec<(String, String)> =
                q.stratum.iter().map(|s| (s.dim.clone(), s.level.clone())).collect();
            v.sort();
            v
        };
        let dims: Vec<String> = q.stratum.iter().map(|s| s.dim.clone()).collect();
        let levels: Vec<String> = q.stratum.iter().map(|s| s.level.clone()).collect();
        let payload = match &dr.quant[i] {
            QuantityResult::Series(s) => LeafPayload::Series(s.clone()),
            QuantityResult::Scalar(v) => LeafPayload::Scalar(*v),
        };
        leaves.push(LeafValue { key, dims, levels, payload });
    }
    let times = if shape == Shape::Series { dr.times.clone() } else { Vec::new() };
    Ok(ShapedValue { shape, leaves, times, dim })
}

/// Combine two operands elementwise, preserving shape — the shape-agreement and
/// dimension-agreement checks live here (located errors).
fn combine(op: &BinOp, l: ShapedValue, r: ShapedValue, cname: &str) -> Result<ShapedValue, String> {
    if l.shape != r.shape {
        return Err(format!(
            "contrast '{cname}': operand shape mismatch — left is a {}, right is a {}. \
             Operands must share shape (both series, or both scalar).",
            l.shape.label(),
            r.shape.label(),
        ));
    }
    let dim = combine_dim(op, l.dim, r.dim, cname)?;

    if l.shape == Shape::Series {
        let n = l.times.len();
        if r.times.len() != n {
            return Err(format!(
                "contrast '{cname}': series operands have different time axes ({} vs {} \
                 snapshots) — the arms must share the output cadence over the window",
                n,
                r.times.len()
            ));
        }
    }

    // Index the right leaves by canonical stratum key; the two operand leaf SETS
    // must match exactly (same strata), else a stratification mismatch.
    let mut right_by_key: HashMap<&Vec<(String, String)>, &LeafValue> = HashMap::new();
    for lf in &r.leaves {
        right_by_key.insert(&lf.key, lf);
    }
    if right_by_key.len() != r.leaves.len() {
        return Err(format!("contrast '{cname}': right operand has duplicate stratum cells"));
    }
    if l.leaves.len() != r.leaves.len() {
        return Err(format!(
            "contrast '{cname}': operand stratification mismatch ({} vs {} cells)",
            l.leaves.len(),
            r.leaves.len()
        ));
    }

    let mut leaves = Vec::with_capacity(l.leaves.len());
    for lf in &l.leaves {
        let rf = right_by_key.get(&lf.key).ok_or_else(|| {
            format!(
                "contrast '{cname}': stratum cell {:?} is present on one operand but not the \
                 other — operands must share strata",
                lf.key
            )
        })?;
        let payload = combine_payload(op, &lf.payload, &rf.payload, cname)?;
        leaves.push(LeafValue {
            key: lf.key.clone(),
            dims: lf.dims.clone(),
            levels: lf.levels.clone(),
            payload,
        });
    }
    Ok(ShapedValue { shape: l.shape, leaves, times: l.times, dim })
}

/// Dimension agreement: `Add`/`Sub` require equal dimensions (a `deaths - rate`
/// contrast is a located error); `Mul`/`Div` add/subtract exponents; other ops
/// keep the left dimension. Mirrors the OCaml E297 check, guarding the Rust path.
fn combine_dim(
    op: &BinOp,
    l: Option<(i32, i32)>,
    r: Option<(i32, i32)>,
    cname: &str,
) -> Result<Option<(i32, i32)>, String> {
    match (l, r) {
        (Some(a), Some(b)) => match op {
            BinOp::Add | BinOp::Sub => {
                if a != b {
                    return Err(format!(
                        "contrast '{cname}': dimension mismatch — left has dimension P^{} T^{}, \
                         right has P^{} T^{}. A contrast difference requires equal dimensions.",
                        a.0, a.1, b.0, b.1
                    ));
                }
                Ok(Some(a))
            }
            BinOp::Mul => Ok(Some((a.0 + b.0, a.1 + b.1))),
            BinOp::Div => Ok(Some((a.0 - b.0, a.1 - b.1))),
            _ => Ok(Some(a)),
        },
        _ => Ok(None),
    }
}

/// Combine two leaf payloads elementwise. Series fold time-by-time; scalars
/// propagate censoring (a difference touching a censored endpoint is censored).
fn combine_payload(
    op: &BinOp,
    l: &LeafPayload,
    r: &LeafPayload,
    cname: &str,
) -> Result<LeafPayload, String> {
    match (l, r) {
        (LeafPayload::Series(a), LeafPayload::Series(b)) => {
            if a.len() != b.len() {
                return Err(format!(
                    "contrast '{cname}': series length mismatch ({} vs {})",
                    a.len(),
                    b.len()
                ));
            }
            Ok(LeafPayload::Series(
                a.iter().zip(b).map(|(x, y)| apply_bin(op, *x, *y)).collect(),
            ))
        }
        (LeafPayload::Scalar(a), LeafPayload::Scalar(b)) => {
            use QuantityDrawValue::*;
            let v = match (a, b) {
                (Censored, _) | (_, Censored) => Censored,
                (Value(x), Value(y)) => Value(apply_bin(op, *x, *y)),
            };
            Ok(LeafPayload::Scalar(v))
        }
        // shape agreement above guarantees matching variants.
        _ => Err(format!("contrast '{cname}': internal payload shape mismatch")),
    }
}

fn apply_bin(op: &BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Pow => a.powf(b),
        BinOp::Mod => a.rem_euclid(b),
        BinOp::Min => a.min(b),
        BinOp::Max => a.max(b),
        BinOp::Eq => (a == b) as i32 as f64,
        BinOp::Neq => (a != b) as i32 as f64,
        BinOp::Lt => (a < b) as i32 as f64,
        BinOp::Gt => (a > b) as i32 as f64,
        BinOp::Le => (a <= b) as i32 as f64,
        BinOp::Ge => (a >= b) as i32 as f64,
    }
}

// ── Banding + tidy/long render ──────────────────────────────────────────────────

/// Band the per-draw ShapedValues of one contrast into a tidy/long TSV:
/// `[time] <dims…> q05 q25 q50 q75 q95 mean n_forkable`, keyed by `(stratum, time)`
/// as the shape carries.
fn band_and_render(name: &str, draws: &[ShapedValue]) -> Result<String, String> {
    let template = draws
        .first()
        .ok_or_else(|| format!("contrast '{name}': no forkable draws to band"))?;
    let shape = template.shape;
    let dims = template.leaves.first().map(|l| l.dims.clone()).unwrap_or_default();

    // Cross-draw consistency: same shape, same per-leaf stratum order, same times.
    for (d, sv) in draws.iter().enumerate() {
        if sv.shape != shape {
            return Err(format!(
                "contrast '{name}': draw {d} has shape {} but draw 0 is {}",
                sv.shape.label(),
                shape.label()
            ));
        }
        if sv.leaves.len() != template.leaves.len() {
            return Err(format!(
                "contrast '{name}': draw {d} has {} leaves but draw 0 has {}",
                sv.leaves.len(),
                template.leaves.len()
            ));
        }
        if shape == Shape::Series && sv.times.len() != template.times.len() {
            return Err(format!(
                "contrast '{name}': draw {d} has {} snapshots but draw 0 has {}",
                sv.times.len(),
                template.times.len()
            ));
        }
        for (li, lf) in sv.leaves.iter().enumerate() {
            if lf.key != template.leaves[li].key {
                return Err(format!(
                    "contrast '{name}': draw {d} leaf {li} stratum {:?} ≠ draw 0 {:?}",
                    lf.key, template.leaves[li].key
                ));
            }
        }
    }

    // Header.
    let mut header: Vec<String> = Vec::new();
    if shape == Shape::Series {
        header.push("time".to_string());
    }
    header.extend(dims.iter().cloned());
    for (_, label) in QUANTILE_LEVELS {
        header.push((*label).to_string());
    }
    header.push("mean".to_string());
    header.push("n_forkable".to_string());

    let mut out = header.join("\t");
    out.push('\n');

    // Rows, in template leaf order, then (series) time order.
    for (li, lf) in template.leaves.iter().enumerate() {
        match shape {
            Shape::Series => {
                for (ti, &t) in template.times.iter().enumerate() {
                    let col: Vec<f64> = draws
                        .iter()
                        .filter_map(|sv| match &sv.leaves[li].payload {
                            LeafPayload::Series(s) => Some(s[ti]),
                            LeafPayload::Scalar(_) => None,
                        })
                        .collect();
                    let mut cells: Vec<String> = vec![fmt_time(t)];
                    cells.extend(lf.levels.iter().cloned());
                    cells.extend(band_cells(&col).map_err(|e| {
                        format!("contrast '{name}' at t={}: {e}", fmt_time(t))
                    })?);
                    out.push_str(&cells.join("\t"));
                    out.push('\n');
                }
            }
            Shape::Scalar => {
                let col: Vec<f64> = draws
                    .iter()
                    .filter_map(|sv| match &sv.leaves[li].payload {
                        LeafPayload::Scalar(QuantityDrawValue::Value(x)) => Some(*x),
                        LeafPayload::Scalar(QuantityDrawValue::Censored) => None,
                        LeafPayload::Series(_) => None,
                    })
                    .collect();
                let mut cells: Vec<String> = Vec::new();
                cells.extend(lf.levels.iter().cloned());
                cells.extend(
                    band_cells(&col).map_err(|e| format!("contrast '{name}': {e}"))?,
                );
                out.push_str(&cells.join("\t"));
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// The band/value cells for one cell's per-draw finite values:
/// `q05 q25 q50 q75 q95 mean n_forkable`. An empty column (every draw censored)
/// renders empty quantile + mean cells and `n_forkable = 0` — never a fabricated
/// band. A non-finite value is rejected by [`band`] (an upstream bug).
fn band_cells(col: &[f64]) -> Result<Vec<String>, String> {
    let mut cells: Vec<String> = Vec::with_capacity(QUANTILE_LEVELS.len() + 2);
    if col.is_empty() {
        for _ in QUANTILE_LEVELS {
            cells.push(String::new());
        }
        cells.push(String::new()); // mean
        cells.push("0".to_string()); // n_forkable
        return Ok(cells);
    }
    let bands = band(col)?;
    cells.extend(bands.iter().map(|b| fmt_value(*b)));
    let mean = col.iter().sum::<f64>() / col.len() as f64;
    cells.push(fmt_value(mean));
    cells.push(col.len().to_string());
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_leaf(x: f64) -> LeafValue {
        LeafValue {
            key: vec![],
            dims: vec![],
            levels: vec![],
            payload: LeafPayload::Scalar(QuantityDrawValue::Value(x)),
        }
    }

    fn series_leaf(xs: &[f64]) -> LeafValue {
        LeafValue {
            key: vec![],
            dims: vec![],
            levels: vec![],
            payload: LeafPayload::Series(xs.to_vec()),
        }
    }

    #[test]
    fn combine_scalar_subtracts_and_keeps_dimension() {
        let l = ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(10.0)], times: vec![], dim: Some((1, 0)) };
        let r = ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(3.0)], times: vec![], dim: Some((1, 0)) };
        let out = combine(&BinOp::Sub, l, r, "averted").unwrap();
        assert_eq!(out.dim, Some((1, 0)));
        match &out.leaves[0].payload {
            LeafPayload::Scalar(QuantityDrawValue::Value(v)) => assert_eq!(*v, 7.0),
            other => panic!("expected scalar 7.0, got {other:?}"),
        }
    }

    #[test]
    fn combine_series_subtracts_elementwise() {
        let l = ShapedValue { shape: Shape::Series, leaves: vec![series_leaf(&[5.0, 6.0, 7.0])], times: vec![0.0, 1.0, 2.0], dim: Some((1, 0)) };
        let r = ShapedValue { shape: Shape::Series, leaves: vec![series_leaf(&[1.0, 2.0, 3.0])], times: vec![0.0, 1.0, 2.0], dim: Some((1, 0)) };
        let out = combine(&BinOp::Sub, l, r, "curve").unwrap();
        match &out.leaves[0].payload {
            LeafPayload::Series(s) => assert_eq!(s, &[4.0, 4.0, 4.0]),
            other => panic!("expected series, got {other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_is_a_located_error() {
        let series = ShapedValue { shape: Shape::Series, leaves: vec![series_leaf(&[1.0, 2.0])], times: vec![0.0, 1.0], dim: Some((1, 0)) };
        let scalar = ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(3.0)], times: vec![], dim: Some((1, 0)) };
        let err = combine(&BinOp::Sub, series, scalar, "bad").unwrap_err();
        assert!(err.contains("shape mismatch"), "got: {err}");
        assert!(err.contains("'bad'"), "names the contrast: {err}");
        assert!(err.contains("series") && err.contains("scalar"), "names both shapes: {err}");
    }

    #[test]
    fn dimension_mismatch_is_a_located_error() {
        // deaths (count, P^1 T^0) − rate (P^0 T^-1) → rejected.
        let deaths = ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(10.0)], times: vec![], dim: Some((1, 0)) };
        let rate = ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(0.1)], times: vec![], dim: Some((0, -1)) };
        let err = combine(&BinOp::Sub, deaths, rate, "deaths_minus_rate").unwrap_err();
        assert!(err.contains("dimension mismatch"), "got: {err}");
        assert!(err.contains("P^1 T^0") && err.contains("P^0 T^-1"), "names both dims: {err}");
    }

    #[test]
    fn scalar_band_columns_and_median() {
        // Per-draw averted values: median 30, mean 30, n_forkable 5.
        let draws: Vec<ShapedValue> = [10.0, 20.0, 30.0, 40.0, 50.0]
            .iter()
            .map(|&x| ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(x)], times: vec![], dim: Some((1, 0)) })
            .collect();
        let tsv = band_and_render("averted", &draws).unwrap();
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "q05\tq25\tq50\tq75\tq95\tmean\tn_forkable");
        let cells: Vec<&str> = lines[1].split('\t').collect();
        // q50 (median) = 30, mean = 30, n_forkable = 5.
        assert_eq!(cells[2], "30", "median");
        assert_eq!(cells[5], "30", "mean");
        assert_eq!(cells[6], "5", "n_forkable");
    }

    #[test]
    fn series_band_has_time_column_per_snapshot() {
        let draws: Vec<ShapedValue> = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            .iter()
            .map(|xs| ShapedValue { shape: Shape::Series, leaves: vec![series_leaf(xs)], times: vec![7.0, 14.0], dim: Some((1, 0)) })
            .collect();
        let tsv = band_and_render("curve", &draws).unwrap();
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "time\tq05\tq25\tq50\tq75\tq95\tmean\tn_forkable");
        // Two snapshot rows (t=7, t=14), median across {1,3,5}=3 and {2,4,6}=4.
        assert_eq!(lines.len(), 3, "header + 2 snapshot rows");
        let r0: Vec<&str> = lines[1].split('\t').collect();
        assert_eq!(r0[0], "7", "first snapshot time");
        assert_eq!(r0[3], "3", "median at t=7");
        let r1: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(r1[0], "14");
        assert_eq!(r1[3], "4", "median at t=14");
    }

    #[test]
    fn censored_scalar_is_excluded_from_the_band() {
        // Two finite, one censored → banded over the 2 finite, n_forkable = 2.
        let draws = vec![
            ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(4.0)], times: vec![], dim: None },
            ShapedValue { shape: Shape::Scalar, leaves: vec![LeafValue { key: vec![], dims: vec![], levels: vec![], payload: LeafPayload::Scalar(QuantityDrawValue::Censored) }], times: vec![], dim: None },
            ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(8.0)], times: vec![], dim: None },
        ];
        let tsv = band_and_render("c", &draws).unwrap();
        let cells: Vec<&str> = tsv.trim_end().lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(cells[5], "6", "mean of the 2 finite values");
        assert_eq!(cells[6], "2", "n_forkable counts only finite draws");
    }
}
