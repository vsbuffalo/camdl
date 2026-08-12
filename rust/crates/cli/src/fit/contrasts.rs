//! Stage C of counterfactual contrasts (proposal 2026-06-25): the two-arm replay
//! reducer that CONSUMES the IR [`ir::contrast::Contrast`] node. Auto-emitted by
//! `fit predict` when the model declares any `contrasts {}`.
//!
//! **The fork is derived, not declared.** A contrast has no `over [..]` window.
//! From the runs a contrast references, the reducer diffs the arms' live
//! intervention sets to find the *toggled* intervention, reads its (constant)
//! fire time, and forks at the **last saved trajectory snapshot strictly before
//! that fire time** (`fork`). Both arms then run `[fork, run_end]`, where
//! `run_end` is the model's simulation horizon. This makes "fork at/after the
//! intervention" unrepresentable.
//!
//! Per forkable posterior draw `i`, per `run` referenced in a contrast (a scenario
//! or the reserved `fitted` no-overlay run):
//!   1. resolve that arm's θ via the 5-tier resolver
//!      ([`crate::params_resolver::resolve_parameters`]) — the fitted draw at tier
//!      3.5, the scenario `set`/`scale` at tier 4 (`fitted` ⇒ no overlay);
//!   2. fork from the smoothed `X_i(fork)` — chain_binomial reads the saved path
//!      with [`io::trajectories::read_state_at`] and resumes via
//!      [`sim::chain_binomial::Resume`]`{ start: Some(..) }`. CRN: both arms share
//!      `X_i(fork)` AND the per-draw seed, so the firing substep is byte-identical
//!      at the fork; post-fork noise desyncs by design;
//!   3. run `[fork, run_end]`, evaluate the operand quantities with the shared
//!      [`sim::quantity::QuantityEvaluator`];
//!   4. difference elementwise, shape- and dimension-preserving;
//!   5. band over the forkable subset → `contrasts/<name>.tsv`.
//!
//! "Averted by week N" is read off the time-indexed output (or by setting the run
//! horizon); a decoupled `by <instant>` clause is a later refinement.
//!
//! Edge cases are loud-deferred (skip-with-note, never silent): a contrast with no
//! toggled intervention (parameter-only scenario → gh#327), or one toggled by a
//! parametric / reactive fire time (no single derivable fork).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use indexmap::{IndexMap, IndexSet};

use ir::contrast::{Contrast, ContrastExpr, RunNamespace};
use ir::expr::{BinOp, Expr, UnOp};
use ir::intervention::{Intervention, InterventionSchedule};
use io::trajectories::SNAPSHOT_TIME_TOL;
use sim::quantity::{QuantityDrawValue, QuantityEvaluator, QuantityResult};

use crate::args::types::ForwardBackend;
use crate::fit::joint::{resolve_joint, LatentPath};
use crate::fit::predict::write_tsv;
use crate::quantile::{band, fmt_time, fmt_value, QUANTILE_LEVELS};
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
             deterministic forking is a named gh#325 follow-up.)",
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

    // #273-class guard: every arm forks on the FULL parameter vector, so each
    // draw must carry every model parameter. The canonical `draws.tsv` writes all
    // of them (estimated columns first, then the fixed values); a draws file
    // missing columns would silently fall back to model defaults — diverging from
    // the fit. Assert coverage loudly rather than fork an incomplete vector. (The
    // join reads only the canonical file today; the guard pins that invariant.)
    let missing: Vec<&str> = model
        .parameters
        .iter()
        .map(|p| p.name.as_str())
        .filter(|name| !forkable[0].0.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "contrasts: the posterior draws are missing {} model parameter(s): {}. A \
             contrast forks each arm on the full parameter vector; a draws.tsv lacking \
             columns would silently fall back to model defaults, diverging from the fit.",
            missing.len(),
            missing.join(", "),
        ));
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

    // Both arms simulate to the model's run-end horizon (the same horizon the
    // predict free-forward path uses). With the fork derived (below), the arm's
    // trajectory already spans exactly [fork, run_end] — no clipping.
    let run_end = model.simulation.t_end;

    // The saved snapshot grid the fork must land on. The output cadence is
    // model-wide, so the first forkable draw's path is representative; a draw that
    // happens to lack the derived fork errors loudly at `read_state_at`.
    let (_, snap_chain, snap_draw) = &forkable[0];
    let snap_traj = stage_dir
        .join(format!("chain_{}", snap_chain + 1))
        .join("trajectories.tsv");
    let snapshot_times = io::trajectories::snapshot_times(&snap_traj, *snap_chain, *snap_draw)?;

    // Per contrast: derive the fork (last saved snapshot before the toggled
    // intervention fires), replay its runs over [fork, run_end] for every forkable
    // draw, walk the body into a per-draw ShapedValue, band over the forkable
    // subset, write.
    let mut written = Vec::new();
    for c in &to_emit {
        let mut runs_c: Vec<String> = Vec::new();
        collect_runs(&c.body, &mut runs_c);

        // Derive the fork from the contrast's toggled intervention. The edge cases
        // (no toggled intervention; a parametric/reactive fire time) are loud
        // skip-with-note deferrals, never silent mis-forks.
        let plan = match derive_fork(model, &arms, &runs_c, &snapshot_times) {
            Ok(p) => p,
            Err(reason) => {
                eprintln!("fit predict: skipping contrast '{}' — {reason}", c.name);
                continue;
            }
        };
        eprintln!(
            "fit predict: contrast '{}' — fork at t={} (last saved snapshot before '{}' \
             fires at t={}); arms run [{}, {}].{}",
            c.name,
            plan.fork_t,
            plan.fire_iv,
            plan.fire_t,
            plan.fork_t,
            run_end,
            if plan.toggled.len() > 1 {
                format!(
                    " ({} interventions toggled [{}]; forking before the earliest)",
                    plan.toggled.len(),
                    plan.toggled.join(", "),
                )
            } else {
                String::new()
            },
        );

        let mut shaped: Vec<ShapedValue> = Vec::with_capacity(forkable.len());
        for (draw_pos, (params_i, chain, draw)) in forkable.iter().enumerate() {
            // CRN: both arms of THIS draw share one seed (run name is NOT mixed in),
            // so the firing substep is byte-identical at the fork.
            let arm_seed = crate::util::derive_chain_seed(seed, draw_pos);
            let mut draw_results: HashMap<String, ArmDrawResult> = HashMap::new();
            for run in &runs_c {
                let res = arms[run].replay(
                    model, run, params_i, &col_spec, &stage_dir, *chain, *draw, dt, arm_seed,
                    plan.fork_t, run_end,
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

    /// Replay this arm for one draw from the derived fork: resolve the draw's θ
    /// (+ scenario overlay), fork from the smoothed `X_i(fork)`, run
    /// `[fork, run_end]`, and evaluate every quantity over that span. A series
    /// operand's time axis is exactly the arm's snapshots over `[fork, run_end]`;
    /// a reduced operand reduces over that same span (no clipping — the fork runs
    /// exactly the right window).
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
        fork_t: f64,
        run_end: f64,
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

        // Read the smoothed latent state X_i(fork) from the saved path. The on-disk
        // chain dir is 1-based (`chain_{N+1}`); the in-file `chain` column is the
        // 0-based key `read_state_at` matches.
        let traj_path = stage_dir.join(format!("chain_{}", chain + 1)).join("trajectories.tsv");
        let (int_s, real_s) =
            io::trajectories::read_state_at(&traj_path, col_spec, chain, draw, fork_t)
                .map_err(|e| format!("contrast arm '{run}': reading X(fork={fork_t}): {e}"))?;

        // Fork from X_i(fork): inject the state at cfg.t_start = fork, fresh RNG from
        // the shared per-draw seed (CRN at the fork; post-fork noise desyncs by design).
        let cfg = sim::ChainBinomialConfig { t_start: fork_t, t_end: run_end, dt };
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
        .map_err(|e| format!("contrast arm '{run}': forking at fork={fork_t}: {e:?}"))?;

        let times: Vec<f64> = traj.snapshots.iter().map(|s| s.t).collect();
        let quant = self.quant_eval.eval_draw(&pvec, &traj, self.compiled.as_ref(), None);
        Ok(ArmDrawResult { quant, times })
    }
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

// ── Derived fork ────────────────────────────────────────────────────────────────

/// The fork derived for one contrast: where to branch the two arms from, and what
/// drove it. The fork is the last saved trajectory snapshot strictly before the
/// toggled intervention's fire time — so "fork at/after the intervention" is
/// unrepresentable by construction.
struct ForkPlan {
    /// The fork time: the latest saved snapshot strictly before `fire_t`.
    fork_t: f64,
    /// The earliest fire time across the toggled interventions — the first instant
    /// the arms diverge.
    fire_t: f64,
    /// The name of the intervention firing at `fire_t` (for the transparency note).
    fire_iv: String,
    /// Every toggled intervention (≥1), sorted; >1 means several differ across the
    /// arms and the fork is taken before the earliest.
    toggled: Vec<String>,
}

/// A toggled intervention's earliest scheduled fire time, or why it can't yield a
/// single constant fork.
enum FireTime {
    /// A constant earliest fire time.
    Const(f64),
    /// A parametric `at [...]` fire time (the fork would differ per draw).
    Parametric,
    /// A reactive fire source — no static schedule to derive a fork from.
    Reactive,
    /// A schedule that produces no fire times in the run window.
    Empty,
}

/// Derive the fork for a contrast. The runs it references are the arms; their live
/// intervention sets differ by the *toggled* intervention(s) (the scenario
/// enable/disable filter already removed the inactive ones, so a live set is
/// exactly what fires in that arm). The fork is the last saved snapshot strictly
/// before the earliest toggled fire time.
///
/// Returns `Err(reason)` for the loud-deferral edge cases: no toggled intervention
/// (a parameter-only counterfactual → gh#327), a parametric / reactive fire time,
/// or a fire time at/before the first saved snapshot (nothing to fork from).
fn derive_fork(
    model: &ir::Model,
    arms: &HashMap<String, Arm>,
    runs: &[String],
    snapshot_times: &[f64],
) -> Result<ForkPlan, String> {
    // The live intervention name-set of each arm (what actually fires in it).
    let live: Vec<HashSet<String>> = runs
        .iter()
        .map(|r| {
            arms[r]
                .compiled
                .model
                .interventions
                .iter()
                .map(|iv| iv.name.clone())
                .collect()
        })
        .collect();
    let toggled = toggled_across_arms(&live);
    if toggled.is_empty() {
        return Err(
            "this contrast has no toggled intervention to derive a fork from; \
             parameter-only counterfactuals need the time-scheduled-parameter \
             primitive (gh#327)"
                .to_string(),
        );
    }

    // The earliest constant fire time across the toggled interventions; a
    // parametric / reactive one is a loud deferral.
    let mut earliest: Option<(f64, String)> = None;
    for name in &toggled {
        let iv = model
            .interventions
            .iter()
            .find(|iv| &iv.name == name)
            .ok_or_else(|| format!("internal: toggled intervention '{name}' not found in model"))?;
        match intervention_earliest_fire(iv) {
            FireTime::Const(t) => {
                if earliest.as_ref().map_or(true, |(e, _)| t < *e) {
                    earliest = Some((t, name.clone()));
                }
            }
            FireTime::Parametric => {
                return Err(format!(
                    "the toggled intervention '{name}' fires at a parameter-dependent time; \
                     contrasts don't yet support parametric intervention times (the fork would \
                     differ per draw) — gh follow-up"
                ))
            }
            FireTime::Reactive => {
                return Err(format!(
                    "the toggled intervention '{name}' is reactive (no static fire time); a \
                     contrast derives its fork from a scheduled fire time"
                ))
            }
            FireTime::Empty => {
                return Err(format!(
                    "the toggled intervention '{name}' has no fire times in the run window"
                ))
            }
        }
    }
    let (fire_t, fire_iv) = earliest.expect("a non-empty toggled set yields at least one fire time");

    // The fork = the largest saved snapshot strictly before the first divergence.
    match fork_before(snapshot_times, fire_t) {
        Some(fork_t) => Ok(ForkPlan { fork_t, fire_t, fire_iv, toggled }),
        None => Err(format!(
            "the toggled intervention fires at t={fire_t}, at or before the first saved \
             snapshot — there is no pre-intervention state to fork from"
        )),
    }
}

/// The largest saved snapshot time strictly before `fire_t` (with a snapshot-time
/// tolerance), or `None` if every snapshot is at/after the fire — the choice that
/// makes "fork at/after the intervention" unrepresentable.
fn fork_before(snapshot_times: &[f64], fire_t: f64) -> Option<f64> {
    snapshot_times
        .iter()
        .copied()
        .filter(|&t| t < fire_t - SNAPSHOT_TIME_TOL)
        .fold(None, |m, t| Some(m.map_or(t, |y: f64| y.max(t))))
}

/// The interventions toggled across the arms: live in some arm but not all (the
/// N-arm generalization of the symmetric difference). Sorted for a deterministic
/// note.
fn toggled_across_arms(live: &[HashSet<String>]) -> Vec<String> {
    let mut union: BTreeSet<String> = BTreeSet::new();
    for s in live {
        union.extend(s.iter().cloned());
    }
    union
        .into_iter()
        .filter(|name| !live.iter().all(|s| s.contains(name)))
        .collect()
}

/// The earliest scheduled fire time of one intervention. Mirrors the runtime fire
/// expansion (`sim::intervention::intervention_fire_times`) for the constant
/// schedules; a parametric `at [...]` (any non-constant expr) or a reactive source
/// is reported so the caller can defer loudly.
fn intervention_earliest_fire(iv: &Intervention) -> FireTime {
    match iv.fire.schedule() {
        None => FireTime::Reactive,
        Some(InterventionSchedule::AtTimes(ts)) => ts
            .iter()
            .copied()
            .fold(None, |m, t| Some(m.map_or(t, |y: f64| y.min(t))))
            .map_or(FireTime::Empty, FireTime::Const),
        Some(InterventionSchedule::AtTimesExpr(exprs)) => {
            let mut min: Option<f64> = None;
            for e in exprs {
                match const_eval_expr(e) {
                    Some(v) => min = Some(min.map_or(v, |m: f64| m.min(v))),
                    None => return FireTime::Parametric,
                }
            }
            min.map_or(FireTime::Empty, FireTime::Const)
        }
        Some(InterventionSchedule::Recurring(rs)) => {
            if rs.period <= 0.0 {
                return FireTime::Empty;
            }
            // Earliest fire ≥ start (mirrors intervention_fire_times): with at_day,
            // `at_day + k0*period` for the smallest k0 ≥ 0 reaching `start`.
            let first = match rs.at_day {
                None => rs.start,
                Some(d) => {
                    let k0 = ((rs.start - d) / rs.period).ceil().max(0.0);
                    d + k0 * rs.period
                }
            };
            if first > rs.end + rs.period * 1e-9 {
                FireTime::Empty
            } else {
                FireTime::Const(first)
            }
        }
    }
}

/// Const-fold an `Expr` to a model-time float, or `None` if it references any
/// non-constant leaf (a parameter, population, time, …). Mirrors the OCaml
/// instant-evaluation: `Const`, `+ - * /`, unary `Neg`, and the dimensional escape
/// `UncheckedDim` (a unit literal like `20 'weeks` lowers to
/// `0 + UncheckedDim{140}`). Anything else is non-constant.
fn const_eval_expr(e: &Expr) -> Option<f64> {
    match e {
        Expr::Const(c) => Some(c.value),
        Expr::UncheckedDim(w) => const_eval_expr(&w.unchecked_dim.inner),
        Expr::UnOp(w) => {
            let v = const_eval_expr(&w.un_op.arg)?;
            match w.un_op.op {
                UnOp::Neg => Some(-v),
                _ => None,
            }
        }
        Expr::BinOp(w) => {
            let l = const_eval_expr(&w.bin_op.left)?;
            let r = const_eval_expr(&w.bin_op.right)?;
            match w.bin_op.op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div => Some(l / r),
                _ => None,
            }
        }
        _ => None,
    }
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
                 this build (gh#326); reduce the stream inside a `quantities {{}}` entry and \
                 contrast the named quantity instead"
            )),
            ContrastExpr::RunMember { ns: RunNamespace::Quantities, member, .. } => {
                if quantity_reduces_observations(model, member) {
                    Err(format!(
                        "quantity '{member}' reduces an `observations.<stream>` source; \
                         observation-sourced contrasts are deferred in this build (gh#326 — the \
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
        // Equal length is necessary but not sufficient: the snapshot VALUES must
        // align too, else an elementwise fold silently differences mismatched
        // times. Compare within the shared snapshot tolerance (a located error).
        for (i, (lt, rt)) in l.times.iter().zip(&r.times).enumerate() {
            if (lt - rt).abs() > SNAPSHOT_TIME_TOL {
                return Err(format!(
                    "contrast '{cname}': series operands disagree on snapshot time {i} \
                     ({lt} vs {rt}) — the arms must share the same output times, not just \
                     the same count",
                ));
            }
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
/// `[time] <dims…> q05 q25 q50 q75 q95 mean n_used`, keyed by `(stratum, time)`
/// as the shape carries. `n_used` is the per-cell count of finite/uncensored
/// draws the band was computed over (NOT the fit-level forkable count).
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
    header.push("n_used".to_string());

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
/// `q05 q25 q50 q75 q95 mean n_used`. An empty column (every draw censored)
/// renders empty quantile + mean cells and `n_used = 0` — never a fabricated
/// band. A non-finite value is rejected by [`band`] (an upstream bug).
fn band_cells(col: &[f64]) -> Result<Vec<String>, String> {
    let mut cells: Vec<String> = Vec::with_capacity(QUANTILE_LEVELS.len() + 2);
    if col.is_empty() {
        for _ in QUANTILE_LEVELS {
            cells.push(String::new());
        }
        cells.push(String::new()); // mean
        cells.push("0".to_string()); // n_used
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
    fn stratum_count_mismatch_is_a_located_error() {
        // A stratified operand (2 strata leaves) minus an unstratified one (1
        // leaf): same shape + dimension, but the strata axes don't match → a
        // located error naming the contrast and the differing cell counts.
        fn keyed_leaf(dim: &str, level: &str, x: f64) -> LeafValue {
            LeafValue {
                key: vec![(dim.to_string(), level.to_string())],
                dims: vec![dim.to_string()],
                levels: vec![level.to_string()],
                payload: LeafPayload::Scalar(QuantityDrawValue::Value(x)),
            }
        }
        let strat = ShapedValue {
            shape: Shape::Scalar,
            leaves: vec![keyed_leaf("patch", "a", 10.0), keyed_leaf("patch", "b", 4.0)],
            times: vec![],
            dim: Some((1, 0)),
        };
        let unstrat = ShapedValue {
            shape: Shape::Scalar,
            leaves: vec![scalar_leaf(3.0)],
            times: vec![],
            dim: Some((1, 0)),
        };
        let err = combine(&BinOp::Sub, strat, unstrat, "by_patch").unwrap_err();
        assert!(err.contains("stratification mismatch"), "got: {err}");
        assert!(err.contains("'by_patch'"), "names the contrast: {err}");
        assert!(err.contains("2 vs 1"), "names the cell counts: {err}");
    }

    #[test]
    fn series_time_axis_value_mismatch_is_a_located_error() {
        // Equal length, different snapshot times: the elementwise fold would
        // silently difference mismatched times → a located error.
        let l = ShapedValue {
            shape: Shape::Series,
            leaves: vec![series_leaf(&[5.0, 6.0])],
            times: vec![7.0, 14.0],
            dim: Some((1, 0)),
        };
        let r = ShapedValue {
            shape: Shape::Series,
            leaves: vec![series_leaf(&[1.0, 2.0])],
            times: vec![7.0, 21.0],
            dim: Some((1, 0)),
        };
        let err = combine(&BinOp::Sub, l, r, "curve").unwrap_err();
        assert!(err.contains("snapshot time 1"), "names the offending index: {err}");
        assert!(err.contains("'curve'"), "names the contrast: {err}");
    }

    #[test]
    fn scalar_band_columns_and_median() {
        // Per-draw averted values: median 30, mean 30, n_used 5.
        let draws: Vec<ShapedValue> = [10.0, 20.0, 30.0, 40.0, 50.0]
            .iter()
            .map(|&x| ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(x)], times: vec![], dim: Some((1, 0)) })
            .collect();
        let tsv = band_and_render("averted", &draws).unwrap();
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "q05\tq25\tq50\tq75\tq95\tmean\tn_used");
        let cells: Vec<&str> = lines[1].split('\t').collect();
        // q50 (median) = 30, mean = 30, n_used = 5.
        assert_eq!(cells[2], "30", "median");
        assert_eq!(cells[5], "30", "mean");
        assert_eq!(cells[6], "5", "n_used");
    }

    #[test]
    fn series_band_has_time_column_per_snapshot() {
        let draws: Vec<ShapedValue> = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            .iter()
            .map(|xs| ShapedValue { shape: Shape::Series, leaves: vec![series_leaf(xs)], times: vec![7.0, 14.0], dim: Some((1, 0)) })
            .collect();
        let tsv = band_and_render("curve", &draws).unwrap();
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "time\tq05\tq25\tq50\tq75\tq95\tmean\tn_used");
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
        // Two finite, one censored → banded over the 2 finite, n_used = 2.
        let draws = vec![
            ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(4.0)], times: vec![], dim: None },
            ShapedValue { shape: Shape::Scalar, leaves: vec![LeafValue { key: vec![], dims: vec![], levels: vec![], payload: LeafPayload::Scalar(QuantityDrawValue::Censored) }], times: vec![], dim: None },
            ShapedValue { shape: Shape::Scalar, leaves: vec![scalar_leaf(8.0)], times: vec![], dim: None },
        ];
        let tsv = band_and_render("c", &draws).unwrap();
        let cells: Vec<&str> = tsv.trim_end().lines().nth(1).unwrap().split('\t').collect();
        assert_eq!(cells[5], "6", "mean of the 2 finite values");
        assert_eq!(cells[6], "2", "n_used counts only finite draws");
    }

    // ── Derived fork ─────────────────────────────────────────────────────────

    use ir::expr::{BinOpExpr, BinOpWrap, ConstExpr, ParamExpr, UnOpExpr, UnOpWrap,
                   UncheckedDimExpr, UncheckedDimWrap};
    use ir::intervention::{Action, FireSource, InterventionSchedule, RecurringSchedule};

    fn set(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn scheduled(name: &str, sched: InterventionSchedule) -> Intervention {
        Intervention {
            name: name.into(),
            base_name: None,
            fire: FireSource::Scheduled(sched),
            actions: Vec::<Action>::new(),
            kind: ir::intervention::InterventionKind::Scenario,
        }
    }

    #[test]
    fn toggled_set_is_the_symmetric_difference_across_arms() {
        // no_sia (sia off) vs with_sia (sia on) → {sia}.
        assert_eq!(toggled_across_arms(&[set(&[]), set(&["sia"])]), vec!["sia"]);
        // Identical live sets (a parameter-only counterfactual) → nothing toggled.
        assert!(toggled_across_arms(&[set(&["sia"]), set(&["sia"])]).is_empty());
        // Three arms; `b` differs in only one → {b}, sorted; shared `a` excluded.
        assert_eq!(
            toggled_across_arms(&[set(&["a"]), set(&["a", "b"]), set(&["a"])]),
            vec!["b"]
        );
    }

    #[test]
    fn fork_is_the_last_snapshot_strictly_before_the_fire() {
        let grid = [0.0, 7.0, 14.0, 21.0, 28.0];
        // Fire at 28 → fork at 21 (28 itself is excluded — fork must be BEFORE).
        assert_eq!(fork_before(&grid, 28.0), Some(21.0));
        // Fire at 22 → fork at 21.
        assert_eq!(fork_before(&grid, 22.0), Some(21.0));
        // Fire at/before the first snapshot → nothing to fork from.
        assert_eq!(fork_before(&grid, 0.0), None);
        assert_eq!(fork_before(&grid, -1.0), None);
    }

    #[test]
    fn const_eval_folds_unit_literal_instants_and_rejects_parameters() {
        // `20 'weeks` lowers to `0 + UncheckedDim{140}`.
        let unit_140 = Expr::BinOp(BinOpWrap {
            bin_op: BinOpExpr {
                op: BinOp::Add,
                left: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
                right: Box::new(Expr::UncheckedDim(UncheckedDimWrap {
                    unchecked_dim: UncheckedDimExpr {
                        inner: Box::new(Expr::Const(ConstExpr { value: 140.0 })),
                        dim: (0, 1),
                        reason: "unit literal 'weeks".into(),
                    },
                })),
            },
        });
        assert_eq!(const_eval_expr(&unit_140), Some(140.0));
        // A parameter-dependent fire time is NOT a compile-time constant.
        let parametric = Expr::BinOp(BinOpWrap {
            bin_op: BinOpExpr {
                op: BinOp::Add,
                left: Box::new(Expr::Const(ConstExpr { value: 14.0 })),
                right: Box::new(Expr::Param(ParamExpr { param: "lag".into() })),
            },
        });
        assert_eq!(const_eval_expr(&parametric), None);
        // Unary negation folds; an unsupported unary op does not.
        let neg = Expr::UnOp(UnOpWrap {
            un_op: UnOpExpr { op: UnOp::Neg, arg: Box::new(Expr::Const(ConstExpr { value: 5.0 })) },
        });
        assert_eq!(const_eval_expr(&neg), Some(-5.0));
    }

    #[test]
    fn earliest_fire_classifies_each_schedule() {
        // AtTimes → the minimum of the list.
        match intervention_earliest_fire(&scheduled("a", InterventionSchedule::AtTimes(vec![140.0, 28.0]))) {
            FireTime::Const(t) => assert_eq!(t, 28.0),
            _ => panic!("AtTimes must be a constant fire time"),
        }
        // AtTimesExpr with a constant unit literal → folds.
        let const_expr = Expr::UncheckedDim(UncheckedDimWrap {
            unchecked_dim: UncheckedDimExpr {
                inner: Box::new(Expr::Const(ConstExpr { value: 28.0 })),
                dim: (0, 1),
                reason: "unit literal 'days".into(),
            },
        });
        match intervention_earliest_fire(&scheduled("a", InterventionSchedule::AtTimesExpr(vec![const_expr]))) {
            FireTime::Const(t) => assert_eq!(t, 28.0),
            _ => panic!("a constant AtTimesExpr must fold to a constant fire time"),
        }
        // AtTimesExpr referencing a parameter → Parametric (loud deferral).
        let param_expr = Expr::Param(ParamExpr { param: "t0".into() });
        assert!(matches!(
            intervention_earliest_fire(&scheduled("a", InterventionSchedule::AtTimesExpr(vec![param_expr]))),
            FireTime::Parametric
        ));
        // Recurring → earliest fire (start, with no at_day).
        let rec = RecurringSchedule { start: 10.0, period: 7.0, end: 80.0, at_day: None };
        match intervention_earliest_fire(&scheduled("a", InterventionSchedule::Recurring(rec))) {
            FireTime::Const(t) => assert_eq!(t, 10.0),
            _ => panic!("Recurring must yield its earliest fire"),
        }
    }
}
