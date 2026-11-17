//! The event-jump sensitivity through the whole pipeline (gh#275 §1g): a `.camdl`
//! source → `camdlc` (which emits each action's `∂amount/∂θ`) → `CompiledModel`
//! → `det_grad`, checked against a central finite difference of the VALUE path
//! (`compute_ode_loglik`, which runs the same effects through `run_ode`).
//!
//! The model is the shape of the report that motivated the feature: an
//! age-structured SEIR whose only effects are yearly school-entry transfers —
//! on a fixed day each year a fixed fraction of every young compartment moves
//! to its old counterpart, four transfers at one instant, each reading the
//! pre-event state — plus an annual importation whose size is estimated. The
//! transfer instants fall mid-step (`dt = 1`, `.5` days), so the RK4 step is
//! split at each.
//!
//! The in-module oracles (`sim::inference::ode_grad::tests`) hand-install the
//! amount gradients; this test is the one place the compiler-emitted ones are
//! consumed end to end. Skipped (with a message) when `camdlc.exe` is not
//! built.

use std::path::PathBuf;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::config::OdeConfig;
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::ode_grad::det_grad;
use sim::inference::{compute_ode_loglik, dense_cells, BoundObs, MultiStreamObsModel};

const MODEL: &str = r#"
time_unit = 'days

compartments { S, E, I, R }

dimensions {
  age = [young, old]
}

stratify(by = age)

let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

parameters {
  beta  : rate  in [0.05, 2.0]
  sigma : rate  in [0.05, 1.0]
  gamma : rate  in [0.05, 1.0]
  seed  : count in [0.0, 50.0]
}

tables {
  C_age : age × age = [[10.0, 3.0], [3.0, 6.0]]
}

transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
  progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
  recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
}

events {
  # School entry: on a fixed day each year a fifth of every young compartment
  # ages into the old one. Four transfers at one instant, each reading the
  # pre-event state. Mid-step instants at dt = 1.
  age_S : transfer(fraction = 0.2, from = S[young], to = S[old]) at [60.5, 120.5]
  age_E : transfer(fraction = 0.2, from = E[young], to = E[old]) at [60.5, 120.5]
  age_I : transfer(fraction = 0.2, from = I[young], to = I[old]) at [60.5, 120.5]
  age_R : transfer(fraction = 0.2, from = R[young], to = R[old]) at [60.5, 120.5]
  # Importation of an estimated number of infectious adults, mid-step.
  import : add(I[old], seed) at [30.5, 90.5]
}

observations {
  cases_young {
    columns   { time : time, cases_young : count }
    covers    = closing_at(time, 7 'days)
    projected = incidence(infection[young])
    cases_young ~ poisson(rate = projected)
  }
  cases_old {
    columns   { time : time, cases_old : count }
    covers    = closing_at(time, 14 'days)
    projected = incidence(infection[old])
    cases_old ~ poisson(rate = projected)
  }
}

init {
  S[young] = 3998
  S[old]   = 6000
  I[young] = 2
}

simulate { from = 0 'days  to = 154 'days }
"#;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn compile(src: &str) -> Option<ir::Model> {
    let camdlc = std::env::var("CAMDLC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo().join("ocaml/_build/default/bin/camdlc.exe"));
    if !camdlc.exists() {
        eprintln!("skip: {} not built (run `make build-ocaml`)", camdlc.display());
        return None;
    }
    // One file per call: the tests of this binary run concurrently.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let k = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("event_jump_sensitivity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("model_{k}.camdl"));
    std::fs::write(&path, src).unwrap();
    let out = std::process::Command::new(&camdlc)
        .env_remove("CLICOLOR_FORCE")
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    Some(ir::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap())
}

/// Incidence of `trs` closed at each of `times`, read off a value-path trajectory.
fn windowed_counts(traj: &sim::state::Trajectory, trs: &[usize], times: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(times.len());
    let mut acc = 0.0;
    let mut next = 0usize;
    for (i, snap) in traj.snapshots.iter().enumerate() {
        if i > 0 {
            acc += trs.iter().map(|&tr| snap.flows.as_real()[tr]).sum::<f64>();
        }
        if next < times.len() && (snap.t - times[next]).abs() < 1e-9 {
            out.push(acc.round());
            acc = 0.0;
            next += 1;
        }
    }
    assert_eq!(out.len(), times.len(), "every obs time must land on a snapshot");
    out
}

#[test]
fn compiled_amount_gradients_carry_the_jump_through_an_age_structured_seir() {
    let Some(mut model) = compile(MODEL) else { return };
    let truth = [("beta", 0.02), ("sigma", 0.2), ("gamma", 0.1), ("seed", 8.0)];
    for p in &mut model.parameters {
        let v = truth.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v).unwrap();
        p.value = p.value.with_value(v);
    }
    // The compiler emitted the jump source term for the estimated amount, and
    // nothing for the constant fractions.
    let import = model.interventions.iter().find(|iv| iv.name == "import").unwrap();
    match &import.actions[0] {
        ir::intervention::Action::Add(a) => assert!(
            matches!(a.count_grad.get("seed"), Some(ir::deriv::DerivEntry::Grad(_))),
            "camdlc must emit ∂count/∂seed on `add(I[old], seed)`: {:?}", a.count_grad
        ),
        other => panic!("expected an add action, got {other:?}"),
    }
    let ageing = model.interventions.iter().find(|iv| iv.name == "age_S").unwrap();
    match &ageing.actions[0] {
        ir::intervention::Action::FractionTransfer(ft) => {
            assert!(ft.fraction_grad.is_empty(), "a constant fraction carries no gradient")
        }
        other => panic!("expected a fraction transfer, got {other:?}"),
    }

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let n = compiled.param_index.len();
    let mut params_true = vec![0.0; n];
    for p in &compiled.model.parameters {
        params_true[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
    }

    // Data from the value path at the truth, so they carry every effect.
    let dt = 1.0;
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let cfg = OdeConfig { t_start, t_end, dt };
    let traj = sim::ode::run_ode(&compiled, &params_true, &cfg, None, None).unwrap();
    let times_young: Vec<f64> = (1..=22).map(|w| (w * 7) as f64).collect();
    let times_old: Vec<f64> = (1..=11).map(|w| (w * 14) as f64).collect();
    let tr = |name: &str| compiled.model.transitions.iter().position(|t| t.name == name).unwrap();
    let data_young = windowed_counts(&traj, &[tr("infection_young")], &times_young);
    let data_old = windowed_counts(&traj, &[tr("infection_old")], &times_old);
    assert!(data_young.iter().sum::<f64>() > 50.0, "young incidence must be substantial");
    assert!(data_old.iter().sum::<f64>() > 50.0, "old incidence must be substantial");

    let specs: Vec<StreamSpec> = compiled
        .model
        .observations
        .iter()
        .map(|om| {
            let (data, times) = match om.name.as_str() {
                "cases_young" => (&data_young, &times_young),
                "cases_old" => (&data_old, &times_old),
                other => panic!("unexpected stream {other}"),
            };
            let projection = StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap();
            StreamSpec {
                times: StreamTimes::contiguous_for(&projection, t_start, times.clone()).unwrap(),
                projection,
                ir_model: om.clone(),
                observations: dense_cells(data.clone()),
                aux: vec![],
            }
        })
        .collect();
    let obs_model =
        MultiStreamObsModel::new(BoundObs::bind(t_start, specs).unwrap().0, compiled.clone()).unwrap();
    let mut union: Vec<f64> = times_young.iter().chain(&times_old).copied().collect();
    union.sort_by(|a, b| a.partial_cmp(b).unwrap());
    union.dedup();

    // Score away from the truth so every gradient is materially nonzero.
    let names = ["beta", "sigma", "gamma", "seed"];
    let mut params = params_true.clone();
    params[compiled.param_index["beta"]] = 0.025;
    params[compiled.param_index["gamma"]] = 0.085;
    params[compiled.param_index["seed"]] = 10.0;
    let est: Vec<usize> = names.iter().map(|n| compiled.param_index[*n]).collect();

    let (ll_grad, grad) = det_grad(&compiled, &obs_model, &union, dt, dt, &params, &est).unwrap();
    let ll_value = compute_ode_loglik(&compiled, &obs_model, &union, dt, &params, dt).unwrap();
    // To 1e-9 relative, not bit for bit: the 14-day stream sums its bins in a
    // different association on the two paths (gh#680). The in-crate oracle
    // (`ode_grad::tests`) pins the recorded per-interval incidence bit for bit.
    assert!(
        (ll_grad - ll_value).abs() <= 1e-9 * ll_value.abs(),
        "gradient-path loglik {ll_grad:.17e} must be the value path's {ll_value:.17e} (Δ = {:.3e})",
        ll_grad - ll_value
    );

    eprintln!("age-structured SEIR with school-entry transfers + estimated import: ll = {ll_grad:.6}");
    eprintln!("  {:<6} {:>18} {:>18} {:>10}", "param", "analytic", "value-path FD", "rel err");
    for (i, &midx) in est.iter().enumerate() {
        let eps = 1e-6 * params[midx].abs().max(1.0);
        let mut pp = params.clone();
        let mut pm = params.clone();
        pp[midx] += eps;
        pm[midx] -= eps;
        let llp = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pp, dt).unwrap();
        let llm = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pm, dt).unwrap();
        let fd = (llp - llm) / (2.0 * eps);
        let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
        eprintln!("  {:<6} {:>18.8} {:>18.8} {:>10.2e}", names[i], grad[i], fd, rel);
        assert!(rel < 1e-4, "∂/∂{} = {} vs value-path FD {} (rel err {rel:.2e})", names[i], grad[i], fd);
        assert!(grad[i].abs() > 1e-2, "∂/∂{} should be materially nonzero, got {}", names[i], grad[i]);
    }
}

/// The same model with the importation written as a fixed-count transfer is
/// refused by the gate before any gradient is taken, naming the event.
#[test]
fn a_fixed_count_transfer_is_refused_by_name_before_any_gradient() {
    let src = MODEL.replace(
        "import : add(I[old], seed) at [30.5, 90.5]",
        "import : transfer(count = seed, from = S[old], to = I[old]) at [30.5, 90.5]",
    );
    let Some(mut model) = compile(&src) else { return };
    for p in &mut model.parameters {
        let v = match p.name.as_str() { "beta" => 0.02, "sigma" => 0.2, "gamma" => 0.1, _ => 8.0 };
        p.value = p.value.with_value(v);
    }
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    let estimated: std::collections::HashSet<&str> = ["beta", "seed"].into_iter().collect();
    let err = sim::inference::gradient_capability::preflight_gradient_ode(&compiled, &params, &estimated)
        .expect_err("a fixed-count transfer has no sensitivity rule");
    let msg = err.to_string();
    assert!(msg.contains("import") && msg.contains("transfer(count") && msg.contains("nl-sbplx"), "{msg}");
}
