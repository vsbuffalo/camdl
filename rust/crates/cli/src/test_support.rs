//! Test-only support helpers shared across the crate's unit tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A collision-free unique temp-directory path for a unit test.
///
/// The previous `temp_dir().join(format!("..._{pid}_{nanos}"))` scheme raced
/// between parallel test threads that hit the same nanosecond, producing
/// intermittent filesystem failures under full-suite load (gh#153). A
/// process-wide monotonic counter guarantees cross-thread uniqueness; the pid
/// keeps it unique across concurrent test processes. The directory is NOT
/// created — callers `create_dir_all` (and clean up) as before.
pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("camdl_{}_{}_{}", prefix, std::process::id(), n))
}

/// A minimal, valid `ir::Model` for unit tests that need a model value but
/// not its dynamics: no compartments, no transitions, no parameters, a daily
/// regular output schedule over `[0, 100]`. Callers set the fields a test
/// reads (`parameters`, `presets`, …).
pub(crate) fn tiny_model() -> ir::Model {
    use ir::model::{
        InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule,
        SimulationConfig,
    };
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "sir".into(),
        version: "1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        initial_conditions: InitialConditions::default(),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: None,
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}
