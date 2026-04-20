use wasm_bindgen::prelude::*;
use serde::Deserialize;
use sim::{CompiledModel, GillespieSim, TauLeapSim, ChainBinomialSim, Simulate};
use sim::config::{SimConfig, GillespieConfig, TauLeapConfig, ChainBinomialConfig};
use ir::model::{CompartmentKind, OutputSchedule};

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Validate IR JSON.
/// Returns `{"ok":true}` or `{"ok":false,"error":"..."}`.
#[wasm_bindgen]
pub fn validate(ir_json: &str) -> String {
    match ir::from_str(ir_json) {
        Err(e) => serde_json::json!({"ok": false, "error": e.to_string()}).to_string(),
        Ok(model) => match ir::validate::validate(&model) {
            Err(errs) => {
                let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
                serde_json::json!({"ok": false, "error": msgs.join("; ")}).to_string()
            }
            Ok(()) => serde_json::json!({"ok": true}).to_string(),
        },
    }
}

#[derive(Deserialize)]
struct WasmSimConfig {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default)]
    seed: u64,
    dt: Option<f64>,
    output_dt: Option<f64>,
}

fn default_backend() -> String {
    "chain_binomial".to_string()
}

/// Simulate a model from IR JSON.
///
/// `config_json`: `{"backend":"gillespie"|"tau_leap"|"chain_binomial","seed":42,"dt":1.0}`
///
/// Returns trajectory JSON or `{"error":"..."}`.
#[wasm_bindgen]
pub fn simulate(ir_json: &str, config_json: &str) -> String {
    match simulate_inner(ir_json, config_json) {
        Ok(result) => result,
        Err(e) => serde_json::json!({"error": e}).to_string(),
    }
}

fn simulate_inner(ir_json: &str, config_json: &str) -> Result<String, String> {
    let model = ir::from_str(ir_json).map_err(|e| e.to_string())?;

    let wasm_cfg: WasmSimConfig = if config_json.is_empty() {
        WasmSimConfig { backend: "chain_binomial".to_string(), seed: 42, dt: None, output_dt: None }
    } else {
        serde_json::from_str(config_json).map_err(|e| e.to_string())?
    };

    let t_start = model.simulation.t_start;
    let t_end   = model.simulation.t_end;
    let seed    = if wasm_cfg.seed == 0 { 42 } else { wasm_cfg.seed };

    // Derive a sensible output_dt for Gillespie — cap trajectory at ~300 points.
    let output_dt = wasm_cfg.output_dt.or_else(|| match &model.output.times {
        OutputSchedule::Regular(r) => Some(r.step),
        _ => Some((t_end - t_start) / 300.0),
    });

    let sim_config = match wasm_cfg.backend.as_str() {
        "tau_leap" => {
            let dt = wasm_cfg.dt.or(model.simulation.dt).unwrap_or(1.0);
            SimConfig::TauLeap(TauLeapConfig { t_start, t_end, dt })
        }
        "chain_binomial" => {
            let dt = wasm_cfg.dt.or(model.simulation.dt).unwrap_or(1.0);
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt })
        }
        _ => SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt }),
    };

    // Collect metadata before consuming model.
    let int_comp_names: Vec<String> = model.compartments.iter()
        .filter(|c| matches!(c.kind, CompartmentKind::Integer))
        .map(|c| c.name.clone())
        .collect();
    let real_comp_names: Vec<String> = model.compartments.iter()
        .filter(|c| matches!(c.kind, CompartmentKind::Real))
        .map(|c| c.name.clone())
        .collect();
    let transition_names: Vec<String> = model.transitions.iter()
        .map(|t| t.name.clone())
        .collect();

    let compiled = CompiledModel::new(model).map_err(|e| e.to_string())?;
    let params   = compiled.default_params.clone();

    let trajectory = match &sim_config {
        SimConfig::Gillespie(_)     => GillespieSim.run(&compiled, &params, seed, &sim_config),
        SimConfig::TauLeap(_)       => TauLeapSim.run(&compiled, &params, seed, &sim_config),
        SimConfig::ChainBinomial(_) => ChainBinomialSim.run(&compiled, &params, seed, &sim_config),
        SimConfig::Ode(_) => return Err("ODE backend not yet supported in WASM".into()),
    }.map_err(|e| e.to_string())?;

    let snapshots: Vec<serde_json::Value> = trajectory.snapshots.iter().map(|snap| {
        serde_json::json!({
            "t":      snap.t,
            "counts": snap.int_state.counts,
            "values": snap.real_state.values,
            "flows":  snap.flows.counts,
        })
    }).collect();

    serde_json::to_string(&serde_json::json!({
        "int_compartment_names":  int_comp_names,
        "real_compartment_names": real_comp_names,
        "transition_names":       transition_names,
        "snapshots":              snapshots,
    })).map_err(|e| e.to_string())
}
