//! Integration test: `camdl render --format graph` emits the structured flow
//! graph (`model.graph.json`) a viewer (camdl-watch) draws the compartmental
//! diagram from without recompiling — the sidecar `fit run` archives beside
//! `model.render.json`.
//!
//! Proposal: docs/dev/proposals/2026-07-20-model-diagram-and-identifiability.md
//!
//! Shells out to the built `camdl` binary and exercises the real
//! compile → render-graph path (`camdl render` delegating to `camdlc render
//! --format graph`, the same emitter `util::render_model_graph_json` reads).
//! The model is an age-stratified SIR carrying every feature the emitter must
//! surface: an `age` plate, a mean-field pool read through `let` bindings, a
//! `consecutive(age)` aging edge, a birth inflow (empty source), and a death
//! outflow (empty sink).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn require_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
dimensions { age = [young, old] }
stratify(by = age)
let N[a in age] = S[a] + I[a] + R[a]
let inf_tot = sum(a in age, I[a])
let ntot    = sum(a in age, N[a])
parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
  mu    : rate in [0.0, 0.1]
}
transitions {
  infection[a in age] : S[a] --> I[a] @ beta * (inf_tot / ntot) * S[a]
  recovery[a in age]  : I[a] --> R[a] @ gamma * I[a]
  aging[c in compartments, (a, a_next) in consecutive(age)] : c[a] --> c[a_next] @ mu * c[a]
  birth : --> S[young] @ mu * ntot
  death[c in compartments, a in age] : c[a] --> @ mu * c[a]
}
init { S[young] = 500  S[old] = 400  I[young] = 10 }
simulate { from = 0 'days  to = 5 'days }
scenarios { baseline { set = { beta = 0.3  gamma = 0.1  mu = 0.01 } } }
"#;

/// Find the edge object with the given `id`.
fn edge<'a>(g: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    g["edges"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == id)
        .unwrap_or_else(|| panic!("no edge with id {id}"))
}

#[test]
fn render_graph_emits_the_flow_graph_a_viewer_draws() {
    let bin = require_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("agepool.camdl");
    std::fs::write(&model, MODEL).unwrap();

    let run = Command::new(&bin)
        .args(["render", &model.to_string_lossy(), "--format", "graph"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn");
    assert!(
        run.status.success(),
        "camdl render --format graph failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let g: serde_json::Value = serde_json::from_slice(&run.stdout)
        .expect("the graph render must be well-formed JSON");

    // Base compartments are the nodes.
    let node_ids: Vec<&str> = g["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_str().unwrap())
        .collect();
    assert_eq!(node_ids, ["S", "I", "R"], "nodes are the base compartments");

    // The declared dimension is a plate with its levels.
    let plates = g["plates"].as_array().unwrap();
    assert_eq!(plates.len(), 1, "one plate (age)");
    assert_eq!(plates[0]["name"], "age");

    // aging steps along the age plate; birth is an inflow; death is an outflow.
    assert_eq!(edge(&g, "aging")["advances"], "age", "aging advances along age");
    assert!(edge(&g, "birth")["from"].is_null(), "birth has no source");
    assert!(edge(&g, "death")["to"].is_null(), "death has no sink");

    // The mean-field pool couples infection (and birth) across age.
    assert_eq!(edge(&g, "birth")["reads_pool"], true);
    assert_eq!(edge(&g, "recovery")["reads_pool"], false);
    let couples_infection = g["couplings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["edge"] == "infection" && c["over"].as_array().unwrap().iter().any(|d| d == "age"));
    assert!(couples_infection, "infection couples over the age pool");
}
