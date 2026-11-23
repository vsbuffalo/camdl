//! gh#1046, on compiled files: one source compiled in full and with
//! `--no-state-grad` is one model.
//!
//! Which source: `tests/fixtures/hoisted_identity/hoisted_rate.camdl`, whose
//! recovery rate has a parameter-only factor that the compiler hoists. The two
//! committed files beside it are its two compiles (`make
//! update-hoisted-identity-golden`; `make check-hoisted-identity-golden` fails
//! when either drifts from the source).
//!
//! What it checks: the two files have the same `model_ir_hash`, and the test
//! bites, because the rate refers to its hoisted factor by a different name in
//! each. Before the fix the hashes were `54f591759354873b…` and
//! `38edb1fe10c6e8f3…`.

use std::path::{Path, PathBuf};

use ir::expr::Expr;

const FULL: &str = "tests/fixtures/hoisted_identity/ir/hoisted_rate.ir.json";
const LEAN: &str = "tests/fixtures/hoisted_identity/ir/hoisted_rate.no_state_grad.ir.json";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn load(rel: &str) -> ir::Model {
    let path = repo_root().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    ir::from_str(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

/// Every hoisted name a rate refers to, in order.
fn rate_refs(m: &ir::Model) -> Vec<String> {
    let mut out = Vec::new();
    for t in &m.transitions {
        let json = serde_json::to_value(&t.rate).expect("serialize rate");
        collect_refs(&json, &mut out);
    }
    out
}

fn collect_refs(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(o) => {
            if let Some(serde_json::Value::String(name)) = o.get("per_eval_ref") {
                out.push(name.clone());
            }
            o.values().for_each(|x| collect_refs(x, out));
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_refs(x, out)),
        _ => {}
    }
}

#[test]
fn a_full_and_a_no_state_grad_compile_are_one_model() {
    let full = load(FULL);
    let lean = load(LEAN);

    // The test bites: both compiles hoist, and they number the hoisted factor
    // the rate reads differently.
    let (full_refs, lean_refs) = (rate_refs(&full), rate_refs(&lean));
    assert!(!full_refs.is_empty(), "the full compile hoists nothing out of a rate");
    assert!(!lean_refs.is_empty(), "the lean compile hoists nothing out of a rate");
    assert_ne!(
        full_refs, lean_refs,
        "the two compiles name the hoisted factor alike, so this fixture no longer \
         exercises gh#1046; find a source whose numbering differs"
    );

    assert_eq!(
        ir::model_hash::model_ir_hash(&full).to_hex(),
        ir::model_hash::model_ir_hash(&lean).to_hex(),
        "a full and a --no-state-grad compile of one source must be one model"
    );
}

/// The factor a rate refers to is still identity: change its body in one file
/// and the model differs.
#[test]
fn the_hoisted_factor_a_rate_reads_is_identity() {
    let base = load(FULL);
    let name = rate_refs(&base).into_iter().next().expect("a hoisted factor");
    let mut changed = base.clone();
    let b = changed
        .per_eval_bindings
        .iter_mut()
        .find(|b| b.name == name)
        .expect("the rate's factor is in the table");
    b.expr = Expr::param("beta");
    assert_ne!(
        ir::model_hash::model_ir_hash(&base).to_hex(),
        ir::model_hash::model_ir_hash(&changed).to_hex()
    );
}
