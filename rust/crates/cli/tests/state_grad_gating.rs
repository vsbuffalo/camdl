//! gh#439 A2: the runtime drives state-Jacobian emission by method.
//!
//! The WrtPop state-Jacobian (`rate_state_grad` / `projection_state_grad`) is
//! read only by `fit --method nuts` on the `ode` backend (the ODE
//! forward-sensitivity gradient). Every other path — `simulate`, IF2, PMMH, PF,
//! PGAS, `mh` — compiles lean (`camdlc --no-state-grad`), dropping a map that
//! reaches 95%+ of the IR on mean-field-coupled models.
//!
//! This checks three things end-to-end against the real `camdlc` / `camdl`:
//!   1. lean vs full emission — `--no-state-grad` empties `rate_state_grad`, the
//!      default emits it on a state-dependent model;
//!   2. run_id-neutrality — a lean and a full compile of the SAME model share the
//!      SAME model identity (`content_hash`), the precondition A2 relies on
//!      (runid SV=2). If this fails, the gradient-independent-identity fix is
//!      incomplete and A2 would re-key the store;
//!   3. the `camdl simulate` runtime path actually compiles lean (the cached IR
//!      it produces carries no state-Jacobian).
//!
//! Skipped when the release binary or camdlc isn't built.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn real_camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn skip_if_unbuilt() -> Option<(PathBuf, PathBuf)> {
    let bin = camdl_bin();
    let cc = real_camdlc();
    if !bin.exists() || !cc.exists() {
        eprintln!("skipping: camdl/camdlc not built");
        return None;
    }
    Some((bin, cc))
}

/// An SIR whose two transitions have state-dependent rates (`beta*S*I/N`,
/// `gamma*I`), so the full compile emits `rate_state_grad` (∂rate/∂compartment)
/// on both — the map `--no-state-grad` skips. Params match the `simulate` call
/// below.
const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 499  I = 1 }
simulate { from = 0 'days  to = 20 'days }
"#;

/// Compile `model` with the real camdlc, optionally passing `--no-state-grad`,
/// and parse the IR emitted on stdout.
fn compile(camdlc: &Path, model: &Path, lean: bool) -> ir::Model {
    let mut cmd = Command::new(camdlc);
    cmd.arg(model);
    if lean {
        cmd.arg("--no-state-grad");
    }
    let out = cmd.output().expect("spawn camdlc");
    assert!(
        out.status.success(),
        "camdlc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8(out.stdout).expect("camdlc output not UTF-8");
    ir::from_str(&json).expect("parse IR")
}

fn any_transition_has_state_grad(m: &ir::Model) -> bool {
    m.transitions.iter().any(|t| !t.rate_state_grad.is_empty())
}
fn all_transitions_state_grad_empty(m: &ir::Model) -> bool {
    m.transitions.iter().all(|t| t.rate_state_grad.is_empty())
}

/// (1) lean vs full emission and (2) run_id-neutrality, proved directly against
/// camdlc — no runtime dispatch involved, so a failure localises to the compiler
/// gate or the identity hash, not the CLI wiring.
#[test]
fn lean_and_full_emission_and_run_id_neutral() {
    use runid::ContentAddressed;
    let Some((_bin, camdlc)) = skip_if_unbuilt() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();

    let full = compile(&camdlc, &model, false);
    let lean = compile(&camdlc, &model, true);

    // Full carries the state-Jacobian; lean drops it on every transition.
    assert!(
        any_transition_has_state_grad(&full),
        "the default compile must emit rate_state_grad on a state-dependent model"
    );
    assert!(
        all_transitions_state_grad_empty(&lean),
        "--no-state-grad must leave every transition's rate_state_grad empty"
    );

    // A2 precondition: dropping the gradient maps must NOT change model identity
    // (runid SV=2, `content_hash` excludes the gradient maps). If these differ,
    // making `simulate` compile lean would re-key the CAS store — the whole point
    // of the identity fix that unblocked A2 is that it does not.
    assert_eq!(
        full.content_hash(),
        lean.content_hash(),
        "lean and full compiles of the same model must share a model identity \
         (run_id-neutral); if they differ, the gradient-independent-identity fix \
         (runid SV=2) is incomplete and A2 would re-key the store"
    );
}

/// (3) The `camdl simulate` runtime path compiles lean. We point the IR cache at
/// a fresh dir, run one simulate, then read the single cached IR back and assert
/// it carries no state-Jacobian — i.e. the runtime passed `--no-state-grad`.
#[test]
fn simulate_runtime_compiles_lean() {
    let Some((bin, camdlc)) = skip_if_unbuilt() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, SIR).unwrap();
    let cache = tmp.path().join("ircache");
    let out_dir = tmp.path().join("out");

    let st = Command::new(&bin)
        .args([
            "simulate",
            model.to_str().unwrap(),
            "--backend",
            "chain_binomial",
            "--seed",
            "1",
            "--param",
            "beta=0.3",
            "--param",
            "gamma=0.1",
            "--param",
            "N0=1000",
            "--output-dir",
            out_dir.to_str().unwrap(),
            "--progress",
            "none",
        ])
        .env("CAMDLC", &camdlc)
        .env("CAMDL_IR_CACHE_DIR", &cache)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status()
        .expect("spawn camdl simulate");
    assert!(st.success(), "simulate should succeed");

    // The cache holds exactly one compiled IR (`<key>.ir.json`); its `.deps`
    // sidecar ends differently, so this filter picks the IR alone.
    let ir_files: Vec<PathBuf> = std::fs::read_dir(&cache)
        .expect("cache dir exists after a compile")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .collect();
    assert_eq!(
        ir_files.len(),
        1,
        "expected exactly one cached IR, found {ir_files:?}"
    );

    let json = std::fs::read_to_string(&ir_files[0]).unwrap();
    let cached: ir::Model = ir::from_str(&json).expect("parse cached IR");
    assert!(
        all_transitions_state_grad_empty(&cached),
        "`camdl simulate` must compile lean (--no-state-grad): the cached IR \
         carried a state-Jacobian, so the runtime failed to gate emission"
    );
}
