//! What the sparse-coupling constant fold does when a patch is empty
//! (gh#1217). Degenerate rates are errors here: every model is compiled with
//! the default (strict) `EvalSettings`.
//!
//! - `fold_empty_patch`: patch p2 is empty and every `W[l, p2]` is a literal 0.
//!   The term `W[l, p2] * I[p2] / N[p2]` is structurally zero, and the fold
//!   drops it before any division exists, as it does for the
//!   `W * (I / N)` spelling. The unfolded model evaluates `0 * 0 / 0` and
//!   reports a division by zero; the folded model runs. This is the decided
//!   behaviour: a contribution the contact matrix zeroes is not evaluated.
//! - `fold_empty_patch_coupled`: `W[p1, p2]` is 0.5, so a term that is not
//!   structurally zero divides by `N[p2] = 0`. Folded or not, the runtime
//!   reports the division by zero.
//!
//! The fixtures are static IR compiled with `--set beta=0.3 --set gamma=0.1`;
//! see each source's header for the commands.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

fn run(name: &str) -> Result<(), String> {
    let path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("tests/fixtures")
        .join(format!("{name}.ir.json"));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));
    let model = ir::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {name}: {e}"));
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: model.simulation.t_start,
        t_end: model.simulation.t_end,
        dt: 1.0,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    let compiled = CompiledModel::new(model).expect("model failed to compile");
    let params = compiled.default_params.clone();
    ChainBinomialSim
        .run(&compiled, &params, 1, &config)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

fn expect_div_by_zero(name: &str) {
    match run(name) {
        Ok(()) => panic!("{name}: expected a division-by-zero error, but the run succeeded"),
        Err(e) => assert!(e.contains("DivByZero"), "{name}: expected DivByZero, got {e}"),
    }
}

#[test]
fn empty_patch_zeroed_by_the_contact_matrix_is_not_evaluated() {
    expect_div_by_zero("fold_empty_patch_unfolded");
    run("fold_empty_patch_folded")
        .unwrap_or_else(|e| panic!("folded model should run: the empty patch's terms are structurally zero; got {e}"));
}

#[test]
fn empty_patch_coupled_by_a_nonzero_cell_still_divides_by_zero() {
    expect_div_by_zero("fold_empty_patch_coupled_unfolded");
    expect_div_by_zero("fold_empty_patch_coupled_folded");
}
