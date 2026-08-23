//! gh#209 — byte-identity guard for the flat-bytecode propensity VM.
//!
//! `CAMDL_EVAL_FLAT` swaps the recursive `eval_resolved` tree-walk for a
//! compiled op tape (`FlatVm`) executed over an unsafe raw-pointer stack. The
//! non-negotiable invariant is that `eval_flat` returns a value **bit-identical**
//! to `eval_resolved` for every rate expression — not merely close. f64 add is
//! non-associative and `+0.0`/`-0.0` differ in bits, so "looks equal" is not
//! good enough: a regrouped fold or a flipped sum seed silently biases an
//! inference run.
//!
//! Before this test, `eval_flat`/`FlatVm` were exercised by a benchmark only —
//! never in `cargo test`, so the invariant rode entirely on a binary nobody runs
//! in CI. This file pins it two ways:
//!   1. Every rate of every golden model, at several times and state variants
//!      (the realistic op coverage: BinOps, IntPopSum, TimeFunc, Cond, bindings).
//!   2. Hand-built latent edge cases the emitter does not currently produce
//!      (empty Reduce / empty sums → the `-0.0` fold seed; a mixed int+real
//!      PopSum → the partial-sum grouping). These are where the seed/grouping
//!      bugs live, and they are not reachable from any golden.

use sim::compiled_model::CompiledModel;
use sim::flat_eval::{build, eval_flat, scratch_capacity, FlatCache, FlatVm};
use sim::propensity::EvalCtx;
use sim::resolved_expr::{eval_resolved, ResolvedExpr};
use sim::state::{IntState, RealState};

/// Golden IR carries declared parameters with bounds but no resolved value
/// (compact IR drops scenario presets), so a bare compile is rejected. Fill any
/// unresolved parameter with the midpoint of its bounds — always in range, and
/// the concrete value is irrelevant to byte-identity (both evaluators run the
/// same ops on it).
fn load_model_filled(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    let mut model: ir::Model =
        ir::from_str(&json).unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e));
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.value.bounds() {
                Some((lo, hi)) => 0.5 * (lo + hi),
                None => 0.5,
            };
            p.value = p.value.with_value(v);
        }
    }
    model
}

/// Bit-for-bit equality, with any-NaN treated as equal (a NaN payload carries no
/// meaning for a propensity and both evaluators take the same arithmetic path).
fn bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn flat_matches_eval_resolved_on_all_goldens() {
    // Match the simulate/inference default: degenerate rates fall back to the
    // legacy 0.0 sentinel rather than NaN, so the two evaluators agree on the
    // div-by-zero arm too.
    sim::eval_stats::set_allow_degenerate_rates(true);

    let dir = "../../../ir/golden";
    let mut checked_models = 0usize;
    let mut checked_evals = 0u64;

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("read golden dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .collect();
    paths.sort();

    for path in &paths {
        let fname = path.file_name().unwrap().to_string_lossy().into_owned();
        let model = load_model_filled(path.to_str().unwrap());
        // A model whose midpoint params can't build a valid initial state isn't a
        // byte-identity failure — it's a fixture-setup limitation. Skip it
        // VISIBLY (logged below) rather than silently, and hold a floor count.
        let cm = match CompiledModel::new(model) {
            Ok(cm) => cm,
            Err(e) => {
                eprintln!("SKIP {fname}: compile with midpoint params: {e}");
                continue;
            }
        };
        let params = cm.default_params.clone();
        let (int_s0, real_s0) = match cm.initial_state_mean(&params) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP {fname}: initial_state with midpoint params: {e}");
                continue;
            }
        };

        let vm: FlatVm = build(&cm.resolved.rates, &cm.resolved.bindings);
        let cap = scratch_capacity(&vm);
        let n_tr = cm.model.transitions.len();

        // State variants exercise different rate magnitudes and flip Cond /
        // div-guard predicates (e.g. an empty compartment).
        let zeroed = IntState::from_vec(vec![0; int_s0.counts.len()]);
        let doubled =
            IntState::from_vec(int_s0.counts.iter().map(|&c| c.saturating_mul(2)).collect());
        let state_variants = [
            (&int_s0, &real_s0),
            (&zeroed, &real_s0),
            (&doubled, &real_s0),
        ];

        for t in [0.0f64, 7.0, 30.0, 180.0, 365.0] {
            for (is, rs) in state_variants {
                let ctx = EvalCtx {
                    model: &cm,
                    int_s: is,
                    real_s: rs,
                    params: &params,
                    t,
                    dt: 1.0,
                    projected: None,
                    aux: None,
                    int_float_override: None, per_eval: None,
                };
                let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
                let mut cache = FlatCache::new(vm.n_bindings);
                for i in 0..n_tr {
                    let r = eval_resolved(&cm.resolved.rates[i], &ctx);
                    let s = eval_flat(&vm, &vm.rates[i], &ctx, &mut scratch, &mut cache);
                    assert!(
                        bits_eq(r, s),
                        "{fname} rate[{i}] @t={t}: eval_resolved={r:?} ({:#018x}) \
                         != eval_flat={s:?} ({:#018x})",
                        r.to_bits(),
                        s.to_bits(),
                    );
                    checked_evals += 1;
                }
            }
        }
        checked_models += 1;
    }

    assert!(
        checked_models >= 10,
        "expected ≥10 golden models exercised, only {checked_models} found in {dir}"
    );
    eprintln!(
        "flat byte-identity: {checked_models} models, {checked_evals} rate evals — all bit-identical"
    );
}

#[test]
fn flat_matches_eval_resolved_on_latent_edge_cases() {
    sim::eval_stats::set_allow_degenerate_rates(true);

    // A real CompiledModel only satisfies `EvalCtx.model`; the synthetic exprs
    // below index directly into the int_s/real_s slices we control and never
    // dereference the model, so sir_basic (3 int compartments, no real ones) is
    // a fine carrier even for the mixed-pop-sum case.
    let cm = CompiledModel::new(load_model_filled("../../../ir/golden/sir_basic.ir.json"))
        .expect("compile sir_basic with midpoint params");
    let params = cm.default_params.clone();

    // int_s[0]=1 and real_s = [1e16, -1e16] make the MixedPopSum grouping
    // observable: eval_resolved computes (int_sum) + (real_sum) = 1.0 + 0.0 =
    // 1.0, whereas a single continuous fold ((-0.0 + 1) + 1e16) + (-1e16) loses
    // the 1 at 1e16 magnitude and yields 0.0. The flat VM must match the former.
    let int_s = IntState::from_vec(vec![1, 0, 0]);
    let real_s = RealState::from_vec(vec![1e16, -1e16]);

    let synthetic: Vec<(&str, ResolvedExpr)> = vec![
        ("empty Reduce", ResolvedExpr::Reduce(vec![])),
        ("empty IntPopSum", ResolvedExpr::IntPopSum(vec![])),
        (
            "empty MixedPopSum",
            ResolvedExpr::MixedPopSum { int_indices: vec![], real_indices: vec![] },
        ),
        ("singleton Reduce", ResolvedExpr::Reduce(vec![ResolvedExpr::Const(0.0)])),
        (
            "mixed int+real grouping",
            ResolvedExpr::MixedPopSum { int_indices: vec![0], real_indices: vec![0, 1] },
        ),
    ];

    let rates: Vec<ResolvedExpr> = synthetic.iter().map(|(_, e)| e.clone()).collect();
    let vm = build(&rates, &[]);
    let cap = scratch_capacity(&vm);
    let ctx = EvalCtx {
        model: &cm,
        int_s: &int_s,
        real_s: &real_s,
        params: &params,
        t: 0.0,
        dt: 1.0,
        projected: None,
        aux: None,
        int_float_override: None, per_eval: None,
    };
    let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
    let mut cache = FlatCache::new(vm.n_bindings);
    for (i, (name, expr)) in synthetic.iter().enumerate() {
        let r = eval_resolved(expr, &ctx);
        let s = eval_flat(&vm, &vm.rates[i], &ctx, &mut scratch, &mut cache);
        assert!(
            bits_eq(r, s),
            "[{name}] eval_resolved={r:?} ({:#018x}) != eval_flat={s:?} ({:#018x})",
            r.to_bits(),
            s.to_bits(),
        );
    }

    // Also pin the *semantics*, not just agreement — byte-identity alone can't
    // catch both evaluators being wrong in the same direction.
    let empty_reduce = eval_resolved(&ResolvedExpr::Reduce(vec![]), &ctx);
    assert_eq!(
        empty_reduce.to_bits(),
        (-0.0f64).to_bits(),
        "empty Reduce must fold from -0.0 (matching Iterator::sum)"
    );
    let grouping = eval_resolved(
        &ResolvedExpr::MixedPopSum { int_indices: vec![0], real_indices: vec![0, 1] },
        &ctx,
    );
    assert_eq!(
        grouping, 1.0,
        "grouped MixedPopSum keeps the int term (1.0); a continuous fold gives 0.0"
    );
}
