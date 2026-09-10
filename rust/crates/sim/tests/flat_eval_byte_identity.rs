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
//!   3. gh#815 — a model carrying gh#272 LICM-hoisted (`per_eval`) bindings, in
//!      both `EvalCtx::per_eval` states (staged prologue and on-demand
//!      fallback). No `ir/golden` model carries one, so without this the gate
//!      says nothing about the `Op::PerEval` arm.
//!
//! Every test here sets `CAMDL_EVAL_FLAT` first (see `enable_flat_toggle`), so
//! `CompiledModel::new` builds the VM through the production guard as well —
//! which is what the gh#815 non-vacuity assertions check.

use std::sync::Once;

use sim::compiled_model::CompiledModel;
use sim::flat_eval::{build, eval_flat, scratch_capacity, FlatCache, FlatProg, FlatVm, Op};
use sim::propensity::EvalCtx;
use sim::resolved_expr::{eval_per_eval_scratch, eval_resolved, ResolvedExpr};
use sim::state::{IntState, RealState};

/// Turn the flat-VM toggle on for this test binary, before anything can read it.
///
/// `flat_eval::eval_flat_enabled` memoizes the env read in a `OnceLock`, and
/// `cargo test` runs the tests in this file on threads of one process. Routing
/// every test through a `Once` that sets the variable as its first action means
/// the memoized value is deterministically `true` — no test can lose a race with
/// another test's `CompiledModel::new`.
fn enable_flat_toggle() {
    static INIT: Once = Once::new();
    INIT.call_once(|| std::env::set_var("CAMDL_EVAL_FLAT", "1"));
}

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
    enable_flat_toggle();
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
    enable_flat_toggle();
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
        // A NaN predicate must take the ELSE branch in both evaluators.
        // `eval_resolved` tests `pred > 0.0`, which is false for NaN; the VM's
        // `JumpIfFalse` has to agree, and `pred <= 0.0` alone would not — it is
        // false for NaN too, so the NaN would fall through to THEN.
        (
            "Cond with a NaN predicate",
            ResolvedExpr::Cond {
                pred: Box::new(ResolvedExpr::Const(f64::NAN)),
                then_: Box::new(ResolvedExpr::Const(1.0)),
                else_: Box::new(ResolvedExpr::Const(2.0)),
            },
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
    let nan_cond = eval_resolved(
        &ResolvedExpr::Cond {
            pred: Box::new(ResolvedExpr::Const(f64::NAN)),
            then_: Box::new(ResolvedExpr::Const(1.0)),
            else_: Box::new(ResolvedExpr::Const(2.0)),
        },
        &ctx,
    );
    assert_eq!(
        nan_cond, 2.0,
        "a NaN predicate must select the else branch, not then"
    );
}

/// Count `Op::PerEval` across every tape in the VM (rates + binding bodies).
fn count_per_eval_ops(vm: &FlatVm) -> usize {
    let in_prog = |p: &FlatProg| p.ops.iter().filter(|o| matches!(o, Op::PerEval(_))).count();
    vm.rates.iter().map(&in_prog).sum::<usize>()
        + vm.binding_progs.iter().map(&in_prog).sum::<usize>()
}

/// gh#815 — byte-identity on a model whose rates carry gh#272 LICM-hoisted
/// (`per_eval`) bindings, which is the case the flat VM used to refuse to build
/// for at all.
///
/// The two tests above iterate `ir/golden`, where no model carries a per-eval
/// binding, so neither says anything about `Op::PerEval`. `licm_ab_on.ir.json`
/// (the gh#272 A/B gate's ON fixture — a 4-patch in-model gravity kernel,
/// compiled with LICM on) has 80 of them.
///
/// Both `EvalCtx::per_eval` states are exercised, because `Op::PerEval` has two
/// arms and only one of them is the fast path:
///   - `Some(scratch)` — the staged prologue a forward backend lends across a
///     θ-stable span; the read is one array index.
///   - `None` — no scratch staged; the body is evaluated on demand.
///
/// Three non-vacuity assertions guard the gate itself, because the historical
/// failure mode here was silence, not a red: the VM was gated off for per-eval
/// models, so this comparison would have compared nothing.
#[test]
fn flat_matches_eval_resolved_on_a_per_eval_model() {
    enable_flat_toggle();
    sim::eval_stats::set_allow_degenerate_rates(true);

    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/licm_ab_on.ir.json");
    let model = load_model_filled(path.to_str().unwrap());

    // NON-VACUITY 1 — the fixture actually carries hoisted bindings. If LICM's
    // output shape ever changes and the fixture is regenerated flat, this fires
    // rather than letting the test pass on a model with nothing to check.
    assert!(
        !model.per_eval_bindings.is_empty(),
        "licm_ab_on fixture has no per_eval_bindings — it no longer exercises \
         the gh#815 path; regenerate it from licm_ab.camdl with LICM on"
    );

    let cm = CompiledModel::new(model).expect("compile licm_ab_on");

    // NON-VACUITY 2 — `CompiledModel::new` built the VM for this model. This is
    // the assertion that fails if the gh#815 guard is ever re-narrowed (e.g. the
    // `per_eval_bindings.is_empty()` clause coming back): the flat VM would be
    // `None`, `CAMDL_EVAL_FLAT` would validate and change nothing, and every
    // comparison below would still be green against a locally-built VM.
    let vm = cm.resolved.flat_vm.as_ref().expect(
        "CAMDL_EVAL_FLAT is set and this model has per-eval bindings, but \
         CompiledModel::new built no FlatVm — the flat path is gated off for \
         LICM'd models again (gh#815), so the toggle is a silent no-op",
    );

    // NON-VACUITY 3 — the emitter really lowered `PerEvalRef` to `Op::PerEval`.
    // Guards against the arm regressing to `Op::Delegate`, which would stay
    // byte-identical (and so invisible to the comparison below) while giving
    // back the whole point of the hoist.
    let n_per_eval = count_per_eval_ops(vm);
    assert!(
        n_per_eval > 0,
        "the flat VM for licm_ab_on emitted no Op::PerEval — PerEvalRef is no \
         longer lowered to a dedicated op, so this test is comparing tapes that \
         never take the gh#815 path"
    );

    let params = cm.default_params.clone();
    let (int_s0, real_s0) = cm.initial_state_mean(&params).expect("initial_state_mean");

    // A per-eval body is param/table/const-only (the gh#272 keystone invariant),
    // so the prologue is constant across `t` and state — stage it once.
    let scratch_pe = eval_per_eval_scratch(&cm, &params, 0.0, 1.0);
    assert_eq!(
        scratch_pe.len(),
        cm.resolved.per_eval_bindings.len(),
        "staged prologue does not cover every per-eval slot"
    );

    let cap = scratch_capacity(vm);
    let n_tr = cm.model.transitions.len();
    let zeroed = IntState::from_vec(vec![0; int_s0.counts.len()]);
    let doubled = IntState::from_vec(int_s0.counts.iter().map(|&c| c.saturating_mul(2)).collect());
    let state_variants = [(&int_s0, &real_s0), (&zeroed, &real_s0), (&doubled, &real_s0)];

    let mut checked_evals = 0u64;
    for t in [0.0f64, 7.0, 30.0, 180.0, 365.0] {
        for (is, rs) in state_variants {
            // `None` is the on-demand arm, `Some` the staged arm. Each is
            // compared against `eval_resolved` under the SAME ctx, so the flat
            // VM is pinned to the recursive path in both.
            for per_eval in [None, Some(&scratch_pe[..])] {
                let staged = per_eval.is_some();
                let ctx = EvalCtx {
                    model: &cm,
                    int_s: is,
                    real_s: rs,
                    params: &params,
                    t,
                    dt: 1.0,
                    projected: None,
                    aux: None,
                    int_float_override: None,
                    per_eval,
                };
                let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
                let mut cache = FlatCache::new(vm.n_bindings);
                for i in 0..n_tr {
                    let r = eval_resolved(&cm.resolved.rates[i], &ctx);
                    let s = eval_flat(vm, &vm.rates[i], &ctx, &mut scratch, &mut cache);
                    assert!(
                        bits_eq(r, s),
                        "licm_ab_on rate[{i}] @t={t} (staged={staged}): \
                         eval_resolved={r:?} ({:#018x}) != eval_flat={s:?} ({:#018x})",
                        r.to_bits(),
                        s.to_bits(),
                    );
                    checked_evals += 1;
                }
            }
        }
    }

    eprintln!(
        "flat byte-identity (per-eval): {n_per_eval} PerEval ops, {} slots, \
         {checked_evals} rate evals (staged and on-demand) — all bit-identical",
        scratch_pe.len()
    );
}
