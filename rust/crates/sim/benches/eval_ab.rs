//! A/B microbench: pre-resolved `eval_resolved` (array-indexed) vs the
//! string-keyed `eval_expr` (per-leaf `HashMap` probes), on a real model's
//! rate expressions.
//!
//!   cargo bench -p sim --bench eval_ab -- <model.ir.json> [label]
//!
//! Both evaluators walk the *same* expression tree against the *same*
//! `EvalCtx`; the only variable is index-access vs HashMap-probe at the
//! leaves (`Param`/`Pop`/`PopSum`/`TableLookup`/`TimeFunc`/`BindingRef`).
//! So the wall-clock ratio is exactly what pre-resolution bought per eval.
//!
//! Emits per-trial rows as TSV to stdout (machine-readable for plotting);
//! a human summary + a bit-exactness check go to stderr.

use std::hint::black_box;
use std::time::{Duration, Instant};

use ir::expr::Expr;
use sim::compiled_model::CompiledModel;
use sim::propensity::{eval_expr, EvalCtx};
use sim::resolved_expr::eval_resolved;

/// Count total AST nodes and the number of leaves that are string-keyed
/// lookups in `eval_expr` (the probes pre-resolution turns into `usize`
/// indexing). Mirrors `eval_expr`'s traversal.
fn count_nodes(e: &Expr, nodes: &mut u64, probes: &mut u64) {
    *nodes += 1;
    match e {
        Expr::Param(_) | Expr::Pop(_) | Expr::ObsColumnRef(_) => *probes += 1,
        Expr::PopSum(ps) => *probes += ps.pop_sum.len() as u64,
        Expr::TimeFunc(_) | Expr::BindingRef(_) | Expr::PerEvalRef(_) => *probes += 1,
        Expr::TableLookup(w) => {
            *probes += 1;
            for ix in &w.table_lookup.indices {
                count_nodes(ix, nodes, probes);
            }
        }
        Expr::BinOp(w) => {
            count_nodes(&w.bin_op.left, nodes, probes);
            count_nodes(&w.bin_op.right, nodes, probes);
        }
        Expr::UnOp(w) => count_nodes(&w.un_op.arg, nodes, probes),
        Expr::Cond(w) => {
            count_nodes(&w.cond.pred, nodes, probes);
            count_nodes(&w.cond.then, nodes, probes);
            count_nodes(&w.cond.else_, nodes, probes);
        }
        Expr::UncheckedDim(w) => count_nodes(&w.unchecked_dim.inner, nodes, probes),
        Expr::Reduce(w) => {
            for t in &w.reduce {
                count_nodes(t, nodes, probes);
            }
        }
        Expr::Const(_) | Expr::Time(_) | Expr::Dt(_) | Expr::Projected(_) => {}
    }
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: eval_ab <model.ir.json> [label]");
        std::process::exit(2);
    }
    let path = &args[1];
    let label = args.get(2).cloned().unwrap_or_else(|| {
        std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into())
    });

    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut model: ir::Model = ir::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    // Inference fixtures leave estimated params value-less; fill them from the
    // first scenario/preset (the baseline), exactly as gate_trajectory_baseline does.
    if let Some(preset) = model.presets.first().cloned() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    let cm = CompiledModel::new(model).unwrap_or_else(|e| panic!("compile {path}: {e}"));
    let params = cm.default_params.clone();
    let (int_s, real_s) = cm.initial_state(&params).expect("initial_state");
    let n_tr = cm.model.transitions.len();

    // ── Static shape: nodes + leaf-probes across all rate exprs ──────────────
    let (mut nodes, mut probes) = (0u64, 0u64);
    for tr in &cm.model.transitions {
        count_nodes(&tr.rate, &mut nodes, &mut probes);
    }
    let probes_per_eval = probes as f64 / n_tr.max(1) as f64;

    // ── Bit-exactness at the value level (independent of the trajectory gate).
    //    Both paths must agree bit-for-bit on every rate at the initial state.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let ctx = EvalCtx {
        model: &cm, int_s: &int_s, real_s: &real_s,
        params: &params, t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    let mut mismatches = 0u64;
    let mut max_abs_diff = 0.0f64;
    for (i, tr) in cm.model.transitions.iter().enumerate() {
        let r = eval_resolved(&cm.resolved.rates[i], &ctx);
        let u = eval_expr(&tr.rate, &ctx).unwrap_or(f64::NAN);
        if r.to_bits() != u.to_bits() && !(r.is_nan() && u.is_nan()) {
            mismatches += 1;
            max_abs_diff = max_abs_diff.max((r - u).abs());
        }
    }

    // ── Timing harness ───────────────────────────────────────────────────────
    let run_resolved = |reps: u64| -> Duration {
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..reps {
            let p = black_box(params.as_slice());
            let ctx = EvalCtx {
                model: &cm, int_s: &int_s, real_s: &real_s,
                params: p, t: black_box(0.0), dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            for i in 0..n_tr {
                acc += eval_resolved(&cm.resolved.rates[i], &ctx);
            }
        }
        black_box(acc);
        t0.elapsed()
    };
    let run_unresolved = |reps: u64| -> Duration {
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..reps {
            let p = black_box(params.as_slice());
            let ctx = EvalCtx {
                model: &cm, int_s: &int_s, real_s: &real_s,
                params: p, t: black_box(0.0), dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            for (i, tr) in cm.model.transitions.iter().enumerate() {
                let _ = i;
                acc += eval_expr(&tr.rate, &ctx).unwrap_or(0.0);
            }
        }
        black_box(acc);
        t0.elapsed()
    };

    // Calibrate reps so a trial ~ 400ms on the resolved path.
    let calib = 4000u64;
    let per_rep = run_resolved(calib).as_secs_f64() / calib as f64;
    let reps = ((0.4 / per_rep) as u64).max(2000);

    // warmup
    run_resolved(reps / 4);
    run_unresolved(reps / 4);

    const TRIALS: usize = 9;
    println!("model\tn_transitions\ttotal_nodes\tprobe_leaves\tprobes_per_eval\tkind\ttrial\treps\tevals\tns_total\tns_per_eval");
    let mut res_nspe = Vec::with_capacity(TRIALS);
    let mut unr_nspe = Vec::with_capacity(TRIALS);
    for trial in 0..TRIALS {
        for (kind, dur) in [("resolved", run_resolved(reps)), ("unresolved", run_unresolved(reps))] {
            let evals = reps * n_tr as u64;
            let ns_total = dur.as_nanos() as f64;
            let ns_per_eval = ns_total / evals as f64;
            if kind == "resolved" { res_nspe.push(ns_per_eval); } else { unr_nspe.push(ns_per_eval); }
            println!(
                "{label}\t{n_tr}\t{nodes}\t{probes}\t{probes_per_eval:.2}\t{kind}\t{trial}\t{reps}\t{evals}\t{ns_total:.0}\t{ns_per_eval:.4}"
            );
        }
    }

    let r_med = median(&mut res_nspe);
    let u_med = median(&mut unr_nspe);
    eprintln!("\n── {label} ─────────────────────────────────────────");
    eprintln!("  transitions={n_tr}  nodes={nodes}  probe_leaves={probes}  probes/eval={probes_per_eval:.2}");
    eprintln!("  resolved   : {r_med:.3} ns/eval (median of {TRIALS})");
    eprintln!("  unresolved : {u_med:.3} ns/eval (median of {TRIALS})");
    eprintln!("  speedup k  : {:.2}x   (Δ {:.3} ns/eval ≈ {:.3} ns/probe)",
        u_med / r_med, u_med - r_med, (u_med - r_med) / probes_per_eval.max(1e-9));
    eprintln!("  bit-exact  : {} ({} mismatches, max|Δ|={max_abs_diff:.2e})",
        if mismatches == 0 { "YES — value-identical" } else { "NO" }, mismatches);
}
