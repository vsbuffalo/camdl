//! gh#209 — flat-bytecode VM (`eval_flat`) vs recursive `eval_resolved`
//! (tree-walk), on a real model's rate exprs.
//!
//!   cargo bench -p sim --bench flat_eval -- <model.ir.json> [label]
//!
//! Steps:
//!   1. Load the IR, build the `FlatVm` (off the hot path, like `resolve_expr`).
//!   2. Bit-exact check vs `eval_resolved` on every rate — FAIL LOUDLY (exit 1)
//!      on any mismatch. Byte-identity is the non-negotiable invariant.
//!   3. Print the op histogram (superinstruction / arith / binding / delegate
//!      counts) so coverage is auditable.
//!   4. Median-of-9 timing of `eval_resolved` (baseline) vs `eval_flat`,
//!      printing ns/eval for each and the speedup.
//!   5. A second `eval_flat` variant with superinstructions LOWERED back to the
//!      generic binary op, so the report shows whether superinstructions helped.

use std::hint::black_box;
use std::time::{Duration, Instant};

use ir::expr::BinOp;
use sim::compiled_model::CompiledModel;
use sim::flat_eval::{
    build, eval_flat, op_histogram, scratch_capacity, FlatCache, FlatProg, FlatVm, Op,
};
use sim::propensity::EvalCtx;
use sim::resolved_expr::eval_resolved;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 }
}

/// Lower the superinstruction ops (Add/Sub/Mul/Div) back to the generic
/// `Op::BinOther`, to measure the superinstruction win in isolation. Pure
/// rewrite of the op tape; everything else is untouched, so it stays bit-exact.
fn without_superinstructions(vm: &FlatVm) -> FlatVm {
    let lower = |p: &FlatProg| -> FlatProg {
        let mut q = p.clone();
        for op in &mut q.ops {
            let lowered = match &*op {
                Op::Add => Some(BinOp::Add),
                Op::Sub => Some(BinOp::Sub),
                Op::Mul => Some(BinOp::Mul),
                Op::Div => Some(BinOp::Div),
                _ => None,
            };
            if let Some(b) = lowered {
                *op = Op::BinOther(b);
            }
        }
        q
    };
    FlatVm {
        rates: vm.rates.iter().map(lower).collect(),
        binding_progs: vm.binding_progs.iter().map(lower).collect(),
        n_bindings: vm.n_bindings,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: flat_eval <model.ir.json> [label]");
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
    let rates = &cm.resolved.rates;
    let bindings = &cm.resolved.bindings;

    // Build the VM once, off the hot path.
    let vm: FlatVm = build(rates, bindings);
    let vm_no_super: FlatVm = without_superinstructions(&vm);
    let cap = scratch_capacity(&vm);
    let hist = op_histogram(&vm);
    let total_ops: usize = vm.rates.iter().map(|p| p.ops.len()).sum::<usize>()
        + vm.binding_progs.iter().map(|p| p.ops.len()).sum::<usize>();

    sim::eval_stats::set_allow_degenerate_rates(true);
    let mk_ctx = |t: f64| EvalCtx {
        model: &cm,
        int_s: &int_s,
        real_s: &real_s,
        params: &params,
        t,
        dt: 1.0,
        projected: None,
        aux: None,
        int_float_override: None, per_eval: None,
    };

    // ── Bit-exactness: eval_flat must match eval_resolved on every rate ──
    let check = |vm: &FlatVm, name: &str| -> u64 {
        let ctx = mk_ctx(0.0);
        let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
        let mut cache = FlatCache::new(vm.n_bindings);
        let mut mismatches = 0u64;
        let mut max_abs = 0.0f64;
        for i in 0..n_tr {
            let r = eval_resolved(&rates[i], &ctx);
            let s = eval_flat(vm, &vm.rates[i], &ctx, &mut scratch, &mut cache);
            if r.to_bits() != s.to_bits() && !(r.is_nan() && s.is_nan()) {
                mismatches += 1;
                max_abs = max_abs.max((r - s).abs());
            }
        }
        if mismatches > 0 {
            eprintln!("  [{name}] BIT-EXACT FAIL: {mismatches} mismatches, max|Δ|={max_abs:.3e}");
        }
        mismatches
    };
    let mm = check(&vm, "eval_flat");
    let mm_ns = check(&vm_no_super, "eval_flat(no-super)");
    if mm != 0 || mm_ns != 0 {
        eprintln!("BIT-EXACT FAILED — refusing to report timings.");
        std::process::exit(1);
    }

    // ── Timers (the full n_tr-rate sweep per rep) ──
    let run_tree = |reps: u64| -> Duration {
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..reps {
            let ctx = EvalCtx {
                model: &cm, int_s: &int_s, real_s: &real_s,
                params: black_box(params.as_slice()), t: black_box(0.0), dt: 1.0,
                projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            for i in 0..n_tr {
                acc += eval_resolved(&rates[i], &ctx);
            }
        }
        black_box(acc);
        t0.elapsed()
    };
    let run_flat = |vm: &FlatVm, reps: u64| -> Duration {
        let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
        let mut cache = FlatCache::new(vm.n_bindings);
        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..reps {
            let ctx = EvalCtx {
                model: &cm, int_s: &int_s, real_s: &real_s,
                params: black_box(params.as_slice()), t: black_box(0.0), dt: 1.0,
                projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            for i in 0..n_tr {
                acc += eval_flat(vm, &vm.rates[i], &ctx, &mut scratch, &mut cache);
            }
        }
        black_box(acc);
        t0.elapsed()
    };

    // Calibrate reps so each trial runs ~0.4 s.
    let calib = 4000u64;
    let per_rep = run_tree(calib).as_secs_f64() / calib as f64;
    let reps = ((0.4 / per_rep) as u64).max(2000);
    // Warm.
    run_tree(reps / 4);
    run_flat(&vm, reps / 4);
    run_flat(&vm_no_super, reps / 4);

    const TRIALS: usize = 9;
    let mut tree_nspe = Vec::with_capacity(TRIALS);
    let mut flat_nspe = Vec::with_capacity(TRIALS);
    let mut flat_ns_nspe = Vec::with_capacity(TRIALS);
    println!("model\tn_transitions\ttotal_ops\tkind\ttrial\treps\tevals\tns_per_eval");
    for trial in 0..TRIALS {
        let dt_tree = run_tree(reps);
        let dt_flat = run_flat(&vm, reps);
        let dt_flat_ns = run_flat(&vm_no_super, reps);
        let evals = reps * n_tr as u64;
        for (kind, dur, sink) in [
            ("eval_resolved", dt_tree, &mut tree_nspe),
            ("eval_flat", dt_flat, &mut flat_nspe),
            ("eval_flat_no_super", dt_flat_ns, &mut flat_ns_nspe),
        ] {
            let ns = dur.as_nanos() as f64 / evals as f64;
            sink.push(ns);
            println!("{label}\t{n_tr}\t{total_ops}\t{kind}\t{trial}\t{reps}\t{evals}\t{ns:.4}");
        }
    }
    let t_med = median(&mut tree_nspe);
    let f_med = median(&mut flat_nspe);
    let fns_med = median(&mut flat_ns_nspe);

    // ── Report ──
    eprintln!("\n── {label} ── flat-bytecode VM vs eval_resolved ─────────────");
    eprintln!("  transitions={n_tr}  total_flat_ops={total_ops}  scratch_cap={cap}");
    eprintln!(
        "  op histogram: superinstr(+-*/)={} bin_other={} int_pop_sum={} mixed_pop_sum={} time_func={} projected={} binding={} delegate={} other={}",
        hist.superinstr, hist.bin_other, hist.int_pop_sum, hist.mixed_pop_sum,
        hist.time_func, hist.projected, hist.binding, hist.delegate, hist.other,
    );
    eprintln!("  bit-exact ({n_tr} rates): {}", if mm == 0 { "YES" } else { "NO!" });
    eprintln!();
    eprintln!("  AGGREGATE (median of {TRIALS}, full {n_tr}-rate sweep):");
    eprintln!("    eval_resolved (tree)           : {t_med:.3} ns/eval   (baseline)");
    eprintln!("    eval_flat (superinstructions)  : {f_med:.3} ns/eval   speedup {:.3}x", t_med / f_med);
    eprintln!("    eval_flat (no superinstr)      : {fns_med:.3} ns/eval   speedup {:.3}x", t_med / fns_med);
    let super_delta = fns_med / f_med;
    eprintln!(
        "    ── superinstructions: {} ({:.3}x vs generic-bin variant)",
        if f_med < fns_med { "HELP" } else { "no win / regress" },
        super_delta,
    );
    eprintln!();
    let best = t_med / f_med;
    eprintln!(
        "  BOTTOM LINE: eval_flat speedup = {best:.3}x  ({})",
        if best > 1.0 { "VM WINS" } else { "tree still wins" },
    );
}
