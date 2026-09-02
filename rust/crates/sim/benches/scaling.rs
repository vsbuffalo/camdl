//! Scaling micro-benchmarks for the FOI blowup study.
//!
//! Isolates the *per-step* compute cost (`eval_propensities`, `step_one`) and
//! the model *load* cost (`ir::from_str` + `CompiledModel::new`) across a
//! (patches P × ages A × coupling) grid. The macro sweep
//! (`scripts/bench_scaling.py`) measures the full pipeline but its `sim_s` is
//! dominated by JSON parse at scale; this bench is what shows that a single
//! chain-binomial step scales O(P²·A) with coupling on vs O(P·A) with it off.
//!
//! Fixtures are generated out-of-band (they can be MBs) by the
//! `make bench-micro` target, which runs `scripts/gen_scaling_models.py` +
//! `camdl compile` into `benches/fixtures/scaling/`. Any fixture that is
//! absent is skipped with a note — run `make bench-micro` to populate them.
//!
//! Run just this bench (the sibling `inference.rs` may not build):
//!     cargo bench -p sim --bench scaling

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sim::chain_binomial::{step_one, StepScratch};
use sim::compiled_model::CompiledModel;
use sim::propensity::eval_propensities;
use sim::rng::StatefulRng;

/// (patches, ages, coupling) grid. grad=minimal: forward eval never reads
/// rate_grad, and minimal IR loads fastest.
const GRID: &[(usize, usize, &str)] = &[
    (4, 1, "on"), (8, 1, "on"), (16, 1, "on"), (32, 1, "on"),
    (4, 1, "off"), (8, 1, "off"), (16, 1, "off"), (32, 1, "off"),
    (8, 7, "on"), (16, 7, "on"), (32, 7, "on"),
    (8, 7, "off"), (16, 7, "off"), (32, 7, "off"),
];

fn fixture_path(p: usize, a: usize, coup: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/fixtures/scaling")
        .join(format!("P{p}_A{a}_{coup}_minimal.ir.json"))
}

/// Load a generated fixture into a CompiledModel + default params, or None if
/// the fixture has not been generated (`make bench-micro`).
fn try_load(p: usize, a: usize, coup: &str) -> Option<(CompiledModel, Vec<f64>)> {
    let path = fixture_path(p, a, coup);
    let json = std::fs::read_to_string(&path).ok()?;
    let model: ir::Model = ir::from_str(&json)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let compiled = CompiledModel::new(model)
        .unwrap_or_else(|e| panic!("compile {}: {e}", path.display()));
    let params = compiled.default_params.clone();
    Some((compiled, params))
}

fn label(p: usize, a: usize, coup: &str) -> String {
    format!("P{p}_A{a}_{coup}")
}

fn bench_eval_propensities(c: &mut Criterion) {
    let mut g = c.benchmark_group("eval_propensities");
    let mut missing = 0;
    for &(p, a, coup) in GRID {
        let Some((model, params)) = try_load(p, a, coup) else { missing += 1; continue };
        let (int_s, real_s) = model.initial_state_mean(&params).unwrap();
        let mut out = Vec::with_capacity(model.model.transitions.len());
        g.throughput(criterion::Throughput::Elements(model.model.transitions.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(label(p, a, coup)), |b| {
            b.iter(|| {
                eval_propensities(&model, &int_s, &real_s, &params, 10.0, 1.0, None, &mut out).unwrap();
            });
        });
    }
    g.finish();
    if missing > 0 {
        eprintln!("note: {missing} scaling fixtures missing — run `make bench-micro` to generate");
    }
}

fn bench_step_one(c: &mut Criterion) {
    let mut g = c.benchmark_group("step_one");
    for &(p, a, coup) in GRID {
        let Some((model, params)) = try_load(p, a, coup) else { continue };
        let n_tr = model.model.transitions.len();
        let (init_int, init_real) = model.initial_state_mean(&params).unwrap();
        let mut scratch = StepScratch::new(&model);
        g.bench_function(BenchmarkId::from_parameter(label(p, a, coup)), |b| {
            b.iter_batched(
                || (init_int.counts.clone(), vec![0u64; n_tr], init_real.clone(), StatefulRng::new(42)),
                |(mut counts, mut flows, mut real, mut rng)| {
                    step_one(&model, &mut counts, &mut flows, &mut real, &params, 0.0, 1.0, None,
                             sim::rng::BinomialAlgorithm::default(), &mut rng, &mut scratch).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_load(c: &mut Criterion) {
    // Parse + resolve cost (the dominant share of forward-sim wall time at
    // scale). Only the coupling=on points, where the tree grows with P.
    let mut g = c.benchmark_group("load_parse_compile");
    g.sample_size(20);
    for &(p, a, coup) in GRID.iter().filter(|(_, _, c)| *c == "on") {
        let path = fixture_path(p, a, coup);
        let Ok(json) = std::fs::read_to_string(&path) else { continue };
        g.throughput(criterion::Throughput::Bytes(json.len() as u64));
        g.bench_function(BenchmarkId::from_parameter(label(p, a, coup)), |b| {
            b.iter(|| {
                let model: ir::Model = ir::from_str(&json).unwrap();
                CompiledModel::new(model).unwrap()
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_eval_propensities, bench_step_one, bench_load);
criterion_main!(benches);
