//! The B-spline validation harness (proposal 2026-10-27-estimable-bsplines
//! §8.1): R4 regression checks, plus a named placeholder for every R1–R3 item
//! that needs the `smooth` coefficient form (the curvature penalty Ω, its
//! whitening, and its prior), which is not built yet.
//!
//! The rest of the harness lives beside the code it checks, and `make
//! validate-bspline` runs all of it, the heavy runs included:
//!
//! - R1 basis values against scipy and R `splines` (clamped uniform, uniform at
//!   absolute time, irregular; degrees 0–5; ends, every breakpoint, between,
//!   outside) and periodic degrees 0–5: `bspline_reference_fixtures.rs`;
//!   against the replaced recursion: `bspline_recursion_oracle.rs`; partition
//!   of unity, non-negativity, end interpolation, constants and the line
//!   reproduced, the Bernstein anchor: `sim::bspline::tests`.
//! - R2 gradients against finite differences, forcing level and the
//!   complete-data likelihood: `bspline_gradient_check.rs`.
//! - R3 PGAS+NUTS integration and recovery: `bspline_inference.rs`.
//! - R4 the `link` no-collision hash: `runid` `ir_spline_link_changes_hash`;
//!   round-trip and reproducibility: here.
//!
//! The placeholders are `#[ignore]`d and fail if run: they are not passing
//! checks, they mark work that does not exist yet.

use std::path::PathBuf;

use sim::compiled_model::CompiledModel;
use sim::resolved_expr::{eval_resolved, ResolvedDerivEntry, ResolvedExpr};

const SPLINE_GOLDENS: [&str; 5] = [
    "bspline_uniform",
    "bspline_explicit_dated",
    "bspline_loglink",
    "bspline_param_coefs",
    "periodic_spline_param_coefs",
];

fn golden(name: &str) -> String {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../../ocaml/golden", &format!("{name}.ir.json")].iter().collect();
    std::fs::read_to_string(p).unwrap()
}

/// R4: the OCaml compiler's JSON for every spline golden parses in Rust, and
/// Rust writes back the same JSON value for every part a spline reaches (the
/// forcing's kind, link, knots and coefficients; each rate and its gradients,
/// with their `forcing_basis` nodes; the observations); and a second Rust
/// round trip changes nothing.
#[test]
fn spline_goldens_round_trip_ocaml_json_rust_json() {
    for name in SPLINE_GOLDENS {
        let ocaml_text = golden(name);
        let ocaml: serde_json::Value = serde_json::from_str(&ocaml_text).unwrap();
        let model = ir::from_str(&ocaml_text).unwrap_or_else(|e| panic!("{name}: Rust cannot parse it: {e}"));
        let rust = serde_json::to_value(&model).unwrap();
        // The two serializers differ elsewhere only in null-versus-absent
        // defaults unrelated to splines (`param_dim`, scenario defaults), so the
        // comparison is over every part a spline reaches.
        for key in ["time_functions", "observations"] {
            assert_eq!(ocaml["model"][key], rust[key], "{name}: `{key}` differs between OCaml and Rust JSON");
        }
        for (o, r) in ocaml["model"]["transitions"].as_array().unwrap().iter()
            .zip(rust["transitions"].as_array().unwrap())
        {
            for key in ["rate", "rate_grad", "rate_state_grad"] {
                assert_eq!(o[key], r[key], "{name}: transition `{}` `{key}` differs", o["name"]);
            }
        }
        // Rust's own round trip. Compared as JSON values, not bytes: the
        // gradient maps are `HashMap`s, so their key order in the text is not
        // stable across instances (general to the IR, not to splines).
        let back: ir::Model = serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
        assert_eq!(rust, serde_json::to_value(&back).unwrap(), "{name}: Rust JSON changed across a round trip");
        assert_eq!(model, back, "{name}: the model changed across a Rust round trip");
    }
}

/// R4: building a spline model twice gives bitwise-identical forcing values
/// and coefficient gradients (the knots, and everything derived from them, are
/// a deterministic function of the IR).
#[test]
fn spline_models_build_reproducibly() {
    for name in SPLINE_GOLDENS {
        let build = || {
            let mut m = ir::from_str(&golden(name)).unwrap();
            if let Some(preset) = m.presets.first().cloned() {
                for p in &mut m.parameters {
                    if let Some(&v) = preset.params.get(&p.name) {
                        p.value = p.value.with_value(v);
                    }
                }
            }
            CompiledModel::new(m).unwrap()
        };
        let (a, b) = (build(), build());
        let bits = |model: &CompiledModel| -> Vec<u64> {
            let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
            let mut out = Vec::new();
            for i in 0..=400 {
                let t = -20.0 + 800.0 * i as f64 / 400.0;
                let ctx = sim::propensity::EvalCtx {
                    model, int_s: &int_s, real_s: &real_s, params: &model.default_params,
                    t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
                };
                for tf in 0..model.time_func_cache.len() {
                    out.push(eval_resolved(&ResolvedExpr::TimeFunc(tf), &ctx).to_bits());
                }
                for grads in &model.resolved.rate_grads_indexed {
                    for (_, g) in grads {
                        if let ResolvedDerivEntry::Grad(e) = g {
                            out.push(eval_resolved(e, &ctx).to_bits());
                        }
                    }
                }
            }
            out
        };
        assert_eq!(bits(&a), bits(&b), "{name}: two builds disagree");
    }
}

// ── Placeholders: items that need the `smooth` coefficient form ────────────
// Each names the proposal item it stands for. They are ignored, and they fail
// if run, so none can be mistaken for a passing check.

#[test]
#[ignore = "R1, needs the curvature penalty Ω (`smooth`, sim/src/spline_penalty.rs): Ω by 2-point Gauss–Legendre against 10-point and against R fda::bsplinepen, ≤ 1e-12·max|Ω|; bandwidth p; rank K − 2"]
fn r1_curvature_penalty_against_references() {
    todo!("R1 penalty Ω: not built")
}

#[test]
#[ignore = "R1, needs Ω (`smooth`): a line has zero curvature penalty — 1ᵀΩ1 and ξᵀΩξ ≤ 1e-12·max|Ω|·(b − a)² for uniform and irregular knots"]
fn r1_a_line_has_zero_curvature_penalty() {
    todo!("R1 null space of Ω: not built")
}

#[test]
#[ignore = "R1, needs the whitening M (`smooth`): MᵀΩM = I to 1e-12; ‖NᵀM‖_max ≤ 1e-12·‖N‖₂‖M‖₂; τ²MMᵀ = τ²Ω⁺ to 1e-10; the induced density equals the improper Gaussian at 100 points"]
fn r1_whitening_identities() {
    todo!("R1 whitening: not built")
}

#[test]
#[ignore = "R2, needs `smooth`: the penalty ½wᵀΩw (gradient Ωw) and the smooth map w(a, b, τ, z) against finite differences"]
fn r2_smooth_gradients() {
    todo!("R2 for smooth: not built")
}

#[test]
#[ignore = "R3, needs the `smooth` prior: prior-only PGAS+NUTS on X --> D @ f(t)·X with X0 = 0, 4 chains × 4000 sweeps; τ ~ exponential(λ) by KS, a and b their normals, cov(w − a·1 − b(ξ − t_m)) = (2/λ²κ²)Ω⁺, |z| < 5 on batch-means MCSE"]
fn r3_prior_only_run_draws_the_smooth_prior() {
    todo!("R3 prior-only run: needs smooth")
}

#[test]
#[ignore = "R4, needs `smooth`: the whitening M is bitwise identical for a fixed model across two builds"]
fn r4_whitening_is_reproducible() {
    todo!("R4 M reproducibility: not built")
}
