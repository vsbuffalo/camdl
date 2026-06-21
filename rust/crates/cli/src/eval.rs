//! `camdl eval` — evaluate time-dependent expressions without simulation.
//!
//! Evaluates forcing functions, let bindings, and inline expressions at a
//! time grid. No compartment state, no RNG, no trajectories. Useful for
//! inspecting covariates, forcing curves, and parameter-derived quantities.

use sim::{
    compiled_model::CompiledModel,
    propensity::{eval_expr, EvalCtx},
    state::{IntState, RealState},
};
use ir::expr::Expr;

/// Resolve a named expression from the compiled model's IR.
/// Checks: forcing functions, then let bindings (already inlined into
/// transition rates — we look for them in the original Model's IR).
/// Returns the Expr and whether it references compartment state.
fn resolve_named_expr(model: &ir::Model, compiled: &CompiledModel, name: &str) -> Result<Expr, String> {
    // 1. Forcing function?
    if compiled.time_func_index.contains_key(name) {
        return Ok(Expr::TimeFunc(ir::expr::TimeFuncWrap {
            time_func: ir::expr::TimeFuncRef { name: name.into() },
        }));
    }

    // 2. Parameter?
    if compiled.param_index.contains_key(name) {
        return Ok(Expr::Param(ir::expr::ParamExpr { param: name.into() }));
    }

    // 3. Scan transitions and let bindings for named sub-expressions.
    // The OCaml expander inlines let bindings, so they don't survive as
    // named entities in the IR. But for simple parameter-only expressions,
    // we can try evaluating the name as a parameter expression.
    // For now: if not found, try parsing as an inline expression.
    // (Inline expression parsing is a future extension.)

    // 4. Check if it's a compartment (error with helpful message)
    if compiled.comp_index.contains_key(name) {
        return Err(format!(
            "expression '{}' references compartment state.\n  \
             Compartment values require a running simulation.\n  \
             Use 'camdl simulate --trace' instead.", name
        ));
    }

    // Not found — check if it matches a table
    if compiled.table_index.contains_key(name) {
        return Err(format!(
            "'{}' is a table, not a scalar expression. Tables require index arguments.", name
        ));
    }

    Err(format!(
        "unknown expression '{}'. Available forcing functions: {:?}, parameters: {:?}",
        name,
        model.time_functions.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
        model.parameters.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
    ))
}

/// Check if an Expr references compartment state (Pop or PopSum nodes).
fn references_compartments(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Pop(w) => Some(w.pop.clone()),
        Expr::PopSum(w) => Some(w.pop_sum.first().cloned().unwrap_or_default()),
        Expr::BinOp(w) => references_compartments(&w.bin_op.left)
            .or_else(|| references_compartments(&w.bin_op.right)),
        Expr::UnOp(w) => references_compartments(&w.un_op.arg),
        Expr::Cond(w) => references_compartments(&w.cond.pred)
            .or_else(|| references_compartments(&w.cond.then))
            .or_else(|| references_compartments(&w.cond.else_)),
        Expr::Reduce(w) => w.reduce.iter().find_map(|t| references_compartments(t)),
        // Hoisted bindings are state-derived → effectively reference compartments.
        Expr::BindingRef(w) => Some(w.binding_ref.clone()),
        _ => None,
    }
}

pub fn cmd_eval(a: &crate::args::EvalArgs) {
    let ir_path = a.model.to_string_lossy();
    let at_points: Option<Vec<f64>> = if a.at.is_empty() { None } else { Some(a.at.clone()) };

    let (model_in, _model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // Unified parameter resolver (2026-05-25 CLI UX rev 2). Maps the
    // legacy `--params FILE` (repeatable) → `fixed_files`, and
    // `--param NAME=VALUE` → `fixed_cli`. `eval` has no scenario, no
    // [estimate] set, and no external tables (the eval pipeline
    // doesn't compile tables independently).
    use crate::params_resolver::{ParameterInputs, resolve_parameters, print_warnings};
    use indexmap::{IndexMap, IndexSet};
    use std::collections::HashMap;

    let fixed_cli: Vec<(String, f64)> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value)).collect();
    let fixed_files: Vec<std::path::PathBuf> = a.model_overrides.params.clone();
    let table_files: HashMap<String, std::path::PathBuf> = a.model_overrides.table.iter()
        .map(|t| (t.name.clone(), t.path.clone())).collect();
    let ftf: IndexMap<String, f64> = IndexMap::new();
    let fte: IndexSet<String> = IndexSet::new();

    let resolved = resolve_parameters(ParameterInputs {
        model: &model_in,
        scenario: None,
        adhoc_enable: &[],
        adhoc_disable: &[],
        fixed_cli: &fixed_cli,
        fixed_files: &fixed_files,
        fit_toml_fixed: &ftf,
        fit_toml_estimate: &fte,
        table_files: &table_files,
    }).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    print_warnings(&resolved);
    if log::log_enabled!(log::Level::Debug) {
        for p in &resolved.params {
            log::debug!("eval: param `{}` = {} ({})",
                p.name, p.value, p.source.tag());
        }
        log::debug!("eval: {} parameter(s) in [estimate]: {:?}",
            resolved.estimate_set.len(),
            resolved.estimate_set.iter().collect::<Vec<_>>());
    }
    let model = resolved.model.clone();
    // Touch role so the variant set is exhaustive — every consumer
    // must handle Fixed/Estimated. For eval (non-inference) every
    // param is `Fixed`; the assertion encodes that invariant.
    for p in &resolved.params {
        debug_assert!(matches!(p.role,
            crate::params_resolver::ParameterRole::Fixed { .. }),
            "eval is non-inference; every param must resolve to Fixed");
    }

    let compiled = CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("compile error: {:?}", e); std::process::exit(1); });
    let params = compiled.default_params.clone();

    let mut resolved: Vec<(String, Expr)> = Vec::new();
    for name in &a.expr {
        match resolve_named_expr(&model, &compiled, name) {
            Ok(expr) => {
                if let Some(comp) = references_compartments(&expr) {
                    eprintln!("error: expression '{}' references compartment '{}'.", name, comp);
                    eprintln!("  Compartment state requires a running simulation.");
                    eprintln!("  Use 'camdl simulate --trace' instead.");
                    std::process::exit(1);
                }
                resolved.push((name.clone(), expr));
            }
            Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
        }
    }

    let times: Vec<f64> = if let Some(pts) = at_points {
        pts
    } else {
        let mut ts = Vec::new();
        let mut t = a.from;
        while t <= a.to + 1e-9 {
            ts.push(t);
            t += a.every;
        }
        ts
    };

    let int_s = IntState::new(compiled.int_local_to_global.len());
    let real_s = RealState::new(compiled.real_local_to_global.len());

    use std::io::Write;
    let mut out: Box<dyn Write> = match &a.output {
        Some(path) => {
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path.display(), e); std::process::exit(1); });
            Box::new(std::io::BufWriter::new(f))
        }
        None => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
    };

    write!(out, "t").unwrap();
    for (name, _) in &resolved {
        write!(out, "\t{}", name).unwrap();
    }
    writeln!(out).unwrap();

    for &t in &times {
        write!(out, "{}", t).unwrap();
        let ctx = EvalCtx { model: &compiled, int_s: &int_s, real_s: &real_s, params: &params, t, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None };
        for (name, expr) in &resolved {
            match eval_expr(expr, &ctx) {
                Ok(val) => write!(out, "\t{:.6}", val).unwrap(),
                Err(e) => {
                    eprintln!("error evaluating '{}' at t={}: {:?}", name, t, e);
                    std::process::exit(1);
                }
            }
        }
        writeln!(out).unwrap();
    }
    drop(out);

    if let Some(ref path) = a.output {
        eprintln!("eval written to {}", path.display());
    }
}
