//! Developer traces that an environment variable switches on and that must
//! reach stderr from inside the engine.
//!
//! The library otherwise never prints: it returns values and records findings,
//! and the command-line crate decides what a user sees (proposal
//! `docs/dev/proposals/2026-11-08-cli-reporting.md`, "The reporting
//! interface"). These traces are the exception because they are not a report
//! of a run: they are a firehose a developer switches on to watch the engine
//! step, emitted at the point the value exists, with no effect on results. Each
//! is one function here, which reads its variable once and writes to stderr.
//!
//! - `CAMDL_TRACE_STEPS=1`: one TSV row per chain-binomial substep
//!   ([`step`], with a header row before the first) and one line per
//!   intervention or event action ([`effect_action`]).
//! - `CAMDL_EVAL_UNRESOLVED` (any value): the propensity evaluator takes the
//!   slow string-keyed path ([`eval_unresolved`]), and says so once.

use std::sync::OnceLock;

use ir::intervention::Action;

use crate::compiled_model::CompiledModel;
use crate::effects::EffectKind;

/// Whether `CAMDL_TRACE_STEPS=1` is set. Read once and cached.
pub fn steps_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CAMDL_TRACE_STEPS").is_ok_and(|v| v == "1"))
}

/// Under `CAMDL_TRACE_STEPS=1`, one row for the chain-binomial substep that
/// ended at `t`: the time, each compartment's count, each transition's flow,
/// the total population, and each transition's propensity. The first call
/// writes the header row (`t`, compartments, `flow_<name>`, `total_pop`,
/// `rate_<name>`) before its row.
pub fn step(model: &CompiledModel, t: f64, counts: &[i64], flows: &[u64], propensities: &[f64]) {
    if !steps_enabled() {
        return;
    }
    static HEADER: OnceLock<bool> = OnceLock::new();
    HEADER.get_or_init(|| {
        eprint!("t");
        for c in &model.model.compartments { eprint!("\t{}", c.name); }
        for tr in &model.model.transitions { eprint!("\tflow_{}", tr.name); }
        eprint!("\ttotal_pop");
        for tr in model.model.transitions.iter() {
            eprint!("\trate_{}", tr.name);
        }
        eprintln!();
        true
    });
    eprint!("{:.1}", t);
    for &c in counts.iter() { eprint!("\t{}", c); }
    for &f in flows.iter() { eprint!("\t{}", f); }
    let total: i64 = counts.iter().sum();
    eprint!("\t{}", total);
    for &p in propensities.iter() { eprint!("\t{:.4}", p); }
    eprintln!();
}

/// Under `CAMDL_TRACE_STEPS=1`, one line for an intervention or event action
/// that fired at `t` with resolved value `v`.
pub fn effect_action(kind: EffectKind, iv_name: &str, action: &Action, v: f64, t: f64) {
    if !steps_enabled() {
        return;
    }
    let k = kind.label();
    match action {
        Action::Add(a) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: add {} += {} (raw={v:.2})",
            a.compartment, v.round() as i64
        ),
        Action::Set(a) => {
            eprintln!("{k} '{iv_name}' at t={t:.1}: set {} = {v:.2}", a.compartment)
        }
        Action::FractionTransfer(ft) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: transfer {} -> {} (frac={:.2})",
            ft.src, ft.dst, v.clamp(0.0, 1.0)
        ),
        Action::AbsoluteTransfer(at) => eprintln!(
            "{k} '{iv_name}' at t={t:.1}: transfer {} -> {} (raw={v:.2})",
            at.src, at.dst
        ),
    }
}

/// Bench / differential-validation switch: route the propensity hot path
/// through the slow string-keyed `eval_expr` instead of the pre-resolved
/// `eval_resolved`. Enabled by setting the env var `CAMDL_EVAL_UNRESOLVED`
/// (any value). Read once and cached, so it costs an atomic load — hoisted
/// out of the per-transition loop in `eval_propensities`, so the default
/// (off) path is unchanged. The first read says on stderr that it is on.
///
/// Two uses:
///  - **benching**: time a `pfilter`/`fit` run with the var off vs on to
///    measure end-to-end what pre-resolution buys (`T_on / T_off`);
///  - **validation**: run any sim/inference under both evaluators and assert
///    byte-identical output — a differential-testing oracle for the resolver,
///    which is inference-critical.
///
/// `eval_resolved` and `eval_expr` are required to agree (see
/// `tests/resolved_expr.rs`); this switch makes that agreement observable on
/// real models at full scale, not just hand-built expressions.
#[inline]
pub fn eval_unresolved() -> bool {
    static EVAL_UNRESOLVED: OnceLock<bool> = OnceLock::new();
    *EVAL_UNRESOLVED.get_or_init(|| {
        let on = std::env::var_os("CAMDL_EVAL_UNRESOLVED").is_some();
        if on {
            eprintln!(
                "[camdl] CAMDL_EVAL_UNRESOLVED set — propensity eval routed through \
                 the slow string-keyed eval_expr (bench/validation mode)"
            );
        }
        on
    })
}
