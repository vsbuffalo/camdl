//! The engine's developer switches for a `sim` test run directly with
//! `cargo test -p sim`.
//!
//! The library reads no environment variable: `camdl` reads them once
//! (`cli::env`) and passes them in. A test that runs the engine without the
//! binary calls [`from_env`] first and compiles its models under the
//! evaluator settings it returns, so that the oracles the notes describe still
//! work there:
//!
//! - `CAMDL_EVAL_UNRESOLVED=1 cargo test -p sim --test gate_trajectory_baseline`
//!   routes every rate through the string-keyed evaluator (the byte-identity
//!   oracle of `docs/dev/notes/2026-06-01-pf-eval-profiling.md`);
//! - `CAMDL_NO_BINDING_CACHE=1` turns the binding cache off;
//! - `CAMDL_TRACE_STEPS=1` traces every chain-binomial substep to stderr.
//!
//! Test code, so the "environment is read in one module" rule
//! (`cli/tests/environment_is_read_in_one_module.rs`) does not cover it. The
//! names and their reading rule (`1` for the trace, any value for the other
//! two) are `cli::env`'s.

/// Configure `sim::trace` from the test process's environment, and return the
/// evaluator switches it sets (`CAMDL_EVAL_UNRESOLVED`,
/// `CAMDL_NO_BINDING_CACHE`) as the settings a test compiles its models under,
/// every other setting at its default. Idempotent; call it at the start of
/// each test that runs the engine.
pub fn from_env() -> sim::EvalSettings {
    sim::trace::configure(sim::trace::Switches {
        steps: std::env::var("CAMDL_TRACE_STEPS").is_ok_and(|v| v == "1"),
    });
    sim::EvalSettings {
        eval_unresolved: std::env::var_os("CAMDL_EVAL_UNRESOLVED").is_some(),
        no_binding_cache: std::env::var_os("CAMDL_NO_BINDING_CACHE").is_some(),
        ..Default::default()
    }
}
