//! Canonical output-path construction for the unified tree.
//!
//! Every filesystem path that camdl writes results into goes through
//! one of the helpers in this module. Centralising construction
//! prevents the `format!("{}/refine", fit.fit.output_dir)`
//! scattered-hand-rolling that let the fit and simulate trees drift
//! apart historically.
//!
//! Introduced in commit 3/6 of the output-tree unification
//! (docs/dev/proposals/2026-04-19-unified-output-tree.md).

use std::path::PathBuf;

/// Default output root: `./output`. Overridden by explicit CLI
/// `--output-dir`, fit.toml `output_dir`, or batch.toml `output_dir`
/// — in that precedence order. Callers should resolve via
/// [`output_root`] so the three entry points can't drift.
/// Default output root. `results/` pairs with `data/` in the research-
/// workflow vocabulary the project's downstream users (book chapters,
/// vignettes) already speak. Was briefly `output/` during the
/// 2026-04-19 unification; reverted per the post-ship review's
/// naming argument — see hardening proposal §ship-now/#7.
pub const DEFAULT_OUTPUT_ROOT: &str = "results";

/// Resolve the output root from the three places a user can set it.
/// Resolve the output root with the project's standard precedence:
/// **CLI override > config-file value > `CAMDL_OUTPUT_DIR` env var
/// > default (`./results`)**.
///
/// The env-var layer makes this function consistent with the
/// reader-side commands (`list` / `show` / `cat`) and the writers
/// `simulate --cas` / `batch run`, all of which honor
/// `CAMDL_OUTPUT_DIR` via clap's `env = "..."` attribute. Before
/// this layer existed, `fit run` and `profile` silently ignored
/// the env var and fell straight through to `DEFAULT_OUTPUT_ROOT`,
/// which made `CAMDL_OUTPUT_DIR=/tmp/foo camdl ...` redirect some
/// commands but not others (audit at
/// `docs/dev/reviews/2026-04-27-output-tree-consistency-review.md`).
///
/// Why env *below* config: project-specific config (fit.toml's
/// `output_dir`, batch.toml's `output_dir`) is more authoritative
/// than ambient shell state. A user who sets `CAMDL_OUTPUT_DIR` to
/// keep dev runs separate shouldn't have it silently override an
/// explicit `output_dir = "results/he2010"` in a fit.toml.
pub fn output_root(cli: Option<&str>, config: Option<&str>) -> PathBuf {
    cli.map(PathBuf::from)
        .or_else(|| config.map(PathBuf::from))
        .or_else(|| std::env::var("CAMDL_OUTPUT_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_ROOT))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize env-touching tests in this file. `output_root` reads
    /// `CAMDL_OUTPUT_DIR`, so any test that asserts its return value
    /// must hold this lock to prevent another test setting the env
    /// concurrently. (Rust's test runner parallelizes by default;
    /// env state is process-global.)
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII helper: snapshot `CAMDL_OUTPUT_DIR` on construction,
    /// restore it on Drop. Use inside a test that wants to mutate
    /// the env var, to leave the process state clean for sibling
    /// tests after this one finishes (whether by success or panic).
    struct EnvSnapshot {
        prior: Option<String>,
    }
    impl EnvSnapshot {
        fn take_and_clear() -> Self {
            let prior = std::env::var("CAMDL_OUTPUT_DIR").ok();
            std::env::remove_var("CAMDL_OUTPUT_DIR");
            Self { prior }
        }
    }
    impl Drop for EnvSnapshot {
        fn drop(&mut self) {
            match self.prior.take() {
                Some(v) => std::env::set_var("CAMDL_OUTPUT_DIR", v),
                None => std::env::remove_var("CAMDL_OUTPUT_DIR"),
            }
        }
    }

    #[test]
    fn output_root_precedence_no_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _snap = EnvSnapshot::take_and_clear();

        // No env, no overrides: hits the default.
        assert_eq!(output_root(None, None), PathBuf::from(DEFAULT_OUTPUT_ROOT));
        assert_eq!(DEFAULT_OUTPUT_ROOT, "results",
            "default output root is 'results/' (pairs with 'data/')");
        // Config wins over default.
        assert_eq!(output_root(None, Some("results")), PathBuf::from("results"));
        // CLI wins over default.
        assert_eq!(output_root(Some("/tmp/abc"), None), PathBuf::from("/tmp/abc"));
        // CLI wins over config.
        assert_eq!(output_root(Some("/cli"), Some("/config")), PathBuf::from("/cli"));
    }

    #[test]
    fn output_root_uses_env_when_cli_and_config_absent() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _snap = EnvSnapshot::take_and_clear();

        std::env::set_var("CAMDL_OUTPUT_DIR", "/tmp/from-env");

        // Env fires when CLI + config are both absent (the regression
        // this layer was added to fix — `fit run` and `profile`
        // calling `output_root(None, ...)` previously fell straight
        // through to DEFAULT_OUTPUT_ROOT).
        assert_eq!(output_root(None, None), PathBuf::from("/tmp/from-env"));
    }

    #[test]
    fn output_root_config_beats_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _snap = EnvSnapshot::take_and_clear();

        std::env::set_var("CAMDL_OUTPUT_DIR", "/tmp/from-env");

        // Config (fit.toml's `output_dir`, batch.toml's manifest
        // setting) is project-specific and more authoritative than
        // ambient shell state. Setting CAMDL_OUTPUT_DIR must NOT
        // silently redirect a fit that explicitly declared
        // `output_dir = "..."` in its toml.
        assert_eq!(output_root(None, Some("results/he2010")),
            PathBuf::from("results/he2010"));
    }

    #[test]
    fn output_root_cli_beats_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _snap = EnvSnapshot::take_and_clear();

        std::env::set_var("CAMDL_OUTPUT_DIR", "/tmp/from-env");

        // Explicit CLI override sits above env.
        assert_eq!(output_root(Some("/cli/dir"), None), PathBuf::from("/cli/dir"));
        assert_eq!(output_root(Some("/cli/dir"), Some("/config")),
            PathBuf::from("/cli/dir"));
    }

}
