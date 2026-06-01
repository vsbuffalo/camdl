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

use std::path::{Path, PathBuf};

use crate::hashing::slug;

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

/// Relative path under `<root>/sims/` for a legacy simulate run path
/// (`<stem>-<sim_hash[:8]>/<scenario-slug>-<scen_hash[:8]>/seed_<N>`).
///
/// M3-DELETION-BOUND (gh#147): the new CAS path is the factored
/// `runid::store_path` (5 levels, `{label}-{hash8}` each). This legacy
/// helper survives only for `batch status`/`batch design` (not yet
/// migrated); delete it when those paths move to `runid::store_path`.
pub fn sim_run_rel(
    model_stem: Option<&str>,
    sim_hash: &str,
    scenario: &str,
    scen_hash: &str,
    seed: u64,
) -> String {
    let hash_prefix = &sim_hash[..8.min(sim_hash.len())];
    let head = match model_stem {
        Some(s) if !s.is_empty() => format!("{}-{}", s, hash_prefix),
        _ => hash_prefix.to_string(),
    };
    format!("{}/{}-{}/seed_{}", head, slug(scenario), &scen_hash[..8.min(scen_hash.len())], seed)
}

/// Top-level directory for a fit.
///
/// With `fit_toml_stem = None`:       `<root>/fits/<fit_hash[:8]>/`
/// With `fit_toml_stem = Some("01")`: `<root>/fits/01-<fit_hash[:8]>/`
///
/// The stem (basename of fit.toml, slugified) is what users see in
/// `ls output/fits/`. The content hash follows as a suffix so two
/// configs that share a name but differ in content get distinct dirs
/// — the hash is the authoritative key.
pub fn fit_run_dir(root: &Path, fit_toml_stem: Option<&str>, fit_hash: &str) -> PathBuf {
    let hash_prefix = &fit_hash[..8.min(fit_hash.len())];
    let dirname = match fit_toml_stem {
        Some(s) if !s.is_empty() => format!("{}-{}", s, hash_prefix),
        _ => hash_prefix.to_string(),
    };
    root.join("fits").join(dirname)
}

// ─── profile layout ──────────────────────────────────────────────────────────
//
// The profile-run-root directory is now produced by `ProfileInputs::cas_path`
// in `cas::typed` (see profile.rs); the function lives there so the typed
// inputs and the path layout stay in one place. The grid-point and start
// helpers below are still hand-rolled — they're a layout convention inside
// a profile run, not a CAS root.

/// Directory for one grid point within a profile:
///
/// ```text
/// <profile_dir>/points/{point_idx:05d}/
/// ```
///
/// Flat-indexed over the Cartesian product of focal axes. The
/// `focal.toml` inside this directory disambiguates which coordinate
/// this point represents without consumers having to parse back
/// from `point_idx` to axis values.
pub fn profile_point_dir(profile_dir: &Path, point_idx: usize) -> PathBuf {
    profile_dir.join("points").join(format!("{:05}", point_idx))
}

/// Directory for one (grid point, start) mini-fit:
///
/// ```text
/// <profile_dir>/points/{point_idx:05d}/start_{start_idx}/
/// ```
///
/// This directory holds a `RunKind::FitStage` run.json — each start is
/// independently cacheable, so a crash that kills one start leaves the
/// others intact and resumable.
pub fn profile_point_start_dir(
    profile_dir: &Path,
    point_idx: usize,
    start_idx: usize,
) -> PathBuf {
    profile_point_dir(profile_dir, point_idx)
        .join(format!("start_{}", start_idx))
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

    #[test]
    fn fit_run_dir_hash_only() {
        let p = fit_run_dir(Path::new("/out"), None, "deadbeef00000000");
        assert_eq!(p, Path::new("/out/fits/deadbeef"));
    }

    #[test]
    fn fit_run_dir_with_stem() {
        let p = fit_run_dir(Path::new("/out"), Some("01"), "deadbeef00000000");
        assert_eq!(p, Path::new("/out/fits/01-deadbeef"));
    }

    #[test]
    fn fit_run_dir_same_stem_different_hash_diverges() {
        // The whole point of the <stem>-<hash[:8]> scheme: two fits
        // with the same human-readable name but different content
        // must land in different directories. Hash is the key, stem
        // is cosmetic.
        let a = fit_run_dir(Path::new("/out"), Some("01"), "aaaaaaaa1111111111");
        let b = fit_run_dir(Path::new("/out"), Some("01"), "bbbbbbbb2222222222");
        assert_ne!(a, b, "same stem, different hash must produce different dirs");
    }

    #[test]
    fn sim_run_rel_same_stem_different_hash_diverges() {
        let a = sim_run_rel(Some("sir"), "aaaaaaaa0000", "baseline", "cccc", 1);
        let b = sim_run_rel(Some("sir"), "bbbbbbbb0000", "baseline", "cccc", 1);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_prefix_tolerates_short_hashes() {
        // Defensive: helpers slice [..8] on the hash. If a caller
        // accidentally passes a short hash, the slice should still
        // work (not panic) so we don't turn a hash-bug into a
        // crash-bug.
        let p = fit_run_dir(Path::new("/out"), None, "abc");
        assert_eq!(p, Path::new("/out/fits/abc"));
    }

    #[test]
    fn profile_grid_point_layout() {
        // The profile-root directory is now produced by ProfileInputs;
        // here we just verify the layout convention inside a profile
        // root holds.
        let root = Path::new("/out/profiles/fit_r0-aaaaaaaa");
        let pt = profile_point_dir(root, 42);
        assert_eq!(pt, Path::new("/out/profiles/fit_r0-aaaaaaaa/points/00042"));
        let st = profile_point_start_dir(root, 42, 2);
        assert_eq!(st, Path::new("/out/profiles/fit_r0-aaaaaaaa/points/00042/start_2"));
    }

    #[test]
    fn profile_point_zero_padded() {
        // Grid sizes up to 99,999 points produce sortable ls output.
        // Larger grids (>100k) fall back to non-padded width but still
        // sort lexicographically within their own width class.
        let root = Path::new("/r/profiles/0000-00000000");
        assert!(profile_point_dir(root, 0)
            .to_str().unwrap().ends_with("00000"));
        assert!(profile_point_dir(root, 42)
            .to_str().unwrap().ends_with("00042"));
        assert!(profile_point_dir(root, 99999)
            .to_str().unwrap().ends_with("99999"));
    }
}
