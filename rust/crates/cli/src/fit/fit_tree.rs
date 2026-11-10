//! Canonical fit-directory walker.
//!
//! One module, two functions, used by every consumer that needs to
//! enumerate fit-method runs:
//!
//! - [`walk_fit_dir`] — given one fit_dir (`results/fits/<stem>-<hash>/`)
//!   returns one [`MethodNode`] per `FitMethod` `runid::RunRecord` leaf found
//!   underneath, each carrying the leaf's [`FitMethodView`] (method, seed,
//!   best_loglik, …) and its path-derived axes.
//! - [`walk_fits_root`] — given the top-level `results/fits/` returns one
//!   [`FitDirEntry`] per fit segment, each carrying its already-parsed
//!   [`FitView`] (fit-level identity + provenance + method views) so callers
//!   filter by model_identity / label / declared methods without a second read.
//!
//! `MethodNode`/`FitMethodView` are method-agnostic by construction: they carry
//! no `fit_state_path` (an IF2-only artifact). Consumers that need the typed
//! result load it via `MethodResult::load_from`.

use std::io;
use std::path::{Path, PathBuf};

use crate::fit::fit_view::{FitMethodView, FitView};

/// One discovered fit-method run inside a fit_dir.
///
/// `axes` (on the method view) carries the (data_kind, fit_seed, sweep_slug)
/// triple extracted from the path relative to the enclosing fit_dir. When the
/// path doesn't fit the canonical v2 layout, `axes` is `None` — the method is
/// still surfaced (its leaf is real) but consumers that need axis grouping skip
/// it.
#[derive(Debug, Clone)]
pub struct MethodNode {
    /// Absolute (or whatever the caller passed) path to the method directory
    /// containing the leaf `run.json`.
    pub method_dir: PathBuf,
    /// The method's projected new-format view.
    pub view: FitMethodView,
}

/// One fit segment returned by [`walk_fits_root`]. The `view` field is
/// pre-parsed so `fit table`'s outer loop can filter by model_identity / label /
/// declared methods without re-reading the leaves per row.
#[derive(Debug, Clone)]
pub struct FitDirEntry {
    pub fit_dir: PathBuf,
    /// The fit-level view (identity + provenance + per-method views).
    pub view: FitView,
}

/// Walk one fit directory and return every `FitMethod` leaf found underneath,
/// each as a [`MethodNode`]. Filesystem layout is intentionally not constrained
/// beyond "every method writes a `runid::RunRecord` leaf": this walker works on
/// real-data fits, synthetic-replicate fits, sweep cells, any combination, and
/// on user-defined non-canonical layouts.
///
/// Returns runs in lexicographic order on `method_dir` (deterministic for
/// tests). A segment with method leaves but no provenance sidecar is skipped
/// loudly (see [`FitView::read`]); the function then yields no nodes.
pub fn walk_fit_dir(fit_dir: &Path) -> io::Result<Vec<MethodNode>> {
    if !fit_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("fit_dir not found: {}", fit_dir.display()),
        ));
    }
    let mut out: Vec<MethodNode> = match FitView::read(fit_dir) {
        Some(view) => view
            .methods
            .into_iter()
            .map(|view| MethodNode {
                method_dir: view.method_dir.clone(),
                view,
            })
            .collect(),
        None => Vec::new(),
    };
    out.sort_by(|a, b| a.method_dir.cmp(&b.method_dir));
    Ok(out)
}

/// Walk `results/fits/` (or whatever root the caller passes) and return one
/// entry per fit segment. A child dir is a fit segment iff it holds `FitMethod`
/// leaves and a provenance sidecar ([`FitView::read`]); other dirs are skipped.
///
/// Returns entries in lexicographic order on `fit_dir`.
pub fn walk_fits_root(root: &Path) -> io::Result<Vec<FitDirEntry>> {
    let mut out = Vec::new();
    if !root.exists() {
        // Empty root → empty list. `walk_fit_dir` errors on a missing dir
        // because that's a single-fit lookup; this is a directory listing, and
        // "no fits yet" is a normal state.
        return Ok(out);
    }
    // gh#147: a CAS fit is a segment dir `fits/{stem}-{h8}/` holding its
    // method-leaf subdirs (`{method}-{h8}/seed_N-{h8}/run.json`, kind
    // FitMethod) plus the fit-level sidecar. `FitView::read` derives ONE
    // fit-level view per segment from those leaves — the same projection
    // `table_row::build_row` consumes, so the table row and its source entry
    // agree. A dir with no FitMethod leaves is not a fit segment and is skipped.
    let entries = std::fs::read_dir(root)?;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(view) = FitView::read(&p) {
            out.push(FitDirEntry {
                fit_dir: p.clone(),
                view,
            });
        }
    }
    out.sort_by(|a, b| a.fit_dir.cmp(&b.fit_dir));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_meta::FitAlgorithm;

    /// Allocate a unique tempdir for one test. Cleaned up by `Drop`.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_fittree_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    /// Write a fit-level sidecar (`fit.meta.json`) under a segment dir so
    /// `FitView::read` treats it as a well-formed fit. Minimal provenance —
    /// `methods_declared`/per-method numbers come from the leaves.
    fn write_sidecar(seg: &Path) {
        std::fs::create_dir_all(seg).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_path":"sir.camdl","model_identity":"f00d","schema_version":2,"source_config":"fit.toml"}"#,
        )
        .unwrap();
    }

    /// Place a `FitMethod` `runid::RunRecord` leaf at `<fit_dir>/<rel>/run.json`.
    /// `ord` seeds a valid-hex run_id; `method`/`method`/`seed` go in `inputs`.
    fn place_method(fit_dir: &Path, rel: &str, ord: u32, name: &str, method: &str, seed: u64) {
        let dir = fit_dir.join(rel);
        std::fs::create_dir_all(&dir).unwrap();
        let run_id = format!("{:0<64}", format!("ae{ord:06x}"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_method","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"deadbeef00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"method","label":"{name}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_{seed}","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"{method}","backend":"chain_binomial","seed":{seed},"n_chains":2,"best_loglik":-100.0,"best_chain":1}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(dir.join("run.json"), rec).unwrap();
    }

    /// gh#147: write a CAS fit segment at `seg` — one FitMethod leaf
    /// (`if2-<h8>/seed_1-<h8>/run.json`) plus the fit-level sidecar — the
    /// shape `walk_fits_root` reads. `fit_h8` (8 hex) seeds the shared
    /// `fit`-level hash so each segment gets a distinct fit hash.
    fn write_cas_fit_seg(seg: &Path, fit_h8: &str) {
        let leaf = seg.join("if2-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let run_id = format!("{:0<64}", format!("{fit_h8}01"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_method","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"{fit_hash}","schema_version":1}},{{"name":"method","label":"if2","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(leaf.join("run.json"), rec).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_identity":"f00d","model_path":"sir.camdl","schema_version":2,"source_config":"fit.toml"}"#,
        )
        .unwrap();
    }

    #[test]
    fn walks_real_only_single_fit() {
        let tmp = tempdir("real_only");
        let fit_dir = tmp.path().join("fit_he-deadbeef");
        write_sidecar(&fit_dir);
        place_method(&fit_dir, "real/fit_1/if2", 1, "if2", "if2", 1);
        place_method(&fit_dir, "real/fit_1/pgas", 2, "pgas", "pgas", 1);

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let method_names: Vec<&str> = nodes.iter().map(|n| n.view.method.as_str()).collect();
        assert!(method_names.contains(&"if2"));
        assert!(method_names.contains(&"pgas"));
    }

    #[test]
    fn walks_synthetic_layout() {
        let tmp = tempdir("synthetic");
        let fit_dir = tmp.path().join("fit_syn-cafebabe");
        write_sidecar(&fit_dir);
        let mut ord = 0;
        for ds in 1..=2usize {
            for fs in [11u64, 22] {
                ord += 1;
                let rel = format!("synthetic/ds_{:02}/fit_{}/mle", ds, fs);
                place_method(&fit_dir, &rel, ord, "mle", "if2", fs);
            }
        }
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 4, "2 datasets × 2 seeds = 4 nodes");
    }

    #[test]
    fn walks_sweep_slug_layout() {
        let tmp = tempdir("sweep");
        let fit_dir = tmp.path().join("fit_sweep-12345678");
        write_sidecar(&fit_dir);
        place_method(&fit_dir, "real/fit_1/R0_1.000/mle", 1, "mle", "if2", 1);
        place_method(&fit_dir, "real/fit_1/R0_2.000/mle", 2, "mle", "if2", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn walks_mixed_method_methods() {
        let tmp = tempdir("mixed");
        let fit_dir = tmp.path().join("fit_mixed-aaaabbbb");
        write_sidecar(&fit_dir);
        place_method(&fit_dir, "real/fit_1/scout", 1, "scout", "if2", 1);
        place_method(&fit_dir, "real/fit_1/pgas", 2, "pgas", "pgas", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let methods: Vec<&str> = nodes.iter().map(|n| n.view.method.as_str()).collect();
        assert!(methods.contains(&"if2"));
        assert!(methods.contains(&"pgas"));
    }

    #[test]
    fn skips_malformed_run_json_does_not_panic() {
        let tmp = tempdir("malformed");
        let fit_dir = tmp.path().join("fit_bad-deadbeef");
        write_sidecar(&fit_dir);
        let method_dir = fit_dir.join("real/fit_1/scout");
        std::fs::create_dir_all(&method_dir).unwrap();
        // Method run.json is garbage — `walk_records` drops it.
        std::fs::write(method_dir.join("run.json"), "{ not valid json }").unwrap();

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert!(
            nodes.is_empty(),
            "malformed method run.json should be skipped, got {:?}",
            nodes
        );
    }

    #[test]
    fn missing_fit_dir_errors() {
        let err = walk_fit_dir(Path::new("/definitely/does/not/exist/here")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn surfaces_non_canonical_layout() {
        // A user might park a fit-method leaf under a one-off path (e.g. for a
        // debugging copy). The walker should still surface it.
        let tmp = tempdir("noncanon");
        let fit_dir = tmp.path().join("fit_x-aabbccdd");
        write_sidecar(&fit_dir);
        place_method(&fit_dir, "weird_layout/scout", 1, "scout", "if2", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn walks_fits_root_returns_one_entry_per_fit() {
        let tmp = tempdir("root");
        let fits_root = tmp.path().join("fits");
        std::fs::create_dir_all(&fits_root).unwrap();

        let dirs = ["fit_a-11111111", "fit_b-22222222"];
        for name in dirs.iter() {
            let d = fits_root.join(name);
            // gh#147: a CAS fit segment (one FitMethod leaf + the fit-level
            // sidecar) — the shape `walk_fits_root`/`FitView::read` reads.
            let h8 = name.split('-').next_back().unwrap();
            write_cas_fit_seg(&d, h8);
        }
        // A dir with no FitMethod leaves (and no sidecar) — should be skipped.
        let intruder = fits_root.join("not_a_fit");
        std::fs::create_dir_all(&intruder).unwrap();
        std::fs::write(intruder.join("README"), b"not a fit").unwrap();

        let entries = walk_fits_root(&fits_root).unwrap();
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e.fit_dir.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["fit_a-11111111", "fit_b-22222222"]);
        // The view is pre-parsed (derived from the method leaves); `methods_declared`
        // comes from the leaves' method levels. gh#147: `estimated`/`fixed`/priors
        // are config detail not carried on the leaves, so they default empty —
        // config-diff reads the archived `fit.toml.original` for those.
        assert_eq!(entries[0].view.methods_declared, vec!["if2"]);
        assert!(!entries[0].view.fit_hash.is_empty());
        assert_eq!(entries[0].view.methods.len(), 1);
    }

    #[test]
    fn walks_fits_root_empty_when_no_root() {
        let entries = walk_fits_root(Path::new("/definitely/not/here/either/fits")).unwrap();
        assert!(entries.is_empty());
    }

    // Keep `FitAlgorithm` exercised so the import isn't unused in the
    // method-agnostic test set above.
    #[test]
    fn method_kind_str_roundtrip() {
        assert_eq!(FitAlgorithm::If2.as_str(), "if2");
        assert_eq!(FitAlgorithm::Pgas.as_str(), "pgas");
    }
}
