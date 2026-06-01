//! Typed CAS inputs for `fit run`.
//!
//! Two levels:
//!
//! - [`FitInputs`] — the umbrella that contains all stages. Its
//!   `content_hash` is the existing `fit_content_hash`
//!   (model IR + data files + fit.toml bytes, seed-free). Its
//!   `cas_path` is `<root>/fits/<stem>-<hash[:8]>/`.
//! - [`StageInputs`] — one fit stage (cell × stage). Its `content_hash`
//!   is the existing `fit_stage_hash` (fit content + stage config + seed,
//!   seed-inclusive). Its `cas_path` is the runner-computed
//!   `<fit>/{cell}/{stage_name}/` directory; the trait's `root` argument
//!   is ignored because cell layout (real / synthetic ds_NN /
//!   fit_<seed>) is too rich to derive from `<root>` alone.
//!
//! Both wrap the existing hashing helpers (`fit_content_hash`,
//! `fit_stage_hash`) — those continue to be the load-bearing
//! implementations; the trait gives a uniform consumer-facing API.

use std::path::{Path, PathBuf};

use crate::cas::typed::{CasInputs, ContentHash};
use crate::run_meta::{FitMeta, RunKind};
use crate::run_paths;

/// Top-level fit run (the umbrella over a fit's stages).
#[derive(Clone)]
pub struct FitInputs {
    /// Pre-computed `fit_content_hash` (model IR + data + fit.toml bytes).
    /// Caller invokes `FitConfigV2::fit_content_hash` once and stashes
    /// the result here.
    pub fit_content_hash: String,
    /// Slugified stem from the fit.toml path (or model basename).
    pub stem: Option<String>,
    /// `FitMeta` payload for the umbrella's `run.json`.
    pub meta: FitMeta,
}

impl CasInputs for FitInputs {
    fn content_hash(&self) -> ContentHash {
        ContentHash::from_hex(self.fit_content_hash.clone())
    }
    fn cas_path(&self, root: &Path) -> PathBuf {
        run_paths::fit_run_dir(root, self.stem.as_deref(), &self.fit_content_hash)
    }
    fn run_kind(&self) -> RunKind {
        RunKind::Fit(self.meta.clone())
    }
}

// `StageInputs` (the legacy per-stage CAS-inputs type) was removed in M3.2:
// fit stages are now content-addressed through `runid` (`fit/cas.rs`
// `resolve_fit_stage` + Mode-B `claim_streaming`/`finalize`), so the legacy
// `fit_stage_hash`-keyed `cas_path` writer no longer has a constructor.
