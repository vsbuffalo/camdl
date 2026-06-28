//! Fit handles: resolve a user-typed fit reference to its sealed fit segment.
//!
//! The modeler loop is `fit → {summarize, predict, scenario, compare}`,
//! referencing the fit each time. A fit is referenced four ways, and resolution
//! is the fallible boundary: a bare hex string and a relative path are
//! genuinely ambiguous, so we classify by cheap syntactic priority
//! (`@name` → label; `*.toml` → config; an existing dir → run dir; else →
//! hash prefix) and let resolution fail with a typed [`ResolveError`] — never a
//! silent guess. Ambiguity is a typed, git-style listed outcome, not a
//! heuristic pick.
//!
//! This subsumes the per-verb `resolve_segment` (which handled only RunDir /
//! Config) and adds `@label` (a label→run lookup over the fit sidecars) and
//! hash-prefix (matched on the fit-level hash — the same one `fit table` and
//! `camdl list` display as `fit_id`, and `camdl label <hash>` resolves).
//!
//! Phase 1b of
//! `docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md`.

use crate::fit::config_v2::FitConfigV2;
use std::path::{Path, PathBuf};

/// A parsed fit reference, before resolution. Classification is syntactic and
/// total; the I/O (and the ambiguity) lives in resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitRef {
    /// `@name` — the leading `@` sigil is the label marker (stripped here).
    Label(String),
    /// A `fit.toml` config → its unique run (or an ambiguity list).
    Config(PathBuf),
    /// An existing fit results (segment) directory.
    RunDir(PathBuf),
    /// A fit-level hash prefix (what `fit table` / `list` display as `fit_id`).
    HashPrefix(String),
}

impl FitRef {
    /// Classify a raw handle by syntactic priority: `@` → Label; `*.toml` →
    /// Config; an existing directory → RunDir; else → HashPrefix. The order is
    /// the contract — a `fit.toml` that also happens to name an existing
    /// directory is still a Config (the `.toml` suffix wins), and anything left
    /// is treated as a hex prefix (which resolution validates).
    pub fn classify(s: &str) -> FitRef {
        if let Some(name) = s.strip_prefix('@') {
            return FitRef::Label(name.to_string());
        }
        if Path::new(s).extension().map(|e| e.eq_ignore_ascii_case("toml")).unwrap_or(false) {
            return FitRef::Config(PathBuf::from(s));
        }
        if Path::new(s).is_dir() {
            return FitRef::RunDir(PathBuf::from(s));
        }
        FitRef::HashPrefix(s.to_string())
    }
}

/// Typed resolution failure — surfaced verbatim; ambiguity is listed git-style.
#[derive(Debug)]
pub enum ResolveError {
    /// No fit matched the handle.
    NotFound(String),
    /// The handle matched more than one fit — the candidates are listed so the
    /// user can disambiguate (a run directory, or a longer hash prefix).
    Ambiguous { handle: String, candidates: Vec<PathBuf> },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(msg) => write!(f, "{msg}"),
            ResolveError::Ambiguous { handle, candidates } => {
                writeln!(f, "{} resolves to {} fits:", handle, candidates.len())?;
                for c in candidates {
                    writeln!(f, "    {}", c.display())?;
                }
                write!(
                    f,
                    "  Pass a run directory or a longer hash prefix to disambiguate."
                )
            }
        }
    }
}

/// A resolved fit: its segment directory + the loaded config.
///
/// The proposal's full `SealedFit` (model IR + posterior ensemble + diagnostics)
/// is Phase 4; the verbs today consume exactly the segment + config that the old
/// `resolve_segment` returned, so that is what resolution yields.
pub struct ResolvedFit {
    pub segment: PathBuf,
    pub config: FitConfigV2,
}

/// Resolve a raw fit handle to its segment directory only (no config load).
/// The cheaper entry point for verbs that operate on the directory (e.g.
/// `fit summary`); [`resolve_fit`] layers the config load on top.
pub fn resolve_fit_segment(s: &str) -> Result<PathBuf, ResolveError> {
    Ok(resolve_inner(s)?.0)
}

/// Resolve a raw fit handle to its segment + the config that governs its data.
pub fn resolve_fit(s: &str) -> Result<ResolvedFit, ResolveError> {
    let (segment, live_config) = resolve_inner(s)?;
    let config = match live_config {
        // The Config (`fit.toml`) handle named the live config — its relative
        // data/model paths already resolve against the config's own directory.
        Some(c) => c,
        // A segment handle (run-dir / @label / hash) has no caller-supplied
        // config; recover one whose data paths resolve correctly.
        None => load_config_for_segment(&segment).map_err(ResolveError::NotFound)?,
    };
    Ok(ResolvedFit { segment, config })
}

/// The 4-way resolution to a segment. The Config branch also returns the
/// already-loaded live config (so the data paths it resolved against the user's
/// directory are preserved); the other branches return `None` and let
/// [`resolve_fit`] recover the config from the segment.
fn resolve_inner(s: &str) -> Result<(PathBuf, Option<FitConfigV2>), ResolveError> {
    match FitRef::classify(s) {
        FitRef::RunDir(dir) => {
            if dir.is_dir() {
                Ok((dir, None))
            } else {
                Err(ResolveError::NotFound(format!("no such fit directory: {}", dir.display())))
            }
        }
        FitRef::Config(path) => {
            let (seg, cfg) = resolve_config(&path)?;
            Ok((seg, Some(cfg)))
        }
        FitRef::Label(name) => {
            let seg = resolve_by(&format!("@{name}"), |seg| {
                crate::run_meta::read_fit_sidecar(seg)
                    .and_then(|side| side.label)
                    .as_deref()
                    == Some(name.as_str())
            })?;
            Ok((seg, None))
        }
        FitRef::HashPrefix(hex) => {
            let seg = resolve_by(&hex, |seg| {
                crate::fit::fit_view::FitView::read(seg)
                    .map(|v| v.fit_hash.starts_with(&hex))
                    .unwrap_or(false)
            })?;
            Ok((seg, None))
        }
    }
}

/// Recover the config for a segment reached by run-dir / `@label` / hash.
///
/// Data archival is not yet part of the sealed packet (Phase 1 archives only
/// the model IR), so the data still lives at the paths the config names — and
/// those paths are relative to the *original* fit.toml directory, not the
/// segment. So we prefer the live original config the sidecar recorded
/// (its data paths resolve correctly), gated on a `fit_toml_hash` match so we
/// never load a different file that happens to share the path. Only when the
/// original is gone or changed do we fall back to the archived copy in the
/// segment (whose relative paths then resolve against the segment — correct
/// only if the data is co-located there).
fn load_config_for_segment(segment: &Path) -> Result<FitConfigV2, String> {
    if let Some(side) = crate::run_meta::read_fit_sidecar(segment) {
        let orig = Path::new(&side.fit_toml_path);
        if orig.is_file() {
            if let Ok(bytes) = std::fs::read(orig) {
                if crate::hashing::sha256_hex(&bytes) == side.fit_toml_hash {
                    if let Ok(cfg) = FitConfigV2::load(&side.fit_toml_path) {
                        return Ok(cfg);
                    }
                }
            }
        }
    }
    let archived = segment.join("fit.toml.original");
    FitConfigV2::load(&archived.to_string_lossy()).map_err(|e| {
        format!(
            "{} is a fit directory but its archived config could not be read: {e}\n  \
             (expected {})",
            segment.display(),
            archived.display()
        )
    })
}

/// The Config (`fit.toml`) branch: hash the config bytes and match against the
/// run store's sidecars (`fit_toml_hash`), under the root the config names.
/// Returns the matched segment and the live config (loaded once here).
fn resolve_config(path: &Path) -> Result<(PathBuf, FitConfigV2), ResolveError> {
    let bytes = std::fs::read(path)
        .map_err(|e| ResolveError::NotFound(format!("cannot read fit config {}: {e}", path.display())))?;
    let toml_hash = crate::hashing::sha256_hex(&bytes);
    let config = FitConfigV2::load(&path.to_string_lossy())
        .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
    let cas_root = crate::run_paths::output_root(None, config.output_dir.as_deref());
    let fits_dir = cas_root.join("fits");
    let mut matches: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&fits_dir) {
        for entry in entries.flatten() {
            let seg = entry.path();
            if !seg.is_dir() {
                continue;
            }
            if let Some(side) = crate::run_meta::read_fit_sidecar(&seg) {
                if side.fit_toml_hash == toml_hash {
                    matches.push(seg);
                }
            }
        }
    }
    matches.sort();
    match matches.len() {
        0 => Err(ResolveError::NotFound(format!(
            "no completed fit found for {} under {}.\n  Run `camdl fit run {}` first.",
            path.display(),
            fits_dir.display(),
            path.display()
        ))),
        1 => Ok((matches.remove(0), config)),
        _ => Err(ResolveError::Ambiguous { handle: path.display().to_string(), candidates: matches }),
    }
}

/// Scan `fits/` segments under the default output root and return the unique
/// one matching `pred`. Used by the label and hash-prefix branches, which have
/// no config to name a custom root — they resolve against the conventional
/// `results/` root (or `CAMDL_OUTPUT_DIR`); a fit under a custom `output_dir`
/// is reached by its run dir or `fit.toml`.
fn resolve_by(
    handle: &str,
    pred: impl Fn(&Path) -> bool,
) -> Result<PathBuf, ResolveError> {
    let fits_dir = crate::run_paths::output_root(None, None).join("fits");
    let mut matches: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&fits_dir) {
        for entry in entries.flatten() {
            let seg = entry.path();
            if seg.is_dir() && pred(&seg) {
                matches.push(seg);
            }
        }
    }
    matches.sort();
    match matches.len() {
        0 => Err(ResolveError::NotFound(format!(
            "no fit found for {handle} under {}",
            fits_dir.display()
        ))),
        1 => Ok(matches.remove(0)),
        _ => Err(ResolveError::Ambiguous { handle: handle.to_string(), candidates: matches }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_prioritizes_label_sigil() {
        assert_eq!(FitRef::classify("@jigawa-baseline"), FitRef::Label("jigawa-baseline".into()));
    }

    #[test]
    fn classify_recognizes_toml_config() {
        assert_eq!(FitRef::classify("fit.toml"), FitRef::Config(PathBuf::from("fit.toml")));
        assert_eq!(FitRef::classify("a/b/run.TOML"), FitRef::Config(PathBuf::from("a/b/run.TOML")));
    }

    #[test]
    fn classify_falls_through_to_hash_prefix() {
        // A hex string that is neither a label, a .toml, nor an existing dir.
        assert_eq!(FitRef::classify("b4aa952d"), FitRef::HashPrefix("b4aa952d".into()));
    }

    #[test]
    fn classify_existing_dir_is_run_dir() {
        let tmp = std::env::temp_dir().join(format!("camdl_handle_classify_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let s = tmp.to_string_lossy().into_owned();
        assert_eq!(FitRef::classify(&s), FitRef::RunDir(tmp.clone()));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ambiguous_lists_candidates() {
        let e = ResolveError::Ambiguous {
            handle: "@dup".into(),
            candidates: vec![PathBuf::from("results/fits/a"), PathBuf::from("results/fits/b")],
        };
        let msg = e.to_string();
        assert!(msg.contains("@dup resolves to 2 fits"));
        assert!(msg.contains("results/fits/a") && msg.contains("results/fits/b"));
    }
}
