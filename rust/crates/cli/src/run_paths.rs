//! Paths camdl shows and the old results root its readers of old stores
//! search.

use std::path::PathBuf;

/// The old results root the readers of old stores search when no root is
/// named. `results/` pairs with `data/` in the research-workflow vocabulary
/// the project's downstream users (book chapters, vignettes) speak.
pub const DEFAULT_OUTPUT_ROOT: &str = "results";

/// Absolutize an output path for DISPLAY, without requiring it to exist.
///
/// gh#507: a run announces where it will write before it has written
/// anything, so `canonicalize` is unavailable (the directory is not there
/// yet) — and echoing the path as-written is what let a fit report
/// `output: ../data/build/camdl/runs/...` while actually writing to a
/// sibling of the repository. A relative string is consistent with more
/// than one base, so it cannot disambiguate; an absolute one can.
///
/// This is `std::path::absolute`: purely lexical, no symlink resolution and
/// no `..` collapsing, so the printed path is exactly the path that will be
/// opened. Falls back to the input if the CWD is unreadable.
pub fn display_absolute(p: &std::path::Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#507: the announced output path must be absolute even when the
    /// root came from the `results/` default or `CAMDL_OUTPUT_DIR` (neither
    /// of which is anchored at the fit.toml), and must survive a path that
    /// does not exist yet.
    #[test]
    fn display_absolute_absolutizes_a_nonexistent_relative_path() {
        let cwd = std::env::current_dir().unwrap();
        let shown = display_absolute(std::path::Path::new(
            "results/fits/fit-deadbeef/no-such-dir"));
        assert!(shown.is_absolute(), "announced path must be absolute: {shown:?}");
        assert!(shown.starts_with(&cwd),
            "a relative root anchors at the CWD: {shown:?} vs {cwd:?}");
        // Absolute in → unchanged (no CWD prefix, no canonicalization).
        assert_eq!(display_absolute(std::path::Path::new("/tmp/x/y")),
            PathBuf::from("/tmp/x/y"));
    }
}
