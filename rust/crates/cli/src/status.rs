//! Tidy, colored one-shot status lines for the CLI — milestones, NOT logs.
//!
//! A small shared vocabulary so every subcommand's user-facing milestones
//! (`compiled`, `storing`, `stored`, …) read identically: a right-aligned
//! bold-green verb + plain detail, cargo-style (`   compiled  …`), with a
//! dimmed indented continuation for hints/next-steps. All to **stderr** so
//! stdout stays data-only.
//!
//! These are distinct from `progress` bars: low-volume, one-shot, and always
//! shown (they're results, not per-iteration chatter) — `--progress none`
//! silences the *bars*, not these milestone lines.
//!
//! Use these instead of ad-hoc `eprintln!("✓ …")` so the look stays unified;
//! when you add a new milestone, reach for `step`/`done`/`hint` first.

use crate::style::{self, Style};

/// Width the verb column is right-aligned to (cargo uses ~12; we use 9 to fit
/// `compiled`/`storing`/`stored`/`ensemble` without truncation).
const VERB_W: usize = 9;

/// An intermediate milestone: bold-green right-aligned `verb` + `detail`, e.g.
/// `  compiled  sir.camdl → /tmp/sir.ir.json   3.3s (1.0 MB)`. (Pad before
/// coloring so the ANSI bytes don't break the alignment.)
pub fn step(verb: &str, detail: impl std::fmt::Display) {
    let p = style::stderr();
    eprintln!("{} {}", p.paint(&p.paint(&format!("{verb:>VERB_W$}"), Style::Bold), Style::Ok), detail);
}

/// A terminal success milestone — same as [`step`] (the verb already carries
/// the meaning, e.g. `stored`); kept as a separate name so call sites read
/// intention-first. Reserved for the last line a command prints.
pub fn done(verb: &str, detail: impl std::fmt::Display) {
    step(verb, detail);
}

/// A dimmed continuation under a milestone — a hint or next-step command,
/// aligned under the detail column, e.g. `           camdl cat <id>`.
pub fn hint(detail: impl std::fmt::Display) {
    eprintln!("{:>VERB_W$} {}", "", style::stderr().paint(&detail.to_string(), Style::Dim));
}

/// A concise display path: relative to the current directory when the path is
/// under it (≈ relative to the project root, since `results/` lives there), so
/// `~/proj/nga/aggfit/model.camdl` shown from the project root reads
/// `aggfit/model.camdl`; otherwise just the file name. Keeps milestone lines
/// short when the user passed a long absolute path.
pub fn concise_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(rel) = p.strip_prefix(&cwd) {
            return rel.display().to_string();
        }
    }
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// Human-readable byte size, no space: `1.0MB`, `12.3KB`, `512B`. The
/// space-less form (à la `ls -lh` / `du -h` / `docker`) is non-SI but reads
/// compactly in a CLI; used for the compile banner's IR size and similar.
pub fn human_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    let b = n as f64;
    if b < KB { format!("{n}B") }
    else if b < KB * KB { format!("{:.1}KB", b / KB) }
    else if b < KB * KB * KB { format!("{:.1}MB", b / (KB * KB)) }
    else { format!("{:.1}GB", b / (KB * KB * KB)) }
}

/// The IR-to-source size ratio for the compile banner, e.g. ` (3.2× source)`
/// or ` (2545× source)` — how much bigger the compiled IR is than the `.camdl`
/// (dominated by stratification expansion for big models; mostly JSON
/// verbosity for tiny ones). Leading space so it appends cleanly; empty when
/// the source size is unknown.
pub fn expansion(ir_bytes: u64, src_bytes: u64) -> String {
    if src_bytes == 0 { return String::new(); }
    let r = ir_bytes as f64 / src_bytes as f64;
    if r >= 10.0 { format!(" ({:.0}\u{d7} source)", r) }
    else { format!(" ({:.1}\u{d7} source)", r) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_scales_no_space() {
        assert_eq!(human_bytes(512), "512B");
        assert_eq!(human_bytes(1536), "1.5KB");
        assert_eq!(human_bytes(1_048_576), "1.0MB");
        assert_eq!(human_bytes(5 * 1_048_576), "5.0MB");
    }

    #[test]
    fn expansion_ratio_formats_and_handles_unknown() {
        assert_eq!(expansion(0, 0), "");          // unknown source → omitted
        assert_eq!(expansion(100, 0), "");
        assert_eq!(expansion(3200, 1000), " (3.2\u{d7} source)");   // <10 → 1 dp
        assert_eq!(expansion(5_600_000, 2200), " (2545\u{d7} source)"); // ≥10 → 0 dp
    }

    #[test]
    fn concise_path_falls_back_to_basename_off_cwd() {
        // An absolute path that cannot be under the test's cwd → just the
        // file name. (The under-cwd → relative branch is env-dependent, so
        // only the deterministic fallback is asserted.)
        assert_eq!(concise_path("/nonexistent/deep/dir/model_C_lga44.camdl"),
            "model_C_lga44.camdl");
    }
}
