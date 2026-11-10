//! What the engine meets during a run that the user should be told once.
//!
//! The library never prints (proposal
//! `docs/dev/proposals/2026-11-08-cli-reporting.md`), and these conditions
//! arise where nothing can be returned: deep in a likelihood evaluation a
//! particle filter repeats thousands of times, or in a plan built once per
//! evaluation. So the engine records each one here. The first occurrence of
//! each distinct note in a process goes to the reporter the command installed
//! ([`set_reporter`]), which says it then; with no reporter installed it waits
//! in a queue that [`take`] drains. Later occurrences of the same note are
//! dropped, as the once-per-process warnings these replace were.
//!
//! A count of how often something happened belongs in
//! [`crate::eval_stats`] instead; this is for a sentence said once.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::effect_plan::SnapNote;
use crate::inference::linalg::NotPositiveDefinite;

/// One thing to tell the user.
#[derive(Clone, Debug, PartialEq)]
pub enum RunNote {
    /// An effect's firing step was chosen by the shortened-last-step rule.
    EffectSnapped(SnapNote),
    /// A `normal(...)` observation likelihood scored a datum that is not an
    /// integer; the likelihood is a discretized count distribution and rounds
    /// it. Said once per process, for the first such datum.
    NonIntegerNormalObservation { observed: f64 },
    /// A Cholesky factorisation the engine needed failed, and it carried on
    /// with a fallback factor.
    NotPositiveDefinite(NotPositiveDefinite),
    /// The complete-data density's gamma-multiplier walk consumed a different
    /// number of multipliers than the trajectory recorded at a substep: the
    /// divergence detector between that walk and the transition density's.
    GammaIndexMismatch { substep: usize, tracked: usize, recorded: usize },
}

impl RunNote {
    /// A stable kebab-case name for the condition, for a notice's code.
    pub fn code(&self) -> &'static str {
        match self {
            RunNote::EffectSnapped(_) => "effect-snapped-to-run-end",
            RunNote::NonIntegerNormalObservation { .. } => "non-integer-normal-observation",
            RunNote::NotPositiveDefinite(_) => "not-positive-definite",
            RunNote::GammaIndexMismatch { .. } => "gamma-index-mismatch",
        }
    }

    /// What makes two notes the same note, so it is said once.
    fn key(&self) -> String {
        match self {
            RunNote::EffectSnapped(n) => {
                format!("{}\u{0}{}\u{0}{}", n.effect, n.declared.to_bits(), n.fires_at.to_bits())
            }
            RunNote::NonIntegerNormalObservation { .. } => String::new(),
            other => other.to_string(),
        }
    }
}

impl std::fmt::Display for RunNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunNote::EffectSnapped(n) => write!(f, "{n}"),
            RunNote::NonIntegerNormalObservation { observed } => write!(
                f,
                "Normal observation likelihood: observed value {:.6} is \
                 not integer-valued. The `normal(...)` likelihood is a \
                 discretized-count distribution (pomp/He et al.); your \
                 data will be rounded to the nearest non-negative integer \
                 before scoring. If you need a continuous Normal PDF, \
                 file a request for a ContinuousNormal variant.",
                observed
            ),
            RunNote::NotPositiveDefinite(e) => write!(f, "{e}"),
            RunNote::GammaIndexMismatch { substep, tracked, recorded } => write!(
                f,
                "gamma index mismatch at substep {}: tracked {} but trajectory recorded {} gammas",
                substep, tracked, recorded
            ),
        }
    }
}

/// Who says a note when it is first recorded.
pub type Reporter = fn(&RunNote);

static REPORTER: OnceLock<Reporter> = OnceLock::new();

struct Book {
    seen: HashSet<(&'static str, String)>,
    pending: Vec<RunNote>,
}

fn book() -> &'static Mutex<Book> {
    static BOOK: OnceLock<Mutex<Book>> = OnceLock::new();
    BOOK.get_or_init(|| Mutex::new(Book { seen: HashSet::new(), pending: Vec::new() }))
}

/// Install the function that says each note as it is first recorded. Set
/// once, by the command-line entry point; a second call is ignored.
pub fn set_reporter(reporter: Reporter) {
    let _ = REPORTER.set(reporter);
}

/// Record `note`. Its first occurrence in the process is reported, or queued
/// when no reporter is installed; a repeat is dropped.
pub fn record(note: RunNote) {
    let first = book()
        .lock()
        .map(|mut b| b.seen.insert((note.code(), note.key())))
        .unwrap_or(true);
    if !first {
        return;
    }
    match REPORTER.get() {
        Some(say) => say(&note),
        None => {
            if let Ok(mut b) = book().lock() {
                b.pending.push(note);
            }
        }
    }
}

/// The notes recorded with no reporter installed, oldest first, emptying the
/// queue.
pub fn take() -> Vec<RunNote> {
    book().lock().map(|mut b| std::mem::take(&mut b.pending)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A note is said once however often it is recorded, and a different
    /// note is said too. The substep numbers are ones no other test records,
    /// since the book is process-wide.
    #[test]
    fn a_note_is_said_once() {
        let note = |substep| RunNote::GammaIndexMismatch { substep, tracked: 1, recorded: 2 };
        record(note(usize::MAX - 7));
        record(note(usize::MAX - 7));
        record(note(usize::MAX - 8));
        let mine: Vec<RunNote> = take().into_iter()
            .filter(|n| matches!(n, RunNote::GammaIndexMismatch { substep, .. } if *substep >= usize::MAX - 8))
            .collect();
        assert_eq!(mine, vec![note(usize::MAX - 7), note(usize::MAX - 8)]);
        assert_eq!(mine[0].to_string(),
            format!("gamma index mismatch at substep {}: tracked 1 but trajectory recorded 2 gammas", usize::MAX - 7));
    }
}
