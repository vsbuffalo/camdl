//! What a command may do with a leaf it has read ([`runid::read_leaf`]).
//!
//! The reader reports every state and every unreadable record; it decides
//! nothing. A command that *consumes a run's results* — its posterior draws,
//! its point estimate, its chain states — requires a completed run, and says
//! which state it found instead ([`require_completed`]). A command that only
//! shows a leaf (`fit summary`, `camdl show`, `camdl list`) shows every state
//! and does not call this.

use std::path::Path;

use runid::{Leaf, LeafState, UnreadableLeaf};

/// Why a leaf in `state` has no result to read, as a clause that follows
/// "is not a completed run: ". `None` for a completed leaf.
pub fn not_completed_reason(state: &LeafState) -> Option<String> {
    Some(match state {
        LeafState::Completed => return None,
        LeafState::Failed { failure } => {
            let message = failure.message.strip_prefix("error: ").unwrap_or(&failure.message);
            format!("it failed: {message}")
        }
        LeafState::FailedReasonUnknown => "it failed, and recorded no reason".to_string(),
        LeafState::Running { pid: Some(pid) } => format!("it is still running (pid {pid})"),
        LeafState::Running { pid: None } => {
            "it is still running (its lock is held, by a pid that cannot be read)".to_string()
        }
        LeafState::OwnerUnknown { pid, host } => format!(
            "it is held by pid {pid} on host {host}, whose liveness this host cannot check, \
             and has recorded no outcome"
        ),
        LeafState::Interrupted { .. } => {
            "it was interrupted: no live process holds it, and it recorded no outcome".to_string()
        }
    })
}

/// `Ok` when `leaf` is a completed run; otherwise the refusal, naming the
/// leaf and its state. The files of a run that did not complete — a
/// `draws.tsv` written before a later step failed, or before the process was
/// killed — are not its result, however complete they look.
pub fn require_completed(leaf: &Leaf) -> Result<(), String> {
    require_completed_state(&leaf.dir, &leaf.state)
}

/// [`require_completed`] for a leaf whose directory and state are in hand.
pub fn require_completed_state(dir: &Path, state: &LeafState) -> Result<(), String> {
    match not_completed_reason(state) {
        None => Ok(()),
        Some(reason) => Err(format!(
            "{} is not a completed run: {reason}.\n  \
             Only a completed run's results are read; `camdl fit summary {}` shows what it recorded.",
            dir.display(),
            dir.display()
        )),
    }
}

/// An unreadable candidate leaf as one sentence: its directory, then why its
/// record could not be read (which names `run.json`).
pub fn unreadable(u: &UnreadableLeaf) -> String {
    format!("{}: {}", u.dir.display(), u.reason)
}

/// [`runid::read_leaf`] for a command that needs a leaf at `dir`: `Ok(None)`
/// when `dir` is no leaf at all, the leaf when its record reads, and the
/// unreadable record as an error sentence ([`unreadable`]).
pub fn read(dir: &Path) -> Result<Option<Leaf>, String> {
    runid::read_leaf(dir).map_err(|u| unreadable(&u))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_completed_state_has_no_reason_and_every_other_state_one() {
        assert_eq!(not_completed_reason(&LeafState::Completed), None);
        let failed = LeafState::Failed {
            failure: runid::Failure {
                message: "error: all 2 chains failed init-eval".into(),
                operation: "pmmh".into(),
            },
        };
        assert_eq!(
            not_completed_reason(&failed).as_deref(),
            Some("it failed: all 2 chains failed init-eval")
        );
        for (state, says) in [
            (LeafState::FailedReasonUnknown, "recorded no reason"),
            (LeafState::Running { pid: Some(41) }, "still running (pid 41)"),
            (LeafState::Running { pid: None }, "still running"),
            (LeafState::OwnerUnknown { pid: 7, host: "h2".into() }, "pid 7 on host h2"),
            (LeafState::Interrupted { pid: Some(9) }, "interrupted"),
            (LeafState::Interrupted { pid: None }, "interrupted"),
        ] {
            let reason = not_completed_reason(&state).unwrap();
            assert!(reason.contains(says), "{state:?}: {reason}");
            let refusal = require_completed_state(Path::new("leaf"), &state).unwrap_err();
            assert!(refusal.starts_with("leaf is not a completed run: ") && refusal.contains(says),
                "{refusal}");
        }
    }
}
