//! Label helpers of the fit store (`fit run`, `camdl label`), and the
//! digest that names a simulation cell's observation blocks
//! ([`obs_subtree_hash`]), which `simulate` folds into each `Obs`
//! by-product's identity.

/// Rewrite a leaf's `run.json` through `edit`, which returns whether it changed
/// the record: the store's one record-update path ([`runid::update_record`]),
/// under the leaf's `.lock` and through the reserved `run.json.tmp`, so no two
/// updates — and no update and an `augment` — interleave, and a crash leaves
/// only files the store's exact-set check ignores (gh#896).
fn rewrite_record(
    leaf_dir: &std::path::Path,
    edit: impl FnOnce(&mut runid::RunRecord) -> Result<bool, String>,
) -> Result<(), String> {
    runid::update_record(leaf_dir, edit).map_err(|e| match e {
        runid::CasError::EditRefused { reason, .. } => reason,
        other => format!("update {}: {other}", leaf_dir.join("run.json").display()),
    })
}

/// Ensure a leaf's `run.json` carries `provenance.label == Some(label)`,
/// rewriting it only when it differs. `provenance` is metadata, not identity —
/// the run_id and exact-set are untouched — applied at write time so an
/// explicit `simulate --label` survives a CAS cache hit, which relabels a sim.
/// Idempotent: a no-op when the leaf already has the requested label. Fit
/// runs use [`set_fit_label`], which also stamps the time.
pub(crate) fn ensure_provenance_label(leaf_dir: &std::path::Path, label: &str) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        if rec.provenance.label.as_deref() == Some(label) {
            return Ok(false);
        }
        rec.provenance.label = Some(label.to_string());
        Ok(true)
    })
}

/// Give a fit run `label`, stamped `label_set_at`: the run becomes the
/// label's holder, since a fit label names the completed run carrying it with
/// the latest stamp. A label the run already carries is re-stamped; another
/// is replaced, and the previous one is returned.
///
/// The one way a fit run's label is written. It changes only the two
/// provenance fields, never anything identity-bearing, and goes through the
/// store's record-update path (the leaf lock, `run.json.tmp`, fsync, rename).
/// The caller holds [`fit_label_lock`] and has checked that no live run
/// carries the label.
pub(crate) fn set_fit_label(
    leaf_dir: &std::path::Path,
    label: &str,
    label_set_at: &str,
) -> Result<Option<String>, String> {
    let mut previous = None;
    rewrite_record(leaf_dir, |rec| {
        previous = rec.provenance.label.clone().filter(|l| l != label);
        rec.provenance.label = Some(label.to_string());
        rec.provenance.label_set_at = Some(label_set_at.to_string());
        Ok(true)
    })?;
    Ok(previous)
}

/// Now, as a `label_set_at` stamp: RFC 3339 UTC with millisecond resolution.
pub(crate) fn label_stamp() -> String {
    crate::cas::iso8601_utc_millis(std::time::SystemTime::now())
}

/// The store-wide lock under which a fit label is checked for uniqueness and
/// written (gh#896): an `flock(2)` exclusive lock on `<root>/.fit-labels.lock`
/// (beside the store's kind directories, not inside `fits/`, which holds only
/// fit folders),
/// so two writers cannot both find a label free and both take it. The kernel
/// releases it when the holder drops it, exits or crashes, so there is no stale
/// lock to reclaim and no pid to trust; the file stays in the store and holds
/// `pid <n> on <host>` of the latest holder, for the refusal message only.
// Off Unix the lock is refused, so nothing constructs one.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) struct LabelLock {
    _file: std::fs::File,
}

impl std::fmt::Debug for LabelLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LabelLock")
    }
}

/// How long [`fit_label_lock`] waits for another holder before refusing.
const LABEL_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

pub(crate) fn fit_label_lock(root: &std::path::Path) -> Result<LabelLock, String> {
    fit_label_lock_within(root, LABEL_LOCK_WAIT)
}

#[cfg(unix)]
fn fit_label_lock_within(
    root: &std::path::Path,
    wait: std::time::Duration,
) -> Result<LabelLock, String> {
    use std::io::{Seek, Write};
    use std::os::unix::io::AsRawFd;
    std::fs::create_dir_all(root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
    let path = root.join(".fit-labels.lock");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let deadline = std::time::Instant::now() + wait;
    loop {
        // SAFETY: flock on an fd we own; LOCK_NB makes it non-blocking.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            let _ = file.set_len(0);
            let _ = file.rewind();
            let _ = write!(file, "pid {} on {}", std::process::id(), hostname());
            return Ok(LabelLock { _file: file });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(format!("cannot lock {}: {err}", path.display()));
        }
        if std::time::Instant::now() >= deadline {
            let recorded = std::fs::read_to_string(&path).unwrap_or_default();
            let recorded = recorded.trim();
            return Err(format!(
                "the store's label lock {} is held by another process (the latest holder \
                 recorded {}); a fit label is being written. Retry when it has finished.",
                path.display(),
                if recorded.is_empty() { "nothing" } else { recorded }
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Off Unix there is no `flock(2)`, and writing a label without the lock would
/// let two writers both find it free and both take it: refused, not skipped.
#[cfg(not(unix))]
fn fit_label_lock_within(
    root: &std::path::Path,
    _wait: std::time::Duration,
) -> Result<LabelLock, String> {
    Err(format!(
        "cannot take the store's label lock {}: fit label locking is unsupported on \
         this platform (it needs flock(2), available only on Unix), and a label is \
         not written without it",
        root.join(".fit-labels.lock").display()
    ))
}

/// This machine's hostname, for the label lock's holder note.
#[cfg(unix)]
fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: gethostname writes at most buf.len() bytes into buf.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return "unknown host".to_string();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Ensure a leaf's `run.json` lists `value` in its `inputs[key]` array,
/// rewriting it only when it is missing. `inputs` is the recorded-not-hashed
/// display projection, so the run_id and the exact-set are untouched. Used for
/// the config identities a fit run was reached from (gh#896): a cache hit from
/// a second config with the same run identity adds that config's.
pub(crate) fn ensure_input_listed(
    leaf_dir: &std::path::Path,
    key: &str,
    value: &str,
) -> Result<(), String> {
    rewrite_record(leaf_dir, |rec| {
        let obj = rec
            .inputs
            .as_object_mut()
            .ok_or_else(|| "`inputs` is not an object".to_string())?;
        let list = obj.entry(key.to_string()).or_insert_with(|| serde_json::json!([]));
        let arr = list
            .as_array_mut()
            .ok_or_else(|| format!("`inputs.{key}` is not a list"))?;
        if arr.iter().any(|v| v.as_str() == Some(value)) {
            return Ok(false);
        }
        arr.push(serde_json::Value::String(value.to_string()));
        Ok(true)
    })
}

/// The obs subtree's identity component.
///
/// A hash of the resolved observation blocks (run-spec: changing a reporting
/// parameter re-samples obs without invalidating the cached trajectory), plus
/// any `--emit-every` override (gh#656) — which changes WHICH times the subtree
/// carries, so two cadences must not address the same directory (the
/// count-in-the-key rule).
///
/// One function, because the hash is computed twice for every cell — once to
/// declare the `obs` child in `run.json`, once to name the subtree directory —
/// and a disagreement between the two would declare one child while writing
/// another. Byte-identical to the historical `sha256(json(observations))`
/// whenever no override is in play, so an unflagged run addresses exactly the
/// subtree it always did.
pub(crate) fn obs_subtree_hash(
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    let mut buf = serde_json::to_string(observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    if let Some(e) = emit {
        buf.push_str(&e.identity_repr());
    }
    // The file FORMAT is part of what the subtree holds (gh#833): the header
    // names the declared columns, a windowed stream carries two boundaries,
    // and a row outside the run is not written. Bytes that change under an
    // unchanged key would be a stored artifact the key no longer describes,
    // so the format version enters the key.
    buf.push_str(OBS_FILE_FORMAT);
    Ok(crate::hashing::sha256_hex(buf.as_bytes()))
}

/// The obs-file format the CAS subtree is written in — bumped whenever the
/// bytes written for an unchanged model would change (the count-in-the-key
/// rule).
///
/// 3 (D59): a row's value is drawn from the state, and with the time, where
/// the row is scored (its instant, or its period's stop), not at its label.
/// For a stream whose label is not its scoring time (`day`, `starting_on`,
/// `ending_on`) whose likelihood reads the state or `t`, the stored draws
/// change, so a subtree written before must not answer for one written now.
const OBS_FILE_FORMAT: &str = "\nobs-file-format:3";

#[cfg(test)]
mod tests {
    /// A leaf directory holding a minimal fit-method `run.json` with `label`.
    fn record_leaf(tag: &str, label: Option<&str>) -> std::path::PathBuf {
        let leaf = crate::test_support::unique_temp_dir(tag).join("if2-00000000").join("seed_1-00000000");
        std::fs::create_dir_all(&leaf).unwrap();
        let mut rec = serde_json::json!({
            "format_version": 1, "kind": "fit_method", "run_id": "ab".repeat(32),
            "hash_version": 1, "calendar": {"days_per_unit": 1.0, "origin": null, "time_unit": "days"}, "ir_version": "0.7", "engine_version": "0.1.0+test",
            "levels": [], "status": "completed", "artifacts": {}, "inputs": {"method": "if2"},
            "provenance": {},
        });
        if let Some(l) = label {
            rec["provenance"]["label"] = serde_json::json!(l);
        }
        std::fs::write(leaf.join("run.json"), rec.to_string()).unwrap();
        leaf
    }

    fn stored_label(leaf: &std::path::Path) -> Option<String> {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
        v["provenance"]["label"].as_str().map(String::from)
    }

    /// gh#896: a fit run's label is compare-and-set — another label on it is
    /// refused and left in place, the same label is a no-op, none is written.
    /// A fit label is written with its stamp: a fresh run takes it, a run
    /// carrying it is re-stamped, and a run carrying another has it replaced,
    /// the old one returned. Nothing but the two provenance fields changes.
    #[test]
    fn a_fit_label_is_written_with_its_stamp() {
        let stamp = |leaf: &std::path::Path| -> Option<String> {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
            v["provenance"]["label_set_at"].as_str().map(String::from)
        };
        let fresh = record_leaf("label_set_fresh", None);
        assert_eq!(set_fit_label(&fresh, "wide", "2026-11-14T00:00:00.000Z").unwrap(), None);
        assert_eq!(stored_label(&fresh).as_deref(), Some("wide"));
        assert_eq!(stamp(&fresh).as_deref(), Some("2026-11-14T00:00:00.000Z"));

        let held = record_leaf("label_set_held", Some("scout"));
        let before: serde_json::Value =
            serde_json::from_slice(&std::fs::read(held.join("run.json")).unwrap()).unwrap();
        assert_eq!(set_fit_label(&held, "scout", "2026-11-14T00:00:01.000Z").unwrap(), None,
            "re-stamping the same label replaces nothing");
        assert_eq!(set_fit_label(&held, "renamed", "2026-11-14T00:00:02.000Z").unwrap().as_deref(),
            Some("scout"), "the replaced label is returned");
        assert_eq!(stored_label(&held).as_deref(), Some("renamed"));
        assert_eq!(stamp(&held).as_deref(), Some("2026-11-14T00:00:02.000Z"));
        let after: serde_json::Value =
            serde_json::from_slice(&std::fs::read(held.join("run.json")).unwrap()).unwrap();
        for key in ["run_id", "levels", "deps", "inputs", "artifacts", "status"] {
            assert_eq!(before[key], after[key], "{key} is untouched");
        }
    }

    #[test]
    fn a_label_stamp_is_rfc3339_with_milliseconds() {
        let s = label_stamp();
        assert_eq!(s.len(), "2026-11-14T00:00:00.000Z".len(), "{s}");
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T' && s.as_bytes()[19] == b'.', "{s}");
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_millis(1_794_614_400_042);
        assert_eq!(crate::cas::iso8601_utc_millis(t), "2026-11-14T00:00:00.042Z");
    }

    /// gh#896: concurrent read-modify-writes of one record lose nothing — each
    /// is one step under the record lock, through its own temporary file.
    #[test]
    fn concurrent_record_edits_lose_no_update() {
        let leaf = record_leaf("record_race", None);
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let leaf = leaf.clone();
                std::thread::spawn(move || {
                    ensure_input_listed(&leaf, "config_identities", &format!("id{i}")).unwrap()
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
        let ids = v["inputs"]["config_identities"].as_array().unwrap();
        assert_eq!(ids.len(), 16, "every writer's entry survives: {ids:?}");
    }

    /// gh#896: the store's label lock excludes a second holder while held, is
    /// free again once dropped, and is never blocked by a lock file whose
    /// recorded holder is not holding it — a crashed process, or a live
    /// process that reused its pid.
    #[cfg(unix)]
    #[test]
    fn the_label_lock_is_held_only_while_held() {
        let root = crate::test_support::unique_temp_dir("label_lock");
        let short = std::time::Duration::from_millis(100);
        let held = fit_label_lock(&root).unwrap();
        let err = fit_label_lock_within(&root, short).expect_err("a held lock excludes");
        assert!(err.contains("held by another process"), "{err}");
        assert!(err.contains(&format!("pid {}", std::process::id())), "names the recorded holder: {err}");
        drop(held);
        let again = fit_label_lock_within(&root, short).expect("free once dropped");
        drop(again);
        // A file naming a live pid (ours) that does not hold the lock.
        std::fs::write(root.join(".fit-labels.lock"), format!("{}", std::process::id())).unwrap();
        fit_label_lock_within(&root, short).expect("a recorded pid alone does not hold the lock");
        std::fs::remove_dir_all(&root).ok();
    }

    use super::*;
}
