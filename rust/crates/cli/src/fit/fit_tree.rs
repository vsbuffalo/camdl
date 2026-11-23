//! The fit runs a run store holds: [`walk_fits_root`] gives one
//! [`FitDirEntry`] per stored fit run, each opened and carrying its parsed
//! [`FitView`], so callers filter by model identity or method without a
//! second read. A view's [`FitMethodView`] carries no method-specific file;
//! a consumer that needs the typed result loads it with
//! `MethodResult::load_from`.

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

/// One stored fit run returned by [`walk_fits_root`], opened for reading:
/// `view` is its view, read from the copy of its files, and `opened` keeps
/// the copy and the run's problem.
#[derive(Clone)]
pub struct FitDirEntry {
    pub view: FitView,
    pub opened: std::sync::Arc<crate::fit_store::OpenedFit>,
}

/// What the store records of `opened`'s run beside its files, written into
/// its view: the fit file it was given (`source_config`) and when it started
/// (`created_at`), from its `run.json`, and `label`.
pub fn fill_from_record(view: &mut FitView, opened: &crate::fit_store::OpenedFit, label: Option<String>) {
    if let Some(run) = &opened.fit.run {
        view.source_config = run.given().pointer("/config/path").and_then(|v| v.as_str()).map(String::from);
        if let Some(m) = view.methods.first_mut() {
            m.created_at = run.started().0.clone();
        }
    }
    if let Some(m) = view.methods.first_mut() {
        m.label = label;
    }
}

/// The label of the latest invocation that asked for the run `run` and held
/// a label, in the store at `root`.
pub fn run_label(root: &Path, run: &store::Key) -> Option<String> {
    let store = crate::camdl_store::open_existing(root)?;
    let mut label = None;
    for inv in store.invocations() {
        let Ok(members) = inv.members::<crate::fit_store::FitRun>() else { continue };
        if members.iter().any(|(id, _)| id.key() == *run) && inv.label().is_some() {
            label = inv.label().map(str::to_string);
        }
    }
    label
}

/// Every fit run stored in the run store at `root`, each opened for reading
/// ([`crate::fit_store::open_fit`]), in run-id order. A root that holds no
/// store holds no fit.
///
/// Each view carries what the store records of the run beside its files: the
/// label of the latest invocation that asked for it and holds a label, the
/// fit file it was given (`source_config`) and when it started
/// (`created_at`), from its `run.json`.
pub fn walk_fits_root(root: &Path) -> io::Result<Vec<FitDirEntry>> {
    let mut out = Vec::new();
    let Some(store) = crate::camdl_store::open_existing(root) else { return Ok(out) };
    let mut labels: std::collections::BTreeMap<store::Key, String> = std::collections::BTreeMap::new();
    let mut order: Vec<store::Key> = Vec::new();
    for inv in store.invocations() {
        let Ok(members) = inv.members::<crate::fit_store::FitRun>() else { continue };
        for (id, _) in members {
            if !order.contains(&id.key()) {
                order.push(id.key());
            }
            if let Some(l) = inv.label() {
                labels.insert(id.key(), l.to_string());
            }
        }
    }
    for key in order {
        // A run that did not complete is not a stored fit.
        let Ok(opened) = crate::fit_store::open_fit(&store, &key.to_hex()) else { continue };
        let Some(dir) = opened.run_dir().map(Path::to_path_buf) else { continue };
        let Some(mut view) = FitView::read(&dir) else { continue };
        fill_from_record(&mut view, &opened, labels.get(&key).cloned());
        out.push(FitDirEntry { view, opened: std::sync::Arc::new(opened) });
    }
    out.sort_by(|a, b| a.view.methods[0].run_id.cmp(&b.view.methods[0].run_id));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_fits_root_empty_when_no_root() {
        let entries = walk_fits_root(Path::new("/definitely/not/here/either/fits")).unwrap();
        assert!(entries.is_empty());
    }
}
