//! The run store's maintenance commands on camdl's command line: `status`,
//! `verify`, `clean` and `reindex`, each the store library's
//! [`store::StoreCommand`] run against the store `--store`, `CAMDL_STORE` or
//! `./camdl-store` names. What each does is the library's
//! (`docs/camdl-run-spec.md` §4.4). The library's `list` and `show` are not
//! mounted: camdl's own `list`, `show` and `cat` read the run store and
//! `results/` together.

use std::io::Write as _;
use std::path::PathBuf;

/// Runs `command` against the run store `root` names, printing its report to
/// stdout. A root with no `store.json` is refused rather than made a store.
pub(crate) fn cmd_store(command: store::StoreCommand, root: Option<PathBuf>) {
    let out = crate::out::Out::new(false);
    let request = crate::camdl_store::store_request(None, false, root);
    let root = crate::camdl_store::root_of(&request);
    if !root.join("store.json").exists() {
        out.fail(format!("error: no run store at {} (it has no store.json)", root.display()));
    }
    let store = crate::camdl_store::open(&request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let mut report = Vec::new();
    command.run(&store, &mut report).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let _ = out.data().write_all(&report);
}
