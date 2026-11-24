//! The run store's maintenance commands on camdl's command line: `status`,
//! `verify`, `clean` and `reindex`, each the store library's
//! [`store::StoreCommand`] run against the store `--store`, `CAMDL_STORE` or
//! `./camdl-store` names, and `receive`, the library's
//! [`store::Store::receive`]. What each does is the library's
//! (`docs/camdl-run-spec.md` §4.4). The library's `list` and `show` are not
//! mounted: camdl's own `list`, `show` and `cat` read the run store and
//! `results/` together.

use std::io::Write as _;
use std::path::PathBuf;

/// Runs `command` against the run store `root` names, printing its report to
/// stdout. A root with no `store.json` is refused rather than made a store.
/// A `verify` that finds a damaged result exits 1 after its report.
pub(crate) fn cmd_store(command: store::StoreCommand, root: Option<PathBuf>) {
    let out = crate::out::Out::new(false);
    let request = crate::camdl_store::store_request(None, false, root);
    let root = crate::camdl_store::root_of(&request);
    if !root.join("store.json").exists() {
        out.fail(format!("error: no run store at {} (it has no store.json)", root.display()));
    }
    let store = crate::camdl_store::open(&request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let mut report = Vec::new();
    let verdict = command.run(&store, &mut report).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let _ = out.data().write_all(&report);
    if let store::Verdict::Damaged { results } = verdict {
        out.fail(format!("error: {results} damaged result(s)"));
    }
}

/// `camdl receive COPY`: places the results of a copy of another run store in
/// the store the request names, creating it when absent, and records them in
/// one invocation under `--label`. Prints one line per result received, saying
/// whether it was placed or was stored here already, and its folder. A refused
/// copy exits 1 with the store's reason, and nothing is placed.
pub(crate) fn cmd_receive(args: crate::args::ReceiveArgs) {
    let out = crate::out::Out::new(false);
    let copy = args.copy.clone();
    let (_, _, _, request, given) = store::Inputs::split(args).into_parts();
    let (label, root) = request.into_parts();
    let request = crate::camdl_store::store_request(label, false, root);
    let store = crate::camdl_store::open(&request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let info = store::RunInfo::new(&given, &(), &request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let received = match store.receive(&copy, &info) {
        Ok(r) => r,
        Err(e) => {
            // Ends the invocation record, if the receive began one, as
            // failed, before the process exits.
            drop(info);
            out.fail(format!("error: {e}"))
        }
    };
    let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let line = |what: &str, key: store::Key| match store.locate(key) {
        Ok(l) => out.data_line(format_args!("{what:<8} {} {}  {}", l.kind(), key.short12(), l.folder().display())),
        Err(e) => out.fail(format!("error: {e}")),
    };
    for key in &received.placed {
        line("placed", *key);
    }
    for key in &received.present {
        line("present", *key);
    }
    out.say(format_args!(
        "receive: {} placed, {} already stored, recorded as invocation {}",
        received.placed.len(),
        received.present.len(),
        invocation.seq()
    ));
    if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
        out.say(format_args!("receive: @{label} now names invocation {}, which held it from invocation {before}", invocation.seq()));
    }
}
