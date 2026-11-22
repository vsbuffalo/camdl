//! The arguments of the run store's maintenance commands: `status`, `verify`,
//! `clean` and `reindex`.

use std::path::PathBuf;

use clap::Args;

use crate::colored_help;

/// The run store a maintenance command reads.
#[derive(Args, Debug, Clone)]
pub struct StoreRootArgs {
    /// The run store's root, where `pfilter` and `simulate` store their
    /// results (default `$CAMDL_STORE`, else `./camdl-store`)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
#[command(after_help = colored_help!("\
Lists every invocation still recorded as running, and every result whose folder
is recorded as running or failed. A run is `running` while its owner, the
process that claimed it, is alive, and `interrupted` once that process has
stopped without finishing; a failed run is listed with the reason it recorded.
An interrupted invocation is listed until `camdl clean` closes its record.
Completed results are not listed: `camdl list` shows them. Writes nothing.

Examples:
  camdl status
  camdl status --store /data/camdl-store
"))]
pub struct StatusArgs {
    #[command(flatten)]
    pub store: StoreRootArgs,
}

#[derive(Args, Debug, Clone)]
#[command(after_help = colored_help!("\
For each completed result, computes the run id again from its `identity.json`
and compares it with the folder's, and computes the SHA-256 of every file its
manifest lists and compares it with the recorded digest. Each result that fails
is reported as damaged, with the file or the id that differs; the next run that
asks for a damaged result moves it to `.quarantine/` and computes it again.
Writes nothing. Exits 0 when every result checked is whole and 1 when any is
damaged.

Examples:
  # Every completed result in the store
  camdl verify

  # One result, by run-id prefix
  camdl verify 318a6e1b

  # Every result of the invocation labelled `baseline`
  camdl verify @baseline
"))]
pub struct VerifyArgs {
    /// The result to check, by run-id prefix or folder, or `@LABEL` for every
    /// result of that invocation (default: every completed result)
    pub handle: Option<String>,

    #[command(flatten)]
    pub store: StoreRootArgs,
}

#[derive(Args, Debug, Clone)]
#[command(after_help = colored_help!("\
Deletes:
  - each staging folder (`.staging/`) of an unfinished atomic result, or
    folder a claim was being built in, whose owner was a process on this host
    that has stopped
  - everything in `.quarantine/`, where the store moves the results `--force`
    replaces, damaged results, and folders it cannot read
  - every `.attempts/` folder: the records an interrupted or failed run left
    before it was run again
  - each group and build directory that is empty

It closes the record of each invocation still running whose process on this
host has stopped, as interrupted, so `status` no longer lists it. Staging whose
owner is on another host is left and listed. It deletes no result other than
those in `.quarantine/`, and never touches a completed result, a running or
interrupted run, a run recorded as failed, or the index. A completed result
leaves the store only when `--force` has moved it to `.quarantine/`.

Examples:
  camdl clean
  camdl clean --store /data/camdl-store
"))]
pub struct CleanArgs {
    #[command(flatten)]
    pub store: StoreRootArgs,
}

#[derive(Args, Debug, Clone)]
#[command(after_help = colored_help!("\
Reads every `identity.json`, `run.json` and invocation record in the run store,
writes the index of completed results and ended invocations as one log, and
removes the index logs of owners known to have stopped. The index is a cache: a
result is found without it, so `reindex` is useful after results were copied
in or removed by hand. `camdl dev reindex` rebuilds the index of
`results/`, the store of `fit` and `profile`.

Examples:
  camdl reindex
  camdl reindex --store /data/camdl-store
"))]
pub struct StoreReindexArgs {
    #[command(flatten)]
    pub store: StoreRootArgs,
}
