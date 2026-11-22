//! The store's commands, a clap subcommand set a project adds to its own.
//! Each works from the index and the records, across every registered kind.

use std::io::Write;
use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::handle::Handle;
use crate::invocation::all_invocations;
use crate::maintenance::walk;
use crate::owner::Liveness;
use crate::record::{read_record, InvocationStatus, Read, RunFile, RUN_FILE};
use crate::storage::{folder_state, State, Store};

/// The store's commands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum StoreCommand {
    /// What results of a kind exist, with status, the labels of the
    /// invocations that asked for them, and time.
    List { kind: String },
    /// What produced a result: identity record, dependencies, files, and the
    /// invocations that asked for it.
    Show { handle: String },
    /// Which runs and batches are running, failed or interrupted.
    Status,
    /// Whether each folder's id matches its identity.json and each file its
    /// digest: every completed result, the one an id prefix or a path names,
    /// or every result of the invocation a label names.
    Verify { handle: Option<String> },
    /// Deletes dead owners' staging, everything in quarantine, every earlier
    /// attempt, and empty group and build directories.
    Clean,
    /// Rebuilds and compacts the index from the records.
    Reindex,
}

/// What a command found wrong in the store. Only `verify` finds anything;
/// every other command returns [`Verdict::Sound`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Sound,
    /// `verify` found this many results damaged.
    Damaged { results: usize },
}

fn io(e: std::io::Error) -> Error {
    Error::Io { op: "write the output of", path: PathBuf::from("a store command"), source: e }
}

impl StoreCommand {
    /// Runs the command against `store`, writing its report to `out`.
    pub fn run(&self, store: &Store, out: &mut dyn Write) -> Result<Verdict> {
        let inner = &store.inner;
        match self {
            StoreCommand::List { kind } => {
                if !inner.kinds.has(kind) {
                    return Err(Error::UnknownKind { kind: kind.clone() });
                }
                let invs = all_invocations(&inner.root);
                for (dir, f) in walk(inner).into_iter().filter(|(_, f)| &f.kind == kind) {
                    let state = match folder_state(&dir, &f.run_id, &inner.kinds) {
                        State::Completed { .. } => "completed",
                        State::Running { .. } | State::HalfMade { .. } | State::Claim { .. } => "running",
                        State::Failed { .. } => "failed",
                        State::Damaged { .. } => "damaged",
                        State::Absent => continue,
                    };
                    let labels: Vec<&str> = invs
                        .iter()
                        .filter(|i| i.results.iter().any(|r| r.run_id == f.run_id))
                        .filter_map(|i| i.label.as_deref())
                        .collect();
                    let time = match read_record::<RunFile>(&dir.join(RUN_FILE)) {
                        Read::Ok(r) => r.finished.unwrap_or(r.started).0,
                        _ => String::new(),
                    };
                    writeln!(out, "{}  {state:<9}  {time}  {}", f.run_id.short12(), labels.join(",")).map_err(io)?;
                }
            }
            StoreCommand::Show { handle } => {
                let h: Handle = handle.parse()?;
                let dir = match &h {
                    Handle::Path(p) => p.clone(),
                    _ => {
                        let inv = store.invocation(&h)?;
                        let key = match &h {
                            Handle::IdPrefix(p) => {
                                walk(inner).into_iter().find(|(_, f)| f.run_id.to_hex().starts_with(p.as_str())).map(|(_, f)| f.run_id)
                            }
                            _ => inv.file.results.first().map(|r| r.run_id),
                        }
                        .ok_or_else(|| Error::NotFound { handle: handle.clone() })?;
                        walk(inner).into_iter().find(|(_, f)| f.run_id == key).map(|(d, _)| d).ok_or(Error::NotFound { handle: handle.clone() })?
                    }
                };
                let ident = std::fs::read_to_string(dir.join(crate::record::IDENTITY_FILE)).map_err(|e| Error::Io { op: "read", path: dir.clone(), source: e })?;
                let run = std::fs::read_to_string(dir.join(RUN_FILE)).map_err(|e| Error::Io { op: "read", path: dir.clone(), source: e })?;
                writeln!(out, "folder: {}\n\nidentity.json:\n{ident}\n\nrun.json:\n{run}", dir.display()).map_err(io)?;
                let id: crate::record::IdentityFile = read_record(&dir.join(crate::record::IDENTITY_FILE)).required(&dir)?;
                if let Read::Ok(run) = read_record::<RunFile>(&dir.join(RUN_FILE)) {
                    let own = inner.ambient_key_of(&id.ambient).map(|k| k.to_hex()).unwrap_or_default();
                    for d in run.deps.iter().filter(|d| d.build != own) {
                        writeln!(out, "cross-build dependency: {} {} was made by build {}", d.kind, d.run_id.short12(), &d.build[..d.build.len().min(12)]).map_err(io)?;
                    }
                }
                for i in all_invocations(&inner.root).iter().filter(|i| i.results.iter().any(|r| r.run_id == id.run_id)) {
                    writeln!(out, "asked for by invocation {}{}", i.seq, i.label.as_ref().map(|l| format!(" (@{l})")).unwrap_or_default()).map_err(io)?;
                }
            }
            StoreCommand::Status => {
                for i in all_invocations(&inner.root).iter().filter(|i| i.status == InvocationStatus::Running) {
                    let l = inner.liveness(&i.owner, std::time::Duration::ZERO);
                    let state = if l == Liveness::Dead { "interrupted" } else { "running" };
                    writeln!(out, "invocation {}  {state}  {} results  {}", i.seq, i.results.len(), i.owner).map_err(io)?;
                }
                for (dir, f) in walk(inner) {
                    match folder_state(&dir, &f.run_id, &inner.kinds) {
                        State::Running { owner, .. } => {
                            let l = inner.liveness(&owner, std::time::Duration::ZERO);
                            let state = if l == Liveness::Dead { "interrupted" } else { "running" };
                            writeln!(out, "{} {}  {state}  {owner}", f.kind, f.run_id.short12()).map_err(io)?;
                        }
                        State::Failed { failure, .. } => {
                            let why = failure.and_then(|f| f.reason).unwrap_or_else(|| "no reason recorded".into());
                            writeln!(out, "{} {}  failed  {why}", f.kind, f.run_id.short12()).map_err(io)?;
                        }
                        _ => {}
                    }
                }
            }
            StoreCommand::Verify { handle } => {
                let only = match handle {
                    None => None,
                    Some(text) => {
                        let h: Handle = text.parse()?;
                        let dirs = match h {
                            Handle::Path(p) if crate::storage::identity_of(&p).is_none() => return Err(Error::NoRecord { path: p }),
                            Handle::Path(p) => vec![p],
                            Handle::IdPrefix(prefix) => {
                                let mut found: Vec<_> =
                                    walk(inner).into_iter().filter(|(_, f)| f.run_id.to_hex().starts_with(prefix.as_str())).collect();
                                found.sort_by_key(|(_, f)| f.run_id);
                                found.dedup_by_key(|(_, f)| f.run_id);
                                match found.len() {
                                    0 => return Err(Error::NotFound { handle: text.clone() }),
                                    1 => found.into_iter().map(|(d, _)| d).collect(),
                                    _ => return Err(Error::Ambiguous { handle: text.clone(), matches: found.into_iter().map(|(_, f)| f.run_id).collect() }),
                                }
                            }
                            label @ Handle::Label { .. } => {
                                let inv = store.invocation(&label)?;
                                let keys: Vec<_> = inv.file.results.iter().map(|r| r.run_id).collect();
                                let dirs: Vec<_> =
                                    walk(inner).into_iter().filter(|(_, f)| keys.contains(&f.run_id)).map(|(d, _)| d).collect();
                                if dirs.is_empty() {
                                    return Err(Error::NotFound { handle: text.clone() });
                                }
                                dirs
                            }
                        };
                        Some(dirs)
                    }
                };
                let failures = store.verify(only.as_deref())?;
                for f in &failures {
                    writeln!(out, "damaged: {} ({}): {}", f.folder.display(), f.run_id.short12(), f.what).map_err(io)?;
                }
                if failures.is_empty() {
                    writeln!(out, "every result verified").map_err(io)?;
                } else {
                    let mut damaged: Vec<_> = failures.iter().map(|f| f.run_id).collect();
                    damaged.sort();
                    damaged.dedup();
                    return Ok(Verdict::Damaged { results: damaged.len() });
                }
            }
            StoreCommand::Clean => {
                let c = store.clean()?;
                writeln!(
                    out,
                    "removed {} staging folders, {} quarantined folders, {} earlier attempts, {} empty directories",
                    c.staging.len(),
                    c.quarantine.len(),
                    c.attempts.len(),
                    c.directories.len()
                )
                .map_err(io)?;
                for p in &c.other_hosts {
                    writeln!(out, "left {}: staged by a process on another host", p.display()).map_err(io)?;
                }
            }
            StoreCommand::Reindex => {
                let n = store.reindex()?;
                writeln!(out, "indexed {n} results").map_err(io)?;
            }
        }
        Ok(Verdict::Sound)
    }
}
