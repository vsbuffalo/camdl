//! Where a running fit method writes its files.
//!
//! A method names each file it writes by its path within the run
//! (`chain_0/trace.tsv`) and is handed an [`OutFile`] to append to; it is
//! never given a folder to write in. In `fit run` the writer is a [`LiveRun`]:
//! each file is a file of the store's in-place run, written in the run's own
//! folder as the method produces it, so a reader of the store sees a chain's
//! trace grow while the fit runs. A name opened twice is refused, except
//! through [`RunWriter::append`], which a resumed chain uses to extend the
//! files carried from its base.
//!
//! A method that reads back what it wrote (a chain's trace, read again to
//! compute its diagnostics; `fit_state.toml`, read for the run's report)
//! reads it from [`RunWriter::folder`], where every byte written through an
//! [`OutFile`] is already on disk.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::fit_store::FitRun;
use crate::result_files::fit_run::{FitOutcome, PlannedMethod};
use crate::result_files::{ParseFile, ResultFile, Role};

/// The files of one run of a fit method, opened by name.
pub(crate) trait RunWriter: Send + Sync {
    /// Opens `name`, a path within the run, as a new, empty file. Refused when
    /// `name` was opened already.
    fn create(&self, name: &str) -> Result<OutFile, String>;
    /// The file `name` opened before, such as a chain file carried from a
    /// resume's base, to append to; or a new, empty file when it was not.
    fn append(&self, name: &str) -> Result<OutFile, String>;
    /// Whether `name` has been opened.
    fn opened(&self, name: &str) -> bool;
    /// The folder the files written so far are read back from.
    fn folder(&self) -> &Path;

    /// Writes `bytes` as the whole of `name`, replacing what it held: for a
    /// file written once, or rewritten whole as the method learns more.
    fn put(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        if self.opened(name) {
            self.append(name)?.replace(bytes)
        } else {
            self.create(name)?.write_all(bytes)
        }
    }
}

/// One open file of a run. Clones write to the same file, so it can be
/// shared between a chain's threads.
#[derive(Clone)]
pub(crate) struct OutFile(Arc<dyn Append>);

/// What an [`OutFile`] writes to: a file of the store's run, or, in tests, a
/// file in a directory.
trait Append: Send + Sync {
    fn append(&self, bytes: &[u8]) -> Result<(), String>;
    fn replace(&self, bytes: &[u8]) -> Result<(), String>;
}

impl Append for store::FileHandle {
    fn append(&self, bytes: &[u8]) -> Result<(), String> {
        self.write_all(bytes).map_err(|e| format!("{}: {e}", self.name()))
    }
    fn replace(&self, bytes: &[u8]) -> Result<(), String> {
        store::FileHandle::replace(self, bytes).map_err(|e| format!("{}: {e}", self.name()))
    }
}

impl OutFile {
    /// Appends `bytes`.
    pub(crate) fn write_all(&self, bytes: &[u8]) -> Result<(), String> {
        self.0.append(bytes)
    }

    /// Replaces the whole content with `bytes`, so a reader sees the old
    /// content or the new and never part of either.
    pub(crate) fn replace(&self, bytes: &[u8]) -> Result<(), String> {
        self.0.replace(bytes)
    }
}

impl std::io::Write for OutFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        OutFile::write_all(self, buf).map_err(std::io::Error::other)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The run a fit method writes into: the store's in-place run, held until
/// the method ends, and the files opened in it so far.
///
/// A file whose name is one of a fit run's files ([`crate::result_files::fit_run::FitRunFile`])
/// is opened in that file's role; any other name is opened as an attempt
/// file, outside the manifest, and recorded, so a completed method that
/// wrote one is refused by name ([`LiveRun::declare`]) and a failed one
/// names it in its reason.
pub(crate) struct LiveRun {
    run: Arc<Mutex<Option<store::Run<FitRun>>>>,
    folder: PathBuf,
    files: Mutex<BTreeMap<String, store::FileHandle>>,
    stray: Mutex<Vec<String>>,
}

impl LiveRun {
    /// The writer of `run`, an in-place run, which it holds from now on.
    pub(crate) fn new(run: store::Run<FitRun>) -> Arc<LiveRun> {
        let folder = run.folder().expect("a fit run is written in its own folder").to_path_buf();
        Arc::new(LiveRun {
            run: Arc::new(Mutex::new(Some(run))),
            folder,
            files: Mutex::default(),
            stray: Mutex::default(),
        })
    }

    /// Takes the run back, for `finish` or `fail`; `None` once taken. Every
    /// later open is refused.
    pub(crate) fn take(&self) -> Option<store::Run<FitRun>> {
        self.run.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Calls `f` on the run while it is held.
    pub(crate) fn with<T>(&self, f: impl FnOnce(&mut store::Run<FitRun>) -> T) -> Option<T> {
        self.run.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(f)
    }

    /// The names opened as attempt files because no fit run's file has them.
    pub(crate) fn stray(&self) -> Vec<String> {
        self.stray.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Declares the files the run holds: those `plan`, the method named
    /// `method`, stores whatever its outcome, with a starts file of extension
    /// `starts`, and those its outcome, as the run's own records state it
    /// ([`FitOutcome::read`]), adds. A completed method that wrote a file
    /// outside that set is refused here, by name; a declared file the method
    /// did not write fails the run at `finish`.
    pub(crate) fn declare(&self, method: &str, plan: &PlannedMethod, n_chains: u32, starts: Option<&str>) -> Result<(), String> {
        let outcome = FitOutcome::read(&self.folder);
        let declared = plan.declared(n_chains, starts, &outcome);
        let written: Vec<String> = self.files.lock().unwrap_or_else(|e| e.into_inner()).keys().cloned().collect();
        let outside = written.iter().find(|name| !declared.iter().any(|f| &f.stored() == *name));
        if let Some(name) = outside {
            let mut names: Vec<String> = declared.iter().map(ResultFile::logical).collect();
            names.sort();
            return Err(format!(
                "the {} method wrote `{name}`, which is not among this run's files: {}",
                method,
                names.join(", ")
            ));
        }
        self.with(|run| run.expect(crate::result_files::expected(declared)))
            .ok_or_else(|| "the run has already ended".to_string())
    }

    fn open(&self, name: &str) -> Result<store::FileHandle, String> {
        let role = crate::result_files::fit_run::FitRunFile::of_stored(name).map(|f| f.role());
        let opened = self.with(|run| {
            match role {
                Some(Role::Result) => run.result_file(name),
                Some(Role::Provenance) => run.provenance_file(name),
                None => run.attempt_file(name),
            }
            .map_err(|e| format!("{name}: {e}"))
        });
        let handle = opened.ok_or_else(|| format!("{name}: the run has already ended"))??;
        if role.is_none() {
            self.stray.lock().unwrap_or_else(|e| e.into_inner()).push(name.to_string());
        }
        self.files.lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string(), handle.clone());
        Ok(handle)
    }
}

impl RunWriter for LiveRun {
    fn create(&self, name: &str) -> Result<OutFile, String> {
        Ok(OutFile(Arc::new(self.open(name)?)))
    }

    fn append(&self, name: &str) -> Result<OutFile, String> {
        let held = self.files.lock().unwrap_or_else(|e| e.into_inner()).get(name).cloned();
        match held {
            Some(h) => Ok(OutFile(Arc::new(h))),
            None => self.create(name),
        }
    }

    fn opened(&self, name: &str) -> bool {
        self.files.lock().unwrap_or_else(|e| e.into_inner()).contains_key(name)
    }

    fn folder(&self) -> &Path {
        &self.folder
    }
}

/// A file of a [`DirWriter`].
#[cfg(test)]
struct DirFile {
    file: Mutex<std::fs::File>,
    path: PathBuf,
}

#[cfg(test)]
impl Append for DirFile {
    fn append(&self, bytes: &[u8]) -> Result<(), String> {
        use std::io::Write as _;
        self.file.lock().unwrap().write_all(bytes).map_err(|e| format!("{}: {e}", self.path.display()))
    }
    fn replace(&self, bytes: &[u8]) -> Result<(), String> {
        store::write_file_atomic(&self.path, bytes).map_err(|e| e.to_string())?;
        *self.file.lock().unwrap() =
            std::fs::OpenOptions::new().append(true).open(&self.path).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// A [`RunWriter`] over a plain directory, for the tests of a method's
/// writers.
#[cfg(test)]
pub(crate) struct DirWriter {
    dir: PathBuf,
    files: Mutex<BTreeMap<String, OutFile>>,
}

#[cfg(test)]
impl DirWriter {
    pub(crate) fn new(dir: &Path) -> DirWriter {
        DirWriter { dir: dir.to_path_buf(), files: Mutex::default() }
    }
}

#[cfg(test)]
impl RunWriter for DirWriter {
    fn create(&self, name: &str) -> Result<OutFile, String> {
        let mut files = self.files.lock().unwrap();
        if files.contains_key(name) {
            return Err(format!("{name} was opened already"));
        }
        let path = self.dir.join(name);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        let f = std::fs::File::create(&path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let out = OutFile(Arc::new(DirFile { file: Mutex::new(f), path }));
        files.insert(name.to_string(), out.clone());
        Ok(out)
    }

    fn append(&self, name: &str) -> Result<OutFile, String> {
        let held = self.files.lock().unwrap().get(name).cloned();
        match held {
            Some(f) => Ok(f),
            None => {
                let path = self.dir.join(name);
                if path.is_file() {
                    let f = std::fs::OpenOptions::new().append(true).open(&path).map_err(|e| e.to_string())?;
                    let out = OutFile(Arc::new(DirFile { file: Mutex::new(f), path }));
                    self.files.lock().unwrap().insert(name.to_string(), out.clone());
                    Ok(out)
                } else {
                    self.create(name)
                }
            }
        }
    }

    fn opened(&self, name: &str) -> bool {
        self.files.lock().unwrap().contains_key(name) || self.dir.join(name).is_file()
    }

    fn folder(&self) -> &Path {
        &self.dir
    }
}
