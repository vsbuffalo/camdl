//! The class checks: one loop that holds every declared input of a command to
//! the promise its class makes.
//!
//! - **Identity.** Varying the field changes a run id.
//! - **Expands.** A smaller set's members keep their run ids inside a larger set.
//! - **Adds.** The upstream's run id is unchanged and a by-product exists.
//! - **Execution.** Two values on a fixture that exercises the field give
//!   identical outputs, and the fixture shows the field was used.
//! - **Export.** A copy has the digest of the stored file it copies. A
//!   rendering, made twice from the same stored files, has the same bytes.
//! - **Render.** No stored file differs across two values.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::hash::Key;
use crate::inputs::{Class, FieldInfo};

/// How to run one command for the class checks. Written once per command by
/// the project.
pub trait InputHarness {
    type Invocation: Clone;
    /// The command with its base arguments.
    fn base(&self) -> Self::Invocation;
    /// The base with one field changed: for identity, a second value; for
    /// expands, a larger set containing the base's; for adds, the request
    /// switched on; for export and render, the option set.
    fn varied(&self, field: &FieldInfo) -> Self::Invocation;
    /// The fixture `exercised_by` names, at two values of the field.
    fn exercised(&self, field: &FieldInfo) -> (Self::Invocation, Self::Invocation);
    /// Whether a run of the fixture shows the field was used.
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool;
    /// Runs in a fresh store, or in `store` when given, and reads back what
    /// was stored.
    fn run(&self, invocation: &Self::Invocation, store: Option<&Path>) -> Result<Observed>;
}

/// What one run left behind, as the checks compare it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Observed {
    pub results: Vec<ObservedResult>,
    /// Each export file's digest.
    pub exports: BTreeMap<PathBuf, Key>,
    /// Each export that is a copy of a stored file, with the file it copies.
    /// An export not named here is a rendering.
    pub copies: BTreeMap<PathBuf, StoredFile>,
}

/// One file of a stored result: the result's run id and the file's name in
/// its folder.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StoredFile {
    pub run_id: Key,
    pub file: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedResult {
    pub kind: String,
    pub run_id: Key,
    /// Each result file's digest.
    pub files: BTreeMap<String, Key>,
    /// A by-product's upstream run id.
    pub upstream: Option<Key>,
    /// A content artifact: an input the command stored for its results to
    /// depend on (a compiled model, a data table), not a result of the
    /// computation.
    pub content: bool,
}

/// One field that broke its class's promise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassFailure {
    pub field: &'static str,
    pub class: Class,
    pub what: String,
}

/// The run ids of the results a computation produced: its own and its
/// by-products', never the content artifacts it stored as inputs. An input
/// that moves only a content artifact's id (a second data file under the
/// same identity) has not reached the result's identity.
fn ids(o: &Observed) -> BTreeSet<Key> {
    o.results.iter().filter(|r| !r.content).map(|r| r.run_id).collect()
}

fn files(o: &Observed) -> BTreeSet<(Key, String, Key)> {
    o.results.iter().flat_map(|r| r.files.iter().map(move |(n, d)| (r.run_id, n.clone(), *d))).collect()
}

/// The digest of the stored file `of` names, among a run's results.
fn stored_digest<'a>(o: &'a Observed, of: &StoredFile) -> Option<&'a Key> {
    o.results.iter().find(|r| r.run_id == of.run_id).and_then(|r| r.files.get(&of.file))
}

fn fresh_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(crate::fsutil::unique_name(&format!("store-class-check-{tag}")))
}

/// Checks every field of one command against its class. Each failure names
/// the field and its class.
pub fn check_input_classes<H: InputHarness>(fields: &[FieldInfo], harness: &H) -> std::result::Result<(), Vec<ClassFailure>> {
    let mut failures = Vec::new();
    let mut run = |inv: &H::Invocation, store: Option<&Path>, f: &FieldInfo| -> Option<Observed> {
        match harness.run(inv, store) {
            Ok(o) => Some(o),
            Err(e) => {
                failures.push(ClassFailure { field: f.name, class: f.class, what: format!("the run failed: {e}") });
                None
            }
        }
    };
    let mut found = Vec::new();
    for f in fields {
        let fail = |what: String| ClassFailure { field: f.name, class: f.class, what };
        match f.class {
            Class::Identity => {
                let (Some(a), Some(b)) = (run(&harness.base(), None, f), run(&harness.varied(f), None, f)) else { continue };
                if ids(&a) == ids(&b) {
                    found.push(fail("a second value left every run id unchanged".into()));
                }
            }
            Class::Expands => {
                let (Some(a), Some(b)) = (run(&harness.base(), None, f), run(&harness.varied(f), None, f)) else { continue };
                let lost: Vec<Key> = ids(&a).difference(&ids(&b)).copied().collect();
                if !lost.is_empty() {
                    found.push(fail(format!("{} member(s) of the smaller set changed run id in the larger", lost.len())));
                }
                if ids(&b).len() <= ids(&a).len() {
                    found.push(fail("the larger set produced no result the smaller did not".into()));
                }
            }
            Class::Adds => {
                let (Some(a), Some(b)) = (run(&harness.base(), None, f), run(&harness.varied(f), None, f)) else { continue };
                if !ids(&a).is_subset(&ids(&b)) {
                    found.push(fail("asking for the by-product changed the upstream's run id".into()));
                }
                let base_ids = ids(&a);
                if !b.results.iter().any(|r| r.upstream.is_some() && !base_ids.contains(&r.run_id)) {
                    found.push(fail("asking for the by-product stored none".into()));
                }
            }
            Class::Execution => {
                let (x, y) = harness.exercised(f);
                let (Some(a), Some(b)) = (run(&x, None, f), run(&y, None, f)) else { continue };
                if files(&a) != files(&b) || ids(&a) != ids(&b) || a.exports != b.exports {
                    found.push(fail("two values gave different outputs".into()));
                }
                if !harness.shows_use(f, &a) || !harness.shows_use(f, &b) {
                    found.push(fail("its fixture does not show the input was used".into()));
                }
            }
            Class::Export => {
                let root = fresh_root("export");
                let inv = harness.varied(f);
                let (Some(a), Some(b)) = (run(&inv, Some(&root), f), run(&inv, Some(&root), f)) else { continue };
                let _ = std::fs::remove_dir_all(&root);
                if a.exports.is_empty() {
                    found.push(fail("setting it wrote no export".into()));
                }
                for (path, digest) in &a.exports {
                    let shown = path.display();
                    match a.copies.get(path) {
                        Some(of) => match stored_digest(&a, of) {
                            None => found.push(fail(format!(
                                "the export {shown} copies `{}` of {}, which the run did not store",
                                of.file, of.run_id
                            ))),
                            Some(d) if d != digest => found.push(fail(format!(
                                "the export {shown} differs from the stored `{}` it copies",
                                of.file
                            ))),
                            Some(_) => {}
                        },
                        None if files(&a) != files(&b) => found.push(fail(format!(
                            "the stored files changed between the two renderings of {shown}"
                        ))),
                        None if b.exports.get(path) != Some(digest) => found.push(fail(format!(
                            "the export {shown} rendered again from the same stored files has other bytes"
                        ))),
                        None => {}
                    }
                }
            }
            Class::Render => {
                let (Some(a), Some(b)) = (run(&harness.base(), None, f), run(&harness.varied(f), None, f)) else { continue };
                if files(&a) != files(&b) || ids(&a) != ids(&b) {
                    found.push(fail("a second value changed a stored file".into()));
                }
            }
            Class::Store | Class::Recorded | Class::Skip => {}
        }
    }
    failures.extend(found);
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

#[cfg(test)]
mod tests;
