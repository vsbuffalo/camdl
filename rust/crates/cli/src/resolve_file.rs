//! Finds a stored file from a handle or an output manifest and a file's name:
//! the one lookup `camdl path`, `camdl cat` and `camdl show` share. The
//! commands differ only in what they print. The rules are
//! `docs/camdl-run-spec.md` §4.5, and the design
//! `docs/dev/proposals/2026-11-23-workflow-bindings.md`, "Looking up a stored
//! file".
//!
//! A [`Source`] names the candidate results ([`candidates`]): for an output
//! manifest, each entry in the manifest's order; for a label, each completed
//! result its invocation listed, in that order; for a run id, a prefix or a
//! folder, the one result and then the by-products stored for it, and, for a
//! name qualified by `model_source`, the model source of the model the result
//! ran ([`with_source`]). [`Narrow`]
//! keeps those of one kind or whose run id begins with a prefix. A name is
//! looked up among them by rules tried in order: a name qualified by a kind
//! (`fit_run:draws`) among that kind's results, a logical name (run spec
//! §2.9), a stored name (`draws.tsv`, or a record, `identity.json` or
//! `run.json`), and, for a handle, a path through a result nested in a
//! candidate's folder (`predict/forecast-<id>/forecast/cases`). The first
//! rule any candidate answers decides. A run id's own result is searched by
//! every rule before its by-products are, so a name the result holds is
//! never ambiguous with one of theirs. A name more than one candidate
//! answers is refused unless every holder was asked for ([`resolve_each`]).
//!
//! A listed result the store holds but cannot serve, damaged or unfinished,
//! is set aside rather than failing the lookup ([`Unavailable`]). It fails a
//! lookup only when it could answer the name: when it is of the kind of a
//! result that does, so that serving the others would silently leave it
//! out, or when nothing else answers.
//!
//! A result file named through a manifest is checked against the manifest:
//! the digest and size the store's `run.json` records must be the ones the
//! manifest records, so a path is printed, or bytes read, only for the file
//! the manifest describes. Provenance files and the two records are not in a
//! manifest, and are not checked against it. A manifest names only its own
//! entries, so no path through a nested result is looked up through one.
//! Reading a file ([`ResolvedFile::read`]) checks its bytes against the
//! digest `run.json` records.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};

use store::{Handle, Key, Located, Store};

use crate::output_manifest::{ManifestFile, OutputManifest};
use crate::result_files::Format;

/// The store's two records of a result, which every result holds beside its
/// files and which are named by their file names.
const RECORDS: [&str; 2] = ["identity.json", "run.json"];

/// What names the results a file is looked up among.
pub(crate) enum Source<'a> {
    /// A result's run id or a prefix of one, its folder, `@L` or `@L~n`.
    Handle(&'a Handle),
    /// An output manifest, read from `path`.
    Manifest { manifest: &'a OutputManifest, path: &'a Path },
}

impl Source<'_> {
    fn shown(&self) -> String {
        match self {
            Source::Handle(h) => format!("`{h}`"),
            Source::Manifest { path, .. } => path.display().to_string(),
        }
    }

    /// Whether the source lists several results by design: a manifest, or a
    /// label, which names an invocation. A run id or a folder names one
    /// result, with its by-products.
    pub(crate) fn lists_results(&self) -> bool {
        matches!(self, Source::Manifest { .. } | Source::Handle(Handle::Label { .. }))
    }
}

/// `--kind` and `--run`: the candidates of one kind, and those whose run id
/// begins with a prefix.
#[derive(Debug, Default, Clone)]
pub(crate) struct Narrow {
    pub kind: Option<String>,
    pub run: Option<String>,
}

impl Narrow {
    fn keeps(&self, kind: &str, run_id: Key) -> bool {
        self.kind.as_deref().is_none_or(|k| k == kind) && self.run.as_deref().is_none_or(|r| run_id.to_hex().starts_with(r))
    }

    fn shown(&self) -> String {
        let mut parts = Vec::new();
        if let Some(k) = &self.kind {
            parts.push(format!("--kind {k}"));
        }
        if let Some(r) = &self.run {
            parts.push(format!("--run {r}"));
        }
        parts.join(" ")
    }

    fn is_empty(&self) -> bool {
        self.kind.is_none() && self.run.is_none()
    }
}

/// One result a source names.
#[derive(Clone)]
pub(crate) struct Candidate {
    pub located: Located,
    /// The labels the manifest or the invocation record gives the result;
    /// `null` when they give none.
    pub labels: serde_json::Value,
    /// For a manifest's entry, the files it records, by logical name.
    promised: Option<BTreeMap<String, ManifestFile>>,
    /// The provenance files `run.json` records, by stored name.
    provenance: Vec<String>,
}

/// A result a source lists that the store holds but cannot serve: damaged,
/// unfinished, or gone.
pub(crate) struct Unavailable {
    kind: String,
    run_id: Key,
    why: String,
}

impl Unavailable {
    /// The sentence a refusal gives it: what it is, why it is not read, and
    /// for a by-product the flag that stores it again.
    fn sentence(&self) -> String {
        let again = by_product_flag(&self.kind)
            .map(|flag| format!("; run the command again with {flag} and --force to store it again"))
            .unwrap_or_default();
        format!("The {} {} is not read: {}{again}", self.kind, self.run_id.short12(), self.why)
    }
}

/// The candidate results of a lookup.
pub(crate) struct Candidates {
    /// Those the store serves, in the source's order.
    found: Vec<Candidate>,
    /// Those it holds but cannot serve.
    unavailable: Vec<Unavailable>,
    /// Whether `found[0]` is the result a run id, a prefix or a folder
    /// names, searched by every rule before the rest.
    named_first: bool,
}

/// One name a candidate answers to.
#[derive(Clone)]
struct Named {
    logical: String,
    stored: String,
    /// What the manifest records of the file, for a manifest's result file.
    promised: Option<ManifestFile>,
}

impl Candidate {
    fn new(located: Located, labels: serde_json::Value, promised: Option<BTreeMap<String, ManifestFile>>) -> Result<Candidate, FileRefusal> {
        let provenance =
            located.provenance_files().map_err(|e| FileRefusal::Store(e.to_string()))?.keys().map(|s| s.to_string()).collect();
        Ok(Candidate { located, labels, promised, provenance })
    }

    fn logical_of(&self, stored: &str) -> String {
        crate::result_files::contract(self.located.kind()).and_then(|c| c.logical(stored)).unwrap_or_else(|| stored.to_string())
    }

    /// The result files and provenance files, by logical name, and the two
    /// records. A manifest's result names its result files as the manifest
    /// records them.
    fn names(&self) -> Vec<Named> {
        let mut names: Vec<Named> = match &self.promised {
            Some(files) => {
                files.iter().map(|(l, f)| Named { logical: l.clone(), stored: f.file.clone(), promised: Some(f.clone()) }).collect()
            }
            None => self
                .located
                .result_files()
                .keys()
                .map(|s| Named { logical: self.logical_of(s), stored: s.to_string(), promised: None })
                .collect(),
        };
        names.extend(self.provenance.iter().map(|s| Named { logical: self.logical_of(s), stored: s.clone(), promised: None }));
        names.extend(RECORDS.iter().map(|r| Named { logical: r.to_string(), stored: r.to_string(), promised: None }));
        names
    }

    /// The result files, by logical name in name order, with their stored
    /// names.
    fn result_names(&self) -> Vec<Named> {
        let results = self.located.result_files();
        let mut names: Vec<Named> =
            self.names().into_iter().filter(|n| n.promised.is_some() || results.contains_key(n.stored.as_str())).collect();
        names.sort_by(|a, b| a.logical.cmp(&b.logical));
        names
    }

    /// Its result files, as (logical name, stored name), by logical name in
    /// name order.
    pub(crate) fn file_names(&self) -> Vec<(String, String)> {
        self.result_names().into_iter().map(|n| (n.logical, n.stored)).collect()
    }

    /// The logical names of its result and provenance files, in name order.
    fn listed(&self) -> Vec<String> {
        let mut v: Vec<String> = self.names().into_iter().filter(|n| !RECORDS.contains(&n.stored.as_str())).map(|n| n.logical).collect();
        v.sort();
        v.dedup();
        v
    }
}

/// A stored file a name resolved to.
pub(crate) struct ResolvedFile {
    /// The result that holds the file.
    pub located: Located,
    /// The labels of the candidate the name was found through.
    pub labels: serde_json::Value,
    /// The logical name, or a record's file name.
    pub logical: String,
    /// The name inside the result's folder.
    pub stored: String,
    /// The file's absolute path.
    pub path: PathBuf,
    /// The digest `run.json` records; `None` for a record.
    pub digest: Option<Key>,
    /// The format the result-file contract gives the file; records are JSON.
    pub format: Option<Format>,
}

impl ResolvedFile {
    /// The file's bytes. A result or provenance file is read through the
    /// store and checked against the digest `run.json` records; a record has
    /// no digest and is read as it lies.
    pub(crate) fn read(&self) -> Result<Vec<u8>, FileRefusal> {
        match self.digest {
            Some(_) => self.located.read(&self.stored).map_err(|e| FileRefusal::Store(e.to_string())),
            None => read_record(&self.path),
        }
    }
}

#[expect(clippy::disallowed_methods, reason = "reads one of the store's own records of a result, which have no digest to check, for display")]
fn read_record(path: &Path) -> Result<Vec<u8>, FileRefusal> {
    std::fs::read(path).map_err(|e| FileRefusal::Store(format!("reading {}: {e}", path.display())))
}

/// Why a file could not be named.
#[derive(Debug)]
pub(crate) enum FileRefusal {
    /// The handle names no result in the store.
    NoMatch,
    /// The store could not name or read a result: a prefix that names
    /// several, a result that did not complete or is damaged, a manifest's
    /// run id this store does not hold.
    Store(String),
    /// `--kind` and `--run` kept none of the source's results.
    Narrowed { source: String, narrow: String, results: Vec<String> },
    /// No candidate holds the name. `holders` gives each candidate and its
    /// file names, and `notes` each listed result that is stored but cannot
    /// be read, and why a path through a nested result was not looked up.
    NoSuchName { name: String, source: String, holders: Vec<(String, Vec<String>)>, notes: Vec<String> },
    /// `KIND:NAME` names a kind none of the candidates is.
    NoSuchKind { kind: String, source: String, kinds: Vec<String> },
    /// A result that could answer the name is stored but cannot be read, so
    /// an answer without it could be the wrong file, or miss one.
    Unreadable { name: String, why: String },
    /// Several candidates hold the name, where one is needed.
    Ambiguous { name: String, source: String, holders: Vec<String>, lists: bool },
    /// The source names several results, where one is needed. `hint`
    /// names a command that reads them together, where there is one.
    NotOneResult { source: String, results: Vec<String>, hint: Option<String> },
    /// The store records the file otherwise than the manifest does.
    ManifestDiffers { source: String, result: String, name: String, what: String },
    /// A result has no default file and holds several.
    NoDefault { result: String, files: Vec<String> },
    /// A label value holds a tab or a line break, which a TSV cell cannot.
    Untabular { what: String },
    /// `cat --each` of a file that is not a table, or of tables whose headers
    /// differ.
    NotOneTable(String),
}

impl fmt::Display for FileRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileRefusal::NoMatch => f.write_str("the handle names no result in the run store"),
            FileRefusal::Store(e) => f.write_str(e),
            FileRefusal::Narrowed { source, narrow, results } => {
                write!(f, "{narrow} keeps none of the results of {source}: {}", results.join(", "))
            }
            FileRefusal::NoSuchName { name, source, holders, notes } => {
                if let [(result, names)] = &holders[..] {
                    write!(f, "the {result} has no `{name}`; its files are {}", names.join(", "))?;
                } else {
                    let each: Vec<String> = holders.iter().map(|(r, names)| format!("{r}: {}", names.join(", "))).collect();
                    write!(f, "no result of {source} has `{name}`; their files are {}", each.join("; "))?;
                }
                for n in notes {
                    write!(f, ". {n}")?;
                }
                Ok(())
            }
            FileRefusal::NoSuchKind { kind, source, kinds } => {
                write!(f, "no result of {source} is a `{kind}`; its results are of the kinds {}", kinds.join(", "))
            }
            FileRefusal::Unreadable { name, why } => write!(f, "`{name}` cannot be resolved: {why}"),
            FileRefusal::Ambiguous { name, source, holders, lists } => {
                let how = if *lists { "pass --kind or --run for one, or --each for all" } else { "pass --kind or --run for one" };
                write!(f, "{} results of {source} have `{name}`: {}; {how}", holders.len(), holders.join(", "))
            }
            FileRefusal::NotOneResult { source, results, hint } => {
                write!(f, "{source} names {} results, where one is needed: {}; name one by its run id, or with `path` and `cat` by --kind or --run", results.len(), results.join(", "))?;
                if let Some(h) = hint {
                    write!(f, "; {h}")?;
                }
                Ok(())
            }
            FileRefusal::ManifestDiffers { source, result, name, what } => {
                write!(f, "{source} does not describe the stored file `{name}` of the {result}: {what}; run the command that wrote the manifest again")
            }
            FileRefusal::NoDefault { result, files } => {
                write!(f, "name a file of the {result}; its files are {}", files.join(", "))
            }
            FileRefusal::Untabular { what } => write!(f, "{what} holds a tab or a line break, which a TSV cell cannot hold"),
            FileRefusal::NotOneTable(why) => f.write_str(why),
        }
    }
}

/// A result as a refusal names it: its kind, or for a fit run its method,
/// and its run id's first 12 hex digits.
fn shown_result(store: &Store, located: &Located) -> String {
    if located.kind() == <crate::fit_store::FitRun as store::Artifact>::KIND {
        if let Ok(run) = store.stored_as::<crate::fit_store::FitRun>(located) {
            return format!("{} run {}", run.identity().chain.method.algorithm(), located.run_id().short12());
        }
    }
    format!("{} {}", located.kind(), located.run_id().short12())
}

/// The results `source` names, kept by `narrow`, in order.
pub(crate) fn candidates(store: &Store, source: &Source, narrow: &Narrow) -> Result<Candidates, FileRefusal> {
    let listed: Vec<(String, Key, serde_json::Value, Option<BTreeMap<String, ManifestFile>>)> = match source {
        Source::Manifest { manifest, .. } => {
            manifest.results.iter().map(|e| (e.kind.clone(), e.run_id, e.labels.clone(), Some(e.files.clone()))).collect()
        }
        Source::Handle(h @ Handle::Label { .. }) => {
            let inv = store.invocation(h).map_err(refusal)?;
            let mut seen = BTreeSet::new();
            inv.entries()
                .into_iter()
                .filter(|(k, id, completed)| *completed && seen.insert((k.to_string(), *id)))
                .map(|(k, id, _)| {
                    let labels = crate::output_manifest::manifest_labels(k, inv.attributes_of(k, id).unwrap_or(serde_json::Value::Null));
                    (k.to_string(), id, labels, None)
                })
                .collect()
        }
        Source::Handle(h) => return one_with_by_products(store, h, narrow),
    };
    let all: Vec<String> = listed.iter().map(|(k, id, ..)| format!("{k} {}", id.short12())).collect();
    let kept: Vec<_> = listed.into_iter().filter(|(k, id, ..)| narrow.keeps(k, *id)).collect();
    if kept.is_empty() {
        return Err(FileRefusal::Narrowed { source: source.shown(), narrow: narrow_or_source(narrow), results: all });
    }
    let mut out = Candidates { found: Vec::new(), unavailable: Vec::new(), named_first: false };
    for (kind, run_id, labels, promised) in kept {
        let located = match (store.locate(run_id), source) {
            (Ok(l), _) => l,
            // A manifest promises each entry, so one the store cannot serve
            // fails every lookup through it.
            (Err(e), Source::Manifest { path, .. }) => {
                return Err(FileRefusal::Store(format!(
                    "{} lists the {kind} {}, which the run store {} does not hold completed: {e}",
                    path.display(),
                    run_id.short12(),
                    store.root().display()
                )))
            }
            (Err(e), Source::Handle(_)) => {
                out.unavailable.push(Unavailable { kind, run_id, why: e.to_string() });
                continue;
            }
        };
        if located.kind() != kind {
            return Err(FileRefusal::ManifestDiffers {
                source: source.shown(),
                result: format!("{kind} {}", run_id.short12()),
                name: "identity.json".into(),
                what: format!("the store's result {} is a `{}`", run_id.short12(), located.kind()),
            });
        }
        out.found.push(Candidate::new(located, labels, promised)?);
    }
    Ok(out)
}

fn narrow_or_source(narrow: &Narrow) -> String {
    if narrow.is_empty() {
        "the source".into()
    } else {
        narrow.shown()
    }
}

/// The result a run id, a prefix or a folder names, then the by-products
/// stored for it, in the order the invocation records first list them; each
/// with the attributes the invocation that answers for it gives it
/// ([`store::Store::listings`]). An entry's
/// upstream is read from its identity record before the entry is located,
/// so a sweep's other cells are never checked.
fn one_with_by_products(store: &Store, handle: &Handle, narrow: &Narrow) -> Result<Candidates, FileRefusal> {
    let main = store.locate_handle(handle).map_err(|e| match e {
        store::Error::NotFound { .. } | store::Error::NoRecord { .. } => FileRefusal::NoMatch,
        e => FileRefusal::Store(not_completed(store, handle, e)),
    })?;
    let invocations = store.invocations();
    let latest_labels = |kind: &str, id: Key| {
        let labels = store
            .listings(&Handle::IdPrefix(id.to_hex()))
            .ok()
            .and_then(|l| l.answering()?.invocation().attributes_of(kind, id))
            .unwrap_or(serde_json::Value::Null);
        crate::output_manifest::manifest_labels(kind, labels)
    };
    let id = main.run_id();
    let main_kind = main.kind().to_string();
    let mut found = vec![Candidate::new(main, latest_labels(&main_kind, id), None)?];
    let mut unavailable = Vec::new();
    let mut seen = BTreeSet::from([id]);
    for inv in invocations.iter() {
        let entries = inv.entries();
        if !entries.iter().any(|(_, k, _)| *k == id) {
            continue;
        }
        for (kind, key, completed) in entries {
            if !completed || crate::result_files::is_content(kind) || kind == main_kind || seen.contains(&key) {
                continue;
            }
            if store.upstream_of(key).ok().flatten() != Some(id) {
                continue;
            }
            seen.insert(key);
            match store.locate(key) {
                Ok(located) => found.push(Candidate::new(located, latest_labels(kind, key), None)?),
                Err(e) => unavailable.push(Unavailable { kind: kind.to_string(), run_id: key, why: e.to_string() }),
            }
        }
    }
    let all: Vec<String> = found
        .iter()
        .map(|c| format!("{} {}", c.located.kind(), c.located.run_id().short12()))
        .chain(unavailable.iter().map(|u| format!("{} {}", u.kind, u.run_id.short12())))
        .collect();
    let named_first = narrow.keeps(&main_kind, id);
    found.retain(|c| narrow.keeps(c.located.kind(), c.located.run_id()));
    unavailable.retain(|u| narrow.keeps(&u.kind, u.run_id));
    if found.is_empty() && unavailable.is_empty() {
        return Err(FileRefusal::Narrowed { source: format!("`{handle}`"), narrow: narrow.shown(), results: all });
    }
    Ok(Candidates { found, unavailable, named_first })
}

/// The store's refusal of a label: [`FileRefusal::NoMatch`] for one no
/// invocation holds.
fn refusal(e: store::Error) -> FileRefusal {
    match e {
        store::Error::NotFound { .. } => FileRefusal::NoMatch,
        e => FileRefusal::Store(e.to_string()),
    }
}

/// A store's refusal of `handle`, or for a fit result that did not complete,
/// the refusal every fit reader gives it: the handle, the folder and the
/// state.
fn not_completed(store: &Store, handle: &Handle, e: store::Error) -> String {
    match e {
        store::Error::NotCompleted { .. } | store::Error::Damaged { .. } => {
            crate::fit_store::not_completed(store, &handle.to_string()).unwrap_or_else(|| e.to_string())
        }
        e => e.to_string(),
    }
}

/// The one result `source` and `narrow` select. A run id, a prefix or a
/// folder names one result; a manifest must hold one after `narrow`. A label
/// whose invocation listed several names the one that is neither a content
/// artifact, a fit problem nor a by-product, when there is one: the fit run
/// of a `fit run`, the cell of a one-cell `simulate`.
pub(crate) fn resolve_result(store: &Store, source: &Source, narrow: &Narrow) -> Result<Candidate, FileRefusal> {
    let Candidates { mut found, unavailable, named_first } = candidates(store, source, narrow)?;
    if named_first && narrow.is_empty() {
        return Ok(found.remove(0));
    }
    if found.is_empty() {
        let why = unavailable.iter().map(Unavailable::sentence).collect::<Vec<_>>().join(". ");
        return Err(FileRefusal::Store(format!("{} names no result the store can read. {why}", source.shown())));
    }
    if found.len() == 1 && unavailable.is_empty() {
        return Ok(found.remove(0));
    }
    let primary: Vec<usize> = (0..found.len()).filter(|&i| is_primary(&found[i].located)).collect();
    if let (Source::Handle(Handle::Label { .. }), [i]) = (source, &primary[..]) {
        if !unavailable.iter().any(|u| u.kind == found[*i].located.kind()) {
            return Ok(found.swap_remove(*i));
        }
    }
    let hint = match source {
        Source::Handle(h @ Handle::Label { .. }) => {
            let kinds: BTreeSet<&str> = primary.iter().map(|&i| found[i].located.kind()).collect();
            let tabled = ["sim", "pfilter", "profile_point"];
            (primary.len() > 1 && kinds.len() == 1 && kinds.iter().all(|k| tabled.contains(k)))
                .then(|| format!("`camdl cat {h}` renders their table"))
        }
        _ => None,
    };
    Err(FileRefusal::NotOneResult {
        source: source.shown(),
        results: found
            .iter()
            .map(|c| shown_result(store, &c.located))
            .chain(unavailable.iter().map(|u| format!("{} {} (not readable)", u.kind, u.run_id.short12())))
            .collect(),
        hint,
    })
}

fn is_primary(located: &Located) -> bool {
    located.upstream().is_none()
        && !crate::result_files::is_content(located.kind())
        && located.kind() != <crate::fit_store::FitProblem as store::Artifact>::KIND
}

/// For a handle and a name qualified by `model_source`
/// (`model_source:render`), `cands` with the model source of the model the
/// named result ran added last, as [`crate::camdl_store::shown_for`] finds
/// it. A label whose invocation stored the source lists it among its results
/// already, and so does a manifest. An unqualified name is looked up among
/// the result and its by-products alone.
fn with_source(store: &Store, source: &Source, name: &str, mut cands: Candidates) -> Result<Candidates, FileRefusal> {
    use crate::camdl_store::model_source::{shown_for, Shown};
    let kind = <crate::camdl_store::ModelSource as store::Artifact>::KIND;
    let qualified = name.split_once(':').is_some_and(|(k, _)| k == kind);
    let Source::Handle(handle) = source else { return Ok(cands) };
    if !qualified || cands.found.iter().any(|c| c.located.kind() == kind) {
        return Ok(cands);
    }
    let Some(model) = named_model(store, handle, &cands) else { return Ok(cands) };
    let shown = shown_for(store, handle, model).map_err(FileRefusal::Store)?;
    if let Some(n) = shown.notice(handle) {
        crate::out::live().notice(&n);
    }
    match shown {
        Shown::Recorded { source, .. } => {
            let key = source.id().key();
            match store.locate(key) {
                Ok(located) => cands.found.push(Candidate::new(located, serde_json::Value::Null, None)?),
                Err(e) => cands.unavailable.push(Unavailable { kind: kind.to_string(), run_id: key, why: e.to_string() }),
            }
        }
        Shown::Unreadable { run_id, why } => {
            cands.unavailable.push(Unavailable { kind: kind.to_string(), run_id, why });
        }
        Shown::NotRecorded { .. } => {}
    }
    Ok(cands)
}

/// The compiled model the result `handle` names ran, by run id: for a
/// label, the one model its invocation's primary results all ran; for a run
/// id, a prefix or a folder, that result's. `None` when there is no such
/// result or model.
fn named_model(store: &Store, handle: &Handle, cands: &Candidates) -> Option<Key> {
    match handle {
        Handle::Label { .. } => {
            let primary: Vec<&Located> = cands.found.iter().map(|c| &c.located).filter(|l| is_primary(l)).collect();
            let models: BTreeSet<Key> = primary.iter().filter_map(|l| crate::camdl_store::model_source::model_of(store, l)).collect();
            match &models.into_iter().collect::<Vec<_>>()[..] {
                [model] => Some(*model),
                _ => None,
            }
        }
        _ => {
            let located = store.locate_handle(handle).ok()?;
            crate::camdl_store::model_source::model_of(store, &located)
        }
    }
}

/// The stored file `name` names among the results `source` and `narrow`
/// select. A name more than one of them holds is refused with the holders.
pub(crate) fn resolve_file(store: &Store, source: &Source, narrow: &Narrow, name: &str) -> Result<ResolvedFile, FileRefusal> {
    let cands = with_source(store, source, name, candidates(store, source, narrow)?)?;
    let mut found = lookup(store, source, &cands, name)?;
    if found.len() == 1 {
        return Ok(found.remove(0));
    }
    Err(FileRefusal::Ambiguous {
        name: name.into(),
        source: source.shown(),
        holders: found.iter().map(|f| shown_result(store, &f.located)).collect(),
        lists: source.lists_results(),
    })
}

/// Every result among those `source` and `narrow` select that holds `name`,
/// in the source's order, with the file it holds under that name.
pub(crate) fn resolve_each(store: &Store, source: &Source, narrow: &Narrow, name: &str) -> Result<Vec<ResolvedFile>, FileRefusal> {
    let cands = with_source(store, source, name, candidates(store, source, narrow)?)?;
    lookup(store, source, &cands, name)
}

/// The file a result's command prints with no name: its kind's summary file
/// (run spec §4.5), else its one result file.
pub(crate) fn default_file(store: &Store, source: &Source, narrow: &Narrow) -> Result<ResolvedFile, FileRefusal> {
    let c = resolve_result(store, source, narrow)?;
    let result_names = c.result_names();
    let named = match default_stored(c.located.kind()).and_then(|d| result_names.iter().find(|n| n.stored == d)) {
        Some(n) => n.clone(),
        None => match &result_names[..] {
            [one] => one.clone(),
            _ => return Err(FileRefusal::NoDefault { result: shown_result(store, &c.located), files: c.listed() }),
        },
    };
    file_of(store, source, &c, &named)
}

/// Every result file of the one result `source` and `narrow` select, by
/// logical name in name order.
pub(crate) fn every_file(store: &Store, source: &Source, narrow: &Narrow) -> Result<Vec<ResolvedFile>, FileRefusal> {
    let c = resolve_result(store, source, narrow)?;
    c.result_names().iter().map(|n| file_of(store, source, &c, n)).collect()
}

/// The stored name of the file `camdl cat` prints for a result of `kind`
/// given no name.
fn default_stored(kind: &str) -> Option<&'static str> {
    use store::Artifact;
    use crate::{fit_store as fs, pfilter_store as ps, profile_store as pr, sim_store as ss};
    Some(match kind {
        k if k == ss::Sim::KIND => ss::TRAJ_FILE,
        k if k == ps::Pfilter::KIND => ps::LOGLIK_FILE,
        k if k == fs::FitRun::KIND => "fit_state.toml",
        k if k == fs::FitProblem::KIND => fs::PROBLEM_FILE,
        k if k == fs::Prediction::KIND => "report.json",
        k if k == fs::Score::KIND => "summary.tsv",
        k if k == pr::ProfilePoint::KIND => pr::MLE_FILE,
        _ => return None,
    })
}

/// `name` among `cands`, by the rules of the module documentation: the
/// named result first, for a run id; then the rest.
fn lookup(store: &Store, source: &Source, cands: &Candidates, name: &str) -> Result<Vec<ResolvedFile>, FileRefusal> {
    let (kind, bare) = match name.split_once(':') {
        Some((kind, rest)) if crate::result_files::contract(kind).is_some() => (Some(kind), rest),
        _ => (None, name),
    };
    let of_kind = |k: &str| kind.is_none_or(|want| want == k);
    let pool: Vec<&Candidate> = cands.found.iter().filter(|c| of_kind(c.located.kind())).collect();
    let unavailable: Vec<&Unavailable> = cands.unavailable.iter().filter(|u| of_kind(&u.kind)).collect();
    if let Some(k) = kind {
        if pool.is_empty() && unavailable.is_empty() {
            let mut kinds: Vec<String> = cands.found.iter().map(|c| c.located.kind().to_string()).collect();
            kinds.dedup();
            return Err(FileRefusal::NoSuchKind { kind: k.into(), source: source.shown(), kinds });
        }
    }
    let named = cands.named_first && pool.first().is_some_and(|c| std::ptr::eq(*c, &cands.found[0]));
    let tiers: Vec<&[&Candidate]> = if named { vec![&pool[..1], &pool[1..]] } else { vec![&pool[..]] };
    for tier in tiers {
        let hits = hits_in(store, source, tier, bare);
        if hits.is_empty() {
            continue;
        }
        // A listed result that cannot be read and is of a holder's kind
        // could hold the name too: an answer without it is not the answer.
        if let Some(u) = unavailable.iter().find(|u| hits.iter().any(|(c, _)| c.located.kind() == u.kind)) {
            return Err(FileRefusal::Unreadable { name: name.into(), why: u.sentence() });
        }
        return hits.into_iter().map(|(c, n)| file_of(store, source, &c, &n)).collect();
    }
    let mut notes: Vec<String> = unavailable.iter().map(|u| u.sentence()).collect();
    if matches!(source, Source::Manifest { .. }) && bare.contains('/') {
        notes.push("A manifest names only its own entries, so no result nested in one is looked up through it".into());
    }
    Err(FileRefusal::NoSuchName {
        name: name.into(),
        source: source.shown(),
        holders: pool.iter().map(|c| (shown_result(store, &c.located), c.listed())).collect(),
        notes,
    })
}

/// The candidates of `tier` that answer `bare` under the first rule any of
/// them answers: its logical name, its stored name, then, for a handle, a
/// path through a result nested in its folder.
fn hits_in(store: &Store, source: &Source, tier: &[&Candidate], bare: &str) -> Vec<(Candidate, Named)> {
    let by: [fn(&Named, &str) -> bool; 2] = [|n, s| n.logical == s, |n, s| n.stored == s];
    for matches in by {
        let hits: Vec<(Candidate, Named)> = tier
            .iter()
            .filter_map(|c| c.names().into_iter().find(|n| matches(n, bare)).map(|n| ((*c).clone(), n)))
            .collect();
        if !hits.is_empty() {
            return hits;
        }
    }
    match source {
        Source::Handle(_) => nested_hits(store, tier, bare),
        Source::Manifest { .. } => Vec::new(),
    }
}

/// The flag that asks for a by-product of `kind`.
fn by_product_flag(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "obs" => "--obs or --obs-dir",
        "event_log" => "--event-log",
        "sim_quantities" => "--quantities-out",
        "trace" => "--trace",
        "pf_health" => "--pf-health",
        "filtering" => "--save-filtering",
        "filtered_streams" => "--save-filtered-streams",
        "final_state" => "--save-final-state",
        "saved_paths" => "--save-paths",
        "prequential" => "--save-prequential",
        _ => return None,
    })
}

/// How many results deep a nested result may sit below the candidate it is
/// named through: a prediction sits one below its fit run, and its scores
/// one below the prediction.
const MAX_NESTING_DEPTH: usize = 4;

/// `name` as a path through a result nested in a candidate's folder, such
/// as a prediction inside its fit run: `predict/<mode>-<id>/` then a name of
/// the nested result. The path's folders must be plain names, never `..`,
/// `.`, empty or absolute, and the nested result's identity must name the
/// candidate as its parent, or as an ancestor within
/// [`MAX_NESTING_DEPTH`] results, so no path reaches a result outside the
/// candidate however the folders are laid out.
fn nested_hits(store: &Store, pool: &[&Candidate], name: &str) -> Vec<(Candidate, Named)> {
    let mut hits = Vec::new();
    for c in pool {
        for (at, _) in name.match_indices('/') {
            let (dir, rest) = (&name[..at], &name[at + 1..]);
            if dir.split('/').any(|part| part.is_empty()) || !Path::new(dir).components().all(|p| matches!(p, Component::Normal(_))) {
                continue;
            }
            let folder = c.located.folder().join(dir);
            if !folder.join(RECORDS[0]).is_file() {
                continue;
            }
            let Ok(located) = store.locate_handle(&Handle::Path(folder)) else { continue };
            if !descends_from(store, &located, c.located.run_id()) {
                continue;
            }
            let Ok(nested) = Candidate::new(located, c.labels.clone(), None) else { continue };
            let names = nested.names();
            if let Some(n) = names.iter().find(|n| n.logical == rest).or_else(|| names.iter().find(|n| n.stored == rest)) {
                let n = n.clone();
                hits.push((nested, n));
                break;
            }
        }
    }
    hits
}

/// Whether `located`'s identity names `ancestor` as its parent, or as the
/// parent of a parent, within [`MAX_NESTING_DEPTH`] results.
fn descends_from(store: &Store, located: &Located, ancestor: Key) -> bool {
    let mut parent = located.parent();
    for _ in 0..MAX_NESTING_DEPTH {
        match parent {
            Some(p) if p == ancestor => return true,
            Some(p) => parent = store.locate(p).ok().and_then(|l| l.parent()),
            None => return false,
        }
    }
    false
}

/// The file `named` of the candidate `c`, checked against what the manifest
/// records of it.
fn file_of(store: &Store, source: &Source, c: &Candidate, named: &Named) -> Result<ResolvedFile, FileRefusal> {
    let located = &c.located;
    let differs = |what: String| FileRefusal::ManifestDiffers {
        source: source.shown(),
        result: shown_result(store, located),
        name: named.logical.clone(),
        what,
    };
    let (path, digest) = if RECORDS.contains(&named.stored.as_str()) {
        (located.folder().join(&named.stored), None)
    } else {
        match located.file(&named.stored) {
            Ok((p, d)) => (p, Some(d)),
            Err(e) if named.promised.is_some() => return Err(differs(format!("the store holds no `{}` ({e})", named.stored))),
            Err(e) => return Err(FileRefusal::Store(e.to_string())),
        }
    };
    if let Some(p) = &named.promised {
        match located.result_files().get(named.stored.as_str()) {
            None => return Err(differs(format!("the store's `{}` is not a result file", named.stored))),
            Some((sha, bytes)) if *sha != p.sha256 || *bytes != p.bytes => {
                return Err(differs(format!(
                    "the store records sha256 {} and {bytes} bytes, and the manifest sha256 {} and {} bytes",
                    sha.to_hex(),
                    p.sha256.to_hex(),
                    p.bytes
                )))
            }
            Some(_) => {}
        }
    }
    let format = if digest.is_none() {
        Some(Format::Json)
    } else {
        crate::result_files::contract(located.kind()).and_then(|k| k.format(&named.stored))
    };
    Ok(ResolvedFile {
        located: located.clone(),
        labels: c.labels.clone(),
        logical: named.logical.clone(),
        stored: named.stored.clone(),
        path: std::path::absolute(&path).unwrap_or(path),
        digest,
        format,
    })
}

// ── `--each` tables ──────────────────────────────────────────────────────────

/// Appends one column per swept parameter, `values` in the order the design
/// names them. A parameter whose name another label column already has is
/// written `param.<name>`.
pub(crate) fn push_swept(columns: &mut Vec<(String, String)>, values: &[(String, f64)]) {
    for (name, v) in values {
        let col = if columns.iter().any(|(c, _)| c == name) { format!("param.{name}") } else { name.clone() };
        columns.push((col, v.to_string()));
    }
}

/// A result's labels as `--each` columns: its kind's rule for a cell and a
/// filter run, and otherwise each field under its name, a scalar as its
/// text and anything else as compact JSON.
fn label_columns(kind: &str, labels: &serde_json::Value) -> Vec<(String, String)> {
    use store::Artifact;
    if kind == crate::sim_store::Sim::KIND {
        if let Ok(l) = serde_json::from_value::<crate::sim_store::CellAttributes>(labels.clone()) {
            return l.columns();
        }
    }
    if kind == crate::pfilter_store::Pfilter::KIND {
        if let Ok(l) = serde_json::from_value::<crate::pfilter_store::PfilterAttributes>(labels.clone()) {
            return l.columns();
        }
    }
    let Some(map) = labels.as_object() else { return Vec::new() };
    map.iter()
        .map(|(k, v)| {
            let text = match v {
                serde_json::Value::Null => String::new(),
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Bool(_) | serde_json::Value::Number(_) => v.to_string(),
                _ => v.to_string(),
            };
            (k.clone(), text)
        })
        .collect()
}

/// The rows' label columns: the union of the files' label keys in the order
/// they are first seen, each row with an empty value where its result lacks
/// a key. A key that `taken` holds is written `label.<key>`.
fn label_table(files: &[ResolvedFile], taken: &[&str]) -> Result<(Vec<String>, Vec<Vec<String>>), FileRefusal> {
    let per_row: Vec<Vec<(String, String)>> = files.iter().map(|f| label_columns(f.located.kind(), &f.labels)).collect();
    let mut keys: Vec<String> = Vec::new();
    for row in &per_row {
        for (k, _) in row {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
    }
    let mut rows = Vec::with_capacity(per_row.len());
    for (f, row) in files.iter().zip(&per_row) {
        let mut cells = Vec::with_capacity(keys.len());
        for k in &keys {
            let v = row.iter().find(|(c, _)| c == k).map(|(_, v)| v.clone()).unwrap_or_default();
            if v.contains(['\t', '\n', '\r']) {
                return Err(FileRefusal::Untabular { what: format!("the label `{k}` of the {} {}", f.located.kind(), f.located.run_id().short12()) });
            }
            cells.push(v);
        }
        rows.push(cells);
    }
    let header = keys.into_iter().map(|k| if taken.contains(&k.as_str()) { format!("label.{k}") } else { k }).collect();
    Ok((header, rows))
}

/// `camdl path --each`: a TSV with one row per file, `run_id` (64 hex
/// digits), the label columns, then `path`.
pub(crate) fn path_table(files: &[ResolvedFile]) -> Result<String, FileRefusal> {
    let (labels, rows) = label_table(files, &["run_id", "path"])?;
    let mut out = String::new();
    let header: Vec<&str> = std::iter::once("run_id").chain(labels.iter().map(String::as_str)).chain(std::iter::once("path")).collect();
    out.push_str(&header.join("\t"));
    out.push('\n');
    for (f, cells) in files.iter().zip(rows) {
        let path = f.path.display().to_string();
        if path.contains(['\t', '\n', '\r']) {
            return Err(FileRefusal::Untabular { what: format!("the path {path}") });
        }
        let line: Vec<String> = std::iter::once(f.located.run_id().to_hex()).chain(cells).chain(std::iter::once(path)).collect();
        out.push_str(&line.join("\t"));
        out.push('\n');
    }
    Ok(out)
}

/// `camdl cat --each`: the rows of every file as one TSV, each row after its
/// result's label columns, under one header. Each file is read through the
/// store and checked against its digest; leading `#` comment lines are
/// dropped. A file that is not a table, and a file whose header differs from
/// the first file's, are refused before anything is returned.
pub(crate) fn cat_table(files: &[ResolvedFile]) -> Result<Vec<u8>, FileRefusal> {
    let mut tables: Vec<(String, Vec<String>)> = Vec::with_capacity(files.len());
    for f in files {
        if f.format != Some(Format::Tsv) {
            let format = f.format.map_or("not a table", Format::name);
            return Err(FileRefusal::NotOneTable(format!(
                "`{}` of the {} {} is {format}; cat --each reads TSV files only",
                f.logical,
                f.located.kind(),
                f.located.run_id().short12()
            )));
        }
        let bytes = f.read()?;
        let text = String::from_utf8(bytes)
            .map_err(|_| FileRefusal::NotOneTable(format!("`{}` of {} is not UTF-8 text", f.logical, f.located.run_id().short12())))?;
        let mut lines = text.lines().skip_while(|l| l.starts_with('#'));
        let header = lines.next().ok_or_else(|| {
            FileRefusal::NotOneTable(format!("`{}` of {} has no header row", f.logical, f.located.run_id().short12()))
        })?;
        tables.push((header.to_string(), lines.map(str::to_string).collect()));
    }
    let first = &tables[0].0;
    if let Some((i, (h, _))) = tables.iter().enumerate().find(|(_, (h, _))| h != first) {
        let f = &files[i];
        return Err(FileRefusal::NotOneTable(format!(
            "the {} {} has a `{}` whose header differs from the first's: `{}` here, `{}` in {}",
            f.located.kind(),
            f.located.run_id().short12(),
            f.logical,
            h.replace('\t', " "),
            first.replace('\t', " "),
            files[0].located.run_id().short12()
        )));
    }
    let columns: Vec<&str> = first.split('\t').collect();
    let (labels, label_rows) = label_table(files, &columns)?;
    let mut out = String::new();
    let header: Vec<&str> = labels.iter().map(String::as_str).chain(columns.iter().copied()).collect();
    out.push_str(&header.join("\t"));
    out.push('\n');
    for ((_, rows), cells) in tables.iter().zip(label_rows) {
        let prefix = cells.join("\t");
        for row in rows {
            if !cells.is_empty() {
                out.push_str(&prefix);
                out.push('\t');
            }
            out.push_str(row);
            out.push('\n');
        }
    }
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests;
