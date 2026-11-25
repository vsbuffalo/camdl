//! A model source: the model file one command compiled, with its views and
//! the quantities its compile produced. The views are what the compiler
//! shows of the file and no computation reads: its `#'` documentation and
//! description, the model as declared, and, for a `.camdl` file, the model's
//! mathematics (`camdlc render --format json`) and its compartment flow graph
//! (`camdlc render --format graph`).
//!
//! A model source is a content artifact of several files, keyed by their
//! bytes, the file's name and the quantities it depends on. No compiled model
//! depends on it, so editing a comment re-keys no simulation or fit. The
//! command that compiles a model stores the quantities, the model source and
//! the compiled model, and records in its invocation record that the compiled
//! model was made from that source ([`put_compiled`], through
//! `RunInfo::record_source`). Two files that differ only in their comments
//! thus share one compiled model, and each invocation records its own source.
//! A command that stores a result read from a stored one, such as a
//! prediction from a stored fit, compiles nothing: it lists the stored
//! compiled model in its own record with the source the result's invocation
//! recorded, or the file it was given ([`link_source`]).
//!
//! A reader finds a result's source by one rule ([`shown_for`]): the source
//! the invocation that answers for its handle (`store::Store::listings`)
//! recorded for the result's compiled model ([`model_of`]). No other
//! invocation stands in, so an invocation that recorded none gives
//! [`Shown::NotRecorded`].

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use store::{Artifact, ContentFiles, Dep, Handle, Hashed, Id, Layout, Placed, Primary, Provenance, Store, Stored, WriteMode};

use super::{CompiledModel, CompiledModelContent, ModelAttributes, Quantities, QuantitiesContent};

/// One file of a model source, as bytes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SourceFile {
    pub bytes: Vec<u8>,
}

impl store::Content for SourceFile {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        Ok(SourceFile { bytes: bytes.to_vec() })
    }
}

fn hashed(bytes: Vec<u8>) -> Hashed<SourceFile> {
    Hashed::new(SourceFile { bytes })
}

/// A model source's identity: the file's name, the digest of each of its
/// files, and the quantities its compile produced.
#[derive(store::Identity, Clone)]
#[identity(tag = "model_source", version = 1)]
pub(crate) struct ModelSourceIdentity {
    /// The model file's name as given, such as `sir.camdl`: `camdlc render`
    /// writes its stem into the views, so two files of one text and two
    /// names have two sources.
    pub file_name: String,
    /// The model file as written: `source.camdl`, or `source.ir.json` for a
    /// model given as compiled IR.
    pub file: Hashed<SourceFile>,
    /// True when `file` is a `.camdl` file.
    pub camdl: bool,
    /// The quantities and contrasts the file's compile produced, which a
    /// prediction from a result of this source evaluates.
    pub quantities: Dep<Quantities>,
    /// `docs.json`: the model's description and its `#'` documentation
    /// ([`SourceDocs`]).
    pub docs: Hashed<SourceFile>,
    /// `declared.json`: the model as its author declared it.
    pub declared: Hashed<SourceFile>,
    /// `model.render.json`: the model's mathematics as KaTeX strings. Absent
    /// for compiled IR, and when camdlc could not render the file.
    pub render: Option<Hashed<SourceFile>>,
    /// `model.graph.json`: the compartment flow graph, absent as `render` is.
    pub graph: Option<Hashed<SourceFile>>,
}

/// The stored name of each file of a model source.
pub(crate) const SOURCE_CAMDL: &str = "source.camdl";
pub(crate) const SOURCE_IR: &str = "source.ir.json";
pub(crate) const DOCS: &str = "docs.json";
pub(crate) const DECLARED: &str = "declared.json";
pub(crate) const RENDER: &str = "model.render.json";
pub(crate) const GRAPH: &str = "model.graph.json";

/// The contents of `docs.json`: the model's `description` and its `#'`
/// documentation, as the envelope holds it.
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub(crate) struct SourceDocs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub docs: ir::ModelDocs,
}

/// A model source, stored once per distinct file, name, quantities and build.
pub(crate) struct ModelSource;

impl Artifact for ModelSource {
    const KIND: &'static str = "model_source";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ModelSourceIdentity;
    type Attributes = ModelAttributes;
}

impl Primary for ModelSource {
    fn layout(_id: &ModelSourceIdentity, attributes: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&attributes.name)
    }
}

impl ContentFiles for ModelSource {
    fn files(id: &ModelSourceIdentity) -> Vec<(String, Vec<u8>)> {
        let file = if id.camdl { SOURCE_CAMDL } else { SOURCE_IR };
        let mut files = vec![
            (file.to_string(), id.file.get().bytes.clone()),
            (DOCS.to_string(), id.docs.get().bytes.clone()),
            (DECLARED.to_string(), id.declared.get().bytes.clone()),
        ];
        files.extend(id.render.iter().map(|r| (RENDER.to_string(), r.get().bytes.clone())));
        files.extend(id.graph.iter().map(|g| (GRAPH.to_string(), g.get().bytes.clone())));
        files
    }
}

/// A `.camdl` file's mathematics and flow graph, as camdlc renders them.
/// Both are absent for compiled IR, which camdlc does not render, and each
/// is absent when its render failed.
pub(crate) struct Rendered {
    pub render: Option<Vec<u8>>,
    pub graph: Option<Vec<u8>>,
}

impl Rendered {
    /// Nothing rendered: for a file no camdlc reads, such as a test's.
    pub fn nothing() -> Self {
        Rendered { render: None, graph: None }
    }

    /// The renders of the model file at `path`. A `.camdl` is rendered twice
    /// by camdlc; a render that fails is reported once as a warning, with
    /// camdlc's error, and is absent. Compiled IR is not rendered, and the
    /// command says so once.
    pub fn of(path: &Path) -> Self {
        if !crate::util::model_is_camdl_source(&path.to_string_lossy()) {
            once(&NOT_RENDERED, || {
                crate::out::notice::Notice::info(
                    "model-not-rendered",
                    format!(
                        "{} is compiled IR, so its stored model source has no rendered mathematics or flow graph; \
                         pass the .camdl to store them",
                        path.display()
                    ),
                )
            });
            return Rendered::nothing();
        }
        let one = |format: &str| match crate::util::camdlc_render(path, format) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                once(&RENDER_FAILED, || crate::out::notice::Notice {
                    severity: crate::out::notice::Severity::Warning,
                    code: "model-render-failed",
                    message: format!(
                        "camdlc could not render {} (`camdlc render --format {format}`), so its stored model source \
                         has no {}: {e}",
                        path.display(),
                        if format == "graph" { "flow graph" } else { "rendered mathematics" }
                    ),
                });
                None
            }
        };
        Rendered { render: one("json"), graph: one("graph") }
    }
}

static NOT_RENDERED: AtomicBool = AtomicBool::new(false);
static RENDER_FAILED: AtomicBool = AtomicBool::new(false);

/// Says `notice` the first time this process reaches it: a command that
/// compiles one model twice (a fit's ordinary compile and its compile with
/// the state Jacobian) reports once.
fn once(said: &AtomicBool, notice: impl FnOnce() -> crate::out::notice::Notice) {
    if !said.swap(true, Ordering::Relaxed) {
        crate::out::live().notice(&notice());
    }
}

/// A model source before it is stored: its files, its name, and the
/// quantities its compile produced, which are stored first so the source can
/// depend on them ([`Self::put`]).
#[derive(Clone)]
pub(crate) struct SourceParts {
    file_name: String,
    file: Hashed<SourceFile>,
    camdl: bool,
    quantities: Hashed<QuantitiesContent>,
    docs: Hashed<SourceFile>,
    declared: Hashed<SourceFile>,
    render: Option<Hashed<SourceFile>>,
    graph: Option<Hashed<SourceFile>>,
}

impl SourceParts {
    /// The source of one compile: the file `file_name` holding `bytes`, the
    /// envelope's description, documentation and declared model, the
    /// renders, and the compile's quantities.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn of(
        file_name: String,
        bytes: &[u8],
        camdl: bool,
        description: Option<String>,
        docs: ir::ModelDocs,
        declared: &declared::DeclaredSection,
        rendered: Rendered,
        quantities: Hashed<QuantitiesContent>,
    ) -> Result<Self, String> {
        let pretty = |what: &str, r: serde_json::Result<Vec<u8>>| r.map_err(|e| format!("encoding the {what}: {e}"));
        Ok(SourceParts {
            file_name,
            file: hashed(bytes.to_vec()),
            camdl,
            quantities,
            docs: hashed(pretty("documentation", serde_json::to_vec_pretty(&SourceDocs { description, docs }))?),
            declared: hashed(pretty("declared model", serde_json::to_vec_pretty(declared))?),
            render: rendered.render.map(hashed),
            graph: rendered.graph.map(hashed),
        })
    }

    /// Stores the quantities, then the source that depends on them, under
    /// `attributes` in the invocation `info`.
    #[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
    pub fn put(self, store: &Store, attributes: &ModelAttributes, info: &store::RunInfo) -> store::Result<Stored<ModelSource>> {
        use store::ContentArtifact;
        let quantities = store.put::<Quantities>(self.quantities, attributes, &Provenance::none(), info)?.dep(Quantities::FILE)?;
        let identity = ModelSourceIdentity {
            file_name: self.file_name,
            file: self.file,
            camdl: self.camdl,
            quantities,
            docs: self.docs,
            declared: self.declared,
            render: self.render,
            graph: self.graph,
        };
        store.put::<ModelSource>(identity, attributes, &Provenance::none(), info)
    }
}

/// The source a compile is recorded with.
pub(crate) enum SourceOf {
    /// The model file the compile was read from, stored with it.
    New(SourceParts),
    /// One already stored: for a compile camdl wrote itself from a stored
    /// result, whose model file is the stored result's.
    Stored(Id<ModelSource>),
    /// None: a compile camdl wrote itself from a stored result that records
    /// no source.
    Unknown,
}

impl From<SourceParts> for SourceOf {
    fn from(p: SourceParts) -> Self {
        SourceOf::New(p)
    }
}

/// Stores the compiled model `model` under `attributes`, with `provenance`,
/// and records in `info` that it was made from `source`, storing that first
/// when it is new; returns the model to depend on.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn put_compiled(
    store: &Store,
    info: &store::RunInfo,
    model: Hashed<CompiledModelContent>,
    source: impl Into<SourceOf>,
    attributes: &ModelAttributes,
    provenance: &Provenance,
) -> store::Result<Dep<CompiledModel>> {
    use store::ContentArtifact;
    let source = source_id(store, info, source.into(), attributes)?;
    let dep = store.put::<CompiledModel>(model, attributes, provenance, info)?.dep(CompiledModel::FILE)?;
    if let Some(source) = source {
        info.record_source(dep.id(), &source)?;
    }
    Ok(dep)
}

/// The run id of `source`, stored under `attributes` first when it is new.
#[allow(clippy::result_large_err)] // as put_compiled's
fn source_id(store: &Store, info: &store::RunInfo, source: SourceOf, attributes: &ModelAttributes) -> store::Result<Option<Id<ModelSource>>> {
    Ok(match source {
        SourceOf::New(p) => Some(*p.put(store, attributes, info)?.id()),
        SourceOf::Stored(id) => Some(id),
        SourceOf::Unknown => None,
    })
}

/// The source an invocation recorded for the model of a stored result, as
/// [`shown_for`] finds it.
pub(crate) enum Shown {
    /// The source the answering invocation recorded, read.
    Recorded { source: Stored<ModelSource>, by: RecordedBy },
    /// The answering invocation, by its sequence number, recorded no source
    /// for the result's model. `None` when no invocation answers: the label
    /// names none, or no invocation listed the result.
    NotRecorded { invocation: Option<u64> },
    /// The answering invocation recorded a source that cannot be read.
    Unreadable { run_id: store::Key, why: String },
}

/// Where a [`Shown::Recorded`] source came from: the invocation it was read
/// from, or the model file this command was given.
pub(crate) enum RecordedBy {
    /// The invocation a label names.
    Label,
    /// The invocation that answers for a run id or a folder, by its sequence
    /// number. `others` holds each different source an earlier invocation
    /// that listed the result recorded for its model.
    Latest { invocation: u64, others: Vec<OtherSource> },
    /// No invocation: the model file this command was given, stored by it.
    Given,
}

/// A source an earlier invocation recorded for a result's model, with the
/// labels that name those invocations (`@a`); empty when none does.
pub(crate) struct OtherSource {
    pub run_id: store::Key,
    pub labels: Vec<String>,
}

/// The source of the model `model` (a compiled model's run id) that the
/// invocation answering for `handle` recorded ([`store::Listings::source`]):
/// for `@label`, the invocation the label names; for a run id, a prefix or a
/// folder, the newest that listed the result computed or reused. No other
/// invocation stands in: one that recorded none gives [`Shown::NotRecorded`].
/// For a run id or a folder, the sources earlier listings of the result
/// recorded are named beside it ([`RecordedBy::Latest`]).
pub(crate) fn shown_for(store: &Store, handle: &Handle, model: store::Key) -> Shown {
    let Ok(listings) = store.listings(handle) else { return Shown::NotRecorded { invocation: None } };
    let Some(seq) = listings.answering().map(|l| l.seq) else { return Shown::NotRecorded { invocation: None } };
    let labelled = |s: u64| -> Vec<String> {
        // A label names the newest invocation that holds it, which may be a
        // later one.
        listings
            .listed()
            .iter()
            .filter(|l| l.seq == s)
            .filter_map(|l| l.label.as_ref())
            .filter(|name| store.invocation(&Handle::Label { name: name.to_string(), back: 0 }).is_ok_and(|i| i.seq() == s))
            .map(|name| format!("@{name}"))
            .collect()
    };
    let by = |others: Vec<OtherSource>| match listings.by() {
        store::By::Label { .. } => RecordedBy::Label,
        store::By::Result { .. } => RecordedBy::Latest { invocation: seq, others },
    };
    let (id, by) = match listings.source::<CompiledModel, ModelSource>(&Id::from_key(model)) {
        store::Recorded::None => return Shown::NotRecorded { invocation: Some(seq) },
        store::Recorded::One { source, .. } => (source, by(Vec::new())),
        store::Recorded::Several { latest, others } => {
            let mut grouped: Vec<OtherSource> = Vec::new();
            for (s, id) in others {
                match grouped.iter_mut().find(|o| o.run_id == id.key()) {
                    Some(o) => o.labels.extend(labelled(s)),
                    None => grouped.push(OtherSource { run_id: id.key(), labels: labelled(s) }),
                }
            }
            (latest, by(grouped))
        }
    };
    match store.get(&id) {
        Ok(Some(source)) => Shown::Recorded { source, by },
        Ok(None) => Shown::Unreadable { run_id: id.key(), why: "it is not stored".into() },
        Err(e) => Shown::Unreadable { run_id: id.key(), why: e.to_string() },
    }
}

impl Shown {
    /// The run id of the recorded source, read or not.
    pub fn run_id(&self) -> Option<store::Key> {
        match self {
            Shown::Recorded { source, .. } => Some(source.id().key()),
            Shown::Unreadable { run_id, .. } => Some(*run_id),
            Shown::NotRecorded { .. } => None,
        }
    }

    /// What a reader that shows the model's documentation tells the user
    /// about it, for the result `handle` names: that no invocation recorded
    /// the model file, so the reader goes on without the documentation; or
    /// that the invocations that listed the result recorded different
    /// files, so the answering one's is shown. `None` otherwise. An
    /// unreadable source is reported by [`Self::restore`].
    pub fn notice(&self, handle: &Handle) -> Option<crate::out::notice::Notice> {
        use crate::out::notice::{Notice, Severity};
        let chosen = |seq: u64| match handle {
            Handle::Label { .. } => format!("the invocation {handle} names (invocation {seq})"),
            _ => format!("the latest invocation that listed it (invocation {seq})"),
        };
        match self {
            Shown::NotRecorded { invocation } => Some(Notice {
                severity: Severity::Warning,
                code: "model-source-not-recorded",
                message: match invocation {
                    Some(seq) => format!(
                        "{handle}: the model's documentation is not shown: {} recorded no model file for the \
                         model the result ran",
                        chosen(*seq)
                    ),
                    None => format!("{handle}: the model's documentation is not shown: no invocation recorded the model file"),
                },
            }),
            Shown::Recorded { by: RecordedBy::Latest { invocation, others }, .. } if !others.is_empty() => {
                let listed: Vec<String> = others
                    .iter()
                    .map(|o| match &o.labels[..] {
                        [] => o.run_id.short12().to_string(),
                        labels => format!("{} ({})", o.run_id.short12(), labels.join(", ")),
                    })
                    .collect();
                Some(Notice::info(
                    "model-source-one-of-several",
                    format!(
                        "{handle}: the model's documentation is the one of the model file {} ran; earlier invocations \
                         that listed the result ran other files (model sources {}), which their labels show",
                        chosen(*invocation),
                        listed.join(", ")
                    ),
                ))
            }
            _ => None,
        }
    }

    /// The source this records, as a compile camdl writes itself from the
    /// stored result is recorded with.
    pub fn source_of(&self) -> SourceOf {
        match self.run_id() {
            Some(key) => SourceOf::Stored(Id::from_key(key)),
            None => SourceOf::Unknown,
        }
    }

    /// Restores into `envelope` the description and the documentation of
    /// the recorded source. No result depends on a source, so one that
    /// cannot be read leaves the envelope without them and is reported as a
    /// warning naming `handle`; the command goes on.
    pub fn restore(&self, handle: &Handle, envelope: &mut ir::IrEnvelope) {
        let read = |source: &Stored<ModelSource>| -> Result<SourceDocs, String> {
            let bytes = super::read_stored(source, DOCS)?;
            serde_json::from_slice(&bytes).map_err(|e| format!("{DOCS}: {e}"))
        };
        let unreadable = match self {
            Shown::Recorded { source, .. } => match read(source) {
                Ok(docs) => {
                    envelope.model.description = docs.description;
                    envelope.docs = docs.docs;
                    return;
                }
                Err(e) => format!("{}: {e}", source.id().key().short12()),
            },
            Shown::Unreadable { run_id, why } => format!("{}: {why}", run_id.short12()),
            Shown::NotRecorded { .. } => return,
        };
        crate::out::live().notice(&crate::out::notice::Notice {
            severity: crate::out::notice::Severity::Warning,
            code: "model-source-unreadable",
            message: format!(
                "{handle}: the model's documentation is not shown, since its stored model source cannot be read: \
                 the model source {unreadable}"
            ),
        });
    }
}

/// The compiled model the stored result `located` was computed from: a
/// compiled model among its dependencies, else among theirs, nearest first,
/// then the same for the result it sits in. Itself, for a compiled model.
/// `None` when no such dependency is stored.
pub(crate) fn model_of(store: &Store, located: &store::Located) -> Option<store::Key> {
    let kind = <CompiledModel as Artifact>::KIND;
    if located.kind() == kind {
        return Some(located.run_id());
    }
    let mut seen = std::collections::BTreeSet::from([located.run_id()]);
    let mut queue: std::collections::VecDeque<store::Located> = std::collections::VecDeque::from([located.clone()]);
    while let Some(at) = queue.pop_front() {
        if let Some(dep) = at.deps().iter().find(|d| d.kind == kind) {
            return Some(dep.run_id);
        }
        let next = at.deps().iter().map(|d| d.run_id).chain(at.parent());
        for key in next {
            if seen.insert(key) {
                if let Ok(l) = store.locate(key) {
                    queue.push_back(l);
                }
            }
        }
    }
    None
}

/// Records in this invocation that the stored compiled model `model` was
/// run from `source`: a new source is stored, and the compiled model is
/// listed in this invocation, as reused, named `name`, with the source
/// recorded. Nothing else is stored, so a compiled model a different build
/// stored keeps its run id. For a command that runs a stored compile of a
/// model file it names, such as a profile of a stored fit, and for one that
/// stores a result read from a stored result, such as a prediction from a
/// stored fit, which records the source that result's invocation recorded
/// or the file it was given.
pub(crate) fn link_source(
    store: &Store,
    info: &store::RunInfo,
    model: &Dep<CompiledModel>,
    source: impl Into<SourceOf>,
    name: &str,
) -> Result<(), String> {
    let attributes = ModelAttributes { name: name.to_string() };
    let source = source_id(store, info, source.into(), &attributes).map_err(|e| format!("storing the model source: {e}"))?;
    store
        .list::<CompiledModel>(model.id(), &attributes, info)
        .map_err(|e| format!("the compiled model {}: {e}", model.id().key().short12()))?;
    if let Some(source) = source {
        info.record_source(model.id(), &source).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
/// A store holding a fit, a prediction, a profile and a simulation, each
/// labelled, for tests of the readers of stored results.
#[cfg(test)]
pub(crate) use tests::store_of_four;
