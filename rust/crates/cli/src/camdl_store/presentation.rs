//! A model's presentation: everything one compile yields that no computation
//! reads. It is the model file as written, its `#'` documentation and
//! description, the model as declared, and, for a `.camdl` source, the
//! model's mathematics (`camdlc render --format json`) and its compartment
//! flow graph (`camdlc render --format graph`).
//!
//! A presentation is a content artifact of several files, keyed by their
//! bytes. No result depends on it, so editing a comment re-keys nothing. The
//! command that compiles a model stores the presentation, then the compiled
//! model, whose labels in that command's invocation record name the
//! presentation ([`put_compiled`]). Two files that differ only in their
//! comments thus share one compiled model, and each invocation names its own
//! presentation. A command that stores a result read from a stored one, such
//! as a prediction from a stored fit, compiles nothing; it lists the stored
//! compiled model in its own record with the presentation the result's
//! invocation named ([`link_presentation`]). So every invocation that lists
//! a result of a model names the model's presentation.
//!
//! A reader finds a result's presentation by one rule ([`shown_for`]): the
//! invocation its handle names (for `@label`, that invocation; for a run id
//! or a folder, the latest that listed the result), and in it the entry for
//! the result's compiled model ([`model_of`]). No other invocation is read,
//! so a record that names none gives [`Shown::NotRecorded`]. The design is
//! `docs/dev/proposals/2026-11-24-model-presentation.md`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use store::{Artifact, ContentFiles, Dep, Handle, Hashed, Layout, Placed, Primary, Provenance, RunRef, Store, Stored, WriteMode};

use super::{CompiledModel, CompiledModelContent, ModelAttributes};

/// One file of a presentation, as bytes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PresentationFile {
    pub bytes: Vec<u8>,
}

impl store::Content for PresentationFile {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        Ok(PresentationFile { bytes: bytes.to_vec() })
    }
}

fn hashed(bytes: Vec<u8>) -> Hashed<PresentationFile> {
    Hashed::new(PresentationFile { bytes })
}

/// A presentation's identity: the digest of each of its files.
#[derive(store::Identity, Clone)]
#[identity(tag = "model_presentation", version = 1)]
pub(crate) struct PresentationIdentity {
    /// The model file as written: `source.camdl`, or `source.ir.json` for a
    /// model given as compiled IR.
    pub source: Hashed<PresentationFile>,
    /// True when `source` is a `.camdl` file.
    pub camdl: bool,
    /// `docs.json`: the model's description and its `#'` documentation
    /// ([`PresentedDocs`]).
    pub docs: Hashed<PresentationFile>,
    /// `declared.json`: the model as its author declared it.
    pub declared: Hashed<PresentationFile>,
    /// `model.render.json`: the model's mathematics as KaTeX strings. Absent
    /// for compiled IR, and when camdlc could not render the source.
    pub render: Option<Hashed<PresentationFile>>,
    /// `model.graph.json`: the compartment flow graph, absent as `render` is.
    pub graph: Option<Hashed<PresentationFile>>,
}

/// The stored name of each file of a presentation.
pub(crate) const SOURCE_CAMDL: &str = "source.camdl";
pub(crate) const SOURCE_IR: &str = "source.ir.json";
pub(crate) const DOCS: &str = "docs.json";
pub(crate) const DECLARED: &str = "declared.json";
pub(crate) const RENDER: &str = "model.render.json";
pub(crate) const GRAPH: &str = "model.graph.json";

/// The contents of `docs.json`: the model's `description` and its `#'`
/// documentation, as the envelope holds it.
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub(crate) struct PresentedDocs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(flatten)]
    pub docs: ir::ModelDocs,
}

/// A model's presentation, stored once per distinct content and build.
pub(crate) struct ModelPresentation;

impl Artifact for ModelPresentation {
    const KIND: &'static str = "model_presentation";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = PresentationIdentity;
    type Attributes = ModelAttributes;
}

impl Primary for ModelPresentation {
    fn layout(_id: &PresentationIdentity, labels: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&labels.name)
    }
}

impl ContentFiles for ModelPresentation {
    fn files(id: &PresentationIdentity) -> Vec<(String, Vec<u8>)> {
        let source = if id.camdl { SOURCE_CAMDL } else { SOURCE_IR };
        let mut files = vec![
            (source.to_string(), id.source.get().bytes.clone()),
            (DOCS.to_string(), id.docs.get().bytes.clone()),
            (DECLARED.to_string(), id.declared.get().bytes.clone()),
        ];
        files.extend(id.render.iter().map(|r| (RENDER.to_string(), r.get().bytes.clone())));
        files.extend(id.graph.iter().map(|g| (GRAPH.to_string(), g.get().bytes.clone())));
        files
    }
}

/// A `.camdl` source's mathematics and flow graph, as camdlc renders them.
/// Both are absent for compiled IR, which camdlc does not render, and each
/// is absent when its render failed.
pub(crate) struct Rendered {
    pub render: Option<Vec<u8>>,
    pub graph: Option<Vec<u8>>,
}

impl Rendered {
    /// Nothing rendered: for a source no camdlc reads, such as a test's.
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
                        "{} is compiled IR, so its stored presentation has no rendered mathematics or flow graph; \
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
                        "camdlc could not render {} (`camdlc render --format {format}`), so its stored presentation \
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

impl PresentationIdentity {
    /// The presentation of one compile: the source file, the envelope's
    /// description, documentation and declared model, and the renders.
    pub(super) fn of(
        source: &[u8],
        camdl: bool,
        description: Option<String>,
        docs: ir::ModelDocs,
        declared: &declared::DeclaredSection,
        rendered: Rendered,
    ) -> Result<Self, String> {
        let pretty = |what: &str, r: serde_json::Result<Vec<u8>>| r.map_err(|e| format!("encoding the {what}: {e}"));
        Ok(PresentationIdentity {
            source: hashed(source.to_vec()),
            camdl,
            docs: hashed(pretty("documentation", serde_json::to_vec_pretty(&PresentedDocs { description, docs }))?),
            declared: hashed(pretty("declared model", serde_json::to_vec_pretty(declared))?),
            render: rendered.render.map(hashed),
            graph: rendered.graph.map(hashed),
        })
    }
}

/// The presentation a compile is stored with.
pub(crate) enum Presentation {
    /// The presentation of the model file the compile was read from, stored
    /// with it.
    New(PresentationIdentity),
    /// One already stored, by run id (hex): for a compile camdl wrote itself
    /// from a stored result, whose model file is the stored result's.
    Stored(String),
    /// None: a compile camdl wrote itself from a stored result that names no
    /// presentation.
    Unknown,
}

impl From<PresentationIdentity> for Presentation {
    fn from(p: PresentationIdentity) -> Self {
        Presentation::New(p)
    }
}

/// Stores the compiled model `model`, whose labels in this invocation name
/// `presentation`, storing that first when it is new; returns the model to
/// depend on. The model's other labels and its provenance are `labels` and
/// `provenance`.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn put_compiled(
    store: &Store,
    info: &store::RunInfo,
    model: Hashed<CompiledModelContent>,
    presentation: impl Into<Presentation>,
    labels: &ModelAttributes,
    provenance: &Provenance,
) -> store::Result<Dep<CompiledModel>> {
    use store::ContentArtifact;
    let labels = ModelAttributes { presentation: presentation_run_id(store, info, presentation.into(), labels)?, ..labels.clone() };
    store.put::<CompiledModel>(model, &labels, provenance, info)?.dep(CompiledModel::FILE)
}

/// The run id (hex) of `presentation`, stored under `labels` first when it
/// is new.
#[allow(clippy::result_large_err)] // as put_compiled's
fn presentation_run_id(
    store: &Store,
    info: &store::RunInfo,
    presentation: Presentation,
    labels: &ModelAttributes,
) -> store::Result<Option<String>> {
    Ok(match presentation {
        Presentation::New(p) => Some(store.put::<ModelPresentation>(p, labels, &Provenance::none(), info)?.id().key().to_hex()),
        Presentation::Stored(hex) => Some(hex),
        Presentation::Unknown => None,
    })
}

/// The presentation an invocation recorded for the model of a stored result,
/// as [`shown_for`] finds it.
pub(crate) enum Shown {
    /// The presentation the chosen invocation names, read.
    Recorded { presentation: Stored<ModelPresentation>, by: RecordedBy },
    /// The chosen invocation, by its sequence number, names no presentation
    /// for the result's model: its record lists the model without one, or
    /// does not list the model. `None` when no invocation is chosen: the
    /// label names none, or no invocation listed the result.
    NotRecorded { invocation: Option<u64> },
    /// The chosen invocation names a presentation that cannot be read.
    Unreadable { run_id: String, why: String },
}

/// Where a [`Shown::Recorded`] presentation came from: the invocation it was
/// read from, or the model file this command was given.
pub(crate) enum RecordedBy {
    /// The invocation a label names.
    Label,
    /// The latest invocation that listed the result, by its sequence number.
    /// `others` holds each different presentation an earlier invocation that
    /// listed the result recorded for its model.
    Latest { invocation: u64, others: Vec<OtherPresentation> },
    /// No invocation: the model file this command was given, stored by it.
    Given,
}

/// A presentation an earlier invocation recorded for a result's model, with
/// the labels that name those invocations (`@a`); empty when none does.
pub(crate) struct OtherPresentation {
    pub run_id: store::Key,
    pub labels: Vec<String>,
}

/// The presentation of the model `model` (a compiled model's run id) that
/// the invocation `handle` names recorded for the stored result `result`:
///
/// - for `@label`, the invocation the label names;
/// - for a run id, a prefix or a folder, the latest invocation that listed
///   `result`;
///
/// and within that invocation, the entry for `model`. No other invocation is
/// read for the presentation: an invocation that names none gives
/// [`Shown::NotRecorded`]. For a run id or a folder, the presentations the
/// earlier invocations that listed `result` recorded are compared with it
/// ([`RecordedBy::Latest`]).
pub(crate) fn shown_for(store: &Store, handle: &Handle, result: store::Key, model: store::Key) -> Shown {
    let recorded = |inv: &store::Invocation| -> Option<String> {
        let labels = inv.attributes_of(<CompiledModel as Artifact>::KIND, model)?;
        serde_json::from_value::<ModelAttributes>(labels).ok()?.presentation
    };
    let (seq, hex, by) = match handle {
        Handle::Label { .. } => {
            let Ok(inv) = store.invocation(handle) else { return Shown::NotRecorded { invocation: None } };
            (inv.seq(), recorded(&inv), RecordedBy::Label)
        }
        Handle::IdPrefix(_) | Handle::Path(_) => {
            let listing: Vec<store::Invocation> =
                store.invocations().iter().filter(|inv| inv.entries().iter().any(|(_, k, _)| *k == result)).cloned().collect();
            let Some((latest, earlier)) = listing.split_last() else { return Shown::NotRecorded { invocation: None } };
            let hex = recorded(latest);
            let mut others: Vec<OtherPresentation> = Vec::new();
            for inv in earlier {
                let Some(run_id) = recorded(inv).filter(|o| Some(o) != hex.as_ref()).and_then(|o| store::Key::from_hex(&o).ok())
                else {
                    continue;
                };
                // A label names the latest invocation that holds it, which
                // may be a later one.
                let label = inv
                    .label()
                    .filter(|name| {
                        store.invocation(&Handle::Label { name: name.to_string(), back: 0 }).is_ok_and(|i| i.seq() == inv.seq())
                    })
                    .map(|name| format!("@{name}"));
                match others.iter_mut().find(|o| o.run_id == run_id) {
                    Some(o) => o.labels.extend(label),
                    None => others.push(OtherPresentation { run_id, labels: label.into_iter().collect() }),
                }
            }
            (latest.seq(), hex, RecordedBy::Latest { invocation: latest.seq(), others })
        }
    };
    let Some(hex) = hex else { return Shown::NotRecorded { invocation: Some(seq) } };
    match store.resolve::<ModelPresentation>(&RunRef::new(Handle::IdPrefix(hex.clone()))) {
        Ok(presentation) => Shown::Recorded { presentation, by },
        Err(e) => Shown::Unreadable { run_id: hex, why: e.to_string() },
    }
}

impl Shown {
    /// The run id (hex) of the recorded presentation, read or not.
    pub fn run_id(&self) -> Option<String> {
        match self {
            Shown::Recorded { presentation, .. } => Some(presentation.id().key().to_hex()),
            Shown::Unreadable { run_id, .. } => Some(run_id.clone()),
            Shown::NotRecorded { .. } => None,
        }
    }

    /// What a reader that shows the model's documentation tells the user
    /// about it, for the result `handle` names: that no invocation recorded
    /// the model file, so the reader goes on without the documentation; or
    /// that the invocations that listed the result recorded different
    /// files, so the latest one's is shown. `None` otherwise. An unreadable
    /// presentation is reported by [`Self::restore`].
    pub fn notice(&self, handle: &Handle) -> Option<crate::out::notice::Notice> {
        use crate::out::notice::{Notice, Severity};
        let chosen = |seq: u64| match handle {
            Handle::Label { .. } => format!("the invocation {handle} names (invocation {seq})"),
            _ => format!("the latest invocation that listed it (invocation {seq})"),
        };
        match self {
            Shown::NotRecorded { invocation } => Some(Notice {
                severity: Severity::Warning,
                code: "model-presentation-not-recorded",
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
                    "model-presentation-one-of-several",
                    format!(
                        "{handle}: the model's documentation is the one of the model file {} ran; earlier invocations \
                         that listed the result ran other files (presentations {}), which their labels show",
                        chosen(*invocation),
                        listed.join(", ")
                    ),
                ))
            }
            _ => None,
        }
    }

    /// The presentation this records, as a compile camdl writes itself from
    /// the stored result is stored with.
    pub fn presentation(&self) -> Presentation {
        match self.run_id() {
            Some(hex) => Presentation::Stored(hex),
            None => Presentation::Unknown,
        }
    }

    /// Restores into `envelope` the description and the documentation of
    /// the recorded presentation. No result depends on a presentation, so
    /// one that cannot be read leaves the envelope without them and is
    /// reported as a warning naming `handle`; the command goes on.
    pub fn restore(&self, handle: &Handle, envelope: &mut ir::IrEnvelope) {
        let read = |presentation: &Stored<ModelPresentation>| -> Result<PresentedDocs, String> {
            let bytes = super::read_stored(presentation, DOCS)?;
            serde_json::from_slice(&bytes).map_err(|e| format!("{DOCS}: {e}"))
        };
        let unreadable = match self {
            Shown::Recorded { presentation, .. } => match read(presentation) {
                Ok(docs) => {
                    envelope.model.description = docs.description;
                    envelope.docs = docs.docs;
                    return;
                }
                Err(e) => format!("{}: {e}", presentation.id().key().short12()),
            },
            Shown::Unreadable { run_id, why } => format!("{}: {why}", &run_id[..12.min(run_id.len())]),
            Shown::NotRecorded { .. } => return,
        };
        crate::out::live().notice(&crate::out::notice::Notice {
            severity: crate::out::notice::Severity::Warning,
            code: "model-presentation-unreadable",
            message: format!(
                "{handle}: the model's documentation is not shown, since its stored presentation cannot be read: \
                 the model's presentation {unreadable}"
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
/// run from the model file `presentation` presents: a new presentation is
/// stored, and the compiled model is listed in this invocation, as reused,
/// with labels that name it. Nothing else is stored, so a compiled model a
/// different build stored keeps its run id. For a command that runs
/// a stored compile of a model file it names, such as a profile of a stored
/// fit, and for one that stores a result read from a stored result, such as
/// a prediction from a stored fit, which names the presentation that
/// result's invocation recorded.
pub(crate) fn link_presentation(
    store: &Store,
    info: &store::RunInfo,
    model: &Dep<CompiledModel>,
    presentation: impl Into<Presentation>,
    name: &str,
) -> Result<(), String> {
    let labels = ModelAttributes { name: name.to_string(), presentation: None };
    let presentation = presentation_run_id(store, info, presentation.into(), &labels)
        .map_err(|e| format!("storing the model's presentation: {e}"))?;
    store
        .list::<CompiledModel>(model.id(), &ModelAttributes { presentation, ..labels }, info)
        .map_err(|e| format!("the compiled model {}: {e}", model.id().key().short12()))
}

#[cfg(test)]
mod tests;
/// A store holding a fit, a prediction, a profile and a simulation, each
/// labelled, for tests of the readers of stored results.
#[cfg(test)]
pub(crate) use tests::store_of_four;
