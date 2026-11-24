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
//! presentation. A reader finds a result's presentation through the
//! invocation that names the result ([`model_presentation`]). The design is
//! `docs/dev/proposals/2026-11-24-model-presentation.md`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use store::{Artifact, ContentFiles, Dep, Handle, Hashed, Layout, Placed, Primary, Provenance, RunRef, Store, Stored, WriteMode};

use super::{CompiledModel, CompiledModelContent, ModelLabels};

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
    type Labels = ModelLabels;
}

impl Primary for ModelPresentation {
    fn layout(_id: &PresentationIdentity, labels: &ModelLabels) -> Placed {
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
    labels: &ModelLabels,
    provenance: &Provenance,
) -> store::Result<Dep<CompiledModel>> {
    use store::ContentArtifact;
    let presentation = match presentation.into() {
        Presentation::New(p) => Some(store.put::<ModelPresentation>(p, labels, &Provenance::none(), info)?.id().key().to_hex()),
        Presentation::Stored(hex) => Some(hex),
        Presentation::Unknown => None,
    };
    let labels = ModelLabels { presentation, ..labels.clone() };
    store.put::<CompiledModel>(model, &labels, provenance, info)?.dep(CompiledModel::FILE)
}

/// The run id (hex) of the presentation the invocation `handle` names
/// recorded for the compiled model `model`: for `@label`, that invocation;
/// for an id or a folder, the latest invocation that listed the result. When
/// that invocation did not compile `model` (a prediction from a stored fit),
/// the latest invocation that compiled it answers. `None` when no invocation
/// recorded one. Nothing is read from the presentation.
pub(crate) fn recorded_presentation(store: &Store, handle: &Handle, model: &Dep<CompiledModel>) -> Option<String> {
    let recorded = |inv: &store::Invocation| -> Option<String> {
        let labels = inv.labels_of(<CompiledModel as Artifact>::KIND, model.id().key())?;
        serde_json::from_value::<ModelLabels>(labels).ok()?.presentation
    };
    store
        .invocation(handle)
        .ok()
        .as_ref()
        .and_then(recorded)
        .or_else(|| store.invocation(&Handle::IdPrefix(model.id().key().to_hex())).ok().as_ref().and_then(recorded))
}

/// The stored presentation [`recorded_presentation`] names, read.
fn model_presentation(
    store: &Store,
    handle: &Handle,
    model: &Dep<CompiledModel>,
) -> Result<Option<Stored<ModelPresentation>>, String> {
    let Some(hex) = recorded_presentation(store, handle, model) else { return Ok(None) };
    store
        .resolve(&RunRef::new(Handle::IdPrefix(hex.clone())))
        .map(Some)
        .map_err(|e| format!("the model's presentation {}: {e}", &hex[..12.min(hex.len())]))
}

/// Restores into `envelope` the description and documentation of the
/// presentation that `handle` names for `model` ([`model_presentation`]).
/// No result depends on a presentation, so one that cannot be read leaves
/// the envelope without them and is reported as a warning; the command goes
/// on.
pub(crate) fn present(store: &Store, handle: &Handle, model: &Dep<CompiledModel>, envelope: &mut ir::IrEnvelope) {
    let read = || -> Result<Option<PresentedDocs>, String> {
        let Some(shown) = model_presentation(store, handle, model)? else { return Ok(None) };
        let bytes = super::read_stored(&shown, DOCS)?;
        serde_json::from_slice(&bytes).map(Some).map_err(|e| format!("{DOCS}: {e}"))
    };
    match read() {
        Ok(Some(docs)) => {
            envelope.model.description = docs.description;
            envelope.docs = docs.docs;
        }
        Ok(None) => {}
        Err(e) => crate::out::live().notice(&crate::out::notice::Notice {
            severity: crate::out::notice::Severity::Warning,
            code: "model-presentation-unreadable",
            message: format!("{handle}: the model's documentation is not shown, since its stored presentation cannot be read: {e}"),
        }),
    }
}
