//! camdl on the run store (`store` crate): the ambient identity, the kinds
//! camdl stores, the store root, the shared directory grouping, the value
//! types identities hold, and the two content artifacts, the compiled model
//! and a data table. The design is
//! `docs/dev/proposals/2026-11-20-run-store-api.md`, Part 2.
//!
//! A command on the store opens it with [`open`] (or [`scratch`] under
//! `--no-store`), builds one [`store::RunInfo`], stores the compiled model and
//! its data tables with `Store::put`, and names them in its identity by a
//! [`store::Dep`]. It begins each result through [`begin`], [`begin_with`] or
//! [`declared`], which declare the files the run will hold from the kind's
//! file enum ([`crate::result_files`]). Its results sit under camdl's build directory,
//! `<root>/<version>-<12 hex of the ambient key>/` ([`camdl_top_level`]).
//!
//! The compiled model is split by [`CompiledParts::split`] into the part every
//! result depends on ([`CompiledModelContent`]), the quantities, the model
//! source ([`model_source`]), the model's name (an attribute) and
//! provenance. The model part is keyed by its own bytes, so a docstring, the
//! description, a quantity or the file's name changes no result's key, and
//! every other field, derivative tables included, does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use store::{
    Artifact, Build, ContentAddressed, ContentArtifact, Dep, Finite, Hashed, Key, Kinds, Layout, Placed, Primary,
    Provenance, Store, StoreRequest, TopLevel, WriteMode,
};

use crate::result_files::Files;

// ── Ambient identity ─────────────────────────────────────────────────────────

/// What every run id holds besides its own identity: the build of camdl and
/// the environment switches that can change a result.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "engine", version = 1)]
pub(crate) struct Engine {
    /// The build fingerprint, [`crate::version::ENGINE_ID`].
    pub build: String,
    pub env: CamdlEnvIdentity,
}

/// The environment variables that can change a result, as read by
/// [`crate::env`]: the two alternative evaluation paths, each true when the
/// variable is set, and every set `CAMDL_EXPERIMENTAL_*` variable with its
/// value. `CAMDL_NO_LICM` and `CAMDL_NO_CONSTANT_FOLD` are not here: they
/// change the compiled model, whose bytes every result is keyed by.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "camdl_env", version = 1)]
pub(crate) struct CamdlEnvIdentity {
    pub eval_unresolved: bool,
    pub no_binding_cache: bool,
    pub experimental: BTreeMap<String, String>,
}

/// This build and this process's environment switches. A build some part of
/// whose fingerprint could not be determined is `Build::Incomplete`, under
/// which the store reuses nothing.
pub(crate) fn ambient() -> Build<Engine> {
    let vars = &crate::env::get().vars;
    let engine = Engine {
        build: crate::version::ENGINE_ID.to_string(),
        env: CamdlEnvIdentity {
            eval_unresolved: vars.eval_unresolved,
            no_binding_cache: vars.no_binding_cache,
            experimental: vars.experimental.clone(),
        },
    };
    if crate::version::ENGINE_REUSE.allows_reuse() {
        Build::Known(engine)
    } else {
        Build::Incomplete(engine)
    }
}

/// camdl groups its results by build, under a directory labelled with its
/// readable version, [`crate::version::VERSION_SHORT`].
pub(crate) fn camdl_top_level() -> TopLevel {
    TopLevel::ByBuild { label: crate::version::VERSION_SHORT.to_string() }
}

/// Every kind camdl stores on the run store.
pub(crate) fn camdl_kinds() -> Kinds {
    let kinds = Kinds::new()
        .register::<CompiledModel>()
        .register::<Quantities>()
        .register::<DataTable>()
        .register::<ModelSource>();
    let kinds = crate::fit_store::register(crate::sim_store::register(crate::pfilter_store::register(kinds)));
    crate::profile_store::register(kinds)
}

// ── The store root and the request ───────────────────────────────────────────

/// The store-class inputs of a command: its `--label`, its `--force`, and the
/// root of its run store, chosen as `docs/camdl-run-spec.md` §2.1 states:
/// `flag` (`--store`), else `fit_file_store` (the `store` key of the fit file
/// the command reads, already anchored at the file), else `CAMDL_STORE`, else
/// [`DEFAULT_ROOT`]. A command that reads no fit file calls [`store_request`].
///
/// `force` asks the store to compute each result again (`Reuse::Force`): a
/// completed result moves to the quarantine and is claimed anew. A claim held
/// by an owner not known to be stopped is refused without `force`; with it,
/// the claim is taken unless its owner is on this host and running, or on
/// another host with a beat that changes. An owner on another host whose beat
/// file is stale or cannot be read is taken, though it may still be running:
/// `force` asserts that it has stopped.
///
/// A `label` [`store::Label::parse`] refuses is refused with
/// [`store::Error::InvalidLabel`].
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn store_request_with_fit_file(
    label: Option<String>,
    force: bool,
    flag: Option<PathBuf>,
    fit_file_store: Option<PathBuf>,
) -> store::Result<StoreRequest> {
    request_from(label, force, flag, fit_file_store, crate::env::get().vars.store.clone())
}

/// [`store_request_with_fit_file`] for a command that reads no fit file: the
/// root is `root` (`--store`), else `CAMDL_STORE`, else [`DEFAULT_ROOT`].
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn store_request(label: Option<String>, force: bool, root: Option<PathBuf>) -> store::Result<StoreRequest> {
    store_request_with_fit_file(label, force, root, None)
}

/// [`store_request`] for a command that names no invocation.
pub(crate) fn unlabelled_request(root: Option<PathBuf>) -> StoreRequest {
    store_request(None, false, root).expect("a request without a label is never refused")
}

/// The request [`store_request_with_fit_file`] builds, given the value of
/// `CAMDL_STORE`.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
fn request_from(
    label: Option<String>,
    force: bool,
    flag: Option<PathBuf>,
    fit_file_store: Option<PathBuf>,
    env_store: Option<PathBuf>,
) -> store::Result<StoreRequest> {
    let env = StoreRequest::new(None, false, env_store)?;
    Ok(StoreRequest::new(label.as_deref(), force, flag.or(fit_file_store))?.merge(env))
}

/// camdl's run store root when no input names one.
pub(crate) const DEFAULT_ROOT: &str = "camdl-store";

/// The root a request names, else [`DEFAULT_ROOT`].
pub(crate) fn root_of(request: &StoreRequest) -> PathBuf {
    request.root(Path::new(DEFAULT_ROOT))
}

/// Opens the store a request names.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn open(request: &StoreRequest) -> store::Result<Store> {
    Store::open(&root_of(request), ambient(), camdl_top_level(), camdl_kinds())
}

/// A temporary store, deleted when dropped, for a command run with
/// `--no-store`.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn scratch() -> store::Result<Store> {
    Store::scratch(ambient(), camdl_kinds())
}

/// The root a reader uses when it is given no store: `CAMDL_STORE`, else the
/// default.
pub(crate) fn read_root() -> PathBuf {
    root_of(&unlabelled_request(None))
}

/// Says, once per command, that the invocation records `unread` under the
/// store at `root` are not read: the store calls it ([`store::report_unread_records_to`])
/// the first time a read of the records meets one. A record of another
/// format was written by another build; one of this format that does not
/// read is damaged. Neither's label or listings are seen by a lookup, and the
/// results they list stay reusable.
pub(crate) fn report_unread_records(root: &Path, unread: &[store::UnreadRecord]) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if unread.is_empty() || SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let listed: Vec<String> = unread.iter().map(|u| format!("\n    {u}")).collect();
    crate::out::live().notice(&crate::out::notice::Notice {
        severity: crate::out::notice::Severity::Warning,
        code: "invocation-records-not-read",
        message: format!(
            "{} invocation record(s) in the run store at {} are not read, so no label or listing they hold is \
             seen; the results they list stay reusable:{}",
            unread.len(),
            root.display(),
            listed.concat()
        ),
    });
}

/// Opens the store at `root` for reading, or `None` when no store is there.
pub(crate) fn open_existing(root: &Path) -> Option<Store> {
    if !root.join("store.json").exists() {
        return None;
    }
    open(&unlabelled_request(Some(root.to_path_buf()))).ok()
}

// ── Beginning a run ──────────────────────────────────────────────────────────

/// Begins `identity` as `Store::begin` does, and declares on a claim that the
/// run will hold `files`, its kind's files as planned before the run's
/// outcome is known. A command whose files depend on the outcome declares
/// them again before it finishes ([`crate::result_files::declare`]).
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn begin<A: Primary + Files>(
    store: &Store,
    identity: &A::Identity,
    labels: &A::Attributes,
    info: &store::RunInfo,
    reuse: store::Reuse,
    files: impl IntoIterator<Item = A::File>,
) -> store::Result<store::Begin<A>> {
    store.begin::<A>(identity, labels, info, reuse).map(|b| declared(b, files))
}

/// [`begin`], also wanting the by-products `wanted`.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn begin_with<A: Primary + Files>(
    store: &Store,
    identity: &A::Identity,
    labels: &A::Attributes,
    wanted: &store::Wanted<A>,
    info: &store::RunInfo,
    reuse: store::Reuse,
    files: impl IntoIterator<Item = A::File>,
) -> store::Result<store::Begin<A>> {
    store.begin_with::<A>(identity, labels, wanted, info, reuse).map(|b| declared(b, files))
}

/// `begun`, with `files` declared on it when it is a claim: for each member
/// of a batch (`Declared::begin_all`), as [`begin`] does for one.
pub(crate) fn declared<A: Files>(begun: store::Begin<A>, files: impl IntoIterator<Item = A::File>) -> store::Begin<A> {
    match begun {
        store::Begin::Claimed(mut run) => {
            crate::result_files::declare(&mut run, files);
            store::Begin::Claimed(run)
        }
        other => other,
    }
}

/// The by-product `B`'s part of `run`, if it was wanted and is not stored,
/// declared to hold `files`.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn also<B: Files + store::ByProduct>(
    run: &mut store::Run<B::Of>,
    files: impl IntoIterator<Item = B::File>,
) -> store::Result<Option<store::ByProductRun<B>>>
where
    B::Identity: Send + Sync,
{
    let Some(mut part) = run.also::<B>()? else { return Ok(None) };
    crate::result_files::declare_by_product(&mut part, files);
    Ok(Some(part))
}

/// A stored result file's bytes, read through the store, which checks them
/// against the file's digest.
pub(crate) fn read_stored<A: Artifact>(stored: &store::Stored<A>, name: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut stored.open(name).map_err(|e| e.to_string())?, &mut bytes)
        .map_err(|e| format!("reading {name}: {e}"))?;
    Ok(bytes)
}

// ── Layout ───────────────────────────────────────────────────────────────────

/// camdl's shared grouping under a kind's directory: the model, then the
/// settings, then the parameters for a kind that has them.
pub(crate) fn camdl_groups<S: ContentAddressed>(
    model_label: &str,
    model: &Dep<CompiledModel>,
    settings_label: &str,
    settings: &S,
    params: Option<(&str, &ResolvedParams)>,
) -> Layout {
    let layout = Layout::in_kind().group(model_label, model).group(settings_label, settings);
    match params {
        Some((label, params)) => layout.group(label, params),
        None => layout,
    }
}

// ── Value types identities hold ──────────────────────────────────────────────

/// Every parameter's value after all sources, by name.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "resolved_params", version = 1)]
pub(crate) struct ResolvedParams {
    pub values: BTreeMap<String, Finite>,
}

impl ResolvedParams {
    /// The values of `pairs`. A value that is NaN or infinite is refused, and
    /// so is a name given twice.
    pub fn new(pairs: impl IntoIterator<Item = (String, f64)>) -> Result<Self, String> {
        let mut values = BTreeMap::new();
        for (name, v) in pairs {
            let f = Finite::new(v).map_err(|e| format!("parameter `{name}`: {e}"))?;
            if values.insert(name.clone(), f).is_some() {
                return Err(format!("parameter `{name}` is given twice"));
            }
        }
        Ok(ResolvedParams { values })
    }
}

/// The interventions a run keeps after its scenario selection (`--scenario`,
/// `--enable`, `--disable`, or a fit's selection), by name: the scenario's
/// effect on the model, without its name. Its effect on parameter values is
/// in [`ResolvedParams`].
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "active_interventions", version = 1)]
pub(crate) struct ActiveInterventions {
    pub names: std::collections::BTreeSet<String>,
}

impl ActiveInterventions {
    pub fn of(model: &ir::Model) -> Self {
        ActiveInterventions { names: model.interventions.iter().map(|iv| iv.name.clone()).collect() }
    }
}

/// The model a run computes on: the stored compiled model with the run's
/// parameter values set and only its active interventions kept.
pub(crate) fn apply(model: &ir::Model, params: &ResolvedParams, active: &ActiveInterventions) -> ir::Model {
    let mut m = model.clone();
    for p in &mut m.parameters {
        if let Some(v) = params.values.get(&p.name) {
            if p.value.resolved_value() != Some(v.get()) {
                p.value = p.value.with_value(v.get());
            }
        }
    }
    m.interventions.retain(|iv| active.names.contains(&iv.name));
    m
}

/// Checks that the model a result is about to run is the stored compile its
/// identity names.
///
/// Refuses `model`, loaded from `path`, unless its model part has the digest
/// `stored`. `what` names the computation (a fit, a prediction, a profile) in
/// the refusal. A result is thus never stored under the identity of a model
/// it did not run.
pub(crate) fn require_stored_model(model: &ir::Model, stored: Key, what: &str, path: &str) -> Result<(), String> {
    if model_part_digest(model) == stored {
        return Ok(());
    }
    Err(format!(
        "internal error: the model this {what} runs ({path}) is not the stored compiled model {} its identity names",
        stored.short12()
    ))
}

/// The digest the model part of `model` would be stored under.
pub(crate) fn model_part_digest(model: &ir::Model) -> Key {
    Key::digest_bytes(&model_part_bytes(model))
}

/// The seed of one process: the seed every random stream of a result is
/// derived from.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "process_seed", version = 1)]
pub(crate) struct ProcessSeed {
    pub seed: u64,
}

// ── The compiled model ───────────────────────────────────────────────────────

/// The part of a compiled model that the engine runs: every field of
/// `ir::Model` except its name, its description, its quantities and its
/// contrasts, derivative tables included. Its bytes are canonical JSON (object
/// keys sorted at every depth) of those fields, so the digest does not depend
/// on the order a map was filled in, with one exception: the initial
/// conditions are written as an array of `[compartment, spec]` pairs in the
/// model's declaration order, which the runtime evaluates by (it breaks ties
/// between independent entries, and so decides the order initial laws are
/// drawn in). Read back, the model is the one that was stored.
pub(crate) struct CompiledModelContent {
    /// The model, with the four fields that are not this part at their empty
    /// values. Nothing reads them from here.
    model: ir::Model,
    bytes: Vec<u8>,
}

impl CompiledModelContent {
    pub fn model(&self) -> &ir::Model {
        &self.model
    }
}

/// The field of `ir::Model` whose order the runtime reads, written as an
/// ordered list.
const INITIAL_CONDITIONS: &str = "initial_conditions";

/// The fields of `ir::Model` that belong to other parts, left out of the
/// model part's bytes.
const NOT_THE_MODEL_PART: [&str; 4] = ["name", "description", "quantities", "contrasts"];

/// The canonical JSON bytes of any value.
fn canonical_bytes(v: &impl serde::Serialize) -> Vec<u8> {
    let value = serde_json::to_value(v).expect("an IR value encodes as JSON");
    serde_json::to_vec(&crate::canonical_json::canonical(&value)).expect("a JSON value encodes")
}

impl CompiledModelContent {
    fn of(model: ir::Model) -> Self {
        let bytes = model_part_bytes(&model);
        CompiledModelContent { model, bytes }
    }
}

/// The bytes the model part of `model` is stored as.
fn model_part_bytes(model: &ir::Model) -> Vec<u8> {
    let mut v = serde_json::to_value(model).expect("an IR model encodes as JSON");
    if let Some(o) = v.as_object_mut() {
        for k in NOT_THE_MODEL_PART {
            o.remove(k);
        }
        let ordered: Vec<serde_json::Value> = model
            .initial_conditions
            .0
            .iter()
            .map(|(name, spec)| serde_json::json!([name, spec]))
            .collect();
        o.insert(INITIAL_CONDITIONS.into(), serde_json::Value::Array(ordered));
    }
    serde_json::to_vec(&crate::canonical_json::canonical(&v)).expect("a JSON value encodes")
}

impl store::Content for CompiledModelContent {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        let parse = |why: String| store::Error::Parse { what: "a stored compiled model".into(), why };
        let mut v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| parse(e.to_string()))?;
        let o = v.as_object_mut().ok_or_else(|| parse("not a JSON object".into()))?;
        o.insert("name".into(), serde_json::Value::String(String::new()));
        o.insert("description".into(), serde_json::Value::Null);
        // The initial conditions, in the order they were stored.
        let ordered = match o.insert(INITIAL_CONDITIONS.into(), serde_json::Value::Object(Default::default())) {
            Some(serde_json::Value::Array(a)) => a,
            _ => return Err(parse("its initial conditions are not an ordered list".into())),
        };
        let mut model: ir::Model = serde_json::from_value(v).map_err(|e| parse(e.to_string()))?;
        for entry in ordered {
            let (name, spec): (String, ir::model::InitSpec) =
                serde_json::from_value(entry).map_err(|e| parse(format!("an initial condition: {e}")))?;
            model.initial_conditions.0.insert(name, spec);
        }
        model.restore_unresolved_horizons();
        Ok(CompiledModelContent { model, bytes: bytes.to_vec() })
    }
}

/// A model's quantities and contrasts, as canonical JSON: the part of a
/// compile `simulate --quantities-out` evaluates, split out so the model part
/// never holds them.
pub(crate) struct QuantitiesContent {
    bytes: Vec<u8>,
}

impl QuantitiesContent {
    /// The quantities, as the evaluator reads them.
    pub fn quantities(&self) -> Result<Vec<ir::quantity::Quantity>, String> {
        let v: serde_json::Value = serde_json::from_slice(&self.bytes).map_err(|e| format!("the stored quantities: {e}"))?;
        serde_json::from_value(v["quantities"].clone()).map_err(|e| format!("the stored quantities: {e}"))
    }

    /// The contrasts, as the model declares them.
    pub fn contrasts(&self) -> Result<Vec<ir::contrast::Contrast>, String> {
        let v: serde_json::Value = serde_json::from_slice(&self.bytes).map_err(|e| format!("the stored contrasts: {e}"))?;
        serde_json::from_value(v["contrasts"].clone()).map_err(|e| format!("the stored contrasts: {e}"))
    }
}

impl store::Content for QuantitiesContent {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        Ok(QuantitiesContent { bytes: bytes.to_vec() })
    }
}

/// The model's name: the stem of the model file, as an attribute of a
/// compiled model, its quantities and its model source. Which source an
/// invocation compiled a model from is the listing's `source`, recorded by
/// [`put_compiled`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ModelAttributes {
    pub name: String,
}

/// Everything one compile yields, each field of the compiled output in
/// exactly one part.
pub(crate) struct CompiledParts {
    pub model: Hashed<CompiledModelContent>,
    pub quantities: Hashed<QuantitiesContent>,
    /// The model file, its name, the description, the `#'` documentation,
    /// the declared model, the renders and the quantities: the model source,
    /// before it is stored.
    pub source: SourceParts,
    pub labels: ModelAttributes,
    /// The envelope's `ir_version` and `validated_by`, and the digests of the
    /// whole compiled output and of the source file.
    pub provenance: Provenance,
}

/// The model file a compile came from: its bytes, its path, whose name is
/// the model source's and whose extension (`.camdl` or not) names its stored
/// file, and its renders.
pub(crate) struct SourceText<'a> {
    pub bytes: &'a [u8],
    pub path: &'a Path,
    pub rendered: Rendered,
}

impl<'a> SourceText<'a> {
    /// The model file at `path`, whose bytes are `bytes`, rendered by camdlc
    /// when it is a `.camdl` ([`Rendered::of`]).
    pub fn read(bytes: &'a [u8], path: &'a Path) -> Self {
        SourceText { bytes, path, rendered: Rendered::of(path) }
    }
}

impl CompiledParts {
    /// Takes the compiled output apart. Every field of the envelope and of
    /// the model is named here, so a field added to the IR does not compile
    /// until it is assigned to a part.
    pub fn split(envelope: ir::IrEnvelope, name: String, source: SourceText<'_>) -> Result<Self, String> {
        let whole = Key::digest_bytes(&canonical_bytes(&envelope));
        let ir::IrEnvelope { ir_version, validated_by, model, docs, declared } = envelope;
        let ir::Model {
            name: _,
            version,
            time_unit,
            description,
            origin,
            origin_rata_die,
            compartments,
            transitions,
            ode_equations,
            time_functions,
            tables,
            interventions,
            observations,
            parameters,
            constants,
            bindings,
            per_eval_bindings,
            initial_conditions,
            ic_grad,
            output,
            simulation,
            presets,
            model_structure,
            balance,
            identity_tracked_compartments,
            quantities,
            contrasts,
        } = model;
        let model_part = ir::Model {
            name: String::new(),
            version,
            time_unit,
            description: None,
            origin,
            origin_rata_die,
            compartments,
            transitions,
            ode_equations,
            time_functions,
            tables,
            interventions,
            observations,
            parameters,
            constants,
            bindings,
            per_eval_bindings,
            initial_conditions,
            ic_grad,
            output,
            simulation,
            presets,
            model_structure,
            balance,
            identity_tracked_compartments,
            quantities: Vec::new(),
            contrasts: Vec::new(),
        };
        let quantities = Hashed::new(QuantitiesContent {
            bytes: canonical_bytes(&serde_json::json!({ "quantities": quantities, "contrasts": contrasts })),
        });
        let camdl = crate::util::model_is_camdl_source(&source.path.to_string_lossy());
        let file_name = source.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let model_source = SourceParts::of(
            file_name,
            source.bytes,
            camdl,
            description,
            docs,
            &declared,
            source.rendered,
            quantities.clone(),
        )?;
        let e = |e: store::Error| e.to_string();
        let provenance = Provenance::none()
            .value("ir_version", &ir_version).map_err(e)?
            .value("validated_by", &validated_by).map_err(e)?
            .value("whole_sha256", &whole.to_hex()).map_err(e)?
            .value("source_sha256", &Key::digest_bytes(source.bytes).to_hex()).map_err(e)?;
        Ok(CompiledParts {
            model: Hashed::new(CompiledModelContent::of(model_part)),
            quantities,
            source: model_source,
            labels: ModelAttributes { name },
            provenance,
        })
    }
}

/// A compiled model, stored once per distinct model part and build, and
/// depended on by every result that runs it.
pub(crate) struct CompiledModel;

impl Artifact for CompiledModel {
    const KIND: &'static str = "compiled_model";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = Hashed<CompiledModelContent>;
    type Attributes = ModelAttributes;
}

impl Primary for CompiledModel {
    fn layout(_id: &Self::Identity, labels: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&labels.name)
    }
}

impl ContentArtifact for CompiledModel {
    type Content = CompiledModelContent;
    const FILE: &'static str = "model.ir.json";
}

/// A model's quantities and contrasts, stored once per distinct content and
/// build, and depended on by each cell's generated quantities.
pub(crate) struct Quantities;

impl Artifact for Quantities {
    const KIND: &'static str = "quantities";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = Hashed<QuantitiesContent>;
    type Attributes = ModelAttributes;
}

impl Primary for Quantities {
    fn layout(_id: &Self::Identity, labels: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&labels.name)
    }
}

impl ContentArtifact for Quantities {
    type Content = QuantitiesContent;
    const FILE: &'static str = "quantities.json";
}

// ── Data tables ──────────────────────────────────────────────────────────────

/// A data file's bytes, as read.
pub(crate) struct Table {
    bytes: Vec<u8>,
}

impl store::Content for Table {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        Ok(Table { bytes: bytes.to_vec() })
    }
}

/// The stream a table is bound to and its path as given.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct TableAttributes {
    pub stream: String,
    pub path: String,
}

/// A data table stored for the results that depend on it.
pub(crate) struct DataTable;

impl Artifact for DataTable {
    const KIND: &'static str = "data_table";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = Hashed<Table>;
    type Attributes = TableAttributes;
}

impl Primary for DataTable {
    fn layout(_id: &Self::Identity, labels: &TableAttributes) -> Placed {
        Layout::in_kind().result(&labels.stream)
    }
}

impl ContentArtifact for DataTable {
    type Content = Table;
    const FILE: &'static str = "table.tsv";
}

pub(crate) mod model_source;
pub(crate) use model_source::{put_compiled, shown_for, ModelSource, Rendered, Shown, SourceOf, SourceParts};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod poisoning;
