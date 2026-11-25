//! A small project on the store, for the store's own tests: a simulation
//! (atomic, grouped by model), its observations (a by-product), a fit (in
//! place), a prediction placed inside a fit, a model (a content artifact of
//! one file) and a model's presentation (a content artifact of several).

pub(crate) use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// Re-exported so each module's tests take the fixtures and these with one glob.
pub(crate) use crate::{
    Artifact, Build, ByProduct, Content, ContentArtifact, ContentFiles, Dep, Finite, Given, Hashed, Identity, Kinds, Layout, Placed,
    Primary, RunInfo, Store, StoreRequest, TopLevel, WriteMode,
};

#[derive(Identity, Clone, Debug)]
#[identity(tag = "test_engine", version = 1)]
pub(crate) struct Engine {
    pub build: String,
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "sim", version = 1)]
pub(crate) struct SimId {
    pub model: String,
    pub dt: Finite,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SimAttributes {
    pub model: String,
    pub seed: u64,
}

pub(crate) struct Sim;
impl Artifact for Sim {
    const KIND: &'static str = "sim";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = SimId;
    type Attributes = SimAttributes;
}
impl Primary for Sim {
    fn layout(id: &SimId, labels: &SimAttributes) -> Placed {
        Layout::in_kind().group(&labels.model, &id.model).result(&format!("seed_{}", labels.seed))
    }
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "obs", version = 1)]
pub(crate) struct ObsId {
    pub sim: SimIdRecord,
    pub rate: Finite,
}

pub(crate) struct Obs;
impl Artifact for Obs {
    const KIND: &'static str = "obs";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ObsId;
    type Attributes = ();
}
impl ByProduct for Obs {
    type Of = Sim;
    fn upstream(id: &ObsId) -> &SimIdRecord {
        &id.sim
    }
    fn label(_: &ObsId, _: &()) -> String {
        "obs".into()
    }
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "fit", version = 1)]
pub(crate) struct FitId {
    pub problem: String,
    pub method: String,
}

pub(crate) struct Fit;
impl Artifact for Fit {
    const KIND: &'static str = "fit";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = FitId;
    type Attributes = ();
}
impl Primary for Fit {
    fn layout(id: &FitId, _: &()) -> Placed {
        Layout::in_kind().group(&id.problem, &id.problem).result(&id.method)
    }
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "fit_trace", version = 1)]
pub(crate) struct TraceId {
    pub fit: FitIdRecord,
}

pub(crate) struct Trace;
impl Artifact for Trace {
    const KIND: &'static str = "trace";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = TraceId;
    type Attributes = ();
}
impl ByProduct for Trace {
    type Of = Fit;
    fn upstream(id: &TraceId) -> &FitIdRecord {
        &id.fit
    }
    fn label(_: &TraceId, _: &()) -> String {
        "trace".into()
    }
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "prediction", version = 1)]
pub(crate) struct PredId {
    pub fit: Dep<Fit>,
    pub n: u64,
}

pub(crate) struct Pred;
impl Artifact for Pred {
    const KIND: &'static str = "prediction";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = PredId;
    type Attributes = ();
}
impl Primary for Pred {
    fn layout(id: &PredId, _: &()) -> Placed {
        Layout::inside(id.fit.id(), "predictions").result(&format!("n{}", id.n))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ModelText(pub String);
impl Content for ModelText {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.as_bytes().to_vec()
    }
    fn from_bytes(b: &[u8]) -> crate::Result<Self> {
        Ok(ModelText(String::from_utf8_lossy(b).into_owned()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ModelAttributes {
    pub name: String,
}

pub(crate) struct Model;
impl Artifact for Model {
    const KIND: &'static str = "compiled_model";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = Hashed<ModelText>;
    type Attributes = ModelAttributes;
}
impl Primary for Model {
    fn layout(_: &Hashed<ModelText>, labels: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&labels.name)
    }
}
impl ContentArtifact for Model {
    type Content = ModelText;
    const FILE: &'static str = "model.txt";
}

/// A model's text and, when it has one, its picture: two files, each held
/// in the identity by its digest.
#[derive(Identity, Clone, Debug)]
#[identity(tag = "presentation", version = 1)]
pub(crate) struct PresentationId {
    pub text: Hashed<ModelText>,
    pub picture: Option<Hashed<ModelText>>,
}

pub(crate) struct Presentation;
impl Artifact for Presentation {
    const KIND: &'static str = "presentation";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = PresentationId;
    type Attributes = ModelAttributes;
}
impl Primary for Presentation {
    fn layout(_: &PresentationId, labels: &ModelAttributes) -> Placed {
        Layout::in_kind().result(&labels.name)
    }
}
impl ContentFiles for Presentation {
    fn files(id: &PresentationId) -> Vec<(String, Vec<u8>)> {
        let mut v = vec![("text.txt".to_string(), id.text.get().to_bytes())];
        v.extend(id.picture.iter().map(|p| ("picture.txt".to_string(), p.get().to_bytes())));
        v
    }
}

pub(crate) fn kinds() -> Kinds {
    Kinds::new()
        .register::<Sim>()
        .register::<Obs>()
        .register::<Fit>()
        .register::<Trace>()
        .register::<Pred>()
        .register::<Model>()
        .register::<Presentation>()
}

/// A fresh, empty directory under the system's temporary directory, removed
/// when dropped.
pub(crate) struct TempRoot(pub PathBuf);
impl TempRoot {
    pub fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(crate::fsutil::unique_name(&format!("store-test-{tag}")));
        std::fs::create_dir_all(&p).unwrap();
        TempRoot(p)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn open(root: &Path) -> Store {
    open_build(root, "b1")
}

pub(crate) fn open_build(root: &Path, build: &str) -> Store {
    Store::open(root, Build::Known(Engine { build: build.into() }), TopLevel::ByBuild { label: "0.1.0".into() }, kinds()).unwrap()
}

pub(crate) fn info() -> RunInfo {
    RunInfo::new(&Given::default(), &(), &StoreRequest::default()).unwrap()
}

pub(crate) fn labelled(label: &str) -> RunInfo {
    RunInfo::new(&Given::default(), &(), &StoreRequest::new(Some(label), false, None).unwrap()).unwrap()
}

pub(crate) fn sim(seed: u64) -> (SimId, SimAttributes) {
    (SimId { model: "sir".into(), dt: Finite::new(0.5).unwrap(), seed }, SimAttributes { model: "sir".into(), seed })
}

/// A file set of the result files `names` and no provenance file.
pub(crate) fn expecting(names: &[&str]) -> crate::ExpectedFiles {
    files(names, &[])
}

/// A file set of the result files `result` and the provenance files
/// `provenance`.
pub(crate) fn files(result: &[&str], provenance: &[&str]) -> crate::ExpectedFiles {
    crate::ExpectedFiles {
        result: result.iter().map(|n| n.to_string()).collect(),
        provenance: provenance.iter().map(|n| n.to_string()).collect(),
    }
}

/// The observations wanted of `run`, if they were, declared as `obs.tsv`.
pub(crate) fn also_obs(run: &mut crate::Run<Sim>) -> Option<crate::ByProductRun<Obs>> {
    let mut obs = run.also::<Obs>().unwrap()?;
    obs.expect(expecting(&["obs.tsv"]));
    Some(obs)
}

/// Begins a simulation and, if claimed, writes `traj.tsv` from its seed and
/// finishes it.
pub(crate) fn run_sim(store: &Store, info: &RunInfo, seed: u64) -> crate::Stored<Sim> {
    let (id, labels) = sim(seed);
    store
        .run::<Sim, crate::Error>(&id, &labels, info, crate::Reuse::IfCompleted, expecting(&["traj.tsv"]), "simulating", |run| {
            run.result_file("traj.tsv")?.write_all(format!("t\tI\n0\t{seed}\n").as_bytes())
        })
        .map_err(|e| e.to_string())
        .unwrap()
}

/// Every folder under `root` holding an `identity.json`, relative to `root`,
/// outside the store's own dot-named entries.
pub(crate) fn result_folders(root: &Path) -> Vec<String> {
    fn go(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                if p.join("identity.json").is_file() {
                    out.push(crate::fsutil::rel_key(root, &p));
                }
                go(&p, root, out);
            }
        }
    }
    let mut out = Vec::new();
    go(root, root, &mut out);
    out.sort();
    out
}

/// A process on this host that has exited, as an owner: its id names no
/// process.
pub(crate) fn dead_owner() -> crate::Owner {
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    crate::Owner { host: crate::this_host().into(), pid, started: crate::Timestamp("2000-01-01T00:00:00.000Z".into()) }
}

/// A live process on this host, as an owner, and the process.
pub(crate) fn live_owner() -> (crate::Owner, std::process::Child) {
    let child = std::process::Command::new("sleep").arg("60").spawn().unwrap();
    let pid = child.id();
    let started = loop {
        if let Some(t) = crate::owner::process_start(pid) {
            break t;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    (crate::Owner { host: crate::this_host().into(), pid, started }, child)
}

/// An owner on another host, with no beat file.
pub(crate) fn remote_owner() -> crate::Owner {
    crate::Owner { host: "another-host.invalid".into(), pid: 4242, started: crate::Timestamp("2026-01-01T00:00:00.000Z".into()) }
}

/// Rewrites the owner a record or lock file at `path` names.
pub(crate) fn set_owner(path: &Path, owner: &crate::Owner) {
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v["owner"] = serde_json::to_value(owner).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}
