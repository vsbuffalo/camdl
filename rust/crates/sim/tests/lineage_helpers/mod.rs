//! Shared test plumbing for the three-layer lineage tests.
//!
//! These helpers exercise the refactored path: a simulation records a Layer-1
//! [`EventLog`] (drawing no identities), then [`realize`] replays it into a
//! line list at a chosen identity seed. This is the in-process equivalent of
//! `camdl simulate --event-log` → `camdl lineage realize` (the CLI binary is
//! exercised separately in `crates/cli/tests/lineage_e2e.rs`).

#![allow(dead_code)] // not every test uses every helper

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use sim::{
    chain_binomial::run_chain_binomial_with_observer,
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig},
    gillespie::run_gillespie_with_observer,
    lineage::{realize, EventLog, EventRecorder, LineListEntry, LineListWriter, RealizeSummary},
    state::Trajectory,
};

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("tests/fixtures")
}

pub fn load_fixture(name: &str) -> ir::Model {
    let path = fixtures_dir().join(format!("{}.ir.json", name));
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("cannot read fixture {}", name));
    ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {}: {}", name, e))
}

pub fn set_params(m: &mut ir::Model, vals: &[(&str, f64)]) {
    for p in &mut m.parameters {
        if let Some((_, v)) = vals.iter().find(|(n, _)| *n == p.name) {
            p.value = p.value.with_value(*v);
        }
    }
}

/// In-memory line-list collector for tests.
#[derive(Clone)]
pub struct VecWriter {
    pub entries: Rc<RefCell<Vec<LineListEntry>>>,
}

impl VecWriter {
    pub fn new() -> Self {
        VecWriter { entries: Rc::new(RefCell::new(Vec::new())) }
    }
}

impl Default for VecWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl LineListWriter for VecWriter {
    fn init(&mut self) -> Result<(), sim::SimError> {
        Ok(())
    }
    fn write(&mut self, e: &LineListEntry) -> Result<(), sim::SimError> {
        self.entries.borrow_mut().push(e.clone());
        Ok(())
    }
    fn finish(&mut self) -> Result<(), sim::SimError> {
        Ok(())
    }
}

/// The backend to record an event log under.
#[derive(Clone, Copy)]
pub enum Backend {
    Gillespie,
    ChainBinomial { dt: f64 },
}

/// Record a Layer-1 event log: run the chosen backend with the identity-free
/// [`EventRecorder`] attached. Returns the count trajectory and the event log.
/// The trajectory is byte-identical to a plain run at the same seed.
pub fn record_event_log(
    m: &ir::Model,
    backend: Backend,
    seed: u64,
    t_end: f64,
) -> (Trajectory, EventLog) {
    let compiled = CompiledModel::new(m.clone()).unwrap();
    let params = compiled.default_params.clone();
    let (initial_int, _) = compiled.initial_state_mean(&params).unwrap();
    let mut recorder = EventRecorder::new(&compiled, &initial_int).unwrap();
    let traj = match backend {
        Backend::Gillespie => {
            let cfg = GillespieConfig { t_start: 0.0, t_end, output_dt: None };
            run_gillespie_with_observer(&compiled, &params, seed, &cfg, Some(&mut recorder), None).unwrap()
        }
        Backend::ChainBinomial { dt } => {
            let cfg = ChainBinomialConfig { t_start: 0.0, t_end, dt };
            run_chain_binomial_with_observer(
                &compiled, &params, seed, &cfg, Some(&mut recorder), None, Default::default(),
            )
            .unwrap()
        }
    };
    (traj, recorder.into_event_log())
}

/// Realize an event log into a line list at `identity_seed`.
pub fn realize_log(log: &EventLog, identity_seed: u64) -> (Vec<LineListEntry>, RealizeSummary) {
    let collector = VecWriter::new();
    let buf = collector.entries.clone();
    let mut writer = collector;
    let summary = realize(log, identity_seed, &mut writer).unwrap();
    let entries = buf.borrow().clone();
    (entries, summary)
}

/// Record + realize in one call (Gillespie), at `identity_seed`. The common
/// case for the structural / frequency tests: one epidemic, one identity draw.
pub fn run_with_lineage(
    m: ir::Model,
    seed: u64,
    t_end: f64,
) -> (Trajectory, Vec<LineListEntry>) {
    run_with_lineage_seeded(m, seed, t_end, seed)
}

/// Record + realize (Gillespie) with an explicit `identity_seed` distinct from
/// the dynamics `seed`.
pub fn run_with_lineage_seeded(
    m: ir::Model,
    seed: u64,
    t_end: f64,
    identity_seed: u64,
) -> (Trajectory, Vec<LineListEntry>) {
    let (traj, log) = record_event_log(&m, Backend::Gillespie, seed, t_end);
    let (entries, _) = realize_log(&log, identity_seed);
    (traj, entries)
}
