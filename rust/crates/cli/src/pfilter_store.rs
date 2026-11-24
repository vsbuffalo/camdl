//! `camdl pfilter` on the run store: the identity of one filter run, the
//! stored result and its by-products, and the rendering `-o` writes
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "By-products of a filter
//! run" and "Layouts, kind by kind").
//!
//! A filter run ([`Pfilter`]) is written in place. It stores `loglik.toml`,
//! and under `--replicates N` each replicate's log-likelihood in
//! `replicates.tsv`; where each scored value came from is the provenance file
//! `provenance.json`. Every diagnostic the command can write is a by-product
//! of the run, stored inside its folder under the by-product's kind, and the
//! flag that asks for it exports a copy to the path it names. A by-product's
//! identity is the run's identity record plus its own settings, so asking for
//! one never changes the run's id, and asking for it after the run is stored
//! runs the filter again and checks that the stored log-likelihood is
//! reproduced.
//!
//! ```text
//! <build>/pfilter/<model>-…/pf-n<particles>-r<replicates>-dt<dt>-…/params-…/seed_<seed>-…/
//!   identity.json  run.json  loglik.toml  provenance.json
//!   saved_paths/paths-…/  paths.tsv
//!   trace/trace-…/        trace.tsv
//!   …
//! ```

use std::collections::BTreeMap;
use std::io::Write;

use store::{Artifact, ByProduct, Dep, Finite, Kinds, Placed, Primary, Rendering, WriteMode};

use crate::camdl_store::{camdl_groups, ActiveInterventions, CompiledModel, DataTable, ProcessSeed, ResolvedParams};

/// The likelihood a filter scores.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "likelihood_objective", version = 1)]
pub(crate) enum Objective {
    /// Every observation.
    Full,
    /// Every observation but the first (a fit config's `ic_free = true`).
    ConditionalOnFirstObservation,
}

impl From<crate::fit::config_v2::LikelihoodObjective> for Objective {
    fn from(o: crate::fit::config_v2::LikelihoodObjective) -> Self {
        use crate::fit::config_v2::LikelihoodObjective as L;
        match o {
            L::Full => Objective::Full,
            L::ConditionalOnFirstObservation => Objective::ConditionalOnFirstObservation,
        }
    }
}

/// The binomial sampler of the chain-binomial draws. The two draw different
/// values from one random stream.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "binomial_sampler", version = 1)]
pub(crate) enum Binomial {
    Btpe,
    Btrs,
}

impl From<sim::rng::BinomialAlgorithm> for Binomial {
    fn from(b: sim::rng::BinomialAlgorithm) -> Self {
        match b {
            sim::rng::BinomialAlgorithm::Btpe => Binomial::Btpe,
            sim::rng::BinomialAlgorithm::Btrs => Binomial::Btrs,
        }
    }
}

/// How a filter run scores: everything besides the model, the parameters
/// and the seed that can change its log-likelihood.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "pfilter_config", version = 1)]
pub(crate) struct PfilterConfig {
    pub particles: u64,
    /// Independent filter runs averaged into the stored log-likelihood.
    pub replicates: u64,
    pub dt: Finite,
    /// Each bound stream's data table, by the name it was bound under.
    pub data: BTreeMap<String, Dep<DataTable>>,
    pub objective: Objective,
    /// A forcing evaluated outside its range holds its end value.
    pub allow_forcing_extrapolation: bool,
    /// A collapsing rate is 0 instead of killing the particle.
    pub allow_degenerate_rates: bool,
    /// Under `--save-final-state` and the fit's `holdout_after`, how many
    /// leading rows of each stream were bound; `None` when no row was cut.
    pub training_rows: Option<BTreeMap<String, u64>>,
    pub binomial: Binomial,
}

impl PfilterConfig {
    /// The settings group's label, `pf-N<particles>-r<replicates>-dt<dt>`.
    pub fn label(&self) -> String {
        config_label(self.particles, self.replicates, self.dt.get())
    }
}

/// `pf-N<particles>-r<replicates>-dt<dt>`, with an integral `dt` written
/// without its decimals.
pub(crate) fn config_label(particles: u64, replicates: u64, dt: f64) -> String {
    format!("pf-N{particles}-r{replicates}-dt{}", crate::sim_store::fmt_dt(dt))
}

/// One particle-filter evaluation.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "pfilter", version = 1)]
pub(crate) struct PfilterIdentity {
    pub model: Dep<CompiledModel>,
    pub config: PfilterConfig,
    pub params: ResolvedParams,
    /// The interventions the scenario selection keeps.
    pub scenario: ActiveInterventions,
    pub seed: ProcessSeed,
    /// The stored fit problem `--fit` named, whose training rows, `[fixed]`
    /// values and settings the run read; `None` without `--fit`, and for a
    /// fit file, which is read in process.
    pub fit: Option<Dep<crate::fit_store::FitProblem>>,
}

/// The model file's stem, the seed the user typed, and for a point of a
/// `--sweep-file` design, the point.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PfilterLabels {
    pub model: String,
    pub seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point: Option<DesignPointLabel>,
}

impl PfilterLabels {
    /// The labels as the columns of a `--each` table (run spec §4.5): the
    /// model and the seed, then for a design point its `row`, its
    /// `point_id` and one column per design column.
    pub(crate) fn columns(&self) -> Vec<(String, String)> {
        let mut c: Vec<(String, String)> = vec![("model".into(), self.model.clone()), ("seed".into(), self.seed.to_string())];
        if let Some(p) = &self.point {
            c.push(("row".into(), p.row.to_string()));
            c.push(("point_id".into(), p.point_id.to_string()));
            crate::resolve_file::push_swept(&mut c, &p.values);
        }
        c
    }
}

/// A design point as its row of the design table shows it: the row's
/// position in the design file, its id, and the value of each design column,
/// sorted by name.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct DesignPointLabel {
    pub row: usize,
    pub point_id: usize,
    pub values: Vec<(String, f64)>,
}

/// One particle-filter run, written in place.
pub(crate) struct Pfilter;

impl Artifact for Pfilter {
    const KIND: &'static str = "pfilter";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = PfilterIdentity;
    type Labels = PfilterLabels;
}

impl Primary for Pfilter {
    fn layout(id: &PfilterIdentity, labels: &PfilterLabels) -> Placed {
        camdl_groups(&labels.model, &id.model, &id.config.label(), &id.config, Some(("params", &id.params)))
            .result(&format!("seed_{}", labels.seed))
    }
}

/// The run's result files.
pub(crate) const LOGLIK_FILE: &str = "loglik.toml";
pub(crate) const REPLICATES_FILE: &str = "replicates.tsv";
pub(crate) const PROVENANCE_FILE: &str = "provenance.json";

/// A by-product of a filter run whose identity is the run's alone, with its
/// kind, its identity's version, its folder's label and its one file.
macro_rules! run_by_product {
    ($(#[$doc:meta])* $name:ident, $identity:ident, $tag:literal, $version:literal, $label:literal, $file:literal) => {
        $(#[$doc])*
        pub(crate) struct $name;

        #[derive(store::Identity, Clone, Debug)]
        #[identity(tag = $tag, version = $version)]
        pub(crate) struct $identity {
            pub filter: PfilterIdentityRecord,
        }

        impl $name {
            pub const FILE: &'static str = $file;
        }

        impl Artifact for $name {
            const KIND: &'static str = $tag;
            const WRITE: WriteMode = WriteMode::Atomic;
            type Identity = $identity;
            type Labels = ();
        }

        impl ByProduct for $name {
            type Of = Pfilter;
            fn upstream(id: &$identity) -> &PfilterIdentityRecord {
                &id.filter
            }
            fn label(_id: &$identity, _labels: &()) -> String {
                $label.into()
            }
        }
    };
}

run_by_product!(
    /// `--trace`: per observation, the log-likelihood increment, the ESS and
    /// the one-step predictive quantiles.
    Trace, TraceIdentity, "trace", 1, "trace", "trace.tsv"
);
run_by_product!(
    /// `--pf-health`: per observation, the ESS, its fraction of the swarm and
    /// the log-weight variance.
    PfHealth, PfHealthIdentity, "pf_health", 1, "pf_health", "pf_health.tsv"
);
run_by_product!(
    /// `--save-filtering`: every particle's state and log-weight at every step.
    Filtering, FilteringIdentity, "filtering", 1, "filtering", "filtering.tsv"
);
run_by_product!(
    /// `--save-filtered-streams`: each stream's conditional mean, summarised
    /// over the swarm before and after each step is scored.
    FilteredStreams, FilteredStreamsIdentity, "filtered_streams", 2, "filtered_streams", "filtered_streams.tsv"
);
run_by_product!(
    /// `--save-final-state`: the swarm after the last observation.
    FinalState, FinalStateIdentity, "final_state", 1, "final_state", "final_state.tsv"
);

/// `--save-paths`: sample paths from the smoothing distribution, traced
/// through the ancestry.
pub(crate) struct SavedPaths;

#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "saved_paths", version = 1)]
pub(crate) struct SavedPathsIdentity {
    pub filter: PfilterIdentityRecord,
    /// `--n-paths`.
    pub paths: u64,
}

impl SavedPaths {
    pub const FILE: &'static str = "paths.tsv";
}

impl Artifact for SavedPaths {
    const KIND: &'static str = "saved_paths";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = SavedPathsIdentity;
    type Labels = ();
}

impl ByProduct for SavedPaths {
    type Of = Pfilter;
    fn upstream(id: &SavedPathsIdentity) -> &PfilterIdentityRecord {
        &id.filter
    }
    fn label(_id: &SavedPathsIdentity, _labels: &()) -> String {
        "paths".into()
    }
}

/// `--save-prequential`: the one-step-ahead predictive scores of the run.
pub(crate) struct Prequential;

#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "prequential", version = 2)]
pub(crate) struct PrequentialIdentity {
    pub filter: PfilterIdentityRecord,
    /// `--score-from`, resolved to model time: the trace scores only the
    /// observations after it.
    pub score_from: Option<Finite>,
    /// The JSON keeps each stream's predictive samples (no
    /// `--no-save-samples`).
    pub samples: bool,
}

impl Prequential {
    pub const JOINT: &'static str = "prequential.tsv";
    pub const STREAMS: &'static str = "prequential.streams.tsv";
    pub const JSON: &'static str = "prequential.json";
}

impl Artifact for Prequential {
    const KIND: &'static str = "prequential";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = PrequentialIdentity;
    type Labels = ();
}

impl ByProduct for Prequential {
    type Of = Pfilter;
    fn upstream(id: &PrequentialIdentity) -> &PfilterIdentityRecord {
        &id.filter
    }
    fn label(_id: &PrequentialIdentity, _labels: &()) -> String {
        "prequential".into()
    }
}

/// Adds every kind of this module to `kinds`.
pub(crate) fn register(kinds: Kinds) -> Kinds {
    kinds
        .register::<Pfilter>()
        .register::<Trace>()
        .register::<PfHealth>()
        .register::<Filtering>()
        .register::<FilteredStreams>()
        .register::<FinalState>()
        .register::<SavedPaths>()
        .register::<Prequential>()
}

// ── The stored log-likelihood ────────────────────────────────────────────────

/// What `loglik.toml` holds.
///
/// A filter that fails at the run's parameters is a result: every particle's
/// weight reached zero at some observation, or the filter stopped with an
/// error the model, the parameters, the data, the particles and the seed
/// determine. Such a run has no log-likelihood, and stores how many of its
/// replicates failed and the first one's message.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Loglik {
    /// The log-likelihood, or under `--replicates N` the mean of the N runs;
    /// absent when a replicate failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglik: Option<f64>,
    pub n_replicates: u64,
    pub n_particles: u64,
    /// The standard deviation over the replicates; absent for one run, and
    /// when a replicate failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglik_sd: Option<f64>,
    /// How many replicates failed; absent when none did.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub n_failed: u64,
    /// The first failed replicate's message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

impl Loglik {
    /// The bytes of `loglik.toml`. Each value is written by `f64`'s
    /// `Display`, which reads back to the same `f64`; `-inf` is written
    /// `-inf`, which TOML reads.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut s = String::new();
        if let Some(ll) = self.loglik {
            s.push_str(&format!("loglik = {}\n", toml_float(ll)));
        }
        s.push_str(&format!("n_replicates = {}\nn_particles = {}\n", self.n_replicates, self.n_particles));
        if let Some(sd) = self.loglik_sd {
            s.push_str(&format!("loglik_sd = {}\n", toml_float(sd)));
        }
        if self.n_failed > 0 {
            s.push_str(&format!("n_failed = {}\n", self.n_failed));
        }
        if let Some(failure) = &self.failure {
            s.push_str(&format!("failure = {}\n", toml::Value::String(failure.clone())));
        }
        s.into_bytes()
    }

    /// Every replicate failed, so the run has no likelihood at all.
    pub fn every_replicate_failed(&self) -> bool {
        self.n_failed > 0 && self.n_failed == self.n_replicates
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Loglik, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| format!("{LOGLIK_FILE}: {e}"))?;
        toml::from_str(text).map_err(|e| format!("{LOGLIK_FILE}: {e}"))
    }
}

fn toml_float(x: f64) -> String {
    if x == f64::NEG_INFINITY {
        "-inf".into()
    } else if x == f64::INFINITY {
        "inf".into()
    } else if x.is_nan() {
        "nan".into()
    } else if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{x:.1}")
    } else {
        format!("{x}")
    }
}

/// Each replicate's seed and log-likelihood, in `replicates.tsv`.
pub(crate) fn replicates_bytes(rows: &[(u64, f64)]) -> Vec<u8> {
    let mut out = b"seed\tloglik\n".to_vec();
    for (seed, ll) in rows {
        let _ = writeln!(out, "{seed}\t{ll}");
    }
    out
}

fn read_replicates(bytes: &[u8]) -> Result<Vec<(u64, f64)>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("{REPLICATES_FILE}: {e}"))?;
    text.lines()
        .skip(1)
        .map(|line| {
            let (s, l) = line.split_once('\t').ok_or_else(|| format!("{REPLICATES_FILE}: a row without two columns"))?;
            let seed = s.parse::<u64>().map_err(|e| format!("{REPLICATES_FILE}: {e}"))?;
            let ll = l.parse::<f64>().map_err(|e| format!("{REPLICATES_FILE}: {e}"))?;
            Ok((seed, ll))
        })
        .collect()
}

/// The log-likelihood table `-o` writes and the terminal shows: one run's
/// log-likelihood to four decimals, or each replicate's seed and
/// log-likelihood.
pub(crate) struct LoglikTable;

#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
impl Rendering for LoglikTable {
    type Of = Pfilter;
    type Options = ();
    fn render(members: &[store::Member<Pfilter>], _options: &(), out: &mut dyn Write) -> store::Result<()> {
        let io = |e: std::io::Error| store::Error::Io { op: "write", path: "the log-likelihood table".into(), source: e };
        let parse = |why: String| store::Error::Parse { what: "a stored pfilter result".into(), why };
        for (stored, _) in members {
            let read = |name: &str| -> store::Result<Vec<u8>> {
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut stored.open(name)?, &mut bytes)
                    .map_err(|e| store::Error::Io { op: "read", path: name.into(), source: e })?;
                Ok(bytes)
            };
            let loglik = Loglik::from_bytes(&read(LOGLIK_FILE)?).map_err(parse)?;
            if loglik.n_replicates > 1 {
                writeln!(out, "seed\tloglik").map_err(io)?;
                for (seed, ll) in read_replicates(&read(REPLICATES_FILE)?).map_err(parse)? {
                    writeln!(out, "{seed}\t{ll:.4}").map_err(io)?;
                }
            } else {
                writeln!(out, "{:.4}", loglik.loglik.unwrap_or(f64::NEG_INFINITY)).map_err(io)?;
            }
        }
        Ok(())
    }
}

/// The table a `--sweep-file` design writes: one row per design point, in the
/// design file's order, with the point's id and design values from the
/// invocation's labels and the rest from the stored run. The points are
/// computed concurrently, so the invocation lists them in the order they
/// reached the store, and the rows are put back in file order here.
///
/// `loglik` is written in the shortest form that reads back to the stored
/// value, and `-inf` when a replicate failed; `loglik_sd` is empty for one
/// replicate and when a replicate failed; `failure` is the stored message on
/// one line; `run_id` is empty when the options leave it out.
pub(crate) struct DesignTable;

/// How a design table is rendered.
pub(crate) struct DesignTableOptions {
    /// Whether `run_id` names each point's stored result. Under `--no-store`
    /// the results live in a temporary store that is deleted, so it is empty.
    pub run_id: bool,
}

#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
impl Rendering for DesignTable {
    type Of = Pfilter;
    type Options = DesignTableOptions;
    fn render(members: &[store::Member<Pfilter>], options: &DesignTableOptions, out: &mut dyn Write) -> store::Result<()> {
        let io = |e: std::io::Error| store::Error::Io { op: "write", path: "the design table".into(), source: e };
        let parse = |why: String| store::Error::Parse { what: "a stored pfilter result".into(), why };
        let no_point = || parse("a member of the invocation is not a design point".into());
        let columns: Vec<&str> = match members.first() {
            Some((_, labels)) => labels.point.as_ref().ok_or_else(no_point)?.values.iter().map(|(k, _)| k.as_str()).collect(),
            None => Vec::new(),
        };
        let mut header = String::from("point_id");
        for c in &columns {
            header.push_str(&format!("\tsweep:{c}"));
        }
        writeln!(out, "{header}\tloglik\tloglik_sd\tn_failed\tfailure\trun_id").map_err(io)?;
        let mut rows = Vec::with_capacity(members.len());
        for (stored, labels) in members {
            rows.push((labels.point.as_ref().ok_or_else(no_point)?, stored));
        }
        rows.sort_by_key(|(point, _)| point.row);
        for (point, stored) in rows {
            let loglik = Loglik::from_bytes(&crate::camdl_store::read_stored(stored, LOGLIK_FILE).map_err(parse)?).map_err(parse)?;
            let mut row = point.point_id.to_string();
            for (_, v) in &point.values {
                row.push_str(&format!("\t{v}"));
            }
            let ll = loglik.loglik.unwrap_or(f64::NEG_INFINITY);
            let sd = loglik.loglik_sd.map(|sd| sd.to_string()).unwrap_or_default();
            let failure = loglik.failure.as_deref().map(one_line).unwrap_or_default();
            let run_id = if options.run_id { stored.id().key().to_hex() } else { String::new() };
            writeln!(out, "{row}\t{ll}\t{sd}\t{}\t{failure}\t{run_id}", loglik.n_failed).map_err(io)?;
        }
        Ok(())
    }
}

/// A message as one cell of a table: without a leading `error: `, and with
/// each tab and newline replaced by a space.
pub(crate) fn one_line(message: &str) -> String {
    message.strip_prefix("error: ").unwrap_or(message)
        .chars()
        .map(|c| if matches!(c, '\t' | '\n' | '\r') { ' ' } else { c })
        .collect()
}

#[cfg(test)]
mod tests;
