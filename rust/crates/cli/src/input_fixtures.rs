//! The second values and the fixtures the input-class checks use
//! (`store::check_input_classes`): each `vary = ...` and `exercised_by = ...`
//! of a command's declared inputs names a function here.
//!
//! Every path a function here returns is relative, and names a file of the
//! fixture directory the command's class check runs in
//! (`crate::pfilter_store::tests`, `crate::sim_store::tests`), which is the
//! working directory of each run. The values are meaningful only there.

use std::path::PathBuf;

use crate::args::types::{DataSpec, ParamOverride, TableSpec};
use crate::args::{FitRunArgs, FitRunInvocation, PfilterCore, PfilterInvocation, SimulateArgs, SimulateInvocation};

/// The pfilter every check starts from:
/// `pfilter sir.ir.json --params p.toml --data cases=cases.tsv --particles 20 --seed 1`.
pub(crate) fn pfilter_base() -> PfilterInvocation {
    pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv", "--particles", "20", "--seed", "1"])
}

/// A pfilter from its command line, without the program and subcommand
/// names, and with the global options at their defaults.
pub(crate) fn pfilter_from(args: &[&str]) -> PfilterInvocation {
    #[derive(clap::Parser)]
    struct Line {
        #[command(flatten)]
        args: crate::args::PfilterArgs,
    }
    let line = <Line as clap::Parser>::try_parse_from(std::iter::once("pfilter").chain(args.iter().copied()))
        .unwrap_or_else(|e| panic!("a fixture's pfilter command line does not parse: {e}"));
    PfilterInvocation {
        args: line.args,
        no_licm: false,
        no_ir_cache: false,
        verbosity: None,
        progress: crate::args::types::ProgressMode::None,
        no_progress: false,
    }
}

pub(crate) fn other_model() -> PathBuf {
    "sir_other.ir.json".into()
}

pub(crate) fn other_params() -> Vec<PathBuf> {
    vec!["p2.toml".into()]
}

pub(crate) fn other_param() -> Vec<ParamOverride> {
    vec![ParamOverride { name: "beta_a".into(), value: 0.45 }]
}

pub(crate) fn other_table() -> Vec<TableSpec> {
    vec![TableSpec { name: "w".into(), path: "w2.tsv".into() }]
}

pub(crate) fn other_scenario() -> Option<String> {
    Some("set_fast".into())
}

/// An intervention of the fixture model, `sia`.
pub(crate) fn an_intervention() -> Vec<String> {
    vec!["sia".into()]
}

pub(crate) fn other_stream() -> Option<String> {
    Some("prevalent".into())
}

pub(crate) fn other_dt() -> Option<f64> {
    Some(0.5)
}

pub(crate) fn other_binomial() -> Option<sim::rng::BinomialAlgorithm> {
    Some(sim::rng::BinomialAlgorithm::Btrs)
}

pub(crate) fn other_data() -> Vec<DataSpec> {
    vec![DataSpec::Named { name: "cases".into(), path: "cases2.tsv".into() }]
}

/// A fit config whose `[config] allow_degenerate_rates` the flags do not set.
pub(crate) fn other_fit() -> Option<PathBuf> {
    Some("fit_adr.toml".into())
}

pub(crate) fn a_final_state() -> Option<PathBuf> {
    Some("state.tsv".into())
}

pub(crate) fn a_score_from() -> Option<String> {
    Some("5".into())
}

/// `--parallel 1` and `--parallel 4`.
pub(crate) fn pfilter_parallel() -> (PfilterCore, PfilterCore) {
    let base = pfilter_base().args.inference;
    (PfilterCore { parallel: 1, ..base.clone() }, PfilterCore { parallel: 4, ..base })
}

/// `--time-format auto` and `--time-format numeric` on a numeric column.
pub(crate) fn pfilter_time_format() -> (PfilterCore, PfilterCore) {
    use crate::caltime_load::TimeFormat;
    let base = pfilter_base().args.inference;
    (PfilterCore { time_format: TimeFormat::Auto, ..base.clone() }, PfilterCore { time_format: TimeFormat::Numeric, ..base })
}

/// The engine's budget and 400, the cost of one pass of the base fixture
/// (twenty particles over twenty one-day windows): the tightest budget that
/// admits the run.
pub(crate) fn pfilter_max_substeps() -> (PfilterCore, PfilterCore) {
    let base = pfilter_base().args.inference;
    (PfilterCore { pf_max_substeps: None, ..base.clone() }, PfilterCore { pf_max_substeps: Some(400), ..base })
}

/// The model from source, compiled with and without the IR cache.
pub(crate) fn pfilter_no_ir_cache() -> (PfilterInvocation, PfilterInvocation) {
    let mut base = pfilter_base();
    base.args.model = "sir.camdl".into();
    (PfilterInvocation { no_ir_cache: false, ..base.clone() }, PfilterInvocation { no_ir_cache: true, ..base })
}

// ── simulate ─────────────────────────────────────────────────────────────────

/// The simulate every check starts from: two seeds of the SIR model, so an
/// execution input has more than one cell to act on.
/// `simulate sir.ir.json --params p.toml --seeds 1,2`.
pub(crate) fn sim_base() -> SimulateInvocation {
    sim_from(&["sir.ir.json", "--params", "p.toml", "--seeds", "1,2"])
}

/// A simulate from its command line, without the program and subcommand
/// names, and with the global options at their defaults.
pub(crate) fn sim_from(args: &[&str]) -> SimulateInvocation {
    #[derive(clap::Parser)]
    struct Line {
        #[command(flatten)]
        args: SimulateArgs,
    }
    let line = <Line as clap::Parser>::try_parse_from(std::iter::once("simulate").chain(args.iter().copied()))
        .unwrap_or_else(|e| panic!("a fixture's simulate command line does not parse: {e}"));
    SimulateInvocation {
        args: line.args,
        no_licm: false,
        no_ir_cache: false,
        verbosity: None,
        progress: crate::args::types::ProgressMode::None,
        no_progress: false,
    }
}

pub(crate) fn sim_other_model() -> PathBuf {
    "sir_other.ir.json".into()
}

pub(crate) fn sim_other_to() -> Option<String> {
    Some("15".into())
}

pub(crate) fn sim_other_from() -> Option<String> {
    Some("2".into())
}

/// A saved state of two particles, so two replicates restore one row each.
pub(crate) fn sim_init_state() -> Option<String> {
    Some("state.tsv".into())
}

pub(crate) fn sim_param_vec() -> Vec<crate::args::types::ParamVecSpec> {
    vec![crate::args::types::ParamVecSpec { prefix: "beta".into(), file: "beta_vec.tsv".into() }]
}

/// A fit config whose `[fixed]` block fills the parameters a draws file
/// lacks.
pub(crate) fn sim_fit() -> Option<PathBuf> {
    Some("fit2.toml".into())
}

pub(crate) fn sim_emit_every() -> Vec<String> {
    vec!["2".into()]
}

pub(crate) fn sim_vocabulary() -> Option<PathBuf> {
    Some("vocabulary.camdl".into())
}

pub(crate) fn other_backend() -> Option<crate::args::types::ForwardBackend> {
    Some(crate::args::types::ForwardBackend::Gillespie)
}

pub(crate) fn other_integrator() -> Option<crate::args::types::IntegratorArg> {
    Some(crate::args::types::IntegratorArg::Rk45)
}

pub(crate) fn other_every() -> Option<f64> {
    Some(2.0)
}

pub(crate) fn other_columns() -> Vec<String> {
    vec!["I".into()]
}

/// `--parallel` unset and `--parallel 4` over the base's two cells.
pub(crate) fn sim_parallel() -> (SimulateArgs, SimulateArgs) {
    let base = sim_base().args;
    (SimulateArgs { parallel: None, ..base.clone() }, SimulateArgs { parallel: Some(4), ..base })
}

/// The model from source, compiled with and without the IR cache.
pub(crate) fn sim_no_ir_cache() -> (SimulateInvocation, SimulateInvocation) {
    let mut base = sim_base();
    base.args.model = "sir.camdl".into();
    (SimulateInvocation { no_ir_cache: false, ..base.clone() }, SimulateInvocation { no_ir_cache: true, ..base })
}

// ── fit run ──────────────────────────────────────────────────────────────────

/// The fit every check starts from: a short IF2 fit of the SIR model to
/// `cases.tsv`. `fit run fit.toml`.
pub(crate) fn fit_base() -> FitRunInvocation {
    fit_from(&["fit.toml"])
}

/// A fit run from its command line, without the program and subcommand
/// names, and with the global options at their defaults.
pub(crate) fn fit_from(args: &[&str]) -> FitRunInvocation {
    #[derive(clap::Parser)]
    struct Line {
        #[command(flatten)]
        args: FitRunArgs,
    }
    let line = <Line as clap::Parser>::try_parse_from(std::iter::once("run").chain(args.iter().copied()))
        .unwrap_or_else(|e| panic!("a fixture's fit run command line does not parse: {e}"));
    FitRunInvocation {
        args: line.args,
        no_licm: false,
        no_ir_cache: false,
        verbosity: None,
        progress: crate::args::types::ProgressMode::None,
        no_progress: false,
    }
}

/// A fit file whose problem differs from the base's: another `[fixed]`
/// value.
pub(crate) fn fit_other_config() -> PathBuf {
    "fit_other.toml".into()
}

/// A cadence the `[synthetic]` generator emits its prevalence stream at.
pub(crate) fn fit_emit_every() -> Vec<String> {
    vec!["prevalent=2".into()]
}

pub(crate) fn fit_starts() -> Option<crate::fit::starts::ChainStarts> {
    Some("lhs".parse().expect("`lhs` is a starts rule"))
}

pub(crate) fn fit_starts_chains() -> Option<crate::chain_selection::ChainSelection> {
    Some(crate::fit::starts::parse_starts_chains("0").expect("`0` is a chain selection"))
}

/// The run the fixture's harness stores under `@base` before a resume.
pub(crate) fn fit_resume() -> Option<String> {
    Some("@base".into())
}

/// Without and with `--allow-nonconverged-source`, starting from a draws
/// file named directly, which carries no verdict to lift.
pub(crate) fn fit_allow_nonconverged_source() -> (FitRunArgs, FitRunArgs) {
    let base = fit_from(&["fit_pmmh.toml", "--starts", "from_posterior=draws.tsv"]).args;
    (FitRunArgs { allow_nonconverged_source: false, ..base.clone() }, FitRunArgs { allow_nonconverged_source: true, ..base })
}

/// `--parallel 1` and `--parallel 4`.
pub(crate) fn fit_parallel() -> (FitRunArgs, FitRunArgs) {
    let base = fit_base().args;
    (FitRunArgs { parallel: 1, ..base.clone() }, FitRunArgs { parallel: 4, ..base })
}

/// The model from source, compiled with and without the IR cache.
pub(crate) fn fit_no_ir_cache() -> (FitRunInvocation, FitRunInvocation) {
    let base = fit_from(&["fit_source.toml"]);
    (FitRunInvocation { no_ir_cache: false, ..base.clone() }, FitRunInvocation { no_ir_cache: true, ..base })
}
