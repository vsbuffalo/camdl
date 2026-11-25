//! The run spec's result-file contract, rendered from the file enums.
//!
//! Each kind lists its files as variants of its enum, with the condition
//! under which a result holds each one. [`render`] writes the section of
//! `docs/camdl-run-spec.md` between [`BEGIN`] and [`END`]; the test
//! `the_contract_section_is_generated` refuses a committed section that
//! differs from it, and `make update-result-file-contract` writes it.
//! `camdl docs result-files` prints it.

use super::fit::{PredictionFile, PredictionPart, ProblemFile, ScoreFile};
use super::fit_run::{
    CommonFile, FitRunFile, If2File, McmcFile, NloptFile, PfilterMethodFile, PgasFile, Sampler,
};
use super::model_source::SourceFile;
use super::pfilter::{
    FilteredStreamsFile, FilteringFile, FinalStateFile, PfHealthFile, PfilterFile, PrequentialFile, SavedPathsFile,
    TraceFile,
};
use super::profile::ProfilePointFile;
use super::sim::{EventLogFile, ObsFile, SimFile, SimQuantitiesFile};
use super::{Column, ResultFile, Role};

/// The line that opens the generated section.
#[cfg(test)]
pub(crate) const BEGIN: &str = "<!-- The result-file contract below is generated from the file enums of \
rust/crates/cli/src/result_files; run `make update-result-file-contract` after changing them. -->";
/// The line that closes it.
#[cfg(test)]
pub(crate) const END: &str = "<!-- End of the generated result-file contract. -->";

/// One file of a kind, with when a result holds it.
pub(super) struct Row {
    pub(super) logical: String,
    pub(super) stored: String,
    role: Role,
    pub(super) format: &'static str,
    pub(super) columns: &'static [Column],
    commented: bool,
    about: &'static str,
    when: &'static str,
}

fn row<F: ResultFile>(f: F, when: &'static str) -> Row {
    let template = |s: String| s.replace("chain_0", "chain_N");
    Row {
        logical: template(f.logical()),
        stored: template(f.stored()),
        role: f.role(),
        format: f.format().name(),
        columns: f.columns(),
        commented: f.commented(),
        about: f.about(),
        when,
    }
}

fn stream() -> String {
    "<stream>".into()
}

/// Each kind: its name, a sentence on what it is, and its files.
pub(super) fn kinds() -> Vec<(&'static str, &'static str, Vec<Row>)> {
    use CommonFile as C;
    let every = "every method";
    let mcmc = "mh, pmmh, pgas";
    vec![
        (
            "fit_run",
            "One fit run of a problem. Its files depend on its method and on how each chain ended, as `chain_starts.tsv` records it: `accepted` (ran to the end), `skipped` (ran and was dropped, IF2) or `refused` (never ran).",
            vec![
                row(FitRunFile::Common(C::Method), every),
                row(FitRunFile::Common(C::Summary), every),
                row(FitRunFile::Common(C::StartsInput("<ext>".into())), "a starts file named directly"),
                row(FitRunFile::Mcmc(McmcFile::Draws), mcmc),
                row(FitRunFile::Mcmc(McmcFile::ChainTrace(Sampler::Mh, 0)), "mh, pmmh, each chain; a refused chain's holds no row"),
                row(FitRunFile::Mcmc(McmcFile::ChainTrace(Sampler::Pgas, 0)), "pgas, each chain; a refused chain's holds no row"),
                row(FitRunFile::Mcmc(McmcFile::ResumeState(0)), "mh, pmmh, pgas, each chain that ran"),
                row(FitRunFile::Mcmc(McmcFile::SamplerSummary(Sampler::Mh)), "mh; pmmh and pgas as `pmmh_summary.json` and `pgas_summary.json`"),
                row(FitRunFile::Mcmc(McmcFile::ChainStarts), "if2, mh, pmmh, pgas"),
                row(FitRunFile::Mcmc(McmcFile::Diagnostics), "every method but pfilter"),
                row(FitRunFile::Mcmc(McmcFile::FitState), "every method but pfilter"),
                row(FitRunFile::Pgas(PgasFile::Trajectories(0)), "pgas, each chain that ran to the end and saved paths (`n_trajectories` above zero, `trajectories.n_saved` above zero in the summary)"),
                row(FitRunFile::Pgas(PgasFile::TrajectoriesMeta(0)), "as `chain_N/trajectories`"),
                row(FitRunFile::Pgas(PgasFile::CrossChainCompat), "pgas, when two or more chains ran to the end"),
                row(FitRunFile::Pgas(PgasFile::FilterEss), "pgas, when its summary holds the `filter_ess` block"),
                row(FitRunFile::Pgas(PgasFile::LatentConvergence), "pgas, when its summary holds the `latent_convergence` block"),
                row(FitRunFile::If2(If2File::ParameterTraces(0)), "if2, each chain: IF2 runs a chain whose every start was refused from its last one"),
                row(FitRunFile::If2(If2File::ChainFinalParams(0)), "if2, each chain that produced a result, as its row in `chain_evaluations.tsv` records"),
                row(FitRunFile::If2(If2File::FinalParams), "if2"),
                row(FitRunFile::If2(If2File::MleParams), "if2, nl-sbplx, nl-bobyqa, nl-lbfgs"),
                row(FitRunFile::If2(If2File::ChainEvaluations), "if2"),
                row(FitRunFile::If2(If2File::IterationDiagnostics), "if2"),
                row(FitRunFile::Nlopt(NloptFile::ChainResults), "nl-sbplx, nl-bobyqa, nl-lbfgs"),
                row(FitRunFile::Pfilter(PfilterMethodFile::Logliks), "pfilter"),
                row(FitRunFile::Pfilter(PfilterMethodFile::PrequentialJoint), "pfilter, when `record_prequential` is on"),
                row(FitRunFile::Pfilter(PfilterMethodFile::PrequentialStreams), "pfilter, when `record_prequential` is on"),
                row(FitRunFile::Pfilter(PfilterMethodFile::PrequentialRecord), "pfilter, when `record_prequential` is on"),
                row(FitRunFile::Common(C::FitConfig), every),
                row(FitRunFile::Common(C::RunInputs), every),
                row(FitRunFile::Common(C::FitMeta), every),
                row(FitRunFile::Common(C::Execution), every),
                row(FitRunFile::Nlopt(NloptFile::OptTrace), "nl-sbplx, nl-bobyqa, nl-lbfgs"),
            ],
        ),
        (
            "model_source",
            "The model file one command compiled, with its name, what the compiler shows of it, and the quantities its compile produced, on which it depends. No simulation or fit depends on it; the compiled model's listing in each invocation record names, as its `source`, the model source of the file that invocation compiled. A command that stores a result from a stored one (`predict RUN`, `score`, `profile MODEL --fit RUN`) lists the stored compiled model in its own record with the source of the file it was given or, when it was given none, the one the stored result's invocation recorded.",
            vec![
                row(SourceFile::Source { camdl: true }, "always; `source.ir.json` for a compiled model"),
                row(SourceFile::Docs, "always"),
                row(SourceFile::Declared, "always"),
                row(SourceFile::Render, "a `.camdl` model that camdlc rendered"),
                row(SourceFile::Graph, "a `.camdl` model that camdlc rendered"),
            ],
        ),
        (
            "fit",
            "A fit problem: the model, the data and the rows that train.",
            vec![
                row(ProblemFile::Problem, "always"),
                row(ProblemFile::Partition, "always"),
                row(ProblemFile::Training { source: "<source>".into(), ext: "<ext>".into() }, "each stream source"),
                row(ProblemFile::Table { name: "<name>".into(), ext: "<ext>".into() }, "each external table"),
                row(ProblemFile::HeldOut { stream: stream() }, "each stream with held-out rows"),
                row(ProblemFile::SyntheticTruth, "a problem on a generated dataset"),
                row(ProblemFile::RecoveryTruth, "a real-data fit with a `[synthetic]` table"),
                row(ProblemFile::RecoveryTable, "a real-data fit with a `[synthetic]` table"),
                row(ProblemFile::ModelSource { camdl: true }, "always; `model.ir.json` for a compiled model"),
            ],
        ),
        (
            "prediction",
            "A prediction from a fit run's posterior or from a fixed point. A name carries a suffix when chains were excluded.",
            vec![
                row(PredictionFile::Manifest { part: PredictionPart::Predictive, stem: "predictive".into() }, "always"),
                row(PredictionFile::Table { part: PredictionPart::Predictive, dir: "predictive".into(), stream: stream() }, "each stream predicted"),
                row(PredictionFile::Manifest { part: PredictionPart::Observed, stem: "observed".into() }, "when observations are bound"),
                row(PredictionFile::Table { part: PredictionPart::Observed, dir: "observed".into(), stream: stream() }, "each stream observed"),
                row(PredictionFile::Manifest { part: PredictionPart::Forecast, stem: "forecast".into() }, "a forecast"),
                row(PredictionFile::Table { part: PredictionPart::Forecast, dir: "forecast".into(), stream: stream() }, "a forecast, each stream"),
                row(PredictionFile::ForecastSamples, "a forecast"),
                row(PredictionFile::ForecastContrasts, "a forecast with contrasts"),
                row(PredictionFile::Manifest { part: PredictionPart::Quantities, stem: "quantities".into() }, "when the model has quantities"),
                row(PredictionFile::Table { part: PredictionPart::Quantities, dir: "quantities-model".into(), stream: "<quantity>".into() }, "each quantity"),
                row(PredictionFile::Manifest { part: PredictionPart::Contrasts, stem: "contrasts".into() }, "a fit run's prediction when the model has contrasts"),
                row(PredictionFile::Table { part: PredictionPart::Contrasts, dir: "contrasts".into(), stream: "<contrast>".into() }, "each contrast forked"),
                row(PredictionFile::Manifest { part: PredictionPart::Report, stem: "report".into() }, "always"),
            ],
        ),
        ("score", "A stored prediction scored against its outcomes.", ScoreFile::ALL.into_iter().map(|f| row(f, "always")).collect()),
        (
            "pfilter",
            "A particle filter run at one parameter point.",
            vec![
                row(PfilterFile::Loglik, "always"),
                row(PfilterFile::Replicates, "with `--replicates` above 1"),
                row(PfilterFile::Provenance, "always"),
            ],
        ),
        ("trace", "`--trace`, a by-product of a filter run.", vec![row(TraceFile::File, "always")]),
        ("pf_health", "`--pf-health`, a by-product of a filter run.", vec![row(PfHealthFile::File, "always")]),
        ("filtering", "`--save-filtering`, a by-product of a filter run.", vec![row(FilteringFile::File, "always")]),
        ("filtered_streams", "`--save-filtered-streams`, a by-product of a filter run.", vec![row(FilteredStreamsFile::File, "always")]),
        ("final_state", "`--save-final-state`, a by-product of a filter run.", vec![row(FinalStateFile::File, "always")]),
        ("saved_paths", "`--save-paths`, a by-product of a filter run.", vec![row(SavedPathsFile::File, "always")]),
        (
            "prequential",
            "`--save-prequential`, a by-product of a filter run.",
            [PrequentialFile::Joint, PrequentialFile::Streams, PrequentialFile::Record].into_iter().map(|f| row(f, "always")).collect(),
        ),
        (
            "sim",
            "One simulated cell.",
            vec![
                row(SimFile::Traj, "when the engine ran"),
                row(SimFile::ReactiveLog, "when the run kept a reactive policy's firings"),
                row(SimFile::Failure, "in place of `traj` when the engine stopped with an error"),
                row(SimFile::Provenance, "always"),
            ],
        ),
        (
            "obs",
            "`--obs`, a cell's simulated observations, a by-product of the cell.",
            vec![row(ObsFile::Stream(stream()), "each stream"), row(ObsFile::Failure, "in place of the streams when they could not be drawn")],
        ),
        (
            "event_log",
            "`--event-log`, a by-product of the cell.",
            vec![row(EventLogFile::EventLog, "when it was recorded"), row(EventLogFile::Failure, "in its place when it could not be")],
        ),
        (
            "sim_quantities",
            "`--quantities-out`, a by-product of the cell.",
            vec![
                row(SimQuantitiesFile::Quantities, "when they were evaluated"),
                row(SimQuantitiesFile::Failure, "in their place when they could not be"),
            ],
        ),
        ("profile_point", "One point of a profile.", ProfilePointFile::ALL.into_iter().map(|f| row(f, "always")).collect()),
    ]
}

fn columns(cs: &[Column]) -> String {
    cs.iter()
        .map(|c| match c {
            Column::Named(n) => format!("`{n}`"),
            Column::Each(what) => (*what).to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The generated section, without its two marker lines.
pub(crate) fn render() -> String {
    let mut s = String::new();
    s.push_str(
        "Each kind's files, by logical name: the stored name, the format, when a result holds the file, \
         what it holds and, for a table, its header row. A provenance file is recorded with its digest \
         and not compared when a result is computed again. `chain_N` stands for each chain, numbered \
         from 0. A run that holds any other file, or lacks one it declared, is not stored.\n",
    );
    s.push_str(&format!(
        "\nEach kind's contract version, which an output manifest records with each result (§2.10), is \
         given with its files; it is bumped when a file is added, removed or renamed, or a column changes. \
         Each of the three content artifacts has a contract version of its own: `compiled_model` {}, \
         `quantities` {} and `data_table` {}. The logical name of each one's file is its stored name up to \
         the first `.`: `model`, `quantities` and `table`.\n",
        super::COMPILED_MODEL_CONTRACT,
        super::QUANTITIES_CONTRACT,
        super::DATA_TABLE_CONTRACT
    ));
    for (kind, what, rows) in kinds() {
        let version = super::contract(kind).expect("every kind in the contract has a version").version;
        s.push_str(&format!("\n#### `{kind}`\n\n{what} Contract version {version}.\n\n"));
        for r in rows {
            let role = if r.role == Role::Provenance { ", provenance" } else { "" };
            s.push_str(&format!("- `{}`: `{}`, {}{} ({}). {}", r.logical, r.stored, r.format, role, r.when, r.about));
            if !r.columns.is_empty() {
                let after = if r.commented { " after `#` comment lines" } else { "" };
                s.push_str(&format!(" Header{after}: {}.", columns(r.columns)));
            }
            s.push('\n');
        }
    }
    s
}

/// `text` as its Markdown blocks, one per line: a heading, a list item, a
/// comment or a paragraph, with its continuation lines joined to it and every
/// run of spaces made one. A reflow of the committed section by the Markdown
/// formatter compares equal; a list item merged into another, a heading
/// joined to the text after it, or a block split in two does not.
#[cfg(test)]
pub(crate) fn normalized(text: &str) -> String {
    let mut blocks: Vec<String> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            open = false;
            continue;
        }
        let heading = line.starts_with('#');
        let starts_block = line.starts_with("- ") || line.starts_with('#') || line.starts_with("<!--");
        match blocks.last_mut() {
            Some(last) if open && !starts_block => {
                last.push(' ');
                last.push_str(&line);
            }
            _ => blocks.push(line),
        }
        // A heading is one line; anything else continues until a blank line
        // or the next block.
        open = !heading;
    }
    blocks.join("\n")
}
