//! The files of a fit problem, a prediction and a score. A fit run's files
//! are [`super::fit_run`]'s.

use super::{find, Files, Format, ParseFile, ResultFile, Role};
use crate::fit_store::{FitProblem, Prediction, Score};

/// The files of a fit problem (`fit`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProblemFile {
    /// `problem.toml`, the problem as the fit file reads it.
    Problem,
    /// `partition.json`, which rows train.
    Partition,
    /// `inputs/<source>.<ext>`, one stream source's training lines as read.
    Training { source: String, ext: String },
    /// `inputs/tables/<name>.<ext>`, one external table as read.
    Table { name: String, ext: String },
    /// `held_out/<stream>.tsv`, one stream's held-out rows without their values.
    HeldOut { stream: String },
    /// `synthetic/truth.toml`, the parameters a generated dataset was drawn at.
    SyntheticTruth,
    /// `synthetic/truth.toml` of a real-data fit's `[synthetic]` table, the
    /// truth `fit recovery` compares with (provenance).
    RecoveryTruth,
    /// `synthetic.toml`, that `[synthetic]` table (provenance).
    RecoveryTable,
    /// `model.camdl` or `model.ir.json`, the model file (provenance).
    ModelSource { camdl: bool },
}

impl ResultFile for ProblemFile {
    fn logical(&self) -> String {
        match self {
            ProblemFile::Problem => "problem".into(),
            ProblemFile::Partition => "partition".into(),
            ProblemFile::Training { source, .. } => format!("inputs/{source}"),
            ProblemFile::Table { name, .. } => format!("inputs/tables/{name}"),
            ProblemFile::HeldOut { stream } => format!("held_out/{stream}"),
            ProblemFile::SyntheticTruth | ProblemFile::RecoveryTruth => "synthetic/truth".into(),
            ProblemFile::RecoveryTable => "synthetic".into(),
            ProblemFile::ModelSource { .. } => "model".into(),
        }
    }
    fn stored(&self) -> String {
        use crate::fit_store as fs;
        match self {
            ProblemFile::Problem => fs::PROBLEM_FILE.into(),
            ProblemFile::Partition => fs::PARTITION_FILE.into(),
            ProblemFile::Training { source, ext } => fs::training_file(source, ext),
            ProblemFile::Table { name, ext } => fs::table_file(name, ext),
            ProblemFile::HeldOut { stream } => fs::held_out_file(stream),
            ProblemFile::SyntheticTruth | ProblemFile::RecoveryTruth => fs::SYNTHETIC_TRUTH_FILE.into(),
            ProblemFile::RecoveryTable => fs::RECOVERY_TABLE_FILE.into(),
            ProblemFile::ModelSource { camdl: true } => "model.camdl".into(),
            ProblemFile::ModelSource { camdl: false } => "model.ir.json".into(),
        }
    }
    fn role(&self) -> Role {
        match self {
            ProblemFile::RecoveryTruth | ProblemFile::RecoveryTable | ProblemFile::ModelSource { .. } => Role::Provenance,
            _ => Role::Result,
        }
    }
    fn format(&self) -> Format {
        match self {
            ProblemFile::Problem | ProblemFile::SyntheticTruth | ProblemFile::RecoveryTruth | ProblemFile::RecoveryTable => Format::Toml,
            ProblemFile::Partition | ProblemFile::ModelSource { camdl: false } => Format::Json,
            ProblemFile::Training { ext, .. } | ProblemFile::Table { ext, .. } if ext == "json" => Format::Json,
            ProblemFile::Training { .. } | ProblemFile::Table { .. } | ProblemFile::HeldOut { .. } => Format::Tsv,
            ProblemFile::ModelSource { camdl: true } => Format::Camdl,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            ProblemFile::Problem => "The problem in fit-file syntax: the model, the data, the estimated and fixed parameters.",
            ProblemFile::Partition => "Which rows of each stream train, and where training ends.",
            ProblemFile::Training { .. } => "One stream source's training lines, as read.",
            ProblemFile::Table { .. } => "One external table, as read.",
            ProblemFile::HeldOut { .. } => "One stream's held-out rows, without their values.",
            ProblemFile::SyntheticTruth => "The true parameters the dataset was generated at.",
            ProblemFile::RecoveryTruth => "The true parameters `fit recovery` compares estimates with.",
            ProblemFile::RecoveryTable => "The fit file's `[synthetic]` table.",
            ProblemFile::ModelSource { .. } => "The model file the problem was compiled from.",
        }
    }
}

impl Files for FitProblem {
    type File = ProblemFile;
    const CONTRACT: u16 = 1;
}

impl ParseFile for ProblemFile {
    fn of_stored(stored: &str) -> Option<Self> {
        let fixed = [
            ProblemFile::Problem,
            ProblemFile::Partition,
            ProblemFile::SyntheticTruth,
            ProblemFile::RecoveryTable,
            ProblemFile::ModelSource { camdl: true },
            ProblemFile::ModelSource { camdl: false },
        ];
        if let Some(f) = find(fixed, stored) {
            return Some(f);
        }
        let named = |rest: &str| rest.rsplit_once('.').filter(|(n, e)| !n.is_empty() && !e.is_empty()).map(|(n, e)| (n.to_string(), e.to_string()));
        if let Some((name, ext)) = stored.strip_prefix("inputs/tables/").and_then(named) {
            return Some(ProblemFile::Table { name, ext });
        }
        if let Some((source, ext)) = stored.strip_prefix("inputs/").filter(|r| !r.contains('/')).and_then(named) {
            return Some(ProblemFile::Training { source, ext });
        }
        let stream = stored.strip_prefix("held_out/")?.strip_suffix(".tsv").filter(|s| !s.is_empty() && !s.contains('/'))?;
        Some(ProblemFile::HeldOut { stream: stream.into() })
    }
}

/// Which part of a prediction a file belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PredictionPart {
    Predictive,
    Observed,
    Forecast,
    Quantities,
    Contrasts,
    Report,
}

impl PredictionPart {
    const ALL: [PredictionPart; 6] = [
        PredictionPart::Predictive,
        PredictionPart::Observed,
        PredictionPart::Forecast,
        PredictionPart::Quantities,
        PredictionPart::Contrasts,
        PredictionPart::Report,
    ];

    fn name(self) -> &'static str {
        match self {
            PredictionPart::Predictive => "predictive",
            PredictionPart::Observed => "observed",
            PredictionPart::Forecast => "forecast",
            PredictionPart::Quantities => "quantities",
            PredictionPart::Contrasts => "contrasts",
            PredictionPart::Report => "report",
        }
    }

    /// The part a stored stem names: the part's name, alone or followed by
    /// `-` and a suffix (a quantities vocabulary's key, a chain exclusion).
    fn of_stem(stem: &str) -> Option<PredictionPart> {
        PredictionPart::ALL.into_iter().find(|p| stem == p.name() || stem.strip_prefix(p.name()).is_some_and(|r| r.starts_with('-')))
    }
}

/// The files of a prediction (`prediction`). Its manifests and tables are
/// named by part, with a suffix when the prediction excluded chains or reads
/// a quantities vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PredictionFile {
    /// `<stem>.json`: a part's manifest, or the report of what failed and
    /// which streams were skipped.
    Manifest { part: PredictionPart, stem: String },
    /// `<dir>/<stream>.tsv`: one stream's (or one quantity's) table of a part.
    Table { part: PredictionPart, dir: String, stream: String },
    /// `forecast-samples.tsv`, one row per forecast sample.
    ForecastSamples,
    /// `forecast-contrasts.tsv`, when the forecast has contrasts.
    ForecastContrasts,
}

impl PredictionFile {
    /// The part of the prediction the file belongs to.
    fn part(&self) -> Option<PredictionPart> {
        match self {
            PredictionFile::Manifest { part, .. } | PredictionFile::Table { part, .. } => Some(*part),
            _ => None,
        }
    }

    /// The file `stored` names, or `None` for a name no prediction stores.
    pub(crate) fn parse(stored: &str) -> Option<PredictionFile> {
        use crate::fit::forecast_output::{FORECAST_CONTRASTS_FILE, FORECAST_SAMPLES_FILE};
        if stored == FORECAST_SAMPLES_FILE {
            return Some(PredictionFile::ForecastSamples);
        }
        if stored == FORECAST_CONTRASTS_FILE {
            return Some(PredictionFile::ForecastContrasts);
        }
        match stored.split_once('/') {
            None => {
                let stem = stored.strip_suffix(".json")?;
                let part = PredictionPart::of_stem(stem)?;
                Some(PredictionFile::Manifest { part, stem: stem.into() })
            }
            Some((dir, file)) => {
                let stream = file.strip_suffix(".tsv").filter(|s| !s.is_empty() && !s.contains('/'))?;
                let part = PredictionPart::of_stem(dir).filter(|p| *p != PredictionPart::Report)?;
                Some(PredictionFile::Table { part, dir: dir.into(), stream: stream.into() })
            }
        }
    }
}

impl PredictionFile {
    /// The files a prediction holds, from what `camdl predict` wrote in
    /// `dir`, its scratch directory: every manifest there, and the tables
    /// the manifests list. The predictive and observed manifests list each
    /// stream's table (`streams[].file`), and a stream they skip has none;
    /// the forecast manifest lists its samples and contrasts files and
    /// names the streams it produced, each with its table under `forecast/`;
    /// the quantities manifest names each quantity, whose table is
    /// `<name>.tsv`; the contrasts manifest lists each contrast's table
    /// (`contrasts[].file`). A table that no manifest lists is left out, so
    /// the store refuses it, and a listed table never written is in the set,
    /// so the store refuses the run as missing it. A file no prediction
    /// stores is refused by name.
    #[expect(clippy::disallowed_methods, reason = "reads the manifests `camdl predict` wrote in its scratch directory, before they are written into the run, to declare the tables they list")]
    pub(crate) fn of_dir(dir: &std::path::Path) -> Result<Vec<PredictionFile>, String> {
        let mut names = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))? {
                let path = e.map_err(|e| e.to_string())?.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    names.push(path.strip_prefix(dir).expect("under the scratch directory").to_string_lossy().into_owned());
                }
            }
        }
        names.sort();
        let found: Vec<PredictionFile> = names
            .iter()
            .map(|n| PredictionFile::parse(n).ok_or_else(|| format!("internal error: the prediction wrote `{n}`, which is not a file a prediction stores")))
            .collect::<Result<_, _>>()?;
        let mut declared: Vec<PredictionFile> = Vec::new();
        // The parts whose manifest lists every table it holds.
        let mut listing: Vec<PredictionPart> = Vec::new();
        for file in &found {
            let PredictionFile::Manifest { part, stem } = file else { continue };
            declared.push(file.clone());
            if *part == PredictionPart::Report {
                continue;
            }
            listing.push(*part);
            let path = dir.join(file.stored());
            let text = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let manifest: serde_json::Value =
                serde_json::from_slice(&text).map_err(|e| format!("internal error: the prediction's {}: {e}", file.stored()))?;
            let mut listed = Vec::new();
            if *part == PredictionPart::Quantities {
                // Each entry names one quantity, whose table is `<name>.tsv`;
                // a quantity evaluated in several design cells has an entry
                // per cell and one table. The model's own block writes its
                // manifest as `quantities[-<suffix>].json` and its tables
                // under `quantities-model[-<suffix>]/` (the store keeps a
                // kind's name, `quantities`, out of a result's top-level
                // directories); a vocabulary's manifest carries its
                // `vocabulary` and names its directory.
                let table_dir = if manifest.get("vocabulary").is_some() {
                    stem.clone()
                } else {
                    stem.replacen("quantities", "quantities-model", 1)
                };
                for name in manifest["quantities"].as_array().into_iter().flatten().filter_map(|q| q["name"].as_str()) {
                    listed.push(PredictionFile::Table { part: PredictionPart::Quantities, dir: table_dir.clone(), stream: name.into() });
                }
                declared.extend(listed);
                continue;
            }
            listed_files(&manifest, &mut listed);
            if *part == PredictionPart::Forecast {
                for s in manifest["produced"].as_array().into_iter().flatten().filter_map(|s| s.as_str()) {
                    listed.push(PredictionFile::Table {
                        part: PredictionPart::Forecast,
                        dir: crate::fit::forecast_output::FORECAST_TABLES_DIR.into(),
                        stream: s.into(),
                    });
                }
            }
            declared.extend(listed);
        }
        for file in found {
            let unlisted_part = match &file {
                PredictionFile::Table { part, .. } => !listing.contains(part),
                PredictionFile::ForecastSamples | PredictionFile::ForecastContrasts => !listing.contains(&PredictionPart::Forecast),
                PredictionFile::Manifest { .. } => false,
            };
            if unlisted_part {
                declared.push(file);
            }
        }
        declared.sort_by_key(ResultFile::stored);
        declared.dedup();
        Ok(declared)
    }
}

/// Every string in `v` that names a table of a prediction, at any depth.
fn listed_files(v: &serde_json::Value, out: &mut Vec<PredictionFile>) {
    match v {
        serde_json::Value::String(s) => {
            if let Some(f) = PredictionFile::parse(s).filter(|f| !matches!(f, PredictionFile::Manifest { .. })) {
                out.push(f);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| listed_files(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| listed_files(x, out)),
        _ => {}
    }
}

impl ResultFile for PredictionFile {
    fn logical(&self) -> String {
        match self {
            PredictionFile::Manifest { stem, .. } => stem.clone(),
            PredictionFile::Table { dir, stream, .. } => format!("{dir}/{stream}"),
            PredictionFile::ForecastSamples => "forecast_samples".into(),
            PredictionFile::ForecastContrasts => "forecast_contrasts".into(),
        }
    }
    fn stored(&self) -> String {
        use crate::fit::forecast_output::{FORECAST_CONTRASTS_FILE, FORECAST_SAMPLES_FILE};
        match self {
            PredictionFile::Manifest { stem, .. } => format!("{stem}.json"),
            PredictionFile::Table { dir, stream, .. } => format!("{dir}/{stream}.tsv"),
            PredictionFile::ForecastSamples => FORECAST_SAMPLES_FILE.into(),
            PredictionFile::ForecastContrasts => FORECAST_CONTRASTS_FILE.into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            PredictionFile::Manifest { .. } => Format::Json,
            _ => Format::Tsv,
        }
    }
    fn about(&self) -> &'static str {
        match (self, self.part()) {
            (PredictionFile::Manifest { .. }, Some(PredictionPart::Report)) => "What failed while the prediction was made, and which streams were skipped.",
            (PredictionFile::Manifest { .. }, _) => "A part's manifest: its schema, its tables and how they were made.",
            (PredictionFile::Table { .. }, Some(PredictionPart::Contrasts)) => "One contrast's bands, per time.",
            (PredictionFile::Table { .. }, Some(PredictionPart::Quantities)) => "One quantity's bands, per time.",
            (PredictionFile::Table { .. }, Some(PredictionPart::Observed)) => "One stream's observed values: `time`, a column per dimension of the stream, and `value`.",
            (PredictionFile::Table { .. }, _) => "One stream's predictive or forecast bands, per time, mode and quantity.",
            (PredictionFile::ForecastSamples, _) => "One row per forecast sample: its draw, particle, stream, time and value.",
            (PredictionFile::ForecastContrasts, _) => "Each forecast contrast's samples.",
        }
    }
}

impl Files for Prediction {
    type File = PredictionFile;
    const CONTRACT: u16 = 2;
}

impl ParseFile for PredictionFile {
    fn of_stored(stored: &str) -> Option<Self> {
        PredictionFile::parse(stored)
    }
}

/// The files of a score (`score`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoreFile {
    /// `scored.tsv`, one row per scored row or target.
    Scored,
    /// `summary.tsv`, one row per stream, mode, role and lead.
    Summary,
    /// `score.json`: the parameter source, the basis, the conditioning and
    /// the notes.
    Score,
}

impl ScoreFile {
    pub(crate) const ALL: [ScoreFile; 3] = [ScoreFile::Scored, ScoreFile::Summary, ScoreFile::Score];
}

impl ResultFile for ScoreFile {
    fn logical(&self) -> String {
        match self {
            ScoreFile::Scored => "scored",
            ScoreFile::Summary => "summary",
            ScoreFile::Score => "score",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            ScoreFile::Scored => "scored.tsv",
            ScoreFile::Summary => "summary.tsv",
            ScoreFile::Score => "score.json",
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            ScoreFile::Score => Format::Json,
            _ => Format::Tsv,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            ScoreFile::Scored => "One row per scored row or target.",
            ScoreFile::Summary => "One row per stream, mode, role and lead.",
            ScoreFile::Score => "The parameter source, the basis, the conditioning and the notes.",
        }
    }
}

impl Files for Score {
    type File = ScoreFile;
    const CONTRACT: u16 = 1;
}

impl ParseFile for ScoreFile {
    fn of_stored(stored: &str) -> Option<Self> {
        find(ScoreFile::ALL, stored)
    }
}
