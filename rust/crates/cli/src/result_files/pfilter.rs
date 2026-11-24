//! The files of a filter run and of its by-products.

use super::{Files, Format, ResultFile, Role};
use crate::pfilter_store::{
    FilteredStreams, Filtering, FinalState, Pfilter, PfHealth, Prequential, SavedPaths, Trace, LOGLIK_FILE,
    PROVENANCE_FILE, REPLICATES_FILE,
};

/// The files of a filter run (`pfilter`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PfilterFile {
    /// `loglik.toml`, the log-likelihood estimate.
    Loglik,
    /// `replicates.tsv`, each replicate filter's estimate, with
    /// `--replicates` above 1.
    Replicates,
    /// `provenance.json`, the parameter values as resolved (provenance).
    Provenance,
}

impl ResultFile for PfilterFile {
    fn logical(&self) -> String {
        match self {
            PfilterFile::Loglik => "loglik",
            PfilterFile::Replicates => "replicates",
            PfilterFile::Provenance => "provenance",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            PfilterFile::Loglik => LOGLIK_FILE,
            PfilterFile::Replicates => REPLICATES_FILE,
            PfilterFile::Provenance => PROVENANCE_FILE,
        }
        .into()
    }
    fn role(&self) -> Role {
        match self {
            PfilterFile::Provenance => Role::Provenance,
            _ => Role::Result,
        }
    }
    fn format(&self) -> Format {
        match self {
            PfilterFile::Loglik => Format::Toml,
            PfilterFile::Replicates => Format::Tsv,
            PfilterFile::Provenance => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            PfilterFile::Loglik => "`loglik`, `n_replicates` and `n_particles`: the log-likelihood estimate.",
            PfilterFile::Replicates => "Each replicate filter's log-likelihood.",
            PfilterFile::Provenance => "The parameter values as resolved, with where each came from.",
        }
    }
}

impl Files for Pfilter {
    type File = PfilterFile;
}

impl PfilterFile {
    /// The files of a run that keeps its replicates' table when `replicates`.
    pub(crate) fn of(replicates: bool) -> Vec<PfilterFile> {
        let mut v = vec![PfilterFile::Loglik, PfilterFile::Provenance];
        if replicates {
            v.push(PfilterFile::Replicates);
        }
        v
    }
}

/// A by-product kind that stores one file: its enum, with that file as its
/// one variant.
macro_rules! one_file {
    ($(#[$doc:meta])* $enum:ident, $kind:ty, $logical:literal, $about:literal) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub(crate) enum $enum {
            File,
        }

        impl ResultFile for $enum {
            fn logical(&self) -> String {
                $logical.into()
            }
            fn stored(&self) -> String {
                <$kind>::FILE.into()
            }
            fn format(&self) -> Format {
                Format::Tsv
            }
            fn about(&self) -> &'static str {
                $about
            }
        }

        impl Files for $kind {
            type File = $enum;
        }
    };
}

one_file!(
    /// The file of `trace`: per observation, the increment, the ESS and the
    /// one-step predictive quantiles.
    TraceFile, Trace, "trace", "Per observation, the log-likelihood increment, the ESS and the one-step predictive quantiles."
);
one_file!(
    /// The file of `pf_health`: per observation, the ESS, its fraction and the
    /// log-weight variance.
    PfHealthFile, PfHealth, "pf_health", "Per observation: `time`, `ESS`, `ESS_frac`, `tau2`."
);
one_file!(
    /// The file of `filtering`: every particle at every step.
    FilteringFile, Filtering, "filtering", "Every particle's state and log-weight at every step."
);
one_file!(
    /// The file of `filtered_streams`: each stream's conditional mean before
    /// and after each step is scored.
    FilteredStreamsFile, FilteredStreams, "filtered_streams", "Each stream's conditional mean, summarised over the swarm before and after each step is scored."
);
one_file!(
    /// The file of `final_state`: the swarm after the last observation.
    FinalStateFile, FinalState, "final_state", "The swarm after the last observation."
);
one_file!(
    /// The file of `saved_paths`: smoothed sample paths.
    SavedPathsFile, SavedPaths, "paths", "Sample paths from the smoothing distribution, traced through the ancestry."
);

/// The files of `prequential`: the run's one-step-ahead predictive scores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrequentialFile {
    Joint,
    Streams,
    Record,
}

impl ResultFile for PrequentialFile {
    fn logical(&self) -> String {
        match self {
            PrequentialFile::Joint => "joint",
            PrequentialFile::Streams => "streams",
            PrequentialFile::Record => "record",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            PrequentialFile::Joint => Prequential::JOINT,
            PrequentialFile::Streams => Prequential::STREAMS,
            PrequentialFile::Record => Prequential::JSON,
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            PrequentialFile::Record => Format::Json,
            _ => Format::Tsv,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            PrequentialFile::Joint => "At each observation time, the joint one-step-ahead log score and the ESS.",
            PrequentialFile::Streams => "At each observation time and stream, the predictive quantiles and the scores.",
            PrequentialFile::Record => "The prequential record: the scores and, unless `--no-save-samples`, each stream's predictive samples.",
        }
    }
}

impl Files for Prequential {
    type File = PrequentialFile;
}
