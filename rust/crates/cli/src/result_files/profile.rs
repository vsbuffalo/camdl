//! The files of a profile point.

use super::{Files, Format, ResultFile, Role};
use crate::profile_store::{ProfilePoint, MLE_FILE, PROVENANCE_FILE};

/// The files of a profile point (`profile_point`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProfilePointFile {
    /// `mle.toml`: the point's final log-likelihood, the profiled value and
    /// the other estimates.
    Mle,
    /// `provenance.json`, the point's settings as resolved (provenance).
    Provenance,
}

impl ProfilePointFile {
    pub(crate) const ALL: [ProfilePointFile; 2] = [ProfilePointFile::Mle, ProfilePointFile::Provenance];
}

impl ResultFile for ProfilePointFile {
    fn logical(&self) -> String {
        match self {
            ProfilePointFile::Mle => "mle",
            ProfilePointFile::Provenance => "provenance",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            ProfilePointFile::Mle => MLE_FILE,
            ProfilePointFile::Provenance => PROVENANCE_FILE,
        }
        .into()
    }
    fn role(&self) -> Role {
        match self {
            ProfilePointFile::Mle => Role::Result,
            ProfilePointFile::Provenance => Role::Provenance,
        }
    }
    fn format(&self) -> Format {
        match self {
            ProfilePointFile::Mle => Format::Toml,
            ProfilePointFile::Provenance => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            ProfilePointFile::Mle => "`final_loglik`, a `[focal]` table with the profiled parameter's value, and an `[mle]` table with the other estimates.",
            ProfilePointFile::Provenance => "The point's settings and starts as resolved.",
        }
    }
}

impl Files for ProfilePoint {
    type File = ProfilePointFile;
}
