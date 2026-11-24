//! A store for the results of scientific computations: identity, provenance
//! and layout derived from the types of a program's inputs.
//!
//! A project declares every input once, with a class, in its argument
//! structs, its settings files and its environment ([`Inputs`],
//! [`FileInputs`], [`EnvInputs`]). Typed resolution steps turn the declared
//! inputs into one [`Identity`] per result. The store derives each result's run
//! id from the kind, the ambient identity and the identity key, places its
//! folder by the kind's [`Layout`], and does every write and every read of a
//! stored file. A result is reused only when it is completed; a completed
//! result's files and records are never written again.
//!
//! The design is `docs/dev/proposals/2026-11-20-run-store-api.md`, Part 1.
//! The library holds no type of any particular project.

extern crate self as store;

mod artifact;
mod check;
mod commands;
mod error;
mod fsutil;
mod handle;
mod harness;
mod hash;
mod identity;
mod index;
mod inputs;
mod invocation;
mod layout;
mod maintenance;
mod owner;
mod receive;
mod record;
mod run;
mod storage;
mod stored;
#[cfg(test)]
mod tests;

pub use artifact::{Artifact, Build, ByProduct, ContentArtifact, Id, Kinds, Primary, TopLevel, WriteMode};
pub use commands::{StoreCommand, Verdict};
pub use error::{Error, Result, RunError};
pub use handle::{Handle, RunRef};
pub use harness::{check_input_classes, ClassFailure, InputHarness, Observed, ObservedResult, StoredFile};
pub use hash::{CanonicalHasher, ContentAddressed, Key, KeyParseError, HASH_VERSION};
pub use identity::{Content, DepInfo, Finite, Hashed, Identity, NotFinite};
pub use inputs::{
    release, Class, EnvInputs, EnvParts, FieldInfo, FileInputs, Given, InputFile, Inputs, Parsed, SourceBytes, Split, Vary,
};
pub use invocation::{
    ByProductRendering, ExportFiles, ExportPath, ExportSink, Invocation, Member, MemberWith, Output, OutputFile, Rendering, RunInfo,
    StoreRequest,
};
pub use layout::{path_label, Layout, Placed, LABEL_CAP};
pub use maintenance::{Cleaned, VerifyFailure};
pub use owner::{this_host, Liveness, Owner, Timestamp, BEAT_INTERVAL, STALE_AFTER};
pub use receive::Received;
pub use record::{Lineage, Status, MAX_PRESERVED_ATTEMPTS, RECORD_FORMAT, STORE_FORMAT};
pub use run::{ByProductRun, ExpectedFiles, FileHandle, Run};
pub use fsutil::write_file_atomic;
pub use storage::{Batch, Begin, PlanState, Planned, Reuse, Store, Wanted};
pub use stored::{Dep, Entry, EntryState, Located, Stored};
pub use store_derive::{EnvInputs, FileInputs, Identity, Inputs};

/// Items the derives expand to. Not for use by hand.
#[doc(hidden)]
pub mod __private {
    pub use crate::inputs::{__prefixed, env};
    pub use serde;
    pub use serde_json;
}

/// What a content artifact's first writer records beside its one result
/// file: provenance files (in the manifest, never compared) and named values
/// in `run.json`.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    pub(crate) files: std::collections::BTreeMap<String, Vec<u8>>,
    pub(crate) values: std::collections::BTreeMap<String, serde_json::Value>,
}

impl Provenance {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn file(mut self, name: &str, bytes: Vec<u8>) -> Self {
        self.files.insert(name.to_string(), bytes);
        self
    }

    pub fn value(mut self, name: &str, value: &impl serde::Serialize) -> Result<Self> {
        let v = serde_json::to_value(value)
            .map_err(|e| Error::Parse { what: format!("the provenance value `{name}`"), why: e.to_string() })?;
        self.values.insert(name.to_string(), v);
        Ok(self)
    }
}
