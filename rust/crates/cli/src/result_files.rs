//! The files each kind of result stores, as types.
//!
//! Each kind's files are an enum whose variants are the files it can store,
//! named by [`Files::File`], and [`ResultFile`] gives a variant's logical name
//! (the stored name without its extension, unique within the kind), its
//! stored name and its role in `run.json`. A command declares the files a run
//! will hold as a list of its kind's variants: when it begins the run
//! ([`crate::camdl_store::begin`]), and again from its outcome before the run
//! finishes ([`declare`]). The store refuses to finish a run whose files
//! differ from its last declaration (`store::Error::FileSetMismatch`), and a
//! run that declared none (`store::Error::NoExpectedFiles`). A content
//! artifact stored with `Store::put` holds the files its identity names and
//! its provenance files, which `put` declares itself; a model's presentation
//! names its files by an enum, as a claimed kind does. The run spec's
//! result-file contract (§2.9) is generated from the same variants, with
//! their formats, meanings and header rows ([`contract`]). The design is
//! `docs/dev/proposals/2026-11-23-workflow-bindings.md`, "Result files as
//! types".

pub(crate) mod contract;
pub(crate) mod fit;
pub(crate) mod fit_run;
pub(crate) mod pfilter;
pub(crate) mod presentation;
pub(crate) mod profile;
pub(crate) mod sim;

/// What a stored file is to the store: a result file, determined by the
/// identity and compared on a recompute, or a provenance file, recorded with
/// its digest and not compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Result,
    Provenance,
}

/// How a stored file is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Tsv,
    Json,
    Toml,
    /// camdl model source.
    Camdl,
    /// Internal state with no format in the contract.
    Binary,
}

impl Format {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Format::Tsv => "TSV",
            Format::Json => "JSON",
            Format::Toml => "TOML",
            Format::Camdl => "camdl",
            Format::Binary => "binary",
        }
    }
}

/// One column, or a run of columns, of a table's header row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Column {
    /// A column of this name.
    Named(&'static str),
    /// Zero or more columns that depend on the model or the fit, such as one
    /// per estimated parameter; the text says which.
    Each(&'static str),
}

/// One file a kind can store.
pub(crate) trait ResultFile {
    /// The file's name in the contract: the stored name without its
    /// extension, unique within the kind.
    fn logical(&self) -> String;
    /// The file's name inside the result's folder.
    fn stored(&self) -> String;
    fn role(&self) -> Role {
        Role::Result
    }
    fn format(&self) -> Format;
    /// The header row of a TSV file, in order, where the contract states
    /// it; empty otherwise.
    fn columns(&self) -> &'static [Column] {
        &[]
    }
    /// True when the header row follows one or more `#` comment lines.
    fn commented(&self) -> bool {
        false
    }
    /// What the file holds, in one sentence.
    fn about(&self) -> &'static str;
}

/// Whether `header`, a TSV file's header row split on tabs, has the columns
/// `columns` describe: the named columns before the first [`Column::Each`]
/// open it, the named columns after the last close it, and every named column
/// between appears in order.
#[cfg(test)]
pub(crate) fn header_matches(columns: &[Column], header: &[&str]) -> bool {
    let name = |c: &Column| match c {
        Column::Named(n) => Some(*n),
        Column::Each(_) => None,
    };
    let Some(first_each) = columns.iter().position(|c| matches!(c, Column::Each(_))) else {
        return columns.iter().filter_map(name).eq(header.iter().copied());
    };
    let last_each = columns.iter().rposition(|c| matches!(c, Column::Each(_))).expect("one run exists");
    let head: Vec<&str> = columns[..first_each].iter().filter_map(name).collect();
    let tail: Vec<&str> = columns[last_each + 1..].iter().filter_map(name).collect();
    if header.len() < head.len() + tail.len() || header[..head.len()] != head[..] || header[header.len() - tail.len()..] != tail[..] {
        return false;
    }
    let mut rest = header[head.len()..header.len() - tail.len()].iter();
    columns[first_each..=last_each].iter().filter_map(name).all(|c| rest.any(|h| *h == c))
}

/// A file enum whose variant can be read back from a stored file's name.
pub(crate) trait ParseFile: ResultFile + Sized {
    /// The variant whose stored name is `stored`, or `None` for a name no
    /// result of the kind stores. Where several variants share a stored name
    /// (the same file written by several methods), any of them is returned:
    /// they share the logical name too.
    fn of_stored(stored: &str) -> Option<Self>;
}

/// The variant among `candidates` whose stored name is `stored`.
fn find<F: ResultFile>(candidates: impl IntoIterator<Item = F>, stored: &str) -> Option<F> {
    candidates.into_iter().find(|f| f.stored() == stored)
}

/// A kind whose files are an enum: every kind camdl stores through a claim,
/// and a model's presentation.
pub(crate) trait Files: store::Artifact {
    type File: ParseFile;
    /// The version of the kind's file contract: which files it stores, under
    /// which names, with which columns. It is bumped when a file is added,
    /// removed or renamed, or a column changes. The test
    /// `every_contract_version_is_pinned_to_its_files` pins each version to a
    /// digest of the kind's contract rows, so a change to the enum fails it
    /// until the version is bumped and the pin updated.
    const CONTRACT: u16;
}

/// The contract version of each content artifact (`compiled_model`,
/// `quantities`, `data_table`), whose one file is its
/// [`store::ContentArtifact::FILE`]. Pinned, with each kind's, by
/// `every_contract_version_is_pinned_to_its_files`.
const CONTENT_CONTRACT: u16 = 1;

/// A kind's file contract as a reader outside camdl relies on it: its
/// version and the logical name of each file it stores.
#[derive(Clone, Copy)]
pub(crate) struct KindContract {
    pub version: u16,
    logical: fn(&str) -> Option<String>,
    format: fn(&str) -> Option<Format>,
}

impl KindContract {
    /// The logical name of the stored file `stored`, or `None` for a name the
    /// kind's contract does not hold.
    pub(crate) fn logical(&self, stored: &str) -> Option<String> {
        (self.logical)(stored)
    }

    /// The format of the stored file `stored`, or `None` for a name the
    /// kind's contract does not hold.
    pub(crate) fn format(&self, stored: &str) -> Option<Format> {
        (self.format)(stored)
    }
}

fn logical_of<F: ParseFile>(stored: &str) -> Option<String> {
    F::of_stored(stored).map(|f| f.logical())
}

fn format_of<F: ParseFile>(stored: &str) -> Option<Format> {
    F::of_stored(stored).map(|f| f.format())
}

/// A content artifact's one file is named in the contract by its stored name
/// up to the first `.`: `model` for `model.ir.json`.
fn content_logical<C: store::ContentArtifact>(stored: &str) -> Option<String> {
    (stored == C::FILE).then(|| C::FILE.split_once('.').map_or(C::FILE, |(stem, _)| stem).to_string())
}

/// A content artifact's one file is a table when its name ends `.tsv`, and
/// JSON otherwise: `table.tsv`, `model.ir.json`, `quantities.json`.
fn content_format<C: store::ContentArtifact>(stored: &str) -> Option<Format> {
    (stored == C::FILE).then(|| if C::FILE.ends_with(".tsv") { Format::Tsv } else { Format::Json })
}

fn claimed<A: Files>() -> (&'static str, KindContract) {
    (A::KIND, KindContract { version: A::CONTRACT, logical: logical_of::<A::File>, format: format_of::<A::File> })
}

fn content<C: store::ContentArtifact>() -> (&'static str, KindContract) {
    (C::KIND, KindContract { version: CONTENT_CONTRACT, logical: content_logical::<C>, format: content_format::<C> })
}

/// The file contract of every kind camdl stores, by kind name.
fn contracts() -> [(&'static str, KindContract); 21] {
    use crate::{camdl_store as cs, fit_store as fs, pfilter_store as ps, profile_store as pr, sim_store as ss};
    [
        content::<cs::CompiledModel>(),
        content::<cs::Quantities>(),
        content::<cs::DataTable>(),
        claimed::<cs::ModelPresentation>(),
        claimed::<fs::FitProblem>(),
        claimed::<fs::FitRun>(),
        claimed::<fs::Prediction>(),
        claimed::<fs::Score>(),
        claimed::<ps::Pfilter>(),
        claimed::<ps::Trace>(),
        claimed::<ps::PfHealth>(),
        claimed::<ps::Filtering>(),
        claimed::<ps::FilteredStreams>(),
        claimed::<ps::FinalState>(),
        claimed::<ps::SavedPaths>(),
        claimed::<ps::Prequential>(),
        claimed::<ss::Sim>(),
        claimed::<ss::Obs>(),
        claimed::<ss::EventLog>(),
        claimed::<ss::SimQuantities>(),
        claimed::<pr::ProfilePoint>(),
    ]
}

/// The file contract of the kind named `kind`, or `None` for a kind camdl
/// does not store.
pub(crate) fn contract(kind: &str) -> Option<KindContract> {
    contracts().into_iter().find(|(k, _)| *k == kind).map(|(_, c)| c)
}

/// Whether `kind` is a content artifact (`compiled_model`, `quantities`,
/// `data_table`, `model_presentation`): a result named by its content, which
/// a command stores beside the results it computes.
pub(crate) fn is_content(kind: &str) -> bool {
    use crate::camdl_store as cs;
    use store::Artifact;
    [cs::CompiledModel::KIND, cs::Quantities::KIND, cs::DataTable::KIND, cs::ModelPresentation::KIND].contains(&kind)
}

/// The expected set of a run that will hold exactly `files`.
pub(crate) fn expected<F: ResultFile>(files: impl IntoIterator<Item = F>) -> store::ExpectedFiles {
    let mut set = store::ExpectedFiles::default();
    for f in files {
        match f.role() {
            Role::Result => set.result.insert(f.stored()),
            Role::Provenance => set.provenance.insert(f.stored()),
        };
    }
    set
}

/// Declares that `run` will hold exactly `files`, replacing its earlier
/// declaration: for a run whose files depend on its outcome.
pub(crate) fn declare<A: Files>(run: &mut store::Run<A>, files: impl IntoIterator<Item = A::File>) {
    run.expect(expected(files));
}

/// Declares that the by-product `run` will hold exactly `files`.
pub(crate) fn declare_by_product<B: Files + store::ByProduct>(run: &mut store::ByProductRun<B>, files: impl IntoIterator<Item = B::File>) {
    run.expect(expected(files));
}

/// Opens `file` in `run`, in its role.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn create<A: Files>(run: &store::Run<A>, file: &A::File) -> store::Result<store::FileHandle> {
    match file.role() {
        Role::Result => run.result_file(&file.stored()),
        Role::Provenance => run.provenance_file(&file.stored()),
    }
}

/// Opens `file` in the by-product `run`. A by-product's files are all result
/// files.
#[allow(clippy::result_large_err)] // the store's error carries the paths and owners a person acts on, returned once
pub(crate) fn create_by_product<B: Files + store::ByProduct>(run: &store::ByProductRun<B>, file: &B::File) -> store::Result<store::FileHandle> {
    run.result_file(&file.stored())
}

#[cfg(test)]
mod tests;
