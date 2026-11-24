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
//! artifact stored with `Store::put` holds the one file its kind names and
//! its provenance files, which `put` declares itself. The run spec's
//! result-file contract (§2.9) is generated from the same variants, with
//! their formats, meanings and header rows ([`contract`]). The design is
//! `docs/dev/proposals/2026-11-23-workflow-bindings.md`, "Result files as
//! types".

pub(crate) mod contract;
pub(crate) mod fit;
pub(crate) mod fit_run;
pub(crate) mod pfilter;
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

/// A kind camdl stores through a claim, with the enum of its files.
pub(crate) trait Files: store::Artifact {
    type File: ResultFile;
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
