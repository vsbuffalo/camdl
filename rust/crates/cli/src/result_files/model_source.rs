//! The files of a model source (`model_source`).

use super::{find, Files, Format, ParseFile, ResultFile};
use crate::camdl_store::model_source as p;
use crate::camdl_store::ModelSource;

/// The files of a model source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SourceFile {
    /// `source.camdl` or `source.ir.json`, the model file as written.
    Source { camdl: bool },
    /// `docs.json`, the description and the `#'` documentation.
    Docs,
    /// `declared.json`, the model as its author declared it.
    Declared,
    /// `model.render.json`, the model's mathematics as KaTeX strings.
    Render,
    /// `model.graph.json`, the compartment flow graph.
    Graph,
}

impl ResultFile for SourceFile {
    fn logical(&self) -> String {
        match self {
            SourceFile::Source { .. } => "source",
            SourceFile::Docs => "docs",
            SourceFile::Declared => "declared",
            SourceFile::Render => "render",
            SourceFile::Graph => "graph",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            SourceFile::Source { camdl: true } => p::SOURCE_CAMDL,
            SourceFile::Source { camdl: false } => p::SOURCE_IR,
            SourceFile::Docs => p::DOCS,
            SourceFile::Declared => p::DECLARED,
            SourceFile::Render => p::RENDER,
            SourceFile::Graph => p::GRAPH,
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            SourceFile::Source { camdl: true } => Format::Camdl,
            _ => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            SourceFile::Source { .. } => "The model file as written.",
            SourceFile::Docs => "The model's `description` and its `#'` documentation, by declaration.",
            SourceFile::Declared => "The model as its author declared it.",
            SourceFile::Render => "The model's mathematics as KaTeX strings, as `camdlc render --format json` writes them.",
            SourceFile::Graph => "The compartment flow graph, as `camdlc render --format graph` writes it.",
        }
    }
}

impl Files for ModelSource {
    type File = SourceFile;
    const CONTRACT: u16 = 1;
}

impl ParseFile for SourceFile {
    fn of_stored(stored: &str) -> Option<Self> {
        find(
            [
                SourceFile::Source { camdl: true },
                SourceFile::Source { camdl: false },
                SourceFile::Docs,
                SourceFile::Declared,
                SourceFile::Render,
                SourceFile::Graph,
            ],
            stored,
        )
    }
}
