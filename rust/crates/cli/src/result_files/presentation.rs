//! The files of a model's presentation (`model_presentation`).

use super::{find, Files, Format, ParseFile, ResultFile};
use crate::camdl_store::presentation as p;
use crate::camdl_store::ModelPresentation;

/// The files of a model's presentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PresentationFile {
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

impl ResultFile for PresentationFile {
    fn logical(&self) -> String {
        match self {
            PresentationFile::Source { .. } => "source",
            PresentationFile::Docs => "docs",
            PresentationFile::Declared => "declared",
            PresentationFile::Render => "render",
            PresentationFile::Graph => "graph",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            PresentationFile::Source { camdl: true } => p::SOURCE_CAMDL,
            PresentationFile::Source { camdl: false } => p::SOURCE_IR,
            PresentationFile::Docs => p::DOCS,
            PresentationFile::Declared => p::DECLARED,
            PresentationFile::Render => p::RENDER,
            PresentationFile::Graph => p::GRAPH,
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            PresentationFile::Source { camdl: true } => Format::Camdl,
            _ => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            PresentationFile::Source { .. } => "The model file as written.",
            PresentationFile::Docs => "The model's `description` and its `#'` documentation, by declaration.",
            PresentationFile::Declared => "The model as its author declared it.",
            PresentationFile::Render => "The model's mathematics as KaTeX strings, as `camdlc render --format json` writes them.",
            PresentationFile::Graph => "The compartment flow graph, as `camdlc render --format graph` writes it.",
        }
    }
}

impl Files for ModelPresentation {
    type File = PresentationFile;
    const CONTRACT: u16 = 1;
}

impl ParseFile for PresentationFile {
    fn of_stored(stored: &str) -> Option<Self> {
        find(
            [
                PresentationFile::Source { camdl: true },
                PresentationFile::Source { camdl: false },
                PresentationFile::Docs,
                PresentationFile::Declared,
                PresentationFile::Render,
                PresentationFile::Graph,
            ],
            stored,
        )
    }
}
