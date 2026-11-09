//! The fit report: what a method's run shows. Never writes; returns lines.
//!
//! `facts` holds the records the report is built from, `build` the one
//! builder, `findings` the diagnostics block, `render` the other sections,
//! `export` the Markdown and LaTeX projections of `fit summary`.

pub mod build;
pub mod exact;
pub mod export;
pub mod facts;
pub mod findings;
pub mod render;

pub use facts::*;
