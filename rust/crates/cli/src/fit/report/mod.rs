//! The fit report: what a method's run shows. Never writes; returns lines.
//!
//! `facts` holds the records the report is built from, `findings` the
//! diagnostics block.

pub mod facts;
pub mod findings;

pub use facts::*;
