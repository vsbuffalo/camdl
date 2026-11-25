//! What a command that stores results writes for a workflow beside the
//! results themselves: its output manifest at `--manifest PATH`
//! (`docs/camdl-run-spec.md` §2.10, [`crate::output_manifest`]) and its link
//! tree under its link root (§2.11, [`crate::link_tree`]).
//!
//! A command claims both ([`claim`]) after its arguments have parsed and
//! before its first call to the store, which removes an earlier manifest at
//! the path and the label's earlier links, and writes both ([`write`]) once
//! it has succeeded, from one reading of its invocation record. The links are
//! written first and the manifest last, so a command whose manifest exists
//! has written its links; a manifest that cannot be written removes the
//! links just written, and the command fails with neither.

use std::path::{Path, PathBuf};

use crate::link_tree::LinkTree;
use crate::output_manifest::{ManifestPath, OutputManifest};

/// A command's claimed manifest path and link tree, each when it has one.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bindings {
    manifest: Option<ManifestPath>,
    links: Option<LinkTree>,
}

/// Claims `manifest` (`--manifest`) and the links of `label` under
/// `link_root`, a root [`crate::link_tree::link_root`] resolved, or fails
/// the command with the refusal. A command that keeps nothing in the store
/// (`--dry-run`, `--no-store`) passes no root, so a `CAMDL_LINK_ROOT` in its
/// environment removes no links.
///
/// `label` is parsed by [`store::Label::parse`] first, the rule the command's
/// store request applies too, so a label it refuses fails the command before
/// the claim moves a previous holder's links aside.
pub(crate) fn claim(out: &crate::out::Out, manifest: Option<&Path>, link_root: Option<PathBuf>, label: Option<&str>) -> Bindings {
    let label = label.map(|l| store::Label::parse(l).unwrap_or_else(|e| out.fail(format!("error: {e}"))));
    let manifest = crate::output_manifest::claim(out, manifest);
    let links = link_root.map(|root| LinkTree::claim(&root, label.as_ref()).unwrap_or_else(|e| out.fail(format!("error: {e}"))));
    Bindings { manifest, links }
}

/// Writes the links and the manifest of `invocation`, an ended invocation of
/// `store`, each when it was claimed, or fails the command.
pub(crate) fn write(out: &crate::out::Out, bindings: &Bindings, store: &store::Store, invocation: &store::Invocation) {
    let (manifest_path, links) = (bindings.manifest.as_ref(), bindings.links.as_ref());
    let flag = match (manifest_path, links) {
        (None, None) => return,
        (Some(m), _) => format!("--manifest {}", m.path().display()),
        (None, Some(_)) => "--link-root".to_string(),
    };
    let manifest = OutputManifest::of(store, invocation).unwrap_or_else(|e| out.fail(format!("error: {flag}: {e}")));
    let written = match links {
        Some(l) => l.write(store, &manifest).unwrap_or_else(|e| out.fail(format!("error: {e}"))),
        None => Vec::new(),
    };
    for dir in &written {
        out.say(format_args!("linked {}", dir.display()));
    }
    if let Some(m) = manifest_path {
        if let Err(e) = m.write(&manifest) {
            LinkTree::remove(&written);
            out.fail(format!("error: {e}"));
        }
    }
}
