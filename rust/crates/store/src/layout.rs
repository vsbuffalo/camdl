//! Where a result's folder sits under the store root.
//!
//! A layout is a view. It groups parts of an identity into directory levels,
//! each `<label>-<12 hex of the part's key>/`, and ends in the result folder,
//! `<label>-<12 hex of the run id>/`. Nothing reads meaning out of a path: a
//! folder is found by its suffixes and confirmed by the run id its
//! `identity.json` holds.

use crate::artifact::Artifact;
use crate::hash::{ContentAddressed, Key};
use crate::Id;

/// The cap on a sanitised label, in bytes.
pub const LABEL_CAP: usize = 64;

/// What a too-long label is cut to before `..` and its hash are appended.
const TRUNC_PREFIX: usize = 46;
const TRUNC_MARKER: &str = "..";
const TRUNC_HASH_HEX: usize = 16;

/// A label made safe for one path component: lowercased, every character
/// outside `a-z`, `0-9`, `_`, `-` and `.` replaced by `_`, and a leading `.`
/// replaced by `_`, so no folder hides among the store's own entries. A
/// sanitised label longer than [`LABEL_CAP`] bytes is cut to a 46-byte prefix
/// and ended with `..` and the first 16 hex digits of the SHA-256 of the whole
/// sanitised label, so it is 64 bytes.
pub fn path_label(label: &str) -> String {
    let mut s: String = label
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') { c } else { '_' })
        .collect();
    if s.starts_with('.') {
        s.replace_range(0..1, "_");
    }
    if s.len() <= LABEL_CAP {
        return s;
    }
    let tag = &Key::digest_bytes(s.as_bytes()).to_hex()[..TRUNC_HASH_HEX];
    format!("{}{TRUNC_MARKER}{tag}", &s[..TRUNC_PREFIX])
}

/// A directory name: a sanitised label, `-`, and a suffix.
pub(crate) fn dir_name(label: &str, suffix: &str) -> String {
    format!("{}-{suffix}", path_label(label))
}

/// Where a layout starts.
#[derive(Debug, Clone)]
pub(crate) enum Base {
    /// The kind's directory, inside the build directory under
    /// [`crate::TopLevel::ByBuild`].
    InKind,
    /// A subdirectory of a parent result's folder.
    Inside { kind: &'static str, parent: Key, subdir: &'static str },
}

/// Where a result's folder will sit: a base and the groups below it, in order.
#[derive(Debug, Clone)]
pub struct Layout {
    pub(crate) base: Base,
    pub(crate) groups: Vec<(String, Key)>,
}

/// A whole placement: the layout and the result folder's own label. Only
/// [`Layout::result`] makes one.
#[derive(Debug, Clone)]
pub struct Placed {
    pub(crate) layout: Layout,
    pub(crate) label: String,
}

impl Layout {
    /// Under the kind's directory: `<root>/<kind>/`, or with
    /// [`crate::TopLevel::ByBuild`] `<root>/<build label>-<12 hex of the
    /// ambient key>/<kind>/`.
    pub fn in_kind() -> Self {
        Layout { base: Base::InKind, groups: Vec::new() }
    }

    /// Under a subdirectory of a parent result, `<parent's folder>/<subdir>/`,
    /// in whichever build's directory the parent sits.
    pub fn inside<P: Artifact>(parent: &Id<P>, subdir: &'static str) -> Self {
        Layout { base: Base::Inside { kind: P::KIND, parent: parent.key(), subdir }, groups: Vec::new() }
    }

    /// One directory level, `<label>-<12 hex of the part's key>/`. Results
    /// whose identities share the part share the directory.
    pub fn group<T: ContentAddressed + ?Sized>(mut self, label: &str, part: &T) -> Self {
        self.groups.push((label.to_string(), part.content_hash()));
        self
    }

    /// The result folder itself, `<label>-<12 hex of the run id>/`.
    pub fn result(self, label: &str) -> Placed {
        Placed { layout: self, label: label.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_label_is_sanitised_and_a_leading_dot_is_replaced() {
        assert_eq!(path_label("Slow Scenario/2"), "slow_scenario_2");
        assert_eq!(path_label(".hidden"), "_hidden");
        assert_eq!(path_label("chain_binomial-dt1.5"), "chain_binomial-dt1.5");
    }

    #[test]
    fn a_long_label_is_cut_to_64_bytes_and_stays_distinct() {
        let a = path_label(&"a".repeat(100));
        let b = path_label(&format!("{}b", "a".repeat(99)));
        assert_eq!(a.len(), LABEL_CAP);
        assert_eq!(b.len(), LABEL_CAP);
        assert_ne!(a, b);
        assert!(a.starts_with(&"a".repeat(TRUNC_PREFIX)));
        assert_eq!(&a[TRUNC_PREFIX..TRUNC_PREFIX + 2], "..");
    }

    #[test]
    fn a_label_of_exactly_the_cap_is_kept() {
        let s = "x".repeat(LABEL_CAP);
        assert_eq!(path_label(&s), s);
    }
}
