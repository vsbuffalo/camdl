/// Full version string: "0.1.0+ce78a5e (2026-04-03)". The `camdl ` prefix is
/// supplied by clap (the bin name); do not repeat it here or `--version` prints
/// "camdl camdl 0.1.0…". `CARGO_PKG_VERSION` is the release version (bumped by
/// `scripts/release.sh`); the `+<source>` is the readable source fingerprint
/// ([`VERSION_SHORT`]), `(date)` the build.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+", env!("CAMDL_SOURCE_LABEL"),
    " (", env!("CAMDL_BUILD_DATE"), ")"
);

/// The readable version for display and provenance: `0.1.0+ce78a5e` for a
/// clean build, `0.1.0+ce78a5e.dirty-3f9c21e0` for one with uncommitted
/// changes to its build inputs, `0.1.0+nogit-1a2b3c4d` without git. Not run
/// identity: that is [`ENGINE_ID`].
pub const VERSION_SHORT: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+", env!("CAMDL_SOURCE_LABEL"),
);

/// Just the git hash, for the camdlc version handshake. Unchanged by
/// uncommitted edits.
pub const GIT_HASH: &str = env!("CAMDL_GIT_HASH");

/// The engine's build fingerprint (32 hex characters): the source (commit and
/// uncommitted changes), `Cargo.lock`, the binary's resolved features, the
/// profile, `rustc -vV`, `RUSTFLAGS` and the target
/// (`engine_fingerprint.rs`). It is what a run's identity records as the
/// engine, and what `run.json`'s `engine_version` holds.
pub const ENGINE_ID: &str = env!("CAMDL_ENGINE_ID");

/// Whether this binary may serve a stored result in place of computing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineReuse {
    /// Every component of [`ENGINE_ID`] was determined at build time.
    Reusable,
    /// Some component could not be determined (`missing`, comma-separated):
    /// the binary computes and records every run and never reads a stored
    /// one, since its identity may stand for another build's.
    Unreusable { missing: &'static str },
}

impl EngineReuse {
    pub fn allows_reuse(self) -> bool {
        matches!(self, EngineReuse::Reusable)
    }
}

const ENGINE_UNREUSABLE: &str = env!("CAMDL_ENGINE_UNREUSABLE");

/// This binary's [`EngineReuse`], from the build script.
pub const ENGINE_REUSE: EngineReuse = if ENGINE_UNREUSABLE.is_empty() {
    EngineReuse::Reusable
} else {
    EngineReuse::Unreusable { missing: ENGINE_UNREUSABLE }
};

/// How a build recorded as `recorded` (a [`VERSION_SHORT`] string) relates to
/// the running one, for reports that say whether a stored result is current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildRelation {
    /// The same readable version.
    Same,
    /// The same release and commit; only the uncommitted changes differ (one
    /// side is clean, or the two dirty digests differ).
    UncommittedChanges,
    /// Another release or commit.
    Stale,
}

/// Compare two readable versions (`<release>+<commit>[.dirty-<digest>]`).
pub fn build_relation(recorded: &str, current: &str) -> BuildRelation {
    if recorded == current {
        return BuildRelation::Same;
    }
    let base = |v: &str| v.split_once(".dirty-").map_or(v, |(b, _)| b).to_string();
    if base(recorded) == base(current) {
        BuildRelation::UncommittedChanges
    } else {
        BuildRelation::Stale
    }
}

/// The sentence a report gives about a result recorded as produced by the
/// readable version `recorded`, or `None` when that is this build:
/// `stale: produced by X, current is Y` for another release or commit, `built
/// from uncommitted changes: produced by X, current is Y` when only the
/// uncommitted changes differ. `quote` wraps each version (backticks in
/// Markdown).
///
/// `recorded_engine` is the run's `ENGINE_ID` (its `run.json`'s
/// `engine_version`), when it is known: under the same readable version and
/// another id, the run was built with [`ANOTHER_CONFIGURATION`].
pub fn build_note(recorded: &str, recorded_engine: Option<&str>, quote: impl Fn(&str) -> String) -> Option<String> {
    let what = match build_relation(recorded, VERSION_SHORT) {
        BuildRelation::Same if recorded_engine.is_some_and(|e| e != ENGINE_ID) => ANOTHER_CONFIGURATION,
        BuildRelation::Same => return None,
        BuildRelation::UncommittedChanges => "built from uncommitted changes",
        BuildRelation::Stale => "stale",
    };
    Some(format!("{what}: produced by {}, current is {}", quote(recorded), quote(VERSION_SHORT)))
}

/// What differs between two builds whose readable versions agree and whose
/// `ENGINE_ID`s do not: the build configuration, which the label leaves out
/// (an uncommitted `Cargo.lock` change moves the id and not the label).
pub const ANOTHER_CONFIGURATION: &str =
    "built with another build configuration (features, profile, toolchain, flags or dependencies)";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dirty_digest_alone_is_uncommitted_changes_and_a_commit_is_stale() {
        let clean = "0.4.0+abc1234";
        let dirty = "0.4.0+abc1234.dirty-3f9c21e0";
        assert_eq!(build_relation(clean, clean), BuildRelation::Same);
        assert_eq!(build_relation(dirty, clean), BuildRelation::UncommittedChanges);
        assert_eq!(build_relation(clean, dirty), BuildRelation::UncommittedChanges);
        assert_eq!(build_relation(dirty, "0.4.0+abc1234.dirty-00000000"), BuildRelation::UncommittedChanges);
        assert_eq!(build_relation("0.4.0+def5678", clean), BuildRelation::Stale);
        assert_eq!(build_relation("0.4.0+def5678.dirty-3f9c21e0", dirty), BuildRelation::Stale);
        assert_eq!(build_relation("0.3.0+abc1234", clean), BuildRelation::Stale);
    }

    #[test]
    fn the_embedded_engine_id_is_128_bits_of_hex_and_not_the_git_hash() {
        assert_eq!(ENGINE_ID.len(), 32, "{ENGINE_ID}");
        assert!(ENGINE_ID.bytes().all(|b| b.is_ascii_hexdigit()), "{ENGINE_ID}");
        assert_ne!(ENGINE_ID, GIT_HASH);
        assert!(VERSION_SHORT.starts_with(env!("CARGO_PKG_VERSION")));
    }

    /// The build under test determined every component of its fingerprint:
    /// a broken wiring (sim's `links` metadata not reaching cli's build
    /// script, `rustc -vV` failing) would make every binary fail closed and
    /// silently never reuse a stored run.
    #[test]
    fn this_build_has_a_reusable_engine_identity() {
        assert!(ENGINE_REUSE.allows_reuse(), "{ENGINE_REUSE:?}");
    }
}
