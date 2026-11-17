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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_engine_id_is_128_bits_of_hex_and_not_the_git_hash() {
        assert_eq!(ENGINE_ID.len(), 32, "{ENGINE_ID}");
        assert!(ENGINE_ID.bytes().all(|b| b.is_ascii_hexdigit()), "{ENGINE_ID}");
        assert_ne!(ENGINE_ID, GIT_HASH);
        assert!(VERSION_SHORT.starts_with(env!("CARGO_PKG_VERSION")));
    }
}
