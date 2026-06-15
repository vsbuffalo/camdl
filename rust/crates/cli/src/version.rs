/// Full version string: "0.1.0+ce78a5e (2026-04-03)". The `camdl ` prefix is
/// supplied by clap (the bin name); do not repeat it here or `--version` prints
/// "camdl camdl 0.1.0…". `CARGO_PKG_VERSION` is the release version (bumped by
/// `scripts/release.sh`); the `+<hash>` is the precise commit, `(date)` the build.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+", env!("CAMDL_GIT_HASH"),
    " (", env!("CAMDL_BUILD_DATE"), ")"
);

/// Short version for embedding in output files: "0.1.0+ce78a5e"
pub const VERSION_SHORT: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "+", env!("CAMDL_GIT_HASH"),
);

/// Just the git hash for comparison.
pub const GIT_HASH: &str = env!("CAMDL_GIT_HASH");
