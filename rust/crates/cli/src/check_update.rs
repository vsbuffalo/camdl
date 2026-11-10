//! `camdl check-update` — tell the user whether a newer camdl release exists.
//!
//! Explicit and synchronous: the user ran it deliberately, so a network
//! round-trip with a tight timeout is fine and a visible failure is correct.
//! No cache, no detached child, no per-command hook, no on-by-default call.
//! See `docs/dev/proposals/2026-06-25-update-availability-check.md`.
//!
//! Design seam: [`decide`] / [`newest_release`] are pure (no I/O) and are the
//! exhaustively unit-tested core; [`fetch_release_tags`] is the only function
//! that touches the network and is never exercised by the test suite.

use std::time::Duration;

/// GitHub REST endpoint for the releases list. Uses `/releases` (NOT
/// `/releases/latest`, which 404s while every tag is a pre-release) so an
/// alpha-tagged project still nudges. Includes pre-releases by design.
const RELEASES_API_URL: &str = "https://api.github.com/repos/vsbuffalo/camdl/releases";

/// Human-facing releases page printed in the "an update is available" line.
const RELEASES_PAGE_URL: &str = "https://github.com/vsbuffalo/camdl/releases";

/// Outcome of the version comparison — derived purely from the current binary
/// version and the available release tags.
#[derive(Debug, PartialEq)]
pub enum UpdateStatus {
    /// A newer release exists.
    Available {
        current: semver::Version,
        latest: semver::Version,
    },
    /// The current binary is at or ahead of the newest release.
    UpToDate { current: semver::Version },
    /// The repository has published no (parseable) releases yet.
    NoReleases,
}

/// Newest release tag that parses as semver (stripping a leading `v`); `None`
/// if the list is empty or nothing parses. Pure — no I/O. Tags that don't parse
/// are skipped rather than failing the whole check.
fn newest_release(tags: &[String]) -> Option<semver::Version> {
    tags.iter()
        .filter_map(|t| semver::Version::parse(t.strip_prefix('v').unwrap_or(t)).ok())
        .max()
}

/// Pure decision. `current` is this binary's `CARGO_PKG_VERSION`, pre-parsed.
/// Available iff the newest release strictly exceeds `current`; equal-or-lower
/// (including a dev build ahead of the newest release) reads as up to date.
/// Pre-release ordering is the semver crate's (SemVer §11).
fn decide(current: &semver::Version, tags: &[String]) -> UpdateStatus {
    match newest_release(tags) {
        None => UpdateStatus::NoReleases,
        Some(latest) if latest > *current => UpdateStatus::Available {
            current: current.clone(),
            latest,
        },
        Some(_) => UpdateStatus::UpToDate {
            current: current.clone(),
        },
    }
}

/// Thin network layer: GET the releases list and pull out the `tag_name`s.
/// `ureq` + rustls (no system OpenSSL), 3 s connect/read timeouts. GitHub
/// rejects requests without a `User-Agent`. NOT called by any test.
fn fetch_release_tags() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    #[derive(serde::Deserialize)]
    struct GhRelease {
        tag_name: String,
    }

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout_read(Duration::from_secs(3))
        .build();

    let body = agent
        .get(RELEASES_API_URL)
        .set("User-Agent", "camdl-check-update")
        .set("Accept", "application/vnd.github+json")
        .set("X-GitHub-Api-Version", "2022-11-28")
        .call()?
        .into_string()?;

    let releases: Vec<GhRelease> = serde_json::from_str(&body)?;
    Ok(releases.into_iter().map(|r| r.tag_name).collect())
}

/// `camdl check-update` entry point. Prints one informational line and exits 0
/// in every reachable case (an unreachable network is not an error — the user
/// asked and we simply couldn't answer).
pub fn cmd_check_update() {
    let out = crate::out::Out::new(false);
    let current = match semver::Version::parse(env!("CARGO_PKG_VERSION")) {
        Ok(v) => v,
        Err(e) => {
            // camdl's own version string is malformed — a build-time bug, not a
            // user error; surface it loudly rather than printing a wrong answer.
            out.fail(format!(
                "error: camdl's own version ({}) is not valid semver: {e}",
                env!("CARGO_PKG_VERSION")
            ));
        }
    };

    let tags = match fetch_release_tags() {
        Ok(tags) => tags,
        Err(_) => {
            out.data_line("couldn't reach GitHub to check for updates (offline?).");
            return;
        }
    };

    match decide(&current, &tags) {
        UpdateStatus::Available { current, latest } => {
            out.data_line(format_args!("camdl {latest} is available (you have {current}). {RELEASES_PAGE_URL}"));
        }
        UpdateStatus::UpToDate { current } => {
            out.data_line(format_args!("camdl {current} is up to date."));
        }
        UpdateStatus::NoReleases => {
            out.data_line("no camdl releases published yet.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semver::Version;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    #[test]
    fn empty_tags_is_no_releases() {
        assert_eq!(decide(&v("0.2.0"), &[]), UpdateStatus::NoReleases);
    }

    #[test]
    fn only_unparseable_tags_is_no_releases() {
        let tags = vec!["latest".to_string(), "nightly".to_string()];
        assert_eq!(decide(&v("0.2.0"), &tags), UpdateStatus::NoReleases);
    }

    #[test]
    fn newer_release_is_available() {
        let tags = vec!["v0.3.0".to_string(), "v0.1.0".to_string()];
        assert_eq!(
            decide(&v("0.2.0"), &tags),
            UpdateStatus::Available {
                current: v("0.2.0"),
                latest: v("0.3.0"),
            }
        );
    }

    #[test]
    fn same_version_is_up_to_date() {
        let tags = vec!["v0.2.0".to_string()];
        assert_eq!(
            decide(&v("0.2.0"), &tags),
            UpdateStatus::UpToDate { current: v("0.2.0") }
        );
    }

    #[test]
    fn leading_v_is_stripped() {
        let tags = vec!["0.3.0".to_string()];
        assert_eq!(newest_release(&tags), Some(v("0.3.0")));
    }

    #[test]
    fn prerelease_orders_below_release() {
        // newest parseable is 0.2.0-rc.1 < 0.2.0 -> up to date
        let tags = vec!["v0.2.0-rc.1".to_string()];
        assert_eq!(
            decide(&v("0.2.0"), &tags),
            UpdateStatus::UpToDate { current: v("0.2.0") }
        );
    }

    #[test]
    fn release_above_own_prerelease_is_available() {
        // current is the rc; the final release supersedes it
        let tags = vec!["v0.2.0".to_string()];
        assert_eq!(
            decide(&v("0.2.0-rc.1"), &tags),
            UpdateStatus::Available {
                current: v("0.2.0-rc.1"),
                latest: v("0.2.0"),
            }
        );
    }

    #[test]
    fn prereleases_are_nudged() {
        // a higher minor pre-release still beats the current release
        let tags = vec!["v0.3.0-alpha".to_string(), "v0.2.0".to_string()];
        assert_eq!(
            decide(&v("0.2.0"), &tags),
            UpdateStatus::Available {
                current: v("0.2.0"),
                latest: v("0.3.0-alpha"),
            }
        );
    }

    #[test]
    fn dev_build_ahead_is_up_to_date() {
        let tags = vec!["v0.3.0".to_string()];
        assert_eq!(
            decide(&v("0.4.0"), &tags),
            UpdateStatus::UpToDate { current: v("0.4.0") }
        );
    }
}
