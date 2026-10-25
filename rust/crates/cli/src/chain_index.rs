//! The chain number — counted from 0, the one index base camdl uses for
//! every index a user sees (gh#990).
//!
//! A fit's chains are named to the user in many places: the `chain_N/`
//! directories, the progress and error lines, `chain_starts.tsv`,
//! `fit_state.toml`'s `best_chain` and `chain_eval_ids`, the per-chain tables
//! of `fit summary`, `--exclude-chains`, the `chain` column of `draws.tsv` and
//! of a `fit predict --by-chain` table. They all carry the same number: the
//! runner's own 0-based index, printed as it is. There is no second base and
//! therefore no conversion — the class of off-by-one that gh#781, gh#823,
//! gh#910 and gh#972 each fixed at one site had two bases as its cause, not a
//! missing `+ 1`.
//!
//! [`ChainIndex`] is that number as a type. Every struct field that names a
//! chain to a user holds one, so a bare `usize` cannot be written into an
//! artifact or a message as a chain by mistake; the runners keep their loop
//! index as `usize` for seeding and indexing and wrap it at the point a chain
//! is named. `Display` and serde render the number itself, so a TOML or JSON
//! consumer sees `best_chain = 2`, not a wrapper.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A chain's number, counted from 0 — the same number in every artifact and
/// message that names a chain, and the `chain` key of `draws.tsv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChainIndex(pub usize);

/// The directory-name prefix: a chain's outputs live under `chain_<n>/`.
const DIR_PREFIX: &str = "chain_";

impl ChainIndex {
    /// The chain numbers `0 .. n`, in order.
    pub fn all(n: usize) -> impl Iterator<Item = ChainIndex> {
        (0..n).map(ChainIndex)
    }

    /// `chain_<n>`: the name of the directory this chain's outputs are under.
    pub fn dir_name(self) -> String {
        format!("{DIR_PREFIX}{}", self.0)
    }

    /// `<stage_dir>/chain_<n>`.
    pub fn dir_in(self, stage_dir: &Path) -> PathBuf {
        stage_dir.join(self.dir_name())
    }

    /// The chain whose directory is named `name`, when `name` is `chain_<n>`.
    pub fn parse_dir_name(name: &str) -> Option<ChainIndex> {
        name.strip_prefix(DIR_PREFIX)?.parse().ok().map(ChainIndex)
    }
}

impl fmt::Display for ChainIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate so width and alignment flags (`{:>5}`) apply to the number.
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::str::FromStr for ChainIndex {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(ChainIndex)
    }
}

/// Refuse a stage leaf whose chain directories were written by a camdl that
/// counted them from 1.
///
/// Before gh#990 the directories were `chain_1/ … chain_N/` while
/// `draws.tsv`'s `chain` key ran `0 … N-1`; now both count from 0. A reader
/// that addressed such a leaf by name under the new rule — `--resume` opening
/// `chain_N/resume_state.bin`, `fit predict` following a draw's `chain` key
/// into `chain_<key>/trajectories.tsv`, `fit summary` labelling
/// `chain_N/trace.tsv` — would read chain `k` where it wanted chain `k - 1`
/// and miss the last chain, with nothing erroring. So the old shape is
/// refused outright, and the shape is unambiguous: every sampler creates a
/// directory for every chain it is asked to run, so a leaf of the new form
/// with any chain directory at all has `chain_0/`, and a leaf with `chain_1/`
/// and no `chain_0/` can only have been written under the old numbering.
pub fn refuse_one_based_layout(stage_dir: &Path) -> Result<(), String> {
    let has = |c: usize| ChainIndex(c).dir_in(stage_dir).is_dir();
    if has(1) && !has(0) {
        return Err(format!(
            "{}: this leaf was written by a camdl that counted chains from 1 \
             (it holds chain_1/ and no chain_0/). camdl now counts every chain \
             index from 0 (gh#990), the same numbering as the `chain` column of \
             draws.tsv, so this leaf's directories, `best_chain` and \
             `chain_eval_ids` cannot be read without shifting every chain by \
             one. Re-run the fit to rewrite the leaf under the current camdl.",
            stage_dir.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_the_bare_number_and_honours_width() {
        assert_eq!(ChainIndex(0).to_string(), "0");
        assert_eq!(ChainIndex(12).to_string(), "12");
        assert_eq!(format!("{:>4}|", ChainIndex(7)), "   7|");
        assert_eq!(format!("{:<3}|", ChainIndex(7)), "7  |");
    }

    #[test]
    fn serde_is_transparent() {
        assert_eq!(serde_json::to_string(&ChainIndex(3)).unwrap(), "3");
        assert_eq!(serde_json::from_str::<ChainIndex>("3").unwrap(), ChainIndex(3));
        assert_eq!(
            serde_json::to_string(&vec![ChainIndex(0), ChainIndex(2)]).unwrap(),
            "[0,2]"
        );
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct S { best_chain: ChainIndex, ids: Vec<ChainIndex> }
        let s = S { best_chain: ChainIndex(1), ids: vec![ChainIndex(0), ChainIndex(1)] };
        let toml = toml::to_string(&s).unwrap();
        assert!(toml.contains("best_chain = 1"), "{toml}");
        assert!(toml.contains("ids = [0, 1]"), "{toml}");
        assert_eq!(toml::from_str::<S>(&toml).unwrap(), s);
    }

    #[test]
    fn directory_names_round_trip() {
        assert_eq!(ChainIndex(0).dir_name(), "chain_0");
        assert_eq!(ChainIndex(0).dir_in(Path::new("/leaf")), PathBuf::from("/leaf/chain_0"));
        assert_eq!(ChainIndex::parse_dir_name("chain_10"), Some(ChainIndex(10)));
        assert_eq!(ChainIndex::parse_dir_name("chain_"), None);
        assert_eq!(ChainIndex::parse_dir_name("chain_x"), None);
        assert_eq!(ChainIndex::parse_dir_name("draws.tsv"), None);
        assert_eq!(ChainIndex::parse_dir_name("chain_starts.tsv"), None);
    }

    #[test]
    fn parses_from_a_cli_token() {
        assert_eq!("0".parse::<ChainIndex>().unwrap(), ChainIndex(0));
        assert_eq!("5".parse::<ChainIndex>().unwrap(), ChainIndex(5));
        assert!("-1".parse::<ChainIndex>().is_err());
        assert!("x".parse::<ChainIndex>().is_err());
    }

    #[test]
    fn all_counts_from_zero() {
        assert_eq!(ChainIndex::all(3).collect::<Vec<_>>(),
            vec![ChainIndex(0), ChainIndex(1), ChainIndex(2)]);
        assert_eq!(ChainIndex::all(0).count(), 0);
    }

    #[test]
    fn a_leaf_counted_from_one_is_refused_and_every_other_shape_passes() {
        let dir = crate::test_support::unique_temp_dir("chain_index_layout");
        std::fs::create_dir_all(&dir).unwrap();
        // No chain directories at all (an optimizer leaf): fine.
        assert!(refuse_one_based_layout(&dir).is_ok());
        // chain_1/ without chain_0/: the old numbering.
        std::fs::create_dir(dir.join("chain_1")).unwrap();
        let e = refuse_one_based_layout(&dir).unwrap_err();
        assert!(e.contains("counted chains from 1") && e.contains("gh#990"), "{e}");
        // chain_0/ present: the current numbering, whatever else is there.
        std::fs::create_dir(dir.join("chain_0")).unwrap();
        assert!(refuse_one_based_layout(&dir).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }
}
