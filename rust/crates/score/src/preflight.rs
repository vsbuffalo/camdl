//! The checks that decide whether fitted models may be compared, as values.
//!
//! A comparison runs every check and records one [`CheckResult`] per
//! [`CheckId`], in declaration order. A result has three parts, because one
//! check can do three things at once: an outcome over the members that carry
//! the fact it reads, the members that do not carry it ([`Unchecked`]), and
//! the members it covered on weaker evidence ([`Qualified`]). A member that
//! lacks a fact is listed as not checked; it never passes by default.
//!
//! A failed check carries a [`Finding`]: the members partitioned by the value
//! that differed, each group sorted by member name and the groups sorted by
//! value, so that a finding does not depend on the order the members were
//! given in. A failed check that is not waived is a [`Refusal`]; one that a
//! flag waived is an [`Override`].
//!
//! Design: `docs/dev/proposals/2026-11-14-scoring-report-library.md`,
//! "Every check is evaluated and recorded".

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::trace::{Coverage, PredictiveBasis, Provenance};

/// One preflight check. The declaration order is the order checks are run
/// and reported in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckId {
    /// The members' traces are on one predictive basis (a particle filter's
    /// estimate, or an ODE trajectory's exact density).
    Basis,
    /// The members are scored under one conditioning mode (in-sample, or
    /// held out).
    Conditioning,
    /// Held-out members sealed training at one boundary.
    TrainingBoundary,
    /// Members re-scored on one stream, in-sample, whose fits assimilated
    /// different streams.
    InSampleAblation,
    /// The members' predictives are of one provenance (plug-in, or a
    /// posterior mixture).
    Provenance,
    /// The members scored one number of steps.
    ScoredSteps,
    /// The members' scores pair: the same quantity, the same times, the same
    /// streams and the same windows at each time.
    Pairing,
    /// The members' fits were bound to the same observations.
    SharedData,
    /// The files a member scores are the bytes its fit was fitted to.
    ChangedData,
    /// The members were scored with the same particle and replicate counts.
    FilterSettings,
}

impl CheckId {
    /// Every check, in declaration order.
    pub const ALL: [CheckId; 10] = [
        CheckId::Basis,
        CheckId::Conditioning,
        CheckId::TrainingBoundary,
        CheckId::InSampleAblation,
        CheckId::Provenance,
        CheckId::ScoredSteps,
        CheckId::Pairing,
        CheckId::SharedData,
        CheckId::ChangedData,
        CheckId::FilterSettings,
    ];

    /// The flag that can waive this check, if any.
    pub fn waiver(self) -> Option<Waiver> {
        match self {
            CheckId::Basis
            | CheckId::Conditioning
            | CheckId::TrainingBoundary
            | CheckId::Pairing => None,
            CheckId::InSampleAblation => Some(Waiver::InSampleAblation),
            CheckId::Provenance => Some(Waiver::MixedProvenance),
            CheckId::ScoredSteps => Some(Waiver::MismatchedHorizon),
            CheckId::SharedData => Some(Waiver::DataMismatch),
            CheckId::ChangedData => Some(Waiver::ChangedData),
            CheckId::FilterSettings => Some(Waiver::MismatchedFilterSettings),
        }
    }
}

/// A preflight check that a `compare` flag lets a comparison render through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Waiver {
    MismatchedHorizon,
    MixedProvenance,
    DataMismatch,
    ChangedData,
    MismatchedFilterSettings,
    InSampleAblation,
}

impl Waiver {
    pub fn flag(self) -> &'static str {
        match self {
            Waiver::MismatchedHorizon => "--allow-mismatched-horizon",
            Waiver::MixedProvenance => "--allow-mixed-provenance",
            Waiver::DataMismatch => "--allow-data-mismatch",
            Waiver::ChangedData => "--allow-changed-data",
            Waiver::MismatchedFilterSettings => "--allow-mismatched-filter-settings",
            Waiver::InSampleAblation => "--allow-in-sample-ablation",
        }
    }

    /// What the failed check does to the reading of the table, in one
    /// sentence. The refusal and every rendering under the flag share it.
    pub fn consequence(self) -> &'static str {
        match self {
            Waiver::MismatchedHorizon =>
                "Δ columns suppressed ('—' in the table, null in the JSON): a Δ pairs \
                 two rows' scores observation by observation, and these rows scored \
                 different numbers of observations.",
            Waiver::MixedProvenance =>
                "A plug-in predictive is under-dispersed relative to a posterior \
                 mixture, so the Δ confounds model difference with dispersion treatment.",
            Waiver::DataMismatch =>
                "Δelpd across them is part model difference and part data difference, \
                 and the table cannot separate the two.",
            Waiver::ChangedData =>
                "`compare` re-reads these files, so the row is scored on data its θ̂ \
                 was not fitted to.",
            Waiver::MismatchedFilterSettings =>
                "The log of a particle-filter likelihood estimate is biased downward, \
                 and by more at fewer particles (and fewer replicates combined), so \
                 part of Δelpd would measure the filters rather than the models.",
            Waiver::InSampleAblation =>
                "Scored in-sample, Δelpd leans toward the fit that assimilated fewer \
                 streams: its θ̂ was chosen to fit the shared stream alone, while the \
                 other fit's traded that stream off against the rest, so on its own \
                 training data it fits it no better and usually slightly worse, \
                 whatever the extra streams contribute to prediction.",
        }
    }
}

/// The recorded result of one check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckResult {
    pub check: CheckId,
    /// The result over the members that carry the fact the check reads.
    pub outcome: CheckOutcome,
    /// Members the check could not cover, and why, sorted by member name.
    pub not_checked: Vec<Unchecked>,
    /// Members covered on weaker evidence, and how, sorted by member name.
    pub qualified: Vec<Qualified>,
    /// The flag that can waive the check, if any.
    pub waiver: Option<Waiver>,
}

impl CheckResult {
    /// A failed check that was not waived: the comparison is refused.
    pub fn refusal(&self) -> Option<Refusal> {
        match &self.outcome {
            CheckOutcome::Failed { finding, waived: false } =>
                Some(Refusal { check: self.check, finding: finding.clone() }),
            _ => None,
        }
    }

    /// A failed check that its waiver let through.
    pub fn override_(&self) -> Option<Override> {
        match (&self.outcome, self.waiver) {
            (CheckOutcome::Failed { finding, waived: true }, Some(waiver)) =>
                Some(Override { check: self.check, waiver, finding: finding.clone() }),
            _ => None,
        }
    }

    /// Whether a check that reads what this one established may run: it
    /// passed, had nothing to compare, or failed and was waived.
    pub fn established(&self) -> bool {
        match self.outcome {
            CheckOutcome::Passed
            | CheckOutcome::Noted { .. }
            | CheckOutcome::NothingToCompare
            | CheckOutcome::Failed { waived: true, .. } => true,
            CheckOutcome::Failed { waived: false, .. } | CheckOutcome::NotEvaluated { .. } => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckOutcome {
    Passed,
    Failed { finding: Finding, waived: bool },
    /// Reported, never refuses.
    Noted { finding: Finding },
    /// Fewer than two members carry the fact the check compares.
    NothingToCompare,
    /// A check this one reads what it established was not established: it
    /// failed unwaived, or was itself not evaluated.
    NotEvaluated { because: CheckId },
}

/// A member a check could not cover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unchecked {
    pub member: String,
    pub reason: NotCheckedReason,
}

/// Why a member is not covered by a check.
///
/// `why` fields carry the explanation recorded when the member was loaded
/// (a missing provenance sidecar, a file that could not be read, a sidecar
/// written before a field existed), as the loader states it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotCheckedReason {
    /// The member records no data identity at all.
    NoDataIdentity { why: String },
    /// The member's record has no usable hash for one source it binds.
    SourceUnresolved { source: String, file: DataFile, why: String },
    /// One file the member scores could not be compared with its record.
    FileNotChecked { stream: String, path: String, why: String },
    /// The trace is older than schema 5, which first recorded filter settings.
    TraceBeforeSchema5,
    /// The trace's producer wrote no derivation block.
    NoDerivationBlock,
}

/// Which of a fit's bound files a data hash is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFile {
    /// A training file.
    Training,
    /// A `[data.holdout]` file.
    Holdout,
}

/// A member a check covered on weaker evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Qualified {
    pub member: String,
    pub how: Qualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qualification {
    /// The member's record holds 8-character hashes (a sidecar written before
    /// full digests were recorded), and its data was compared on that prefix.
    HashPrefixOnly,
}

/// A failed, unwaived check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refusal {
    pub check: CheckId,
    pub finding: Finding,
}

/// A waiver that was given and needed: the check failed, and the flag let
/// the comparison render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Override {
    pub check: CheckId,
    pub waiver: Waiver,
    pub finding: Finding,
}

/// Members grouped by one value: each group's members sorted by name, the
/// groups sorted by value.
pub type Groups<V> = Vec<(V, Vec<String>)>;

/// What a failed check found. One variant per way a check can fail; each
/// carries the members partitioned by the value that differed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Finding {
    /// `Basis`: the members' predictive bases.
    BasisDiffers { groups: Groups<PredictiveBasis> },
    /// `Conditioning`: the members' conditioning modes.
    ConditioningDiffers { groups: Groups<ConditioningMode> },
    /// `TrainingBoundary`: the held-out members' training boundaries, members
    /// within the holdout time tolerance of a group's value in one group.
    TrainingBoundaryDiffers { groups: Groups<f64> },
    /// `InSampleAblation`: the streams each member's fit assimilated. A
    /// member whose set is not recorded is a group of its own, since an
    /// unknown set is not known to equal any other.
    InSampleAblation { groups: Groups<Option<Vec<String>>> },
    /// `Provenance`: the members' predictive provenance.
    ProvenanceDiffers { groups: Groups<Provenance> },
    /// `ScoredSteps`: the number of steps each member scored.
    ScoredStepsDiffer { groups: Groups<usize> },
    /// `Pairing`: what each member's step scores are of: every stream its fit
    /// observed (`None`), or the listed streams alone.
    ScoredQuantityDiffers { groups: Groups<Option<Vec<String>>> },
    /// `Pairing`: the observation time of scored step `step` (counted from 1),
    /// the first step at which the members' times differ.
    ObservationTimesDiffer { step: usize, groups: Groups<f64> },
    /// `Pairing`: the streams scored at time `t`, the first time at which
    /// they differ. An empty set is a step with no per-stream breakdown.
    /// `shared_streams` are the streams every member in the groups scores
    /// somewhere, sorted.
    StreamSetsDiffer { t: f64, groups: Groups<Vec<String>>, shared_streams: Vec<String> },
    /// `Pairing`: what `stream`'s value at time `t` covers.
    CoverageDiffers { t: f64, stream: String, groups: Groups<Coverage> },
    /// `SharedData`: per source, the recorded hash of each member's file
    /// (`None`: the member does not bind it), compared on the length of the
    /// shortest; and the members that record no hash for the data the table
    /// scores.
    DataDiffers { sources: Vec<SourceDigests>, unhashed: Vec<UnhashedData> },
    /// `ChangedData`: the files whose bytes changed after their fit ran.
    DataChanged { files: Vec<ChangedFile> },
    /// `FilterSettings`: the particle and replicate counts of the members that
    /// record them. A dimension in which the members agree has one group.
    FilterSettingsDiffer { particles: Groups<usize>, replicates: Groups<usize> },
}

/// A member's conditioning mode, as the conditioning check compares it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningMode {
    InSample,
    HoldOutTail { train_end: f64 },
}

/// One source's recorded hashes across the members that bind it or could.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDigests {
    pub source: String,
    pub file: DataFile,
    /// The compared prefix of each member's hash, or `None` where the member
    /// does not bind the source.
    pub groups: Groups<Option<String>>,
}

/// A member that records no hash for the data a one-stream table scores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnhashedData {
    pub member: String,
    pub what: Unhashed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unhashed {
    /// The member's fit does not bind the stream the table scores.
    StreamNotBound { stream: String },
    /// The member's record has no hash for the source the table scores.
    NoHashForSource { source: String },
}

/// One file scored on bytes other than those its fit read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedFile {
    pub member: String,
    pub stream: String,
    pub path: String,
    /// The hash the fit recorded, abbreviated.
    pub recorded: String,
    /// The hash of the file now, abbreviated.
    pub current: String,
}

impl Finding {
    /// Every member the finding names, sorted and without repeats.
    pub fn members(&self) -> Vec<String> {
        fn of<V>(g: &Groups<V>) -> impl Iterator<Item = &String> {
            g.iter().flat_map(|(_, m)| m)
        }
        let mut out: Vec<String> = match self {
            Finding::BasisDiffers { groups } => of(groups).cloned().collect(),
            Finding::ConditioningDiffers { groups } => of(groups).cloned().collect(),
            Finding::TrainingBoundaryDiffers { groups } => of(groups).cloned().collect(),
            Finding::InSampleAblation { groups } => of(groups).cloned().collect(),
            Finding::ProvenanceDiffers { groups } => of(groups).cloned().collect(),
            Finding::ScoredStepsDiffer { groups } => of(groups).cloned().collect(),
            Finding::ScoredQuantityDiffers { groups } => of(groups).cloned().collect(),
            Finding::ObservationTimesDiffer { groups, .. } => of(groups).cloned().collect(),
            Finding::StreamSetsDiffer { groups, .. } => of(groups).cloned().collect(),
            Finding::CoverageDiffers { groups, .. } => of(groups).cloned().collect(),
            Finding::DataDiffers { sources, unhashed } => sources.iter()
                .flat_map(|s| of(&s.groups).cloned())
                .chain(unhashed.iter().map(|u| u.member.clone()))
                .collect(),
            Finding::DataChanged { files } => files.iter().map(|f| f.member.clone()).collect(),
            Finding::FilterSettingsDiffer { particles, replicates } =>
                of(particles).chain(of(replicates)).cloned().collect(),
        };
        out.sort();
        out.dedup();
        out
    }
}

/// Partition `items` (a value and a member name each) into groups: items
/// are sorted by `order` on the value, then by name, and an item joins the
/// group before it when `same(group's value, its value)`; otherwise it starts
/// a group. The result depends only on the multiset of items, not their
/// order, provided `order` is a total order and `same` holds only between
/// values `order` places together.
pub fn partition<V: Clone>(
    mut items: Vec<(V, String)>,
    order: impl Fn(&V, &V) -> Ordering,
    same: impl Fn(&V, &V) -> bool,
) -> Groups<V> {
    items.sort_by(|(va, na), (vb, nb)| order(va, vb).then_with(|| na.cmp(nb)));
    let mut groups: Groups<V> = Vec::new();
    for (v, name) in items {
        match groups.last_mut() {
            Some((gv, members)) if same(gv, &v) => members.push(name),
            _ => groups.push((v, vec![name])),
        }
    }
    groups
}

/// Sort `entries` by member name, keeping the order of each member's own
/// entries: a check's lists then do not depend on the order of the members.
pub fn sort_by_member<T>(entries: &mut [T], member: impl Fn(&T) -> &str) {
    entries.sort_by(|a, b| member(a).cmp(member(b)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_lists_every_check_once_in_declaration_order() {
        // Exhaustive: a new variant fails to compile here until it is placed.
        let position = |c: CheckId| match c {
            CheckId::Basis => 0,
            CheckId::Conditioning => 1,
            CheckId::TrainingBoundary => 2,
            CheckId::InSampleAblation => 3,
            CheckId::Provenance => 4,
            CheckId::ScoredSteps => 5,
            CheckId::Pairing => 6,
            CheckId::SharedData => 7,
            CheckId::ChangedData => 8,
            CheckId::FilterSettings => 9,
        };
        for (i, c) in CheckId::ALL.iter().enumerate() {
            assert_eq!(position(*c), i, "{c:?}");
        }
        assert!(CheckId::ALL.windows(2).all(|w| w[0] < w[1]), "declaration order");
    }

    #[test]
    fn a_partition_does_not_depend_on_the_order_of_its_items() {
        let items = vec![(3, "c".to_string()), (1, "b".into()), (3, "a".into()), (2, "d".into())];
        let want = vec![(1, vec!["b".to_string()]), (2, vec!["d".into()]),
                        (3, vec!["a".into(), "c".into()])];
        let mut rev = items.clone();
        rev.reverse();
        for it in [items, rev] {
            assert_eq!(partition(it, |a, b| a.cmp(b), |a, b| a == b), want);
        }
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// One result for every outcome, every finding, every reason and every
    /// qualification: a `Vec<CheckResult>` reads back equal to what was
    /// written.
    #[test]
    fn check_results_round_trip_through_json() {
        let findings = vec![
            Finding::BasisDiffers { groups: vec![
                (PredictiveBasis::Filtered, names(&["a"])),
                (PredictiveBasis::OdeFreeForward { samples_per_draw: 100 }, names(&["b"]))] },
            Finding::ConditioningDiffers { groups: vec![
                (ConditioningMode::InSample, names(&["a"])),
                (ConditioningMode::HoldOutTail { train_end: 42.5 }, names(&["b", "c"]))] },
            Finding::TrainingBoundaryDiffers { groups: vec![(21.0, names(&["a"])), (21.5, names(&["b"]))] },
            Finding::InSampleAblation { groups: vec![
                (None, names(&["pq"])), (Some(names(&["cases"])), names(&["b"]))] },
            Finding::ProvenanceDiffers { groups: vec![
                (Provenance::PlugIn, names(&["a"])),
                (Provenance::Posterior { n_draws: 4 }, names(&["b"]))] },
            Finding::ScoredStepsDiffer { groups: vec![(2, names(&["b"])), (3, names(&["a"]))] },
            Finding::ScoredQuantityDiffers { groups: vec![
                (None, names(&["joint"])), (Some(names(&["cases"])), names(&["projected"]))] },
            Finding::ObservationTimesDiffer { step: 3, groups: vec![(21.0, names(&["a"])), (28.0, names(&["b"]))] },
            Finding::StreamSetsDiffer { t: 7.0, groups: vec![
                (vec![], names(&["v1"])), (names(&["north", "south"]), names(&["a"]))],
                shared_streams: names(&["north"]) },
            Finding::CoverageDiffers { t: 6.0, stream: "cases".into(), groups: vec![
                (Coverage::Unrecorded, names(&["a"])), (Coverage::Instant, names(&["b"])),
                (Coverage::Interval { start: 4.0, stop: 6.0 }, names(&["c"]))] },
            Finding::DataDiffers {
                sources: vec![SourceDigests { source: "cases".into(), file: DataFile::Holdout,
                    groups: vec![(None, names(&["c"])), (Some("aaaaaaaa".into()), names(&["a"]))] }],
                unhashed: vec![
                    UnhashedData { member: "b".into(), what: Unhashed::StreamNotBound { stream: "deaths".into() } },
                    UnhashedData { member: "c".into(), what: Unhashed::NoHashForSource { source: "cases".into() } }],
            },
            Finding::DataChanged { files: vec![ChangedFile { member: "b".into(), stream: "cases".into(),
                path: "cases.tsv".into(), recorded: "aaaaaaaa".into(), current: "bbbbbbbb".into() }] },
            Finding::FilterSettingsDiffer { particles: vec![(200, names(&["b"])), (1000, names(&["a"]))],
                replicates: vec![(5, names(&["a", "b"]))] },
        ];
        let mut results: Vec<CheckResult> = findings.iter().enumerate().map(|(i, f)| CheckResult {
            check: CheckId::ALL[i % CheckId::ALL.len()],
            outcome: if i % 2 == 0 {
                CheckOutcome::Failed { finding: f.clone(), waived: false }
            } else {
                CheckOutcome::Failed { finding: f.clone(), waived: true }
            },
            not_checked: vec![],
            qualified: vec![],
            waiver: CheckId::ALL[i % CheckId::ALL.len()].waiver(),
        }).collect();
        let reasons = vec![
            NotCheckedReason::NoDataIdentity { why: "no sidecar".into() },
            NotCheckedReason::SourceUnresolved { source: "cases".into(), file: DataFile::Training,
                why: "the fit recorded no hash for it".into() },
            NotCheckedReason::FileNotChecked { stream: "cases".into(), path: "cases.tsv".into(),
                why: "it could not be read".into() },
            NotCheckedReason::TraceBeforeSchema5,
            NotCheckedReason::NoDerivationBlock,
        ];
        results.push(CheckResult {
            check: CheckId::SharedData,
            outcome: CheckOutcome::Passed,
            not_checked: reasons.into_iter()
                .map(|reason| Unchecked { member: "a".into(), reason }).collect(),
            qualified: vec![Qualified { member: "b".into(), how: Qualification::HashPrefixOnly }],
            waiver: Some(Waiver::DataMismatch),
        });
        for outcome in [CheckOutcome::NothingToCompare,
                        CheckOutcome::NotEvaluated { because: CheckId::Conditioning },
                        CheckOutcome::Noted { finding: findings[0].clone() }] {
            results.push(CheckResult { check: CheckId::Pairing, outcome, not_checked: vec![],
                qualified: vec![], waiver: None });
        }
        let text = serde_json::to_string(&results).unwrap();
        let back: Vec<CheckResult> = serde_json::from_str(&text).unwrap();
        assert_eq!(back, results, "{text}");
        assert!(text.contains("\"shared_data\"") && text.contains("\"hash_prefix_only\"")
                && text.contains("\"not_evaluated\""), "snake case names: {text}");
    }

    #[test]
    fn a_waiver_round_trips_under_its_snake_case_name() {
        for (w, name) in [(Waiver::MismatchedHorizon, "mismatched_horizon"),
                          (Waiver::InSampleAblation, "in_sample_ablation")] {
            let text = serde_json::to_string(&w).unwrap();
            assert_eq!(text, format!("\"{name}\""));
            assert_eq!(serde_json::from_str::<Waiver>(&text).unwrap(), w);
        }
    }
}
