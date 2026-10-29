//! Parameter roles: what a run does with each `parameters {}` entry.
//!
//! A run holds a parameter at one value (**fixed**), searches over it
//! (**estimated**), or sets it per cell from a grid the user wrote
//! (**design**). That assignment is the run's [`Partition`], resolved here
//! from role declarations only — `[estimate]`, `--estimate`, `[fixed]`,
//! `--fixed`, `--fixed-file` and `--sweep` — before any value is resolved.
//! Each parameter has at most one declaration; two of different roles are an
//! error ([`RoleError::TwoDeclarations`]), except for the special cases of
//! [`SpecialCase`].
//!
//! A typed constant (`let omega : rate = 0.01`) is not a parameter: it is in
//! the IR's `constants`, not its `parameters`, so it is in no partition and no
//! declaration can name it. One that tries is [`RoleError::UnknownName`], whose
//! message carries the hint that makes it settable.
//!
//! The partition holds no values. The stored record ([`PartitionRecord`])
//! pairs it with the values the run resolved, and the legacy `estimated` and
//! `fixed` fields of a run's sidecar are generated from that record by
//! [`PartitionRecord::legacy_fields`] and nowhere else.
//!
//! This module reads role declarations, never a command's algorithm flags:
//! `profile`'s legacy `--rw-sd` selection is translated into [`RoleInputs`]
//! by `profile::legacy_role_inputs`, so no rule here depends on it.
//!
//! The sampled role (a measure's varying set, `simulate --draws`) is not here
//! yet: no command that resolves roles takes a sample axis until `simulate`
//! and `fit predict` move onto the partition.
//!
//! Proposal: `docs/dev/proposals/2026-10-26-parameter-roles.md` §6.

use std::collections::BTreeMap;
use std::path::PathBuf;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sim::inference::types::Transform;

use crate::fit::config_v2::{EstimateSpecV2, SimplexGroup};
use crate::fit::priors_precedence::{PriorSource, ResolvedPrior};
use crate::params_resolver::ResolvedParameters;

// ─── The partition ───────────────────────────────────────────────────────────

/// What a run does with one `parameters {}` entry.
#[derive(Debug, Clone)]
pub enum Role {
    /// One value per scenario arm, from the value tiers.
    Fixed,
    /// Searched by the method.
    Estimated(Box<Estimated>),
    /// Set per cell from the design axis; fixed inside each cell.
    Design,
}

impl Role {
    pub fn kind(&self) -> RoleKind {
        match self {
            Role::Fixed => RoleKind::Fixed,
            Role::Estimated(_) => RoleKind::Estimated,
            Role::Design => RoleKind::Design,
        }
    }
}

/// The model's admissible values: `in [lo, hi]`, else the kind's range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Support {
    pub lower: f64,
    pub upper: f64,
    pub from: SupportFrom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportFrom {
    Declared,
    Kind,
}

/// Where the method searches: within the support. Part of a likelihood
/// estimand, because a narrower box can change a maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchBox {
    pub lower: f64,
    pub upper: f64,
    pub from: BoxFrom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoxFrom {
    FitConfig,
    Cli,
    /// The support: the model's declared range, or the kind's finite one.
    Support,
    /// Nothing declared and the box differs from the support: the runner's
    /// `[0, ∞)` (`fit::runner::default_search_box`) for a parameter whose IR
    /// carries no kind, whose support is the whole line. A `real` or
    /// `instant` parameter searches its support, `(−∞, ∞)`.
    Fallback,
}

/// An estimated parameter: its support, search box and prior are distinct.
#[derive(Debug, Clone)]
pub struct Estimated {
    pub support: Support,
    pub search: SearchBox,
    /// Clamps to `search`.
    pub transform: Transform,
    pub transform_from: TransformFrom,
    /// Read by the Bayesian methods only; always recorded.
    pub prior: ResolvedPrior,
    /// The prior in the canonical `family(arg=value, ...)` form, for the
    /// record.
    pub prior_text: String,
    /// `[estimate].start`, if written.
    pub start: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformFrom {
    FitConfig,
    Kind,
}

/// How a `[fixed]` entry reached the fit config.
#[derive(Debug, Clone, PartialEq)]
pub enum FitFixedVia {
    Inline,
    FromFile(PathBuf),
    FromScenario(String),
}

/// Which flag declared a design parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DesignOrigin {
    /// `--sweep` (`fit run`, `profile`).
    Sweep,
}

/// What declared a parameter's role.
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    /// Named by nothing: fixed.
    Default,
    FitEstimate,
    FitFixed(FitFixedVia),
    CliEstimate,
    /// `--fixed`.
    CliFixed,
    /// `--fixed-file`.
    CliFixedFile(PathBuf),
    Design(DesignOrigin),
}

impl Declaration {
    /// The role a declaration of this kind declares.
    fn role_kind(&self) -> RoleKind {
        match self {
            Declaration::Default
            | Declaration::FitFixed(_)
            | Declaration::CliFixed
            | Declaration::CliFixedFile(_) => RoleKind::Fixed,
            Declaration::FitEstimate | Declaration::CliEstimate => RoleKind::Estimated,
            Declaration::Design(_) => RoleKind::Design,
        }
    }

    /// The value tier a fixed declaration supplies (higher wins): `[fixed]`
    /// is tier 2, `--fixed-file` tier 3, `--fixed` tier 5.
    fn fixed_tier(&self) -> u8 {
        match self {
            Declaration::FitFixed(_) => 2,
            Declaration::CliFixedFile(_) => 3,
            Declaration::CliFixed => 5,
            _ => 0,
        }
    }

    /// The declaration as a reader would write it.
    pub fn describe(&self) -> String {
        match self {
            Declaration::Default => "no declaration (fixed by default)".into(),
            Declaration::FitEstimate => "[estimate] in the fit config".into(),
            Declaration::FitFixed(FitFixedVia::Inline) => "[fixed] in the fit config".into(),
            Declaration::FitFixed(FitFixedVia::FromFile(p)) => {
                format!("[fixed] from_file = \"{}\" in the fit config", p.display())
            }
            Declaration::FitFixed(FitFixedVia::FromScenario(s)) => {
                format!("[fixed] from_scenario = \"{s}\" in the fit config")
            }
            Declaration::CliEstimate => "--estimate".into(),
            Declaration::CliFixed => "--fixed".into(),
            Declaration::CliFixedFile(p) => format!("--fixed-file {}", p.display()),
            Declaration::Design(DesignOrigin::Sweep) => "--sweep".into(),
        }
    }

    /// The record's `declared_by` tag and, where one applies, the file or
    /// scenario it names and how a `[fixed]` entry reached the config.
    fn record_parts(&self) -> (&'static str, Option<&'static str>, Option<String>) {
        match self {
            Declaration::Default => ("default", None, None),
            Declaration::FitEstimate => ("fit_estimate", None, None),
            Declaration::FitFixed(FitFixedVia::Inline) => ("fit_fixed", Some("inline"), None),
            Declaration::FitFixed(FitFixedVia::FromFile(p)) => {
                ("fit_fixed", Some("from_file"), Some(p.display().to_string()))
            }
            Declaration::FitFixed(FitFixedVia::FromScenario(s)) => {
                ("fit_fixed", Some("from_scenario"), Some(s.clone()))
            }
            Declaration::CliEstimate => ("cli_estimate", None, None),
            Declaration::CliFixed => ("cli_fixed", None, None),
            Declaration::CliFixedFile(p) => ("cli_fixed_file", None, Some(p.display().to_string())),
            Declaration::Design(DesignOrigin::Sweep) => ("cli_sweep", None, None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleKind {
    Fixed,
    Estimated,
    Design,
}

impl RoleKind {
    fn word(self) -> &'static str {
        match self {
            RoleKind::Fixed => "fixed",
            RoleKind::Estimated => "estimated",
            RoleKind::Design => "design (swept)",
        }
    }
}

/// One of the operations that names a parameter twice without a conflict,
/// when it applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialCase {
    /// `profile --sweep` over a fit config's `[estimate]` entry: the focal
    /// parameter; the entry is not used for it.
    ProfileFocal,
    /// `--sweep` over a `[fixed]` entry: a design coordinate at job level,
    /// fixed at the grid value in each cell; the `[fixed]` value is not used.
    SweepOverFixed,
}

#[derive(Debug, Clone)]
pub struct RoleEntry {
    pub role: Role,
    pub declared_by: Declaration,
    pub special: Option<SpecialCase>,
}

/// A `[[simplex_groups]]` group with the role its members share.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSimplex {
    pub members: Vec<String>,
    pub role: SimplexRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimplexRole {
    /// Every member estimated: the method perturbs the group jointly.
    Estimated,
    /// Every member fixed or design: the group is a check that the held
    /// values sum to one.
    Held,
}

/// Exhaustive: exactly one entry per `parameters {}` name, in declaration
/// order. Built only by [`resolve_roles`].
#[derive(Debug, Clone)]
pub struct Partition {
    entries: IndexMap<String, RoleEntry>,
    simplex_groups: Vec<ResolvedSimplex>,
    /// Entries of a fit artifact passed as `--fixed-file` that name a design
    /// or estimated parameter: not used, and listed.
    unused_file_entries: Vec<(PathBuf, String)>,
    /// The estimated names in the order their declarations gave them — the
    /// order of every method's parameter columns. `entries` keep model order.
    estimate_order: Vec<String>,
}

impl Partition {
    /// The names whose role is `kind`, in declaration order.
    pub fn names_with(&self, kind: RoleKind) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .filter(move |(_, e)| e.role.kind() == kind)
            .map(|(n, _)| n.as_str())
    }

    /// Every parameter's name, in declaration order.
    pub fn all_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// One info line per special case that applied and per unused file entry.
    pub fn notices(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, e) in &self.entries {
            match e.special {
                Some(SpecialCase::ProfileFocal) => out.push(format!(
                    "{name} is the focal parameter (--sweep); its [estimate] entry in the \
                     fit config is not used")),
                Some(SpecialCase::SweepOverFixed) => out.push(format!(
                    "{name} is swept (--sweep); its [fixed] value in the fit config is not \
                     used")),
                None => {}
            }
        }
        for (path, name) in &self.unused_file_entries {
            out.push(format!(
                "{name}: its entry in the fit artifact {} is not used ({name} is {})",
                path.display(),
                self.entries.get(name).map(|e| e.role.kind().word()).unwrap_or("not held")));
        }
        out
    }

    /// The record of the job's partition: design parameters as design.
    pub fn record(&self, values: &ResolvedParameters) -> PartitionRecord {
        self.record_at(values, None)
    }

    /// The partition one cell runs: each design parameter fixed at that
    /// cell's value. `point` must give a value to every design parameter and
    /// to nothing else.
    pub fn cell<'p>(&'p self, point: &'p [(String, f64)]) -> Result<CellPartition<'p>, String> {
        let design: Vec<&str> = self.names_with(RoleKind::Design).collect();
        for name in &design {
            if !point.iter().any(|(n, _)| n == name) {
                return Err(format!("design parameter `{name}` has no value at this cell"));
            }
        }
        if let Some((n, _)) = point.iter().find(|(n, _)| !design.contains(&n.as_str())) {
            return Err(format!("`{n}` has a cell value but is not a design parameter"));
        }
        Ok(CellPartition { partition: self, point })
    }

    fn record_at(
        &self,
        values: &ResolvedParameters,
        point: Option<&[(String, f64)]>,
    ) -> PartitionRecord {
        let value_of = |name: &str| values.params.iter().find(|rp| rp.name == name);
        let mut parameters = IndexMap::new();
        for (name, e) in &self.entries {
            let (declared_by, via, declared_in) = e.declared_by.record_parts();
            let declared = DeclaredRecord {
                declared_by: declared_by.to_string(),
                via: via.map(str::to_string),
                declared_in,
            };
            let rec = match (&e.role, point) {
                (Role::Fixed, _) => {
                    let rp = value_of(name);
                    ParamRecord::Fixed {
                        job_role: None,
                        special: e.special,
                        declared,
                        value: rp.map(|rp| rp.value),
                        value_from: rp.map(|rp| ValueFrom::from_source(&rp.source)),
                    }
                }
                (Role::Design, Some(point)) => {
                    let v = point.iter().find(|(n, _)| n == name).map(|(_, v)| *v);
                    ParamRecord::Fixed {
                        job_role: Some(RoleKind::Design),
                        special: e.special,
                        declared,
                        value: v,
                        value_from: Some(ValueFrom::design_point()),
                    }
                }
                (Role::Design, None) => ParamRecord::Design { special: e.special, declared },
                (Role::Estimated(est), _) => ParamRecord::Estimated {
                    special: e.special,
                    declared,
                    support: [Bound(est.support.lower), Bound(est.support.upper)],
                    support_from: est.support.from,
                    search: [Bound(est.search.lower), Bound(est.search.upper)],
                    search_from: est.search.from,
                    transform: transform_tag(&est.transform).to_string(),
                    transform_from: est.transform_from,
                    prior: PriorRecord {
                        dist: est.prior_text.clone(),
                        from: est.prior.source,
                    },
                    start: est.start,
                },
            };
            parameters.insert(name.clone(), rec);
        }
        PartitionRecord {
            parameters,
            estimate_order: self.estimate_order.clone(),
            simplex_groups: self.simplex_groups.iter()
                .map(|g| SimplexRecord { members: g.members.clone(), role: g.role })
                .collect(),
            notices: self.notices(),
        }
    }
}

/// The partition one cell runs ([`Partition::cell`]).
pub struct CellPartition<'p> {
    partition: &'p Partition,
    point: &'p [(String, f64)],
}

impl CellPartition<'_> {
    pub fn record(&self, values: &ResolvedParameters) -> PartitionRecord {
        self.partition.record_at(values, Some(self.point))
    }
}

fn transform_tag(t: &Transform) -> &'static str {
    match t {
        Transform::Log { .. } => "log",
        Transform::Logit { .. } => "logit",
        Transform::None => "identity",
    }
}

// ─── The stored record ───────────────────────────────────────────────────────

/// A run's `partition` field: the resolved partition, or — for a run today's
/// rules accept and [`resolve_roles`] refuses — the refusal, so no record
/// claims a partition that was not resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PartitionField {
    Resolved(PartitionRecord),
    Unresolved { reason: String },
}

/// A run's `partition` field and the legacy `estimated` / `fixed` fields,
/// generated from the record. For roles that form no partition — a run
/// today's rules accept and [`resolve_roles`] refuses, or one whose values
/// did not resolve — the field is `unresolved` and the legacy fields
/// are the caller's `unpartitioned` ones, as the command wrote them before the
/// partition existed. That fall-back goes when the command refuses such runs
/// (Stages 2 and 3 of the proposal).
pub struct RoleRecord {
    pub partition: PartitionField,
    pub estimated: Vec<String>,
    pub fixed: BTreeMap<String, f64>,
}

impl RoleRecord {
    pub fn new(
        record: Result<PartitionRecord, String>,
        unpartitioned: impl FnOnce() -> (Vec<String>, BTreeMap<String, f64>),
    ) -> Self {
        match record {
            Ok(record) => {
                let (estimated, fixed) = record.legacy_fields();
                RoleRecord { partition: PartitionField::Resolved(record), estimated, fixed }
            }
            Err(reason) => {
                let (estimated, fixed) = unpartitioned();
                RoleRecord { partition: PartitionField::Unresolved { reason }, estimated, fixed }
            }
        }
    }
}

/// The authoritative record of a run's (or a cell's) roles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionRecord {
    pub parameters: IndexMap<String, ParamRecord>,
    /// The estimated names in declaration order, for [`Self::legacy_fields`]:
    /// the legacy `estimated` list has always followed the columns. Not
    /// stored — the order is that of `[estimate]` / `--estimate` — so a
    /// record read back carries none, and its legacy list falls back to
    /// model order.
    #[serde(skip)]
    pub estimate_order: Vec<String>,
    pub simplex_groups: Vec<SimplexRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notices: Vec<String>,
}

impl PartitionRecord {
    /// The legacy `estimated` and `fixed` sidecar fields, generated here and
    /// nowhere else: the estimated names in declaration order, and every held
    /// parameter's value — fixed parameters and design parameters at their
    /// cell value.
    pub fn legacy_fields(&self) -> (Vec<String>, BTreeMap<String, f64>) {
        // Declaration order first; any estimated name it lacks (a record read
        // back from disk) follows in model order.
        let mut estimated: Vec<String> = self.estimate_order.iter()
            .filter(|n| matches!(self.parameters.get(*n), Some(ParamRecord::Estimated { .. })))
            .cloned()
            .collect();
        let mut fixed = BTreeMap::new();
        for (name, rec) in &self.parameters {
            match rec {
                ParamRecord::Estimated { .. } => {
                    if !self.estimate_order.contains(name) {
                        estimated.push(name.clone());
                    }
                }
                ParamRecord::Fixed { value: Some(v), .. } => {
                    fixed.insert(name.clone(), *v);
                }
                ParamRecord::Fixed { value: None, .. } | ParamRecord::Design { .. } => {}
            }
        }
        (estimated, fixed)
    }
}

/// What declared a parameter, as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredRecord {
    pub declared_by: String,
    /// How a `[fixed]` entry reached the fit config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
    /// The file or scenario the declaration names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_in: Option<String>,
}

/// One parameter's record. A cell's design parameter is `fixed` with
/// `job_role: design`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ParamRecord {
    Fixed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        job_role: Option<RoleKind>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        special: Option<SpecialCase>,
        #[serde(flatten)]
        declared: DeclaredRecord,
        value: Option<f64>,
        value_from: Option<ValueFrom>,
    },
    Estimated {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        special: Option<SpecialCase>,
        #[serde(flatten)]
        declared: DeclaredRecord,
        support: [Bound; 2],
        support_from: SupportFrom,
        search: [Bound; 2],
        search_from: BoxFrom,
        transform: String,
        transform_from: TransformFrom,
        prior: PriorRecord,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start: Option<f64>,
    },
    Design {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        special: Option<SpecialCase>,
        #[serde(flatten)]
        declared: DeclaredRecord,
    },
}

/// Where a held value came from: a value-tier tag
/// ([`crate::params_resolver::ValueSource::tag`]) or `design_point`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueFrom {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl ValueFrom {
    fn from_source(src: &crate::params_resolver::ValueSource) -> Self {
        use crate::params_resolver::{StartsOrigin, ValueSource};
        ValueFrom {
            source: src.tag().to_string(),
            scenario: match src {
                ValueSource::Scenario(n) => Some(n.clone()),
                _ => None,
            },
            path: match src {
                ValueSource::FixedFile { path }
                | ValueSource::Starts(StartsOrigin::Params { path }) => {
                    Some(path.display().to_string())
                }
                ValueSource::Starts(StartsOrigin::Mle { fit }) => Some(fit.clone()),
                _ => None,
            },
        }
    }

    fn design_point() -> Self {
        ValueFrom { source: "design_point".into(), scenario: None, path: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriorRecord {
    pub dist: String,
    pub from: PriorSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimplexRecord {
    pub members: Vec<String>,
    pub role: SimplexRole,
}

/// An interval end: a number when finite, `"inf"` / `"-inf"` otherwise (JSON
/// has no infinity, and `serde_json` would write `null`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound(pub f64);

impl Serialize for Bound {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            s.serialize_f64(self.0)
        } else if self.0 > 0.0 {
            s.serialize_str("inf")
        } else {
            s.serialize_str("-inf")
        }
    }
}

impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(f64),
            Text(String),
        }
        match Raw::deserialize(d)? {
            Raw::Num(v) => Ok(Bound(v)),
            Raw::Text(t) if t == "inf" => Ok(Bound(f64::INFINITY)),
            Raw::Text(t) if t == "-inf" => Ok(Bound(f64::NEG_INFINITY)),
            Raw::Text(t) => Err(serde::de::Error::custom(format!("not a bound: {t}"))),
        }
    }
}

// ─── Which roles a command accepts ───────────────────────────────────────────

/// The roles a command accepts (proposal §4.4). Fixed is always accepted.
pub struct CommandRoles {
    pub command: &'static str,
    pub estimated: Allowed,
    pub design: Allowed,
    /// Whether a `--sweep` over a fit config's `[estimate]` entry is the
    /// focal parameter of a profile ([`SpecialCase::ProfileFocal`]).
    pub profiles: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allowed {
    No,
    Optional,
    AtLeastOne,
    /// At least one, unless the run says it is a slice (`profile --slice`).
    AtLeastOneUnlessSlice,
}

pub const FIT_RUN: CommandRoles = CommandRoles {
    command: "fit run",
    estimated: Allowed::Optional,
    design: Allowed::Optional,
    profiles: false,
};
pub const SURVEY: CommandRoles = CommandRoles {
    command: "survey",
    estimated: Allowed::AtLeastOne,
    design: Allowed::No,
    profiles: false,
};
pub const PROFILE: CommandRoles = CommandRoles {
    command: "profile",
    estimated: Allowed::AtLeastOneUnlessSlice,
    design: Allowed::AtLeastOne,
    profiles: true,
};

// ─── Inputs ──────────────────────────────────────────────────────────────────

/// `--estimate NAME[=LO:HI]`.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimateDecl {
    pub name: String,
    pub bounds: Option<(f64, f64)>,
}

/// A `--fixed-file` and the names it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedFileDecl {
    pub path: PathBuf,
    pub names: Vec<String>,
    /// A params file a fit wrote (it carries a `[provenance]` block): its
    /// entries for design or estimated names are not used, and are listed.
    pub fit_artifact: bool,
}

/// The parameters one scenario arm sets or scales.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioFootprint {
    pub scenario: String,
    pub params: Vec<String>,
}

/// The role declarations. Values are not here.
pub struct RoleInputs<'a> {
    /// The model as compiled, before any value is written into it.
    pub model: &'a ir::Model,
    /// The fit config's `[estimate]`; empty without one.
    pub fit_estimate: &'a IndexMap<String, EstimateSpecV2>,
    pub fit_fixed: &'a [(String, FitFixedVia)],
    pub simplex_groups: &'a [SimplexGroup],
    pub cli_estimate: &'a [EstimateDecl],
    /// `--fixed` names.
    pub cli_fixed: &'a [String],
    pub cli_fixed_files: &'a [FixedFileDecl],
    pub design: &'a [(String, DesignOrigin)],
    /// The run is a likelihood slice: nothing estimated, said explicitly.
    pub slice: bool,
    /// Each scenario arm's footprint, checked against the roles (§4.3).
    pub scenarios: &'a [ScenarioFootprint],
}

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RoleError {
    TwoDeclarations { name: String, first: Declaration, second: Declaration },
    /// `constant_hint` is set when `name` is a model constant
    /// ([`ir::Model::constant_hint`]); the message then says how to make it
    /// configurable.
    UnknownName { name: String, by: Declaration, candidates: Vec<String>,
                  constant_hint: Option<String> },
    NotAccepted { name: String, role: RoleKind, command: &'static str },
    Missing { role: RoleKind, command: &'static str },
    ScenarioOnRole { scenario: String, name: String, role: RoleKind },
    SimplexPartlyHeld { group: Vec<String>, held: Vec<String> },
    /// A given search box leaves the model's declared range or, where none is
    /// declared, its kind's support.
    BoxOutsideSupport { message: String },
}

impl std::fmt::Display for RoleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoleError::TwoDeclarations { name, first, second } => {
                let fix = match (first.role_kind(), second.role_kind()) {
                    (RoleKind::Estimated, RoleKind::Fixed) | (RoleKind::Fixed, RoleKind::Estimated)
                        if matches!(first, Declaration::FitEstimate)
                            || matches!(second, Declaration::FitEstimate) =>
                        format!("to hold `{name}`, move it from [estimate] to [fixed] in a copy \
                                 of the fit config; that copy then names the problem it runs"),
                    _ => format!("remove one of the two declarations of `{name}`"),
                };
                write!(f,
                    "parameter `{name}` is declared {} by {} and {} by {}; each parameter \
                     has one role. Fix: {fix}.",
                    first.role_kind().word(), first.describe(),
                    second.role_kind().word(), second.describe())
            }
            RoleError::UnknownName { name, by, candidates, constant_hint } => {
                write!(f, "{} names `{name}`, which is not a parameter of this model",
                    by.describe())?;
                if !candidates.is_empty() {
                    write!(f, "; `{name}` is an indexed family, whose parameters are {}",
                        candidates.join(", "))?;
                }
                if let Some(hint) = constant_hint {
                    write!(f, "\n  hint: {hint}")?;
                }
                Ok(())
            }
            RoleError::NotAccepted { name, role, command } => write!(f,
                "`{command}` does not take a {} parameter, and `{name}` is declared one",
                role.word()),
            RoleError::Missing { role, command } => write!(f,
                "`{command}` needs at least one {} parameter, and none is declared",
                role.word()),
            RoleError::ScenarioOnRole { scenario, name, role } => write!(f,
                "scenario `{scenario}` sets or scales `{name}`, which this run declares {}; \
                 a scenario gives values to fixed parameters only",
                role.word()),
            RoleError::BoxOutsideSupport { message } => write!(f, "{message}"),
            RoleError::SimplexPartlyHeld { group, held } => write!(f,
                "simplex group [{}] is partly held ({} not estimated); the free members \
                 would have to sum to one minus the held ones, which a simplex group cannot \
                 express. Estimate every member, or hold every member",
                group.join(", "), held.join(", ")),
        }
    }
}

// ─── The resolver ────────────────────────────────────────────────────────────

/// One declaration of one name, in input order.
struct Decl<'a> {
    name: &'a str,
    by: Declaration,
    /// An entry of a fit artifact passed as `--fixed-file` (D10).
    fit_artifact: bool,
    /// `--estimate NAME=LO:HI`'s bounds.
    cli_bounds: Option<(f64, f64)>,
}

/// Resolve the run's partition from its role declarations
/// (proposal §4): one declaration per parameter, the special cases of
/// [`SpecialCase`], and errors otherwise; then the command's accepted roles,
/// each scenario arm against the roles, and the simplex groups.
pub fn resolve_roles<'a>(inputs: &'a RoleInputs<'a>, command: &CommandRoles)
    -> Result<Partition, RoleError>
{
    let model = inputs.model;
    let configurable: Vec<&ir::parameter::Parameter> = model.parameters.iter().collect();

    // Every declaration, in input order.
    let decl = |name: &'a str, by: Declaration| Decl { name, by, fit_artifact: false, cli_bounds: None };
    let mut decls: Vec<Decl<'a>> = Vec::new();
    decls.extend(inputs.fit_estimate.keys().map(|n| decl(n, Declaration::FitEstimate)));
    decls.extend(inputs.cli_estimate.iter().map(|e| Decl {
        cli_bounds: e.bounds, ..decl(&e.name, Declaration::CliEstimate)
    }));
    decls.extend(inputs.fit_fixed.iter().map(|(n, via)| decl(n, Declaration::FitFixed(via.clone()))));
    decls.extend(inputs.cli_fixed.iter().map(|n| decl(n, Declaration::CliFixed)));
    for file in inputs.cli_fixed_files {
        decls.extend(file.names.iter().map(|n| Decl {
            fit_artifact: file.fit_artifact,
            ..decl(n, Declaration::CliFixedFile(file.path.clone()))
        }));
    }
    decls.extend(inputs.design.iter().map(|(n, origin)| decl(n, Declaration::Design(*origin))));

    // Every declared name is a parameter.
    for d in &decls {
        if !configurable.iter().any(|p| p.name == d.name) {
            let prefix = format!("{}_", d.name);
            return Err(RoleError::UnknownName {
                name: d.name.to_string(),
                by: d.by.clone(),
                candidates: configurable.iter()
                    .filter(|p| p.name.starts_with(&prefix))
                    .map(|p| p.name.clone())
                    .collect(),
                constant_hint: inputs.model.constant_hint(d.name),
            });
        }
    }

    // One role per parameter.
    let mut entries: IndexMap<String, RoleEntry> = IndexMap::new();
    let mut unused_file_entries: Vec<(PathBuf, String)> = Vec::new();
    for p in &configurable {
        let mine: Vec<&Decl<'_>> = decls.iter().filter(|d| d.name == p.name).collect();
        let est: Vec<&Decl<'_>> =
            mine.iter().copied().filter(|d| d.by.role_kind() == RoleKind::Estimated).collect();
        let des: Vec<&Decl<'_>> =
            mine.iter().copied().filter(|d| d.by.role_kind() == RoleKind::Design).collect();
        let held: Vec<&Decl<'_>> = mine.iter().copied()
            .filter(|d| d.by.role_kind() == RoleKind::Fixed && !d.fit_artifact)
            .collect();
        let artifact: Vec<&Decl<'_>> = mine.iter().copied().filter(|d| d.fit_artifact).collect();
        let two = |a: &Decl<'_>, b: &Decl<'_>| RoleError::TwoDeclarations {
            name: p.name.clone(), first: a.by.clone(), second: b.by.clone(),
        };
        if est.len() > 1 {
            return Err(two(est[0], est[1]));
        }
        if des.len() > 1 {
            return Err(two(des[0], des[1]));
        }
        let entry = match (des.first(), est.first()) {
            (Some(d), Some(e)) => {
                // Case 1: profiling a fit config's estimated parameter.
                if !(command.profiles && e.by == Declaration::FitEstimate) {
                    return Err(two(e, d));
                }
                if let Some(h) = held.first() {
                    return Err(two(h, d));
                }
                unused_file_entries.extend(artifact.iter().map(|a| (file_of(&a.by), p.name.clone())));
                RoleEntry { role: Role::Design, declared_by: d.by.clone(),
                            special: Some(SpecialCase::ProfileFocal) }
            }
            (Some(d), None) => {
                // Case 2: sweeping a held parameter — a `[fixed]` entry only.
                let special = match held.as_slice() {
                    [] => None,
                    [h] if matches!(h.by, Declaration::FitFixed(_)) =>
                        Some(SpecialCase::SweepOverFixed),
                    [h, ..] => {
                        let other = held.iter()
                            .find(|h| !matches!(h.by, Declaration::FitFixed(_)))
                            .unwrap_or(h);
                        return Err(two(other, d));
                    }
                };
                unused_file_entries.extend(artifact.iter().map(|a| (file_of(&a.by), p.name.clone())));
                RoleEntry { role: Role::Design, declared_by: d.by.clone(), special }
            }
            (None, Some(e)) => {
                if let Some(h) = held.first() {
                    return Err(two(e, h));
                }
                unused_file_entries.extend(artifact.iter().map(|a| (file_of(&a.by), p.name.clone())));
                let spec = match e.by {
                    Declaration::FitEstimate => inputs.fit_estimate.get(&p.name),
                    _ => None,
                };
                RoleEntry {
                    role: Role::Estimated(Box::new(estimated(p, spec, e.cli_bounds, inputs)?)),
                    declared_by: e.by.clone(),
                    special: None,
                }
            }
            (None, None) => {
                // Held: by its highest-tier fixed declaration (a fit
                // artifact's entry is one here), else by nothing (D3).
                let by = mine.iter()
                    .max_by_key(|d| d.by.fixed_tier())
                    .map(|d| d.by.clone())
                    .unwrap_or(Declaration::Default);
                RoleEntry { role: Role::Fixed, declared_by: by, special: None }
            }
        };
        entries.insert(p.name.clone(), entry);
    }

    // The command's accepted roles.
    for (kind, allowed) in [(RoleKind::Estimated, command.estimated), (RoleKind::Design, command.design)] {
        let mut names = entries.iter().filter(|(_, e)| e.role.kind() == kind).map(|(n, _)| n);
        let first = names.next();
        match (allowed, first) {
            (Allowed::No, Some(name)) => return Err(RoleError::NotAccepted {
                name: name.clone(), role: kind, command: command.command,
            }),
            (Allowed::AtLeastOne, None) => return Err(RoleError::Missing {
                role: kind, command: command.command,
            }),
            (Allowed::AtLeastOneUnlessSlice, None) if !inputs.slice => {
                return Err(RoleError::Missing { role: kind, command: command.command });
            }
            _ => {}
        }
    }

    // A scenario gives values to fixed parameters only (§4.3).
    for arm in inputs.scenarios {
        for name in &arm.params {
            if let Some(e) = entries.get(name) {
                match e.role.kind() {
                    RoleKind::Fixed => {}
                    role => return Err(RoleError::ScenarioOnRole {
                        scenario: arm.scenario.clone(), name: name.clone(), role,
                    }),
                }
            }
        }
    }

    // Simplex groups: all members one role.
    let mut simplex_groups = Vec::new();
    for g in inputs.simplex_groups {
        let mut held = Vec::new();
        for m in &g.params {
            match entries.get(m) {
                Some(e) if e.role.kind() == RoleKind::Estimated => {}
                Some(_) => held.push(m.clone()),
                None => return Err(RoleError::UnknownName {
                    name: m.clone(), by: Declaration::FitEstimate, candidates: Vec::new(),
                    constant_hint: inputs.model.constant_hint(m),
                }),
            }
        }
        let role = if held.is_empty() {
            SimplexRole::Estimated
        } else if held.len() == g.params.len() {
            SimplexRole::Held
        } else {
            return Err(RoleError::SimplexPartlyHeld { group: g.params.clone(), held });
        };
        simplex_groups.push(ResolvedSimplex { members: g.params.clone(), role });
    }

    let mut estimate_order: Vec<String> = Vec::new();
    for d in &decls {
        let estimated = entries.get(d.name).is_some_and(|e| e.role.kind() == RoleKind::Estimated);
        if estimated && d.by.role_kind() == RoleKind::Estimated
            && !estimate_order.iter().any(|n| n == d.name)
        {
            estimate_order.push(d.name.to_string());
        }
    }
    Ok(Partition { entries, simplex_groups, unused_file_entries, estimate_order })
}

fn file_of(by: &Declaration) -> PathBuf {
    match by {
        Declaration::CliFixedFile(p) => p.clone(),
        _ => PathBuf::new(),
    }
}

/// The model's admissible values for `p`: its declared `in [lo, hi]`, else
/// its kind's range — `probability` `[0, 1]`; `rate`, `positive`, `count`,
/// `duration` `[0, ∞)`; `real`, `instant` and an unannotated parameter
/// `(−∞, ∞)`.
pub fn support(p: &ir::parameter::Parameter) -> Support {
    if let Some((lower, upper)) = p.bounds() {
        return Support { lower, upper, from: SupportFrom::Declared };
    }
    let (lower, upper) = crate::fit::runner::kind_support(p)
        .map_or((f64::NEG_INFINITY, f64::INFINITY), |(range, _)| range);
    Support { lower, upper, from: SupportFrom::Kind }
}

fn estimated(
    p: &ir::parameter::Parameter,
    spec: Option<&EstimateSpecV2>,
    cli_bounds: Option<(f64, f64)>,
    inputs: &RoleInputs<'_>,
) -> Result<Estimated, RoleError> {
    let support = support(p);
    // The box the run searches is the one `resolve_search_box` gives the
    // method, so the record describes what ran.
    let given_box = spec.and_then(|s| s.bounds).or(cli_bounds);
    let (lower, upper) = crate::fit::runner::resolve_search_box(&p.name, given_box, p)
        .map_err(|message| RoleError::BoxOutsideSupport { message })?;
    let from = match (spec.and_then(|s| s.bounds), cli_bounds) {
        (Some(_), _) => BoxFrom::FitConfig,
        (None, Some(_)) => BoxFrom::Cli,
        (None, None) if (lower, upper) == (support.lower, support.upper) => BoxFrom::Support,
        (None, None) => BoxFrom::Fallback,
    };
    let search = SearchBox { lower, upper, from };
    let given = spec.and_then(|s| s.transform.as_ref()).map(|t| t.as_str());
    let transform = crate::fit::runner::derive_transform_with_bounds(
        p, given, (search.lower, search.upper));
    let transform_from = if given.is_some() { TransformFrom::FitConfig } else { TransformFrom::Kind };
    let prior = crate::fit::priors_precedence::resolve_priors_with_precedence(
        std::slice::from_ref(&p.name), inputs.fit_estimate, inputs.model,
    ).remove(0);
    let prior_text = match prior.source {
        PriorSource::FitToml | PriorSource::FlatExplicit => spec
            .and_then(|s| s.prior.as_ref())
            .map(crate::fit::config_diff::format_prior)
            .unwrap_or_else(|| format!("{:?}", prior.prior)),
        PriorSource::ModelIr => crate::fit::priors_precedence::model_prior_text(&p.name, inputs.model)
            .unwrap_or_else(|| format!("{:?}", prior.prior)),
        PriorSource::FlatFallback => "flat()".to_string(),
    };
    Ok(Estimated {
        support,
        search,
        transform,
        transform_from,
        prior,
        prior_text,
        start: spec.and_then(|s| s.start),
    })
}

#[cfg(test)]
mod tests {
    //! One test per row of the proposal's §4.2 and §4.3 tables that a command
    //! resolving roles can reach (the `--draws` rows need the sampled role,
    //! which arrives with `simulate`), plus §4.4's accepted roles, §4.5's
    //! scalar names, D3, the simplex rule and the record.
    use super::*;
    use ir::parameter::{ParamKind, ParamValue, Parameter, PriorDist, PriorSpec};

    fn param(name: &str, kind: ParamKind, bounds: Option<(f64, f64)>) -> Parameter {
        Parameter {
            name: name.into(),
            value: match bounds {
                Some(b) => ParamValue::Estimated {
                    init: None,
                    bounds: Some(b),
                    prior: PriorSpec::Dist(PriorDist::HalfNormal(ir::parameter::HalfNormalPrior { sigma: 10.0 })),
                    transform: ir::parameter::Transform::Identity,
                },
                None => ParamValue::Required,
            },
            param_kind: Some(kind),
            param_dim: None,
        }
    }

    /// The SIR recovery model's parameters, a parameter `delta` with a value
    /// in the IR, an indexed family `c[region]` compiled to `c_north`,
    /// `c_south`, and a model constant `omega` (not a parameter).
    fn model() -> ir::Model {
        let mut m = crate::test_support::tiny_model();
        m.parameters = vec![
            param("beta", ParamKind::Rate, Some((0.05, 1.0))),
            param("gamma", ParamKind::Rate, Some((0.01, 0.5))),
            param("N0", ParamKind::Count, None),
            param("I0", ParamKind::Count, None),
            param("rho", ParamKind::Probability, Some((0.1, 0.9))),
            param("k", ParamKind::Positive, Some((1.0, 100.0))),
            param("tau", ParamKind::Real, None),
            param("c_north", ParamKind::Rate, None),
            param("c_south", ParamKind::Rate, None),
            Parameter {
                name: "delta".into(),
                value: ParamValue::Fixed { value: 0.01 },
                param_kind: Some(ParamKind::Rate),
                param_dim: None,
            },
        ];
        m.constants = vec![ir::parameter::Constant {
            name: "omega".into(), value: 0.01, kind: ParamKind::Rate,
        }];
        m
    }

    fn estimate(entries: &[(&str, &str)]) -> IndexMap<String, EstimateSpecV2> {
        entries.iter()
            .map(|(n, spec)| (n.to_string(), toml::from_str(spec).expect("estimate spec")))
            .collect()
    }

    /// The declarations of one run, owned so a test can set what it needs.
    #[derive(Default)]
    struct Run {
        fit_estimate: IndexMap<String, EstimateSpecV2>,
        fit_fixed: Vec<(String, FitFixedVia)>,
        simplex: Vec<SimplexGroup>,
        cli_estimate: Vec<EstimateDecl>,
        cli_fixed: Vec<String>,
        files: Vec<FixedFileDecl>,
        design: Vec<(String, DesignOrigin)>,
        slice: bool,
        scenarios: Vec<ScenarioFootprint>,
    }

    impl Run {
        fn fixed(mut self, names: &[&str]) -> Self {
            self.fit_fixed = names.iter().map(|n| (n.to_string(), FitFixedVia::Inline)).collect();
            self
        }
        fn sweep(mut self, names: &[&str]) -> Self {
            self.design = names.iter().map(|n| (n.to_string(), DesignOrigin::Sweep)).collect();
            self
        }
        fn resolve(&self, command: &CommandRoles) -> Result<Partition, RoleError> {
            let m = model();
            let inputs = RoleInputs {
                model: &m,
                fit_estimate: &self.fit_estimate,
                fit_fixed: &self.fit_fixed,
                simplex_groups: &self.simplex,
                cli_estimate: &self.cli_estimate,
                cli_fixed: &self.cli_fixed,
                cli_fixed_files: &self.files,
                design: &self.design,
                slice: self.slice,
                scenarios: &self.scenarios,
            };
            resolve_roles(&inputs, command)
        }
    }

    /// The SIR recovery fit config: `beta`, `gamma` estimated, the rest in
    /// `[fixed]`; `delta` undeclared, so fixed by default.
    fn sir_fit() -> Run {
        Run {
            fit_estimate: estimate(&[("beta", "bounds = [0.1, 0.9]"), ("gamma", "")]),
            ..Run::default()
        }
        .fixed(&["N0", "I0", "rho", "k", "tau", "c_north", "c_south"])
    }

    fn kind(p: &Partition, name: &str) -> RoleKind {
        p.entries[name].role.kind()
    }

    fn two(e: RoleError) -> (String, Declaration, Declaration) {
        match e {
            RoleError::TwoDeclarations { name, first, second } => (name, first, second),
            other => panic!("expected TwoDeclarations, got {other:?}"),
        }
    }

    // ── §4.2 ────────────────────────────────────────────────────────────────

    #[test]
    fn estimate_and_fixed_in_one_file_is_refused() {
        let run = Run { fit_estimate: estimate(&[("beta", "")]), ..Run::default() }
            .fixed(&["beta", "gamma"]);
        let (name, first, second) = two(run.resolve(&FIT_RUN).unwrap_err());
        assert_eq!(name, "beta");
        assert_eq!(first, Declaration::FitEstimate);
        assert_eq!(second, Declaration::FitFixed(FitFixedVia::Inline));
    }

    #[test]
    fn profiling_an_estimated_parameter_makes_it_focal() {
        let p = sir_fit().sweep(&["beta"]).resolve(&PROFILE).unwrap();
        let beta = &p.entries["beta"];
        assert_eq!(beta.role.kind(), RoleKind::Design);
        assert_eq!(beta.special, Some(SpecialCase::ProfileFocal));
        assert_eq!(beta.declared_by, Declaration::Design(DesignOrigin::Sweep));
        assert_eq!(kind(&p, "gamma"), RoleKind::Estimated);
        assert_eq!(p.notices().len(), 1, "{:?}", p.notices());
    }

    #[test]
    fn sweeping_an_estimated_parameter_outside_profile_is_refused() {
        let (name, first, second) = two(sir_fit().sweep(&["beta"]).resolve(&FIT_RUN).unwrap_err());
        assert_eq!((name.as_str(), first, second),
            ("beta", Declaration::FitEstimate, Declaration::Design(DesignOrigin::Sweep)));
    }

    #[test]
    fn sweeping_a_fixed_entry_is_design_at_job_level() {
        for command in [&FIT_RUN, &PROFILE] {
            let p = sir_fit().sweep(&["N0"]).resolve(command).unwrap();
            let n0 = &p.entries["N0"];
            assert_eq!(n0.role.kind(), RoleKind::Design, "{}", command.command);
            assert_eq!(n0.special, Some(SpecialCase::SweepOverFixed), "{}", command.command);
        }
    }

    #[test]
    fn fixed_cli_and_sweep_on_one_name_is_refused() {
        let mut run = sir_fit().sweep(&["N0"]);
        run.cli_fixed = vec!["N0".into()];
        let (name, first, second) = two(run.resolve(&FIT_RUN).unwrap_err());
        assert_eq!((name.as_str(), first, second),
            ("N0", Declaration::CliFixed, Declaration::Design(DesignOrigin::Sweep)));
    }

    #[test]
    fn estimate_cli_and_fixed_cli_on_one_name_is_refused() {
        let run = Run {
            cli_estimate: vec![EstimateDecl { name: "gamma".into(), bounds: None }],
            cli_fixed: vec!["gamma".into()],
            ..Run::default()
        };
        let (name, first, second) = two(run.resolve(&SURVEY).unwrap_err());
        assert_eq!((name.as_str(), first, second),
            ("gamma", Declaration::CliEstimate, Declaration::CliFixed));
    }

    #[test]
    fn fixed_cli_over_estimate_is_refused_naming_the_fix() {
        let mut run = sir_fit();
        run.fit_estimate.insert("rho".into(), toml::from_str("").unwrap());
        run.fit_fixed.retain(|(n, _)| n != "rho");
        run.cli_fixed = vec!["rho".into()];
        let err = run.resolve(&SURVEY).unwrap_err();
        assert!(err.to_string().contains("move it from [estimate] to [fixed]"), "{err}");
        let (name, first, second) = two(err);
        assert_eq!((name.as_str(), first, second),
            ("rho", Declaration::FitEstimate, Declaration::CliFixed));
    }

    #[test]
    fn a_fit_artifacts_entry_for_a_swept_name_is_unused_and_listed() {
        let mut run = sir_fit().sweep(&["beta"]);
        run.files = vec![FixedFileDecl {
            path: "mle_params.toml".into(),
            names: vec!["beta".into(), "gamma".into(), "k".into()],
            fit_artifact: true,
        }];
        let p = run.resolve(&PROFILE).unwrap();
        assert_eq!(kind(&p, "beta"), RoleKind::Design);
        assert_eq!(kind(&p, "gamma"), RoleKind::Estimated);
        // Its entry for a name with no other role holds that name.
        assert_eq!(p.entries["k"].declared_by, Declaration::CliFixedFile("mle_params.toml".into()));
        assert_eq!(p.unused_file_entries, vec![
            (PathBuf::from("mle_params.toml"), "beta".to_string()),
            (PathBuf::from("mle_params.toml"), "gamma".to_string()),
        ]);
        assert_eq!(p.notices().iter().filter(|n| n.contains("mle_params.toml")).count(), 2);
    }

    #[test]
    fn a_plain_files_entry_for_a_swept_name_is_refused() {
        let mut run = Run {
            cli_estimate: vec![EstimateDecl { name: "gamma".into(), bounds: None }],
            ..Run::default()
        }.sweep(&["beta"]);
        run.files = vec![FixedFileDecl {
            path: "p.toml".into(), names: vec!["beta".into()], fit_artifact: false,
        }];
        let (name, first, second) = two(run.resolve(&PROFILE).unwrap_err());
        assert_eq!((name.as_str(), first, second),
            ("beta", Declaration::CliFixedFile("p.toml".into()), Declaration::Design(DesignOrigin::Sweep)));
    }

    #[test]
    fn a_plain_files_entry_for_an_estimated_name_is_refused() {
        let run = Run {
            cli_estimate: vec![EstimateDecl { name: "gamma".into(), bounds: None }],
            files: vec![FixedFileDecl {
                path: "p.toml".into(), names: vec!["gamma".into()], fit_artifact: false,
            }],
            ..Run::default()
        };
        assert_eq!(two(run.resolve(&SURVEY).unwrap_err()).0, "gamma");
    }

    #[test]
    fn two_fixed_declarations_are_tiers_not_a_conflict() {
        let mut run = sir_fit();
        run.cli_fixed = vec!["k".into()];
        let p = run.resolve(&FIT_RUN).unwrap();
        assert_eq!(kind(&p, "k"), RoleKind::Fixed);
        assert_eq!(p.entries["k"].declared_by, Declaration::CliFixed,
            "the record names the declaration whose value wins");
    }

    // ── §4.3 ────────────────────────────────────────────────────────────────

    fn under(run: Run, param: &str) -> Run {
        Run {
            scenarios: vec![ScenarioFootprint {
                scenario: "baseline".into(), params: vec![param.to_string()],
            }],
            ..run
        }
    }

    #[test]
    fn a_scenario_on_a_fixed_parameter_is_a_value() {
        assert!(under(sir_fit(), "k").resolve(&FIT_RUN).is_ok());
    }

    #[test]
    fn a_scenario_on_an_estimated_parameter_is_refused() {
        match under(sir_fit(), "gamma").resolve(&FIT_RUN).unwrap_err() {
            RoleError::ScenarioOnRole { scenario, name, role } => {
                assert_eq!((scenario.as_str(), name.as_str(), role),
                    ("baseline", "gamma", RoleKind::Estimated));
            }
            other => panic!("expected ScenarioOnRole, got {other:?}"),
        }
    }

    #[test]
    fn a_scenario_on_a_design_parameter_is_refused() {
        match under(sir_fit().sweep(&["N0"]), "N0").resolve(&FIT_RUN).unwrap_err() {
            RoleError::ScenarioOnRole { role, .. } => assert_eq!(role, RoleKind::Design),
            other => panic!("expected ScenarioOnRole, got {other:?}"),
        }
    }

    // ── §4.4 and §4.5 ───────────────────────────────────────────────────────

    #[test]
    fn a_role_the_command_does_not_take_is_refused() {
        match sir_fit().sweep(&["N0"]).resolve(&SURVEY).unwrap_err() {
            RoleError::NotAccepted { name, role, command } => {
                assert_eq!((name.as_str(), role, command), ("N0", RoleKind::Design, "survey"));
            }
            other => panic!("expected NotAccepted, got {other:?}"),
        }
    }

    #[test]
    fn a_profile_with_nothing_estimated_needs_slice() {
        let run = Run::default().sweep(&["beta"]);
        assert_eq!(run.resolve(&PROFILE).unwrap_err(),
            RoleError::Missing { role: RoleKind::Estimated, command: "profile" });
        let slice = Run { slice: true, ..run };
        assert!(slice.resolve(&PROFILE).is_ok());
        assert_eq!(Run::default().resolve(&SURVEY).unwrap_err(),
            RoleError::Missing { role: RoleKind::Estimated, command: "survey" });
    }

    #[test]
    fn a_family_name_is_unknown_and_its_cells_are_listed() {
        let run = Run { cli_fixed: vec!["c".into()], ..sir_fit() };
        match run.resolve(&FIT_RUN).unwrap_err() {
            RoleError::UnknownName { name, candidates, .. } => {
                assert_eq!(name, "c");
                assert_eq!(candidates, vec!["c_north", "c_south"]);
            }
            other => panic!("expected UnknownName, got {other:?}"),
        }
    }

    /// D2/D9: a model constant is not a parameter, so naming one in any
    /// declaration is the unknown-name refusal, and its message says how to
    /// make it configurable.
    #[test]
    fn a_model_constant_is_an_unknown_name_with_a_hint() {
        let runs = [
            Run { cli_fixed: vec!["omega".into()], ..sir_fit() },
            Run { cli_estimate: vec![EstimateDecl { name: "omega".into(), bounds: None }],
                  ..Run::default() },
            sir_fit().sweep(&["omega"]),
        ];
        for run in runs {
            let err = run.resolve(&PROFILE).unwrap_err();
            match &err {
                RoleError::UnknownName { name, constant_hint: Some(_), .. } =>
                    assert_eq!(name, "omega"),
                other => panic!("expected UnknownName with a constant hint, got {other:?}"),
            }
            let msg = err.to_string();
            assert!(msg.contains("`omega` is a constant of the model (`let omega : rate = 0.01`)")
                && msg.contains("declare it in `parameters {}` to set it from outside"),
                "the hint must name the constant and the fix: {msg}");
        }
        // A misspelt parameter gets no constant hint.
        let err = Run { cli_fixed: vec!["omegaa".into()], ..sir_fit() }
            .resolve(&FIT_RUN).unwrap_err();
        assert!(matches!(err, RoleError::UnknownName { constant_hint: None, .. }), "{err:?}");
    }

    // ── D3, the support and the simplex rule ────────────────────────────────

    #[test]
    fn an_undeclared_parameter_is_fixed_by_default() {
        let run = Run {
            cli_estimate: vec![EstimateDecl { name: "beta".into(), bounds: Some((0.1, 0.5)) }],
            ..Run::default()
        };
        let p = run.resolve(&SURVEY).unwrap();
        assert_eq!(kind(&p, "gamma"), RoleKind::Fixed);
        assert_eq!(p.entries["gamma"].declared_by, Declaration::Default);
        assert_eq!(p.names_with(RoleKind::Estimated).collect::<Vec<_>>(), vec!["beta"]);
        assert_eq!(p.all_names().count(), 10, "every parameter");
    }

    #[test]
    fn support_box_and_transform_are_kept_apart() {
        let mut run = sir_fit();
        run.fit_estimate.insert("tau".into(), toml::from_str("").unwrap());
        run.fit_fixed.retain(|(n, _)| n != "tau");
        let p = run.resolve(&FIT_RUN).unwrap();
        let Role::Estimated(beta) = &p.entries["beta"].role else { panic!("beta estimated") };
        assert_eq!(beta.support, Support { lower: 0.05, upper: 1.0, from: SupportFrom::Declared });
        assert_eq!(beta.search, SearchBox { lower: 0.1, upper: 0.9, from: BoxFrom::FitConfig });
        assert!(matches!(beta.transform, Transform::Log { lo, hi } if lo == 0.1 && hi == 0.9));
        let Role::Estimated(tau) = &p.entries["tau"].role else { panic!("tau estimated") };
        assert_eq!(tau.support, Support {
            lower: f64::NEG_INFINITY, upper: f64::INFINITY, from: SupportFrom::Kind });
        // An unbounded `real` searches its whole support: nothing would
        // enforce a `[0, ∞)` box under the identity transform, so the record
        // must not claim one.
        assert_eq!(tau.search, SearchBox {
            lower: f64::NEG_INFINITY, upper: f64::INFINITY, from: BoxFrom::Support });
        assert!(matches!(tau.transform, Transform::None));
    }

    #[test]
    fn a_box_leaving_the_declared_range_is_refused_as_the_run_refuses_it() {
        let run = Run { fit_estimate: estimate(&[("beta", "bounds = [0.01, 2.0]")]), ..sir_fit() };
        match run.resolve(&FIT_RUN).unwrap_err() {
            RoleError::BoxOutsideSupport { message } => {
                assert!(message.contains("tighten bounds but not loosen"), "{message}");
            }
            other => panic!("expected BoxOutsideSupport, got {other:?}"),
        }
    }

    #[test]
    fn a_partly_held_simplex_is_refused() {
        let mut run = sir_fit();
        run.simplex = vec![SimplexGroup { params: vec!["beta".into(), "rho".into()] }];
        assert_eq!(run.resolve(&FIT_RUN).unwrap_err(), RoleError::SimplexPartlyHeld {
            group: vec!["beta".into(), "rho".into()], held: vec!["rho".into()],
        });
        run.simplex = vec![SimplexGroup { params: vec!["k".into(), "rho".into()] }];
        assert_eq!(run.resolve(&FIT_RUN).unwrap().simplex_groups[0].role, SimplexRole::Held);
        run.simplex = vec![SimplexGroup { params: vec!["beta".into(), "gamma".into()] }];
        assert_eq!(run.resolve(&FIT_RUN).unwrap().simplex_groups[0].role, SimplexRole::Estimated);
    }

    // ── The record ──────────────────────────────────────────────────────────

    fn values(pairs: &[(&str, f64)]) -> ResolvedParameters {
        use crate::params_resolver::{FixReason, ParameterRole, ResolvedParameter, ValueSource};
        ResolvedParameters {
            params: pairs.iter().map(|(n, v)| ResolvedParameter {
                name: n.to_string(),
                value: *v,
                source: ValueSource::FitTomlFixed,
                role: ParameterRole::Fixed { reason: FixReason::NotInEstimate },
                overrides: vec![],
            }).collect(),
            estimate_set: Default::default(),
            model: model(),
            pinned_over_estimate: vec![],
            context: Default::default(),
        }
    }

    #[test]
    fn legacy_fields_are_generated_from_the_record() {
        let p = sir_fit().sweep(&["N0"]).resolve(&FIT_RUN).unwrap();
        let vals = values(&[("N0", 1.0), ("I0", 10.0), ("rho", 0.6), ("k", 10.0), ("tau", 0.0),
                            ("c_north", 1.0), ("c_south", 2.0), ("delta", 0.01)]);
        // The job: N0 is a design coordinate, with no single value.
        let (est, fixed) = p.record(&vals).legacy_fields();
        assert_eq!(est, vec!["beta", "gamma"]);
        assert!(!fixed.contains_key("N0"));
        assert_eq!(fixed.get("delta"), Some(&0.01), "an undeclared parameter is held");
        assert!(!fixed.contains_key("omega"), "a model constant is in no partition");
        // A cell: N0 is fixed at its grid value.
        let point = vec![("N0".to_string(), 5000.0)];
        let rec = p.cell(&point).unwrap().record(&vals);
        assert_eq!(rec.legacy_fields().1.get("N0"), Some(&5000.0));
        match &rec.parameters["N0"] {
            ParamRecord::Fixed { job_role, special, value_from, .. } => {
                assert_eq!(*job_role, Some(RoleKind::Design));
                assert_eq!(*special, Some(SpecialCase::SweepOverFixed));
                assert_eq!(value_from.as_ref().unwrap().source, "design_point");
            }
            other => panic!("N0 in a cell is fixed, got {other:?}"),
        }
        assert!(p.cell(&[]).is_err(), "a cell gives every design parameter a value");
    }

    #[test]
    fn the_legacy_estimated_list_keeps_the_declaration_order() {
        let run = Run {
            fit_estimate: estimate(&[("gamma", ""), ("beta", "")]),
            ..Run::default()
        }.fixed(&["N0", "I0", "rho", "k", "tau", "c_north", "c_south"]);
        let rec = run.resolve(&FIT_RUN).unwrap().record(&values(&[]));
        assert_eq!(rec.legacy_fields().0, vec!["gamma", "beta"], "the columns' order");
        let in_record: Vec<&str> = rec.parameters.keys().map(String::as_str).take(2).collect();
        assert_eq!(in_record, vec!["beta", "gamma"], "the record keeps model order");
    }

    #[test]
    fn the_record_round_trips_with_infinite_bounds() {
        let mut run = sir_fit();
        run.fit_estimate.insert("tau".into(), toml::from_str("").unwrap());
        run.fit_fixed.retain(|(n, _)| n != "tau");
        let rec = run.resolve(&FIT_RUN).unwrap().record(&values(&[("N0", 1.0)]));
        let field = PartitionField::Resolved(rec);
        let json = serde_json::to_value(&field).unwrap();
        assert_eq!(json["status"], "resolved");
        assert_eq!(json["parameters"]["tau"]["support"], serde_json::json!(["-inf", "inf"]));
        assert_eq!(json["parameters"]["tau"]["search"], serde_json::json!(["-inf", "inf"]));
        assert_eq!(json["parameters"]["tau"]["search_from"], "support");
        assert_eq!(json["parameters"]["beta"]["declared_by"], "fit_estimate");
        assert_eq!(json["parameters"]["N0"]["via"], "inline");
        let back: PartitionField = serde_json::from_value(json).unwrap();
        // The declaration order is not stored; everything else round-trips.
        let PartitionField::Resolved(mut rec) = field else { unreachable!() };
        rec.estimate_order.clear();
        assert_eq!(back, PartitionField::Resolved(rec));
    }
}
