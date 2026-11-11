//! The typed record of what a fit method's run shows.
//!
//! Every fact the end of `fit run` prints is a value here before it is text,
//! and is recorded in the method leaf so `fit summary` can rebuild it from the
//! files (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`, "The
//! report"). This module holds the record types; each says where it is stored.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The run's opening block, as `fit run` prints it under `fit:`. Recorded in
/// `run.json`'s `inputs.header` (display inputs, never hashed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunHeader {
    /// The config file as given on the command line.
    pub config: String,
    /// The method's name (`pmmh`, `nl-sbplx`, …).
    pub method: String,
    /// The model path as the config writes it.
    pub model: String,
    /// Estimated parameters, in the config's order.
    pub estimate: Vec<String>,
    /// Fixed parameters, in the order the config resolves them.
    pub fixed: Vec<String>,
    /// The starts rule and why it applies, as the `starts:` line reads
    /// (`StartsResolution::describe`).
    pub starts: String,
    /// The fit directory, relative to the working directory the run was
    /// started from when it is under it ([`display_path`]).
    pub output: String,
}

/// A path for display: relative to the working directory when it is under
/// it, otherwise absolute. Relative is what a reader can paste back into a
/// command from where they stand; a path outside the working directory is
/// absolute so it cannot be read against the wrong base (gh#507).
pub fn display_path(p: &Path) -> String {
    let abs = crate::run_paths::display_absolute(p);
    match std::env::current_dir() {
        Ok(cwd) => match abs.strip_prefix(&cwd) {
            Ok(rel) if !rel.as_os_str().is_empty() => rel.display().to_string(),
            _ => abs.display().to_string(),
        },
        Err(_) => abs.display().to_string(),
    }
}

/// One chain's adaptive warm-up, as it ended. Recorded in
/// `pmmh_summary.json` (or `mh_summary.json`) under `warmup`; empty when
/// `adapt = false`, where nothing adapts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainWarmup {
    /// The chain's number, as its `chain_N/` directory carries it.
    pub chain: usize,
    /// Steps on which the Robbins–Monro scale adapted.
    pub warmup_steps: usize,
    /// Of those, steps on which it sat at its lower bound.
    pub steps_at_scale_floor: usize,
    /// The scale λ the kept draws were proposed under.
    #[serde(with = "sim::inference::json_f64")]
    pub final_scale: f64,
    /// The acceptance rate the adaptation chased.
    #[serde(with = "sim::inference::json_f64")]
    pub target_accept: f64,
}

impl ChainWarmup {
    /// The record of a chain resumed from `base`: the resumed call adapts only
    /// over whatever warm-up `base` had left, so its steps add to the base's
    /// and the scale it ends at is the one its draws were taken under.
    pub fn continued_from(base: &ChainWarmup, this_call: &ChainWarmup) -> ChainWarmup {
        ChainWarmup {
            chain: this_call.chain,
            warmup_steps: base.warmup_steps + this_call.warmup_steps,
            steps_at_scale_floor: base.steps_at_scale_floor + this_call.steps_at_scale_floor,
            final_scale: this_call.final_scale,
            target_accept: this_call.target_accept,
        }
    }
}

/// How a PMMH or `mh` method sampled. Recorded in its summary JSON under
/// `plan`; the same values the `pmmh:` line prints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingPlan {
    pub n_chains: usize,
    pub n_steps: usize,
    /// Particles per likelihood evaluation; 0 on the deterministic `mh` path.
    pub n_particles: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub adapt: bool,
    /// Observation times the likelihood sums over (the `T` of the PMMH
    /// caveat). 0 in a summary written before it was recorded.
    #[serde(default)]
    pub series_len: usize,
    /// The initial proposal SD per estimated parameter, on the transformed
    /// scale.
    pub proposal_sd: BTreeMap<String, f64>,
}

/// What a PMMH or `mh` method's sampling cost. Recorded in its summary JSON
/// under `cost`; the values the `complete in` line prints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingCost {
    /// Wall time of the chains, start of the first to end of the last.
    #[serde(with = "sim::inference::json_f64")]
    pub sampling_seconds: f64,
    /// Likelihood evaluations this call ran, as the completion line counts
    /// them: one per MCMC step, so a resumed run counts only its own steps.
    pub evaluations: usize,
    /// Wall time per evaluation within one chain, the chains running in
    /// parallel.
    #[serde(with = "sim::inference::json_f64")]
    pub ms_per_evaluation: f64,
}

/// The `warmup` record a method's summary JSON holds, or empty when the file
/// or the field is absent (a leaf written before it was recorded).
pub fn read_warmup(summary_path: &Path) -> Vec<ChainWarmup> {
    #[derive(Deserialize)]
    struct WithWarmup {
        #[serde(default)]
        warmup: Vec<ChainWarmup>,
    }
    std::fs::read_to_string(summary_path)
        .ok()
        .and_then(|t| serde_json::from_str::<WithWarmup>(&t).ok())
        .map(|w| w.warmup)
        .unwrap_or_default()
}

/// A check made before sampling, shown in the run's opening section. Checks
/// are not findings: the findings count does not count them.
///
/// Derived, not stored twice: `FilterNoise` from `fit_state.toml`'s
/// `pf_noise`, `ObsStream` from the model, `MethodCaveat` from the method
/// registry, `Starts` from the run's header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    /// One observation stream the fit scores: its name, what it projects
    /// (`incidence(infection)`), and its likelihood family.
    ObsStream { name: String, carrier: String, family: String },
    /// The particle-filter noise measurement before PMMH samples. The record
    /// carries its own `log L̂` mean (`PfNoiseCheck::ll_mean`).
    FilterNoise { check: crate::fit::pf_noise::PfNoiseCheck },
    /// The method registry's caveat for this method and backend, and the
    /// series length it is said with, when the run recorded it.
    MethodCaveat { method: String, note: String, series_len: Option<usize> },
    /// A starts rule that discards the declared `[estimate].start` values of
    /// `params`, with the note the run prints.
    Starts { mode: String, params: Vec<String>, note: String },
}

/// Whether a projection reads the state at an instant (a prevalence or
/// snapshot) rather than accumulating a flow.
pub fn projection_is_snapshot(p: &ir::observation::Projection) -> bool {
    use ir::observation::Projection as P;
    match p {
        P::CumulativeFlow(_) | P::CumulativeFlowSum(_) | P::FlowRatio { .. } => false,
        P::CurrentPop(_) | P::CurrentPopSum(_) | P::DerivedExpr(_) => true,
    }
}

/// One `ObsStream` check per observation stream the model scores, in
/// declaration order: what the run's `observations (…)` block lists.
pub fn obs_stream_checks(model: &ir::Model) -> Vec<Check> {
    use ir::observation::{Likelihood as L, Projection as P};
    model.observations.iter().map(|obs| {
        let carrier = match &obs.projection {
            P::CumulativeFlow(name) => format!("incidence({})", name),
            P::CumulativeFlowSum(names) => format!("incidence({})", names.join(" + ")),
            P::FlowRatio { numerator, denominator } => format!("incidence({}) / incidence({})",
                numerator.join(" + "), denominator.join(" + ")),
            P::CurrentPop(name) => name.to_string(),
            P::CurrentPopSum(names) => names.join(" + "),
            P::DerivedExpr(_) => "derived expression".to_string(),
        };
        let family = match &obs.likelihood {
            L::NegBinomial(_) => "NegBinomial",
            L::Poisson(_) => "Poisson",
            L::Normal(_) => "Normal",
            L::Binomial(_) => "Binomial",
            L::BetaBinomial(_) => "BetaBinomial",
            L::Beta(_) => "Beta",
            L::Bernoulli(_) => "Bernoulli",
            L::ZeroInflatedNegBinomial(_) => "ZeroInflatedNegBinomial",
        };
        Check::ObsStream { name: obs.name.clone(), carrier, family: family.to_string() }
    }).collect()
}

/// The verdict on a method, decided by its R̂ alone ([`verdict`]). Serialized
/// with the report so a consumer reads it rather than recomputing it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    Converged { max_rhat: f64, param: Option<String> },
    NotConverged { max_rhat: f64, param: Option<String>, band: crate::fit::method_result::RhatBand },
    /// These chains took no step after burn-in (gh#1032): not converged,
    /// whatever any R̂ says.
    ChainsFrozen { chains: Vec<crate::chain_index::ChainIndex> },
    /// R̂ could not be computed for these parameters for a reason that is a
    /// problem: not converged, with no number to show.
    Unassessable { params: Vec<String> },
    /// R̂ was never computable for a structural reason (too few chains, a
    /// point start): not assessed.
    NotAssessed { reason: String },
    /// IF2, NLopt and pfilter: no chains to compare.
    NotApplicable,
}

/// The verdict on `result`, from its max R̂ and the band `fit summary`
/// glyphs it in (`RhatBand`).
pub fn verdict(result: &crate::fit::method_result::MethodResult) -> Verdict {
    use crate::fit::method_result::{MaxRhat, MethodResult, RhatBand};
    let diag = match result {
        MethodResult::Pgas(r) => &r.diagnostics,
        MethodResult::Pmmh(r) => &r.diagnostics,
        MethodResult::If2(_) | MethodResult::Nlopt(_) | MethodResult::Pfilter(_) => {
            return Verdict::NotApplicable;
        }
    };
    let param = diag.per_param.iter()
        .filter_map(|(n, p)| p.rhat().finite().map(|r| (n, r)))
        .chain(diag.per_param.iter()
            .filter(|(_, p)| matches!(p.rhat(), crate::fit::method_result::Stat::Infinite))
            .map(|(n, _)| (n, f64::INFINITY)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(n, _)| n.clone());
    match diag.max_rhat_status() {
        MaxRhat::Reported(v) => match RhatBand::of(v) {
            RhatBand::Converged => Verdict::Converged { max_rhat: v, param },
            band => Verdict::NotConverged { max_rhat: v, param, band },
        },
        MaxRhat::ChainsFrozen { chains } => Verdict::ChainsFrozen { chains },
        MaxRhat::Unassessable { params } => Verdict::Unassessable { params },
        MaxRhat::NotApplicable { reason } => Verdict::NotAssessed { reason: reason.describe().to_string() },
        MaxRhat::NoParams => Verdict::NotAssessed { reason: "no parameter was assessed".to_string() },
    }
}

/// Where the leaf landed and what it took in all. `None` until the leaf is
/// finalized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stored {
    /// The leaf directory, for display ([`display_path`]).
    pub leaf: String,
    /// Wall time of the whole method, `run.json`'s `inputs.wall_time_seconds`.
    pub total_seconds: Option<f64>,
}

/// Everything a method's report is built from. Two producers fill it: the end
/// of the method's run, from memory, and `fit summary`'s loader, from the leaf
/// (`fit_summary::load_method_facts`). Nothing else builds a `MethodReport`
/// ([`crate::fit::report::build::build_method_report`]).
#[derive(Debug, Clone)]
pub struct MethodFacts {
    /// The method leaf's label, its method (`pmmh`, `nl-sbplx`, …) and
    /// directory.
    pub name: String,
    pub method: String,
    pub method_dir: std::path::PathBuf,
    pub header: Option<RunHeader>,
    pub result: crate::fit::method_result::MethodResult,
    pub checks: Vec<Check>,
    /// One record per chain; empty when nothing adapted.
    pub warmup: Vec<ChainWarmup>,
    /// One record per finding, as `diagnostics.json` holds them.
    pub findings: Vec<sim::inference::diagnostic::Diagnostic>,
    /// What in `diagnostics.json` could not be read; empty from the run.
    pub findings_unreadable: Vec<String>,
    pub stored: Option<Stored>,
    /// What the summary's text reads from the leaf beside the typed result;
    /// `None` for a view that does not show it.
    pub chain_detail: Option<ChainDetail>,
}

/// What a leaf's `diagnostics.json` holds: the findings that read, and one
/// sentence for each thing that did not.
#[derive(Debug, Clone, Default)]
pub struct ReadFindings {
    pub findings: Vec<sim::inference::diagnostic::Diagnostic>,
    /// The whole file when it is not a JSON array, else each entry that does
    /// not parse, named by position, type and field. Empty when all read.
    pub unreadable: Vec<String>,
}

/// The findings a leaf recorded (`diagnostics.json`), or none when it holds no
/// such file.
///
/// Read entry by entry, so one record that does not parse costs that record
/// and not the file: an R̂ of `+∞` that an earlier binary wrote as `null` must
/// not take the `bad_init` records beside it out of the summary. Nothing that
/// fails to read is read as "no findings": each is named in
/// [`ReadFindings::unreadable`], which the summary prints and its JSON
/// carries. A file that exists and cannot be opened is an error.
pub fn read_findings(method_dir: &Path) -> Result<ReadFindings, String> {
    use sim::inference::diagnostic::Diagnostic;
    let path = method_dir.join("diagnostics.json");
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ReadFindings::default()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let entries = match serde_json::from_str::<Vec<serde_json::Value>>(&raw) {
        Ok(entries) => entries,
        Err(e) => {
            return Ok(ReadFindings {
                findings: Vec::new(),
                unreadable: vec![format!(
                    "diagnostics.json could not be read, so refused chains are not \
                     listed: {}: {e}",
                    path.display()
                )],
            })
        }
    };
    let mut out = ReadFindings::default();
    for (i, entry) in entries.iter().enumerate() {
        match serde_json::from_value::<Diagnostic>(entry.clone()) {
            Ok(d) => out.findings.push(d),
            Err(e) => out.unreadable.push(unreadable_entry(&path, i, entry, &e)),
        }
    }
    Ok(out)
}

/// The sentence for entry `i` (0-based) of `diagnostics.json`, which did not
/// parse with `err`.
///
/// When the entry's `null`s are what stop it — a file written before
/// non-finite numbers were spelled out, where `null` stood for `+∞`, `−∞` and
/// `NaN` alike — the sentence names those fields and says so. The entry is
/// still refused: admitting the `null` as any one of the three would be a
/// guess, and the guess decides a finding's severity.
fn unreadable_entry(path: &Path, i: usize, entry: &serde_json::Value, err: &serde_json::Error) -> String {
    let kind = entry
        .pointer("/kind/type")
        .and_then(|t| t.as_str())
        .unwrap_or("unknown type");
    let head = format!("{} entry {} (`{kind}`) could not be read", path.display(), i + 1);
    let culprits = null_culprits(entry);
    if culprits.is_empty() {
        return format!("{head}: {err}");
    }
    let fields: Vec<String> = culprits.iter().map(|f| format!("`{f}`")).collect();
    format!(
        "{head}: {} {} null. The file was written before camdl spelled a non-finite \
         number as \"inf\", \"-inf\" or \"NaN\"; null stood for all three, so the value \
         is not recoverable from it. The file's other entries are read.",
        fields.join(", "),
        if fields.len() == 1 { "is" } else { "are" },
    )
}

/// The fields of `entry` whose `null` alone stops it parsing as a
/// [`Diagnostic`](sim::inference::diagnostic::Diagnostic), as dotted paths
/// under `kind` (`rhat`, `attempts.0.time`).
///
/// Found by trial rather than by schema, so it cannot drift from the type: with
/// every `null` replaced by the non-finite spelling `"NaN"` — which a number
/// field, an optional number and an optional string all accept — the entry
/// must parse, or its fault is something else and nothing is named. Each
/// `null` is then restored alone; the ones the entry cannot carry are the
/// culprits, and a `null` that is a value (an attempt's `projected_max` when no
/// particle was live) is not blamed.
fn null_culprits(entry: &serde_json::Value) -> Vec<String> {
    use serde_json::Value;
    use sim::inference::diagnostic::Diagnostic;

    fn nulls(v: &Value, at: &str, out: &mut Vec<String>) {
        match v {
            Value::Null => out.push(at.to_string()),
            Value::Array(xs) => {
                for (i, x) in xs.iter().enumerate() {
                    nulls(x, &format!("{at}/{i}"), out);
                }
            }
            Value::Object(m) => {
                for (k, x) in m {
                    let k = k.replace('~', "~0").replace('/', "~1");
                    nulls(x, &format!("{at}/{k}"), out);
                }
            }
            _ => {}
        }
    }
    let parses = |v: &Value| serde_json::from_value::<Diagnostic>(v.clone()).is_ok();
    let set = |v: &mut Value, at: &str, to: Value| {
        if let Some(slot) = v.pointer_mut(at) {
            *slot = to;
        }
    };

    let mut paths = Vec::new();
    nulls(entry, "", &mut paths);
    let mut filled = entry.clone();
    for at in &paths {
        set(&mut filled, at, Value::from("NaN"));
    }
    if paths.is_empty() || !parses(&filled) {
        return Vec::new();
    }
    paths
        .into_iter()
        .filter(|at| {
            let mut one = filled.clone();
            set(&mut one, at, Value::Null);
            !parses(&one)
        })
        .map(|at| {
            let at = at.strip_prefix("/kind/").unwrap_or(at.trim_start_matches('/'));
            at.split('/')
                .map(|seg| seg.replace("~1", "/").replace("~0", "~"))
                .collect::<Vec<_>>()
                .join(".")
        })
        .collect()
}

/// The chains a method refused at their start, from the `bad_init` records
/// among its findings (gh#607, gh#663) — the durable record, which does not
/// depend on anything having captured stderr.
#[derive(Debug, Clone, Default)]
pub struct RefusedChains {
    /// `(chain, reason)`, in chain order.
    pub records: Vec<(crate::chain_index::ChainIndex, String)>,
    /// What in `diagnostics.json` could not be read ([`ReadFindings::unreadable`]),
    /// so the summary says it could not look rather than implying nothing was
    /// refused.
    pub unreadable: Vec<String>,
}

impl RefusedChains {
    pub fn from_findings(findings: &[sim::inference::diagnostic::Diagnostic], unreadable: &[String]) -> Self {
        use sim::inference::diagnostic::DiagnosticKind;
        let mut records: Vec<_> = findings.iter().filter_map(|d| match &d.kind {
            DiagnosticKind::BadInit { chain_id, reason, .. } => {
                Some((crate::chain_index::ChainIndex(*chain_id), reason.clone()))
            }
            _ => None,
        }).collect();
        records.sort_by_key(|(id, _)| *id);
        RefusedChains { records, unreadable: unreadable.to_vec() }
    }

    /// `"6 of 8 chains ran — refused at start: 3, 5"`, or `None` when every
    /// chain ran. The count that ran is `fit_state.toml`'s `n_good_chains`
    /// when the sampler recorded it, else the total less the refusal records;
    /// if the two sources disagree — a count with no record behind it — the
    /// line says so rather than choosing one.
    pub fn ran_of(&self, n_chains: usize, n_good_chains: Option<usize>) -> Option<String> {
        let n_refused = self.records.len();
        let ran = n_good_chains.unwrap_or(n_chains.saturating_sub(n_refused));
        if ran >= n_chains && n_refused == 0 {
            return None;
        }
        let ids: Vec<String> = self.records.iter().map(|(id, _)| id.to_string()).collect();
        let mut text = format!("{ran} of {n_chains} chains ran");
        if !ids.is_empty() {
            text.push_str(&format!(" — refused at start: {}", ids.join(", ")));
        }
        if ran + n_refused != n_chains {
            text.push_str(&format!(
                " ({} chain(s) not accounted for by a refusal record in diagnostics.json; \
                 chain_starts.tsv gives each chain's status)",
                n_chains.abs_diff(ran + n_refused)
            ));
        }
        Some(text)
    }
}

/// The note a starts rule that discards declared starts prints, for the
/// parameters in `params`. `still_read` is said for the methods (PMMH, `mh`)
/// whose pre-run likelihood check and initial proposal still read the
/// declared values.
pub fn unused_starts_note(rule_tag: &str, params: &[String], still_read: bool) -> String {
    let still = if still_read {
        " It still sets the point of the pre-run likelihood check and the \
         scale of the initial proposal."
    } else {
        ""
    };
    format!(
        "`starts = \"{rule_tag}\"` draws every chain's start, so `[estimate].start` does \
         not set where the chains begin for: {}.{still} Use `starts = \"single\"` to \
         start every chain at the declared values.",
        params.join(", "))
}

/// Everything one Bayesian method's text block reads off disk, gathered once.
///
/// The verdict line and the sections under it are views of the same facts — a
/// chain the `chains` table flags is one of the outliers the verdict counts,
/// and the chains that left latent paths are what the identity line counts —
/// so they are read once, here, rather than by each section separately. Two
/// renderings of one fit that each go to disk for themselves is how they drift.
#[derive(Debug, Clone, Default)]
pub struct ChainDetail {
    /// Per-chain mean log-likelihood and the trace column it was read from.
    /// `None` when no per-chain trace is readable at all.
    pub logliks: Option<crate::fit::chain_diagnostics::ChainLoglikMeans>,
    /// Each chain's comparison of its mean log-likelihood with the other
    /// chains' ([`crate::fit::chain_diagnostics::assess_chain_logliks`]).
    /// Empty when no cross-chain comparison was possible — fewer than two
    /// chains with traces, or no scored column.
    pub scores: Vec<crate::fit::chain_diagnostics::ChainLoglikScore>,
    /// Per-chain saved / forkable latent-path counts. `None` for a method that
    /// saved none, and for one whose `trajectories` block predates the chain
    /// ids that key the rows (gh#727 follow-up: a row's position is not an
    /// answer once a chain has been refused at its start).
    pub saved_paths: Option<crate::fit::pgas::SavedPathReport>,
    /// Latent-path convergence over the paths on disk, or the reason there is
    /// none. `None` when the method saved no path at all.
    pub latent: Option<Result<crate::fit::latent_convergence::LatentConvergence, String>>,
    /// Whether this read wrote the per-cell latent table because the leaf
    /// predates it: `Some(Ok)` written now, `Some(Err)` could not be written,
    /// `None` when the leaf already had it or there was nothing to write.
    pub latent_table_written: Option<Result<(), String>>,
    /// How many chains left a latent path behind — counted from the paths
    /// themselves, so it is right even when the summary block cannot key them.
    pub n_chains_with_paths: usize,
    /// The conditional filter's ESS by observation (`pgas_summary.json`'s
    /// `filter_ess` block); `None` for a method that wrote none.
    pub filter_ess: Option<crate::fit::filter_ess::FilterEss>,
    /// Trajectory renewal by tenth of the series (`pgas_summary.json`'s
    /// `path_renewal` block); `None` for a method that wrote none.
    pub path_renewal: Option<crate::fit::path_renewal::PathRenewal>,
    /// The cross-chain path/parameter compatibility numbers
    /// (`cross_chain_compat.json`); `None` when the method wrote none.
    pub cross_chain: Option<crate::fit::cross_chain_compat::CrossChainReading>,
    /// Per-chain count of retained draws whose log-posterior is `-inf`.
    pub neginf: Option<Vec<crate::fit::chain_diagnostics::ChainNegInf>>,
    /// How many posterior draws a counterfactual could fork from, over the
    /// same chains the rest of the summary reports.
    pub forkable: Option<Forkable>,
    /// The chains `--exclude-chains` kept and dropped, when it is active.
    pub subset: Option<crate::chain_selection::SubsetInfo>,
    /// How many chains the sampler ran and where they began, from
    /// `fit_state.toml`; `None` when the leaf has none (IF2 carries its own).
    pub chain_starts: Option<ChainStarts>,
    /// The run id a stored-fit `starts` source resolved to when this run
    /// started (`run.json` `inputs.starts_source.run_id`, gh#896).
    pub source_run: Option<String>,
}

/// How many chains a sampler ran and where they began, as `fit_state.toml`
/// records it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStarts {
    pub n_chains: usize,
    pub n_good_chains: Option<usize>,
    /// The starts rule as written (`from_prior`, `from_mle @mle`).
    pub source: Option<String>,
    pub kind: Option<crate::fit::starts::ChainStartsKind>,
}

/// The keyed-joint forkable count of a posterior: of `n_total` draws (after
/// any chain selection), `n_forkable` pair with a saved latent path or are
/// deterministic (gh#322).
#[derive(Debug, Clone, PartialEq)]
pub struct Forkable {
    pub n_forkable: usize,
    pub n_total: usize,
    /// The chain selection the count is over; `None` is the whole posterior.
    pub selection: Option<crate::chain_selection::SubsetInfo>,
}

impl ChainDetail {
    /// Everything the summary's Bayesian block reads from the leaf beside the
    /// typed result. `subset` and `forkable` were resolved by the caller under
    /// the command's `--exclude-chains`.
    ///
    /// A PGAS leaf written before the per-cell latent table existed has the
    /// table written here, once, from its saved paths: the same bytes a re-run
    /// would leave.
    pub fn gather(
        method_dir: &Path,
        ll_kind: crate::fit::loglik::LoglikType,
        is_pgas: bool,
        subset: Option<crate::chain_selection::SubsetInfo>,
        forkable: Option<Forkable>,
    ) -> Self {
        use crate::fit::chain_diagnostics as cd;
        let (logliks, scores) = match cd::read_chain_loglik_scores(method_dir, ll_kind) {
            Some((means, scores)) => (Some(means), scores),
            None => (None, Vec::new()),
        };
        let mut detail = Self {
            logliks,
            scores,
            saved_paths: None,
            latent: None,
            latent_table_written: None,
            n_chains_with_paths: 0,
            filter_ess: None,
            path_renewal: None,
            cross_chain: None,
            neginf: cd::read_chain_neginf(method_dir),
            forkable,
            subset,
            // Read for the identity line's chain count and the seeded-from
            // line: display details, not a verdict.
            chain_starts: crate::fit::state::FitState::load(&method_dir.to_string_lossy()).ok()
                .map(|st| ChainStarts {
                    n_chains: st.n_chains,
                    n_good_chains: st.n_good_chains,
                    source: st.chain_init_source,
                    kind: st.chain_starts_kind,
                }),
            source_run: None,
        };
        // PGAS is the one Bayesian method that writes latent paths, so for PMMH
        // / MH there is nothing to read and nothing to omit.
        if !is_pgas {
            return detail;
        }
        let summary = std::fs::read_to_string(
            method_dir.join(crate::run_meta::FitAlgorithm::Pgas.summary_filename()),
        )
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok());
        detail.saved_paths = summary
            .as_ref()
            .and_then(crate::fit::pgas::read_saved_path_counts)
            .filter(|r| !r.per_chain.is_empty());
        detail.filter_ess = summary.as_ref().and_then(crate::fit::filter_ess::FilterEss::read);
        detail.path_renewal = summary.as_ref()
            .and_then(|s| s.get(crate::fit::path_renewal::PATH_RENEWAL_KEY))
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        detail.cross_chain = crate::fit::cross_chain_compat::CrossChainReading::read(method_dir);
        let (latent, n_chains_with_paths) =
            match crate::fit::latent_convergence::read_method_paths(method_dir) {
                Ok(Some((chains, columns))) => {
                    let n = chains.len();
                    let lc = crate::fit::latent_convergence::latent_convergence(&chains, &columns)
                        .map_err(|e| e.to_string());
                    (Some(lc), n)
                }
                Ok(None) => (None, 0),
                Err(e) => (Some(Err(e)), 0),
            };
        if let Some(Ok(lc)) = &latent {
            let table = method_dir.join(crate::fit::latent_convergence::LATENT_CONVERGENCE_TSV);
            if !table.is_file() {
                detail.latent_table_written = Some(lc.write_tsv(&table));
            }
        }
        detail.latent = latent;
        detail.n_chains_with_paths = n_chains_with_paths;
        detail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resumed_chain_adds_its_warmup_to_the_base() {
        let base = ChainWarmup {
            chain: 2, warmup_steps: 600, steps_at_scale_floor: 40,
            final_scale: 0.5, target_accept: 0.337,
        };
        let resumed = ChainWarmup {
            chain: 2, warmup_steps: 400, steps_at_scale_floor: 10,
            final_scale: 0.4, target_accept: 0.337,
        };
        let c = ChainWarmup::continued_from(&base, &resumed);
        assert_eq!((c.warmup_steps, c.steps_at_scale_floor), (1000, 50));
        assert_eq!(c.final_scale, 0.4);
    }

    #[test]
    fn a_summary_without_a_warmup_record_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("pmmh_summary.json");
        std::fs::write(&p, r#"{"method": "pmmh"}"#).unwrap();
        assert!(read_warmup(&p).is_empty());
        assert!(read_warmup(&dir.path().join("absent.json")).is_empty());
    }
}
