//! Resolve a fit run's canonical posterior parameter draws.
//!
//! A Bayesian fit method (PGAS / PMMH / MH) writes `<method_dir>/draws.tsv`: the
//! post-warm-up, thinned parameter draws, concatenated across chains, carrying
//! *every* model parameter (estimated columns first, then the fixed values).
//! That file — not the raw per-chain `trace.tsv`, which now carries warm-up
//! rows (PGAS sweeps and PMMH/MH steps below `burn_in`, for live burn-in
//! observability) — is the canonical posterior. Reading `trace.tsv` would
//! silently fold warm-up draws into a "posterior" band.
//!
//! Resolution is **by artifact, not by method name** (proposal §"types first"):
//! a method has a posterior iff it wrote a `draws.tsv`. An optimizer-only fit
//! (IF2 / NLopt) wrote none, so it resolves to an error, not a silent single
//! point dressed up as a distribution.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::chain_selection::{ChainSelection, SubsetInfo};
use crate::fit::fit_view::FitView;
use crate::run_meta::{FitAlgorithm, InferenceBackend};
use crate::KeyedDraw;

/// The method leaf that produced a posterior draws cloud, located within a
/// resolved fit directory.
#[derive(Debug, Clone)]
pub struct PosteriorDrawsRef {
    /// Bare method name (`scout`, `refine`, `pgas`) the draws came from.
    pub name: String,
    /// `<method_dir>/draws.tsv` — the canonical post-warm-up posterior draws.
    pub draws_path: PathBuf,
    /// The method's inference algorithm, when recoverable from the fit view.
    /// Informational (the band's label); `None` when a method dir was passed
    /// directly with no fit-level view to read the method from.
    pub method: Option<FitAlgorithm>,
    /// The simulation backend the method ran on (`chain_binomial` / `ode`), so a
    /// downstream predictive replays on the SAME forward simulator the fit used
    /// — not a hardcoded default. `None` when a method dir was passed directly.
    pub backend: Option<InferenceBackend>,
    /// Read-side chain selection (`--exclude-chains`) to apply when the cloud is
    /// loaded — `None` for every consumer that does not surface the flag, so the
    /// filter is a no-op there (byte-identical to the pre-selection behaviour).
    /// Attached via [`PosteriorDrawsRef::with_selection`]; applied ONCE, in the
    /// load methods, so the filter lives at the draws authority rather than in
    /// each consumer.
    pub chain_selection: Option<ChainSelection>,
}

const DRAWS_FILE: &str = "draws.tsv";

/// Why [`resolve_posterior_draws`] returned no cloud.
///
/// Several readers have a fallback for a fit without a posterior — `compare`
/// scores an optimizer fit at its single winner, `fit table --quantity` renders
/// `—` — and before this type they took it on *any* error. That turned a
/// refusal (a leaf whose chain directories count from 1, gh#990) or an
/// unknown method name into a silent point-estimate answer. The two cases are
/// distinct values so a caller can fall back on the first and must propagate
/// the second.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosteriorDrawsError {
    /// The fit has no posterior cloud: no method (or not the named method)
    /// wrote `draws.tsv` — an optimizer fit, or a sampler that has not
    /// finished. The point-estimate fallback is the right answer.
    NoPosterior(String),
    /// Anything else — not a directory, an unknown method name, a leaf refused
    /// for its chain numbering. Never a reason to fall back.
    Other(String),
}

impl std::fmt::Display for PosteriorDrawsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PosteriorDrawsError::NoPosterior(m) | PosteriorDrawsError::Other(m) => f.write_str(m),
        }
    }
}

impl From<PosteriorDrawsError> for String {
    fn from(e: PosteriorDrawsError) -> String {
        e.to_string()
    }
}

/// [`crate::leaf_use::require_completed`] for a method leaf of a fit view: its
/// state read from its `run.json` and `.lock`, so the refusal says whether it
/// failed (and why), is running, or was interrupted.
fn require_completed(m: &crate::fit::fit_view::FitMethodView) -> Result<(), PosteriorDrawsError> {
    use PosteriorDrawsError::Other;
    if m.completed() {
        return Ok(());
    }
    match crate::leaf_use::read(&m.method_dir).map_err(Other)? {
        Some(leaf) => crate::leaf_use::require_completed(&leaf).map_err(Other),
        None => Err(Other(format!("{} holds no run.json", m.method_dir.display()))),
    }
}

/// [`crate::chain_index::refuse_one_based_layout`], as a resolution failure
/// that is not "no posterior".
fn refuse_old_layout(method_dir: &Path) -> Result<(), PosteriorDrawsError> {
    crate::chain_index::refuse_one_based_layout(method_dir).map_err(PosteriorDrawsError::Other)
}

/// Resolve a fit reference to its canonical posterior draws.
///
/// `fit_ref` is a path to a fit results directory — either the segment
/// (`results/fits/<stem>-<hash>/`) or a single leaf directory. The last leaf
/// (in label order) that wrote a `draws.tsv` is chosen; `method` names one
/// leaf by its label instead — an internal selector, used by the readers that
/// render one leaf's stanza. Errors (never guesses) when the directory holds
/// no posterior draws or the named leaf has none
/// ([`PosteriorDrawsError::NoPosterior`]), and for every other failure,
/// including a leaf refused for its chain numbering
/// ([`PosteriorDrawsError::Other`]).
///
/// Only a completed run's `draws.tsv` is a posterior. A run that failed after
/// writing it, or was killed after writing it, leaves a file that looks
/// complete; such a leaf — named, or among the runs with draws — is refused
/// with its state ([`crate::leaf_use::require_completed`]), never read and
/// never passed over in silence.
pub fn resolve_posterior_draws(
    fit_ref: &str,
    method: Option<&str>,
) -> Result<PosteriorDrawsRef, PosteriorDrawsError> {
    use PosteriorDrawsError::{NoPosterior, Other};
    let dir = Path::new(fit_ref);
    if !dir.is_dir() {
        return Err(Other(format!(
            "not a fit results directory: {fit_ref}\n  \
             pass the fit directory, e.g. results/fits/<stem>-<hash>/ \
             (the path `camdl fit run` printed)"
        )));
    }

    // Segment case: the fit-level view enumerates the method leaves in label
    // order.
    if let Some(view) = FitView::read(dir) {
        // Methods that actually wrote a posterior cloud — the artifact gate.
        let with_draws: Vec<&crate::fit::fit_view::FitMethodView> = view
            .methods
            .iter()
            .filter(|s| s.method_dir.join(DRAWS_FILE).is_file())
            .collect();

        if let Some(want) = method {
            let named: Vec<&crate::fit::fit_view::FitMethodView> =
                view.methods.iter().filter(|s| s.method.as_str() == want).collect();
            let chosen = match named.as_slice() {
                [] => {
                    let names: Vec<&str> = view.methods.iter().map(|s| s.method.as_str()).collect();
                    return Err(Other(format!(
                        "no method named '{want}' in {fit_ref}\n  available methods: {}",
                        names.join(", ")
                    )));
                }
                [one] => *one,
                several => return Err(Other(several_runs(fit_ref, &format!("'{want}' runs"), several))),
            };
            require_completed(chosen)?;
            let draws_path = chosen.method_dir.join(DRAWS_FILE);
            if !draws_path.is_file() {
                return Err(NoPosterior(no_draws_for_method(want, chosen.method)));
            }
            refuse_old_layout(&chosen.method_dir)?;
            return Ok(PosteriorDrawsRef {
                name: chosen.method.as_str().to_string(),
                draws_path,
                method: Some(chosen.method),
                backend: Some(chosen.backend),
                chain_selection: None,
            });
        }

        // No selector: the one leaf that wrote a posterior cloud. Several are
        // refused with them listed rather than one chosen by label order
        // (gh#896). A leaf whose chain directories count from 1 is refused
        // here, before any reader follows a draw's `chain` key into
        // `chain_<key>/` (gh#990). A run with draws that did not complete is
        // refused before any is chosen.
        for s in &with_draws {
            require_completed(s)?;
        }
        let chosen = match with_draws.as_slice() {
            [] => return Err(NoPosterior(no_posterior_error(fit_ref, &view))),
            [one] => *one,
            several => {
                return Err(Other(several_runs(fit_ref, "runs with posterior draws", several)))
            }
        };
        let draws_path = chosen.method_dir.join(DRAWS_FILE);
        refuse_old_layout(&chosen.method_dir)?;
        return Ok(PosteriorDrawsRef {
            name: chosen.method.as_str().to_string(),
            draws_path,
            method: Some(chosen.method),
            backend: Some(chosen.backend),
            chain_selection: None,
        });
    }

    // Leaf-dir-passed-directly case: no fit-level view, but the directory
    // itself may hold a draws.tsv — a leaf outside a fit folder, or a bare
    // directory of draws. A leaf must be a completed run, and its label comes
    // from its own `run.json`; a record that cannot be read is refused rather
    // than taken as a bare directory.
    let draws_path = dir.join(DRAWS_FILE);
    if draws_path.is_file() {
        let leaf = crate::leaf_use::read(dir).map_err(Other)?;
        if let Some(leaf) = &leaf {
            crate::leaf_use::require_completed(leaf).map_err(Other)?;
        }
        refuse_old_layout(dir)?;
        let method = leaf
            .and_then(|l| l.record.inputs.get("method").and_then(|s| s.as_str()).map(String::from))
            .unwrap_or_default();
        return Ok(PosteriorDrawsRef {
            name: method,
            draws_path,
            method: None,
            backend: None,
            chain_selection: None,
        });
    }

    Err(NoPosterior(format!(
        "no posterior draws under {fit_ref}\n  \
         expected a Bayesian fit (PGAS / PMMH / MH) that wrote {DRAWS_FILE}; \
         found no fit methods there"
    )))
}

impl PosteriorDrawsRef {
    /// Attach a read-side chain selection to apply when the cloud is loaded.
    /// `None` leaves the cloud unfiltered (the default for every consumer that
    /// does not surface `--exclude-chains`).
    pub fn with_selection(mut self, selection: Option<ChainSelection>) -> Self {
        self.chain_selection = selection;
        self
    }

    /// Read the keyed posterior cloud from `draws_path`, applying the attached
    /// chain selection (if any). Returns the retained rows and — when a
    /// selection was active — the [`SubsetInfo`] provenance record.
    ///
    /// This is the single point where the chain filter meets the cloud: the raw
    /// parse ([`crate::load_draws_tsv_keyed`]) then, once, the shared filter
    /// ([`ChainSelection::apply_keyed`]). Every consumer that reads a cloud
    /// through the ref inherits the filter here rather than re-deriving it.
    pub fn load_keyed_with_info(&self) -> Result<(Vec<KeyedDraw>, Option<SubsetInfo>), String> {
        let rows = crate::load_draws_tsv_keyed(&self.draws_path.to_string_lossy())?;
        match &self.chain_selection {
            Some(sel) => {
                let (kept, info) = sel.apply_keyed(rows)?;
                Ok((kept, Some(info)))
            }
            None => Ok((rows, None)),
        }
    }

    /// Param-only rows (the `(chain, draw)` key dropped), selection applied.
    /// Second return is the [`SubsetInfo`] when a selection was active.
    pub fn load_params_with_info(
        &self,
    ) -> Result<(Vec<HashMap<String, f64>>, Option<SubsetInfo>), String> {
        let (keyed, info) = self.load_keyed_with_info()?;
        Ok((keyed.into_iter().map(|d| d.params).collect(), info))
    }
}

/// The "this method has no `draws.tsv`" error, framed by the leaf's algorithm
/// (gh#343). A Bayesian sampler (PGAS / PMMH / MH) writes its cloud at
/// completion, so a missing file means the method is incomplete or still running
/// — not an optimizer. An optimizer (IF2 / NLopt) never has a band.
fn no_draws_for_method(name: &str, method: FitAlgorithm) -> String {
    let label = fit_algorithm_label(method);
    if method.is_posterior_sampler() {
        format!(
            "method '{name}' ({label}) has not written its posterior draws \
             ({DRAWS_FILE}) yet — it is a Bayesian sampler whose cloud is written at \
             completion, so the method is likely incomplete or still running."
        )
    } else {
        format!(
            "method '{name}' produced no posterior draws ({DRAWS_FILE}).\n  \
             {label} is an optimizer method: it returns a single best-fit point, \
             not a distribution."
        )
    }
}

/// A refusal naming every candidate run, for a reader that needs one (gh#896).
fn several_runs(fit_ref: &str, what: &str, runs: &[&crate::fit::fit_view::FitMethodView]) -> String {
    let mut msg = format!(
        "{fit_ref} holds {} {what}, and this reads one; pass the run's directory (or its \
         @label or run-id prefix):",
        runs.len()
    );
    for r in runs {
        msg.push_str(&format!("\n  {} {}", r.method.as_str(), r.method_dir.display()));
    }
    msg
}

/// Build the actionable "no posterior cloud here" error. A Bayesian sampler
/// leaf with no `draws.tsv` is incomplete / still running — never an
/// "optimizer fit" (gh#343); only a folder of genuine optimizers is
/// redirected to the plug-in-simulate workflow.
fn no_posterior_error(fit_ref: &str, view: &FitView) -> String {
    let methods: Vec<String> = view
        .methods
        .iter()
        .map(|s| format!("{} ({})", s.method.as_str(), fit_algorithm_label(s.method)))
        .collect();
    if let Some(term) = view.methods.iter().find(|s| s.method.is_posterior_sampler()) {
        let label = fit_algorithm_label(term.method);
        return format!(
            "no {DRAWS_FILE} under {fit_ref}: the Bayesian method '{}' \
             ({label}) has not written its posterior draws cloud.\n  \
             methods found: {}\n  \
             {DRAWS_FILE} is written at method completion — if the fit is still \
             running or was interrupted, let it finish (or re-run it); a completed \
             {label} method has a full posterior cloud to band.",
            term.method.as_str(),
            methods.join(", "),
        );
    }
    format!(
        "no method in {fit_ref} produced a posterior draws cloud ({DRAWS_FILE}).\n  \
         methods found: {}\n  \
         An optimizer fit (IF2 / NLopt) returns a single best-fit parameter set, \
         not a distribution, so there is no band to draw. Get those parameters with\n    \
         camdl fit summary {fit_ref} --params-only\n  \
         and run\n    \
         camdl simulate <model> --params <(camdl fit summary {fit_ref} --params-only) \
         --obs-only-dir out/",
        methods.join(", ")
    )
}

fn fit_algorithm_label(m: FitAlgorithm) -> &'static str {
    match m {
        FitAlgorithm::If2 => "IF2",
        FitAlgorithm::Pgas => "PGAS",
        FitAlgorithm::Pmmh => "PMMH",
        FitAlgorithm::Mh => "MH",
        FitAlgorithm::Pfilter => "particle filter",
        FitAlgorithm::NlSbplx => "NLopt/sbplx",
        FitAlgorithm::NlBobyqa => "NLopt/bobyqa",
        FitAlgorithm::NlLbfgs => "NLopt/lbfgs",
    }
}

/// Resolve a fit's `[fixed]` block, for #273 backfill: fill any parameter
/// *absent* from a raw draws TSV with the value the fit held it fixed at.
///
/// `fit_ref` is either a fit-config TOML (read the `[fixed]` block) or a fit
/// results directory (read the `fixed` map off its `fit.meta.json` sidecar). A
/// raw posterior trace tail carries only the estimated columns, so without this
/// a `simulate --draws tail.tsv` run would fall back to model defaults for the
/// fixed parameters — silently diverging from the fit.
///
/// `model` resolves a config's `[fixed] from_scenario = "name"`, which names
/// one of the model's scenarios; a stored fit's sidecar is already resolved.
pub fn resolve_fixed_for_backfill(
    fit_ref: &str,
    model: &ir::Model,
) -> Result<Vec<(String, f64)>, String> {
    let path = Path::new(fit_ref);
    if path.is_dir() {
        let side = crate::run_meta::read_fit_sidecar(path)?.ok_or_else(|| {
            format!("no fit.meta.json under {fit_ref} to read the fit's [fixed] block from")
        })?;
        let mut out: Vec<(String, f64)> = side.fixed.into_iter().collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        return Ok(out);
    }
    let cfg = crate::fit::config_v2::Problem::load(fit_ref)
        .map_err(|e| format!("failed to load --fit config '{fit_ref}': {e}"))?;
    let fixed = cfg
        .fixed
        .resolve_with_model(model)
        .map_err(|e| format!("resolving [fixed] from '{fit_ref}': {e}"))?;
    Ok(fixed.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Write one realistic `FitMethod` leaf into a segment, mirroring the on-disk
    /// shape the runner writes: `<seg>/<label>-<h8>/seed_1-<h8>/run.json`.
    /// Returns the leaf dir (where `draws.tsv` would live). `method` is the
    /// inference algorithm tag; `draws` writes a `draws.tsv` iff `Some`.
    fn write_method_leaf(
        seg: &Path,
        method_label: &str,
        method: &str,
        draws: Option<&str>,
    ) -> PathBuf {
        let leaf = seg.join(format!("{method_label}-1fb03eee")).join("seed_1-06cbd6b3");
        fs::create_dir_all(&leaf).unwrap();
        let fit_hash = "abc12345".to_string() + &"0".repeat(56);
        // run_id must be valid 64-char hex (ContentHash rejects non-hex), and
        // unique per leaf — derive it from the label's bytes.
        let ord_hex: String = method_label.bytes().map(|b| format!("{b:02x}")).collect();
        let run_id = format!("{:0<64}", format!("{ord_hex}abcdef"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_method","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"toy","hash":"{fit_hash}","schema_version":1}},{{"name":"method","label":"{method_label}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"{method}","backend":"chain_binomial","seed":1,"n_chains":4,"best_loglik":-12.3,"best_chain":1}},"provenance":{{"created_at":"2026-06-22T00:00:0{ord}Z","argv":["camdl","fit","run"]}}}}"#,
            ord = method_label.len() % 10,
        );
        fs::write(leaf.join("run.json"), rec).unwrap();
        if let Some(content) = draws {
            fs::write(leaf.join(DRAWS_FILE), content).unwrap();
        }
        leaf
    }

    /// A well-formed segment (with the required sidecar) holding the given
    /// method leaves. Each entry is `(leaf_label, method, draws?)` — the label
    /// and the method are the same string on every leaf the runner writes.
    fn fixture_segment(root: &Path, methods: &[(&str, &str, Option<&str>)]) -> PathBuf {
        let seg = root.join("fits").join("toy-abc12345");
        fs::create_dir_all(&seg).unwrap();
        for (label, method, draws) in methods {
            write_method_leaf(&seg, label, method, *draws);
        }
        // Sidecar with non-empty resolved_priors so a Bayesian segment is
        // treated as well-formed (FitView flags empty priors as a bug).
        let sidecar = crate::run_meta::FitSidecar {
            estimated: vec!["beta".into()],
            resolved_priors: vec![crate::run_meta::ResolvedPriorEntry {
                param: "beta".into(),
                source: "model_ir".into(),
                overridden_model_prior: None,
            }],
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&seg, Path::new("nonexistent.toml"), Path::new(""), &sidecar).unwrap();
        seg
    }

    const TWO_DRAWS: &str = "beta\tgamma\n0.5\t0.1\n0.6\t0.12\n";

    #[test]
    fn resolves_terminal_method_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_ok");
        // An IF2 leaf (no draws) beside a PGAS leaf (draws) — two methods of
        // one problem: the leaf with a cloud wins.
        let seg = fixture_segment(&tmp, &[
            ("if2",  "if2", None),
            ("pgas", "pgas", Some(TWO_DRAWS)),
        ]);
        let r = resolve_posterior_draws(seg.to_str().unwrap(), None).expect("resolves");
        assert_eq!(r.name, "pgas");
        assert_eq!(r.method, Some(FitAlgorithm::Pgas));
        assert!(r.draws_path.ends_with(DRAWS_FILE));
        assert!(r.draws_path.is_file());
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_no_method_has_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_none");
        // An optimizer-only fit: no posterior cloud anywhere.
        let seg = fixture_segment(&tmp, &[
            ("if2",      "if2", None),
            ("nl-sbplx", "nl-sbplx", None),
        ]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::NoPosterior(_)), "{err:?}");
        let err = err.to_string();
        assert!(err.contains("no method"), "actionable error, got: {err}");
        assert!(err.contains("--params-only"), "points to the optimizer workflow: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn method_override_selects_named_method() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_methodpick");
        let seg = fixture_segment(&tmp, &[
            ("pgas",  "pgas", Some("beta\n0.1\n")),
            ("pmmh",  "pmmh", Some(TWO_DRAWS)),
        ]);
        // Without a selector, two runs with draws are refused, both listed,
        // rather than the last in label order read silently (gh#896)...
        let err = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap_err().to_string();
        assert!(err.contains("holds 2 runs with posterior draws"), "{err}");
        assert!(err.contains("pgas-") && err.contains("pmmh-"), "both runs are listed: {err}");
        // ...with the selector naming pgas, that leaf is chosen.
        let r = resolve_posterior_draws(seg.to_str().unwrap(), Some("pgas")).unwrap();
        assert_eq!(r.name, "pgas");
        assert_eq!(r.method, Some(FitAlgorithm::Pgas));
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_named_method_has_no_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_namednone");
        let seg = fixture_segment(&tmp, &[
            ("if2",  "if2",  None),
            ("pgas", "pgas", Some(TWO_DRAWS)),
        ]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), Some("if2")).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::NoPosterior(_)), "{err:?}");
        let err = err.to_string();
        assert!(err.contains("no posterior draws") || err.contains("optimizer"), "got: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn pgas_method_without_draws_is_not_called_an_optimizer() {
        // gh#343: a Bayesian sampler method (pgas) that has not written draws.tsv
        // — incomplete or still running — must NOT be framed as an optimizer fit,
        // on either the default (last leaf) or the name-selected path.
        let tmp = crate::test_support::unique_temp_dir("pdraws_pgas_nodraws");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", None)]);

        let err = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::NoPosterior(_)), "{err:?}");
        let err = err.to_string();
        assert!(!err.contains("optimizer"), "a pgas method is not an optimizer: {err}");
        assert!(err.contains(DRAWS_FILE), "names the missing posterior draws: {err}");

        let err2 = resolve_posterior_draws(seg.to_str().unwrap(), Some("pgas")).unwrap_err();
        assert!(matches!(err2, PosteriorDrawsError::NoPosterior(_)), "{err2:?}");
        let err2 = err2.to_string();
        assert!(!err2.contains("optimizer"), "named pgas method is not an optimizer: {err2}");
        assert!(err2.contains(DRAWS_FILE), "named method names the missing draws: {err2}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_on_unknown_method_name() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_badmethod");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", Some(TWO_DRAWS))]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), Some("nope")).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::Other(_)), "an unknown name is not 'no posterior': {err:?}");
        let err = err.to_string();
        assert!(err.contains("no method named 'nope'"), "got: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    /// Rewrite a fixture leaf's `run.json` as a run that did not complete:
    /// `status` and, for a failed run, its `failure`.
    fn mark(leaf: &Path, status: &str, failure: Option<&str>) {
        let mut rec: serde_json::Value =
            serde_json::from_slice(&fs::read(leaf.join("run.json")).unwrap()).unwrap();
        rec["status"] = status.into();
        if let Some(message) = failure {
            rec["failure"] = serde_json::json!({"message": message, "operation": "pgas"});
        }
        fs::write(leaf.join("run.json"), rec.to_string()).unwrap();
    }

    /// A run that failed after writing its `draws.tsv` is not a posterior: it
    /// is refused, with its state and reason, whether it is the folder's only
    /// run with draws, is named by its method, or is the leaf passed — never
    /// read, and never `NoPosterior` (which callers answer with a point
    /// estimate).
    #[test]
    fn a_failed_run_with_draws_is_refused_naming_its_state() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_failed");
        let seg = fixture_segment(&tmp, &[
            ("if2",  "if2", None),
            ("pgas", "pgas", Some(TWO_DRAWS)),
        ]);
        let leaf = seg.join("pgas-1fb03eee").join("seed_1-06cbd6b3");
        mark(&leaf, "failed", Some("error: --resume lineage: cannot read the base"));
        for (handle, method) in [(seg.clone(), None), (seg.clone(), Some("pgas")), (leaf.clone(), None)] {
            let err = resolve_posterior_draws(handle.to_str().unwrap(), method).unwrap_err();
            assert!(matches!(err, PosteriorDrawsError::Other(_)), "{handle:?} {method:?}: {err:?}");
            let err = err.to_string();
            assert!(err.contains(&format!("{} is not a completed run", leaf.display())), "{err}");
            assert!(err.contains("it failed: --resume lineage: cannot read the base"), "{err}");
        }
        fs::remove_dir_all(&tmp).ok();
    }

    /// The same for a run still `running` whose `.lock` names no live process
    /// (killed after writing `draws.tsv`): refused as interrupted.
    #[test]
    fn an_interrupted_run_with_draws_is_refused_as_interrupted() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_interrupted");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", Some(TWO_DRAWS))]);
        let leaf = seg.join("pgas-1fb03eee").join("seed_1-06cbd6b3");
        mark(&leaf, "running", None);
        let err = resolve_posterior_draws(leaf.to_str().unwrap(), None).unwrap_err().to_string();
        assert!(err.contains("it was interrupted: no live process holds it"), "{err}");
        fs::remove_dir_all(&tmp).ok();
    }

    /// A directory holding a `draws.tsv` beside a `run.json` that is not a run
    /// record is refused, naming the file's problem, rather than read as a
    /// bare directory of draws.
    #[test]
    fn a_dir_with_draws_and_an_unreadable_run_json_is_refused() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_unreadable");
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(DRAWS_FILE), TWO_DRAWS).unwrap();
        fs::write(tmp.join("run.json"), "{ truncated").unwrap();
        let err = resolve_posterior_draws(tmp.to_str().unwrap(), None).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::Other(_)), "{err:?}");
        let err = err.to_string();
        assert!(err.contains(&tmp.display().to_string()) && err.contains("its run.json is not a run record"),
            "{err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolves_dir_with_draws_directly() {
        // A bare directory holding a draws.tsv (no fit view) resolves via the
        // fallback — for power users who point at a method dir directly.
        let tmp = crate::test_support::unique_temp_dir("pdraws_direct");
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(DRAWS_FILE), TWO_DRAWS).unwrap();
        let r = resolve_posterior_draws(tmp.to_str().unwrap(), None).expect("resolves");
        assert_eq!(r.draws_path, tmp.join(DRAWS_FILE));
        assert_eq!(r.method, None);
        fs::remove_dir_all(&tmp).ok();
    }

    /// gh#990: a leaf whose chain directories count from 1 is refused as
    /// `Other`, never as `NoPosterior` — the variant callers fall back on —
    /// on the segment path and on a leaf passed directly.
    #[test]
    fn a_one_based_leaf_is_refused_not_reported_as_no_posterior() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_onebased");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", Some(TWO_DRAWS))]);
        let leaf = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap()
            .draws_path.parent().unwrap().to_path_buf();
        fs::create_dir_all(leaf.join("chain_1")).unwrap();
        fs::create_dir_all(leaf.join("chain_2")).unwrap();
        for target in [seg.as_path(), leaf.as_path()] {
            let err = resolve_posterior_draws(target.to_str().unwrap(), None).unwrap_err();
            match &err {
                PosteriorDrawsError::Other(m) => assert!(m.contains("gh#990"), "{m}"),
                other => panic!("{}: expected the gh#990 refusal, got {other:?}", target.display()),
            }
        }
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_not_a_directory() {
        let err = resolve_posterior_draws("/no/such/dir/at/all", None).unwrap_err();
        assert!(matches!(err, PosteriorDrawsError::Other(_)), "{err:?}");
        let err = err.to_string();
        assert!(err.contains("not a fit results directory"), "got: {err}");
    }

    // ─── #273: reading a fit's [fixed] ────────────────────────────────────

    #[test]
    fn resolve_fixed_reads_sidecar_fixed_from_dir() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_fixed_dir");
        fs::create_dir_all(&tmp).unwrap();
        let sidecar = crate::run_meta::FitSidecar {
            fixed: std::collections::BTreeMap::from([("N0".to_string(), 1000.0), ("sigma".to_string(), 0.11)]),
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&tmp, Path::new("nonexistent.toml"), Path::new(""), &sidecar).unwrap();
        let mut got = resolve_fixed_for_backfill(tmp.to_str().unwrap(), &crate::test_support::tiny_model())
            .expect("reads sidecar");
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![("N0".to_string(), 1000.0), ("sigma".to_string(), 0.11)]);
        fs::remove_dir_all(&tmp).ok();
    }
}
