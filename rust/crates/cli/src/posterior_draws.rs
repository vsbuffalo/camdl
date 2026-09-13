//! Resolve a fit run's canonical posterior parameter draws.
//!
//! A Bayesian fit stage (PGAS / PMMH / MH) writes `<stage_dir>/draws.tsv`: the
//! post-warm-up, thinned parameter draws, concatenated across chains, carrying
//! *every* model parameter (estimated columns first, then the fixed values).
//! That file — not the raw per-chain `trace.tsv`, which now carries warm-up
//! rows (PGAS sweeps and PMMH/MH steps below `burn_in`, for live burn-in
//! observability) — is the canonical posterior. Reading `trace.tsv` would
//! silently fold warm-up draws into a "posterior" band.
//!
//! Resolution is **by artifact, not by method name** (proposal §"types first"):
//! a stage has a posterior iff it wrote a `draws.tsv`. An optimizer-only fit
//! (IF2 / NLopt) wrote none, so it resolves to an error, not a silent single
//! point dressed up as a distribution.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::chain_selection::{ChainSelection, SubsetInfo};
use crate::fit::fit_view::FitView;
use crate::run_meta::{FitAlgorithm, InferenceBackend};
use crate::KeyedDraw;

/// The method leaf that produced a posterior draws cloud, located within a
/// resolved fit directory.
#[derive(Debug, Clone)]
pub struct PosteriorDrawsRef {
    /// Bare stage name (`scout`, `refine`, `pgas`) the draws came from.
    pub stage: String,
    /// `<stage_dir>/draws.tsv` — the canonical post-warm-up posterior draws.
    pub draws_path: PathBuf,
    /// The stage's inference algorithm, when recoverable from the fit view.
    /// Informational (the band's label); `None` when a stage dir was passed
    /// directly with no fit-level view to read the method from.
    pub method: Option<FitAlgorithm>,
    /// The simulation backend the stage ran on (`chain_binomial` / `ode`), so a
    /// downstream predictive replays on the SAME forward simulator the fit used
    /// — not a hardcoded default. `None` when a stage dir was passed directly.
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

/// Resolve a fit reference to its canonical posterior draws.
///
/// `fit_ref` is a path to a fit results directory — either the segment
/// (`results/fits/<stem>-<hash>/`) or a single leaf directory. The last leaf
/// (in label order) that wrote a `draws.tsv` is chosen; `stage` names one
/// leaf by its label instead — an internal selector, used by the readers that
/// render one leaf's stanza. Errors (never guesses) when the directory holds
/// no posterior draws or the named leaf has none.
pub fn resolve_posterior_draws(
    fit_ref: &str,
    stage: Option<&str>,
) -> Result<PosteriorDrawsRef, String> {
    let dir = Path::new(fit_ref);
    if !dir.is_dir() {
        return Err(format!(
            "not a fit results directory: {fit_ref}\n  \
             pass the fit directory, e.g. results/fits/<stem>-<hash>/ \
             (the path `camdl fit run` printed)"
        ));
    }

    // Segment case: the fit-level view enumerates the method leaves in label
    // order.
    if let Some(view) = FitView::read(dir) {
        // Stages that actually wrote a posterior cloud — the artifact gate.
        let with_draws: Vec<&crate::fit::fit_view::FitStageView> = view
            .stages
            .iter()
            .filter(|s| s.stage_dir.join(DRAWS_FILE).is_file())
            .collect();

        if let Some(want) = stage {
            let chosen = view.stages.iter().find(|s| s.method.as_str() == want).ok_or_else(|| {
                let names: Vec<&str> = view.stages.iter().map(|s| s.method.as_str()).collect();
                format!(
                    "no method named '{want}' in {fit_ref}\n  available methods: {}",
                    names.join(", ")
                )
            })?;
            let draws_path = chosen.stage_dir.join(DRAWS_FILE);
            if !draws_path.is_file() {
                return Err(no_draws_for_stage(want, chosen.method));
            }
            return Ok(PosteriorDrawsRef {
                stage: chosen.method.as_str().to_string(),
                draws_path,
                method: Some(chosen.method),
                backend: Some(chosen.backend),
                chain_selection: None,
            });
        }

        // No selector: the last leaf that wrote a posterior cloud.
        let chosen = with_draws.last().ok_or_else(|| no_posterior_error(fit_ref, &view))?;
        let draws_path = chosen.stage_dir.join(DRAWS_FILE);
        return Ok(PosteriorDrawsRef {
            stage: chosen.method.as_str().to_string(),
            draws_path,
            method: Some(chosen.method),
            backend: Some(chosen.backend),
            chain_selection: None,
        });
    }

    // Leaf-dir-passed-directly case: no fit-level view, but the directory
    // itself may be a leaf holding draws.tsv. Its label comes from its own
    // `run.json` when it has one.
    let draws_path = dir.join(DRAWS_FILE);
    if draws_path.is_file() {
        let stage = std::fs::read(dir.join("run.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<runid::RunRecord>(&b).ok())
            .and_then(|r| r.inputs.get("method").and_then(|s| s.as_str()).map(String::from))
            .unwrap_or_default();
        return Ok(PosteriorDrawsRef {
            stage,
            draws_path,
            method: None,
            backend: None,
            chain_selection: None,
        });
    }

    Err(format!(
        "no posterior draws under {fit_ref}\n  \
         expected a Bayesian fit (PGAS / PMMH / MH) that wrote {DRAWS_FILE}; \
         found no fit stages there"
    ))
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

/// The "this stage has no `draws.tsv`" error, framed by the stage's method
/// (gh#343). A Bayesian sampler (PGAS / PMMH / MH) writes its cloud at
/// completion, so a missing file means the stage is incomplete or still running
/// — not an optimizer. An optimizer (IF2 / NLopt) never has a band.
fn no_draws_for_stage(stage: &str, method: FitAlgorithm) -> String {
    let label = fit_algorithm_label(method);
    if method.is_posterior_sampler() {
        format!(
            "stage '{stage}' ({label}) has not written its posterior draws \
             ({DRAWS_FILE}) yet — it is a Bayesian sampler whose cloud is written at \
             completion, so the stage is likely incomplete or still running."
        )
    } else {
        format!(
            "stage '{stage}' produced no posterior draws ({DRAWS_FILE}).\n  \
             {label} is an optimizer stage: it returns a single best-fit point, \
             not a distribution."
        )
    }
}

/// Build the actionable "no posterior cloud here" error, framed by the terminal
/// stage's method. A terminal Bayesian sampler with no `draws.tsv` is incomplete
/// / still running — never an "optimizer fit" (gh#343); only a genuine optimizer
/// is redirected to the plug-in-simulate workflow.
fn no_posterior_error(fit_ref: &str, view: &FitView) -> String {
    let methods: Vec<String> = view
        .stages
        .iter()
        .map(|s| format!("{} ({})", s.method.as_str(), fit_algorithm_label(s.method)))
        .collect();
    if let Some(term) = view.stages.last().filter(|s| s.method.is_posterior_sampler()) {
        let label = fit_algorithm_label(term.method);
        return format!(
            "no {DRAWS_FILE} under {fit_ref}: the terminal Bayesian stage '{}' \
             ({label}) has not written its posterior draws cloud.\n  \
             stages found: {}\n  \
             {DRAWS_FILE} is written at stage completion — if the fit is still \
             running or was interrupted, let it finish (or re-run it); a completed \
             {label} stage has a full posterior cloud to band.",
            term.method.as_str(),
            methods.join(", "),
        );
    }
    format!(
        "no stage in {fit_ref} produced a posterior draws cloud ({DRAWS_FILE}).\n  \
         stages found: {}\n  \
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
        FitAlgorithm::Nuts => "NUTS",
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
pub fn resolve_fixed_for_backfill(fit_ref: &str) -> Result<Vec<(String, f64)>, String> {
    let path = Path::new(fit_ref);
    if path.is_dir() {
        let side = crate::run_meta::read_fit_sidecar(path).ok_or_else(|| {
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
        .resolve()
        .map_err(|e| format!("resolving [fixed] from '{fit_ref}': {e}"))?;
    Ok(fixed.into_iter().collect())
}

/// #273 backfill: insert each `(name, value)` from `fixed` into every draw row
/// that lacks that column, **never** overwriting a column the row provides.
/// Returns the set of names actually filled (for the user-facing report). The
/// "never overwrite" rule is what keeps a posterior draw winning over a value
/// the fit merely held fixed.
pub fn backfill_fixed(
    draws: &mut [HashMap<String, f64>],
    fixed: &[(String, f64)],
) -> BTreeSet<String> {
    let mut filled = BTreeSet::new();
    for row in draws.iter_mut() {
        for (name, value) in fixed {
            if !row.contains_key(name) {
                row.insert(name.clone(), *value);
                filled.insert(name.clone());
            }
        }
    }
    filled
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Write one realistic `FitStage` leaf into a segment, mirroring the on-disk
    /// shape the runner writes: `<seg>/<label>-<h8>/seed_1-<h8>/run.json`.
    /// Returns the leaf dir (where `draws.tsv` would live). `method` is the
    /// inference algorithm tag; `draws` writes a `draws.tsv` iff `Some`.
    fn write_stage_leaf(
        seg: &Path,
        stage_label: &str,
        method: &str,
        draws: Option<&str>,
    ) -> PathBuf {
        let leaf = seg.join(format!("{stage_label}-1fb03eee")).join("seed_1-06cbd6b3");
        fs::create_dir_all(&leaf).unwrap();
        let fit_hash = "abc12345".to_string() + &"0".repeat(56);
        // run_id must be valid 64-char hex (ContentHash rejects non-hex), and
        // unique per leaf — derive it from the label's bytes.
        let ord_hex: String = stage_label.bytes().map(|b| format!("{b:02x}")).collect();
        let run_id = format!("{:0<64}", format!("{ord_hex}abcdef"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"toy","hash":"{fit_hash}","schema_version":1}},{{"name":"method","label":"{stage_label}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"{method}","backend":"chain_binomial","seed":1,"n_chains":4,"best_loglik":-12.3,"best_chain":1}},"provenance":{{"created_at":"2026-06-22T00:00:0{ord}Z","argv":["camdl","fit","run"]}}}}"#,
            ord = stage_label.len() % 10,
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
    fn fixture_segment(root: &Path, stages: &[(&str, &str, Option<&str>)]) -> PathBuf {
        let seg = root.join("fits").join("toy-abc12345");
        fs::create_dir_all(&seg).unwrap();
        for (label, method, draws) in stages {
            write_stage_leaf(&seg, label, method, *draws);
        }
        // Sidecar with non-empty resolved_priors so a Bayesian segment is
        // treated as well-formed (FitView flags empty priors as a bug).
        let sidecar = crate::run_meta::FitSidecar {
            estimated: vec!["beta".into()],
            resolved_priors: vec![crate::run_meta::ResolvedPriorEntry {
                param: "beta".into(),
                source: "model_ir".into(),
            }],
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&seg, Path::new("nonexistent.toml"), &sidecar).unwrap();
        seg
    }

    const TWO_DRAWS: &str = "beta\tgamma\n0.5\t0.1\n0.6\t0.12\n";

    #[test]
    fn resolves_terminal_stage_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_ok");
        // An IF2 leaf (no draws) beside a PGAS leaf (draws) — two methods of
        // one problem: the leaf with a cloud wins.
        let seg = fixture_segment(&tmp, &[
            ("if2",  "if2", None),
            ("pgas", "pgas", Some(TWO_DRAWS)),
        ]);
        let r = resolve_posterior_draws(seg.to_str().unwrap(), None).expect("resolves");
        assert_eq!(r.stage, "pgas");
        assert_eq!(r.method, Some(FitAlgorithm::Pgas));
        assert!(r.draws_path.ends_with(DRAWS_FILE));
        assert!(r.draws_path.is_file());
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_no_stage_has_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_none");
        // An optimizer-only fit: no posterior cloud anywhere.
        let seg = fixture_segment(&tmp, &[
            ("if2",      "if2", None),
            ("nl-sbplx", "nl-sbplx", None),
        ]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap_err();
        assert!(err.contains("no stage"), "actionable error, got: {err}");
        assert!(err.contains("--params-only"), "points to the optimizer workflow: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn stage_override_selects_named_stage() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_stagepick");
        let seg = fixture_segment(&tmp, &[
            ("pgas",  "pgas", Some("beta\n0.1\n")),
            ("pmmh",  "pmmh", Some(TWO_DRAWS)),
        ]);
        // Without a selector, the last leaf in label order (pmmh) wins...
        assert_eq!(resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap().stage, "pmmh");
        // ...with the selector naming pgas, that leaf is chosen.
        let r = resolve_posterior_draws(seg.to_str().unwrap(), Some("pgas")).unwrap();
        assert_eq!(r.stage, "pgas");
        assert_eq!(r.method, Some(FitAlgorithm::Pgas));
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_named_stage_has_no_draws() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_namednone");
        let seg = fixture_segment(&tmp, &[
            ("if2",  "if2",  None),
            ("pgas", "pgas", Some(TWO_DRAWS)),
        ]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), Some("if2")).unwrap_err();
        assert!(err.contains("no posterior draws") || err.contains("optimizer"), "got: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn pgas_stage_without_draws_is_not_called_an_optimizer() {
        // gh#343: a Bayesian sampler stage (pgas) that has not written draws.tsv
        // — incomplete or still running — must NOT be framed as an optimizer fit,
        // on either the default (terminal-stage) or the --stage-named path.
        let tmp = crate::test_support::unique_temp_dir("pdraws_pgas_nodraws");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", None)]);

        let err = resolve_posterior_draws(seg.to_str().unwrap(), None).unwrap_err();
        assert!(!err.contains("optimizer"), "a pgas stage is not an optimizer: {err}");
        assert!(err.contains(DRAWS_FILE), "names the missing posterior draws: {err}");

        let err2 = resolve_posterior_draws(seg.to_str().unwrap(), Some("pgas")).unwrap_err();
        assert!(!err2.contains("optimizer"), "named pgas stage is not an optimizer: {err2}");
        assert!(err2.contains(DRAWS_FILE), "named stage names the missing draws: {err2}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_on_unknown_stage_name() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_badstage");
        let seg = fixture_segment(&tmp, &[("pgas", "pgas", Some(TWO_DRAWS))]);
        let err = resolve_posterior_draws(seg.to_str().unwrap(), Some("nope")).unwrap_err();
        assert!(err.contains("no method named 'nope'"), "got: {err}");
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolves_dir_with_draws_directly() {
        // A bare directory holding a draws.tsv (no fit view) resolves via the
        // fallback — for power users who point at a stage dir directly.
        let tmp = crate::test_support::unique_temp_dir("pdraws_direct");
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(DRAWS_FILE), TWO_DRAWS).unwrap();
        let r = resolve_posterior_draws(tmp.to_str().unwrap(), None).expect("resolves");
        assert_eq!(r.draws_path, tmp.join(DRAWS_FILE));
        assert_eq!(r.method, None);
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn errors_when_not_a_directory() {
        let err = resolve_posterior_draws("/no/such/dir/at/all", None).unwrap_err();
        assert!(err.contains("not a fit results directory"), "got: {err}");
    }

    // ─── #273: [fixed] backfill ───────────────────────────────────────────

    #[test]
    fn backfill_fills_absent_never_overwrites_present() {
        let mut draws = vec![
            HashMap::from([("beta".to_string(), 0.5)]),  // gamma absent
            HashMap::from([("beta".to_string(), 0.6), ("gamma".to_string(), 0.2)]), // both present
        ];
        let fixed = vec![("beta".to_string(), 9.9), ("gamma".to_string(), 0.1)];
        let filled = backfill_fixed(&mut draws, &fixed);
        // beta is present in both rows → never overwritten; only gamma in row 0 fills.
        assert_eq!(draws[0]["beta"], 0.5, "a present column wins over [fixed]");
        assert_eq!(draws[0]["gamma"], 0.1, "an absent column is backfilled");
        assert_eq!(draws[1]["beta"], 0.6);
        assert_eq!(draws[1]["gamma"], 0.2, "row 1 already had gamma — untouched");
        assert_eq!(filled.into_iter().collect::<Vec<_>>(), vec!["gamma".to_string()]);
    }

    #[test]
    fn resolve_fixed_reads_sidecar_fixed_from_dir() {
        let tmp = crate::test_support::unique_temp_dir("pdraws_fixed_dir");
        fs::create_dir_all(&tmp).unwrap();
        let sidecar = crate::run_meta::FitSidecar {
            fixed: std::collections::BTreeMap::from([("N0".to_string(), 1000.0), ("sigma".to_string(), 0.11)]),
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&tmp, Path::new("nonexistent.toml"), &sidecar).unwrap();
        let mut got = resolve_fixed_for_backfill(tmp.to_str().unwrap()).expect("reads sidecar");
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![("N0".to_string(), 1000.0), ("sigma".to_string(), 0.11)]);
        fs::remove_dir_all(&tmp).ok();
    }
}
