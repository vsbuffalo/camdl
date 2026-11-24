//! Provenance hashing: input_hash (computation identity) and content_hash (tamper detection).

use sha2::{Sha256, Digest};

use score::convergence::ChainIndex;
use crate::version;

// `compute_input_hash` moved to `crate::hashing::fit_input_hash`
// as part of the 2026-04-19 output-tree unification. Call sites
// updated in lockstep.

/// MLE-params tamper-detection hash. Hashes the canonicalised (name,
/// value) pairs from an `mle_params.toml` file so we can detect post-
/// write edits to the parameter values. Naming note: this is **not**
/// a general content-hash helper (unlike `crate::hashing::sha256_hex`); it's
/// mle-specific because it canonicalises numeric formatting with `{:.12}` precision before hashing. Lives in
/// `fit::provenance` (not `crate::hashing`) for that reason.
pub fn mle_params_tamper_hash(params: &std::collections::BTreeMap<String, f64>) -> String {
    let mut pairs: Vec<(&String, &f64)> = params.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    let mut h = Sha256::new();
    for (name, value) in pairs {
        h.update(format!("{}={:.12}\x00", name, value).as_bytes());
    }
    hex::encode(&h.finalize()[..4]) // 8 hex chars
}

/// The `[provenance]` block of a params file a fit wrote: an optimizer's
/// `mle_params.toml`, or what `fit summary --point` prints for a sampler run.
/// `camdl simulate --params` reads `backend` and `dt` back and replays the
/// values on that simulator
/// (`docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`).
///
/// Only fields that are `Some` get serialized; legacy fit-produced
/// files may omit any of the optional fields.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MleProvenance {
    pub camdl_version: String,
    /// Tamper-detection hash over the numeric parameter values in the
    /// file below (NOT over this `[provenance]` block). Editing a
    /// parameter invalidates this; editing a provenance field does
    /// not.
    pub content_hash: String,
    /// Full fit-level hash of the originating fit, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_hash: Option<String>,
    /// Dynamics fields — load-bearing for the backend-guardrail.
    pub backend: crate::args::types::ForwardBackend,
    pub dt: f64,
    /// The model the fit ran on, by content. Where its file was is not
    /// recorded here: this file is part of the leaf's stored bytes, which are a
    /// function of the computation, and a path relative to the store changes
    /// with where the store is. The fit folder's `fit.meta.json` records the
    /// model's file (`source_model`).
    pub model_identity: String,
    pub seed: u64,
    /// The method that produced this estimate, by the label of its `method`
    /// store level (`if2`, `nl-sbplx`, `pmmh`, …) — the same word the config,
    /// the store path and the CLI use (gh#890).
    pub method: String,
    /// What the values are: one point the method found, or a statistic of its
    /// posterior draws. Written as the variant's fields, at this level.
    #[serde(flatten)]
    pub estimate: Estimate,
}

/// What the parameter values of a fit-written params file are. The block
/// carries no tag: a winner's fields begin with `chain`, a posterior
/// summary's with `statistic`, and a reader tries the winner first.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Estimate {
    /// An optimizer's best point (IF2, NLopt).
    Winner(WinnerRecord),
    /// A statistic of a sampler's posterior draws (MH, PMMH, PGAS), taken per
    /// parameter on its natural scale.
    PosteriorSummary(PosteriorSummaryRecord),
}

/// The point an optimizer run found, with the log-likelihood it was scored at.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WinnerRecord {
    /// The chain whose estimate this is — the number of its `chain_N/`
    /// directory and of `fit_state.toml`'s `best_chain` (gh#990).
    pub chain: ChainIndex,
    pub log_likelihood: f64,
    /// Monte Carlo standard error of `log_likelihood`: that of the re-score
    /// that produced it (IF2), `0.0` for a deterministic likelihood (NLopt on
    /// the ODE backend).
    pub loglik_sd: f64,
    /// Particles per filter replicate of that re-score; `0` when no particle
    /// filter scored it.
    pub n_particles: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ess_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ess_min: Option<f64>,
}

/// A per-parameter statistic of a sampler run's posterior draws. Each
/// estimated parameter's value is the statistic of its column of the run's
/// `draws.tsv`; each fixed parameter's is the value the fit held it at. The
/// values are not a point any chain visited, and no log-likelihood was
/// computed at them.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PosteriorSummaryRecord {
    pub statistic: crate::args::PointStatistic,
    /// The posterior draws the statistic was taken over: every row of the
    /// run's `draws.tsv`, all chains pooled.
    pub n_draws: usize,
    /// The fit run whose draws these are, by its full run id.
    pub run_id: String,
}

/// The text of a fit-written params file: `header` as comment lines, the
/// parameter values as top-level `name = value` lines in name order, then the
/// `[provenance]` block `provenance` builds from the file's `content_hash`.
///
/// The hash covers the values as the file holds them. `format_param_value`
/// writes ten decimals, so a hash over the unrounded `f64` would declare a
/// value no reader of the file can recompute, and a check against it would
/// call every real fit's file edited.
///
/// The values come before the block: a bare `key = value` line that follows
/// a `[table]` header parses as a field of that table, so values written
/// after `[provenance]` would be read as fields of it by
/// `util::load_params_toml` and by any other TOML reader.
pub fn params_file_text(
    header: &[&str],
    params: &std::collections::BTreeMap<String, f64>,
    provenance: impl FnOnce(String) -> MleProvenance,
) -> Result<String, String> {
    let as_written: std::collections::BTreeMap<String, f64> = params
        .iter()
        .map(|(k, v)| {
            let written = crate::fit::runner::format_param_value(*v);
            written
                .parse::<f64>()
                .map(|read_back| (k.clone(), read_back))
                .map_err(|_| format!("`{}` formats to {:?}, which is not a number", k, written))
        })
        .collect::<Result<_, String>>()?;
    let prov = provenance(mle_params_tamper_hash(&as_written));
    let mut text = String::new();
    for line in header {
        text.push_str(&format!("# {line}\n"));
    }
    for (name, value) in params {
        text.push_str(&format!("{} = {}\n", name, crate::fit::runner::format_param_value(*value)));
    }
    text.push('\n');
    #[derive(serde::Serialize)]
    struct ProvWrapper<'a> {
        provenance: &'a MleProvenance,
    }
    let block = toml::to_string(&ProvWrapper { provenance: &prov })
        .map_err(|e| format!("cannot serialize provenance: {}", e))?;
    text.push_str(&block);
    Ok(text)
}

/// Write an optimizer's winner, `all_params`, to `path` as `mle_params.toml`:
/// the values, then a `[provenance]` block ([`params_file_text`]).
pub fn write_mle_params(
    path: &str,
    all_params: &std::collections::BTreeMap<String, f64>,
    metadata: &MleMetadata,
) -> Result<(), String> {
    use std::io::Write;
    let header = [
        "camdl fit MLE — parameter values first, followed by a [provenance] block.",
        "Editing any value in this params section invalidates provenance.content_hash.",
    ];
    let text = params_file_text(&header, all_params, |content_hash| MleProvenance {
        camdl_version: version::VERSION_SHORT.to_string(),
        content_hash,
        fit_hash: Some(metadata.input_hash.clone()),
        backend: metadata.backend,
        dt: metadata.dt,
        model_identity: metadata.model_identity.clone(),
        seed: metadata.seed,
        method: metadata.method.clone(),
        estimate: Estimate::Winner(WinnerRecord {
            chain: metadata.best_chain,
            log_likelihood: metadata.loglik,
            loglik_sd: metadata.loglik_sd,
            n_particles: metadata.n_particles,
            ess_mean: metadata.ess_at_mle.map(|(m, _)| m),
            ess_min: metadata.ess_at_mle.map(|(_, mn)| mn),
        }),
    })
    .map_err(|e| format!("cannot write {}: {}", path, e))?;
    let mut f = std::fs::File::create(path).map_err(|e| format!("cannot write {}: {}", path, e))?;
    f.write_all(text.as_bytes()).map_err(|e| format!("cannot write {}: {}", path, e))
}

/// Read the `[provenance]` block from an mle_params.toml file. Returns
/// None if the file has no such block (legacy format or a non-fit
/// params file); Err on malformed TOML.
#[expect(clippy::disallowed_methods, reason = "reads a params file a command names with `--params` (declared `#[input(identity)]` on `ModelOverrides`) or `--fixed-file` (declared on `InferenceModelOverrides`), for its `[provenance]` block")]
pub fn read_mle_provenance(path: &str) -> Result<Option<MleProvenance>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        #[serde(default)]
        provenance: Option<MleProvenance>,
    }
    let w: Wrapper = toml::from_str(&contents)
        .map_err(|e| format!("parse error in {}: {}", path, e))?;
    Ok(w.provenance)
}

/// Whether the parameter values in a fit-written params file still hash to
/// the `content_hash` its `[provenance]` block declares.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentVerification {
    Valid,
    /// The values were edited after the block was written: `declared` is the
    /// block's hash, `computed` the hash of the values now in the file.
    Modified { declared: String, computed: String },
}

/// Recompute the tamper hash over the file's top-level numeric parameters —
/// the values `write_mle_params` hashed into `content_hash` — and compare it
/// with `declared`, the hash of the `[provenance]` block the caller has read.
/// The scope is the parameter values only: editing a provenance field does
/// not invalidate the hash, editing a value does. A `[provenance]` table is
/// skipped exactly as `util::load_params_toml` skips it.
#[expect(clippy::disallowed_methods, reason = "reads a params file a command names with `--params` (declared `#[input(identity)]` on `ModelOverrides`) or `--fixed-file` (declared on `InferenceModelOverrides`), to check its values against its `[provenance]` block")]
pub fn verify_content_hash(path: &str, declared: &str) -> Result<ContentVerification, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let table: toml::Table = contents.parse()
        .map_err(|e| format!("parse error in {}: {}", path, e))?;
    let params: std::collections::BTreeMap<String, f64> = table
        .into_iter()
        .filter(|(k, _)| k != "provenance")
        .filter_map(|(k, v)| match v {
            toml::Value::Float(f) => Some((k, f)),
            toml::Value::Integer(i) => Some((k, i as f64)),
            _ => None,
        })
        .collect();
    let computed = mle_params_tamper_hash(&params);
    Ok(if computed == declared {
        ContentVerification::Valid
    } else {
        ContentVerification::Modified { declared: declared.to_string(), computed }
    })
}

pub struct MleMetadata {
    pub input_hash: String,
    pub model_identity: String,
    pub seed: u64,
    /// The method label, written to `[provenance] method`.
    pub method: String,
    /// The winning chain — written to `[provenance] chain` as it is, the same
    /// number as [`crate::fit::state::FitState::best_chain`] and the
    /// `chain_N/` directory (gh#912, gh#990).
    pub best_chain: ChainIndex,
    /// Simulation backend the fit used. Load-bearing for the
    /// backend-provenance guardrail in `camdl simulate --params`
    /// — downstream can only auto-match if we record this.
    pub backend: crate::args::types::ForwardBackend,
    /// Timestep used by the fit. Paired with `backend`.
    pub dt: f64,
    pub loglik: f64,
    /// See [`MleProvenance::loglik_sd`].
    pub loglik_sd: f64,
    /// See [`MleProvenance::n_particles`].
    pub n_particles: usize,
    pub ess_at_mle: Option<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#890: the `[provenance]` block names the method by the word every
    /// other surface uses. The file is machine-read — `camdl simulate
    /// --params` matches the backend off this block — so the key is a
    /// contract, and after the `[stages]` → `[method]` split `stage` named a
    /// concept the config, the store path and the CLI no longer have.
    #[test]
    fn the_provenance_block_names_the_method_not_a_stage() {
        let dir = crate::test_support::unique_temp_dir("mle_params_provenance");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.3)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 1,
            method: "if2".into(),
            best_chain: ChainIndex(0),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
        };
        write_mle_params(&path.to_string_lossy(), &params, &meta).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("method = \"if2\""),
            "the block must name the method:\n{text}");
        assert!(!text.contains("stage = "),
            "and must not spell it `stage`, which no other surface does:\n{text}");
        // Read back through the reader that consumers use, so the rename is
        // pinned on both sides of the round trip.
        let prov = read_mle_provenance(&path.to_string_lossy()).unwrap().unwrap();
        assert_eq!(prov.method, "if2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The block records the model by content and the data not at all: the
    /// file is a result file, whose bytes are a function of the run's
    /// identity, and the identity holds the training lines and the held-out
    /// dates, not a data file's path or whole-file digest. The same fit
    /// written from two directories writes the same block.
    #[test]
    fn the_provenance_block_records_no_path() {
        let root = crate::test_support::unique_temp_dir("mle_params_no_path");
        let params: std::collections::BTreeMap<String, f64> = [("beta".to_string(), 0.3)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 1,
            method: "if2".into(),
            best_chain: ChainIndex(0),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
        };
        let mut written = Vec::new();
        for sub in ["a/results/fits/sir-abcd1234/if2-0/seed_1-0", "b/deeper/results/fits/sir-abcd1234/if2-0/seed_1-0"] {
            let leaf = root.join(sub);
            std::fs::create_dir_all(&leaf).unwrap();
            let path = leaf.join("mle_params.toml");
            write_mle_params(&path.to_string_lossy(), &params, &meta).unwrap();
            written.push(std::fs::read_to_string(&path).unwrap());
        }
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(written[0], written[1], "the block depends on where the fit ran");
        assert!(!written[0].contains("source") && !written[0].contains("timestamp"), "{}", written[0]);
        assert!(!written[0].contains("[provenance.data"), "{}", written[0]);
    }

    /// The file a fit writes verifies; a value edited afterwards is reported
    /// against the declared hash; an edited provenance field is not a value
    /// edit.
    #[test]
    fn verify_content_hash_tells_an_edited_value_from_an_unedited_file() {
        let dir = crate::test_support::unique_temp_dir("mle_params_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let path_s = path.to_string_lossy().into_owned();
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 1,
            method: "if2".into(),
            best_chain: ChainIndex(0),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
        };
        write_mle_params(&path_s, &params, &meta).unwrap();
        let prov = read_mle_provenance(&path_s).unwrap().unwrap();
        assert_eq!(
            verify_content_hash(&path_s, &prov.content_hash).unwrap(),
            ContentVerification::Valid,
            "the file a fit writes verifies against its own block"
        );

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("beta = 0.3\n"), "{text}");
        std::fs::write(&path, text.replace("beta = 0.3\n", "beta = 0.35\n")).unwrap();
        match verify_content_hash(&path_s, &prov.content_hash).unwrap() {
            ContentVerification::Modified { declared, computed } => {
                assert_eq!(declared, prov.content_hash);
                assert_eq!(computed, mle_params_tamper_hash(
                    &[("beta".to_string(), 0.35), ("gamma".to_string(), 0.1)].into()));
                assert_ne!(declared, computed);
            }
            other => panic!("an edited value must be reported, got {other:?}"),
        }

        // Only the values are hashed: a provenance field may be corrected.
        assert!(text.contains("model_identity = \"f00d\""), "{text}");
        std::fs::write(&path, text.replace("model_identity = \"f00d\"", "model_identity = \"beef\"")).unwrap();
        assert_eq!(verify_content_hash(&path_s, &prov.content_hash).unwrap(), ContentVerification::Valid);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bytes an optimizer's `mle_params.toml` holds. The file is a result
    /// file of the fit run, so its bytes are a function of the run's identity:
    /// a change to how the block is written changes every stored fit's file.
    #[test]
    fn a_winner_block_is_written_as_pinned() {
        let dir = crate::test_support::unique_temp_dir("mle_params_pinned");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 7,
            method: "if2".into(),
            best_chain: ChainIndex(2),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 0.5,
            loglik: -12.5,
            loglik_sd: 0.25,
            n_particles: 100,
            ess_at_mle: Some((40.0, 12.5)),
        };
        write_mle_params(&path.to_string_lossy(), &params, &meta).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let expected = format!(
            "# camdl fit MLE — parameter values first, followed by a [provenance] block.\n\
             # Editing any value in this params section invalidates provenance.content_hash.\n\
             beta = 0.3\ngamma = 0.1\n\n\
             [provenance]\ncamdl_version = \"{}\"\ncontent_hash = \"88849be8\"\nfit_hash = \"deadbeef\"\n\
             backend = \"chain_binomial\"\ndt = 0.5\nmodel_identity = \"f00d\"\nseed = 7\nmethod = \"if2\"\n\
             chain = 2\nlog_likelihood = -12.5\nloglik_sd = 0.25\nn_particles = 100\n\
             ess_mean = 40.0\ness_min = 12.5\n",
            version::VERSION_SHORT
        );
        assert_eq!(text, expected);
    }

    /// A posterior summary's block reads back as one, a winner's as a winner,
    /// and the summary's values verify against the hash it declares.
    #[test]
    fn a_posterior_summary_block_round_trips() {
        let dir = crate::test_support::unique_temp_dir("posterior_summary_block");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mean.toml");
        let path_s = path.to_string_lossy().into_owned();
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.31234567891234), ("gamma".to_string(), 0.1)].into();
        let summary = Estimate::PosteriorSummary(PosteriorSummaryRecord {
            statistic: crate::args::PointStatistic::PosteriorMean,
            n_draws: 80,
            run_id: "ab".repeat(32),
        });
        let text = params_file_text(&["a header"], &params, |content_hash| MleProvenance {
            camdl_version: "test".into(),
            content_hash,
            fit_hash: Some("cd".repeat(32)),
            backend: crate::args::types::ForwardBackend::Ode,
            dt: 0.25,
            model_identity: "f00d".into(),
            seed: 3,
            method: "pmmh".into(),
            estimate: summary.clone(),
        })
        .unwrap();
        assert!(text.starts_with("# a header\nbeta = 0.3123456789\ngamma = 0.1\n\n[provenance]\n"), "{text}");
        assert!(text.contains("statistic = \"posterior_mean\"\nn_draws = 80\n"), "{text}");
        assert!(!text.contains("chain ="), "a summary names no chain:\n{text}");
        std::fs::write(&path, &text).unwrap();
        let prov = read_mle_provenance(&path_s).unwrap().unwrap();
        assert_eq!(prov.estimate, summary);
        assert_eq!((prov.backend, prov.dt), (crate::args::types::ForwardBackend::Ode, 0.25));
        assert_eq!(verify_content_hash(&path_s, &prov.content_hash).unwrap(), ContentVerification::Valid);

        let winner = dir.join("mle_params.toml");
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 1,
            method: "if2".into(),
            best_chain: ChainIndex(1),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
        };
        write_mle_params(&winner.to_string_lossy(), &params, &meta).unwrap();
        let prov = read_mle_provenance(&winner.to_string_lossy()).unwrap().unwrap();
        assert!(matches!(prov.estimate, Estimate::Winner(WinnerRecord { chain: ChainIndex(1), .. })), "{prov:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn content_hash_stable() {
        let params: std::collections::BTreeMap<String, f64> = [
            ("beta".to_string(), 0.3),
            ("gamma".to_string(), 0.1),
        ].into();
        let h1 = mle_params_tamper_hash(&params);
        let h2 = mle_params_tamper_hash(&params);
        assert_eq!(h1, h2, "same params must produce same hash");
        assert_eq!(h1.len(), 8, "content hash is 8 hex chars");
    }

    #[test]
    fn content_hash_changes_on_value_change() {
        let params1: std::collections::BTreeMap<String, f64> = [("beta".to_string(), 0.3)].into();
        let params2: std::collections::BTreeMap<String, f64> = [("beta".to_string(), 0.31)].into();
        assert_ne!(
            mle_params_tamper_hash(&params1),
            mle_params_tamper_hash(&params2),
            "different values must produce different hashes"
        );
    }

    /// The declared `content_hash` is recomputable from the file: the file
    /// holds `format_param_value`'s ten decimals, so the hash must cover the
    /// values as written, not the unrounded estimate.
    #[test]
    fn content_hash_is_over_the_values_as_written() {
        let dir = crate::test_support::unique_temp_dir("mle_params_as_written");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mle_params.toml");
        let path_s = path.to_string_lossy().into_owned();
        // Thirteen decimals: more than the file carries.
        let params: std::collections::BTreeMap<String, f64> =
            [("beta".to_string(), 0.4321098765432_f64), ("gamma".to_string(), 0.2)].into();
        let meta = MleMetadata {
            input_hash: "deadbeef".into(),
            model_identity: "f00d".into(),
            seed: 1,
            method: "if2".into(),
            best_chain: ChainIndex(0),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            loglik: -12.5,
            loglik_sd: 0.0,
            n_particles: 100,
            ess_at_mle: None,
        };
        write_mle_params(&path_s, &params, &meta).unwrap();
        let declared = read_mle_provenance(&path_s).unwrap().unwrap().content_hash;
        let read_back: std::collections::BTreeMap<String, f64> =
            crate::util::load_params_toml(&path_s).unwrap().into_iter().collect();
        assert_eq!(read_back["beta"], 0.4321098765, "the file holds ten decimals");
        assert_eq!(
            mle_params_tamper_hash(&read_back),
            declared,
            "the declared hash must be the hash of the values the file holds"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
