//! End-to-end driver: load a case, run camdl, summarise, compare.

use crate::compare::{self, CheckResult, Outcome};
use crate::hashing;
use crate::manifest::{CamdlMode, CaseManifest, ExpectedManifest, FixtureManifest, ReferenceSpec, SummarySpec};
use crate::subprocess;
use crate::summarise;
use crate::summary::Summary;
use std::path::{Path, PathBuf};

pub const HARNESS_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug)]
pub struct CaseOutcome {
    pub name: String,
    pub status: Status,
    pub checks: Vec<CheckResult>,
    /// Where per-seed stdout/stderr and summary.tsv were written.
    pub run_dir: PathBuf,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Stale(String),
    Crash(String),
    ToleranceFail,
}

/// Run a single case's fast-path. Returns without invoking reference
/// scripts — staleness checks come first, then camdl runs, then compare.
pub fn run_case(case_dir: &Path) -> anyhow::Result<CaseOutcome> {
    let case: CaseManifest = load_toml(&case_dir.join("case.toml"))?;
    let expected: ExpectedManifest = load_toml(&case_dir.join("expected.toml"))?;
    validate_expected(&expected)?;

    let fixtures_dir = case_dir.join("fixtures");
    let fixture_manifest_path = fixtures_dir.join("MANIFEST.toml");
    let fixture_summary_path = fixtures_dir.join("summary.tsv");

    // Prepare run directory.
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();
    let run_dir = case_dir.join("..").join("..").join("runs").join(&case.name).join(&ts);
    std::fs::create_dir_all(&run_dir)
        .map_err(|e| anyhow::anyhow!("create {}: {}", run_dir.display(), e))?;

    // Staleness checks.
    let fixture: FixtureManifest = load_toml(&fixture_manifest_path)?;
    if let Some(msg) = check_staleness(&case, case_dir, &fixture)? {
        return Ok(CaseOutcome {
            name: case.name.clone(),
            status: Status::Stale(msg),
            checks: vec![],
            run_dir,
        });
    }

    // Camdl runs. Two shapes:
    //   PerSeed (default):  n_seeds separate invocations, each → per-seed
    //                        obs.tsv, summariser walks the dir tree.
    //   BatchReplicated:    one invocation with replicate flags → single
    //                        long-format TSV; same summariser path as the
    //                        reference side.
    let camdl_summary = match case.camdl.mode {
        CamdlMode::PerSeed => {
            let mut runs = Vec::with_capacity(case.camdl.n_seeds);
            for i in 0..case.camdl.n_seeds {
                let seed = case.camdl.seed_base + i as u64;
                let run = subprocess::run_camdl_seed(&case.camdl, case_dir, &run_dir, seed)?;
                if !run.succeeded() {
                    return Ok(CaseOutcome {
                        name: case.name.clone(),
                        status: Status::Crash(format!(
                            "camdl exit={:?} on seed={} (see {})",
                            run.exit_code, run.seed, run.stderr_path.display())),
                        checks: vec![],
                        run_dir,
                    });
                }
                runs.push(run);
            }
            match &case.summary {
                SummarySpec::Prebaked => {
                    return Err(anyhow::anyhow!(
                        "summary.kind = 'prebaked' is for the reference side only; \
                         case.toml must declare a summary spec for camdl's output"));
                }
                SummarySpec::EnsembleStats { .. } => summarise::summarise_runs(&case.summary, &runs)?,
            }
        }
        CamdlMode::BatchReplicated => {
            let batch = subprocess::run_camdl_batch(&case.camdl, case_dir, &run_dir)?;
            if !batch.succeeded() {
                return Ok(CaseOutcome {
                    name: case.name.clone(),
                    status: Status::Crash(format!(
                        "camdl exit={:?} (see {})",
                        batch.exit_code, batch.stderr_path.display())),
                    checks: vec![],
                    run_dir,
                });
            }
            summarise::summarise_long_tsv(&case.summary, &batch.output_tsv,
                &case.camdl.batch_seed_col)?
        }
    };
    camdl_summary.write_tsv(&run_dir.join("camdl_summary.tsv"))?;

    let reference_summary = Summary::read_tsv(&fixture_summary_path)?;

    // Compare.
    let checks = compare::run_all(&expected, &camdl_summary, &reference_summary);
    let any_fail = checks.iter().any(|c| c.outcome == Outcome::Fail);
    let status = if any_fail { Status::ToleranceFail } else { Status::Pass };

    Ok(CaseOutcome { name: case.name.clone(), status, checks, run_dir })
}

fn load_toml<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {}", path.display(), e))?;
    toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("parse {}: {}", path.display(), e))
}

fn validate_expected(m: &ExpectedManifest) -> anyhow::Result<()> {
    // Principle #5: rationale required. Catch at load time so a case
    // author can't merge a check with an empty rationale.
    for (stat, check) in &m.checks {
        if check.rationale.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "expected.toml check '{}' has empty rationale — rationale is required \
                 and must include a Monte Carlo power statement (see proposal #5)",
                stat));
        }
    }
    Ok(())
}

fn check_staleness(
    case: &CaseManifest,
    case_dir: &Path,
    fixture: &FixtureManifest,
) -> anyhow::Result<Option<String>> {
    // 1. harness_version
    if fixture.harness_version != HARNESS_VERSION {
        return Ok(Some(format!(
            "harness version changed: fixture generated by {}, current {}",
            fixture.harness_version, HARNESS_VERSION)));
    }

    // 2. reference_sha
    let current_ref_sha = compute_reference_sha(&case.reference, case_dir)?;
    if current_ref_sha != fixture.reference_sha {
        return Ok(Some(format!(
            "reference script/directory changed:\n  \
             MANIFEST.reference_sha = {} (fixture)\n  \
             current                = {}\n  \
             Re-run with CAMDL_REGEN_EXTERNAL=1 to regenerate.",
            fixture.reference_sha, current_ref_sha)));
    }

    // 3. case_sha (model + params + case.toml + expected.toml)
    let current_case_sha = compute_case_sha(case, case_dir)?;
    if current_case_sha != fixture.case_sha {
        return Ok(Some(format!(
            "model/params/manifest changed since fixture generation:\n  \
             MANIFEST.case_sha = {} (fixture)\n  \
             current           = {}\n  \
             Re-run with CAMDL_REGEN_EXTERNAL=1 to regenerate.",
            fixture.case_sha, current_case_sha)));
    }

    Ok(None)
}

pub fn compute_reference_sha(spec: &ReferenceSpec, case_dir: &Path) -> anyhow::Result<String> {
    match spec {
        ReferenceSpec::Analytical { derivation } => {
            hashing::sha256_file(&case_dir.join(derivation))
        }
        ReferenceSpec::RPomp { run, fingerprint_dir, .. } => {
            let target = fingerprint_dir
                .clone()
                .unwrap_or_else(|| run.parent().unwrap_or_else(|| Path::new(".")).to_path_buf());
            hashing::sha256_dir(&case_dir.join(target))
        }
        ReferenceSpec::PyNumpyro { run, fingerprint_dir }
        | ReferenceSpec::Stan { run, fingerprint_dir } => {
            let target = fingerprint_dir
                .clone()
                .unwrap_or_else(|| run.parent().unwrap_or_else(|| Path::new(".")).to_path_buf());
            hashing::sha256_dir(&case_dir.join(target))
        }
    }
}

/// Invoke the reference script and compute a fresh summary from its
/// output. Writes the new `fixtures/summary.tsv` and `fixtures/MANIFEST.toml`.
/// Returns the new fixture MANIFEST so the caller can proceed to rerun
/// the fast path against the fresh fixture.
pub fn regen_case(case_dir: &Path) -> anyhow::Result<FixtureManifest> {
    let case: CaseManifest = load_toml(&case_dir.join("case.toml"))?;

    let (run_script, ensemble_tsv, seed_col) = match &case.reference {
        ReferenceSpec::Analytical { .. } => {
            return Err(anyhow::anyhow!(
                "case '{}' has reference.kind = 'analytical'; regen is manual \
                 (edit reference/derivation.md and fixtures/summary.tsv together, \
                 then `external-harness bootstrap --write`).",
                case.name));
        }
        ReferenceSpec::RPomp { run, ensemble_tsv, seed_col, .. } => {
            (run.clone(), ensemble_tsv.clone(), seed_col.clone())
        }
        ReferenceSpec::PyNumpyro { .. } | ReferenceSpec::Stan { .. } => {
            return Err(anyhow::anyhow!(
                "regen for this reference kind not yet wired (the ensemble_tsv \
                 / seed_col fields live on RPomp for v1; generalise when we \
                 actually add a numpyro case)"));
        }
    };

    // Prepare a timestamped run directory for the reference log.
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();
    let run_dir = case_dir.join("..").join("..").join("runs").join(&case.name).join(format!("regen-{}", ts));
    std::fs::create_dir_all(&run_dir)
        .map_err(|e| anyhow::anyhow!("create {}: {}", run_dir.display(), e))?;

    eprintln!("running reference: {}", run_script.display());
    let log_path = run_dir.join("reference.log");
    let status = crate::subprocess::run_reference(&case_dir.join(&run_script), case_dir, &log_path)?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "reference script exited {:?}; log: {}",
            status.code(), log_path.display()));
    }

    // Read the ensemble and summarise.
    let ensemble_path = case_dir.join(&ensemble_tsv);
    if !ensemble_path.exists() {
        return Err(anyhow::anyhow!(
            "reference script did not produce expected ensemble TSV at {}",
            ensemble_path.display()));
    }
    let (summary, n_seeds_reference) = crate::summarise::summarise_long_tsv_with_seed_count(
        &case.summary, &ensemble_path, &seed_col)?;

    let fixture_summary_path = case_dir.join("fixtures").join("summary.tsv");
    summary.write_tsv(&fixture_summary_path)?;

    // Rewrite MANIFEST with fresh hashes and provenance.
    let reference_sha = compute_reference_sha(&case.reference, case_dir)?;
    let case_sha = compute_case_sha(&case, case_dir)?;
    let fixture_sha = hashing::sha256_file(&fixture_summary_path)?;

    let fm = FixtureManifest {
        reference_sha, case_sha,
        harness_version: HARNESS_VERSION.to_string(),
        fixture_sha,
        pomp_version: detect_pomp_version(case_dir),
        r_version: detect_r_version(),
        python_version: None,
        generated_at: chrono::Utc::now().to_rfc3339(),
        generated_on: Some(format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)),
        generated_command: Some(format!("external-harness regen (via {})", run_script.display())),
        generated_in_docker: std::env::var("CAMDL_EXTERNAL_USE_DOCKER")
            .ok().is_some_and(|v| v == "1"),
        n_seeds_reference,
        seed_base: case.camdl.seed_base,
    };
    let manifest_path = case_dir.join("fixtures").join("MANIFEST.toml");
    let text = toml::to_string_pretty(&fm)
        .map_err(|e| anyhow::anyhow!("serialize MANIFEST: {}", e))?;
    std::fs::write(&manifest_path, text)
        .map_err(|e| anyhow::anyhow!("write {}: {}", manifest_path.display(), e))?;
    eprintln!("wrote {}", manifest_path.display());
    Ok(fm)
}

fn detect_pomp_version(case_dir: &Path) -> Option<String> {
    // Read from renv.lock if present — cheap string search for the pomp
    // package version, avoiding a full JSON parse dep.
    let renv = case_dir.join("reference").join("renv.lock");
    let text = std::fs::read_to_string(&renv).ok()?;
    // Look for `"pomp": { ... "Version": "X.Y" ...}`. Small state machine.
    let bytes = text.as_bytes();
    let needle = b"\"pomp\"";
    let mut i = 0;
    while i + needle.len() < bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            // Find the next `"Version"`.
            let rest = &text[i..];
            if let Some(v_idx) = rest.find("\"Version\"") {
                let after = &rest[v_idx + "\"Version\"".len()..];
                if let Some(q1) = after.find('"') {
                    let after2 = &after[q1 + 1..];
                    if let Some(q2) = after2.find('"') {
                        return Some(after2[..q2].to_string());
                    }
                }
            }
            break;
        }
        i += 1;
    }
    None
}

fn detect_r_version() -> Option<String> {
    let out = std::process::Command::new("Rscript")
        .arg("-e")
        .arg("cat(paste(R.version$major, R.version$minor, sep='.'))")
        .output().ok()?;
    if !out.status.success() { return None; }
    String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
}

pub fn compute_case_sha(case: &CaseManifest, case_dir: &Path) -> anyhow::Result<String> {
    let model_path = case_dir.join(&case.camdl.model);
    let case_toml_path = case_dir.join("case.toml");
    let expected_toml_path = case_dir.join("expected.toml");
    let params_path = case.camdl.params.as_ref().map(|p| case_dir.join(p));

    let mut paths: Vec<PathBuf> = vec![model_path, case_toml_path, expected_toml_path];
    if let Some(p) = params_path {
        if p.exists() { paths.push(p); }
    }
    let refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    hashing::sha256_files(&refs)
}
