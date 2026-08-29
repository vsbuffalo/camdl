//! Spatial polio AFP + ES multi-cadence fixture — forward-sim smoke.
//!
//! Fixture for the multi-stream multi-cadence union-axis lift
//! (docs/dev/proposals/2026-06-10-multi-stream-multi-cadence-union-axis.md §6).
//!
//! The model `tests/fixtures/polio_afp_es_2patch.camdl` declares TWO per-patch
//! stratified observation streams on DIFFERENT cadences:
//!   - `afp[p in patch]` — monthly (every 30 d) incidence of the `paralysis`
//!     flow, neg_binomial, low/zero-heavy counts;
//!   - `es[p in patch]`  — biweekly (every 14 d) prevalence of `I_shed`, poisson.
//!
//! What this pins (Phase 2b opens the gate — `bind` merges the per-stream
//! schedules to the union axis and the per-stream incidence reset scores each
//! stream over its own cadence):
//!   1. the model SIMULATES and `--obs-only-dir` emits one TSV per stratum leaf,
//!      AFP on a 30-day grid and ES on a 14-day grid (two distinct cadences);
//!   2. each source's long-form file (`time, patch, <scored>`) LOADS through the
//!      §4.2 long-form router and scores a finite loglik (rows routed by name);
//!   3. binding BOTH sources at once (AFP monthly + ES biweekly) now FITS — the
//!      heterogeneous-cadence gate is open; the union axis carries both cadences
//!      and the filter scores a finite loglik over all four stratum leaves;
//!   4. the recover-known-params end-to-end fit (`camdl fit run` IF2 scout on the
//!      committed multi-cadence data) recovers the well-identified transmission +
//!      reporting params within a Monte-Carlo tolerance.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn fixture(rel: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../tests/fixtures").join(rel)
}

/// The dune-built compiler. We compile the `.camdl` fixture to IR with THIS
/// binary (not whatever `camdlc` is on PATH — a stale `~/.local/bin/camdlc`
/// predates the §4.2 stratified-observation header and rejects the model),
/// then run `camdl` on the `.ir.json`. Mirrors `fit_sparse_holes.rs`.
fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} — run `make build-ocaml`", p.display());
    p
}

/// Compile the fixture model to IR in `dir` and return the `.ir.json` path.
fn compile_model(dir: &Path) -> PathBuf {
    let ir = dir.join("polio_afp_es_2patch.ir.json");
    let out = Command::new(camdlc_bin())
        .arg(fixture("polio_afp_es_2patch.camdl"))
        .output()
        .expect("spawn camdlc");
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

fn params() -> PathBuf {
    fixture("polio_afp_es_2patch.params.toml")
}

/// Read a one-value-per-leaf wide TSV (`time\t<col>`), returning the time column.
fn read_times(path: &Path) -> Vec<f64> {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    txt.lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .next()
                .unwrap()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("time parse in {}: {l:?}", path.display()))
        })
        .collect()
}

/// Pivot the per-leaf wide TSVs (`{source}_{level}.tsv`, cols `time, {col}`)
/// into one long-form file (`time, patch, {scored}`) — the shape the §4.2
/// long-form fit loader consumes. (simulate emits per-leaf wide today; a
/// stratified `: dim` stream loads long-form. This bridges the two.)
fn pivot_long_form(obs_dir: &Path, out: &Path, source: &str, scored: &str, levels: &[&str]) {
    let mut rows: Vec<(f64, String, String)> = Vec::new();
    for &lvl in levels {
        let leaf = obs_dir.join(format!("{source}_{lvl}.tsv"));
        let txt = std::fs::read_to_string(&leaf)
            .unwrap_or_else(|e| panic!("read leaf {}: {e}", leaf.display()));
        for line in txt.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let mut it = line.split('\t');
            let t: f64 = it.next().unwrap().parse().unwrap();
            let v = it.next().unwrap().to_string();
            rows.push((t, lvl.to_string(), v));
        }
    }
    // Sort by (time, patch) for a stable, chronological long-form file.
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut body = format!("time\tpatch\t{scored}\n");
    for (t, p, v) in rows {
        // Emit integer times without a trailing .0 (matches the simulator).
        body.push_str(&format!("{}\t{p}\t{v}\n", t as i64));
    }
    std::fs::write(out, body).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}

/// Generate the synthetic data into `dir`: simulate at truth, then pivot to
/// long-form per-source files. Returns (afp_long, es_long) paths.
fn generate_data(bin: &Path, ir: &Path, dir: &Path) -> (PathBuf, PathBuf) {
    let obs_dir = dir.join("obs");
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "simulate", &ir.to_string_lossy(),
            "--params", &params().to_string_lossy(),
            "--backend", "chain_binomial", "--dt", "1", "--seed", "1",
            "--obs-only-dir", &obs_dir.to_string_lossy(),
            "--output-dir", &dir.join("results").to_string_lossy(),
        ])
        .output()
        .expect("spawn simulate");
    assert!(
        out.status.success(),
        "simulate failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Property 1: one TSV per stratum leaf, AFP monthly + ES biweekly.
    let afp_u = read_times(&obs_dir.join("afp_urban.tsv"));
    let afp_r = read_times(&obs_dir.join("afp_rural.tsv"));
    let es_u = read_times(&obs_dir.join("es_urban.tsv"));
    let es_r = read_times(&obs_dir.join("es_rural.tsv"));

    // Both AFP leaves share a 30-day grid; both ES leaves share a 14-day grid.
    assert_eq!(afp_u, afp_r, "AFP leaves must share one cadence");
    assert_eq!(es_u, es_r, "ES leaves must share one cadence");
    assert_eq!(afp_u[0], 0.0);
    assert_eq!(afp_u[1] - afp_u[0], 30.0, "AFP is monthly (every 30 days)");
    assert_eq!(es_u[1] - es_u[0], 14.0, "ES is biweekly (every 14 days)");
    // Distinct cadences → distinct grids: AFP at 30 is not an ES time.
    assert!(!es_u.contains(&30.0) || es_u.contains(&28.0),
        "AFP (30d) and ES (14d) grids must differ");

    let afp = dir.join("afp.tsv");
    let es = dir.join("es.tsv");
    pivot_long_form(&obs_dir, &afp, "afp", "cases", &["urban", "rural"]);
    pivot_long_form(&obs_dir, &es, "es", "conc", &["urban", "rural"]);
    (afp, es)
}

fn run_pfilter(bin: &Path, ir: &Path, data_args: &[String]) -> (bool, String, String) {
    let mut args = vec![
        "pfilter".to_string(),
        ir.to_string_lossy().into_owned(),
        "--params".to_string(),
        params().to_string_lossy().into_owned(),
        "--particles".to_string(),
        "300".to_string(),
        "--dt".to_string(),
        "1".to_string(),
        "--seed".to_string(),
        "1".to_string(),
    ];
    args.extend_from_slice(data_args);
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(&args)
        .output()
        .expect("spawn pfilter");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn simulates_two_streams_at_distinct_cadences_and_each_source_loads() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = compile_model(tmp.path());
    let (afp, es) = generate_data(&bin, &ir, tmp.path());

    // Property 2a: the AFP source (monthly) loads long-form → finite loglik.
    let (ok_afp, out_afp, err_afp) =
        run_pfilter(&bin, &ir, &[format!("--data=afp={}", afp.to_string_lossy())]);
    assert!(ok_afp, "AFP pfilter failed:\nstdout={out_afp}\nstderr={err_afp}");
    assert!(err_afp.contains("afp_urban") && err_afp.contains("afp_rural"),
        "both AFP leaves must bind from the long-form file: {err_afp}");
    let ll_afp: f64 = out_afp.trim().parse()
        .unwrap_or_else(|_| panic!("AFP loglik parse: {out_afp:?}"));
    assert!(ll_afp.is_finite() && ll_afp < 0.0, "AFP loglik must be finite: {ll_afp}");

    // Property 2b: the ES source (biweekly) loads long-form → finite loglik.
    let (ok_es, out_es, err_es) =
        run_pfilter(&bin, &ir, &[format!("--data=es={}", es.to_string_lossy())]);
    assert!(ok_es, "ES pfilter failed:\nstdout={out_es}\nstderr={err_es}");
    assert!(err_es.contains("es_urban") && err_es.contains("es_rural"),
        "both ES leaves must bind from the long-form file: {err_es}");
    let ll_es: f64 = out_es.trim().parse()
        .unwrap_or_else(|_| panic!("ES loglik parse: {out_es:?}"));
    assert!(ll_es.is_finite() && ll_es < 0.0, "ES loglik must be finite: {ll_es}");
}

#[test]
fn binding_both_cadences_now_fits() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let ir = compile_model(tmp.path());
    let (afp, es) = generate_data(&bin, &ir, tmp.path());

    // Property 3 (Phase 2b — the opened gate): AFP (monthly) + ES (biweekly)
    // bound TOGETHER now load and score. `bind` merges the two cadences to the
    // union observation axis; the per-stream incidence reset closes each stream's
    // bin on its own schedule. The old heterogeneous-cadence rejection ("must
    // share identical observation times") is gone — this pins the opened gate.
    // (proposal 2026-06-10-multi-stream-multi-cadence-union-axis.md §3.3, Phase 2b.)
    let (ok, out, err) = run_pfilter(
        &bin,
        &ir,
        &[
            format!("--data=afp={}", afp.to_string_lossy()),
            format!("--data=es={}", es.to_string_lossy()),
        ],
    );
    assert!(ok,
        "two streams on different cadences must now FIT (Phase 2b opened the gate):\n\
         stdout={out}\nstderr={err}");
    assert!(!err.contains("identical observation times"),
        "the heterogeneous-cadence rejection must be gone: {err}");
    // All four stratum leaves bind (both sources, both patches).
    assert!(
        err.contains("afp_urban") && err.contains("afp_rural")
            && err.contains("es_urban") && err.contains("es_rural"),
        "all four leaves must bind from the two long-form sources: {err}");
    let ll: f64 = out.trim().parse()
        .unwrap_or_else(|_| panic!("multi-cadence loglik parse: {out:?}"));
    assert!(ll.is_finite() && ll < 0.0,
        "the merged multi-cadence loglik must be finite and negative: {ll}");
}

/// Read a `mle_params.toml` value by key (`<name> = <float>` lines in the
/// leading bare-key section, before the `[provenance]` table). Returns `None`
/// if the key is absent. (The header comment mentions "[provenance]" in prose,
/// so we stop at the first real TOML table header line, not a substring match.)
fn read_mle_param(mle_toml: &str, key: &str) -> Option<f64> {
    for line in mle_toml.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        if line.starts_with('[') { break; } // first table header → end of bare keys
        if let Some((name, val)) = line.split_once('=') {
            if name.trim() == key {
                return val.trim().parse::<f64>().ok();
            }
        }
    }
    None
}

/// Recover-known-params end-to-end fit — the Phase 2b proof that the
/// heterogeneous multi-cadence path FITS, not just loads.
///
/// Runs the IF2 scout (`--stage scout`) of `polio_afp_es/fit.toml` on the
/// committed multi-cadence data (AFP monthly + ES biweekly long-form files) and
/// asserts the fitted point estimate (`mle_params.toml`, the best chain) lands
/// near the truth for the WELL-IDENTIFIED parameters: the two transmission rates
/// (R0_urban, R0_rural) and the AFP reporting fraction (rho).
///
/// Scope + tolerance (flagged deliberately): this runs the IF2 SCOUT only (the
/// PGAS posterior stage is the slow part and is not needed to prove the gate is
/// open). From a SINGLE synthetic realization (gen_data.sh seed 1) the scout's
/// likelihood surface is multimodal, so the shedding / spatial-coupling / ES
/// parameters (delta, kappa, gamma, lambda) are only weakly identified and are
/// NOT asserted — asserting them would be a flaky test that fails for the wrong
/// reason. R0 (±25%) and rho (±20%) are the robustly identifiable signals; the
/// θ̂ is content-addressed and seed-pinned, so the check is deterministic.
#[test]
#[ignore = "costly: ~85s 4-chain, 2000-particle, 50-iteration IF2 scout recovery \
            fit — opt-in via `cargo test -- --ignored` (run before releases / in \
            nightly CI), not in the default `make test` gate"]
fn synthetic_fit_recovers_params() {
    let bin = skip_if_missing_binary();
    // Absolute toml path: every path inside the toml resolves against the
    // toml's own directory (gh#22, and `output_dir` too as of gh#507), so they
    // bind correctly regardless of CWD. The fixture therefore declares NO
    // `output_dir` — one would put the run tree inside the committed fixture
    // directory — and the destination is pinned here with CAMDL_OUTPUT_DIR,
    // the layer that applies when no `output_dir` is declared. Pinning it
    // explicitly also keeps the test from inheriting a developer's ambient
    // CAMDL_OUTPUT_DIR.
    let fit_toml = fixture("polio_afp_es/fit.toml");
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(&bin)
        .current_dir(tmp.path())
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        // Compile the model with the dune-built camdlc (a stale PATH camdlc
        // predates the stratified-observation header), mirroring compile_model.
        .env("CAMDLC", camdlc_bin())
        .args([
            "fit", "run", &fit_toml.to_string_lossy(),
            "--stage", "scout",
            "--allow-nonconverged-scout",
        ])
        .output()
        .expect("spawn fit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "multi-cadence IF2 scout fit failed:\nstderr={stderr}",
    );
    // Proof the heterogeneous union path actually ran: all four leaves bound.
    assert!(
        stderr.contains("4 observation streams")
            && stderr.contains("afp_urban") && stderr.contains("es_urban"),
        "the scout must have run over all four multi-cadence leaves:\n{stderr}",
    );

    // Locate the stored best-chain point estimate (`mle_params.toml`).
    let mle_path = {
        let mut found = None;
        for entry in walk_files(&tmp.path().join("results")) {
            if entry.file_name().is_some_and(|n| n == "mle_params.toml") {
                found = Some(entry);
                break;
            }
        }
        found.unwrap_or_else(|| panic!(
            "no mle_params.toml under {}/results — fit did not store a point estimate",
            tmp.path().display()))
    };
    let mle = std::fs::read_to_string(&mle_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", mle_path.display()));

    // Truth (../polio_afp_es_2patch.params.toml) for the well-identified params.
    let checks = [
        ("R0_urban", 3.0, 0.25),
        ("R0_rural", 2.2, 0.25),
        ("rho", 0.6, 0.20),
    ];
    for (name, truth, rel_tol) in checks {
        let est = read_mle_param(&mle, name)
            .unwrap_or_else(|| panic!("{name} missing from mle_params.toml:\n{mle}"));
        let rel_err = (est - truth).abs() / truth;
        assert!(
            rel_err <= rel_tol,
            "{name}: θ̂ = {est:.4}, truth = {truth}, rel.err = {:.1}% > tol {:.0}%\n\
             (multi-cadence recovery — IF2 scout on the committed AFP+ES data)",
            rel_err * 100.0, rel_tol * 100.0,
        );
    }
}

/// Recursively list files under `root` (depth-first). Empty if `root` is absent.
fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); } else { out.push(p); }
        }
    }
    out
}
