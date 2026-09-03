//! A PGAS stage publishes the convergence of its latent path: R̂ and ESS of
//! every state at every substep across the chains' saved paths, as
//! `latent_convergence.tsv` (one row per cell) and a `latent_convergence`
//! block in `pgas_summary.json` (the per-bin reduction and the agreement
//! horizon).
//!
//! The in-crate unit tests pin the classification and the bin accounting on
//! hand-built blocks. This pins what a downstream reader finds on disk, and
//! that it has not drifted from the sampler's own record:
//!
//! * the table has exactly one row per (substep, column) of the chains'
//!   `trajectories.tsv`, and every column of that file is assessed;
//! * every `mixed` cell's R̂ and ESS are what `rank_convergence` returns on the
//!   per-chain vectors read back from the chains' `trajectories.tsv` — the
//!   same draws, the same estimator, recomputed by a reader holding only the
//!   artifacts;
//! * every `frozen_disagree` / `constant` cell is one by the estimator's own
//!   rule on those vectors;
//! * the JSON bins are the counts down the table's `status` column, and
//!   `agree_from` is the substep after the table's last `frozen_disagree` row;
//! * the block is additive — everything the summary carried before is still
//!   there.
//!
//! Skipped when the release binary or camdlc isn't present.

use sim::inference::convergence::{rank_convergence, ConvergenceError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_latentconv_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Spelled out rather than imported: this test speaks for a reader who has
/// only the artifacts.
const N_BINS: usize = 10;

const T_END: usize = 40;
const SWEEPS: usize = 40;
const BURN_IN: usize = 10;
const CHAINS: usize = 3;

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = {T_END} 'days }}
"#);
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let mut data = String::from("time\tcases\n");
    for t in 1..=T_END {
        let x = t as f64;
        let n = (60.0 * (-((x - 18.0) / 9.0).powi(2)).exp()).round() as i64 + 4;
        data.push_str(&format!("{t}\t{n}\n"));
    }
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path, data).unwrap();
    (ir_path, data_path)
}

/// `thin = 1` and the default `n_trajectories` (200) exceeds the retained
/// count, so every retained sweep saves a path and the per-chain vectors the
/// stage reduced are exactly the ones in `trajectories.tsv`.
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path) -> (PathBuf, PathBuf) {
    let out = dir.join("results");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = {CHAINS}
particles = 40
sweeps = {SWEEPS}
burn_in = {BURN_IN}
thin = 1
"#,
        out = out.display(), ir = ir.display(), data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    (p, out)
}

fn stage_leaf(out: &Path) -> PathBuf {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return d;
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

/// A TSV as header + rows, skipping `#` provenance lines.
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let header: Vec<String> =
        lines.next().expect("header").split('\t').map(String::from).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split('\t').map(String::from).collect::<Vec<_>>())
        .collect();
    for r in &rows {
        assert_eq!(r.len(), header.len(), "{}: row width must match the header", path.display());
    }
    (header, rows)
}

fn col<'a>(header: &[String], row: &'a [String], name: &str) -> &'a str {
    let i = header.iter().position(|c| c == name)
        .unwrap_or_else(|| panic!("no `{name}` column; header: {header:?}"));
    &row[i]
}

/// Every chain's saved paths, as `paths[chain][draw][substep][column]` plus
/// the data column names, read back from `trajectories.tsv`.
fn read_paths(stage: &Path) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<String>) {
    let mut chains = Vec::new();
    let mut columns: Option<Vec<String>> = None;
    for c in 1..=CHAINS {
        let (header, rows) = read_tsv(&stage.join(format!("chain_{c}/trajectories.tsv")));
        let data_cols: Vec<String> = header.iter()
            .filter(|h| !matches!(h.as_str(), "chain" | "draw" | "time" | "date"))
            .cloned()
            .collect();
        match &columns {
            None => columns = Some(data_cols.clone()),
            Some(prev) => assert_eq!(prev, &data_cols, "chains share one column layout"),
        }
        // Group rows by draw, in file order (which is sweep order).
        let mut by_draw: BTreeMap<usize, Vec<Vec<f64>>> = BTreeMap::new();
        for row in &rows {
            let draw: usize = col(&header, row, "draw").parse().expect("draw parses");
            let vals: Vec<f64> = data_cols.iter()
                .map(|n| col(&header, row, n).parse::<f64>().expect("value parses"))
                .collect();
            by_draw.entry(draw).or_default().push(vals);
        }
        chains.push(by_draw.into_values().collect::<Vec<_>>());
    }
    (chains, columns.expect("at least one chain"))
}

#[test]
fn pgas_stage_publishes_latent_path_convergence_that_matches_its_own_paths() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("e2e");
    let (ir, data) = write_fixture(tmp.path());
    let (fit, out) = write_fit_toml(tmp.path(), &ir, &data);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit).arg("--seed").arg("11")
        .output().expect("spawn camdl");
    assert!(r.status.success(),
        "fit run failed: {}", String::from_utf8_lossy(&r.stderr));
    let stage = stage_leaf(&out);

    // ── The paths the stage reduced ───────────────────────────────────────
    let (paths, columns) = read_paths(&stage);
    let n_draws = paths.iter().map(|c| c.len()).min().unwrap();
    assert_eq!(n_draws, SWEEPS - BURN_IN, "thin = 1 and n_trajectories > retained: every retained sweep saved a path");
    let n_substeps = paths[0][0].len();
    // The saved path leads with the initial-condition row at t_start (gh#270),
    // then one row per substep.
    assert_eq!(n_substeps, T_END + 1, "t_start row + one per day at dt = 1");
    let n_cols = columns.len();

    // ── One row per cell, every column assessed ───────────────────────────
    let (header, rows) = read_tsv(&stage.join("latent_convergence.tsv"));
    assert_eq!(rows.len(), n_substeps * n_cols,
        "one row per (substep, column): {} substeps × {} columns", n_substeps, n_cols);
    let mut seen: BTreeMap<(usize, String), usize> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        let s: usize = col(&header, row, "substep").parse().unwrap();
        let name = col(&header, row, "column").to_string();
        assert!(columns.contains(&name), "`{name}` is not a trajectories.tsv column");
        assert!(seen.insert((s, name), i).is_none(), "duplicate cell");
    }

    // ── Every cell is the estimator's own answer on the read-back vectors ─
    let mut n_mixed = 0;
    let mut n_frozen = 0;
    let mut n_constant = 0;
    let mut last_frozen: Option<usize> = None;
    let mut by_bin_status: Vec<BTreeMap<String, usize>> = vec![BTreeMap::new(); N_BINS];
    // Per bin, over non-constant cells: chains that never moved, counted from
    // the read-back vectors rather than the published `n_frozen_chains`.
    let mut by_bin_frozen_chains: Vec<usize> = vec![0; N_BINS];
    for row in &rows {
        let s: usize = col(&header, row, "substep").parse().unwrap();
        let name = col(&header, row, "column");
        let k = columns.iter().position(|c| c == name).unwrap();
        let status = col(&header, row, "status");
        let vectors: Vec<Vec<f64>> = paths.iter()
            .map(|chain| chain[..n_draws].iter().map(|draw| draw[s][k]).collect())
            .collect();
        let bin = (s * N_BINS / n_substeps).min(N_BINS - 1);
        *by_bin_status[bin].entry(status.to_string()).or_default() += 1;
        if status != "constant" {
            let frozen = vectors.iter()
                .filter(|v| v.iter().all(|x| *x == v[0]))
                .count();
            assert_eq!(col(&header, row, "n_frozen_chains"), frozen.to_string(),
                "`{name}` substep {s}: published frozen-chain count vs the vectors");
            by_bin_frozen_chains[bin] += frozen;
        }
        match (status, rank_convergence(&vectors)) {
            ("mixed", Ok(rc)) => {
                n_mixed += 1;
                assert!(!rc.all_chains_frozen);
                let published: f64 = col(&header, row, "rhat").parse().unwrap();
                assert!((published - rc.rhat).abs() < 1e-6,
                    "`{name}` substep {s}: published R̂ {published}, recomputed {}", rc.rhat);
                let ess = col(&header, row, "ess_bulk");
                if rc.ess_bulk.is_finite() {
                    let e: f64 = ess.parse().unwrap();
                    assert!((e - rc.ess_bulk).abs() < 1e-6,
                        "`{name}` substep {s}: published ESS {e}, recomputed {}", rc.ess_bulk);
                } else {
                    assert_eq!(ess, "NA");
                }
                // The R̂ ingredients are published too, and must be consistent
                // with the vectors: a frozen chain count of 0 here.
                assert_eq!(col(&header, row, "n_frozen_chains"), "0",
                    "a mixed cell with a frozen chain is possible but not on this fixture");
            }
            ("frozen_disagree", Ok(rc)) => {
                n_frozen += 1;
                assert!(rc.all_chains_frozen, "`{name}` substep {s}: not every chain constant");
                assert_eq!(col(&header, row, "rhat"), "NA");
                assert_eq!(col(&header, row, "n_frozen_chains"), CHAINS.to_string());
                last_frozen = Some(last_frozen.map_or(s, |m| m.max(s)));
            }
            ("constant", Err(ConvergenceError::ConstantDraws { .. })) => {
                n_constant += 1;
                assert_eq!(col(&header, row, "rhat"), "NA");
                // Constant across the pool: the chain means agree exactly.
                assert_eq!(col(&header, row, "chain_mean_min"), col(&header, row, "chain_mean_max"));
            }
            (st, res) => panic!("`{name}` substep {s}: status `{st}` but the estimator says {res:?}"),
        }
    }
    assert_eq!(n_mixed + n_frozen + n_constant, rows.len());
    assert!(n_mixed > 0, "a 40-particle SIR fit over 30 draws must have mixing cells");

    // ── The JSON block is the table, reduced ──────────────────────────────
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(stage.join("pgas_summary.json")).unwrap()).unwrap();
    for key in ["stage", "n_chains", "acceptance_rates", "thin", "trajectories",
                "path_renewal", "rhat", "ess", "ess_tail"] {
        assert!(v.get(key).is_some(), "additive: `{key}` must still be in pgas_summary.json");
    }
    let lc = v.get("latent_convergence").unwrap_or_else(|| panic!(
        "a PGAS stage must write a `latent_convergence` block; keys are {:?}",
        v.as_object().map(|o| o.keys().collect::<Vec<_>>())));
    assert_eq!(lc["n_chains"].as_u64(), Some(CHAINS as u64));
    assert_eq!(lc["n_draws"].as_u64(), Some(n_draws as u64));
    assert_eq!(lc["n_substeps"].as_u64(), Some(n_substeps as u64));
    assert_eq!(lc["n_columns"].as_u64(), Some(n_cols as u64));
    assert_eq!(lc["table"], serde_json::json!("latent_convergence.tsv"));
    assert!(lc["ess_over"].as_str().unwrap().contains("saved paths"),
        "the block must say the ESS is over saved paths, not sweeps");
    let bins = lc["bins"].as_array().expect("bins");
    assert_eq!(bins.len(), N_BINS);
    for (b, bin) in bins.iter().enumerate() {
        let n = bin["n_cells"].as_u64().unwrap() as usize;
        let counted: usize = by_bin_status[b].values().sum();
        assert_eq!(n, counted, "bin {b}: n_cells is the table's row count in that tenth");
        let frac = |st: &str| *by_bin_status[b].get(st).unwrap_or(&0) as f64 / n as f64;
        for (key, st) in [("frac_frozen_disagree", "frozen_disagree"),
                          ("frac_constant", "constant"), ("frac_mixed", "mixed")] {
            let published = bin[key].as_f64().unwrap_or_else(|| panic!("bin {b}: {key}"));
            assert!((published - frac(st)).abs() < 1e-12,
                "bin {b}: {key} published {published}, table says {}", frac(st));
        }
        let non_constant = n - by_bin_status[b].get("constant").copied().unwrap_or(0);
        match bin["frozen_chain_frac"].as_f64() {
            Some(published) => {
                assert!(non_constant > 0, "bin {b}: frozen_chain_frac published over no cell");
                let expect = by_bin_frozen_chains[b] as f64 / (non_constant * CHAINS) as f64;
                assert!((published - expect).abs() < 1e-12,
                    "bin {b}: frozen_chain_frac published {published}, vectors say {expect}");
            }
            None => assert_eq!(non_constant, 0, "bin {b}: frozen_chain_frac missing"),
        }
    }
    let expect_agree = match last_frozen {
        None => Some(0),
        Some(s) if s + 1 < n_substeps => Some(s + 1),
        Some(_) => None,
    };
    assert_eq!(lc["agree_from"].as_u64().map(|x| x as usize), expect_agree,
        "agree_from is the substep after the table's last frozen_disagree row");

    // ── It reaches the terminal, under the renewal profile ───────────────
    let stderr = String::from_utf8_lossy(&r.stderr);
    let renewal_at = stderr.find("path renewal").expect("renewal profile printed");
    let latent_at = stderr.find("latent-path convergence").expect("latent profile printed");
    assert!(latent_at > renewal_at, "the two profiles are read together: latent under renewal");
    assert!(stderr.contains("frozen-disagree"), "stderr was:\n{stderr}");

    // ── `fit summary` recomputes the same block from the paths on disk ────
    // The six profile lines (title through `ESS min`), compared trimmed: the
    // summary indents them into its own section.
    fn profile_lines(text: &str) -> Vec<String> {
        text.lines()
            .skip_while(|l| !l.contains("latent-path convergence ("))
            .take(6)
            .map(|l| l.trim().to_string())
            .collect()
    }
    let s = Command::new(&bin)
        .arg("fit").arg("summary").arg(&fit)
        .output().expect("spawn camdl");
    assert!(s.status.success(),
        "fit summary failed: {}", String::from_utf8_lossy(&s.stderr));
    let summary = String::from_utf8_lossy(&s.stdout);
    let at_stage_end = profile_lines(&stderr);
    let in_summary = profile_lines(&summary);
    assert_eq!(at_stage_end.len(), 6, "stage-end profile:\n{stderr}");
    assert_eq!(in_summary, at_stage_end, "fit summary output:\n{summary}");
}
