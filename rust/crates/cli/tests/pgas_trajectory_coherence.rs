//! gh#264 / gh#270 + latent-trajectory-output consolidation (2026-06-09): the
//! saved PGAS posterior trajectories must be (a) a single directionally-coherent
//! path — `S` monotone non-increasing, each step's `−ΔS` equal to the infection
//! flow, the AGGREGATE `Σ flow_infection == S₀ − S_final` (gh#270: requires the
//! path to carry its `t_start` initial row), population conserved — and (b)
//! emitted in the shared
//! tidy/long `trajectories.tsv` format: a `# camdl-trajectories v1` header,
//! leading `chain  draw  time` id columns, int/real compartments,
//! `flow_<transition>`, then `inc_<stream>` columns whose values are the
//! observation model's `FlowSum` projection of the substep flows (the gh#48
//! safe path) — NOT a finite-difference of compartment counts.
//!
//! This is the END-TO-END guard: it runs a tiny PGAS fit and audits the
//! written `trajectories.tsv` + `trajectories.json`. (The
//! `SubstepRecord → Snapshot` adapter and the `inc_<stream>` projection are
//! unit-tested in
//! `sim::inference::pgas::grid_tests::to_trajectory_projects_incidence_from_flows_not_count_diff`;
//! the coherent-path reconstruction in
//! `coherent_counts_after_removes_as_join_backflow`. This test covers the
//! writer wiring + the two-line posterior-band load.)
//!
//! Skipped when the release binary or camdlc isn't present.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust`", p.display());
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

/// Find the single per-chain `trajectories.tsv` under the fit results.
fn find_trajectories_tsv(dir: &Path) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("trajectories.tsv") {
                return Some(p);
            }
        }
    }
    None
}

fn find_manifest(dir: &Path) -> Option<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("trajectories.json") {
                return Some(p);
            }
        }
    }
    None
}

#[test]
fn saved_pgas_trajectories_are_coherent_and_tidy() {
    let bin = camdl_bin();
    let Some(camdlc) = camdlc_bin() else { return };
    let tmp = std::env::temp_dir().join(format!("camdl_traj_coh_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // Tiny SIR — S leaves only via infection, so a coherent path has S
    // monotone non-increasing and −ΔS == flow_infection at every step. The
    // `cases` stream uses an INCIDENCE projection (incidence(infection)), so a
    // per-substep `inc_cases` column appears and must equal flow_infection (the
    // FlowSum of the infection flow), NOT a count diff like ΔI = infection −
    // recovery.
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 12 'days }
"#;
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, src).unwrap();
    let ir = tmp.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(
        tmp.join("cases.tsv"),
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t12\n5\t9\n6\t7\n7\t5\n8\t4\n9\t3\n10\t2\n11\t1\n12\t1\n",
    )
    .unwrap();

    let fit = tmp.join("fit.toml");
    std::fs::write(&fit, format!(r#"
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
chains = 1
particles = 30
sweeps = 12
burn_in = 2
n_trajectories = 4
"#,
        out = tmp.join("results").display(),
        ir = ir.display(),
        data = tmp.join("cases.tsv").display(),
    )).unwrap();

    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit).arg("--seed").arg("1")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn");
    assert!(r.status.success(), "fit run failed: {}", String::from_utf8_lossy(&r.stderr));

    let traj = find_trajectories_tsv(&tmp.join("results"))
        .expect("a saved trajectories.tsv under the fit results");
    let text = std::fs::read_to_string(&traj).unwrap();
    let mut lines = text.lines();

    // (1) Version header.
    let version_line = lines.next().expect("version header");
    assert!(version_line.starts_with("# camdl-trajectories v1"),
        "expected camdl-trajectories header, got: {version_line}");
    assert!(version_line.contains("method=pgas"), "header must name method: {version_line}");
    assert!(version_line.contains("granularity=substep"),
        "header must name granularity: {version_line}");

    // (2) Tidy/long column header with leading id columns.
    let header: Vec<&str> = lines.next().expect("col header").split('\t').collect();
    assert_eq!(&header[0..3], &["chain", "draw", "time"],
        "tidy format leads with chain/draw/time, got {header:?}");
    let col = |name: &str| header.iter().position(|h| *h == name)
        .unwrap_or_else(|| panic!("column {name} not in header {header:?}"));
    let (ci, di, ti) = (col("chain"), col("draw"), col("time"));
    let (si, ii, ri) = (col("S"), col("I"), col("R"));
    let (fi_inf, fi_rec) = (col("flow_infection"), col("flow_recovery"));
    let inc_i = col("inc_cases"); // incidence projection ⇒ inc_<stream> column

    // (3) Per-draw coherence + the inc_<stream>-is-the-projection property.
    // Group rows by (chain, draw); within a draw audit S-monotonicity,
    // population conservation, −ΔS == flow_infection, and
    // inc_cases == flow_infection (FlowSum of the infection flow — NOT
    // ΔI = infection − recovery, which a count-diff would give).
    let rows: Vec<Vec<f64>> = lines
        .map(|line| line.split('\t')
            .map(|v| v.parse::<f64>().unwrap_or(0.0))
            .collect())
        .collect();
    assert!(!rows.is_empty(), "trajectories.tsv had no data rows");

    let mut by_draw: BTreeMap<(i64, i64), Vec<&Vec<f64>>> = BTreeMap::new();
    for r in &rows {
        by_draw.entry((r[ci] as i64, r[di] as i64)).or_default().push(r);
    }
    assert!(by_draw.len() >= 1, "expected at least one (chain,draw) group");

    let mut total_rows = 0usize;
    let mut audited_seeded_draw = false;
    for ((_chain, _draw), draw_rows) in &by_draw {
        let mut prev_s: Option<i64> = None;
        let mut pop0: Option<i64> = None;
        let mut prev_t: Option<f64> = None;
        let mut sum_inf: i64 = 0;
        let (mut s_first, mut s_last): (Option<i64>, i64) = (None, 0);
        for r in draw_rows {
            let s = r[si] as i64;
            let inf = r[fi_inf] as i64;
            let rec = r[fi_rec] as i64;
            let inc = r[inc_i] as i64;
            let pop = s + (r[ii] as i64) + (r[ri] as i64);

            // inc_cases is the FlowSum projection of the infection flow.
            assert_eq!(inc, inf,
                "inc_cases must equal flow_infection (FlowSum projection), got inc={inc} inf={inf}");
            // ... and (in this model) is NOT a count-diff ΔI = infection −
            // recovery. Only assert the distinction when recovery actually fired,
            // else the two coincide trivially.
            if rec > 0 {
                assert_ne!(inc, inf - rec,
                    "inc_cases must be the projection, not ΔI = infection − recovery");
            }

            if let Some(ps) = prev_s {
                assert!(s <= ps, "S must be monotone non-increasing, got {ps} -> {s}");
                assert_eq!(ps - s, inf, "−ΔS must equal flow_infection (coherent path)");
            }
            let p0 = *pop0.get_or_insert(pop);
            assert_eq!(pop, p0, "S+I+R must be conserved across the path");
            // Time strictly increasing within a draw.
            if let Some(pt) = prev_t {
                assert!(r[ti] > pt, "time must increase within a draw, {pt} -> {}", r[ti]);
            }
            sum_inf += inf;
            s_first.get_or_insert(s);
            s_last = s;
            prev_s = Some(s);
            prev_t = Some(r[ti]);
            total_rows += 1;
        }
        // gh#270: the AGGREGATE must reconcile, not just consecutive steps. The
        // path includes its t_start initial row, so the first written S is the
        // true S₀ and every infection — including the first substep's — has its
        // S decrement recorded. Without the t_start row this fails by exactly
        // flow_infection[0], the seed-stratum residual. (The per-step check
        // above can't see this: it exempts the first row.)
        let s0 = s_first.expect("draw had no rows");
        assert_eq!(sum_inf, s0 - s_last,
            "Σ flow_infection ({sum_inf}) must equal S₀−S_final ({}) over the whole path (gh#270)",
            s0 - s_last);
        // With I₀ = 10 the seed's first substep almost always infects someone,
        // so this fixture genuinely exercises the residual (flow_infection[0]>0),
        // not the vacuous I₀=0 case where the identity holds trivially.
        if sum_inf > 0 { audited_seeded_draw = true; }
    }
    assert!(total_rows > 0, "no trajectory rows audited");
    assert!(audited_seeded_draw,
        "no draw had any infection flow — fixture not exercising the seed residual");

    // (4) The §4b "two-line load" acceptance test, in Rust: load the tidy file
    // and compute a posterior median band over S grouped by time. This is the
    // groupby(time) a researcher runs in pandas/readr; if the format is right
    // it is a few lines here too.
    let mut s_by_time: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for r in &rows {
        // bucket time to integer (dt = 1 day) for the band
        s_by_time.entry(r[ti] as i64).or_default().push(r[si]);
    }
    assert!(s_by_time.len() > 1, "band needs more than one time point");
    // Median S must be non-increasing across the band (SIR: S only decreases).
    let medians: Vec<f64> = s_by_time.values().map(|v| {
        let mut vv = v.clone();
        vv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vv[vv.len() / 2]
    }).collect();
    for w in medians.windows(2) {
        assert!(w[1] <= w[0] + 1.0, // +1 slack for ties at the median across draws
            "posterior-median S band should be ~non-increasing, got {:?}", medians);
    }

    // (5) Manifest discoverability: trajectories.json records the conditioned
    // flag, method, granularity, and the column list.
    let manifest_path = find_manifest(&tmp.join("results"))
        .expect("a trajectories.json manifest");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest.contains("\"method\": \"pgas\""), "manifest method: {manifest}");
    assert!(manifest.contains("\"granularity\": \"substep\""), "manifest granularity");
    assert!(manifest.contains("\"conditioned\": true"),
        "manifest must mark PGAS smoother paths conditioned: {manifest}");
    assert!(manifest.contains("inc_cases"), "manifest columns must list inc_cases");

    let _ = std::fs::remove_dir_all(&tmp);
}
