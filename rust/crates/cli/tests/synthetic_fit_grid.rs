//! End-to-end tests for the replicate-grid machinery in
//! `camdl fit run`. Encodes the canonical modes from
//! docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.
//!
//! Shells out to the built `camdl` binary; skipped silently when
//! the release binary or `camdlc.exe` isn't present so the suite
//! stays runnable in rust-only CI and when tests run before a build.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_sim() -> PathBuf {
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

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gridtest_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// Fit directories are named `<stem>-<fit_hash[:8]>/` — the test can't
/// know the hash up front, so it discovers the single directory under
/// `<out>/fits/` and asserts it starts with the expected stem.
fn find_fit_dir(out: &Path, stem: &str) -> PathBuf {
    let fits = out.join("fits");
    let entries: Vec<_> = std::fs::read_dir(&fits)
        .unwrap_or_else(|_| panic!("no fits/ dir under {}", out.display()))
        .flatten().map(|e| e.path()).collect();
    assert_eq!(entries.len(), 1,
        "expected exactly one fit dir under {}, got {:?}", fits.display(), entries);
    let p = &entries[0];
    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let prefix = format!("{}-", stem);
    assert!(name.starts_with(&prefix),
        "expected {}-<hash> under fits/, got {}", stem, name);
    p.clone()
}

/// Minimal SIR with Poisson prevalence obs, fit config skeleton.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdl = camdlc().expect("camdlc.exe");
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
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 10 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let output = Command::new(&camdl).arg(&model_path).output().unwrap();
    assert!(output.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&output.stderr));
    std::fs::write(&ir_path, &output.stdout).unwrap();

    let truth_path = dir.join("truth.toml");
    std::fs::write(&truth_path, "beta = 0.8\ngamma = 0.3\nN0 = 1000\n").unwrap();

    (ir_path, truth_path)
}

fn stages_block() -> &'static str {
    // Deliberately cheap — we're testing the grid structure, not
    // convergence. One stage, very few iterations, tiny particle
    // count so a 2×2 grid finishes in seconds.
    r#"
[estimate]
beta  = { bounds = [0.01, 5.0], start = 1.0 }
gamma = { bounds = [0.01, 1.0], start = 0.3 }

[fixed]
N0 = 1000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 5
cooling = 0.7
"#
}

fn run_fit(bin: &Path, fit_toml: &Path) {
    let status = Command::new(bin)
        .arg("fit").arg("run")
        .arg(fit_toml)
        .status()
        .expect("camdl fit run must invoke");
    assert!(status.success(), "fit run failed for {}", fit_toml.display());
}

/// gh#147 (M3.2): the CAS stage leaf for `stage_substr` under `fit_dir` —
/// `<fit_dir>/<NN>-<stage>-<h8>/seed_<N>-<h8>/` (the dir holding a `fit_stage`
/// run.json whose `stage` level contains `stage_substr`). Replaces the pre-M3.2
/// `real/fit_<seed>/<stage>` / `synthetic/ds_NN/fit_<seed>/<stage>` probe.
fn cas_stage_leaf(fit_dir: &Path, stage_substr: &str) -> PathBuf {
    let mut stack = vec![fit_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    let stage = v["levels"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .find(|l| l["name"].as_str() == Some("stage"))
                        .and_then(|l| l["label"].as_str())
                        .unwrap_or("");
                    if stage.contains(stage_substr) {
                        return d;
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no CAS '{}' stage leaf under {}", stage_substr, fit_dir.display());
}

/// Every CAS fit-stage leaf dir under `root` whose `stage` level contains
/// `stage_substr` — the multi-cell analog of `cas_stage_leaf` (one entry per
/// `seed_<S>` leaf, across one or more fit-bases).
fn all_stage_leaves(root: &Path, stage_substr: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(d.join("run.json")).unwrap_or_default(),
        ) {
            if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                let stage = v["levels"].as_array().into_iter().flatten()
                    .find(|l| l["name"].as_str() == Some("stage"))
                    .and_then(|l| l["label"].as_str()).unwrap_or("");
                if stage.contains(stage_substr) { out.push(d.clone()); }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    out
}

/// The `seed` level (`seed_<N>`) of a CAS fit-stage leaf, parsed to `N`.
fn seed_of(leaf: &Path) -> u64 {
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    v["levels"].as_array().into_iter().flatten()
        .find(|l| l["name"].as_str() == Some("seed"))
        .and_then(|l| l["label"].as_str())
        .and_then(|s| s.strip_prefix("seed_"))
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no seed level on {}", leaf.display()))
}

/// The distinct CAS fit-base dirs under `fits/` holding an `mle` stage leaf —
/// the cell fits (one per dataset for a synthetic grid), excluding the
/// synthetic dataset-generation segment (which carries no stage leaf).
fn cell_fit_bases(out: &Path) -> Vec<PathBuf> {
    let fits = out.join("fits");
    let mut bases: Vec<PathBuf> = std::fs::read_dir(&fits).unwrap()
        .flatten().map(|e| e.path())
        .filter(|p| p.is_dir() && !all_stage_leaves(p, "mle").is_empty())
        .collect();
    bases.sort();
    bases
}

/// The synthetic dataset-generation segment under `fits/` — the fit-base that
/// holds `synthetic/data/` (generated `ds_NN.tsv` + `truth.toml`), distinct
/// from the per-dataset cell fits.
fn datagen_segment(out: &Path) -> PathBuf {
    let fits = out.join("fits");
    std::fs::read_dir(&fits).unwrap().flatten().map(|e| e.path())
        .find(|p| p.join("synthetic").join("data").is_dir())
        .unwrap_or_else(|| panic!("no synthetic dataset-gen segment under {}", fits.display()))
}

// ── mode 1: a single fit lands at its CAS stage leaf ───────────────────
#[test]
fn single_fit_lands_at_cas_stage_leaf() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("single");
    let (ir, _) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let data_tsv = tmp.path().join("cases.tsv");
    std::fs::write(&data_tsv, "time\tcases\n1\t5\n2\t7\n3\t12\n4\t18\n5\t25\n6\t30\n7\t28\n8\t22\n9\t15\n10\t10\n").unwrap();
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[data.observations]
cases = "{}"
{}
"#, out.display(), ir.display(), data_tsv.display(), stages_block())).unwrap();

    run_fit(&bin, &fit_toml);

    // gh#147 (M3.2): the `mle` stage lands at the content-addressed leaf
    // `<fit_dir>/<NN>-mle-<h8>/seed_<N>-<h8>/`, not the legacy
    // `real/fit_1/mle/` wrapper.
    let fit_dir = find_fit_dir(&out, "fit");
    let leaf = cas_stage_leaf(&fit_dir, "mle");
    assert!(leaf.join("run.json").is_file(),
        "the mle stage leaf {} must hold a run.json", leaf.display());
    // The retired per-seed wrapper must NOT exist.
    assert!(!fit_dir.join("real").exists(),
        "the legacy real/fit_<seed>/ wrapper must NOT exist under {}", fit_dir.display());
}

// ── mode 2: fit_seeds list → one CAS stage leaf per seed ───────────────
#[test]
fn fit_seeds_list_produces_per_seed_dirs() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("list");
    let (ir, _) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let data_tsv = tmp.path().join("cases.tsv");
    std::fs::write(&data_tsv, "time\tcases\n1\t5\n2\t7\n3\t12\n4\t18\n5\t25\n6\t30\n7\t28\n8\t22\n9\t15\n10\t10\n").unwrap();
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"
fit_seeds = [11, 22, 33]

[model]
camdl = "{}"

[data.observations]
cases = "{}"
{}
"#, out.display(), ir.display(), data_tsv.display(), stages_block())).unwrap();

    run_fit(&bin, &fit_toml);

    // Real multi-seed: one fit-base (same model + data + config), one CAS
    // `mle` stage leaf per fit-seed. The legacy `real/fit_<seed>/` wrapper is
    // retired; the cross-seed summary.tsv is the deferred M4 view (gh#150).
    let fit_dir = find_fit_dir(&out, "fit");
    let leaves = all_stage_leaves(&fit_dir, "mle");
    let seeds: std::collections::HashSet<u64> = leaves.iter().map(|l| seed_of(l)).collect();
    assert_eq!(seeds, [11u64, 22, 33].into_iter().collect(),
        "expected one mle stage leaf per fit-seed {{11,22,33}}; got {:?} from {:?}",
        seeds, leaves);
    for l in &leaves {
        assert!(l.join("mle_params.toml").is_file(),
            "each seed leaf must hold mle_params.toml: {}", l.display());
    }
    assert!(!fit_dir.join("real").exists(),
        "the legacy real/fit_<seed>/ wrapper must NOT exist under {}", fit_dir.display());
}

// ── gh#110 follow-up: IF2 skips a degenerate chain instead of aborting ──
/// When one IF2 chain's search wanders into the PF-degenerate region, the
/// runner must skip that chain (with a loud diagnostic) and finish the fit
/// on the survivors — matching PMMH's gh#110 skip-and-continue. Previously
/// the runner `process::exit(1)`-ed on the first chain error, killing the
/// whole fit even when other chains were healthy.
///
/// Base seed 33 is chosen because its IF2 search deterministically collapses
/// ESS around obs window 9 (verified); with several chains at least one
/// survives, so a correct runner completes the fit.
#[test]
fn fit_run_skips_degenerate_if2_chain_and_continues() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("degenskip");
    let (ir, _) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let data_tsv = tmp.path().join("cases.tsv");
    std::fs::write(&data_tsv, "time\tcases\n1\t5\n2\t7\n3\t12\n4\t18\n5\t25\n6\t30\n7\t28\n8\t22\n9\t15\n10\t10\n").unwrap();
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"
fit_seeds = [33]

[model]
camdl = "{}"

[data.observations]
cases = "{}"

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 6
particles = 100
iterations = 5
cooling = 0.7
"#, out.display(), ir.display(), data_tsv.display())).unwrap();

    let res = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit_toml)
        .args(["--progress", "none"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run must invoke");
    let stderr = String::from_utf8_lossy(&res.stderr);

    // Acceptance 1: the fit completes despite a degenerate chain.
    assert!(res.status.success(),
        "fit must succeed when an IF2 chain hits PFDegenerate — the bad \
         chain should be skipped, not abort the whole fit.\nstderr:\n{}",
        stderr);

    // Acceptance 2 (non-vacuity): a chain was actually skipped for
    // degeneracy. Without this the test could pass without ever exercising
    // the skip path (e.g. if no chain degenerated).
    assert!(stderr.to_lowercase().contains("degenerate"),
        "expected a degenerate-chain skip diagnostic in stderr — the test \
         must actually exercise the skip path; got:\n{}", stderr);

    // Acceptance 3: the surviving chains still produced the fit output —
    // the `mle` stage leaf (gh#147 M3.2 CAS layout).
    let leaf = cas_stage_leaf(&find_fit_dir(&out, "fit"), "mle");
    assert!(leaf.join("run.json").is_file(),
        "surviving chains must still write the mle stage leaf at {}", leaf.display());
}

// ── mode 3: synthetic generation — N datasets → N content-addressed cells ──
#[test]
fn synthetic_generates_n_datasets_and_fits() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("syn");
    let (ir, truth) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[synthetic]
true_params = "{}"
sim_seeds = [1, 2, 3]
{}
"#, out.display(), ir.display(), truth.display(), stages_block())).unwrap();

    run_fit(&bin, &fit_toml);

    // The dataset-generation segment holds the N generated datasets + truth;
    // each dataset is then fit as its own content-addressed cell (distinct
    // data digest → distinct fit-base). The cross-dataset summary.tsv and the
    // parameter-recovery coverage.tsv are the deferred M4 views (gh#150).
    let datagen = datagen_segment(&out);
    for i in 1..=3 {
        assert!(datagen.join("synthetic").join("data").join(format!("ds_{:02}.tsv", i)).is_file(),
            "ds_{:02}.tsv must be generated under {}", i, datagen.display());
    }
    assert!(datagen.join("synthetic").join("truth.toml").is_file(),
        "truth.toml must be recorded for provenance");
    let bases = cell_fit_bases(&out);
    assert_eq!(bases.len(), 3,
        "3 datasets → 3 distinct content-addressed cell fits; got {:?}", bases);
    for b in &bases {
        assert!(!all_stage_leaves(b, "mle").is_empty(),
            "each cell fit must have an mle stage leaf: {}", b.display());
    }
}

// ── mode 4: synthetic × fit_seeds full matrix ─────────────────────────
#[test]
fn synthetic_and_fit_seeds_full_matrix() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("matrix");
    let (ir, truth) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"
fit_seeds = [1, 2]

[model]
camdl = "{}"

[synthetic]
true_params = "{}"
sim_seeds = [10, 20]
{}
"#, out.display(), ir.display(), truth.display(), stages_block())).unwrap();

    run_fit(&bin, &fit_toml);

    // 2 datasets × 2 fit-seeds: 2 distinct cell fit-bases (one per dataset),
    // each with one CAS `mle` stage leaf per fit-seed. The 2×2 summary.tsv is
    // the deferred M4 view (gh#150).
    let bases = cell_fit_bases(&out);
    assert_eq!(bases.len(), 2, "2 datasets → 2 cell fit-bases; got {:?}", bases);
    for b in &bases {
        let seeds: std::collections::HashSet<u64> =
            all_stage_leaves(b, "mle").iter().map(|l| seed_of(l)).collect();
        assert_eq!(seeds, [1u64, 2].into_iter().collect(),
            "cell fit {} must have a leaf per fit-seed {{1,2}}; got {:?}",
            b.display(), seeds);
    }
}

// ── per-chain random starts: an IF2 stage with N > 1 chains and no
//    starts_from must give each chain its own draw over bounds, and
//    IF2 must actually start from those draws (not just record them
//    decoratively in chain_starts.tsv). Regression against the
//    2026-04-18 finding that v2 dispatch collapsed all chains to the
//    same base_params at iter 0. ─────────────────────────────────────
#[test]
fn v2_if2_chains_diverge_at_iter_0_when_no_starts_from() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("chain_starts");
    let (ir, truth) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    // 8 chains to get a readable spread across the beta bounds.
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[synthetic]
true_params = "{}"
sim_seeds = [1]

[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 1.0 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.3 }}

[fixed]
N0 = 1000

[stages.mle]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 8
particles  = 50
iterations = 2
cooling    = 0.9
"#, out.display(), ir.display(), truth.display())).unwrap();
    run_fit(&bin, &fit_toml);

    // gh#147 (M3.2): the synthetic fit's `mle` stage lands at its CAS leaf,
    // not the legacy `synthetic/ds_01/fit_1/mle/` wrapper. A synthetic fit
    // writes two segments (the generated dataset + the fit), so search the
    // `fits/` root for the stage leaf rather than assuming one segment.
    let stage = cas_stage_leaf(&out.join("fits"), "mle");
    let starts_text = std::fs::read_to_string(stage.join("chain_starts.tsv"))
        .expect("chain_starts.tsv must exist");
    let starts: Vec<Vec<f64>> = starts_text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("chain"))
        .map(|l| l.split('\t').skip(1)  // skip chain id
             .map(|s| s.parse::<f64>().unwrap()).collect())
        .collect();
    assert_eq!(starts.len(), 8, "need 8 chain rows");

    // Assertion 1: chain_starts.tsv shows genuine spread, not 8 copies
    // of the seeded start. Take the beta column (index 0).
    let betas: Vec<f64> = starts.iter().map(|r| r[0]).collect();
    let (min_b, max_b) = betas.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY),
              |(lo, hi), &x| (lo.min(x), hi.max(x)));
    let range = max_b - min_b;
    // Bounds span 5.0 - 0.01 = 4.99. 8 uniform draws over that range
    // should easily span > 20% of the bounds range.
    assert!(range > 1.0,
        "beta starts must span > 1.0 of the 4.99-wide bounds; got range={} \
         from values {:?}. If this fails, the v2 dispatch isn't building \
         per-chain random starts.", range, betas);

    // Assertion 2: IF2 actually *used* those starts (the 2026-04-18 incident:
    // the runner built per-chain specs and `run_if2` then overwrote them with
    // `base_params`). chain_starts.tsv alone can't show that — it is written
    // from the same `per_chain_params` slice the engine receives, so it would
    // look identical either way. The iter-0 trace row is the engine's own
    // report, so it is the one that can distinguish them.
    //
    // Pick the two chains by their ACTUAL extreme starts, not by index. The
    // hard-coded "chain 1 vs chain 8" here assumed chain_starts came out
    // ordered; it does not — the LHS draws land in arbitrary chain order, so
    // the pair could be any two of the eight. A measured run:
    //
    //   chain  start_beta  iter0_beta
    //     1      0.0366      1.9054
    //     2      0.9311      0.9385
    //     3      0.0214      2.2312   <- min start
    //     4      0.8730      0.7535
    //     5      0.0509      2.0929
    //     6      0.0553      1.5353
    //     7      0.0604      1.3580
    //     8      1.7323      1.6384   <- max start
    //
    // chains 1 and 8 happen to land 0.267 apart while the swarm-wide spread is
    // 1.477 (0.754 → 2.231). Choosing by extremity of start gives |2.2312 −
    // 1.6384| = 0.593 on that run. The 0.3 threshold below is UNCHANGED.
    //
    // Note what the table shows about the property itself: the chains that
    // started near the shared fallback (2, 4, 8 — starts 0.87…1.73) stay in
    // 0.75…1.64, while the chains that started near the lower bound (1, 3, 5,
    // 6, 7 — starts 0.02…0.06) are carried to 1.36…2.23. A run where every
    // chain started from the shared `start = 1.0` could not produce the second
    // group. The exact, threshold-free version of this check lives in
    // `sim/tests/if2_honours_per_chain_initial.rs`; this one guards the
    // runner→engine wiring end to end.
    let read_iter_0_beta = |chain: usize| -> f64 {
        let path = stage.join(format!("chain_{}", chain))
            .join("parameter_traces.tsv");
        let text = std::fs::read_to_string(&path).unwrap();
        let first_data = text.lines()
            .find(|l| !l.starts_with('#') && !l.starts_with("iteration"))
            .unwrap();
        // iteration\tloglik\tif2_perturbed_loglik\tbeta\tgamma
        first_data.split('\t').nth(3).unwrap().parse().unwrap()
    };
    // Chain ids are 1-based; `betas[i]` is chain i+1.
    let argmin = betas.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 + 1;
    let argmax = betas.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 + 1;
    let iter0_lo = read_iter_0_beta(argmin);
    let iter0_hi = read_iter_0_beta(argmax);
    let iter0_spread = (iter0_lo - iter0_hi).abs();
    assert!(iter0_spread > 0.3,
        "the lowest-start chain ({}, start {:.4}) and the highest-start chain \
         ({}, start {:.4}) must have iter-0 betas that differ meaningfully \
         (> 0.3); got {:.4} vs {:.4} (spread {:.4}). If the spread is ~rw_sd \
         ({:.3}), IF2 started both chains from the same base_params and only \
         the per-chain RNG diverged them — the .initial-authoritative fix \
         didn't land.",
         argmin, betas[argmin - 1], argmax, betas[argmax - 1],
         iter0_lo, iter0_hi, iter0_spread, 0.03);
}

// ── seeding parity: --obs-only and [synthetic] must produce byte-identical
//    data at the same nominal seed. Regression against the 2026-04-18
//    parameter-recovery bias discrepancy. ──────────────────────────────────
#[test]
fn obs_only_and_synthetic_agree_byte_for_byte_at_same_seed() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("seed_parity");
    let (ir, truth) = write_fixture(tmp.path());

    // Path A: --obs-only at seed=10
    let cli_tsv = tmp.path().join("cli.tsv");
    let cli_status = Command::new(&bin).arg("simulate")
        .arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--seed", "10"])
        .args(["--backend", "chain_binomial", "--dt", "1"])
        .args(["--obs-only"]).arg(&cli_tsv)
        .status().expect("--obs-only must invoke");
    assert!(cli_status.success());

    // Path B: [synthetic] with sim_seeds = [10]
    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[synthetic]
true_params = "{}"
sim_seeds = [10]
{}
"#, out.display(), ir.display(), truth.display(), stages_block())).unwrap();
    run_fit(&bin, &fit_toml);

    let syn_tsv = datagen_segment(&out).join("synthetic").join("data").join("ds_01.tsv");

    let cli_bytes = std::fs::read(&cli_tsv).unwrap();
    let syn_bytes = std::fs::read(&syn_tsv).unwrap();
    assert_eq!(cli_bytes, syn_bytes,
        "--obs-only (seed=N) and [synthetic] (sim_seeds=[N]) must produce \
         byte-identical observations. Diverging these paths caused the \
         2026-04-18 parameter-recovery bias discrepancy. CLI:\n{}\nsynthetic:\n{}",
        String::from_utf8_lossy(&cli_bytes),
        String::from_utf8_lossy(&syn_bytes));
}

// ── mode 5: [data] + [synthetic] errors cleanly ───────────────────────
#[test]
fn data_and_synthetic_errors_cleanly() {
    let bin = camdl_sim();
    if camdlc().is_none() { return; }
    let tmp = tempdir("mutex");
    let (ir, truth) = write_fixture(tmp.path());
    let out = tmp.path().join("out");
    let data_tsv = tmp.path().join("cases.tsv");
    std::fs::write(&data_tsv, "time\tcases\n1\t5\n").unwrap();
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[data.observations]
cases = "{}"

[synthetic]
true_params = "{}"
sim_seeds = [1]
{}
"#, out.display(), ir.display(), data_tsv.display(), truth.display(), stages_block())).unwrap();

    let output = Command::new(&bin).arg("fit").arg("run")
        .arg(&fit_toml).output().unwrap();
    assert!(!output.status.success(),
        "[data]+[synthetic] must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("[data]") && stderr.contains("[synthetic]"),
        "error must name both blocks: {}", stderr);
}
