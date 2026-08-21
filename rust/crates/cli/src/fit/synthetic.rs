//! Synthetic-data generation for `[synthetic]` fit configs.
//!
//!
//! Runs the simulation backend once per `sim_seed`, samples each
//! observation stream through its declared likelihood, and writes one
//! wide-format TSV per dataset into `<fit_dir>/synthetic/data/`. The
//! resulting file paths can be handed to the fit runner verbatim, as
//! if the user had supplied them via `[data.observations]`.
//!
//! Generation is a thin wrapper over the existing `simulate --obs`
//! pipeline: `util::run_simulation` produces the trajectory,
//! `main::project_all_obs_times` computes the projection per
//! observation tick, and `sim::inference::obs_model::compile_obs_sample_pf`
//! samples the likelihood. No new simulation machinery — just a
//! write-loop and deterministic path layout.
//!
//! See docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::rng::StatefulRng;

use super::config_v2::{SyntheticSpec, format_dataset_dir};
use crate::util::{load_params_toml, SimRun, run_simulation};

/// One generated synthetic dataset. Writers of summary / coverage
/// tables consume this to find each cell's data file; the runner
/// dispatches a fit per entry.
#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    /// 1-based dataset index (matches `ds_NN` in the output directory).
    pub idx: usize,
    /// Wide-format TSV at `<fit_dir>/synthetic/data/ds_NN.tsv`.
    pub path: PathBuf,
}

/// Generate `len(sim_seeds)` synthetic datasets into
/// `<fit_dir>/synthetic/data/`. Returns the dataset descriptors in
/// 1-based index order.
///
/// Each dataset is a single run of the simulation backend at
/// `spec.true_params` with the given `sim_seed`, with every
/// observation block in the model sampled through its declared
/// likelihood at the declared schedule. The resulting TSV has one
/// column per observation stream (plus `time`).
///
/// `emit` is `fit run --emit-every` (gh#656). This is the one path where the
/// emission cadence determines data that is then FITTED, so it is
/// identity-bearing here: it changes the generated TSV's bytes, and the fit
/// hashes each training stream's bytes (`FitDigest.data`), so the fit re-keys.
/// Correct — the data changed.
pub fn generate_synthetic_datasets(
    spec: &SyntheticSpec,
    model_path: &str,
    fit_dir: &Path,
    dt: f64,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<Vec<SyntheticDataset>, String> {
    let data_dir = fit_dir.join("synthetic").join("data");
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("cannot create {}: {}", data_dir.display(), e))?;

    // Also copy truth.toml into synthetic/ for provenance — downstream
    // summary/coverage computation reads this, and it ties the whole
    // synthetic run to the specific truth the user declared.
    let truth_bytes = std::fs::read(&spec.true_params)
        .map_err(|e| format!("cannot read true_params {}: {}", spec.true_params, e))?;
    let synthetic_dir = fit_dir.join("synthetic");
    std::fs::write(synthetic_dir.join("truth.toml"), &truth_bytes)
        .map_err(|e| format!("cannot write synthetic/truth.toml: {}", e))?;

    let seeds = spec.sim_seeds.to_vec()
        .map_err(|e| format!("[synthetic] sim_seeds: {}", e))?;
    // gh#656: an override changes what is GENERATED, but the fit-level container
    // this writes into is keyed on model + config before any data exists — so
    // two cadences would otherwise write the same `ds_NN.tsv` path, the second
    // silently replacing the first's dataset beside a cell fit still keyed on
    // the first's bytes. Tagging the filename keeps them side by side. Without
    // the flag the name is exactly `ds_NN.tsv`, as it has always been.
    let tag = emit
        .map(|e| {
            let h = crate::hashing::sha256_hex(e.identity_repr().as_bytes());
            format!("-emit{}", &h[..8])
        })
        .unwrap_or_default();
    let mut out = Vec::with_capacity(seeds.len());
    for (i, &sim_seed) in seeds.iter().enumerate() {
        let idx = i + 1;
        let path = data_dir.join(format!("{}{tag}.tsv", format_dataset_dir(idx)));
        // Generate the dataset (returns a content hash but we don't persist
        // it — the per-cell content_hash flow lived in the v1 grid summary
        // path, which has been deleted).
        let _ = generate_one_dataset(
            spec, model_path, sim_seed, &path, dt, emit,
        )?;
        out.push(SyntheticDataset { idx, path });
    }
    Ok(out)
}

/// Generate a single synthetic dataset at a given seed. Writes the
/// wide-format TSV to `out_path`. Returns a short hex content hash.
fn generate_one_dataset(
    spec: &SyntheticSpec,
    model_path: &str,
    sim_seed: u64,
    out_path: &Path,
    dt: f64,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    // Build a SimRun matching `simulate --obs` semantics — load the
    // model, apply `true_params` as overrides, apply the synthetic
    // scenario if declared, run the backend at `sim_seed`.
    let truth_overrides = load_params_toml(&spec.true_params)
        .map_err(|e| format!("parsing [synthetic] true_params {}: {}", spec.true_params, e))?;
    let run = SimRun {
        ir_path: model_path.to_string(),
        params_files: vec![],
        overrides: truth_overrides,
        point_overrides: Default::default(),
        set_vec_entries: vec![],
        table_files: Default::default(),
        scenario_name: spec.scenario.clone(),
        // Synthetic-data generation runs the model forward before any
        // data exists, so an anchored model has nothing to anchor TO;
        // `CompiledModel::new` refuses it by name.
        obs_anchors: None,
        t_end_override: None, // fit refuses horizons (gh#561)
        init_state: None, // synthetic data-gen starts from the model's init {}
        adhoc_enable: vec![],
        adhoc_disable: vec![],
        scenario_inline_name: None,
        scenario_inline_set: vec![],
        scenario_inline_scale: vec![],
        backend: spec.backend,
        dt,
        seed: sim_seed,
        integrator: None, // synthetic data-gen uses the model's declared integrator
    };
    let (traj, model) = run_simulation(&run)?;

    if model.observations.is_empty() {
        return Err("model has no `observations { }` block — synthetic-data fits \
             require at least one observation stream in the .camdl file".to_string());
    }

    // gh#656: refuse an override that names no stream, names a fit-only stream,
    // or targets an `at [...]` list — before any data is written, so a
    // mis-typed label never yields a silently unchanged dataset.
    if let Some(e) = emit {
        e.validate(&model.observations)?;
    }

    // Compile observation samplers once per dataset. The RNG stream
    // uses `sim_seed ^ util::SEED_MIX_OBS` so that observation noise
    // is deterministic from `sim_seed` alone — re-running a dataset
    // with the same seed reproduces the same draws bit-for-bit.
    //
    // The decorrelation constant is shared with `camdl simulate
    // --obs-only` (main.rs), so the same nominal seed produces
    // identical observation bytes whether generated via CLI or via
    // the `[synthetic]` block. Diverging these constants in the past
    // caused a parameter-recovery discrepancy that looked like a +59% β
    // bias — see the 2026-04-18 downstream incident report.
    let compiled = Arc::new(
        CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?,
    );
    let params = compiled.default_params.clone();
    let mut obs_rng = StatefulRng::new(sim_seed ^ crate::util::SEED_MIX_OBS);

    // Per-stream: declared times, projected values per time, drawn values.
    let mut all_times: Vec<Vec<f64>> = Vec::with_capacity(model.observations.len());
    let mut all_draws: Vec<Vec<f64>> = Vec::with_capacity(model.observations.len());
    for obs_ir in &model.observations {
        // The generated dataset must not extend past the window the trajectory
        // beside it was run over — under `[synthetic] scenario = "X"` with a
        // shortened horizon the surplus rows are fabricated, and a recovery
        // study would then fit invented data (gh#561).
        let times = crate::obs_emit_schedule_times(obs_ir, None, model.simulation.t_end, emit)?;
        // `None` — the synthetic dataset IS the data; nothing to condition on
        // (gh#702).
        let projected =
            crate::project_all_obs_times(&traj, obs_ir, &model, &times, None)?;

        let sampler = sim::inference::obs_model::compile_obs_sample_pf(
            obs_ir, compiled.clone(), &params,
        );
        // GH #6: pass compartment state at each obs time so likelihood
        // arg expressions (e.g. p = projected / N) resolve correctly.
        let draws: Vec<f64> = times.iter().enumerate().map(|(ti, &obs_t)| {
            let snap = crate::snap_at(&traj, obs_t);
            sampler(projected[ti], obs_t, &snap.int_state.counts, &[], &mut obs_rng)
        }).collect();
        all_times.push(times);
        all_draws.push(draws);
    }

    // Write wide-format TSV: time column + one column per obs stream.
    // Uses the union of all obs times across streams, sorted; missing
    // values are blank (NA-compatible with the fit-loader). Streams
    // that share the same schedule (the common case for parameter-
    // recovery studies) collapse to one row per time, no NAs.
    let mut union_times: Vec<f64> = all_times.iter().flatten().copied().collect();
    union_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    union_times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut buf = String::new();
    buf.push_str("time");
    for obs_ir in &model.observations {
        buf.push('\t');
        buf.push_str(&obs_ir.name);
    }
    buf.push('\n');

    for &t in &union_times {
        buf.push_str(&format_time(t));
        for (si, _obs_ir) in model.observations.iter().enumerate() {
            buf.push('\t');
            // Find the tick in this stream matching t (within tolerance)
            let hit = all_times[si].iter()
                .position(|&ot| (ot - t).abs() < 1e-9);
            match hit {
                Some(ti) => buf.push_str(&format_value(all_draws[si][ti])),
                None     => {} // blank cell — fit-loader treats as missing
            }
        }
        buf.push('\n');
    }

    std::fs::write(out_path, &buf)
        .map_err(|e| format!("cannot write {}: {}", out_path.display(), e))?;

    // Short content hash for provenance. The downstream grid runner
    // uses this in its per-cell hash so that regenerating with an
    // unchanged seed + truth is a cache hit.
    let hash = {
        use sha2::{Sha256, Digest};
        let result = Sha256::digest(buf.as_bytes());
        hex::encode(&result[..4])
    };

    Ok(hash)
}

/// Render a time value without trailing zeros, keeping integer-valued
/// times as integers (`5` not `5.0`) to match the existing observation
/// file conventions.
fn format_time(t: f64) -> String {
    if (t.round() - t).abs() < 1e-9 {
        format!("{}", t.round() as i64)
    } else {
        format!("{}", t)
    }
}

fn format_value(v: f64) -> String {
    if (v.round() - v).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config_v2::SeedsSpec;

    #[test]
    fn format_time_keeps_integers_clean() {
        assert_eq!(format_time(0.0), "0");
        assert_eq!(format_time(7.0), "7");
        assert_eq!(format_time(7.5), "7.5");
    }

    #[test]
    fn format_value_keeps_count_data_clean() {
        // Counts should render as integers so the output looks like
        // observed incidence / prevalence, not scientific notation.
        assert_eq!(format_value(42.0), "42");
        assert_eq!(format_value(0.0), "0");
        assert_eq!(format_value(3.14159), "3.14159");
    }

    // ── End-to-end: generation against a tiny compiled SIR fixture.
    //    Requires the OCaml `camdlc` binary built at
    //    `ocaml/_build/default/bin/camdlc.exe`. Skipped automatically
    //    when that binary is absent so the suite stays runnable in
    //    rust-only CI. ──────────────────────────────────────────────

    fn camdlc_path() -> Option<std::path::PathBuf> {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
        let p = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/_build/default/bin/camdlc.exe");
        if p.exists() { Some(p) } else { None }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &std::path::Path { &self.0 }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!(
            "camdl_synth_{}_{}_{}", tag, std::process::id(), ns));
        std::fs::create_dir_all(&base).unwrap();
        TempDir(base)
    }

    fn write_fixture(dir: &std::path::Path, camdlc: &std::path::Path)
        -> (PathBuf, PathBuf)
    {
        let model_src = r#"
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
        std::fs::write(&model_path, model_src).unwrap();

        let output = std::process::Command::new(camdlc)
            .arg(&model_path)
            .output()
            .expect("camdlc invocation must succeed");
        assert!(output.status.success(),
            "camdlc compile failed: {}", String::from_utf8_lossy(&output.stderr));
        let ir_path = dir.join("sir.ir.json");
        std::fs::write(&ir_path, &output.stdout).unwrap();

        let truth_path = dir.join("truth.toml");
        std::fs::write(&truth_path, "beta = 0.8\ngamma = 0.3\nN0 = 1000\n").unwrap();

        (ir_path, truth_path)
    }

    #[test]
    fn generates_one_file_per_sim_seed() {
        let Some(camdlc) = camdlc_path() else {
            eprintln!("skipping: camdlc.exe not built; run `cd ocaml && dune build` first");
            return;
        };
        let tmp = tempdir("one_per");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);

        let fit_dir = tmp.path().join("fit_out");
        let spec = SyntheticSpec {
            true_params: truth_path.to_string_lossy().to_string(),
            sim_seeds: SeedsSpec::List(vec![1, 2, 3]),
            datasets: None,
            scenario: None,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
        };
        let datasets = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(), &fit_dir,
            1.0, None,
        ).expect("generation must succeed on minimal SIR");

        assert_eq!(datasets.len(), 3);
        for (i, ds) in datasets.iter().enumerate() {
            assert_eq!(ds.idx, i + 1);
            assert!(ds.path.exists(),
                "ds_{:02}.tsv must exist at {}", i + 1, ds.path.display());
            let contents = std::fs::read_to_string(&ds.path).unwrap();
            assert!(contents.lines().next().unwrap().contains("cases"),
                "header must declare the obs stream name");
            assert!(contents.lines().count() >= 10,
                "≥10 daily obs rows expected, got {}",
                contents.lines().count().saturating_sub(1));
        }
        assert!(fit_dir.join("synthetic").join("truth.toml").exists(),
            "truth.toml must be copied for provenance");
    }

    #[test]
    fn same_seed_produces_identical_content_hash() {
        let Some(camdlc) = camdlc_path() else { return; };
        let tmp = tempdir("det");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);

        let spec = SyntheticSpec {
            true_params: truth_path.to_string_lossy().to_string(),
            sim_seeds: SeedsSpec::List(vec![42]),
            datasets: None,
            scenario: None,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
        };
        let a = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("run_a"), 1.0, None,
        ).unwrap();
        let b = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("run_b"), 1.0, None,
        ).unwrap();

        assert_eq!(std::fs::read(&a[0].path).unwrap(),
                   std::fs::read(&b[0].path).unwrap(),
                   "same seed + same truth must produce identical datasets");
    }

    #[test]
    fn different_seeds_produce_different_data() {
        let Some(camdlc) = camdlc_path() else { return; };
        let tmp = tempdir("diff");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);
        let spec = SyntheticSpec {
            true_params: truth_path.to_string_lossy().to_string(),
            sim_seeds: SeedsSpec::List(vec![1, 999]),
            datasets: None,
            scenario: None,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
        };
        let ds = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("fit"), 1.0, None,
        ).unwrap();
        assert_ne!(std::fs::read(&ds[0].path).unwrap(),
                   std::fs::read(&ds[1].path).unwrap(),
                   "different sim seeds must produce different data realizations");
    }

    #[test]
    fn spec_roundtrips_seed_lists() {
        let s = SeedsSpec::List(vec![1, 2, 3]);
        assert_eq!(s.to_vec().unwrap(), vec![1u64, 2, 3]);
    }
}
