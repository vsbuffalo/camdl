//! Synthetic-data generation for `[synthetic]` fit configs.
//!
//! Runs the simulation backend once per `sim_seed` and writes one dataset per
//! seed under `<fit_dir>/synthetic/data/ds_NN/` — one file per observation
//! stream, under the stream's own declared column names. The paths are handed
//! to the fit runner verbatim, as if the user had supplied them via
//! `[data.observations]`, so a synthetic dataset is read back by the same
//! loader the real data uses and nothing stands between generation and fit.
//!
//! Generation is [`crate::obs_emit::simulate_dataset`] on the model's own
//! declared design — a `[synthetic]` config binds no data (`[data]` and
//! `[synthetic]` are mutually exclusive), so there is no observed design to
//! preserve. `util::run_simulation` produces the trajectory, the stream's
//! `emit_schedule` and `covers` fix its rows, and the declared likelihood draws
//! each value. No simulation machinery of its own.
//!
//! See docs/dev/proposals/2026-04-17-synthetic-fit-replicates.md.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use super::config_v2::{SyntheticSpec, format_dataset_dir};
use crate::obs_emit::{ObservationDesign, simulate_dataset};
use crate::util::{SimRun, load_params_toml};

/// One generated synthetic dataset. Writers of summary / coverage
/// tables consume this to find each cell's data files; the runner
/// dispatches a fit per entry.
#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    /// 1-based dataset index (matches `ds_NN` in the output directory).
    pub idx: usize,
    /// The generated files, keyed by the `source` each binds to in
    /// `[data.observations]`, in the model's declaration order.
    pub files: IndexMap<String, PathBuf>,
}

/// Generate `len(sim_seeds)` synthetic datasets into
/// `<fit_dir>/synthetic/data/`. Returns the dataset descriptors in
/// 1-based index order.
///
/// Each dataset is a single run of the simulation backend at
/// `spec.true_params` with the given `sim_seed`, with every
/// observation block in the model sampled through its declared
/// likelihood at the declared schedule, written as one file per stream.
///
/// `emit` is `fit run --emit-every` (gh#656). This is the one path where the
/// emission cadence determines data that is then FITTED, so it is
/// identity-bearing here: it changes the generated files' bytes, and the fit
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

    // A synthetic dataset is fitted by THIS model, so its cadence must be one
    // the model's own declaration reads. Checked once, before any run, so a
    // refused cadence writes nothing.
    let (model, _) = crate::util::load_model(model_path)?;
    check_emit_every_matches_declared_windows(&model, emit)?;

    // gh#656: an override changes what is GENERATED, but the fit-level container
    // this writes into is keyed on model + config before any data exists — so
    // two cadences would otherwise write the same `ds_NN/` path, the second
    // silently replacing the first's dataset beside a cell fit still keyed on
    // the first's bytes. Tagging the directory keeps them side by side. Without
    // the flag the name is exactly `ds_NN`, as it has always been.
    let tag = emit
        .map(|e| {
            let h = crate::hashing::sha256_hex(e.identity_repr().as_bytes());
            format!("-emit{}", &h[..8])
        })
        .unwrap_or_default();
    let mut out = Vec::with_capacity(seeds.len());
    for (i, &sim_seed) in seeds.iter().enumerate() {
        let idx = i + 1;
        let dir = data_dir.join(format!("{}{tag}", format_dataset_dir(idx)));
        let run = synthetic_sim_run(spec, model_path, sim_seed, dt)?;
        let written = simulate_dataset(&run, ObservationDesign::Declared { emit }, &dir)?;
        let files: IndexMap<String, PathBuf> =
            written.into_iter().map(|f| (f.source, f.path)).collect();
        out.push(SyntheticDataset { idx, files });
    }
    Ok(out)
}

/// The forward run one synthetic dataset is drawn from: the model at
/// `true_params`, under the `[synthetic]` scenario if declared, at `sim_seed`.
///
/// The observation RNG is decorrelated from the process RNG by
/// `util::SEED_MIX_OBS` inside `simulate_dataset` — the constant `camdl
/// simulate --obs-only-dir` shares — so the same nominal seed produces
/// identical observation bytes whether the dataset came from the CLI or from
/// `[synthetic]`. Diverging these constants in the past caused a
/// parameter-recovery discrepancy that looked like a +59% β bias; see the
/// 2026-04-18 downstream incident report.
fn synthetic_sim_run(
    spec: &SyntheticSpec,
    model_path: &str,
    sim_seed: u64,
    dt: f64,
) -> Result<SimRun, String> {
    let truth_overrides = load_params_toml(&spec.true_params)
        .map_err(|e| format!("parsing [synthetic] true_params {}: {}", spec.true_params, e))?;
    Ok(SimRun {
        ir_path: model_path.to_string(),
        overrides: truth_overrides,
        scenario_name: spec.scenario.clone(),
        // Synthetic-data generation runs the model forward before any
        // data exists, so an anchored model has nothing to anchor TO;
        // `CompiledModel::new` refuses it by name.
        obs_anchors: None,
        t_end_override: None, // fit refuses horizons (gh#561)
        init_state: None,     // synthetic data-gen starts from the model's init {}
        integrator: None,     // and uses the model's declared integrator
        backend: spec.backend,
        dt,
        seed: sim_seed,
        ..SimRun::default()
    })
}

/// Refuse an `--emit-every` cadence a stream's own `covers` declaration
/// contradicts (gh#656 × gh#833).
///
/// `--emit-every` re-spaces a stream declared with a uniform window of another
/// width; a `[synthetic]` dataset is fitted by this same model, which would
/// then read those rows as windows of the DECLARED width with gaps between.
/// Refused, not rescaled — the flag exists for `simulate --obs`, whose output
/// nothing has to read back.
fn check_emit_every_matches_declared_windows(
    model: &ir::Model,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<(), String> {
    let Some(emit) = emit else { return Ok(()) };
    for obs_ir in &model.observations {
        let (Some(step), Some(covers)) =
            (emit.resolve_for(&obs_ir.source), obs_ir.covers.as_ref())
        else {
            continue;
        };
        let Some((start, stop)) = covers.period_of(0.0) else { continue };
        let span = stop - start;
        if (span - step).abs() > crate::OBS_SNAP_EPS {
            return Err(format!(
                "--emit-every {step} would write '{}' rows covering {step} {unit} \
                 each, but the model declares `covers` with {span}-{unit} windows, \
                 and a [synthetic] dataset is fitted by this same model — which \
                 would read those rows as {span}-{unit} windows with gaps between. \
                 Drop the override for this stream, or declare the window you want \
                 to emit (`closing_at({time}, {step} '{unit})`) in the model.",
                obs_ir.name,
                unit = model.time_unit,
                time = crate::pfilter::obs_time_column(obs_ir).unwrap_or("time"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config_v2::SeedsSpec;

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

    fn spec_for(truth: &std::path::Path, seeds: Vec<u64>) -> SyntheticSpec {
        SyntheticSpec {
            true_params: truth.to_string_lossy().to_string(),
            sim_seeds: SeedsSpec::List(seeds),
            datasets: None,
            scenario: None,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
        }
    }

    #[test]
    fn generates_one_dataset_directory_per_sim_seed() {
        let Some(camdlc) = camdlc_path() else {
            eprintln!("skipping: camdlc.exe not built; run `cd ocaml && dune build` first");
            return;
        };
        let tmp = tempdir("one_per");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);

        let fit_dir = tmp.path().join("fit_out");
        let spec = spec_for(&truth_path, vec![1, 2, 3]);
        let datasets = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(), &fit_dir,
            1.0, None,
        ).expect("generation must succeed on minimal SIR");

        assert_eq!(datasets.len(), 3);
        for (i, ds) in datasets.iter().enumerate() {
            assert_eq!(ds.idx, i + 1);
            // One file per stream, keyed by the `source` the fit binds it to.
            let path = ds.files.get("cases")
                .unwrap_or_else(|| panic!("ds_{:02} must bind the `cases` source: {:?}",
                    i + 1, ds.files));
            assert!(path.exists(), "{} must exist", path.display());
            let contents = std::fs::read_to_string(path).unwrap();
            // The header names the DECLARED columns, so the file re-loads
            // under the model that generated it (gh#830, gh#833).
            assert_eq!(contents.lines().next().unwrap(), "time\tcases");
            assert!(contents.lines().count() >= 10,
                "≥10 daily obs rows expected, got {}",
                contents.lines().count().saturating_sub(1));
        }
        assert!(fit_dir.join("synthetic").join("truth.toml").exists(),
            "truth.toml must be copied for provenance");
    }

    #[test]
    fn same_seed_produces_identical_content() {
        let Some(camdlc) = camdlc_path() else { return; };
        let tmp = tempdir("det");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);
        let spec = spec_for(&truth_path, vec![42]);

        let a = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("run_a"), 1.0, None,
        ).unwrap();
        let b = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("run_b"), 1.0, None,
        ).unwrap();

        assert_eq!(std::fs::read(&a[0].files["cases"]).unwrap(),
                   std::fs::read(&b[0].files["cases"]).unwrap(),
                   "same seed + same truth must produce identical datasets");
    }

    #[test]
    fn different_seeds_produce_different_data() {
        let Some(camdlc) = camdlc_path() else { return; };
        let tmp = tempdir("diff");
        let (ir_path, truth_path) = write_fixture(tmp.path(), &camdlc);
        let spec = spec_for(&truth_path, vec![1, 999]);
        let ds = generate_synthetic_datasets(
            &spec, ir_path.to_str().unwrap(),
            &tmp.path().join("fit"), 1.0, None,
        ).unwrap();
        assert_ne!(std::fs::read(&ds[0].files["cases"]).unwrap(),
                   std::fs::read(&ds[1].files["cases"]).unwrap(),
                   "different sim seeds must produce different data realizations");
    }

    #[test]
    fn spec_roundtrips_seed_lists() {
        let s = SeedsSpec::List(vec![1, 2, 3]);
        assert_eq!(s.to_vec().unwrap(), vec![1u64, 2, 3]);
    }
}
