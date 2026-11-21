//! What each of `camdl pfilter`'s two observation-drawing diagnostics writes
//! is a function of the run's seed and inputs alone: passing `--trace` leaves
//! every `--save-prequential` file byte-identical, and passing
//! `--save-prequential` leaves the `--trace` file byte-identical.
//!
//! The two diagnostics draw observations from the same particles at the same
//! stops. If they shared a random stream, one diagnostic's draws at a stop
//! would advance the stream the other's are then drawn from: the CRPS and PIT
//! columns and the samples in the JSON trace, or the trace's `obs_q05`,
//! `obs_q50` and `obs_q95` columns, would change with a flag that is not meant
//! to touch them.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn model() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ir/golden/seir_spatial_5_inference.ir.json")
}

fn camdl(bin: &Path, args: &[&str]) -> std::process::Output {
    let out = Command::new(bin)
        .env_remove("CLICOLOR_FORCE").env("CAMDL_STORE", isolated_store())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .output()
        .expect("spawn camdl");
    assert!(
        out.status.success(),
        "camdl {:?} failed:\nstdout={}\nstderr={}",
        args,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// Runs `pfilter`, with `--save-prequential <dir>/preq` when `prequential` is
/// set and `--trace <dir>/trace.tsv` when `trace` is set, and returns the
/// prequential stem.
fn pfilter(bin: &Path, obs: &Path, dir: &Path, prequential: bool, trace: bool) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let stem = dir.join("preq");
    let trace_path = dir.join("trace.tsv");
    let model = model();
    let data1 = format!("cases_p1={}", obs.display());
    let data2 = format!("cases_p2={}", obs.display());
    let stem_s = stem.to_string_lossy().into_owned();
    let trace_s = trace_path.to_string_lossy().into_owned();
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--scenario", "true_params",
        "--data", &data1,
        "--data", &data2,
        "--particles", "200",
        "--dt", "1",
        "--seed", "7",
    ];
    if prequential {
        args.extend(["--save-prequential", &stem_s]);
    }
    if trace {
        args.extend(["--trace", &trace_s]);
    }
    camdl(bin, &args);
    if trace {
        assert!(trace_path.exists(), "--trace wrote no file");
    }
    stem
}

/// Simulates observations from the fixture into `dir/obs.tsv`.
fn synth_obs(bin: &Path, dir: &Path) -> PathBuf {
    let obs = dir.join("obs.tsv");
    camdl(bin, &[
        "simulate", model().to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", "42",
        "--scenario", "true_params",
        "--obs-only", obs.to_str().unwrap(),
    ]);
    obs
}

fn release_binary() -> PathBuf {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} - run `make build-rust`", bin.display());
    bin
}

#[test]
fn save_prequential_files_are_identical_with_and_without_trace() {
    let bin = release_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, tmp.path());

    let without = pfilter(&bin, &obs, &tmp.path().join("without"), true, false);
    let with = pfilter(&bin, &obs, &tmp.path().join("with"), true, true);

    for ext in ["tsv", "streams.tsv", "json"] {
        let a = std::fs::read(without.with_extension(ext))
            .unwrap_or_else(|e| panic!("read preq.{ext} (without --trace): {e}"));
        let b = std::fs::read(with.with_extension(ext))
            .unwrap_or_else(|e| panic!("read preq.{ext} (with --trace): {e}"));
        assert!(!a.is_empty(), "preq.{ext} is empty");
        assert!(
            a == b,
            "preq.{ext} differs when --trace is passed beside --save-prequential \
             (one seed, same inputs): the prequential samples depend on --trace"
        );
    }
}

#[test]
fn trace_file_is_identical_with_and_without_save_prequential() {
    let bin = release_binary();
    let tmp = tempfile::tempdir().unwrap();
    let obs = synth_obs(&bin, tmp.path());

    let without = tmp.path().join("without");
    let with = tmp.path().join("with");
    pfilter(&bin, &obs, &without, false, true);
    pfilter(&bin, &obs, &with, true, true);

    let a = std::fs::read(without.join("trace.tsv")).expect("read trace (without --save-prequential)");
    let b = std::fs::read(with.join("trace.tsv")).expect("read trace (with --save-prequential)");
    let header = String::from_utf8_lossy(&a).lines().next().unwrap_or("").to_string();
    assert!(header.contains("obs_q05"),
        "the trace carries no prediction columns, so it cannot show the dependence: {header}");
    assert!(
        a == b,
        "trace.tsv differs when --save-prequential is passed beside --trace \
         (one seed, same inputs): the prediction draws depend on --save-prequential"
    );
}

/// A run store of its own for one command, so tests that run the same filter
/// at once do not meet in a shared root (a filter run another live process
/// holds is refused).
fn isolated_store() -> std::path::PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "camdl-store-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}
