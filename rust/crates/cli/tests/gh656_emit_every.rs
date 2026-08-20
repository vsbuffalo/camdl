//! gh#656: `--emit-every`, a per-stream override of a model's emission cadence.
//!
//! `emit_schedule` decides at what times a FORWARD simulation emits synthetic
//! observations. It never enters a likelihood — a fit against real data scores
//! at its data file's own times — so one model can serve a daily and a weekly
//! emission without a regex rewrite of its source.
//!
//! The load-bearing half of this file is IDENTITY. The override is applied at
//! the consumption sites, never by rematerializing the compiled IR the way
//! `--output-every` does, so:
//!
//!   - two cadences address two DISTINCT obs artifacts (they carry different
//!     times, so sharing one address would serve one for the other);
//!   - the same cadence twice addresses the same one;
//!   - the trajectory leaf's `run_id` — the model identity a `fit` against REAL
//!     data shares — does NOT move, so a cosmetic emission change cannot orphan
//!     a completed fit (the gh#653 harm, which an IR rewrite would reintroduce).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// Two streams on a daily cadence, so a per-stream override has a sibling to
/// leave alone.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = projected)
  }
  prevalent {
    columns       { time : time, prevalent : count }
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    prevalent     ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 28 'days }
"#;

/// One stream whose emission times are a fixed list, not a cadence.
const AT_LIST_MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = at [0 'days, 7 'days, 14 'days, 21 'days]
    cases         ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 28 'days }
"#;

fn setup(dir: &Path, src: &str) -> (PathBuf, PathBuf) {
    let m = dir.join("m.camdl");
    std::fs::write(&m, src).unwrap();
    let p = dir.join("p.toml");
    std::fs::write(&p, "beta = 0.5\ngamma = 0.1\n").unwrap();
    (m, p)
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn skip_if_missing_binary() -> bool {
    if binary().exists() {
        return false;
    }
    eprintln!("skipping: release binary not built; run `make build` first");
    true
}

/// The `time` column of a one-stream obs TSV (header + `time\t<stream>` rows).
fn obs_times(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').next().unwrap().parse::<f64>().unwrap())
        .collect()
}

/// Every dir containing a `run.json`, at any depth — the factored CAS sim path
/// is several levels deep, and the obs ensemble is a declared `obs/` child below
/// it (which carries no `run.json` of its own).
fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                }
            }
        }
    }
    out
}

/// The `obs/{obs_hash8}-{obs_seed}` subtree names under a sim leaf — the obs
/// artifacts' content addresses.
fn obs_subtree_names(leaf: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(leaf.join("obs"))
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn run_id_of(leaf: &Path) -> String {
    let txt = std::fs::read_to_string(leaf.join("run.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&txt).unwrap();
    v["run_id"].as_str().expect("run.json carries a run_id").to_string()
}

// ── Behaviour ────────────────────────────────────────────────────────────────

#[test]
fn the_bare_form_sets_every_streams_cadence() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let dir = tmp.path().join("obs");

    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        "--emit-every", "7",
        "--obs-dir", dir.to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--emit-every 7 must run:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    for stream in ["cases", "prevalent"] {
        let times = obs_times(&dir.join(format!("{stream}.tsv")));
        assert_eq!(
            times,
            vec![0.0, 7.0, 14.0, 21.0, 28.0],
            "'{stream}' must emit weekly, not on its declared daily cadence"
        );
    }
}

#[test]
fn the_labelled_form_sets_one_stream_and_leaves_its_sibling_alone() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let dir = tmp.path().join("obs");

    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        "--emit-every", "cases=7",
        "--obs-dir", dir.to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "--emit-every cases=7 must run:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(
        obs_times(&dir.join("cases.tsv")),
        vec![0.0, 7.0, 14.0, 21.0, 28.0],
        "the named stream must take the override"
    );
    assert_eq!(
        obs_times(&dir.join("prevalent.tsv")).len(),
        29,
        "the unnamed sibling must keep its declared daily cadence (t = 0..28)"
    );
}

// ── Refusals ─────────────────────────────────────────────────────────────────

#[test]
fn mixing_the_bare_and_labelled_forms_is_refused() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--emit-every", "cases=1",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "mixed forms must be refused\nstderr={stderr}");
    assert!(stderr.contains("mutually exclusive"), "stderr={stderr}");
}

#[test]
fn an_unknown_stream_label_lists_the_valid_ones() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "caes=7",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a typo'd label must be refused\nstderr={stderr}");
    assert!(
        stderr.contains("'caes' is not an observation stream")
            && stderr.contains("cases")
            && stderr.contains("prevalent"),
        "the error must list the valid labels, got:\n{stderr}"
    );
}

#[test]
fn an_at_list_stream_is_refused_by_name() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), AT_LIST_MODEL);
    let obs = tmp.path().join("obs.tsv");
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--obs", obs.to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "converting a fixed emission list to a cadence must be refused\nstderr={stderr}"
    );
    assert!(
        stderr.contains("'cases'") && stderr.contains("at [...]"),
        "the refusal must name the stream and its declared form, got:\n{stderr}"
    );
    assert!(!obs.exists(), "nothing may be emitted when the override is refused");
}

#[test]
fn a_run_that_emits_no_observations_refuses_the_flag() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "7",
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a flag that could do nothing must refuse, not silently no-op\nstderr={stderr}"
    );
    assert!(stderr.contains("emits none"), "stderr={stderr}");
}

#[test]
fn the_dsl_tick_spelling_is_refused_with_a_plain_number_hint() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--emit-every", "1 'weeks",
        "--obs-dir", tmp.path().join("obs").to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "stderr={stderr}");
    assert!(
        stderr.contains("tick") && stderr.contains("plain number"),
        "the tick spelling must hint the plain-number form, got:\n{stderr}"
    );
}

// ── Identity ─────────────────────────────────────────────────────────────────

/// Two cadences are two artifacts, and one cadence twice is one artifact.
///
/// The obs subtree is addressed `obs/{obs_hash8}-{obs_seed}` under the sim leaf,
/// and `obs_hash` folds the override (`batch::obs_subtree_hash`). Drop that fold
/// and the two cadences collide on one directory — the second run silently
/// overwriting the first's weekly file with daily rows under an address that
/// still claims to be the first's.
#[test]
fn distinct_cadences_are_distinct_obs_artifacts_and_the_same_cadence_collides() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let store = tmp.path().join("r");

    let sim = |every: &str, obs_dir: &str| {
        let out = run(&[
            "simulate", model.to_str().unwrap(),
            "--params", params.to_str().unwrap(),
            "--seed", "3",
            "--emit-every", every,
            "--obs-dir", obs_dir,
            "--output-dir", store.to_str().unwrap(),
        ]);
        assert!(
            out.status.success(),
            "--emit-every {every} must run:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    sim("7", tmp.path().join("o7").to_str().unwrap());
    let leaves = run_leaves(&store);
    assert_eq!(leaves.len(), 1, "one cell, one sim leaf: {leaves:?}");
    let leaf = leaves[0].clone();
    let after_7 = obs_subtree_names(&leaf);
    assert_eq!(after_7.len(), 1, "the first run writes one obs subtree: {after_7:?}");

    // Same cadence again: the same address, no second artifact.
    sim("7", tmp.path().join("o7b").to_str().unwrap());
    assert_eq!(
        obs_subtree_names(&leaf),
        after_7,
        "the same cadence must address the same obs artifact"
    );

    // A different cadence: a second, distinct address alongside the first.
    sim("14", tmp.path().join("o14").to_str().unwrap());
    let after_14 = obs_subtree_names(&leaf);
    assert_eq!(
        after_14.len(),
        2,
        "a different cadence must be a DISTINCT obs artifact, not an \
         overwrite of the first: {after_14:?}"
    );
    assert!(after_14.contains(&after_7[0]), "the weekly artifact must survive: {after_14:?}");

    // And the two carry what they claim to.
    assert_eq!(obs_times(&tmp.path().join("o7").join("cases.tsv")).len(), 5);
    assert_eq!(obs_times(&tmp.path().join("o14").join("cases.tsv")).len(), 3);
}

/// The model identity does not move.
///
/// A `fit` against REAL data keys on the model's IR (`FitDigest.model`), which
/// is the same digest the sim leaf's `run_id` folds. Lowering `--emit-every`
/// into the IR — the way `--output-every` does — would move that digest, so a
/// purely cosmetic change to emitted output would orphan a completed fit. It
/// must not: same leaf, same `run_id`, with and without the flag.
#[test]
fn the_flag_does_not_move_the_model_identity_a_real_data_fit_keys_on() {
    if skip_if_missing_binary() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path(), MODEL);
    let store = tmp.path().join("r");
    let model_bytes_before = std::fs::read(&model).unwrap();

    let base = |extra: &[&str], obs_dir: &str| {
        let mut args: Vec<String> = vec![
            "simulate".into(), model.to_string_lossy().into(),
            "--params".into(), params.to_string_lossy().into(),
            "--seed".into(), "3".into(),
            "--obs-dir".into(), obs_dir.into(),
            "--output-dir".into(), store.to_string_lossy().into(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let out = run(&refs);
        assert!(
            out.status.success(),
            "run must succeed:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    base(&[], tmp.path().join("plain").to_str().unwrap());
    let plain_leaves = run_leaves(&store);
    assert_eq!(plain_leaves.len(), 1);
    let plain_id = run_id_of(&plain_leaves[0]);

    base(&["--emit-every", "7"], tmp.path().join("weekly").to_str().unwrap());
    let after = run_leaves(&store);
    assert_eq!(
        after.len(),
        1,
        "--emit-every must not fork the trajectory leaf — the trajectory bytes \
         do not depend on the emission cadence: {after:?}"
    );
    assert_eq!(
        run_id_of(&after[0]),
        plain_id,
        "the sim run_id — the model identity a real-data fit shares — must be \
         byte-identical with and without --emit-every"
    );
    assert_eq!(
        std::fs::read(&model).unwrap(),
        model_bytes_before,
        "the model source must not be rewritten"
    );
    // Two cadences, two obs artifacts, one trajectory.
    assert_eq!(obs_subtree_names(&after[0]).len(), 2);
}
