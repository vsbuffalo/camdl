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

// ── `fit run` ────────────────────────────────────────────────────────────────

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    p.exists().then_some(p)
}

/// Compile the two-stream model to IR and write a truth file for `[synthetic]`.
fn fit_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc().expect("checked by the caller");
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(&camdlc).arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let truth = dir.join("truth.toml");
    std::fs::write(&truth, "beta = 0.5\ngamma = 0.1\n").unwrap();
    (ir, truth)
}

const FIT_STAGES: &str = r#"
[estimate]
beta = { bounds = [0.01, 2.0], start = 0.5 }

[fixed]
gamma = 0.1

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 3
cooling = 0.7
"#;

fn run_fit(fit_toml: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(binary())
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .args(extra)
        .arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run")
}

fn fit_bases(out: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(out.join("fits"))
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// On a fit with no `[synthetic]` block the flag can do nothing — a fit against
/// real data scores at its data file's own times. Refuse, naming why, rather
/// than accepting a flag whose effect is nil. And nothing may be created: a
/// completed real-data fit keeps its identity.
#[test]
fn fit_run_without_a_synthetic_block_refuses_the_flag_and_re_keys_nothing() {
    if skip_if_missing_binary() || camdlc().is_none() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (ir, _truth) = fit_fixture(tmp.path());
    let out_dir = tmp.path().join("out");

    // A real-data fit: two daily streams over the model window. Times start at
    // t = 1 — `cases` is an incidence stream, and t = 0 has no preceding
    // accumulation interval.
    let mut data = String::from("time\tcases\tprevalent\n");
    for t in 1..=28 {
        data.push_str(&format!("{t}\t{}\t{}\n", 5 + t % 4, 10 + t % 7));
    }
    let data_path = tmp.path().join("obs.tsv");
    std::fs::write(&data_path, &data).unwrap();

    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(
        &fit_toml,
        format!(
            r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"
prevalent = "{data}"
{stages}
"#,
            out = out_dir.display(),
            ir = ir.display(),
            data = data_path.display(),
            stages = FIT_STAGES,
        ),
    )
    .unwrap();

    let plain = run_fit(&fit_toml, &[]);
    assert!(
        plain.status.success(),
        "the real-data fit must run:\nstderr={}",
        String::from_utf8_lossy(&plain.stderr)
    );
    let bases_before = fit_bases(&out_dir);
    assert_eq!(bases_before.len(), 1, "one fit base: {bases_before:?}");

    let flagged = run_fit(&fit_toml, &["--emit-every", "7"]);
    let stderr = String::from_utf8_lossy(&flagged.stderr);
    assert!(
        !flagged.status.success(),
        "--emit-every on a real-data fit must refuse, not silently no-op\nstderr={stderr}"
    );
    assert!(
        stderr.contains("[synthetic]") && stderr.contains("never enters the likelihood"),
        "the refusal must say why the flag cannot apply, got:\n{stderr}"
    );
    assert_eq!(
        fit_bases(&out_dir),
        bases_before,
        "a real-data fit must not be re-keyed — or even touched — by this flag"
    );
}

/// `[synthetic]` is the one fit path the cadence reaches: it changes the data
/// that is then FITTED. So the generated file's times change, and — because the
/// fit hashes each training stream's bytes — the fit re-keys onto a new base.
/// That is correct: the data changed.
#[test]
fn a_synthetic_fit_re_keys_because_the_generated_data_changed() {
    if skip_if_missing_binary() || camdlc().is_none() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let (ir, truth) = fit_fixture(tmp.path());

    let fit_for = |tag: &str, out_dir: &Path| -> PathBuf {
        let p = tmp.path().join(format!("fit_{tag}.toml"));
        std::fs::write(
            &p,
            format!(
                r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[synthetic]
true_params = "{truth}"
sim_seeds = [1]
{stages}
"#,
                out = out_dir.display(),
                ir = ir.display(),
                truth = truth.display(),
                stages = FIT_STAGES,
            ),
        )
        .unwrap();
        p
    };

    // Both cadences into ONE store, so "did it re-key?" is "are there two
    // dataset-generation segments?" rather than a cross-store comparison.
    let out_dir = tmp.path().join("out");
    let toml = fit_for("a", &out_dir);

    for every in ["1", "7"] {
        let out = run_fit(&toml, &["--emit-every", every]);
        assert!(
            out.status.success(),
            "fit run --emit-every {every} must run:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // The generated datasets: one per cadence, with the cadence's own times.
    let mut generated: Vec<PathBuf> = Vec::new();
    let mut stack = vec![out_dir.clone()];
    while let Some(d) = stack.pop() {
        if d.file_name().map(|n| n == "data").unwrap_or(false)
            && d.parent().map(|p| p.ends_with("synthetic")).unwrap_or(false)
        {
            for f in std::fs::read_dir(&d).unwrap().flatten() {
                if f.path().extension().map(|e| e == "tsv").unwrap_or(false) {
                    generated.push(f.path());
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
    let mut row_counts: Vec<usize> =
        generated.iter().map(|p| obs_times(p).len()).collect();
    row_counts.sort();
    assert_eq!(
        row_counts,
        vec![5, 29],
        "each cadence must generate its own dataset (weekly = 5 rows, daily = \
         29) — got {generated:?}"
    );

    // …and the fit followed the data onto a distinct base. The fit-level
    // CONTAINER is keyed on model + config before any data exists, so both
    // cadences share it (which is why the generated files are tagged, above);
    // the CELL fits fold the generated bytes, so they are two.
    let cells = cell_fit_bases(&out_dir);
    assert_eq!(
        cells.len(),
        2,
        "each cadence's fit must key on its own generated data: {cells:?}"
    );
}

/// Fit bases holding at least one `fit_stage` leaf — the cell fits, as opposed
/// to the dataset-generation container (which carries no stage leaf).
fn cell_fit_bases(out: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(out.join("fits"))
        .map(|d| {
            d.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && has_stage_leaf(p))
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn has_stage_leaf(base: &Path) -> bool {
    let mut stack = vec![base.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(d.join("run.json")).unwrap_or_default(),
        ) {
            if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                return true;
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
    false
}
