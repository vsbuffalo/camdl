//! A simulation leaf's `run.json` carries the model's calendar axis
//! (`docs/dev/temporal-architecture.md` §5.3; M1, gh#838), so the stored
//! trajectory's `t` column can be dated without the model source.
//!
//! Shells out to the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

fn model(origin_line: &str) -> String {
    format!(
        r#"
time_unit = 'days
{origin_line}

compartments {{ S, I, R }}

let N = S + I + R

parameters {{
  beta  : rate in [0.01, 5.0]
  gamma : rate in [0.01, 5.0]
}}

init {{ S = 990  I = 10  R = 0 }}

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

simulate {{ from = 0 'days  to = 10 'days }}
"#
    )
}

/// The `run.json` of the one `Sim` leaf a `simulate` of `model_src` writes.
fn sim_record(model_src: &str) -> serde_json::Value {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.camdl"), model_src).unwrap();
    std::fs::write(dir.path().join("p.toml"), "beta = 0.4\ngamma = 0.2\n").unwrap();
    let out = Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir.path())
        .args([
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "0.25", "--seed", "3", "--output-dir", "out",
        ])
        .output()
        .expect("spawn camdl");
    assert!(out.status.success(), "simulate failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let mut stack = vec![dir.path().join("out").join("sims")];
    let mut records = Vec::new();
    while let Some(d) = stack.pop() {
        if d.join("run.json").is_file() {
            records.push(std::fs::read_to_string(d.join("run.json")).unwrap());
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            }
        }
    }
    assert_eq!(records.len(), 1, "one sim leaf");
    serde_json::from_str(&records[0]).unwrap()
}

#[test]
fn a_dated_simulation_record_carries_its_calendar() {
    let rec = sim_record(&model(r#"origin = date("2020-03-01")"#));
    assert_eq!(
        rec["calendar"],
        serde_json::json!({"days_per_unit": 1.0, "origin": "2020-03-01", "time_unit": "days"}),
        "run.json: {rec:#}"
    );
}

/// A model with no origin still records its axis, with the origin stated as
/// absent rather than left out.
#[test]
fn an_undated_simulation_record_says_it_has_no_origin() {
    let rec = sim_record(&model(""));
    assert_eq!(
        rec["calendar"],
        serde_json::json!({"days_per_unit": 1.0, "origin": null, "time_unit": "days"}),
        "run.json: {rec:#}"
    );
}

/// Fit a dated model with if2 in `dir`, writing `results/fits/…`.
fn run_dated_fit(dir: &Path) {
    let src = model(r#"origin = date("2020-03-01")"#).replace(
        "simulate {",
        "observations {\n  cases {\n    columns { time : time, cases : count }\n    \
         covers = closing_at(time, 1 'days)\n    projected = incidence(infection)\n    \
         cases ~ poisson(rate = projected)\n  }\n}\n\nsimulate {",
    );
    std::fs::write(dir.join("m.camdl"), src).unwrap();
    std::fs::write(dir.join("cases.tsv"),
        "time\tcases\n2020-03-02\t3\n2020-03-03\t4\n2020-03-04\t6\n").unwrap();
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"m.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = { bounds = [0.01, 5.0], start = 0.4 }\n[fixed]\ngamma = 0.2\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
         particles = 10\niterations = 1\ncooling = 0.7\n").unwrap();
    let out = camdl(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\n{}", String::from_utf8_lossy(&out.stderr));
}

fn camdl(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl")
}

/// Every file named `name` under `root`.
fn files_named(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join(name).is_file() {
            found.push(d.join(name));
        }
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            }
        }
    }
    found
}

/// Every record carries the axis, not only a simulation's (D59): a fit
/// method's `run.json` and its segment's `fit.meta.json` both record it.
#[test]
fn a_fit_records_its_calendar_in_run_json_and_fit_meta() {
    let dir = tempfile::tempdir().unwrap();
    run_dated_fit(dir.path());
    let expected = serde_json::json!({"days_per_unit": 1.0, "origin": "2020-03-01", "time_unit": "days"});
    let fits = dir.path().join("results").join("fits");
    let (runs, metas) = (files_named(&fits, "run.json"), files_named(&fits, "fit.meta.json"));
    assert!(!runs.is_empty() && !metas.is_empty(), "a fit writes both records");
    for path in runs.iter().chain(&metas) {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(v["calendar"], expected, "{}: {v:#}", path.display());
    }
}

/// A `run.json` written before the calendar block was required is named as
/// such: `camdl list` warns once and skips it, and `camdl show` refuses it
/// with the same sentence instead of a parse error.
#[test]
fn a_record_written_before_the_calendar_block_is_named_by_list_and_show() {
    let dir = tempfile::tempdir().unwrap();
    run_dated_fit(dir.path());
    let runs = files_named(&dir.path().join("results").join("fits"), "run.json");
    assert_eq!(runs.len(), 1, "one fit method leaf: {runs:?}");
    // The shape an older camdl wrote: the same record with no `calendar`.
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&runs[0]).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("calendar").expect("the block was there");
    std::fs::write(&runs[0], serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    let sentence = "written before the calendar axis was recorded";

    let out = camdl(dir.path(), &["list", "results"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(stderr.matches(sentence).count(), 1, "list names the record once:\n{stderr}");
    let rel = runs[0].strip_prefix(dir.path()).unwrap();
    assert!(stderr.contains(&rel.display().to_string()), "and its path:\n{stderr}");

    let leaf = runs[0].parent().unwrap();
    let out = camdl(dir.path(), &["show", leaf.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "show refuses it:\n{stderr}");
    assert!(stderr.contains(sentence) && !stderr.contains("missing field"),
        "show names why, not the parse error:\n{stderr}");
}
