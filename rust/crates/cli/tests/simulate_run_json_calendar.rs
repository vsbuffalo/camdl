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
