//! `camdl profile --table NAME=FILE` supplies a model table, as it does on
//! `simulate` and `pfilter`: a model that declares an `external()` table is
//! profiled with the file's values, and two files give two computations.
//! The flag was accepted and never read, so such a model could not be
//! profiled at all (the run refused for want of the table it was given).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0]  rho : probability in [0.05, 0.95] }
tables {
  w : patch = external("w")
}
let N[l in patch] = S[l] + I[l] + R[l]
transitions {
  infection[l in patch] : S[l] --> I[l] @ beta * w[l] * S[l] * I[l] / N[l]
  recovery[l in patch]  : I[l] --> R[l] @ gamma * I[l]
}
init {
  S[p0] = 990  I[p0] = 10
  S[p1] = 1000  I[p1] = 5
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = sum(l in patch, incidence(infection[l]))
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected + 0.5)
  }
}
simulate { from = 0 'days  to = 10 'days }
"#;

fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl missing: {} — run `make build`", p.display());
    p
}

fn profile(dir: &Path, w: &str) -> Output {
    let p = |n: &str| dir.join(n).to_string_lossy().into_owned();
    Command::new(bin())
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        .env("CAMDL_STORE", dir.join("store"))
        .args(["profile", &p("patch.camdl"), "--fit", &p("fit.toml"), "--starts", &format!("from_params={}", p("start.toml"))])
        .args(["--table", &format!("w={}", p(w)), "--sweep", "gamma=0.2"])
        .args(["--rw-sd", "beta=0.05", "--particles", "20", "--iterations", "2", "--n-starts", "1", "--progress", "none"])
        .output()
        .expect("spawn camdl")
}

/// The run ids of the profile points stored under `dir`'s store.
fn points(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.join("store")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "identity.json") {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                if v["kind"] == "profile_point" {
                    out.push(v["run_id"].as_str().unwrap().to_string());
                }
            }
        }
    }
    out.sort();
    out
}

#[test]
fn profile_applies_the_table_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("patch.camdl"), PATCH).unwrap();
    std::fs::write(dir.join("w1.tsv"), "1.0\n0.5\n").unwrap();
    std::fs::write(dir.join("w2.tsv"), "1.0\n0.7\n").unwrap();
    std::fs::write(dir.join("start.toml"), "beta = 0.4\n").unwrap();
    std::fs::write(dir.join("cases.tsv"), (1..=10).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 2 + t))).unwrap();
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"patch.camdl\"\n\n[data.observations]\ncases = \"cases.tsv\"\n\n\
         [estimate]\nbeta = { bounds = [0.01, 2.0] }\ngamma = { bounds = [0.01, 1.0] }\n\n\
         [fixed]\nrho = 0.5\n").unwrap();
    let mut ids = Vec::new();
    for w in ["w1.tsv", "w2.tsv"] {
        let o = profile(dir, w);
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success(), "--table w={w} is applied:\n{err}");
        let found = points(dir);
        assert_eq!(found.len(), ids.len() + 1, "one more point: {found:?}");
        ids = found;
    }
}
