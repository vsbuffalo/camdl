//! A fit store made by the release `camdl` for tests of the readers of a
//! stored fit and of `camdl dev migrate`: a small SIR model observed weekly,
//! a PMMH fit config, and helpers to run `camdl` in a cleared environment and
//! to read the store's files.

#![allow(dead_code)] // each test file uses part of it

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability
}

let N = S + I + R

transitions {
  infection : S --> I @ beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10 }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }
}

simulate { from = 0 'days  to = 70 'days  dt = 1 'days }
"#;

pub const METHOD: &str = "[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nparticles = 20\n\
                      chains = 1\niterations = 20\nburn_in = 5\n";

/// A fit config: `fixed` is the body of its `[fixed]`.
pub fn fit_config(fixed: &str) -> String {
    format!(
        "[model]\ncamdl = \"m.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.3, prior = {{ uniform = {{}} }} }}\n[fixed]\n{fixed}\n{METHOD}"
    )
}

pub struct Fixture {
    pub dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
}

impl Fixture {
    pub fn new(name: &str) -> Fixture {
        static N: AtomicUsize = AtomicUsize::new(0);
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        let version = Command::new(&camdl).arg("--version").output().unwrap();
        let version = String::from_utf8_lossy(&version.stdout);
        assert!(
            version.contains(&format!("+{} ", env!("CAMDL_SOURCE_LABEL"))),
            "{} is not built from this source ({}): {version} — run `make build`",
            camdl.display(),
            env!("CAMDL_SOURCE_LABEL")
        );
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{name}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("home")).unwrap();
        let fx = Fixture { dir, camdl, camdlc };
        fx.write("m.camdl", MODEL);
        let cases: String = (1..=10).map(|w| format!("{}\t{}\n", 7 * w, 2 + 3 * w)).collect();
        fx.write("cases.tsv", &format!("time\tcases\n{cases}"));
        fx
    }

    pub fn write(&self, name: &str, text: &str) {
        std::fs::write(self.dir.join(name), text).unwrap();
    }

    pub fn camdl(&self, args: &[&str]) -> Output {
        Command::new(&self.camdl)
            .current_dir(&self.dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.dir.join("home"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &self.camdlc)
            .env("CAMDL_IR_CACHE_DIR", self.dir.join("ir-cache"))
            .args(args)
            .output()
            .expect("spawn camdl")
    }

    /// `fit run`, returning every leaf it stored or reused, relative to the
    /// fixture directory.
    pub fn fit(&self, args: &[&str]) -> Vec<String> {
        let mut all = vec!["fit", "run"];
        all.extend(args);
        all.extend(["--progress", "none"]);
        let stderr = ok(&self.camdl(&all));
        let leaves: Vec<String> = stderr
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.strip_prefix("stored ").and_then(|r| r.split(" · ").nth(1)).map(str::to_string)
                    .or_else(|| l.strip_prefix("cache hit — reusing ").map(|r| format!("results/{r}")))
            })
            .collect();
        assert!(!leaves.is_empty(), "no stored leaf in:\n{stderr}");
        leaves
    }

    /// The fit folder holding `leaf`.
    pub fn folder(&self, leaf: &str) -> PathBuf {
        self.dir.join(leaf).ancestors().nth(2).unwrap().to_path_buf()
    }

    /// `fit predict <leaf>`'s files, by path under the folder's `predict/`.
    pub fn predict(&self, leaf: &str) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let folder = self.folder(leaf);
        let _ = std::fs::remove_dir_all(folder.join("predict"));
        let out = self.camdl(&["fit", "predict", &self.dir.join(leaf).to_string_lossy(), "--n-draws", "5",
                               "--seed", "1", "--progress", "none"]);
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).into_owned());
        }
        Ok(files_under(&folder.join("predict")))
    }

    pub fn migrate(&self, args: &[&str]) -> Output {
        let mut all = vec!["dev", "migrate"];
        all.extend(args);
        self.camdl(&all)
    }
}

pub fn ok(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "camdl failed:\n{stderr}\n{}", String::from_utf8_lossy(&out.stdout));
    stderr
}

pub fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

pub fn files_under(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.strip_prefix(dir).unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// Every file named `name` under `dir`.
pub fn named(dir: &Path, name: &str) -> Vec<PathBuf> {
    files_under(dir).into_keys().filter(|k| k.ends_with(name)).map(|k| dir.join(k)).collect()
}

pub fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn set_json(path: &Path, value: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

