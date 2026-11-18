//! External inputs end to end: a model's `external("NAME")` tables and
//! `data = external("NAME")` forcings, bound to files when the model is run.
//!
//! Drives the release `camdl` (built from this source, checked) against models
//! compiled by this source's `camdlc`:
//!
//! - **Parity.** A forcing bound at run time reads its file exactly as the
//!   compiler reads `data = "path"`: the two forms give byte-identical
//!   trajectories over one file that exercises every reading rule (a `#`
//!   comment, an indexed forcing's key column, a `'per_week` unit scale, ISO
//!   dates under an origin, the `constant` method).
//! - **Refusals**, each naming the external input: no binding, a binding the
//!   model does not declare, a missing column, no rows for a stratum, a table
//!   file with the wrong number of cells.
//! - **Identity.** The bound file's bytes are a run input: the same bytes under
//!   another path give the same run id, and one added byte gives a new one.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A model whose two forcings read files written into the model, and the same
/// model reading them as external inputs. Everything else is identical.
fn model(imp: &str, school: &str) -> String {
    format!(
        r#"time_unit = 'days
origin = date("2024-01-01")
compartments {{ S, I }}
dimensions {{ patch = [north, south] }}
stratify(by = patch)
let N[p in patch] = S[p] + I[p]
parameters {{ beta : rate  gamma : rate }}
forcing {{
  school : interpolated 'ratio {{
    data = {school}  time_col = "date"  value_col = "in_term"  method = constant
  }}
  imp[p in patch] : interpolated 'per_week {{
    data = {imp}  key_col = "patch"  time_col = "t"  value_col = "v"
  }}
}}
transitions {{
  infection[p in patch] : S[p] --> I[p] @ (beta * school * I[p] / N[p] + imp[p]) * S[p]
  recovery[p in patch]  : I[p] --> S[p] @ gamma * I[p]
}}
init {{ S[north] = 990  I[north] = 10  S[south] = 495  I[south] = 5 }}
simulate {{ from = 0 'days  to = 20 'days  dt = 0.25 'days }}
"#
    )
}

const IMP: &str = "# provenance: synthetic test data\n\
                   patch\tt\tv\n\
                   north\t0\t0.07\nnorth\t10\t0.14\nnorth\t20\t0.07\n\
                   south\t0\t0.035\nsouth\t10\t0\nsouth\t20\t0.035\n";

const SCHOOL: &str = "date\tin_term\n2024-01-01\t1\n2024-01-06\t0\n2024-01-13\t1\n2024-01-21\t1\n";

/// A table read at run time, over a dimension of three levels.
const TABLE_MODEL: &str = r#"time_unit = 'days
compartments { S, I }
dimensions { patch = [a, b, c] }
stratify(by = patch)
parameters { gamma : rate }
tables { w : patch = external("w") }
transitions { recovery[p in patch] : I[p] --> S[p] @ gamma * w[p] * I[p] }
init { S[a] = 10  S[b] = 10  S[c] = 10  I[a] = 5  I[b] = 5  I[c] = 5 }
simulate { from = 0 'days  to = 5 'days }
"#;

struct Fixture {
    dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
}

impl Fixture {
    fn new() -> Fixture {
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
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "external_inputs-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("home")).unwrap();
        let fx = Fixture { dir, camdl, camdlc };
        fx.write("imp.tsv", IMP);
        fx.write("school.tsv", SCHOOL);
        fx.write("file.camdl", &model("\"imp.tsv\"", "\"school.tsv\""));
        fx.write("ext.camdl", &model("external(\"imp\")", "external(\"school\")"));
        fx.write("table.camdl", TABLE_MODEL);
        fx
    }

    fn write(&self, name: &str, text: &str) {
        let p = self.dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(p, text).unwrap();
    }

    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }

    /// `camdl simulate` in a cleared environment, into `store` under the
    /// fixture directory.
    fn simulate(&self, store: &str, args: &[&str]) -> Output {
        Command::new(&self.camdl)
            .current_dir(&self.dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.dir.join("home"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &self.camdlc)
            .env("CAMDL_IR_CACHE_DIR", self.dir.join("ir-cache"))
            .arg("simulate")
            .args(args)
            .args(["--progress", "none", "--output-dir"])
            .arg(self.dir.join(store))
            .output()
            .expect("spawn camdl simulate")
    }
}

fn ok(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "camdl failed:\n{stderr}");
    stderr
}

fn refused(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!out.status.success(), "camdl should have refused, but succeeded:\n{stderr}");
    stderr
}

/// The run id `simulate` reports (`camdl cat <id>`).
fn run_id(stderr: &str) -> String {
    stderr
        .lines()
        .find_map(|l| l.trim().strip_prefix("camdl cat ").map(str::to_string))
        .unwrap_or_else(|| panic!("no `camdl cat <id>` line in:\n{stderr}"))
}

/// The one trajectory under `store`.
fn trajectory(fx: &Fixture, store: &str) -> String {
    let mut stack = vec![fx.dir.join(store)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "traj.tsv") {
                return std::fs::read_to_string(p).unwrap();
            }
        }
    }
    panic!("no traj.tsv under {store}");
}

const PARAMS: [&str; 6] = ["--param", "beta=0.3", "--param", "gamma=0.1", "--seed", "7"];

fn ext_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["ext.camdl"];
    v.extend(PARAMS);
    v.extend(extra);
    v
}

#[test]
fn external_forcing_matches_file_forcing() {
    let fx = Fixture::new();
    let mut file_args = vec!["file.camdl"];
    file_args.extend(PARAMS);
    ok(&fx.simulate("file-store", &file_args));
    ok(&fx.simulate("ext-store", &ext_args(&["--table", "imp=imp.tsv", "--table", "school=school.tsv"])));
    let file = trajectory(&fx, "file-store");
    let ext = trajectory(&fx, "ext-store");
    assert!(file.lines().count() > 20, "the trajectory should cover the run:\n{file}");
    assert_eq!(
        file, ext,
        "a forcing bound at run time must read its file exactly as the compiler reads data = \"path\""
    );
}

#[test]
fn unbound_external_forcing_is_refused_naming_it() {
    let fx = Fixture::new();
    let err = refused(&fx.simulate("s", &ext_args(&["--table", "school=school.tsv"])));
    assert!(err.contains("external input 'imp' has no file"), "{err}");
    assert!(err.contains("forcing 'imp_north'") && err.contains("forcing 'imp_south'"), "{err}");
    assert!(err.contains("--table imp=FILE") && err.contains("[data.tables] imp"), "{err}");
}

#[test]
fn binding_an_undeclared_name_is_refused() {
    let fx = Fixture::new();
    let err = refused(&fx.simulate(
        "s",
        &ext_args(&["--table", "imp=imp.tsv", "--table", "school=school.tsv", "--table", "schol=school.tsv"]),
    ));
    assert!(err.contains("external input 'schol' is bound (--table schol="), "{err}");
    assert!(err.contains("does not declare it") && err.contains("imp, school"), "{err}");
}

#[test]
fn forcing_file_without_its_column_is_refused() {
    let fx = Fixture::new();
    fx.write("school_bad.tsv", "date\tterm\n2024-01-01\t1\n");
    let err = refused(&fx.simulate("s", &ext_args(&["--table", "imp=imp.tsv", "--table", "school=school_bad.tsv"])));
    assert!(err.contains("external input 'school'") && err.contains("forcing 'school'"), "{err}");
    assert!(err.contains("no column 'in_term'") && err.contains("date, term"), "{err}");
}

#[test]
fn forcing_file_without_rows_for_a_stratum_is_refused() {
    let fx = Fixture::new();
    fx.write("imp_north_only.tsv", "patch\tt\tv\nnorth\t0\t0.1\nnorth\t20\t0.1\n");
    let err = refused(&fx.simulate(
        "s",
        &ext_args(&["--table", "imp=imp_north_only.tsv", "--table", "school=school.tsv"]),
    ));
    assert!(err.contains("forcing 'imp_south'") && err.contains("no rows where column 'patch' = 'south'"), "{err}");
}

#[test]
fn table_file_with_the_wrong_cell_count_is_refused() {
    let fx = Fixture::new();
    fx.write("w3.tsv", "1\n2\n3\n");
    fx.write("w4.tsv", "1\n2\n3\n4\n");
    let args = |w: &'static str| vec!["table.camdl", "--param", "gamma=0.1", "--seed", "1", "--table", w];
    ok(&fx.simulate("s3", &args("w=w3.tsv")));
    let err = refused(&fx.simulate("s4", &args("w=w4.tsv")));
    assert!(err.contains("external input 'w'") && err.contains("table 'w'"), "{err}");
    assert!(err.contains("patch (3 levels) — 3 cells") && err.contains("holds 4 numbers"), "{err}");
}

#[test]
fn bound_bytes_not_path_are_the_identity() {
    let fx = Fixture::new();
    fx.write("copy/imp.tsv", IMP);
    fx.write("edited/imp.tsv", &format!("{IMP}# one more line\n"));
    let id = |store: &str, imp: &str| {
        let imp = format!("imp={imp}");
        run_id(&ok(&fx.simulate(store, &ext_args(&["--table", &imp, "--table", "school=school.tsv"]))))
    };
    let a = id("a", "imp.tsv");
    let b = id("b", &fx.path("copy/imp.tsv"));
    let c = id("c", "edited/imp.tsv");
    assert_eq!(a, b, "the same bytes under another path are the same run");
    assert_ne!(a, c, "one changed byte in a bound file is a different run");
}
