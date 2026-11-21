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
//! - **Fit configs.** `[data.tables]` binds the same inputs for `fit run`,
//!   `pfilter --fit` and `profile --fit`, with `--table` laid
//!   over it; two calendars are two fits with two log-likelihoods; a fit
//!   records each bound file's digest and stores its bytes; `fit predict` and
//!   `simulate --fit` replay those stored bytes after the source is edited.

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

// ── Fits: `[data.tables]` in the fit config ─────────────────────────────────

/// A model fitted to daily cases, whose transmission is scaled by a school
/// calendar read from the external input `school`.
const FIT_MODEL: &str = r#"time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 2.0]  ~ log_normal(mu = -1.0, sigma = 0.3)
  gamma : rate        in [0.01, 1.0]  ~ log_normal(mu = -1.5, sigma = 0.3)
  rho   : probability in [0.05, 0.95]
}
forcing {
  school : interpolated 'ratio {
    data = external("school")  time_col = "t"  value_col = "in_term"  method = constant
  }
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * school * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10  R = 0 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected)
  }
}
simulate { from = 0 'days  to = 20 'days }
"#;

/// Two school calendars: term breaks (transmission × 0.6) at different times.
const CALENDAR_A: &str = "t\tin_term\n0\t1\n7\t0.6\n14\t1\n20\t1\n";
const CALENDAR_B: &str = "t\tin_term\n0\t1\n3\t0.6\n10\t1\n20\t1\n";

const PFILTER_METHOD: &str =
    "[method]\nalgorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 30\nreplicates = 1\n";
const PMMH_METHOD: &str = "[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nparticles = 20\n\
                           chains = 2\niterations = 20\nburn_in = 5\n";

/// A fit config whose `[data]` ends with `tables` (the `[data.tables]` body,
/// or nothing to leave the section out).
fn fit_config(tables: &str, method: &str) -> String {
    let tables = if tables.is_empty() { String::new() } else { format!("[data.tables]\n{tables}\n") };
    format!(
        "[model]\ncamdl = \"fit.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n{tables}\
         [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\n[fixed]\ngamma = 0.2\nrho = 0.5\n{method}"
    )
}

impl Fixture {
    fn with_fit_files() -> Fixture {
        let fx = Fixture::new();
        fx.write("fit.camdl", FIT_MODEL);
        let cases: String = (1..=20).map(|t| format!("{t}\t{}\n", (2.0 * 1.12f64.powi(t)).round())).collect();
        fx.write("cases.tsv", &format!("time\tcases\n{cases}"));
        fx.write("cal/a.tsv", CALENDAR_A);
        fx.write("cal/b.tsv", CALENDAR_B);
        fx
    }

    /// Any `camdl` subcommand in the cleared environment, in the fixture
    /// directory (so a fit stores under `results/` there).
    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(&self.camdl)
            .current_dir(&self.dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.dir.join("home"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &self.camdlc)
            .env("CAMDL_IR_CACHE_DIR", self.dir.join("ir-cache"))
            .args(args)
            .args(["--progress", "none"])
            .output()
            .expect("spawn camdl")
    }
}

/// The leaf `fit run` stored (`stored <method> · <leaf>`) or reused (`cache
/// hit — reusing <leaf under results/>`), relative to the fixture directory.
fn stored_leaf(stderr: &str) -> String {
    stderr
        .lines()
        .find_map(|l| {
            let l = l.trim();
            l.strip_prefix("stored ").and_then(|r| r.split(" · ").nth(1)).map(str::to_string)
                .or_else(|| l.strip_prefix("cache hit — reusing ").map(|r| format!("results/{r}")))
        })
        .unwrap_or_else(|| panic!("no `stored … · <leaf>` or `cache hit — reusing <leaf>` line in:\n{stderr}"))
}

fn run_json(fx: &Fixture, leaf: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fx.dir.join(leaf).join("run.json")).unwrap()).unwrap()
}

/// The single log-likelihood a `pfilter` leaf recorded.
fn leaf_loglik(fx: &Fixture, leaf: &str) -> f64 {
    let text = std::fs::read_to_string(fx.dir.join(leaf).join("logliks.tsv")).unwrap();
    text.lines().nth(1).and_then(|l| l.split('\t').nth(1)).unwrap().parse().unwrap()
}

/// The number line `camdl pfilter` prints: its log-likelihood.
fn printed_loglik(stdout_and_stderr: &Output) -> f64 {
    let text = format!("{}{}", String::from_utf8_lossy(&stdout_and_stderr.stdout), String::from_utf8_lossy(&stdout_and_stderr.stderr));
    text.lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no log-likelihood line in:\n{text}"))
}

fn sha256_hex(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
fn fit_under_two_tables_gives_two_fits_and_two_logliks() {
    let fx = Fixture::with_fit_files();
    fx.write("a.toml", &fit_config("school = \"cal/a.tsv\"", PFILTER_METHOD));
    fx.write("b.toml", &fit_config("school = \"cal/b.tsv\"", PFILTER_METHOD));
    let a = stored_leaf(&ok(&fx.camdl(&["fit", "run", "a.toml"])));
    let b = stored_leaf(&ok(&fx.camdl(&["fit", "run", "b.toml"])));
    assert_ne!(run_json(&fx, &a)["run_id"], run_json(&fx, &b)["run_id"], "two calendars are two fits");
    let (la, lb) = (leaf_loglik(&fx, &a), leaf_loglik(&fx, &b));
    assert!(la.is_finite() && lb.is_finite() && la != lb, "two calendars must score differently: {la} vs {lb}");

    // The fit folder records the bound file's digest and stores its bytes.
    for (leaf, cal) in [(&a, "cal/a.tsv"), (&b, "cal/b.tsv")] {
        let folder = fx.dir.join(leaf).ancestors().nth(2).unwrap().to_path_buf();
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("fit.meta.json")).unwrap()).unwrap();
        assert_eq!(meta["table_hashes"]["school"], sha256_hex(&fx.dir.join(cal)), "{leaf}: recorded digest");
        assert_eq!(
            std::fs::read(folder.join("inputs/school.tsv")).unwrap(),
            std::fs::read(fx.dir.join(cal)).unwrap(),
            "{leaf}: the stored copy holds the bound bytes"
        );
    }
}

#[test]
fn same_table_bytes_under_two_paths_is_one_fit() {
    let fx = Fixture::with_fit_files();
    fx.write("copy/school.tsv", CALENDAR_A);
    fx.write("edited/school.tsv", &format!("{CALENDAR_A}# one more line\n"));
    fx.write("a.toml", &fit_config("school = \"cal/a.tsv\"", PFILTER_METHOD));
    fx.write("a_copy.toml", &fit_config("school = \"copy/school.tsv\"", PFILTER_METHOD));
    fx.write("a_edited.toml", &fit_config("school = \"edited/school.tsv\"", PFILTER_METHOD));
    let id = |config: &str| {
        let leaf = stored_leaf(&ok(&fx.camdl(&["fit", "run", config])));
        run_json(&fx, &leaf)["run_id"].as_str().unwrap().to_string()
    };
    let (a, copy, edited) = (id("a.toml"), id("a_copy.toml"), id("a_edited.toml"));
    assert_eq!(a, copy, "the same bytes under another path are the same fit");
    assert_ne!(a, edited, "one changed byte in a bound file is a different fit");
}

#[test]
fn replays_of_a_stored_fit_read_its_stored_table() {
    let fx = Fixture::with_fit_files();
    fx.write("school.tsv", CALENDAR_A);
    fx.write("pm.toml", &fit_config("school = \"school.tsv\"", PMMH_METHOD));
    let leaf = stored_leaf(&ok(&fx.camdl(&["fit", "run", "pm.toml"])));
    let folder = fx.dir.join(&leaf).ancestors().nth(2).unwrap().to_path_buf();

    let predict = || -> std::collections::BTreeMap<String, Vec<u8>> {
        let _ = std::fs::remove_dir_all(folder.join("predict"));
        ok(&fx.camdl(&["fit", "predict", "pm.toml", "--n-draws", "8"]));
        files_under(&folder.join("predict"))
    };
    let replay = |store: &str, extra: &[&str]| -> Vec<u8> {
        let mut args = vec!["simulate", "fit.camdl", "--fit", "pm.toml", "--draws", "posterior", "-n", "3",
                            "--seed", "1", "--output-dir", store];
        args.extend(extra);
        ok(&fx.camdl(&args));
        files_under(&fx.dir.join(store)).into_iter().filter(|(k, _)| k.ends_with("traj.tsv")).flat_map(|(_, v)| v).collect()
    };
    let predicted = predict();
    let replayed = replay("sim1", &[]);
    assert!(!predicted.is_empty() && !replayed.is_empty());

    // Edit the source file the fit was bound to: the fit's replays do not move.
    fx.write("school.tsv", "t\tin_term\n0\t0.2\n20\t0.2\n");
    assert_eq!(predicted, predict(), "fit predict must replay the table bytes the fit stored");
    assert_eq!(replayed, replay("sim2", &[]), "simulate --fit must replay the table bytes the fit stored");

    // `--table` replaces the fit's file on a replay, as on every command.
    assert_ne!(replayed, replay("sim3", &["--table", "school=cal/b.tsv"]), "--table must win over the fit's table");
}

/// Every file under `dir`, by path relative to it, with its bytes.
fn files_under(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
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

#[test]
fn fit_refuses_an_unbound_undeclared_or_mismatched_table() {
    let fx = Fixture::with_fit_files();
    fx.write("cal/bad.tsv", "t\topen\n0\t1\n20\t1\n");
    fx.write("unbound.toml", &fit_config("", PFILTER_METHOD));
    fx.write("undeclared.toml", &fit_config("school = \"cal/a.tsv\"\nschol = \"cal/b.tsv\"", PFILTER_METHOD));
    fx.write("mismatch.toml", &fit_config("school = \"cal/bad.tsv\"", PFILTER_METHOD));

    let err = refused(&fx.camdl(&["fit", "run", "unbound.toml"]));
    assert!(err.contains("external input 'school' has no file") && err.contains("forcing 'school'"), "{err}");

    let err = refused(&fx.camdl(&["fit", "run", "undeclared.toml"]));
    assert!(err.contains("external input 'schol' is bound ([data.tables] schol = "), "{err}");
    assert!(err.contains("does not declare it"), "{err}");

    let err = refused(&fx.camdl(&["fit", "run", "mismatch.toml"]));
    assert!(err.contains("external input 'school' ([data.tables] school = ") && err.contains("forcing 'school'"), "{err}");
    assert!(err.contains("no column 'in_term'") && err.contains("t, open"), "{err}");
}

#[test]
fn pfilter_and_profile_bind_the_config_table_under_table_flag() {
    let fx = Fixture::with_fit_files();
    fx.write("a.toml", &fit_config("school = \"cal/a.tsv\"", PFILTER_METHOD));
    fx.write("b.toml", &fit_config("school = \"cal/b.tsv\"", PFILTER_METHOD));
    let pf = |fit: &str, extra: &[&str]| {
        let mut args = vec!["pfilter", "fit.camdl", "--fit", fit, "--param", "beta=0.4", "--param", "gamma=0.2",
                            "--param", "rho=0.5", "--particles", "50", "--seed", "1"];
        args.extend(extra);
        let out = fx.camdl(&args);
        ok(&out);
        printed_loglik(&out)
    };
    let a = pf("a.toml", &[]);
    let b = pf("b.toml", &[]);
    let a_with_b = pf("a.toml", &["--table", "school=cal/b.tsv"]);
    assert_ne!(a, b, "pfilter --fit binds the config's calendar");
    assert_eq!(a_with_b, b, "--table replaces the config's calendar");

    let profile = |extra: &[&str]| -> String {
        let mut args = vec!["profile", "fit.camdl", "--fit", "a.toml", "--sweep", "beta=0.4", "--particles", "20",
                            "--iterations", "2", "--n-starts", "1", "--seed", "1"];
        args.extend(extra);
        let out = fx.camdl(&args);
        ok(&out);
        String::from_utf8_lossy(&out.stdout).lines().last().unwrap_or_default().to_string()
    };
    assert_ne!(profile(&[]), profile(&["--table", "school=cal/b.tsv"]), "profile must bind --table, not drop it");
}

/// A stored fit read back by a reader of its config binds the table bytes
/// the fit ran under. `fit.resolved.toml` keeps `[data.tables]`, and a reader
/// of a run handle (`pfilter --fit <leaf>`) binds the copy stored in the fit
/// folder, so editing the source file after the fit does not move the
/// result; a stored copy whose bytes are not the ones the fit recorded is
/// refused rather than read.
#[test]
fn a_stored_fits_reader_binds_the_tables_it_ran_under() {
    let fx = Fixture::with_fit_files();
    fx.write("school.tsv", CALENDAR_A);
    fx.write("a.toml", &fit_config("school = \"school.tsv\"", PFILTER_METHOD));
    let leaf = stored_leaf(&ok(&fx.camdl(&["fit", "run", "a.toml"])));
    let folder = fx.dir.join(&leaf).ancestors().nth(2).unwrap().to_path_buf();
    let resolved = std::fs::read_to_string(folder.join("fit.resolved.toml")).unwrap();
    assert!(resolved.contains("[data.tables]\nschool = \"school.tsv\"\n"), "{resolved}");

    let pf = |fit: &str| -> Result<f64, String> {
        let out = fx.camdl(&["pfilter", "fit.camdl", "--fit", fit, "--param", "beta=0.4", "--param", "gamma=0.2",
                             "--param", "rho=0.5", "--particles", "50", "--seed", "1"]);
        if out.status.success() { Ok(printed_loglik(&out)) } else { Err(String::from_utf8_lossy(&out.stderr).into_owned()) }
    };
    let leaf_dir = fx.path(&leaf);
    let as_ran = pf(&leaf_dir).unwrap();
    assert_eq!(as_ran, pf("a.toml").unwrap(), "the leaf reads the table the file binds");

    // The source file changes: the file now names another problem, the
    // stored fit does not.
    fx.write("school.tsv", CALENDAR_B);
    assert_ne!(pf("a.toml").unwrap(), as_ran, "the table matters to the log-likelihood");
    assert_eq!(pf(&leaf_dir).unwrap(), as_ran, "a stored fit binds the bytes it ran under");

    // The stored copy itself changes: refused, naming the input and the
    // digest the fit recorded.
    std::fs::write(folder.join("inputs/school.tsv"), CALENDAR_B).unwrap();
    let err = pf(&leaf_dir).expect_err("a stored copy whose bytes are not the fit's");
    let recorded = sha256_hex(&fx.dir.join("cal/a.tsv"));
    assert!(err.contains("external input 'school'") && err.contains(&recorded), "{err}");
}
