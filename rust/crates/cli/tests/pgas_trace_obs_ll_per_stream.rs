//! gh#742: the PGAS trace carries `obs_ll` resolved by observation stream.
//!
//! `chain_<n>/trace.tsv` gains one `obs_ll_<stream>` column per declared stream
//! alongside the existing `obs_ll`, so a multi-stream fit can be asked which
//! stream it is straining against without re-running the filter — which for a
//! national model is minutes per chain and gives a different realisation unless
//! the seed is pinned.
//!
//! What this pins that the in-crate `sim` test cannot: the columns are LABELLED,
//! ALIGNED and POPULATED on disk. The `sim` test
//! (`pgas_obs_ll_per_stream.rs`) checks the decomposition at full `f64`
//! precision in memory; here the values have been through the trace writer's
//! 4-decimal formatting, so agreement is asserted to that precision. A trace
//! whose header and rows disagree in width, or whose values are written in a
//! different order from the header, mislabels every diagnostic on the row —
//! silently, since every column is numeric.
//!
//! The multi-stream fixture is deliberately MULTI-CADENCE: `cases` weekly
//! against `confirmations` daily. Every sweep evaluates the whole likelihood, so
//! both columns must carry a number on every row — the weekly stream does not go
//! blank on the days it has no observation, because the row is a SWEEP, not an
//! observation time.
//!
//! Skipped when camdlc isn't present; the release binary is required (gh#105).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_pgas_obsll_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Daily prevalence counts over a 21-day SIR epidemic, drawn from the model
/// itself so the starting parameters can explain them (a zero-density start
/// would have the sampler fighting its way back from `-inf` instead of
/// exercising the ordinary path).
const CONFIRMATIONS: &str = "time\tconfirmations\n\
    1\t40\n2\t58\n3\t75\n4\t98\n5\t154\n6\t179\n7\t235\n8\t292\n9\t298\n10\t305\n\
    11\t298\n12\t270\n13\t232\n14\t196\n15\t160\n16\t149\n17\t139\n18\t117\n\
    19\t83\n20\t68\n21\t57\n";

/// The weekly incidence of the same epidemic — three observations against the
/// daily stream's twenty-one.
const CASES: &str = "time\tcases\n7\t381\n14\t449\n21\t88\n";

/// Compile `src` and write the data files named in `data`. Returns the IR path.
fn write_fixture(dir: &Path, src: &str, data: &[(&str, &str)]) -> PathBuf {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    for (name, text) in data {
        std::fs::write(dir.join(format!("{name}.tsv")), text).unwrap();
    }
    ir_path
}

/// SIR with a weekly incidence stream (`cases`) and a daily prevalence stream
/// (`confirmations`) — genuinely different cadences over one latent path.
const TWO_STREAM_MODEL: &str = r#"
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
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = projected)
  }
  confirmations {
    columns       { time : time, confirmations : count }
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    confirmations ~ poisson(rate = projected)
  }
}
init { S = 980  I = 20 }
simulate { from = 0 'days  to = 21 'days }
"#;

/// The same model with only the daily prevalence stream declared.
const ONE_STREAM_MODEL: &str = r#"
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
  confirmations {
    columns       { time : time, confirmations : count }
    projected     = prevalence(I)
    emit_schedule = every 1 'days
    confirmations ~ poisson(rate = projected)
  }
}
init { S = 980  I = 20 }
simulate { from = 0 'days  to = 21 'days }
"#;

fn write_fit_toml(dir: &Path, ir: &Path, streams: &[&str]) -> PathBuf {
    let obs: String = streams.iter()
        .map(|s| format!("{s} = \"{}\"\n", dir.join(format!("{s}.tsv")).display()))
        .collect();
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
{obs}
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 30
sweeps = 12
burn_in = 2
"#,
        out = dir.join("results").display(),
        ir  = ir.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// The `chain_1/trace.tsv` of the `post` stage leaf under `<out>/fits/`.
fn trace_of_post_leaf(out: &Path) -> String {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return std::fs::read_to_string(d.join("chain_1/trace.tsv"))
                        .expect("read chain_1/trace.tsv");
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

fn run_fit(bin: &Path, fit: &Path) {
    let r = Command::new(bin)
        .arg("fit").arg("run").arg(fit).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r.status.success(), "PGAS run failed: {}", String::from_utf8_lossy(&r.stderr));
}

/// Split a trace into (header, data rows), asserting every row is the header's
/// width — a ragged row mislabels every column after the gap.
fn parse(text: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = text.lines();
    let header = lines.next().expect("trace has a header line");
    let cols: Vec<String> = header.split('\t').map(str::to_string).collect();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<String> = l.split('\t').map(str::to_string).collect();
            assert_eq!(
                f.len(), cols.len(),
                "row has {} fields but the header names {} columns — every diagnostic \
                 after the mismatch is mislabelled; row was: {l}",
                f.len(), cols.len(),
            );
            f
        })
        .collect();
    assert!(!rows.is_empty(), "trace has no data rows");
    (cols, rows)
}

#[test]
fn multi_cadence_trace_decomposes_obs_ll_by_stream() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("two");
    let ir = write_fixture(tmp.path(), TWO_STREAM_MODEL,
        &[("cases", CASES), ("confirmations", CONFIRMATIONS)]);
    let fit = write_fit_toml(tmp.path(), &ir, &["cases", "confirmations"]);
    run_fit(&bin, &fit);

    let text = trace_of_post_leaf(&tmp.path().join("results"));
    let (cols, rows) = parse(&text);

    // `obs_ll` stays — `fit summary` ranks PGAS chains on it (gh#667) — and each
    // declared stream gains its own column beside it.
    for c in ["obs_ll", "obs_ll_cases", "obs_ll_confirmations"] {
        assert!(
            cols.iter().any(|h| h == c),
            "PGAS trace must carry the `{c}` column (gh#742); header was: {}",
            cols.join("\t"),
        );
    }
    // Exactly one column per DECLARED stream, not one per stream × observation
    // time and not one per union index.
    let per_stream: Vec<&String> = cols.iter()
        .filter(|c| c.starts_with("obs_ll_"))
        .collect();
    assert_eq!(
        per_stream.len(), 2,
        "two declared streams ⇒ two per-stream columns; got {per_stream:?}",
    );

    let idx = |name: &str| cols.iter().position(|c| c == name).unwrap();
    let (i_obs, i_cases, i_conf) =
        (idx("obs_ll"), idx("obs_ll_cases"), idx("obs_ll_confirmations"));

    // The trace writes 4 decimals, so each of the (2 + 1) values carries up to
    // 5e-5 of rounding; the sum can therefore sit that far from `obs_ll` without
    // anything being wrong. The full-precision identity is asserted in
    // `sim/tests/pgas_obs_ll_per_stream.rs`.
    let tol = 5e-5 * (per_stream.len() + 1) as f64;

    for row in &rows {
        let parse_at = |i: usize, name: &str| -> f64 {
            row[i].parse().unwrap_or_else(|e| panic!(
                "{name} = {:?} must be a number ({e}); row was: {}", row[i], row.join("\t")))
        };
        let obs = parse_at(i_obs, "obs_ll");
        let cases = parse_at(i_cases, "obs_ll_cases");
        let conf = parse_at(i_conf, "obs_ll_confirmations");

        // Every column is populated on every row: a sweep evaluates the WHOLE
        // likelihood, so the weekly stream contributes its own sum even though
        // most days carry no `cases` observation. A blank, an `NA`, or a zero
        // here would mean the weekly stream's terms were dropped.
        assert!(
            cases.is_finite() && cases != 0.0,
            "obs_ll_cases must carry the weekly stream's own sum on every sweep, \
             got {cases}; row was: {}", row.join("\t"),
        );
        assert!(
            conf.is_finite() && conf != 0.0,
            "obs_ll_confirmations must carry the daily stream's own sum on every \
             sweep, got {conf}; row was: {}", row.join("\t"),
        );
        assert!(
            (cases + conf - obs).abs() <= tol,
            "the per-stream columns must decompose obs_ll: {cases} + {conf} = {} \
             against obs_ll = {obs} (|Δ| = {:.3e}, tol {:.3e}); row was: {}",
            cases + conf, (cases + conf - obs).abs(), tol, row.join("\t"),
        );
        // The two streams score different data on different cadences, so equal
        // values across every row would mean one number was broadcast.
        assert_ne!(
            cases, conf,
            "the two streams must not report the same value; row was: {}",
            row.join("\t"),
        );
        // Which column is which. `confirmations` contributes 21 Poisson terms
        // and `cases` 3, on counts of comparable size, so the daily stream's sum
        // is necessarily the more negative of the two by a wide margin. This is
        // what fails if the values are written in a different order from the
        // header — a swap that the sum check above cannot see, since addition
        // commutes.
        assert!(
            conf < cases,
            "obs_ll_confirmations ({conf}) sums 21 observation terms against \
             obs_ll_cases' ({cases}) 3, so it must be the more negative; a \
             reversal here means the values are written in a different order \
             from the header. Row was: {}", row.join("\t"),
        );
    }
}

#[test]
fn single_stream_trace_carries_one_column_equal_to_obs_ll() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("one");
    let ir = write_fixture(tmp.path(), ONE_STREAM_MODEL,
        &[("confirmations", CONFIRMATIONS)]);
    let fit = write_fit_toml(tmp.path(), &ir, &["confirmations"]);
    run_fit(&bin, &fit);

    let text = trace_of_post_leaf(&tmp.path().join("results"));
    let (cols, rows) = parse(&text);

    let per_stream: Vec<&String> = cols.iter()
        .filter(|c| c.starts_with("obs_ll_"))
        .collect();
    assert_eq!(
        per_stream, vec!["obs_ll_confirmations"],
        "one declared stream ⇒ exactly one per-stream column, named for it; \
         header was: {}", cols.join("\t"),
    );

    let i_obs = cols.iter().position(|c| c == "obs_ll").unwrap();
    let i_one = cols.iter().position(|c| c == "obs_ll_confirmations").unwrap();
    for row in &rows {
        let obs: f64 = row[i_obs].parse().expect("obs_ll parses");
        let one: f64 = row[i_one].parse().expect("obs_ll_confirmations parses");
        assert_eq!(
            one, obs,
            "with one stream the single per-stream column IS obs_ll — same value, \
             same 4-decimal rendering; row was: {}", row.join("\t"),
        );
    }
}
