//! gh#728: `*_summary.json` says which contract its numbers were written
//! under, and a reader that does not recognise the tag refuses the file.
//!
//! Two keys of this artifact changed *meaning* without changing name.
//! `rhat` was the classic Gelman–Rubin statistic and is now
//! `max(rank-normalized split-R̂, folded split-R̂)`; `ess` was a sum of
//! per-chain Geyer estimates, suppressed to NaN above R̂ 1.1, and is now the
//! cross-chain bulk-ESS, never suppressed. Both vintages sit in one store for
//! as long as a project keeps its fits — for a paper, indefinitely — and
//! nothing in an untagged file separates them: a summary with no nulls is
//! either a new fit or an old fit whose parameters all converged. A viewer
//! that labels a column "R̂" is therefore labelling two different estimators
//! and cannot know which.
//!
//! The pin is the whole contract, because half of it is worthless: the tag
//! must be *written* (or there is nothing to key on), and it must be
//! *required* (or a stale file is read under today's meaning anyway). The
//! third assertion is that the refusal names the file and the tag, since a
//! reader who hits it is holding a fit they may not be able to re-run
//! cheaply and needs to know what it is.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

const DATA: &str = "time\tcases\n1\t20\n2\t35\n3\t60\n4\t90\n5\t120\n6\t150\n";

const FIT_TOML: &str = r#"output_dir = "results"
[model]
camdl = "sir.camdl"
[data.observations]
cases = "cases.tsv"
[config]
dt = 1.0
[estimate]
beta  = { bounds = [0.01, 5.0], prior = { log_normal = { mu = -0.3, sigma = 0.5 } }, start = 0.3 }
gamma = { bounds = [0.01, 1.0], prior = { log_normal = { mu = -1.2, sigma = 0.5 } }, start = 0.1 }
[fixed]
N0 = 1000
[method]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 2
particles = 8
sweeps    = 8
burn_in   = 2
thin      = 1
starts    = "uniform_unconstrained"
"#;

/// The directory holding `pgas_summary.json` under `root`.
fn leaf_with_summary(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("pgas_summary.json").is_file() {
            return d;
        }
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no pgas_summary.json under {}", root.display());
}

#[test]
fn a_fit_summary_declares_its_schema_and_a_reader_requires_it() {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    let tmp = std::env::temp_dir().join(format!("camdl_gh728_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let run = |args: &[&str]| -> std::process::Output {
        Command::new(&bin)
            .args(args)
            .current_dir(&tmp)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    };

    let out = run(&["fit", "run", "fit.toml", "--seed", "1", "--no-progress"]);
    assert!(
        out.status.success(),
        "fit run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ── Written: the tag leads the file, as in every other camdl manifest ──
    let leaf = leaf_with_summary(&tmp.join("results"));
    let path = leaf.join("pgas_summary.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let summary: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        summary["schema"], "camdl.fit-summary/v2",
        "the summary must declare which contract its `rhat` and `ess` keys \
         were written under:\n{text}"
    );
    // The premise: the keys the tag is about are present, so the tag is
    // describing something rather than decorating an empty file.
    assert!(summary.get("rhat").is_some(), "{text}");
    assert!(summary.get("ess").is_some(), "{text}");

    // gh#901, the rename the v2 tag is about: the algorithm is named under
    // `method`, the word the store levels, the fit config and the CLI use.
    // Two-sided, so the rename cannot be half-reverted — the old key must be
    // gone from the file, not merely joined by a new one.
    assert_eq!(
        summary["method"], "pgas",
        "the summary must name its algorithm under `method` (gh#901):\n{text}"
    );
    assert!(
        summary.get("stage").is_none(),
        "and must not carry the old `stage` key alongside it — a consumer \
         that kept reading `stage` would never learn the vocabulary \
         changed:\n{text}"
    );

    let segment = leaf.parent().unwrap().parent().unwrap();
    let seg = segment.to_string_lossy().into_owned();
    let out = run(&["fit", "summary", &seg]);
    let tagged_stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "a tagged summary reads normally:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        tagged_stdout.contains("beta"),
        "premise: the tagged read reports the parameters, so the absence \
         asserted below is the refusal and not an empty fit:\n{tagged_stdout}"
    );

    // ── Required: strip the tag, and the reader refuses rather than reading
    //    the numbers under today's meaning ──
    let mut stripped = summary.clone();
    stripped.as_object_mut().unwrap().remove("schema");
    std::fs::write(&path, serde_json::to_string_pretty(&stripped).unwrap()).unwrap();

    let out = run(&["fit", "summary", &seg]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        !stdout.contains("beta"),
        "an untagged summary's numbers must not be reported as though they \
         were written under today's definitions:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stderr.contains("pgas_summary.json") && stderr.contains("schema"),
        "and the reason must name the file and what it is missing, because a \
         reader who hits this is holding a fit they may not be able to re-run \
         cheaply:\n{stderr}"
    );
    // The refusal explains what the untagged numbers *are*, not only that the
    // tag is absent — that is the difference between a message a reader can
    // act on and one that sends them to the source.
    assert!(
        stderr.contains("Gelman"),
        "the refusal says which estimators the old file holds:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
