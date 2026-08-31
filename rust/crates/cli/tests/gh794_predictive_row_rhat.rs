//! gh#794 — `predictive/<stream>.tsv` carries the convergence of the value in
//! each row, not only the producing stage's worst-parameter stamp.
//!
//! `fit_rhat_max` / `fit_ess_min` are the fit's worst parameter, copied from
//! the stage summary: one number repeated on every row of every stream. Under
//! their former names (`rhat_max` / `ess_min`) they read as the R̂ *of the
//! prediction beside them*, and a user acting on that draws the wrong
//! conclusion in either direction — a reportable quantity can be far better
//! determined than the parameters behind it, and a forecast can be
//! undetermined while the parameters look settled.
//!
//! What these tests pin, end to end through a real two-chain fit:
//!
//!   * `per_row_convergence_columns_are_written_and_vary_down_the_file` — the
//!     four new columns exist in the documented order, carry real numbers, and
//!     — the point of the change — differ from row to row, while
//!     `fit_rhat_max` stays constant. A per-row column that never moves has
//!     not been computed per row.
//!   * `one_step_rows_leave_both_per_row_pairs_empty` — the one-step horizon
//!     pools over particles as well as draws, so neither channel is emitted
//!     there. Both are empty, so a reader cannot pick up the diluted one by
//!     accident.
//!   * `predictive_manifest_lists_the_new_diagnostics` — the join contract
//!     beside the TSVs names them, so a consumer discovers them without
//!     reading the header.
//!   * `by_chain_adds_a_chain_column_and_leaves_the_pooled_rows_byte_identical`
//!     — `--by-chain` adds a leading `chain` column and one band per chain, and
//!     nothing else: strip the column and the per-chain rows and the file is
//!     byte-identical to the one written without the flag.
//!   * `by_chain_decomposes_the_one_step_horizon_too` — the in-sample half:
//!     does each chain explain the observed record, as against the
//!     free-forward rows' does each chain project the same future.
//!   * `by_chain_composes_with_exclude_chains_without_renumbering_or_a_second_
//!     address` — only the exclusion keys the artifact address (gh#795);
//!     `--by-chain` writes a superset of the pooled file and needs none. Chain
//!     ids are the fit's own, so a subset artifact's `2` names the same chain
//!     the pooled artifact's `2` does.
//!   * `reported_quantities_carry_their_own_rhat_and_ess` — the same reduction
//!     on `quantities/<name>.tsv`, where the case is stronger still: those are
//!     the numbers that get published.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build` or `make test`",
        bin.display()
    );
    bin
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(root.join("fits")).ok()?.flatten() {
        let p = e.path().join(sub).join(format!("{stream}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(root.join("fits")).ok()?.flatten() {
        let p = e.path().join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Header index of `name`, panicking with the whole header when absent.
fn col(header: &[&str], name: &str) -> usize {
    header
        .iter()
        .position(|h| *h == name)
        .unwrap_or_else(|| panic!("column `{name}` missing from header {header:?}"))
}

/// Every row's cells, split, for the rows whose `horizon` matches.
fn rows_of<'a>(tsv: &'a str, horizon: &str) -> (Vec<&'a str>, Vec<Vec<&'a str>>) {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let c_hor = col(&header, "horizon");
    let rows: Vec<Vec<&str>> = lines
        .map(|l| l.split('\t').collect::<Vec<&str>>())
        .filter(|c| c[c_hor] == horizon)
        .collect();
    (header, rows)
}

/// A closed SIR observed weekly, fitted with two PGAS chains of 200 retained
/// sweeps. Two chains and 100 retained draws each (after the strided subsample
/// to the 200-draw cap) is well above the estimator's floor of 2 chains and 4
/// draws per chain, so every assessable row reports a number — and the fit is
/// long enough for its own parameter R̂ to be reportable, which is what makes
/// the stamp-versus-row contrast below a real comparison.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  prevalence  = I / N
  peak_burden = max(I)
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 300
burn_in = 100
thin = 1
"#;

fn setup(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = std::env::temp_dir()
        .join(format!("camdl_gh794_{}_{}_{}", tag, std::process::id(), ns));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();
    tmp
}

fn fit_then_predict(bin: &Path, tmp: &Path) {
    fit_then_predict_with(bin, tmp, &[])
}

fn fit_then_predict_with(bin: &Path, tmp: &Path, extra: &[&str]) {
    let out = run(bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    predict_with(bin, tmp, extra);
}

fn predict_with(bin: &Path, tmp: &Path, extra: &[&str]) {
    let mut args = vec!["fit", "predict", "--fit", "fit.toml"];
    args.extend_from_slice(extra);
    let out = run(bin, tmp, &args);
    assert!(
        out.status.success(),
        "fit predict {extra:?} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn per_row_convergence_columns_are_written_and_vary_down_the_file() {
    let bin = skip_if_missing_binary();
    let tmp = setup("rows");
    fit_then_predict(&bin, &tmp);

    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let txt = std::fs::read_to_string(&pred).unwrap();
    let (header, rows) = rows_of(&txt, "free_forward");
    assert!(!rows.is_empty(), "free-forward rows must exist:\n{txt}");

    // The documented layout: the stage stamp, then the two per-row pairs, then
    // n_draws. A consumer reading positionally must not find them interleaved.
    let (c_rmax, c_emin) = (col(&header, "fit_rhat_max"), col(&header, "fit_ess_min"));
    let (c_rmean, c_emean) = (col(&header, "rhat_mean"), col(&header, "ess_mean"));
    let (c_rpred, c_epred) = (col(&header, "rhat_pred"), col(&header, "ess_pred"));
    let c_n = col(&header, "n_draws");
    assert_eq!(c_emin, c_rmax + 1);
    assert_eq!(c_rmean, c_emin + 1);
    assert_eq!(c_emean, c_rmean + 1);
    assert_eq!(c_rpred, c_emean + 1);
    assert_eq!(c_epred, c_rpred + 1);
    assert_eq!(c_n, c_epred + 1);

    // Every assessable row carries all four numbers. This fit has two chains of
    // thirty retained draws, so the only refusal available is a constant row.
    let assessed: Vec<&Vec<&str>> =
        rows.iter().filter(|c| !c[c_rmean].is_empty()).collect();
    assert!(
        assessed.len() * 2 >= rows.len(),
        "most free-forward rows must report a per-row R̂; {} of {} did:\n{txt}",
        assessed.len(),
        rows.len()
    );
    for c in &assessed {
        for (name, i) in [("ess_mean", c_emean), ("rhat_pred", c_rpred), ("ess_pred", c_epred)] {
            let v: f64 = c[i]
                .parse()
                .unwrap_or_else(|_| panic!("`{name}` cell `{}` must parse as a number", c[i]));
            assert!(v.is_finite() && v > 0.0, "`{name}` must be a positive finite number, got {v}");
        }
    }

    // The defect being fixed: the stage stamp is one number repeated on every
    // row, while the per-row channel actually moves with the row.
    let stamps: std::collections::BTreeSet<&str> =
        rows.iter().map(|c| c[c_rmax]).collect();
    assert_eq!(
        stamps.len(),
        1,
        "fit_rhat_max is the producing stage's worst parameter — provenance, constant \
         down the file. Got {stamps:?}"
    );
    let stamp: f64 = stamps.iter().next().unwrap().parse().expect(
        "this fit's worst parameter has an assessable R̂, so the stamp is a number",
    );
    let per_row: std::collections::BTreeSet<&str> =
        assessed.iter().map(|c| c[c_rmean]).collect();
    assert!(
        per_row.len() > 1,
        "rhat_mean must be computed per row: it took only the value(s) {per_row:?} \
         across {} assessed rows, which is what a provenance stamp looks like\n{txt}",
        assessed.len()
    );
    // And it is a materially different number, not a rounding of the stamp. On
    // this fit the worst parameter sits near 2.7 while the predicted curve is
    // near 1.05 — the reportable quantity is far better determined than the
    // parameters behind it, which is the case the issue opens with.
    for c in &assessed {
        let per_row: f64 = c[c_rmean].parse().unwrap();
        assert!(
            (per_row - stamp).abs() > 0.1,
            "rhat_mean {per_row} must describe the row, not repeat the stage stamp {stamp}"
        );
    }

    // The two per-row channels are distinct reductions, not one number written
    // twice, and they come apart in the documented direction: the predictive
    // draw carries observation noise, which inflates the within-chain variance,
    // so `ess_pred` reads as MORE information than `ess_mean` on most rows.
    assert!(
        assessed.iter().any(|c| c[c_rmean] != c[c_rpred]),
        "rhat_mean and rhat_pred reduce different operands (the latent mean and \
         the predictive draw), so they cannot be identical on every row\n{txt}"
    );
    let diluted = assessed
        .iter()
        .filter(|c| {
            let (m, p): (f64, f64) = (c[c_emean].parse().unwrap(), c[c_epred].parse().unwrap());
            p >= m
        })
        .count();
    assert!(
        diluted * 2 > assessed.len(),
        "adding observation noise to a series inflates its apparent effective \
         sample size, so ess_pred >= ess_mean on most rows; it held on {diluted} \
         of {}\n{txt}",
        assessed.len()
    );
}

#[test]
fn one_step_rows_leave_both_per_row_pairs_empty() {
    let bin = skip_if_missing_binary();
    let tmp = setup("onestep");
    fit_then_predict(&bin, &tmp);

    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let txt = std::fs::read_to_string(&pred).unwrap();
    let (header, rows) = rows_of(&txt, "one_step");
    assert!(!rows.is_empty(), "a chain-binomial fit emits one_step rows too:\n{txt}");
    for name in ["rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
        let i = col(&header, name);
        assert!(
            rows.iter().all(|c| c[i].is_empty()),
            "the one-step cell pools over particles as well as draws, so `{name}` \
             is withheld there — both channels empty, never just the diluted one"
        );
    }
    // The stage stamp still rides on the one-step rows: it is provenance, so it
    // is horizon-independent and carries the SAME value the free-forward rows do.
    let i = col(&header, "fit_rhat_max");
    let (_, ff) = rows_of(&txt, "free_forward");
    let ff_stamp = ff[0][i];
    assert!(
        rows.iter().all(|c| c[i] == ff_stamp),
        "the stage stamp is the same on every horizon (got {:?} against the \
         free-forward {ff_stamp:?})",
        rows.iter().map(|c| c[i]).collect::<Vec<_>>()
    );
}

#[test]
fn predictive_manifest_lists_the_new_diagnostics() {
    let bin = skip_if_missing_binary();
    let tmp = setup("manifest");
    fit_then_predict(&bin, &tmp);

    let path = find_file(&tmp.join("results"), "predictive.json")
        .expect("predictive.json must be written");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let diags = v["streams"][0]["diagnostics"]
        .as_array()
        .expect("a stream entry lists its diagnostics")
        .iter()
        .map(|d| d.as_str().unwrap().to_string())
        .collect::<Vec<String>>();
    for name in ["fit_rhat_max", "fit_ess_min", "rhat_mean", "ess_mean", "rhat_pred", "ess_pred", "n_draws"]
    {
        assert!(diags.contains(&name.to_string()), "manifest must list `{name}`: {diags:?}");
    }
}

/// `--by-chain` adds a leading `chain` column with `all` on the pooled rows and
/// one band per chain beside them — and, the property every existing consumer
/// depends on, changes nothing else: strip the `chain` column and the per-chain
/// rows and the file is byte-identical to the one written without the flag.
#[test]
fn by_chain_adds_a_chain_column_and_leaves_the_pooled_rows_byte_identical() {
    let bin = skip_if_missing_binary();
    let tmp = setup("bychain");

    // The pooled file first, then the same fit re-predicted with the flag.
    fit_then_predict(&bin, &tmp);
    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pooled_only = std::fs::read_to_string(&pred).unwrap();
    assert!(
        !pooled_only.lines().next().unwrap().starts_with("chain\t"),
        "no --by-chain ⇒ no `chain` column"
    );

    predict_with(&bin, &tmp, &["--by-chain"]);
    let with_chain = std::fs::read_to_string(&pred).unwrap();
    let header: Vec<&str> = with_chain.lines().next().unwrap().split('\t').collect();
    assert_eq!(header[0], "chain", "the chain column leads: {header:?}");

    // Both chains of the fit appear, 1-based, beside the pooled `all` rows.
    let chains: std::collections::BTreeSet<&str> = with_chain
        .lines()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        chains,
        ["1", "2", "all"].into_iter().collect::<std::collections::BTreeSet<&str>>(),
        "the pooled band plus one band per chain, 1-based"
    );

    // The pooled rows survive untouched: drop the per-chain rows and the leading
    // cell and the two files agree byte for byte.
    let stripped: String = with_chain
        .lines()
        .filter(|l| l.starts_with("all\t") || l.starts_with("chain\t"))
        .map(|l| format!("{}\n", l.split_once('\t').unwrap().1))
        .collect();
    assert_eq!(
        stripped, pooled_only,
        "--by-chain must add rows and one column, never alter the pooled band"
    );

    // A per-chain row is a band over one chain, so it carries no between-chain
    // statistic — and its n_draws is its own chain's, not the pooled count.
    let ix = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (c_n, c_hor) = (ix("n_draws"), ix("horizon"));
    let mut pooled_n = String::new();
    let mut per_chain_n: Vec<String> = Vec::new();
    for l in with_chain.lines().skip(1) {
        let c: Vec<&str> = l.split('\t').collect();
        if c[c_hor] != "free_forward" {
            continue;
        }
        if c[0] == "all" {
            pooled_n = c[c_n].to_string();
            continue;
        }
        for name in ["rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
            assert_eq!(
                c[ix(name)], "",
                "a single-chain band has no `{name}`: R-hat compares chains"
            );
        }
        per_chain_n.push(c[c_n].to_string());
    }
    assert!(!per_chain_n.is_empty(), "free-forward per-chain rows exist:\n{with_chain}");
    let pooled: usize = pooled_n.parse().unwrap();
    for n in &per_chain_n {
        let n: usize = n.parse().unwrap();
        assert!(
            n < pooled && n > 0,
            "a chain's band is over its own draws ({n}), not the pooled {pooled}"
        );
    }

    // Both horizons are decomposed: the one-step cell pools over particles as
    // well as draws, but that affects a per-chain band's WIDTH exactly as it
    // already affects the pooled band's, and grouping is not a variance
    // decomposition. (What the particle pooling does rule out is R-hat/ESS on
    // those rows — gh#798, asserted just below.)
    let one_step_chains: std::collections::BTreeSet<&str> = with_chain
        .lines()
        .skip(1)
        .map(|l| l.split('\t').collect::<Vec<&str>>())
        .filter(|c| c[c_hor] == "one_step")
        .map(|c| c[0])
        .collect();
    assert_eq!(
        one_step_chains,
        ["1", "2", "all"].into_iter().collect::<std::collections::BTreeSet<&str>>(),
        "one_step carries a pooled band plus one per chain"
    );

    // The manifest declares the new coordinate, so a consumer discovers it
    // without parsing the header.
    let path = find_file(&tmp.join("results"), "predictive.json").unwrap();
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let coords: Vec<String> = v["streams"][0]["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_string())
        .collect();
    assert_eq!(coords.first().map(String::as_str), Some("chain"), "{coords:?}");
}

/// `--by-chain` and `--exclude-chains` compose, and only one of them keys the
/// artifact address.
///
/// `--exclude-chains` does key it (gh#795): a chain subset is a different
/// posterior, and writing it at the pooled address replaced the run's canonical
/// predictive with a cherry-picked one. `--by-chain` does not: its file is a
/// strict superset of the pooled one — the `all` rows are byte-identical — so
/// re-running with the flag adds rows and a column rather than replacing an
/// artifact with a different object, exactly as `--scenario` and `--sweep` do.
///
/// The sharp part is the numbering. `ChainSelection::apply_keyed` filters draws
/// and leaves each row's `chain` value alone, so the ids in a subset artifact
/// name the same chains as in the pooled one. Excluding chain 1 of a two-chain
/// fit must therefore leave a `chain` column reading `2`, not a renumbered `1`
/// — a renumbering would make the two artifacts silently incomparable.
#[test]
fn by_chain_composes_with_exclude_chains_without_renumbering_or_a_second_address() {
    let bin = skip_if_missing_binary();
    let tmp = setup("compose");

    // Pooled first, so the canonical artifact exists to compare against.
    fit_then_predict(&bin, &tmp);
    let pooled = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("the pooled predictive is written first");
    let pooled_before = std::fs::read_to_string(&pooled).unwrap();

    // Drop chain 1 and ask for the per-chain decomposition at the same time.
    predict_with(&bin, &tmp, &["--by-chain", "--exclude-chains", "1"]);

    // The exclusion keys the address; --by-chain adds none of its own.
    let excl = find_artifact(&tmp.join("results"), "predictive-excl1", "weekly_cases")
        .expect("--exclude-chains 1 writes predictive-excl1/, not predictive/");
    let txt = std::fs::read_to_string(&excl).unwrap();
    assert_eq!(
        std::fs::read_to_string(&pooled).unwrap(),
        pooled_before,
        "the pooled artifact is untouched by a chain-subset run (gh#795)"
    );

    let header: Vec<&str> = txt.lines().next().unwrap().split('\t').collect();
    assert_eq!(header[0], "chain", "--by-chain still writes its column: {header:?}");
    let chains: std::collections::BTreeSet<&str> =
        txt.lines().skip(1).map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(
        chains,
        ["2", "all"].into_iter().collect::<std::collections::BTreeSet<&str>>(),
        "the retained chain keeps its own id 2 — an exclusion filters draws, it \
         does not renumber, so these rows name the same chain the pooled \
         artifact's `2` rows do"
    );

    // One retained chain, so the between-chain statistics are refused rather
    // than computed over a single chain. The consequence of excluding down to
    // one chain, made visible rather than papered over with a number.
    let ix = |name: &str| col(&header, name);
    let c_hor = ix("horizon");
    let mut saw_free_forward = false;
    for l in txt.lines().skip(1) {
        let c: Vec<&str> = l.split('\t').collect();
        if c[c_hor] != "free_forward" {
            continue;
        }
        saw_free_forward = true;
        for name in ["rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
            assert_eq!(
                c[ix(name)], "",
                "one retained chain leaves `{name}` empty on every row, pooled \
                 and per-chain alike: R-hat compares chains"
            );
        }
    }
    assert!(saw_free_forward, "free-forward rows exist in the subset artifact:\n{txt}");

    // The manifest travels to the keyed name and still declares `chain`.
    let mf = find_file(&tmp.join("results"), "predictive-excl1.json")
        .expect("the keyed manifest is written beside the keyed directory");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&mf).unwrap()).unwrap();
    let coords: Vec<String> = v["streams"][0]["coordinates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_string())
        .collect();
    assert_eq!(coords.first().map(String::as_str), Some("chain"), "{coords:?}");
    assert_eq!(
        v["streams"][0]["file"], "predictive-excl1/weekly_cases.tsv",
        "the manifest points at the keyed location"
    );
}

/// gh#794 on the reported estimands: `quantities/<name>.tsv` carries the R-hat
/// and bulk-ESS of the value in each row — the same chain-grouped reduction the
/// predictive rows get, on the numbers that actually get published.
#[test]
fn reported_quantities_carry_their_own_rhat_and_ess() {
    let bin = skip_if_missing_binary();
    let tmp = setup("quantities");
    fit_then_predict(&bin, &tmp);

    // A series quantity: one row per snapshot, each with its own reduction.
    let series = find_artifact(&tmp.join("results"), "quantities", "prevalence")
        .expect("quantities/prevalence.tsv must be written");
    let txt = std::fs::read_to_string(&series).unwrap();
    let header: Vec<&str> = txt.lines().next().unwrap().split('\t').collect();
    let ix = |name: &str| col(&header, name);
    assert_eq!(
        header[ix("n_draws") + 1],
        "rhat",
        "rhat follows n_draws: {header:?}"
    );
    assert_eq!(header[ix("n_draws") + 2], "ess");

    let mut rhats: Vec<f64> = Vec::new();
    for l in txt.lines().skip(1) {
        let c: Vec<&str> = l.split('\t').collect();
        let (r, e) = (c[ix("rhat")], c[ix("ess")]);
        if r.is_empty() {
            continue; // a constant row is refused, not fabricated
        }
        let r: f64 = r.parse().expect("rhat parses");
        let e: f64 = e.parse().expect("ess parses");
        assert!(r.is_finite() && r > 0.0, "rhat is a positive finite number, got {r}");
        assert!(e.is_finite() && e > 0.0, "ess is a positive finite number, got {e}");
        rhats.push(r);
    }
    assert!(!rhats.is_empty(), "a two-chain fit reports rhat on its series rows:\n{txt}");
    let distinct: std::collections::BTreeSet<String> =
        rhats.iter().map(|r| format!("{r:.4}")).collect();
    assert!(
        distinct.len() > 1,
        "rhat is computed per row, so it moves down the file; got only {distinct:?}"
    );

    // A scalar quantity: one row, still carrying its own reduction.
    let scalar = find_artifact(&tmp.join("results"), "quantities", "peak_burden")
        .expect("quantities/peak_burden.tsv must be written");
    let txt = std::fs::read_to_string(&scalar).unwrap();
    let header: Vec<&str> = txt.lines().next().unwrap().split('\t').collect();
    let cells: Vec<&str> = txt.lines().nth(1).unwrap().split('\t').collect();
    let r = cells[col(&header, "rhat")];
    assert!(
        !r.is_empty() && r.parse::<f64>().is_ok(),
        "a scalar estimand reports its own rhat too, got {r:?}\n{txt}"
    );
}

/// `--by-chain` decomposes the **one-step** horizon as well as the free-forward
/// one, and that is the half the diagnostic was asked for by name.
///
/// The two horizons answer different questions and a reader has to know which
/// they are looking at. Free-forward per-chain bands say whether the chains
/// *project* the same future — disagreement there mixes mixing pathology with
/// extrapolation uncertainty, because the bands are running free past the data.
/// One-step per-chain bands say whether each chain *explains the observed
/// record*: they are re-anchored to the data at every step, so a separation
/// between them is disagreement about the fitted trajectory itself, with the
/// extrapolation removed. That is the sharper statement about mixing.
///
/// Grouping the pool by chain is not a variance decomposition: each chain's
/// band pools its own draws x particles exactly as the pooled band pools all of
/// them. The particle pooling that rules out a bulk-ESS on these rows (gh#798)
/// does not rule out banding them.
#[test]
fn by_chain_decomposes_the_one_step_horizon_too() {
    let bin = skip_if_missing_binary();
    let tmp = setup("onestep_bychain");

    fit_then_predict(&bin, &tmp);
    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pooled_only = std::fs::read_to_string(&pred).unwrap();

    predict_with(&bin, &tmp, &["--by-chain"]);
    let with_chain = std::fs::read_to_string(&pred).unwrap();
    let header: Vec<&str> = with_chain.lines().next().unwrap().split('\t').collect();
    let ix = |name: &str| col(&header, name);
    let (c_hor, c_n, c_time) = (ix("horizon"), ix("n_draws"), ix("time"));

    let one_step: Vec<Vec<&str>> = with_chain
        .lines()
        .skip(1)
        .map(|l| l.split('\t').collect::<Vec<&str>>())
        .filter(|c| c[c_hor] == "one_step")
        .collect();
    assert!(!one_step.is_empty(), "a chain-binomial fit emits one_step rows:\n{with_chain}");

    // One band per chain beside the pooled one, 1-based.
    let chains: std::collections::BTreeSet<&str> = one_step.iter().map(|c| c[0]).collect();
    assert_eq!(
        chains,
        ["1", "2", "all"].into_iter().collect::<std::collections::BTreeSet<&str>>(),
        "the one-step horizon is decomposed by chain, not left pooled-only"
    );

    // Each chain covers the same observation axis the pooled rows do — a
    // per-chain band that silently dropped time points would look like a
    // shorter series rather than a missing one.
    let times_of = |chain: &str| -> Vec<&str> {
        one_step.iter().filter(|c| c[0] == chain).map(|c| c[c_time]).collect()
    };
    let pooled_times = times_of("all");
    assert!(!pooled_times.is_empty());
    for chain in ["1", "2"] {
        assert_eq!(times_of(chain), pooled_times, "chain {chain} covers the pooled time axis");
    }

    // No between-chain statistic on a single chain's rows — the same rule the
    // free-forward per-chain rows follow, and the pooled one-step rows keep
    // their empty cells for the gh#798 reason.
    for c in &one_step {
        for name in ["rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
            assert_eq!(
                c[ix(name)], "",
                "`{name}` is withheld on every one-step row (gh#798), pooled and \
                 per-chain alike"
            );
        }
    }

    // A chain's band is over its *own* draws, not a relabelled copy of the pooled
    // one. Without this the grouping could be a no-op that renames rows.
    let q50 = ix("q50");
    let band_of = |chain: &str| -> Vec<&str> {
        one_step.iter().filter(|c| c[0] == chain).map(|c| c[q50]).collect()
    };
    let pooled_band = band_of("all");
    for chain in ["1", "2"] {
        assert_ne!(
            band_of(chain),
            pooled_band,
            "chain {chain}'s one-step band must be over its own draws — an \
             identical band means the pooled cell was banded and relabelled"
        );
    }

    // A chain's band is over its own draws: its n_draws is smaller than the
    // pooled count, and positive.
    let pooled_n: usize = one_step
        .iter()
        .find(|c| c[0] == "all")
        .map(|c| c[c_n].parse().unwrap())
        .unwrap();
    for chain in ["1", "2"] {
        let n: usize = one_step
            .iter()
            .find(|c| c[0] == chain)
            .map(|c| c[c_n].parse().unwrap())
            .unwrap();
        assert!(
            n > 0 && n < pooled_n,
            "chain {chain} bands over its own {n} draws, not the pooled {pooled_n}"
        );
    }

    // And the pooled one-step rows are byte-identical to the no-flag run's.
    // This is the property the whole change has to keep: the partition is a
    // regrouping of the same samples, and `band` sorts a copy, so the pooled
    // quantiles cannot move.
    let strip = |txt: &str, keep_all: bool| -> String {
        txt.lines()
            .filter(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                if keep_all {
                    c[0] == "chain" || (c[0] == "all" && c[c_hor] == "one_step")
                } else {
                    // The no-flag file has no `chain` column; its horizon
                    // column sits one to the left.
                    l.starts_with("scenario\t") || c[c_hor - 1] == "one_step"
                }
            })
            .map(|l| {
                if keep_all { l.split_once('\t').unwrap().1.to_string() } else { l.to_string() }
            })
            .map(|l| format!("{l}\n"))
            .collect()
    };
    assert_eq!(
        strip(&with_chain, true),
        strip(&pooled_only, false),
        "the pooled one-step rows must survive --by-chain byte for byte"
    );
}
