//! gh#795 — a chain-subset `fit predict` must not be written at the pooled
//! run's address.
//!
//! `--exclude-chains` reports a DIFFERENT posterior (its own warning says so:
//! "Post-hoc chain exclusion BIASES the posterior toward the retained mode").
//! Before this fix the excluded set reached `predictive.json` as a
//! `chain_selection` stamp but never reached the artifact's location, so
//! `fit predict --exclude-chains 3` replaced the run's canonical predictive
//! with a three-chain one, and a second exclusion replaced that. A reader —
//! `camdl-scope`, or a person — then rendered a cherry-picked subset as the
//! posterior predictive of the fit, with nothing in the run directory saying so.
//!
//! What is pinned here:
//!
//! - two different exclusions land at two addresses, and NEITHER is the pooled
//!   one — the pooled artifact survives both runs byte-for-byte;
//! - the address is keyed on the normalised SET, so `4,2` and `2,4` are one
//!   address and a repeat collapses;
//! - the three addresses hold genuinely different clouds (120 / 90 / 60 draws);
//! - with no `--exclude-chains`, the artifact names are exactly the historical
//!   ones — a `camdl-scope` reader of an unexcluded run is unaffected.
//!
//! Fixture: one tiny real PGAS fit for a valid self-contained segment, then its
//! `draws.tsv` is replaced by a controlled 4-chain cloud (three tight chains
//! plus one deliberate outlier), mirroring `exclude_chains_e2e.rs` — a KNOWN
//! chain structure a converging sampler would not reliably produce.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the latest `fit predict` of a run in `seg` wrote:
/// `<seg>/predict/<run id>[-<options key>]/` (gh#896), the directory whose
/// `report*.json` (one written by every predict) is newest. A path that does not
/// exist while nothing has been predicted.
fn predict_out(seg: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(seg.join("predict"))
        .into_iter().flatten().flatten().map(|e| e.path())
        .filter_map(|d| {
            // Newest `report*.json` (keyed by chain selection) in the directory.
            let t = std::fs::read_dir(&d).ok()?.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.starts_with("report") && n.ends_with(".json")
                })
                .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
                .max()?;
            Some((t, d))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, d)| d)
        .unwrap_or_else(|| seg.join("predict").join("none"))
}


fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn require_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation and a `quantities {}`
/// block, so the run writes BOTH a predictive and a quantities artifact — the
/// two families that collide.
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
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  peak = max(I / N)
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

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 40
burn_in = 10
thin = 1
"#;

/// Draws per chain in the controlled cloud.
const DRAWS_PER_CHAIN: usize = 30;

/// A controlled 4-chain cloud: chains 0,1,2 tight around a slow epidemic,
/// chain 3 a stuck outlier at a fast one. The
/// per-chain phase shift gives non-identical sequences with the same mean.
fn build_cloud() -> String {
    let mut s = String::from("chain\tdraw\tbeta\tgamma\tN0\tI0\trho\tk\n");
    let n = DRAWS_PER_CHAIN;
    for chain in 0..4 {
        for draw in 0..n {
            let i = (draw + chain) % n;
            let jitter = ((i % 11) as f64 - 5.0) * 0.002;
            let (beta, gamma) = if chain == 3 {
                (0.85 + jitter, 0.05)
            } else {
                (0.35 + jitter, 0.14 + jitter * 0.1)
            };
            s.push_str(&format!(
                "{chain}\t{draw}\t{beta:.4}\t{gamma:.4}\t10000\t10\t0.6\t10\n"
            ));
        }
    }
    s
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn segment_dir(results: &Path) -> PathBuf {
    std::fs::read_dir(results.join("fits"))
        .expect("results/fits exists")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("one fit segment")
}

fn find_draws_tsv(dir: &Path) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = find_draws_tsv(&p) {
                return Some(found);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some("draws.tsv") {
            return Some(p);
        }
    }
    None
}

struct Fixture {
    tmp: PathBuf,
    seg: PathBuf,
}

impl Fixture {
    /// Run `fit predict`, optionally under an exclusion. Free-forward only —
    /// the one-step horizon is not what this test is about and it is slower.
    fn predict(&self, bin: &Path, exclude: Option<&str>) -> std::process::Output {
        let seg = self.seg.to_string_lossy().into_owned();
        let mut args: Vec<&str> = vec!["fit", "predict", &seg, "--mode", "free_forward"];
        if let Some(ids) = exclude {
            args.push("--exclude-chains");
            args.push(ids);
        }
        run(bin, &self.tmp, &args)
    }

    /// The predictive/quantities artifact names directly under the run's
    /// predictions directory (`<segment>/predict/<run id>/`, gh#896),
    /// sorted. This IS the run directory as a reader (`ls`, `camdl-scope`) sees
    /// it, which is the surface the bug corrupts.
    fn artifact_names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(predict_out(&self.seg))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("predictive") || n.starts_with("quantities"))
            .collect();
        names.sort();
        names
    }

    fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(predict_out(&self.seg).join(rel))
            .unwrap_or_else(|e| panic!("reading {rel} under the segment: {e}"))
    }

    fn exists(&self, rel: &str) -> bool {
        predict_out(&self.seg).join(rel).exists()
    }
}

fn assert_ok(out: &std::process::Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Set up a fit segment whose `draws.tsv` is the controlled 4-chain cloud.
fn setup(bin: &Path, label: &str) -> Fixture {
    let tmp = std::env::temp_dir().join(format!(
        "camdl_gh795_{}_{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let out = run(bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert_ok(&out, "fit run");

    let seg = segment_dir(&tmp.join("results"));
    let draws = find_draws_tsv(&seg).expect("draws.tsv written by the fit");
    std::fs::write(&draws, build_cloud()).unwrap();
    Fixture { tmp, seg }
}

/// The distinct `n_draws` values of the free-forward rows — the cloud size the
/// bands were computed over, which is what distinguishes the three artifacts.
fn free_forward_n_draws(pred_tsv: &str) -> BTreeSet<usize> {
    let mut lines = pred_tsv.lines();
    let header: Vec<&str> = lines.next().expect("a header row").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("no `{name}` column in:\n{pred_tsv}"))
    };
    let (hi, ni) = (col("horizon"), col("n_draws"));
    let mut out = BTreeSet::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f.get(hi).copied() == Some("free_forward") {
            out.insert(f[ni].parse().expect("n_draws parses"));
        }
    }
    assert!(!out.is_empty(), "no free_forward rows in:\n{pred_tsv}");
    out
}

// ── The pooled address survives every exclusion ─────────────────────────────

#[test]
fn two_exclusions_are_two_addresses_and_neither_is_the_pooled_one() {
    let bin = require_binary();
    let f = setup(&bin, "addresses");

    // Each selection is its own `fit predict` address (`predict/<run>-<key>/`,
    // the key covering the chain selection), so each directory is read as the
    // one its predict wrote.
    let read = |dir: &Path, rel: &str| -> String {
        std::fs::read_to_string(dir.join(rel))
            .unwrap_or_else(|e| panic!("reading {rel} under {}: {e}", dir.display()))
    };

    // 1. The pooled run: the canonical artifact.
    assert_ok(&f.predict(&bin, None), "pooled predict");
    let pooled_dir = predict_out(&f.seg);
    let pooled_tsv = read(&pooled_dir, "predictive/weekly_cases.tsv");
    let pooled_json = read(&pooled_dir, "predictive.json");
    assert_eq!(
        free_forward_n_draws(&pooled_tsv),
        BTreeSet::from([120]),
        "the pooled cloud is all 4 chains × 30 draws"
    );

    // 2. Drop the outlier chain 3. This must NOT be the pooled address.
    assert_ok(&f.predict(&bin, Some("3")), "predict --exclude-chains 3");
    let excl3_dir = predict_out(&f.seg);
    assert_ne!(excl3_dir, pooled_dir, "a chain subset is its own address");
    assert_eq!(
        read(&pooled_dir, "predictive/weekly_cases.tsv"),
        pooled_tsv,
        "a chain-subset predict must not overwrite the pooled predictive TSV — \
         camdl-scope reads `predictive/` and would render a 3-chain subset as \
         the posterior predictive of the fit"
    );
    assert_eq!(read(&pooled_dir, "predictive.json"), pooled_json, "…nor the pooled manifest");
    assert!(
        !excl3_dir.join("predictive").exists(),
        "the subset's directory holds no unsuffixed predictive to mistake for the pooled one"
    );
    let excl3_tsv = read(&excl3_dir, "predictive-excl3/weekly_cases.tsv");
    assert_eq!(
        free_forward_n_draws(&excl3_tsv),
        BTreeSet::from([90]),
        "the subset cloud is 3 chains × 30 draws"
    );

    // 3. A second, different exclusion. All three artifacts coexist.
    assert_ok(&f.predict(&bin, Some("1,3")), "predict --exclude-chains 1,3");
    let excl13_dir = predict_out(&f.seg);
    assert!(excl13_dir != pooled_dir && excl13_dir != excl3_dir, "a third address");
    assert_eq!(
        read(&pooled_dir, "predictive/weekly_cases.tsv"),
        pooled_tsv,
        "the pooled predictive survives the second exclusion too"
    );
    assert_eq!(
        read(&excl3_dir, "predictive-excl3/weekly_cases.tsv"),
        excl3_tsv,
        "…and so does the FIRST exclusion's artifact — the reporter ran five \
         exclusions and each overwrote the last"
    );
    assert_eq!(
        free_forward_n_draws(&read(&excl13_dir, "predictive-excl1,3/weekly_cases.tsv")),
        BTreeSet::from([60]),
        "two chains remain"
    );

    // 4. Each address carries its own manifest, stamped with its own selection.
    let pooled: serde_json::Value = serde_json::from_str(&pooled_json).unwrap();
    assert!(
        pooled.get("chain_selection").is_none(),
        "the pooled manifest records no selection"
    );
    for (dir, name, excluded, kept) in [
        (&excl3_dir, "predictive-excl3.json", vec![3], vec![0, 1, 2]),
        (&excl13_dir, "predictive-excl1,3.json", vec![1, 3], vec![0, 2]),
    ] {
        let j: serde_json::Value = serde_json::from_str(&read(dir, name)).unwrap();
        let cs = j
            .get("chain_selection")
            .unwrap_or_else(|| panic!("{name} stamps chain_selection"));
        assert_eq!(cs["excluded"], serde_json::json!(excluded), "{name}");
        assert_eq!(cs["kept"], serde_json::json!(kept), "{name}");
        assert_eq!(j["options"]["excluded_chains"], serde_json::json!(excluded),
            "{name} records the selection among the options that key its address");
        // The manifest's declared file location must be the keyed one, or a
        // consumer that follows it lands back on the pooled artifact.
        let file = j["streams"][0]["file"].as_str().expect("a stream file path");
        assert!(
            file.starts_with(name.trim_end_matches(".json")),
            "{name} must point at its OWN directory, got {file}"
        );
        assert!(dir.join(file).exists(), "{name} declares a file that exists: {file}");
    }

    // 5. The quantities sidecar is posterior-derived too. The pooled table must
    //    survive a chain-subset run.
    assert!(
        pooled_dir.join("quantities/peak.tsv").exists(),
        "the pooled quantities table survives the exclusions"
    );
    assert!(
        excl3_dir.join("quantities-excl3/peak.tsv").exists()
            && excl13_dir.join("quantities-excl1,3/peak.tsv").exists(),
        "each exclusion writes its own quantities table"
    );

    // 6. Discovery: `camdl show <fit>` enumerates every address the fit holds,
    //    so a user finds the subsets without knowing the naming rule. A
    //    fixed-name envelope would list only the pooled artifact and read as
    //    "the subset was never generated".
    let seg = f.seg.to_string_lossy().into_owned();
    let out = run(&bin, &f.tmp, &["show", &seg]);
    assert_ok(&out, "camdl show <segment>");
    let shown = String::from_utf8_lossy(&out.stdout);
    for expected in [
        "predictive/weekly_cases.tsv",
        "predictive-excl3/weekly_cases.tsv",
        "predictive-excl1,3/weekly_cases.tsv",
        "predictive.json",
        "predictive-excl3.json",
        "predictive-excl1,3.json",
        "quantities-excl3/peak.tsv",
    ] {
        assert!(
            shown.contains(expected),
            "the fit envelope must list {expected}; got:\n{shown}"
        );
    }

    // …and the name `show` prints is a name `cat --stream` takes, so the
    // discovery loop closes: no keyed artifact is listed but unreachable.
    // `--mode free_forward` keys the predict's address (gh#896), so its
    // files are listed and read under `predict/<dir>/`.
    let prefix = format!("predict/{}/", excl3_dir.file_name().unwrap().to_string_lossy());
    let stream = format!("{prefix}predictive-excl3/weekly_cases.tsv");
    assert!(shown.contains(&stream), "show lists {stream}; got:\n{shown}");
    let out = run(&bin, &f.tmp, &["cat", &seg, "--stream", &stream]);
    assert_ok(&out, "camdl cat --stream predictive-excl3/weekly_cases.tsv");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        excl3_tsv,
        "cat must return the keyed artifact's own bytes"
    );

    let _ = std::fs::remove_dir_all(&f.tmp);
}

/// The address is keyed on the normalised SET, exactly as `ChainSelection`
/// normalises it: order and whitespace do not make a second artifact, and a
/// repeated id is refused at parse rather than collapsed, so it writes nothing.
#[test]
fn the_address_is_the_normalised_set_not_the_string() {
    let bin = require_binary();
    let f = setup(&bin, "normalised");

    assert_ok(&f.predict(&bin, Some("1,3")), "predict --exclude-chains 1,3");
    let after_first = f.artifact_names();
    let canonical = f.read("predictive-excl1,3/weekly_cases.tsv");

    let repeated = f.predict(&bin, Some("1,3,1"));
    assert!(!repeated.status.success(), "a repeated chain id is refused");
    let stderr = String::from_utf8_lossy(&repeated.stderr);
    assert!(stderr.contains("--exclude-chains lists chain 1 twice"), "{stderr}");
    assert_eq!(f.artifact_names(), after_first, "a refused selection writes nothing");

    for spelling in ["3,1", " 3 , 1 "] {
        assert_ok(
            &f.predict(&bin, Some(spelling)),
            &format!("predict --exclude-chains {spelling}"),
        );
        assert_eq!(
            f.artifact_names(),
            after_first,
            "`{spelling}` is the same exclusion as `2,4` and must resolve to the \
             same address, not a second one"
        );
        assert_eq!(
            f.read("predictive-excl1,3/weekly_cases.tsv"),
            canonical,
            "`{spelling}` rewrites the same artifact (a correct collision)"
        );
    }

    let _ = std::fs::remove_dir_all(&f.tmp);
}

/// The regression guard for every existing reader: with no `--exclude-chains`
/// the run directory holds exactly the historical names, and nothing else.
#[test]
fn without_the_flag_the_artifact_names_are_the_historical_ones() {
    let bin = require_binary();
    let f = setup(&bin, "unexcluded");

    assert_ok(&f.predict(&bin, None), "pooled predict");
    assert_eq!(
        f.artifact_names(),
        vec![
            "predictive",
            "predictive.json",
            "quantities",
            "quantities.json"
        ],
        "an unexcluded run must land exactly where it always has — camdl-scope \
         reads `predictive/`"
    );
    assert!(f.exists("predictive/weekly_cases.tsv"));
    assert!(f.exists("quantities/peak.tsv"));

    // `observed/` is the fit's DATA, identical under any selection, so it never
    // moves — a keyed copy would be a pointless duplicate of the same bytes.
    let observed = f.read("observed/weekly_cases.tsv");
    assert_ok(&f.predict(&bin, Some("3")), "predict --exclude-chains 3");
    assert_eq!(
        f.read("observed/weekly_cases.tsv"),
        observed,
        "the observed half does not depend on the chain selection"
    );
    assert!(
        !predict_out(&f.seg).join("observed-excl3").exists(),
        "…so it is not keyed either"
    );

    let _ = std::fs::remove_dir_all(&f.tmp);
}

// ── `--chains`: the keeping polarity of the one chain selector ───────────────

/// `--chains 0,1,2` on the four-chain cloud is `--exclude-chains 3`: one
/// stored prediction, one address, the same bytes, and a warning on stderr
/// worded by polarity. Both flags together are refused.
#[test]
fn chains_and_its_complementary_exclusion_are_one_prediction() {
    let bin = require_binary();
    let f = setup(&bin, "keep");
    let seg = f.seg.to_string_lossy().into_owned();
    let predict = |flag: &str, ids: &str| {
        run(&bin, &f.tmp, &["fit", "predict", &seg, "--mode", "free_forward", flag, ids])
    };

    let kept = predict("--chains", "0,1,2");
    assert_ok(&kept, "predict --chains 0,1,2");
    let stderr = String::from_utf8_lossy(&kept.stderr);
    assert!(
        stderr.contains("--chains: keeping only chain(s) 0,1,2; dropping 3 (3 of 4 chains kept)"),
        "the warning names what was kept and what was dropped:\n{stderr}"
    );
    assert!(!String::from_utf8_lossy(&kept.stdout).contains("keeping only"),
        "the warning is on stderr, never stdout");
    let names = f.artifact_names();
    assert!(names.contains(&"predictive-excl3".to_string()), "{names:?}");
    let via_keep = f.read("predictive-excl3/weekly_cases.tsv");

    assert_ok(&predict("--exclude-chains", "3"), "predict --exclude-chains 3");
    assert_eq!(f.artifact_names(), names, "the complement writes no second artifact");
    assert_eq!(f.read("predictive-excl3/weekly_cases.tsv"), via_keep,
        "the two spellings band the same cloud");

    let both = run(&bin, &f.tmp, &["fit", "predict", &seg, "--mode", "free_forward",
        "--chains", "0", "--exclude-chains", "3"]);
    assert!(!both.status.success(), "both flags together are refused");

    // The parse tightenings and the keep-everything refusal reach the CLI.
    for (ids, want) in [("0,0", "lists chain 0 twice"), ("0,,1", "empty chain id"),
                        ("0,1,2,3", "keeps every chain"), ("7", "chain 7 not in this fit")] {
        let o = predict("--chains", ids);
        assert!(!o.status.success(), "--chains {ids} is refused");
        let e = String::from_utf8_lossy(&o.stderr);
        assert!(e.contains(want), "--chains {ids}: {e}");
    }

    let _ = std::fs::remove_dir_all(&f.tmp);
}

/// `cat --chains` on `draws.tsv` prints the kept chains' rows verbatim, and is
/// refused for a stream whose recorded schema has no `chain` column.
#[test]
fn cat_selects_the_rows_of_draws_tsv_by_chain() {
    let bin = require_binary();
    let f = setup(&bin, "cat");
    let draws = find_draws_tsv(&f.seg).unwrap();
    let leaf = draws.parent().unwrap().to_string_lossy().into_owned();

    let out = run(&bin, &f.tmp, &["cat", &leaf, "--stream", "draws.tsv", "--chains", "3"]);
    assert_ok(&out, "cat --chains 3");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut want = String::new();
    for (i, line) in build_cloud().lines().enumerate() {
        if i == 0 || line.starts_with("3\t") {
            want.push_str(line);
            want.push('\n');
        }
    }
    assert_eq!(text, want, "the header and chain 3's rows, byte for byte");
    assert!(String::from_utf8_lossy(&out.stderr).contains("--chains: keeping only chain(s) 3"));

    let out = run(&bin, &f.tmp, &["cat", &leaf, "--stream", "draws.tsv", "--exclude-chains", "3"]);
    assert_ok(&out, "cat --exclude-chains 3");
    let n_rows = String::from_utf8_lossy(&out.stdout).lines().count() - 1;
    assert_eq!(n_rows, 3 * DRAWS_PER_CHAIN);

    let out = run(&bin, &f.tmp, &["cat", &leaf, "--stream", "fit_state.toml", "--chains", "0"]);
    assert!(!out.status.success(), "a stream without a chain column is refused");
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(e.contains("`fit_state.toml` has no `chain` column"), "{e}");

    let _ = std::fs::remove_dir_all(&f.tmp);
}

/// `simulate --draws posterior --chains 3` replays only chain 3's draws: every
/// replayed `beta` is the outlier chain's.
#[test]
fn simulate_replays_only_the_selected_chains() {
    let bin = require_binary();
    let f = setup(&bin, "simulate");
    let draws = find_draws_tsv(&f.seg).unwrap();
    let leaf = draws.parent().unwrap().to_string_lossy().into_owned();
    let read_betas = |path: &Path| -> Vec<f64> {
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let col = header.iter().position(|c| *c == "beta").expect("a beta column");
        lines.map(|l| l.split('\t').nth(col).unwrap().parse().unwrap()).collect()
    };

    let out = run(&bin, &f.tmp, &["simulate", "model.camdl", "--draws", "posterior",
        "--fit", &leaf, "--chains", "3", "-n", "5", "--draws-out", "kept.tsv"]);
    assert_ok(&out, "simulate --chains 3");
    assert!(String::from_utf8_lossy(&out.stderr).contains("--chains: keeping only chain(s) 3"));
    let betas = read_betas(&f.tmp.join("kept.tsv"));
    assert_eq!(betas.len(), 5);
    assert!(betas.iter().all(|b| *b > 0.8), "every draw is chain 3's: {betas:?}");

    let out = run(&bin, &f.tmp, &["simulate", "model.camdl", "--draws", "posterior",
        "--fit", &leaf, "--exclude-chains", "3", "-n", "5", "--draws-out", "dropped.tsv"]);
    assert_ok(&out, "simulate --exclude-chains 3");
    assert!(read_betas(&f.tmp.join("dropped.tsv")).iter().all(|b| *b < 0.5));

    let out = run(&bin, &f.tmp, &["simulate", "model.camdl", "--chains", "3"]);
    assert!(!out.status.success(), "--chains needs --draws posterior");
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs --draws posterior"));

    let _ = std::fs::remove_dir_all(&f.tmp);
}
