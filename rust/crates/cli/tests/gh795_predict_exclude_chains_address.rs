//! gh#795 — a chain-subset `fit predict` must not be written at the pooled
//! run's address.
//!
//! `--exclude-chains` reports a DIFFERENT posterior (its own warning says so:
//! "Post-hoc chain exclusion BIASES the posterior toward the retained mode").
//! Before this fix the excluded set reached `predictive.json` as a
//! `chain_selection` stamp but never reached the artifact's location, so
//! `fit predict --exclude-chains 4` replaced the run's canonical predictive
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

[stages.posterior]
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

/// A controlled 4-chain cloud: chains 0,1,2 (the user's 1,2,3) tight around a
/// slow epidemic, chain 3 (the user's 4) a stuck outlier at a fast one. The
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
    Command::new(bin)
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
        let mut args: Vec<&str> = vec!["fit", "predict", &seg, "--horizon", "free_forward"];
        if let Some(ids) = exclude {
            args.push("--exclude-chains");
            args.push(ids);
        }
        run(bin, &self.tmp, &args)
    }

    /// The predictive/quantities artifact names directly under the segment,
    /// sorted. This IS the run directory as a reader (`ls`, `camdl-scope`) sees
    /// it, which is the surface the bug corrupts.
    fn artifact_names(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.seg)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("predictive") || n.starts_with("quantities"))
            .collect();
        names.sort();
        names
    }

    fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.seg.join(rel))
            .unwrap_or_else(|e| panic!("reading {rel} under the segment: {e}"))
    }

    fn exists(&self, rel: &str) -> bool {
        self.seg.join(rel).exists()
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

    // 1. The pooled run: the canonical artifact, at the historical address.
    assert_ok(&f.predict(&bin, None), "pooled predict");
    let pooled_tsv = f.read("predictive/weekly_cases.tsv");
    let pooled_json = f.read("predictive.json");
    assert_eq!(
        free_forward_n_draws(&pooled_tsv),
        BTreeSet::from([120]),
        "the pooled cloud is all 4 chains × 30 draws"
    );

    // 2. Drop the outlier chain 4. This must NOT be the pooled address.
    assert_ok(&f.predict(&bin, Some("4")), "predict --exclude-chains 4");
    assert_eq!(
        f.read("predictive/weekly_cases.tsv"),
        pooled_tsv,
        "a chain-subset predict must not overwrite the pooled predictive TSV — \
         camdl-scope reads `predictive/` and would render a 3-chain subset as \
         the posterior predictive of the fit"
    );
    assert_eq!(
        f.read("predictive.json"),
        pooled_json,
        "…nor the pooled manifest"
    );
    let excl4_tsv = f.read("predictive-excl4/weekly_cases.tsv");
    assert_eq!(
        free_forward_n_draws(&excl4_tsv),
        BTreeSet::from([90]),
        "the subset cloud is 3 chains × 30 draws"
    );

    // 3. A second, different exclusion. All three artifacts coexist.
    assert_ok(&f.predict(&bin, Some("2,4")), "predict --exclude-chains 2,4");
    assert_eq!(
        f.read("predictive/weekly_cases.tsv"),
        pooled_tsv,
        "the pooled predictive survives the second exclusion too"
    );
    assert_eq!(
        f.read("predictive-excl4/weekly_cases.tsv"),
        excl4_tsv,
        "…and so does the FIRST exclusion's artifact — the reporter ran five \
         exclusions and each overwrote the last"
    );
    assert_eq!(
        free_forward_n_draws(&f.read("predictive-excl2,4/weekly_cases.tsv")),
        BTreeSet::from([60]),
        "two chains remain"
    );

    // 4. Each address carries its own manifest, stamped with its own selection.
    let pooled: serde_json::Value = serde_json::from_str(&pooled_json).unwrap();
    assert!(
        pooled.get("chain_selection").is_none(),
        "the pooled manifest records no selection"
    );
    for (name, excluded, kept) in [
        ("predictive-excl4.json", vec![4], vec![1, 2, 3]),
        ("predictive-excl2,4.json", vec![2, 4], vec![1, 3]),
    ] {
        let j: serde_json::Value = serde_json::from_str(&f.read(name)).unwrap();
        let cs = j
            .get("chain_selection")
            .unwrap_or_else(|| panic!("{name} stamps chain_selection"));
        assert_eq!(cs["excluded"], serde_json::json!(excluded), "{name}");
        assert_eq!(cs["kept"], serde_json::json!(kept), "{name}");
        // The manifest's declared file location must be the keyed one, or a
        // consumer that follows it lands back on the pooled artifact.
        let file = j["streams"][0]["file"].as_str().expect("a stream file path");
        assert!(
            file.starts_with(name.trim_end_matches(".json")),
            "{name} must point at its OWN directory, got {file}"
        );
        assert!(f.exists(file), "{name} declares a file that exists: {file}");
    }

    // 5. The quantities sidecar is posterior-derived too, and collides the same
    //    way. The pooled table must survive a chain-subset run.
    assert!(
        f.exists("quantities/peak.tsv"),
        "the pooled quantities table survives the exclusions"
    );
    assert!(
        f.exists("quantities-excl4/peak.tsv") && f.exists("quantities-excl2,4/peak.tsv"),
        "each exclusion writes its own quantities table, got {:?}",
        f.artifact_names()
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
        "predictive-excl4/weekly_cases.tsv",
        "predictive-excl2,4/weekly_cases.tsv",
        "predictive.json",
        "predictive-excl4.json",
        "predictive-excl2,4.json",
        "quantities-excl4/peak.tsv",
    ] {
        assert!(
            shown.contains(expected),
            "the fit envelope must list {expected}; got:\n{shown}"
        );
    }
    // The pooled address leads its family, so a reader scanning the envelope
    // meets the canonical artifact before the subsets.
    let order = |a: &str, b: &str| {
        let (ia, ib) = (shown.find(a), shown.find(b));
        assert!(
            ia.is_some() && ia < ib,
            "{a} must be listed before {b}; got:\n{shown}"
        );
    };
    order("predictive/weekly_cases.tsv", "predictive-excl2,4/weekly_cases.tsv");
    order("\n  predictive.json", "predictive-excl2,4.json");

    // …and the name `show` prints is a name `cat --stream` takes, so the
    // discovery loop closes: no keyed artifact is listed but unreachable.
    let out = run(&bin, &f.tmp, &["cat", &seg, "--stream", "predictive-excl4/weekly_cases.tsv"]);
    assert_ok(&out, "camdl cat --stream predictive-excl4/weekly_cases.tsv");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        excl4_tsv,
        "cat must return the keyed artifact's own bytes"
    );

    let _ = std::fs::remove_dir_all(&f.tmp);
}

/// The address is keyed on the normalised SET, exactly as `ChainSelection`
/// normalises it: order and repeats do not make a second artifact.
#[test]
fn the_address_is_the_normalised_set_not_the_string() {
    let bin = require_binary();
    let f = setup(&bin, "normalised");

    assert_ok(&f.predict(&bin, Some("2,4")), "predict --exclude-chains 2,4");
    let after_first = f.artifact_names();
    let canonical = f.read("predictive-excl2,4/weekly_cases.tsv");

    for spelling in ["4,2", "2,4,2", " 4 , 2 "] {
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
            f.read("predictive-excl2,4/weekly_cases.tsv"),
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
    assert_ok(&f.predict(&bin, Some("4")), "predict --exclude-chains 4");
    assert_eq!(
        f.read("observed/weekly_cases.tsv"),
        observed,
        "the observed half does not depend on the chain selection"
    );
    assert!(
        !f.seg.join("observed-excl4").exists(),
        "…so it is not keyed either"
    );

    let _ = std::fs::remove_dir_all(&f.tmp);
}
