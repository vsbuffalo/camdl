//! gh#695 — `--exclude-chains` must reach the counterfactual contrast path.
//!
//! `fit predict` applies a read-side chain selection to the free-forward
//! posterior cloud. Before this fix it did NOT apply it to the `(θ, X)` joint the
//! contrast reducer forks from, so `contrasts/<name>.tsv` banded over the very
//! chain `predictive.json`'s `chain_selection` recorded as excluded — the
//! artifact and its manifest disagreeing, silently, on a number a policy
//! question is answered from.
//!
//! The measurements here are the issue's own: with a two-chain PGAS fit,
//!
//!   * the contrast artifact under `--exclude-chains 2` must DIFFER from the
//!     artifact of the same run without the flag (before the fix it was
//!     byte-identical);
//!   * the contrast's `n_used` must equal the free-forward rows' `n_draws` —
//!     both bands over one cloud;
//!   * that cloud must be exactly the retained chain's draws.
//!
//! Plus the two edges a chain filter adds on top of the forkable subset (draws
//! with a saved latent path): excluding chains narrows an already-partial set,
//! so an EMPTY intersection is refused by name rather than emitted as a tiny or
//! empty band, and excluding a chain that had no saved paths still behaves.

use std::collections::HashMap;
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
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
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

/// SIRD + a week-4 SIA, two scenarios toggling it, and one scalar contrast
/// (`deaths averted`). The contrast forks from the last saved snapshot before
/// the SIA fires, so its value is a real function of the posterior draws it
/// bands over — which is what makes "did the excluded chain contribute?"
/// answerable from the artifact.
const MODEL: &str = r#"
time_unit = 'days
origin     = date("2020-01-01")
compartments { S, I, R, D, V }
parameters {
  beta  : rate         in [0.05, 1.5]  ~ log_normal(mu = -0.5, sigma = 0.5)
  gamma : rate         in [0.05, 0.5]  ~ log_normal(mu = -1.5, sigma = 0.5)
  mu    : rate         in [0.0, 0.3]
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R + D + V
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
  death     : I --> D  @ mu * I
}
init { S = N0 - I0  I = I0 }
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 4 'weeks]
}
scenarios {
  no_sia   { disable = [sia] }
  with_sia { enable  = [sia] }
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
  total = final(D)
}
contrasts {
  averted = no_sia.quantities.total - with_sia.quantities.total
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// Two PGAS chains. `n_trajectories` defaults to 200 and only 25 post-burn-in
/// sweeps exist per chain, so the save stride is 1: every posterior draw carries
/// a saved latent path. The forkable subset is therefore the whole cloud, and
/// any shortfall in the contrast's `n_used` is the chain filter, not the stride.
const FIT_TOML: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.5], start = 0.5 }
gamma = { bounds = [0.05, 0.5], start = 0.15 }
[fixed]
mu  = 0.05
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 120
sweeps = 40
burn_in = 15
thin = 1
"#;

fn workspace(tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("camdl_gh695_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();
    tmp
}

fn fit_run(bin: &Path, tmp: &Path) {
    let out = run(bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The single fit segment under `results/fits/`.
fn segment(tmp: &Path) -> PathBuf {
    let fits = tmp.join("results").join("fits");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&fits)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", fits.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert_eq!(dirs.len(), 1, "one fit segment expected, got {dirs:?}");
    dirs.pop().unwrap()
}

/// The stage leaf holding `draws.tsv` and the `chain_*/` path dirs.
fn stage_dir(seg: &Path) -> PathBuf {
    fn walk(dir: &Path) -> Option<PathBuf> {
        if dir.join("draws.tsv").is_file() {
            return Some(dir.to_path_buf());
        }
        for e in std::fs::read_dir(dir).ok()?.flatten() {
            if e.path().is_dir() {
                if let Some(p) = walk(&e.path()) {
                    return Some(p);
                }
            }
        }
        None
    }
    walk(seg).unwrap_or_else(|| panic!("no draws.tsv under {}", seg.display()))
}

/// Draw-row counts per 0-based chain id, read off the fit's own `draws.tsv` —
/// so the expected cloud size is the fit's, never a number restated from the
/// sampler config.
fn draws_per_chain(seg: &Path) -> HashMap<usize, usize> {
    let path = stage_dir(seg).join("draws.tsv");
    let txt = std::fs::read_to_string(&path).unwrap();
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("draws header").split('\t').collect();
    let ci = header
        .iter()
        .position(|c| *c == "chain")
        .unwrap_or_else(|| panic!("draws.tsv has no chain column: {header:?}"));
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for l in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = l.split('\t').collect();
        let c: usize = f[ci].parse().expect("chain id");
        *counts.entry(c).or_default() += 1;
    }
    counts
}

fn contrast_path(seg: &Path, name: &str) -> PathBuf {
    seg.join("contrasts").join(format!("{name}.tsv"))
}

/// The `n_used` cell of a scalar contrast's single band row — the count of
/// draws the band was actually computed over.
fn contrast_n_used(path: &Path) -> usize {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let i = header
        .iter()
        .position(|c| *c == "n_used")
        .unwrap_or_else(|| panic!("no n_used column: {header:?}"));
    let row: Vec<&str> = lines.next().expect("one band row").split('\t').collect();
    row[i].parse().unwrap_or_else(|e| panic!("n_used not a count: {} ({e})", row[i]))
}

/// The distinct `n_draws` values across a free-forward predictive stream — the
/// size of the posterior cloud each banded cell pooled.
fn predictive_n_draws(seg: &Path, stream: &str) -> Vec<usize> {
    let path = seg.join("predictive").join(format!("{stream}.tsv"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let i = header
        .iter()
        .position(|c| *c == "n_draws")
        .unwrap_or_else(|| panic!("no n_draws column: {header:?}"));
    let mut seen: Vec<usize> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            f[i].parse::<usize>().expect("n_draws is a count")
        })
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen
}

/// The `chain_selection` object `fit predict` stamps into `predictive.json`.
fn manifest_chain_selection(seg: &Path) -> Option<serde_json::Value> {
    let path = seg.join("predictive.json");
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&txt).expect("predictive.json parses");
    v.get("chain_selection").cloned()
}

fn predict(bin: &Path, tmp: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward"];
    args.extend_from_slice(extra);
    run(bin, tmp, &args)
}

/// The headline: a contrast must band over the cloud the run's own manifest
/// describes. Before the fix, `contrasts/averted.tsv` was byte-identical with
/// and without `--exclude-chains 2` while `predictive.json` recorded chain 2 as
/// excluded — the artifact and the manifest disagreeing about which draws
/// produced the number.
#[test]
fn contrast_bands_over_the_retained_chains_only() {
    let bin = skip_if_missing_binary();
    let tmp = workspace("retained");
    fit_run(&bin, &tmp);
    let seg = segment(&tmp);

    let per_chain = draws_per_chain(&seg);
    assert_eq!(per_chain.len(), 2, "the fixture fit must have two chains, got {per_chain:?}");
    let n_all: usize = per_chain.values().sum();
    let n_kept = per_chain[&0]; // 1-based chain 1 = 0-based 0, the retained one

    // (a) The full-cloud run — the baseline both later comparisons are made
    //     against.
    let out = predict(&bin, &tmp, &[]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let averted = contrast_path(&seg, "averted");
    let full_bytes = std::fs::read(&averted).expect("contrasts/averted.tsv must be emitted");
    let full_n_used = contrast_n_used(&averted);
    assert_eq!(
        full_n_used, n_all,
        "sanity: with no selection the contrast bands the whole cloud ({n_all} draws)"
    );
    assert_eq!(
        manifest_chain_selection(&seg),
        None,
        "an unselected run records no chain_selection"
    );

    // (b) The same run with chain 2 excluded.
    let out = predict(&bin, &tmp, &["--exclude-chains", "2"]);
    assert!(
        out.status.success(),
        "fit predict --exclude-chains failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let sub_bytes = std::fs::read(&averted).unwrap();
    let sub_n_used = contrast_n_used(&averted);

    // The artifact must MOVE. Byte-identity here is the whole bug: it means the
    // excluded chain's draws were forked into the band regardless.
    assert_ne!(
        sub_bytes, full_bytes,
        "the contrast artifact must differ when a chain is excluded; it was \
         byte-identical, so the excluded chain still contributed to the band"
    );

    // The contrast's denominator and the free-forward rows' denominator are the
    // same cloud — the issue's own measurement (`n_used` vs `n_draws`).
    let ff = predictive_n_draws(&seg, "weekly_cases");
    assert_eq!(
        ff,
        vec![n_kept],
        "every free-forward cell bands the {n_kept} retained draws, got {ff:?}"
    );
    assert_eq!(
        sub_n_used, n_kept,
        "the contrast must band the {n_kept} retained draws, not the full {n_all}"
    );
    assert_eq!(
        sub_n_used, ff[0],
        "contrast n_used ({sub_n_used}) and free-forward n_draws ({}) must be one cloud",
        ff[0]
    );

    // The manifest and the data now agree: the manifest names chain 2 as
    // excluded, and the contrast really did drop it.
    let sel = manifest_chain_selection(&seg).expect("a selected run stamps chain_selection");
    assert_eq!(sel["excluded"], serde_json::json!([2]), "manifest: {sel}");
    assert_eq!(sel["kept"], serde_json::json!([1]), "manifest: {sel}");
    assert_eq!(sel["n_total"], serde_json::json!(2), "manifest: {sel}");

    // The count is visible, not only inferable from the file: the reducer names
    // the retained-chain scope on stderr.
    assert!(
        stderr.contains("contrasts band over the RETAINED chains")
            && stderr.contains("chain(s) 2 excluded"),
        "the reducer must name the retained-chain scope and the dropped chain; \
         stderr:\n{stderr}"
    );

    // (c) Negative control: with no selection the artifact returns to exactly
    //     the bytes of (a). The fix touches the selected path only.
    let out = predict(&bin, &tmp, &[]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::read(&averted).unwrap(),
        full_bytes,
        "an unselected re-run must reproduce the full-cloud artifact byte-for-byte"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The forkable subset and the chain filter intersect. A contrast can only fork
/// a draw with a saved latent path; excluding chains narrows that set again. If
/// the intersection is empty the run must refuse BY NAME — "this fit has no
/// forkable draws" would blame the fit for a consequence of the selection, and
/// an empty band would be a fabricated answer.
///
/// Constructed by deleting one chain's saved paths, which is the on-disk shape
/// of "this chain saved nothing" (a PGAS stage run with `n_trajectories = 0`,
/// or an interrupted write).
#[test]
fn empty_forkable_intersection_refuses_by_name() {
    let bin = skip_if_missing_binary();
    let tmp = workspace("empty");
    fit_run(&bin, &tmp);
    let seg = segment(&tmp);
    let per_chain = draws_per_chain(&seg);
    let n_chain0 = per_chain[&0];

    // Drop 0-based chain 1's saved paths (the on-disk dir is 1-based: chain_2).
    let orphaned = stage_dir(&seg).join("chain_2");
    assert!(orphaned.is_dir(), "expected saved paths at {}", orphaned.display());
    std::fs::remove_dir_all(&orphaned).unwrap();

    // Excluding chain 1 retains only the chain whose paths are gone → the
    // (retained ∩ forkable) intersection is empty.
    let out = predict(&bin, &tmp, &["--exclude-chains", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "an empty forkable intersection must be refused, not banded; it \
         succeeded.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("--exclude-chains 1") && stderr.contains("no forkable posterior draws"),
        "the refusal must name the selection as the cause; stderr:\n{stderr}"
    );
    assert!(
        !contrast_path(&seg, "averted").exists(),
        "a refused contrast must leave no artifact behind"
    );

    // Negative control: excluding the chain that has NO saved paths is not an
    // error at all — the retained chain still carries a full forkable cloud, and
    // the band is over exactly its draws.
    let out = predict(&bin, &tmp, &["--exclude-chains", "2"]);
    assert!(
        out.status.success(),
        "excluding the path-less chain must still predict:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let averted = contrast_path(&seg, "averted");
    assert!(averted.is_file(), "the retained chain's contrast must be emitted");
    assert_eq!(
        contrast_n_used(&averted),
        n_chain0,
        "the band is over the retained chain's {n_chain0} draws"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
