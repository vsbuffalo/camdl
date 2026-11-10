//! A resumed PMMH or PGAS leaf holds the whole chain.
//!
//! `fit run --resume <base>` writes a leaf of its own; that leaf is the
//! complete artifact for the longer chain. Each chain's `trace.tsv` (and, for
//! PGAS, `trajectories.tsv`) starts as the base's file byte for byte and the
//! new segment is appended, the index continuing from the base's without a
//! repeated or restarted row; `draws.tsv` and the summary's R̂ / ESS are over
//! every retained draw of the chain; and `run.json` `inputs.resume` names the
//! base and the index each segment begins at.
//!
//! A resumed chain's random stream differs from an uninterrupted one's (the
//! sampler reseeds at the boundary), so the uninterrupted run of the same total
//! length is compared on its counts, not its values.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A work directory holding the reporting fixture's SIR model compiled to IR
/// (so `camdl` needs no compiler) and its case counts.
fn workdir(tag: &str) -> (Tmp, PathBuf) {
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_whole_chain_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let fixtures = manifest().join("tests/fixtures/reporting");
    std::fs::copy(fixtures.join("cases.tsv"), work.join("cases.tsv")).unwrap();
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(fixtures.join("sir.camdl"))
        .output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(work.join("sir.ir.json"), &out.stdout).unwrap();
    (Tmp(work.clone()), work)
}

fn camdl(work: &Path, args: &[&str]) {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let out = Command::new(&bin)
        .env_remove("CLICOLOR_FORCE")
        .current_dir(work)
        .args(args)
        .args(["--progress", "none"])
        .output()
        .unwrap();
    assert!(out.status.success(), "camdl {args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// The single method leaf under `results/fits/<stem>-*/`.
fn leaf(work: &Path, stem: &str) -> PathBuf {
    let only = |dir: &Path, prefix: &str| -> PathBuf {
        let hits: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().starts_with(prefix))
            .collect();
        assert_eq!(hits.len(), 1, "one `{prefix}*` under {}: {hits:?}", dir.display());
        hits.into_iter().next().unwrap()
    };
    let fit = only(&work.join("results/fits"), &format!("{stem}-"));
    let method = only(&fit, "");
    only(&method, "seed_")
}

fn run_json(leaf: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap()
}

/// The `index` column of a trace, in row order.
fn trace_index(trace: &Path, column: &str) -> Vec<usize> {
    let text = std::fs::read_to_string(trace).unwrap();
    let mut lines = text.lines();
    let at = lines.next().unwrap().split('\t').position(|c| c == column).unwrap();
    lines.map(|l| l.split('\t').nth(at).unwrap().parse().unwrap()).collect()
}

/// `draws.tsv`'s rows of one chain, as written.
fn chain_draw_rows(leaf: &Path, chain: usize) -> Vec<String> {
    std::fs::read_to_string(leaf.join("draws.tsv")).unwrap().lines().skip(1)
        .filter(|l| l.split('\t').next() == Some(chain.to_string().as_str()))
        .map(str::to_string)
        .collect()
}

/// One parameter's draws for one chain, from `draws.tsv`.
fn chain_draws(leaf: &Path, chain: usize, param: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(leaf.join("draws.tsv")).unwrap();
    let at = text.lines().next().unwrap().split('\t').position(|c| c == param).unwrap();
    chain_draw_rows(leaf, chain).iter().map(|l| l.split('\t').nth(at).unwrap().parse().unwrap()).collect()
}

/// The checks both samplers share. `n1` and `n2` are the base's and the
/// resumed run's length; every row of the trace is a step (PMMH at `thin = 1`)
/// or sweep (PGAS) of the chain.
fn assert_whole_chain(base: &Path, resumed: &Path, fresh: &Path, index: &str, chains: usize, n1: usize, n2: usize) {
    let summary_file = if index == "sweep" { "pgas_summary.json" } else { "pmmh_summary.json" };
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(resumed.join(summary_file)).unwrap()).unwrap();
    let mut distinguished = 0usize;
    for c in 0..chains {
        let trace = |leaf: &Path| leaf.join(format!("chain_{c}/trace.tsv"));
        let parent_bytes = std::fs::read(trace(base)).unwrap();
        let resumed_bytes = std::fs::read(trace(resumed)).unwrap();
        assert!(resumed_bytes.starts_with(&parent_bytes),
            "chain {c}: the resumed trace begins with the base's, byte for byte");

        let parent_idx = trace_index(&trace(base), index);
        let idx = trace_index(&trace(resumed), index);
        assert_eq!(parent_idx.len(), n1, "chain {c}: the base's trace");
        assert_eq!(idx.len(), parent_idx.len() + (n2 - n1),
            "chain {c}: the resumed trace holds the base's rows and the new ones");
        assert_eq!(idx, (0..n2).collect::<Vec<_>>(),
            "chain {c}: `{index}` runs 0..{n2} by one across the boundary at {n1}, \
             the boundary row once, the numbering continued");
        assert_eq!(idx.len(), trace_index(&trace(fresh), index).len(),
            "chain {c}: as many rows as an uninterrupted run of the same length");

        let parent_rows = chain_draw_rows(base, c);
        let rows = chain_draw_rows(resumed, c);
        assert!(!parent_rows.is_empty() && rows.len() > parent_rows.len(), "chain {c}: draws");
        assert_eq!(rows[..parent_rows.len()], parent_rows[..],
            "chain {c}: draws.tsv opens with the base's draws");
        assert_eq!(rows.len(), chain_draw_rows(fresh, c).len(),
            "chain {c}: as many draws as an uninterrupted run of the same length");

        // Precondition for the ESS checks below: the sampler moved in both
        // segments. A chain whose θ never changes has an ESS at its floor over
        // any span, so the whole chain and the segment could not be told
        // apart — and a frozen sampler is a defect to report, not a pass.
        for p in ["beta", "gamma"] {
            let whole = chain_draws(resumed, c, p);
            let moved = |span: &[f64]| span.windows(2).any(|w| w[0] != w[1]);
            assert!(moved(&whole[..parent_rows.len()]) && moved(&whole[parent_rows.len()..]),
                "chain {c} {p}: θ must move in the base's draws and in the new ones; \
                 a frozen chain makes the ESS checks uninformative");
        }

        // The stored per-chain ESS is the one over every draw of the chain —
        // the base's included — not over the ones this call sampled.
        for p in ["beta", "gamma"] {
            let whole = chain_draws(resumed, c, p);
            let stored = summary["ess_per_chain"][p][c].as_f64().unwrap();
            let over_whole = sim::inference::pmmh::mcmc_ess(&whole);
            assert!((stored - over_whole).abs() <= 1e-9 * over_whole.abs().max(1.0),
                "chain {c} {p}: ESS {stored} is over the whole chain ({over_whole})");
            let over_segment = sim::inference::pmmh::mcmc_ess(&whole[parent_rows.len()..]);
            if (stored - over_segment).abs() > 1e-9 {
                distinguished += 1;
            }
        }
    }
    // The negative control: the equality above must be able to fail, so at
    // least one chain and parameter must give different values over the whole
    // chain and over the new segment (the chains moved, checked above).
    assert!(distinguished > 0, "no ESS distinguishes the whole chain from the new segment");

    // The lineage: the base, and where each segment begins.
    let base_id = run_json(base)["run_id"].as_str().unwrap().to_string();
    let rec = run_json(resumed);
    let lineage = &rec["inputs"]["resume"];
    assert_eq!(lineage["parent_run_id"].as_str(), Some(base_id.as_str()), "{lineage}");
    assert_eq!(lineage["index_column"].as_str(), Some(index), "{lineage}");
    let segments = lineage["segments"].as_array().unwrap();
    assert_eq!(segments.len(), 2, "{lineage}");
    assert_eq!(segments[0]["run_id"].as_str(), Some(base_id.as_str()), "{lineage}");
    assert_eq!(segments[0]["first_index"], serde_json::json!(vec![0; chains]), "{lineage}");
    assert_eq!(segments[1]["run_id"], rec["run_id"], "{lineage}");
    assert_eq!(segments[1]["first_index"], serde_json::json!(vec![n1; chains]),
        "the boundary is the base's length, per chain: {lineage}");
    assert!(run_json(fresh)["inputs"].get("resume").is_none(), "a fresh run records no lineage");
}

/// A `fit_state.toml` number (`best_chain`, `acceptance_rate`, `best_loglik`).
fn state_number(leaf: &Path, key: &str) -> f64 {
    let t = std::fs::read_to_string(leaf.join("fit_state.toml")).unwrap();
    t.lines().find_map(|l| l.strip_prefix(&format!("{key} = ")))
        .unwrap_or_else(|| panic!("no `{key}` in {}/fit_state.toml", leaf.display()))
        .trim().parse().unwrap()
}

const MODEL: &str = r#"
[model]
camdl = "sir.ir.json"
[data.observations]
cases = "cases.tsv"
[estimate]
beta  = { start = 0.5 }
gamma = { start = 0.2 }
[fixed]
rho = 0.5
N0  = 2000
"#;

#[test]
fn a_resumed_pmmh_leaf_holds_the_whole_chain() {
    let (_guard, work) = workdir("pmmh");
    let (n1, n2) = (40, 70);
    for (stem, n) in [("base", n1), ("longer", n2), ("fresh", n2)] {
        std::fs::write(work.join(format!("{stem}.toml")), format!("{MODEL}[method]
algorithm  = \"pmmh\"
backend    = \"chain_binomial\"
chains     = 2
particles  = 20
iterations = {n}
burn_in    = 10
thin       = 1
")).unwrap();
    }
    camdl(&work, &["fit", "run", "base.toml", "--seed", "1"]);
    let base = leaf(&work, "base");
    camdl(&work, &["fit", "run", "longer.toml", "--seed", "1", "--resume", base.to_str().unwrap()]);
    camdl(&work, &["fit", "run", "fresh.toml", "--seed", "1"]);
    let resumed = leaf(&work, "longer");
    assert_whole_chain(&base, &resumed, &leaf(&work, "fresh"), "step", 2, n1, n2);

    // `fit_state.toml`'s acceptance rate is the best chain's over every
    // post-burn-in step of the whole chain — at `thin = 1` the `accepted`
    // column of the steps at or past `burn_in` in its trace.
    let burn_in = 10;
    let trace_rate = |leaf: &Path, n: usize| -> f64 {
        let chain = state_number(leaf, "best_chain") as usize;
        let text = std::fs::read_to_string(leaf.join(format!("chain_{chain}/trace.tsv"))).unwrap();
        let mut lines = text.lines();
        let cols: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let (step, acc) = (cols.iter().position(|c| *c == "step").unwrap(),
            cols.iter().position(|c| *c == "accepted").unwrap());
        let accepted = lines.map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|f| f[step].parse::<usize>().unwrap() >= burn_in && f[acc] == "1")
            .count();
        accepted as f64 / (n - burn_in) as f64
    };
    for (leaf, n) in [(&base, n1), (&resumed, n2)] {
        let (stored, counted) = (state_number(leaf, "acceptance_rate"), trace_rate(leaf, n));
        assert!((stored - counted).abs() < 1e-12,
            "{}: acceptance_rate {stored} is the whole chain's {counted}", leaf.display());
    }
}

#[test]
fn a_resumed_pgas_leaf_holds_the_whole_chain() {
    let (_guard, work) = workdir("pgas");
    // A warm-up long enough for NUTS's step size to adapt: at `burn_in = 10`
    // the frozen step size made every post-burn-in NUTS step diverge and θ
    // never moved.
    let (n1, n2) = (150, 250);
    for (stem, n) in [("base", n1), ("longer", n2), ("fresh", n2)] {
        std::fs::write(work.join(format!("{stem}.toml")), format!("{MODEL}[method]
algorithm = \"pgas\"
backend   = \"chain_binomial\"
chains    = 2
particles = 50
sweeps    = {n}
burn_in   = 100
thin      = 1
n_trajectories = 4
")).unwrap();
    }
    camdl(&work, &["fit", "run", "base.toml", "--seed", "1"]);
    let base = leaf(&work, "base");
    camdl(&work, &["fit", "run", "longer.toml", "--seed", "1", "--resume", base.to_str().unwrap()]);
    let resumed = leaf(&work, "longer");
    camdl(&work, &["fit", "run", "fresh.toml", "--seed", "1"]);
    assert_whole_chain(&base, &resumed, &leaf(&work, "fresh"), "sweep", 2, n1, n2);

    // The saved latent paths: the base's file, then the new segment's, and
    // the manifest and the summary count both.
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(resumed.join("pgas_summary.json")).unwrap()).unwrap();
    for c in 0..2 {
        let tsv = |leaf: &Path| leaf.join(format!("chain_{c}/trajectories.tsv"));
        let parent = std::fs::read(tsv(&base)).unwrap();
        let whole = std::fs::read(tsv(&resumed)).unwrap();
        assert!(whole.len() > parent.len() && whole.starts_with(&parent),
            "chain {c}: trajectories.tsv extends the base's");
        let text = String::from_utf8(whole).unwrap();
        let mut draws: Vec<usize> = text.lines().skip(2)
            .map(|l| l.split('\t').nth(1).unwrap().parse().unwrap()).collect();
        draws.dedup();
        assert!(draws.windows(2).all(|w| w[0] < w[1]), "chain {c}: each path once, in order: {draws:?}");
        assert!(draws.iter().any(|&d| d < n1) && draws.iter().any(|&d| d >= n1), "{draws:?}");
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(resumed.join(format!("chain_{c}/trajectories.json"))).unwrap()).unwrap();
        assert_eq!(manifest["n_draws"].as_u64(), Some(draws.len() as u64), "chain {c}: manifest");
        assert_eq!(summary["trajectories"]["n_saved"][c].as_u64(), Some(draws.len() as u64),
            "chain {c}: the summary counts every saved path");
    }

    // The best sweep is the whole chain's: never below the base's.
    let best = |leaf: &Path| state_number(leaf, "best_loglik");
    assert!(best(&resumed) >= best(&base), "{} < {}", best(&resumed), best(&base));

    // The acceptance rate. The base records its best chain's rate only, so
    // when the resumed leaf's best chain is that chain its rate combines the
    // base's with the new segment's by count; the new segment's rate it
    // implies is then a rate, in [0, 1]. (The combination's arithmetic is
    // pinned exactly by the `count_weighted` unit test; the new segment's own
    // rate is not in any file.)
    let chain = state_number(&resumed, "best_chain") as usize;
    if state_number(&base, "best_chain") as usize == chain {
        let n_base = chain_draw_rows(&base, chain).len() as f64;
        let n_whole = chain_draw_rows(&resumed, chain).len() as f64;
        let implied = (state_number(&resumed, "acceptance_rate") * n_whole
            - state_number(&base, "acceptance_rate") * n_base) / (n_whole - n_base);
        assert!((-1e-9..=1.0 + 1e-9).contains(&implied),
            "the new segment's implied acceptance rate {implied} is not a rate");
    }
}
