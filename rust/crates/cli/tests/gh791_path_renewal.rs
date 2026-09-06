//! gh#791: `pgas_summary.json` carries the per-bin trajectory-renewal profile.
//!
//! `trajectory_renewal` is a weighted mean over ten time bins whose LATE terms
//! are high in most runs — the traceback's lineages have not yet coalesced when
//! it reaches the late states, so the tail of the path renews freely. A run
//! whose conditional-SMC genealogy has coalesced, so the EARLY path is held at
//! the reference, therefore still averages a healthy-looking third. The per-bin
//! data already existed in `trace.tsv` (`renewal_b0 … renewal_b9`, gh#688);
//! nothing summarised it.
//!
//! The in-crate unit tests pin what the two derived numbers MEAN. This pins
//! what a downstream reader (`camdl-scope`) actually finds on disk:
//!
//! * the `path_renewal` block exists for a PGAS stage, with the documented
//!   keys, and states what a bin spans — `bins[0] = 0.04` is uninterpretable
//!   without that, and today the fact lives only in a source comment;
//! * the published profile IS the mean down the `renewal_b<n>` columns of the
//!   chains' `trace.tsv`, over the retained post-burn-in sweeps. Without this
//!   the block could drift from the sampler's own record and nothing would say
//!   so;
//! * `prefix` and `gradient` are the stated functions of `bins`, recomputable
//!   by a consumer holding only the JSON;
//! * the ancestor-sampling acceptance rate is in the SAME block — the two are
//!   only legible read side by side;
//! * everything the summary carried before is still there, spelled the same.
//!   The block is additive; `camdl-scope` reads this file.
//!
//! Skipped when the release binary or camdlc isn't present.

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
        "camdl_gh791_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Bins the profile is resolved into. The same ten as `RENEWAL_BINS`, spelled
/// out here rather than imported: this test speaks for a downstream reader who
/// has only the JSON, and a reader who imported the constant could not catch a
/// change to it.
const N_BINS: usize = 10;

/// 40 days at dt = 1, so every one of the ten bins holds four substeps and the
/// profile has no `null` in it — the case where the check below is strongest.
const T_END: usize = 40;
const SWEEPS: usize = 40;
const BURN_IN: usize = 10;
const CHAINS: usize = 2;

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = {T_END} 'days }}
"#);
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    // A single epidemic wave, one row per day.
    let mut data = String::from("time\tcases\n");
    for t in 1..=T_END {
        let x = t as f64;
        let n = (60.0 * (-((x - 18.0) / 9.0).powi(2)).exp()).round() as i64 + 4;
        data.push_str(&format!("{t}\t{n}\n"));
    }
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path, data).unwrap();

    (ir_path, data_path)
}

/// `thin = 1`, so the sweeps the summary averages over are exactly the trace
/// rows with `sweep >= BURN_IN` — which is what makes the trace recomputation
/// below an independent check rather than a restatement.
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path) -> (PathBuf, PathBuf) {
    let out = dir.join("results");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
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
chains = {CHAINS}
particles = 40
sweeps = {SWEEPS}
burn_in = {BURN_IN}
thin = 1
"#,
        out = out.display(), ir = ir.display(), data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    (p, out)
}

fn stage_leaf(out: &Path) -> PathBuf {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return d;
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

/// One chain's `trace.tsv` as a header plus one row of fields per sweep.
fn trace(stage: &Path, chain_1based: usize) -> (Vec<String>, Vec<Vec<String>>) {
    let path = stage.join(format!("chain_{chain_1based}/trace.tsv"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> =
        lines.next().expect("trace header").split('\t').map(String::from).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split('\t').map(String::from).collect::<Vec<_>>())
        .collect();
    assert!(!rows.is_empty(), "chain {chain_1based} wrote no trace rows");
    for r in &rows {
        assert_eq!(r.len(), header.len(), "trace row width must match the header");
    }
    (header, rows)
}

fn col<'a>(header: &[String], row: &'a [String], name: &str) -> &'a str {
    let i = header.iter().position(|c| c == name)
        .unwrap_or_else(|| panic!("trace.tsv has no `{name}` column; header: {header:?}"));
    &row[i]
}

/// The mean down each `renewal_b<n>` column and down `trajectory_renewal`,
/// over every chain's retained post-burn-in sweeps. `NA` — a bin holding no
/// substep on that sweep — is skipped, never counted as a renewal of zero.
///
/// Returns `(per-bin means, aggregate mean, sweeps counted)`.
fn profile_from_traces(stage: &Path) -> ([Option<f64>; N_BINS], f64, usize) {
    let mut sum = [0.0f64; N_BINS];
    let mut n = [0usize; N_BINS];
    let mut agg_sum = 0.0;
    let mut n_sweeps = 0usize;
    for c in 1..=CHAINS {
        let (header, rows) = trace(stage, c);
        for row in &rows {
            let sweep: usize = col(&header, row, "sweep").parse().expect("sweep parses");
            // thin = 1, so every post-burn-in sweep is a retained draw.
            if sweep < BURN_IN { continue; }
            n_sweeps += 1;
            agg_sum += col(&header, row, "trajectory_renewal")
                .parse::<f64>().expect("trajectory_renewal parses");
            for (b, s) in sum.iter_mut().enumerate() {
                let v = col(&header, row, &format!("renewal_b{b}"));
                if v == "NA" { continue; }
                *s += v.parse::<f64>()
                    .unwrap_or_else(|e| panic!("renewal_b{b} = {v:?} ({e})"));
                n[b] += 1;
            }
        }
    }
    assert!(n_sweeps > 0, "no retained post-burn-in sweep in any chain's trace");
    let mut bins = [None; N_BINS];
    for (b, slot) in bins.iter_mut().enumerate() {
        if n[b] > 0 { *slot = Some(sum[b] / n[b] as f64); }
    }
    (bins, agg_sum / n_sweeps as f64, n_sweeps)
}

/// gh#864: the per-sweep ancestor-weight ESS columns, over the same retained
/// post-burn-in sweeps, `NA` skipped. Returns `(pre-mask, post-mask)`.
///
/// `NA` is a sweep with no ancestor-sampling step to measure — dropped here,
/// exactly as the summary drops it, and never read as an ESS of zero.
fn ancestor_ess_from_traces(stage: &Path) -> (Vec<f64>, Vec<f64>) {
    let mut pre = Vec::new();
    let mut post = Vec::new();
    for c in 1..=CHAINS {
        let (header, rows) = trace(stage, c);
        for row in &rows {
            let sweep: usize = col(&header, row, "sweep").parse().expect("sweep parses");
            if sweep < BURN_IN { continue; }
            for (name, sink) in [("as_ess_pre", &mut pre), ("as_ess_post", &mut post)] {
                let v = col(&header, row, name);
                if v == "NA" { continue; }
                sink.push(v.parse::<f64>()
                    .unwrap_or_else(|e| panic!("{name} = {v:?} ({e})")));
            }
        }
    }
    (pre, post)
}

/// Median of a sample, or `None` when it is empty — the same convention the
/// summary uses, recomputed here so the published number is checked against
/// the sampler's own record rather than restated.
fn median_of(mut xs: Vec<f64>) -> Option<f64> {
    if xs.is_empty() { return None; }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    Some(if n % 2 == 0 { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 } else { xs[n / 2] })
}

#[test]
fn pgas_summary_carries_the_per_bin_renewal_profile() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("profile");
    let (ir, data) = write_fixture(tmp.path());
    let (fit, out) = write_fit_toml(tmp.path(), &ir, &data);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit).arg("--seed").arg("11")
        .output().expect("spawn camdl");
    // A coalesced profile is a DIAGNOSTIC and must never fail a run.
    assert!(r.status.success(),
        "fit run failed: {}", String::from_utf8_lossy(&r.stderr));

    let stage = stage_leaf(&out);
    let summary_path = stage.join("pgas_summary.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&summary_path).unwrap())
            .expect("parse pgas_summary.json");

    // ── Nothing that was already there moved (camdl-scope reads this) ─────
    for key in ["stage", "n_chains", "acceptance_rates", "thin", "trajectories",
                "rhat", "ess", "ess_tail"] {
        assert!(v.get(key).is_some(),
            "gh#791 is additive: `{key}` must still be in pgas_summary.json; \
             keys are {:?}", v.as_object().map(|o| o.keys().collect::<Vec<_>>()));
    }
    assert_eq!(v["stage"], serde_json::json!("pgas"));
    assert_eq!(v["n_chains"], serde_json::json!(CHAINS));

    // ── The block, and what it says about itself ──────────────────────────
    let pr = v.get("path_renewal").unwrap_or_else(|| panic!(
        "a PGAS stage must write a `path_renewal` block; keys are {:?}",
        v.as_object().map(|o| o.keys().collect::<Vec<_>>())));
    assert_eq!(pr["n_bins"], serde_json::json!(N_BINS));
    assert_eq!(pr["n_prefix_bins"], serde_json::json!(N_BINS / 2));
    let span = pr["bin_span"].as_str().expect("bin_span is a string");
    assert!(span.contains("[b/10, (b+1)/10)"),
        "the artifact must say what a bin spans — `bins[0] = 0.04` means nothing \
         without it; got {span:?}");

    let bins: Vec<Option<f64>> =
        serde_json::from_value(pr["bins"].clone()).expect("bins");
    assert_eq!(bins.len(), N_BINS, "one entry per bin");
    for (b, v) in bins.iter().enumerate() {
        if let Some(x) = v {
            assert!((0.0..=1.0).contains(x),
                "bin {b} is a fraction of substeps and must lie in [0,1]; got {x}");
        }
    }

    // ── The block IS the mean down the trace columns ──────────────────────
    // The trace writes four decimals, so the tolerance is the rounding.
    let (from_trace, agg_from_trace, n_sweeps) = profile_from_traces(&stage);
    assert_eq!(pr["n_sweeps"].as_u64(), Some(n_sweeps as u64),
        "the block must say how many retained sweeps it averaged");
    assert_eq!(pr["n_chains"].as_u64(), Some(CHAINS as u64));
    for b in 0..N_BINS {
        match (bins[b], from_trace[b]) {
            (Some(published), Some(recomputed)) => assert!(
                (published - recomputed).abs() < 1e-4,
                "bin {b}: the summary publishes {published}, but the mean of the \
                 `renewal_b{b}` column over the {n_sweeps} retained sweeps of the \
                 chains' trace.tsv is {recomputed} — the block has drifted from \
                 the sampler's own record"),
            (None, None) => {}
            (p, t) => panic!(
                "bin {b}: summary says {p:?} while the traces say {t:?}; a bin is \
                 null exactly when no retained sweep recorded a substep in it"),
        }
    }
    let aggregate = pr["aggregate"].as_f64().expect("aggregate");
    assert!((aggregate - agg_from_trace).abs() < 1e-4,
        "the aggregate in the block must be the mean of the `trajectory_renewal` \
         column over the same sweeps: {aggregate} vs {agg_from_trace}");

    // ── The derived numbers are the stated functions of `bins` ────────────
    let observed: Vec<f64> = bins[..N_BINS / 2].iter().flatten().copied().collect();
    let expect_prefix = (!observed.is_empty())
        .then(|| observed.iter().sum::<f64>() / observed.len() as f64);
    let close = |got: Option<f64>, want: Option<f64>| match (got, want) {
        (Some(a), Some(b)) => (a - b).abs() < 1e-12,
        (None, None) => true,
        _ => false,
    };
    assert!(close(pr["prefix"].as_f64(), expect_prefix),
        "prefix must be the mean over the observed bins of the first half: \
         published {:?}, recomputed from `bins` {expect_prefix:?}", pr["prefix"]);
    let expect_gradient = match (bins[N_BINS - 1], bins[0]) {
        (Some(l), Some(f)) => Some(l - f),
        _ => None,
    };
    assert!(close(pr["gradient"].as_f64(), expect_gradient),
        "gradient must be the LAST bin minus the FIRST: published {:?}, \
         recomputed from `bins` {expect_gradient:?}", pr["gradient"]);

    // ── Ancestor sampling is reported beside it, not elsewhere ────────────
    let n_prop = pr["n_as_proposed"].as_u64().expect("n_as_proposed");
    let n_acc = pr["n_as_accepted"].as_u64().expect("n_as_accepted");
    match pr["as_accept"].as_f64() {
        Some(rate) => {
            assert!(n_prop > 0, "an acceptance rate needs a denominator");
            assert!((rate - n_acc as f64 / n_prop as f64).abs() < 1e-12,
                "as_accept must be the pooled n_as_accepted / n_as_proposed");
        }
        None => assert_eq!(n_prop, 0,
            "as_accept is null only when the Metropolis step never ran — which is \
             a different diagnosis from an acceptance rate of 0"),
    }

    // ── The ancestor weights' ESS, both sides of the mask (gh#864) ────────
    // A count of admissible candidates is not a count of choices: the ancestor
    // index is drawn from a categorical over those weights, and one dominant
    // weight leaves the move with one real option however many are admissible.
    // Both sides are published because the guard can lower the count and raise
    // the ESS, by removing a dominant candidate whose splice was infeasible.
    let (ess_pre, ess_post) = ancestor_ess_from_traces(&stage);
    for (key, from_trace) in [("as_ess_pre", &ess_pre), ("as_ess_post", &ess_post)] {
        let block = &pr[key];
        assert_eq!(block["n_sweeps"].as_u64(), Some(from_trace.len() as u64),
            "{key} must say how many retained sweeps measured an ESS — the \
             non-`NA` rows of its own trace column ({} of {n_sweeps})",
            from_trace.len());
        match (block["median"].as_f64(), median_of(from_trace.clone())) {
            (Some(published), Some(recomputed)) => {
                // The column prints two decimals, so the recomputation agrees
                // to that rather than exactly.
                assert!((published - recomputed).abs() < 0.01,
                    "{key} must be the median of its own trace column over the \
                     retained sweeps: published {published}, recomputed \
                     {recomputed}");
                assert!((1.0..=40.0).contains(&published),
                    "{key} is an effective particle count, so it lies in \
                     [1, particles = 40]: {published}");
            }
            (None, None) => {}
            (p, t) => panic!(
                "{key}: summary says {p:?} while the traces say {t:?}; the \
                 median is null exactly when no retained sweep measured one"),
        }
    }

    // ── The aggregate is not replaced ─────────────────────────────────────
    // `trajectory_renewal` is what downstream tools read today, and it stays
    // where they read it.
    let (header, _) = trace(&stage, 1);
    for c in ["trajectory_renewal", "renewal_b0", "renewal_b9", "as_accept",
              "as_ess_pre", "as_ess_post"] {
        assert!(header.iter().any(|h| h == c),
            "trace.tsv must keep the `{c}` column; header was {header:?}");
    }

    // ── The profile reaches the terminal, beside the aggregate ────────────
    let stderr = String::from_utf8_lossy(&r.stderr);
    assert!(stderr.contains("path renewal"),
        "the end-of-stage output must print the profile; stderr was:\n{stderr}");
    assert!(stderr.contains("gradient") && stderr.contains("prefix"),
        "with both derived numbers; stderr was:\n{stderr}");
    assert!(stderr.contains("ancestor-sampling acceptance"),
        "and the ancestor-sampling acceptance rate beside them; stderr was:\n{stderr}");
    // gh#864: the acceptance rate on its own leaves the reader unable to tell a
    // move that was rejected from a move that had nothing to choose among.
    assert!(stderr.contains("ancestor-weight ESS"),
        "and the ancestor-weight ESS beside the acceptance rate it qualifies; \
         stderr was:\n{stderr}");
    assert!(stderr.contains("tenth"),
        "and it must say what a bin spans wherever it prints one; stderr was:\n{stderr}");
}
