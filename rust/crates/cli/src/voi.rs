//! `camdl voi run` — Expected Value of Sample Information (EVSI).
//!
//! Reads a `voi.toml` that references a completed design experiment and
//! computes EVSI for each defined study via preposterior importance sampling
//! (Raiffa & Schlaifer 1961; Strong et al. 2014).
//!
//! No new simulations are run. All computation is O(N × M × |A|) where
//! N = prior samples, M = MC draws (~1000), |A| = number of actions.
//!
//! Pipeline:
//!   camdl simulate batch sweep.toml    ← runs the design (gh #2)
//!   camdl voi run voi.toml             ← reads outputs.tsv, writes evsi.tsv

use std::collections::HashMap;
use serde::Deserialize;

// ─── TOML deserialization types ───────────────────────────────────────────────

#[derive(Deserialize)]
struct VoiFile {
    voi:      VoiBlock,
    decision: DecisionBlock,
    #[serde(default)]
    study:    HashMap<String, StudyBlock>,
}

#[derive(Deserialize)]
struct VoiBlock {
    batch:  String,   // path to the simulate-batch sweep.toml
    design: String,   // which design's outputs.tsv to use
}

#[derive(Deserialize)]
struct DecisionBlock {
    actions:   Vec<String>,   // scenario names
    reference: String,        // baseline scenario
    utility:   String,        // column name in outputs.tsv
    #[serde(default = "default_direction")]
    direction: String,        // "minimize" | "maximize"
    #[serde(default)]
    cost:      HashMap<String, f64>,  // action -> cost in utility units (subtracted from U)
}

fn default_direction() -> String { "minimize".to_string() }

#[derive(Deserialize)]
struct StudyBlock {
    parameter:      String,        // which parameter the study informs
    likelihood:     String,        // "binomial" | "log_normal" | "normal"
    #[serde(default)]
    sample_sizes:   Vec<usize>,    // for discrete studies
    #[serde(default)]
    observation_sd: Vec<f64>,      // for continuous studies (sd in observation space)
}

// ─── Domain types ─────────────────────────────────────────────────────────────

struct EvsiRow {
    study:          String,
    size:           f64,    // sample_size or obs_sd depending on study type
    evsi:           f64,
    evsi_se:        f64,
    current_eu:     f64,
    optimal_action: String,
    p_switch:       f64,
    switch_label:   String,
    min_ess:        f64,
    ess_warning:    bool,
}

// ─── Simple LCG RNG ──────────────────────────────────────────────────────────
//
// Knuth's MMIX LCG: multiplier 6364136223846793005, addend 1442695040888963407.
// Period 2^64. Adequate for M=1000 MC draws per EVSI estimate.
// seed+1 in new() avoids the degenerate fixed-point at state=0.

struct Lcg { state: u64 }

impl Lcg {
    fn new(seed: u64) -> Self { Lcg { state: seed.wrapping_add(1) } }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller normal sample.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Binomial(n, p) sample.
    /// Uses normal approximation only when n > 30 AND np >= 5 AND n(1-p) >= 5.
    /// Falls back to exact Bernoulli trials otherwise (handles small-p edge cases).
    fn binomial(&mut self, n: usize, p: f64) -> usize {
        if p <= 0.0 { return 0; }
        if p >= 1.0 { return n; }
        let np = n as f64 * p;
        if n > 30 && np >= 5.0 && n as f64 * (1.0 - p) >= 5.0 {
            let sigma = (np * (1.0 - p)).sqrt();
            ((np + sigma * self.normal()).round() as isize).max(0).min(n as isize) as usize
        } else {
            (0..n).filter(|_| self.next_f64() < p).count()
        }
    }
}

// ─── Study likelihood / data simulation ──────────────────────────────────────

/// Simulate one study observation under true parameter `theta`, at the given
/// size (n for discrete, sigma for continuous).
fn simulate_data(rng: &mut Lcg, likelihood: &str, theta: f64, size: f64) -> f64 {
    match likelihood {
        "binomial" | "beta_binomial" => {
            rng.binomial(size as usize, theta.clamp(0.0, 1.0)) as f64
        }
        "log_normal" => {
            let sigma = size;
            (theta.max(1e-300).ln() + sigma * rng.normal()).exp()
        }
        _ => { // "normal"
            theta + size * rng.normal()
        }
    }
}

/// Log-likelihood p(d | theta, size).  Constants that cancel in normalisation omitted.
fn log_lik(likelihood: &str, d: f64, theta: f64, size: f64) -> f64 {
    match likelihood {
        "binomial" | "beta_binomial" => {
            let k = d;
            let n = size;
            let t = theta.clamp(1e-10, 1.0 - 1e-10);
            k * t.ln() + (n - k) * (1.0 - t).ln()
        }
        "log_normal" => {
            let sigma = size;
            let diff  = d.max(1e-300).ln() - theta.max(1e-300).ln();
            -0.5 * diff * diff / (sigma * sigma)
        }
        _ => { // "normal"
            let sigma = size;
            let diff  = d - theta;
            -0.5 * diff * diff / (sigma * sigma)
        }
    }
}

// ─── Core EVSI computation ────────────────────────────────────────────────────

/// Compute EVSI for one (study, size) combination.
///
/// `theta_vals`   : N values of the study parameter, one per prior sample.
/// `utility`      : N × |actions| matrix — utility[i][a] = U(action_a | θᵢ).
/// `actions`      : action names, matching utility columns.
/// `likelihood`   : study likelihood model.
/// `size`         : sample size (binomial) or obs_sd (continuous).
/// `mc_draws`     : number of Monte Carlo draws over study outcomes.
fn compute_evsi(
    theta_vals: &[f64],
    utility:    &[Vec<f64>],
    actions:    &[String],
    likelihood: &str,
    size:       f64,
    mc_draws:   usize,
    rng:        &mut Lcg,
) -> EvsiRow {
    let n         = theta_vals.len();
    let n_actions = actions.len();

    // ── Prior expected utilities ──────────────────────────────────────────────
    let eu_prior: Vec<f64> = (0..n_actions).map(|a| {
        utility.iter().map(|row| row[a]).sum::<f64>() / n as f64
    }).collect();

    let (best_a, current_eu) = eu_prior.iter().enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    let optimal_action = actions[best_a].clone();

    // ── Monte Carlo over study outcomes ───────────────────────────────────────
    let mut v_j: Vec<f64> = Vec::with_capacity(mc_draws);
    let mut switches: Vec<(usize, usize)> = Vec::new();  // (from, to) action indices
    let mut min_ess = f64::INFINITY;

    for j in 0..mc_draws {
        // "True" parameter drawn by cycling through prior samples
        let true_theta = theta_vals[j % n];

        // Simulate study data
        let d = simulate_data(rng, likelihood, true_theta, size);

        // Log-importance weights
        let lw_max = theta_vals.iter()
            .map(|&t| log_lik(likelihood, d, t, size))
            .fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = theta_vals.iter()
            .map(|&t| (log_lik(likelihood, d, t, size) - lw_max).exp())
            .collect();
        let w_sum: f64 = w.iter().sum();

        // ESS = 1 / Σ(wᵢ/w_sum)²
        let ess = {
            let sq: f64 = w.iter().map(|wi| (wi / w_sum).powi(2)).sum();
            if sq > 0.0 { 1.0 / sq } else { 0.0 }
        };
        if ess < min_ess { min_ess = ess; }

        // Posterior-weighted expected utility
        let eu_post: Vec<f64> = (0..n_actions).map(|a| {
            w.iter().zip(utility.iter()).map(|(wi, row)| wi * row[a]).sum::<f64>() / w_sum
        }).collect();

        let (best_post, v) = eu_post.iter().enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &val)| {
                if val > bv { (i, val) } else { (bi, bv) }
            });
        v_j.push(v);
        if best_post != best_a {
            switches.push((best_a, best_post));
        }
    }

    // ── Aggregate ─────────────────────────────────────────────────────────────
    let v_mean: f64 = v_j.iter().sum::<f64>() / mc_draws as f64;
    let v_var:  f64 = v_j.iter().map(|v| (v - v_mean).powi(2)).sum::<f64>() / mc_draws as f64;
    let evsi        = v_mean - current_eu;
    let evsi_se     = v_var.sqrt() / (mc_draws as f64).sqrt();

    let p_switch = switches.len() as f64 / mc_draws as f64;

    // Most common switch label
    let mut switch_counts: HashMap<String, usize> = HashMap::new();
    for (from, to) in &switches {
        let label = format!("{} → {}", actions[*from], actions[*to]);
        *switch_counts.entry(label).or_insert(0) += 1;
    }
    let switch_label = switch_counts.into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(s, _)| s)
        .unwrap_or_else(|| format!("{} → {}", optimal_action, optimal_action));

    let ess_warning = min_ess < (n as f64 / 5.0);

    EvsiRow {
        study: String::new(), // filled by caller
        size,
        evsi,
        evsi_se,
        current_eu,
        optimal_action,
        p_switch,
        switch_label,
        min_ess,
        ess_warning,
    }
}

// ─── TSV helpers ──────────────────────────────────────────────────────────────

/// Read outputs.tsv (full-granularity format from `camdl simulate batch`).
/// Returns: (col_names, rows) where each row is (point_id, scenario, seed, vals).
fn read_outputs_tsv(path: &str)
    -> Result<(Vec<String>, Vec<(usize, String, u64, Vec<f64>)>), String>
{
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = src.lines();
    let header_line = lines.next().ok_or("empty outputs.tsv")?;
    let header: Vec<&str> = header_line.split('\t').collect();

    let pid_idx  = header.iter().position(|h| *h == "point_id")
        .ok_or("outputs.tsv missing 'point_id' column")?;
    let scen_idx = header.iter().position(|h| *h == "scenario")
        .ok_or("outputs.tsv missing 'scenario' column — run 'camdl simulate batch' first")?;
    let seed_idx = header.iter().position(|h| *h == "seed")
        .ok_or("outputs.tsv missing 'seed' column — run 'camdl simulate batch' first")?;

    const META: &[&str] = &["point_id", "scenario", "seed"];
    let data_indices: Vec<usize> = header.iter().enumerate()
        .filter(|(_, h)| !META.contains(h))
        .map(|(i, _)| i)
        .collect();
    let col_names: Vec<String> = header.iter().enumerate()
        .filter(|(_, h)| !META.contains(h))
        .map(|(_, h)| h.to_string())
        .collect();

    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        let point_id = parts.get(pid_idx).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let scenario  = parts.get(scen_idx).copied().unwrap_or("").to_string();
        let seed      = parts.get(seed_idx).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let vals: Vec<f64> = data_indices.iter()
            .map(|&i| parts.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN))
            .collect();
        rows.push((point_id, scenario, seed, vals));
    }
    Ok((col_names, rows))
}

/// Read parameter_points.tsv.
/// Returns: (param_names, rows) where each row is (point_id, vals).
fn read_param_points(path: &str) -> Result<(Vec<String>, Vec<(usize, Vec<f64>)>), String> {
    let src = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = src.lines();
    let header_line = lines.next().ok_or("empty parameter_points.tsv")?;
    let header: Vec<&str> = header_line.split('\t').collect();
    let pid_idx = header.iter().position(|h| *h == "point_id").unwrap_or(0);
    let param_names: Vec<String> = header.iter().enumerate()
        .filter(|(i, _)| *i != pid_idx)
        .map(|(_, h)| h.to_string())
        .collect();
    let data_idx: Vec<usize> = (0..header.len()).filter(|i| *i != pid_idx).collect();

    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        let pid = parts.get(pid_idx).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let vals: Vec<f64> = data_idx.iter()
            .map(|&i| parts.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN))
            .collect();
        rows.push((pid, vals));
    }
    Ok((param_names, rows))
}

// ─── Build utility matrix ─────────────────────────────────────────────────────

/// Build U(action, θᵢ) matrix from outputs.tsv.
///
/// Groups rows by (point_id, scenario), averages over seeds.
/// Then computes U(a, θᵢ) = Ū(reference, θᵢ) - Ū(action, θᵢ)  [minimize]
///                           Ū(action, θᵢ) - Ū(reference, θᵢ)  [maximize]
///
/// Returns (N, utility_matrix) where utility_matrix[i][a] = U(action_a | θᵢ).
fn build_utility_matrix(
    col_names:  &[String],
    rows:       &[(usize, String, u64, Vec<f64>)],
    actions:    &[String],
    reference:  &str,
    utility:    &str,
    direction:  &str,
    n_points:   usize,
    cost:       &HashMap<String, f64>,
) -> Result<Vec<Vec<f64>>, String> {
    let util_col = col_names.iter().position(|c| c == utility)
        .ok_or_else(|| format!("utility column '{}' not found in outputs.tsv. Available: {}",
            utility, col_names.join(", ")))?;

    // Group by (point_id, scenario), sum over seeds + count
    let mut sums: HashMap<(usize, String), (f64, usize)> = HashMap::new();
    for (pid, scen, _, vals) in rows {
        let entry = sums.entry((*pid, scen.clone())).or_insert((0.0, 0));
        if let Some(&v) = vals.get(util_col) {
            entry.0 += v;
            entry.1 += 1;
        }
    }

    // Average over seeds
    let avg: HashMap<(usize, &str), f64> = sums.iter()
        .map(|((pid, scen), (sum, count))| {
            ((*pid, scen.as_str()), sum / (*count as f64).max(1.0))
        })
        .collect();

    // Check that reference scenario exists
    let has_reference = avg.keys().any(|(_, s)| *s == reference);
    if !has_reference {
        let scenarios: Vec<&str> = {
            let mut s: Vec<&str> = avg.keys().map(|(_, s)| *s).collect();
            s.sort();
            s.dedup();
            s
        };
        return Err(format!(
            "reference scenario '{}' not found in outputs.tsv. Available: {}",
            reference, scenarios.join(", ")
        ));
    }

    // Build utility matrix: utility[point_id][action_idx]
    let mut utility_matrix: Vec<Vec<f64>> = vec![vec![0.0; actions.len()]; n_points];
    for (i, action) in actions.iter().enumerate() {
        let action_cost = cost.get(action.as_str()).copied().unwrap_or(0.0);
        for pid in 0..n_points {
            let ref_val = avg.get(&(pid, reference)).copied().unwrap_or(f64::NAN);
            let act_val = avg.get(&(pid, action.as_str())).copied().unwrap_or(f64::NAN);
            let raw = if direction == "maximize" {
                act_val - ref_val
            } else {
                ref_val - act_val
            };
            utility_matrix[pid][i] = raw - action_cost;
        }
    }
    Ok(utility_matrix)
}

// ─── Output writers ───────────────────────────────────────────────────────────

fn write_evsi_tsv(path: &str, rows: &[EvsiRow]) {
    let mut tsv = String::from(
        "study\tsample_size\tEVSI\tEVSI_se\tcurrent_EU\toptimal_action_current\tess_warning\n"
    );
    for r in rows {
        tsv.push_str(&format!("{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\n",
            r.study, r.size, r.evsi, r.evsi_se, r.current_eu,
            r.optimal_action, if r.ess_warning { "yes" } else { "no" }));
    }
    let _ = std::fs::write(path, &tsv);
}

fn write_action_sensitivity_tsv(path: &str, rows: &[EvsiRow]) {
    let mut tsv = String::from("study\tsample_size\tP_switch\tmost_common_switch\n");
    for r in rows {
        tsv.push_str(&format!("{}\t{}\t{:.4}\t{}\n",
            r.study, r.size, r.p_switch, r.switch_label));
    }
    let _ = std::fs::write(path, &tsv);
}

fn write_diminishing_returns_tsv(path: &str, rows: &[EvsiRow]) {
    // Group by study, compute marginal EVSI between consecutive sizes
    let mut tsv = String::from("study\tsample_size\tEVSI\tmarginal_EVSI\n");
    let mut study_rows: HashMap<&str, Vec<&EvsiRow>> = HashMap::new();
    for r in rows { study_rows.entry(&r.study).or_default().push(r); }
    let mut study_names: Vec<&str> = study_rows.keys().copied().collect();
    study_names.sort();
    for name in study_names {
        let sr = &study_rows[name];
        for (i, r) in sr.iter().enumerate() {
            let marginal = if i == 0 {
                r.evsi
            } else {
                r.evsi - sr[i-1].evsi
            };
            tsv.push_str(&format!("{}\t{}\t{:.4}\t{:.4}\n",
                r.study, r.size, r.evsi, marginal));
        }
    }
    let _ = std::fs::write(path, &tsv);
}

fn write_assumptions_txt(path: &str, voi: &VoiFile, n: usize, mc_draws: usize, rows: &[EvsiRow]) {
    let mut txt = String::from("VOI analysis assumptions:\n\n");
    txt.push_str("Method:\n");
    txt.push_str("  Preposterior importance sampling (Raiffa & Schlaifer 1961;\n");
    txt.push_str("  Strong et al. 2014 for health economics)\n\n");
    txt.push_str(&format!("Prior:\n  N = {} samples from design '{}'\n\n",
        n, voi.voi.design));
    txt.push_str("Decision:\n");
    txt.push_str(&format!("  Actions: {}\n", voi.decision.actions.join(", ")));
    txt.push_str(&format!("  Reference: {}\n", voi.decision.reference));
    txt.push_str(&format!("  Utility: {} ({})\n", voi.decision.utility, voi.decision.direction));
    if !voi.decision.cost.is_empty() {
        let mut costs: Vec<(&String, &f64)> = voi.decision.cost.iter().collect();
        costs.sort_by(|(a, _), (b, _)| a.cmp(b));
        txt.push_str("  Costs (utility units subtracted per action):\n");
        for (action, c) in &costs {
            txt.push_str(&format!("    {}: {}\n", action, c));
        }
    }
    txt.push('\n');
    txt.push_str(&format!("Monte Carlo draws per study configuration: M = {}\n\n", mc_draws));
    txt.push_str("Studies:\n");
    let mut study_names: Vec<&String> = voi.study.keys().collect();
    study_names.sort();
    for name in &study_names {
        let s = &voi.study[*name];
        let sizes = if !s.sample_sizes.is_empty() {
            s.sample_sizes.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
        } else {
            s.observation_sd.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", ")
        };
        txt.push_str(&format!("  {}: {}({}), sizes/sds: {}\n",
            name, s.likelihood, s.parameter, sizes));
    }
    txt.push('\n');
    let warns: Vec<&EvsiRow> = rows.iter().filter(|r| r.ess_warning).collect();
    if warns.is_empty() {
        txt.push_str("Effective sample size: all configurations had ESS > N/5 (reliable)\n");
    } else {
        txt.push_str("ESS WARNINGS (importance sampling may be unreliable):\n");
        for r in &warns {
            txt.push_str(&format!("  {} n={}: min ESS = {:.0}\n", r.study, r.size, r.min_ess));
        }
    }
    txt.push_str("\nLimitations:\n");
    txt.push_str("  - Each study is assumed to inform exactly one parameter\n");
    txt.push_str("  - Prior is approximated by the design sample, not an analytic form\n");
    txt.push_str("  - Importance sampling degrades when the study is very informative\n");
    let _ = std::fs::write(path, &txt);
}

// ─── CLI entry point ──────────────────────────────────────────────────────────

pub fn cmd_voi_run(args: &[String]) {
    let mut toml_path: Option<String> = None;
    let mut output_override: Option<String> = None;
    let mut mc_draws: usize = 1000;
    let mut rng_seed: u64 = 42;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output"   => { i += 1; output_override = Some(args[i].clone()); }
            "--mc-draws" => { i += 1; mc_draws = args[i].parse().unwrap_or(1000); }
            "--seed"     => { i += 1; rng_seed  = args[i].parse().unwrap_or(42); }
            s if s.starts_with("--") => {
                eprintln!("unknown flag: {}", s); voi_usage();
            }
            p => { toml_path = Some(p.to_string()); }
        }
        i += 1;
    }

    let toml_path = toml_path.unwrap_or_else(|| { voi_usage(); });
    let toml_src = std::fs::read_to_string(&toml_path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", toml_path, e);
        std::process::exit(1);
    });
    let voi_file: VoiFile = toml::from_str(&toml_src).unwrap_or_else(|e| {
        eprintln!("error: voi.toml parse error: {}", e);
        std::process::exit(1);
    });

    // Resolve batch output_dir
    let batch_toml_src = std::fs::read_to_string(&voi_file.voi.batch).unwrap_or_else(|e| {
        eprintln!("error: cannot read batch '{}': {}", voi_file.voi.batch, e);
        std::process::exit(1);
    });
    let batch_info = crate::util::parse_batch_toml(&batch_toml_src).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let batch_output_dir = batch_info.output_dir;
    let design_dir = format!("{}/designs/{}", batch_output_dir, voi_file.voi.design);

    // Output directory
    let out_dir = output_override.unwrap_or_else(|| {
        format!("{}/analysis/voi", batch_output_dir)
    });
    std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        eprintln!("error: cannot create {}: {}", out_dir, e);
        std::process::exit(1);
    });

    // Load parameter_points.tsv
    let pts_path = format!("{}/parameter_points.tsv", design_dir);
    let (param_names, param_rows) = read_param_points(&pts_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let n_points = param_rows.iter().map(|(pid, _)| *pid).max().map(|m| m + 1).unwrap_or(0);
    eprintln!("Loaded {} parameter points, {} parameters", n_points, param_names.len());

    // Load outputs.tsv
    let outputs_path = format!("{}/outputs.tsv", design_dir);
    let (col_names, output_rows) = read_outputs_tsv(&outputs_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        eprintln!("  Run 'camdl simulate batch {}' first.", voi_file.voi.batch);
        std::process::exit(1);
    });

    // Build utility matrix
    let utility_matrix = build_utility_matrix(
        &col_names, &output_rows,
        &voi_file.decision.actions,
        &voi_file.decision.reference,
        &voi_file.decision.utility,
        &voi_file.decision.direction,
        n_points,
        &voi_file.decision.cost,
    ).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // Build param value lookup: param_name -> Vec<f64> indexed by point_id
    let mut param_by_name: HashMap<String, Vec<f64>> = HashMap::new();
    for name in &param_names {
        param_by_name.insert(name.clone(), vec![0.0; n_points]);
    }
    for (pid, vals) in &param_rows {
        for (j, name) in param_names.iter().enumerate() {
            if let Some(pv) = param_by_name.get_mut(name) {
                if let Some(v) = vals.get(j) {
                    if *pid < pv.len() { pv[*pid] = *v; }
                }
            }
        }
    }

    // Run EVSI for all studies
    let mut rng = Lcg::new(rng_seed);
    let mut all_rows: Vec<EvsiRow> = Vec::new();

    let mut study_names: Vec<&String> = voi_file.study.keys().collect();
    study_names.sort();

    for study_name in &study_names {
        let study = &voi_file.study[*study_name];

        let theta_vals = param_by_name.get(&study.parameter).cloned()
            .unwrap_or_else(|| {
                eprintln!("error: study parameter '{}' not found in parameter_points.tsv",
                    study.parameter);
                eprintln!("  Available: {}", param_names.join(", "));
                std::process::exit(1);
            });

        // Enumerate study sizes
        let sizes: Vec<f64> = if !study.sample_sizes.is_empty() {
            study.sample_sizes.iter().map(|&n| n as f64).collect()
        } else {
            study.observation_sd.clone()
        };

        for &size in &sizes {
            eprint!("  {} n={} ...", study_name, size);
            let mut row = compute_evsi(
                &theta_vals,
                &utility_matrix,
                &voi_file.decision.actions,
                &study.likelihood,
                size,
                mc_draws,
                &mut rng,
            );
            row.study = (*study_name).clone();
            if row.ess_warning {
                eprint!(" [ESS warning: min={:.0}]", row.min_ess);
            }
            eprintln!(" EVSI={:.3} (se={:.3})", row.evsi, row.evsi_se);
            all_rows.push(row);
        }
    }

    // Write outputs
    write_evsi_tsv(&format!("{}/evsi.tsv", out_dir), &all_rows);
    write_action_sensitivity_tsv(&format!("{}/action_sensitivity.tsv", out_dir), &all_rows);
    write_diminishing_returns_tsv(&format!("{}/diminishing_returns.tsv", out_dir), &all_rows);
    write_assumptions_txt(&format!("{}/assumptions.txt", out_dir),
        &voi_file, n_points, mc_draws, &all_rows);

    eprintln!("Wrote {} to {}/", ["evsi.tsv", "action_sensitivity.tsv",
        "diminishing_returns.tsv", "assumptions.txt"].join(", "), out_dir);
}

fn voi_usage() -> ! {
    eprintln!("usage: camdl voi run VOI.toml [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --output DIR    output directory (default: <output_dir>/analysis/voi/)");
    eprintln!("  --mc-draws N    Monte Carlo draws per study (default: 1000)");
    eprintln!("  --seed N        RNG seed (default: 42)");
    std::process::exit(1);
}


