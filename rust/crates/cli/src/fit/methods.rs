//! Single source of truth for the `(algorithm, backend)` matrix.
//!
//! The Phase-1 ODE-inference proposal
//! (`docs/dev/proposals/2026-05-04-ode-inference-three-phase.md`) splits the
//! old `method = "..."` field into explicit `algorithm` + `backend` fields.
//! Each algorithm structurally requires a specific backend — PF-based
//! algorithms (if2 / pgas / pmmh) need the stochastic process kernel
//! (`chain_binomial`); deterministic-optimizer or exact-likelihood algorithms
//! (nl-sbplx / nl-bobyqa, and Phase 2/3's `mh` / `nuts`) need the deterministic
//! `ode` skeleton.
//!
//! `METHODS` is the canonical list of supported pairs. The fit.toml validator,
//! `camdl fit methods` subcommand, runtime status banners, and invalid-pair
//! error messages all read from it. Adding an algorithm = one entry here plus
//! its dispatcher arm in `fit/mod.rs`.

use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodStatus {
    /// Validated against published / vignette use cases; production-ready.
    Stable,
    /// Shipped and exercised but downstream validation still accumulating.
    /// Surfaced as `[beta]`; runtime banner names the caveat.
    Beta,
    /// Known limitations that affect correctness in some regime.
    /// Surfaced as `[experimental]`; runtime banner is loud.
    Experimental,
}

impl MethodStatus {
    fn as_tag(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
            Self::Experimental => "experimental",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodCategory {
    /// Inference algorithm — fits parameters to data (MLE or Bayesian).
    Inference,
    /// Diagnostic stage — evaluates the likelihood at fixed parameters.
    /// Not a parameter-inference method; surfaced separately.
    Diagnostic,
}

/// One supported `(algorithm, backend)` combination.
#[derive(Debug, Clone, Copy)]
pub struct InferenceMethod {
    pub algorithm: &'static str,
    pub backend: &'static str,
    pub category: MethodCategory,
    pub status: MethodStatus,
    /// One-line summary surfaced in `camdl fit methods` and error messages.
    pub one_liner: &'static str,
    /// "Use for:" sub-line in `camdl fit methods` rendering. May be empty.
    pub use_for: &'static str,
    /// Banner text for Beta / Experimental methods. Empty for Stable.
    pub status_note: &'static str,
}

/// Canonical method registry. Order is rendering order in
/// `camdl fit methods`; group by backend, then by category, then by status.
pub const METHODS: &[InferenceMethod] = &[
    // ─── chain_binomial backend (stochastic process kernel) ───────────────
    InferenceMethod {
        algorithm: "if2",
        backend: "chain_binomial",
        category: MethodCategory::Inference,
        status: MethodStatus::Stable,
        one_liner: "Iterated filtering MLE — perturbation-and-filter loop.",
        use_for: "scout/refine pipelines on stochastic models.",
        status_note: "",
    },
    InferenceMethod {
        algorithm: "pgas",
        backend: "chain_binomial",
        category: MethodCategory::Inference,
        status: MethodStatus::Stable,
        one_liner: "Particle Gibbs + NUTS-on-θ; production Bayesian path.",
        use_for: "Bayesian posteriors on stochastic models.",
        status_note: "",
    },
    InferenceMethod {
        algorithm: "pmmh",
        backend: "chain_binomial",
        category: MethodCategory::Inference,
        status: MethodStatus::Experimental,
        one_liner: "Pseudo-marginal MH; PF-inside-MH Bayesian sampler.",
        use_for: "small-T posterior sampling when PGAS isn't a fit.",
        status_note:
            "PMMH acceptance rates degrade for T > 500 observations. \
             Correlated pseudo-marginal (rho config) helps but has limits \
             on discrete-state models. PGAS is the production Bayesian path.",
    },
    InferenceMethod {
        algorithm: "pfilter",
        backend: "chain_binomial",
        category: MethodCategory::Diagnostic,
        status: MethodStatus::Stable,
        one_liner: "Bootstrap particle filter — likelihood evaluation only.",
        use_for: "post-fit diagnostic loglik (mean ± SD across replicates) \
                  and prequential scoring.",
        status_note: "",
    },
    // ─── ode backend (deterministic skeleton; new in Phase 1) ─────────────
    InferenceMethod {
        algorithm: "nl-sbplx",
        backend: "ode",
        category: MethodCategory::Inference,
        status: MethodStatus::Beta,
        one_liner: "Sbplx via NLopt — Nelder-Mead variant, robust to \
                    boundary non-smoothness.",
        use_for: "default deterministic MLE; equilibrium / large-population \
                  fits where PF is structurally redundant.",
        status_note:
            "Phase 1 typhoid validation passed; other model classes still \
             gathering downstream feedback.",
    },
    InferenceMethod {
        algorithm: "nl-bobyqa",
        backend: "ode",
        category: MethodCategory::Inference,
        status: MethodStatus::Beta,
        one_liner: "BOBYQA via NLopt — quadratic-trust-region.",
        use_for: "smooth deterministic objectives where Sbplx is overkill; \
                  faster than Sbplx on quadratic-shaped likelihoods.",
        status_note:
            "Requires smooth objective in the search box; fails at \
             parameter-bound boundaries where Sbplx succeeds. Prefer \
             nl-sbplx unless you've confirmed the boundary is interior.",
    },
];

/// Look up a method by (algorithm, backend). Returns `None` if the pair
/// isn't in the registry — caller renders the structured error.
pub fn lookup(algorithm: &str, backend: &str) -> Option<&'static InferenceMethod> {
    METHODS
        .iter()
        .find(|m| m.algorithm == algorithm && m.backend == backend)
}

/// Validate a `(algorithm, backend)` pair at config-load time.
///
/// On failure returns a fully-formed multi-line error message that names a
/// structural reason and suggests the right alternative. On success the entry
/// is returned, but the runtime caveat banner is driven by `status_note` /
/// `emit_status_banner` below rather than by callers inspecting the `Ok`.
pub fn validate_combo(
    algorithm: &str,
    backend: &str,
) -> Result<&'static InferenceMethod, String> {
    if let Some(m) = lookup(algorithm, backend) {
        return Ok(m);
    }
    Err(render_invalid_combo(algorithm, backend))
}

/// The registry caveat for a `(algorithm, backend)` pair — its `status_note` if
/// the pair is registered and carries a non-empty note, else `None`. Single
/// source of truth for the runtime caveat banner (`emit_status_banner`); the
/// same field drives `camdl fit methods`, so the two can never drift.
pub fn status_note(algorithm: &str, backend: &str) -> Option<&'static str> {
    lookup(algorithm, backend)
        .map(|m| m.status_note)
        .filter(|s| !s.is_empty())
}

/// Print the registry caveat banner to stderr when the chosen method is
/// Beta/Experimental (non-empty `status_note`). No-op for Stable methods and
/// for unregistered pairs (those fail earlier in `validate_combo`). Driven
/// entirely by the registry so the banner text and `camdl fit methods` stay in
/// lockstep — this replaces the previously hand-coded, PMMH-only banner.
pub fn emit_status_banner(algorithm: &str, backend: &str) {
    use owo_colors::OwoColorize;
    if let Some(note) = status_note(algorithm, backend) {
        eprintln!("{}", format!("⚠ {note}").yellow());
        eprintln!();
    }
}

/// Per-pair structural reasons for known invalid combinations. Hand-crafted
/// per the proposal's "error messages are a feature, not polish" principle —
/// the message must point at the right alternative, not just say "no".
fn rejection_reason(algorithm: &str, backend: &str) -> Option<&'static str> {
    match (algorithm, backend) {
        ("if2", "ode") => Some(
            "IF2 (Iterated Filtering 2) is a particle-filter-based MLE \
             algorithm. It perturbs parameters across particles and uses \
             the between-particle trajectory variance to drive the \
             optimization. Under the ODE backend all particles produce \
             identical trajectories per parameter point — there is no \
             between-particle variance for IF2 to exploit. The algorithm \
             collapses to a noisy gradient-free hill-climber that is \
             structurally a worse optimizer than the deterministic \
             alternatives.\n\n  \
             If you want MLE on the ODE backend, use:\n    \
             algorithm = \"nl-sbplx\"   default deterministic MLE; robust \
                                          to boundary non-smoothness\n    \
             algorithm = \"nl-bobyqa\"  faster than Sbplx on smooth \
                                          objectives",
        ),
        ("pgas", "ode") => Some(
            "PGAS (Particle Gibbs with Ancestor Sampling) is a particle-\
             filter-based Bayesian sampler — its CSMC step needs \
             stochastic process variance to refresh the trajectory \
             between θ updates. Under ODE all particles produce identical \
             trajectories per θ, so the CSMC step is degenerate.\n\n  \
             If you want Bayesian inference on the ODE backend, use:\n    \
             algorithm = \"mh\"     vanilla MH on the deterministic \
                                       likelihood (Phase 2)\n    \
             algorithm = \"nuts\"   gradient-based NUTS via forward \
                                       sensitivity (Phase 3)",
        ),
        ("pmmh", "ode") => Some(
            "PMMH (Pseudo-Marginal Metropolis-Hastings) wraps a particle \
             filter inside an MH acceptance step — the PF wrapping is \
             exactly what makes the sampler unbiased on a stochastic \
             likelihood. Under ODE the PF wrapping is degenerate \
             (1-particle, exact); the algorithm collapses to vanilla MH \
             on the deterministic marginal likelihood.\n\n  \
             If you want Bayesian inference on the ODE backend, use:\n    \
             algorithm = \"mh\"     vanilla MH on the deterministic \
                                       likelihood directly (Phase 2)",
        ),
        ("nl-sbplx", "chain_binomial") | ("nl-bobyqa", "chain_binomial") => Some(
            "NLopt deterministic optimizers (Sbplx, BOBYQA) operate on a \
             smooth objective. Under the chain_binomial backend the \
             single-trajectory loglik is a noisy estimator of the true \
             marginal likelihood — the optimizer sees ranking noise that \
             defeats convergence. IF2's perturbation-and-filter loop is \
             the right tool for MLE on a stochastic objective.\n\n  \
             If you want MLE on the chain_binomial backend, use:\n    \
             algorithm = \"if2\"   Iterated filtering MLE",
        ),
        ("mh", "chain_binomial") => Some(
            "Vanilla MH on a noisy single-trajectory loglik gives biased \
             posteriors — the PF wrapping is exactly what makes PMMH \
             unbiased on a stochastic likelihood. Use PMMH if you need a \
             Bayesian sampler on the chain_binomial backend.\n\n  \
             If you want Bayesian inference on the chain_binomial \
             backend, use:\n    \
             algorithm = \"pgas\"   Particle Gibbs (production Bayesian path)\n    \
             algorithm = \"pmmh\"   Pseudo-marginal MH (experimental)",
        ),
        ("nuts", "chain_binomial") => Some(
            "Vanilla NUTS on a stochastic likelihood is not a coherent \
             algorithm — gradients are noisy under PF wrapping. PGAS \
             handles this by integrating NUTS into a Gibbs sweep over \
             trajectories.\n\n  \
             If you want gradient-based Bayesian inference on the \
             chain_binomial backend, use:\n    \
             algorithm = \"pgas\"   integrates NUTS-on-θ inside a Gibbs sweep",
        ),
        _ => None,
    }
}

fn render_invalid_combo(algorithm: &str, backend: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "stage has algorithm = \"{}\" with backend = \"{}\", which is not \
         a supported inference method.",
        algorithm, backend
    );
    out.push('\n');
    if let Some(reason) = rejection_reason(algorithm, backend) {
        out.push_str("  ");
        // Indent each line of the reason for readability under the header.
        for (i, line) in reason.lines().enumerate() {
            if i > 0 {
                out.push_str("\n  ");
            }
            out.push_str(line);
        }
        out.push('\n');
    } else {
        let known_alg = METHODS.iter().any(|m| m.algorithm == algorithm);
        let known_be = METHODS.iter().any(|m| m.backend == backend);
        if !known_alg && !known_be {
            let _ = writeln!(
                out,
                "  Unknown algorithm \"{}\" and unknown backend \"{}\".",
                algorithm, backend
            );
        } else if !known_alg {
            let _ = writeln!(out, "  Unknown algorithm \"{}\".", algorithm);
        } else if !known_be {
            let _ = writeln!(
                out,
                "  Unknown backend \"{}\". Supported backends: \
                 chain_binomial, ode.",
                backend
            );
        } else {
            let _ = writeln!(
                out,
                "  This algorithm/backend combination is not in the \
                 supported matrix."
            );
        }
    }
    out.push('\n');
    out.push_str("  Supported (algorithm, backend) pairs:\n");
    for m in METHODS {
        let _ = writeln!(
            out,
            "    ({:<10} {:<14}) {}",
            format!("{},", m.algorithm),
            m.backend,
            m.one_liner
                .lines()
                .next()
                .unwrap_or("")
        );
    }
    out.push('\n');
    out.push_str(
        "  Note: camdl computes a different statistical object on each \
         backend\n  (chain_binomial → p(y|θ); ode → p(y|θ, ODE_skeleton)). \
         In low-noise\n  regimes these converge empirically. See \
         docs/inference.md for guidance.\n",
    );
    out
}

/// Verify the compiled model's required capabilities are supported by
/// the requested backend. Returns a structured error pointing at the
/// right alternative when the model needs more than the backend
/// provides.
///
/// `validate_combo` is structural-only — it knows the (algorithm,
/// backend) registry but not the model. This helper closes the
/// model-capability gap: a model with `overdispersed(rate, σ²)`
/// transitions running on the deterministic ODE backend silently
/// produces the deterministic-skeleton likelihood (ignoring σ²)
/// and was the bug that motivated this check (see `camdl simulate
/// --backend ode` which already enforces the same gate via
/// `util::run_simulation`).
///
/// Call from every dispatch site that resolves a (algorithm, backend,
/// model) triple. For backends whose capability set covers everything
/// the model needs, this is a no-op.
/// How observation times relate to the integrator's `dt` grid — the
/// `obs_alignment` choice (`fit.toml [backend]`, alongside `dt`). See
/// docs/dev/proposals/2026-06-05-unified-timeline-effect-architecture.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ObsAlignment {
    /// Step exactly to each observation time (shortened final substep).
    Exact,
    /// Round observation times onto the `dt` grid (uniform stepping).
    Snap,
}

/// The single `(algorithm × obs-alignment)` support gate at the fit-dispatch seam.
/// Returns the resolved alignment, or a clean error for an unsupported
/// combination — converting today's *silent* fallbacks into loud errors:
/// `exact` + PGAS silently snaps to a uniform grid, and `exact` + off-grid
/// correlated-PMMH silently falls back to fresh RNG (decorrelating the CPM
/// estimator). The threading of the resolved alignment into the filters'
/// `Schedule` policy, and the unimplemented modes (PF-snap, PGAS-exact), are
/// Stage 3; today this is the validation seam.
///
/// * `requested = None` is the default: "exact where the algorithm supports it".
/// * `correlated` — a PMMH run with `rho` set (CPM); its pre-drawn-noise layout
///   assumes a fixed substep count per observation window, so `exact` requires
///   on-grid obs.
/// * `obs_on_grid` — every observation time is an integer multiple of `dt`.
pub fn resolve_obs_alignment(
    algorithm: &str,
    correlated: bool,
    requested: Option<ObsAlignment>,
    obs_on_grid: bool,
) -> Result<ObsAlignment, String> {
    use ObsAlignment::{Exact, Snap};
    match algorithm {
        // Exact-steppers: land exactly on any obs. No `snap` inference path exists.
        "if2" | "pfilter" => match requested {
            None | Some(Exact) => Ok(Exact),
            Some(Snap) => Err(format!(
                "{algorithm}: obs_alignment = \"snap\" is not implemented — it steps \
                 exactly to observation times. Use \"exact\" (the default)."
            )),
        },
        "pmmh" => match (requested, correlated, obs_on_grid) {
            // Plain PMMH (no `rho`) is the bootstrap PF: exact on any obs.
            (None | Some(Exact), false, _) => Ok(Exact),
            // Correlated PMMH (CPM, `rho` set): exact only on-grid.
            (None | Some(Exact), true, true) => Ok(Exact),
            (None, true, false) => Err(
                "pmmh with rho (correlated pseudo-marginal): observations are off the \
                 dt grid, but the correlated-PF noise layout assumes a fixed substep \
                 count per observation window. Put observations on the dt grid, or \
                 unset rho (plain PMMH steps exactly to any obs)."
                    .into(),
            ),
            (Some(Exact), true, false) => Err(
                "pmmh: obs_alignment = \"exact\" with rho (correlated) requires \
                 on-grid observations — the correlated-PF noise layout assumes a \
                 fixed substep count per observation window."
                    .into(),
            ),
            (Some(Snap), _, _) => {
                Err("pmmh: obs_alignment = \"snap\" is not implemented.".into())
            }
        },
        // PGAS uses a uniform grid; exact-PGAS is planned but not yet built.
        "pgas" => match requested {
            None | Some(Snap) => Ok(Snap),
            Some(Exact) => Err(
                "pgas: obs_alignment = \"exact\" is not yet implemented (PGAS uses a \
                 uniform grid; exact-PGAS is planned). Use \"snap\", or algorithm = \
                 if2 / pfilter for exact alignment."
                    .into(),
            ),
        },
        other => Err(format!("resolve_obs_alignment: unknown algorithm '{other}'")),
    }
}

/// The `(algorithm × ic_free)` support gate at the fit-dispatch seam (F1).
///
/// `ic_free = true` requests IC-free / conditional-likelihood inference:
/// weight-and-resample at the first observation (pinning the initial state)
/// but drop y₁ from the accumulated log-likelihood. This is honored only by
/// the cells that actually skip the first increment:
///
///   * `if2`     — `if2.rs` guards `total_loglik` at `obs_idx == 0`.
///   * `pfilter` — `particle_filter.rs` guards the increment at `obs_idx == 0`.
///   * `pmmh` **without** `rho` (plain / uncorrelated) — wraps the bootstrap
///     PF, so it inherits the guard.
///
/// The remaining cells score every observation unconditionally, so
/// `ic_free = true` would *silently* compute the UNCONDITIONAL likelihood
/// while the startup banner claims "conditioning on y₁":
///
///   * `pgas`                  — no conditioning field anywhere in `pgas.rs`.
///   * `nl-sbplx` / `nl-bobyqa` — score via `runner::compute_ode_loglik`,
///     which sums over every obs time with no skip.
///   * `pmmh` **with** `rho` (correlated PMMH) — routes to
///     `correlated_pf::bootstrap_filter_correlated`, which adds every
///     increment unconditionally.
///
/// For those, this hard-errors at config-load time, naming the limitation
/// and the supported cells — converting a silent wrong answer into a loud
/// failure. `correlated` is `true` for a PMMH stage with `rho` set.
pub fn validate_ic_free(algorithm: &str, correlated: bool) -> Result<(), String> {
    match algorithm {
        // Honoring cells: the first increment is dropped from the loglik.
        "if2" | "pfilter" => Ok(()),
        // Plain PMMH wraps the bootstrap PF (honors it); correlated PMMH
        // routes to the correlated PF, which does not.
        "pmmh" if !correlated => Ok(()),
        "pmmh" => Err(
            "ic_free = true is not supported with correlated PMMH (a `pmmh` \
             stage with `rho` set). The correlated particle filter \
             (correlated_pf) accumulates every observation's log-likelihood \
             increment unconditionally, so it would silently compute the \
             UNCONDITIONAL likelihood while reporting that it conditioned on \
             y₁.\n\n  \
             ic_free is honored by: if2, pfilter, and plain pmmh (no rho).\n  \
             Either unset `rho` on this stage (plain PMMH honors ic_free), or \
             remove `ic_free = true`."
                .into(),
        ),
        "pgas" => Err(
            "ic_free = true is not supported with the `pgas` algorithm. PGAS \
             accumulates every observation's log-likelihood increment \
             unconditionally (no conditioning field exists in its CSMC / \
             ancestor-sampling path), so it would silently compute the \
             UNCONDITIONAL likelihood while reporting that it conditioned on \
             y₁.\n\n  \
             ic_free is honored by: if2, pfilter, and plain pmmh (no rho).\n  \
             Use one of those for IC-free inference, or remove \
             `ic_free = true` from the fit."
                .into(),
        ),
        "nl-sbplx" | "nl-bobyqa" => Err(format!(
            "ic_free = true is not supported with the `{algorithm}` algorithm \
             (ODE-MLE). The deterministic likelihood (compute_ode_loglik) sums \
             over every observation time with no first-observation skip, so it \
             would silently compute the UNCONDITIONAL likelihood while \
             reporting that it conditioned on y₁.\n\n  \
             ic_free is honored by: if2, pfilter, and plain pmmh (no rho).\n  \
             Use one of those for IC-free inference, or remove \
             `ic_free = true` from the fit."
        )),
        other => Err(format!(
            "validate_ic_free: unknown algorithm '{other}'"
        )),
    }
}

pub fn check_model_capabilities(
    backend: &str,
    compiled: &sim::CompiledModel,
) -> Result<(), String> {
    use sim::Capabilities;
    let backend_caps = match backend {
        // chain_binomial-inference grants OVERDISPERSION (NegBinomial draws)
        // and BALANCE (`balance{}` is applied via step_one in the filter
        // loops — gh#192; it was wrongly withheld, so `profile` falsely
        // rejected balance{} models). It intentionally does NOT advertise
        // REAL_COMPARTMENTS for INFERENCE: the filter loops carry no real
        // state and never advance a reservoir, so a real-coupled model would
        // be fit with its real compartments frozen at their init value —
        // silently mis-fit (gh#191). Re-grant REAL_COMPARTMENTS here once
        // inference advances real state.
        // RUNTIME_DT (gh#54): both inference backends realize a substep `dt`
        // — chain_binomial via the PGAS StepClock substeps
        // (gate_dt_rate_exact_clip.rs) and ode via RK4 flow accumulation
        // (ode_dt_rate_flow.rs) — so a `dt`-in-rate model fits on either. The
        // requirement only excludes gillespie, which is not an inference
        // backend here.
        "chain_binomial" => {
            Capabilities::OVERDISPERSION | Capabilities::BALANCE | Capabilities::RUNTIME_DT
        }
        "ode"            => Capabilities::REAL_COMPARTMENTS | Capabilities::RUNTIME_DT,
        other            => return Err(format!(
            "check_model_capabilities: unknown backend '{}'", other
        )),
    };
    let required = compiled.required_capabilities();
    let unsupported = required - backend_caps;
    if unsupported.is_empty() {
        return Ok(());
    }
    // Iterate the unsupported bitflags (bitflags 2.x `iter_names`) rather than
    // a hand if-ladder, so EVERY flag renders a non-blank message — an
    // unsupported flag with no hand-written branch previously produced a
    // blank `  - ` line (gh#192). `capability_hint` carries the rich
    // per-flag hint text.
    let features: Vec<String> = unsupported
        .iter_names()
        .map(|(name, flag)| capability_hint(name, flag))
        .collect();
    Err(format!(
        "model requires capabilities not supported by backend '{}':\n  - {}",
        backend,
        features.join("\n  - "),
    ))
}

/// Per-capability hint text for the unsupported-capability error. Keyed on the
/// `Capabilities` flag; `name` is the bitflags constant name (used as the
/// non-blank fallback for any flag without bespoke guidance, so the message
/// can never be empty — gh#192).
fn capability_hint(name: &str, flag: sim::Capabilities) -> String {
    use sim::Capabilities;
    match flag {
        Capabilities::OVERDISPERSION =>
            "OVERDISPERSION: the model has `overdispersed(...)` transitions \
             whose process noise (σ²) the deterministic ODE skeleton \
             ignores. Switch to backend = \"chain_binomial\" (algorithms \
             if2 / pgas / pmmh) for stochastic-process inference, or \
             remove the overdispersed wrapper if the noise isn't \
             load-bearing for your inference question.".to_string(),
        Capabilities::REAL_COMPARTMENTS =>
            "REAL_COMPARTMENTS: stochastic inference (backend = \
             \"chain_binomial\") does not yet advance real-valued \
             (ODE-coupled) compartments — they would be held frozen at their \
             initial value, silently mis-fitting any transition rate that \
             couples to them (gh#191). For a deterministic-skeleton fit use \
             backend = \"ode\" (which integrates the real compartments); \
             otherwise remove the real compartments, or use forward \
             simulation (`camdl simulate`) for real-coupled stochastic \
             dynamics.".to_string(),
        Capabilities::BALANCE =>
            "BALANCE: the model has a `balance{}` block (a population-residual \
             compartment). Its firing semantics are chain-binomial-only \
             (substep residual after transitions and events); the ODE backend \
             conserves population algebraically and has no substep to apply \
             it. Use backend = \"chain_binomial\", or remove the `balance{}` \
             block.".to_string(),
        Capabilities::RUNTIME_DT =>
            "RUNTIME_DT: the model uses `dt` in a rate (the runtime substep \
             length). gillespie has no substep — its SSA loop would freeze \
             `dt` to the nominal `simulation.dt`-or-`1.0`, silently changing \
             the rate. Use backend = \"chain_binomial\" or \"ode\" (both \
             evaluate the rate at the realized substep length), or remove the \
             `dt` factor from the rate.".to_string(),
        // Any other flag (e.g. LINEAGES) still gets a named, non-blank line.
        _ => format!(
            "{name}: required by the model but not supported by this backend."
        ),
    }
}

/// Render the registry as a user-facing reference table.
/// Output goes to `camdl fit methods` and is also embedded in `--help`.
pub fn render_matrix() -> String {
    let mut out = String::new();
    let backends = [
        (
            "chain_binomial",
            "CHAIN_BINOMIAL backend (stochastic process kernel)",
        ),
        (
            "ode",
            "ODE backend (deterministic skeleton; new in this release)",
        ),
    ];
    for (be_name, header) in backends {
        let _ = writeln!(out, "{}\n", header);
        let methods_for_be: Vec<_> =
            METHODS.iter().filter(|m| m.backend == be_name).collect();
        if methods_for_be.is_empty() {
            continue;
        }
        // Inference algorithms first, diagnostics second.
        for cat in [MethodCategory::Inference, MethodCategory::Diagnostic] {
            for m in methods_for_be.iter().filter(|m| m.category == cat) {
                let cat_label = match m.category {
                    MethodCategory::Inference => "",
                    MethodCategory::Diagnostic => " (diagnostic)",
                };
                let _ = writeln!(
                    out,
                    "  algorithm = \"{}\"  [{}{}]",
                    m.algorithm,
                    m.status.as_tag(),
                    cat_label
                );
                for line in m.one_liner.lines() {
                    let _ = writeln!(out, "    {}", line.trim_start());
                }
                if !m.use_for.is_empty() {
                    let _ = writeln!(out, "    Use for: {}", m.use_for);
                }
                if !m.status_note.is_empty() {
                    let _ = writeln!(out, "    ⚠ {}", m.status_note);
                }
                out.push('\n');
            }
        }
    }
    out.push_str(
        "Methods compute different statistical objects across backends:\n  \
         chain_binomial → p(y|θ) under stochastic process noise\n  \
         ode            → p(y|θ, ODE_skeleton) — Jensen's inequality bias\n\
         In low-noise regimes these converge empirically. See \
         docs/inference.md\nfor guidance on when to pick which backend.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase1_method_present() {
        for (a, b) in [
            ("if2", "chain_binomial"),
            ("pgas", "chain_binomial"),
            ("pmmh", "chain_binomial"),
            ("pfilter", "chain_binomial"),
            ("nl-sbplx", "ode"),
            ("nl-bobyqa", "ode"),
        ] {
            assert!(
                lookup(a, b).is_some(),
                "expected ({a}, {b}) in METHODS"
            );
        }
    }

    #[test]
    fn obs_alignment_exact_steppers_any_obs() {
        use ObsAlignment::{Exact, Snap};
        // if2/pfilter are exact on any obs; default and explicit exact both ok.
        for algo in ["if2", "pfilter"] {
            assert_eq!(resolve_obs_alignment(algo, false, None, true), Ok(Exact));
            assert_eq!(resolve_obs_alignment(algo, false, None, false), Ok(Exact));
            assert_eq!(resolve_obs_alignment(algo, false, Some(Exact), false), Ok(Exact));
            // snap is not implemented for the exact-steppers.
            assert!(resolve_obs_alignment(algo, false, Some(Snap), true).is_err());
        }
    }

    #[test]
    fn obs_alignment_pmmh_is_rho_dependent() {
        use ObsAlignment::Exact;
        // Plain PMMH (uncorrelated) = bootstrap PF: exact on any obs.
        assert_eq!(resolve_obs_alignment("pmmh", false, None, false), Ok(Exact));
        // Correlated PMMH (rho set): exact OK on-grid...
        assert_eq!(resolve_obs_alignment("pmmh", true, None, true), Ok(Exact));
        // ...but off-grid + correlated is a CLEAN ERROR (was silent fresh-RNG
        // decorrelation, #17), under both default and explicit exact.
        assert!(resolve_obs_alignment("pmmh", true, None, false).is_err());
        assert!(resolve_obs_alignment("pmmh", true, Some(Exact), false).is_err());
    }

    #[test]
    fn obs_alignment_pgas_snap_only_exact_is_clean_error() {
        use ObsAlignment::{Exact, Snap};
        // PGAS defaults to snap (its only mode today)...
        assert_eq!(resolve_obs_alignment("pgas", false, None, true), Ok(Snap));
        assert_eq!(resolve_obs_alignment("pgas", false, Some(Snap), true), Ok(Snap));
        // ...and exact is a CLEAN ERROR (was a silent snap), naming the fix.
        let err = resolve_obs_alignment("pgas", false, Some(Exact), true).unwrap_err();
        assert!(err.contains("not yet implemented"), "should name exact-PGAS as unimplemented: {err}");
        assert!(err.contains("if2") || err.contains("snap"), "should suggest a fix: {err}");
    }

    // ── ic_free / conditioning support gate (F1) ───────────────────────────
    //
    // `ic_free = true` (IC-free / conditional likelihood) is honored only by
    // the cells that actually drop y₁ from the accumulated loglik: IF2, the
    // bootstrap particle filter (`pfilter`), and plain PMMH (uncorrelated —
    // it wraps the bootstrap PF). PGAS, the ODE-MLE optimizers
    // (`nl-sbplx` / `nl-bobyqa`, via `compute_ode_loglik`), and correlated
    // PMMH (rho set, via `bootstrap_filter_correlated`) score every
    // observation unconditionally — running them with `ic_free = true`
    // silently computes the UNCONDITIONAL likelihood while the banner claims
    // conditioning. The gate hard-errors those cells.

    #[test]
    fn ic_free_honored_cells_succeed() {
        // IF2 and the bootstrap PF honor conditioning — must pass.
        assert!(validate_ic_free("if2", false).is_ok());
        assert!(validate_ic_free("pfilter", false).is_ok());
        // Plain PMMH (no rho / uncorrelated) wraps the bootstrap PF — honors it.
        assert!(validate_ic_free("pmmh", false).is_ok());
    }

    #[test]
    fn ic_free_pgas_is_hard_error_naming_the_limitation() {
        let err = validate_ic_free("pgas", false).unwrap_err();
        assert!(err.contains("ic_free"), "must name ic_free: {err}");
        assert!(err.contains("pgas"), "must name the offending algorithm: {err}");
        // Points the user at a supported alternative.
        assert!(
            err.contains("if2") || err.contains("pfilter"),
            "must name a supported cell: {err}"
        );
    }

    #[test]
    fn ic_free_ode_mle_is_hard_error() {
        // Both NLopt deterministic optimizers score every obs via
        // compute_ode_loglik — conditioning is silently ignored.
        for algo in ["nl-sbplx", "nl-bobyqa"] {
            let err = validate_ic_free(algo, false).unwrap_err();
            assert!(err.contains("ic_free"), "{algo}: must name ic_free: {err}");
            assert!(err.contains(algo), "{algo}: must name the algorithm: {err}");
        }
    }

    #[test]
    fn ic_free_correlated_pmmh_is_hard_error_but_plain_pmmh_is_ok() {
        // Correlated PMMH (rho set) routes to bootstrap_filter_correlated,
        // which adds every increment unconditionally → reject.
        let err = validate_ic_free("pmmh", true).unwrap_err();
        assert!(err.contains("ic_free"), "must name ic_free: {err}");
        assert!(
            err.contains("correlated") || err.contains("rho"),
            "must name the correlated/rho condition: {err}"
        );
        // ...but plain PMMH (uncorrelated) honors conditioning.
        assert!(validate_ic_free("pmmh", false).is_ok());
    }

    #[test]
    fn invalid_pf_method_on_ode_names_nlopt_alternative() {
        let err = validate_combo("if2", "ode").unwrap_err();
        assert!(err.contains("nl-sbplx"), "message should suggest nl-sbplx; got:\n{err}");
        assert!(err.contains("MLE on the ODE backend"));
    }

    #[test]
    fn invalid_nlopt_on_chain_binomial_names_if2() {
        let err = validate_combo("nl-sbplx", "chain_binomial").unwrap_err();
        assert!(err.contains("if2"), "message should suggest if2; got:\n{err}");
    }

    #[test]
    fn unknown_algorithm_yields_clear_error() {
        let err = validate_combo("not-a-method", "ode").unwrap_err();
        assert!(err.contains("Unknown algorithm"), "got:\n{err}");
    }

    #[test]
    fn unknown_backend_yields_clear_error() {
        let err = validate_combo("if2", "not-a-backend").unwrap_err();
        assert!(err.contains("Unknown backend"), "got:\n{err}");
    }

    #[test]
    fn rejection_message_lists_full_matrix() {
        let err = validate_combo("if2", "ode").unwrap_err();
        for m in METHODS {
            assert!(
                err.contains(m.algorithm),
                "expected algorithm {} listed in error; got:\n{err}",
                m.algorithm
            );
        }
    }

    #[test]
    fn render_matrix_groups_by_backend() {
        let out = render_matrix();
        let cb_pos = out
            .find("CHAIN_BINOMIAL backend")
            .expect("chain_binomial header");
        let ode_pos = out.find("ODE backend").expect("ode header");
        assert!(
            cb_pos < ode_pos,
            "chain_binomial section should come before ode section"
        );
        // Pfilter labelled as diagnostic.
        let pf_idx = out.find("\"pfilter\"").expect("pfilter listed");
        let pf_line_end = out[pf_idx..]
            .find('\n')
            .map(|n| pf_idx + n)
            .unwrap_or(out.len());
        assert!(
            out[pf_idx..pf_line_end].contains("diagnostic"),
            "pfilter line should mark it as diagnostic"
        );
    }

    #[test]
    fn status_note_is_the_single_source_for_runtime_caveats() {
        // G4 (docs/dev/capabilities-system.md): the runtime caveat banner is
        // driven by the registry `status_note`, not hand-coded per method.
        // `status_note()` is what the fit/profile dispatch paths call, so this
        // pins the contract — a hand-coded banner can no longer drift from the
        // registry text, and Beta methods can't silently lack a runtime caveat.
        assert!(
            status_note("pmmh", "chain_binomial").is_some_and(|n| n.contains("T > 500")),
            "experimental PMMH must surface its caveat at runtime"
        );
        // The bug this closes: Beta NLopt caveats never reached runtime before
        // — only PMMH had a hand-coded banner.
        assert!(
            status_note("nl-sbplx", "ode").is_some_and(|n| n.contains("Phase 1")),
            "Beta nl-sbplx caveat must surface at runtime, not just in `fit methods`"
        );
        assert!(
            status_note("nl-bobyqa", "ode").is_some(),
            "Beta nl-bobyqa carries a caveat"
        );
        // Stable methods: no banner.
        assert_eq!(status_note("if2", "chain_binomial"), None);
        assert_eq!(status_note("pgas", "chain_binomial"), None);
        assert_eq!(status_note("pfilter", "chain_binomial"), None);
        // Unregistered pair: no banner (validate_combo emits the hard error).
        assert_eq!(status_note("pgas", "ode"), None);
    }

    #[test]
    fn chain_binomial_inference_rejects_real_compartments() {
        // gh#191: the inference path carries no real state and never advances a
        // reservoir, so a real-coupled model would be fit with its real
        // compartments frozen at init — silently mis-fit. The capability gate
        // must REJECT it (before gh#191 it was silently accepted). Forward sim
        // handles real compartments correctly (see #3 / 5c7585c).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../ocaml/golden/sir_reservoir_mixed.ir.json"
        );
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        // Golden IR is enveloped: { ir_version, validated_by, model: {...} },
        // and stores parameter values as null (resolved at run time). Fill them
        // with an in-bounds placeholder so the model compiles — the values are
        // irrelevant to the capability check.
        let env: serde_json::Value =
            serde_json::from_str(&json).expect("parse sir_reservoir_mixed envelope");
        let mut model: ir::Model = serde_json::from_value(env["model"].clone())
            .expect("deserialize sir_reservoir_mixed model");
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        let compiled = sim::CompiledModel::new(model).expect("compile sir_reservoir_mixed");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::REAL_COMPARTMENTS),
            "fixture must actually have real compartments"
        );
        let err = check_model_capabilities("chain_binomial", &compiled)
            .expect_err("chain_binomial inference must reject real-coupled models");
        assert!(err.contains("gh#191"), "should cite the tracking issue: {err}");
        assert!(err.contains("frozen"), "should explain the frozen-reservoir reason: {err}");
        // ode integrates real compartments — still accepted.
        assert!(check_model_capabilities("ode", &compiled).is_ok());
    }

    /// Build a `CompiledModel` from the sir_basic golden with a `balance{}`
    /// block injected (target = integer compartment `R`, expr = a resolvable
    /// param). The chain-binomial inference path applies balance via
    /// `step_one`, so the gate must ACCEPT it; the only required capability is
    /// `BALANCE`. Avoids invoking camdlc (no version-guard dependency).
    fn compiled_sir_with_balance() -> sim::CompiledModel {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../ocaml/golden/sir_basic.ir.json"
        );
        let json =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let envv: serde_json::Value =
            serde_json::from_str(&json).expect("parse sir_basic envelope");
        let mut model: ir::Model = serde_json::from_value(envv["model"].clone())
            .expect("deserialize sir_basic model");
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        // `R = N0` — target R is an integer compartment, N0 a declared param;
        // value irrelevant to the capability scan, just needs to resolve.
        model.balance = Some(ir::model::BalanceSpec {
            target: "R".to_string(),
            expr: ir::expr::Expr::param("N0"),
        });
        sim::CompiledModel::new(model).expect("compile sir_basic + balance")
    }

    #[test]
    fn chain_binomial_inference_accepts_balance() {
        // gh#192: `balance{}` is a chain-binomial-only construct that the
        // inference filter loops apply via step_one — so the capability gate
        // must ACCEPT it on chain_binomial. Before the fix the gate granted
        // chain_binomial only OVERDISPERSION, so a balance{} model was
        // falsely rejected on `profile` (the one path that calls the gate),
        // even though fit/survey/pfilter ran it.
        let compiled = compiled_sir_with_balance();
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::BALANCE),
            "fixture must actually require BALANCE"
        );
        check_model_capabilities("chain_binomial", &compiled).unwrap_or_else(|e| {
            panic!("chain_binomial inference must ACCEPT balance{{}} models: {e}")
        });
    }

    #[test]
    fn unsupported_capability_message_is_never_blank() {
        // gh#192 part 2: the error builder only had hand-written branches for
        // OVERDISPERSION / REAL_COMPARTMENTS, so any OTHER unsupported flag
        // (e.g. BALANCE on `ode`) rendered a blank `  - ` line that never
        // named the missing capability. Drive an unsupported BALANCE through
        // the ode backend (ode grants only REAL_COMPARTMENTS) and assert the
        // message names the capability rather than printing an empty entry.
        let compiled = compiled_sir_with_balance();
        let err = check_model_capabilities("ode", &compiled)
            .expect_err("balance{} on ode must be rejected");
        assert!(
            err.contains("BALANCE"),
            "error must NAME the unsupported capability, not print a blank line: {err:?}"
        );
        // No empty bullet: every `  - ` entry must carry text after it.
        for line in err.lines() {
            assert_ne!(line.trim_end(), "  -", "bare blank bullet: {err:?}");
            if let Some(rest) = line.trim_end().strip_prefix("  - ") {
                assert!(
                    !rest.trim().is_empty(),
                    "blank capability bullet in message: {err:?}"
                );
            }
        }
    }

    /// gh#54: a `dt`-in-rate model requires RUNTIME_DT. Both inference
    /// backends realize a substep `dt` (chain_binomial via PGAS StepClock,
    /// ode via RK4 flow accumulation), so the inference capability gate must
    /// ACCEPT it on both — a missing grant here would falsely reject a
    /// legitimate dt-rate fit. (gillespie is not an inference backend.)
    #[test]
    fn inference_accepts_dt_in_rate_on_both_backends() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../tests/fixtures/corner_cases/ir/dt_rate.ir.json"
        );
        let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let model = ir::from_str(&json).expect("parse dt_rate IR");
        let compiled = sim::CompiledModel::new(model).expect("compile dt_rate");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::RUNTIME_DT),
            "fixture must actually require RUNTIME_DT"
        );
        check_model_capabilities("chain_binomial", &compiled).unwrap_or_else(|e| {
            panic!("chain_binomial inference must ACCEPT dt-in-rate models: {e}")
        });
        check_model_capabilities("ode", &compiled).unwrap_or_else(|e| {
            panic!("ode inference must ACCEPT dt-in-rate models: {e}")
        });
    }

    /// The RUNTIME_DT hint must name the feature (`dt`) and the fix
    /// (chain_binomial / ode) — not fall through to the blank-safe generic
    /// fallback. Exercised by driving an unsupported RUNTIME_DT through a
    /// backend grant that lacks it.
    #[test]
    fn runtime_dt_hint_names_feature_and_fix() {
        let hint = capability_hint("RUNTIME_DT", sim::Capabilities::RUNTIME_DT);
        assert!(hint.contains("RUNTIME_DT"), "hint must name the capability: {hint}");
        assert!(hint.contains("dt"), "hint must name the `dt` feature: {hint}");
        assert!(
            hint.contains("chain_binomial") && hint.contains("ode"),
            "hint must name the fix backends: {hint}"
        );
    }
}
