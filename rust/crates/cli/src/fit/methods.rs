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

use crate::run_meta::{FitAlgorithm, InferenceBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodStatus {
    /// Validated against published / vignette use cases; production-ready.
    Stable,
    /// Shipped and exercised but downstream validation still accumulating.
    /// Surfaced as `[beta]`; runtime banner names the caveat.
    Beta,
}

impl MethodStatus {
    fn as_tag(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
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
    pub algorithm: FitAlgorithm,
    pub backend: InferenceBackend,
    pub category: MethodCategory,
    pub status: MethodStatus,
    /// One-line summary surfaced in `camdl fit methods` and error messages.
    pub one_liner: &'static str,
    /// "Use for:" sub-line in `camdl fit methods` rendering. May be empty.
    pub use_for: &'static str,
    /// Runtime caveat banner text — surfaced for any method that carries one
    /// (drives `emit_status_banner`). Beta methods name their limitation here;
    /// a Stable method may carry usage guidance (e.g. PMMH's "prefer PGAS for
    /// long series"). Empty = no banner.
    pub status_note: &'static str,
}

/// Canonical method registry. Order is rendering order in
/// `camdl fit methods`; group by backend, then by category, then by status.
pub const METHODS: &[InferenceMethod] = &[
    // ─── chain_binomial backend (stochastic process kernel) ───────────────
    InferenceMethod {
        algorithm: FitAlgorithm::If2,
        backend: InferenceBackend::ChainBinomial,
        category: MethodCategory::Inference,
        status: MethodStatus::Stable,
        one_liner: "Iterated filtering MLE — perturbation-and-filter loop.",
        use_for: "scout/refine pipelines on stochastic models.",
        status_note: "",
    },
    InferenceMethod {
        algorithm: FitAlgorithm::Pgas,
        backend: InferenceBackend::ChainBinomial,
        category: MethodCategory::Inference,
        status: MethodStatus::Stable,
        one_liner: "Particle Gibbs + NUTS-on-θ; default Bayesian path.",
        use_for: "Bayesian posteriors on stochastic models.",
        status_note: "",
    },
    InferenceMethod {
        algorithm: FitAlgorithm::Pmmh,
        backend: InferenceBackend::ChainBinomial,
        category: MethodCategory::Inference,
        status: MethodStatus::Stable,
        one_liner: "Pseudo-marginal MH; PF-inside-MH Bayesian sampler.",
        use_for: "Bayesian posteriors on stochastic models; \
                  short-to-moderate series and freeze-then-sample workflows.",
        status_note:
            "Acceptance rates degrade for long observation series \
             (T > 500); prefer PGAS — the default Bayesian path — on long \
             chain-binomial series. Correlated pseudo-marginal (rho config) \
             helps but has limits on discrete-state models.",
    },
    InferenceMethod {
        algorithm: FitAlgorithm::Pfilter,
        backend: InferenceBackend::ChainBinomial,
        category: MethodCategory::Diagnostic,
        status: MethodStatus::Stable,
        one_liner: "Bootstrap particle filter — likelihood evaluation only.",
        use_for: "post-fit diagnostic loglik (mean ± SD across replicates) \
                  and prequential scoring.",
        status_note: "",
    },
    // ─── ode backend (deterministic skeleton; new in Phase 1) ─────────────
    InferenceMethod {
        algorithm: FitAlgorithm::NlSbplx,
        backend: InferenceBackend::Ode,
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
        algorithm: FitAlgorithm::NlBobyqa,
        backend: InferenceBackend::Ode,
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
    InferenceMethod {
        algorithm: FitAlgorithm::Mh,
        backend: InferenceBackend::Ode,
        category: MethodCategory::Inference,
        status: MethodStatus::Beta,
        one_liner: "Metropolis-Hastings on the deterministic ODE marginal likelihood.",
        use_for: "Bayesian posteriors on ODE/equilibrium models without gradients.",
        status_note: "",
    },
    InferenceMethod {
        algorithm: FitAlgorithm::Nuts,
        backend: InferenceBackend::Ode,
        category: MethodCategory::Inference,
        status: MethodStatus::Beta,
        one_liner: "No-U-Turn Sampler on the deterministic ODE marginal likelihood \
                    (gradient-based, via forward sensitivities).",
        use_for: "Bayesian posteriors on ODE/equilibrium models — the gradient \
                  sampler; scales to correlated, moderate-dimension posteriors \
                  better than the gradient-free `mh`.",
        status_note:
            "gh#275 Phase 2. Requires a differentiable model (the capability gate \
             refuses an unsupported rate/observation gradient, an adaptive \
             integrator, a scheduled effect, or a parameterized initial \
             condition). On stochastic backends, use `pgas` (which runs NUTS on \
             the conditioned trajectory).",
    },
];

/// Look up a method by (algorithm, backend). Returns `None` if the pair
/// isn't in the registry — caller renders the structured error.
pub fn lookup(
    algorithm: FitAlgorithm,
    backend: InferenceBackend,
) -> Option<&'static InferenceMethod> {
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
    algorithm: FitAlgorithm,
    backend: InferenceBackend,
) -> Result<&'static InferenceMethod, String> {
    if let Some(m) = lookup(algorithm, backend) {
        return Ok(m);
    }
    Err(render_invalid_combo(algorithm, backend))
}

/// Parse a user-supplied `(algorithm, backend)` string pair (the CLI boundary
/// for `camdl profile`/`fit`) into the typed registry entry. Strings enter the
/// typed world *here*; on any failure the error names the problem and points at
/// the matrix. `fit.toml` does not use this — its `Stage` is already typed, so
/// it calls [`validate_combo`] with `stage.method_kind()` / `stage.backend()`.
pub fn parse_combo(
    algorithm: &str,
    backend: &str,
) -> Result<&'static InferenceMethod, String> {
    match (parse_algorithm(algorithm), parse_backend(backend)) {
        (Some(a), Some(b)) => validate_combo(a, b),
        (a, b) => Err(render_unknown_combo(algorithm, backend, a, b)),
    }
}

/// Wire-string → [`FitAlgorithm`]; `None` for any name not in the registry
/// vocabulary (the inverse of [`FitAlgorithm::as_str`]).
fn parse_algorithm(s: &str) -> Option<FitAlgorithm> {
    Some(match s {
        "if2" => FitAlgorithm::If2,
        "pgas" => FitAlgorithm::Pgas,
        "pmmh" => FitAlgorithm::Pmmh,
        "mh" => FitAlgorithm::Mh,
        "nuts" => FitAlgorithm::Nuts,
        "pfilter" => FitAlgorithm::Pfilter,
        "nl-sbplx" => FitAlgorithm::NlSbplx,
        "nl-bobyqa" => FitAlgorithm::NlBobyqa,
        _ => return None,
    })
}

/// Wire-string → [`InferenceBackend`]; `None` for any unknown backend name.
fn parse_backend(s: &str) -> Option<InferenceBackend> {
    Some(match s {
        "chain_binomial" => InferenceBackend::ChainBinomial,
        "ode" => InferenceBackend::Ode,
        _ => return None,
    })
}

/// The registry caveat for a `(algorithm, backend)` pair — its `status_note` if
/// the pair is registered and carries a non-empty note, else `None`. Single
/// source of truth for the runtime caveat banner (`emit_status_banner`); the
/// same field drives `camdl fit methods`, so the two can never drift.
pub fn status_note(algorithm: FitAlgorithm, backend: InferenceBackend) -> Option<&'static str> {
    lookup(algorithm, backend)
        .map(|m| m.status_note)
        .filter(|s| !s.is_empty())
}

/// Print the registry caveat banner to stderr when the chosen method carries
/// a caveat (non-empty `status_note` — every Beta method, plus any Stable
/// method with usage guidance). No-op for methods without a note and for
/// unregistered pairs (those fail earlier in `validate_combo`). Driven entirely
/// by the registry so the banner text and `camdl fit methods` stay in lockstep
/// — this replaces the previously hand-coded, PMMH-only banner.
pub fn emit_status_banner(algorithm: FitAlgorithm, backend: InferenceBackend) {
    use owo_colors::OwoColorize;
    if let Some(note) = status_note(algorithm, backend) {
        eprintln!("{}", format!("⚠ {note}").yellow());
        eprintln!();
    }
}

/// Per-pair structural reasons for known invalid combinations. Hand-crafted
/// per the proposal's "error messages are a feature, not polish" principle —
/// the message must point at the right alternative, not just say "no".
fn rejection_reason(
    algorithm: FitAlgorithm,
    backend: InferenceBackend,
) -> Option<&'static str> {
    use FitAlgorithm as A;
    use InferenceBackend as B;
    match (algorithm, backend) {
        (A::If2, B::Ode) => Some(
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
        (A::Pgas, B::Ode) => Some(
            "PGAS (Particle Gibbs with Ancestor Sampling) is a particle-\
             filter-based Bayesian sampler — its CSMC step needs \
             stochastic process variance to refresh the trajectory \
             between θ updates. Under ODE all particles produce identical \
             trajectories per θ, so the CSMC step is degenerate.\n\n  \
             If you want Bayesian inference on the ODE backend, use:\n    \
             algorithm = \"nuts\"   gradient-based NUTS via forward \
                                       sensitivities (the gradient sampler)\n    \
             algorithm = \"mh\"     gradient-free MH on the deterministic \
                                       likelihood",
        ),
        (A::Pmmh, B::Ode) => Some(
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
        (A::NlSbplx, B::ChainBinomial) | (A::NlBobyqa, B::ChainBinomial) => Some(
            "NLopt deterministic optimizers (Sbplx, BOBYQA) operate on a \
             smooth objective. Under the chain_binomial backend the \
             single-trajectory loglik is a noisy estimator of the true \
             marginal likelihood — the optimizer sees ranking noise that \
             defeats convergence. IF2's perturbation-and-filter loop is \
             the right tool for MLE on a stochastic objective.\n\n  \
             If you want MLE on the chain_binomial backend, use:\n    \
             algorithm = \"if2\"   Iterated filtering MLE",
        ),
        (A::Mh, B::ChainBinomial) => Some(
            "Vanilla MH on a noisy single-trajectory loglik gives biased \
             posteriors — the PF wrapping is exactly what makes PMMH \
             unbiased on a stochastic likelihood. Use PMMH if you need a \
             Bayesian sampler on the chain_binomial backend.\n\n  \
             If you want Bayesian inference on the chain_binomial \
             backend, use:\n    \
             algorithm = \"pgas\"   Particle Gibbs (default Bayesian path)\n    \
             algorithm = \"pmmh\"   Pseudo-marginal MH",
        ),
        (A::Nuts, B::ChainBinomial) => Some(
            "NUTS needs a closed-form gradient of log p(y | θ), which only a \
             directly-differentiable (deterministic) likelihood provides. \
             Under the chain_binomial backend the marginal likelihood is an \
             intractable integral over latent trajectories — its gradient is \
             not available in closed form, so vanilla NUTS is not a coherent \
             algorithm here. PGAS handles this by running NUTS-on-θ *inside* a \
             Gibbs sweep, conditioned on a sampled trajectory.\n\n  \
             If you want gradient-based Bayesian inference on the \
             chain_binomial backend, use:\n    \
             algorithm = \"pgas\"   integrates NUTS-on-θ inside a Gibbs sweep",
        ),
        _ => None,
    }
}

/// Render the structured error for a *valid-but-unsupported* typed pair — both
/// the algorithm and backend are registry vocabulary, but the pair is not a
/// supported method (e.g. `if2` + `ode`). Called by [`validate_combo`].
fn render_invalid_combo(algorithm: FitAlgorithm, backend: InferenceBackend) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "stage has algorithm = \"{}\" with backend = \"{}\", which is not \
         a supported inference method.",
        algorithm, backend
    );
    out.push('\n');
    if let Some(reason) = rejection_reason(algorithm, backend) {
        append_indented(&mut out, reason);
    } else {
        out.push_str(
            "  This algorithm/backend combination is not in the supported \
             matrix.\n",
        );
    }
    append_matrix_footer(&mut out);
    out
}

/// Render the structured error for an *unparsed* `(algorithm, backend)` string
/// pair at the CLI boundary ([`parse_combo`]): an unknown algorithm and/or
/// backend name. `parsed_*` carry the parse results so the message names which
/// side failed. Every algorithm in the registry vocabulary parses, so a
/// valid-but-unsupported *pair* (e.g. `nuts` + `chain_binomial`) is handled by
/// [`validate_combo`]/[`rejection_reason`], not here.
fn render_unknown_combo(
    algorithm: &str,
    backend: &str,
    parsed_algo: Option<FitAlgorithm>,
    parsed_be: Option<InferenceBackend>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "stage has algorithm = \"{}\" with backend = \"{}\", which is not \
         a supported inference method.",
        algorithm, backend
    );
    out.push('\n');
    match (parsed_algo.is_none(), parsed_be.is_none()) {
        (true, true) => {
            let _ = writeln!(
                out,
                "  Unknown algorithm \"{}\" and unknown backend \"{}\".",
                algorithm, backend
            );
        }
        (true, false) => {
            let _ = writeln!(out, "  Unknown algorithm \"{}\".", algorithm);
        }
        (false, true) => {
            let _ = writeln!(
                out,
                "  Unknown backend \"{}\". Supported backends: \
                 chain_binomial, ode.",
                backend
            );
        }
        // Both parsed → a valid-but-unsupported pair, handled by validate_combo.
        (false, false) => {}
    }
    append_matrix_footer(&mut out);
    out
}

/// Append a reason/hint block, indenting each line two spaces under the header.
fn append_indented(out: &mut String, text: &str) {
    out.push_str("  ");
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            out.push_str("\n  ");
        }
        out.push_str(line);
    }
    out.push('\n');
}

/// The shared footer: the supported-pairs listing and the per-backend
/// statistical-object note. Identical across the invalid-pair and unknown-name
/// renderers so the two can never drift.
fn append_matrix_footer(out: &mut String) {
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
    algorithm: FitAlgorithm,
    correlated: bool,
    requested: Option<ObsAlignment>,
    obs_on_grid: bool,
) -> Result<ObsAlignment, String> {
    use FitAlgorithm as A;
    use ObsAlignment::{Exact, Snap};
    match algorithm {
        // Exact-steppers: land exactly on any obs. No `snap` inference path exists.
        A::If2 | A::Pfilter => match requested {
            None | Some(Exact) => Ok(Exact),
            Some(Snap) => Err(format!(
                "{algorithm}: obs_alignment = \"snap\" is not implemented — it steps \
                 exactly to observation times. Use \"exact\" (the default)."
            )),
        },
        A::Pmmh => match (requested, correlated, obs_on_grid) {
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
        A::Pgas => match requested {
            None | Some(Snap) => Ok(Snap),
            Some(Exact) => Err(
                "pgas: obs_alignment = \"exact\" is not yet implemented (PGAS uses a \
                 uniform grid; exact-PGAS is planned). Use \"snap\", or algorithm = \
                 if2 / pfilter for exact alignment."
                    .into(),
            ),
        },
        // ODE-backend algorithms never reach here — both call sites gate on the
        // PF algorithms (if2/pgas/pmmh/pfilter). The arm exists for exhaustiveness;
        // obs alignment is a particle-filter concept (ODE scores on the integrator
        // grid), so it is a clear error rather than a panic.
        A::NlSbplx | A::NlBobyqa | A::Mh | A::Nuts => Err(format!(
            "obs_alignment does not apply to the ODE algorithm '{algorithm}' — \
             observations are scored on the integrator grid."
        )),
    }
}

/// Does the model's `init { }` DRAW any compartment from a law
/// (`I ~ poisson(rate = I0)`), rather than computing every one of them from an
/// expression?
///
/// The one MODEL fact [`validate_ic_free`] needs, and the reason that check
/// cannot be settled from the algorithm alone. Carried as a named type rather
/// than a bare `bool` so a call site says which fact it is asserting; the
/// producer is `Model::initial_conditions` (`InitSpec::is_law`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitLaw {
    /// Every `init { }` entry is an expression. Every particle of a bootstrap
    /// swarm therefore starts at the SAME x₀.
    Absent,
    /// At least one entry is a law, so each particle draws its own x₀.
    Declared,
}

/// The `(algorithm × model × ic_free)` support check at the fit-dispatch seam
/// (F1).
///
/// `ic_free = true` requests IC-free / conditional-likelihood inference:
/// weight-and-resample at the first observation (pinning the initial state on
/// y₁) but drop y₁ from the accumulated log-likelihood. That estimand needs
/// **two** properties, and only the first is a property of the algorithm alone:
///
/// 1. **The algorithm drops the first increment.** Otherwise it silently
///    returns the UNCONDITIONAL likelihood while the startup banner claims
///    conditioning.
/// 2. **The particles differ in x₀.** The reweight at y₁ is what pins the
///    initial state; with every particle carrying the SAME x₀ that reweight
///    is a no-op — every particle scores identically — and `ic_free`
///    degenerates to *dropping* y₁ rather than conditioning on it.
///
/// `if2` has both for any model: it perturbs θ per particle at t=0 and each
/// particle then draws its own x₀ from its own θ (gh#364, pinned by
/// `sim/tests/gh364_if2_per_particle_initial_state.rs`) — which is what a
/// `perturb_only_at_t0 = true` parameter buys, and it buys it under IF2 only.
///
/// `pfilter` and plain `pmmh` have property 1 always and property 2 **only for
/// a model that declares an `init { }` law** (gh#732). Both run the bootstrap
/// particle filter, which draws x₀ per particle from that particle's own RNG
/// stream. Under a law those draws differ; under a deterministic `init { }`
/// they are all the same state, the swarm has no spread at t=0, and admitting
/// `ic_free` would do the exact thing this check exists to prevent. So the two
/// cells split on the MODEL, not on the algorithm.
///
/// Property 1 is why the rest are refused for every model:
///
///   * `pgas`                   — no conditioning field anywhere in `pgas.rs`.
///   * `nl-sbplx` / `nl-bobyqa` / `mh` / `nuts` — score via
///     `runner::compute_ode_loglik`, which sums over every obs time with no
///     skip.
///   * `pmmh` **with** `rho` (correlated PMMH) — routes to
///     `correlated_pf::bootstrap_filter_correlated`, which adds every
///     increment unconditionally. (That filter also refuses a declared
///     `init { }` law outright, for a reason of its own: its pre-drawn
///     correlated randoms cover the transition kernel only.)
///
/// Each is a hard error at config-load time naming its own reason —
/// converting a silent wrong answer into a loud failure. `correlated` is
/// `true` for a PMMH stage with `rho` set.
pub fn validate_ic_free(
    algorithm: FitAlgorithm,
    correlated: bool,
    init_law: InitLaw,
) -> Result<(), String> {
    use FitAlgorithm as A;
    match algorithm {
        // Drops the first increment, and gives each particle its own x₀ drawn
        // from its own perturbed θ — for any model.
        A::If2 => Ok(()),
        // Correlated PMMH fails property 1 as well, and for a different
        // reason than the bootstrap-PF cells — say which. Ordered before the
        // bootstrap-PF arms so `rho` decides first.
        A::Pmmh if correlated => Err(
            "ic_free = true is not supported with correlated PMMH (a `pmmh` \
             stage with `rho` set). The correlated particle filter \
             (correlated_pf) accumulates every observation's log-likelihood \
             increment unconditionally, so it would silently compute the \
             UNCONDITIONAL likelihood while reporting that it conditioned on \
             y₁.\n\n  \
             Drop `rho` to run plain PMMH, use `algorithm = if2`, or remove \
             `ic_free = true`."
                .into(),
        ),
        // gh#732. The bootstrap PF draws x₀ per particle, so a declared law
        // gives these two the spread the first reweight needs.
        A::Pfilter | A::Pmmh if init_law == InitLaw::Declared => Ok(()),
        // …and a deterministic `init { }` gives them none, because every
        // particle's draw returns the same state.
        A::Pfilter | A::Pmmh => Err(format!(
            "ic_free = true is not supported for this model with the \
             `{algorithm}` algorithm, because the model's `init {{ }}` computes \
             every compartment from an expression. ic_free conditions the \
             initial state on y₁ by weighting and resampling at the first \
             observation, which requires the particles to DIFFER in their \
             initial state — otherwise the first reweight scores every particle \
             identically and ic_free degenerates to silently dropping y₁ \
             instead of conditioning on it (gh#732).\n\n  \
             `{algorithm}` runs the bootstrap particle filter, which DOES draw \
             x₀ per particle — but a deterministic `init {{ }}` returns the same \
             state on every draw, so the swarm has no spread at t=0. Declaring \
             the initial state as a law gives it that spread:\n\n      \
             init {{ I ~ poisson(rate = I0) }}\n\n  \
             Otherwise use `algorithm = if2`, which perturbs θ per particle at \
             t=0 and lets each particle draw its own x₀ from its own θ \
             (gh#364), or remove `ic_free = true` from the fit."
        )),
        A::Pgas => Err(
            "ic_free = true is not supported with the `pgas` algorithm. PGAS \
             accumulates every observation's log-likelihood increment \
             unconditionally (no conditioning field exists in its CSMC / \
             ancestor-sampling path), so it would silently compute the \
             UNCONDITIONAL likelihood while reporting that it conditioned on \
             y₁.\n\n  \
             ic_free is honored by `if2`, and by `pfilter` / plain `pmmh` on a \
             model whose `init { }` declares a law.\n  \
             Use one of those, or remove `ic_free = true` from the fit."
                .into(),
        ),
        A::NlSbplx | A::NlBobyqa | A::Mh | A::Nuts => Err(format!(
            "ic_free = true is not supported with the `{algorithm}` algorithm \
             (ODE backend). The deterministic likelihood (compute_ode_loglik) \
             sums over every observation time with no first-observation skip, so \
             it would silently compute the UNCONDITIONAL likelihood while \
             reporting that it conditioned on y₁.\n\n  \
             ic_free is honored by `if2`, and by `pfilter` / plain `pmmh` on a \
             model whose `init {{ }}` declares a law.\n  \
             Use one of those, or remove `ic_free = true` from the fit."
        )),
    }
}

/// Can a stage running `algorithm` make sense of `perturb_only_at_t0`?
///
/// `if2` **reads** it: the flag is its perturbation schedule — "perturb this
/// parameter once at t=0 rather than again at every observation" (`if2.rs`,
/// the inner loop that skips exactly these entries).
///
/// `pfilter` **tolerates** it: it estimates nothing, it evaluates the
/// likelihood at a fixed θ, so the flag is as inert there as `rw_sd` or
/// `transform` are — inert, not wrong.
///
/// Everything else proposes θ from a kernel with no notion of "when", so there
/// is no schedule for the flag to modify and the declaration is read nowhere.
/// `nuts` is in this group: a parameter-estimating ODE sampler with no
/// perturbation schedule, i.e. the same case as `mh`.
///
/// The match is exhaustive on purpose — a new algorithm must decide.
pub fn stage_tolerates_perturb_only_at_t0(algorithm: FitAlgorithm) -> bool {
    use FitAlgorithm as A;
    match algorithm {
        A::If2 | A::Pfilter => true,
        A::Pgas | A::Pmmh | A::Mh | A::Nuts | A::NlSbplx | A::NlBobyqa => false,
    }
}

/// The `perturb_only_at_t0` support check (axis 3) — **fit-level, not
/// per-stage**.
///
/// The declaration is refused only when NO stage in the fit can use it. That
/// asymmetry is forced by the config's shape: `[estimate]` is global to the fit
/// while the algorithm is per stage, so the flag is a property of the fit and
/// has to be judged against the fit.
///
/// Judging it per stage was wrong, and wrong in the expensive direction. It
/// refused the ordinary scout-then-refine shape — an `if2` scout that needs the
/// flag, followed by a `pgas` posterior that ignores it — and left the user two
/// escapes, both worse than the status quo: drop the flag, which makes the IF2
/// scout perturb an initial-value parameter at every observation (exactly the
/// thing the flag exists to prevent), or split one fit into two.
///
/// The defect this check exists for is still caught. Under a fit with no stage
/// that can use it, the declaration is parsed, folded into the fit hash, and
/// read nowhere: a modeller writes it believing they have said something about
/// the initial state, and has said nothing. That fit is refused.
///
/// One cell is deliberately let through: a fit whose only stage is `pfilter`.
/// The flag is inert there too, but `pfilter` estimates nothing at all, so the
/// declaration is no more meaningful — and no less — than the `rw_sd` sitting
/// beside it. Refusing on that basis would be a rule about `pfilter` configs
/// generally, not about this flag.
pub fn validate_perturb_only_at_t0(
    stage_algorithms: &[FitAlgorithm],
    declared_on: &[&str],
) -> Result<(), String> {
    if stage_algorithms.iter().copied().any(stage_tolerates_perturb_only_at_t0) {
        return Ok(());
    }
    let stages: Vec<&str> = stage_algorithms.iter().map(|a| a.as_str()).collect();
    Err(format!(
        "perturb_only_at_t0 = true is declared on {}, but no stage in this fit \
         can use it. It is an IF2 perturbation schedule — \"perturb this \
         parameter at t=0 only, not again at every observation\" — and this \
         fit's stages ({}) have no perturbation schedule to modify, so the \
         declaration would be read nowhere and silently do nothing.\n\n  \
         Add an `if2` stage (the flag then applies to it, and the other stages \
         simply ignore it), or drop `perturb_only_at_t0 = true` from the \
         [estimate] entries — an initial-state parameter is estimated by these \
         algorithms like any other.",
        declared_on.join(", "),
        stages.join(", "),
    ))
}

pub fn check_model_capabilities(
    backend: InferenceBackend,
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
        InferenceBackend::ChainBinomial => {
            Capabilities::OVERDISPERSION | Capabilities::BALANCE | Capabilities::RUNTIME_DT
        }
        InferenceBackend::Ode => Capabilities::REAL_COMPARTMENTS | Capabilities::RUNTIME_DT,
    };
    // gh#122: a source that mixes a `deterministic(...)` exit with another exit
    // is unsupported on the stochastic (chain_binomial) inference producer — the
    // competing-risk draw would over-draw the source. This is a STRUCTURAL model
    // property (not a backend-feature bitflag), so it is checked here rather than
    // via `Capabilities`, which lets the error name the offending compartment and
    // transitions. ODE inference runs every transition as a deterministic flow
    // and is exempt. This is the single inference chokepoint every stochastic
    // fit stage / profile / survey routes through.
    if matches!(backend, InferenceBackend::ChainBinomial) {
        compiled
            .validate_deterministic_source_exits()
            .map_err(|e| e.to_string())?;
        // gh#121: a multi-source stochastic transition (`A + B --> C`) is bounded
        // by only its first source on the chain-binomial producer, driving the
        // secondary source negative. Same structural-model rejection as gh#122
        // (not a Capabilities bitflag, so the error can name the transition and
        // its source compartments). ODE inference runs each transition as a
        // flow and is exempt.
        compiled
            .validate_single_source_transitions()
            .map_err(|e| e.to_string())?;
    }

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

/// gh#166 B2: a model that references the step size in a rate (`Expr::Dt` /
/// RUNTIME_DT) keeps the FIRST-ORDER Euler incidence on the ODE backend — the
/// high-order augmented flow (Q1B) is undefined when a rate depends on the step
/// size. This is allowed and not a capability error, but it silently degrades
/// incidence accuracy relative to every other model, so it is surfaced LOUDLY,
/// once per invocation, by the ODE dispatch chokepoints (`util::run_simulation`
/// for `simulate`, `gate_run_stages_against_model` for `fit`). No-op unless the
/// model is RUNTIME_DT (callers gate on the ODE backend).
pub fn warn_if_ode_euler_flow(compiled: &sim::CompiledModel) {
    if compiled
        .required_capabilities()
        .contains(sim::Capabilities::RUNTIME_DT)
    {
        eprintln!(
            "\x1b[33m⚠ model references `dt` in a rate (Expr::Dt): on the ODE \
             backend its incidence is computed with the first-order Euler method \
             — the high-order augmented flow is undefined when a rate depends on \
             the step size. `dt` in a rate is a discrete-time construct; consider \
             whether it belongs on the continuous ODE backend (every other model \
             gets high-order incidence).\x1b[0m"
        );
    }
}

/// gh#95: the predicate behind [`warn_if_gillespie_time_dep`] — true iff the
/// model has at least one transition whose rate is time-varying (a `TimeFunc`
/// forcing, a bare `t`, or anything that transitively reads one; the same set
/// Gillespie must re-evaluate as time advances, `time_dep_transitions`).
/// Factored out so the classification is unit-testable without capturing stderr.
pub(crate) fn gillespie_time_dep_warn(compiled: &sim::CompiledModel) -> bool {
    !compiled.time_dep_transitions.is_empty()
}

/// gh#95: Gillespie's next-event draw holds the total propensity CONSTANT over
/// each exponential inter-event wait, so a time-varying rate is effectively
/// frozen within the wait and only refreshed at grid boundaries — a
/// piecewise-constant approximation to the true inhomogeneous Poisson process
/// (seasonal forcing, a bare-`t` ramp, importation forcing). A fine output grid
/// shrinks the bias, so this is surfaced as a WARNING rather than a hard error
/// (mirrors `warn_if_ode_euler_flow`). Emitted once, at the gillespie forward
/// dispatch chokepoint. No-op unless the model has a time-varying rate.
pub fn warn_if_gillespie_time_dep(compiled: &sim::CompiledModel) {
    if gillespie_time_dep_warn(compiled) {
        eprintln!(
            "\x1b[33m⚠ model has time-varying transition rate(s): on the \
             gillespie backend the next-event draw holds the total rate constant \
             over each exponential wait, so a time-varying rate is treated as \
             piecewise-constant on the output grid — biasing the inhomogeneous \
             Poisson process. Mitigations: use a fine output grid (smaller steps \
             → smaller bias), or prefer backend = \"chain_binomial\", which \
             re-evaluates the rate every substep. (gh#95)\x1b[0m"
        );
    }
}

/// Per-capability hint text for the unsupported-capability error. Keyed on the
/// `Capabilities` flag; `name` is the bitflags constant name (used as the
/// non-blank fallback for any flag without bespoke guidance, so the message
/// can never be empty — gh#192).
pub(crate) fn capability_hint(name: &str, flag: sim::Capabilities) -> String {
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
        Capabilities::REACTIVE_INTERVENTIONS =>
            "REACTIVE_INTERVENTIONS: the model has a `reactive_interventions{}` \
             policy (a state/observation-triggered campaign). It is parsed and \
             validated, but the reactive agenda is not yet executed by any \
             backend (gh#204) — running it would silently drop the policy. \
             Remove the reactive policy, or replace it with an equivalent fixed \
             schedule (`interventions {}` with `at [...]`) for now.".to_string(),
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
            METHODS.iter().filter(|m| m.backend.as_str() == be_name).collect();
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

    /// gh#122: a source that mixes a `deterministic(...)` exit with another exit
    /// is rejected on the chain_binomial inference producer (the over-draw
    /// hazard) with a located, gh#122-tagged message, but still accepted on the
    /// ODE inference backend (which runs every transition as a deterministic
    /// flow). Proves the inference dispatch gate is wired to the shared
    /// `CompiledModel::validate_deterministic_source_exits`.
    #[test]
    fn mixed_deterministic_source_rejected_on_chain_binomial_inference_only() {
        use ir::{
            expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
            model::{
                Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
                RegularOutputSchedule, SimulationConfig,
            },
            parameter::{ParamValue, Parameter},
            transition::{DrawMethod, StoichiometryEntry, Transition},
            Model,
        };
        use std::collections::HashMap;

        let mul = |l: Expr, r: Expr| {
            Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(l), right: Box::new(r) },
            })
        };
        let rate = |p: &str, c: &str| mul(
            Expr::Param(ParamExpr { param: p.into() }),
            Expr::Pop(PopExpr { pop: c.into() }),
        );
        let tr = |name: &str, dst: &str, dm: DrawMethod, r: Expr| Transition {
            rate_state_grad: Default::default(),
            name: name.into(),
            stoichiometry: vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry(dst.into(), 1)],
            rate: r,
            metadata: None,
            draw_method: dm,
            rate_grad: Default::default(),
            lineage: None,
        };

        // Source `I` mixes a deterministic recovery with a Poisson death.
        let model = Model {
            ic_grad: Default::default(),
            name: "mixed_source_infer".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: ["I", "R", "D"]
                .iter()
                .map(|c| Compartment { name: (*c).into(), kind: CompartmentKind::Integer })
                .collect(),
            transitions: vec![
                tr("recover", "R", DrawMethod::Deterministic, rate("gamma", "I")),
                tr("die", "D", DrawMethod::Poisson, rate("mu", "I")),
            ],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![
                Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None },
                Parameter { name: "mu".into(), value: ParamValue::Fixed { value: 0.02 }, param_kind: None, param_dim: None },
            ],
            initial_conditions: InitialConditions::constants(
                [("I", 1000.0), ("R", 0.0), ("D", 0.0)]
                    .iter()
                    .map(|(k, v)| ((*k).into(), *v))
                    .collect::<HashMap<String, f64>>(),
            ),
            output: OutputConfig {
                times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 5.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(7), integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
            quantities: vec![],
            contrasts: vec![],
        };
        let compiled = sim::CompiledModel::new(model).expect("mixed model still compiles");

        let err = check_model_capabilities(InferenceBackend::ChainBinomial, &compiled)
            .expect_err("chain_binomial inference must reject a mixed deterministic source");
        assert!(err.contains("gh#122"), "must cite the issue: {err}");
        assert!(err.contains("I"), "must name the source compartment: {err}");
        assert!(err.contains("recover"), "must name the deterministic exit: {err}");

        // ODE inference integrates every transition as a flow — still accepted.
        assert!(
            check_model_capabilities(InferenceBackend::Ode, &compiled).is_ok(),
            "ODE inference must accept a mixed deterministic source"
        );
    }

    /// gh#95: `gillespie_time_dep_warn` is TRUE for a model with a time-varying
    /// transition rate (here a bare `t` factor) and FALSE for a time-free model.
    /// This is the predicate behind the gillespie inhomogeneous-Poisson warning.
    #[test]
    fn gillespie_time_dep_warn_detects_time_varying_rate() {
        use ir::{
            expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, TimeExpr},
            model::{
                Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
                RegularOutputSchedule, SimulationConfig,
            },
            parameter::{ParamValue, Parameter},
            transition::{DrawMethod, StoichiometryEntry, Transition},
            Model,
        };
        use std::collections::HashMap;

        let mul = |l: Expr, r: Expr| {
            Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(l), right: Box::new(r) },
            })
        };
        // Build a one-transition S --> I model whose infection rate is `rate`.
        let build = |name: &str, rate: Expr| -> sim::CompiledModel {
            let model = Model {
                ic_grad: Default::default(),
                name: name.into(),
                version: "0.3".into(),
                time_unit: "days".into(),
                description: None,
                origin: None,
                origin_rata_die: None,
                compartments: ["S", "I"]
                    .iter()
                    .map(|c| Compartment { name: (*c).into(), kind: CompartmentKind::Integer })
                    .collect(),
                transitions: vec![Transition {
                    rate_state_grad: Default::default(),
                    name: "infect".into(),
                    stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                    rate,
                    metadata: None,
                    draw_method: DrawMethod::Poisson,
                    rate_grad: Default::default(),
                    lineage: None,
                }],
                ode_equations: vec![],
                time_functions: vec![],
                tables: vec![],
                interventions: vec![],
                observations: vec![],
                bindings: vec![],
                per_eval_bindings: vec![],
                parameters: vec![Parameter {
                    name: "beta".into(), value: ParamValue::Fixed { value: 0.001 },
                    param_kind: None, param_dim: None,
                }],
                initial_conditions: InitialConditions::constants(
                    [("S", 990.0), ("I", 10.0)].iter().map(|(k, v)| ((*k).into(), *v))
                        .collect::<HashMap<String, f64>>(),
                ),
                output: OutputConfig {
                    times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
                    format: "tsv".into(), trajectory: true, observations: false,
                },
                simulation: SimulationConfig {
                    t_start: 0.0, t_end: 5.0, time_semantics: "continuous".into(),
                    dt: Some(1.0), rng_seed: Some(7), integrator: Default::default(),
                    t_end_anchor: None,
                },
                presets: vec![],
                model_structure: None,
                balance: None,
                identity_tracked_compartments: vec![],
                quantities: vec![],
                contrasts: vec![],
            };
            sim::CompiledModel::new(model).expect("compile")
        };

        let beta_i = || mul(
            Expr::Param(ParamExpr { param: "beta".into() }),
            Expr::Pop(PopExpr { pop: "I".into() }),
        );
        // Time-varying: rate = beta * I * t.
        let time_varying = build("tvar", mul(beta_i(), Expr::Time(TimeExpr { time: () })));
        assert!(
            gillespie_time_dep_warn(&time_varying),
            "a rate with a bare `t` factor must be flagged time-varying (gh#95)"
        );
        // Time-free control: rate = beta * I.
        let time_free = build("tfree", beta_i());
        assert!(
            !gillespie_time_dep_warn(&time_free),
            "a time-free rate must NOT be flagged"
        );
    }

    #[test]
    fn every_phase1_method_present() {
        for (a, b) in [
            (FitAlgorithm::If2, InferenceBackend::ChainBinomial),
            (FitAlgorithm::Pgas, InferenceBackend::ChainBinomial),
            (FitAlgorithm::Pmmh, InferenceBackend::ChainBinomial),
            (FitAlgorithm::Pfilter, InferenceBackend::ChainBinomial),
            (FitAlgorithm::NlSbplx, InferenceBackend::Ode),
            (FitAlgorithm::NlBobyqa, InferenceBackend::Ode),
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
        for algo in [FitAlgorithm::If2, FitAlgorithm::Pfilter] {
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
        assert_eq!(resolve_obs_alignment(FitAlgorithm::Pmmh, false, None, false), Ok(Exact));
        // Correlated PMMH (rho set): exact OK on-grid...
        assert_eq!(resolve_obs_alignment(FitAlgorithm::Pmmh, true, None, true), Ok(Exact));
        // ...but off-grid + correlated is a CLEAN ERROR (was silent fresh-RNG
        // decorrelation, #17), under both default and explicit exact.
        assert!(resolve_obs_alignment(FitAlgorithm::Pmmh, true, None, false).is_err());
        assert!(resolve_obs_alignment(FitAlgorithm::Pmmh, true, Some(Exact), false).is_err());
    }

    #[test]
    fn obs_alignment_pgas_snap_only_exact_is_clean_error() {
        use ObsAlignment::{Exact, Snap};
        // PGAS defaults to snap (its only mode today)...
        assert_eq!(resolve_obs_alignment(FitAlgorithm::Pgas, false, None, true), Ok(Snap));
        assert_eq!(resolve_obs_alignment(FitAlgorithm::Pgas, false, Some(Snap), true), Ok(Snap));
        // ...and exact is a CLEAN ERROR (was a silent snap), naming the fix.
        let err = resolve_obs_alignment(FitAlgorithm::Pgas, false, Some(Exact), true).unwrap_err();
        assert!(err.contains("not yet implemented"), "should name exact-PGAS as unimplemented: {err}");
        assert!(err.contains("if2") || err.contains("snap"), "should suggest a fix: {err}");
    }

    // ── ic_free / conditioning support check (F1) ──────────────────────────
    //
    // `ic_free = true` (IC-free / conditional likelihood) needs BOTH: an
    // algorithm that drops y₁ from the accumulated loglik, and particles that
    // differ in x₀. PGAS, the ODE algorithms (via `compute_ode_loglik`) and
    // correlated PMMH (via `bootstrap_filter_correlated`) fail the first for
    // every model. `pfilter` and plain `pmmh` pass the first always and the
    // second only when the MODEL declares an `init { }` law — the bootstrap PF
    // draws x₀ per particle, and a deterministic `init { }` returns the same
    // state on every draw (gh#732). Those are the four cells below.

    #[test]
    fn ic_free_is_accepted_for_if2_under_either_model() {
        // IF2 perturbs θ per particle at t=0 and each particle draws its own
        // x₀ from its own θ (gh#364), so its spread does not depend on the
        // model's `init { }`.
        for law in [InitLaw::Absent, InitLaw::Declared] {
            assert!(validate_ic_free(FitAlgorithm::If2, false, law).is_ok(),
                "if2 must be admitted with init law {law:?}");
        }
    }

    /// gh#732, the cell this change OPENS. The bootstrap particle filter draws
    /// x₀ per particle from that particle's own stream, so a model whose
    /// `init { }` declares a law gives the swarm real spread at t=0 — which is
    /// exactly the property `ic_free`'s first reweight needs.
    #[test]
    fn ic_free_bootstrap_pf_cells_are_admitted_when_the_model_draws_x0() {
        for algo in [FitAlgorithm::Pfilter, FitAlgorithm::Pmmh] {
            assert!(
                validate_ic_free(algo, false, InitLaw::Declared).is_ok(),
                "{algo} with a declared init law must be admitted: {:?}",
                validate_ic_free(algo, false, InitLaw::Declared)
            );
        }
    }

    /// gh#732, the cell that stays SHUT. With a deterministic `init { }` every
    /// particle's draw returns the same state, so the first reweight scores
    /// them identically and `ic_free` degenerates to dropping y₁. Refused, and
    /// the refusal must say WHY and how to fix it — not merely that it refused.
    #[test]
    fn ic_free_bootstrap_pf_cells_are_refused_on_a_deterministic_init() {
        for algo in [FitAlgorithm::Pfilter, FitAlgorithm::Pmmh] {
            let err = validate_ic_free(algo, false, InitLaw::Absent).unwrap_err();
            assert!(err.contains("ic_free"), "{algo}: must name ic_free: {err}");
            assert!(err.contains(algo.as_str()),
                "{algo}: must name the offending algorithm: {err}");
            // The REASON, not just the refusal.
            assert!(err.contains("bootstrap particle filter"),
                "{algo}: must name the mechanism that fails: {err}");
            assert!(err.contains("no spread at t=0"),
                "{algo}: must say what is missing: {err}");
            assert!(err.contains("gh#732"),
                "{algo}: must cite the issue: {err}");
            // …and the two ways out, the model one first.
            assert!(err.contains("poisson(rate = I0)"),
                "{algo}: must show the init-law form that fixes it: {err}");
            assert!(err.contains("if2"),
                "{algo}: must name the algorithm that works regardless: {err}");
        }
    }

    #[test]
    fn ic_free_pgas_is_hard_error_naming_the_limitation() {
        // PGAS fails property 1 (it scores every obs), which no `init { }` law
        // changes — so both model shapes are refused.
        for law in [InitLaw::Absent, InitLaw::Declared] {
            let err = validate_ic_free(FitAlgorithm::Pgas, false, law).unwrap_err();
            assert!(err.contains("ic_free"), "must name ic_free: {err}");
            assert!(err.contains("pgas"), "must name the offending algorithm: {err}");
            // Points the user at the supported alternative.
            assert!(err.contains("if2"), "must name a supported cell: {err}");
        }
    }

    #[test]
    fn ic_free_ode_mle_is_hard_error() {
        // Both NLopt deterministic optimizers score every obs via
        // compute_ode_loglik — conditioning is silently ignored.
        for algo in [FitAlgorithm::NlSbplx, FitAlgorithm::NlBobyqa] {
            let err = validate_ic_free(algo, false, InitLaw::Declared).unwrap_err();
            assert!(err.contains("ic_free"), "{algo}: must name ic_free: {err}");
            assert!(err.contains(algo.as_str()), "{algo}: must name the algorithm: {err}");
        }
    }

    #[test]
    fn ic_free_correlated_pmmh_names_the_correlated_reason() {
        // Correlated PMMH (rho set) routes to bootstrap_filter_correlated,
        // which adds every increment unconditionally. That refusal is
        // independent of the model: `rho` decides before the init law does, so
        // a declared law does NOT open this cell, and the message must still
        // be the rho one — a user who unsets `rho` must not be told to unset
        // it again.
        for law in [InitLaw::Absent, InitLaw::Declared] {
            let err = validate_ic_free(FitAlgorithm::Pmmh, true, law).unwrap_err();
            assert!(err.contains("ic_free"), "must name ic_free: {err}");
            assert!(
                err.contains("correlated") || err.contains("rho"),
                "must name the correlated/rho condition: {err}"
            );
        }
        // …and plain PMMH on the same deterministic model is refused for the
        // OTHER reason, so the two messages stay distinguishable.
        let plain = validate_ic_free(FitAlgorithm::Pmmh, false, InitLaw::Absent).unwrap_err();
        assert!(plain.contains("bootstrap particle filter"),
            "plain PMMH is refused for the missing t=0 spread, not for rho: {plain}");
    }

    // ── perturb_only_at_t0 (axis 3, fit-level) ─────────────────────────────
    //
    // The flag is refused only when NO stage in the fit can use it. `[estimate]`
    // is global to the fit while the algorithm is per stage, so a scout-then-
    // refine pipeline that declares the flag for its `if2` scout must be
    // accepted even though its `pgas` posterior ignores it.

    use FitAlgorithm as FA;

    #[test]
    fn perturb_only_at_t0_stage_predicate_is_if2_and_pfilter() {
        assert!(stage_tolerates_perturb_only_at_t0(FA::If2),
            "IF2 reads the flag — it is IF2's own perturbation schedule");
        assert!(stage_tolerates_perturb_only_at_t0(FA::Pfilter),
            "pfilter estimates nothing; the flag is inert there, not wrong");
        for algo in [FA::Pgas, FA::Pmmh, FA::Mh, FA::Nuts, FA::NlSbplx, FA::NlBobyqa] {
            assert!(!stage_tolerates_perturb_only_at_t0(algo),
                "{algo} has no perturbation schedule for the flag to modify");
        }
    }

    /// The case the per-stage rule got wrong: an `if2` scout that needs the
    /// flag, followed by a `pgas` posterior that ignores it. One stage can use
    /// it, so the fit is accepted.
    #[test]
    fn perturb_only_at_t0_accepted_when_any_stage_is_if2() {
        validate_perturb_only_at_t0(&[FA::If2, FA::Pgas], &["I0"])
            .expect("if2 scout + pgas posterior must be accepted");
        validate_perturb_only_at_t0(&[FA::If2, FA::Pmmh, FA::Pfilter], &["I0"])
            .expect("the if2 stage is enough, wherever it sits in the pipeline");
    }

    /// ...but a fit where the declaration genuinely does nothing is still
    /// refused, and the message says it needs an `if2` stage.
    #[test]
    fn perturb_only_at_t0_refused_when_no_stage_can_use_it() {
        let err = validate_perturb_only_at_t0(&[FA::Pgas], &["I0"]).unwrap_err();
        assert!(err.contains("perturb_only_at_t0"), "must name the flag: {err}");
        assert!(err.contains("no stage in this fit"),
            "must say the refusal is about the FIT, not one stage: {err}");
        assert!(err.contains("`if2`"),
            "must name the stage kind that would give the flag meaning: {err}");
        assert!(err.contains("I0"),
            "must name the parameter that declared it: {err}");
        assert!(err.contains("perturbation schedule"),
            "must say WHY, not just that it refused: {err}");

        // Every non-tolerating algorithm, alone, is refused.
        for algo in [FA::Pgas, FA::Pmmh, FA::Mh, FA::Nuts, FA::NlSbplx, FA::NlBobyqa] {
            validate_perturb_only_at_t0(&[algo], &["I0"])
                .expect_err("{algo} alone cannot use the flag");
        }
        // ...and so is a pipeline built only from them.
        validate_perturb_only_at_t0(&[FA::Pgas, FA::Pmmh], &["I0"])
            .expect_err("no if2 anywhere in the pipeline");
    }

    /// A `pfilter`-only fit is let through. The flag is inert there, but
    /// `pfilter` estimates nothing at all, so refusing on that basis would be a
    /// rule about pfilter configs rather than about this flag.
    #[test]
    fn perturb_only_at_t0_pfilter_only_fit_is_tolerated() {
        validate_perturb_only_at_t0(&[FA::Pfilter], &["I0"])
            .expect("pfilter tolerates the flag; see stage_tolerates_perturb_only_at_t0");
    }

    #[test]
    fn invalid_pf_method_on_ode_names_nlopt_alternative() {
        let err = validate_combo(FitAlgorithm::If2, InferenceBackend::Ode).unwrap_err();
        assert!(err.contains("nl-sbplx"), "message should suggest nl-sbplx; got:\n{err}");
        assert!(err.contains("MLE on the ODE backend"));
    }

    #[test]
    fn invalid_nlopt_on_chain_binomial_names_if2() {
        let err = validate_combo(FitAlgorithm::NlSbplx, InferenceBackend::ChainBinomial).unwrap_err();
        assert!(err.contains("if2"), "message should suggest if2; got:\n{err}");
    }

    #[test]
    fn unknown_algorithm_yields_clear_error() {
        let err = parse_combo("not-a-method", "ode").unwrap_err();
        assert!(err.contains("Unknown algorithm"), "got:\n{err}");
    }

    #[test]
    fn unknown_backend_yields_clear_error() {
        let err = parse_combo("if2", "not-a-backend").unwrap_err();
        assert!(err.contains("Unknown backend"), "got:\n{err}");
    }

    #[test]
    fn nuts_on_ode_is_supported() {
        // `nuts` is a real method on the ODE backend (gh#275 Phase 2): it parses
        // to `FitAlgorithm::Nuts` and validates to the registry entry.
        let m = parse_combo("nuts", "ode").expect("nuts+ode must be a supported method");
        assert_eq!(m.algorithm, FitAlgorithm::Nuts);
        assert_eq!(m.backend, InferenceBackend::Ode);
    }

    #[test]
    fn nuts_on_stochastic_backend_steers_to_pgas() {
        // `nuts` now *parses* (it is a known algorithm), so the stochastic-backend
        // rejection is a valid-but-unsupported *typed pair* handled by
        // `validate_combo` — NOT a bare "unknown algorithm" at the parse boundary.
        // The tailored steer-to-pgas hint must survive there: vanilla NUTS has no
        // closed-form gradient under PF wrapping, so the user is pointed at PGAS.
        let err = parse_combo("nuts", "chain_binomial").unwrap_err();
        assert!(
            !err.contains("Unknown algorithm"),
            "nuts parses now — must NOT degrade to an unknown-algorithm error: {err}"
        );
        assert!(err.contains("NUTS"), "carries the tailored NUTS explanation: {err}");
        assert!(err.contains("pgas"), "points the user at the supported alternative: {err}");
    }

    #[test]
    fn parse_combo_round_trips_every_registry_pair() {
        // The string boundary parses every canonical (algorithm, backend) wire
        // spelling back to its typed registry entry — pins `parse ≡ as_str`.
        for m in METHODS {
            let parsed = parse_combo(m.algorithm.as_str(), m.backend.as_str())
                .unwrap_or_else(|e| panic!("registry pair must parse: {e}"));
            assert_eq!(parsed.algorithm, m.algorithm);
            assert_eq!(parsed.backend, m.backend);
        }
    }

    #[test]
    fn rejection_message_lists_full_matrix() {
        let err = validate_combo(FitAlgorithm::If2, InferenceBackend::Ode).unwrap_err();
        for m in METHODS {
            assert!(
                err.contains(m.algorithm.as_str()),
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
            status_note(FitAlgorithm::Pmmh, InferenceBackend::ChainBinomial).is_some_and(|n| n.contains("T > 500")),
            "PMMH (stable, production) must still surface its prefer-PGAS-for-long-series guidance at runtime"
        );
        // The bug this closes: Beta NLopt caveats never reached runtime before
        // — only PMMH had a hand-coded banner.
        assert!(
            status_note(FitAlgorithm::NlSbplx, InferenceBackend::Ode).is_some_and(|n| n.contains("Phase 1")),
            "Beta nl-sbplx caveat must surface at runtime, not just in `fit methods`"
        );
        assert!(
            status_note(FitAlgorithm::NlBobyqa, InferenceBackend::Ode).is_some(),
            "Beta nl-bobyqa carries a caveat"
        );
        // Stable methods: no banner.
        assert_eq!(status_note(FitAlgorithm::If2, InferenceBackend::ChainBinomial), None);
        assert_eq!(status_note(FitAlgorithm::Pgas, InferenceBackend::ChainBinomial), None);
        assert_eq!(status_note(FitAlgorithm::Pfilter, InferenceBackend::ChainBinomial), None);
        // Unregistered pair: no banner (validate_combo emits the hard error).
        assert_eq!(status_note(FitAlgorithm::Pgas, InferenceBackend::Ode), None);
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
        let err = check_model_capabilities(InferenceBackend::ChainBinomial, &compiled)
            .expect_err("chain_binomial inference must reject real-coupled models");
        assert!(err.contains("gh#191"), "should cite the tracking issue: {err}");
        assert!(err.contains("frozen"), "should explain the frozen-reservoir reason: {err}");
        // ode integrates real compartments — still accepted.
        assert!(check_model_capabilities(InferenceBackend::Ode, &compiled).is_ok());
    }

    /// gh#204: the inference capability gate rejects an active reactive policy on
    /// EVERY inference backend (no backend executes the reactive agenda yet), so
    /// fit / pfilter never silently drops the policy. Uses the committed reactive
    /// golden as the model under test.
    #[test]
    fn reactive_policy_rejected_on_inference_backends() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json"
        );
        let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let env: serde_json::Value = serde_json::from_str(&json).expect("parse reactive envelope");
        let mut model: ir::Model =
            serde_json::from_value(env["model"].clone()).expect("deserialize reactive model");
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                p.value = p.value.with_value(0.5);
            }
        }
        let compiled = sim::CompiledModel::new(model).expect("compile reactive golden");
        assert!(
            compiled
                .required_capabilities()
                .contains(sim::Capabilities::REACTIVE_INTERVENTIONS),
            "fixture must carry a reactive fire source"
        );
        for backend in [InferenceBackend::ChainBinomial, InferenceBackend::Ode] {
            let err = check_model_capabilities(backend, &compiled)
                .expect_err("inference must reject an active reactive policy");
            assert!(
                err.contains("REACTIVE_INTERVENTIONS"),
                "{backend:?} rejection must name the capability: {err}"
            );
        }
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
        check_model_capabilities(InferenceBackend::ChainBinomial, &compiled).unwrap_or_else(|e| {
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
        let err = check_model_capabilities(InferenceBackend::Ode, &compiled)
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
        check_model_capabilities(InferenceBackend::ChainBinomial, &compiled).unwrap_or_else(|e| {
            panic!("chain_binomial inference must ACCEPT dt-in-rate models: {e}")
        });
        check_model_capabilities(InferenceBackend::Ode, &compiled).unwrap_or_else(|e| {
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
