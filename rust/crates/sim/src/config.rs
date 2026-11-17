/// Config for the Gillespie (exact SSA) backend.
/// No `dt` — time steps are drawn from exponential distributions.
#[derive(Debug, Clone)]
pub struct GillespieConfig {
    pub t_start: f64,
    pub t_end: f64,
    /// How often to record output. If None, record at every event.
    pub output_dt: Option<f64>,
}

/// Config for the discrete-time chain-binomial backend.
#[derive(Debug, Clone)]
pub struct ChainBinomialConfig {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
    /// The algorithm every binomial draw of a fresh run uses (gh#747): the
    /// transitions and the initial-state draw. Both draw exactly Binomial(n, p),
    /// but from different random numbers, so it is part of what a run is. A run
    /// resumed with a saved generator keeps that generator's algorithm.
    pub binomial: crate::rng::BinomialAlgorithm,
}

/// Config for the deterministic ODE (RK4) backend.
/// Seed is accepted at the call site but ignored — runs are deterministic.
#[derive(Debug, Clone)]
pub struct OdeConfig {
    pub t_start: f64,
    pub t_end: f64,
    pub dt: f64,
}

/// Dispatch enum — lets cross-backend test loops iterate over all backends.
/// Use the type system: `dt` only exists in the variants that need it.
#[derive(Debug, Clone)]
pub enum SimConfig {
    Gillespie(GillespieConfig),
    ChainBinomial(ChainBinomialConfig),
    Ode(OdeConfig),
}

impl SimConfig {
    pub fn t_start(&self) -> f64 {
        match self {
            SimConfig::Gillespie(c) => c.t_start,
            SimConfig::ChainBinomial(c) => c.t_start,
            SimConfig::Ode(c) => c.t_start,
        }
    }

    pub fn t_end(&self) -> f64 {
        match self {
            SimConfig::Gillespie(c) => c.t_end,
            SimConfig::ChainBinomial(c) => c.t_end,
            SimConfig::Ode(c) => c.t_end,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            SimConfig::Gillespie(_) => "Gillespie",
            SimConfig::ChainBinomial(_) => "ChainBinomial",
            SimConfig::Ode(_) => "Ode",
        }
    }
}
