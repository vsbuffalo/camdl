//! Tallies for silent fallback paths in expression evaluation.
//!
//! RM2 in 2026-04-19 engine review: eval_resolved, eval_expr, and
//! rng all have silent fallback paths (Div-by-zero → 0, Pow-NaN → 0,
//! NegBinomial degenerate → Poisson, etc.). Logging on each hit is
//! either ignored (default log level) or a firehose for inference
//! runs with millions of steps. Atomic counters give a cheap summary
//! the caller can check at sim end: if the count is non-zero, the
//! model hit a degenerate regime the user should know about.
//!
//! Counters are process-global. Callers that care about per-sim
//! isolation should snapshot at start and diff at end.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub static DIV_BY_ZERO:       AtomicU64 = AtomicU64::new(0);
pub static POW_NAN_INF:       AtomicU64 = AtomicU64::new(0);
pub static UNOP_NAN:          AtomicU64 = AtomicU64::new(0);
pub static NEG_BINOMIAL_POIS: AtomicU64 = AtomicU64::new(0);
pub static BINOMIAL_FALLBACK: AtomicU64 = AtomicU64::new(0);
/// gh#619: a likelihood argument (mean, rate, p, …) evaluated to NaN during
/// observation *sampling* — typically 0/0 from a collapsed-compartment
/// denominator in a degenerate prior draw. The sampler draws 0 (the honest
/// dead-epidemic value) instead of panicking or handing NaN to rand_distr.
pub static OBS_SAMPLE_NAN:    AtomicU64 = AtomicU64::new(0);
/// gh#127 (#12): out-of-range table lookups hit by the fast evaluator. Unlike
/// the others this is NOT a silent-fallback path — the lookup returns NaN and
/// the run hard-errors at the `eval_propensities` boundary — but it is counted
/// here for the same end-of-run summary so a user sees an OOB happened.
pub static TABLE_OOB:         AtomicU64 = AtomicU64::new(0);
/// A `balance {}` target the substep left negative (`lifecycle.rs`): the
/// model is inconsistent at these parameters. The run continues with the
/// negative count, so the summary is where the user learns it happened.
pub static BALANCE_NEGATIVE:  AtomicU64 = AtomicU64::new(0);

/// gh#audit-C6 / S1. Process-global opt-in for the legacy silent-zero
/// behaviour in eval_expr. Default false → numerical-collapse paths
/// produce SimError::NumericalCollapse; the CLI sets this true if
/// `--allow-degenerate-rates` is present. Counters above still
/// increment under either mode so the user sees how often the
/// fallback fired. Process-global mirrors the surrounding atomic-
/// counter pattern; tests that exercise both modes must save / restore
/// or run serially.
pub static ALLOW_DEGENERATE_RATES: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn allow_degenerate_rates() -> bool {
    ALLOW_DEGENERATE_RATES.load(Ordering::Relaxed)
}

#[inline]
pub fn set_allow_degenerate_rates(allow: bool) {
    ALLOW_DEGENERATE_RATES.store(allow, Ordering::Relaxed);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvalStats {
    pub div_by_zero:       u64,
    pub pow_nan_inf:       u64,
    pub unop_nan:          u64,
    pub neg_binomial_pois: u64,
    pub binomial_fallback: u64,
    pub obs_sample_nan:    u64,
    pub table_oob:         u64,
    pub balance_negative:  u64,
}

impl EvalStats {
    pub fn snapshot() -> Self {
        EvalStats {
            div_by_zero:       DIV_BY_ZERO.load(Ordering::Relaxed),
            pow_nan_inf:       POW_NAN_INF.load(Ordering::Relaxed),
            unop_nan:          UNOP_NAN.load(Ordering::Relaxed),
            neg_binomial_pois: NEG_BINOMIAL_POIS.load(Ordering::Relaxed),
            binomial_fallback: BINOMIAL_FALLBACK.load(Ordering::Relaxed),
            obs_sample_nan:    OBS_SAMPLE_NAN.load(Ordering::Relaxed),
            table_oob:         TABLE_OOB.load(Ordering::Relaxed),
            balance_negative:  BALANCE_NEGATIVE.load(Ordering::Relaxed),
        }
    }

    pub fn diff_since(&self, earlier: &Self) -> Self {
        EvalStats {
            div_by_zero:       self.div_by_zero.saturating_sub(earlier.div_by_zero),
            pow_nan_inf:       self.pow_nan_inf.saturating_sub(earlier.pow_nan_inf),
            unop_nan:          self.unop_nan.saturating_sub(earlier.unop_nan),
            neg_binomial_pois: self.neg_binomial_pois.saturating_sub(earlier.neg_binomial_pois),
            binomial_fallback: self.binomial_fallback.saturating_sub(earlier.binomial_fallback),
            obs_sample_nan:    self.obs_sample_nan.saturating_sub(earlier.obs_sample_nan),
            table_oob:         self.table_oob.saturating_sub(earlier.table_oob),
            balance_negative:  self.balance_negative.saturating_sub(earlier.balance_negative),
        }
    }

    pub fn total(&self) -> u64 {
        self.div_by_zero + self.pow_nan_inf + self.unop_nan
            + self.neg_binomial_pois + self.binomial_fallback
            + self.obs_sample_nan + self.table_oob + self.balance_negative
    }
}

#[inline]
pub fn inc_div_by_zero()       { DIV_BY_ZERO.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_pow_nan_inf()       { POW_NAN_INF.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_unop_nan()          { UNOP_NAN.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_neg_binomial_pois() { NEG_BINOMIAL_POIS.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_binomial_fallback() { BINOMIAL_FALLBACK.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_obs_sample_nan()    { OBS_SAMPLE_NAN.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_table_oob()         { TABLE_OOB.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_balance_negative()  { BALANCE_NEGATIVE.fetch_add(1, Ordering::Relaxed); }

impl std::fmt::Display for EvalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "eval-stats summary (counts of fallback paths hit during this run):")?;
        if self.div_by_zero       > 0 { writeln!(f, "  div_by_zero:       {}", self.div_by_zero)?; }
        if self.pow_nan_inf       > 0 { writeln!(f, "  pow_nan_inf:       {}", self.pow_nan_inf)?; }
        if self.unop_nan          > 0 { writeln!(f, "  unop_nan:          {}", self.unop_nan)?; }
        if self.neg_binomial_pois > 0 { writeln!(f, "  neg_binomial_pois: {}", self.neg_binomial_pois)?; }
        if self.binomial_fallback > 0 { writeln!(f, "  binomial_fallback: {}", self.binomial_fallback)?; }
        if self.obs_sample_nan    > 0 { writeln!(f, "  obs_sample_nan:    {}", self.obs_sample_nan)?; }
        if self.table_oob         > 0 { writeln!(f, "  table_oob:         {}", self.table_oob)?; }
        if self.balance_negative  > 0 { writeln!(f, "  balance_negative:  {}", self.balance_negative)?; }
        Ok(())
    }
}

/// gh#audit-H5. Convenience helper used by every CLI entry point that
/// runs simulation or inference. Snapshot at the start of `cmd_*`, call
/// `report_if_nonzero(start)` at the end. Prints a compact summary to
/// stderr if any counter incremented during the run; silent otherwise.
/// Does not write JSON — `eval_stats.json` was the audit's recommendation
/// for fit runs with a results dir; left as future work for now.
pub fn report_if_nonzero(start: &EvalStats) {
    let end  = EvalStats::snapshot();
    let diff = end.diff_since(start);
    if diff.total() > 0 {
        eprintln!();
        eprint!("{}", diff);
    }
}
