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
use std::sync::OnceLock;

pub static DIV_BY_ZERO:       AtomicU64 = AtomicU64::new(0);
pub static POW_NAN_INF:       AtomicU64 = AtomicU64::new(0);
pub static UNOP_NAN:          AtomicU64 = AtomicU64::new(0);
pub static NEG_BINOMIAL_POIS: AtomicU64 = AtomicU64::new(0);
pub static BINOMIAL_FALLBACK: AtomicU64 = AtomicU64::new(0);
/// gh#127 (#12): out-of-range table lookups hit by the fast evaluator. Unlike
/// the others this is NOT a silent-fallback path — the lookup returns NaN and
/// the run hard-errors at the `eval_propensities` boundary — but it is counted
/// here for the same end-of-run summary so a user sees an OOB happened.
pub static TABLE_OOB:         AtomicU64 = AtomicU64::new(0);
/// gh#517: a Poisson draw was asked for a non-finite rate. NaN is the one that
/// matters — before the guard it was laundered into a draw of ~10^15 by
/// `f64::min`, producing a finite, plausible-looking count from an undefined
/// rate. It now returns 0. +inf keeps hitting the 1e15 clamp (its limiting
/// case) and is counted here too. Either way the model produced an undefined
/// rate, which is a defect upstream — not a numerical corner to absorb.
pub static POISSON_NONFINITE: AtomicU64 = AtomicU64::new(0);

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

/// Bench / differential-validation switch: route the propensity hot path
/// through the slow string-keyed `eval_expr` instead of the pre-resolved
/// `eval_resolved`. Enabled by setting the env var `CAMDL_EVAL_UNRESOLVED`
/// (any value). Read once and cached, so it costs an atomic load — hoisted
/// out of the per-transition loop in `eval_propensities`, so the default
/// (off) path is unchanged.
///
/// Two uses:
///  - **benching**: time a `pfilter`/`fit` run with the var off vs on to
///    measure end-to-end what pre-resolution buys (`T_on / T_off`);
///  - **validation**: run any sim/inference under both evaluators and assert
///    byte-identical output — a differential-testing oracle for the resolver,
///    which is inference-critical.
///
/// `eval_resolved` and `eval_expr` are required to agree (see
/// `tests/resolved_expr.rs`); this switch makes that agreement observable on
/// real models at full scale, not just hand-built expressions.
static EVAL_UNRESOLVED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn eval_unresolved() -> bool {
    *EVAL_UNRESOLVED.get_or_init(|| {
        let on = std::env::var_os("CAMDL_EVAL_UNRESOLVED").is_some();
        if on {
            eprintln!(
                "[camdl] CAMDL_EVAL_UNRESOLVED set — propensity eval routed through \
                 the slow string-keyed eval_expr (bench/validation mode)"
            );
        }
        on
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EvalStats {
    pub div_by_zero:       u64,
    pub pow_nan_inf:       u64,
    pub unop_nan:          u64,
    pub neg_binomial_pois: u64,
    pub binomial_fallback: u64,
    pub table_oob:         u64,
    /// gh#517: a Poisson draw asked for a non-finite rate. NaN returns 0;
    /// +inf hits the 1e15 clamp. Either way the model produced an undefined
    /// rate, which is a defect upstream, not a numerical corner.
    pub poisson_nonfinite: u64,
}

impl EvalStats {
    pub fn snapshot() -> Self {
        EvalStats {
            div_by_zero:       DIV_BY_ZERO.load(Ordering::Relaxed),
            pow_nan_inf:       POW_NAN_INF.load(Ordering::Relaxed),
            unop_nan:          UNOP_NAN.load(Ordering::Relaxed),
            neg_binomial_pois: NEG_BINOMIAL_POIS.load(Ordering::Relaxed),
            binomial_fallback: BINOMIAL_FALLBACK.load(Ordering::Relaxed),
            table_oob:         TABLE_OOB.load(Ordering::Relaxed),
            poisson_nonfinite: POISSON_NONFINITE.load(Ordering::Relaxed),
        }
    }

    pub fn diff_since(&self, earlier: &Self) -> Self {
        EvalStats {
            div_by_zero:       self.div_by_zero.saturating_sub(earlier.div_by_zero),
            pow_nan_inf:       self.pow_nan_inf.saturating_sub(earlier.pow_nan_inf),
            unop_nan:          self.unop_nan.saturating_sub(earlier.unop_nan),
            neg_binomial_pois: self.neg_binomial_pois.saturating_sub(earlier.neg_binomial_pois),
            binomial_fallback: self.binomial_fallback.saturating_sub(earlier.binomial_fallback),
            table_oob:         self.table_oob.saturating_sub(earlier.table_oob),
            poisson_nonfinite: self.poisson_nonfinite.saturating_sub(earlier.poisson_nonfinite),
        }
    }

    pub fn total(&self) -> u64 {
        self.div_by_zero + self.pow_nan_inf + self.unop_nan
            + self.neg_binomial_pois + self.binomial_fallback + self.table_oob
            + self.poisson_nonfinite
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
pub fn inc_table_oob()         { TABLE_OOB.fetch_add(1, Ordering::Relaxed); }
#[inline]
pub fn inc_poisson_nonfinite() { POISSON_NONFINITE.fetch_add(1, Ordering::Relaxed); }

impl std::fmt::Display for EvalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "eval-stats summary (counts of fallback paths hit during this run):")?;
        if self.div_by_zero       > 0 { writeln!(f, "  div_by_zero:       {}", self.div_by_zero)?; }
        if self.pow_nan_inf       > 0 { writeln!(f, "  pow_nan_inf:       {}", self.pow_nan_inf)?; }
        if self.unop_nan          > 0 { writeln!(f, "  unop_nan:          {}", self.unop_nan)?; }
        if self.neg_binomial_pois > 0 { writeln!(f, "  neg_binomial_pois: {}", self.neg_binomial_pois)?; }
        if self.binomial_fallback > 0 { writeln!(f, "  binomial_fallback: {}", self.binomial_fallback)?; }
        if self.table_oob         > 0 { writeln!(f, "  table_oob:         {}", self.table_oob)?; }
        // gh#517. Named with its consequence, because unlike its neighbours a
        // non-finite rate is not a numerical corner the run absorbed — it is a
        // rate the model could not define, and the count is how many draws
        // were made from it.
        if self.poisson_nonfinite > 0 {
            writeln!(f, "  poisson_nonfinite: {}  \
                (a rate evaluated to NaN or inf; those draws returned 0 — \
                check for a division by zero or an unset covariate)",
                self.poisson_nonfinite)?;
        }
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
