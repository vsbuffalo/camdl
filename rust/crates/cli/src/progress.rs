//! Progress-output policy for long-running subcommands.
//!
//! Resolves the `--progress {auto,pretty,plain,none}` CLI flag (GH #14) into
//! an effective mode stored in a process-wide `OnceLock`. Call sites consult
//! `draw_target()` when constructing indicatif bars and `is_plain()` when
//! deciding whether to emit plain-text progress lines alongside (or instead
//! of) the bar updates.
//!
//! The `auto` mode resolves to `Pretty` when stderr is a TTY and `Plain`
//! otherwise — matching the pattern documented by `cargo --color auto`,
//! `docker build --progress auto`, and `tqdm`'s auto-fallback.
//!
//! Plain mode emits one line per significant event without carriage returns
//! or ANSI escapes, throttled per (chain, event-type) pair. Designed to be
//! safe under `tee`, `&> log`, `ssh host 'camdl ...'`, and CI pipelines —
//! the motivating use cases from the camdl-book CLAUDE.md guidance about
//! `script(1)` wrapping, which this replaces.

use std::io::IsTerminal;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::args::types::ProgressMode;

/// Effective progress mode after resolving `Auto` against the terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolved {
    Pretty,
    Plain,
    None,
}

static RESOLVED: OnceLock<Resolved> = OnceLock::new();

/// Install the process-wide progress mode from the CLI flag. Safe to call
/// more than once; subsequent calls are ignored.
pub fn init(mode: ProgressMode) {
    let r = match mode {
        ProgressMode::Auto => {
            if std::io::stderr().is_terminal() { Resolved::Pretty }
            else { Resolved::Plain }
        }
        ProgressMode::Pretty => Resolved::Pretty,
        ProgressMode::Plain  => Resolved::Plain,
        ProgressMode::None   => Resolved::None,
    };
    let _ = RESOLVED.set(r);
}

/// Current effective mode. Defaults to `Pretty` if `init` was never called
/// (e.g., in unit tests that instantiate a bar directly).
pub fn resolved() -> Resolved {
    RESOLVED.get().copied().unwrap_or(Resolved::Pretty)
}

/// Indicatif draw target to use for bars. In `Plain` and `None` modes this
/// is `hidden()` — the bar still exists (so position/message updates don't
/// have to be gated at every call site) but nothing renders.
pub fn draw_target() -> ProgressDrawTarget {
    match resolved() {
        Resolved::Pretty => ProgressDrawTarget::stderr(),
        Resolved::Plain | Resolved::None => ProgressDrawTarget::hidden(),
    }
}

/// True when plain-text progress lines should be emitted by callbacks.
pub fn is_plain() -> bool { resolved() == Resolved::Plain }

/// True when no progress output of any kind should happen.
pub fn is_none() -> bool { resolved() == Resolved::None }

/// Time-throttled emitter for plain-mode progress lines. One instance per
/// (chain, event-type) avoids flooding the log when callbacks fire every
/// few milliseconds at the end of a run.
///
/// Usage:
/// ```ignore
/// let mut throttle = Throttle::new(Duration::from_secs(5));
/// for iter in 0..n {
///     // ... work ...
///     if throttle.ready() {
///         log::info!("chain {} iter {}/{} ll={:.1}", chain_id, iter, n, ll);
///     }
/// }
/// ```
/// Default cadence for plain-mode per-chain progress lines. Chosen to
/// produce a handful of lines for a typical 2-hour scout (36 chains ×
/// one line per 30s = ~240 lines total) — enough for `tail -f` to show
/// motion without overwhelming the log. Consumers should prefer
/// `Throttle::default()` over hard-coding this value.
///
/// If/when `--progress-interval` lands (GH #14 stretch), this becomes
/// the default the flag overrides.
pub const DEFAULT_THROTTLE: Duration = Duration::from_secs(30);

pub struct Throttle {
    min_interval: Duration,
    last: Option<Instant>,
}

impl Default for Throttle {
    /// 30-second cadence — see `DEFAULT_THROTTLE`.
    fn default() -> Self { Self::new(DEFAULT_THROTTLE) }
}

impl Throttle {
    pub fn new(min_interval: Duration) -> Self {
        Self { min_interval, last: None }
    }

    /// Returns true at most once per `min_interval`. Always returns true
    /// on first call.
    pub fn ready(&mut self) -> bool {
        let now = Instant::now();
        match self.last {
            None => { self.last = Some(now); true }
            Some(prev) if now.duration_since(prev) >= self.min_interval => {
                self.last = Some(now); true
            }
            _ => false,
        }
    }

}

// ─── Rendering layer (Reporter / Task) ──────────────────────────────────────
//
// The shared rendering API every subcommand uses, so the per-subcommand
// hand-rolled `MultiProgress` + `ProgressStyle` blocks collapse to one place
// and all bars look identical. `Reporter` is a factory; the `Task`s it hands
// out are self-sufficient (each holds a clone of the `MultiProgress`, which is
// `Arc`-backed, so the `Reporter` may be dropped while bars keep rendering).
// See docs/dev/proposals/2026-06-03-progress-system.md.

/// The shared count-bar style: `<prefix> <bar> pos/len  <rate>  ETA  <metric>`.
/// One definition so every subcommand's bars are visually identical. The rate
/// is a custom `{rate}` key that labels the per-second figure with the work
/// `unit` (`6.0 cells/s`, `0.40 it/s`) — indicatif's built-in `{per_sec}` is
/// the unitless `0.4/s` that the user found uninformative. Precision adapts to
/// the magnitude. `ETA` comes free; `{msg}` carries the optional metric.
fn count_style(unit: &str) -> ProgressStyle {
    let unit = unit.to_string();
    ProgressStyle::with_template(
        "  {prefix:<22} {bar:24.cyan/dim} {pos}/{len} {rate:>13} ETA {eta:<5} {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .with_key("rate", move |s: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
        let _ = write!(w, "{}", fmt_rate(s.per_sec(), &unit));
    })
    .progress_chars("\u{2501}\u{2578}\u{2500}")
}

/// Format a per-second rate with its `unit`, precision adapting to magnitude:
/// `44 cells/s`, `4.4 it/s`, `0.40 pts/s`, and `-- reps/s` for a not-yet-known
/// rate (zero / non-finite). Pulled out of the `{rate}` key closure so the
/// format is unit-testable without a live terminal.
fn fmt_rate(per_sec: f64, unit: &str) -> String {
    if !per_sec.is_finite() || per_sec == 0.0 {
        format!("-- {unit}/s")
    } else if per_sec >= 10.0 {
        format!("{per_sec:.0} {unit}/s")
    } else if per_sec >= 1.0 {
        format!("{per_sec:.1} {unit}/s")
    } else {
        format!("{per_sec:.2} {unit}/s")
    }
}

/// A group of progress bars for one subcommand invocation. Honors the
/// resolved mode: `Pretty` renders bars on stderr, `Plain` emits throttled log
/// lines, `None` is silent. Replaces the hand-rolled `MultiProgress` blocks.
pub struct Reporter {
    mp: MultiProgress,
    mode: Resolved,
}

impl Reporter {
    pub fn new() -> Self {
        Self { mp: MultiProgress::with_draw_target(draw_target()), mode: resolved() }
    }

    /// A count bar (`pos/len`) for `len` units of work labelled `label`
    /// (rendered as the bar prefix), with `unit` naming the work item so the
    /// rate reads `6.0 cells/s` / `0.40 it/s` (not the unitless `0.4/s`).
    /// Multiple `task()` calls on one `Reporter` share its `MultiProgress`, so
    /// they render as a coordinated stack (e.g. one bar per fit chain).
    pub fn task(&self, len: u64, label: impl Into<String>, unit: &str) -> Task {
        let label = label.into();
        let pb = self.mp.add(ProgressBar::new(len));
        pb.set_style(count_style(unit));
        pb.set_prefix(label.clone());
        // Steady tick so the bar paints and the ETA advances between `inc`
        // calls. A no-op against a hidden draw target (Plain / None).
        pb.enable_steady_tick(Duration::from_millis(120));
        Task {
            pb,
            _mp: self.mp.clone(),
            mode: self.mode,
            throttle: std::sync::Mutex::new(Throttle::default()),
            metric: std::sync::Mutex::new(String::new()),
            label,
        }
    }
}

impl Default for Reporter {
    fn default() -> Self { Self::new() }
}

/// One bar within a [`Reporter`]. Advance with [`Task::inc`], attach a
/// researcher metric with [`Task::set`], close with [`Task::finish`].
/// `Pretty` redraws; `Plain` emits a throttled `label pos/len <metric>` log
/// line; `None` is silent.
///
/// `inc`/`set` take `&self` (interior-mutable throttle + metric) so one bar
/// can be shared across rayon workers — `survey` ticks a single overall bar
/// from a parallel point sweep, `CasSink` ticks it from the sequential merge
/// loop, and fit gives each chain its own bar. `Task` is `Send + Sync`.
pub struct Task {
    pb: ProgressBar,
    /// A clone of the owning `MultiProgress`, held only to keep it alive for
    /// this bar's lifetime (it is `Arc`-backed; dropping the `Reporter` must
    /// not stop the bar rendering).
    _mp: MultiProgress,
    mode: Resolved,
    throttle: std::sync::Mutex<Throttle>,
    /// Last metric string, kept so Plain-mode lines carry it too.
    metric: std::sync::Mutex<String>,
    label: String,
}

impl Task {
    /// Advance by `n`. `Pretty`: redraw. `Plain`: a throttled
    /// `label pos/len <metric>` log line (position is tracked even against a
    /// hidden target). `None`: no-op.
    pub fn inc(&self, n: u64) {
        match self.mode {
            Resolved::Pretty => self.pb.inc(n),
            Resolved::Plain => {
                self.pb.inc(n);
                let ready = self.throttle.lock().map(|mut t| t.ready()).unwrap_or(false);
                if ready {
                    let m = self.metric.lock().map(|s| s.clone()).unwrap_or_default();
                    let sep = if m.is_empty() { "" } else { "  " };
                    log::info!("{} {}/{}{}{}", self.label, self.pb.position(),
                        self.pb.length().unwrap_or(0), sep, m);
                }
            }
            Resolved::None => {}
        }
    }

    /// Attach a researcher-facing metric (e.g. [`best_ll`]). `Pretty` shows it
    /// on the bar as `{msg}`; `Plain` folds it into the next throttled line;
    /// `None` is a no-op. Callers format the string (standard forms documented
    /// in the progress-system proposal).
    pub fn set(&self, metric: impl Into<String>) {
        let m = metric.into();
        if self.mode == Resolved::Pretty {
            self.pb.set_message(m.clone());
        }
        if let Ok(mut g) = self.metric.lock() {
            *g = m;
        }
    }

    /// Finish and clear the bar. `Pretty` removes it from the terminal (the
    /// caller prints any end-of-run summary); `Plain`/`None` have nothing to
    /// clear.
    pub fn finish(self) {
        self.pb.finish_and_clear();
    }
}

/// Standard "best log-likelihood so far" metric string for search-style work
/// (`survey`, `profile`). Kept here so the live bar reads identically across
/// subcommands.
pub fn best_ll(x: f64) -> String {
    if x.is_finite() { format!("best ll={:.1}", x) } else { "best ll=-inf".to_string() }
}

/// Current log-likelihood metric string, prefixed by the loglik *class*:
/// `ll(complete)=` for PGAS's complete-data value, `ll=` for a marginal
/// `log p(y | θ)` (gh#280). This is the live-feed analogue of the
/// `log_complete_data_ll` trace column — it stops a complete-data value being
/// read as a marginal on the bar a human or agent watches mid-run. Mirrors
/// [`best_ll`]'s finite / -inf handling.
pub fn ll_kind(x: f64, kind: crate::fit::loglik::LoglikType) -> String {
    let prefix = kind.metric_prefix();
    if x.is_finite() { format!("{prefix}={:.1}", x) } else { format!("{prefix}=-inf") }
}

/// Standard "current log-likelihood" metric string for marginal-reporting
/// iterative fitters (`fit` IF2, `pfilter`). A thin [`ll_kind`] wrapper
/// fixing the class to marginal (`ll=`); PGAS passes `CompleteData` to
/// [`ll_kind`] directly so its feed reads `ll(complete)=`.
pub fn ll(x: f64) -> String {
    ll_kind(x, crate::fit::loglik::LoglikType::Marginal)
}

/// Standard MCMC metric string carrying the current log-likelihood and the
/// acceptance fraction (PMMH / PGAS-MCMC), e.g. `"ll=-12.3  acc=24%"`. The
/// `accept` argument is a fraction in `[0, 1]`. Reuses [`ll`] for the
/// log-likelihood term and its -inf handling.
pub fn mcmc(loglik: f64, accept: f64) -> String {
    format!("{}  acc={:.0}%", ll(loglik), accept * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ll_finite_and_neg_inf() {
        assert_eq!(ll(-12.34), "ll=-12.3");
        assert_eq!(ll(f64::NEG_INFINITY), "ll=-inf");
        // NaN / +inf are not finite → the -inf label (matches best_ll).
        assert_eq!(ll(f64::NAN), "ll=-inf");
    }

    #[test]
    fn ll_kind_carries_the_class_prefix() {
        use crate::fit::loglik::LoglikType;
        // Marginal class → `ll=`; complete-data → `ll(complete)=` (gh#280).
        assert_eq!(ll_kind(-12.34, LoglikType::Marginal), "ll=-12.3");
        assert_eq!(ll_kind(-12.34, LoglikType::If2), "ll=-12.3");
        assert_eq!(ll_kind(-12.34, LoglikType::CompleteData), "ll(complete)=-12.3");
        // -inf / NaN / +inf all fall to the `<prefix>=-inf` label.
        assert_eq!(ll_kind(f64::NEG_INFINITY, LoglikType::CompleteData), "ll(complete)=-inf");
        assert_eq!(ll_kind(f64::NAN, LoglikType::Marginal), "ll=-inf");
        // `ll` is exactly the marginal-default wrapper.
        assert_eq!(ll(-12.34), ll_kind(-12.34, LoglikType::Marginal));
    }

    #[test]
    fn mcmc_reuses_ll_and_formats_accept() {
        assert_eq!(mcmc(-12.34, 0.24), "ll=-12.3  acc=24%");
        assert_eq!(mcmc(f64::NEG_INFINITY, 0.0), "ll=-inf  acc=0%");
        assert_eq!(mcmc(-1.0, 1.0), "ll=-1.0  acc=100%");
    }

    #[test]
    fn fmt_rate_labels_units_with_adaptive_precision() {
        assert_eq!(fmt_rate(44.0, "cells"), "44 cells/s");   // ≥10 → 0 dp
        assert_eq!(fmt_rate(4.4, "it"), "4.4 it/s");          // ≥1  → 1 dp
        assert_eq!(fmt_rate(0.4, "cells"), "0.40 cells/s");   // <1  → 2 dp (the user's case, now labelled)
        assert_eq!(fmt_rate(0.0, "reps"), "-- reps/s");       // not yet known
        assert_eq!(fmt_rate(f64::INFINITY, "pts"), "-- pts/s");
    }
}
