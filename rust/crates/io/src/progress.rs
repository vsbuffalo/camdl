//! Per-run liveness/progress heartbeat (gh#278).
//!
//! A long fit run (a national-scale PGAS sweep can take days, with minutes
//! between trace writes) leaves no reliable on-disk liveness signal: the
//! `.lock` PID is same-host-only and reuse-prone, and `trace.tsv` mtime is not
//! a heartbeat (one sweep is a full particle-filter pass). This module emits a
//! small `progress.json`, refreshed on a **fixed wall-clock timer independent
//! of step cadence**, so any consumer (a remote dashboard, CI) reads one
//! contract for liveness + progress instead of reverse-engineering it — and can
//! spot a fit that is broken early and stop it.
//!
//! The progress model is **algorithm-agnostic** so EVERY fitting method maps
//! onto it: a generic `step`/`total` counter plus a [`Phase`] (MCMC
//! burn-in/sampling, optimizer search, profile grid). The [`Heartbeat`]
//! constructor picks how the phase is derived ([`Heartbeat::mcmc`] /
//! [`Heartbeat::optimizing`] / [`Heartbeat::profiling`]); the loop just
//! [`bump`](Heartbeat::bump)s a step counter.
//!
//! A multi-chain method reports one more thing (gh#751): **which chains are
//! actually sampling**. A chain is refused at its start, so how many of the 24
//! paid for are running is knowable in the first minutes — and before this it
//! was reported only after the stage finished, in `fit_state.toml` and
//! `diagnostics.json`, which on a multi-hour national fit is hours late. The
//! only signal in the meantime was counting `chain_*/trace.tsv` files with more
//! than a header, which reads a private layout and still cannot separate a
//! chain that was REFUSED from one QUEUED behind `--parallel`. Those call for
//! opposite responses — respecify, or wait — so [`ChainLiveness`] distinguishes
//! them: a chain that has not been picked up by a worker has no row at all, and
//! `not_started` counts them.
//!
//! The honest model: a SIGKILLed run *cannot* write "I died". So the artifact
//! records the run's last self-report ([`RunState`]); **deadness is a consumer
//! inference from staleness**, never a self-claim. [`liveness`] folds the
//! artifact + the clock into that judgement once ([`RunLiveness`]).
//!
//! The heartbeat is a **pure observer**: the step loop only stores into a shared
//! atomic ([`Heartbeat::bump`]); a background thread does the file I/O. It reads
//! nothing the inference writes and consumes no RNG — it cannot change a fit
//! number.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The artifact filename written into a run's seed/stage directory.
pub const PROGRESS_FILE: &str = "progress.json";

/// What kind of work a run is doing — algorithm-agnostic, so every fitting
/// method maps onto the one progress type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// MCMC warmup — no trace rows yet (gh#278 motivation 3). PGAS / PMMH / `mh` on ode.
    BurnIn,
    /// MCMC sampling — trace rows accrue. PGAS / PMMH / `mh` on ode.
    Sampling,
    /// Searching for the MLE — IF2's cooling iterations or an NLopt eval loop.
    Optimizing,
    /// Stepping a profile-likelihood grid.
    Profiling,
}

/// How a run derives its [`Phase`] from the step counter. A [`Heartbeat`]'s
/// constructor picks the rule — MCMC splits on `burn_in`; every other algorithm
/// has a single fixed phase. This is what lets one progress type serve all of
/// them without a phase that some algorithm can't fill in.
#[derive(Debug, Clone, Copy)]
enum PhaseRule {
    Mcmc { burn_in: u64 },
    Fixed(Phase),
}

impl PhaseRule {
    fn at(&self, step: u64) -> Phase {
        match self {
            PhaseRule::Mcmc { burn_in } => {
                if step < *burn_in { Phase::BurnIn } else { Phase::Sampling }
            }
            PhaseRule::Fixed(p) => *p,
        }
    }
}

/// The run's last self-report. An ADT, so incoherent combinations (a failure
/// with no reason, `running` with no counter) are unrepresentable. Serializes
/// externally-tagged: `{"running": {…}}` / `"done"` / `{"failed": {…}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// Live: `step` of `total` units done. The unit is the algorithm's
    /// (sweeps / IF2 iterations / NLopt evals / profile grid points); `phase`
    /// gives the context. Coherent only while running.
    Running { phase: Phase, step: u64, total: u64 },
    /// Clean completion.
    Done,
    /// Clean, caught failure — carries the reason. An *un*caught death
    /// (SIGKILL/panic) instead leaves the last `Running` on disk going stale;
    /// see [`RunLiveness`].
    Failed { reason: String },
}

/// How the `chain` field of a [`ChainStatus`] is numbered. Stated in the
/// artifact so a consumer holding only `progress.json` can map a row to a
/// directory. It agrees with `pgas_summary.json`'s `chain_numbering`, and is
/// deliberately the opposite of `chain_starts.tsv`, whose own `chain_id`
/// column is 0-based and whose header says so.
pub const CHAIN_NUMBERING: &str = "1-based, matching the chain_N/ directories";

/// Why a chain did not run.
///
/// A short tag rather than the refusal's prose, because this is what a
/// dashboard or a `make watch` keys on; the full reason, with the log-posterior
/// terms and the start vector, is the stage's `diagnostics.json` `bad_init`
/// record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusedReason {
    /// The start's complete-data log-posterior was non-finite and did not
    /// recover on the first trajectory update, after every permitted redraw
    /// (gh#607, gh#887). Today the only way a chain is refused.
    NonFiniteStart,
}

/// One chain's last self-report. Internally tagged under `status`, so a refusal
/// carries its reason in a sibling field and every row of the table has the
/// same flat shape: `{"chain": 3, "status": "refused", "reason": "non_finite_start"}`.
///
/// There is deliberately no "queued" variant. A chain that has not been picked
/// up by a worker has said nothing, and inventing a report for it would be a
/// claim the run cannot make — its absence from [`ChainLiveness::chains`] is
/// the honest form, and `not_started` counts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ChainState {
    /// The chain has begun; a worker is running its sweeps.
    Running,
    /// The chain finished the sweeps it was asked for.
    Completed,
    /// The chain never ran: its start could not be scored.
    Refused { reason: RefusedReason },
}

/// One row of the per-chain table: which chain, and what it last said.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainStatus {
    /// The chain, numbered per [`CHAIN_NUMBERING`].
    pub chain: u64,
    #[serde(flatten)]
    pub state: ChainState,
}

/// How many chains of this stage are sampling, finished, or refused — and the
/// per-chain rows behind those counts.
///
/// `running + completed` is the number of chains that got past their start;
/// `total - refused` is not the same thing mid-run, because a chain that has
/// not started yet is neither good nor bad. Both are recoverable from the four
/// counts, and none of them is named `n_good_chains`, which is the stage's
/// FINAL verdict and belongs to `fit_state.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainLiveness {
    /// Chains this stage was configured to run.
    pub total: u64,
    pub running: u64,
    pub completed: u64,
    pub refused: u64,
    /// Chains that have not been picked up by a worker yet — queued behind
    /// `--parallel`, not refused.
    pub not_started: u64,
    /// What each chain that has reported last said, in chain order. A chain
    /// with no row has not started.
    pub chains: Vec<ChainStatus>,
    /// [`CHAIN_NUMBERING`], carried so the file explains its own `chain` column.
    pub numbering: String,
}

impl ChainLiveness {
    /// Fold the per-chain slots — one per chain, 0-based, `None` for a chain
    /// that has not reported — into the artifact's block.
    fn of(slots: &[Option<ChainState>]) -> ChainLiveness {
        let mut live = ChainLiveness {
            total: slots.len() as u64,
            running: 0,
            completed: 0,
            refused: 0,
            not_started: 0,
            chains: Vec::new(),
            numbering: CHAIN_NUMBERING.to_string(),
        };
        for (i, slot) in slots.iter().enumerate() {
            match slot {
                None => live.not_started += 1,
                Some(state) => {
                    match state {
                        ChainState::Running => live.running += 1,
                        ChainState::Completed => live.completed += 1,
                        ChainState::Refused { .. } => live.refused += 1,
                    }
                    live.chains.push(ChainStatus { chain: i as u64 + 1, state: *state });
                }
            }
        }
        live
    }
}

/// The on-disk artifact: an always-present envelope around the state ADT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    /// Unix epoch SECONDS of the last write. Freshness of this — not the
    /// `state` field — is the liveness signal (a killed run can't update it).
    pub updated_at: u64,
    /// The writing process's PID. Informational only; liveness must NOT depend
    /// on it (cross-host + PID-reuse fragile — the whole point of this artifact).
    pub pid: u32,
    /// The run's last self-report.
    pub state: RunState,
    /// Per-chain liveness, for a method that has chains (gh#751). Absent —
    /// not null — for one that does not (an NLopt search, a profile grid), so
    /// those artifacts are byte-identical to before. Purely additive: the three
    /// fields above are unchanged for every reader that already has them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chains: Option<ChainLiveness>,
}

impl Progress {
    fn now(state: RunState, chains: Option<ChainLiveness>) -> Progress {
        Progress {
            updated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            pid: std::process::id(),
            state,
            chains,
        }
    }
}

/// A consumer's judgement, folding the artifact + the clock into one ADT so the
/// staleness/terminal logic lives in a single parse, not re-derived per reader.
#[derive(Debug, Clone, PartialEq)]
pub enum RunLiveness {
    /// `Running` and the heartbeat is fresh.
    Alive(RunState),
    /// `Running` but the heartbeat is stale → presumed dead or hung. This is
    /// the SIGKILL case, named honestly: an inference, not a self-claim.
    PresumedDead(RunState),
    /// `Done`/`Failed` — a clean terminal write; freshness is irrelevant.
    Terminal(RunState),
}

/// Fold a [`Progress`] read + the current time into a [`RunLiveness`].
/// `now_unix` and `max_stale_secs` are seconds. A clean terminal state
/// short-circuits the freshness check; only a stale `Running` is `PresumedDead`.
pub fn liveness(p: &Progress, now_unix: u64, max_stale_secs: u64) -> RunLiveness {
    match &p.state {
        RunState::Done | RunState::Failed { .. } => RunLiveness::Terminal(p.state.clone()),
        RunState::Running { .. } => {
            let age = now_unix.saturating_sub(p.updated_at);
            if age <= max_stale_secs {
                RunLiveness::Alive(p.state.clone())
            } else {
                RunLiveness::PresumedDead(p.state.clone())
            }
        }
    }
}

/// Atomically write `progress.json` into `dir` (temp file + rename), so a
/// concurrent reader never observes a half-written file.
pub fn write_progress(dir: &Path, p: &Progress) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(p).map_err(io::Error::other)?;
    let final_path = dir.join(PROGRESS_FILE);
    let tmp = dir.join(format!("{}.{}.tmp", PROGRESS_FILE, std::process::id()));
    fs::write(&tmp, &json)?;
    fs::rename(&tmp, &final_path)?;
    Ok(())
}

/// Read and parse a run's `progress.json`.
pub fn read_progress(dir: &Path) -> io::Result<Progress> {
    let bytes = fs::read(dir.join(PROGRESS_FILE))?;
    serde_json::from_slice(&bytes).map_err(io::Error::other)
}

// ── The heartbeat handle ─────────────────────────────────────────────────────

struct Shared {
    dir: PathBuf,
    rule: PhaseRule,
    total: u64,
    step: AtomicU64, // monotonic (fetch_max) — furthest any chain has reached
    stop: AtomicBool,
    /// One slot per chain, 0-based, `None` until that chain reports (gh#751).
    /// Empty for a method with no chains, which then writes no `chains` block
    /// at all. A `Mutex` and not an atomic because a refusal carries a reason;
    /// it is taken a handful of times per chain per run, never on the sweep
    /// path, so it cannot contend with the sampler.
    chains: std::sync::Mutex<Vec<Option<ChainState>>>,
}

impl Shared {
    /// The chain block for this write — `None` when the method has no chains.
    fn chain_liveness(&self) -> Option<ChainLiveness> {
        let slots = self.chains.lock().expect("no thread panicked holding the chain slots");
        if slots.is_empty() { None } else { Some(ChainLiveness::of(&slots)) }
    }
}

/// A background heartbeat for a run directory. The constructor encodes the
/// algorithm's progress shape ([`mcmc`](Heartbeat::mcmc) /
/// [`optimizing`](Heartbeat::optimizing) / [`profiling`](Heartbeat::profiling));
/// the step loop calls [`bump`](Heartbeat::bump) (a cheap atomic, no I/O); a
/// timer thread writes `progress.json` every `interval`. [`finish`](Heartbeat::finish)
/// writes the clean terminal state and joins. If dropped without `finish`
/// (panic/early return), the thread stops and the last `Running` is left to go
/// stale — a consumer then reads `PresumedDead`.
pub struct Heartbeat {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl Heartbeat {
    fn spawn(
        dir: PathBuf,
        rule: PhaseRule,
        total: u64,
        interval: Duration,
        n_chains: usize,
    ) -> Heartbeat {
        let shared = Arc::new(Shared {
            dir,
            rule,
            total,
            step: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            chains: std::sync::Mutex::new(vec![None; n_chains]),
        });
        let s = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("camdl-heartbeat".into())
            .spawn(move || {
                // Sleep in short ticks so `stop` is responsive without a condvar.
                let tick = Duration::from_millis(250).min(interval);
                loop {
                    let step = s.step.load(Ordering::Relaxed);
                    let p = Progress::now(
                        RunState::Running {
                            phase: s.rule.at(step),
                            step,
                            total: s.total,
                        },
                        s.chain_liveness(),
                    );
                    let _ = write_progress(&s.dir, &p);
                    let mut waited = Duration::ZERO;
                    while waited < interval {
                        if s.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(tick);
                        waited += tick;
                    }
                }
            })
            .expect("spawn heartbeat thread");
        Heartbeat { shared, handle: Some(handle) }
    }

    /// MCMC heartbeat (PGAS / PMMH / `mh` on ode): the phase reads `BurnIn`
    /// below `burn_in` sweeps and `Sampling` at/after it. `total` is the total
    /// sweeps/steps. `interval` is a fixed wall-clock period (5–10 s).
    ///
    /// `n_chains` is the roster the stage was configured to run, which is what
    /// makes "1 of 24 sampling" sayable at all — the count has to be known
    /// before any chain reports, or a run whose chains all refused would be
    /// indistinguishable from one that has not started (gh#751). Report each
    /// chain's state with [`chain`](Heartbeat::chain).
    pub fn mcmc(
        dir: PathBuf,
        burn_in: u64,
        total: u64,
        interval: Duration,
        n_chains: usize,
    ) -> Heartbeat {
        Self::spawn(dir, PhaseRule::Mcmc { burn_in }, total, interval, n_chains)
    }

    /// Optimizer heartbeat (IF2 cooling iterations / NLopt eval loop): a single
    /// `Optimizing` phase. `total` is the iteration / max-eval budget.
    pub fn optimizing(dir: PathBuf, total: u64, interval: Duration) -> Heartbeat {
        Self::spawn(dir, PhaseRule::Fixed(Phase::Optimizing), total, interval, 0)
    }

    /// Profile heartbeat: a single `Profiling` phase. `total` is the grid size.
    pub fn profiling(dir: PathBuf, total: u64, interval: Duration) -> Heartbeat {
        Self::spawn(dir, PhaseRule::Fixed(Phase::Profiling), total, interval, 0)
    }

    /// Report what `chain` (0-based, the index the sampler works in; the
    /// artifact renders it 1-based) is doing (gh#751).
    ///
    /// Called when a worker picks the chain up, when it is refused, and when it
    /// finishes — three times per chain, never on the sweep path. A chain that
    /// has not been reported has no row, which is how the artifact separates
    /// "refused" from "queued behind `--parallel`". Out-of-range ids are
    /// ignored rather than panicking: this is a diagnostic and must not be able
    /// to take a fit down.
    pub fn chain(&self, chain: usize, state: ChainState) {
        if let Ok(mut slots) = self.shared.chains.lock() {
            if let Some(slot) = slots.get_mut(chain) {
                *slot = Some(state);
            }
        }
    }

    /// Report the furthest step reached. Cheap (one relaxed `fetch_max`) — safe
    /// from the hot loop and from multiple parallel chains; monotonic, so
    /// progress never jitters backward. Does no I/O.
    pub fn bump(&self, step: u64) {
        self.shared.step.fetch_max(step, Ordering::Relaxed);
    }

    /// Stop the timer and write the clean terminal state (`Done` / `Failed`).
    pub fn finish(mut self, state: RunState) {
        self.stop_thread();
        let chains = self.shared.chain_liveness();
        let _ = write_progress(&self.shared.dir, &Progress::now(state, chains));
    }

    fn stop_thread(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        // finish() already joined; this only fires on an un-finished drop
        // (panic/early return). Stop the thread and leave the last Running on
        // disk — the consumer infers PresumedDead from its staleness.
        self.stop_thread();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_state_serializes_as_tagged_adt() {
        let r = RunState::Running { phase: Phase::Optimizing, step: 3, total: 10 };
        let j = serde_json::to_string(&r).unwrap();
        assert!(j.contains("\"running\"") && j.contains("\"optimizing\"") && j.contains("\"step\":3"));
        assert_eq!(serde_json::to_string(&RunState::Done).unwrap(), "\"done\"");
        let f = serde_json::to_string(&RunState::Failed { reason: "boom".into() }).unwrap();
        assert!(f.contains("\"failed\"") && f.contains("boom"));
        let back: RunState = serde_json::from_str(&j).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn phase_rule_covers_every_algorithm_shape() {
        // MCMC: derived from burn_in.
        let mcmc = PhaseRule::Mcmc { burn_in: 5 };
        assert_eq!(mcmc.at(4), Phase::BurnIn);
        assert_eq!(mcmc.at(5), Phase::Sampling);
        // Optimizer / profile: fixed phase regardless of step.
        assert_eq!(PhaseRule::Fixed(Phase::Optimizing).at(999), Phase::Optimizing);
        assert_eq!(PhaseRule::Fixed(Phase::Profiling).at(0), Phase::Profiling);
    }

    #[test]
    fn liveness_distinguishes_alive_stale_terminal() {
        let running = |t: u64| Progress {
            updated_at: t, pid: 1,
            state: RunState::Running { phase: Phase::Sampling, step: 5, total: 10 },
            chains: None,
        };
        assert!(matches!(liveness(&running(100), 105, 30), RunLiveness::Alive(_)));
        assert!(matches!(liveness(&running(100), 200, 30), RunLiveness::PresumedDead(_)));
        let done = Progress { updated_at: 1, pid: 1, state: RunState::Done, chains: None };
        assert!(matches!(liveness(&done, 9_999_999, 30), RunLiveness::Terminal(_)));
    }

    #[test]
    fn write_then_read_round_trips_and_is_atomic_named() {
        let dir = tempfile::tempdir().unwrap();
        let p =
            Progress::now(RunState::Running { phase: Phase::BurnIn, step: 7, total: 20 }, None);
        write_progress(dir.path(), &p).unwrap();
        let tmp = dir.path().join(format!("{}.{}.tmp", PROGRESS_FILE, std::process::id()));
        assert!(!tmp.exists(), "temp file should be renamed away");
        assert_eq!(read_progress(dir.path()).unwrap().state, p.state);
    }

    // ── gh#751: per-chain liveness ───────────────────────────────────────

    /// The counts and the rows for the case the issue is about: 24 chains
    /// paid for, 23 refused, 1 sampling. Before this, `progress.json` said
    /// only `{"running": {"phase": "burn_in", "step": 276, "total": 2000}}` —
    /// byte-for-byte what a healthy 24-chain fit says, for hours.
    #[test]
    fn a_fit_that_refused_all_but_one_chain_says_so() {
        let mut slots: Vec<Option<ChainState>> = vec![
            Some(ChainState::Refused { reason: RefusedReason::NonFiniteStart });
            24
        ];
        slots[7] = Some(ChainState::Running);
        let live = ChainLiveness::of(&slots);

        assert_eq!(live.total, 24);
        assert_eq!(live.running, 1);
        assert_eq!(live.refused, 23);
        assert_eq!(live.completed, 0);
        assert_eq!(live.not_started, 0);
        assert_eq!(live.chains.len(), 24);
        // 1-based, so the eighth slot is chain 8 and matches `chain_8/`.
        let sampling: Vec<u64> = live
            .chains
            .iter()
            .filter(|c| c.state == ChainState::Running)
            .map(|c| c.chain)
            .collect();
        assert_eq!(sampling, vec![8], "the chain id is 1-based: chain_8/");
        assert_eq!(live.numbering, CHAIN_NUMBERING);
    }

    /// A refused chain and one queued behind `--parallel` are the two cases
    /// that call for opposite responses — respecify, or wait — and an empty
    /// chain directory cannot tell them apart. A chain that has not reported
    /// has no row and is counted in `not_started`.
    #[test]
    fn a_queued_chain_is_not_a_refused_one() {
        let slots = vec![
            Some(ChainState::Running),
            Some(ChainState::Refused { reason: RefusedReason::NonFiniteStart }),
            None, // queued behind --parallel
            None,
        ];
        let live = ChainLiveness::of(&slots);
        assert_eq!((live.running, live.refused, live.not_started), (1, 1, 2));
        let ids: Vec<u64> = live.chains.iter().map(|c| c.chain).collect();
        assert_eq!(ids, vec![1, 2], "chains 3 and 4 have said nothing, so no row");
    }

    /// The artifact's shape, which is the part a dashboard or `make watch`
    /// keys on: the three existing top-level fields are untouched and the
    /// per-chain rows are flat, with the refusal's short tag beside its status.
    #[test]
    fn the_chain_block_is_additive_and_its_rows_are_flat() {
        let slots = vec![
            Some(ChainState::Completed),
            Some(ChainState::Refused { reason: RefusedReason::NonFiniteStart }),
        ];
        let p = Progress::now(RunState::Done, Some(ChainLiveness::of(&slots)));
        let v: serde_json::Value = serde_json::to_value(&p).unwrap();

        // Unchanged for every reader that already has them.
        assert!(v["updated_at"].is_u64());
        assert!(v["pid"].is_u64());
        assert_eq!(v["state"], "done");

        assert_eq!(v["chains"]["total"], 2);
        assert_eq!(v["chains"]["completed"], 1);
        assert_eq!(v["chains"]["refused"], 1);
        assert_eq!(v["chains"]["chains"][0]["chain"], 1);
        assert_eq!(v["chains"]["chains"][0]["status"], "completed");
        assert_eq!(v["chains"]["chains"][1]["status"], "refused");
        assert_eq!(
            v["chains"]["chains"][1]["reason"], "non_finite_start",
            "the reason sits beside the status, not nested under it"
        );

        // And it round-trips, so a reader gets the typed value back.
        let back: Progress = serde_json::from_value(v).unwrap();
        assert_eq!(back, p);
    }

    /// A method with no chains writes no `chains` key at all — not `null` —
    /// so an optimizer's or a profile grid's artifact is byte-identical to
    /// what it was before, and an old file still parses.
    #[test]
    fn a_chainless_method_writes_no_chain_key_and_an_old_file_still_parses() {
        let p = Progress::now(RunState::Running { phase: Phase::Optimizing, step: 3, total: 10 }, None);
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("chains"), "no key at all, not a null: {json}");

        let old = r#"{"updated_at":1787670244,"pid":2882,
                      "state":{"running":{"phase":"burn_in","step":276,"total":2000}}}"#;
        let parsed: Progress = serde_json::from_str(old).unwrap();
        assert_eq!(parsed.pid, 2882);
        assert!(parsed.chains.is_none());
    }

    /// The heartbeat carries the chain block onto disk, and `finish` writes
    /// the final one — so a completed fit's artifact still says how many of
    /// the chains it paid for ever ran.
    #[test]
    fn the_heartbeat_writes_the_chain_block_and_finish_keeps_it() {
        let dir = tempfile::tempdir().unwrap();
        let hb = Heartbeat::mcmc(dir.path().to_path_buf(), 5, 30, Duration::from_millis(20), 3);
        hb.chain(0, ChainState::Running);
        hb.chain(2, ChainState::Refused { reason: RefusedReason::NonFiniteStart });
        std::thread::sleep(Duration::from_millis(60));

        let mid = read_progress(dir.path()).unwrap();
        let live = mid.chains.expect("a chain block while running");
        assert_eq!((live.total, live.running, live.refused, live.not_started), (3, 1, 1, 1));

        hb.chain(0, ChainState::Completed);
        hb.finish(RunState::Done);
        let end = read_progress(dir.path()).unwrap();
        assert_eq!(end.state, RunState::Done);
        let live = end.chains.expect("the block survives the terminal write");
        assert_eq!((live.completed, live.refused, live.not_started), (1, 1, 1));
    }

    /// An out-of-range chain id is ignored rather than panicking: this is a
    /// diagnostic, and it must not be able to take a fit down.
    #[test]
    fn an_out_of_range_chain_report_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let hb = Heartbeat::mcmc(dir.path().to_path_buf(), 1, 2, Duration::from_millis(20), 2);
        hb.chain(9, ChainState::Running);
        hb.finish(RunState::Done);
        let live = read_progress(dir.path()).unwrap().chains.unwrap();
        assert_eq!((live.total, live.not_started), (2, 2));
        assert!(live.chains.is_empty());
    }

    #[test]
    fn heartbeat_writes_then_finish_marks_terminal() {
        let dir = tempfile::tempdir().unwrap();
        // optimizing: fixed phase, step 12 of 30.
        let hb = Heartbeat::optimizing(dir.path().to_path_buf(), 30, Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(40));
        hb.bump(12);
        std::thread::sleep(Duration::from_millis(40));
        let mid = read_progress(dir.path()).unwrap();
        assert!(matches!(mid.state, RunState::Running { step: 12, phase: Phase::Optimizing, .. }));
        hb.finish(RunState::Done);
        assert_eq!(read_progress(dir.path()).unwrap().state, RunState::Done);
    }
}
