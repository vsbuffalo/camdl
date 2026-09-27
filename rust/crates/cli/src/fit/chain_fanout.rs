//! Running a fit's chains in parallel so that every started chain keeps
//! advancing (gh#821).
//!
//! A fit nests parallelism: chains run side by side, and each chain's particle
//! filter runs its own `par_iter` over particles on the same Rayon pool. Fanning
//! the chains out as an ordinary parallel iterator (`(0..n).into_par_iter()`)
//! makes each not-yet-started chain an ordinary, stealable Rayon job. A worker
//! that is blocked inside one chain's particle `par_iter`, waiting for a half
//! another worker stole, looks for work while it waits — and may steal a
//! pending chain and run it to completion *on top of* the chain it was already
//! running. The first chain cannot resume until the stolen one finishes: it
//! stops advancing for the whole length of a chain, while the scheduler shows
//! two chains starting at the instant one finished. That is the gh#821 report
//! (16 chains, `--parallel 5`: two chains frozen at 163 and 116 of 2000 sweeps
//! for hours), reproduced by `nested_chains_never_share_a_worker` below.
//!
//! The fix keeps the pool and its size, and changes only how chains reach it:
//! [`rayon::broadcast`] places one chain-worker loop on
//! *each* pool thread, and each loop pulls the next chain index from a shared
//! counter. A broadcast job sits in its own thread's private queue, which no
//! other worker steals from, and chain bodies are never pushed as ordinary
//! jobs — so the only work a blocked worker can pick up is a fragment of some
//! chain's particle step. A chain can then be delayed by at most one particle
//! step of another chain, never by a whole chain.
//!
//! What `--parallel N` means is unchanged: one pool of N threads, at most N
//! chains running at once, each chain started only when a worker has finished
//! its previous chain, and never more than N busy threads. A worker whose loop
//! finds no chain left returns to the pool and helps the remaining chains'
//! particle filters, so the tail of a run (or a run where most chains were
//! refused) still uses all N threads.
//!
//! Results do not depend on which worker runs which chain: each chain draws
//! from its own seed, and the particle filters are thread-count invariant
//! (`sim/tests/gate_pgas_thread_invariance.rs`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Run `f(chain)` for every chain in `0..n_chains` on the current Rayon pool,
/// at most one chain per worker at a time. Returns the outputs in chain order.
///
/// "Current pool" is Rayon's: the pool this is called from inside (under
/// `ThreadPool::install`), else the global pool.
pub fn run_chains<T, F>(n_chains: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    dispatch(n_chains, f, |_| false)
        .into_iter()
        .map(|out| out.expect("run_chains: every chain is dispatched when none is fatal"))
        .collect()
}

/// [`run_chains`] for chains that can fail structurally: once a chain returns
/// `Err`, no further chain is started (chains already running finish), and the
/// error of the lowest-numbered failed chain is returned. This is the
/// short-circuit `collect::<Result<Vec<_>, _>>()` gave over a parallel iterator
/// — a config error surfaces without first running every remaining chain.
pub fn try_run_chains<T, E, F>(n_chains: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    let outs = dispatch(n_chains, f, |r| r.is_err());
    let mut ok = Vec::with_capacity(n_chains);
    let mut not_run = false;
    for out in outs {
        match out {
            Some(Ok(t)) => ok.push(t),
            Some(Err(e)) => return Err(e),
            None => not_run = true,
        }
    }
    // A chain is left undispatched only after some chain failed, and that
    // failure returned above.
    assert!(!not_run, "try_run_chains: a chain was skipped with no chain failing");
    Ok(ok)
}

/// The one chain dispatcher: a chain-worker loop on each pool thread, pulling
/// chain indices from a shared counter until they run out or `is_fatal` has
/// held for some output. `None` marks a chain never started.
fn dispatch<T, F, S>(n_chains: usize, f: F, is_fatal: S) -> Vec<Option<T>>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
    S: Fn(&T) -> bool + Sync,
{
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<T>>> = (0..n_chains).map(|_| Mutex::new(None)).collect();
    rayon::broadcast(|_| loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let chain = next.fetch_add(1, Ordering::Relaxed);
        if chain >= n_chains {
            break;
        }
        let out = f(chain);
        if is_fatal(&out) {
            stop.store(true, Ordering::Release);
        }
        *slots[chain].lock().unwrap_or_else(|p| p.into_inner()) = Some(out);
    });
    slots.into_iter().map(|m| m.into_inner().unwrap_or_else(|p| p.into_inner())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::cell::Cell;

    thread_local! {
        /// How many chain bodies are live on this thread's stack.
        static CHAIN_DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    /// What a chain body saw: the deepest chain nesting on any thread, and the
    /// most chains ever running at once.
    #[derive(Default)]
    struct Observed {
        max_depth: AtomicUsize,
        max_running: AtomicUsize,
        running: AtomicUsize,
    }

    /// A stand-in chain with the shape of a PGAS/PMMH/IF2 chain: many
    /// sequential iterations, each an inner `par_iter` over particles on the
    /// same pool. Records nesting and concurrency; returns a value computed
    /// from the chain id alone.
    fn chain_body(chain: usize, obs: &Observed) -> u64 {
        let depth = CHAIN_DEPTH.with(|d| {
            d.set(d.get() + 1);
            d.get()
        });
        obs.max_depth.fetch_max(depth, Ordering::SeqCst);
        let now = obs.running.fetch_add(1, Ordering::SeqCst) + 1;
        obs.max_running.fetch_max(now, Ordering::SeqCst);
        // Chains of unequal length, so they finish at different times and
        // workers free up while other chains are mid-particle-step.
        let iterations = 30 + 15 * (chain % 4);
        let mut particles = vec![chain as f64; 256];
        for _ in 0..iterations {
            particles.par_iter_mut().for_each(|p| {
                for i in 0..400 {
                    *p = (*p + i as f64).sin();
                }
            });
        }
        obs.running.fetch_sub(1, Ordering::SeqCst);
        CHAIN_DEPTH.with(|d| d.set(d.get() - 1));
        chain as u64 * 1_000 + iterations as u64
    }

    fn expected(n: usize) -> Vec<u64> {
        (0..n).map(|c| c as u64 * 1_000 + (30 + 15 * (c % 4)) as u64).collect()
    }

    /// gh#821. More chains than pool threads, each nesting particle-level
    /// parallelism: no worker may ever hold two chain bodies on its stack
    /// (the displaced one would freeze until the other finished), and no more
    /// chains than threads may be running at once.
    ///
    /// Green is structural under `run_chains`, not a timing outcome. The same
    /// body fanned out with `(0..n).into_par_iter()` — the pre-gh#821 pattern —
    /// reaches depth ≥ 2 in most rounds (9 of 10 at 5 threads × 16 chains in
    /// the reproduction); `ROUNDS` repeats make that red reliable.
    #[test]
    fn nested_chains_never_share_a_worker() {
        const THREADS: usize = 4;
        const CHAINS: usize = 13;
        const ROUNDS: usize = 8;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(THREADS).build().unwrap();
        for _ in 0..ROUNDS {
            let obs = Observed::default();
            let out = pool.install(|| run_chains(CHAINS, |c| chain_body(c, &obs)));
            assert_eq!(out, expected(CHAINS), "outputs in chain order");
            assert_eq!(obs.max_depth.load(Ordering::SeqCst), 1,
                "a worker ran a second chain on top of the one it was running");
            assert!(obs.max_running.load(Ordering::SeqCst) <= THREADS,
                "more chains running than pool threads");
        }
    }

    /// Called from outside any pool (as `camdl fit` does), the chains run on
    /// the global pool with the same guarantees.
    #[test]
    fn global_pool_from_outside() {
        let obs = Observed::default();
        let out = run_chains(9, |c| chain_body(c, &obs));
        assert_eq!(out, expected(9));
        assert_eq!(obs.max_depth.load(Ordering::SeqCst), 1);
        assert!(obs.max_running.load(Ordering::SeqCst) <= rayon::current_num_threads());
    }

    #[test]
    fn zero_chains_is_empty() {
        assert!(run_chains(0, |c| c).is_empty());
        assert_eq!(try_run_chains(0, Ok::<usize, ()>), Ok(vec![]));
    }

    /// A structural failure stops further chains from starting and reports the
    /// lowest-numbered failure. With one worker the order is sequential, so
    /// "no chain after the failure started" is exact.
    #[test]
    fn try_run_chains_stops_dispatching_after_an_error() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let started = AtomicUsize::new(0);
        let r: Result<Vec<usize>, String> = pool.install(|| {
            try_run_chains(10, |c| {
                started.fetch_add(1, Ordering::SeqCst);
                if c == 3 { Err(format!("chain {c} failed")) } else { Ok(c) }
            })
        });
        assert_eq!(r, Err("chain 3 failed".to_string()));
        assert_eq!(started.load(Ordering::SeqCst), 4, "chains after the failure must not start");

        let all_ok: Result<Vec<usize>, String> = pool.install(|| try_run_chains(5, Ok));
        assert_eq!(all_ok, Ok(vec![0, 1, 2, 3, 4]));
    }
}
