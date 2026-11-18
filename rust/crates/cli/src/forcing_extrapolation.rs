//! What a command does when a run evaluates a forcing outside the range on
//! which it is defined (spec `docs/dev/temporal-architecture.md` §4.6, D44 and
//! D48): a data-table forcing before its first or after its last data point,
//! or a `bspline` outside its window.
//!
//! The engine holds the end value and records, once per forcing and side,
//! that it did, as a run note ([`sim::forcing_range`]). This module is the
//! policy, as a value: each command builds a [`ForcingPolicy`] from its
//! resolved `--allow-forcing-extrapolation` (or a fit config's `[config]
//! allow_forcing_extrapolation = true`), installs it for its duration
//! ([`ForcingPolicy::install`]), and hands it to every store write it makes
//! ([`crate::resolve::RecordMeta::new`]). By default a note is an error: the
//! command says which forcing, its range and the side it was evaluated on,
//! and exits before it commits anything. A command that holds a store claim
//! while its model runs (`fit run`, `survey`) takes the refusal itself
//! ([`ForcingPolicy::defer_while_claimed`]), so the claim is failed through
//! its own error path rather than left `running` by an exit from the engine's
//! worker thread. Under the override the note is said once as a warning, and
//! every `run.json` the command writes records the override and the forcings
//! it held.
//!
//! The one process-wide piece is the routing slot ([`report`]): the run-note
//! reporter is a function installed once at start-up (`sim::run_notes` is
//! process-wide by design), so it cannot be handed the command's value and
//! looks up the installed one instead.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sim::forcing_range::OutsideRange;

/// The notice code of the warning, and of the error's kind.
pub const CODE: &str = "forcing-outside-range";

/// One command's forcing-extrapolation policy, and what its run met under
/// it: the forcings it held, and the refusal it kept while it held a store
/// claim. Clones share that record, so the value the command installs, the
/// one its store writes hold and the one its claim defers on are one.
#[derive(Clone, Debug)]
pub struct ForcingPolicy(Arc<State>);

#[derive(Debug)]
struct State {
    /// Whether this run may hold a forcing's end value.
    allowed: bool,
    held: Mutex<Vec<OutsideRange>>,
    /// The command holds a store claim and fails it itself on a refusal.
    deferred: AtomicBool,
    /// A refusal was kept for the claim-holding command ([`ForcingPolicy::refused`]);
    /// the message is in `refusal`.
    refused: AtomicBool,
    refusal: Mutex<Option<String>>,
}

/// The policy the run-note reporter routes forcing notes into: the only
/// process-wide state of this module. Set by [`ForcingPolicy::install`] and
/// cleared by the guard it returns. With none installed, a note is refused,
/// as under the default policy.
static ROUTE: Mutex<Option<ForcingPolicy>> = Mutex::new(None);

impl ForcingPolicy {
    /// The policy of a command given (`true`) or not given the override.
    pub fn new(allowed: bool) -> Self {
        ForcingPolicy(Arc::new(State {
            allowed,
            held: Mutex::new(Vec::new()),
            deferred: AtomicBool::new(false),
            refused: AtomicBool::new(false),
            refusal: Mutex::new(None),
        }))
    }

    /// Whether this run may hold a forcing's end value.
    pub fn allowed(&self) -> bool {
        self.0.allowed
    }

    /// Route the engine's forcing notes into this policy until the returned
    /// guard drops. Installed before the command evaluates its model, and
    /// held for the rest of the command. Installing another policy replaces
    /// this one (a command that learns from its fit config that the override
    /// is on); a guard empties the slot only while its own policy is the one
    /// installed, so the replaced policy's guard leaves the new one in place.
    pub fn install(&self) -> Installed {
        *ROUTE.lock().unwrap_or_else(|e| e.into_inner()) = Some(self.clone());
        Installed(self.clone())
    }

    /// Say a new [`OutsideRange`]: under the override a warning, otherwise
    /// the error that ends the command. While the command holds a claim
    /// ([`defer_while_claimed`](Self::defer_while_claimed)) the error is kept
    /// for it to report instead ([`refused`](Self::refused)); otherwise
    /// nothing is committed yet, and the command ends here.
    fn say(&self, o: &OutsideRange) {
        let s = &self.0;
        if !s.allowed {
            if s.deferred.load(Ordering::Relaxed) {
                if let Ok(mut r) = s.refusal.lock() {
                    r.get_or_insert_with(|| refusal(o));
                }
                s.refused.store(true, Ordering::Release);
                return;
            }
            crate::out::detached().fail(refusal(o));
        }
        if let Ok(mut held) = s.held.lock() {
            held.push(o.clone());
        }
        crate::out::live().notice(&crate::out::notice::Notice {
            severity: crate::out::notice::Severity::Warning,
            code: CODE,
            message: o.to_string(),
        });
    }

    /// While the guard lives, a refusal does not end the process: it is kept
    /// for the command, which holds a store claim, to fail that claim with
    /// ([`refused`](Self::refused)). Taken right after the claim, dropped
    /// after its finalize.
    pub fn defer_while_claimed(&self) -> DeferredRefusal {
        self.0.deferred.store(true, Ordering::Relaxed);
        DeferredRefusal(self.clone())
    }

    /// The refusal kept while a claim was held, if the run evaluated a
    /// forcing outside its range without the override. A cheap check: one
    /// atomic load when nothing was refused, so a method's progress callback
    /// may poll it every iteration.
    pub fn refused(&self) -> Option<String> {
        if !self.0.refused.load(Ordering::Acquire) {
            return None;
        }
        self.0.refusal.lock().ok().and_then(|r| r.clone())
    }

    /// Record the override, and the forcings held so far, in a `run.json`'s
    /// display inputs. Nothing is added when the override is off: such a run
    /// held nothing, or it was refused before it committed.
    pub fn record_in(&self, inputs: &mut serde_json::Value) {
        if !self.0.allowed {
            return;
        }
        let Some(obj) = inputs.as_object_mut() else { return };
        let held: Vec<serde_json::Value> = self
            .0
            .held
            .lock()
            .map(|h| {
                h.iter()
                    .map(|o| {
                        serde_json::json!({
                            "forcing": o.forcing,
                            "defined_on": [o.lo, o.hi],
                            "side": match o.side {
                                sim::forcing_range::Side::Before => "before",
                                sim::forcing_range::Side::After => "after",
                            },
                            "held_value_at": o.held_end(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        obj.insert("forcing_extrapolation".into(), serde_json::json!({ "allowed": true, "held": held }));
    }
}

/// The guard [`ForcingPolicy::install`] returns.
#[must_use = "the policy is uninstalled when the guard drops"]
pub struct Installed(ForcingPolicy);

impl Drop for Installed {
    fn drop(&mut self) {
        let mut route = ROUTE.lock().unwrap_or_else(|e| e.into_inner());
        if route.as_ref().is_some_and(|p| Arc::ptr_eq(&p.0, &self.0 .0)) {
            *route = None;
        }
    }
}

/// The guard [`ForcingPolicy::defer_while_claimed`] returns.
pub struct DeferredRefusal(ForcingPolicy);

impl Drop for DeferredRefusal {
    fn drop(&mut self) {
        self.0 .0.deferred.store(false, Ordering::Relaxed);
    }
}

/// Said by the run-note reporter for each new [`OutsideRange`], into the
/// installed policy ([`ForcingPolicy::install`]), or, with none installed,
/// refused as under the default policy.
pub fn report(o: &OutsideRange) {
    let installed = ROUTE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    installed.unwrap_or_else(|| ForcingPolicy::new(false)).say(o);
}

/// The error for a run that evaluated a forcing outside its range without
/// the override.
pub fn refusal(o: &OutsideRange) -> String {
    let fix = match o.side {
        sim::forcing_range::Side::After => format!("end the run by t = {}", o.hi),
        sim::forcing_range::Side::Before => format!("start the run at or after t = {}", o.lo),
    };
    format!(
        "error: forcing `{}` was evaluated {} (it is defined on t = {} to t = {}, {}){}.\n  \
         camdl does not extend a forcing past its range silently. Extend {} to cover the run, {}, \
         or pass --allow-forcing-extrapolation (in a fit config, `[config] \
         allow_forcing_extrapolation = true`) to hold its value at t = {}.",
        o.forcing,
        o.side_phrase(),
        o.lo,
        o.hi,
        o.range_phrase(),
        o.lag_clause(),
        o.range_phrase(),
        fix,
        o.held_end(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::forcing_range::{RangeKind, Side};

    fn past_end(name: &str) -> OutsideRange {
        OutsideRange {
            forcing: name.to_string(),
            kind: RangeKind::DataTable,
            lo: 0.0,
            hi: 50.0,
            side: Side::After,
            lagged: false,
        }
    }

    fn held(inputs: &serde_json::Value) -> Vec<String> {
        inputs["forcing_extrapolation"]["held"]
            .as_array()
            .map(|a| a.iter().map(|h| h["forcing"].as_str().unwrap().to_string()).collect())
            .unwrap_or_default()
    }

    /// The tests that install a policy share the one routing slot.
    static SLOT: Mutex<()> = Mutex::new(());

    /// Two commands in one process, both under the override: the second held
    /// nothing, so its record names nothing held.
    #[test]
    fn a_second_commands_record_names_only_what_it_held() {
        let first = ForcingPolicy::new(true);
        first.say(&past_end("first"));
        let mut record = serde_json::json!({});
        first.record_in(&mut record);
        assert_eq!(held(&record), vec!["first"]);

        let second = ForcingPolicy::new(true);
        let mut record = serde_json::json!({});
        second.record_in(&mut record);
        assert_eq!(held(&record), Vec::<String>::new(), "the second command held nothing");
    }

    /// A refusal kept for one command's claim is not another command's.
    #[test]
    fn a_kept_refusal_belongs_to_its_command() {
        let first = ForcingPolicy::new(false);
        {
            let _deferred = first.defer_while_claimed();
            first.say(&past_end("refused"));
            assert!(first.refused().is_some_and(|m| m.contains("refused")));
        }
        let second = ForcingPolicy::new(false);
        assert_eq!(second.refused(), None, "a fresh command has refused nothing");
    }

    /// The reporter routes a note into the installed policy, and the guard
    /// uninstalls it.
    #[test]
    fn the_reporter_routes_into_the_installed_policy() {
        let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let policy = ForcingPolicy::new(true);
        {
            let _route = policy.install();
            report(&past_end("routed"));
        }
        assert!(ROUTE.lock().unwrap().is_none(), "the guard uninstalls its policy");
        let mut record = serde_json::json!({});
        policy.record_in(&mut record);
        assert_eq!(held(&record), vec!["routed"]);
    }

    /// A command that widens its policy (its fit config turns the override
    /// on) installs the wider one; dropping the first guard leaves it.
    #[test]
    fn a_replacing_install_survives_the_first_guard() {
        let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let mut policy = ForcingPolicy::new(false);
        let mut _route = policy.install();
        policy = ForcingPolicy::new(true);
        _route = policy.install();
        report(&past_end("widened"));
        let mut record = serde_json::json!({});
        policy.record_in(&mut record);
        assert_eq!(held(&record), vec!["widened"]);
        drop(_route);
        assert!(ROUTE.lock().unwrap().is_none(), "the last guard empties the slot");
    }
}
