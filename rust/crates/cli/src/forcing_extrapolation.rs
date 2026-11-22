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
//! while its model runs (`fit run`) takes the refusal itself
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
//!
//! **A policy is told only by a model compiled after it was installed.** The
//! engine reports a forcing outside its range once per side per compiled
//! model: the set-once flag is on the compiled forcing (`DefinedRange` in
//! `sim::forcing_range`), shared by every clone of the model. A model that
//! already held a forcing under one policy does not report it again under the
//! next. So a command installs its policy before it compiles the models it
//! runs under it, and a command that runs under several policies in turn
//! (`compare`, one per in-process fit) compiles a model for each after
//! installing it. Each policy says a forcing once however many of its models
//! report it.

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
    /// Under the override, a held forcing is recorded but not said as it is
    /// met: the command reports it itself, from [`ForcingPolicy::trace_warnings`].
    quiet: bool,
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
        Self::with_quiet(allowed, false)
    }

    /// A policy that, under the override, records each held forcing without
    /// saying it as it is met; the command reports what it held itself.
    /// `compare` scores each in-process fit under one and prints the holds on
    /// the fit's rows, as it does for a fit its `camdl pfilter` child scores.
    /// A refusal is said as under [`ForcingPolicy::new`].
    pub fn recording(allowed: bool) -> Self {
        Self::with_quiet(allowed, true)
    }

    fn with_quiet(allowed: bool, quiet: bool) -> Self {
        ForcingPolicy(Arc::new(State {
            allowed,
            quiet,
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

    /// The forcings held so far, as the warnings a prequential trace scored
    /// under this policy carries, so a reader of the trace (`compare`'s rows)
    /// sees them with the scores they affected.
    pub fn trace_warnings(&self) -> Vec<score::trace::PrequentialWarning> {
        self.0
            .held
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|o| score::trace::PrequentialWarning::ForcingHeld {
                forcing: o.forcing.clone(),
                defined_on: [o.lo, o.hi],
                side: match o.side {
                    sim::forcing_range::Side::Before => score::trace::ForcingSide::Before,
                    sim::forcing_range::Side::After => score::trace::ForcingSide::After,
                },
                held_value_at: o.held_end(),
                bound: Some(match o.kind {
                    sim::forcing_range::RangeKind::DataTable => score::trace::ForcingBound::Data,
                    sim::forcing_range::RangeKind::Window => score::trace::ForcingBound::Window,
                }),
                lagged: Some(o.lagged),
            })
            .collect()
    }

    /// Route the engine's forcing notes into this policy until the returned
    /// guard drops. Install it before compiling the models that run under it,
    /// and hold it for the rest of the command: a model reports a forcing
    /// once, so one compiled earlier and already past a forcing's range does
    /// not tell this policy (see the module doc). An install nests: `camdl predict`
    /// installs the stored fit's policy inside its own, and the inner guard's
    /// drop puts back the policy it replaced.
    pub fn install(&self) -> Installed {
        let previous = ROUTE.lock().unwrap_or_else(|e| e.into_inner()).replace(self.clone());
        Installed { policy: self.clone(), previous }
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
                s.refusal.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(|| refusal(o));
                s.refused.store(true, Ordering::Release);
                return;
            }
            crate::out::detached().fail(refusal(o));
        }
        // Said once per policy: the engine reports a forcing each time a
        // compiled model first holds it (`sim::run_notes`), and one command
        // can compile many.
        {
            let mut held = s.held.lock().unwrap_or_else(|e| e.into_inner());
            if held.contains(o) {
                return;
            }
            held.push(o.clone());
        }
        if s.quiet {
            return;
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
        self.0.refusal.lock().unwrap_or_else(|e| e.into_inner()).clone()
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
            .unwrap_or_else(|e| e.into_inner())
            .iter()
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
            .collect();
        obj.insert("forcing_extrapolation".into(), serde_json::json!({ "allowed": true, "held": held }));
    }
}

/// The guard [`ForcingPolicy::install`] returns.
#[must_use = "the policy is uninstalled when the guard drops"]
pub struct Installed {
    policy: ForcingPolicy,
    /// What the install replaced, put back when the guard drops.
    previous: Option<ForcingPolicy>,
}

impl Drop for Installed {
    /// Put back the policy this install replaced, while this guard's policy
    /// is still the one installed (guards drop innermost first).
    fn drop(&mut self) {
        let mut route = ROUTE.lock().unwrap_or_else(|e| e.into_inner());
        if route.as_ref().is_some_and(|p| Arc::ptr_eq(&p.0, &self.policy.0)) {
            *route = self.previous.take();
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

    /// A `bspline` forcing defined on t = 0 to 120 days, compiled into a
    /// fresh model.
    fn bspline_model() -> sim::compiled_model::CompiledModel {
        let golden = format!("{}/../../../ocaml/golden/bspline_uniform.ir.json", env!("CARGO_MANIFEST_DIR"));
        let (mut m, _) = crate::util::load_ir(&golden).unwrap();
        for p in &mut m.parameters {
            p.value = p.value.with_value(1.0);
        }
        sim::compiled_model::CompiledModel::new(m).unwrap()
    }

    /// Evaluate the model's forcing at `t`, as a run does.
    fn evaluate_forcing(model: &sim::compiled_model::CompiledModel, t: f64) {
        let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
        let ctx = sim::propensity::EvalCtx {
            model, int_s: &int_s, real_s: &real_s, params: &model.default_params,
            t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        sim::resolved_expr::eval_resolved(&sim::resolved_expr::ResolvedExpr::TimeFunc(0), &ctx);
    }

    /// Two policies installed one after the other in one process (as
    /// `compare` installs one per fit), each with a model compiled after it
    /// was installed and evaluated past its forcing's range: each is told of
    /// the forcing through the engine's run notes, and says it once however
    /// often its runs met it.
    #[test]
    fn each_policy_in_a_process_is_told_of_what_its_runs_held() {
        let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        sim::run_notes::set_reporter(crate::util::report_run_note);
        let names = |p: &ForcingPolicy| {
            let mut record = serde_json::json!({});
            p.record_in(&mut record);
            held(&record)
        };
        let first = ForcingPolicy::recording(true);
        {
            let _route = first.install();
            for model in [bspline_model(), bspline_model()] {
                evaluate_forcing(&model, 200.0);
                evaluate_forcing(&model, 300.0);
            }
        }
        let second = ForcingPolicy::recording(true);
        {
            let _route = second.install();
            evaluate_forcing(&bspline_model(), 200.0);
        }
        assert_eq!(names(&first), vec!["contact"], "said once by the first, though two of its models met it");
        assert_eq!(names(&second), vec!["contact"], "and told to the second");
    }

    /// A held forcing reaches a trace with what bounds its range and whether
    /// it is lagged.
    #[test]
    fn a_trace_warning_carries_the_bound_and_the_lag() {
        let policy = ForcingPolicy::new(true);
        policy.say(&OutsideRange { kind: RangeKind::Window, lagged: true, ..past_end("lagged") });
        assert_eq!(
            policy.trace_warnings(),
            vec![score::trace::PrequentialWarning::ForcingHeld {
                forcing: "lagged".into(),
                defined_on: [0.0, 50.0],
                side: score::trace::ForcingSide::After,
                held_value_at: 50.0,
                bound: Some(score::trace::ForcingBound::Window),
                lagged: Some(true),
            }]
        );
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

    /// A nested install (`camdl predict` installs the stored fit's policy
    /// inside the command's own) routes into the inner policy while its
    /// guard lives; dropping that guard puts the outer one back, and
    /// dropping the outer one empties the slot.
    #[test]
    fn a_nested_install_restores_the_outer_policy() {
        let _slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
        let installed = || ROUTE.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let outer = ForcingPolicy::new(true);
        let inner = ForcingPolicy::new(true);
        {
            let _outer_route = outer.install();
            {
                let _inner_route = inner.install();
                report(&past_end("inner"));
            }
            assert!(installed().is_some_and(|p| Arc::ptr_eq(&p.0, &outer.0)),
                "dropping the inner guard puts the outer policy back");
            report(&past_end("outer"));
        }
        assert!(installed().is_none(), "dropping the outer guard empties the slot");
        let names = |p: &ForcingPolicy| {
            let mut record = serde_json::json!({});
            p.record_in(&mut record);
            held(&record)
        };
        assert_eq!(names(&inner), vec!["inner"]);
        assert_eq!(names(&outer), vec!["outer"]);
    }

    /// A refusal kept while a thread that held the policy's locks panicked is
    /// still a refusal, and a held forcing still recorded: a poisoned lock
    /// does not read as "nothing refused" or "nothing held".
    #[test]
    fn a_poisoned_lock_does_not_hide_a_refusal_or_a_hold() {
        let refusing = ForcingPolicy::new(false);
        let allowing = ForcingPolicy::new(true);
        for p in [&refusing, &allowing] {
            let q = p.clone();
            let _ = std::thread::spawn(move || {
                let _r = q.0.refusal.lock().unwrap();
                let _h = q.0.held.lock().unwrap();
                panic!("poison the policy's locks");
            })
            .join();
            assert!(p.0.refusal.is_poisoned() && p.0.held.is_poisoned());
        }
        {
            let _deferred = refusing.defer_while_claimed();
            refusing.say(&past_end("refused"));
        }
        assert!(refusing.refused().is_some_and(|m| m.contains("refused")), "the refusal is kept");
        allowing.say(&past_end("held"));
        let mut record = serde_json::json!({});
        allowing.record_in(&mut record);
        assert_eq!(held(&record), vec!["held"], "the hold is recorded");
    }
}
