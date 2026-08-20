//! `--emit-every`: a per-stream override of a model's `emit_schedule` cadence
//! (gh#656).
//!
//! `emit_schedule` decides at what times a FORWARD simulation emits synthetic
//! observations. It does **not** enter the likelihood — a fit against real data
//! scores at the data file's own times and never consults it. So one model can
//! serve a daily and a weekly emission without editing its source, which is what
//! this override is for: the prior-predictive band and the projections follow the
//! cadence, and because `incidence()` accumulates over the emit interval, a
//! weekly emit against daily observations puts the band roughly sevenfold off.
//!
//! **Identity.** The override is applied at the CONSUMPTION sites (see
//! [`apply_override`]), never by rematerializing the compiled IR the way
//! `--output-every` does. That distinction is the whole point: rewriting the IR
//! moves the model hash, so a `fit` against REAL data — where `emit_schedule` is
//! never read — would re-key and orphan a completed fit over a cosmetic emission
//! change. Instead each path keys what the override actually determines: the
//! emitted obs artifact ([`crate::batch::obs_subtree_hash`]) and, on the
//! `[synthetic]` path, the generated data's own bytes (which the fit already
//! hashes). A run without the flag keys exactly as it always did.

use std::borrow::Cow;
use std::collections::BTreeMap;

use ir::observation::{ObservationModel, ObservationSchedule, RegularSchedule};

/// A resolved `--emit-every` override.
///
/// Two surface forms, mutually exclusive in one invocation — the same grammar
/// `--data` uses ([`crate::util::resolve_data_specs`]):
///
/// - [`EmitEvery::All`] — `--emit-every N`, one cadence for every stream.
/// - [`EmitEvery::PerStream`] — `--emit-every NAME=N` (repeatable), keyed by the
///   stream's observation-block label (the IR `source`, what `--data NAME=PATH`
///   binds to), so one flag covers every leaf of a stratified family.
///
/// `N` is a plain number in the model's own time unit, never the DSL tick
/// spelling: `8 'weeks` is a shell-quoting hazard, and gh#626 already rejects
/// ticks on the CLI for that reason.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitEvery {
    /// `--emit-every N` — every schedule-bearing stream.
    All(f64),
    /// `--emit-every NAME=N` — the named streams only. `BTreeMap` so the
    /// identity rendering is deterministic.
    PerStream(BTreeMap<String, f64>),
}

const FLAG: &str = "--emit-every";

impl EmitEvery {
    /// Resolve repeatable `--emit-every` values. `Ok(None)` for an empty list
    /// (no flag given), so callers can keep their historical behaviour exactly.
    ///
    /// Bare and labelled forms do not compose: `--emit-every 7 --emit-every
    /// cases=1` is a hard error rather than a silent precedence rule.
    pub fn from_cli_specs(specs: &[String]) -> Result<Option<EmitEvery>, String> {
        if specs.is_empty() {
            return Ok(None);
        }
        // A spec is `NAME=N` only when the text before `=` looks like a stream
        // label: no whitespace, non-empty.
        let split = |raw: &String| -> Option<(String, String)> {
            let (label, value) = raw.split_once('=')?;
            let label = label.trim();
            if label.is_empty() || label.contains(char::is_whitespace) {
                return None;
            }
            Some((label.to_string(), value.trim().to_string()))
        };
        let n_named = specs.iter().filter(|s| split(s).is_some()).count();
        let n_bare = specs.len() - n_named;
        if n_named > 0 && n_bare > 0 {
            return Err(format!(
                "{FLAG} N and {FLAG} NAME=N are mutually exclusive; pick one \
                 form.\n  \
                 Use {FLAG} N to set every stream's cadence, or {FLAG} NAME=N \
                 (repeatable, one per stream) to set individual streams."
            ));
        }
        if n_named == 0 {
            if n_bare > 1 {
                return Err(format!(
                    "{FLAG} N given {n_bare} times; use one {FLAG} flag (the \
                     all-streams form takes a single cadence). For individual \
                     streams use {FLAG} NAME=N (repeatable)."
                ));
            }
            return Ok(Some(EmitEvery::All(parse_step(&specs[0])?)));
        }
        let mut map: BTreeMap<String, f64> = BTreeMap::new();
        for raw in specs {
            let (label, value) = split(raw).expect("all-named form checked above");
            let step = parse_step(&value)?;
            if map.insert(label.clone(), step).is_some() {
                return Err(format!("{FLAG}: stream '{label}' given twice."));
            }
        }
        Ok(Some(EmitEvery::PerStream(map)))
    }

    /// The cadence this override sets for the stream labelled `label` (its
    /// observation-block label / IR `source`), or `None` when it does not name
    /// that stream.
    pub fn resolve_for(&self, label: &str) -> Option<f64> {
        match self {
            EmitEvery::All(step) => Some(*step),
            EmitEvery::PerStream(map) => map.get(label).copied(),
        }
    }

    /// Validate the override against the model's declared observation streams,
    /// BEFORE anything runs. Three hard errors, each naming the stream:
    ///
    /// 1. a label that names no observation block — listing the valid labels;
    /// 2. a named stream that is fit-only (no `emit_schedule` at all), where the
    ///    flag has no cadence to override;
    /// 3. a stream whose declared schedule is an `at [...]` list — see
    ///    [`apply_override`] for why that is a refusal and not a conversion.
    ///
    /// (3) delegates to [`apply_override`], so the up-front check and the
    /// per-emission one cannot disagree about what is legal.
    pub fn validate(&self, observations: &[ObservationModel]) -> Result<(), String> {
        if let EmitEvery::PerStream(map) = self {
            for label in map.keys() {
                if !observations.iter().any(|o| &o.source == label) {
                    let mut labels: Vec<&str> =
                        observations.iter().map(|o| o.source.as_str()).collect();
                    labels.sort_unstable();
                    labels.dedup();
                    return Err(format!(
                        "{FLAG} {label}=…: '{label}' is not an observation \
                         stream. The label is the observation block's own name \
                         (what `--data NAME=PATH` binds to); valid labels are: \
                         {}.",
                        if labels.is_empty() {
                            "<the model declares no observation blocks>".to_string()
                        } else {
                            labels.join(", ")
                        }
                    ));
                }
            }
        }
        for obs in observations {
            match &obs.emit_schedule {
                Some(sched) => {
                    apply_override(Some(self), obs, sched)?;
                }
                // A fit-only stream has no cadence to override. Under the
                // all-streams form that is a no-op (it emits nothing either
                // way); named explicitly, the user asked for something the flag
                // cannot do, so say so.
                None => {
                    if let EmitEvery::PerStream(map) = self {
                        if map.contains_key(&obs.source) {
                            return Err(format!(
                                "{FLAG} {}=…: stream '{}' is fit-only — it \
                                 declares no `emit_schedule`, so there is no \
                                 emission cadence to override. Add \
                                 `emit_schedule = every N 'unit` to the \
                                 observation block to generate synthetic data \
                                 for it.",
                                obs.source, obs.name
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The canonical bytes this override contributes to an emitted artifact's
    /// content address.
    ///
    /// Distinct overrides must render distinctly, so the per-stream map is
    /// written out in its `BTreeMap` order with `\n` and `=` delimiters — neither
    /// can occur inside a stream label, which is a DSL identifier. `{:?}` on the
    /// step is the round-trippable float form, so two cadences that differ in the
    /// last bit render differently.
    ///
    /// Note `All(7)` and `PerStream({only_stream: 7})` render differently even on
    /// a one-stream model, where they emit identical bytes. That is
    /// over-invalidation (two addresses for one content), which costs a re-run;
    /// the alternative — normalizing against the model here — would make the
    /// address depend on which model it was computed against.
    pub fn identity_repr(&self) -> String {
        match self {
            EmitEvery::All(step) => format!("\nemit_every:all={step:?}"),
            EmitEvery::PerStream(map) => {
                let mut s = String::from("\nemit_every:per_stream");
                for (label, step) in map {
                    s.push_str(&format!("\n{label}={step:?}"));
                }
                s
            }
        }
    }
}

/// The schedule `obs` emits on once `emit` is applied — the model's own schedule
/// when the override does not name the stream.
///
/// **An `at [...]` list is refused, not converted.** Silently replacing a fixed
/// list of emission times with a cadence would change what the stream means (the
/// author listed specific times) while looking like a formatting flag, and for an
/// incidence stream it would also re-bin every accumulation interval.
pub fn apply_override<'a>(
    emit: Option<&EmitEvery>,
    obs: &ObservationModel,
    sched: &'a ObservationSchedule,
) -> Result<Cow<'a, ObservationSchedule>, String> {
    let Some(step) = emit.and_then(|e| e.resolve_for(&obs.source)) else {
        return Ok(Cow::Borrowed(sched));
    };
    match sched {
        // `end` rides along unchanged: since gh#143/gh#561 the runtime derives
        // emission from the RUN horizon and ignores this field.
        ObservationSchedule::Regular(reg) => {
            Ok(Cow::Owned(ObservationSchedule::Regular(RegularSchedule {
                start: reg.start,
                step,
                end: reg.end,
            })))
        }
        ObservationSchedule::AtTimes(times) => Err(format!(
            "{FLAG} cannot set the cadence of stream '{}': it declares \
             `emit_schedule = at [...]`, a fixed list of {} emission times, not \
             a recurring cadence. Converting the list to `every {step}` would \
             emit at different times than the model declares. Fix: change the \
             block to `emit_schedule = every N 'unit`, or drop '{}' from {FLAG}.",
            obs.name,
            times.len(),
            obs.source
        )),
    }
}

/// Parse one cadence value: a plain, positive, finite number in the model's own
/// time unit.
///
/// The DSL tick spelling (`8 'weeks`) is rejected with a hint rather than
/// accepted, following gh#626's `--to`: a leading `'` is a shell-quoting hazard,
/// so the CLI grammar and the DSL grammar are deliberately not the same here.
fn parse_step(raw: &str) -> Result<f64, String> {
    let s = raw.trim();
    if let Ok(v) = s.parse::<f64>() {
        if !v.is_finite() || v <= 0.0 {
            return Err(format!(
                "{FLAG} = \"{raw}\": the cadence must be a positive, finite \
                 number of model time units; got {s}."
            ));
        }
        return Ok(v);
    }
    // `N unit` / `N 'unit` — name the plain-number form rather than failing as
    // "not a number".
    let mut it = s.split_whitespace();
    if let (Some(n_tok), Some(unit_tok), None) = (it.next(), it.next(), it.next()) {
        if n_tok.parse::<f64>().is_ok() {
            let unit = unit_tok.trim_start_matches('\'');
            return Err(format!(
                "{FLAG} = \"{raw}\": a unit is not accepted here (nor the DSL \
                 tick spelling '{unit}) — the cadence is a plain number in the \
                 model's own `time_unit`. On a `time_unit = 'days` model, \
                 `every {n_tok} '{unit}` is `{FLAG} <that many days>`."
            ));
        }
    }
    Err(format!(
        "{FLAG} = \"{raw}\": expected a positive number of model time units \
         (e.g. `{FLAG} 7`), or `{FLAG} NAME=7` to set one stream."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::deriv::Diffable;
    use ir::expr::{ConstExpr, Expr};
    use ir::observation::{Likelihood, PoissonLikelihood, Projection};

    fn spec(s: &str) -> String {
        s.to_string()
    }

    fn obs(name: &str, source: &str, sched: Option<ObservationSchedule>) -> ObservationModel {
        ObservationModel {
            name: name.to_string(),
            source: source.to_string(),
            columns: Vec::new(),
            scored: name.to_string(),
            emit_schedule: sched,
            stratum: Vec::new(),
            projection: Projection::CurrentPop("I".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: Diffable::new(Expr::Const(ConstExpr { value: 1.0 })),
            }),
        }
    }

    fn regular(start: f64, step: f64) -> ObservationSchedule {
        ObservationSchedule::Regular(RegularSchedule { start, step, end: 100.0 })
    }

    #[test]
    fn empty_specs_resolve_to_no_override() {
        assert_eq!(EmitEvery::from_cli_specs(&[]).unwrap(), None);
    }

    #[test]
    fn bare_form_sets_every_stream() {
        let e = EmitEvery::from_cli_specs(&[spec("7")]).unwrap().unwrap();
        assert_eq!(e, EmitEvery::All(7.0));
        assert_eq!(e.resolve_for("cases"), Some(7.0));
        assert_eq!(e.resolve_for("deaths"), Some(7.0));
    }

    #[test]
    fn labelled_form_sets_only_the_named_stream() {
        let e = EmitEvery::from_cli_specs(&[spec("cases=7")]).unwrap().unwrap();
        assert_eq!(e.resolve_for("cases"), Some(7.0));
        assert_eq!(e.resolve_for("deaths"), None, "a sibling stream is untouched");
    }

    #[test]
    fn bare_and_labelled_forms_together_are_refused() {
        let err = EmitEvery::from_cli_specs(&[spec("7"), spec("cases=1")]).unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "mixed forms must be refused, not silently precedence-ordered: {err}"
        );
    }

    #[test]
    fn two_bare_cadences_are_refused() {
        let err = EmitEvery::from_cli_specs(&[spec("7"), spec("14")]).unwrap_err();
        assert!(err.contains("given 2 times"), "{err}");
    }

    #[test]
    fn a_repeated_label_is_refused() {
        let err = EmitEvery::from_cli_specs(&[spec("cases=7"), spec("cases=1")]).unwrap_err();
        assert!(err.contains("given twice"), "{err}");
    }

    #[test]
    fn the_dsl_tick_spelling_is_refused_with_a_plain_number_hint() {
        let err = EmitEvery::from_cli_specs(&[spec("8 'weeks")]).unwrap_err();
        assert!(
            err.contains("tick") && err.contains("plain number"),
            "a tick unit must hint the plain-number spelling: {err}"
        );
    }

    #[test]
    fn a_non_positive_or_non_finite_cadence_is_refused() {
        for bad in ["0", "-7", "inf", "NaN"] {
            let err = EmitEvery::from_cli_specs(&[spec(bad)]).unwrap_err();
            assert!(
                err.contains("positive") || err.contains("expected"),
                "'{bad}' must be refused: {err}"
            );
        }
    }

    #[test]
    fn an_unknown_label_lists_the_valid_ones() {
        let model = vec![
            obs("cases", "cases", Some(regular(0.0, 1.0))),
            obs("deaths", "deaths", Some(regular(0.0, 1.0))),
        ];
        let e = EmitEvery::from_cli_specs(&[spec("caes=7")]).unwrap().unwrap();
        let err = e.validate(&model).unwrap_err();
        assert!(err.contains("'caes' is not an observation stream"), "{err}");
        assert!(err.contains("cases") && err.contains("deaths"), "must list valid labels: {err}");
    }

    #[test]
    fn a_family_root_covers_every_expanded_leaf() {
        // A stratified `cases[p in patch]` expands to leaves that share one
        // `source`, so one flag sets the whole family.
        let model = vec![
            obs("cases_p1", "cases", Some(regular(0.0, 1.0))),
            obs("cases_p2", "cases", Some(regular(0.0, 1.0))),
        ];
        let e = EmitEvery::from_cli_specs(&[spec("cases=7")]).unwrap().unwrap();
        e.validate(&model).expect("the family root is a valid label");
        for o in &model {
            let s = o.emit_schedule.clone().unwrap();
            let applied = apply_override(Some(&e), o, &s).unwrap();
            assert_eq!(*applied, regular(0.0, 7.0));
        }
    }

    #[test]
    fn an_at_list_stream_is_refused_by_name() {
        let model = vec![
            obs("cases", "cases", Some(ObservationSchedule::AtTimes(vec![1.0, 5.0, 9.0]))),
        ];
        let e = EmitEvery::from_cli_specs(&[spec("7")]).unwrap().unwrap();
        let err = e.validate(&model).unwrap_err();
        assert!(err.contains("'cases'"), "must name the stream: {err}");
        assert!(err.contains("at [...]"), "must name the declared form: {err}");
    }

    #[test]
    fn a_fit_only_stream_is_a_no_op_under_the_bare_form_and_refused_when_named() {
        let model = vec![obs("cases", "cases", None)];
        EmitEvery::from_cli_specs(&[spec("7")])
            .unwrap()
            .unwrap()
            .validate(&model)
            .expect("the all-streams form skips a stream that emits nothing");
        let err = EmitEvery::from_cli_specs(&[spec("cases=7")])
            .unwrap()
            .unwrap()
            .validate(&model)
            .unwrap_err();
        assert!(err.contains("fit-only"), "{err}");
    }

    #[test]
    fn an_unnamed_stream_keeps_its_declared_cadence() {
        let o = obs("deaths", "deaths", Some(regular(0.0, 3.0)));
        let s = o.emit_schedule.clone().unwrap();
        let e = EmitEvery::from_cli_specs(&[spec("cases=7")]).unwrap().unwrap();
        let applied = apply_override(Some(&e), &o, &s).unwrap();
        assert!(matches!(applied, Cow::Borrowed(_)), "an untouched stream must not clone");
        assert_eq!(*applied, regular(0.0, 3.0));
    }

    #[test]
    fn no_override_borrows_the_declared_schedule() {
        let o = obs("cases", "cases", Some(regular(0.0, 1.0)));
        let s = o.emit_schedule.clone().unwrap();
        let applied = apply_override(None, &o, &s).unwrap();
        assert!(matches!(applied, Cow::Borrowed(_)));
    }

    #[test]
    fn the_override_keeps_the_schedule_start() {
        // The cadence is re-anchored to the DECLARED start, never to t_start —
        // a stream that begins emitting at t = 3 still begins at t = 3.
        let o = obs("cases", "cases", Some(regular(3.0, 1.0)));
        let s = o.emit_schedule.clone().unwrap();
        let e = EmitEvery::All(7.0);
        let applied = apply_override(Some(&e), &o, &s).unwrap();
        assert_eq!(*applied, regular(3.0, 7.0));
    }

    #[test]
    fn distinct_overrides_render_distinct_identity_bytes() {
        let a = EmitEvery::All(7.0);
        let b = EmitEvery::All(14.0);
        let c = EmitEvery::from_cli_specs(&[spec("cases=7")]).unwrap().unwrap();
        let d = EmitEvery::from_cli_specs(&[spec("deaths=7")]).unwrap().unwrap();
        let reprs = [a.identity_repr(), b.identity_repr(), c.identity_repr(), d.identity_repr()];
        for i in 0..reprs.len() {
            for j in (i + 1)..reprs.len() {
                assert_ne!(reprs[i], reprs[j], "overrides {i} and {j} must not alias");
            }
        }
    }

    #[test]
    fn identity_bytes_are_order_independent_for_the_labelled_form() {
        let a = EmitEvery::from_cli_specs(&[spec("cases=7"), spec("deaths=1")])
            .unwrap()
            .unwrap();
        let b = EmitEvery::from_cli_specs(&[spec("deaths=1"), spec("cases=7")])
            .unwrap()
            .unwrap();
        assert_eq!(a.identity_repr(), b.identity_repr(), "flag order is not identity");
    }
}
