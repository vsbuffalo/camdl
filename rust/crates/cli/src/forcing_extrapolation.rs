//! What a command does when a run evaluates a forcing outside the range on
//! which it is defined (spec `docs/dev/temporal-architecture.md` §4.6, D44 and
//! D48): a data-table forcing before its first or after its last data point,
//! or a `bspline` outside its window.
//!
//! The engine holds the end value and records, once per forcing and side,
//! that it did, as a run note ([`sim::forcing_range`]). This module is the
//! policy. By default the note is an error: the command says which forcing,
//! its range and the side it was evaluated on, and exits before it commits
//! anything. Under `--allow-forcing-extrapolation` (or a fit config's
//! `[config] allow_forcing_extrapolation = true`) the note is said once as a
//! warning, and every `run.json` the command writes records the override and
//! the forcings it held.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use sim::forcing_range::OutsideRange;

/// The notice code of the warning, and of the error's kind.
pub const CODE: &str = "forcing-outside-range";

static ALLOWED: AtomicBool = AtomicBool::new(false);
static HELD: Mutex<Vec<OutsideRange>> = Mutex::new(Vec::new());

/// Set the policy for this process, before the command evaluates its model.
pub fn set_allowed(allowed: bool) {
    ALLOWED.store(allowed, Ordering::Relaxed);
}

/// Whether this run may hold a forcing's end value.
pub fn allowed() -> bool {
    ALLOWED.load(Ordering::Relaxed)
}

/// Said by the run-note reporter for each new [`OutsideRange`]: under the
/// override a warning, otherwise the error that ends the command.
pub fn report(o: &OutsideRange) {
    if !allowed() {
        crate::out::detached().fail(refusal(o));
    }
    if let Ok(mut held) = HELD.lock() {
        held.push(o.clone());
    }
    crate::out::live().notice(&crate::out::notice::Notice {
        severity: crate::out::notice::Severity::Warning,
        code: CODE,
        message: o.to_string(),
    });
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

/// Whether a stored leaf may not answer for this run: the leaf was made under
/// the override and held a forcing, and this run is not under it. Such a leaf
/// is a cache miss, so the run evaluates its model and is refused as a fresh
/// one would be, rather than served the held values. (A fit keys the override
/// in its config and never reaches this; a simulate leaf's identity does not
/// carry it.)
pub fn leaf_held_without_override(leaf: &std::path::Path) -> bool {
    let Ok(Some(l)) = runid::read_leaf(leaf) else { return false };
    held_without_override(&l.record)
}

/// [`leaf_held_without_override`] for a record already read, as a store
/// lookup returns it.
pub fn held_without_override(record: &runid::record::RunRecord) -> bool {
    if allowed() {
        return false;
    }
    record
        .inputs
        .pointer("/forcing_extrapolation/held")
        .and_then(|h| h.as_array())
        .is_some_and(|h| !h.is_empty())
}

/// Record the override, and the forcings held so far, in a `run.json`'s
/// display inputs. Nothing is added when the override is off: such a run held
/// nothing, or it was refused before it committed.
pub fn record_in(inputs: &mut serde_json::Value) {
    if !allowed() {
        return;
    }
    let Some(obj) = inputs.as_object_mut() else { return };
    let held: Vec<serde_json::Value> = HELD
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
