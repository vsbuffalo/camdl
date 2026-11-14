//! The OCaml codec and these types agree on the wire format.
//!
//! `fixtures/all_variants.json` is written by the OCaml side
//! (`ocaml/lib/ir/declared.ml`, from the hand-written value in
//! `ocaml/test/declared_fixture.ml`, via `make update-declared-fixture`). It
//! holds two documents: `all_variants`, which uses every type and every enum
//! variant at least once, and `minimal`, with every optional absent and every
//! list empty. Here each is read through the Rust types and written back, and
//! the result must be the JSON that was read. The OCaml test
//! `ocaml/test/test_declared.ml` checks the same file against the same value.

use declared::*;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/all_variants.json");

fn document(key: &str) -> Value {
    let all: Value = serde_json::from_str(FIXTURE).expect("the fixture is JSON");
    all.get(key).unwrap_or_else(|| panic!("the fixture has no `{key}` document")).clone()
}

fn read(key: &str) -> DeclaredModel {
    serde_json::from_value(document(key))
        .unwrap_or_else(|e| panic!("`{key}` does not read as a DeclaredModel: {e}"))
}

/// Reading a document written by OCaml and writing it back gives the same
/// JSON: no field renamed, dropped, added or retyped on either side.
#[test]
fn rust_reads_and_rewrites_what_ocaml_wrote() {
    for key in ["all_variants", "minimal"] {
        let written = serde_json::to_value(read(key)).expect("serializes");
        assert_eq!(written, document(key), "`{key}` changed across a Rust round trip");
    }
}

#[test]
fn the_fixture_is_at_the_schema_version_this_crate_writes() {
    assert_eq!(read("all_variants").schema_version.get(), SCHEMA_VERSION);
    assert_eq!(read("minimal").schema_version.get(), SCHEMA_VERSION);
}

/// `minimal` is the absent case of every optional and the empty case of every
/// list.
#[test]
fn minimal_has_every_optional_absent() {
    let m = read("minimal");
    assert!(m.time_unit.is_none() && m.description.is_none() && m.origin.is_none());
    assert!(m.simulation.is_none());
    assert_eq!(m.output, Some(OutputDecl { trajectories: None, source_text: None }));
    assert!(m.dimensions.is_empty() && m.transitions.is_empty() && m.contrasts.is_empty());
}

/// Every variant of every expression-level enum appears in `all_variants`.
/// Checked on the compact serialization, where each variant has one spelling;
/// the patterns rely on serde writing a struct's fields in declaration order.
#[test]
fn all_variants_uses_every_expression_level_variant() {
    let text = serde_json::to_string(&read("all_variants")).expect("serializes");
    let mut patterns: Vec<String> = Vec::new();
    for tag in [
        "const", "unit", "ident", "index", "bin_op", "un_op", "sum", "cond", "call", "list",
        "range", "obs_access", "run_member", "str",
    ] {
        patterns.push(format!("{{\"{tag}\":{{"));
    }
    for kind in [
        "compartment", "parameter", "definition", "table", "time_input", "index_var",
        "level", "builtin", "data_column", "quantity", "stream", "transition", "dimension",
        "obs_anchor", "keyword", "unresolved",
    ] {
        patterns.push(format!("\"kind\":\"{kind}\"}}"));
    }
    for unit in [
        "days", "weeks", "months", "years", "per_day", "per_week", "per_month", "per_year",
        "count", "ratio",
    ] {
        patterns.push(format!("\"unit\":\"{unit}\""));
    }
    for op in ["add", "sub", "mul", "div", "pow", "eq", "neq", "lt", "gt", "le", "ge"] {
        patterns.push(format!("\"op\":\"{op}\",\"lhs\""));
    }
    for op in
        ["neg", "exp", "log", "sqrt", "abs", "floor", "ceil", "sin", "cos", "tanh", "total"]
    {
        patterns.push(format!("\"op\":\"{op}\",\"arg\""));
    }
    for op in ["lt", "le", "gt", "ge", "eq", "ne"] {
        patterns.push(format!("\"op\":\"{op}\",\"operand\""));
    }
    for literal in [
        "\"namespace\":\"quantities\"",
        "\"namespace\":\"observations\"",
        "{\"positional\":{",
        "{\"named\":{",
        "{\"in\":{\"var\"",
        "{\"consecutive\":{\"var\"",
        "{\"compartments\":{\"var\"",
        "{\"eq\":{\"lhs\":\"",
        "{\"neq\":{\"lhs\":\"",
        "{\"table\":{\"table\"",
        "{\"and\":{\"lhs\":{\"eq\"",
        "\"guard\":{\"or\":{",
        "{\"num\":{\"value\"",
        "{\"name\":{\"name\"",
    ] {
        patterns.push(literal.to_string());
    }
    let missing: Vec<&String> = patterns.iter().filter(|p| !text.contains(p.as_str())).collect();
    assert!(missing.is_empty(), "the fixture has no instance of: {missing:?}");
}

/// Every variant of every declaration-level enum appears in `all_variants`.
#[test]
fn all_variants_uses_every_declaration_level_variant() {
    let m = read("all_variants");

    assert!(m.dimensions.iter().any(|d| d.source == DimSource::Inline));
    assert!(m.dimensions.iter().any(|d| matches!(d.source, DimSource::Read { .. })));
    for kind in [CompartmentKind::Integer, CompartmentKind::Real] {
        assert!(m.compartments.iter().any(|c| c.kind == kind), "{kind:?}");
    }
    let param_kinds = [
        ParamKind::Rate,
        ParamKind::Probability,
        ParamKind::Count,
        ParamKind::Positive,
        ParamKind::Real,
        ParamKind::Instant,
        ParamKind::Duration,
    ];
    for kind in param_kinds {
        assert!(m.parameters.iter().any(|p| p.kind == kind), "{kind:?}");
    }

    let arg_values = || m.time_inputs.iter().flat_map(|t| t.args.iter().map(|a| &a.value));
    assert!(arg_values().any(|v| matches!(v, TimeInputValue::Str { .. })));
    assert!(arg_values().any(|v| matches!(v, TimeInputValue::Expr { .. })));

    assert!(m.transitions.iter().any(|t| matches!(t.to, Destination::Sum { .. })));
    assert!(m.transitions.iter().any(|t| matches!(t.to, Destination::Branch { .. })));
    assert!(m.transitions.iter().any(|t| matches!(t.dynamics, Dynamics::Rate { .. })));
    assert!(m.transitions.iter().any(|t| matches!(t.dynamics, Dynamics::Via { .. })));

    assert!(m.initial.iter().any(|i| matches!(i.value, InitialValue::Expr { .. })));
    for family in [
        Family::NegBinomial,
        Family::Poisson,
        Family::Normal,
        Family::Binomial,
        Family::BetaBinomial,
        Family::Beta,
        Family::Bernoulli,
        Family::ZeroInflatedNegBinomial,
    ] {
        assert!(
            m.initial.iter().any(
                |i| matches!(&i.value, InitialValue::Law { law } if law.family == family)
            ),
            "{family:?}"
        );
    }

    let roles = || m.observations.iter().flat_map(|o| o.columns.iter().map(|c| &c.role));
    assert!(roles().any(|r| *r == ColumnRole::Time));
    assert!(roles().any(|r| matches!(r, ColumnRole::Dim { .. })));
    assert!(roles().any(|r| *r == ColumnRole::WindowStart));
    assert!(roles().any(|r| *r == ColumnRole::WindowStop));
    for kind in param_kinds {
        assert!(roles().any(|r| *r == ColumnRole::Value { kind }), "{kind:?}");
    }
    assert!(m.observations.iter().any(|o| matches!(o.projection, Projection::Incidence { .. })));
    assert!(m.observations.iter().any(|o| matches!(o.projection, Projection::Derived { .. })));
    let emits = || m.observations.iter().filter_map(|o| o.emit_schedule.as_ref());
    assert!(emits().any(|s| matches!(s, ScheduleCore::Every { .. })));
    assert!(emits().any(|s| matches!(s, ScheduleCore::At { .. })));

    let entries = || m.interventions.iter().chain(m.events.iter());
    let actions = || {
        entries()
            .flat_map(|i| i.actions.iter())
            .chain(m.reactive_interventions.iter().map(|r| &r.action))
    };
    assert!(actions().any(|a| matches!(a, Action::Transfer { .. })));
    assert!(actions().any(|a| matches!(a, Action::Set { .. })));
    assert!(actions().any(|a| matches!(a, Action::Add { .. })));
    assert!(entries().any(|i| matches!(i.schedule, Schedule::AtTimes { .. })));
    assert!(entries().any(|i| matches!(i.schedule, Schedule::Recurring { .. })));
    assert!(entries().any(|i| matches!(i.schedule, Schedule::EveryAtDay { .. })));

    fn trigger_tags(p: &TriggerPred, seen: &mut [bool; 4]) {
        match p {
            TriggerPred::Atom { .. } => seen[0] = true,
            TriggerPred::And { lhs, rhs } => {
                seen[1] = true;
                trigger_tags(lhs, seen);
                trigger_tags(rhs, seen);
            }
            TriggerPred::Or { lhs, rhs } => {
                seen[2] = true;
                trigger_tags(lhs, seen);
                trigger_tags(rhs, seen);
            }
            TriggerPred::Not { arg } => {
                seen[3] = true;
                trigger_tags(arg, seen);
            }
        }
    }
    let mut seen = [false; 4];
    for r in &m.reactive_interventions {
        trigger_tags(&r.when, &mut seen);
    }
    assert_eq!(seen, [true; 4], "trigger variants atom, and, or, not");

    let fields = || m.scenarios.iter().flat_map(|s| s.fields.iter());
    assert!(fields().any(|f| matches!(f, ScenarioField::Label { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Enable { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Disable { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Set { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Scale { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Compose { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::TEnd { .. })));
    assert!(fields().any(|f| matches!(f, ScenarioField::Extends { .. })));
}

/// A declaration's text as written crosses the two codecs byte for byte: a line
/// break, a tab, a quoted string, a backslash and a non-ASCII character are
/// each read back as the one character the OCaml side wrote. `None` is the
/// "not recorded" case, and it is `null` on the wire, never an empty string.
#[test]
fn source_text_is_read_back_exactly() {
    let m = read("all_variants");
    assert_eq!(
        m.transitions[0].source_text.as_deref(),
        Some(
            "#[lineage] infection[a in age] : S[a] --> I[a] + R[age = adult]\n\
             \t  @ beta × S[a]   # per \"capita\" \\ rate\n\
             \t  where a != b"
        )
    );
    assert_eq!(m.compartments[1].source_text, None);
    assert_eq!(document("all_variants")["compartments"][1]["source_text"], Value::Null);
}
