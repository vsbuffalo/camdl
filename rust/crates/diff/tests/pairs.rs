//! The diff of each fixture variant against the reference model, with the
//! answer stated in advance.
//!
//! Which models: `tests/fixtures/diff_pairs`. `base.camdl` is a
//! three-compartment SIR model with one observation stream; every other file
//! is that model with one thing changed, named in its header. The expected
//! rows follow from what the change is; the expected outcome follows from how
//! the compiler and the model hash behave, and is checked here against the
//! hashes of the committed compiled files, not assumed.
//!
//! Each test states every row that is not `Same` and every `Same` row that
//! reads something changed, so an extra row fails it as surely as a missing
//! one.

mod common;

use common::{compare, pair_file};
use diff::{AffectedBy, Category, DiffReport, ModelOutcome, Row, Status};

fn against_base(variant: &str) -> DiffReport {
    compare(&pair_file("base"), &pair_file(variant)).unwrap_or_else(|e| panic!("base vs {variant}: {e}"))
}

fn status_word(row: &Row) -> String {
    match &row.status {
        Status::Same => "same".to_string(),
        Status::Changed { fields } => {
            let paths: Vec<&str> = fields.iter().map(|f| f.field.as_str()).collect();
            format!("changed {}", paths.join(" "))
        }
        Status::OnlyLeft => "only-left".to_string(),
        Status::OnlyRight => "only-right".to_string(),
        Status::NotCompared { .. } => "not-compared".to_string(),
    }
}

fn affected_word(row: &Row) -> String {
    match &row.affected_by {
        AffectedBy::Known { changed } if changed.is_empty() => String::new(),
        AffectedBy::Known { changed } => {
            let names: Vec<&str> = changed.iter().map(|e| e.name.as_str()).collect();
            format!(" <- {}", names.join(" "))
        }
        AffectedBy::Unknown { .. } => " <- unknown".to_string(),
    }
}

/// Every row worth a reader's attention, one line each:
/// `<category> <name>: <status>[ <- <changed things it reads>]`.
fn notable(report: &DiffReport) -> Vec<String> {
    report
        .rows
        .iter()
        .filter(|r| !r.status.is_same() || !r.affected_by.is_clear())
        .map(|r| format!("{} {}: {}{}", r.category.label(), r.name, status_word(r), affected_word(r)))
        .collect()
}

#[test]
fn a_reformatted_file_is_identical() {
    let report = against_base("reformatted");
    assert_eq!(notable(&report), Vec::<String>::new());
    assert_eq!(report.model_outcome, ModelOutcome::Identical);
    assert!(report.identical());
    // The declarations are the same while their text is not: the row status
    // does not read the text.
    let infection = report.rows.iter().find(|r| r.name == "infection").expect("the infection row");
    assert_ne!(infection.left_text, infection.right_text);
}

#[test]
fn renaming_a_let_that_reads_a_parameter_is_written_differently() {
    let report = against_base("let_renamed");
    assert_eq!(
        notable(&report),
        [
            "definition foi: only-left",
            "definition lambda_t: only-right",
            "transition infection: changed dynamics.rate.rate <- foi lambda_t",
        ]
    );
    assert_eq!(report.model_outcome, ModelOutcome::WrittenDifferently);
    assert!(report.model_identical);
}

#[test]
fn renaming_a_let_that_reads_no_parameter_is_a_different_model() {
    let report = against_base("constant_let_renamed");
    assert_eq!(
        notable(&report),
        [
            "definition N: only-left",
            "definition foi: changed body <- N Npop",
            "definition Npop: only-right",
            "transition infection: same <- N Npop foi",
        ]
    );
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_changed_let_body_lights_the_rows_that_read_it() {
    let report = against_base("let_body_changed");
    assert_eq!(
        notable(&report),
        [
            "definition N: changed body",
            "definition foi: same <- N",
            "transition infection: same <- N",
        ]
    );
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn writing_a_let_out_in_full_is_written_differently() {
    let report = against_base("let_inlined");
    assert_eq!(notable(&report), ["transition infection: changed dynamics.rate.rate"]);
    assert_eq!(report.model_outcome, ModelOutcome::WrittenDifferently);
}

#[test]
fn a_changed_rate_is_one_row() {
    let report = against_base("rate_changed");
    assert_eq!(notable(&report), ["transition recovery: changed dynamics.rate.rate"]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn an_added_parameter_and_the_rate_that_reads_it() {
    let report = against_base("parameter_added");
    assert_eq!(
        notable(&report),
        ["parameter kappa: only-right", "transition recovery: changed dynamics.rate.rate"]
    );
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn reordered_transitions_are_one_order_row() {
    let report = against_base("transitions_reordered");
    assert_eq!(notable(&report), ["declaration order transition: changed order"]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_changed_prior_family_is_a_parameter_row() {
    let report = against_base("prior_changed");
    assert_eq!(notable(&report), [
            "parameter gamma: changed prior.args[0].name prior.args[0].value prior.args[1].name \
             prior.args[1].value prior.family"
        ]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_changed_observation_family() {
    let report = against_base("observation_family_changed");
    assert_eq!(
        notable(&report),
        [
            "parameter k: only-right",
            "observation stream cases: changed measurement.likelihood.args measurement.likelihood.family",
        ]
    );
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn an_added_scenario() {
    let report = against_base("scenario_added");
    assert_eq!(notable(&report), ["scenario baseline: only-right"]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_changed_simulation_end() {
    let report = against_base("simulation_end_changed");
    assert_eq!(notable(&report), ["simulation simulate: changed simulate.to"]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_compile_time_set_value_is_a_compile_input_row() {
    let report = against_base("base.set_beta");
    assert_eq!(notable(&report), ["compile input --set beta: only-right"]);
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}

#[test]
fn a_changed_quantity_leaves_the_model_identical() {
    let report = against_base("quantity_changed");
    assert_eq!(notable(&report), ["quantity peak_infectious: changed body"]);
    assert_eq!(report.model_outcome, ModelOutcome::Identical);
    assert!(report.model_identical);
    assert!(!report.reporting_identical);
    assert!(!report.identical());
    let row = report.rows.iter().find(|r| r.category == Category::Quantity).expect("the quantity row");
    assert_eq!(row.layer, diff::Layer::Reporting);
}

/// A level added to a stratifying dimension. No transition is written
/// differently, so there is no changed transition row: the dimension row, the
/// contact table and the new group's initial condition carry the whole
/// difference, and the infection rate is marked as reading the changed table.
#[test]
fn a_level_added_to_a_dimension_changes_no_transition_as_written() {
    let report = compare(&pair_file("age_base"), &pair_file("age_level_added")).expect("a report");
    assert_eq!(
        notable(&report),
        [
            "dimension age: changed levels",
            "table C_age: changed value",
            "transition infection: same <- C_age",
            "initial condition S[elder]: only-right",
        ]
    );
    assert_eq!(report.model_outcome, ModelOutcome::Different);
}
