//! The diff's invariants, on every compiled golden and on the fixture pairs.
//!
//! Numbered as in `docs/dev/proposals/2026-11-13-model-structure-diff.md`,
//! "Invariants". Source: the committed compiled models under `ocaml/golden` and
//! `tests/fixtures/*/ir` (primary: what the compiler emits), each read once.

mod common;

use common::{compare, golden_files, pair_file, Compiled};
use declared::{DeclaredError, DeclaredModel, DeclaredSection, FileInput};
use diff::{AffectedBy, Category, DiffReport, ModelOutcome, Refusal, Row, Status, Which};

fn goldens() -> Vec<Compiled> {
    let all: Vec<Compiled> = golden_files().iter().map(|p| Compiled::read(p)).collect();
    assert!(all.len() > 90, "expected the whole golden set, found {} files", all.len());
    all
}

fn recorded(c: &Compiled) -> &DeclaredModel {
    c.declared.recorded().unwrap_or_else(|e| panic!("{}: {e}", c.name))
}

fn with_declared(c: &Compiled, model: DeclaredModel) -> Compiled {
    Compiled {
        name: c.name.clone(),
        declared: DeclaredSection::Recorded(Box::new(model)),
        model_hash: c.model_hash.clone(),
        compiler: c.compiler.clone(),
    }
}

/// Invariant 1: a model against itself has only `Same` rows that read nothing
/// changed, and is identical.
#[test]
fn a_model_against_itself_is_identical() {
    for g in goldens() {
        let report = compare(&g, &g).unwrap_or_else(|e| panic!("{}: {e}", g.name));
        for row in &report.rows {
            assert!(row.status.is_same(), "{}: {} {} is not Same", g.name, row.category.label(), row.name);
            assert!(row.affected_by.is_clear(), "{}: {} reads something changed", g.name, row.name);
        }
        assert_eq!(report.model_outcome, ModelOutcome::Identical, "{}", g.name);
        assert!(report.identical(), "{}", g.name);
    }
}

/// A row with left and right exchanged.
fn mirrored(row: &Row) -> Row {
    let mut m = row.clone();
    std::mem::swap(&mut m.left_text, &mut m.right_text);
    m.status = match &row.status {
        Status::OnlyLeft => Status::OnlyRight,
        Status::OnlyRight => Status::OnlyLeft,
        Status::Changed { fields } => Status::Changed {
            fields: fields
                .iter()
                .map(|f| diff::FieldChange { field: f.field.clone(), left: f.right.clone(), right: f.left.clone() })
                .collect(),
        },
        other => other.clone(),
    };
    m
}

fn sorted(mut rows: Vec<Row>) -> Vec<Row> {
    rows.sort_by(|a, b| (a.layer, a.category, &a.name).cmp(&(b.layer, b.category, &b.name)));
    rows
}

const VARIANTS: [&str; 14] = [
    "reformatted",
    "let_renamed",
    "constant_let_renamed",
    "let_body_changed",
    "let_inlined",
    "rate_changed",
    "parameter_added",
    "transitions_reordered",
    "prior_changed",
    "observation_family_changed",
    "scenario_added",
    "simulation_end_changed",
    "quantity_changed",
    "base.set_beta",
];

/// Invariant 2: exchanging the two models exchanges only-left with only-right,
/// and the two sides of each change, and nothing else. Rows are compared as a
/// set: their order follows the left model's declarations.
#[test]
fn exchanging_the_models_mirrors_the_report() {
    let base = pair_file("base");
    for variant in VARIANTS {
        let other = pair_file(variant);
        let forward = compare(&base, &other).expect("base against a variant");
        let backward = compare(&other, &base).expect("a variant against base");
        assert_eq!(forward.model_outcome, backward.model_outcome, "{variant}");
        assert_eq!(forward.reporting_identical, backward.reporting_identical, "{variant}");
        let mirrored_back: Vec<Row> = backward.rows.iter().map(mirrored).collect();
        // A message of a not-compared row names a side, and no fixture has one.
        assert_eq!(sorted(forward.rows.clone()), sorted(mirrored_back), "{variant}");
    }
}

/// Invariant 3, over every pair of compiled goldens. 3a: `model_identical` is
/// the comparison of the two hashes. 3b: declarations that are all the same
/// never sit over different compiled models; the diff refuses that case, so the
/// assertion is that no pair is refused. 3c: equal hashes under differing
/// declarations are reported as written differently, rows kept.
#[test]
fn declared_rows_agree_with_the_compiled_models_on_every_pair() {
    let all = goldens();
    let mut identical = Vec::new();
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            let report: DiffReport = match compare(a, b) {
                Ok(r) => r,
                Err(e) => panic!("{} against {}: {e}", a.name, b.name),
            };
            let same_hash = a.model_hash == b.model_hash;
            assert_eq!(report.model_identical, same_hash, "{} against {}", a.name, b.name);
            let rows_same =
                report.rows.iter().filter(|r| r.layer == diff::Layer::Model).all(|r| r.status.is_same());
            let expected = match (rows_same, same_hash) {
                (true, true) => ModelOutcome::Identical,
                (false, true) => ModelOutcome::WrittenDifferently,
                (false, false) => ModelOutcome::Different,
                (true, false) => unreachable!("refused above"),
            };
            assert_eq!(report.model_outcome, expected, "{} against {}", a.name, b.name);
            if report.model_outcome == ModelOutcome::Identical {
                identical.push(format!("{} = {}", a.name, b.name));
            }
        }
    }
    // The test bites in both directions: some pairs of distinct files are one
    // model as declared and as compiled, among them the reformatted fixture
    // and one source compiled with and without `--no-state-grad` (gh#1046).
    assert!(identical.contains(&"base.ir.json = reformatted.ir.json".to_string()), "{identical:?}");
    assert!(
        identical.contains(&"hoisted_rate.ir.json = hoisted_rate.no_state_grad.ir.json".to_string()),
        "{identical:?}"
    );
}

/// Invariant 5: every declaration of each model has exactly one row. Counted
/// from the declared model's own lists: one row for each entry, plus the three
/// rows every model has (description, simulation, output).
#[test]
fn every_declaration_has_exactly_one_row() {
    fn declarations(m: &DeclaredModel) -> usize {
        m.compile_inputs.set_values.len()
            + m.compile_inputs.files.len()
            + m.dimensions.len()
            + m.compartments.len()
            + m.stratify.len()
            + m.parameters.len()
            + m.definitions.len()
            + m.tables.len()
            + m.time_inputs.len()
            + m.timepoints.len()
            + m.transitions.len()
            + m.ode_equations.len()
            + m.balance.len()
            + m.initial.len()
            + m.observations.len()
            + m.interventions.len()
            + m.events.len()
            + m.reactive_interventions.len()
            + m.scenarios.len()
            + m.quantities.len()
            + m.contrasts.len()
            + 3
    }
    let base = pair_file("base");
    for g in goldens() {
        let report = match compare(&base, &g) {
            Ok(r) => r,
            Err(e) => panic!("base against {}: {e}", g.name),
        };
        let rows: Vec<&Row> = report.rows.iter().filter(|r| r.category != Category::DeclarationOrder).collect();
        let on_left = rows.iter().filter(|r| !matches!(r.status, Status::OnlyRight)).count();
        let on_right = rows.iter().filter(|r| !matches!(r.status, Status::OnlyLeft)).count();
        assert_eq!(on_left, declarations(recorded(&base)), "left rows, against {}", g.name);
        assert_eq!(on_right, declarations(recorded(&g)), "right rows, {}", g.name);
        let mut names: Vec<(Category, &str)> = rows.iter().map(|r| (r.category, r.name.as_str())).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "a row name is repeated against {}", g.name);
    }
}

/// Invariant 8: the report is the same bytes on every run.
#[test]
fn the_report_is_the_same_bytes_on_every_run() {
    let base = pair_file("base");
    for variant in VARIANTS {
        let other = pair_file(variant);
        let once = serde_json::to_string(&compare(&base, &other).expect("a report")).expect("serialize");
        let twice = serde_json::to_string(&compare(&base, &other).expect("a report")).expect("serialize");
        assert_eq!(once, twice, "{variant}");
        let read_back: DiffReport = serde_json::from_str(&once).expect("the report reads back");
        assert_eq!(serde_json::to_string(&read_back).expect("serialize"), once, "{variant}");
    }
}

#[test]
fn a_side_without_a_readable_declared_model_is_refused_and_named() {
    let base = pair_file("base");
    let bare = Compiled {
        name: "bare.ir.json".to_string(),
        declared: DeclaredSection::NotRecorded,
        model_hash: base.model_hash.clone(),
        compiler: None,
    };
    assert_eq!(
        compare(&base, &bare),
        Err(Refusal::DeclaredUnavailable {
            which: Which::Right,
            name: "bare.ir.json".to_string(),
            why: DeclaredError::NotRecorded,
        })
    );
    let newer: DeclaredSection =
        serde_json::from_str(r#"{"schema_version": 99, "anything": true}"#).expect("an unsupported section reads");
    let unsupported = Compiled { declared: newer, ..bare };
    match compare(&unsupported, &base) {
        Err(Refusal::DeclaredUnavailable {
            which: Which::Left,
            why: DeclaredError::UnsupportedVersion { found: 99, .. },
            ..
        }) => {}
        other => panic!("expected a refusal naming version 99, got {other:?}"),
    }
}

/// The fourth outcome: the same declarations over different compiled models.
/// For one compiler it is refused with both hashes; for two it is reported as
/// such. It is never "identical".
#[test]
fn same_declarations_over_different_models_are_never_identical() {
    let base = pair_file("base");
    let other_hash = "0".repeat(64);
    let moved = Compiled {
        name: "moved".to_string(),
        declared: base.declared.clone(),
        model_hash: other_hash.clone(),
        compiler: base.compiler.clone(),
    };
    assert_eq!(
        compare(&base, &moved),
        Err(Refusal::SameDeclarationsDifferentModels {
            left_hash: base.model_hash.clone(),
            right_hash: other_hash.clone(),
        })
    );
    // Not known to be different compilers: still refused.
    let unknown = Compiled { compiler: None, name: "unknown".to_string(), declared: base.declared.clone(), model_hash: other_hash.clone() };
    assert!(compare(&base, &unknown).is_err());

    let newer = Compiled {
        compiler: Some("another-compiler".to_string()),
        name: "newer".to_string(),
        declared: base.declared.clone(),
        model_hash: other_hash,
    };
    let report = compare(&base, &newer).expect("two compilers are reported, not refused");
    assert_eq!(
        report.model_outcome,
        ModelOutcome::SameDeclarationsDifferentCompilers {
            left: base.compiler.clone().expect("the fixture names its compiler"),
            right: "another-compiler".to_string(),
        }
    );
    assert!(!report.model_identical);
    assert!(!report.identical());
}

/// A row that reads a name the compiler could not classify has unknown
/// dependencies, and so does every row that reads it through a definition. It
/// never has an empty `affected_by`, which would say nothing it reads changed.
#[test]
fn an_unresolved_name_makes_dependencies_unknown() {
    let base = pair_file("base");
    let text = serde_json::to_string(recorded(&base)).expect("serialize");
    let resolved = r#"{"name":"N","kind":"definition"}"#;
    assert!(text.contains(resolved), "the fixture's force of infection reads N");
    let edited = text.replace(resolved, r#"{"name":"N","kind":"unresolved"}"#);
    let right = with_declared(&base, DeclaredModel::from_json_str(&edited).expect("the edited section reads"));

    let report = compare(&base, &right).expect("a report");
    let row = |name: &str| report.rows.iter().find(|r| r.name == name).unwrap_or_else(|| panic!("no row {name}"));
    let unknown = |name: &str| match &row(name).affected_by {
        AffectedBy::Unknown { unresolved, .. } => unresolved.clone(),
        AffectedBy::Known { .. } => panic!("{name} claims to know its dependencies"),
    };
    assert_eq!(unknown("foi"), ["N"]);
    assert_eq!(unknown("infection"), ["N"], "through the definition foi");
    assert!(row("infection").status.is_same());
    assert!(row("recovery").affected_by.is_clear(), "recovery reads neither");
}

/// A file the compiler read is the same only when both digests are recorded
/// and equal. Without a digest the row is not compared, and the models are not
/// reported as identical.
#[test]
fn a_file_without_a_digest_is_never_the_same() {
    let base = pair_file("base");
    let with_file = |sha256: Option<&str>| {
        let mut model = recorded(&base).clone();
        model
            .compile_inputs
            .files
            .push(FileInput { path: "contacts.tsv".to_string(), sha256: sha256.map(str::to_string) });
        with_declared(&base, model)
    };
    let file_row = |report: &DiffReport| {
        report.rows.iter().find(|r| r.name == "file contacts.tsv").expect("the file's row").status.clone()
    };

    let both = compare(&with_file(Some("aa")), &with_file(Some("aa"))).expect("a report");
    assert_eq!(file_row(&both), Status::Same);
    assert_eq!(both.model_outcome, ModelOutcome::Identical);

    let differ = compare(&with_file(Some("aa")), &with_file(Some("bb"))).expect("a report");
    assert!(matches!(file_row(&differ), Status::Changed { .. }));

    for (left, right) in [(None, Some("aa")), (Some("aa"), None), (None, None)] {
        let report = compare(&with_file(left), &with_file(right)).expect("a report");
        assert!(matches!(file_row(&report), Status::NotCompared { .. }), "{left:?} {right:?}");
        assert_ne!(report.model_outcome, ModelOutcome::Identical, "{left:?} {right:?}");
    }
}
