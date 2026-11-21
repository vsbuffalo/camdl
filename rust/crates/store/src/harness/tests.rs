use super::*;

/// A harness whose runs return fixed observations: the base run, and the run
/// with the field varied.
struct Fixed {
    base: Observed,
    varied: Observed,
}

impl InputHarness for Fixed {
    type Invocation = bool;
    fn base(&self) -> bool {
        false
    }
    fn varied(&self, _field: &FieldInfo) -> bool {
        true
    }
    fn exercised(&self, _field: &FieldInfo) -> (bool, bool) {
        (false, true)
    }
    fn shows_use(&self, _field: &FieldInfo, _seen: &Observed) -> bool {
        true
    }
    fn run(&self, varied: &bool, _store: Option<&Path>) -> Result<Observed> {
        Ok(if *varied { self.varied.clone() } else { self.base.clone() })
    }
}

fn result(kind: &str, byte: u8, content: bool) -> ObservedResult {
    ObservedResult { kind: kind.into(), run_id: Key::from_bytes([byte; 32]), files: BTreeMap::new(), upstream: None, content }
}

fn identity_field() -> FieldInfo {
    FieldInfo { name: "data", class: Class::Identity, vary: None, exercised_by: None }
}

/// An identity input that changes only a content artifact the run stored (a
/// second data file) and leaves the run's own id alone was dropped from the
/// identity: the check must say so, though the set of every stored id
/// differs.
#[test]
fn an_identity_input_that_moves_only_a_content_artifact_fails() {
    let h = Fixed {
        base: Observed { results: vec![result("run", 1, false), result("table", 2, true)], exports: BTreeMap::new() },
        varied: Observed { results: vec![result("run", 1, false), result("table", 3, true)], exports: BTreeMap::new() },
    };
    let failures = check_input_classes(&[identity_field()], &h).expect_err("the run's id did not move");
    assert_eq!(failures[0].field, "data");
}

/// The same input moving the run's own id keeps its promise.
#[test]
fn an_identity_input_that_moves_the_run_passes() {
    let h = Fixed {
        base: Observed { results: vec![result("run", 1, false), result("table", 2, true)], exports: BTreeMap::new() },
        varied: Observed { results: vec![result("run", 4, false), result("table", 3, true)], exports: BTreeMap::new() },
    };
    assert!(check_input_classes(&[identity_field()], &h).is_ok());
}

fn expands_field() -> FieldInfo {
    FieldInfo { name: "points", class: Class::Expands, vary: None, exercised_by: None }
}

/// An `expands` input the command ignores keeps every member's id in the
/// larger set, because the larger set is the smaller: the check must say so.
#[test]
fn an_ignored_expands_input_fails() {
    let same = Observed { results: vec![result("run", 1, false), result("table", 2, true)], exports: BTreeMap::new() };
    let h = Fixed { base: same.clone(), varied: same };
    let failures = check_input_classes(&[expands_field()], &h).expect_err("the larger set added nothing");
    assert_eq!(failures[0].field, "points");
}

/// A larger set that keeps the smaller set's ids and adds one keeps its
/// promise.
#[test]
fn an_expands_input_that_adds_a_member_passes() {
    let h = Fixed {
        base: Observed { results: vec![result("run", 1, false)], exports: BTreeMap::new() },
        varied: Observed { results: vec![result("run", 1, false), result("run", 5, false)], exports: BTreeMap::new() },
    };
    assert!(check_input_classes(&[expands_field()], &h).is_ok());
}
