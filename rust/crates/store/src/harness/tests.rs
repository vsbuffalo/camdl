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
        base: Observed { results: vec![result("run", 1, false), result("table", 2, true)], ..Observed::default() },
        varied: Observed { results: vec![result("run", 1, false), result("table", 3, true)], ..Observed::default() },
    };
    let failures = check_input_classes(&[identity_field()], &h).expect_err("the run's id did not move");
    assert_eq!(failures[0].field, "data");
}

/// The same input moving the run's own id keeps its promise.
#[test]
fn an_identity_input_that_moves_the_run_passes() {
    let h = Fixed {
        base: Observed { results: vec![result("run", 1, false), result("table", 2, true)], ..Observed::default() },
        varied: Observed { results: vec![result("run", 4, false), result("table", 3, true)], ..Observed::default() },
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
    let same = Observed { results: vec![result("run", 1, false), result("table", 2, true)], ..Observed::default() };
    let h = Fixed { base: same.clone(), varied: same };
    let failures = check_input_classes(&[expands_field()], &h).expect_err("the larger set added nothing");
    assert_eq!(failures[0].field, "points");
}

/// A larger set that keeps the smaller set's ids and adds one keeps its
/// promise.
#[test]
fn an_expands_input_that_adds_a_member_passes() {
    let h = Fixed {
        base: Observed { results: vec![result("run", 1, false)], ..Observed::default() },
        varied: Observed { results: vec![result("run", 1, false), result("run", 5, false)], ..Observed::default() },
    };
    assert!(check_input_classes(&[expands_field()], &h).is_ok());
}

/// A harness whose runs return the given observations in turn, for the
/// export check, which runs the same invocation twice into one store.
struct InTurn {
    runs: Vec<Observed>,
    next: std::cell::Cell<usize>,
}

impl InTurn {
    fn new(runs: Vec<Observed>) -> Self {
        InTurn { runs, next: std::cell::Cell::new(0) }
    }
}

impl InputHarness for InTurn {
    type Invocation = ();
    fn base(&self) {}
    fn varied(&self, _field: &FieldInfo) {}
    fn exercised(&self, _field: &FieldInfo) -> ((), ()) {
        ((), ())
    }
    fn shows_use(&self, _field: &FieldInfo, _seen: &Observed) -> bool {
        true
    }
    fn run(&self, _: &(), _store: Option<&Path>) -> Result<Observed> {
        let i = self.next.get();
        self.next.set(i + 1);
        Ok(self.runs[i].clone())
    }
}

fn export_field() -> FieldInfo {
    FieldInfo { name: "output", class: Class::Export, vary: None, exercised_by: None }
}

/// A run that stored `traj.tsv` with digest `stored`, and exported a file
/// with digest `exported`, as a copy of `traj.tsv` when `copy`.
fn exported(stored: u8, exported: u8, copy: bool) -> Observed {
    let mut run = result("run", 1, false);
    run.files.insert("traj.tsv".into(), Key::from_bytes([stored; 32]));
    let path = PathBuf::from("out.tsv");
    let mut o = Observed { results: vec![run], ..Observed::default() };
    o.exports.insert(path.clone(), Key::from_bytes([exported; 32]));
    if copy {
        o.copies.insert(path, StoredFile { run_id: Key::from_bytes([1; 32]), file: "traj.tsv".into() });
    }
    o
}

/// A copy whose bytes differ from the stored file it copies fails, though it
/// has the same bytes each time it is made.
#[test]
fn a_copy_that_differs_from_its_stored_file_fails() {
    let h = InTurn::new(vec![exported(7, 8, true), exported(7, 8, true)]);
    let failures = check_input_classes(&[export_field()], &h).expect_err("the copy is not the stored file");
    assert_eq!(failures[0].field, "output");
}

/// A copy with the stored file's digest keeps its promise.
#[test]
fn a_copy_of_its_stored_file_passes() {
    let h = InTurn::new(vec![exported(7, 7, true), exported(7, 7, true)]);
    assert!(check_input_classes(&[export_field()], &h).is_ok());
}

/// A copy that names a file the run did not store fails.
#[test]
fn a_copy_of_a_file_never_stored_fails() {
    let mut a = exported(7, 7, true);
    a.copies.values_mut().for_each(|f| f.file = "other.tsv".into());
    let h = InTurn::new(vec![a.clone(), a]);
    assert!(check_input_classes(&[export_field()], &h).is_err());
}

/// A rendering made twice from the same stored files with other bytes the
/// second time fails.
#[test]
fn a_rendering_that_differs_when_made_again_fails() {
    let h = InTurn::new(vec![exported(7, 8, false), exported(7, 9, false)]);
    assert!(check_input_classes(&[export_field()], &h).is_err());
}

/// A rendering made twice from the same stored files with the same bytes
/// keeps its promise.
#[test]
fn a_rendering_made_again_with_the_same_bytes_passes() {
    let h = InTurn::new(vec![exported(7, 8, false), exported(7, 8, false)]);
    assert!(check_input_classes(&[export_field()], &h).is_ok());
}

/// Two renderings with the same bytes, made from stored files that changed
/// between them, were not made from the same stored files: the check fails.
#[test]
fn a_rendering_over_stored_files_that_changed_fails() {
    let h = InTurn::new(vec![exported(7, 8, false), exported(6, 8, false)]);
    assert!(check_input_classes(&[export_field()], &h).is_err());
}
