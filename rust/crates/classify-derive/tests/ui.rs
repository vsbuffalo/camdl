//! Compile-time contract of `#[derive(Classify)]`: an unclassified, doubly
//! classified or mis-classified input does not compile, and a flattened group
//! takes its classes from its own declaration (and must derive `Classify`).

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass_*.rs");
    t.compile_fail("tests/ui/fail_*.rs");
}
