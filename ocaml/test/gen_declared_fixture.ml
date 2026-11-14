(* Writes the cross-language declared-model fixture to stdout. Run through
   `make update-declared-fixture`, which redirects it to
   `rust/crates/declared/tests/fixtures/all_variants.json`. *)

let () = print_string Declared_fixture.fixture_text
