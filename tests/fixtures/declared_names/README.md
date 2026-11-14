# Declared-section name-kind fixtures

Three small models, each holding a name of a kind that no other compiled golden
has. The declared-model section of a compiled model
(`docs/dev/proposals/2026-11-13-model-structure-diff.md`) tags every name in an
expression with what it refers to; these models make three of those kinds appear
in compiler output, where before only a hand-written fixture reached them.

| Model                        | Kind          | Where the name is                                             |
| ---------------------------- | ------------- | ------------------------------------------------------------- |
| `obs_anchor.camdl`           | `obs_anchor`  | `last_obs` and `first_obs` in two `quantities`                |
| `aux_data_column.camdl`      | `data_column` | `tested`, the `n` of a binomial likelihood                    |
| `table_time_dimension.camdl` | `dimension`   | `week`, the `time_dim` of a table-backed interpolated forcing |

They are compile-only: nothing simulates them. `make update-golden`
(`update-declared-names-golden`) writes `ir/*.ir.json`, and
`make check-declared-names-golden`, which runs in `make test`, fails if a model
no longer compiles byte-for-byte to its committed file.

What reads them:

- `ocaml/test/test_declared_emit.ml` compiles each with every other golden and
  checks the section (every name has one kind; `reads` is the tree's names; the
  section agrees with `ModelStructure` and the flow graph), and asserts that
  each model holds its kind.
- `rust/crates/declared/tests/golden_sections.rs` reads each committed section
  through the Rust types.
- `rust/crates/runid/tests/golden_model_hashes.rs` pins each model's identity.

`data/` holds the two files `table_time_dimension.camdl` reads. Both are
synthetic, written for this fixture, with no external source: a contact
multiplier for two patches at three weeks, and the list of those weeks.
