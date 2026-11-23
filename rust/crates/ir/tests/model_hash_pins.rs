//! The structural model hash of four compiled golden models, pinned to its
//! hex, so a change to the encoding fails here rather than passing unseen.
//!
//! Source: the committed `ocaml/golden/*.ir.json` files named below
//! (primary). A pin that moves means every stored id built from a model hash,
//! and every `model_hash` a PGAS trajectory header prints, moved with it.

use std::path::Path;

use ir::model_hash::model_ir_hash;

/// `(golden file, model_ir_hash)`.
const PINS: &[(&str, &str)] = &[
    (
        "sir_basic.ir.json",
        "b3625fc83ec89908b6072a5094f76ce05e5ff97b579dbdcf8eb50737ce25ba3d",
    ),
    (
        "seir_observations.ir.json",
        "705f21e3a3cbbcbb507295599bcc6103c78db56c02d8d2b801435d30829b2096",
    ),
    (
        "sir_two_patch_long_obs.ir.json",
        "756309b6211ee01f63c71254ea9a249c7a8f3c31e3d6afb3cf715997bb0f300e",
    ),
    (
        "init_dependency_order.ir.json",
        "851b61f02bf6e0664d47c28b7fa19b73193ee24778a49b9357e1c4d93fa36fe6",
    ),
];

#[test]
fn the_model_hash_of_four_goldens_is_pinned() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden");
    for (file, plain) in PINS {
        let path = golden.join(file);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let model = ir::from_str(&text).unwrap_or_else(|e| panic!("parse {file}: {e}"));
        assert_eq!(model_ir_hash(&model).to_hex(), *plain, "{file}: model_ir_hash");
    }
}
