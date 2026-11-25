use super::*;
use store::{Content, Identity};

fn envelope(model: ir::Model) -> ir::IrEnvelope {
    ir::IrEnvelope::wrap(model, None, declared::DeclaredSection::not_recorded())
}

fn parts(model: ir::Model) -> CompiledParts {
    CompiledParts::split(envelope(model), "sir".into(), ModelSource { bytes: b"source", path: Path::new("sir.camdl"), rendered: Rendered::nothing() })
        .expect("a tiny model splits")
}

fn model_digest(model: ir::Model) -> Key {
    parts(model).model.digest()
}

/// The model part is keyed by its bytes, which do not depend on the order a
/// hash map was filled in: two models whose `ic_grad` maps hold the same
/// entries, inserted in opposite orders, have one digest.
#[test]
fn the_model_part_is_canonical_over_map_order() {
    let grad = |names: &[&str]| -> ir::Model {
        let mut m = crate::test_support::tiny_model();
        for n in names {
            m.ic_grad.insert(n.to_string(), Default::default());
        }
        m
    };
    let names: Vec<String> = (0..32).map(|i| format!("c{i}")).collect();
    let forward: Vec<&str> = names.iter().map(String::as_str).collect();
    let backward: Vec<&str> = forward.iter().rev().copied().collect();
    assert_eq!(model_digest(grad(&forward)), model_digest(grad(&backward)));
}

/// The bytes a model part is stored as read back to the same bytes, so a
/// result that loads the stored model reads what its key names.
#[test]
fn the_model_part_reads_back_to_its_own_bytes() {
    let p = parts(crate::test_support::tiny_model());
    let bytes = p.model.get().to_bytes();
    let back = CompiledModelContent::from_bytes(&bytes).expect("the stored bytes read");
    assert_eq!(back.to_bytes(), bytes);
    assert_eq!(back.model().simulation.t_end, 100.0);
}

/// The initial conditions read back in the order they were declared, which
/// the runtime evaluates them by and draws initial laws in: a model whose
/// compartments are declared out of alphabetical order is the same model
/// after it is stored, by its order and by its structural hash.
#[test]
fn the_model_part_keeps_the_initial_conditions_order() {
    let mut m = crate::test_support::tiny_model();
    m.initial_conditions = ir::model::InitialConditions::constants([
        ("S".to_string(), 990.0),
        ("I".to_string(), 10.0),
        ("R".to_string(), 0.0),
    ]);
    let p = parts(m.clone());
    let back = CompiledModelContent::from_bytes(&p.model.get().to_bytes()).expect("the stored bytes read");
    let names: Vec<&str> = back.model().initial_conditions.0.keys().map(String::as_str).collect();
    assert_eq!(names, ["S", "I", "R"]);
    let mut stored = back.model().clone();
    stored.name = m.name.clone();
    stored.description = m.description.clone();
    assert_eq!(ir::model_hash::model_ir_hash(&stored), ir::model_hash::model_ir_hash(&m));
    let mut swapped = m.clone();
    swapped.initial_conditions.0.swap_indices(0, 1);
    assert_ne!(model_digest(swapped), p.model.digest(), "the order is part of the model");
}

/// The name, the description, the quantities and the documentation are not
/// the model part: none of them moves its digest. A change to what the
/// engine runs does.
#[test]
fn only_what_the_engine_runs_keys_the_model_part() {
    let base = model_digest(crate::test_support::tiny_model());
    let mut renamed = crate::test_support::tiny_model();
    renamed.name = "other".into();
    renamed.description = Some("a description".into());
    assert_eq!(model_digest(renamed), base, "the name and description are labels and provenance");

    let mut documented = envelope(crate::test_support::tiny_model());
    documented.docs = serde_json::from_value(serde_json::json!({ "model": { "text": "a docstring" } }))
        .expect("a model doc reads");
    assert!(!documented.docs.is_empty());
    let p = CompiledParts::split(documented, "sir".into(), ModelSource { bytes: b"other source", path: Path::new("x.camdl"), rendered: Rendered::nothing() })
        .unwrap();
    assert_eq!(p.model.digest(), base, "the documentation and the source file are provenance");

    let mut longer = crate::test_support::tiny_model();
    longer.simulation.t_end = 200.0;
    assert_ne!(model_digest(longer), base, "the simulation window is the model");
}

/// An environment switch is part of the ambient identity, and so of every
/// run id; the default environment is one identity.
#[test]
fn an_engine_switch_changes_the_ambient_key() {
    let env = |eval_unresolved: bool| Engine {
        build: "b".into(),
        env: CamdlEnvIdentity { eval_unresolved, no_binding_cache: false, experimental: BTreeMap::new() },
    };
    assert_ne!(env(false).key(), env(true).key());
    assert_eq!(env(false).key(), env(false).key());
}

/// `--store` wins over `CAMDL_STORE`, which wins over `./camdl-store`; the old
/// root is never the default.
#[test]
fn the_root_is_the_flag_then_the_environment_then_the_default() {
    let env = StoreRequest::new(None, false, Some("from-env".into())).unwrap();
    let flag = StoreRequest::new(None, false, Some("from-flag".into())).unwrap();
    assert_eq!(root_of(&flag.clone().merge(env.clone())), PathBuf::from("from-flag"));
    assert_eq!(root_of(&StoreRequest::new(None, false, None).unwrap().merge(env)), PathBuf::from("from-env"));
    assert_eq!(root_of(&StoreRequest::new(None, false, None).unwrap()), PathBuf::from("camdl-store"));
}

/// The root is `--store`, then the fit file's `store` key, then
/// `CAMDL_STORE`, then `./camdl-store`.
#[test]
fn a_fit_files_store_key_comes_between_the_flag_and_the_environment() {
    let p = |s: &str| Some(PathBuf::from(s));
    let root = |flag, key, env| root_of(&request_from(None, false, flag, key, env).unwrap());
    assert_eq!(root(p("flag"), p("key"), p("env")), PathBuf::from("flag"));
    assert_eq!(root(None, p("key"), p("env")), PathBuf::from("key"));
    assert_eq!(root(None, None, p("env")), PathBuf::from("env"));
    assert_eq!(root(None, None, None), PathBuf::from("camdl-store"));
}

/// Every parameter value is finite before it is keyed.
#[test]
fn a_non_finite_parameter_is_refused() {
    assert!(ResolvedParams::new([("beta".to_string(), f64::NAN)]).is_err());
    assert!(ResolvedParams::new([("beta".to_string(), f64::INFINITY)]).is_err());
    assert!(ResolvedParams::new([("beta".to_string(), 0.3), ("beta".to_string(), 0.4)]).is_err());
    let a = ResolvedParams::new([("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)]).unwrap();
    let b = ResolvedParams::new([("gamma".to_string(), 0.1), ("beta".to_string(), 0.3)]).unwrap();
    assert_eq!(a.key(), b.key(), "the parameters are a map, not a list");
}

/// A model is accepted as the stored compile only when its model part has the
/// stored digest; another model is refused, naming the digest it lacks.
#[test]
fn only_the_stored_compile_is_accepted_as_the_model_a_result_runs() {
    let model = crate::test_support::tiny_model();
    let stored = model_digest(model.clone());
    assert!(require_stored_model(&model, stored, "fit", "m.ir.json").is_ok());
    let mut other = model;
    other.simulation.dt = Some(other.simulation.dt.unwrap_or(1.0) * 2.0);
    let err = require_stored_model(&other, stored, "fit", "m.ir.json").unwrap_err();
    assert!(err.contains("is not the stored compiled model") && err.contains(&stored.short12()), "{err}");
}
