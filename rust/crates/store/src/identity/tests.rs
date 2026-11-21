//! The derives: the identity encoding, record forms, and the input split.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::tests::*;
use crate::{CanonicalHasher, Class, ContentAddressed, EnvInputs, FileInputs, InputFile, Inputs, Key, HASH_VERSION};

#[derive(Identity, Clone, Debug)]
#[identity(tag = "pair", version = 2)]
struct PairAB {
    a: u64,
    b: String,
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "pair", version = 2)]
struct PairBA {
    b: String,
    a: u64,
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "pair", version = 2)]
struct PairRenamed {
    a: u64,
    c: String,
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "pair", version = 3)]
struct PairV3 {
    a: u64,
    b: String,
}

#[test]
fn a_struct_writes_its_tag_version_and_each_field_by_name_in_name_order() {
    let v = PairBA { b: "x".into(), a: 7 };
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_str("pair");
    h.write_u16(2);
    h.write_str("a");
    h.write_u64(7);
    h.write_str("b");
    h.write_str("x");
    assert_eq!(v.content_hash(), h.finalize());
    assert_eq!(v.key(), PairAB { a: 7, b: "x".into() }.key(), "field order does not matter");
    assert_ne!(v.key(), PairRenamed { a: 7, c: "x".into() }.key(), "a field's name does");
    assert_ne!(v.key(), PairV3 { a: 7, b: "x".into() }.key(), "the version does");
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "method", version = 1)]
enum MethodXY {
    Mh { steps: u64 },
    Fixed(u32, bool),
    Off,
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "method", version = 1)]
enum MethodYX {
    Off,
    Fixed(u32, bool),
    Mh { steps: u64 },
}

#[test]
fn an_enum_writes_its_variant_by_name() {
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_str("method");
    h.write_u16(1);
    h.write_str("Fixed");
    h.write_u32(3);
    h.write_bool(true);
    assert_eq!(MethodXY::Fixed(3, true).key(), h.finalize());
    assert_eq!(MethodXY::Mh { steps: 9 }.key(), MethodYX::Mh { steps: 9 }.key(), "variant order does not matter");
    assert_eq!(MethodXY::Off.key(), MethodYX::Off.key());
    assert_eq!(MethodXY::Fixed(1, true).key(), MethodYX::Fixed(1, true).key());
    assert_ne!(MethodXY::Off.key(), MethodXY::Fixed(0, false).key());
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "problem", version = 1)]
struct Problem {
    rows: Hashed<ModelText>,
    method: MethodXY,
    seeds: Vec<u64>,
    by_name: BTreeMap<String, Finite>,
    maybe: Option<Finite>,
}

#[test]
fn a_record_has_the_key_of_its_identity_and_lists_held_content() {
    let rows = Hashed::new(ModelText("a\n1\n".into()));
    let p = Problem {
        rows: rows.clone(),
        method: MethodXY::Mh { steps: 3 },
        seeds: vec![1, 2],
        by_name: [("beta".to_string(), Finite::new(0.25).unwrap())].into(),
        maybe: None,
    };
    let rec = p.record();
    assert_eq!(rec.rows, rows.digest());
    assert_eq!(rec.rows, Key::digest_bytes(b"a\n1\n"), "a content's digest is the SHA-256 of its bytes");
    assert_eq!(rec.content_hash(), p.key());
    assert_eq!(p.held_content(), vec![rows.digest()]);
    assert!(rec.held_content().is_empty());
    let json = serde_json::to_value(&rec).unwrap();
    let back: ProblemRecord = serde_json::from_value(json).unwrap();
    assert_eq!(back.content_hash(), p.key(), "a record read back hashes as the identity");
}

#[test]
fn a_float_is_finite_and_its_zeros_are_one_key() {
    assert!(Finite::new(f64::NAN).is_err());
    assert!(Finite::new(f64::INFINITY).is_err());
    assert_eq!(Finite::new(-0.0).unwrap().content_hash(), Finite::new(0.0).unwrap().content_hash());
}

#[test]
fn a_run_id_hashes_the_version_the_kind_the_ambient_key_and_the_identity_key() {
    let ambient = Engine { build: "b".into() }.key();
    let (id, _) = sim(1);
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_str("sim");
    h.write_key(&ambient);
    h.write_key(&id.key());
    assert_eq!(crate::Id::<Sim>::of(&ambient, &id).key(), h.finalize());
    let other = Engine { build: "c".into() }.key();
    assert_ne!(crate::Id::<Sim>::of(&ambient, &id), crate::Id::<Sim>::of(&other, &id), "a new build reuses nothing");
}

mod fixtures {
    use super::*;
    pub fn other_model() -> InputFile<ModelText> {
        InputFile::new("other.camdl")
    }
    pub fn other_rows() -> InputFile<ModelText> {
        InputFile::new("other_rows.csv")
    }
    pub fn two_threads() -> (ExampleArgs, ExampleArgs) {
        let mut a = base_args();
        a.parallel = 1;
        let mut b = base_args();
        b.parallel = 4;
        (a, b)
    }
}

#[derive(clap::Args, Inputs, Clone, Debug)]
pub(crate) struct NumericsArgs {
    #[arg(long)]
    #[input(identity)]
    pub dt: f64,
    #[arg(long)]
    #[input(identity)]
    pub binomial: bool,
}

#[derive(clap::Parser, Inputs, Clone, Debug)]
pub(crate) struct ExampleArgs {
    #[command(flatten)]
    #[input(group)]
    pub numerics: NumericsArgs,
    #[input(identity, vary = fixtures::other_model)]
    pub model: InputFile<ModelText>,
    #[arg(long)]
    #[input(expands)]
    pub seeds: Vec<u64>,
    #[arg(long)]
    #[input(execution, exercised_by = fixtures::two_threads)]
    pub parallel: usize,
    #[arg(long)]
    #[input(export)]
    pub output: Option<String>,
    #[arg(long)]
    #[input(render)]
    pub dates: bool,
    #[arg(long)]
    #[input(store)]
    pub label: Option<String>,
    #[arg(long)]
    #[input(recorded)]
    pub note: Option<String>,
    #[arg(long)]
    #[input(skip)]
    pub quiet: bool,
}

fn base_args() -> ExampleArgs {
    <ExampleArgs as clap::Parser>::parse_from(["example", "model.camdl", "--dt", "0.5", "--seeds", "1", "--seeds", "2", "--parallel", "2", "--label", "base", "--note", "first try", "--quiet"])
}

#[test]
fn the_split_takes_every_field_apart_by_class() {
    let (declared, execution, exports, request, given) = base_args().split().into_parts();
    let (numerics, model, seeds) = declared.into_parts();
    let (dt, binomial) = numerics.into_parts();
    assert_eq!((dt, binomial), (0.5, false));
    assert_eq!(model.typed(), std::path::Path::new("model.camdl"));
    assert_eq!(seeds, vec![1, 2]);
    assert_eq!(execution.parallel, 2);
    let _: () = execution.numerics;
    let ((), output, dates) = exports.into_parts();
    assert_eq!((output, dates), (None, false));
    let ((), label) = request.into_parts();
    assert_eq!(label.as_deref(), Some("base"));
    let f = given.fields();
    assert_eq!(f["numerics"], serde_json::json!({"dt": 0.5, "binomial": false}));
    assert_eq!(f["model"], "model.camdl");
    assert_eq!(f["note"], "first try");
    assert!(!f.contains_key("label"), "store fields are not in the given");
    assert!(!f.contains_key("quiet"), "skip fields are recorded nowhere");
}

#[test]
fn the_field_list_names_group_fields_and_varied_gives_a_second_value() {
    let names: Vec<(&str, Class)> = ExampleArgs::fields().iter().map(|f| (f.name, f.class)).collect();
    assert!(names.contains(&("numerics.dt", Class::Identity)));
    assert!(names.contains(&("parallel", Class::Execution)));
    assert!(names.contains(&("quiet", Class::Skip)));
    let p = ExampleArgs::fields().iter().find(|f| f.name == "parallel").unwrap();
    assert_eq!(p.exercised_by, Some("fixtures::two_threads"));
    let base = base_args();
    assert_eq!(base.varied("numerics.dt").numerics.dt, 1.0);
    assert_eq!(base.varied("model").model.typed(), std::path::Path::new("other.camdl"));
    assert!(base.varied("dates").dates);
    let (a, b) = ExampleArgs::exercised("parallel");
    assert_ne!(a.parallel, b.parallel);
}

#[derive(serde::Serialize, serde::Deserialize, FileInputs, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct DataTable {
    #[input(identity, vary = fixtures::other_rows)]
    pub path: InputFile<ModelText>,
    #[input(identity)]
    #[serde(default)]
    pub holdout_after: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize, FileInputs, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExampleToml {
    #[input(identity)]
    pub values: BTreeMap<String, f64>,
    #[input(expands)]
    #[serde(default)]
    pub fit_seeds: Vec<u64>,
    #[input(group)]
    #[serde(default)]
    pub data: Option<DataTable>,
    #[input(recorded)]
    #[serde(default, rename = "comment-text")]
    pub comment: Option<String>,
}

#[test]
fn a_settings_file_is_parsed_split_and_reads_its_paths_relative_to_itself() {
    let root = TempRoot::new("toml");
    let dir = root.path().join("configs");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("rows.csv"), "x\n1\n").unwrap();
    let toml = "values = { beta = 0.5 }\nfit_seeds = [1, 2]\ncomment-text = \"why\"\n\n[data]\npath = \"rows.csv\"\n";
    std::fs::write(dir.join("fit.toml"), toml).unwrap();
    let parsed = InputFile::<ExampleToml>::new(dir.join("fit.toml")).parse().unwrap();
    let (declared, _execution, _request, given, source) = parsed.into_parts();
    let (values, seeds, data) = declared.into_parts();
    assert_eq!(values["beta"], 0.5);
    assert_eq!(seeds, vec![1, 2]);
    let (path, holdout) = data.expect("[data] present").into_parts();
    assert_eq!(holdout, None);
    assert_eq!(path.typed(), std::path::Path::new("rows.csv"), "recorded as typed");
    assert_eq!(path.content().unwrap().get().0, "x\n1\n", "read relative to the file's directory");
    assert_eq!(given.fields()["comment-text"], "why", "a settings key is named as written");
    assert_eq!(source.digest(), Key::digest_bytes(toml.as_bytes()));
    let names: Vec<&str> = ExampleToml::fields().iter().map(|f| f.name).collect();
    assert!(names.contains(&"data.path"));
    let with = crate::Given::default().with_file("fit", given, &source);
    assert_eq!(with.fields()["fit"]["sha256"], source.digest().to_hex());
    assert_eq!(with.fields()["fit"]["keys"]["data"]["path"], "rows.csv");

    std::fs::write(dir.join("bad.toml"), "values = {}\nunknown = 1\n").unwrap();
    assert!(matches!(InputFile::<ExampleToml>::new(dir.join("bad.toml")).parse(), Err(crate::Error::Parse { .. })));
    std::fs::write(dir.join("bare.toml"), "values = {}\n").unwrap();
    let (d, ..) = InputFile::<ExampleToml>::new(dir.join("bare.toml")).parse().unwrap().into_parts();
    assert!(d.into_parts().2.is_none(), "an absent table stays distinct");
}

fn first_line(v: Option<String>) -> Result<bool, String> {
    Ok(v.as_deref() == Some("1"))
}

#[derive(EnvInputs)]
pub(crate) struct TestEnv {
    #[env("STORE_TEST_ENV_FLAG")]
    #[input(identity)]
    pub flag: bool,
    #[env(prefix = "STORE_TEST_ENV_EXP_")]
    #[input(identity)]
    pub experimental: BTreeMap<String, String>,
    #[env("STORE_TEST_ENV_ONE", parse = first_line)]
    #[input(identity)]
    pub one: bool,
    #[env("STORE_TEST_ENV_DIR")]
    #[input(store)]
    pub dir: Option<PathBuf>,
}

#[test]
fn environment_variables_are_read_by_their_field_types() {
    // The variables are this test's own; no other test reads them.
    std::env::set_var("STORE_TEST_ENV_FLAG", "0");
    std::env::set_var("STORE_TEST_ENV_EXP_A", "x");
    std::env::set_var("STORE_TEST_ENV_ONE", "2");
    std::env::remove_var("STORE_TEST_ENV_DIR");
    let (identity, (), request, given) = TestEnv::read().unwrap();
    assert!(identity.flag, "a bool variable is true when set, whatever its value");
    assert_eq!(identity.experimental.get("STORE_TEST_ENV_EXP_A").map(String::as_str), Some("x"));
    assert!(!identity.one, "a field's own parser decides");
    assert_eq!(request.into_parts(), (None,));
    assert_eq!(given.fields()["STORE_TEST_ENV_FLAG"], true);
    assert!(given.argv().is_empty());
    let key = identity.key();
    std::env::remove_var("STORE_TEST_ENV_FLAG");
    assert_ne!(TestEnv::read().unwrap().0.key(), key);
}
