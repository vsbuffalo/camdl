use super::*;
use crate::sim_store::{CellAttributes, PointAttributes};

fn cell(point: PointAttributes) -> CellAttributes {
    CellAttributes {
        model: "sir".into(),
        params: "beta=0.2".into(),
        scenario: "(none)".into(),
        replicate: 3,
        seed: 7,
        seed_replicate: 1,
        process_seed: 8,
        point,
    }
}

fn names(columns: &[(String, String)]) -> Vec<&str> {
    columns.iter().map(|(c, _)| c.as_str()).collect()
}

#[test]
fn a_cells_attributes_flatten_by_its_point() {
    let fixed = ["model", "params", "scenario", "replicate", "seed", "seed_replicate", "process_seed"];
    assert_eq!(names(&cell(PointAttributes::None).columns()), fixed);
    let draw = cell(PointAttributes::Draw { index: 4 }).columns();
    assert_eq!(names(&draw), [&fixed[..], &["draw"]].concat());
    assert_eq!(draw.last().unwrap().1, "4");
    let sweep = cell(PointAttributes::Sweep { point_id: 2, values: vec![("beta".into(), 0.25), ("gamma".into(), 0.1)] }).columns();
    assert_eq!(names(&sweep), [&fixed[..], &["point_id", "beta", "gamma"]].concat());
    let values: Vec<&str> = sweep.iter().map(|(_, v)| v.as_str()).collect();
    assert_eq!(values[7..], ["2", "0.25", "0.1"]);
}

#[test]
fn a_swept_parameter_named_like_an_attribute_is_prefixed() {
    let sweep = cell(PointAttributes::Sweep { point_id: 0, values: vec![("seed".into(), 3.0)] }).columns();
    assert_eq!(sweep.last().unwrap(), &("param.seed".to_string(), "3".to_string()));
}

#[test]
fn other_attributes_give_a_column_per_field() {
    let attributes = serde_json::json!({ "method": "mh", "seed": 7, "scenarios": ["a", "b"], "note": null });
    let cols = attribute_columns("fit_run", &attributes);
    let got: BTreeMap<&str, &str> = cols.iter().map(|(c, v)| (c.as_str(), v.as_str())).collect();
    assert_eq!(got, BTreeMap::from([("method", "mh"), ("seed", "7"), ("scenarios", r#"["a","b"]"#), ("note", "")]));
    assert!(attribute_columns("obs", &serde_json::Value::Null).is_empty(), "a by-product's null attributes give none");
}

#[test]
fn an_attribute_named_like_a_table_column_is_prefixed() {
    let keys = vec!["seed".to_string(), "path".to_string()];
    assert_eq!(attribute_header(keys, &["run_id", "path"]), ["seed", "attribute.path"]);
}
