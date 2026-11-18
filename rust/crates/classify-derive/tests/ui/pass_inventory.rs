include!("stub.rs");
use classify_derive::Classify;
use input_class::Classify as _;

#[derive(Classify)]
struct Group {
    #[class(keyed(config))]
    dt: f64,
}

#[derive(Classify)]
struct Surface {
    #[class(keyed(model))]
    model: String,
    #[class(execution)]
    parallel: usize,
    #[command(flatten)]
    group: Group,
    #[class(expands)]
    seeds: Vec<u64>,
    #[class(removed)]
    _removed_cas: bool,
}

fn main() {
    let inv = Surface::inventory();
    let names: Vec<&str> = inv.iter().map(|i| i.field).collect();
    assert_eq!(names, ["model", "parallel", "dt", "seeds", "_removed_cas"]);
    assert_eq!(inv[0].class, input_class::InputClass::Keyed(input_class::Level::Model));
    assert_eq!(inv[2].class, input_class::InputClass::Keyed(input_class::Level::Config));
    assert_eq!(inv[3].class, input_class::InputClass::Expands);
    let s = Surface { model: String::new(), parallel: 1, group: Group { dt: 1.0 }, seeds: vec![], _removed_cas: false };
    let _ = (s.model, s.parallel, s.group.dt, s.seeds, s._removed_cas);
}
