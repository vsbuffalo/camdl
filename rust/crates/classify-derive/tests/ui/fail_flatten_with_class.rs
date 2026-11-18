include!("stub.rs");
use classify_derive::Classify;

#[derive(Classify)]
struct Group {
    #[class(keyed(config))]
    dt: f64,
}

#[derive(Classify)]
struct Surface {
    #[class(keyed(config))]
    #[command(flatten)]
    group: Group,
}

fn main() {}
