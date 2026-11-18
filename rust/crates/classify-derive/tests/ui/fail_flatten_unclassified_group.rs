include!("stub.rs");
use classify_derive::Classify;

// A flattened group that does not derive `Classify`: its inputs would be
// unaccounted for, so the surface that flattens it does not compile.
struct Group {
    dt: f64,
}

#[derive(Classify)]
struct Surface {
    #[command(flatten)]
    group: Group,
}

fn main() {}
