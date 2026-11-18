include!("stub.rs");
use classify_derive::Classify;

#[derive(Classify)]
struct Surface {
    #[class(keyed(engine))]
    parallel: usize,
}

fn main() {}
