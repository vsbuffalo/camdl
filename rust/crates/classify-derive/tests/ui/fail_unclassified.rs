include!("stub.rs");
use classify_derive::Classify;

#[derive(Classify)]
struct Surface {
    #[class(keyed(model))]
    model: String,
    parallel: usize,
}

fn main() {}
