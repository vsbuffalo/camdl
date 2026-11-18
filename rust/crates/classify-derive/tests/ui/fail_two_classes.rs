include!("stub.rs");
use classify_derive::Classify;

#[derive(Classify)]
struct Surface {
    #[class(execution)]
    #[class(provenance)]
    parallel: usize,
}

fn main() {}
