include!("stub.rs");
use classify_derive::Classify;

#[derive(Classify)]
struct Surface {
    #[class(semantic)]
    parallel: usize,
}

fn main() {}
