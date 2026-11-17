//! The model and how it is made.

use other::shapes::Shape;

/// A model of one shape.
#[derive(Debug, Clone)]
pub struct Model {
    /// Its shape, from the other crate.
    pub shape: Shape,
    /// How many there are. See [`Shape`] and [`make`].
    pub n: u32, // a trailing comment
    hidden: crate::inner::Hidden,
}

impl other::Area for Model {
    fn area(&self) -> f64 {
        0.0
    }
}

impl crate::Step for Model {
    fn step(&mut self, _dt: f64) {}
}

/// Makes a model like `m`.
pub fn make(m: &Model) -> Model {
    let Model { n, .. } = m;
    let _ = n;
    m.clone()
}

impl Model {
    /// How many there are.
    pub fn count(&self) -> u32 {
        self.n
    }
}
