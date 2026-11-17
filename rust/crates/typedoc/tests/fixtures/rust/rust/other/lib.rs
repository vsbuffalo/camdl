//! A second crate, so that links across crates are tested.

/// Shapes.
pub mod shapes {
    /// A shape.
    pub enum Shape {
        /// A circle, by its radius.
        Circle(f64),
        /// A square.
        Square { side: f64 },
        Point,
    }
}

/// Something with an area.
pub trait Area {
    /// The area.
    fn area(&self) -> f64;
}
