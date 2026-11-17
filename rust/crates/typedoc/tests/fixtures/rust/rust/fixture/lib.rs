//! The fixture crate. Its main type is [`Model`].

pub mod model;

pub use model::Model;

/// A module written inline.
mod inner {
    /// Private, and documented anyway.
    pub struct Hidden(pub u32);
}

/// A step of a model.
pub trait Step: other::Area {
    /// Advances by `dt`.
    fn step(&mut self, dt: f64);

    /// Advances twice, by default.
    fn twice(&mut self, dt: f64) {
        <Self as other::Area>::area(self);
        self.step(dt);
        self.step(dt);
    }
}

/// Counts, by name.
pub type Counts = std::collections::BTreeMap<String, u64>;
