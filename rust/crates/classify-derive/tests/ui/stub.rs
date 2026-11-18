// The types the derive's output names, as the `cli` crate defines them.
mod input_class {
    #[derive(Debug, PartialEq)]
    pub enum Level {
        Model,
        Config,
        Params,
        Scenario,
        Seed,
        Obs,
    }
    #[derive(Debug, PartialEq)]
    pub enum InputClass {
        Keyed(Level),
        Execution,
        Provenance,
        Presentation,
        Expands,
        Removed,
    }
    #[derive(Debug, PartialEq)]
    pub struct Input {
        pub field: &'static str,
        pub class: InputClass,
    }
    pub trait Classify {
        fn inventory() -> Vec<Input>;
    }
}
