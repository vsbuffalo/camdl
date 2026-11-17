//! Publishes `sim`'s resolved Cargo features to `cli`'s build script, which
//! folds them into the engine's build fingerprint (`ENGINE_ID`).
//!
//! A build script sees only its own package's features, and `sim`'s are not
//! a function of `cli`'s (`io` depends on `sim` with its default features), so
//! `cli` cannot derive them. `links` metadata is how a dependency hands a value
//! to its direct dependents' build scripts: this emits `features`, which `cli`
//! reads as `DEP_CAMDL_SIM_FEATURES`.

fn main() {
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(str::to_lowercase))
        .collect();
    features.sort();
    println!("cargo:features={}", features.join(","));
    println!("cargo:rerun-if-changed=build.rs");
}
