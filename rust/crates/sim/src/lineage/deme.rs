//! Per-compartment deme (stratum / patch) assignment.
//!
//! Phase 2 of the individual-sampling layer (2026-05-19 proposal,
//! §"Mathematical structure" — stratified case) makes [`super::DemeId`] real.
//! A compartment in stratum `b` lives in pool `(b, comp)`; when a `#[lineage]`
//! event in stratum `a` fires, the parent is attributed to whichever
//! stratum-specific pool `(b, I_b)` the per-class weights select, and the line
//! list records `parent_deme = b`, `child_deme = a`.
//!
//! ## How a compartment's deme is derived
//!
//! The IR that reaches Rust is *fully expanded*: stratification leaves no
//! shorthand, only flat compartments named `<base>_<v1>_<v2>…` (the OCaml
//! expander joins the base name and its dimension values with `_`, in
//! cartesian-product order — see `ocaml/lib/compiler/expander.ml`). The
//! model carries `model_structure.dimensions` (ordered value lists) and
//! `model_structure.compartment_dims` (base name → the dimension names it is
//! stratified by).
//!
//! We do **not** parse compartment names by splitting on `_` (a dimension
//! value may itself contain `_`). Instead we *reconstruct* the expanded name
//! for every `(base, dimension-value-tuple)` exactly as the expander does, and
//! build a map from expanded name to a stratum index. The stratum index — the
//! `DemeId` — is the position of the compartment's dimension-value tuple in
//! the cartesian product of *all* model dimensions, projected onto the
//! dimensions that compartment is stratified by. Compartments sharing a
//! stratum (`S_a`, `I_a`, `R_a`) share a deme; `S_a` and `S_b` differ.
//!
//! Unstratified compartments — and every compartment in a model with no
//! `model_structure` or no dimensions — map to deme 0. That makes the
//! single-population slice the `DemeId = 0` special case exactly, so the
//! Phase-1 behaviour (and Tier-2a byte identity) is preserved: the deme is
//! pure line-list metadata, never consulted by the count dynamics.

use std::collections::HashMap;

use ir::model::ModelStructure;
use ir::Model;

use super::{CompartmentId, DemeId};

/// Maps every global compartment id to its deme (stratum index).
///
/// Built once per run from the model structure. `deme_of` returns 0 for any
/// compartment not explicitly stratified, which is the correct single-
/// population / unstratified answer.
pub struct DemeMap {
    /// Indexed by global compartment id.
    by_comp: Vec<DemeId>,
}

impl DemeMap {
    /// Build the map from the model. `comp_index` maps compartment name to
    /// global id (matches `CompiledModel::comp_index`).
    pub fn build(model: &Model, comp_index: &HashMap<String, usize>) -> Self {
        let n = comp_index.len();
        let mut by_comp = vec![DemeId(0); n];

        let Some(ms) = &model.model_structure else {
            // No stratification: every compartment is deme 0.
            return DemeMap { by_comp };
        };
        if ms.dimensions.is_empty() {
            return DemeMap { by_comp };
        }

        // Ordered (dim name → ordered values) for fast lookup, and a stable
        // global dimension order taken from `ms.dimensions`.
        let dim_values: HashMap<&str, &[String]> = ms
            .dimensions
            .iter()
            .map(|d| (d.name.as_str(), d.values.as_slice()))
            .collect();

        // The global stratum enumeration is the cartesian product of *all*
        // dimensions, in `ms.dimensions` order. A compartment stratified by a
        // subset of dimensions is assigned the deme of the product cell whose
        // values match the compartment's, with the unreferenced dimensions
        // pinned to their first value. This gives a single consistent stratum
        // numbering shared across compartments that carry the same dimensions
        // (the spatial / age-structured case), which is what Phase 2 needs.
        //
        // We enumerate by reconstructing names rather than parsing them.
        for (base, dims) in &ms.compartment_dims {
            // Value lists for this compartment's dimensions, in declared order.
            let mut val_lists: Vec<&[String]> = Vec::with_capacity(dims.len());
            for dname in dims {
                match dim_values.get(dname.as_str()) {
                    Some(vs) => val_lists.push(vs),
                    // Unknown dimension on a compartment — leave its expansions
                    // at deme 0 rather than guess. (Validation upstream should
                    // already reject this; we stay defensive.)
                    None => {
                        val_lists.clear();
                        break;
                    }
                }
            }
            if val_lists.is_empty() && !dims.is_empty() {
                continue;
            }

            // Cartesian product of this compartment's dimension values, in the
            // same order the expander uses (`String.concat "_"`).
            for combo in cartesian(&val_lists) {
                let expanded = std::iter::once(base.clone())
                    .chain(combo.iter().cloned())
                    .collect::<Vec<_>>()
                    .join("_");
                if let Some(&gid) = comp_index.get(&expanded) {
                    by_comp[gid] = global_stratum_index(ms, dims, &combo);
                }
            }
        }

        DemeMap { by_comp }
    }

    /// The deme of a global compartment id. Defaults to 0 for unstratified
    /// compartments.
    pub fn deme_of(&self, comp: CompartmentId) -> DemeId {
        self.by_comp.get(comp.0).copied().unwrap_or(DemeId(0))
    }

    /// Number of distinct demes (max index + 1). Used by tests/diagnostics.
    pub fn n_demes(&self) -> usize {
        self.by_comp.iter().copied().max().map_or(1, |m| m.0 as usize + 1)
    }
}

/// Index of a compartment's stratum within the cartesian product of ALL model
/// dimensions (in `ms.dimensions` order). Dimensions the compartment is *not*
/// stratified by are pinned to value index 0; the referenced dimensions take
/// the value indices given by `combo`. This yields a consistent global stratum
/// numbering: compartments carrying the same dimension set with the same
/// values get the same deme.
fn global_stratum_index(
    ms: &ModelStructure,
    comp_dims: &[String],
    combo: &[String],
) -> DemeId {
    // Map each dimension name → its chosen value index (0 if unreferenced).
    let mut idx_by_dim: HashMap<&str, usize> = HashMap::new();
    for (dname, val) in comp_dims.iter().zip(combo.iter()) {
        if let Some(d) = ms.dimensions.iter().find(|d| &d.name == dname) {
            if let Some(pos) = d.values.iter().position(|v| v == val) {
                idx_by_dim.insert(dname.as_str(), pos);
            }
        }
    }
    // Mixed-radix encode in `ms.dimensions` order (first dimension is the most
    // significant digit). Radix of each digit is that dimension's cardinality.
    let mut deme: u64 = 0;
    for d in &ms.dimensions {
        let radix = d.values.len().max(1) as u64;
        let digit = idx_by_dim.get(d.name.as_str()).copied().unwrap_or(0) as u64;
        deme = deme * radix + digit;
    }
    DemeId(deme as u32)
}

/// Cartesian product of value lists, preserving order. Returns an empty
/// single-element product (`[[]]`) when there are no lists, matching the
/// expander's behaviour for unstratified bases.
fn cartesian(lists: &[&[String]]) -> Vec<Vec<String>> {
    let mut acc: Vec<Vec<String>> = vec![Vec::new()];
    for list in lists {
        let mut next = Vec::with_capacity(acc.len() * list.len());
        for prefix in &acc {
            for v in *list {
                let mut row = prefix.clone();
                row.push(v.clone());
                next.push(row);
            }
        }
        acc = next;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::model::{
        Compartment, CompartmentKind, Dimension, InitialConditions, ModelStructure, OutputConfig,
        OutputSchedule, SimulationConfig,
    };

    fn ms(dims: Vec<(&str, Vec<&str>)>, comp_dims: Vec<(&str, Vec<&str>)>) -> ModelStructure {
        ModelStructure {
            dimensions: dims
                .into_iter()
                .map(|(n, vs)| Dimension {
                    name: n.to_string(),
                    values: vs.into_iter().map(String::from).collect(),
                })
                .collect(),
            compartment_dims: comp_dims
                .into_iter()
                .map(|(n, ds)| (n.to_string(), ds.into_iter().map(String::from).collect()))
                .collect(),
            base_compartments: vec![],
            transmission_transitions: vec![],
            infectious_compartments: vec![],
        }
    }

    #[test]
    fn single_dimension_two_patches() {
        // patch = [a, b]; S/I/R stratified by patch.
        let structure = ms(
            vec![("patch", vec!["a", "b"])],
            vec![("S", vec!["patch"]), ("I", vec!["patch"]), ("R", vec!["patch"])],
        );
        let mut model = ir::Model {
            model_structure: Some(structure),
            ..minimal_model()
        };
        // Expanded names in cartesian order: S_a S_b I_a I_b R_a R_b.
        let names = ["S_a", "S_b", "I_a", "I_b", "R_a", "R_b"];
        let comp_index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect();
        model.compartments = names
            .iter()
            .map(|n| Compartment { name: n.to_string(), kind: CompartmentKind::Integer })
            .collect();

        let dm = DemeMap::build(&model, &comp_index);
        // Patch a → deme 0, patch b → deme 1, shared across S/I/R.
        assert_eq!(dm.deme_of(CompartmentId(comp_index["S_a"])), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_a"])), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["R_a"])), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["S_b"])), DemeId(1));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_b"])), DemeId(1));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["R_b"])), DemeId(1));
        assert_eq!(dm.n_demes(), 2);
    }

    #[test]
    fn unstratified_is_all_deme_zero() {
        let model = minimal_model();
        let comp_index: HashMap<String, usize> =
            [("S", 0), ("I", 1), ("R", 2)].iter().map(|(n, i)| (n.to_string(), *i)).collect();
        let dm = DemeMap::build(&model, &comp_index);
        assert_eq!(dm.deme_of(CompartmentId(0)), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(1)), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(2)), DemeId(0));
        assert_eq!(dm.n_demes(), 1);
    }

    #[test]
    fn two_dimensions_mixed_radix() {
        // age = [child, adult], patch = [a, b] → 4 demes, mixed-radix encoded.
        let structure = ms(
            vec![("age", vec!["child", "adult"]), ("patch", vec!["a", "b"])],
            vec![("I", vec!["age", "patch"])],
        );
        let mut model = ir::Model { model_structure: Some(structure), ..minimal_model() };
        let names = ["I_child_a", "I_child_b", "I_adult_a", "I_adult_b"];
        let comp_index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect();
        model.compartments = names
            .iter()
            .map(|n| Compartment { name: n.to_string(), kind: CompartmentKind::Integer })
            .collect();
        let dm = DemeMap::build(&model, &comp_index);
        // age is the most significant digit (radix 2), patch the least.
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_child_a"])), DemeId(0)); // 0*2 + 0
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_child_b"])), DemeId(1)); // 0*2 + 1
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_adult_a"])), DemeId(2)); // 1*2 + 0
        assert_eq!(dm.deme_of(CompartmentId(comp_index["I_adult_b"])), DemeId(3)); // 1*2 + 1
        assert_eq!(dm.n_demes(), 4);
    }

    fn minimal_model() -> ir::Model {
        ir::Model {
            ic_grad: Default::default(),
            name: "t".into(),
            version: "0".into(),
            time_unit: "days".into(),
            description: None,
            origin: None, origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![],
            initial_conditions: InitialConditions::Explicit(HashMap::new()),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 1.0,
                time_semantics: "continuous".into(),
                dt: None,
                rng_seed: None,
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }
}
