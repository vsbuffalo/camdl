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
//! ## Two questions, one table
//!
//! The primitive is the **dimension-value assignment**: for each compartment,
//! which value it takes along each declared dimension, or "carries no value
//! here". The `DemeId` is a mixed-radix digest of that row. [`DemeMap`] stores
//! the rows and answers both questions from them:
//!
//! - [`DemeMap::deme_of`] — the digest, for the lineage pools.
//! - [`DemeMap::stratum_members`] — the compartments lying in the stratum a
//!   given compartment names, for the IVP Binomial denominator N₀ (gh#649).
//!
//! The two differ where a model is only *partially* stratified
//! (`stratify(by = latent_stage, only = [E])`): the digest pins a dimension a
//! compartment does not carry to value 0, which lumps unstratified `S` in with
//! `E_e1`. That is the right pooling for lineage attribution and the wrong one
//! for a population denominator, so `stratum_members` reads the row directly
//! instead of comparing digests.
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

/// A compartment carries no value along this dimension — it is not stratified
/// by it. Distinct from value index 0, which is a real value.
const NOT_STRATIFIED: u32 = u32::MAX;

/// Maps every global compartment id to its dimension-value assignment, and to
/// the deme (stratum index) that assignment encodes.
///
/// Built once per run from the model structure. `deme_of` returns 0 for any
/// compartment not explicitly stratified, which is the correct single-
/// population / unstratified answer.
pub struct DemeMap {
    /// Indexed by global compartment id.
    by_comp: Vec<DemeId>,
    /// Flattened `n_comps × n_dims` table: `assign[c * n_dims + d]` is the
    /// index of compartment `c`'s value within `ms.dimensions[d].values`, or
    /// [`NOT_STRATIFIED`] when `c` carries no value along dimension `d`.
    /// `by_comp` is the mixed-radix digest of this table.
    assign: Vec<u32>,
    /// Number of declared dimensions — the row width of `assign`. Zero when
    /// the model declares none, in which case `assign` is empty.
    n_dims: usize,
}

impl DemeMap {
    /// Build the map from the model. `comp_index` maps compartment name to
    /// global id (matches `CompiledModel::comp_index`).
    pub fn build(model: &Model, comp_index: &HashMap<String, usize>) -> Self {
        let n = comp_index.len();
        let unstratified = || DemeMap { by_comp: vec![DemeId(0); n], assign: Vec::new(), n_dims: 0 };

        let Some(ms) = &model.model_structure else {
            // No stratification: every compartment is deme 0.
            return unstratified();
        };
        if ms.dimensions.is_empty() {
            return unstratified();
        }

        let n_dims = ms.dimensions.len();
        let mut assign = vec![NOT_STRATIFIED; n * n_dims];

        // Ordered (dim name → ordered values) for fast lookup, and a stable
        // global dimension order taken from `ms.dimensions`.
        let dim_values: HashMap<&str, &[String]> = ms
            .dimensions
            .iter()
            .map(|d| (d.name.as_str(), d.values.as_slice()))
            .collect();
        // Dimension name → its column in `assign` (its position in
        // `ms.dimensions`, which is also its mixed-radix digit position).
        let dim_col: HashMap<&str, usize> =
            ms.dimensions.iter().enumerate().map(|(i, d)| (d.name.as_str(), i)).collect();

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
                let Some(&gid) = comp_index.get(&expanded) else { continue };
                for (dname, val) in dims.iter().zip(combo.iter()) {
                    let (Some(&col), Some(vals)) =
                        (dim_col.get(dname.as_str()), dim_values.get(dname.as_str()))
                    else {
                        continue;
                    };
                    if let Some(pos) = vals.iter().position(|v| v == val) {
                        assign[gid * n_dims + col] = pos as u32;
                    }
                }
            }
        }

        let by_comp = (0..n)
            .map(|c| encode_deme(ms, &assign[c * n_dims..(c + 1) * n_dims]))
            .collect();
        DemeMap { by_comp, assign, n_dims }
    }

    /// The deme of a global compartment id. Defaults to 0 for unstratified
    /// compartments.
    pub fn deme_of(&self, comp: CompartmentId) -> DemeId {
        self.by_comp.get(comp.0).copied().unwrap_or(DemeId(0))
    }

    /// The compartments lying in the stratum that `comp` names, `comp`
    /// included.
    ///
    /// A compartment names the stratum given by *its own* dimensions at its
    /// own values: `S_p1_child` names `(patch = p1, age = child)`. Another
    /// compartment lies in that stratum when it carries **every one of those
    /// dimensions with the same value** — `I_p1_child` does; `I_p2_child` and
    /// `I_p1_adult` do not. A compartment carrying no dimension names no
    /// constraint, so every compartment lies in its stratum: the whole model.
    /// That makes the unstratified model the one-stratum case exactly, and
    /// keeps the answer for the unstratified `S` of a partially stratified
    /// model (`stratify(by = latent_stage, only = [E])`) the whole population
    /// rather than whichever cells happen to share its deme digest.
    ///
    /// This is the grouping the IVP Binomial denominator N₀ needs (gh#649):
    /// `S₀ ~ Binom(N₀, s₀)` where N₀ is the population `S₀` is a fraction of.
    pub fn stratum_members(&self, comp: CompartmentId) -> impl Iterator<Item = CompartmentId> + '_ {
        let nd = self.n_dims;
        // An out-of-range compartment (or a model with no dimensions) yields an
        // empty constraint row, i.e. the whole model.
        let target: &[u32] = nd
            .checked_mul(comp.0)
            .and_then(|start| self.assign.get(start..start + nd))
            .unwrap_or(&[]);
        (0..self.by_comp.len())
            .filter(move |&j| {
                target.iter().enumerate().all(|(d, &t)| {
                    t == NOT_STRATIFIED || self.assign[j * nd + d] == t
                })
            })
            .map(CompartmentId)
    }

    /// Number of distinct demes (max index + 1). Used by tests/diagnostics.
    pub fn n_demes(&self) -> usize {
        self.by_comp.iter().copied().max().map_or(1, |m| m.0 as usize + 1)
    }
}

/// Index of a compartment's stratum within the cartesian product of ALL model
/// dimensions (in `ms.dimensions` order), from its `assign` row. Dimensions the
/// compartment is *not* stratified by are pinned to value index 0. This yields
/// a consistent global stratum numbering: compartments carrying the same
/// dimension set with the same values get the same deme.
///
/// Mixed-radix, `ms.dimensions` order — the first dimension is the most
/// significant digit, each digit's radix that dimension's cardinality.
fn encode_deme(ms: &ModelStructure, row: &[u32]) -> DemeId {
    let mut deme: u64 = 0;
    for (d, dim) in ms.dimensions.iter().enumerate() {
        let radix = dim.values.len().max(1) as u64;
        let digit = match row.get(d) {
            Some(&v) if v != NOT_STRATIFIED => v as u64,
            _ => 0,
        };
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

    /// `stratum_members` groups by the compartment's own dimension values, not
    /// by the deme digest — so a two-dimensional cell stays a cell.
    #[test]
    fn stratum_members_respects_every_dimension() {
        let structure = ms(
            vec![("age", vec!["child", "adult"]), ("patch", vec!["a", "b"])],
            vec![("S", vec!["age", "patch"]), ("I", vec!["age", "patch"])],
        );
        let mut model = ir::Model { model_structure: Some(structure), ..minimal_model() };
        let names = [
            "S_child_a", "S_child_b", "S_adult_a", "S_adult_b",
            "I_child_a", "I_child_b", "I_adult_a", "I_adult_b",
        ];
        let comp_index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect();
        model.compartments = names
            .iter()
            .map(|n| Compartment { name: n.to_string(), kind: CompartmentKind::Integer })
            .collect();
        let dm = DemeMap::build(&model, &comp_index);

        let members: Vec<&str> = dm
            .stratum_members(CompartmentId(comp_index["S_child_a"]))
            .map(|c| names[c.0])
            .collect();
        assert_eq!(members, vec!["S_child_a", "I_child_a"]);
    }

    /// Partial stratification (`stratify(by = latent_stage, only = [E])`): the
    /// unstratified `S` names the whole model, while `E_e1` names only itself.
    /// The deme digest cannot express this — it puts `S` and `E_e1` both in
    /// deme 0 — which is why `stratum_members` reads the assignment row.
    #[test]
    fn stratum_members_separates_unstratified_from_the_first_cell() {
        let structure = ms(
            vec![("latent_stage", vec!["e1", "e2", "e3"])],
            vec![("S", vec![]), ("I", vec![]), ("E", vec!["latent_stage"])],
        );
        let mut model = ir::Model { model_structure: Some(structure), ..minimal_model() };
        let names = ["S", "I", "E_e1", "E_e2", "E_e3"];
        let comp_index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect();
        model.compartments = names
            .iter()
            .map(|n| Compartment { name: n.to_string(), kind: CompartmentKind::Integer })
            .collect();
        let dm = DemeMap::build(&model, &comp_index);

        // The digest lumps them together …
        assert_eq!(dm.deme_of(CompartmentId(comp_index["S"])), DemeId(0));
        assert_eq!(dm.deme_of(CompartmentId(comp_index["E_e1"])), DemeId(0));
        // … the assignment row does not.
        let s_members: Vec<&str> =
            dm.stratum_members(CompartmentId(comp_index["S"])).map(|c| names[c.0]).collect();
        assert_eq!(s_members, vec!["S", "I", "E_e1", "E_e2", "E_e3"]);
        let e1_members: Vec<&str> =
            dm.stratum_members(CompartmentId(comp_index["E_e1"])).map(|c| names[c.0]).collect();
        assert_eq!(e1_members, vec!["E_e1"]);
    }

    /// A model with no dimensions has one stratum containing everything, even
    /// when a compartment name contains `_`.
    #[test]
    fn stratum_members_of_an_unstratified_model_is_everything() {
        let mut model = minimal_model();
        let names = ["S", "I", "n_vax"];
        let comp_index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect();
        model.compartments = names
            .iter()
            .map(|n| Compartment { name: n.to_string(), kind: CompartmentKind::Integer })
            .collect();
        let dm = DemeMap::build(&model, &comp_index);
        for n in names {
            let members: Vec<&str> =
                dm.stratum_members(CompartmentId(comp_index[n])).map(|c| names[c.0]).collect();
            assert_eq!(members, vec!["S", "I", "n_vax"], "stratum of {n}");
        }
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
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }
}
