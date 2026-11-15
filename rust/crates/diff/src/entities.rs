//! A declared model as the list of things a diff has a row for.
//!
//! Each declaration becomes one [`Entity`]: its category, the name it is
//! matched by, and its document form. The whole of a [`DeclaredModel`] is
//! covered: the destructuring in [`entities`] names every member, so a member
//! added to the declared model does not compile here until it is given a row.

use std::collections::BTreeMap;

use declared::{DeclaredModel, ExprTree, IndexBinding, IndexItem, SourceText};
use serde::Serialize;
use serde_json::{json, Value};

use crate::{document, Category};

pub struct Entity {
    pub category: Category,
    /// The name the entity is matched by, unique within its category.
    pub name: String,
    /// The declaration as the `declared` crate writes it.
    pub doc: Value,
    pub source_text: Option<String>,
    /// Every name the declaration reads, with its kind.
    pub reads: Vec<(String, String)>,
    /// The names, with their kind, by which an expression refers to this
    /// entity. Empty for an entity nothing can read.
    pub provides: Vec<(String, String)>,
}

fn document_of<T: Serialize>(declaration: &T) -> Value {
    // The declared types are plain records and enums with string keys; their
    // serialization to a value cannot fail.
    serde_json::to_value(declaration).expect("a declared type serializes to JSON")
}

fn entity<T: Serialize>(category: Category, name: String, declaration: &T, source_text: &SourceText) -> Entity {
    let doc = document_of(declaration);
    let reads = document::names(&doc);
    Entity { category, name, doc, source_text: source_text.clone(), reads, provides: Vec::new() }
}

fn provided(mut e: Entity, kind: &str, names: &[String]) -> Entity {
    e.provides = names.iter().map(|n| (n.clone(), kind.to_string())).collect();
    e
}

/// An index expression as text, for a row's name. Names and numbers are
/// written as the author would; anything else falls back to its document form,
/// which is unambiguous if not pretty.
fn expr_text(tree: &ExprTree) -> String {
    match tree {
        ExprTree::Ident { name } => name.name.clone(),
        ExprTree::Const { value } => value.to_string(),
        ExprTree::Index { name, indices } => format!("{}{}", name.name, index_text(indices)),
        other => serde_json::to_string(other).expect("an expression serializes to JSON"),
    }
}

fn index_text(indices: &[IndexItem]) -> String {
    if indices.is_empty() {
        return String::new();
    }
    let items: Vec<String> = indices
        .iter()
        .map(|item| match item {
            IndexItem::Positional { value } => expr_text(value),
            IndexItem::Named { dim, value } => format!("{dim} = {}", expr_text(value)),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

fn binding_text(bindings: &[IndexBinding]) -> String {
    if bindings.is_empty() {
        return String::new();
    }
    let items: Vec<String> = bindings
        .iter()
        .map(|b| match b {
            IndexBinding::In { var, dim } => format!("{var} in {dim}"),
            IndexBinding::Consecutive { var, var_next, dim } => {
                format!("({var}, {var_next}) in consecutive({dim})")
            }
            IndexBinding::Compartments { var } => format!("{var} in compartments"),
        })
        .collect();
    format!(" [{}]", items.join(", "))
}

/// Give the second and later entities of one name in one category a numbered
/// name, so that every entity is matched exactly once. Two declarations of one
/// name are then matched in the order they were written.
fn number_repeats(entities: &mut [Entity]) {
    let mut seen: BTreeMap<(Category, String), usize> = BTreeMap::new();
    for e in entities.iter_mut() {
        let count = seen.entry((e.category, e.name.clone())).or_insert(0);
        *count += 1;
        if *count > 1 {
            e.name = format!("{} ({})", e.name, count);
        }
    }
}

/// Every entity of a declared model, grouped by category in the order of
/// [`Category`], and in declaration order within a category.
pub fn entities(model: &DeclaredModel) -> Vec<Entity> {
    // Every member is named, with no `..`: a new member is a compile error
    // here until it has a row.
    let DeclaredModel {
        schema_version: _,
        compile_inputs,
        time_unit,
        description,
        origin,
        dimensions,
        compartments,
        stratify,
        parameters,
        definitions,
        tables,
        time_inputs,
        timepoints,
        transitions,
        ode_equations,
        balance,
        initial,
        observations,
        interventions,
        events,
        reactive_interventions,
        scenarios,
        simulation,
        output,
        quantities,
        contrasts,
    } = model;

    let mut out: Vec<Entity> = Vec::new();

    // A compile input and the description have no span in the model file. The
    // text shown for each is what was given: the flag as typed, the file by
    // path and digest, the description itself.
    for set in &compile_inputs.set_values {
        let shown = Some(format!("--set {}={}", set.name, set.value));
        out.push(entity(Category::CompileInput, format!("--set {}", set.name), set, &shown));
    }
    for file in &compile_inputs.files {
        let digest = file.sha256.as_deref().unwrap_or("digest not recorded");
        let shown = Some(format!("{}  sha256 {digest}", file.path));
        out.push(entity(Category::CompileInput, format!("file {}", file.path), file, &shown));
    }

    out.push(entity(
        Category::Description,
        "description".to_string(),
        &json!({ "description": description }),
        description,
    ));

    for d in dimensions {
        out.push(entity(Category::Dimension, d.name.clone(), d, &d.source_text));
    }
    for c in compartments {
        out.push(entity(Category::Compartment, c.name.clone(), c, &c.source_text));
    }
    for s in stratify {
        out.push(entity(Category::Stratification, s.dim.clone(), s, &s.source_text));
    }
    for p in parameters {
        out.push(entity(Category::Parameter, p.name.clone(), p, &p.source_text));
    }
    for d in definitions {
        let e = entity(Category::Definition, d.name.clone(), d, &d.source_text);
        out.push(provided(e, "definition", std::slice::from_ref(&d.name)));
    }
    for t in tables {
        let e = entity(Category::Table, t.names.join(", "), t, &t.source_text);
        out.push(provided(e, "table", &t.names));
    }
    for t in time_inputs {
        let e = entity(Category::TimeInput, t.name.clone(), t, &t.source_text);
        out.push(provided(e, "time_input", std::slice::from_ref(&t.name)));
    }
    for t in timepoints {
        out.push(entity(Category::Timepoint, t.name.clone(), t, &t.source_text));
    }
    for t in transitions {
        out.push(entity(Category::Transition, t.name.clone(), t, &t.source_text));
    }
    for o in ode_equations {
        out.push(entity(Category::OdeEquation, o.compartment.clone(), o, &o.source_text));
    }
    for b in balance {
        out.push(entity(Category::Balance, b.compartment.clone(), b, &b.source_text));
    }
    for i in initial {
        let name = format!("{}{}{}", i.compartment, index_text(&i.indices), binding_text(&i.bindings));
        out.push(entity(Category::InitialCondition, name, i, &i.source_text));
    }
    for o in observations {
        out.push(entity(Category::Observation, o.name.clone(), o, &o.source_text));
    }
    for i in interventions {
        out.push(entity(Category::Intervention, i.name.clone(), i, &i.source_text));
    }
    for e in events {
        out.push(entity(Category::Event, e.name.clone(), e, &e.source_text));
    }
    for r in reactive_interventions {
        out.push(entity(Category::ReactiveIntervention, r.name.clone(), r, &r.source_text));
    }
    for s in scenarios {
        out.push(entity(Category::Scenario, s.name.clone(), s, &s.source_text));
    }

    // One row for the whole of the simulation settings, present whether or not
    // the model has a `simulate` block: the time unit and the origin are
    // top-level declarations, and an absent block is itself a setting.
    let simulate_text = simulation.as_ref().and_then(|s| s.source_text.clone());
    out.push(entity(
        Category::Simulation,
        "simulate".to_string(),
        &json!({ "time_unit": time_unit, "origin": origin, "simulate": simulation }),
        &simulate_text,
    ));
    let output_text = output.as_ref().and_then(|o| o.source_text.clone());
    out.push(entity(Category::Output, "output".to_string(), &json!({ "output": output }), &output_text));

    for q in quantities {
        let e = entity(Category::Quantity, q.name.clone(), q, &q.source_text);
        out.push(provided(e, "quantity", std::slice::from_ref(&q.name)));
    }
    for c in contrasts {
        out.push(entity(Category::Contrast, c.name.clone(), c, &c.source_text));
    }

    number_repeats(&mut out);
    out
}
