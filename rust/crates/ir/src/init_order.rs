//! Dependency order for `init { }` (gh#733).
//!
//! An initial-condition spec may read another compartment's initial value
//! (`init { I = I0   S = N0 - I }`), so the entries form a directed graph over
//! compartment references and have to be evaluated in topological order:
//! dependencies first, each entry against the partially built state. A
//! reference cycle has no evaluation order and is rejected.
//!
//! Two consumers: [`crate::validate`] rejects a cycle at the load boundary, and
//! `sim::CompiledModel::new` sorts the entries once and evaluates in that
//! order. `ocaml/lib/ir/init_order.ml` is the OCaml half of the same agreement
//! — it sorts the same graph the same way so the compiler's IC-gradient
//! inlining and the runtime cannot disagree about the order.

use std::collections::{HashMap, HashSet};

use crate::expr::Expr;
use crate::model::{Binding, InitialConditions};

/// Compartments an initial-condition expression reads.
///
/// `BindingRef` is followed into the model's hoisted bindings: `let N = S+I+R`
/// used in an init RHS is a real dependency on S, I and R, because the runtime
/// evaluates the binding body against whatever state exists at that moment.
/// Treating it as a leaf would order the entry wrongly and read zeros.
/// Bindings are emitted in dependency order and are acyclic; the `seen` set is
/// a belt-and-braces guard so a drifted or hand-written IR cannot spin here.
pub fn deps(expr: &Expr, bindings: &[Binding]) -> Vec<String> {
    let bodies: HashMap<&str, &Expr> =
        bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect();
    let mut out: Vec<String> = Vec::new();
    let mut seen_bindings: HashSet<&str> = HashSet::new();
    walk(expr, &bodies, &mut seen_bindings, &mut out);
    out
}

fn walk<'a>(
    expr: &'a Expr,
    bodies: &HashMap<&'a str, &'a Expr>,
    seen_bindings: &mut HashSet<&'a str>,
    out: &mut Vec<String>,
) {
    let push = |c: &String, out: &mut Vec<String>| {
        if !out.iter().any(|x| x == c) {
            out.push(c.clone());
        }
    };
    match expr {
        Expr::Const(_)
        | Expr::Param(_)
        | Expr::Time(_)
        | Expr::Dt(_)
        | Expr::TimeFunc(_)
        | Expr::Projected(_)
        | Expr::ObsColumnRef(_)
        | Expr::ObsAnchor(_)
        | Expr::PerEvalRef(_) => {}
        Expr::Pop(p) => push(&p.pop, out),
        Expr::PopSum(ps) => {
            for name in &ps.pop_sum {
                push(name, out);
            }
        }
        Expr::BinOp(w) => {
            walk(&w.bin_op.left, bodies, seen_bindings, out);
            walk(&w.bin_op.right, bodies, seen_bindings, out);
        }
        Expr::UnOp(w) => walk(&w.un_op.arg, bodies, seen_bindings, out),
        Expr::Cond(w) => {
            walk(&w.cond.pred, bodies, seen_bindings, out);
            walk(&w.cond.then, bodies, seen_bindings, out);
            walk(&w.cond.else_, bodies, seen_bindings, out);
        }
        Expr::TableLookup(w) => {
            for i in &w.table_lookup.indices {
                walk(i, bodies, seen_bindings, out);
            }
        }
        Expr::UncheckedDim(w) => walk(&w.unchecked_dim.inner, bodies, seen_bindings, out),
        Expr::Reduce(w) => {
            for t in &w.reduce {
                walk(t, bodies, seen_bindings, out);
            }
        }
        Expr::BindingRef(w) => {
            let name = w.binding_ref.as_str();
            if seen_bindings.insert(name) {
                if let Some(body) = bodies.get(name) {
                    walk(body, bodies, seen_bindings, out);
                }
            }
        }
    }
}

/// Topologically sort the init entries.
///
/// `Ok(order)` lists **indices into `ic`** with every dependency before its
/// dependant; ties are broken by declaration order, so the result is
/// deterministic. `Err(cycle)` names the compartments on one reference cycle,
/// in the order they close it (`A -> B -> A` reports `["A", "B"]`).
///
/// A referenced compartment with no init entry is NOT an edge: it starts at 0
/// (the default) and constrains nothing.
pub fn topo(ic: &InitialConditions, bindings: &[Binding]) -> Result<Vec<usize>, Vec<String>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Grey,
        Black,
    }
    let mut mark: HashMap<usize, Mark> = HashMap::new();
    let mut order: Vec<usize> = Vec::with_capacity(ic.len());
    let mut cycle: Option<Vec<String>> = None;

    // Iterative DFS with an explicit stack: an init block is small, but a
    // recursive walk here would be a stack-depth cliff on a generated model
    // with a long dependency chain.
    enum Step {
        Enter(usize),
        Exit(usize),
    }
    for root in 0..ic.len() {
        if mark.contains_key(&root) {
            continue;
        }
        // `path` mirrors the Enter frames still open, so a Grey hit can name
        // the cycle.
        let mut path: Vec<usize> = Vec::new();
        let mut stack: Vec<Step> = vec![Step::Enter(root)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Exit(i) => {
                    mark.insert(i, Mark::Black);
                    path.pop();
                    order.push(i);
                }
                Step::Enter(i) => {
                    match mark.get(&i) {
                        Some(Mark::Black) => continue,
                        Some(Mark::Grey) => {
                            if cycle.is_none() {
                                // The cycle is the open path from `i` onwards.
                                let start = path.iter().position(|&p| p == i).unwrap_or(0);
                                cycle = Some(
                                    path[start..]
                                        .iter()
                                        .map(|&p| ic.0.get_index(p).unwrap().0.clone())
                                        .collect(),
                                );
                            }
                            continue;
                        }
                        None => {}
                    }
                    mark.insert(i, Mark::Grey);
                    path.push(i);
                    stack.push(Step::Exit(i));
                    // EVERY expression the spec evaluates is an edge source: a
                    // law's arguments are evaluated against the partially built
                    // state exactly as a deterministic RHS is, so
                    // `I ~ binomial(n = N0 - R, p = q)` depends on `R`.
                    let spec = ic.0.get_index(i).unwrap().1;
                    // Push dependencies in reverse so the first-declared
                    // dependency is visited first.
                    let d: Vec<String> = {
                        let mut acc: Vec<String> = Vec::new();
                        for e in spec.exprs() {
                            for name in deps(e, bindings) {
                                if !acc.contains(&name) {
                                    acc.push(name);
                                }
                            }
                        }
                        acc
                    };
                    for name in d.iter().rev() {
                        if let Some(j) = ic.0.get_index_of(name.as_str()) {
                            if mark.get(&j) != Some(&Mark::Black) {
                                stack.push(Step::Enter(j));
                            }
                        }
                    }
                }
            }
        }
    }
    match cycle {
        Some(c) => Err(c),
        None => Ok(order),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::BinOp;

    fn ic(entries: &[(&str, Expr)]) -> InitialConditions {
        InitialConditions::exprs(entries.iter().map(|(k, e)| (k.to_string(), e.clone())))
    }

    /// Resolve a topological order back to names, so the assertions read as the
    /// evaluation order a modeller would reason about.
    fn order_names(ic: &InitialConditions, bindings: &[Binding]) -> Vec<String> {
        topo(ic, bindings)
            .expect("acyclic")
            .into_iter()
            .map(|i| ic.0.get_index(i).unwrap().0.clone())
            .collect()
    }

    #[test]
    fn a_dependency_is_ordered_before_its_dependant() {
        let block = ic(&[
            ("B", Expr::bin_op(BinOp::Sub, Expr::param("N"), Expr::pop("A"))),
            ("A", Expr::param("N")),
        ]);
        assert_eq!(order_names(&block, &[]), vec!["A", "B"]);
    }

    /// Independent entries keep the order the model file wrote them in, so two
    /// models that differ only in an irrelevant reordering do not silently
    /// evaluate in a different order (and, through `ContentAddressed`, key
    /// differently for a reason nobody can see).
    #[test]
    fn independent_entries_keep_declaration_order() {
        let block = ic(&[
            ("C", Expr::const_(3.0)),
            ("A", Expr::const_(1.0)),
            ("B", Expr::const_(2.0)),
        ]);
        assert_eq!(order_names(&block, &[]), vec!["C", "A", "B"]);
    }

    /// A compartment named in an init RHS that has no init entry of its own is
    /// NOT an edge: it starts at 0 (the runtime default) and constrains
    /// nothing. Treating it as an edge would be a dangling reference, and
    /// treating it as a cycle would reject a legal model.
    #[test]
    fn a_reference_to_an_unseeded_compartment_imposes_no_order() {
        let block = ic(&[
            ("B", Expr::bin_op(BinOp::Sub, Expr::param("N"), Expr::pop("Unseeded"))),
            ("A", Expr::const_(1.0)),
        ]);
        assert_eq!(order_names(&block, &[]), vec!["B", "A"]);
    }

    /// `deps` follows a `BindingRef` into the binding body, transitively. A
    /// binding is evaluated against whatever state exists when it is read, so
    /// `let outer = inner` / `let inner = S + I` makes an init entry that reads
    /// `outer` depend on both S and I.
    #[test]
    fn deps_follows_a_binding_chain_to_the_compartments_at_the_end() {
        let bindings = vec![
            Binding { name: "outer".into(), expr: Expr::binding_ref("inner") },
            Binding { name: "inner".into(), expr: Expr::pop_sum(vec!["S".into(), "I".into()]) },
        ];
        let got = deps(&Expr::binding_ref("outer"), &bindings);
        assert_eq!(got, vec!["S".to_string(), "I".to_string()]);
    }

    /// A binding naming itself cannot make the walk spin. The OCaml compiler
    /// emits bindings in dependency order and never produces this, so the guard
    /// is for a hand-written or drifted IR — but "it cannot happen" is not a
    /// reason to hang instead of returning.
    #[test]
    fn a_self_referential_binding_terminates() {
        let bindings = vec![Binding { name: "loop".into(), expr: Expr::binding_ref("loop") }];
        assert!(deps(&Expr::binding_ref("loop"), &bindings).is_empty());
    }

    /// The reported cycle is the whole loop, in the order the references close
    /// it, so the caller can render `A -> B -> A` rather than an unordered set.
    #[test]
    fn a_cycle_is_reported_in_the_order_it_closes() {
        let block = ic(&[
            ("A", Expr::pop("B")),
            ("B", Expr::pop("C")),
            ("C", Expr::pop("A")),
        ]);
        assert_eq!(
            topo(&block, &[]).expect_err("cyclic"),
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );
    }
}
