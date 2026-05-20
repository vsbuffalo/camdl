//! Offline projection: line list → transmission tree → Newick.
//!
//! Pure functions over a parsed line list. No simulation dependency — the
//! input is the on-disk artifact ([`super::writer`] columns), the output is a
//! Newick string. Independently unit-testable and re-runnable (the proposal's
//! content-addressable "offline" half).
//!
//! Pipeline:
//!   1. [`TransmissionForest::from_entries`] — build parent→child edges from
//!      lineage events (`parent_kind == "individual"`). Each child has exactly
//!      one parent; seed / import individuals are roots.
//!   2. [`SamplingScheme`] — select observed tips. Phase 1 ships [`Flat`]
//!      (each candidate tip sampled i.i.d. with a fixed probability).
//!   3. [`TransmissionForest::prune_to`] — restrict to the minimal subtree
//!      spanning the sampled tips, suppressing unary internal nodes.
//!   4. [`PrunedTree::to_newick`] — render Newick with branch lengths in time
//!      units.

use std::collections::{HashMap, HashSet};

use crate::error::SimError;
use crate::rng::StatefulRng;

use super::writer::LineListEntry;
use super::{IndividualId, ParentRef};

/// A single node in the raw transmission forest: an individual, the time it
/// was born (its lineage-event time), and its parent (if any).
#[derive(Debug, Clone)]
pub struct Node {
    pub id: IndividualId,
    pub birth_time: f64,
    pub parent: Option<IndividualId>,
    pub children: Vec<IndividualId>,
}

/// The full parent→child structure recovered from a line list. May be a forest
/// (multiple seed/import roots).
#[derive(Debug, Clone)]
pub struct TransmissionForest {
    pub nodes: HashMap<IndividualId, Node>,
    pub roots: Vec<IndividualId>,
}

impl TransmissionForest {
    /// Build the forest from line-list entries. Only lineage events
    /// (`parent_kind == individual`) create edges; the focal `individual` of
    /// such an event is the child, `parent_id` is its parent.
    ///
    /// An individual that is born at an import / seed event (or first seen as a
    /// parent) is a root. Birth time is the event time at which the individual
    /// appears as the focal `individual`.
    pub fn from_entries(entries: &[LineListEntry]) -> Self {
        let mut nodes: HashMap<IndividualId, Node> = HashMap::new();

        // First pass: register every focal individual with its birth time and
        // parent. The earliest event mentioning an individual as focal is its
        // birth.
        for e in entries {
            let parent = match e.parent {
                ParentRef::Individual(p) => Some(p),
                _ => None,
            };
            nodes
                .entry(e.individual)
                .and_modify(|n| {
                    if e.time < n.birth_time {
                        n.birth_time = e.time;
                        n.parent = parent.or(n.parent);
                    }
                })
                .or_insert(Node {
                    id: e.individual,
                    birth_time: e.time,
                    parent,
                    children: Vec::new(),
                });
            // Ensure any referenced parent exists as a node (it may have been
            // a seed individual that only ever appears as a parent).
            if let Some(p) = parent {
                nodes.entry(p).or_insert(Node {
                    id: p,
                    birth_time: 0.0,
                    parent: None,
                    children: Vec::new(),
                });
            }
        }

        // Second pass: wire children lists and collect roots.
        let ids: Vec<IndividualId> = nodes.keys().copied().collect();
        for id in &ids {
            let parent = nodes[id].parent;
            if let Some(p) = parent {
                if let Some(pn) = nodes.get_mut(&p) {
                    pn.children.push(*id);
                }
            }
        }
        let mut roots: Vec<IndividualId> = nodes
            .values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();
        roots.sort();
        // Deterministic child ordering for reproducible Newick.
        for n in nodes.values_mut() {
            n.children.sort();
        }

        TransmissionForest { nodes, roots }
    }

    /// All individuals that are leaves of the *full* forest (no children) —
    /// the candidate tips a sampling scheme draws from.
    pub fn leaves(&self) -> Vec<IndividualId> {
        let mut v: Vec<IndividualId> = self
            .nodes
            .values()
            .filter(|n| n.children.is_empty())
            .map(|n| n.id)
            .collect();
        v.sort();
        v
    }

    /// Prune the forest to the minimal subtree spanning `sampled`, suppressing
    /// unary internal nodes (degree-2 path compression). Branch lengths
    /// accumulate the birth-time differences across suppressed nodes.
    ///
    /// Returns one [`PrunedTree`] per root that retains ≥ 1 sampled descendant.
    pub fn prune_to(&self, sampled: &HashSet<IndividualId>) -> Vec<PrunedTree> {
        // Mark every ancestor of a sampled tip as "retained".
        let mut retained: HashSet<IndividualId> = HashSet::new();
        for &s in sampled {
            let mut cur = Some(s);
            while let Some(id) = cur {
                if !retained.insert(id) {
                    break; // already walked this ancestor chain
                }
                cur = self.nodes.get(&id).and_then(|n| n.parent);
            }
        }

        let mut trees = Vec::new();
        for &root in &self.roots {
            if !retained.contains(&root) {
                continue;
            }
            if let Some(t) = self.build_pruned(root, sampled, &retained) {
                trees.push(t);
            }
        }
        trees
    }

    /// Recursively build a path-compressed pruned subtree rooted at `id`.
    /// Returns `None` if the subtree contains no sampled tip.
    fn build_pruned(
        &self,
        id: IndividualId,
        sampled: &HashSet<IndividualId>,
        retained: &HashSet<IndividualId>,
    ) -> Option<PrunedNode> {
        let node = self.nodes.get(&id)?;
        let kept_children: Vec<IndividualId> = node
            .children
            .iter()
            .copied()
            .filter(|c| retained.contains(c))
            .collect();

        let is_sampled_tip = sampled.contains(&id);

        // Build retained children subtrees.
        let mut child_subtrees: Vec<PrunedNode> = Vec::new();
        for c in &kept_children {
            if let Some(sub) = self.build_pruned(*c, sampled, retained) {
                child_subtrees.push(sub);
            }
        }

        // Path compression: a node that is not a sampled tip and has exactly
        // one retained child collapses into that child, summing branch lengths.
        if !is_sampled_tip && child_subtrees.len() == 1 {
            let mut only = child_subtrees.pop().unwrap();
            // Add this node's own incoming branch length onto the child by
            // shifting the child's birth reference up to this node's parent.
            only.branch_from = node.parent.and_then(|p| self.nodes.get(&p)).map(|p| p.birth_time);
            return Some(only);
        }

        if !is_sampled_tip && child_subtrees.is_empty() {
            return None;
        }

        let branch_from = node.parent.and_then(|p| self.nodes.get(&p)).map(|p| p.birth_time);
        Some(PrunedNode {
            id,
            birth_time: node.birth_time,
            branch_from,
            children: child_subtrees,
            is_sampled_tip,
        })
    }
}

/// A node in a pruned tree. `branch_from` is the birth time of the most-recent
/// retained ancestor (for branch-length computation); `None` at the root.
#[derive(Debug, Clone)]
pub struct PrunedNode {
    pub id: IndividualId,
    pub birth_time: f64,
    pub branch_from: Option<f64>,
    pub children: Vec<PrunedNode>,
    pub is_sampled_tip: bool,
}

/// A pruned, path-compressed tree rooted at one forest root.
pub type PrunedTree = PrunedNode;

impl PrunedNode {
    /// Render Newick. Tips are labelled `ind<id>`; internal nodes are unlabelled.
    /// Branch lengths are `birth_time − branch_from` (0 at the root).
    pub fn to_newick(&self) -> String {
        let mut s = String::new();
        self.write_newick(&mut s);
        s.push(';');
        s
    }

    fn write_newick(&self, out: &mut String) {
        if self.children.is_empty() {
            out.push_str(&format!("ind{}", self.id.0));
        } else {
            out.push('(');
            for (i, c) in self.children.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                c.write_newick(out);
            }
            out.push(')');
            // Internal nodes are unlabelled; label sampled internal tips for
            // completeness (rare: a sampled individual that also infected).
            if self.is_sampled_tip {
                out.push_str(&format!("ind{}", self.id.0));
            }
        }
        let bl = self.branch_length();
        out.push_str(&format!(":{}", bl));
    }

    /// Branch length subtending this node (0 at a root).
    pub fn branch_length(&self) -> f64 {
        match self.branch_from {
            Some(from) => (self.birth_time - from).max(0.0),
            None => 0.0,
        }
    }

    /// Count of sampled tips in this tree (Sackin / structural tests).
    pub fn tip_count(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            self.children.iter().map(|c| c.tip_count()).sum()
        }
    }

    /// Sackin index: sum over tips of the number of internal nodes on the
    /// path from the tip to the root (tip depth in edges). A standard tree
    /// imbalance statistic.
    pub fn sackin(&self) -> usize {
        self.sackin_at(0)
    }

    fn sackin_at(&self, depth: usize) -> usize {
        if self.children.is_empty() {
            depth
        } else {
            self.children.iter().map(|c| c.sackin_at(depth + 1)).sum()
        }
    }
}

/// A scheme that selects which candidate tips become observed samples. Phase 1
/// implements only [`Flat`]; the trait fixes the shape so later schemes
/// (per-deme, time-varying, conditional-on-removal) slot in without changing
/// callers.
pub trait SamplingScheme {
    /// Given the full forest and a deterministic RNG, return the set of
    /// individuals selected as observed tips.
    fn select(&self, forest: &TransmissionForest, rng: &mut StatefulRng) -> HashSet<IndividualId>;
}

/// Flat sampling: each leaf of the full forest is sampled i.i.d. with
/// probability `rate`.
pub struct Flat {
    pub rate: f64,
}

impl Flat {
    pub fn new(rate: f64) -> Self {
        Flat { rate }
    }
}

impl SamplingScheme for Flat {
    fn select(&self, forest: &TransmissionForest, rng: &mut StatefulRng) -> HashSet<IndividualId> {
        let p = self.rate.clamp(0.0, 1.0);
        forest
            .leaves()
            .into_iter()
            .filter(|_| rng.uniform() < p)
            .collect()
    }
}

// ── Line-list reading (offline input) ─────────────────────────────────────────

/// Parse a TSV line list (the [`super::writer::COLUMNS`] layout) back into
/// entries. Used by the offline `camdl lineage tree` path and tests.
pub fn read_tsv(path: &std::path::Path) -> Result<Vec<LineListEntry>, SimError> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| SimError::Validation(format!("read line list {}: {}", path.display(), e)))?;
    let mut lines = body.lines();
    let header = lines
        .next()
        .ok_or_else(|| SimError::Validation("empty line list".to_string()))?;
    if header != super::writer::COLUMNS.join("\t") {
        return Err(SimError::Validation(format!(
            "line list header mismatch: expected '{}', got '{}'",
            super::writer::COLUMNS.join("\t"),
            header
        )));
    }
    let mut out = Vec::new();
    for (lineno, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != super::writer::COLUMNS.len() {
            return Err(SimError::Validation(format!(
                "line list row {}: expected {} columns, got {}",
                lineno + 2,
                super::writer::COLUMNS.len(),
                f.len()
            )));
        }
        let parse_i64 = |s: &str| -> Result<i64, SimError> {
            s.parse::<i64>()
                .map_err(|e| SimError::Validation(format!("line list parse '{}': {}", s, e)))
        };
        let comp_opt = |v: i64| -> Option<usize> {
            if v < 0 {
                None
            } else {
                Some(v as usize)
            }
        };
        let time: f64 = f[0]
            .parse()
            .map_err(|e| SimError::Validation(format!("line list time '{}': {}", f[0], e)))?;
        let transition = parse_i64(f[1])? as usize;
        let individual = IndividualId(parse_i64(f[2])? as u64);
        let source = comp_opt(parse_i64(f[3])?);
        let destination = comp_opt(parse_i64(f[4])?);
        let deme = parse_i64(f[5])? as u32;
        let parent_id = parse_i64(f[7])?;
        let parent = match f[6] {
            "individual" => ParentRef::Individual(IndividualId(parent_id as u64)),
            "import" => ParentRef::Import,
            "seed" => ParentRef::Seed,
            "none" => ParentRef::None,
            other => {
                return Err(SimError::Validation(format!(
                    "line list row {}: unknown parent_kind '{}'",
                    lineno + 2,
                    other
                )))
            }
        };
        out.push(LineListEntry {
            time,
            transition,
            individual,
            source,
            destination,
            deme,
            parent,
        });
    }
    Ok(out)
}

#[cfg(feature = "lineage-parquet")]
pub fn read_parquet(path: &std::path::Path) -> Result<Vec<LineListEntry>, SimError> {
    use arrow::array::{Float64Array, Int64Array, StringArray, UInt32Array, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path)
        .map_err(|e| SimError::Validation(format!("open line list {}: {}", path.display(), e)))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| SimError::Validation(format!("parquet reader: {}", e)))?;
    let reader = builder
        .build()
        .map_err(|e| SimError::Validation(format!("parquet reader build: {}", e)))?;

    let mut out = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| SimError::Validation(format!("parquet batch: {}", e)))?;
        let col = |i: usize| batch.column(i);
        let time = col(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let transition = col(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        let individual = col(2).as_any().downcast_ref::<UInt64Array>().unwrap();
        let source = col(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let destination = col(4).as_any().downcast_ref::<Int64Array>().unwrap();
        let deme = col(5).as_any().downcast_ref::<UInt32Array>().unwrap();
        let parent_kind = col(6).as_any().downcast_ref::<StringArray>().unwrap();
        let parent_id = col(7).as_any().downcast_ref::<Int64Array>().unwrap();
        let comp_opt = |v: i64| if v < 0 { None } else { Some(v as usize) };
        for r in 0..batch.num_rows() {
            let parent = match parent_kind.value(r) {
                "individual" => ParentRef::Individual(IndividualId(parent_id.value(r) as u64)),
                "import" => ParentRef::Import,
                "seed" => ParentRef::Seed,
                _ => ParentRef::None,
            };
            out.push(LineListEntry {
                time: time.value(r),
                transition: transition.value(r) as usize,
                individual: IndividualId(individual.value(r)),
                source: comp_opt(source.value(r)),
                destination: comp_opt(destination.value(r)),
                deme: deme.value(r),
                parent,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineage_entry(t: f64, ind: u64, parent: u64, dst: usize) -> LineListEntry {
        LineListEntry {
            time: t,
            transition: 0,
            individual: IndividualId(ind),
            source: Some(0),
            destination: Some(dst),
            deme: 0,
            parent: ParentRef::Individual(IndividualId(parent)),
        }
    }

    #[test]
    fn forest_builds_edges_and_roots() {
        // Seed individual 0 (appears only as a parent → root).
        // 0 -> 1 at t=1, 0 -> 2 at t=2, 1 -> 3 at t=3.
        let entries = vec![
            lineage_entry(1.0, 1, 0, 1),
            lineage_entry(2.0, 2, 0, 1),
            lineage_entry(3.0, 3, 1, 1),
        ];
        let f = TransmissionForest::from_entries(&entries);
        assert_eq!(f.roots, vec![IndividualId(0)]);
        assert_eq!(f.nodes[&IndividualId(0)].children, vec![IndividualId(1), IndividualId(2)]);
        assert_eq!(f.nodes[&IndividualId(1)].children, vec![IndividualId(3)]);
        // Leaves are 2 and 3.
        assert_eq!(f.leaves(), vec![IndividualId(2), IndividualId(3)]);
    }

    #[test]
    fn every_child_has_exactly_one_parent() {
        let entries = vec![
            lineage_entry(1.0, 1, 0, 1),
            lineage_entry(3.0, 3, 1, 1),
            lineage_entry(2.0, 2, 0, 1),
        ];
        let f = TransmissionForest::from_entries(&entries);
        for n in f.nodes.values() {
            if n.id != IndividualId(0) {
                assert!(n.parent.is_some(), "non-root {} must have a parent", n.id.0);
            }
        }
    }

    #[test]
    fn prune_suppresses_unary_nodes_and_keeps_sampled_tips() {
        // Chain: 0 -> 1 -> 2 -> 3, plus 0 -> 4. Sample {3, 4}.
        // Pruned tree should connect root 0 to tips 3 and 4 with the
        // intermediate unary nodes 1,2 suppressed.
        let entries = vec![
            lineage_entry(1.0, 1, 0, 1),
            lineage_entry(2.0, 2, 1, 1),
            lineage_entry(3.0, 3, 2, 1),
            lineage_entry(1.5, 4, 0, 1),
        ];
        let f = TransmissionForest::from_entries(&entries);
        let sampled: HashSet<IndividualId> =
            [IndividualId(3), IndividualId(4)].into_iter().collect();
        let trees = f.prune_to(&sampled);
        assert_eq!(trees.len(), 1);
        let t = &trees[0];
        // Two tips.
        assert_eq!(t.tip_count(), 2);
        // No unary internal nodes: the root has exactly the two tip children.
        assert_eq!(t.children.len(), 2);
        for c in &t.children {
            assert!(c.children.is_empty(), "tips must be leaves");
            assert!(c.is_sampled_tip);
        }
        // Pruned tips equal the sampled set.
        let mut tip_ids: Vec<u64> = t.children.iter().map(|c| c.id.0).collect();
        tip_ids.sort();
        assert_eq!(tip_ids, vec![3, 4]);
        // Newick parses to a 2-tip cherry.
        let nwk = t.to_newick();
        assert!(nwk.starts_with('(') && nwk.ends_with(';'));
        assert!(nwk.contains("ind3") && nwk.contains("ind4"));
    }

    #[test]
    fn flat_scheme_is_deterministic_given_seed() {
        let entries = vec![
            lineage_entry(1.0, 1, 0, 1),
            lineage_entry(2.0, 2, 0, 1),
            lineage_entry(3.0, 3, 1, 1),
        ];
        let f = TransmissionForest::from_entries(&entries);
        let mut r1 = StatefulRng::new(123);
        let mut r2 = StatefulRng::new(123);
        let s1 = Flat::new(0.5).select(&f, &mut r1);
        let s2 = Flat::new(0.5).select(&f, &mut r2);
        assert_eq!(s1, s2);
    }
}
