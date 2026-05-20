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
//!   2. [`IndividualSummary`] / [`summarize`] — per-individual infection time,
//!      removal time, and deme, derived from the line list. The candidate set
//!      for sampling is **all** individuals, not just leaves.
//!   3. [`SamplingScheme`] — decide, per individual, whether it is sampled and
//!      at what *sampling time* a pendant tip is placed. Ships [`Flat`] (i.i.d.
//!      probability over all individuals) and [`Stratified`] (per-deme rates).
//!   4. [`TransmissionForest::prune_to`] — restrict to the minimal subtree
//!      spanning the sampled tips, placing each sampled individual's pendant tip
//!      at its sampling time and suppressing unsampled unary internal nodes.
//!   5. [`PrunedTree::to_newick`] — render Newick with branch lengths in time
//!      units (tip branches reach the sampling time).

use std::collections::{HashMap, HashSet};

use crate::error::SimError;
use crate::rng::StatefulRng;

use super::writer::LineListEntry;
use super::{DemeId, IndividualId, ParentRef};

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

    /// Prune the forest to the minimal subtree spanning the sampled
    /// individuals, suppressing unsampled unary internal nodes (degree-2 path
    /// compression). `sampled` maps each sampled individual to its **sampling
    /// time** — the time its pendant tip reaches.
    ///
    /// A sampled individual always contributes exactly one tip:
    ///   - if it has no retained children it is a leaf, placed at its sampling
    ///     time;
    ///   - if it is an infector with retained children (an internal node), it
    ///     becomes an internal node *plus* an extra pendant-tip child placed at
    ///     its sampling time. Its onward-transmission subtree is preserved.
    ///
    /// Returns one [`PrunedTree`] per root that retains ≥ 1 sampled descendant.
    pub fn prune_to(&self, sampled: &HashMap<IndividualId, f64>) -> Vec<PrunedTree> {
        // Mark every ancestor of a sampled individual as "retained".
        let mut retained: HashSet<IndividualId> = HashSet::new();
        for &s in sampled.keys() {
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
    /// Returns `None` if the subtree contains no sampled individual.
    fn build_pruned(
        &self,
        id: IndividualId,
        sampled: &HashMap<IndividualId, f64>,
        retained: &HashSet<IndividualId>,
    ) -> Option<PrunedNode> {
        let node = self.nodes.get(&id)?;
        let kept_children: Vec<IndividualId> = node
            .children
            .iter()
            .copied()
            .filter(|c| retained.contains(c))
            .collect();

        let sampling_time = sampled.get(&id).copied();
        let is_sampled = sampling_time.is_some();

        // Build retained children subtrees.
        let mut child_subtrees: Vec<PrunedNode> = Vec::new();
        for c in &kept_children {
            if let Some(sub) = self.build_pruned(*c, sampled, retained) {
                child_subtrees.push(sub);
            }
        }

        // Path compression: an *unsampled* node with exactly one retained child
        // collapses into that child, summing branch lengths.
        if !is_sampled && child_subtrees.len() == 1 {
            let mut only = child_subtrees.pop().unwrap();
            // Add this node's own incoming branch length onto the child by
            // shifting the child's branch reference up to this node's parent.
            only.branch_from = node.parent.and_then(|p| self.nodes.get(&p)).map(|p| p.birth_time);
            return Some(only);
        }

        if !is_sampled && child_subtrees.is_empty() {
            return None;
        }

        let branch_from = node.parent.and_then(|p| self.nodes.get(&p)).map(|p| p.birth_time);

        if let Some(t_sample) = sampling_time {
            if child_subtrees.is_empty() {
                // A sampled leaf: the node *is* the tip, ending at its sampling
                // time. Branch length spans birth → sampling.
                return Some(PrunedNode {
                    id,
                    node_time: t_sample,
                    branch_from,
                    children: Vec::new(),
                    is_sampled_tip: true,
                });
            }
            // A sampled infector (internal node): keep the transmission subtree
            // and add a pendant tip at the sampling time. The internal node sits
            // at the infection (birth) time; the pendant tip hangs off it with
            // branch length (sampling − birth).
            child_subtrees.push(PrunedNode {
                id,
                node_time: t_sample,
                branch_from: Some(node.birth_time),
                children: Vec::new(),
                is_sampled_tip: true,
            });
            return Some(PrunedNode {
                id,
                node_time: node.birth_time,
                branch_from,
                children: child_subtrees,
                is_sampled_tip: false,
            });
        }

        // Unsampled internal node with ≥ 2 retained children: a real coalescence
        // at its birth (transmission) time.
        Some(PrunedNode {
            id,
            node_time: node.birth_time,
            branch_from,
            children: child_subtrees,
            is_sampled_tip: false,
        })
    }
}

/// A node in a pruned tree. `node_time` is the node's calendar time: a sampled
/// tip's *sampling* time, an internal node's transmission (birth) time.
/// `branch_from` is the calendar time of the most-recent retained ancestor (for
/// branch-length computation); `None` at the root.
#[derive(Debug, Clone)]
pub struct PrunedNode {
    pub id: IndividualId,
    pub node_time: f64,
    pub branch_from: Option<f64>,
    pub children: Vec<PrunedNode>,
    pub is_sampled_tip: bool,
}

/// A pruned, path-compressed tree rooted at one forest root.
pub type PrunedTree = PrunedNode;

impl PrunedNode {
    /// Render Newick. Tips (leaves) are labelled `ind<id>`; internal nodes are
    /// unlabelled. Branch lengths are `node_time − branch_from` (0 at the root),
    /// so a sampled tip's branch reaches its sampling time.
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
            // Internal nodes are unlabelled. A sampled infector contributes its
            // own tip as a pendant-leaf child, so it is never labelled here.
        }
        let bl = self.branch_length();
        out.push_str(&format!(":{}", bl));
    }

    /// Branch length subtending this node (0 at a root).
    pub fn branch_length(&self) -> f64 {
        match self.branch_from {
            Some(from) => (self.node_time - from).max(0.0),
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

/// Per-individual facts derived from the line list, independent of the forest
/// topology. The sampling layer draws on these (deme + removal time); the tree
/// builder uses the resulting per-individual sampling time to place pendant
/// tips.
///
/// - `infection_time`: when the individual was born — its earliest focal event
///   time (the lineage event that created it, or its seed/import time). Equal to
///   the forest [`Node::birth_time`].
/// - `removal_time`: when the individual stopped being infectious — the latest
///   time at which it is the focal individual of a *non-lineage* event
///   (progression / recovery / death; `parent_kind == none`). `None` if it never
///   appears in such an event (still infected at the simulation horizon).
/// - `deme`: the individual's stratum, read from the `deme` column of its birth
///   event (its destination stratum at infection / seeding).
#[derive(Debug, Clone, PartialEq)]
pub struct IndividualSummary {
    pub id: IndividualId,
    pub infection_time: f64,
    pub removal_time: Option<f64>,
    pub deme: DemeId,
}

impl IndividualSummary {
    /// The sampling time to use when this individual is observed but never
    /// removed: fall back to the simulation horizon `sim_end` so the pendant tip
    /// reaches the end of observation rather than its (unknown) removal.
    pub fn removal_or(&self, sim_end: f64) -> f64 {
        self.removal_time.unwrap_or(sim_end)
    }
}

/// Build the per-individual summaries from a line list and report the
/// simulation horizon (the maximum event time, used as the sampling time for
/// never-removed individuals). Returns `(summaries, sim_end)`.
///
/// Every individual that appears as a focal individual gets a summary; so do
/// pure-parent seeds (individuals that only ever appear as a `parent_id`), with
/// `infection_time = 0` and `removal_time = None` — matching the forest's
/// treatment of seed roots.
pub fn summarize(entries: &[LineListEntry]) -> (HashMap<IndividualId, IndividualSummary>, f64) {
    let mut map: HashMap<IndividualId, IndividualSummary> = HashMap::new();
    let mut sim_end = 0.0_f64;

    for e in entries {
        sim_end = sim_end.max(e.time);
        let is_lineage = matches!(e.parent, ParentRef::Individual(_));

        // Focal individual: born at its earliest focal-event time; the deme at
        // that earliest event is its stratum.
        let s = map.entry(e.individual).or_insert(IndividualSummary {
            id: e.individual,
            infection_time: e.time,
            removal_time: None,
            deme: e.deme,
        });
        if e.time < s.infection_time {
            s.infection_time = e.time;
            s.deme = e.deme;
        }
        // A non-lineage focal event is a progression / removal: take the latest
        // such time as the removal time.
        if !is_lineage {
            s.removal_time = Some(match s.removal_time {
                Some(t) => t.max(e.time),
                None => e.time,
            });
        }

        // Ensure a pure-parent seed (only ever a parent_id) exists as a root
        // individual: infection_time 0, never removed, deme from parent_deme.
        if let ParentRef::Individual(p) = e.parent {
            map.entry(p).or_insert(IndividualSummary {
                id: p,
                infection_time: 0.0,
                removal_time: None,
                deme: e.parent_deme.unwrap_or(0),
            });
        }
    }

    (map, sim_end)
}

/// A scheme that decides, per individual, whether it is sampled and the
/// *sampling time* at which its pendant tip is placed. The candidate set is
/// **all** individuals (an infector can be a tip), not just chain endpoints.
/// The trait fixes the shape so later schemes (time-varying, conditional-on-
/// removal) slot in without changing callers.
pub trait SamplingScheme {
    /// Given an individual's summary and a deterministic RNG, return its
    /// sampling time (a pendant tip is placed there) or `None` if excluded.
    fn sample(&self, s: &IndividualSummary, rng: &mut StatefulRng) -> Option<f64>;
}

/// Apply a [`SamplingScheme`] to every individual in `summaries`, returning the
/// map `sampled individual → sampling time`. Individuals are visited in id order
/// so the RNG draw order — and therefore the sampled set for a given seed — is
/// deterministic regardless of `HashMap` iteration order.
pub fn select_samples(
    scheme: &dyn SamplingScheme,
    summaries: &HashMap<IndividualId, IndividualSummary>,
    rng: &mut StatefulRng,
) -> HashMap<IndividualId, f64> {
    let mut ids: Vec<IndividualId> = summaries.keys().copied().collect();
    ids.sort();
    let mut out = HashMap::new();
    for id in ids {
        if let Some(t) = scheme.sample(&summaries[&id], rng) {
            out.insert(id, t);
        }
    }
    out
}

/// Flat sampling: each individual is sampled i.i.d. with probability `rate`,
/// over **all** individuals (not just leaves). A sampled individual's tip is
/// placed at its removal time (or the simulation horizon if it was never
/// removed). `sim_end` is the horizon from [`summarize`].
pub struct Flat {
    pub rate: f64,
    pub sim_end: f64,
}

impl Flat {
    pub fn new(rate: f64, sim_end: f64) -> Self {
        Flat { rate, sim_end }
    }
}

impl SamplingScheme for Flat {
    fn sample(&self, s: &IndividualSummary, rng: &mut StatefulRng) -> Option<f64> {
        let p = self.rate.clamp(0.0, 1.0);
        if rng.uniform() < p {
            Some(s.removal_or(self.sim_end))
        } else {
            None
        }
    }
}

/// Stratified sampling: each individual is sampled at its deme's rate (falling
/// back to `default` for demes without an explicit rate). Sampling time = the
/// individual's removal time (or the horizon if never removed). Rates are keyed
/// on the integer deme index; the model-block path that resolves stratum
/// *names* and rates-as-parameters is a future milestone.
pub struct Stratified {
    pub rates: HashMap<DemeId, f64>,
    pub default: f64,
    pub sim_end: f64,
}

impl Stratified {
    pub fn new(rates: HashMap<DemeId, f64>, default: f64, sim_end: f64) -> Self {
        Stratified { rates, default, sim_end }
    }

    /// The sampling probability that applies to deme `d`.
    pub fn rate_for(&self, d: DemeId) -> f64 {
        self.rates.get(&d).copied().unwrap_or(self.default).clamp(0.0, 1.0)
    }
}

impl SamplingScheme for Stratified {
    fn sample(&self, s: &IndividualSummary, rng: &mut StatefulRng) -> Option<f64> {
        let p = self.rate_for(s.deme);
        if rng.uniform() < p {
            Some(s.removal_or(self.sim_end))
        } else {
            None
        }
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
        // parent_deme: -1 sentinel → None (non-lineage event).
        let parent_deme = match parse_i64(f[8])? {
            v if v < 0 => None,
            v => Some(v as u32),
        };
        let attribution_logprob: f64 = f[9].parse().map_err(|e| {
            SimError::Validation(format!("line list attribution_logprob '{}': {}", f[9], e))
        })?;
        out.push(LineListEntry {
            time,
            transition,
            individual,
            source,
            destination,
            deme,
            parent,
            parent_deme,
            attribution_logprob,
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
        let parent_deme = col(8).as_any().downcast_ref::<Int64Array>().unwrap();
        let attribution_logprob = col(9).as_any().downcast_ref::<Float64Array>().unwrap();
        let comp_opt = |v: i64| if v < 0 { None } else { Some(v as usize) };
        for r in 0..batch.num_rows() {
            let parent = match parent_kind.value(r) {
                "individual" => ParentRef::Individual(IndividualId(parent_id.value(r) as u64)),
                "import" => ParentRef::Import,
                "seed" => ParentRef::Seed,
                _ => ParentRef::None,
            };
            let pdeme = match parent_deme.value(r) {
                v if v < 0 => None,
                v => Some(v as u32),
            };
            out.push(LineListEntry {
                time: time.value(r),
                transition: transition.value(r) as usize,
                individual: IndividualId(individual.value(r)),
                source: comp_opt(source.value(r)),
                destination: comp_opt(destination.value(r)),
                deme: deme.value(r),
                parent,
                parent_deme: pdeme,
                attribution_logprob: attribution_logprob.value(r),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineage_entry(t: f64, ind: u64, parent: u64, dst: usize) -> LineListEntry {
        lineage_entry_deme(t, ind, parent, dst, 0)
    }

    fn lineage_entry_deme(t: f64, ind: u64, parent: u64, dst: usize, deme: u32) -> LineListEntry {
        LineListEntry {
            time: t,
            transition: 0,
            individual: IndividualId(ind),
            source: Some(0),
            destination: Some(dst),
            deme,
            parent: ParentRef::Individual(IndividualId(parent)),
            parent_deme: Some(0),
            attribution_logprob: 0.0,
        }
    }

    /// A non-lineage focal event (progression / recovery / removal): focal
    /// individual moves `src -> dst`, no parent. Used to give individuals a
    /// removal time.
    fn removal_entry(t: f64, ind: u64, src: usize, dst: Option<usize>) -> LineListEntry {
        LineListEntry {
            time: t,
            transition: 1,
            individual: IndividualId(ind),
            source: Some(src),
            destination: dst,
            deme: 0,
            parent: ParentRef::None,
            parent_deme: None,
            attribution_logprob: 0.0,
        }
    }

    /// Sampled set with all sampling times equal to `t` (a convenience for
    /// structural prune tests where the exact tip time is not under test).
    fn sampled_at(ids: &[u64], t: f64) -> HashMap<IndividualId, f64> {
        ids.iter().map(|&i| (IndividualId(i), t)).collect()
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
        let sampled = sampled_at(&[3, 4], 5.0);
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
        let (summaries, sim_end) = summarize(&entries);
        let mut r1 = StatefulRng::new(123);
        let mut r2 = StatefulRng::new(123);
        let s1 = select_samples(&Flat::new(0.5, sim_end), &summaries, &mut r1);
        let s2 = select_samples(&Flat::new(0.5, sim_end), &summaries, &mut r2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn summarize_infection_removal_and_deme() {
        // 0 -> 1 (infected t=1, deme 0), 1 recovers at t=4 (I->R).
        // 0 -> 2 (infected t=2, deme 1), never recovers.
        let entries = vec![
            lineage_entry_deme(1.0, 1, 0, 1, 0),
            removal_entry(4.0, 1, 1, Some(2)),
            lineage_entry_deme(2.0, 2, 0, 1, 1),
        ];
        let (s, sim_end) = summarize(&entries);
        assert_eq!(sim_end, 4.0);
        let s1 = &s[&IndividualId(1)];
        assert_eq!(s1.infection_time, 1.0);
        assert_eq!(s1.removal_time, Some(4.0));
        assert_eq!(s1.deme, 0);
        let s2 = &s[&IndividualId(2)];
        assert_eq!(s2.infection_time, 2.0);
        assert_eq!(s2.removal_time, None);
        assert_eq!(s2.deme, 1);
        // Never-removed individual falls back to sim_end.
        assert_eq!(s2.removal_or(sim_end), 4.0);
        // Seed (pure parent) exists as a root with infection_time 0.
        let s0 = &s[&IndividualId(0)];
        assert_eq!(s0.infection_time, 0.0);
        assert_eq!(s0.removal_time, None);
    }

    #[test]
    fn flat_all_individuals_can_sample_an_infector() {
        // Chain 0 -> 1 -> 2. Individual 1 is an infector (internal node).
        // With rate 1.0 over ALL individuals, 1 must be in the sampled set —
        // impossible under the old leaf-only scheme.
        let entries = vec![lineage_entry(1.0, 1, 0, 1), lineage_entry(2.0, 2, 1, 1)];
        let (summaries, sim_end) = summarize(&entries);
        let mut rng = StatefulRng::new(1);
        let sampled = select_samples(&Flat::new(1.0, sim_end), &summaries, &mut rng);
        assert!(sampled.contains_key(&IndividualId(1)), "infector must be sampleable");

        let f = TransmissionForest::from_entries(&entries);
        // Confirm 1 is genuinely an infector (has a child) in the full forest.
        assert!(!f.nodes[&IndividualId(1)].children.is_empty());

        // In the pruned tree, sampling {1, 2} makes 1 an internal node with a
        // pendant tip plus its child subtree (tip for 2).
        let sel = sampled_at(&[1, 2], 5.0);
        let trees = f.prune_to(&sel);
        assert_eq!(trees.len(), 1);
        let t = &trees[0];
        // Tips equal the sampled set {1, 2}.
        let mut tip_ids = collect_tip_ids(t);
        tip_ids.sort();
        assert_eq!(tip_ids, vec![1, 2], "pruned tips must equal the sampled set");
    }

    #[test]
    fn pendant_tip_is_placed_at_sampling_time() {
        // 0 -> 1 (infected t=1), 1 recovers (removed) at t=6. Sample {1}.
        // Its tip branch must reach the removal time (6), not the infection
        // time (1).
        let entries = vec![lineage_entry(1.0, 1, 0, 1), removal_entry(6.0, 1, 1, Some(2))];
        let f = TransmissionForest::from_entries(&entries);
        let (summaries, sim_end) = summarize(&entries);
        let t_sample = summaries[&IndividualId(1)].removal_or(sim_end);
        assert_eq!(t_sample, 6.0);

        let sel: HashMap<IndividualId, f64> = [(IndividualId(1), t_sample)].into_iter().collect();
        let trees = f.prune_to(&sel);
        assert_eq!(trees.len(), 1);
        // The single tip's node_time is the sampling (removal) time.
        let tip = find_tip(&trees[0], IndividualId(1)).expect("tip 1 present");
        assert_eq!(tip.node_time, 6.0, "tip placed at sampling time, not infection time");
    }

    #[test]
    fn stratified_rates_skew_tip_composition() {
        // Two demes: 100 individuals each. Deme 0 sampled at 0.5, deme 1 at
        // 0.05. Over many seeds, observed per-deme frequencies match the rates
        // and the tip stratum is skewed toward deme 0.
        let mut entries = Vec::new();
        // All infected by a single seed (0), so they are all candidate tips.
        for i in 1..=100u64 {
            entries.push(lineage_entry_deme(1.0, i, 0, 1, 0)); // deme 0
        }
        for i in 101..=200u64 {
            entries.push(lineage_entry_deme(1.0, i, 0, 1, 1)); // deme 1
        }
        let (summaries, sim_end) = summarize(&entries);
        let rates: HashMap<DemeId, f64> = [(0u32, 0.5), (1u32, 0.05)].into_iter().collect();
        let scheme = Stratified::new(rates, 0.0, sim_end);

        let trials = 400;
        let mut sum_d0 = 0usize;
        let mut sum_d1 = 0usize;
        for seed in 0..trials {
            let mut rng = StatefulRng::new(seed);
            let sampled = select_samples(&scheme, &summaries, &mut rng);
            for id in sampled.keys() {
                if summaries[id].deme == 0 {
                    sum_d0 += 1;
                } else {
                    sum_d1 += 1;
                }
            }
        }
        // Observed per-deme frequency = sampled / (trials * pop_per_deme).
        let n_per_deme = 100.0 * trials as f64;
        let freq_d0 = sum_d0 as f64 / n_per_deme;
        let freq_d1 = sum_d1 as f64 / n_per_deme;
        // 3σ MC tolerance: sigma = sqrt(p(1-p)/N).
        let sigma = |p: f64| (p * (1.0 - p) / n_per_deme).sqrt();
        assert!(
            (freq_d0 - 0.5).abs() < 3.0 * sigma(0.5),
            "deme-0 freq {} not within 3σ of 0.5",
            freq_d0
        );
        assert!(
            (freq_d1 - 0.05).abs() < 3.0 * sigma(0.05),
            "deme-1 freq {} not within 3σ of 0.05",
            freq_d1
        );
        // Composition skewed: ~10x more deme-0 tips than deme-1.
        assert!(sum_d0 > 5 * sum_d1, "tip composition must skew toward deme 0");
    }

    // ── helpers for tree-shape assertions ─────────────────────────────────

    fn collect_tip_ids(n: &PrunedNode) -> Vec<u64> {
        if n.children.is_empty() {
            vec![n.id.0]
        } else {
            n.children.iter().flat_map(collect_tip_ids).collect()
        }
    }

    fn find_tip<'a>(n: &'a PrunedNode, id: IndividualId) -> Option<&'a PrunedNode> {
        if n.children.is_empty() {
            return if n.id == id { Some(n) } else { None };
        }
        n.children.iter().find_map(|c| find_tip(c, id))
    }
}
