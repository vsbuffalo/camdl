//! A resumed sampler leaf holds the whole chain.
//!
//! `fit run --resume <base>` extends a PMMH or PGAS run into a leaf of its
//! own. That leaf is the complete artifact for the longer chain: each
//! `chain_N/` starts as a copy of the parent's per-chain files
//! ([`CARRIED_CHAIN_FILES`]) and the new segment is appended to them, so every
//! reader of the leaf — `fit summary`, R̂/ESS, `draws.tsv`, `fit predict`,
//! `fit compare` — sees the whole chain without following the lineage back.
//!
//! The lineage itself is recorded, not inferred: run.json `inputs.resume`
//! ([`ResumeLineage`]) names the parent and the index at which every segment
//! of the chain begins.

use std::path::Path;

use crate::chain_index::ChainIndex;

/// The per-chain files a resumed leaf copies from its parent before the new
/// segment runs. `resume_state.bin` is the chain state the sampler continues
/// from (and overwrites); `trace.tsv` (both methods) and `trajectories.tsv`
/// (PGAS, when it saved paths) are extended in place by the new segment's
/// rows. `trajectories.json` is not carried: it is rewritten whole.
pub(crate) const CARRIED_CHAIN_FILES: [&str; 3] = ["resume_state.bin", "trace.tsv", "trajectories.tsv"];

/// Copy the parent leaf's [`CARRIED_CHAIN_FILES`] into the resumed leaf, chain
/// by chain. The parent is read only.
///
/// `std::fs::copy` clones the file on APFS (`fclonefileat`, copy-on-write: the
/// copy shares the parent's blocks until the append writes new ones), and on
/// Linux goes through `copy_file_range`, which reflinks on filesystems that
/// support it; elsewhere it is a plain copy.
pub(crate) fn copy_parent_chains(parent: &Path, leaf: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if ChainIndex::parse_dir_name(&name).is_none() || !entry.path().is_dir() {
            continue;
        }
        let dst = leaf.join(name.as_ref());
        std::fs::create_dir_all(&dst)?;
        for f in CARRIED_CHAIN_FILES {
            let src = entry.path().join(f);
            if src.is_file() {
                std::fs::copy(&src, dst.join(f))?;
            }
        }
    }
    Ok(())
}

/// One chain's retained posterior draws that came from the parent: the rows of
/// the chain's `trace.tsv` below the resume boundary that the burn-in and
/// thinning rule keeps.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InheritedDraws {
    /// The step (PMMH) or sweep (PGAS) index of each retained row.
    pub index: Vec<usize>,
    /// `values[r][p]`: estimated parameter `p` (in the order asked for) at row `r`.
    pub values: Vec<Vec<f64>>,
}

impl InheritedDraws {
    /// The values of parameter `p` down the inherited rows.
    pub fn column(&self, p: usize) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().map(move |row| row[p])
    }
}

/// Whether the sampler keeps the draw at `index`: the rule both samplers apply
/// when they record a step (`index >= burn_in` and a multiple of `thin` past
/// it; a `thin` of 0 keeps only the draw at `burn_in`).
pub(crate) fn is_retained(index: usize, burn_in: usize, thin: usize) -> bool {
    index >= burn_in && (index - burn_in).is_multiple_of(thin)
}

/// Read the retained rows below `boundary` from a resumed chain's `trace.tsv`,
/// which by then holds the parent's rows followed by the new segment's.
///
/// The whole file is checked on the way: the index column must increase
/// strictly from the first row to the last, so a boundary row written twice,
/// or a segment whose numbering restarted, is refused rather than read as more
/// draws. Parameter values are the trace's shortest round-trip rendering, so
/// they parse back to the exact values the parent sampled.
pub(crate) fn inherited_draws(
    trace: &Path,
    index_col: &str,
    boundary: usize,
    burn_in: usize,
    thin: usize,
    names: &[String],
) -> Result<InheritedDraws, String> {
    let text = std::fs::read_to_string(trace)
        .map_err(|e| format!("cannot read {}: {e}", trace.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let col = |name: &str| {
        header.iter().position(|c| *c == name)
            .ok_or_else(|| format!("{}: no `{name}` column", trace.display()))
    };
    let index_at = col(index_col)?;
    let param_at: Vec<usize> = names.iter().map(|n| col(n)).collect::<Result<_, _>>()?;
    let mut out = InheritedDraws { index: Vec::new(), values: Vec::new() };
    let mut previous: Option<usize> = None;
    for (n, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let row = n + 2;
        let index: usize = fields.get(index_at).and_then(|s| s.parse().ok()).ok_or_else(|| {
            format!("{}: row {row} has no `{index_col}` index", trace.display())
        })?;
        if let Some(p) = previous {
            if index <= p {
                return Err(format!(
                    "{}: `{index_col}` {index} at row {row} follows {p}; a resumed chain's \
                     index must increase strictly across the parent's rows and the new ones",
                    trace.display()));
            }
        }
        previous = Some(index);
        if index >= boundary || !is_retained(index, burn_in, thin) {
            continue;
        }
        let values = param_at.iter().zip(names)
            .map(|(&i, name)| fields.get(i).and_then(|s| s.parse::<f64>().ok()).ok_or_else(|| {
                format!("{}: row {row} has no value for `{name}`", trace.display())
            }))
            .collect::<Result<Vec<f64>, String>>()?;
        out.index.push(index);
        out.values.push(values);
    }
    Ok(out)
}

/// The sweeps below `boundary` a resumed PGAS chain's `trajectories.tsv` holds
/// a saved path for (its `draw` column), in file order, each once. Empty when
/// the parent saved no paths for the chain.
pub(crate) fn inherited_saved_sweeps(trajectories: &Path, boundary: usize) -> Result<Vec<usize>, String> {
    if !trajectories.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(trajectories)
        .map_err(|e| format!("cannot read {}: {e}", trajectories.display()))?;
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let draw_at = header.iter().position(|c| *c == "draw")
        .ok_or_else(|| format!("{}: no `draw` column", trajectories.display()))?;
    let mut sweeps: Vec<usize> = Vec::new();
    for line in lines {
        let Some(draw) = line.split('\t').nth(draw_at).and_then(|s| s.parse::<usize>().ok()) else {
            return Err(format!("{}: a row has no `draw` sweep", trajectories.display()));
        };
        if draw < boundary && sweeps.last() != Some(&draw) {
            sweeps.push(draw);
        }
    }
    Ok(sweeps)
}

/// run.json `inputs.resume` of a resumed leaf: where the chain it holds came
/// from.
///
/// `segments` lists every run that contributed rows to the chain, oldest
/// first, ending with this leaf's own run. A row of chain `c` with index `i`
/// (the `index_column` of `chain_c/trace.tsv`, the `draw` of `trajectories.tsv`,
/// and the sweep in PGAS's `draws.tsv`) belongs to the last segment whose
/// `first_index[c] <= i`. The last segment's `first_index` is therefore the
/// resume boundary: the index at which this run's rows begin, one past the
/// parent's last. Copied forward from the parent's own record when it was
/// itself resumed, so no reader walks back through the ancestors.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ResumeLineage {
    /// The `run_id` of the leaf this run resumed (`--resume <base>`).
    pub parent_run_id: String,
    /// `step` (PMMH) or `sweep` (PGAS): the column the indices count.
    pub index_column: String,
    pub segments: Vec<Segment>,
}

/// One run's contribution to a resumed chain.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Segment {
    /// The run that sampled these rows.
    pub run_id: String,
    /// Per chain, indexed by chain number: the index of the segment's first
    /// row.
    pub first_index: Vec<usize>,
    /// The run's method wall time (its run.json `inputs.wall_time_seconds`);
    /// `null` when that record has none.
    pub wall_time_seconds: Option<f64>,
}

impl ResumeLineage {
    /// The lineage of a run that resumed `parent` and began each chain at
    /// `boundary[c]`: the parent's own segments when it was resumed, else the
    /// parent as one segment from index 0, then this run.
    pub fn extend(
        parent: &Path,
        index_column: &str,
        run_id: &str,
        boundary: Vec<usize>,
        wall_time_seconds: f64,
    ) -> Result<Self, String> {
        let rec = crate::leaf_use::read(parent)?
            .ok_or_else(|| format!("{} holds no run.json", parent.display()))?
            .record;
        let parent_run_id = rec.run_id.to_hex();
        let mut segments = match Self::read(&rec.inputs)? {
            Some(l) => l.segments,
            None => vec![Segment {
                run_id: parent_run_id.clone(),
                first_index: vec![0; boundary.len()],
                wall_time_seconds: rec.inputs.get("wall_time_seconds").and_then(|v| v.as_f64()),
            }],
        };
        segments.push(Segment {
            run_id: run_id.to_string(),
            first_index: boundary,
            wall_time_seconds: Some(wall_time_seconds),
        });
        Ok(ResumeLineage { parent_run_id, index_column: index_column.to_string(), segments })
    }

    /// The lineage a run.json `inputs` records, `None` for a run that did not
    /// resume.
    pub fn read(inputs: &serde_json::Value) -> Result<Option<Self>, String> {
        match inputs.get(RESUME_KEY) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => serde_json::from_value(v.clone())
                .map(Some)
                .map_err(|e| format!("run.json inputs.{RESUME_KEY}: {e}")),
        }
    }

    /// The whole chain's method wall time: the sum over its segments, `None`
    /// when any segment's is unknown.
    pub fn wall_time_seconds(&self) -> Option<f64> {
        self.segments.iter().map(|s| s.wall_time_seconds).sum()
    }
}

/// The run.json `inputs` key [`ResumeLineage`] is recorded under.
pub(crate) const RESUME_KEY: &str = "resume";

/// The method wall time behind every draw a leaf holds — the denominator of
/// ESS per second. A fresh run's `inputs.wall_time_seconds`; for a resumed
/// leaf, whose draws span every segment, the segments' total (`None` when one
/// is unknown, rather than one segment's time set against the whole chain's
/// ESS).
pub(crate) fn chain_wall_time_seconds(inputs: &serde_json::Value) -> Option<f64> {
    match ResumeLineage::read(inputs) {
        Ok(Some(l)) => l.wall_time_seconds(),
        Ok(None) => inputs.get("wall_time_seconds").and_then(|v| v.as_f64()),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let d = crate::test_support::unique_temp_dir(tag);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn inherited_draws_keeps_the_retained_rows_below_the_boundary() {
        let d = tmp("inherited_draws_keeps");
        let p = d.join("trace.tsv");
        // Warm-up rows every step (0..4), then the thinned sampling rows, then
        // the new segment from 8.
        std::fs::write(&p, "step\tlog_likelihood\tlog_posterior\taccepted\tbeta\n\
            0\t-1\t-1\t1\t0.1\n1\t-1\t-1\t1\t0.2\n2\t-1\t-1\t1\t0.3\n3\t-1\t-1\t1\t0.4\n\
            4\t-1\t-1\t1\t0.5\n6\t-1\t-1\t1\t0.6000000000000001\n8\t-1\t-1\t1\t0.7\n").unwrap();
        let got = inherited_draws(&p, "step", 8, 4, 2, &["beta".into()]).unwrap();
        assert_eq!(got.index, vec![4, 6]);
        assert_eq!(got.column(0).collect::<Vec<_>>(), vec![0.5, 0.6000000000000001],
            "values parse back exactly");
    }

    #[test]
    fn inherited_draws_refuses_a_repeated_or_restarted_index() {
        let d = tmp("inherited_draws_refuses");
        let p = d.join("trace.tsv");
        for body in ["0\t0.1\n1\t0.2\n1\t0.2\n", "0\t0.1\n1\t0.2\n0\t0.3\n"] {
            std::fs::write(&p, format!("sweep\tbeta\n{body}")).unwrap();
            let e = inherited_draws(&p, "sweep", 1, 0, 1, &["beta".into()]).unwrap_err();
            assert!(e.contains("increase strictly"), "{e}");
        }
    }

    #[test]
    fn inherited_saved_sweeps_reads_each_draw_below_the_boundary_once() {
        let d = tmp("inherited_saved_sweeps");
        let p = d.join("trajectories.tsv");
        std::fs::write(&p, "# camdl-trajectories v1\nchain\tdraw\ttime\n\
            0\t4\t0\n0\t4\t1\n0\t9\t0\n0\t9\t1\n0\t14\t0\n").unwrap();
        assert_eq!(inherited_saved_sweeps(&p, 12).unwrap(), vec![4, 9]);
        assert!(inherited_saved_sweeps(&d.join("absent.tsv"), 12).unwrap().is_empty());
    }

    #[test]
    fn the_chain_wall_time_sums_every_segment() {
        let seg = |w: Option<f64>| Segment { run_id: "r".into(), first_index: vec![0], wall_time_seconds: w };
        let l = ResumeLineage {
            parent_run_id: "p".into(), index_column: "step".into(),
            segments: vec![seg(Some(2.0)), seg(Some(3.0))],
        };
        let inputs = serde_json::json!({ "wall_time_seconds": 3.0, RESUME_KEY: l });
        assert_eq!(chain_wall_time_seconds(&inputs), Some(5.0));
        let l = ResumeLineage { segments: vec![seg(None), seg(Some(3.0))], ..l };
        let inputs = serde_json::json!({ "wall_time_seconds": 3.0, RESUME_KEY: l });
        assert_eq!(chain_wall_time_seconds(&inputs), None, "an unknown segment leaves it unknown");
        assert_eq!(chain_wall_time_seconds(&serde_json::json!({ "wall_time_seconds": 4.0 })), Some(4.0));
    }
}
