//! The filtered-state file — one writer, one reader, one format (gh#641).
//!
//! `camdl pfilter --save-final-state PATH` writes the bootstrap filter's
//! post-resampling particle states after the last observation: an unweighted
//! sample from `p(x_T | y_{1:T})` at the filter's θ. `camdl simulate
//! --init-state PATH` reads it back and starts a forward run from those states
//! instead of the model's `init {}` block — the forecast-from-filtered-state
//! workflow.
//!
//! Writer and reader live in one module so the header, the column layout, and
//! the "which columns are state" rule cannot drift between them. The column
//! layout itself is described by [`TrajColumnSpec`], shared with
//! `trajectories.tsv`, so the two state-bearing formats cannot disagree about
//! which compartments a model has or what order they are in.
//!
//! ## The format
//!
//! ```text
//! # camdl-final-state v1	t=196
//! particle	S	E	I	R
//! 0	9812	14	31	143
//! 1	9805	19	28	148
//! ```
//!
//! - The header line records **`t`, the model time the states are at** (the
//!   filter's last observation time). Without it a reader cannot know where the
//!   forecast origin is, and a separately-typed `--from` flag could silently
//!   disagree with the file it accompanies. A file lacking the header is
//!   refused — there is no headerless fall-back (alpha posture).
//! - One row per particle. `particle` is the row index (provenance; the reader
//!   keys on row order, not on this value).
//! - One column per **integer** compartment, in model order. Nothing else.
//!
//! ## Why compartments and nothing else
//!
//! **No flow accumulators.** An earlier form of this file carried a
//! `flow_<transition>` column per transition. Every one of them is structurally
//! zero: the filter's observation loop calls `reset_flows()` on every particle
//! after the final resample (`particle_filter.rs`), so the saved swarm's
//! accumulators are always cleared. That put columns of zeros in the file and in
//! its content digest, and invited a reader to take `flow_infection = 0` for a
//! measurement of no infections.
//!
//! Nothing is lost, because a forecast could not have used them: it opens a
//! fresh accumulation window at the origin, and the gh#322 resume seam emits
//! that first row with zeroed flows by construction.
//!
//! **No real compartments.** The particle filter's `ParticleState` carries
//! integer counts only, so a model with a real-valued compartment cannot be
//! represented here at all — the reader refuses such a model by name rather
//! than defaulting the reservoir to zero.

use std::io::Write;
use std::path::Path;

use crate::trajectories::TrajColumnSpec;

/// The header line's format tag. A file whose first line does not start with
/// this is refused — including a file written before the header existed.
pub const FORMAT_TAG: &str = "# camdl-final-state v1";

/// The leading row-index column's name.
const PARTICLE_COL: &str = "particle";

/// Prefix of the per-transition flow-accumulator columns this format used to
/// carry. No longer written or accepted (they were structurally zero — see the
/// module doc); kept only to give a file that still has them a diagnostic that
/// names the cause instead of "unknown compartment".
const RETIRED_FLOW_PREFIX: &str = "flow_";

/// A parsed filtered-state file, already permuted into the model's integer
/// compartment order.
///
/// Parsed at the boundary, not validated downstream: constructing this proves
/// that every one of the model's integer compartments has a column, that no
/// foreign compartment column is present, and that every row parsed as `i64`
/// at every compartment column. A consumer indexes `counts[particle]` exactly
/// like `sim::IntState::counts`.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalStates {
    /// The forecast origin: the model time these states are at, from the file
    /// header. A `simulate --init-state` run uses it as `simulation.t_start`.
    pub origin_t: f64,
    /// `counts[particle][local_int_index]`, in the model's integer-compartment
    /// order.
    pub counts: Vec<Vec<i64>>,
}

impl FinalStates {
    /// Number of particle rows — the number of forecast replicates this file
    /// can seed.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
}

/// Write the filtered particle states at `origin_t` (the filter's last
/// observation time).
///
/// Columns: `particle`, then every integer compartment in model order.
/// `states[i].counts` is in that same order (the filter's `ParticleState`
/// mirrors `IntState`), so the emit is positional against the header this
/// function itself writes. A row whose width disagrees with the header is a
/// hard error — never a silent truncate or zero-fill.
///
/// Flow accumulators are deliberately not written: the filter clears them on
/// every particle after the final resample, so they would be columns of zeros
/// (module doc, "Why compartments and nothing else").
pub fn write_final_states(
    path: &Path,
    states: &[sim::inference::ParticleState],
    columns: &TrajColumnSpec,
    origin_t: f64,
) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path.display(), e))?;
    let mut w = std::io::BufWriter::new(f);
    let io_err = |e: std::io::Error| format!("cannot write {}: {}", path.display(), e);

    writeln!(w, "{}\tt={}", FORMAT_TAG, origin_t).map_err(io_err)?;

    write!(w, "{}", PARTICLE_COL).map_err(io_err)?;
    for c in &columns.int_comps {
        write!(w, "\t{}", c).map_err(io_err)?;
    }
    writeln!(w).map_err(io_err)?;

    for (i, state) in states.iter().enumerate() {
        if state.counts.len() != columns.int_comps.len() {
            return Err(format!(
                "final-state writer: particle {} has {} counts, but the header \
                 declares {} compartments",
                i,
                state.counts.len(),
                columns.int_comps.len()
            ));
        }
        write!(w, "{}", i).map_err(io_err)?;
        for &c in &state.counts {
            write!(w, "\t{}", c).map_err(io_err)?;
        }
        writeln!(w).map_err(io_err)?;
    }

    // Explicit flush: BufWriter swallows write errors on drop, which would
    // silently truncate the file if the disk filled during the final drain.
    w.flush().map_err(io_err)?;
    Ok(())
}

/// Read a filtered-state file and permute it into the model's integer
/// compartment order (`columns.int_comps`).
///
/// Structural checks, all hard errors — a state file that does not describe
/// *this* model's compartments would otherwise seed a forecast with counts
/// filed under the wrong names, which no downstream check could catch:
///
/// - the header must be present and carry a finite `t`;
/// - a model carrying any **real** compartment is refused by name: the particle
///   filter has no real state to save, so the file cannot restore one;
/// - every integer compartment must have a column, resolved **by name**, never
///   positionally — a missing one errors instead of zero-filling;
/// - any column that is not `particle` and does not name a model integer
///   compartment is refused (the file came from a different model, or from a
///   camdl that still wrote the retired `flow_*` columns);
/// - every row must parse as `i64` at every compartment column.
pub fn read_final_states(path: &Path, columns: &TrajColumnSpec) -> Result<FinalStates, String> {
    if !columns.real_comps.is_empty() {
        return Err(format!(
            "this model has real-valued compartment(s) [{}], which a filtered-state \
             file cannot carry: the particle filter's state is integer counts only, \
             so the reservoir's value at the forecast origin is recorded nowhere. \
             Refusing rather than restarting the reservoir from `init {{}}` while the \
             counts come from the filter.",
            columns.real_comps.join(", ")
        ));
    }

    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let mut lines = txt.lines();

    let header = lines.next().unwrap_or_default();
    if !header.starts_with(FORMAT_TAG) {
        return Err(format!(
            "{} is not a camdl filtered-state file (its first line does not start \
             `{}`).\n  Write one with: camdl pfilter <model> --data <data> --params \
             <params> --save-final-state {}",
            path.display(),
            FORMAT_TAG,
            path.display(),
        ));
    }
    let origin_t = parse_origin_t(header, path)?;

    let col_header = lines.next().ok_or_else(|| {
        format!("{}: filtered-state file has a header but no columns", path.display())
    })?;
    let cols: Vec<&str> = col_header.split('\t').map(str::trim).collect();

    // Resolve every model integer compartment to a column BY NAME.
    let mut idx: Vec<usize> = Vec::with_capacity(columns.int_comps.len());
    for name in &columns.int_comps {
        let i = cols.iter().position(|c| c == name).ok_or_else(|| {
            format!(
                "{}: filtered-state file has no column for compartment `{}`. Its \
                 columns are [{}]. The file was written for a different model — \
                 regenerate it from the model being simulated.",
                path.display(),
                name,
                cols.join(", ")
            )
        })?;
        idx.push(i);
    }
    // Any other column names something this model does not have. The retired
    // `flow_*` columns get their own diagnostic: "not a compartment" would be
    // true but useless to someone holding a file an older camdl wrote.
    for c in &cols {
        if *c == PARTICLE_COL || columns.int_comps.iter().any(|n| n == c) {
            continue;
        }
        if c.starts_with(RETIRED_FLOW_PREFIX) {
            return Err(format!(
                "{}: filtered-state file carries `{}` and other flow-accumulator \
                 columns, which this format no longer has — they were always zero \
                 (the filter clears them after the final resample), so they were \
                 removed rather than left to look like a measurement.\n  \
                 Fix: regenerate the file with a current camdl \
                 (`camdl pfilter ... --save-final-state`).",
                path.display(),
                c
            ));
        }
        return Err(format!(
            "{}: filtered-state file has column `{}`, which is not a compartment of \
             this model. The file was written for a different model — regenerate it \
             from the model being simulated.",
            path.display(),
            c
        ));
    }

    let mut counts: Vec<Vec<i64>> = Vec::new();
    for (line_no, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let mut row = Vec::with_capacity(idx.len());
        for (k, &i) in idx.iter().enumerate() {
            let v = f
                .get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    format!(
                        "{}: row {} has {} fields, but compartment `{}` is at column {}",
                        path.display(),
                        line_no + 1,
                        f.len(),
                        columns.int_comps[k],
                        i
                    )
                })?;
            let n: i64 = v.parse().map_err(|_| {
                format!(
                    "{}: row {}, compartment `{}`: `{}` is not an integer count",
                    path.display(),
                    line_no + 1,
                    columns.int_comps[k],
                    v
                )
            })?;
            row.push(n);
        }
        counts.push(row);
    }

    if counts.is_empty() {
        return Err(format!(
            "{}: filtered-state file has no particle rows",
            path.display()
        ));
    }
    Ok(FinalStates { origin_t, counts })
}

/// Pull `t=<value>` out of the header line.
fn parse_origin_t(header: &str, path: &Path) -> Result<f64, String> {
    let field = header
        .split('\t')
        .map(str::trim)
        .find_map(|f| f.strip_prefix("t="))
        .ok_or_else(|| {
            format!(
                "{}: filtered-state header carries no `t=<time>` field, so the forecast \
                 origin is unknown. Regenerate the file with a current camdl.",
                path.display()
            )
        })?;
    let t: f64 = field.trim().parse().map_err(|_| {
        format!(
            "{}: filtered-state header has `t={}`, which is not a number",
            path.display(),
            field
        )
    })?;
    if !t.is_finite() {
        return Err(format!(
            "{}: filtered-state header has a non-finite origin time `t={}`",
            path.display(),
            field
        ));
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::ParticleState;

    fn cols(real: &[&str]) -> TrajColumnSpec {
        TrajColumnSpec {
            int_comps: vec!["S".into(), "I".into(), "R".into()],
            real_comps: real.iter().map(|s| s.to_string()).collect(),
            flows: vec!["flow_infection".into(), "flow_recovery".into()],
            incidence: vec![],
        }
    }

    fn states() -> Vec<ParticleState> {
        let mk = |c: Vec<i64>, f: Vec<u64>| ParticleState {
            counts: c,
            flow_accumulators: f,
            acc: vec![],
        };
        vec![mk(vec![90, 5, 5], vec![7, 3]), mk(vec![80, 11, 9], vec![13, 6])]
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("camdl_final_state_{}_{}", tag, std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    #[test]
    fn round_trips_counts_and_origin() {
        let d = tmpdir("rt");
        let p = d.join("final.tsv");
        write_final_states(&p, &states(), &cols(&[]), 196.0).unwrap();

        let text = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with(FORMAT_TAG), "got: {}", lines[0]);
        assert!(lines[0].contains("t=196"), "got: {}", lines[0]);
        assert_eq!(lines[1], "particle\tS\tI\tR", "compartments only — no flow_* columns");
        assert_eq!(lines[2], "0\t90\t5\t5");

        let fs = read_final_states(&p, &cols(&[])).unwrap();
        assert_eq!(fs.origin_t, 196.0);
        assert_eq!(fs.counts, vec![vec![90, 5, 5], vec![80, 11, 9]]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A fractional origin must survive the text round-trip bit-exactly — an
    /// origin off by one float ULP misses the output-emit grid the resume seam
    /// requires it to land on.
    #[test]
    fn fractional_origin_round_trips_exactly() {
        let d = tmpdir("frac");
        let p = d.join("final.tsv");
        let t = 0.1 + 0.2; // 0.30000000000000004
        write_final_states(&p, &states(), &cols(&[]), t).unwrap();
        let fs = read_final_states(&p, &cols(&[])).unwrap();
        assert_eq!(fs.origin_t.to_bits(), t.to_bits());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Columns resolve BY NAME: a file whose compartment columns are permuted
    /// relative to the model still restores each count under its own name.
    #[test]
    fn columns_resolve_by_name_not_position() {
        let d = tmpdir("perm");
        let p = d.join("final.tsv");
        std::fs::write(
            &p,
            format!("{FORMAT_TAG}\tt=10\nparticle\tR\tS\tI\n0\t5\t90\t7\n"),
        )
        .unwrap();
        let fs = read_final_states(&p, &cols(&[])).unwrap();
        // Model order is S, I, R — not the file's R, S, I.
        assert_eq!(fs.counts, vec![vec![90, 7, 5]]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_header_is_refused() {
        let d = tmpdir("nohdr");
        let p = d.join("final.tsv");
        std::fs::write(&p, "particle\tS\tI\tR\n0\t90\t5\t5\n").unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("not a camdl filtered-state file"), "got: {err}");
        assert!(err.contains("--save-final-state"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn header_without_time_is_refused() {
        let d = tmpdir("not");
        let p = d.join("final.tsv");
        std::fs::write(&p, format!("{FORMAT_TAG}\nparticle\tS\tI\tR\n0\t90\t5\t5\n")).unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("no `t=<time>` field"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn missing_compartment_column_is_refused_never_zero_filled() {
        let d = tmpdir("miss");
        let p = d.join("final.tsv");
        std::fs::write(&p, format!("{FORMAT_TAG}\tt=10\nparticle\tS\tI\n0\t90\t5\n")).unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("no column for compartment `R`"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn foreign_compartment_column_is_refused() {
        let d = tmpdir("foreign");
        let p = d.join("final.tsv");
        std::fs::write(
            &p,
            format!("{FORMAT_TAG}\tt=10\nparticle\tS\tI\tR\tE\n0\t90\t5\t5\t1\n"),
        )
        .unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("column `E`"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A real-valued compartment cannot be restored from a file the particle
    /// filter wrote — refused by name, never silently defaulted.
    #[test]
    fn real_compartment_model_is_refused_by_name() {
        let d = tmpdir("real");
        let p = d.join("final.tsv");
        write_final_states(&p, &states(), &cols(&[]), 10.0).unwrap();
        let err = read_final_states(&p, &cols(&["P"])).unwrap_err();
        assert!(err.contains("real-valued compartment(s) [P]"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn non_integer_count_is_refused() {
        let d = tmpdir("nonint");
        let p = d.join("final.tsv");
        std::fs::write(
            &p,
            format!("{FORMAT_TAG}\tt=10\nparticle\tS\tI\tR\n0\t90.5\t5\t5\n"),
        )
        .unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("not an integer count"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn empty_body_is_refused() {
        let d = tmpdir("empty");
        let p = d.join("final.tsv");
        std::fs::write(&p, format!("{FORMAT_TAG}\tt=10\nparticle\tS\tI\tR\n")).unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("no particle rows"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A file from a camdl that still wrote `flow_*` columns must be refused
    /// with a diagnostic naming the cause — "not a compartment" would be true
    /// and useless.
    #[test]
    fn a_file_with_the_retired_flow_columns_names_the_cause() {
        let d = tmpdir("retired");
        let p = d.join("final.tsv");
        std::fs::write(
            &p,
            format!(
                "{FORMAT_TAG}\tt=10\nparticle\tS\tI\tR\tflow_infection\n0\t90\t5\t5\t7\n"
            ),
        )
        .unwrap();
        let err = read_final_states(&p, &cols(&[])).unwrap_err();
        assert!(err.contains("flow-accumulator columns"), "got: {err}");
        assert!(err.contains("always zero"), "got: {err}");
        assert!(err.contains("--save-final-state"), "the fix must be spelled out: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn writer_refuses_a_state_whose_width_disagrees_with_the_header() {
        let d = tmpdir("width");
        let p = d.join("final.tsv");
        let bad = vec![ParticleState {
            counts: vec![90, 5],
            flow_accumulators: vec![7, 3],
            acc: vec![],
        }];
        let err = write_final_states(&p, &bad, &cols(&[]), 1.0).unwrap_err();
        assert!(err.contains("2 counts"), "got: {err}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
