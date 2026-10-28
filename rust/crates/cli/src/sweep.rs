//! The design axis of `camdl simulate`: `--sweep PARAM=GRID` and
//! `--sweep-file FILE` expanded into design points, the resolved design that
//! is recorded with the run, and the checks the flags share with
//! `fit predict --sweep` (proposal 2026-10-24-batch-to-simulate §3).
//!
//! One expansion serves both verbs, so `simulate --sweep` and
//! `fit predict --sweep` produce the same `point_id` for the same flags: the
//! Cartesian product of the swept parameters **sorted by name**, the first
//! name varying slowest, `point_id` being a point's 0-based position in it.
//! A design coordinate is a *user-authored* parameter value (the
//! `ParamSource::Sweep` variant), never a draw: it labels its leaf, it never
//! enters the process seed (decision D-B of the coordinate-kind proposal), and
//! a scenario or a `--param` that touches a swept parameter is refused rather
//! than silently winning over every point.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;

use crate::args::types::SweepSpec;
use crate::sim_ensemble_cas::{ResolvedDesign, SweptParam};

/// One design point: the swept parameters and their values at this point,
/// sorted by parameter name.
pub type DesignPoint = Vec<(String, f64)>;

/// Expand `--sweep` specs into design points: the Cartesian product of the
/// specs sorted by parameter name, the first name varying slowest, each point
/// a sorted-by-name `(param, value)` list. No specs give a single empty point,
/// so a caller with no sweep runs exactly once.
pub fn expand_sweep_specs(specs: &[SweepSpec]) -> Vec<DesignPoint> {
    if specs.is_empty() {
        return vec![Vec::new()];
    }
    let mut sorted: Vec<&SweepSpec> = specs.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut cells: Vec<DesignPoint> = vec![Vec::new()];
    for spec in sorted {
        let values = spec.grid.expand();
        let mut next = Vec::with_capacity(cells.len() * values.len());
        for cell in &cells {
            for &v in &values {
                let mut c = cell.clone();
                c.push((spec.name.clone(), v));
                next.push(c);
            }
        }
        cells = next;
    }
    cells
}

/// Refuse a sweep that names a parameter twice or one the model does not
/// declare. Either would vary a `sweep:<param>` column while the dynamics
/// never move (a silent no-op), so both are errors before anything runs.
pub fn check_swept_names(specs: &[SweepSpec], model: &ir::Model) -> Result<(), String> {
    let distinct: BTreeSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    if distinct.len() != specs.len() {
        return Err(
            "--sweep names the same parameter more than once; sweep each \
             parameter at most once (the grid is the Cartesian product of \
             distinct parameters)"
                .to_string(),
        );
    }
    let model_params: HashSet<&str> = model.parameters.iter().map(|p| p.name.as_str()).collect();
    let unknown: Vec<&str> = distinct.iter().copied().filter(|p| !model_params.contains(p)).collect();
    if !unknown.is_empty() {
        return Err(format!(
            "--sweep names parameter(s) the model does not declare: {} \
             (a sweep must vary a real model parameter){}",
            unknown.join(", "),
            crate::util::constant_hints(model, unknown.iter().copied())
        ));
    }
    Ok(())
}

/// The resolved design of `--sweep` specs: each swept parameter with its
/// expanded values, name-sorted, and the point count.
pub fn design_from_specs(specs: &[SweepSpec]) -> ResolvedDesign {
    let mut params: Vec<SweptParam> = specs
        .iter()
        .map(|s| SweptParam { name: s.name.clone(), values: s.grid.expand() })
        .collect();
    params.sort_by(|a, b| a.name.cmp(&b.name));
    let n_points = params.iter().map(|p| p.values.len()).product();
    ResolvedDesign::Sweep { params, n_points }
}

/// The `--sweep-file` header column that names the design points instead of
/// a parameter (decision D-C of the coordinate-kind proposal): its value on a
/// row is the `point_id` the row is written under.
pub const POINT_ID_COLUMN: &str = "point_id";

/// A `--sweep-file` read and checked: its design points in file order, each
/// with the id it is written under, and the resolved design to record (the
/// header's parameters, the file digest and the ids).
#[derive(Debug)]
pub struct SweepFile {
    /// `(point_id, point)` per row, in file order. The id is the row's own
    /// `point_id` cell when the file has that column, else its 0-based row;
    /// the row position stays the engine's index either way (the design never
    /// enters the seed, decision D-B). The point is the row's name-sorted
    /// values.
    pub points: Vec<(usize, DesignPoint)>,
    pub design: ResolvedDesign,
}

/// Read a `--sweep-file`: a TSV whose header names model parameters and whose
/// rows are design points. An optional `point_id` column gives each row the
/// id it is written under (in the combined and observation tables and in the
/// ensemble record); without it the id is the 0-based row. Refused: a column
/// that names no model parameter (with the parameter list), a file with zero
/// rows, a duplicate row (both line numbers named, decision D-E), a cell that
/// is not a number, and a `point_id` that is not a non-negative integer or
/// repeats an earlier row's. The file enters the design by its content
/// digest, never its path.
pub fn read_sweep_file(path: &Path, model: &ir::Model) -> Result<SweepFile, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read --sweep-file {}: {}", path.display(), e))?;
    let text = String::from_utf8(bytes.clone())
        .map_err(|_| format!("--sweep-file {} is not UTF-8 text", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| format!("--sweep-file {} is empty", path.display()))?;
    let columns: Vec<String> = header
        .split('\t')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    // `point_id` names the points, not a parameter: its column is read for
    // the ids and leaves the parameter list.
    let id_col: Option<usize> = columns.iter().position(|c| c == POINT_ID_COLUMN);
    let param_cols: Vec<(usize, &str)> = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| *c != POINT_ID_COLUMN)
        .map(|(i, c)| (i, c.as_str()))
        .collect();
    let params: Vec<String> = param_cols.iter().map(|(_, c)| c.to_string()).collect();
    if params.is_empty() {
        return Err(format!("--sweep-file {}: the header names no parameter", path.display()));
    }
    let model_params: Vec<&str> = model.parameters.iter().map(|p| p.name.as_str()).collect();
    let unknown: Vec<&str> =
        params.iter().map(|s| s.as_str()).filter(|p| !model_params.contains(p)).collect();
    if !unknown.is_empty() {
        return Err(format!(
            "--sweep-file {}: column(s) {} name no model parameter.\n  \
             The header must name model parameters (plus an optional `{}` column); \
             this model declares: {}.",
            path.display(),
            unknown.iter().map(|u| format!("`{u}`")).collect::<Vec<_>>().join(", "),
            POINT_ID_COLUMN,
            model_params.join(", ")
        ));
    }
    let distinct: BTreeSet<&str> = columns.iter().map(|s| s.as_str()).collect();
    if distinct.len() != columns.len() {
        return Err(format!(
            "--sweep-file {}: a column is named more than once",
            path.display()
        ));
    }

    let mut points: Vec<(usize, DesignPoint)> = Vec::new();
    // Line number (1-based, header is line 1) of each row seen so far, keyed
    // by the row's exact values, for the duplicate diagnostic.
    let mut seen: Vec<(Vec<u64>, usize)> = Vec::new();
    // Line number of each `point_id` seen so far, for the repeated-id
    // diagnostic.
    let mut seen_ids: Vec<(usize, usize)> = Vec::new();
    for (i, line) in lines.enumerate() {
        let line_no = i + 2;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
        if fields.len() < columns.len() {
            return Err(format!(
                "--sweep-file {} line {}: expected {} column(s), got {}",
                path.display(), line_no, columns.len(), fields.len()
            ));
        }
        let mut point: DesignPoint = Vec::with_capacity(params.len());
        for (col, name) in &param_cols {
            let field = fields[*col];
            let v: f64 = field.parse().map_err(|_| format!(
                "--sweep-file {} line {}, column `{}`: cannot parse '{}' as a number",
                path.display(), line_no, name, field
            ))?;
            if !v.is_finite() {
                return Err(format!(
                    "--sweep-file {} line {}, column `{}`: {} is not a finite value",
                    path.display(), line_no, name, field
                ));
            }
            point.push((name.to_string(), v));
        }
        point.sort_by(|a, b| a.0.cmp(&b.0));
        let key: Vec<u64> = point.iter().map(|(_, v)| v.to_bits()).collect();
        if let Some((_, first)) = seen.iter().find(|(k, _)| *k == key) {
            return Err(format!(
                "--sweep-file {}: line {} repeats the design point of line {} ({}).\n  \
                 Every row is one design point; a repeated row would run and label the \
                 same point twice.\n  Fix: remove one of the two rows.",
                path.display(),
                line_no,
                first,
                point.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", ")
            ));
        }
        seen.push((key, line_no));
        let point_id: usize = match id_col {
            Some(col) => {
                let field = fields[col];
                let id: usize = field.parse().map_err(|_| format!(
                    "--sweep-file {} line {}, column `{}`: '{}' is not a non-negative integer",
                    path.display(), line_no, POINT_ID_COLUMN, field
                ))?;
                if let Some((_, first)) = seen_ids.iter().find(|(seen_id, _)| *seen_id == id) {
                    return Err(format!(
                        "--sweep-file {}: line {} repeats point_id {} of line {}.\n  \
                         A point_id names one design point; two rows cannot share it.\n  \
                         Fix: give each row a distinct point_id, or drop the column to \
                         number the rows from 0.",
                        path.display(), line_no, id, first
                    ));
                }
                seen_ids.push((id, line_no));
                id
            }
            None => points.len(),
        };
        points.push((point_id, point));
    }
    if points.is_empty() {
        return Err(format!(
            "--sweep-file {} has a header but no design points (zero rows)",
            path.display()
        ));
    }
    let design = ResolvedDesign::File {
        params,
        digest: runid::ContentHash::digest_bytes(&bytes).to_hex(),
        n_points: points.len(),
        point_ids: points.iter().map(|(id, _)| *id).collect(),
    };
    Ok(SweepFile { points, design })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::types::Grid;

    #[test]
    fn expand_sweep_specs_empty_is_one_null_cell_and_cartesian_is_name_sorted() {
        // No specs → exactly one empty cell (the no-sweep single-pass path).
        assert_eq!(expand_sweep_specs(&[]), vec![Vec::<(String, f64)>::new()]);
        // Two params → Cartesian product, each cell sorted by param name, the
        // first name varying slowest — whatever order the flags were given in.
        let rho_then_k = vec![
            SweepSpec { name: "rho".to_string(), grid: Grid::List(vec![0.3, 0.5]) },
            SweepSpec { name: "k".to_string(), grid: Grid::List(vec![8.0, 12.0]) },
        ];
        let k_then_rho = vec![rho_then_k[1].clone(), rho_then_k[0].clone()];
        let cells = expand_sweep_specs(&rho_then_k);
        assert_eq!(
            cells,
            vec![
                vec![("k".to_string(), 8.0), ("rho".to_string(), 0.3)],
                vec![("k".to_string(), 8.0), ("rho".to_string(), 0.5)],
                vec![("k".to_string(), 12.0), ("rho".to_string(), 0.3)],
                vec![("k".to_string(), 12.0), ("rho".to_string(), 0.5)],
            ],
            "k sorts before rho and varies slowest; the grid is the Cartesian product"
        );
        assert_eq!(cells, expand_sweep_specs(&k_then_rho), "flag order does not change point_id");
    }

    #[test]
    fn design_from_specs_is_name_sorted_with_the_point_count() {
        let specs = vec![
            SweepSpec { name: "rho".to_string(), grid: Grid::List(vec![0.3, 0.5]) },
            SweepSpec { name: "k".to_string(), grid: Grid::Linear { min: 1.0, max: 3.0, n: 3 } },
        ];
        assert_eq!(
            design_from_specs(&specs),
            ResolvedDesign::Sweep {
                params: vec![
                    SweptParam { name: "k".to_string(), values: vec![1.0, 2.0, 3.0] },
                    SweptParam { name: "rho".to_string(), values: vec![0.3, 0.5] },
                ],
                n_points: 6,
            }
        );
    }

    fn model_with(params: &[&str]) -> ir::Model {
        let mut m = crate::test_support::tiny_model();
        m.parameters = params
            .iter()
            .map(|n| ir::parameter::Parameter {
                name: n.to_string(),
                value: ir::parameter::ParamValue::Fixed { value: 0.5 },
                param_kind: None,
                param_dim: None,
            })
            .collect();
        m
    }

    #[test]
    fn check_swept_names_refuses_a_repeat_and_an_unknown_name() {
        let m = model_with(&["beta", "gamma"]);
        let twice = vec![
            SweepSpec { name: "beta".to_string(), grid: Grid::List(vec![0.1]) },
            SweepSpec { name: "beta".to_string(), grid: Grid::List(vec![0.2]) },
        ];
        assert!(check_swept_names(&twice, &m).unwrap_err().contains("more than once"));
        let unknown = vec![SweepSpec { name: "delta".to_string(), grid: Grid::List(vec![0.1]) }];
        assert!(check_swept_names(&unknown, &m).unwrap_err().contains("delta"));
        let ok = vec![SweepSpec { name: "gamma".to_string(), grid: Grid::List(vec![0.1]) }];
        assert!(check_swept_names(&ok, &m).is_ok());
    }

    fn write(dir: &Path, name: &str, text: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn read_sweep_file_reads_rows_as_points_in_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let m = model_with(&["beta", "gamma"]);
        let p = write(dir.path(), "d.tsv", "gamma\tbeta\n0.1\t0.2\n0.1\t0.3\n");
        let f = read_sweep_file(&p, &m).unwrap();
        assert_eq!(
            f.points,
            vec![
                (0, vec![("beta".to_string(), 0.2), ("gamma".to_string(), 0.1)]),
                (1, vec![("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)]),
            ],
            "rows are points in file order, written under their 0-based row; each point's pairs are name-sorted"
        );
        match f.design {
            ResolvedDesign::File { params, digest, n_points, point_ids } => {
                assert_eq!(params, vec!["gamma", "beta"]);
                assert_eq!(n_points, 2);
                assert_eq!(point_ids, vec![0, 1]);
                assert_eq!(digest.len(), 64, "the file enters by content digest");
            }
            other => panic!("expected a file design, got {other:?}"),
        }
    }

    /// Decision D-C: a `point_id` column names the points instead of a
    /// parameter. Its values become the written ids, in the points and in the
    /// recorded design; the row position is unchanged.
    #[test]
    fn read_sweep_file_honours_a_point_id_column() {
        let dir = tempfile::tempdir().unwrap();
        let m = model_with(&["beta", "gamma"]);
        let p = write(dir.path(), "ids.tsv", "beta\tpoint_id\tgamma\n0.2\t3\t0.1\n0.3\t7\t0.1\n0.4\t12\t0.1\n");
        let f = read_sweep_file(&p, &m).unwrap();
        let ids: Vec<usize> = f.points.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![3, 7, 12], "the file's ids are the written ids, in row order");
        assert_eq!(
            f.points[1].1,
            vec![("beta".to_string(), 0.3), ("gamma".to_string(), 0.1)],
            "the `point_id` column is not a parameter"
        );
        match f.design {
            ResolvedDesign::File { params, point_ids, n_points, .. } => {
                assert_eq!(params, vec!["beta", "gamma"], "the design names the parameters only");
                assert_eq!(point_ids, vec![3, 7, 12], "and records the ids the points were written under");
                assert_eq!(n_points, 3);
            }
            other => panic!("expected a file design, got {other:?}"),
        }
        // Without the column the id is the 0-based row, as before.
        let p = write(dir.path(), "rows.tsv", "beta\n0.2\n0.3\n");
        let f = read_sweep_file(&p, &m).unwrap();
        assert_eq!(f.points.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![0, 1]);
        match f.design {
            ResolvedDesign::File { point_ids, .. } => assert_eq!(point_ids, vec![0, 1]),
            other => panic!("expected a file design, got {other:?}"),
        }
    }

    /// A repeated id, a non-integer id and a negative id are each refused
    /// naming the row; an id is a label, so two rows cannot share one.
    #[test]
    fn read_sweep_file_refuses_a_repeated_or_malformed_point_id() {
        let dir = tempfile::tempdir().unwrap();
        let m = model_with(&["beta", "gamma"]);
        let e = read_sweep_file(&write(dir.path(), "dup.tsv", "point_id\tbeta\n3\t0.2\n3\t0.3\n"), &m)
            .unwrap_err();
        assert!(e.contains("line 3 repeats point_id 3 of line 2"), "{e}");
        let e = read_sweep_file(&write(dir.path(), "frac.tsv", "point_id\tbeta\n1.5\t0.2\n"), &m)
            .unwrap_err();
        assert!(e.contains("line 2") && e.contains("'1.5'") && e.contains("non-negative integer"), "{e}");
        let e = read_sweep_file(&write(dir.path(), "neg.tsv", "point_id\tbeta\n-1\t0.2\n"), &m)
            .unwrap_err();
        assert!(e.contains("line 2") && e.contains("'-1'"), "{e}");
        // A file whose only column is `point_id` names no parameter.
        let e = read_sweep_file(&write(dir.path(), "only.tsv", "point_id\n0\n"), &m).unwrap_err();
        assert!(e.contains("names no parameter"), "{e}");
    }

    #[test]
    fn read_sweep_file_refuses_unknown_column_zero_rows_and_duplicate_row() {
        let dir = tempfile::tempdir().unwrap();
        let m = model_with(&["beta", "gamma"]);
        let e = read_sweep_file(&write(dir.path(), "u.tsv", "beta\tdelta\n0.1\t0.2\n"), &m).unwrap_err();
        assert!(e.contains("`delta`") && e.contains("beta, gamma"), "{e}");
        let e = read_sweep_file(&write(dir.path(), "z.tsv", "beta\n"), &m).unwrap_err();
        assert!(e.contains("zero rows"), "{e}");
        let e = read_sweep_file(&write(dir.path(), "d.tsv", "beta\n0.1\n0.2\n0.1\n"), &m).unwrap_err();
        assert!(e.contains("line 4 repeats the design point of line 2"), "{e}");
    }
}
