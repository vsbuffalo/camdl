//! The K1 and K2 known-profile fixtures (`tests/external/known_profiles/`)
//! compile, and their exact answers are committed beside them in the shape
//! the profile tests read (proposal
//! `docs/dev/proposals/2026-10-26-parameter-roles.md` §11.1, Stage 1).
//!
//! The answers are computed by `gen/known_profiles.py` from the committed
//! data, independently of camdl; this test needs neither Python nor the
//! generator, only the files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

fn fixtures() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../tests/external/known_profiles")
}

/// The rows of a reference TSV: its header and every row's numbers.
fn reference(name: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(fixtures().join("ref").join(name)).unwrap();
    let mut lines = text.lines();
    let header = lines.next().unwrap().split('\t').map(str::to_string).collect();
    let rows = lines
        .map(|l| l.split('\t').map(|v| v.parse::<f64>().unwrap()).collect())
        .collect();
    (header, rows)
}

/// The fit config's `[estimate].<name>.bounds`.
fn bounds(fit: &str, name: &str) -> (f64, f64) {
    let text = std::fs::read_to_string(fixtures().join("fits").join(fit)).unwrap();
    let t: toml::Table = toml::from_str(&text).unwrap();
    let b = t["estimate"][name]["bounds"].as_array().unwrap();
    (b[0].as_float().unwrap(), b[1].as_float().unwrap())
}

#[test]
fn both_models_compile() {
    let tmp = tempfile::tempdir().unwrap();
    for m in ["k1", "k2"] {
        let out = Command::new(binary())
            .args(["dev", "compile"])
            .arg(fixtures().join("models").join(format!("{m}.camdl")))
            .arg("-o")
            .arg(tmp.path().join(format!("{m}.ir.json")))
            .output()
            .expect("spawn");
        assert!(out.status.success(), "{m}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

#[test]
fn k1_answers_cover_the_grid_inside_the_box() {
    let (header, rows) = reference("k1_profile.tsv");
    assert_eq!(header, ["mu", "s_hat", "loglik_profiled", "loglik_held_s_2"]);
    assert_eq!(rows.len(), 13, "lin(8.5, 11.5, 13)");
    let (lo, hi) = bounds("k1.toml", "s");
    for r in &rows {
        assert!(r.iter().all(|v| v.is_finite()), "{r:?}");
        // The nuisance maximizer is interior to its box, so the profile
        // follows the likelihood rather than the box edge.
        assert!(lo < r[1] && r[1] < hi, "s_hat {} outside [{lo}, {hi}]", r[1]);
        // A profile is never below the curve with the nuisance held.
        assert!(r[2] >= r[3] - 1e-12, "{r:?}");
    }
}

#[test]
fn k2_answers_are_flat_inside_the_box_and_fall_off_outside() {
    let (header, rows) = reference("k2_profile.tsv");
    assert_eq!(header,
        ["rho", "x0_hat", "loglik_profiled", "loglik_held_x0_40", "loglik_saturated"]);
    assert_eq!(rows.len(), 16, "lin(0.2, 0.95, 16)");
    let (lo, hi) = bounds("k2.toml", "X0");
    let (mut flat, mut edge) = (0, 0);
    for r in &rows {
        let (x0_hat, profiled, saturated) = (r[1], r[2], r[4]);
        assert!((lo..=hi).contains(&x0_hat), "{r:?}");
        if x0_hat > lo && x0_hat < hi {
            assert!((profiled - saturated).abs() < 1e-9, "flat inside the box: {r:?}");
            flat += 1;
        } else {
            assert!(profiled < saturated, "below the maximum at the box edge: {r:?}");
            edge += 1;
        }
    }
    assert!(flat > 0 && edge > 0, "the grid crosses both box edges: {flat} flat, {edge} at an edge");
}
