//! Indexed `set` targets in effect families (proposal
//! 2026-11-01-indexed-set-targets), checked end to end through `camdlc` and the
//! `camdl` runtime on every backend.
//!
//! A `set` family whose right-hand side reads other cells is a simultaneous map:
//! every instance due at one instant reads the same pre-event state (spec
//! section 13.10). The age shift below, written as a `set` family, must therefore
//! give exactly what the equivalent `transfer` family gives. A cascading
//! (one-cell-at-a-time) application would differ at `a1`: it would move half of
//! the already-updated 110 instead of half of the original 20.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn real_camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn built() -> Option<(PathBuf, PathBuf)> {
    let (bin, cc) = (camdl_bin(), real_camdlc());
    if !bin.exists() || !cc.exists() {
        eprintln!("skipping: camdl/camdlc not built");
        return None;
    }
    Some((bin, cc))
}

/// Three ages, no dynamics (`gamma = 0`), and one event at t = 1 given by
/// `effect`. `age_out` is the share of each age moving up; `younger[a, b]` is 1
/// where band `b` feeds band `a`.
fn model(effect: &str) -> String {
    format!(
        r#"time_unit = 'years
compartments {{ S, I }}
dimensions {{ age = [a0, a1, a2] }}
stratify(by = age)
parameters {{ gamma : rate }}
tables {{
  age_out : age = [0.9, 0.5, 0.0]
  younger : age × age = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
}}
transitions {{ recover[a in age] : I[a] --> S[a] @ gamma * I[a] }}
init {{ S[a0] = 100  S[a1] = 20  S[a2] = 3  I[a0] = 10  I[a1] = 2  I[a2] = 0 }}
events {{
{effect}
}}
simulate {{ from = 0 'years  to = 2 'years }}
"#
    )
}

const SET_SHIFT: &str = r#"  shift[c in compartments, a in age] : {
    c[a] = (1 - age_out[a]) * c[a] + sum(b in age, younger[a, b] * age_out[b] * c[b])
    at = [1]
  }"#;

const TRANSFER_SHIFT: &str = r#"  shift[c in compartments, (a, a_next) in consecutive(age)] :
    transfer(fraction = age_out[a], from = c[a], to = c[a_next]) at [1]"#;

/// Simulate `src` on `backend`; returns (status ok, stdout, stderr).
fn simulate(src: &str, backend: &str) -> (bool, String, String) {
    let (bin, cc) = built().expect("built");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.camdl");
    std::fs::write(&path, src).unwrap();
    let out = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args([
            "simulate", path.to_str().unwrap(), "--backend", backend,
            "--param", "gamma=0", "--seed", "1", "--to", "2", "--output-every", "1",
            "--stdout", "--progress", "none",
        ])
        .env("CAMDLC", cc)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_NO_IR_CACHE", "1")
        .output()
        .expect("spawn camdl");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The `[S_a0, S_a1, S_a2, I_a0, I_a1, I_a2]` row at time `t`.
fn row_at(tsv: &str, t: &str) -> Vec<f64> {
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let row: Vec<&str> = lines
        .find(|l| l.split('\t').next() == Some(t))
        .unwrap_or_else(|| panic!("no row at t = {t}"))
        .split('\t')
        .collect();
    ["S_a0", "S_a1", "S_a2", "I_a0", "I_a1", "I_a2"]
        .iter()
        .map(|c| {
            let i = header.iter().position(|h| h == c).unwrap_or_else(|| panic!("no column {c}"));
            row[i].parse().unwrap()
        })
        .collect()
}

#[test]
fn set_family_shift_is_simultaneous_on_every_backend() {
    if built().is_none() {
        return;
    }
    let simultaneous = vec![10.0, 100.0, 13.0, 1.0, 10.0, 1.0];
    for backend in ["ode", "chain_binomial", "gillespie"] {
        let (ok_t, out_t, err_t) = simulate(&model(TRANSFER_SHIFT), backend);
        assert!(ok_t, "{backend}: transfer family failed: {err_t}");
        assert_eq!(row_at(&out_t, "1"), simultaneous,
            "{backend}: the transfer family is the reference for a simultaneous shift");

        let (ok_s, out_s, err_s) = simulate(&model(SET_SHIFT), backend);
        assert!(ok_s, "{backend}: set family failed: {err_s}");
        assert_eq!(row_at(&out_s, "1"), simultaneous,
            "{backend}: the set family must read one snapshot, as the transfer family does");
    }
}

/// A `set` family and a `transfer` event touching `S_a1` at one instant: the
/// existing runtime conflict check refuses it, as it would for two hand-written
/// events.
#[test]
fn set_family_conflicting_with_a_transfer_is_refused() {
    if built().is_none() {
        return;
    }
    let effect = format!(
        "{SET_SHIFT}\n  drain : transfer(fraction = 0.5, from = S[a1], to = I[a1]) at [1]"
    );
    let (ok, _out, err) = simulate(&model(&effect), "ode");
    assert!(!ok, "a set and a transfer on one cell at one instant must be refused");
    assert!(err.contains("same instant"), "expected the same-instant conflict error, got: {err}");
}
