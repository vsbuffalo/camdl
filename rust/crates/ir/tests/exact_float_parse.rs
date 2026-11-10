//! Every JSON read path parses a float to exactly the double it was written
//! from (gh#1018).
//!
//! serde_json's default float parser is fast but not correctly rounded: it
//! reads `0.9724999999999999` as 0.9725, one unit in the last place (ULP)
//! away. A value the runtime writes to a summary or `fit.meta` JSON and reads
//! back was then not the value it computed, and an IR constant the compiler
//! wrote was not the constant the runtime simulated with. The workspace
//! enables serde_json's `float_roundtrip` feature (`rust/Cargo.toml`), which
//! makes the parser correctly rounded. Every crate that reads JSON depends on
//! `ir`, so cargo's feature unification turns the feature on for all of them
//! whenever this crate builds — which is why the invariant is pinned here.
//!
//! Two properties are checked:
//!
//! - **Round trip.** `bits(parse(serialize(x))) == bits(x)` for every finite
//!   `f64` tried: a few hundred thousand random bit patterns (uniform over the
//!   encoding, so every exponent is exercised), the subnormals, the extremes,
//!   and short decimals such as `0.1` and `1/3` whose shortest representation
//!   is where an inexact parser slips.
//! - **IR constants.** Every float literal in every committed `*.ir.json`
//!   parses to the nearest double of its written decimal (Rust's `str::parse`
//!   is correctly rounded and is the reference), and loading the IR through
//!   [`ir::envelope_from_str`] carries those exact doubles into the model.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// splitmix64: a fixed, dependency-free stream of well-mixed `u64`s, so the
/// sample of bit patterns is the same on every run.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Values where a parser that is not correctly rounded is most likely to
/// miss: subnormals, the ends of the range, ties near powers of ten, and
/// short decimals that are not exact in binary.
fn edge_values() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        f64::from_bits(1),                     // smallest subnormal, 5e-324
        f64::from_bits(2),
        f64::from_bits(0x000F_FFFF_FFFF_FFFF), // largest subnormal
        f64::MIN_POSITIVE,                     // smallest normal
        f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1),
        f64::MAX,
        f64::MIN,
        f64::from_bits(f64::MAX.to_bits() - 1),
        f64::EPSILON,
        1.0,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
        0.1,
        0.2,
        0.3,
        0.1 + 0.2,
        1.0 / 3.0,
        2.0 / 3.0,
        1e23,
        8.98846567431158e307,
        // The decimal that hung strtod implementations; written as text
        // because it carries more digits than its double needs.
        "2.2250738585072011e-308".parse().unwrap(),
        // The value gh#1018 was found on, and the IR constants the default
        // parser read one ULP away.
        0.9724999999999999,
        12783.487500000001,
        -1.9459101490553135,
        0.0013698630136986301,
        0.24840292533115854,
        0.09027013094248869,
    ];
    // Every power of ten in range, and its neighbours either side.
    for e in -323..=308 {
        let p: f64 = format!("1e{e}").parse().unwrap();
        v.push(p);
        v.push(f64::from_bits(p.to_bits() + 1));
        v.push(f64::from_bits(p.to_bits() - 1));
    }
    // Every subnormal exponent band.
    for k in 0..52 {
        v.push(f64::from_bits(1u64 << k));
        v.push(f64::from_bits((1u64 << (k + 1)) - 1));
    }
    let negated: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(negated);
    v
}

fn assert_round_trips(x: f64) {
    let text = serde_json::to_string(&x).unwrap();
    let back: f64 = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("{text} does not parse back: {e}"));
    assert_eq!(
        back.to_bits(),
        x.to_bits(),
        "serialize→parse moved {x:e} (bits {:#018x}): wrote {text}, read {back:e} (bits {:#018x})",
        x.to_bits(),
        back.to_bits()
    );
    // The reader-based entry point (`from_reader`, used for IR files and
    // streamed records) takes a different input path to the same parser.
    let via_reader: f64 = serde_json::from_reader(text.as_bytes()).unwrap();
    assert_eq!(via_reader.to_bits(), x.to_bits(), "from_reader moved {x:e}: wrote {text}");
}

#[test]
fn every_finite_f64_tried_survives_serialize_then_parse() {
    let mut state = 0x1018_u64;
    let mut tried = 0usize;
    while tried < 300_000 {
        let x = f64::from_bits(splitmix64(&mut state));
        if !x.is_finite() {
            continue;
        }
        assert_round_trips(x);
        tried += 1;
    }
    for x in edge_values() {
        assert_round_trips(x);
    }
}

/// A float inside a document — an object field and an array element, the
/// shape every summary and IR file has — round-trips like a bare one.
#[test]
fn floats_inside_a_document_survive_serialize_then_parse() {
    let mut state = 0xF00D_u64;
    let xs: Vec<f64> = std::iter::from_fn(|| Some(f64::from_bits(splitmix64(&mut state))))
        .filter(|x| x.is_finite())
        .take(10_000)
        .chain(edge_values())
        .collect();
    let doc = serde_json::json!({ "values": xs, "first": xs[0] });
    let text = serde_json::to_string(&doc).unwrap();
    let back: Value = serde_json::from_str(&text).unwrap();
    let read: Vec<f64> = back["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(read.len(), xs.len());
    for (a, b) in xs.iter().zip(&read) {
        assert_eq!(a.to_bits(), b.to_bits(), "document round trip moved {a:e} to {b:e}");
    }
    assert_eq!(back["first"].as_f64().unwrap().to_bits(), xs[0].to_bits());
}

/// The number tokens of a JSON text, outside strings, in document order.
fn number_tokens(s: &str) -> Vec<&str> {
    let b = s.as_bytes();
    let (mut i, mut out, mut in_str, mut esc) = (0, Vec::new(), false, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_str = true;
            i += 1;
            continue;
        }
        if c == b'-' || c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_digit() || b"-+.eE".contains(&b[i])) {
                i += 1;
            }
            out.push(&s[start..i]);
            continue;
        }
        i += 1;
    }
    out
}

fn collect_floats(v: &Value, out: &mut HashSet<u64>) {
    match v {
        Value::Number(n) => {
            if let Some(x) = n.as_f64() {
                out.insert(x.to_bits());
            }
        }
        Value::Array(xs) => xs.iter().for_each(|x| collect_floats(x, out)),
        Value::Object(m) => m.values().for_each(|x| collect_floats(x, out)),
        _ => {}
    }
}

fn ir_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries {
            let p = e.expect("dir entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.to_string_lossy().ends_with(".ir.json") {
                out.push(p);
            }
        }
    }
    let root = repo_root();
    let mut out = Vec::new();
    for d in ["ir/golden", "ocaml/golden", "tests", "rust/crates/sim/tests/fixtures"] {
        walk(&root.join(d), &mut out);
    }
    out.sort();
    out
}

#[test]
fn ir_constants_parse_to_the_nearest_double_of_their_written_decimal() {
    let files = ir_files();
    // The committed set is 135 files; a much smaller count means the walk
    // missed a directory and the test would pass vacuously.
    assert!(files.len() >= 130, "found only {} IR files", files.len());
    let mut n_literals = 0usize;
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        for tok in number_tokens(&text) {
            if !tok.contains(['.', 'e', 'E']) {
                continue;
            }
            let nearest: f64 = tok.parse().unwrap();
            let parsed: f64 = serde_json::from_str(tok).unwrap();
            assert_eq!(
                parsed.to_bits(),
                nearest.to_bits(),
                "{}: `{tok}` parsed to {parsed:e}, not the nearest double {nearest:e}",
                path.display()
            );
            n_literals += 1;
        }
    }
    assert!(n_literals > 10_000, "only {n_literals} float literals checked");
}

/// Loading an IR file through the real entry point carries the exact doubles
/// into the model: re-serializing the loaded envelope to a `Value` (no text
/// in between, so nothing is re-parsed) contains the nearest double of every
/// float literal the file holds. The six files that held the 39 constants the
/// default parser moved are named so a reader can see they are covered.
#[test]
fn ir_load_carries_the_exact_written_constants_into_the_model() {
    let mut loaded = 0usize;
    let mut covered = HashSet::new();
    for path in ir_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        // The frozen `ir/golden/` set and a few fixtures may lag the current
        // `ir/VERSION`; this test is about float parsing, not the version
        // handshake, so it takes the files the current loader accepts.
        let Ok(env) = ir::envelope_from_str(&text) else { continue };
        let mut in_model = HashSet::new();
        collect_floats(&serde_json::to_value(&env).unwrap(), &mut in_model);
        for tok in number_tokens(&text) {
            if !tok.contains(['.', 'e', 'E']) {
                continue;
            }
            let nearest: f64 = tok.parse().unwrap();
            if !nearest.is_finite() {
                continue;
            }
            assert!(
                in_model.contains(&nearest.to_bits()),
                "{}: `{tok}` is not in the loaded model as its nearest double {nearest:e}",
                path.display()
            );
        }
        covered.insert(path.file_name().unwrap().to_string_lossy().into_owned());
        loaded += 1;
    }
    assert!(loaded >= 100, "only {loaded} IR files loaded");
    for name in [
        "sir_five_age.ir.json",
        "bspline_loglink.ir.json",
        "seir_spatial_5_inference.ir.json",
        "seir_k12_truth.ir.json",
        "bspline_forcing_grad.ir.json",
        "bspline_smooth_grad.ir.json",
    ] {
        assert!(covered.contains(name), "{name} was not loaded");
    }
}
