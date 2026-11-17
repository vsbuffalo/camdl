//! Runs rustdoc on two fixture crates, extracts them with
//! `typedoc extract-rust`, and checks the JSON and the rendered pages: links
//! across crates, enum variants, trait items and impls, resolved doc links,
//! source links only where rustdoc resolved the name, and the refusal of a
//! rustdoc JSON format the extractor was not written for.
//!
//! The fixture's rustdoc JSON is generated here rather than checked in,
//! because rustdoc writes absolute paths of the standard library into it.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// The fixture repository: `rust/` below it holds the crates, as the real
/// repository's `rust/` holds the workspace.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rust")
}

const SHA: &str = "78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0";

fn run(cmd: &mut Command) {
    let o = cmd.output().expect("command runs");
    assert!(
        o.status.success(),
        "{cmd:?}\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

/// The toolchain `rust/rust-toolchain.toml` pins: the one whose rustdoc
/// writes the JSON format the extractor reads.
///
/// rustup picks a toolchain by the directory cargo is started in and passes
/// its choice to child processes as `RUSTUP_TOOLCHAIN`, so `cargo test
/// --manifest-path rust/Cargo.toml` started from the repository root runs
/// this test, and every `rustdoc` it starts, on the default toolchain
/// instead. Naming the pinned toolchain explicitly makes the result
/// independent of where the test was started. Outside rustup the variable
/// is ignored, and the extractor's format check reports a mismatch.
fn pinned_toolchain() -> String {
    let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../rust-toolchain.toml");
    let text = fs::read_to_string(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    text.lines()
        .find_map(|l| {
            let (key, value) = l.split_once('=')?;
            (key.trim() == "channel").then(|| value.trim().trim_matches('"').to_string())
        })
        .unwrap_or_else(|| panic!("{}: no channel", file.display()))
}

/// Writes `other.json` and `fixture.json` into `out`, as `cargo rustdoc`
/// would with the docs build's flags, on the pinned toolchain.
fn rustdoc_json(out: &Path) {
    let ws = repo().join("rust");
    let toolchain = pinned_toolchain();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustdoc = std::env::var("RUSTDOC").unwrap_or_else(|_| "rustdoc".into());
    run(Command::new(rustc)
        .env("RUSTUP_TOOLCHAIN", &toolchain)
        .current_dir(&ws)
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--crate-name",
            "other",
        ])
        .arg("other/lib.rs")
        .arg("--out-dir")
        .arg(out));
    let json = |name: &str, file: &str, extra: &[&str]| {
        run(Command::new(&rustdoc)
            .env("RUSTUP_TOOLCHAIN", &toolchain)
            .current_dir(&ws)
            .env("RUSTC_BOOTSTRAP", "1")
            .args([
                "--edition",
                "2021",
                "--crate-type",
                "lib",
                "--crate-name",
                name,
            ])
            .arg(file)
            .args(extra)
            .args([
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                "--document-private-items",
                "-o",
            ])
            .arg(out));
    };
    json("other", "other/lib.rs", &[]);
    let rlib = out.join("libother.rlib");
    let ext = format!("other={}", rlib.display());
    let dir = out.display().to_string();
    json("fixture", "fixture/lib.rs", &["--extern", &ext, "-L", &dir]);
}

fn extract(out: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("extract-rust")
        .arg("--crate")
        .arg(format!("fixture={}", out.join("fixture.json").display()))
        .arg("--crate")
        .arg(format!("other={}", out.join("other.json").display()))
        .args(["--workspace", "rust"])
        .arg("--repo-root")
        .arg(repo())
        .arg("--out")
        .arg(out.join("input.json"))
        .output()
        .expect("typedoc runs")
}

/// The extracted input, from a fresh rustdoc run.
fn extracted(out: &Path) -> Value {
    rustdoc_json(out);
    let o = extract(out);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_str(&fs::read_to_string(out.join("input.json")).unwrap()).unwrap()
}

fn module<'a>(input: &'a Value, name: &str) -> &'a Value {
    input["modules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == name)
        .unwrap_or_else(|| panic!("no module {name}"))
}

fn ty<'a>(m: &'a Value, name: &str) -> &'a Value {
    m["types"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == name)
        .unwrap_or_else(|| panic!("no type {name}"))
}

/// The text of each linked span of a declaration or file, with its target
/// as `module/type`.
fn links(text: &str, spans: &Value) -> Vec<(usize, String, String)> {
    let lines: Vec<&str> = text.lines().collect();
    spans
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s.get("target").is_some())
        .map(|s| {
            let line = s["line"].as_u64().unwrap() as usize;
            let col = s["col"].as_u64().unwrap() as usize;
            let len = s["len"].as_u64().unwrap() as usize;
            let t = &s["target"];
            (
                line,
                lines[line - 1][col..col + len].to_string(),
                format!(
                    "{}/{}",
                    t["module"].as_str().unwrap(),
                    t["type"].as_str().unwrap()
                ),
            )
        })
        .collect()
}

#[test]
fn extracts_modules_types_members_and_impls() {
    let dir = tempfile::tempdir().unwrap();
    let input = extracted(dir.path());
    assert_eq!(input["language"], "rust");
    let names: Vec<&str> = input["modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "fixture",
            "fixture::model",
            "fixture::inner",
            "other",
            "other::shapes"
        ]
    );
    let root = module(&input, "fixture");
    assert_eq!(root["path"], "rust/fixture/lib.rs");
    assert_eq!(
        root["doc_links"],
        serde_json::json!([{"text": "Model", "to": {"kind": "type", "module": "fixture::model", "type": "Model"}}]),
        "the crate doc's link, resolved by rustdoc through the re-export"
    );
    assert_eq!(
        module(&input, "fixture::inner")["path"],
        "rust/fixture/lib.rs"
    );

    // An enum's variants, with their payloads and docs.
    let shape = ty(module(&input, "other::shapes"), "Shape");
    assert_eq!(shape["kind"], "enum");
    let variants: Vec<(&str, &str, Option<&str>)> = shape["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["name"].as_str().unwrap(),
                m["type"].as_str().unwrap(),
                m["doc"].as_str(),
            )
        })
        .collect();
    assert_eq!(
        variants,
        [
            ("Circle", "(f64)", Some("A circle, by its radius.")),
            ("Square", "{ side: f64 }", Some("A square.")),
            ("Point", "", None),
        ]
    );

    // A struct: comments gone from the declaration, field types as written,
    // the field type from the other crate linked to it.
    let model = ty(module(&input, "fixture::model"), "Model");
    assert_eq!(model["kind"], "struct");
    assert_eq!(model["line"], 7);
    assert_eq!(
        model["decl"],
        "pub struct Model {\n    pub shape: Shape,\n    pub n: u32,\n    hidden: crate::inner::Hidden,\n}"
    );
    assert_eq!(
        links(model["decl"].as_str().unwrap(), &model["decl_spans"]),
        [
            (1, "Model".into(), "fixture::model/Model".into()),
            (2, "Shape".into(), "other::shapes/Shape".into()),
            (
                4,
                "crate::inner::Hidden".into(),
                "fixture::inner/Hidden".into()
            ),
        ]
    );
    let fields: Vec<(&str, &str)> = model["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["name"].as_str().unwrap(), m["type"].as_str().unwrap()))
        .collect();
    assert_eq!(
        fields,
        [
            ("shape", "Shape"),
            ("n", "u32"),
            ("hidden", "crate::inner::Hidden")
        ]
    );
    // A positional field's type ends at the field: not the `)` that closes
    // the tuple, nor the `,` before the next field.
    let weighted = ty(module(&input, "fixture::model"), "Weighted");
    let fields: Vec<(&str, &str)> = weighted["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["name"].as_str().unwrap(), m["type"].as_str().unwrap()))
        .collect();
    assert_eq!(fields, [("0", "u32"), ("1", "f64")]);
    let n_links = &model["members"][1]["doc_links"];
    assert_eq!(
        n_links,
        &serde_json::json!([
            {"text": "Shape", "to": {"kind": "type", "module": "other::shapes", "type": "Shape"}},
            {"text": "make", "to": {"kind": "source", "path": "rust/fixture/model.rs", "line": 26}},
        ]),
        "a type links to its page, a function to its line"
    );
    assert_eq!(
        model["refs"],
        serde_json::json!(["Shape", "crate::inner::Hidden"])
    );

    // Impls: derived ones by name, written ones as written, a documented
    // trait linked.
    assert_eq!(
        model["impls"],
        serde_json::json!([
            {"trait": "Debug", "derived": true, "path": null, "line": null},
            {"trait": "Clone", "derived": true, "path": null, "line": null},
            {"trait": "other::Area", "target": {"module": "other", "type": "Area"},
             "derived": false, "path": "rust/fixture/model.rs", "line": 15},
            {"trait": "crate::Step", "target": {"module": "fixture", "type": "Step"},
             "derived": false, "path": "rust/fixture/model.rs", "line": 21},
        ])
    );

    // A trait: its items' signatures, a provided method's body elided.
    let step = ty(root, "Step");
    assert_eq!(step["kind"], "trait");
    assert!(
        step["decl"]
            .as_str()
            .unwrap()
            .contains("fn twice(&mut self, dt: f64) { … }"),
        "{}",
        step["decl"]
    );
    let items: Vec<&str> = step["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        items,
        [
            "fn step(&mut self, dt: f64)",
            "fn twice(&mut self, dt: f64)"
        ]
    );
    assert_eq!(ty(root, "Counts")["kind"], "alias");

    // The values an engineering page may link to: a module's function, and
    // a method of an inherent impl, but no trait impl's method (`area`,
    // `step`).
    assert_eq!(
        module(&input, "fixture::model")["values"],
        serde_json::json!([
            {"name": "make", "path": "rust/fixture/model.rs", "line": 26},
            {"name": "Model::count", "path": "rust/fixture/model.rs", "line": 34},
        ])
    );
    assert!(root.get("values").is_none(), "{}", root["values"]);
    assert_eq!(
        ty(module(&input, "fixture::inner"), "Hidden")["kind"],
        "tuple_struct"
    );
}

#[test]
fn source_links_only_where_rustdoc_resolved_the_name() {
    let dir = tempfile::tempdir().unwrap();
    let input = extracted(dir.path());
    let file = input["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["path"] == "rust/fixture/model.rs")
        .unwrap();
    let text = fs::read_to_string(repo().join("rust/fixture/model.rs")).unwrap();
    assert_eq!(file["bytes"].as_u64().unwrap() as usize, text.len());
    let got = links(&text, &file["spans"]);
    // Line 26 is `make`'s signature, whose `Model`s rustdoc resolved; line 27
    // is its body, which rustdoc does not record, so its `Model` is plain.
    // Line 3's private `use` is not linked: rustdoc's JSON leaves private
    // imports out even with --document-private-items.
    let model = "fixture::model/Model".to_string();
    let expected: Vec<(usize, String, String)> = vec![
        (7, "Model".into(), model.clone()),
        (9, "Shape".into(), "other::shapes/Shape".into()),
        (
            12,
            "crate::inner::Hidden".into(),
            "fixture::inner/Hidden".into(),
        ),
        (15, "other::Area".into(), "other/Area".into()),
        (15, "Model".into(), model.clone()),
        (21, "crate::Step".into(), "fixture/Step".into()),
        (21, "Model".into(), model.clone()),
        (26, "Model".into(), model.clone()),
        (26, "Model".into(), model.clone()),
        // The inherent impl's header.
        (32, "Model".into(), model),
        // A type's name where it is defined.
        (40, "Weighted".into(), "fixture::model/Weighted".into()),
    ];
    assert_eq!(got, expected);

    // A public re-export is in rustdoc's JSON, and links.
    let lib = input["sources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["path"] == "rust/fixture/lib.rs")
        .unwrap();
    let lib_text = fs::read_to_string(repo().join("rust/fixture/lib.rs")).unwrap();
    // The supertrait on line 14 links; the same path in the provided
    // method's body on line 20 does not, though it lies inside the trait's
    // declaration. Mutation check: dropping the body exclusion in
    // `CrateExtract::source` links line 20 and fails this assertion.
    let lib_links = links(&lib_text, &lib["spans"]);
    assert_eq!(
        lib_links,
        [
            (5, "model::Model".into(), "fixture::model/Model".into()),
            (10, "Hidden".into(), "fixture::inner/Hidden".into()),
            (14, "Step".into(), "fixture/Step".into()),
            (14, "other::Area".into(), "other/Area".into()),
            (27, "Counts".into(), "fixture/Counts".into()),
        ]
    );

    let classes: Vec<&str> = file["spans"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["line"] == 11)
        .map(|s| s["class"].as_str().unwrap())
        .collect();
    assert_eq!(classes, ["keyword", "builtin_type", "comment"], "line 11");
}

#[test]
fn renders_crate_and_module_pages() {
    let dir = tempfile::tempdir().unwrap();
    extracted(dir.path());
    let out = dir.path().join("pages");
    let o = Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("render")
        .arg("--overview")
        .arg(repo().join("overview.md"))
        .arg("--input")
        .arg(dir.path().join("input.json"))
        .arg("--out")
        .arg(&out)
        .args(["--commit", SHA])
        .arg("--source-root")
        .arg(repo())
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let mut pages: Vec<String> = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    pages.sort();
    assert_eq!(
        pages,
        [
            "fixture-model.qmd",
            "fixture.qmd",
            "index.qmd",
            "other.qmd",
            "src"
        ]
    );
    let read = |f: &str| fs::read_to_string(out.join(f)).unwrap();
    let index = read("index.qmd");
    assert!(
        index.contains("See [`fixture::model::Model`](fixture-model.qmd#t-Model)."),
        "{index}"
    );
    assert!(index.contains("| [fixture](fixture.qmd) |"), "{index}");
    assert!(
        !index.contains("| [fixture::model]"),
        "a module file is listed on its crate's page, not the index"
    );
    let krate = read("fixture.qmd");
    assert!(krate.contains("## Modules {#modules}"), "{krate}");
    assert!(
        krate.contains("| [fixture::model](fixture-model.qmd) |"),
        "{krate}"
    );
    assert!(krate.contains("## Module inner {#m-inner}"), "{krate}");
    assert!(
        krate.contains("**Implemented by:** [`fixture::model::Model`](fixture-model.qmd#t-Model)"),
        "{krate}"
    );
    let page = read("fixture-model.qmd");
    assert!(
        page.contains("A module of the crate [`fixture`](fixture.qmd)."),
        "{page}"
    );
    assert!(
        page.contains(
            "See [`Shape`](other.qmd#t-shapes--Shape) and [`make`](src/fixture-model.qmd#L26)."
        ),
        "{page}"
    );
    assert!(
        page.contains(
            "**Implements:** `Debug`, `Clone` (derived); [`other::Area`](other.qmd#t-Area) \
             ([impl](src/fixture-model.qmd#L15){.td-source})"
        ),
        "{page}"
    );
}

#[test]
fn a_foreign_format_version_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("future.json");
    fs::write(&json, r#"{"format_version": 99, "root": 0, "index": {}}"#).unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("extract-rust")
        .arg("--crate")
        .arg(format!("x={}", json.display()))
        .output()
        .unwrap();
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("rustdoc JSON format_version 99 is not 57"),
        "{err}"
    );
    assert!(err.contains("rustdoc-types"), "{err}");
}
