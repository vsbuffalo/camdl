//! Renders the fixture through the `typedoc` binary and checks the pages.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn render(manifest: &Path, out: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("render")
        .arg("--manifest")
        .arg(manifest)
        .arg("--input")
        .arg(fixture("compiler.json"))
        .arg("--out")
        .arg(out)
        .output()
        .expect("typedoc runs")
}

fn render_ok(out: &Path) {
    let o = render(&fixture("compiler.toml"), out);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

fn read(dir: &Path, name: &str) -> String {
    fs::read_to_string(dir.join(name)).unwrap()
}

fn listing(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

#[test]
fn writes_one_page_per_module_and_an_index() {
    let dir = tempfile::tempdir().unwrap();
    render_ok(dir.path());
    assert_eq!(listing(dir.path()), ["ast.qmd", "index.qmd", "ir.qmd"]);

    let index = read(dir.path(), "index.qmd");
    assert!(index.starts_with("---\ntitle: \"The compiler\"\n"));
    assert!(index.contains("order: 100\n"));
    assert!(index.contains("## Syntax tree\n"));
    assert!(index.contains(
        "| [Ast](ast.qmd) | The parsed form of a camdl model, before names are resolved. | 6 | 83% | 1 |"
    ));
    assert!(index.contains("| [Ir](ir.qmd) | [undocumented]{.td-mark .td-undoc} | 2 | 100% | 1 |"));
    assert!(index.contains("## Reading the marks"));

    let ast = read(dir.path(), "ast.qmd");
    for line in [
        "title: \"Ast\"",
        "description: \"The parsed form of a camdl model, before names are resolved.\"",
        "order: 101",
        "Source: [`ocaml/lib/compiler/ast.ml`](https://github.com/vsbuffalo/camdl/blob/main/ocaml/lib/compiler/ast.ml){.td-source}",
        // The odoc doc: a camdl example, code, a list, a cross-module reference.
        "## destination_form {#t-destination_form}",
        "```camdl\ninfection : S --> I @ beta * S * I / N",
        "The two forms:\n\n- `DstSum [I]` is a sum of compartments;\n- `DstBranch` is a probabilistic branch.",
        "Compare [`Ir.expr`](ir.qmd#t-expr).",
        "```ocaml\ntype destination_form =\n  | DstSum    of stoich_ref list",
        // Member docs, with the history mark on the one that narrates history.
        "`DstSum of stoich_ref list`\n:   Every listed compartment gains one individual per firing.",
        "`DstBranch of (stoich_ref * expr) list`\n:   [history: wave 2, malaria #2, gh#616]{.td-mark .td-history}",
        // Cross-references, with the self reference dropped.
        "**Refers to:** [`stoich_ref`](#t-stoich_ref), [`expr`](#t-expr)\n",
        "**Used by:** [`Ir.transition`](ir.qmd#t-transition)",
        "[source](https://github.com/vsbuffalo/camdl/blob/main/ocaml/lib/compiler/ast.ml#L188){.td-source}",
        // The `and` group.
        "**Defined together with:** [`call`](#t-call)",
        "**Defined together with:** [`expr`](#t-expr)",
        // Marks.
        "[capitals: ONLY]{.td-mark .td-capitals} [1 of 2 constructors undocumented]{.td-mark .td-undoc}",
        "[undocumented]{.td-mark .td-undoc} [2 of 2 fields undocumented]{.td-mark .td-undoc}",
        "[3 bool fields]{.td-mark .td-flags} [string-valued: backend_kind]{.td-mark .td-string-enum}",
        // An unresolvable reference is code, not a link; predefined types are left out.
        "**Refers to:** `Yojson.Safe.t`\n",
        // The nested module, after the parent's types.
        "## Module Located {#m-Located}",
        "### Located.t {#t-Located-t}",
        "| [`'a Located.t`](#t-Located-t) | A located value. |  |",
        "**Refers to:** [`Located.t`](#t-Located-t)",
    ] {
        assert!(ast.contains(line), "ast.qmd lacks:\n{line}\n---\n{ast}");
    }
    assert!(
        ast.find("### Located.t").unwrap() > ast.find("## settings").unwrap(),
        "nested module types follow the parent's"
    );

    let ir = read(dir.path(), "ir.qmd");
    assert!(!ir.contains("description:"), "Ir has no module doc");
    assert!(ir.contains("[history: Previously]{.td-mark .td-history}"));
    assert!(ir.contains(
        "**Refers to:** [`expr`](#t-expr), [`Ast.destination_form`](ast.qmd#t-destination_form)"
    ));
}

#[test]
fn rendering_twice_gives_identical_bytes() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    render_ok(a.path());
    render_ok(b.path());
    render_ok(b.path());
    for name in listing(a.path()) {
        assert_eq!(
            fs::read(a.path().join(&name)).unwrap(),
            fs::read(b.path().join(&name)).unwrap(),
            "{name} differs between runs"
        );
    }
}

#[test]
fn stale_pages_are_deleted_and_other_files_kept() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("gone.qmd"), "old").unwrap();
    fs::write(dir.path().join("notes.txt"), "keep").unwrap();
    render_ok(dir.path());
    assert_eq!(
        listing(dir.path()),
        ["ast.qmd", "index.qmd", "ir.qmd", "notes.txt"]
    );
}

#[test]
fn a_manifest_that_does_not_cover_the_input_fails_naming_modules() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("m.toml");
    fs::write(
        &manifest,
        "title='T'\nsummary='S'\nrepo_blob='https://h/'\norder=1\n\
         [[group]]\ntitle='G'\nmodules=['Ast', 'Parser_errors']\n",
    )
    .unwrap();
    let out = dir.path().join("out");
    let o = render(&manifest, &out);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("input modules not in the manifest: Ir"),
        "{err}"
    );
    assert!(
        err.contains("manifest modules absent from the input: Parser_errors"),
        "{err}"
    );
    assert!(!out.exists(), "nothing is written when the check fails");
}
