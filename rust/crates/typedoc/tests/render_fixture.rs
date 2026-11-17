//! Renders the fixture through the `typedoc` binary and checks the pages.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The commit the fixture's source links are pinned to.
const SHA: &str = "78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0";

fn render(overview: &Path, out: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("render")
        .arg("--overview")
        .arg(overview)
        .arg("--input")
        .arg(fixture("compiler.json"))
        .arg("--out")
        .arg(out)
        .arg("--commit")
        .arg(SHA)
        .output()
        .expect("typedoc runs")
}

fn render_ok(out: &Path) {
    let o = render(&fixture("compiler.mld"), out);
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
    let ast_row =
        "| [Ast](ast.qmd) | The parsed form of a camdl model, before names are resolved. | 6 | 83% | 1 |";
    let ir_row = "| [Ir](ir.qmd) | [undocumented]{.td-mark .td-undoc} | 2 | 100% | 1 |";
    // The overview, in page order: the opening sentence as the description,
    // the rest of the prose, each section's prose around its module table, a
    // subsection heading, and the legend last.
    let expected_body = [
        "---\ntitle: \"The compiler\"\n",
        "description: \"How camdlc turns a .camdl file into the IR the runtime reads, described by its types.\"\n",
        "order: 100\n",
        "---\n\nIt reads the source into [`Ast`](ast.qmd), then lowers it to [`Ir.expr`](ir.qmd#t-expr).\n\n",
        "```camdl\ninfection : S --> I @ beta * S * I / N\n```\n\n",
        "## Syntax tree {#syntax}\n\nThe parser's output.\n\n",
        ast_row,
        "\n\n### What the tree keeps\n\nEvery node keeps its location.\n\n",
        "## Intermediate representation\n\n| Module",
        ir_row,
        "\n\nWhat the runtime reads.\n\n## Reading the marks",
    ];
    let mut at = 0;
    for piece in expected_body {
        match index[at..].find(piece) {
            Some(k) => at += k + piece.len(),
            None => panic!("index.qmd lacks, after byte {at}:\n{piece}\n---\n{index}"),
        }
    }
    assert_eq!(
        index.matches("How camdlc turns").count(),
        1,
        "the opening sentence is printed once, as the description"
    );

    let ast = read(dir.path(), "ast.qmd");
    // A module page has no description: its doc's opening sentence would
    // otherwise appear twice under the title.
    assert!(
        !ast.contains("description:"),
        "module page has a description:\n{ast}"
    );
    for line in [
        "title: \"Ast\"",
        "order: 101",
        "Source: [`ocaml/lib/compiler/ast.ml`](https://github.com/vsbuffalo/camdl/blob/78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0/ocaml/lib/compiler/ast.ml){.td-source}",
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
        "[source](https://github.com/vsbuffalo/camdl/blob/78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0/ocaml/lib/compiler/ast.ml#L188){.td-source}",
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
fn an_overview_that_does_not_cover_the_input_fails_naming_modules() {
    let dir = tempfile::tempdir().unwrap();
    let overview = dir.path().join("index.mld");
    fs::write(
        &overview,
        "{0 T}\n\nSummary.\n\n{1 G}\n\n{!modules: Ast Parser_errors}\n",
    )
    .unwrap();
    let out = dir.path().join("out");
    let o = render(&overview, &out);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("index.mld: "),
        "the error names the file: {err}"
    );
    assert!(
        err.contains("input modules in no {!modules: ...} list of the overview: Ir"),
        "{err}"
    );
    assert!(
        err.contains("overview modules absent from the input: Parser_errors"),
        "{err}"
    );
    assert!(!out.exists(), "nothing is written when the check fails");
}

#[test]
fn a_malformed_overview_fails_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let overview = dir.path().join("index.mld");
    fs::write(&overview, "{0 T}\n\nSummary.\n\n{1 G}\n\nNo list.\n").unwrap();
    let out = dir.path().join("out");
    let o = render(&overview, &out);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("line 5: section \"G\" has no {!modules: ...} list"),
        "{err}"
    );
    assert!(!out.exists(), "nothing is written when the check fails");
}

#[test]
fn the_commit_is_required_and_checked() {
    let dir = tempfile::tempdir().unwrap();
    let base = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_typedoc"));
        c.arg("render")
            .arg("--overview")
            .arg(fixture("compiler.mld"))
            .arg("--input")
            .arg(fixture("compiler.json"))
            .arg("--out")
            .arg(dir.path().join("out"));
        c
    };
    let o = base().output().unwrap();
    assert!(!o.status.success(), "rendering without --commit is refused");
    assert!(String::from_utf8_lossy(&o.stderr).contains("--commit"));
    let o = base().arg("--commit").arg("main").output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("is not a commit hash"));
    assert!(!dir.path().join("out").exists());
}
