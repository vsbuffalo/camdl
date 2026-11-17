//! Renders engineering pages through the `typedoc` binary against a
//! hand-written Rust input and the OCaml fixture, and checks the references,
//! the embedded declaration, the rewritten relative links and the failures.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SHA: &str = "78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A docs tree in `root`, as the repository's `docs/` is laid out: the
/// fixture's guide page, and `pages` as `docs/engineering/<name>.md`.
fn docs_tree(root: &Path, pages: &[(&str, &str)]) {
    let docs = root.join("docs");
    fs::create_dir_all(docs.join("engineering")).unwrap();
    fs::copy(fixture("engineering/docs/guide.md"), docs.join("guide.md")).unwrap();
    for (name, text) in pages {
        fs::write(docs.join("engineering").join(name), text).unwrap();
    }
}

/// Runs `typedoc engineering` on `root/docs/engineering`, writing into
/// `root/docs/internals/engineering`.
fn engineering(root: &Path) -> Output {
    let docs = root.join("docs");
    let site = |dir: &str, overview: &str, input: &str| {
        format!(
            "{}={},{}",
            docs.join("internals").join(dir).display(),
            fixture(overview).display(),
            fixture(input).display()
        )
    };
    Command::new(env!("CARGO_BIN_EXE_typedoc"))
        .arg("engineering")
        .arg("--pages")
        .arg(docs.join("engineering"))
        .arg("--out")
        .arg(docs.join("internals/engineering"))
        .arg("--site")
        .arg(site("compiler", "compiler.mld", "compiler.json"))
        .arg("--site")
        .arg(site(
            "runtime",
            "engineering/runtime.md",
            "engineering/runtime.json",
        ))
        .args(["--commit", SHA])
        .output()
        .expect("typedoc runs")
}

/// The rendered page of the fixture's `run-identity.md`.
fn rendered() -> String {
    let dir = tempfile::tempdir().unwrap();
    let text = fs::read_to_string(fixture("engineering/docs/engineering/run-identity.md")).unwrap();
    docs_tree(dir.path(), &[("run-identity.md", &text)]);
    let o = engineering(dir.path());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    fs::read_to_string(
        dir.path()
            .join("docs/internals/engineering/run-identity.qmd"),
    )
    .unwrap()
}

/// The error of a page that fails to render.
fn failure(page: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    docs_tree(dir.path(), &[("bad.md", page)]);
    let o = engineering(dir.path());
    assert!(!o.status.success(), "rendered: {page}");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn the_title_and_first_sentence_are_the_front_matter() {
    let page = rendered();
    assert!(
        page.starts_with(
            "---\ntitle: \"Run identity\"\n\
             description: \"A run is named by the hash of what it was made from.\"\n\
             order: 1\nbody-classes: td-page\n---\n\n\
             The rest of this page says which inputs those are.\n"
        ),
        "{page}"
    );
}

#[test]
fn references_resolve_to_entries_for_both_languages() {
    let page = rendered();
    for link in [
        // A Rust type, a field of it (its type's entry), a method and a
        // function (their lines on the source page), and a module.
        "[`Seed`](../runtime/demo-inputs.qmd#t-Seed)",
        "[`value`](../runtime/demo-inputs.qmd#t-Seed)",
        "[`Seed::new`](../runtime/src/demo-inputs.qmd#L20)",
        "[`run_id`](../runtime/src/demo.qmd#L7)",
        "[`demo::inputs`](../runtime/demo-inputs.qmd)",
        // An OCaml type, a type of a nested module, and a record field, with
        // its link title kept.
        "[`Ir.expr`](../compiler/ir.qmd#t-expr)",
        "[`Ast.Located.t`](../compiler/ast.qmd#t-Located-t)",
        "[the `fn` field](../compiler/ast.qmd#t-call \"a field\")",
    ] {
        assert!(page.contains(link), "missing {link}:\n{page}");
    }
}

#[test]
fn a_decl_marker_embeds_the_entry_with_its_links() {
    let page = rendered();
    let expected = [
        "::: {.td-embed}\n\n",
        "**[`demo::inputs::Seed`](../runtime/demo-inputs.qmd#t-Seed)** · ",
        "[source](../runtime/src/demo-inputs.qmd#L4){.td-source} · ",
        &format!(
            "[on GitHub](https://github.com/vsbuffalo/camdl/blob/{SHA}/rust/demo/inputs.rs#L4){{.td-source}}\n\n"
        ),
        "The seed of a run. See [`SimConfig`](../runtime/demo-inputs.qmd#t-SimConfig).\n\n",
        "```{=html}\n",
    ]
    .concat();
    assert!(page.contains(&expected), "{page}");
    // The declaration links the other type, in HTML, to its rendered page;
    // the type's own name is not linked.
    assert!(
        page.contains(
            "<a class=\"td-ref\" href=\"../runtime/demo-inputs.html#t-SimConfig\">SimConfig</a>"
        ),
        "{page}"
    );
    assert!(
        !page.contains("href=\"../runtime/demo-inputs.html#t-Seed\""),
        "{page}"
    );
    assert!(
        page.contains(
            "**Fields**\n\n`value: u64`\n:   The number the generator starts from.\n\n`config: SimConfig`\n"
        ),
        "{page}"
    );
    assert!(page.contains("\n:::\n"), "{page}");
    assert!(
        !page.contains("<!-- decl: rust:demo::inputs::Seed"),
        "{page}"
    );
}

#[test]
fn relative_links_are_rewritten_for_the_output_directory() {
    let page = rendered();
    assert!(page.contains("[the guide](../../guide.md#seeds)"), "{page}");
    assert!(page.contains("\n[guide]: ../../guide.md\n"), "{page}");
}

#[test]
fn code_is_left_alone() {
    let page = rendered();
    assert!(page.contains("`[not a link](rust:demo::Gone)`"), "{page}");
    assert!(
        page.contains(
            "```text\n[still not a link](rust:demo::Gone)\n<!-- decl: rust:demo::Gone -->\n```"
        ),
        "{page}"
    );
}

const HEAD: &str = "# Bad\n\nA page that fails.\n\n";

#[test]
fn an_unknown_reference_fails_with_page_and_line() {
    // Mutation check: making `Context::destination` return the destination
    // unchanged when the reference names nothing renders this page, and
    // this test fails.
    let err = failure(&format!("{HEAD}See [`Gone`](rust:demo::inputs::Gone).\n"));
    assert!(
        err.contains(
            "docs/engineering/bad.md:5: the reference rust:demo::inputs::Gone names nothing \
             the generated pages document"
        ),
        "{err}"
    );
    // A path of the other language does not resolve.
    let err = failure(&format!("{HEAD}See [`Seed`](ocaml:demo::inputs::Seed).\n"));
    assert!(
        err.contains("bad.md:5: the reference ocaml:demo::inputs::Seed"),
        "{err}"
    );
}

#[test]
fn a_decl_marker_must_name_a_type() {
    let err = failure(&format!("{HEAD}<!-- decl: rust:demo::inputs::Gone -->\n"));
    assert!(
        err.contains(
            "bad.md:5: the declaration marker names rust:demo::inputs::Gone, which names \
             nothing the generated pages document"
        ),
        "{err}"
    );
    let err = failure(&format!("{HEAD}<!-- decl: rust:demo::run_id -->\n"));
    assert!(
        err.contains(
            "bad.md:5: the declaration marker names rust:demo::run_id, which is not a type"
        ),
        "{err}"
    );
    let err = failure(&format!(
        "{HEAD}Text <!-- decl: rust:demo::inputs::Seed -->\n"
    ));
    assert!(err.contains("bad.md:5: a declaration marker"), "{err}");
}

#[test]
fn a_relative_link_to_no_file_fails() {
    let err = failure(&format!("{HEAD}See [this](../gone.md).\n"));
    assert!(
        err.contains("bad.md:5: the link \"../gone.md\" names"),
        "{err}"
    );
    assert!(err.contains("which does not exist"), "{err}");
}

#[test]
fn a_page_needs_a_title_and_an_opening_paragraph() {
    let err = failure("Some text.\n");
    assert!(
        err.contains("bad.md:1: an engineering page opens with its title"),
        "{err}"
    );
    let err = failure("# Bad\n\n<!-- decl: rust:demo::inputs::Seed -->\n");
    assert!(
        err.contains("bad.md:2: the title is followed by a paragraph"),
        "{err}"
    );
}

#[test]
fn no_pages_is_no_error_and_stale_pages_are_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("docs/internals/engineering");
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("old.qmd"), "stale").unwrap();
    // No docs/engineering directory at all.
    let o = engineering(dir.path());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!out.join("old.qmd").exists());
    // Only the index, so the sidebar's section is not empty, which Quarto
    // fails to render.
    let names: Vec<String> = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(names, ["index.qmd"]);
    let index = fs::read_to_string(out.join("index.qmd")).unwrap();
    assert!(
        index.contains("There are no engineering pages yet."),
        "{index}"
    );
}

#[test]
fn the_index_lists_the_pages_in_file_name_order() {
    let dir = tempfile::tempdir().unwrap();
    docs_tree(
        dir.path(),
        &[
            ("b-second.md", "# Second\n\nThe second page.\n"),
            ("a-first.md", "# First\n\nThe first page. More.\n"),
        ],
    );
    let o = engineering(dir.path());
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = dir.path().join("docs/internals/engineering");
    let index = fs::read_to_string(out.join("index.qmd")).unwrap();
    assert!(
        index.starts_with("---\ntitle: \"Engineering\"\n"),
        "{index}"
    );
    assert!(index.contains("order: 0\n"), "{index}");
    assert!(
        index.contains(
            "| Page | Summary |\n|:--|:--|\n\
             | [First](a-first.qmd) | The first page. |\n\
             | [Second](b-second.qmd) | The second page. |\n"
        ),
        "{index}"
    );
    let second = fs::read_to_string(out.join("b-second.qmd")).unwrap();
    assert!(second.contains("order: 2\n"), "{second}");
}
