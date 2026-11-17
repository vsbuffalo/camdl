//! Writes the pages: an index of the modules by group, and one page per
//! top-level module with every type declaration, its documentation, its
//! marks and its cross-references.

use std::fmt::Write as _;

use crate::markdown::{
    code_span, escape_text, fence, first_sentence, indent, rustdoc_to_markdown, yaml_string,
};
use crate::marks::{self, Mark};
use crate::model::{Kind, Language, Member};
use crate::odoc;
use crate::site::{Site, TypeId, INDEX_SLUG};

/// A generated page: its file name in the output directory, and its text.
pub struct PageFile {
    pub file_name: String,
    pub text: String,
}

/// Renders every page of the site: the index first, then the module pages in
/// manifest order.
pub fn render(site: &Site) -> Vec<PageFile> {
    let mut files = vec![PageFile {
        file_name: format!("{INDEX_SLUG}.qmd"),
        text: index_page(site),
    }];
    for (p, page) in site.pages.iter().enumerate() {
        files.push(PageFile {
            file_name: format!("{}.qmd", page.slug),
            text: module_page(site, p),
        });
    }
    files
}

/// Converts a doc comment written in module `m` to Markdown for page `page`,
/// resolving references from that module.
fn convert_doc(site: &Site, m: usize, page: usize, doc: &str) -> String {
    let resolve = |path: &str| -> Option<String> {
        site.resolve_type(m, path)
            .map(|id| site.type_href(page, id))
            .or_else(|| {
                site.resolve_module(m, path)
                    .map(|t| site.module_href(page, t))
            })
    };
    match site.modules[m].language {
        Language::Ocaml => odoc::to_markdown(doc, &resolve),
        Language::Rust => rustdoc_to_markdown(doc, &resolve),
    }
}

/// A doc comment, if present and not blank.
fn present(doc: &Option<String>) -> Option<&str> {
    doc.as_deref().filter(|d| !d.trim().is_empty())
}

/// The summary sentence of a doc comment, or `None` when it has none.
fn summary(site: &Site, m: usize, page: usize, doc: &Option<String>) -> Option<String> {
    present(doc).and_then(|d| first_sentence(&convert_doc(site, m, page, d)))
}

/// Text for a pipe-table cell: on one line, with `|` escaped.
fn cell(s: &str) -> String {
    s.replace('\n', " ").replace('|', "\\|")
}

fn front_matter(out: &mut String, title: &str, description: Option<&str>, order: i64) {
    out.push_str("---\n");
    let _ = writeln!(out, "title: {}", yaml_string(title));
    if let Some(d) = description {
        let _ = writeln!(out, "description: {}", yaml_string(d));
    }
    let _ = writeln!(out, "order: {order}");
    // Scopes the internals-only styles in docs/_web/theme.scss.
    out.push_str("body-classes: td-page\n");
    out.push_str("---\n\n");
}

/// Counts, for the index, over every module shown on a page.
struct PageStats {
    types: usize,
    documented: usize,
    history: usize,
}

fn page_stats(site: &Site, page: usize) -> PageStats {
    let mut stats = PageStats {
        types: 0,
        documented: 0,
        history: 0,
    };
    let is_history = |doc: &Option<String>, lang| {
        present(doc).is_some_and(|d| {
            marks::doc_marks(Some(d), lang)
                .iter()
                .any(|m| matches!(m, Mark::History(_)))
        })
    };
    for &m in &site.pages[page].modules {
        let e = &site.modules[m];
        stats.history += usize::from(is_history(&e.module.doc, e.language));
        for t in &e.module.types {
            stats.types += 1;
            stats.documented += usize::from(marks::is_documented(t));
            stats.history += usize::from(is_history(&t.doc, e.language));
            stats.history += t
                .members
                .iter()
                .filter(|mem| is_history(&mem.doc, e.language))
                .count();
        }
    }
    stats
}

/// The share of `part` in `whole` as a whole percentage, rounded half up.
fn percent(part: usize, whole: usize) -> String {
    if whole == 0 {
        "–".to_string()
    } else {
        format!("{}%", (100 * part + whole / 2) / whole)
    }
}

fn index_page(site: &Site) -> String {
    let manifest = site.manifest;
    let mut out = String::new();
    front_matter(
        &mut out,
        &manifest.title,
        Some(&manifest.summary),
        manifest.order,
    );
    // The summary is the page's description, which Quarto shows under the
    // title; repeating it as the first paragraph would print it twice.
    for (g, group) in manifest.groups.iter().enumerate() {
        let _ = writeln!(out, "## {}\n", escape_text(&group.title));
        out.push_str("| Module | Summary | Types | Documented | History marks |\n");
        out.push_str("|:--|:--|--:|--:|--:|\n");
        for (p, page) in site
            .pages
            .iter()
            .enumerate()
            .filter(|(_, pg)| pg.group == g)
        {
            let top = page.modules[0];
            let entry = &site.modules[top];
            let summary = if entry.local.is_empty() {
                summary(site, top, p, &entry.module.doc)
                    .unwrap_or_else(|| Mark::Undocumented.render())
            } else {
                Mark::Undocumented.render()
            };
            let stats = page_stats(site, p);
            let _ = writeln!(
                out,
                "| [{}]({}.qmd) | {} | {} | {} | {} |",
                escape_text(&page.name),
                page.slug,
                cell(&summary),
                stats.types,
                percent(stats.documented, stats.types),
                stats.history
            );
        }
        out.push('\n');
    }
    out.push_str(LEGEND);
    out
}

/// The legend of the index page: one sentence per mark on what it flags and
/// why a reviewer would look.
const LEGEND: &str = "\
## Reading the marks

The pages mark what a reviewer may want to question. A mark is a prompt, not an
error: some records do need three independent flags, and the marks exist so the
question is asked where the answer can be seen.

- [undocumented]{.td-mark .td-undoc} A type, constructor, field or module has no
  doc comment, so a reader has to work out what it means from the code. On a
  type, the mark also counts its undocumented constructors or fields.
- [history]{.td-mark .td-history} The doc comment names an issue, a proposal, a
  date or an earlier version: it says how the code came to be rather than what
  it is, and goes stale when the code changes again. The patterns are broad on
  purpose, so a sampler's proposal distribution is marked too.
- [capitals]{.td-mark .td-capitals} The doc comment uses a word in capitals for
  emphasis, where prose would carry the stress.
- [3 bool fields]{.td-mark .td-flags} A record has three or more `bool` fields,
  where a variant type might rule out combinations that cannot occur.
- [3 option fields]{.td-mark .td-options} A record has three or more optional
  fields, where some combinations of present and absent may mean nothing.
- [string-valued]{.td-mark .td-string-enum} A `string` field is named like a
  choice from a fixed set (`kind`, `mode`, `type`, `method`, `status`,
  `strategy`, `backend`, `op`), where a variant type would let the compiler
  check its values. The test reads only the field's name, so it misses some
  such fields and marks some that hold free text.
";

fn module_page(site: &Site, p: usize) -> String {
    let page = &site.pages[p];
    let top = page.modules[0];
    let top_entry = &site.modules[top];
    let top_doc = if top_entry.local.is_empty() {
        &top_entry.module.doc
    } else {
        &None
    };
    let mut out = String::new();
    let description = summary(site, top, p, top_doc);
    front_matter(&mut out, &page.name, description.as_deref(), page.order);

    let path = &top_entry.module.path;
    let _ = writeln!(
        out,
        "Source: [{}]({}{path}){{.td-source}}\n",
        code_span(path),
        site.manifest.repo_blob
    );

    // The top-level module's documentation; nested modules carry their own
    // under their section heading.
    if top_entry.local.is_empty() {
        module_doc(&mut out, site, top, p);
    }

    // The compact table of every type on the page.
    let all_types: Vec<TypeId> = page
        .modules
        .iter()
        .flat_map(|&m| {
            (0..site.modules[m].module.types.len()).map(move |ty| TypeId { module: m, ty })
        })
        .collect();
    if !all_types.is_empty() {
        out.push_str("## Types {#types}\n\n");
        out.push_str("| Type | Summary | Marks |\n|:--|:--|:--|\n");
        for &id in &all_types {
            let e = &site.modules[id.module];
            let t = &e.module.types[id.ty];
            let s = summary(site, id.module, p, &t.doc).unwrap_or_default();
            let marks = marks::type_marks(t, e.language);
            let _ = writeln!(
                out,
                "| [{}]({}) | {} | {} |",
                code_span(&with_params(
                    e.language,
                    &t.params,
                    &site.type_display(p, id)
                )),
                site.type_href(p, id),
                cell(&s),
                cell(&marks::render_line(&marks))
            );
        }
        out.push('\n');
    }

    for &m in &page.modules {
        let e = &site.modules[m];
        let level = if e.local.is_empty() {
            "##"
        } else {
            let anchor = site.module_anchor(m).unwrap_or_default();
            let _ = writeln!(out, "## Module {} {{#{anchor}}}\n", escape_text(&e.local));
            module_doc(&mut out, site, m, p);
            "###"
        };
        for ty in 0..e.module.types.len() {
            type_section(&mut out, site, p, TypeId { module: m, ty }, level);
        }
    }
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    out.push('\n');
    out
}

/// A module's marks and converted doc comment.
fn module_doc(out: &mut String, site: &Site, m: usize, p: usize) {
    let e = &site.modules[m];
    let marks = marks::doc_marks(e.module.doc.as_deref(), e.language);
    if !marks.is_empty() {
        let _ = writeln!(out, "{}\n", marks::render_line(&marks));
    }
    if let Some(doc) = present(&e.module.doc) {
        let _ = writeln!(out, "{}\n", convert_doc(site, m, p, doc));
    }
}

fn type_section(out: &mut String, site: &Site, p: usize, id: TypeId, level: &str) {
    let e = &site.modules[id.module];
    let t = &e.module.types[id.ty];
    let anchor = &e.anchors[id.ty];
    let _ = writeln!(
        out,
        "{level} {} {{#{anchor}}}\n",
        escape_text(&site.type_display(p, id))
    );

    let marks = marks::type_marks(t, e.language);
    if !marks.is_empty() {
        let _ = writeln!(out, "{}\n", marks::render_line(&marks));
    }
    if let Some(doc) = present(&t.doc) {
        let _ = writeln!(out, "{}\n", convert_doc(site, id.module, p, doc));
    }
    let _ = writeln!(out, "{}\n", fence(e.language.fence(), &t.decl));

    if !t.members.is_empty() {
        let noun = t.kind.member_noun();
        let mut heading = noun.to_string();
        heading[..1].make_ascii_uppercase();
        let _ = writeln!(out, "**{heading}**\n");
        for mem in &t.members {
            let _ = writeln!(out, "{}", code_span(&member_term(t.kind, e.language, mem)));
            let mut body = Vec::new();
            let mem_marks = marks::doc_marks(mem.doc.as_deref(), e.language);
            if !mem_marks.is_empty() {
                body.push(marks::render_line(&mem_marks));
            }
            if let Some(doc) = present(&mem.doc) {
                body.push(convert_doc(site, id.module, p, doc));
            }
            let body = body.join("\n\n");
            let indented = indent(&body, 4);
            let _ = writeln!(out, ":   {}\n", indented.trim_start());
        }
    }

    let together: Vec<String> = match t.recursive_group {
        Some(g) => e
            .module
            .types
            .iter()
            .enumerate()
            .filter(|(i, other)| *i != id.ty && other.recursive_group == Some(g))
            .map(|(i, _)| {
                link(
                    site,
                    p,
                    TypeId {
                        module: id.module,
                        ty: i,
                    },
                )
            })
            .collect(),
        None => Vec::new(),
    };
    if !together.is_empty() {
        let _ = writeln!(out, "**Defined together with:** {}\n", together.join(", "));
    }

    let refers = refers_to(site, p, id);
    if !refers.is_empty() {
        let _ = writeln!(out, "**Refers to:** {}\n", refers.join(", "));
    }

    let users: Vec<String> = site.used_by(id).iter().map(|&u| link(site, p, u)).collect();
    if !users.is_empty() {
        let _ = writeln!(out, "**Used by:** {}\n", users.join(", "));
    }

    let _ = writeln!(
        out,
        "[source]({}{}#L{}){{.td-source}}\n",
        site.manifest.repo_blob, e.module.path, t.line
    );
}

/// A type's name applied to its parameters, as the language writes it:
/// `'a t` or `('a, 'b) t` in OCaml, `T<A, B>` in Rust.
fn with_params(language: Language, params: &[String], name: &str) -> String {
    match (language, params) {
        (_, []) => name.to_string(),
        (Language::Ocaml, [one]) => format!("{one} {name}"),
        (Language::Ocaml, many) => format!("({}) {name}", many.join(", ")),
        (Language::Rust, many) => format!("{name}<{}>", many.join(", ")),
    }
}

/// A link to a type, its text the name as seen from page `p`.
fn link(site: &Site, p: usize, id: TypeId) -> String {
    format!(
        "[{}]({})",
        code_span(&site.type_display(p, id)),
        site.type_href(p, id)
    )
}

/// The "Refers to" entries of a type: a link for each path that resolves to
/// another declaration, and a code span for each that resolves to nothing
/// and is not one of the language's own types. Paths naming the type itself
/// are left out, as are repeats.
fn refers_to(site: &Site, p: usize, id: TypeId) -> Vec<String> {
    let t = &site.modules[id.module].module.types[id.ty];
    let mut seen_ids = Vec::new();
    let mut seen_text: Vec<&str> = Vec::new();
    let mut out = Vec::new();
    for r in &t.refs {
        let r = r.trim();
        match site.resolve_type(id.module, r) {
            Some(target) if target == id => {}
            Some(target) => {
                if !seen_ids.contains(&target) {
                    seen_ids.push(target);
                    out.push(format!("[{}]({})", code_span(r), site.type_href(p, target)));
                }
            }
            None => {
                if !site.is_predefined(id.module, r) && !seen_text.contains(&r) {
                    seen_text.push(r);
                    out.push(code_span(r));
                }
            }
        }
    }
    out
}

/// A member as it reads in source: `Name of t` or `name : t` in OCaml,
/// `Name(T)` or `name: T` in Rust.
fn member_term(kind: Kind, language: Language, m: &Member) -> String {
    let ty = m.ty.trim();
    let mutable = if m.mutable { "mutable " } else { "" };
    match (language, kind.has_fields()) {
        (Language::Ocaml, true) => format!("{mutable}{} : {ty}", m.name),
        (Language::Ocaml, false) if ty.is_empty() => m.name.clone(),
        (Language::Ocaml, false) => format!("{} of {ty}", m.name),
        (Language::Rust, true) => format!("{}: {ty}", m.name),
        (Language::Rust, false) if ty.is_empty() => m.name.clone(),
        (Language::Rust, false) if kind == Kind::Enum => {
            if ty.starts_with('(') || ty.starts_with('{') {
                format!("{}{ty}", m.name)
            } else {
                format!("{}({ty})", m.name)
            }
        }
        (Language::Rust, false) => format!("{}: {ty}", m.name),
    }
}
