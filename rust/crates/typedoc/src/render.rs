//! Writes the pages: an index of the modules by group, and one page per
//! top-level module with every type declaration, its documentation, its
//! marks and its cross-references.

use std::fmt::Write as _;

use std::collections::BTreeMap;

use crate::code::{self, Highlighted};
use crate::markdown::{
    code_span, escape_text, fence, first_sentence, indent, rustdoc_to_markdown,
    split_first_sentence, yaml_string,
};
use crate::marks::{self, Mark};
use crate::model::Target;
use crate::model::{DocLink, Kind, Language, LinkTo, Member};
use crate::odoc;
use crate::overview::Format;
use crate::site::{From, Site, SourceEntry, TypeId, INDEX_SLUG, SOURCE_DIR};

/// Where source links point: a repository on GitHub, at one commit, so that
/// a link's line number names the line the page was generated from.
#[derive(Debug, Clone)]
pub struct SourceLinks {
    repo: String,
    commit: String,
}

impl SourceLinks {
    /// Source links into `repo` (a repository URL such as
    /// `https://github.com/vsbuffalo/camdl`) at `commit`, a full or
    /// abbreviated commit hash. Refuses a commit that is not 7 to 64
    /// hexadecimal digits, so a failed `git rev-parse` cannot produce links
    /// to nothing.
    pub fn new(repo: &str, commit: &str) -> Result<SourceLinks, String> {
        let ok = (7..=64).contains(&commit.len()) && commit.chars().all(|c| c.is_ascii_hexdigit());
        if !ok {
            return Err(format!(
                "--commit {commit:?} is not a commit hash (7 to 64 hexadecimal digits)"
            ));
        }
        let repo = repo.trim_end_matches('/');
        if repo.is_empty() {
            return Err("--repo is empty".to_string());
        }
        Ok(SourceLinks {
            repo: repo.to_string(),
            commit: commit.to_ascii_lowercase(),
        })
    }

    /// The URL of a repository-relative file, at a line when given.
    pub fn url(&self, path: &str, line: Option<u32>) -> String {
        let anchor = line.map_or(String::new(), |n| format!("#L{n}"));
        format!("{}/blob/{}/{path}{anchor}", self.repo, self.commit)
    }
}

/// A generated page: its path relative to the output directory, and its
/// text.
pub struct PageFile {
    pub file_name: String,
    pub text: String,
}

/// Renders every page of the site: the index first, then the module pages in
/// overview order, then one page per source file. `texts` holds each source
/// file's text by [`crate::model::SourceFile::path`].
///
/// Fails when a source text is missing or differs in size from the one the
/// extractor read, or when a span does not fit its text.
pub fn render(
    site: &Site,
    source: &SourceLinks,
    texts: &BTreeMap<String, String>,
) -> Result<Vec<PageFile>, String> {
    let mut files = vec![PageFile {
        file_name: format!("{INDEX_SLUG}.qmd"),
        text: index_page(site),
    }];
    for (p, page) in site.pages.iter().enumerate() {
        files.push(PageFile {
            file_name: format!("{}.qmd", page.slug),
            text: module_page(site, source, p)?,
        });
    }
    for s in &site.sources {
        let path = &s.file.path;
        let text = texts
            .get(path)
            .ok_or_else(|| format!("{path}: no text given for the source file"))?;
        files.push(PageFile {
            file_name: format!("{SOURCE_DIR}/{}.qmd", s.slug),
            text: source_page(site, source, s, text)?,
        });
    }
    Ok(files)
}

/// The page of one source file: the whole text, highlighted, every line
/// anchored `#L<n>`, and every type name the extractor resolved linked to
/// its documentation.
fn source_page(
    site: &Site,
    links: &SourceLinks,
    s: &SourceEntry,
    text: &str,
) -> Result<String, String> {
    let path = &s.file.path;
    let lines = text.lines().count();
    if text.len() != s.file.bytes || lines != s.file.lines {
        return Err(format!(
            "{path}: the file has {} bytes in {lines} lines, but the extractor read {} bytes in {} \
             lines; extract again",
            text.len(),
            s.file.bytes,
            s.file.lines
        ));
    }
    let h = Highlighted::new(text, &s.file.spans).map_err(|e| format!("{path}: {e}"))?;
    let href = |t: &Target| {
        site.find_target(t)
            .map(|id| site.type_href_html(From::Source, id))
    };
    let page = &site.pages[s.page];
    let name = path.rsplit('/').next().unwrap_or(path);
    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(out, "title: {}", yaml_string(name));
    // Reached by links from the module pages and kept out of the sidebar by
    // docs/_quarto.yml; kept out of the search index too, which would
    // otherwise match every identifier in the source.
    out.push_str("search: false\ntoc: false\n");
    out.push_str("body-classes: td-page td-source-page\n");
    out.push_str("---\n\n");
    let _ = writeln!(
        out,
        "The text of {}, documented on [{}](../{}.qmd). Type names link to their \
         documentation. [On GitHub]({}){{.td-source}}\n",
        code_span(path),
        escape_text(&page.name),
        page.slug,
        links.url(path, None)
    );
    let block = code::block(&h.lines_html(&href), "td-src", true);
    let _ = writeln!(out, "{}", fence("{=html}", &block));
    Ok(out)
}

/// Converts a doc comment written in module `m` to Markdown for a page at
/// `from`.
/// An OCaml comment's references are resolved from that module; a Rust
/// comment's links are the ones rustdoc resolved, `links`, and a link
/// rustdoc did not resolve to something on the pages is left unlinked.
fn convert_doc(site: &Site, m: usize, from: From, doc: &str, links: &[DocLink]) -> String {
    match site.modules[m].language {
        Language::Ocaml => {
            let resolve = |path: &str| -> Option<String> {
                site.resolve_type(m, path)
                    .map(|id| site.type_href(from, id))
                    .or_else(|| {
                        site.resolve_module(m, path)
                            .map(|t| site.module_href(from, t))
                    })
            };
            odoc::to_markdown(doc, &resolve)
        }
        Language::Rust => {
            let resolve = |text: &str| -> Option<String> {
                let link = links
                    .iter()
                    .find(|l| l.text == text || l.text.trim_end_matches("()") == text)?;
                doc_link_href(site, from, &link.to)
            };
            rustdoc_to_markdown(doc, &resolve)
        }
    }
}

/// The link target, from `from`, of a resolved doc link.
fn doc_link_href(site: &Site, from: From, to: &LinkTo) -> Option<String> {
    match to {
        LinkTo::Type { module, ty } => site
            .find_target(&Target {
                module: module.clone(),
                ty: ty.clone(),
            })
            .map(|id| site.type_href(from, id)),
        LinkTo::Module { module } => site.module_named(module).map(|m| site.module_href(from, m)),
        LinkTo::Source { path, line } => site.source_href(from, path, *line),
    }
}

/// A doc comment, if present and not blank.
fn present(doc: &Option<String>) -> Option<&str> {
    doc.as_deref().filter(|d| !d.trim().is_empty())
}

/// The summary sentence of a doc comment, or `None` when it has none.
fn summary(
    site: &Site,
    m: usize,
    page: usize,
    doc: &Option<String>,
    links: &[DocLink],
) -> Option<String> {
    present(doc).and_then(|d| first_sentence(&convert_doc(site, m, From::Page(page), d, links)))
}

/// Text for a pipe-table cell: on one line, with `|` escaped.
fn cell(s: &str) -> String {
    s.replace('\n', " ").replace('|', "\\|")
}

pub fn front_matter(out: &mut String, title: &str, description: Option<&str>, order: i64) {
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

/// Converts overview text to Markdown, resolving its references from the
/// index page.
fn convert_overview(site: &Site, text: &str) -> String {
    let resolve = |path: &str| site.index_href(path);
    match site.overview.format {
        Format::Odoc => odoc::page_to_markdown(text, &resolve),
        Format::Markdown => rustdoc_to_markdown(text, &resolve),
    }
}

/// Converted text on one line, for a title or a heading.
fn one_line(md: &str) -> String {
    md.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn index_page(site: &Site) -> String {
    let overview = site.overview;
    let mut out = String::new();
    let intro = convert_overview(site, &overview.intro);
    // The opening sentence is the page's description, which Quarto shows
    // under the title; the body carries on after it rather than printing it
    // twice. Overview::parse refuses an overview without one.
    let (description, rest) = split_first_sentence(&intro).unwrap_or_default();
    front_matter(
        &mut out,
        &one_line(&convert_overview(site, &overview.title)),
        Some(&description),
        site.order,
    );
    push_prose(&mut out, &rest);
    for (g, group) in overview.groups.iter().enumerate() {
        let anchor = group
            .label
            .as_ref()
            .map_or(String::new(), |l| format!(" {{#{l}}}"));
        let _ = writeln!(
            out,
            "## {}{anchor}\n",
            one_line(&convert_overview(site, &group.title))
        );
        push_prose(&mut out, &convert_overview(site, &group.before));
        module_table(&mut out, site, g);
        push_prose(&mut out, &convert_overview(site, &group.after));
    }
    out.push_str(LEGEND);
    out
}

/// Appends converted prose as its own blocks, when there is any.
fn push_prose(out: &mut String, md: &str) {
    let md = md.trim();
    if !md.is_empty() {
        let _ = writeln!(out, "{md}\n");
    }
}

/// The table of a group's module pages, as the overview lists them: each
/// module's summary and counts. A Rust crate's counts cover all its files.
fn module_table(out: &mut String, site: &Site, g: usize) {
    let pages: Vec<usize> = (0..site.pages.len())
        .filter(|&p| site.pages[p].group == g && site.pages[p].parent.is_none())
        .collect();
    page_table(out, site, &pages, true);
}

/// A table of pages, linked from a page in the same directory, with their
/// summaries and counts, which include their child pages when `with_children`.
fn page_table(out: &mut String, site: &Site, pages: &[usize], with_children: bool) {
    out.push_str("| Module | Summary | Types | Documented | History marks |\n");
    out.push_str("|:--|:--|--:|--:|--:|\n");
    for &p in pages {
        let page = &site.pages[p];
        let top = page.modules[0];
        let entry = &site.modules[top];
        let summary = if entry.local.is_empty() {
            summary(site, top, p, &entry.module.doc, &entry.module.doc_links)
                .unwrap_or_else(|| Mark::Undocumented.render())
        } else {
            Mark::Undocumented.render()
        };
        let mut stats = page_stats(site, p);
        if with_children {
            for c in site.children(p) {
                let s = page_stats(site, c);
                stats.types += s.types;
                stats.documented += s.documented;
                stats.history += s.history;
            }
        }
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

fn module_page(site: &Site, source: &SourceLinks, p: usize) -> Result<String, String> {
    let page = &site.pages[p];
    let top = page.modules[0];
    let top_entry = &site.modules[top];
    let mut out = String::new();
    // No description: the module doc opens with its summary sentence, and a
    // description would print that sentence twice under the title.
    front_matter(&mut out, &page.name, None, page.order);

    let path = &top_entry.module.path;
    let _ = writeln!(
        out,
        "Source: [{}]({}/{}.qmd){{.td-source}} · [on GitHub]({}){{.td-source}}\n",
        code_span(path),
        SOURCE_DIR,
        site.source_of(top).slug,
        source.url(path, None)
    );

    if let Some(parent) = page.parent {
        let crate_page = &site.pages[parent];
        let _ = writeln!(
            out,
            "A module of the crate [{}]({}.qmd).\n",
            code_span(&crate_page.name),
            crate_page.slug
        );
    }

    // The top-level module's documentation; nested modules carry their own
    // under their section heading.
    if top_entry.local.is_empty() {
        module_doc(&mut out, site, top, p);
    }

    // A crate's page lists the pages of its other files.
    let children: Vec<usize> = site.children(p).collect();
    if !children.is_empty() {
        out.push_str("## Modules {#modules}\n\n");
        page_table(&mut out, site, &children, false);
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
            let s = summary(site, id.module, p, &t.doc, &t.doc_links).unwrap_or_default();
            let marks = marks::type_marks(t, e.language);
            let _ = writeln!(
                out,
                "| [{}]({}) | {} | {} |",
                code_span(&with_params(
                    e.language,
                    &t.params,
                    &site.type_display(From::Page(p), id)
                )),
                site.type_href(From::Page(p), id),
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
            type_section(
                &mut out,
                site,
                source,
                From::Page(p),
                TypeId { module: m, ty },
                level,
            )?;
        }
    }
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    out.push('\n');
    Ok(out)
}

/// A module's marks and converted doc comment.
fn module_doc(out: &mut String, site: &Site, m: usize, p: usize) {
    let e = &site.modules[m];
    let marks = marks::doc_marks(e.module.doc.as_deref(), e.language);
    if !marks.is_empty() {
        let _ = writeln!(out, "{}\n", marks::render_line(&marks));
    }
    if let Some(doc) = present(&e.module.doc) {
        let _ = writeln!(
            out,
            "{}\n",
            convert_doc(site, m, From::Page(p), doc, &e.module.doc_links)
        );
    }
}

fn type_section(
    out: &mut String,
    site: &Site,
    source: &SourceLinks,
    from: From,
    id: TypeId,
    level: &str,
) -> Result<(), String> {
    let e = &site.modules[id.module];
    let t = &e.module.types[id.ty];
    let anchor = &e.anchors[id.ty];
    let _ = writeln!(
        out,
        "{level} {} {{#{anchor}}}\n",
        escape_text(&site.type_display(from, id))
    );

    let marks = marks::type_marks(t, e.language);
    if !marks.is_empty() {
        let _ = writeln!(out, "{}\n", marks::render_line(&marks));
    }
    type_body(out, site, from, id)?;

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
                    from,
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

    let refers = refers_to(site, from, id);
    if !refers.is_empty() {
        let _ = writeln!(out, "**Refers to:** {}\n", refers.join(", "));
    }

    let users: Vec<String> = site
        .used_by(id)
        .iter()
        .map(|&u| link(site, from, u))
        .collect();
    if !users.is_empty() {
        let _ = writeln!(out, "**Used by:** {}\n", users.join(", "));
    }

    let impls = implements(site, source, from, id);
    if !impls.is_empty() {
        let _ = writeln!(out, "**Implements:** {impls}\n");
    }

    let implementors: Vec<String> = site
        .implemented_by(id)
        .iter()
        .map(|&u| link(site, from, u))
        .collect();
    if !implementors.is_empty() {
        let _ = writeln!(out, "**Implemented by:** {}\n", implementors.join(", "));
    }

    let _ = writeln!(out, "{}\n", source_line(site, source, from, id));
    Ok(())
}

/// What a type's section shows between its heading and its relations: its
/// converted doc comment, its declaration, and each member with its marks
/// and doc comment. An engineering page embeds the same.
pub fn type_body(out: &mut String, site: &Site, from: From, id: TypeId) -> Result<(), String> {
    let e = &site.modules[id.module];
    let t = &e.module.types[id.ty];
    if let Some(doc) = present(&t.doc) {
        let _ = writeln!(
            out,
            "{}\n",
            convert_doc(site, id.module, from, doc, &t.doc_links)
        );
    }
    let _ = writeln!(out, "{}\n", declaration(site, from, id)?);

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
                body.push(convert_doc(site, id.module, from, doc, &mem.doc_links));
            }
            let body = body.join("\n\n");
            let indented = indent(&body, 4);
            let _ = writeln!(out, ":   {}\n", indented.trim_start());
        }
    }
    Ok(())
}

/// The links to where a type is written: its line on its source page and
/// on GitHub.
pub fn source_line(site: &Site, source: &SourceLinks, from: From, id: TypeId) -> String {
    let e = &site.modules[id.module];
    let t = &e.module.types[id.ty];
    let path = &e.module.path;
    // Site::build refuses a module whose file has no source page.
    let on_page = site.source_href(from, path, t.line).unwrap_or_default();
    format!(
        "[source]({on_page}){{.td-source}} · [on GitHub]({}){{.td-source}}",
        source.url(path, Some(t.line))
    )
}

/// The traits a Rust type implements: the derived ones first, then each
/// written impl, linked to the trait's documentation when an input declares
/// it, with a link to where the impl is written.
fn implements(site: &Site, source: &SourceLinks, from: From, id: TypeId) -> String {
    let t = &site.modules[id.module].module.types[id.ty];
    let trait_link = |imp: &crate::model::Impl| -> String {
        let shown = code_span(&imp.trait_path);
        match imp.target.as_ref().and_then(|x| site.find_target(x)) {
            Some(tr) => format!("[{shown}]({})", site.type_href(from, tr)),
            None => shown,
        }
    };
    let derived: Vec<String> = t
        .impls
        .iter()
        .filter(|i| i.derived)
        .map(trait_link)
        .collect();
    let written: Vec<String> = t
        .impls
        .iter()
        .filter(|i| !i.derived)
        .map(|i| match (&i.path, i.line) {
            (Some(path), Some(line)) => {
                let at = site
                    .source_href(from, path, line)
                    .unwrap_or_else(|| source.url(path, Some(line)));
                format!("{} ([impl]({at}){{.td-source}})", trait_link(i))
            }
            _ => trait_link(i),
        })
        .collect();
    let mut parts = Vec::new();
    if !derived.is_empty() {
        parts.push(format!("{} (derived)", derived.join(", ")));
    }
    parts.extend(written);
    parts.join("; ")
}

/// A type's declaration as a highlighted block, each type name in it linked
/// to that type's documentation, except the type's own name.
fn declaration(site: &Site, from: From, id: TypeId) -> Result<String, String> {
    let e = &site.modules[id.module];
    let t = &e.module.types[id.ty];
    let h = Highlighted::new(&t.decl, &t.decl_spans)
        .map_err(|err| format!("{}: declaration of {}: {err}", e.module.path, t.name))?;
    let href = |target: &Target| {
        site.find_target(target)
            .filter(|&other| other != id)
            .map(|other| site.type_href_html(from, other))
    };
    let class = format!("td-decl {}", e.language.fence());
    Ok(fence(
        "{=html}",
        &code::block(&h.lines_html(&href), &class, false),
    ))
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

/// A link to a type, its text the name as seen from `from`.
fn link(site: &Site, from: From, id: TypeId) -> String {
    format!(
        "[{}]({})",
        code_span(&site.type_display(from, id)),
        site.type_href(from, id)
    )
}

/// The "Refers to" entries of a type: a link for each other declaration its
/// declaration names, as the extractor resolved it ([`Site::refers_to`]),
/// shown as written; then a code span for each path it mentions that names
/// no declaration and is not one of the language's own types, such as
/// `Yojson.Safe.t`. Repeats are left out.
fn refers_to(site: &Site, from: From, id: TypeId) -> Vec<String> {
    let t = &site.modules[id.module].module.types[id.ty];
    let decl_lines: Vec<&str> = t.decl.lines().collect();
    // The text of each span that links, with what it links to. Site::build
    // and declaration() checked that every span fits the text.
    let linked: Vec<(String, Option<TypeId>)> = t
        .decl_spans
        .iter()
        .filter_map(|s| {
            let target = s.target.as_ref()?;
            let line = decl_lines.get(s.line.checked_sub(1)?)?;
            let text = line.get(s.col..s.col + s.len)?;
            Some((text.split_whitespace().collect(), site.find_target(target)))
        })
        .collect();
    let mut out: Vec<String> = site
        .refers_to(id)
        .into_iter()
        .map(|target| {
            let written = linked
                .iter()
                .find(|(_, found)| *found == Some(target))
                .map_or(String::new(), |(text, _)| text.clone());
            format!(
                "[{}]({})",
                code_span(&written),
                site.type_href(from, target)
            )
        })
        .collect();
    let mut seen_text: Vec<&str> = Vec::new();
    for r in &t.refs {
        let r = r.trim();
        let resolved = linked.iter().any(|(text, _)| text == r);
        if !resolved && !site.is_predefined(id.module, r) && !seen_text.contains(&r) {
            seen_text.push(r);
            out.push(code_span(r));
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
        (Language::Rust, false) if kind == Kind::Trait && !ty.is_empty() => ty.to_string(),
        (Language::Rust, false) => format!("{}: {ty}", m.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_links_are_pinned_to_the_commit() {
        let sha = "78b0a4af4e1f2c3d4b5a69788796a5b4c3d2e1f0";
        let links = SourceLinks::new("https://github.com/vsbuffalo/camdl/", sha).unwrap();
        assert_eq!(
            links.url("ocaml/lib/compiler/ast.ml", Some(188)),
            format!("https://github.com/vsbuffalo/camdl/blob/{sha}/ocaml/lib/compiler/ast.ml#L188")
        );
        assert_eq!(
            links.url("ocaml/lib/compiler/ast.ml", None),
            format!("https://github.com/vsbuffalo/camdl/blob/{sha}/ocaml/lib/compiler/ast.ml")
        );
    }

    #[test]
    fn a_commit_that_is_not_a_hash_is_refused() {
        for bad in ["", "main", "HEAD", "abc12", "78b0a4af4 "] {
            let e = SourceLinks::new("https://github.com/vsbuffalo/camdl", bad).unwrap_err();
            assert!(e.contains("is not a commit hash"), "{bad:?}: {e}");
        }
    }
}
