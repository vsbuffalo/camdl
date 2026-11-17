//! Engineering pages: hand-written Markdown, one page per engineering
//! feature, whose references to the code are checked and whose type
//! declarations are taken from the code when the pages are rendered.
//!
//! A page is a Markdown file that opens with a `# Title` line and then a
//! paragraph. The title is the page's title, and the paragraph's first
//! sentence its description, which Quarto shows under the title; the body
//! carries on after that sentence. In the body:
//!
//! - A link whose destination is `rust:<path>` or `ocaml:<path>` is a
//!   reference to the code, by its full path: `[`Seed`](rust:runid::inputs::Seed)`,
//!   `[`Ir.model`](ocaml:Ir.model)`. It is rewritten to the generated page of
//!   what the path names ([`Site::named`]): a type's entry, the entry of the
//!   type a field, variant or trait item belongs to, a function's or method's
//!   line on its source page, or a module's page.
//! - A line holding only `<!-- decl: rust:<path> -->` is replaced by that
//!   type's entry as its own page shows it ([`render::type_body`]): its doc
//!   comment, its declaration with its type names linked, and its members,
//!   under a link to the entry and to its source.
//! - A relative link is written relative to the page's own directory and is
//!   rewritten to point at the same file from the output directory, so links
//!   to `docs/` pages keep working and `docs/_web/unpublished-links.lua`
//!   still sees which `.md` file they name.
//!
//! Text inside fenced code blocks and code spans is left alone. A reference
//! or a declaration marker that names nothing, and a relative link to a file
//! that does not exist, fail the render with the page, line and target.

use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};

use regex::Regex;

use crate::markdown::{code_span, fence_marker, split_first_sentence};
use crate::model::Language;
use crate::render::{self, front_matter, source_line, SourceLinks};
use crate::site::{From, Named, Site};

/// The prefix of a reference to code in `language`, written before its path.
fn prefix_of(language: Language) -> &'static str {
    match language {
        Language::Ocaml => "ocaml:",
        Language::Rust => "rust:",
    }
}

/// A site the pages link into: its pages' directory as a path relative to
/// the engineering pages' output directory, such as `../runtime`.
pub struct Linked<'s> {
    pub site: &'s Site<'s>,
    pub dir: String,
}

/// Where a page's relative links point from and must point from after
/// rendering: both directories absolute and without `.` or `..`.
pub struct Relocate {
    pub source_dir: PathBuf,
    pub out_dir: PathBuf,
}

/// A path made absolute against the current directory, its `.` and `..`
/// components resolved by name, without reading the file system.
pub fn absolute(p: &Path) -> Result<PathBuf, String> {
    let abs = std::path::absolute(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut out = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    Ok(out)
}

/// The path of `to` relative to the directory `from`, both absolute and
/// resolved by [`absolute`], with `/` between segments.
pub fn relative(from: &Path, to: &Path) -> String {
    let a: Vec<Component> = from.components().collect();
    let b: Vec<Component> = to.components().collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut segs: Vec<String> = vec!["..".to_string(); a.len() - common];
    segs.extend(
        b[common..]
            .iter()
            .map(|c| c.as_os_str().to_string_lossy().into_owned()),
    );
    if segs.is_empty() {
        ".".to_string()
    } else {
        segs.join("/")
    }
}

impl Relocate {
    /// A relative link destination, as written in a page of `source_dir`,
    /// rewritten for a page of `out_dir`. Fails when it names no file.
    fn rewrite(&self, dest: &str) -> Result<String, String> {
        let cut = dest.find(['#', '?']).unwrap_or(dest.len());
        let (path, frag) = dest.split_at(cut);
        let target = absolute(&self.source_dir.join(path))?;
        if !target.exists() {
            return Err(format!(
                "the link {dest:?} names {}, which does not exist",
                target.display()
            ));
        }
        Ok(format!("{}{frag}", relative(&self.out_dir, &target)))
    }
}

/// What a page needs to resolve its references and links.
pub struct Context<'c> {
    pub sites: &'c [Linked<'c>],
    pub source: &'c SourceLinks,
    pub relocate: &'c Relocate,
}

impl Context<'_> {
    /// What a reference names, and the site it is in.
    fn named(&self, language: Language, path: &str) -> Option<(&Linked<'_>, Named<'_>)> {
        self.sites
            .iter()
            .find_map(|l| l.site.named(language, path).map(|n| (l, n)))
    }

    /// A reference `rust:<path>` or `ocaml:<path>`, if `dest` is one: its
    /// language and path.
    fn reference(dest: &str) -> Option<(Language, &str)> {
        [Language::Rust, Language::Ocaml]
            .into_iter()
            .find_map(|l| dest.strip_prefix(prefix_of(l)).map(|p| (l, p)))
    }

    /// A link destination as the rendered page writes it.
    fn destination(&self, dest: &str) -> Result<String, String> {
        if let Some((language, path)) = Self::reference(dest) {
            let (l, named) = self.named(language, path).ok_or_else(|| {
                format!("the reference {dest} names nothing the generated pages document")
            })?;
            let from = From::Outside(&l.dir);
            return Ok(match named {
                Named::Type(id) | Named::Member(id, _) => l.site.type_href(from, id),
                Named::Module(m) => l.site.module_href(from, m),
                // Site::build checked that the value's file has a page.
                Named::Value(v) => l
                    .site
                    .source_href(from, &v.path, v.line)
                    .unwrap_or_default(),
            });
        }
        // A URL: a scheme, `[A-Za-z][A-Za-z0-9+.-]*`, then a colon.
        let scheme = dest.split_once(':').is_some_and(|(s, _)| {
            s.starts_with(|c: char| c.is_ascii_alphabetic())
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "+.-".contains(c))
        });
        if dest.is_empty() || dest.starts_with(['#', '/']) || scheme {
            return Ok(dest.to_string());
        }
        self.relocate.rewrite(dest)
    }

    /// A type's entry, embedded where a page's `<!-- decl: ... -->` line is.
    fn embed(&self, target: &str) -> Result<String, String> {
        let not_type = |what: &str| format!("the declaration marker names {target}, {what}");
        let (language, path) = Self::reference(target)
            .ok_or_else(|| not_type("which is not rust:<path> or ocaml:<path>"))?;
        let (l, named) = self
            .named(language, path)
            .ok_or_else(|| not_type("which names nothing the generated pages document"))?;
        let Named::Type(id) = named else {
            return Err(not_type("which is not a type"));
        };
        let from = From::Outside(&l.dir);
        let mut out = String::from("::: {.td-embed}\n\n");
        let _ = writeln!(
            out,
            "**[{}]({})** · {}\n",
            code_span(&l.site.type_display(from, id)),
            l.site.type_href(from, id),
            source_line(l.site, self.source, from, id)
        );
        render::type_body(&mut out, l.site, from, id)?;
        out.push_str(":::\n");
        Ok(out)
    }
}

/// The type a `<!-- decl: ... -->` line names, if the line is one.
fn decl_marker(line: &str) -> Option<&str> {
    let inner = line
        .trim()
        .strip_prefix("<!--")?
        .strip_suffix("-->")?
        .trim();
    inner.strip_prefix("decl:").map(str::trim)
}

/// Rewrites every inline link destination on one line, outside code spans.
fn rewrite_inline(
    line: &str,
    dest: &dyn Fn(&str) -> Result<String, String>,
) -> Result<String, String> {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut copied = 0;
    while i < b.len() {
        if b[i] == b'`' {
            let n = b[i..].iter().take_while(|&&c| c == b'`').count();
            // The closing run of exactly n backticks, if any.
            let mut j = i + n;
            let mut close = None;
            while j < b.len() {
                if b[j] == b'`' {
                    let m = b[j..].iter().take_while(|&&c| c == b'`').count();
                    if m == n {
                        close = Some(j + m);
                        break;
                    }
                    j += m;
                } else {
                    j += 1;
                }
            }
            i = close.unwrap_or(i + n);
            continue;
        }
        if b[i] == b']' && b.get(i + 1) == Some(&b'(') {
            let start = i + 2;
            let mut depth = 0usize;
            let mut k = start;
            while k < b.len() {
                match b[k] {
                    b'(' => depth += 1,
                    b')' if depth == 0 => break,
                    b')' => depth -= 1,
                    _ => {}
                }
                k += 1;
            }
            if k == b.len() {
                break;
            }
            let inner = &line[start..k];
            // A destination, then optionally a title after white space.
            let lead = inner.len() - inner.trim_start().len();
            let body = inner.trim_start();
            let (raw, title) = match body.strip_prefix('<') {
                Some(r) => match r.find('>') {
                    Some(e) => (&r[..e], &r[e + 1..]),
                    None => (body, ""),
                },
                None => body.split_at(body.find(char::is_whitespace).unwrap_or(body.len())),
            };
            out.push_str(&line[copied..start]);
            out.push_str(&inner[..lead]);
            let new = dest(raw)?;
            if body.starts_with('<') {
                let _ = write!(out, "<{new}>");
            } else {
                out.push_str(&new);
            }
            out.push_str(title);
            copied = k;
            i = k;
            continue;
        }
        i += 1;
    }
    out.push_str(&line[copied..]);
    Ok(out)
}

/// A rendered engineering page.
pub struct Rendered {
    pub title: String,
    pub description: String,
    pub text: String,
}

/// The index of the engineering pages, at sidebar position 0: a table of
/// the pages, `(file name without extension, page)` each, in sidebar order.
/// It is written when there are no pages too, so the sidebar's section is
/// never empty, which Quarto does not render.
pub fn index_page(pages: &[(String, Rendered)]) -> String {
    let mut out = String::new();
    front_matter(
        &mut out,
        "Engineering",
        Some("How the larger features of camdl are built, and why."),
        0,
    );
    out.push_str(
        "Each page explains one feature across the crates and modules it spans. Its \
         links to the code and the declarations it shows are taken from the code when \
         the site is built, so they follow the code as it changes.\n\n",
    );
    if pages.is_empty() {
        out.push_str("There are no engineering pages yet.\n");
        return out;
    }
    out.push_str("| Page | Summary |\n|:--|:--|\n");
    for (slug, p) in pages {
        let _ = writeln!(
            out,
            "| [{}]({slug}.qmd) | {} |",
            p.title.replace('|', "\\|"),
            p.description.replace('|', "\\|")
        );
    }
    out
}

/// One engineering page, rendered: its front matter at sidebar position
/// `order`, then its body with references resolved, declarations embedded
/// and relative links rewritten. `name` is the page's path, for messages.
pub fn render_page(name: &str, text: &str, order: i64, cx: &Context) -> Result<Rendered, String> {
    let at = |line: usize, e: String| format!("{name}:{line}: {e}");
    let lines: Vec<&str> = text.lines().collect();
    let first = lines
        .iter()
        .position(|l| !l.trim().is_empty())
        .ok_or_else(|| at(1, "the page is empty".to_string()))?;
    let title = lines[first]
        .strip_prefix("# ")
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            at(
                first + 1,
                "an engineering page opens with its title, a line `# Title`".to_string(),
            )
        })?;
    let refdef = Regex::new(r"^( {0,3}\[[^\]^][^\]]*\]:[ \t]+)(\S+)(.*)$").expect("valid regex");
    let mut body = String::new();
    let mut fence: Option<(char, usize)> = None;
    for (k, line) in lines.iter().enumerate().skip(first + 1) {
        let n = k + 1;
        if let Some((c, len)) = fence {
            if fence_marker(line).is_some_and(|(c2, n2)| c2 == c && n2 >= len)
                && line.trim().chars().all(|x| x == c)
            {
                fence = None;
            }
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if let Some(m) = fence_marker(line) {
            fence = Some(m);
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if line.starts_with("# ") {
            return Err(at(
                n,
                "a second `# ` heading; the page's title is its only one".to_string(),
            ));
        }
        if let Some(target) = decl_marker(line) {
            body.push_str(&cx.embed(target).map_err(|e| at(n, e))?);
            continue;
        }
        if line.contains("<!-- decl:") {
            return Err(at(
                n,
                "a declaration marker `<!-- decl: ... -->` must be alone on its line".to_string(),
            ));
        }
        let rewritten = match refdef.captures(line) {
            Some(c) => format!(
                "{}{}{}",
                &c[1],
                cx.destination(&c[2]).map_err(|e| at(n, e))?,
                &c[3]
            ),
            None => rewrite_inline(line, &|d| cx.destination(d)).map_err(|e| at(n, e))?,
        };
        body.push_str(&rewritten);
        body.push('\n');
    }
    let opens_with_prose = body
        .lines()
        .find(|l| !l.trim().is_empty())
        .is_some_and(|l| !l.starts_with(['#', ':', '<', '|']));
    let split = split_first_sentence(body.trim_start()).filter(|_| opens_with_prose);
    let Some((description, rest)) = split else {
        return Err(at(
            first + 2,
            "the title is followed by a paragraph, whose first sentence is the page's \
             description"
                .to_string(),
        ));
    };
    if description.contains("](") {
        return Err(at(
            first + 2,
            "the first sentence is the page's description, shown under its title, and may \
             not hold a link"
                .to_string(),
        ));
    }
    let mut out = String::new();
    front_matter(&mut out, title, Some(&description), order);
    out.push_str(rest.trim_end());
    out.push('\n');
    Ok(Rendered {
        title: title.to_string(),
        description,
        text: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_climb_to_the_common_directory() {
        let p = |s: &str| PathBuf::from(s);
        assert_eq!(
            relative(&p("/r/docs/internals/engineering"), &p("/r/docs/run.md")),
            "../../run.md"
        );
        assert_eq!(
            relative(
                &p("/r/docs/internals/engineering"),
                &p("/r/docs/internals/runtime")
            ),
            "../runtime"
        );
        assert_eq!(relative(&p("/r/a"), &p("/r/a")), ".");
    }

    #[test]
    fn code_spans_are_left_alone() {
        let f = |d: &str| Ok(format!("<{d}>"));
        assert_eq!(
            rewrite_inline("see `[a](b)` and [c](d \"t\") and [e](<f g>)", &f).unwrap(),
            "see `[a](b)` and [c](<d> \"t\") and [e](<<f g>>)"
        );
    }

    #[test]
    fn a_decl_marker_is_a_whole_line() {
        assert_eq!(decl_marker("<!-- decl: rust:a::B -->"), Some("rust:a::B"));
        assert_eq!(
            decl_marker("  <!--decl:ocaml:Ir.t-->  "),
            Some("ocaml:Ir.t")
        );
        assert_eq!(decl_marker("<!-- note -->"), None);
    }
}
