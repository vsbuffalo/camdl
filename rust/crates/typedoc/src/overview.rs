//! The overview page: an odoc standalone page (`.mld`) or a Markdown page
//! (`.md`) that gives a set of internals pages its title, its index prose,
//! and the groups the modules are listed under. The OCaml compiler's is odoc,
//! as its doc comments are; the Rust runtime's is Markdown, as its doc
//! comments are. Both are held to the same rules.
//!
//! The Markdown page's one `# ...` heading, first in the file, is the title,
//! and its opening paragraph's first sentence the description. Each
//! `## ...` heading opens a group (`## Heading {#label}` gives its anchor),
//! and the group's one `<!-- modules: a b c -->` line lists its top-level
//! modules in order; the HTML comment keeps the list out of a plain Markdown
//! preview. Text in fenced code blocks is never structure.
//!
//! The odoc page:
//!
//! The page is ordinary odoc. Its one `{0 ...}` heading, first in the file,
//! is the title. Its opening paragraph's first sentence is the page
//! description. Each `{1 ...}` heading opens a group, and the group's one
//! `{!modules: A B C}` directive lists the group's top-level modules in
//! order; the module table is drawn where the directive sits. Everything else
//! is prose, converted by [`crate::odoc::page_to_markdown`] when the index is
//! written. A page that breaks any of these rules is refused, naming the line.

use crate::markdown::first_sentence;
use crate::odoc;

/// The markup an overview page is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Odoc,
    Markdown,
}

/// A parsed overview page. The text fields hold source in [`Overview::format`].
#[derive(Debug, Clone)]
pub struct Overview {
    /// The markup of the text fields.
    pub format: Format,
    /// The text of the `{0 ...}` heading.
    pub title: String,
    /// The prose between the title and the first `{1 ...}` heading.
    pub intro: String,
    /// The `{1 ...}` sections, in page order.
    pub groups: Vec<Group>,
}

/// A `{1 ...}` section of the overview: a heading, prose, and the list of
/// modules the index shows as a table at the directive's place.
#[derive(Debug, Clone)]
pub struct Group {
    /// The heading's text.
    pub title: String,
    /// The heading's label, from `{1:label ...}`, used as its anchor.
    pub label: Option<String>,
    /// The prose between the heading and the `{!modules: ...}` directive.
    pub before: String,
    /// The top-level modules listed, by name, in order. A module nested in
    /// one of them (`Ast.Located` in `Ast`) is shown on its parent's page and
    /// is not listed.
    pub modules: Vec<String>,
    /// The prose between the directive and the next `{1 ...}` heading.
    pub after: String,
}

/// A top-level construct the structure of the page depends on.
enum Event {
    Heading {
        level: u8,
        label: Option<String>,
        text: String,
        start: usize,
        end: usize,
    },
    Modules {
        names: Vec<String>,
        start: usize,
        end: usize,
    },
}

impl Event {
    fn start(&self) -> usize {
        match self {
            Event::Heading { start, .. } | Event::Modules { start, .. } => *start,
        }
    }
}

const MODULES_DIRECTIVE: &str = "{!modules:";

/// The opening of a Markdown overview's module list.
const MD_MODULES: &str = "<!-- modules:";

/// The fence a Markdown line opens, if it does: its character and length.
fn md_fence(line: &str) -> Option<(char, usize)> {
    let t = line.trim_start();
    let c = t.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let n = t.chars().take_while(|x| *x == c).count();
    (n >= 3).then_some((c, n))
}

impl Overview {
    /// Reads an overview page from its odoc source.
    pub fn parse(src: &str) -> Result<Overview, String> {
        let s: Vec<char> = src.chars().collect();
        let events = scan(&s)?;
        let text = |a: usize, b: usize| -> String { s[a..b].iter().collect() };
        let line = |i: usize| 1 + s[..i].iter().filter(|c| **c == '\n').count();

        let (title, title_end) = match events.first() {
            Some(Event::Heading {
                level: 0,
                text: t,
                start,
                end,
                ..
            }) => {
                if !text(0, *start).trim().is_empty() {
                    return Err(format!(
                        "line {}: text before the {{0 ...}} title heading; the title comes first",
                        line(0)
                    ));
                }
                (t.trim().to_string(), *end)
            }
            _ => {
                let at = events.first().map_or(String::new(), |e| {
                    format!(
                        " (the first heading or directive is on line {})",
                        line(e.start())
                    )
                });
                return Err(format!(
                    "the overview does not open with a {{0 ...}} title heading{at}"
                ));
            }
        };
        for e in &events[1..] {
            if let Event::Heading {
                level: 0, start, ..
            } = e
            {
                return Err(format!(
                    "line {}: a second {{0 ...}} heading; the page has exactly one title",
                    line(*start)
                ));
            }
        }

        // Split at the `{1 ...}` headings.
        let sections: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, Event::Heading { level: 1, .. }))
            .map(|(k, _)| k)
            .collect();
        let first_section = sections.first().copied().unwrap_or(events.len());
        for e in &events[1..first_section] {
            if let Event::Modules { start, .. } = e {
                return Err(format!(
                    "line {}: a {MODULES_DIRECTIVE} ...}} list before the first {{1 ...}} \
                     heading; every list belongs to a section",
                    line(*start)
                ));
            }
        }
        let intro_end = events.get(first_section).map_or(s.len(), Event::start);
        let intro = text(title_end, intro_end);

        let mut groups = Vec::new();
        for (n, &k) in sections.iter().enumerate() {
            let next = sections.get(n + 1).copied().unwrap_or(events.len());
            let Event::Heading {
                label,
                text: heading,
                start,
                end: body_start,
                ..
            } = &events[k]
            else {
                unreachable!("sections index headings");
            };
            let body_end = events.get(next).map_or(s.len(), Event::start);
            let lists: Vec<(&[String], usize, usize)> = events[k + 1..next]
                .iter()
                .filter_map(|e| match e {
                    Event::Modules { names, start, end } => Some((names.as_slice(), *start, *end)),
                    Event::Heading { .. } => None,
                })
                .collect();
            let heading = heading.trim().to_string();
            let (names, d_start, d_end) = match lists.as_slice() {
                [one] => *one,
                [] => {
                    return Err(format!(
                        "line {}: section \"{heading}\" has no {MODULES_DIRECTIVE} ...}} list",
                        line(*start)
                    ))
                }
                many => {
                    let at: Vec<String> =
                        many.iter().map(|(_, a, _)| line(*a).to_string()).collect();
                    return Err(format!(
                        "line {}: section \"{heading}\" has {} {MODULES_DIRECTIVE} ...}} lists \
                         (lines {}); a section lists its modules once",
                        line(*start),
                        many.len(),
                        at.join(", ")
                    ));
                }
            };
            if names.is_empty() {
                return Err(format!(
                    "line {}: section \"{heading}\" has an empty {MODULES_DIRECTIVE} ...}} list",
                    line(d_start)
                ));
            }
            groups.push(Group {
                title: heading,
                label: label.clone(),
                before: text(*body_start, d_start),
                modules: names.to_vec(),
                after: text(d_end, body_end),
            });
        }

        let unresolved = |_: &str| None;
        if first_sentence(&odoc::page_to_markdown(&intro, &unresolved)).is_none() {
            return Err(format!(
                "line {}: the overview's first paragraph, under the title, gives the page's \
                 description, but the text there opens with no sentence",
                line(title_end)
            ));
        }

        Ok(Overview {
            format: Format::Odoc,
            title,
            intro,
            groups,
        })
    }

    /// Reads an overview page from its Markdown source.
    pub fn parse_markdown(src: &str) -> Result<Overview, String> {
        let lines: Vec<&str> = src.split('\n').collect();
        let mut fence: Option<(char, usize)> = None;
        let mut title: Option<String> = None;
        let mut intro = String::new();
        // Each group, the line of its heading, and the lines of its lists.
        let mut groups: Vec<(Group, usize, Vec<usize>)> = Vec::new();
        for (n, &line) in lines.iter().enumerate() {
            let lineno = n + 1;
            let mut prose = true;
            if let Some((c, k)) = fence {
                let t = line.trim();
                if t.chars().take_while(|x| *x == c).count() >= k && t.chars().all(|x| x == c) {
                    fence = None;
                }
            } else if let Some(m) = md_fence(line) {
                fence = Some(m);
            } else if let Some(t) = line.strip_prefix("# ") {
                if title.is_some() {
                    return Err(format!(
                        "line {lineno}: a second \"# ...\" heading; the page has exactly one title"
                    ));
                }
                title = Some(t.trim().to_string());
                prose = false;
            } else if let Some(h) = line.strip_prefix("## ") {
                let h = h.trim();
                let (text, label) = match h.strip_suffix('}').and_then(|r| r.rsplit_once("{#")) {
                    Some((t, l)) => (t.trim().to_string(), Some(l.trim().to_string())),
                    None => (h.to_string(), None),
                };
                groups.push((
                    Group {
                        title: text,
                        label,
                        before: String::new(),
                        modules: Vec::new(),
                        after: String::new(),
                    },
                    lineno,
                    Vec::new(),
                ));
                prose = false;
            } else if let Some(names) = line
                .trim()
                .strip_prefix(MD_MODULES)
                .and_then(|r| r.strip_suffix("-->"))
            {
                let Some((g, _, lists)) = groups.last_mut() else {
                    return Err(format!(
                        "line {lineno}: a {MD_MODULES} ... --> list before the first \"## ...\" \
                         heading; every list belongs to a section"
                    ));
                };
                lists.push(lineno);
                g.modules = names.split_whitespace().map(str::to_string).collect();
                prose = false;
            } else if line.trim_start().starts_with(MD_MODULES) {
                return Err(format!(
                    "line {lineno}: a module list must be one line, {MD_MODULES} a b c -->"
                ));
            }
            if title.is_none() {
                if !line.trim().is_empty() {
                    return Err(format!(
                        "the overview does not open with a \"# ...\" title heading (line {lineno} \
                         comes first)"
                    ));
                }
                continue;
            }
            if !prose {
                continue;
            }
            let target = match groups.last_mut() {
                None => &mut intro,
                Some((g, _, lists)) if lists.is_empty() => &mut g.before,
                Some((g, _, _)) => &mut g.after,
            };
            target.push_str(line);
            target.push('\n');
        }
        let Some(title) = title else {
            return Err("the overview does not open with a \"# ...\" title heading".to_string());
        };
        let mut out = Vec::new();
        for (g, at, lists) in groups {
            match lists.as_slice() {
                [_] => {}
                [] => {
                    return Err(format!(
                        "line {at}: section \"{}\" has no {MD_MODULES} ... --> list",
                        g.title
                    ))
                }
                many => {
                    let l: Vec<String> = many.iter().map(usize::to_string).collect();
                    return Err(format!(
                        "line {at}: section \"{}\" has {} {MD_MODULES} ... --> lists (lines {}); \
                         a section lists its modules once",
                        g.title,
                        many.len(),
                        l.join(", ")
                    ));
                }
            }
            if g.modules.is_empty() {
                return Err(format!(
                    "line {}: section \"{}\" has an empty {MD_MODULES} ... --> list",
                    lists[0], g.title
                ));
            }
            out.push(g);
        }
        if first_sentence(&crate::markdown::tidy(&intro)).is_none() {
            return Err(
                "the overview's first paragraph, under the title, gives the page's description, \
                 but the text there opens with no sentence"
                    .to_string(),
            );
        }
        Ok(Overview {
            format: Format::Markdown,
            title,
            intro,
            groups: out,
        })
    }

    /// The listed module names in page order: groups in order, and modules in
    /// order within each group.
    pub fn module_names(&self) -> impl Iterator<Item = &str> {
        self.groups
            .iter()
            .flat_map(|g| g.modules.iter().map(String::as_str))
    }
}

fn starts(s: &[char], i: usize, pat: &str) -> bool {
    pat.chars()
        .enumerate()
        .all(|(k, c)| s.get(i + k) == Some(&c))
}

/// The index just past the first `pat` at or after `i`, or the end of input.
fn past(s: &[char], mut i: usize, pat: &str) -> usize {
    while i < s.len() {
        if starts(s, i, pat) {
            return i + pat.chars().count();
        }
        i += 1;
    }
    s.len()
}

/// When a construct whose content is not markup starts at `i` (an escape, a
/// code span or a code block), the index just past it. Unterminated ones run
/// to the end of input, as the converter reads them.
fn skip_verbatim(s: &[char], i: usize) -> Option<usize> {
    match s[i] {
        '\\' => Some((i + 2).min(s.len())),
        '[' => {
            let mut depth = 0usize;
            let mut j = i;
            while j < s.len() {
                match s[j] {
                    '\\' => j += 1,
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            Some(s.len())
        }
        '`' => {
            let n = s[i..].iter().take_while(|c| **c == '`').count();
            let mut j = i + n;
            while j < s.len() {
                if s[j] == '`' {
                    let m = s[j..].iter().take_while(|c| **c == '`').count();
                    if m == n {
                        return Some(j + m);
                    }
                    j += m;
                } else {
                    j += 1;
                }
            }
            Some(i + n)
        }
        '{' if starts(s, i, "{[") => Some(past(s, i + 2, "]}")),
        '{' if starts(s, i, "{@") => Some(past(s, past(s, i + 2, "["), "]}")),
        '{' if starts(s, i, "{v") && s.get(i + 2).is_some_and(|c| c.is_whitespace()) => {
            Some(past(s, i + 2, "v}"))
        }
        '{' if starts(s, i, "{%") => Some(past(s, i + 2, "%}")),
        _ => None,
    }
}

/// A heading tag at `i`: `{N text}` or `{N:label text}` with `N` from 0 to 5.
/// Returns the level, the label and the index where the text starts.
fn heading_open(s: &[char], i: usize) -> Option<(u8, Option<String>, usize)> {
    let level = s.get(i + 1)?.to_digit(10).filter(|d| *d <= 5)? as u8;
    match s.get(i + 2)? {
        c if c.is_whitespace() => Some((level, None, i + 2)),
        ':' => {
            let label: String = s[i + 3..]
                .iter()
                .take_while(|c| !c.is_whitespace() && **c != '}')
                .collect();
            let after = i + 3 + label.chars().count();
            (!label.is_empty()).then_some((level, Some(label), after))
        }
        _ => None,
    }
}

/// The index of the `}` closing a tag whose content starts at `i`.
fn closing_brace(s: &[char], mut i: usize) -> Option<usize> {
    let mut depth = 1usize;
    while i < s.len() {
        if let Some(j) = skip_verbatim(s, i) {
            i = j;
            continue;
        }
        match s[i] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// The headings and `{!modules: ...}` directives of a page, in order. Text
/// inside code spans and code blocks is skipped, so an example that shows the
/// markup is not read as structure.
fn scan(s: &[char]) -> Result<Vec<Event>, String> {
    let line = |i: usize| 1 + s[..i].iter().filter(|c| **c == '\n').count();
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut i = 0;
    while i < s.len() {
        if let Some(j) = skip_verbatim(s, i) {
            i = j;
            continue;
        }
        match s[i] {
            '{' if starts(s, i, MODULES_DIRECTIVE) => {
                if depth > 0 {
                    return Err(format!(
                        "line {}: a {MODULES_DIRECTIVE} ...}} list inside other markup",
                        line(i)
                    ));
                }
                let body = i + MODULES_DIRECTIVE.chars().count();
                let Some(close) = s[body..].iter().position(|c| *c == '}') else {
                    return Err(format!(
                        "line {}: a {MODULES_DIRECTIVE} ...}} list is never closed",
                        line(i)
                    ));
                };
                let names: String = s[body..body + close].iter().collect();
                out.push(Event::Modules {
                    names: names.split_whitespace().map(str::to_string).collect(),
                    start: i,
                    end: body + close + 1,
                });
                i = body + close + 1;
            }
            '{' => match heading_open(s, i) {
                Some((level, label, text_start)) if depth == 0 => {
                    let Some(close) = closing_brace(s, text_start) else {
                        return Err(format!(
                            "line {}: a {{{level} ...}} heading is never closed",
                            line(i)
                        ));
                    };
                    out.push(Event::Heading {
                        level,
                        label,
                        text: s[text_start..close].iter().collect(),
                        start: i,
                        end: close + 1,
                    });
                    i = close + 1;
                }
                _ => {
                    depth += 1;
                    i += 1;
                }
            },
            '}' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            _ => i += 1,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "\
{0 The compiler}

How camdlc reads a model. It has {b stages}, as in {!Ast.model}.

{[
{1 Not a heading} {!modules: Not_a_list}
]}

{1:source Reading the source}

The parser builds the tree.

{!modules: Ast Parser_errors}

After the table.

{2 A subsection}

More.

{1 The IR}

{!modules:
  Ir
  Serde}
";

    #[test]
    fn title_intro_and_groups() {
        let o = Overview::parse(GOOD).unwrap();
        assert_eq!(o.title, "The compiler");
        assert!(o.intro.contains("How camdlc reads a model."));
        assert!(
            o.intro.contains("{1 Not a heading} {!modules: Not_a_list}"),
            "markup inside a code block is prose: {}",
            o.intro
        );
        assert_eq!(o.groups.len(), 2);
        let g = &o.groups[0];
        assert_eq!(g.title, "Reading the source");
        assert_eq!(g.label.as_deref(), Some("source"));
        assert_eq!(g.before.trim(), "The parser builds the tree.");
        assert_eq!(g.modules, ["Ast", "Parser_errors"]);
        assert!(g.after.contains("After the table."));
        assert!(
            g.after.contains("{2 A subsection}"),
            "a {{2}} heading stays prose"
        );
        assert_eq!(o.groups[1].title, "The IR");
        assert_eq!(o.groups[1].label, None);
        assert_eq!(o.groups[1].modules, ["Ir", "Serde"]);
        assert_eq!(
            o.module_names().collect::<Vec<_>>(),
            ["Ast", "Parser_errors", "Ir", "Serde"]
        );
    }

    fn err(src: &str) -> String {
        Overview::parse(src).unwrap_err()
    }

    #[test]
    fn a_missing_title_fails() {
        let e = err("Some text.\n\n{1 G}\n{!modules: A}\n");
        assert!(e.contains("does not open with a {0 ...} title"), "{e}");
        let e = err("Text first.\n{0 T}\nBody.\n{1 G}\n{!modules: A}\n");
        assert!(e.contains("line 1: text before the {0 ...} title"), "{e}");
    }

    #[test]
    fn a_second_title_fails() {
        let e = err("{0 T}\nBody.\n{1 G}\n{!modules: A}\n{0 Again}\n");
        assert!(e.contains("line 5: a second {0 ...} heading"), "{e}");
    }

    #[test]
    fn a_section_without_a_list_fails() {
        let e = err("{0 T}\nBody.\n{1 G}\nText.\n{1 H}\n{!modules: A}\n");
        assert!(
            e.contains("line 3: section \"G\" has no {!modules: ...} list"),
            "{e}"
        );
    }

    #[test]
    fn a_section_with_two_lists_fails() {
        let e = err("{0 T}\nBody.\n{1 G}\n{!modules: A}\n\n{!modules: B}\n");
        assert!(
            e.contains("line 3: section \"G\" has 2 {!modules: ...} lists (lines 4, 6)"),
            "{e}"
        );
    }

    #[test]
    fn a_list_outside_a_section_fails() {
        let e = err("{0 T}\nBody.\n{!modules: A}\n{1 G}\n{!modules: B}\n");
        assert!(
            e.contains("line 3: a {!modules: ...} list before the first"),
            "{e}"
        );
    }

    #[test]
    fn an_empty_list_fails() {
        let e = err("{0 T}\nBody.\n{1 G}\n{!modules: }\n");
        assert!(e.contains("empty {!modules: ...} list"), "{e}");
    }

    #[test]
    fn an_overview_without_an_opening_sentence_fails() {
        let e = err("{0 T}\n{[ code ]}\n{1 G}\n{!modules: A}\n");
        assert!(e.contains("opens with no sentence"), "{e}");
    }

    const GOOD_MD: &str = "\
# The runtime

How camdl runs a model. It reads [`Model`].

```
## Not a heading
<!-- modules: not_a_list -->
```

## Types first {#types}

The IR.

<!-- modules: ir declared -->

After the table.

### A subsection

## Engine

<!-- modules: sim -->
";

    #[test]
    fn markdown_title_intro_and_groups() {
        let o = Overview::parse_markdown(GOOD_MD).unwrap();
        assert_eq!(o.format, Format::Markdown);
        assert_eq!(o.title, "The runtime");
        assert!(
            o.intro.contains("<!-- modules: not_a_list -->"),
            "{}",
            o.intro
        );
        assert_eq!(o.groups.len(), 2);
        let g = &o.groups[0];
        assert_eq!(g.title, "Types first");
        assert_eq!(g.label.as_deref(), Some("types"));
        assert_eq!(g.before.trim(), "The IR.");
        assert_eq!(g.modules, ["ir", "declared"]);
        assert!(g.after.contains("### A subsection"), "{}", g.after);
        assert_eq!(
            o.module_names().collect::<Vec<_>>(),
            ["ir", "declared", "sim"]
        );
    }

    fn md_err(src: &str) -> String {
        Overview::parse_markdown(src).unwrap_err()
    }

    #[test]
    fn markdown_structure_errors_name_the_line() {
        let e = md_err("Text.\n# T\nBody.\n## G\n<!-- modules: a -->\n");
        assert!(e.contains("does not open with a \"# ...\" title"), "{e}");
        let e = md_err("# T\nBody.\n## G\n<!-- modules: a -->\n# Again\n");
        assert!(e.contains("line 5: a second"), "{e}");
        let e = md_err("# T\nBody.\n## G\nText.\n## H\n<!-- modules: a -->\n");
        assert!(e.contains("line 3: section \"G\" has no"), "{e}");
        let e = md_err("# T\nBody.\n## G\n<!-- modules: a -->\n<!-- modules: b -->\n");
        assert!(
            e.contains("has 2 <!-- modules: ... --> lists (lines 4, 5)"),
            "{e}"
        );
        let e = md_err("# T\nBody.\n<!-- modules: a -->\n## G\n<!-- modules: b -->\n");
        assert!(
            e.contains("line 3: a <!-- modules: ... --> list before the first"),
            "{e}"
        );
        let e = md_err("# T\nBody.\n## G\n<!-- modules: -->\n");
        assert!(e.contains("empty"), "{e}");
        let e = md_err("# T\n```\ncode\n```\n## G\n<!-- modules: a -->\n");
        assert!(e.contains("opens with no sentence"), "{e}");
        let e = md_err("# T\nBody.\n## G\n<!-- modules: a\nb -->\n");
        assert!(e.contains("must be one line"), "{e}");
    }

    #[test]
    fn unclosed_markup_fails() {
        let e = err("{0 T}\nBody.\n{1 G\n");
        assert!(
            e.contains("line 3: a {1 ...} heading is never closed"),
            "{e}"
        );
        let e = err("{0 T}\nBody.\n{1 G}\n{!modules: A\n");
        assert!(e.contains("never closed"), "{e}");
    }
}
