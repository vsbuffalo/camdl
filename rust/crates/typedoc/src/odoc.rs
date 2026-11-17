//! Converts odoc markup, the language of OCaml doc comments, to Markdown.
//!
//! The conversion covers what camdl's doc comments use:
//!
//! - `[code]` becomes a code span; brackets inside it nest, as in `[f [x]]`.
//! - `{[ ... ]}` becomes an `ocaml` code block, `{@lang[ ... ]}` a `lang` code
//!   block, and `{v ... v}` a code block with no language.
//! - `{b x}` becomes bold, `{i x}` and `{e x}` italic.
//! - `{!ref}` becomes a link when the resolver finds a target for it, and a
//!   code span otherwise; `{{!ref} text}` and `{{:url} text}` become links
//!   with that text.
//! - `{ul {li a} {li b}}` and `{ol ...}` become lists, as do lines starting
//!   `- ` (unordered) or `+ ` (ordered).
//! - `{0 ...}` to `{5 ...}` become bold paragraphs in a doc comment, which
//!   sits under the page's own headings, and Markdown headings one level
//!   deeper (`{1}` is `##`) in a standalone page ([`page_to_markdown`]).
//! - Markdown code spans written with backticks, and `*emphasis*`, pass
//!   through unchanged.
//!
//! Any other `{tag ...}` keeps its content and loses the tag.

use crate::markdown::{code_span, dedent, fence, push_text_char, tidy};

/// Resolves the target of a `{!ref}`: a type or module path, with any odoc
/// kind prefix (`type:`, `module-`) removed. Returns the link target.
pub type Resolver<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Converts one odoc doc comment to Markdown.
pub fn to_markdown(src: &str, resolve: &Resolver) -> String {
    convert(src, resolve, false)
}

/// Converts the text of an odoc standalone page (`.mld`) to Markdown, where
/// headings are the page's own: `{1 ...}` becomes `##`, `{2 ...}` `###`.
pub fn page_to_markdown(src: &str, resolve: &Resolver) -> String {
    convert(src, resolve, true)
}

fn convert(src: &str, resolve: &Resolver, page: bool) -> String {
    let text = dedent(src);
    let mut conv = Conv {
        s: text.chars().collect(),
        i: 0,
        resolve,
        page,
    };
    tidy(&conv.seq(false).finish())
}

/// Markdown being built: text, with blocks set off by blank lines.
struct Out {
    buf: String,
    /// Whether the next text starts a line, where `-`, `+`, `#` and `>` mean
    /// something.
    line_start: bool,
    /// Whether whitespace is being dropped because a block just ended.
    skip_ws: bool,
}

impl Out {
    fn new() -> Out {
        Out {
            buf: String::new(),
            line_start: true,
            skip_ws: false,
        }
    }

    fn last(&self) -> Option<char> {
        self.buf.chars().next_back()
    }

    /// Appends text that is already Markdown.
    fn raw(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.skip_ws = false;
        self.line_start = false;
        self.buf.push_str(s);
    }

    /// Appends a block, separated from what surrounds it by blank lines.
    fn block(&mut self, s: &str) {
        let trimmed = self.buf.trim_end_matches([' ', '\t']).len();
        self.buf.truncate(trimmed);
        if !self.buf.is_empty() {
            while !self.buf.ends_with("\n\n") {
                self.buf.push('\n');
            }
        }
        self.buf.push_str(s);
        self.buf.push_str("\n\n");
        self.line_start = true;
        self.skip_ws = true;
    }

    fn finish(self) -> String {
        self.buf
    }
}

struct Conv<'a> {
    s: Vec<char>,
    i: usize,
    resolve: &'a Resolver<'a>,
    /// Whether the text is a standalone page, whose headings are Markdown
    /// headings, rather than a doc comment.
    page: bool,
}

impl Conv<'_> {
    fn peek(&self, k: usize) -> Option<char> {
        self.s.get(self.i + k).copied()
    }

    fn starts_with(&self, pat: &str) -> bool {
        pat.chars()
            .enumerate()
            .all(|(k, c)| self.peek(k) == Some(c))
    }

    /// The raw text up to the first occurrence of `end`, which is consumed.
    /// Runs to the end of input when `end` never occurs.
    fn until(&mut self, end: &str) -> String {
        let mut out = String::new();
        while self.i < self.s.len() && !self.starts_with(end) {
            out.push(self.s[self.i]);
            self.i += 1;
        }
        self.i = (self.i + end.chars().count()).min(self.s.len());
        out
    }

    /// The tag name after a `{`: the characters up to whitespace or a brace.
    fn tag_name(&self) -> String {
        self.s[self.i + 1..]
            .iter()
            .take_while(|c| !c.is_whitespace() && **c != '}' && **c != '{' && **c != '[')
            .collect()
    }

    /// Converts a sequence of text and markup. With `in_brace`, stops at the
    /// `}` closing the enclosing tag and consumes it.
    fn seq(&mut self, in_brace: bool) -> Out {
        let mut out = Out::new();
        while let Some(c) = self.peek(0) {
            match c {
                '}' if in_brace => {
                    self.i += 1;
                    return out;
                }
                '\\' => {
                    let next = self.peek(1);
                    self.i += if next.is_some() { 2 } else { 1 };
                    let lit = next.unwrap_or('\\');
                    out.skip_ws = false;
                    out.line_start = false;
                    push_text_char(&mut out.buf, lit, None, None);
                }
                '[' => {
                    self.i += 1;
                    let code = self.bracketed();
                    out.raw(&code_span(&code));
                }
                '`' => self.backtick_code(&mut out),
                '{' => self.tag(&mut out),
                '\n' => {
                    self.i += 1;
                    if !out.skip_ws {
                        out.buf.push('\n');
                        out.line_start = true;
                    }
                }
                c if c.is_whitespace() => {
                    self.i += 1;
                    if out.skip_ws {
                        continue;
                    }
                    if out.line_start {
                        // Indentation of four or more spaces would make an
                        // indented code block, which odoc does not have.
                        let col = out.buf.len() - out.buf.rfind('\n').map_or(0, |p| p + 1);
                        if col >= 3 {
                            continue;
                        }
                    }
                    out.buf.push(' ');
                }
                _ => {
                    self.i += 1;
                    self.text_char(&mut out, c);
                }
            }
        }
        out
    }

    /// Appends one character of prose.
    fn text_char(&mut self, out: &mut Out, c: char) {
        out.skip_ws = false;
        if out.line_start {
            out.line_start = false;
            if c == '+' && self.peek(0) == Some(' ') {
                out.buf.push_str("1.");
                return;
            }
            if c == '-' && self.peek(0) == Some(' ') {
                out.buf.push('-');
                return;
            }
            if matches!(c, '#' | '>' | '-' | '+' | '=') || (c == '*' && self.peek(0) == Some(' ')) {
                out.buf.push('\\');
                out.buf.push(c);
                return;
            }
        }
        if c == '*' {
            // Authors of camdl's odoc comments write `*word*` meaning
            // emphasis, so it is kept. Markdown reads a `*` with spaces on
            // both sides, as in `beta * S`, literally.
            out.buf.push(c);
            return;
        }
        let prev = out.last();
        push_text_char(&mut out.buf, c, prev, self.peek(0));
    }

    /// The content of `[...]` after the opening bracket, which is consumed
    /// with its closing bracket. Nested brackets balance.
    fn bracketed(&mut self) -> String {
        let mut depth = 1;
        let mut out = String::new();
        while let Some(c) = self.peek(0) {
            self.i += 1;
            match c {
                '\\' if matches!(self.peek(0), Some('[' | ']')) => {
                    out.push(self.peek(0).unwrap_or(c));
                    self.i += 1;
                    continue;
                }
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return out;
                    }
                }
                _ => {}
            }
            out.push(c);
        }
        out
    }

    /// Copies a Markdown code span written with backticks unchanged.
    fn backtick_code(&mut self, out: &mut Out) {
        let n = self.s[self.i..].iter().take_while(|c| **c == '`').count();
        let delim: String = "`".repeat(n);
        let start = self.i + n;
        let mut j = start;
        while j < self.s.len() {
            if self.s[j] == '`' {
                let m = self.s[j..].iter().take_while(|c| **c == '`').count();
                if m == n {
                    let body: String = self.s[start..j].iter().collect();
                    self.i = j + m;
                    out.raw(&format!("{delim}{body}{delim}"));
                    return;
                }
                j += m;
            } else {
                j += 1;
            }
        }
        // No closing run: the backticks are literal.
        self.i += n;
        for _ in 0..n {
            out.raw("\\`");
        }
    }

    /// Converts a `{...}` construct at the cursor.
    fn tag(&mut self, out: &mut Out) {
        if self.starts_with("{[") {
            self.i += 2;
            let body = self.until("]}");
            out.block(&fence("ocaml", &code_body(&body)));
        } else if self.starts_with("{@") {
            self.i += 2;
            let lang = self.until("[");
            let body = self.until("]}");
            out.block(&fence(lang.trim(), &code_body(&body)));
        } else if self.starts_with("{v") && self.peek(2).is_some_and(char::is_whitespace) {
            self.i += 2;
            let body = self.until("v}");
            out.block(&fence("", &code_body(&body)));
        } else if self.starts_with("{%") {
            // Raw markup for one output format: keep its text.
            self.i += 2;
            let body = self.until("%}");
            let body = body.split_once(':').map_or(body.as_str(), |(_, b)| b);
            out.raw(body.trim());
        } else if self.starts_with("{!") {
            self.i += 2;
            let target = self.until("}");
            out.raw(&self.reference(target.trim(), None));
        } else if self.starts_with("{{!") || self.starts_with("{{:") {
            self.i += 3;
            let is_url = self.s[self.i - 1] == ':';
            let target = self.until("}");
            let text = self.seq(true).finish();
            let text = text.trim();
            if is_url {
                out.raw(&format!("[{text}]({})", target.trim()));
            } else {
                out.raw(&self.reference(target.trim(), Some(text)));
            }
        } else if self.starts_with("{:") {
            self.i += 2;
            let url = self.until("}");
            out.raw(&format!("<{}>", url.trim()));
        } else {
            let name = self.tag_name();
            if let Some((level, label)) = heading_level(&name) {
                self.i += 1 + name.len();
                let inner = self.seq(true).finish();
                if self.page {
                    // A Markdown heading is one line.
                    let inner = inner.split_whitespace().collect::<Vec<_>>().join(" ");
                    let anchor = label.map_or(String::new(), |l| format!(" {{#{l}}}"));
                    out.block(&format!("{} {inner}{anchor}", "#".repeat(level + 1)));
                } else {
                    out.block(&format!("**{}**", inner.trim()));
                }
                return;
            }
            match name.as_str() {
                "b" | "i" | "e" => {
                    self.i += 1 + name.len();
                    let inner = self.seq(true).finish();
                    let inner = inner.trim();
                    if !inner.is_empty() {
                        let mark = if name == "b" { "**" } else { "*" };
                        out.raw(&format!("{mark}{inner}{mark}"));
                    }
                }
                "ul" | "ol" => {
                    self.i += 1 + name.len();
                    let list = self.list(name == "ol");
                    out.block(&list);
                }
                "" => {
                    // A `{` that opens no tag is literal.
                    self.i += 1;
                    self.text_char(out, '{');
                }
                _ => {
                    // An unknown tag: keep the content.
                    self.i += 1 + name.len();
                    let inner = self.seq(true).finish();
                    out.raw(inner.trim());
                }
            }
        }
    }

    /// The Markdown for a `{!ref}`, with `text` as the link text if given.
    fn reference(&self, target: &str, text: Option<&str>) -> String {
        let path = strip_kinds(target);
        let shown = match text {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => code_span(&path),
        };
        match (self.resolve)(&path) {
            Some(href) => format!("[{shown}]({href})"),
            None => shown,
        }
    }

    /// The items of a `{ul ...}` or `{ol ...}` after its tag, consumed with
    /// its closing brace, as a Markdown list.
    fn list(&mut self, ordered: bool) -> String {
        let mut items = Vec::new();
        loop {
            while self.peek(0).is_some_and(char::is_whitespace) {
                self.i += 1;
            }
            match self.peek(0) {
                None => break,
                Some('}') => {
                    self.i += 1;
                    break;
                }
                Some('{') if self.starts_with("{li") || self.starts_with("{-") => {
                    self.i += if self.starts_with("{li") { 3 } else { 2 };
                    let body = tidy(&self.seq(true).finish()).trim_start().to_string();
                    items.push(body);
                }
                Some(_) => {
                    // Text between items is not odoc; keep it as an item.
                    let body = tidy(&self.seq(true).finish()).trim_start().to_string();
                    if !body.is_empty() {
                        items.push(body);
                    }
                    break;
                }
            }
        }
        items
            .iter()
            .map(|body| {
                let marker = if ordered { "1. " } else { "- " };
                let pad = " ".repeat(marker.len());
                let mut lines = body.lines();
                let mut s = format!("{marker}{}", lines.next().unwrap_or(""));
                for l in lines {
                    s.push('\n');
                    if !l.is_empty() {
                        s.push_str(&pad);
                        s.push_str(l);
                    }
                }
                s
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The level and label of a heading tag's name: `1` is level 1, and
/// `1:intro` is level 1 labelled `intro`. `None` for any other name.
fn heading_level(name: &str) -> Option<(usize, Option<&str>)> {
    let (digit, label) = match name.split_once(':') {
        Some((d, l)) if !l.is_empty() => (d, Some(l)),
        Some(_) => return None,
        None => (name, None),
    };
    match digit {
        "0" | "1" | "2" | "3" | "4" | "5" => Some((digit.parse().ok()?, label)),
        _ => None,
    }
}

/// The body of a code block: its own common indentation removed, and blank
/// lines at either end dropped.
fn code_body(body: &str) -> String {
    let lines: Vec<&str> = body.lines().collect();
    let common = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut out: Vec<&str> = lines
        .iter()
        .map(|l| {
            if l.len() >= common {
                &l[common..]
            } else {
                l.trim_start()
            }
        })
        .collect();
    while out.first().is_some_and(|l| l.trim().is_empty()) {
        out.remove(0);
    }
    while out.last().is_some_and(|l| l.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// The odoc kinds a reference may be qualified with, as `type:t` or as a
/// path component `module-M`.
const REFERENCE_KINDS: &[&str] = &[
    "module",
    "modtype",
    "module-type",
    "type",
    "val",
    "class",
    "class-type",
    "exception",
    "extension",
    "field",
    "constructor",
    "method",
    "instance-variable",
    "label",
    "page",
    "section",
];

/// A reference path with its odoc kind qualifiers removed:
/// `module-Ir.type-expr` and `type:Ir.expr` are both `Ir.expr`.
fn strip_kinds(target: &str) -> String {
    let target = match target.split_once(':') {
        Some((k, rest)) if REFERENCE_KINDS.contains(&k) => rest,
        _ => target,
    };
    target
        .split('.')
        .map(|seg| {
            REFERENCE_KINDS
                .iter()
                .find_map(|k| seg.strip_prefix(k).and_then(|r| r.strip_prefix('-')))
                .unwrap_or(seg)
        })
        .collect::<Vec<_>>()
        .join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(s: &str) -> String {
        let resolve = |p: &str| match p {
            "Ir.expr" => Some("ir.qmd#t-expr".to_string()),
            "stoich_ref" => Some("#t-stoich_ref".to_string()),
            _ => None,
        };
        to_markdown(s, &resolve)
    }

    #[test]
    fn inline_code_nests_brackets() {
        assert_eq!(conv("Calls [f [x]] then [g]."), "Calls `f [x]` then `g`.");
        assert_eq!(conv("The [`A] tag."), "The `` `A `` tag.");
    }

    #[test]
    fn code_blocks_get_fences_and_lose_indentation() {
        assert_eq!(
            conv("Example:\n    {[\n      let x = [1]\n    ]}\n    Done."),
            "Example:\n\n```ocaml\nlet x = [1]\n```\n\nDone."
        );
        assert_eq!(
            conv("{@camdl[\n  infection : S --> I @ beta * S\n]}"),
            "```camdl\ninfection : S --> I @ beta * S\n```"
        );
        assert_eq!(conv("{v\n a [b] *c*\nv}"), "```\na [b] *c*\n```");
    }

    #[test]
    fn emphasis() {
        assert_eq!(
            conv("{b bold} and {i it} and {e em}."),
            "**bold** and *it* and *em*."
        );
    }

    #[test]
    fn references_link_when_resolved() {
        assert_eq!(conv("See {!Ir.expr}."), "See [`Ir.expr`](ir.qmd#t-expr).");
        assert_eq!(
            conv("See {!type:stoich_ref}."),
            "See [`stoich_ref`](#t-stoich_ref)."
        );
        assert_eq!(
            conv("See {!module-Ir.type-expr}."),
            "See [`Ir.expr`](ir.qmd#t-expr)."
        );
        assert_eq!(conv("See {!Missing.t}."), "See `Missing.t`.");
        assert_eq!(
            conv("See {{!Ir.expr} the IR}."),
            "See [the IR](ir.qmd#t-expr)."
        );
        assert_eq!(conv("At {{:https://x.org} x}."), "At [x](https://x.org).");
    }

    #[test]
    fn explicit_lists() {
        assert_eq!(
            conv("Two:\n{ul {li one [a]}\n    {li two}}\nEnd."),
            "Two:\n\n- one `a`\n- two\n\nEnd."
        );
        assert_eq!(conv("{ol {- a} {- b}}"), "1. a\n1. b");
    }

    #[test]
    fn list_item_with_a_code_block_stays_inside_the_item() {
        assert_eq!(
            conv("{ul {li see\n {[x]}}}"),
            "- see\n\n  ```ocaml\n  x\n  ```"
        );
    }

    #[test]
    fn dash_lines_stay_lists_with_a_blank_line_before() {
        assert_eq!(
            conv("Cases:\n    - [A] one\n      more\n    - [B] two"),
            "Cases:\n\n- `A` one\n  more\n- `B` two"
        );
        assert_eq!(
            conv("Steps:\n+ first\n+ second"),
            "Steps:\n\n1. first\n1. second"
        );
    }

    #[test]
    fn markdown_specials_in_prose_are_escaped() {
        assert_eq!(
            conv("w_A * rate, <x>, $5, @param, a_b, _c"),
            "w_A * rate, \\<x>, \\$5, \\@param, a_b, \\_c"
        );
        assert_eq!(conv("x\n# not heading"), "x\n\\# not heading");
        assert_eq!(conv("x\n* not a list"), "x\n\\* not a list");
    }

    #[test]
    fn asterisk_emphasis_passes_through() {
        assert_eq!(conv("An *estimated* one"), "An *estimated* one");
        assert_eq!(conv("*Only* this"), "*Only* this");
    }

    #[test]
    fn backtick_code_passes_through() {
        assert_eq!(
            conv("A `+`-separated list of `[x]`."),
            "A `+`-separated list of `[x]`."
        );
    }

    #[test]
    fn unknown_tags_keep_content() {
        assert_eq!(conv("{custom kept [x]} text"), "kept `x` text");
        assert_eq!(conv("{1 Heading}\nBody."), "**Heading**\n\nBody.");
    }

    #[test]
    fn page_headings_are_markdown_headings() {
        let resolve = |_: &str| None;
        assert_eq!(
            page_to_markdown("Intro.\n{2 A {b part}}\nBody.\n{3:deep Deeper}", &resolve),
            "Intro.\n\n### A **part**\n\nBody.\n\n#### Deeper {#deep}"
        );
        assert_eq!(conv("{2:x Sub}\nBody."), "**Sub**\n\nBody.");
    }

    #[test]
    fn escapes_are_literal() {
        assert_eq!(conv("a \\{b\\} \\[c\\]"), "a {b} \\[c\\]");
    }

    #[test]
    fn deep_indentation_does_not_make_a_code_block() {
        assert_eq!(conv("a\n  x\n        b"), "a\nx\n   b");
    }
}
