//! Highlighted source text as HTML, with links on the type names an
//! extractor resolved. Used for whole source files and for the declaration
//! blocks on module pages. Knows nothing of any language: the extractor says
//! what each span is ([`Span`]), and this module only escapes and wraps.
//!
//! The markup reuses the classes of the highlighting Quarto applies to fenced
//! code (`kw`, `co`, `st`, ...) inside `div.sourceCode > pre > code`, so the
//! site's light and dark highlight styles color it with no styles of its own.

use std::fmt::Write as _;

use crate::model::{Span, SpanClass, Target};

/// Escapes text for HTML element content and attribute values.
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// The class of Quarto's highlighting a span is drawn with, or `None` for
/// text shown in the plain color.
fn quarto_class(class: SpanClass) -> Option<&'static str> {
    match class {
        SpanClass::Keyword => Some("kw"),
        SpanClass::Comment => Some("co"),
        SpanClass::Doc => Some("do"),
        SpanClass::String => Some("st"),
        SpanClass::Char => Some("ch"),
        SpanClass::Number => Some("dv"),
        SpanClass::BuiltinType => Some("dt"),
        SpanClass::Type => None,
    }
}

/// A text split into lines, each line's spans checked against it: on the
/// line, on character boundaries, in order and not overlapping. Building one
/// is the only check; rendering cannot fail.
#[derive(Debug)]
pub struct Highlighted<'a> {
    lines: Vec<(&'a str, Vec<&'a Span>)>,
}

impl<'a> Highlighted<'a> {
    /// Places `spans` (lines counted from 1) on the lines of `text`. Fails,
    /// naming the span, when one does not fit the text.
    pub fn new(text: &'a str, spans: &'a [Span]) -> Result<Highlighted<'a>, String> {
        let mut lines: Vec<(&str, Vec<&Span>)> = text.lines().map(|l| (l, Vec::new())).collect();
        let mut prev: Option<(usize, usize)> = None;
        for s in spans {
            let bad = |why: &str| {
                format!(
                    "span at line {} column {} (length {}): {why}",
                    s.line, s.col, s.len
                )
            };
            if s.len == 0 {
                return Err(bad("is empty"));
            }
            if let Some((line, end)) = prev {
                if s.line < line || (s.line == line && s.col < end) {
                    return Err(bad("is out of order or overlaps the one before"));
                }
            }
            let Some((line, on_line)) = s.line.checked_sub(1).and_then(|i| lines.get_mut(i)) else {
                return Err(bad("lies past the last line"));
            };
            let end = s.col + s.len;
            if end > line.len() {
                return Err(bad("runs past the end of its line"));
            }
            if !line.is_char_boundary(s.col) || !line.is_char_boundary(end) {
                return Err(bad("cuts a character in two"));
            }
            on_line.push(s);
            prev = Some((s.line, end));
        }
        Ok(Highlighted { lines })
    }

    /// Each line as HTML, without its line end. A span with a target links
    /// to the URL `href` gives it; one `href` has no URL for is shown
    /// unlinked.
    pub fn lines_html(&self, href: &dyn Fn(&Target) -> Option<String>) -> Vec<String> {
        self.lines
            .iter()
            .map(|(line, spans)| {
                let mut out = String::new();
                let mut at = 0;
                for s in spans {
                    out.push_str(&escape_html(&line[at..s.col]));
                    let text = escape_html(&line[s.col..s.col + s.len]);
                    let url = s.target.as_ref().and_then(href);
                    match (url, quarto_class(s.class)) {
                        (Some(url), Some(c)) => {
                            let _ = write!(
                                out,
                                "<a class=\"td-ref\" href=\"{}\"><span class=\"{c}\">{text}</span></a>",
                                escape_html(&url)
                            );
                        }
                        (Some(url), None) => {
                            let _ = write!(
                                out,
                                "<a class=\"td-ref\" href=\"{}\">{text}</a>",
                                escape_html(&url)
                            );
                        }
                        (None, Some(c)) => {
                            let _ = write!(out, "<span class=\"{c}\">{text}</span>");
                        }
                        (None, None) => out.push_str(&text),
                    }
                    at = s.col + s.len;
                }
                out.push_str(&escape_html(&line[at..]));
                out
            })
            .collect()
    }
}

/// A code block of highlighted lines: the markup Quarto gives a fenced
/// block, so it is styled as one. `extra` is a further class on the block.
/// With `anchors`, line `n` is an element with id `L<n>`, and the lines are
/// numbered by the site's styles.
pub fn block(lines: &[String], extra: &str, anchors: bool) -> String {
    let mut out = format!(
        "<div class=\"sourceCode td-code {extra}\"><pre class=\"sourceCode\"><code class=\"sourceCode\">"
    );
    // Each line is its own element, as in Quarto's blocks: its styles reset
    // the color of the code element's direct children, so a highlighted span
    // there would be drawn plain.
    for (i, l) in lines.iter().enumerate() {
        if anchors {
            let _ = write!(out, "<span class=\"td-line\" id=\"L{}\">{l}</span>", i + 1);
        } else {
            let _ = write!(out, "<span class=\"td-line\">{l}</span>");
        }
        if i + 1 < lines.len() {
            out.push('\n');
        }
    }
    out.push_str("</code></pre></div>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(
        line: usize,
        col: usize,
        len: usize,
        class: SpanClass,
        target: Option<(&str, &str)>,
    ) -> Span {
        Span {
            line,
            col,
            len,
            class,
            target: target.map(|(m, t)| Target {
                module: m.to_string(),
                ty: t.to_string(),
            }),
        }
    }

    fn href(t: &Target) -> Option<String> {
        (t.module == "Ast").then(|| format!("../ast.html#t-{}", t.ty))
    }

    #[test]
    fn escapes_text_and_links_resolved_targets_only() {
        let text = "type t = (a, \"<b>\") Ast.expr & Ir.expr\n";
        let spans = [
            span(1, 0, 4, SpanClass::Keyword, None),
            span(1, 5, 1, SpanClass::Type, Some(("Ast", "t"))),
            span(1, 13, 5, SpanClass::String, None),
            span(1, 20, 8, SpanClass::Type, Some(("Ast", "expr"))),
            span(1, 31, 7, SpanClass::Type, Some(("Ir", "expr"))),
        ];
        let h = Highlighted::new(text, &spans).unwrap();
        assert_eq!(
            h.lines_html(&href),
            [concat!(
                "<span class=\"kw\">type</span> ",
                "<a class=\"td-ref\" href=\"../ast.html#t-t\">t</a> = (a, ",
                "<span class=\"st\">&quot;&lt;b&gt;&quot;</span>) ",
                "<a class=\"td-ref\" href=\"../ast.html#t-expr\">Ast.expr</a> &amp; Ir.expr"
            )]
        );
    }

    #[test]
    fn a_resolved_target_links_to_its_own_type() {
        // Mutation check: linking every Type span to one fixed URL, or
        // ignoring the target's type name, fails here.
        let text = "a b";
        let spans = [
            span(1, 0, 1, SpanClass::Type, Some(("Ast", "a"))),
            span(1, 2, 1, SpanClass::Type, Some(("Ast", "b"))),
        ];
        let html = Highlighted::new(text, &spans).unwrap().lines_html(&href);
        assert_eq!(
            html,
            ["<a class=\"td-ref\" href=\"../ast.html#t-a\">a</a> <a class=\"td-ref\" href=\"../ast.html#t-b\">b</a>"]
        );
    }

    #[test]
    fn blocks_anchor_every_line() {
        let text = "x\n\ny <z>\n";
        let h = Highlighted::new(text, &[]).unwrap();
        let html = block(&h.lines_html(&href), "td-src", true);
        assert_eq!(
            html,
            "<div class=\"sourceCode td-code td-src\"><pre class=\"sourceCode\"><code class=\"sourceCode\">\
             <span class=\"td-line\" id=\"L1\">x</span>\n\
             <span class=\"td-line\" id=\"L2\"></span>\n\
             <span class=\"td-line\" id=\"L3\">y &lt;z&gt;</span></code></pre></div>"
        );
        let decl = block(&h.lines_html(&href), "td-decl", false);
        assert!(!decl.contains("id="), "{decl}");
        assert!(
            decl.contains("<code class=\"sourceCode\"><span class=\"td-line\">x</span>\n"),
            "{decl}"
        );
    }

    #[test]
    fn spans_that_do_not_fit_are_refused() {
        let text = "ab\né\n";
        for (s, why) in [
            (
                span(3, 0, 1, SpanClass::Keyword, None),
                "past the last line",
            ),
            (
                span(0, 0, 1, SpanClass::Keyword, None),
                "past the last line",
            ),
            (
                span(1, 1, 2, SpanClass::Keyword, None),
                "past the end of its line",
            ),
            (span(2, 0, 1, SpanClass::Keyword, None), "cuts a character"),
            (span(1, 0, 0, SpanClass::Keyword, None), "is empty"),
        ] {
            let e = Highlighted::new(text, std::slice::from_ref(&s)).unwrap_err();
            assert!(e.contains(why), "{s:?}: {e}");
        }
        let overlapping = [
            span(1, 0, 2, SpanClass::Keyword, None),
            span(1, 1, 1, SpanClass::Keyword, None),
        ];
        let e = Highlighted::new(text, &overlapping).unwrap_err();
        assert!(e.contains("overlaps"), "{e}");
    }
}
