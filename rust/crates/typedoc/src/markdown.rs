//! Writing Markdown that Quarto reads as `commonmark_x`: escaping, code spans
//! and fences, and the tidying every converted doc comment goes through.

/// Characters escaped wherever they occur in text, because `commonmark_x`
/// gives each a meaning: emphasis (`*`), raw HTML (`<`), TeX math (`$`), a
/// citation (`@`), subscript and strikeout (`~`), superscript (`^`), and link
/// brackets (`[`, `]`).
const ALWAYS_ESCAPED: &[char] = &['*', '<', '$', '@', '~', '^', '[', ']', '\\', '`'];

/// Escapes one character of text for Markdown. `prev` and `next` are its
/// neighbours, which decide whether `_` can open or close emphasis.
pub fn push_text_char(out: &mut String, c: char, prev: Option<char>, next: Option<char>) {
    let word = |c: Option<char>| c.is_some_and(char::is_alphanumeric);
    let escape = ALWAYS_ESCAPED.contains(&c) || (c == '_' && !(word(prev) && word(next)));
    if escape {
        out.push('\\');
    }
    out.push(c);
}

/// Text with every Markdown-significant character escaped, for a string that
/// must be read literally: a name in a heading, a phrase quoted in a mark.
pub fn escape_text(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| chars[j]);
        push_text_char(&mut out, c, prev, chars.get(i + 1).copied());
    }
    if out.starts_with('#') || out.starts_with('>') {
        out.insert(0, '\\');
    }
    out
}

/// The length of the longest run of `c` in `s`.
fn longest_run(s: &str, c: char) -> usize {
    let mut best = 0;
    let mut cur = 0;
    for x in s.chars() {
        if x == c {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

/// An inline code span holding `s` literally, on one line. The delimiter is
/// longer than any run of backticks inside, so OCaml's polymorphic variants
/// (`` [`A | `B] ``) survive.
pub fn code_span(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let ticks = "`".repeat(longest_run(&flat, '`') + 1);
    let pad = if flat.starts_with('`') || flat.ends_with('`') {
        " "
    } else {
        ""
    };
    format!("{ticks}{pad}{flat}{pad}{ticks}")
}

/// A fenced code block holding `body`, with `info` as its info string (empty
/// for none). The fence is longer than any run of backticks in the body.
pub fn fence(info: &str, body: &str) -> String {
    let ticks = "`".repeat((longest_run(body, '`') + 1).max(3));
    format!("{ticks}{info}\n{}\n{ticks}", body.trim_end())
}

/// Removes the indentation common to a doc comment's lines.
///
/// The first line is treated apart: in `(** Summary.\n    More. *)` it
/// starts right after the opening delimiter, so its own indentation says
/// nothing about the others'. Blank lines at either end are dropped.
pub fn dedent(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let indent = |l: &str| l.len() - l.trim_start().len();
    let common = lines[1..]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| indent(l))
        .min()
        .unwrap_or(0);
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    out.push(lines[0].trim_start());
    for l in &lines[1..] {
        out.push(if l.len() >= common {
            &l[common..]
        } else {
            l.trim_start()
        });
    }
    while out.first().is_some_and(|l| l.trim().is_empty()) {
        out.remove(0);
    }
    while out.last().is_some_and(|l| l.trim().is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Whether a line opens a list item: `-`, `+` or `*`, or a number followed by
/// `.` or `)`, then a space, after at most three spaces.
fn is_list_item(line: &str) -> bool {
    let t = line.trim_start();
    if line.len() - t.len() > 3 {
        return false;
    }
    let mut chars = t.chars();
    match chars.next() {
        Some('-' | '+' | '*') => chars.next() == Some(' '),
        Some(c) if c.is_ascii_digit() => {
            let rest = t.trim_start_matches(|c: char| c.is_ascii_digit());
            rest.starts_with(". ") || rest.starts_with(") ")
        }
        _ => false,
    }
}

/// The fence a line opens or closes, if it is one: its character and length.
fn fence_marker(line: &str) -> Option<(char, usize)> {
    let t = line.trim_start();
    let c = t.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let n = t.chars().take_while(|x| *x == c).count();
    (n >= 3).then_some((c, n))
}

/// Tidies converted Markdown outside code blocks: strips trailing spaces,
/// collapses runs of blank lines into one, and puts a blank line before a
/// list that follows a paragraph, which Markdown requires and odoc does not.
pub fn tidy(s: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut open: Option<(char, usize)> = None;
    let mut in_list = false;
    let mut after_blank = false;
    for line in s.lines() {
        if let Some((c, n)) = open {
            if fence_marker(line).is_some_and(|(c2, n2)| c2 == c && n2 >= n)
                && line.trim().chars().all(|x| x == c)
            {
                open = None;
            }
            out.push(line.trim_end().to_string());
            continue;
        }
        if let Some(m) = fence_marker(line) {
            open = Some(m);
            in_list = false;
            out.push(line.trim_end().to_string());
            continue;
        }
        let line = line.trim_end();
        if line.is_empty() {
            if out.last().is_some_and(|l| !l.is_empty()) {
                out.push(String::new());
            }
            after_blank = true;
            continue;
        }
        if is_list_item(line) {
            if !in_list && out.last().is_some_and(|l| !l.is_empty()) {
                out.push(String::new());
            }
            in_list = true;
        } else if after_blank && !line.starts_with(' ') {
            in_list = false;
        }
        after_blank = false;
        out.push(line.to_string());
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    while out.first().is_some_and(|l| l.is_empty()) {
        out.remove(0);
    }
    out.join("\n")
}

/// The first sentence of converted Markdown, for a summary line: the first
/// paragraph, on one line, cut after the first `.`, `?` or `!` that is
/// followed by a space or ends the paragraph and is not inside a code span or
/// part of `e.g.` or `i.e.`. `None` when the text opens with a code block or
/// a list, which has no first sentence.
pub fn first_sentence(md: &str) -> Option<String> {
    split_first_sentence(md).map(|(head, _)| head)
}

/// The first sentence of converted Markdown, as [`first_sentence`] finds it,
/// and the Markdown that follows it: the rest of the first paragraph, on one
/// line, then the remaining blocks unchanged.
pub fn split_first_sentence(md: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = md.lines().collect();
    let para: Vec<&str> = lines
        .iter()
        .copied()
        .take_while(|l| !l.trim().is_empty() && fence_marker(l).is_none())
        .collect();
    if para.is_empty() || is_list_item(para[0]) {
        return None;
    }
    let following = lines[para.len()..].join("\n");
    let following = following.trim_start_matches('\n');
    let text = para.iter().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = text.chars().collect();
    let mut ticks = 0usize;
    let mut i = 0;
    let mut end = chars.len();
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            let n = chars[i..].iter().take_while(|x| **x == '`').count();
            ticks = if ticks == 0 {
                n
            } else if ticks == n {
                0
            } else {
                ticks
            };
            i += n;
            continue;
        }
        let ends = matches!(c, '.' | '?' | '!')
            && ticks == 0
            && chars.get(i + 1).is_none_or(|n| n.is_whitespace())
            && !(i > 0 && chars[i - 1] == '\\');
        if ends {
            let lower: String = chars[..=i].iter().collect::<String>().to_lowercase();
            if !(lower.ends_with("e.g.") || lower.ends_with("i.e.")) {
                end = i + 1;
                break;
            }
        }
        i += 1;
    }
    let head: String = chars[..end].iter().collect();
    let tail: String = chars[end..].iter().collect();
    let rest = match (tail.trim(), following) {
        ("", f) => f.to_string(),
        (t, "") => t.to_string(),
        (t, f) => format!("{t}\n\n{f}"),
    };
    Some((head, rest))
}

/// Indents every non-blank line of `s` by `n` spaces.
pub fn indent(s: &str, n: usize) -> String {
    let pad = " ".repeat(n);
    s.lines()
        .map(|l| {
            if l.is_empty() {
                String::new()
            } else {
                format!("{pad}{l}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A string as a double-quoted YAML scalar.
pub fn yaml_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Converts a Rust doc comment, which is Markdown already, to the page's
/// Markdown. Intra-doc links written ``[`Name`]`` become links when `link`
/// resolves the name, and code spans otherwise, since the page has no link
/// definitions for them.
pub fn rustdoc_to_markdown(src: &str, link: &dyn Fn(&str) -> Option<String>) -> String {
    let text = dedent(src);
    let mut out = String::with_capacity(text.len());
    let mut in_fence: Option<(char, usize)> = None;
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if let Some((c, n)) = in_fence {
            if fence_marker(line).is_some_and(|(c2, n2)| c2 == c && n2 >= n) {
                in_fence = None;
            }
            out.push_str(line);
            continue;
        }
        if let Some(m) = fence_marker(line) {
            in_fence = Some(m);
            out.push_str(line);
            continue;
        }
        out.push_str(&rust_intra_links(line, link));
    }
    tidy(&out)
}

/// Rewrites the ``[`Name`]`` intra-doc links on one line.
fn rust_intra_links(line: &str, link: &dyn Fn(&str) -> Option<String>) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(start) = rest.find("[`") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("`]") else { break };
        let name = &after[..end];
        let tail = &after[end + 2..];
        out.push_str(&rest[..start]);
        if tail.starts_with('(') || tail.starts_with('[') || name.contains('`') {
            out.push_str(&rest[start..start + 2 + end + 2]);
        } else {
            let target = name.trim_end_matches("()");
            match link(target) {
                Some(href) => out.push_str(&format!("[{}]({href})", code_span(name))),
                None => out.push_str(&code_span(name)),
            }
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_span_outgrows_inner_backticks() {
        assert_eq!(code_span("int list"), "`int list`");
        assert_eq!(code_span("[ `A | `B ]"), "``[ `A | `B ]``");
        assert_eq!(code_span("`A"), "`` `A ``");
        assert_eq!(code_span("a\n   b"), "`a b`");
    }

    #[test]
    fn dedent_ignores_first_line_indentation() {
        assert_eq!(
            dedent(" Summary.\n    More.\n      Indented.\n  "),
            "Summary.\nMore.\n  Indented."
        );
    }

    #[test]
    fn tidy_puts_a_blank_line_before_a_list() {
        assert_eq!(
            tidy("Intro:\n- a\n  more a\n- b\nAfter."),
            "Intro:\n\n- a\n  more a\n- b\nAfter."
        );
        assert_eq!(tidy("a\n\n\n\nb  "), "a\n\nb");
        assert_eq!(tidy("```\nx\n\n\n```"), "```\nx\n\n\n```");
    }

    #[test]
    fn first_sentence_skips_code_and_abbreviations() {
        assert_eq!(
            first_sentence("Reads `a.b`. Then more.").as_deref(),
            Some("Reads `a.b`.")
        );
        assert_eq!(
            first_sentence("A rate, e.g. per day. More.").as_deref(),
            Some("A rate, e.g. per day.")
        );
        assert_eq!(
            first_sentence("No period\nat all").as_deref(),
            Some("No period at all")
        );
        assert_eq!(first_sentence("```ocaml\nx\n```"), None);
    }

    #[test]
    fn split_first_sentence_keeps_what_follows() {
        let split = |s: &str| split_first_sentence(s).unwrap();
        assert_eq!(
            split("One here.\nTwo, then\nthree.\n\n## Next\n\nBody."),
            (
                "One here.".to_string(),
                "Two, then three.\n\n## Next\n\nBody.".to_string()
            )
        );
        assert_eq!(
            split("Only one.\n\nBody."),
            ("Only one.".to_string(), "Body.".to_string())
        );
        assert_eq!(split("Only one."), ("Only one.".to_string(), String::new()));
    }

    #[test]
    fn escape_text_neutralises_markdown() {
        assert_eq!(
            escape_text("a*b <c> $x @y snake_case _lead"),
            "a\\*b \\<c> \\$x \\@y snake_case \\_lead"
        );
        assert_eq!(escape_text("# not a heading"), "\\# not a heading");
    }

    #[test]
    fn rust_intra_doc_links() {
        let link = |n: &str| (n == "IrError").then(|| "ir.qmd#t-IrError".to_string());
        let md = rustdoc_to_markdown(
            "Refused with [`IrError`] or [`Other`].\n\n```\nlet x = [`IrError`];\n```",
            &link,
        );
        assert_eq!(
            md,
            "Refused with [`IrError`](ir.qmd#t-IrError) or `Other`.\n\n```\nlet x = [`IrError`];\n```"
        );
    }
}
