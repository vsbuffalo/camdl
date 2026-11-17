//! A lexer for Rust source, for highlighting and for finding the paths a
//! declaration writes. It splits text into tokens and classifies them; it
//! does not parse, and it never decides what a name refers to — that comes
//! from rustdoc ([`crate::rustdoc`]).
//!
//! Comments are tokens here, unlike in `proc-macro2`, because the pages show
//! them and the declarations shown must leave them out.

/// What a token is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tok {
    /// An identifier that is not a keyword: `Model`, `r#type`.
    Ident,
    /// A keyword: `struct`, `fn`, `crate`, `Self`.
    Keyword,
    /// A primitive type's name: `f64`, `usize`, `str`.
    Primitive,
    /// A comment that is not a doc comment.
    Comment,
    /// A doc comment: `///`, `//!`, `/** */`, `/*! */`.
    Doc,
    /// A string literal, including byte, C and raw strings.
    Str,
    /// A character or byte literal.
    Char,
    /// A number literal.
    Number,
    /// A lifetime or label: `'a`, `'static`.
    Lifetime,
    /// Punctuation; `::` is one token.
    Punct,
    /// Whitespace.
    Space,
}

/// A token: its kind and its byte range in the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub kind: Tok,
    pub start: usize,
    pub end: usize,
}

const KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "unsafe", "use", "where", "while",
];

const PRIMITIVES: &[&str] = &[
    "bool", "char", "str", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
    "i128", "isize", "f32", "f64",
];

/// The keywords that can begin or continue a path: `crate::x`, `self::x`,
/// `super::x`, `Self::X`.
pub fn is_path_keyword(s: &str) -> bool {
    matches!(s, "crate" | "self" | "super" | "Self")
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Splits `src` into tokens that cover it exactly, in order. Never fails: a
/// literal or comment left open runs to the end of the text.
pub fn lex(src: &str) -> Vec<Token> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let char_at = |i: usize| src[i..].chars().next();
    while i < b.len() {
        let start = i;
        let c = char_at(i).expect("i is on a character boundary");
        let kind = if c.is_whitespace() {
            while let Some(c) = char_at(i).filter(|c| c.is_whitespace()) {
                i += c.len_utf8();
            }
            Tok::Space
        } else if src[i..].starts_with("//") {
            let rest = &src[i..];
            let doc =
                (rest.starts_with("///") && !rest.starts_with("////")) || rest.starts_with("//!");
            i += rest.find('\n').unwrap_or(rest.len());
            if doc {
                Tok::Doc
            } else {
                Tok::Comment
            }
        } else if src[i..].starts_with("/*") {
            let rest = &src[i..];
            let doc =
                (rest.starts_with("/**") && !rest.starts_with("/***") && !rest.starts_with("/**/"))
                    || rest.starts_with("/*!");
            i += 2;
            let mut depth = 1;
            while i < b.len() && depth > 0 {
                if src[i..].starts_with("/*") {
                    depth += 1;
                    i += 2;
                } else if src[i..].starts_with("*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += char_at(i).map_or(1, char::len_utf8);
                }
            }
            if doc {
                Tok::Doc
            } else {
                Tok::Comment
            }
        } else if let Some(end) = string_at(src, i) {
            i = end;
            Tok::Str
        } else if c == 'b' && src[i..].starts_with("b'") {
            i = char_literal_end(src, i + 1).unwrap_or(b.len());
            Tok::Char
        } else if src[i..].starts_with("r#") && char_at(i + 2).is_some_and(is_ident_start) {
            i += 2;
            while let Some(c) = char_at(i).filter(|c| is_ident_continue(*c)) {
                i += c.len_utf8();
            }
            Tok::Ident
        } else if is_ident_start(c) {
            while let Some(c) = char_at(i).filter(|c| is_ident_continue(*c)) {
                i += c.len_utf8();
            }
            let word = &src[start..i];
            if KEYWORDS.contains(&word) {
                Tok::Keyword
            } else if PRIMITIVES.contains(&word) {
                Tok::Primitive
            } else {
                Tok::Ident
            }
        } else if c == '\'' {
            match char_literal_end(src, i) {
                Some(end) => {
                    i = end;
                    Tok::Char
                }
                None => {
                    i += 1;
                    while let Some(c) = char_at(i).filter(|c| is_ident_continue(*c)) {
                        i += c.len_utf8();
                    }
                    Tok::Lifetime
                }
            }
        } else if c.is_ascii_digit() {
            i = number_end(src, i);
            Tok::Number
        } else if src[i..].starts_with("::") {
            i += 2;
            Tok::Punct
        } else {
            i += c.len_utf8();
            Tok::Punct
        };
        out.push(Token {
            kind,
            start,
            end: i,
        });
    }
    out
}

/// The end of a character literal whose opening quote is at `i`, or `None`
/// when the quote opens a lifetime instead.
fn char_literal_end(src: &str, i: usize) -> Option<usize> {
    let rest = &src[i + 1..];
    let mut chars = rest.char_indices();
    let (_, first) = chars.next()?;
    if first == '\\' {
        // An escape: runs to the next quote.
        let close = rest[1..].find('\'')?;
        return Some(i + 1 + 1 + close + 1);
    }
    let (k, second) = chars.next()?;
    (second == '\'' && first != '\'').then_some(i + 1 + k + 1)
}

/// The end of a string literal starting at `i`, if one starts there: `"..."`,
/// `b"..."`, `c"..."`, or a raw string `r"..."`, `r#"..."#`, `br#"..."#`.
fn string_at(src: &str, i: usize) -> Option<usize> {
    let rest = &src[i..];
    let prefix = ["br", "cr", "r", "b", "c", ""].into_iter().find(|p| {
        let Some(after) = rest.strip_prefix(p) else {
            return false;
        };
        after.starts_with('"') || (p.ends_with('r') && after.starts_with('#'))
    })?;
    let after = &rest[prefix.len()..];
    if prefix.ends_with('r') {
        let hashes = after.chars().take_while(|c| *c == '#').count();
        if !after[hashes..].starts_with('"') {
            return None;
        }
        let close = format!("\"{}", "#".repeat(hashes));
        let body = i + prefix.len() + hashes + 1;
        return Some(
            src[body..]
                .find(&close)
                .map_or(src.len(), |k| body + k + close.len()),
        );
    }
    let mut j = i + prefix.len() + 1;
    let b = src.as_bytes();
    while j < b.len() {
        match b[j] {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    Some(src.len())
}

/// The end of a number literal starting at `i`: digits, letters and `_`
/// (for a base prefix, a suffix or an exponent), a fraction when a `.` is
/// followed by a digit, and an exponent's sign.
fn number_end(src: &str, mut i: usize) -> usize {
    let b = src.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut seen_dot = false;
    while i < b.len() {
        let c = b[i];
        if word(c) {
            i += 1;
            if (c == b'e' || c == b'E')
                && i < b.len()
                && (b[i] == b'+' || b[i] == b'-')
                && !src[..i].starts_with("0x")
            {
                i += 1;
            }
        } else if c == b'.' && !seen_dot && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
            seen_dot = true;
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// A path as written: identifiers and path keywords joined by `::`, with no
/// space between them, such as `ir::model::Model` or `Self`. Given as a
/// range of token positions, `first..=last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathRun {
    pub first: usize,
    pub last: usize,
}

/// The paths in a token list, each as long as it can be. A path may start
/// with `::`. Turbofish and generic arguments end a path: `Vec::<T>::new`
/// gives `Vec`, then `new`.
pub fn paths(src: &str, toks: &[Token]) -> Vec<PathRun> {
    let is_seg = |t: &Token| match t.kind {
        Tok::Ident => true,
        Tok::Keyword => is_path_keyword(&src[t.start..t.end]),
        _ => false,
    };
    let is_sep = |t: &Token| t.kind == Tok::Punct && &src[t.start..t.end] == "::";
    let mut out = Vec::new();
    let mut k = 0;
    while k < toks.len() {
        // A leading `::` names the extern prelude, except after `>`, where
        // it continues a path with generic arguments (`Vec::<T>::new`).
        let after_generics = k
            .checked_sub(1)
            .is_some_and(|j| &src[toks[j].start..toks[j].end] == ">");
        let lead = is_sep(&toks[k]) && !after_generics && toks.get(k + 1).is_some_and(is_seg);
        if !(is_seg(&toks[k]) || lead) {
            k += 1;
            continue;
        }
        let first = k;
        let mut last = if lead { k + 1 } else { k };
        while toks.get(last + 1).is_some_and(is_sep) && toks.get(last + 2).is_some_and(is_seg) {
            last += 2;
        }
        out.push(PathRun { first, last });
        k = last + 1;
    }
    out
}

/// Byte offsets to lines and columns: line counting from 1, column in bytes
/// from 0.
pub struct Lines {
    starts: Vec<usize>,
}

impl Lines {
    pub fn new(src: &str) -> Lines {
        let mut starts = vec![0];
        starts.extend(src.match_indices('\n').map(|(i, _)| i + 1));
        Lines { starts }
    }

    /// The line (from 1) and byte column (from 0) of a byte offset.
    pub fn at(&self, offset: usize) -> (usize, usize) {
        let line = self.starts.partition_point(|&s| s <= offset);
        (line, offset - self.starts[line - 1])
    }

    /// The byte offset where a line (from 1) starts, if the text has it.
    pub fn start(&self, line: usize) -> Option<usize> {
        line.checked_sub(1)
            .and_then(|l| self.starts.get(l))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<(Tok, &str)> {
        lex(src)
            .into_iter()
            .filter(|t| t.kind != Tok::Space)
            .map(|t| (t.kind, &src[t.start..t.end]))
            .collect()
    }

    #[test]
    fn comments_doc_comments_and_nesting() {
        assert_eq!(
            kinds("/// doc\n// plain\n//// four\n/* a /* nested */ b */ x /** d */"),
            [
                (Tok::Doc, "/// doc"),
                (Tok::Comment, "// plain"),
                (Tok::Comment, "//// four"),
                (Tok::Comment, "/* a /* nested */ b */"),
                (Tok::Ident, "x"),
                (Tok::Doc, "/** d */"),
            ]
        );
    }

    #[test]
    fn literals_lifetimes_and_paths() {
        assert_eq!(
            kinds(r##"'a' '\n' 'b b'x' "s\"t" r#"raw"# 1.5e-3 1..2 x.0 &'static str"##),
            [
                (Tok::Char, "'a'"),
                (Tok::Char, "'\\n'"),
                (Tok::Lifetime, "'b"),
                (Tok::Char, "b'x'"),
                (Tok::Str, "\"s\\\"t\""),
                (Tok::Str, "r#\"raw\"#"),
                (Tok::Number, "1.5e-3"),
                (Tok::Number, "1"),
                (Tok::Punct, "."),
                (Tok::Punct, "."),
                (Tok::Number, "2"),
                (Tok::Ident, "x"),
                (Tok::Punct, "."),
                (Tok::Number, "0"),
                (Tok::Punct, "&"),
                (Tok::Lifetime, "'static"),
                (Tok::Primitive, "str"),
            ]
        );
    }

    #[test]
    fn paths_join_segments_and_stop_at_generics() {
        let src = "crate::ir::Model<T> + Vec::<u8>::new + ::std::fmt::Debug + a :: b";
        let toks = lex(src);
        let texts: Vec<&str> = paths(src, &toks)
            .iter()
            .map(|p| &src[toks[p.first].start..toks[p.last].end])
            .collect();
        assert_eq!(
            texts,
            [
                "crate::ir::Model",
                "T",
                "Vec",
                "new",
                "::std::fmt::Debug",
                "a",
                "b"
            ]
        );
    }

    #[test]
    fn lines_and_columns() {
        let l = Lines::new("ab\ncd\n");
        assert_eq!(l.at(0), (1, 0));
        assert_eq!(l.at(4), (2, 1));
        assert_eq!(l.start(2), Some(3));
    }
}
