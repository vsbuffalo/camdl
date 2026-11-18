//! Reading a crate's own source as code: the files under a directory, with
//! comments blanked and `#[cfg(test)]` items removed. Shared by the source
//! rules `sim_prints_nothing` (sim) and `environment_is_read_in_one_module`
//! (cli, which includes this file by path).

use std::path::{Path, PathBuf};

pub fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).flatten() {
        let p = entry.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// `src` with comments blanked to spaces (newlines kept, so line numbers
/// hold). String and char literals are kept as they are, and are skipped
/// over so a `//` inside one is not read as a comment.
pub fn without_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while i < b.len() {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    let c = src[i..].chars().next().unwrap();
                    out.push(blank(c));
                    i += c.len_utf8();
                }
            }
        } else if b[i] == b'"' {
            // A string literal, escapes honoured; raw strings with `#`
            // delimiters do not occur in sim's print sites.
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    out.push('\\');
                    i += 1;
                }
                let c = src[i..].chars().next().unwrap();
                out.push(c);
                i += c.len_utf8();
            }
            if i < b.len() {
                out.push('"');
                i += 1;
            }
        } else if b[i] == b'\'' && b.get(i + 2) == Some(&b'\'') {
            // A one-byte char literal such as '"', which would otherwise open
            // a string.
            out.push_str(&src[i..i + 3]);
            i += 3;
        } else if b[i] == b'\'' && b.get(i + 1) == Some(&b'\\') && b.get(i + 3) == Some(&b'\'') {
            out.push_str(&src[i..i + 4]);
            i += 4;
        } else {
            let c = src[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// `code` (already without comments) with every `#[cfg(test)]` item blanked:
/// from the attribute to the end of the item it annotates — the matching `}`
/// of its first `{`, or its `;` when it has no body.
pub fn without_test_items(code: &str) -> String {
    let mut out: Vec<char> = code.chars().collect();
    let text: String = code.to_string();
    let mut search = 0;
    while let Some(off) = text[search..].find("#[cfg(test)]") {
        let start = search + off;
        let mut depth = 0usize;
        let mut end = text.len();
        for (j, c) in text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + j + 1;
                        break;
                    }
                }
                ';' if depth == 0 => {
                    end = start + j + 1;
                    break;
                }
                _ => {}
            }
        }
        let (cs, ce) = (text[..start].chars().count(), text[..end].chars().count());
        for c in &mut out[cs..ce] {
            if *c != '\n' {
                *c = ' ';
            }
        }
        search = end;
    }
    out.into_iter().collect()
}
