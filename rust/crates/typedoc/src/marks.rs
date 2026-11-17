//! Reviewer marks: visible prompts on a type, member or module that a reader
//! might want to question. A mark is never an error; it says where to look.

use std::sync::OnceLock;

use regex::Regex;

use crate::markdown::escape_text;
use crate::model::{Language, Member, TypeDecl};

/// Phrases that narrate how the code came to be rather than what it is.
/// Matched case-insensitively against a doc comment's prose, with code blocks
/// removed (a calendar date in a camdl example is not history).
///
/// - `gh#616`: an issue reference.
/// - `malaria #2`, `issue #12`, or a bare `#123` after a space or `(`: a
///   numbered issue or work item. A bare `#` needs a digit right after it, so
///   Markdown headings and Rust attributes (`#[derive]`) do not match.
/// - `§ 4.2`: a section number of some other document, usually a proposal.
/// - `proposal`, `incident`, `wave 2`: names of the work that produced the code.
///   `proposal` also matches the proposal distribution of a sampler, which is
///   not history; a reviewer reads the phrase and judges.
/// - `previously`, `formerly`, `no longer`, `used to`, `superseded`,
///   `deprecated`, `legacy`, `renamed`: a comparison with an earlier version.
/// - `2026-06-09`: a date, which in a doc comment almost always dates a change.
/// - `v1`, `v2`: a version of the design.
const HISTORY_PATTERNS: &[&str] = &[
    r"\bgh#\w+",
    r"\b(?:issues?|malaria|bug|ticket|pr|fix(?:es)?|closes|see)\s*#\d+\b",
    r"(?:^|[\s(])#\d+\b",
    r"§\s*\d+(?:\.\d+)*",
    r"\bproposals?\b",
    r"\bincidents?\b",
    r"\bwave \d+",
    r"\b(?:previously|formerly|no longer|used to|superseded|deprecated|legacy|renamed)\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bv[12]\b",
];

/// Words of four or more capitals that are acronyms or names in camdl's
/// sources, not emphasis.
const CAPITALS_ALLOWED: &[&str] = &[
    "ASCII", "BDBV", "BFGS", "BINV", "BTPE", "BTRS", "CAMDLC", "CRPS", "CSMC", "HTML", "IEEE",
    "JSON", "LBFGS", "LICM", "MCMC", "MCSE", "NLOPT", "NUTS", "PGAS", "PMMH", "POIS", "SEIR",
    "SEIRS", "TODO", "TOML", "UTF", "YAML", "YYYY",
];

/// The number of fields of one kind (`bool`, or optional) at which a record
/// is marked: the point where a variant type might rule out combinations of
/// them that cannot occur.
pub const MANY_FIELDS: usize = 3;

/// Field names, or final `_`-separated parts of field names, that suggest a
/// `string` field holds one of a fixed set of values.
const ENUM_LIKE_NAMES: &[&str] = &[
    "kind", "mode", "type", "method", "status", "strategy", "backend", "op",
];

/// The most phrases or words a mark quotes before it ends in an ellipsis.
const MAX_QUOTED: usize = 3;

/// One reviewer mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mark {
    /// The type, member or module has no doc comment.
    Undocumented,
    /// Some of a type's constructors or fields have no doc comment.
    UndocumentedMembers {
        undocumented: usize,
        total: usize,
        noun: &'static str,
    },
    /// The doc comment narrates history; the phrases that matched.
    History(Vec<String>),
    /// The doc comment uses capitals for emphasis; the words.
    Capitals(Vec<String>),
    /// A record has this many `bool` fields.
    Flags(usize),
    /// A record has this many optional fields.
    Options(usize),
    /// `string` fields whose names suggest a fixed set of values.
    StringEnum(Vec<String>),
}

impl Mark {
    /// The mark as a Markdown span with the mark's classes.
    pub fn render(&self) -> String {
        let (text, class) = match self {
            Mark::Undocumented => ("undocumented".to_string(), "td-undoc"),
            Mark::UndocumentedMembers {
                undocumented,
                total,
                noun,
            } => (
                format!("{undocumented} of {total} {noun} undocumented"),
                "td-undoc",
            ),
            Mark::History(p) => (format!("history: {}", quote(p)), "td-history"),
            Mark::Capitals(w) => (format!("capitals: {}", quote(w)), "td-capitals"),
            Mark::Flags(n) => (format!("{n} bool fields"), "td-flags"),
            Mark::Options(n) => (format!("{n} option fields"), "td-options"),
            Mark::StringEnum(f) => (format!("string-valued: {}", quote(f)), "td-string-enum"),
        };
        format!("[{}]{{.td-mark .{class}}}", escape_text(&text))
    }
}

/// Renders marks as one line, separated by spaces.
pub fn render_line(marks: &[Mark]) -> String {
    marks.iter().map(Mark::render).collect::<Vec<_>>().join(" ")
}

fn quote(items: &[String]) -> String {
    let mut s = items
        .iter()
        .take(MAX_QUOTED)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > MAX_QUOTED {
        s.push_str(", …");
    }
    s
}

fn history_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let alternation = HISTORY_PATTERNS.join("|");
        Regex::new(&format!("(?i)(?:{alternation})")).expect("history patterns are valid")
    })
}

fn capitals_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Z]{4,}\b").expect("capitals pattern is valid"))
}

/// Pushes `s` onto `v` unless an equal string, ignoring case, is there.
fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x.eq_ignore_ascii_case(s)) {
        v.push(s.to_string());
    }
}

/// The marks a doc comment earns by its text: history and capitals. An absent
/// doc earns [`Mark::Undocumented`].
pub fn doc_marks(doc: Option<&str>, language: Language) -> Vec<Mark> {
    let Some(doc) = doc.filter(|d| !d.trim().is_empty()) else {
        return vec![Mark::Undocumented];
    };
    let mut marks = Vec::new();
    let prose = without_code_blocks(doc, language);
    let mut phrases = Vec::new();
    for m in history_regex().find_iter(&prose) {
        let phrase = m
            .as_str()
            .trim_start_matches(|c: char| c.is_whitespace() || c == '(');
        push_unique(&mut phrases, phrase);
    }
    if !phrases.is_empty() {
        marks.push(Mark::History(phrases));
    }
    let words_text = without_inline_code(&prose, language);
    let mut words = Vec::new();
    for m in capitals_regex().find_iter(&words_text) {
        if !CAPITALS_ALLOWED.contains(&m.as_str()) {
            push_unique(&mut words, m.as_str());
        }
    }
    if !words.is_empty() {
        marks.push(Mark::Capitals(words));
    }
    marks
}

/// The marks shown on a type's mark line: its own doc's marks, undocumented
/// members, and the shape marks of a record.
pub fn type_marks(t: &TypeDecl, language: Language) -> Vec<Mark> {
    let mut marks = doc_marks(t.doc.as_deref(), language);
    let undocumented = t.members.iter().filter(|m| is_undocumented(m)).count();
    if undocumented > 0 {
        marks.push(Mark::UndocumentedMembers {
            undocumented,
            total: t.members.len(),
            noun: t.kind.member_noun(),
        });
    }
    if t.kind.has_fields() {
        let flags = t.members.iter().filter(|m| is_bool(&m.ty)).count();
        if flags >= MANY_FIELDS {
            marks.push(Mark::Flags(flags));
        }
        let options = t
            .members
            .iter()
            .filter(|m| is_option(&m.ty, language))
            .count();
        if options >= MANY_FIELDS {
            marks.push(Mark::Options(options));
        }
        let stringly: Vec<String> = t
            .members
            .iter()
            .filter(|m| is_string(&m.ty, language) && has_enum_like_name(&m.name))
            .map(|m| m.name.clone())
            .collect();
        if !stringly.is_empty() {
            marks.push(Mark::StringEnum(stringly));
        }
    }
    marks
}

/// Whether a member has no doc comment.
pub fn is_undocumented(m: &Member) -> bool {
    m.doc.as_deref().is_none_or(|d| d.trim().is_empty())
}

/// Whether a type has a doc comment.
pub fn is_documented(t: &TypeDecl) -> bool {
    t.doc.as_deref().is_some_and(|d| !d.trim().is_empty())
}

fn is_bool(ty: &str) -> bool {
    ty.trim() == "bool"
}

fn is_option(ty: &str, language: Language) -> bool {
    let ty = ty.trim();
    match language {
        Language::Ocaml => ty == "option" || ty.ends_with(" option") || ty.ends_with(")option"),
        Language::Rust => ty.starts_with("Option<") || ty.starts_with("std::option::Option<"),
    }
}

fn is_string(ty: &str, language: Language) -> bool {
    let ty: String = ty.split_whitespace().collect::<Vec<_>>().join(" ");
    match language {
        Language::Ocaml => ty == "string" || ty == "string option",
        Language::Rust => matches!(
            ty.as_str(),
            "String" | "&str" | "&'static str" | "Option<String>" | "Option<&str>"
        ),
    }
}

fn has_enum_like_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    let last = lower.rsplit('_').next().unwrap_or(&lower);
    ENUM_LIKE_NAMES.contains(&last)
}

/// A doc comment with its code blocks removed.
fn without_code_blocks(doc: &str, language: Language) -> String {
    match language {
        Language::Ocaml => {
            let mut out = String::new();
            let mut rest = doc;
            loop {
                let starts = [rest.find("{["), rest.find("{@"), rest.find("{v")];
                let Some(start) = starts.iter().flatten().min().copied() else {
                    break;
                };
                let end_marker = if rest[start..].starts_with("{v") {
                    "v}"
                } else {
                    "]}"
                };
                out.push_str(&rest[..start]);
                match rest[start + 2..].find(end_marker) {
                    Some(e) => rest = &rest[start + 2 + e + 2..],
                    None => {
                        rest = "";
                        break;
                    }
                }
            }
            out.push_str(rest);
            out
        }
        Language::Rust => {
            let mut out = String::new();
            let mut in_fence = false;
            for line in doc.lines() {
                let t = line.trim_start();
                if t.starts_with("```") || t.starts_with("~~~") {
                    in_fence = !in_fence;
                    continue;
                }
                if !in_fence {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out
        }
    }
}

/// Prose with inline code removed: backtick spans, and in OCaml `[...]`.
fn without_inline_code(prose: &str, language: Language) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    let mut in_ticks = false;
    for c in prose.chars() {
        match c {
            '`' => in_ticks = !in_ticks,
            '[' if language == Language::Ocaml && !in_ticks => depth += 1,
            ']' if language == Language::Ocaml && !in_ticks && depth > 0 => depth -= 1,
            _ if in_ticks || depth > 0 => {}
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;

    fn member(name: &str, ty: &str, doc: Option<&str>) -> Member {
        Member {
            name: name.into(),
            ty: ty.into(),
            mutable: false,
            doc: doc.map(Into::into),
        }
    }

    fn record(members: Vec<Member>) -> TypeDecl {
        TypeDecl {
            name: "r".into(),
            line: 1,
            kind: Kind::Record,
            params: vec![],
            recursive_group: None,
            decl: String::new(),
            decl_spans: vec![],
            doc: Some("A record.".into()),
            members,
            refs: vec![],
        }
    }

    fn history(doc: &str) -> Vec<String> {
        match doc_marks(Some(doc), Language::Ocaml)
            .into_iter()
            .find(|m| matches!(m, Mark::History(_)))
        {
            Some(Mark::History(p)) => p,
            _ => vec![],
        }
    }

    #[test]
    fn history_phrases() {
        assert_eq!(history("See gh#616 and GH#616."), vec!["gh#616"]);
        assert_eq!(
            history("(wave 2 / malaria #2) branch"),
            vec!["wave 2", "malaria #2"]
        );
        assert_eq!(
            history("Per §4.2 of the proposal."),
            vec!["§4.2", "proposal"]
        );
        assert_eq!(
            history("No longer used; previously a list."),
            vec!["No longer", "previously"]
        );
        assert_eq!(
            history("Changed 2026-06-09 (#812)."),
            vec!["2026-06-09", "#812"]
        );
        assert_eq!(history("The v2 path."), vec!["v2"]);
    }

    #[test]
    fn history_ignores_non_history() {
        assert!(history("A rate per day. # heading\n#[derive] x#1 and v10.").is_empty());
        assert!(history("Example {@camdl[ t0 = 2026-01-01 ]} done.").is_empty());
    }

    #[test]
    fn history_mark_quotes_at_most_three() {
        let m = Mark::History(vec!["a".into(), "b".into(), "c".into(), "d".into()]);
        assert_eq!(m.render(), "[history: a, b, c, …]{.td-mark .td-history}");
    }

    #[test]
    fn capitals_skip_acronyms_and_code() {
        let marks = doc_marks(
            Some("Read the JSON ONLY once, [NEVER] `ALWAYS`; MUST hold."),
            Language::Ocaml,
        );
        assert_eq!(
            marks,
            vec![Mark::Capitals(vec!["ONLY".into(), "MUST".into()])]
        );
    }

    #[test]
    fn undocumented_type_and_members() {
        let mut t = record(vec![
            member("a", "int", None),
            member("b", "int", Some("B.")),
        ]);
        t.doc = None;
        assert_eq!(
            type_marks(&t, Language::Ocaml),
            vec![
                Mark::Undocumented,
                Mark::UndocumentedMembers {
                    undocumented: 1,
                    total: 2,
                    noun: "fields"
                }
            ]
        );
        assert_eq!(
            type_marks(&t, Language::Ocaml)[1].render(),
            "[1 of 2 fields undocumented]{.td-mark .td-undoc}"
        );
    }

    #[test]
    fn flags_options_and_string_enums() {
        let d = Some("D.");
        let t = record(vec![
            member("a", "bool", d),
            member("b", "bool", d),
            member("c", "bool", d),
            member("x", "int option", d),
            member("y", "(int * int) option", d),
            member("z", "string option", d),
            member("backend_kind", "string", d),
            member("op", "string option", d),
            member("loop", "string", d),
        ]);
        assert_eq!(
            type_marks(&t, Language::Ocaml),
            vec![
                Mark::Flags(3),
                Mark::Options(4),
                Mark::StringEnum(vec!["backend_kind".into(), "op".into()])
            ]
        );
        let two = record(vec![member("a", "bool", d), member("b", "bool", d)]);
        assert!(type_marks(&two, Language::Ocaml).is_empty());
    }

    #[test]
    fn shape_marks_only_on_records() {
        let mut t = record(vec![
            member("A", "bool", Some("x")),
            member("B", "bool", Some("x")),
            member("C", "bool", Some("x")),
        ]);
        t.kind = Kind::Variant;
        assert!(type_marks(&t, Language::Ocaml).is_empty());
    }
}
