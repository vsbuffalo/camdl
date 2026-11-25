//! The label a command's invocation is named by, reached as `@label`.
//!
//! [`Label::parse`] is the one rule every command's `--label` passes, applied
//! by [`crate::StoreRequest::new`], so an invocation records only a label a
//! handle can name. `~` is refused because `@L~n` names the `n`th previous
//! holder of `L`; `/` because a label names a directory of the link tree; `.`
//! and `..` because they are not directory names.

use crate::error::{Error, Result};

/// Longest label, in characters, after trimming.
const MAX_CHARS: usize = 64;

/// A label: one to 64 characters after trimming surrounding whitespace, each
/// an ASCII letter or digit, a space, a comma, a dot, an underscore or a
/// hyphen; and not `.` or `..`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label(String);

impl Label {
    /// The label `raw` names, trimmed. Refused with [`Error::InvalidLabel`],
    /// saying which part of the rule it breaks.
    pub fn parse(raw: &str) -> Result<Label> {
        let refuse = |why: String| Err(Error::InvalidLabel { label: raw.to_string(), why });
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return refuse("it is empty after trimming".into());
        }
        let n = trimmed.chars().count();
        if n > MAX_CHARS {
            return refuse(format!("it is {n} characters after trimming, and at most {MAX_CHARS} are allowed"));
        }
        if let Some((i, c)) = trimmed.chars().enumerate().find(|(_, c)| !allowed(*c)) {
            return refuse(format!(
                "`{c}` at position {} is not allowed: a label holds letters, digits, spaces, commas, dots, \
                 underscores and hyphens",
                i + 1
            ));
        }
        if trimmed == "." || trimmed == ".." {
            return refuse("`.` and `..` are not directory names".into());
        }
        Ok(Label(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn allowed(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, ' ' | ',' | '.' | '_' | '-')
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl serde::Serialize for Label {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

/// A record's label is read by [`Label::parse`], so a record holding one the
/// rule refuses does not read.
impl<'de> serde::Deserialize<'de> for Label {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Label::parse(&raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::Label;

    fn why(raw: &str) -> String {
        match Label::parse(raw) {
            Err(crate::Error::InvalidLabel { why, .. }) => why,
            other => panic!("`{raw}` must be refused as a label, got {other:?}"),
        }
    }

    #[test]
    fn the_documented_shapes_are_labels_as_given() {
        for ok in ["narrow R0, take 1", "iota free", "log_normal R0 prior", "a", "a-b_c.d 0,1", "...", ".a"] {
            assert_eq!(Label::parse(ok).unwrap().as_str(), ok);
        }
    }

    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(Label::parse("   narrow R0   ").unwrap().as_str(), "narrow R0");
    }

    #[test]
    fn an_empty_label_is_refused() {
        for empty in ["", "   ", "\t \n"] {
            assert!(why(empty).contains("empty"));
        }
    }

    #[test]
    fn a_label_is_at_most_64_characters() {
        assert!(Label::parse(&"a".repeat(64)).is_ok());
        assert!(why(&"a".repeat(65)).contains("64"));
    }

    #[test]
    fn each_character_outside_the_rule_is_named() {
        for (raw, bad) in [("f~1", "~"), ("R0/2", "/"), ("alpha=2", "="), ("name:tag", ":"), ("a;b", ";"), ("emoji 🎯", "🎯")] {
            assert!(why(raw).contains(bad), "the refusal of `{raw}` names `{bad}`: {}", why(raw));
        }
    }

    #[test]
    fn dot_and_dot_dot_are_refused() {
        for raw in [".", "..", " .. "] {
            assert!(why(raw).contains("directory"), "{}", why(raw));
        }
    }
}
