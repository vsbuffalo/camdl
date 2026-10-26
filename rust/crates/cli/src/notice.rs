//! A user-facing notice: one line telling the user something the command did
//! that they did not spell out, at a stated severity.
//!
//! A command collects its notices into a [`Notices`] and emits them once,
//! rather than printing ad hoc `eprintln!("warning: …")` lines from wherever
//! the condition is noticed. Collecting first is what lets a multi-cell run
//! report a condition once instead of once per cell, and lets the same lines
//! be recorded beside the run (`run.json`'s display inputs) as well as shown.
//!
//! Notices are presentation: recorded, never hashed into a run's identity.

use serde::Serialize;

/// How much a notice asks of the user.
///
/// - `Warning`: the command did something the user is likely not to have
///   meant, or that changes what the run means — a parameter dropped from
///   `[estimate]`, for example.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Warning,
}

impl Severity {
    /// The stderr prefix.
    pub fn label(self) -> &'static str {
        match self {
            Severity::Warning => "warning",
        }
    }
}

/// One notice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Notice {
    pub severity: Severity,
    /// A stable kebab-case identifier for the condition (`param-override`),
    /// so a test or a reader of `run.json` can select notices without
    /// matching on prose.
    pub code: &'static str,
    /// One line naming the subject and the values involved. No severity
    /// prefix; [`Notice::render`] adds it.
    pub message: String,
}

impl Notice {
    pub fn warning(code: &'static str, message: impl Into<String>) -> Self {
        Notice { severity: Severity::Warning, code, message: message.into() }
    }

    /// The line as printed: `<severity>: <message>`.
    pub fn render(&self) -> String {
        format!("{}: {}", self.severity.label(), self.message)
    }
}

/// The notices a command collected, in the order they were pushed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Notices(Vec<Notice>);

impl Notices {
    /// Print every notice to stderr, one rendered line each.
    pub fn emit_stderr(&self) {
        for n in &self.0 {
            eprintln!("{}", n.render());
        }
    }
}

impl FromIterator<Notice> for Notices {
    fn from_iter<I: IntoIterator<Item = Notice>>(iter: I) -> Self {
        Notices(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prefixes_the_severity() {
        assert_eq!(Notice::warning("param-override", "y").render(), "warning: y");
    }

    #[test]
    fn notices_serialise_as_a_list_of_records() {
        let ns: Notices = [Notice::warning("kicked-from-estimate", "m")].into_iter().collect();
        let v = serde_json::to_value(&ns).unwrap();
        assert_eq!(
            v,
            serde_json::json!([
                { "severity": "warning", "code": "kicked-from-estimate", "message": "m" }
            ])
        );
    }
}
