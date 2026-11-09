//! The `── diagnostics ──` block: a method's findings, printed once and merged.
//!
//! Findings whose [`MergeKey`]s are equal print as one entry naming every
//! chain, each chain's value beside its id ([`render_entry`]); every other
//! finding prints on its own. Entries keep the order of their first finding.
//! The count line counts findings, not entries, so merging never changes it.

use sim::inference::diagnostic::{
    render_entry, Diagnostic, DiagnosticCollector, DiagnosticKind, HintContext, MergeKey, Severity,
};

use crate::out::paint::{Line, Span, Style};

/// Indent of a multi-line finding's continuation lines: under the text after
/// `  ! `, as the warnings these sentences came from were laid out.
const CONTINUATION: &str = "           ";

/// The findings grouped for printing, in order of each group's first finding.
fn groups(diags: &[Diagnostic]) -> Vec<Vec<&Diagnostic>> {
    let mut out: Vec<Vec<&Diagnostic>> = Vec::new();
    let mut keys: Vec<MergeKey> = Vec::new();
    for d in diags {
        let key = d.kind.merge_key();
        if key != MergeKey::Never {
            if let Some(i) = keys.iter().position(|k| *k == key) {
                out[i].push(d);
                continue;
            }
        }
        keys.push(key);
        out.push(vec![d]);
    }
    out
}

/// The block's lines, or none when there are no findings.
pub fn render(diags: &[Diagnostic], ctx: HintContext) -> Vec<Line> {
    if diags.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![
        Line::default(),
        Line(vec![Span::plain("── diagnostics ──────────────────────────────────────")]),
    ];
    for group in groups(diags) {
        let (icon, style) = match group[0].severity {
            Severity::Info => ("i", Style::Info),
            Severity::Warning => ("!", Style::Warn),
            Severity::Error => ("x", Style::Err),
        };
        let kinds: Vec<&DiagnosticKind> = group.iter().map(|d| &d.kind).collect();
        let text = render_entry(&kinds);
        for (i, part) in text.lines().enumerate() {
            lines.push(if i == 0 {
                Line(vec![Span::plain("  "), Span::styled(icon, style), Span::plain(format!(" {part}"))])
            } else {
                Line(vec![Span::plain(format!("{CONTINUATION}{part}"))])
            });
        }
        for hint in kinds[0].hints(ctx) {
            lines.push(Line(vec![Span::plain(format!("    -> {hint}"))]));
        }
    }
    let count = |s: Severity| diags.iter().filter(|d| d.severity == s).count();
    lines.push(Line(vec![Span::plain(format!(
        "  {} error(s), {} warning(s), {} info",
        count(Severity::Error), count(Severity::Warning), count(Severity::Info)))]));
    lines
}

/// Print the collector's findings to stderr.
#[expect(dead_code, reason = "every run prints its findings in its run-end block now; deleted \
    in its own commit")]
pub fn print(collector: &DiagnosticCollector, ctx: HintContext) {
    let lines = render(&collector.snapshot(), ctx);
    // stderr is unbuffered and a failed write to it has nowhere to be reported.
    let _ = crate::out::paint::stderr().write(&mut std::io::stderr(), &lines);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::diagnostic::{acceptance_diagnostic, AcceptanceKernel};

    fn text(lines: &[Line]) -> String {
        lines.iter().map(|l| l.text()).collect::<Vec<_>>().join("\n")
    }

    /// Two chains' acceptance findings print as one entry; a finding between
    /// them that does not share the key keeps its place; the count is of
    /// findings.
    #[test]
    fn equal_keys_print_once_and_the_count_counts_findings() {
        let c = DiagnosticCollector::new("pmmh");
        let rw = AcceptanceKernel::RandomWalk;
        c.push(acceptance_diagnostic(Some(0), 0.0005, None, rw).unwrap());
        c.push(DiagnosticKind::RhatHigh { param: "beta".into(), rhat: 2.25, threshold: 1.1 });
        c.push(acceptance_diagnostic(Some(1), 0.001, None, rw).unwrap());
        for (chain, n) in [(0, 241), (1, 365)] {
            c.push(DiagnosticKind::ScaleAtFloor {
                chain_id: chain, steps_at_floor: n, warmup_steps: 1000, target_accept: 0.337,
            });
        }
        let out = text(&render(&c.snapshot(), HintContext::default()));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[1], "── diagnostics ──────────────────────────────────────");
        assert_eq!(lines[2], "  x chains 0, 1: chain acceptance rates 0.050% and 0.10% are outside \
                              healthy range [15%, 50%] (random-walk MH).");
        assert!(lines[3].starts_with("  x rank-normalized split R̂ for 'beta' is 2.250"), "{out}");
        assert_eq!(lines[4], "  ! chains 0, 1: the adaptive proposal scale sat at its lower bound (lambda = 4.5e-5)");
        assert_eq!(lines[5], "           for 241 and 365 of 1000 warm-up steps: no scale reached the");
        assert_eq!(lines.last().copied(), Some("  3 error(s), 2 warning(s), 0 info"));
    }

    #[test]
    fn no_findings_print_nothing() {
        assert!(render(&[], HintContext::default()).is_empty());
    }
}
