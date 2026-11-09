//! Sections of a method's report, as lines. Never writes: the caller hands
//! the lines to the stream's painter.

use std::collections::BTreeMap;

use sim::inference::diagnostic::{fmt_rate, AcceptanceKernel};

use super::ChainWarmup;
use crate::out::paint::{Line, Span, Style};

/// The acceptance table at the end of a PMMH or `mh` run: one row per chain
/// in chain order, its acceptance rate against the kernel's healthy band, and
/// beside it the scale its draws were proposed under and how many warm-up
/// steps the scale sat at its floor. A chain with no warm-up record (nothing
/// adapted) shows `—` there.
pub fn acceptance_table(
    rates: &BTreeMap<usize, f64>,
    warmup: &[ChainWarmup],
    kernel: AcceptanceKernel,
) -> Vec<Line> {
    let (lo, hi) = kernel.healthy_band();
    let mut lines = vec![
        Line::default(),
        Line(vec![Span::plain("acceptance rates:")]),
        Line(vec![Span::plain(format!(
            "  {:<5}  {:>10} {:<10}  {:>16}  {:>14}",
            "chain", "acceptance", "", "λ end of warm-up", "steps at floor"))]),
    ];
    for (&chain, &rate) in rates {
        let (glyph, style) = if rate < lo {
            ("✗ too low", Style::Err)
        } else if rate > hi {
            ("~ high", Style::Warn)
        } else {
            ("✓", Style::Ok)
        };
        let (lambda, floor) = match warmup.iter().find(|w| w.chain == chain) {
            Some(w) => (format!("{:.3e}", w.final_scale),
                format!("{} / {}", w.steps_at_scale_floor, w.warmup_steps)),
            None => ("—".to_string(), "—".to_string()),
        };
        lines.push(Line(vec![
            Span::plain(format!("  {:<5}  {:>10} ", chain, fmt_rate(rate))),
            Span::styled(format!("{glyph:<10}"), style),
            Span::plain(format!("  {lambda:>16}  {floor:>14}")),
        ]));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_row_per_chain_with_its_warmup() {
        let rates = BTreeMap::from([(0, 0.0005), (2, 0.396)]);
        let warmup = vec![ChainWarmup {
            chain: 0, warmup_steps: 1000, steps_at_scale_floor: 241,
            final_scale: 4.54e-5, target_accept: 0.337,
        }];
        let lines = acceptance_table(&rates, &warmup, AcceptanceKernel::RandomWalk);
        let text: Vec<String> = lines.iter().map(Line::text).collect();
        assert_eq!(text[1], "acceptance rates:");
        assert!(text[3].starts_with("  0") && text[3].contains("0.050% ✗ too low")
            && text[3].contains("4.540e-5") && text[3].ends_with("241 / 1000"), "{}", text[3]);
        assert!(text[4].starts_with("  2") && text[4].contains("39.6% ✓")
            && text[4].ends_with("—"), "{}", text[4]);
    }
}
