//! Sections of a method's report, as lines. Never writes: the caller hands
//! the lines to the stream's painter.

use std::collections::BTreeMap;

use sim::inference::diagnostic::{fmt_rate, AcceptanceKernel};

use super::ChainWarmup;
use crate::fit::method_result::{PosteriorDiagnostics, Stat};
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

// ─── Text layout shared by the sections ───────────────────────────────────────

/// Significant figures the posterior-mean column carries. Four separates the
/// values a reader compares across rows (`0.001854` against `240.8`) without
/// implying a precision the Monte-Carlo error does not support.
///
/// The export formats (`--format md` / `latex` / `json`) deliberately do NOT
/// share this: they stay at a fixed `{:.6}`. A terminal column is read by eye
/// and scanned across rows, so relative precision and a narrow column win;
/// an export is parsed or typeset downstream, where a fixed number of decimals
/// is easier to align, diff and re-read than a width that varies per value.
/// The two are different jobs, not a drift to reconcile — do not "fix" one to
/// match the other.
pub(crate) const POSTERIOR_MEAN_SIG_FIGS: usize = 4;

/// Significant figures the two efficiency headlines carry.
///
/// Fixed decimals were the wrong invariant here for a sharper reason than in
/// the mean column: the quantity is a ratio whose numerator is an ESS and whose
/// denominator is an iteration or a second count, so it gets SMALLER exactly as
/// the fit gets worse. A min-parameter ESS of 8 over 48,000 raw iterations is
/// 1.7e-4, which `{:.3}` renders `0.000` — the headline reads as "nothing" on
/// the one fit whose reader most needs the number, and two such fits an order
/// of magnitude apart render identically. Significant figures keep three digits
/// at any magnitude, and [`sig_figs`] falls back to scientific notation below
/// `1e-4` rather than growing a run of leading zeros.
pub(crate) const EFFICIENCY_SIG_FIGS: usize = 3;

/// Column the section rules and the `--explain` prose wrap to. Wide enough for
/// the per-chain log-likelihood table, narrow enough to survive an 80-column
/// terminal without wrapping.
pub(crate) const SECTION_WIDTH: usize = 76;

/// Widest a parameter-name column is allowed to grow before names are
/// ellipsized instead. Stratified names are built by suffixing (`I0_ituri`,
/// `phi_split_haut_uele`) and a deeply stratified model can reach a width no
/// terminal helps with; past this point a readable grid is worth more than a
/// complete name.
pub(crate) const NAME_COL_MAX: usize = 44;

/// The width a name column must have for every name to fit, bounded by
/// [`NAME_COL_MAX`] and never narrower than `min` (which keeps a short-named
/// model's table looking as it always did).
///
/// Rust's `{:14}` is a MINIMUM width: it pads a short name and passes a long
/// one through whole, shoving every later column right. Sizing the column to
/// the names actually present is what keeps the grid a grid.
pub(crate) fn name_col_width<'a>(names: impl Iterator<Item = &'a str>, min: usize) -> usize {
    names
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(min, NAME_COL_MAX)
}

/// The width a column of already-rendered numbers must have for every one of
/// them to fit, never narrower than `min`.
///
/// The numeric twin of [`name_col_width`], and it has no upper bound for the
/// same reason that one does: a name can be ellipsized and still identify its
/// row, a number cannot be shortened without changing what it says. Callers
/// render each value once and size the column to those strings — `{:<12.6}`
/// bounds the decimals but not the integer part, so a value at or above `1e6`
/// is fourteen characters in a twelve-wide field and moves every column after
/// it on that row while its neighbours stay put.
pub(crate) fn num_col_width<'a>(rendered: impl Iterator<Item = &'a str>, min: usize) -> usize {
    rendered.map(|v| v.chars().count()).max().unwrap_or(0).max(min)
}

/// Codepoint ranges [`display_width`] treats as occupying no cell of their own:
/// the combining diacritical marks (U+0302, the accent in `R̂`) and the
/// combining marks for symbols.
const COMBINING_MARKS: [(char, char); 2] = [('\u{0300}', '\u{036F}'), ('\u{20D0}', '\u{20F0}')];

/// How many terminal cells `s` occupies.
///
/// `chars().count()` — which is what Rust's own `{:>8}` pads by — is the wrong
/// measure wherever a string carries a combining mark: `R̂` is `R` followed by
/// U+0302 COMBINING CIRCUMFLEX ACCENT, two `char`s the terminal composes into
/// one cell, so a field padded by `char` count comes out a cell short of the
/// column it is meant to fill. Subtracting the combining marks is the whole of
/// the correction these tables need — their labels are ASCII plus a diacritic,
/// with nothing double-width in them — so a full Unicode width table would be a
/// dependency bought for a handful of fixed strings.
pub(crate) fn display_width(s: &str) -> usize {
    s.chars()
        .filter(|c| !COMBINING_MARKS.iter().any(|(lo, hi)| c >= lo && c <= hi))
        .count()
}

/// `s` right-aligned in a field `width` display cells wide: `{:>width$}` with
/// [`display_width`] in place of `chars().count()`.
///
/// Reach for it wherever a padded cell can carry a combining mark, and leave
/// `{:>N}` where the cell is plain ASCII. A hand-placed compensating space
/// would do the same thing here and stop doing it the next time the string
/// changes.
pub(crate) fn right_align(s: &str, width: usize) -> String {
    " ".repeat(width.saturating_sub(display_width(s))) + s
}

/// `name` cut to `width` characters, ellipsized in the MIDDLE.
///
/// The head gives way, not the tail: a stratified parameter's distinguishing
/// part is its suffix (`..._nord_kivu`), so a tail-truncated name would
/// collapse every stratum of one parameter onto the same unreadable row.
pub(crate) fn fit_name(name: &str, width: usize) -> String {
    let n = name.chars().count();
    if n <= width {
        return name.to_string();
    }
    // One char for the ellipsis; the remainder splits with the extra going to
    // the tail, which is the half that identifies the row.
    let keep = width.saturating_sub(1);
    let head = keep / 2;
    let tail = keep - head;
    let chars: Vec<char> = name.chars().collect();
    let mut out: String = chars[..head].iter().collect();
    out.push('\u{2026}');
    out.extend(&chars[n - tail..]);
    out
}

/// A number at `sig` significant figures, in the shortest form that keeps
/// them -- the `%g` rule.
///
/// Fixed decimals give every parameter the same ABSOLUTE precision, which is
/// the wrong invariant when one is a coupling rate near `0.002` and the next
/// is a seed size near `241`: six decimals is four wasted columns on one and
/// three meaningless digits on the other. Significant figures give each the
/// same RELATIVE precision, so the column can be scanned.
pub(crate) fn sig_figs(v: f64, sig: usize) -> String {
    debug_assert!(sig >= 1, "at least one significant figure");
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    // `%g`'s own switch. Below the window the fixed form is mostly leading
    // zeros; above it, the fixed form would print MORE significant digits than
    // the budget claims (`6322125` asserts seven), which is the failure this
    // function exists to avoid.
    if exp < -4 || exp >= sig as i32 {
        return format!("{:.*e}", sig - 1, v);
    }
    let decimals = (sig as i32 - 1 - exp).max(0) as usize;
    format!("{v:.decimals$}")
}

/// `--explain` prose, one entry per section, keyed by the section's own name.
///
/// A term is defined ONCE, in the section it is primarily read in, and never
/// repeated: R̂ appears as a column in `posterior` and again in `latent paths`,
/// but it is `convergence` that shows both its halves side by side, so that is
/// where it is explained. `posterior` therefore has no entry — every column it
/// carries is defined under `convergence`.
///
/// Written as one paragraph each; [`wrap`] lays them out at the rendering
/// width, so re-flowing here changes nothing on screen.
pub(crate) const EXPLAIN: &[(&str, &str)] = &[
    (
        SECTION_CONVERGENCE,
        "R̂ compares within-chain to between-chain variance; 1.0 is perfect \
         agreement. Rank-normalized, so a heavy tail cannot inflate it. Two \
         halves: `bulk` worse means the chains disagree about where the \
         posterior sits; `folded` worse means they agree on location and \
         disagree on spread. `classic` is the unnormalized statistic, shown \
         because a large gap to it is itself informative — it says the \
         disagreement lives in the tails.",
    ),
    (
        SECTION_CHAINS,
        "Ranked on `obs ll` = log p(y | X): does this chain reproduce the data. \
         Transition and complete ll are path densities at each chain's own \
         path, so they are shown but not ranked. `mod-z` is a \
         median-absolute-deviation z-score — robust, so one bad chain cannot \
         inflate the scale that judges it; |mod-z| > 3.5 flags. `forkable` \
         counts draws whose latent path was saved, so a counterfactual can be \
         branched from them; only path-saved draws qualify.",
    ),
    (
        SECTION_LATENT,
        "`frozen-disagree`: cells where every chain sits constant at its own \
         value — each chain holds a single draw, so R̂ is undefined rather than \
         large. `chains frozen`: over the non-constant cells, the fraction of \
         chains that never moved; a cell counts as mixed once one chain moves \
         once, so this is the honest denominator.",
    ),
];

/// Section names. Constants because [`EXPLAIN`] is keyed on them: a renamed
/// section that silently lost its prose would be invisible.
pub(crate) const SECTION_POSTERIOR: &str = "posterior";
pub(crate) const SECTION_CONVERGENCE: &str = "convergence";
pub(crate) const SECTION_CHAINS: &str = "chains";
pub(crate) const SECTION_LATENT: &str = "latent paths";

/// `text` greedily wrapped to `width` columns, one `String` per line.
///
/// Deliberately simple: whitespace-separated, no hyphenation, no re-flowing of
/// a word longer than the width (it takes its own over-long line rather than
/// being cut, because the over-long words here are identifiers). Deterministic,
/// so a rendered block can be asserted whole.
pub(crate) fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        // `chars().count()`, not `len()`: `R̂` and `—` are multi-byte.
        let w = word.chars().count();
        if !line.is_empty() && line.chars().count() + 1 + w > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// The model's parameter legend from its `#'` docs — symbol, name, the prose,
/// and the citation — or nothing.
///
/// Nothing unless `wanted`. The legend is the priors, their citations and
/// their caveats: reference material a modeller reads deliberately, and not
/// the answer `fit summary` is run to get. On the ebola province model it is
/// thirty-six lines carrying two DOIs, all of it above the first number, which
/// is why it moved behind `--parameters` rather than staying the first thing
/// on screen.
pub(crate) fn parameter_legend(
    documented: &BTreeMap<String, ir::parameter::DocBlock>,
    wanted: bool,
) -> String {
    if !wanted || documented.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n  parameters\n");
    for (name, d) in documented {
        s.push_str("    ");
        if let Some(sym) = &d.symbol {
            s.push_str(sym);
            s.push_str("  ");
        }
        s.push_str(name);
        if let Some(t) = &d.text {
            s.push_str("  —  ");
            s.push_str(t);
        }
        if let Some(r) = &d.reference {
            s.push_str("  [");
            s.push_str(r);
            s.push(']');
        }
        s.push('\n');
    }
    s
}

/// The model's constants — each typed `let` with a constant body, with its
/// value, its declared kind and its `#'` prose — or nothing.
///
/// A constant is part of the model, not a parameter: no fit estimated or held
/// it, so it appears in no method's table. It is listed here, after the
/// parameter legend and under the same flag, because its value is an
/// assumption the fit rests on. Every constant is listed, documented or not.
pub(crate) fn constant_legend(
    constants: &[ir::parameter::Constant],
    documented: &BTreeMap<String, ir::parameter::DocBlock>,
    wanted: bool,
) -> String {
    if !wanted || constants.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n  constants (typed `let`s: part of the model, not parameters)\n");
    for c in constants {
        s.push_str("    ");
        let d = documented.get(&c.name);
        if let Some(sym) = d.and_then(|d| d.symbol.as_ref()) {
            s.push_str(sym);
            s.push_str("  ");
        }
        s.push_str(&format!("{} = {} ({})", c.name, c.value, c.kind.as_str()));
        if let Some(t) = d.and_then(|d| d.text.as_ref()) {
            s.push_str("  —  ");
            s.push_str(t);
        }
        if let Some(r) = d.and_then(|d| d.reference.as_ref()) {
            s.push_str("  [");
            s.push_str(r);
            s.push(']');
        }
        s.push('\n');
    }
    s
}

/// Parameter names worst-R̂ first — the order every per-parameter table in the
/// text format uses.
///
/// Alphabetical order buries the parameter a reader opened the summary to find
/// among the healthy ones; on the ebola province fit the worst of thirty sat
/// sixth. Parameters carrying no R̂ sort last: they are a different finding,
/// and `convergence` names each one's reason under its table. Ties break
/// alphabetically, so the order is stable between runs of the same fit.
pub(crate) fn by_rhat_desc<'a>(
    diag: &PosteriorDiagnostics,
    names: impl Iterator<Item = &'a str>,
) -> Vec<&'a str> {
    let mut v: Vec<&str> = names.collect();
    v.sort_by(|a, b| {
        // `(has_no_rhat, value)`: `∞` is a real answer and the worst one, so it
        // sorts above every finite R̂ rather than with the unscored.
        let key = |n: &str| match diag.per_param.get(n).map(|p| p.rhat()) {
            Some(Stat::Infinite) => (0u8, f64::INFINITY),
            Some(Stat::Value(x)) => (0, x),
            _ => (1, 0.0),
        };
        let (ka, va) = key(a);
        let (kb, vb) = key(b);
        ka.cmp(&kb)
            .then(vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.cmp(b))
    });
    v
}

/// The parameter carrying the worst R̂, and that R̂ — the name
/// [`PosteriorDiagnostics::max_rhat_status`] computes the value of but does not
/// return, so the verdict can say WHICH parameter failed in the same breath as
/// saying that one did.
///
/// The same filter as that method's `Reported` arm: finite values only. Ties
/// keep the first in name order, so the verdict is stable between runs.
pub(crate) fn worst_rhat_param(diag: &PosteriorDiagnostics) -> Option<(&str, f64)> {
    let mut best: Option<(&str, f64)> = None;
    for (name, p) in &diag.per_param {
        let Some(v) = p.rhat().finite() else { continue };
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((name.as_str(), v)),
        }
    }
    best
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
