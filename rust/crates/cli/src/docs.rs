//! `camdl docs [TOPIC]` — embedded, version-locked usage documentation.
//!
//! Bodies are baked into the binary via `include_str!`, so a `camdl docs`
//! result always matches THIS build of camdl and works with no network and
//! no checkout — the discoverability gap a downstream agent hits when
//! `AGENTS.md`/`docs/` aren't on its filesystem.
//!
//! The `TOPICS` table is the curation boundary: only user-facing docs get a
//! slug, so dev-internal specs never surface here.
//!
//! v1 emits raw markdown to stdout (pipe/agent friendly). TTY rendering +
//! tree-sitter syntax highlighting are a planned v2 layer; see
//! `docs/dev/proposals/2026-06-01-camdl-docs-subcommand.md`.

/// A servable documentation topic. `body` is embedded at compile time.
struct Topic {
    slug: &'static str,
    aliases: &'static [&'static str],
    summary: &'static str,
    body: &'static str,
    /// The doc's path under `docs/` when the website publishes it, so a figure
    /// marker can be printed as a link to the drawing (`terminal_text`).
    /// `None` for an embedded file the site does not render.
    page: Option<&'static str>,
}

/// Where the website is served; `site-url` in `docs/_quarto.yml`, which a test
/// holds this to.
const SITE_URL: &str = "https://vsbuffalo.github.io/camdl/";

// Paths are four levels up from this file (`rust/crates/cli/src/docs.rs`) to
// the repo root — the same `include_str!` convention as
// `rust/crates/ir/src/envelope.rs` (`../../../../ir/VERSION`).
const TOPICS: &[Topic] = &[
    Topic {
        slug: "agents",
        aliases: &["agent", "ai"],
        summary: "Orientation for agents using camdl: canonical workflow, idioms, when to ask the human",
        body: include_str!("../../../../docs/agents.md"),
        page: Some("agents.md"),
    },
    Topic {
        slug: "getting-started",
        aliases: &["intro", "start", "tutorial", "modeling", "model"],
        summary: "Write your first .camdl by example: compartments, transitions, rates, stratification",
        body: include_str!("../../../../docs/intro.md"),
        page: Some("intro.md"),
    },
    Topic {
        slug: "language",
        aliases: &["dsl", "spec", "syntax"],
        summary: "Full DSL reference: units & dimensions, parameter kinds, tables, forcings",
        body: include_str!("../../../../docs/camdl-language-spec.md"),
        page: Some("camdl-language-spec.md"),
    },
    Topic {
        slug: "language-changes",
        aliases: &["lang-changes", "migrations", "breaking", "what-changed"],
        summary: "Breaking DSL changes with migrations (old → new) — check here when a model that should compile is rejected",
        body: include_str!("../../../../docs/language-changes.md"),
        page: Some("language-changes.md"),
    },
    Topic {
        slug: "changelog",
        aliases: &["release-notes", "releases", "history"],
        summary: "Full changelog (all changes by version): DSL, CLI, inference, formats",
        body: include_str!("../../../../CHANGELOG.md"),
        page: None,
    },
    Topic {
        slug: "inference",
        aliases: &["fit", "fitting", "mcmc"],
        summary: "Fitting: particle filter, IF2, PGAS, ODE fitting (mh, NLopt), profiles, diagnostics",
        body: include_str!("../../../../docs/inference.md"),
        page: Some("inference.md"),
    },
    Topic {
        slug: "diagnosing-fits",
        aliases: &["diagnose", "diagnostics", "troubleshoot"],
        summary: "When a fit won't behave: model vs inference, the synthetic self-consistency test, which diagnostic to read",
        body: include_str!("../../../../docs/diagnosing-fits.md"),
        page: Some("diagnosing-fits.md"),
    },
    Topic {
        slug: "workflow",
        aliases: &["calibrate", "pipeline"],
        summary: "The canonical fit workflow: check → simulate → survey → fit → diagnose → refine → validate",
        body: include_str!("../../../../docs/workflow.md"),
        page: Some("workflow.md"),
    },
    Topic {
        slug: "fit-toml",
        aliases: &["fittoml", "fit-config"],
        summary: "fit.toml reference: model/data/estimate/fixed/method/config, chain starts, priors, method algorithms",
        body: include_str!("../../../../docs/fit-toml.md"),
        page: Some("fit-toml.md"),
    },
    Topic {
        slug: "concepts",
        aliases: &["why", "identifiability"],
        summary: "The reasoning: identifiability, what priors are for, why a failing gate is information",
        body: include_str!("../../../../docs/concepts.md"),
        page: Some("concepts.md"),
    },
    Topic {
        slug: "features",
        aliases: &["catalogue", "catalog"],
        summary: "Feature catalogue, with the pomp comparison",
        body: include_str!("../../../../docs/user-features.md"),
        page: Some("user-features.md"),
    },
    Topic {
        slug: "backends",
        aliases: &["runtimes", "simulate"],
        summary: "Simulation backends: Gillespie, chain-binomial, ODE",
        body: include_str!("../../../../docs/runtimes.md"),
        page: Some("runtimes.md"),
    },
    Topic {
        slug: "data",
        aliases: &["observations", "obs"],
        summary: "Observation data format (a table over time x dims)",
        body: include_str!("../../../../docs/camdl-data-spec.md"),
        page: Some("camdl-data-spec.md"),
    },
    Topic {
        slug: "dates",
        aliases: &["calendar", "time", "origin", "anchored"],
        summary: "Calendar time: anchoring a model with `origin`, dated data columns, reading results back as dates",
        body: include_str!("../../../../docs/dates.md"),
        page: Some("dates.md"),
    },
    Topic {
        slug: "debugging",
        aliases: &["debug", "eval", "trace"],
        summary: "Debugging via `camdl eval` and the substep tracer",
        body: include_str!("../../../../docs/debugging.md"),
        page: Some("debugging.md"),
    },
    Topic {
        slug: "mre",
        aliases: &["bundle", "repro", "reproducible", "report-bug"],
        summary: "Package a minimal reproducible example (`camdl mre fit`) to send the maintainer when a fit misbehaves",
        body: include_str!("../../../../docs/mre.md"),
        page: Some("mre.md"),
    },
    Topic {
        slug: "model-comparison",
        aliases: &["compare", "comparison", "elpd", "evidence", "jeffreys", "holdout"],
        summary: "Reading `camdl compare`: elpd and the prequential score, LR and the Jeffreys tiers, se(Δ), PIT coverage, held-out evaluation",
        body: include_str!("../../../../docs/methods/model-comparison.md"),
        page: Some("methods/model-comparison.md"),
    },
];

fn resolve(name: &str) -> Option<&'static Topic> {
    let n = name.to_lowercase();
    TOPICS
        .iter()
        .find(|t| t.slug == n || t.aliases.contains(&n.as_str()))
}

/// The body of one topic, by slug or alias — the same text `camdl docs <slug>`
/// prints.
///
/// The seam another command uses to serve a guide from its own surface
/// (`camdl compare --explain`). It exists so there is exactly one
/// `include_str!` per doc: a second one would put a second copy of the file in
/// the binary, and the two would drift the first time a topic is renamed or
/// re-pathed.
pub(crate) fn topic_body(name: &str) -> Option<String> {
    resolve(name).map(terminal_text)
}

/// The text `camdl docs` prints for an embedded doc: the markdown, with the
/// website's figure markers turned into links to the drawings.
///
/// A doc places a web-only figure with an HTML comment on its own line,
///
///     <!-- figure: transition-anatomy -->
///
/// and may close it with `<!-- /figure -->` after a text diagram, which the
/// website then replaces with the drawing (`docs/_web/figures.lua`; the
/// conventions are in `docs/dev/documentation.md`). The markers are
/// instructions to the site build, so the terminal never sees them: an
/// opening marker becomes one line pointing at the figure's anchor on the
/// website, and a closing marker is dropped. A text diagram between the two is
/// kept, since it is the terminal's version of the figure. A marker may span
/// lines (a long caption). For a doc the site does not publish, an opening
/// marker is dropped together with the blank line after it. Every other
/// comment, the doc-test preamble markers among them, is printed unchanged.
///
/// The website follows the latest main, so the drawing a link reaches may be
/// newer than this binary.
fn terminal_text(topic: &Topic) -> String {
    let (body, page) = (topic.body, topic.page);
    let mut out = String::with_capacity(body.len());
    let mut lines = body.lines();
    let mut drop_blank = false;
    while let Some(line) = lines.next() {
        let t = line.trim();
        let opens = t.starts_with("<!-- figure:");
        if opens || t == "<!-- /figure -->" {
            if !t.ends_with("-->") {
                // Skip the rest of a multi-line marker.
                for rest in lines.by_ref() {
                    if rest.trim_end().ends_with("-->") {
                        break;
                    }
                }
            }
            match (opens, figure_url(page, t)) {
                (true, Some(url)) => {
                    out.push_str(&format!("[Figure on the website: {url}]\n"));
                }
                (true, None) => drop_blank = true,
                (false, _) => {}
            }
            continue;
        }
        if drop_blank && t.is_empty() {
            drop_blank = false;
            continue;
        }
        drop_blank = false;
        out.push_str(line);
        out.push('\n');
    }
    if !body.ends_with('\n') {
        out.pop();
    }
    out
}

/// The website address of the figure an opening marker places:
/// `<site>/<page>.html#fig-<name>`, the anchor `figures.lua` gives the
/// `<figure>` element.
fn figure_url(page: Option<&str>, marker: &str) -> Option<String> {
    let page = page?;
    let name = marker
        .strip_prefix("<!-- figure:")?
        .trim_start()
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .next()
        .filter(|n| !n.is_empty())?;
    let html = page.strip_suffix(".md").unwrap_or(page);
    Some(format!("{SITE_URL}{html}.html#fig-{name}"))
}

fn print_index(out: &crate::out::Out) {
    out.report_line("camdl docs — usage guides embedded in this binary (offline, version-matched).\n");
    let w = TOPICS.iter().map(|t| t.slug.len()).max().unwrap_or(0);
    for t in TOPICS {
        out.report_line(format_args!("  {:<w$}  {}", t.slug, t.summary, w = w));
    }
    out.report_line("");
    out.report_line("  camdl docs <topic>        print a guide");
    out.report_line("  camdl docs --search TERM  find where a term is discussed");
    out.report_line("  camdl docs --all          print every guide (full corpus)");
    out.report_line("  camdl docs --json         machine-readable topic index");
}

fn print_json(out: &crate::out::Out) {
    let topics: Vec<serde_json::Value> = TOPICS
        .iter()
        .map(|t| {
            serde_json::json!({
                "slug": t.slug,
                "aliases": t.aliases,
                "summary": t.summary,
            })
        })
        .collect();
    let index = serde_json::json!({ "topics": topics });
    out.report_line(serde_json::to_string_pretty(&index).expect("serialize topic index"));
}

fn print_all(out: &crate::out::Out) {
    for t in TOPICS {
        let body = terminal_text(t);
        out.report_line(format_args!("<!-- camdl docs: {} -->", t.slug));
        out.report_raw(format_args!("{}", body));
        if !body.ends_with('\n') {
            out.report_line("");
        }
        out.report_line("");
    }
}

/// Linear scan, no index (the corpus is a few hundred KB). Default match is
/// case-insensitive, multi-term AND, line-oriented: a line matches if it
/// contains every whitespace-separated term. Output is `slug:lineno: line`,
/// grouped by topic ordered by hit count — greppable, agent-friendly.
fn search(out: &crate::out::Out, query: &str) {
    let terms: Vec<String> = query.to_lowercase().split_whitespace().map(String::from).collect();
    if terms.is_empty() {
        out.fail("error: empty search query");
    }

    let mut results: Vec<(&Topic, Vec<(usize, String)>)> = Vec::new();
    for t in TOPICS {
        // Search the printed text, so a reported line number is a line of
        // what `camdl docs <slug>` shows.
        let hits: Vec<(usize, String)> = terminal_text(t)
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                let lc = line.to_lowercase();
                terms
                    .iter()
                    .all(|term| lc.contains(term.as_str()))
                    .then(|| (i + 1, line.trim().to_string()))
            })
            .collect();
        if !hits.is_empty() {
            results.push((t, hits));
        }
    }
    results.sort_by_key(|(_, hits)| std::cmp::Reverse(hits.len()));

    if results.is_empty() {
        out.fail(format!("no matches for '{}'", query));
    }
    for (t, hits) in &results {
        for (lineno, line) in hits {
            out.report_line(format_args!("{}:{}: {}", t.slug, lineno, line));
        }
    }
}

pub fn cmd_docs(a: &crate::args::DocsArgs) {
    let out = crate::out::Out::new(false);
    if let Some(q) = &a.search {
        search(&out, q);
        return;
    }
    if a.all {
        print_all(&out);
        return;
    }
    if a.json {
        print_json(&out);
        return;
    }
    match &a.topic {
        None => print_index(&out),
        Some(name) => match resolve(name) {
            Some(t) => out.report_raw(format_args!("{}", terminal_text(t))),
            None => {
                out.say(format_args!("error: unknown topic '{}'", name));
                out.fail(format!(
                    "available topics: {}",
                    TOPICS.iter().map(|t| t.slug).collect::<Vec<_>>().join(", ")
                ));
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topics_present_and_substantial() {
        assert!(!TOPICS.is_empty(), "no topics registered");
        for t in TOPICS {
            assert!(!t.slug.is_empty(), "empty slug");
            assert!(!t.summary.is_empty(), "empty summary for {}", t.slug);
            // Each embedded body is a real doc, not an accidentally-empty include.
            assert!(t.body.len() > 200, "topic {} body suspiciously short ({} bytes)", t.slug, t.body.len());
        }
    }

    #[test]
    fn slugs_and_aliases_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for t in TOPICS {
            assert!(seen.insert(t.slug), "duplicate slug: {}", t.slug);
            for a in t.aliases {
                assert!(seen.insert(*a), "alias '{}' collides with another slug/alias", a);
            }
        }
    }

    fn topic(body: &'static str, page: Option<&'static str>) -> Topic {
        Topic { slug: "t", aliases: &[], summary: "s", body, page }
    }

    #[test]
    fn a_published_doc_prints_each_figure_as_a_link() {
        let doc = "Intro.\n\n<!-- figure: a -->\n\nTable.\n\n<!-- figure: b, caption: B. -->\n```\nA --> B\n```\n<!-- /figure -->\n\nEnd.\n";
        assert_eq!(
            terminal_text(&topic(doc, Some("guide/x.md"))),
            "Intro.\n\n[Figure on the website: https://vsbuffalo.github.io/camdl/guide/x.html#fig-a]\n\nTable.\n\n\
             [Figure on the website: https://vsbuffalo.github.io/camdl/guide/x.html#fig-b]\n```\nA --> B\n```\n\nEnd.\n"
        );
    }

    #[test]
    fn an_unpublished_doc_drops_figure_markers_and_keeps_the_text_diagram() {
        let doc = "Intro.\n\n<!-- figure: a -->\n\nTable.\n\n<!-- figure: b -->\n```\nA --> B\n```\n<!-- /figure -->\n\nEnd.\n";
        assert_eq!(terminal_text(&topic(doc, None)), "Intro.\n\nTable.\n\n```\nA --> B\n```\n\nEnd.\n");
    }

    #[test]
    fn a_multi_line_marker_is_one_link() {
        let doc = "Before.\n\n<!-- figure: a, caption: a long\ncaption that wraps -->\n\nAfter.\n";
        assert_eq!(
            terminal_text(&topic(doc, Some("p.md"))),
            "Before.\n\n[Figure on the website: https://vsbuffalo.github.io/camdl/p.html#fig-a]\n\nAfter.\n"
        );
    }

    #[test]
    fn other_comments_are_printed_unchanged() {
        // The doc-test preamble markers are read by agents through `camdl docs`.
        let doc = "<!-- camdl-doctest-preamble -->\n```camdl\nx\n```\n";
        assert_eq!(terminal_text(&topic(doc, Some("p.md"))), doc);
    }

    #[test]
    fn no_embedded_topic_prints_a_raw_figure_marker() {
        // Positive control: the intro carries a marker, so the rewrite is exercised.
        let intro = resolve("intro").unwrap();
        assert!(intro.body.contains("<!-- figure:"));
        assert!(terminal_text(intro).contains("intro.html#fig-transition-anatomy"));
        for t in TOPICS {
            let text = terminal_text(t);
            assert!(!text.contains("<!-- figure:"), "topic {} prints a figure marker", t.slug);
            assert!(!text.contains("<!-- /figure -->"), "topic {} prints a figure close marker", t.slug);
        }
    }

    /// A link is only right if the page path names the file that was embedded,
    /// the site publishes that file, and the site is where SITE_URL says.
    #[test]
    fn page_paths_name_the_embedded_file_and_a_published_page() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let quarto = std::fs::read_to_string(repo.join("docs/_quarto.yml")).expect("read docs/_quarto.yml");
        assert!(quarto.contains(&format!("site-url: {SITE_URL}")), "SITE_URL disagrees with site-url in docs/_quarto.yml");
        for t in TOPICS {
            let Some(page) = t.page else { continue };
            let on_disk = std::fs::read_to_string(repo.join("docs").join(page)).expect("read page");
            assert_eq!(on_disk, t.body, "topic {}: page {} is not the embedded file", t.slug, page);
            assert!(quarto.contains(&format!("- {page}\n")), "topic {}: {} is not in the site's render list", t.slug, page);
        }
    }

    #[test]
    fn resolve_handles_slug_alias_case_and_miss() {
        // exact slug
        assert_eq!(resolve("inference").map(|t| t.slug), Some("inference"));
        // alias → canonical slug
        assert_eq!(resolve("fit").map(|t| t.slug), Some("inference"));
        // case-insensitive
        assert_eq!(resolve("INFERENCE").map(|t| t.slug), Some("inference"));
        // negative control: a name that is neither a slug nor an alias
        assert!(resolve("definitely-not-a-topic").is_none());
    }
}
