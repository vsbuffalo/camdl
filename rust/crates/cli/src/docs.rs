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
}

// Paths are four levels up from this file (`rust/crates/cli/src/docs.rs`) to
// the repo root — the same `include_str!` convention as
// `rust/crates/ir/src/envelope.rs` (`../../../../ir/VERSION`).
const TOPICS: &[Topic] = &[
    Topic {
        slug: "agents",
        aliases: &["agent", "ai"],
        summary: "Orientation for agents using camdl: canonical workflow, idioms, when to ask the human",
        body: include_str!("../../../../docs/agents.md"),
    },
    Topic {
        slug: "getting-started",
        aliases: &["intro", "start", "tutorial", "modeling", "model"],
        summary: "Write your first .camdl by example: compartments, transitions, rates, stratification",
        body: include_str!("../../../../docs/intro.md"),
    },
    Topic {
        slug: "language",
        aliases: &["dsl", "spec", "syntax"],
        summary: "Full DSL reference: units & dimensions, parameter kinds, tables, forcings",
        body: include_str!("../../../../docs/camdl-language-spec.md"),
    },
    Topic {
        slug: "language-changes",
        aliases: &["lang-changes", "migrations", "breaking", "what-changed"],
        summary: "Breaking DSL changes with migrations (old → new) — check here when a model that should compile is rejected",
        body: include_str!("../../../../docs/language-changes.md"),
    },
    Topic {
        slug: "changelog",
        aliases: &["release-notes", "releases", "history"],
        summary: "Full changelog (all changes by version): DSL, CLI, inference, formats",
        body: include_str!("../../../../CHANGELOG.md"),
    },
    Topic {
        slug: "inference",
        aliases: &["fit", "fitting", "mcmc"],
        summary: "Fitting: particle filter, IF2, PGAS, ODE gradient sampling (nuts/mh), profiles, diagnostics",
        body: include_str!("../../../../docs/inference.md"),
    },
    Topic {
        slug: "diagnosing-fits",
        aliases: &["diagnose", "diagnostics", "troubleshoot"],
        summary: "When a fit won't behave: model vs inference, the synthetic self-consistency test, which diagnostic to read",
        body: include_str!("../../../../docs/diagnosing-fits.md"),
    },
    Topic {
        slug: "workflow",
        aliases: &["calibrate", "pipeline"],
        summary: "The canonical fit workflow: check → simulate → survey → fit → diagnose → refine → validate",
        body: include_str!("../../../../docs/workflow.md"),
    },
    Topic {
        slug: "fit-toml",
        aliases: &["fittoml", "fit-config"],
        summary: "fit.toml reference: model/data/estimate/fixed/stages, priors, stage algorithms",
        body: include_str!("../../../../docs/fit-toml.md"),
    },
    Topic {
        slug: "concepts",
        aliases: &["why", "identifiability"],
        summary: "The reasoning: identifiability, what priors are for, why a failing gate is information",
        body: include_str!("../../../../docs/concepts.md"),
    },
    Topic {
        slug: "features",
        aliases: &["catalogue", "catalog"],
        summary: "Feature catalogue, with the pomp comparison",
        body: include_str!("../../../../docs/user-features.md"),
    },
    Topic {
        slug: "backends",
        aliases: &["runtimes", "simulate"],
        summary: "Simulation backends: Gillespie, chain-binomial, ODE",
        body: include_str!("../../../../docs/runtimes.md"),
    },
    Topic {
        slug: "data",
        aliases: &["observations", "obs"],
        summary: "Observation data format (a table over time x dims)",
        body: include_str!("../../../../docs/camdl-data-spec.md"),
    },
    Topic {
        slug: "dates",
        aliases: &["calendar", "time", "origin", "anchored"],
        summary: "Calendar time: anchoring a model with `origin`, dated data columns, reading results back as dates",
        body: include_str!("../../../../docs/dates.md"),
    },
    Topic {
        slug: "debugging",
        aliases: &["debug", "eval", "trace"],
        summary: "Debugging via `camdl eval` and the substep tracer",
        body: include_str!("../../../../docs/debugging.md"),
    },
    Topic {
        slug: "mre",
        aliases: &["bundle", "repro", "reproducible", "report-bug"],
        summary: "Package a minimal reproducible example (`camdl mre fit`) to send the maintainer when a fit misbehaves",
        body: include_str!("../../../../docs/mre.md"),
    },
    Topic {
        slug: "model-comparison",
        aliases: &["compare", "comparison", "elpd", "evidence", "jeffreys", "holdout"],
        summary: "Reading `camdl compare`: elpd and the prequential score, LR and the Jeffreys tiers, se(Δ), PIT coverage, held-out evaluation",
        body: include_str!("../../../../docs/methods/model-comparison.md"),
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
pub(crate) fn topic_body(name: &str) -> Option<&'static str> {
    resolve(name).map(|t| t.body)
}

fn print_index() {
    println!("camdl docs — usage guides embedded in this binary (offline, version-matched).\n");
    let w = TOPICS.iter().map(|t| t.slug.len()).max().unwrap_or(0);
    for t in TOPICS {
        println!("  {:<w$}  {}", t.slug, t.summary, w = w);
    }
    println!();
    println!("  camdl docs <topic>        print a guide");
    println!("  camdl docs --search TERM  find where a term is discussed");
    println!("  camdl docs --all          print every guide (full corpus)");
    println!("  camdl docs --json         machine-readable topic index");
}

fn print_json() {
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
    let out = serde_json::json!({ "topics": topics });
    println!("{}", serde_json::to_string_pretty(&out).expect("serialize topic index"));
}

fn print_all() {
    for t in TOPICS {
        println!("<!-- camdl docs: {} -->", t.slug);
        print!("{}", t.body);
        if !t.body.ends_with('\n') {
            println!();
        }
        println!();
    }
}

/// Linear scan, no index (the corpus is a few hundred KB). Default match is
/// case-insensitive, multi-term AND, line-oriented: a line matches if it
/// contains every whitespace-separated term. Output is `slug:lineno: line`,
/// grouped by topic ordered by hit count — greppable, agent-friendly.
fn search(query: &str) {
    let terms: Vec<String> = query.to_lowercase().split_whitespace().map(String::from).collect();
    if terms.is_empty() {
        eprintln!("error: empty search query");
        std::process::exit(1);
    }

    let mut results: Vec<(&Topic, Vec<(usize, String)>)> = Vec::new();
    for t in TOPICS {
        let hits: Vec<(usize, String)> = t
            .body
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
        eprintln!("no matches for '{}'", query);
        std::process::exit(1);
    }
    for (t, hits) in &results {
        for (lineno, line) in hits {
            println!("{}:{}: {}", t.slug, lineno, line);
        }
    }
}

pub fn cmd_docs(a: &crate::args::DocsArgs) {
    if let Some(q) = &a.search {
        search(q);
        return;
    }
    if a.all {
        print_all();
        return;
    }
    if a.json {
        print_json();
        return;
    }
    match &a.topic {
        None => print_index(),
        Some(name) => match resolve(name) {
            Some(t) => print!("{}", t.body),
            None => {
                eprintln!("error: unknown topic '{}'", name);
                eprintln!(
                    "available topics: {}",
                    TOPICS.iter().map(|t| t.slug).collect::<Vec<_>>().join(", ")
                );
                std::process::exit(1);
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
