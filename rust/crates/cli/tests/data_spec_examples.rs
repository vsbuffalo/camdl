//! Execute the observation-file examples printed in `docs/camdl-data-spec.md`
//! through the loader `fit run` uses, so a documented format that does not load
//! fails the gate (gh#879).
//!
//! The language spec's `camdl` snippets are compiled by `camdlc doctest --gate`
//! (`make test-docs`). The data spec's examples cannot be: each is a TSV file
//! that only means something under the stream declaration printed beside it,
//! and the claim to check is that it *loads with the periods the prose says it
//! has* — a run through the real reader, not a compile. The consequence of not
//! checking was concrete: the documented dated window form failed to load in
//! every code path, and the first person to try it was a modeller following
//! this document.
//!
//! ## How an example is marked
//!
//! Pairing is by name in the fence's info string, never by adjacency:
//!
//! ```text
//! ```camdl data-example=ituri-windows preamble=ituri
//! ```tsv   data-example=ituri-windows
//! ```
//!
//! `preamble` names a model under `tests/fixtures/data_spec/` — a complete
//! `.camdl` with no `observations { }` block, plus a `.params.toml` beside it.
//! The harness appends the document's stream to it, compiles, and binds the
//! document's file under that stream.
//!
//! A file the document prints as what *not* to write carries `refused="…"`,
//! whose value is the text the diagnostic must contain:
//!
//! ```text
//! ```tsv data-example=uniform-gap refused="covered by neither"
//! ```
//!
//! The markers live in the fence info string rather than in HTML comments
//! before the fence, which is what makes `dprint fmt` safe on this file: the
//! formatter reflows prose and leaves an info string and a fenced body
//! untouched. (The language spec's `<!-- camdl-doctest-preamble -->` markers
//! are the opposite case, and why that file is on the no-format list.)
//!
//! ## Fail-closed
//!
//! Inside the guarded sections, every `camdl` and every `tsv` fence must carry
//! a `data-example` marker. An unmarked example is a gate failure, not a
//! silent skip — otherwise the harness quietly stops covering the document as
//! the document grows.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The sections whose examples are executed. Every fenced `camdl`/`tsv` block
/// between one of these headings and the next `## ` heading must be marked.
const GUARDED_SECTIONS: &[&str] = &[
    "## What a row covers: `covers` and the window columns",
    "## Missing observations: `NA` is a hole, not a zero",
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap()
}

fn spec_path() -> PathBuf {
    repo_root().join("docs/camdl-data-spec.md")
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/data_spec")
}

fn camdl_bin() -> PathBuf {
    let p = repo_root().join("rust/target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        p.display()
    );
    p
}

fn camdlc() -> PathBuf {
    let p = repo_root().join("ocaml/_build/default/bin/camdlc.exe");
    assert!(
        p.exists(),
        "camdlc missing: {} - run `make build-ocaml`, or gate with `make test-data-spec`",
        p.display()
    );
    p
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir()
        .join(format!("camdl_data_spec_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

// ── Reading the document ────────────────────────────────────────────────────

/// One fenced block inside a guarded section.
struct Fence {
    lang: String,
    info: BTreeMap<String, String>,
    body: String,
    /// 1-based line of the opening fence, so a failure points at the document.
    line: usize,
}

/// Split a fence info string into its language and `key=value` directives.
/// A value may be double-quoted, so it can carry spaces.
fn parse_info(info: &str) -> (String, BTreeMap<String, String>) {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for c in info.chars() {
        match c {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    let lang = tokens.first().cloned().unwrap_or_default();
    let directives = tokens[1.min(tokens.len())..]
        .iter()
        .map(|t| match t.split_once('=') {
            Some((k, v)) => (k.to_string(), v.to_string()),
            None => (t.to_string(), String::new()),
        })
        .collect();
    (lang, directives)
}

/// Every fenced block inside the guarded sections, in document order.
fn fences_in_guarded_sections(text: &str) -> Vec<Fence> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut guarded = false;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("## ") {
            guarded = GUARDED_SECTIONS.contains(&line.trim_end());
        }
        if guarded && line.starts_with("```") {
            let (lang, info) = parse_info(line.trim_start_matches('`'));
            let open = i;
            let mut body = String::new();
            i += 1;
            while i < lines.len() && !lines[i].starts_with("```") {
                body.push_str(lines[i]);
                body.push('\n');
                i += 1;
            }
            out.push(Fence { lang, info, body, line: open + 1 });
        }
        i += 1;
    }
    out
}

/// A stream declaration and the file it is printed beside.
struct Example {
    name: String,
    preamble: String,
    /// The stream block, as the document prints it.
    declaration: String,
    declaration_line: usize,
    file: String,
    file_line: usize,
    /// `Some(text)` when the document prints this file as what not to write;
    /// the loader's diagnostic must contain `text`.
    refused: Option<String>,
}

fn collect_examples(text: &str) -> Vec<Example> {
    let fences = fences_in_guarded_sections(text);
    let mut decls: BTreeMap<String, (String, String, usize)> = BTreeMap::new();
    let mut files: BTreeMap<String, (String, usize, Option<String>)> = BTreeMap::new();

    for f in &fences {
        if f.lang != "camdl" && f.lang != "tsv" {
            continue;
        }
        let name = f.info.get("data-example").unwrap_or_else(|| {
            panic!(
                "{}:{}: a `{}` block in a guarded section carries no \
                 `data-example=<name>` marker. Every example in these sections is \
                 executed; mark it, or move it out of the section.",
                spec_path().display(),
                f.line,
                f.lang
            )
        });
        assert!(!name.is_empty(), "{}:{}: `data-example` needs a name", spec_path().display(), f.line);
        if f.lang == "camdl" {
            let preamble = f.info.get("preamble").unwrap_or_else(|| {
                panic!(
                    "{}:{}: example '{name}' declares a stream but names no \
                     `preamble=<stem>` under tests/fixtures/data_spec/",
                    spec_path().display(),
                    f.line
                )
            });
            let prior = decls.insert(name.clone(), (preamble.clone(), f.body.clone(), f.line));
            assert!(prior.is_none(), "example '{name}' declares a stream twice");
        } else {
            let refused = f.info.get("refused").cloned();
            let prior = files.insert(name.clone(), (f.body.clone(), f.line, refused));
            assert!(prior.is_none(), "example '{name}' prints a file twice");
        }
    }

    let mut out = Vec::new();
    for (name, (preamble, declaration, declaration_line)) in decls {
        let (file, file_line, refused) = files.remove(&name).unwrap_or_else(|| {
            panic!("example '{name}' declares a stream but prints no `tsv` file for it")
        });
        out.push(Example {
            name,
            preamble,
            declaration,
            declaration_line,
            file,
            file_line,
            refused,
        });
    }
    assert!(
        files.is_empty(),
        "these examples print a file with no stream declaring it: {:?}",
        files.keys().collect::<Vec<_>>()
    );
    out
}

// ── What the document says a row covers ─────────────────────────────────────

/// A guarded section's `origin` and `time_unit`, read off the preamble model.
struct Calendar {
    origin: String,
    time_unit: String,
}

impl Calendar {
    fn of(preamble_src: &str) -> Calendar {
        let field = |key: &str| -> String {
            preamble_src
                .lines()
                .find_map(|l| l.trim().strip_prefix(key)?.split_once('=').map(|(_, v)| v.trim().to_string()))
                .unwrap_or_else(|| panic!("the preamble must declare `{key}`"))
        };
        let origin = field("origin")
            .trim_start_matches("date(\"")
            .trim_end_matches("\")")
            .to_string();
        let time_unit = field("time_unit").trim_start_matches('\'').to_string();
        Calendar { origin, time_unit }
    }

    /// A temporal cell as model time — an ISO date read through the origin, or
    /// a number read as it stands.
    fn time_of(&self, cell: &str) -> f64 {
        if cell.contains('-') {
            ir::caltime::date_to_internal(&self.origin, cell, &self.time_unit)
                .unwrap_or_else(|e| panic!("cannot read '{cell}' through the origin: {e:?}"))
        } else {
            cell.parse().unwrap_or_else(|e| panic!("cannot read '{cell}' as a time: {e}"))
        }
    }

    /// A duration written `<n> '<unit>`, in the model's own time unit.
    fn width_of(&self, n: f64, unit: &str) -> f64 {
        let days = |u: &str| {
            ir::caltime::days_per_unit(u).unwrap_or_else(|e| panic!("unit '{u}': {e:?}"))
        };
        n * days(unit) / days(&self.time_unit)
    }

    fn one_day(&self) -> f64 {
        self.width_of(1.0, "days")
    }
}

/// The periods the document's own declaration gives its own file — computed
/// here from the rule the spec's table states, not asked of the loader.
fn periods_the_document_states(ex: &Example, cal: &Calendar) -> Vec<(f64, f64)> {
    let (header, rows) = {
        let mut lines = ex.file.lines();
        let header: Vec<&str> = lines.next().expect("a header row").split('\t').collect();
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
        (header, rows)
    };
    let column = |name: &str| -> usize {
        header.iter().position(|h| *h == name).unwrap_or_else(|| {
            panic!(
                "{}:{}: example '{}' declares the column '{name}', which the file's \
                 header {header:?} does not have",
                spec_path().display(),
                ex.file_line,
                ex.name
            )
        })
    };

    // The scored column is the one on the `~` line; a row with `NA` there
    // states its period and scores nothing, so it contributes no term.
    let scored = ex
        .declaration
        .lines()
        .find_map(|l| Some(l.trim().split_once(" ~ ")?.0.trim().to_string()))
        .unwrap_or_else(|| panic!("example '{}' has no `<column> ~ …` line", ex.name));
    let scored_ix = column(&scored);
    let observed = |r: &Vec<&str>| r[scored_ix].trim() != "NA";

    let role = |role: &str| -> Option<String> {
        ex.declaration.split_once(&format!(": {role}"))?.0.rsplit(&[',', '{'][..]).next()
            .map(|s| s.trim().to_string())
    };

    // The per-row form: both boundaries are in the file.
    if let (Some(start), Some(stop)) = (role("window_start"), role("window_stop")) {
        let (a, b) = (column(&start), column(&stop));
        return rows
            .iter()
            .filter(|r| observed(r))
            .map(|r| (cal.time_of(r[a].trim()), cal.time_of(r[b].trim())))
            .collect();
    }

    // A uniform form: the rule in the spec's own table, applied to the label.
    let covers = ex
        .declaration
        .lines()
        .find_map(|l| Some(l.trim().strip_prefix("covers")?.split_once('=')?.1.trim().to_string()))
        .unwrap_or_else(|| {
            panic!(
                "{}:{}: example '{}' has neither window columns nor a `covers = …` \
                 declaration, so nothing states what its rows cover",
                spec_path().display(),
                ex.declaration_line,
                ex.name
            )
        });
    let (form, args) = covers.split_once('(').expect("covers = form(…)");
    let args: Vec<&str> = args.trim_end_matches(')').split(',').map(str::trim).collect();
    let label_ix = column(args[0]);
    let width = |args: &[&str]| -> f64 {
        let (n, unit) = args[1].split_once(" '").expect("a width written `<n> '<unit>`");
        cal.width_of(n.trim().parse().expect("a numeric width"), unit.trim())
    };
    let day = cal.one_day();
    let period: Box<dyn Fn(f64) -> (f64, f64)> = match form.trim() {
        "day" => Box::new(move |d| (d, d + day)),
        "starting_on" => {
            let w = width(&args);
            Box::new(move |d| (d, d + w))
        }
        "ending_on" => {
            let w = width(&args);
            Box::new(move |d| (d - w + day, d + day))
        }
        "closing_at" => {
            let w = width(&args);
            Box::new(move |d| (d - w, d))
        }
        other => panic!(
            "example '{}': the harness has no reading for `covers = {other}(…)`. \
             Add it from the spec's own table rather than skipping the example.",
            ex.name
        ),
    };
    rows.iter()
        .filter(|r| observed(r))
        .map(|r| period(cal.time_of(r[label_ix].trim())))
        .collect()
}

// ── Running one example ─────────────────────────────────────────────────────

/// The periods a run actually scored, off the prequential trace.
fn periods_the_run_scored(trace: &serde_json::Value) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for step in trace["steps"].as_array().expect("steps") {
        for ps in step["per_stream"].as_array().expect("per_stream") {
            let c = &ps["coverage"];
            assert_eq!(
                c["kind"], "interval",
                "an incidence stream's observation covers a period, not an instant"
            );
            out.push((c["start"].as_f64().unwrap(), c["stop"].as_f64().unwrap()));
        }
    }
    out
}

/// Bind one documented example through the loader. `Ok(())` when it behaved as
/// the document says; `Err(explanation)` otherwise.
fn run_example(ex: &Example) -> Result<(), String> {
    let dir = tempdir(&ex.name);
    let preamble_src = std::fs::read_to_string(fixtures().join(format!("{}.camdl", ex.preamble)))
        .map_err(|e| format!("preamble '{}': {e}", ex.preamble))?;
    let params = fixtures().join(format!("{}.params.toml", ex.preamble));
    let cal = Calendar::of(&preamble_src);

    let model = dir.join("model.camdl");
    std::fs::write(&model, format!("{preamble_src}\nobservations {{\n{}}}\n", ex.declaration))
        .unwrap();
    let ir = dir.join("model.ir.json");
    let out = Command::new(camdlc()).arg(&model).arg("-o").arg(&ir).output().unwrap();
    if !out.status.success() {
        return Err(format!(
            "the stream declared at {}:{} does not compile under preamble '{}':\n{}",
            spec_path().display(),
            ex.declaration_line,
            ex.preamble,
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let compiled: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ir).unwrap()).unwrap();
    let source = compiled["model"]["observations"][0]["source"]
        .as_str()
        .expect("a stream source")
        .to_string();

    let data = dir.join("data.tsv");
    std::fs::write(&data, &ex.file).unwrap();
    let stem = dir.join("preq");
    let out = Command::new(camdl_bin())
        .args([
            "pfilter",
            ir.to_str().unwrap(),
            "--particles",
            "50",
            "--dt",
            "0.5",
            "--seed",
            "1",
            "--params",
            params.to_str().unwrap(),
            "--data",
            &format!("{source}={}", data.display()),
            "--save-prequential",
            stem.to_str().unwrap(),
        ])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    if let Some(must_say) = &ex.refused {
        if out.status.success() {
            return Err(format!(
                "the file at {}:{} is printed as what not to write, and loaded anyway",
                spec_path().display(),
                ex.file_line
            ));
        }
        if !stderr.contains(must_say) {
            return Err(format!(
                "the file at {}:{} was refused, but the diagnostic does not say \
                 {must_say:?}:\n{stderr}",
                spec_path().display(),
                ex.file_line
            ));
        }
        let _ = std::fs::remove_dir_all(&dir);
        return Ok(());
    }

    if !out.status.success() {
        return Err(format!(
            "the file at {}:{} is printed as a documented format and does not load:\n{stderr}",
            spec_path().display(),
            ex.file_line
        ));
    }
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}.json", stem.display())).unwrap())
            .unwrap();
    let scored = periods_the_run_scored(&trace);
    let stated = periods_the_document_states(ex, &cal);
    if scored != stated {
        return Err(format!(
            "the file at {}:{} loads over periods the document does not state\n  \
             scored: {scored:?}\n  stated: {stated:?}",
            spec_path().display(),
            ex.file_line
        ));
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// ── The gate ────────────────────────────────────────────────────────────────

/// Every observation-file example the data spec prints in the guarded sections
/// loads under the stream printed beside it, over the periods the document
/// states — or, where the document prints it as what not to write, is refused
/// with the diagnostic the document names.
#[test]
fn the_data_spec_examples_load_over_the_periods_the_document_states() {
    let text = std::fs::read_to_string(spec_path()).unwrap();
    let examples = collect_examples(&text);

    // Non-vacuous three ways: both guarded sections carry an example, and the
    // expected-to-refuse marker is exercised. Without these the harness could
    // pass on a document that had quietly stopped printing examples.
    assert!(
        examples.len() >= 2,
        "the guarded sections must carry examples; found {}",
        examples.len()
    );
    for heading in GUARDED_SECTIONS {
        let start = text.find(heading).unwrap_or_else(|| {
            panic!("`{heading}` is gone from the data spec; update GUARDED_SECTIONS")
        });
        let body_start = text[..start].lines().count() + 1;
        let body_end = text[start + heading.len()..]
            .find("\n## ")
            .map(|o| text[..start + heading.len() + o].lines().count() + 1)
            .unwrap_or(usize::MAX);
        assert!(
            examples.iter().any(|e| (body_start..body_end).contains(&e.file_line)),
            "`{heading}` prints no executed example"
        );
    }
    assert!(
        examples.iter().any(|e| e.refused.is_some()),
        "no example is marked `refused=…`, so the what-not-to-write path is unexercised"
    );

    let mut failures: Vec<String> = Vec::new();
    for ex in &examples {
        if let Err(why) = run_example(ex) {
            failures.push(format!("example '{}': {why}", ex.name));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} documented examples did not behave as documented:\n\n{}",
        failures.len(),
        examples.len(),
        failures.join("\n\n")
    );
}
