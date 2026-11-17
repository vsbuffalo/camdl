//! Execute the `simulate` output tables printed in `docs/dates.md` and compare
//! their time and date columns with what the command prints today (gh#1089).
//!
//! `docs/dates.md` states where a generated grid falls on the calendar and
//! shows it with printed output. `make test-docs` compiles the page's
//! `camdl` blocks and `make test-cli-docs` parses its `camdl …` command lines;
//! neither runs a model, so a printed table could say one thing while the
//! compiler did another. gh#1072 is what that cost: the page said a generated
//! grid counts from `origin` while the compiler counts it from
//! `simulate { from }`, and the page's only example had `from = origin`.
//!
//! ## How a table is marked
//!
//! The fence's info string names the model and the command:
//!
//! ```text
//! ```tsv dates-example=weekly-wednesday model=weekly_from_origin cmd="simulate --obs --dates"
//! ```
//!
//! `model` names a stem under `tests/fixtures/dates_doc/`: a complete
//! `<stem>.camdl` and the `<stem>.params.toml` it is run with. `cmd` is the
//! `camdl` subcommand and its flags, minus the model and the parameters. The
//! harness supplies the plumbing a reader does not need to see: `--obs` is run
//! as `--obs-only <file>` and that file is read; without `--obs` the
//! trajectory is read from `--stdout`.
//!
//! What is compared: every column the table's header names must exist in the
//! output (a renamed column is drift), and the time and date columns among
//! them (`t`, `time`, `t_start`, `t_stop`, `date`, `date_start`, `date_stop`)
//! must match cell for cell. Count and state columns are seed- and
//! backend-dependent and are not compared. By default the table is the whole
//! output, row for row; a fence carrying `excerpt` is a contiguous run of
//! output rows instead, located by its first row.
//!
//! The markers live in the fence info string, never in an HTML comment before
//! the fence, so `dprint fmt` is safe on the page (CLAUDE.md, "Markdown"; the
//! same arrangement as `data_spec_examples.rs`).
//!
//! ## Fail-closed
//!
//! A fenced block anywhere on the page whose first line reads like a
//! `simulate` header — a first column `t` or `time` and a `date` column — must
//! carry a `dates-example` marker. An unmarked output table is a gate failure,
//! so the check cannot quietly stop covering a table added later.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Columns whose values are a function of the model's calendar and schedule
/// alone, never of the seed or the backend's noise.
const TEMPORAL: &[&str] = &["t", "time", "t_start", "t_stop", "date", "date_start", "date_stop"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap()
}

/// The page under test. `CAMDL_DATES_DOC` points the harness at another copy,
/// which is how the red step (a deliberately wrong date column) is shown
/// without editing the committed page.
fn doc_path() -> PathBuf {
    match std::env::var_os("CAMDL_DATES_DOC") {
        Some(p) => PathBuf::from(p),
        None => repo_root().join("docs/dates.md"),
    }
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dates_doc")
}

fn camdl_bin() -> PathBuf {
    let p = repo_root().join("rust/target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust`, or gate with `make test-dates-doc`",
        p.display()
    );
    p
}

fn camdlc() -> PathBuf {
    let p = repo_root().join("ocaml/_build/default/bin/camdlc.exe");
    assert!(
        p.exists(),
        "camdlc missing: {} - run `make build-ocaml`, or gate with `make test-dates-doc`",
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
        .join(format!("camdl_dates_doc_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

// ── Reading the document ────────────────────────────────────────────────────

/// One fenced block anywhere on the page.
struct Fence {
    info: BTreeMap<String, String>,
    body: String,
    /// 1-based line of the opening fence, so a failure points at the page.
    line: usize,
}

/// Split a fence info string into its `key=value` directives (the language
/// token is dropped). A value may be double-quoted, so it can carry spaces; a
/// bare token becomes a key with an empty value.
fn parse_info(info: &str) -> BTreeMap<String, String> {
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
    tokens
        .iter()
        .skip(1)
        .map(|t| match t.split_once('=') {
            Some((k, v)) => (k.to_string(), v.to_string()),
            None => (t.to_string(), String::new()),
        })
        .collect()
}

fn fences(text: &str) -> Vec<Fence> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("```") {
            let info = parse_info(lines[i].trim_start_matches('`'));
            let open = i;
            let mut body = String::new();
            i += 1;
            while i < lines.len() && !lines[i].starts_with("```") {
                body.push_str(lines[i]);
                body.push('\n');
                i += 1;
            }
            out.push(Fence { info, body, line: open + 1 });
        }
        i += 1;
    }
    out
}

/// A printed table: a header and rows, cells split on runs of whitespace (the
/// page aligns its columns with spaces; the CLI separates them with tabs).
struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn table_of(text: &str) -> Table {
    let mut lines = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let header = lines
        .next()
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    let rows = lines
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect();
    Table { header, rows }
}

/// Does this block read like printed `simulate` output?
fn looks_like_output(body: &str) -> bool {
    let t = table_of(body);
    matches!(t.header.first().map(String::as_str), Some("t") | Some("time"))
        && t.header.iter().any(|h| h == "date")
}

struct Example {
    name: String,
    model: String,
    cmd: Vec<String>,
    excerpt: bool,
    table: Table,
    line: usize,
}

fn collect_examples(text: &str) -> Vec<Example> {
    let mut out: Vec<Example> = Vec::new();
    for f in fences(text) {
        let Some(name) = f.info.get("dates-example") else {
            assert!(
                !looks_like_output(&f.body),
                "{}:{}: this block prints `simulate` output but carries no \
                 `dates-example=<name> model=<stem> cmd=\"…\"` marker. Every output \
                 table on this page is executed; mark it.",
                doc_path().display(),
                f.line
            );
            continue;
        };
        assert!(!name.is_empty(), "{}:{}: `dates-example` needs a name", doc_path().display(), f.line);
        assert!(
            out.iter().all(|e| &e.name != name),
            "{}:{}: example '{name}' is printed twice",
            doc_path().display(),
            f.line
        );
        let need = |key: &str| -> String {
            f.info.get(key).cloned().unwrap_or_else(|| {
                panic!("{}:{}: example '{name}' names no `{key}=`", doc_path().display(), f.line)
            })
        };
        out.push(Example {
            name: name.clone(),
            model: need("model"),
            cmd: need("cmd").split_whitespace().map(str::to_string).collect(),
            excerpt: f.info.contains_key("excerpt"),
            table: table_of(&f.body),
            line: f.line,
        });
    }
    out
}

// ── Running one example ─────────────────────────────────────────────────────

/// Run the example's command and return what it printed.
fn run(ex: &Example, dir: &Path) -> Result<Table, String> {
    let model = fixtures().join(format!("{}.camdl", ex.model));
    let params = fixtures().join(format!("{}.params.toml", ex.model));
    let (sub, flags) = ex
        .cmd
        .split_first()
        .ok_or_else(|| "`cmd` is empty".to_string())?;

    let mut args: Vec<String> = vec![
        sub.clone(),
        model.display().to_string(),
        "--params".into(),
        params.display().to_string(),
        "--output-dir".into(),
        dir.join("results").display().to_string(),
    ];
    let obs_file = dir.join("obs.tsv");
    let mut reads_obs = false;
    for flag in flags {
        if flag == "--obs" {
            reads_obs = true;
            args.push("--obs-only".into());
            args.push(obs_file.display().to_string());
        } else {
            args.push(flag.clone());
        }
    }
    if !reads_obs {
        args.push("--stdout".into());
    }

    let out = Command::new(camdl_bin())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .current_dir(dir)
        .args(&args)
        .output()
        .expect("camdl must invoke");
    if !out.status.success() {
        return Err(format!(
            "`camdl {}` failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let printed = if reads_obs {
        std::fs::read_to_string(&obs_file).map_err(|e| format!("reading the obs file: {e}"))?
    } else {
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    Ok(table_of(&printed))
}

/// Compare the printed table with the run. `Ok(())` when every temporal cell
/// the page prints is the one the command prints.
fn check(ex: &Example, got: &Table) -> Result<(), String> {
    let doc = &ex.table;
    let mut compared: Vec<(usize, usize, &str)> = Vec::new();
    for (di, h) in doc.header.iter().enumerate() {
        let gi = got.header.iter().position(|g| g == h).ok_or_else(|| {
            format!("the page prints a column '{h}', which the command does not: it prints {:?}", got.header)
        })?;
        if TEMPORAL.contains(&h.as_str()) {
            compared.push((di, gi, h.as_str()));
        }
    }
    if compared.is_empty() {
        return Err(format!("the table prints no time or date column ({TEMPORAL:?}), so nothing is checked"));
    }
    if doc.rows.is_empty() {
        return Err("the table prints no rows".into());
    }
    for (k, r) in doc.rows.iter().enumerate() {
        if r.len() != doc.header.len() {
            return Err(format!("table row {} has {} cells for {} columns", k + 1, r.len(), doc.header.len()));
        }
    }

    let key = |row: &Vec<String>, side_doc: bool| -> Vec<String> {
        compared
            .iter()
            .map(|(di, gi, _)| row[if side_doc { *di } else { *gi }].clone())
            .collect()
    };
    let offset = if ex.excerpt {
        let first = key(&doc.rows[0], true);
        got.rows.iter().position(|r| key(r, false) == first).ok_or_else(|| {
            format!(
                "the excerpt's first row {first:?} is not in the output; the command prints \
                 these rows (columns {:?}):\n{}",
                compared.iter().map(|c| c.2).collect::<Vec<_>>(),
                got.rows.iter().map(|r| key(r, false).join("  ")).collect::<Vec<_>>().join("\n")
            )
        })?
    } else {
        if doc.rows.len() != got.rows.len() {
            return Err(format!(
                "the table prints {} rows and the command prints {} (mark the fence \
                 `excerpt` if the page shows only some of them):\n{}",
                doc.rows.len(),
                got.rows.len(),
                got.rows.iter().map(|r| key(r, false).join("  ")).collect::<Vec<_>>().join("\n")
            ));
        }
        0
    };
    for (k, drow) in doc.rows.iter().enumerate() {
        let grow = got.rows.get(offset + k).ok_or_else(|| {
            format!("table row {} runs past the end of the output", k + 1)
        })?;
        for (di, gi, h) in &compared {
            if drow[*di] != grow[*gi] {
                return Err(format!(
                    "table row {} prints {h} = {}, the command prints {h} = {}",
                    k + 1,
                    drow[*di],
                    grow[*gi]
                ));
            }
        }
    }
    Ok(())
}

// ── The gate ────────────────────────────────────────────────────────────────

/// Every `simulate` output table `docs/dates.md` prints has the time and date
/// columns its command prints today.
#[test]
fn the_dates_page_output_tables_match_their_commands() {
    let text = std::fs::read_to_string(doc_path()).unwrap();
    let examples = collect_examples(&text);

    // Non-vacuous: the page prints three output tables (the trajectory excerpt
    // and the two weekly grids). Fewer means a marker was lost.
    assert!(
        examples.len() >= 3,
        "expected at least three executed tables on the page, found {}",
        examples.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for ex in &examples {
        let dir = tempdir(&ex.name);
        let result = run(ex, &dir).and_then(|got| check(ex, &got));
        match result {
            Ok(()) => {
                let _ = std::fs::remove_dir_all(&dir);
            }
            Err(why) => failures.push(format!(
                "{}:{}: example '{}': {why}",
                doc_path().display(),
                ex.line,
                ex.name
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} printed tables do not match their commands:\n\n{}",
        failures.len(),
        examples.len(),
        failures.join("\n\n")
    );
}
