//! `typedoc`: renders the internals pages of the camdl documentation site.
//!
//! A language extractor reads source files and writes their type
//! declarations as JSON ([`model::Input`]): `typedoc_ocaml` for the OCaml
//! compiler, and this tool's `extract-rust` subcommand ([`rustdoc`]) for the
//! Rust crates, from rustdoc's JSON output. The `render` subcommand reads one
//! or more such files and an overview page ([`overview::Overview`], an odoc
//! `.mld` file or a Markdown `.md` file) that gives the index its title and
//! prose and groups the modules, and
//! writes Quarto pages: the index, and one page per top-level module showing
//! each type's declaration, documentation, reviewer marks, cross-references
//! and a link to its source at one commit. The `engineering` subcommand
//! renders hand-written pages about the code ([`engineering`]), resolving
//! their references against the same inputs. The output is the same, byte
//! for byte, for the same inputs and commit.
#![expect(
    clippy::disallowed_methods,
    reason = "a documentation generator outside the engine: it reads source files, \
              rustdoc JSON and the pages it wrote before, none of which is an input of a \
              stored result"
)]

mod code;
mod engineering;
mod markdown;
mod marks;
mod model;
mod odoc;
mod overview;
mod render;
mod rustdoc;
mod rustlex;
mod site;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "typedoc",
    about = "Render internals pages from extracted type declarations"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the index page and one page per module into a directory.
    ///
    /// The directory is the generator's own: any `.qmd` file in it that this
    /// run does not write is deleted.
    Render {
        /// The overview page: the index's title and prose, and one section
        /// per group listing its modules. An odoc `.mld` file has one
        /// `{1 ...}` section with a `{!modules: ...}` list per group; a
        /// Markdown `.md` file has one `## ...` section with a
        /// `<!-- modules: ... -->` line per group.
        #[arg(long)]
        overview: PathBuf,
        /// An extractor's JSON output. Repeat for several languages.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,
        /// The directory to write the pages into. Created if absent.
        #[arg(long)]
        out: PathBuf,
        /// The sidebar position of the index page. The module pages follow
        /// it, at order + 1, order + 2, ... in overview order.
        #[arg(long, default_value_t = 100)]
        order: i64,
        /// The repository the source links point into.
        #[arg(long, default_value = "https://github.com/vsbuffalo/camdl")]
        repo: String,
        /// The commit the source links point at, so their line numbers stay
        /// right after the files change.
        #[arg(long)]
        commit: String,
        /// The directory the inputs' source paths are relative to: the
        /// repository root. Each source file is read from it and shown whole
        /// on a page of its own.
        #[arg(long, default_value = ".")]
        source_root: PathBuf,
    },
    /// Write the engineering pages: each Markdown file of a directory, its
    /// references to the code resolved against the generated pages.
    ///
    /// The output directory is the generator's own: any `.qmd` file in it
    /// that this run does not write is deleted.
    Engineering {
        /// The directory of the hand-written pages, `<topic>.md` each. A
        /// missing or empty directory writes no pages.
        #[arg(long)]
        pages: PathBuf,
        /// The directory to write the pages into. Created if absent.
        #[arg(long)]
        out: PathBuf,
        /// A generated site the pages refer to, as
        /// `dir=overview,input.json`: the directory its pages are written to,
        /// and the overview and extractor output its `render` was given.
        /// Repeat for each language.
        #[arg(long = "site", required = true, value_parser = SiteSpec::parse)]
        sites: Vec<SiteSpec>,
        /// The repository the source links point into.
        #[arg(long, default_value = "https://github.com/vsbuffalo/camdl")]
        repo: String,
        /// The commit the source links point at.
        #[arg(long)]
        commit: String,
    },
    /// Write the renderer's JSON input for Rust crates, from the JSON
    /// rustdoc writes for each (`cargo rustdoc -- -Z unstable-options
    /// --output-format json --document-private-items`).
    ExtractRust {
        /// A crate, as `label=path/to/crate.json`: the label prefixes its
        /// module paths on the pages. Repeat, in page order.
        #[arg(long = "crate", required = true, value_parser = rustdoc::CrateSpec::parse)]
        crates: Vec<rustdoc::CrateSpec>,
        /// The cargo workspace's directory, relative to the repository root;
        /// rustdoc writes source paths relative to it.
        #[arg(long, default_value = "rust")]
        workspace: String,
        /// The repository root, which source files are read from.
        #[arg(long, default_value = ".")]
        repo_root: PathBuf,
        /// The file to write. Standard output when absent.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// A generated site, as `--site dir=overview,input.json` gives it.
#[derive(Debug, Clone)]
struct SiteSpec {
    dir: PathBuf,
    overview: PathBuf,
    input: PathBuf,
}

impl SiteSpec {
    fn parse(s: &str) -> Result<SiteSpec, String> {
        let parsed = s.split_once('=').and_then(|(dir, rest)| {
            let (overview, input) = rest.split_once(',')?;
            let ok = [dir, overview, input].iter().all(|x| !x.is_empty());
            ok.then(|| SiteSpec {
                dir: dir.into(),
                overview: overview.into(),
                input: input.into(),
            })
        });
        parsed.ok_or_else(|| format!("--site {s:?} is not dir=overview,input.json"))
    }
}

fn run_extract_rust(
    crates: &[rustdoc::CrateSpec],
    workspace: &str,
    repo_root: &Path,
    out: Option<&Path>,
) -> Result<(), String> {
    let input = rustdoc::extract(crates, workspace, repo_root)?;
    let json = serde_json::to_string(&input).map_err(|e| e.to_string())?;
    match out {
        Some(p) => fs::write(p, json + "\n").map_err(|e| format!("{}: {e}", p.display())),
        None => {
            println!("{json}");
            Ok(())
        }
    }
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

struct RenderArgs<'a> {
    overview: &'a Path,
    inputs: &'a [PathBuf],
    out: &'a Path,
    order: i64,
    source: render::SourceLinks,
    source_root: &'a Path,
}

/// An overview page: odoc markup, or Markdown for a `.md` file.
fn load_overview(path: &Path) -> Result<overview::Overview, String> {
    let text = read(path)?;
    if path.extension().is_some_and(|x| x == "md") {
        overview::Overview::parse_markdown(&text)
    } else {
        overview::Overview::parse(&text)
    }
    .map_err(|e| format!("{}: {e}", path.display()))
}

fn load_input(path: &Path) -> Result<model::Input, String> {
    serde_json::from_str(&read(path)?).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes `files` (names relative to `out`) into `out`, leaving a file
/// whose bytes are unchanged untouched, and deletes every other `.qmd` file
/// in each of `dirs`.
fn write_pages(out: &Path, files: &[render::PageFile], dirs: &[&Path]) -> Result<(), String> {
    let written: BTreeSet<PathBuf> = files.iter().map(|f| out.join(&f.file_name)).collect();
    for f in files {
        let path = out.join(&f.file_name);
        let unchanged = fs::read(&path).is_ok_and(|old| old == f.text.as_bytes());
        if !unchanged {
            fs::write(&path, &f.text).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    for dir in dirs {
        let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
            if path.extension().is_some_and(|x| x == "qmd")
                && path.is_file()
                && !written.contains(&path)
            {
                fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
    }
    Ok(())
}

fn run_render(args: &RenderArgs) -> Result<usize, String> {
    let overview = load_overview(args.overview)?;
    let inputs: Vec<model::Input> = args
        .inputs
        .iter()
        .map(|p| load_input(p))
        .collect::<Result<_, String>>()?;
    let site = site::Site::build(&overview, args.order, &inputs)
        .map_err(|e| format!("{}: {e}", args.overview.display()))?;
    let mut texts = BTreeMap::new();
    for s in &site.sources {
        let path = &s.file.path;
        texts.insert(path.clone(), read(&args.source_root.join(path))?);
    }
    let files = render::render(&site, &args.source, &texts)?;

    let out = args.out;
    let src_dir = out.join(site::SOURCE_DIR);
    fs::create_dir_all(&src_dir).map_err(|e| format!("{}: {e}", src_dir.display()))?;
    write_pages(out, &files, &[out, src_dir.as_path()])?;
    Ok(files.len())
}

/// The engineering pages of `pages`: every `.md` file in it, in file-name
/// order, which is their sidebar order.
fn page_sources(pages: &Path) -> Result<Vec<PathBuf>, String> {
    if !pages.exists() {
        return Ok(Vec::new());
    }
    let entries = fs::read_dir(pages).map_err(|e| format!("{}: {e}", pages.display()))?;
    let mut out = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("{}: {e}", pages.display()))?
            .path();
        if path.extension().is_some_and(|x| x == "md") && path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn run_engineering(
    pages: &Path,
    out: &Path,
    specs: &[SiteSpec],
    source: &render::SourceLinks,
) -> Result<usize, String> {
    let overviews: Vec<overview::Overview> = specs
        .iter()
        .map(|s| load_overview(&s.overview))
        .collect::<Result<_, _>>()?;
    let inputs: Vec<Vec<model::Input>> = specs
        .iter()
        .map(|s| load_input(&s.input).map(|i| vec![i]))
        .collect::<Result<_, _>>()?;
    let sites: Vec<site::Site> = specs
        .iter()
        .zip(&overviews)
        .zip(&inputs)
        .map(|((s, o), i)| {
            site::Site::build(o, 0, i).map_err(|e| format!("{}: {e}", s.overview.display()))
        })
        .collect::<Result<_, _>>()?;
    let out_dir = engineering::absolute(out)?;
    let linked: Vec<engineering::Linked> = specs
        .iter()
        .zip(&sites)
        .map(|(s, site)| {
            let dir = engineering::relative(&out_dir, &engineering::absolute(&s.dir)?);
            Ok(engineering::Linked { site, dir })
        })
        .collect::<Result<_, String>>()?;
    let relocate = engineering::Relocate {
        source_dir: engineering::absolute(pages)?,
        out_dir,
    };
    let cx = engineering::Context {
        sites: &linked,
        source,
        relocate: &relocate,
    };
    let mut rendered = Vec::new();
    for (k, path) in page_sources(pages)?.iter().enumerate() {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let name = path.display().to_string();
        // Quarto renders no file whose name starts with `_` or `.`.
        if stem.starts_with(['_', '.']) {
            return Err(format!(
                "{name}: Quarto skips a page whose name starts with `_` or `.`; rename it"
            ));
        }
        if stem == site::INDEX_SLUG {
            return Err(format!(
                "{name}: {}.qmd is the index of the engineering pages; rename the page",
                site::INDEX_SLUG
            ));
        }
        let page = engineering::render_page(&name, &read(path)?, k as i64 + 1, &cx)?;
        rendered.push((stem, page));
    }
    let mut files = vec![render::PageFile {
        file_name: format!("{}.qmd", site::INDEX_SLUG),
        text: engineering::index_page(&rendered),
    }];
    files.extend(rendered.into_iter().map(|(stem, page)| render::PageFile {
        file_name: format!("{stem}.qmd"),
        text: page.text,
    }));
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    write_pages(out, &files, &[out])?;
    Ok(files.len() - 1)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Engineering {
            pages,
            out,
            sites,
            repo,
            commit,
        } => {
            let result = render::SourceLinks::new(&repo, &commit)
                .and_then(|source| run_engineering(&pages, &out, &sites, &source));
            match result {
                Ok(n) => {
                    eprintln!("typedoc: wrote {n} engineering pages to {}", out.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("typedoc: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Command::ExtractRust {
            crates,
            workspace,
            repo_root,
            out,
        } => match run_extract_rust(&crates, &workspace, &repo_root, out.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("typedoc: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Render {
            overview,
            inputs,
            out,
            order,
            repo,
            commit,
            source_root,
        } => {
            let result = render::SourceLinks::new(&repo, &commit).and_then(|source| {
                run_render(&RenderArgs {
                    overview: &overview,
                    inputs: &inputs,
                    out: &out,
                    order,
                    source,
                    source_root: &source_root,
                })
            });
            match result {
                Ok(n) => {
                    eprintln!("typedoc: wrote {n} pages to {}", out.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("typedoc: {e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
