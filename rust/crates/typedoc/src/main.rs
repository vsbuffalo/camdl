//! `typedoc`: renders the internals pages of the camdl documentation site.
//!
//! A language extractor reads source files and writes their type
//! declarations as JSON ([`model::Input`]). This tool reads one or more such
//! files and an overview page ([`overview::Overview`], an odoc `.mld` file)
//! that gives the index its title and prose and groups the modules, and
//! writes Quarto pages: the index, and one page per top-level module showing
//! each type's declaration, documentation, reviewer marks, cross-references
//! and a link to its source at one commit. The output is the same, byte for
//! byte, for the same inputs and commit.

mod code;
mod markdown;
mod marks;
mod model;
mod odoc;
mod overview;
mod render;
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
        /// The overview page (odoc `.mld`): the index's title and prose, and
        /// one `{1 ...}` section with a `{!modules: ...}` list per group.
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

fn run_render(args: &RenderArgs) -> Result<usize, String> {
    let overview = overview::Overview::parse(&read(args.overview)?)
        .map_err(|e| format!("{}: {e}", args.overview.display()))?;
    let inputs: Vec<model::Input> = args
        .inputs
        .iter()
        .map(|p| serde_json::from_str(&read(p)?).map_err(|e| format!("{}: {e}", p.display())))
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
    let written: BTreeSet<PathBuf> = files.iter().map(|f| out.join(&f.file_name)).collect();
    for f in &files {
        let path = out.join(&f.file_name);
        let unchanged = fs::read(&path).is_ok_and(|old| old == f.text.as_bytes());
        if !unchanged {
            fs::write(&path, &f.text).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    for dir in [out, src_dir.as_path()] {
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
    Ok(files.len())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
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
