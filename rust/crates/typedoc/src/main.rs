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

mod markdown;
mod marks;
mod model;
mod odoc;
mod overview;
mod render;
mod site;

use std::collections::BTreeSet;
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
    let files = render::render(&site, &args.source);

    let out = args.out;
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let written: BTreeSet<&str> = files.iter().map(|f| f.file_name.as_str()).collect();
    for f in &files {
        let path = out.join(&f.file_name);
        let unchanged = fs::read(&path).is_ok_and(|old| old == f.text.as_bytes());
        if !unchanged {
            fs::write(&path, &f.text).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    let entries = fs::read_dir(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| format!("{}: {e}", out.display()))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.extension().is_some_and(|x| x == "qmd") && path.is_file() && !written.contains(name)
        {
            fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
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
        } => {
            let result = render::SourceLinks::new(&repo, &commit).and_then(|source| {
                run_render(&RenderArgs {
                    overview: &overview,
                    inputs: &inputs,
                    out: &out,
                    order,
                    source,
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
