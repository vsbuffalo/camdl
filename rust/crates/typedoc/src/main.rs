//! `typedoc`: renders the internals pages of the camdl documentation site.
//!
//! A language extractor reads source files and writes their type
//! declarations as JSON ([`model::Input`]). This tool reads one or more such
//! files and a manifest ([`manifest::Manifest`]) that groups the modules, and
//! writes Quarto pages: an index of the modules, and one page per top-level
//! module showing each type's declaration, documentation, reviewer marks and
//! cross-references. The output is the same, byte for byte, for the same
//! inputs.

mod manifest;
mod markdown;
mod marks;
mod model;
mod odoc;
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
        /// The manifest naming the pages' title and the module groups.
        #[arg(long)]
        manifest: PathBuf,
        /// An extractor's JSON output. Repeat for several languages.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,
        /// The directory to write the pages into. Created if absent.
        #[arg(long)]
        out: PathBuf,
    },
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn run_render(manifest: &Path, inputs: &[PathBuf], out: &Path) -> Result<usize, String> {
    let manifest = manifest::Manifest::from_toml(&read(manifest)?)
        .map_err(|e| format!("{}: {e}", manifest.display()))?;
    let inputs: Vec<model::Input> = inputs
        .iter()
        .map(|p| serde_json::from_str(&read(p)?).map_err(|e| format!("{}: {e}", p.display())))
        .collect::<Result<_, String>>()?;
    let site = site::Site::build(&manifest, &inputs)?;
    let files = render::render(&site);

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
            manifest,
            inputs,
            out,
        } => match run_render(&manifest, &inputs, &out) {
            Ok(n) => {
                eprintln!("typedoc: wrote {n} pages to {}", out.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("typedoc: {e}");
                ExitCode::FAILURE
            }
        },
    }
}
