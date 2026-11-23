//! `camdl diff A B`: what differs between two models, declaration by
//! declaration (docs/dev/proposals/2026-11-13-model-structure-diff.md).
//!
//! This file only gathers the inputs and prints. Each side is compiled (a
//! `.camdl`) or read (a compiled `.ir.json`); its declared section and the
//! hash of its compiled model go to `diff::diff`, which decides every row and
//! the outcome. The hash is `ir::model_hash::model_ir_hash`, the one model
//! identity.
//!
//! Exit status: 0 when the two are identical in every layer present, 1 when
//! they differ, 2 when no report could be made (an input that does not load,
//! a side with no declared model, or the same declarations over different
//! compiled models).

use std::path::Path;

use crate::args::{DiffArgs, DiffFormat};
use crate::out::{CommandError, ExitCode, Out};

/// The two models differ. Not an error: the report was printed.
const DIFFERENT: ExitCode = ExitCode(1);
/// No report could be made.
const REFUSED: ExitCode = ExitCode(2);

/// One side, loaded: what `diff::Side` borrows from.
struct Loaded {
    name: String,
    declared: declared::DeclaredSection,
    model_hash: String,
    compiler: Option<String>,
}

impl Loaded {
    fn side(&self) -> diff::Side<'_> {
        diff::Side {
            name: &self.name,
            declared: &self.declared,
            model_hash: &self.model_hash,
            compiler: self.compiler.as_deref(),
        }
    }
}

fn refuse(out: &Out, message: String, hint: Option<String>) -> ! {
    out.error(CommandError { code: REFUSED, message, hint }).exit()
}

fn load(out: &Out, path: &Path, compile: crate::util::CompileSettings) -> Loaded {
    let name = path.to_string_lossy().into_owned();
    let is_model = name.ends_with(".camdl") || name.ends_with(".json");
    if !is_model {
        refuse(
            out,
            format!("error: camdl diff compares two model files; {name} is not a .camdl or a compiled .ir.json"),
            Some("To compare two fit configurations, use `camdl fit diff A.toml B.toml`.".to_string()),
        );
    }
    let envelope = crate::util::load_model_envelope(&name, compile)
        .unwrap_or_else(|e| refuse(out, format!("error: {e}"), None));
    Loaded {
        name,
        model_hash: ir::model_hash::model_ir_hash(&envelope.model).to_hex(),
        compiler: envelope.validated_by,
        declared: envelope.declared,
    }
}

/// `camdl diff A B`
pub(crate) fn cmd_diff(args: &DiffArgs, compile: crate::util::CompileSettings) {
    // No `--no-color`: the report is plain text, and errors follow the
    // environment and whether stderr is a terminal.
    let out = Out::new(false);
    let left = load(&out, &args.a, compile);
    let right = load(&out, &args.b, compile);

    let report = diff::diff(&left.side(), &right.side())
        .unwrap_or_else(|refusal| refuse(&out, format!("error: {refusal}"), None));

    match args.format {
        DiffFormat::Table => out.report_raw(diff::render::table(&report)),
        DiffFormat::Json => {
            let json = serde_json::to_string_pretty(&report)
                .unwrap_or_else(|e| refuse(&out, format!("error: cannot write the report as JSON: {e}"), None));
            out.report_line(json);
        }
    }
    if !report.identical() {
        DIFFERENT.exit();
    }
}
