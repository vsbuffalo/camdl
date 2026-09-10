//! `camdl if2` — removed (gh#147, M3.3).
//!
//! The standalone `if2` subcommand had its own non-content-addressed
//! fit runner. Every fit, however invoked, is now a content-addressed
//! fit of one `[method]`, so an IF2 fit is a fit whose method is
//! `algorithm = "if2"`, run through `camdl fit run`. There is no separate
//! "one-off" shape to maintain.
//!
//! This file is a deprecation stub: it accepts (and ignores) any
//! arguments and prints an actionable migration message. Per CLAUDE.md
//! alpha posture it is a redirect, not a back-compat shim — the old
//! behaviour is gone.

/// Print the migration message and exit non-zero. `camdl if2` no longer
/// runs anything.
pub fn cmd_if2(_a: &crate::args::If2Args) {
    eprintln!("{}", DEPRECATION);
    std::process::exit(2);
}

const DEPRECATION: &str = r#"error: `camdl if2` has been removed.

Run IF2 as a fit through the content-addressed fit runner.
Write a fit.toml:

  [model]
  camdl = "model.camdl"

  [data.observations]
  cases = "cases.tsv"

  [estimate]
  beta = { bounds = [0.1, 2.0], start = 0.6 }

  [method]
  algorithm  = "if2"
  backend    = "chain_binomial"
  chains     = 4
  particles  = 300
  iterations = 100
  cooling    = 0.7

then run it:

  camdl fit run fit.toml --seed 1

The fit lands in the content-addressed store
(fits/<stem>-<h8>/if2-<h8>/); browse it with `camdl list --kind fit`,
`camdl show <hash>`, `camdl cat <hash>`."#;
