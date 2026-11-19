//! `ResolvedFitConfig`: the one resolved configuration of a `camdl fit run`,
//! from which both the fit's identity and what each method runs are built
//! (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`, phase 3
//! "The shape"; step 4 for `fit run`).
//!
//! ```text
//! FitRunArgs + fit.toml + defaults ──resolve──► ResolvedFitConfig
//!                                                 │            │
//!                          FitMethodCtx (per leaf)│            │ FitRunConfig::build, the runners
//!                                                 ▼            ▼
//!                               cas::resolve_fit_method     the engine
//! ```
//!
//! [`resolve`] applies every flag that changes what the fit computes to the
//! configuration — `--allow-forcing-extrapolation` into `[config]`,
//! `--starts`, `--starts-chains` and the sampler flags into `[method]` — then
//! compiles the model under the command's compile settings
//! ([`crate::util::CompileSettings`], `--no-licm` passed to the compile rather
//! than read from a process-wide setting), resolves the observation anchors,
//! the step (the dt rule), `[fixed] from_scenario` and the chain-starts rule,
//! and validates the whole, all before anything runs. `cmd_fit_run_v2`
//! destructures the value with every field named, so a field added here does
//! not compile there until it is used.
//!
//! **What reaches the identity, and how.** The fit's identity is still the
//! canonical-JSON hash of its configuration (`fit::cas`; the typed identity is
//! the follow-up proposal): the problem half of [`ResolvedFitConfig::config`]
//! at the `fit` level, its method at the `method` level, each seed at the
//! `seed` level. [`ResolvedFitConfig::model`] and [`ResolvedFitConfig::engine`]
//! (this build and the engine switches set, `env::Environment::engine_identity`)
//! are the fit level's model and engine. The compile settings reach it through
//! the IR they produce. `--sweep` and `--seed`/`fit_seeds` expand the run into
//! leaves; `--emit-every` keys a synthetic fit through the data it generates;
//! `--resume` through the method level's target length and its dep on the base.
//!
//! **What the fit folder stores.** The file is read once, in [`resolve`]
//! ([`ResolvedFitConfig::config_text`]), and every fit folder the run writes
//! keeps two configs. `fit.toml.original` is that text byte for byte, comments
//! included: what the user wrote, for display and provenance; the folder's
//! `fit_toml_hash` is its hash. `fit.resolved.toml` is the configuration the
//! folder's fits ran under ([`resolved_config_text`]): the text with every
//! problem-half value the command line set written in ([`ProblemFlags`]) and
//! the folder's own `[fixed]` (a sweep point's values applied), deterministic
//! and marked as generated. An edit to the file while the fit runs reaches
//! neither.
//!
//! Who reads the resolved config: a run-directory, label or run-id handle
//! (`fit predict <leaf>`, `compare <leaf>`) recovers the problem from it, and
//! so runs under the override a flag set (gh#1268) and at a sweep point's own
//! values; `fit table`'s config diff compares it. A `fit.toml` handle names
//! only the run the file alone makes (gh#896), so it refuses a flag-made fit,
//! naming its leaf. `pfilter`, `profile` and `survey` take the same handle
//! forms for `--fit` (`fit::handle::resolve_fit_problem`): a run handle reads
//! the resolved config, and a `fit.toml` names the file's problem, which does
//! not hold a flag's value. The resolved config's `[method]` table is the file's:
//! a fit folder holds every method run on its problem, so a method's own
//! resolved settings are its leaf's (`run.json`'s `inputs`).
//!
//! **What is not here.** `--parallel` and `--force` are execution: how the
//! leaves are computed, never what. `--label` is provenance, and the progress
//! flags presentation; neither is a field of this configuration, so neither
//! can reach the engine through it.

use super::chain_starts;
use super::config_v2::{self, FitConfig};

/// The resolved configuration of one `camdl fit run`.
pub(crate) struct ResolvedFitConfig {
    /// The fit file's text, read once: what `config` was parsed from, what each
    /// fit folder keeps as `fit.toml.original` and builds its
    /// `fit.resolved.toml` from ([`resolved_config_text`]), and what its config
    /// identity and `fit_toml_hash` are taken from. Nothing after resolution
    /// reads the file again, so an edit to it while the fit runs reaches no
    /// leaf.
    pub config_text: String,
    /// The configuration the fit computes from, keys and stores: the file with
    /// every flag applied, the step resolved, `[fixed] from_scenario`
    /// expanded, the starts rule resolved and `problem.compiled_ir` the
    /// compiled model's file.
    pub config: FitConfig,
    /// The model compiled under the command's compile settings, observation
    /// anchors resolved and external inputs bound: what every method loads
    /// (from `config.problem.compiled_ir`) and what the fit level hashes.
    pub model: ir::Model,
    /// The files `[data.tables]` bound into `model`, read once: their digests
    /// enter the fit level (`FitDigest.tables`) and their bytes are stored in
    /// each fit folder. Empty for a model with no external input.
    pub tables: crate::external_inputs::BoundFiles,
    /// The engine the identity folds: this build and the engine switches set
    /// ([`super::fit_engine`]).
    pub engine: runid::inputs::EngineIdentity,
    /// `[config] allow_forcing_extrapolation`, as the policy every model the
    /// fit evaluates runs under and every record it writes names; built once,
    /// from `config`.
    pub forcing: crate::forcing_extrapolation::ForcingPolicy,
    /// The fit seeds: the file's `fit_seeds`, else `--seed`, else 1. One leaf
    /// each.
    pub seeds: Vec<u64>,
    /// `--sweep`: each point's values replace `[fixed]` ones; one fit folder
    /// per point.
    pub sweep: Vec<(String, Vec<f64>)>,
    /// `--emit-every`: the cadence the `[synthetic]` generator emits at.
    pub emit_every: Option<crate::emit_every::EmitEvery>,
    /// `--resume`: the base a PGAS or PMMH chain continues, as named. Keyed
    /// at the method level, through the target length and the dep on the
    /// base it resolves to.
    pub resume: Option<String>,
    /// The NUTS tuning flags given on the command line (`--max-tree-depth`,
    /// `--diagonal-mass`), by spelling, for the note a PGAS fit gives when the
    /// model cannot run NUTS. Their values are in `config`'s method.
    pub nuts_tuning_flags: Vec<&'static str>,
    /// What the command line set in the problem half of `config`, for the
    /// resolved config each fit folder stores.
    pub problem_flags: ProblemFlags,
    /// How each `[fixed]` name reached the config, read before the
    /// `from_scenario` expansion, for the fit's partition.
    pub fixed_via: Result<Vec<(String, crate::roles::FitFixedVia)>, String>,
    /// The chain-starts rule as resolved.
    pub starts: StartsPlan,
    /// What the file alone, with no option, resolves to (gh#896).
    pub file_only: FileOnly,
}

/// The chain-starts rule, resolved.
pub(crate) struct StartsPlan {
    /// A sourced rule's resolution (its handle read, its source's verdict
    /// checked), taken before the model was compiled; `None` for a bare rule.
    pub early: Option<chain_starts::ResolvedStarts>,
    /// Where the rule came from and what it is, as the run header says it.
    pub resolution: config_v2::StartsResolution,
    /// `--allow-nonconverged-source`: a source fit whose verdict is not
    /// converged is used rather than refused. It decides whether the run
    /// starts, never what it computes once started.
    pub allow_nonconverged_source: bool,
    /// Where the rule came from, for the prefix of an error about it:
    /// `--starts <spec>` or the fit file.
    pub origin: String,
}

/// The method and seeds the file alone resolves to, so the run `fit run
/// <file>` with no option would make can be told from this one (gh#896).
pub(crate) struct FileOnly {
    pub method: config_v2::Method,
    pub seeds: Vec<u64>,
}

/// What the command line set in the problem half of the configuration: the
/// part a fit folder's `fit.resolved.toml` holds. Each field is `true` when the flag
/// set a value the file did not already hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProblemFlags {
    /// `--allow-forcing-extrapolation`: `[config] allow_forcing_extrapolation
    /// = true`.
    pub allow_forcing_extrapolation: bool,
}

impl ProblemFlags {
    /// `problem` without what the command line set in it: the problem the fit
    /// file alone resolves to, for the run `fit run <file>` with no option
    /// would make (gh#896). Each field is the inverse of the fold in
    /// [`resolve`].
    pub fn unfold(&self, problem: &config_v2::Problem) -> config_v2::Problem {
        let ProblemFlags { allow_forcing_extrapolation } = *self;
        let mut file = problem.clone();
        if allow_forcing_extrapolation {
            file.config.allow_forcing_extrapolation = false;
        }
        file
    }

    /// Write every value the command line set in the problem half into `doc`,
    /// the fit file's tables. Each field is the fold [`resolve`] applied to the
    /// typed config.
    fn fold_into(&self, doc: &mut toml::Table) -> Result<(), String> {
        let ProblemFlags { allow_forcing_extrapolation } = *self;
        if allow_forcing_extrapolation {
            table_mut(doc, "config")?
                .insert("allow_forcing_extrapolation".to_string(), toml::Value::Boolean(true));
        }
        Ok(())
    }
}

/// The comment `fit.resolved.toml` opens with: what the file is, and that it is
/// generated.
pub(crate) const RESOLVED_CONFIG_HEADER: &str = "\
# fit.resolved.toml: the configuration the fits in this folder ran under.
# Generated by `camdl fit run`; do not edit. It is the fit file as read (kept
# byte for byte beside it as fit.toml.original) with every value the command
# line set in the problem half written in, and [fixed] this folder's own
# values: a sweep point's applied, a `from_scenario` and a `from_file` read in,
# so it names no fixed-values file. Relative [model], [data] and output_dir
# paths resolve against the fit file's directory (`source_config` in
# fit.meta.json), except that a reader binds each [data.tables] input to the
# copy of its bytes this folder stores in inputs/, checked against the digest
# fit.meta.json records. [estimate], [data.observations] and [data.holdout]
# keep the file's order, which is meaningful; every other table's keys are
# sorted. [method] is the file's; each run's own method settings are in its
# run.json.
";

/// The tables whose key order means something (camdl reads them in order: the
/// estimated parameters', the streams'), by `[table]` header path.
/// [`resolved_config_text`] writes them in the file's order and sorts every
/// other table.
const ORDERED_TABLES: [&[&str]; 3] = [&["estimate"], &["data", "observations"], &["data", "holdout"]];

/// The text of a fit folder's `fit.resolved.toml`
/// ([`crate::run_meta::RESOLVED_CONFIG`]): the configuration its fits ran
/// under, which every reader of a stored fit's config reads.
///
/// It is `text` (the fit file, as read once) with every value of the problem
/// half that `flags` set written in, and its `[fixed]` replaced by `fixed`, the
/// folder's own values as the fit resolved them (a `from_scenario` expanded, a
/// `from_file` read, a sweep point's values applied), so the stored config
/// names no fixed-values file. It is written from the parsed tables, keeping no
/// comment or layout of the file's: the keys of every table are sorted except
/// those of [`ORDERED_TABLES`], which keep the file's order, so the same
/// configuration always gives the same bytes. It opens with
/// [`RESOLVED_CONFIG_HEADER`], and is refused unless the production parser
/// reads it back, with every ordered table in the file's order.
pub(crate) fn resolved_config_text(
    text: &str,
    flags: ProblemFlags,
    fixed: &indexmap::IndexMap<String, f64>,
) -> Result<String, String> {
    resolved_config_text_with(RESOLVED_CONFIG_HEADER, text, flags, fixed)
}

/// [`resolved_config_text`] under another opening comment, `header`: the one
/// construction of the text, for a writer other than `fit run` that must say
/// it wrote the file (`camdl dev migrate`, [`crate::store_migrate`]).
pub(crate) fn resolved_config_text_with(
    header: &str,
    text: &str,
    flags: ProblemFlags,
    fixed: &indexmap::IndexMap<String, f64>,
) -> Result<String, String> {
    let mut doc: toml::Table = text.parse().map_err(|e| format!("cannot parse the config to store it: {e}"))?;
    let written = FitConfig::from_toml_str(text).map_err(|e| format!("cannot parse the config to store it: {e}"))?;
    let orders = ordered_keys(&written);
    flags.fold_into(&mut doc)?;
    let table: toml::Table = fixed.iter().map(|(k, v)| (k.clone(), toml::Value::Float(*v))).collect();
    // Written even when empty: the parser requires a `[fixed]` table.
    doc.insert("fixed".to_string(), toml::Value::Table(table));
    let mut ordered: Vec<(&[&str], toml::Table)> = Vec::new();
    for path in ORDERED_TABLES {
        if let Some(t) = remove_table(&mut doc, path) {
            ordered.push((path, t));
        }
    }
    let mut body = toml::to_string(&sorted(doc)).map_err(|e| format!("cannot write the config to store it: {e}"))?;
    for (path, table) in ordered {
        let order: &[String] = orders.iter().find(|(p, _)| *p == path).map_or(&[], |(_, o)| o.as_slice());
        let mut keys: Vec<&String> = order.iter().filter(|k| table.contains_key(*k)).collect();
        keys.extend(table.keys().filter(|k| !order.contains(k)));
        body.push_str(&format!("\n[{}]\n", path.join(".")));
        for key in keys {
            body.push_str(&format!("{} = {}\n", bare_or_quoted(key), sorted_value(table[key].clone())));
        }
    }
    let text = format!("{header}\n{body}");
    // Every reader parses it with the production parser; a text it refuses,
    // or reads in another order, is refused here, before a fit folder holds a
    // config no reader can open or one that reads differently.
    let back = FitConfig::from_toml_str_named(&text, "fit.resolved.toml")
        .map_err(|e| format!("the resolved config this fit would store does not parse back: {e}"))?;
    if ordered_keys(&back) != orders {
        return Err("the resolved config this fit would store reads its [estimate] or [data] \
                    streams in another order than the file"
            .to_string());
    }
    Ok(text)
}

/// The key order of each of [`ORDERED_TABLES`] in `config`.
fn ordered_keys(config: &FitConfig) -> Vec<(&'static [&'static str], Vec<String>)> {
    let p = &config.problem;
    let data = p.data.as_ref();
    vec![
        (ORDERED_TABLES[0], p.estimate.keys().cloned().collect()),
        (ORDERED_TABLES[1], data.map(|d| d.observations.keys().cloned().collect()).unwrap_or_default()),
        (ORDERED_TABLES[2], data.and_then(|d| d.holdout.as_ref()).map(|h| h.keys().cloned().collect()).unwrap_or_default()),
    ]
}

/// Remove and return the table at `path` in `doc`, if it is one.
fn remove_table(doc: &mut toml::Table, path: &[&str]) -> Option<toml::Table> {
    let (last, parents) = path.split_last()?;
    let mut at = doc;
    for p in parents {
        at = at.get_mut(*p)?.as_table_mut()?;
    }
    match at.remove(*last)? {
        toml::Value::Table(t) => Some(t),
        other => {
            at.insert((*last).to_string(), other);
            None
        }
    }
}

/// `key` as a TOML key: bare when it can be, quoted otherwise.
fn bare_or_quoted(key: &str) -> String {
    if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        key.to_string()
    } else {
        toml::Value::String(key.to_string()).to_string()
    }
}

/// `v`, with the keys of every table in it sorted.
fn sorted_value(v: toml::Value) -> toml::Value {
    match v {
        toml::Value::Table(t) => toml::Value::Table(sorted(t)),
        toml::Value::Array(a) => toml::Value::Array(a.into_iter().map(sorted_value).collect()),
        other => other,
    }
}

/// The table `doc[key]`, made when absent.
fn table_mut<'a>(doc: &'a mut toml::Table, key: &str) -> Result<&'a mut toml::Table, String> {
    doc.entry(key)
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .ok_or_else(|| format!("cannot store the config: its `{key}` is not a table"))
}

/// `doc` with the keys of every table, at every depth, in sorted order, so its
/// text does not depend on the order the file wrote them in (whichever map
/// `toml::Table` is built on).
fn sorted(doc: toml::Table) -> toml::Table {
    fn value(v: toml::Value) -> toml::Value {
        match v {
            toml::Value::Table(t) => toml::Value::Table(sorted(t)),
            toml::Value::Array(a) => toml::Value::Array(a.into_iter().map(value).collect()),
            other => other,
        }
    }
    let mut entries: Vec<(String, toml::Value)> = doc.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.into_iter().map(|(k, v)| (k, value(v))).collect()
}

/// Resolve `fit run`'s configuration from its arguments, its file and the
/// defaults, under the compile settings `compile`; refuse (through `out`) a
/// combination that cannot run. Returns the value and the forcing policy's
/// installation, held for the command.
pub(crate) fn resolve(
    a: &crate::args::FitRunArgs,
    compile: crate::util::CompileSettings,
    out: &crate::out::Out,
) -> (ResolvedFitConfig, crate::forcing_extrapolation::Installed) {
    // Removed flags fail loudly before any work, each naming its replacement
    // (alpha posture: a redirect, not a shim). Every one present is reported,
    // so `--stage posterior --init from_prior` is answered in one message.
    if let Some(msg) = super::removed_flag_message(a) {
        out.fail(format!("error: {}", msg));
    }
    // PGAS parallel tempering was removed (gh#789); the flag is trapped so a
    // stale script is answered with the fix rather than clap's "unexpected
    // argument".
    if let Some(raw) = a._removed_tempering.as_deref() {
        out.fail(format!(
            "error: --tempering is no longer accepted on `camdl fit run`: PGAS parallel \
             tempering was removed (gh#789), and every PGAS chain samples the posterior \
             directly, as the ladder's cold rung did. Drop `--tempering {raw}`; a \
             `tempering = [...]` line under [method] must be deleted too. Fits stored \
             before the removal are not reused and cannot be resumed; they run again."
        ));
    }
    // Correlated pseudo-marginal PMMH was removed; `--rho` is trapped the same
    // way.
    if let Some(raw) = a._removed_rho.as_deref() {
        out.fail(format!(
            "error: --rho is no longer accepted on `camdl fit run`: correlated \
             pseudo-marginal PMMH was removed, and `pmmh` runs the bootstrap particle \
             filter with independent likelihood evaluations. Drop `--rho {raw}`; a \
             `rho = ...` line under [method] must be deleted too. PMMH fits stored \
             before the removal are not reused and cannot be resumed; they run again."
        ));
    }
    let fit_path              = a.config.to_string_lossy().into_owned();
    let base_seed             = a.seed.unwrap_or(1);
    // The file is read here, once; everything below parses, checks and stores
    // this text.
    let config_text = std::fs::read_to_string(&fit_path).unwrap_or_else(|e| {
        out.fail(format!("error: cannot read {fit_path}: {e}"));
    });

    // gh#540: every CLI flag that changes what the method computes or stores,
    // collected once so the method identity can see all of them. Applied to
    // the in-memory method below, before the CAS claim — the dispatch sites
    // no longer override anything, so there is no route around the key.
    let cli_overrides = crate::fit::config_v2::CliMethodOverrides {
        max_tree_depth:       a.max_tree_depth,
        trajectory_warmup:    a.trajectory_warmup,
        csmc_sweeps_per_nuts: a.csmc_sweeps_per_nuts,
        n_trajectories:       a.n_trajectories,
        diagonal_mass:        a.diagonal_mass,
        no_nuts:              a.no_nuts,
        no_ancestor_sampling: a.no_ancestor_sampling,
        no_adapt:             a.no_adapt,
        adapt_start:          a.adapt_start,
        cooling_target_iters: a.cooling_target_iters,
        decibans_thresh:      a.decibans_thresh,
        no_dt_check:          a.no_dt_check,
        dt_check_halvings:    a.dt_check_halvings,
        dt_check_strict:      a.dt_check_strict,
        binomial:             a.binomial,
        record_ancestry:      a.record_ancestry,
        record_prequential:   a.record_prequential,
    };
    let sweep_specs: Vec<(String, Vec<f64>)> = a.sweep.iter()
        .map(|s| (s.name.clone(), s.grid.expand()))
        .collect();

    // Load the file: the problem half and the one `[method]`.
    let path = std::path::Path::new(&fit_path);
    let mut config = FitConfig::parse_anchored_at(&config_text, path, path).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // `fit run` runs the file's one method and refuses a file that has none,
    // naming the table. Every other reader takes the problem alone.
    if let Err(e) = config.method() {
        out.fail(format!("error: {}: {}", fit_path, e));
    }
    // `[method]` keys the method's own settings leave inert. Here and not in
    // the shared parser, which also reads stored fits' archived configs.
    {
        let method = config.inference.method.as_ref().expect("checked above");
        if let Err(e) = config_v2::refuse_inert_method_keys_in(&config_text, method) {
            out.fail(format!("error: {fit_path}: {e}"));
        }
    }
    // gh#audit-C6 / gh#189: degenerate-rate handling is a keyed `[config]` field
    // (folds into the fit-identity hash via the config blob). Every model this
    // fit compiles carries it (`FitBackendConfig::eval_settings`): the
    // capability scan and gradient preflight below, `FitRunConfig::build` for
    // each method, and the `[synthetic]` generator's runs.
    // A forcing outside its range: the flag sets the keyed `[config]` field,
    // so the override reaches the fit identity either way.
    // What the flag folds into the problem is recorded, so the fit folder
    // stores the config the fit ran under (gh#1268).
    let problem_flags = ProblemFlags {
        allow_forcing_extrapolation: a.forcing_extrapolation.allow_forcing_extrapolation
            && !config.problem.config.allow_forcing_extrapolation,
    };
    if a.forcing_extrapolation.allow_forcing_extrapolation {
        config.problem.config.allow_forcing_extrapolation = true;
    }
    // The policy every model this fit evaluates runs under, and every record
    // it writes names: installed for the rest of the command.
    let forcing =
        crate::forcing_extrapolation::ForcingPolicy::new(config.problem.config.allow_forcing_extrapolation);
    let _forcing_route = forcing.install();

    // gh#656: `--emit-every` reaches exactly one thing on this command — the
    // `[synthetic]` generator, which is the only fit path where the emission
    // cadence determines data that is then fitted. A fit against REAL data
    // scores at its data files' own times and never consults `emit_schedule`,
    // so the flag would silently do nothing; refuse and say why rather than
    // leaving the user to wonder which cadence they got.
    //
    // Deliberately NOT written into `config`: the fit-identity hash serializes
    // that document, so parking it there would re-key a real-data fit over a
    // knob that fit cannot see. The override travels as an argument to
    // generation, and keys the fit the honest way — through the generated
    // data's bytes, which `FitDigest.data` already hashes.
    let emit_every = crate::emit_every::EmitEvery::from_cli_specs(&a.emit_every)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    if emit_every.is_some() && !config.problem.is_synthetic_fit() {
        out.fail(format!(
            "error: --emit-every sets the cadence at which SYNTHETIC \
             observations are generated, and {} {}.\n  \
             A fit against real data scores each stream at its own data file's \
             times — `emit_schedule` never enters the likelihood — so this flag \
             would change nothing.\n  \
             Drop the flag, or add a `[synthetic]` block to generate data.",
            fit_path,
            if config.problem.synthetic.is_some() {
                "declares `[data]` beside `[synthetic]`, so it fits the real data"
            } else {
                "declares no `[synthetic]` block"
            }
        ));
    }

    // gh#896: the method as the file alone spells it, before any CLI override,
    // so a run can be told apart from the run `fit run <file>` would make.
    let mut file_method: config_v2::Method =
        config.inference.method.clone().expect("checked above");
    // gh#514 / gh#540: the CLI overrides — `--starts` and the sampler knobs —
    // all change where the chains begin or what the method computes, and
    // therefore the stored output. They are written INTO the method here,
    // before `cas::fit_level_hash` and before the claim, so a different
    // override is a different artifact. Leaving them unset leaves the toml's
    // values in place.
    {
        let method = config.inference.method.as_mut().expect("checked above");
        if !cli_overrides.is_empty() {
            // A flag the method does not read is refused here, before any
            // work, rather than dropped (`OverrideFlag::methods`).
            if let Err(e) = method.algorithm.apply_cli_overrides(&cli_overrides) {
                out.fail(format!("error: {e}"));
            }
        }
        if let Some(starts) = &a.starts {
            method.starts = Some(starts.clone());
        }
        // gh#1056: `--starts-chains` qualifies the rule in force, from the
        // flag or the file; only `from_posterior` takes it.
        if let Some(subset) = &a.starts_chains {
            method.starts = Some(
                crate::fit::starts::apply_starts_chains(method.starts.clone(), subset, "fit")
                    .unwrap_or_else(|e| out.fail(format!("error: {e}"))));
        }
    }

    // A sourced `starts` rule reads a stored fit or a file: resolve the handle
    // now, before the model is compiled, so a bad handle is refused at once
    // and the dep folded into every leaf is one resolution. The source's
    // convergence verdict is checked here (`--allow-nonconverged-source`
    // lifts the refusal) — the one seam an upstream fit is consumed. A bare
    // rule has nothing to resolve until the default is known, below.
    let early_resolved_starts: Option<chain_starts::ResolvedStarts> = {
        let method = config.inference.method.as_ref().expect("checked above");
        match &method.starts {
            Some(rule) if rule.source().is_some() => Some(
                chain_starts::resolve_starts(rule, a.allow_nonconverged_source).unwrap_or_else(|e| {
                    out.fail(format!("error: {}", e));
                }),
            ),
            _ => None,
        }
    };

    // Compile `model.camdl` → IR EXACTLY ONCE for the whole fit. Every
    // per-(cell × sweep point) `FitRunConfig::build` then loads this
    // pre-compiled IR instead of re-invoking camdlc per unit (a swept fit
    // otherwise recompiled the model dozens of times). The resolved path is
    // recorded on `config.problem.compiled_ir`; `model.camdl` is left
    // untouched so the fit's content hash still hashes the original `.camdl`
    // source bytes (identity is unchanged by the hoist). The temp IR persists
    // for the process — `resolve_ir_path`'s returned path is not a drop guard.
    //
    // gh#439 A2: only `nl-lbfgs` on the `ode` backend reads the WrtPop
    // state-Jacobian (`rate_state_grad` / `projection_state_grad`, via the ODE
    // forward-sensitivity gradient in `ode_grad::det_grad`). Otherwise compile lean
    // (`--no-state-grad`), which omits the dense ~O(G^3) Jacobian that dominates
    // coupled-model IR. The bit is folded into the IR-cache key, so a lean entry is
    // never reused for an nl-lbfgs+ode fit (and run identity is gradient-independent,
    // so lean vs full share the same model digest).
    let needs_state_grad = config.needs_state_grad();
    let (compiled_ir, _ir_tmp) = crate::util::resolve_ir_path_with(
        &config.problem.model.camdl, needs_state_grad, None, compile)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // gh#616: if the model declares observation anchors, resolve them ONCE here
    // — from this fit's own `[data]` — and re-emit the compiled IR with the
    // anchors substituted, so every downstream loader (the runner, each sweep
    // point, the archived copy) reads the SAME resolved model.
    //
    // The alternative, substituting in memory at each of those loads, would put
    // the resolution in half a dozen places that must agree; this puts it in
    // one. A FRESH temp is written rather than mutating `compiled_ir` in place,
    // because that path can be a user-supplied `.ir.json` (`resolve_ir_path`
    // returns it unchanged) and a command must never rewrite the user's file.
    //
    // The config's `[simulate] from` is resolved here too. Fit identity keys
    // on the model loaded from the re-emitted IR, so the resolved start and
    // horizon are in it: two windows are two fits, and one start spelled two
    // ways is one.
    let compiled_ir = super::resolve_anchors_into_temp_ir(out, &compiled_ir, &config.problem)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // Bind the model's external tables and forcings to the files
    // `[data.tables]` names, once, the same way: every loader after this reads
    // the bound model, and what was read is kept for the identity and the fit
    // folder.
    let (compiled_ir, tables) = super::bind_tables_into_temp_ir(&compiled_ir, &config.problem)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {fit_path}: {e}"));
        });
    config.problem.compiled_ir = Some(compiled_ir.clone());

    // Load model and validate completeness (from the pre-compiled IR).
    let (model, _) = crate::util::load_model(&compiled_ir).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // The one dt rule (`crate::dt_rule`): the model's declared step is the
    // integration step for every method; a `[config] dt` the file writes that
    // differs is refused. The resolved step is written into the config before
    // anything reads it — the observation grid, the synthetic generator, every
    // runner, and the fit identity, which so keys on the step the fit ran at.
    {
        let written = config_v2::table_declares_key(&config_text, "config", "dt")
            .then_some(config.problem.config.dt);
        match crate::dt_rule::effective_dt(model.simulation.dt, written) {
            Ok(dt) => config.problem.config.dt = dt,
            Err(m) => {
                out.fail(format!("error: {fit_path}: {}", m.message("[config] dt")));
            }
        }
    }
    // How each `[fixed]` name reached the config, for the partition — read
    // before the expansion below folds a scenario's names into the inline
    // values. An error is the expansion's to report.
    let fixed_via = config.problem.fixed.declarations(&model);
    // gh#33: expand `[fixed] from_scenario = "name"` into the inline
    // values map by looking up the named scenario in the model. Must
    // happen after model load but before validate, so the every-param-
    // resolved check sees the expanded values.
    config.problem.expand_fixed_from_scenario(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    let model_params: Vec<String> = model.parameters.iter().map(|p| p.name.clone()).collect();
    // The one model fact `validate` needs beyond the parameter names: does
    // `init { }` DRAW a compartment from a law? It decides the ic_free ×
    // pfilter/pmmh cells (gh#732) — under the bootstrap particle filter a
    // declared law is the whole source of the swarm's spread at t=0.
    let init_law = if model.initial_conditions.iter().any(|(_, s)| s.is_law()) {
        crate::fit::methods::InitLaw::Declared
    } else {
        crate::fit::methods::InitLaw::Absent
    };
    config.validate(&model_params, init_law).unwrap_or_else(|e| {
        // A model constant named in [estimate] or [fixed] is an unknown
        // parameter; the hint says how to make it configurable.
        let fixed = config.problem.fixed.resolve().unwrap_or_default();
        let named = config.problem.estimate.keys().chain(fixed.keys()).map(|s| s.as_str());
        out.fail(format!("error: {}{}", e, crate::util::constant_hints(&model, named)));
    });
    // gh#75: prior-presence check honoring the IR fallback. Has to run
    // after partition validation but with the model IR in scope.
    let ir_prior_params: std::collections::BTreeSet<&str> = model.parameters.iter()
        .filter(|p| p.prior_dist().is_some() || p.hierarchical().is_some())
        .map(|p| p.name.as_str())
        .collect();
    config.validate_priors_present(&ir_prior_params).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    // gh#369: a fit.toml prior that displaces a model `~` prior is said once,
    // here, before the grid — a property of the config, not of a cell/chain.
    let estimated_names: Vec<String> = config.problem.estimate.keys().cloned().collect();
    if let Some(w) = crate::fit::priors_precedence::format_prior_override_warning(
        &crate::fit::priors_precedence::prior_overrides(
            &estimated_names, &config.problem.estimate, &model),
    ) {
        out.say_raw(format_args!("{}", w));
    }
    // `nl-lbfgs` is the one method whose objective needs a gradient of this
    // model. Ask the gradient capability gate now, before a leaf is claimed, so
    // a model it refuses costs a message rather than a half-run method.
    config.validate_gradient_capability(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_held_hierarchical_leaves(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_hierarchical_leaf_boxes(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_smooth_scale(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    config.validate_smooth_coordinates(&model).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });

    // The chain-starts rule, resolved against the problem before the identity
    // is taken (proposal §3.4): `from_prior` when every estimated parameter
    // declares a prior, `uniform_unconstrained` otherwise, unless the file or
    // `--starts` spelled one. Said at startup either way, so the rule a run
    // used is never a thing to infer.
    let starts_resolution = {
        let method = config.inference.method.as_mut().expect("checked above");
        method.resolve_starts(&config.problem, &model).unwrap_or_else(|e| {
            out.fail(format!("error: {}: {e}", super::starts_origin(a, &fit_path)));
        })
    };
    let starts_origin = super::starts_origin(a, &fit_path);
    // From here the method is read-only: the identity below sees exactly what
    // the runners consume.
    let method: &config_v2::Method = config.inference.method.as_ref().expect("checked above");
    // gh#896: the method the file alone resolves to, for the identity of the
    // run `fit run <file>` with no CLI options would key (below, per cell).
    let _ = file_method.resolve_starts(&config.problem, &model);
    // The seeds the file alone runs: its `fit_seeds`, else the default seed.
    let file_seeds: Vec<u64> = config.inference.fit_seeds.clone().unwrap_or_else(|| vec![1]);
    let allow_nonconverged_source = a.allow_nonconverged_source;
    // Only a fit source carries a convergence verdict to lift.
    if allow_nonconverged_source {
        let rule = method.starts().unwrap_or_else(|e| { out.fail(format!("error: {e}")) });
        if let Err(e) = chain_starts::refuse_inert_allow_nonconverged(rule, "fit") {
            out.fail(format!("error: {e}"));
        }
    }
    if let Some(msg) = config.point_start_multichain_note() {
        // gh#71: a point-started multi-chain sampler's R̂ is uninformative;
        // the summary will report it as not assessed. A note, not an error —
        // the sample is still valid.
        out.say(format_args!("{} {}", crate::out::paint::note_tag(), msg));
    }
    // `--resume` continues a PGAS or PMMH chain. Other methods have no
    // extension dimension (IF2's cooling depends on total iterations, the
    // particle filter is one pass), so resuming would be statistically
    // incoherent.
    if a.resume.is_some()
        && !matches!(method.algorithm, config_v2::Algorithm::PGAS { .. } | config_v2::Algorithm::PMMH { .. })
    {
        out.fail(format!("error: --resume is only supported for PGAS and PMMH methods; \
                   this fit's method is '{}'.", method.algorithm.method_name()));
    }
    if a.seed.is_some() && config.inference.fit_seeds.is_some() {
        out.fail(format!("error: --seed sets the single seed of a fit, and {fit_path} declares \
                   `fit_seeds`, which replaces it, so the flag would change nothing. Drop \
                   --seed, or delete `fit_seeds` from the file."));
    }
    let fit_seeds: Vec<u64> = match &config.inference.fit_seeds {
        Some(list) => list.clone(),
        None       => vec![base_seed],
    };

    let nuts_tuning_flags: Vec<&'static str> =
        [(a.max_tree_depth.is_some(), "--max-tree-depth"), (a.diagonal_mass, "--diagonal-mass")]
            .into_iter()
            .filter_map(|(on, name)| on.then_some(name))
            .collect();
    let resolved = ResolvedFitConfig {
        config_text,
        config,
        model,
        tables,
        engine: super::fit_engine(),
        forcing,
        seeds: fit_seeds,
        sweep: sweep_specs,
        emit_every,
        resume: a.resume.clone(),
        nuts_tuning_flags,
        problem_flags,
        fixed_via,
        starts: StartsPlan {
            early: early_resolved_starts,
            resolution: starts_resolution,
            allow_nonconverged_source,
            origin: starts_origin,
        },
        file_only: FileOnly { method: file_method, seeds: file_seeds },
    };
    (resolved, _forcing_route)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_FLAG: ProblemFlags = ProblemFlags { allow_forcing_extrapolation: false };

    /// The least a fit file can hold, ending inside its `[fixed]` table.
    const PROBLEM: &str = "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.01, 1.0] }\n[fixed]\n";

    fn fixed(values: &[(&str, f64)]) -> indexmap::IndexMap<String, f64> {
        values.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn parse(text: &str) -> toml::Table {
        text.parse().unwrap_or_else(|e| panic!("{e}\n{text}"))
    }

    #[test]
    fn the_forcing_flag_is_written_into_the_resolved_config() {
        let flags = ProblemFlags { allow_forcing_extrapolation: true };
        let with_dt = format!("{PROBLEM}[config]\ndt = 0.5\n");
        for text in [PROBLEM, with_dt.as_str()] {
            let stored = resolved_config_text(text, flags, &fixed(&[])).unwrap();
            let doc = parse(&stored);
            assert_eq!(doc["config"]["allow_forcing_extrapolation"].as_bool(), Some(true), "{stored}");
            assert_eq!(doc["model"]["camdl"].as_str(), Some("m.camdl"), "the rest is the file's: {stored}");
        }
        let doc = parse(&resolved_config_text(&with_dt, flags, &fixed(&[])).unwrap());
        assert_eq!(doc["config"]["dt"].as_float(), Some(0.5));
        let doc = parse(&resolved_config_text(&with_dt, NO_FLAG, &fixed(&[])).unwrap());
        assert!(doc["config"].get("allow_forcing_extrapolation").is_none(), "no flag, no value written");
    }

    /// `[fixed]` is the folder's own: a sweep point's value replaces the
    /// file's, and the rest of the file's values stay.
    #[test]
    fn the_folders_fixed_values_replace_the_files() {
        let text = "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.01, 1.0] }\n\
                    [fixed]\nk = 0.3\ngamma = 2\n";
        let stored = resolved_config_text(text, NO_FLAG, &fixed(&[("k", 0.4), ("gamma", 2.0)])).unwrap();
        let doc = parse(&stored);
        assert_eq!(doc["fixed"]["k"].as_float(), Some(0.4), "{stored}");
        assert_eq!(doc["fixed"]["gamma"].as_float(), Some(2.0), "{stored}");
        let config = FitConfig::from_toml_str(&stored).unwrap();
        assert_eq!(config.problem.fixed.values.get("k"), Some(&0.4));
        // An empty [fixed] is kept: the parser requires the table.
        let empty = "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.01, 1.0] }\n[fixed]\n";
        let stored = resolved_config_text(empty, NO_FLAG, &fixed(&[])).unwrap();
        assert!(parse(&stored)["fixed"].as_table().is_some_and(|t| t.is_empty()), "{stored}");
        FitConfig::from_toml_str(&stored).unwrap_or_else(|e| panic!("an empty [fixed] parses back: {e}\n{stored}"));
    }

    /// `[estimate]`, `[data.observations]` and `[data.holdout]` keep the file's
    /// order, which camdl reads them in; every other table is sorted. The
    /// parsed config reads them back in that order.
    #[test]
    fn the_ordered_tables_keep_the_files_order() {
        let text = "[model]\ncamdl = \"m.camdl\"\n[data]\nholdout_after = 3.0\n\
                    [data.observations]\nzeta = \"z.tsv\"\nalpha = \"a.tsv\"\n\
                    [data.holdout]\nzeta = \"zh.tsv\"\nalpha = \"ah.tsv\"\n\
                    [estimate]\ngamma = { bounds = [0.1, 1.0] }\nbeta = { bounds = [0.01, 1.0], start = 0.5 }\n\
                    [estimate.alpha]\nbounds = [0.0, 2.0]\n[estimate.alpha.prior.uniform]\n\
                    [fixed]\nk = 1.0\n";
        let stored = resolved_config_text(text, NO_FLAG, &fixed(&[("k", 1.0)])).unwrap();
        let back = FitConfig::from_toml_str(&stored).unwrap_or_else(|e| panic!("{e}\n{stored}"));
        let names: Vec<&str> = back.problem.estimate.keys().map(String::as_str).collect();
        assert_eq!(names, ["gamma", "beta", "alpha"], "{stored}");
        let data = back.problem.data.as_ref().unwrap();
        assert_eq!(data.observations.keys().collect::<Vec<_>>(), ["zeta", "alpha"], "{stored}");
        assert_eq!(data.holdout.as_ref().unwrap().keys().collect::<Vec<_>>(), ["zeta", "alpha"], "{stored}");
        assert!(stored.find("gamma =").unwrap() < stored.find("beta =").unwrap(), "{stored}");
        assert_eq!(back.problem.estimate["beta"].start, Some(0.5));
    }

    /// The text is generated, says so, and depends on the configuration only:
    /// comments, layout, key order and number spelling do not reach it.
    #[test]
    fn the_resolved_config_is_canonical_and_marked_generated() {
        let a = format!("# my fit\n{PROBLEM}k = 0.30\n\n[config]\ndt = 0.50\nallow_forcing_extrapolation = false\n");
        let b = "[config]\nallow_forcing_extrapolation = false\ndt = 0.5 # half a day\n[fixed]\nk = 0.3\n\
                 [estimate]\nbeta = { bounds = [0.01, 1.0] }\n[model]\n  camdl = \"m.camdl\"\n";
        let ra = resolved_config_text(&a, NO_FLAG, &fixed(&[("k", 0.3)])).unwrap();
        let rb = resolved_config_text(b, NO_FLAG, &fixed(&[("k", 0.3)])).unwrap();
        assert_eq!(ra, rb);
        assert!(ra.starts_with(RESOLVED_CONFIG_HEADER), "{ra}");
        assert!(ra.contains("Generated by `camdl fit run`"), "{ra}");
        assert!(!ra.contains("my fit") && !ra.contains("half a day"), "{ra}");
        assert_eq!(resolved_config_text(&a, NO_FLAG, &fixed(&[("k", 0.3)])).unwrap(), ra, "same bytes every time");
    }
}
