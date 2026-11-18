//! The environment: every environment variable camdl reads, read once, here
//! (`docs/dev/proposals/2026-11-17-honest-input-addressed-store.md`, part E).
//!
//! Each variable is a field of [`Env`], listed in [`VARS`] with its name, its
//! class by effect on results (as a command-line flag is classified,
//! [`crate::input_class`]), a stability tag and, for a variable that can change
//! a result, the route by which it enters the run identity.
//! [`get`] reads them all on first use and holds the values for the life of the
//! process; nothing else in `cli` or `sim` reads the environment
//! (`rust/crates/cli/tests/environment_is_read_in_one_module.rs`). The engine
//! is told through values: [`install_engine_switches`] passes the evaluator
//! switches to `sim` once, at start-up.
//!
//! **Stability.** `stable` is a supported setting; `debug` is a diagnostic or an
//! alternative evaluation path meant to agree with the default one;
//! `experimental` is an unreleased feature, enabled by a variable named
//! `CAMDL_EXPERIMENTAL_<NAME>`. Every such variable that is set is an
//! experimental input of the run, whether or not a feature reads it yet: it is
//! keyed, and `run.json`'s provenance lists it with its value
//! ([`Environment::experimental_inputs`]). A feature reads its switch from
//! [`Env::experimental`].
//!
//! **Identity.** A variable that can change a result is keyed. The alternative
//! evaluation paths (`CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_BINDING_CACHE`) are meant
//! to be equivalent to the default path and pass the current differential tests,
//! but finite tests do not establish equivalence on every model, so they are
//! keyed until stronger justification exists. They and the experimental inputs
//! enter the engine identity ([`Environment::engine_identity`]): a run made with
//! one set never shares an address with a default run, and a default run's
//! identity is unchanged. `CAMDL_NO_CONSTANT_FOLD` and `CAMDL_NO_LICM` change the
//! IR the compiler emits and are keyed through the model level. Only `simulate`
//! puts [`Environment::engine_identity`] into its identity so far; the other
//! commands adopt it as they are converted (proposal phase 3 step 4).
//!
//! Not here: `clap`'s `env = "CAMDL_…"` attributes (`CAMDL_SEED`, `CAMDL_OUTPUT`,
//! `CAMDL_OUTPUT_DIR` for `simulate`, `CAMDL_PARALLEL`), which are another
//! spelling of a flag and are classified with it; `std::env::args`,
//! `current_dir` and `temp_dir`, which are the process's context, not settings;
//! and compile-time `env!` constants.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::OnceLock;

/// The prefix that names an experimental input.
pub(crate) const EXPERIMENTAL_PREFIX: &str = "CAMDL_EXPERIMENTAL_";

/// How settled a variable is. See the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stability {
    Stable,
    Experimental,
    Debug,
}

/// How a variable that can change a result enters the run identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Route {
    /// It does not change a result (its class is not `keyed`).
    None,
    /// Through the engine identity string in the model level
    /// ([`Environment::engine_identity`]).
    Engine,
    /// Through the compiled IR the model level hashes.
    CompiledModel,
}

/// One environment variable camdl reads, or one family of them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Var {
    /// The variable's name; for the experimental family, the prefix.
    pub name: &'static str,
    /// The [`Env`] field it is read into.
    pub field: &'static str,
    /// What it may do to a stored result.
    pub class: InputClass,
    pub stability: Stability,
    pub route: Route,
}

const fn var(name: &'static str, field: &'static str, class: InputClass, stability: Stability, route: Route) -> Var {
    Var { name, field, class, stability, route }
}

use crate::input_class::InputClass::{self, Execution, Keyed, Presentation};
use crate::input_class::Level::Model;

/// Every variable, in [`Env`] field order.
pub(crate) const VARS: &[Var] = &[
    var("CAMDL_TRACE_STEPS", "trace_steps", Presentation, Stability::Debug, Route::None),
    var("CAMDL_EVAL_UNRESOLVED", "eval_unresolved", Keyed(Model), Stability::Debug, Route::Engine),
    var("CAMDL_NO_BINDING_CACHE", "no_binding_cache", Keyed(Model), Stability::Debug, Route::Engine),
    var(EXPERIMENTAL_PREFIX, "experimental", Keyed(Model), Stability::Experimental, Route::Engine),
    var("CAMDL_NO_CONSTANT_FOLD", "no_constant_fold", Keyed(Model), Stability::Debug, Route::CompiledModel),
    var("CAMDL_NO_LICM", "no_licm", Keyed(Model), Stability::Debug, Route::CompiledModel),
    var("CAMDL_NO_IR_CACHE", "no_ir_cache", Execution, Stability::Stable, Route::None),
    var("CAMDL_IR_CACHE_DIR", "ir_cache_dir", Execution, Stability::Stable, Route::None),
    var("CAMDL_SKIP_VERSION_CHECK", "skip_version_check", Execution, Stability::Debug, Route::None),
    var("CAMDLC_PATH", "camdlc_path", Execution, Stability::Stable, Route::None),
    var("CAMDLC", "camdlc", Execution, Stability::Stable, Route::None),
    var("CAMDL_OUTPUT_DIR", "output_dir", Execution, Stability::Stable, Route::None),
    var("CAMDL_UNLABELED_THRESHOLD", "unlabeled_threshold", Presentation, Stability::Stable, Route::None),
    var("PATH", "path", Execution, Stability::Stable, Route::None),
    var("XDG_CACHE_HOME", "xdg_cache_home", Execution, Stability::Stable, Route::None),
    var("HOME", "home", Execution, Stability::Stable, Route::None),
    var("NO_COLOR", "no_color", Presentation, Stability::Stable, Route::None),
    var("CLICOLOR_FORCE", "clicolor_force", Presentation, Stability::Stable, Route::None),
    var("CI", "ci", Presentation, Stability::Stable, Route::None),
];

/// The value of every variable in [`VARS`], typed.
#[derive(Debug, Clone, Default)]
pub(crate) struct Env {
    /// `CAMDL_TRACE_STEPS=1`: one stderr row per chain-binomial substep and one
    /// line per effect action (`sim::trace`). A diagnostic; writes nothing kept.
    pub trace_steps: bool,
    /// `CAMDL_EVAL_UNRESOLVED` (any value): propensities through the slow
    /// string-keyed evaluator, an alternative path meant to agree with the
    /// default.
    pub eval_unresolved: bool,
    /// `CAMDL_NO_BINDING_CACHE` (any value): bindings recomputed on demand
    /// instead of cached, an alternative path meant to agree with the default.
    pub no_binding_cache: bool,
    /// Every set `CAMDL_EXPERIMENTAL_<NAME>`, by full name, with its value.
    pub experimental: BTreeMap<String, String>,
    /// `CAMDL_NO_CONSTANT_FOLD` (any value): camdlc skips constant folding. It
    /// reads the variable itself; camdl reads it to key the IR cache.
    pub no_constant_fold: bool,
    /// `CAMDL_NO_LICM` (any value): the equivalent of `--no-licm`.
    pub no_licm: bool,
    /// `CAMDL_NO_IR_CACHE` (any value): the equivalent of `--no-ir-cache`.
    pub no_ir_cache: bool,
    /// `CAMDL_IR_CACHE_DIR`: where the compiled-IR cache lives.
    pub ir_cache_dir: Option<PathBuf>,
    /// `CAMDL_SKIP_VERSION_CHECK` (any value): skip the camdl/camdlc version
    /// handshake. The IR the compiler emits is still hashed in the model level.
    pub skip_version_check: bool,
    /// `CAMDLC_PATH`, then `CAMDLC`: the compiler to run, when it is not beside
    /// the camdl binary.
    pub camdlc_path: Option<PathBuf>,
    pub camdlc: Option<PathBuf>,
    /// `CAMDL_OUTPUT_DIR`: the store root for a command without
    /// `--output-dir` and a config that names none.
    pub output_dir: Option<PathBuf>,
    /// `CAMDL_UNLABELED_THRESHOLD`: how many unlabelled fits `fit table` lists
    /// before it suggests labels.
    pub unlabeled_threshold: Option<String>,
    /// `PATH`: where a `camdlc` on the path is looked for.
    pub path: Option<OsString>,
    /// `XDG_CACHE_HOME`, then `HOME`: the base of the default IR cache directory.
    pub xdg_cache_home: Option<PathBuf>,
    pub home: Option<PathBuf>,
    /// `NO_COLOR` (any value): no colour (no-color.org).
    pub no_color: bool,
    /// `CLICOLOR_FORCE` (any value): colour even when not on a terminal.
    pub clicolor_force: bool,
    /// `CI`: `true` or `1` turns on `fit summary --strict`.
    pub ci: Option<String>,
}

/// The environment as read.
#[derive(Debug, Clone, Default)]
pub(crate) struct Environment {
    pub vars: Env,
}

impl Environment {
    /// Read every variable in [`VARS`]: the named ones through `lookup`, the
    /// experimental family from `all` (every variable of the process).
    pub fn read(
        lookup: impl Fn(&str) -> Option<OsString>,
        all: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Self {
        let raw = |name: &str| -> Option<OsString> {
            debug_assert!(VARS.iter().any(|v| v.name == name), "{name} is not in VARS");
            lookup(name)
        };
        let is_set = |name: &str| raw(name).is_some();
        let path = |name: &str| raw(name).map(PathBuf::from);
        let string = |name: &str| raw(name).map(|v| v.to_string_lossy().into_owned());
        let experimental = all
            .into_iter()
            .filter_map(|(k, v)| {
                let k = k.to_str()?.to_string();
                k.starts_with(EXPERIMENTAL_PREFIX).then(|| (k, v.to_string_lossy().into_owned()))
            })
            .collect();
        let vars = Env {
            trace_steps: raw("CAMDL_TRACE_STEPS").is_some_and(|v| v == "1"),
            eval_unresolved: is_set("CAMDL_EVAL_UNRESOLVED"),
            no_binding_cache: is_set("CAMDL_NO_BINDING_CACHE"),
            experimental,
            no_constant_fold: is_set("CAMDL_NO_CONSTANT_FOLD"),
            no_licm: is_set("CAMDL_NO_LICM"),
            no_ir_cache: is_set("CAMDL_NO_IR_CACHE"),
            ir_cache_dir: path("CAMDL_IR_CACHE_DIR").filter(|p| !p.as_os_str().is_empty()),
            skip_version_check: is_set("CAMDL_SKIP_VERSION_CHECK"),
            camdlc_path: path("CAMDLC_PATH"),
            camdlc: path("CAMDLC"),
            output_dir: path("CAMDL_OUTPUT_DIR"),
            unlabeled_threshold: string("CAMDL_UNLABELED_THRESHOLD"),
            path: raw("PATH"),
            xdg_cache_home: path("XDG_CACHE_HOME"),
            home: path("HOME"),
            no_color: is_set("NO_COLOR"),
            clicolor_force: is_set("CLICOLOR_FORCE"),
            ci: string("CI"),
        };
        Environment { vars }
    }

    /// The engine switches that are on, each as its name and, for an
    /// experimental input, its value, in [`VARS`] order: the keyed variables
    /// whose route is [`Route::Engine`].
    fn engine_switches(&self) -> Vec<(String, Option<String>)> {
        let mut on = Vec::new();
        for v in VARS.iter().filter(|v| matches!(v.class, Keyed(_)) && v.route == Route::Engine) {
            match v.field {
                "eval_unresolved" if self.vars.eval_unresolved => on.push((v.name.to_string(), None)),
                "no_binding_cache" if self.vars.no_binding_cache => on.push((v.name.to_string(), None)),
                "experimental" => on.extend(self.vars.experimental.iter().map(|(k, x)| (k.clone(), Some(x.clone())))),
                "eval_unresolved" | "no_binding_cache" => {}
                other => unreachable!("`{other}` is routed to the engine identity but not read into it"),
            }
        }
        on
    }

    /// The engine identity a run's model level folds: `engine_id` (the build
    /// fingerprint) followed, for each engine switch that is on, by
    /// `+<len>:<NAME>` and, for an experimental input, `=<len>:<VALUE>`, the
    /// lengths in bytes. Unchanged when no switch is on, so a default run keeps
    /// its identity. The lengths make the encoding injective: a value that
    /// spells `+` or `=` cannot pass for a second switch or a longer name.
    pub fn engine_identity(&self, engine_id: &str) -> String {
        let mut id = engine_id.to_string();
        for (name, value) in self.engine_switches() {
            id.push_str(&format!("+{}:{name}", name.len()));
            if let Some(value) = value {
                id.push_str(&format!("={}:{value}", value.len()));
            }
        }
        id
    }

    /// Every experimental input set for this process, by name, with its value:
    /// what `run.json`'s provenance lists.
    pub fn experimental_inputs(&self) -> BTreeMap<String, String> {
        let experimental = VARS.iter().any(|v| v.stability == Stability::Experimental && v.field == "experimental");
        debug_assert!(experimental, "the experimental family is in VARS");
        if experimental { self.vars.experimental.clone() } else { BTreeMap::new() }
    }
}

/// The process's environment, read on first use and never again.
pub(crate) fn get() -> &'static Environment {
    static ENV: OnceLock<Environment> = OnceLock::new();
    ENV.get_or_init(|| Environment::read(|name| std::env::var_os(name), std::env::vars_os()))
}

/// Pass the evaluator switches to the engine. Called once, at start-up, before
/// any command runs a model.
pub(crate) fn install_engine_switches(env: &Env) {
    sim::trace::configure(sim::trace::Switches {
        steps: env.trace_steps,
        eval_unresolved: env.eval_unresolved,
    });
    sim::resolved_expr::set_binding_cache_off_by_default(env.no_binding_cache);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_class::Level;
    use std::cell::RefCell;
    use std::collections::BTreeSet;

    #[test]
    fn every_field_has_one_variable_and_every_variable_one_field() {
        // An exhaustive destructure: a field added to `Env` does not compile
        // until it is named here, and the list below must then name it too.
        let Env {
            trace_steps: _,
            eval_unresolved: _,
            no_binding_cache: _,
            experimental: _,
            no_constant_fold: _,
            no_licm: _,
            no_ir_cache: _,
            ir_cache_dir: _,
            skip_version_check: _,
            camdlc_path: _,
            camdlc: _,
            output_dir: _,
            unlabeled_threshold: _,
            path: _,
            xdg_cache_home: _,
            home: _,
            no_color: _,
            clicolor_force: _,
            ci: _,
        } = Env::default();
        let fields = [
            "trace_steps", "eval_unresolved", "no_binding_cache", "experimental", "no_constant_fold", "no_licm",
            "no_ir_cache", "ir_cache_dir", "skip_version_check", "camdlc_path", "camdlc", "output_dir",
            "unlabeled_threshold", "path", "xdg_cache_home", "home", "no_color", "clicolor_force", "ci",
        ];
        let listed: Vec<&str> = VARS.iter().map(|v| v.field).collect();
        assert_eq!(fields.to_vec(), listed, "VARS lists Env's fields, in order");
        let names: BTreeSet<&str> = VARS.iter().map(|v| v.name).collect();
        assert_eq!(names.len(), VARS.len(), "a variable is listed once");
    }

    #[test]
    fn a_variable_that_can_change_a_result_has_an_identity_route() {
        for v in VARS {
            let class = v.class;
            let keyed = matches!(class, InputClass::Keyed(_));
            assert_eq!(keyed, v.route != Route::None, "{}: keyed exactly when routed into identity", v.name);
            if keyed {
                assert_eq!(class, InputClass::Keyed(Level::Model), "{} enters at the model level", v.name);
            }
            assert_eq!(
                v.stability == Stability::Experimental,
                v.name == EXPERIMENTAL_PREFIX,
                "{}: the experimental family is the one experimental entry",
                v.name
            );
        }
    }

    #[test]
    fn read_asks_for_every_named_variable_and_no_other() {
        let asked = RefCell::new(BTreeSet::new());
        let _ = Environment::read(
            |name| {
                asked.borrow_mut().insert(name.to_string());
                None
            },
            Vec::new(),
        );
        let named: BTreeSet<String> =
            VARS.iter().filter(|v| v.name != EXPERIMENTAL_PREFIX).map(|v| v.name.to_string()).collect();
        assert_eq!(*asked.borrow(), named);
    }

    fn with(pairs: &[(&str, &str)]) -> Environment {
        let pairs: Vec<(OsString, OsString)> =
            pairs.iter().map(|(k, v)| (OsString::from(k), OsString::from(v))).collect();
        let lookup = |name: &str| pairs.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone());
        Environment::read(lookup, pairs.clone())
    }

    #[test]
    fn an_engine_switch_keys_the_engine_and_the_default_does_not() {
        assert_eq!(with(&[]).engine_identity("0.4.0+abc"), "0.4.0+abc");
        assert_eq!(
            with(&[("CAMDL_EVAL_UNRESOLVED", "1")]).engine_identity("0.4.0+abc"),
            "0.4.0+abc+21:CAMDL_EVAL_UNRESOLVED"
        );
        assert_eq!(
            with(&[("CAMDL_NO_BINDING_CACHE", ""), ("CAMDL_EVAL_UNRESOLVED", "x")]).engine_identity("e"),
            "e+21:CAMDL_EVAL_UNRESOLVED+22:CAMDL_NO_BINDING_CACHE"
        );
        assert_eq!(
            with(&[("CAMDL_EXPERIMENTAL_FOO", "2")]).engine_identity("e"),
            "e+22:CAMDL_EXPERIMENTAL_FOO=1:2"
        );
        // A diagnostic changes no result, and a compile switch is keyed
        // through the IR, so neither touches the engine string.
        assert_eq!(with(&[("CAMDL_TRACE_STEPS", "1"), ("CAMDL_NO_LICM", "1")]).engine_identity("e"), "e");
    }

    /// Two different switch sets never give one engine identity, whatever the
    /// values hold: one variable whose value spells a second variable is not
    /// the two variables.
    #[test]
    fn the_engine_identity_is_injective_in_the_switches() {
        let spelled = with(&[("CAMDL_EXPERIMENTAL_A", "1+CAMDL_EXPERIMENTAL_B=2")]).engine_identity("e");
        let two = with(&[("CAMDL_EXPERIMENTAL_A", "1"), ("CAMDL_EXPERIMENTAL_B", "2")]).engine_identity("e");
        assert_ne!(spelled, two);
        let eq_in_value = with(&[("CAMDL_EXPERIMENTAL_A", "1=2")]).engine_identity("e");
        let eq_in_name = with(&[("CAMDL_EXPERIMENTAL_A=1", "2")]).engine_identity("e");
        assert_ne!(eq_in_value, eq_in_name);
    }

    #[test]
    fn every_engine_routed_variable_is_an_engine_switch() {
        // Each `Route::Engine` variable, set, appears in the engine identity.
        for v in VARS.iter().filter(|v| matches!(v.class, Keyed(_)) && v.route == Route::Engine) {
            let name = if v.name == EXPERIMENTAL_PREFIX { format!("{EXPERIMENTAL_PREFIX}Z") } else { v.name.to_string() };
            let env = with(&[(name.as_str(), "1")]);
            assert_ne!(env.engine_identity("e"), "e", "{} is routed to the engine but does not key it", v.name);
        }
    }

    #[test]
    fn trace_steps_needs_the_value_one() {
        assert!(with(&[("CAMDL_TRACE_STEPS", "1")]).vars.trace_steps);
        assert!(!with(&[("CAMDL_TRACE_STEPS", "0")]).vars.trace_steps);
    }

    #[test]
    fn experimental_inputs_lists_every_set_experimental_variable_and_nothing_else() {
        let env = with(&[("CAMDL_EXPERIMENTAL_X", "on"), ("CAMDL_TRACE_STEPS", "1"), ("CAMDL_SEED", "3")]);
        assert_eq!(
            env.experimental_inputs(),
            BTreeMap::from([("CAMDL_EXPERIMENTAL_X".to_string(), "on".to_string())])
        );
        assert!(with(&[("CAMDL_TRACE_STEPS", "1")]).experimental_inputs().is_empty());
    }
}
