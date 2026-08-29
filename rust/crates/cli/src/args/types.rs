use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

// ─── ParamOverride ────────────────────────────────────────────────────────────

/// `--param NAME=VALUE`  e.g. `--param R0=2.5`
#[derive(Clone, Debug)]
pub struct ParamOverride {
    pub name:  String,
    pub value: f64,
}

impl FromStr for ParamOverride {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, val) = s.split_once('=')
            .ok_or_else(|| format!("expected NAME=VALUE, got '{}'", s))?;
        let value = val.parse::<f64>()
            .map_err(|_| format!("'{}' is not a valid float in --param {}", val, name))?;
        Ok(Self { name: name.to_string(), value })
    }
}

// ─── TableSpec ────────────────────────────────────────────────────────────────

/// `--table NAME=FILE`  e.g. `--table contact=matrix.tsv`
#[derive(Clone, Debug)]
pub struct TableSpec {
    pub name: String,
    pub path: PathBuf,
}

impl FromStr for TableSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, path) = s.split_once('=')
            .ok_or_else(|| format!("expected NAME=FILE, got '{}'", s))?;
        Ok(Self { name: name.to_string(), path: PathBuf::from(path) })
    }
}

// ─── DataSpec ─────────────────────────────────────────────────────────────────

/// `--data PATH`  or  `--data NAME=PATH` (repeatable).
///
/// gh#90: the primary multi-stream surface for `camdl profile` /
/// `camdl pfilter`. Polymorphic — same flag accepts both forms, mirroring
/// the existing `TableSpec` precedent (`--table NAME=FILE`). The two
/// forms are mutually exclusive within a single invocation (the
/// dispatch-side resolver enforces this).
///
/// - `--data PATH` — single-stream. Binds to the model's single obs
///   block (or the one named by `--obs`).
/// - `--data NAME=PATH` — multi-stream. Each pair binds one obs block
///   by name. Repeat the flag for every stream.
///
/// Empty `NAME` (`--data =foo.tsv`) is treated as a single PATH whose
/// filename literally starts with `=`, not as a malformed NAME=PATH.
/// The guard (`!name.is_empty()`) intentionally lets such a path
/// through rather than rejecting it — single-PATH form swallows the
/// whole argument, and the file-existence check downstream catches
/// the (unlikely) typo.
#[derive(Clone, Debug)]
pub enum DataSpec {
    /// `--data PATH` — no `=`, or `=` at position 0.
    Single(PathBuf),
    /// `--data NAME=PATH` — explicit stream-name binding.
    Named { name: String, path: PathBuf },
}

impl FromStr for DataSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once('=') {
            Some((name, path)) if !name.is_empty() => Ok(DataSpec::Named {
                name: name.to_string(),
                path: PathBuf::from(path),
            }),
            _ => Ok(DataSpec::Single(PathBuf::from(s))),
        }
    }
}


// ─── Backend ──────────────────────────────────────────────────────────────────

/// Simulation backend.  Used as a clap `ValueEnum` so `--help` lists variants.
///
/// Canonical names use snake_case on the CLI (`--backend chain_binomial`)
/// to match the IR JSON and run.json `backend` field. Clap 4's default
/// ValueEnum rendering is kebab-case, which would diverge from
/// everywhere else in the tool — hence the explicit `#[value(name=...)]`.
/// The kebab form is kept as an alias so old scripts don't break.
///
/// Serde derives use the same canonical names so this enum can
/// substitute for `String` in `run.json` (`SimulateMeta.backend`),
/// `fit.toml` (`[config].backend`), and `SimRun.backend` without
/// changing the wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum,
         serde::Serialize, serde::Deserialize)]
pub enum ForwardBackend {
    #[value(name = "gillespie")]
    #[serde(rename = "gillespie")]
    Gillespie,
    #[value(name = "chain_binomial", alias = "chain-binomial")]
    #[serde(rename = "chain_binomial", alias = "chain-binomial")]
    ChainBinomial,
    #[value(name = "ode")]
    #[serde(rename = "ode")]
    Ode,
}

impl ForwardBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gillespie     => "gillespie",
            Self::ChainBinomial => "chain_binomial",
            Self::Ode           => "ode",
        }
    }
}

/// Which predictive horizon `camdl fit predict` emits. ABSENT on the command
/// line = "all applicable" (the default): a chain-binomial fit emits both
/// `free_forward` and `one_step`; an ODE fit emits `free_forward` only (its
/// one-step is identical, so it is simply not applicable — no error). An
/// explicit `--horizon one_step` on an ODE fit is a hard error with a redirect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HorizonArg {
    #[value(name = "free_forward", alias = "free-forward")]
    FreeForward,
    #[value(name = "one_step", alias = "one-step")]
    OneStep,
}

/// CLI override for the ODE integrator METHOD (gh#166). Method-only by design:
/// the tolerances are a model property (the `simulate { integrator = rk45 { … } }`
/// block, where the type forbids orphan tolerances), so there is no `--atol`/
/// `--rtol` flag to orphan. Forcing rk45 preserves the model's tolerances when it
/// declared them, else uses the runtime defaults; forcing rk4 drops them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum IntegratorArg {
    #[value(name = "rk4")]
    Rk4,
    #[value(name = "rk45")]
    Rk45,
}

impl std::fmt::Display for ForwardBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── ProgressMode ─────────────────────────────────────────────────────────────

/// Controls how long-running subcommands (`fit run`, `simulate`, `pfilter`,
/// ...) report progress. Rationale and semantics: see GH #14.
///
/// - `auto` — pretty indicatif bars if stderr is a TTY, otherwise plain
///   timestamped log lines. The default.
/// - `pretty` — force indicatif bars regardless of TTY detection.
/// - `plain` — force plain text lines (no `\r`, no ANSI). The mode to use
///   under `tee`, `&> log`, `ssh`, CI, or any non-interactive driver that
///   wants to tail/grep progress.
/// - `none`   — suppress progress output entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ProgressMode {
    #[default]
    #[value(name = "auto")]
    Auto,
    #[value(name = "pretty")]
    Pretty,
    #[value(name = "plain")]
    Plain,
    #[value(name = "none")]
    None,
}

impl ProgressMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto   => "auto",
            Self::Pretty => "pretty",
            Self::Plain  => "plain",
            Self::None   => "none",
        }
    }
}

impl std::fmt::Display for ProgressMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── SeedSpec ─────────────────────────────────────────────────────────────────

/// `--seeds 1:100`  or  `--seeds 1,2,42`
#[derive(Clone, Debug)]
pub enum SeedSpec {
    Range { from: u64, to: u64 },
    List(Vec<u64>),
}

impl SeedSpec {
    pub fn expand(&self) -> Vec<u64> {
        match self {
            Self::Range { from, to } => (*from..=*to).collect(),
            Self::List(v)            => v.clone(),
        }
    }
}

impl FromStr for SeedSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((a, b)) = s.split_once(':') {
            let from = a.parse::<u64>()
                .map_err(|_| format!("invalid seed range start '{}' in '{}'", a, s))?;
            let to   = b.parse::<u64>()
                .map_err(|_| format!("invalid seed range end '{}' in '{}'", b, s))?;
            if from > to {
                return Err(format!("seed range {}:{} is empty (from > to)", from, to));
            }
            Ok(Self::Range { from, to })
        } else {
            let list = s.split(',')
                .map(|t| t.trim().parse::<u64>()
                    .map_err(|_| format!("'{}' is not a valid seed in '{}'", t, s)))
                .collect::<Result<Vec<_>, _>>()?;
            if list.is_empty() {
                return Err(format!("empty seed list '{}'", s));
            }
            Ok(Self::List(list))
        }
    }
}

// ─── SweepSpec ────────────────────────────────────────────────────────────────

/// `--sweep NAME=SPEC` where SPEC is one of:
///   - `V1,V2,...`         explicit list, e.g. `beta=0.1,0.2,0.3`
///   - `lin(min,max,n)`    n linearly-spaced values, endpoints inclusive
///   - `log10(min,max,n)`  n log10-spaced values, endpoints inclusive
///
/// Naming note: function names are pinned to the base (`log10`, no
/// generic `log`) to avoid colliding with the camdl DSL's `log(...)`,
/// which means natural log inside rate expressions
/// (`ocaml/lib/compiler/expander.ml` and `rust/crates/sim/src/propensity.rs`
/// both implement it as `ln`). Rebinding `log` here would make
/// `log(0.1, 10, 5)` mean two different things in two contexts.
#[derive(Clone, Debug)]
pub struct SweepSpec {
    pub name: String,
    pub grid: Grid,
}

#[derive(Clone, Debug)]
pub enum Grid {
    List(Vec<f64>),
    Linear { min: f64, max: f64, n: usize },
    Log10  { min: f64, max: f64, n: usize },
}

impl Grid {
    /// Concrete value list. `n=1` collapses to `[min]`; `n>=2` lays out
    /// endpoints inclusively on the chosen scale.
    pub fn expand(&self) -> Vec<f64> {
        match *self {
            Grid::List(ref v) => v.clone(),
            Grid::Linear { min, max, n } => {
                if n == 1 { return vec![min]; }
                (0..n).map(|i| {
                    let t = i as f64 / (n - 1) as f64;
                    min + (max - min) * t
                }).collect()
            }
            Grid::Log10 { min, max, n } => {
                if n == 1 { return vec![min]; }
                let lmin = min.log10();
                let lmax = max.log10();
                (0..n).map(|i| {
                    let t = i as f64 / (n - 1) as f64;
                    10f64.powf(lmin + (lmax - lmin) * t)
                }).collect()
            }
        }
    }
}

impl FromStr for SweepSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, vals) = s.split_once('=')
            .ok_or_else(|| format!("expected NAME=SPEC, got '{}'", s))?;
        if name.is_empty() {
            return Err(format!("empty parameter name in --sweep '{}'", s));
        }
        let grid = parse_grid(vals.trim(), name)?;
        Ok(Self { name: name.to_string(), grid })
    }
}

/// Parse the value-side of `--sweep NAME=SPEC`. Tries shorthand first
/// (`lin(...)`, `log10(...)`); falls back to comma-list.
fn parse_grid(spec: &str, name: &str) -> Result<Grid, String> {
    if let Some(args) = strip_call(spec, "lin") {
        let (min, max, n) = parse_min_max_n(args, "lin", name)?;
        // NaN arm explicit: `min >= max` alone is false for NaN.
        if min.is_nan() || max.is_nan() || min >= max {
            return Err(format!(
                "lin(min, max, n) requires min < max for --sweep {} (got min={}, max={})",
                name, min, max));
        }
        return Ok(Grid::Linear { min, max, n });
    }
    if let Some(args) = strip_call(spec, "log10") {
        let (min, max, n) = parse_min_max_n(args, "log10", name)?;
        // NaN arm explicit: `min <= 0.0` alone is false for NaN.
        if min.is_nan() || min <= 0.0 {
            return Err(format!(
                "log10(min, max, n) requires min > 0 for --sweep {} (got min={})",
                name, min));
        }
        if min.is_nan() || max.is_nan() || min >= max {
            return Err(format!(
                "log10(min, max, n) requires min < max for --sweep {} (got min={}, max={})",
                name, min, max));
        }
        return Ok(Grid::Log10 { min, max, n });
    }
    // Helpful nudge: someone wrote `log(...)` thinking it would mean
    // log10. Disambiguate before silently parsing it as a list (it
    // wouldn't even succeed — `log(...)` doesn't parse as floats — so
    // this catch is mostly for clarity of the error.)
    if strip_call(spec, "log").is_some() {
        return Err(format!(
            "--sweep {}: `log(...)` is not a sweep function; use `log10(min, max, n)` \
             (the camdl DSL reserves `log` for natural log, so the CLI doesn't bind \
             that name to log10 spacing)", name));
    }
    // Fallback: explicit comma list.
    let values = spec.split(',')
        .map(|v| v.trim().parse::<f64>()
            .map_err(|_| format!("'{}' is not a valid float in --sweep {}", v.trim(), name)))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("no values for --sweep {}", name));
    }
    Ok(Grid::List(values))
}

/// If `spec` has the shape `head(...)`, return the inner string; else `None`.
fn strip_call<'a>(spec: &'a str, head: &str) -> Option<&'a str> {
    let rest = spec.strip_prefix(head)?;
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    Some(inner)
}

fn parse_min_max_n(args: &str, fname: &str, name: &str) -> Result<(f64, f64, usize), String> {
    let parts: Vec<&str> = args.split(',').map(|p| p.trim()).collect();
    if parts.len() != 3 {
        return Err(format!(
            "{}(min, max, n) takes 3 arguments for --sweep {} (got {})",
            fname, name, parts.len()));
    }
    let min = parts[0].parse::<f64>()
        .map_err(|_| format!("{}(...): '{}' is not a valid float (--sweep {})",
            fname, parts[0], name))?;
    let max = parts[1].parse::<f64>()
        .map_err(|_| format!("{}(...): '{}' is not a valid float (--sweep {})",
            fname, parts[1], name))?;
    let n = parts[2].parse::<usize>()
        .map_err(|_| format!("{}(...): '{}' is not a valid positive integer (--sweep {})",
            fname, parts[2], name))?;
    if n < 2 {
        return Err(format!(
            "{}(min, max, n) requires n >= 2 for --sweep {} (got n={})",
            fname, name, n));
    }
    Ok((min, max, n))
}

// ─── RwSd ─────────────────────────────────────────────────────────────────────

/// `--rw-sd auto`  or  `--rw-sd "beta=0.05,rho=0.01"`  or  `--rw-sd "beta=auto,rho=0.01"`
///
/// `None` values in the Map mean "auto" (use heuristic from parameter bounds).
#[derive(Clone, Debug)]
pub enum RwSd {
    Auto,
    Map(HashMap<String, Option<f64>>),
}

impl FromStr for RwSd {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let mut map = HashMap::new();
        for part in s.split(',') {
            let (k, v) = part.trim().split_once('=')
                .ok_or_else(|| format!("expected NAME=VALUE in --rw-sd, got '{}'", part))?;
            let val = if v.eq_ignore_ascii_case("auto") {
                None
            } else {
                Some(v.parse::<f64>()
                    .map_err(|_| format!("'{}' is not a valid float in --rw-sd {}", v, k))?)
            };
            map.insert(k.to_string(), val);
        }
        Ok(Self::Map(map))
    }
}

// ─── ParamVecSpec ─────────────────────────────────────────────────────────────

/// `--param-vec PREFIX=FILE`  e.g. `--param-vec beta=params.tsv`
#[derive(Clone, Debug)]
pub struct ParamVecSpec {
    pub prefix: String,
    pub file:   String,
}

impl FromStr for ParamVecSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (prefix, file) = s.split_once('=')
            .ok_or_else(|| format!("expected PREFIX=FILE, got '{}'", s))?;
        Ok(Self { prefix: prefix.to_string(), file: file.to_string() })
    }
}

// ─── EstimateBoundsSpec ───────────────────────────────────────────────────────

/// `--estimate NAME=LO:HI` for `camdl survey` inline mode.
///
/// e.g. `--estimate "beta=0.001:1.0"`. The colon separator avoids
/// confusion with `--param NAME=VALUE` (single value, no range) and
/// matches the `--sweep NAME=lin(...)` spirit of compositional CLI
/// surfaces. Survey uses these directly as the LHS box.
#[derive(Clone, Debug)]
pub struct EstimateBoundsSpec {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
}

impl FromStr for EstimateBoundsSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, range) = s.split_once('=')
            .ok_or_else(|| format!(
                "expected NAME=LO:HI, got '{}'", s))?;
        let (lo_s, hi_s) = range.split_once(':')
            .ok_or_else(|| format!(
                "expected NAME=LO:HI (with `:` between bounds), got '{}'", s))?;
        let lo = lo_s.parse::<f64>()
            .map_err(|_| format!(
                "'{}' is not a valid float in --estimate {}", lo_s, name))?;
        let hi = hi_s.parse::<f64>()
            .map_err(|_| format!(
                "'{}' is not a valid float in --estimate {}", hi_s, name))?;
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(format!(
                "--estimate {}={}:{}: bounds must both be finite", name, lo, hi));
        }
        if lo >= hi {
            return Err(format!(
                "--estimate {}={}:{}: lower bound must be < upper bound",
                name, lo, hi));
        }
        Ok(Self { name: name.to_string(), lo, hi })
    }
}

// ─── ListDuration ─────────────────────────────────────────────────────────────

/// `--since 1h` / `30m` / `2d`  (for `camdl list`)
#[derive(Clone, Debug)]
pub struct ListDuration(pub std::time::Duration);

impl FromStr for ListDuration {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (num_str, unit) = s
            .find(|c: char| c.is_alphabetic())
            .map(|i| s.split_at(i))
            .ok_or_else(|| format!("expected duration like 1h/30m/2d, got '{}'", s))?;
        let n = num_str.parse::<u64>()
            .map_err(|_| format!("'{}' is not a valid number in duration '{}'", num_str, s))?;
        let secs = match unit {
            "s"           => n,
            "m"           => n * 60,
            "h"           => n * 3600,
            "d"           => n * 86_400,
            "w"           => n * 604_800,
            other         => return Err(format!("unknown duration unit '{}' in '{}' (use s/m/h/d/w)", other, s)),
        };
        Ok(Self(std::time::Duration::from_secs(secs)))
    }
}

#[cfg(test)]
mod sweep_tests {
    use super::*;

    fn parse(s: &str) -> SweepSpec { s.parse::<SweepSpec>().expect(s) }

    #[test]
    fn list_form_parses_unchanged() {
        let s = parse("beta=0.1,0.2,0.3");
        assert_eq!(s.name, "beta");
        assert_eq!(s.grid.expand(), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn lin_endpoints_inclusive() {
        let g = parse("beta=lin(1.0,4.0,4)").grid.expand();
        assert_eq!(g, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn lin_eleven_point_grid_matches_user_example() {
        // The motivating call: --sweep beta=1.0,1.3,...,4.0 (11 vals).
        let g = parse("beta=lin(1.0,4.0,11)").grid.expand();
        assert_eq!(g.len(), 11);
        assert!((g[0] - 1.0).abs() < 1e-12);
        assert!((g[10] - 4.0).abs() < 1e-12);
        assert!((g[1] - 1.3).abs() < 1e-12);
    }

    #[test]
    fn log10_endpoints_inclusive_and_powers_of_ten() {
        let g = parse("k=log10(0.001,1000,7)").grid.expand();
        assert_eq!(g.len(), 7);
        for (i, expected) in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0].iter().enumerate() {
            assert!((g[i] - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "g[{}] = {} expected {}", i, g[i], expected);
        }
    }

    #[test]
    fn log10_value_form_not_exponent_form() {
        // The whole point of value form: log10(0.05, 5.0, ...) means
        // "from 0.05 to 5.0", NOT "from 10^0.05 to 10^5.0".
        let g = parse("beta=log10(0.05,5.0,3)").grid.expand();
        assert!((g[0] - 0.05).abs() < 1e-12, "g[0] = {}", g[0]);
        assert!((g[2] - 5.0).abs() < 1e-12,  "g[2] = {}", g[2]);
        // Geometric mean of 0.05 and 5.0 is 0.5.
        assert!((g[1] - 0.5).abs() < 1e-12,  "g[1] = {}", g[1]);
    }

    #[test]
    fn lin_rejects_n_below_two() {
        let e = "beta=lin(1.0,4.0,1)".parse::<SweepSpec>().unwrap_err();
        assert!(e.contains("n >= 2"), "got: {}", e);
    }

    #[test]
    fn lin_rejects_min_ge_max() {
        let e = "beta=lin(4.0,1.0,5)".parse::<SweepSpec>().unwrap_err();
        assert!(e.contains("min < max"), "got: {}", e);
    }

    #[test]
    fn log10_rejects_nonpositive_min() {
        let e = "beta=log10(0.0,10.0,5)".parse::<SweepSpec>().unwrap_err();
        assert!(e.contains("min > 0"), "got: {}", e);
        let e = "beta=log10(-1.0,10.0,5)".parse::<SweepSpec>().unwrap_err();
        assert!(e.contains("min > 0"), "got: {}", e);
    }

    #[test]
    fn log_alone_gets_helpful_error_pointing_at_dsl_collision() {
        // A user who reaches for `log(...)` should be redirected to
        // `log10(...)` rather than seeing a generic float-parse error.
        let e = "beta=log(0.1,10,5)".parse::<SweepSpec>().unwrap_err();
        assert!(e.contains("log10"), "want hint to log10, got: {}", e);
        assert!(e.contains("natural log") || e.contains("DSL"),
            "want DSL collision context, got: {}", e);
    }

    #[test]
    fn whitespace_inside_call_is_tolerated() {
        let g = parse("beta=lin( 1.0 , 4.0 , 4 )").grid.expand();
        assert_eq!(g, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn empty_name_rejected() {
        assert!("=1,2,3".parse::<SweepSpec>().is_err());
    }

    #[test]
    fn missing_equals_rejected() {
        assert!("betalin(1,4,4)".parse::<SweepSpec>().is_err());
    }
}

#[cfg(test)]
mod data_spec_tests {
    use super::*;

    #[test]
    fn data_spec_parses_single_path() {
        let d: DataSpec = "cases.tsv".parse().unwrap();
        match d {
            DataSpec::Single(p) => assert_eq!(p, PathBuf::from("cases.tsv")),
            DataSpec::Named { .. } => panic!("expected Single, got Named"),
        }
    }

    #[test]
    fn data_spec_parses_named_form() {
        let d: DataSpec = "cases=path/to/cases.tsv".parse().unwrap();
        match d {
            DataSpec::Named { name, path } => {
                assert_eq!(name, "cases");
                assert_eq!(path, PathBuf::from("path/to/cases.tsv"));
            }
            DataSpec::Single(_) => panic!("expected Named, got Single"),
        }
    }

    #[test]
    fn data_spec_named_empty_name_treated_as_path() {
        // `--data =foo.tsv` has empty NAME; we fall through to Single
        // with the literal `=foo.tsv` path. This is the documented
        // behaviour — empty-NAME pairs are not a NAME=PATH binding.
        let d: DataSpec = "=foo.tsv".parse().unwrap();
        match d {
            DataSpec::Single(p) => assert_eq!(p, PathBuf::from("=foo.tsv")),
            DataSpec::Named { .. } => panic!(
                "empty NAME must NOT be treated as Named binding"),
        }
    }

    #[test]
    fn data_spec_path_with_equals_in_filename_is_named_if_lhs_nonempty() {
        // `cases=2026=foo.tsv` parses as Named { name: "cases", path:
        // "2026=foo.tsv" }. `split_once` takes the *first* `=`.
        // Documented; users with `=` in filenames must rename or use
        // the Single form with NAME-free paths.
        let d: DataSpec = "cases=2026=foo.tsv".parse().unwrap();
        match d {
            DataSpec::Named { name, path } => {
                assert_eq!(name, "cases");
                assert_eq!(path, PathBuf::from("2026=foo.tsv"));
            }
            _ => panic!("expected Named"),
        }
    }

}
