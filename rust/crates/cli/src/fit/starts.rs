//! Where a method's chains begin: the `starts` key of `[method]`.
//!
//! The top-level split is the fact a reader of `fit summary` needs: are the
//! starts a *distribution* (one independent draw per chain) or a *point*
//! (every chain at one vector)? R̂ compares between-chain disagreement to
//! within-chain variance, so chains that begin at one point begin in perfect
//! agreement and an R̂ near 1 is guaranteed by construction (Gelman, Vehtari,
//! McElreath et al. 2026, §30.5). The type makes that fact a variant rather
//! than a property one has to know per rule, so every consumer that reports a
//! between-chain statistic can ask one question of it.
//!
//! Wire form — a bare string for the parameterless rules, a one-key inline
//! table for the sourced ones:
//!
//! ```toml
//! starts = "from_prior"                   # one draw per chain from the priors
//! starts = { from_posterior = "@base" }   # one posterior row of @base per chain
//! starts = { from_posterior = "@base", chains = [0, 2] }  # rows of chains 0 and 2 only
//! starts = { from_mle = "@mle" }          # every chain at @mle's estimate
//! starts = { from_params = "theta.toml" } # every chain at a hand-written point
//! ```
//!
//! The CLI spells the same rules as `--starts from_prior` and
//! `--starts from_posterior=@base` — a name, or a name and a source — and the
//! chain subset as `--starts-chains 0,2`.
//!
//! When the key is absent the rule is resolved against the problem: `from_prior`
//! when every estimated parameter declares a prior, `uniform_unconstrained`
//! otherwise ([`crate::fit::config_v2::Method::resolve_starts`]).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::chain_selection::{ChainSelection, Polarity};

/// A fit reference or a file, as written: `@label`, a fit-id hash prefix, a
/// run directory, a `fit.toml` path, or — for `from_posterior` — a draws TSV.
/// Resolved where it is consumed (`crate::fit::chain_starts`), never here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handle(pub String);

/// The CLI spelling of a `from_posterior` chain subset, `--starts-chains
/// 0,1,2,4` (gh#1056): a [`ChainSelection`] that keeps the named chains.
///
/// The reason to name chains is a source diagnosed as multimodal: some chains
/// sit in a mode with negligible posterior weight, and the next run should
/// start from the ones that agree. The subset is judged by its own R̂ when the
/// source is resolved (`crate::fit::chain_starts::resolve_starts`).
pub fn parse_starts_chains(s: &str) -> Result<ChainSelection, String> {
    ChainSelection::parse(s, Polarity::Keep, "--starts-chains")
}

/// Apply `--starts-chains` to the starts rule in force, `rule` (from
/// `--starts`, or a fit config's `starts`): only a `from_posterior` rule takes
/// a chain subset. `run` names what has no rule when `rule` is absent
/// (`"fit"`, `"profile"`). `fit run` and `profile` both qualify their rule
/// here, so the flag means one thing on both.
pub fn apply_starts_chains(
    rule: Option<ChainStarts>,
    subset: &ChainSelection,
    run: &str,
) -> Result<ChainStarts, String> {
    let rule = rule.ok_or_else(|| format!(
        "--starts-chains {} restricts a `from_posterior` rule, and this {run} names no \
         starts rule; add `--starts from_posterior=@handle`",
        subset.ids_csv()))?;
    rule.with_chains(subset.clone()).map_err(|e| format!("--starts-chains: {e}"))
}

impl std::fmt::Display for Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a method's chains begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainStarts {
    /// One independent draw per chain. The between-chain comparison R̂ makes
    /// is a comparison.
    Spread(Spread),
    /// Every chain at one vector. R̂ is not assessable; `fit summary` says so.
    Point(Point),
}

/// The spread rules: each chain gets its own point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spread {
    /// Stan's initialization: an i.i.d. draw on the unconstrained scale
    /// (radius 2), mapped into the bounds. Boundary-avoiding and
    /// scale-invariant; the default when a parameter lacks a prior.
    UniformUnconstrained,
    /// Latin-hypercube stratified within the bounds, transform-aware.
    Lhs,
    /// Chain 1 at the declared start; chains 2..N uniform within the bounds.
    Uniform,
    /// One draw per chain from each parameter's prior. The default when every
    /// estimated parameter declares one.
    FromPrior,
    /// One row per chain, drawn uniformly with replacement from a posterior
    /// cloud: a fit handle whose leaf wrote `draws.tsv`, or a draws TSV.
    /// `chains` restricts the cloud to those source chains' rows (gh#1056).
    FromPosterior { source: Handle, chains: Option<ChainSelection> },
}

/// The point rules: every chain at one vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Point {
    /// `[estimate].start`, else the model's declared value. Wire `"single"`.
    Declared,
    /// The point estimate of a stored fit, read from its `fit_state.toml`.
    /// The escape hatch that replaced in-file chaining.
    FromMle { source: Handle },
    /// A hand-written flat params TOML (top-level `name = value` lines).
    FromParams { path: PathBuf },
}

/// The two-way fact `fit_state.toml` records beside the rule's tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainStartsKind {
    Spread,
    Point,
}

impl ChainStartsKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChainStartsKind::Spread => "spread",
            ChainStartsKind::Point => "point",
        }
    }
}

/// The bare spellings, for error messages.
pub const BARE_RULES: &str = "uniform_unconstrained, lhs, uniform, from_prior, single";
/// The sourced spellings, for error messages.
pub const SOURCED_RULES: &str =
    "{ from_posterior = \"@handle\" } (optionally with `chains = [..]`), { from_mle = \"@handle\" }, \
     { from_params = \"params.toml\" }";

impl ChainStarts {
    /// The default when the problem gives no reason to prefer the prior.
    pub fn uniform_unconstrained() -> Self {
        ChainStarts::Spread(Spread::UniformUnconstrained)
    }

    pub fn from_prior() -> Self {
        ChainStarts::Spread(Spread::FromPrior)
    }

    pub fn kind(&self) -> ChainStartsKind {
        match self {
            ChainStarts::Spread(_) => ChainStartsKind::Spread,
            ChainStarts::Point(_) => ChainStartsKind::Point,
        }
    }

    /// Does this rule put every chain at the same point?
    pub fn is_point(&self) -> bool {
        matches!(self, ChainStarts::Point(_))
    }

    /// The rule's tag: the bare spelling, or the sourced rule's key. One
    /// spelling per rule across `fit_state.toml`, `chain_starts.tsv` and
    /// `run.json` (gh#873).
    pub fn tag(&self) -> &'static str {
        match self {
            ChainStarts::Spread(Spread::UniformUnconstrained) => "uniform_unconstrained",
            ChainStarts::Spread(Spread::Lhs) => "lhs",
            ChainStarts::Spread(Spread::Uniform) => "uniform",
            ChainStarts::Spread(Spread::FromPrior) => "from_prior",
            ChainStarts::Spread(Spread::FromPosterior { .. }) => "from_posterior",
            ChainStarts::Point(Point::Declared) => "single",
            ChainStarts::Point(Point::FromMle { .. }) => "from_mle",
            ChainStarts::Point(Point::FromParams { .. }) => "from_params",
        }
    }

    /// The rule as a run's identity hashes it (gh#896). A sourced rule is
    /// its kind alone (`"from_mle"`): the source it read is identity through
    /// the run's `deps` — the source run's id and the consumed file's digest —
    /// so `@scout`, the scout's leaf directory and a prefix of its run id key
    /// one run. The handle as typed is recorded, unhashed, in run.json
    /// `inputs.starts_source`. A parameterless rule hashes as it serializes.
    ///
    /// A chain subset is NOT carried by the deps — the dep is the whole
    /// `draws.tsv`'s digest — so it is hashed here: `from_posterior` with
    /// `chains` keys as `{"from_posterior": {"chains": [..]}}`, distinct from
    /// the whole-cloud rule and from every other subset (gh#1056). Without
    /// `chains` the value is the bare tag, byte-identical to before. The list
    /// is the kept chain ids as plain integers, built here rather than
    /// serialized from the selection, so a change to [`ChainSelection`]'s
    /// representation cannot re-key a run.
    pub fn identity_value(&self) -> serde_json::Value {
        if let Some(chains) = self.chains() {
            return serde_json::json!({ self.tag(): { "chains": kept_ids(chains) } });
        }
        match self.source() {
            Some(_) => serde_json::Value::String(self.tag().to_string()),
            None => serde_json::to_value(self).unwrap_or(serde_json::Value::Null),
        }
    }

    /// The source chains a `from_posterior` rule is restricted to, if any.
    pub fn chains(&self) -> Option<&ChainSelection> {
        match self {
            ChainStarts::Spread(Spread::FromPosterior { chains, .. }) => chains.as_ref(),
            _ => None,
        }
    }

    /// Restrict the rule to `chains` of its source: the `--starts-chains`
    /// flag. Only `from_posterior` draws per-chain rows, so any other rule
    /// refuses, naming what it is. The selection names the chains to start
    /// from, so it must keep them.
    pub fn with_chains(self, chains: ChainSelection) -> Result<Self, String> {
        if chains.polarity() != Polarity::Keep {
            return Err(format!(
                "a starts chain subset names the chains to start from; got a selection \
                 that drops chains {}", chains.ids_csv()));
        }
        match self {
            ChainStarts::Spread(Spread::FromPosterior { source, .. }) => {
                Ok(ChainStarts::Spread(Spread::FromPosterior { source, chains: Some(chains) }))
            }
            other => Err(format!(
                "chain subset `{}` given, but the starts rule is `{}`: only `from_posterior` \
                 draws rows from a source's chains. Write `--starts from_posterior=@handle` \
                 (or `starts = {{ from_posterior = \"@handle\", chains = [..] }}`)",
                chains.ids_csv(), other.tag()
            )),
        }
    }

    /// The handle or path a sourced rule reads, as written; `None` for the
    /// parameterless rules.
    pub fn source(&self) -> Option<String> {
        match self {
            ChainStarts::Spread(Spread::FromPosterior { source, .. })
            | ChainStarts::Point(Point::FromMle { source }) => Some(source.0.clone()),
            ChainStarts::Point(Point::FromParams { path }) => {
                Some(path.to_string_lossy().into_owned())
            }
            _ => None,
        }
    }

    /// The rule as a reader of `fit.toml` would write it: the tag, then the
    /// source when there is one (`from_mle @mle`).
    pub fn spelled(&self) -> String {
        match (self.source(), self.chains()) {
            (Some(src), Some(ch)) => format!("{} {} chains {}", self.tag(), src, ch.ids_csv()),
            (Some(src), None) => format!("{} {}", self.tag(), src),
            (None, _) => self.tag().to_string(),
        }
    }

    /// One clause for `fit summary`'s header, saying which of the two things
    /// the chains were given.
    pub fn describe(&self) -> String {
        match self.kind() {
            ChainStartsKind::Spread => {
                format!("{} (one independent draw per chain)", self.spelled())
            }
            ChainStartsKind::Point => format!("{} (every chain at one point)", self.spelled()),
        }
    }

    /// Does this rule discard the base point — and with it `[estimate].start`?
    ///
    /// gh#506: `start` is load-bearing under some rules and inert under
    /// others. Declaring a start the rule then ignores is not an error (the
    /// spread rules ignore it on purpose), but it is a silent no-op the user
    /// who wrote the value deserves to hear about. `n_chains` matters because
    /// the spread rules that draw within the bounds fall back to the base
    /// point at one chain: there is nothing to spread.
    ///
    /// Per rule, not per parameter: a name missing from a `from_mle` /
    /// `from_params` source falls back to bounds-uniform (or to the base point
    /// when the model declares no range), so for *that* parameter the declared
    /// `start` is still load-bearing. The note this drives is therefore
    /// conservative — it can say "unused" about a start that one parameter out
    /// of several still used.
    pub fn ignores_base_point(&self, n_chains: usize) -> bool {
        match self {
            ChainStarts::Point(Point::Declared) => false,
            ChainStarts::Spread(Spread::Uniform) => false,
            ChainStarts::Spread(Spread::Lhs) | ChainStarts::Spread(Spread::UniformUnconstrained) => {
                n_chains >= 2
            }
            ChainStarts::Spread(Spread::FromPrior)
            | ChainStarts::Spread(Spread::FromPosterior { .. })
            | ChainStarts::Point(Point::FromMle { .. })
            | ChainStarts::Point(Point::FromParams { .. }) => true,
        }
    }

    /// Parse one wire spelling: a bare rule name, or a sourced rule with its
    /// source. Shared by the TOML deserializer (bare string / one-key table)
    /// and the CLI (`name` / `name=source`).
    fn from_parts(name: &str, source: Option<&str>) -> Result<Self, String> {
        let bare = |rule: ChainStarts| match source {
            None => Ok(rule),
            Some(src) => Err(format!(
                "`{name}` takes no source (got `{src}`); write `starts = \"{name}\"`"
            )),
        };
        let sourced = |what: &str| -> Result<String, String> {
            match source {
                Some(src) if !src.trim().is_empty() => Ok(src.trim().to_string()),
                _ => Err(format!(
                    "`{name}` needs {what}: write `starts = {{ {name} = \"…\" }}` in \
                     fit.toml or `--starts {name}=…` on the command line"
                )),
            }
        };
        match name {
            "uniform_unconstrained" | "uniform-unconstrained" => {
                bare(ChainStarts::Spread(Spread::UniformUnconstrained))
            }
            "lhs" => bare(ChainStarts::Spread(Spread::Lhs)),
            "uniform" => bare(ChainStarts::Spread(Spread::Uniform)),
            "from_prior" | "from-prior" => bare(ChainStarts::Spread(Spread::FromPrior)),
            "single" => bare(ChainStarts::Point(Point::Declared)),
            "from_posterior" | "from-posterior" => {
                let src = sourced("a fit handle or a draws TSV")?;
                Ok(ChainStarts::Spread(Spread::FromPosterior { source: Handle(src), chains: None }))
            }
            "from_mle" | "from-mle" => {
                let src = sourced("a fit handle")?;
                Ok(ChainStarts::Point(Point::FromMle { source: Handle(src) }))
            }
            "from_params" | "from-params" => {
                let src = sourced("a flat params TOML path")?;
                Ok(ChainStarts::Point(Point::FromParams { path: PathBuf::from(src) }))
            }
            "survey_top_k" => Err(
                "`survey_top_k` was removed: a survey landscape is not a posterior. Run a \
                 short fit and start from its cloud with `starts = { from_posterior = \
                 \"@handle\" }`, or use `from_prior`."
                    .to_string(),
            ),
            // A bare number was `camdl profile --starts <N>` before gh#889
            // gave that verb this same rule grammar. Say which flag now
            // carries the count rather than listing rules at someone who
            // meant a number.
            other if other.parse::<u32>().is_ok() => Err(format!(
                "`starts` is the rule the chains begin from, not how many there are, \
                 and `{other}` is a number. On `camdl profile` the count of \
                 independent starts per grid point is `--n-starts {other}`; on \
                 `camdl fit run` the number of chains is `[method] chains`. The rules \
                 are {BARE_RULES}, or a sourced rule {SOURCED_RULES}."
            )),
            other => Err(format!(
                "unknown starts rule `{other}`; expected one of {BARE_RULES}, or a sourced \
                 rule {SOURCED_RULES}"
            )),
        }
    }
}

impl std::fmt::Display for ChainStarts {
    /// The tag alone — what `chain_starts.tsv`'s `source` column and
    /// `fit_state.toml`'s `chain_init_source` carry.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}

impl std::str::FromStr for ChainStarts {
    type Err = String;

    /// The CLI grammar: `name`, or `name=source`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        match s.split_once('=') {
            Some((name, source)) => ChainStarts::from_parts(name.trim(), Some(source)),
            None => ChainStarts::from_parts(s, None),
        }
    }
}

/// The two shapes the key takes on the wire.
#[derive(Deserialize)]
#[serde(untagged)]
enum Wire {
    Bare(String),
    Table(BTreeMap<String, WireValue>),
}

/// A table entry: a rule's source (a string), or `from_posterior`'s
/// `chains` (a list of chain ids).
#[derive(Deserialize)]
#[serde(untagged)]
enum WireValue {
    Source(String),
    Chains(Vec<i64>),
}

impl<'de> Deserialize<'de> for ChainStarts {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        match Wire::deserialize(de).map_err(|_| {
            D::Error::custom(format!(
                "`starts` is a bare rule name ({BARE_RULES}) or a one-key table \
                 ({SOURCED_RULES})"
            ))
        })? {
            Wire::Bare(name) => ChainStarts::from_parts(&name, None).map_err(D::Error::custom),
            Wire::Table(mut map) => {
                // `chains` qualifies `from_posterior`; every other key names a rule.
                let chains = map.remove("chains");
                if map.len() != 1 {
                    return Err(D::Error::custom(format!(
                        "`starts` names one sourced rule, got {} keys ({}); expected one of \
                         {SOURCED_RULES}",
                        map.len(),
                        map.keys().cloned().collect::<Vec<_>>().join(", ")
                    )));
                }
                let (name, source) = map.into_iter().next().expect("one entry");
                let source = match source {
                    WireValue::Source(s) => s,
                    WireValue::Chains(_) => return Err(D::Error::custom(format!(
                        "`starts.{name}` is a source (a fit handle or a file path), not a list"
                    ))),
                };
                let rule = ChainStarts::from_parts(&name, Some(&source)).map_err(D::Error::custom)?;
                match chains {
                    None => Ok(rule),
                    Some(WireValue::Chains(ids)) => {
                        let ids = ids
                            .into_iter()
                            .map(|i| usize::try_from(i).map_err(|_| format!(
                                "`chains` lists {i}: chain ids are counted from 0")))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(D::Error::custom)?;
                        let subset = ChainSelection::from_ids(ids, Polarity::Keep, "`chains`")
                            .map_err(D::Error::custom)?;
                        rule.with_chains(subset).map_err(D::Error::custom)
                    }
                    Some(WireValue::Source(s)) => Err(D::Error::custom(format!(
                        "`chains` is a list of chain ids (`chains = [0, 1]`), got the string `{s}`"
                    ))),
                }
            }
        }
    }
}

impl Serialize for ChainStarts {
    /// The same shape a user writes, so the identity payload's bytes match a
    /// hand-written equivalent: a bare string, or a one-key map.
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self.source() {
            None => ser.serialize_str(self.tag()),
            Some(src) => {
                use serde::ser::SerializeMap;
                let chains = self.chains();
                let mut m = ser.serialize_map(Some(1 + usize::from(chains.is_some())))?;
                m.serialize_entry(self.tag(), &src)?;
                if let Some(ch) = chains {
                    m.serialize_entry("chains", &kept_ids(ch))?;
                }
                m.end()
            }
        }
    }
}

/// The chain ids a start-side selection keeps, as plain integers: the shape
/// `fit.toml` spells and the identity value hashes.
pub fn kept_ids(chains: &ChainSelection) -> Vec<usize> {
    chains.ids().iter().map(|c| c.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keep(ids: Vec<usize>) -> ChainSelection {
        ChainSelection::from_ids(ids, Polarity::Keep, "`chains`").unwrap()
    }

    #[derive(Deserialize, Serialize)]
    struct Doc {
        starts: ChainStarts,
    }

    fn parse(s: &str) -> Result<ChainStarts, String> {
        toml::from_str::<Doc>(s).map(|d| d.starts).map_err(|e| e.to_string())
    }

    #[test]
    fn bare_rules_parse_to_their_variant() {
        assert_eq!(
            parse("starts = \"uniform_unconstrained\"").unwrap(),
            ChainStarts::Spread(Spread::UniformUnconstrained)
        );
        assert_eq!(parse("starts = \"lhs\"").unwrap(), ChainStarts::Spread(Spread::Lhs));
        assert_eq!(parse("starts = \"uniform\"").unwrap(), ChainStarts::Spread(Spread::Uniform));
        assert_eq!(parse("starts = \"from_prior\"").unwrap(), ChainStarts::Spread(Spread::FromPrior));
        assert_eq!(parse("starts = \"single\"").unwrap(), ChainStarts::Point(Point::Declared));
    }

    #[test]
    fn sourced_rules_parse_from_a_one_key_table() {
        assert_eq!(
            parse("starts = { from_posterior = \"@base\" }").unwrap(),
            ChainStarts::Spread(Spread::FromPosterior { source: Handle("@base".into()), chains: None })
        );
        assert_eq!(
            parse("starts = { from_mle = \"@mle\" }").unwrap(),
            ChainStarts::Point(Point::FromMle { source: Handle("@mle".into()) })
        );
        assert_eq!(
            parse("starts = { from_params = \"theta.toml\" }").unwrap(),
            ChainStarts::Point(Point::FromParams { path: PathBuf::from("theta.toml") })
        );
    }

    #[test]
    fn a_sourced_rule_written_bare_says_what_it_needs() {
        let err = parse("starts = \"from_mle\"").unwrap_err();
        assert!(err.contains("needs a fit handle"), "{err}");
        assert!(err.contains("{ from_mle = "), "{err}");
    }

    #[test]
    fn a_bare_rule_with_a_source_is_refused() {
        let err = parse("starts = { lhs = \"x\" }").unwrap_err();
        assert!(err.contains("takes no source"), "{err}");
    }

    #[test]
    fn survey_top_k_is_named_as_removed() {
        let err = parse("starts = \"survey_top_k\"").unwrap_err();
        assert!(err.contains("removed"), "{err}");
        assert!(err.contains("from_posterior"), "{err}");
    }

    #[test]
    fn an_unknown_rule_lists_the_spellings() {
        let err = parse("starts = \"lhss\"").unwrap_err();
        assert!(err.contains("unknown starts rule `lhss`"), "{err}");
        assert!(err.contains("uniform_unconstrained"), "{err}");
        assert!(err.contains("from_params"), "{err}");
    }

    #[test]
    fn a_two_key_table_is_refused() {
        let err = parse("starts = { from_mle = \"@a\", from_posterior = \"@b\" }").unwrap_err();
        assert!(err.contains("one sourced rule"), "{err}");
    }

    /// gh#1056: `chains` qualifies `from_posterior`, is normalized to sorted
    /// order, and round-trips through the wire form a user writes.
    #[test]
    fn from_posterior_takes_a_chain_subset() {
        let r = parse("starts = { from_posterior = \"@base\", chains = [4, 0, 2] }").unwrap();
        assert_eq!(r, ChainStarts::Spread(Spread::FromPosterior {
            source: Handle("@base".into()),
            chains: Some(keep(vec![0, 2, 4])),
        }));
        assert_eq!(kept_ids(r.chains().unwrap()), vec![0, 2, 4]);
        assert_eq!(r.spelled(), "from_posterior @base chains 0,2,4");
        let back = toml::to_string(&Doc { starts: r.clone() }).unwrap();
        assert_eq!(parse(&back).unwrap(), r, "{back}");
    }

    #[test]
    fn a_chain_subset_is_refused_where_it_cannot_apply() {
        for (toml, want) in [
            ("starts = { from_mle = \"@m\", chains = [0] }", "only `from_posterior`"),
            ("starts = { from_posterior = \"@b\", chains = [] }", "names no chain"),
            ("starts = { from_posterior = \"@b\", chains = [1, 1] }", "chain 1 twice"),
            ("starts = { from_posterior = \"@b\", chains = [-1] }", "counted from 0"),
            ("starts = { from_posterior = \"@b\", chains = \"0,1\" }", "list of chain ids"),
            ("starts = { chains = [0, 1] }", "one sourced rule"),
        ] {
            let err = parse(toml).expect_err(toml);
            assert!(err.contains(want), "{toml}: {err}");
        }
    }

    #[test]
    fn the_cli_chain_list_parses_and_with_chains_refuses_other_rules() {
        assert_eq!(kept_ids(&parse_starts_chains("0, 2,1").unwrap()), vec![0, 1, 2]);
        assert!(parse_starts_chains("0,x").unwrap_err().contains("not a chain id"));
        assert!(parse_starts_chains("").unwrap_err().contains("names no chain"));
        // An empty token was skipped before the selectors were one type; it is
        // now refused, as `--exclude-chains` always refused it.
        let e = parse_starts_chains("0,,1").unwrap_err();
        assert!(e.contains("--starts-chains") && e.contains("empty chain id"), "{e}");
        assert!(parse_starts_chains("1,1").unwrap_err().contains("chain 1 twice"));
        let sub = keep(vec![0]);
        let err = ChainStarts::from_prior().with_chains(sub.clone()).unwrap_err();
        assert!(err.contains("from_prior") && err.contains("only `from_posterior`"), "{err}");
        let post = "from_posterior=@b".parse::<ChainStarts>().unwrap().with_chains(sub).unwrap();
        assert_eq!(kept_ids(post.chains().unwrap()), vec![0]);
        // A dropping selection cannot qualify a start rule.
        let drop = ChainSelection::parse_exclude("0").unwrap();
        let err = "from_posterior=@b".parse::<ChainStarts>().unwrap().with_chains(drop).unwrap_err();
        assert!(err.contains("drops chains 0"), "{err}");
    }

    /// The subset is identity (the dep is the whole `draws.tsv`, so nothing
    /// else would key it); the whole-cloud rule keeps its old value.
    #[test]
    fn a_chain_subset_enters_the_identity_value_and_only_then() {
        let whole = "from_posterior=@b".parse::<ChainStarts>().unwrap();
        assert_eq!(whole.identity_value(), serde_json::json!("from_posterior"));
        let a = whole.clone().with_chains(keep(vec![2, 0])).unwrap();
        let b = whole.clone().with_chains(keep(vec![0, 2])).unwrap();
        let c = whole.clone().with_chains(keep(vec![0, 1])).unwrap();
        assert_eq!(a.identity_value(), serde_json::json!({"from_posterior": {"chains": [0, 2]}}));
        assert_eq!(a.identity_value(), b.identity_value(), "order is not meaningful");
        assert_ne!(a.identity_value(), c.identity_value());
        assert_ne!(a.identity_value(), whole.identity_value());
    }

    #[test]
    fn the_cli_grammar_is_name_or_name_equals_source() {
        assert_eq!("from_prior".parse::<ChainStarts>().unwrap(), ChainStarts::from_prior());
        assert_eq!(
            "from_posterior=@base".parse::<ChainStarts>().unwrap(),
            ChainStarts::Spread(Spread::FromPosterior { source: Handle("@base".into()), chains: None })
        );
        assert_eq!(
            "from_mle=results/fits/x-1234/pgas-abcd/seed_1-ef01".parse::<ChainStarts>().unwrap(),
            ChainStarts::Point(Point::FromMle {
                source: Handle("results/fits/x-1234/pgas-abcd/seed_1-ef01".into())
            })
        );
        let err = "from_mle".parse::<ChainStarts>().unwrap_err();
        assert!(err.contains("--starts from_mle=…"), "{err}");
    }

    #[test]
    fn serialization_round_trips_in_the_written_shape() {
        for src in [
            "starts = \"lhs\"",
            "starts = \"from_prior\"",
            "starts = { from_posterior = \"@base\" }",
            "starts = { from_mle = \"@mle\" }",
            "starts = { from_params = \"p.toml\" }",
        ] {
            let parsed = parse(src).unwrap();
            let json = serde_json::to_value(&parsed).unwrap();
            let back: ChainStarts = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(back, parsed, "{src} -> {json}");
        }
        assert_eq!(
            serde_json::to_value(ChainStarts::from_prior()).unwrap(),
            serde_json::json!("from_prior")
        );
        assert_eq!(
            serde_json::to_value(ChainStarts::Point(Point::FromMle {
                source: Handle("@mle".into())
            }))
            .unwrap(),
            serde_json::json!({ "from_mle": "@mle" })
        );
    }

    #[test]
    fn point_and_spread_are_told_apart() {
        assert!(ChainStarts::Point(Point::Declared).is_point());
        assert!(ChainStarts::Point(Point::FromParams { path: "p".into() }).is_point());
        assert!(!ChainStarts::Spread(Spread::Uniform).is_point());
        assert!(!ChainStarts::Spread(Spread::FromPosterior { source: Handle("@a".into()), chains: None }).is_point());
        assert_eq!(ChainStarts::from_prior().kind().as_str(), "spread");
        assert_eq!(ChainStarts::Point(Point::Declared).kind().as_str(), "point");
    }

    #[test]
    fn the_tag_is_one_spelling_per_rule_and_has_no_hyphen() {
        for rule in [
            ChainStarts::Spread(Spread::UniformUnconstrained),
            ChainStarts::Spread(Spread::Lhs),
            ChainStarts::Spread(Spread::Uniform),
            ChainStarts::Spread(Spread::FromPrior),
            ChainStarts::Spread(Spread::FromPosterior { source: Handle("@a".into()), chains: None }),
            ChainStarts::Point(Point::Declared),
            ChainStarts::Point(Point::FromMle { source: Handle("@a".into()) }),
            ChainStarts::Point(Point::FromParams { path: "p".into() }),
        ] {
            let tag = rule.tag();
            assert!(!tag.contains('-'), "{tag}");
            assert_eq!(rule.to_string(), tag);
            // The bare tag parses back to the rule's shape (with a dummy
            // source for the sourced ones).
            let back = ChainStarts::from_parts(tag, rule.source().as_deref()).unwrap();
            assert_eq!(back.tag(), tag);
        }
    }

    #[test]
    fn describe_says_spread_or_point() {
        assert_eq!(
            ChainStarts::from_prior().describe(),
            "from_prior (one independent draw per chain)"
        );
        assert_eq!(
            ChainStarts::Point(Point::FromMle { source: Handle("@mle".into()) }).describe(),
            "from_mle @mle (every chain at one point)"
        );
    }

    #[test]
    fn ignores_base_point_matches_the_rules_documented() {
        assert!(!ChainStarts::Point(Point::Declared).ignores_base_point(4));
        assert!(!ChainStarts::Spread(Spread::Uniform).ignores_base_point(4));
        assert!(ChainStarts::Spread(Spread::Lhs).ignores_base_point(2));
        assert!(!ChainStarts::Spread(Spread::Lhs).ignores_base_point(1));
        assert!(ChainStarts::uniform_unconstrained().ignores_base_point(2));
        assert!(!ChainStarts::uniform_unconstrained().ignores_base_point(1));
        assert!(ChainStarts::from_prior().ignores_base_point(1));
        assert!(ChainStarts::Point(Point::FromMle { source: Handle("@a".into()) }).ignores_base_point(1));
    }
}
