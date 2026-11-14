//! Which observation streams a fit's `[data]` block binds, and to which files.
//!
//! A model's `observations { }` block can expand one declared block into
//! several concrete streams: an indexed `cases[p in patch]` becomes
//! `cases_urban`, `cases_rural`, and several blocks can read one data source
//! through `from <label>`. Each concrete stream is a *leaf*
//! (`ObservationModel.name`); the label a data file is bound to is its
//! *source* (`ObservationModel.source`). For a plain stream they are the same
//! string.
//!
//! `[data.observations]` and `[data.holdout]` are keyed by source, and the
//! one-file shorthand `[data] file = "..."` binds that file to every source.
//! [`StreamBindings::resolve`] is the one place that reads `[data]` against a
//! model; every command that loads, hashes or checks a fit's data asks it.

use indexmap::IndexMap;

use super::config_v2::DataSpec;

/// The observation streams a fit binds: for every bound leaf, the source its
/// data is bound by, and the file bound to each source (training and, for an
/// explicit `[data.holdout]`, holdout).
#[derive(Debug, Clone)]
pub(crate) struct StreamBindings {
    /// Whether `[data]` used the one-file shorthand. Read only by
    /// [`Self::identity_keyed_training`].
    shorthand: bool,
    /// Source → training file. Under `[data.observations]`, in the order
    /// written; under the shorthand, sorted by source.
    training: IndexMap<String, String>,
    /// Source → holdout file; empty without an explicit `[data.holdout]`.
    holdout: IndexMap<String, String>,
    /// Every bound leaf and its source, sorted by leaf name.
    leaves: Vec<(String, String)>,
}

impl StreamBindings {
    /// The only constructor. Expands the one-file shorthand by source, refuses
    /// a `[data.observations]` or `[data.holdout]` key that names no source the
    /// model declares, and refuses a holdout source with no training file.
    pub(crate) fn resolve(model: &ir::Model, data: &DataSpec) -> Result<Self, String> {
        let training: IndexMap<String, String> = match &data.file {
            Some(file) => {
                let mut sources: Vec<&str> =
                    model.observations.iter().map(|o| o.source.as_str()).collect();
                sources.sort_unstable();
                sources.dedup();
                if sources.is_empty() {
                    return Err(format!(
                        "[data] file = \"{file}\" but the model declares no observation \
                         streams. Either add an `observations {{ }}` block to the \
                         .camdl file, or remove [data] from fit.toml."));
                }
                sources.into_iter().map(|s| (s.to_string(), file.clone())).collect()
            }
            None => {
                check_bound_sources(model, "[data.observations]", &data.observations)?;
                data.observations.clone()
            }
        };
        let holdout = data.holdout.clone().unwrap_or_default();
        check_bound_sources(model, "[data.holdout]", &holdout)?;
        if let Some(source) = holdout.keys().find(|s| !training.contains_key(*s)) {
            return Err(format!(
                "[data.holdout] names stream '{source}', which is not a bound \
                 observation stream (bound: {}).",
                training.keys().map(String::as_str).collect::<Vec<_>>().join(", ")));
        }
        let mut leaves: Vec<(String, String)> = model.observations.iter()
            .filter(|o| training.contains_key(&o.source))
            .map(|o| (o.name.clone(), o.source.clone()))
            .collect();
        leaves.sort();
        Ok(StreamBindings { shorthand: data.file.is_some(), training, holdout, leaves })
    }

    /// Source → training file.
    pub(crate) fn training_by_source(&self) -> &IndexMap<String, String> {
        &self.training
    }

    /// Source → holdout file; empty without an explicit `[data.holdout]`.
    pub(crate) fn holdout_by_source(&self) -> &IndexMap<String, String> {
        &self.holdout
    }

    /// The training files keyed as the run identities of `fit`, `survey`,
    /// `profile` and `pfilter --fit` have always been built over:
    /// `[data.observations]` as written (by source), and under the one-file
    /// shorthand the file once per model observation leaf. These identities
    /// fold the files' digests in key order with the keys dropped, so under
    /// the shorthand the number of leaves is part of every such `run_id`.
    ///
    /// Not a binding: nothing looks a stream up by these keys. Changing the
    /// layout re-keys every stored shorthand run, which is a deliberate,
    /// version-bumped change, never a side effect of a refactor.
    pub(crate) fn identity_keyed_training(&self) -> IndexMap<String, String> {
        if !self.shorthand {
            return self.training.clone();
        }
        self.leaves.iter()
            .map(|(leaf, source)| (leaf.clone(), self.training[source].clone()))
            .collect()
    }
}

/// Reject a data-binding key that names no observation stream `source`.
///
/// `[data.observations]`, `[data.holdout]` and `--data NAME=PATH` all key by
/// observation SOURCE (the `from <label>`, defaulting to the stream name;
/// §2.4). A key matching none of them binds a file to nothing — at best a
/// mistyped stream name, at worst a *top-level* fit.toml setting that TOML
/// scoping captured into the table by accident, which then silently does not
/// apply (a top-level key such as `ic_free` written below the
/// `[data.observations]` header is the motivating case: it looks set and is
/// not).
///
/// `origin` names the table (or flag) the keys were typed in, so the message
/// points at what the user wrote.
pub(crate) fn check_bound_sources(
    model: &ir::Model,
    origin: &str,
    bound: &IndexMap<String, String>,
) -> Result<(), String> {
    let mut sources: Vec<&str> =
        model.observations.iter().map(|o| o.source.as_str()).collect();
    sources.sort_unstable();
    sources.dedup();
    for (key, path) in bound {
        if sources.contains(&key.as_str()) {
            continue;
        }
        // A key whose value is not even a readable file is far likelier to be a
        // top-level setting swallowed by the preceding `[table]` header than a
        // mistyped stream name — and the two have different fixes, so say which
        // one the evidence points at rather than offering both.
        let hint = if std::path::Path::new(path).is_file() {
            String::new()
        } else {
            format!(
                " Its value (\"{path}\") is not a readable file either — if \
                 `{key}` was meant as a TOP-LEVEL fit.toml key, note that TOML \
                 binds every key after a `[table]` header to that table, so it \
                 must sit ABOVE the first `[table]`."
            )
        };
        return Err(format!(
            "{origin}: '{key}' is not an observation source. Keys here bind a \
             data file to an observation stream declared in the model's \
             `observations {{ }}` block.{hint} Available sources: {}",
            sources.join(", ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_patch() -> ir::Model {
        let path = format!(
            "{}/../../../ocaml/golden/sir_two_patch_long_obs.ir.json",
            env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        serde_json::from_value(v["model"].clone()).unwrap()
    }

    fn data(toml: &str) -> DataSpec {
        toml::from_str(toml).expect("fixture [data] parses")
    }

    #[test]
    fn the_shorthand_binds_by_source_and_names_every_leaf() {
        let b = StreamBindings::resolve(&two_patch(), &data("file = \"x.tsv\"")).unwrap();
        assert_eq!(b.training_by_source().iter().collect::<Vec<_>>(),
            vec![(&"cases".to_string(), &"x.tsv".to_string())]);
        assert!(b.holdout_by_source().is_empty());
        assert_eq!(b.identity_keyed_training().keys().collect::<Vec<_>>(),
            vec!["cases_rural", "cases_urban"], "the shorthand identity folds the file per leaf");
    }

    #[test]
    fn the_per_source_form_binds_what_it_names() {
        let b = StreamBindings::resolve(&two_patch(), &data(
            "[observations]\ncases = \"x.tsv\"\n[holdout]\ncases = \"h.tsv\"")).unwrap();
        assert_eq!(b.training_by_source()["cases"], "x.tsv");
        assert_eq!(b.holdout_by_source()["cases"], "h.tsv");
        assert_eq!(b.identity_keyed_training(), *b.training_by_source());
    }

    #[test]
    fn a_key_that_names_no_source_is_refused() {
        let leaf = StreamBindings::resolve(&two_patch(), &data(
            "[observations]\ncases_urban = \"x.tsv\"")).unwrap_err();
        assert!(leaf.contains("[data.observations]: 'cases_urban' is not an observation source"),
            "{leaf}");
        let holdout = StreamBindings::resolve(&two_patch(), &data(
            "file = \"x.tsv\"\n[holdout]\ndeaths = \"h.tsv\"")).unwrap_err();
        assert!(holdout.contains("[data.holdout]: 'deaths'"), "{holdout}");
    }
}
