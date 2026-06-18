//! Byte-digest, path-label, and provenance helpers shared across the CLI.
//!
//! # Not run identity (gh#241)
//!
//! Artifact run identity is built exclusively by the `runid`/`resolve` paths
//! (`resolve::resolve_trajectory`, `cas::fit_level_hash`, the `runid` factored
//! levels). NOTHING in this module participates in a `run_id`. The functions
//! here are one of three kinds:
//!
//! - **byte digests**: [`sha256_hex`], [`file_hash`] — hash some bytes.
//! - **path labels**: [`path_stem_slug`], [`slug`] — make a directory name
//!   human-readable; the content hash that follows is the authoritative key.
//! - **legacy cross-check / layout digests**: [`model_hash`],
//!   [`fit_content_hash`] — NOT artifact run identity, and NOT casual display
//!   strings either. `model_hash` is byte-compared by the `init = survey_top_k`
//!   warm-start cross-check ([`crate::fit::init::build_chain_starts_from_survey`]),
//!   and `fit_content_hash` backs the synthetic-fit `fit_dir()` on-disk path
//!   layout. Because the cross-check's two sides must agree byte-for-byte and
//!   the directory layout is persisted, these cannot be retargeted onto `runid`
//!   in isolation — migrating the coupled set is a separate future task (a
//!   gh#241 follow-up), not this module's to drop. Treat them as legacy
//!   cross-check/layout digests, never as harmless display strings.
//!
//! **DO NOT** introduce a new run-identity construction here. A field that
//! changes stored bytes is identity and must re-key through `runid`; a
//! re-encoding of the same values is presentation. The `legacy_identity_*`
//! allowlist test below pins the exact set of display call sites — a new
//! call site to [`model_hash`] / [`fit_content_hash`] fails it.

use sha2::{Sha256, Digest};

use crate::version;

/// Structural hash of the IR: every field that affects the computed trajectory.
///
/// Legacy cross-check digest (gh#241) — NOT artifact run identity, and not a
/// casual display string: it is recorded in `survey`/`fit` `run.json` `inputs`
/// and read back + byte-compared by the `init = "survey_top_k"` warm-start
/// cross-check ([`crate::fit::init`]). Both sides must produce the same string,
/// so it is not migrated to `runid` piecemeal — that's a separate future task.
///
/// Presentation-only fields (`output.format`, `simulation.time_semantics`) are
/// excluded so `--format` / date rendering stay inert; `dt`/seed are not part
/// of the model hash. serde_json's Map is backed by BTreeMap (sorted keys),
/// so serialization is deterministic.
///
/// The on-disk IR is an *envelope* — `{ ir_version, validated_by, model: {…} }`
/// — and every field below lives inside `model`. We descend into it.
/// gh#135: the previous code scanned the envelope's top level, found none of
/// these keys, fed nothing to the hasher, and returned `SHA256("")` for every
/// model. That made the sim cache blind to model structure: two different
/// models with the same params/backend/dt/seed collided to one CAS entry and
/// the second run was silently served the first model's trajectory. The `model`
/// key is absent only when the input is already a bare inner model; we fall
/// back to scanning it directly so that case still hashes, and the post-hash
/// guard catches the empty-input digest either way.
///
/// gh#147: the allowlist previously omitted the trajectory-determining
/// non-structural fields — output cadence (`output.times`), the horizon
/// (`simulation.t_start`/`t_end`), the calendar `origin`/`origin_rata_die`, and
/// `time_unit`. Two models differing only in one of those hashed *equal*. They
/// are now folded in so the structural digest tracks the full trajectory-
/// determining surface. We hash the trajectory-determining *sub-fields* of
/// `output`/`simulation` rather
/// than the whole blocks, so the presentation fields stay inert — mirroring the
/// runid path's `resolve::normalize_for_hash` invariant.
pub fn model_hash(ir_json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(ir_json)
        .expect("model_hash: invalid JSON");
    let envelope = v.as_object().expect("model_hash: expected object");
    // Descend into the `model` envelope key; tolerate a bare inner model.
    let obj = match envelope.get("model").and_then(|m| m.as_object()) {
        Some(model_obj) => model_obj,
        None => envelope,
    };

    let mut h = Sha256::new();
    let structural_keys = [
        "compartments", "transitions", "parameters", "tables",
        "time_functions", "interventions", "observations",
        "ode_equations", "initial_conditions",
        // gh#147: calendar/time-axis context. `origin`/`origin_rata_die` anchor
        // wall-clock dates and calendar-aware forcings; `time_unit` fixes the
        // meaning of the numeric time axis. All change the run.
        "origin", "origin_rata_die", "time_unit",
    ];
    for key in &structural_keys {
        if let Some(val) = obj.get(*key) {
            h.update(key.as_bytes());
            h.update(b"\x00");
            h.update(serde_json::to_string(val).unwrap().as_bytes());
            h.update(b"\x00");
        }
    }
    // gh#147: the output cadence — `output.times` (Regular{start,step,end} or
    // AtTimes[…]) — determines which rows the trajectory emits. `output.format`
    // and the `trajectory`/`observations` selection flags are presentation, so
    // we hash only `times`.
    if let Some(times) = obj.get("output").and_then(|o| o.as_object()).and_then(|o| o.get("times")) {
        h.update(b"output.times\x00");
        h.update(serde_json::to_string(times).unwrap().as_bytes());
        h.update(b"\x00");
    }
    // gh#147: the simulation horizon `t_start`/`t_end` bounds the run. `dt` and
    // the seed ride in sim_hash; `time_semantics` is presentation — so we hash
    // only the horizon bounds here.
    if let Some(sim) = obj.get("simulation").and_then(|s| s.as_object()) {
        for key in ["t_start", "t_end"] {
            if let Some(val) = sim.get(key) {
                h.update(b"simulation.");
                h.update(key.as_bytes());
                h.update(b"\x00");
                h.update(serde_json::to_string(val).unwrap().as_bytes());
                h.update(b"\x00");
            }
        }
    }
    if let Some(val) = obj.get("version") {
        h.update(b"version\x00");
        h.update(serde_json::to_string(val).unwrap().as_bytes());
    }
    let digest = hex::encode(h.finalize());
    // Defense in depth against the gh#135 failure mode recurring under a
    // future schema rename: a real model always has at least `compartments`,
    // so an empty-input digest means we hashed nothing and every model would
    // collide. Fail loudly here rather than silently serve a wrong trajectory.
    assert_ne!(
        digest, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "model_hash hashed no structural fields (SHA256 of empty) — the IR \
         envelope shape likely changed and model_hash no longer finds the \
         model's structural keys; see gh#135"
    );
    digest
}

/// Full 64-char SHA-256 of a byte slice, hex-encoded. Used where the
/// caller wants a full content hash (e.g. fit_toml_hash in the top-level
/// fit run record); the 8-char truncated form is only appropriate when
/// the hash is paired with a richer identifier.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Hash the contents of a file (first 4 bytes of SHA256, 8 hex chars).
/// Returns `None` when the file can't be read — callers use this to
/// surface `<unreadable>` in provenance records rather than failing
/// the whole run.
///
/// Shared between simulate (data-file hashing for scen_hash / run
/// metadata) and fit (data-file hashing for fit_stage_hash /
/// per-stage provenance). Was `fit::provenance::file_content_hash`
/// before the 2026-04-19 unification.
pub fn file_hash(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(hex::encode(&Sha256::digest(&bytes)[..4]))
}

/// Canonicalise a TOML document for hashing. Parses to a value, then
/// serialises back through `toml::to_string` which sorts keys and
/// strips comments + non-semantic whitespace. Purpose: cache-
/// invalidation based on semantic content, not textual form — editing
/// a comment or reformatting the file doesn't bust the cache.
///
/// Falls back to raw bytes on parse failure: if the TOML is
/// unparseable, the config is broken anyway and downstream will
/// produce a better error than "can't canonicalise your hash input."
/// We prefer to still produce a hash (for cache-staleness detection)
/// rather than refuse, since the caller handles real errors on the
/// primary `FitConfigV2::load` path.
fn canonicalise_toml(raw: &[u8]) -> Vec<u8> {
    let s = match std::str::from_utf8(raw) {
        Ok(s) => s,
        Err(_) => return raw.to_vec(), // non-UTF-8: pass through
    };
    match toml::from_str::<toml::Value>(s) {
        Ok(v) => match toml::to_string(&v) {
            Ok(canonical) => canonical.into_bytes(),
            Err(_) => raw.to_vec(),
        },
        Err(_) => raw.to_vec(),
    }
}

/// Legacy LAYOUT digest (gh#241) — NOT artifact run identity. It backs the
/// synthetic-fit `FitConfigV2::fit_dir()` on-disk path, so retargeting it onto
/// `runid` would change the directory layout; that migration is a separate
/// future task, not a free rename. (Real-data fits identify via the `runid`
/// `cas::fit_level_hash` path; only the synthetic fit *dir* keys off this.)
///
/// Content hash for a fit's *directory* (seed-independent). Keyed on
/// `(model IR, data files, canonicalised fit.toml, version)` — deliberately
/// omits seed so re-running the same fit config with different seeds lands
/// in the same `results/fits/<stem>-<hash>/` directory, with seeds
/// differentiated via the `fit_<seed>` subdirectory.
///
/// Used by `FitConfigV2::fit_dir()` to produce the content-addressable
/// suffix on the fit-directory name. The proposal's "edit your
/// fit.toml and get a new dir" invariant falls out of this: any
/// *semantic* edit to model, data, or fit.toml changes the hash;
/// seed alone doesn't, and neither do comment edits or whitespace
/// reformatting (TOML is canonicalised before hashing).
pub fn fit_content_hash(
    model_ir_bytes: &[u8],
    data_files: &mut [(String, Vec<u8>)],
    fit_toml_bytes: &[u8],
) -> String {
    let mut h = Sha256::new();
    h.update(b"model\x00");
    h.update(model_ir_bytes);
    h.update(b"\x00data\x00");
    data_files.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, bytes) in data_files.iter() {
        h.update(name.as_bytes());
        h.update(b"\x00");
        h.update(bytes);
        h.update(b"\x00");
    }
    h.update(b"fit\x00");
    h.update(canonicalise_toml(fit_toml_bytes));
    h.update(b"\x00version\x00");
    h.update(version::VERSION_SHORT.as_bytes());
    // Full 64-char hex. Directory-name truncation happens at the
    // path layer via `run_paths::fit_run_dir`, not here —
    // decoupling the storage key from the display prefix (an 8-char
    // key would risk collisions at ~65k fits).
    hex::encode(h.finalize())
}

/// Extract a directory-safe stem from a file path: basename without
/// extension(s), slugified. Used to label fit / sim output
/// directories so `ls output/fits/` shows recognisable names alongside
/// their content hashes. Multi-dot extensions (`foo.ir.json` →
/// `foo`) collapse by stripping from the first dot.
pub fn path_stem_slug(path: &str) -> Option<String> {
    let name = std::path::Path::new(path).file_name()?.to_str()?;
    let stem = name.split('.').next().filter(|s| !s.is_empty())?;
    Some(slug(stem))
}

/// Convert a scenario name to a filesystem-safe slug: lowercase, non-[a-z0-9_] → '_'.
pub fn slug(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Frozen golden hashes (regression guard) ──────────────────────────────
    //
    // These assertions lock each retained hash helper to a known byte-
    // for-byte output for a fixed input. If someone refactors the
    // hashing code and the bytes move, CI fails with a crisp diff —
    // forcing the refactor to either justify the break (and update
    // this test as a conscious decision) or preserve the hash. The
    // `model_hash` golden is doubly load-bearing: the `survey`↔`fit`
    // cross-check (gh#51) depends on both sides producing this string.

    #[test]
    fn golden_hash_model_hash() {
        // Realistic on-disk shape: an IR *envelope* whose structural
        // fields live under `model` (not at the top level). Pre-gh#135
        // this hashed to SHA256("") because the scanner looked at the
        // envelope's top level and found none of its keys.
        let ir = r#"{"ir_version":"3","validated_by":"camdlc","model":{"compartments":["S","I"],"parameters":[{"name":"beta"}]}}"#;
        assert_eq!(model_hash(ir),
            "53b7d24e97c71b0fb35e58a95d21ccd8b7178a22317e3115df5770c856d9180b");
    }

    // SHA-256 of the empty byte string — the digest a Hasher produces
    // when nothing is fed to it. gh#135: model_hash returned exactly
    // this for every model because it scanned the wrong nesting level.
    const EMPTY_SHA256: &str =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn model_hash_envelope_is_not_empty_hash() {
        // gh#135 regression: a real enveloped IR must hash its inner
        // structural fields, NOT collapse to SHA256("").
        let ir = r#"{"ir_version":"3","validated_by":"camdlc","model":{"compartments":["S","I","R"],"transitions":[{"name":"inf"}],"parameters":[{"name":"beta","value":0.3}]}}"#;
        assert_ne!(model_hash(ir), EMPTY_SHA256,
            "model_hash must hash the inner `model`, not the empty envelope top level (gh#135)");
    }

    #[test]
    fn model_hash_senses_structural_difference() {
        // gh#135 regression: two structurally different models must
        // produce different model_hash. Pre-fix both collapsed to
        // SHA256("") and collided, so the cross-check could not tell
        // them apart.
        let v1 = r#"{"ir_version":"3","validated_by":"camdlc","model":{"compartments":["S","I","R"],"transitions":[{"name":"inf","rate":"beta*S*I"}],"parameters":[{"name":"beta","value":15.0}]}}"#;
        let v2 = r#"{"ir_version":"3","validated_by":"camdlc","model":{"compartments":["S","I","R"],"transitions":[{"name":"inf","rate":"beta*S*I"}],"parameters":[{"name":"beta","value":30.0}]}}"#;
        assert_ne!(model_hash(v1), model_hash(v2),
            "models differing in a parameter value must hash differently (gh#135)");
        assert_ne!(model_hash(v1), EMPTY_SHA256);
        assert_ne!(model_hash(v2), EMPTY_SHA256);
    }

    // gh#147: model_hash's allowlist previously omitted trajectory-determining
    // fields (`output` cadence, `simulation.t_end`, calendar `origin`,
    // `time_unit`). Two models differing only in one of those hashed EQUAL.
    // These pin that each trajectory-determining field re-keys the hash, while
    // presentation-only fields (`output.format`, `simulation.time_semantics`)
    // stay inert.

    /// Minimal enveloped IR with templated output schedule, simulation block,
    /// origin, and time_unit so each test perturbs exactly one field.
    fn ir_with(times: &str, t_end: f64, origin: &str, origin_rd: &str, time_unit: &str,
               format: &str, time_semantics: &str) -> String {
        format!(
            r#"{{"ir_version":"0.15","validated_by":"camdlc","model":{{
                "compartments":["S","I","R"],
                "transitions":[{{"name":"inf","rate":"beta*S*I"}}],
                "parameters":[{{"name":"beta","value":0.3}}],
                "time_unit":"{time_unit}",
                "origin":{origin},
                "origin_rata_die":{origin_rd},
                "output":{{"times":{times},"format":"{format}","trajectory":true,"observations":false}},
                "simulation":{{"t_start":0.0,"t_end":{t_end},"time_semantics":"{time_semantics}","dt":1.0,"rng_seed":null}}
            }}}}"#
        )
    }

    #[test]
    fn model_hash_t_end_invalidates() {
        // gh#147: two models differing only in simulation.t_end must hash
        // differently — a longer horizon is a different trajectory.
        let a = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        let b = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 200.0,
                        "null", "null", "days", "tsv", "continuous");
        assert_ne!(model_hash(&a), model_hash(&b),
            "a t_end change must re-key model_hash (gh#147)");
    }

    #[test]
    fn model_hash_output_cadence_invalidates() {
        // gh#147: two models differing only in the output schedule (cadence)
        // must hash differently — they emit different rows.
        let a = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        let b = ir_with("{\"regular\":{\"start\":0.0,\"step\":7.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        assert_ne!(model_hash(&a), model_hash(&b),
            "an output-cadence change must re-key model_hash (gh#147)");
    }

    #[test]
    fn model_hash_output_at_times_invalidates() {
        // gh#147: an explicit at-times output schedule change must re-key.
        let a = ir_with("{\"at_times\":[1.0,2.0,3.0]}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        let b = ir_with("{\"at_times\":[1.0,2.0,4.0]}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        assert_ne!(model_hash(&a), model_hash(&b),
            "an at-times output schedule change must re-key model_hash (gh#147)");
    }

    #[test]
    fn model_hash_origin_invalidates() {
        // gh#147: the calendar origin maps internal t to wall-clock dates and
        // anchors calendar-aware forcings; changing it changes the run.
        let a = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "\"2020-01-01\"", "737425", "days", "tsv", "continuous");
        let b = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "\"2021-01-01\"", "737791", "days", "tsv", "continuous");
        assert_ne!(model_hash(&a), model_hash(&b),
            "a calendar-origin change must re-key model_hash (gh#147)");
    }

    #[test]
    fn model_hash_time_unit_invalidates() {
        // gh#147: time_unit sets the meaning of the time axis (days vs weeks);
        // the same numeric schedule under a different unit is a different run.
        let a = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        let b = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "weeks", "tsv", "continuous");
        assert_ne!(model_hash(&a), model_hash(&b),
            "a time_unit change must re-key model_hash (gh#147)");
    }

    #[test]
    fn model_hash_presentation_fields_are_inert() {
        // gh#147 (and the resolve.rs `presentation_fields_are_inert` invariant):
        // output.format (tsv/csv) and simulation.time_semantics never affect the
        // computed trajectory — they render views. Folding them into the key
        // would over-invalidate the cache. The base case is identical; only the
        // presentation fields differ.
        let a = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "tsv", "continuous");
        let b = ir_with("{\"regular\":{\"start\":0.0,\"step\":1.0,\"end\":100.0}}", 100.0,
                        "null", "null", "days", "csv", "discrete");
        assert_eq!(model_hash(&a), model_hash(&b),
            "output.format / simulation.time_semantics must NOT re-key model_hash");
    }

    #[test]
    fn fit_content_hash_ignores_comments_and_whitespace() {
        // Two fit.tomls that differ only in comments and whitespace
        // must produce the same fit_content_hash after canonicalisation.
        // Hardening #6 — comments are for humans, not provenance inputs.
        let model = b"ir:{}";
        let mut data1: Vec<(String, Vec<u8>)> = vec![];
        let mut data2: Vec<(String, Vec<u8>)> = vec![];
        let toml_a = b"# top comment\n[estimate]\nbeta = { bounds = [0.1, 2.0] }\n";
        let toml_b = b"[estimate]\n   beta  =  { bounds = [0.1,  2.0] }\n# trailing\n";
        let h_a = fit_content_hash(model, &mut data1, toml_a);
        let h_b = fit_content_hash(model, &mut data2, toml_b);
        assert_eq!(h_a, h_b,
            "canonicalised TOML must ignore comments + whitespace");
    }

    #[test]
    fn fit_content_hash_still_senses_real_changes() {
        // Sanity check the inverse: a semantic change (different
        // bounds) must produce a different hash.
        let model = b"ir:{}";
        let mut data1: Vec<(String, Vec<u8>)> = vec![];
        let mut data2: Vec<(String, Vec<u8>)> = vec![];
        let toml_a = b"[estimate]\nbeta = { bounds = [0.1, 2.0] }\n";
        let toml_b = b"[estimate]\nbeta = { bounds = [0.1, 3.0] }\n";
        let h_a = fit_content_hash(model, &mut data1, toml_a);
        let h_b = fit_content_hash(model, &mut data2, toml_b);
        assert_ne!(h_a, h_b,
            "changing a numeric bound must change the hash");
    }

    #[test]
    fn canonicalise_toml_falls_back_on_invalid_input() {
        // Unparseable TOML: return raw bytes. The caller produces a
        // better error than we would; we just need to not panic.
        let garbage = b"this = is = not = valid = toml";
        let out = canonicalise_toml(garbage);
        assert_eq!(out, garbage.to_vec());
    }

    #[test]
    fn golden_hash_fit_content_hash_is_full_64_chars() {
        // Lock the full-width (64-hex) output so a future truncation
        // regression fails CI instead of silently reinstating collision
        // risk (the directory prefix is truncated at the path layer, not
        // in the content hash).
        let model_ir = r#"{"compartments":["S","I"],"parameters":[]}"#;
        let fit_toml = b"[estimate]\nbeta = { bounds = [0.1, 2.0] }\n";
        let mut data: Vec<(String, Vec<u8>)> = vec![
            ("cases".into(), b"time\tvalue\n1\t5\n2\t7\n".to_vec()),
        ];
        let h = fit_content_hash(model_ir.as_bytes(), &mut data, fit_toml);
        assert_eq!(h.len(), 64,
            "fit_content_hash must return full 64-char hex (was truncated \
             to 8 pre-hardening; see hardening-proposal §ship-now/#1)");
        // Call it twice — must be deterministic.
        let mut data2: Vec<(String, Vec<u8>)> = vec![
            ("cases".into(), b"time\tvalue\n1\t5\n2\t7\n".to_vec()),
        ];
        let h2 = fit_content_hash(model_ir.as_bytes(), &mut data2, fit_toml);
        assert_eq!(h, h2);
    }

    // There is no single `run_hash(sim, scen, seed)` content hash: a run's
    // identity is the factored `runid` identity (`runid::run_id` over the
    // per-level hashes; see `crate::resolve::resolve_trajectory`).

    // ── legacy-identity call-site allowlist (gh#241) ──────────────────────────

    /// The exact set of production call sites permitted to call the retained
    /// "display / provenance / cross-check" hashers ([`model_hash`],
    /// [`fit_content_hash`]). Each entry is a `(file, hasher)` pair where
    /// `file` is relative to the cli crate's `src/`. These are the only places
    /// allowed to use these functions, and EVERY use here is display /
    /// provenance / a `survey`↔`fit` cross-check ([`crate::fit::init`]) /
    /// synthetic-fit directory keying — NOT a `runid` artifact identity.
    ///
    /// If you add a new call site, first confirm it is display/provenance (not
    /// identity — identity is built via `runid`/`resolve`), then add it here.
    /// A new use that is not listed fails this test. The intent is that
    /// `model_hash`/`fit_content_hash` cannot quietly grow a run-identity
    /// consumer: the north star is that identity is `runid`-only.
    const LEGACY_IDENTITY_ALLOWLIST: &[(&str, &str)] = &[
        // model_hash — recorded-not-keyed structural digest.
        ("survey.rs", "model_hash"),       // run.json inputs (gh#51 cross-check)
        ("profile.rs", "model_hash"),      // profile leaf content-bearing input
        ("fit/pgas.rs", "model_hash"),     // SurveyFitContext cross-check
        ("fit/pmmh.rs", "model_hash"),     // SurveyFitContext cross-check
        ("fit/mod.rs", "model_hash"),      // SurveyFitContext + mle/provenance/sidecar
        // fit_content_hash — synthetic-fit directory keying + announcement.
        ("fit/config_v2.rs", "fit_content_hash"), // fit_dir() + the method itself
        ("fit/mod.rs", "fit_content_hash"),        // synthetic-fit announcement
    ];

    /// gh#241 — pin the legacy-identity call sites to the known display set.
    ///
    /// Scans every `.rs` under the cli crate's `src/` (except this module and
    /// `args/`) for production call forms of the retained hashers, and asserts
    /// the set of `(file, hasher)` pairs equals [`LEGACY_IDENTITY_ALLOWLIST`].
    /// Test code (`#[cfg(test)]` / `#[test]`) and the definitions in this
    /// module are excluded — only real call sites count.
    #[test]
    fn legacy_identity_call_sites_match_allowlist() {
        use std::collections::BTreeSet;

        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        // The retained hashers and the call-text fragments that denote a
        // production call to each. `sim_hash`/`scen_hash`/`canonical_params`
        // are deleted; if any reappears it is dead-code-eliminated or this
        // test (and the acceptance grep) catches the stray reference.
        let hashers = ["model_hash", "fit_content_hash"];

        let mut found: BTreeSet<(String, String)> = BTreeSet::new();
        let mut stack = vec![src_root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let rel = path.strip_prefix(&src_root).unwrap()
                    .to_string_lossy().replace('\\', "/");
                // Skip this module (defs + this test's own references) and
                // `args/` (clap structs never hash).
                if rel == "hashing.rs" || rel.starts_with("args/") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut in_test = false;
                for line in text.lines() {
                    let trimmed = line.trim_start();
                    // Coarse but sufficient: once a `#[cfg(test)]` / `#[test]`
                    // attribute or a `mod tests` is seen, treat the rest of the
                    // file as test code (test modules live at the file tail by
                    // convention in this crate).
                    if trimmed.starts_with("#[cfg(test)]")
                        || trimmed.starts_with("#[test]")
                        || trimmed.starts_with("mod tests")
                    {
                        in_test = true;
                    }
                    if in_test {
                        continue;
                    }
                    for h in &hashers {
                        // Production call forms: `crate::hashing::<h>(` or a
                        // method call `.<h>(` / `self.<h>(`. Doc/comment lines
                        // (`//`) are not calls.
                        if trimmed.starts_with("//") {
                            continue;
                        }
                        let call = format!("{}(", h);
                        if line.contains(&call) {
                            found.insert((rel.clone(), (*h).to_string()));
                        }
                    }
                }
            }
        }

        let allowed: BTreeSet<(String, String)> = LEGACY_IDENTITY_ALLOWLIST
            .iter()
            .map(|(f, h)| ((*f).to_string(), (*h).to_string()))
            .collect();

        let unexpected: Vec<_> = found.difference(&allowed).collect();
        assert!(
            unexpected.is_empty(),
            "new legacy-identity call site(s) not in the allowlist (gh#241): {:?}\n\
             These hashers are display/provenance only — use runid/resolve for \
             run identity, or add the site to LEGACY_IDENTITY_ALLOWLIST if it is \
             genuinely display/provenance.",
            unexpected
        );
        // Also flag a stale allowlist entry whose call site was removed, so the
        // list stays honest as files are refactored.
        let missing: Vec<_> = allowed.difference(&found).collect();
        assert!(
            missing.is_empty(),
            "allowlist entry no longer present in the source (remove it): {:?}",
            missing
        );
    }

    // ── slug ─────────────────────────────────────────────────────────────────

    #[test]
    fn slug_alphanumeric_passthrough() {
        assert_eq!(slug("baseline"), "baseline");
        assert_eq!(slug("with_sia"), "with_sia");
    }

    #[test]
    fn slug_lowercases() {
        assert_eq!(slug("WithSIA"), "withsia");
    }

    #[test]
    fn slug_replaces_spaces_and_specials() {
        assert_eq!(slug("with sia!"), "with_sia_");
        assert_eq!(slug("r0=3.0"), "r0_3_0");
    }

    // ── file_hash / fit_input_hash (relocated from fit::provenance) ─────────

    #[test]
    fn file_hash_returns_8_hex() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_hash_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::write(&tmp, b"hello world").unwrap();
        let h = file_hash(tmp.to_str().unwrap()).unwrap();
        assert_eq!(h.len(), 8, "file_hash should return 8 hex chars");
        // SHA256("hello world")[..4] is b94d27b9 in hex.
        assert_eq!(h, "b94d27b9");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn file_hash_returns_none_for_missing() {
        assert!(file_hash("/does/not/exist/at/all").is_none());
    }

}
