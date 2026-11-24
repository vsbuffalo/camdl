//! Test-only support helpers shared across the crate's unit tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A collision-free unique temp-directory path for a unit test.
///
/// The previous `temp_dir().join(format!("..._{pid}_{nanos}"))` scheme raced
/// between parallel test threads that hit the same nanosecond, producing
/// intermittent filesystem failures under full-suite load (gh#153). A
/// process-wide monotonic counter guarantees cross-thread uniqueness; the pid
/// keeps it unique across concurrent test processes. The directory is NOT
/// created — callers `create_dir_all` (and clean up) as before.
pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("camdl_{}_{}_{}", prefix, std::process::id(), n))
}

/// A minimal, valid `ir::Model` for unit tests that need a model value but
/// not its dynamics: no compartments, no transitions, no parameters, a daily
/// regular output schedule over `[0, 100]`. Callers set the fields a test
/// reads (`parameters`, `presets`, …).
pub(crate) fn tiny_model() -> ir::Model {
    use ir::model::{
        InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule,
        SimulationConfig,
    };
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "sir".into(),
        version: "1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        initial_conditions: InitialConditions::default(),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            dt: Some(1.0),
            rng_seed: None,
            integrator: Default::default(),
            t_end_anchor: None, t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

/// The execution inputs the first result of `kind` under the store `root`
/// recorded in its `run.json`, each under `recorded.<field>`, with `<field>`
/// the input's dotted name in the command's `fields()`. For a harness's
/// `InputHarness::shows_use`.
pub(crate) fn recorded_execution(root: &std::path::Path, kind: &str) -> std::collections::BTreeMap<String, String> {
    use std::collections::BTreeMap;
    fn flatten(prefix: &str, v: &serde_json::Value, out: &mut BTreeMap<String, String>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, x) in m {
                    flatten(&if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") }, x, out);
                }
            }
            serde_json::Value::String(x) => {
                out.insert(format!("recorded.{prefix}"), x.clone());
            }
            other => {
                out.insert(format!("recorded.{prefix}"), other.to_string());
            }
        }
    }
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.join("identity.json").is_file() {
                let id: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("identity.json")).unwrap()).unwrap();
                if id["kind"] == kind {
                    let run: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("run.json")).unwrap()).unwrap();
                    flatten("", &run["execution"], &mut out);
                    return out;
                }
            }
            stack.push(p);
        }
    }
    out
}

/// Whether the IR cache directory `dir` holds a compiled model, which a
/// compile through the cache writes and `--no-ir-cache` does not.
pub(crate) fn ir_cache_filled(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir)
        .map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".ir.json")))
        .unwrap_or(false)
}

/// A pin table: `(file, digest)` per case, for one platform.
pub(crate) type Pins = &'static [(&'static str, &'static [(&'static str, &'static str)])];

/// Asserts that `got`, the digests a platform computed per case, are the
/// pins `pins` holds for this `(target_os, target_arch)`. Digests that
/// depend on the platform's libm are pinned per platform. On a platform with
/// no pins it fails, printing the table it computed and the row to add.
pub(crate) fn assert_pinned(
    pins: &[((&str, &str), Pins)],
    got: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>,
    what: &str,
) {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let Some((_, pinned)) = pins.iter().find(|((o, a), _)| *o == os && *a == arch) else {
        let name = format!("{os}_{arch}").to_uppercase();
        panic!(
            "no digests are pinned for {os}/{arch}; this platform computed the table below. \
             Add it, and the row ((\"{os}\", \"{arch}\"), {name}) to PINS:\n\n{}",
            render_pins(&name, got)
        );
    };
    let want: std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>> = pinned
        .iter()
        .map(|(m, fs)| (m.to_string(), fs.iter().map(|(f, d)| (f.to_string(), d.to_string())).collect()))
        .collect();
    assert_eq!(got, &want, "{what}");
}

/// `got` as the Rust source of a pin table named `name`.
fn render_pins(name: &str, got: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>) -> String {
    let mut s = format!("const {name}: Pins = &[\n");
    for (m, fs) in got {
        s += &format!("    (\"{m}\", &[\n");
        for (f, d) in fs {
            s += &format!("        (\"{f}\", \"{d}\"),\n");
        }
        s += "    ]),\n";
    }
    s + "];\n"
}

/// Each link of the link tree at `root` (relative to `cwd`), as an export
/// that copies the stored file it points at: the bytes' digest in `exports`
/// and the file in `copies`, each under the link's path relative to `cwd`.
/// A link directory's record of its links is not a stored file's copy and is
/// left out. For a harness's `InputHarness::run`.
pub(crate) fn link_exports(
    cwd: &std::path::Path,
    root: &std::path::Path,
    exports: &mut std::collections::BTreeMap<PathBuf, store::Key>,
    copies: &mut std::collections::BTreeMap<PathBuf, store::StoredFile>,
) {
    let mut stack = vec![cwd.join(root)];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            let meta = std::fs::symlink_metadata(&p).unwrap();
            if meta.is_dir() {
                stack.push(p);
                continue;
            }
            if !meta.file_type().is_symlink() {
                continue;
            }
            let target = p.canonicalize().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let folder = target.ancestors().find(|a| a.join("identity.json").is_file()).expect("a link points into a result's folder");
            let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(folder.join("identity.json")).unwrap()).unwrap();
            let run_id = store::Key::from_hex(identity["run_id"].as_str().unwrap()).unwrap();
            let file = target.strip_prefix(folder).unwrap().to_string_lossy().into_owned();
            let rel = p.strip_prefix(cwd).unwrap().to_path_buf();
            exports.insert(rel.clone(), store::Key::digest_bytes(&std::fs::read(&target).unwrap()));
            copies.insert(rel, store::StoredFile { run_id, file });
        }
    }
}
