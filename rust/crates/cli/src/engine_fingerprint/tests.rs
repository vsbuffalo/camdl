//! The engine fingerprint, on injected inputs and on a scratch repository laid
//! out like this one. A real rebuild per case would take minutes; the
//! functions tested here are the ones `build.rs` calls, so what they return for
//! a tree is what a build of that tree embeds. That cargo re-runs the script
//! when such a tree changes is the `rerun-if-changed` list, checked by
//! `the_watched_paths_cover_every_crate_source_dir`.

use super::*;
use std::collections::HashMap;

fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    move |k: &str| map.get(k).cloned()
}

const BUILD_ENV: &[(&str, &str)] = &[
    ("CARGO_PKG_VERSION", "0.4.0"),
    ("DEP_CAMDL_SIM_FEATURES", "default,lineage-parquet,ode"),
    ("OUT_DIR", "/t/debug/build/cli-0123/out"),
    ("PROFILE", "debug"),
    ("OPT_LEVEL", "0"),
    ("DEBUG", "true"),
    ("CARGO_CFG_DEBUG_ASSERTIONS", ""),
    ("CARGO_ENCODED_RUSTFLAGS", ""),
    ("TARGET", "aarch64-apple-darwin"),
];

fn rustc() -> Option<String> {
    Some("rustc 1.94.1 (test)\nLLVM version: 21\n".into())
}

fn cli_features() -> Vec<String> {
    vec!["cli:default".into(), "cli:lineage_parquet".into(), "cli:ode".into()]
}

/// The fingerprint a build of the tree at `root` would embed, under a fixed
/// build environment.
fn build_of(root: &Path) -> Fingerprint {
    let env = env_of(BUILD_ENV);
    fingerprint(&gather(root, &env, &rustc, cli_features()), || panic!("every component is known"))
}

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"])
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

/// A tree with this repository's layout: two crates (`sim`, a dependency, and
/// `cli`, the binary), their tests and benches, the workspace files, `ir/VERSION`
/// and a document. Committed when `commit` is set.
fn scratch_tree(commit: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for (rel, text) in [
        ("rust/Cargo.toml", "[workspace]\n"),
        ("rust/Cargo.lock", "# lock\n"),
        ("rust/crates/sim/Cargo.toml", "[package]\nname = \"sim\"\n"),
        ("rust/crates/sim/build.rs", "fn main() {}\n"),
        ("rust/crates/sim/src/lib.rs", "pub fn rate() -> f64 { 1.0 }\n#[cfg(test)]\nmod tests;\n"),
        ("rust/crates/sim/src/tests.rs", "#[test] fn t() {}\n"),
        ("rust/crates/sim/tests/sampler.rs", "#[test] fn s() {}\n"),
        ("rust/crates/sim/benches/b.rs", "fn main() {}\n"),
        ("rust/crates/cli/Cargo.toml", "[package]\nname = \"cli\"\n"),
        ("rust/crates/cli/build.rs", "fn main() {}\n"),
        ("rust/crates/cli/src/main.rs", "fn main() {}\n"),
        ("ir/VERSION", "0.43\n"),
        ("docs/guide.md", "# guide\n"),
    ] {
        write(root, rel, text);
    }
    if commit {
        git(root, &["init", "-q"]);
        git(root, &["add", "-A"]);
        git(root, &["commit", "-q", "-m", "init"]);
    }
    dir
}

/// The regression the fingerprint exists for: two builds that differ only by
/// an uncommitted edit to the engine's source must not share an identity.
/// Under the commit-only engine version they did, so the second build was
/// served the first's stored runs.
#[test]
fn an_uncommitted_edit_to_a_dependency_crate_changes_the_engine_id() {
    let tree = scratch_tree(true);
    let root = tree.path();
    let clean = build_of(root);
    assert!(clean.missing.is_empty(), "{:?}", clean.missing);
    assert!(!clean.source_label.contains("dirty"), "{}", clean.source_label);

    write(root, "rust/crates/sim/src/lib.rs", "pub fn rate() -> f64 { 2.0 }\n#[cfg(test)]\nmod tests;\n");
    let edited = build_of(root);
    assert_ne!(edited.engine_id, clean.engine_id, "an edit to sim/src must re-key the engine");
    assert!(edited.source_label.starts_with(&format!("{}.dirty-", clean.source_label)), "{}", edited.source_label);

    // Undoing the edit is the clean build again.
    write(root, "rust/crates/sim/src/lib.rs", "pub fn rate() -> f64 { 1.0 }\n#[cfg(test)]\nmod tests;\n");
    assert_eq!(build_of(root), clean);
}

#[test]
fn edits_in_cli_and_new_source_files_change_the_engine_id_and_a_commit_changes_it_again() {
    let tree = scratch_tree(true);
    let root = tree.path();
    let clean = build_of(root);

    write(root, "rust/crates/cli/src/main.rs", "fn main() { let _ = 1; }\n");
    let cli_edit = build_of(root);
    assert_ne!(cli_edit.engine_id, clean.engine_id, "an edit to cli/src");

    // Two different uncommitted states are two identities.
    write(root, "rust/crates/cli/src/main.rs", "fn main() { let _ = 2; }\n");
    assert_ne!(build_of(root).engine_id, cli_edit.engine_id, "a second edit");
    write(root, "rust/crates/cli/src/main.rs", "fn main() {}\n");

    // An untracked, not-ignored source file and a deletion count.
    write(root, "rust/crates/sim/src/new_kernel.rs", "pub fn k() {}\n");
    assert_ne!(build_of(root).engine_id, clean.engine_id, "an untracked source file");
    std::fs::remove_file(root.join("rust/crates/sim/src/new_kernel.rs")).unwrap();
    std::fs::remove_file(root.join("rust/crates/cli/build.rs")).unwrap();
    assert_ne!(build_of(root).engine_id, clean.engine_id, "a deleted build script");
    git(root, &["checkout", "--", "rust/crates/cli/build.rs"]);
    assert_eq!(build_of(root), clean);

    for rel in ["rust/Cargo.toml", "rust/crates/sim/Cargo.toml", "rust/crates/sim/build.rs", "ir/VERSION"] {
        let before = std::fs::read_to_string(root.join(rel)).unwrap();
        write(root, rel, &format!("{before}# edit\n"));
        assert_ne!(build_of(root).engine_id, clean.engine_id, "{rel}");
        write(root, rel, &before);
    }

    // Committing an edit: a new commit, no uncommitted changes, a new id.
    write(root, "rust/crates/sim/src/lib.rs", "pub fn rate() -> f64 { 3.0 }\n#[cfg(test)]\nmod tests;\n");
    let dirty = build_of(root);
    git(root, &["commit", "-q", "-am", "edit"]);
    let committed = build_of(root);
    assert!(!committed.source_label.contains("dirty"), "{}", committed.source_label);
    assert_ne!(committed.engine_id, clean.engine_id);
    assert_ne!(committed.engine_id, dirty.engine_id);
}

#[test]
fn editing_tests_benches_or_documents_leaves_the_engine_id_unchanged() {
    let tree = scratch_tree(true);
    let root = tree.path();
    let clean = build_of(root);
    for rel in [
        "rust/crates/sim/tests/sampler.rs",
        "rust/crates/sim/src/tests.rs",
        "rust/crates/sim/benches/b.rs",
        "docs/guide.md",
    ] {
        let before = std::fs::read_to_string(root.join(rel)).unwrap();
        write(root, rel, &format!("{before}// edit\n"));
        assert_eq!(build_of(root), clean, "{rel} is not a build input");
        write(root, rel, &before);
    }
    write(root, "rust/crates/sim/tests/new_test.rs", "#[test] fn n() {}\n");
    assert_eq!(build_of(root), clean, "a new test file");
}

#[test]
fn without_git_the_source_is_a_digest_of_the_build_inputs() {
    let tree = scratch_tree(false);
    let root = tree.path();
    let a = build_of(root);
    assert!(a.missing.is_empty(), "{:?}", a.missing);
    assert!(a.source_label.starts_with("nogit-"), "{}", a.source_label);
    assert_eq!(build_of(root), a, "deterministic");

    write(root, "rust/crates/sim/tests/sampler.rs", "// edit\n");
    assert_eq!(build_of(root), a, "a test file");
    write(root, "rust/crates/sim/src/lib.rs", "pub fn rate() -> f64 { 2.0 }\n");
    assert_ne!(build_of(root).engine_id, a.engine_id, "a source file");
}

fn inputs() -> BuildInputs {
    BuildInputs {
        pkg_version: "0.4.0".into(),
        source: Some(SourceState::Git { commit: "abc1234ffff".into(), short: "abc1234".into(), dirty: None }),
        cargo_lock: Some("11".into()),
        features: Some(vec!["cli:default".into(), "sim:lineage-parquet".into(), "sim:ode".into()]),
        profile: Some("dir=debug".into()),
        rustc: Some("rustc 1.94.1".into()),
        rustflags: Some(String::new()),
        target: Some("aarch64-apple-darwin".into()),
    }
}

fn id(i: &BuildInputs) -> String {
    fingerprint(i, || panic!("every component is known")).engine_id
}

#[test]
fn every_component_enters_the_engine_id() {
    let base = id(&inputs());
    let variants: Vec<(&str, BuildInputs)> = vec![
        ("pkg_version", BuildInputs { pkg_version: "0.5.0".into(), ..inputs() }),
        ("commit", BuildInputs {
            source: Some(SourceState::Git { commit: "def5678ffff".into(), short: "abc1234".into(), dirty: None }),
            ..inputs()
        }),
        ("dirty", BuildInputs {
            source: Some(SourceState::Git { commit: "abc1234ffff".into(), short: "abc1234".into(), dirty: Some("3f9c21e0aa".into()) }),
            ..inputs()
        }),
        ("cargo_lock", BuildInputs { cargo_lock: Some("12".into()), ..inputs() }),
        ("a sim feature switched off", BuildInputs {
            features: Some(vec!["cli:default".into(), "sim:ode".into()]),
            ..inputs()
        }),
        ("profile", BuildInputs { profile: Some("dir=release".into()), ..inputs() }),
        ("rustc", BuildInputs { rustc: Some("rustc 1.95.0".into()), ..inputs() }),
        ("rustflags", BuildInputs { rustflags: Some("-Ctarget-cpu=native".into()), ..inputs() }),
        ("target", BuildInputs { target: Some("x86_64-unknown-linux-gnu".into()), ..inputs() }),
    ];
    for (what, v) in variants {
        assert_ne!(id(&v), base, "{what} must change ENGINE_ID");
    }
}

/// `sim`'s features arrive through `links` metadata, not `cli`'s own: switching
/// one in the build environment changes the id; with it absent the identity
/// is not determined.
#[test]
fn a_sim_feature_reaches_the_engine_id_through_its_build_script() {
    let tree = scratch_tree(true);
    let root = tree.path();
    let base = build_of(root);
    let mut env: Vec<(&str, &str)> = BUILD_ENV.to_vec();
    for (k, v) in env.iter_mut() {
        if *k == "DEP_CAMDL_SIM_FEATURES" {
            *v = "default,ode";
        }
    }
    let switched = fingerprint(&gather(root, &env_of(&env), &rustc, cli_features()), || unreachable!());
    assert_ne!(switched.engine_id, base.engine_id);

    env.retain(|(k, _)| *k != "DEP_CAMDL_SIM_FEATURES");
    let unknown = fingerprint(&gather(root, &env_of(&env), &rustc, cli_features()), || "n".into());
    assert_eq!(unknown.missing, vec!["features"]);
}

#[test]
fn a_missing_component_is_reported_and_the_id_is_unique_per_build() {
    let mut i = inputs();
    i.rustc = None;
    i.source = None;
    let a = fingerprint(&i, || "build-a".into());
    let b = fingerprint(&i, || "build-b".into());
    assert_eq!(a.missing, vec!["source", "rustc"]);
    assert_eq!(a.source_label, "unknown");
    assert_ne!(a.engine_id, b.engine_id, "two undetermined builds never share an id");
    assert_eq!(a.engine_id.len(), 32);
}

#[test]
fn build_inputs_are_sources_manifests_and_build_scripts_not_tests() {
    for (rel, want) in [
        ("ir/VERSION", true),
        ("rust/Cargo.toml", true),
        ("rust/Cargo.lock", false),
        ("rust/src/lib.rs", true),
        ("rust/tests/golden_deser.rs", false),
        ("rust/crates/sim/Cargo.toml", true),
        ("rust/crates/sim/build.rs", true),
        ("rust/crates/sim/src/inference/pgas.rs", true),
        ("rust/crates/sim/src/tests.rs", false),
        ("rust/crates/cli/src/resolve/tests.rs", false),
        ("rust/crates/cli/src/x/tests/helper.rs", false),
        ("rust/crates/sim/tests/sampler.rs", false),
        ("rust/crates/sim/benches/scaling.rs", false),
        ("rust/crates/sim/README.md", false),
        ("rust/target/debug/build/x.rs", false),
        ("ocaml/lib/compiler/compiler.ml", false),
        ("docs/guide.md", false),
    ] {
        assert_eq!(is_build_input(rel), want, "{rel}");
    }
}

fn this_repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).unwrap().to_path_buf()
}

/// The test-only files kept inside `src/` are excluded from the fingerprint,
/// which is sound only while each is compiled under `#[cfg(test)]`: a
/// `tests.rs` that reached the binary would change it without re-keying.
#[test]
fn every_tests_rs_under_a_src_dir_is_declared_cfg_test() {
    let root = this_repo();
    let files = {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else {
                    out.push(p);
                }
            }
        }
        let mut out = Vec::new();
        for p in watched_source_paths(&root) {
            if p.is_dir() && p.file_name().is_some_and(|n| n == "src") {
                walk(&p, &mut out);
            }
        }
        out
    };
    assert!(!files.is_empty(), "the walk found the repository's src dirs");
    // Nothing under src sits in a `tests` directory today; the exclusion
    // would need the same check as `tests.rs` if something did.
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        let in_src = rel.split_once("/src/").map(|(_, t)| t).unwrap_or("");
        assert!(!in_src.split('/').any(|c| c == "tests"), "{rel}");
    }
    for f in files.into_iter().filter(|p| p.file_name().is_some_and(|n| n == "tests.rs")) {
        let dir = f.parent().unwrap();
        // `a/b/tests.rs` is declared in `a/b.rs`, `a/b/mod.rs`, or (for a
        // crate root's own `src/tests.rs`) `lib.rs` / `main.rs`.
        let parents = [
            dir.with_extension("rs"),
            dir.join("mod.rs"),
            dir.join("lib.rs"),
            dir.join("main.rs"),
        ];
        let declared = parents.iter().filter(|p| p.is_file()).any(|p| {
            let text = std::fs::read_to_string(p).unwrap();
            text.contains("#[cfg(test)]\nmod tests;")
        });
        assert!(declared, "{} is not declared `#[cfg(test)] mod tests;`", f.display());
    }
}

#[test]
fn the_watched_paths_cover_every_crate_source_dir() {
    let root = this_repo();
    let watched = watched_source_paths(&root);
    for rel in ["rust/crates/sim/src", "rust/crates/cli/src", "rust/crates/runid/src", "rust/crates/sim/build.rs",
                "rust/crates/cli/build.rs", "rust/Cargo.lock", "rust/Cargo.toml", "ir/VERSION"] {
        assert!(watched.contains(&root.join(rel)), "{rel} is not watched");
    }
    assert!(watched.iter().all(|p| p.exists()), "a watched path that does not exist re-runs every build");
}

/// A commit moves the branch ref the script watches, so committing re-runs
/// it; every watched git path exists (a missing one re-runs every build).
#[test]
fn the_watched_git_paths_are_head_and_its_branch_ref() {
    let tree = scratch_tree(true);
    let root = tree.path();
    let watched = watched_git_paths(root);
    let git_dir = root.join(".git");
    assert!(watched.iter().any(|p| p.ends_with(".git/HEAD")), "{watched:?}");
    assert!(watched.contains(&git_dir.join("refs/heads/main")), "{watched:?}");
    assert!(watched.iter().all(|p| p.exists()), "{watched:?}");
    // A packed ref has no loose file: its nearest existing directory is watched.
    git(root, &["pack-refs", "--all"]);
    let packed = watched_git_paths(root);
    assert!(packed.iter().all(|p| p.exists()), "{packed:?}");
    assert!(packed.contains(&git_dir.join("packed-refs")), "{packed:?}");
    assert!(packed.contains(&git_dir.join("refs/heads")), "{packed:?}");
}
