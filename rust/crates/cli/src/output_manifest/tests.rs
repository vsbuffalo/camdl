use super::*;

fn key(byte: u8) -> Key {
    Key::from_hex(&format!("{byte:02x}").repeat(32)).unwrap()
}

/// A manifest of two results, the second a by-product of the first.
fn sample(n_files: usize) -> OutputManifest {
    let files = |kind: &str| -> BTreeMap<String, ManifestFile> {
        (0..n_files)
            .map(|i| (format!("{kind}_{i}"), ManifestFile { file: format!("{kind}_{i}.tsv"), sha256: key(i as u8), bytes: i as u64 * 10 }))
            .collect()
    };
    OutputManifest {
        schema: SCHEMA.into(),
        camdl: BuildRecord { version: "0.1.0+abc".into(), engine_id: "0".repeat(32), ambient_key: key(0xaa) },
        label: Some("run".into()),
        results: vec![
            ManifestEntry {
                kind: "pfilter".into(),
                run_id: key(1),
                upstream: None,
                contract: 1,
                labels: serde_json::json!({ "model": "sir", "seed": 7 }),
                files: files("loglik"),
            },
            ManifestEntry {
                kind: "pf_health".into(),
                run_id: key(2),
                upstream: Some(key(1)),
                contract: 1,
                labels: serde_json::Value::Null,
                files: files("pf_health"),
            },
        ],
    }
}

#[test]
fn a_manifest_round_trips_through_its_type() {
    let m = sample(3);
    let back: OutputManifest = serde_json::from_slice(&m.to_bytes()).unwrap();
    assert_eq!(back, m);
    assert!(m.to_bytes().ends_with(b"}\n"), "the file ends with a newline");
}

#[test]
fn a_manifest_with_a_field_the_type_does_not_hold_is_refused() {
    let mut v: serde_json::Value = serde_json::from_slice(&sample(1).to_bytes()).unwrap();
    v["results"][0]["path"] = "/somewhere".into();
    assert!(serde_json::from_value::<OutputManifest>(v).is_err());
}

/// The example in run spec §2.10, a manifest `camdl simulate` wrote, reads
/// as the type.
#[test]
fn the_run_spec_example_reads_as_the_type() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/camdl-run-spec.md");
    let text = std::fs::read_to_string(&path).unwrap();
    let section = &text[text.find("### 2.10 The output manifest").expect("the run spec has §2.10")..];
    let start = section.find("```json\n").expect("§2.10 holds a JSON example") + "```json\n".len();
    let end = start + section[start..].find("```").unwrap();
    let m: OutputManifest = serde_json::from_str(&section[start..end]).unwrap();
    assert_eq!(m.schema, SCHEMA);
    assert!(m.results.iter().any(|r| r.upstream.is_some()), "the example shows a by-product");
    for r in &m.results {
        let contract = crate::result_files::contract(&r.kind).expect("a kind camdl stores");
        assert_eq!(r.contract, contract.version);
        for (logical, f) in &r.files {
            assert_eq!(contract.logical(&f.file).as_deref(), Some(logical.as_str()), "{} names `{}`", r.kind, f.file);
        }
    }
}

#[test]
fn claiming_a_path_removes_the_earlier_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.json");
    std::fs::write(&path, sample(1).to_bytes()).unwrap();
    ManifestPath::claim(&path).unwrap();
    assert!(!path.exists());
    ManifestPath::claim(&path).expect("a path with no manifest is claimed");
    std::fs::write(&path, "").unwrap();
    ManifestPath::claim(&path).expect("an empty file is claimed");
    assert!(!path.exists());
    std::fs::write(&path, r#"{"schema": "camdl.output-manifest/v9", "other": 1}"#).unwrap();
    ManifestPath::claim(&path).expect("a manifest of another version is claimed");
    assert!(!path.exists());
}

/// A file that is not an output manifest is refused and left as it is:
/// data, JSON of another schema, and JSON with no schema.
#[test]
fn a_file_that_is_not_a_manifest_is_never_removed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.tsv");
    for text in ["time\tcases\n1\t2\n", r#"{"schema": "camdl.predictive/v4"}"#, r#"{"results": []}"#, "[1, 2]"] {
        std::fs::write(&path, text).unwrap();
        let e = ManifestPath::claim(&path).unwrap_err();
        assert!(e.contains("holds a file that is not an output manifest"), "{e}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "the file is left as it was");
    }
}

#[test]
fn a_directory_or_a_missing_parent_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let e = ManifestPath::claim(dir.path()).unwrap_err();
    assert!(e.contains("is a directory"), "{e}");
    let e = ManifestPath::claim(&dir.path().join("absent/m.json")).unwrap_err();
    assert!(e.contains("does not exist"), "{e}");
}

/// A reader polling the path while the manifest is written again and again
/// finds no file or the whole manifest, never part of one.
#[test]
fn a_reader_never_sees_a_partial_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.json");
    let claimed = ManifestPath::claim(&path).unwrap();
    let bytes = sample(4000).to_bytes();
    assert!(bytes.len() > 500_000, "premise: the manifest is larger than one write");
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let reader = s.spawn(|| {
            let mut whole = 0usize;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                match std::fs::read(&path) {
                    Ok(read) => {
                        assert_eq!(read.len(), bytes.len(), "a reader saw {} of {} bytes", read.len(), bytes.len());
                        assert!(read == bytes, "a reader saw other bytes");
                        whole += 1;
                    }
                    Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
                }
            }
            whole
        });
        for _ in 0..40 {
            claimed.write_bytes(&bytes).unwrap();
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(reader.join().unwrap() > 0, "the reader read the manifest at least once");
    });
}

/// A write that fails, here into a directory that cannot be written, leaves
/// no manifest and no temporary file at the path.
#[cfg(unix)]
#[test]
fn a_failed_write_leaves_no_manifest() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("build");
    std::fs::create_dir(&sub).unwrap();
    let path = sub.join("m.json");
    std::fs::write(&path, sample(1).to_bytes()).unwrap();
    let claimed = ManifestPath::claim(&path).unwrap();
    std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o555)).unwrap();
    let written = claimed.write_bytes(&sample(1).to_bytes());
    std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(written.is_err(), "premise: the directory cannot be written");
    let left: Vec<_> = std::fs::read_dir(&sub).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "nothing is left at the manifest's path: {left:?}");
}
