//! Tests for PGAS chain resume: bincode serialization, config hash.

use sim::inference::pgas::{ChainResumeState, PGASTrajectory, SubstepRecord};
use sim::inference::nuts::MassMatrix;

/// T1: Bincode round-trip — serialize and deserialize a ChainResumeState
/// with all field types (trajectory, dense mass matrix, etc.)
#[test]
fn test_resume_state_bincode_roundtrip() {
    let trajectory = PGASTrajectory {
        initial_counts: vec![1000, 10, 0],
        substeps: vec![
            SubstepRecord {
                counts_before: vec![995, 15, 0],
                counts_after: vec![993, 17, 0],
                flows: vec![5, 3],
                gammas: vec![1.02],
                t0: 0.0,
                dt_substep: 1.0,
            },
            SubstepRecord {
                counts_before: vec![990, 18, 2],
                counts_after: vec![988, 20, 2],
                flows: vec![5, 2],
                gammas: vec![0.98],
                t0: 1.0,
                dt_substep: 1.0,
            },
        ],
    };

    let mass = MassMatrix::dense_from_covariance(&[
        1.0, 0.5, 0.5, 2.0,
    ], 2);

    let state = ChainResumeState {
        config_hash: "abc123def456".into(),
        completed_sweeps: 5000,
        params: vec![0.4, 0.1, 1000.0],
        transformed: vec![-0.916, -2.302, 6.908],
        param_names: vec!["beta".into(), "gamma".into(), "N0".into()],
        trajectory,
        mass_matrix: mass,
        nuts_step_size: 0.0234,
        log_proposal_sd: vec![-5.0, -6.0, -3.0],
        total_accepted: vec![2200, 1800, 2500],
        current_ll: -131456.78,
    };

    // Serialize
    let encoded = bincode::serialize(&state).unwrap();
    eprintln!("  resume state size: {} bytes", encoded.len());

    // Deserialize
    let decoded: ChainResumeState = bincode::deserialize(&encoded).unwrap();

    // Verify all fields
    assert_eq!(decoded.config_hash, "abc123def456");
    assert_eq!(decoded.completed_sweeps, 5000);
    assert_eq!(decoded.params, vec![0.4, 0.1, 1000.0]);
    assert_eq!(decoded.transformed, vec![-0.916, -2.302, 6.908]);
    assert_eq!(decoded.trajectory.initial_counts, vec![1000, 10, 0]);
    assert_eq!(decoded.trajectory.substeps.len(), 2);
    assert_eq!(decoded.trajectory.substeps[0].counts_before, vec![995, 15, 0]);
    assert_eq!(decoded.trajectory.substeps[0].flows, vec![5, 3]);
    assert!((decoded.trajectory.substeps[0].gammas[0] - 1.02).abs() < 1e-10);
    assert!((decoded.nuts_step_size - 0.0234).abs() < 1e-10);
    assert_eq!(decoded.log_proposal_sd, vec![-5.0, -6.0, -3.0]);
    assert_eq!(decoded.total_accepted, vec![2200, 1800, 2500]);
    assert!((decoded.current_ll - (-131456.78)).abs() < 1e-6);

    // Verify mass matrix round-trips (dense)
    match &decoded.mass_matrix {
        MassMatrix::Dense { dim, .. } => assert_eq!(*dim, 2),
        _ => panic!("expected Dense mass matrix after round-trip"),
    }
}

/// T2: Bincode round-trip with diagonal mass matrix
#[test]
fn test_resume_state_diagonal_mass_roundtrip() {
    let state = ChainResumeState {
        config_hash: "test".into(),
        completed_sweeps: 100,
        params: vec![1.0],
        transformed: vec![0.0],
        param_names: vec!["x".into()],
        trajectory: PGASTrajectory { initial_counts: vec![100], substeps: vec![] },
        mass_matrix: MassMatrix::identity(3),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0],
        total_accepted: vec![50],
        current_ll: -100.0,
    };

    let encoded = bincode::serialize(&state).unwrap();
    let decoded: ChainResumeState = bincode::deserialize(&encoded).unwrap();

    match &decoded.mass_matrix {
        MassMatrix::Diagonal(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v, &vec![1.0, 1.0, 1.0]);
        }
        _ => panic!("expected Diagonal mass matrix"),
    }
}

/// T3: Config hash stability — same inputs produce same hash.
#[test]
fn test_config_hash_deterministic() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let compute = || {
        let mut h = DefaultHasher::new();
        "model_json_content".hash(&mut h);
        "data_content".hash(&mut h);
        "R0:bounds=(1,100):prior=lognormal(3.9,0.4)".hash(&mut h);
        100_usize.hash(&mut h);
        1.0_f64.to_bits().hash(&mut h);
        h.finish()
    };
    let hash1 = compute();
    let hash2 = compute();
    assert_eq!(hash1, hash2, "same config must produce same hash");
}

/// T4: Config hash sensitivity — changing invalidating fields changes the hash.
#[test]
fn test_config_hash_sensitivity() {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let compute = |model: &str, n_particles: usize, prior: &str| {
        let mut h = DefaultHasher::new();
        model.hash(&mut h);
        "data".hash(&mut h);
        prior.hash(&mut h);
        n_particles.hash(&mut h);
        1.0_f64.to_bits().hash(&mut h);
        h.finish()
    };

    let base = compute("model_v1", 100, "flat");

    // Changed model → different hash
    assert_ne!(base, compute("model_v2", 100, "flat"));
    // Changed particles → different hash
    assert_ne!(base, compute("model_v1", 200, "flat"));
    // Changed prior → different hash
    assert_ne!(base, compute("model_v1", 100, "lognormal(3.9,0.4)"));
    // Same everything → same hash
    assert_eq!(base, compute("model_v1", 100, "flat"));
}

/// T5: Resume state with mismatched hash is detected.
#[test]
fn test_resume_hash_mismatch_detection() {
    let state = ChainResumeState {
        config_hash: "original_hash_abc123".into(),
        completed_sweeps: 1000,
        params: vec![1.0],
        transformed: vec![0.0],
        param_names: vec!["x".into()],
        trajectory: PGASTrajectory { initial_counts: vec![100], substeps: vec![] },
        mass_matrix: MassMatrix::identity(1),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0],
        total_accepted: vec![500],
        current_ll: -100.0,
    };

    let current_hash = "different_hash_xyz789";
    assert_ne!(state.config_hash, current_hash,
        "mismatched hashes should be detected by the caller");

    // Matching hash should pass
    let matching_hash = "original_hash_abc123";
    assert_eq!(state.config_hash, matching_hash);
}

/// T6: Large trajectory serialization (realistic size)
#[test]
fn test_resume_state_large_trajectory() {
    // Simulate a realistic trajectory size (1000 substeps, 4 compartments)
    let substeps: Vec<SubstepRecord> = (0..1000).map(|i| SubstepRecord {
        counts_before: vec![10000 - i as i64, i as i64, 0, 0], counts_after: vec![10000 - i as i64 - 1, i as i64 + 1, 0, 0],
        flows: vec![1, 0, 0, 0, 0, 0],
        gammas: vec![1.0],
        t0: i as f64,
        dt_substep: 1.0,
    }).collect();

    let state = ChainResumeState {
        config_hash: "large_test".into(),
        completed_sweeps: 10000,
        params: vec![50.0, 0.08, 0.55, 0.035],
        transformed: vec![3.91, -2.53, 0.20, -3.35],
        param_names: vec!["R0".into(), "gamma".into(), "amplitude".into(), "kappa".into()],
        trajectory: PGASTrajectory { initial_counts: vec![10000, 0, 0, 0], substeps },
        mass_matrix: MassMatrix::dense_from_covariance(&[
            1.0, 0.9, 0.1, 0.2,
            0.9, 1.0, 0.3, 0.1,
            0.1, 0.3, 1.0, 0.5,
            0.2, 0.1, 0.5, 1.0,
        ], 4),
        nuts_step_size: 0.015,
        log_proposal_sd: vec![-4.0; 4],
        total_accepted: vec![4500; 4],
        current_ll: -135000.0,
    };

    let encoded = bincode::serialize(&state).unwrap();
    let decoded: ChainResumeState = bincode::deserialize(&encoded).unwrap();

    eprintln!("  large trajectory: {} bytes for {} substeps",
        encoded.len(), decoded.trajectory.substeps.len());
    assert_eq!(decoded.trajectory.substeps.len(), 1000);
    assert_eq!(decoded.completed_sweeps, 10000);
    assert_eq!(decoded.params.len(), 4);
}

/// T7: Resume state file protects trace data.
/// When a resume state exists, the trace must be opened in append mode.
/// When no resume state exists, creating a new trace is safe.
#[test]
fn test_resume_trace_file_safety() {
    use std::io::Write;

    let dir = std::env::temp_dir().join("camdl_test_resume_safety");
    let chain_dir = dir.join("chain_1");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&chain_dir).unwrap();

    let trace_path = chain_dir.join("trace.tsv");
    let resume_path = chain_dir.join("resume_state.bin");

    // Write a fake trace with 100 lines of data
    {
        let mut f = std::fs::File::create(&trace_path).unwrap();
        writeln!(f, "sweep\tlog_likelihood\tparam1").unwrap();
        for i in 1..=100 {
            writeln!(f, "{}\t-1000.0\t0.5", i).unwrap();
        }
    }
    let original_size = std::fs::metadata(&trace_path).unwrap().len();
    assert!(original_size > 0, "trace file should have content");

    // Write a resume state
    let state = ChainResumeState {
        config_hash: "test".into(),
        completed_sweeps: 100,
        params: vec![0.5],
        transformed: vec![0.0],
        param_names: vec!["x".into()],
        trajectory: PGASTrajectory { initial_counts: vec![1000], substeps: vec![] },
        mass_matrix: MassMatrix::identity(1),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0],
        total_accepted: vec![50],
        current_ll: -1000.0,
    };
    let encoded = bincode::serialize(&state).unwrap();
    std::fs::write(&resume_path, &encoded).unwrap();

    // Key invariant: when resume state exists and we open the trace
    // in append mode, original data is preserved.
    {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&trace_path)
            .unwrap();
        writeln!(f, "101\t-999.0\t0.51").unwrap();
    }
    let appended_size = std::fs::metadata(&trace_path).unwrap().len();
    assert!(appended_size > original_size, "append should grow file, not truncate");

    // Verify original content is intact
    let content = std::fs::read_to_string(&trace_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 102, "header + 100 original + 1 appended");
    assert!(lines[0].starts_with("sweep"), "header preserved");
    assert!(lines[1].starts_with("1\t"), "first original line preserved");
    assert!(lines[100].starts_with("100\t"), "last original line preserved");
    assert!(lines[101].starts_with("101\t"), "appended line present");

    // Anti-pattern: File::create would DESTROY existing data
    let would_destroy_size = {
        let path2 = chain_dir.join("trace_destroyed.tsv");
        std::fs::copy(&trace_path, &path2).unwrap();
        {
            let mut f = std::fs::File::create(&path2).unwrap();
            writeln!(f, "sweep\tlog_likelihood\tparam1").unwrap();
            writeln!(f, "101\t-999.0\t0.51").unwrap();
        }
        std::fs::metadata(&path2).unwrap().len()
    };
    assert!(would_destroy_size < original_size,
        "File::create destroys existing content — this is why --resume must use append");

    let _ = std::fs::remove_dir_all(&dir);
}

/// T8: param_names round-trip — names are stored and recovered.
#[test]
fn test_resume_param_names_roundtrip() {
    let state = ChainResumeState {
        config_hash: "test".into(),
        completed_sweeps: 100,
        params: vec![20.0, 0.06, 0.3],
        transformed: vec![3.0, -2.81, -1.20],
        param_names: vec!["R0".into(), "s0".into(), "amplitude".into()],
        trajectory: PGASTrajectory { initial_counts: vec![100], substeps: vec![] },
        mass_matrix: MassMatrix::identity(3),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0; 3],
        total_accepted: vec![50; 3],
        current_ll: -1000.0,
    };

    let encoded = bincode::serialize(&state).unwrap();
    let decoded: ChainResumeState = bincode::deserialize(&encoded).unwrap();

    assert_eq!(decoded.param_names, vec!["R0", "s0", "amplitude"]);
    assert_eq!(decoded.transformed, vec![3.0, -2.81, -1.20]);
}

/// T9: name-based z-value recovery when parameter order differs.
/// Simulates the HashMap ordering bug: saved state has [R0, s0, amplitude]
/// but the current run's if2_params are in order [amplitude, R0, s0].
#[test]
fn test_resume_reorder_z_by_name() {
    // Saved state: order [R0, s0, amplitude]
    let state = ChainResumeState {
        config_hash: "test".into(),
        completed_sweeps: 100,
        params: vec![20.0, 0.06, 0.3],  // full model param vec
        transformed: vec![3.0, -2.81, -1.20],  // z for [R0, s0, amplitude]
        param_names: vec!["R0".into(), "s0".into(), "amplitude".into()],
        trajectory: PGASTrajectory { initial_counts: vec![100], substeps: vec![] },
        mass_matrix: MassMatrix::identity(3),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0; 3],
        total_accepted: vec![50; 3],
        current_ll: -1000.0,
    };

    // Build name→z lookup (same logic as run_pgas resume path)
    let saved_z: std::collections::HashMap<&str, f64> = state.param_names.iter()
        .zip(state.transformed.iter())
        .map(|(name, &z)| (name.as_str(), z))
        .collect();

    // Current run's parameter order: [amplitude, R0, s0] (different!)
    let current_order = ["amplitude", "R0", "s0"];
    let reordered: Vec<f64> = current_order.iter()
        .map(|name| *saved_z.get(name).unwrap())
        .collect();

    // amplitude should get -1.20, R0 should get 3.0, s0 should get -2.81
    assert!((reordered[0] - (-1.20)).abs() < 1e-10, "amplitude z should be -1.20, got {}", reordered[0]);
    assert!((reordered[1] - 3.0).abs() < 1e-10, "R0 z should be 3.0, got {}", reordered[1]);
    assert!((reordered[2] - (-2.81)).abs() < 1e-10, "s0 z should be -2.81, got {}", reordered[2]);
}

/// T10: param_names mismatch detection — saved names must match current config.
#[test]
fn test_resume_param_names_mismatch_detected() {
    let state = ChainResumeState {
        config_hash: "test".into(),
        completed_sweeps: 100,
        params: vec![20.0, 0.06, 0.3],
        transformed: vec![3.0, -2.81, -1.20],
        param_names: vec!["R0".into(), "s0".into(), "amplitude".into()],
        trajectory: PGASTrajectory { initial_counts: vec![100], substeps: vec![] },
        mass_matrix: MassMatrix::identity(3),
        nuts_step_size: 0.1,
        log_proposal_sd: vec![-3.0; 3],
        total_accepted: vec![50; 3],
        current_ll: -1000.0,
    };

    // Build name→z lookup
    let saved_z: std::collections::HashMap<&str, f64> = state.param_names.iter()
        .zip(state.transformed.iter())
        .map(|(name, &z)| (name.as_str(), z))
        .collect();

    // Missing param in saved state should be detected
    let current_names = ["R0", "s0", "amplitude", "new_param"];
    let missing: Vec<&&str> = current_names.iter()
        .filter(|name| !saved_z.contains_key(**name))
        .collect();
    assert_eq!(missing, vec![&"new_param"],
        "should detect that 'new_param' is not in saved state");
}

/// T11: DiagnosticCollector round-trip and severity classification.
#[test]
fn test_diagnostic_collector_basics() {
    use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind, Severity};

    let c = DiagnosticCollector::new("test");

    c.push(DiagnosticKind::RhatHigh { param: "R0".into(), rhat: 1.2, threshold: 1.1 });
    c.push(DiagnosticKind::RhatHigh { param: "sigma".into(), rhat: 1.8, threshold: 1.1 });
    c.push(DiagnosticKind::AutoRwSd { param: "beta".into(), rw_sd: 0.05 });
    c.push(DiagnosticKind::InitialLoglikInfinite);

    assert!(c.has_errors(), "rhat=1.8 and -inf should be errors");
    assert!(c.has_warnings());

    let diags = c.drain();
    assert_eq!(diags.len(), 4);
    assert_eq!(diags[0].severity, Severity::Warning); // rhat 1.2
    assert_eq!(diags[1].severity, Severity::Error);   // rhat 1.8
    assert_eq!(diags[2].severity, Severity::Info);     // auto rw_sd
    assert_eq!(diags[3].severity, Severity::Error);    // -inf

    // JSON round-trip
    let json = serde_json::to_string(&diags).unwrap();
    let decoded: Vec<sim::inference::diagnostic::Diagnostic> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded[0].method, "test");
}
