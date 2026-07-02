//! Paired pathogen-vs-human migration test (2026-05-21 migration proposal).
//!
//! Two 2-patch SIR models that couple patches differently:
//!   - `pathogen_migration` — cross-patch force of infection (people don't move),
//!   - `human_migration`    — local transmission + infectives move between patches.
//!
//! They produce **structurally opposite genealogies**, which this test pins:
//!   - cross-deme transmission fraction: pathogen > 0, human = 0;
//!   - migration events: human > 0, pathogen = 0;
//!   - and the regression guard: scoring the human model by *birth* deme (the
//!     pre-fix behaviour) spuriously shows cross-deme transmissions — the
//!     deme-trajectory fix is what keeps the human signal at 0.
//!
//! Shells the release `camdl` for simulate + realize (silent-skip if unbuilt),
//! then reads the TSV line list and computes the statistics in-process.

use std::path::{Path, PathBuf};
use std::process::Command;

use sim::lineage::tree::{
    cross_deme_transmission_fraction, migration_event_count, read_tsv, summarize,
};
use sim::lineage::{LineListEntry, ParentRef};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn fixture(name: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join(format!("../sim/tests/fixtures/{name}.ir.json"))
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_mig_e2e_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(camdl: &Path, args: &[&str]) {
    let o = Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke");
    assert!(o.status.success(), "camdl {:?} failed: {}", args, String::from_utf8_lossy(&o.stderr));
}

/// Simulate `model` at `params`, realize, and return the parsed line list.
fn line_list(camdl: &Path, dir: &Path, model: &str, params: &[&str], seed: &str) -> Vec<LineListEntry> {
    let ir = fixture(model);
    let ev = dir.join(format!("{model}.tsv"));
    let traj = dir.join(format!("{model}_traj.tsv"));
    let ll = dir.join(format!("{model}_ll.tsv"));
    let mut sim = vec![
        "simulate", ir.to_str().unwrap(),
        "--backend", "chain_binomial", "--dt", "1", "--seed", seed,
    ];
    sim.extend_from_slice(params);
    sim.extend(["--event-log", ev.to_str().unwrap(), "--tsv", "--output", traj.to_str().unwrap()]);
    run(camdl, &sim);
    run(camdl, &["lineage", "realize", ev.to_str().unwrap(), "--identity-seed", seed, "-o", ll.to_str().unwrap()]);
    read_tsv(&ll).expect("parse line list")
}

#[test]
fn pathogen_and_human_migration_have_opposite_genealogies() {
    let camdl = camdl_bin();
    let dir = tempdir("mig");

    let p = line_list(
        &camdl, &dir, "pathogen_migration",
        &["--param", "beta=0.5", "--param", "gamma=0.2", "--param", "kappa=0.01"],
        "7",
    );
    let h = line_list(
        &camdl, &dir, "human_migration",
        &["--param", "beta=0.5", "--param", "gamma=0.2", "--param", "m=0.005"],
        "7",
    );

    let fp = cross_deme_transmission_fraction(&p).expect("pathogen has transmissions");
    let fh = cross_deme_transmission_fraction(&h).expect("human has transmissions");

    // The decisive contrast: pathogen migration crosses demes at transmission
    // nodes; human migration's transmission is always local.
    assert!(fp > 0.0, "pathogen migration must have cross-deme transmissions, got {fp}");
    assert_eq!(fh, 0.0, "human migration must have ZERO cross-deme transmissions, got {fh}");

    // The mirror image: migration events live on branches under human migration.
    assert!(migration_event_count(&h).unwrap() > 0, "human migration must have migration events");
    assert_eq!(migration_event_count(&p).unwrap(), 0, "pathogen migration must have no migration events");

    // Regression guard: scoring the human model by *birth* deme (the pre-fix
    // IndividualSummary behaviour) spuriously reports cross-deme transmissions,
    // because a migrant born in a and transmitting in b looks like an a→b edge.
    // The deme-trajectory fix is what keeps the event-time signal (fh) at 0.
    let (summaries, _) = summarize(&h).unwrap();
    let (mut n, mut cross_birth) = (0u64, 0u64);
    for e in &h {
        if let ParentRef::Individual(pid) = e.parent {
            n += 1;
            let parent_birth = summaries[&pid].trajectory.birth_deme();
            let child_birth = summaries[&e.individual].trajectory.birth_deme();
            if parent_birth != child_birth {
                cross_birth += 1;
            }
        }
    }
    let birth_frac = cross_birth as f64 / n as f64;
    assert!(
        birth_frac > 0.0,
        "birth-deme scoring should spuriously show cross-deme transmissions for the \
         human model (the bug the deme-trajectory fix avoids); got {birth_frac}, while \
         the correct event-time fraction is {fh}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
