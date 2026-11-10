//! gh#1009 — when `camdl compare` cannot derive a prequential because the
//! `camdl pfilter` it runs failed for a reason unrelated to the posterior draw
//! it was scoring (here, the fit's data file cannot be read), the error says so. It
//! must not lead with the diagnosis that belongs to a draw the filter rejected
//! ("the posterior holds θ the filter cannot track"), which sends the reader
//! to drop chains from a good fit.
//!
//! The rejected-draw diagnosis itself stays, for the case it describes:
//! `compare_auto_prequential.rs` pins it on a deliberately stuck chain.
//!
//! The file is made unreadable rather than removed: a data file that is not
//! there at all is refused before any filter runs, when the fit's config is
//! recovered (`gh1009_stored_paths_resolve_from_any_directory.rs`), so only a
//! file that exists but cannot be read still reaches `camdl pfilter`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// A closed SIR with a weekly negative-binomial observation; `rho` is fixed
/// per fit so the two fits are distinct.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(rho: f64) -> String {
    format!(
        r#"[model]
camdl = "sir.camdl"

[data.observations]
weekly_cases = "data/weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = {rho}
k   = 10.0

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 300
burn_in = 100
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

#[test]
fn a_missing_data_file_is_reported_as_itself_not_as_an_unscorable_draw() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());

    let tmp = std::env::temp_dir().join(format!("camdl_gh1009_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("data")).unwrap();
    std::fs::write(tmp.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("data/weekly_cases.tsv"), DATA).unwrap();
    for (rho, label) in [(0.5, "a"), (0.6, "b")] {
        std::fs::write(tmp.join("fit.toml"), fit_toml(rho)).unwrap();
        let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--label", label, "--seed", "1"]);
        assert!(
            out.status.success(),
            "fit run {label} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // The fits are sound; only their data cannot be read.
    use std::os::unix::fs::PermissionsExt;
    let data = tmp.join("data/weekly_cases.tsv");
    std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o000)).unwrap();

    let out = run(
        &bin,
        &tmp,
        &["compare", "@a", "@b", "--draws", "4", "--replicates", "1", "--particles", "100"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!out.status.success(), "compare cannot score without data:\n{stderr}");
    assert!(
        stderr.contains("data/weekly_cases.tsv: Permission denied"),
        "the cause is named:\n{stderr}"
    );
    for blame in ["could not be scored", "the posterior holds θ", "--exclude-chains"] {
        assert!(
            !stderr.contains(blame),
            "a missing data file is not a rejected posterior draw ({blame:?}):\n{stderr}"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
