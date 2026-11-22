//! Readers of a stored fit (a run handle to `pfilter`, `profile` or
//! `simulate --fit`) take what they read of its configuration from the
//! problem's `problem.toml`, the configuration the fit ran under, and read
//! only bytes the store's records check. End to end through the release
//! binary, over the store `stored_fit_fixture` makes.

mod stored_fit_fixture;

use stored_fit_fixture::*;

fn pfilter(fx: &Fixture, fit: &str) -> std::process::Output {
    fx.camdl(&["pfilter", "m.camdl", "--fit", fit, "--param", "beta=0.3", "--param", "gamma=0.2", "--param", "rho=0.5",
               "--particles", "20", "--seed", "1",
               "--progress", "none"])
}

fn profile(fx: &Fixture, fit: &str) -> std::process::Output {
    fx.camdl(&["profile", "m.camdl", "--fit", fit, "--sweep", "beta=0.3", "--particles", "20", "--iterations", "2",
               "--n-starts", "1", "--seed", "1", "--progress", "none"])
}

fn simulate(fx: &Fixture, fit: &str) -> std::process::Output {
    fx.camdl(&["simulate", "m.camdl", "--fit", fit, "--param", "beta=0.3", "--param", "gamma=0.2",
               "--param", "rho=0.5", "--seed", "1", "--stdout", "--progress", "none"])
}

/// `pfilter`, `profile` and `simulate` read a stored fit named by its run
/// folder and by its problem folder.
#[test]
fn a_stored_fit_is_read_by_its_run_and_its_problem() {
    let fx = Fixture::new("stored_fit_readers");
    fx.write("fit.toml", &format!("{}[simulate]\nfrom = -7\n", fit_config("gamma = 0.2\nrho = 0.5")));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let problem = fx.folder(&leaf).to_string_lossy().into_owned();
    for handle in [&leaf, &problem] {
        ok(&pfilter(&fx, handle));
        ok(&profile(&fx, handle));
        ok(&simulate(&fx, handle));
    }
}

/// A stored fit whose `problem.toml` is not the file the store recorded (its
/// SHA-256 is not the recorded digest) is refused by every reader, naming the
/// file, rather than read under bytes no record checks. The edit sets
/// `[config] dt` to a step the model does not declare, so a reader that read
/// the edited file would refuse it for its step instead. The fit sets
/// `[simulate] from`, so `simulate --fit` without `--draws` reads the fit
/// rather than refusing for want of a start time.
#[test]
fn a_stored_fit_whose_problem_file_is_edited_is_refused() {
    let fx = Fixture::new("stored_fit_readers");
    fx.write("fit.toml", &format!("{}[simulate]\nfrom = -7\n", fit_config("gamma = 0.2\nrho = 0.5")));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let path = fx.folder(&leaf).join("problem.toml");
    let body = std::fs::read_to_string(&path).unwrap();
    assert!(body.contains("dt = 1.0"), "{body}");
    std::fs::write(&path, body.replace("dt = 1.0", "dt = 0.5")).unwrap();
    for (what, out) in [("pfilter", pfilter(&fx, &leaf)), ("profile", profile(&fx, &leaf)),
                        ("simulate", simulate(&fx, &leaf))] {
        let said = text(&out);
        assert!(!out.status.success(), "{what} must not read an edited problem.toml:\n{said}");
        assert!(said.contains("problem.toml"), "{what} names the file: {said}");
    }
}
