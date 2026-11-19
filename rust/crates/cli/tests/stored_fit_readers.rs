//! Readers of a stored fit (a run handle to `pfilter`, `profile` or
//! `simulate --fit`) take what they read of its configuration from the
//! folder's `fit.resolved.toml`, the configuration the fit ran under, and key
//! only bytes the folder's records check. End to end through the release
//! binary, over the store `stored_fit_fixture` makes.

mod stored_fit_fixture;

use stored_fit_fixture::*;

/// The folder's `fit.resolved.toml` with `[config] dt` set to `dt`.
fn set_resolved_dt(fx: &Fixture, leaf: &str, dt: f64) {
    let path = fx.folder(leaf).join("fit.resolved.toml");
    let mut doc: toml::Table = std::fs::read_to_string(&path).unwrap().parse().unwrap();
    doc.entry("config")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .unwrap()
        .insert("dt".to_string(), toml::Value::Float(dt));
    std::fs::write(&path, toml::to_string(&doc).unwrap()).unwrap();
}

fn pfilter(fx: &Fixture, fit: &str) -> std::process::Output {
    fx.camdl(&["pfilter", "m.camdl", "--fit", fit, "--param", "beta=0.3", "--param", "gamma=0.2", "--param", "rho=0.5",
               "--particles", "20", "--seed", "1",
               "--progress", "none"])
}

fn profile(fx: &Fixture, fit: &str) -> std::process::Output {
    fx.camdl(&["profile", "m.camdl", "--fit", fit, "--sweep", "beta=0.3", "--particles", "20", "--iterations", "2",
               "--n-starts", "1", "--seed", "1", "--progress", "none"])
}

/// The `[config] dt` `pfilter` and `profile` check against the model's step
/// is the stored fit's resolved one, not its `fit.toml.original`'s. The
/// model declares `dt = 1 'days`; a resolved config stating another step is
/// refused, while the original, which states none, would pass.
#[test]
fn a_stored_fits_written_dt_is_read_from_its_resolved_config() {
    let fx = Fixture::new("stored_fit_readers");
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let handle = fx.dir.join(&leaf).to_string_lossy().into_owned();
    ok(&pfilter(&fx, &handle));
    ok(&profile(&fx, &handle));

    set_resolved_dt(&fx, &leaf, 0.5);
    for (what, out) in [("pfilter", pfilter(&fx, &handle)), ("profile", profile(&fx, &handle))] {
        let said = text(&out);
        assert!(!out.status.success(), "{what} must read the resolved config's dt:\n{said}");
        assert!(said.contains("dt"), "{what}: {said}");
    }
}

/// A stored fit whose `fit.toml.original` is not the file the fit read (its
/// SHA-256 is not `fit_toml_hash`) is refused by the readers that key or read
/// it, rather than profiled under bytes no record checks.
#[test]
fn a_stored_fit_whose_original_is_not_the_files_is_refused() {
    let fx = Fixture::new("stored_fit_readers");
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let handle = fx.dir.join(&leaf).to_string_lossy().into_owned();
    let original = fx.folder(&leaf).join("fit.toml.original");
    let edited = format!("{}# edited after the fit\n", std::fs::read_to_string(&original).unwrap());
    std::fs::write(&original, edited).unwrap();
    for (what, out) in [("pfilter", pfilter(&fx, &handle)), ("profile", profile(&fx, &handle))] {
        let said = text(&out);
        assert!(!out.status.success(), "{what}: {said}");
        assert!(said.contains("fit_toml_hash") && said.contains("fit.toml.original"), "{what}: {said}");
    }
}

/// `simulate --fit <run handle>` reads the config's `[simulate] from` from
/// the stored fit. A fit folder whose config cannot be read (here, one an
/// older camdl wrote, at layout 0) is refused naming the migration, not
/// passed over to run from the model's own start.
#[test]
fn simulate_refuses_a_stored_fit_whose_config_it_cannot_read() {
    let fx = Fixture::new("stored_fit_readers");
    fx.write("fit.toml", &format!("{}[simulate]\nfrom = -7\n", fit_config("gamma = 0.2\nrho = 0.5")));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let handle = fx.dir.join(&leaf).to_string_lossy().into_owned();
    let sim = || fx.camdl(&["simulate", "m.camdl", "--fit", &handle, "--param", "beta=0.3", "--param", "gamma=0.2",
                            "--param", "rho=0.5", "--seed", "1", "--stdout", "--progress", "none"]);
    ok(&sim());

    let folder = fx.folder(&leaf);
    std::fs::remove_file(folder.join("fit.resolved.toml")).unwrap();
    let meta = folder.join("fit.meta.json");
    let mut v = json(&meta);
    v.as_object_mut().unwrap().remove("store_layout");
    set_json(&meta, &v);
    let out = sim();
    let said = text(&out);
    assert!(!out.status.success(), "a layout-0 fit folder must not be passed over:\n{said}");
    assert!(said.contains(&format!("camdl dev migrate {}", folder.display())), "{said}");
}
