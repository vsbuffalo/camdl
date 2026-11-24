use super::*;
use store::{Begin, ContentArtifact, Given, Identity, Reuse, RunInfo, StoreRequest};

/// Two daily incidence streams of one SIR, which one wide file can carry.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95]
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
  recoveries {
    columns       { time : time, recoveries : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(recovery)
    emit_schedule = every 1 'days
    recoveries ~ poisson(rate = projected)
  }
}
simulate { from = 0 'days  to = 20 'days }
"#;

/// Twenty days of both streams in one table, a hole in each.
fn wide() -> String {
    let mut s = String::from("time\tcases\trecoveries\n");
    for d in 1..=20 {
        let cases = if d == 4 { "NA".to_string() } else { (d * 3).to_string() };
        let rec = if d == 15 { "NA".to_string() } else { (d * 2).to_string() };
        s.push_str(&format!("{d}\t{cases}\t{rec}\n"));
    }
    s
}

const METHOD: &str = "[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\n\
                      iterations = 2\ncooling = 0.7\nstarts = \"from_prior\"\n";

/// A fit file over `data` (the body of `[data]`), with `rw_sd` for beta.
fn fit_toml(data: &str, rw_sd: f64) -> String {
    format!(
        "[model]\ncamdl = \"model.camdl\"\n[data]\n{data}\n[estimate]\n\
         beta = {{ bounds = [0.1, 0.9], rw_sd = {rw_sd}, start = 0.4 }}\n\
         gamma = {{ bounds = [0.01, 0.5], prior = {{ uniform = {{}} }} }}\n\
         [fixed]\nN0 = 1000\nI0 = 5\nrho = 0.5\n{METHOD}"
    )
}

struct Fixture {
    dir: tempfile::TempDir,
    model: ir::Model,
}

/// `src` compiled by the camdlc this checkout built, in a directory that
/// holds the wide table.
fn compiled(src: &str) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.camdl"), src).unwrap();
    std::fs::write(dir.path().join("wide.tsv"), wide()).unwrap();
    let camdlc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing at {}: run `make build-ocaml`", camdlc.display());
    let out = std::process::Command::new(&camdlc).arg(dir.path().join("model.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir_path = dir.path().join("model.ir.json");
    std::fs::write(&ir_path, &out.stdout).unwrap();
    let model = crate::util::load_ir(ir_path.to_str().unwrap()).unwrap().0;
    Fixture { dir, model }
}

fn fixture() -> Fixture {
    compiled(MODEL)
}

impl Fixture {
    fn config(&self, text: &str) -> FitConfig {
        let path = self.dir.path().join("fit.toml");
        std::fs::write(&path, text).unwrap();
        FitConfig::parse_anchored_at(text, &path, &path).unwrap()
    }
}

struct Opened {
    _root: tempfile::TempDir,
    store: Store,
    info: RunInfo,
}

fn open() -> Opened {
    let root = tempfile::tempdir().unwrap();
    let store = Store::open(
        &root.path().join("camdl-store"),
        crate::camdl_store::ambient(),
        crate::camdl_store::camdl_top_level(),
        crate::camdl_store::camdl_kinds(),
    )
    .unwrap();
    let info = RunInfo::new(&Given::default(), &(), &StoreRequest::default()).unwrap();
    Opened { _root: root, store, info }
}

fn model_dep(o: &Opened, model: &ir::Model) -> Dep<CompiledModel> {
    let envelope = ir::IrEnvelope::wrap(model.clone(), None, declared::DeclaredSection::not_recorded());
    let parts = crate::camdl_store::CompiledParts::split(
        envelope,
        "model".into(),
        crate::camdl_store::ModelSource { bytes: MODEL.as_bytes(), path: Path::new("model.camdl") },
    )
    .unwrap();
    o.store.put::<CompiledModel>(parts.model, &parts.labels, &parts.provenance, &o.info).unwrap().dep(CompiledModel::FILE).unwrap()
}

fn try_resolved(fx: &Fixture, o: &Opened, config: &FitConfig) -> Result<ResolvedProblem, String> {
    try_resolved_as(fx, o, config, DataOrigin::Files)
}

fn try_resolved_as(fx: &Fixture, o: &Opened, config: &FitConfig, origin: DataOrigin) -> Result<ResolvedProblem, String> {
    resolve_problem(
        &config.problem,
        &fx.model,
        model_dep(o, &fx.model),
        ModelSourceFile { bytes: MODEL.as_bytes().to_vec(), camdl: true },
        origin,
    )
}

/// A problem posed on the data files `config` names.
fn real_data(fx: &Fixture, o: &Opened, config: &FitConfig) -> ResolvedProblem {
    try_resolved(fx, o, config).unwrap()
}

fn resolved(fx: &Fixture, o: &Opened, config: &FitConfig) -> ResolvedProblem {
    try_resolved(fx, o, config).unwrap()
}

fn store_problem(o: &Opened, problem: &ResolvedProblem) -> Stored<FitProblem> {
    let labels = ProblemLabels { model: "model".into(), sweep_point: None };
    match crate::camdl_store::begin::<FitProblem>(&o.store, &problem.identity, &labels, &o.info, Reuse::IfCompleted, problem_files(problem)).unwrap() {
        Begin::Claimed(mut run) => {
            write_problem(&mut run, problem).unwrap();
            run.finish().unwrap()
        }
        Begin::Reused(stored) => stored,
        Begin::Busy(_) => panic!("busy"),
    }
}

fn run_identity(problem: &Stored<FitProblem>, config: &FitConfig) -> (FitRunIdentity, RunLabels) {
    let method = config.inference.method.as_ref().unwrap();
    let (settings, length) = MethodIdentity::of(method).unwrap();
    let (_, perturbation) = split_estimate(&config.problem.estimate).unwrap();
    let labels = RunLabels { method: method.algorithm.method_name().to_string(), seed: 1, quantities: None };
    let chain = ChainIdentity {
        problem: problem.dep_all(),
        method: settings,
        perturbation,
        jacobian_model: None,
        seed: ProcessSeed { seed: 1 },
        starts: StartSource::of(method, StartInput::None).unwrap(),
    };
    (FitRunIdentity { chain, length, resumed_from: None }, labels)
}

fn store_run(o: &Opened, problem: &Stored<FitProblem>, config: &FitConfig) -> Stored<FitRun> {
    let (id, labels) = run_identity(problem, config);
    match crate::camdl_store::begin::<FitRun>(&o.store, &id, &labels, &o.info, Reuse::IfCompleted, [crate::result_files::fit_run::FitRunFile::Common(crate::result_files::fit_run::CommonFile::Method)]).unwrap() {
        Begin::Claimed(mut run) => {
            write_method(&mut run, config.inference.method.as_ref().unwrap()).unwrap();
            run.finish().unwrap()
        }
        _ => panic!("the run is new"),
    }
}

/// Each stream's rows as `problem` binds them, cut to the first `prefix`
/// rows of each when given.
fn bound_rows(model: &ir::Model, problem: &Problem, prefix: Option<&TrainingPartitionRecord>) -> BTreeMap<String, String> {
    let data = problem.data.as_ref().unwrap();
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data).unwrap();
    let opts = time_opts(model, 1.0);
    bindings
        .leaves()
        .map(|(leaf, source)| {
            let obs = model.observations.iter().find(|o| o.name == leaf).unwrap();
            let siblings: Vec<_> = model.observations.iter().filter(|o| o.source == source).collect();
            let mut rows =
                crate::fit::runner::load_observations(&bindings.training_by_source()[source], obs, &siblings, &opts)
                    .unwrap();
            if let Some(p) = prefix {
                rows.truncate(p.streams.iter().find(|s| s.stream == leaf).unwrap().rows);
            }
            (leaf.to_string(), format!("{rows:?}"))
        })
        .collect()
}

/// A wide table bound to every stream is stored once per source it is bound
/// to, cut to the training lines, and the rebuilt problem binds each stream to
/// exactly the rows it trained on: the same rows, holes and positions
/// included, as the wide table cut by the partition.
#[test]
fn a_wide_table_stored_per_stream_binds_identical_rows() {
    let fx = fixture();
    let o = open();
    for data in ["file = \"wide.tsv\"\nholdout_after = 10", "file = \"wide.tsv\""] {
        let config = fx.config(&fit_toml(data, 0.02));
        let problem = resolved(&fx, &o, &config);
        let stored = store_problem(&o, &problem);
        let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
        let rd = fit.config.problem.data.as_ref().unwrap();
        assert!(rd.file.is_none() && rd.holdout_after.is_none() && rd.holdout.is_none(), "{data}");
        assert_eq!(rd.observations.keys().collect::<Vec<_>>(), ["cases", "recoveries"], "{data}");
        let scratch = tempfile::tempdir().unwrap();
        let rebuilt = fit.bind_in(scratch.path()).unwrap();
        let original = bound_rows(&fx.model, &config.problem, Some(&fit.partition));
        assert_eq!(bound_rows(&fx.model, &rebuilt, None), original, "{data}");
        let cases = String::from_utf8(fit.training[0].1.bytes.clone()).unwrap();
        if data.contains("holdout_after") {
            assert_eq!(original["cases"].matches("position").count(), 10, "ten training rows");
            assert_eq!(cases.lines().count(), 11, "the header and the ten training lines: {cases}");
        } else {
            assert_eq!(cases, wide(), "with nothing held out the stored file is the source's bytes");
        }
    }
}

/// A long-form file sorted by stratum rather than by time: each stratum
/// trains on its first rows, which are not the file's first lines. The stored
/// file keeps each stratum's training lines, and binds each stratum to
/// exactly its training rows.
#[test]
fn a_long_form_file_sorted_by_stratum_binds_identical_rows() {
    let ir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden/sir_two_patch_long_obs.ir.json");
    let model = crate::util::load_ir(ir.to_str().unwrap()).unwrap().0;
    let dir = tempfile::tempdir().unwrap();
    let mut long = String::from("time\tpatch\tcases\n");
    for patch in ["urban", "rural"] {
        for (i, t) in [7, 14, 21, 28].iter().enumerate() {
            long.push_str(&format!("{t}\t{patch}\t{}\n", 10 + i));
        }
    }
    std::fs::write(dir.path().join("long.tsv"), &long).unwrap();
    let fx = Fixture { dir, model };
    let config = fx.config(
        "[model]\ncamdl = \"model.camdl\"\n[data]\nholdout_after = 14\n[data.observations]\ncases = \"long.tsv\"\n\
         [estimate]\nbeta = { bounds = [0.01, 0.5] }\n[fixed]\ngamma = 0.1\nrho = 0.6\nk = 5.0\n",
    );
    let o = open();
    let stored = store_problem(&o, &resolved(&fx, &o, &config));
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    assert!(fit.partition.streams.iter().all(|p| p.rows == 2), "each stratum trains on two rows");
    let kept = String::from_utf8(fit.training[0].1.bytes.clone()).unwrap();
    assert_eq!(kept, "time\tpatch\tcases\n7\turban\t10\n14\turban\t11\n7\trural\t10\n14\trural\t11\n");
    let scratch = tempfile::tempdir().unwrap();
    let rebuilt = fit.bind_in(scratch.path()).unwrap();
    assert_eq!(bound_rows(&fx.model, &rebuilt, None), bound_rows(&fx.model, &config.problem, Some(&fit.partition)));
}

/// Streams that read one source but train on different numbers of its lines
/// cannot be stored with only training rows: here `recoveries` reads source
/// `cases` and measures the day that opens at its time, so its row at the
/// cutoff is held out while `cases`'s row on that line trains. The problem is
/// refused, naming it. Two sources bound to one file are stored apart and
/// are not refused.
#[test]
fn streams_of_one_file_training_on_different_lines_are_refused() {
    let opening = MODEL.replace(
        "covers        = closing_at(time, 1 'days)\n    projected     = incidence(recovery)",
        "covers        = starting_on(time, 1 'days)\n    projected     = incidence(recovery)",
    );
    assert_ne!(opening, MODEL);
    let o = open();
    let apart = compiled(&opening);
    let config = apart.config(&fit_toml("file = \"wide.tsv\"\nholdout_after = 10", 0.02));
    let stored = store_problem(&o, &resolved(&apart, &o, &config));
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    let rows: Vec<usize> = fit.partition.streams.iter().map(|p| p.rows).collect();
    assert_eq!(rows, [10, 9], "the two streams train on different numbers of rows");
    let fx = compiled(&opening.replace("  recoveries {", "  recoveries from cases {"));
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"\nholdout_after = 10", 0.02));
    let err = try_resolved(&fx, &o, &config).err().expect("refused");
    assert!(err.contains("train on different numbers of its lines") && err.contains("'recoveries'"), "{err}");
}

/// The held-out rows are stored as dates without values: the file, the
/// position, the label and the period of each row past the cutoff.
#[test]
fn held_out_rows_are_stored_without_their_values() {
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"\nholdout_after = 10", 0.02));
    let stored = store_problem(&o, &resolved(&fx, &o, &config));
    let text = String::from_utf8(read_stored(&stored, &held_out_file("cases")).unwrap()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "file\tposition\tlabel\tmeasures\tstart\tend");
    assert_eq!(lines.len(), 11, "ten held-out rows: {text}");
    assert_eq!(lines[1], "training\t10\t11\tperiod\t10\t11");
    assert!(!text.contains("33"), "no value of the stream is stored: {text}");
    let none = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    assert!(resolved(&fx, &o, &none).identity.held_out.is_empty(), "nothing is held out without a holdout");
}

/// The tree is the proposal's: the problem under `fit/`, named by the model;
/// the run inside it under `runs/`, grouped by the method and named by the
/// seed typed; a posterior prediction inside the run under `predict/`.
#[test]
fn the_tree_is_the_layout() {
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let problem = store_problem(&o, &resolved(&fx, &o, &config));
    let run = store_run(&o, &problem, &config);
    let root = o.store.root().to_path_buf();
    let rel = |p: &Path| p.strip_prefix(&root).unwrap().components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>();
    let named = |dir: &str, label: &str| {
        let (l, hex) = dir.rsplit_once('-').unwrap();
        l == label && hex.len() == 12 && hex.chars().all(|c| c.is_ascii_hexdigit())
    };
    let p = rel(problem.folder());
    assert_eq!(p.len(), 3, "{p:?}");
    assert!(p[0].starts_with(crate::version::VERSION_SHORT.replace('+', "_").as_str()), "{p:?}");
    assert_eq!(p[1], "fit");
    assert!(named(&p[2], "model"), "{p:?}");
    let r = rel(run.folder());
    assert_eq!(&r[..3], &p[..], "{r:?}");
    assert_eq!(r[3], "runs");
    assert!(named(&r[4], "if2") && named(&r[5], "seed_1"), "{r:?}");
    for f in ["problem.toml", "partition.json", "inputs/cases.tsv", "inputs/recoveries.tsv", "model.camdl"] {
        assert!(problem.folder().join(f).is_file(), "{f}");
    }
    assert!(run.folder().join(METHOD_FILE).is_file());

    let id = PredictIdentity {
        source: ParameterSource::Posterior { draws: run.dep_all(), problem: problem.dep_all() },
        data: BTreeMap::new(),
        quantities: None,
        options: PredictOptions { mode: PredictMode::Forecast, settings: "{\"reach\":\"7 days\"}".into(), design: BTreeMap::new() },
    };
    let labels = PredictionLabels { mode: PredictMode::Forecast.label().into(), scenarios: Vec::new(), model: "model".into() };
    let prediction = match crate::camdl_store::begin::<Prediction>(&o.store, &id, &labels, &o.info, Reuse::IfCompleted, crate::result_files::fit::PredictionFile::parse("predictive.json")).unwrap() {
        Begin::Claimed(run) => {
            run.result_file("predictive.json").unwrap().write_all(b"{}").unwrap();
            run.finish().unwrap()
        }
        _ => panic!("the prediction is new"),
    };
    let q = rel(prediction.folder());
    assert_eq!(&q[..6], &r[..], "{q:?}");
    assert_eq!(q[6], "predict");
    assert!(named(&q[7], "forecast"), "{q:?}");
}

/// A prediction declares the tables its manifests list: a stream table the
/// predictive manifest lists and `camdl predict` never wrote is refused as
/// missing when the prediction is stored; a predictive table no manifest
/// lists is left out of the set, and so refused; the forecast's produced
/// streams and its samples file are declared. The quantities manifest lists
/// one table per quantity it names, under `quantities-model/` for the model's
/// own block and under the manifest's own name for a vocabulary's; the
/// contrasts manifest lists its files. A quantity or contrast table no
/// manifest lists is refused, and one listed and never written is missing.
#[test]
fn a_prediction_declares_the_tables_its_manifests_list() {
    use crate::result_files::fit::PredictionFile;
    use crate::result_files::ResultFile;
    let scratch = tempfile::tempdir().unwrap();
    let write = |name: &str, text: &str| {
        let p = scratch.path().join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    };
    write("predictive.json", r#"{"streams": [{"file": "predictive/cases.tsv"}, {"file": "predictive/deaths.tsv"}], "skipped": ["hosp"]}"#);
    write("predictive/cases.tsv", "t\n");
    write("predictive/hosp.tsv", "t\n");
    write("forecast.json", r#"{"produced": ["cases"], "samples": {"file": "forecast-samples.tsv"}}"#);
    write("forecast/cases.tsv", "t\n");
    write("forecast-samples.tsv", "t\n");
    write("contrasts.json", r#"{"contrasts": [{"name": "averted", "file": "contrasts/averted.tsv"}]}"#);
    write("contrasts/averted.tsv", "t\n");
    write("contrasts/stray.tsv", "t\n");
    write("quantities.json", r#"{"quantities": [{"name": "peak", "scenario": "a"}, {"name": "peak", "scenario": "b"}, {"name": "onset"}]}"#);
    write("quantities-model/peak.tsv", "t\n");
    write("quantities-model/stray.tsv", "t\n");
    write("quantities-ab12cd34.json", r#"{"quantities": [{"name": "size"}], "vocabulary": {"file": "v.toml"}}"#);
    write("quantities-ab12cd34/size.tsv", "t\n");
    write("report.json", "{}");
    let declared: Vec<String> = PredictionFile::of_dir(scratch.path()).unwrap().iter().map(ResultFile::stored).collect();
    assert_eq!(
        declared,
        [
            "contrasts.json", "contrasts/averted.tsv", "forecast-samples.tsv", "forecast.json", "forecast/cases.tsv",
            "predictive.json", "predictive/cases.tsv", "predictive/deaths.tsv", "quantities-ab12cd34.json",
            "quantities-ab12cd34/size.tsv", "quantities-model/onset.tsv", "quantities-model/peak.tsv",
            "quantities.json", "report.json",
        ]
    );

    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let problem = store_problem(&o, &resolved(&fx, &o, &config));
    let run = store_run(&o, &problem, &config);
    let id = PredictIdentity {
        source: ParameterSource::Posterior { draws: run.dep_all(), problem: problem.dep_all() },
        data: BTreeMap::new(),
        quantities: None,
        options: PredictOptions { mode: PredictMode::All, settings: "{}".into(), design: BTreeMap::new() },
    };
    let labels = PredictionLabels { mode: PredictMode::All.label().into(), scenarios: Vec::new(), model: "model".into() };
    let declared = PredictionFile::of_dir(scratch.path()).unwrap();
    let Begin::Claimed(prediction) =
        crate::camdl_store::begin::<Prediction>(&o.store, &id, &labels, &o.info, Reuse::IfCompleted, declared).unwrap()
    else {
        panic!("the prediction is new")
    };
    for name in ["predictive.json", "predictive/cases.tsv", "predictive/hosp.tsv", "forecast.json", "forecast/cases.tsv",
                 "forecast-samples.tsv", "contrasts.json", "contrasts/averted.tsv", "contrasts/stray.tsv",
                 "quantities.json", "quantities-model/peak.tsv", "quantities-model/stray.tsv",
                 "quantities-ab12cd34.json", "quantities-ab12cd34/size.tsv", "report.json"] {
        prediction.result_file(name).unwrap().write_all(b"x").unwrap();
    }
    match prediction.finish() {
        Err(store::Error::FileSetMismatch { mut missing, mut unexpected, .. }) => {
            missing.sort();
            unexpected.sort();
            assert_eq!(missing, ["predictive/deaths.tsv", "quantities-model/onset.tsv"], "a listed table never written");
            assert_eq!(
                unexpected,
                ["contrasts/stray.tsv", "predictive/hosp.tsv", "quantities-model/stray.tsv"],
                "a table no manifest lists"
            );
        }
        other => panic!("{other:?}"),
    }

    // A quantity or contrast table written with no manifest of its part is
    // refused, never declared as written; the model's block under a chain
    // selection reads its tables under `quantities-model-excl…/`.
    for (manifest, table) in [("quantities-excl1.json", "contrasts/averted.tsv"), ("contrasts.json", "quantities-model/peak.tsv")] {
        let d = tempfile::tempdir().unwrap();
        let put = |name: &str, text: &str| {
            let p = d.path().join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        put(manifest, r#"{"quantities": [{"name": "size"}], "contrasts": []}"#);
        if manifest.starts_with("quantities") {
            put("quantities-model-excl1/size.tsv", "t\n");
        }
        put(table, "t\n");
        let err = PredictionFile::of_dir(d.path()).unwrap_err();
        assert!(err.contains(&format!("wrote `{table}` and no")), "{err}");
    }
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("quantities-excl1.json"), r#"{"quantities": [{"name": "size"}]}"#).unwrap();
    let declared: Vec<String> = PredictionFile::of_dir(d.path()).unwrap().iter().map(ResultFile::stored).collect();
    assert_eq!(declared, ["quantities-excl1.json", "quantities-model-excl1/size.tsv"]);
}

/// A completed run that wrote a file its method does not store is refused by
/// name: an MH run's output refuses IF2's `mle_params.toml`. A file no fit
/// run has is opened as an attempt file, outside the manifest, and named, so
/// a failing run can say what it kept.
#[test]
fn a_method_output_refuses_a_file_its_method_does_not_store() {
    use crate::fit::output::{LiveRun, RunWriter as _};
    use crate::result_files::fit_run::{Mh, PlannedMethod};
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let problem = store_problem(&o, &resolved(&fx, &o, &config));
    let (id, labels) = run_identity(&problem, &config);
    let Begin::Claimed(run) = o.store.begin::<FitRun>(&id, &labels, &o.info, Reuse::IfCompleted).unwrap() else {
        panic!("the run is new")
    };
    let live = LiveRun::new(run);
    live.put("draws.tsv", b"chain\tdraw\n").unwrap();
    live.put("mle_params.toml", b"beta = 0.4\n").unwrap();
    let err = live.declare("mh", &PlannedMethod::Mh(Mh), 1, None).unwrap_err();
    assert!(err.contains("the mh method wrote `mle_params.toml`") && err.contains("chain_0/trace"), "{err}");
    drop(live);

    let Begin::Claimed(run) = o.store.begin::<FitRun>(&id, &labels, &o.info, Reuse::IfCompleted).unwrap() else {
        panic!("a failed run is claimed again")
    };
    let live = LiveRun::new(run);
    live.put("draws.tsv", b"chain\tdraw\n").unwrap();
    live.put("scratch.txt", b"x\n").unwrap();
    assert_eq!(live.stray(), vec!["scratch.txt".to_string()], "the file no fit run has is named");
    let folder = live.folder().to_path_buf();
    live.take().unwrap().fail("fitting", "stopped").unwrap();
    assert!(folder.join("draws.tsv").is_file());
    assert_eq!(std::fs::read_to_string(folder.join("scratch.txt")).unwrap(), "x\n", "kept in the failed folder");
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(folder.join("run.json")).unwrap()).unwrap();
    assert_eq!(record["status"], "failed", "{record}");
}

/// Writes into `live` every file of a two-chain MH run whose presence does
/// not depend on the outcome, and `chain_starts.tsv` recording each chain's
/// end.
fn mh_files(live: &crate::fit::output::LiveRun, ends: [&str; 2]) {
    use crate::fit::output::RunWriter as _;
    use crate::result_files::fit_run::{MethodFiles, Mh};
    use crate::result_files::ResultFile as _;
    for f in Mh.files(2) {
        if f.stored() != "chain_starts.tsv" {
            live.put(&f.stored(), b"x\n").unwrap();
        }
    }
    let rows: String = ends.iter().enumerate().map(|(c, e)| format!("{c}\t0\t{e}\tsingle\t0.4\t\t\n")).collect();
    live.put("chain_starts.tsv", format!("# camdl chain_starts\nchain_id\tattempt\tstatus\tsource\tbeta\tess\treason\n{rows}").as_bytes()).unwrap();
}

/// A completed run is stored only with every file its recorded outcome
/// states: a two-chain MH run that wrote no chain's trace is refused at
/// `finish`, naming each chain's missing files; with them it is stored; a
/// chain `chain_starts.tsv` records as refused has its trace and no resume
/// state, and a resume state it wrote is refused.
#[test]
fn a_completed_run_missing_a_chains_files_is_refused() {
    use crate::fit::output::{LiveRun, RunWriter as _};
    use crate::result_files::fit_run::{Mh, PlannedMethod};
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let problem = store_problem(&o, &resolved(&fx, &o, &config));
    let (id, labels) = run_identity(&problem, &config);
    let claim = || match o.store.begin::<FitRun>(&id, &labels, &o.info, Reuse::IfCompleted).unwrap() {
        Begin::Claimed(run) => LiveRun::new(run),
        _ => panic!("the run is claimed"),
    };
    let plan = PlannedMethod::Mh(Mh);

    let live = claim();
    mh_files(&live, ["accepted", "accepted"]);
    live.declare("mh", &plan, 2, None).unwrap();
    match live.take().unwrap().finish() {
        Err(store::Error::FileSetMismatch { missing, unexpected, .. }) => {
            assert_eq!(missing, ["chain_0/resume_state.bin", "chain_0/trace.tsv", "chain_1/resume_state.bin", "chain_1/trace.tsv"]);
            assert!(unexpected.is_empty(), "{unexpected:?}");
        }
        other => panic!("a run with no chain's trace is refused: {other:?}"),
    }

    let chain_files = ["chain_0/trace.tsv", "chain_0/resume_state.bin", "chain_1/trace.tsv", "chain_1/resume_state.bin"];
    let live = claim();
    mh_files(&live, ["accepted", "refused"]);
    for f in chain_files {
        live.put(f, b"x\n").unwrap();
    }
    let err = live.declare("mh", &plan, 2, None).unwrap_err();
    assert!(err.contains("`chain_1/resume_state.bin`"), "a refused chain writes no resume state: {err}");
    drop(live);

    let live = claim();
    mh_files(&live, ["accepted", "refused"]);
    for f in chain_files.iter().filter(|f| **f != "chain_1/resume_state.bin") {
        live.put(f, b"x\n").unwrap();
    }
    live.declare("mh", &plan, 2, None).unwrap();
    let stored = live.take().unwrap().finish().expect("every file the outcome states is there");
    assert!(stored.folder().join("chain_1/trace.tsv").is_file() && !stored.folder().join("chain_1/resume_state.bin").exists());
}

/// A posterior prediction's parameter source encodes as it did when the
/// source had no other variant: an enum writes its variant's name first, so
/// adding `FixedPoint` re-keys no stored posterior prediction. Checked
/// against an enum with the same tag and version and only the posterior
/// variant.
#[test]
fn the_fixed_point_variant_re_keys_no_posterior_prediction() {
    #[derive(store::Identity, Clone, Debug)]
    #[identity(tag = "parameter_source", version = 1)]
    enum PosteriorOnly {
        Posterior { draws: Dep<FitRun>, problem: Dep<FitProblem> },
    }
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let problem = store_problem(&o, &resolved(&fx, &o, &config));
    let run = store_run(&o, &problem, &config);
    let now = ParameterSource::Posterior { draws: run.dep_all(), problem: problem.dep_all() };
    let before = PosteriorOnly::Posterior { draws: run.dep_all(), problem: problem.dep_all() };
    assert_eq!(now.key(), before.key());
}

/// `rw_sd`, `perturb_only_at_t0` and `start` are the run's: two fit files
/// that differ only in `rw_sd` pose one problem and are two runs on it, and
/// a run read back carries its own `rw_sd` and method.
#[test]
fn rw_sd_belongs_to_the_run() {
    let fx = fixture();
    let o = open();
    let a = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let b = fx.config(&fit_toml("file = \"wide.tsv\"", 0.05));
    let (pa, pb) = (resolved(&fx, &o, &a), resolved(&fx, &o, &b));
    assert_eq!(pa.identity.key(), pb.identity.key(), "rw_sd is not the problem's");
    let problem = store_problem(&o, &pa);
    let (ra, _) = run_identity(&problem, &a);
    let (rb, _) = run_identity(&problem, &b);
    assert_ne!(ra.key(), rb.key(), "rw_sd is the run's");
    assert_ne!(ra.chain.perturbation.key(), rb.chain.perturbation.key());

    let run = store_run(&o, &problem, &b);
    let fit = read_fit(&o.store, &Handle::IdPrefix(run.id().key().to_hex())).unwrap();
    assert_eq!(fit.problem.id(), problem.id());
    let beta = &fit.config.problem.estimate["beta"];
    assert_eq!((beta.rw_sd, beta.start), (Some(0.05), Some(0.4)));
    assert!(fit.config.problem.estimate["gamma"].prior.is_some(), "the prior is the problem's");
    let method = fit.config.inference.method.as_ref().unwrap();
    assert_eq!(serde_json::to_value(method).unwrap(), serde_json::to_value(b.inference.method.as_ref().unwrap()).unwrap());

    let by_problem = read_fit(&o.store, &Handle::IdPrefix(problem.id().key().to_hex())).unwrap();
    assert!(by_problem.run.is_none() && by_problem.config.inference.method.is_none());
    assert_eq!(by_problem.config.problem.estimate["beta"].rw_sd, None, "a problem holds no rw_sd");
}

/// A method's length is the run's, outside its settings: a PMMH run of 200
/// and one of 400 iterations share one method group.
#[test]
fn a_method_length_is_outside_its_settings() {
    let parse = |n: u32| -> Method {
        let text = format!(
            "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\n\
             iterations = {n}\nstarts = \"from_prior\"\n"
        );
        toml::from_str(&text).unwrap()
    };
    let (a, la) = MethodIdentity::of(&parse(200)).unwrap();
    let (b, lb) = MethodIdentity::of(&parse(400)).unwrap();
    assert_eq!((la, lb), (200, 400));
    assert_eq!(a.key(), b.key());
    assert_eq!(a.record().algorithm(), "pmmh");
}

/// `[estimate]` keeps the file's order through the stored problem, which
/// camdl reads in order: a set written `gamma` before `beta` reads back in that
/// order, not sorted.
#[test]
fn the_estimated_set_keeps_its_order() {
    let fx = fixture();
    let o = open();
    let text = fit_toml("file = \"wide.tsv\"", 0.02);
    let (head, rest) = text.split_once("beta = ").unwrap();
    let (beta, rest) = rest.split_once('\n').unwrap();
    let (gamma, rest) = rest.split_once('\n').unwrap();
    let swapped = format!("{head}{gamma}\nbeta = {beta}\n{rest}");
    let config = fx.config(&swapped);
    assert_eq!(config.problem.estimate.keys().collect::<Vec<_>>(), ["gamma", "beta"]);
    let stored = store_problem(&o, &resolved(&fx, &o, &config));
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    assert_eq!(fit.config.problem.estimate.keys().collect::<Vec<_>>(), ["gamma", "beta"]);
}

/// A NaN or infinite method setting is refused rather than keyed: JSON writes
/// both as `null`, so `cooling = nan` and `cooling = inf` would share a key.
#[test]
fn a_non_finite_method_setting_is_refused() {
    for cooling in ["nan", "inf"] {
        let text = format!(
            "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\n\
             iterations = 2\ncooling = {cooling}\nstarts = \"from_prior\"\n"
        );
        let method: Method = toml::from_str(&text).unwrap();
        assert!(MethodIdentity::of(&method).is_err(), "cooling = {cooling}");
        assert!(StartSource::of(&method, StartInput::None).is_err(), "cooling = {cooling}");
    }
}

/// A stored fit hands its readers the compile it was posed on and the bytes
/// of each input, read through the store, and no path into its folder.
#[test]
fn a_stored_fit_gives_its_compile_and_its_inputs() {
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"\nholdout_after = 10", 0.02));
    let problem = resolved(&fx, &o, &config);
    let stored = store_problem(&o, &problem);
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    assert_eq!(fit.model.digest(), problem.identity.model.digest());
    assert_eq!(fit.model.id(), problem.identity.model.id());
    let names: Vec<&str> = fit.training.iter().map(|(_, i)| i.name.as_str()).collect();
    assert_eq!(names, ["inputs/cases.tsv", "inputs/recoveries.tsv"]);
    assert_eq!(fit.config.problem.data.as_ref().unwrap().observations["cases"], "inputs/cases.tsv",
        "the config names the stored file; it is not a path into the folder");
    assert_eq!(fit.held_out.keys().collect::<Vec<_>>(), ["cases", "recoveries"]);
}

/// A problem whose data a `[synthetic]` table generated keeps what generated
/// it: the true parameters file, stored and checked, the dataset's one seed
/// and the table's settings, so the problem read back is the one posed. The
/// truth is part of the problem: another truth is another problem.
#[test]
fn a_synthetic_problem_reads_back_with_its_truth() {
    use store::ContentAddressed;
    let fx = fixture();
    let o = open();
    let truth = "beta = 0.4\ngamma = 0.2\n";
    std::fs::write(fx.dir.path().join("truth.toml"), truth).unwrap();
    let with = |seed: u64| {
        fit_toml("file = \"wide.tsv\"", 0.02).replace(
            "[estimate]",
            &format!("[synthetic]\ntrue_params = \"{}\"\nsim_seeds = [{seed}]\n[estimate]",
                fx.dir.path().join("truth.toml").display()),
        )
    };
    let config = fx.config(&with(7));
    let generated = |c: &FitConfig| try_resolved_as(&fx, &o, c, DataOrigin::Generated).unwrap();
    let problem = generated(&config);
    let stored = store_problem(&o, &problem);
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let bound = fit.bind_in(scratch.path()).unwrap();
    let spec = bound.synthetic.as_ref().expect("the [synthetic] table is part of the stored problem");
    assert_eq!(std::fs::read_to_string(&spec.true_params).unwrap(), truth, "the truth is the file's bytes");
    assert_eq!(spec.sim_seeds.to_vec().unwrap(), [7]);
    assert_eq!(spec.backend, config.problem.synthetic.as_ref().unwrap().backend);
    assert!(bound.data.is_some(), "the generated data is the problem's data");

    let other_seed = generated(&fx.config(&with(8)));
    assert_ne!(other_seed.identity.content_hash(), problem.identity.content_hash(), "the seed is the problem's");
    std::fs::write(fx.dir.path().join("truth.toml"), "beta = 0.5\ngamma = 0.2\n").unwrap();
    let other_truth = generated(&fx.config(&with(7)));
    assert_ne!(other_truth.identity.content_hash(), problem.identity.content_hash(), "the truth is the problem's");
}

/// A fit of real data whose file also declares `[synthetic]` (the truth `fit
/// recovery` compares an estimate with) stores the truth with the problem,
/// outside its identity: the problem is the data's, so another truth is the
/// same problem. Any number of `sim_seeds` is accepted, and the problem read
/// back carries the table and the truth.
#[test]
fn a_real_data_fit_keeps_its_recovery_truth_outside_its_identity() {
    use store::ContentAddressed;
    let fx = fixture();
    let o = open();
    let truth = fx.dir.path().join("truth.toml");
    std::fs::write(&truth, "beta = 0.4\ngamma = 0.2\n").unwrap();
    let text = fit_toml("file = \"wide.tsv\"", 0.02).replace(
        "[estimate]",
        &format!("[synthetic]\ntrue_params = \"{}\"\nsim_seeds = [1, 2, 3]\n[estimate]", truth.display()),
    );
    let config = fx.config(&text);
    let problem = real_data(&fx, &o, &config);
    let stored = store_problem(&o, &problem);
    let fit = read_fit(&o.store, &Handle::IdPrefix(stored.id().key().to_hex())).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let bound = fit.bind_in(scratch.path()).unwrap();
    let spec = bound.synthetic.as_ref().expect("the recovery truth is read back with the problem");
    assert_eq!(std::fs::read_to_string(&spec.true_params).unwrap(), "beta = 0.4\ngamma = 0.2\n");
    assert_eq!(spec.sim_seeds.to_vec().unwrap(), [1, 2, 3]);

    std::fs::write(&truth, "beta = 0.5\ngamma = 0.2\n").unwrap();
    let other = real_data(&fx, &o, &fx.config(&text));
    assert_eq!(other.identity.content_hash(), problem.identity.content_hash(), "the truth does not key a real-data fit");
}

/// A compile whose derivative tables differ (what a camdlc with an edited
/// autodiff emits) is another compiled model, and so poses another fit
/// problem: every run of it re-keys. A method that reads no gradient re-keys
/// too, since the problem names the whole compile; that over-keying costs a
/// recompute and is accepted, where reusing a run fitted under other
/// gradients would not be.
#[test]
fn a_change_to_the_derivative_tables_re_keys_the_fit() {
    use ir::deriv::DerivEntry;
    use ir::expr::Expr;
    use store::ContentAddressed;
    let fx = fixture();
    let o = open();
    let config = fx.config(&fit_toml("file = \"wide.tsv\"", 0.02));
    let mut edited = fx.model.clone();
    let t = edited.transitions.first_mut().expect("the fixture has a transition");
    t.rate_grad.insert("beta".to_string(), DerivEntry::Grad(Expr::const_(2.0)));
    let as_problem = |model: &ir::Model| {
        resolve_problem(
            &config.problem,
            model,
            model_dep(&o, model),
            ModelSourceFile { bytes: MODEL.as_bytes().to_vec(), camdl: true },
            DataOrigin::Files,
        )
        .unwrap()
    };
    assert_ne!(model_dep(&o, &edited).id(), model_dep(&o, &fx.model).id(), "another compiled model");
    assert_ne!(as_problem(&edited).identity.content_hash(), as_problem(&fx.model).identity.content_hash(),
        "and so another fit problem");
}
