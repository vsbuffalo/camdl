//! Handles, labels, renderings and exports.

use std::io::Write;

use crate::tests::*;
use crate::record::{InvocationFile, InvocationStatus, Outcome, ResultEntry, RECORD_FORMAT};
use crate::{Begin, Error, ExportPath, Handle, Reuse, RunRef, Stored};

fn resolve_sim(store: &Store, h: &str) -> crate::Result<Stored<Sim>> {
    store.resolve(&h.parse::<RunRef<Sim>>()?)
}

#[test]
fn a_handle_parses_by_its_form() {
    assert_eq!("@base".parse::<Handle>().unwrap(), Handle::Label { name: "base".into(), back: 0 });
    assert_eq!("@base~2".parse::<Handle>().unwrap(), Handle::Label { name: "base".into(), back: 2 });
    assert_eq!("3fa9c1".parse::<Handle>().unwrap(), Handle::IdPrefix("3fa9c1".into()));
    assert_eq!("results/sim".parse::<Handle>().unwrap(), Handle::Path("results/sim".into()));
}

#[test]
fn an_id_prefix_names_one_result_and_an_ambiguous_one_lists_every_match() {
    let root = TempRoot::new("prefix");
    let store = open(root.path());
    let sims: Vec<Stored<Sim>> = (1..=20).map(|s| run_sim(&store, &info(), s)).collect();
    let hex = sims[4].id().key().to_hex();
    assert_eq!(resolve_sim(&store, &hex[..10]).unwrap().id(), sims[4].id());
    let mut by_first: std::collections::BTreeMap<char, usize> = Default::default();
    for s in &sims {
        *by_first.entry(s.id().key().to_hex().chars().next().unwrap()).or_default() += 1;
    }
    let shared = by_first.iter().find(|(_, n)| **n > 1).map(|(c, _)| c.to_string()).unwrap();
    assert!(matches!(resolve_sim(&store, &shared), Err(Error::Ambiguous { .. })));
    assert!(matches!(resolve_sim(&store, "fffffffffffffffffff0"), Err(Error::NotFound { .. })));
    assert!(matches!(store.resolve(&hex[..10].parse::<RunRef<Fit>>().unwrap()), Err(Error::WrongKind { .. })));
}

#[test]
fn a_path_names_its_result_and_a_group_directory_is_not_a_handle() {
    let root = TempRoot::new("path");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    let h = s.folder().display().to_string();
    assert_eq!(resolve_sim(&store, &h).unwrap().id(), s.id());
    let group = s.folder().parent().unwrap().display().to_string();
    assert!(matches!(resolve_sim(&store, &group), Err(Error::NoRecord { .. })));
}

#[test]
fn a_label_names_the_latest_invocation_whose_results_all_completed() {
    let root = TempRoot::new("labels");
    let store = open(root.path());
    let i1 = labelled("base");
    let first = run_sim(&store, &i1, 1);
    assert_eq!(i1.finish(&store).unwrap().took_label_from(), None);
    let i2 = labelled("base");
    let second = run_sim(&store, &i2, 2);
    assert_eq!(i2.finish(&store).unwrap().took_label_from(), Some(1));
    assert_eq!(resolve_sim(&store, "@base").unwrap().id(), second.id());
    assert_eq!(resolve_sim(&store, "@base~1").unwrap().id(), first.id());

    // An invocation with a failed result does not take the name.
    let i3 = labelled("base");
    let (id, labels) = sim(3);
    let _ = store.run::<Sim, String>(&id, &labels, &i3, Reuse::IfCompleted, "simulating", |_| Err("no".into()));
    i3.finish(&store).unwrap();
    assert_eq!(resolve_sim(&store, "@base").unwrap().id(), second.id());

    // A label on several results names the invocation, not one of them.
    let i4 = labelled("grid");
    run_sim(&store, &i4, 4);
    run_sim(&store, &i4, 5);
    i4.finish(&store).unwrap();
    assert!(matches!(resolve_sim(&store, "@grid"), Err(Error::NotOneResult { .. })));
    let inv = store.invocation(&"@grid".parse().unwrap()).unwrap();
    assert_eq!(inv.members::<Sim>().unwrap().len(), 2);
    assert_eq!(store.invocation(&second.folder().display().to_string().parse().unwrap()).unwrap().seq(), 2);
}

fn running_invocation(root: &std::path::Path, seq: u64, label: &str, owner: crate::Owner) {
    let f = InvocationFile {
        format: RECORD_FORMAT,
        seq,
        record: "invocation".into(),
        status: InvocationStatus::Running,
        owner,
        started: crate::Timestamp::now(),
        ended: None,
        label: Some(label.into()),
        argv: vec![],
        given: serde_json::json!({}),
        config_files: vec![],
        results: vec![ResultEntry { kind: "sim".into(), run_id: crate::Key::digest_bytes(b"r"), labels: serde_json::Value::Null, outcome: Outcome::Planned, failure: None }],
        counts: Default::default(),
        progress: None,
    };
    std::fs::create_dir_all(root.join(".invocations")).unwrap();
    std::fs::write(root.join(format!(".invocations/{seq}.json")), serde_json::to_vec(&f).unwrap()).unwrap();
}

#[test]
fn a_label_is_refused_while_an_invocation_carrying_it_runs() {
    let root = TempRoot::new("label-in-use");
    let store = open(root.path());
    let (owner, mut child) = live_owner();
    running_invocation(root.path(), 7, "base", owner.clone());
    let (id, labels) = sim(1);
    match store.begin::<Sim>(&id, &labels, &labelled("base"), Reuse::IfCompleted) {
        Err(Error::LabelInUse { owner: o, running, .. }) => {
            assert_eq!(o, owner);
            assert_eq!(running, Some(crate::Key::digest_bytes(b"r")));
        }
        _ => panic!("a label held by a running invocation is refused"),
    }
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(matches!(store.begin::<Sim>(&id, &labels, &labelled("base"), Reuse::IfCompleted).unwrap(), Begin::Claimed(_)));
    running_invocation(root.path(), 20, "other", dead_owner());
    let i = labelled("other");
    run_sim(&store, &i, 2);
    assert!(i.finish(&store).unwrap().seq() > 20, "sequence numbers follow the highest taken");
}

struct Combined;
impl crate::Rendering for Combined {
    type Of = Sim;
    type Options = bool;
    fn render(members: &[(Stored<Sim>, SimLabels)], header: &bool, out: &mut dyn Write) -> crate::Result<()> {
        if *header {
            writeln!(out, "seed\tbytes").unwrap();
        }
        for (s, l) in members {
            let mut b = Vec::new();
            std::io::Read::read_to_end(&mut s.open("traj.tsv")?, &mut b).unwrap();
            writeln!(out, "{}\t{}", l.seed, b.len()).unwrap();
        }
        Ok(())
    }
}

#[test]
fn a_rendering_combines_an_invocations_results_with_their_labels() {
    let root = TempRoot::new("render");
    let store = open(root.path());
    let i = labelled("pair");
    run_sim(&store, &i, 2);
    run_sim(&store, &i, 1);
    i.finish(&store).unwrap();
    let inv = store.invocation(&"@pair".parse().unwrap()).unwrap();
    let to = ExportPath(root.path().join("out.tsv"));
    inv.render::<Combined>(&store, &true, &to).unwrap();
    let mut screen = Vec::new();
    inv.render_to::<Combined>(&store, &true, &mut screen).unwrap();
    assert_eq!(std::fs::read(&to.0).unwrap(), screen, "the path and the screen agree");
    assert_eq!(String::from_utf8(screen).unwrap(), "seed\tbytes\n2\t8\n1\t8\n", "in the order asked for");

    let i = labelled("broken");
    run_sim(&store, &i, 3);
    let (id, labels) = sim(4);
    let _ = store.run::<Sim, String>(&id, &labels, &i, Reuse::IfCompleted, "simulating", |_| Err("no".into()));
    let inv = i.finish(&store).unwrap();
    assert!(matches!(inv.render_to::<Combined>(&store, &false, &mut Vec::new()), Err(Error::MemberNotCompleted { .. })));

    let s = run_sim(&store, &info(), 5);
    let copy = ExportPath(root.path().join("copy.tsv"));
    s.export("traj.tsv", &copy).unwrap();
    assert_eq!(std::fs::read(&copy.0).unwrap(), b"t\tI\n0\t5\n");
}

#[derive(serde::Serialize, serde::Deserialize, crate::FileInputs, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct ConfigToml {
    #[input(identity)]
    values: std::collections::BTreeMap<String, f64>,
}

#[test]
fn the_runs_a_configuration_file_produced_are_listed_from_the_invocation_records() {
    let root = TempRoot::new("config");
    let store = open(root.path());
    std::fs::write(root.path().join("c.toml"), "values = {}\n").unwrap();
    let (declared, _, _, given, source) =
        crate::InputFile::<ConfigToml>::new(root.path().join("c.toml")).parse().unwrap().into_parts();
    let (values,) = declared.into_parts();
    assert!(values.is_empty());
    let given = crate::Given::default().with_file("config", given, &source);
    let i = RunInfo::new(&given, &(), &StoreRequest::default()).unwrap();
    let s = run_sim(&store, &i, 1);
    i.finish(&store).unwrap();
    assert_eq!(store.runs_of_configuration(source.digest()), vec![s.id().key()]);
    assert!(matches!(store.configuration_not_a_run(&source), Error::ConfigurationNotARun { runs, .. } if runs == vec![s.id().key()]));
}
