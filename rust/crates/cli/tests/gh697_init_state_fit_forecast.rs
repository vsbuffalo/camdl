//! gh#697 — `camdl simulate --init-state fit`: forecast from the paired
//! `(θ_i, X_i(T))` posterior, not from one θ's particle swarm.
//!
//! `--init-state FILE` (gh#641) conditions on where the epidemic *is* but runs
//! at a single θ; `--draws posterior` propagates parameter uncertainty but
//! starts every draw from the model's `init {}` at t = 0. Neither is a
//! forecast. The paired source is the terminal row of each PGAS draw's saved
//! latent path — at the last observation time the smoothing distribution
//! equals the filtering distribution, so that row is a draw from
//! `p(x_T | y_{1:T})` carrying its own θ.
//!
//! These tests shell out to the release binary, so they exercise the real
//! `fit run` → `trajectories.tsv` → `simulate` → CAS path.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing() -> PathBuf {
    let b = binary();
    assert!(
        b.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        b.display()
    );
    b
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
scenarios {
  control_50 {
    label = "halve transmission over the forecast window"
    scale = { beta = 0.5 }
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The last observation time in `DATA` — the forecast origin the paired source
/// must resolve to.
const LAST_OBS: f64 = 56.0;

/// The forecast horizon every arm below runs to, via `--to` (past the model's
/// own `to = 80`, so the run has somewhere to go after the origin).
const HORIZON: &str = "120";

/// `sweeps - burn_in = 40` post-burn-in sweeps per chain with `thin = 1` gives
/// 40 posterior draws per chain; `n_trajectories = 8` saves a path every 5th
/// sweep. The forkable subset is therefore a STRICT subset of the parameter
/// posterior — the 606-saved / 300-forkable shape of a real run, in miniature.
const FIT_TOML: &str = r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }
[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
n_trajectories = 8
"#;

// ── Fixture: one PGAS fit, shared by every test in this binary ───────────────

struct Fixture {
    dir: PathBuf,
    model: PathBuf,
    /// The fit results segment (`results/fits/<stem>-<hash>/`).
    fit: PathBuf,
    /// The stage directory holding `draws.tsv` + `chain_*/trajectories.tsv`.
    stage: PathBuf,
}

fn run_in(bin: &Path, dir: &Path, args: &[String]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let bin = skip_if_missing();
        let dir = std::env::temp_dir().join(format!("camdl_gh697_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
        std::fs::write(dir.join("weekly_cases.tsv"), DATA).unwrap();
        std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();

        let out = run_in(
            &bin,
            &dir,
            &["fit".into(), "run".into(), "fit.toml".into(), "--seed".into(), "1".into()],
        );
        assert!(
            out.status.success(),
            "fit run failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        let fits = dir.join("results").join("fits");
        let fit = std::fs::read_dir(&fits)
            .unwrap_or_else(|e| panic!("read {}: {e}", fits.display()))
            .flatten()
            .map(|e| e.path())
            .find(|p| p.is_dir())
            .expect("one fit segment");
        let stage = find_stage(&fit).expect("a stage leaf holding draws.tsv");

        let model = dir.join("model.camdl");
        Fixture { dir, model, fit, stage }
    })
}

/// The stage leaf under a fit segment: `<fit>/NN-<stage>-<hash>/seed_N-<hash>/`,
/// found by the artifact it holds rather than by its path shape.
fn find_stage(root: &Path) -> Option<PathBuf> {
    if root.join("draws.tsv").is_file() {
        return Some(root.to_path_buf());
    }
    let mut kids: Vec<PathBuf> =
        std::fs::read_dir(root).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    kids.sort();
    kids.iter().find_map(|k| find_stage(k))
}

// ── TSV helpers ──────────────────────────────────────────────────────────────

/// Read a TSV, skipping `#` comment lines: (header, rows).
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let hdr: Vec<String> = lines.next().unwrap().split('\t').map(str::to_string).collect();
    let rows: Vec<Vec<String>> =
        lines.map(|l| l.split('\t').map(str::to_string).collect()).collect();
    (hdr, rows)
}

fn idx(hdr: &[String], name: &str) -> usize {
    hdr.iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("column `{name}` among {hdr:?}"))
}

/// The compartment triple this model's state is: `(S, I, R)`.
type Sir = (i64, i64, i64);

/// `draws.tsv` in file order: `(chain, draw, beta, gamma)`.
fn read_draws(stage: &Path) -> Vec<(usize, usize, f64, f64)> {
    let (hdr, rows) = read_tsv(&stage.join("draws.tsv"));
    let (ci, di) = (idx(&hdr, "chain"), idx(&hdr, "draw"));
    let (bi, gi) = (idx(&hdr, "beta"), idx(&hdr, "gamma"));
    rows.iter()
        .map(|r| {
            (
                r[ci].parse().unwrap(),
                r[di].parse().unwrap(),
                r[bi].parse().unwrap(),
                r[gi].parse().unwrap(),
            )
        })
        .collect()
}

/// The TERMINAL snapshot of every saved latent path, keyed by `(chain, draw)`:
/// `(t, S, I, R)`. This is the object `--init-state fit` must restore.
fn read_terminal_states(stage: &Path) -> BTreeMap<(usize, usize), (f64, Sir)> {
    let mut out: BTreeMap<(usize, usize), (f64, Sir)> = BTreeMap::new();
    for e in std::fs::read_dir(stage).unwrap().flatten() {
        let traj = e.path().join("trajectories.tsv");
        if !traj.is_file() {
            continue;
        }
        let (hdr, rows) = read_tsv(&traj);
        let (ci, di, ti) = (idx(&hdr, "chain"), idx(&hdr, "draw"), idx(&hdr, "time"));
        let (si, ii, ri) = (idx(&hdr, "S"), idx(&hdr, "I"), idx(&hdr, "R"));
        for r in &rows {
            let key: (usize, usize) = (r[ci].parse().unwrap(), r[di].parse().unwrap());
            let t: f64 = r[ti].parse().unwrap();
            let sir: Sir = (
                r[si].parse::<f64>().unwrap() as i64,
                r[ii].parse::<f64>().unwrap() as i64,
                r[ri].parse::<f64>().unwrap() as i64,
            );
            match out.get(&key) {
                Some((prev, _)) if *prev >= t => {}
                _ => {
                    out.insert(key, (t, sir));
                }
            }
        }
    }
    out
}

/// The paired ensemble the CLI must build, derived independently here: every
/// `draws.tsv` row, in file order, that has a saved latent path.
fn expected_pairing(stage: &Path) -> Vec<((usize, usize), (f64, f64), Sir)> {
    let terminal = read_terminal_states(stage);
    read_draws(stage)
        .into_iter()
        .filter_map(|(c, d, beta, gamma)| {
            terminal.get(&(c, d)).map(|(_, sir)| ((c, d), (beta, gamma), *sir))
        })
        .collect()
}

/// A wide `-o` trajectory mirror, grouped by 1-based `draw`: `draw → [(t, SIR)]`.
fn traj_by_draw(path: &Path) -> BTreeMap<usize, Vec<(f64, Sir)>> {
    let (hdr, rows) = read_tsv(path);
    let di = idx(&hdr, "draw");
    let ti = idx(&hdr, "t");
    let (si, ii, ri) = (idx(&hdr, "S"), idx(&hdr, "I"), idx(&hdr, "R"));
    let mut out: BTreeMap<usize, Vec<(f64, Sir)>> = BTreeMap::new();
    for r in &rows {
        out.entry(r[di].parse().unwrap()).or_default().push((
            r[ti].parse().unwrap(),
            (
                r[si].parse::<f64>().unwrap() as i64,
                r[ii].parse::<f64>().unwrap() as i64,
                r[ri].parse::<f64>().unwrap() as i64,
            ),
        ));
    }
    out
}

fn at_time(rows: &[(f64, Sir)], t: f64) -> Option<Sir> {
    rows.iter().find(|(rt, _)| (rt - t).abs() < 1e-9).map(|(_, sir)| *sir)
}

// ── 1. The headline oracle ───────────────────────────────────────────────────

/// A fit-sourced forecast is NOT a free-forward replay. Both arms run the same
/// θ cloud, in the same order, at the same seed, to the same horizon — the ONLY
/// difference is where the trajectory starts. If `--init-state fit` were a
/// no-op, the two would coincide over the whole overlapping window.
#[test]
fn fit_sourced_forecast_differs_from_free_forward_at_the_same_theta_cloud() {
    let bin = skip_if_missing();
    let f = fixture();

    // Arm B: the conditioned forecast. `--draws-out` records the exact θ cloud
    // it used, so arm A can be run on precisely the same rows in the same order.
    let b_traj = f.dir.join("b_forecast.tsv");
    let theta = f.dir.join("b_theta.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
            "--seed".into(),
            "7".into(),
            "-o".into(),
            s(&b_traj),
            "--draws-out".into(),
            s(&theta),
        ],
    );
    assert!(
        out.status.success(),
        "conditioned forecast failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Arm A: the free-forward replay of the SAME θ rows, same seed, same
    // horizon — starting from `init {}` at t = 0.
    let a_traj = f.dir.join("a_freeforward.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--draws".into(),
            s(&theta),
            "--to".into(),
            HORIZON.into(),
            "--seed".into(),
            "7".into(),
            "-o".into(),
            s(&a_traj),
        ],
    );
    assert!(
        out.status.success(),
        "free-forward replay failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let b = traj_by_draw(&b_traj);
    let a = traj_by_draw(&a_traj);
    assert_eq!(a.len(), b.len(), "both arms run the same number of draws");
    assert!(!b.is_empty(), "the forkable subset must be non-empty");

    // The forecast begins at the origin; the replay begins at the model's start.
    let b_min = b[&1].iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min);
    let a_min = a[&1].iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min);
    assert_eq!(b_min, LAST_OBS, "the forecast starts at the last observation time");
    assert_eq!(a_min, 0.0, "the free-forward replay starts at the model's t_start");

    // The substantive difference: at the SHARED time t = last_obs, the
    // conditioned arm sits at the inferred state while the replay sits wherever
    // `init {}` took it. Identical values for every draw would mean the state
    // was never restored.
    //
    // And not just different from the replay — different from `init {}` too:
    // moving `t_start` to the origin WITHOUT restoring the state would also
    // produce two different rows here, and would still be a free-forward run
    // wearing a forecast's window.
    let init_block: Sir = (9990, 10, 0);
    let mut same = 0usize;
    for d in b.keys() {
        let bb = at_time(&b[d], LAST_OBS).expect("forecast has a row at the origin");
        let aa = at_time(&a[d], LAST_OBS).expect("replay has a row at the origin");
        assert_ne!(
            bb, init_block,
            "draw {d} started from `init {{}}` at the forecast origin — the origin \
             state was not restored"
        );
        if aa == bb {
            same += 1;
        }
    }
    assert_eq!(
        same,
        0,
        "every draw's state at t = {LAST_OBS} must differ between the conditioned \
         forecast and the free-forward replay — {same}/{} coincided, which is what \
         a silently-ignored --init-state looks like",
        b.len()
    );
}

// ── 2. The pairing ───────────────────────────────────────────────────────────

/// Draw *i*'s state goes with draw *i*'s θ. A shuffled pairing still produces a
/// plausible cloud, so this is asserted against the fit's own artifacts:
/// `draws.tsv` order restricted to the rows that have a saved path, joined by
/// `(chain, draw)` to the terminal snapshot of that path.
#[test]
fn each_draw_restores_its_own_terminal_state_under_its_own_theta() {
    let bin = skip_if_missing();
    let f = fixture();

    let traj = f.dir.join("pairing.tsv");
    let theta = f.dir.join("pairing_theta.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
            "--seed".into(),
            "3".into(),
            "-o".into(),
            s(&traj),
            "--draws-out".into(),
            s(&theta),
        ],
    );
    assert!(
        out.status.success(),
        "conditioned forecast failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let expected = expected_pairing(&f.stage);
    assert!(
        !expected.is_empty(),
        "the fixture must produce at least one forkable draw"
    );

    let by_draw = traj_by_draw(&traj);
    assert_eq!(
        by_draw.len(),
        expected.len(),
        "one forecast cell per forkable posterior draw"
    );

    // θ side: the draws-out rows are the forkable draws' parameters, in order.
    let (thdr, trows) = read_tsv(&theta);
    let (bi, gi) = (idx(&thdr, "beta"), idx(&thdr, "gamma"));
    assert_eq!(trows.len(), expected.len());

    for (i, ((chain, draw), (beta, gamma), sir)) in expected.iter().enumerate() {
        // X side: cell i's first emitted row IS draw i's terminal latent state.
        let rows = &by_draw[&(i + 1)];
        let got = at_time(rows, LAST_OBS)
            .unwrap_or_else(|| panic!("forecast cell {} has a row at t = {LAST_OBS}", i + 1));
        assert_eq!(
            got,
            *sir,
            "forecast cell {} must restore the terminal state of (chain {chain}, draw \
             {draw}) — a mismatch here is the shuffled pairing this feature exists to \
             prevent",
            i + 1
        );
        // θ side: the same cell's parameters are that same draw's parameters.
        let got_beta: f64 = trows[i][bi].parse().unwrap();
        let got_gamma: f64 = trows[i][gi].parse().unwrap();
        assert!(
            (got_beta - beta).abs() < 1e-9 && (got_gamma - gamma).abs() < 1e-9,
            "forecast cell {} must run at (chain {chain}, draw {draw})'s θ: expected \
             beta={beta} gamma={gamma}, got beta={got_beta} gamma={got_gamma}",
            i + 1
        );
    }
}

// ── 3. The forkable subset is reported, never silently reduced ───────────────

/// Only draws with a saved trajectory can be forked. The fixture deliberately
/// saves fewer paths than it retains draws, and the run must SAY so — a cloud
/// quietly banded over a fifth of the posterior is the failure mode here.
#[test]
fn the_forkable_subset_is_counted_out_loud() {
    let bin = skip_if_missing();
    let f = fixture();

    let n_total = read_draws(&f.stage).len();
    let n_forkable = expected_pairing(&f.stage).len();
    assert!(
        n_forkable < n_total,
        "fixture precondition: the saved-path subset must be strict \
         ({n_forkable} of {n_total})"
    );

    let traj = f.dir.join("subset.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
            "-o".into(),
            s(&traj),
        ],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains(&format!("{n_forkable}/{n_total}")),
        "the run must report the forkable subset as `{n_forkable}/{n_total}`; stderr was:\n{err}"
    );
    assert!(
        err.contains("saved latent path"),
        "the report must say what makes a draw forkable; stderr was:\n{err}"
    );
}

// ── 4. Composition: the scenario overlay and the anchored horizon ────────────

/// The actual forecast invocation: the fit supplies the paired origin, `--to
/// "last_obs + N"` supplies the horizon (gh#626), and `--scenario` overlays the
/// counterfactual. All three at once, and the scenario must still bite.
#[test]
fn composes_with_a_scenario_overlay_and_an_anchored_horizon() {
    let bin = skip_if_missing();
    let f = fixture();

    let arm = |name: &str, scenario: Option<&str>| -> BTreeMap<usize, Vec<(f64, Sir)>> {
        let traj = f.dir.join(format!("compose_{name}.tsv"));
        let mut args: Vec<String> = vec![
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            "last_obs + 8 weeks".into(),
            "--seed".into(),
            "5".into(),
            "-o".into(),
            s(&traj),
        ];
        if let Some(sc) = scenario {
            args.push("--scenario".into());
            args.push(sc.into());
        }
        let out = run_in(&bin, &f.dir, &args);
        assert!(
            out.status.success(),
            "compose arm {name} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        traj_by_draw(&traj)
    };

    let base = arm("baseline", None);
    let ctrl = arm("control", Some("control_50"));

    // The window: origin from the paired source, horizon from the anchor.
    let times: Vec<f64> = base[&1].iter().map(|(t, _)| *t).collect();
    let tmin = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(tmin, LAST_OBS, "origin from --init-state fit");
    assert_eq!(tmax, LAST_OBS + 56.0, "horizon from --to \"last_obs + 8 weeks\"");

    // Both arms start from the same restored states (the scenario patches θ,
    // not X) and diverge afterwards — the overlay is live on the forecast.
    for d in base.keys() {
        assert_eq!(
            at_time(&base[d], LAST_OBS),
            at_time(&ctrl[d], LAST_OBS),
            "draw {d}: both arms fork from the same inferred state"
        );
    }
    let diverged = base
        .keys()
        .filter(|d| at_time(&base[d], tmax) != at_time(&ctrl[d], tmax))
        .count();
    assert!(
        diverged > 0,
        "halving beta over the whole forecast window must change the horizon \
         state for at least one draw — a scenario that changes nothing means the \
         overlay was dropped"
    );
}

// ── 5. Negative control: the single-θ file path is untouched ─────────────────

/// `--init-state FILE` is still one θ's particle swarm, and still refuses
/// `--draws`. The refusal now points at the paired source instead of naming a
/// blocker that has landed.
#[test]
fn a_state_file_still_refuses_draws_and_names_the_paired_source() {
    let bin = skip_if_missing();
    let f = fixture();

    // A `pfilter --save-final-state` file: p(x_T | y_{1:T}) at ONE θ.
    let params = f.dir.join("point.toml");
    std::fs::write(
        &params,
        "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n",
    )
    .unwrap();
    let state = f.dir.join("final_state.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "pfilter".into(),
            s(&f.model),
            "--params".into(),
            s(&params),
            format!("--data={}", s(&f.dir.join("weekly_cases.tsv"))),
            "--particles".into(),
            "8".into(),
            "--seed".into(),
            "1".into(),
            "--save-final-state".into(),
            s(&state),
        ],
    );
    assert!(out.status.success(), "pfilter: {}", String::from_utf8_lossy(&out.stderr));

    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            s(&state),
            "--to".into(),
            HORIZON.into(),
        ],
    );
    assert!(!out.status.success(), "a state FILE crossed with --draws must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--init-state fit"),
        "the refusal must name the paired source that IS coherent; stderr was:\n{err}"
    );
}

/// `--draws posterior` without `--init-state` is unchanged: a free-forward
/// replay of the whole parameter posterior from the model's own start.
#[test]
fn draws_without_init_state_still_free_forwards_from_t_start() {
    let bin = skip_if_missing();
    let f = fixture();

    let traj = f.dir.join("plain_draws.tsv");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "-n".into(),
            "4".into(),
            "--to".into(),
            HORIZON.into(),
            "-o".into(),
            s(&traj),
        ],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let by_draw = traj_by_draw(&traj);
    assert_eq!(by_draw.len(), 4, "the whole parameter posterior is available, capped at -n");
    for (d, rows) in &by_draw {
        let tmin = rows.iter().map(|(t, _)| *t).fold(f64::INFINITY, f64::min);
        assert_eq!(tmin, 0.0, "draw {d} must free-forward from the model's t_start");
        assert_eq!(
            at_time(rows, 0.0).unwrap(),
            (9990, 10, 0),
            "draw {d} must start from `init {{}}`"
        );
    }
}

/// A fit whose stage saved no latent paths cannot supply a paired origin. It
/// must be refused BY NAME — never quietly fall back to `init {}`, which would
/// look like a forecast and be a free-forward replay.
///
/// The classification is by ARTIFACT, not by method name (`fit::joint`), so the
/// control is built by removing the artifact: a copy of the fit with its
/// `chain_*/trajectories.tsv` deleted is exactly the PMMH / particle-filter
/// case, deterministically and without a second fit.
#[test]
fn a_fit_with_no_saved_trajectories_refuses_by_name() {
    let bin = skip_if_missing();
    let f = fixture();

    let bare = f.dir.join("fit_no_paths");
    let _ = std::fs::remove_dir_all(&bare);
    copy_tree(&f.fit, &bare);
    let saved = find_all(&bare, "trajectories.tsv");
    assert!(!saved.is_empty(), "the fixture fit must have had saved paths to remove");
    for traj in saved {
        std::fs::remove_file(&traj).unwrap();
    }

    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&bare),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
        ],
    );
    assert!(
        !out.status.success(),
        "a fit with no saved latent paths must be refused, not silently free-forwarded"
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("no forkable posterior draws") && err.contains("trajectories.tsv"),
        "the refusal must name the missing artifact; stderr was:\n{err}"
    );
}

/// `--init-state fit` without the posterior it pairs against is refused: there
/// is no θ_i to put with X_i.
#[test]
fn fit_source_without_draws_posterior_is_refused() {
    let bin = skip_if_missing();
    let f = fixture();

    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
        ],
    );
    assert!(!out.status.success(), "--init-state fit without --draws posterior must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--draws posterior"),
        "the refusal must name the flag that completes the pairing; stderr was:\n{err}"
    );
}

// ── 6. Identity ──────────────────────────────────────────────────────────────

/// Two different paired ensembles must not share a store leaf. The state a cell
/// restores is part of what it computed, so it keys the run: re-running the same
/// command hits the cache, while a run whose restored states changed misses it.
#[test]
fn a_changed_paired_ensemble_re_keys_every_cell() {
    let bin = skip_if_missing();
    let f = fixture();

    let root_a = f.dir.join("cas_a");
    let args = |root: &Path| -> Vec<String> {
        vec![
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&f.fit),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
            "--seed".into(),
            "11".into(),
            "--output-dir".into(),
            s(root),
        ]
    };
    let out = run_in(&bin, &f.dir, &args(&root_a));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let first = sim_leaf_ids(&root_a);
    assert!(!first.is_empty(), "the run must commit sim leaves");

    // Same command, same store → every cell is a cache hit (identical ids).
    let out = run_in(&bin, &f.dir, &args(&root_a));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(sim_leaf_ids(&root_a), first, "an unchanged ensemble must re-key to itself");

    // Perturb ONE restored state in the fit's saved paths and re-run into a
    // fresh store: the ensemble digest changes, so no cell may reuse an id.
    let bumped = f.dir.join("fit_bumped");
    let _ = std::fs::remove_dir_all(&bumped);
    copy_tree(&f.fit, &bumped);
    let traj = find_all(&bumped, "trajectories.tsv")
        .into_iter()
        .next()
        .expect("a saved path to perturb");
    {
        let txt = std::fs::read_to_string(&traj).unwrap();
        let mut lines: Vec<String> = txt.lines().map(str::to_string).collect();
        let hdr_pos = lines.iter().position(|l| !l.starts_with('#')).unwrap();
        let hdr: Vec<String> = lines[hdr_pos].split('\t').map(str::to_string).collect();
        let si = idx(&hdr, "S");
        let last = lines.len() - 1;
        let mut fields: Vec<String> = lines[last].split('\t').map(str::to_string).collect();
        let v: f64 = fields[si].parse().unwrap();
        fields[si] = format!("{}", v + 1.0);
        lines[last] = fields.join("\t");
        std::fs::write(&traj, lines.join("\n") + "\n").unwrap();
    }

    let root_b = f.dir.join("cas_b");
    let out = run_in(
        &bin,
        &f.dir,
        &[
            "simulate".into(),
            s(&f.model),
            "--fit".into(),
            s(&bumped),
            "--draws".into(),
            "posterior".into(),
            "--init-state".into(),
            "fit".into(),
            "--to".into(),
            HORIZON.into(),
            "--seed".into(),
            "11".into(),
            "--output-dir".into(),
            s(&root_b),
        ],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let second = sim_leaf_ids(&root_b);
    assert!(!second.is_empty());
    for id in &second {
        assert!(
            !first.contains(id),
            "a perturbed paired ensemble must re-key EVERY cell — run_id {id} was \
             reused, so the store would serve a stale forecast"
        );
    }
}

// ── shared helpers ───────────────────────────────────────────────────────────

/// Every file named `name` under `root`, in sorted path order.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(p: &Path, name: &str, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, name, out);
                } else if path.file_name().is_some_and(|n| n == name) {
                    out.push(path);
                }
            }
        }
    }
    walk(root, name, &mut out);
    out.sort();
    out
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let from = e.path();
        let to = dst.join(e.file_name());
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

/// Every `sim`-kind leaf's `run_id` under a CAS root.
fn sim_leaf_ids(root: &Path) -> Vec<String> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.file_name().is_some_and(|n| n == "run.json") {
                    out.push(path);
                }
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    let mut out: Vec<String> = files
        .iter()
        .filter_map(|f| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
            (v["kind"] == "sim").then(|| v["run_id"].as_str().unwrap().to_string())
        })
        .collect();
    out.sort();
    out.dedup();
    out
}
