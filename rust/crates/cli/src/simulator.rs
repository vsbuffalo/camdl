//! The simulator a run integrates with: its forward backend, its integration
//! step, its ODE integrator and its binomial sampler.
//!
//! A [`Simulator`] is built only by [`Simulator::resolve`], from what the
//! model declares ([`ModelSimulate`]), what the caller wrote
//! ([`WrittenSimulator`]) and which of the rules of `docs/camdl-run-spec.md`
//! §1.3.1 applies ([`Basis`]). Its fields are private, so a command that
//! holds one cannot change one setting of it, and a command that needs a
//! simulator cannot build one any other way. The step rule itself is
//! `crate::dt_rule`; this module decides which of its functions applies, and
//! the backend, integrator and sampler beside it.
//!
//! `--integrator` is refused on a backend other than ODE and `--binomial` on a
//! backend other than chain-binomial, under every basis: either flag would
//! select nothing on that backend, yet change the run's identity.

use store::Finite;

use crate::args::types::{ForwardBackend, IntegratorArg};
use crate::fit::fitted_model::FittedSettings;
use ir::model::Integrator;
use sim::rng::BinomialAlgorithm;

/// The simulator a run integrates with. Fields are private and the only
/// constructor is [`Simulator::resolve`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Simulator {
    backend: ForwardBackend,
    dt: Finite,
    /// The full method, tolerances included.
    integrator: Integrator,
    binomial: BinomialAlgorithm,
    source: SimulatorSource,
}

/// Where a resolved simulator came from, for the run's notes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SimulatorSource {
    /// Nothing written: the model's declared step and integrator, or the
    /// defaults.
    Model,
    /// The caller wrote settings the rule accepted, named as written
    /// (`--dt`, `[config] dt`, `--backend`, `--integrator`, `--binomial`).
    Written { by: Vec<&'static str> },
    /// A replay of a stored fit, or of a params file's provenance. `moved`
    /// lists the settings a flag moved off the fit's; when it is not empty,
    /// [`Simulator::resolve`] returns the note naming both simulators.
    Fit { name: String, moved: Vec<Setting> },
}

/// One of the four settings of a simulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Setting {
    Backend,
    Dt,
    Integrator,
    Binomial,
}

impl Setting {
    /// The flag that sets it.
    fn flag(self) -> &'static str {
        match self {
            Setting::Backend => "--backend",
            Setting::Dt => "--dt",
            Setting::Integrator => "--integrator",
            Setting::Binomial => "--binomial",
        }
    }
}

/// What the model declares in `simulate {}`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ModelSimulate<'a> {
    dt: Option<f64>,
    integrator: &'a Integrator,
}

impl<'a> ModelSimulate<'a> {
    /// A model's `simulate {}` block, `model.simulation`.
    pub fn of(simulation: &'a ir::model::SimulationConfig) -> Self {
        ModelSimulate { dt: simulation.dt, integrator: &simulation.integrator }
    }
}

/// What the caller wrote, from flags and a fit file. `None` means not written.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WrittenSimulator {
    /// `--backend`.
    pub backend: Option<ForwardBackend>,
    /// `--dt`.
    pub dt: Option<f64>,
    /// A fit file's `[config] dt`, when the file writes one.
    pub config_dt: Option<f64>,
    /// `--integrator`.
    pub integrator: Option<IntegratorArg>,
    /// `--binomial`.
    pub binomial: Option<BinomialAlgorithm>,
}

/// Which of run spec §1.3.1's three rules applies.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Basis<'a> {
    /// `fit run`, `predict` from a model, `profile`, `pfilter`: a written step
    /// must agree with the model's. `backend` is the command's backend when no
    /// `--backend` is written; `run` names the run in a refusal ("this
    /// prediction"), and `file` the fit file in a step error
    /// (`dt_rule::effective_dt_for_command`).
    #[cfg_attr(not(test), expect(dead_code, reason = "the fitting commands resolve their simulator through `Simulator::resolve` in the next step of docs/dev/proposals/2026-11-24-simulator-resolver.md"))]
    Fitting { backend: ForwardBackend, run: &'static str, file: &'static str },
    /// A stored fit or a params file's provenance: the fit's simulator, and a
    /// flag that differs runs with a note.
    Replay(&'a StoredSimulator),
    /// `simulate` without a fit: a flag overrides the model.
    Exploratory,
}

/// The simulator a fit ran on, as its record or a params file's provenance
/// block states it. `settings` is `None` for a provenance block, which records
/// no `[config]`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoredSimulator {
    /// The fit, as messages name it (`@r0`, a file path).
    pub name: String,
    /// `None` for a fit config with no `[method]`, which names no backend.
    pub backend: Option<ForwardBackend>,
    pub dt: f64,
    pub integrator: Integrator,
    /// `None` when the record states no sampler.
    pub binomial: Option<BinomialAlgorithm>,
    pub settings: Option<FittedSettings>,
}

impl StoredSimulator {
    /// `backend` at `dt`, with the integrator on ODE, for messages.
    pub fn describe(&self) -> String {
        describe(self.backend, self.dt, &self.integrator)
    }
}

fn integrator_name(i: &Integrator) -> &'static str {
    match i {
        Integrator::Rk4 => "rk4",
        Integrator::Rk45 { .. } => "rk45",
    }
}

/// `backend` at `dt`, with the integrator on ODE.
fn describe(backend: Option<ForwardBackend>, dt: f64, integrator: &Integrator) -> String {
    match backend {
        Some(ForwardBackend::Ode) => format!("ode ({}) at dt = {dt}", integrator_name(integrator)),
        Some(b) => format!("{} at dt = {dt}", b.as_str()),
        None => format!("dt = {dt}"),
    }
}

/// The refusal of `--integrator` on a run whose backend has no integrator.
/// `what` names the run ("this replay of @r0", "this simulation").
pub(crate) fn refuse_integrator_off_ode(integrator: Option<IntegratorArg>, backend: ForwardBackend, what: &str) -> Result<(), String> {
    if integrator.is_some() && backend != ForwardBackend::Ode {
        return Err(format!(
            "--integrator selects the ODE integrator, but {what} runs on {}, which has none.\n  \
             Fix: drop --integrator, or pass --backend ode.",
            backend.as_str()
        ));
    }
    Ok(())
}

/// The refusal of `--binomial` on a run whose backend draws no binomials.
fn refuse_binomial_off_chain_binomial(
    binomial: Option<BinomialAlgorithm>,
    backend: ForwardBackend,
    what: &str,
) -> Result<(), String> {
    if binomial.is_some() && backend != ForwardBackend::ChainBinomial {
        return Err(format!(
            "--binomial sets the sampler of the chain-binomial draws, and {what} runs on the {backend} \
             backend, which draws none."
        ));
    }
    Ok(())
}

fn finite_dt(dt: f64) -> Result<Finite, String> {
    Finite::new(dt).map_err(|e| format!("the integration step: {e}"))
}

impl Simulator {
    /// The run's simulator, and the note to print when a replay moved off the
    /// fit's. Refuses a disagreeing step under [`Basis::Fitting`],
    /// `--integrator` off ODE and `--binomial` off chain-binomial under every
    /// basis.
    ///
    /// - **Fitting**: the step is `crate::dt_rule::effective_dt_for_command`
    ///   over `--dt`, `[config] dt` and the model's; the integrator is the
    ///   model's, with `--integrator`'s method; the sampler is `--binomial`,
    ///   else BTPE.
    /// - **Replay**: each written flag replaces its own setting of the fit's
    ///   and nothing else, so the fit's step carries across a `--backend`
    ///   change. A fit that names no backend runs on `--backend`, else
    ///   chain-binomial. A fit's integrator matters only on ODE, so a moved
    ///   integrator is noted only on an ODE replay of an ODE fit.
    /// - **Exploratory**: `--backend`, else chain-binomial; `--dt`, else the
    ///   model's step, else `crate::dt_rule::DEFAULT_DT`; the model's
    ///   integrator with `--integrator`'s method; `--binomial`, else BTPE.
    pub fn resolve(
        model: ModelSimulate,
        written: WrittenSimulator,
        basis: Basis,
    ) -> Result<(Simulator, Option<String>), String> {
        let with_method = |declared: &Integrator| match written.integrator {
            Some(m) => crate::util::overridden_integrator(declared, m),
            None => declared.clone(),
        };
        let binomial = written.binomial.unwrap_or_default();
        match basis {
            Basis::Fitting { backend, run, file } => {
                let backend = written.backend.unwrap_or(backend);
                refuse_integrator_off_ode(written.integrator, backend, run)?;
                refuse_binomial_off_chain_binomial(written.binomial, backend, run)?;
                let dt = crate::dt_rule::effective_dt_for_command(written.dt, written.config_dt, model.dt, file)?;
                let simulator = Simulator {
                    backend,
                    dt: finite_dt(dt)?,
                    integrator: with_method(model.integrator),
                    binomial,
                    source: written.source(),
                };
                Ok((simulator, None))
            }
            Basis::Exploratory => {
                let backend = written.backend.unwrap_or(ForwardBackend::ChainBinomial);
                refuse_integrator_off_ode(written.integrator, backend, "this simulation")?;
                refuse_binomial_off_chain_binomial(written.binomial, backend, "this simulation")?;
                let dt = written.dt.or(model.dt).unwrap_or(crate::dt_rule::DEFAULT_DT);
                let simulator = Simulator {
                    backend,
                    dt: finite_dt(dt)?,
                    integrator: with_method(model.integrator),
                    binomial,
                    source: written.source(),
                };
                Ok((simulator, None))
            }
            Basis::Replay(fit) => {
                let what = format!("this replay of {}", fit.name);
                let backend = written.backend.or(fit.backend).unwrap_or(ForwardBackend::ChainBinomial);
                refuse_integrator_off_ode(written.integrator, backend, &what)?;
                refuse_binomial_off_chain_binomial(written.binomial, backend, &what)?;
                let dt = written.dt.unwrap_or(fit.dt);
                let integrator = with_method(&fit.integrator);
                let binomial = written.binomial.or(fit.binomial).unwrap_or_default();
                let moved: Vec<Setting> = [
                    (fit.backend.is_some_and(|b| b != backend), Setting::Backend),
                    (dt != fit.dt, Setting::Dt),
                    (
                        fit.backend == Some(ForwardBackend::Ode)
                            && backend == ForwardBackend::Ode
                            && integrator != fit.integrator,
                        Setting::Integrator,
                    ),
                    (
                        backend == ForwardBackend::ChainBinomial && binomial != fit.binomial.unwrap_or_default(),
                        Setting::Binomial,
                    ),
                ]
                .into_iter()
                .filter_map(|(on, s)| on.then_some(s))
                .collect();
                let simulator = Simulator {
                    backend,
                    dt: finite_dt(dt)?,
                    integrator,
                    binomial,
                    source: SimulatorSource::Fit { name: fit.name.clone(), moved },
                };
                let note = simulator.replay_note(fit);
                Ok((simulator, note))
            }
        }
    }

    /// The note a replay says when a flag moved it off `fit`'s simulator.
    fn replay_note(&self, fit: &StoredSimulator) -> Option<String> {
        let SimulatorSource::Fit { name, moved } = &self.source else {
            return None;
        };
        if moved.is_empty() {
            return None;
        }
        let flags: Vec<&str> = moved.iter().map(|s| s.flag()).collect();
        Some(format!(
            "note: {name} was fitted on {}; this replay runs on {} ({} as given), so the \
             parameters drive a different simulator from the one they were estimated on",
            fit.describe(),
            self.describe(),
            flags.join(", ")
        ))
    }

    pub fn backend(&self) -> ForwardBackend {
        self.backend
    }

    pub fn dt(&self) -> Finite {
        self.dt
    }

    pub fn binomial(&self) -> BinomialAlgorithm {
        self.binomial
    }

    /// The backend at the step, with the integrator on ODE, and the settings
    /// the caller wrote, for messages: `chain_binomial at dt = 0.5 (--dt)`.
    pub fn describe(&self) -> String {
        let base = describe(Some(self.backend), self.dt.get(), &self.integrator);
        match &self.source {
            SimulatorSource::Written { by } => format!("{base} ({})", by.join(", ")),
            SimulatorSource::Model | SimulatorSource::Fit { .. } => base,
        }
    }
}

impl WrittenSimulator {
    /// The source of a simulator resolved from these settings alone.
    fn source(&self) -> SimulatorSource {
        let by: Vec<&'static str> = [
            (self.backend.is_some(), "--backend"),
            (self.dt.is_some(), "--dt"),
            (self.config_dt.is_some(), "[config] dt"),
            (self.integrator.is_some(), "--integrator"),
            (self.binomial.is_some(), "--binomial"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect();
        if by.is_empty() {
            SimulatorSource::Model
        } else {
            SimulatorSource::Written { by }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CB: ForwardBackend = ForwardBackend::ChainBinomial;
    const ODE: ForwardBackend = ForwardBackend::Ode;

    fn model(dt: Option<f64>, integrator: &Integrator) -> ModelSimulate<'_> {
        ModelSimulate { dt, integrator }
    }

    fn fit(backend: Option<ForwardBackend>, dt: f64) -> StoredSimulator {
        StoredSimulator {
            name: "F".into(),
            backend,
            dt,
            integrator: Integrator::Rk4,
            binomial: None,
            settings: None,
        }
    }

    fn fitting(backend: ForwardBackend) -> Basis<'static> {
        Basis::Fitting { backend, run: "this prediction", file: crate::dt_rule::FIT_FLAG_FILE }
    }

    fn resolve(m: ModelSimulate, w: WrittenSimulator, b: Basis) -> (Simulator, Option<String>) {
        Simulator::resolve(m, w, b).unwrap()
    }

    #[test]
    fn under_fitting_a_written_step_must_agree_with_the_models() {
        let rk4 = Integrator::Rk4;
        let (s, note) = resolve(model(Some(0.5), &rk4), WrittenSimulator::default(), fitting(CB));
        assert_eq!((s.backend(), s.dt().get(), s.binomial(), note), (CB, 0.5, BinomialAlgorithm::Btpe, None));
        assert_eq!(s.source, SimulatorSource::Model);

        let agreeing = WrittenSimulator { dt: Some(0.5), config_dt: Some(0.5), ..Default::default() };
        let (s, _) = resolve(model(Some(0.5), &rk4), agreeing, fitting(CB));
        assert_eq!(s.source, SimulatorSource::Written { by: vec!["--dt", "[config] dt"] });

        let err = Simulator::resolve(
            model(Some(0.5), &rk4),
            WrittenSimulator { dt: Some(0.25), ..Default::default() },
            fitting(CB),
        )
        .unwrap_err();
        assert!(err.contains("--dt = 0.25") && err.contains("dt = 0.5"), "{err}");
        let err = Simulator::resolve(
            model(Some(0.5), &rk4),
            WrittenSimulator { config_dt: Some(0.25), ..Default::default() },
            fitting(CB),
        )
        .unwrap_err();
        assert!(err.contains("[config] dt = 0.25"), "{err}");

        let (s, _) =
            resolve(model(None, &rk4), WrittenSimulator { config_dt: Some(0.25), ..Default::default() }, fitting(CB));
        assert_eq!(s.dt().get(), 0.25);
        let (s, _) = resolve(model(None, &rk4), WrittenSimulator::default(), fitting(CB));
        assert_eq!(s.dt().get(), crate::dt_rule::DEFAULT_DT);
    }

    #[test]
    fn exploratory_simulation_lets_a_flag_override_the_models_step() {
        let rk4 = Integrator::Rk4;
        let (s, _) =
            resolve(model(Some(0.5), &rk4), WrittenSimulator { dt: Some(0.25), ..Default::default() }, Basis::Exploratory);
        assert_eq!((s.backend(), s.dt().get()), (CB, 0.25));
        assert_eq!(s.describe(), "chain_binomial at dt = 0.25 (--dt)");
        let (s, _) = resolve(model(Some(0.5), &rk4), WrittenSimulator::default(), Basis::Exploratory);
        assert_eq!(s.dt().get(), 0.5);
        let (s, _) = resolve(model(None, &rk4), WrittenSimulator::default(), Basis::Exploratory);
        assert_eq!(s.dt().get(), crate::dt_rule::DEFAULT_DT);
    }

    #[test]
    fn a_non_finite_step_is_refused() {
        let rk4 = Integrator::Rk4;
        let w = WrittenSimulator { dt: Some(f64::NAN), ..Default::default() };
        assert!(Simulator::resolve(model(None, &rk4), w, Basis::Exploratory).is_err());
    }

    #[test]
    fn integrator_takes_the_method_and_keeps_the_models_tolerances() {
        let declared = Integrator::Rk45 { atol: Some(1e-8), rtol: Some(1e-6) };
        let w = WrittenSimulator { backend: Some(ODE), integrator: Some(IntegratorArg::Rk45), ..Default::default() };
        let (s, _) = resolve(model(None, &declared), w, Basis::Exploratory);
        assert_eq!(s.integrator, declared);

        let w = WrittenSimulator { backend: Some(ODE), integrator: Some(IntegratorArg::Rk4), ..Default::default() };
        let (s, _) = resolve(model(None, &declared), w, Basis::Exploratory);
        assert_eq!(s.integrator, Integrator::Rk4);

        let w = WrittenSimulator { backend: Some(ODE), integrator: Some(IntegratorArg::Rk45), ..Default::default() };
        let (s, _) = resolve(model(None, &Integrator::Rk4), w, Basis::Exploratory);
        assert_eq!(s.integrator, Integrator::Rk45 { atol: None, rtol: None });

        let (s, _) = resolve(model(None, &declared), WrittenSimulator::default(), Basis::Exploratory);
        assert_eq!(s.integrator, declared, "no flag: the model's own");
    }

    #[test]
    fn integrator_off_ode_and_binomial_off_chain_binomial_are_refused_under_every_basis() {
        let rk4 = Integrator::Rk4;
        let stored = fit(Some(ODE), 1.0);
        let integ = WrittenSimulator { integrator: Some(IntegratorArg::Rk45), ..Default::default() };
        let binom =
            WrittenSimulator { backend: Some(ODE), binomial: Some(BinomialAlgorithm::Btrs), ..Default::default() };
        let cb_integ = WrittenSimulator { backend: Some(CB), ..integ };
        for (w, basis, needle) in [
            (integ, fitting(CB), "this prediction runs on chain_binomial"),
            (integ, Basis::Exploratory, "this simulation runs on chain_binomial"),
            (cb_integ, Basis::Replay(&stored), "this replay of F runs on chain_binomial"),
            (binom, fitting(CB), "this prediction runs on the ode backend"),
            (binom, Basis::Exploratory, "this simulation runs on the ode backend"),
            (binom, Basis::Replay(&stored), "this replay of F runs on the ode backend"),
        ] {
            let err = Simulator::resolve(model(None, &rk4), w, basis).unwrap_err();
            assert!(err.contains(needle), "{err}");
        }
        let ok = WrittenSimulator { binomial: Some(BinomialAlgorithm::Btrs), ..Default::default() };
        let (s, _) = resolve(model(None, &rk4), ok, fitting(CB));
        assert_eq!(s.binomial(), BinomialAlgorithm::Btrs);
    }

    #[test]
    fn a_replay_runs_on_the_fits_simulator_silently() {
        let rk4 = Integrator::Rk4;
        let stored = fit(Some(CB), 0.5);
        let (s, note) = resolve(model(Some(1.0), &rk4), WrittenSimulator::default(), Basis::Replay(&stored));
        assert_eq!((s.backend(), s.dt().get(), note), (CB, 0.5, None));

        let equal = WrittenSimulator { backend: Some(CB), dt: Some(0.5), ..Default::default() };
        let (_, note) = resolve(model(Some(1.0), &rk4), equal, Basis::Replay(&stored));
        assert_eq!(note, None, "a flag equal to the fit's changes nothing");

        let stored_ode = fit(Some(ODE), 1.0);
        let w = WrittenSimulator { integrator: Some(IntegratorArg::Rk4), ..Default::default() };
        let (_, note) = resolve(model(None, &rk4), w, Basis::Replay(&stored_ode));
        assert_eq!(note, None);

        let btrs = StoredSimulator { binomial: Some(BinomialAlgorithm::Btrs), ..fit(Some(CB), 0.5) };
        let (s, note) = resolve(model(None, &rk4), WrittenSimulator::default(), Basis::Replay(&btrs));
        assert_eq!((s.binomial(), note), (BinomialAlgorithm::Btrs, None));
    }

    #[test]
    fn a_differing_flag_runs_the_replay_with_one_note_naming_both_simulators() {
        let rk4 = Integrator::Rk4;
        let stored = fit(Some(CB), 0.5);
        let w = WrittenSimulator { backend: Some(ODE), ..Default::default() };
        let (s, note) = resolve(model(None, &rk4), w, Basis::Replay(&stored));
        assert_eq!((s.backend(), s.dt().get()), (ODE, 0.5), "a --backend change keeps the fit's step");
        let n = note.unwrap();
        assert!(
            n.contains("F was fitted on chain_binomial at dt = 0.5")
                && n.contains("runs on ode (rk4) at dt = 0.5")
                && n.contains("(--backend as given)"),
            "{n}"
        );

        let w = WrittenSimulator { dt: Some(1.0), ..Default::default() };
        let n = resolve(model(None, &rk4), w, Basis::Replay(&stored)).1.unwrap();
        assert!(n.contains("runs on chain_binomial at dt = 1 (--dt as given)"), "{n}");

        let w = WrittenSimulator { integrator: Some(IntegratorArg::Rk45), ..Default::default() };
        let n = resolve(model(None, &rk4), w, Basis::Replay(&fit(Some(ODE), 1.0))).1.unwrap();
        assert!(n.contains("runs on ode (rk45)") && n.contains("--integrator"), "{n}");

        let w = WrittenSimulator { binomial: Some(BinomialAlgorithm::Btrs), ..Default::default() };
        let n = resolve(model(None, &rk4), w, Basis::Replay(&stored)).1.unwrap();
        assert!(n.contains("(--binomial as given)"), "{n}");
    }

    #[test]
    fn a_fit_that_names_no_backend_runs_on_the_flag_or_chain_binomial() {
        let rk4 = Integrator::Rk4;
        let stored = fit(None, 0.25);
        let (s, note) = resolve(model(None, &rk4), WrittenSimulator::default(), Basis::Replay(&stored));
        assert_eq!((s.backend(), s.dt().get(), note), (CB, 0.25, None));
        let w = WrittenSimulator { backend: Some(ForwardBackend::Gillespie), ..Default::default() };
        let (s, note) = resolve(model(None, &rk4), w, Basis::Replay(&stored));
        assert_eq!((s.backend(), note), (ForwardBackend::Gillespie, None));
    }
}
