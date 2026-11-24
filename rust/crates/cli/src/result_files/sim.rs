//! The files of a simulated cell and of its by-products.
//!
//! A cell stores its trajectory, or, when the engine stopped with an error the
//! cell's identity determines, `failure.toml` in its place; a cell whose model
//! has an active reactive policy also stores the policy's firings. Each
//! by-product stores its own files, or `failure.toml` when it could not be
//! written from the cell.

use super::{Files, Format, ResultFile, Role};
use crate::sim_store::{EventLog, Obs, Sim, SimQuantities, FAILURE_FILE, PROVENANCE_FILE, REACTIVE_LOG_FILE, TRAJ_FILE};

/// The files of a simulated cell (`sim`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SimFile {
    /// `traj.tsv`, the trajectory at the output times.
    Traj,
    /// `reactive_log.tsv`, each firing of a reactive policy.
    ReactiveLog,
    /// `failure.toml`, the error the engine stopped with.
    Failure,
    /// `provenance.json`, where each parameter value came from and the
    /// notices of the resolution (provenance).
    Provenance,
}

impl ResultFile for SimFile {
    fn logical(&self) -> String {
        match self {
            SimFile::Traj => "traj",
            SimFile::ReactiveLog => "reactive_log",
            SimFile::Failure => "failure",
            SimFile::Provenance => "provenance",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            SimFile::Traj => TRAJ_FILE,
            SimFile::ReactiveLog => REACTIVE_LOG_FILE,
            SimFile::Failure => FAILURE_FILE,
            SimFile::Provenance => PROVENANCE_FILE,
        }
        .into()
    }
    fn role(&self) -> Role {
        match self {
            SimFile::Provenance => Role::Provenance,
            _ => Role::Result,
        }
    }
    fn format(&self) -> Format {
        match self {
            SimFile::Traj | SimFile::ReactiveLog => Format::Tsv,
            SimFile::Failure => Format::Toml,
            SimFile::Provenance => Format::Json,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            SimFile::Traj => "The trajectory at the output times: `t`, `t_start`, `t_stop`, each compartment and each `flow_<name>`.",
            SimFile::ReactiveLog => "Each firing of a reactive policy.",
            SimFile::Failure => "The operation and the error the engine stopped with.",
            SimFile::Provenance => "Where each parameter value came from, and the notices of the resolution.",
        }
    }
}

impl Files for Sim {
    type File = SimFile;
}

impl SimFile {
    /// The files of a cell before its outcome is known: the trajectory and
    /// the provenance record.
    pub(crate) fn planned() -> Vec<SimFile> {
        vec![SimFile::Traj, SimFile::Provenance]
    }

    /// The files of a cell whose engine ran: the trajectory, the firings
    /// when `reactive` (the run kept a reactive log), and the provenance
    /// record; or, when the engine stopped with an error, `failure.toml` and
    /// the provenance record.
    pub(crate) fn of_outcome(completed: bool, reactive: bool) -> Vec<SimFile> {
        match (completed, reactive) {
            (false, _) => vec![SimFile::Failure, SimFile::Provenance],
            (true, false) => SimFile::planned(),
            (true, true) => vec![SimFile::Traj, SimFile::ReactiveLog, SimFile::Provenance],
        }
    }
}

/// The files of a cell's simulated observations (`obs`): one table per
/// stream, or `failure.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObsFile {
    /// `<stream>.tsv`, one stream's simulated observations.
    Stream(String),
    Failure,
}

impl ResultFile for ObsFile {
    fn logical(&self) -> String {
        match self {
            ObsFile::Stream(s) => s.clone(),
            ObsFile::Failure => "failure".into(),
        }
    }
    fn stored(&self) -> String {
        match self {
            ObsFile::Stream(s) => format!("{s}.tsv"),
            ObsFile::Failure => FAILURE_FILE.into(),
        }
    }
    fn format(&self) -> Format {
        match self {
            ObsFile::Stream(_) => Format::Tsv,
            ObsFile::Failure => Format::Toml,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            ObsFile::Stream(_) => "One stream's simulated observations: `time` and the stream's columns.",
            ObsFile::Failure => "The error that kept the observations from being drawn.",
        }
    }
}

impl Files for Obs {
    type File = ObsFile;
}

/// The files of a cell's event log (`event_log`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EventLogFile {
    /// `event_log.tsv`, every event of the run under the recorder.
    EventLog,
    Failure,
}

impl ResultFile for EventLogFile {
    fn logical(&self) -> String {
        match self {
            EventLogFile::EventLog => "event_log",
            EventLogFile::Failure => "failure",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            EventLogFile::EventLog => EventLog::FILE,
            EventLogFile::Failure => FAILURE_FILE,
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            EventLogFile::EventLog => Format::Tsv,
            EventLogFile::Failure => Format::Toml,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            EventLogFile::EventLog => "Every event of the cell's run under the recorder.",
            EventLogFile::Failure => "The error that kept the event log from being recorded.",
        }
    }
}

impl Files for EventLog {
    type File = EventLogFile;
}

/// The files of a cell's generated quantities (`sim_quantities`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SimQuantitiesFile {
    /// `quantities.json`, the quantities evaluated on the cell.
    Quantities,
    Failure,
}

impl ResultFile for SimQuantitiesFile {
    fn logical(&self) -> String {
        match self {
            SimQuantitiesFile::Quantities => "quantities",
            SimQuantitiesFile::Failure => "failure",
        }
        .into()
    }
    fn stored(&self) -> String {
        match self {
            SimQuantitiesFile::Quantities => SimQuantities::FILE,
            SimQuantitiesFile::Failure => FAILURE_FILE,
        }
        .into()
    }
    fn format(&self) -> Format {
        match self {
            SimQuantitiesFile::Quantities => Format::Json,
            SimQuantitiesFile::Failure => Format::Toml,
        }
    }
    fn about(&self) -> &'static str {
        match self {
            SimQuantitiesFile::Quantities => "The model's quantities evaluated on the cell.",
            SimQuantitiesFile::Failure => "The error that kept the quantities from being evaluated.",
        }
    }
}

impl Files for SimQuantities {
    type File = SimQuantitiesFile;
}
