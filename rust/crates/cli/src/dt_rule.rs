//! The one integration-step rule, for every backend and every command that
//! scores data against a model (`fit run`, `fit predict`, `camdl profile`,
//! `camdl pfilter`).
//!
//! The model's declared step (`simulate { dt = … }`) is the integration step.
//! A step written by the caller — a fit-config `[config] dt`, a `--dt` flag —
//! must agree with it, or it is refused: a written value that disagrees would
//! otherwise be silently replaced, and the run would not be the one asked for.
//! When the model declares no step, the written value is the step, and
//! [`DEFAULT_DT`] is the step when neither does.
//!
//! `camdl simulate --dt` is not under this rule: exploratory simulation fits
//! nothing, and its flag keeps overriding the model's step.

/// The step when neither the model nor the caller declares one.
pub const DEFAULT_DT: f64 = 1.0;

/// A written step that disagrees with the model's declared one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DtMismatch {
    pub model: f64,
    pub written: f64,
}

impl DtMismatch {
    /// The refusal, naming where the written value came from (`what`, e.g.
    /// "`[config] dt`" or "--dt").
    pub fn message(&self, what: &str) -> String {
        format!(
            "{what} = {} is not the integration step: the model declares \
             `simulate {{ dt = {} }}`, which every run of it integrates at. \
             Drop {what}, or change the model's `dt`.",
            self.written, self.model
        )
    }
}

/// The integration step, from the model's declared step and the caller's
/// written one (`None` when the caller did not write one).
pub fn effective_dt(model_dt: Option<f64>, written: Option<f64>) -> Result<f64, DtMismatch> {
    match (model_dt, written) {
        (Some(model), Some(written)) if written != model => Err(DtMismatch { model, written }),
        (Some(model), _) => Ok(model),
        (None, Some(written)) => Ok(written),
        (None, None) => Ok(DEFAULT_DT),
    }
}

/// The step for a command that takes both a `--dt` and a `--fit` file
/// (`camdl profile`, `camdl pfilter`): the model's declared step; a written
/// one — `--dt`, or the fit file's `[config] dt` — must agree with it and with
/// each other; 1.0 when nothing declares one. The error names the source.
pub fn effective_dt_for_command(
    cli_dt: Option<f64>,
    fit_dt: Option<f64>,
    model_dt: Option<f64>,
) -> Result<f64, String> {
    if let (Some(cli), Some(fit)) = (cli_dt, fit_dt) {
        if cli != fit {
            return Err(format!(
                "--dt {cli} and the --fit file's `[config] dt = {fit}` disagree; drop one."
            ));
        }
    }
    let what = if cli_dt.is_some() { "--dt" } else { "the --fit file's [config] dt" };
    effective_dt(model_dt, cli_dt.or(fit_dt)).map_err(|m| m.message(what))
}

/// The step a run already checked by [`effective_dt`] used, recovered from the
/// model and the config value it carried (`[config] dt`, the written value or
/// its default). For readers of a stored fit, which was validated when it ran.
pub fn resolved_dt(model_dt: Option<f64>, config_dt: f64) -> f64 {
    model_dt.unwrap_or(config_dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_models_step_is_the_step_and_a_disagreeing_written_one_is_refused() {
        assert_eq!(effective_dt(Some(0.5), None), Ok(0.5));
        assert_eq!(effective_dt(Some(0.5), Some(0.5)), Ok(0.5));
        assert_eq!(effective_dt(Some(0.5), Some(0.25)),
            Err(DtMismatch { model: 0.5, written: 0.25 }));
        assert_eq!(effective_dt(None, Some(0.25)), Ok(0.25));
        assert_eq!(effective_dt(None, None), Ok(DEFAULT_DT));
        assert_eq!(resolved_dt(Some(0.5), 1.0), 0.5);
        assert_eq!(resolved_dt(None, 0.25), 0.25);
        let msg = DtMismatch { model: 0.5, written: 0.25 }.message("--dt");
        assert!(msg.contains("--dt = 0.25") && msg.contains("dt = 0.5"), "{msg}");
    }

    /// `camdl profile` / `camdl pfilter`: a written --dt or --fit
    /// `[config] dt` must agree with the model's declared step and with each
    /// other.
    #[test]
    fn a_command_reads_both_dt_and_a_fit_files_config_dt() {
        assert_eq!(effective_dt_for_command(None, None, Some(0.5)), Ok(0.5));
        assert_eq!(effective_dt_for_command(Some(0.5), Some(0.5), Some(0.5)), Ok(0.5));
        assert_eq!(effective_dt_for_command(None, None, None), Ok(1.0));
        assert_eq!(effective_dt_for_command(Some(0.25), None, None), Ok(0.25));
        assert_eq!(effective_dt_for_command(None, Some(0.25), None), Ok(0.25));
        let err = effective_dt_for_command(Some(0.25), None, Some(0.5)).unwrap_err();
        assert!(err.contains("--dt = 0.25") && err.contains("dt = 0.5"), "{err}");
        let err = effective_dt_for_command(None, Some(0.25), Some(0.5)).unwrap_err();
        assert!(err.contains("--fit file's [config] dt = 0.25"), "{err}");
        let err = effective_dt_for_command(Some(0.5), Some(0.25), None).unwrap_err();
        assert!(err.contains("disagree"), "{err}");
    }

}
