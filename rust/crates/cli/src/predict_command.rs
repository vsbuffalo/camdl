//! `camdl predict RUN` on the run store: the prediction of a stored fit run's
//! posterior, stored inside that run at `predict/<mode>-<id>/`
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "Predictions").
//!
//! The command's inputs are declared in [`crate::args::PredictInvocation`],
//! each with its class. Every flag is `identity` except the global
//! `--no-licm` and `--no-ir-cache`, which reach only the compile of a
//! `--quantities` vocabulary and are `execution`. The prediction is computed
//! by [`crate::fit::predict::predict`]; this module splits the inputs, refuses
//! the removed spellings and opens the invocation's record.

use crate::args::{PosteriorFlags, PredictArgs, PredictFlags, PredictInvocation};
use crate::out::Out;

/// The refusal of `camdl fit predict`, which `camdl predict` replaced.
pub(crate) fn refuse_fit_predict() {
    Out::new(false).fail(
        "error: `camdl fit predict` has been removed. A prediction is made from a stored fit run by \
         `camdl predict @run`, where `@run` is the run's @label, a run-id prefix or its folder; \
         every other flag keeps its meaning (`camdl predict --help`)."
            .to_string(),
    );
}

/// `camdl predict RUN`.
pub(crate) fn cmd_predict(inv: PredictInvocation) {
    let out = Out::new(false);
    if let Some(fit) = inv.args._removed_fit.as_deref() {
        out.fail(format!(
            "error: --fit has been removed; the run is the positional argument: `camdl predict {fit}`."
        ));
    }
    let (declared, execution, _exports, _request, given) = store::Inputs::split(inv).into_parts();
    let (args,) = declared.into_parts();
    let (source, flags, posterior) = args.into_parts();
    let (
        forcing_extrapolation,
        stream,
        scenarios,
        enable,
        disable,
        sweep,
        mode,
        horizon,
        design,
        quantities,
        n_draws,
        particles,
        seed,
    ) = flags.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let (by_chain, exclude_chains, chains, realised_design) = posterior.into_parts();
    let crate::args::PredictInvocationExecution { no_licm, no_ir_cache, .. } = &execution;
    let compile = crate::util::CompileSettings::resolve(*no_licm, *no_ir_cache);

    // The prediction reads the arguments reassembled, every declared input
    // placed back in the struct it reads.
    let args = PredictArgs {
        source,
        flags: PredictFlags {
            forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
            stream,
            scenarios,
            enable,
            disable,
            sweep,
            mode,
            horizon,
            design,
            quantities,
            n_draws,
            particles,
            seed,
        },
        posterior: PosteriorFlags { by_chain, exclude_chains, chains, realised_design },
        _removed_fit: None,
    };
    let info = store::RunInfo::new(&given, &execution, &store::StoreRequest::default())
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    crate::fit::predict::predict(&args, compile, info);
}

#[cfg(test)]
mod tests;
