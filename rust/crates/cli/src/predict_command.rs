//! `camdl predict RUN|MODEL` on the run store: the prediction of a stored fit
//! run's posterior, stored inside that run at `predict/<mode>-<id>/`, or of
//! one point given with a model file, stored at
//! `prediction/<model>-<id>/<mode>-<id>/`
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "Predictions" and
//! "`camdl predict` from a fixed point").
//!
//! The command's inputs are declared in [`crate::args::PredictInvocation`],
//! each with its class: the flags both sources share ([`PredictFlags`]), the
//! flags that read a posterior ([`PosteriorFlags`]) and the flags that pose a
//! fixed point ([`FixedPointFlags`]). A flag of one source's group given with
//! the other source is refused here, naming the source it belongs to. The
//! prediction is computed by [`crate::fit::predict::predict`]; this module
//! splits the inputs, refuses the removed spellings and opens the
//! invocation's record.

use crate::args::{FixedPointFlags, PosteriorFlags, PredictArgs, PredictFlags, PredictInvocation};
use crate::out::Out;

/// Refuse a flag of one source's group given with the other source: the
/// posterior flags name a fit's chains or read its held-out rows, and the
/// fixed-point flags give what a stored fit problem already holds.
pub(crate) fn refuse_other_source_flags(args: &PredictArgs) -> Result<(), String> {
    let Some(source) = args.source.as_deref() else { return Ok(()) };
    if crate::fit::fixed_point::names_a_model(source) {
        let given = args.posterior.given();
        if !given.is_empty() {
            return Err(format!(
                "{} read(s) a stored fit's posterior, and {source} is a model file, predicted at one \
                 point: --by-chain, --exclude-chains and --chains name a fit's chains, \
                 --realised-design reads held-out rows' design values, which a fixed-point prediction \
                 does not keep, and --model reports a stored fit through another model file. Name a stored fit run (`camdl predict @run`), or drop {}.",
                given.join(", "),
                given.join(", ")
            ));
        }
    } else {
        let given = args.fixed_point.given();
        if !given.is_empty() {
            return Err(format!(
                "{} pose(s) a prediction from a model file, and {source} names a stored fit run, whose \
                 problem holds its model, data, tables and numerics, and whose draws are the parameters. \
                 Drop {}, or predict from a model: `camdl predict MODEL.camdl --params FILE --data ...`.",
                given.join(", "),
                given.join(", ")
            ));
        }
    }
    Ok(())
}

/// `camdl predict RUN|MODEL`.
pub(crate) fn cmd_predict(inv: PredictInvocation) {
    let out = Out::new(false);
    if let Err(e) = refuse_other_source_flags(&inv.args) {
        out.fail(format!("error: {e}"));
    }
    let time_format = inv.args.fixed_point.time_format;
    // The export and the store request, read before the split hands them to
    // their own parts.
    let output = inv.args.flags.output.clone();
    let link_root = crate::link_tree::link_root(inv.args.flags.link_root.as_deref());
    let bindings =
        crate::bindings::claim(&out, inv.args.flags.manifest.as_deref(), link_root, inv.args.flags.label.as_deref());
    let request = crate::camdl_store::store_request(
        inv.args.flags.label.clone(),
        inv.args.flags.force,
        inv.args.flags.store.clone(),
    );
    let (declared, execution, _exports, _request, given) = store::Inputs::split(inv).into_parts();
    let (args, no_licm) = declared.into_parts();
    let (source, flags, posterior, fixed_point) = args.into_parts();
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
        covariate,
        n_draws,
        particles,
        seed,
    ) = flags.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let (by_chain, exclude_chains, chains, realised_design, model) = posterior.into_parts();
    let (params, param, table, data, obs, holdout_after, backend, dt, integrator, binomial, allow_degenerate_rates) =
        fixed_point.into_parts();
    let crate::args::PredictInvocationExecution { no_ir_cache, .. } = &execution;
    let compile = crate::util::CompileSettings::resolve(no_licm, *no_ir_cache);

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
            covariate,
            n_draws,
            particles,
            seed,
            output: None,
            manifest: None,
            link_root: None,
            force: false,
            label: None,
            store: None,
        },
        posterior: PosteriorFlags { by_chain, exclude_chains, chains, realised_design, model },
        fixed_point: FixedPointFlags {
            params,
            param,
            table,
            data,
            obs,
            holdout_after,
            backend,
            dt,
            integrator,
            binomial,
            allow_degenerate_rates,
            time_format,
        },
    };
    if let Some(dir) = &output {
        if let Err(e) = crate::util::refuse_missing_export_dir("-o", dir) {
            out.fail(format!("error: {e}"));
        }
    }
    // A run handle is looked up in the store the request names; a folder
    // handle names its own.
    crate::fit_store::set_reader_root(crate::camdl_store::root_of(&request));
    let info = store::RunInfo::new(&given, &execution, &request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    crate::fit::predict::predict(&args, compile, info, &request, output.as_deref(), &bindings);
}

#[cfg(test)]
mod tests;
