//! A resumed PGAS chain's NUTS counters cover the whole chain.
//!
//! The CLI judges `n_divergent_post_burn` and `n_max_treedepth_post_burn`
//! against every post-burn-in sweep of the chain, `n_sweeps - burn_in`, the
//! base's sweeps included (`cli/src/fit/pgas.rs`, the `DivergentTransitions`
//! and `MaxTreeDepthHits` findings). So the counts must include the base's
//! hits too, as `total_accepted` does, or a resumed run under-reports its
//! divergences and tree-depth saturation over the same denominator.

use std::cell::Cell;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, ChainResumeState, PGASConfig, PGASResult, PGASSweep, PGASTrajectory};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

const SEED: u64 = 20261109;
const DT: f64 = 1.0;

fn host_model() -> ir::Model {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_overdispersion.ir.json")
        .expect("read sir_overdispersion golden");
    let mut model = ir::from_str(&json).expect("parse sir_overdispersion");
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.name.as_str() {
                "beta" => 0.3,
                "gamma" => 0.1,
                "sigma_se" => 0.1,
                "N0" => 1000.0,
                "I0" => 10.0,
                _ => 0.5,
            };
            p.value = p.value.with_value(v);
        }
    }
    model
}

/// A Poisson stream over the infection flow, weekly.
fn poisson_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "weekly_cases".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        covers: None,
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

/// The hits of one call, counted sweep by sweep from what the sampler
/// reports: (post-burn-in sweeps at the tree-depth cap, post-burn-in
/// divergences, all sweeps at the cap, all divergences).
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Hits {
    treedepth_post_burn: usize,
    divergent_post_burn: usize,
    treedepth: usize,
    divergent: usize,
}

/// Run the fixture's chain to `n_sweeps`, fresh or from `resume`.
fn run(n_sweeps: usize, resume: Option<ChainResumeState>) -> (PGASResult, Hits) {
    let mut model = host_model();
    model.observations = vec![poisson_obs_block()];
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(SEED);
    let truth = simulate_reference(&compiled, &params, t_end, DT, &mut rng).unwrap();
    let mut cum: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth.substeps.iter().enumerate() {
        cum += rec.flows[0];
        let t = ((s + 1) as f64) * DT;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum as f64 });
            cum = 0;
        }
    }
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![StreamSpec::dense(
            0.0,
            StreamProjection::FlowSum(vec![0]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();

    let pgas_params = vec![EstimatedParam {
        name: "beta".into(),
        index: compiled.param_index["beta"],
        initial: 0.3,
        rw_sd: 0.02,
        transform: Transform::for_domain(0.01, 2.0, true).expect("domain"),
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];
    // A tree-depth cap of 1, so most NUTS sweeps reach it and the count has
    // something to carry.
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 32,
        n_sweeps,
        burn_in: BURN_IN,
        thin: 1,
        dt: DT,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 1,
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    let hits = Cell::new(Hits::default());
    let on_sweep = |sweep: usize, s: &PGASSweep, _: &PGASTrajectory| {
        let mut h = hits.get();
        let capped = s.nuts.tree_depth >= config.max_tree_depth;
        h.treedepth += usize::from(capped);
        h.divergent += s.nuts.n_divergent;
        if sweep >= BURN_IN {
            h.treedepth_post_burn += usize::from(capped);
            h.divergent_post_burn += s.nuts.n_divergent;
        }
        hits.set(h);
    };
    let result = run_pgas(
        &compiled, &pgas_params, &priors, &params, &config, &obs, &obs_model,
        SEED, Some(&on_sweep), None, resume, "resumed_nuts_counts".into(),
    )
    .expect("run_pgas must succeed");
    (result, hits.get())
}

const BURN_IN: usize = 4;

/// The resumed call's counters equal the base's hits plus its own segment's,
/// for a base that ended past burn-in. Totals and post-burn-in counts alike.
#[test]
fn a_resumed_chain_counts_its_bases_nuts_hits() {
    let (base, base_hits) = run(12, None);
    let (resumed, segment_hits) = run(20, Some(base.resume_state));
    assert!(base_hits.treedepth_post_burn > 0 && segment_hits.treedepth_post_burn > 0,
        "the fixture must reach the tree-depth cap after burn-in in both calls: \
         base {base_hits:?}, segment {segment_hits:?}");
    let got = Hits {
        treedepth_post_burn: resumed.n_max_treedepth_post_burn,
        divergent_post_burn: resumed.n_divergent_post_burn,
        treedepth: resumed.n_max_treedepth_total,
        divergent: resumed.n_divergent_total,
    };
    let expected = Hits {
        treedepth_post_burn: base_hits.treedepth_post_burn + segment_hits.treedepth_post_burn,
        divergent_post_burn: base_hits.divergent_post_burn + segment_hits.divergent_post_burn,
        treedepth: base_hits.treedepth + segment_hits.treedepth,
        divergent: base_hits.divergent + segment_hits.divergent,
    };
    assert_eq!(got, expected,
        "resumed counters vs base {base_hits:?} + segment {segment_hits:?}");
}
