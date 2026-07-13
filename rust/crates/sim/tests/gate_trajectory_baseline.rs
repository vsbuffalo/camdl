//! Byte-identical trajectory baseline gate (the refactor ratchet).
//!
//! For every `ocaml/golden/*.ir.json`, simulate under each supported backend at a
//! fixed seed and assert the full trajectory hashes to a committed baseline. This
//! is the gate for behavior-preserving compiler/runtime refactors (the shared-
//! bindings + reduction work, docs/dev/proposals/2026-05-29-shared-bindings-and-
//! reduction.md): if D/B1 perturb a single count, this fails loudly and names the
//! model+backend, rather than the change passing on associativity-blind small
//! goldens (a 3-term `N=S+I+R` sums identically in any order; only a large/mixed
//! sum exposes a reassociation regression — see the gate models added alongside).
//!
//! Baselines are machine/toolchain-specific (libm `exp`/`sqrt` can differ by a
//! ULP across platforms). This is a *development ratchet*: capture on the dev
//! machine, run before/after each refactor phase on the same machine. Re-capture
//! with `CAMDL_CAPTURE_BASELINE=1 cargo test -p sim --test gate_trajectory_baseline
//! -- --nocapture` and paste the printed table below.
//!
//! Mirrors smoke_all_golden.rs (discovery, backend matrix, capability-skip) but
//! asserts trajectory identity, not just invariants.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;

fn ocaml_golden_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("../../../ocaml/golden")
}

fn discover_models() -> Vec<String> {
    let dir = ocaml_golden_dir();
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {:?}: {}", dir, e))
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            name.strip_suffix(".ir.json").map(|s| s.to_owned())
        })
        .collect();
    names.sort();
    names
}

fn load_and_apply_baseline(name: &str) -> ir::Model {
    let path = ocaml_golden_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    let mut model: ir::Model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e));
    if let Some(preset) = model.presets.first().cloned() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    model
}

/// FNV-1a/64 over the full trajectory numeric content. Deterministic and
/// platform-independent given identical inputs (no std hasher RNG).
fn trajectory_hash(traj: &sim::state::Trajectory) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for snap in &traj.snapshots {
        mix(&snap.t.to_bits().to_le_bytes());
        for &c in &snap.int_state.counts {
            mix(&c.to_le_bytes());
        }
        for &v in &snap.real_state.values {
            mix(&v.to_bits().to_le_bytes());
        }
        match &snap.flows {
            sim::state::Flows::Int(fs) => {
                for &f in fs {
                    mix(&f.to_le_bytes());
                }
            }
            // ODE flow is real-valued (rounding it re-introduces the sub-unit
            // bug); hash the f64 bits. Int hashing is byte-identical to the
            // pre-`Flows` `snap.flows.counts`, so gillespie/chain_binomial
            // baselines are unchanged; only the `ode` baselines move.
            sim::state::Flows::Real(fs) => {
                for &f in fs {
                    mix(&f.to_bits().to_le_bytes());
                }
            }
        }
    }
    h
}

/// FNV-1a/64 over the STATE-ONLY trajectory content: `t`, integer counts, and
/// real values — EXCLUDING flows. The Phase-A→B instrument (gh#166, proposal
/// gate #5): the Euler→augmented flow change (Q1B) moves the full
/// `trajectory_hash` (which mixes flows) for every ODE model, but the
/// compartment integration is independent of the flow accumulators (nothing in
/// dX/dt reads them), so this hash must stay byte-identical across that change —
/// proving prevalence is untouched and only incidence moves.
fn ode_state_hash(traj: &sim::state::Trajectory) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for snap in &traj.snapshots {
        mix(&snap.t.to_bits().to_le_bytes());
        for &c in &snap.int_state.counts {
            mix(&c.to_le_bytes());
        }
        for &v in &snap.real_state.values {
            mix(&v.to_bits().to_le_bytes());
        }
        // snap.flows DELIBERATELY excluded — see the doc comment.
    }
    h
}

/// Committed baselines: (model, backend) -> trajectory hash, captured on the dev
/// machine against the current compiler+runtime. Re-capture per the header.
const BASELINES: &[(&str, &str, u64)] = &[
    ("bimolecular", "gillespie", 0x54a38d360dcf4c01),
    // gh#121: `bimolecular` is a MULTI-SOURCE transition (`A + B --> C`), now
    // rejected on chain_binomial (bounded by only the first source). The gate's
    // per-run `Err(_) => continue` drops it, so no CB baseline exists for it.
    ("branching_si_symp_asym", "gillespie", 0x325b8b153b1b16d4),
    ("branching_si_symp_asym", "chain_binomial", 0xae1bb55ced8410bd),
    // Ebola onset-to-outcome as a CFR-split `via hyper_erlang(...)` mixture
    // (Phase 4, PER-BRANCH endpoints): two parallel Erlang-3 chains drained out
    // of `I`, the fatal arm exiting to `D` and the recover arm to `R`, the entry
    // into `I` split `cfr` / `1−cfr`, and the FOI's bare `I` summing all six
    // infectious stages. No manual twin in the corpus; the hashes stand alone
    // (the per-branch-destination IR isomorphism is pinned by test_hyper_erlang).
    ("ebola_outcome_hyper", "gillespie", 0xe0536f62dd9a2ebc),
    ("ebola_outcome_hyper", "chain_binomial", 0x34833164a6ba99c1),
    ("ebola_outcome_hyper", "ode", 0x4a7a3a00b5d8d557),
    ("malaria_two_species", "gillespie", 0x5ed03d7812021914),
    ("malaria_two_species", "chain_binomial", 0x132e1d7efc2da7d4),
    ("polio_age", "gillespie", 0x968a5308fde3affb),
    ("polio_age", "chain_binomial", 0x7b8b1e77dbccab4b),
    // Polio bimodal shedding as a SAME-endpoint `via hyper_erlang(...)` mixture
    // (Phase 4): an Erlang-2 typical arm and an Erlang-1 prolonged arm, both
    // draining `I` to the shared `--> R`, the entry split `p` / `1−p`, and the
    // FOI's bare `I` summing all three stages. Hashes stand alone (the
    // same-endpoint IR shape is pinned by test_hyper_erlang).
    ("multi_index_beta", "gillespie", 0xb5b316ed9463ebf2),
    ("multi_index_beta", "chain_binomial", 0xb63048d396fb309a),
    ("multi_index_beta", "ode", 0x85f898d465a420b5),
    ("polio_shedding_bimodal", "gillespie", 0xef5d276a6774bc31),
    ("polio_shedding_bimodal", "chain_binomial", 0xdd348d9d5a752f05),
    ("polio_shedding_bimodal", "ode", 0x4b0e6b49f73269b4),
    ("polio_spatial_5", "gillespie", 0x5516309d3eedfda4),
    ("polio_spatial_5", "chain_binomial", 0x3b8831126ad37aeb),
    ("ross_macdonald", "gillespie", 0xb8a901ca29312b3e),
    ("ross_macdonald", "chain_binomial", 0xfaa942e09da2009a),
    ("seir_age", "gillespie", 0x42aa86e0753ea235),
    ("seir_age", "chain_binomial", 0x1ea29e011a7eba67),
    // Same dynamics as seir_age; differ only in the observations block
    // (un-indexed incidence strata-sum / let-bound projection). The
    // trajectory hash excludes observations, so these match seir_age.
    ("seir_age_incidence_sum", "gillespie", 0x42aa86e0753ea235),
    ("seir_age_incidence_sum", "chain_binomial", 0x1ea29e011a7eba67),
    ("seir_age_let_projection", "gillespie", 0x42aa86e0753ea235),
    ("seir_age_let_projection", "chain_binomial", 0x1ea29e011a7eba67),
    ("seir_age_table_rates", "gillespie", 0xaefb0972f1798fc5),
    ("seir_age_table_rates", "chain_binomial", 0x87d0504d39dc8044),
    // Age-stratified SEIR with an Erlang-3 INFECTIOUS period via the `via`
    // clause (Phase 2b): `I` is staged AND age-stratified, the per-age FOI's
    // `I[b]` is rewritten to sum over stages. Unlike seir_erlang_via, this has
    // no manually-staged twin in the corpus, so its hashes stand on their own
    // (the T6 anchor test pins the via↔manual IR isomorphism instead).
    ("seir_age_erlang_via", "gillespie", 0x656107884de56623),
    ("seir_age_erlang_via", "chain_binomial", 0xca5ff916a5280453),
    // Joint patch×age stratification (4 patches × 3 ages = 48 compartments,
    // 120 transitions): the only golden exercising the 2-axis cross-product
    // expander + cross-dimension transitions (aging within patch, spatial FOI
    // summed across patches within an age band, age-contact mixing within a
    // patch). Single-axis goldens reassociate identically; this one doesn't.
    ("seir_cross_dim", "gillespie", 0x238f6ca444053ff0),
    ("seir_cross_dim", "chain_binomial", 0x1385ec55aeecbc76),
    ("seir_cross_dim", "ode", 0xcadf49dec610af29),
    ("seir_defines_adj", "gillespie", 0x6f777f70cb7742ca),
    ("seir_defines_adj", "chain_binomial", 0xa443c47393008cf7),
    ("seir_defines_patch", "gillespie", 0xa7c867674ed33cf9),
    ("seir_defines_patch", "chain_binomial", 0x35818731f63a2b8b),
    ("seir_erlang", "gillespie", 0x9678d01f75671b6f),
    ("seir_erlang", "chain_binomial", 0x08b695ddf690d3f0),
    // seir_erlang_via: the `via erlang(...)` clause desugars to exactly
    // seir_erlang's manual staged chain, so its trajectory is BYTE-IDENTICAL to
    // seir_erlang under every backend (same hashes below) — an end-to-end
    // confirmation that the staging lowering changes nothing dynamical.
    ("seir_erlang_via", "gillespie", 0x9678d01f75671b6f),
    ("seir_erlang_via", "chain_binomial", 0x08b695ddf690d3f0),
    ("seir_erlang_staged", "gillespie", 0xee741459747732f2),
    ("seir_erlang_staged", "chain_binomial", 0xd5463d6b91a7545d),
    ("seir_observations", "gillespie", 0x1512c82543641dbc),
    ("seir_observations", "chain_binomial", 0x1620e4f54e9021bf),
    ("seir_seasonal_patch", "gillespie", 0xbab747d305e59679),
    ("seir_seasonal_patch", "chain_binomial", 0x973dccbdeba49bb5),
    ("seir_spatial_5_inference", "chain_binomial", 0xfc6f6fe0c603429e),
    ("seir_vaccine", "gillespie", 0x17257cd9fa3ce428),
    ("seir_vaccine", "chain_binomial", 0xfb6b6f6bdba7e7d3),
    ("seir_vaccine_seasonal", "gillespie", 0xaadce0ddf1d680fd),
    ("seir_vaccine_seasonal", "chain_binomial", 0xdce773319d7251e7),
    ("sia_anchored_dates", "gillespie", 0xa07df71463113b70),
    ("sia_anchored_dates", "chain_binomial", 0x5a594b3abc78f56b),
    ("sia_instance_enable", "gillespie", 0x7e32d1580b333e86),
    ("sia_instance_enable", "chain_binomial", 0x410202f77d80ff93),
    ("sir_basic", "gillespie", 0xc58ddb854d12660a),
    ("sir_basic", "chain_binomial", 0x233d5bb24557cb84),
    ("sir_coupling", "gillespie", 0xfa90685fe7e20637),
    ("sir_coupling", "chain_binomial", 0x909c2ae3a066dd5c),
    ("sir_demography", "gillespie", 0xf6238b4be3d98bcb),
    ("sir_demography", "chain_binomial", 0x57c2f3c4272fb8e0),
    ("sir_dim_annotated", "gillespie", 0xc8cc8178959c656f),
    ("sir_dim_annotated", "chain_binomial", 0x3a9a794fc35272cd),
    ("sir_dt", "gillespie", 0xb26d184abde13883),
    ("sir_dt", "chain_binomial", 0x419976994c4a9b4d),
    ("sir_five_age", "gillespie", 0x0b027f6bf099d5ec),
    ("sir_five_age", "chain_binomial", 0xd8287dd6daf58eb1),
    ("sir_init_table", "gillespie", 0xde69b188f2e80a65),
    ("sir_init_table", "chain_binomial", 0xfb5d74fc9bbb470e),
    ("sir_overdispersion", "chain_binomial", 0x7be63a31759824b8),
    ("sir_patches_5", "gillespie", 0xbb5266c0c72c32b4),
    ("sir_patches_5", "chain_binomial", 0xb2a247f9ca9c2afe),
    ("sir_priors", "gillespie", 0xc58ddb854d12660a),
    ("sir_priors", "chain_binomial", 0x233d5bb24557cb84),
    ("sir_reservoir", "gillespie", 0x47bfd5ec6fefdb43),
    ("sir_reservoir", "chain_binomial", 0x6f5c2c8af8307f5c),
    // Mixed int/real >=8-term aggregate (Fix-B trap #1 gate): a binding
    // extraction that reassociates the MixedPopSum fold order changes these.
    ("sir_reservoir_mixed", "gillespie", 0xa3b890243e0932a5),
    ("sir_reservoir_mixed", "chain_binomial", 0x0597d93ff326fb1b),
    ("sir_spatial_sum", "gillespie", 0x65d363618fc40fb4),
    ("sir_spatial_sum", "chain_binomial", 0xd38ed3b3bfe9c9fa),
    // overdispersion model: gillespie/ode capability-skip, so chain-binomial only
    ("sir_two_overdispersed", "chain_binomial", 0x47b4ab5edd2fb5c4),
    ("sir_two_patch", "gillespie", 0xe9f432f7882e9b70),
    ("sir_two_patch", "chain_binomial", 0xa1c9f945649cc4fa),
    ("sirv_anchored_calendar", "gillespie", 0xec592cdf358a308e),
    ("sirv_anchored_calendar", "chain_binomial", 0x557cef37b9b035b1),
    // ODE backend (deterministic; added per the four-backend landing
    // condition). Captured against the post-Fix-B compiler/runtime.
    ("bimolecular", "ode", 0x19390c52ffe65914),
    ("branching_si_symp_asym", "ode", 0x62b8c2365ff30f4d),
    ("malaria_two_species", "ode", 0xbfc172d9173156ee),
    ("polio_age", "ode", 0xfcae9872a53fbfd5),
    ("polio_spatial_5", "ode", 0x8a87376834332b83),
    ("ross_macdonald", "ode", 0x649e56498d64d76d),
    ("seir_age", "ode", 0x3e30c77d579b38a7),
    ("seir_age_incidence_sum", "ode", 0x3e30c77d579b38a7),
    ("seir_age_let_projection", "ode", 0x3e30c77d579b38a7),
    ("seir_age_table_rates", "ode", 0x96d4a0f0287ecb7b),
    ("seir_age_erlang_via", "ode", 0x093aa76b68d31054),
    ("seir_defines_adj", "ode", 0xb6b63bd987b59e8c),
    ("seir_defines_patch", "ode", 0xdb957d113668b48e),
    ("seir_erlang", "ode", 0xaabb48fd40c23438),
    ("seir_erlang_via", "ode", 0xaabb48fd40c23438),   // == seir_erlang (desugars identically)
    ("seir_erlang_staged", "ode", 0xe60a8e49be37e706),
    ("seir_observations", "ode", 0x93753fb7da6c81e5),
    ("seir_seasonal_patch", "ode", 0x3184c3472b16c420),
    ("seir_vaccine", "ode", 0x24b4bb52d195fad9),
    ("seir_vaccine_seasonal", "ode", 0x9820f1bddecdf334),
    ("sia_anchored_dates", "ode", 0x20c5c2abc81cf6a6),
    ("sia_instance_enable", "ode", 0xe173cf74c0d0e008),
    ("sir_basic", "ode", 0xc5f3f8c75b2a5cc8),
    ("sir_coupling", "ode", 0xd80e32cdc3656de6),
    ("sir_demography", "ode", 0x4c938f0b4442e646),
    ("sir_dim_annotated", "ode", 0xdc4237fabde1e1a8),
    ("sir_dt", "ode", 0xaf5cd1d73847a50d),
    ("sir_five_age", "ode", 0x807e47630c0c1225),
    ("sir_init_table", "ode", 0x841ab0abdaf56108),
    ("sir_patches_5", "ode", 0xe86f08d9d8e88bf6),
    ("sir_priors", "ode", 0xc5f3f8c75b2a5cc8),
    ("sir_reservoir", "ode", 0x4ec00b5be816b0a9),
    ("sir_reservoir_mixed", "ode", 0x973c801291584d80),
    ("sir_spatial_sum", "ode", 0x2c7b8a06628edfc5),
    ("sir_two_patch", "ode", 0x10d330cc876c377f),
    ("sirv_anchored_calendar", "ode", 0x3cb6557a75653f74),
    // Six feature-coverage goldens added in c760b230 (forcing-from-data,
    // unchecked phenom mixing, population balance, seasonal importation,
    // guarded FOI, surveillance likelihoods). Captured deterministically;
    // the existing 105 entries re-verified unchanged in the same run.
    // seir_pop_balance is chain_binomial-only (capability-skip on gillespie/ode).
    ("flu_data_forcing", "gillespie", 0xbefc2c0366093bcf),
    ("flu_data_forcing", "chain_binomial", 0x7b037870200ca140),
    ("flu_data_forcing", "ode", 0x19cffc0fbaebe987),
    ("phenom_mixing_unchecked", "gillespie", 0x5113d45f49fcb942),
    // gh#122: the sole-exit deterministic `waning : R --> S @ deterministic(omega*R)`
    // was silently FROZEN on chain_binomial (never fired); it now fires
    // `round(omega*R*dt)`, so this trajectory legitimately changed. gillespie/ode
    // never had the freeze (both run `deterministic()` as a flow), so their
    // baselines above/below are unchanged. This is the ONLY golden with a sourced
    // deterministic transition; every deterministic-free baseline is byte-identical.
    ("phenom_mixing_unchecked", "chain_binomial", 0xeb226deae32a4e86),
    ("phenom_mixing_unchecked", "ode", 0x38dcf13c1d2f0570),
    ("seir_pop_balance", "chain_binomial", 0xc3eb97c9311a8dca),
    ("seir_seasonal_importation", "gillespie", 0xba237ff576896498),
    ("seir_seasonal_importation", "chain_binomial", 0xba237ff576896498),
    ("seir_seasonal_importation", "ode", 0xba237ff576896498),
    ("sir_guarded_foi", "gillespie", 0x5eca6018289b72bd),
    ("sir_guarded_foi", "chain_binomial", 0x80ef48856ef6bf15),
    ("sir_guarded_foi", "ode", 0xaeb58e6a6b2cf3db),
    ("surveillance_likelihoods", "gillespie", 0xd289093f707a3cea),
    ("surveillance_likelihoods", "chain_binomial", 0xcfa2b111d613f1e7),
    ("surveillance_likelihoods", "ode", 0xc4164552649db984),
    // §4.2 long-form stratified-observation fixture (sir_two_patch_long_obs):
    // a 2-patch SIR with an indexed `cases[p in patch]` header. The trajectory
    // hash excludes observations, so it pins only the dynamics (identical-shape
    // to sir_two_patch but distinct params/levels).
    ("sir_two_patch_long_obs", "gillespie", 0x695d50d1cbec83fc),
    ("sir_two_patch_long_obs", "chain_binomial", 0xcd2756793661993f),
    ("sir_two_patch_long_obs", "ode", 0x447fa76c8402b0ee),
    // Restricted-sum `where` coupling (gh#185): 6-patch line, radius support
    // pruned at compile time, seeded p0 → traveling wave across in-radius
    // neighbours. The first golden exercising `sum(q where dist[p,q] < r, …)`.
    // (Hashes re-captured post-rebase against IR 0.15 + augmented-flow ode.)
    ("sir_spatial_where", "gillespie", 0x093f980c2089c00c),
    ("sir_spatial_where", "chain_binomial", 0x3a8c0b0bb6dd86ac),
    ("sir_spatial_where", "ode", 0x414139babb1ed18a),
];

/// State-only ODE baselines (gh#166 Phase A): model -> `ode_state_hash`, captured
/// in the Euler-flow era. Phase B (augmented flow) must leave EVERY one of these
/// unchanged — the proof that unifying flow accounting did not perturb the
/// compartment integration. Re-capture per the file header with
/// `CAMDL_CAPTURE_BASELINE=1`.
const ODE_STATE_BASELINES: &[(&str, u64)] = &[
    ("bimolecular", 0x1bd688a80a4578f1),
    ("branching_si_symp_asym", 0x35a833278ee1fecc),
    ("ebola_outcome_hyper", 0xfda4ae754f35311c),
    ("flu_data_forcing", 0xd55c543de04d2062),
    ("malaria_two_species", 0xfd4699acf8596e87),
    ("multi_index_beta", 0x31fc0e11eb647500),
    ("phenom_mixing_unchecked", 0x46f766b4f10b0138),
    ("polio_age", 0x3feecf44d4f3c67a),
    ("polio_shedding_bimodal", 0x794dd970f8dd41e2),
    ("polio_spatial_5", 0x14cfd1ded179cc60),
    ("ross_macdonald", 0x3d5f33467f72bafd),
    ("seir_age", 0x310c060d6ceebe19),
    ("seir_age_incidence_sum", 0x310c060d6ceebe19),
    ("seir_age_let_projection", 0x310c060d6ceebe19),
    ("seir_age_table_rates", 0x7740d7f2d7d93a9b),
    ("seir_age_erlang_via", 0x92b78a5c2fe58b8c),
    ("seir_cross_dim", 0x56539345e82ada6f),
    ("seir_defines_adj", 0x34100c3629bb7053),
    ("seir_defines_patch", 0xcfaedfa885954f5f),
    ("seir_erlang", 0xd67553e482930e56),
    ("seir_erlang_via", 0xd67553e482930e56),   // == seir_erlang (desugars identically)
    ("seir_erlang_staged", 0x9d71c13925516443),
    ("seir_observations", 0xd98853739e669231),
    ("seir_seasonal_importation", 0x674f33759aab5fb8),
    ("seir_seasonal_patch", 0x647c238e73b173d3),
    ("seir_vaccine", 0x4e897aed24c09e15),
    ("seir_vaccine_seasonal", 0xe953521cacf5427e),
    ("sia_anchored_dates", 0x483f6805d1c71332),
    ("sia_instance_enable", 0xee4e306f1d08e9fc),
    ("sir_basic", 0xbfe29deac5a25942),
    ("sir_coupling", 0xe2121c407df25c01),
    ("sir_demography", 0x1070553187edc261),
    ("sir_dim_annotated", 0x8053d63f41130840),
    ("sir_dt", 0x0625b77d9571b8d7),
    ("sir_five_age", 0xca5a77e2fcbfc66f),
    ("sir_guarded_foi", 0xf61bb2663c7f6d65),
    ("sir_init_table", 0x8f76471fa24057b2),
    ("sir_patches_5", 0x64d27ef986e34fc9),
    ("sir_priors", 0xbfe29deac5a25942),
    ("sir_reservoir", 0x68ce5f90016f7ae7),
    ("sir_reservoir_mixed", 0xeb5efcb47704c2e5),
    ("sir_spatial_sum", 0xcd94b01eae00eeb9),
    ("sir_spatial_where", 0x416cb8b9aa0f7f83),
    ("sir_two_patch", 0xa589042453cfa6bc),
    ("sir_two_patch_long_obs", 0x84dd19dfc276b148),
    ("sirv_anchored_calendar", 0x59095d189f6f3b42),
    ("surveillance_likelihoods", 0x1797802b02ef6f71),
];

#[test]
fn gate_golden_trajectories_are_byte_identical() {
    // Match smoke_all_golden: legacy degenerate-rate mode + no interventions, so
    // the baseline behavior is well-defined for the whole corpus.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let capture = std::env::var("CAMDL_CAPTURE_BASELINE").is_ok();

    let models = discover_models();
    assert!(!models.is_empty(), "no *.ir.json in ocaml/golden/");

    let lookup = |name: &str, backend: &str| -> Option<u64> {
        BASELINES.iter()
            .find(|(n, b, _)| *n == name && *b == backend)
            .map(|(_, _, h)| *h)
    };

    let mut captured: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    for name in &models {
        let mut model = load_and_apply_baseline(name);
        model.interventions.clear();
        let compiled = match CompiledModel::new(model.clone()) {
            Ok(c) => c,
            Err(_) => continue, // models that don't compile under baseline are out of scope here
        };
        let params = compiled.default_params.clone();
        let t_start = model.simulation.t_start;
        let t_end = model.simulation.t_end.min(30.0);

        // ODE is deterministic (seed ignored); capability-skip drops models it
        // can't run (e.g. overdispersion), and the run-error `continue` below
        // drops any it errors on.
        let backends: &[(&str, SimConfig)] = &[
            ("gillespie", SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None })),
            ("chain_binomial", SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 })),
            ("ode", SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 })),
        ];
        let required = compiled.required_capabilities();
        for (backend, config) in backends {
            let sim: &dyn Simulate = match *backend {
                "gillespie" => &GillespieSim,
                "ode" => &OdeSim,
                _ => &ChainBinomialSim,
            };
            if !(required - sim.capabilities()).is_empty() {
                continue;
            }
            let traj = match sim.run(&compiled, &params, SEED, config) {
                Ok(t) => t,
                Err(_) => continue, // baseline-time sim errors are not this gate's concern
            };
            let hash = trajectory_hash(&traj);
            if capture {
                captured.push(format!("    (\"{name}\", \"{backend}\", 0x{hash:016x}),"));
            } else {
                match lookup(name, backend) {
                    Some(expected) => assert_eq!(
                        hash, expected,
                        "TRAJECTORY CHANGED for {name}/{backend}: a refactor perturbed \
                         the trajectory (got 0x{hash:016x}, expected 0x{expected:016x})"
                    ),
                    None => missing.push(format!("{name}/{backend}")),
                }
            }
        }
    }

    if capture {
        eprintln!("\n// <<CAPTURED-BASELINES>> — paste into BASELINES:");
        for line in &captured {
            eprintln!("{line}");
        }
        eprintln!("// ({} entries)\n", captured.len());
    } else {
        assert!(
            missing.is_empty(),
            "no baseline for: {missing:?} — run with CAMDL_CAPTURE_BASELINE=1 and paste the table"
        );
    }
}

#[test]
fn gate_ode_state_only_hash_is_stable() {
    // Mirror the full gate's setup so the trajectory is the SAME run, then hash
    // state only (int+real, no flows). See `ode_state_hash`.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let capture = std::env::var("CAMDL_CAPTURE_BASELINE").is_ok();

    let models = discover_models();
    assert!(!models.is_empty(), "no *.ir.json in ocaml/golden/");

    let lookup = |name: &str| -> Option<u64> {
        ODE_STATE_BASELINES.iter().find(|(n, _)| *n == name).map(|(_, h)| *h)
    };

    let mut captured: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    for name in &models {
        let mut model = load_and_apply_baseline(name);
        model.interventions.clear();
        let compiled = match CompiledModel::new(model.clone()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let params = compiled.default_params.clone();
        let t_start = model.simulation.t_start;
        let t_end = model.simulation.t_end.min(30.0);

        // ODE backend only; capability-skip drops models it can't run.
        let required = compiled.required_capabilities();
        if !(required - OdeSim.capabilities()).is_empty() {
            continue;
        }
        let cfg = SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 });
        let traj = match OdeSim.run(&compiled, &params, SEED, &cfg) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let hash = ode_state_hash(&traj);
        if capture {
            captured.push(format!("    (\"{name}\", 0x{hash:016x}),"));
        } else {
            match lookup(name) {
                Some(expected) => assert_eq!(
                    hash, expected,
                    "ODE STATE-ONLY hash CHANGED for {name}: the compartment \
                     integration moved (got 0x{hash:016x}, expected 0x{expected:016x}). \
                     Phase B (augmented flow) must leave prevalence byte-identical."
                ),
                None => missing.push(name.clone()),
            }
        }
    }

    if capture {
        eprintln!("\n// <<CAPTURED-ODE-STATE-BASELINES>> — paste into ODE_STATE_BASELINES:");
        for line in &captured {
            eprintln!("{line}");
        }
        eprintln!("// ({} entries)\n", captured.len());
    } else {
        assert!(
            missing.is_empty(),
            "no ODE state baseline for: {missing:?} — run with CAMDL_CAPTURE_BASELINE=1 \
             and paste the table"
        );
    }
}
