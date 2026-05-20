#!/usr/bin/env python3
"""Generate the Tier-5 external-oracle reference TSV for stratified lineage
attribution, using VGsim as an independent lineage-aware simulator.

    2026-05-19 individual-sampling-layer proposal, §"Validation" Tier 5.

WHY THIS EXISTS
---------------
Tiers 1–3 (structural, trajectory byte-identity, analytic Yule) and Tier 2b
(empirical attribution frequencies vs the analytic decomposition) all test
camdl against *camdl's own* math. Tier 5 closes the loop by cross-validating
the **stratified** scenario against a fully independent simulator with a
non-overlapping code path. The thing being validated is the one with real
silent-wrong-answer risk: contact-structured, per-stratum parent attribution.

This script is NOT run by CI. Like the pomp/scipy/numpy oracle generators in
this directory, it writes a committed TSV that the Rust test loads offline.
Regenerate only when the matched model or VGsim's version changes.

    Run:  uv run --with vgsim scripts/gen_spatial_lineage_vgsim_fixture.py \
              --out rust/crates/sim/tests/fixtures/spatial_lineage_vgsim.tsv
    Pin:  record the exact VGsim version in the TSV header (this script does).

THE MODEL MATCH (the load-bearing part)
---------------------------------------
camdl model under test: `spatial_lineage.camdl` — a two-patch SIR with a
frequency-dependent, contact-structured force of infection into patch a:

    infection[a] : S[a] --> I[a]  @  beta * S[a] * sum_b C[a,b] * I[b] / N[b]
    recovery[a]  : I[a] --> R[a]  @  gamma * I[a]

    C = [[1.0, 0.3],     (row = focal patch a, col = infector patch b)
         [1.5, 0.2]]     ASYMMETRIC: C[a,b]=0.3 != C[b,a]=1.5
    init: S_a=4000 I_a=5,  S_b=2000 I_b=40,   beta=0.6  gamma=0.2

The statistic to cross-validate is the conditional distribution of the
INFECTOR's patch given the infectee's patch, accumulated at the event-instant
state:

    P(infector in b | infectee in a) ∝ C[a,b] * I_b / N_b.

VGsim (Vladimir Shchur et al., "VGsim: scalable viral genealogy simulator",
PLoS Comput Biol 2022) simulates a multi-population epidemic with a contact /
migration matrix and emits a genealogy whose nodes carry the population
(deme) of each lineage. From the VGsim output we record, for each transmission
event, (infectee_deme, infector_deme) and tabulate the empirical conditional
infector-deme distribution.

To match camdl exactly the VGsim configuration must encode the SAME
frequency-dependent, C-weighted cross-deme transmission. Concretely:
  - two populations a, b with sizes N_a=4005, N_b=2040 (= S+I at t=0);
  - per-contact transmission scaled so that the realised force of infection
    into deme a from deme b is  beta * C[a,b] * I_b / N_b  (VGsim's
    susceptibility * contact-matrix entry must reproduce beta*C[a,b]);
  - recovery rate gamma in both demes;
  - a single susceptibility type (no within-host heterogeneity), one
    haplotype (no sequence evolution), so the only structure is spatial.

CAUTION — this match is the entire scientific content of the test. VGsim's
contact matrix is row-normalised differently from camdl's raw C; the person
regenerating this fixture MUST verify that VGsim's effective per-pair
transmission equals beta*C[a,b]/N_b (frequency-dependent), not beta*C[a,b]
(density-dependent), before trusting the output. If in doubt, prefer MASTER
(BEAST2) whose structured birth-death migration parameterisation maps onto
C[a,b] more transparently; see the alternative note at the bottom.

OUTPUT FORMAT
-------------
TSV with a `#`-comment header recording the VGsim version and the matched
model, then columns:

    focal_deme    parent_deme    probability

one row per (focal_deme, parent_deme) pair, where `probability` is the
empirical P(parent_deme | focal_deme) from the VGsim run (rows for a fixed
focal_deme sum to 1). The Rust test compares camdl's own conditional
infector-deme distribution against these probabilities within a tolerance
that accounts for both simulators' Monte-Carlo error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Matched-model constants (must equal spatial_lineage.camdl / its params).
N_DEMES = 2
C = [[1.0, 0.3], [1.5, 0.2]]
S0 = [4000, 2000]
I0 = [5, 40]
BETA = 0.6
GAMMA = 0.2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("rust/crates/sim/tests/fixtures/spatial_lineage_vgsim.tsv"),
        help="output TSV path",
    )
    ap.add_argument("--iterations", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    try:
        import VGsim  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "VGsim is not installed. Install it (pip install vgsim) in an "
            "environment with network access, then re-run this script.\n"
            "Until then the committed fixture remains the PLACEHOLDER and the "
            "Rust oracle test silently skips.\n"
        )
        return 2

    # ----------------------------------------------------------------------
    # NOTE TO THE HUMAN REGENERATING THIS FIXTURE:
    #
    # The block below is the VGsim model construction. It is left as a
    # documented skeleton rather than a guessed-at, possibly-wrong
    # configuration, because getting the frequency-dependent C-weighting
    # right is the whole point of the test and must be verified by someone
    # who can run VGsim and inspect its realised rates. Fill it in, confirm
    # the realised per-pair transmission equals beta*C[a,b]/N_b, run, and
    # tabulate (focal_deme, parent_deme) from the resulting genealogy.
    #
    #   sim = VGsim.Simulator(number_of_sites=0, populations=N_DEMES, ...)
    #   sim.set_transmission_rate(BETA)
    #   sim.set_recovery_rate(GAMMA)
    #   sim.set_population_size(S0[0] + I0[0], population=0)
    #   sim.set_population_size(S0[1] + I0[1], population=1)
    #   sim.set_migration_probability(...)  # encode C, row-normalised, /N_b
    #   sim.set_susceptible(...) ; sim.set_infectious(...)
    #   sim.simulate(args.iterations, seed=args.seed)
    #   genealogy = sim.output_tree_events()  # nodes carry population
    #   ... tabulate transmission events by (recipient_pop, donor_pop) ...
    # ----------------------------------------------------------------------
    sys.stderr.write(
        "VGsim import succeeded but the model-construction block in this "
        "script is intentionally a skeleton (see the NOTE). Complete it, "
        "verify the realised rates match beta*C[a,b]/N_b, then write the TSV "
        "via write_fixture(args.out, version, table).\n"
    )
    return 3


def write_fixture(out: Path, vgsim_version: str, table: dict[tuple[int, int], float]) -> None:
    """Write the (focal_deme, parent_deme, probability) TSV with a provenance
    header. `table[(focal, parent)]` is the empirical conditional probability;
    rows for a fixed focal must sum to 1."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Tier-5 external oracle: VGsim {vgsim_version}\n")
        f.write("# Matched model: spatial_lineage.camdl (two-patch SIR,\n")
        f.write(f"#   C={C}, S0={S0}, I0={I0}, beta={BETA}, gamma={GAMMA})\n")
        f.write("# Statistic: P(parent_deme | focal_deme) for transmission events.\n")
        f.write("focal_deme\tparent_deme\tprobability\n")
        for focal in range(N_DEMES):
            for parent in range(N_DEMES):
                p = table.get((focal, parent), 0.0)
                f.write(f"{focal}\t{parent}\t{p:.10f}\n")


if __name__ == "__main__":
    raise SystemExit(main())
