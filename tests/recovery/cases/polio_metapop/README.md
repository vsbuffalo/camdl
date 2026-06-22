# `polio_metapop` — spatial cVDPV2 metapopulation recovery case

A parameter-recovery case for a polio-shaped spatial metapopulation, mirroring
the structure of Daniel Klein's Sokoto cVDPV2 model (gh#207/#209): per
(ward × age) stratum `S, I_v, I_c, R` (vaccine-derived / circulating infecteds,
with `I_v → I_c` reversion), age-mixed force of infection, gravity-kernel
spatial coupling, full demography (aging, age-split waning, births, deaths), and
a single un-stratified `AFP` paralysis accumulator observed as monthly
incidence.

This case exists to validate the **fit machinery + scaling** on the spatial
polio structure, and to empirically answer "how many particles does PGAS need?"
on a model in this class (sweeping particle count vs recovery quality + the
`--pf-health` ESS/τ² diagnostic).

## Provenance — the model is generated

`model.camdl` is generated, not hand-written. Regenerate with:

```
python3 scripts/gen_scaling_models.py -P 12 -A 3 --shape polio \
    --coupling-degree 4 --observe --grad full --to "3 'years" \
    -o tests/recovery/cases/polio_metapop/model.camdl
```

P=12 wards × 3 age × {S,I_v,I_c,R} + AFP = 145 compartments, 577 transitions —
small enough to iterate, structurally faithful to the national model. The
committed `model.camdl` is a snapshot; `truth.toml` is the single source of
ground-truth parameters.

## Stages

| stage | what it tests |
| --- | --- |
| **Step 1 (here)** | `rho_afp = 0.3` (informative AFP), estimate only `R0_c` → clean recovery validates the machinery. |
| Step 2 (next) | sweep `particles ∈ {100,500,2000}` → recovery quality + `--pf-health` ESS/τ² + CSMC mixing → how many particles. |
| Step 3 | dial `rho_afp → ~1/200` (realistic opaque surveillance) → how far identifiability degrades. |
| Step 4 | scale `P → 244` once the small case is solid. |

## Run

```
make -f tests/recovery/Makefile CASE=tests/recovery/cases/polio_metapop synth
make -f tests/recovery/Makefile CASE=tests/recovery/cases/polio_metapop pgas
```

Recovery target: the `R0_c` posterior should concentrate near the truth (6.0),
started off-truth at 4.0. `data/` and `results/` are gitignored — the recipe
(`model.camdl` + `truth.toml` + `pgas.toml`) is the committed artifact.
