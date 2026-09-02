# Where a PGAS sweep spends its time on the province model, and where the next factors are

Date: 2026-09-01 (extended 2026-09-02)\
Project: camdl\
Tags: profiling, pgas, inference, samply, binomial, flat-eval,
ancestor-sampling, memory

Sibling to
[`2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`](2026-08-24-pgas-binomial-sampler-is-half-the-fit.md),
which profiled the 2-province `bvd_province_nu_carec` and found the binomial
sampler at ~50% of the fit. This note profiles the model the ebola project
actually runs now — `bvd_province_hier3_ksmooth.camdl` under
`fit_province_hier3_ksmooth_19200_16c.toml`: 3 provinces, 42 integer
compartments, 48 transitions, 12 weekly observation streams, ~104 daily
substeps, 19,200 particles, 16 chains × 2,000 sweeps — and the mix is different.
It then measures the levers the profile exposes, and closes with the two places
anything larger than ~1.1× can come from: the ancestor-sampling pass, and sweep
efficiency.

## The breakdown

Share of CPU work for one chain, serial run, kernel-wait samples excluded so
every sample is work. One `█` ≈ 2%.

```
████████████████████████████                 free-particle propagation                 56%
    ██████████▌                                binomial draws (BTPE + BINV)             21%
    █████████▊                                 rate-expression eval (recursive walk)    20%
    ███                                        step_one bookkeeping (splits, stoich)     6%
    ██▌                                        ChaCha8 random bytes                      5%
    ██                                         forcing, gamma noise, libm, misc          4%
████████████▋                                ancestor-sampling density pass            25%
    █████▌                                     rate-expression eval                     11%
    ███▏                                       libm exp/ln (binomial probabilities)      6%
    █▊                                         binomial log-pmfs                         4%
    ██                                         scratch, context setup, misc              4%
█████                                        observation likelihoods                   10%
    ██                                         rate/dispersion expression eval           4%
    █▏                                         libm exp/ln                               2%
    ▋                                          neg-binomial / beta-binomial log-pmfs     1%
    █▌                                         projection fold, misc                     3%
███▎                                         resample, history, traceback, misc         7%
▊                                            init pass (once per chain)                 2%
▏                                            AS splice suffix ratios                   <1%
                                             NUTS θ-move                               ~0%  ← caveat below
```

Three headline facts, against the sibling model's profile:

1. **Rate-expression evaluation is the biggest single cost (~35% across all
   phases)**, not the binomial sampler (~26% here vs ~50% there). This model has
   twice the transitions, forcings in every infection rate, and an importation
   sum.
2. **The ancestor-sampling density pass quadrupled as a share** (6% → 25%).
   Twelve weekly streams mean a couple dozen resample substeps per sweep, and
   each one evaluates a full transition density for all 19,200 particles.
3. **Allocation is invisible** (<1% incl. malloc/free), replicating the sibling
   note's finding 2(a). Memory is a _footprint_ problem here, not a speed
   problem — see gh#814.

The `~0%` NUTS row carries the same caveat as the sibling note: in this probe
the chain started infeasible, so the θ-move did nothing (`n_leapfrog = 1`, 0%
acceptance). On a healthy chain NUTS performs up to 1,023 serial gradient walks
of the whole trajectory per sweep; its true share on a converged production
chain is unmeasured. Re-profile a healthy chain before trusting these shares to
the last few points.

### Reading guide: what happens in one sweep

PGAS alternates two moves. The **θ-move** (NUTS) updates parameters against the
current latent trajectory. The **X-move** (`csmc_as`) refreshes the trajectory:
19,199 free particles are simulated forward substep by substep (`step_one`:
evaluate all 48 rates, draw binomial exit counts, split them across competing
transitions), while slot 19,200 replays the retained _reference_ trajectory from
the previous sweep. At each observation the particles are weighted and
resampled, and _ancestor sampling_ tries to splice the reference onto a
different particle's history — that is the move that lets the past of the
trajectory change. At the end, one particle's ancestral path becomes the new
trajectory. The flamegraph rows are exactly these pieces.

## Method, briefly

`--profile profiling` build; samply with `--unstable-presymbolicate`; serial run
(`--parallel 1`, 1 chain, 3 sweeps, seed 5) so idle threads can be excluded
cleanly (92,993 busy samples). Wall-clock numbers below use user-CPU from
`/usr/bin/time` with arms interleaved, because two production fits shared the
machine throughout; within-process attribution is unaffected. Repro commands at
the end.

## Levers measured this session

### The `binomial = "btrs"` field: 1.051× on this model

The typed sampler field (proposal
[`2026-08-24-faster-binomial-sampler.md`](../proposals/2026-08-24-faster-binomial-sampler.md)
step 2) made the whole-fit factor measurable for the first time. Serial,
interleaved, seed 11, 5 sweeps + init pass; spread across reps under 1%:

| arm    | user s (2 reps) | ratio      |
| ------ | --------------- | ---------- |
| `btpe` | 113.8 / 114.0   | 1.00×      |
| `btrs` | 108.2 / 108.5   | **1.051×** |

Below the ~1.09× that Amdahl on a 26% share at the bench's 1.48× sampler ratio
would predict, for two visible reasons: a quarter of this fit draws no binomials
at all (the AS pass), and Haut-Uele's small counts push more split draws under
`BINV_THRESHOLD`, where both arms are the same BINV code. On the
sampler-dominated `nu_carec` family the implied factor is ~1.15×; measuring it
there is now one `binomial = "btrs"` line in the stage TOML.

### `CAMDL_EVAL_FLAT` is a silent no-op on every LICM model (gh#815)

The flat rows of the A/B moved nothing — 1.00× — and the reason is a finding
about the toggle, not the evaluator: the flat VM is built only when
`per_eval_bindings` is empty (`compiled_model.rs`, behind a comment that still
says LICM is off by default). LICM is **on** by default and hoists 30 bindings
from this model, so the env var validates and then does nothing. Verified by
profile, not inference: under `CAMDL_EVAL_FLAT=1` the top leaf is still
`eval_resolved` (33.8%) and no `eval_flat` frame appears.

Forcing the comparison by disabling LICM:

| configuration                   | user s (2 reps) | vs baseline |
| ------------------------------- | --------------- | ----------- |
| baseline (LICM, recursive eval) | 112.5 / 113.7   | 1.00×       |
| `CAMDL_NO_LICM=1`               | 119.8 / 120.9   | 0.94×       |
| `CAMDL_NO_LICM=1` + flat        | 105.5 / 105.3   | **1.073×**  |

Read the rows together: LICM's hoist is worth 1.064×, and the flat tape beats
the recursive evaluator like-for-like by 1.14× — enough that flat on the
_un-hoisted_ tree still beats LICM on the hoisted one. Finishing the gh#272
step-1.4 per-eval tape should stack the two: roughly **1.10–1.14×** on
eval-heavy models. Filed as gh#815 with these numbers, plus the immediate fix
(warn when the toggle cannot engage).

### Memory: one chain is ~3 GB, sixteen chains is the machine (gh#814)

`csmc_as` stores four full per-substep ensemble histories as nested vectors; at
19,200 particles that is ~3.0 GB per chain, so the production 16-chain config
sits at the 48 GB boundary — and a swapping fit loses more wall time than every
lever in this note combined. `history_counts_before` is exactly derivable from
`counts_after` plus the ancestor vectors, and flat arenas remove ~8M allocations
per sweep; together ~2× RSS, byte-identical. Filed as gh#814. A speed non-event
(see headline fact 3); an operational necessity.

## The ancestor-sampling pass: 25% of the fit for a move that almost never fires

This is the largest _engineering_ target left, and it is really a statistics
question wearing an engineering costume.

What the money buys today: at each resample substep, `fill_ancestor_log_weights`
evaluates log f_θ(x′ₛ | x_{s−1}^j) for **every** particle j — 19,200 transition
densities — to draw one categorical sample: the reference's new ancestor. The
diagnostics say that draw then goes almost nowhere: AS acceptance 0.027 on this
probe, and trajectory renewal 0.000 over the first seven tenths of the series.

Why it never fires: for a discrete-state model the density is unforgiving.
Another particle's state can host the reference's recorded flows only if every
source compartment holds enough individuals _and_ the state consumes exactly the
recorded number of gamma multipliers (the gh#607 −inf rules). With 42 integer
compartments the chance that an independently-evolved particle satisfies all of
it is small, so almost every ancestor weight is −inf, the categorical
concentrates on the reference itself, and the rare proposal dies on the suffix
ratio. This is intrinsic to ancestor-splicing in high-dimensional integer state
spaces, not a defect in the implementation.

Three responses, in increasing ambition. Each names its decision statistic up
front; the right one is **ESS(θ) per CPU-second** from `draws.tsv`, read beside
the renewal profile — not renewal alone (a slot-identity statistic), and not
wall time alone.

**E1 — measure what AS is worth (free).** Run matched probes with ancestor
sampling disabled against current. If ESS/CPU-s and the renewal profile are
indistinguishable — which acceptance 0.027 and the zeroed early bins predict —
the pass is nearly pure overhead on this model, and any cheaper valid proposal
is a free win. This experiment gates everything below.

**E2 — a multiple-try ancestor proposal (~1.3× if E1 reads as expected).** The
code already treats the screened weights as an _independence proposal_ with an
exact MH correction (the LJS §6.1 structure gh#607 built), so the expensive part
is the proposal, not the correction. Replace it with a multiple-try move:
subsample M candidates from the filter weights (already computed, free),
evaluate densities only for those, select among them, and accept with the
multiple-try Metropolis ratio (Liu, Liang & Wong 2000, JASA 95(449):121–134 —
transcribe the reverse-set construction from the paper, not from memory). Cost
falls from N to ~2M densities per AS step; at M = 256, N = 19,200 the 25%
becomes ~1%, a ~1.3× whole-fit factor, with little mixing at risk given how
little the full-price move currently buys. Tunable by M, exact for any M.

**E3 — backward simulation (PGBS): spend more, to buy back sweeps.** The
structural alternative is to replace the single splice point with a full
backward-simulation traceback (Whiteley's discussion of Andrieu, Doucet &
Holenstein 2010, JRSS-B 72:269–342; Lindsten & Schön 2013, FnT ML 6(1):1–143):
draw the new trajectory backward _through the particle cloud_, renewing every
segment rather than one prefix. It costs N densities at every substep — roughly
3–4× the current AS budget, so ~1.7× slower sweeps — and it attacks exactly the
coalescence that is frozen here. The same −inf hosting problem applies to its
backward weights, so it may disappoint for the same reason AS does; that is
precisely what the experiment is for. If it unfreezes the prefix, the sweep
count (and possibly the particle count) falls, and those multiply — this is the
one candidate for an integer factor.

## Sweeps, not seconds

Everything above shaves cost-per-sweep. The fit's total cost is chains × sweeps
× cost(sweep), and the first two factors are statistical:

- **The frozen prefix is a scientific issue before it is a performance one.**
  Renewal 0.000 over bins b0–b7 means the latent path over the early window is
  never re-drawn across the whole run — the early-window posterior path is
  inherited from initialization, and apparent stationarity there is not evidence
  of mixing. Whatever fixes this (E3, more particles, a different early-window
  treatment) is worth more than any 1.1×. Note the direction: the sibling note's
  "4× by cutting N" hope does **not** read across to this model — the coalesced
  prefix argues N is too small here, not too large.
- **`csmc_sweeps_per_nuts = 3–5`** exists precisely for series where the
  trajectory move is the bottleneck (its own doc says so) and costs one line in
  the stage TOML. Cheap arm to add to the E1–E3 sweep.
- **~44% of prior draws are refused at init** (the config's own header), so a
  material fraction of chains contribute nothing. The chain-viability work in
  flight addresses the same waste from the other end.

## Ceiling arithmetic

If the sampler _and_ all expression eval were literally free, this model's fit
would run 1/(1 − 0.26 − 0.35) ≈ 2.6× faster — the bound on all engineering
micro-work combined. Bankable today: btrs (1.051×) stacked with the per-eval
flat tape (~1.07–1.14×, gh#815) ≈ **1.12–1.2×**, plus a plausible ~1.05–1.08×
from the Phase-2 PRNG swap. E2 adds ~1.3× if E1 reads as predicted. Integer
factors, if they exist, are in E3 and in sweep efficiency.

## Status ledger

- `binomial` stage field: PR#816 (this branch); an independent implementation
  carrying the algorithm on `StatefulRng` is in flight and is the better seam —
  the branches share the TOML surface and identity bytes, so the measurements
  here transfer to either.
- gh#814 (history memory, ~2× RSS), gh#815 (flat×LICM, ~1.1×): filed with
  measurements.
- gh#761 (make BTRS the only sampler) stays blocked on gh#802/gh#803, the
  backend-wide baseline re-capture, and a healthy-chain regime re-harvest.
- E1–E3 need a short proposal with the MTM correction and the PGBS backward
  kernel derived for the chain-binomial density before any code.

## Repro

```bash
make build   # both toolchains
export CAMDLC=$PWD/ocaml/_build/default/bin/camdlc.exe

# work-only profile (serial; exclude kernel-wait leaves in analysis)
cargo build --profile profiling -p cli
samply record --save-only --unstable-presymbolicate -o prof.json.gz -- \
  rust/target/profiling/camdl fit run probe.toml --parallel 1 --seed 5

# view interactively (Firefox Profiler UI, served locally; nothing uploads)
samply load prof.json.gz

# whole-fit A/B: same config ± `binomial = "btrs"`, serial, interleaved,
# `--force`, read user-CPU from /usr/bin/time
```

The probe configs are the production TOML with `chains = 1`, `sweeps = 3..10`,
`burn_in = 0`; particle count kept at 19,200 so shares are the production
shares.
