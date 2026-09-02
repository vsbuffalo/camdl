# Where a 3-province PGAS fit spends its time: hier3_ksmooth at 19,200 particles

Date: 2026-09-01\
Project: camdl\
Tags: profiling, pgas, inference, samply, binomial, flat-eval, memory

Sibling to
[`2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`](2026-08-24-pgas-binomial-sampler-is-half-the-fit.md),
which profiled the 2-province `bvd_province_nu_carec` (24 transitions, 9,600
particles) and found the binomial sampler at ~50% of the fit. This note profiles
the heavier sibling the ebola project actually runs now —
`bvd_province_hier3_ksmooth.camdl` under
`fit_province_hier3_ksmooth_19200_16c.toml` (42 compartments, 48 transitions, 12
observation streams at weekly cadence, ~104 substeps at dt = 1, 19,200
particles, 16 chains × 2,000 sweeps) — and the mix is different: rate-expression
eval overtakes the sampler, and the ancestor-sampling density pass grows from 6%
to a quarter of the work.

## Method

`--profile profiling` build, samply at default rate with
`--unstable-presymbolicate`, **serial** run (`--parallel 1`, 1 chain, 3 sweeps,
seed 5) so every busy sample is work; kernel-wait leaves (`__psynch_cvwait`,
`__semwait_signal`, `swtch_pri`) excluded. 92,993 busy samples. The machine was
also running two production fits during capture; that contaminates wall-clock
figures, not within-process attribution. Caveat carried from the sibling note:
NUTS did nothing in this probe (0% acceptance from an infeasible start at this
seed), so the θ-move's cost — up to 1,023 serial gradient walks per sweep on a
healthy chain — is absent from these shares.

## Phase attribution (share of busy samples)

| phase                                                  | share | `eval_resolved` leaf within |
| ------------------------------------------------------ | ----- | --------------------------- |
| free-particle propagation (`step_one`)                 | 56.0% | 19.5 pp                     |
| ancestor-sampling density (`log_transition_density_…`) | 25.3% | 11.2 pp                     |
| observation likelihood                                 | 10.1% | 4.0 pp                      |
| other (traceback, resample, history, …)                | 6.7%  | —                           |
| init pass / splice ratio / NUTS grad                   | 2.0%  | —                           |

Leaf view: `eval_resolved` 35.0%, binomial sampler + RNG ≈ 26% (`rand_distr`
BTPE 10.9 + `StatefulRng::binomial` 9.3 + ChaCha8 block 5.0 + `__powidf2` 0.7),
`lgamma` 3.3%, malloc/free ≈ 2%.

Against the 2-province note: sampler ~50% → ~26%, eval ~19% → ~35%, AS density
~6% → ~25%. The model is eval-heavier (double the transitions, forcings, the
importation sum), and 12 weekly streams mean many more resample substeps each
paying a 19,200-particle density pass.

## Levers, sized on this model

- **`binomial = "btrs"`** (typed stage field, this commit's sibling): sampler
  share ≈26% × measured 1.48× sampler ratio → implied ~1.09× on this fit.
  Measured end-to-end below now that the knob exists.
- **`CAMDL_EVAL_FLAT`**: targets the 35% eval share; the sibling corpus measured
  1.07× at a ~19% share, byte-identical. Expect more here; measured below. Still
  blocked from being a default by gh#746 (not in run identity) and the missing
  run-level A/B gate.
- **Memory, not speed**: one chain at 19,200 particles holds ~3.0 GB RSS —
  dominated by `csmc_as`'s four per-substep ensemble histories — so the 16-chain
  production config sits at the 48 GB machine's boundary. Filed as its own
  issue; ~2× reduction is available byte-identically (drop the derivable
  `history_counts_before`, arena-allocate the rest).
- **Not a lever here — the particle count.** The sibling note's "4× if renewal
  holds at N/4" does not read across: this probe's renewal profile is flat 0.000
  over bins b0–b7 with 0.760 in b9 (a coalesced conditional-SMC genealogy — the
  early path never renews), aggregate 7.8%, AS acceptance 0.027. On the fixture
  family, raising N is what moved a frozen prefix. The statistical read stays
  the maintainer's.

## End-to-end A/B — the first whole-fit sampler measurement

Serial (`--parallel 1`), interleaved arms, `--force`, seed 11, 1 chain, 5
sweeps + init pass per run; the two production fits above were running
throughout, so read user-CPU. Spread across reps is under 1%.

| arm         | user s (2 reps) | ratio vs btpe |
| ----------- | --------------- | ------------- |
| `btpe`      | 113.81 / 114.03 | 1.00×         |
| `btrs`      | 108.17 / 108.51 | **1.051×**    |
| `btpe`+flat | 113.23 / 114.92 | 1.00×         |
| `btrs`+flat | 108.76 / 108.42 | 1.049×        |

**BTRS delivers 1.05× on this model** — real, but below the 1.09× that Amdahl on
the ≈26% sampler share at the bench's 1.48× would imply, and well below the
sibling model's implied 1.15×. Two mechanisms fit: a quarter of this fit is the
ancestor-sampling density pass, which draws no binomials at all; and Haut-Uele's
small counts push more split draws under `BINV_THRESHOLD`, where the two arms
are the same code. The measured number is the one to quote.

**The flat rows are a finding about the toggle, not the evaluator.**
`CAMDL_EVAL_FLAT` moved nothing because the flat VM never engaged: a samply
profile of the flat run still shows `eval_resolved` as the top leaf (33.8%) and
no `eval_flat` frames. Cause, in `compiled_model.rs` (`flat_vm` construction):
the flat VM is built only when `model.per_eval_bindings` is empty — the gh#272
per-eval tape was deferred — and the guarding comment says "Default-off LICM ⇒
this is never hit today". **LICM is on by default**, and this model hoists 30
per-eval bindings, so on every model where LICM finds anything,
`CAMDL_EVAL_FLAT` is a silent no-op. The sibling note's measured 1.07× was
presumably a model with nothing to hoist. Head-to-head LICM-off numbers below
say whether finishing the per-eval tape is worth it on an eval-heavy model.

Byte-identity: `draws.tsv` under the flat env var is identical to without —
vacuously, since the VM never ran. The sibling's expression-level byte gate
(`flat_eval_byte_identity.rs`) remains the evidence for the evaluator itself.

## LICM against the flat tape, head to head

Flat only engages with LICM off, so the trade is measurable directly (same
protocol as above):

| configuration                   | user s (2 reps) | vs baseline |
| ------------------------------- | --------------- | ----------- |
| baseline (LICM, recursive eval) | 112.5 / 113.7   | 1.00×       |
| `CAMDL_NO_LICM=1`               | 119.8 / 120.9   | 0.94×       |
| `CAMDL_NO_LICM=1` + flat        | 105.5 / 105.3   | **1.073×**  |

LICM's hoist is worth 1.064× here (first two rows), and the flat tape beats the
recursive evaluator like-for-like by 1.14× (rows two and three) — enough that
flat on the _un-hoisted_ tree still beats LICM on the hoisted one. A flat VM
that also reads the per-eval scratch (gh#272 step 1.4) should stack both:
roughly **1.10–1.14×** over today's default on eval-heavy models. Filed with
these numbers as gh#815, together with the silent-no-op fix (warn when the
toggle cannot engage).

## What this session landed and what remains

- `binomial = "btrs"` typed stage field, threaded as a value to `step_one`
  (proposal step 2), commit `fb436aa3` — measured **1.051×** on this model. The
  flip (gh#761) remains blocked on gh#802/gh#803 and the healthy-chain regime
  re-harvest.
- gh#814: the csmc history memory lever (~3 GB/chain → 16 chains is the machine;
  ~2× reduction available byte-identically).
- gh#815: the flat×LICM gap above.
- Stacked today, `btrs` + (no-LICM + flat) ≈ **1.12×** is the measured available
  engineering factor on this model; the per-eval tape raises it to ~1.15–1.2×.
  The remaining large factors are structural: the ancestor-sampling density pass
  (25% — scales with streams × particles) and the statistical particle-count
  question, which this model's coalesced early-window renewal profile argues
  should not be answered by cutting N.
