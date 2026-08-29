# Correlated PMMH on a reporting series with a missing day

Date: 2026-08-28 Related: gh#193 (the CPM obs-grid gate), gh#669 (the junction
workaround), PR #762

A downstream Ebola team could not run correlated pseudo-marginal PMMH (CPM) on
their national series. Their observation grid has 96 of 97 daily times: one day
carries no situation report. CPM refused the config, and they asked which of two
workarounds to build around. Both answers were in our code rather than in our
judgement, so this note records them with their sites, and what changed as a
result.

## An absent row and an `NA` row are different objects

They proposed writing `NA` in the value column for the missing day, on the guess
that this would be equivalent to omitting it. It is not, and the difference is
load-bearing for any stream projecting `incidence(...)`.

`long_form_absent_row_is_not_scheduled` (`rust/crates/cli/src/pfilter.rs`) draws
the distinction directly. An **absent** row means the time is not on that
stream's own axis: no score, and no reset of the incidence accumulator, so the
flow keeps accumulating into the following window. An **`NA`** row means the
time _is_ on the axis and the cell is a hole: no likelihood term, but the
accumulator still resets there, because that is what an observation time means
(pomp's `accumvars` fixed-bin semantics).

For a daily case series with no report on one day, the absent row is the correct
encoding: the next reported count covers two days of flow, and that is what the
model should be asked to explain. Writing `NA` would close the bin on the
unreported day and attribute two days of incidence to a one-day window — a quiet
mis-specification, not an error.

That is why the workaround the team was ready to adopt was the wrong one, and
why the grid had to reach the filter irregular.

## What CPM actually required

CPM pre-draws the particle filter's random numbers so that the same draw is
reused at the same (window, particle, substep) across MCMC iterations; that
reuse is what makes the likelihood ratio in the acceptance step far less noisy
than either estimate. The draws are stored per observation window
(`PFRandomState.gamma_noise` / `binomial_noise`, both `Vec<Vec<f64>>`), so the
structure was already jagged — the inner arrays were uniform only because
`draw_fresh` took a scalar substep count and applied it to every window.

The team's own proposal was to size every block from the longest window and read
a prefix, accepting the wasted memory. That price was not necessary: sizing each
row at its own window's substep count costs nothing and wastes nothing. CPM's
correctness requirement survives untouched, because the irregularity is static —
the grid and `dt` are fixed at setup, so window `i` always has the same substep
count and always maps to the same slots. The Crank-Nicolson update is
elementwise and length-agnostic.

The work was in the read sites, which computed
`particle * steps_per_obs +
substep` from one global stride. There are exactly
two, both in `bootstrap_filter_correlated`'s per-particle loop, and each now
strides by its own window's count. A wrong stride there does not fail: it reads
a valid float from another particle's slot, so two particles share a draw and
the swarm loses the independence a particle filter needs, with nothing to see.

Landed in PR #762, which also replaced the uniformity gate with a pre-run
reconciliation of the drawn noise against the walk the filter is about to
perform, so the indexing is sound by construction rather than by validation.

## A stale paragraph that would have sent them the wrong way

`docs/camdl-language-spec.md` §12.1 said `incidence(a) + incidence(b)` does not
sum, and recommended routing distinct flows through one junction transition.
Both claims are stale. The summed form compiles today and lowers to a single
`CumulativeFlowSum` over both flows (`d528e024`), and the junction — a
compartment nobody is in, plus a magic rate that imposes a real dwell — is the
workaround gh#669 argues against and whose recommendation was removed from the
`E203` hint in `70247af6`. §16 of the same file has documented the summed form
correctly since B1a landed, so this was one file contradicting itself.

The team's `cases_national` is a four-term sum of exactly that shape. Had they
read §12.1 before writing the model, it would have cost them a junction
compartment per route.

Corrected in this note's commit. The same paragraph's companion at §12.1's
"head-position sugar" entry was stale in the same direction: wrapping a
projection in arithmetic is now `E341`, named, rather than `E100`, undeclared
function — and addition of incidence terms is admitted, while weighting,
subtraction, and mixing with a state read are not.
