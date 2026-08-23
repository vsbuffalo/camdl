# PGAS has been sampling from the wrong distribution since April

- Date: 2026-08-23
- Issue: gh#718
- Defect 1: fixed by `55178dd1`
- Defect 2: confirmed; fix decided (Part 7), implementation in progress
- Class: **code-vs-code** — the code disagreed with the mathematics it is an
  implementation of. Both defects are errors of composition rather than of any
  single function; see Part 5.

## Summary in one paragraph

PGAS is camdl's default method for stochastic Bayesian fitting. Its correctness
rests on a single property, and we found two separate places where the code
violates it. The consequence is not "noisier answers" or "slower convergence" —
it is that the posterior distributions PGAS reports are not the posterior
distributions that were asked for. They are shifted, systematically, in a way
that **no convergence diagnostic we compute can detect**. Every PGAS fit
produced since 2026-04-05 is affected. One defect is fixed; the second is
confirmed and needs a design decision before it can be fixed.

---

## Part 0: the vocabulary, so the rest of this document parses

This section defines every term the investigation uses. Skip it if the machinery
is already familiar.

**The thing we are trying to sample.** When we fit a model, there are two kinds
of unknown: the _parameters_ (θ — transmission rate, recovery rate, and so on)
and the _latent trajectory_ (X — how many people were actually in each
compartment at each moment, which we never observe directly). PGAS alternates
between updating θ and updating X. The target for the X update is written
`p(X | θ, y)`: the distribution of plausible trajectories, given the parameters
and the observed data `y`. This document calls it **the target**.

**Particles.** To update X, camdl simulates many candidate trajectories forward
in time at once. Each candidate is a **particle**. In production runs there are
typically a few thousand; the experiments below use 5, deliberately, because a
small ensemble makes any defect large enough to measure.

**Substeps.** The simulation advances in small fixed increments called substeps.
Observations are much sparser than substeps — a model might take 15 substeps but
only have data at 3 of them.

**Weights and resampling.** When the simulation reaches a time where data
exists, each particle is scored by how well it matches that observation. That
score is the particle's **weight**. Particles with low weight are wasteful to
keep simulating, so the ensemble is **resampled**: a new ensemble of the same
size is drawn from the old one, with each particle chosen in proportion to its
weight. Good particles get duplicated, bad ones are dropped. Between
observations there is no new information, so all weights are equal and camdl
skips resampling entirely — this skip turns out to matter enormously.

**The reference trajectory, and why this is "conditional".** PGAS is a Markov
chain: each sweep produces one trajectory, which becomes the starting point for
the next sweep. That carried-over trajectory is the **reference trajectory**.
Ordinary particle filtering has no such thing; PGAS forces one particle slot to
hold the reference and be immune from being discarded. That is what makes the
algorithm _conditional_ SMC rather than ordinary SMC, and it is the single
structural difference between them. In camdl the reference occupies the last
slot, `j_ref = n - 1`.

**Ancestor sampling (AS).** This is the "AS" in PGAS. Without it, the reference
trajectory stays attached to its own past forever and the chain mixes very
slowly. Ancestor sampling lets the reference _detach_ from its own history at
some substep and _re-attach_ to one of the other particles' histories instead.
This document calls that move a **splice**. Whether a proposed splice is
accepted is decided by a Metropolis accept/reject test.

**Invariance — the property that has to hold.** PGAS is correct if and only if
one sweep of it leaves the target unchanged: feed it a trajectory drawn from
`p(X | θ, y)`, and what comes out must also be a draw from `p(X | θ, y)`. This
is called **invariance**. If it fails, the chain still converges — it just
converges to the wrong distribution. That is bias, and bias does not go away
with more iterations.

**Systematic vs multinomial resampling.** Two ways to draw the new ensemble.
_Multinomial_ means each new slot independently picks a particle at random with
probability equal to its weight — simple, and every slot is independent of every
other. _Systematic_ lays the particles end to end on a line segment of length 1,
each occupying a piece as wide as its weight, then places `n` evenly spaced tick
marks along that line and takes whichever particle each tick lands in.
Systematic has less randomness and so wastes fewer particles, which is why it is
the standard choice — but the `n` picks are strongly dependent on each other,
because they are locked to a rigid grid of ticks. Each tick's interval is called
a **stratum**.

> **A warning about the word "stratum".** In this document it means one of those
> `n` evenly spaced intervals of the resampling line — a purely numerical
> construct inside the resampler, with no epidemiological content. It has
> **nothing to do with `stratum` in the camdl DSL**, where it names a
> stratification of the population (age band, patch, health zone) declared in an
> observation block. The two are unrelated and it is unfortunate that the
> standard term for each is the same word.

---

## Part 1: how we measured "wrong"

The problem that had blocked this investigation for days was the absence of a
ground truth. To ask "does one sweep leave the target unchanged?" you need to
know the target exactly, and for a realistic model you cannot: you can only
estimate it by simulation, and then every comparison is polluted by the
estimate's own error. The original gh#718 measurement needed 800,000 simulated
draws to build a ground truth, and its verdict rested on a gap of 11 standard
errors. A follow-up run at 32 particles lost the ability to detect a bias it was
certain was present.

**The fix was to shrink the model instead of growing the sample.** A plain SIR
model without overdispersion has no continuous randomness anywhere — a
trajectory is completely determined by how many people moved along each arrow at
each substep, and those are small whole numbers. With a population of 6 over 4
substeps there are exactly **3,538 possible trajectories**. That is few enough
to write them all down, score each one, and normalise. The target is then known
_exactly_, to floating-point precision, with no simulation error at all.

The test becomes the literal definition of invariance:

```
draw X₀ from the exact target
X₁ = one PGAS sweep applied to X₀
question: is X₁ also a draw from the exact target?
```

Repeat that a few hundred thousand times, tally which of the 3,538 trajectories
came out, and compare the tally to the known probabilities. Under the null
hypothesis "the kernel is invariant" the tally is a multinomial sample with
known probabilities, so this is an ordinary goodness-of-fit test. We report it
as **z**, a standardised chi-square: **z near 0 means the kernel passed, z above
about 5 means it failed.** Because the target has no Monte-Carlo error, a large
z cannot be blamed on the ground truth.

**Before trusting any of this we checked the foundation.** The exact target is
built from `complete_data_loglik`, the function that scores a trajectory's
probability. If that function is not the actual law the simulator draws from,
the whole exercise is meaningless — and note this assumption was also silently
underneath gh#718's original 800,000-draw ground truth. So we drew one substep
400,000 times from the simulator and compared the observed frequencies to the
scoring function cell by cell:

```
Sum of scored probabilities over the substep's support = 1.000000
Worst cell disagreement over 400,000 draws:            |z| = 2.33
```

The scoring function is exactly the simulator's law. The instrument is sound.

**Three fixtures.** A _fixture_ here means one specific tiny model plus one
specific schedule of when observations occur. Three were used, and the
differences between them turn out to be the whole story:

| name       | shape                                             | why                                                                                                                                              |
| ---------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DENSE**  | 4 substeps, an observation at _every_ substep     | resampling happens at every substep; the "skip resampling" path is never taken                                                                   |
| **SPARSE** | 4 substeps, observations at substeps 1 and 3 only | 3 of the 4 substeps have equal weights, so resampling is skipped there                                                                           |
| **TRAP**   | 2 substeps, one observation at the very end       | built later, and deliberately: the one meaningful ancestor-sampling move happens with resampling skipped, and **nothing afterwards can undo it** |

---

## Part 2: the investigation, hypothesis by hypothesis

Each heading carries its verdict, so the two that mattered are findable without
reading the one that did not.

### Hypothesis 1 (REFUTED) — the ancestor weight was being scored at a stale state

**What we thought.** `fill_ancestor_log_weights` scores each candidate the
reference could re-attach to. It took the reference's own state through a
separate argument named `ref_counts_before`. That name means "the reference
trajectory's _recorded_ starting state", and since gh#607 an accepted splice can
move the reference away from that recorded state, the name suggested the code
was scoring a stale value — which would break the cancellation the accept/reject
test depends on.

**How we tested it.** Read the call site and traced every write to the buffers
involved; then ran an independent adversarial review of the same claim.

**Result: refuted.** The call site passes `prev_counts[j_ref]`, which is the
reference's _current_ state, not its recorded one. The argument's name was stale
but its value was correct, and the branch selecting it was choosing between two
buffers holding identical values. The upstream review reached the same
conclusion independently, making three separate confirmations.

**What it means.** No behavioural defect. But the misleading name had now sent
two readers down the same wrong path, so it was removed in `b7279b5e` — a
behaviour-neutral change that makes the file stop suggesting a bug that isn't
there.

### Hypothesis 2 (CONFIRMED — this is defect 1) — the resampling discards one of its own picks

**What we thought.** Reading the resampling block, `systematic_resample` is
asked for `n` picks — one per particle — but the result is only used for the
slots that are _not_ the reference:

```rust
for j in 0..n_particles {
    if j == j_ref { new_counts.push(counts[j_ref].clone()); }   // reference keeps itself
    else          { new_counts.push(counts[indices[j]].clone()); }
}
```

With `j_ref = n - 1`, the pick `indices[n-1]` is computed and never read. In
systematic resampling the picks are ordered — pick 0 comes from the leftmost
stratum, pick `n-1` from the rightmost. **So throwing away the last pick is not
throwing away a random one. It throws away the far right of the line**, and the
reference particle is laid down last, so the far right of the line is exactly
where the reference sits.

**How we tested it.** Directly, with no model and no simulation: run the
resampler many times on a fixed set of weights and count how often each particle
is chosen for the free slots, against how often it should be.

**Result: confirmed, and the effect is total rather than marginal.**

| reference's weight | share it is owed | share it actually got | every other particle |
| -----------------: | ---------------: | --------------------: | -------------------: |
|               0.10 |           0.1000 |            **0.0000** |       inflated ×1.25 |
|               0.40 |           0.4000 |                0.2500 |       inflated ×1.25 |

When the reference's weight is below `1/n`, it gets **no descendants at all**.
Not few — none. And this does not improve with more particles, because the
amount lost is about one slot regardless of `n`, while a typical particle only
expects about one slot to begin with:

| particles | share of its fair descendants the reference loses |
| --------: | ------------------------------------------------: |
|         5 |                                               59% |
|        20 |                                               61% |
|       100 |                                               63% |
|       500 |                                               65% |

**What it means.** The reference particle can take a free particle's history
(via ancestor sampling) but the free particles can almost never take the
reference's. History flows one way. That asymmetry is a direct violation of the
invariance requirement.

**End-to-end confirmation on the DENSE fixture** (M = 400,000 draws each):

| ancestor sampling | resampling scheme              |         z |
| ----------------- | ------------------------------ | --------: |
| off               | as shipped                     |  **6.52** |
| off               | n−1 strata (an ad-hoc variant) |      1.27 |
| off               | multinomial                    |      1.54 |
| **on**            | **as shipped**                 | **10.76** |
| on                | n−1 strata                     |      2.81 |
| on                | multinomial                    |  **0.05** |

Two things to read off this table. First, **the top row is the important one**:
with ancestor sampling switched off entirely — plain particle Gibbs, the arm
gh#718 used as its known-good control — the kernel _still_ fails. The defect is
upstream of ancestor sampling. Second, fixing only the resampling (bottom row)
takes the full shipped kernel from 10.76 to 0.05.

**The fix.** Chopin & Singh (2015), _On particle Gibbs sampling_, Bernoulli
21(3):1855–1883, address exactly this. They require the resampling scheme to be
**marginally unbiased** — the chance of any one slot picking particle `m` must
equal `m`'s weight — and give the conditional systematic construction that
achieves it as their Algorithm 4. It has three steps: condition the starting
offset so the reference is guaranteed an offspring with the correct distribution
for how many; run ordinary systematic selection at that offset; then rotate the
result so one of the reference's own copies lands in the reference slot.
`conditional_systematic_resample` implements it. Validated at 3× the power:

| ancestor sampling | z (M = 1,200,000) |
| ----------------- | ----------------: |
| off               |             −0.11 |
| on                |             +0.39 |

All three steps of the algorithm were mutation-checked — each deliberately
broken in turn to confirm a test catches it. One early attempt survived mutation
because the test's weights made `n × w_ref` exactly 2.0, which makes step one of
the algorithm a no-op; the test now uses three weight vectors chosen so every
branch is exercised.

### Hypothesis 3 (CONFIRMED — this is defect 2) — the ancestor move is invalid wherever resampling is skipped

**Where this came from.** An upstream review of gh#718, which did not agree with
our diagnosis and proposed a different and deeper one.

**What it said.** The accept/reject test for a splice is
`α = S_new / S_current`, where `S` covers the trajectory's remaining
probability. That simple form is only correct if the resampling picks are
_independent of each other_, which is true for multinomial and false for
systematic. More sharply: **when weights are equal, camdl skips resampling
altogether, so every particle keeps its own history.** Ancestor sampling then
moves the reference onto someone else's history anyway — producing an ancestry
that the resampling step had assigned probability _zero_. The reference takes,
and nothing can give back, because there is no resampling step to give with.

**How we tested it, part one: reproduce their counterexample independently.**
The review included a two-state hidden Markov model with one transition, small
enough to enumerate every random choice exactly — no simulation, no sampling
error at all. We re-implemented it from their description without looking at
their code:

| particles | keep-own-history ancestry + AS | multinomial ancestry + AS |
| --------: | -----------------------------: | ------------------------: |
|         2 |                  **8.124e-03** |                   5.6e-17 |
|         3 |                  **8.351e-03** |                   6.7e-16 |
|         4 |                  **7.046e-03** |                   3.8e-14 |
|         5 |                  **5.682e-03** |                   2.9e-12 |

(The multinomial column is machine-precision zero throughout; its drift is
accumulated floating-point from a progressively larger enumeration, not signal.)
Our numbers match theirs to three significant figures — they reported 8.12e-3
and 1.7e-16. **The error does not shrink with more particles.**

**How we tested it, part two — and this is where we were initially wrong.** We
ran the SPARSE fixture, which has 3 of its 4 substeps on the skip-resampling
path. It came back clean (M = 150,000):

| SPARSE                                      |     z |
| ------------------------------------------- | ----: |
| shipped                                     | −0.36 |
| force multinomial at every substep          | −1.25 |
| suppress AS wherever resampling was skipped | −0.29 |
| ancestor sampling off (control)             | −1.10 |

No difference between any arm. At this point our reading was that the review's
mechanism might not apply to camdl. **That reading was wrong, and the reason is
instructive**: SPARSE has one substep that _does_ resample, and that step lets
the reference's history flow back to the free particles, repairing the asymmetry
one substep later. The review's counterexample has only one transition and so no
opportunity for repair.

**How we tested it, part three: build a fixture with no escape hatch.** The TRAP
fixture is two substeps with the only observation at the very end. Substep 0 has
equal weights, but every particle shares the same deterministic starting state,
so its ancestor move changes nothing. Substep 1 also has equal weights — and by
then the particles have diverged, so _its_ ancestor move is real, it happens
with resampling skipped, and **there is no later resampling step to repair it**.
This is the camdl equivalent of the review's counterexample.

**Result: confirmed.** At M = 400,000 the shipped arm was the only elevated one
(z = 2.36 against −0.37, −0.57, 0.43), suggestive but not conclusive. The
fixture is tiny and cheap, so we bought more power — and a real bias grows with
sample size while noise does not:

| TRAP, M = 3,000,000                         | chi-square / df |         z |
| ------------------------------------------- | --------------: | --------: |
| **shipped**                                 |           1.745 | **+4.99** |
| force multinomial at every substep          |           1.160 |     +1.28 |
| suppress AS wherever resampling was skipped |           1.121 |     +0.98 |
| ancestor sampling off (control)             |           1.039 |     +0.35 |

The shipped arm's excess grew from 1.32 to 1.75 as the sample grew — the
signature of genuine bias. Both proposed remedies remove it.

**What it means.** The review is right, this is a second and independent defect,
and **the fix for defect 1 does not touch it** — defect 1 lives on substeps that
resample, defect 2 lives on substeps that don't.

---

## Part 3: the two defects are one pathology

Both defects are the same sentence: **the reference particle can take history
from the free particles, but the free particles can never take history from the
reference.**

- On substeps _with_ an observation, the resampling discarded the stratum the
  reference sits in, so the free particles could not inherit from it.
- On substeps _without_ an observation, resampling is skipped, so the free
  particles could not inherit from anything — while ancestor sampling let the
  reference keep taking.

Every measurement in this document is an instance of that asymmetry.

---

## Part 4: what this cost

**Every PGAS posterior since 2026-04-05 is drawn from the wrong distribution.**
Not noisier — shifted.

**No diagnostic we compute could have caught it.** R̂ compares chains against
each other and ESS measures within-chain autocorrelation. Both defects bias
every chain _identically_, so the chains agree perfectly: R̂ ≈ 1.00, healthy ESS,
wrong answer. This is exactly the failure mode CLAUDE.md names as the worst
outcome available.

**It probably made mixing look better.** Starving the reference of descendants
makes the chain abandon its current trajectory more readily than it should. So
the bug would have presented as _good_ trajectory renewal, never as a symptom.

**It sent a multi-day investigation to the wrong surface.** gh#718 measured the
non-invariance correctly and attributed it to the ancestor-sampling splice,
because it used plain particle Gibbs as a _provably-correct control_. Provably
correct as an algorithm — but this implementation of it shares the resampler, so
the paired comparison subtracted one biased kernel from a more biased one and
read the residual as a splice defect. Everything gh#718 _excluded_ was excluded
correctly. The splice itself was never wrong.

**Not affected:** the bootstrap particle filter, IF2, correlated PF, and PMMH.
All of them fill every slot from every pick, because none of them has a
reference particle to protect. Only `csmc_as` does.

---

## Part 5: what kind of errors were these?

Neither defect is a bug in any single function. `systematic_resample` is
correct. The loop that applies it is correct — its bounds are right and it
touches every slot it means to. `splice_log_ratio` is correct, and was verified
against an independent oracle before any of this. **Both defects are errors of
composition**: correct pieces assembled under an assumption that nothing in the
program recorded.

**Defect 1's class: calling a primitive that answers a neighbouring question,
then discarding the mismatch.** The operation actually needed was "draw the
ancestors for the `n-1` slots that are not the reference". No such function
existed. Instead the code called the function for a _different_ question — "draw
ancestors for all `n` slots, with no reference to protect" — and dropped the
part of the answer that did not fit. Dropping part of a joint draw silently
changes its distribution, which is invisible at the call site because the
discarded value is simply never read.

This has the _appearance_ of an off-by-one, since `j_ref = n - 1` and the
discarded pick is `indices[n - 1]`. It is not one, and that distinction matters
for prevention: re-indexing would not have fixed it. Had the reference been
placed at slot 0, the code would have discarded the leftmost stratum instead and
starved the reference just as thoroughly. The error is that a joint draw was
treated as if its components were separable.

**Defect 2's class: a borrowed derivation whose stated precondition was never
carried into the code.** The ancestor-sampling accept/reject formula comes from
Lindsten, Jordan & Schön (2014), where it is derived for a particle system whose
resampling picks are _independent across slots_ — multinomial. camdl resamples
systematically, and skips resampling entirely when weights are equal. The
formula was transcribed faithfully; the condition under which it is a valid
formula was not. Nothing in the code, its types, its comments, or its tests
recorded that the formula had a precondition at all, so there was nothing to
notice was violated.

**What the two share.** In both cases a correctness precondition lived only in
the mathematics — "the resampling must be marginally unbiased", "the ancestor
picks must be independent across slots" — and had no representation in the
program. A precondition with no representation cannot be checked, cannot be
tested, and cannot fail loudly. It can only be violated silently.

## Part 6: retrospective — how this could have been caught; prevention

### Why four months passed

The suite had four tests on this kernel and all four passed throughout. They
pinned the accept/reject ratio against an independent oracle, the
ancestor-sampling weight formula, the continuity of the returned path, and a
byte-level digest. Every one of them checks a _part_. None asked the question
the algorithm exists to answer: does a sweep applied to a draw from the target
return a draw from the target?

Nothing else could have surfaced it either. Both defects bias every chain
identically, so R̂ stays at 1.00 and ESS stays healthy — the diagnostics were
working correctly and reporting truthfully on a quantity that is blind to this
entire class of error. And because starving the reference makes the chain
abandon its current trajectory more readily, the bug plausibly made mixing look
_better_, so it never presented as a symptom.

### What would have caught it, in order of cost

1. **A unit test asserting the resampling's defining property.** The cheapest by
   a wide margin. Chopin & Singh state it in one sentence: the chance of any one
   slot picking particle `m` must equal `m`'s weight. As a test that is twenty
   lines, needs no model, and runs in 0.13 s. Against the old code it fails at
   0.0000 versus 0.1000 owed — not a marginal failure, a total one. It now
   exists (`conditional_systematic_matches_rejection_sampling_*`).
2. **An invariance test on an exactly-enumerable fixture.** Minutes to run,
   about a day to conceive, and it would have failed on day one. It now exists
   (`csmc_exact_invariance`).
3. **Recording the preconditions of a borrowed derivation as executable
   assertions.** When a formula is transcribed from a paper, the paper's
   assumptions are part of what is being transcribed. "Valid only under a
   resampling scheme that is marginally unbiased and independent across slots"
   belongs next to the formula as something the program can check, not as prose
   in a comment — or as nothing at all, which is what happened.

### Rules to carry forward

1. **Test the defining property, not the parts.** A kernel whose whole purpose
   is to leave a distribution invariant needs a test that measures invariance.
   Four tests on the parts passed on a kernel that was not invariant.
2. **Never call a primitive and discard part of its answer.** If the operation
   you need is not the operation the function performs, write the function you
   need and name it for the question it answers. The discarded value is the tell
   — a computed result that is never read is either dead work or a silently
   changed distribution.
3. **When you implement from a paper, implement its preconditions too.** Copy
   the formula and the conditions in the same commit, and make the conditions
   checkable. A derivation's assumptions are not commentary; they are part of
   the specification.
4. **A control is only a control if it has been tested on its own.** "Plain
   particle Gibbs is provably correct" was true of the algorithm and false of
   this implementation of it. A control that shares code with the arm under test
   cancels shared defects out of the comparison and relocates them into the
   difference — which is precisely how gh#718 came to indict the wrong surface.
   Where a control carries the weight of a conclusion, give it an absolute
   check, not just its role in a paired contrast.
5. **Shrink the model before growing the sample.** The instrument that settled
   this in minutes is a 6-person SIR over 4 substeps. The one that could not
   settle it in days was a 15-substep model needing 800,000 draws for a ground
   truth. When a property is exactly checkable on a small enough state space,
   make the state space small — the payoff is a comparison with no Monte-Carlo
   error on either side.
6. **When a prediction fails to reproduce, compare the geometry before
   concluding the prediction is wrong.** The SPARSE fixture came back clean and
   we briefly read that as evidence against the review's diagnosis. SPARSE
   contained a repair mechanism we had not thought about. Finding TRAP meant
   asking what structure the counterexample had that our fixture lacked.
7. **`--method pgas` has no off-switch for ancestor sampling.** Every plain
   particle Gibbs arm in this investigation existed only because `pgas.rs` was
   temporarily patched. Given rule 4, the control ought to be reachable without
   editing the source.

## Part 7: the decision on defect 2

Three options, with what each costs.

**Option 1 — use multinomial resampling at every substep, including the ones
with equal weights.** Measured clean (z = 1.28). But resampling on equal weights
duplicates particles for no benefit: only about 63% of the ensemble survives as
distinct trajectories each time. Over the 7–15 substeps camdl typically runs
between observations, the ensemble collapses. **Rejected.**

**Option 2 — where resampling is skipped, also skip ancestor sampling; use
multinomial resampling and ancestor sampling only where an ancestry is actually
drawn.** Measured clean (z = 0.98). Lindsten, Jordan & Schön explicitly permit
performing ancestor sampling only occasionally. **This is the option chosen**,
and see "What it actually cost" below — the prediction that it would hurt mixing
did not survive measurement.

**Option 3 — keep systematic resampling and carry the full resampling
probability through the accept/reject test.** On equal-weight substeps that
probability is zero for any move, so the test would reject every splice — which
is Option 2 arrived at the long way, plus considerable machinery.

**A consequence of Option 2 worth stating plainly:** it makes
`conditional_systematic_resample` unnecessary. Under multinomial resampling the
conditional draw is just `n−1` independent picks, and defect 1 cannot occur by
construction. The defect-1 fix (`55178dd1`) would be _revised_, not extended,
and the Algorithm 4 implementation deleted.

**The one claim we left open is now settled, and the review was right.** Their
§8 argued that conditional _systematic_ resampling is invalid in combination
with ancestor sampling even on substeps that do resample. We recorded it as
unresolved and said our DENSE and SPARSE runs were "moderate evidence against
it". **Both of those statements were wrong.**

_The evidence was not moderate; it was absent._ The goodness-of-fit statistic
scales as `z ≈ M·D / sqrt(2·df)` for a divergence `D`, so at fixed `D` it falls
as `1/sqrt(df)` — and DENSE has 1,956 scored bins against TRAP's 126, a 15×
difference nobody costed. Comparing raw `z` across fixtures was the error. A
defect of exactly TRAP's measured size would have registered on DENSE at **z =
0.60**. DENSE had 0.6σ of power against the question it was being cited to
answer, and SPARSE had less. This is the same shape of mistake as reading
SPARSE's clean result as refuting the review: a property of the fixture's
geometry that was not accounted for.

_And the defect is real, measured exactly._ Enumerating the resampling law
directly — no model, no Monte Carlo — the ancestry `csmc_as` realises under
conditional systematic resampling differs from the target's by a total variation
of 0.30 to 0.80, and **33–75% of its mass sits on ancestries the target gives
density exactly zero.** Systematic resampling pins each particle's offspring
count to `floor(N·W)` or `floor(N·W)+1` and emits them in cyclic order, so given
the other slots only one or two values of the reference's ancestor are
admissible at all; ancestor sampling proposes over all `N`. That is a support
violation, and no reweighting of an accept/reject ratio repairs one. On a
near-uniform weight vector the reference's history is erased from the ensemble
on 81% of steps where the target says 5% — defect 1's pathology reappearing on
substeps that _do_ resample.

The identical comparison under multinomial returns machine zero (≤ 6e−15), which
is what makes this a statement about systematic resampling rather than about our
algebra.

**So multinomial was not a way of sidestepping the question. It was the
answer.** Chopin & Singh say as much, in §5.3: the label dependence induced by
"particularly systematic resampling … makes it more difficult to update one
single component of this vector", and updating one single component is exactly
what ancestor sampling does. They add that adapting ancestor sampling to those
schemes is "straightforward" but give no algorithm and call the mixing gains
"deserv[ing] further investigation".

_If anyone later wants systematic's lower variance back_, the adaptation is to
draw the reference's ancestor **before** the free block rather than overwriting
it after: AS-draw `A^r = j` from the unchanged Eq.-(17) weight, then draw the
free ancestry from `ρ_W(· | A^r = j)`. This costs nothing structurally in camdl
— the ancestor-sampling block reads only pre-resample snapshots and the incoming
weights, never anything propagation produces, so it can be hoisted above the
resample unchanged.

### The chosen fix, and the condition it must be gated on

**Option 2 is chosen, with one clarification that changes the implementation.**

Ancestor sampling must run only where **a valid resampling operation actually
took place** — not merely where the substep happens to carry an observation.
Those two conditions look equivalent and are not. A substep can carry an
observation and still produce equal weights, if every particle happens to score
the observation identically; the resampling is then skipped, and an ancestor
move there would be exactly the invalid move defect 2 describes. Gating on "is
there an observation here?" would leave that case broken while appearing to fix
it.

So the gate is a fact about what the code _did_, not about what the schedule
says: the resampling step sets a flag when it has actually drawn a new ancestry,
and ancestor sampling reads that flag. Anything else re-introduces the defect
through a narrower door.

### Follow-up

`--method pgas` has no way to disable ancestor sampling, so the plain particle
Gibbs control used throughout this investigation only existed by patching
`pgas.rs`. An `ancestor_sampling = false` setting is to be added once the
defect-2 fix lands, so that control is reachable without editing the source.

### What it actually cost

The expectation, stated before measuring, was that ancestor sampling firing at
far fewer substeps would reduce trajectory renewal and with it the mixing rate.
**That is wrong.** Measured before and after at matched observation cadence
(2500 sweeps, 256 particles, 80 substeps, `sir_overdispersion`):

| cadence           | arm       | renewal | ms/sweep | ESS/sweep | ESS/second |
| ----------------- | --------- | ------: | -------: | --------: | ---------: |
| every 7 substeps  | before    |  0.9957 |     23.3 |     1.000 |      42.94 |
|                   | **after** |  0.9760 | **13.7** |     0.936 |  **68.07** |
| every 14 substeps | before    |  0.9965 |     21.3 |     0.872 |      40.87 |
|                   | **after** |  0.9831 | **11.2** |     0.923 |  **82.42** |

Three things to read off it.

**Per-sweep efficiency is unchanged within noise** — ESS/sweep moves −6% at one
cadence and +6% at the other. Despite 7× and 16× fewer ancestor-sampling
opportunities, renewal falls only from ~0.996 to ~0.976, and `renewal_by_bin`
shows no early-time degeneracy. The reason is that renewal is not proportional
to opportunity count: one splice accepted at an observation boundary re-anchors
the whole trajectory downstream of it.

**Sweeps are 1.7–1.9× cheaper**, because `fill_ancestor_log_weights` and
`splice_log_ratio` are skipped wherever the gate closes, and they are not cheap.

**ESS per second improves 1.6–2.0×.** On this fixture the correct kernel is also
the faster one.

Scope, stated rather than implied: one model, 256 particles. The production
province fit runs 9,600 particles over 141 substeps, where the saving should be
larger — `fill_ancestor_log_weights` is O(particles) per substep — but the
magnitude does not transfer. The "before" column is a biased kernel, so this
measures throughput honestly and statistical quality only loosely. ESS is for a
single functional, prevalence at the terminal substep, chosen as the hardest
coordinate to renew.

**Consequence for tuning:** do not raise `csmc_sweeps_per_nuts` to compensate
for the gate. On this evidence there is nothing to compensate for.

### How many opportunities the gate leaves, on real schedules

The count is a property of the observation cadence, not the model, and it is off
by one from the obvious reading: the weights an observation produces are
consumed by the FOLLOWING substep, so a terminal observation drives no
resampling at all. Computed from the ebola project's own data files, at
`dt = 1`:

| fit config                                        | union observation days | substeps | ancestor-sampling opportunities |
| ------------------------------------------------- | ---------------------: | -------: | ------------------------------: |
| `fit_province_direct_sum_8k` (all streams weekly) |                     15 |      141 |                   **14** (9.9%) |
| `fit_national_delay_od_lab_direct_sum_8k` (daily) |                     85 |      146 |                  **84** (57.5%) |

The per-sweep trace now carries `as_opportunity` next to `as_proposed`, so this
is readable from any run rather than recomputed from the schedule.
