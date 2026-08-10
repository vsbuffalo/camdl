# Parallel-tempering swap accepted the wrong exchanges for four months

|            |                                                                                               |
| ---------- | --------------------------------------------------------------------------------------------- |
| Date       | 2026-08-10                                                                                    |
| Severity   | **High (P1)** — critical for any tempered result that informed a conclusion                   |
| Scope      | `pgas` stages with a `tempering` ladder longer than one rung. **Default PGAS is unaffected.** |
| Issue      | gh#550                                                                                        |
| Introduced | `163c80d7`, 2026-04-09                                                                        |
| Status     | **Fixed** — sign corrected and pinned by tests in the same push; §4b open                     |

---

## Summary in one paragraph

camdl's optional parallel-tempering mode runs several copies of a chain at
different "temperatures" and periodically lets neighbouring copies trade states,
so that a good region found by an exploratory copy reaches the copy whose
samples we actually report. The rule deciding whether to accept a trade had its
sign inverted. It therefore rejected the trades the method exists to perform and
accepted their opposites, pushing states from the deliberately-flattened copies
into the reported one. Nothing looked wrong from the outside: the run completes,
likelihoods are finite, and the swap-rate diagnostic sits in its healthy band.
Ordinary single-rung PGAS never executes the code and is not affected.

---

## 1. What tempering is, and why the bug is invisible

Some posteriors have several separated modes. A sampler that finds one may never
cross to another, so it reports a confident answer about the wrong mode.

Parallel tempering runs `K` copies ("rungs") of the chain at once. Rung `k`
targets a **flattened** version of the posterior:

```
π_k(θ)  ∝  L(θ)^{β_k} · p(θ)
```

where `L` is the likelihood, `p` the prior, and `β_k` an "inverse temperature".
Rung 0 has `β = 1` — that is the true posterior, the **cold** chain, and the
only one whose draws are reported. Later rungs have smaller β, which flattens
the surface — peaks shrink, valleys fill in — so those chains wander between
modes easily. Periodically, neighbouring rungs propose **trading their states**.
A good region discovered by a hot, free-roaming chain migrates down the ladder
into the cold chain. That migration is the entire point of the method.

The bug is in the rule that decides whether a proposed trade is accepted.

It is invisible because every surface signal stays healthy. The run finishes.
Log-likelihoods are finite. Trajectories look epidemiologically sensible. Swaps
are accepted at a plausible rate. The individual PGAS moves are correct. What is
wrong is only _which_ trades get accepted — and nothing was watching that.

---

## 2. The math, plainly

### The question a swap asks

Rung `i` currently holds state `x_i`; rung `j` holds `x_j`. Should they trade?

Metropolis says: compare how "happy" the pair of chains is now against how happy
they would be after trading, and accept with probability equal to the ratio
(capped at 1). "Happy" means the joint density of the configuration.

Write `ℓ_i` for the log-likelihood of state `x_i` — **untempered**, no β
applied.

**Now:** rung `i` holds `x_i`, rung `j` holds `x_j`.

```
log(density now)  =  β_i·ℓ_i  +  β_j·ℓ_j  +  log p(x_i) + log p(x_j)
```

**After trading:** rung `i` holds `x_j`, rung `j` holds `x_i`.

```
log(density after) =  β_i·ℓ_j  +  β_j·ℓ_i  +  log p(x_j) + log p(x_i)
```

The two prior terms are the same on both lines — the same two states are present
either way, just parked in different rungs — so they cancel. Subtracting:

```
log α  =  (β_i·ℓ_j + β_j·ℓ_i) − (β_i·ℓ_i + β_j·ℓ_j)
       =  β_i(ℓ_j − ℓ_i) − β_j(ℓ_j − ℓ_i)
       =  (β_i − β_j)(ℓ_j − ℓ_i)                    ← the correct rule
```

### Reading that rule in words

Rung `i` is the colder one, so `β_i > β_j` and the first bracket is
**positive**. The sign of the whole thing is therefore the sign of
`(ℓ_j − ℓ_i)`:

> **Accept when the hotter rung is holding the better state.**

Which is exactly what you would want, stated without any algebra: if the
free-roaming chain found somewhere good, move it down to the chain we report.

### What the code did

`pgas.rs:2699` computed

```
(β_i − β_j) · (ℓ_i − ℓ_j)          ← the two ℓ terms are the other way round
```

which is the negation. In words, it accepted when the **colder** rung already
had the better state — i.e. it took the worse state from the flattened chain and
moved it _into_ the chain we report, and refused the trades that would have
helped.

### A concrete instance

Cold rung `β_i = 1.0`, hot rung `β_j = 0.5`. The hot rung has found something
much better: `ℓ_j = −100` against the cold rung's `ℓ_i = −200`. This is
precisely the situation tempering is built for.

|              | log α                         | probability       | outcome                                         |
| ------------ | ----------------------------- | ----------------- | ----------------------------------------------- |
| correct rule | `0.5 × (−100 − (−200)) = +50` | 1                 | accept — the good state moves to the cold chain |
| shipped code | `0.5 × (−200 − (−100)) = −50` | `e^{−50}` ≈ 2e−22 | reject                                          |

The joint density genuinely rises by a factor of `e^50` under this trade, so
accepting is mandatory. The code declined it with probability
0.999999999999999999999.

### Where the sign was lost

The physics literature writes this in **energies**, with Boltzmann weight
`e^{−βE}`. Our weight is `L^β = e^{βℓ}`, so energy and log-likelihood are
related by `E = −ℓ`. The standard criterion (Hukushima & Nemoto 1996, eq. 6) is

```
Δ = (β_i − β_j)(E_i − E_j)
```

Substitute `E = −ℓ` **correctly** and you get `(β_i − β_j)(ℓ_j − ℓ_i)` — the
right rule. Substitute `ℓ` for `E` and forget the minus sign, and you get
exactly what shipped. It is a one-character slip with no visible symptom.

---

## 3. How it was detected

Not by a test, and not by anyone looking at tempering.

1. An unrelated blocker (gh#471) was fixed in the same function — a different
   acceptance test that mis-rejected `+∞`.
2. While fixing it, a sibling sweep found the swap site and concluded it was
   _decision-equivalent_ to the fixed one, so a 16-line comment was added
   blessing it and explaining why it was deliberately left alone.
3. That comment is what dragged the site into scope for the adversarial review
   commissioned on the gh#471 diff. The reviewer's words: file it separately,
   "or the comment will read as an audit that cleared it."
4. The reviewer derived the sign error. It was then derived independently a
   second time in-house, and confirmed a third time by external review at ~99%
   confidence.

**Worth recording honestly:** a confident comment written about code that had
not been verified is what caused the code to be verified. Had the site been left
untouched and unremarked, nobody would have looked at it.

---

## 4. Root cause, and a second finding

### 4a. The sign (confirmed)

Above. The code, the inline comment, **and the original commit message** all
state the same inverted formula:

> `163c80d7` (2026-04-09): "acceptance alpha = min(1, exp((beta_i - beta_j) *
> (LL_i - LL_j)))"

So this was not a regression or a later edit — the sign came from the initial
derivation, and every artefact in the tree agrees with every other. Code review
saw internal consistency and had nothing to catch.

### 4b. The rungs may not target what the swap assumes (FLAGGED, not confirmed)

The swap ratio is derived from each rung targeting `π_k ∝ L^{β_k}·p`. Checking
whether they actually do:

- **The parameter move is tempered.** `pgas.rs:2615`:
  `log_alpha = beta * (proposed_ll − rungs[rung].ll) + (prior diff) + (jacobian diff)`
  — β scales the likelihood difference, the prior is untempered. Consistent with
  `L^β · p`.
- **The trajectory move is not.** `csmc_as` takes no β at all, and the comment
  at `pgas.rs:2652` says so outright: _"CSMC always runs at β=1 — the trajectory
  must match the data."_

So for `β < 1` the within-rung kernel alternates a θ-move targeting `L^β·p` with
an x-move targeting the `β = 1` conditional `p(x | θ, y)`. Those two do not
share a stationary distribution, so rung `k`'s actual invariant law is **not**
`L(θ,x)^{β_k} p(θ)` — which is the distribution the swap ratio (even corrected)
is derived from.

This was independently anticipated by the external reviewer, who asked to
inspect "exactly where beta enters conditional SMC/ancestor sampling and
parameter MH/NUTS, to verify that all rungs really target `p(x)e^{βℓ(x)}`" and
warned that a mismatch would be "another correctness issue independent of this
sign."

**This is a derivation, not a confirmed defect**, and it is a design question as
much as a bug: tempering the complete-data likelihood and tempering only the
observation likelihood are different intents, and which was meant is not
recorded anywhere. It needs the same external scrutiny the sign got.

**Sequencing — a position we reversed.** An earlier draft of this report argued
the sign fix should wait until §4b was settled, on the grounds that a corrected
ratio derived for a target the rungs do not have is still not right. That was
wrong, and the fix shipped first. Whatever §4b turns out to be, the corrected
direction moves _higher_-likelihood states toward the cold rung, where the
inverted one moved worse ones there — strictly an improvement. Holding a known
inverted kernel in place behind an open design question would have been the
worse call. §4b is tracked separately.

---

## 5. Impact

| scope                                                      | assessment                                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------------- |
| PGAS with no `tempering` key, or `[1.0]` (the default)     | **unaffected** — `while i + 1 < n_rungs` never executes at one rung |
| PGAS latent-trajectory machinery (CSMC, ancestor sampling) | **no evidence of a defect**                                         |
| The swap kernel itself                                     | **incorrect**, conditional on the stated target                     |
| Tempered cold-chain posteriors                             | **not theoretically valid**                                         |
| Previously generated tempered analyses                     | **should be rerun after the fix**                                   |

The correct framing for anyone outside the project is **"the optional
replica-exchange transition in tempered PGAS violates its intended acceptance
ratio"** — not "PGAS is broken." The total kernel is
`K_total = K_PGAS/MWG ∘ K_swap`; the first factor can be entirely sound, and
composing it with a non-invariant second factor is what destroys invariance
overall. That "PGAS appears to work" is fully compatible with this bug.

**On the direction of the resulting bias, we are deliberately not claiming
much.** An earlier draft asserted the cold intervals would come out too wide.
That overstates what is known: correct within-rung kernels are fighting an
incorrect swap kernel, and the stationary law of the composition is not a simple
mixture of the cold and hot targets. The defensible statement, per the external
review:

> The cold-chain marginal is no longer guaranteed to equal the intended
> posterior. The erroneous swap systematically favours assignments disfavoured
> by the intended product target, plausibly flattening the cold marginal — but
> the magnitude and direction of downstream parameter bias must be determined
> empirically.

---

## 6. Remediation

### The fix is one line

```rust
// pgas.rs:2699
let log_alpha = (betas[i] - betas[j]) * (rungs[j].ll - rungs[i].ll);
```

plus correcting the comment above it, which states the same wrong formula.

It is one line, but it is **not** a one-line change to land: §4b must be settled
first, and every tempered result in the store becomes suspect on landing, which
is a release-note item.

### It should be extracted, not fixed in place

```rust
fn swap_log_alpha(beta_i: f64, beta_j: f64, ll_i: f64, ll_j: f64) -> f64 {
    (beta_i - beta_j) * (ll_j - ll_i)
}
```

The expression is currently unreachable from a test without running a full fit,
which is the direct reason no test pins it. Extracting it is what makes §7
possible.

---

## 7. Testing: what would have caught this, and what we are adding

The existing tempering test asserts that swap rates lie in `[0, 1]` and that
log-likelihoods are finite. That is **mechanical operation**, not
**stationarity** — and it is the general lesson here:

> A sampler can look excellent while sampling the wrong distribution. Tests that
> assert the machinery runs cannot detect a target error; only tests that assert
> agreement with a known distribution can.

Four checks, cheapest first.

**1. The ratio against an independently computed product target.** For arbitrary
`β_i > β_j` and arbitrary `ℓ_i, ℓ_j`, compute `log Π_proposed − log Π_current`
directly from the definition and require `swap_log_alpha` to equal it. This
tests the _invariant_, not one example — a single worked case can be satisfied
by a coincidentally-correct wrong formula.

**2. The direction that matters, as a named case.**
`assert!(swap_log_alpha(1.0, 0.5, -200.0, -100.0) > 0.0)` — a hot rung holding a
much better state must be accepted. Red on the shipped expression, green on the
corrected one.

**3. Exact-posterior agreement on a tiny model.** Build a discrete model whose
`π_β` is enumerable in closed form. Run one-rung PGAS and 2–4-rung tempered
PGAS. **The cold marginals must agree with the exact `β = 1` distribution
regardless of the ladder.** This is the test that catches §4b as well as the
sign, because it tests the composed kernel rather than one factor.

**4. Tempering must not move the estimand.** On a small tractable model,
long-run cold-chain summaries under `[1.0]` and under `[1.0, 0.7, 0.4]` must
agree within Monte Carlo error. Tempering is allowed to change autocorrelation
and mixing — never the answer.

### Two runtime diagnostics worth adding

- **Log the sign of the correct Δ, not just the total swap rate.** A proposal
  with `Δ > 0` must always be accepted; a counter of "Δ > 0 but rejected" would
  have been non-zero from day one.
- **Monotone mean likelihood across the ladder.** Under the intended family,
  `d/dβ E_β[ℓ] = Var_β(ℓ) ≥ 0`, so at equilibrium **colder rungs should have
  higher mean untempered log-likelihood than hotter ones**. That ordering is
  cheap to compute per-sweep and is violated by an inverted swap.

### A correction to our own reasoning

Our first write-up claimed the acceptance-rate diagnostic _cannot_ distinguish
the two signs. The external review corrected this and the correction matters:
correct code accepts with `min(1, e^Δ)` and buggy code with `min(1, e^{−Δ})`,
which can differ dramatically. They only look alike when the distribution of `Δ`
is roughly symmetric and concentrated near zero — which is what happens in our
small test fixture, and is why _our_ rates looked healthy. The diagnostic is not
intrinsically sign-blind; **our test model was too easy for it to matter.** That
is a more useful conclusion, because it says the fixture needs a real likelihood
gap between rungs, not that the diagnostic needs replacing.

---

## 8. Process changes this suggests

1. **A shared numerical primitive that is derived, not transcribed, gets its
   derivation in a comment and a test against that derivation.** Acceptance
   ratios, Jacobians, and gradient formulas all qualify. The comment here
   restated the formula rather than deriving it, so it could not disagree with
   the code.

2. **An optional feature that the default path never exercises needs its own
   gate.** Tempering is off by default, so the entire PGAS test suite ran none
   of this code. Any feature in that category should be named in the testing
   docs with the test that covers it.

3. **Sibling sweeps must not bless what they do not verify.** The comment added
   during the gh#471 sweep asserted the swap site was correct on the strength of
   a _decision-equivalence_ check about `±∞` handling — a question that had
   nothing to do with whether the formula was right. Verifying one property and
   writing a sentence that reads as clearing the site is worse than saying
   nothing.

4. **The `E ↔ −log L` conversion deserves a standing note.** It is the single
   most likely place for this class of error, and it will recur wherever the
   physics literature is the source.

---

## References

- **Hukushima, K., & Nemoto, K. (1996).** Exchange Monte Carlo method and
  application to spin glass simulations. _J. Phys. Soc. Japan_, 65(6),
  1604–1608. DOI: 10.1143/JPSJ.65.1604 — the canonical criterion, in energy
  form. Load-bearing because the `E = −ℓ` substitution is where the sign was
  dropped.
- **Geyer, C. J. (1991).** Markov chain Monte Carlo maximum likelihood.
  _Computing Science and Statistics: Proc. 23rd Symposium on the Interface_,
  156–163. — origin of Metropolis-coupled MCMC; states the swap as a Metropolis
  step on the product chain, which is the derivation in §2.
- **Earl, D. J., & Deem, M. W. (2005).** Parallel tempering: theory,
  applications, and new perspectives. _PCCP_, 7, 3910–3916. DOI:
  10.1039/B509983H — review; discusses acceptance-rate tuning, the diagnostic §7
  revisits.
- **Lindsten, F., Jordan, M. I., & Schön, T. B. (2014).** Particle Gibbs with
  ancestor sampling. _JMLR_, 15, 2145–2184. — the inference method the tempering
  sits on top of. Notably does **not** specify a tempering scheme, which is why
  the ratio came from the physics literature in the first place.
