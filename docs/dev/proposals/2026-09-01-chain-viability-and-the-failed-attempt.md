# A refused chain is a failed attempt, not an infeasible draw

Date: 2026-09-01 Status: proposed\
Related: gh#780 (open), gh#783 (partly landed), gh#784 (landed, `2e00e135`),
gh#751 (open), gh#607, gh#334, PR#782 (closed, not merged)\
Note ref: `docs/dev/notes/2026-08-30-pgas-bad-init-criterion.md`

Re-keying authorisation: **required, not yet given** — see §11.

## 1. What camdl claims today, and why the claim is wrong

A user asks for 16 chains. camdl draws 16 starting parameter vectors `θ₀`, seven
of them never produce a sample, and the fit proceeds with nine. The summary
reports R̂ over those nine.

The seven are reported as `BadInit`. That word makes a statistical claim — _this
starting point is infeasible_ — that camdl is not in a position to make, and in
the case that prompted this proposal the claim was not merely unsupported but
inverted.

On a national Ebola fit (16 chains, `--init from_prior`), the refused and
sampled starts separate perfectly on a single coordinate, the initial infectious
count `I0`:

```
refused   I0 = 15.1, 26.9, 35.1, 35.8, 39.4, 49.9, 51.4     max   51.4
sampled   I0 = 112.9 … 615.0                                min  112.9
```

No other coordinate separates them — a refused draw had `r_eff` 1.86 and a
sampled one 1.14. The posterior that fit then produced, over 13,500 draws:

```
I0    5% 56.1    25% 69.8    median 82.1    75% 96.1    95% 121.7
```

**91.5% of the posterior mass sits below `112.9`, the smallest start that was
kept.** Every chain that ran began above the 91.5th percentile of the answer.

The modelling team read "44% of my prior draws are infeasible" as _my prior is
too vague_ and came close to tightening it — which would have cut the lower tail
of `I0`, the region carrying almost all the posterior. That near-miss is the
motivating failure. It is a naming failure before it is an algorithmic one.

### The three mechanisms behind one word

`DiagnosticKind::BadInit` is pushed from three sites with three unrelated
meanings:

| site                     | fires when                                                                                                          | what it actually establishes                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `cli/fit/pgas.rs:1026`   | `SimError::NonFiniteChainStart` — complete-data density zero at `(θ₀, X₀)` and still zero after **one** Gibbs sweep | one sampled trajectory failed to explain the data                   |
| `cli/fit/pmmh.rs:640`    | `SimError::PFDegenerate`                                                                                            | the likelihood **estimator** was unhealthy                          |
| `cli/fit/runner.rs:2162` | `SimError::PFDegenerate`, raised at `sim/inference/if2.rs:667` **inside an iteration's filter loop**                | a stochastic optimisation run degenerated, possibly at iteration 40 |

The engine layer already distinguishes these — `SimError` carries
`NonFiniteChainStart` and `PFDegenerate` as separate variants with separate
payloads, and classifies recoverability at `sim/error.rs:425-480`. It is the
**reporting** layer that flattens them, and it flattens them onto a word that
asserts something none of the three has measured.

## 2. Measurement: a failed sweep is not a repeat

The note this proposal supersedes argues for extending PGAS's probation from one
sweep to `N`, and models successive probation sweeps as i.i.d. Bernoulli trials
so that `N` can be chosen from `1 − (1−p)^N`. That model requires

```
X_{s+1} = X_s   whenever probation sweep s fails
```

— only then is the next sweep an identical retry with fresh randomness.

It is not what the code does, and it is not what production runs show.

**Mechanism.** `csmc_as` reconstructs its outgoing trajectory by walking the
ancestor array backward from the selected index
(`sim/inference/pgas.rs:2740-2761`):

```rust
let mut particle = k;
for s in (0..n_substeps).rev() {
    let renewed = particle != j_ref;
    if !renewed { n_from_ref += 1; }
    trajectory_substeps.push(SubstepRecord { /* … history at [s][particle] … */ });
    particle = ancestors[s][particle];
}
let trajectory_renewal = 1.0 - n_from_ref as f64 / n_substeps as f64;
```

Ancestor sampling has been writing into `ancestors` throughout the sweep. So
even on the gh#783 collapse path, where the final draw falls back to the
reference **index** (`k = j_ref`, `pgas.rs:2707-2711`), the reconstructed
**path** follows re-anchored ancestry and need not be the reference.

`trajectory_renewal` is therefore an exact indicator of the property in
question: it is `0.0` if and only if the traceback sat on `j_ref` at every
substep, i.e. the outgoing trajectory _is_ the incoming reference. It is already
computed every sweep and already written to `trace.tsv` (`cli/fit/pgas.rs:795`,
`:875`), alongside per-decile bins `renewal_b0…b9`.

**Result.** Over every committed PGAS trace in the Ebola project — 2,606 chains,
9,259,970 sweep rows:

| quantity                                                    | value               |
| ----------------------------------------------------------- | ------------------- |
| sweeps with `log_complete_data_ll = -inf`                   | 988,440 (10.7%)     |
| …of which `trajectory_renewal == 0.0` exactly               | 29,726 (3.0%)       |
| …of which `trajectory_renewal != 0.0`                       | **958,714 (97.0%)** |
| runs of ≥2 consecutive `-inf` sweeps                        | 35,703              |
| …in which **every** sweep had renewal `0.0` (a true repeat) | **0**               |
| longest consecutive `-inf` run                              | 40,000 sweeps       |

Renewal on non-recovering sweeps: p5 0.057, p25 0.125, **median 0.239**, p75
0.852, p95 0.955, mean 0.442.

The 29,726 zero-renewal rows are not a regime: 29,571 of them (99.5%) come from
a single fit, `fit_national_delay_od_lab_holed`.

Reproduction is a scan of committed artifacts; no run is required.

**Consequence.** Across 35,703 multi-sweep failure runs, not one was a sequence
of identical retries. The success probability of probation sweep `s` is
`p(θ₀, X_s)` with `X_s` moving — typically by a quarter of its substeps at the
median and nearly all of them in the upper quartile. There is no constant `p` to
plug into a geometric model.

A second, independent defect stands even if the mechanism above were repaired.
The note estimates `p ≈ 1/17` from a single observed first-recovery at sweep 17.
One success in seventeen trials has an exact (Clopper–Pearson) 95% interval of
**[0.0015, 0.2869]** — two orders of magnitude. The note's `N = 100 → 99.8%` is
the plug-in point estimate; at the interval's lower end, 100 consecutive
failures still occur with probability **0.862**.

**So `N` is retained as an operational recovery budget and abandoned as a
calibrated test.** `N = 100` remains a reasonable default — recoveries were
observed at sweeps 15 and 17, and a probation sweep costs 0.9–4.0 s (median 1.2)
against a 2,000-sweep run — but the justification is compute, not coverage.

## 3. Why the three algorithms should keep three criteria

The differences are structural, and a unified predicate would have to erase one
of them.

**PGAS** (Lindsten, Jordan & Schön 2014, _JMLR_ 15:2145-2184) alternates
`θ | X, y` and `X | θ, y`, so its Gibbs state is the **pair** `(θ, X)`. The
quantity it evaluates is the complete-data density `p(y, X | θ) p(θ)`, not the
marginal likelihood `p(y | θ) = ∫ p(y, X | θ) dX`. A pair can have zero
complete-data density while `θ` is entirely reasonable, because the one sampled
trajectory predicts zero incidence in a window with positive observed counts.
That is precisely the Ebola case, and it is why PGAS alone has a _repair_ — move
2 draws a new `X` at the same `θ`.

**PMMH** (Andrieu, Doucet & Holenstein 2010, _JRSS-B_ 72:269-342) is
Metropolis–Hastings on an extended state `(θ, U)`, where `U` is the particle
filter's auxiliary randomness and the posterior is the marginal. It retains no
trajectory between iterations; each proposal draws a fresh likelihood estimate.
Its exactness rests on that estimate being **positive and unbiased**, so an ESS
watchdog firing reports an unusable estimator — _not_ `p(y | θ) = 0`. camdl's
own initialization code already states this bar: an empty finite-particle swarm
"is never a claim about `p(y | θ₀)`" (`pgas_init.rs`,
`UnconditionalPass::NoSupport`).

**IF2** (Ionides et al. 2015, _PNAS_ 112:719-724) is stochastic optimisation
toward the MLE, not posterior sampling. It has no support to fall outside of,
and its failure is not even an initialisation event: `check_pf_degeneracy` fires
inside an iteration's per-observation filter loop (`if2.rs:667`), so a run that
completed forty iterations before degenerating is today reported as a bad
_init_.

There is consequently no predicate that means the same thing in all three.
"Finite complete-data density" is undefined for PMMH and IF2; "the filter did
not degenerate" is a numerical-health check, not a statement about `θ`.

**The seam:** share the _vocabulary_ and the _accounting_; keep the _criteria_
and their _repairs_ distinct, each with its justification beside it.

## 4. The principle

> A sampler or optimiser failing to operate from a starting point is not
> evidence that the starting point is statistically infeasible. Hold `θ₀` fixed,
> repair whatever auxiliary state that algorithm actually owns, and on
> exhaustion record a **failed attempt from `θ₀`** — never an infeasible draw.

## 5. gh#334 and gh#607 are not in conflict once separated

`inference/mod.rs:99` defines `mh_accept` as bare `u_ln < log_alpha`, with no
finiteness guard, and its test comment (gh#334) states the rule: a chain at `−∞`
**must** be able to escape to a finite proposal, since `finite − (−∞) = +∞` and
the true Metropolis acceptance probability is 1. The module comment places it
deliberately as a _cross-algorithm_ predicate.

gh#607 then added `NonFiniteChainStart`, refusing a PGAS chain still at `−∞`
after one sweep. Its motivation was real: a run had seeded every chain at `−∞`
and sampled 40,000 sweeps, yielding one distinct parameter vector across 7,600
retained draws. §2's measurement contains that exact pathology — the longest
observed run of consecutive `-inf` sweeps is 40,000.

The two answer different questions and both survive:

- **gh#334 is a correctness rule.** Being at `−∞` must never _prohibit_ a valid
  move. Unchanged.
- **gh#607 becomes a bounded-compute guard.** "We spent the recovery budget and
  stopped" — an operational fact, carrying no claim about the target's support.

## 6. Types

### 6.1 The reporting enum

```rust
// sim/src/inference/diagnostic.rs — CURRENT
pub enum DiagnosticKind {
    // …
    BadInit {
        chain_id: usize,
        /// Estimated parameter name → starting value, natural scale.
        params: std::collections::BTreeMap<String, f64>,
        /// One-line cause from the upstream PFDegenerateKind / fallback message.
        reason: String,          // ← the only thing separating three causes
    },
    // …
}
```

`params` is already carried, so the start vector is already preserved — more
than the external review credited. What is missing is the _type_ of the cause
and the run-level denominator.

```rust
// sim/src/inference/diagnostic.rs — PROPOSED
pub enum DiagnosticKind {
    // …
    /// A chain that was requested and produced no draws. Deliberately NOT named
    /// for initialisation: `cause` says what failed, and for two of the three
    /// variants the failure is not at the start at all.
    ChainNotSampled {
        chain_id: usize,
        params: std::collections::BTreeMap<String, f64>,   // unchanged
        cause: NotSampledCause,
    },
    // …
}

/// Why a requested chain produced no draws. One variant per algorithm, because
/// the three establish genuinely different facts. `chain_id` and `params` live
/// on the parent — they are common to all three; only the evidence differs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NotSampledCause {
    /// PGAS. `θ₀` was held fixed throughout. The sampler could not find a
    /// latent trajectory `X` giving the pair finite complete-data density
    /// within the recovery budget. Establishes NOTHING about `p(y | θ₀)`,
    /// which was never evaluated.
    PgasRecoveryExhausted {
        sweeps_attempted: usize,
        budget: usize,
        /// gh#784: how `X₀` was obtained. `ForwardDraw(_)` means the
        /// unconditional SMC pass could not supply a valid path — an
        /// initialisation failure. `UnconditionalFilter` means initialisation
        /// SUCCEEDED and the chain was refused anyway, a different finding.
        init: InitSource,
        /// The four complete-data terms at the last attempt. Same fields the
        /// current `NonFiniteChainStart` already carries.
        last: CompleteDataTerms,
        /// Per probation sweep, in order. `0.0` means that sweep returned the
        /// reference unchanged. §2 measured this as near-universally non-zero,
        /// which is why the budget is operational rather than calibrated —
        /// recording it keeps that measurable per run instead of assumed.
        trajectory_renewal: Vec<f64>,
    },

    /// PMMH. The particle filter never produced a usable estimate of
    /// `p(y | θ₀)`. Pseudo-marginal exactness needs a positive unbiased
    /// estimate; a watchdog bail means the ESTIMATOR failed. It is not
    /// evidence of zero marginal support.
    PmmhFilterUnavailable {
        kind: PFDegenerateKind,
        obs_window: usize,
        attempts: usize,
    },

    /// IF2. The stochastic search degenerated. `iteration` may be 0 or 40:
    /// this is a failed optimisation RUN, and IF2 has no posterior support to
    /// violate. Keep it in the multi-start denominator — coverage of the
    /// parameter space IS the method.
    If2SearchDegenerated {
        iteration: usize,
        kind: PFDegenerateKind,
        obs_window: usize,
    },
}

/// The complete-data decomposition, lifted out of `SimError` so the error and
/// the diagnostic carry one type rather than five parallel `f64`s.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CompleteDataTerms {
    pub log_posterior: f64,
    /// Non-finite here means a `step_one` / `log_transition_density_substep`
    /// disagreement — a bug, not a bad start (gh#80).
    pub transition: f64,
    /// `log p(y | X₀)`. The common cause: the trajectory predicts zero where
    /// the data is positive.
    pub observation: f64,
    pub ivp: f64,
    /// `Σ log p(θ₀)`. Non-finite here means the start is outside its own
    /// prior's support — the ONE case where "infeasible draw" is the correct
    /// description, and it is checkable without running anything.
    pub log_prior: f64,
}
```

That last field deserves emphasis: a non-finite `log_prior` _is_ a genuinely
infeasible draw, and it is separable from every other cause by inspection. The
proposal's whole argument is that the other four terms do not license the same
word.

### 6.2 The engine error

`SimError` needs no restructuring — one rename, because the current name and
message assert the claim being retracted.

```rust
// sim/src/error.rs — CURRENT
#[error("chain start has zero posterior density and did not recover on its \
         first trajectory update: …")]
NonFiniteChainStart { log_posterior, transition, observation, ivp, log_prior, init }

// PROPOSED
#[error("PGAS did not recover a finite joint state from this start within \
         {sweeps_attempted} sweeps (budget {budget}). This is a recovery \
         budget exhausted, NOT evidence that θ₀ is infeasible: the \
         complete-data density is a property of the PAIR (θ, X), and \
         p(y | θ₀) was never evaluated. Last terms: {last}; {init}")]
PgasStartRecoveryExhausted {
    sweeps_attempted: usize,   // new
    budget: usize,             // new
    last: CompleteDataTerms,   // replaces five loose f64 fields
    init: InitSource,          // unchanged
}
```

`sim/error.rs:425-480`'s recoverability classification keeps the same answer
(`false` — per-chain, not per-particle, not fatal).

### 6.3 The run-level denominator

The genuinely new type, and the one that fixes what users actually hit.

```rust
/// What the fit was asked for versus what it got. Emitted ALWAYS, not only on
/// failure, so `requested == sampled` is a positive assertion rather than the
/// absence of a complaint. Consumers must be able to tell "16 requested, 16
/// sampled" from "16 requested, unknown".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainAccounting {
    pub requested: usize,
    pub sampled: usize,
    /// One entry per requested-but-unsampled chain, each carrying its `θ₀`.
    pub not_sampled: Vec<DiagnosticKind>,
    /// Present when refusal is associated with a coordinate of `θ₀`: the
    /// parameter name and the separation between refused and sampled starts.
    /// This is the number that would have told the Ebola team the refusal was
    /// about algorithmic accessibility rather than about their prior.
    pub refusal_separation: Option<RefusalSeparation>,
}

/// Refused and sampled starts, compared coordinate by coordinate. Computed from
/// data the diagnostics already carry (`params` on each `ChainNotSampled`, plus
/// `chain_starts.tsv`), so this costs no new instrumentation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefusalSeparation {
    pub param: String,
    pub refused_max: f64,
    pub sampled_min: f64,
    /// Fraction of the posterior below `sampled_min`, when a posterior exists.
    /// 0.915 in the motivating case.
    pub posterior_below_sampled_min: Option<f64>,
}
```

## 7. The JSON sidecar

### 7.1 What exists

`diagnostics.json` is written per stage at `stage_dir/diagnostics.json`, and
`DiagnosticKind` already derives `Serialize`/`Deserialize`, so `BadInit` already
reaches consumers as structured JSON with `chain_id`, `params` and `reason`.
Three properties of the current arrangement are load-bearing for this proposal:

1. **It is terminal-only.** All three write sites are ends of the road:
   `cli/fit/pgas.rs:1171` and `:1431` are the all-chains-refused error paths,
   and `:1573` runs after the stage completes. During a ten-hour fit there is no
   supported way to learn that seven chains were refused in the first thirty
   seconds. That is gh#751 and the downstream request that "would have saved us
   the most time this month."
2. **The write result is discarded.** All three sites are
   `let _ = collector.write_json(&diag_path.to_string_lossy());`. A failed
   sidecar write is silent.
3. **There is no schema tag.** `diagnostics.json` carries no version field, so a
   consumer cannot tell which shape it is parsing. This is the same defect
   gh#728 records for `*_summary.json`, and this proposal changes the shape.

### 7.2 What changes

`ChainAccounting` is emitted into `diagnostics.json` as a **top-level sibling**
of the diagnostic list, not as another entry in it — the denominator is not a
diagnostic, it is the frame every diagnostic is read against:

```json
{
  "schema": "camdl.diagnostics/v2",
  "chain_accounting": {
    "requested": 16,
    "sampled": 9,
    "not_sampled": [
      {
        "kind": "chain_not_sampled",
        "chain_id": 3,
        "params": { "I0": 35.1, "r_eff": 1.86, "…": 0.0 },
        "cause": {
          "kind": "pgas_recovery_exhausted",
          "sweeps_attempted": 100,
          "budget": 100,
          "init": "unconditional_filter",
          "last": {
            "log_posterior": null,
            "transition": -2245.09,
            "observation": null,
            "ivp": -26.28,
            "log_prior": -33.83
          },
          "trajectory_renewal": [0.91, 0.88, 0.94, "…"]
        }
      }
    ],
    "refusal_separation": {
      "param": "I0",
      "refused_max": 51.4,
      "sampled_min": 112.9,
      "posterior_below_sampled_min": 0.915
    }
  },
  "diagnostics": [/* … existing DiagnosticKind list … */]
}
```

Non-finite floats serialise as JSON `null` — camdl already runs a finiteness
gate on raw floats rather than on built `Value`s (`140ef57f`), because
`json!`/`to_value` collapse NaN and infinity to `Null` before any gate can see
them. `CompleteDataTerms` must therefore be serialised through that same path,
with the sentinel documented, or a consumer cannot distinguish `-inf` from
absent.

Three further changes, each small and each closing a live complaint:

- **Add `"schema": "camdl.diagnostics/v2"`.** This proposal changes the shape; a
  consumer that cannot detect the change will mis-parse silently. Same argument
  as gh#728.
- **Flush incrementally.** Write `diagnostics.partial.json` as refusals occur,
  replaced atomically by `diagnostics.json` at stage end. A ten-hour fit that
  refused seven chains in the first thirty seconds becomes abandonable in the
  first minute. This is the single most requested change from downstream.
- **Stop discarding the write result.** `let _ =` on all three sites becomes a
  logged warning at minimum. A diagnostics file that silently failed to write is
  indistinguishable from a fit with no diagnostics.

### 7.3 Consumers

`camdl-viewer` reads these artifacts and currently conveys refused chains
poorly, which is a direct consequence of §7.1: the only machine-readable signal
is a `BadInit` entry whose `reason` is prose, with no denominator anywhere. With
`chain_accounting` present a viewer can render "9 of 16 chains sampled" as a
first-class fact, list the seven starts, and — where `refusal_separation` is
populated — say which coordinate the refusals track.

The `output_schema` map already in `run.json` (`cli/src/resolve.rs:270`) is the
precedent for telling consumers a shape; `diagnostics.json` should carry its own
tag by the same logic.

**Action required before implementation:** enumerate the downstream consumers of
`diagnostics.json`. `camdl-viewer` is one; the Ebola project's workflow is
another. Per `VERSIONING.md` this is a pre-1.0 breaking change and gets no
compatibility shim, but the consumers should be told rather than discover it.

## 8. What the user sees

Today: seven chains vanish, R̂ is computed over nine, nothing records that the
design changed.

Proposed, on stderr and in the summary:

```
chains: 9 of 16 sampled, 7 not sampled (PGAS recovery budget exhausted)
refusal tracks I0: refused ≤ 51.4, sampled ≥ 112.9
  → 91.5% of the posterior lies below the smallest start that was kept.
    This is algorithmic accessibility, not a statement about your prior.
R̂ computed from 9 of 16 requested chains — treat as conditional on that.
```

The third line is the one that matters. It is computable from data the
diagnostics already carry, and it is the sentence that would have stopped a
modelling group from tightening a prior away from its own posterior.

Also fixed as a side effect: a refused chain is currently indistinguishable from
one still queued behind `--parallel`, because both show as a short or absent
`trace.tsv`. `chain_accounting` distinguishes them by construction.

## 9. Non-goals

**Not unifying the predicate.** Three algorithms, three criteria, each with its
justification beside it. A single function taking a method discriminant would
erase that one algorithm carries latent state in its chain, one does not, and
one is not a chain.

**Not auto-redrawing `θ₀`.** Redrawing until `K` chains are viable is an
unreported rejection sampler from `q₀(θ)·a(θ)`. It does not bias the posterior —
each surviving chain is invariant to its start, and the motivating fit confirmed
this empirically, with all nine survivors reaching the bulk within 100–250
sweeps inside a 500-sweep burn-in. What it destroys is R̂'s diagnostic power,
which depends on starts being **overdispersed** relative to the posterior
(Gelman & Rubin 1992, _Statist. Sci._ 7:457-472). Starts selected for filter
friendliness are tight and displaced in one direction, so chains agree before
they have explored. Rank-normalised split-R̂ (Vehtari et al. 2021, _Bayesian
Anal._ 16:667-718) repairs heavy tails and unequal scales; it cannot
reconstitute chains removed before sampling.

For IF2 this is stronger than a diagnostic concern. There is no invariance to
fall back on: multi-start coverage _is_ the method, so conditioning the
surviving runs on "the filter never degenerated" changes which basins are
reachable.

**Not weighting posterior draws by `1/a(θ)`.** The selection acts on starts, not
on samples. An importance correction here would introduce a bias rather than
remove one.

**Not resurrecting PR#782.** Deciding PGAS's refusal on a marginal-likelihood
probe was rejected because one PF realisation is seed-sensitive and an ESS
failure is not a test of support. PGAS has a more direct quantity available:
whether its own joint state recovered.

## 10. Implementation order

Each step is independently landable and green.

1. **`CompleteDataTerms`**, replacing the five loose `f64`s on
   `NonFiniteChainStart`. Pure refactor, no behaviour change, no re-key.
2. **`ChainNotSampled` + `NotSampledCause`**, with the three call sites
   converted. `BadInit` is deleted outright — alpha posture, no alias. Changes
   `diagnostics.json`; add the `schema` tag in the same commit.
3. **`ChainAccounting`** including `refusal_separation`, emitted by all three
   drivers. This is the user-visible win and depends on step 2.
4. **Incremental flush** (`diagnostics.partial.json`) plus handling the write
   result. Independent of 1-3; could land first if downstream needs it sooner.
5. **The probation budget.** `PgasStartRecoveryExhausted` replaces
   `NonFiniteChainStart`, with `probation_sweeps` (default 100) as a typed field
   on `Stage::PGAS`. **This is the step that re-keys** — see §11.
6. **IF2 relabelling.** `If2SearchDegenerated` carrying `iteration`. Requires
   threading the iteration index to the error, which `if2.rs:667` does not
   currently have in scope.

Steps 1-4 and 6 are behaviour-preserving for the sampler and change only what is
reported. Step 5 changes what runs.

## 11. Re-key inventory

Step 5 adds `probation_sweeps` to `Stage::PGAS`. `Stage::identity_payload` is
subtractive — it serialises the whole variant and removes two named keys — so a
new field is hashed by default. Note this is a **serde** property, not the
compile-enforced `RunInput` guarantee: `Stage` derives
`Debug, Clone, Deserialize, Serialize` and **not** `RunInput`
(`cli/fit/config_v2.rs:1043`), so a `#[serde(skip)]` would silently drop the
field from the key. The include-by-default behaviour is correct here; the
enforcement is weaker than the surrounding documentation implies.

An always-serialised field emits `"probation_sweeps": 100` on **every** PGAS
stage, which re-keys every existing PGAS leaf and invalidates every
`resume_state.bin`. The alternative — `skip_serializing_if` on the default —
avoids that but means a stage's stored payload no longer records the budget the
run used.

**Recommendation: take the full re-key.** A stored posterior whose address does
not name the budget that produced it is the failure the identity system exists
to prevent, and pre-1.0 invalidation is cheap.

**This is the same decision gh#802's sibling work faces** for the binomial
sampler field. Both should land in one bump rather than two.

## 12. Design calls

**D1. Does `probation_sweeps` serialise always, or skip on default?** Full
re-key (always) versus preserved cache (skip). Recommendation: always, and land
it with the binomial field in one bump. Confidence: **leaning** — what would
flip me is an unfinished long PGAS run whose `resume_state.bin` you intend to
resume.

**D2. Should an exhausted recovery budget be reported, or should the fit
refuse?** Today a fit aborts only when _every_ chain is refused
(`cli/fit/pgas.rs:1171`). With a denominator now reported, there is an argument
for a threshold — refuse the fit when fewer than some fraction of requested
chains sampled, since R̂ over 9 of 16 on a family whose known problem is a
mechanistic R̂ near 1.2 is a weak test. Recommendation: report, do not refuse,
and let `--min-sampled-chains` express it if a user wants the gate. Confidence:
**need you** — this is a question about what a fit should be allowed to return,
not one the code answers.

**D3. Should `refusal_separation` be computed for every fit, or only when
refusals occur?** It is cheap either way. Computing it always gives a "refusals
do not track any coordinate" negative result, which is itself informative.
Recommendation: compute when `not_sampled` is non-empty; a separation statistic
over an empty set is not defined. Confidence: **solid** — proceeding this way
unless told otherwise.
