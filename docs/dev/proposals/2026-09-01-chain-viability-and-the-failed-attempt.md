# A refused chain is a failed attempt, not an infeasible draw

Date: 2026-09-01 Status: proposed\
Related: gh#780 (open), gh#783 (partly landed), gh#784 (landed, `2e00e135`),
gh#751 (open), gh#607, gh#334, PR#782 (closed, not merged)\
Note ref: `docs/dev/notes/2026-08-30-pgas-bad-init-criterion.md`

Re-keying authorisation: **given** (2026-09-01). Land in one bump with gh#802's
binomial sampler field — see §12.

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

## 2. Why the probation budget is operational, not calibrated

The note this proposal supersedes extends PGAS's probation from one sweep to `N`
and sizes `N` from `1 − (1−p)^N`, treating successive probation sweeps as i.i.d.
Bernoulli trials. That model needs a failed sweep to be an identical retry —
`X_{s+1} = X_s` — and needs a `p` worth plugging in. Neither holds.

**A failed sweep can hand the next sweep a different trajectory.** `csmc_as`
rebuilds its outgoing path by walking the ancestor array backward from the
selected index (`sim/inference/pgas.rs:2740-2761`), and ancestor sampling writes
into that array during the sweep. Even on the gh#783 collapse path, where the
final draw falls back to the reference _index_ (`pgas.rs:2707-2711`), the
reconstructed _path_ follows re-anchored ancestry. So the success probability of
sweep `s` is `p(θ₀, X_s)` with `X_s` moving, not a constant `p(θ₀)`. This is a
property of the code, not an empirical claim.

**And `p` was never estimable from what we had.** The note infers `p ≈ 1/17`
from a single observed first recovery at sweep 17. One success in seventeen
trials has an exact 95% interval of `[0.0015, 0.2869]` — a factor of 190. The
note's `N = 100 → 99.8%` is the plug-in point estimate; at the interval's lower
end, 100 consecutive failures still occur with probability 0.862. The conclusion
survives every alternative interval (Jeffreys 0.525, Wilson 0.349), so it is not
an artifact of choosing the conservative one. Strictly the datum is a geometric
waiting time rather than a fixed-`n` binomial, but the lower bound is
algebraically identical under both designs, and only the lower bound is used
here.

**So `N = 100` is an operational recovery budget.** Recoveries were observed at
sweeps 15 and 17, and a probation sweep costs 0.9-4.0 s (median 1.2) against a
2,000-sweep run, so 100 is affordable and comfortably past the observed
recoveries. It carries no coverage guarantee and this document claims none. Once
the budget exists it logs its own outcomes, and the default can be revised
against those rather than against a model.

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
filter's auxiliary randomness (ADH §4.4, p. 298). Two clarifications that an
earlier draft got wrong: the invariant distribution is the **joint**
`p(θ, x | y)` (ADH §2.4.2, p. 278) — the marginal is what justifies the
acceptance ratio, not what is targeted — and PMMH **does** carry a trajectory
across iterations (Step 2(c) sets `X(i) = X(i−1)` on rejection). The distinction
that matters here is narrower and survives both: PMMH never **conditions** the
next filter on the retained trajectory, so there is no PGAS-style "redraw `X` at
the same `θ`" repair available to it. Its exactness rests on that estimate being
**positive and unbiased**, so an ESS watchdog firing reports an unusable
estimator — _not_ `p(y | θ) = 0`. camdl's own initialization code already states
this bar: an empty finite-particle swarm "is never a claim about `p(y | θ₀)`"
(`pgas_init.rs`, `UnconditionalPass::NoSupport`).

**IF2** (Ionides et al. 2015, _PNAS_ 112:719-724) is stochastic optimisation
toward the MLE, not posterior sampling, so a filter failure is not a statement
about posterior support. It is not free of a positivity condition either —
Ionides et al.'s Theorem 1 condition (B4) bounds the observation density away
from zero — but a degenerated run is that condition failing somewhere along a
search, which is a reason to report a failed search, not to relabel its start.
Its failure is not even an initialisation event: `check_pf_degeneracy` fires
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
after one sweep. Its motivation was real, and precise — `sim/error.rs:139-142`
records it: on a 40,000-sweep, 8-chain production fit, **one chain** failed
40,000 consecutive times, contributing one distinct parameter vector across
7,600 retained draws — an eighth of a 2 h 29 m run pooled into the posterior and
R̂. One chain of eight, not the whole fit; a fit where every chain is refused
takes the different `all_results.is_empty()` path at `cli/fit/pgas.rs:1171`.
§2's measurement contains that exact pathology — the longest observed run of
consecutive `-inf` sweeps is 40,000.

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
// The tag is `cause`, not `kind`: two variants below carry a FIELD named
// `kind`, and serde rejects an internal tag colliding with a field name
// ("variant field name `kind` conflicts with internal tag" — verified against
// the workspace's serde 1.0.228).
#[serde(tag = "cause", rename_all = "snake_case")]
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

### 6.3 The run-level denominator — replacing `n_good_chains`, not joining it

camdl already counts this, twice, and neither place is good enough.
`n_good_chains` (`cli/fit/pgas.rs:1154`) prints `ran 9 of 16 chains` to stderr
and lands in `fit_state.toml` — but only as `Some(n)` when something failed
(`pgas.rs:1468`), which is exactly the "absence of a complaint" the honest
version has to fix. `chain_starts.tsv` carries a third copy of the requested
count in its header (`init.rs:1220`).

An earlier draft of this section added a fourth. **`ChainAccounting` replaces
`n_good_chains` outright**; `n_good_chains` and its `fit_state.toml` field are
deleted in the same commit, and `chain_starts.tsv`'s header count is derived
from the accounting rather than recomputed. One counter, one definition, always
present.

```rust
/// What the fit was asked for versus what it got. The single authority — this
/// REPLACES `n_good_chains`, which existed only on failure. Emitted always, so
/// `requested == sampled` is a positive assertion rather than silence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainAccounting {
    pub requested: usize,
    pub sampled: usize,
    /// One entry per requested-but-unsampled chain. `ChainNotSampled` is its
    /// own struct precisely so this cannot hold an unrelated diagnostic;
    /// `DiagnosticKind::ChainNotSampled(ChainNotSampled)` wraps the same value
    /// for the diagnostics list. One definition, two views.
    pub not_sampled: Vec<ChainNotSampled>,
    /// Populated when refusal tracks a coordinate of `θ₀` — see below.
    pub refusal_separation: Option<RefusalSeparation>,
}

/// Refused versus sampled starts, compared coordinate by coordinate. Reads the
/// `params` already on each `ChainNotSampled` and the starts already written to
/// `chain_starts.tsv`; no new instrumentation, and no second copy of either.
///
/// Direction is explicit rather than assumed. An earlier draft had only
/// `refused_max`/`sampled_min`, which silently presumes refusals sit BELOW the
/// sampled starts — true in the motivating case, and a printed falsehood in the
/// mirror case.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefusalSeparation {
    pub param: String,
    /// Which side the refused starts fall on.
    pub direction: Side,               // RefusedBelow | RefusedAbove
    /// The gap: the extreme refused value and the nearest sampled one, ordered
    /// by `direction`.
    pub refused_bound: f64,
    pub sampled_bound: f64,
    /// Fraction of the posterior on the refused side of `sampled_bound`, when a
    /// posterior exists. 0.915 in the motivating case. `None` at the two write
    /// sites that have no posterior (all chains refused) and for IF2.
    pub posterior_on_refused_side: Option<f64>,
}
```

**Selection rule, stated so it is not invented.** For each estimated parameter,
compute the refused and sampled ranges over `θ₀`. A coordinate qualifies when
the two ranges are disjoint. Report the qualifying coordinate with the largest
gap relative to the sampled range's span; report `None` when none qualifies,
which is itself informative — it says refusal does not track any single
parameter. Perfect separation only: a partial-overlap statistic invites
over-reading, and the claim this field exists to support ("the refusals are
systematically over here") needs the strong version to be worth printing.

Computed only when `not_sampled` is non-empty — a separation statistic over an
empty set is undefined.

## 7. The JSON sidecar

### 7.1 What exists

`diagnostics.json` is written per stage at `stage_dir/diagnostics.json`, and
`DiagnosticKind` already derives `Serialize`/`Deserialize`, so `BadInit` already
reaches consumers as structured JSON with `chain_id`, `params` and `reason`.
Four properties of the current arrangement are load-bearing for this proposal,
and the first is the one an implementer must not discover late:

1. **The top level is a JSON array, not an object.** `write_json`
   (`sim/inference/diagnostic.rs:628`) serialises `&*self.diagnostics.lock()` —
   a bare `Vec<Diagnostic>`, where each entry wraps the kind:
   `{"kind": {"type": "bad_init", …}, "severity": …, "message": …, "stage": …,
   "timestamp": …}`.
   Note the variant tag is `"type"`, and `"kind"` is the wrapper's field name —
   tests navigate `d["kind"]["type"]`. **§7.2 changes the top level from array
   to object.** That is the largest breaking change in this proposal, larger
   than any field rename, and four in-repo helpers call `.as_array()` on this
   file and break immediately: `cli/tests/pmmh_bad_init_skip.rs:315` (literally
   `expect("diagnostics.json is an array")`), `pgas_bad_init_skip.rs:273` and
   `:286`, `gh226_degenerate_fit.rs:210`. `docs/camdl-run-spec.md:335` names the
   file as a leaf artifact and needs the same edit.
2. **It is terminal-only.** All eight write sites are ends of the road:
   `cli/fit/pgas.rs:1171` and `:1431` are the all-chains-refused error paths,
   and `:1573` runs after the stage completes. During a ten-hour fit there is no
   supported way to learn that seven chains were refused in the first thirty
   seconds. That is gh#751 and the downstream request that "would have saved us
   the most time this month."
3. **The write result is discarded.** All eight sites are
   `let _ = collector.write_json(&diag_path.to_string_lossy());`. A failed
   sidecar write is silent.
4. **There is no schema tag.** `diagnostics.json` carries no version field, so a
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
        "type": "chain_not_sampled",
        "chain_id": 3,
        "params": { "I0": 35.1, "r_eff": 1.86, "…": 0.0 },
        "cause": {
          "cause": "pgas_recovery_exhausted",
          "sweeps_attempted": 100,
          "budget": 100,
          "init": "unconditional_filter",
          "last": {
            "log_posterior": "-inf",
            "transition": -2245.09,
            "observation": "-inf",
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

### 7.3 `-inf` is a value, and JSON must carry it as one

A log-density lives on the extended reals `[−∞, +∞]`. `−∞` means _this state is
off the support_ — a measurement, not a failure to take one. Every field of
`CompleteDataTerms` is such a density, and the observation term is `−∞` in
essentially every refusal this proposal is about.

JSON has no encoding for it, and `serde_json` maps every non-finite `f64` to
`null` — which is also what an absent field looks like. Collapsing those two
throws away the only signal a refusal record carries.

camdl has already made this decision one layer up. `chain_loglik_cell`
(`fit/fit_summary.rs:1326`) renders `-inf` loudly and `—` for "nothing
readable", and says why: "softening either one hides the only signal there is."
The terminal gets it right; JSON does not. So this is not a new convention, it
is an existing one crossing a layer boundary.

```rust
/// A quantity on the EXTENDED reals `[−∞, +∞]`, encoded so JSON does not lose
/// the distinction between "off the support" and "not computed".
///
/// Encoding: a finite value is a JSON **number**; a non-finite one is one of
/// the **strings** `"-inf"`, `"inf"`, `"nan"`. `null` keeps its ordinary
/// meaning — the field was not computed. A consumer branches on the JSON type,
/// which is why the file carries a `schema` tag (§7.2).
///
/// `NaN` round-trips as `"nan"` rather than vanishing, so it stays visible;
/// `diagnostic.rs` already holds the line that "`NaN` and `0.0` are different
/// diagnoses and must not be collapsed". Read it precisely: a `NaN` in
/// `transition`, `observation`, `ivp` or `log_prior` is a bug upstream. A `NaN`
/// in `log_posterior` with both a `+inf` and a `-inf` among its terms is a
/// genuine `∞ − ∞` — reachable without any bug, since camdl admits `beta` and
/// `gamma` priors (`ir/parameter.rs:60`) whose log-density diverges to `+∞` at
/// a boundary. Since all five terms travel together, a reader can tell which.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtendedReal(pub f64);

impl serde::Serialize for ExtendedReal {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() { s.serialize_f64(self.0) }
        else if self.0.is_nan() { s.serialize_str("nan") }
        else if self.0 > 0.0    { s.serialize_str("inf") }
        else                    { s.serialize_str("-inf") }
    }
}
// Deserialize accepts a number or one of the three strings; anything else errs.
```

`CompleteDataTerms` therefore becomes:

```rust
pub struct CompleteDataTerms {
    pub log_posterior: ExtendedReal,
    pub transition:    ExtendedReal,
    pub observation:   ExtendedReal,   // `-inf` in essentially every refusal
    pub ivp:           ExtendedReal,
    pub log_prior:     ExtendedReal,
}
```

**One more type is in scope than it looks.**
`InitFallback::NonFiniteDensity { transition, observation, initial_state }`
(`sim/error.rs:336`) carries three raw `f64` that are non-finite **by
construction** — it is built at `pgas_init.rs:136-140` precisely when the traced
lineage's density is not finite. Left as `f64`, the same refusal record would
carry `"observation": "-inf"` in `CompleteDataTerms` and `"observation": null`
in `init`. It takes `ExtendedReal` too.

**Scope it deliberately.** `ExtendedReal` is for densities and scores, not a
blanket replacement for `f64` in artifacts. Most floats camdl writes are finite
by construction, and widening this beyond the fields where non-finiteness is
_meaningful_ would make every consumer branch on a type union for no gain.

**One hazard, stated because it is not obvious.** `ensure_finite`
(`cli/fit/cas.rs:380`) rejects non-finite floats before hashing, by driving a
custom `FiniteCheck` serializer over the value — it exists because
`json!`/`to_value` collapse non-finites to `Null` before any gate can see them
(`140ef57f`). An `ExtendedReal` serialises a non-finite as a **string**, which
`FiniteCheck` does not inspect. **Putting an `ExtendedReal` into anything that
enters the run address would silently bypass the finiteness gate.**

The boundary is therefore: `ExtendedReal` is an **output** type, for diagnostics
and reports, and never an **input** type. Identity payloads keep raw `f64` and
keep the gate. This should be enforced by a test asserting no type reachable
from an identity payload contains an `ExtendedReal`, not left to the reader —
the failure mode is silent and the gate is load-bearing.

Three further changes, each small and each closing a live complaint:

- **Add `"schema": "camdl.diagnostics/v2"`.** This proposal changes the shape; a
  consumer that cannot detect the change will mis-parse silently. Same argument
  as gh#728.
- **Report refusals mid-run through `progress.json`, not a second artifact.** An
  earlier draft proposed a `diagnostics.partial.json` flushed as refusals occur.
  That would have been a second mid-run reporting channel beside one that
  already exists and is already the thing consumers poll. `RunState`
  (`io/src/progress.rs:83`) is an ADT written precisely "so incoherent
  combinations are unrepresentable", with one type serving every algorithm.
  Extend its `Running` variant instead:

  ```rust
  Running { phase: Phase, step: u64, total: u64, chains: ChainProgress },
  ```

  where `ChainProgress { requested, sampled, not_sampled, running }` accounts
  for every requested chain at every instant. That is what makes a ten-hour fit
  abandonable in its first minute — the single most requested change from
  downstream — and it lands in the file they already read, with no second
  artifact to keep consistent. It also fixes what §8 previously claimed and
  could not deliver: a chain **stranded** by the `--parallel` scheduler is
  `running` with no progress, which is visibly different from `not_sampled`.
- **Stop discarding the write result.** `let _ = collector.write_json(…)`
  appears at **eight** sites — `pgas.rs:1172, 1432, 1574`;
  `pmmh.rs:903, 929,
  1134`; `runner.rs:2269`; `nuts.rs:410` — each becoming a
  logged warning. A diagnostics file that silently failed to write is
  indistinguishable from a fit with no diagnostics. Note `nuts.rs` is a fourth
  driver the rest of this proposal must also cover, or its file carries a
  `schema` tag it does not honour.

### 7.4 Consumers

`camdl 'scope` (`camdl-watch`, a separate repo) reads these artifacts and
currently conveys refused chains poorly, which is a direct consequence of §7.1:
the only machine-readable signal is a `BadInit` entry whose `reason` is prose,
with no denominator anywhere. With `chain_accounting` present a viewer can
render "9 of 16 chains sampled" as a first-class fact, list the seven starts,
and — where `refusal_separation` is populated — say which coordinate the
refusals track.

The `output_schema` map already in `run.json` (`cli/src/resolve.rs:270`) is the
precedent for telling consumers a shape; `diagnostics.json` should carry its own
tag by the same logic.

**Action required before implementation:** enumerate the downstream consumers of
`diagnostics.json`. `camdl 'scope` is one — it "refreshes a run that is still
sampling" (`docs/agents.md:545`), so §7.2's `RunState` change is the one it
feels; the Ebola project's workflow is another. Per `VERSIONING.md` this is a
pre-1.0 breaking change and gets no compatibility shim, but the consumers should
be told rather than discover it.

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

One thing this deliberately does **not** claim. An earlier draft said the
accounting distinguishes a refused chain from one stranded by the `--parallel`
scheduler. It does not: `diagnostics.json` is written after `collect()`, and a
stranded chain blocks `collect()`, so in the case that matters the file is never
written at all. The distinction comes from §7.2's `RunState` change instead — a
stranded chain reports `running` and stops advancing, which is visibly different
from `not_sampled`, and `progress.json` is written while the fit is still going.

## 9. Non-goals

**Not unifying the predicate.** Three algorithms, three criteria, each with its
justification beside it. A single function taking a method discriminant would
erase that one algorithm carries latent state in its chain, one does not, and
one is not a chain.

**Not auto-redrawing `θ₀`.** Redrawing until `K` chains are viable is an
unreported rejection sampler from `q₀(θ)·a(θ)`. It does not bias the posterior —
each surviving chain is invariant to its start, and the motivating fit confirmed
this empirically, with all nine survivors reaching the bulk within 100–250
sweeps inside a 500-sweep burn-in. What it destroys is R̂'s diagnostic power.
Gelman & Rubin §2.1 requires that the starting distribution "**covers** the
target distribution in the same sense that an approximate distribution for
rejection sampling should cover the exact distribution" (1992, _Statist. Sci._
7:457-472) — so selecting starts on filter viability is not merely bad practice,
it violates a stated requirement of the diagnostic, in the same
rejection-sampling frame. Starts selected for filter friendliness are tight and
displaced in one direction, so chains agree before they have explored.
Rank-normalised split-R̂ (Vehtari et al. 2021, _Bayesian Anal._ 16:667-718)
repairs heavy tails and unequal scales; it cannot reconstitute chains removed
before sampling.

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

1. **`ExtendedReal` and `CompleteDataTerms`**, with a hand-written `Display` for
   `CompleteDataTerms` (the `#[error]` string interpolates it, and
   `sim/error.rs:658` and `cli/tests/pgas_bad_init_skip.rs:401` assert exact
   substrings such as `observation -inf`, so the format is pinned, not free).
   Replacing the five loose `f64`s on `NonFiniteChainStart` (§7.3). Pure
   refactor of the error's shape, no behaviour change, no re-key. Includes the
   test asserting no identity payload can reach an `ExtendedReal`, since that
   hazard is silent.
2. **`ChainNotSampled` + `NotSampledCause`**, with the three call sites
   converted through the single `From<&SimError>` mapping (§13). Prerequisite,
   unglamorous and easy to miss: `InitSource`, `InitFallback` and
   `PFDegenerateKind` (`sim/error.rs:283`, `:326`, `:257`) derive only
   `Debug, Clone, PartialEq` and need serde, with `rename_all` so
   `InitSource::UnconditionalFilter` emits `unconditional_filter`. `BadInit` is
   deleted outright — alpha posture, no alias. Changes `diagnostics.json`; add
   the `schema` tag in the same commit.
3. **`ChainAccounting`** including `refusal_separation`, emitted by all three
   drivers. This is the user-visible win and depends on step 2.
4. **Incremental flush** (`diagnostics.partial.json`) plus handling the write
   result. Independent of 1-3; could land first if downstream needs it sooner.
5. **The probation budget.** `PgasStartRecoveryExhausted` replaces
   `NonFiniteChainStart`, with `probation_sweeps` (default 100) as a typed field
   on `Stage::PGAS`. **This is the step that re-keys** — see §11.
6. **IF2 relabelling.** `If2SearchDegenerated` carrying `iteration`. The index
   IS in scope (`for iter in 0..config.n_iterations`, `if2.rs:386`); the real
   obstacle is that `pf_bail_error` (`degeneracy.rs:299`) and
   `SimError::PFDegenerate` are shared with PMMH and the particle filter across
   five call sites, so an iteration field must either be optional or IF2 must
   carry it outside the error. Decide that here, not mid-edit. Because step 2's
   `If2SearchDegenerated` variant needs the value, steps 2 and 6 land together.

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

## 12. Decisions taken

These were open when this document was drafted and are now settled. Recorded
with their consequence so the implementer does not reopen them.

**`probation_sweeps` is always serialised, and the re-key is accepted.** Every
existing PGAS leaf gets a new address and every `resume_state.bin` is
invalidated — a stale one does not degrade gracefully, it prints and
`std::process::exit(1)` (`cli/fit/pgas.rs:455-464`), so an in-flight `--resume`
must be re-run with `--force`. Accepted deliberately: a stored posterior whose
address does not name the budget that produced it is the failure the identity
system exists to prevent. **Land in one bump with gh#802's binomial field**
rather than re-keying twice. The field needs `#[serde(default = "…")]` or every
existing `fit.toml` fails to deserialize — a missing field with no default is a
hard serde error, independent of the always-serialise decision.

**A fit with too few sampled chains reports; it does not refuse.** The stage
continues and the accounting says what happened. No threshold gate and no
`--min-sampled-chains` flag — an unused knob is a knob to maintain, and the
number a user would set it to is not knowable in advance. If demand appears
later it is additive.

**`ChainAccounting` replaces `n_good_chains`.** See §6.3. Not additive.

**`N = 100` is operational and stays.** See §2. No further measurement is
required before implementation; the budget logs its own outcomes and the default
is revisable against those.

## 13. Nothing here adds a second implementation

camdl's standing rule is that a knob, a counter, or a policy has exactly one
definition. This proposal deletes more than it adds, and the places where an
implementer could accidentally fork are named here.

**Deleted outright** — alpha posture, no aliases, no compatibility shims:

| deleted                                                                                                                                              | replaced by                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `DiagnosticKind::BadInit` and its `severity()`, message and `suggestions()` arms                                                                     | `DiagnosticKind::ChainNotSampled(ChainNotSampled)`, one arm per cause |
| `n_good_chains` (`cli/fit/pgas.rs:1154`, `:1468`) and its `fit_state.toml` field                                                                     | `ChainAccounting` (§6.3)                                              |
| `SimError::NonFiniteChainStart`'s five loose `f64` fields                                                                                            | `CompleteDataTerms` (§6.1)                                            |
| `start_at_zero_density: Option<(f64, f64, f64, f64, f64)>` (`sim/inference/pgas.rs:3498`) — the same five floats one layer up, as an anonymous tuple | `Option<CompleteDataTerms>`                                           |
| `option_finite` (`cli/compare.rs:1492`), the ad-hoc `is_finite → null` collapse                                                                      | `ExtendedReal` (§7.3)                                                 |
| the `diagnostics.partial.json` an earlier draft proposed                                                                                             | `RunState::Running`'s chain counts (§7.2)                             |

**Single-sited by construction.** Three drivers currently build a `BadInit`
independently (`pgas.rs:1026`, `pmmh.rs:640`, `runner.rs:2162`), each mapping a
`SimError` to a diagnostic in its own way. Replace all three with one
`impl From<&SimError> for Option<NotSampledCause>`, so the taxonomy exists once
and each call site is a single line. The mapping must be total: a `SimError`
that reaches a driver and is neither structural nor a known cause is a
compile-time gap, not a `_` arm.

**Two enums that look like duplication and are not.** `SimError` is the engine's
transient failure channel; `NotSampledCause` is the serialized reporting
vocabulary with a schema tag and downstream consumers. They live at different
layers with different lifetimes and different stability guarantees. Merging them
would couple the engine's error handling to an artifact format — the wrong seam.
What must not happen is a _third_ place that classifies the same three events;
the `From` impl above is the only mapping.

**Adjacent code that is not duplicated and must not be reimplemented.**
`chain_diagnostics.rs:517`'s `read_chain_neginf` screens per-chain non-finite
counts over _retained trace rows_ — a different population (chains that sampled)
reached by a different route (reading `trace.tsv`, which a not-sampled chain
does not have). An implementer looking for "count the bad chains" will find it;
it answers a different question and stays.

**A pre-existing name collision to leave alone.** `sim::error::InitSource` and
`cli::fit::chain_starts::InitSource` are unrelated enums sharing a name. This
proposal touches the first. Renaming either is out of scope and would enlarge
the diff without improving it — noted so it is not mistaken for something this
change introduced.
