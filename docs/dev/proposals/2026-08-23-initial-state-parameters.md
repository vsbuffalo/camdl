# Initial-state parameters: one name per concept, one declared distribution

Status: proposed\
Supersedes: nothing\
Closes: gh#719 (design half), gh#723

## Background the reader is assumed to bring

The `Capabilities` bitflags and the three compatibility axes
([`docs/dev/capabilities-system.md`](../capabilities-system.md)); PGAS's
complete-data likelihood decomposition (`sim/src/inference/pgas.rs`,
`complete_data_loglik`). Everything else is stated here.

## The problem, in one sentence

Two unrelated mechanisms are both called "IVP", one of them decides whether to
apply itself by finite-differencing a rounded integer, and neither is declared
by the modeller.

## The two things called IVP

**1. A perturbation schedule (IF2).** `EstimatedParam.ivp: bool`
(`sim/src/inference/types.rs:67`) means "perturb this parameter only at `t = 0`,
not at every observation". It is read in exactly one place, `if2.rs:522`:

```rust
if spec.ivp || simplex_member_indices.contains(&spec.index) { continue; }
```

It skips the random-walk jitter. There is no density term anywhere near it.
camdl's own doc comment says this matches pomp's `ivp()` in `rw.sd`; that
correspondence is **relayed from the comment, not verified against pomp** (pomp
is not installed in this checkout).

**2. An initial-state density (PGAS).** `IVPMapping` (`pgas.rs:294`) causes
`complete_data_loglik` to add

```
log Binom( x₀[c] ;  N_patch,  θ )
```

reading the parameter `θ` as the **probability** that each of `N_patch`
individuals starts in compartment `c`. This makes `x₀` stochastic so the Gibbs
structure can sample `θ` through it.

These share a name and nothing else. One is a schedule, the other is a
likelihood term. **PGAS never reads the flag**, and IF2 never adds the density.

### What that costs today

A modeller who writes `ivp = true` in a fit config and runs `algorithm = pgas`
gets silence: the key is parsed, folded into the fit hash, reported in the
pre-flight summary (`fit/mod.rs:706`), and then ignored. That is the exact
pathology axis 3 exists to prevent — "a model accepted on a backend by one
algorithm is rejected on the same backend by another", except here it is not
rejected, it is silently inert.

Conversely, a modeller who declares nothing can still get mechanism 2, because
PGAS infers it (`detect_ivp_mappings`) by nudging each estimated parameter and
asking whether a **rounded** initial count moved. Nobody asked for it and
nothing reports it as a modelling choice.

## Design

### Types

Replace the inferred boolean with a declared distribution, per parameter.

```rust
/// The law the initial compartment count follows, given θ.
///
/// Declared per estimated parameter; never inferred. `Deterministic` is the
/// default and adds no term to the complete-data likelihood.
pub enum InitialStateLaw {
    /// `x₀[c] = round(f(θ))`. No density term; θ is estimated through the
    /// trajectory density alone.
    Deterministic,
    /// `x₀[c] ~ Binomial(N_patch, θ)`. Requires `ParamKind::Probability`.
    /// The estimand is an initial *prevalence*.
    Binomial,
    /// `x₀[c] ~ Poisson(θ)`. Requires `ParamKind::Count` or `Positive`.
    /// The estimand is an initial *count* — an introduction size.
    Poisson,
}

/// Which compartment an initial-state law attaches to, and under which law.
/// Replaces `IVPMapping`.
pub struct InitialStateDensity {
    pub param_idx: usize,
    pub model_param_idx: usize,
    pub compartment_idx: usize,
    pub law: InitialStateLaw,
}
```

`EstimatedParam` carries the declaration and loses the overloaded name:

```rust
pub struct EstimatedParam {
    // …
    /// IF2 only: perturb at t = 0 and not thereafter. Was `ivp`.
    pub perturb_only_at_t0: bool,
    /// The initial-state law this parameter carries. `Deterministic` unless
    /// declared. Read by PGAS/PMMH; ignored by IF2, which has no
    /// complete-data likelihood to add it to.
    pub initial_state: InitialStateLaw,
}
```

`LogLikComponents.ivp` becomes `initial_state`, and gains a trace column
(`initial_state_ll`) next to `transition_ll` and `obs_ll`. A constant component
of the target that can only be recovered as
`log_complete_data_ll − transition_ll − obs_ll` is how gh#719 took a
trace-forensics pass to find.

### Surface

```toml
[estimate]
# an introduction size we believe as a count
I0 = { start = 550.0, bounds = [2.5, 4100.0], initial_state = "poisson" }
# an initial prevalence
p0 = { start = 3e-5, bounds = [1e-6, 1e-3], initial_state = "binomial" }
# ordinary: absent means Deterministic
r_eff = { start = 1.2, bounds = [0.5, 3.0] }
```

`ivp` stays spelled `ivp` at the user surface. It is pomp's word, modellers
arriving from pomp look for it, and renaming a user-facing key to fix an
_internal_ collision would be paying the wrong party. The collision is resolved
by giving mechanism 2 its own name — `initial_state` — and by renaming the
internal types, which no user reads.

### Why Poisson for the count case

This is the case the current code cannot express, and it is the common one: an
outbreak seeded by a known or estimated number of introductions. Binomial is the
wrong shape for it — it needs a denominator `N_patch` that carries no meaning
for a seed count, and it caps the seed at the patch population. Poisson needs no
denominator, has no upper bound, and is the `N → ∞, Np → λ` limit of exactly the
Binomial the current code writes. A modeller who knows "about three people were
infectious at t₀" is describing a Poisson mean, not a binomial proportion.

### Resolution of the fraction-versus-count question

**Both, declared, never inferred.** The two parameterisations are not
alternatives to choose between globally — they answer different questions, and
which one applies is a property of the parameter, not of the model or the
backend. A model may reasonably carry one of each.

### Capability gate

`ivp = true` under `algorithm = pgas` or `pmmh` becomes a **hard error at config
load**, naming the flag, the algorithm, and the fix: it is an IF2 perturbation
schedule and those algorithms have none. Routed through axis 3 alongside the
existing `requires_priors` and hierarchical-prior checks (`config_v2.rs`,
`pgas.rs`), which is where a config-key × algorithm rejection belongs.

`initial_state = "binomial" | "poisson"` under `algorithm = if2` is likewise a
hard error: IF2 has no complete-data likelihood to add a density term to.

Kind mismatches (`binomial` on a `count`, `poisson` on a `probability`) are
config-load errors naming the parameter, its declared kind, and the law it was
given.

### Removal

`detect_ivp_mappings` and `PROBE_STEP` are deleted. Nothing infers an
initial-state law.

## Modeller UX risks, and what each is worth

1. **An opt-in declaration is easy to forget.** A modeller who wants a
   stochastic seed and omits `initial_state` silently gets a deterministic one.
   This is the main risk of the design and it is not fully removable —
   mitigation is that the fit pre-flight always prints one line per estimated
   parameter carrying a non-deterministic law, and prints "all initial
   conditions deterministic" when none does. Silence must never be ambiguous
   between "not declared" and "not supported".

2. **The word `ivp` will keep misleading people** as long as it means only the
   IF2 schedule while looking like it means initial-value estimation generally.
   The hard error under PGAS/PMMH converts that from a silent no-op into a
   message that teaches the distinction at the moment it matters.

3. **Migration is a no-op for correct models.** A model whose initial-state
   parameter is `probability`-kinded and currently auto-detected must add
   `initial_state = "binomial"` to keep the term. Nothing else changes. Across
   the ebola corpus — 542 stored chains, 75 fits — **zero** fits carried a
   correct initial-state fraction, so nothing there regresses; the parameters
   that did enter the path were `I0` (`count`) and `tau`/`gamma`/`omega_base`
   (`rate`), all of which the interim kind guard already excludes.

4. **Deterministic is a real choice, not a degradation.** It is what 530 of
   those 542 chains ran, and those are the fits that produced usable posteriors.
   The docs must say so, or modellers will read the default as "off" and add a
   law they do not need.

## Staging

1. Rename internals (`ivp` → `perturb_only_at_t0`, `IVPMapping` →
   `InitialStateDensity`, `LogLikComponents.ivp` → `initial_state`); add the
   `initial_state_ll` trace column. Behaviour-neutral.
2. Add `InitialStateLaw` + the `initial_state` config key, honoured for
   `Binomial`. Keep detection as a deprecated fallback that warns when it fires
   and names the declaration that would replace it.
3. Add `Poisson`.
4. Delete detection and `PROBE_STEP`; add the axis-3 gates.

Steps 1 and 4 are the ones that move stored output: 1 renames a trace column and
4 removes an auto-applied likelihood term. Both re-key. Land 1 and 2 together,
then 3 and 4 together, rather than four separate invalidations.

## What is already done

The interim safety fix is on `main` at `c988b91b`: only a
`ParamKind::Probability` parameter may enter the Binomial term. That removes the
`-4.2e8` class and makes detection deterministic in the chain's start, without
deciding anything this proposal decides. It is not a substitute for the design —
detection is still an inference where a declaration belongs — but it means the
design is not urgent.
