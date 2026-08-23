# Initial state: one spec per compartment, evaluated in dependency order

Status: proposed\
Closes: gh#719, gh#723, gh#732, gh#733\
Requires: `ir/VERSION` 0.33 → 0.34 (52 goldens, atomic OCaml + Rust)

## Assumed background

The three compatibility axes
([`capabilities-system.md`](../capabilities-system.md)); PGAS's complete-data
likelihood decomposition (`sim/src/inference/pgas.rs`, `complete_data_loglik`);
the golden / `ir/VERSION` human-loop rule (CLAUDE.md, `VERSIONING.md`).
Everything else is stated here.

## What is wrong today

**An initial condition cannot say it is stochastic.** `init {}` accepts only
`compartment = expr`. PGAS nonetheless adds a Binomial density on some initial
compartments, choosing which by finite-differencing a rounded integer — a probe
whose outcome depends on the chain's starting draw, so two chains of one fit can
carry different targets (gh#719).

**A compartment reference in `init {}` silently reads zero.**
`CompiledModel::initial_state` evaluates every parameterized IC against a
throwaway zero state (`compiled_model.rs:1912`), so

```camdl
init { A = A0
       B = A0 - A }        # A0 = 500, A = 500
```

yields `B = 500`, not `0`. It compiles with no diagnostic and the reference
survives into the IR. There is no ordering to appeal to: `Parameterized` is a
`HashMap<String, Expr>`, which has no iteration order (gh#733).

**A mixed block is unrepresentable.** `InitialConditions` is an enum over the
_whole_ block, and the variant is a constant-folding decision in the expander
(`expander.ml:6157`: all-const → `Explicit`, else `Parameterized`). One
deterministic entry beside one stochastic entry has nowhere to live.

**`ivp` means three things.** It is an IF2 perturbation schedule (`if2.rs:522`),
a precondition for `ic_free` (`runner.rs:482`), and an exemption from the scout
Â gate (`gating.rs:163`). Under PGAS it is parsed, folded into the fit hash, and
read nowhere.

**`ic_free` + plain `pmmh` is a live silent-wrong answer.** `validate_ic_free`
admits it (`methods.rs:588`) and `runner.rs:482` accepts an `ivp = true`
declaration as proof of per-particle spread — but the bootstrap PF copies one
deterministic initial state to every particle (`particle_filter.rs:129-133`), so
the spread the guard checks for never exists and `ic_free` degenerates to
dropping y₁, exactly as that guard's own comment warns (gh#732).

## Design

### The spec is per compartment

```rust
/// What one compartment's initial value is.
pub enum InitSpec {
    /// `S = N0 - I`. May reference parameters AND other compartments.
    Deterministic(Expr),
    /// `I ~ poisson(rate = I0)`. Integer compartments only.
    Count(InitCountLaw),
    /// A real compartment drawn from a continuous law.
    Real(InitRealLaw),
}

/// Ordered: init entries are evaluated in dependency order, and the order is
/// part of the model's identity.
pub struct InitialConditions(pub IndexMap<String, InitSpec>);
```

`Explicit` and `Parameterized` both collapse into `Deterministic`; constant
folding becomes a `CompiledModel` build detail rather than an IR-visible
variant. `FromDistribution` is **deleted**, not retyped — it has been present
since the first IR commit (`076e8397`), has no producer in either language
(`expander.ml` emits only the other two), and is typed
`HashMap<String, PriorDist>` against a vocabulary with no discrete laws, so it
could never have expressed a count seed.

### The law vocabulary reuses the argument structs, not the enum

```rust
pub enum InitCountLaw {              // CompartmentKind::Integer
    Poisson(PoissonLikelihood),         // { rate: Diffable }
    Binomial(BinomialLikelihood),       // { n: Expr (θ-independent), p: Diffable }
    NegBinomial(NegBinomialLikelihood), // { mean: Diffable, dispersion: Diffable }
}
pub enum InitRealLaw {               // CompartmentKind::Real
    Normal(NormalLikelihood),
}
```

Reusing the `Likelihood` _enum_ would admit `Bernoulli`, `Beta` and
`ZeroInflatedNegBinomial` on an integer compartment — illegal states, made
representable. Reusing its _structs_ keeps what matters: `Diffable` carries the
expression together with its per-parameter classified gradient from the OCaml
obs-gradient autodiff pass (`observation.rs:60-73`), which is the mechanism that
makes a new law's gradient correct by construction rather than by hand.
`BinomialLikelihood.n` is already sealed `#[differentiate(skip)]` with "must be
θ-independent"; importing that seal turns a silently-dropped `∂N/∂θ` into a
compile-time refusal.

`CompartmentKind` selects the admissible set at construction, so a continuous
law on an integer compartment does not compile.

`NegBinomial` ships in v1 rather than later. Introduction counts are clustered
rather than Poisson (Lloyd-Smith, Schreiber, Kopp & Getz 2005, _Nature_
438:355–359, doi:10.1038/nature04153 — offspring dispersion `k ≈ 0.16` for
SARS), and `NegBinomial` contains Poisson as `k → ∞`. Shipping Poisson alone
would be the special case presented as the general one.

### Syntax

```camdl
init {
  I ~ poisson(rate = I0)
  S = N0 - I               # reads the DRAWN value
}
```

`~` already means "is distributed as" in two grammar positions — parameter
priors (`parser.mly:429/437/445/453`) and observation likelihoods
(`parser.mly:877`, `scored ~ obs_likelihood`), both parsed as `funcall(kwargs)`.
Extending `init_entry` (`parser.mly:1333`) is the third consistent use, and
`obs_likelihood` is reusable verbatim.

**Measured, not assumed:** adding `comp [idx] ~ obs_likelihood` and
`comp idxs ~ obs_likelihood` to `init_entry` and running
`menhir --explain --strict` gives 1 shift/reduce and 2 reduce/reduce — identical
to the unmodified grammar, one state renumbered. The decision point is a single
terminal (`EQ` vs `TILDE`) after a shared prefix.

`=` keeps its present meaning exactly. The two forms then carry the distinction
that matters — deterministic versus stochastic — visible in the model file, with
no default to forget.

### Dependency-ordered evaluation

Init entries form a DAG over compartment references. Build it from each RHS's
`Expr::Pop` set, topologically sort, and evaluate against the **partially
built** state rather than a zero state. A cycle is a compile error naming the
cycle.

This is what makes the population budget hold without a `balance {}` block:
`I ~ poisson(rate = I0)` is drawn, then `S = N0 - I` reads the drawn value, so
the total is `N0` by construction rather than by a balance rewrite. It is also
the fix for gh#733 — the same change, and the reason to do them together.

No golden references a compartment in an init RHS (a walk over all 52
`ocaml/golden/*.ir.json` matching `{"pop": "<name>"}` inside
`initial_conditions` returns zero files), so nothing depends on the zero-state
behaviour.

### A law on a `balance {}` target is a compile error

The balance stage overwrites its target after every substep
(`lifecycle.rs:71-82`), so a draw there is discarded. Three sites already know
this by hand — the IVP detector skips it (`pgas.rs`), `csmc_as` recomputes it
(`pgas.rs:2166`), `lifecycle.rs:91` exempts it from the negativity check. It
becomes one validate-time rule with an E-code.

While there: `csmc_as` hardcodes `total_pop − Σothers` where `lifecycle.rs:75`
evaluates the declared `bal.expr`. Any model whose balance expression is not
exactly that gets two different initial states from the two paths. `csmc_as`
adopts the declared expression.

### The seam: one initial-state producer, four questions

`CompiledModel::initial_state` is a deterministic producer with ~18 call sites
(three forward backends, the bootstrap PF, the correlated PF, IF2's per-particle
loop, the ODE sensitivity seed). It splits:

```rust
fn initial_state_mean(&self, params) -> (IntState, RealState)        // ODE, render, preflight
fn initial_state_draw(&self, params, rng) -> (IntState, RealState)   // every stochastic forward path
fn initial_state_logpdf(&self, x0, params) -> f64                    // PGAS complete-data term
fn initial_state_logpdf_grad(&self, x0, params) -> Vec<f64>          // PGAS + NUTS
```

A law is a sampler _and_ a density _and_ a gradient. Today three sites hardcode
Binomial independently (`pgas.rs:2154` sampler, `pgas.rs:1237` density,
`pgas_grad.rs:454` gradient); adding a law to one and not the others gives NUTS
a gradient identically zero on that coordinate — the silent-bias class camdl
already hard-rejects for parametric `DerivedExpr` projections. The split makes
that structural. `poisson_logpmf` (`obs_loglik.rs:445`), its gradient
(`obs_loglik.rs:171`) and `StatefulRng::poisson` (`rng.rs:119`) already exist.

With `initial_state_draw` in place, the bootstrap PF draws per-particle initial
states, and gh#732 closes as a consequence rather than as a patch.

### `ivp` is renamed and gated

`EstimatedParam.ivp` → `perturb_only_at_t0`, at the user surface too. Keeping
pomp's word costs more than it saves: a pomp user reads `ivp()` as "this is an
initial-value parameter", which in pomp is simultaneously a schedule statement
and a modelling statement because pomp's `rinit` is where `i_0` lives. camdl
splits those and would be keeping the word for only the schedule half.
(`capabilities-system.md` already records `ivp` as "a fit-layer concept, not an
IR property".) Under the alpha posture this is one renamed TOML key.

It becomes a hard error under every algorithm with no perturbation schedule
(`pgas`, `pmmh`, `mh`, `nl-*`), routed through axis 3 beside the existing
`requires_priors` and hierarchical-prior checks — **after** `runner.rs:482`'s
`ic_free` precondition is re-expressed as what it actually needs (per-particle
spread at t=0), which today only IF2 delivers. Gating the flag before fixing the
precondition would make the `ic_free` + `pmmh` cell unsatisfiable rather than
correct.

## Out of scope, with reasons

**Zero-truncation.** Poisson and Binomial both put mass on `x₀ = 0`, which makes
the outbreak impossible and the observation likelihood `−∞`; under CSMC those
particles are dead weight. Conditioning on "an outbreak was observed" is a
zero-truncated law. Deferred to a follow-up issue rather than guessed at,
because whether to condition is a modelling choice and the right default is not
obvious.

**Joint seeds.** Because init expands per cell before emission
(`expander.ml:6118-6153`), `I[p in patch] ~ poisson(rate = λ)` means
_independent_ draws per patch. "One introduction, location unknown" is
Multinomial over cells and is not expressible. `simplex_groups` (`if2.rs:59`) is
the existing surface for an initial composition and is where that belongs; a
follow-up issue.

## Staging

1. **gh#733 alone**: `IndexMap`, dependency order, cycle error, partial-state
   evaluation. `Explicit`/`Parameterized` collapse to `Deterministic`. IR bump.
2. **The law types + `~` grammar**, `Deterministic` still the only reachable
   spec for existing models. Goldens regenerate once more with the new
   `InitSpec` shape.
3. **The `initial_state` split** (mean/draw/logpdf/logpdf_grad), all call sites.
   Behaviour-neutral for deterministic models.
4. **Wire the laws** through sampler/density/gradient; delete
   `detect_ivp_mappings` and `PROBE_STEP`; balance-target E-code.
5. **gh#732 + the rename + the axis-3 gates.**

Steps 1 and 2 each bump `ir/VERSION`. Land them adjacently so the goldens
regenerate twice rather than five times.

Stored runs are _already_ invalidated by any commit — `FitDigest.engine` folds
`VERSION_SHORT`, which includes the git hash (`cli/src/version.rs:12`), so
grouping saves nothing in the run store. What steps 4 and 5 invalidate is
**scientific**: a fit computed with an auto-applied initial-state term is not
comparable with one computed without it. That is the sentence to put in the
release notes.

## Already landed

`c988b91b` — only a `ParamKind::Probability` parameter may enter the existing
Binomial term. This removes the `−4.2 × 10⁸` class and makes the probe's outcome
deterministic in the chain's start, so nothing here is urgent. It is not a
substitute: detection remains an inference where a declaration belongs.
