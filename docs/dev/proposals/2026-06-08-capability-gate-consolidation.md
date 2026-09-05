---
date: 2026-06-08
status: proposal (v2 — revised after adversarial review)
area: cli / sim / capability dispatch
related:
  - 2026-06-05-unified-timeline-effect-architecture.md
issues: gh#192, gh#191, gh#15(review), gh#95
cross-ref: gh#119 (related; fixed separately — not a capability flag)
---

# Consolidate the capability gate: one source of truth, one dispatch seam

> **Revision note (v2).** v1 proposed
> `inference_capabilities(backend) =
> backend.capabilities() − REAL_COMPARTMENTS`.
> A four-lens adversarial review showed this is **wrong**:
> `OdeSim::capabilities()` is `REAL_COMPARTMENTS` only (`ode.rs:34`), so the
> subtraction zeroes ODE inference — which _does_ integrate real compartments
> (the deterministic-skeleton fit at `nlopt_stage.rs:92` /
> `survey --eval simulate`), pinned by the existing test `methods.rs:670`. The
> withholding of `REAL_COMPARTMENTS` is a property of the **chain_binomial
> stateless filter loops**, not of "inference." v2 fixes the formula
> (per-(backend, algorithm), declared not blanket-subtracted), enumerates all
> gate sites (including the _ungated_ `survey --eval pfilter` and `pfilter`
> paths and the _per-stage_ fit-run reality), makes RUNTIME_DT position-aware,
> narrows the Gillespie correction, resolves the LINEAGES seam, and preserves
> the rich hint text.

## Problem

"Can backend B run model M under algorithm A?" is answered inconsistently across
commands, by **two divergent capability definitions**, and the divergence has
already produced bugs that point in opposite directions.

1. **Real source of truth** — `Simulate::capabilities()`. `ChainBinomialSim` =
   `OVERDISPERSION | REAL_COMPARTMENTS | BALANCE | LINEAGES`
   (`chain_binomial.rs:112`); `OdeSim` = `REAL_COMPARTMENTS` (`ode.rs:34`);
   `GillespieSim` = `REAL_COMPARTMENTS | LINEAGES` (`gillespie.rs:43`). Used by
   forward `simulate` (`util.rs:1698`).
2. **Hand-rolled duplicate** — `check_model_capabilities`
   (`fit/methods.rs:407`),
   `match backend { "chain_binomial" => OVERDISPERSION,
   "ode" => REAL_COMPARTMENTS, … }`.
   Used by `profile`, `survey --eval simulate` (hardcoding `"ode"`), and
   `nlopt_stage`.

The duplicate has **drifted**: it grants chain_binomial only `OVERDISPERSION`,
omitting `BALANCE` (which the inference path _does_ apply, via the same
`step_one` kernel). And its error builder (`methods.rs:424-447`) only has
`features.push` branches for `OVERDISPERSION`/`REAL_COMPARTMENTS`, so an
unsupported `BALANCE` joins to empty → a blank `-` line.

### The gate sites today (verified on `main`)

| site                     | gate                                                                        | verdict on `balance{}` + chain_binomial | gap                                                                                            |
| ------------------------ | --------------------------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `simulate` (forward)     | real `backend.capabilities()` (`util.rs:1698`)                              | ✅ accept (correct verdict)             | error msg is bare `{:?}` Debug (worst of all)                                                  |
| `fit run`                | **none** (`FitRunConfig::build` + per-stage dispatch call nothing)          | ✅ accept                               | **gh#191 under-gate**; also _per-stage_ backends (nlopt scout=ode, pgas refine=chain_binomial) |
| `profile`                | hardcoded `check_model_capabilities`                                        | ❌ **false reject** + blank name        | **gh#192**                                                                                     |
| `survey --eval simulate` | hardcoded, hardcodes `"ode"` (`survey.rs:183`)                              | n/a (ode path)                          | only this branch is gated                                                                      |
| `survey --eval pfilter`  | **none** (`ChainBinomialProcess`, `survey.rs:382`)                          | ✅ accept                               | **gh#191 under-gate (missed in v1)**                                                           |
| `survey --eval auto`     | `required_capabilities().contains(OVERDISPERSION)` router (`survey.rs:156`) | routes                                  | "model picks backend" logic that must agree with the gate                                      |
| `pfilter`                | **none** (hardcodes `ChainBinomialProcess`, `pfilter.rs:279`)               | ✅ accept                               | **gh#191 under-gate (missed in v1)**                                                           |

So the same question gets answered too-strict (profile) and too-lax (fit run,
`survey --eval pfilter`, `pfilter`) for the same model class. A third symptom:
`Expr::Dt` models are gated nowhere off Gillespie/ODE (review finding #15).

The unified-timeline proposal already scoped this consolidation
(`2026-06-05-unified-timeline-effect-architecture.md`, §"The gate, and the
consolidation it forces"); it has not landed.

## Root cause (corrected)

`REAL_COMPARTMENTS` withholding is **not** a property of "inference" — it is a
property of the **chain_binomial stochastic filter loops**, whose
`ParticleState` carries no real state, so a real reservoir is frozen at init
(gh#191). The **ODE deterministic-skeleton** inference path
(`compute_ode_loglik` via `nlopt_stage` and `survey --eval simulate`) integrates
real compartments correctly and must keep `REAL_COMPARTMENTS`. The capability
profile is therefore **per-(backend, algorithm)**, not per-backend, and
certainly not a blanket subtraction.

## Design

### 1. One Result-returning gate function, in `sim`, that every site calls

The structural anti-fork mechanism is not a test — it is that **all
accept/reject decisions route through one function**. Define in `sim`:

```rust
/// The capability profile a given (backend, algorithm) actually HONORS at run
/// time — declared per profile, NOT derived by blanket subtraction (a blanket
/// `caps - REAL_COMPARTMENTS` is wrong for ode; subtraction-by-default risks
/// auto-granting a flag the loop doesn't honor — the gh#95/#119 family).
fn inference_capabilities(backend: Backend, algorithm: Algorithm) -> Capabilities;

fn check_capabilities(
    required: Capabilities,        // from CompiledModel::required_capabilities()
    provided: Capabilities,        // forward: backend.capabilities(); inference: inference_capabilities(b,a)
    extra_required: Capabilities,   // request-raised, e.g. LINEAGES from --lineages (forward seam only)
) -> Result<(), CapabilityError>;
```

Declared profiles (audited per flag, with the gh#191 reason documented at the
one exclusion site):

- **chain_binomial filter family** (if2 / pgas / pmmh / pfilter, incl.
  `survey --eval pfilter`): `OVERDISPERSION | BALANCE` (+ `LINEAGES` **only if**
  an audit confirms the filter loops honor lineage recording — otherwise omit;
  do **not** auto-grant by subtraction). Withholds `REAL_COMPARTMENTS` (gh#191).
  Fixes gh#192 (BALANCE now granted) and keeps gh#191.
- **ode skeleton** (nlopt, `survey --eval simulate`): `OdeSim::capabilities()`
  unchanged — **keeps `REAL_COMPARTMENTS`** (preserves the `methods.rs:670`
  invariant; ODE integrates real compartments).
- **forward `simulate`**: keeps the full `Simulate::capabilities()` trait gate
  (a real-coupled chain_binomial _forward_ sim must still be accepted — guard
  against accidentally routing forward sim through the inference profile).

### 2. Route every site through it (the corrected enumeration)

- `fit run`: **per stage**, not once at `build()`. Each fit stage has its own
  `backend`/`algorithm` (`config_v2.rs` per-variant `backend` fields; the
  `backend()` accessor exists). Iterate stages, call
  `check_capabilities(required, inference_capabilities(stage.backend,
  stage.algorithm), …)`;
  **delete** the redundant hardcoded `nlopt_stage.rs:92`
  `check_model_capabilities("ode", …)`.
- `profile`, `pfilter`, `survey --eval {simulate, pfilter}`: call the same gate
  with their actual (backend, algorithm). This closes the gh#191 under-gate on
  `pfilter` and `survey --eval pfilter` that v1 missed.
- `survey --eval auto` router (`survey.rs:156`): add an invariant test that the
  router never resolves to a (backend, algorithm) the gate would then reject.
- `simulate` (forward): keep the trait gate, but route its message through the
  shared builder (below) so its wording stops being the worst of the set.

### 3. Error quality: preserve the rich hints; fix the blank structurally

- The blank `-` is a **missing-branch** bug, not a missing-name bug. bitflags
  2.x already gives `iter_names()`; iterate `unsupported.iter_names()` so a flag
  can never render blank — no new `name()` needed.
- **Preserve the rich hint text.** Move the existing multi-sentence
  OVERDISPERSION / REAL_COMPARTMENTS guidance (`methods.rs:425-447`, citing
  gh#191 / "frozen" / the backend-switch fix) verbatim into
  `Capabilities::hint(flag) -> &'static str` (a const table in `sim`). The gate
  builds its message from `(name, hint)` per unsupported flag.
- **Upgrade `simulate`'s `{:?}` Debug message** (`util.rs:1702`) to the same
  builder — a net error-quality _improvement_ on every path, regression on none.

### 4. RUNTIME_DT (gh#15) — position-aware, with a real AST walk

- `required_capabilities()` (`compiled_model.rs:1015`) **does not currently walk
  rate ASTs** (only draw_method / real_comp_indices / balance). Add a recursive
  `Expr::Dt` scan reusing the shape of `expr_is_time_dependent`
  (`compiled_model.rs:195`): recurse BinOp/UnOp/Cond/TableLookup-indices/Reduce
  **and resolve `BindingRef`** against model bindings.
- **Position-aware:** raise `RUNTIME_DT` only from `Dt` reachable in
  **rate/transition (and overdispersion σ²)** expressions — NOT from observation
  or initial-condition expressions, where `Dt` evaluates to a hardcoded `0.0` on
  _every_ backend incl. chain_binomial (`obs_model.rs:71`,
  `compiled_model.rs
  :1066`). An IR-wide `any(Dt)` scan would make the gate
  _falsely accept_ `Dt`-in-observation on chain_binomial (which silently yields
  0.0) — relocating the very bug we're closing. `Dt`-in-observation/IC, if it is
  a bug, needs its own diagnostic; flag separately.
- Declare `RUNTIME_DT` provided by chain_binomial forward + the chain_binomial
  filters (verified: the filters pass the realized `dt_substep` into the
  binomial/gamma densities, `pgas.rs:787`), not by Gillespie/ODE.

### 5. Gillespie `REAL_COMPARTMENTS` (gh#95) — narrow, don't blanket-withdraw

Blanket withdrawal **over-rejects**: Gillespie samples a real-coupled model
_correctly_ when no transition rate reads a real compartment that moves between
events (pure accumulators / non-feedback ODE sub-systems). The unsafe set is "a
transition **rate reads** a between-events-varying real compartment."

- Add `expr_reads_real_compartment` (sibling of `expr_is_time_dependent`) and
  raise a finer `REAL_COUPLED_RATE` requirement only when a rate reads a real
  compartment; gate that off Gillespie. This rejects exactly the gh#95-unsafe
  class and leaves time-homogeneous real-accumulator models running.
- If a blanket interim withdrawal is taken instead, the proposal must state it
  over-rejects the safe subset as a deliberate conservative choice — not present
  it as a cost-free capability-truth fix.

## Staging (acceptance-changing stages flagged honestly)

- **Stage 1 (no acceptance change):** `iter_names()` + `Capabilities::hint()`
  table; route `simulate`'s message through it. Kills the blank-name class and
  upgrades simulate's wording. **Re-baseline list:** any test asserting
  `simulate`'s old `{:?}` string (`util.rs:1703`) or
  `check_model_capabilities`'s prose (`methods.rs:448`).
- **Stage 2 (closes gh#192 + gh#191 — acceptance change on the under-gated
  paths):** single gate + declared inference profiles; route fit-run (per
  stage), profile, pfilter, survey paths through it. **Newly rejects**
  real-compartment models under `pfilter` / `survey --eval pfilter` /
  `survey --eval auto` / chain_binomial fit stages (the gh#191 fix) with a named
  error. Re-baseline those. **Green-keepers:** ode/nlopt real-compartment still
  `Ok` (`methods.rs:670`); forward `simulate` chain_binomial real-compartment
  still accepts.
- **Stage 3 (acceptance change incl. FORWARD sim — needs migration notes):**
  `REAL_COUPLED_RATE` off Gillespie (gh#95) and `RUNTIME_DT`. Withholding from
  Gillespie changes **forward `simulate`** acceptance via the shared trait gate
  → re-bless `smoke_all_golden` / `gate_*_baseline` fixtures that run
  real-coupled models on Gillespie, and add a `docs/language-changes.md` entry
  naming the backend swap (gillespie → chain_binomial for real-coupled rates),
  per CLAUDE.md "breaking language changes must signpost the migration."

## Testing

- **In-process exhaustive matrix on the ONE gate function** (the commands
  `process::exit` and have no Result seam, so test the function, not the
  binary): over (backend × algorithm × feature-model {balance, overdispersed,
  real-accumulator, real-coupled-rate, Dt-in-rate, Dt-in-obs}), assert the
  accept/reject verdict. The command layer is a thin Err→exit shell,
  spot-checked by one integration test.
- **Reachability, not enumeration:** the anti-fork guarantee is that every
  command routes through the one function — not a hand-listed matrix that a new
  command can silently dodge. (A new ungated command is exactly how this fork
  was born.)
- **Red tests:** real-compartment under `pfilter`, under `survey --eval pfilter`
  / `--eval auto`, and under a chain_binomial fit stage (before: accepted;
  after: named REAL_COMPARTMENTS error); `balance{}` under `profile` (was the
  false reject); Stage 3: real-coupled-rate under `simulate --backend gillespie`
  (before accepted, after rejected) + a fit config with an ode-nlopt scout +
  chain_binomial-pgas refine on an overdispersed model (must reject naming the
  _ode_ stage, accept once the nlopt stage is removed).
- **Green-keepers (regression guards):** `methods.rs:670` ode real-compartment
  `Ok`; forward `simulate` chain_binomial real-compartment accepts.
- **Hint-text guard:** assert the REAL_COMPARTMENTS message still contains
  "gh#191" and "frozen" (mirror `methods.rs:667-668`) so the re-baseline can't
  silently downgrade the hints.
- **Every-flag-has-a-name+hint** test (guards the next added flag from the blank
  class).

## Out of scope / cross-references

- gh#95 _sampler_ fix (thinning) — its own RFC; here we only correct the
  capability _declaration_ (`REAL_COUPLED_RATE`).
- gh#191 _full_ fix (carry real state in `ParticleState`) — here we only ensure
  the gate fires on every chain_binomial-filter path. Re-granting
  `REAL_COMPARTMENTS` is that fix's completion criterion.
- **gh#119** (frozen parameterized caches) — _related, fixed separately_: it is
  not a capability flag (one-line cross-reference only, to avoid implying a flag
  fix this proposal doesn't make).
- LINEAGES is a **forward-simulate/event-log** request-raised requirement with
  **no inference caller** (verified: zero references in `fit/`, `profile`,
  `survey`, `pfilter`); it flows through `extra_required` at the _forward_ seam
  (`util.rs:1798`), not the inference profile.

## Backwards compatibility

Alpha — not a concern. Error wording changes (re-baselined); Stage 2/3 newly
reject silently-wrong models, with named errors + a language-changes entry for
the Stage-3 forward-sim change.
