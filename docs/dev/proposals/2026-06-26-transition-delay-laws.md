# Transition delay laws: non-exponential transit delays via phase-type lowering

Date: 2026-06-26 Status: Draft Relates: gh#313 (non-exponential sojourns),
gh#314 (forcing lag — landed)

## 1. The problem

A transition `A --> B @ rate` is exponential: residence in `A` is memoryless,
coefficient of variation 1. Real infectious-disease waiting times are not.
Latent and incubation periods are peaked (CV well below 1) — a measles latent
period is ~8 days with modest spread, not an exponential with a fat tail of
instant and very-late progressors. The number of "stages" in such a period
materially changes epidemic speed, peak timing, control thresholds, and the
inferred reproduction number from the same data (Wearing, Rohani & Keeling 2005,
_PLoS Med_ 2:e174; Lloyd 2001, _Proc. R. Soc. B_ 268:985).

The standard fix is the **method of stages** (the linear chain trick): an
Erlang-`k` delay with mean `τ` is exactly a chain of `k` exponential sub-stages
each with rate `k/τ`. The sum of `k` exponentials is Erlang, mean `τ`, variance
`τ²/k`, CV `1/√k` — so `k` tunes the spread and `k→∞` approaches a fixed delay.

camdl can already express this, manually. The committed golden
`ocaml/golden/seir_erlang.camdl` builds an Erlang-3 latent period by hand:

```camdl
dimensions { latent_stage = [e1, e2, e3] }
stratify(by = latent_stage, only = [E])
transitions {
  infection : S --> E[e1]                         @ beta * S * I / N
  latent[(s, s_next) in consecutive(latent_stage)]
            : E[s] --> E[s_next]                   @ 3 * sigma * E[s]
  onset     : E[e3] --> I                          @ 3 * sigma * E[e3]
}
```

Three costs:

1. **A silent-wrong footgun.** The per-stage rate is `3 * sigma` — `k` times the
   exponential-equivalent rate. Write `sigma` and the mean is wrong by `k`, with
   no error — a silent inference bug, the class camdl exists to make
   unrepresentable.
2. **Verbosity that scales with `k`.** A near-fixed delay wants large `k` —
   enumerating `k` dimension labels by hand. The cost grows where it hurts most.
3. **Buried intent.** The stratify-plus-`consecutive` incantation hides the one
   fact the modeller cares about: this is an Erlang-3 latent period of a given
   mean.

We want one line that says that, and lowers to exactly the chain above.

### What this feature does — and does not — cover

The phase-type chain is a **transit delay** between two compartments: the
in-transit individuals sit in hidden intermediate stages. This is correct
precisely when those individuals are **inert** — not referenced by any rate.

- **Inert delays (in scope):** latent / incubation periods (incubating
  individuals do not transmit — `E` appears in no rate), and onset-to-report
  delays. This is the canonical 80–90% case, and it is exactly the Garki/DMT
  incubation that motivated gh#313.
- **Active residences (out of scope, follow-up):** a non-exponential
  **infectious** period. Infectious individuals transmit, so they must live in a
  **visible** compartment that feeds `beta·S·I/N`; a hidden pipeline would
  silently drop them from the force of infection. Reshaping an active residence
  needs visible sub-staging (the stages stay referenceable and their sum feeds
  rates) — a separate feature, flagged in §7.

## 2. Design — the `via` clause

A transition carries an optional **delay law** introduced by `via`. The arrow is
still the biological event; `@` is still a propensity; `via` says how the
in-transit time between leaving `FROM` and arriving at `TO` is distributed.

```
NAME : FROM --> TO
  @ ENTRY_FORCE
  via LAW(args...)
```

`@ ENTRY_FORCE` is the propensity at which individuals **leave `FROM` and begin
the journey** to `TO` — a force of infection, an importation rate. It is _not_ a
progression rate. `via LAW` governs the **transit delay**: the time from
beginning the journey to arriving at `TO`, spent in a hidden phase chain. Absent
`via`, the transition is exactly today's exponential transition (zero transit
delay — arrival is immediate on firing).

```camdl
infection : S --> I
  @ beta * S * I / N
  via erlang(stages = 3, mean = 7 'days)
```

Reading: individuals get infected at the force of infection (residence in `S` =
time-to-infection), then spend an Erlang-3 latent period (mean 7 days) in a
hidden pipeline, then become infectious (`I`). The hidden pipeline _is_ the
latent class — no explicit `E` needed.

Block-form equivalent (the brace transition body already exists, accepting
`rate` and `where`; `via` is a third field):

```camdl
infection : S --> I {
  rate = beta * S * I / N
  via  = erlang(stages = 3, mean = 7 'days)
}
```

### Types

The transition AST (`ast.ml`) gains one optional field plus a reserved
competing-exit slot (empty in v1, see §7.1); nothing else changes:

```ocaml
type transition_decl = {
  (* ... trname, trindices, trsrc, trdst, trrate, trguard, trlineage ... *)
  trvia       : via_call option;        (* NEW: None ⇒ ordinary exponential transition *)
  trcompeting : competing_exit list;    (* reserved, [] in v1 (§7.1) *)
}

(* What the PARSER produces: a raw, untyped law call (name + kwargs). The parser
   has no typed-int / branch-record, so the typed via_spec below is built in the
   EXPANDER, where stages/mean/rate/weight are extracted and validated. *)
and via_call = string * (string * expr) list

(* What the EXPANDER lowers a via_call into, after validation. *)
type via_spec =
  | Erlang      of { stages : pos_int;        mean : mean_spec }
  | HyperErlang of { branches : hyper_branch list }
  (* deferred: Coxian, ApproxGamma — see §7 *)

and hyper_branch = {
  weight : expr option;   (* None on the LAST branch ⇒ 1 − Σ others (decision below) *)
  stages : pos_int;
  mean   : mean_spec;
  label  : string;
}

and mean_spec = Mean of expr | Rate of expr   (* exactly one — encodes the XOR *)
```

`stages` is a **compile-time positive integer** — the number of hidden
compartments, a structural quantity (like a stratification cardinality), not a
fittable parameter. It is parsed as a `pos_int` (a smart constructor folding
extract-literal + positivity into one seam, erroring at construction), not a
plain `int` — so `stages = 0` is unrepresentable, not validated downstream.
`mean` / `rate` / `weight` are **expressions** and may reference parameters;
they land in ordinary mass-action rates and are fully estimable (see §6).

### Grammar

`via` is the only new reserved keyword. Law names (`erlang`, `hyper_erlang`),
argument names (`stages`, `mean`, `rate`, `weight`, `label`), and the per-branch
constructor `branch(...)` stay ordinary identifiers, dispatched by string in the
expander — exactly how any function call `f(k = expr, …)` parses in a rate
expression (`atom_expr` → `EFuncCall`). The inline rule gains an optional `via`
clause between the rate and the `where` guard; the block body gains a
`via = LAW(...)` entry. (`rate` is a reserved keyword but is already accepted as
a kwarg name, so `erlang(rate = ρ)` needs no grammar change.) Adding the clause
is conflict-neutral — measured against the existing transition grammar, the
shift/reduce count is unchanged.

Hyper-Erlang groups each branch into its own `branch(...)` call rather than
positionally-aligned parallel lists — each branch is self-contained, so there
are no per-attribute lists to keep index-aligned (the fragile failure mode).

## 3. The laws — scoped to what 80–90% of users need

| Law                            | Covers                                                                 | Phase                           |
| ------------------------------ | ---------------------------------------------------------------------- | ------------------------------- |
| `erlang(stages, mean\|rate)`   | Unimodal non-exponential inert delays — latent/incubation periods      | **First**                       |
| `hyper_erlang(branch(...), …)` | Bimodal / mixture delays: short-vs-long incubation, fast-vs-slow onset | **Second** (cheap given Erlang) |
| `coxian(...)`                  | General acyclic phase-type — approximates any positive delay           | Deferred (§7)                   |
| `approx_gamma(shape, mean)`    | Non-integer gamma shape (fittable _shape_)                             | Deferred (§7)                   |

`erlang` is the must-have. `hyper_erlang` is nearly free once `erlang` exists (a
probability-weighted entry into parallel Erlang chains, §4) and covers the
bimodal case, so it ships in the same push. `coxian` and `approx_gamma` are the
long tail.

**Fixed (deterministic) delays.** A point delay — everyone takes exactly `τ` —
is common in ID modelling (a near-fixed incubation is the DMT case). The Erlang
chain **cannot** produce one: CV is `1/√k`, so a near-deterministic delay needs
very large `k` (CV 0.05 ⇒ `k = 400` compartments), reintroducing the cost-2
verbosity. v1 expresses a fixed delay as a high-`k` `erlang` **as an explicit
stopgap** — `O(k)` cost, residual CV `1/√k`, not exact. An _exact_ deterministic
delay needs a discrete conveyor (a shift register advancing one slot per `dt`),
which is different machinery and only well-defined in discrete time; that is a
named follow-up, not v1.

## 4. Lowering

`via` lowering is an **AST→AST pre-pass that runs during expansion, before
stratification and the duplicate-name guard** (`check_declaration_names`). It
rewrites a `via` transition into explicit hidden `compartment_decl`s + ordinary
`transition_decl`s; stratification then replicates them per-stratum for free,
and the later passes (validate → dimcheck → lint → **autodiff**) see only
ordinary transitions — so the generated rates are differentiated by the existing
machinery (§6). The synthesis mirrors stratification's compartment/transition
generation; it is new code modelled on existing patterns, not a literal call
into `consecutive` (a parser binder form) — only the `DstBranch` rate-weighting
(§4, hyper-Erlang) is reused as a code path.

**Naming.** Hidden compartments are `<TO>__<transition>[__<label>]__<i>`. The
`__` mangling is **newly reserved** (no compartment may contain `__` today —
verified empty across the corpus; the reservation is a new check). The pass
enforces both via-vs-user and **via-vs-via** uniqueness (two branches with the
same label is a hard error). **The entry transition keeps the original
transition name** so `incidence(NAME)` / `CumulativeFlow(NAME)` resolves to the
biological event (the entry flow); internal and exit transitions get mangled
names.

### Erlang — worked example

Source:

```camdl
infection : S --> I  @ beta * S * I / N  via erlang(stages = 3, mean = 7 'days)
```

Lowers to (stage rate `r = stages / mean = 3 / 7 'days`):

```camdl
infection          : S --> I__infection__1  @ beta * S * I / N     # entry: keeps the name, fires at FOI
infection__prog_1  : I__infection__1 --> I__infection__2  @ r * I__infection__1
infection__prog_2  : I__infection__2 --> I__infection__3  @ r * I__infection__2
infection__exit    : I__infection__3 --> I                @ r * I__infection__3
```

The three hidden compartments _are_ the latent class. The transit delay
(entering the pipeline to arriving at `I`) is Erlang(3, 3/τ) = mean 7 days; the
`S`-residence (time-to-infection) is governed by the force of infection, as in
any SEIR. With `rate = sigma` instead of `mean`, this is **isomorphic, modulo
hidden-compartment names, to the expanded `seir_erlang.camdl` golden**
(`hᵢ = Eᵢ`, stage rate `3·sigma`) — the equivalence T1 pins.

### Hyper-Erlang — worked example

Source (the last branch omits `weight` ⇒ `1 − p_fast`, normalized by
construction — see §5):

```camdl
infection : S --> I
  @ beta * S * I / N
  via hyper_erlang(
    branch(label = fast, weight = p_fast, stages = 2, mean = 5 'days),
    branch(label = long,                  stages = 1, mean = 60 'days)
  )
```

Each `branch(...)` is a self-contained record (the surface form of one
`hyper_branch`). Lowers to a **single** probability-weighted entry transition
named `infection` — a `DstBranch` (`S --> { … : w₁, … : w₂ } @ FOI`, which the
flow resolver aggregates as a `CumulativeFlowSum` over both destinations, so
`incidence(infection)` captures _both_ branches) — into parallel Erlang chains:

```camdl
infection : S --> { I__infection__fast__1 : p_fast, I__infection__long__1 : 1 - p_fast }  @ beta * S * I / N
infection__fast_1    : I__infection__fast__1 --> I__infection__fast__2  @ (2 / 5 'days) * I__infection__fast__1
infection__fast_exit : I__infection__fast__2 --> I                      @ (2 / 5 'days) * I__infection__fast__2
infection__long_exit : I__infection__long__1 --> I                      @ (1 / 60 'days) * I__infection__long__1
```

The latent period is a mixture: with probability `p_fast` an Erlang-2 of mean 5
days, otherwise an exponential of mean 60 days — a hyper-Erlang.

## 5. Semantics and resolved decisions

- **`@` is the entry force, not a progression rate.** Individuals leave `FROM`
  at the `@` propensity and begin the transit; the law governs only the transit
  delay. Put a force of infection / importation rate in `@`, never a
  per-stage/latent rate — the latter would double-count (an `Exp(@)` residence
  _plus_ the law's delay). The natural use is on an **event** transition
  (infection), where `@` is a real upstream force and the delay is the
  natural-history latent period that follows.
- **In-transit individuals are inert.** The hidden stages are not user-
  referenceable and appear in no rate — correct for a
  latent/incubation/reporting delay. A non-exponential _infectious_ period
  (active residence) is **not** expressible this way and is a follow-up (§7).
- **`stages = 1` is the exponential delay, not a no-op.** It lowers to one
  hidden compartment (`S --> h₁ @ FOI`, `h₁ --> I @ (1/mean)·h₁`) — the classic
  exponential latent period. It is _not_ byte-identical to the plain `S --> I`
  (which has zero transit delay).
- **Mean → stage rate is the compiler's job.** `erlang(stages = k, mean = τ)`
  emits stage rate `k/τ`; `erlang(stages = k, rate = ρ)` emits `k·ρ`. Exactly
  one of `mean`/`rate` (the `mean_spec` type enforces the XOR); both, or
  neither, is a compile error.
- **Hyper-Erlang weights: the last branch's weight is implicit** (`1 − Σ`
  others), so the mixture is **normalized by construction**. This makes a
  _fitted_ weight always valid (no compile-time sum-to-1 check, which is
  impossible for parameter-valued weights). A weight on the last branch, or
  weights that can't be normalized, is a compile error.
- **`incidence(NAME)` is the entry flow** (the biological event); the entry
  transition keeps the original name; hyper-Erlang aggregates over branches.
- **Hidden compartments init to 0** (the default for any compartment absent from
  `init {}`) — the correct cold-start for an in-transit pool.

## 6. Gradients, and what's estimable

Because the `via` pass lowers to ordinary transitions **before** the autodiff
pass, every generated rate is differentiated by the existing transition autodiff
with no special cases. A fitted `mean` appears as `k/mean` in the stage rates;
autodiff gives `∂(k/mean·X)/∂mean = −k/mean²·X` like any other division. A
fitted hyper-Erlang `weight` is a linear coefficient on the entry rates (and,
via `1 − p`, on the sibling branch with opposite sign), differentiated on both.
So, unlike the forcing lag (gh#314, whose gradient flows through a forcing's
time argument), delay laws create **no** autodiff gap — they never leave the
mass-action world.

**What's estimable:**

| Parameter             | Fittable? | Why                                                                                    |
| --------------------- | --------- | -------------------------------------------------------------------------------------- |
| `mean` (the period)   | **Yes**   | Lands in `k/mean`; real gradient under PGAS+NUTS and gradient-free methods.            |
| `rate` (= 1/mean)     | **Yes**   | Lands as `k·rate`.                                                                     |
| hyper-Erlang `weight` | **Yes**   | Linear on entry rates; gradients on both branches (last-implicit keeps it normalized). |
| `stages` (k)          | **No**    | Sets how many compartments exist — model _structure_, not a continuous knob.           |

So a Garki-style fit estimates the mean incubation period and the fast/slow
mixture weight (all gradient-correct), and _fixes_ the stage counts — which is
the standard treatment (the stage number is the shape, chosen from prior
knowledge or selected by discrete comparison over a few `k`; Wearing–Rohani–
Keeling 2005). The one thing not fittable in v1 is a **continuous shape** (a
non-integer gamma shape), which an integer stage chain cannot represent and
which has no smooth gradient w.r.t. compartment count — that is exactly what the
deferred `approx_gamma` / `coxian` would unlock.

## 7. Interactions and scope boundaries

- **`balance {}` is a hard error with `via`.** A user-written conservation
  budget (`R = N − S − E − I`) cannot see the hidden in-transit compartments, so
  the slack compartment would be silently under-counted by the in-transit
  population. The compiler rejects `via` + `balance` with a diagnostic naming
  the transition and the budget. (Events / interventions resolve compartments by
  explicit name, so they neither hit nor need the hidden pool — those
  interactions are safe.)
- **Active residences (infectious period) — the visibility boundary.** As §1
  states, the hidden pipeline only models _inert_ delays. A non-exponential
  infectious period (the stages must transmit, i.e. feed `beta·S·I/N`) needs
  **visible** sub-staging — the stages stay referenceable and the `FROM` name
  resolves to their sum in rate expressions. This is a clean follow-up; it
  shares the chain machinery but not the hiding. Until then, the manual
  `consecutive` form expresses it.
- **Competing hazards during the delay.** Even for inert delays, the hidden
  stages cannot carry a competing exit (e.g. mortality during incubation, the
  DMT `(1−δ)^N` survival). `via` expresses a single-exit delay; a competing
  hazard mid-pipeline awaits §7.1.
- **Backends.** The lowered form is ordinary transitions, so all three backends
  (Gillespie, chain-binomial, ODE) and the autodiff are untouched.
- **Run identity.** The hashed IR is the post-lowering structure, so a `via`
  model re-keys automatically — no machinery needed.
- **Deferred laws.** `coxian` and `approx_gamma` are named follow-ups.

### 7.1 Sketched extension — multi-exit (competing hazard during the delay)

A later extension lets delayed individuals also leave to a competing destination
mid-chain. Sketched now so the v1 AST reserves the `competing` slot. Two shapes:

A **`during { … }` clause** for a uniform per-stage hazard (the common case):

```camdl
incubation : E --> I
  @ sigma * E
  via erlang(stages = 15, mean = 15 'days)
  during { death : --> D @ mu }     # per-capita hazard mu from every hidden stage to D
```

An **addressable delay process** when the hazard is stage-specific:

```camdl
delay incubation : E ~> I  via erlang(stages = 15, mean = 15 'days)
transitions {
  death_in_incubation : incubation[s] --> D  @ mu[s] * incubation[s]
}
```

Two semantics to settle when built: the competing exit takes a **per-capita
hazard** (the hidden stages can't be referenced) — a mild inconsistency with the
entry `@` (a propensity) that the addressable form removes; and under leakage
the survivors' arrival-time distribution is the Erlang _conditioned on no
competing exit_ (a defective distribution), so `mean` no longer equals the
survivors' mean arrival. Exact Dirac-plus-daily-survival `(1−δ)^N` additionally
wants the deferred conveyor.

## 8. Testing strategy

The feature is a desugar to ordinary transitions, so correctness spans three
layers: the desugar is _structurally right_ (IR), the _dynamics are the intended
distribution_ (statistical known-answer), and it composes across the _backend ×
method matrix_ and the _gradient_ path. Each phase (§9) lands with its tier
green.

### T1 — Lowering correctness (IR-level, OCaml expander)

- **Anchor equivalence.** `via erlang(stages = 3, rate = sigma)` on
  `S --> I @ FOI` must lower to a compartment+transition set equal — modulo
  generated names — to the latent sub-structure of `seir_erlang.camdl`
  (`hᵢ = Eᵢ`): identical stoichiometry and identical rate ASTs after
  constant-fold, asserted **at the AST level** so it cannot pass vacuously.
- **Stage-rate.** `mean = τ` ⇒ internal rate `stages/τ`; `rate = ρ` ⇒ `stages·ρ`
  (assert the emitted AST).
- **Entry-name preservation**, including a **hyper-Erlang case asserting
  `incidence(NAME)` sums all branch entries** (the DstBranch aggregation — the
  test that would catch a dropped branch).
- **Naming + collision**: hidden names follow the scheme; via-vs-user _and_
  via-vs-via (same label) collisions are hard errors.
- **`stages = 1`** lowers to one hidden compartment (the exponential delay) —
  and is asserted to differ from the plain transition.
- **Init**: hidden compartments default to 0.
- **Golden IR fixtures** for one `erlang` and one `hyper_erlang` model,
  committed and human-reviewed.

### T2 — Distributional correctness (known-answer — the scientific anchor)

The point of the feature is the _shape_, so the highest-value check is the
sojourn against the analytic law. The **deterministic ODE = analytic
gamma-kernel** check is the **primary** anchor (no sampling noise): under the
ODE backend the chain's `TO`-arrival impulse response equals the closed-form
gamma density `g(t; k, k/τ)` within integrator tolerance. The stochastic check
is secondary: seed a pulse cohort, no inflow, record arrival times; the
empirical sojourn matches Erlang(`k`, `k/τ`) — mean `τ`, variance `τ²/k`, CV
`1/√k`, and the CDF within a stated band (committed `n`, seed, and the exact
reference call — `scipy.stats.gamma(a = k, scale = τ/k)` / R
`pgamma(x, shape = k, rate = k/τ)`, noting the `scale = τ/k` footgun).
Hyper-Erlang: the mixture matches `Σ wᵢ·Erlang(kᵢ, kᵢ/τᵢ)` in mean, variance,
and survival function. Tolerances are sized to Monte-Carlo SE and committed —
never silently loosened.

**Per-law DSL-to-distribution fixtures (expensive tier).** Every law gets a
fixture under `tests/external/` that compiles real DSL → simulates → checks the
realized sojourn distribution against the analytic law, with committed reference
values. These run in CI (and `make test`) but are skipped by the
`make test-fast` inner loop — the same tier as the `he2010` pomp-oracle and the
sparse-oracle. No law ships without one.

### T3 — Backend × method coverage (no silent matrix gap)

- The same `via` model runs on **all three** forward backends; the stochastic
  mean converges to the ODE field.
- It runs under **each** inference method the model admits (PF, IF2, PGAS, PMMH)
  — the lowered transitions are ordinary, so every applicable cell must run, not
  be skipped: assert a loglik or a short fit per cell, no capability rejection.

### T4 — Gradient correctness (the "for free" claim)

- A fitted `mean` **and** a hyper-Erlang `weight` each yield a non-empty
  `rate_grad` on the generated transitions.
- **Finite-difference check**: the emitted `∂rate/∂mean` matches a central
  finite difference at sampled states/params.
- **Recovery**: simulate with a known `mean` _and_ a known `weight`, fit with
  IF2 and PGAS+NUTS, recover within the interval — gradients present _and_
  correct, on both the `mean` path and the dual-branch `weight` path.

### T5 — Validation & error quality

- Dimcheck: non-duration `mean`, non-rate `rate`, non-positive-integer `stages`
  each → a distinct E-code naming the transition.
- `mean` XOR `rate`: both → error; neither → error.
- `via` + `balance` → hard error (§7).
- Hyper-Erlang weights: a weight on the **last** branch → error; a
  parameter-valued non-last weight is accepted (normalized by construction).
- Each asserted via the JSON-diagnostics test path (code + transition name).

### T6 — Parser, round-trip, regression, and the realistic shapes

- Inline and block `via` parse; the AST round-trips; IR round-trips.
- **Stratified `via`**: a `via` on an age-stratified `S[a] --> I[a]` yields
  `k × n_strata` hidden compartments with correct per-stratum rates and unique
  names, and `incidence` sums over strata without double-counting branches.
  (Every realistic model is stratified — this is not optional.)
- **Observation-scored `via` flow**: score `cases ~ incidence(infection)`
  against a known truth — the test that exercises the full entry-flow
  resolution.
- **Multi-`via` model**: two `via` transitions in one model (no cross-chain name
  collision, independent rates).
- **Golden-neutrality**: every existing model without `via` compiles to
  byte-identical IR.
- **Doc-tests**: the spec §5 `via` snippets compile under the doctest harness.

## 9. Phasing and implementation plan

Each phase lands with its named test tier green before the next.

1. **AST + parser.** Add `trvia : via_call option` (a raw `(name, kwargs)`) and
   the reserved `trcompeting` slot; the inline `via LAW(...)` clause and the
   block `via = LAW(...)` entry; the `via` keyword. The parser stores the raw
   `EFuncCall`-shaped call — it does **not** build the typed `via_spec`. →
   **T6** (parse, round-trip, golden-neutrality).
2. **Erlang lowering.** The AST→AST pre-pass (before stratification /
   `check_declaration_names`): parse the raw call into a typed `via_spec` with
   `stages : pos_int` extraction, `mean` XOR `rate`, and reserved-name +
   via-vs-via collision checks; generate the hidden compartments + chain
   transitions + entry/exit; scale the stage rate; preserve the entry name. →
   **T1**, then **T2/T3** and **T4**.
3. **Dimcheck + validation.** `mean` `[T]`, `rate` `[1/T]`, `stages` positive
   integer; `mean` XOR `rate`; `via` + `balance` hard error. Errors name the
   transition. → **T5**.
4. **Hyper-Erlang.** Single `DstBranch` weighted entry into parallel chains;
   `branch(...)` labels into the naming; last-weight-implicit normalization;
   weight validation. → **T1/T2/T5** repeated for the mixture (incl. the
   incidence-sums-branches and parameter-weight cases).
5. **Docs.** Spec §5 prose + the `via` examples (the **T6** doc-tests);
   `docs/dsl-cheatsheet.md`.

Deferred follow-ups (named, not open questions): `coxian`; `approx_gamma`
(continuous-shape fitting); a discrete conveyor for exact fixed delays;
**visible sub-staging for active residences** (non-exponential infectious
period); the multi-exit / competing-hazard extension (§7.1); exit-flow (arrival)
observation access.
