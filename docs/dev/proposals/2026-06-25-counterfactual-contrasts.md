# Counterfactual contrasts (cases averted)

Status: **v1 implemented** (gh#322). The infrastructure prerequisites — the
joint keyed `(θ, X)` output (#1), the start-from-state engine seam (#2), and the
`LatentPath` fork-validity classifier (#4) — landed, as did the DSL
`contrasts {}` surface + IR `Contrast` node, the stored quantity dimension (#5),
and the two-arm replay reducer (#3) auto-emitted on `fit predict`. v1 covers
**chain_binomial** fits with **state-sourced** operands (the
cases/deaths-averted headline). Two parts are loud-deferred with tracked
follow-ups: ODE/deterministic forks (**gh#325**) and observation-sourced
operands (**gh#326**) — see "v1 implementation status" below. (Generated
quantities, a prerequisite, shipped via `2026-06-25-generated-quantities.md`.)

Splits the counterfactual half out of
`2026-06-24-generated-quantities-and-counterfactuals.md` (superseded).
Supersedes `2026-06-04-experiment-compare.md`.

## What this delivers

The headline policy number — **cases averted**: how many cases an intervention
(an SIA) prevented, with full posterior uncertainty, as a banded
`contrasts/<name>.tsv`.

## Why not a forward-only version (the deferral rationale)

A counterfactual can be posed two ways, and they answer different questions:

- **Free-forward (prospective):** replay the posterior _parameters_ from `t0`
  under both arms. Answers "in outbreaks like the ones we inferred, how many
  cases would the SIA avert _on average_." A population-design object.
- **Conditioned (retrospective):** condition on the _observed_ outbreak — branch
  both arms from the fit's inferred latent state at the intervention time.
  Answers "for _this_ outbreak we saw, how many did the SIA avert."

"Cases averted" almost always means the **retrospective** question, and a user
who runs a forward-only `contrasts` and reads the prospective number as the
realized one has a silent-wrong on a policy headline. So this surface ships
**only** with the conditioned fork; a forward-only `contrasts` block is not an
acceptable interim. (The prospective object is recoverable via plain
`simulate --draws` under two scenarios for users who genuinely want it; it is
not dressed up as "cases averted".)

## The design (converged)

The `contrasts {}` block is **forward simulation that reads a fit's output** —
it never invokes the inference kernels. Per posterior draw `i`:

```
read θ_i  and  the fit's inferred latent state X_i(T*) at the fork T*
arm A (factual):        forward-sim from X_i(T*), no intervention,  seed s_i
arm B (counterfactual): forward-sim from X_i(T*), with intervention, seed s_i   ← CRN
averted_i = quantity(arm A) − quantity(arm B)
band averted over draws
```

Both arms branch from the **same** `X_i(T*)`, so there is **zero pre-fork
variance**; the band over `averted_i` carries the **joint (θ, X) posterior**
(parameter _and_ latent uncertainty), since a stochastic-fit draw is the joint
`(θ_i, X_i)`.

**CRN, stated honestly.** Sharing the seed makes the firing substep
byte-identical across arms (the SIA `transfer` is RNG-free). After the transfer
changes the compartment counts, the two streams desynchronize (the backend
consumes a rate-dependent number of RNG words), so the post-fork coupling is
correlated and **decays** — CRN does _not_ cancel the post-fork forward noise.
The real variance-reduction win is the shared `X_i(T*)` (eliminating latent
variance at the fork), not forward-noise cancellation.

### Why the smoothed `X_i(T*)`, not a filtered one

`X_i(T*)` is the fit's **smoothed** latent state — conditioned on the _whole_
observed series `y_{1:T}`, including data _after_ the fork (PGAS samples the
joint smoothing distribution `p(θ, X | y_{1:T})`). Using post-`T*` data to set
the state we fork from looks suspect, so it is worth saying why it is correct —
and when it would not be.

This surface ships the **retrospective** estimand: "for _this_ outbreak we
actually observed, how many cases did the SIA avert?" That is a
**counterfactual** query, and a counterfactual is _defined_ by conditioning on
the factual evidence (Pearl's abduction → action → prediction: condition the
latent variables on _all_ the evidence, then `do(no SIA)`, then predict).
Conditioning the latent state on all of `y_{1:T}` **is** smoothing. The
post-`T*` data is not the future leaking into the counterfactual: `X_i(T*)` is a
**pre-fork, shared** quantity — both arms are identical up to `T*` and diverge
only when the SIA is applied _after_ the fork — so the later observations
sharpen our estimate of the one true state that existed at `T*`, _through the
known factual forward model_ (`X(T*) → SIA →
dynamics → y_{>T*}`). A fast
post-`T*` decline tells us how many were really infected at `T*`; the smoother
uses it, a filter (state given `y_{1:T*}` only) throws it away.

Filtering would be correct for a **different** estimand — the _prospective_,
decision-time question "with only what was knowable at `T*`, how much would an
SIA be expected to avert?" — which deliberately ignores hindsight. We do not
ship that one (a user reading the prospective number as the realized one is the
silent-wrong this proposal exists to avoid). So: retrospective ⇒ smoothing (read
`X` from the fit's smoothed trajectory, never a re-run filter).

### Validity per inference method — classified by `LatentPath`

The conditioned fork is valid only for methods that produced a conditionable
latent state, and that differs by method. Fork-validity is classified by the
3-state `LatentPath` ADT from
[`2026-06-28-keyed-joint-param-trajectory-output.md`](2026-06-28-keyed-joint-param-trajectory-output.md)
§3 — `Deterministic` (ODE, valid: recompute X from θ) / `Sampled` (a stored
path, valid) / `NotSaved` (a stochastic fit with no saved path, rejected) — and
a point-estimate fit (no posterior cloud) rejects outright. This keys on the
latent **artifact**, not the method name. It does **not** extend the
particle-filter-drive `FilterableFit` witness (`predict.rs:197`): that witness
_rejects_ ODE as `NotFilterable::Deterministic`, the exact opposite of the
fork's verdict (ODE is the easy, valid case), so reusing it would overload one
enum arm with two opposite meanings. The validity table:

| method    | backend        | latent state                 | conditioned fork                                    |
| --------- | -------------- | ---------------------------- | --------------------------------------------------- |
| PGAS      | chain_binomial | smoothed path per draw       | valid (CRN)                                         |
| PMMH / PF | chain_binomial | path _if saved_ (today: not) | valid once saved (CRN)                              |
| IF2       | chain_binomial | none (MLE point)             | rejected — `PointEstimate`                          |
| MH        | ODE            | θ-determined                 | valid, deterministic (no CRN); v1-deferred → gh#325 |
| **NLopt** | ODE            | θ-determined **point**       | **rejected — `PointEstimate`**                      |

NLopt on ODE is a point estimate (no posterior), so it rejects exactly like IF2
— the classifier keys on posterior-vs-point (the artifact), not on
`backend == Ode`. PMMH/PF are `Sampled` once their fits save a latent path, and
`NotSaved` (rejected) until then.

### Surface

```
observations { reported { columns { time : time, cases : count }
                          projected = incidence(infection)
                          cases ~ neg_binomial(mean = rho * projected, r = k) } }
interventions { sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 20 'weeks] }
scenarios     { no_sia { disable = [sia] }   with_sia { enable = [sia] } }
quantities    { total_deaths = final(D)                       # scalar  (time collapsed)
                prevalence   = I / N                          # series  (no temporal reduce)
                cum_reported = integral(observations.reported) }  # scalar  (series reduced)

contrasts {
  # scalar — total deaths averted over the window
  deaths_averted   = no_sia.quantities.total_deaths - with_sia.quantities.total_deaths   over [origin + 20 'weeks, origin + 52 'weeks]
  # series — the averted reported *curve* (a raw obs series; no reducer needed)
  reported_averted = no_sia.observations.reported   - with_sia.observations.reported     over [origin + 20 'weeks, origin + 52 'weeks]
  # scalar from a series — reduce inside a named quantity, then contrast it
  cases_averted    = no_sia.quantities.cum_reported - with_sia.quantities.cum_reported   over [origin + 20 'weeks, origin + 52 'weeks]
}
```

A contrast operand is a run-rooted reference (`<scenario>.quantities.<q>` or
`<scenario>.observations.<stream>`) combined by arithmetic. **Reductions live in
`quantities {}`, not inside the contrast** (v1 takes no inline reducer in a
contrast expression): a series operand contrasts directly to an averted series,
and to collapse it to a scalar you name a quantity (`cum_reported` above) —
which is exactly what the `quantities {}` block is for.

Each contrast bands over the **forkable** posterior subset (the joined count
surfaced, per the `(θ, X)` partial-join contract) and is emitted as a tidy/long
`contrasts/<name>.tsv` keyed by whatever axes the operands carry (`stratum`,
`time`) with `q05…q95 / mean / n_forkable` columns.

**Invocation: auto-emitted on `fit predict`.** A contrast has no verb or flag of
its own. When the model declares a `contrasts {}` block,
`camdl fit predict <fit>` computes each contrast against that fit and writes
`contrasts/<name>.tsv` under the predict output directory, alongside the
predictive / quantities / observed sidecars. The fit handle is the predict
invocation argument, so `fitted` resolves to the no-overlay run of that one fit
(the fit's identity is the output path). A fit with no forkable draws (no saved
latent paths) or a deterministic backend emits no file and a located note rather
than a band; a point-estimate fit is already refused by `fit predict` before any
output.

**v1 implementation status.** Shipped now: **chain_binomial** fits with
**state-sourced** operands — the `deaths_averted` line above (the CRN headline).
Two parts of this surface are landed-but-loud-deferred, each with a tracked
follow-up:

- **ODE / deterministic forks** (`MH` on ODE) → **gh#325**. The fork is
  well-defined (recompute `X(T*)` from θ, two-leg integration) but needs the
  start-from-state seam extended to `run_ode`; until then an ODE fit emits the
  located note above, not a band.
- **Observation-sourced operands** (`reported_averted`, `cases_averted` above,
  i.e. anything reading `run.observations.<stream>`) → **gh#326**. Blocked on
  the obs-time axis over a fork window (the deferred `last_obs`/`first_obs`
  time-source); a contrast referencing them is skipped with a note while the
  model's state-sourced contrasts still emit.

**The namespace is the run, with two uniform sub-namespaces — `quantities` and
`observations`.** Dot is one operator — "member of a run" — and a run member is
always `<run>.<quantities|observations>.<name>`. The two sub-namespaces are
symmetric: neither is special-cased, and a quantity named `observations` (or a
stream named `deaths`) can never collide with the other namespace.

- In a `contrasts {}` expression the run is _explicit_ (a scenario name):
  `no_sia.quantities.total_deaths` is the `total_deaths` quantity on the no_sia
  run; `no_sia.observations.reported` is the no_sia run's simulated `reported`
  series.
- In a `quantities {}` recipe the run is _implicit_ (the recipe applies to
  whatever run evaluates it), so the run prefix drops: `observations.reported`
  is _this run's_ series, and a bare compartment (`D` in `final(D)`) is _this
  run's_ state. `observations.reported` is exactly
  `<this run>.observations.reported` with the run elided.

(`DOT` already exists — `lexer.mll:217` — and `observations.<stream>` already
uses it via `OBSERVATIONS DOT IDENT`, `parser.mly:1156`; this adds the
run-prefixed forms `<scenario> DOT observations DOT <stream>` and
`<scenario> DOT
quantities DOT <quantity>`, so it is a grammar addition, not a
new token. `.5`/`1.5` stay floats by maximal munch, so `no_sia.quantities` lexes
`IDENT DOT QUANTITIES` while a stray `no_sia.5` lexes `IDENT FLOAT`.)

- The block is named **`contrasts {}`**, not `compare {}` — `camdl compare` is
  already a CLI subcommand (model Δelpd comparison, `compare.rs`); a model block
  and a CLI verb do not collide at parse time, but `contrasts` is the precise
  word and avoids the conceptual overload.
- The window's `from` instant is the fork; endpoints are **instants** in the
  existing typed-time system (`origin + 20 'weeks`, `date(...)`; verified to
  parse). A new **endpoint type check** rejects a bare duration (today
  `at [20 'weeks]` compiles silently — this is new code, reconciled with the
  same `at` loophole).
- Each entry is `name = <contrast_expr> over [<instant>, <instant>]`, with
  `over` a new keyword binding looser than arithmetic — a real grammar
  production with explicit precedence (`over` below the additive/subtractive
  operators, so `a - b over [..]` parses as `(a - b) over [..]`), not an
  asserted precedence.
- Two cross-context diagnostics (the error-quality bar): `observations.x` used
  as a _contrast operand_ (it is a run sub-namespace, not a scenario), and
  `<scenario>.x` used _inside a quantity recipe_ (the run is implicit there),
  each fail with a located message naming the rule and the fix — not a bare
  syntax error.

### Shape model — contrasts inherit operand shape

A quantity value already lives on `{time?} × {strata}`: `ir/src/quantity.rs`
carries `reduce: Option<TemporalReduce>` (`None` ⇒ a series, one value per
output snapshot; `Some` ⇒ time collapsed to a scalar) and
`stratum: Vec<StratumKey>` (the IR is fully expanded — one leaf per cell). A
contrast is **elementwise `armA − armB`, shape-preserving**, banded per draw
over the forkable subset:

| operand shape   | example                      | contrast result               |
| --------------- | ---------------------------- | ----------------------------- |
| scalar          | `final(D)`, unstratified     | one banded number             |
| series (time)   | `I / N` (no reduce)          | averted **over time** — curve |
| vector (strata) | `final(D)`, stratified       | averted **by region**         |
| time × strata   | incidence series, stratified | banded per `(region, time)`   |

Averted-over-time and averted-by-region therefore need **no special syntax** —
they fall out of contrasting a series / stratified quantity. Two checks guard
the arithmetic:

- **Shape agreement** — operands must share axes; `series − scalar`, or two
  differently-stratified operands, is a located error naming both shapes.
- **Dimension agreement** (prerequisite #5) — `deaths − rate` is rejected,
  naming both dimensions.

`over [window]` is orthogonal to shape: it clips the time axis of a series
operand and scopes the reduction window of a reduced one.

**Deferred (the only stratification deferral):** sub-cell / sub-time _selection_
— `no_sia.quantities.deaths[region = north]`, a `[..]` index picking one cell or
instant. A refinement on top of shape inheritance; _contrasting_ stratified or
series quantities (whole-vector results) is in v1.

### Parameters in a contrast arm

Each arm is a scenario applied to one posterior draw, then forked from `X_i(T*)`
— so parameter resolution is the **existing 5-tier chain**
(`params_resolver.rs`; `docs/camdl-run-spec.md §1.3`), no new mechanism. The
fitted draw and the scenario sit at adjacent tiers:

- **Tier 3.5** — the fitted draw θ_i (from `draws.tsv`).
- **Tier 4** — the scenario: `set` _overrides_ the draw (absolute), `scale`
  _multiplies_ it (`params_resolver.rs:99`).

So for a fitted parameter there are two distinct, well-defined counterfactuals:

- **`scale = { beta = 0.5 }`** → per draw, `θ_i.beta × 0.5` — **preserves the
  posterior** (every draw perturbed, uncertainty carried through). The clean way
  to compare the fit against a perturbed-parameter scenario.
- **`set = { beta = 0.2 }`** → forces an absolute value — `do(β = 0.2)`,
  **collapsing that parameter's posterior to a point**. Valid, but the fitted
  uncertainty on β is discarded.

The no-overlay arm — the fitted model with no scenario patch — is **`fitted`**,
the reserved no-overlay sentinel `camdl fit predict` already emits in its
`scenario` column (a `scenarios {}` preset named `fitted` is rejected, E291). A
contrast references it directly; it is not declared:

```
scenarios { lower_trans { scale = { beta = 0.5 } } }   # only the counterfactual is declared
contrasts {
  averted = fitted.quantities.cases - lower_trans.quantities.cases  over [...]
}
```

`fitted` is **fit-relative** — it means the no-overlay run of the one fit this
contrast is computed against (the fit handle is the invocation argument; an
ambiguous handle is already a hard error). With several fits of a model (PMMH,
PGAS) you run the contrast against each handle; the fit's identity is the output
path, not the `scenario` column. Comparing two fits' inferences is a distinct
object — there is no shared `X(T*)` across fits to fork from — and is out of
scope for a conditioned contrast.

**Post-fork only.** A scenario's parameter override acts on the **forward
dynamics from T\***; it does **not** re-infer `X_i(T*)`, the factual shared
anchor both arms branch from. So a contrast expresses post-fork parameter
counterfactuals ("transmission halved _after_ the SIA"). A _from-t₀_ parameter
counterfactual ("β had always been lower," which would change the pre-fork
epidemic too) is the **prospective** question — `simulate --draws` under two
scenarios, not a conditioned contrast (where the deferral rationale above
already sends it). Forcing a pre-fork change onto a frozen `X_i(T*)` would be
silently incoherent, so it is rejected, not approximated.

## Prerequisites

Three of the five are built (gh#322); the remaining two — the `contrasts {}`
surface + reducer (#3) and the stored quantity dimension (#5) — are this
proposal's build.

1. **[done] Joint, keyed `(θ, X)` fit output** — specced in
   [`2026-06-28-keyed-joint-param-trajectory-output.md`](2026-06-28-keyed-joint-param-trajectory-output.md),
   landed. `draws.tsv` carries a leading `(chain, draw)` key inner-joined to the
   **path-saved subset** of `trajectories.tsv` (a partial join, with the
   forkable count surfaced via `fit::joint::classify_joint`); the shared loader
   strips the key so every param-only reader is unchanged. A PMMH/PF latent-path
   writer and full-coverage join are deferred follow-ups there.
2. **[done] A start-from-state engine seam** — landed. `chain_binomial`'s
   `run_chain_binomial_with_observer` resumes from an injected `X(T*)` at
   `cfg.t_start = T*` via `Resume{ start: Some(StartState{..}) }`: state inject,
   output-cursor re-seat, fresh/restored RNG, with reactive/off-grid rejections.
   Validated by the splice invariant. Reads `io::trajectories::read_state_at`
   for a `Sampled` path; an ODE arm recomputes `X(T*)` from θ.
3. **[build] The `contrasts {}` surface + two-arm replay reducer.** The DSL
   block (run-rooted DOT member-access, `over` keyword, endpoint type check),
   then the Rust reducer: for each forkable draw, replay arm A and arm B from
   `X_i(T*)` via the engine seam, evaluate the operand quantities on each arm,
   and difference them **elementwise, preserving shape** (scalar / series /
   stratified / time × strata — see the shape model), banding over the forkable
   subset. Today `fit predict` builds one inline baseline (`predict.rs:860`);
   the paired two-arm replay + shape-preserving differencing + per-draw band are
   net-new.
4. **[done] Fork-validity classifier** — the `LatentPath` ADT
   (`Deterministic | Sampled | NotSaved`) landed in prerequisite #1's
   `fit::joint`, with a point-estimate (no-posterior) rejection. NOT an
   extension of `FilterableFit` (the PF-drive witness, which rejects ODE — see
   "Validity per inference method" above).
5. **[build] A stored quantity dimension** for the contrast binop-agreement
   check (`no_sia.quantities.deaths - with_sia.quantities.deaths` requires equal
   dims) — `dimcheck` does not persist computed dimensions today
   (`dimcheck.ml`). Owned here; see the "IR-side stored quantity dimension"
   sketch below. (The companion **shape agreement** check — operands must share
   `{time?} × {strata}` axes — is a Rust check in the reducer over the evaluated
   operand shapes, needing no IR change.)

### IR-side stored quantity dimension (prerequisite #5)

The `contrasts {}` binop `no_sia.quantities.deaths - with_sia.quantities.deaths`
is an arithmetic combination of two quantity values, so the dimensional checker
must verify the two operands agree (both `deaths`, a count) — otherwise a
`deaths - rate` contrast either silently produces a meaningless number or fails
opaquely. `dimcheck.ml` checks dimensions during expansion but does not
**persist** the computed dimension of a declared `quantities {}` entry into the
IR. This is a small OCaml/IR-side add: carry each quantity's resolved dimension
on its IR node (an `ir/schema.json` field on the quantity, mirrored OCaml↔Rust),
so the Rust `contrasts {}` reducer can check operand-dimension agreement (E-code
on mismatch, naming both quantities and their dimensions) before differencing.
No new unit literals or DSL surface — purely persisting a dimension `dimcheck`
already computes. (Sized as a follow-up alongside the contrast reducer, not a
blocker for prerequisites #1–#2.)

## Deferred to a follow-up (explicitly out of v1 scope)

These are named non-goals for v1, not unresolved design questions. v1 ships
shape-polymorphic contrasts (scalar / series / stratified / time × strata, by
shape inheritance) on **chain_binomial** fits with **state-sourced** operands;
the deferrals below each have a tracked issue:

- **ODE / deterministic forks** — **gh#325**. Recompute `X(T*)` from θ via a
  start-from-state seam on `run_ode` (two-leg factual/arm integration); v1 forks
  chain_binomial only.
- **Observation-sourced operands** (`run.observations.<stream>`) — **gh#326**.
  Blocked on the obs-time axis over a fork window; v1 operands are state-sourced
  quantities.
- Sub-cell / sub-time **selection** —
  `no_sia.quantities.deaths[region = north]`, a `[..]` index picking one cell or
  instant. _Contrasting_ stratified / series quantities (whole-vector results)
  is in v1; only picking a sub-element is deferred.
- **Inline reducers in a contrast expression**
  (`integral(no_sia.observations.reported)`). v1 reduces in `quantities {}` and
  references the named quantity; a series operand still contrasts directly to an
  averted series. Adding reducer calls to the contrast-expr grammar is a later
  convenience, not a v1 need.
- Decoupling the conditioning instant from the accumulation window.
- `last_obs`/`first_obs` as named instants — define the resolver's time source
  for multi-stream (ragged) models.
- Reconcile the `free_forward` naming with the existing `Horizon::FreeForward`
  predict axis if a prospective surface is ever added.

## Decisions recorded

- Ship only the conditioned fork; no forward-only `contrasts` (misread risk).
- The `contrasts {}` block is forward sim that _reads_ the fit's latent `X(T*)`;
  no re-filter.
- **The namespace root is the run, with two symmetric sub-namespaces**
  (`quantities`, `observations`). A run member is
  `<run>.<quantities|observations>.<name>`: explicit run (scenario) in a
  `contrasts {}` expression (`no_sia.quantities.deaths`,
  `no_sia.observations.cases`), implicit run in a `quantities {}` recipe
  (`observations.cases`, the prefix elided). Neither sub-namespace is
  special-cased; quantity/stream names cannot collide across them. (`DOT`,
  `QUANTITIES`, `OBSERVATIONS` are existing tokens; this adds productions, not
  tokens.)
- **Contrasts are shape-polymorphic — they inherit the operand's shape** (scalar
  / series / stratified / time × strata) and difference elementwise, banding per
  draw. Averted-over-time and averted-by-region need no special syntax. Guarded
  by a shape-agreement check (Rust, in the reducer) and a dimension-agreement
  check (#5, IR-persisted). Only sub-element _selection_ (`[p]`) is deferred.
- Validity gated by the `LatentPath` ADT (`Deterministic | Sampled | NotSaved`),
  keyed on the latent **artifact** (posterior-vs-point), not the method name or
  the backend axis — explicitly NOT an extension of `FilterableFit` (which
  rejects ODE, the opposite verdict; see "Validity per inference method").
- CRN's win is the shared `X(T*)`, not post-fork noise cancellation (stated
  honestly).
- The infrastructure prerequisites (joint keyed `(θ, X)` output #1,
  start-from-state seam #2, `LatentPath` classifier #4) are named, not buried,
  and are now built (gh#322); this proposal builds the surface + reducer (#3)
  and the stored quantity dimension (#5).
