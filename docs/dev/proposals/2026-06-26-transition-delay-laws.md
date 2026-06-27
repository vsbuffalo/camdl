# Staged residences: non-exponential dwell times via compartment sub-staging

Date: 2026-06-26 Status: Draft Relates: gh#313 (non-exponential sojourns),
gh#314 (forcing lag — landed) Supersedes: the earlier hidden-pipeline framing of
this proposal

## 1. The problem

A compartment's residence time in camdl is exponential: while in `A`, an
individual leaves at the total exit hazard — memoryless, with coefficient of
variation one. Real infectious-disease dwell times are not. The number of
"stages" in a latent or infectious period materially changes epidemic speed,
peak timing, control thresholds, and the inferred reproduction number from the
same data (Wearing, Rohani & Keeling 2005, _PLoS Med_ 2:e174; Lloyd 2001, _Proc.
R. Soc. B_ 268:985).

The four diseases this is built for each break the exponential assumption, in
ways that map onto a small set of distributions:

| Disease             | Dwell time                                                      | Shape                                      | Law                                                                         |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------- |
| Malaria (Garki/DMT) | human incubation, liver→patency (~fixed 15 d)                   | deterministic                              | `fixed`                                                                     |
| Malaria             | gametocyte / infectious period                                  | peaked                                     | `erlang`                                                                    |
| Polio               | incubation → paralysis onset                                    | gamma                                      | `erlang`                                                                    |
| Polio               | shedding: weeks (typical) vs years (immunodeficient iVDPV)      | bimodal, one endpoint                      | `hyper_erlang`                                                              |
| Ebola               | incubation (mean ~9–11 d; WHO Ebola Response Team 2014, _NEJM_) | gamma                                      | `erlang`                                                                    |
| Ebola               | onset → death (~8 d) vs recovery (~12 d), split by CFR          | mixture, _different durations + endpoints_ | `hyper_erlang` (per-branch destination)                                     |
| TB                  | latent → fast primary progression vs slow lifelong reactivation | mixture + a low-rate arm                   | `hyper_erlang` (partial; reactivation arm is a competing exit, deferred §7) |

camdl can already express the unimodal case manually — the committed golden
`ocaml/golden/seir_erlang.camdl` builds an Erlang-3 latent period by sub-staging
`E` with `consecutive`:

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

Three costs: the per-stage rate `3 * sigma` is a **silent-wrong footgun** (write
`sigma` and the mean is wrong by `k`, no error); the verbosity scales with `k`;
and the modelling intent — "the latent period is Erlang-3, mean 7 days" — is
buried.

We want one clause that says that, and lowers to exactly this chain.

## 2. The object: a staged residence

A non-exponential dwell time is a property of a **compartment** — how long an
individual stays in it — not of an edge. The realization is the method of stages
(the linear chain trick): replace the compartment with `k` internal sub-stages,
each an exponential of rate `k/τ`, so the total dwell is Erlang(`k`, `k/τ`) with
mean `τ`, variance `τ²/k`, CV `1/√k`.

Critically, **the sub-stages are visible** — they are an ordinary stratification
of the compartment. So a bare reference to the compartment in any rate
automatically sums over its stages, exactly as a bare age-stratified name sums
over age today (`R[a]` with strata `[age, immunity]` already means
`R[a,natural] + R[a,immunity]`; `expander.ml:2560`, spec §5.1). This is the
whole game: because the stages are a sub-stratification, the force of infection,
population total, and observations all see the right thing for free.

This visibility distinction is the load-bearing one. A _latent_ period and an
_infectious_ period are the **same** staged construction (Wearing-Rohani-Keeling
treat them symmetrically); they differ only in whether the staged compartment
appears in the force of infection — which falls out of whether the modeller
references it, not from a second kind of object. Latent stages aren't in the
FOI; infectious stages are. One primitive, both cases.

## 3. Design — the `via` clause

A transition that **drains** a compartment carries an optional dwell law via the
`via` clause. The clause stages the **source** compartment; the law governs its
residence:

```camdl
onset    : E --> I  via erlang(stages = 3, mean = 7 'days)    # E's residence is Erlang-3
recovery : I --> R  via erlang(stages = 3, mean = 10 'days)   # I's residence is Erlang-3
```

A transition has **either** `@ rate` **or** `via law` — never both. `@ rate` is
an ordinary exponential transition (the rate is the propensity); `via law` is a
staged residence (the law supplies the rate, `k/τ` per stage). This is the crisp
resolution of the `@`-ambiguity that dogged the earlier framing: there is no
"entry force vs residence rate" confusion because a `via` transition has no `@`.
The force that _fills_ the compartment lives on a separate, ordinary transition.

A worked SEIR with non-exponential latent **and** infectious periods:

```camdl
compartments { S, E, I, R }
let N = S + E + I + R
transitions {
  infection : S --> E  @ beta * S * I / N                       # ordinary event: the FOI
  onset     : E --> I  via erlang(stages = 3, rate = sigma)     # E's residence = Erlang latent
  recovery  : I --> R  via erlang(stages = 3, rate = gamma)     # I's residence = Erlang infectious
}
```

This lowers to (E → E_1,E_2,E_3; I → I_1,I_2,I_3, all visible):

```camdl
let N = S + (E_1+E_2+E_3) + (I_1+I_2+I_3) + R
transitions {
  infection : S --> E_1 @ beta * S * (I_1+I_2+I_3) / N    # FOI sums the I-stages; inflow → stage 1
  onset_1   : E_1 --> E_2 @ (3*sigma) * E_1
  onset_2   : E_2 --> E_3 @ (3*sigma) * E_2
  onset_3   : E_3 --> I_1 @ (3*sigma) * E_3               # E-chain exits into I-stage-1
  recov_1   : I_1 --> I_2 @ (3*gamma) * I_1
  recov_2   : I_2 --> I_3 @ (3*gamma) * I_2
  recov_3   : I_3 --> R   @ (3*gamma) * I_3
}
```

Everything the lowering does is existing stratification behaviour: the FOI's
bare `I` sums the stages (`PopSum`), `N`'s bare `E`/`I` sum, and the inflow
`S --> E` redirects to `E_1` (the require-full-index rule already forces this).
It is isomorphic, modulo stage names, to the `seir_erlang.camdl` golden — the
anchor test (§8).

### The rate is right — R₀ preserved, only the shape changes

Transmission happens at `beta*S/N` per infectious individual _regardless of
stage_ — the `3*gamma` scaling is only on the **progression** through stages,
never on the FOI. So the mean infectious duration is
`3 × 1/(3*gamma) = 1/gamma`, unchanged, and `R₀ = beta/gamma`, unchanged.
Staging changes the dwell-time _distribution_ (and thus the generation-time
shape and epidemic speed) but not the total transmission. Correctness follows
from transmission and progression being separate rates.

### Types

```ocaml
type transition_decl = {
  (* ... trname, trindices, trsrc, trdst, trrate, trguard, trlineage ... *)
  trvia : via_call option;   (* NEW: Some ⇒ staged residence; trrate must be absent *)
}

(* PARSER output: a raw, untyped law call. The typed via_spec is built in the
   EXPANDER (stages/mean/rate/weight extraction + validation). *)
and via_call = string * (string * expr) list

(* EXPANDER output. *)
type via_spec =
  | Erlang      of { stages : pos_int; mean : mean_spec }
  | HyperErlang of { branches : hyper_branch list }
  | Fixed       of { duration : expr; precision : pos_int option }   (* → high-k Erlang, §4 *)

and hyper_branch = {
  weight : expr option;          (* None on the LAST branch ⇒ 1 − Σ others *)
  stages : pos_int;
  mean   : mean_spec;
  to_    : stoich_ref option;    (* per-branch destination; None ⇒ the transition's TO *)
  label  : string;
}

and mean_spec = Mean of expr | Rate of expr   (* exactly one — encodes the XOR *)
```

`stages` is a compile-time `pos_int` (structural, not fittable — see §6). `mean`
/ `rate` / `weight` are expressions and may reference parameters.

## 4. The laws

Three laws ship in the first push; they cover the disease table in §1.

**`erlang(stages, mean | rate)`** — the workhorse. Integer-shape gamma. Every
unimodal latent/incubation/infectious period. `stages = 1` is the ordinary
exponential dwell (one stage — _not_ a no-op: it's the exponential SEIR, which
is the point of the `k` knob).

**`hyper_erlang(branch(...), …)`** — a finite mixture of Erlangs, branched at
entry. Each `branch(label, weight, stages, mean | rate, to?)` is a
self-contained record (no fragile parallel lists). The mixture covers two shapes
the diseases need:

- **Same endpoint, different durations** (polio shedding — typical vs
  prolonged):

  ```camdl
  clearance : I --> R via hyper_erlang(
    branch(label = typical,   weight = p, stages = 2, mean = 4 'weeks),
    branch(label = prolonged,             stages = 1, mean = 2 'years)
  )
  ```

- **Different endpoints + durations** (Ebola — fatal vs recover, with `to`). The
  branches carry destinations, so the transition has no single arrow target:

  ```camdl
  outcome : I via hyper_erlang(
    branch(label = fatal,   weight = cfr, stages = 3, mean =  8 'days, to = D),
    branch(label = recover,               stages = 3, mean = 12 'days, to = R)
  )
  ```

  This is **not** reducible to two competing exits: two exponential hazards out
  of `I` give exponential, _coupled_ outcome times; gamma onset-to-death with a
  different gamma onset-to-recovery requires branching the outcome at entry and
  giving each arm its own chain and endpoint. Under staging, `I` becomes two
  parallel visible chains; the bare `I` in the FOI sums _all_ their stages (all
  infectious); each chain exits to its own destination. The last branch's weight
  is implicit (`1 − Σ others`), so the mixture is normalized by construction and
  a _fitted_ weight is always valid.

**`fixed(τ)`** — a near-deterministic delay (malaria incubation; fixed
treatment/quarantine windows). The Erlang chain cannot be exactly deterministic
(CV `1/√k`), so v1 lowers `fixed(τ)` to a high-`k` Erlang, with `k` chosen for a
documented residual CV (default targeting CV ≈ 0.1; `fixed(τ, precision = k)` to
override). **It is honestly an approximation** — the diagnostic and docs say so
— and an exact discrete conveyor is a deferred backend (§7).

**Deferred, with reasons:** `coxian` / general phase-type (no standard model
among these four needs unequal-rate chains; Erlang + hyper-Erlang cover them);
`approx_gamma` non-integer shape (round to the nearest Erlang — standard
practice); the exact conveyor for `fixed`; and **competing exits during a
residence** (TB's slow reactivation arm; death-at-constant-hazard throughout an
infectious period) — §7.

## 5. Lowering

`via` lowering is an AST→AST pre-pass that runs during expansion, **before
stratification and the duplicate-name guard**. It rewrites a staged-residence
transition by treating the source compartment's stages as a **sub-stratification
dimension**, so all downstream machinery (the bare-name `PopSum` rule, the
require-full-index stoichiometry rule, observation projection, balance,
autodiff) applies unchanged. Concretely, for
`onset : E --> I via erlang(stages = k, …)`:

1. Add a stage dimension to `E` → compartments `E_1 … E_k`.
2. Emit the chain `E_i --> E_{i+1} @ (k/τ)·E_i`, and the exit
   `E_k --> I @
   (k/τ)·E_k` (or `--> I_1` if `I` is itself staged).
3. Redirect every existing inflow to `E` so it lands in `E_1` (the require-full-
   index rule already rejects a bare staged destination, so this is the only
   legal target).
4. Leave every _rate-position_ reference to `E` alone — the bare-name sum rule
   turns it into `PopSum([E_1…E_k])` automatically.

`hyper_erlang` additionally branches the entry: the inflow to the staged
compartment becomes a `DstBranch` into each branch's first stage
(`weight_i ·
inflow`), and each branch is its own chain exiting to its own `to`.
(`DstBranch` already lowers a weighted multi-destination transition,
`expander.ml:3043` — only the weighting is reused; the chain synthesis is new.)
`fixed` lowers to `erlang` with the chosen `k`.

Because the lowered form is ordinary compartments + transitions, the engine, all
three backends, and the source-to-source autodiff are untouched — `via` adds
**zero** new IR algebra; it is a macro over existing nodes.

**Naming.** Stage compartments follow the stratification naming (`E_1`, …, or
`E__<label>__i` for hyper-Erlang branches), with the stage dimension reserved so
a user compartment cannot collide.

## 6. Gradients, and what's estimable

Because lowering precedes autodiff and produces ordinary mass-action rates,
every continuous law parameter is gradient-estimable with no special case —
confirmed against `autodiff.ml` (the `PopSum` factor over stages is a constant
w.r.t. parameters; a fitted `mean` in `k/mean` differentiates as `−k/mean²·X`).

| Parameter             | Fittable? | Why                                                                         |
| --------------------- | --------- | --------------------------------------------------------------------------- |
| `mean` (the period)   | **Yes**   | Lands in `k/mean`; real gradient under PGAS+NUTS and gradient-free methods. |
| `rate` (= 1/mean)     | **Yes**   | Lands as `k·rate`.                                                          |
| hyper-Erlang `weight` | **Yes**   | Linear on the branched entry rates (both arms, via `1 − p`).                |
| `stages` (k)          | **No**    | Sets how many compartments exist — model structure, not a continuous knob.  |

So a Garki/Ebola fit estimates the mean incubation/infectious period and the
mixture weight (all gradient-correct) and fixes the stage counts — the standard
treatment (the stage number is the shape, chosen or compared over a few `k`).
Continuous-_shape_ fitting (non-integer gamma) is what the deferred
`approx_gamma` / `coxian` would unlock.

## 7. Interactions and scope boundaries

- **Stratification (Phase 0 prerequisite).** Staging is a sub-stratification, so
  a staged _and_ age/space-stratified compartment (`I` with dims `[age, stage]`)
  must support partial indexing — `I[a]` summing over the stage dimension. That
  "omit a dimension → sum over it" rule is **specified** (spec §5.1/§5.3, with
  `E[a]` over `latent_stage` as the literal example) but **unimplemented**
  (`expander.ml:2161` concatenates only the supplied index → E100; the
  prevalence path emits a dangling `CurrentPop` caught only at Rust runtime).
  Every realistic model is stratified, so this is a **prerequisite**, not an
  edge case — and the fix is bounded (teach the `EIndex` compartment branch and
  `prevalence_projection` to consult `comp_dims` and emit `PopSum`/
  `CurrentPopSum` over the omitted dims), and it retires a live doc-vs-code gap.
- **`balance {}`.** A bare staged name sums correctly in a conservation
  expression (same resolver), so `balance` composes — no hard error needed (this
  is a genuine improvement over the hidden-pipeline design, which would have
  silently mis-conserved).
- **Competing exits during a residence (deferred).** A hazard that races with
  the dwell throughout — TB's lifelong reactivation out of latency, or death at
  a constant hazard during an infectious period — is a per-stage competing
  transition. It is _not_ the same as `hyper_erlang`'s branch-at-entry (where
  the outcome is decided on arrival). Reserved as a follow-up; the AST keeps
  room for it. Until then the manual per-stage form expresses it.
- **Reporting delays are a different subsystem.** An onset-to-report delay that
  moves no population is a convolution in the **measurement** layer (the
  unimplemented `DelayedFlow` / `Convolved` projection over an incidence
  history, `compartmental-ir-spec.md:407`), not a staged residence. EpiNow2 /
  epidemia encode it exactly this way. Out of scope here; tracked separately.
- **Backends & run identity.** The lowered form is ordinary transitions, so all
  three backends and the autodiff are untouched, and the hashed (post-lowering)
  IR re-keys automatically.

## 8. Testing strategy

Correctness spans the desugar (IR), the dynamics (the intended distribution),
and the matrix (backend × method × stratification). Each phase (§9) lands with
its tier green. The per-law distributional validations live in the **expensive
`tests/external/` tier** (CI + `make test`, skipped by `make test-fast`) — the
same tier as the `he2010` pomp-oracle.

**T0 — partial-dimension summing** (the Phase-0 prerequisite). `I[a]` over a
compartment with `[age, stage]` (or any two dims) sums the omitted dimension in
rates, observations, and balance; the dangling-`CurrentPop` runtime error is
gone; the spec §5.3 worked example compiles and simulates.

**T1 — lowering correctness (IR-level).** Anchor:
`onset : E --> I via
erlang(stages = 3, rate = sigma)` lowers to a
compartment+transition set equal — modulo stage names — to `seir_erlang.camdl`
(identical stoichiometry and constant-folded rate ASTs, AST-level assertion so
it can't pass vacuously). Plus: stage-rate (`k/τ`); inflow lands in stage 1;
bare source name sums in the FOI; `hyper_erlang` per-branch destinations and
entry weighting; `via` ⊻ `@` (both → error); stage-name collision is a hard
error. Golden IR fixtures for one `erlang`, one same-destination `hyper_erlang`,
and one per-destination `hyper_erlang` (Ebola).

**T2 — distributional correctness (known-answer, the scientific anchor).**
Primary: the deterministic ODE-backend impulse response of a chain equals the
closed-form gamma kernel `g(t; k, k/τ)` within integrator tolerance (no sampling
noise). Secondary: a stochastic pulse cohort's empirical dwell matches
Erlang(`k`, `k/τ`) — mean `τ`, variance `τ²/k`, CV `1/√k`, CDF within a
committed band (exact `scipy.stats.gamma(a=k, scale=τ/k)` / R
`pgamma(shape=k, rate=k/τ)` call committed, with the `scale = τ/k` footgun
noted). `hyper_erlang`: the mixture matches `Σ wᵢ·Erlang(kᵢ, kᵢ/τᵢ)` in mean,
variance, survival, and — for the per-destination case — the **branch split
fractions** (Ebola CFR). `fixed`: the realized CV matches `1/√k` for the chosen
`k`. One fixture per law.

**T3 — backend × method coverage.** The same staged model runs on all three
forward backends (stochastic mean → ODE field) and under each inference method
the model admits (PF, IF2, PGAS, PMMH) with no capability rejection.

**T4 — gradient correctness.** A fitted `mean` and a `hyper_erlang` `weight`
each yield non-empty `rate_grad`; a finite-difference check on `∂rate/∂mean`;
and a recovery test (simulate known `mean` + `weight`, fit with IF2 and
PGAS+NUTS, recover within the interval — gradients present and correct, on both
paths).

**T5 — validation & error quality.** `mean` XOR `rate`; `via` ⊻ `@`;
non-duration `mean` / non-rate `rate` / non-positive-integer `stages`; a weight
on the last hyper-Erlang branch; `fixed` precision; each a distinct E-code
naming the transition.

**T6 — stratified, observed, multi-via, regression.** A staged **and**
age-stratified `I` (the realistic case, exercising T0): `k × n_age`
compartments, FOI sums correctly, per-age `I[a]` works. An observation scored
off a staged flow against a known truth. Two staged residences in one model
(latent + infectious). Golden-neutrality: every non-`via` model byte-identical.
Spec doc-tests compile.

## 9. Phasing and implementation plan

Each phase lands with its test tier green before the next.

0. **Partial-dimension summing.** Implement the spec §5.1/§5.3 omit-a-dimension
   rule in the `EIndex` compartment branch and `prevalence_projection` — a
   bounded fix that staged∧stratified models require and that retires a
   doc-vs-code gap. → **T0**. (Prerequisite for every realistic staged model.)
1. **AST + parser.** `trvia : via_call option`, the `via law(...)` clause, the
   `via` ⊻ `@` rule, the `via` keyword. Parser stores the raw call. → **T6**
   (parse, round-trip, golden-neutrality).
2. **Erlang staging.** The AST→AST pre-pass: stage the source compartment as a
   sub-dimension, emit the chain + exit, redirect inflow to stage 1, scale the
   rate; build the typed `via_spec` with `pos_int` + `mean` XOR `rate` +
   collision checks. → **T1**, then **T2/T3**, **T4**.
3. **Validation + `fixed`.** Dimchecks, `via` ⊻ `@`, `fixed` → high-`k` Erlang
   with documented CV. → **T5**.
4. **Hyper-Erlang.** Branched entry (`DstBranch` weighting) into parallel
   chains; per-branch `to`; last-weight-implicit normalization. → **T1/T2/T5**
   repeated, incl. the Ebola per-destination and CFR-split cases.
5. **Docs.** Spec §5 prose + the `via` examples (the T6 doc-tests); cheatsheet.

Deferred follow-ups (named, not open questions): `coxian`; `approx_gamma`
(continuous-shape fitting); the exact discrete conveyor for `fixed`; competing
exits during a residence (TB reactivation; death-during-infectiousness); and the
measurement-layer reporting-delay convolution (`DelayedFlow`).
