# Loop-invariant code motion: per-eval staging of param/table-only subexpressions

Date: 2026-06-20 Status: Implemented and **default-on** (ODE/forward +
stochastic inference; `--no-licm` / `CAMDL_NO_LICM` is the escape hatch) Issue:
gh#272 Schema: 0.18 → 0.19 (adds `per_eval_bindings` + `Expr::PerEvalRef`)

The runtime half is **Design C — a staged per-eval scratch threaded as data on
`EvalCtx`** (`per_eval: Option<&[f64]>`, the sibling of `int_float_override`).
The compiler half — the OCaml LICM pass, the IR nodes, the keystone validation —
is shared with the original sketch; only the runtime evaluation of `PerEvalRef`
differs from the first cut. Measured on the real MRE kernel: ~4.2× faster per
ODE step, bringing the in-model fittable-γ kernel to ~precomputed-matrix speed.
The staged scratch is threaded through every backend AND every stochastic
inference producer (PF / IF2 / PGAS / PMMH), so the in-model fittable kernel
reaches fixed-kernel parity on the production Bayesian path, not just the ODE
skeleton. **LICM is on by default** (mirroring `constant_fold`); `--no-licm` /
`CAMDL_NO_LICM` is the escape hatch. The flip was golden-neutral — no golden
model has hoistable structure, so `make update-golden` under LICM-on changed
zero files; run identity re-keys only for models that actually hoist (a user
in-model kernel), which is the intended behaviour, and a non-hoisting model's IR
is byte-identical to pre-flip (so existing CAS entries stay valid). Remaining
follow-ons: the flat-eval per-eval tape and the strength-reduction peephole.

## Problem

Computing a coupling/mixing matrix **in-model** — so its parameters are fittable
— is measured at **3.77× slower per likelihood eval** than reading a precomputed
matrix, because the matrix is re-evaluated on every integration step even though
it is loop-invariant: it depends only on parameters and tables, not on state or
time.

### Reproduction (verified 2026-06-20)

Two models identical except for the gravity coupling kernel (Sierra Leone
14-district metapopulation SEIRD, MH on the ODE skeleton, single chain, dt=1,
490-day trajectories; bundles in `ebola_camdl/mre`). 1500 MH iterations each:

| model                                                        | kernel   | ms/eval  | it/s |
| ------------------------------------------------------------ | -------- | -------- | ---- |
| precomputed `read("kernel.tsv")`, γ baked                    | baseline | **9.5**  | ~105 |
| in-model `W[p,q] ∝ N0[q]·exp(−γ·log(dratio[p,q]))`, γ fitted | target   | **35.8** | ~28  |

`35.8 / 9.5 = 3.77×`. The two configs draw the same RNG, score the same
`neg_binomial` likelihood on the same observations, and differ only by `gamma_k`
and the kernel form. So the entire **26.3 ms/eval delta is the forward
integration of the kernel** — ~73% of every in-model eval is loop-invariant
rebuild.

(Measured speedups with LICM on, and the on/off byte-identity evidence, are in
`docs/dev/notes/2026-06-22-licm-kernel-benchmark.md`.)

The motivation for the in-model form is the Xia/Bjørnstad/Grenfell idiom: make
the gravity distance-decay exponent γ a _fitted_ parameter with a posterior,
instead of profiling over rebuilt `kernel.tsv` files. This is the _correct_
inference object here — the companion analysis (`ebola_camdl`,
`2026-06-18-scale-selects-the-tool.md`) records that at this scale the
deterministic ODE likelihood with MH is the well-conditioned method and the
exact stochastic machinery (IF2/PMMH/PGAS) degenerates. So the gradient-free
forward path the benchmark exercises is the production workhorse for
national-scale spatial fits, not a niche.

### Root cause (verified against code)

The kernel never becomes a binding. The OCaml expander's `let_is_hoistable`
(`ocaml/lib/compiler/expander.ml:1905`) refuses to hoist any `let` whose body
references a parameter (`body_refs_param_or_let`, `:1855`), because the autodiff
pass differentiates a `BindingRef` to zero (`ocaml/lib/ir/autodiff.ml:324`,
`BindingRef _ -> Known (Const 0.0)`) — correct only under the invariant that
bindings are **param-free**, and enforced by a Rust safety net
(`rust/crates/sim/src/compiled_model.rs:803`, rejects a param inside a binding
body). So `beta`, `dfac`, `bc`, and the whole kernel are **inlined** at every
use site.

Confirmed with `camdlc inspect --cost-report`: the 14 emitted bindings are all
the param-free `N_*` (= S+E+I+R per patch); `infection_Kailahun` carries 426
rate nodes; 392 `Reduce` terms, 0% collapsed by `constant_fold` (it is a
`read()`-loaded, dense matrix — structurally outside that pass); the in-model IR
is **13× larger** (17.9 MB vs 1.35 MB). The ODE backend then re-evaluates that
whole tree on every RK stage of every one of ~490 steps (`ode_derivs` is called
4× per RK4 step, `rust/crates/sim/src/ode.rs:150–174`) — ~2000 identical kernel
rebuilds per likelihood eval.

Per step, for each stratum `p`, the kernel evaluates a numerator sum over `q`
**and** a normalization sum over `r`:

```
W[p,q] = N0[q]·exp(−γ·log(dratio[p,q])) / Σ_r N0[r]·exp(−γ·log(dratio[p,r]))
```

≈ `2·n²` calls to `exp(−γ·log(·))` per step (≈784 transcendentals at n=14).
Every one of those depends only on `(γ, N0, dratio)` — all constant within a
single trajectory. Only `I[q]/N[q]` in the coupling is state-dependent.

## Design

A **loop-invariant code-motion (LICM) pass in the OCaml compiler**, a sibling of
the existing `Constant_fold.fold_model`, that extracts maximal param/table-only
subexpressions out of the (already-differentiated) **dynamics** expression trees
into a new per-eval binding category, replacing them with references that the
Rust runtime evaluates **once per θ-stable scope** instead of once per step.

### Why this seam

There is already a post-autodiff IR pass of exactly this shape.
`Constant_fold.fold_model` (`ocaml/lib/ir/constant_fold.ml`) runs in the slot
`validate → dimcheck → lint → differentiate_transitions → maybe_constant_fold →
serialize`
(`ocaml/lib/compiler/compiler.ml`), rewrites the rate, rate_grad, bindings, and
ODE derivatives (`:103–117`), is value-preserving, and is pinned by an A/B
byte-identity gate (`rust/crates/sim/tests/gate_constant_fold_ab.rs`). LICM is
its sibling: same pipeline slot, same value-preserving contract, same gate
pattern. Keeping the analysis in OCaml puts all compiler logic in one place; the
Rust side only _executes_ the result.

### Run the pass _after_ autodiff, not before

Placing LICM after `differentiate_transitions` is strictly safer than before:

- **After**: autodiff differentiates the inlined rate exactly as today — **zero
  lines change in autodiff, the highest-risk module.** LICM then CSE-hoists the
  loop-invariant subtrees out of both the rate and the already-computed
  rate_grad. The gradient path gets the same speedup for free, because
  rate_grad's invariant subterms (∂W/∂γ, and W itself in the product-rule terms)
  are param/table-only too.
- **Before**: autodiff would have to learn to differentiate a hoisted
  param-binding (look through to its body), editing the scariest file for no
  additional benefit.

**The exact slot, and order vs `constant_fold`.** LICM is a pure `model → model`
pass — template `Constant_fold.fold_model` (`constant_fold.ml:103`), **not** the
expander's `ctx`-bound hoisting machinery (`hoisted_rev` /
`register_hoisted_binding`), which is already drained into `model.bindings` and
gone by the time LICM runs. It builds its **own** CSE-keyed,
topologically-ordered accumulator over `Ir.expr` trees. It wraps the final
expression of `finish_compile` — `Ok (maybe_licm (maybe_constant_fold m))` at
`compiler.ml:449`, gated on `CAMDL_LICM` exactly as `maybe_constant_fold` guards
on its env var (`:402`) — so both entry points (`compile`, `compile_with_reads`)
are covered by the one `finish_compile` site. Running **after**
`maybe_constant_fold` means LICM hoists already-folded subtrees and
`constant_fold` never sees a `PerEvalRef` (its `fold` is exhaustive —
`constant_fold.ml:64`); `fold_model` does **not** iterate `per_eval_bindings`
and need not, since they are created after it runs. (`constant_fold` is a no-op
on the `read()`-loaded kernel this targets; the ordering matters only for
inline-literal models where both fire.)

### Two axes: param-free vs param-carrying, invariant vs variant

Every subexpression sits on two independent axes, and conflating them is the
trap the existing `BindingRef` invariant was protecting against:

- **Gradient axis** — _param-free_ (∂/∂θ ≡ 0) vs _param-carrying_ (∂/∂θ may be
  nonzero). This is what makes `autodiff`'s `BindingRef → 0` sound.
- **Lifetime axis** — _invariant_ within a trajectory (depends only on
  params/tables/constants) vs _variant_ (depends on state, time, `dt`, or a
  forcing). This is what determines cache lifetime: per-eval vs per-step.

Existing `bindings` / `BindingRef` are **param-free** — and that is _all_ they
are. Hoist eligibility (`expander.ml:1855`) rejects only parameters and other
`let`s; it explicitly allows compartments, tables, forcings, time, and
constants. So a `BindingRef` may be variant (`N[l] = S+E+I+R`, the common case)
or invariant (`log(dens[l])`, pure table); either way its param-gradient is
zero, which is the only property the three consumers (`autodiff`,
`pgas::collect_param_refs`, `eval_resolved_deriv`) and the
`binding_param_free_guard.rs` test rely on.

The new node occupies the remaining quadrant — **param-carrying + invariant** —
which is exactly the loop-invariant-but-fittable kernel:

```
IR (schema 0.19):
  model.per_eval_bindings : [ { name, expr } ]      // new top-level list, sibling of `bindings`
  Expr::PerEvalRef(name)                            // new expr node, sibling of BindingRef
```

- `bindings` / `BindingRef`: **unchanged** — param-free, `→0` everywhere,
  `binding_param_free_guard` stays green.
- `per_eval_bindings` / `PerEvalRef`: param/table/constant only, never state,
  time, `dt`, or forcing. Bodies are topologically ordered (a per-eval binding
  may reference earlier ones via CSE).

Reusing `BindingRef` for the param-carrying case would relax the param-free
invariant across the inference math; a distinct node keeps the existing meaning
untouched and confines the new behaviour to one variant.

(The issue's third "load-time const" stage is folded into per-eval: computing a
table-only value once-per-eval vs once-ever is negligible once amortized over
~490 steps, and avoids a third category. Existing param-free-but-invariant
bindings like `log(dens[l])` stay per-step-cached — a minor missed optimization,
out of scope.)

### The LICM pass (OCaml): coverage and the reason for it

LICM rewrites **exactly the dynamics surfaces — transition `rate`, transition
`rate_grad`, and `ode_equations` derivatives.** It deliberately does **not**
touch:

- existing `bindings` (param-free by construction — no `PerEvalRef` would carry
  a param, and they are already per-step-cached);
- transition **overdispersion** σ² expressions;
- **observation** projections and likelihood expressions.

The reason is a hard contract, not a convenience: the three dynamics surfaces
are the ones the runtime evaluates with `eval_resolved` (the forward value path
— `pgas_grad.rs:129/251` evaluate the compiler-emitted `rate_grad` _as a value_,
not by re-differentiating it). The excluded surfaces are the ones touched by the
**secondary** gradient consumers — `eval_resolved_deriv` (overdispersion σ² at
`pgas_grad.rs:368`, obs-likelihood params in `obs_model.rs`) and
`collect_param_refs` (used in exactly one place, `pgas.rs:2170`, scanning
`DerivedExpr` obs projections for the gh#76 silent-zero guard). By keeping
`PerEvalRef` out of those surfaces, **neither secondary consumer ever encounters
it**, so neither needs to change. This avoids a real hazard:
`eval_resolved_deriv` groups `TableLookup → 0.0` (`resolved_expr.rs:692`), and
inline table cells _can_ be param-valued
(`TableSource::Inline { values: Vec<Expr> }`; `compiled_model.rs` notes "a
param-referencing inline-table value tracks the params slice"), so a
`PerEvalRef` over such a cell on that path would silently produce a zero
derivative.

`autodiff` needs **zero change** (it ran before LICM and never sees a
`PerEvalRef`). The dynamics gradient stays correct because LICM is
value-preserving and `rate_grad` is evaluated, not re-differentiated.

### Extraction rule

Classify each node:

- **invariant** iff it is `Const`, `Param`, a `TableLookup` over invariant
  indices, or a `BinOp`/`UnOp`/`Cond`/`Reduce`/`UncheckedDim` of invariant
  children;
- **variant** if it (transitively) contains `Pop`, `PopSum`, `Time`, `Dt`,
  `Projected`, `ObsColumnRef`, `TimeFunc` (a forcing — time-varying), or a
  `BindingRef` (which may be state-dependent).

  **Do not reuse the existing predicates verbatim as the variant test.** Each
  misses a case that is fatal here: `compiled_model.rs`'s
  `expr_is_time_dependent` (`:216`) has **no `Dt` arm** — `Dt` falls through its
  `_ => false`, so a `dt`-bearing subtree would be misclassified _invariant_ and
  frozen at the first substep's `dt`. Substeps are clipped at interval/output
  boundaries (`ode.rs:255`, `self.dt.min(h_max)`; `schedule.substeps` yields a
  per-substep `step_dt`), so a frozen `dt` is a silent-wrong trajectory for any
  model with `dt` in a rate (gh#54). The LICM variant test is therefore a
  **new** predicate — `references_state(e) || references_time(e)`, where
  `references_time` explicitly covers `Time`, `Dt`, **and** `TimeFunc` — using
  the existing functions only as references for the state and forcing halves,
  never as the whole test.

Hoist each **maximal invariant subtree**, defined relative to a **virtual
variant parent at the use site**: the rate / rate_grad-term / ODE-derivative
root is treated as if its parent were variant. So a subtree is hoisted iff it is
invariant and either its real parent is variant or it _is_ the root. This
catches both the kernel (invariant subtree under the variant coupling sum) **and
a fully-invariant root** (e.g. a zero-order import rate `iota·exp(−k)` whose
whole expression is param/table-only) — the parent-based rule alone would miss
the latter.

Gate on a cost threshold so trivial subtrees are left inline (only hoist
subtrees containing a transcendental, a `Pow`, or a `Reduce`, or exceeding N
nodes) — hoisting a bare `Param`/`Const` saves nothing and only grows the IR.

CSE-dedup identical subtrees by a **bitwise expression hash/equality** — _not_
OCaml's polymorphic `=`, which treats `-0.0` and `0.0` as equal and `nan` as
unequal to itself. The distinction is load-bearing: `-0.0` is the documented
seed of a left-folded `Reduce` (`resolved_expr.rs:587`), so merging a
`Const(-0.0)` subtree with a `Const(0.0)` one could perturb a sum. With bitwise
CSE the kernel numerator term `N0[q]·exp(−γ·log(dratio[p,q]))` — shared by the
numerator sum and the normalization sum — is stored once. The existing
`inspect.ml::expr_hash` (`:371`) uses `Hashtbl.hash`, which collides
`-0.0`/`0.0` — it is precisely the wrong seam to reach for here; LICM needs its
own bitwise hash.

**Before/after** (the coupling term). Only `I[q]/N[q]` is variant:

```
before:  Σ_q ( N0[q]·exp(−γ·log(dratio[p,q])) · I[q]/N[q] )
              / Σ_r ( N0[r]·exp(−γ·log(dratio[p,r])) )

after:   Σ_q ( PerEvalRef(Wnum_p_q) · I[q]/N[q] ) / PerEvalRef(Z_p)
  per_eval_bindings:
    Wnum_p_q = N0[q]·exp(−γ·log(dratio[p,q]))      // one per (p,q), CSE-shared
    Z_p      = Σ_r PerEvalRef(Wnum_p_r)            // references earlier per-eval bindings
```

The ~784 transcendentals/step collapse to once-per-eval. Substituting a
`PerEvalRef` for a bitwise-identical subtree preserves evaluation order, so the
trajectory and loglik are byte-identical (see "Gates" for the precise scope of
that claim).

Follow-on peephole (separate, optional): strength-reduce
`exp(−γ·log d) → d^(−γ)` (one `powf` instead of `log`+`mul`+`exp`) — a local
rewrite inside the hoisted body, not load-bearing for the main win.

### Runtime (Rust) — execution only, no compiler logic

1. `ResolvedExpr::PerEvalRef(usize)` +
   `ResolvedModel.per_eval_bindings:
   Vec<ResolvedExpr>`, resolved by slot at
   `CompiledModel::new()` via a new `per_eval_index` field on `ResolveCtx`
   (mirroring `binding_index`, `resolved_expr.rs:125`). Build `per_eval_index`
   with an **assert-unique on insert** — the existing `binding_index`
   `.collect()` (`compiled_model.rs:1032`) silently last-writer-wins on a
   duplicate, which here would mis-resolve a self-reference; LICM must mint
   collision-proof names (a reserved prefix the lexer forbids in user
   identifiers). A Rust-side validation, run as a **hard precondition at the top
   of `CompiledModel::new` — before `required_capabilities` is derived** (so a
   `Dt`/forcing smuggled into a body can't escape the `RUNTIME_DT` gate), and
   modeled on the param-free net (`compiled_model.rs:802`), asserts each
   `per_eval_binding` body (a) references only param/table/constant and earlier
   per-eval slots — no state/time/`dt`/forcing/`BindingRef` — and (b) is
   **topologically ordered** (`PerEvalRef(j)` ⇒ `j < i`), the invariant the
   `BindingRef` path gets for free from the expander's reverse-topological
   `hoisted_rev` but LICM's fresh accumulator must re-establish.
2. **Staged per-eval scratch on `EvalCtx`.** `EvalCtx` gains
   `per_eval: Option<&'a [f64]>` (the sibling of `int_float_override`). For a
   θ-stable span, the values of `model.resolved.per_eval_bindings` are computed
   **once** into an owned `Vec<f64>` by
   `eval_per_eval_scratch(model, params, t,
   dt)` (`resolved_expr.rs`) and
   lent into every rate eval of that span. The eval arm is a slice read:

   ```rust
   ResolvedExpr::PerEvalRef(slot) => match ctx.per_eval {
       Some(scratch) => scratch[*slot],
       None => eval_resolved(&ctx.model.resolved.per_eval_bindings[*slot], ctx),
   }
   ```

   The scratch is built in topological order, lending the already-filled prefix
   (`per_eval: Some(&scratch[..i])`) while evaluating body `i`, so a body that
   references an earlier slot (`Z_p = Σ_r PerEvalRef(Wnum_p_r)` — the benchmark
   shape) reads the staged prefix value. No thread-local cache, no generation,
   no `RefCell` borrow dance: the scratch is owned by the caller and passed as
   data.
3. **Safe fallback (`None` ⇒ on-demand eval).** When the caller has not staged a
   scratch (`per_eval == None`), `PerEvalRef` evaluates its body directly —
   byte-identical to the staged read, just unamortized. So `PerEvalRef` is
   **correct on every eval site regardless of whether that site stages a
   scratch**; the phase split (below) is safe by construction, not by "don't use
   it there." Every non-LICM eval site (obs likelihoods, interventions, priors,
   inference producer steps) passes `None`.
4. **Flat evaluator** (`flat_eval.rs`, opt-in `CAMDL_EVAL_FLAT`) is an
   independent opt-in path that returns before the `EvalCtx` eval and does not
   read `per_eval`. Wiring the staged scratch into the flat VM (an
   `Op::PerEval(slot)` tape read) is a follow-on; until then, the `CAMDL_LICM` ×
   `CAMDL_EVAL_FLAT` combination is handled by the flat builder (both flags are
   off by default).
5. **Staging sites — one per θ-stable span.** The scratch is computed once at
   the entry of each fixed-θ span and threaded down to the eval sites it covers:
   - ODE: at the top of `run_ode` (`ode.rs`), before the
     `while next_stop {
     stepper.advance(…) }` driver loop, lent through
     `OdeStepper::advance` → `rk4_step`/`dopri5_try_step` → `ode_derivs` (and
     the euler-flow `eval_propensities`). θ is fixed for the whole `run_ode`
     call; that one site covers **both** ODE inference
     (`compute_ode_loglik → run_ode`) and forward ODE simulate.
   - Forward simulate (chain_binomial/gillespie): staged once at the top of
     `run_chain_binomial_with_observer` / `run_gillespie_with_observer` and lent
     into every `step_one` / `eval_propensities` of the run. `step_one` — the
     producer step shared with inference — takes `per_eval` as a parameter, so
     forward passes the staged scratch and inference passes `None`.
   - Stochastic inference producers, each staged at its θ-stable boundary above
     the substep loop and threaded through the shared `step_one` /
     `log_transition_density_substep`:
     - **PF** (`bootstrap_filter`) and **PMMH** (`bootstrap_filter_correlated`):
       θ is global to the filter → stage once at the top, lend into every
       particle's every substep.
     - **IF2** (`run_if2`): θ is per-particle → stage inside the per-particle
       closure from `pp`, before its substep walk. Because the scratch is
       owned/lent data, "one scratch per particle" is just a local `Vec`
       structurally bound to that particle's θ; there is no shared cache to
       mis-scope, so IF2's per-particle perturbed θ cannot serve one particle's
       kernel to another's θ — the correctness obligation the original cache
       design carried as placement discipline is dissolved into the type.
     - **PGAS** (`csmc_as` per sweep; `complete_data_loglik`[`_grad`];
       `simulate_reference_on_grid`): θ is fixed per call → stage at the top,
       thread into the producer, the per-substep density/gradient evals, and the
       rate_grad `EvalCtx`.

### Why Design C — the alternatives considered

The runtime evaluates `PerEvalRef` via **Design C** (the staged scratch above).
Two other designs were considered and rejected; the comparison is the design
record. The compiler half — the LICM pass, the IR nodes, the keystone validation
— is identical under all three; only the runtime evaluation of `PerEvalRef`
differs.

**Design A — a thread-local per-eval cache tier + RAII `EvalScope`.** A first
cut used a second generation on the thread-local `BindingCache`, invalidated per
θ-stable span by an `EvalScope` RAII guard each backend enters. It works (≈4.2×
measured), but the per-eval cache is shared thread-local state invalidated by a
generation a per-backend `EvalScope` bumps. Correctness then depends on each
backend placing the scope at a θ-stable boundary — and the inference loops nest
θ differently (IF2 perturbs θ _per particle_). A scope entered around the
parallel region instead of per particle silently serves one particle's kernel to
another's θ. The design carries a correctness obligation as placement discipline
rather than as a type, and it accretes a second of every cache primitive
(generation, `active`, override, hit counter, plus a separate flat-VM tier).

**Design C — a staged prologue, threaded as data (implemented).** Compute the
per-eval bindings once for the θ-stable span into an owned `Vec<f64>` scratch,
and thread it as a borrow on `EvalCtx` (`per_eval: Option<&[f64]>`, the exact
sibling of the existing `int_float_override: Option<&[f64]>`).
`PerEvalRef(slot)` is `ctx.per_eval[slot]` — a slice read — falling through to
on-demand eval when `None` (byte-identical, so an un-staged path is correct,
just unamortized).

This is **correct by construction**: the scratch is owned/lent, not ambient, so
there is no shared mutable cache to alias across particles — the value is
structurally bound to the θ it was computed at, and the `if2.rs` per-particle
case is just "one scratch per particle." A missed wiring is a _compile error_ (a
typed field), not a silent stale read. It carries no thread-local cache tier, no
`EvalScope` RAII, no borrow-before-recurse dance, and no flat-VM cache tier. And
it is the extensible seam: a future `const` stage and the gradient-path per-eval
stage each compose as another owned scratch read by index, where Design A would
need a third and fourth thread-local tier.

(Design B — auto-invalidate by keying the cache on the param vector — was
rejected: IF2/PGAS/PMMH **mutate the θ buffer in place** (`if2.rs:391,480`;
`pmmh.rs:467`), so pointer-identity keying serves stale values, and
content-hashing has no existing identity to hang on and costs a per-eval hash.)

**What landed.** Steps 1.1 (IR substrate) and 1.2 (the LICM pass) are as
originally specced. The runtime adds `per_eval: Option<&[f64]>` to `EvalCtx`
(every construction site `None` except the rate / rate_grad / ode-derivative
eval path, since `PerEvalRef` appears only there), `eval_per_eval_scratch`
computed at each θ-span entry, and threads it through `OdeStepper::advance` →
`ode_derivs` and through `eval_propensities` / `step_one`. The A/B gate
(`gate_licm_ab.rs` §4) asserts `scratch present == scratch absent` at expression
granularity, replacing the original cache-on/off comparison.

### Consumer surface: three classes

Adding an `Expr`/`ResolvedExpr` variant touches many traversals. They fall into
three classes, and the distinction is the difference between "the build tells
you" and "it silently does the wrong thing." The earlier framing — that only the
two secondary gradient consumers matter — conflated _runtime-unreachable_ (the
scoping argument) with _compile-exhaustive_ (every `match` needs an arm). Both
are real and separate.

**Class A — compile-forced, mechanical arm (safe; the build breaks until
done).** Every exhaustive `match` over the expr type needs a `PerEvalRef` arm or
it won't compile. This is the desired safety property — none can be silently
missed. Non-exhaustive (so easy to under-count) but each is mechanical:

- OCaml: `serde.ml` (serialize + deserialize, _and_ the `model_of_json` record
  literal for `per_eval_bindings`), `dimcheck.ml` (`infer` + 3 helpers),
  `autodiff.ml::mentions` (so "autodiff changes zero lines" is true only of its
  _logic_, not its exhaustiveness — `mentions` needs an arm),
  `constant_fold.ml`, `validate.ml` (`references_param` / `check_expr_refs`),
  `expr_analysis.ml::dep_of_expr`, `lineage.ml` (`classify_parents`,
  `deriv_num_wrt_pop` — and LICM must exclude lineage transitions from hoisting
  the same way `lineage.ml:145` already disables it for state `BindingRef`s),
  `pp_expr.ml`, `inspect.ml` (≈7 expr matches incl. the cost-report walkers).
- Rust: `ir/src/expr.rs` deserialize, `ir/src/validate.rs::check_expr`,
  `resolved_expr.rs` (`resolve_expr`, `eval_resolved`, `references_state`),
  `flat_eval.rs` (the emit match),
  `multi_stream_obs.rs::collect_obs_column_refs`,
  `hierarchical.rs::eval_prior_arg`, `cli/.../coeff_guard.rs`, and the ~20
  `Model {
  … }` struct-literal sites (each needs `per_eval_bindings: vec![]`).
  The `eval_ab` bench's `count_nodes` is exhaustive but only compiles under
  `cargo
  bench` — a latent break `make test` won't surface; fix it
  deliberately.

**Class B — silent wildcard arms that rely on the per-eval invariant (the
keystone).** Several traversals end in `_ => …` and will accept a `PerEvalRef`
without complaint, doing the right thing **only if** per-eval bodies never
reference state/time/`dt`/forcing: `compiled_model.rs::collect_int_comp_deps`
(`_ => {}`, builds the Gillespie dependency graph), `expr_is_time_dependent`
(`_ => false`), `required_capabilities`/`expr_contains_dt` (which stops at the
`PerEvalRef` leaf and does _not_ descend into the body — so a `Dt` hidden in a
body would escape the `RUNTIME_DT` capability gate), and **both**
`references_state` functions — the `Expr`-level one in `compiled_model.rs`
(`_ => false`) **and** the `ResolvedExpr`-level one at `resolved_expr.rs:90`,
which returns `true` for `BindingRef` (`:105`) and gates overdispersion
validation (`compiled_model.rs:1086`) plus the per-particle-vs-fixed-state
decision. Give the latter an explicit `PerEvalRef(_) => false` arm (not
wildcard) — the contrast with `BindingRef(_) => true` is itself the keystone
boundary. Plus a few CLI/eval helpers. These are safe, but only because of the
**keystone invariant**: a `per_eval_binding` body is param/table/constant only.
That invariant must be _enforced_, not assumed — by the Rust-side constructor
validation (Design step 1) and the OCaml LICM pass's own variant test — and it
deserves a **dedicated unit test** that feeds a per-eval body referencing
state/time/`dt`/forcing/`BindingRef` and asserts rejection. Prefer an explicit
`PerEvalRef(_) => false` arm with a comment over leaving these as wildcard
fall-through, so the dependency is visible.

**Class C — hand-written, non-exhaustive traversals (silent if missed — the
blocker).** The run-identity IR hash is hand-written and lists fields
explicitly; a new field is silently dropped. See "Run identity" below — this is
the one true silent hazard and is **not** caught by adding a `match` arm.

The gradient consumers deserve special care. `eval_resolved_deriv`,
`eval_expr_deriv` (the unresolved-tree differentiator on the
`CAMDL_EVAL_UNRESOLVED` oracle path, `propensity.rs:291`), and
`collect_param_refs` are all exhaustive, so each needs a `PerEvalRef` arm to
compile — even though the scoping argument proves a `PerEvalRef` never _reaches_
them at runtime. Their arm must be **`unreachable!()` / a hard error, not a
silent `0.0` / `{}`**: a param-carrying node differentiated to zero (the way the
param-_free_ `BindingRef → 0` arm correctly does) would be silently wrong if the
scoping invariant ever broke. Making the arm panic turns "scoped correctly" from
an assumption into an enforced invariant. (The resolved-tree `eval_expr` forward
path, by contrast, _does_ legitimately evaluate a `PerEvalRef` — resolve it by
name against `per_eval_bindings`, like `BindingRef`.)

## Atomic IR-schema change (procedure)

Per CLAUDE.md "Changing the IR schema":

1. `ir/schema.json`: add `per_eval_bindings` to the model (same shape as
   `bindings`, `:76–85`) and a `per_eval_ref` variant to the `expr` `oneOf`
   (string payload, mirroring `binding_ref`, `:235–241`). Bump `ir/VERSION` 0.18
   → 0.19.
2. OCaml `ocaml/lib/ir/ir.ml`: add `PerEvalRef of string` to `type expr` (beside
   `BindingRef of string`, `:39`) and `per_eval_bindings : binding list` to
   `type model`. Serde in `ocaml/lib/ir/serde.ml` — `model_to_json` **must omit
   `per_eval_bindings` when empty** (the `| [] -> []` pattern `bindings` uses at
   `serde.ml:1259`), or every golden churns. The Class-A matches above each get
   a `PerEvalRef` arm; the pre-LICM passes (`dimcheck`/`validate`/`lint`)
   hard-error on an unexpected pre-LICM `PerEvalRef` rather than reimplementing
   inference.
3. Rust `rust/crates/ir/src/expr.rs`: add `Expr::PerEvalRef(PerEvalRefWrap)`
   (tag `per_eval_ref`, mirroring `BindingRefWrap`, `:125`) to the enum (`:224`)
   and the hand-written deserialize match (`:363`; Serialize is derived
   `untagged`, auto-handled). `rust/crates/ir/src/model.rs`: add
   `per_eval_bindings:
   Vec<Binding>` with
   `#[serde(skip_serializing_if = "Vec::is_empty")]` (copy `bindings`, `:185`) —
   the Rust-side half of the omit-when-empty requirement.
4. **Run identity** (`rust/crates/runid/src/ir_hash.rs`): add
   `self.per_eval_bindings.hash_into(h)` to `Model::hash_into` (beside
   `self.bindings.hash_into(h)`, `:1063`). This hash is hand-written and
   field-by-field — _not_ an exhaustive match — so a missing line is silent. See
   "Run identity" below.
5. `make test-unit` — fix exhaustiveness errors across both languages (Class A).
6. `make update-golden && make update-expected` — with the pass **default-off**,
   goldens are byte-unchanged **iff** both serializers omit the empty field
   (steps 2–3); the schema addition is otherwise additive. The golden _content_
   moves only when the pass is flipped on (Rollout below), as its own reviewed
   commit.
7. Commit schema + both languages + run-id hash + (empty-by-default) golden
   touch atomically.

### Run identity

The IR hash that feeds `run_id` (`runid/src/ir_hash.rs::Model::hash_into`) is
hand-written and enumerates fields; adding `per_eval_bindings` to the struct
does **not** force a hash update the way an exhaustive `match` would. If the
`hash_into` line is omitted, two models that differ _only_ in their LICM output
get the **same `run_id`** — a silent CAS collision (turning the pass on could
return a stale cached result keyed to the off-form). This is the single
silent-wrong hazard in the change, and it is exactly the class CLAUDE.md's "CAS
/ run-identity" rule guards: a field that changes stored bytes is identity and
must re-key.

The existing golden-hash test will _not_ catch the omission on its own:
`representative_model()` is a struct literal, so it compile-errors on the new
field and the author satisfies it with `per_eval_bindings: vec![]` — an empty
vec hashes to nothing, so the missing line stays invisible. The fix must
therefore **also** extend the representative model with a _non-empty_
`per_eval_bindings` and assert the hash _changes_ — a deliberate distinctness
test, not just a compile fix.

There are in fact **two re-keys**, both intended. (1) Adding the `hash_into`
line moves the model hash for _every_ model at the 0.19 schema commit — even
default-off — because `Vec::hash_into` writes an 8-byte length prefix for the
(empty) field; the `model_golden_hash` GOLDEN constant must be updated in that
commit, like prior wholesale re-keys at 0.10/0.11/0.12/0.17. (2) Flipping the
pass default-on later re-keys again for models the pass actually touches
(populated field). So "zero `run_id` churn" is true only of trajectory
_content_; the identity hash re-keys once at the schema bump and again at the
flip.

**Identity, not presentation — and the warm-start consequence.**
`per_eval_bindings` stays identity (it is _not_ added to `normalize_for_hash`).
Presentation-stripping it would be incoherent: the `PerEvalRef`-bearing
`rate`/`rate_grad`/`ode` trees _themselves_ differ between LICM-on/off and are
hashed and unstripped, so stripping the binding list alone could not make the
two forms hash-equal, and stripping the whole dynamics surface is absurd. The
cost is that the survey→fit warm-start cross-check (`init.rs:909`,
`cross_check_survey` hard-asserts `model_identity` equality) will **reject a
survey compiled at the other LICM setting** with a spurious "model edit" error —
the same invalidation any IR-version bump causes. It fails _closed_ (safe
direction), so this is a release-notes / rollout item: the default-on flip is a
survey-invalidating release, not a silent-wrong risk.

## Gates and rollout

### The A/B variant flag is the correctness apparatus

LICM ships as a **built-in on/off variant flag**, the same pattern several
compiler/runtime parts already use, because flipping it and asserting
**everything is byte-identical** is how value-preservation is _proven_, not
asserted. The established siblings:

- `constant_fold` — OCaml compile-time pass, `CAMDL_NO_CONSTANT_FOLD` escape
  hatch, gated by `gate_constant_fold_ab.rs` (compile both ways → identical
  trajectory).
- binding cache — runtime, `CAMDL_NO_BINDING_CACHE` plus a **per-thread**
  `set_binding_cache_disabled(bool)` override (`resolved_expr.rs:354`) so one
  process runs cache-on and cache-off and diffs; gated by
  `gate_binding_cache_ab.rs`.
- flat VM (`CAMDL_EVAL_FLAT`) and the unresolved-eval oracle
  (`CAMDL_EVAL_UNRESOLVED`) — same shape.

LICM has **two** independent toggles because it spans compile and runtime, and
each must be provably value-preserving on its own:

1. **`CAMDL_LICM`** (OCaml compile-time, opt-in, default-off) — does the pass
   hoist or not. Off → the IR has no `per_eval_bindings` (today's bytes
   exactly); on → the hoisted IR. The **hoist A/B gate** compiles the model both
   ways and asserts identical trajectories under every backend: proves the hoist
   is value-preserving.
2. **per-eval cache toggle** (runtime, sibling of `CAMDL_NO_BINDING_CACHE` with
   a per-thread override) — given a hoisted IR, does the runtime cache
   `PerEvalRef` per scope or evaluate it on-demand each time. The **cache A/B
   gate** flips this in-process and asserts identical trajectories: proves the
   cache returns exactly what on-demand evaluation would.

Together these two flips bracket the whole change: hoisting (compile) and
caching (runtime) are each independently proven byte-identical to the
un-optimized path. That is the load-bearing safety mechanism — default-off means
a regression can never reach a user before the flip, and the in-process A/B
means the proof runs in CI on every commit. **`run_id` consequence**:
`CAMDL_LICM=on` changes the emitted IR, so it re-keys `run_id` (the emitted IR
is identity — `normalize_for_hash` strips only `output.format`/`time_semantics`,
`resolve.rs:89`), exactly as toggling `constant_fold` does. The default-off
rollout means zero churn in trajectory _content_; the identity hash itself
re-keys at the 0.19 schema bump and again at the flip (see "Run identity").

### Gates

Mirror the sparse-fold / constant-fold rollout (land behind a flag, prove, then
flip). **Byte-identity throughout means the trajectory / loglik**, not
diagnostic side effects: caching evaluates an invariant subtree once instead of
~490×, so `eval_stats` counters (div-by-zero, pow-nan) and table-OOB records
fire fewer times. This is the same property the per-step `BindingCache` already
has, and `gate_binding_cache_ab.rs` already asserts trajectory identity, not
counter counts.

- **Forward byte-identity**: copy `gate_constant_fold_ab.rs` — compile both ways
  (LICM off / on), assert the simulated trajectory is byte-identical under every
  backend, with a non-vacuity guard that the pass hoisted ≥1 transcendental
  subtree. **Include a model with `dt` in a rate** (gh#54) so the `Dt`-variant
  classification is exercised — a `Dt` mis-hoisted as invariant would diverge at
  a clipped substep boundary and fail this gate. The fixture is a small
  **inline-table** model compiled both ways **offline** into committed
  `_off`/`_on` IR JSON (exactly how `gate_constant_fold_ab.rs` regenerates its
  fixtures) — the `read()`-loaded kernel cannot be an in-test fixture (no tables
  on disk) and doesn't need to be; it rides the performance-acceptance bench.
  **Also include a guarded-`Cond` model** where LICM hoists an invariant factor
  out of a guarded branch (the `if N[q]>0 then …·I[q]/N[q] else 0` FOI idiom),
  ideally with a hoisted subtree that is non-finite when the guard is false
  (e.g. a distance-table cell of 0 under an `I[q]>0` guard). Hoisting moves
  _when_ that subtree is evaluated (out of the lazy branch, into the per-eval
  prologue), so the gate must pin that the preserved `Cond` still gates the
  value identically and the unused non-finite never reaches the trajectory —
  converting the speculative-hoist soundness argument from reasoned to tested.
- **Staged-scratch A/B** (`gate_licm_ab.rs` §4): on the hoisted (ON) model,
  stage the per-eval scratch at a fixed (state, θ) and assert
  `eval(Some(scratch)) == eval(None)` (on-demand) bitwise for every rate /
  rate_grad / ode-derivative expression. This isolates the runtime read
  mechanism from hoist soundness (§2-3) and end-to-end identity (§2, which
  simulates the ON model — `run_ode` stages the scratch — byte-identical to the
  fully-inlined OFF model). Non-vacuity: the ON model has non-empty
  `per_eval_bindings`, so the scratch is non-empty and the rate surface carries
  exercised `PerEvalRef` nodes.
- **Inference producer A/B** (`gate_licm_inference_producer_byte_identical`):
  the inference-path analogue of §2. PF / IF2 / PGAS / PMMH all advance
  particles via `ProcessModel::step` → `chain_binomial::step_one` (the one
  shared producer seam). Stepping the ON model with its staged scratch and the
  OFF model on-demand under the same seed must yield byte-identical particle
  counts AND flow accumulators across the window — a wrong-θ stage (the IF2
  silent-wrong risk) would change the draws and diverge. Counts are integer, so
  this is exact. The full inference loglik is a deterministic function of these
  producer states plus per_eval-free observation scoring, so its byte-identity
  follows.
- **PGAS loglik A/B** (`gate_licm_pgas_loglik_byte_identical`): the result-level
  standing gate. Drives all three PGAS surfaces that carry `PerEvalRef` — the
  CSMC producer (`simulate_reference_on_grid` → `step_one`), the transition
  density (`complete_data_loglik` → `log_transition_density_substep`), and the
  NUTS gradient (`complete_data_loglik_grad`) — on the ON and OFF fixtures at
  the same seed, and asserts the complete-data log-likelihood AND its full
  gradient are byte-identical. This is the permanent regression guard that
  flipping `--licm` cannot move a PGAS fit's numbers, on a self-contained
  fixture, fast and in-tree.
- **ODE-loglik under rayon** (no silent gap): both existing trajectory gates
  call `OdeSim::run` directly, never the inference seam. A gate that flips
  `CAMDL_LICM` and asserts an identical loglik from `compute_ode_loglik` under a
  multi-θ rayon `par_iter` (MH/PMMH/nlopt/profile, `pmmh.rs:566`,
  `nlopt_stage.rs:162`, `profile.rs:1209`) pins the parallel path. Under Design
  C the scratch is owned/lent per `run_ode` call (no thread-local to alias), so
  this is a belt-and-braces regression guard rather than a correctness crux.
- **Gradient correctness**: `gradient_check*.rs` must stay green with the pass
  on. Add a **value** check: evaluate `rate_grad` LICM-off vs LICM-on at a fixed
  (state, θ) and assert equality (the _serialized_ `rate_grad` necessarily
  differs — it contains `PerEvalRef` — so this is evaluated-value equality plus
  a serialized non-vacuity check, not an IR byte-diff). Add a model with
  **param- valued overdispersion σ² and a param-dependent obs-likelihood arg**,
  pass ON, asserting those gradients are unchanged — the direct regression for
  the scoping invariant (those surfaces must stay `PerEvalRef`-free).
- **Keystone validation**: a unit test feeding a `per_eval_binding` body that
  references state / time / `dt` / forcing / `BindingRef` and asserting the
  constructor rejects it (Class B depends on this invariant holding).
- **Run identity**: the hash-distinctness test from "Run identity" above — a
  non-empty `per_eval_bindings` must change `run_id`.
- **Performance acceptance**: in-model ms/eval approaches the precomputed 9.5 ms
  baseline (residual = one kernel build per eval vs zero).
- **OCaml**: pin the hoist count + the term-count (the `Reduce` over
  `PerEvalRef` vs inlined) in `test_compiler.ml`, as `constant_fold` does.

Rollout: land the pass **default-off** (env opt-in, e.g. `CAMDL_LICM`), prove
all gates, then flip default-on with the golden regen
(`make update-golden && make
update-expected`) as its own reviewed, human-loop
commit — a golden diff is never collateral.

## Phasing

1. **IR node + pass + forward path.** Schema 0.19, the LICM pass, the Rust
   runtime (`PerEvalRef` eval + the staged `EvalCtx.per_eval` scratch + the
   on-demand fallback), staged at the ODE and forward-simulate θ-span entries.
   No gradient-consumer _logic_ changes are needed (the pass is scoped to the
   `eval_resolved` surfaces); the exhaustive gradient consumers
   (`eval_resolved_deriv`, `eval_expr_deriv`, `collect_param_refs`) get the
   mechanical `unreachable!()` arm from Class A to compile, but no
   differentiation behaviour changes. Nails the benchmark. Default-off. The
   on-demand fallback makes this safe on the inference backends even before they
   stage a scratch.
2. **Stochastic backends (done).** Staged scratch threaded through the PF / IF2
   / PGAS / PMMH producers at their θ-stable boundaries (filter-global for
   PF/PMMH, per-particle for IF2, per-sweep/per-call for PGAS). Gated by the
   existing inference oracles (LICM-off byte-identity) plus a new producer A/B
   (`gate_licm_inference_producer_byte_identical`): ON staged vs OFF on-demand
   through `step_one` yields byte-identical counts + flow accumulators.
3. Flip default-on; golden regen. Precede with an opt-in `--licm` flag wired
   into the run identity (the `config` level), so `camdl fit` can pick LICM up
   before the flip — today `CAMDL_LICM` is a no-op through fit (its IR cache
   keys on source + camdlc hash, not the flag).
4. (Optional) strength-reduction peephole; flat-eval per-eval tape.

## Risks, non-goals, open questions

- **Per-particle staging (done).** IF2 computes the scratch from exactly that
  particle's θ. Because the scratch is owned/lent data (not ambient thread-local
  state), this is a local `Vec` inside the per-particle closure — there is no
  scope to mis-bracket. The producer A/B and the on-demand fallback bound the
  blast radius: a missing scratch costs performance, never correctness.
- **Bitwise CSE** is a correctness requirement, not an optimization detail (the
  `-0.0` Reduce seed). The CSE key must be a bitwise expr hash.
- **CSE cost threshold** (when a subtree is worth hoisting) is a heuristic;
  start conservative (transcendental / `Pow` / `Reduce` present, or ≥ N nodes)
  and tune against the cost-report.
- **Non-goals**: param-only **overdispersion** σ² and param-dependent **obs**
  expressions are not hoisted (they live on the `eval_resolved_deriv` /
  `collect_param_refs` surfaces, deliberately kept `PerEvalRef`-free). They stay
  inline and correct, just unoptimized — a missed optimization, not a gap.
  Param-free-but-invariant existing `bindings` likewise stay per-step-cached.
- **Diagnostics, not correctness**: a table-OOB inside a hoisted subtree is
  recorded once (on the first `PerEvalRef` eval). `eval_propensities` clears the
  per-thread OOB record before _each_ rate (`propensity.rs:647`), so a second
  rate that reads the cached subtree and gets a NaN may surface a generic
  `NumericalCollapse` instead of the named `SimError::TableLookup`. The
  trajectory is unaffected (NaN → −∞ either way); only the named-error quality
  degrades for cached subtrees. Acceptable, but worth noting against the "error
  messages are a feature" bar.
- **Forcing coefficients (gh#119) are covered, not a gap**: param-dependent
  forcing coefficients are differentiated into `rate_grad`, which is _evaluated_
  via `eval_resolved` (the explicit note at `resolved_expr.rs:681–684`), never
  re-differentiated. A `PerEvalRef` in a forcing-coef-bearing `rate_grad` term
  evaluates through the cache like any other — the same "evaluated, not
  re-differentiated" argument that covers σ² and obs.
- **Memory**: one n² per-eval scratch buffer per worker — same footprint as the
  precomputed matrix the in-model form replaces.
