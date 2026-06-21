# Loop-invariant code motion: per-eval staging of param/table-only subexpressions

Date: 2026-06-20 Status: Phase 1 implemented (ODE/forward; default-off) Issue:
gh#272 Schema: 0.18 → 0.19 (adds `per_eval_bindings` + `Expr::PerEvalRef`)

Implemented: ea4300cc (step 1.1 — IR substrate + on-demand eval), 8cfe9b8a (step
1.2 — the OCaml LICM pass), de71c112 (step 1.3 — per-eval cache tier +
EvalScope). Measured on the real MRE kernel: ~4.2× faster per ODE step, bringing
the in-model fittable-γ kernel to ~precomputed-matrix speed. Still opt-in
(`CAMDL_LICM`). Remaining: Phase 2 (stochastic per-particle EvalScope — pure
performance; all methods are already _correct_ via the on-demand fallback), the
default-on flip (a deliberate, run-id-re-keying release decision), and the
flat-eval tape / strength-reduction follow-ons.

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
2. **Per-eval cache tier**: a second generation on the existing thread-local
   `BindingCache` (`rust/crates/sim/src/resolved_expr.rs`). The existing
   generation bumps per `eval_propensities` (per step); the new generation bumps
   per **`EvalScope`** (per θ-stable boundary). `PerEvalRef` memoizes against
   the per-eval generation; `BindingRef` keeps memoizing against the per-step
   one. The two tiers index **disjoint slot spaces** (`bindings` vs
   `per_eval_bindings`), so the per-eval tier needs its **own**
   `val`/`stamp`/`gen` **and its own `active` flag** in the `BindingCache`
   struct — not a shared buffer (slot `k` would collide) and **not the shared
   `active`**. The shared `active` is the trap: the per-step `CacheScope` is
   dropped **per RK stage** on the ODE path (`ode.rs:89`, "dropped at the end of
   this stage"), and `CacheScope::Drop` sets `active = false`
   (`resolved_expr.rs:404`). Reusing that one flag, the first nested stage's
   `Drop` deactivates per-eval caching for the rest of the trajectory (no
   speedup) — and "fixing" it by not clearing the flag breaks the per-step
   tier's own invalidation (silent-wrong `N[p]`). `CacheScope::enter`'s per-step
   resize (`resolved_expr.rs:382`, `if c.val.len() != n_bindings { … }`) must
   likewise leave the per-eval vectors untouched. Same struct, two fully
   independent tiers.

   **Borrow discipline.** A per-eval body references earlier per-eval bindings
   (`Z_p = Σ_r PerEvalRef(Wnum_p_r)`), so `PerEvalRef` eval must **release the
   `RefCell` borrow before recursing** into an earlier slot — exactly as the
   `BindingRef` arm does (`resolved_expr.rs:597–621`: borrow scoped to the hit
   check, recursion only after). Holding `borrow_mut()` across the recursion
   panics (`BorrowMutError`) on the `Z_p → Wnum` shape — which _is_ the
   benchmark — so the keystone test must exercise a per-eval body that
   references an earlier one.
3. **Safe fallback (no active scope ⇒ on-demand eval).** When no `EvalScope` is
   active, `PerEvalRef` evaluates its body directly — exactly as `BindingRef`
   already does on a cache miss (`resolved_expr.rs:608`,
   `None => eval_resolved(
   &bindings[slot], ctx)`). So `PerEvalRef` is
   **correct on every backend regardless of whether that backend has wired an
   `EvalScope` yet** — it is only _faster_ where a scope is present. The phase
   split (below) is safe by construction, not by "don't use it there."
4. **Flat evaluator** (`flat_eval.rs`, opt-in `CAMDL_EVAL_FLAT`) is a
   **first-class sub-deliverable, not a one-line arm.** It does _not_ use the
   thread-local `BindingCache`; it has its own `FlatVm` with a `binding_progs`
   tape and a `FlatCache` (`val`/`stamp`/`gen`/`active`) threaded through
   `run`'s signature and the `FLAT_STATE` thread-local (`propensity.rs`).
   `PerEvalRef` needs the parallel build-out: a second `per_eval_progs` tape on
   `FlatVm` (`build` gains a third arg), a **second `FlatCache` tier** (disjoint
   slots, per-eval generation) plumbed through `run`/`eval_flat` and
   `FLAT_STATE`, an `Op::PerEval(slot)` mirroring `Op::Binding`
   (`flat_eval.rs:254`), and `EvalScope::enter` must bump the per-eval
   generation in **both** the `BINDING_CACHE` and `FLAT_STATE` thread-locals (or
   the flat path silently never reuses — correct via fallback, but no speedup).
5. **`EvalScope`** RAII (sibling of `CacheScope`), entered at each backend's
   θ-stable boundary:
   - ODE: **at the top of `OdeSim::run` (`ode.rs:508`), wrapping the
     `while next_stop { stepper.advance(…) }` driver loop (`:602`) — _not_
     inside `ode_derivs` / next to the per-stage `CacheScope` (`ode.rs:89`).** θ
     is fixed for the whole `run_ode` call; that is the unique θ-stable span.
     Placing it where the per-stage `CacheScope` lives (the obvious copy-paste)
     bumps the per-eval generation ~2000× per integration → zero cross-step
     reuse → the 3.77× evaporates. This one site covers **both** ODE inference
     (`compute_ode_loglik → OdeSim::run`, `runner.rs:775/792`) and forward ODE
     simulate, since both route through `OdeSim::run`.
   - Forward simulate (chain_binomial/gillespie): once per run.
   - PF / IF2: **inside** each particle's closure body, _per particle_. The
     rayon parallel region opens at `particle_filter.rs:274` (`par_iter_mut`);
     the per-particle closure body begins at `:278`; the substep loop is `:284`.
     The `EvalScope` must be entered **inside that closure (per particle)**, and
     `EvalScope::enter` must **bump the per-eval generation** (mirroring
     `CacheScope::enter`, `resolved_expr.rs:387`), so when a rayon worker
     processes particle 3 then particle 17, particle 3's cached kernel is
     invalidated before particle 17 runs. This is load-bearing for IF2, where
     each particle carries a distinct perturbed θ (`if2.rs:444`): a scope
     entered _around_ the parallel region (on the calling thread) would leave
     each worker holding a stale per-eval cache from a different particle's θ →
     silent-wrong gradient. Entered per particle, the kernel is constant across
     that particle's interval substeps (the win) and correct across the
     per-particle θ change (the invariant).
   - PGAS: per CSMC sweep (θ fixed during the conditional filter, perturbed by
     NUTS between sweeps).

### Runtime invalidation: staged per-eval scratch (Design C — the chosen architecture)

The thread-local cache + RAII `EvalScope` above (call it Design A) is what Phase
1 shipped, and it works (≈4.2× measured). But three independent design reviews
converged on a cleaner architecture for the runtime half, and it is what the
remaining runtime work should adopt. The compiler half — the LICM pass, the IR
nodes, the keystone validation — is **unchanged**; only the way the runtime
evaluates `PerEvalRef` changes.

**The problem with Design A.** The per-eval cache is shared thread-local state
invalidated by a generation a per-backend `EvalScope` bumps. Correctness then
depends on each backend placing the scope at a θ-stable boundary — and the
inference loops nest θ differently (IF2 perturbs θ _per particle_). A scope
entered around the parallel region instead of per particle silently serves one
particle's kernel to another's θ. The design carries a correctness obligation as
placement discipline rather than as a type, and it accretes a second of every
cache primitive (generation, `active`, override, hit counter, plus a separate
flat-VM tier).

**Design C — a staged prologue, threaded as data.** Compute the per-eval
bindings once for the θ-stable span into an owned `Vec<f64>` scratch, and thread
it as a borrow on `EvalCtx` (`per_eval: Option<&[f64]>`, the exact sibling of
the existing `int_float_override: Option<&[f64]>`). `PerEvalRef(slot)` becomes
`ctx.per_eval[slot]` — a slice read — falling through to on-demand eval when
`None` (byte-identical, so an un-staged path is correct, just unamortized).

This is **correct by construction**: the scratch is owned/lent, not ambient, so
there is no shared mutable cache to alias across particles — the value is
structurally bound to the θ it was computed at, and the `if2.rs` per-particle
case is just "one scratch per particle." A missed wiring is a _compile error_ (a
typed field), not a silent stale read. It **deletes** Design A's second cache
tier, the `EvalScope` RAII, the borrow-before-recurse dance, and the flat-VM
tier. And it is the extensible seam: the future `const` stage and the
gradient-path per-eval stage each compose as another owned scratch read by
index, where Design A would need a third and fourth thread-local tier.

(Design B — auto-invalidate by keying the cache on the param vector — was
rejected: IF2/PGAS/PMMH **mutate the θ buffer in place** (`if2.rs:391,480`;
`pmmh.rs:467`), so pointer-identity keying serves stale values, and
content-hashing has no existing identity to hang on and costs a per-eval hash.)

**Migration (the chosen plan).** Keep steps 1.1 (IR substrate) and 1.2 (the LICM
pass) verbatim. Replace step 1.3's thread-local tier + `EvalScope` with the
staged scratch: add `per_eval: Option<&[f64]>` to `EvalCtx` (~70 construction
sites, mostly `None`; only rate / rate_grad / ode-derivative eval sites carry
the scratch since `PerEvalRef` appears only there), a `PerEvalScratch` computed
at each θ-span entry, and thread it through `ode_derivs` (via the `OdeStepper`
stepper) and `eval_propensities`. The A/B gate transfers directly
(`scratch present vs absent` replaces `cache on vs off`). It is a real but
bounded refactor of the eval hot path — to be done as a careful, gated focused
pass, not rushed.

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
- **Per-eval cache A/B**: extend `gate_binding_cache_ab.rs` to the per-eval tier
  (cache on/off → identical trajectory). Assert hits > 0 on a **distinct**
  `take_per_eval_cache_hits` counter — the existing per-step counter is nonzero
  regardless, so reusing it makes the non-vacuity check vacuous. **Include an
  IF2 run with per-particle θ perturbation** so the per-particle scope reset is
  exercised — a scope entered _around_ the parallel region instead of per
  particle would alias stale θ and fail this gate.
- **ODE-loglik under rayon** (no silent gap): both existing gates call
  `OdeSim::run` directly, never the inference seam. The benchmark's own path —
  `compute_ode_loglik` run under a multi-θ rayon `par_iter`
  (MH/PMMH/nlopt/profile, `pmmh.rs:566`, `nlopt_stage.rs:162`,
  `profile.rs:1209`) — is where a mis-scoped thread-local `EvalScope` would
  alias a stale θ's kernel. Add a gate that flips `CAMDL_LICM` and asserts an
  identical loglik from `compute_ode_loglik` under that parallel context. Name
  `correlated_pf`/PMMH explicitly in the cache-gate matrix too (the fallback
  keeps them _correct_ without a scope, but "covered by the IF2 clause" is the
  exact hand-wave to avoid).
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
   runtime (`PerEvalRef` eval + per-eval cache tier + flat-eval `Op` + the
   on-demand fallback), and `EvalScope` for the ODE backend and forward
   simulate. No gradient-consumer _logic_ changes are needed (the pass is scoped
   to the `eval_resolved` surfaces); the exhaustive gradient consumers
   (`eval_resolved_deriv`, `eval_expr_deriv`, `collect_param_refs`) get the
   mechanical `unreachable!()` arm from Class A to compile, but no
   differentiation behaviour changes. Nails the benchmark. Default-off. The
   on-demand fallback makes this safe on the stochastic backends even before
   they wire a scope.
2. **Stochastic backends**: `EvalScope` inside the PF/IF2/PGAS per-particle (and
   CSMC) boundaries. Per-particle-interval reuse. Gated by the existing
   inference oracles.
3. Flip default-on; golden regen.
4. (Optional) strength-reduction peephole.

## Risks, non-goals, open questions

- **EvalScope boundary correctness** is the crux for the stochastic backends —
  the scope must bracket exactly a θ-stable span. ODE/forward (Phase 1) is
  trivial (one scope per trajectory). The PF/IF2/PGAS boundaries (Phase 2) need
  the exact call sites pinned and gated against the inference oracles before
  flipping on. The on-demand fallback bounds the blast radius: a missing or
  mis-scoped `EvalScope` costs performance, never correctness.
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
