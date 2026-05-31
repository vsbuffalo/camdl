# Total-input-hash CAS: one typed input surface per command, the type IS the cache key

**Date:** 2026-05-31
**Status:** Draft — design proposal. §8 reproduction verified (red→green);
Phase B landed (gh#142), Phase C (the `RunInputs` enum) pending review.
**Supersedes/absorbs:** the `CasEntry` ADT sketch in
[`2026-05-31-run-system-unification-map.md`](2026-05-31-run-system-unification-map.md)
§4 (this is that idea, sharpened around the hashing invariant).
**Class:** code-vs-code (cache-key incompleteness → latent silent-wrong-answer)
+ architecture (consolidate the input surface across commands).
**Stakes:** a cache that returns the wrong run's output is the
silent-wrong-answer class this software cannot tolerate. The current keys are
hand-maintained subsets and have already drifted (see §3). This is priority-
zero correctness, not ergonomics.

---

## 0. The shape, in one diagram

The whole design is: **every command funnels its inputs (CLI args *and*/or
TOML) into one typed value; that value IS the cache key; everything else
(path, run.json, cache-hit) is a function of it.**

```
  the ONLY places args/TOML are parsed          one complete typed value      derived, never hand-written
  ────────────────────────────────────          ───────────────────────       ──────────────────────────
  simulate CLI args ┐                                                        ┌─ input_hash(r)  → cache key
  batch TOML        ├─→  resolve(...) ─────────→   RunInputs (per command) ──┼─ cas_path(r)    → readable path
  fit.toml          ┘     (ONE funnel/command)     = exactly the inputs to   └─ run.json(r)    → metadata
  pfilter/profile  ─┘                                the pure function f       (one structural traversal each)
```

If `RunInputs` contains *exactly* the inputs to `f`, and the key is
`H(RunInputs)`, then soundness is **structural**: you cannot forget an input,
because there is no field list to forget — you hash the whole value.

---

## 1. The principle: the cache key must be a *total* function of the run's inputs

A run is `output = f(inputs; seed)`, `f` deterministic. So the output is never
hashed (it's reproducible); the **cache key is `H(all inputs)`**. Two standard
framings:

- **Sound caching.** `key = H(all inputs)` where the key function is **total
  over the input set** — *every* input the simulator reads is in the key. The
  failure mode — *"if we miss an input we get clashes"* — is exactly a
  **non-total / unsound key**: the key is a function of a *subset* of inputs,
  so two genuinely different runs collide and the cache serves the wrong one.
- **"The type IS the key"** (correct-by-construction hashing /
  derive-don't-handwrite). Instead of a function that *lists* which fields to
  hash (where one can be forgotten), hash the **whole input value** via a
  derived/structural traversal. Then *"add a field" → "it's automatically in
  the key,"* with no separate hash site to update.

**The invariant, stated sharply:**

> There is one type per command that contains **exactly** the inputs to its
> pure function `f`, and the cache key is a **structural hash of that whole
> value**. Adding/removing an input is a change to the type — which the
> compiler propagates to the key automatically. **No hand-written field list
> anywhere.**

That is both the *consolidation* and the *validation* of the
inputs→pure-`f`→CAS idea: if the inputs type is complete and the key is
`H(whole type)`, soundness is structural, not vigilance.

---

## 2. Auto-invalidation falls out (IR, params, seed, config, …)

Because the key is `H(all inputs)`:
- change the **IR** → key changes → recompute (IR affects everything);
- change a **param / scenario delta** → key changes for that subtree, siblings reused;
- change **any config knob** (backend, dt, t_end, output_dt, a future `atol`)
  → key changes (it's *in* the hashed value);
- add a **seed** → no existing key changes (pure addition).

"Auto cache invalidation when IR / params / seeds / config change" is not extra
code — it's the definition of `H(all inputs)`, *provided the inputs type is
complete*. That proviso is the whole game.

---

## 3. The code violates this today — three hand-written field lists (verified)

Each `*_hash` is a manually-enumerated **subset** of inputs, separately
maintained. Verified against source 2026-05-31 (HEAD `fbedb5a`):

- **`sim_hash(model_hash, params_canonical, backend, dt)`**
  (`hashing.rs:65`) — folds `model_hash, params, backend, dt, version`. But the
  thing it represents, `SimConfig` (`sim/config.rs`), *also* has
  **`t_start`, `t_end`, `output_dt`** — **not in the key.** → change `t_end`
  or `output_dt`, same key, **stale-hit collision.** *(Latent bug; §"Verified"
  will pin it with a red repro.)*
- **`scen_hash(enable, disable, params)`** (`hashing.rs:88`) — a *different*
  hand-listed set.
- **`fit_stage_hash(model, observations, estimate, fixed, simplex_groups,
  stage_name, stage, seed)`** (`fit/provenance.rs:303`) — an **8-arg**
  hand-listed set, separately maintained, reading data files inline.

Three lists, three maintainers, no shared structure → guaranteed to diverge,
and the `t_end`/`output_dt` gap shows they already have. **Every arg list is a
place an input can be silently forgotten = a place clashes are born.** That is
the smell, made concrete.

Corroborating: `SimulateInputs` (the would-be consolidated type,
`cas/sim_inputs.rs:26`) is constructed in **two** places independently
(`batch.rs:825`, `main.rs:1299` via `prepare_cas_ctx`) — two hand-rolled
args/TOML→inputs mappings, which is how `simulate` and `batch` drift.

---

## 4. The target: one `RunInputs` **enum** per command, hashed as one value

### 4.1 It must be an enum, NOT a trait — and that distinction is the whole point

Today there is already a *unified interface*: `trait CasInputs`
(`cas/typed.rs:94`) with `fn content_hash(&self) -> ContentHash`, implemented
by **five** types — `SimulateInputs`, `FitInputs`, `StageInputs`,
`ProfileInputs`, `SurveyInputs` (verified 2026-05-31). **A trait does not give
totality.** "Implement `content_hash` however you like" is *exactly* the
hand-written-subset hole, five times over — and one of them already lost
`t_end` (gh#142). The current trait unifies the *interface* (you can be
hashed) and leaves *completeness* per-impl (what you hash is your business).
That gap is the bug.

So the target is **one enum, hashed as one value** — not a trait each command
implements:

```rust
/// THE complete input to one deterministic run `f(inputs) -> output`.
/// Built ONLY by the per-command resolver (CLI args | TOML → this).
/// Hashed as ONE structural value — there is NO per-variant hash method.
enum RunInputs {
    Simulate { model: ModelInput, config: SimConfig, scenario: ScenarioDelta,
               params: ParamSet, seed: Seed },
    FitStage { model: ModelInput, data: DataInputs, estimate: EstimateSet,
               fixed: FixedSet, config: SimConfig, stage: StageConfig, seed: Seed },
    Pfilter  { model: ModelInput, data: DataInputs, params: ParamSet,
               config: SimConfig, particles: usize, seed: Seed },
    Profile  { model: ModelInput, data: DataInputs, estimate: EstimateSet,
               fixed: FixedSet, focal: FocalAxes, config: SimConfig,
               stage: StageConfig, seed: Seed },
    Eval     { model: ModelInput, params: ParamSet, exprs: Vec<String>, times: TimeGrid },
}

/// ONE hash, total over the whole value. No field list anywhere.
fn input_hash(r: &RunInputs) -> ContentHash;   // structural traversal of the value
```

`content_hash` stops being a *method you implement* (five holes) and becomes
*one function of the whole value* (zero holes). Add a field to a variant → in
the key, free, compiler-propagated. Drop `t_end` → impossible: it's a field of
`SimConfig`, which is a field of the variant.

**`config: SimConfig` is the WHOLE bundle** (backend, dt, t_start, t_end,
output_dt, future atol/integrator…), not four cherry-picked fields — which is
precisely what fixes gh#142 and makes the `config-<hash>` path segment
complete (§6).

### 4.2 "Hash the whole value" = a *total structural* hash, NOT literal `#[derive(Hash)]`

Honesty about the implementation: it is **one explicitly-total structural
hash**, derive-*style*, not the stdlib derive. Three reasons the macro can't
be used blindly (all already true in the current code, just scattered):
- **`f64` fields** (dt, params, t_end) aren't `std::hash::Hash` (no `Eq`) —
  hash via `.to_bits()` (the current code does this for `dt`); pick a NaN/`-0.0`
  canonicalization once, centrally.
- **maps** (params, data streams) must be **key-sorted** before hashing or the
  digest is nondeterministic (the current `canonical_params` does this for one
  case; the enum applies it uniformly).
- so the win is **one** total hash over the enum vs **five** hand-written
  methods — and ideally a small derive-macro (`#[derive(InputHash)]`) that
  hashes every field so "add a field, forget the hash" is a compile concern,
  not a review concern. (Macro = stretch goal; one hand-written total fold over
  the enum is the floor and already eliminates the five-holes problem.)

### 4.3 The escape hatch is *typed*: input vs display (this is the dangerous direction)

Not every field of a config is a *semantic* input. Two kinds of non-inputs
must be **excluded from the hash**:
1. **display / provenance** — the model *path*, `--label`, timestamps. If these
   were hashed, moving a file or relabelling would bust the cache.
2. **output-shaping that may not change the trajectory** — e.g. `output_dt`
   (a per-field judgment; §8/gh#142).

⚠️ **Excluding is the *under-invalidation* direction — the silent-wrong-answer
risk.** Folding more *in* is monotone-safe (worst case: a spurious miss →
recompute → correct). Leaving something *out* that should be in → serve the
wrong run. So the exclusion set is the one place this design can go wrong, and
it must be **a typed, reviewed-once decision, not a per-impl omission**:
`ModelInput` carries the IR *content* (hashed) and the path as a *separate,
explicitly-`#[hash_skip]` (or sibling `Display`-struct) field that is never
hashed*. The rule: **everything in the input type is hashed; non-inputs live
in a sibling display/provenance type.** One exclusion list, reviewed once —
vs today's five implicit ones.

### 4.4 One funnel per front-end — TOML and CLI both resolve INTO `RunInputs`

The unification is real only if **every front-end produces the same value**:

```
  the ONLY places args/TOML are parsed            ONE typed value
  ────────────────────────────────────            ───────────────
  simulate CLI args ──→ resolve_simulate(args)  ─┐
  batch TOML        ──→ resolve_batch(toml)      ─┤
  fit.toml          ──→ resolve_fit(toml)        ─┼──→  Vec<RunInputs>   (a sweep / ensemble = many)
  pfilter CLI       ──→ resolve_pfilter(args)    ─┤      one RunInputs    (a single run)
  profile/eval      ──→ resolve_*(args)          ─┘
                                                       └──→ input_hash / cas_path / run.json
```

TOML is **not** a parallel input type — it is one of several parsers feeding
the single funnel. A batch sweep or a multi-seed run is `Vec<RunInputs>` (each
cell a complete value); a lone `camdl simulate` is one `RunInputs`. This is
what makes it a *true* unified input rather than today's two-constructor drift
(`SimulateInputs` is built independently in `batch.rs:825` and
`main.rs:1299` — two hand-rolled args/TOML→inputs mappings, the seam where
simulate and batch diverge).

---

## 5. The one necessary escape hatch: input vs display

"Hash the whole value" is *not* a blind `#[derive(Hash)]` — because not every
field is a *semantic* input to `f`. Two kinds of non-inputs must be excluded:

1. **Display / provenance** — the model *path*, labels, timestamps. Must NOT
   be in the key (else moving a file invalidates the cache). These live in a
   **separate display/provenance type that is never hashed**.
2. **Output-shaping that may not change the computed trajectory** — e.g.
   `output_dt` changes which timepoints are *emitted*; whether that's a
   semantic input or a view concern is a per-field judgment (§"Verified" #2).

So the rule: **the input-bearing type contains exactly the semantic inputs and
is hashed whole; display/provenance is a sibling type, never hashed.** That
input-vs-display split is the real design work — it's the same hash-vs-display
split the *path segments* already need (readable label + authoritative hash),
applied at the type level.

---

## 6. Path = the same inputs, rendered (readable, complete, stable shape)

The CAS path is `H(inputs)` rendered as a layered, browsable chain — each
segment is **one bundle hashed as a unit**, named by what the bundle *is*, not
one-knob-per-segment (so the number of config knobs never changes the tree
shape):

```
results/
  <model_stem>-<model_hash8>/         # Model    = H(structural IR)
    <backend>-<config_hash8>/         # Config   = H(WHOLE SimConfig)   ← fixes the dt/t_end/output_dt gap
      <scenario_slug>-<scen_hash8>/   # Scenario = H(enable+disable+param delta)
        seed_<n>/                     # Seed     (atomic; the value IS the readable key)
          traj.tsv  run.json  obs/<obs_hash>-<obs_seed>/<stream>.tsv
```

- **hash** = Merkle fold up this chain (each layer folds in its bundle hash) →
  scoped reuse + auto-invalidation.
- **path** = the chain rendered: `<readable-label>-<bundle_hash8>` per segment;
  the *full decoded bundle* lives in `run.json` so `camdl show <run>` expands
  `config-a1b2c3d4` → `backend=chain_binomial dt=1 t_end=4388 output_dt=…`.
- **readability tradeoff (your call, recorded):** `config-<hash>` is less
  `ls`-readable than `chain_binomial.dt1`, but a readable-but-*incomplete* key
  is a silent-wrong-answer risk. Lean: `<backend>-<config_hash8>` — backend
  name as the readable prefix (glanceable), full-config hash as the
  authoritative suffix (complete). Best of both, mirrors the existing
  `<slug>-<hash8>` pattern everywhere else.

This is the layered expression of §4: each path layer = one input bundle,
hashed whole. The `RunKind`-style human-readable structure is *preserved and
improved* — the path now spells out the full input lineage, not two opaque
hashes.

---

## 7. Relationship to the rest of the run-system unification

The `CasEntry`/`Inner`/`Leaf` recursive tree (unification-map §4) is the
*grouping* view (how N seeds / scenarios nest); **this doc is the *key* view
(what's hashed).** They're the same structure from two ends and must agree:
each `Inner(dim)` layer corresponds to one `Binding`/bundle in the hash chain;
each `Leaf` is a complete `RunInputs`. A later proposal unifies the *grouping*
(`ReplicateSet`/batch-grid/fit-ad-hoc → one tree); this proposal unifies the
*key* (three hand-lists → one structural hash). Land the key invariant first —
it's the correctness floor everything else stands on.

---

## 8. Verified — the §3 collision is real (reproduced 2026-05-31, red→green)

The §3 `simulation`-block omission is reproduced concretely and pinned by
`rust/crates/cli/tests/cas_tend_in_key.rs`.

**Mechanism (verified by dumping trajectory rows).** Output row count is
governed solely by `output.times.regular.end` (80 in the golden) —
`simulation.t_end` does *not* set the horizon, so t_end ∈ {40, 80, 160} all
emit 81 rows ending at t=80. But t_end=40 stops the integrator at t=40 and the
writer emits the *frozen* t=40 state for t=41..80 (incidence cols zeroed), so
its bytes differ from t_end=80 (the real decline). `t_end ≥ output.end` is a
no-op (80 vs 160 byte-identical). The test therefore uses **40 vs 80** — the
pair that genuinely differs in output; 80 vs 160 would pass for the wrong
reason. (The silent frozen tail when `t_end < output.end` is a separate
concern, gh#143 — not a horizon cap.)

**Reproduction.** Two IRs, same filename (`model.ir.json` — identical path
stem, so only the hash can separate them), differing ONLY in
`simulation.t_end` (40 vs 80), into the same `--cas` dir:

- **pre-fix:** both collapse to one CAS entry (`model-3b6f3f33/…`); the second
  run is a `cache hit` served the first's trajectory → `Found 1 dir(s),
  left:1 right:2`.
- **post-fix:** two distinct entries (`model-26240390` / `model-e2217ead`),
  distinct trajectory bytes.

**Root cause (source-confirmed).** `sim_hash` folds only `model_hash, params,
backend, dt, version`; `model_hash` hashed only an allowlist of *structural*
keys — the `simulation` block (holding `t_end`) was in neither. This is
exactly the §1 totality violation: a hand-listed subset silently omits a real
input. Phase B (§10) folds `simulation` + `time_unit` into `model_hash`,
closing gh#142.

**Test-design note.** The collision is invisible if the two models have
*different* filenames — the path stem (`<model_stem>-<hash>`) then differs even
when the hash is identical, yielding two dirs that *look* fine while sharing a
key. The regression test MUST use the same filename, or it passes vacuously.
This is itself evidence for §6: the *path* must derive from the *hash*, not
from an incidental filename.

## 9. Decisions for the maintainer

1. **Path config segment:** `<backend>-<config_hash8>` (lean) vs pure
   `config-<config_hash8>` vs readable `chain_binomial.dt1-<hash>`?
2. **`output_dt`:** semantic input (in key) or view concern (display only)?
   (Test #2 informs; final call is yours.)
3. **Scope/sequencing:** land the *key* invariant (this doc) as its own
   refactor first, before the *grouping* unification and before CAS-default
   output rides on it? (Lean: yes — key soundness is the floor.)
4. **`fit_stage_hash` migration:** fold fit onto the same `RunInputs`/
   `input_hash` machinery now, or in a follow-up? (It's inference-adjacent →
   careful, but it's one of the three drifting lists this is meant to kill.)

---

## 10. Phasing & execution (the monotone-safety split)

The decisive engineering fact for *how* to land this: **folding inputs INTO a
hash is monotone-safe; EXCLUDING is not.** Adding an input can only over-
invalidate (spurious miss → recompute → still correct). Excluding an input
that should be in the key under-invalidates → serves the wrong run. So the
phases are ordered by which direction they move, and which can run unattended.

### Phase B — complete the key (MONOTONE-SAFE, lands autonomously) — closes gh#142
Fold the missing `simulation` block (`t_end`, `t_start`, `output_dt`) into the
simulate cache key. This is purely *additive* to the hash, so it **cannot**
introduce a wrong-answer — the worst case is a recompute that would have been a
hit. TDD: the gh#142 reproduction (two models differ only in `t_end`, same
filename + same `--cas` dir → must become 2 distinct CAS entries) is the red
test. Gate: `ir/expected/*.tsv` byte-identical (dynamics unchanged proves this
is a *key* change, not a *behaviour* change); `determinism_pin` +
`progress_tick_invariance` green; only hash-value goldens (`golden_hash_sim_hash`
etc.) legitimately move — and they move because the input set legitimately grew.
Bounded, safe to land without a human watching the goldens, *because the failure
direction is recompute-not-wrong.*

### Phase C — the `RunInputs` enum + total structural hash (EXCLUSION-RISK, needs review)
The full §4 type. Its risk is §4.3's exclusion set — every `#[hash_skip]` is an
under-invalidation hazard (the wrong-answer direction). It also regenerates the
entire CAS golden surface and touches inference-adjacent `fit` (`StageInputs` /
`fit_stage_hash`). **This does NOT land unattended.** It's staged as its own
determinism-gated change with the exclusion list reviewed field-by-field by the
maintainer. Phase B is forward-compatible with it (Phase C subsumes the
hand-fold of Phase B into the structural hash).

### Phase D — grouping unification (separate; the unification-map doc)
`ReplicateSet`/batch-grid/fit-ad-hoc → one `Inner`/`Leaf` tree. Orthogonal to
the *key* (this doc); sequenced after.

**Tonight's autonomous scope = Phase B only.** Phase C/D are staged for review:
the cache-key *layer* of public-health software is not rewritten unattended
when the failure mode is silent-wrong-answer. Phase B kills the actual filed
bug (gh#142) by the one move that structurally cannot make things worse.
