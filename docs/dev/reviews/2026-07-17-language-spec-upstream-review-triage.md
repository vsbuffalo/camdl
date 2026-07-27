# Upstream language-spec review — verification triage

Date: 2026-07-17 Scope: `docs/camdl-language-spec.md` (v0.3-draft) Method: every
claim verified against the OCaml compiler + Rust backend, not the spec prose
alone. Classifications and evidence below are file-checked; the one code bug
(item 47) carries a reproduced command + output.

## Overall assessment

The review was produced by an agent with access to the **spec only**, not the
code. Its central thesis — "the spec is not semantically closed; identical
syntax means different things in different contexts" — is **overstated**. On
verification the compiler is consistently _more_ principled than the spec reads:
partial indexing hard-errors everywhere it isn't an index binder (E287), the
silent-marginalization case is explicitly guarded (E280), branch weights lower
deterministically, run-identity is a sound factored scheme, and negative-count
interventions hard-error. Several "direct contradictions" are the reviewer
reading a stale spec passage as live behavior, or conflating a documented
carve-out with the general rule.

That said, the review did real work. It surfaced **one genuine silent-wrong code
bug**, a cluster of **real spec-vs-code drifts** (whole stale sections), several
**error-quality / ergonomic code warts**, and a strong list of
**underspecified-but-defined behaviors** that should be written down before beta
so a modeler can predict them. Net: high value as a pre-beta punch list once the
misreads are filtered out.

Counts: ~1 code bug, ~6 real code/error-quality issues, ~16 stale/wrong spec
passages, ~9 document-the-behavior gaps, ~13 needs-Vince design calls, ~9
reviewer misreads.

---

## Tier 0 — Genuine code bug (fix before beta)

### Item 47 — default output window emits pre-`from` rows with scrambled time

`expander.ml:5284` sets the default trajectory start to `min 0.0 t_start`. The
inline comment shows this was a deliberate gh#143 choice to cover negative
`t_start` (from-before-origin) while "preserving the existing start=0 behaviour
for unanchored models." The consequence: **any model with `t_start > 0`**
(anchored `from` later than `origin`, or unanchored `from > 0`) and **no
explicit `output` block** gets `start = 0`, so the trajectory is emitted — and
the dynamics run — over `[0, t_end]` instead of the requested `[from, to]`.

Reproduction (minimal SIR, `origin = 2020-01-01`, `from = 2020-07-01`,
`to = 2020-07-12` → an 11-day window):

```
$ camdl simulate anchored_window.camdl --params p.toml --seed 1 --dates --stdout
t     date        S    I   R   flow_infection  flow_recovery
182   2020-07-01  990  10  0   0               0
1     2020-01-02  988  10  2   2               2
2     2020-01-03  988  10  2   0               0
...
193   2020-07-12  861  86  53  25              10
```

195 data rows for an 11-day window; time is **non-monotonic** (182, then
1,2,…,193); and the epidemic evolves across the whole origin→`to` span, not the
requested window. This is worse than "frozen initial-state rows" — the run
ignores `from` for the dynamics.

Class: code-vs-code. This is a facet of the reopened gh#143 (output window vs
sim window). Fix is Vince's call — `start = t_start` for `t_start ≥ 0` while
keeping the negative-`t_start` handling — because the `min 0.0` was intentional
and the interaction with the state-at-obs-time snapshot path (gh#143) needs
care.

---

## Tier 1 — Real code / error-quality issues worth fixing

| #   | Issue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Where                    | Fix                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 30  | Multi-value `read()` **value** columns are pure-positional (no name check): swapping the two value columns of `pop, init_sus : patch = read(...)` silently swaps the tables (0 warnings). Self-product tables (`patch × patch`, `age × age`) can't validate axis order — a column-swapped asymmetric kernel is silently transposed behind a non-discriminating W201. Dimension columns _are_ name-checked (E216/W201), so the blanket "positional" claim is wrong, but the value-column gap is real data corruption on exactly the national-scale demographic + spatial-kernel files. | `expander.ml` table load | name-check value columns against LHS names; hard axis-role check for self-product tables                           |
| 34b | `set` intervention requires the **mangled IR name** (`I_child_p1 = 500`); `I[child, p1] = 500` is `E001` syntax error. DSL exposing its own lowering — against "design the DSL for humans first."                                                                                                                                                                                                                                                                                                                                                                                     | `parser.mly:1067`        | accept indexed l-values (and stratum-expanding partial forms, as `transfer` already does), lower internally        |
| 34a | Data-derived level names are **not sanitized**: a level `"kano dala"` yields compartment `S_kano dala` (embedded space) that `set` then can't address at all. (Collision detection _does_ exist — E278 — so the reviewer's "no check" is wrong.)                                                                                                                                                                                                                                                                                                                                      | `expander.ml` mangling   | validate/sanitize data-derived levels; make E278 name the colliding tuples                                         |
| 17  | `log(x ≤ 0) → −∞` **silently**, inconsistent with its sibling `sqrt(neg)` which raises a typed `NumericalCollapse::SqrtNegative`; and no `+∞` guard on the final propensity. (The rest of the reviewer's "no invalid-value policy" is a misread — there is a strict default with typed collapses gated by `--allow-degenerate-rates`.)                                                                                                                                                                                                                                                | `types.rs` UnOp eval     | route `log(x≤0)` through the same domain-error path as `sqrt`; add an `is_finite` guard on the resolved propensity |
| 23  | `at_day` "exactly one fire per period regardless of dt" is false: fire times are `round(t/dt)` collected into a **dedup BTreeSet**, so when `dt ≥ period` (or two targets within < dt) fires silently merge; the strict `< 0.5·dt` is really `≤` at the exact midpoint.                                                                                                                                                                                                                                                                                                               | `time.rs:143`            | split the integrator step at agenda times instead of proximity-detecting; soften the §13.7 guarantee               |
| 19  | Gillespie silently evaluates `Expr::Dt` to the nominal `dt` (unrelated to event spacing) with no capability gate, while rk45 correctly hard-errors on `Expr::Dt`.                                                                                                                                                                                                                                                                                                                                                                                                                     | `gillespie.rs:209`       | gate `Expr::Dt` on Gillespie, or document that it means the nominal step                                           |
| 1b  | Partial index in **stoichiometry** produces a mislocated `E503: unknown compartment 'R_adult'` (name-mangle failure) instead of the §26.3-promised "all dimensions must be specified" diagnostic.                                                                                                                                                                                                                                                                                                                                                                                     | `expander.ml`            | emit the intended E303-style cross-dimension diagnostic at the right span                                          |

---

## Tier 2 — Stale / wrong spec (doc-fix only; code is correct)

| #        | Passage                                             | Problem                                                                                                                                                                                                                                         | Verified behavior                                                                                                                                                                                                                                                    |
| -------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 39/41/43 | §19 "Content-Addressable Output" + §1 hashing prose | Documents a **dead** `sim_hash`/`scen_hash`/`model_hash` two-level scheme (`grep` → 0 matches in code). Overclaims scenario-rename cache reuse (§19.2); has no config-level hash.                                                               | Shipped: factored 5-level `run_id(model, config, params, scenario, seed)`, structural hashing, full-64-char verify + collision disambiguation. Rename → new path → harmless cache miss. Column-selection _is_ a hashed config field. **Rewrite §19 + the §1 intro.** |
| 2        | §8.3 "No Localization, No Magic"                    | Says the compiler transforms a global infection formula via "coupling rules (§10)" and that global sugar and indexed form "produce the same IR." Contradicts §10 (sugar removed) and §8.1.                                                      | Unindexed transition over stratified compartments hard-errors `E272`. Delete the auto-transform account.                                                                                                                                                             |
| 14       | §26.9 Self-Loop Detection                           | Says generated self-loops are a **warning** that "Gillespie fires."                                                                                                                                                                             | Self-loops collapse to empty stoichiometry → hard `E310`; no self-loop warning code exists. Fix §26.9.                                                                                                                                                               |
| 29       | §12 canonical `detection` example                   | `projected = prevalence(I)` then `detection ~ bernoulli(p = p_detect)` — `projected` is never referenced, so the likelihood is state-independent. Teaches a vacuous obs model.                                                                  | Real, verbatim, compiles (no unused-`projected` lint). Rewrite so `p` depends on `projected`; consider an unused-`projected` warning.                                                                                                                                |
| 52       | §27 Primitive Summary, forcing examples             | Show `sinusoidal { ... }` with **no unit literal**, but §7 requires a tier-3 unit (E001 if omitted). Agents copy summaries.                                                                                                                     | Unitless forcing → `E001`. Add `'unit` to the §27 examples.                                                                                                                                                                                                          |
| 54       | §26.6 unit-error example; E25x placeholders         | §26.6 calls `count` "dimensionless," contradicting §2 and the dimchecker (`count ≡ [P]`). Normative prose uses literal `E25x/E25y/E25z` placeholders (real codes are E255–E259). E100/E289 are code-families used across ≥3–5 distinct classes. | Fix the example to "count (dimension P)"; give real codes; note E100/E289 are families.                                                                                                                                                                              |
| 6        | §4.1 `count : integer ≥ 0`                          | Integrality is **not enforced**; `let iota : count = 1e-6` compiles.                                                                                                                                                                            | `count` carries dimension [P] only. State that; integrality is not checked.                                                                                                                                                                                          |
| 1a       | §5.1 L821–822                                       | Comments `S[patch=p1] # sum over age` describe removed sum-over-dropped-dim behavior and contradict the section's own E287 rule.                                                                                                                | Projections do **not** marginalize dropped dims. Delete the stale comments.                                                                                                                                                                                          |
| 11       | §9.8 overdispersion prose                           | Calls σ² "the variance of the Gamma noise multiplier."                                                                                                                                                                                          | The multiplier variance is σ²/dt (runtime draws `G ~ Gamma(dt/σ², σ²/dt)`). Fix prose — and see Tier 4 for the dimension question.                                                                                                                                   |
| 33       | §13.3 schedule example                              | Uses integer positions `sia_day[p, 0]` / `sia_day[p, 1]`; table indices must be level names, and `round` is undeclared.                                                                                                                         | Compiler rejects with `E263`. Rewrite with a `round` binder / level names (see `ocaml/golden/sia_anchored_dates.camdl`).                                                                                                                                             |
| 53       | §26/grammar runnable-model requirements             | Claims `transitions` is mandatory and `check` requires `parameters`.                                                                                                                                                                            | Code is _more permissive_: pure-ODE (no transitions) and param-free models both compile, check, and simulate. Fix the doc to "transitions OR ode"; drop the parameters-for-check claim.                                                                              |
| 35       | Parameter-precedence summary                        | "External configuration overrides in-file presets" is false — in-file scenario `set`/`scale` beats external params.toml and sweep points.                                                                                                       | Real order: defaults → params files → draw/sweep → **scenario** → CLI `--param`. The explicit list two sentences later is correct; the summary is the wrong side.                                                                                                    |
| 38       | §17.1 scenario patch-ops                            | `simulate.to` is a real inheritable scenario field (`ScTEnd`, `preset_t_end`) but is omitted from the patch-operation list (only the §17.2 merge table mentions it).                                                                            | Add it to the §17.1 list.                                                                                                                                                                                                                                            |
| 51       | §26 name-resolution order                           | Gives a 5-namespace resolution order, but E278 forbids cross-namespace duplicates across exactly those 5 — so the order can never fire.                                                                                                         | Remove/clarify (it applies to reserved-id precedence & stratum shadowing, not the 5).                                                                                                                                                                                |
| 45n      | §14 timepoints "planned for v0.2"                   | Stale version note (now v0.3-draft).                                                                                                                                                                                                            | (See Tier 4 for the inert-syntax decision.)                                                                                                                                                                                                                          |
| 55       | Header `Date: 2026-03-16`                           | Body cites the 2026-05-22 typed-time proposal and 2026-05-25 CLI revision.                                                                                                                                                                      | Bump the header date.                                                                                                                                                                                                                                                |

---

## Tier 3 — Underspecified in spec but definite in code (document the behavior)

These are not bugs; the code is deterministic and sound. Writing them down lets
a modeler predict outcomes — highest value first.

| #   | Behavior to document                                                                                                                                                                                                                                                                                                                                                                                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 22  | **Same-time phase order** (highest value): transition draws + inflow events → residual events (`set`/drain) → interventions (Stage 3) → balance (Stage 4) → negative-count check → output recorded **post-effect**. An observation at a vaccination time sees the **post-vaccination** state. A modeler currently can't predict this.                                                                                      |
| 28a | `neg_binomial(mean, r)` is **NB2**: variance = μ + μ²/r (`k → ∞` ⇒ Poisson). Load-bearing for priors on `r`, and invisible in the spec today.                                                                                                                                                                                                                                                                              |
| 27  | `incidence(transition)` accumulation window: accumulates from **sim start** (so the first interval is "since t0," not "since last obs"), interval `(prev_obs, curr_obs]`, per-stream reset, and **two data rows at the same time is a hard error** (gh#188). Changes likelihoods — worth a precise paragraph.                                                                                                              |
| 20  | ODE backend: every transition adds `stoich·rate` to derivatives; scheduled effects are exact discontinuities with post-effect output and solver restart; balance-on-ODE is a hard capability error (chain-binomial only).                                                                                                                                                                                                  |
| 15  | Multi-source (`A + B --> C`) on chain-binomial is a **hard error** (only the first source bounds the draw); use Gillespie/ODE. Document the gate.                                                                                                                                                                                                                                                                          |
| 18  | `deterministic(rate)` count is clamped to `[0, n_src]`; two deterministic transitions on the same source is rejected (gh#122).                                                                                                                                                                                                                                                                                             |
| 36  | Scenario patch algebra: `set` then `scale` (scale multiplies the set value); `compose` list in order, parent's own patch last (wins on collision); explicit `disable` beats `enable`.                                                                                                                                                                                                                                      |
| 21  | `via` staged-residence entry: `init`/inflow land in stage 1; bare `E` / `prevalence(E)` sum stages; `hyper_erlang` on a stratified source is E248; a bare staged name in an intervention `transfer(to = E)` is E264 (must name `E_s1`) — an undocumented asymmetry vs init/inflow.                                                                                                                                         |
| 26  | Reactive-intervention engine semantics (all pinned in `reactive.rs`): window `(now − w, now]` open-left/closed-right; **level** predicate at emit boundaries gated by `once`/cooldown; cooldown measured from **trigger** time. Also **soften** the "shared across particles" sentence — reactive is chain-binomial-forward-only and never runs in inference, so that sentence describes a feature that doesn't exist yet. |

---

## Tier 4 — Real ergonomic / design issues — needs-Vince

| #   | Issue                                                                                                                                                                                                                                                                                            | Note                                                                                     |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 8   | `positive`/`real` unit literal **discards scale**: `importn : positive 'per_year` is byte-identical to `'per_day`, so bounds `in [1e-4, 0.1]` are read as per-**day** (365× off) — while forcings/tables _do_ rescale `'per_year`. The reviewer's strongest trap; documented intent, but silent. | Warn/reject non-day rate literals on params, or honor the scale.                         |
| 44  | `output { format = ... }` parses but is **inert** (`format = banana_nonsense` compiles clean, then stripped before hashing). Violates "no loose semantics."                                                                                                                                      | Wire it as default, or remove the field and hard-error.                                  |
| 45  | `timepoints { }` parses then is **discarded** (`DTimepoints _ -> ()`, zero consumers). Same principle.                                                                                                                                                                                           | Implement, or reject with a capability error.                                            |
| 12  | Branch weights needn't sum to 1 — but the spec's "implicit 'other' destination" is **mathematically wrong**: `scaled_rate = weight · raw_rate`, so weights summing to 0.8 just reduce the exit hazard to 0.8·r; nobody gets the missing mass. sum > 1 inflates the exit rate.                    | Fix the prose; decide whether to enforce sum = 1 or add a residual `otherwise` branch.   |
| 13  | Erlang `via rate = r` means **1/overall-mean** (per-stage rate `k·r`), not the conventional Erlang stage rate λ (mean k/λ). A statistician reading `rate` will be k× off.                                                                                                                        | Rename (`inverse_mean`?) or document loudly.                                             |
| 28b | `poisson(rate = projected)` names an expected **count** `rate`, confusing in a language with a `rate` physical type (the dimchecker itself flags it as [P], not per-time).                                                                                                                       | Rename to `mean`? Breaking DSL change.                                                   |
| 9   | Every compartment is forced to dimension [P]; `W : real [1]` is `E001`. Environmental reservoirs can't carry a concentration/dimensionless dimension.                                                                                                                                            | Allow a state dimension annotation (`real [C]`, `real [1]`).                             |
| 46  | Explicit `at` output time beyond `t_end` is silently dropped (no diagnostic).                                                                                                                                                                                                                    | Add a warning listing omitted times.                                                     |
| 50  | Built-in names (`log`, `poisson`, `date`, `projected`, `baseline`) are legal parameter names (not reserved); resolution is position-dependent.                                                                                                                                                   | Lint (W) when a param shadows a builtin fn.                                              |
| 31  | Multi-value `read()` can't express per-column units (one unit for all value columns); can't read a single column from a multi-column file. Workaround: split files.                                                                                                                              | Per-column unit syntax, or document the constraint.                                      |
| 32  | Data-derived dimension order = first-row-occurrence; only load-bearing for `consecutive(dim)` over a data-derived dim (positional integer indexing isn't user-reachable).                                                                                                                        | Nominal-vs-ordered distinction, or warn when `consecutive()` targets a data-derived dim. |
| 37  | Scenario inheritance **appends** `enable`/`disable`/`compose` lists (with W310 when the append changes the resolved list) where users may expect replace. Already documented + warned.                                                                                                           | Optional `+=` / `=` operators (enhancement, not a bug).                                  |
| 16  | No stoichiometric coefficients (`2A --> C`); `A + A` gives net −2 but propensity uses the raw expression, not `A(A−1)/2`. By design (compartmental co-events, not chemistry).                                                                                                                    | Document the idiom; decide if mass-action coefficients are in scope.                     |

---

## Reviewer misreads — no action

- **Item 10** (flagship "canonical five-age example fails its own dimensional
  checker"): **does not reproduce.** Contact matrices are dimensionless
  (§2.5/§10.2) and the worked examples compile clean. The reviewer assumed a
  `C_age : age × age 'per_day` form no example uses — and even that compiles,
  because a unit literal on a multi-dim table stamps scale only, never a cell
  dimension.
- **Item 1** (mostly): "four indexing dialects" — the code is principled
  (partial index → E287 in any read/write/projection; index binders expand;
  silent-sum guarded by E280). Only the small §5.1 stale-comment defect (Tier 2,
  1a) and the E503 mislocation (Tier 1, 1b) survive.
- **Item 3**: `baseline` coherently sentinels — identity patch when the model
  defines no `baseline`, value-carrying preset when it does. Not a
  contradiction.
- **Item 5**: `ic_free` (fit-level conditional-likelihood setting) and
  parameterized init (`init { I = I0 }`, `I0` an ordinary estimated param) are
  complementary, not rival mechanisms. Note: there is **no `ivp` ParamKind** —
  the `ivp: true` in ARCHITECTURE.md/CLAUDE.md is stale (`ivp` survives only as
  a CLI gating label).
- **Item 7**: rate/probability boundary values are handled via interior clamps
  (`LOG_PROB_FLOOR`, `PROB_FRACTION_EPS`); no non-finite transform values arise.
- **Item 25**: `add` producing a negative compartment is a **hard error**
  (`NegativeCount::InterventionNegative`); only the balance _target_ warns
  (documented, intentional).
- **Item 40**: 8-hex path segments are a display convenience backed by
  full-64-char identity gates + collision disambiguation — no birthday-collision
  risk.
- **Item 42**: `quantities {}` are deliberately non-identity regenerated
  sidecars (`--quantities-out`), never in the CAS leaf — so changing them can't
  mutate an immutable run's bytes.
- **Item 48**: missing `simulate` → `t_start=0, t_end=100` is an intended,
  spec-disclosed default.

---

## Suggested sequencing

1. **Item 47** — the one silent-wrong bug. TDD red (the repro above), then the
   `start = t_start` fix with gh#143-aware care. Highest priority.
2. **Tier 2 doc sweep** — the stale sections actively mislead agents copying the
   spec. §19 hashing rewrite, §8.3, §26.9, the §12 detection example, the §27
   forcing units, and the §26.6 `count`-dimensionless example are the ones most
   likely to produce broken generated models.
3. **Tier 3 documentation** — item 22 (phase order) and item 28a (NB2 variance)
   first; both are load-bearing for correctness a user can't currently predict.
4. **Tier 1 code warts** — items 30 and 34b are the highest-value (data
   corruption + humans-first violation); batch the rest as small fixes.
5. **Tier 4** — Vince's design calls; item 8 (silent per-year footgun) and items
   44/45 (inert syntax) are the ones that touch shipped surface.
