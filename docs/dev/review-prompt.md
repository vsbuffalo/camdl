# Code Review Prompt — CAMDL / Compartmental

This document is a self-contained prompt for reviewing code in the `compartmental`
monorepo. It can be given verbatim to a human or AI reviewer. No prior reading of
`CLAUDE.md` is assumed.

---

## Project context

`compartmental` is a stochastic compartmental epidemic modelling framework. It has
two subsystems joined by a shared JSON IR:

- **OCaml frontend** (`ocaml/`): DSL source → stratification expansion → IR JSON
- **Rust backend** (`rust/`): IR JSON → simulation → inference → trajectory output

The IR schema (`ir/schema.json`) is the formal contract between them.

**Stakes:** This software informs major public health decisions. A silent wrong
answer — one that compiles clean, runs without error, and produces plausible but
wrong numbers — is the worst failure mode. Prioritize findings accordingly.

---

## Output format

Produce a Markdown document saved to `docs/dev/reviews/YYYY-MM-DD-{scope}.md`.
The format must match the convention used in all previous reviews in that directory
so that findings can be cross-referenced in PRs and tracked to resolution.

### Document template

~~~markdown
---
status: open
date: YYYY-MM-DD
scope: {one-line description of what was reviewed}
reviewer: {name or "agent"}
---

## Resolution status

| Finding | Status | Notes |
|---------|--------|-------|

---

# {Title} — YYYY-MM-DD

{One paragraph: what files were reviewed, what this review focuses on, what
prior reviews (if any) it builds on and does not repeat.}

## Summary

**Strong:** {What is well-designed and should be preserved — 2–4 sentences.
Name specific patterns, modules, or decisions. This is not courtesy; it tells
future reviewers what not to change.}

**Needs work:** {The 2–4 most important problems, one sentence each. These
should map to Major findings below.}

## Findings

### Major

**{CODE}. {Short title — 10 words or fewer}**

{Prose: what the problem is, why it matters, the specific correctness or
maintenance risk it creates.}

{Bad code block (quoted from the actual source with file:line reference).}

{Good code block or prose fix.}

---

### Minor

**{CODE}. {Short title}**

{Same structure, shorter.}

---

### Nit

**{CODE}. {Short title}**

{One short paragraph + optional fix.}

---

## Cross-cutting themes

{2–4 named themes that unite multiple findings. Each theme: one sentence
stating the pattern, one sentence on the systemic fix.}
~~~

### Finding codes

Each finding gets a unique alphanumeric code for cross-referencing. Codes
survive into PR descriptions, commit messages, and the resolution table above.

Format: `{prefix}{severity}{n}`

| Part | Values | Meaning |
|------|--------|---------|
| `prefix` | 2-letter scope abbreviation | Derived from what was reviewed (see below) |
| `severity` | `M` / `m` / `n` | Major / minor / nit |
| `n` | 1, 2, 3, … | Sequential within prefix+severity |

Standard prefix abbreviations:

| Prefix | Scope |
|--------|-------|
| `Oc` | OCaml compiler (`ocaml/lib/compiler/`) |
| `Ir` | OCaml IR types (`ocaml/lib/ir/`) |
| `Si` | Rust sim crate (`rust/crates/sim/src/`, excl. inference) |
| `In` | Rust inference stack (`rust/crates/sim/src/inference/`) |
| `Cl` | Rust CLI (`rust/crates/cli/`) |
| `Io` | Rust I/O / observe crates |
| `Sc` | IR schema (`ir/`) |
| `Ts` | Test coverage |
| `Rd` | Rust design (cross-crate design quality pass) |

A code never changes after it is written. If a finding is resolved, update the
resolution table — do not renumber remaining codes.

---

## Review procedure

The review runs in five ordered passes. **Do not merge passes.** An agent that
tries to evaluate all ten criteria simultaneously against a large diff will miss
correctness issues by diffusing attention into nits.

### Pass 1 — Orientation (no findings yet)

Read without evaluating:
- The PR description or commit message
- The proposal in `docs/dev/proposals/` that this change implements (if any)
- The list of changed files and their crate/module context
- The diff at a high level — what kind of change is this?

Output of this pass: a one-paragraph scope statement for the review document
header, and a mental model of what the change is trying to do.

### Pass 2 — Correctness (§1–§2)

Apply §1 (scientific correctness) and §2 (IR contract integrity).

If any **Major** finding surfaces here: document it, set `status: open`, and
**stop**. Report the Major finding immediately. Do not proceed to Passes 3–4
until correctness blockers are resolved. A nit in §7 is irrelevant if the
inference math is wrong.

### Pass 3 — Design (§3–§6)

Apply §3 (type design), §4 (abstraction boundaries), §5 (diagnostic quality),
§6 (DSL/CLI UX).

Findings here are Major or Minor. Nits in this pass are rare — if something
in type design is merely a style preference, record it as a Nit or omit it.

### Pass 4 — Quality (§7–§9)

Apply §7 (DRY/SOLID/magic values), §8 (performance), §9 (test coverage).

Most Minor and Nit findings live here. A Major finding in §8 (e.g., allocation
in a per-particle loop) is possible but should be escalated to the Pass 2
summary if the performance impact is correctness-adjacent (e.g., OOM under
production particle counts).

### Pass 5 — Synthesis

Write the **Summary** section (Strong / Needs work) and the **Cross-cutting
themes** section. These are written last because themes only become visible
after all four substantive passes.

Fill in the **resolution table** with one row per finding, `Status: Open`.

---

## How to read the criteria sections

Criteria are ordered by **blast radius**: the severity of the worst-case silent
failure if the criterion is violated. A nit at the bottom is genuinely a nit;
a finding in §1 can corrupt inference posteriors.

---

## §1 — Scientific correctness

*Blast radius: silently wrong inference, wrong simulation dynamics, invalid posteriors.*

This is the highest-priority section. Findings here can produce results that look
plausible but are numerically wrong — the failure class of the 2026-04-21
table-unit incident (compiler silently ignored unit annotations on table values,
scaling was never applied).

**1.1 Proposal adherence.** If the change implements a proposal under
`docs/dev/proposals/`, verify it matches the proposal exactly. Flag any deviation
that is not documented inline with a reason. Do not accept "this seems equivalent"
— verify it is equivalent by comparing against the mathematical derivation.

**1.2 Inference math correctness.** For any change touching these files, treat
it as high-risk regardless of how mechanical it looks:

```
rust/crates/sim/src/inference/pgas.rs
rust/crates/sim/src/inference/pgas_grad.rs
rust/crates/sim/src/inference/obs_loglik.rs
rust/crates/sim/src/inference/particle_filter.rs
rust/crates/sim/src/inference/if2.rs
rust/crates/sim/src/inference/nuts.rs
```

Read the *entire* function before evaluating any part of it. Verify:
- Log-probability computations handle the zero/underflowing case explicitly (not silently)
- Gradient expressions match the symbolic derivatives in the autodiff output
- Particle resampling indices are used consistently before and after the resample step
- Ancestor sampling backward pass consumes gamma values in exactly the same order
  as the forward pass emits them (this is a cross-function invariant with no type enforcement)

**1.3 RNG consumption order.** The paired-seed CRN guarantee (same seed → same
pre-intervention trajectory for `enable`/`disable` scenarios) holds only when the
RNG is consumed in the same order on both scenario branches. Any structural change
that reorders RNG draws in `step()` or reorders transitions silently breaks the
coupling. Flag any such change.

**1.4 Unit handling.** The compiler performs dimensional analysis. Any new path
that loads numeric values from an external source (CSV, TSV, inline table, user
parameter) must either:
- Apply the model time-unit conversion (`Days | Weeks | Months | Years`) explicitly, or
- Carry a unit annotation in the IR and apply conversion in the reader

"Load values verbatim" is wrong unless the spec explicitly says no conversion is needed.

**1.5 Spec claims without tests.** If the change introduces a new compiler
behaviour claimed in the spec (e.g., "this forcing form produces Erlang-k
distribution", "scenario `scale` multiplies the parameter at runtime"), verify
there is a test that would fail if the claim were violated. Spec claims without
tests are the primary source of silent regressions in this codebase.

---

## §2 — IR contract integrity

*Blast radius: cross-language desync, parse failures at runtime, data loss in golden files.*

**2.1 Atomic schema changes.** Any change to `ir/schema.json` or `ir/VERSION` must
be accompanied in the *same commit* by:
- Updated OCaml types in `ocaml/lib/ir/` (`ir.ml`, `serialize.ml`, `deserialize.ml`)
- Updated Rust types in `rust/crates/ir/src/`
- Regenerated golden files (`make update-golden && make update-expected`)
- Bumped `ir/VERSION`

A partial update that updates only one language is never acceptable. Flag any
schema-touching PR that is missing any of these four components.

**2.2 No silent defaults for required fields.** New IR fields that carry semantic
meaning must be `required` in the schema, not defaulted to a value that silently
produces wrong behaviour. The pattern `#[serde(default = "...")]` is only
appropriate for fields whose absence genuinely means "use the default behaviour
specified in the spec."

Bad:
```rust
#[serde(default)]   // defaults to false, but false means "never fire this event"
pub enabled: bool,
```

Good:
```rust
pub enabled: bool,  // required; schema enforces presence
```

**2.3 IR is fully flat.** Verify no stratification syntax survives into the IR.
The OCaml compiler is responsible for all expansion; the Rust backend receives a
flat list of compartments, transitions, and observations. Any IR field that
encodes "iterate over dimension X" is a design violation — the expansion should
have happened at compile time.

---

## §3 — Type design

*Blast radius: wrong values accepted at compile time, type confusion between domain concepts.*

### OCaml

**3.1 Located types on all AST nodes.** Every construct that a user wrote in source
should carry a `loc` (file, line, col) for diagnostic reporting. A node without a
location produces locationless error messages — users must grep their model to find
the mistake.

Bad:
```ocaml
type error =
  | UnknownParameter of string
  | DuplicateCompartment of string
```

Good:
```ocaml
type error =
  | UnknownParameter of string * Diagnostics.loc
  | DuplicateCompartment of string * Diagnostics.loc
```

**3.2 Discriminated unions over booleans for distinct cases.** If two values of the
same "type" have meaningfully different structure or behaviour, they should be
separate constructors, not a boolean flag.

Bad:
```ocaml
type destination = {
  compartment: string;
  is_branching: bool;   (* true = DstBranch, false = DstSum *)
  weights: expr list;   (* only meaningful when is_branching = true *)
}
```

Good:
```ocaml
type destination =
  | DstSum  of { compartment: string }
  | DstBranch of { compartment: string; weight: expr }
```

**3.3 No stringly-typed identifiers across module boundaries.** Compartment names,
parameter names, and dimension names should resolve to indices as early as possible
in the compiler pipeline. String lookups in the hot evaluation path are an
O(n) linear scan that should have been a compile-time O(1) index.

**3.4 Module interface boundaries.** Internal types should not leak across module
boundaries without an `.mli` guard. Accessing `d.ctx.diags.diags` directly from
`compiler.ml` (reaching into the `Diagnostics` module's internal fields) is a red
flag. Every field access on a type owned by another module should go through that
module's public interface.

Bad:
```ocaml
(* compiler.ml *)
if d.ctx.diags.diags <> [] then
```

Good:
```ocaml
(* diagnostics.mli *)
val has_any : t -> bool
(* compiler.ml *)
if Diagnostics.has_any d.ctx.diags then
```

### Rust

**3.5 Newtype wrappers for domain concepts in public APIs.** Raw `f64` for time and
`usize` for indices are interchangeable at compile time. In function signatures that
take multiple `f64` or multiple `usize` arguments, silent argument transposition is
the most common call-site bug.

At minimum, flag parameter lists like:
```rust
fn step(&self, state: &mut S, params: &[f64], t: f64, dt: f64, ...)
```
where `t` and `dt` could be swapped with no compiler error. This is an accepted
technical debt item (Rdn1 in the 2026-04-20 Rust design review) — do not introduce
*new* instances of it.

**3.6 `thiserror` enums, not `String` errors.** No public function in a `sim` or
`ir` crate should return `Result<_, String>`. Errors that escape a crate boundary
must be typed so callers can match on them.

Bad:
```rust
fn validate(...) -> Result<(), String> {
    Err("unknown compartment".to_string())
}
```

Good:
```rust
fn validate(...) -> Result<(), SimError> {
    Err(SimError::UnknownCompartment(name))
}
```

**3.7 Types belong in the module that owns them, not the first module that needed them.**
The canonical anti-pattern in this codebase is types like `EstimatedParam` and
`Transform` defined in `if2.rs` and imported by PGAS and PMMH via `use super::if2`.
If a type is used by more than one algorithm, it belongs in `inference/types.rs`
or `inference/mod.rs`, not in an algorithm-specific file.

---

## §4 — Abstraction boundaries and leaky interfaces

*Blast radius: algorithms that bypass trait contracts, making future refactors dangerous.*

**4.1 Inference algorithms use traits, not concrete types.** All simulation and
observation logic in the inference layer must go through these three traits:

- `ProcessModel` — forward simulation step, initial state, scratch allocation
- `ObservationModel` — log-likelihood, observation count, sampling
- `DensityProcess: ProcessModel` — log-transition density for PGAS

An algorithm that reaches directly into a `CompiledModel`, `ChainBinomial`, or
other concrete type bypasses the abstraction and ties the algorithm to an
implementation detail. Flag any function in `inference/` that takes a concrete
simulation type rather than a trait object or generic bound.

**4.2 The `expand` module encapsulates all stratification logic.** No code outside
`ocaml/lib/compiler/expander.ml` should perform or undo stratification. If a
post-expansion pass needs to understand the stratified structure of a model, it
should read the `dimensions` field in the IR — not attempt to reverse-engineer
compartment names.

**4.3 Algorithms do not share state through global mutable references.** The
`Compiler.no_dim_check` pattern (a mutable global ref set by the CLI) is a known
design smell in this codebase. Do not introduce new mutable globals. Prefer passing
config through function arguments or a context struct.

---

## §5 — Diagnostic quality

*Blast radius: users cannot find or fix their errors; diagnostic regressions go unnoticed.*

Every new error path must carry **all four** of:

| Field | What it must contain |
|-------|---------------------|
| `code` | Stable identifier, e.g. `"E303"` — enables grep, CI filtering, suppression |
| `loc` | Source file + line + column of the *actual mistake*, not a secondary symptom |
| `message` | One sentence: what went wrong |
| `hint` | What the user should do to fix it (when a fix is known) |

The `detail` field is optional but recommended when the root cause is non-obvious.

**5.1 `failwith` and `assert false` are never acceptable for user-facing errors.**
These produce stack traces instead of diagnostics. Any `failwith` in a code path
reachable from a user model is a bug.

Bad:
```ocaml
failwith (Printf.sprintf "autodiff: mod w.r.t. param '%s' not representable" p)
```

Good:
```ocaml
(* Return an error result and emit E600 in compiler.ml with the transition name and loc *)
Error (ModOverParam { param = p })
```

**5.2 Severity must be correct.**
- `Error` — blocks compilation; user cannot proceed
- `Warning` — suspicious but legal; user should see it but compilation continues
- `Info` — informational (e.g., "dimension could not be inferred"); non-blocking

Downgrading a real error to `Warning` or `Info` is a silent failure mode. Upgrading
an advisory to `Error` is a usability regression.

**5.3 Error codes point to the actual mistake, not a downstream symptom.** When two
codes could fire for the same root cause, prefer the one that gives the user the
cross-construct context they need.

Example: a parameter used with inconsistent dimensions across transitions should
produce `E303` ("conflicting dimensions in transition A vs B"), not `E302`
("dimension mismatch in addition") — even though E302 is technically triggered.
E303 gives the user the cross-transition context; E302 gives them a symptom.

**5.4 New diagnostic codes need error fixtures.** Every new `E###`, `W###`, or
`I###` code must have a corresponding golden fixture in `ocaml/golden/errors/`
named `e{NNN}_{description}.camdl`. This is the regression tripwire that ensures
the diagnostic can never silently disappear in a refactor.

---

## §6 — DSL and CLI user experience

*Blast radius: users write invalid models that silently compile, or get unusable error messages.*

**6.1 No silent acceptance of invalid input.** If a construct looks like it means
something, it either means exactly that or produces a clear error. There is no
"ignored but accepted" middle ground.

Bad: a keyword argument with a typo (`amplitde = 0.5`) is silently ignored and
the forcing function uses its default amplitude.

Good: `E401: unknown argument 'amplitde' to sinusoidal; did you mean 'amplitude'?`

**6.2 New syntax follows the grammar conventions in the spec.** Review additions
against `docs/camdl-language-spec.md`:
- Expressions use the existing `expr` grammar; no ad hoc extensions
- Identifiers follow the `snake_case` convention
- Blocks use `{ }` with consistent indentation rules
- Unit literals follow the tier-1/2/3 system

**6.3 New CLI flags follow existing conventions.** Check that:
- Long flags use `--kebab-case`
- Flags with values use `--flag VALUE`, not `--flag=VALUE` or `--flagVALUE`
- Mutually exclusive flags produce a clear error, not silent precedence rules
- New subcommand shapes are consistent with the `camdl {simulate, infer, check, ...}` hierarchy

**6.4 No backwards-compatibility shims.** This is unreleased software. When a field
is renamed, rename it everywhere atomically. Do not add `#[serde(alias = "old_name")]`,
deprecated re-exports, or `// kept for backwards compat` comments. Clean design
beats legacy support.

---

## §7 — Code structure: DRY, SOLID, and magic values

*Blast radius: maintenance hazards, silent divergence between copies, numeric correctness errors from magic value inconsistency.*

**7.1 Named constants for all numeric magic values.** Every bare numeric literal
that represents a deliberate algorithmic choice — a floor value, a stream offset,
a convergence threshold — must be a named constant with a doc comment explaining
the choice. The canonical anti-pattern in this codebase is `1e-300` appearing at
eight sites with no name and no explanation.

Bad:
```rust
// Multiple sites across multiple files:
let safe = val.max(1e-300);
```

Good:
```rust
// In inference/types.rs:
/// Floor for ln() args to avoid −∞ log-weights. ln(LOG_PROB_FLOOR) ≈ −690,
/// safely above the underflow threshold for any realistic particle count.
/// Do not lower below f64::MIN_POSITIVE (5e-324).
pub const LOG_PROB_FLOOR: f64 = 1e-300;
```

**7.2 No magic seeding patterns without documentation.** RNG seeds and stream
offsets are correctness-critical. Every seed computation must be documented with
its intent.

Bad:
```rust
let resample_rng = StatefulRng::new(seed.wrapping_add(0xdeadbeef));
```

Good:
```rust
// Use a reserved high stream index that cannot collide with per-particle streams
let resample_rng = StatefulRng::new_stream(seed, RESAMPLE_RNG_STREAM);
// where RESAMPLE_RNG_STREAM: u64 = u64::MAX is defined in types.rs
```

**7.3 Shared types defined in the module that owns them.** Types used across
multiple algorithm files belong in `inference/types.rs`. Types used across multiple
OCaml compiler passes belong in a shared module, not in whichever pass needed them
first.

**7.4 Functions > ~80 lines of correctness-critical math need decomposition.** Long
functions in the inference core (`pgas.rs`, `pgas_grad.rs`, `if2.rs`) make the
relationship between code and mathematical derivation unverifiable by inspection.
Decompose along the logical stages of the algorithm.

Each sub-function should map to one named paragraph of the derivation:

```rust
// Logically: p(y_t | x_t^(i)) · p(x_t^(i) | x_{t-1}^(a_i)) / q(x_t^(i) | ...)
let obs_ll = observation_log_likelihood(&obs, &state, &params);
let trans_ll = log_transition_density_substep(&model, &prev, &flows, &gammas, ...);
let weight = obs_ll + trans_ll - proposal_log_density;
```

**7.5 DRY across algorithm files.** The three inference algorithms (IF2, PGAS, PMMH)
share boilerplate for: particle RNG initialization, config field access, and resume
state serialization. When this boilerplate diverges (e.g., IF2 uses
`stream_base | i` and PF uses bare `i`), the difference must be intentional,
documented, and exercised through a shared helper that makes the variant visible.

Three similar lines is acceptable. The same 20-line block copy-pasted across three
algorithm files with subtle variations is not.

---

## §8 — Performance sensitivity

*Blast radius: inference that takes 10× longer with no warning; cache thrashing in particle loops.*

The simulation inner loop and inference particle loop are the hot paths. Any change
that allocates heap memory in these paths, or introduces O(n) lookups where O(1)
was used before, is a performance regression.

**8.1 No allocation in `step()` or per-particle loops.** The `ProcessModel::step()`
contract provides a `scratch: &mut Self::Scratch` parameter specifically so
temporary buffers can be pre-allocated at model construction. New code in `step()`
must not call `Vec::new()`, `HashMap::new()`, `Box::new()`, or any other allocator.

**8.2 Data structures chosen for evaluation performance, not construction convenience.**
The canonical anti-pattern: `rate_grads: Vec<Vec<(String, ResolvedExpr)>>` uses
string keys because the OCaml compiler emits parameter names, but the Rust runtime
resolved those names to indices at construction time. The hot-loop lookup should
use indices, not strings.

When reviewing data structures in the inference layer, ask: "does this lookup happen
once at model construction (acceptable) or once per particle per substep (must be O(1))?"

**8.3 Pre-computed resolved expressions.** The `resolved_expr.rs` module proves the
right pattern: expressions are compiled to index-offset closures at model
construction, so the inner loop touches no hash maps. Any new expression evaluation
that uses a `HashMap` lookup in a per-step context is wrong.

---

## §9 — Test coverage

*Blast radius: regressions ship silently; spec claims go unverified.*

**9.1 Golden fixture for every new compiler feature.** A new DSL construct must have
at least one `.camdl` fixture in `ocaml/golden/` whose `.ir.json` output is committed.
This makes the feature a regression tripwire.

**9.2 Error fixture for every new diagnostic code.** A new `E###`, `W###`, or `I###`
code must have a minimum-reproducer `.camdl` fixture in `ocaml/golden/errors/`. The
negative-golden test suite will assert this fixture triggers the expected code and
no other error.

**9.3 End-to-end tests for runtime application of spec claims.** Spec claims of the
form "this runtime operation produces result X" must have a test that would fail if
X were not applied. Compile-time golden tests verify IR *shape*, not runtime
*behaviour*. Example: `scenarios { sa { scale = { beta = 2.0 } } }` must have a
Rust integration test that asserts the trajectory under `--scenario sa` differs
from the baseline in the expected direction.

**9.4 Test suite passes before and after.** Every PR must demonstrate:
1. `cargo test --workspace` passed on the base branch
2. `cargo test --workspace` passes after the change
3. `dune runtest` passes in `ocaml/`

A change to inference math that does not run the full test suite is not reviewable.

---

## §10 — Proposal adherence and documentation

*Blast radius: design rationale lost; implementation diverges from reviewed design.*

**10.1 Changes implementing a proposal must cite it.** If a PR implements
`docs/dev/proposals/2026-04-11-inference-traits.md`, the PR description and any
non-obvious implementation decision should cite the proposal by filename. A reviewer
must be able to verify implementation fidelity by comparing code to proposal.

**10.2 Deviations from a proposal are documented inline.** If the implementation
differs from the proposal, the code must include a comment at the deviation point
explaining why (a constraint discovered during implementation, a correctness issue
with the proposed approach, etc.). "We decided to do it differently" with no
explanation is not acceptable.

**10.3 No improvised design changes during implementation.** A PR that was scoped
to implement a proposal must not also restructure unrelated code, add new IR fields,
or change CLI behaviour. Scope creep during implementation is reviewed as if it
were a separate, unreviewed proposal — with correspondingly higher scrutiny.

**10.4 `CLAUDE.md` is not a substitute for a proposal.** General engineering
principles in `CLAUDE.md` govern implementation style. Major design decisions —
new IR fields, new inference algorithms, new compiler passes — require a proposal
in `docs/dev/proposals/` reviewed before implementation begins.

---

## Checklist summary

Use this as a final pass before approving:

- [ ] §1: Does the change touch inference math? If so, is the math correct relative
      to the derivation/proposal? Is RNG order stable?
- [ ] §2: If schema changed, are OCaml types, Rust types, golden files, and VERSION
      all updated atomically in this commit?
- [ ] §3: Do all new OCaml AST nodes carry source locations? Are new Rust error types
      using `thiserror` enums? Are shared types in the right module?
- [ ] §4: Do inference algorithms use the `ProcessModel`/`ObservationModel`/
      `DensityProcess` traits? No concrete type bypass?
- [ ] §5: Does every new error path carry code, loc, message, and hint? No `failwith`
      for user-facing errors? Is there an error fixture for every new code?
- [ ] §6: Does invalid input produce a clear error, not silent acceptance?
- [ ] §7: Are all numeric magic values named constants? Is duplicated boilerplate
      extracted to a shared helper?
- [ ] §8: Does anything in a hot path allocate or do a string lookup?
- [ ] §9: Golden fixture for new feature? Error fixture for new diagnostic code?
      End-to-end test for runtime spec claims? Test suite green before and after?
- [ ] §10: If implementing a proposal, is it cited? Are deviations documented inline?
