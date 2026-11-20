### Two kinds of comment

A **doc comment** says what a module, type, field, constructor or function _is_:
what a value of it means, what it guarantees, and what it is for. In Rust it is
`///` on an item and `//!` at the top of a module; in OCaml it is `(** ... *)`.
The internals pages on the website publish doc comments and nothing else, so a
doc comment is published prose and follows the rest of this style guide.

An **implementation comment** says why a line of code is written as it is, when
the code cannot say so itself. In Rust it is `//`; in OCaml it is `(* ... *)`.
It is never published, and it is read only by someone already looking at that
line.

Both describe the code as it is today. Neither is a record of how the code came
to be.

### What a doc comment says

The first sentence is a summary that stands alone, because it is what the index
of a module shows. It names what the thing represents in the model or the
program, not its shape: "A transition's destination: one or more compartments,
or a probabilistic choice between them", not "A list of stoichiometry
references". The declaration printed beside it already gives the shape.

After the summary, a doc comment adds only what the declaration cannot show:

- **For a type**: what each constructor or field stands for, in camdl's terms;
  units, scales and time conventions (per day or per year, a probability or a
  rate, an instant or a duration); and any invariant the type does not enforce
  (a list that is never empty, names that are already resolved).
- **For a function**: its contract. What it computes, what it requires of its
  inputs, what it guarantees about its output, and the errors it returns and
  when. Not how it computes it; that is the code's job, helped where needed by
  implementation comments.
- **For a module**: its design as it stands. What it receives and hands on,
  which types carry its work, what it guarantees to its callers, and why it is
  shaped this way.
- **An example**, where one helps: preferably the camdl source that produces the
  value. A camdl example is fenced `camdl` in Rust and marked `{@camdl[ ... ]}`
  in OCaml, and the documentation build compiles it. A Rust example fenced
  `rust` is a doctest, run by `cargo test`. Both are code: an example that
  cannot stand alone is marked as a fragment, not left to drift.

### Explain a design once

A design is explained in one place: the module's doc comment, or a published
specification it implements. A function's doc says only what is particular to
that function, and links to the place that explains the rest. Repeating a design
in every function that takes part in it multiplies the comments that go stale
when the design changes, and the copies drift apart.

When the code implements a specification, link to it rather than restating it. A
restated precedence chain or rule list is a second copy of the specification,
and it will disagree with the first.

### Reasons, not history

Write the reasons for the code as they stand today. "Names are resolved before
this pass, so no expression here can refer to an unknown compartment" is a
reason: it holds now, and it tells a reader what they may rely on. "This pass
was moved after name resolution to fix a crash" is history. A reason names the
property the design protects; history names the event that produced it. When the
honest reason is that no one has needed anything better, say what the code does
and leave the reason out.

So a comment carries no issue or proposal numbers, no dates, no names of the
work stream in which a feature arrived, no "previously", "now", "no longer",
"the new …", "replaces" or "superseded by", and no list of the functions or
files the code replaced. Git history holds that record. History is deleted, not
moved into a document.

Two exceptions. A regression test names the issue it reproduces. And code names
an issue when it works around a defect outside camdl that the issue documents.

### State guarantees precisely

Replace a slogan with the guarantee it stands for. "Single source of truth",
"the one authority", "makes it impossible" and "robust" say that something is
good without saying what holds. Say what holds: "the only function in `cli` that
writes `parameters[i].value`", "an envelope whose `ir_version` differs from
`IR_VERSION` is refused with `IrError::VersionMismatch`". A precise claim can be
checked, and so can be found false.

### Keep the scientific argument

The most valuable comments in camdl explain what a model operation means, why a
change that looks harmless would alter simulation or inference results, or why a
combination of inputs is refused rather than resolved. Keep these, and write
them in the terms of the model rather than the code. They are the comments that
stop a careful reader from "fixing" code that is correct, and they cannot be
recovered from the code alone.

### Check every claim against the code

A comment is a claim about the code, and a wrong claim is worse than none:
readers, human or agent, act on it without checking. Before writing or keeping a
statement, check it against the code that produces and consumes the value, not
against an older comment or a specification.

When a comment disagrees with the code, correct the comment. When the code
disagrees with a specification, or a comment's claim is false and correcting it
would hide a defect, report the disagreement and change neither: which side is
right is a decision, not an edit.

### A long comment is a design signal

A doc comment that needs more than a short paragraph to explain a type or a
function is often compensating for the code: a name that does not say what it
means, a function doing two jobs, a `bool` or a string standing in for a variant
type, a boundary in the wrong place. Ask whether a better name, a smaller
function, a typed representation or a clearer boundary would make the comment
unnecessary. Report that as a suggested change; do not make it in the same
change as the comment.

### Implementation comments are short and local

An implementation comment earns its place by stopping a careful reader from
"fixing" something that is correct, or by naming a fact the code depends on that
the code cannot show: an ordering that matters, a numerical hazard, a property
of the input that a check upstream established. One or two sentences, next to
the line in question. Delete a comment that restates the code below it, narrates
the change that introduced it, or reasons aloud.

### Examples

**Instead of** (history, a restated specification, and a slogan):

```rust
//! Unified parameter-value resolver — single source of truth for the
//! precedence chain documented in `docs/camdl-run-spec.md §1.3`.
//!
//! Background
//! ----------
//!
//! Before this module, three half-resolvers + several inline blocks
//! implemented the same precedence rules independently:
//!
//!   - `util::resolve_run_model` for `simulate` / `lineage` …
//!   - inline blocks in `profile.rs:437-453`, `if2.rs:109-168`, …
//!
//! Precedence (last wins)
//! ----------------------
//!
//! This implements `docs/camdl-run-spec.md §1.3` exactly:
//!   1. Model parameter default …
```

**Write:**

```rust
//! Resolves the value of every parameter of a run from the sources the run
//! names: the compiled model, a `--starts` rule, a fit config's `[fixed]`
//! table, params files, a draw or sweep point, a scenario and `--fixed`. The
//! order in which they override one another is `docs/camdl-run-spec.md` §1.3.
//!
//! Each resolved value records where it came from
//! ([`ResolvedParameter::source`]). A parameter named by `--fixed` or in a
//! `--fixed-file` is also removed from the fit's `[estimate]` set, and the
//! run says so in an info notice.
```

The first version names functions that no longer resolve parameters, points at
line numbers that no longer exist, and restates a specification it no longer
matches. The second says what the module does today and leaves the precedence
order to the one document that defines it.

**Instead of:**

```rust
/// Deserialise a `Model` from a JSON string. gh#audit-C8: enforces
/// the `IrEnvelope` wrapper with version handshake. Returns
/// `IrError::VersionMismatch` if the envelope's `ir_version` doesn't
/// match the `IR_VERSION` baked from `ir/VERSION` at compile time.
pub fn from_str(s: &str) -> Result<Model, IrError> {
```

**Write:**

```rust
/// Reads a compiled model from the JSON `camdlc` writes.
///
/// The input must be a whole `IrEnvelope`, not a bare model. An envelope
/// whose `ir_version` differs from [`IR_VERSION`], the version this runtime
/// reads, is refused with [`IrError::VersionMismatch`].
pub fn from_str(s: &str) -> Result<Model, IrError> {
```

**Instead of:**

```ocaml
(** Transition destination form.
    - [DstSum] is the ordinary case: a `+`-separated list of destination
      compartments, each contributing +1 to stoichiometry. Singleton =
      classic, ≥ 2 = multi-dest (wave 1 / malaria #1).
    - [DstBranch] (wave 2 / malaria #2) is a probabilistic branch:
      `X --> { A : w_A, B : w_B } @ rate`. The expander desugars each
      branch into its own IR transition with rate `w_i * rate`. *)
type destination_form =
  | DstSum    of stoich_ref list
  | DstBranch of (stoich_ref * expr) list
```

**Write:**

```ocaml
(** Where a transition sends the individuals it moves.

    {@camdl[
      infection : S --> I @ beta * S * I / N              # DstSum [I]
      split     : I --> R + Q @ gamma * I                 # DstSum [R; Q]
      recovery  : I --> { R : 0.9, D : 0.1 } @ gamma * I  # DstBranch
      death     : Q --> @ mu * Q                          # DstSum []
    ]} *)
type destination_form =
  | DstSum    of stoich_ref list
      (** Every listed compartment gains one individual per firing. The empty
          list stands for two things the tree does not tell apart: a
          transition whose individuals leave the model ([Q --> @ mu * Q], a
          death or emigration), and a [via] transition with no arrow target,
          [I via hyper_erlang(...)], each of whose branches names its own
          destination. *)
  | DstBranch of (stoich_ref * expr) list
      (** One flow per branch, at the branch's weight times the transition's
          rate, so the source leaves at the rate times the sum of the
          weights. Weights that are all constants must sum to one (E423). *)
```

The first version mixes what the type means, how the expander lowers it, and
when each constructor was added. The second keeps only the first. The lowering
is still worth explaining, in an implementation comment at the expander's
lowering code.

### Checklist

Before handing in comments, their author, human or agent, can answer _yes_ to
each of these:

- Does every doc comment open with a sentence saying what the thing represents
  or does, not its shape?
- Is each design explained once, with specifications linked rather than
  restated?
- Is every comment free of history: no issue or proposal numbers, dates, work
  stream names, "previously", "now", "no longer" or "replaces"?
- Is every claim checked against the code, and every disagreement with a
  specification reported rather than settled by an edit?
- Is every scientific reason kept: what an operation means, why a harmless-
  looking change would alter results, why a combination is refused?
- Is the change to comments only, with every line of code as it was?
- Has every design problem found while writing been reported, not hidden behind
  a vague phrase or quietly fixed in the same change?
