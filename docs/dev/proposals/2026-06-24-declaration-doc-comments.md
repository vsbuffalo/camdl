---
status: draft
date: 2026-06-24
implemented: Stage 0 (spec fix) + Stage 1 (#' prose on parameters + compartments, OCaml-only, surfaced in `inspect`). Stages 1b/2/3 pending.
---

# Declaration Doc Comments (`#'`) and Parameter Annotation

## Problem

A `.camdl` model declares _what exists_ — parameters, compartments, dimensions,
transitions, observations — but has no place to record _what each thing means_.
The name and the dimension are all the structure carries. So authors (humans
and, increasingly, coding agents) jam the meaning into ordinary `#` comments,
which the lexer discards (`ocaml/lib/compiler/lexer.mll:154-155`). That
documentation is invisible to every downstream consumer: it can't appear in
`camdl inspect`, can't label a posterior plot, can't be surfaced in a fit
report.

Two failure modes follow.

**1. Meaning has no home, so it rots in throwaway comments.** A
`# transmission
rate` next to `beta : rate` is unparsed free text with no link
to `beta`. Rename `beta` and the comment silently orphans.

**2. Agents shadow-default values into comments.** A recurring anti-pattern in
agent-authored models:

```camdl
parameters {
  beta : rate   # FIXED = 0.3
}
```

This is a _shadow `[fixed]` entry_. camdl deliberately keeps parameter values
out of the model — they live in a `--params` TOML, which already has the
structured home for exactly this:

```toml
# from tests/fixtures/polio_afp_es/fit.toml
[fixed]
sigma = 0.2
k = 10.0
```

`# FIXED = 0.3` crosses the one cleavage the design rests on, and crosses it
twice: it puts a **value** _and_ an **estimation decision** (fixed-vs-free) into
the reusable model file, where both are unparsed, drift-prone, and duplicate a
field that has an authoritative home elsewhere.

**The spec actively licenses #2.** There is a live self-contradiction (verified:
`grep -n "Default values may\|never.*specified inside" docs/camdl-language-spec.md`):

- `docs/camdl-language-spec.md:504` — "Default values may optionally be
  specified in the model file."
- `docs/camdl-language-spec.md:622` — "Parameter values are **never** specified
  inside `.camdl` files."

622 is the design; 504 is stale. An agent that reads 504 is _told_ it may put
defaults in the model, and `# FIXED = 0.3` is what that looks like when there is
no syntactic slot for it. This is a pure `doc-vs-doc` defect — a one-line fix
that ships first and independently (Stage 0 below), because deleting 504 removes
the _license_ for the anti-pattern regardless of the rest of this work.

## The cleavage that decides every annotation

The model file is **intrinsic and reusable** — what exists, what it means, its
type/dimension; the same across every study that uses the model. The `--params`
TOML is **study-specific** — the numbers for _this_ analysis.

> Would this be the same across every study that uses this model? → model. Does
> it change per analysis? → `--params` TOML.

A model doc annotation describes the parameter; **it never carries a number the
TOML owns** (value, bound, prior, start, fixed-vs-free). The moment a "plausible
range" or "typical value" enters the model as a _machine-read field_, the
default is back and so is the drift.

**The line is between prose and tags, not between text and numbers.** Numbers in
_prose_ are documentation and are fine:
`#' mean latent period ≈ 5.1 days (Lauer
2020)` is a perfectly good description.
What is refused is a _structured, number-bearing tag_ that a consumer would
parse and act on — `@default 5.1`, `@plausible 0.1 0.5`, `@fixed`. A tag a tool
reads as a value is a shadow `[fixed]`; a sentence that mentions a number is
just a sentence.

## Design

A roxygen-style doc comment, `#'`, that attaches to the **following**
declaration. (R precedent: `#'` is roxygen2's doc-comment marker; the
recognition value for the target audience is high.) The model-level description
is _already_ handled by the existing `description = "..."` top-level declaration
(`parser.mly:140`, round-trips through `ir.ml:609` / `model.rs:164` /
`schema.json:26`), so this proposal covers the **per-declaration** docs that
have no home today.

### Syntax

`#'` introduces a doc line; consecutive `#'` lines immediately preceding a
declaration form that declaration's **doc block**. Attachment is _leading only_
(like roxygen) — there is one rule to hold in your head. Within the block,
free-text prose is the description; lines beginning with a recognized `@tag`
carry structured fields (Stage 2 — see staging).

```camdl
#' Two-patch SEIR for cVDPV2, fit to AFP + environmental surveillance.
description = "Two-patch SEIR for cVDPV2"   # model-level: existing field
time_unit   = 'days

dimensions {
  #' spatial patches — the two health districts under surveillance
  patch = [urban, rural]
}

compartments {
  #' fully susceptible
  S,
  #' latent: infected, not yet shedding
  E,
  #' infectious and shedding virus
  I,
  #' recovered, immune
  R
}

stratify(by = patch)

parameters {
  #' basic reproduction number (per patch)
  #' @symbol R₀
  R0[patch] : positive in [1.0, 6.0]

  #' mean latent period is 1/sigma
  #' @symbol σ
  sigma : rate in [0.01, 0.3]

  #' AFP reporting fraction — detected cases per true infection
  #' @symbol ρ
  rho : probability in [0.1, 0.9]
}

transitions {
  #' force of infection (frequency-dependent), per patch
  infection[p in patch] : S[p] --> E[p] @ R0[p] * gamma * S[p] * I[p] / N[p]
}

observations {
  #' weekly AFP case counts, negative-binomial reporting
  afp ~ ...
}
```

Three load-bearing details, each verified against the grammar:

- **Multi-line is already legal.** `compartments { S, I, R }` parses as
  `separated_list(COMMA, compartment_decl)` (`parser.mly:213-214`) and newlines
  are whitespace; the spec already shows the one-per-line form
  (`camdl-language-spec.md:477-481`). `#'`-per-compartment needs each
  compartment on its own line, which requires _no new block form_ — only the
  leading doc slot. (Note: `separated_list`, so **no trailing comma** after the
  last compartment — verified, and it remains a syntax error.)

- **Indexed params/compartments carry one doc for the whole family.**
  `R0[patch]` expands to `R0_urban`/`R0_rural`; both inherit the one doc,
  exactly as they already share one `bounds` (`camdl-language-spec.md:715`).

- **Patches are _not_ a declaration site.** A patch is a _level_ of a dimension
  (`dimensions { patch = [urban, rural] }` + `stratify(by = patch)`); `S_urban`
  is synthesized by expansion and has no source site to annotate. The
  documentable thing is the **dimension**. Per-level human labels
  (`urban = "Lagos metro"`) are _data_, not model structure — when levels come
  from `read(...)` there is no source site for them at all — and belong in a
  level→label table, not a source annotation.

### Vocabulary (deliberately tiny)

Keep the entire grammar in a head (the human-first DSL principle). Three
carriers:

| Form             | Carries                         | Where it lives | Example                       |
| ---------------- | ------------------------------- | -------------- | ----------------------------- |
| free-text prose  | the description / meaning       | model          | `#' per-capita recovery rate` |
| `@symbol <text>` | display label for plots/reports | model          | `#' @symbol γ`                |
| `@ref <text>`    | citation for the _definition_   | model          | `#' @ref Anderson & May 1991` |

Prose is Stage 1; the two tags are Stage 2 (staging below). `@symbol` is the
sleeper-valuable one: the fit-report / posterior-predictive / `fit predict` plot
surfaces render parameter _names_, and `β`/`γ`/`R₀` are far better axis labels
than `beta`/`gamma`/`R0`.

**`@symbol` rendering is decided, not punted.** The value is a **Unicode
literal** (`β`, `R₀`), not LaTeX. Rationale: `.camdl` is already UTF-8 (table
syntax uses `×`), the literal renders the same in a plaintext `inspect` dump and
a plot axis, and "the consumer decides" would mean the same model renders
differently across tools — the implicit-convention-in-the-head failure the
human-first principle warns against. A consumer that cannot render a glyph falls
back to the parameter name. No LaTeX in the DSL.

**Refused tags — the smell in a costume.** Any tag that smuggles a _machine-read
number_ into the model is rejected at parse time with a hard error naming the
migration (the "breaking change signposts the migration" bar). Unknown `@tag` →
hard error (no loose semantics):

> unknown doc tag `@plausible`; parameter values, bounds, and priors belong in
> the `--params` TOML, not the model — recognized tags are `@symbol`, `@ref`.

(Numbers in _prose_ are unaffected — only `@tag`-shaped fields are parsed.)

## Types first: the mechanism

### Lexer (`ocaml/lib/compiler/lexer.mll`)

Add a token for the doc line. **Ordering is load-bearing**: the existing comment
rule (`lexer.mll:155`) is `'#' [^'\n' '['] [^'\n']*`, whose char-class
`[^'\n'
'[']` _includes_ `'`, so `#' …` already matches it to end-of-line.
ocamllex breaks a longest-match tie by _source order_ (earliest rule wins). The
`DOC` rule must therefore sit **above** the comment rules (immediately after the
`#[` carve-out at `:153`), or every `#'` line is silently swallowed as a comment
and the feature is inert:

```
| "#["            { HASH_LBRACKET }
| "#'" [^'\n']*   { DOC <raw text after #'> }   (* MUST precede the comment rules *)
| '#'                       { token lexbuf }
| '#' [^'\n' '['] [^'\n']*  { token lexbuf }
```

The token carries the raw line text (trimmed). `#'days` does **not** collide
with the `'days` unit literal — the two-char `"#'"` prefix is matched before the
lexer reaches the unit rules. A lexer unit test asserting that `#'`, `#[`, `#`,
and a bare `#` each lex to the right token is **required** (this is a
three-member `#`-sigil family now; the disambiguation must be pinned).

### Parser (`ocaml/lib/compiler/parser.mly`)

Add a reusable `doc_opt` production — a direct analogue of the existing
`lineage_attr_opt` (`parser.mly:462`), which already attaches optional leading
metadata to a transition:

```
doc_opt:
  | (* empty *)              { [] }
  | ds = nonempty_list(DOC)  { ds }
```

Each documentable declaration gains a leading `d = doc_opt`. The surface is
larger than it looks: `param_decl` is **eight** productions (4 `PScalar` ×
{±bounds} × {±prior}, 4 `PIndexed` likewise — `parser.mly:231-264`), plus
`compartment_decl`, `dim_entry`, `transition_decl` (where `doc_opt` sits just
before the existing `lineage_attr_opt`), and `obs_decl`. Adding `doc_opt`
introduces **no Menhir LR conflicts** (verified by building a minimal grammar
with `doc_opt` prefixing `list(...)`, `separated_list(...)`, and the
double-optional `doc_opt
lineage_attr_opt IDENT …` — `.conflicts` empty in every
case; `DOC` is a fresh terminal with a disjoint first-set). The implementer must
still confirm the generated `parser.conflicts` is empty —
`ocaml/lib/compiler/dune` runs `menhir` **without `--strict`**, so a conflict
would be a silent warning, not a build failure.

**Dangling / misplaced `#'` is a hard parse error.** A `#'` block not
immediately followed by a documentable declaration (e.g. before a `}`, before
`stratify(...)`, or trailing inline after a complete decl) leaves a `DOC` token
the grammar cannot attach: the parse entry point
(`ocaml/lib/compiler/compiler.ml:76`) reports a located `E001: syntax error` and
the compile fails (verified: a `#'` before a block-closing `}` exits non-zero
with `error[E001]` at the orphan's line/col). It is **rejected, never silently
accepted** — which satisfies "no loose semantics." A _targeted_ diagnostic ("doc
comment `#'` must precede a declaration") would need an explicit error
production at each block tail, and that fights the grammar: with `doc_opt`
inside the list element (`list(param_decl)`, `param_decl = doc_opt
IDENT …`),
trailing `DOC`s are greedily consumed by the next element's `doc_opt` before any
tail production sees them, so a clean named error is not cheaply expressible.
The located E001 is the honest Stage-1 behavior; the targeted message is a noted
enhancement, not a blocker.

### Tag splitting (OCaml, plain function — Stage 2, not the grammar)

The grammar stays minimal: one `DOC` token, prose-or-`@tag` distinction lives in
a normal OCaml pass over the accumulated block. This keeps the tag vocabulary in
code where it gets good errors (unknown-`@tag` → E-code with the migration hint)
and is trivially extensible, rather than baking each tag into the parser.

```ocaml
type doc = { description : string option;   (* joined prose lines *)
             symbol      : string option;   (* @symbol, Stage 2 *)
             reference   : string option }  (* @ref,    Stage 2 *)
```

### Expander (`ocaml/lib/compiler/expander.ml`) — Stage 3 only

This is **not** needed for the Stage-1 `inspect` consumer: `camdlc inspect`
reads the _pre-expansion_ AST declarations directly off the expander context
(`ctx.comp_decls : compartment_decl list`, `ctx.param_decls`), so a doc on the
source declaration surfaces with no propagation at all (verified —
`inspect
--parameters` / `--compartments` print the doc, and an indexed
`R0[patch]`'s one doc rides both `R0_urban`/`R0_rural` because the lookup is by
the source decl). Propagation matters only when the **IR** carries per-leaf docs
for the Stage-3 Rust consumers (report/plot labels read the serialized IR, not
the AST):

Stratification copies the source declaration's `doc` to every expanded leaf
(`R0[patch]` → `R0_urban`, `R0_rural` each carry it), exactly as `pbounds` /
`pkind` / `punit` already propagate through resolution. This is **not a
one-liner** and **not** a single site: the relevant construction paths include
indexed-param expansion (`expander.ml:1267-1341`) and compartment-name expansion
(`expander.ml:955`), with the `read(...)`-loaded-levels case (no source site)
explicitly carrying no doc. Each path that constructs an expanded
param/compartment must thread the doc, or a documented entity silently loses its
doc post-expansion — the silently-dropped-at-one-path failure this codebase
treats as the villain. The test strategy asserts propagation for **both** an
indexed param **and** a stratified compartment, not just one example.

### IR schema (`ir/schema.json`, bump `ir/VERSION` 0.19 → 0.20) — Stage 3

A single shared optional sub-object, defined once and referenced from each
documented object, rather than three flat fields × five objects (name the
concept once; fewer places to drift):

```jsonc
"doc_block": {
  "type": ["object", "null"],
  "properties": {
    "text":   { "type": ["string", "null"] },   // joined description prose
    "symbol": { "type": ["string", "null"] },    // @symbol; null when absent
    "ref":    { "type": ["string", "null"] }     // @ref;    null when absent
  }
}
// parameter / compartment / dimension / transition / observation each gain:
"doc": { "$ref": "#/definitions/doc_block" }     // omitted entirely when absent
```

**Golden impact — stated honestly (the "byte-identical" claim was wrong).** Two
facts, both verified:

1. The IR envelope bakes the version string into every file
   (`serde.ml:1318/1326`); all 17 goldens carry `"ir_version": "0.19"` /
   `"validated_by":
   "ocaml-compiler-v0.19"`. The 0.20 bump rewrites those two
   lines in **every** golden. Goldens are therefore **not** byte-identical —
   they take a mechanical 2-line envelope bump regenerated by
   `make update-golden`.

2. The compact serializer emits absent optionals as `null`, it does **not** omit
   them (`serde.ml:1028-1029`; `sir_basic.ir.json:82` shows
   `"param_dim": null`). To keep the _model body_ of an undocumented model
   unchanged, `doc` **must** use the **omit-when-None** pattern that `origin`
   and `lineage` already use — OCaml
   `@ (match p.doc with None -> [] | Some d -> [("doc", doc_block_to_json d)])`
   (`serde.ml:1237`, `:310`), and Rust
   `#[serde(default, skip_serializing_if =
   "Option::is_none")]`
   (`model.rs:165`). With omit-when-None, an undocumented param/compartment line
   is byte-identical; only the envelope version lines move. Naively emitting
   `null` (the `param_dim` pattern) would append `"doc": null` to every line in
   every golden — the opposite of neutral. **Step verification:** the golden
   diff for an undocumented model must show _only_ the two envelope lines.

**Run identity — docs are presentation, excluded from the content hash.** The
Rust IR feeds the content-addressed `run_id`, hashed field-by-field
(`runid/src/ir_hash.rs:431-437`). A doc-text edit must **not** invalidate the
CAS cache or break paired-seed reproducibility, so `doc` is **not** added to any
`hash_into` / `ContentAddressed` impl. A test pins that two models differing
only in doc text produce the same `run_id`. (CLAUDE.md's run-identity
required-reading rule: a field that changes stored bytes is identity;
presentation is stripped. Docs are presentation.)

**Why the IR at all, and why Stage 3 ships _with_ its Rust consumer.** The
consumers that motivate the IR fields — fit-report param tables, posterior plot
labels — are Rust-side. An OCaml-only doc serves only `camdlc inspect`. Putting
the doc in the serialized IR is the point: it reaches Rust. But IR fields with
no live Rust reader are a stub primitive (the `Schedule::next_stop`
anti-pattern), so Stage 3 lands the schema/Rust fields **together with the first
Rust consumer** (the report or plot label), not as a round-trip-only shell. The
~116 Rust struct-literal sites (`Parameter {…}` ×108, `Compartment {…}` ×8,
mostly tests) get the new field defaulted; a `Parameter::new(name, value)` smart
constructor (the "parse at the boundary" preference) localizes the default
rather than threading `None` through every test.

## Implementation staging

The user asked to "knock it out as a quick commit." Stage 1 is that quick
commit; the rest are explicit, separately-reviewed follow-ups. The IR change is
_not_ in the quick commit, because its only real consumer is Rust (Stage 3).

**Stage 0 — spec fix (independent, ship first).** Delete the stale
`camdl-language-spec.md:504` clause. Pure doc-vs-doc, zero code, removes the
license for `# FIXED = 0.3` immediately.

**Stage 1 — `#'` prose on parameters + compartments, OCaml-only, surfaced in
`inspect` (the quick commit, landed).**

1. Lexer: `DOC` token, ordered **above** the comment rules.
2. Parser: `doc_opt` (the empty/`nonempty_list(DOC)` analogue of
   `lineage_attr_opt`) on `compartment_decl` and the eight `param_decl`
   productions. A misplaced `#'` is a hard `E001` parse error. No new Menhir
   conflict (the one pre-existing shift/reduce is unrelated — confirmed by a
   baseline build).
3. Carry prose on the OCaml `compartment_decl.cdoc` / `param_decl.pdoc` AST
   fields (`string option`, no `@tag` parsing yet). **No IR / Rust / schema /
   golden / VERSION change** — docs never reach the IR (proven byte-identical:
   documented model vs stripped twin serialize the same).
4. `camdlc inspect --parameters` / `--compartments` print the doc (it reads the
   pre-expansion AST off the expander context; `camdl inspect` forwards here).
5. `make test` green; five `test_compiler` cases cover compile / IR-neutrality /
   dangling-rejection / both inspect views.

**Stage 1b — extend to dimensions, transitions, observations.** Same `doc_opt`
mechanism on `dim_entry` / `transition_decl` (before `lineage_attr_opt`) /
`obs_decl`, plus their `inspect` rendering. Deferred because those decls lack a
detailed `inspect` view today, so their docs would land unrendered (a stub);
they ship when their view (or the Stage-3 IR consumer) exists.

**Stage 2 — the `@symbol` / `@ref` vocabulary.** The OCaml tag-split pass,
unknown-`@tag` and refused-tag hard errors, Unicode-symbol rule. Still
OCaml-side; `inspect` shows symbol/ref.

**Stage 3 — IR fields + first Rust consumer (shipped wired).** `doc_block` in
schema (omit-when-None) + `ir/VERSION` 0.20 + OCaml serde + Rust types
(`skip_serializing_if`, excluded from `run_id` hash) + the smart constructor +
one new documented golden + `update-golden`/`update-expected` + the first Rust
reader (fit-report `@symbol` axis label). Each Rust consumer beyond the first is
a `gh#NN` issue.

## Considered and rejected

**A `camdl check` lint flagging `# FIXED = 0.3` comments.** Rejected. It could
only ever be a _warning_ (the `#` text is discarded by the lexer; a hard error
on comment content is impossible), and CLAUDE.md is explicit that "warnings are
noise an agent will suppress and a non-specialist will skim." So a lint would be
a mechanism the governing doc classifies as noise, aimed at the exact audience
that ignores it. The root cause is already removed by Stage 0 (delete spec 504,
which _licenses_ the anti-pattern) + Stage 1 (give meaning a real home via
`#'`). If a retraining signal is wanted later, it is a separate `gh#NN`, not
part of this work.

**Extending `#[...]` to `#[doc: "..."]` instead of a new `#'` channel.**
Rejected. `#[lineage]` is a _semantic_ attribute — it threads `trlineage` into a
`transition_lineage` IR record that changes compilation (individual-sampling). A
doc comment is _non-semantic_ metadata. Conflating documentation into the
semantic-attribute namespace is a category error; and
`#[doc: "long prose, with
commas and \"quotes\""]` would need a string-valued,
escaping-aware attribute grammar that does not exist and is hostile to the
multi-line prose blocks that are the entire point. Keeping docs a distinct
channel is the "natural seam" — this is the one place the "reuse the existing
seam" rule does not apply, and the reason it could not serve is the category
difference plus the multi-line-prose requirement.

## Test strategy

- Lexer: `#'`, `#[`, `#`, bare `#` each lex to the right token (the
  three-member-sigil disambiguation).
- Parser: `#'` attaches to the right decl; multi-line block joins; dangling /
  trailing `#'` yields the _named_ error (not E001); (Stage 2) unknown `@tag`
  and refused tags (`@default`, `@plausible`, `@fixed`) error with the migration
  hint.
- Expander: documented `R0[patch]` → every leaf carries the doc; documented
  stratified compartment → every leaf carries the doc.
- (Stage 3) Round-trip: documented model → IR → back, doc preserved; an
  _undocumented_ golden's diff shows **only** the two envelope version lines;
  two models differing only in doc text produce the **same** `run_id`.
- `inspect`: doc (and Stage 2 symbol) appear in `--summary`.

## Decisions (resolved — no open questions)

- **Attachment**: leading-only, position-attached (no name duplication).
  Dangling / trailing `#'` is a hard `E001` parse error (rejected, not silent);
  a targeted message is a noted enhancement (it fights the greedy `doc_opt` list
  boundary).
- **Scope**: per-declaration docs on parameters, compartments, dimensions,
  transitions, observations. Model-level stays the existing
  `description = "..."`.
- **Vocabulary**: prose (Stage 1) + `@symbol` + `@ref` (Stage 2). Every
  machine-read number-bearing tag is a hard error naming the TOML migration;
  numbers in prose are unaffected.
- **`@symbol` rendering**: Unicode literal, not LaTeX; consumers that can't
  render fall back to the parameter name.
- **IR field shape**: one shared optional `doc_block { text, symbol, ref }`
  sub-object referenced from each documented object; serialized
  **omit-when-None**; **excluded** from the `run_id` content hash (presentation,
  not identity).
- **`#'` vs `#[doc:]`**: a dedicated `#'` channel (rationale in "Considered and
  rejected").
- **Sequencing**: Stage 0 (spec fix) and Stage 1 (`#'` prose, OCaml-only, wired
  to `inspect`) are the quick commit; tags (Stage 2) and the IR/Rust flow
  shipped with its first Rust consumer (Stage 3) are separate reviewed commits.
  The IR is _not_ in the quick commit — its only real consumer is Rust.
