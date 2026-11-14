# Proposal: principles for the published comparison viewer

**Status:** Proposal **Date:** 2026-11-13 **Motivation:** camdl fits are about
to be published as immutable bundles and compared in a public viewer; this sets
the rules that keep every pixel of that viewer traceable to a typed value camdl
produced, so that the design mockups are a specification and not decoration.

**Drafted with Claude** from the design conversation of 2026-11-13. Code
references are to camdl at `c1f27d9d` (2026-11-13), to the compare cleanup
branch `worktree-peppy-jingling-kite` where named, and to camdl-scope at
`3d07fed` (2026-11-11, its remote `main`).

This is proposal 0 of a set. It defines vocabulary and rules. It specifies no
file format and no code; those are in the proposals listed under
[The set](#the-set). Where a rule needs something camdl does not have today, the
rule says so and names the proposal that owes it.

## The rule this proposal sets

**Nothing appears in the published viewer unless it is a typed value in a
bundle, or a fixed rendering of one.** The viewer selects, orders, windows and
formats. It does not estimate, score, threshold or judge. A reader who distrusts
a number can follow it to the artifact field it came from, and from there to the
command that wrote it.

Three consequences shape everything else:

1. Every statistical decision is made once, in camdl, and recorded.
   `2026-11-08-cli-reporting.md` set this rule for the terminal ("every fact a
   command shows is a value in a typed report before it is text") and
   implemented it for `fit run` and `fit summary`. `camdl compare` does not yet
   satisfy it: it has no report type, and computes its score differences
   separately inside each of three renderers (`rust/crates/cli/src/compare.rs`).
   This set extends the rule to `compare` first (proposal 3) and then to a
   public page.
2. A design mockup is accepted only when each element on it names its source. An
   element with no source is a requirement on camdl or it is removed.
3. The page is built so that the wrong conclusion is hard to reach: limits sit
   beside claims, untested ranges are marked, and every declared assumption is
   one click away.

## Why now

Scope is a local instrument for the person running fits. The published viewer is
read by people who did not run them: collaborators, reviewers, funders,
decision-makers. They cannot inspect the run directory, and many will read only
the first screen. The cost of a misleading first screen is higher, and the
reader's ability to check it is lower.

The motivating failure is the Columbia debris-assessment slide: a reassuring
title, "Review of Test Data Indicates Conservatism for Tile Penetration", with
the fact that the flight condition was "significantly outside of test database"
in a low-level bullet (Columbia Accident Investigation Board, _Report_, vol. 1,
August 2003, p. 191; discussed in E. Tufte,
["Columbia Accident Investigation Board: the Boeing PowerPoint slide"](https://www.edwardtufte.com/notebook/columbia-accident-investigation-board-the-boeing-powerpoint-slide/)).
The output travelled; the assumption did not. The audit of our own first mockups
found elements of the same shape (see [Mock debt](#mock-debt)).

## Vocabulary

Several of these words are already in use in camdl with other meanings. Each is
defined here against the existing one.

- **Bundle.** The immutable published form of one fit: model source, the
  declared model (proposal 4), fit configuration facts, posterior summaries,
  predictive summaries, the score artifact, the data the fit saw, and a bundle
  record. Content-addressed. Not the `camdl mre` reproduction bundle
  (`2026-06-09-mre-bundle.md`), which packages what is needed to re-run a
  failure. Its relation to `camdl store pack`
  (`2026-08-21-cas-lifecycle-labels-and-sharing.md` §3, under revision) and to
  the sealed fit of
  `2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` is settled by
  proposal 1, which either builds on them or supersedes them.
- **Bundle record.** The file inside a bundle listing its contents, their
  hashes, the schema version and the camdl version that wrote it.
- **Repository index.** The file an author commits to a repository, listing the
  bundles published from it, the comparison sets they belong to, and each
  bundle's status (current, superseded, withdrawn). It is the one mutable file.
- **Comparison set.** Bundles whose score artifacts pass, against each other,
  the preflight checks `camdl compare` runs. Being comparable is a relation
  among members and not a property of one bundle. On the compare cleanup branch
  those checks are: same observations by content hash, same scored steps, same
  filter settings, compatible score provenance, and the stream rules of
  `2026-11-13-compare-on-selected-streams.md`.
- **Published comparison.** A file in one author's repository naming member
  bundles (from any repositories), carrying the report `camdl compare` computed
  for them on the author's machine, and that author's annotations. It has an
  author.
- **Ad hoc comparison.** Members named in the viewer URL. It has no author, no
  annotations and no report: the viewer shows each member's own numbers side by
  side and no score differences, because nothing has computed them (R3).
- **Report.** The typed result of scoring a comparison: per-model rows,
  preflight results, refusals, overrides, the camdl version. It does not exist
  as a type today; proposal 3 owes it.
- **Annotation.** Text written by a comparison's author, anchored to a specific
  computed result. It lives in the comparison file.
- **Description.** A fit's own free-text account of itself, from the
  `[provenance]` table of its fit configuration, which is not hashed
  (`rust/crates/cli/src/fit/cas.rs:417-437`).
- **Publish sidecar.** A file read by `camdl publish` that supplies presentation
  for one bundle: geography, display labels, layout hints. Not the fit sidecar
  `fit.meta.json`.
- **Limits.** Computed statements that bound what a comparison can support.
- **Assumption ledger.** The typed list of what a model declares it assumes,
  generated from the declared model and the fit configuration facts. It lists
  declared assumptions only: fixed values, priors, the observation model, data,
  scenario. Assumptions made by omission (homogeneous mixing, exponentially
  distributed dwell times, a closed population) are in no ledger, and every
  count of assumptions is labelled "declared".
- **Lead time.** How far ahead of the last conditioning observation a prediction
  is made. **Scored steps** is the number of observations scored. `compare` uses
  "horizon" for the second; this set does not use the word.

A bundle is **publishable** when it is a complete fit with a persisted verdict
and provenance for every method it reports.

## The rules

Each rule states what it requires and how it is enforced. "Enforced" means a
build or test fails. A rule that cannot be enforced that way is labelled a
review convention.

### R1. Every displayed element names its source

Each screen has a view contract: a table with one row per visible element.

| Column  | Meaning                                      |
| ------- | -------------------------------------------- |
| Element | What the reader sees                         |
| Source  | Type and field path it reads                 |
| Class   | Authored, computed, or fixed wording (R2)    |
| Absent  | What is shown when the field is missing (R5) |

Enforced by: a test that parses each view contract and checks every Source path
resolves in the generated schema (R4); and a fixture test that every rendered
text node carries an attribute naming a schema path or a fixed-wording key.
Typed component props alone do not enforce this: they stop a wrong-typed value,
not a literal string.

### R2. Three provenance classes, visibly distinct

| Class         | What it is                                | Treatment                          |
| ------------- | ----------------------------------------- | ---------------------------------- |
| Computed      | Numbers and typed labels written by camdl | Default ink                        |
| Authored      | Descriptions and annotations              | Green, labelled, attributed, dated |
| Fixed wording | The viewer's sentence for a typed value   | Default ink; one lookup table      |

Rules for annotations:

- Only the comparison's author annotates. A member bundle's publisher is shown
  as its source ("source: owner/repo@commit; not a participant in this
  comparison") and never as a co-author of the comparison.
- Each is anchored to one computed result (a comparison, model, stream or
  quantity), never to a page.
- It sits on top of the computed value and cannot hide, replace or restate it.
- It records the hash of the report it was written against. If the comparison is
  republished it is shown as written for an earlier version.
- It is attributed to the owner of the repository that holds the comparison
  file. There is no signature: the repository in the URL is the only identity.
- It is distinguished by a label and icon as well as colour.

Labels that come from a computed gate ("within filter noise", a failed fairness
check, "not converged") are computed, never authored. If they could be
annotations, a reader could not tell a checked result from an opinion. Today
such labels are strings (`evidence_label` returns `&'static str`,
`rust/crates/cli/src/evidence.rs:259`); R4 requires each to be a Rust enum.

Enforced by: annotations are a distinct schema type rendered by one component; a
fixture test that an annotated element still contains its computed value.

### R3. The viewer computes nothing statistical

Allowed in the viewer: choosing which models and streams to show, sorting,
choosing a time window, unit and number formatting, laying out a figure
(including mapping values to pixels).

Not allowed: any estimate, score, difference of scores, interval, coverage,
calibration statistic, convergence statistic, or threshold.

**Scores are computed once, natively, by `camdl`, and stored.** A bundle carries
its own score artifact; a published comparison carries the report its author's
`camdl compare` produced. The viewer never scores. An ad hoc comparison
therefore shows no score differences. Scoring in the browser, by camdl's code
compiled to WASM, is a possible later addition and not part of this set's first
release; it would need a scoring crate with no file, process, thread or
system-RNG dependency, which does not exist (the comparison code is in the `cli`
crate, a binary with no library target, `rust/crates/cli/Cargo.toml:24`, and
derives traces by spawning `camdl pfilter`).

Scope keeps this rule in part: `verdict.py` "holds no threshold" and
`compare.py` "never recomputes prequential scores". It also computes statistics
locally that the published viewer may not: in `diagnostics.py`, R̂ and ESS with
arviz, Monte Carlo standard errors, a plateau test with its own threshold,
prior-to-posterior contraction, and a bound-pressure flag. In published mode
each of these comes from the bundle or is not shown. camdl has no contraction
measure today; proposal 5 owes one before any "prior-driven" label appears.

Enforced by: the viewer package has no statistics dependency; score and interval
values arrive as a branded type that only the scale and format functions of one
module accept, so arithmetic on them elsewhere does not type-check.

### R4. Types flow from Rust

The bundle schema, the report and the diff report are Rust types that derive
both serialization directions. TypeScript types are generated from them. Every
label R2 calls computed is an enum, handled exhaustively, so a new variant fails
the viewer's build until someone decides how it is shown. Each schema carries a
version.

Build-time exhaustiveness protects only a rebuilt viewer. Bundles are immutable
and the viewer is deployed, so a bundle written by a newer camdl reaches an
older viewer at run time. The viewer therefore validates every loaded document
against the schema, refuses a major version it does not know, and renders an
unknown enum value as "this viewer does not know the value `x`" (R5), never as
blank.

No Rust-to-TypeScript generator is in the workspace today (none of `ts-rs`,
`schemars`, `specta`, `typeshare`, `tsify` is in `rust/Cargo.lock`), and
`compare`'s JSON is hand-built with no schema version. Proposal 3 owes the typed
report; proposal 7 chooses the generator. Scope's `make types` generates from
its Python API, where the verdict kind is a plain string.

Enforced by: a CI job regenerates the TypeScript types and fails on a diff;
schema validation runs on every fixture bundle.

### R5. Missing is a designed state

`Option` and absent artifacts each have a rendering: not recorded, not defined
in this model, not checked (with the reason), refused (with the refusal),
unknown to this viewer. A blank, a zero or a silently omitted row is a bug.
Older bundles that predate a field show "not recorded", and never pass a check
they cannot satisfy.

Enforced by: R15's fixtures include, for every optional field, a bundle in which
it is absent.

### R6. Fairness before results

The preflight results are shown before any score. A refused comparison shows the
refusal and no Δ columns. An override used when a trace was produced or scored
is recorded in the artifact and shown. (On `main` overrides are untyped flags;
the compare cleanup branch adds an `overrides` list to the comparison output and
to the trace.) Today the first failed check exits, so later results never exist;
proposal 3's report evaluates and records every check.

"Fair" is not "adequate". The page says so next to the fairness badge: a fair
comparison of four poor models is still fair.

Enforced by: a fixture test that a refused comparison renders no Δ cell, and
that preflight precedes scores in document order.

### R7. Limits sit beside interpretation, at equal weight

Wherever an author's interpretation appears, a computed limits block appears
next to it, at comparable size, and stays directly beneath it when the layout
stacks at phone width. It states at least: the scored lead time, the number of
scored steps, any member admitted under a convergence override, any quantity
that is prior-driven, and the number of declared assumptions shared by all
members. The author cannot edit or suppress it.

Enforced by: the limits block is a typed value produced by camdl (proposal 5); a
layout test that it renders wherever the annotation component does, at both
widths.

### R8. The tested range is marked

Accuracy is measured at a lead time. Any prediction or projection shown at a
lead time that was not scored is marked as untested, on the figure itself. The
default held-out view shows predictions at the scored lead time and does not
show free-running projections beyond it.

Scoring at longer lead times is designed and unbuilt (forecast-mode scoring,
`2026-08-29-honest-predictive-evaluation.md` §3.7.2, gh#808). The first release
marks the untested range and does not wait for it.

Enforced by: each predictive series carries a typed scored lead time (owed by
proposal 2); a fixture test that a series beyond it renders the untested mark.

### R9. Every declared assumption is indicated and one click away

Every figure carries a count of the declared assumptions that feed it, labelled
as declared. One click opens that subset of the assumption ledger. Assumptions
shared by every member of a comparison are listed explicitly, because a
comparison cannot test them and a difference view cannot show them. Members are
matched by name, so two members making one assumption under different names are
not listed as shared unless the comparison declares a rename (proposal 4); the
shared list says so.

This is a review convention until proposal 5 defines the ledger and the
output-to-assumption dependencies; it then becomes a fixture test.

### R10. Context travels with the figure

An exported or embedded figure carries a footer: what it shows, the scored lead
time, the data date, the bundle hash and a permalink. A figure never leaves the
site without the facts that bound it.

Enforced by: a snapshot test on the export function's footer fields.

### R11. Immutable, pinned, withdrawable

A bundle is never edited. Links pin a commit and carry the bundle's content
hash, and the viewer checks the fetched bundle against that hash.

A pinned URL returns fixed content, so it cannot carry a withdrawal. Status
(current, superseded, withdrawn) is read on every load from the repository index
at the default branch of the bundle's home repository, for every member of a
comparison. When that fetch fails the page says "status could not be checked". A
withdrawn or superseded bundle says so prominently and points to its
replacement.

Permanence is conditional and the page says on what: a link works while the
author's repository exists, is public, and has not rewritten the pinned commit.
camdl does not host bundles.

Enforced by: fixture tests for a hash mismatch, a withdrawn member, and an
unreachable index.

### R12. Presentation never enters identity

What is supplied at publish time never changes a model identity, a fit identity
or a run id:

- A fit's description lives in `[provenance]` of its fit configuration, which
  the fit digest excludes along with `output_dir`. The rest of the problem half
  of the configuration is hashed whole (`fit/cas.rs:417-437`), so nothing else
  presentational may live there.
- Comparison annotations live in the comparison file.
- Geography, display labels and layout hints live in the publish sidecar. gh#306
  proposes packing per-dimension geometry into fit artifacts instead; proposal 6
  settles which.

This rule covers presentation supplied at publish time. Text declared inside the
model that the model hash covers is identity today: `normalize_for_hash` removes
only `output.format` and `simulation.time_semantics`
(`rust/crates/runid/src/inputs.rs:332-337`). A `#'` doc comment is not such
text, a `@symbol` tag included: two sources that differ only in doc comments
compile to byte-equal model JSON (`ocaml/test/test_declared_emit.ml`, "a doc
comment changes renderings only, and no model").

A run id also folds in the IR version string, so every stored run and fit gets a
new address when `ir/VERSION` changes. That is a property of the camdl version
and not of anything supplied at publish time.

Enforced by: a `camdl publish` test that changing each of the three leaves the
model digest, fit digest and run id byte-identical.

### R13. Bundles are untrusted input

The viewer treats bundle content as data. No bundle text is rendered as HTML.
The source repository and commit are always visible, with a plain statement that
camdl has not reviewed it. A curated gallery is visually distinct from an
arbitrary link.

Enforced by: a lint ban on raw-HTML insertion; a strict content security policy;
a fixture bundle with markup in every string field.

### R14. One figure grammar across tools

A predictive band, a Δelpd interval and a flow diagram look the same in scope
and in the published viewer, with the same colours meaning the same things. The
chrome differs: scope is dense and terminal-like, the published viewer reads as
a publication. Anything that must be told apart by colour also differs in shape
or line style.

A review convention, unless both viewers come to import one figure package.

### R15. Fixtures decide

A set of fixture bundles, real and pathological, must render on every screen, at
desktop and phone width for the first screen: one model, forty models, a member
admitted under a convergence override, a refused comparison, a model missing a
quantity the others define, a bundle from an older schema, a bundle with every
optional field absent, a withdrawn member. A mockup is a proposal until it
renders from a fixture.

Enforced by: these are CI tests with assertions on the rendered document, not
screenshots for review.

## Published bundles are read forever

The repository's alpha rule is no fallback readers: a format changes and the old
one is gone. Published bundles are the exception, because they will be cited.

- The bundle schema, the report schema and the declared-model schema are each a
  separately versioned surface. From version 1 of each, every published version
  stays readable by the viewer.
- A bundle is self-sufficient for viewing. Nothing the viewer reads requires
  camdl's runtime reader for compiled models, which refuses any IR version but
  its own (`rust/crates/ir/src/envelope.rs:126-133`). The declared model is
  carried as its own versioned document. A compiled model may be included for
  reproduction; the viewer does not read it.
- Because scores are stored and not recomputed (R3), a published page shows the
  same numbers for as long as it renders, and each report names the camdl
  version that computed it.

## Convergence

The viewer shows camdl's verdict and adds none of its own. camdl records a
verdict per fitting method, with six kinds
(`rust/crates/cli/src/fit/report/facts.rs:198-214`), and for methods without
chains a separate gate result with six values
(`rust/crates/cli/src/fit/method_result.rs:152-166`). A fit that has not
finished has no verdict; that is a run status, not a verdict kind.

| Verdict                                   | Publishable                 | In a comparison                                            |
| ----------------------------------------- | --------------------------- | ---------------------------------------------------------- |
| Converged                                 | yes                         | yes                                                        |
| Not converged                             | yes, marked loudly          | refused by `camdl compare` unless overridden               |
| Chains frozen                             | yes, marked loudly          | refused unless overridden                                  |
| Unassessable                              | yes, marked loudly          | refused unless overridden                                  |
| Not assessed                              | yes, marked with the reason | yes, marked with camdl's recorded reason, listed in limits |
| Not applicable, gate passed               | yes                         | yes                                                        |
| Not applicable, gate failed or incomplete | yes, marked loudly          | yes, marked, listed in limits                              |
| Not applicable, gate not assessed         | yes, marked                 | yes, marked, listed in limits                              |
| No report recorded; run failed or running | no                          | no                                                         |

Three things here are requirements on camdl, owed by proposal 3:

- `camdl compare` has no convergence check today (gh#716 records a chain that
  never moved reordering a table by 13 log-score units). It gains a preflight
  that refuses a member whose verdict is not converged, chains frozen or
  unassessable. The override is recorded in the report, and every score from
  such a member is labelled as computed from a sample that did not converge.
- The verdict records the rule it was decided by. Today `Converged` carries the
  largest R̂ and its parameter but not the threshold, a constant 1.05
  (`method_result.rs:787`), which gh#725 proposes to tighten. Without the rule
  in the record, bundles published before and after such a change would both
  read "Converged" under different rules.
- ODE fits emit no score artifact (gh#312), so they are publishable and never
  comparable until that is fixed.

The word shown for a passing fit is "Converged". It is not "validated":
convergence says the sampler agreed with itself, not that the model is right.

## Loading

There is no central repository and no server. The viewer is static files, served
at `models.camdl.dev`. Paths are versioned and carry the content hash; proposal
7 specifies them, and they do not change after launch. Three shapes of URL:

1. One repository at a commit: the browser reads that repository's index.
2. An ad hoc comparison: several bundles named in the URL.
3. A published comparison: one file, in one repository, naming bundles anywhere.

Bundles are fetched by the reader's browser with plain GET requests. Observed on
2026-11-13 with a cross-origin request: `raw.githubusercontent.com` answers with
`access-control-allow-origin: *` at a branch and at a commit; a release-asset
URL redirects to a host that sends no such header. So bundles live in the
repository, a Pages site, or a bucket the author controls. This is observed
behaviour and not a documented guarantee; proposal 7 records rate and size
limits before relying on it.

A server can be added later without rework if the viewer obtains data only
through one "load bundle from reference" interface.

The first screen works at phone width: per-model columns stack, and each limits
block stays under its interpretation (R7). Other screens are desktop layouts.

## Mock debt

Elements in the 2026-11-13 mockups that have no source in camdl today. Each
becomes a requirement in the named proposal, or is removed.

| Element                                  | Needs                                                                                                                        | Proposal |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------- |
| Calibration reading ("U-shaped")         | Author annotation, or a typed finding from camdl                                                                             | 3, 6     |
| Pooled calibration and coverage          | Removed. Calibration and CRPS are reported per stream, with no combined figure (`2026-11-13-compare-on-selected-streams.md`) | —        |
| Per-stream Δelpd                         | Exists in the trace; must be labelled marginal, no total                                                                     | 3        |
| "Prior-driven" flag                      | A contraction measure and rule defined in camdl                                                                              | 5        |
| Declared-assumption counts per figure    | Ledger plus output-to-assumption dependencies                                                                                | 5        |
| "Shared by all models" list              | N-way diff                                                                                                                   | 4        |
| Structure difference sentence            | Typed diff report                                                                                                            | 4        |
| One-week-ahead versus free-running bands | Both exist (`one_step`, `free_forward`); a typed scored lead time on each                                                    | 2        |
| Projection beyond scored lead time       | Not shown by default; marked untested (R8); gh#808 for scoring it                                                            | 2        |
| Reproduction number by province          | A declared quantity indexed by the dimension, shown across models (gh#707)                                                   | 5, 6     |
| Headline sentence                        | Author annotation only; no generated claim                                                                                   | 6        |
| Data "as of" date                        | Recorded with the data digest; digests exist, no date does                                                                   | 2        |

## The set

| # | Proposal                                                                                            | Depends on                 |
| - | --------------------------------------------------------------------------------------------------- | -------------------------- |
| 0 | Principles (this document)                                                                          | —                          |
| 1 | Bundle, repository index and `camdl publish`                                                        | 0, 2, 4                    |
| 2 | Score and predictive artifacts: persisted trace, scored lead time, data digests and dates           | compare cleanup branch     |
| 3 | Typed comparison report: every preflight recorded, convergence preflight, verdict rule, enum labels | 2                          |
| 4 | Model structure diff                                                                                | 0 (steps 1–7); 1 to bundle |
| 5 | Assumption ledger and limits                                                                        | 1, 3, 4                    |
| 6 | Presentation: comparison file and annotations, publish sidecar                                      | 1, 3                       |
| 7 | Viewer: loader, URL scheme, type generator, view contracts, fixtures, export                        | 1, 3, 4, 5, 6              |

Parked: scoring in the browser; export to forecast-hub formats; a registry;
cloud fitting (`2026-06-23-cloud-fit-dispatch.md`, a draft). Proposal 1 should
keep a dense quantile set per step so hub export stays possible.
