# Proposal: principles for the published comparison viewer

**Status:** Proposal **Date:** 2026-11-13 **Motivation:** camdl fits are about
to be published as immutable bundles and compared in a public viewer; this sets
the rules that keep every pixel of that viewer traceable to a typed value camdl
produced, so that the design mockups are a specification and not decoration.

**Drafted with Claude** from the design conversation of 2026-11-13. Code
references are to `main` at `2e2f663` (2026-11-09) and to camdl-scope at
`3d07fed` (2026-11-11). Nothing here has been checked against the compare
cleanup branch (`worktree-peppy-jingling-kite`); where a field is said to exist,
verify it there before relying on it.

This is proposal 0 of a set. It defines vocabulary and rules. It specifies no
file format and no code; those are in the proposals listed under
[The set](#the-set).

## The rule this proposal sets

**Nothing appears in the published viewer unless it is a typed value in a
bundle, or a fixed rendering of one.** The viewer selects, orders, windows and
formats. It does not estimate, score, threshold or judge. A reader who distrusts
a number can follow it to the artifact field it came from, and from there to the
command that wrote it.

Three consequences shape everything else:

1. Every statistical decision is made once, in camdl, and recorded. This is the
   rule `2026-11-08-cli-reporting.md` set for the terminal, extended to a public
   page.
2. A design mockup is accepted only when each element on it names its source. An
   element with no source is a requirement on camdl or it is removed.
3. The page is built so that the wrong conclusion is hard to reach: limits sit
   beside claims, untested ranges are marked, and every assumption is one click
   away.

## Why now

Scope is a local instrument for the person running fits. The published viewer is
read by people who did not run them: collaborators, reviewers, funders,
decision-makers. They cannot inspect the run directory, and many will read only
the first screen. The cost of a misleading first screen is higher, and the
reader's ability to check it is lower.

The motivating failure is the Columbia debris-assessment slide: a reassuring
title, with the fact that the test data lay far outside the conditions being
assessed in a low-level bullet. The output travelled; the assumption did not.
The audit of our own first mockups found the same shape seven times (see
[Mock debt](#mock-debt)).

## Vocabulary

- **Bundle.** The immutable published form of one fit: model source, compiled
  model, fit configuration, posterior summaries, predictive summaries, score
  artifact, and a manifest. Content-addressed.
- **Manifest.** The index file an author commits to a repository, listing the
  bundles published from it and the comparison sets they belong to.
- **Comparison set.** Bundles that can be scored against each other: same
  observations, same scoring window, same streams, same conditioning, same
  filter settings.
- **Published comparison.** A file in one author's repository naming member
  bundles (from any repositories) plus that author's annotations. It has an
  author.
- **Ad hoc comparison.** Members named in the viewer URL. It has no author and
  therefore no interpretation layer.
- **Report.** The typed result of scoring a comparison: per-model rows,
  preflight results, refusals, caveats.
- **Annotation.** Text written by the publisher, anchored to a specific computed
  result.
- **Limits.** Computed statements that bound what a comparison can support.
- **Assumption ledger.** The typed list of what a model assumes, generated from
  the compiled model and fit configuration.

Two tiers, kept separate everywhere:

- **Publishable:** a complete fit with a persisted verdict and provenance. It
  can be viewed.
- **Comparable:** publishable, and it carries a score artifact that passes the
  fairness preflights against the other members.

## The rules

Each rule states what it requires and how it is enforced. "Enforced" means a
build or test fails, not that a reviewer is expected to notice.

### R1. Every displayed element names its source

Each screen has a view contract: a table with one row per visible element.

| Column  | Meaning                                      |
| ------- | -------------------------------------------- |
| Element | What the reader sees                         |
| Source  | Type and field path it reads                 |
| Class   | Authored, computed, or fixed wording (R2)    |
| Absent  | What is shown when the field is missing (R5) |

Enforced by: the viewer's components take typed props generated from the bundle
schema (R4); a component cannot be given a value that is not a schema field. The
view contract tables live in the viewer proposal and are reviewed with it.

### R2. Three provenance classes, visibly distinct

| Class         | What it is                                       | Treatment                      |
| ------------- | ------------------------------------------------ | ------------------------------ |
| Computed      | Numbers and typed labels written by camdl        | Default ink                    |
| Authored      | The publisher's words: descriptions, annotations | Green, labelled, signed, dated |
| Fixed wording | The viewer's sentence for a typed value          | Default ink; one lookup table  |

Rules for annotations:

- Each is anchored to one computed result (a comparison, model, stream or
  quantity), never to a page.
- It sits on top of the computed value and cannot hide, replace or restate it.
- It records the bundle hash it was written against. If the results are
  republished it is shown as written for an earlier version.
- It lives in a sidecar read by `camdl publish` and never enters fit or model
  identity (R12).
- It is distinguished by a label and icon as well as colour.
- Only the publisher annotates, for now.

Labels that come from a computed gate ("within filter noise", a failed fairness
check, "not converged") are computed, never authored. If they could be
annotations, a reader could not tell a checked result from an opinion.

### R3. The viewer computes nothing statistical

Allowed in the viewer: choosing which models and streams to show, sorting,
choosing a time window, unit and number formatting, laying out a figure.

Not allowed: any estimate, score, difference of scores, interval, coverage,
calibration statistic, convergence statistic, or threshold. These come from
camdl, either precomputed in the bundle or by running camdl's own code compiled
to WASM.

This is scope's existing rule. Its modules already state it: `verdict.py` "holds
no threshold", `compare.py` "never recomputes prequential scores". One current
exception to retire: `diagnostics.py` computes R̂ and ESS with arviz. In
published mode those numbers must come from the bundle.

Enforced by: the viewer has no statistics dependency; a lint rule on the
component package forbids arithmetic on score fields outside the WASM boundary.

### R4. Types flow from Rust

The bundle schema, the report and the diff report are Rust types. TypeScript
types are generated from them. Every enum is handled exhaustively, so a new
variant in camdl fails the viewer's build until someone decides how it is shown.
Each schema carries a version; the viewer refuses a major version it does not
know and says so.

Scope does this today from its Python API (`make types`). The published viewer
does it from Rust.

### R5. Missing is a designed state

`Option` and absent artifacts each have a rendering: not recorded, not defined
in this model, not checked (with the reason), refused (with the refusal). A
blank, a zero or a silently omitted row is a bug. Older bundles that predate a
field show "not recorded", and never pass a check they cannot satisfy.

### R6. Fairness before results

The preflight results are shown before any score. A refused comparison shows the
refusal and no Δ columns. An override used when a trace was produced or scored
is recorded in the artifact and shown.

"Fair" is not "adequate". The page says so next to the fairness badge: a fair
comparison of four poor models is still fair.

### R7. Limits sit beside interpretation, at equal weight

Wherever an author's interpretation appears, a computed limits block appears
next to it, at comparable size. It states at least: the scored horizon, the
number of scored steps, any member that did not converge, any quantity that is
prior-driven, and the number of assumptions shared by all members. The author
cannot edit or suppress it.

### R8. The tested range is marked

Accuracy is measured at a horizon. Any prediction or projection shown at a
horizon that was not scored is marked as untested, on the figure itself. The
default held-out view shows predictions at the scored horizon.

### R9. Every assumption is indicated and one click away

Every figure carries a count of the assumptions that feed it. One click opens
that subset of the assumption ledger: fixed values, priors, observation model,
data, scenario. Assumptions shared by every member of a comparison are listed
explicitly, because a comparison cannot test them and a difference view cannot
show them.

### R10. Context travels with the figure

An exported or embedded figure carries a footer: what it shows, the scored
horizon, the data date, the bundle hash and a permalink. A figure never leaves
the site without the facts that bound it.

### R11. Immutable, pinned, withdrawable

A bundle is never edited. Links pin a commit. A bundle can be marked superseded
or withdrawn by its publisher; the old link then says so prominently and points
to the replacement. URLs are permanent from launch, because they will be cited.

### R12. Presentation never enters identity

Annotations, geography, display names and layout hints are read at publish time
from sidecars. Changing them never changes a model identity, a fit identity or a
run id. (Today the whole canonicalized `fit.toml` is hashed into the fit digest,
`runid/src/inputs.rs:425-445`, so none of this may live there.)

### R13. Bundles are untrusted input

The viewer treats bundle content as data. No bundle text is rendered as HTML.
The source repository and commit are always visible, with a plain statement that
camdl has not reviewed it. A curated gallery is visually distinct from an
arbitrary link.

### R14. One figure grammar across tools

A predictive band, a Δelpd interval and a flow diagram look the same in scope
and in the published viewer, with the same colours meaning the same things. The
chrome differs: scope is dense and terminal-like, the published viewer reads as
a publication. Anything that must be told apart by colour also differs in shape
or line style.

### R15. Fixtures decide

A set of fixture bundles, real and pathological, must render on every screen:
one model, forty models, an unconverged member, a refused comparison, a model
missing a quantity the others define, a bundle from an older schema. A mockup is
a proposal until it renders from a fixture.

## Convergence tiers

The viewer shows camdl's verdict and adds none of its own. Scope already carries
the verdict kinds (`converged`, `not_converged`, `unassessable`, `not_assessed`,
`not_applicable`, `chains_frozen`, `gate_pass`, `gate_fail`; see camdl-scope
`api/models.py`, `VerdictModel`).

| Verdict                      | Publishable        | In a comparison                      |
| ---------------------------- | ------------------ | ------------------------------------ |
| converged                    | yes                | yes                                  |
| not converged, chains frozen | yes, marked loudly | yes, marked loudly, listed in limits |
| unassessable, not assessed   | yes, marked        | decision needed (open question 2)    |
| failed, running              | no                 | no                                   |

The word shown for a passing fit is "Converged". It is not "validated":
convergence says the sampler agreed with itself, not that the model is right.

## Loading

There is no central repository and, at launch, no server. The viewer is static
files. Three shapes of URL:

1. One repository at a commit: the browser reads that repository's manifest.
2. An ad hoc comparison: several bundles named in the URL.
3. A published comparison: one file, in one repository, naming bundles anywhere.

Bundles are fetched by the reader's browser with plain GET requests. GitHub raw
file URLs allow this; GitHub release assets do not, so bundles live in the
repository, a Pages site, or a bucket the author controls.

A server can be added later without rework if two things hold: the viewer
obtains data only through one "load bundle from reference" interface, and all
computation is camdl code (R3).

## Mock debt

Elements in the 2026-11-13 mockups that have no source in camdl today. Each
becomes a requirement in the named proposal, or is removed.

| Element                                  | Needs                                                       | Proposal |
| ---------------------------------------- | ----------------------------------------------------------- | -------- |
| Calibration reading ("U-shaped")         | Author annotation, or a typed finding from camdl            | 3, 6     |
| Pooled calibration and coverage          | Per-stream PIT per step, pooled by camdl                    | 3        |
| Per-stream Δelpd                         | Exists as marginals; must be labelled marginal, no total    | 3        |
| "Prior-driven" flag                      | A contraction measure and rule defined in camdl             | 5        |
| Assumption counts per figure             | Ledger plus output-to-assumption dependencies               | 5        |
| "Shared by all models" list              | N-way diff                                                  | 4        |
| Structure difference sentence            | Typed diff report                                           | 4        |
| One-week-ahead versus free-running bands | Both named and typed in the predictive artifact             | 2        |
| Projection beyond scored horizon         | Scored horizon recorded; ideally scoring at longer horizons | 3        |
| Reproduction number by province          | A declared generated quantity indexed by the dimension      | 5, 6     |
| Headline sentence                        | Author annotation only; no generated claim                  | 6        |
| Data "as of" date                        | Recorded with the data digest                               | 2        |

## The set

| # | Proposal                                                                                        | Depends on             |
| - | ----------------------------------------------------------------------------------------------- | ---------------------- |
| 0 | Principles (this document)                                                                      | —                      |
| 1 | Bundle, manifest and `camdl publish`                                                            | 0                      |
| 2 | Score artifact: persisted trace, holdout identity, publish profile, data digests, calendar time | compare cleanup branch |
| 3 | Scoring core: typed report and refusals, WASM, pooled calibration, longer horizons              | 2                      |
| 4 | Model structure diff                                                                            | 0                      |
| 5 | Assumption ledger and limits                                                                    | 4                      |
| 6 | Presentation sidecars: annotations, geography                                                   | 1                      |
| 7 | Viewer: loader, URL scheme, view contracts, fixtures, export                                    | 1, 3, 4, 5, 6          |

Parked: export to forecast-hub formats, a registry, cloud fitting. Proposal 1
should keep a dense quantile set per step so hub export stays possible.

## Open questions

1. Name and subdomain. `models.camdl.dev` is assumed here. The URL scheme is the
   part that cannot change after launch.
2. Can a member whose convergence camdl could not assess be in a comparison?
3. Does the summary screen have to work at phone width? If so the per-model
   columns stack and the side-by-side scan is lost.
4. Should scoring at longer horizons (R8) be in the first release, or is marking
   the untested range enough?
5. Who may publish a comparison that includes someone else's bundle without
   their knowledge? Technically anyone; say whether the member's author is shown
   as a participant or only as a source.
