# camdl style

House style for a modelling project: what belongs in a `.camdl` header, and
where the project's files live. Nothing here is enforced by the compiler.

## Model files

A `.camdl` file is read cold, by someone who did not write it, trying to answer
three questions fast: **what is this, where did it come from, and what does it
change.** The header answers those three and then gets out of the way.

### The header

```camdl
# National SEIR with a facility-death delay: cases and deaths from one
# confirmation flow, deaths lagged through an isolation compartment.
#
# Base:    bvd_national_twocfr.camdl
# Adds:    nothing.
# Changes: f_cfr_unret becomes free with a beta(2,2) prior instead of being
#          derived from f_cfr, so the data can violate the care ordering.
#
# Fitted to: confirmed cases (daily), facility deaths (daily), community
# deaths (daily), lab confirmations (daily, a proportion over specimens).
#
# Why: with f_cfr_unret derived, f_cfr did not move (0.419 vs 0.429) and the
# care-death stream sat at -3.49 nats against a -2.95 ceiling. If the ordering
# constraint is what binds f_cfr, freeing it should move f_cfr_unret above
# f_cfr. Full argument: notes/2026-08-22-cfr-ordering.md
```

**One line first, saying what the model is.** Not what it branches from — what
it _is_, in words a reader who has never seen the project can follow. This is
the line that is most often missing, and it is the one a reader needs first.

**Then `Base:` / `Adds:` / `Changes:`.** One change per variant wherever
possible, so the contrast is readable. `Base: none — root of the <name> line`
for a root model. `Adds: nothing.` and `Changes: nothing.` are both good answers
and should be written rather than omitted.

**Then what it is fitted to**, one clause per stream saying what the stream
measures. If a stream is doing something subtle — a proportion over a subset, a
stock rather than a flow, a second pipeline onto the same latent quantity — that
is the place to say so.

**Then `Why:`** — the observation that motivates the change, and the
pre-registered read that would confirm or refute it. Both are scientific
content; do not compress them to hit a line count. Keep proportion, though: a
header is not a paper, and a `Why:` that runs to paragraphs is an argument, and
an argument belongs in a dated note that the header cites by filename.

Past forty lines, something in the header belongs somewhere else.

### History versus motivation

The test: **does it explain the model in front of me, or does it narrate how it
got here?**

| Keep — explains the model in front of you                                                                                        | Cut — narrates how it got here                       |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| "the base hit 37% divergences at these settings; this reparameterises `beta` as `r_eff * (tau + gamma)` in response"             | "this used to be `beta`-primary, changed 2026-08-12" |
| "the observed deaths-per-case ratio rises 0.08 → 0.52 then plateaus, which a constant multiple cannot produce — hence the delay" | "superseded by `bvd_national_twocfr.camdl`"          |

A motivating observation — the divergences, the failed check, the shape in the
data that the variant responds to — is context a reader needs to evaluate the
model, so it stays, even though it came out of an earlier fit. A changelog is
git's job. The header copy is a second record of the same thing, and it is the
one that goes stale, because nothing forces anyone to update it.

Running commentary on what earlier arms produced belongs in a dated note for a
sharper reason than tidiness: results go stale the next time you fit, while a
pre-registered read does not. One observation stated because it explains this
file's shape is not commentary; a rolling log of every arm's numbers is.

### What else does not belong

**Project conventions.** A naming scheme, a directory layout, a description of
how the family branches — these are true of every file in the family, so a copy
in each file is a copy that can drift, and a reader who has opened three models
has read it three times. Write it down once, in the project README or a note,
and let each header be about its own model.

### Parameter docstrings are the exception

`#'` docstrings carry the **justification** for a prior — the citation, the
derivation, the caution about what the parameter does and does not mean. That is
documentation, not history, and it should stay:

```camdl
#' Case fatality among those reaching facility care. Beta(4,5) has mean 0.44
#' and 90% [0.19, 0.71], covering the Bundibugyo-specific anchors: Roddy 0.42
#' (26 hospitalised confirmed, 2007), Kratz 0.50 (9/18, 2012 Isiro).
#' @ref Van Kerkhove et al. 2015, Sci Data 2:150019, Table 5.
f_cfr : probability in [0.0, 1.0] ~ beta(alpha = 4.0, beta = 5.0)
```

Keep every citation load-bearing and verified. A decorative reference in a model
file is worse than none, because it looks like evidence.

Neither kind of comment re-keys a model — a `#` comment never reaches the IR,
and `#'` docs are carried in the IR envelope, outside the hashed `model` object
— so a wrong citation can be corrected without orphaning any fit.

### The header moves to `#'` when gh#750 lands

A `#'` must attach to a declaration today, so a model-level one is a syntax
error and the header above is written in `#` comments, which no tool can read.
gh#750 adds a model-level doc slot; the header's shape does not change, only the
sigil, and `camdl render`, the viewer and `camdl fit summary` will then be able
to say what a model is without anyone opening the file. Write `#` headers until
it lands.

### Body comments

Comment at the seams, not line by line. What earns a comment:

- **a coefficient that is not obvious** — where `3.386` came from, why the
  hazard is split this way
- **a modelling choice a reader might mistake for a typo** — `I = 0.5 * I0`
  splitting a seed across two classes at a constant rather than at `p_hard`
- **anything approximate**, said plainly, so nobody reads it as exact

What does not:

- restating the code (`# recovery from I to R` above `recover : I --> R`)
- section banners in a file the reader can already see the shape of

### Checklist

- [ ] First line says what the model **is**, in plain words
- [ ] `Base:` / `Adds:` / `Changes:` present, one change if possible
- [ ] Every observation stream named, with what it measures
- [ ] `Why:` gives the motivating observation and the pre-registered read, or
      cites the note that argues it
- [ ] No project conventions, no changelog, no running results
- [ ] Every `#'` citation checked against the source

## Project layout

The directories the tools expect — `models/`, `data/`, `results/`, `scripts/`,
`workflow/`, `tests/`, `notes/`, `Makefile` — are tabulated in
[`agents.md`](agents.md), "The layout the tools expect" (`camdl docs agents`),
with the argument for keeping every run under one `results/` root. What that
table does not say:

**A `fit.toml` sits beside its `.camdl` in `models/`, not in a `configs/`.**
They are one artifact pair: the model alone is not runnable, because the config
is what names the data, the bounds and priors, and the stages. Split them across
two trees and every variant, rename and copy has to be done twice in two places,
and the paths inside the config — which resolve relative to the config's own
directory — grow a `../` for no gain.

**`notes/` is dated (`<YYYY-MM-DD>-<slug>.md`)** because a note records what was
true on a day: what a fit showed, which read held, a dead end worth not
repeating. That is exactly the material a header sheds — the argument too long
for `Why:`, and the commentary on what earlier arms produced. A file named for
its topic invites editing to keep it current, and so can go stale; a file named
for its date cannot.

**Nothing derived is ever written into a `results/` leaf.** A leaf belongs to
camdl: its `run.json` manifests exactly the files camdl put there, and an
unlisted file makes the leaf fail that check — the next run reports it stale
(`OrphanFiles`) rather than taking the cache hit, and recomputes. A hand-written
summary inside a leaf therefore costs you the cache hit and does not survive.
Derived artifacts go beside the store, keyed by run id.
