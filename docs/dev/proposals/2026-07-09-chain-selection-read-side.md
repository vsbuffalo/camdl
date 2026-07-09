# Read-side chain selection (`--exclude-chains` / `--chains`)

Date: 2026-07-09 Status: Draft (gated on maintainer review) Tags: inference,
cli, posterior, diagnostics, cas

Depends on: the per-chain diagnostics (gh#406) — you must be able to _see_ which
chains are outliers before excluding them. That is the companion change and
lands first; this proposal is the escape hatch built on top of it.

## Problem

There is no way to select or exclude MCMC chains on the read side. `fit predict`
takes `--stage/--scenario/--sweep/--seed/--n-draws` but nothing chain-level;
`fit summary`, `compare`, `fit table` have none. The gate pass/fails the whole
fit; it never lets you drop the offending chains.

Concrete case (camdl-garki): a genuinely non-identified parameter — `g`
(transmission ceiling; the profile shows a flat likelihood above ~0.3) — leaves
a flat ridge, and a minority of chains wander into non-representative side
modes. A 6-chain mh-ODE fit: 2 of 6 chains stuck → R̂(q3)=1.5, ESS=NaN. Excluding
those 2 → R̂(q3)=1.05, `q3` cleanly identified. This is being done by hand in
polars today — **invisibly** — so the sealed θ̂ and the watcher's
predictive/compare are computed from the polluted 6-chain cloud with no record
that anyone intervened. The feature's value is largely in making that
already-happening manual practice _auditable_.

## Classification and the load-bearing caveat

This is a new read-side CLI feature (not a bug fix). It is a **loaded gun**:
post-hoc chain exclusion can bias the posterior (cherry-picking modes). R̂ > 1.1
with chains in different modes is the model telling you something true — the
sampler has not converged, or the posterior is genuinely multimodal /
unidentified. The _primary_ remedy is to ask whether a parameter is unidentified
and fix it (which garki is doing for `g`). So this feature is the explicit,
warned escape hatch for the case where you understand _why_ a few chains strayed
(a known ridge) and want the dominant mode — never a default cleanup. Every
design decision below exists to keep it from becoming "delete chains until R̂
looks good."

## The one seam

`crate::posterior_draws::resolve_posterior_draws` (`posterior_draws.rs:49`) is
the draws-cloud authority — the single function every read-side consumer already
routes through: `compare` (`compare.rs:353`), `fit summary` (`main.rs:992`),
`fit predict` (`predict.rs`), `contrasts`, `fit_table`, `joint`. The combined
`draws.tsv` carries a `chain` column as its first field (verified on a garki
fit: `chain\tdraw\ta2\tg\t…`).

**Design: chain filtering lives here, once.** `resolve_posterior_draws` gains an
optional chain selector; it drops rows whose `chain` is excluded before
returning the cloud. Every consumer inherits `--exclude-chains` uniformly — the
no-silent-gaps property is structural, not per-command discipline. This is the
"reach for the existing seam" move: one filter at the authority, not four
parallel ones.

## Flag surface

**One flag: `--exclude-chains 3,5`** — a comma-separated list of 1-based chain
ids to drop (matching the `chain_N/` dirs and the per-chain summary table),
parsed once at the boundary into a typed `ChainSelection`. No `--chains`
keep-form in v1: camdl fits are 4–8 chains and the operation is "drop the one or
two I diagnosed as stuck", so the drop verb is the natural one (and the literal
"ignore" framing), and a second flag with a mutual-exclusion rule is surface we
do not need yet. It is a trivial additive follow-up if a many-chain keep-most
case appears.

Exposed on the **single-fit** read-side commands: `fit predict`, `fit summary`,
`fit table`. `compare` is **out of v1** — `compare @a @b --exclude-chains 3,5`
is ambiguous (fit A's chain 3 is unrelated to fit B's), so it needs per-fit
syntax (`@a:3,5`) that is a separate, larger design; deferred to a named
follow-up rather than shipped ambiguous. `contrasts` / `joint` likewise consume
the cloud through the seam and inherit the _capability_, but the flag is not
surfaced on them in v1 (no demonstrated need); the seam makes adding it later
free.

Errors are hard, never silent: a chain id not in the fit →
`error: chain 7 not in this fit (chains 1..6)`; excluding every chain →
`error: --exclude-chains leaves an empty posterior`.

## Identity and provenance (verified against code)

The initial instinct was "the exclusion set feeds `run_id` → a new identity
level." Reading the code, that is **not** the mechanism:

- **`fit predict`** writes `predictive/<stream>.tsv` into the fit's own segment
  (`write_tsv(segment, …)`, `predict.rs:525`); it does not mint a new CAS
  `run_id`. `--n-draws` already subsets the cloud and is likewise not folded
  into any hash (verified: no `n_draws` in `resolve.rs` / `fit/cas.rs` hashing).
  So `--exclude-chains` follows the existing `--n-draws` pattern: it changes
  what is written into the segment, an overwrite, with **no cache collision** to
  guard.
- **`compare`** seals θ̂ as the posterior MEAN over the draws
  (`resolve_posterior_draws`) and derives the prequential by invoking the
  canonical `camdl pfilter` at that θ̂ (`compare.rs:184`, `246`). A chain-subset
  θ̂ is a _different parameter vector_, so the derived `pfilter` run re-keys
  **naturally** through the existing params level of `run_id` — no special
  handling.
- **`fit summary`** is display-only; `--exclude-chains` is a view.

So there is **no new `run_id` level**. What is mandatory instead is
**provenance**: every artifact or sealed θ̂ produced under a chain selection must
record the selection where it is written —

1. `predictive.json` (and the `predictive/*.tsv` are already regenerable views)
   gains a `chain_selection` field alongside the existing `n_draws` /`seed`.
2. `compare`'s derived-θ̂ provenance records it, so a chain-subset comparison is
   never mistakable for a full-cloud one.
3. `fit summary` prints the active selection in its header when one is set.

A chain-subset seal must be **auditable**, never silently indistinguishable from
a full-cloud seal. That is the identity requirement here, and it is satisfied by
stamping, not by re-keying.

## Guardrail (load-bearing, not polish)

- **Loud warning** whenever a selector is active: one line naming the excluded
  chains and stating that post-hoc exclusion can bias the posterior.
- **Identifiability nudge**: when the per-chain outlier signal is strong (gh#406
  already computes it), `fit summary` prints "chains N,M disagree — is a
  parameter unidentified? see `camdl docs …`" _before_ pointing at the flag. The
  flag is the second thing the user reads, not the first.
- The warning is not quietable — the failure mode (a cherry-picked posterior
  read as a normal one) is silent, so the warning must not be.

## Decision: no auto-quarantine

The reported ask included an optional "gate mode that flags chains failing a
per-chain criterion and drops them from the sealed θ̂." **Rejected.**
Automatically dropping chains is the one part that turns a warned escape hatch
into a silent-bias generator — it automates the exact cherry-picking the caveat
warns against, and "drops with a warning" is how a warning becomes noise an
agent suppresses and a non-specialist skims. The gate's job is to _flag and
name_ (gh#406); the human makes the explicit, recorded exclusion. Recorded here
so a future reader sees it was considered and rejected, not overlooked.

## Phasing

1. **gh#406 (companion, lands first)** — per-chain diagnostics that name the
   outlier chains in `fit summary` + the gate. Prerequisite: you cannot
   responsibly exclude what you cannot see.
2. **This proposal** — `ChainSelection` parsed at the boundary; filter in
   `resolve_posterior_draws`; expose the flag on the read-side commands;
   provenance stamping; the warning + nudge. Red→green: a fit with a known
   outlier chain → `fit predict --exclude-chains <it>` produces a predictive
   whose bands differ from the full-cloud run AND whose `predictive.json`
   records the selection; a bad chain id hard-errors; `--chains` and
   `--exclude-chains` together is a parse error. Full `make test`, goldens
   byte-identical (no golden touches this — it is read-side).

## Decisions (no open questions)

1. Filtering lives once in `resolve_posterior_draws`; consumers inherit it.
   Rationale: the no-silent-gaps property becomes structural.
2. **One flag, `--exclude-chains`** (drop-set), parsed into a typed
   `ChainSelection` at the boundary. No `--chains` keep-form in v1 (smaller
   surface; the drop verb fits the "ignore the stuck ones" workflow). A
   nonexistent chain id, or excluding all chains, is a hard error.
3. No new `run_id` level. Identity is preserved by provenance stamping
   (`predictive.json` `chain_selection` field, summary header) — verified that
   predict overwrites the segment (like `--n-draws`, which is in no hash) and
   compare would re-key via the changed θ̂ params.
4. v1 surfaces the flag on the single-fit commands only: `fit predict`,
   `fit summary`, `fit table`. `compare` is deferred (needs per-fit `@a:3,5`
   syntax — ambiguous otherwise); `contrasts` / `joint` inherit the seam
   capability but the flag is not surfaced in v1 (no demonstrated need).
5. A loud, non-quietable warning + an identifiability nudge accompany every
   active selection. No auto-quarantine.
