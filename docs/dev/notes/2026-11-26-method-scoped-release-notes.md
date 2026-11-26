# Release notes draft: method-scoped settings and the Simulator resolver

Date: 2026-11-26

Status: draft for the next release's notes. The `release-notes` skill folds it
into `RELEASE_NOTES-<version>.md` when the release is cut, then deletes it.

Covers method-scoped settings Series A and C
(`docs/dev/proposals/2026-10-29-method-scoped-settings.md`, ending at
`6d4247982`) and the Simulator resolver
(`docs/dev/proposals/2026-11-24-simulator-resolver.md`, ending at `51568c468`).

## Breaking changes

**Step sizes moved from `[estimate]` to `[method]`, and are written in z.** Here
z is the unconstrained coordinate each method moves in. The old `rw_sd` was a
natural-scale step: it set IF2's perturbation, and one fifth of PMMH's and MH's
initial proposal. Each step is now an entry in `[method.perturbation]` (IF2) or
`[method.proposal]` (PMMH, MH). The old `perturb_only_at_t0 = true` is now
`initial_state = true` inside the IF2 entry.

A config that uses the old keys is refused. For each parameter, the message
prints two things: the z value that gives your old step at the parameter's
start, and the natural step the default would give. Deleting a line is not
neutral, because it switches that parameter to the default step. `pgas` and the
`nl-*` optimizers never read `rw_sd`, so delete it for those methods.

**IF2 checks where each estimated parameter enters the model.**

- It refuses a parameter that enters only the initial state unless that
  parameter is marked `initial_state = true`.
- It refuses the mark on a parameter that enters the dynamics or an observation.
- IF2 and the `nl-*` optimizers refuse an estimated parameter that the
  likelihood does not depend on.
- PMMH, MH and PGAS refuse a parameter with neither a likelihood connection nor
  a proper prior. They warn about one that has only a proper prior.
  `prior_only = "refuse"` turns that warning into a refusal (`e86b59d32`).

**`fit_state.toml` carries `schema = "camdl.fit-state/1"`.** An untagged file is
refused as an input, and can still be read for inspection.

**`profile --rw-sd` is still natural-scale.** It is converted to z at the
parameter's base value. Without it, an IF2 profile reads the `--fit` file's
`[method.perturbation]`. Every other command takes steps in z, so no release tag
may carry both units: Series D replaces `--rw-sd` with `--perturbation NAME=SD`
in z.

## Run identity

- Every fit re-keys at this release.
- `[estimate].start` no longer changes a fit's problem hash. Fits that differ
  only in their starts or steps share one problem folder.
- `pfilter --fit` and `profile --fit` now use the fit's binomial sampler, which
  re-keys them (`86c5842f9`).
- Stored fits made before this release are still listed and addressable. Their
  old keys are set aside with a notice.

## CLI

- `fit run --dt` sets the integration step under the dt rule (run spec §1.3.1).
- `design … --fit` and `simulate --fit` with an anchored `--to`, or with an
  anchored model, refuse a fit file whose written `[config] dt` disagrees with
  the model's `simulate { dt }`. This is the same refusal `fit run`, `pfilter`
  and `profile` give (`1b9ff9ce2`).
- One window moves. Take a model that declares no step, beside a fit file that
  writes one. Its observed window is now read at the file's step, the step the
  fit reads it at, where it was read at 1.0.

## Fixed

- An explicit `binomial = "btpe"` is accepted under IF2, PMMH and pfilter. It
  was refused as an unknown key (`7c9eb63fd`).
- IF2's per-iteration `effective_sd` diagnostic, and profile's `cooling_final`,
  report the cooling at the end of each iteration. They were one iteration
  behind (`5136a916b`). `cooling_final` is the last iteration's cooling factor
  (`e52ee2ec5`).
