# Quantities as a separable, shareable reporting layer

Status: proposed Issue: gh#618 (identity question, answered here), downstream
ask in `ebola-bdbv-camdl/agent-channel.md` Scope: CLI + OCaml surface. No IR
change. No identity change.

## Problem

Twelve model files in one downstream project carry near-identical 50-line
`quantities {}` blocks, and the copies have drifted: `bvd_national_dwell.camdl`
computes `reff` and `generation_interval` with exponential-dwell formulas that
are wrong under its Erlang staging, and reports P(R > 1) = 0.20 for an epidemic
that grew tenfold. A copy-pasted reporting block did not follow its model.

Two further things are true and change the design:

1. **Quantities are already outside model identity.** `Model::hash_into` skips
   `quantities` and `contrasts`, pinned by `ir_quantities_excluded_from_hash`
   ("the one Model field outside the run-id walk"). Editing a reporting formula
   does NOT re-key a sim or a fit. The downstream belief that it does — and the
   `scripts/staged_quantities.py` workaround built on that belief — is mistaken,
   and the workaround is unnecessary.
2. **But `fit predict` reads the ARCHIVED model** (`predict.rs:928`,
   `segment.join("model.ir.json")`), so correcting a quantity in the source has
   no effect on an existing fit. It does not orphan the fit; it simply does
   nothing. That, not identity, is the real obstacle.

So the ask is not "get quantities out of identity" (done) but "let a corrected
reporting vocabulary be applied to a fit that already exists."

## Design

**A quantities file is an ordinary `.camdl` file containing only a
`quantities {}` block**, supplied at the point of use:

```
camdl fit predict <fit> --quantities reporting/national.camdl
camdl simulate model.camdl --quantities reporting/national.camdl ...
```

Deliberately NOT an import statement in the model. camdl has no import
mechanism, and adding one brings scoping, cycles, and search-path questions for
a construct that is evaluated post-hoc from stored trajectories and needs none
of it. The file is resolved and type-checked against the model it is applied to,
at the point it is applied. A model's own `quantities {}` block remains the
default when no override is given; `--quantities` REPLACES it wholesale (not
merges — a merge rule is a silent-precedence surface, and replacement is what
"swap the vocabulary" means).

**Missing symbols are a hard error** naming the symbol and the file. A shared
vocabulary will reference `f_cfr`, which exists only in the delay family, or
`beta`, a parameter in one model and a `let` in another. The alternative — an
optional quantity that silently disappears — is how a reporting table loses a
column and nobody notices. Separate vocabularies per family is the right answer,
and the error tells the author which family they are in. This matches the
"declared branches over silent defaults" rule.

## Identity

Fit identity: **unchanged**, and must stay unchanged. The likelihood does not
read quantities.

Artifact identity: **the quantities file's content hash keys any artifact whose
content it determines** — the predict `quantities/*.tsv` and `quantities.json`,
and `simulate --quantities-out`. Two vocabularies applied to one fit produce two
different tables; without the key they collide at one content address, which is
the exact class fixed twice this week (gh#626 `--to`, gh#641 `--init-state`).
Key the file's BYTES, not its path, so an in-place edit re-keys and two copies
of one file share identity.

Provenance: the resolved quantities file path + hash is recorded in the
artifact's run record, so a table can always be traced to the vocabulary that
produced it.

## What this does not do

- It does not let a model NAME a shared vocabulary (`use quantities "..."`).
  That is an import mechanism; if it is wanted later it layers on top of this
  without redesign.
- It does not merge model-declared and file-supplied quantities.
- It does not change `contrasts`, which share the identity exclusion but have
  their own evaluation path.
- It does not resolve `value_at(…, last_obs)` under a data-free `simulate`
  (ebola F23) — unchanged, and still refused there.

## Testing

- A quantities file applied to a fit produces a table; the model's own block is
  not consulted.
- A missing symbol is a hard error naming the symbol AND the file (both, or the
  author cannot tell which vocabulary is wrong).
- Two different vocabularies on one fit produce two artifacts with DISTINCT
  content addresses; the same vocabulary twice collides correctly.
  Mutation-check by dropping the key and confirming the collision test reddens.
- Fit identity is byte-identical with and without `--quantities` (the fit's
  `run_id` must not move).
- An in-place edit of the quantities file re-keys the artifact.
