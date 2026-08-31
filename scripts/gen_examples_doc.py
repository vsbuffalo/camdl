#!/usr/bin/env python3
"""gen_examples_doc.py — regenerate `docs/examples.md`, the catalogue of every
model shipped in this repository.

The catalogue is DERIVED, never hand-maintained:

  * the prose summary of a model is the leading `#` comment block of its
    `.camdl` source (first paragraph, joined to one line);
  * its structure and feature flags come from the COMPILED `.ir.json` where one
    exists, so the catalogue cannot claim a feature the compiler did not emit;
  * the section layout comes from `SECTIONS` below — the one curated thing here,
    because "what is this directory FOR" is not derivable from the files.

Usage:
    scripts/gen_examples_doc.py            # rewrite docs/examples.md
    scripts/gen_examples_doc.py --check    # exit 3 if the committed doc is stale

`--check` compares whitespace-NORMALISED text (each blank-line-separated block
collapsed to a single spaced line), so a later `mdfmt`/`dprint fmt` pass — which
re-wraps prose and re-aligns table columns — never trips the gate. Only a real
content change does.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "examples.md"

# ── Curation ─────────────────────────────────────────────────────────────────
# (globs, heading, blurb). Order here is the order in the doc, and the first
# section to claim a file wins. A `.camdl` under the repo that matches no glob is
# reported as UNFILED rather than silently dropped — that is what keeps the
# catalogue exhaustive as the tree grows.
SECTIONS: list[tuple[list[str], str, str]] = [
    (
        ["ocaml/golden/*.camdl"],
        "Example models",
        "The primary example set. Every file here is automatically enrolled in the "
        "IR round-trip, Rust smoke, and cross-language simulate tests, so each one is "
        "known to compile, survive the OCaml↔Rust IR contract, and simulate on both "
        "stochastic backends.",
    ),
    (
        ["tests/external/cases/*/model.camdl"],
        "Literature replications and analytic references",
        "Models reproducing a published result or a closed-form answer. These are the "
        "external-validation surface: camdl's output is checked against the reference, "
        "not merely against itself.",
    ),
    (
        ["tests/external/ode_oracle/models/*.camdl"],
        "ODE oracles",
        "Deterministic models whose trajectories are checked against an independent "
        "ODE integrator (gh#166).",
    ),
    (
        ["tests/recovery/cases/*/model.camdl"],
        "Parameter-recovery cases",
        "Models used to fit synthetic data generated from known truth, to check that "
        "the inference stack recovers the parameters it was given.",
    ),
    (
        ["rust/crates/sim/tests/fixtures/*.camdl"],
        "Engine fixtures",
        "Models pinning specific simulation-engine behaviour: coupling semantics, seed "
        "timing, lineage tracking, and optimiser A/B gates.",
    ),
    (
        ["tests/fixtures/corner_cases/*.camdl"],
        "Corner cases",
        "Models pinning behaviour at awkward boundaries — off-grid observations and "
        "interventions, coincident lifecycle events, fractional end times.",
    ),
    (
        ["tests/fixtures/*.camdl", "tests/fixtures/*/*.camdl", "tests/fixtures/*/*/*.camdl"],
        "Feature and regression fixtures",
        "Models exercising one feature or reproducing one fixed bug.",
    ),
    (
        ["docs/dev/proposals/fixtures/*.camdl"],
        "Proposal fixtures",
        "Before/after models attached to a design proposal, showing what a language "
        "change buys.",
    ),
]

# Build output and VCS metadata: `ocaml/_build/` contains a dune-generated
# `META.camdl`, which is a package manifest, not a model.
IGNORED_DIRS = {".git", "_build", "target", "node_modules", "review-zips"}

# Directories of DELIBERATELY-INVALID sources (they exist to be rejected). Not
# models, so they are counted in the doc but never catalogued.
ERROR_DIRS = [
    ("ocaml/golden/errors", "dimension/type errors the compiler must reject"),
    ("ocaml/test/errors", "lex/parse/name-resolution errors"),
    ("ocaml/test/lints", "lint and diagnostic fixtures (clean + expected-warning pairs)"),
]

# Hand-picked entry points, with the reason each is worth opening first. Every
# name must resolve to a catalogued model or generation FAILS — so a rename or
# deletion is caught here rather than rotting in the doc.
SPOTLIGHT: list[tuple[str, str]] = [
    ("sir_basic", "the smallest complete model — start here"),
    ("seir_observations", "how observations attach to a model"),
    ("sir_priors", "declaring priors for inference"),
    ("sir_two_patch", "indexed parameters over a dimension"),
    ("seir_age", "stratification and a contact matrix"),
    ("seir_vaccine", "an intervention plus a scenario to switch it on"),
    ("seir_erlang_via", "non-exponential dwell times via `via`"),
    ("sirv_anchored_calendar", "calendar time: real dates, seasonal forcing, dated campaigns"),
    ("ross_macdonald", "multi-species host-vector transmission"),
    ("seir_spatial_5_inference", "a spatial model set up as an inference stress test"),
]


# ── Extraction ───────────────────────────────────────────────────────────────
SUMMARY_CAP = 160


def header_summary(src: Path) -> str:
    """First paragraph of the leading `#` comment block, capped to one table cell.

    Some headers open with several paragraphs of design notes; the catalogue
    wants the identifying sentence, and the file itself has the rest. So: take
    the first paragraph, stop at the first sentence end if there is one, and
    otherwise truncate on a word boundary.
    """
    lines: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line and not lines:
            continue  # tolerate blank lines before the block
        if not line.startswith("#"):
            break
        body = line.lstrip("#").strip()
        if not body:  # blank comment line ends the first paragraph
            break
        lines.append(body)
    text = " ".join(lines)

    sentence = re.match(r"(.+?[.?!])(?:\s|$)", text)
    if sentence and len(sentence.group(1)) <= SUMMARY_CAP:
        return sentence.group(1)
    if len(text) <= SUMMARY_CAP:
        return text
    return text[:SUMMARY_CAP].rsplit(" ", 1)[0] + " …"


def find_ir(src: Path) -> Path | None:
    """The compiled IR for `src`, if one is committed.

    Fixtures compile either beside the source (`foo.ir.json`) or into a sibling
    `ir/` directory; A/B gate fixtures compile to several variants of one source
    (`licm_ab.camdl` -> `licm_ab_on.ir.json`), in which case any variant answers
    the structural questions asked here.
    """
    stem = src.stem
    for cand in (src.with_suffix(".ir.json"), src.parent / "ir" / f"{stem}.ir.json"):
        if cand.exists():
            return cand
    for parent in (src.parent, src.parent / "ir"):
        variants = sorted(parent.glob(f"{stem}*.ir.json")) if parent.is_dir() else []
        if variants:
            return variants[0]
    return None


def compartments_from_source(src: Path) -> list[str]:
    """Fallback for a model with no committed IR: read `compartments { … }`."""
    m = re.search(r"compartments\s*\{([^}]*)\}", src.read_text(encoding="utf-8"))
    if not m:
        return []
    return [c.strip() for c in m.group(1).replace("\n", ",").split(",") if c.strip()]


def describe(src: Path) -> dict:
    """Everything the catalogue knows about one model."""
    text = src.read_text(encoding="utf-8")
    rel = src.relative_to(REPO).as_posix()
    info: dict = {
        "name": src.parent.name if src.stem == "model" else src.stem,
        "path": rel,
        "summary": header_summary(src),
        "flags": [],
    }

    ir_path = find_ir(src)
    if ir_path is None:
        info["base"] = compartments_from_source(src)
        info["dims"] = []
        info["from_ir"] = False
    else:
        model = json.loads(ir_path.read_text(encoding="utf-8"))["model"]
        struct = model.get("model_structure", {})
        info["from_ir"] = True
        info["base"] = struct.get("base_compartments") or [
            c["name"] for c in model.get("compartments", [])
        ]
        info["dims"] = [(d["name"], len(d["values"])) for d in struct.get("dimensions", [])]

        flags = info["flags"]
        all_dims = {name for name, _ in info["dims"]}
        per_compartment = struct.get("compartment_dims") or {}
        if all_dims and any(
            set(per_compartment.get(c, [])) != all_dims for c in info["base"]
        ):
            # `stratify(by = …, only = [...])` — the `×` in Structure is the model's
            # dimension list, not a promise that every compartment carries all of it.
            flags.append("partial-strat")
        if model.get("observations"):
            flags.append("obs")
        if model.get("interventions"):
            flags.append("intervention")
        if len(model.get("scenarios", [])) > 1:
            flags.append("scenarios")
        if model.get("tables"):
            flags.append("tables")
        if model.get("time_functions"):
            flags.append("forcing")
        if model.get("ode_equations"):
            flags.append("ode")
        if (model.get("simulation") or {}).get("dt") is not None:
            flags.append("dt")
        if any(
            (p.get("value") or {}).get("prior") not in (None, "flat")
            for p in model.get("parameters", [])
        ):
            flags.append("priors")

    if re.search(r"^\s*origin\s*=", text, re.MULTILINE):
        info["flags"].append("calendar")
    if "read(" in text:
        info["flags"].append("data")

    siblings = {p.name for p in src.parent.iterdir()}
    if f"{src.stem}.params.toml" in siblings or "params.toml" in siblings:
        info["flags"].append("params")
    if {"if2.toml", "fit.toml"} & siblings:
        info["flags"].append("fit-config")

    return info


def structure(info: dict) -> str:
    base = ",".join(info["base"]) if info["base"] else "—"
    for name, size in info["dims"]:
        base += f" × {name}[{size}]"
    return base


# ── Rendering ────────────────────────────────────────────────────────────────
def table(rows: list[dict]) -> list[str]:
    header = ["Model", "Structure", "Features", "Description"]
    body = [
        [
            f"[`{r['name']}`]({relative_link(r['path'])})",
            structure(r),
            ", ".join(r["flags"]) or "—",
            (r["summary"] or "_(no header comment)_").replace("|", "\\|"),
        ]
        for r in rows
    ]
    widths = [max(len(c[i]) for c in [header, *body]) for i in range(4)]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |"]
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in body]
    return out


def relative_link(path: str) -> str:
    """A repo-relative path, rewritten relative to `docs/`."""
    return "../" + path


def render(sections: list[tuple[str, str, str, list[dict]]], counts: dict) -> str:
    total = sum(len(rows) for _, _, _, rows in sections)
    error_total = sum(counts[d] for d, _ in ERROR_DIRS)

    L: list[str] = []
    L.append("# Example models")
    L.append("")
    L.append(
        f"Every model shipped in this repository — {total} `.camdl` files, plus "
        f"{error_total} deliberately-invalid sources used to pin compiler diagnostics. "
        "Together they are the worked-example corpus: whatever you are trying to "
        "express, something here probably expresses it already."
    )
    L.append("")
    L.append(
        "**This file is generated.** Run `make examples-doc` after adding or renaming a "
        "model; `scripts/gen_examples_doc.py --check` gates it. Edit the generator, not "
        "this file."
    )
    L.append("")
    L.append("## Reading the tables")
    L.append("")
    L.append(
        "**Structure** is the unstratified compartment list, followed by the dimensions "
        "it is expanded over and their sizes — so `S,E,I,R × age[2] × patch[5]` is a "
        "4-compartment model that expands to 40 states. **Features** are flags read out "
        "of the compiled IR:"
    )
    L.append("")
    for flag, meaning in [
        ("`obs`", "has an `observations` block (can be fitted to data)"),
        ("`intervention`", "declares at least one intervention"),
        ("`scenarios`", "declares scenarios beyond the baseline"),
        ("`tables`", "reads an indexed table (contact matrix, populations, …)"),
        ("`forcing`", "has a time-varying forcing function"),
        ("`ode`", "has explicit ODE equations (real-valued compartments)"),
        ("`dt`", "pins an explicit discretisation step"),
        ("`priors`", "declares a non-flat prior on at least one parameter"),
        ("`calendar`", "anchored to real dates via `origin`"),
        ("`data`", "reads an external file with `read(…)`"),
        ("`params`", "ships a `.params.toml`, so it runs without you supplying values"),
        ("`fit-config`", "ships a fit configuration (`fit.toml` / `if2.toml`)"),
    ]:
        L.append(f"- {flag} — {meaning}")
    L.append("")
    L.append("Descriptions are each model's own header comment.")
    L.append("")

    L.append("## Start here")
    L.append("")
    L.append(
        "If you are looking for a model to copy rather than a model to study, these ten "
        "cover most of the language between them:"
    )
    L.append("")
    by_name: dict[str, dict] = {}
    for _, _, _, rows in sections:
        for r in rows:
            by_name.setdefault(r["name"], r)  # earlier sections win a name collision
    for name, why in SPOTLIGHT:
        row = by_name[name]  # KeyError here = a spotlighted model was renamed/removed
        L.append(f"- [`{name}`]({relative_link(row['path'])}) — {why}")
    L.append("")

    for globs, heading, blurb, rows in sections:
        L.append(f"## {heading}")
        L.append("")
        L.append(f"{blurb} — {', '.join(f'`{g}`' for g in globs)} ({len(rows)} models)")
        L.append("")
        L += table(rows)
        L.append("")

    L.append("## Not models")
    L.append("")
    L.append(
        f"{error_total} further `.camdl` files exist only to be REJECTED — they pin the "
        "text and code of a compiler diagnostic, and none of them describe a disease. "
        "They are excluded from the tables above."
    )
    L.append("")
    for d, what in ERROR_DIRS:
        L.append(f"- `{d}/` ({counts[d]}) — {what}")
    L.append("")

    L.append("## Running one")
    L.append("")
    L.append("```bash")
    L.append("camdl check   ocaml/golden/sir_basic.camdl")
    L.append("camdl simulate ocaml/golden/sir_basic.camdl --params ocaml/golden/sir_basic.params.toml")
    L.append("```")
    L.append("")
    L.append(
        "A model without a `params` flag leaves its parameters estimated, so `simulate` "
        "needs values supplied with `--params` or `--set`."
    )
    L.append("")
    L.append(
        "Reading this through `camdl docs examples` with no checkout on disk? The "
        "sources are a sparse clone away (~5 MB), and the paths above are relative to "
        "its root:"
    )
    L.append("")
    L.append("```bash")
    L.append("git clone --depth 1 --filter=blob:none --sparse \\")
    L.append("    https://github.com/vsbuffalo/camdl .camdl-source")
    L.append('cd .camdl-source && git sparse-checkout set docs ocaml/golden && cd ..')
    L.append("```")
    L.append("")
    L.append(
        "You do not need this repository to build models of your own — see "
        "`camdl docs getting-started`."
    )
    L.append("")
    return "\n".join(L)


# ── Drive ────────────────────────────────────────────────────────────────────
def collect() -> tuple[list, dict]:
    error_dirs = [REPO / d for d, _ in ERROR_DIRS]
    claimed: set[Path] = set()
    sections = []
    for globs, heading, blurb in SECTIONS:
        rows = []
        for glob in globs:
            for src in sorted(REPO.glob(glob)):
                if src in claimed or any(d in src.parents for d in error_dirs):
                    continue
                claimed.add(src)
                rows.append(describe(src))
        rows.sort(key=lambda r: r["name"])
        sections.append((globs, heading, blurb, rows))

    unfiled = [
        p.relative_to(REPO).as_posix()
        for p in sorted(REPO.rglob("*.camdl"))
        if p not in claimed
        and not any(d in p.parents for d in error_dirs)
        and not (IGNORED_DIRS & set(p.parts))
    ]
    if unfiled:
        sys.exit(
            "error: these models match no section glob, so the catalogue would be "
            "incomplete. Add a section to SECTIONS in this script:\n  "
            + "\n  ".join(unfiled)
        )

    counts = {d: len(list((REPO / d).glob("*.camdl"))) for d, _ in ERROR_DIRS}
    return sections, counts


def normalise(text: str) -> str:
    """Collapse formatting-only differences: table padding and prose re-wrapping."""
    blocks = re.split(r"\n\s*\n", text)
    return "\n".join(" ".join(b.split()) for b in blocks if b.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if docs/examples.md is stale")
    args = ap.parse_args()

    sections, counts = collect()
    generated = render(sections, counts)

    if args.check:
        if not DOC.exists():
            print(f"error: {DOC.relative_to(REPO)} is missing; run `make examples-doc`")
            return 3
        if normalise(DOC.read_text(encoding="utf-8")) != normalise(generated):
            print(
                f"error: {DOC.relative_to(REPO)} is stale — the model tree has changed "
                "since it was generated. Run `make examples-doc` and commit the result."
            )
            return 3
        print(f"ok: {DOC.relative_to(REPO)} matches the model tree")
        return 0

    DOC.write_text(generated, encoding="utf-8")
    total = sum(len(rows) for _, _, _, rows in sections)
    print(f"wrote {DOC.relative_to(REPO)} — {total} models in {len(sections)} sections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
