# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data parameter-recovery harness — sibling to tests/external/.
#
# tests/external/ compares camdl against EXTERNAL references (pomp, Stan, …).
# This is self-contained: plant θ (truth.toml), simulate the observation
# streams, fit, and recover. Manual today (run a target, eyeball θ̂ vs truth);
# the planned successor is a `cargo test` harness that asserts recovery within
# Monte-Carlo tolerance. It also doubles as the place to *watch the progress
# bars* across every fitter.
#
# Targets (run from anywhere; `-f tests/recovery/Makefile` is enough):
#
#   make -f tests/recovery/Makefile synth      # simulate at truth.toml → obs streams
#   make -f tests/recovery/Makefile if2        # IF2 fit     — per-chain ll bars
#   make -f tests/recovery/Makefile pgas       # PGAS fit    — per-chain ll bars
#   make -f tests/recovery/Makefile pmmh       # PMMH fit    — per-chain ll + acc bars
#   make -f tests/recovery/Makefile ensemble   # 30-cell simulate — the cells bar
#   make -f tests/recovery/Makefile survey     # likelihood survey — best-ll bar
#   make -f tests/recovery/Makefile all        # synth + the three fits
#   make -f tests/recovery/Makefile clean      # remove generated data + results
#
# Append `--progress plain` to any target for the CI/plain bar form, or
# `--no-progress` to silence. Uses `cargo run` (dev build) so it works from a
# fresh checkout with no install step. CASE selects the case dir
# (default cases/seir_age). Generated data/ + results/ are gitignored — the
# recipe (model + truth + configs) is the committed artifact.
# ─────────────────────────────────────────────────────────────────────────────

# Anchor to this Makefile's own location so targets work from any CWD.
THIS := $(abspath $(lastword $(MAKEFILE_LIST)))
HERE := $(patsubst %/,%,$(dir $(THIS)))
REPO := $(abspath $(HERE)/../..)
CASE ?= $(HERE)/cases/seir_age

# `cargo run` builds a dev `camdl` whose embedded git-hash need not match the
# dune-built camdlc; the version handshake guards mismatched *release* artifacts,
# not two binaries built from the same source tree, so we skip it here.
export CAMDLC_PATH              := $(REPO)/ocaml/_build/default/bin/camdlc.exe
export CAMDL_SKIP_VERSION_CHECK := 1

CAMDL := cargo run --quiet --manifest-path $(REPO)/rust/Cargo.toml -p cli --
MODEL := model.camdl              # paths below resolve inside $(CASE)
TRUTH := --params truth.toml      # planted ground truth (single source)
# Stream-name-agnostic synth sentinel: a case may emit cases_child/adult.tsv,
# afp.tsv, … — gate the fits on a stamp, not on a specific stream filename.
SYNTH := $(CASE)/data/.synth-stamp

.PHONY: help ocaml synth if2 pgas pmmh ensemble survey all clean
.DEFAULT_GOAL := help

help: ## show this help
	@grep -hE '^[a-z][a-zA-Z0-9-]*:.*## ' $(THIS) | sed 's/:.*## /\t/' | \
	  awk -F'\t' '{printf "  make -f tests/recovery/Makefile %-9s  %s\n", $$1, $$2}'

# Build camdlc once (the IR cache then reuses the compiled model across targets).
ocaml:
	@cd $(REPO)/ocaml && dune build 2>/dev/null

# Synthetic data: simulate at the truth + sample every observation stream the
# model declares (one TSV per stream under data/). The stamp makes the recipe
# stream-name-agnostic so cases with different obs blocks share this harness.
$(SYNTH): $(CASE)/$(MODEL) $(CASE)/truth.toml | ocaml
	cd $(CASE) && $(CAMDL) simulate $(MODEL) --backend chain_binomial --seed 1 $(TRUTH) \
	  --obs-only-dir data --output-dir results
	@touch $(SYNTH)
synth: $(SYNTH) ## simulate at truth.toml → obs streams

if2:  $(SYNTH) ; cd $(CASE) && $(CAMDL) fit run if2.toml  --allow-nonconverged-scout ## IF2 fit
pgas: $(SYNTH) ; cd $(CASE) && $(CAMDL) fit run pgas.toml --allow-nonconverged-scout ## PGAS fit
pmmh: $(SYNTH) ; cd $(CASE) && $(CAMDL) fit run pmmh.toml --allow-nonconverged-scout ## PMMH fit

ensemble: | ocaml ## 30-replicate forward sim (cells bar)
	cd $(CASE) && $(CAMDL) simulate $(MODEL) --backend chain_binomial $(TRUTH) \
	  --replicates 30 --output-dir results

survey: $(SYNTH) ## likelihood survey over if2.toml bounds (best-ll bar)
	cd $(CASE) && $(CAMDL) survey $(MODEL) --fit if2.toml --n-points 400 --output results

all: if2 pgas pmmh ## synth + all three fits

clean: ## remove generated data/ + results/
	rm -rf $(CASE)/data $(CASE)/results
