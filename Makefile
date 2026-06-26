SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := build

# ── Paths ─────────────────────────────────────────────────────────────────────

CAMDLC  := ocaml/_build/default/bin/camdlc.exe
CAMDL   := rust/target/release/camdl
INSTALL_DIR ?= $(HOME)/.local/bin

OCAML_GOLDENS := $(wildcard ocaml/golden/*.camdl)

# ── Build ─────────────────────────────────────────────────────────────────────

.PHONY: build build-ocaml build-rust build-benches

build: build-ocaml build-rust

# gh#audit-C8 follow-up. ir/VERSION is the canonical IR schema version
# (Rust reads it via include_str! at compile time). OCaml's dune project
# root is `ocaml/`, which puts ir/VERSION outside dune's source tree —
# so we generate a tiny .ml constant module from the file *before* dune
# runs, guaranteeing both languages bake the same value at build time.
# The generated file is .gitignore'd; bumping ir/VERSION + `make build`
# re-emits it.
OCAML_IR_VERSION_GEN := ocaml/lib/ir/ir_version_generated.ml

$(OCAML_IR_VERSION_GEN): ir/VERSION
	@printf '(* GENERATED from ir/VERSION by Makefile — do not edit. *)\nlet value = "%s"\n' \
	    "$$(tr -d '[:space:]' < ir/VERSION)" > $@

build-ocaml: $(OCAML_IR_VERSION_GEN)
	cd ocaml && dune build

build-rust:
	cd rust && cargo build --release --workspace --bins

# Compile-only build of every bench target so bench bit-rot fails fast (gh#222).
# Benches are NOT built by `cargo test` or `cargo build --bins`, and — verified
# empirically — `cargo clippy --all-targets` does NOT catch a bench that fails to
# type-check (a criterion bench supplies its own `main`, which clippy's check
# pass leaves unlinted). So a bench that has rotted against a signature change is
# dead code that still reads as live until someone runs this. No run needed; the
# benchmark fixtures and timings are irrelevant — we only need the compiler to
# prove the bench still matches the API it benchmarks.
build-benches:
	cd rust && cargo build --workspace --benches

# ── Install ───────────────────────────────────────────────────────────────────

.PHONY: install uninstall

# Git hash embedded in both binaries for version-skew detection.
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

install: build
	@mkdir -p $(INSTALL_DIR)
	@# camdlc: dune uses .exe on all platforms; install without the suffix.
	@# Also install as camdlc-<hash> so camdl can confirm an exact version
	@# match via a filesystem stat (no subprocess needed).
	install -m 755 $(CAMDLC) $(INSTALL_DIR)/camdlc
	install -m 755 $(CAMDLC) $(INSTALL_DIR)/camdlc-$(GIT_HASH)
	install -m 755 $(CAMDL)  $(INSTALL_DIR)/camdl
	@echo "Installed to $(INSTALL_DIR)  [camdlc-$(GIT_HASH)]"
	@echo "Make sure $(INSTALL_DIR) is on your PATH."
	@# Postflight: detect when another `camdl` (typically a leftover
	@# `cargo install --path crates/cli` in ~/.cargo/bin/) wins on PATH
	@# ahead of the binary we just wrote. Without this check the user
	@# only finds out at first invocation, and the runtime error tells
	@# them to "run make install" — which they just did. Catch it now.
	@expected=$(INSTALL_DIR)/camdl; \
	first=$$(command -v camdl 2>/dev/null || true); \
	if [ -n "$$first" ] && [ "$$first" != "$$expected" ]; then \
	  echo ""; \
	  echo "warning: another \`camdl\` is shadowing this install on your PATH."; \
	  echo "  Resolves first on PATH: $$first"; \
	  echo "  Just installed:         $$expected"; \
	  echo "  Fix: \`rm $$first\`, or put $(INSTALL_DIR) ahead of $${first%/*} on your PATH."; \
	fi

uninstall:
	rm -f $(INSTALL_DIR)/camdlc $(INSTALL_DIR)/camdl
	rm -f $(INSTALL_DIR)/camdlc-$(GIT_HASH)
	@echo "Removed from $(INSTALL_DIR)"

.PHONY: install-camdlc dev-camdlc

# Sync ONLY camdlc to the current HEAD (rebuilds OCaml, not Rust; does not
# install camdl). Use when the installed/`cargo run` camdl reports a camdlc
# version mismatch after a commit — realigns the installed camdlc with HEAD.
install-camdlc: build-ocaml
	@mkdir -p $(INSTALL_DIR)
	install -m 755 $(CAMDLC) $(INSTALL_DIR)/camdlc
	install -m 755 $(CAMDLC) $(INSTALL_DIR)/camdlc-$(GIT_HASH)
	@echo "Installed camdlc to $(INSTALL_DIR)  [camdlc-$(GIT_HASH)]"

# Branch dev loop: drop a hash-matched camdlc beside the cargo-built camdl so
# `cargo run -p cli -- ...` resolves it via the exact-match path (find_camdlc
# rule 1a in util.rs: a `camdlc-<hash>` in the running binary's directory IS the
# version check — no subprocess, no global install, no ~/.local/bin clobber).
# Re-run after each commit (HEAD's hash changes, so camdl looks for a new name).
dev-camdlc: build-ocaml
	@mkdir -p rust/target/debug rust/target/release
	install -m 755 $(CAMDLC) rust/target/debug/camdlc-$(GIT_HASH)
	install -m 755 $(CAMDLC) rust/target/release/camdlc-$(GIT_HASH)
	@echo "camdlc-$(GIT_HASH) placed beside the cargo binaries — \`cargo run -p cli\` exact-matches it (no install needed)."

# ── Test ──────────────────────────────────────────────────────────────────────

.PHONY: test test-ocaml test-rust test-inference test-integration test-docs test-cli-docs test-install

# `make test` runs the full surface. The Rust suite is split into two groups so
# CI can run and badge them independently (see .github/workflows/): test-rust =
# everything except the sim crate; test-inference = the sim crate (simulation
# engine + the inference stack). Their union is the whole workspace.
test: test-ocaml check-reactive-golden check-quantities-golden test-rust test-inference test-integration test-docs test-cli-docs test-install

# Inner-loop gate: the whole Rust workspace (unit + integration + doctests) via
# `cargo test`. Deliberately SKIPS the slow cross-language / doc phases
# (test-ocaml's dune runtest, check-reactive-golden, test-integration's OCaml→Rust
# shell suite, test-docs, test-cli-docs). Use it while iterating; it is NOT the
# authoritative gate. The full `make test` — and CI, which mirrors every phase
# (see .github/workflows/ + docs/dev/testing.md "Tiered gate") — must pass before
# a change lands. Anything test-fast skips is therefore still caught by CI.
# (cargo-nextest was tried as the runner but its parallel test-list enumeration
# spawns a burst of process launches that wedges macOS syspolicyd / code-signing,
# hanging every launch in dyld; `cargo test` runs binaries sequentially and is
# unaffected. sccache stays — it wraps rustc, no process burst.)
.PHONY: test-fast
test-fast: build-ocaml build-rust
	@mkdir -p $(CAMDLC_BIN)
	@ln -sf $(CAMDLC_ABS) $(CAMDLC_BIN)/camdlc
	cd rust && PATH="$(CAMDLC_BIN):$$PATH" CAMDL_SKIP_VERSION_CHECK=1 \
	  CAMDL="$(abspath $(CAMDL))" $(CARGO_WRAP) \
	  cargo test --no-fail-fast --workspace

# build-ocaml regenerates the gitignored ir_version_generated.ml from
# ir/VERSION; without this dep, `dune runtest` runs against a stale version
# constant after an ir/VERSION bump (emits the old version, mismatches the
# regenerated goldens) — the OCaml-side instance of the gh#178 staleness.
test-ocaml: build-ocaml
	cd ocaml && dune runtest

# Resolve the freshly-built camdlc by putting it FIRST on PATH (not via the
# CAMDLC env var): several tests (compile_once, ir_cache) inject their own
# camdlc shim through PATH, and a CAMDLC env (find_camdlc priority 2) would
# override them. PATH-prepend works WITH that design — a test that prepends
# its own shim still wins. This shadows any stale shared ~/.local/bin/camdlc,
# so camdl uses the compiler under test, never a stale PATH fallback (no
# divergence). Skip the version handshake too: it is an end-user safeguard
# against drifted *installed* binaries; with the fresh camdlc pinned on PATH
# it would only false-red on a cargo-cached stale camdl binary — which, having
# unchanged Rust, is schema-compatible by construction. See docs/dev/testing.md.
CAMDLC_BIN := $(abspath rust/target/_camdlc_bin)
# Optional compile cache (sccache): used only when it's on PATH; empty otherwise,
# which cargo treats as unset — so CI and contributors without it are unaffected.
# Shares artifacts across reruns AND across git worktrees (each has its own
# target/), which directly speeds the worktree-parallel workflow. `brew install
# sccache` to enable. (sccache wraps rustc only — no process-spawn burst, so it
# is safe on macOS, unlike cargo-nextest; see test-fast's note.)
SCCACHE := $(shell command -v sccache 2>/dev/null)
CARGO_WRAP := $(if $(SCCACHE),RUSTC_WRAPPER=$(SCCACHE),)
# build-rust is required: `cargo test` builds debug artifacts, but tests that
# spawn the binary (e.g. cli/tests/simulate_dt_knob.rs) use the *release*
# rust/target/release/camdl. Without this dep `make test` runs them against a
# stale release binary (false red/green — gh#178). build-rust is shared with
# test-integration's `build`, so make runs it once.
test-rust: build-ocaml build-rust
	@mkdir -p $(CAMDLC_BIN)
	@ln -sf $(CAMDLC_ABS) $(CAMDLC_BIN)/camdlc
	cd rust && PATH="$(CAMDLC_BIN):$$PATH" CAMDL_SKIP_VERSION_CHECK=1 \
	  CAMDL="$(abspath $(CAMDL))" $(CARGO_WRAP) \
	  cargo test --no-fail-fast --workspace --exclude sim

# The sim crate — simulation engine (Gillespie/tau-leap/ODE/chain-binomial) plus
# the inference stack (particle filter, IF2, PGAS, PMMH, NUTS, gradient checks)
# — is the heaviest, highest-stakes test group, so CI gives it its own workflow
# and badge. Same camdlc-on-PATH shim as test-rust (some sim tests compile .camdl
# fixtures via camdlc). `make test` runs this alongside test-rust.
test-inference: build-ocaml build-rust
	@mkdir -p $(CAMDLC_BIN)
	@ln -sf $(CAMDLC_ABS) $(CAMDLC_BIN)/camdlc
	cd rust && PATH="$(CAMDLC_BIN):$$PATH" CAMDL_SKIP_VERSION_CHECK=1 $(CARGO_WRAP) \
	  cargo test --no-fail-fast -p sim

test-integration: build
	CAMDLC="$(CAMDLC)" CAMDL="$(CAMDL)" bash tests/test_ocaml_to_rust.sh

# Compile every ```camdl block in the specs against the real compiler and gate
# on any complete-model block that fails (drift detector for documented
# examples). Fragments / legends / data-dependent blocks are auto-skipped by
# the compiler's verdict; see `camdlc doctest --help`. Catches code changes
# (grammar churn) that break documented models; a doc-only-PR gate needs a
# separate doc-triggered CI job since ci.yml ignores docs/** paths.
DOCTEST_DOCS := docs/camdl-language-spec.md docs/intro.md docs/user-features.md \
                docs/dsl-cheatsheet.md docs/dates.md docs/camdl-run-spec.md
test-docs: build-ocaml
	$(CAMDLC) doctest --gate $(DOCTEST_DOCS)

# Gate every documented `camdl …` invocation in CLI_DOCS against the real CLI
# parser (the binary's hidden `camdl __check-args` parse-only mode). Fails on
# DRIFT — a documented subcommand/flag/arg shape the binary does not expose —
# while tolerating EXPECTED input-layer failures (missing file, placeholder
# path), which the parse-only check never even reaches. The script's
# `--selftest` is the NON-VACUOUS guard: it asserts the gate catches synthetic
# drift AND does not over-flag valid-but-input-missing commands, so this target
# can never silently degrade into a no-op.
#
# CLI_DOCS is the set of docs verified drift-free / kept-clean. Extend it as
# other docs are brought to green.
CLI_DOCS := docs/workflow.md docs/inference.md docs/debugging.md docs/diagnosing-fits.md
test-cli-docs: build-rust
	bash scripts/check_cli_docs.sh --selftest
	bash scripts/check_cli_docs.sh $(CLI_DOCS)

# install.sh fast tier: shellcheck (if present) + offline unit tests
# (version_ge, the cmake>=3.13 gate, cmake_plat, and the no-sudo contract — a
# sudo shim aborts if any ensure_* reaches for root). Fast and offline, so it
# rides in the authoritative `make test`. The full end-to-end build is the
# container test, tests/install/Dockerfile.ubuntu1804 (amd64, run in CI/nightly).
test-install:
	@if command -v shellcheck >/dev/null 2>&1; then \
	  shellcheck install.sh tests/install_sh_test.sh; \
	else \
	  echo "shellcheck not found — skipping lint (CI installs it)"; \
	fi
	bash tests/install_sh_test.sh

# ── Golden file management ────────────────────────────────────────────────────

.PHONY: update-golden update-ocaml-golden update-corner-golden update-regression-golden update-reactive-golden update-quantities-golden check-quantities-golden

# Recompile all DSL fixtures → ocaml/golden/*.ir.json
update-ocaml-golden: build-ocaml
	@echo "Recompiling OCaml golden files..."
	@for src in $(OCAML_GOLDENS); do \
		out="$${src%.camdl}.ir.json"; \
		echo "  $$src → $$out"; \
		$(CAMDLC) "$$src" > "$$out"; \
	done

update-golden: update-ocaml-golden update-corner-golden update-regression-golden update-reactive-golden update-quantities-golden

# Recompile the corner-case fixtures (params baked via --set) →
# tests/fixtures/corner_cases/ir/*.ir.json. These pin the off-grid /
# coincident / fractional / lifecycle FORWARD trajectories in
# gate_corner_case_baseline.rs. Re-run after a schema bump, then re-capture
# the gate (CAMDL_CAPTURE_BASELINE=1).
CORNER_DIR := tests/fixtures/corner_cases
update-corner-golden: build-ocaml
	@echo "Recompiling corner-case fixtures..."
	@$(CAMDLC) $(CORNER_DIR)/off_grid_intervention.camdl       --set beta=1.0 --set gamma=0.2 --set cull=0.5               -o $(CORNER_DIR)/ir/off_grid_intervention.ir.json
	@$(CAMDLC) $(CORNER_DIR)/coincident_obs_intervention.camdl --set beta=1.0 --set gamma=0.2 --set cull=0.5               -o $(CORNER_DIR)/ir/coincident_obs_intervention.ir.json
	@$(CAMDLC) $(CORNER_DIR)/fractional_output_end.camdl       --set beta=1.0 --set gamma=0.2                              -o $(CORNER_DIR)/ir/fractional_output_end.ir.json
	@$(CAMDLC) $(CORNER_DIR)/off_grid_obs.camdl                --set beta=1.0 --set gamma=0.2                              -o $(CORNER_DIR)/ir/off_grid_obs.ir.json
	@$(CAMDLC) $(CORNER_DIR)/all_lifecycle.camdl               --set beta=1.0 --set gamma=0.2 --set cull=0.5 --set N0=1000 -o $(CORNER_DIR)/ir/all_lifecycle.ir.json
	@$(CAMDLC) $(CORNER_DIR)/seasonal_drift.camdl              --set beta=0.6 --set gamma=0.4 --set alpha=0.4 -o $(CORNER_DIR)/ir/seasonal_drift.ir.json
	@$(CAMDLC) $(CORNER_DIR)/event_intervention_agree.camdl    --set k=0.0 --set keep=0.5 -o $(CORNER_DIR)/ir/event_intervention_agree.ir.json
	@$(CAMDLC) $(CORNER_DIR)/gh70_absorbing_importation.camdl  --set k=0.0 -o $(CORNER_DIR)/ir/gh70_absorbing_importation.ir.json
	@$(CAMDLC) $(CORNER_DIR)/multi_effect_same_time.camdl      --set k=0.0 -o $(CORNER_DIR)/ir/multi_effect_same_time.ir.json
	@$(CAMDLC) $(CORNER_DIR)/event_drain_fusion.camdl          --set k=0.3 --set f=0.2 -o $(CORNER_DIR)/ir/event_drain_fusion.ir.json
	@$(CAMDLC) $(CORNER_DIR)/dt_rate.camdl                     --set beta=1.0 --set gamma=0.2 --set tau=1.0 -o $(CORNER_DIR)/ir/dt_rate.ir.json

# Recompile the regression fixtures (params baked via --set) →
# tests/fixtures/regression/ir/*.ir.json. These reproduce specific fixed bugs;
# unlike the corner-case corpus they are NOT auto-discovered by a baseline gate
# (a sim error must not break baseline capture), only loaded by their own
# regression tests. Re-run after a schema bump.
REGRESSION_DIR := tests/fixtures/regression
update-regression-golden: build-ocaml
	@echo "Recompiling regression fixtures..."
	@$(CAMDLC) $(REGRESSION_DIR)/gh208_sparse_negative_rate.camdl --set beta=2.0 --set gamma=0.2 --set omega=1.0 --set cap=9 -o $(REGRESSION_DIR)/ir/gh208_sparse_negative_rate.ir.json

# Recompile the reactive (gh#204) compiler/IR goldens →
# tests/fixtures/reactive/ir/*.ir.json. Compile-only: the runtime rejects an
# active reactive policy, so these pin the IR SHAPE (FireSource::Reactive,
# TriggerExpr, stratified expansion), deserialised cross-language by
# rust/crates/ir reactive_golden tests. No --set (params stay estimated; the IR
# emits without values). Re-run after a schema bump.
REACTIVE_DIR := tests/fixtures/reactive
update-reactive-golden: build-ocaml
	@echo "Recompiling reactive fixtures..."
	@mkdir -p $(REACTIVE_DIR)/ir
	@$(CAMDLC) $(REACTIVE_DIR)/reactive_sir_observed_threshold.camdl -o $(REACTIVE_DIR)/ir/reactive_sir_observed_threshold.ir.json
	@$(CAMDLC) $(REACTIVE_DIR)/reactive_indexed_patch_sia.camdl      -o $(REACTIVE_DIR)/ir/reactive_indexed_patch_sia.ir.json

# Drift gate (gh#204): each reactive .camdl must still compile BYTE-FOR-BYTE to
# its committed .ir.json. `update-reactive-golden` regenerates; this FAILS if
# source and golden have diverged (a grammar/expander change that moved the IR
# without re-running update). Runs in `make test`.
.PHONY: check-reactive-golden
check-reactive-golden: build-ocaml
	@echo "Checking reactive goldens match their .camdl..."
	@fail=0; for src in reactive_sir_observed_threshold reactive_indexed_patch_sia; do \
	  $(CAMDLC) $(REACTIVE_DIR)/$$src.camdl -o $(REACTIVE_DIR)/ir/$$src.ir.json.tmp; \
	  if ! diff -q $(REACTIVE_DIR)/ir/$$src.ir.json $(REACTIVE_DIR)/ir/$$src.ir.json.tmp >/dev/null 2>&1; then \
	    echo "  DRIFT: $$src.camdl no longer compiles to the committed .ir.json — run: make update-reactive-golden"; \
	    fail=1; \
	  fi; \
	  rm -f $(REACTIVE_DIR)/ir/$$src.ir.json.tmp; \
	done; \
	if [ $$fail -ne 0 ]; then exit 1; fi; \
	echo "  reactive goldens in sync"

# Recompile the generated-quantities showcase fixture →
# tests/fixtures/quantities/ir/*.ir.json. Compile-only: pins the IR SHAPE of
# every quantity variant (state/observation source, value/time reductions,
# integral, Derived, stratified expansion), deserialised cross-language by the
# `rust/crates/ir quantities_golden` test. The quantity OUTPUT values are pinned
# separately by `quantities_surface.rs`. Re-run after a schema/grammar change.
QUANTITIES_DIR := tests/fixtures/quantities
update-quantities-golden: build-ocaml
	@echo "Recompiling quantities fixtures..."
	@mkdir -p $(QUANTITIES_DIR)/ir
	@$(CAMDLC) $(QUANTITIES_DIR)/quantities_showcase.camdl -o $(QUANTITIES_DIR)/ir/quantities_showcase.ir.json

# Drift gate: the showcase .camdl must still compile BYTE-FOR-BYTE to its
# committed .ir.json. `update-quantities-golden` regenerates; this FAILS if
# source and golden diverged. Runs in `make test`.
.PHONY: check-quantities-golden
check-quantities-golden: build-ocaml
	@echo "Checking quantities goldens match their .camdl..."
	@fail=0; for src in quantities_showcase; do \
	  $(CAMDLC) $(QUANTITIES_DIR)/$$src.camdl -o $(QUANTITIES_DIR)/ir/$$src.ir.json.tmp; \
	  if ! diff -q $(QUANTITIES_DIR)/ir/$$src.ir.json $(QUANTITIES_DIR)/ir/$$src.ir.json.tmp >/dev/null 2>&1; then \
	    echo "  DRIFT: $$src.camdl no longer compiles to the committed .ir.json — run: make update-quantities-golden"; \
	    fail=1; \
	  fi; \
	  rm -f $(QUANTITIES_DIR)/ir/$$src.ir.json.tmp; \
	done; \
	if [ $$fail -ne 0 ]; then exit 1; fi; \
	echo "  quantities goldens in sync"

# ── Release / changelog ───────────────────────────────────────────────────────

.PHONY: changelog version-bump release-suggest release-prep release-publish

# Cutting a release — the short path (full runbook: RELEASING.md, policy:
# VERSIONING.md). Three steps:
#   make release-suggest                 # what changed since last tag + suggested bump
#   make release-prep VERSION=0.2.0      # bump manifests + regenerate changelog (review it)
#   ... draft RELEASE_NOTES-0.2.0.md (/release-notes skill), run `make test` ...
#   make release-publish VERSION=0.2.0   # commit + tag + push + gh release (irreversible)
release-suggest:
	@scripts/release.sh suggest
release-prep:
	@test -n "$(VERSION)" || { echo "usage: make release-prep VERSION=0.2.0"; exit 1; }
	@scripts/release.sh prep "$(VERSION)"
release-publish:
	@test -n "$(VERSION)" || { echo "usage: make release-publish VERSION=0.2.0"; exit 1; }
	@scripts/release.sh publish "$(VERSION)"

# Deterministic changelog spine from Conventional Commits (last tag -> HEAD).
# git-cliff is the renderer: `brew install git-cliff` or `cargo install git-cliff`.
# The /release-notes skill turns this spine into user-facing notes. See
# VERSIONING.md and cliff.toml.
changelog:
	@command -v git-cliff >/dev/null || { \
	  echo "git-cliff not found — install it: brew install git-cliff (or cargo install git-cliff)"; \
	  exit 1; }
	git-cliff -o CHANGELOG.md
	@echo "wrote CHANGELOG.md (embedded into \`camdl docs changelog\`; regenerate before a release/build)"

# Print the SemVer version git-cliff recommends from the unreleased commits.
version-bump:
	@command -v git-cliff >/dev/null || { echo "install git-cliff first"; exit 1; }
	@git-cliff --bumped-version

# ── Quick simulation helpers ──────────────────────────────────────────────────

.PHONY: sim

# Usage: make sim MODEL=ir/golden/sir_basic.ir.json ARGS="--set beta=0.3 ..."
sim: build-rust
	$(CAMDL) simulate $(MODEL) $(ARGS)

# ── Benchmarks & profiling (FOI scaling study) ────────────────────────────────
#
# See docs/dev/notes/2026-05-29-foi-scaling-bench.md. The toy model generator
# is scripts/gen_scaling_models.py; macro sweep scripts/bench_scaling.py.

.PHONY: bench-scaling bench-compile bench-micro bench-micro-fixtures flamegraph-real flamegraph-bench profile-pmmh

CAMDLC_ABS := $(abspath $(CAMDLC))
GEN        := scripts/gen_scaling_models.py
FX         := rust/crates/sim/benches/fixtures/scaling
PROFILE_CAMDL := rust/target/profiling/camdl

# Real model timed alongside the synthetic ladder by bench-compile. Default is
# the main-checkout-relative path to the playpen Kano SEIRV model; from a git
# worktree (deeper tree) pass an absolute KANO_MODEL=... instead. The harness
# skips it cleanly with a message if the path is absent.
KANO_MODEL ?= ../playpen-camdl-measles/projects/nga/getting-started-simple/model/kano_lga_seirv.camdl

# (P,A,coupling) grid for the micro-bench fixtures — matches GRID in scaling.rs.
MICRO_GRID := 4/1/on 8/1/on 16/1/on 32/1/on 4/1/off 8/1/off 16/1/off 32/1/off \
              8/7/on 16/7/on 32/7/on 8/7/off 16/7/off 32/7/off

# Macro sweep: full compile→simulate pipeline across scales → TSV + plot.
bench-scaling: build
	CAMDLC="$(CAMDLC_ABS)" python3 scripts/bench_scaling.py
	uv run --with matplotlib --with numpy scripts/plot_scaling.py

# Compiler-only sweep: time camdlc.exe alone (parse→expand→dimcheck→autodiff→
# serialize), no Rust runtime — the compile-side analogue of bench-scaling.
# Writes docs/dev/notes/assets/compile/compile_baseline.tsv + curves. Override
# OUT=... to record a labelled variant (e.g. _flambda) for before/after plots.
OUT ?= docs/dev/notes/assets/compile/compile_baseline.tsv
bench-compile: build-ocaml
	CAMDLC="$(CAMDLC_ABS)" python3 scripts/bench_compile.py --out "$(OUT)" --real "$(KANO_MODEL)"
	uv run --with matplotlib --with numpy scripts/plot_compile.py

# Generate the (gitignored) IR fixtures the micro-bench loads.
bench-micro-fixtures: build
	@mkdir -p $(FX)
	@for spec in $(MICRO_GRID); do \
	  P=$${spec%%/*}; rest=$${spec#*/}; A=$${rest%%/*}; C=$${rest##*/}; \
	  out=$(FX)/P$${P}_A$${A}_$${C}_minimal.ir.json; \
	  python3 $(GEN) -P $$P -A $$A --coupling $$C --grad minimal -o /tmp/_micro.camdl 2>/dev/null; \
	  CAMDL_SKIP_VERSION_CHECK=1 CAMDLC="$(CAMDLC_ABS)" $(CAMDL) compile /tmp/_micro.camdl --no-dim-check -o $$out >/dev/null; \
	done
	@echo "fixtures → $(FX)"

# Per-step eval / load micro-benchmarks (criterion): the `scaling` bench.
bench-micro: bench-micro-fixtures
	cd rust && cargo bench -p sim --bench scaling

# Flamegraph the real-model regime: generate the anchor (P=44,A=21,coupling=on,
# grad=full ≈ the Kano model), then profile `simulate`. Produces a static SVG
# (macOS `sample` → inferno; no sudo) that serves cleanly over HTTP, plus a
# samply profile for interactive exploration. Point at a different IR (e.g. the
# real Kano model) to profile that instead.
# Prereqs: `cargo install inferno samply`.
FG_SVG := docs/dev/notes/assets/scaling/flamegraph_real.svg
flamegraph-real: build-ocaml
	cd rust && cargo build --profile profiling -p cli --bin camdl
	python3 $(GEN) -P 44 -A 21 --coupling on --grad full -o /tmp/fg_anchor.camdl
	CAMDL_SKIP_VERSION_CHECK=1 CAMDLC="$(CAMDLC_ABS)" $(PROFILE_CAMDL) \
	  compile /tmp/fg_anchor.camdl --no-dim-check -o /tmp/fg_anchor.ir.json
	@echo "sampling simulate (~12s)..."
	@TMPDIR=/tmp CAMDL_SKIP_VERSION_CHECK=1 CAMDLC="$(CAMDLC_ABS)" $(PROFILE_CAMDL) \
	   simulate /tmp/fg_anchor.ir.json --backend chain_binomial --scenario baseline \
	   -o /tmp/fg_traj.tsv & \
	 PID=$$!; sample $$PID 12 -file /tmp/camdl_sample.txt >/dev/null 2>&1; wait $$PID
	inferno-collapse-sample /tmp/camdl_sample.txt | \
	  inferno-flamegraph --title "camdl simulate anchor (P=44,A=21,on,full)" > $(FG_SVG)
	@echo "wrote $(FG_SVG)  (also: samply record -- $(PROFILE_CAMDL) simulate … for interactive)"

# Flamegraph the per-step hot path via the scaling bench binary.
flamegraph-bench: bench-micro-fixtures
	cd rust && cargo build --profile profiling -p sim --bench scaling
	@echo "run: samply record -- rust/target/profiling/deps/scaling-* --bench --profile-time 10 eval_propensities"

# Flamegraph PMMH inference steps. `--observe` makes the generated spatial model
# fittable (a weekly_cases stream over prevalence(I)); we synthesize data, then
# sample a PMMH run → static inferno SVG. PMMH is particle-filter-based (uses
# `rate` only, no rate_grad path); `--grad full` only supplies free FOI params
# for PMMH to estimate. Memory-safe at moderate P — small IR, PF state is just
# N_particles × compartments — so unlike `flamegraph-real` (P=44,A=21 full grad,
# the ~15 GB OOM anchor) this stays small. Tune: PMMH_P/PMMH_A/PMMH_STEPS/PMMH_PARTICLES.
PMMH_P ?= 16
PMMH_A ?= 7
PMMH_STEPS ?= 100
PMMH_PARTICLES ?= 200
FG_PMMH_SVG := docs/dev/notes/assets/scaling/flamegraph_pmmh.svg
profile-pmmh: build-ocaml
	cd rust && cargo build --profile profiling -p cli --bin camdl
	python3 $(GEN) -P $(PMMH_P) -A $(PMMH_A) --coupling on --grad full --observe -o /tmp/pmmh_anchor.camdl
	CAMDL_SKIP_VERSION_CHECK=1 CAMDLC="$(CAMDLC_ABS)" $(PROFILE_CAMDL) \
	  compile /tmp/pmmh_anchor.camdl --no-dim-check -o /tmp/pmmh_anchor.ir.json
	CAMDL_SKIP_VERSION_CHECK=1 $(PROFILE_CAMDL) simulate /tmp/pmmh_anchor.ir.json \
	  --backend chain_binomial --dt 1 --seed 42 --scenario baseline --obs-dir /tmp/pmmh_obs >/dev/null
	@echo "sampling PMMH (~15s); P=$(PMMH_P) A=$(PMMH_A) particles=$(PMMH_PARTICLES) steps=$(PMMH_STEPS)..."
	@TMPDIR=/tmp CAMDL_OUTPUT_DIR=/tmp/pmmh_prof_out CAMDL_SKIP_VERSION_CHECK=1 $(PROFILE_CAMDL) \
	   profile /tmp/pmmh_anchor.ir.json --scenario baseline \
	   --data /tmp/pmmh_obs/weekly_cases.tsv --obs weekly_cases --flow infection \
	   --sweep 'R0=lin(14,16,2)' --particles $(PMMH_PARTICLES) \
	   --algorithm pmmh --pmmh-steps $(PMMH_STEPS) --pmmh-particles $(PMMH_PARTICLES) --pmmh-rho 0.99 \
	   --starts 1 --rw-sd auto --fixed sigma=0.125 --fixed kappa=0.05 --fixed amplitude=0.25 --fixed iota=1e-7 \
	   --output /tmp/pmmh_prof.tsv --seed 1 >/tmp/pmmh_prof.log 2>&1 & \
	 PID=$$!; sample $$PID 15 -file /tmp/pmmh_sample.txt >/dev/null 2>&1; wait $$PID
	inferno-collapse-sample /tmp/pmmh_sample.txt | \
	  inferno-flamegraph --title "camdl PMMH step (P=$(PMMH_P),A=$(PMMH_A),coupling on)" > $(FG_PMMH_SVG)
	@echo "wrote $(FG_PMMH_SVG)"

# ── Tree-sitter / Neovim ──────────────────────────────────────────────────────

TS_DIR      := tree-sitter
NVIM_PARSER := $(HOME)/.local/share/nvim/lazy/nvim-treesitter/parser/camdl.so
NVIM_QUERIES := $(HOME)/.config/nvim/after/queries/camdl

.PHONY: install-nvim-ts

# Compile the camdl tree-sitter parser and install it into Neovim.
# Requires: a C compiler on PATH.
install-nvim-ts:
	@echo "Compiling tree-sitter parser..."
	cc -shared -fPIC -o $(TS_DIR)/camdl.so -I $(TS_DIR)/src $(TS_DIR)/src/parser.c
	install -m 644 $(TS_DIR)/camdl.so $(NVIM_PARSER)
	@echo "Installing queries..."
	@mkdir -p $(NVIM_QUERIES)
	install -m 644 $(TS_DIR)/queries/highlights.scm $(NVIM_QUERIES)/highlights.scm
	install -m 644 $(TS_DIR)/queries/locals.scm     $(NVIM_QUERIES)/locals.scm
	@echo "Done. Restart Neovim and open a .camdl file."

# ── Housekeeping ──────────────────────────────────────────────────────────────

.PHONY: clean

clean:
	cd ocaml && dune clean
	cd rust && cargo clean
