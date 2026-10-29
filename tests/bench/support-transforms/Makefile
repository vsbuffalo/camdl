# IF2 default step in z, per map family (support-keyed transforms proposal
# §12.6, D8): the benchmark behind
# docs/dev/notes/2026-10-29-support-keyed-transforms-benchmark.md.
#
#   make binaries   pinned builds in bin/: br and ie at BR_SHA, main at MAIN_SHA,
#                   each in its own git worktree under work/
#   make grid       base grid, z-steps 2/3 .. 1/50, every fixture (+ main)
#   make grid-ext   1/100 and 1/200 on seir_age and ebola
#   make polio-r8   polio, 2/3 .. 1/200, seeds 1-2, 8 clean-evaluation replicates
#   make gh912-32   the gh912 short-budget cell, 1/20 and 1/50, seeds 1-32
#   make far        far starts (a box corner), 30 and 150 IF2 iterations
#   make results    results.tsv and summary.tsv from runs/ (committed extract)
#   make check-data re-simulate the committed fixture data and compare
#
# Each run set is resumable: a case whose runs/<id>/status.json exists is
# skipped. Run one set at a time; a run's wall and CPU time are recorded.
PY        := python3 -I
REPO      := $(abspath ../../..)
BR_SHA    := 877c517a
MAIN_SHA  := 0abe59f8
CARGO_REL  = cd $(1)/rust && cargo build --release -p cli -q
SETS      := grid grid-ext polio-r8 gh912-32 far

.PHONY: binaries results check-data $(SETS)

binaries: bin/camdl-br bin/camdlc-br bin/camdl-ie bin/camdl-main bin/camdlc-main

# A pinned worktree per build, created on first use.
WORKTREE = test -d work/$(1) || git -C $(REPO) worktree add --detach $(abspath work/$(1)) $(2)

bin/camdl-br bin/camdlc-br:
	$(call WORKTREE,br,$(BR_SHA))
	$(call CARGO_REL,work/br)
	$(MAKE) -C work/br build-ocaml
	mkdir -p bin && cp work/br/rust/target/release/camdl bin/camdl-br
	cp work/br/ocaml/_build/default/bin/camdlc.exe bin/camdlc-br

# Interval everywhere: the br tree with Transform::for_domain's LogInterval arm
# disabled, built and then restored.
bin/camdl-ie: bin/camdl-br
	sed -i.orig 's/(true, true) if positive_kind && lo > 0.0 =>/(true, true) if positive_kind \&\& lo > 0.0 \&\& false =>/' \
	  work/br/rust/crates/sim/src/inference/types.rs
	$(call CARGO_REL,work/br)
	cp work/br/rust/target/release/camdl bin/camdl-ie
	mv work/br/rust/crates/sim/src/inference/types.rs.orig work/br/rust/crates/sim/src/inference/types.rs

bin/camdl-main bin/camdlc-main:
	$(call WORKTREE,main,$(MAIN_SHA))
	$(call CARGO_REL,work/main)
	$(MAKE) -C work/main build-ocaml
	mkdir -p bin && cp work/main/rust/target/release/camdl bin/camdl-main
	cp work/main/ocaml/_build/default/bin/camdlc.exe bin/camdlc-main

$(SETS): binaries
	$(PY) bench.py run $@ >> runs-$@.log 2>&1

results:
	$(PY) table.py > /dev/null

# The data were simulated once at seed 1 (sir and seir_age by main at MAIN_SHA,
# ebola by br) and are committed; this re-simulates them and compares.
check-data: bin/camdl-main bin/camdl-br
	rm -rf work/data && mkdir -p work/data
	for f in sir seir_age; do \
	  CAMDL_SKIP_VERSION_CHECK=1 CAMDLC=$(abspath bin/camdlc-main) bin/camdl-main simulate \
	    $(REPO)/tests/recovery/cases/$$f/model.camdl --backend chain_binomial --seed 1 \
	    --params $(REPO)/tests/recovery/cases/$$f/truth.toml --obs-only-dir work/data/$$f; done
	CAMDL_SKIP_VERSION_CHECK=1 CAMDLC=$(abspath bin/camdlc-br) bin/camdl-br simulate \
	  fixtures/ebola/model.camdl --backend chain_binomial --seed 1 \
	  --params fixtures/ebola/truth.toml --obs-only-dir work/data/ebola
	for f in sir seir_age ebola; do diff -r work/data/$$f fixtures/$$f/data || exit 1; done
	@echo "fixture data reproduce"
