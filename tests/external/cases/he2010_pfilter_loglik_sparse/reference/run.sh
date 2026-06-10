#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

if [[ "${CAMDL_EXTERNAL_USE_DOCKER:-}" == "1" ]]; then
    echo "docker regen not yet wired for this case; run locally" >&2
    exit 1
fi

command -v Rscript >/dev/null || {
    echo "Rscript not found on PATH. Install R (e.g. 'brew install r') or rerun with CAMDL_EXTERNAL_USE_DOCKER=1." >&2
    exit 1
}

if [[ ! -d "renv/library" ]]; then
    echo "installing R dependencies via renv (first-run bootstrap)…" >&2
    Rscript -e 'if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", repos = "https://cloud.r-project.org"); renv::restore(prompt = FALSE)'
fi

# reference.R scores the HOLED series: it reads the NA pattern from the
# committed camdl-side data file ../../data/he2010_london_cases_holed.tsv
# ../data/he2010_london_cases_holed.tsv (relative to reference/) — the
# same file camdl scores — and overlays it on the upstream measles
# time axis. No separate holed-series derivation lives in reference/.
Rscript reference.R

[[ -s "out/pomp_pfilter_loglik.tsv" ]] || {
    echo "reference.R did not produce out/pomp_pfilter_loglik.tsv" >&2
    exit 1
}

echo "reference regen complete: $(wc -l < out/pomp_pfilter_loglik.tsv) rows → out/pomp_pfilter_loglik.tsv"
