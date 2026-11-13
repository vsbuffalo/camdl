#!/usr/bin/env bash
# Render the link-preview card (Open Graph / Twitter) for camdl.dev.
#
# docs/assets/social-card.svg lays out the title, tagline and domain around
# docs/assets/social-card-diagram.svg, a card-only variant of the site's
# architecture figure. A preview card must be a raster image, so this renders
# them to the committed docs/assets/social-card.png, which docs/_quarto.yml
# names. Re-run after changing either SVG.
#
# Needs: rsvg-convert (librsvg).
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root/docs/assets"
rsvg-convert social-card.svg -o social-card.png
echo "wrote docs/assets/social-card.png"
