#!/usr/bin/env Rscript
# Generate a reference TSV of the periodic B-spline forcing computed via
# pomp::periodic_bspline_basis.
#
# Second independent oracle for camdl's evaluator (complements the scipy
# fixture). camdl matches pomp's basis-indexing convention directly, so
# no coefficient roll is needed here.
#
# This script is NOT run by CI — the TSV it produces is committed to the
# repo so the cross-validation test stays offline-safe. Regenerate only
# if pomp's algorithm changes (it has been stable for over a decade) or
# you intentionally change camdl's convention.
#
# Run: make fixtures-bspline

if (!requireNamespace("pomp", quietly = TRUE)) {
  stop("install pomp first: install.packages('pomp')")
}

period  <- 4.0
n_basis <- 6
degree  <- 3
n_pts   <- 200
coefs   <- c(0.7, 1.2, 0.9, 0.5, 1.1, 0.8)

ts <- seq(0, period, length.out = n_pts + 1)[-(n_pts + 1)]
# basis is n_pts x n_basis
basis <- pomp::periodic_bspline_basis(ts,
                                       nbasis = n_basis,
                                       degree = degree,
                                       period = period)
ys <- as.vector(basis %*% coefs)

out_path <- "rust/crates/sim/tests/fixtures/periodic_bspline_pomp.tsv"
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

con <- file(out_path, "w")
writeLines(c(
  "# gh#59 v2 oracle: periodic B-spline via pomp::periodic_bspline_basis",
  "# fixture parameters:",
  sprintf("#   period=%g  n_basis=%d  degree=%d", period, n_basis, degree),
  sprintf("#   coefs=[%s]", paste(coefs, collapse = ", ")),
  "# camdl uses pomp's basis-indexing convention (centering shift); no",
  "# coef roll is needed when comparing this fixture to camdl output.",
  "# Regenerate with: Rscript scripts/gen_periodic_bspline_pomp_fixture.R",
  "t\ty"
), con)
write.table(data.frame(t = ts, y = ys),
            con,
            sep = "\t", quote = FALSE,
            row.names = FALSE, col.names = FALSE)
close(con)
cat(sprintf("wrote %d rows to %s\n", n_pts, out_path))
