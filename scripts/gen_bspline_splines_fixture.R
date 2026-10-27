#!/usr/bin/env Rscript
# Reference clamped B-spline basis values from R's splines::splineDesign, a
# second oracle for rust/crates/sim/src/bspline.rs independent of scipy
# (proposal docs/dev/proposals/2026-10-27-estimable-bsplines.md §8.1, R1).
#
# Same layouts and points as scripts/gen_bspline_scipy_fixtures.py, written to
# bspline_basis_splines.tsv. Outside the window the value is splineDesign's at
# t clamped to [a, b] (camdl holds a spline at its end value).
#
# Run: make fixtures-bspline

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: gen_bspline_splines_fixture.R <out-dir>")
out_path <- file.path(args[[1]], "bspline_basis_splines.tsv")

max_degree <- 5
uniform <- function(a, b, n) { h <- (b - a) / n; a + (0:n) * h }
layouts <- list(
  uniform = uniform(0, 168, 9),
  uniform_offset = uniform(1000, 1168, 9),
  irregular = c(0, 10, 25, 30, 60, 90, 100, 130, 150, 168)
)

points <- function(xs) {
  a <- xs[1]; b <- xs[length(xs)]
  lo <- xs[-length(xs)]; hi <- xs[-1]
  inner <- c(lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo), lo + 0.75 * (hi - lo))
  sort(unique(c(xs, a - 5, b + 7, inner)))
}

fmt <- function(x) sprintf("%.17g", x)
rows <- character(0)
for (name in names(layouts)) {
  xs <- layouts[[name]]
  a <- xs[1]; b <- xs[length(xs)]
  for (p in 0:max_degree) {
    u <- c(rep(a, p), xs, rep(b, p))
    ts <- points(xs)
    tc <- pmin(pmax(ts, a), b)
    basis <- splines::splineDesign(knots = u, x = tc, ord = p + 1, outer.ok = FALSE)
    stopifnot(all(abs(rowSums(basis) - 1) < 1e-13))
    for (i in seq_along(ts)) {
      rows <- c(rows, paste(name, p, fmt(ts[i]), seq_len(ncol(basis)) - 1, fmt(basis[i, ]), sep = "\t"))
    }
  }
}

header <- c(
  "# Clamped B-spline basis values B_k(t) from R splines::splineDesign",
  sprintf("# (R %s, splines %s). Written by scripts/gen_bspline_splines_fixture.R;",
          paste(R.version$major, R.version$minor, sep = "."), packageVersion("splines")),
  "# regenerate with `make fixtures-bspline`. An oracle, not an input.",
  "# Outside the window the value is splineDesign's at t clamped to [a, b].",
  "# Layouts as in bspline_basis_scipy.tsv.",
  "layout\tdegree\tt\tk\tvalue"
)
writeLines(c(header, rows), out_path)
cat(sprintf("wrote %d rows to %s\n", length(rows), out_path))
