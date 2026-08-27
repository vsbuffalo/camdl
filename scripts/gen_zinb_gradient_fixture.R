#!/usr/bin/env Rscript
# Generate the external oracle for camdl's zero-inflated negative-binomial
# (ZINB) observation likelihood and its gradient.
#
# The ZINB mixes a point mass at zero over an NB2 base:
#
#   P(y = 0) = pi + (1 - pi) * f_NB(0 | mu, k)
#   P(y = k) =      (1 - pi) * f_NB(k | mu, k)      for k > 0
#
# where mu is the mean, k the NB2 dispersion (variance = mu + mu^2/k), and pi
# the structural-zero probability.
#
# Why this is an oracle and not a second camdl implementation:
#
#   * the NB2 density comes from base R's `dnbinom(x, size = k, mu = mu)` --
#     an independent, long-established implementation, not a restatement of
#     camdl's `negbin_logpmf`;
#   * the gradient comes from `numDeriv::grad`, which differentiates that
#     density numerically by Richardson extrapolation. It has no knowledge of
#     the closed form camdl evaluates, so agreement is evidence about the
#     derivation and not merely about arithmetic.
#
# The one line this script and camdl would otherwise share is the mixture
# itself. That line is therefore checked here against two properties derived
# independently of the pmf algebra, and the script aborts rather than writing
# a fixture if either fails:
#
#   * normalization: sum_y P(y) = 1 over a truncation carrying > 1 - 1e-12;
#   * moments: E[Y] = (1 - pi) * mu  and
#              Var[Y] = (1 - pi) * mu * (1 + mu/k + pi*mu).
#
# The fixture is committed so the Rust test stays offline and CI never needs R.
# Regenerate only when deliberately re-pinning.
#
# Run: Rscript scripts/gen_zinb_gradient_fixture.R

if (!requireNamespace("numDeriv", quietly = TRUE)) {
  stop("install numDeriv first: install.packages('numDeriv')")
}
suppressPackageStartupMessages(library(numDeriv))

pkg_version <- as.character(utils::packageVersion("numDeriv"))
r_version <- paste(R.version$major, R.version$minor, sep = ".")

repo_root <- tryCatch(
  system2("git", c("rev-parse", "--show-toplevel"), stdout = TRUE),
  error = function(e) stop("run inside the camdl git repository")
)
out_path <- file.path(repo_root, "rust", "crates", "sim", "tests",
                      "fixtures", "zinb_gradient_ref.tsv")

# ── the density under test, in R ───────────────────────────────────────────
# `dnbinom` supplies f_NB; only the mixture is written here, and it is the
# part the structural checks below exist to police.
zinb_logpmf <- function(y, mu, k, pi) {
  stopifnot(mu > 0, k > 0, pi >= 0, pi <= 1)
  log_f <- dnbinom(y, size = k, mu = mu, log = TRUE)
  if (y == 0) {
    # log(pi + (1 - pi) * f0), computed by log-sum-exp so a tiny f0 does not
    # lose the pi term and pi = 0 does not produce log(0) * 0.
    a <- if (pi > 0) log(pi) else -Inf
    b <- if (pi < 1) log1p(-pi) + log_f else -Inf
    m <- max(a, b)
    if (is.infinite(m)) return(m)
    m + log(exp(a - m) + exp(b - m))
  } else {
    if (pi >= 1) return(-Inf)
    log1p(-pi) + log_f
  }
}

# ── structural checks on the mixture ───────────────────────────────────────
check_normalization <- function(mu, k, pi, tol = 1e-10) {
  # Truncate at the NB quantile carrying all but 1e-15 of the mass, and
  # account for the remaining tail exactly with `pnbinom` rather than assuming
  # it is zero. A small k (heavy overdispersion) gives a near-geometric tail
  # with ratio mu/(k+mu), which a fixed sd-based cutoff badly under-covers.
  ymax <- max(200, qnbinom(1 - 1e-15, size = k, mu = mu))
  ys <- 0:ymax
  p <- vapply(ys, function(y) exp(zinb_logpmf(y, mu, k, pi)), numeric(1))
  tail_mass <- (1 - pi) * pnbinom(ymax, size = k, mu = mu, lower.tail = FALSE)
  total <- sum(p) + tail_mass
  if (abs(total - 1) > tol) {
    stop(sprintf("normalization failed at mu=%g k=%g pi=%g: sum = %.17g",
                 mu, k, pi, total))
  }
  list(mean = sum(ys * p), var = sum(ys^2 * p) - sum(ys * p)^2,
       tail_mass = tail_mass)
}

check_moments <- function(mu, k, pi, tol = 1e-6) {
  m <- check_normalization(mu, k, pi)
  want_mean <- (1 - pi) * mu
  want_var <- (1 - pi) * mu * (1 + mu / k + pi * mu)
  if (abs(m$mean - want_mean) > tol * max(1, abs(want_mean))) {
    stop(sprintf("mean identity failed at mu=%g k=%g pi=%g: %.17g vs %.17g",
                 mu, k, pi, m$mean, want_mean))
  }
  if (abs(m$var - want_var) > tol * max(1, abs(want_var))) {
    stop(sprintf("variance identity failed at mu=%g k=%g pi=%g: %.17g vs %.17g",
                 mu, k, pi, m$var, want_var))
  }
  invisible(TRUE)
}

# ── the grid ───────────────────────────────────────────────────────────────
# Each row names what it is for; a regression that only breaks one regime
# should say which regime in its failure message.
grid <- rbind(
  data.frame(case = "zero_no_inflation",    y = 0,   mu = 5.0,   k = 2.0,   pi = 0.0),
  data.frame(case = "positive_no_inflation", y = 7,  mu = 5.0,   k = 2.0,   pi = 0.0),
  data.frame(case = "zero_mid_pi",          y = 0,   mu = 5.0,   k = 2.0,   pi = 0.4),
  data.frame(case = "positive_mid_pi",      y = 7,   mu = 5.0,   k = 2.0,   pi = 0.4),
  data.frame(case = "zero_small_pi",        y = 0,   mu = 5.0,   k = 2.0,   pi = 0.001),
  data.frame(case = "zero_large_pi",        y = 0,   mu = 5.0,   k = 2.0,   pi = 0.99),
  data.frame(case = "positive_large_pi",    y = 3,   mu = 5.0,   k = 2.0,   pi = 0.99),
  data.frame(case = "zero_large_mu",        y = 0,   mu = 200.0, k = 3.0,   pi = 0.3),
  data.frame(case = "zero_tiny_k",          y = 0,   mu = 5.0,   k = 0.05,  pi = 0.3),
  data.frame(case = "positive_tiny_k",      y = 12,  mu = 5.0,   k = 0.05,  pi = 0.3),
  data.frame(case = "zero_large_k",         y = 0,   mu = 5.0,   k = 500.0, pi = 0.2),
  data.frame(case = "positive_large_k",     y = 4,   mu = 5.0,   k = 500.0, pi = 0.2),
  data.frame(case = "positive_small_mu",    y = 1,   mu = 0.05,  k = 1.5,   pi = 0.25),
  data.frame(case = "zero_small_mu",        y = 0,   mu = 0.05,  k = 1.5,   pi = 0.25),
  data.frame(case = "positive_far_tail",    y = 60,  mu = 5.0,   k = 2.0,   pi = 0.3)
)

for (i in seq_len(nrow(grid))) {
  check_moments(grid$mu[i], grid$k[i], grid$pi[i])
}
cat(sprintf("structural checks passed on %d parameter settings\n", nrow(grid)))

# ── reference values ───────────────────────────────────────────────────────
# numDeriv differentiates in (mu, k, pi). At a pi boundary a central
# difference would step outside [0, 1], so those rows are differentiated
# one-sided into the interior.
ref_row <- function(case, y, mu, k, pi) {
  f <- function(theta) zinb_logpmf(y, theta[1], theta[2], theta[3])
  # `side = NA` is the two-sided Richardson difference, accurate to ~1e-10
  # relative. At a pi boundary a two-sided step would leave [0, 1], so pi is
  # differentiated one-sided there and is roughly five orders of magnitude
  # less accurate. That is a property of the differencing, not a disagreement,
  # so it is recorded per row and the consumer widens its tolerance rather
  # than every row paying for the boundary ones.
  side <- c(NA, NA, NA)
  onesided <- (pi <= 0) || (pi >= 1)
  if (pi <= 0) side[3] <- 1
  if (pi >= 1) side[3] <- -1
  g <- numDeriv::grad(f, c(mu, k, pi), side = side,
                      method = "Richardson")
  data.frame(case = case, y = y, mu = mu, k = k, pi = pi,
             logpmf = f(c(mu, k, pi)),
             d_mu = g[1], d_k = g[2], d_pi = g[3],
             d_pi_onesided = if (onesided) 1L else 0L)
}

ref <- do.call(rbind, lapply(seq_len(nrow(grid)), function(i)
  ref_row(grid$case[i], grid$y[i], grid$mu[i], grid$k[i], grid$pi[i])))

g17 <- function(x) vapply(x, function(v) sprintf("%.17g", v), character(1))

con <- file(out_path, "w")
writeLines(c(
  "# ZINB log-pmf and its gradient in (mu, k, pi), for camdl's zero_inflated",
  "# observation likelihood.",
  sprintf("# Oracle: base R `dnbinom` for the NB2 density and `numDeriv` %s",
          pkg_version),
  sprintf("# (Richardson extrapolation) for the gradient, under R %s.", r_version),
  "# The mixture is checked against normalization and the ZINB moment",
  "# identities before this file is written; see the generator.",
  "# Regenerate with: Rscript scripts/gen_zinb_gradient_fixture.R",
  "# Values are printed at %.17g so they round-trip through IEEE-754 exactly.",
  "# d_pi_onesided = 1 marks a row whose pi sits on a boundary, where the",
  "# reference d_pi is a one-sided difference and carries ~1e-4 relative",
  "# accuracy instead of ~1e-10. d_mu and d_k stay two-sided on every row.",
  "case\ty\tmu\tk\tpi\tlogpmf\td_mu\td_k\td_pi\td_pi_onesided"
), con)
writeLines(sprintf("%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d",
                   ref$case, as.integer(ref$y), g17(ref$mu), g17(ref$k),
                   g17(ref$pi), g17(ref$logpmf), g17(ref$d_mu), g17(ref$d_k),
                   g17(ref$d_pi), ref$d_pi_onesided), con)
close(con)

cat(sprintf("wrote %d rows to %s\n", nrow(ref), out_path))
print(ref, row.names = FALSE)
