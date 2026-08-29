#!/usr/bin/env Rscript
# Generate the external oracle for camdl's prequential scoring-rule kernels:
# sample CRPS (both estimators), the mixture log score, and the randomized
# PIT (gh#628, gh#629).
#
# Why this is an oracle and not a second camdl implementation:
#
#   * the edf CRPS comes from `scoringRules::crps_sample(method = "edf")`
#     (Jordan, Kruger & Lerch 2019, J. Stat. Software 90(12)) -- an
#     independent, published implementation of the same estimator camdl's
#     `crps_sample` computes via the sorted-sample identity, so agreement is
#     expected to machine precision, not tolerance;
#   * the closed-form rows score large committed ensembles against
#     `crps_pois` / `crps_nbinom` / `crps_norm` -- analytic CRPS values that
#     know nothing about sample estimators, so agreement (within the stated
#     Monte Carlo tolerance) is evidence about the estimator itself, for the
#     fair form including its finite-ensemble unbiasedness;
#   * the log-score rows are checked in R with densities from `dpois` /
#     `dnbinom`, and the point-mass mixture rows against the parametric
#     density directly, which the mixture must reproduce exactly.
#
# The fair CRPS (Ferro 2014) and the randomized PIT (Smith 1985; Brockwell
# 2007) have no closed-form scoringRules entry; their reference values are
# computed here from their definitional forms (the O(S^2) pairwise sum; the
# literal atom split), independent of camdl's order-statistic and counting
# implementations. Before writing, the script checks both against structure
# they must satisfy -- the fair/edf normalization identity on every row, and
# calibration of the randomized PIT on a Poisson forecast -- and aborts
# rather than writing a fixture if either fails.
#
# The fixture is committed so the Rust test stays offline and CI never needs
# R. Regenerate only when deliberately re-pinning.
#
# Run: Rscript scripts/gen_prequential_scores_fixture.R

if (!requireNamespace("scoringRules", quietly = TRUE)) {
  stop("install scoringRules first: install.packages('scoringRules')")
}

pkg_version <- as.character(utils::packageVersion("scoringRules"))
r_version <- paste(R.version$major, R.version$minor, sep = ".")

repo_root <- tryCatch(
  system2("git", c("rev-parse", "--show-toplevel"), stdout = TRUE),
  error = function(e) stop("run inside the camdl git repository")
)
out_path <- file.path(repo_root, "rust", "crates", "sim", "tests",
                      "fixtures", "prequential_scores_ref.tsv")

g17 <- function(x) vapply(x, function(v) sprintf("%.17g", v), character(1))
join17 <- function(x) paste(g17(x), collapse = ",")

# ── definitional reference forms ───────────────────────────────────────────
# Fair CRPS by its pairwise definition (Ferro 2014; as written in Zamo &
# Naveau 2018). O(S^2) on purpose: no shared shortcut with camdl's
# order-statistic implementation.
crps_fair_pairwise <- function(x, y) {
  s <- length(x)
  if (s == 1) return(abs(x - y))
  mean(abs(x - y)) - sum(abs(outer(x, x, "-"))) / (2 * s * (s - 1))
}

# Randomized PIT at a supplied v: the literal atom split.
pit_at_v <- function(x, y, v) (sum(x < y) + v * sum(x == y)) / length(x)

# Numerically stable log(mean(exp(ll))).
logmeanexp <- function(ll) {
  m <- max(ll)
  m + log(mean(exp(ll - m)))
}

# ── structural checks (abort before writing if any fails) ──────────────────
check_fair_edf_identity <- function(x, y, tol = 1e-12) {
  s <- length(x)
  t1 <- mean(abs(x - y))
  edf <- scoringRules::crps_sample(y, x, method = "edf")
  fair <- crps_fair_pairwise(x, y)
  want <- t1 - (s / (s - 1)) * (t1 - edf)
  if (abs(fair - want) > tol * max(1, abs(want))) {
    stop(sprintf("fair/edf identity failed at y=%g: %.17g vs %.17g",
                 y, fair, want))
  }
}

check_pointmass_mixture <- function(ll_point, dens, tol = 1e-12) {
  got <- logmeanexp(ll_point)
  if (abs(got - dens) > tol * max(1, abs(dens))) {
    stop(sprintf("point-mass mixture != parametric density: %.17g vs %.17g",
                 got, dens))
  }
}

check_randomized_pit_calibration <- function() {
  # A calibrated Poisson(3) forecast: the randomized PIT must average 0.5.
  set.seed(101)
  n_rep <- 4000
  u <- replicate(n_rep, {
    x <- rpois(400, 3)
    y <- rpois(1, 3)
    pit_at_v(x, y, runif(1))
  })
  if (abs(mean(u) - 0.5) > 0.02) {
    stop(sprintf("randomized PIT mis-calibrated on Poisson(3): mean %.4f",
                 mean(u)))
  }
}

check_randomized_pit_calibration()

# ── exact-agreement cases: CRPS (both estimators) and PIT ──────────────────
# Small vectors with ties, unsorted input, negatives, fractional values.
exact_cases <- list(
  list(case = "ties_at_obs",     x = c(1, 2, 2, 3),              y = 2),
  list(case = "frac_with_ties",  x = c(0.5, 1, 2, 2, 7.5),       y = 1.5),
  list(case = "unsorted_counts", x = c(5, 1, 4, 4, 4, 9, 0),     y = 4),
  list(case = "negatives_zero",  x = c(-2.5, 0, 0, 3.25),        y = 0),
  list(case = "obs_below_all",   x = c(3, 4, 7, 7),              y = 1),
  list(case = "obs_above_all",   x = c(3, 4, 7, 7),              y = 11),
  list(case = "two_draws",       x = c(2, 5),                    y = 3)
)

rows <- list()
for (cs in exact_cases) {
  check_fair_edf_identity(cs$x, cs$y)
  rows[[length(rows) + 1]] <- data.frame(
    case = cs$case, axis = "crps_exact", y = cs$y,
    oracle_a = scoringRules::crps_sample(cs$y, cs$x, method = "edf"),
    oracle_b = crps_fair_pairwise(cs$x, cs$y),
    oracle_c = NA_real_, tol = 1e-12,
    samples = join17(cs$x))
  rows[[length(rows) + 1]] <- data.frame(
    case = cs$case, axis = "pit_exact", y = cs$y,
    oracle_a = pit_at_v(cs$x, cs$y, 0),
    oracle_b = pit_at_v(cs$x, cs$y, 0.5),
    oracle_c = pit_at_v(cs$x, cs$y, 1),
    tol = 1e-12,
    samples = join17(cs$x))
}

# ── log-score cases: the mixture density over per-particle likelihoods ─────
# `samples` carries per-particle log-likelihoods (the kernel's input);
# densities from dpois/dnbinom, mixture combined by stable logmeanexp.
ls_cases <- list(
  list(case = "pois_mixture",
       ll = dpois(3, c(1.5, 3.0, 6.0, 2.2), log = TRUE)),
  list(case = "nbinom_mixture",
       ll = dnbinom(7, size = 1.5, mu = c(4, 8, 16, 2, 9), log = TRUE)),
  list(case = "deep_negative_tail",
       ll = c(-745, -746, -750, -744.5))
)
for (cs in ls_cases) {
  rows[[length(rows) + 1]] <- data.frame(
    case = cs$case, axis = "logscore", y = NA_real_,
    oracle_a = logmeanexp(cs$ll), oracle_b = NA_real_, oracle_c = NA_real_,
    tol = 1e-12,
    samples = join17(cs$ll))
}
# Point-mass mixture must reproduce the parametric density exactly; the
# script checks this identity itself, then commits it as a row.
ll_point <- rep(dpois(4, 2.7, log = TRUE), 8)
check_pointmass_mixture(ll_point, dpois(4, 2.7, log = TRUE))
rows[[length(rows) + 1]] <- data.frame(
  case = "pois_point_mass", axis = "logscore", y = NA_real_,
  oracle_a = dpois(4, 2.7, log = TRUE), oracle_b = NA_real_,
  oracle_c = NA_real_, tol = 1e-12,
  samples = join17(ll_point))

# ── closed-form cases: large ensembles vs analytic CRPS ────────────────────
# The committed ensemble is deterministic, so `tol` covers the fixed,
# realized Monte Carlo error of THIS ensemble; per-case tol is set a priori
# from the estimator's standard error sd(|X−y|)/sqrt(n) at ~3 sigma, and the
# generator verifies the realized error before writing.
#
# Count ensembles are committed as a histogram ("value:count,...") — CRPS and
# PIT are permutation-invariant, so the ensemble reconstructs exactly, and
# n = 1e5 costs a few dozen tokens instead of megabytes. The continuous
# normal ensemble has no compact encoding and stays at n = 2000 raw draws
# with a correspondingly honest tolerance.
#
# The fair estimator's pairwise term is computed here from the histogram
# (grouped double sum) rather than `outer` on 1e5 values.
crps_fair_hist <- function(vals, cnts, y) {
  n <- sum(cnts)
  t1 <- sum(cnts * abs(vals - y)) / n
  pair <- 0
  for (i in seq_along(vals)) {
    pair <- pair + sum(cnts[i] * cnts * abs(vals[i] - vals))
  }
  t1 - pair / (2 * n * (n - 1))
}
crps_edf_hist <- function(vals, cnts, y) {
  n <- sum(cnts)
  t1 <- sum(cnts * abs(vals - y)) / n
  pair <- 0
  for (i in seq_along(vals)) {
    pair <- pair + sum(cnts[i] * cnts * abs(vals[i] - vals))
  }
  t1 - pair / (2 * n * n)
}
join_hist <- function(vals, cnts) {
  paste(sprintf("%s:%d", g17(vals), cnts), collapse = ",")
}

set.seed(20260829)
n_hist <- 100000
hist_cases <- list(
  list(case = "pois_closed_form",
       draws = as.numeric(rpois(n_hist, 4.2)), y = 3, tol = 0.03,
       analytic = scoringRules::crps_pois(3, lambda = 4.2)),
  list(case = "nbinom_closed_form",
       draws = as.numeric(rnbinom(n_hist, size = 2, mu = 8)), y = 6, tol = 0.07,
       analytic = scoringRules::crps_nbinom(6, size = 2, mu = 8))
)
for (cs in hist_cases) {
  tab <- table(cs$draws)
  vals <- as.numeric(names(tab))
  cnts <- as.integer(tab)
  for (est in c(crps_edf_hist(vals, cnts, cs$y),
                crps_fair_hist(vals, cnts, cs$y))) {
    if (abs(est - cs$analytic) > cs$tol) {
      stop(sprintf(
        "%s: realized MC error %.4f exceeds tol %.3f; enlarge the ensemble",
        cs$case, abs(est - cs$analytic), cs$tol))
    }
  }
  rows[[length(rows) + 1]] <- data.frame(
    case = cs$case, axis = "crps_closed_form_hist", y = cs$y,
    oracle_a = cs$analytic, oracle_b = NA_real_, oracle_c = NA_real_,
    tol = cs$tol,
    samples = join_hist(vals, cnts))
}

norm_draws <- rnorm(2000, mean = 3, sd = 2)
norm_y <- 2.5
norm_tol <- 0.15  # sd(|X−y|)/sqrt(2000) ≈ 0.045; ~3 sigma
norm_analytic <- scoringRules::crps_norm(norm_y, mean = 3, sd = 2)
for (est in c(scoringRules::crps_sample(norm_y, norm_draws, method = "edf"),
              crps_fair_pairwise(norm_draws, norm_y))) {
  if (abs(est - norm_analytic) > norm_tol) {
    stop(sprintf(
      "norm_closed_form: realized MC error %.4f exceeds tol %.3f",
      abs(est - norm_analytic), norm_tol))
  }
}
rows[[length(rows) + 1]] <- data.frame(
  case = "norm_closed_form", axis = "crps_closed_form", y = norm_y,
  oracle_a = norm_analytic, oracle_b = NA_real_, oracle_c = NA_real_,
  tol = norm_tol,
  samples = join17(norm_draws))

ref <- do.call(rbind, rows)

fmt_opt <- function(x) ifelse(is.na(x), "NA", g17(x))

con <- file(out_path, "w")
writeLines(c(
  "# External oracle for camdl's prequential scoring-rule kernels (gh#628,",
  "# gh#629): sample CRPS (edf and fair estimators), mixture log score, and",
  "# randomized PIT.",
  sprintf("# Oracle: scoringRules %s under R %s for the edf CRPS and the",
          pkg_version, r_version),
  "# analytic crps_pois/crps_nbinom/crps_norm; definitional pairwise/atom",
  "# forms (checked structurally; see the generator) for fair CRPS and PIT;",
  "# dpois/dnbinom densities for the log-score mixtures.",
  "# Regenerate with: Rscript scripts/gen_prequential_scores_fixture.R",
  "# Values are printed at %.17g so they round-trip through IEEE-754 exactly.",
  "# Row meaning by axis:",
  "#   crps_exact:       samples = ensemble; a = edf CRPS, b = fair CRPS",
  "#   pit_exact:        samples = ensemble; a/b/c = PIT at v = 0, 0.5, 1",
  "#   logscore:         samples = per-particle log-likelihoods; a = mixture",
  "#                     log score log(mean(exp(ll)))",
  "#   crps_closed_form: samples = large raw ensemble from the named family;",
  "#                     a = analytic CRPS; both estimators within tol",
  "#   crps_closed_form_hist: samples = ensemble as value:count histogram",
  "#                     (order-invariant scores reconstruct it exactly);",
  "#                     a = analytic CRPS; both estimators within tol",
  "case\taxis\ty\toracle_a\toracle_b\toracle_c\ttol\tsamples"
), con)
writeLines(sprintf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
                   ref$case, ref$axis, fmt_opt(ref$y), fmt_opt(ref$oracle_a),
                   fmt_opt(ref$oracle_b), fmt_opt(ref$oracle_c),
                   g17(ref$tol), ref$samples), con)
close(con)

cat(sprintf("wrote %d rows to %s\n", nrow(ref), out_path))
print(ref[, c("case", "axis", "y", "oracle_a", "oracle_b", "oracle_c", "tol")],
      row.names = FALSE)
