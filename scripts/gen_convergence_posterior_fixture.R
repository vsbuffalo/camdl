#!/usr/bin/env Rscript
# Generate the external oracle for camdl's rank-normalized convergence
# diagnostics (gh#84): Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021),
# "Rank-normalization, folding, and localization: An improved R-hat for
# assessing convergence of MCMC", Bayesian Analysis 16(2):667-718.
#
# The oracle is the R `posterior` package — the reference implementation the
# paper's authors maintain, and the same algorithm Stan reports. camdl must
# reproduce its numbers; an independent camdl-side reimplementation would not
# be independent evidence (a misread of, say, the rank-offset convention would
# be misread identically in both halves).
#
# This script writes BOTH halves of the fixture:
#   * the draws themselves (so camdl and posterior score identical numbers),
#   * posterior's statistics on those draws (the reference values).
#
# It is NOT run by CI — both TSVs are committed so the Rust test stays
# offline-safe and needs no R. Regenerate only when deliberately re-pinning
# against a newer `posterior`.
#
# Run: Rscript scripts/gen_convergence_posterior_fixture.R

if (!requireNamespace("posterior", quietly = TRUE)) {
  stop("install posterior first: install.packages('posterior')")
}
suppressPackageStartupMessages(library(posterior))

pkg_version <- as.character(utils::packageVersion("posterior"))
r_version <- paste(R.version$major, R.version$minor, sep = ".")

set.seed(20260821)

# ── draw generators ────────────────────────────────────────────────────────
# Each returns an (iterations x chains) matrix, the layout `posterior`'s
# matrix methods expect.

ar1 <- function(n, phi, sd = 1, start = 0) {
  x <- numeric(n)
  x[1] <- start
  for (i in seq_len(n - 1) + 1) x[i] <- phi * x[i - 1] + rnorm(1, 0, sd)
  x
}

# A random-walk Metropolis chain on a N(0,1) target. A rejected proposal
# repeats the previous state EXACTLY — the tie structure that PMMH produces
# and that naive (non-average) ranking would distort.
rwm <- function(n, prop_sd, start = 0) {
  x <- numeric(n)
  cur <- start
  for (i in seq_len(n)) {
    prop <- cur + rnorm(1, 0, prop_sd)
    if (log(runif(1)) < dnorm(prop, log = TRUE) - dnorm(cur, log = TRUE)) {
      cur <- prop
    }
    x[i] <- cur
  }
  x
}

cases <- list()

# 1. Four well-mixed AR(1) chains on a common target. The baseline: R-hat near
#    1, ESS a sizable fraction of N.
cases$ar1_mixed <- sapply(1:4, function(k) ar1(250, 0.6, start = rnorm(1)))

# 2. Chain MEANS agree while each chain drifts across its own run in
#    alternating directions. Classic (unsplit) R-hat is blind to this; split
#    R-hat is not. This is the failure mode measured on the ebola 8-chain PGAS
#    fit in gh#84.
cases$within_chain_drift <- sapply(1:4, function(k) {
  n <- 250
  slope <- c(1, -1, 1, -1)[k] * 6 / n
  ar1(n, 0.5) + slope * (seq_len(n) - n / 2)
})

# 3. Chains agree on location and disagree on SCALE. Only the folded statistic
#    (R-hat of |x - median(x)|) sees it. Particle filters produce exactly this
#    when per-chain particle diversity differs.
cases$scale_disagree <- sapply(1:4, function(k) rnorm(250, 0, c(0.5, 1, 2, 4)[k]))

# 4. Random-walk Metropolis with a deliberately wide proposal: ~75% of steps
#    are rejections, i.e. exact repeats. Average-ranked ties.
cases$rwm_ties <- sapply(1:4, function(k) rwm(250, 6, start = rnorm(1)))

# 5. Heavy-tailed, right-skewed marginal with chains centred differently.
#    Classic R-hat assumes finite variance; the rank-normalized statistic is
#    invariant to the monotone transform.
cases$heavy_tail <- sapply(1:4, function(k) exp(ar1(250, 0.7) + c(0, 0.4, -0.3, 0.8)[k]))

# 6. An ODD number of draws per chain, which forces the split convention to
#    drop the middle draw rather than overlap the halves.
cases$odd_draws <- sapply(1:4, function(k) ar1(101, 0.4, start = rnorm(1)))

# 7. Strongly ANTITHETIC chains (MA(1) with negative coefficient, lag-1
#    autocorrelation about -0.5). Exercises the Geyer truncation branch where
#    the very first pair sum is non-positive, and ESS exceeds the draw count.
cases$antithetic <- sapply(1:4, function(k) {
  e <- rnorm(251)
  e[-1] - 0.95 * e[-251]
})

# 8. One chain frozen at a single value while the others mix — a chain that
#    never accepted a move. Its within-chain variance is exactly zero, but the
#    pooled draws are NOT constant, so a statistic is still defined.
cases$one_stuck_chain <- cbind(
  sapply(1:3, function(k) ar1(250, 0.5, start = rnorm(1))),
  rep(0.37, 250)
)

# 9. A probability parameter piled at its LOWER bound: about half the draws are
#    exactly 0. Massive ties, and the 5% tail indicator is still informative.
cases$lower_bound_pileup <- sapply(1:4, function(k) pmax(0, ar1(250, 0.6, start = rnorm(1))))

# 10. A probability parameter piled at its UPPER bound: the top 5% of draws are
#     all exactly 1, so the 95% tail indicator is constant and tail-ESS is not
#     defined. posterior returns NA here; camdl must not invent a number.
cases$upper_bound_pileup <- sapply(1:4, function(k) pmin(1, 0.55 + 0.35 * ar1(250, 0.6)))

# 11. The smallest input the estimator accepts: 2 chains of 8 draws.
cases$two_chains_short <- sapply(1:2, function(k) ar1(8, 0.3, start = rnorm(1)))

# 12. Every draw identical across every chain — the degenerate-variance guard.
#     posterior returns NA for all four statistics.
cases$all_constant <- matrix(2.5, nrow = 50, ncol = 4)

# ── reference statistics ───────────────────────────────────────────────────
# rhat            posterior::rhat        max(rank-normalized split R-hat,
#                                        folded rank-normalized split R-hat)
# rhat_split      posterior::rhat_basic  split R-hat on the RAW scale
# rhat_classic    posterior::rhat_basic(split = FALSE)  Gelman & Rubin (1992)
# ess_bulk        posterior::ess_bulk    rank-normalized split ESS
# ess_tail        posterior::ess_tail    min ESS of the 5% / 95% tail indicators

stat_row <- function(name, mat) {
  quiet <- function(expr) suppressWarnings(withCallingHandlers(expr,
    message = function(m) invokeRestart("muffleMessage")))
  data.frame(
    case = name,
    statistic = c("n_chains", "n_draws", "rhat", "rhat_split", "rhat_classic",
                  "ess_bulk", "ess_tail"),
    value = c(
      ncol(mat),
      nrow(mat),
      quiet(rhat(mat)),
      quiet(rhat_basic(mat)),
      quiet(rhat_basic(mat, split = FALSE)),
      quiet(ess_bulk(mat)),
      quiet(ess_tail(mat))
    ),
    stringsAsFactors = FALSE
  )
}

g17 <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))

chains_path <- "rust/crates/sim/tests/fixtures/convergence_chains.tsv"
ref_path <- "rust/crates/sim/tests/fixtures/convergence_posterior_ref.tsv"
dir.create(dirname(chains_path), recursive = TRUE, showWarnings = FALSE)

header <- function(what) c(
  sprintf("# %s", what),
  sprintf("# Oracle: R package `posterior` %s under R %s.", pkg_version, r_version),
  "# Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), Bayesian Analysis",
  "# 16(2):667-718. doi:10.1214/20-BA1221",
  "# Regenerate with: Rscript scripts/gen_convergence_posterior_fixture.R",
  "# Values are printed at %.17g so they round-trip through IEEE-754 exactly."
)

con <- file(chains_path, "w")
writeLines(c(
  header("Draws scored by both posterior and camdl (gh#84)."),
  "# chain and draw indices are 0-based.",
  "case\tchain\tdraw\tvalue"
), con)
for (nm in names(cases)) {
  mat <- cases[[nm]]
  for (j in seq_len(ncol(mat))) {
    writeLines(sprintf("%s\t%d\t%d\t%s", nm, j - 1L, seq_len(nrow(mat)) - 1L,
                       g17(mat[, j])), con)
  }
}
close(con)

ref <- do.call(rbind, lapply(names(cases), function(nm) stat_row(nm, cases[[nm]])))
con <- file(ref_path, "w")
writeLines(c(
  header("posterior's convergence statistics on convergence_chains.tsv (gh#84)."),
  "# `NA` means posterior declines to report the statistic for that case.",
  "case\tstatistic\tvalue"
), con)
writeLines(sprintf("%s\t%s\t%s", ref$case, ref$statistic, g17(ref$value)), con)
close(con)

cat(sprintf("wrote %d cases to %s and %s\n", length(cases), chains_path, ref_path))
print(ref, row.names = FALSE)
