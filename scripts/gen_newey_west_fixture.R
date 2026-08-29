#!/usr/bin/env Rscript
# Generate the external oracle for camdl's Diebold-Mariano / Newey-West
# long-run variance of a loss-differential series (Stage 4.3 of the
# 2026-08-29 honest-predictive-evaluation proposal).
#
# Why this is an oracle and not a second camdl implementation: the reference
# long-run variance comes from sandwich::lrvar (Zeileis 2004, J. Stat.
# Software 11(10); Newey & West 1987), an independent, widely used
# implementation. The lag choice is pinned to sandwich's own default
# bandwidth floor(4*(T/100)^(2/9)) so a future "improvement" cannot silently
# change reported SEs. The HLN small-sample correction (Harvey, Leybourne &
# Newbold 1997) and lag-1 autocorrelation are computed here from their
# definitional formulas.
#
# The fixture is committed so the Rust test stays offline and CI never
# needs R. Regenerate only when deliberately re-pinning.
#
# Run: Rscript scripts/gen_newey_west_fixture.R

if (!requireNamespace("sandwich", quietly = TRUE)) {
  stop("install sandwich first: install.packages('sandwich')")
}

pkg_version <- as.character(utils::packageVersion("sandwich"))
r_version <- paste(R.version$major, R.version$minor, sep = ".")

repo_root <- tryCatch(
  system2("git", c("rev-parse", "--show-toplevel"), stdout = TRUE),
  error = function(e) stop("run inside the camdl git repository")
)
out_path <- file.path(repo_root, "rust", "crates", "cli", "tests",
                      "fixtures", "newey_west_ref.tsv")
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

g17 <- function(x) vapply(x, function(v) sprintf("%.17g", v), character(1))
join17 <- function(x) paste(g17(x), collapse = ",")

# sandwich's default bandwidth for NeweyWest (bwNeweyWest prewhitened is
# different; we pin the plain textbook rule the proposal states).
nw_lag <- function(t) floor(4 * (t / 100)^(2 / 9))

# Long-run variance of the MEAN of d via the Bartlett kernel:
#   sigma2_NW = gamma0 + 2 * sum_{k=1..L} (1 - k/(L+1)) gamma_k
# with gamma_k the biased (1/T) autocovariances. se(mean) =
# sqrt(sigma2_NW / T). sandwich::lrvar(d, type = "Newey-West",
# prewhite = FALSE, adjust = FALSE, lag = L) returns sigma2_NW / T
# directly (the variance of the mean).
ref_row <- function(case, d) {
  t <- length(d)
  lag <- nw_lag(t)
  v <- sandwich::lrvar(d, type = "Newey-West", prewhite = FALSE,
                       adjust = FALSE, lag = lag)
  se_mean <- sqrt(v)
  # HLN correction on the DM statistic for h = 1:
  #   c = sqrt((T + 1 - 2h + h(h-1)/T) / T) = sqrt(1 - 1/T) at h = 1.
  hln <- sqrt((t + 1 - 2 + 0) / t)
  r1 <- {
    dm <- d - mean(d)
    sum(dm[-1] * dm[-t]) / sum(dm^2)
  }
  data.frame(case = case, t = t, lag = lag,
             se_mean = se_mean, hln = hln, lag1 = r1,
             d = join17(d))
}

set.seed(48)
cases <- list(
  ref_row("iid_normal", rnorm(60, mean = 0.3, sd = 1.0)),
  ref_row("ar1_positive", as.numeric(
    arima.sim(list(ar = 0.6), n = 80) + 0.2)),
  ref_row("ar1_negative", as.numeric(
    arima.sim(list(ar = -0.5), n = 40))),
  ref_row("short_series", rnorm(12, mean = 1.0, sd = 0.5)),
  ref_row("long_persistent", as.numeric(
    arima.sim(list(ar = 0.85), n = 200)))
)
ref <- do.call(rbind, cases)

con <- file(out_path, "w")
writeLines(c(
  "# Newey-West long-run SE of a loss-differential mean, HLN correction,",
  "# and lag-1 autocorrelation, for camdl compare's Diebold-Mariano se(delta)",
  "# (Stage 4.3 of the 2026-08-29 honest-predictive-evaluation proposal).",
  sprintf("# Oracle: sandwich %s (lrvar, type = Newey-West, prewhite = FALSE,",
          pkg_version),
  sprintf("# adjust = FALSE) under R %s; lag pinned to floor(4*(T/100)^(2/9)).", r_version),
  "# se_mean = sqrt(lrvar) is the HAC SE of mean(d); hln = the Harvey-",
  "# Leybourne-Newbold h = 1 factor sqrt(1 - 1/T); lag1 = the sample lag-1",
  "# autocorrelation of d.",
  "# Regenerate with: Rscript scripts/gen_newey_west_fixture.R",
  "# Values are printed at %.17g so they round-trip through IEEE-754 exactly.",
  "case\tt\tlag\tse_mean\thln\tlag1\td"
), con)
writeLines(sprintf("%s\t%d\t%d\t%s\t%s\t%s\t%s",
                   ref$case, ref$t, ref$lag, g17(ref$se_mean),
                   g17(ref$hln), g17(ref$lag1), ref$d), con)
close(con)

cat(sprintf("wrote %d rows to %s\n", nrow(ref), out_path))
print(ref[, c("case", "t", "lag", "se_mean", "hln", "lag1")], row.names = FALSE)
