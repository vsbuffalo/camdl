#!/usr/bin/env Rscript
# pomp pfilter log-likelihood at the He et al. (2010) London MLE, on a
# HOLED (sparse) version of the weekly measles series.
#
# This is the SPARSE sibling of
#   tests/external/cases/he2010_pfilter_loglik/reference/reference.R
# (the DENSE case). It is byte-for-byte identical in model, parameters,
# particle count, replicate count, and seeds; the ONLY differences are:
#
#   1. The observed `cases` series has 222 weeks blanked to NA. The NA
#      pattern is taken DIRECTLY from the committed case data file
#      ../data/he2010_london_cases_holed.tsv (relative to reference/) —
#      the SAME file camdl scores — so both sides necessarily score the
#      identical holes.
#      ALL 1096 weekly time points remain in the grid; only VALUES go
#      missing.
#   2. The `dmeasure` Csnippet handles NA explicitly: a hole returns
#      likelihood 1 (log-contribution 0), so it adds no term to the
#      total log-likelihood.
#
# WHY THE WEEKLY GRID IS UNCHANGED (the whole point of this oracle):
#   pomp accumulator variables (accumvars = c("C","W")) are reset to 0 at
#   the START of each inter-observation rprocess interval (t[k], t[k+1]),
#   for EVERY k in the observation-time grid — NOT only at weeks whose
#   value is present. Source: pomp 6.4 `?accumvars` ("if a is a
#   state-variable named in the pomp's accumvars argument, then for each
#   interval (t[k],t[k+1]), k=0,1,2,..., a will be set to zero prior to
#   any rprocess computation over that interval"); King, Nguyen & Ionides
#   (2016) J. Stat. Soft. 69(12), doi:10.18637/jss.v069.i12. Because we
#   keep all 1096 weekly time points, every hole week's reset STILL fires.
#   The incidence bin C over a hole week therefore accumulates exactly one
#   week of new cases (fixed-bin incidence): the bin width is set by the
#   schedule, not by whether the value was observed. camdl's accumulator
#   reset must match this to agree.
#
# Data citation: He, Ionides & King (2010) J. R. Soc. Interface
# 7(43):271-283, doi:10.1098/rsif.2009.0151; via pomp's twentycities.rda.
#
# Output: out/pomp_pfilter_loglik.tsv  (sim, loglik) — one row per replicate.

library(pomp)
library(dplyr)

# ── Load demography covariates from the upstream data (same as dense) ─────────

# Download to a tempfile (NOT into reference/, so no generated artifact
# pollutes the reference_sha staleness fingerprint — same as the dense case).
url <- "https://kingaa.github.io/pomp/vignettes/twentycities.rda"
tmp <- tempfile(fileext = ".rda")
download.file(url, tmp, quiet = TRUE)
load(tmp)

TOWN <- "London"
demog |> filter(town == TOWN) |> select(-town) -> dem
dem |> mutate(birthrate = births / pop) -> dem
delay <- 4
t_fine <- with(dem, seq(from = min(year), to = max(year), by = 1/12))

covar <- covariate_table(
  t        = t_fine,
  pop      = predict(smooth.spline(dem$year, dem$pop), x = t_fine)$y,
  birthrate= predict(smooth.spline(dem$year + delay, dem$birthrate), x = t_fine)$y,
  times    = "t",
  order    = "constant"
)

theta <- c(
  R0 = 56.8, mu = 0.02, delay = 4, sigma = 28.9, gamma = 30.4,
  rho = 0.488, amplitude = 0.554, alpha = 0.976, iota = 2.9,
  cohort = 0.557, psi = 0.116, sigmaSE = 0.0878,
  S_0 = 0.0297, E_0 = 5.17e-05, I_0 = 5.14e-05, R_0 = 0.97
)
paramnames <- names(theta)

# ── Build the pomp time axis (year) exactly as the dense case does ────────────
# The dense case derives `year` from measles$date. We reuse that axis and
# overlay the holes read from the COMMITTED camdl-side data file, so the
# two implementations score identical NA weeks by construction.

measles |>
  filter(town == TOWN) |>
  mutate(year = as.numeric(format(date, "%Y")) +
                as.numeric(format(date, "%j")) / 365.25) |>
  arrange(year) |>
  select(year, cases) -> dat
stopifnot(nrow(dat) == 1096L, sum(is.na(dat$cases)) == 0L)

# ── Overlay the holes from the camdl-side data file ───────────────────────────
# Single source of truth: the holed series camdl reads. Its `weekly_cases`
# column is "NA" on hole weeks; the row order matches the date-sorted grid.
holed_path <- "../data/he2010_london_cases_holed.tsv"
stopifnot(file.exists(holed_path))
holed <- read.delim(holed_path, sep = "\t",
                    na.strings = "NA",
                    colClasses = c("numeric", "integer"))
stopifnot(nrow(holed) == 1096L,
          all(c("time", "weekly_cases") %in% names(holed)))
is_na_week <- is.na(holed$weekly_cases)
dat$cases[is_na_week] <- NA_integer_

n_na  <- sum(is.na(dat$cases))
n_obs <- sum(!is.na(dat$cases))
message(sprintf("loaded holed series: %d weeks, %d NA, %d observed",
                nrow(dat), n_na, n_obs))
stopifnot(n_na == 222L, n_obs == 874L)

m1 <- dat |>
  pomp(
    times = "year", t0 = with(dat, min(year) - 1/52),
    covar = covar, accumvars = c("C", "W"),
    rprocess = euler(
      step.fun = Csnippet("
        double beta, br, seas, foi, dw;
        double rate[6], trans[6];
        if (fabs(t - floor(t) - 251.0/365.0) < 0.5*dt)
          br = cohort*birthrate/dt + (1-cohort)*birthrate;
        else br = (1-cohort)*birthrate;
        double t_day = (t - floor(t)) * 365.25;
        if ((t_day>=7 && t_day<=100) || (t_day>=115 && t_day<=199) ||
            (t_day>=252 && t_day<=300) || (t_day>=308 && t_day<=356))
          seas = 1.0 + amplitude * 0.2411/0.7589;
        else seas = 1.0 - amplitude;
        beta = R0 * seas * (1.0 - exp(-(gamma+mu)*dt)) / dt;
        foi = beta * pow(I + iota, alpha) / pop;
        dw = rgammawn(sigmaSE, dt);
        rate[0] = foi * dw/dt; rate[1] = mu;
        rate[2] = sigma;       rate[3] = mu;
        rate[4] = gamma;       rate[5] = mu;
        reulermultinom(2, nearbyint(S), &rate[0], dt, &trans[0]);
        reulermultinom(2, nearbyint(E), &rate[2], dt, &trans[2]);
        reulermultinom(2, nearbyint(I), &rate[4], dt, &trans[4]);
        S += nearbyint(pop*br*dt) - trans[0] - trans[1];
        E += trans[0] - trans[2] - trans[3];
        I += trans[2] - trans[4] - trans[5];
        R = nearbyint(pop) - S - E - I;
        W += (dw - dt)/sigmaSE;
        C += trans[4];
      "),
      delta.t = 1/365.25
    ),
    rinit = Csnippet("
      double m = pop / (S_0 + E_0 + I_0 + R_0);
      S = nearbyint(m * S_0);
      E = nearbyint(m * E_0);
      I = nearbyint(m * I_0);
      R = nearbyint(m * R_0);
      W = 0; C = 0;
    "),
    # NA-AWARE dmeasure. The ONLY change from the dense case is the leading
    # `if (ISNA(cases))` branch: a hole returns likelihood 1 (log 0), so it
    # contributes no term to the total log-lik. The non-NA branches are
    # byte-identical to the dense reference. (pomp's C API exposes ISNA();
    # `give_log` selects mass vs log-mass, matching the dense snippet.)
    dmeasure = Csnippet("
      if (ISNA(cases)) {
        lik = (give_log) ? 0.0 : 1.0;
      } else {
        double m = rho * C;
        double v = m * (1.0 - rho + psi*psi*m);
        double tol = 1.0e-18;
        if (cases > 0.0) {
          lik = pnorm(cases+0.5, m, sqrt(v)+tol, 1, 0) -
                pnorm(cases-0.5, m, sqrt(v)+tol, 1, 0) + tol;
        } else {
          lik = pnorm(0.5, m, sqrt(v)+tol, 1, 0) + tol;
        }
        if (give_log) lik = log(lik);
      }
    "),
    statenames = c("S", "E", "I", "R", "C", "W"),
    paramnames = paramnames
  )

# ── pfilter replicates ───────────────────────────────────────────────────────

N_PARTICLES <- 2000
N_REPS      <- 20
SEED_BASE   <- 42L

message("Running ", N_REPS, " pfilter replicates × ", N_PARTICLES, " particles…")

set.seed(SEED_BASE)
logliks <- numeric(N_REPS)
cond_na_sum   <- numeric(N_REPS)  # sum of per-week conditional ll over HOLE weeks
cond_obs_sum  <- numeric(N_REPS)  # sum of per-week conditional ll over OBSERVED weeks
for (i in seq_len(N_REPS)) {
  pf <- pfilter(m1, params = theta, Np = N_PARTICLES)
  logliks[i] <- logLik(pf)
  # cond_logLik(pf) is the per-observation-time conditional log-lik
  # (length 1096; one entry per week in the grid). Holes must be exactly 0.
  cll <- cond_logLik(pf)
  cond_na_sum[i]  <- sum(cll[is_na_week])
  cond_obs_sum[i] <- sum(cll[!is_na_week])
  if (i %% 5 == 0) message("  rep ", i, "/", N_REPS, ": ll = ", round(logliks[i], 2))
}

# ── VERIFICATION 1: holes contribute exactly zero ─────────────────────────────
# If the NA-aware dmeasure works, every hole week's conditional log-lik is
# exactly 0, so the total log-lik == sum over the OBSERVED weeks only.
max_hole_contrib <- max(abs(cond_na_sum))
max_recon_err    <- max(abs(logliks - cond_obs_sum))
message(sprintf(
  "VERIFY-1 holes-are-zero: max |sum of hole-week cond.ll| = %.3e (want 0); ",
  max_hole_contrib))
message(sprintf(
  "VERIFY-1 partial-sum: max |total - sum(observed-week cond.ll)| = %.3e (want 0)",
  max_recon_err))
stopifnot(max_hole_contrib < 1e-9, max_recon_err < 1e-9)

# ── VERIFICATION 2: weekly reset of accumvars fires across holes ──────────────
# Probe C (cumulative incidence-in-bin) at every weekly time point: simulate
# the latent path once with the accumvars active and confirm C never carries a
# hole week's count into the next bin. Concretely: the C in a hole week is of
# the same order as its non-hole neighbours (one week of incidence), NOT ~2x
# (which is what we'd see if the reset were skipped for holes and the bin
# doubled).
set.seed(SEED_BASE + 1000L)
sim1 <- simulate(m1, params = theta, format = "data.frame", nsim = 1)
Cvals <- sim1$C
ratios <- c()
for (k in which(is_na_week)) {
  if (k > 1 && k < length(Cvals) && !is_na_week[k-1] && !is_na_week[k+1]) {
    nb <- mean(c(Cvals[k-1], Cvals[k+1]))
    if (nb > 0) ratios <- c(ratios, Cvals[k] / nb)
  }
}
message(sprintf(
  "VERIFY-2 weekly-reset: hole-week C vs neighbour-mean — median ratio %.3f (want ~1, NOT ~2); n=%d isolated holes probed",
  median(ratios), length(ratios)))
# A median ratio near 1 means the accumulator was reset at the hole week's
# grid time (one-week bin). A ratio near 2 would indicate a doubled bin
# (reset skipped for holes). Bound generously: epidemic dynamics make
# week-to-week C noisy, so we only assert it is far from the doubled regime.
stopifnot(median(ratios) > 0.4, median(ratios) < 1.6)

# ── Write output ─────────────────────────────────────────────────────────────

out_dir <- "out"
dir.create(out_dir, showWarnings = FALSE)

data.frame(sim = seq_len(N_REPS), loglik = logliks) |>
  write.table(file.path(out_dir, "pomp_pfilter_loglik.tsv"),
              sep = "\t", row.names = FALSE, quote = FALSE)

message("  → out/pomp_pfilter_loglik.tsv (",
        N_REPS, " replicates; mean ll = ", round(mean(logliks), 2),
        ", sd = ", round(sd(logliks), 2), ")")
message(sprintf("  observed weeks = %d, hole weeks = %d (%.2f%% observed)",
                n_obs, n_na, 100 * n_obs / nrow(dat)))
