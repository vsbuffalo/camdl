#!/usr/bin/env Rscript
# Run once after installing R: initializes renv and installs dependencies.
# Usage: cd bench && Rscript setup.R

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

renv::init(bare = TRUE)
renv::install(c("pomp", "dplyr", "tidyr"))
renv::snapshot()

message("Done. renv.lock created. Run 'make all' to generate comparison data.")
