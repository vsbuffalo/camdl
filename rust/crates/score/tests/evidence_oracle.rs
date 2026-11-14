//! The Newey–West standard error, the HLN factor and the lag-1
//! autocorrelation of `score::evidence`, checked against the committed
//! external oracle `tests/fixtures/newey_west_ref.tsv`. It is an
//! integration test rather than a unit test in `src/evidence.rs` because it
//! reads the fixture from disk, and `tests/source.rs` forbids file access
//! anywhere under `src/`.

mod tests {
    use score::evidence::{hln_factor, lag1_autocorrelation, newey_west_lag, newey_west_se_of_sum};

    /// Stage 4.3: the Newey–West HAC SE, HLN factor, and lag-1
    /// autocorrelation against the committed external oracle
    /// (`sandwich::lrvar`, Zeileis 2004; definitional HLN/lag-1 forms
    /// computed in R). Regenerate with
    /// `Rscript scripts/gen_newey_west_fixture.R`. `lrvar` returns the
    /// variance of the MEAN, so `se_sum = T · se_mean`. Machine-precision
    /// agreement: same estimator, independent implementation.
    #[test]
    fn newey_west_matches_the_sandwich_oracle() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/newey_west_ref.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut n_rows = 0;
        for line in text.lines() {
            if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 7, "malformed fixture row: {line}");
            let case = f[0];
            let t: usize = f[1].parse().unwrap();
            let lag: usize = f[2].parse().unwrap();
            let se_mean: f64 = f[3].parse().unwrap();
            let hln: f64 = f[4].parse().unwrap();
            let lag1: f64 = f[5].parse().unwrap();
            let d: Vec<f64> = f[6].split(',').map(|v| v.parse().unwrap()).collect();
            assert_eq!(d.len(), t, "{case}: series length");

            assert_eq!(newey_west_lag(t), lag,
                "{case}: the pinned bandwidth rule must match sandwich's default");
            let se_sum = newey_west_se_of_sum(&d);
            let want = t as f64 * se_mean;
            assert!((se_sum - want).abs() <= 1e-10 * want.max(1.0),
                "{case}: NW se of sum {se_sum} vs sandwich {want}");
            assert!((hln_factor(t) - hln).abs() < 1e-12,
                "{case}: HLN factor");
            assert!((lag1_autocorrelation(&d) - lag1).abs() < 1e-12,
                "{case}: lag-1 autocorrelation");
            n_rows += 1;
        }
        assert!(n_rows >= 5, "fixture unexpectedly short: {n_rows} rows");
    }
}
