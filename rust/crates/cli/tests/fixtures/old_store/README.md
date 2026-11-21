# An old-store leaf

`root/` is an old-store root (the `results/` directory `runid` writes) holding
one leaf (`camdl simulate` of `../reporting/sir.camdl`,
`--param beta=0.35 --param gamma=0.12 --param rho=0.5
--param N0=2000`, seed 1),
committed so a test has an old-store record that no command has to write:
`stored_survey_leaf_is_not_read.rs` copies it in as a real leaf, and uses its
`run.json` as the template for a leaf of a kind this camdl does not register.
The record's `argv[0]` is `camdl`, not the path of the binary that wrote it.
