# IF2 Transforms, Scaling, and Fitting Philosophy

**Status:** Mostly implemented. Remaining: boundary-hit diagnostic, preflight
report.

---

## Implementation status

| Item                                            | Status      | Commit                 |
| ----------------------------------------------- | ----------- | ---------------------- |
| `param_kind` in IR                              | ✅ shipped  | 49bf22e                |
| Type-driven transforms (param_kind → log/logit) | ✅ shipped  | 49bf22e                |
| Transform carries bounds (Log { lo, hi })       | ✅ shipped  | 4a90daa                |
| Log clamp on from_transformed                   | ✅ shipped  | 4a90daa                |
| Auto rw_sd /20 log, /6 logit                    | ✅ shipped  | 4a90daa                |
| cooling_target_iters = n_iterations             | ✅ shipped  | 4a90daa                |
| Scout rw_sd_scale 1.5 → 1.0                     | ✅ shipped  | 4a90daa                |
| global_step powf (overflow fix)                 | ✅ shipped  | eb804b4                |
| Gamma/Binomial/Poisson sampler guards           | ✅ shipped  | faad081, 351733b       |
| NaN in MAD fix                                  | ✅ shipped  | 1c12faf                |
| Sampler stress tests (26 tests)                 | ✅ shipped  | 1c12faf                |
| Cooling verification                            | ✅ verified | (math check, not code) |
| **Boundary-hit diagnostic**                     | 🔲 TODO     | ~15 lines              |
| **Preflight transform report**                  | 🔲 TODO     | ~30 lines              |
| **Diagnostics in summary JSON**                 | 🔲 TODO     | ~10 lines              |

---

## Remaining work

### 1. Boundary-hit diagnostic (~15 lines in IF2 engine)

Count per-parameter clamp activations per iteration. When >10% of particle-steps
hit the clamp, record in `ParamIterDiag` and report to stderr. The count goes
into `parameter_traces.tsv` and the method's summary JSON for agent consumption.

### 2. Preflight transform report (~30 lines in runner/if2)

Before IF2 starts, print the transform regime:

- Parameter name, transform type, bounds, transformed position
- Compression warning for logit params near boundaries (|z| > 2)
- Auto rw_sd values with per-step transformed-scale equivalent
- Cooling schedule preview (% at iterations 1, mid, end)

### 3. Diagnostics in summary JSON (~10 lines)

Add to scout/refine/validate summary JSONs:

- `preflight`: transform table, compression warnings
- `boundary_hits`: per-parameter clamp fraction
- `diagnostics.wvr` and `diagnostics.q_ratio` summaries

This enables an agent to read the JSON and diagnose fitting problems without
parsing stderr.
