# FOI scaling study — figures & data

Artifacts for
[`../../2026-05-29-foi-scaling-bench.md`](../../2026-05-29-foi-scaling-bench.md)
(the profiling investigation + optimization staging E→D→B). All are regenerable
from the committed harness — see the `make` targets noted per item. How to run
the benches/profilers:
[`rust/crates/sim/benches/README.md`](../../../../../rust/crates/sim/benches/README.md).

## Figures

| file                          | what it shows                                                                                                                                                                                                       | regenerate                                                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `scaling_curves.png`          | 4-panel scaling characterization: ① IR size O(P²) coupling-on vs O(P) off; ② the ~5× `rate_grad` multiplier; ③ forward-sim time ∝ IR bytes (parse-bound); ④ RSS ≈ 2.7× IR bytes                                     | `make bench-scaling` (`scripts/plot_scaling.py`)                                                                                        |
| `deser_load_before_after.png` | **Fix E** before/after: model load (`ir::from_str` + `CompiledModel::new`), derived `#[serde(untagged)]` vs hand-written single-pass `Deserialize` (2.5×–5.8×, grows with size)                                     | `make bench-micro` + `scripts/plot_deser_before_after.py`                                                                               |
| `d_reduce_ir_cliff.png`       | **Fix D** before/after: IR size, deep `BinOp(Add)` chain vs flat `Reduce` node — 1.3–2.3× smaller + the parse cliff removed (P>50 hit serde's recursion limit, were unparseable)                                    | `scripts/plot_d_reduce.py` (reads `d_reduce_ir_cliff.tsv`)                                                                              |
| `fix_b_before_after.png`      | **Fix B** before/after: 3-panel IR / peak RSS / `simulate` wall, inlined vs hoisted bindings, at the Kano anchor (P=44: 3.5× / 5.2× / 6.9×)                                                                         | `scripts/plot_scaling_before_after.py` (reads `scaling_before_b.tsv`, `scaling_after.tsv`)                                              |
| `flamegraph_real.svg`         | flamegraph of `simulate` on the ~2 GB anchor (P=44,A=21). **gitignored** (~4 MB). ⚠️ `make flamegraph-real` regenerates the **full-grad P=44,A=21** anchor — the ~15 GB-RSS OOM model; run only with memory headroom | `make flamegraph-real`                                                                                                                  |
| `pmmh_scale.png`              | **PMMH/IF2 scaling**: wall vs P at 100/400 particles — ~O(P²) in patches, linear in particles. Inputs to the national-scale roadmap                                                                                 | ad-hoc sweep (see the roadmap note); a committed `bench-inference-scale` is a TODO                                                      |
| `method_scale.png`            | **Cross-method bench**: IF2 vs PGAS wall vs P. PGAS is cheap per-sweep + clean O(P^1.5); IF2 number confounded by the refine stage. The measured basis for the PGAS-centric roadmap                                 | ad-hoc sweep (see [`../2026-05-29-inference-scaling-and-national-roadmap.md`](../2026-05-29-inference-scaling-and-national-roadmap.md)) |

## Data (raw, committed)

| file                                         | columns                                                                                                                                                                                        |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scaling.tsv`                                | macro sweep, **pre-E baseline** (older generator — not directly comparable to the current sweeps): `slice P A coupling grad n_compartments n_transitions ir_bytes compile_s sim_s peak_rss_mb` |
| `scaling_before_b.tsv` / `scaling_after.tsv` | same schema — the Fix B **before (E+D)** / **after (E+D+B)** sweeps on the _current_ generator (the clean same-generator comparison the figure uses)                                           |
| `d_reduce_ir_cliff.tsv`                      | Fix D IR-size sweep (before = Add-chain, after = Reduce; includes the P>50 cliff points)                                                                                                       |
| `deser_load_before_after.tsv`                | `model ir_mb before_us after_us` (criterion `load_parse_compile`; before = untagged, after = single-pass)                                                                                      |

## Live viewing

While a server is running
(`python3 -m http.server 8910 --directory <this dir>`), the figures are at
`http://thuja:8910/<file>` over Tailscale.
