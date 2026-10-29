"""Summarise runs/* (made by the Makefile targets) into
  results.tsv  one row per run;
  summary.tsv  one row per fixture x variant x family x condition x candidate.
The final log-likelihood is the clean evaluation in chain_evaluations.tsv: a
fresh particle filter at each chain's final estimate, several replicates
(8 x 4000 particles; polio/gh912 "" 2 x 1000, "r8"/"far20" 8 x 1000), not IF2's
running value. A chain reached the basin when its clean log-likelihood is within
BASIN nats of the best clean value any run on that fixture's data reached.
usage: python3 -I table.py"""
import glob, json, math, os, statistics
S = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(S, "bench.py")).read().split("if __name__")[0]
g = {"__file__": os.path.join(S, "bench.py")}; exec(src, g)
cases, case_id, RUNS, ALL = g["cases"], g["case_id"], g["RUNS"], g["ALL"]
BASIN = 3.0
DATA = {"sir": "sir", "sir_0hi": "sir", "sir_wide": "sir", "sir_lb": "sir", "ebola": "ebola",
        "ebola_lb": "ebola", "seir_age": "seir_age", "polio": "polio", "gh912": "polio"}

def read_tsv(p):
    rows = [l.rstrip("\n").split("\t") for l in open(p) if not l.startswith("#")]
    return rows[0], rows[1:]

runs = []
for c in cases():
    fx, v, fam, cand, seed, cond = c
    d = os.path.join(RUNS, case_id(c)); st = os.path.join(d, "status.json")
    if not os.path.exists(st): continue
    status = json.load(open(st))
    leafs = glob.glob(os.path.join(d, "out/fits/*/*/seed_*"))
    ce = os.path.join(leafs[0], "chain_evaluations.tsv") if leafs else ""
    chains = []
    if ce and os.path.exists(ce):
        h, rr = read_tsv(ce)
        chains = [(float(r[h.index("loglik")]), float(r[h.index("se")])) for r in rr]
    runs.append(dict(c=c, exit=status["exit"], wall=status["wall"], cpu=status.get("cpu"), chains=chains))

best = {}
for r in runs:
    for ll, _ in r["chains"]:
        if math.isfinite(ll):
            k = DATA[r["c"][0]]; best[k] = max(best.get(k, -math.inf), ll)

with open(os.path.join(S, "results.tsv"), "w") as f:
    f.write("fixture\tvariant\tfamily\tc\tseed\tcondition\texit\twall_s\tcpu_s\tll_best\tll_mean\tn_chains\tn_inf\tin_basin\n")
    for r in runs:
        fx, v, fam, cand, seed, cond = r["c"]
        fin = [ll for ll, _ in r["chains"] if math.isfinite(ll)]
        inb = sum(ll >= best[DATA[fx]] - BASIN for ll in fin)
        f.write("\t".join(map(str, [fx, v, fam, cand, seed, cond, r["exit"], round(r["wall"], 1),
            round(r["cpu"], 1) if r["cpu"] is not None else "", round(max(fin), 2) if fin else "",
            round(statistics.mean(fin), 2) if fin else "", len(r["chains"]),
            len(r["chains"]) - len(fin), inb])) + "\n")

groups = {}
for r in runs:
    fx, v, fam, cand, seed, cond = r["c"]
    groups.setdefault((fx, v, fam, cond), {}).setdefault(cand, []).append(r)
out = []
for (fx, v, fam, cond), byc in sorted(groups.items()):
    stats = {}
    for cand, rs in byc.items():
        bests, pf, nch, inb, walls = [], [], 0, 0, []
        for r in rs:
            fin = [(ll, se) for ll, se in r["chains"] if math.isfinite(ll)]
            reps = 8 if (fx not in ("polio", "gh912") or cond in ("r8", "far20")) else 2
            nch += max(len(r["chains"]), 1)
            inb += sum(ll >= best[DATA[fx]] - BASIN for ll, _ in fin)
            if fin:
                bests.append(max(ll for ll, _ in fin))
                pf += [se * math.sqrt(reps) for _, se in fin if math.isfinite(se)]
            walls.append(r["wall"])
        stats[cand] = dict(n=len(rs), failed=len(rs) - len(bests),
            mean=statistics.mean(bests) if bests else -math.inf,
            sd=statistics.stdev(bests) if len(bests) > 1 else float("nan"),
            pf=statistics.mean(pf) if pf else float("nan"),
            basin=inb / nch, wall=statistics.mean(walls))
    top = max(s["mean"] for s in stats.values())
    for cand in [c for c in ALL + ["auto"] if c in stats]:
        s = stats[cand]; gap = top - s["mean"]
        out.append([fx, v, fam, cond or "base", cand, s["n"], s["failed"], round(s["mean"], 2),
                    round(s["sd"], 2), round(gap, 2), round(s["pf"], 3), round(s["basin"], 2),
                    round(s["wall"], 1), "yes" if gap <= 2 * s["pf"] else "no"])
with open(os.path.join(S, "summary.tsv"), "w") as f:
    f.write("fixture\tvariant\tfamily\tcondition\tc\tseeds\tfailed_seeds\tmean_best\tsd_best_over_seeds\t"
            "gap_to_top\tpf_sd\tfrac_chains_in_basin\tmean_wall_s\tserves\n")
    for r in out: f.write("\t".join(map(str, r)) + "\n")
print(open(os.path.join(S, "summary.tsv")).read())
