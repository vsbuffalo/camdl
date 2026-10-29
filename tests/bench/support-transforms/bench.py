"""IF2 default-step benchmark per map family (support-keyed transforms §12.6, D8).

Run through the Makefile beside this file (`make grid`, `make far`, ...); this
script is its engine:
  python3 -I bench.py run <set>      runs every case of a set not yet done
  python3 -I bench.py list [set] | show <case id>

A case = (fixture, variant, family under test, z-step c, seed, condition).
Every step is written as an explicit natural-scale rw_sd chosen so the
branch's one-time conversion (series (i): at the box midpoint for the interval
maps, at the start for the half-bounded ones) gives exactly the intended step
in z. The families not under test take the provisional defaults (interval 2/3,
log-interval 1/5, half-bounded 1/5). Variants: br = feat/support-transforms at
877c517a (log-interval for a positive kind with 0 < lo); ie = the same tree with
the log-interval arm of Transform::for_domain disabled (Interval everywhere,
the D11 alternative); main = main at 0abe59f8 (its own automatic rw_sd,
converted at each particle), the reference.
"""
import math, os, subprocess, sys, time, json, re

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(S)))  # tests/bench/support-transforms
BIN = os.path.join(S, "bin"); FX = os.path.join(S, "fixtures")
RUNS = os.environ.get("BENCH_RUNS", os.path.join(S, "runs"))
WT = REPO
VARIANTS = {"br": (os.path.join(BIN, "camdl-br"), os.path.join(BIN, "camdlc-br")),
            "ie": (os.path.join(BIN, "camdl-ie"), os.path.join(BIN, "camdlc-br")),
            "main": (os.path.join(BIN, "camdl-main"), os.path.join(BIN, "camdlc-main"))}
# Where each fixture's model and data live: the recovery cases' models are used
# as they are; their data, and the Ebola-like fixture's, are committed here.
MODEL = {"sir": "tests/recovery/cases/sir/model.camdl",
         "seir_age": "tests/recovery/cases/seir_age/model.camdl"}
DATA_DIR = {"sir": "sir", "sir_0hi": "sir", "sir_wide": "sir", "sir_lb": "sir",
            "seir_age": "seir_age", "ebola": "ebola", "ebola_lb": "ebola"}
CANDS = ["2/3", "1/3", "1/5", "1/10", "1/20", "1/50"]
CANDS_EXT = ["1/100", "1/200"]
ALL = CANDS + CANDS_EXT
# Far starts: every chain at a corner of the box (or far up a half-line),
# away from the optimum, for the far-start / short-budget conditions.
FAR = {
 "sir_wide": {"lb": -7.5, "gamma": 0.45},
 "sir_0hi": {"beta": 0.9, "gamma": 0.45},
 "seir_age": {"beta": 0.4, "sigma": 0.02, "gamma": 0.9},
 "ebola": {"beta": 1.8, "sigma": 0.9, "cfr": 0.05},
 "sir_lb": {"beta": 4.0, "gamma": 2.0},
 "ebola_lb": {"beta": 3.0, "sigma": 2.0, "cfr": 0.05},
 "polio": {"R0_urban": 5.5, "R0_rural": 5.5, "kappa": 0.25, "gamma": 0.45, "delta": 0.25,
           "rho": 0.9, "lambda": 0.08},
}
DEFAULT = {"interval": 2 / 3, "logint": 0.2, "half": 0.2}
POS = ("rate", "positive", "count", "duration")

def family(variant, kind, lo, hi):
    if lo is None or hi is None: return "half"
    if variant == "br" and kind in POS and lo > 0: return "logint"
    return "interval"

def natural(fam, lo, hi, start, c):
    if fam == "interval": return c * (hi - lo) / 4
    if fam == "logint": return c * math.sqrt(lo * hi) * math.log(hi / lo) / 4
    return c * (start - (lo or 0.0))  # LowerBounded{lo}, converted at the start

IF2 = ('algorithm = "if2"\nbackend = "chain_binomial"\nchains = 8\nparticles = 2000\n'
       'iterations = 150\ncooling = 0.7\nstarts = "{starts}"\n')
SIR_FIXED = {"N0": 10000, "I0": 10, "rho": 0.6, "k": 10.0}
EBOLA_FIXED = {"gamma": 0.125, "N0": 50000, "I0": 10, "rho": 0.7, "k": 10.0}
WEEKLY = {"weekly_cases": "data/weekly_cases.tsv"}
EBOBS = {"weekly_onsets": "data/weekly_onsets.tsv", "weekly_deaths": "data/weekly_deaths.tsv"}
FIXTURES = {
 "sir": dict(obs=WEEKLY, fixed=SIR_FIXED, starts="lhs",
   est=[("beta", "rate", 0.05, 1.0, 0.25), ("gamma", "rate", 0.01, 0.5, 0.10)]),
 "sir_0hi": dict(obs=WEEKLY, fixed=SIR_FIXED, starts="lhs",
   est=[("beta", "rate", 0.05, 1.0, 0.25), ("gamma", "rate", 0.0, 0.5, 0.10)]),
 "sir_wide": dict(obs=WEEKLY, fixed=SIR_FIXED, starts="lhs",
   est=[("lb", "real", -8.0, 2.0, math.log(0.25)), ("gamma", "rate", 0.01, 0.5, 0.10)]),
 "seir_age": dict(obs={"cases_child": "data/cases_child.tsv", "cases_adult": "data/cases_adult.tsv"},
   fixed={"rho": 0.6, "k": 10.0}, starts="lhs",
   est=[("beta", "rate", 0.001, 0.5, 0.04), ("sigma", "rate", 0.01, 1.0, 0.25), ("gamma", "rate", 0.01, 1.0, 0.12)]),
 "ebola": dict(obs=EBOBS, fixed=EBOLA_FIXED, starts="lhs",
   est=[("beta", "rate", 0.05, 2.0, 0.5), ("sigma", "rate", 0.02, 1.0, 0.3), ("cfr", "probability", 0.0, 1.0, 0.4)]),
 "sir_lb": dict(obs=WEEKLY, fixed=SIR_FIXED, starts="single",
   est=[("beta", "rate", None, None, 0.25), ("gamma", "rate", None, None, 0.10)]),
 "ebola_lb": dict(obs=EBOBS, fixed=EBOLA_FIXED, starts="single",
   est=[("beta", "rate", None, None, 0.5), ("sigma", "rate", None, None, 0.3),
        ("cfr", "probability", 0.0, 1.0, 0.4)]),
}
POLIO_EST = [("R0_urban", "positive", 1.0, 6.0, 2.2), ("R0_rural", "positive", 1.0, 6.0, 3.0),
             ("kappa", "rate", 0.001, 0.3, 0.05), ("gamma", "rate", 0.02, 0.5, 0.1),
             ("delta", "rate", 0.01, 0.3, 0.05), ("rho", "probability", 0.1, 0.95, 0.40),
             ("lambda", "positive", 0.001, 0.1, 0.01)]
POLIO_FIXED = ("[fixed]\nsigma = 0.2\npara_frac = 0.0005\nomega = 0.003\nN0_urban = 500000\n"
               "N0_rural = 150000\nI0 = 20\nk = 10.0\nk_es = 10.0\n")
POLIO_METHOD = {
 "polio": ('algorithm = "if2"\nbackend = "chain_binomial"\nchains = 4\nparticles = 2000\n'
           'iterations = 50\ncooling = 0.5\nloglik_eval = { n_particles = 1000, n_replicates = 2 }\n'
           'dt_check = { enabled = false }\n'),
 "gh912": ('algorithm = "if2"\nbackend = "chain_binomial"\nchains = 2\nparticles = 50\n'
           'iterations = 3\ncooling = 0.5\nloglik_eval = { n_particles = 1000, n_replicates = 2 }\n'
           'dt_check = { enabled = false }\n'),
}

def cases():
    """(fixture, variant, family, candidate, seed, condition). condition "" is
    the base grid; "far30"/"far150" start every chain at FAR[fixture] with 30 or
    150 IF2 iterations; "r8" re-runs polio / gh912 with 8 clean-evaluation
    replicates."""
    out = []
    def add(fx, v, fam, seeds, cands=CANDS, cond=""):
        for c in cands:
            for s in seeds: out.append((fx, v, fam, c, s, cond))
    s3 = (1, 2, 3)
    add("sir", "br", "logint", s3); add("sir", "ie", "interval", s3)
    add("sir_0hi", "br", "interval", s3)
    add("sir_wide", "br", "interval", s3)
    add("seir_age", "br", "logint", (1, 2)); add("seir_age", "ie", "interval", (1, 2))
    add("ebola", "br", "logint", s3); add("ebola", "br", "interval", s3); add("ebola", "ie", "interval", s3)
    add("sir_lb", "br", "half", s3); add("ebola_lb", "br", "half", s3)
    add("polio", "br", "logint", (1,)); add("polio", "br", "interval", (1,))
    add("gh912", "br", "logint", range(1, 9)); add("gh912", "br", "interval", range(1, 9))
    for fx in ("sir", "sir_0hi", "sir_wide", "seir_age", "ebola"):
        for s in (s3 if fx != "seir_age" else (1, 2)):
            out.append((fx, "main", "-", "auto", s, ""))
    # ── extension (maintainer, T1) ──
    add("seir_age", "br", "logint", (1, 2), CANDS_EXT)
    add("ebola", "br", "logint", s3, CANDS_EXT)
    add("polio", "br", "logint", (1, 2), ALL, "r8")
    add("gh912", "br", "logint", range(1, 33), ["1/20", "1/50"], "r8")
    for cond in ("far30", "far150"):
        for fx, fam in (("sir_wide", "interval"), ("sir_0hi", "interval"), ("seir_age", "logint"),
                        ("ebola", "logint"), ("ebola", "interval"), ("sir_lb", "half"),
                        ("ebola_lb", "half")):
            add(fx, "br", fam, s3, ALL, cond)
    add("polio", "br", "logint", (1,), ALL, "far20")
    return out

def sets():
    """Named subsets, one per Makefile target."""
    c = cases()
    return {
        "grid": [x for x in c if x[5] == "" and x[3] not in CANDS_EXT],
        "grid-ext": [x for x in c if x[5] == "" and x[3] in CANDS_EXT],
        "polio-r8": [x for x in c if x[0] == "polio" and x[5] == "r8"],
        "gh912-32": [x for x in c if x[0] == "gh912" and x[5] == "r8"],
        "far": [x for x in c if x[5].startswith("far")],
    }

def case_id(c):
    fx, v, fam, cand, s, cond = c
    base = f"{fx}__{v}__{fam}__{cand.replace('/', '_')}__s{s}"
    return base + (f"__{cond}" if cond else "")

def est_lines(est, v, fam_tested, cand, far=None):
    lines = []
    for name, kind, lo, hi, start in est:
        if far: start = far[name]
        b = f"bounds = [{lo}, {hi}], " if lo is not None else ""
        if v == "main":
            lines.append(f"{name} = {{ {b}start = {start} }}"); continue
        fam = family(v, kind, lo, hi)
        c = eval(cand) if fam == fam_tested else DEFAULT[fam]
        lines.append(f"{name} = {{ {b}start = {start}, rw_sd = {natural(fam, lo, hi, start, c):.10g} }}")
    return lines

def fit_toml(c):
    fx, v, fam, cand, s, cond = c
    far = FAR.get(fx) if cond.startswith("far") else None
    its = int(cond[3:]) if far else None
    if fx in ("polio", "gh912"):
        fd = os.path.join(WT, "tests/fixtures/polio_afp_es")
        lines = ['output_dir = "out"', "[model]", 'camdl = "m.ir.json"', "[data.observations]",
                 f'afp = "{fd}/data/afp.tsv"', f'es = "{fd}/data/es.tsv"', "[estimate]"]
        lines += [l.replace(" }", ", prior = { flat = {} } }") for l in est_lines(POLIO_EST, v, fam, cand, far)]
        m = POLIO_METHOD[fx]
        if cond == "r8": m = m.replace("n_replicates = 2", "n_replicates = 8")
        if far: m = m.replace("n_replicates = 2", "n_replicates = 8").replace("iterations = 50", f"iterations = {its}") + 'starts = "single"\n'
        return "\n".join(lines) + "\n" + POLIO_FIXED + "[method]\n" + m
    f = FIXTURES[fx]
    lines = ['output_dir = "out"', "[model]", 'camdl = "model.camdl"', "[data.observations]"]
    lines += [f'{k} = "{p}"' for k, p in f["obs"].items()]
    lines.append("[estimate]"); lines += est_lines(f["est"], v, fam, cand, far)
    lines.append("[fixed]"); lines += [f"{k} = {val}" for k, val in f["fixed"].items()]
    method = IF2.format(starts="single" if far else f["starts"])
    if far: method = method.replace("iterations = 150", f"iterations = {its}")
    lines.append("[method]"); lines.append(method)
    return "\n".join(lines) + "\n"

def run(c):
    cid = case_id(c); d = os.path.join(RUNS, cid)
    if os.path.exists(os.path.join(d, "status.json")): return
    os.makedirs(d, exist_ok=True)
    fx, v, fam, cand, seed, cond = c
    camdl, camdlc = VARIANTS[v]
    if fx in ("polio", "gh912"):
        ir = subprocess.run([camdlc, os.path.join(WT, "tests/fixtures/polio_afp_es_2patch.camdl")],
                            capture_output=True, text=True, check=True).stdout
        open(os.path.join(d, "m.ir.json"), "w").write(ir)
    else:
        model = os.path.join(REPO, MODEL[fx]) if fx in MODEL else os.path.join(FX, fx, "model.camdl")
        open(os.path.join(d, "model.camdl"), "w").write(open(model).read())
        for p in FIXTURES[fx]["obs"].values():
            os.makedirs(os.path.join(d, os.path.dirname(p)), exist_ok=True)
            open(os.path.join(d, p), "w").write(open(os.path.join(FX, DATA_DIR[fx], p)).read())
    open(os.path.join(d, "fit.toml"), "w").write(fit_toml(c))
    env = dict(os.environ, CAMDL_SKIP_VERSION_CHECK="1", CAMDLC=camdlc)
    import resource
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    r = subprocess.run([camdl, "fit", "run", "fit.toml", "--seed", str(seed), "--progress", "none"],
                       cwd=d, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    open(os.path.join(d, "stderr.log"), "w").write(r.stderr)
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    json.dump({"exit": r.returncode, "wall": wall, "cpu": cpu}, open(os.path.join(d, "status.json"), "w"))
    print(f"{cid}: exit {r.returncode} in {wall:.0f}s", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "run":
        for c in sets()[sys.argv[2]]: run(c)
    elif sys.argv[1] == "list":
        for c in (sets()[sys.argv[2]] if len(sys.argv) > 2 else cases()): print(case_id(c))
    elif sys.argv[1] == "show":
        for c in cases():
            if case_id(c) == sys.argv[2]: print(fit_toml(c))
