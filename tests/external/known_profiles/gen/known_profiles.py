"""Write the K1 and K2 known-profile fixtures: data, fit configs, exact answers.

K1 and K2 are the known-profile oracles of the parameter-roles proposal
(docs/dev/proposals/2026-10-26-parameter-roles.md §11.1). Each model has one
compartment `X` with no transitions that holds 1, and the parameters enter the
observation directly, so camdl's likelihood under the ODE backend is an iid
sample's:

- K1: integer y_t scored by camdl's `normal(mean = mu, sd = s)`, which is the
  discretized normal P(y) = Phi((y + 0.5 - mu)/s) - Phi((y - 0.5 - mu)/s) for
  y > 0 (sim/src/inference/obs_loglik.rs, `discretized_normal_logpmf_tol`;
  every y here is positive). Holding s = s0 the curve in `mu` is that sum in
  closed form; re-maximizing `s` has no closed form, so the profile is
  computed by a one-dimensional golden-section search to 1e-13 in log s,
  inside the search box of `s` written into fits/k1.toml.
- K2: y_t ~ poisson(rate = rho * X0). The likelihood depends on rho * X0
  only. Profiling `rho` with `X0` re-maximized over its search box [lo, hi]
  gives the saturated maximum sum[y log ybar - ybar - log y!] wherever
  ybar / rho lies in the box, and the Poisson curve at the box edge outside
  it; holding X0 = X0_0 gives sum[y log(rho X0_0) - rho X0_0 - log y!].

camdl does not have a continuous normal likelihood and rounds a compartment's
initial value to an integer, which is why K1's data are integers and why `mu`
and `X0` multiply the projection instead of initialising `X`.

Everything is computed here, from the data this script writes, with nothing
from camdl. Deterministic: rerunning reproduces every file byte for byte.

    python3 gen/known_profiles.py        # from tests/external/known_profiles/
"""

import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 20
TIMES = list(range(1, N + 1))


def lin(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def fmt(x: float) -> str:
    return repr(round(x, 12))


def write_tsv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    lines = ["\t".join(header)] + ["\t".join(fmt(v) if isinstance(v, float) else str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's multiplication method: exact for the small rates used here."""
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def dnorm_logpmf(y: int, mu: float, sd: float) -> float:
    """camdl's discretized normal for y > 0, in the same erfc-stable form."""
    assert y > 0
    z_lo, z_hi = (y - 0.5 - mu) / sd, (y + 0.5 - mu) / sd
    r2 = math.sqrt(2.0)
    if z_lo + z_hi >= 0.0:
        p = 0.5 * (math.erfc(z_lo / r2) - math.erfc(z_hi / r2))
    else:
        p = 0.5 * (math.erfc(-z_hi / r2) - math.erfc(-z_lo / r2))
    return math.log(p)


def golden_max(f, lo: float, hi: float, tol: float = 1e-13) -> float:
    """The maximizer of a unimodal `f` on [lo, hi]."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


def k1() -> None:
    rng = random.Random(20261027)
    y = [max(1, round(10.0 + 2.0 * rng.gauss(0.0, 1.0))) for _ in TIMES]
    write_tsv(ROOT / "data" / "k1_y.tsv", ["time", "y"], [[t, v] for t, v in zip(TIMES, y)])
    grid = lin(8.5, 11.5, 13)
    s0 = 2.0

    def loglik(mu: float, sd: float) -> float:
        return sum(dnorm_logpmf(v, mu, sd) for v in y)

    # The search box of `s`: [0.5 min, 2 max] of the continuous sigma_hat over
    # the grid, which contains the discretized maximizer at every grid point.
    s_cont = [math.sqrt(sum((v - mu) ** 2 for v in y) / len(y)) for mu in grid]
    lo = math.floor(0.5 * min(s_cont) * 1e4) / 1e4
    hi = math.ceil(2.0 * max(s_cont) * 1e4) / 1e4
    rows = []
    for mu in grid:
        log_s = golden_max(lambda ls: loglik(mu, math.exp(ls)), math.log(lo), math.log(hi))
        s_hat = math.exp(log_s)
        assert lo * 1.01 < s_hat < hi / 1.01, f"s_hat {s_hat} at the box edge for mu = {mu}"
        rows.append([mu, s_hat, loglik(mu, s_hat), loglik(mu, s0)])
    write_tsv(ROOT / "ref" / "k1_profile.tsv", ["mu", "s_hat", "loglik_profiled", "loglik_held_s_2"], rows)
    (ROOT / "fits" / "k1.toml").write_text(f"""# K1: profile `mu` (lin(8.5, 11.5, 13)) with `s` estimated, against
# ref/k1_profile.tsv. The box of `s` contains its maximizer at every grid
# point; written by gen/known_profiles.py.
[model]
camdl = "../models/k1.camdl"

[data.observations]
y = "../data/k1_y.tsv"

[estimate]
mu = {{ bounds = [8.0, 12.0], start = 10.0 }}
s  = {{ bounds = [{lo}, {hi}], start = 2.0 }}

[fixed]

[method]
algorithm = "nl-bobyqa"
backend = "ode"
chains = 3
tolerance = 1e-10
""")


def k2() -> None:
    rng = random.Random(20261028)
    y = [poisson(rng, 20.0) for _ in TIMES]
    write_tsv(ROOT / "data" / "k2_y.tsv", ["time", "y"], [[t, v] for t, v in zip(TIMES, y)])
    n = len(y)
    ybar = sum(y) / n
    log_fact = sum(math.lgamma(v + 1) for v in y)
    lo, hi = 25.0, 60.0
    x0_held = 40.0

    def loglik(lam: float) -> float:
        return sum(y) * math.log(lam) - n * lam - log_fact

    saturated = loglik(ybar)
    rows = []
    for rho in lin(0.2, 0.95, 16):
        x0_hat = min(max(ybar / rho, lo), hi)
        rows.append([rho, x0_hat, loglik(rho * x0_hat), loglik(rho * x0_held), saturated])
    write_tsv(ROOT / "ref" / "k2_profile.tsv",
              ["rho", "x0_hat", "loglik_profiled", "loglik_held_x0_40", "loglik_saturated"], rows)
    (ROOT / "fits" / "k2.toml").write_text(f"""# K2: profile `rho` (lin(0.2, 0.95, 16)) with `X0` estimated over
# [{lo}, {hi}], against ref/k2_profile.tsv. Flat at the saturated maximum where
# ybar / rho lies in the box ({ybar / hi:.4f} <= rho <= {ybar / lo:.4f}), the
# Poisson curve at the box edge outside it.
[model]
camdl = "../models/k2.camdl"

[data.observations]
y = "../data/k2_y.tsv"

[estimate]
rho = {{ bounds = [0.1, 1.0], start = 0.5 }}
X0  = {{ bounds = [{lo}, {hi}], start = 40.0 }}

[fixed]

[method]
algorithm = "nl-bobyqa"
backend = "ode"
chains = 3
tolerance = 1e-10
""")


if __name__ == "__main__":
    k1()
    k2()
