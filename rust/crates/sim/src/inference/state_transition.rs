//! State-to-state transition density for the state-space trajectory kernel
//! (`docs/dev/notes/2026-09-02-state-pgbs-spike.md`).
//!
//! `p_θ(Z' | Z)` for `Z = (X, A)` — integer compartment counts plus the open
//! interval-stream accumulators — marginalized over the flow vectors
//! consistent with the edge:
//!
//! ```text
//!   p(Z' | Z) = Σ_{F ≥ 0 : S·F = ΔX, H·F = ΔA}  p(F | X, θ)
//! ```
//!
//! where `S` is the stoichiometry matrix and `H` maps flows to interval-stream
//! accumulators. The classification is COMPUTED from the compiled model, never
//! hand-derived: identical `[S; H]` columns are collapsed into flow classes
//! (their internal split marginalizes exactly — see
//! [`log_marginal_density_of_class_flows`]), the integer nullspace of the
//! class matrix gives the ambiguity directions, and the bounded lattice is
//! enumerated per edge.
//!
//! Correctness posture: each lattice term is scored by the EXISTING
//! [`log_transition_density_substep`], so this module cannot drift from
//! `step_one`'s Euler-multinomial semantics — the only new numerics are exact
//! rational linear algebra (done once per model) and one binomial/multinomial
//! split-probability correction per merged class (an exact identity of the
//! multinomial: marginalizing a subset's internal split given its total).
//!
//! Scope gate (the spike's prototype class): models with overdispersed
//! transitions, deterministic draws, events, balances, or interventions are
//! REFUSED loudly at analysis time. Feature parity with the innovation kernel
//! is the landing bar (nothing ships for one model); this gate is the
//! transition state of the committed arc, not an endpoint.

use std::collections::HashMap;

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::inference::multi_stream_obs::MultiStreamObsModel;
use crate::inference::pgas::log_transition_density_substep;

// ── Exact rational arithmetic (i128) for the one-time model analysis ────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frac {
    n: i128,
    d: i128, // always > 0
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

impl Frac {
    fn int(n: i128) -> Self {
        Frac { n, d: 1 }
    }
    fn is_zero(self) -> bool {
        self.n == 0
    }
    fn norm(n: i128, d: i128) -> Self {
        debug_assert!(d != 0);
        let s = if d < 0 { -1 } else { 1 };
        let g = gcd(n, d);
        Frac { n: s * n / g, d: s * d / g }
    }
    fn add(self, o: Frac) -> Frac {
        Frac::norm(self.n * o.d + o.n * self.d, self.d * o.d)
    }
    fn sub(self, o: Frac) -> Frac {
        Frac::norm(self.n * o.d - o.n * self.d, self.d * o.d)
    }
    fn mul(self, o: Frac) -> Frac {
        Frac::norm(self.n * o.n, self.d * o.d)
    }
    fn div(self, o: Frac) -> Frac {
        debug_assert!(o.n != 0);
        Frac::norm(self.n * o.d, self.d * o.n)
    }
}

// ── Analysis (once per model) ───────────────────────────────────────────────

/// The per-model classification behind the state-transition density: flow
/// classes, the reduced `[S; H]` system, and the integer nullspace basis.
pub struct StateTransitionAnalysis {
    n_tr: usize,
    n_comp: usize,
    n_streams: usize,
    /// class → member transitions, in transition-index order. The FIRST
    /// member is the canonical split target (see the density fn).
    class_members: Vec<Vec<usize>>,
    /// Row-transform of the RREF over the class matrix `M` (rows =
    /// n_comp + n_streams): applying `t[r]` to a stacked delta vector gives
    /// row `r`'s reduced right-hand side.
    t: Vec<Vec<Frac>>,
    pivot_col_of_row: Vec<usize>,
    /// Free (non-pivot) class columns — the lattice dimensions.
    free_cols: Vec<usize>,
    /// Integer nullspace basis, one vector (length n_class) per free column,
    /// with `+1` at its own free column.
    null_basis: Vec<Vec<i64>>,
}

/// Enumeration guard: a lattice this large means the analysis mis-classified
/// the model (or the model is outside what bounded enumeration should ever
/// serve) — refuse loudly rather than spin.
const MAX_LATTICE_TERMS: usize = 200_000;

impl StateTransitionAnalysis {
    /// Build the analysis, refusing models outside the prototype class.
    pub fn from_model(
        model: &CompiledModel,
        obs_model: &MultiStreamObsModel,
    ) -> Result<Self, SimError> {
        // ── Prototype class gate (loud, per the spike) ──
        for (i, tr) in model.model.transitions.iter().enumerate() {
            match tr.draw_method {
                ir::transition::DrawMethod::Overdispersed { .. } => {
                    return Err(SimError::Validation(format!(
                        "state-space trajectory kernel (prototype) does not yet \
                         support overdispersed transitions: `{}` (index {i}). \
                         Gamma marginalization is deliberately deferred until the \
                         fixed-θ experiment reads out — see \
                         docs/dev/notes/2026-09-02-state-pgbs-spike.md.",
                        tr.name
                    )));
                }
                ir::transition::DrawMethod::Deterministic => {
                    return Err(SimError::Validation(format!(
                        "state-space trajectory kernel (prototype) does not yet \
                         support deterministic transitions: `{}` (index {i}).",
                        tr.name
                    )));
                }
                ir::transition::DrawMethod::Poisson => {}
            }
        }
        if !model.model.interventions.is_empty() {
            return Err(SimError::Validation(
                "state-space trajectory kernel (prototype) does not yet support \
                 interventions"
                    .into(),
            ));
        }
        if model.balance.is_some() {
            return Err(SimError::Validation(
                "state-space trajectory kernel (prototype) does not yet support \
                 balance {}"
                    .into(),
            ));
        }

        let n_tr = model.model.transitions.len();
        let n_comp = model.int_local_to_global.len();
        let streams = obs_model.incidence_streams();
        let n_streams = streams.len();
        let n_rows = n_comp + n_streams;

        // ── Full [S; H] columns per transition ──
        let mut cols: Vec<Vec<i64>> = vec![vec![0; n_rows]; n_tr];
        for (j, stoich) in model.transition_stoich.iter().enumerate() {
            for &(local, delta) in stoich {
                cols[j][local] += delta;
            }
        }
        for (k, (_, idxs)) in streams.iter().enumerate() {
            for &j in idxs {
                cols[j][n_comp + k] += 1;
            }
        }

        // ── Collapse identical columns into flow classes ──
        let mut class_members: Vec<Vec<usize>> = Vec::new();
        let mut seen: HashMap<Vec<i64>, usize> = HashMap::new();
        for j in 0..n_tr {
            let id = *seen.entry(cols[j].clone()).or_insert_with(|| {
                class_members.push(Vec::new());
                class_members.len() - 1
            });
            class_members[id].push(j);
        }
        let n_class = class_members.len();

        // ── RREF of the class matrix over exact rationals ──
        let mut m: Vec<Vec<Frac>> = (0..n_rows)
            .map(|r| {
                (0..n_class)
                    .map(|c| Frac::int(cols[class_members[c][0]][r] as i128))
                    .collect()
            })
            .collect();
        // Row-transform accumulator: t = I, updated alongside m.
        let mut t: Vec<Vec<Frac>> = (0..n_rows)
            .map(|r| (0..n_rows).map(|c| Frac::int((r == c) as i128)).collect())
            .collect();

        let mut pivot_col_of_row = Vec::new();
        let mut rank = 0usize;
        for c in 0..n_class {
            let Some(piv) = (rank..n_rows).find(|&r| !m[r][c].is_zero()) else {
                continue;
            };
            m.swap(rank, piv);
            t.swap(rank, piv);
            let inv = m[rank][c];
            for x in m[rank].iter_mut() {
                *x = x.div(inv);
            }
            for x in t[rank].iter_mut() {
                *x = x.div(inv);
            }
            for r in 0..n_rows {
                if r != rank && !m[r][c].is_zero() {
                    let f = m[r][c];
                    for cc in 0..n_class {
                        m[r][cc] = m[r][cc].sub(f.mul(m[rank][cc]));
                    }
                    for cc in 0..n_rows {
                        t[r][cc] = t[r][cc].sub(f.mul(t[rank][cc]));
                    }
                }
            }
            pivot_col_of_row.push(c);
            rank += 1;
            if rank == n_rows {
                break;
            }
        }

        let free_cols: Vec<usize> =
            (0..n_class).filter(|c| !pivot_col_of_row.contains(c)).collect();

        // ── Integer nullspace basis (scaled to integers) ──
        let mut null_basis = Vec::with_capacity(free_cols.len());
        for &f in &free_cols {
            let mut v: Vec<Frac> = vec![Frac::int(0); n_class];
            v[f] = Frac::int(1);
            for (r, &pc) in pivot_col_of_row.iter().enumerate() {
                v[pc] = Frac::int(0).sub(m[r][f]);
            }
            let lcm = v.iter().fold(1i128, |acc, x| acc / gcd(acc, x.d) * x.d);
            let iv: Vec<i64> = v
                .iter()
                .map(|x| {
                    let scaled = x.n * (lcm / x.d);
                    i64::try_from(scaled).expect("nullspace entry fits i64")
                })
                .collect();
            null_basis.push(iv);
        }

        Ok(StateTransitionAnalysis {
            n_tr,
            n_comp,
            n_streams,
            class_members,
            t,
            pivot_col_of_row,
            free_cols,
            null_basis,
        })
    }

    /// Number of lattice (ambiguity) dimensions.
    pub fn n_free_dims(&self) -> usize {
        self.free_cols.len()
    }

    /// Class ids with more than one member transition (collapsed columns).
    pub fn merged_classes(&self) -> Vec<usize> {
        (0..self.class_members.len())
            .filter(|&c| self.class_members[c].len() > 1)
            .collect()
    }

    /// Solve for class flows given the stacked edge delta `[ΔX; ΔA]`:
    /// the particular solution (free vars = 0) and a consistency verdict.
    /// `None` = the edge is unreachable in one substep (density zero).
    fn particular_solution(&self, delta: &[i64]) -> Option<Vec<i64>> {
        debug_assert_eq!(delta.len(), self.n_comp + self.n_streams);
        let n_rows = self.n_comp + self.n_streams;
        let n_class = self.class_members.len();
        let rank = self.pivot_col_of_row.len();

        // Reduced RHS: rhs[r] = (T · delta)[r].
        let rhs = |r: usize| -> Frac {
            let mut acc = Frac::int(0);
            for c in 0..n_rows {
                if !self.t[r][c].is_zero() && delta[c] != 0 {
                    acc = acc.add(self.t[r][c].mul(Frac::int(delta[c] as i128)));
                }
            }
            acc
        };
        // Consistency: zero rows of the RREF must have zero reduced RHS.
        for r in rank..n_rows {
            if !rhs(r).is_zero() {
                return None;
            }
        }
        let mut f = vec![0i64; n_class];
        for (r, &pc) in self.pivot_col_of_row.iter().enumerate() {
            let v = rhs(r);
            if v.d != 1 {
                return None; // non-integer flow ⇒ unreachable edge
            }
            f[pc] = i64::try_from(v.n).ok()?;
        }
        Some(f)
    }

    /// Enumerate the feasible lattice: every non-negative integer class-flow
    /// vector consistent with `delta`, as offsets `m` applied to the
    /// particular solution along the nullspace basis. Calls `visit` with the
    /// full class-flow vector for each point. Returns the number of points,
    /// or an error if the lattice exceeds [`MAX_LATTICE_TERMS`].
    fn enumerate_lattice(
        &self,
        particular: &[i64],
        mut visit: impl FnMut(&[i64]),
    ) -> Result<usize, SimError> {
        let k = self.null_basis.len();
        let n_class = self.class_members.len();
        if k == 0 {
            if particular.iter().all(|&x| x >= 0) {
                visit(particular);
                return Ok(1);
            }
            return Ok(0);
        }

        // Conservative per-dimension bounds by fixpoint interval tightening on
        // the constraints  particular[c] + Σ_i m_i · basis_i[c] ≥ 0.
        let big = particular.iter().map(|&x| x.abs()).sum::<i64>().max(1) * 2 + 2;
        let mut lo = vec![-big; k];
        let mut hi = vec![big; k];
        for _ in 0..(4 * k + 4) {
            let mut moved = false;
            for c in 0..n_class {
                for i in 0..k {
                    let a_i = self.null_basis[i][c];
                    if a_i == 0 {
                        continue;
                    }
                    // worst-case contribution of the other dims
                    let mut rest = particular[c] as i128;
                    for j in 0..k {
                        if j == i {
                            continue;
                        }
                        let a = self.null_basis[j][c] as i128;
                        rest += if a >= 0 { a * hi[j] as i128 } else { a * lo[j] as i128 };
                    }
                    // a_i·m_i ≥ −rest
                    if a_i > 0 {
                        let bound = (-rest).div_euclid(a_i as i128) as i64
                            + i64::from((-rest).rem_euclid(a_i as i128) != 0);
                        if bound > lo[i] {
                            lo[i] = bound;
                            moved = true;
                        }
                    } else {
                        let bound = rest.div_euclid((-a_i) as i128) as i64;
                        if bound < hi[i] {
                            hi[i] = bound;
                            moved = true;
                        }
                    }
                }
            }
            if !moved {
                break;
            }
        }
        for i in 0..k {
            if lo[i] > hi[i] {
                return Ok(0);
            }
        }
        let box_size: i128 = (0..k).map(|i| (hi[i] - lo[i] + 1) as i128).product();
        if box_size > MAX_LATTICE_TERMS as i128 {
            return Err(SimError::Validation(format!(
                "state-transition lattice box has {box_size} points \
                 (> {MAX_LATTICE_TERMS}); this model/edge is outside what bounded \
                 enumeration should serve — see the spike note's production DP path"
            )));
        }

        // Nested walk over the box with a final exact feasibility check.
        let mut n_points = 0usize;
        let mut m = lo.clone();
        let mut flows = vec![0i64; n_class];
        'outer: loop {
            let mut ok = true;
            for c in 0..n_class {
                let mut v = particular[c];
                for i in 0..k {
                    v += m[i] * self.null_basis[i][c];
                }
                if v < 0 {
                    ok = false;
                    break;
                }
                flows[c] = v;
            }
            if ok {
                visit(&flows);
                n_points += 1;
            }
            // increment odometer
            for i in 0..k {
                m[i] += 1;
                if m[i] <= hi[i] {
                    continue 'outer;
                }
                m[i] = lo[i];
            }
            break;
        }
        Ok(n_points)
    }

    /// Expand class flows to a per-transition flow vector under the canonical
    /// split: a merged class's total is assigned entirely to its FIRST member.
    fn expand_canonical(&self, class_flows: &[i64], out: &mut [u64]) {
        out.iter_mut().for_each(|x| *x = 0);
        for (c, members) in self.class_members.iter().enumerate() {
            out[members[0]] = class_flows[c] as u64;
        }
    }
}

// ── The density ─────────────────────────────────────────────────────────────

/// `log p_θ(Z' | Z)` for one substep: log-sum-exp over the feasible flow
/// lattice of the innovation-conditional density, with merged-class splits
/// marginalized exactly.
///
/// `d_counts` = X' − X (length n_comp); `d_acc` = the accumulator deltas this
/// substep contributes (length n_interval_streams) — the CALLER owns the
/// reset convention (spike note: `A_{t+1} = (due_reset(t) ? 0 : A_t) + H·F`).
///
/// Each lattice term is scored by [`log_transition_density_substep`] on the
/// canonical split, then corrected by the exact multinomial identity: the
/// marginal over a merged class's internal split, given its total `n` and the
/// canonical all-to-first assignment, is the full density minus
/// `log P(split = canonical | total)` = `n · log(r_first / r_class)`.
#[allow(clippy::too_many_arguments)]
pub fn log_state_transition_density(
    model: &CompiledModel,
    analysis: &StateTransitionAnalysis,
    counts_before: &[i64],
    d_counts: &[i64],
    d_acc: &[i64],
    params: &[f64],
    t: f64,
    dt: f64,
    per_eval: Option<&[f64]>,
) -> Result<f64, SimError> {
    debug_assert_eq!(d_counts.len(), analysis.n_comp);
    debug_assert_eq!(d_acc.len(), analysis.n_streams);

    let mut delta = Vec::with_capacity(analysis.n_comp + analysis.n_streams);
    delta.extend_from_slice(d_counts);
    delta.extend_from_slice(d_acc);

    let Some(particular) = analysis.particular_solution(&delta) else {
        return Ok(f64::NEG_INFINITY);
    };

    // Rates for the canonical-split correction (class rate = Σ member rates).
    // Evaluated once per edge; `eval_propensities` needs an IntState.
    let mut int_s = crate::state::IntState::new(analysis.n_comp);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = crate::state::RealState::new(model.real_local_to_global.len());
    let mut propensities = vec![0.0; analysis.n_tr];
    crate::propensity::eval_propensities(
        model, &int_s, &real_s, params, t, dt, per_eval, &mut propensities,
    )?;
    let merged = analysis.merged_classes();

    let mut terms: Vec<f64> = Vec::new();
    let mut flows = vec![0u64; analysis.n_tr];
    let mut inner_err: Option<SimError> = None;
    analysis.enumerate_lattice(&particular, |class_flows| {
        if inner_err.is_some() {
            return;
        }
        analysis.expand_canonical(class_flows, &mut flows);
        let td = match log_transition_density_substep(
            model, counts_before, &flows, &[], params, t, dt, per_eval,
        ) {
            Ok(v) => v,
            Err(e) => {
                inner_err = Some(e);
                return;
            }
        };
        if td == f64::NEG_INFINITY {
            return;
        }
        let mut lp = td;
        for &c in &merged {
            let n = class_flows[c];
            if n == 0 {
                continue;
            }
            let members = &analysis.class_members[c];
            let r_first = propensities[members[0]];
            let r_class: f64 = members.iter().map(|&j| propensities[j]).sum();
            if r_class <= 0.0 {
                // Positive total flow with zero class rate: the canonical term
                // was −inf already, so we cannot be here; guard anyway.
                return;
            }
            // marginal = full − log P(canonical split | total)
            //          = full − n·log(r_first / r_class)
            lp -= n as f64 * (r_first / r_class).ln();
        }
        terms.push(lp);
    })?;
    if let Some(e) = inner_err {
        return Err(e);
    }
    if terms.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }
    let mx = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(mx + terms.iter().map(|v| (v - mx).exp()).sum::<f64>().ln())
}

/// Draw a per-transition flow vector from the lattice-restricted conditional
/// `p(F | Z, Z', θ)` — the reconstruction seam's per-edge draw. Two stages,
/// both exact: choose the lattice point ∝ its (split-marginalized) density,
/// then draw each merged class's internal split from its multinomial
/// conditional given the class total.
#[allow(clippy::too_many_arguments)]
pub fn sample_edge_flows(
    model: &CompiledModel,
    analysis: &StateTransitionAnalysis,
    counts_before: &[i64],
    d_counts: &[i64],
    d_acc: &[i64],
    params: &[f64],
    t: f64,
    dt: f64,
    per_eval: Option<&[f64]>,
    rng: &mut crate::rng::StatefulRng,
) -> Result<Option<Vec<u64>>, SimError> {
    let mut delta = Vec::with_capacity(analysis.n_comp + analysis.n_streams);
    delta.extend_from_slice(d_counts);
    delta.extend_from_slice(d_acc);
    let Some(particular) = analysis.particular_solution(&delta) else {
        return Ok(None);
    };

    let mut int_s = crate::state::IntState::new(analysis.n_comp);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = crate::state::RealState::new(model.real_local_to_global.len());
    let mut propensities = vec![0.0; analysis.n_tr];
    crate::propensity::eval_propensities(
        model, &int_s, &real_s, params, t, dt, per_eval, &mut propensities,
    )?;
    let merged = analysis.merged_classes();

    // Stage 1: categorical over lattice points.
    let mut points: Vec<Vec<i64>> = Vec::new();
    let mut logw: Vec<f64> = Vec::new();
    let mut flows = vec![0u64; analysis.n_tr];
    let mut inner_err: Option<SimError> = None;
    analysis.enumerate_lattice(&particular, |class_flows| {
        if inner_err.is_some() {
            return;
        }
        analysis.expand_canonical(class_flows, &mut flows);
        match log_transition_density_substep(
            model, counts_before, &flows, &[], params, t, dt, per_eval,
        ) {
            Ok(td) if td > f64::NEG_INFINITY => {
                let mut lp = td;
                for &c in &merged {
                    let n = class_flows[c];
                    if n == 0 {
                        continue;
                    }
                    let members = &analysis.class_members[c];
                    let r_first = propensities[members[0]];
                    let r_class: f64 = members.iter().map(|&j| propensities[j]).sum();
                    if r_class <= 0.0 {
                        return;
                    }
                    lp -= n as f64 * (r_first / r_class).ln();
                }
                points.push(class_flows.to_vec());
                logw.push(lp);
            }
            Ok(_) => {}
            Err(e) => inner_err = Some(e),
        }
    })?;
    if let Some(e) = inner_err {
        return Err(e);
    }
    if points.is_empty() {
        return Ok(None);
    }
    let chosen = match crate::inference::pgas::sample_categorical_log(&logw, rng) {
        Some(i) => &points[i],
        None => return Ok(None),
    };

    // Stage 2: internal splits of merged classes ~ multinomial(total, r_i/r_class),
    // drawn as sequential conditional binomials (matching step_one's convention).
    let mut out = vec![0u64; analysis.n_tr];
    for (c, members) in analysis.class_members.iter().enumerate() {
        let total = chosen[c] as u64;
        if members.len() == 1 {
            out[members[0]] = total;
            continue;
        }
        let mut remaining = total;
        let mut rate_remaining: f64 = members.iter().map(|&j| propensities[j]).sum();
        for (k, &j) in members.iter().enumerate() {
            if k == members.len() - 1 {
                out[j] = remaining;
            } else if remaining > 0 && rate_remaining > 0.0 {
                let p = (propensities[j] / rate_remaining).clamp(0.0, 1.0);
                let draw = rng.binomial(remaining, p);
                out[j] = draw;
                remaining -= draw;
                rate_remaining -= propensities[j];
            } else {
                out[j] = 0;
            }
        }
    }
    Ok(Some(out))
}
