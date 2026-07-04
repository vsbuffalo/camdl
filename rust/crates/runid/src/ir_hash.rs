//! Hand-written `ContentAddressed` impls for the `ir` type tree reachable
//! from [`ir::Model`].
//!
//! Why hand-written, not derived: `ir::Model` is a *foreign* type tree with
//! ~60 nested foreign types pervaded by `HashMap<String, f64>`,
//! `HashMap<String, Expr>` (`rate_grad`), and raw `f64`. You cannot
//! `#[derive]` on types you don't own, and hashing the IR via serde bytes
//! would be unsound (NaN → `null`, `HashMap` order not guaranteed sorted).
//! The orphan rule allows these impls (local trait, foreign types).
//!
//! Two policies applied throughout:
//!
//! - **Structural IR floats.** Every raw `f64` is hashed via
//!   [`CanonicalHasher::write_f64_bits`] (raw `to_bits`), keeping `±0.0`
//!   and NaN payloads distinct — matching the IR's own `ConstExpr::PartialEq`
//!   (`expr.rs`), which compares `to_bits()` precisely so two ASTs differing
//!   only in zero sign or NaN payload are observably distinct. Routing IR
//!   floats through `FiniteF64` would erase that distinction *and* reject
//!   NaN-bearing consts at hash time (a totality break).
//! - **Sorted maps.** Every `HashMap`/`BTreeMap` is hashed in sorted key
//!   order (`rate_grad` is a `HashMap` inside the IR).
//!
//! `Expr` is a `Box`-recursive tree; `eval_expr` already recurses it plainly
//! and production handles multi-GB IRs without overflow, so a recursive
//! `hash_into` is exactly as safe as the engine itself.

use ir::deriv::{DerivEntry, UnsupportedReason};
use ir::expr::{BinOp, Expr, UnOp};
use ir::intervention::{
    Action, CmpOp, FireSource, Intervention, InterventionKind,
    InterventionSchedule, ObsReducer, ReactiveTrigger, RecurringSchedule, TriggerExpr,
    TriggerQuantity, TriggerThreshold,
};
use ir::model::{
    BalanceSpec, Binding, Compartment, CompartmentKind, Dimension, InitialConditions, Model,
    ModelStructure, OutputConfig, OutputSchedule, Preset, RegularOutputSchedule, SimulationConfig,
};
use ir::observation::{
    ColumnRole, Likelihood, ObsColumn, ObservationModel, ObservationSchedule, Projection,
    RegularSchedule, StratumKey,
};
use ir::ode_equation::OdeEquation;
use ir::parameter::{
    HierarchicalKind, HierarchicalPrior, ParamKind, ParamValue, Parameter, PriorDist, PriorSpec,
    Transform,
};
use ir::table::{OobPolicy, Table, TableSource};
use ir::time_func::{TimeFuncKind, TimeFunction};
use ir::transition::{
    DrawMethod, StoichiometryEntry, Transition, TransitionLineage, TransitionMetadata,
};

use crate::hash::{CanonicalHasher, ContentAddressed};

/// Schema version for the `ir` tree's encoding. A change to *how* the IR is
/// hashed (a field's policy) bumps this; bumping it re-keys IR-derived
/// identities. (The IR's own *content* version is `ir_version`; this is the
/// hashing policy version, which is independent.)
const SV: u16 = 1;

/// Write the per-type domain-separation header: tag + schema version.
fn header(h: &mut CanonicalHasher, tag: &str) {
    h.write_type_tag(tag);
    h.write_schema_version(SV);
}

/// Hash an `Option<f64>` under the structural-float policy (Option is
/// generic over `ContentAddressed`, but `f64` deliberately is not, so the
/// f64-valued options are spelled out here).
fn hash_opt_f64(h: &mut CanonicalHasher, x: &Option<f64>) {
    match x {
        None => h.write_u8(0),
        Some(v) => {
            h.write_u8(1);
            h.write_f64_bits(*v);
        }
    }
}

// ── deriv.rs ─────────────────────────────────────────────────────────────────

impl ContentAddressed for UnsupportedReason {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::deriv::UnsupportedReason");
        // Positional discriminant; variants append at the end (see the type).
        let idx: u32 = match self {
            UnsupportedReason::Lag => 0,
            UnsupportedReason::PeriodicCoeff => 1,
            UnsupportedReason::StructuralForcing => 2,
            UnsupportedReason::NonConstTableIndex => 3,
            UnsupportedReason::Mod => 4,
            UnsupportedReason::ParametricN => 5,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for DerivEntry {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::deriv::DerivEntry");
        match self {
            DerivEntry::Grad(e) => {
                h.write_u32(0);
                e.hash_into(h);
            }
            // Hash the stable `code`, NEVER the free-text `node` label — a
            // message/label copy-edit must not re-key run_id (§4.1).
            DerivEntry::Unsupported { node: _, code } => {
                h.write_u32(1);
                code.hash_into(h);
            }
        }
    }
}

// ── expr.rs ──────────────────────────────────────────────────────────────────

impl ContentAddressed for BinOp {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::expr::BinOp");
        let idx: u32 = match self {
            BinOp::Add => 0,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            BinOp::Div => 3,
            BinOp::Pow => 4,
            BinOp::Mod => 5,
            BinOp::Min => 6,
            BinOp::Max => 7,
            BinOp::Eq => 8,
            BinOp::Neq => 9,
            BinOp::Lt => 10,
            BinOp::Gt => 11,
            BinOp::Le => 12,
            BinOp::Ge => 13,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for UnOp {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::expr::UnOp");
        let idx: u32 = match self {
            UnOp::Neg => 0,
            UnOp::Exp => 1,
            UnOp::Log => 2,
            UnOp::Sqrt => 3,
            UnOp::Abs => 4,
            UnOp::Floor => 5,
            UnOp::Ceil => 6,
            UnOp::Sin => 7,
            UnOp::Cos => 8,
            UnOp::Tanh => 9,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for Expr {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::expr::Expr");
        // Variant index in declaration order, then the meaningful payload —
        // the single-field wrapper structs (ConstExpr, BinOpWrap, …) are
        // inlined; the variant index already provides domain separation.
        match self {
            Expr::Const(c) => {
                h.write_u32(0);
                // Structural float: raw bits, ±0.0/NaN-distinct.
                h.write_f64_bits(c.value);
            }
            Expr::Param(p) => {
                h.write_u32(1);
                h.write_str(&p.param);
            }
            Expr::Pop(p) => {
                h.write_u32(2);
                h.write_str(&p.pop);
            }
            Expr::PopSum(p) => {
                h.write_u32(3);
                p.pop_sum.hash_into(h);
            }
            Expr::Time(_) => {
                h.write_u32(4);
            }
            Expr::Dt(_) => {
                h.write_u32(5);
            }
            Expr::BinOp(w) => {
                h.write_u32(6);
                w.bin_op.op.hash_into(h);
                w.bin_op.left.hash_into(h);
                w.bin_op.right.hash_into(h);
            }
            Expr::UnOp(w) => {
                h.write_u32(7);
                w.un_op.op.hash_into(h);
                w.un_op.arg.hash_into(h);
            }
            Expr::Cond(w) => {
                h.write_u32(8);
                w.cond.pred.hash_into(h);
                w.cond.then.hash_into(h);
                w.cond.else_.hash_into(h);
            }
            Expr::TimeFunc(w) => {
                h.write_u32(9);
                h.write_str(&w.time_func.name);
            }
            Expr::TableLookup(w) => {
                h.write_u32(10);
                h.write_str(&w.table_lookup.table);
                w.table_lookup.indices.hash_into(h);
            }
            Expr::Projected(_) => {
                h.write_u32(11);
            }
            Expr::UncheckedDim(w) => {
                h.write_u32(12);
                w.unchecked_dim.inner.hash_into(h);
                w.unchecked_dim.dim.hash_into(h);
                h.write_str(&w.unchecked_dim.reason);
            }
            Expr::Reduce(w) => {
                h.write_u32(13);
                w.reduce.hash_into(h);
            }
            Expr::BindingRef(w) => {
                h.write_u32(14);
                h.write_str(&w.binding_ref);
            }
            Expr::ObsColumnRef(w) => {
                h.write_u32(15);
                h.write_str(&w.obs_column_ref);
            }
            // gh#272 LICM: fresh variant index 16 (not inserted between 14/15) so
            // existing nodes keep their hash — only the new node and the empty
            // `per_eval_bindings` field shift the model hash at 0.19.
            Expr::PerEvalRef(w) => {
                h.write_u32(16);
                h.write_str(&w.per_eval_ref);
            }
        }
    }
}

// ── transition.rs ────────────────────────────────────────────────────────────

impl ContentAddressed for StoichiometryEntry {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::transition::StoichiometryEntry");
        h.write_str(&self.0);
        h.write_i64(self.1);
    }
}

impl ContentAddressed for TransitionMetadata {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::transition::TransitionMetadata");
        self.origin_kind.hash_into(h);
        self.source_compartment.hash_into(h);
        self.dest_compartment.hash_into(h);
    }
}

impl ContentAddressed for DrawMethod {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::transition::DrawMethod");
        match self {
            DrawMethod::Poisson => h.write_u32(0),
            DrawMethod::Overdispersed { sigma_sq, sigma_sq_grad } => {
                h.write_u32(1);
                sigma_sq.hash_into(h);
                // sigma_sq_grad: HashMap<String, DerivEntry> — sorted by key
                // (mirrors the transition rate_grad). Empty ⇒ length-0 prefix.
                h.write_str_map(sigma_sq_grad.iter());
            }
            DrawMethod::Deterministic => h.write_u32(2),
        }
    }
}

impl ContentAddressed for TransitionLineage {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::transition::TransitionLineage");
        self.is_lineage_event.hash_into(h);
        self.parent_pool_weights.hash_into(h);
    }
}

impl ContentAddressed for Transition {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::transition::Transition");
        h.write_str(&self.name);
        self.stoichiometry.hash_into(h);
        self.rate.hash_into(h);
        self.metadata.hash_into(h);
        self.draw_method.hash_into(h);
        // rate_grad: HashMap<String, Expr> — sorted by key.
        h.write_str_map(self.rate_grad.iter());
        self.lineage.hash_into(h);
    }
}

// ── parameter.rs ─────────────────────────────────────────────────────────────

impl ContentAddressed for PriorDist {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::PriorDist");
        // The single-field prior structs are inlined (1:1 with variants).
        match self {
            PriorDist::Uniform(p) => {
                h.write_u32(0);
                h.write_f64_bits(p.lower);
                h.write_f64_bits(p.upper);
            }
            PriorDist::Normal(p) => {
                h.write_u32(1);
                h.write_f64_bits(p.mean);
                h.write_f64_bits(p.sd);
            }
            PriorDist::LogNormal(p) => {
                h.write_u32(2);
                h.write_f64_bits(p.mu);
                h.write_f64_bits(p.sigma);
            }
            PriorDist::HalfNormal(p) => {
                h.write_u32(3);
                h.write_f64_bits(p.sigma);
            }
            PriorDist::Beta(p) => {
                h.write_u32(4);
                h.write_f64_bits(p.alpha);
                h.write_f64_bits(p.beta);
            }
            PriorDist::Gamma(p) => {
                h.write_u32(5);
                h.write_f64_bits(p.shape);
                h.write_f64_bits(p.rate);
            }
            PriorDist::Exponential(p) => {
                h.write_u32(6);
                h.write_f64_bits(p.rate);
            }
            PriorDist::Fixed(v) => {
                h.write_u32(7);
                h.write_f64_bits(*v);
            }
            // New variants take fresh discriminants (8, 9) so existing priors'
            // content hashes are unchanged.
            PriorDist::LogUniform(p) => {
                h.write_u32(8);
                h.write_f64_bits(p.lower);
                h.write_f64_bits(p.upper);
            }
            PriorDist::TruncatedNormal(p) => {
                h.write_u32(9);
                h.write_f64_bits(p.mean);
                h.write_f64_bits(p.sd);
                h.write_f64_bits(p.lower);
                h.write_f64_bits(p.upper);
            }
        }
    }
}

impl ContentAddressed for HierarchicalKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::HierarchicalKind");
        let idx: u32 = match self {
            HierarchicalKind::Uniform => 0,
            HierarchicalKind::Normal => 1,
            HierarchicalKind::LogNormal => 2,
            HierarchicalKind::HalfNormal => 3,
            HierarchicalKind::Beta => 4,
            HierarchicalKind::Gamma => 5,
            HierarchicalKind::Exponential => 6,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for HierarchicalPrior {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::HierarchicalPrior");
        self.kind.hash_into(h);
        // args: BTreeMap<String, Expr> — sorted by key (already, but the
        // helper sorts regardless).
        h.write_str_map(self.args.iter());
        h.write_str(&self.pool_over);
    }
}

impl ContentAddressed for Transform {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::Transform");
        let idx: u32 = match self {
            Transform::Log => 0,
            Transform::Logit => 1,
            Transform::Identity => 2,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for ParamKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::ParamKind");
        // Permanent variant indices (run-id stability) — new kinds append.
        let idx: u32 = match self {
            ParamKind::Rate        => 0,
            ParamKind::Probability => 1,
            ParamKind::Count       => 2,
            ParamKind::Positive    => 3,
            ParamKind::Real        => 4,
            ParamKind::Instant     => 5,
            ParamKind::Duration    => 6,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for PriorSpec {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::PriorSpec");
        // Permanent variant indices (run-id stability).
        match self {
            PriorSpec::Flat => h.write_u32(0),
            PriorSpec::Dist(d) => {
                h.write_u32(1);
                d.hash_into(h);
            }
            PriorSpec::Hierarchical(hp) => {
                h.write_u32(2);
                hp.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for ParamValue {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::ParamValue");
        // Permanent variant indices (run-id stability).
        match self {
            ParamValue::Fixed { value } => {
                h.write_u32(0);
                h.write_f64_bits(*value);
            }
            ParamValue::Estimated { init, bounds, prior, transform } => {
                h.write_u32(1);
                hash_opt_f64(h, init);
                // bounds: Option<(f64, f64)> — structural floats, inlined.
                match bounds {
                    None => h.write_u8(0),
                    Some((lo, hi)) => {
                        h.write_u8(1);
                        h.write_f64_bits(*lo);
                        h.write_f64_bits(*hi);
                    }
                }
                prior.hash_into(h);
                transform.hash_into(h);
            }
            ParamValue::Required => h.write_u32(2),
        }
    }
}

impl ContentAddressed for Parameter {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::parameter::Parameter");
        h.write_str(&self.name);
        self.value.hash_into(h);
        self.param_kind.hash_into(h);
        self.param_dim.hash_into(h);
    }
}

// ── observation.rs ───────────────────────────────────────────────────────────

impl ContentAddressed for Projection {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::Projection");
        // Run-id stability: each variant's index is PERMANENT. A new variant
        // takes the next unused index and is appended at the end; existing
        // indices are NEVER renumbered — renumbering churns the run_id of
        // every stored run whose model uses the shifted variant.
        match self {
            Projection::CumulativeFlow(s) => {
                h.write_u32(0);
                h.write_str(s);
            }
            Projection::CurrentPop(s) => {
                h.write_u32(1);
                h.write_str(s);
            }
            Projection::CurrentPopSum(v) => {
                h.write_u32(2);
                v.hash_into(h);
            }
            Projection::DerivedExpr(e) => {
                h.write_u32(3);
                e.hash_into(h);
            }
            // Added gh#160 (strata-summed incidence) — appended at 4.
            Projection::CumulativeFlowSum(v) => {
                h.write_u32(4);
                v.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for Likelihood {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::Likelihood");
        match self {
            // Each arg's gradient map is hashed right after its expression —
            // sorted by key (mirrors the transition rate_grad); empty ⇒ length-0
            // prefix. `n` (Binomial/BetaBinomial) carries no grad (θ-independent).
            Likelihood::Poisson(l) => {
                h.write_u32(0);
                l.rate.hash_into(h);
                h.write_str_map(l.rate_grad.iter());
            }
            Likelihood::NegBinomial(l) => {
                h.write_u32(1);
                l.mean.hash_into(h);
                h.write_str_map(l.mean_grad.iter());
                l.dispersion.hash_into(h);
                h.write_str_map(l.dispersion_grad.iter());
            }
            Likelihood::Normal(l) => {
                h.write_u32(2);
                l.mean.hash_into(h);
                h.write_str_map(l.mean_grad.iter());
                l.sd.hash_into(h);
                h.write_str_map(l.sd_grad.iter());
            }
            Likelihood::Binomial(l) => {
                h.write_u32(3);
                l.n.hash_into(h);
                l.p.hash_into(h);
                h.write_str_map(l.p_grad.iter());
            }
            Likelihood::BetaBinomial(l) => {
                h.write_u32(4);
                l.n.hash_into(h);
                l.alpha.hash_into(h);
                h.write_str_map(l.alpha_grad.iter());
                l.beta.hash_into(h);
                h.write_str_map(l.beta_grad.iter());
            }
            Likelihood::Bernoulli(l) => {
                h.write_u32(5);
                l.p.hash_into(h);
                h.write_str_map(l.p_grad.iter());
            }
        }
    }
}

impl ContentAddressed for RegularSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::RegularSchedule");
        h.write_f64_bits(self.start);
        h.write_f64_bits(self.step);
        h.write_f64_bits(self.end);
    }
}

impl ContentAddressed for ObservationSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::ObservationSchedule");
        match self {
            ObservationSchedule::AtTimes(times) => {
                h.write_u32(0);
                h.write_f64_slice(times);
            }
            ObservationSchedule::Regular(r) => {
                h.write_u32(1);
                r.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for ColumnRole {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::ColumnRole");
        // Permanent variant indices (run-id stability) — new roles append.
        match self {
            ColumnRole::Time => h.write_u32(0),
            ColumnRole::Dim(d) => {
                h.write_u32(1);
                h.write_str(d);
            }
            ColumnRole::Value(k) => {
                h.write_u32(2);
                k.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for ObsColumn {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::ObsColumn");
        h.write_str(&self.name);
        self.role.hash_into(h);
    }
}

impl ContentAddressed for StratumKey {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::StratumKey");
        h.write_str(&self.dim);
        h.write_str(&self.level);
    }
}

impl ContentAddressed for ObservationModel {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::observation::ObservationModel");
        h.write_str(&self.name);
        h.write_str(&self.source);
        self.columns.hash_into(h);
        h.write_str(&self.scored);
        self.emit_schedule.hash_into(h);
        // `stratum` is hashed ONLY when non-empty: an empty stratum (every
        // model without a stratified observation header) writes nothing, so
        // existing run_ids are byte-identical. A non-empty stratum exists only
        // on new stratified-obs leaves (no stored run_ids yet) and is
        // load-bearing — it routes file rows to this leaf — so it must enter
        // the hash. (`Vec::hash_into` would write a `len=0` prefix even when
        // empty, churning every existing id; guard against that.)
        if !self.stratum.is_empty() {
            self.stratum.hash_into(h);
        }
        self.projection.hash_into(h);
        self.likelihood.hash_into(h);
    }
}

// ── ode_equation.rs ──────────────────────────────────────────────────────────

impl ContentAddressed for OdeEquation {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::ode_equation::OdeEquation");
        h.write_str(&self.compartment);
        self.derivative.hash_into(h);
    }
}

// ── table.rs ─────────────────────────────────────────────────────────────────

impl ContentAddressed for OobPolicy {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::table::OobPolicy");
        // Keep Error at index 2 (its value when Clamp=0 / Wrap=1 existed) so
        // removing those dead variants stayed run_id-neutral — do NOT renumber.
        let idx: u32 = match self {
            OobPolicy::Error => 2,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for TableSource {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::table::TableSource");
        match self {
            TableSource::Inline { values } => {
                h.write_u32(0);
                values.hash_into(h);
            }
            TableSource::External { external } => {
                h.write_u32(1);
                h.write_str(external);
            }
        }
    }
}

impl ContentAddressed for Table {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::table::Table");
        h.write_str(&self.name);
        self.source.hash_into(h);
        self.out_of_bounds.hash_into(h);
        self.cell_kind.hash_into(h);
    }
}

// ── time_func.rs ─────────────────────────────────────────────────────────────

impl ContentAddressed for TimeFuncKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::time_func::TimeFuncKind");
        match self {
            TimeFuncKind::Sinusoidal(s) => {
                h.write_u32(0);
                s.amplitude.hash_into(h);
                s.period.hash_into(h);
                s.phase.hash_into(h);
                s.baseline.hash_into(h);
            }
            TimeFuncKind::Piecewise(p) => {
                h.write_u32(1);
                p.breakpoints.hash_into(h);
                p.values.hash_into(h);
            }
            TimeFuncKind::Interpolated(i) => {
                h.write_u32(2);
                i.times.hash_into(h);
                i.values.hash_into(h);
                // InterpMethod inlined.
                let m: u32 = match i.method {
                    ir::time_func::InterpMethod::Linear => 0,
                    ir::time_func::InterpMethod::Constant => 1,
                    ir::time_func::InterpMethod::Spline => 2,
                };
                h.write_u32(m);
            }
            TimeFuncKind::Periodic(p) => {
                h.write_u32(3);
                p.period.hash_into(h);
                p.values.hash_into(h);
            }
            TimeFuncKind::Fourier(f) => {
                h.write_u32(4);
                f.period.hash_into(h);
                // harmonics: Vec<(Expr, Expr)>.
                f.harmonics.hash_into(h);
            }
            TimeFuncKind::PeriodicSpline(s) => {
                h.write_u32(5);
                s.period.hash_into(h);
                s.n_basis.hash_into(h);
                s.degree.hash_into(h);
                s.coefs.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for TimeFunction {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::time_func::TimeFunction");
        h.write_str(&self.name);
        self.kind.hash_into(h);
        self.dim.hash_into(h);
        // gh#314: lag is identity — two models that differ only by a forcing's
        // evaluation-time shift produce different trajectories and must re-key.
        // The Option impl tags presence, so `None` (no lag) stays distinct from
        // any `Some(lag)`.
        self.lag.hash_into(h);
    }
}

// ── intervention.rs ──────────────────────────────────────────────────────────

impl ContentAddressed for RecurringSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::RecurringSchedule");
        h.write_f64_bits(self.start);
        h.write_f64_bits(self.period);
        h.write_f64_bits(self.end);
        hash_opt_f64(h, &self.at_day);
    }
}

impl ContentAddressed for InterventionSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::InterventionSchedule");
        match self {
            InterventionSchedule::AtTimes(times) => {
                h.write_u32(0);
                h.write_f64_slice(times);
            }
            InterventionSchedule::AtTimesExpr(exprs) => {
                h.write_u32(1);
                exprs.hash_into(h);
            }
            InterventionSchedule::Recurring(r) => {
                h.write_u32(2);
                r.hash_into(h);
            }
        }
    }
}


impl ContentAddressed for CmpOp {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::CmpOp");
        // Permanent variant indices (run-id stability) — new ops append.
        let idx: u32 = match self {
            CmpOp::Lt => 0,
            CmpOp::Le => 1,
            CmpOp::Gt => 2,
            CmpOp::Ge => 3,
            CmpOp::Eq => 4,
            CmpOp::Neq => 5,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for ObsReducer {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::ObsReducer");
        // Permanent variant indices (run-id stability) — new reducers append.
        let idx: u32 = match self {
            ObsReducer::Latest => 0,
            ObsReducer::Sum => 1,
            ObsReducer::Mean => 2,
            ObsReducer::Max => 3,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for TriggerQuantity {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::TriggerQuantity");
        match self {
            TriggerQuantity::Observed { stream, window, reducer } => {
                h.write_u32(0);
                h.write_str(stream);
                hash_opt_f64(h, window);
                reducer.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for TriggerThreshold {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::TriggerThreshold");
        match self {
            TriggerThreshold::Const(v) => {
                h.write_u32(0);
                h.write_f64_bits(*v);
            }
            TriggerThreshold::Param(name) => {
                h.write_u32(1);
                h.write_str(name);
            }
        }
    }
}

impl ContentAddressed for TriggerExpr {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::TriggerExpr");
        match self {
            TriggerExpr::Cmp { lhs, op, rhs } => {
                h.write_u32(0);
                lhs.hash_into(h);
                op.hash_into(h);
                rhs.hash_into(h);
            }
            TriggerExpr::And(a, b) => {
                h.write_u32(1);
                a.hash_into(h);
                b.hash_into(h);
            }
            TriggerExpr::Or(a, b) => {
                h.write_u32(2);
                a.hash_into(h);
                b.hash_into(h);
            }
            TriggerExpr::Not(a) => {
                h.write_u32(3);
                a.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for ReactiveTrigger {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::ReactiveTrigger");
        self.when_.hash_into(h);
        h.write_f64_bits(self.after);
        self.once.hash_into(h);
        hash_opt_f64(h, &self.cooldown);
    }
}

impl ContentAddressed for FireSource {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::FireSource");
        match self {
            FireSource::Scheduled(s) => {
                h.write_u32(0);
                s.hash_into(h);
            }
            FireSource::Reactive(t) => {
                h.write_u32(1);
                t.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for Action {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::Action");
        match self {
            Action::FractionTransfer(a) => {
                h.write_u32(0);
                h.write_str(&a.src);
                h.write_str(&a.dst);
                a.fraction.hash_into(h);
            }
            Action::AbsoluteTransfer(a) => {
                h.write_u32(1);
                h.write_str(&a.src);
                h.write_str(&a.dst);
                a.count.hash_into(h);
            }
            Action::Set(a) => {
                h.write_u32(2);
                h.write_str(&a.compartment);
                a.value.hash_into(h);
            }
            Action::Add(a) => {
                h.write_u32(3);
                h.write_str(&a.compartment);
                a.count.hash_into(h);
            }
        }
    }
}

impl ContentAddressed for Intervention {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::Intervention");
        h.write_str(&self.name);
        self.base_name.hash_into(h);
        self.fire.hash_into(h);
        self.actions.hash_into(h);
        self.kind.hash_into(h);
    }
}

impl ContentAddressed for InterventionKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::intervention::InterventionKind");
        // Permanent variant indices (run-id stability) — new kinds append.
        let idx: u32 = match self {
            InterventionKind::Scenario => 0,
            InterventionKind::Event    => 1,
        };
        h.write_u32(idx);
    }
}

// ── model.rs ─────────────────────────────────────────────────────────────────

impl ContentAddressed for CompartmentKind {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::CompartmentKind");
        let idx: u32 = match self {
            CompartmentKind::Integer => 0,
            CompartmentKind::Real => 1,
        };
        h.write_u32(idx);
    }
}

impl ContentAddressed for Compartment {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::Compartment");
        h.write_str(&self.name);
        self.kind.hash_into(h);
    }
}

impl ContentAddressed for InitialConditions {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::InitialConditions");
        match self {
            InitialConditions::Explicit(m) => {
                h.write_u32(0);
                h.write_str_f64_map(m.iter());
            }
            InitialConditions::Parameterized(m) => {
                h.write_u32(1);
                h.write_str_map(m.iter());
            }
            InitialConditions::FromDistribution(m) => {
                h.write_u32(2);
                h.write_str_map(m.iter());
            }
        }
    }
}

impl ContentAddressed for RegularOutputSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::RegularOutputSchedule");
        h.write_f64_bits(self.start);
        h.write_f64_bits(self.step);
    }
}

impl ContentAddressed for OutputSchedule {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::OutputSchedule");
        match self {
            OutputSchedule::Regular(r) => {
                h.write_u32(0);
                r.hash_into(h);
            }
            OutputSchedule::AtTimes(times) => {
                h.write_u32(1);
                h.write_f64_slice(times);
            }
        }
    }
}

impl ContentAddressed for OutputConfig {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::OutputConfig");
        self.times.hash_into(h);
        h.write_str(&self.format);
        self.trajectory.hash_into(h);
        self.observations.hash_into(h);
    }
}

impl ContentAddressed for SimulationConfig {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::SimulationConfig");
        h.write_f64_bits(self.t_start);
        h.write_f64_bits(self.t_end);
        h.write_str(&self.time_semantics);
        hash_opt_f64(h, &self.dt);
        self.rng_seed.hash_into(h);
        // gh#166: hash the integrator ONLY when non-default (Rk45 + its
        // tolerances, tagged so atol/rtol can't collide), so a default-Rk4 model
        // keeps its pre-gh#166 run-id (no cache churn) while rk45 / explicit
        // tolerances — which produce different trajectories — get a distinct
        // content address.
        if let ir::model::Integrator::Rk45 { atol, rtol } = &self.integrator {
            h.write_str("rk45");
            if let Some(a) = atol { h.write_str("atol"); h.write_f64_bits(*a); }
            if let Some(r) = rtol { h.write_str("rtol"); h.write_f64_bits(*r); }
        }
    }
}

impl ContentAddressed for Preset {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::Preset");
        h.write_str(&self.name);
        h.write_str(&self.label);
        h.write_str_f64_map(self.params.iter());
        self.enable.hash_into(h);
        self.disable.hash_into(h);
        h.write_str_f64_map(self.scale.iter());
        self.compose.hash_into(h);
        hash_opt_f64(h, &self.t_end);
    }
}

impl ContentAddressed for Dimension {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::Dimension");
        h.write_str(&self.name);
        self.values.hash_into(h);
    }
}

impl ContentAddressed for ModelStructure {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::ModelStructure");
        self.dimensions.hash_into(h);
        // compartment_dims: HashMap<String, Vec<String>>.
        h.write_str_map(self.compartment_dims.iter());
        self.base_compartments.hash_into(h);
        self.transmission_transitions.hash_into(h);
        self.infectious_compartments.hash_into(h);
    }
}

impl ContentAddressed for BalanceSpec {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::BalanceSpec");
        h.write_str(&self.target);
        self.expr.hash_into(h);
    }
}

impl ContentAddressed for Binding {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::Binding");
        h.write_str(&self.name);
        self.expr.hash_into(h);
    }
}

impl ContentAddressed for Model {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        header(h, "ir::model::Model");
        h.write_str(&self.name);
        h.write_str(&self.version);
        h.write_str(&self.time_unit);
        self.description.hash_into(h);
        self.origin.hash_into(h);
        self.origin_rata_die.hash_into(h);
        self.compartments.hash_into(h);
        self.transitions.hash_into(h);
        self.ode_equations.hash_into(h);
        self.time_functions.hash_into(h);
        self.tables.hash_into(h);
        self.interventions.hash_into(h);
        self.observations.hash_into(h);
        self.parameters.hash_into(h);
        self.bindings.hash_into(h);
        // gh#272 LICM: identity field (the emitted IR is hashed). Empty by default
        // — but the empty Vec's length prefix still shifts the model hash at the
        // 0.19 schema bump (a deliberate, version-bumped re-key); pinned by the
        // distinctness test in this module.
        self.per_eval_bindings.hash_into(h);
        self.initial_conditions.hash_into(h);
        self.output.hash_into(h);
        self.simulation.hash_into(h);
        self.presets.hash_into(h);
        self.model_structure.hash_into(h);
        self.balance.hash_into(h);
        self.identity_tracked_compartments.hash_into(h);
    }
}

#[cfg(test)]
mod tests;
