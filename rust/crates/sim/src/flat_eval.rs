//! gh#209 — a flat-bytecode evaluator for rate expressions, the canonical
//! measurement prototype for an alternative to the recursive `eval_resolved`
//! tree-walk.
//!
//! A `ResolvedExpr` tree is flattened once (off the hot path, the analogue of
//! `resolve_expr`) into a contiguous `Vec<Op>` plus side-tables, then executed
//! with a stack machine. The winning configuration this module distils:
//!
//! 1. **Full flatten, no delegation for the common nodes.** `IntPopSum` /
//!    `MixedPopSum` become dedicated ops over a side-table of index lists; a
//!    `BindingRef` compiles its body to its own tape (`binding_progs`), so a
//!    binding never re-enters the tree-walk. The single deliberately-delegated
//!    node is `TableLookup` (the table-OOB thread-local recording machinery is
//!    genuinely complex and rare; delegating it is an explicit, documented
//!    choice — see the `emit` match).
//! 2. **Superinstruction arithmetic.** The dominant binary ops (`+ - * /`) get
//!    dedicated opcodes (`Op::Add`/`Sub`/`Mul`/`Div`) with direct executor arms,
//!    so the hot path dispatches once instead of `Op::Bin` → `match BinOp`. The
//!    remaining binary ops fall to `Op::BinOther` → `apply_bin`.
//! 3. **Bounds-check-free stack.** The tape's max stack depth is computed at
//!    flatten time; the executor runs over a pre-sized raw buffer with
//!    `get_unchecked`, so the inner loop has no push/pop bookkeeping and no
//!    bounds checks.
//! 4. **`&mut` binding cache.** A generation-stamped cache threaded by `&mut`
//!    (not a thread-local) — direct field reads, no macOS `_tlv_get_addr` TLS
//!    cost per `Binding` op. In the bench the cache is inactive, so every
//!    binding misses → runs its body tape → byte-identical to `eval_resolved`'s
//!    on-demand path.
//!
//! Byte-identity vs `eval_resolved` on every rate is the non-negotiable
//! invariant, pinned by the bench's bit-exact check.
//!
//! Wiring: opt-in behind the `CAMDL_EVAL_FLAT` env toggle (default OFF, see
//! `eval_flat_enabled`). When OFF, the `FlatVm` is never built and
//! `eval_propensities` takes the unchanged `eval_resolved` path — default
//! behaviour is byte-for-byte identical to not having this module wired at all.
//! When ON, `eval_propensities` runs the flat tape with the same per-rate
//! NaN/table-OOB/negative-rate handling as the recursive path.

use std::sync::OnceLock;

use ir::expr::{BinOp, UnOp};

use crate::propensity::{eval_forcing, EvalCtx};
use crate::resolved_expr::{eval_resolved, ResolvedExpr};

/// Opt-in toggle for the flat-bytecode evaluator on the production propensity
/// path. Presence-based (matches `CAMDL_NO_BINDING_CACHE` / `CAMDL_EVAL_UNRESOLVED`):
/// any value of `CAMDL_EVAL_FLAT` (even empty) turns it on. Default OFF — when
/// unset, `ResolvedModel::flat_vm` is `None` and `eval_propensities` takes the
/// unchanged `eval_resolved` path, so default behaviour is byte-for-byte
/// identical to today.
pub fn eval_flat_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CAMDL_EVAL_FLAT").is_some())
}

/// A flat opcode. Kept small (tag + payload) so the tape is dense and
/// cache-friendly. Index-list ops (`IntPopSum`/`MixedPopSum`), bindings, and
/// delegated sub-trees carry a `u32` index into a side-table on `FlatProg`.
#[derive(Clone, Debug)]
pub enum Op {
    Const(f64),
    Param(u32),
    IntPop(u32),
    RealPop(u32),
    Time,
    Dt,
    /// `ctx.projected.unwrap_or(0.0)` (observation-likelihood context only).
    Projected,
    /// Forcing evaluation; payload indexes `ctx.model.time_func_cache`.
    TimeFunc(u32),
    /// Sum of integer compartments; payload indexes `FlatProg::isets`.
    IntPopSum(u32),
    /// Sum mixing int + real compartments; payload indexes `FlatProg::msets`.
    MixedPopSum(u32),
    // Superinstruction arithmetic: each pops 2, pushes 1, dispatched directly.
    Add,
    Sub,
    Mul,
    Div,
    /// Remaining binary ops (Pow/Mod/Min/Max/Eq/Neq/Lt/Gt/Le/Ge) → `apply_bin`.
    BinOther(BinOp),
    Un(UnOp), // pops 1, pushes 1
    SumN(u32), // pops N, pushes left-fold sum (Reduce)
    JumpIfFalse(u32),
    Jump(u32),
    /// Inlined binding cache lookup; payload is the binding slot. On a hit the
    /// cached value is pushed; on a miss `FlatProg::binding_progs[slot]` runs.
    Binding(u32),
    /// `TableLookup`: push `eval_resolved(&subs[i], ctx)`. The only delegated
    /// node — its table-OOB thread-local recording is complex and rare.
    Delegate(u32),
}

/// Index lists for `MixedPopSum`.
#[derive(Clone, Debug)]
pub struct MixedSet {
    pub int_indices: Vec<u32>,
    pub real_indices: Vec<u32>,
}

/// A compiled rate (or binding body) op tape plus its side-tables.
#[derive(Clone, Debug, Default)]
pub struct FlatProg {
    pub ops: Vec<Op>,
    /// Sub-trees for the delegated node (`TableLookup` only).
    pub subs: Vec<ResolvedExpr>,
    /// Index lists for `IntPopSum` ops.
    pub isets: Vec<Vec<u32>>,
    /// Index lists for `MixedPopSum` ops.
    pub msets: Vec<MixedSet>,
    /// Max stack depth this tape needs (computed at flatten time; drives the
    /// bounds-check-free stack buffer sizing).
    pub max_depth: u32,
}

/// A full compiled model surface: one tape per rate, one per binding body, and
/// the binding count (for cache sizing). Mirrors `cm.resolved`.
#[derive(Clone, Debug, Default)]
pub struct FlatVm {
    pub rates: Vec<FlatProg>,
    pub binding_progs: Vec<FlatProg>,
    pub n_bindings: usize,
}

/// Generation-stamped binding cache, threaded by `&mut` through the executor.
/// Direct field reads — no thread-local, no `_tlv_get_addr` per `Binding` op.
/// In the bench `active` is false, so every binding misses → runs its body →
/// byte-identical to `eval_resolved`'s on-demand path.
pub struct FlatCache {
    val: Vec<f64>,
    stamp: Vec<u32>,
    gen: u32,
    active: bool,
}

impl FlatCache {
    pub fn new(n_bindings: usize) -> Self {
        FlatCache { val: vec![0.0; n_bindings], stamp: vec![0; n_bindings], gen: 0, active: false }
    }

    /// True iff this cache is sized for exactly `n` bindings. The per-thread
    /// cache is rebuilt (via `FlatCache::new`) when this returns false, mirroring
    /// `CacheScope::enter`'s `c.val.len() != n_bindings` re-alloc guard.
    #[inline]
    pub fn is_sized(&self, n: usize) -> bool {
        self.val.len() == n
    }

    /// Activate the cache for one propensity-vector evaluation: bump the
    /// generation (invalidating the prior state's cached values) and mark active.
    /// Mirrors `resolved_expr::CacheScope::enter` exactly so the binding hit/miss
    /// protocol — and therefore the values returned — match the `eval_resolved`
    /// path bit-for-bit. The caller is responsible for sizing the cache to the
    /// model's binding count first (see `is_sized`).
    #[inline]
    pub fn activate(&mut self) {
        let n = self.val.len();
        if self.stamp.len() != n {
            self.stamp = vec![0; n];
            self.gen = 0;
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // Generation wrapped: clear stamps so no stale slot aliases gen 0.
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.gen = 1;
        }
        self.active = true;
    }
}

/// Compile a `ResolvedExpr` rate tree into a flat tape.
pub fn flatten(e: &ResolvedExpr) -> FlatProg {
    let mut f = FlatProg::default();
    emit(e, &mut f);
    f.max_depth = compute_max_depth(&f);
    f
}

/// Build the full `FlatVm` from a compiled model's resolved surface.
pub fn build(rates: &[ResolvedExpr], bindings: &[ResolvedExpr]) -> FlatVm {
    FlatVm {
        rates: rates.iter().map(flatten).collect(),
        binding_progs: bindings.iter().map(flatten).collect(),
        n_bindings: bindings.len(),
    }
}

/// Lower one `ResolvedExpr` node onto the tape.
///
/// Exhaustive by design — there is NO catch-all arm. A new `ResolvedExpr`
/// variant must add an arm here, which the compiler enforces (it becomes a
/// compile error otherwise). This keeps the conversion type-total: an unhandled
/// node can never silently fall through to a delegate.
fn emit(e: &ResolvedExpr, f: &mut FlatProg) {
    match e {
        ResolvedExpr::Const(v) => f.ops.push(Op::Const(*v)),
        ResolvedExpr::Param(i) => f.ops.push(Op::Param(*i as u32)),
        ResolvedExpr::IntPop(i) => f.ops.push(Op::IntPop(*i as u32)),
        ResolvedExpr::RealPop(i) => f.ops.push(Op::RealPop(*i as u32)),
        ResolvedExpr::Time => f.ops.push(Op::Time),
        ResolvedExpr::Dt => f.ops.push(Op::Dt),
        ResolvedExpr::Projected => f.ops.push(Op::Projected),
        ResolvedExpr::TimeFunc(idx) => f.ops.push(Op::TimeFunc(*idx as u32)),
        ResolvedExpr::IntPopSum(indices) => {
            let idx = f.isets.len() as u32;
            f.isets.push(indices.iter().map(|&i| i as u32).collect());
            f.ops.push(Op::IntPopSum(idx));
        }
        ResolvedExpr::MixedPopSum { int_indices, real_indices } => {
            let idx = f.msets.len() as u32;
            f.msets.push(MixedSet {
                int_indices: int_indices.iter().map(|&i| i as u32).collect(),
                real_indices: real_indices.iter().map(|&i| i as u32).collect(),
            });
            f.ops.push(Op::MixedPopSum(idx));
        }
        ResolvedExpr::BinOp { op, left, right } => {
            emit(left, f);
            emit(right, f);
            // Superinstructions for the hot arithmetic; the rest go via apply_bin.
            f.ops.push(match op {
                BinOp::Add => Op::Add,
                BinOp::Sub => Op::Sub,
                BinOp::Mul => Op::Mul,
                BinOp::Div => Op::Div,
                other => Op::BinOther(other.clone()),
            });
        }
        ResolvedExpr::UnOp { op, arg } => {
            emit(arg, f);
            f.ops.push(Op::Un(op.clone()));
        }
        ResolvedExpr::Reduce(terms) => {
            for t in terms {
                emit(t, f);
            }
            f.ops.push(Op::SumN(terms.len() as u32));
        }
        ResolvedExpr::Cond { pred, then_, else_ } => {
            emit(pred, f);
            let jf = f.ops.len();
            f.ops.push(Op::JumpIfFalse(0)); // backpatched
            emit(then_, f);
            let j = f.ops.len();
            f.ops.push(Op::Jump(0)); // backpatched
            f.ops[jf] = Op::JumpIfFalse(f.ops.len() as u32); // else target
            emit(else_, f);
            f.ops[j] = Op::Jump(f.ops.len() as u32); // end target
        }
        ResolvedExpr::UncheckedDim { inner } => emit(inner, f), // transparent
        ResolvedExpr::BindingRef(slot) => f.ops.push(Op::Binding(*slot as u32)),
        // gh#272: the flat VM's per-eval tape is deferred (step 1.4). Until then
        // `build` is gated off for models with per-eval bindings (see
        // `CompiledModel::new`), so this node never reaches the emitter.
        ResolvedExpr::PerEvalRef(_) =>
            unreachable!("flat VM emitted for a per-eval model; build() must be gated off"),
        // The one deliberately-delegated node: TableLookup. Reimplementing the
        // OOB thread-local recording + per-policy machinery as opcodes is large
        // and risky for a rare node; delegating its whole sub-tree to
        // eval_resolved is an explicit, documented choice (not a catch-all).
        ResolvedExpr::TableLookup { .. } => {
            let idx = f.subs.len() as u32;
            f.subs.push(e.clone());
            f.ops.push(Op::Delegate(idx));
        }
        // `ObsColumnRef` reads a per-observation aux value from `ctx.aux` and
        // appears ONLY in likelihood expressions, never in a transition rate, so
        // a rate tape never actually emits this op. Delegated (like
        // `TableLookup`) rather than given a dedicated opcode: this keeps `emit`
        // type-total — a future `ResolvedExpr` variant is still a compile error —
        // without adding executor machinery for a node the propensity path cannot
        // reach. If it ever did appear, `Op::Delegate` evaluates the exact
        // `eval_resolved` semantics (aux lookup, 0.0 floor when absent).
        ResolvedExpr::ObsColumnRef(_) => {
            let idx = f.subs.len() as u32;
            f.subs.push(e.clone());
            f.ops.push(Op::Delegate(idx));
        }
    }
}

/// Symbolically execute the tape's stack effect to find the peak depth. The tape
/// is a postorder tree serialization, so a single linear pass gives a safe upper
/// bound: both arms of a `Cond` net +1 from the same base and are laid out
/// sequentially, so the linear max bounds the true runtime max. Safety (not
/// tightness) is what the pre-sized buffer needs.
fn compute_max_depth(f: &FlatProg) -> u32 {
    let mut depth: i64 = 0;
    let mut max: i64 = 0;
    for op in &f.ops {
        match op {
            Op::Const(_) | Op::Param(_) | Op::IntPop(_) | Op::RealPop(_)
            | Op::Time | Op::Dt | Op::Projected | Op::TimeFunc(_)
            | Op::IntPopSum(_) | Op::MixedPopSum(_)
            | Op::Binding(_) | Op::Delegate(_) => depth += 1, // push 1
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::BinOther(_) => depth -= 1, // pop 2, push 1
            Op::Un(_) => {}                  // pop 1, push 1
            Op::SumN(n) => depth -= *n as i64 - 1, // pop n, push 1
            Op::JumpIfFalse(_) => depth -= 1, // pop predicate
            Op::Jump(_) => {}
        }
        if depth > max {
            max = depth;
        }
    }
    // +2 slack guards any off-by-one in the Cond branch accounting.
    (max + 2).max(1) as u32
}

/// Remaining binary ops (everything except + - * /). Byte-identical to
/// `eval_resolved`'s BinOp arm for these operators.
#[inline(always)]
fn apply_bin(op: &BinOp, a: f64, b: f64, allow: bool) -> f64 {
    match op {
        // The four superinstruction ops never reach here (handled inline), but
        // we keep them exhaustive for total matching.
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            if b == 0.0 {
                if allow { 0.0 } else { f64::NAN }
            } else {
                a / b
            }
        }
        BinOp::Pow => {
            let r = a.powf(b);
            if r.is_nan() || r.is_infinite() {
                if allow { 0.0 } else { f64::NAN }
            } else {
                r
            }
        }
        BinOp::Mod => {
            if b == 0.0 {
                if allow { 0.0 } else { f64::NAN }
            } else {
                a.rem_euclid(b)
            }
        }
        BinOp::Min => a.min(b),
        BinOp::Max => a.max(b),
        BinOp::Eq => if a == b { 1.0 } else { 0.0 },
        BinOp::Neq => if a != b { 1.0 } else { 0.0 },
        BinOp::Lt => if a < b { 1.0 } else { 0.0 },
        BinOp::Gt => if a > b { 1.0 } else { 0.0 },
        BinOp::Le => if a <= b { 1.0 } else { 0.0 },
        BinOp::Ge => if a >= b { 1.0 } else { 0.0 },
    }
}

#[inline(always)]
fn apply_un(op: &UnOp, a: f64, allow: bool) -> f64 {
    let result = match op {
        UnOp::Neg => -a,
        UnOp::Exp => a.exp(),
        UnOp::Log => if a > 0.0 { a.ln() } else if allow { 0.0 } else { f64::NAN },
        UnOp::Sqrt => if a >= 0.0 { a.sqrt() } else if allow { 0.0 } else { f64::NAN },
        UnOp::Abs => a.abs(),
        UnOp::Floor => a.floor(),
        UnOp::Ceil => a.ceil(),
        UnOp::Sin => a.sin(),
        UnOp::Cos => a.cos(),
        UnOp::Tanh => a.tanh(),
    };
    if result.is_nan() {
        if allow { 0.0 } else { f64::NAN }
    } else {
        result
    }
}

/// Read an integer compartment count as f64, honoring the ODE float override.
#[inline(always)]
fn int_val(ctx: &EvalCtx<'_>, i: usize) -> f64 {
    match ctx.int_float_override {
        Some(fl) => fl[i],
        None => ctx.int_s.counts[i] as f64,
    }
}

/// Total scratch capacity to reserve before a top-level eval, so the buffer
/// never reallocates mid-evaluation (the executor holds a raw pointer across the
/// whole tape). A binding-body eval runs ABOVE the caller's frame, so the worst
/// case is the rate's depth plus, for each nesting level, the deepest binding
/// body. Binding nesting is bounded by the binding count (a binding only
/// references earlier ones — a DAG), so `rate + n_bindings * max_binding` is a
/// safe (loose) ceiling. Computed once.
pub fn scratch_capacity(vm: &FlatVm) -> usize {
    let max_rate = vm.rates.iter().map(|p| p.max_depth as usize).max().unwrap_or(0);
    let max_bind = vm.binding_progs.iter().map(|p| p.max_depth as usize).max().unwrap_or(0);
    max_rate + vm.n_bindings.max(1) * max_bind + 8
}

/// Top-level rate evaluation. `scratch` must have capacity ≥
/// `scratch_capacity(vm)`. The executor holds a raw pointer for the whole eval;
/// capacity is pre-reserved so no realloc occurs.
#[inline]
pub fn eval_flat(
    vm: &FlatVm,
    prog: &FlatProg,
    ctx: &EvalCtx<'_>,
    scratch: &mut Vec<f64>,
    cache: &mut FlatCache,
) -> f64 {
    let allow = crate::eval_stats::allow_degenerate_rates();
    // SAFETY: caller guarantees capacity ≥ scratch_capacity(vm); no op grows the
    // buffer, so the base pointer stays valid for the whole eval. The executor
    // returns the top value and never relies on the buffer's logical len.
    unsafe {
        let buf = scratch.as_mut_ptr();
        run(vm, prog, ctx, allow, buf, 0, cache)
    }
}

/// Bounds-check-free executor over a raw buffer. `base` is the frame's stack
/// base; the tape pushes above it and the result is the single value at `base`
/// on exit.
///
/// SAFETY: `buf` must have ≥ `base + (this frame's depth)` slots available; the
/// top-level caller pre-reserves the global ceiling via `scratch_capacity`.
unsafe fn run(
    vm: &FlatVm,
    prog: &FlatProg,
    ctx: &EvalCtx<'_>,
    allow: bool,
    buf: *mut f64,
    base: usize,
    cache: &mut FlatCache,
) -> f64 {
    let mut sp = base;
    let ops = &prog.ops;
    let mut pc = 0usize;
    let n = ops.len();
    macro_rules! push {
        ($v:expr) => {{
            *buf.add(sp) = $v;
            sp += 1;
        }};
    }
    while pc < n {
        match ops.get_unchecked(pc) {
            Op::Const(v) => push!(*v),
            Op::Param(i) => push!(*ctx.params.get_unchecked(*i as usize)),
            Op::IntPop(i) => push!(int_val(ctx, *i as usize)),
            Op::RealPop(i) => push!(*ctx.real_s.values.get_unchecked(*i as usize)),
            Op::Time => push!(ctx.t),
            Op::Dt => push!(ctx.dt),
            Op::Projected => push!(ctx.projected.unwrap_or(0.0)),
            Op::TimeFunc(idx) => {
                // gh#314: apply the same evaluation-time shift as the standard
                // path (`propensity::eval_expr`'s `Expr::TimeFunc` arm) so the
                // flat-bytecode path stays byte-identical for lagged forcings.
                let tf = &ctx.model.time_func_cache[*idx as usize];
                let t_eff = match &tf.lag {
                    Some(lag) => ctx.t - eval_resolved(lag, ctx),
                    None => ctx.t,
                };
                let v = eval_forcing(&tf.kind, t_eff, ctx);
                push!(v);
            }
            Op::IntPopSum(idx) => {
                let set = prog.isets.get_unchecked(*idx as usize);
                // Seed -0.0 to mirror `Iterator::sum::<f64>()` in
                // eval_resolved (std folds from -0.0, so empty → -0.0 and a
                // lone element keeps its sign-of-zero). Byte-identity matters
                // for the CAMDL_EVAL_FLAT oracle.
                let mut s = -0.0;
                for &i in set {
                    s += int_val(ctx, i as usize);
                }
                push!(s);
            }
            Op::MixedPopSum(idx) => {
                let set = prog.msets.get_unchecked(*idx as usize);
                // Mirror eval_resolved exactly: two separate partial sums
                // (each seeded -0.0) added as `int_sum + real_sum`. A single
                // continuous fold would regroup the terms and diverge by ULPs
                // (f64 add is non-associative) once both sides are non-empty.
                let mut int_s = -0.0;
                for &i in &set.int_indices {
                    int_s += int_val(ctx, i as usize);
                }
                let mut real_s = -0.0;
                for &i in &set.real_indices {
                    real_s += *ctx.real_s.values.get_unchecked(i as usize);
                }
                push!(int_s + real_s);
            }
            Op::Add => {
                let b = *buf.add(sp - 1);
                let a = *buf.add(sp - 2);
                sp -= 2;
                push!(a + b);
            }
            Op::Sub => {
                let b = *buf.add(sp - 1);
                let a = *buf.add(sp - 2);
                sp -= 2;
                push!(a - b);
            }
            Op::Mul => {
                let b = *buf.add(sp - 1);
                let a = *buf.add(sp - 2);
                sp -= 2;
                push!(a * b);
            }
            Op::Div => {
                let b = *buf.add(sp - 1);
                let a = *buf.add(sp - 2);
                sp -= 2;
                let r = if b == 0.0 {
                    if allow { 0.0 } else { f64::NAN }
                } else {
                    a / b
                };
                push!(r);
            }
            Op::BinOther(op) => {
                let b = *buf.add(sp - 1);
                let a = *buf.add(sp - 2);
                sp -= 2;
                push!(apply_bin(op, a, b, allow));
            }
            Op::Un(op) => {
                let a = *buf.add(sp - 1);
                *buf.add(sp - 1) = apply_un(op, a, allow);
            }
            Op::SumN(nn) => {
                let cnt = *nn as usize;
                let at = sp - cnt;
                // Mirror Reduce's `Iterator::sum::<f64>()` (seeds -0.0); an
                // empty Reduce must yield -0.0, not +0.0, for byte-identity.
                let mut s = -0.0f64;
                for k in 0..cnt {
                    s += *buf.add(at + k);
                }
                sp = at;
                push!(s);
            }
            Op::JumpIfFalse(t) => {
                let pred = *buf.add(sp - 1);
                sp -= 1;
                // Take the else branch when the predicate is not strictly
                // positive, NaN included — matching eval_resolved. The NaN arm
                // is explicit because `pred <= 0.0` alone is false for NaN.
                if pred.is_nan() || pred <= 0.0 {
                    pc = *t as usize;
                    continue;
                }
            }
            Op::Jump(t) => {
                pc = *t as usize;
                continue;
            }
            Op::Binding(slot) => {
                let s = *slot as usize;
                // Direct field reads — no thread-local, no `_tlv_get_addr`.
                let v = if cache.active && s < cache.stamp.len() && cache.stamp[s] == cache.gen {
                    cache.val[s]
                } else {
                    let body = &vm.binding_progs[s];
                    // Body runs ABOVE this frame at base `sp`.
                    let r = run(vm, body, ctx, allow, buf, sp, cache);
                    if cache.active && s < cache.val.len() {
                        cache.val[s] = r;
                        cache.stamp[s] = cache.gen;
                    }
                    r
                };
                push!(v);
            }
            Op::Delegate(i) => {
                // eval_resolved does NOT touch `buf` — safe to call mid-tape.
                let v = eval_resolved(prog.subs.get_unchecked(*i as usize), ctx);
                push!(v);
            }
        }
        pc += 1;
    }
    *buf.add(sp - 1)
}

/// Diagnostic: count ops by category across the whole VM (for the bench report).
pub fn op_histogram(vm: &FlatVm) -> OpHistogram {
    let mut h = OpHistogram::default();
    let mut tally = |progs: &[FlatProg]| {
        for p in progs {
            for op in &p.ops {
                match op {
                    Op::Add | Op::Sub | Op::Mul | Op::Div => h.superinstr += 1,
                    Op::BinOther(_) => h.bin_other += 1,
                    Op::Delegate(_) => h.delegate += 1,
                    Op::Binding(_) => h.binding += 1,
                    Op::IntPopSum(_) => h.int_pop_sum += 1,
                    Op::MixedPopSum(_) => h.mixed_pop_sum += 1,
                    Op::TimeFunc(_) => h.time_func += 1,
                    Op::Projected => h.projected += 1,
                    _ => h.other += 1,
                }
            }
        }
    };
    tally(&vm.rates);
    tally(&vm.binding_progs);
    h
}

#[derive(Default, Debug)]
pub struct OpHistogram {
    /// Add/Sub/Mul/Div superinstructions.
    pub superinstr: usize,
    /// Remaining binary ops (Pow/Mod/Min/Max/comparisons).
    pub bin_other: usize,
    pub int_pop_sum: usize,
    pub mixed_pop_sum: usize,
    pub time_func: usize,
    pub projected: usize,
    pub binding: usize,
    pub delegate: usize,
    /// Const/Param/Pop/Un/SumN/Jump/etc.
    pub other: usize,
}
