(* IR type definitions — mirror of rust/crates/ir/src/ *)
[@@@warning "-30-50"]  (* allow duplicate record field names; suppress docstring warning *)

(* ── Expression ─────────────────────────────────────────────────────────────── *)

type bin_op = Add | Sub | Mul | Div | Pow | Mod | Min | Max | Eq | Neq | Lt | Gt | Le | Ge

type un_op = Neg | Exp | Log | Sqrt | Abs | Floor | Ceil
           | Sin | Cos | Tanh                    (* gh#58 *)

(* ── Observation anchors (gh#616) ───────────────────────────────────────────── *)

(* Which end of the observed record an anchor names. Resolved PER RUN from the
   run's bound observation streams — `AnchorLast` is the maximum observation
   time over them, `AnchorFirst` the minimum — so it is not a release constant:
   two fit configs binding different stream sets resolve different values from
   the same compiled model. *)
type obs_anchor =
  | AnchorFirst
  | AnchorLast

(* An observation anchor plus a constant offset ALREADY FOLDED to model time
   units at compile time (`1 'weeks` under `time_unit = days` → 7.0). Only the
   anchor's value binds later, so the compiled model stays data-independent and
   the offset never needs the observation series to be interpreted.
   `offset = 0.0` is a bare anchor. *)
type anchored_time = {
  anchor: obs_anchor;
  offset: float;   (* model time units; signed *)
}

type bin_op_expr = { op: bin_op; left: expr; right: expr }
and un_op_expr   = { op: un_op;  arg:  expr }
and cond_expr    = { pred: expr; then_: expr; else_: expr }

and expr =
  | Const  of float
  | Param  of string
  | Pop    of string
  | PopSum of string list
  | Time
  | Dt                            (* runtime integrator step (gh#54) *)
  | BinOp  of bin_op_expr
  | UnOp   of un_op_expr
  | Cond   of cond_expr
  | TimeFunc of string           (* name of the time function *)
  | TableLookup of string * expr list  (* table name, index exprs *)
  (* n-ary sum over already-substituted terms (proposal
     2026-05-29-shared-bindings-and-reduction, Fix D). Replaces the deep
     left-nested Add chain that `sum(...)` over a dimension used to lower to,
     which tripped serde's recursion limit past ~50 patches. Sum semantics only;
     Prod is deferred (the FOI needs only sum, and a half-built Prod is worse
     than none — documented deviation from the proposal's {op,terms} shape). *)
  | Reduce of expr list
  (* Reference to a model-level binding by name (Fix B, shared per-coordinate
     bindings). Resolved to a slot at CompiledModel::new (like Param/Pop) and
     evaluated on-demand from the binding's body. Hoisted FOI aggregates
     (N[l], I_agg[l], spatial force F[l]) are defined once in `model.bindings`
     instead of being inlined into every (patch,age) rate. *)
  | BindingRef of string
  (* Reference to a model-level PER-EVAL binding by name (gh#272 LICM). Like
     BindingRef, but the body is param/table-only (loop-invariant within a
     trajectory) and may be param-CARRYING — so it is cached once per θ-stable
     scope, not per step. Produced by the LICM pass (post-autodiff; on by
     default, `CAMDL_NO_LICM` disables); absent only when LICM is disabled or
     nothing hoists. Resolved to a slot at CompiledModel::new against
     `model.per_eval_bindings`. *)
  | PerEvalRef of string
  | Projected                    (* refers to projection output in likelihoods *)
  (* Per-observation auxiliary data column referenced by name in a likelihood
     (e.g. binomial `n = tested`). The Rust binder resolves it against the
     enclosing stream's bound aux columns and evaluates it per observation —
     symmetric to [Projected], which the binder fills per observation from the
     projection. Only valid inside a likelihood argument expression
     (2026-06-10 observation data-entry §3, §6.1). *)
  | ObsColumnRef of string
  (* An observation anchor ± a compile-folded constant offset (gh#616).
     UNRESOLVED until a run binds its data: the runtime resolver substitutes a
     `Const` before anything evaluates, and `CompiledModel::new` refuses a model
     that still carries one — so this variant never reaches the evaluator, and
     no backend has a case for it. Emitted ONLY into a forcing `breakpoints`
     list (the interception is scoped to that key), which is why it is a
     structural knot rather than a value in the dynamics. *)
  | ObsAnchor of anchored_time
  (* Per-expression dimensional escape: asserts the wrapped subexpression
     has dimension `(dim_p, dim_t)` without the checker verifying. The
     user-supplied `reason` is retained for audit trails (run.json) and
     is not consumed by runtime evaluation. Runtime semantics: identity —
     evaluates `inner` and returns its value. See
     docs/dev/proposals/notes/unchecked-dim-escape.md. *)
  | UncheckedDim of unchecked_dim_expr

and unchecked_dim_expr = {
  inner:  expr;
  dim_p:  int;
  dim_t:  int;
  reason: string;
}

(* ── Derivative entries (obs/σ² gradient surface) ────────────────────────────────

   The observation/σ² analogue of the transition [rate_grad], but classified so a
   live-but-omitted coefficient never masquerades as a genuine zero. The autodiff
   pass ([autodiff.ml]) classifies each differentiable position as
   [Known | Omitted | Unsupported]; the obs/σ² driver (a later phase) collapses
   that to the two states the IR carries — a real gradient [DEGrad], or a loud,
   coded refusal [DEUnsupported] — so the fit-time gate consumes the reason
   (proposal 2026-07-03-unified-obs-gradient-autodiff §4.1). An ABSENT map key is
   a genuine zero.

   Unlike a rate [Unsupported] (a compile-time error, never serialized), an obs/σ²
   [DEUnsupported] is serialized into the IR and hashed into run_id identity, so it
   carries a STABLE [unsupported_reason] code — the only hashed part — while the
   human label ([node]) and message are derived for display, keeping a copy-edit
   from re-keying run_id. New reasons append at the END (positional hash index). *)
type unsupported_reason =
  | URLag
  | URPeriodicCoeff
  | URStructuralForcing
  | URNonConstTableIndex
  | URMod
  | URParametricN
  | URNonsmoothState   (* gh#275: a nonsmooth function of state (floor/ceil/abs/
                          min/max of a compartment) in a WrtPop derivative — the
                          derivative is not smooth, so a gradient method is refused *)
  | URZeroInflated     (* a param reaching a zero_inflated_neg_binomial likelihood,
                          which is scoring-only (no emitted gradient). Detected at
                          the Rust fit gate; never emitted into IR by the OCaml
                          autodiff (ZI has no diffable field), so mirrored here only
                          to keep the wire-name enum complete. *)

type deriv_entry =
  | DEGrad of expr
  | DEUnsupported of { node : string; code : unsupported_reason }

(* snake_case wire names — must match the Rust `#[serde(rename_all="snake_case")]`
   on `ir::deriv::UnsupportedReason`. *)
let unsupported_reason_name = function
  | URLag                -> "lag"
  | URPeriodicCoeff      -> "periodic_coeff"
  | URStructuralForcing  -> "structural_forcing"
  | URNonConstTableIndex -> "non_const_table_index"
  | URMod                -> "mod"
  | URParametricN        -> "parametric_n"
  | URNonsmoothState     -> "nonsmooth_state"
  | URZeroInflated       -> "zero_inflated"

let unsupported_reason_of_name = function
  | "lag"                   -> Some URLag
  | "periodic_coeff"        -> Some URPeriodicCoeff
  | "structural_forcing"    -> Some URStructuralForcing
  | "non_const_table_index" -> Some URNonConstTableIndex
  | "mod"                   -> Some URMod
  | "parametric_n"          -> Some URParametricN
  | "nonsmooth_state"       -> Some URNonsmoothState
  | "zero_inflated"         -> Some URZeroInflated
  | _                       -> None

(* ── Compartment ─────────────────────────────────────────────────────────────── *)

type compartment_kind = Integer | Real

type compartment = {
  name: string;
  kind: compartment_kind;
}

(* ── Transition ─────────────────────────────────────────────────────────────── *)

type stoichiometry_entry = string * int

type transition_metadata = {
  origin_kind:        string option;
  source_compartment: string option;
  dest_compartment:   string option;
}

type draw_method =
  | DrawPoisson
  (* [sigma_sq_grad] is the ∂σ²/∂param map (empty ⇒ not computed; absent key ⇒
     genuine zero), carried alongside its expression so a derivative can never be
     written without a slot for it (the Diffable principle, proposal §4.1). *)
  | DrawOverdispersed of { sigma_sq : expr; sigma_sq_grad : (string * deriv_entry) list }
  | DrawDeterministic

(* Lineage (individual-sampling) annotation, emitted for `#[lineage]`
   transitions (2026-05-19 proposal). Present iff the transition was
   marked `#[lineage]` and passed the linear-in-parents check.

   [parent_pool_weights] is the linear decomposition of the rate over
   parent pools: a list of (parent compartment, per-pool weight
   expression) pairs. For `β·S·I/N` with parent I this is
   [("I", β·S/N)]; multi-pool `β·S·(β_I·I + β_A·A)/N` →
   [("I", β·β_I·S/N); ("A", β·β_A·S/N)]. The runtime samples parent
   pool b with P(b) ∝ weight_b · count_b, then uniform within the
   pool. The weight expression is a frozen coefficient at the event
   instant (normalizers like 1/N are evaluated at the current state),
   so it must NOT itself reference the parent count linearly — that
   linear dependence has been factored out into the per-pool entry. *)
type transition_lineage = {
  is_lineage_event:    bool;
  parent_pool_weights: (string * expr) list;
}

type transition = {
  name:            string;
  stoichiometry:   stoichiometry_entry list;
  rate:            expr;
  metadata:        transition_metadata option;
  draw_method:     draw_method;
  rate_grad:       (string * deriv_entry) list;  (** ∂rate/∂param, classified [Grad | Unsupported] (a [grad_map]); empty if not computed. A live-but-omitted coefficient is a serialized [DEUnsupported] the fit gate refuses on (gh#342). *)
  rate_state_grad: (string * deriv_entry) list;  (** ∂rate/∂compartment, the [rate_grad] sibling keyed by compartment ([J_x]'s ingredient for the ODE forward sensitivities, gh#275). Empty until the [WrtPop] autodiff pass emits it. *)
  lineage:         transition_lineage option;  (** Some iff `#[lineage]`; None for ordinary transitions. *)
}

(* ── ODE equation ────────────────────────────────────────────────────────────── *)

type ode_equation = {
  compartment: string;
  derivative:  expr;
}

(* ── Time functions ──────────────────────────────────────────────────────────── *)

type sinusoidal = { amplitude: expr; period: expr; phase: expr; baseline: expr }
type piecewise  = { breakpoints: expr list; values: expr list }
type interpolated = { times: expr list; values: expr list; method_: string }
type periodic   = { period: expr; values: expr list }
(* gh#59: finite Fourier series with N estimable cos/sin pairs.
   `harmonics[k] = (a_k, b_k)` for k = 1, 2, ... — k=0 is the
   baseline (caller modulates as `1 + sum_k a_k cos + b_k sin`). *)
type fourier = { period: expr; harmonics: (expr * expr) list }
(* gh#59 (revised 2026-05-12): periodic B-spline with uniform knots.
   Knots are implicit: dx = period / n_basis, knots at k*dx for
   k = -degree..n_basis+degree. coefs has length = n_basis. degree
   defaults to 3 (cubic). Standard de Boor recurrence + periodic
   wrap-fold; algorithm from de Boor 1978 §X, Eilers & Marx 1996,
   Wand & Ormerod 2008. *)
type periodic_spline = {
  period:  expr;
  n_basis: int;
  degree:  int;
  coefs:   expr list;
}

type time_func_kind =
  | Sinusoidal    of sinusoidal
  | Piecewise     of piecewise
  | Interpolated  of interpolated
  | Periodic      of periodic
  | Fourier       of fourier            (* gh#59 *)
  | PeriodicSpline of periodic_spline   (* gh#59 *)

(* Compile-time provenance for a forcing declared `data = "path"`. The knots
   are read from the file and INLINED into [Interpolated] at compile time, so
   the runtime never opens it — which is exactly why the file has to be
   recorded here. Without this, a completed fit's provenance is silently
   incomplete on the one input most likely to change underneath it: the file
   is not named in the IR, not named in the run record, and not recoverable
   from the baked values.

   NEITHER FIELD IS RUN IDENTITY (see `Model::hash_into` in
   `rust/crates/runid/src/ir_hash.rs`, and the two exclusion tests beside it):

   - [sha256] is redundant with the inlined knots, which ARE hashed. A file
     edit that changes a value already re-keys through the values. Folding
     the byte hash in as well would additionally re-key on edits that change
     no compiled value — a comment line, a trailing newline, CRLF, a column
     reorder, rows for a stratum this model does not read — i.e. it would
     invalidate the cache for a model that is bit-for-bit the same model.
   - [path] must not re-key: the SAME bytes read from a second location (a
     copy, a checkout at a different relative prefix) are the same model.

   Both are therefore provenance in the `#[run_input(provenance)]` sense:
   recorded and displayed, never hashed. *)
type data_source = {
  (* The path AS WRITTEN in the model, not the resolved absolute one, so it
     stays portable and comparable across machines and checkouts. *)
  path:   string;
  (* Lowercase 64-char SHA-256 of the file's raw bytes — reproducible with
     `shasum -a 256 <path>` from the model's directory. *)
  sha256: string;
}

type time_function = {
  name: string;
  kind: time_func_kind;
  (* Required declared dimension from the forcing's tier-3 unit literal
     (e.g. `'count`, `'per_year`, `'ratio`). The dim-checker uses this
     authoritatively. The expander has already applied the scale
     factor to stored values so runtime interpolation returns values
     in the model's `time_unit`. GH #8. *)
  dim:  int * int;
  (* gh#314: optional evaluation-time shift. When [Some lag], every forcing
     kind is evaluated at [t − lag] instead of [t] via one shared shift.
     [lag] is a duration expression already rescaled to the model's
     `time_unit` (a literal like `10 'days`, or a `Param` reference — the
     lag-as-parameter case is a primary motivation). [None] ⇒ no shift,
     byte-identical to a forcing declared without `lag`. *)
  lag:  expr option;
  (* The external file this forcing's knots were read from, when it declared
     `data = "path"`. [None] for every other forcing kind (and for the
     `table =` / inline `times`/`values` forms of `interpolated`), which is
     why it serializes only when present. *)
  data_source: data_source option;
}

(* ── Tables ──────────────────────────────────────────────────────────────────── *)

(* Table out-of-range lookups fail loud (a model bug), never silently
   clamp/flat-extrapolate or wrap — those masked errors with surprising
   behaviour and were removed (the evaluator was built but unselectable). *)
type oob_policy = Error

type table_source =
  | Inline   of expr list  (** values resolved at compile time *)
  | External of string     (** logical name; values supplied via --table name=file at runtime *)

type table = {
  name:          string;
  source:        table_source;
  out_of_bounds: oob_policy;
  (* Optional cell-type annotation (gh#32). When present, declares the
     dimensional kind of every value cell — same vocabulary as
     [parameter.param_kind] ("rate", "probability", "positive",
     "count", "real"). Absent = dimensionless cells (legacy
     behaviour). The dim-checker treats this as authoritative, the
     same way it treats `parameter.param_kind`. *)
  cell_kind:     string option;
}

(* ── Interventions ───────────────────────────────────────────────────────────── *)

type recurring_schedule = { start: float; period: float; end_: float; at_day: float option }

type intervention_schedule =
  | AtTimes     of float list
  | AtTimesExpr of expr list
  (* gh#69: parametric `at [...]` lists. Each element is an arbitrary IR
     expression — typically `Param "t_seed"` or arithmetic of parameters
     and constants. The runtime evaluates each expression once per
     simulation start against the current `params` vector to obtain the
     concrete fire times. The expander emits the legacy `AtTimes` form
     when every entry is a compile-time constant so existing golden IRs
     stay byte-identical. *)
  | Recurring of recurring_schedule

type fraction_transfer = { src: string; dst: string; fraction: expr }
type absolute_transfer = { src: string; dst: string; count: expr }
type set_action        = { compartment: string; value: expr }
type add_action        = { add_compartment: string; add_count: expr }

type action =
  | FractionTransfer of fraction_transfer
  | AbsoluteTransfer of absolute_transfer
  | Set              of set_action
  | AddAction        of add_action

(* Distinguishes the two DSL constructs that both lower to [intervention]:
   `interventions {}` (scenario-toggled) and `events {}` (fire every substep
   unconditionally). Replaces the former [always_active: bool] (gh#107) — a
   named enum names the distinction and extends to a future kind (e.g.
   reactive, gh#204) instead of bolting on a second bool. *)
type intervention_kind =
  | Scenario   (* interventions {} — toggled by enable/disable/set/scale *)
  | Event      (* events {}        — fires unconditionally every substep   *)

(* gh#204. Reactive trigger predicate — a dedicated ADT, NOT the shared [expr]:
   boolean-valued by construction, and its observed() leaves cannot leak into
   rate expressions. Wire strings mirror the Rust serde (snake_case). *)
type cmp_op = CmpLt | CmpLe | CmpGt | CmpGe | CmpEq | CmpNeq

type obs_reducer = RedLatest | RedSum | RedMean | RedMax
  (* observed(s) -> RedLatest; sum_observed(s, window=..) -> RedSum *)

type trigger_quantity =
  | TQObserved of { stream : string; window : float option; reducer : obs_reducer }

type trigger_threshold =
  | TTConst of float
  | TTParam of string

type trigger_expr =
  | TECmp of trigger_quantity * cmp_op * trigger_threshold
  | TEAnd of trigger_expr * trigger_expr
  | TEOr  of trigger_expr * trigger_expr
  | TENot of trigger_expr

(* gh#204. A reactive (state/observation-triggered) fire source: fire when
   [when_] holds, [after] a non-negative lag, optionally rate-limited by
   [cooldown]. The action grammar and effect resolution are shared with
   scheduled interventions — only the fire source differs. *)
type reactive_trigger = {
  when_:    trigger_expr;    (* trigger predicate; wire key "when" *)
  after:    float;          (* non-negative lag before the effect fires    *)
  once:     bool;           (* fire-and-disable; mutually exclusive w/ cooldown *)
  cooldown: float option;   (* min time between firings when [once = false] *)
}

(* gh#204. How an intervention's fire times are produced — orthogonal to
   [intervention_kind] (the toggling/structural axis). A reactive policy is
   [kind = Scenario, fire = Reactive ..]; splitting the fire source from the
   kind makes the illegal pairings unrepresentable. *)
type fire_source =
  | Scheduled of intervention_schedule
  | Reactive  of reactive_trigger

type intervention = {
  name:          string;
  base_name:     string option;
  fire:          fire_source;
  actions:       action list;
  kind:          intervention_kind;
}

(* ── Observation model ───────────────────────────────────────────────────────── *)

type projection =
  | CumulativeFlow    of string
  | CurrentPop        of string
  | CurrentPopSum     of string list
  | DerivedExpr       of expr
  (* New variants append last — keeps parity with the Rust run_id hash,
     whose variant indices are positional and permanent. *)
  | CumulativeFlowSum of string list

(* Each differentiable likelihood argument carries its ∂arg/∂param map (empty ⇒
   not computed; absent key ⇒ genuine zero), the obs analogue of the transition
   [rate_grad]. `n` (Binomial/BetaBinomial) has NO grad — it must be θ-independent
   (rounded to an integer). Proposal 2026-07-03 §4.1, §4.4/§4.5. *)
type grad_map = (string * deriv_entry) list

(* A differentiable likelihood argument: its expression paired with its
   per-parameter classified gradient. Bundling the two means a gradient can never
   be written without a slot for its expression; and the producing pass
   ([Autodiff.differentiate_likelihood]) reconstructs every [diffable] fully, so a
   new differentiable argument is a compile error until it is differentiated
   (proposal 2026-07-06 §4.1/§4.3). `n` (Binomial/BetaBinomial) is NOT a
   [diffable] — it is θ-independent and carries no gradient. On the wire a
   [diffable] is the nested shape [{"expr": …, "grad": …}] (grad omitted when
   empty), so a field [mean] serialises as [{"mean": {"expr": …, "grad": …}}]. *)
(* [proj_grad] is ∂expr/∂[Projected] — the observation FACTOR-2 chain ingredient
   (gh#275): [None] = a genuine zero (the argument does not depend on the
   projection output), [Some (Grad e)] = the derivative, [Some (DEUnsupported _)]
   = a nonsmooth-of-projection refusal the fit-time gate consumes. Sibling of
   [grad] (∂expr/∂θ). *)
type diffable = { expr: expr; grad: grad_map; proj_grad: deriv_entry option }
type poisson_likelihood      = { rate: diffable }
type neg_binomial_likelihood = { mean: diffable; dispersion: diffable }
type normal_likelihood       = { mean: diffable; sd: diffable }
type binomial_likelihood     = { n: expr; p: diffable }
type beta_binomial_likelihood = { n: expr; alpha: diffable; beta: diffable }
(* Beta likelihood for a continuous proportion [x ∈ (0, 1)] — an observed rate,
   coverage, or positivity given directly as a fraction (not a k-of-n count, which
   is [beta_binomial_likelihood]). Mean-linked like [neg_binomial_likelihood]: the
   model predicts [mean] and [concentration] (φ) is the dispersion knob, with shape
   parameters [α = mean·φ], [β = (1 − mean)·φ]. *)
type beta_likelihood         = { mean: diffable; concentration: diffable }
type bernoulli_likelihood    = { p: diffable }
(* Zero-inflated negative binomial: a structural-zero mass [pi] mixed with a
   NegBinomial count. `P(Y=0) = pi + (1-pi)·f(0)`, `P(Y=k>0) = (1-pi)·f(k)`,
   with `f` the NegBinomial(mean, dispersion) pmf. Scoring-only: [mean],
   [dispersion], and [pi] are bare exprs (NOT [diffable]), so the family carries
   no gradient and gradient-based inference (PGAS/NUTS) refuses a model
   containing one via the Rust capability gate; MH/PMMH/PF/IF2 score it. The
   surface is the `zero_inflated(base = neg_binomial(...), pi = ...)` wrapper,
   desugared to this flat variant at parse time. See §12.2. *)
type zi_neg_binomial_likelihood = { mean: expr; dispersion: expr; pi: expr }

type likelihood =
  | Poisson              of poisson_likelihood
  | NegBinomial          of neg_binomial_likelihood
  | Normal               of normal_likelihood
  | Binomial             of binomial_likelihood
  | BetaBinomial         of beta_binomial_likelihood
  | Beta                 of beta_likelihood
  | Bernoulli            of bernoulli_likelihood
  | ZeroInflatedNegBinomial of zi_neg_binomial_likelihood

(* DSL parameter-type keyword (the [param_kind] production in parser.mly:
   `rate`, `probability`, `count`, `positive`, `real`, `instant`,
   `duration`). Was a free [string option]; the typed enum is rejected at IR
   deserialisation rather than re-parsed by every consumer (the gh#191
   stringly-typed surface). The dimensional meaning of each kind lives in
   [Dimcheck.param_dim_of_kind]; `instant`/`duration` are time-typed ([T]) per
   the 2026-05-22 calendar-time proposal. *)
type param_kind =
  | Rate
  | Probability
  | Count
  | Positive
  | Real
  | Instant
  | Duration

let param_kind_name = function
  | Rate        -> "rate"
  | Probability -> "probability"
  | Count       -> "count"
  | Positive    -> "positive"
  | Real        -> "real"
  | Instant     -> "instant"
  | Duration    -> "duration"

let param_kind_of_name = function
  | "rate"        -> Some Rate
  | "probability" -> Some Probability
  | "count"       -> Some Count
  | "positive"    -> Some Positive
  | "real"        -> Some Real
  | "instant"     -> Some Instant
  | "duration"    -> Some Duration
  | _             -> None

type regular_obs_schedule = { start: float; step: float; end_: float }

type observation_schedule =
  | ObsAtTimes of float list
  | ObsRegular of regular_obs_schedule

(* The role of a declared file column (2026-06-10 observation data-entry §2.2):
   - [RoleTime]    — the time axis (the FIT time source).
   - [RoleDim d]   — a model dimension `d`; values bind to its levels.
   - [RoleValue k] — an observed value of DSL type `k` (count/real/…). *)
type obs_column_role =
  | RoleTime
  | RoleDim   of string
  | RoleValue of param_kind

type obs_column = {
  col_name: string;
  col_role: obs_column_role;
}

type observation_model = {
  name:          string;
  (* `from <label>` — the data SOURCE key the Rust loader binds a file to
     (`--data label=file`); defaults to [name]. Named [obs_source] (not
     [source]) to avoid colliding with [table.source : table_source]; the IR
     JSON key is "source". *)
  obs_source:    string;
  (* The explicit file schema (`columns { name : role }`). The Rust loader
     binds the file's columns by these names. *)
  columns:       obs_column list;
  (* The `~` LHS — the declared value column the likelihood scores. *)
  scored:        string;
  (* SIMULATE-only emission cadence (`emit_schedule`); the fit path reads the
     data file's time column and never consults this. [None] for a fit-only
     model that omits it. *)
  emit_schedule: observation_schedule option;
  (* For a stratified observation stream (`cases[p in patch] ~ ...`), the
     (dimension, level) pairs identifying which stratum cell this expanded
     leaf observes — e.g. [("patch", "p1")]. Empty ([]) for an unstratified
     stream. The Rust long-form loader routes each data-file row to the leaf
     whose [stratum] matches the row's `: dim` column values BY NAME. *)
  stratum:       (string * string) list;
  projection:    projection;
  (* ∂projection/∂compartment for a [DerivedExpr] (nonlinear) projection — the
     WrtPop differentiation the ODE observation gradient's factor-2 chain consumes
     (gh#275 §1h). Empty for a linear projection (CurrentPop*/CumulativeFlow* — a
     trivial selection). The [rate_state_grad] analogue for a projection. *)
  projection_state_grad: (string * deriv_entry) list;
  likelihood:    likelihood;
}

(* ── Parameters ──────────────────────────────────────────────────────────────── *)

type uniform_prior    = { lower: float; upper: float }
type normal_prior     = { mean: float; sd: float }
type log_normal_prior = { mu: float; sigma: float }
type half_normal_prior = { sigma: float }
type beta_prior       = { alpha: float; beta: float }
type gamma_prior      = { shape: float; rate: float }
type exponential_prior = { rate: float }
type log_uniform_prior = { lu_lower: float; lu_upper: float }
type truncated_normal_prior =
  { tn_mean: float; tn_sd: float; tn_lower: float; tn_upper: float }

type prior_dist =
  | Uniform     of uniform_prior
  | Normal_p    of normal_prior
  | LogNormal   of log_normal_prior
  | HalfNormal  of half_normal_prior
  | Beta        of beta_prior
  | Gamma       of gamma_prior
  | Exponential of exponential_prior
  | LogUniform      of log_uniform_prior
  | TruncatedNormal of truncated_normal_prior
  | Fixed       of float

type transform = Log | Logit | Identity

(** Distribution family for a hierarchical (pooled) prior leaf.
    Mirrors [prior_dist] variants but excludes Fixed (no meaning for
    a hierarchically-parameterised prior). Serialises to/from the same
    snake_case strings used in [prior_dist]: "uniform", "normal",
    "log_normal", "half_normal", "beta", "gamma", "exponential". *)
type hierarchical_kind =
  | HkUniform
  | HkNormal
  | HkLogNormal
  | HkHalfNormal
  | HkBeta
  | HkGamma
  | HkExponential

let hierarchical_kind_of_name = function
  | "uniform"     -> HkUniform
  | "normal"      -> HkNormal
  | "log_normal"  -> HkLogNormal
  | "half_normal" -> HkHalfNormal
  | "beta"        -> HkBeta
  | "gamma"       -> HkGamma
  | "exponential" -> HkExponential
  | s -> failwith (Printf.sprintf "unknown hierarchical kind '%s'" s)

let hierarchical_kind_name = function
  | HkUniform     -> "uniform"
  | HkNormal      -> "normal"
  | HkLogNormal   -> "log_normal"
  | HkHalfNormal  -> "half_normal"
  | HkBeta        -> "beta"
  | HkGamma       -> "gamma"
  | HkExponential -> "exponential"

(** Hierarchical prior (wave 2 / malaria #3). When a parameter's prior
    references other parameters, we can't fold the args down to floats
    — they're evaluated at inference time against the current
    hyperparameter values. [hkind] is the distribution family (typed
    enum). [hargs] are keyword → expression pairs (e.g.
    [("mu", Param "mu_h"), ("sigma", Param "sigma_h")]). [hpool_over]
    is the dimension name from the `| age` pooling clause — empty string
    when the leaf is a flat scalar with hyperparent references (no
    indexed pooling). *)
type hierarchical_prior = {
  hkind:       hierarchical_kind;
  hargs:       (string * expr) list;
  hpool_over:  string;
}

(** The prior on an *estimated* parameter (gh#191). Collapses the former
    [prior: prior_dist option] + [hierarchical: hierarchical_prior option]
    (declared "mutually exclusive" by comment) into one slot: both-set is
    unrepresentable and the ambiguous both-[None] becomes the explicit [Flat].
    JSON: [Flat] → ["flat"]; [Dist d] → [{"dist": <prior_dist>}];
    [Hierarchical h] → [{"hierarchical": <hierarchical_prior>}]. *)
type prior_spec =
  | Flat
  | Dist         of prior_dist
  | Hierarchical of hierarchical_prior

(** The three real meanings the former [value: float option] conflated
    (gh#191). Inference config ([est_init]/[est_bounds]/[est_prior]/
    [est_transform]) exists *only* on [Estimated]. JSON: internally tagged on
    ["mode"] — [{"mode":"fixed","value":_}],
    [{"mode":"estimated","bounds":_,"prior":_,"transform":_}],
    [{"mode":"required"}]. *)
type param_value =
  | Fixed     of float
  | Estimated of {
      est_init:      float option;
      est_bounds:    (float * float) option;
      est_prior:     prior_spec;
      est_transform: transform;
    }
  | Required

(* A `#'` doc block carried into the IR for non-OCaml consumers (e.g. a fit
   report's parameter table). Presentation metadata only: excluded from the
   content-addressed [run_id], and omitted from serialization when empty. *)
type doc = {
  text:      string option;   (* joined prose description *)
  symbol:    string option;   (* @symbol — display label for plots / reports *)
  reference: string option;   (* @ref — citation for the definition *)
}

(* The model's `#'` documentation dictionary: base declaration name → doc, by
   category. Built once from the source declarations and serialized at the IR
   *envelope* level (not the model body) — it is presentation metadata, outside
   the content-addressed [run_id], and the single home for every entity's doc.
   A downstream consumer (a sidecar, a plot, a report) labels any output column
   by joining its name against this index. *)
type doc_index = {
  (* The model's own doc — the file-header `#'` block (gh#750): what the model
     is, what it is fitted to, what it branches from. [None] when the file has
     no header block. Unlike [model.description], which sits in the hashed model
     body, this is envelope metadata and can be corrected without re-keying. *)
  di_model:        doc option;
  di_parameters:   (string * doc) list;
  di_compartments: (string * doc) list;
  di_transitions:  (string * doc) list;
  di_observations: (string * doc) list;
  di_dimensions:   (string * doc) list;
  di_quantities:   (string * doc) list;
}

let empty_doc_index = {
  di_model = None;
  di_parameters = []; di_compartments = []; di_transitions = [];
  di_observations = []; di_dimensions = []; di_quantities = [];
}

type parameter = {
  name:          string;
  value:         param_value;
  param_kind:    param_kind option;  (* DSL type keyword; see [param_kind] above *)
  param_dim:     (int * int) option;  (* explicit dimension annotation: (P exponent, T exponent) *)
  (* Parameter `#'` documentation lives in the model's [doc_index] (serialized at
     the envelope level), not here — it is presentation metadata, kept out of the
     computational record and out of [run_id]. *)
}

(* Accessors recovering the former flat [parameter] fields from [value].
   Inference config exists only on [Estimated]; these return [None] for
   [Fixed]/[Required]. [param_concrete_value] is the former [value] (the
   number present iff [Fixed]). *)
let param_concrete_value (p : parameter) : float option =
  match p.value with Fixed v -> Some v | Estimated _ | Required -> None

let param_bounds (p : parameter) : (float * float) option =
  match p.value with Estimated e -> e.est_bounds | _ -> None

let param_initial_value (p : parameter) : float option =
  match p.value with Estimated e -> e.est_init | _ -> None

let param_transform (p : parameter) : transform option =
  match p.value with Estimated e -> Some e.est_transform | _ -> None

let param_prior_dist (p : parameter) : prior_dist option =
  match p.value with Estimated { est_prior = Dist d; _ } -> Some d | _ -> None

let param_hierarchical (p : parameter) : hierarchical_prior option =
  match p.value with Estimated { est_prior = Hierarchical h; _ } -> Some h | _ -> None

(* ── Initial conditions ──────────────────────────────────────────────────────── *)

(** A law an INTEGER compartment's initial count is drawn from.

    The argument records are the OBSERVATION ones ([poisson_likelihood] and
    friends), reused deliberately — but NOT the [likelihood] variant, which
    would make `bernoulli` or `beta` on a count compartment representable.
    Each argument is a [diffable]: the expression together with its
    per-parameter classified ∂arg/∂θ from the autodiff pass, which is what
    makes a law's density gradient correct by construction rather than by hand.
    [binomial_likelihood.n] is a bare expr — θ-independent, no gradient. *)
type init_count_law =
  | InitPoisson     of poisson_likelihood
  | InitBinomial    of binomial_likelihood
  | InitNegBinomial of neg_binomial_likelihood

(** A law a REAL compartment's initial value is drawn from. Two enums rather
    than one, so the compartment kind selects the admissible set at
    construction: a continuous law seeds no count and a count law seeds no
    reservoir. *)
type init_real_law =
  | InitNormal of normal_likelihood

(** What one compartment's initial value is.

    [Deterministic e] is `S = N0 - I`: an expression over constants, parameters
    and other compartments' initial values. Whether [e] happens to be constant
    is a runtime build detail, not a distinction the IR draws.

    [InitCount l] is `I ~ poisson(rate = I0)` — the compartment is DRAWN from
    [l] at t_start, once per particle. [InitReal l] is the same for a real
    compartment. *)
type init_spec =
  | Deterministic of expr
  | InitCount     of init_count_law
  | InitReal      of init_real_law

(** One spec per compartment, in declaration order.

    Ordered: the runtime evaluates the entries in DEPENDENCY order (an entry may
    read another compartment's initial value), and the declaration order is part
    of the model's identity. The association list carries that order; the Rust
    side is an [IndexMap] over the same JSON object. *)
type initial_conditions = (string * init_spec) list

(** Every expression a spec evaluates, in argument order.

    Dependency extraction walks ALL of them, not just the mean: a
    `binomial(n = N0 - R, p = q)` reads `R`'s seeded value through `n`, so `n`
    is a real edge in the init DAG. *)
let init_spec_exprs (s : init_spec) : expr list =
  match s with
  | Deterministic e -> [e]
  | InitCount (InitPoisson l) -> [l.rate.expr]
  | InitCount (InitBinomial l) -> [l.n; l.p.expr]
  | InitCount (InitNegBinomial l) -> [l.mean.expr; l.dispersion.expr]
  | InitReal (InitNormal l) -> [l.mean.expr; l.sd.expr]

(** The expression whose value is this compartment's MEAN initial value — what
    the deterministic paths (the ODE skeleton, render, the pre-flight state)
    start from, and what the ∂init/∂θ forward-sensitivity seed [ic_grad]
    differentiates. `binomial(n, p)` has mean `n * p`, which is not a
    subexpression of the spec, so this constructs rather than selects. *)
let init_spec_mean_expr (s : init_spec) : expr =
  match s with
  | Deterministic e -> e
  | InitCount (InitPoisson l) -> l.rate.expr
  | InitCount (InitBinomial l) -> BinOp { op = Mul; left = l.n; right = l.p.expr }
  | InitCount (InitNegBinomial l) -> l.mean.expr
  | InitReal (InitNormal l) -> l.mean.expr

(** The distribution keyword, or [None] for a deterministic entry. *)
let init_spec_law_name (s : init_spec) : string option =
  match s with
  | Deterministic _ -> None
  | InitCount (InitPoisson _) -> Some "poisson"
  | InitCount (InitBinomial _) -> Some "binomial"
  | InitCount (InitNegBinomial _) -> Some "neg_binomial"
  | InitReal (InitNormal _) -> Some "normal"

(* ── Output ──────────────────────────────────────────────────────────────────── *)

type regular_output_schedule = { start: float; step: float }

type output_schedule =
  | OutRegular          of regular_output_schedule
  | OutAtTimes          of float list

type output_config = {
  times:        output_schedule;
  format:       string;
  trajectory:   bool;
  observations: bool;
}

(* ── Simulation config ───────────────────────────────────────────────────────── *)

(* gh#166: ODE integrator. Rk45 carries its dimensionless adaptive tolerances, so
   the orphan state (tolerances without rk45) is unrepresentable. *)
type integrator =
  | Rk4
  | Rk45 of { atol: float option; rtol: float option }

type simulation_config = {
  t_start:        float;
  t_end:          float;
  time_semantics: string;
  dt:             float option;
  rng_seed:       int option;
  integrator:     integrator;
  (* gh#616: `simulate { to = last_obs + 4 'weeks }`. When present, `t_end`
     carries NO usable horizon — the compiler bakes NaN there so every
     equality-based horizon guard fails closed — and this field IS the
     unresolved marker: the Rust resolver substitutes the resolved time into
     `t_end` and CLEARS this, and `CompiledModel::new` refuses any model that
     still carries it. `None` for a literal horizon. *)
  t_end_anchor:   anchored_time option;
}

(* ── Presets (named parameter sets for web UI / CLI) ─────────────────────────── *)

type preset = {
  preset_name    : string;
  preset_label   : string;
  preset_params  : (string * float) list;
  preset_enable  : string list;
  preset_disable : string list;
  preset_scale   : (string * float) list;
  preset_compose : string list;
  preset_t_end   : float option;
  (* gh#616: a preset's own anchored horizon (`scenarios { s { simulate { to =
     last_obs + 8 'weeks } } }`). Same contract as `simulation_config`'s:
     `preset_t_end` is NaN while this is present, and the resolver clears it. *)
  preset_t_end_anchor : anchored_time option;
}

(* ── Model structure ─────────────────────────────────────────────────────────── *)

type dimension = {
  dim_name  : string;
  dim_values: string list;
}

type model_structure = {
  dimensions              : dimension list;
  compartment_dims        : (string * string list) list; (* base → [dim_name, ...] *)
  base_compartments       : string list;
  transmission_transitions: string list;
  infectious_compartments : string list; (* base names of source_compartment in transmission transitions *)
}

(* ── Balance constraint ──────────────────────────────────────────────────────── *)

type balance_spec = {
  balance_target: string;
  balance_expr:   expr;
}

(* ── Top-level model ─────────────────────────────────────────────────────────── *)

(* A model-level shared binding (Fix B): a named value (e.g. N[l], I_agg[l],
   spatial force F[l]) referenced by Expr.BindingRef, defined once instead of
   inlined into every (patch,age) rate. Topologically ordered — a binding's body
   may reference earlier bindings. Evaluated on-demand in B-inc1; a later
   increment may cache values per step. *)
type binding = {
  bname: string;
  bexpr: expr;
}

(* ── Generated quantities (proposal 2026-06-25) ──────────────────────────────
   Named reductions of what a simulation produces — the non-scored twin of an
   observation. Mirror of rust/crates/ir/src/quantity.rs; the wire format is
   pinned there. Stratum is (dim, level) pairs, identical to observation_model.
   A quantity's *state expression* reuses the shared [expr]; reduction
   arithmetic uses the dedicated [scalar_expr] ADT (the trigger_expr precedent)
   so a reduced scalar can never appear in a propensity, and a rate leaf can
   never appear in reduction arithmetic. *)

(* A reference to an earlier *scalar* quantity, carrying the stratum it resolves
   in (populated per-cell by the expander, like stratified observations).
   `qref_stratum` is omitted on the wire when empty. *)
type qref = {
  qref_name:    string;
  qref_stratum: (string * string) list;  (* (dim, level) cells *)
}

(* Reduction-arithmetic expression: closed, total, scalar-valued. Externally
   tagged single-key objects; `op` reuses the shared bin_op/un_op encoding. *)
type scalar_expr =
  | SConst of float
  | SParam of string
  | SQRef  of qref
  | SUnOp  of { op : un_op;  arg : scalar_expr }
  | SBinOp of { op : bin_op; left : scalar_expr; right : scalar_expr }
  | SCond  of { pred : scalar_expr; then_ : scalar_expr; else_ : scalar_expr }

(* When a `value_at` reads its series: a constant time expression, or a symbolic
   observation anchor (± a compile-folded constant offset) resolved at evaluation
   time (the compiled model stays data-independent; a data-free context
   hard-errors at evaluator construction). Deliberately NOT `expr` alone: an
   anchor must be unrepresentable outside its three granted positions, so it
   cannot leak into dynamics. *)
type time_anchor =
  | ATime of expr
  | AObs  of anchored_time

(* A reduction whose result has the same dimension as the series. *)
type value_reduce =
  | VFinal
  | VMax
  | VMin
  | VMean
  | VCountAbove of expr
  | VCountBelow of expr
  (* The series value at the last output time <= the anchor (LOCF); censored
     outside the trajectory window — never clamped (proposal 2026-08-17). *)
  | VValueAt of time_anchor

(* A reduction whose result is a *time* (dimension T). *)
type time_reduce =
  | TimeOfMax
  | TimeOfMin
  | FirstAbove of expr
  | FirstBelow of expr
  | LastAbove  of expr
  | LastBelow  of expr

(* A reduction over *time*: Value preserves the series dimension, Time yields a
   time, Integral the trapezoidal area (person-time). *)
type temporal_reduce =
  | RValue of value_reduce
  | RTime  of time_reduce
  | RIntegral

(* What a Reduced quantity folds over. Externally tagged; the v1.1
   `Observation { stream }` variant appends additively. *)
type quantity_source =
  | QSState of expr
  | QSObservation of string   (* v1.1: observations.<stream> — reduces y_sim *)

(* Either a reduction of a source series, or reduction arithmetic over earlier
   scalar quantities. `reduce = None` ⇒ a series; Some ⇒ a scalar. *)
type quantity_body =
  | QBReduced of { source : quantity_source; reduce : temporal_reduce option }
  | QBDerived of scalar_expr

(* The non-scored twin of an observation_model: a named reduction of a
   simulation output, with no likelihood. Stratified quantities are fully
   expanded (one leaf per cell, tagged with `q_stratum`). *)
type quantity = {
  q_name:    string;
  q_stratum: (string * string) list;  (* (dim, level) cells; omitted when empty *)
  q_body:    quantity_body;
  (* Resolved dimension of the reduced value, as (P exponent, T exponent)
     (prerequisite #5 of the counterfactual-contrasts proposal). Computed by
     `dimcheck` and stored so the Rust contrast reducer can check operand-
     dimension agreement without re-deriving. `None` when the dimension is
     undetermined (an unconstrained `positive`/`real` leaf); omitted on the wire
     when absent, so non-quantity-dimension goldens are byte-identical. *)
  q_dimension: (int * int) option;
}

(* ── Counterfactual contrasts (proposal 2026-06-25) ────────────────────────────
   A named difference of two run-rooted operands (cases averted). The DSL block
   parses + dim-checks + lowers here; the Rust two-arm replay reducer (stage C)
   evaluates it against a fit's keyed (θ, X) output. Reporting-only and
   non-identity — excluded from the run-id walk, like quantities. *)

(* The two symmetric sub-namespaces of a run member. *)
type run_namespace = NsQuantities | NsObservations

(* A contrast body: arithmetic over run-rooted operands. A dedicated ADT
   (the trigger_expr / scalar_expr precedent) so a run-member reference can
   never appear in a propensity, and a rate leaf can never appear in a
   contrast. *)
type contrast_expr =
  | CRunMember of { run : string; ns : run_namespace; member : string }
  | CBinOp     of { op : bin_op; left : contrast_expr; right : contrast_expr }

type contrast = {
  c_name:   string;
  c_body:   contrast_expr;
}

type model = {
  name:               string;
  version:            string;
  time_unit:          string;           (* declared time unit, e.g. "days" *)
  description:        string option;
  origin:             string option;    (* ISO date string, e.g. "2020-01-01" *)
  origin_rata_die:    int option;       (* compiler-derived proleptic-Gregorian
                                           day number of `origin`; the runtime
                                           reads this so it never re-parses the
                                           origin string (2026-05-22 §6.2). *)
  compartments:       compartment list;
  transitions:        transition list;
  ode_equations:      ode_equation list;
  time_functions:     time_function list;
  tables:             table list;
  interventions:      intervention list;
  observations:       observation_model list;
  parameters:         parameter list;
  bindings:           binding list;       (* Fix B: shared per-coordinate bindings, topo-ordered *)
  per_eval_bindings:  binding list;       (* gh#272 LICM: param/table-only loop-invariant bindings,
                                             topo-ordered; produced by the LICM pass, empty by default *)
  initial_conditions: initial_conditions;
  ic_grad:            (string * (string * deriv_entry) list) list;
                      (** ∂(initial_state)/∂θ per parameterized IC compartment — the
                          forward-sensitivity seed [S(t_start)] for the ODE gradient
                          (gh#275). Keyed [compartment -> (param -> deriv_entry)];
                          the [rate_grad] analogue for the init map, empty until the
                          [WrtParam]-over-init pass emits it. *)
  output:             output_config;
  simulation:         simulation_config;
  presets:            preset list;
  model_structure:    model_structure option;
  balance:            balance_spec option;
  (* Compartments whose individuals carry tracked IDs, computed by
     forward reachability from {lineage-event destinations ∪ parent
     pools} closed under transitions (2026-05-19 proposal, §Identity-
     tracked subgraph). Empty when no `#[lineage]` annotations exist —
     in that case the lineage subsystem is statically inert. Cached
     here so the runtime does not recompute it. *)
  identity_tracked_compartments: string list;
  (* `#'` documentation dictionary (presentation metadata). Serialized at the
     envelope level, never in the model body, so it stays outside [run_id]. *)
  doc_index:          doc_index;
  (* Generated quantities (proposal 2026-06-25): named, non-scored reductions of
     simulation output. Reporting-only, fully expanded. Empty by default; the
     frontend does not yet produce these. *)
  quantities: quantity list;
  (* Counterfactual contrasts (proposal 2026-06-25): named differences of two
     run-rooted operands (cases averted). Reporting-only and non-identity (out
     of the run-id walk, like quantities). Empty by default. *)
  contrasts: contrast list;
}
