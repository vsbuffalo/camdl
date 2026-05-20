(* IR type definitions — mirror of rust/crates/ir/src/ *)
[@@@warning "-30-50"]  (* allow duplicate record field names; suppress docstring warning *)

(* ── Expression ─────────────────────────────────────────────────────────────── *)

type bin_op = Add | Sub | Mul | Div | Pow | Mod | Min | Max | Eq | Neq | Lt | Gt | Le | Ge

type un_op = Neg | Exp | Log | Sqrt | Abs | Floor | Ceil
           | Sin | Cos | Tanh                    (* gh#58 *)

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
  | Projected                    (* refers to projection output in likelihoods *)
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
  | DrawOverdispersed of expr
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
  rate_grad:       (string * expr) list;  (** ∂rate/∂param for each estimated param. Empty if not computed. *)
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

type time_function = {
  name: string;
  kind: time_func_kind;
  (* Required declared dimension from the forcing's tier-3 unit literal
     (e.g. `'count`, `'per_year`, `'ratio`). The dim-checker uses this
     authoritatively. The expander has already applied the scale
     factor to stored values so runtime interpolation returns values
     in the model's `time_unit`. GH #8. *)
  dim:  int * int;
}

(* ── Tables ──────────────────────────────────────────────────────────────────── *)

type oob_policy = Clamp | Wrap | Error

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
  | AtTimes   of float list
  | Recurring of recurring_schedule
  | External  of string

type fraction_transfer = { src: string; dst: string; fraction: expr }
type absolute_transfer = { src: string; dst: string; count: expr }
type set_action        = { compartment: string; value: expr }
type add_action        = { add_compartment: string; add_count: expr }

type action =
  | FractionTransfer of fraction_transfer
  | AbsoluteTransfer of absolute_transfer
  | Set              of set_action
  | AddAction        of add_action

type intervention = {
  name:          string;
  base_name:     string option;
  schedule:      intervention_schedule;
  actions:       action list;
  always_active: bool;
}

(* ── Observation model ───────────────────────────────────────────────────────── *)

type projection =
  | CumulativeFlow of string
  | CurrentPop     of string
  | CurrentPopSum  of string list
  | DerivedExpr    of expr

type poisson_likelihood      = { rate:       expr }
type neg_binomial_likelihood = { mean: expr; dispersion: expr }
type normal_likelihood       = { mean: expr; sd: expr }
type binomial_likelihood     = { n:    expr; p:  expr }
type beta_binomial_likelihood = { n: expr; alpha: expr; beta: expr }
type bernoulli_likelihood    = { p: expr }

type likelihood =
  | Poisson      of poisson_likelihood
  | NegBinomial  of neg_binomial_likelihood
  | Normal       of normal_likelihood
  | Binomial     of binomial_likelihood
  | BetaBinomial of beta_binomial_likelihood
  | Bernoulli    of bernoulli_likelihood

type regular_obs_schedule = { start: float; step: float; end_: float }

type observation_schedule =
  | ObsAtTimes of float list
  | ObsRegular of regular_obs_schedule
  | ObsFromData

type observation_model = {
  name:        string;
  data_stream: string;
  schedule:    observation_schedule;
  projection:  projection;
  likelihood:  likelihood;
}

(* ── Parameters ──────────────────────────────────────────────────────────────── *)

type uniform_prior    = { lower: float; upper: float }
type normal_prior     = { mean: float; sd: float }
type log_normal_prior = { mu: float; sigma: float }
type half_normal_prior = { sigma: float }
type beta_prior       = { alpha: float; beta: float }
type gamma_prior      = { shape: float; rate: float }
type exponential_prior = { rate: float }

type prior_dist =
  | Uniform     of uniform_prior
  | Normal_p    of normal_prior
  | LogNormal   of log_normal_prior
  | HalfNormal  of half_normal_prior
  | Beta        of beta_prior
  | Gamma       of gamma_prior
  | Exponential of exponential_prior
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

type parameter = {
  name:          string;
  value:         float option;  (* None = must be supplied at runtime via --params / --set *)
  bounds:        (float * float) option;  (* optional [lo, hi] constraint for inference/validation *)
  prior:         prior_dist option;
  hierarchical:  hierarchical_prior option;  (* Some iff a leaf in a hierarchical pool; mutually exclusive with prior. *)
  transform:     transform option;
  initial_value: float option;
  param_kind:    string option;  (* DSL type: "rate", "probability", "positive", "count", "real" *)
  param_dim:     (int * int) option;  (* explicit dimension annotation: (P exponent, T exponent) *)
}

(* ── Initial conditions ──────────────────────────────────────────────────────── *)

type initial_conditions =
  | Explicit        of (string * float) list
  | Parameterized   of (string * expr)  list
  | FromDistribution of (string * prior_dist) list

(* ── Output ──────────────────────────────────────────────────────────────────── *)

type regular_output_schedule = { start: float; step: float; end_: float }

type output_schedule =
  | OutRegular          of regular_output_schedule
  | OutAtTimes          of float list
  | OutMatchObservations

type output_config = {
  times:        output_schedule;
  format:       string;
  trajectory:   bool;
  observations: bool;
}

(* ── Simulation config ───────────────────────────────────────────────────────── *)

type simulation_config = {
  t_start:        float;
  t_end:          float;
  time_semantics: string;
  dt:             float option;
  rng_seed:       int option;
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

type model = {
  name:               string;
  version:            string;
  time_unit:          string;           (* declared time unit, e.g. "days" *)
  description:        string option;
  origin:             string option;    (* ISO date string, e.g. "2020-01-01" *)
  compartments:       compartment list;
  transitions:        transition list;
  ode_equations:      ode_equation list;
  time_functions:     time_function list;
  tables:             table list;
  interventions:      intervention list;
  observations:       observation_model list;
  parameters:         parameter list;
  initial_conditions: initial_conditions;
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
}
