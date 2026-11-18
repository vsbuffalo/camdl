(** The intermediate representation (IR): a camdl model after compilation, as
    [camdlc] writes it to JSON for the Rust runtime.

    The IR is fully expanded. Every stratified declaration has become one
    entry per cell, named by joining the base name to its levels with [_]
    ([S] over [patch = [urban, rural]] becomes [S_urban] and [S_rural]), and
    every name an expression mentions refers to an entry of the {!model}.
    Times, durations and rates are numbers in the model's [time_unit]; the
    compiler has already converted every unit literal into it. The JSON codec
    is [Serde]; the Rust types in [rust/crates/ir/src/] read the same JSON,
    and [ir/schema.json] is the contract between the two. *)

[@@@warning "-30-50"]  (* allow duplicate record field names; suppress docstring warning *)

(* ── Expression ─────────────────────────────────────────────────────────────── *)

(** A binary operator of an expression.

    The comparisons [Eq] to [Ge] evaluate to [1.0] when they hold and [0.0]
    when they do not. [Mod] is Euclidean: [a mod b] is at least 0 and less than [|b|]. A
    division or [Mod] by zero, or a [Pow] whose result is not finite,
    evaluates to NaN, which the runtime reports as a numerical error (or
    treats as [0.0] under [--allow-degenerate-rates]). *)
type bin_op = Add | Sub | Mul | Div | Pow | Mod | Min | Max | Eq | Neq | Lt | Gt | Le | Ge

(** A unary function of an expression. [Log] of a number that is not positive
    and [Sqrt] of a negative number evaluate to NaN, handled as for
    {!bin_op}. *)
type un_op = Neg | Exp | Log | Sqrt | Abs | Floor | Ceil
           | Sin | Cos | Tanh

(* ── Observation anchors ────────────────────────────────────────────────────── *)

(** Which end of a run's observation record an anchor names: written
    [first_obs] and [last_obs] in a model.

    An anchor is resolved per run, from the observation streams that run binds
    to data, so two runs of one compiled model can resolve it to different
    times. *)
type obs_anchor =
  | AnchorFirst  (** The earliest observation time over the run's bound streams. *)
  | AnchorLast   (** The latest observation time over the run's bound streams. *)

(** A time stated relative to the observed data, such as
    [last_obs + 4 'weeks]: an anchor whose value a run supplies, plus a
    constant offset the compiler has already converted to the model's time
    unit.

    It serialises as the bare anchor name when the offset is zero, and as an
    object otherwise. In a days model, [last_obs + 4 'weeks] is

    {@json[
      {"anchor": "last_obs", "offset": 28.0}
    ]} *)
type anchored_time = {
  anchor: obs_anchor;
  offset: float;   (** Signed, in model time units. *)
}

(** The operands of a {!bin_op}. *)
type bin_op_expr = { op: bin_op; left: expr; right: expr }

(** The operand of a {!un_op}. *)
and un_op_expr   = { op: un_op;  arg:  expr }

(** A conditional, written [if pred then a else b]: [then_] when [pred]
    evaluates to a positive number, [else_] otherwise. *)
and cond_expr    = { pred: expr; then_: expr; else_: expr }

(** A numeric expression: a transition rate, an initial value, a likelihood
    argument, a derivative, or any other quantity the runtime evaluates.

    The language is pure and first order: no variable binding, no recursion,
    so an expression evaluates in bounded time. Every name it mentions is
    already expanded and resolved. The rates of a model fragment such as

    {@camdl[
      let omega : rate = 0.01
      let N[p in patch] = S[p] + I[p] + R[p]
      transitions {
        infection[p in patch] : S[p] --> I[p] @ beta * season * S[p] * sum(q in patch, C[p, q] * I[q] / N[q])
        waning[p in patch]    : R[p] --> S[p] @ omega * R[p]
        import[p in patch]    : --> I[p] @ (if t < 30 'days then 0.001 'per_day else 0.0 'per_day) * N[p]
      }
    ]}

    use [Param "beta"], [TimeFunc "season"], [Pop "S_a"], a [Reduce] over
    [q] with the table cells folded to [Const 0.9] and [Const 0.1],
    [BindingRef "N_a"], [ConstRef "omega"], and a [Cond] on [Time] whose
    branches are [UnitLiteral]s. *)
and expr =
  | Const  of float
      (** A number: a literal, or a value the compiler computed. *)
  | Param  of string
      (** The value of a model parameter, named by an entry of
          [model.parameters]. *)
  | ConstRef of string
      (** The value of a model constant, a typed [let] with a constant body
          such as [let omega : rate = 0.01], named by an entry of
          [model.constants]. It is not a parameter: nothing can set or
          estimate it, and its derivative with respect to every parameter is
          zero. It stays a reference rather than its number so the dimension
          checker sees the constant's declared kind and displays keep its
          name. *)
  | Pop    of string
      (** The current value of one compartment: a count for an integer
          compartment, a real number for a real one. *)
  | PopSum of string list
      (** The sum of the current values of several compartments. A bare
          stratified compartment name in a rate, or a [let] such as
          [N = S + I + R], lowers to it. *)
  | Time
      (** The current model time, in the model's time unit, measured from the
          model's [origin] when it declares one. *)
  | Dt
      (** The length of the current step, in model time units. Only the
          stepping backends (chain-binomial and the ODE integrators) give it a
          value; the Gillespie backend refuses a model whose rates read it,
          and an initial condition reads [0.0]. *)
  | BinOp  of bin_op_expr
  | UnOp   of un_op_expr
  | Cond   of cond_expr
  | TimeFunc of string           (** The value of a forcing, named by an entry of [model.time_functions], at the current time shifted back by the forcing's [lag]. *)
  | TableLookup of string * expr list
      (** One cell of a table, named by an entry of [model.tables], at a
          0-based index into its values. The list always holds exactly one
          index: the compiler flattens a multi-dimensional lookup [C[p, q]]
          into one constant row-major index, and the runtime refuses any other
          length. The index is rounded down, and one outside the table is a
          runtime error. Where the cell is a literal number the compiler
          writes the cell itself, as a [Const], in place of the lookup. *)
  | Reduce of expr list
      (** The sum of the terms, added left to right. A [sum(q in patch, ...)]
          over a dimension lowers to it, one term per level. *)
  | BindingRef of string
      (** The value of a shared binding, named by an entry of
          [model.bindings]: a [let] that several rates read, such as
          [N_urban], defined once instead of copied into each rate. *)
  | PerEvalRef of string
      (** The value of a per-evaluation binding, named by an entry of
          [model.per_eval_bindings]: a subexpression of parameters, constants
          and tables only, which the compiler lifts out of the rates so the
          runtime computes it once per parameter value rather than at every
          step. *)
  | Projected                    (** In a likelihood argument, the stream's projected value for the observation being scored: [projected] in the model. *)
  | ObsColumnRef of string
      (** In a likelihood argument, the value of a declared data column in
          the row being scored, such as [tested] in
          [cases ~ binomial(n = tested, p = rho)]. *)
  | ObsAnchor of anchored_time
      (** A time stated relative to the observed data, such as
          [last_obs - 2 'weeks]. It appears only among a forcing's knots: a
          [piecewise] forcing's [breakpoints], and a [bspline]'s [window] ends
          or [breakpoints]. The runtime substitutes a [Const] for each one once
          a run binds its data, and refuses to build a model that still
          carries one, so no evaluator ever sees it. *)
  | ForcingBasis of { forcing : string; index : int; space : basis_space }
      (** The derivative of a spline forcing with respect to one of its
          coefficients: basis function [index] (0-based) in [space] of the
          forcing named [forcing], at the forcing's own evaluation time (the
          current time minus its [lag], held to a [bspline]'s window or
          wrapped by a [periodic_spline]'s period). It appears only in
          gradient maps; a model cannot write it. *)
  | UncheckedDim of unchecked_dim_expr
      (** [unchecked_dim(e, dim = ..., reason = ...)]: the value of [e], with a
          dimension the author asserts and the dimension checker does not
          verify. *)
  | UnitLiteral of { value : float; dim_p : int; dim_t : int; unit : string }
      (** A literal written with a unit, such as [0.2 'per_day]. [value] is
          already in the model's time unit, [(dim_p, dim_t)] is its dimension
          as exponents of population and time, and [unit] is the unit as
          written. In a days model [20 'weeks] has [value = 140.0] and
          [unit = "weeks"]. It evaluates to [value], and the dimension checker
          takes its dimension as fact. *)

(** Which function a {!expr.ForcingBasis} node evaluates: [Space_spline] the
    [index]-th B-spline basis function; [Space_whitened] the [index]-th
    whitened curvature mode of a [smooth] coefficient form, divided by its
    scale constant κ; [Space_linear] the slope direction [t - t_m], with [t_m]
    the middle of the window and [index] 0. Only a [bspline] whose
    coefficients are [smooth] has the whitened and linear spaces. *)
and basis_space = Space_spline | Space_whitened | Space_linear

(** The operand and asserted dimension of an [unchecked_dim(...)]. *)
and unchecked_dim_expr = {
  inner:  expr;
  dim_p:  int;     (** Asserted exponent of population. *)
  dim_t:  int;     (** Asserted exponent of time. *)
  reason: string;  (** The author's stated reason, kept for the record; it does not affect evaluation. *)
}

(* ── Derivative entries ───────────────────────────────────────────────────────── *)

(** Why the compiler could not write the derivative of some quantity with
    respect to a parameter (or, for [URNonsmoothState], a compartment). A
    gradient-based method refuses a model that carries one for a parameter it
    estimates; methods that use no gradient are unaffected. The wire spelling
    of each reason is {!unsupported_reason_name}. *)
type unsupported_reason =
  | URLag                 (** The parameter sets a forcing's [lag]. *)
  | URPeriodicCoeff       (** The parameter sets a [periodic] forcing's step value or period. *)
  | URStructuralForcing   (** The parameter reaches forcing data that is fixed when the model is built: [piecewise] or [interpolated] knots and values, or a spline's knots, window or period. *)
  | URNonConstTableIndex  (** The parameter reaches a table cell through an index that is not constant. *)
  | URMod                 (** The parameter enters through [mod], which has no derivative. *)
  | URParametricN         (** The parameter reaches the [n] of a binomial or beta-binomial, which must not depend on parameters. The compiler never writes this reason; the runtime's gradient check produces it. *)
  | URNonsmoothState      (** A derivative with respect to a compartment meets [floor], [ceil], [abs], [min] or [max] of a compartment, where it is not smooth. *)

(** One entry of a gradient map: a derivative, or a refusal to give one. A
    key absent from the map means the derivative is exactly zero.

    {@json[
      {"grad": {"param": "rho"}}
    ]} *)
type deriv_entry =
  | DEGrad of expr
      (** The derivative, as an expression. *)
  | DEUnsupported of { node : string; code : unsupported_reason }
      (** No derivative: [code] says why, and [node] names the construct for
          messages, such as ["mod expression"]. *)

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

let unsupported_reason_of_name = function
  | "lag"                   -> Some URLag
  | "periodic_coeff"        -> Some URPeriodicCoeff
  | "structural_forcing"    -> Some URStructuralForcing
  | "non_const_table_index" -> Some URNonConstTableIndex
  | "mod"                   -> Some URMod
  | "parametric_n"          -> Some URParametricN
  | "nonsmooth_state"       -> Some URNonsmoothState
  | _                       -> None

(* ── Compartment ─────────────────────────────────────────────────────────────── *)

(** What a compartment holds: [Integer], the default, a whole number of
    individuals; [Real], declared [W : real], a continuous quantity such as a
    pathogen reservoir evolved by an [ode] equation. *)
type compartment_kind = Integer | Real

(** One compartment of the expanded model, such as [S_urban]. *)
type compartment = {
  name: string;
  kind: compartment_kind;
}

(* ── Transition ─────────────────────────────────────────────────────────────── *)

(** One compartment's change in value each time a transition fires, such as
    [("S", -1)]. A compartment named on both sides of a transition appears
    once, with its net change, and a net change of zero is left out. *)
type stoichiometry_entry = string * int

(** A classification of a transition, for tooling; the runtime does not read
    it. The compiler gives every transition one. *)
type transition_metadata = {
  origin_kind:        string option;
      (** One of ["inflow"] (the transition has no single source
          compartment), ["outflow"] (no single destination), ["transmission"]
          (the rate reads a compartment other than the source) or
          ["intrinsic"] (it does not). The test sees only compartments the
          rate names directly: one read through a [sum(...)] or a shared
          [let] is missed, so such a transmission is labelled
          ["intrinsic"], and a transition with two sources, [A + B --> C], is
          labelled ["inflow"]. *)
  source_compartment: string option;
      (** The compartment whose value the transition lowers, when there is
          exactly one. *)
  dest_compartment:   string option;
      (** The compartment whose value the transition raises, when there is
          exactly one. *)
}

(** How a stepping backend draws the number of times a transition fires in a
    step, from the step's expected count [rate * dt]. *)
type draw_method =
  | DrawPoisson
      (** The backend's ordinary random draw: a rate written [@ rate]. *)
  | DrawOverdispersed of { sigma_sq : expr; sigma_sq_grad : (string * deriv_entry) list }
      (** [overdispersed(rate, sigma_sq)]: the rate is multiplied in each step
          by gamma-distributed noise with mean 1 and variance
          [sigma_sq / dt], so [sigma_sq] has the dimension of time. Only the
          chain-binomial backend supports it. [sigma_sq_grad] is the
          derivative of [sigma_sq] with respect to each parameter, empty when
          not computed. *)
  | DrawDeterministic
      (** [deterministic(rate)]: the transition fires [rate * dt] times,
          rounded to the nearest integer, with no noise. *)

(** How a [#[lineage]] transition chooses which existing individual caused
    each firing, so that the runtime can record a genealogy. The compiler
    refuses (E601) a [#[lineage]] transition whose rate is not linear in the
    counts of its parent compartments, including a product of two parent
    counts such as [beta * S * I * I / (N * N)], so the weights times the
    parent counts sum to the rate. *)
type transition_lineage = {
  is_lineage_event:    bool;
      (** Always [true]: the compiler writes a lineage record only for a
          [#[lineage]] transition. *)
  parent_pool_weights: (string * expr) list;
      (** The rate split over parent compartments: each pair is a parent and
          its weight, the rate with that parent's count factored out. A
          firing picks parent pool [b] with probability proportional to
          [weight_b * count_b], then an individual uniformly within it. For
          [beta * S * I / N] with parent [I] this is [[("I", beta * S / N)]]. *)
}

(** One transition of the expanded model: a stochastic flow between
    compartments at a rate, such as [infection_urban]. A transition split by
    [{ R : w_R, D : w_D }] becomes one transition per branch, its name
    suffixed by the branch. *)
type transition = {
  name:            string;
  stoichiometry:   stoichiometry_entry list;
  rate:            expr;
      (** The expected number of firings per model time unit (the
          propensity). The runtime refuses a rate that evaluates to a negative
          number, and one that is not finite unless [--allow-degenerate-rates]
          makes it zero. *)
  metadata:        transition_metadata option;
  draw_method:     draw_method;
  rate_grad:       (string * deriv_entry) list;  (** The derivative of [rate] with respect to each parameter, keyed by parameter name; empty when not computed. *)
  rate_state_grad: (string * deriv_entry) list;  (** The derivative of [rate] with respect to each compartment it reads, keyed by compartment name, which the ODE gradient needs; empty when not computed, for instance under [camdlc --no-state-grad]. *)
  lineage:         transition_lineage option;  (** Present exactly for a [#[lineage]] transition. *)
}

(* ── ODE equation ────────────────────────────────────────────────────────────── *)

(** The time derivative of one real compartment, from an [ode] block:
    [ode { W = xi * I - delta * W }] gives [compartment = "W"] and the
    derivative of [W] per model time unit. *)
type ode_equation = {
  compartment: string;
  derivative:  expr;
}

(* ── Time functions ──────────────────────────────────────────────────────────── *)

(** A [sinusoidal] forcing: [baseline + amplitude * sin(2π (t - phase) / period)].
    [phase] and [period] are in model time units; [phase] is a shift in time,
    not an angle. *)
type sinusoidal = { amplitude: expr; period: expr; phase: expr; baseline: expr }

(** A [piecewise] forcing, a step function: [values[0]] before the first
    breakpoint, then [values[i + 1]] from [breakpoints[i]] on. A breakpoint may
    be an {!expr.ObsAnchor}. *)
type piecewise  = { breakpoints: expr list; values: expr list }

(** An [interpolated] forcing through the knots [(times[i], values[i])].
    [method_] is ["linear"] (straight lines between knots, held at the end
    values outside them), ["constant"] (the value of the last knot at or
    before [t]) or ["spline"] (a cubic spline).
    A forcing declared [data = "path"] has its knots read from the file at
    compile time. *)
type interpolated = { times: expr list; values: expr list; method_: string }

(** A [periodic] forcing: [period] split into [List.length values] equal
    intervals, each holding its value, repeating forever. *)
type periodic   = { period: expr; values: expr list }

(** A [fourier] forcing:
    [Σ_k a_k cos(2π k t / period) + b_k sin(2π k t / period)], with
    [harmonics] holding [(a_k, b_k)] for [k = 1, 2, ...] in order. It has no
    constant term: a model writes [1 + f] for a modulation around 1. *)
type fourier = { period: expr; harmonics: (expr * expr) list }

(** How a spline's basis sum [g(t) = Σ c_k B_k(t)] becomes the forcing value:
    [Link_identity] gives [g], [Link_log] gives [exp g], positive whatever the
    coefficients. It serialises only when it is [Link_log]. *)
type spline_link = Link_identity | Link_log

(** A [periodic_spline] forcing: a periodic B-spline of [degree] with
    [n_basis] equally spaced knots over one [period], and one coefficient per
    basis function. *)
type periodic_spline = {
  period:  expr;
  n_basis: int;
  degree:  int;
  coefs:   expr list;
  link:    spline_link;
}

(** The knots of a [bspline] forcing. Both forms are fixed when the model is
    built: a parameter cannot move them, though an end may be an
    {!expr.ObsAnchor}. *)
type bspline_knots =
  | Knots_uniform  of { lower : expr; upper : expr; n_intervals : int }
      (** [window = [lower, upper]] cut into [n_intervals] equal intervals. *)
  | Knots_explicit of expr list
      (** [breakpoints = [...]], strictly increasing; the first and last are
          the window's ends. *)

(** The coefficients of a [bspline] forcing. *)
type spline_coefs =
  | Coefs_explicit of expr list
      (** One coefficient per basis function. *)
  | Coefs_smooth of { level : expr; slope : expr; scale : expr; modes : expr list; per : expr }
      (** [smooth(level = ..., slope = ..., scale = ..., modes = ..., per = ...)]:
          a curve with penalised curvature,
          [level + slope * (t - t_m) + (scale / κ) * M * modes] on the basis,
          where [t_m] is the middle of the window, [M] whitens the curvature
          penalty and is built by the runtime from the knots, [modes] has two
          fewer entries than there are basis functions, and
          [κ = sqrt(K - 2) * per^(3/2)] for [K] basis functions and the
          reference duration [per].

          {@camdl[
            coefs = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
          ]} *)

(** A [bspline] forcing: a clamped B-spline of [degree] over its window, held
    at its end values outside it, with [n_intervals + degree] basis
    functions. *)
type bspline = {
  knots:  bspline_knots;
  degree: int;
  coefs:  spline_coefs;
  link:   spline_link;
}

(** The shape of a forcing, one constructor per forcing type keyword. *)
type time_func_kind =
  | Sinusoidal    of sinusoidal
  | Piecewise     of piecewise
  | Interpolated  of interpolated
  | Periodic      of periodic
  | Fourier       of fourier
  | PeriodicSpline of periodic_spline
  | BSpline       of bspline

(** The file a [data = "path"] forcing's knots were read from. The compiler
    copies the knots into the forcing, so the runtime never opens the file;
    this record keeps the provenance. Neither field is part of the run's
    identity: the copied values already are, and the same bytes at another
    path, or an edit that changes no value read, are the same model. *)
type data_source = {
  path:   string;
      (** The path as written in the model, not resolved, so it stays
          comparable across machines and checkouts. *)
  sha256: string;
      (** The lowercase hexadecimal SHA-256 of the file's bytes, as
          [shasum -a 256 <path>] prints it from the model's directory. *)
}

(** One forcing of the expanded model: a function of time, named for
    {!expr.TimeFunc}, such as [seasonal_p1] for the cell [p1] of an indexed
    forcing. *)
type time_function = {
  name: string;
  kind: time_func_kind;
  dim:  int * int;
      (** The dimension of the forcing's declared unit, as exponents of
          population and time: [(0, -1)] for ['per_day], [(1, 0)] for
          ['count], [(0, 0)] for ['ratio]. Its values are already scaled to
          the model's time unit. *)
  lag:  expr option;
      (** A delay, in model time units: when present the forcing is
          evaluated at [t - lag]. It may be a parameter. *)
  data_source: data_source option;
      (** The file the knots came from, for a forcing declared
          [data = "path"]. *)
}

(* ── Tables ──────────────────────────────────────────────────────────────────── *)

(** What a lookup outside a table does. The only policy is a runtime error. *)
type oob_policy = Error

(** Where a table's values come from. *)
type table_source =
  | Inline   of expr list  (** The values in row-major order (the last index varies fastest), each an expression of constants and parameters. *)
  | External of string     (** Values supplied when the model is run, under this name, from [external("name")]; the runtime refuses to build a model whose external table was not supplied. *)

(** A table of the expanded model, read by {!expr.TableLookup}. A table over
    [patch × patch] declared as [[[0.9, 0.1], [0.1, 0.9]]] holds the four
    values [[0.9; 0.1; 0.1; 0.9]]. *)
type table = {
  name:          string;
  source:        table_source;
  out_of_bounds: oob_policy;
  cell_kind:     string option;
      (** The declared kind of every cell, from a table declared
          [aging_rate : age :rate = [...]]: a {!param_kind} keyword, as a
          string such as ["rate"]. Absent, the cells are dimensionless. *)
}

(* ── Interventions ───────────────────────────────────────────────────────────── *)

(** A recurring fire schedule, in model time units. Without [at_day] it fires
    at [start], [start + period], ... up to [end_]; with [at_day] it fires at
    [at_day + k * period] for each [k] that falls in the window. [end_] is NaN,
    serialised as [null], when it was taken from a model horizon that is still
    an unresolved {!anchored_time}. *)
type recurring_schedule = { start: float; period: float; end_: float; at_day: float option }

(** When a scheduled intervention or event fires. *)
type intervention_schedule =
  | AtTimes     of float list
      (** At these model times, from an [at [...]] list whose entries are
          all plain numbers once compiled. *)
  | AtTimesExpr of expr list
      (** At these times, each an expression the runtime evaluates against
          the parameters when a simulation starts, so a fire time can be a
          parameter: [at [t_seed]]. The compiler writes this form for any list
          with an entry that does not compile to a bare number, including a
          constant date expression such as [at [origin + 20 'weeks]]. *)
  | Recurring of recurring_schedule

(** [transfer(fraction = f, from = src, to = dst)]: moves the fraction [f] of
    [src] into [dst]. *)
type fraction_transfer = { src: string; dst: string; fraction: expr }

(** [transfer(count = n, from = src, to = dst)]: moves [n] individuals from
    [src] into [dst]. *)
type absolute_transfer = { src: string; dst: string; count: expr }

(** Sets a compartment to a value. *)
type set_action        = { compartment: string; value: expr }

(** [add(C, n)]: adds [n] to compartment [C]. *)
type add_action        = { add_compartment: string; add_count: expr }

(** One change an intervention or event makes to the state when it fires. *)
type action =
  | FractionTransfer of fraction_transfer
  | AbsoluteTransfer of absolute_transfer
  | Set              of set_action
  | AddAction        of add_action

(** Whether scenarios switch an intervention on, or it is always on. Both
    kinds fire at the times their {!fire_source} gives. *)
type intervention_kind =
  | Scenario   (** From [interventions {}] or [reactive_interventions {}]: inactive unless a scenario or [--enable] enables it. *)
  | Event      (** From [events {}]: active unless a scenario or [--disable] disables it. *)

(** The comparison of a reactive trigger. *)
type cmp_op = CmpLt | CmpLe | CmpGt | CmpGe | CmpEq | CmpNeq

(** How a reactive trigger reduces an observation stream's history to one
    number. [RedLatest], [observed(s)], is the most recent value and ignores
    the window. [RedSum], [sum_observed(s, window = w)], is the sum over the
    trailing window: observations later than [t - window] and no later than
    [t], the current time. [RedMean] and [RedMax] are the mean and maximum over that window;
    no model syntax produces them. A window that holds no observation reduces
    to [0.0]. *)
type obs_reducer = RedLatest | RedSum | RedMean | RedMax

(** The quantity a reactive trigger watches: the simulated observations of a
    stream, the data a policymaker would see, never the latent state. *)
type trigger_quantity =
  | TQObserved of { stream : string; window : float option; reducer : obs_reducer }
      (** The observations of the expanded stream [stream], over a trailing
          window in model time units ([None]: the latest observation only),
          reduced by [reducer]. *)

(** The value a reactive trigger compares against. *)
type trigger_threshold =
  | TTConst of float
  | TTParam of string  (** A parameter, by name. *)

(** The condition of a reactive intervention: a boolean combination of
    comparisons, written after [when]. It is its own type, not an {!expr}, so
    it is boolean by construction and its observation reads cannot appear in a
    rate. *)
type trigger_expr =
  | TECmp of trigger_quantity * cmp_op * trigger_threshold
  | TEAnd of trigger_expr * trigger_expr
  | TEOr  of trigger_expr * trigger_expr
  | TENot of trigger_expr

(** The condition and timing of a reactive intervention. The entry

    {@camdl[
      reactive_interventions {
        sia2 : when sum_observed(weekly_cases, window = 28 'days) >= 5 {
          after  = 21 'days
          action = transfer(count = 10, from = S, to = V)
          once   = true
        }
      }
    ]}

    applies its action 21 days after the 28-day sum of [weekly_cases] first
    reaches 5, and fires no more. *)
type reactive_trigger = {
  when_:    trigger_expr;    (** The condition, written after [when]; it serialises under ["when"]. *)
  after:    float;          (** The delay, in model time units and not negative, between the condition holding and the actions applying. *)
  once:     bool;           (** Fires at most once. The compiler refuses [once] together with a [cooldown]. *)
  cooldown: float option;   (** The least time, in model time units, between firings. *)
}

(** What decides when an intervention fires. *)
type fire_source =
  | Scheduled of intervention_schedule
  | Reactive  of reactive_trigger
      (** Fires when observed data meet a condition. Only the chain-binomial
          backend's forward simulation runs these; every other backend, and
          every inference method, refuses a model that carries one. *)

(** One intervention or event of the expanded model: actions applied to the
    state when its fire source says so. *)
type intervention = {
  name:          string;
  base_name:     string option;
      (** For a cell of an indexed intervention, the name it was declared
          under, by which scenarios enable or disable the whole family;
          [None] for an intervention declared without indices. *)
  fire:          fire_source;
  actions:       action list;
  kind:          intervention_kind;
}

(* ── Observation model ───────────────────────────────────────────────────────── *)

(** What the model predicts for an observation stream: the [projected = ...]
    line. The flow projections ([CumulativeFlow], [CumulativeFlowSum],
    [FlowRatio]) accumulate over the period each data row covers, which the
    stream's [covers] states; the others are read at the observation time. *)
type projection =
  | CumulativeFlow    of string
      (** [incidence(tr)]: how many times one transition fired in the row's
          period. *)
  | CurrentPop        of string
      (** The value of one compartment at the observation time. *)
  | CurrentPopSum     of string list
      (** The sum of several compartments at the observation time, such as a
          bare stratified compartment name. *)
  | DerivedExpr       of expr
      (** Any other expression of the state, read at the observation time,
          such as [projected = R / N]. *)
  | CumulativeFlowSum of string list
      (** The total firings of several transitions in the row's period, such
          as [sum(p in patch, incidence(infection[p]))]. *)
  | FlowRatio         of { numerator : string list; denominator : string list }
      (** The firings of the [numerator] transitions divided by those of the
          [denominator] transitions, both counted over the row's period:
          [incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))].
          Neither list is empty. It is NaN when the denominator is zero. *)

(** A derivative with respect to each parameter, keyed by parameter name. An
    absent key is an exact zero; an empty map means the derivatives were not
    computed. *)
type grad_map = (string * deriv_entry) list

(** A likelihood argument, together with its derivatives. For
    [mean = rho * projected]:

    {@json[
      {"expr": {"bin_op": {"op": "mul", "left": {"param": "rho"}, "right": {"projected": null}}},
       "grad": {"rho": {"grad": {"projected": null}}},
       "proj_grad": {"grad": {"param": "rho"}}}
    ]}

    [grad] is the derivative of [expr] with respect to each parameter;
    [proj_grad] its derivative with respect to {!expr.Projected}, [None] when
    [expr] does not depend on it. *)
type diffable = { expr: expr; grad: grad_map; proj_grad: deriv_entry option }

(** The arguments of [poisson(rate = ...)]. *)
type poisson_likelihood      = { rate: diffable }

(** The arguments of [neg_binomial(mean = ..., r = ...)]: variance
    [mean + mean² / dispersion]. *)
type neg_binomial_likelihood = { mean: diffable; dispersion: diffable }

(** The arguments of [normal(mean = ..., sd = ...)]. *)
type normal_likelihood       = { mean: diffable; sd: diffable }

(** The arguments of [binomial(n = ..., p = ...)]. [n] is rounded to an
    integer and must not depend on parameters, so it carries no derivative;
    it is often a data column, [n = tested]. *)
type binomial_likelihood     = { n: expr; p: diffable }

(** The arguments of [beta_binomial(n = ..., alpha = ..., beta = ...)], with
    [n] as for {!binomial_likelihood}. *)
type beta_binomial_likelihood = { n: expr; alpha: diffable; beta: diffable }

(** The arguments of [beta(mean = ..., concentration = ...)], for a proportion
    in [(0, 1)] reported directly rather than as [k] of [n]: shape parameters
    [mean * concentration] and [(1 - mean) * concentration]. *)
type beta_likelihood         = { mean: diffable; concentration: diffable }

(** The argument of [bernoulli(p = ...)]. *)
type bernoulli_likelihood    = { p: diffable }

(** The arguments of
    [zero_inflated(base = neg_binomial(mean = ..., r = ...), pi = ...)]: a
    zero with probability [pi], otherwise a negative binomial count, so
    [P(Y = 0) = pi + (1 - pi) f(0)] and [P(Y = k) = (1 - pi) f(k)] for
    [k > 0]. *)
type zi_neg_binomial_likelihood =
  { mean: diffable; dispersion: diffable; pi: diffable }

(** The distribution of an observation given its projected value: the
    right-hand side of [cases ~ ...]. *)
type likelihood =
  | Poisson              of poisson_likelihood
  | NegBinomial          of neg_binomial_likelihood
  | Normal               of normal_likelihood
  | Binomial             of binomial_likelihood
  | BetaBinomial         of beta_binomial_likelihood
  | Beta                 of beta_likelihood
  | Bernoulli            of bernoulli_likelihood
  | ZeroInflatedNegBinomial of zi_neg_binomial_likelihood

(** The kind keyword of a parameter, constant or data column, which fixes its
    dimension unless an explicit [dim] annotation does: [Rate] is per time,
    [Probability] dimensionless, [Count] population, [Instant] and [Duration]
    time; [Positive] and [Real] leave the dimension to be inferred. *)
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

(** [emit_schedule = every step]: from [start] to [end_], in model time units.
    The compiler sets [start] and [end_] to the model's horizon; [end_] is NaN,
    serialised as [null], while that horizon is an unresolved
    {!anchored_time}. *)
type regular_obs_schedule = { start: float; step: float; end_: float }

(** When [simulate] emits a stream's synthetic observations. *)
type observation_schedule =
  | ObsAtTimes of float list
  | ObsRegular of regular_obs_schedule

(** What one declared data column of a stream holds:
    [columns { time : time, patch : dim, cases : count }]. *)
type obs_column_role =
  | RoleTime
      (** The observation time. A stream has either this or a window pair. *)
  | RoleDim   of string
      (** A level of the named dimension, which routes a row to its cell. *)
  | RoleValue of param_kind
      (** A value of that kind: the observed value, or a column a
          likelihood reads as {!expr.ObsColumnRef}. *)
  | RoleWindowStart
      (** The start of the period the row covers. *)
  | RoleWindowStop
      (** The end of the period the row covers, which also serves as the
          row's observation time. *)

(** One declared data column: its header name and role. *)
type obs_column = {
  col_name: string;
  col_role: obs_column_role;
}

(** The period each data row of a flow stream covers, from its
    [covers = ...] line, with the calendar arithmetic already done.

    With [t] the row's time and [one_day] one day in model time units, the
    forms lower as follows; the last column is the period of a row dated
    11 July, with [d] seven days:

    {v
      covers = day(t)        CoversFrom  (0, one_day)   [11 Jul, 12 Jul)
      starting_on(t, d)      CoversFrom  (0, d)         [11 Jul, 18 Jul)
      ending_on(t, d)        CoversUntil (one_day, d)   [5 Jul, 12 Jul)
      closing_at(t, d)       CoversUntil (0, d)         [4 Jul, 11 Jul)
      window columns         CoversWindowColumns        as written
    v}

    [ending_on] and [closing_at] differ by a day: a "week ending 11 July"
    includes the 11th, while a period "closing at 11 July" ends at its
    start. *)
type covers =
  | CoversFrom  of float * float
      (** [(offset, span)]: the period starts at [t + offset] and lasts
          [span]. *)
  | CoversUntil of float * float
      (** [(offset, span)]: the period ends at [t + offset] and lasts
          [span]. *)
  | CoversWindowColumns
      (** The period's start and end are read from each row's
          [window_start] and [window_stop] columns: the only form whose rows
          can differ in length or leave gaps. *)

(** One observation stream of the expanded model: a data column, what the
    model predicts for it, and the distribution relating the two. A stream
    indexed over [patch] becomes one per level, each scoring the rows of its
    level:

    {@json[
      {"name": "cases_urban", "source": "cases",
       "columns": [{"name": "time", "role": "time"},
                   {"name": "patch", "role": {"dim": "patch"}},
                   {"name": "cases", "role": {"value": "count"}}],
       "scored": "cases",
       "stratum": [{"dim": "patch", "level": "urban"}],
       "covers": {"kind": "until", "offset": 0.0, "span": 7.0},
       "projection": {"cumulative_flow": "infection_urban"}}
    ]} *)
type observation_model = {
  name:          string;
  obs_source:    string;
      (** The name a data file is bound to, [--data cases=file.tsv]: the
          [from] clause, or the stream's declared name. It serialises as
          ["source"]. *)
  columns:       obs_column list;
  scored:        string;
      (** The column the likelihood scores: the left-hand side of [~]. *)
  emit_schedule: observation_schedule option;
      (** When [simulate] emits synthetic observations of the stream; a fit
          reads the times from the data instead. *)
  stratum:       (string * string) list;
      (** The [(dimension, level)] pairs of the cell this stream observes;
          empty for a stream declared without indices. *)
  covers:        covers option;
      (** The period each row covers. Present for every flow projection,
          for which the compiler requires a [covers] line; absent for a
          stream read at an instant. *)
  projection:    projection;
  projection_state_grad: (string * deriv_entry) list;
      (** For a {!projection.DerivedExpr}, its derivative with respect to each
          compartment, which the ODE gradient needs; empty otherwise. *)
  likelihood:    likelihood;
}

(* ── Parameters ──────────────────────────────────────────────────────────────── *)

(** [uniform(lower = ..., upper = ...)]. *)
type uniform_prior    = { lower: float; upper: float }

(** [normal(mean = ..., sd = ...)]. *)
type normal_prior     = { mean: float; sd: float }

(** [log_normal(mu = ..., sigma = ...)]: [log X ~ Normal(mu, sigma)]. *)
type log_normal_prior = { mu: float; sigma: float }

(** [half_normal(sigma = ...)]: the absolute value of a [Normal(0, sigma)]. *)
type half_normal_prior = { sigma: float }

(** [beta(alpha = ..., beta = ...)]. *)
type beta_prior       = { alpha: float; beta: float }

(** [gamma(shape = ..., rate = ...)]. *)
type gamma_prior      = { shape: float; rate: float }

(** [exponential(rate = ...)]. *)
type exponential_prior = { rate: float }

(** [log_uniform(lower = ..., upper = ...)]:
    [log X ~ Uniform(log lower, log upper)], for a scale uncertain over
    orders of magnitude. *)
type log_uniform_prior = { lu_lower: float; lu_upper: float }

(** [truncated_normal(mean = ..., sd = ...)], truncated to [[tn_lower, tn_upper]],
    which the compiler copies from the parameter's declared range. *)
type truncated_normal_prior =
  { tn_mean: float; tn_sd: float; tn_lower: float; tn_upper: float }

(** A prior distribution with constant arguments, from [~ family(...)] on a
    parameter. *)
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
      (** A point value. The compiler never writes it, and the samplers treat
          it as a flat prior. *)

(** The distribution family of a {!hierarchical_prior}: the families of
    {!prior_dist} except [Fixed]. It serialises as the same names ("uniform",
    "normal", "log_normal", "half_normal", "beta", "gamma",
    "exponential"). *)
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

(** A prior whose arguments read other parameters (hyperparameters), so that
    they are evaluated during inference rather than fixed at compile time. In

    {@camdl[
      mu_r  : rate ~ half_normal(sigma = 0.1)
      gamma : rate ~ log_normal(mu = mu_r, sigma = 0.5)
    ]}

    [gamma]'s prior is hierarchical, with [hargs] holding
    [("mu", Param "mu_r")] and [("sigma", Const 0.5)]. *)
type hierarchical_prior = {
  hkind:       hierarchical_kind;
  hargs:       (string * expr) list;
      (** Each argument by keyword, such as [("mu", Param "mu_r")]. *)
  hpool_over:  string;
      (** The dimension of a pooled prior, from a trailing [| age]; the empty
          string when there is none. *)
}

(** The prior on an estimated parameter.

    {@json[
      "flat"
    ]} *)
type prior_spec =
  | Flat
      (** No prior beyond the parameter's bounds: a parameter declared with
          [in [lo, hi]] and no [~]. *)
  | Dist         of prior_dist
  | Hierarchical of hierarchical_prior

(** How a parameter gets its value. Inference settings exist only on
    [Estimated]; each sampler derives its change of variables from the
    parameter's kind and bounds.

    {@json[
      {"mode": "estimated", "bounds": [0.001, 2.0], "prior": "flat"}
    ]} *)
type param_value =
  | Fixed     of float
      (** A known value, as [camdlc --set beta=0.3] or a run's own override
          writes it. A [parameters] declaration never compiles to it. *)
  | Estimated of {
      est_init:      float option;
      est_bounds:    (float * float) option;
      est_prior:     prior_spec;
    }
      (** A parameter declared with bounds or a prior, which inference may
          estimate. [est_init] is a starting point, which the compiler leaves
          out. *)
  | Required
      (** A parameter declared with neither bounds nor a prior, whose value
          a run must supply. *)

(** A [#'] documentation block in the model source, kept for display. It is
    not part of the run's identity. *)
type doc = {
  text:      string option;   (** The prose. *)
  symbol:    string option;   (** The [@symbol] line: a display label for plots and reports, such as β. *)
  reference: string option;   (** The [@ref] line: a citation. It serialises as ["ref"]. *)
}

(** Every [#'] block of the model, by category and declared (not expanded)
    name. It is serialised beside the model in the JSON envelope, not inside
    it, so correcting documentation never changes the run's identity. *)
type doc_index = {
  di_model:        doc option;
      (** The block at the head of the file, about the model as a whole. *)
  di_parameters:   (string * doc) list;
  di_constants:    (string * doc) list;
  di_compartments: (string * doc) list;
  di_transitions:  (string * doc) list;
  di_observations: (string * doc) list;
  di_dimensions:   (string * doc) list;
  di_quantities:   (string * doc) list;
}

let empty_doc_index = {
  di_model = None;
  di_parameters = []; di_constants = []; di_compartments = []; di_transitions = [];
  di_observations = []; di_dimensions = []; di_quantities = [];
}

(** A model constant: a typed [let] with a constant body, such as
    [let omega : rate = 0.01], evaluated at compile time and read through
    {!expr.ConstRef}. *)
type constant = {
  c_name:  string;
  c_value: float;
  c_kind:  param_kind;   (** The declared kind, which gives the constant its dimension. *)
}

(** One parameter of the expanded model. An indexed parameter becomes one per
    cell. *)
type parameter = {
  name:          string;
  value:         param_value;
  param_kind:    param_kind option;   (** The declared kind keyword; the compiler always writes one. *)
  param_dim:     (int * int) option;  (** An explicit [dim] annotation, such as [dim 1/T], as exponents of population and time. *)
}

(* Accessors over [param_value]. Inference settings exist only on [Estimated],
   so these return [None] for [Fixed] and [Required]; [param_concrete_value]
   is the number of a [Fixed] parameter. *)
let param_concrete_value (p : parameter) : float option =
  match p.value with Fixed v -> Some v | Estimated _ | Required -> None

let param_bounds (p : parameter) : (float * float) option =
  match p.value with Estimated e -> e.est_bounds | _ -> None

let param_initial_value (p : parameter) : float option =
  match p.value with Estimated e -> e.est_init | _ -> None

let param_prior_dist (p : parameter) : prior_dist option =
  match p.value with Estimated { est_prior = Dist d; _ } -> Some d | _ -> None

let param_hierarchical (p : parameter) : hierarchical_prior option =
  match p.value with Estimated { est_prior = Hierarchical h; _ } -> Some h | _ -> None

(* ── Initial conditions ──────────────────────────────────────────────────────── *)

(** A distribution an integer compartment's initial count is drawn from.

    The arguments are the observation likelihoods' argument records, so each
    differentiable argument carries its derivatives, but not the
    {!likelihood} type itself, which would let a count be drawn from
    [bernoulli] or [beta]. *)
type init_count_law =
  | InitPoisson     of poisson_likelihood
  | InitBinomial    of binomial_likelihood
  | InitNegBinomial of neg_binomial_likelihood

(** A distribution a real compartment's initial value is drawn from. It is a
    separate type from {!init_count_law} so that the compartment's kind
    decides which distributions it may use. *)
type init_real_law =
  | InitNormal of normal_likelihood

(** A compartment's initial value. *)
type init_spec =
  | Deterministic of expr
      (** [S = N0 - I]: an expression of constants, parameters and other
          compartments' initial values. *)
  | InitCount     of init_count_law
      (** [I ~ poisson(rate = I0)] on an integer compartment: drawn afresh
          for each simulation or particle. *)
  | InitReal      of init_real_law
      (** [W ~ normal(mean = w0, sd = s)] on a real compartment. *)

(** The [init] block: one entry per compartment it names, in the order it
    names them. A compartment it does not name starts at zero. The runtime
    evaluates the entries in dependency order, since an entry may read
    another compartment's initial value, and the order written breaks ties;
    it rounds an integer compartment's value as it places it. The JSON is an
    object whose key order is that order:

    {@json[
      {"S": {"deterministic": {"bin_op": {"op": "sub", "left": {"param": "N0"}, "right": {"param": "I0"}}}},
       "I": {"count": {"poisson": {"rate": {"expr": {"param": "I0"}, "grad": {"I0": {"grad": {"const": 1.0}}}}}}}}
    ]} *)
type initial_conditions = (string * init_spec) list

(** Every expression a spec evaluates, in argument order.

    Dependency extraction walks all of them, not only the mean: a
    [binomial(n = N0 - R, p = q)] reads [R]'s initial value through [n], so
    [n] is an edge in the dependency graph. *)
let init_spec_exprs (s : init_spec) : expr list =
  match s with
  | Deterministic e -> [e]
  | InitCount (InitPoisson l) -> [l.rate.expr]
  | InitCount (InitBinomial l) -> [l.n; l.p.expr]
  | InitCount (InitNegBinomial l) -> [l.mean.expr; l.dispersion.expr]
  | InitReal (InitNormal l) -> [l.mean.expr; l.sd.expr]

(** The expression for this compartment's mean initial value: where the
    deterministic paths (the ODE skeleton, [render], the pre-flight state)
    start, and what [ic_grad] differentiates. [binomial(n, p)] has mean
    [n * p], which is not a subexpression of the spec, so this builds the
    expression rather than selecting it. *)
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

(** Trajectory rows every [step] from [start], in model time units, up to the
    simulation's [t_end]. *)
type regular_output_schedule = { start: float; step: float }

(** When a simulation records the state: [output { trajectories { every = 7 'days } }],
    or every model time unit from [t_start] when the model does not say. *)
type output_schedule =
  | OutRegular          of regular_output_schedule
  | OutAtTimes          of float list

(** What a simulation writes. *)
type output_config = {
  times:        output_schedule;
  format:       string;   (** The declared file format, ["tsv"] unless the model says otherwise. It does not enter the run's identity. *)
  trajectory:   bool;     (** Always [true]. *)
  observations: bool;     (** Always [true]. *)
}

(* ── Simulation config ───────────────────────────────────────────────────────── *)

(** The ODE integrator, from [simulate { integrator = ... }]. *)
type integrator =
  | Rk4
      (** Fixed-step classical Runge–Kutta, the default. *)
  | Rk45 of { atol: float option; rtol: float option }
      (** Adaptive Dormand–Prince with dimensionless absolute and relative
          tolerances; [None] takes the runtime's default. *)

(** The [simulate] block: the window to simulate and how. A model without one
    simulates [[0, 100]]. *)
type simulation_config = {
  t_start:        float;    (** The start of the window, in model time units. *)
  t_end:          float;
      (** The end of the window, in model time units. While [t_end_anchor]
          is present it is NaN, serialised as [null], so no comparison of
          horizons can pass by coincidence. *)
  time_semantics: string;   (** Always ["continuous"]. It does not enter the run's identity. *)
  dt:             float option;
      (** The step of the stepping backends, in model time units, from
          [simulate { dt = ... }]. [None] leaves it to the command, which
          defaults to one model time unit. *)
  rng_seed:       int option;  (** Always [None] in compiled output; the runtime does not read it. *)
  integrator:     integrator;
  t_end_anchor:   anchored_time option;
      (** An end stated relative to the data, [to = last_obs + 4 'weeks].
          The runtime writes the resolved time into [t_end] and clears this
          field before it builds the model, and refuses a model that still
          carries it. *)
}

(* ── Presets (named parameter sets for web UI / CLI) ─────────────────────────── *)

(** A named scenario from the [scenarios] block: changes a run can apply by
    name. It serialises under the model's ["scenarios"] key. *)
type preset = {
  preset_name    : string;
  preset_label   : string;                 (** A display label; the name when the scenario has none. *)
  preset_params  : (string * float) list;  (** Parameter values the scenario sets. *)
  preset_enable  : string list;            (** Interventions it switches on, by name or family name. *)
  preset_disable : string list;            (** Interventions and events it switches off. *)
  preset_scale   : (string * float) list;  (** Factors it multiplies parameters by. *)
  preset_compose : string list;
      (** Other scenarios applied first, whose parameter values this one's
          override; a composed scenario may not itself compose. *)
  preset_t_end   : float option;
      (** The scenario's own end of the window, in model time units. It is
          [Some nan], serialised as [null], while [preset_t_end_anchor] is
          present. *)
  preset_t_end_anchor : anchored_time option;
      (** The scenario's end stated relative to the data, resolved and
          cleared by the runtime as for {!simulation_config}. *)
}

(* ── Model structure ─────────────────────────────────────────────────────────── *)

(** A declared dimension and its levels, in order:
    [dimensions { patch = [urban, rural] }]. *)
type dimension = {
  dim_name  : string;
  dim_values: string list;
}

(** The model's declarations before expansion, kept so tooling and the
    lineage recorder can map expanded names back to their base names and
    levels. *)
type model_structure = {
  dimensions              : dimension list;
  compartment_dims        : (string * string list) list;
      (** Each declared compartment's dimensions, in order; [S] stratified by
          [patch] gives [("S", ["patch"])]. *)
  base_compartments       : string list;  (** The declared compartment names. *)
  transmission_transitions: string list;
      (** The declared names of transitions whose metadata says
          ["transmission"]. *)
  infectious_compartments : string list;
      (** The declared names of the compartments, other than the source,
          that those transitions' rates read. *)
}

(* ── Balance constraint ──────────────────────────────────────────────────────── *)

(** A [balance] block: at the end of every step, after its transitions,
    interventions and events, the target compartment is set to the
    expression's value, so that it absorbs whatever keeps the total on a
    prescribed course (typically [pop(t) - S - E - I]). *)
type balance_spec = {
  balance_target: string;
  balance_expr:   expr;
}

(* ── Top-level model ─────────────────────────────────────────────────────────── *)

(** A named expression that several rates share, read through
    {!expr.BindingRef} or {!expr.PerEvalRef}: [let N[p in patch] = ...]
    becomes [N_urban] and [N_rural]. A binding may read only bindings listed
    before it. *)
type binding = {
  bname: string;
  bexpr: expr;
}

(* ── Generated quantities ────────────────────────────────────────────────────── *)

(** A reference to an earlier scalar quantity, in the cell it is read in. *)
type qref = {
  qref_name:    string;
  qref_stratum: (string * string) list;  (** The [(dimension, level)] pairs of the cell; empty for a quantity declared without indices. *)
}

(** Arithmetic over scalar quantities, parameters and numbers. It is a
    separate type from {!expr} so that a reduced value cannot appear in a
    rate, nor a compartment here. *)
type scalar_expr =
  | SConst of float
  | SParam of string
  | SQRef  of qref
  | SUnOp  of { op : un_op;  arg : scalar_expr }
  | SBinOp of { op : bin_op; left : scalar_expr; right : scalar_expr }
  | SCond  of { pred : scalar_expr; then_ : scalar_expr; else_ : scalar_expr }

(** The time at which [value_at] reads a series. *)
type time_anchor =
  | ATime of expr
      (** A time expression of constants and parameters, such as [50] or a
          date. *)
  | AObs  of anchored_time
      (** A time relative to the data, such as [last_obs]. *)

(** A reduction of a series to a value of the series' own dimension. Each
    reads the series at the output times. *)
type value_reduce =
  | VFinal                    (** [final(x)]: the last value. *)
  | VMax                      (** [max(x)]. *)
  | VMin                      (** [min(x)]. *)
  | VMean                     (** The arithmetic mean over output times. *)
  | VCountAbove of expr       (** [count_above(x, c)]: the number of output times with [x > c]. *)
  | VCountBelow of expr       (** [count_below(x, c)]: the number of output times with [x < c]. *)
  | VValueAt of time_anchor
      (** [value_at(x, t)]: the value at the last output time at or before
          [t]. It is censored, not clamped, when [t] lies outside the
          trajectory. *)

(** A reduction of a series to a time. A crossing that never happens leaves
    the value censored. *)
type time_reduce =
  | TimeOfMax                 (** The first time the series reaches its maximum. *)
  | TimeOfMin                 (** The first time the series reaches its minimum. *)
  | FirstAbove of expr        (** [first_above(x, c)]: the first output time with [x > c]. *)
  | FirstBelow of expr        (** The first output time with [x < c]. *)
  | LastAbove  of expr        (** The last output time with [x > c]. *)
  | LastBelow  of expr        (** The last output time with [x < c]. *)

(** A reduction over time, with the dimension of its result fixed by the
    constructor. *)
type temporal_reduce =
  | RValue of value_reduce    (** Same dimension as the series. *)
  | RTime  of time_reduce     (** A time. *)
  | RIntegral                 (** The area under the series by the trapezoid rule, in the series' dimension times time (person-time for a count). *)

(** The series a quantity reduces. *)
type quantity_source =
  | QSState of expr
      (** An expression of the state, evaluated at each output time:
          [I], [N0 - S]. *)
  | QSObservation of string
      (** The simulated observations of the named stream. *)

(** How a quantity is computed. *)
type quantity_body =
  | QBReduced of { source : quantity_source; reduce : temporal_reduce option }
      (** A series, or with [reduce] a single number reduced from it. *)
  | QBDerived of scalar_expr
      (** Arithmetic over earlier scalar quantities. *)

(** One generated quantity of the expanded model: a named summary of what a
    run produced, reported and never scored. It is not part of the run's
    identity. A cell of an indexed quantity is its own entry.

    {@camdl[
      quantities {
        total = final(D)
        onset = first_above(I, thr)
      }
    ]} *)
type quantity = {
  q_name:    string;
  q_stratum: (string * string) list;  (** The [(dimension, level)] pairs of the cell; empty for a quantity declared without indices. *)
  q_body:    quantity_body;
  q_dimension: (int * int) option;
      (** The dimension of the result as exponents of population and time,
          as the dimension checker resolved it; [None] when it could not. *)
}

(* ── Counterfactual contrasts ────────────────────────────────────────────────── *)

(** Which part of a run a contrast operand names: [NsQuantities] one of its
    quantities, [NsObservations] one of its simulated observation streams. *)
type run_namespace = NsQuantities | NsObservations

(** The arithmetic of a contrast over members of named runs. It is a separate
    type from {!expr}, so a run member cannot appear in a rate. *)
type contrast_expr =
  | CRunMember of { run : string; ns : run_namespace; member : string }
      (** [no_sia.quantities.total]: [member] of run [run], where a run is a
          declared scenario or [fitted], the fitted model with no scenario
          applied. *)
  | CBinOp     of { op : bin_op; left : contrast_expr; right : contrast_expr }
      (** One of [+ - * /]. *)

(** A named difference between runs, such as deaths averted, from the
    [contrasts] block. It is reported, never scored, and not part of the
    run's identity.

    {@camdl[
      contrasts {
        averted = no_sia.quantities.total - with_sia.quantities.total
      }
    ]} *)
type contrast = {
  c_name:   string;
  c_body:   contrast_expr;
}

(** A compiled model: the whole of a [.camdl] file, expanded. *)
type model = {
  name:               string;
  version:            string;           (** Always ["0.3"]. *)
  time_unit:          string;           (** The unit of every time, duration and rate in the model, such as ["days"]. *)
  description:        string option;
  origin:             string option;    (** The calendar date of model time zero, as an ISO 8601 date string. *)
  origin_rata_die:    int option;       (** The proleptic Gregorian day number of [origin], so the runtime need not parse the date. *)
  compartments:       compartment list;
  transitions:        transition list;
  ode_equations:      ode_equation list;
  time_functions:     time_function list;
  tables:             table list;
  interventions:      intervention list;
      (** Interventions, events and reactive interventions alike, told apart
          by their [kind] and [fire]. *)
  observations:       observation_model list;
  parameters:         parameter list;
  constants:          constant list;
  bindings:           binding list;
  per_eval_bindings:  binding list;
      (** Empty when the compiler found nothing to lift out of the rates, or
          when [CAMDL_NO_LICM] is set. *)
  initial_conditions: initial_conditions;
  ic_grad:            (string * (string * deriv_entry) list) list;
                      (** The derivative of each compartment's mean initial
                          value with respect to each parameter, keyed by
                          compartment and then by parameter: the starting
                          point of the ODE gradient. A compartment whose
                          initial value depends on no parameter is left
                          out. *)
  output:             output_config;
  simulation:         simulation_config;
  presets:            preset list;
  model_structure:    model_structure option;  (** Always present in compiled output. *)
  balance:            balance_spec option;
  identity_tracked_compartments: string list;
      (** The compartments whose individuals the lineage recorder tracks:
          those reachable through transitions from a [#[lineage]]
          transition's destinations and parent pools. Empty when no
          transition is [#[lineage]]. *)
  doc_index:          doc_index;  (** Serialised beside the model in the envelope, under ["docs"]. *)
  quantities: quantity list;
  contrasts: contrast list;
}
