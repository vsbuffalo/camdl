(** The dimension checker of an expanded model: it infers the physical
    dimension of every expression and parameter, and reports each place the
    model's dimensions disagree.

    {!check_model} takes the expanded {!Ir.model}, with side inputs from the
    expander that the IR does not keep, and returns a {!result}: every
    diagnostic, and the dimensions it resolved for parameters and generated
    quantities. It changes nothing in the model. How the check works, as one
    system of linear equations over integer exponents, is described in the
    implementation's module documentation. *)

open Ir

(** A physical dimension, as integer exponents of the two base dimensions:
    [d.(0)] is the exponent of population (P) and [d.(1)] that of time (T),
    so [[|1; -1|]] is a number of individuals per unit time. Every value has
    exactly two entries; the type does not enforce it. Build one with
    {!make}, combine with {!dim_mul} and {!dim_div}, compare with {!dim_eq},
    render with {!formal_dim} or {!display_dim}. *)
type dim_vec = int array

(** [make p t] is the dimension with population exponent [p] and time
    exponent [t] (e.g. [make 0 (-1)] is a per-capita rate). *)
val make : int -> int -> dim_vec

val dim_mul : dim_vec -> dim_vec -> dim_vec
val dim_div : dim_vec -> dim_vec -> dim_vec
val dim_eq  : dim_vec -> dim_vec -> bool

(** Compact algebraic rendering, e.g. ["P*T^-1"]. *)
val formal_dim : dim_vec -> string

(** Human-readable rendering, e.g. ["per-capita rate"]. *)
val display_dim : dim_vec -> string

(** How serious a diagnostic is. An [Error] stops the compile. An [Info]
    does not: the checker uses it for a parameter whose dimension the model
    leaves undetermined (I300) and for a suspicious log-normal prior
    argument (I316). *)
type severity = Error | Info

(** The construct a diagnostic concerns. The IR carries no source locations,
    so the compiler looks up the construct's declaration to place the
    diagnostic; a diagnostic without a subject is printed without a
    location. *)
type subject = Dim_sites.subject =
  | STransition  of string   (** A transition, by expanded name. *)
  | SOde         of string   (** The ODE equation of the named compartment. *)
  | SObservation of string   (** An observation stream. *)
  | SBinding     of string   (** A shared binding. The compiler has no location for one. *)
  | SContrast    of string   (** A contrast between runs. *)
  | SInit        of string   (** The initial condition of the named compartment. *)
  | SParameter   of string   (** A parameter, for a diagnostic about its prior or bounds. *)
  | SQuantity    of string   (** A generated quantity. *)
  | SIntervention of string  (** An intervention, event or reactive trigger. *)
  | SLet         of string   (** A typed [let], for its annotation. *)
  | SForcing     of string   (** A forcing, by expanded name, for its arguments. *)
  | STable       of string   (** A table, for its inline cells. *)

(** One finding of the checker, before the compiler turns it into a
    located diagnostic. *)
type diagnostic = {
  severity : severity;
  code     : string;          (** The diagnostic code, such as ["E300"]. *)
  message  : string;
  detail   : string option;   (** What was expected and what was found. *)
  hint     : string option;   (** How to fix it. *)
  subject  : subject option;
}

(** Everything the checker reports about one model. *)
type result = {
  diagnostics   : diagnostic list;
  (** The errors from the equations, at most one per subject, code and
      message; then the findings of the prior check (E316, I316); then one
      I300 for each undetermined parameter that some model position reads. *)
  param_dims    : (string * dim_vec) list;
  (** Every parameter and typed constant whose dimension the model
      determines, by name. *)
  quantity_dims : ((string * (string * string) list) * dim_vec) list;
  (** Every generated quantity whose dimension the model determines, keyed by
      its name and the [(dimension, level)] pairs of its cell. The compiler
      writes these onto the quantities of the compiled model. *)
}

(** A prior or a pair of bounds as written, before the expander folds a
    constant prior's arguments to numbers: each argument resolved to an IR
    expression with its unit literals intact, so that the checker can see
    which unit each one asserted.

    {@camdl[
      parameters {
        gamma : rate ~ normal(mu = 0.2 'per_day, sigma = 0.05 'per_day)
      }
    ]} *)
type prior_source = {
  param  : string;
  family : string;
  (** The family's name, such as ["normal"] or ["log_normal"], or
      ["bounds"] for a parameter's [in [lo, hi]] bounds, whose arguments are
      ["lower"] and ["upper"]. *)
  args   : (string * expr) list;   (** Each argument, by name. *)
}

(** Dimensionally check an expanded model, returning every diagnostic plus the
    resolved parameter and generated-quantity dimensions. [prior_args] are the
    model's priors and bounds as written; without them only hierarchical
    priors, whose arguments the IR keeps, are checked for explicit units. *)
val check_model :
  ?prior_args:prior_source list -> ?constant_dims:(string * (int * int)) list ->
  ?table_dims:(string * (int * int)) list ->
  ?typed_lets:(string * (int * int) * expr) list -> model -> result
(** [constant_dims] are the [dim] clauses of typed constants, which the IR's
    constant record does not carry; [table_dims] the dimensions of tables
    declared with a unit annotation, which the IR's table record does not
    carry either. [typed_lets] are the expanded bodies of typed, non-constant
    [let]s with their declared dimension; the IR substitutes them into the
    expressions that use them, so it keeps no node for the annotation
    (E318). All of these come from the expander; the compiler and
    [camdlc inspect] pass them. A model checked without them (an IR read
    back from JSON, a model built in a test) gets weaker results, never
    contradictory ones: a constant with a [dim] clause is inferred from its
    use instead of declared, and the unit literals of constant priors are not
    checked, since the IR holds those arguments only as numbers.

    An exponent too large for a native integer gives one error, E310, in
    place of every other result. *)
