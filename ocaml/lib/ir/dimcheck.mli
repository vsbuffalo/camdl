(** Dimensional analysis of an expanded IR model.

    Every expression-bearing position of the model ([Dim_sites.sites]) is
    turned into linear equations over integer exponent vectors in two bases
    (population, time), with one more equation for the dimension its position
    requires, and the whole system is solved at once by [Dim_solve]. Every
    contradiction and every integer infeasibility is a diagnostic at the
    position it surfaces at; a parameter the system leaves free gets I300 with
    whatever the model does determine about it. The checker records a
    [subject] on every diagnostic (the IR has no source spans); the compiler
    resolves that to a declaration location. *)

open Ir

(** A dimension vector over the fixed base set: [d.(0)] is the population
    exponent, [d.(1)] the time exponent. Build one with [make], combine with
    [dim_mul]/[dim_div], compare with [dim_eq], render with
    [formal_dim]/[display_dim]. *)
type dim_vec = int array

(** [make p t] is the dimension with population exponent [p] and time
    exponent [t] (e.g. [make 1 (-1)] is a per-capita rate). *)
val make : int -> int -> dim_vec

val dim_mul : dim_vec -> dim_vec -> dim_vec
val dim_div : dim_vec -> dim_vec -> dim_vec
val dim_eq  : dim_vec -> dim_vec -> bool

(** Compact algebraic rendering, e.g. ["P*T^-1"]. *)
val formal_dim : dim_vec -> string

(** Human-readable rendering, e.g. ["per-capita rate"]. *)
val display_dim : dim_vec -> string

type severity = Error | Info

(** The construct a diagnostic concerns; the compiler maps it to a source
    location. *)
type subject = Dim_sites.subject =
  | STransition  of string
  | SOde         of string
  | SObservation of string
  | SBinding     of string
  | SContrast    of string
  | SInit        of string
  | SParameter   of string   (** a parameter, for a diagnostic about its prior *)
  | SQuantity    of string
  | SIntervention of string  (** an intervention, event or reactive trigger *)
  | SLet         of string   (** a typed `let`, for its annotation *)

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  subject  : subject option;
}

type result = {
  diagnostics   : diagnostic list;
  param_dims    : (string * dim_vec) list;
  quantity_dims : ((string * (string * string) list) * dim_vec) list;
}

(** A prior as written, before the expander folds a constant prior's arguments
    to floats: each argument resolved to an IR expression, unit literals intact
    (as [UncheckedDim] nodes whose reason starts "unit literal"). *)
type prior_source = {
  param  : string;
  family : string;                  (** "normal", "log_normal", ... *)
  args   : (string * expr) list;
}

(** Dimensionally check an expanded model, returning every diagnostic plus the
    resolved parameter and generated-quantity dimensions. [prior_args] are the
    model's priors as written; without them only hierarchical priors, whose
    arguments the IR keeps, are checked for explicit units. *)
val check_model :
  ?prior_args:prior_source list -> ?constant_dims:(string * (int * int)) list ->
  ?typed_lets:(string * (int * int) * expr) list -> model -> result
(** [constant_dims] are the `dim` clauses of typed constants, which the IR's
    constant record does not carry. [typed_lets] are the expanded bodies of
    typed, non-constant `let`s with their declared dimension; the IR inlines
    them, so it keeps no node for the annotation (E318). Both side inputs come from the expander;
    the compiler and `camdlc inspect` pass them. A model checked without them
    (an IR reloaded from JSON, a model built in a test) gets weaker results,
    never contradictory ones: a constant with a `dim` clause is inferred from
    its use instead of declared, and the unit literals of constant priors are
    not checked, since the IR no longer holds them. The next IR schema bump
    puts the clause on the constant record, which removes the first gap. *)
