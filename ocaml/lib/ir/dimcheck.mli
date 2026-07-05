(** Dimensional analysis of an expanded IR model.

    Runs Hindley–Milner-style unification over a two-base dimension algebra
    (population, time) to check that every rate, ODE derivative, observation
    argument, and contrast is dimensionally consistent, and to infer each
    parameter's resolved dimension. The checker records a [subject] on every
    diagnostic (the IR has no source spans); the compiler resolves that to a
    declaration location. *)

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
type subject =
  | STransition  of string
  | SOde         of string
  | SObservation of string
  | SBinding     of string
  | SContrast    of string

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

(** Dimensionally check an expanded model, returning every diagnostic plus the
    resolved parameter and generated-quantity dimensions. *)
val check_model : model -> result
