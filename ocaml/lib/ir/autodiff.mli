(** Source-to-source symbolic differentiation of camdl expressions.

    Differentiates rate / likelihood / overdispersion expressions with respect
    to named parameters, emitting derivative [expr] trees the Rust backend
    evaluates via [eval_expr]. Forcing/table coefficients are differentiated
    through their definitions (hence the [time_function]/[table] arguments):
    a parameter entering a rate only via a forcing still has a real derivative. *)

open Ir

(** The coefficient sub-expressions of a forcing definition, in evaluation
    order — the expressions a parameter can enter a rate through when it
    appears only inside that forcing. *)
val forcing_coeff_exprs : time_func_kind -> expr list

(** Repeatedly simplify an expression until it reaches a fixed point. *)
val simplify_fixpoint : expr -> expr

(** ∂rate/∂pⱼ for each parameter name (in [param_names] order), or [Error msg]
    when a coefficient derivative is structurally unsupported. *)
val differentiate_rate :
  expr -> string list -> time_function list -> table list ->
  ((string * expr) list, string) result

(** Attach per-parameter gradients to a single likelihood's arguments. *)
val differentiate_likelihood :
  projection -> likelihood -> string list -> time_function list -> table list ->
  likelihood

(** Attach per-parameter gradients to every observation model's likelihood. *)
val differentiate_observations :
  observation_model list -> string list -> time_function list -> table list ->
  observation_model list

(** Attach per-parameter gradients to each overdispersed transition's σ². *)
val differentiate_overdispersion :
  transition list -> string list -> time_function list -> table list ->
  transition list
