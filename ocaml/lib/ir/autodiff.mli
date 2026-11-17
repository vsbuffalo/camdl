(** Symbolic differentiation of IR expressions: the derivatives the
    compiler writes into the gradient fields of the IR, for the runtime's
    gradient-based methods. The design is described in [autodiff.ml]. *)

open Ir

(** The coefficient sub-expressions of a forcing definition, in evaluation
    order — the expressions a parameter can enter a rate through when it
    appears only inside that forcing. *)
val forcing_coeff_exprs : time_func_kind -> expr list

(** Repeatedly simplify an expression until it reaches a fixed point:
    additive and multiplicative identities, arithmetic on numeric literals,
    and zero terms of a sum. *)
val simplify_fixpoint : expr -> expr

(** The derivative of a rate with respect to each parameter, in
    [param_names] order: a transition's [rate_grad]. A zero derivative is
    left out and a live coefficient whose derivative is not written becomes
    [DEUnsupported]. [Error msg] when a parameter reaches data fixed when the
    model is built, which the caller reports as [E600]. *)
val differentiate_rate :
  expr -> string list -> time_function list -> table list ->
  ((string * deriv_entry) list, string) result

(** The derivative of a rate with respect to each compartment: a
    transition's [rate_state_grad], for the ODE forward sensitivities. It
    never fails: a coefficient whose derivative is not written, and a
    function that is not smooth in the state, become [DEUnsupported]. An
    absent key is a zero derivative. The [binding list] is needed because a
    binding's body is differentiated with respect to the state. *)
val differentiate_rate_state :
  expr -> string list -> time_function list -> table list -> binding list ->
  (string * deriv_entry) list

(** The derivative of an observation's projection with respect to each
    compartment: its [projection_state_grad], for the ODE observation
    gradient. Only a [DerivedExpr] projection has entries; the other
    projections give an empty list. *)
val differentiate_projection :
  projection -> string list -> time_function list -> table list -> binding list ->
  (string * deriv_entry) list

(** The derivative of one compartment's initial value with respect to each
    parameter: that compartment's entry in the model's [ic_grad], the
    starting value of the ODE state sensitivities. A zero derivative is left
    out; one that cannot be written becomes [DEUnsupported]. *)
val differentiate_ic :
  expr -> string list -> time_function list -> table list ->
  (string * deriv_entry) list

(** Fill the per-parameter gradients of every argument of every drawn
    initial state ([I ~ poisson(rate = I0)]), so the runtime can
    differentiate the log density of the initial state. A compartment in an
    argument differentiates to 0, because the density is scored with the
    initial state held fixed. Distinct from [differentiate_ic], which
    differentiates a deterministic initial value. *)
val differentiate_initial_conditions :
  initial_conditions -> string list -> time_function list -> table list ->
  initial_conditions

(** Fill the per-parameter gradients of a single likelihood's arguments. *)
val differentiate_likelihood :
  projection -> likelihood -> string list -> time_function list -> table list ->
  likelihood

(** Fill the per-parameter gradients of every observation model's
    likelihood. *)
val differentiate_observations :
  observation_model list -> string list -> time_function list -> table list ->
  observation_model list

(** Fill the per-parameter gradients of each overdispersed transition's
    variance. *)
val differentiate_overdispersion :
  transition list -> string list -> time_function list -> table list ->
  transition list
