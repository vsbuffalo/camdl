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

(** ∂rate/∂pⱼ for each parameter name (in [param_names] order) as a classified
    [grad_map] ([Grad] / [DEUnsupported] for a live-but-omitted coefficient), or
    [Error msg] when a coefficient derivative is structurally unsupported (E600). *)
val differentiate_rate :
  expr -> string list -> time_function list -> table list ->
  ((string * deriv_entry) list, string) result

(** ∂rate/∂compartment for each compartment name — the transition's
    [rate_state_grad] map ([J_x]'s ingredient for the ODE forward sensitivities,
    gh#275). Both a live-but-omitted and a nonsmooth-of-state coefficient become a
    [DEUnsupported] the fit-time gradient gate refuses on (unlike the rate-θ
    driver's E600); an absent key is a genuine zero. The [binding list] is threaded
    because a hoisted binding body is state-bearing under [WrtPop]. *)
val differentiate_rate_state :
  expr -> string list -> time_function list -> table list -> binding list ->
  (string * deriv_entry) list

(** ∂projection/∂compartment for an observation projection — the model's
    [projection_state_grad] (gh#275 §1h), the ODE observation gradient's factor-2
    ingredient. Only a [DerivedExpr] projection has a non-trivial state gradient
    (reuses [differentiate_rate_state]); linear projections emit nothing. *)
val differentiate_projection :
  projection -> string list -> time_function list -> table list -> binding list ->
  (string * deriv_entry) list

(** ∂(initial-condition expression)/∂θ for each parameter — one compartment's
    entry in the model's [ic_grad] map (the ODE forward-sensitivity seed
    S(t_start), gh#275 §1c). Same defer POLICY as the obs/σ² driver: a genuine
    zero is dropped, an [Omitted]/[Unsupported] coefficient becomes a
    [DEUnsupported] the fit-time gradient gate refuses ODE-NUTS on. *)
val differentiate_ic :
  expr -> string list -> time_function list -> table list ->
  (string * deriv_entry) list

(** Attach per-parameter gradients to every argument of every DRAWN initial
    condition (`I ~ poisson(rate = I0)`), so `∂/∂θ log p(x₀ | θ)` has an
    emitted derivative to chain through. [Pop] differentiates to 0 by design:
    the density is scored with x₀ held fixed, so the only θ-dependence is
    through the law's arguments. Distinct from [differentiate_ic], which
    differentiates the MEAN for the ODE forward-sensitivity seed. *)
val differentiate_initial_conditions :
  initial_conditions -> string list -> time_function list -> table list ->
  initial_conditions

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
