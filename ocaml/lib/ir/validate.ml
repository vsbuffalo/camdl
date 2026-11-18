(** The structural check of an expanded model: duplicate names, references
    to names the model does not declare, and combinations of declarations the
    runtime cannot execute.

    {!validate} reads an {!Ir.model} and returns [Ok ()] or every {!error} it
    found, in the order found; it changes nothing. It is the first check the
    compiler runs on the expanded model, and the compile stops after it if it
    reports anything. The compiler turns each error into a diagnostic with a
    code from E500 to E519 and places it at the declaration of the named
    construct or, for a reference to a name that does not exist, at the
    construct that contains the reference ({!site}).

    Almost every check here repeats one the expander makes, with a source
    location, while it builds the model. This module is the check at the
    boundary of the IR itself: it also catches an IR that was written by
    hand, produced by another tool, or built by an expander that has
    drifted from its own rules. The Rust runtime validates a model again
    when it loads one ([rust/crates/ir/src/validate.rs]), and the checks of
    initial-state laws and intervention targets here mirror its.

    The checks are:

    - no two compartments, transitions or parameters share a name;
    - every name read by a transition rate, an ODE derivative, a likelihood
      argument or an intervention action's expression is a declared
      compartment, parameter, constant, table or forcing, and no such
      expression holds a spline basis function, which only gradient maps may;
    - every compartment named in a stoichiometry, a projection or an action
      target exists, every transition named by a projection exists, and a
      union of flows names no flow twice;
    - a compartment of real kind appears in no stoichiometry and has an ODE
      equation, and no other compartment has one (two equations for one
      compartment are not detected);
    - no transition changes a compartment by zero;
    - no shared binding reads a parameter (see {!first_param});
    - no sub-expression made only of literals has a value that is NaN or
      infinite ({!first_non_finite}), in any expression walked for
      references and also in shared bindings, derived projections,
      overdispersion variances, initial conditions and quantities;
    - every initial condition names a compartment, its law suits the
      compartment's kind, no drawn compartment is the balance target, and
      the initial conditions do not read each other in a cycle
      ({!Init_order.topo}).

    Some expressions are not walked for references: initial conditions,
    shared bindings, derived projections, forcing arguments, overdispersion
    variances, quantities and contrasts. A [BindingRef] is not resolved
    either; the Rust runtime resolves binding names when it builds the
    model. Duplicate constants, tables, forcings, observations and bindings
    are left to the expander. *)

open Ir

(** The construct that contains a reference, where a diagnostic about a
    name that does not exist is placed. *)
type site =
  | InTransition  of string   (** A transition, by expanded name. *)
  | InOde         of string   (** The ODE equation of the named compartment. *)
  | InObservation of string   (** An observation stream. *)
  | InIntervention of string  (** An intervention or event. *)
  | InBinding     of string   (** A shared binding (a hoisted [let]), by name. *)
  | InInit        of string   (** The initial condition of the named compartment. *)
  | InQuantity    of string   (** A generated quantity, by name. *)

(** One structural defect of an expanded model. The compiler's code for each
    is given in brackets. *)
type error =
  | DuplicateCompartment  of string   (** [E500]. *)
  | DuplicateTransition   of string   (** [E501]. *)
  | DuplicateParameter    of string   (** [E502]. *)
  | UnknownCompartment    of string * site   (** [E503]. *)
  | UnknownParameter      of string * site   (** [E504]. *)
  | UnknownConstant       of string * site   (** [E518]. *)
  | UnknownTable          of string * site   (** [E505]. *)
  | UnknownTimeFunction   of string * site   (** [E506]: no forcing of that name. *)
  | ForcingBasisOutsideGradient of string * site
      (** [E519]: a model expression holds a basis function of the named
          forcing, a node only the derivatives autodiff writes may hold. No
          camdl source produces it. *)
  | UnknownTransition     of string * site   (** [E507]: a projection names no transition. *)
  | DuplicateFlowInUnion  of string * site  (** [E514]: a sum or ratio of flows names one flow twice, which would count each of its events twice. *)
  | RealCompartmentInStoichiometry of string * string  (** [E508]: (transition, compartment); a real compartment changes only by its ODE. *)
  | MissingOdeEquation    of string   (** [E509]: a real compartment with no ODE equation. *)
  | OdeForNonRealComp     of string   (** [E510]: an ODE equation for a compartment that is not real, or does not exist. *)
  | ZeroDelta             of string * string  (** [E511]: (transition, compartment) with a change of zero. *)
  | ParamInBinding        of string * string  (** [E512]: (binding, parameter) read by it. *)
  | InitUnknownCompartment of string          (** [E513]: an initial condition for a compartment that does not exist. *)
  | InitDependencyCycle   of string list      (** [E515]: initial conditions that read each other, listed in the order they close the cycle. *)
  | InitLawKindMismatch   of string * string * string
      (** [E516]: (compartment, law family, the kind of compartment the law
          seeds); an integer law on a real compartment or the reverse. *)
  | InitLawOnBalanceTarget of string          (** [E517]: a compartment both drawn from a law and set by the balance, which overwrites the draw. *)
  | NonFiniteConstant of expr * float * site
      (** [E520]: a sub-expression made only of literals whose value is NaN or
          infinite, such as [(0 - 8) ^ 0.5], with that value. The innermost
          such sub-expression is reported, once per expression. *)

let error_to_string = function
  | DuplicateCompartment s -> Printf.sprintf "duplicate compartment: %s" s
  | DuplicateTransition  s -> Printf.sprintf "duplicate transition: %s" s
  | DuplicateParameter   s -> Printf.sprintf "duplicate parameter: %s" s
  | UnknownCompartment  (s, _) -> Printf.sprintf "unknown compartment: %s" s
  | UnknownParameter    (s, _) -> Printf.sprintf "unknown parameter: %s" s
  | UnknownConstant     (s, _) -> Printf.sprintf "unknown constant: %s" s
  | UnknownTable        (s, _) -> Printf.sprintf "unknown table: %s" s
  | UnknownTimeFunction (s, _) -> Printf.sprintf "unknown time_function: %s" s
  | ForcingBasisOutsideGradient (s, _) ->
    Printf.sprintf "forcing_basis of '%s' outside a gradient map; only derivative entries may hold it" s
  | UnknownTransition   (s, _) -> Printf.sprintf "unknown transition: %s" s
  | DuplicateFlowInUnion (s, _) ->
    Printf.sprintf
      "flow '%s' appears twice in a cumulative_flow_sum — a union of flows must \
       be disjoint, or every event on it is counted twice" s
  | RealCompartmentInStoichiometry (tr, c) ->
    Printf.sprintf "real compartment '%s' in stoichiometry of '%s'" c tr
  | MissingOdeEquation s -> Printf.sprintf "real compartment '%s' has no ODE equation" s
  | OdeForNonRealComp  s -> Printf.sprintf "ODE equation for non-real compartment '%s'" s
  | ZeroDelta (tr, c)    -> Printf.sprintf "zero delta for '%s' in transition '%s'" c tr
  | ParamInBinding (b, p) ->
    Printf.sprintf "parameter '%s' reachable from hoisted binding '%s'" p b
  | InitUnknownCompartment s ->
    Printf.sprintf "initial condition names unknown compartment: %s" s
  | InitDependencyCycle cyc ->
    Printf.sprintf "initial conditions reference each other in a cycle: %s"
      (String.concat " -> " (cyc @ [List.hd cyc]))
  | InitLawKindMismatch (comp, family, seeds) ->
    Printf.sprintf "initial condition '%s' is drawn from '%s', which seeds %s"
      comp family seeds
  | InitLawOnBalanceTarget comp ->
    Printf.sprintf "initial condition '%s' is drawn from a law but is the balance target"
      comp
  | NonFiniteConstant (_, v, _) ->
    Printf.sprintf "a sub-expression of literals evaluates to %F" v

module SS = Set.Make(String)

(** The first parameter an expression reads directly, if any.

    A shared binding must read no parameter, because {!Autodiff}
    differentiates a [BindingRef] to zero: a parameter inside a binding would
    silently lose its gradient. The walk does not follow a [BindingRef] (each
    binding is checked on its own, so a parameter read through another
    binding is reported on that one), nor the cells of a table or the
    coefficients of a forcing: a parameter reached only through a table or
    forcing is not found here. *)
let first_param (e : expr) : string option =
  let rec go = function
    | Param p -> Some p
    | Const _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _ | BindingRef _
    | Projected | ObsColumnRef _ | ObsAnchor _ | UnitLiteral _ -> None
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
    | BinOp b -> (match go b.left with Some _ as r -> r | None -> go b.right)
    | UnOp u  -> go u.arg
    | Cond c  ->
      (match go c.pred with
       | Some _ as r -> r
       | None -> (match go c.then_ with Some _ as r -> r | None -> go c.else_))
    | TableLookup (_, idxs) -> List.find_map go idxs
    | Reduce terms -> List.find_map go terms
    | UncheckedDim u -> go u.inner
  in
  go e

let references_param (e : expr) : bool = first_param e <> None

let uniq_check name_of xs constructor errors =
  let seen = Hashtbl.create 16 in
  List.iter (fun x ->
    let n = name_of x in
    if Hashtbl.mem seen n
    then errors := constructor n :: !errors
    else Hashtbl.add seen n ()
  ) xs;
  let set = Hashtbl.fold (fun k () s -> SS.add k s) seen SS.empty in
  set

(** The value of a binary operation on two numbers, as the runtime evaluator
    computes it for finite operands, or [None] for an operator this check
    does not evaluate (the comparisons, whose value is always finite). *)
let literal_bin_op op a b : float option =
  match op with
  | Add -> Some (a +. b)
  | Sub -> Some (a -. b)
  | Mul -> Some (a *. b)
  | Div -> Some (a /. b)
  | Pow -> Some (a ** b)
  | Min -> Some (Float.min a b)
  | Max -> Some (Float.max a b)
  | Mod -> Some (if b = 0.0 then Float.nan else Float.rem a b)
  | Eq | Neq | Lt | Gt | Le | Ge -> None

let literal_un_op op a : float =
  match op with
  | Neg -> -. a | Exp -> exp a | Log -> log a | Sqrt -> sqrt a
  | Abs -> abs_float a | Floor -> floor a | Ceil -> ceil a
  | Sin -> sin a | Cos -> cos a | Tanh -> tanh a

(** The innermost sub-expression of [e] made only of literals whose value is
    NaN or infinite, with that value, if there is one.

    A literal is a [Const] or a [UnitLiteral]; arithmetic and the elementary
    functions of literals are literal too. A literal whose value is not
    finite, such as [(0 - 8) ^ 0.5] or [1 / 0], is a mistake in the model
    wherever it stands, even in a branch that never runs, and JSON cannot
    carry its value. The first one found, left to right, is returned. *)
let first_non_finite (e : expr) : (expr * float) option =
  let found = ref None in
  (* The value of a literal node, or [None] when it is not literal or its
     value is not finite, so that only the innermost one is kept. *)
  let literal node v =
    if Float.is_finite v then Some v
    else begin
      (if !found = None then found := Some (node, v));
      None
    end
  in
  let rec go node : float option =
    match node with
    | Const v -> literal node v
    | UnitLiteral u -> literal node u.value
    | BinOp b ->
      let l = go b.left in
      let r = go b.right in
      (match l, r with
       | Some a, Some c ->
         (match literal_bin_op b.op a c with Some v -> literal node v | None -> None)
       | _ -> None)
    | UnOp u ->
      (match go u.arg with Some a -> literal node (literal_un_op u.op a) | None -> None)
    | UncheckedDim u -> go u.inner
    | Cond c -> ignore (go c.pred); ignore (go c.then_); ignore (go c.else_); None
    | TableLookup (_, idxs) -> List.iter (fun i -> ignore (go i)) idxs; None
    | Reduce terms -> List.iter (fun t -> ignore (go t)) terms; None
    | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _
    | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ | ObsAnchor _ -> None
  in
  ignore (go e);
  !found

(** A quantity's scalar arithmetic as an {!expr}, for {!first_non_finite}:
    a number stays a number, and a reference to a parameter or another
    quantity becomes a non-literal leaf. *)
let rec scalar_as_expr (e : scalar_expr) : expr =
  match e with
  | SConst v -> Const v
  | SParam p -> Param p
  | SQRef _ -> Time
  | SUnOp { op; arg } -> UnOp { op; arg = scalar_as_expr arg }
  | SBinOp { op; left; right } ->
    BinOp { op; left = scalar_as_expr left; right = scalar_as_expr right }
  | SCond { pred; then_; else_ } ->
    Cond { pred = scalar_as_expr pred; then_ = scalar_as_expr then_;
           else_ = scalar_as_expr else_ }

(** Every expression a quantity's body holds: the series, the threshold or
    time of its reduction, or its scalar arithmetic. *)
let quantity_exprs (q : quantity) : expr list =
  match q.q_body with
  | QBDerived s -> [scalar_as_expr s]
  | QBReduced { source; reduce } ->
    let src = match source with QSState e -> [e] | QSObservation _ -> [] in
    let red = match reduce with
      | Some (RValue (VCountAbove c | VCountBelow c))
      | Some (RTime (FirstAbove c | FirstBelow c | LastAbove c | LastBelow c)) -> [c]
      | Some (RValue (VValueAt (ATime t))) -> [t]
      | Some (RValue (VFinal | VMax | VMin | VMean | VValueAt (AObs _)))
      | Some (RTime (TimeOfMax | TimeOfMin)) | Some RIntegral | None -> []
    in
    src @ red

(** Record an E520 error for the first non-finite literal of [e]. *)
let check_finite ~site errors e =
  match first_non_finite e with
  | Some (node, v) -> errors := NonFiniteConstant (node, v, site) :: !errors
  | None -> ()

let check_expr_refs ~site ~comps ~params ~constants ~tables ~tfs errors e =
  let rec go = function
    | Const _ | UnitLiteral _ | Time | Dt | Projected | ObsColumnRef _ | ObsAnchor _ -> ()
    | Param p -> if not (SS.mem p params) then errors := UnknownParameter (p, site) :: !errors
    | ConstRef c -> if not (SS.mem c constants) then errors := UnknownConstant (c, site) :: !errors
    | Pop   c -> if not (SS.mem c comps)  then errors := UnknownCompartment (c, site) :: !errors
    | PopSum cs -> List.iter (fun c -> if not (SS.mem c comps) then errors := UnknownCompartment (c, site) :: !errors) cs
    | BinOp b -> go b.left; go b.right
    | UnOp u  -> go u.arg
    | Cond c  -> go c.pred; go c.then_; go c.else_
    | TimeFunc n ->
      if not (SS.mem n tfs) then errors := UnknownTimeFunction (n, site) :: !errors
    (* Model expressions only (gradients are not walked here). *)
    | ForcingBasis { forcing; _ } -> errors := ForcingBasisOutsideGradient (forcing, site) :: !errors
    | TableLookup (t, idxs) ->
      (if not (SS.mem t tables) then errors := UnknownTable (t, site) :: !errors);
      List.iter go idxs
    | UncheckedDim u -> go u.inner
    | Reduce terms -> List.iter go terms
    | BindingRef _ -> ()   (* leaf; binding name resolution happens at CompiledModel::new *)
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  in
  go e;
  check_finite ~site errors e

let validate (m : model) : (unit, error list) result =
  let errors = ref [] in

  (* Unique-name checks; the returned name sets are reused by the reference
     checks below. *)
  let comp_names = uniq_check (fun (c: compartment)     -> c.name) m.compartments (fun n -> DuplicateCompartment n) errors in
  let tr_set     = uniq_check (fun (t: transition)      -> t.name) m.transitions  (fun n -> DuplicateTransition  n) errors in
  let params     = uniq_check (fun (p: parameter)       -> p.name) m.parameters   (fun n -> DuplicateParameter   n) errors in

  let real_comps = List.filter_map (fun (c: compartment)     -> if c.kind = Real then Some c.name else None) m.compartments |> SS.of_list in
  let tables     = List.map (fun (t: table)         -> t.name) m.tables        |> SS.of_list in
  let tfs        = List.map (fun (f: time_function) -> f.name) m.time_functions |> SS.of_list in

  let constants  = List.map (fun (c : constant) -> c.c_name) m.constants |> SS.of_list in
  let check_expr_r ~site e = check_expr_refs ~site ~comps:comp_names ~params ~constants ~tables ~tfs errors e in

  (* stoichiometry *)
  List.iter (fun (tr: transition) ->
    List.iter (fun (comp, delta) ->
      if not (SS.mem comp comp_names)
      then errors := UnknownCompartment (comp, InTransition tr.name) :: !errors
      else if SS.mem comp real_comps
      then errors := RealCompartmentInStoichiometry (tr.name, comp) :: !errors;
      if delta = 0
      then errors := ZeroDelta (tr.name, comp) :: !errors
    ) tr.stoichiometry;
    check_expr_r ~site:(InTransition tr.name) tr.rate;
    (match tr.draw_method with
     | DrawOverdispersed { sigma_sq; _ } ->
       check_finite ~site:(InTransition tr.name) errors sigma_sq
     | DrawPoisson | DrawDeterministic -> ())
  ) m.transitions;

  (* ODE equations *)
  let ode_comps = List.map (fun (e: ode_equation) -> e.compartment) m.ode_equations |> SS.of_list in
  SS.iter (fun rc ->
    if not (SS.mem rc ode_comps) then errors := MissingOdeEquation rc :: !errors
  ) real_comps;
  List.iter (fun (eq: ode_equation) ->
    if not (SS.mem eq.compartment real_comps)
    then errors := OdeForNonRealComp eq.compartment :: !errors;
    check_expr_r ~site:(InOde eq.compartment) eq.derivative
  ) m.ode_equations;

  (* observations *)
  List.iter (fun (obs: observation_model) ->
    let here = InObservation obs.name in
    (* Projection reference check: every flow and compartment a projection
       names must exist. A projection of a partly indexed family, such as
       `I[child]` on an `[age, patch]` family, lowers to a name like `I_child`
       that is no compartment. *)
    (match obs.projection with
     | CumulativeFlow tn ->
       if not (SS.mem tn tr_set) then errors := UnknownTransition (tn, here) :: !errors
     | CumulativeFlowSum tns ->
       List.iter (fun tn ->
         if not (SS.mem tn tr_set) then errors := UnknownTransition (tn, here) :: !errors
       ) tns;
       (* The expander checks this where it lowers a union; this covers an IR
          from any other producer. *)
       let seen = Hashtbl.create 8 in
       List.iter (fun tn ->
         if Hashtbl.mem seen tn then
           errors := DuplicateFlowInUnion (tn, here) :: !errors
         else Hashtbl.add seen tn ()) tns
     | FlowRatio { numerator; denominator } ->
       (* Each side is a flow union in its own right: every name must be a
          transition and no side may name one twice. The same flow on both
          sides is the expected shape (`a / (a + b)`) and is not a collision. *)
       List.iter (fun tns ->
         List.iter (fun tn ->
           if not (SS.mem tn tr_set) then errors := UnknownTransition (tn, here) :: !errors
         ) tns;
         let seen = Hashtbl.create 8 in
         List.iter (fun tn ->
           if Hashtbl.mem seen tn then
             errors := DuplicateFlowInUnion (tn, here) :: !errors
           else Hashtbl.add seen tn ()) tns
       ) [numerator; denominator]
     | CurrentPop cn ->
       if not (SS.mem cn comp_names) then errors := UnknownCompartment (cn, here) :: !errors
     | CurrentPopSum cns ->
       List.iter (fun cn ->
         if not (SS.mem cn comp_names) then errors := UnknownCompartment (cn, here) :: !errors
       ) cns
     | DerivedExpr e -> check_finite ~site:here errors e);
    (* Every name in every likelihood argument. *)
    let chk e = check_expr_r ~site:here e in
    (match obs.likelihood with
     | Poisson      { rate }                    -> chk rate.expr
     | NegBinomial  { mean; dispersion }        -> chk mean.expr; chk dispersion.expr
     | Normal       { mean; sd }                -> chk mean.expr; chk sd.expr
     | Binomial     { n; p }                    -> chk n; chk p.expr
     | BetaBinomial { n; alpha; beta }          -> chk n; chk alpha.expr; chk beta.expr
     | Beta         { mean; concentration }     -> chk mean.expr; chk concentration.expr
     | Bernoulli    { p }                       -> chk p.expr
     | ZeroInflatedNegBinomial { mean; dispersion; pi } ->
       chk mean.expr; chk dispersion.expr; chk pi.expr)
  ) m.observations;

  (* No binding reads a parameter (see [first_param]). The expander hoists
     only parameter-free lets, so this fires only if that rule breaks. *)
  List.iter (fun (b : binding) ->
    (match first_param b.bexpr with
     | Some p -> errors := ParamInBinding (b.bname, p) :: !errors
     | None -> ());
    check_finite ~site:(InBinding b.bname) errors b.bexpr
  ) m.bindings;

  (* Every initial condition names a compartment of the model. The expander
     reports the same mistake with a location (E277). *)
  List.iter (fun (k, spec) ->
    if not (SS.mem k comp_names)
    then errors := InitUnknownCompartment k :: !errors;
    List.iter (check_finite ~site:(InInit k) errors) (init_spec_exprs spec)
  ) m.initial_conditions;

  (* Initial-state law admissibility. The compartment's kind selects which laws
     may seed it, and the `balance { }` target may be seeded by none of them —
     the balance stage overwrites it after every substep, so a draw there is
     discarded. The frontend reports both with a located E344/E345; these are
     the contract-boundary nets, and they mirror `rust/crates/ir/src/validate.rs`
     so the two sides agree by construction rather than by comment. *)
  List.iter (fun (comp, spec) ->
    let real = SS.mem comp real_comps in
    (match spec with
     | Deterministic _ -> ()
     | InitCount _ when real ->
       errors := InitLawKindMismatch
                   (comp, Option.get (init_spec_law_name spec),
                    "an integer compartment") :: !errors
     | InitReal _ when not real && SS.mem comp comp_names ->
       errors := InitLawKindMismatch
                   (comp, Option.get (init_spec_law_name spec),
                    "a real compartment") :: !errors
     | InitCount _ | InitReal _ -> ());
    if init_spec_law_name spec <> None
       && (match m.balance with Some b -> b.balance_target = comp | None -> false)
    then errors := InitLawOnBalanceTarget comp :: !errors
  ) m.initial_conditions;

  (* Initial-condition dependency cycle. An init entry may read another
     compartment's initial value, so the entries are evaluated in topological
     order; `A = B + 1` beside `B = A - 1` has no order to evaluate in and no
     value to report. Rejecting it here is what lets both the runtime and the
     IC-gradient inlining assume a DAG. *)
  (match Init_order.topo m.initial_conditions ~bindings:m.bindings with
   | Ok _ -> ()
   | Error cyc -> errors := InitDependencyCycle cyc :: !errors);

  (* Intervention/event action targets. A dangling target is a silent
     no-op at best. The expander enforces this at the frontend (E265); this is
     the contract-boundary net, and it mirrors the Rust validator
     (`rust/crates/ir/src/validate.rs`, `check_target`) so the two sides agree
     by construction rather than by comment. *)
  List.iter (fun (iv : intervention) ->
    let site = InIntervention iv.name in
    let check_target c =
      if not (SS.mem c comp_names)
      then errors := UnknownCompartment (c, site) :: !errors
    in
    List.iter (fun a ->
      match a with
      | FractionTransfer ft -> check_target ft.src; check_target ft.dst;
                               check_expr_r ~site ft.fraction
      | AbsoluteTransfer at -> check_target at.src; check_target at.dst;
                               check_expr_r ~site at.count
      | Set s               -> check_target s.compartment;
                               check_expr_r ~site s.value
      | AddAction a         -> check_target a.add_compartment;
                               check_expr_r ~site a.add_count
    ) iv.actions
  ) m.interventions;

  (* Quantities: a non-finite literal in a reported summary is the same
     mistake (E520) as in a rate. *)
  List.iter (fun (q : quantity) ->
    List.iter (check_finite ~site:(InQuantity q.q_name) errors) (quantity_exprs q)
  ) m.quantities;

  if !errors = [] then Ok ()
  else Error (List.rev !errors)
