open Ir

(* Where a reference error occurred — the enclosing construct, so the compiler
   can point the diagnostic at its declaration (the referenced name itself does
   not exist, so it has no decl loc). *)
type site =
  | InTransition  of string   (* transition name *)
  | InOde         of string   (* compartment whose ODE derivative *)
  | InObservation of string   (* observation name *)
  | InIntervention of string  (* intervention/event name *)

type error =
  | DuplicateCompartment  of string
  | DuplicateTransition   of string
  | DuplicateParameter    of string
  | UnknownCompartment    of string * site
  | UnknownParameter      of string * site
  | UnknownTable          of string * site
  | UnknownTimeFunction   of string * site
  | UnknownTransition     of string * site
  | DuplicateFlowInUnion  of string * site  (* gh#678: a flow union must be disjoint *)
  | RealCompartmentInStoichiometry of string * string  (* transition, compartment *)
  | MissingOdeEquation    of string
  | OdeForNonRealComp     of string
  | ZeroDelta             of string * string  (* transition, compartment *)
  | ParamInBinding        of string * string  (* binding name, param name *)
  | InitUnknownCompartment of string          (* init key naming no compartment *)
  | InitDependencyCycle   of string list      (* gh#733: init entries that read each other *)

let error_to_string = function
  | DuplicateCompartment s -> Printf.sprintf "duplicate compartment: %s" s
  | DuplicateTransition  s -> Printf.sprintf "duplicate transition: %s" s
  | DuplicateParameter   s -> Printf.sprintf "duplicate parameter: %s" s
  | UnknownCompartment  (s, _) -> Printf.sprintf "unknown compartment: %s" s
  | UnknownParameter    (s, _) -> Printf.sprintf "unknown parameter: %s" s
  | UnknownTable        (s, _) -> Printf.sprintf "unknown table: %s" s
  | UnknownTimeFunction (s, _) -> Printf.sprintf "unknown time_function: %s" s
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

module SS = Set.Make(String)

(* Does this expression reach a [Param] node? Used to enforce the
   hoist/autodiff contract: a hoisted [model.bindings] body must be
   param-free, because [autodiff.ml] differentiates [BindingRef] to 0.
   A param leaking into a binding would silently zero its gradient.
   This is a structural reachability check — it does NOT resolve
   [BindingRef] transitively (bindings are topo-ordered and each is
   checked in turn, so a param reachable through an earlier binding
   surfaces on that binding). *)
let first_param (e : expr) : string option =
  let rec go = function
    | Param p -> Some p
    | Const _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | BindingRef _
    | Projected | ObsColumnRef _ | ObsAnchor _ -> None
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

let check_expr_refs ~site ~comps ~params ~tables ~tfs errors e =
  let rec go = function
    | Const _ | Time | Dt | Projected | ObsColumnRef _ | ObsAnchor _ -> ()
    | Param p -> if not (SS.mem p params) then errors := UnknownParameter (p, site) :: !errors
    | Pop   c -> if not (SS.mem c comps)  then errors := UnknownCompartment (c, site) :: !errors
    | PopSum cs -> List.iter (fun c -> if not (SS.mem c comps) then errors := UnknownCompartment (c, site) :: !errors) cs
    | BinOp b -> go b.left; go b.right
    | UnOp u  -> go u.arg
    | Cond c  -> go c.pred; go c.then_; go c.else_
    | TimeFunc n ->
      if not (SS.mem n tfs) then errors := UnknownTimeFunction (n, site) :: !errors
    | TableLookup (t, idxs) ->
      (if not (SS.mem t tables) then errors := UnknownTable (t, site) :: !errors);
      List.iter go idxs
    | UncheckedDim u -> go u.inner
    | Reduce terms -> List.iter go terms
    | BindingRef _ -> ()   (* leaf; binding name resolution happens at CompiledModel::new *)
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  in
  go e

let validate (m : model) : (unit, error list) result =
  let errors = ref [] in

  (* Unique-name checks. The returned sets double as the
     {comps, params, tables, tfs, tr_set} used by check_expr_refs
     below — m10 in the 2026-04-19 review. Prior version bound two
     of these to `_tr_names` / `_param_names` and then rebuilt them
     via `List.map |> SS.of_list`, doing the walk twice for each
     list. *)
  let comp_names = uniq_check (fun (c: compartment)     -> c.name) m.compartments (fun n -> DuplicateCompartment n) errors in
  let tr_set     = uniq_check (fun (t: transition)      -> t.name) m.transitions  (fun n -> DuplicateTransition  n) errors in
  let params     = uniq_check (fun (p: parameter)       -> p.name) m.parameters   (fun n -> DuplicateParameter   n) errors in

  let real_comps = List.filter_map (fun (c: compartment)     -> if c.kind = Real then Some c.name else None) m.compartments |> SS.of_list in
  let tables     = List.map (fun (t: table)         -> t.name) m.tables        |> SS.of_list in
  let tfs        = List.map (fun (f: time_function) -> f.name) m.time_functions |> SS.of_list in

  let check_expr_r ~site e = check_expr_refs ~site ~comps:comp_names ~params ~tables ~tfs errors e in

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
    check_expr_r ~site:(InTransition tr.name) tr.rate
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
    (* Projection reference check. The compartment arms were missing (gh#478):
       `CurrentPop`/`CurrentPopSum` fell through the wildcard, so a projection
       naming no real cell — e.g. `prevalence(I[child])` on a `[age, patch]`
       family, which lowers to the non-existent `I_child` — passed `camdlc
       check` and only failed at run time. Transitions were already checked;
       both sides now are. *)
    (match obs.projection with
     | CumulativeFlow tn ->
       if not (SS.mem tn tr_set) then errors := UnknownTransition (tn, here) :: !errors
     | CumulativeFlowSum tns ->
       List.iter (fun tn ->
         if not (SS.mem tn tr_set) then errors := UnknownTransition (tn, here) :: !errors
       ) tns;
       (* gh#678: covers every producer, including a hand-written or generated
          IR that never passed through the expander's lowering-site check. *)
       let seen = Hashtbl.create 8 in
       List.iter (fun tn ->
         if Hashtbl.mem seen tn then
           errors := DuplicateFlowInUnion (tn, here) :: !errors
         else Hashtbl.add seen tn ()) tns
     | CurrentPop cn ->
       if not (SS.mem cn comp_names) then errors := UnknownCompartment (cn, here) :: !errors
     | CurrentPopSum cns ->
       List.iter (fun cn ->
         if not (SS.mem cn comp_names) then errors := UnknownCompartment (cn, here) :: !errors
       ) cns
     | _ -> ());
    (* Walk observation-likelihood expressions. The likelihood AST
       may reference parameters, populations, tables, and the special
       `Projected` variable; we check every identifier in the
       distribution's payload so e.g.
         cases : poisson(rate = bata * Projected)
       catches the `bata` typo here. m9 in the 2026-04-19 review —
       previously this branch was commented out, so these checks ran
       nowhere. *)
    let chk e = check_expr_r ~site:here e in
    (match obs.likelihood with
     | Poisson      { rate }                    -> chk rate.expr
     | NegBinomial  { mean; dispersion }        -> chk mean.expr; chk dispersion.expr
     | Normal       { mean; sd }                -> chk mean.expr; chk sd.expr
     | Binomial     { n; p }                    -> chk n; chk p.expr
     | BetaBinomial { n; alpha; beta }          -> chk n; chk alpha.expr; chk beta.expr
     | Beta         { mean; concentration }     -> chk mean.expr; chk concentration.expr
     | Bernoulli    { p }                       -> chk p.expr
     | ZeroInflatedNegBinomial { mean; dispersion; pi } -> chk mean; chk dispersion; chk pi)
  ) m.observations;

  (* Hoist/autodiff contract (defensive invariant). Every entry in
     [m.bindings] must be param-free: [autodiff.ml] differentiates
     [BindingRef] unconditionally to 0, so a param reachable from a
     hoisted binding body would silently zero its gradient — an
     un-estimable parameter with no error (the gh#186 failure class).
     The expander's [let_is_hoistable] only hoists param-free lets, so
     on the clean corpus this never fires; it converts a latent
     silent-wrong-gradient into a loud compile-time failure if that
     eligibility heuristic ever regresses. *)
  List.iter (fun (b : binding) ->
    match first_param b.bexpr with
    | Some p -> errors := ParamInBinding (b.bname, p) :: !errors
    | None -> ()
  ) m.bindings;

  (* Initial-condition reference check (gh#114). Every init key must name a
     real compartment in the (already fully-expanded) IR. The OCaml expander
     enforces this at the frontend (E277); this is the contract-boundary net
     so a hand-written or drifted IR cannot start a cell that doesn't exist. *)
  List.iter (fun (k, _) ->
    if not (SS.mem k comp_names)
    then errors := InitUnknownCompartment k :: !errors
  ) m.initial_conditions;

  (* Initial-condition dependency cycle (gh#733). An init entry may read another
     compartment's initial value, so the entries are evaluated in topological
     order; `A = B + 1` beside `B = A - 1` has no order to evaluate in and no
     value to report. Rejecting it here is what lets both the runtime and the
     IC-gradient inlining assume a DAG. *)
  (match Init_order.topo m.initial_conditions ~bindings:m.bindings with
   | Ok _ -> ()
   | Error cyc -> errors := InitDependencyCycle cyc :: !errors);

  (* Intervention/event action targets (gh#461). A dangling target is a silent
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

  if !errors = [] then Ok ()
  else Error (List.rev !errors)
