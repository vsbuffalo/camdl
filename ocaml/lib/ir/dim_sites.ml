(* Every expression-bearing position of an expanded model, with the dimension
   its expression must have.

   This module is the list of positions the dimension checker reads, and the
   only one. [sites] destructures every IR record it visits with a closed
   pattern and every variant without a catch-all, under warnings 4, 8 and 9 as
   errors: a new IR field or constructor does not compile here until someone
   writes its rule or an exemption. The coverage test (ocaml/test/
   test_dim_sites.ml) checks the other half, that no expression-bearing field
   is bound to [_]: every expression node of the serialized model must be the
   expression of some site, or lie under an exempt key.

   The rules are those of proposal 2026-11-06-dimension-checking-as-one-
   constraint-system §5.1. *)

[@@@warning "@4@8@9"]
(* 4: a [_] catch-all over a variant is an error, so a new constructor cannot
      fall into an existing arm.
   8: a non-exhaustive match is an error.
   9: a record pattern that omits a field is an error, so a new field cannot be
      skipped by not naming it. *)

open Ir

type dim_vec = int array

(* The construct a diagnostic concerns; the compiler maps it to a source
   location (the IR carries no spans). *)
type subject =
  | STransition  of string
  | SOde         of string   (* compartment whose ODE derivative *)
  | SObservation of string
  | SBinding     of string
  | SContrast    of string
  | SInit        of string   (* compartment whose initial condition *)
  | SParameter   of string   (* parameter whose prior *)

(* Things whose dimension other positions refer to. *)
type key =
  | K_param    of string   (* a parameter or a typed constant *)
  | K_table    of string   (* a table's cell dimension *)
  | K_binding  of string
  | K_stream   of string   (* an observation stream's projected dimension *)
  | K_column   of string * string   (* (stream, column) *)
  | K_quantity of string   (* a generated quantity's reported dimension *)
  | K_series   of string   (* the series a quantity reduces *)

type site_expr =
  | E        of expr
  | Scalar   of scalar_expr     (* a derived quantity's body *)
  | Contrast of contrast_expr

type exemption =
  | Constant_argument   (* a forcing argument fixed at compile time is read
                           in the forcing's own unit *)

type obligation =
  | Has     of dim_vec   (* a fixed dimension: P·T⁻¹, P, 1, T, ... *)
  | Equals  of key       (* the dimension of [key] *)
  | Equals_t of key      (* the T exponent of [key]'s; P unconstrained, and
                            only the T components of the expression's own
                            equations hold (count noise, §5.6) *)
  | Defines of key       (* this expression's dimension is [key]'s *)
  | Free                 (* operator rules only; nothing required of the value *)
  | Exempt  of exemption (* not read *)

type literals =
  | Literals_dimensionless   (* a literal is dimensionless; 0 takes any *)
  | Whole_terms              (* a literal that is an entire additive term of
                                the expression takes the position's dimension:
                                `count = 50`, `at [t_seed + 7]` (§5.2) *)
  | Literals_free            (* every literal takes the dimension its context
                                needs: initial conditions, until the literal
                                rule of §5.2 lands *)

type strength =
  | Binding   (* a model position: its equations enter the solve *)
  | Checked   (* a prior argument: never constrains the model (§5.9) *)

type site = {
  id         : int;
  subject    : subject option;
  role       : string;           (* "transition 'infection' rate", ... *)
  expr       : site_expr;
  obligation : obligation;
  literals   : literals;
  code       : string;           (* the code a violated obligation reports *)
  strength   : strength;
  stream     : string option;    (* the observation whose Projected and
                                    column references this expression reads *)
  note       : string option;    (* guidance a violation's diagnostic carries *)
}

let dimensionless = [| 0; 0 |]
let population    = [| 1; 0 |]
let time          = [| 0; 1 |]
let rate_total    = [| 1; -1 |]
let per_time      = [| 0; -1 |]

(* An expression with no leaf but literals: fixed at compile time. *)
let rec is_constant_expr (e : expr) : bool =
  match e with
  | Const _ -> true
  | BinOp { op = _; left; right } -> is_constant_expr left && is_constant_expr right
  | UnOp { op = _; arg } -> is_constant_expr arg
  | Cond { pred; then_; else_ } ->
    is_constant_expr pred && is_constant_expr then_ && is_constant_expr else_
  | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | TableLookup _ | Reduce _ | BindingRef _ | PerEvalRef _ | Projected
  | ObsColumnRef _ | ObsAnchor _ | ForcingBasis _ | UncheckedDim _ -> false

(* ── Building sites ────────────────────────────────────────────────────── *)

type acc = { mutable next : int; mutable out : site list }

let add acc ?(subject : subject option) ?(literals = Literals_dimensionless) ?(strength = Binding)
    ?stream ?note ~role ~code obligation expr =
  let s = { id = acc.next; subject; role; expr; obligation; literals; code;
            strength; stream; note } in
  acc.next <- acc.next + 1;
  acc.out <- s :: acc.out


let transition_sites acc (tr : transition) =
  let { name; stoichiometry = _; rate; metadata = _; draw_method;
        rate_grad = _;          (* derivative map: autodiff's, not the user's *)
        rate_state_grad = _;    (* derivative map *)
        lineage } = tr in
  let subject = STransition name in
  add acc ~subject ~role:(Printf.sprintf "transition '%s' rate" name) ~code:"E300"
    (Has rate_total) (E rate);
  (match draw_method with
   | DrawOverdispersed { sigma_sq; sigma_sq_grad = _ (* derivative map *) } ->
     add acc ~subject ~role:(Printf.sprintf "overdispersion sigma^2 in '%s'" name)
       ~code:"E308" (Has dimensionless) (E sigma_sq)
   | DrawPoisson | DrawDeterministic -> ());
  match lineage with
  | None -> ()
  | Some { is_lineage_event = _; parent_pool_weights = _ } ->
    (* Each weight is the rate with one parent count factored out, derived by
       the expander from the rate checked above. *)
    ()

let ode_site acc ({ compartment; derivative } : ode_equation) =
  add acc ~subject:(SOde compartment)
    ~role:(Printf.sprintf "ODE d(%s)/dt" compartment) ~code:"E306"
    (Has rate_total) (E derivative)

let forcing_sites acc (tf : time_function) =
  let { name; kind; dim = (fp, ft); lag; data_source = _ } = tf in
  let subject = None in
  let value_dim = [| fp; ft |] in
  let need what e want =
    let role = Printf.sprintf "forcing '%s': `%s`" name what in
    if is_constant_expr e then
      add acc ?subject ~role ~code:"E312" (Exempt Constant_argument) (E e)
    else add acc ?subject ~role ~code:"E312" (Has want) (E e)
  in
  let need_all what es want =
    List.iteri (fun k e -> need (Printf.sprintf "%s[%d]" what k) e want) es in
  (* A log-link coefficient is a log-value: dimensionless, constant or not. *)
  let need_log what e =
    add acc ?subject ~role:(Printf.sprintf "forcing '%s': %s" name what) ~code:"E301"
      ~note:"under `link = log` the coefficients are log-values, and the forcing \
             takes its unit from its declaration; give the coefficient's \
             parameter `dim 1`"
      (Has dimensionless) (E e)
  in
  (match lag with
   | None -> ()
   | Some e ->
     add acc ?subject ~role:(Printf.sprintf "forcing '%s': `lag`" name) ~code:"E309"
       ~note:"it is subtracted from the evaluation time; use a duration literal \
              (`lag = 10 'days`) or a parameter declared `: duration`"
       (Has time) (E e));
  match kind with
  | Sinusoidal { amplitude; period; phase; baseline } ->
    need "amplitude" amplitude value_dim;
    need "baseline" baseline value_dim;
    need "period" period time;
    need "phase" phase time
  | Piecewise { breakpoints; values } ->
    need_all "breakpoints" breakpoints time;
    need_all "values" values value_dim
  | Interpolated { times; values; method_ = _ } ->
    need_all "times" times time;
    need_all "values" values value_dim
  | Periodic { period; values } ->
    need "period" period time;
    need_all "values" values value_dim
  | Fourier { period; harmonics } ->
    need "period" period time;
    List.iteri (fun k (a, b) ->
      need (Printf.sprintf "harmonics[%d] cos" k) a value_dim;
      need (Printf.sprintf "harmonics[%d] sin" k) b value_dim) harmonics
  | PeriodicSpline { period; n_basis = _; degree = _; coefs; link } ->
    need "period" period time;
    (match link with
     | Link_identity -> need_all "coefs" coefs value_dim
     | Link_log -> List.iteri (fun k c -> need_log (Printf.sprintf "coefficient %d" (k + 1)) c) coefs)
  | BSpline { knots; degree = _; coefs; link } ->
    (match knots with
     | Knots_uniform { lower; upper; n_intervals = _ } ->
       need "knot lower" lower time; need "knot upper" upper time
     | Knots_explicit ks -> need_all "knots" ks time);
    (match coefs, link with
     | Coefs_explicit cs, Link_identity -> need_all "coefs" cs value_dim
     | Coefs_explicit cs, Link_log ->
       List.iteri (fun k c -> need_log (Printf.sprintf "coefficient %d" (k + 1)) c) cs
     | Coefs_smooth { level; slope; scale; modes; per }, link ->
       let role what = Printf.sprintf "forcing '%s': smooth `%s`" name what in
       add acc ?subject ~role:(role "per") ~code:"E301" (Has time) (E per);
       (match link with
        | Link_log ->
          add acc ?subject ~role:(role "level") ~code:"E301" (Has dimensionless)
            (E level);
          add acc ?subject ~role:(role "slope") ~code:"E301" (Has per_time) (E slope);
          add acc ?subject ~role:(role "scale") ~code:"E301" (Has dimensionless) (E scale)
        | Link_identity ->
          (* w = level + slope·(t − t_m) + (scale/κ)·M·z is the forcing's value:
             the penalty Ω integrates B''² over t, so M (MᵀΩM = I) and
             κ = √(K − 2)·per^{3/2} are both T^{3/2} and cancel (§5.8,
             spline_penalty.rs, compiled_model.rs). *)
          add acc ?subject ~role:(role "level") ~code:"E312" (Has value_dim) (E level);
          add acc ?subject ~role:(role "slope") ~code:"E312"
            (Has (Array.mapi (fun i x -> x + per_time.(i)) value_dim)) (E slope);
          add acc ?subject ~role:(role "scale") ~code:"E312" (Has value_dim) (E scale));
       (* the modes are standard normal draws: pure numbers, under either link *)
       List.iteri (fun k e ->
         add acc ?subject ~role:(role (Printf.sprintf "mode %d" k)) ~code:"E312"
           (Has dimensionless) (E e)) modes)

let table_sites acc (tbl : table) =
  let { name; source; out_of_bounds = _; cell_kind = _ } = tbl in
  match source with
  | External _ -> ()
  | Inline values ->
    List.iteri (fun k e ->
      let role = Printf.sprintf "table '%s' value %d" name k in
      (* A literal cell is read in the table's declared unit, like a constant
         forcing argument. *)
      let obligation =
        if is_constant_expr e then Exempt Constant_argument else Equals (K_table name) in
      add acc ~role ~code:"E302" obligation (E e)) values

(* A transfer moves a fraction of a compartment, or a number of people; an
   assignment or an addition is a number of people. *)
let action_sites acc ?subject ~role:what (a : action) =
  let role field = Printf.sprintf "%s `%s`" what field in
  let count field e =
    add acc ?subject ~role:(role field) ~code:"E314" ~literals:Whole_terms
      (Has population) (E e) in
  match a with
  | FractionTransfer { src = _; dst = _; fraction } ->
    add acc ?subject ~role:(role "fraction") ~code:"E314" (Has dimensionless) (E fraction)
  | AbsoluteTransfer { src = _; dst = _; count = c } -> count "count" c
  | Set { compartment = _; value } -> count "value" value
  | AddAction { add_compartment = _; add_count } -> count "count" add_count

(* A threshold is compared with the observed values of a stream, so it has
   the dimension of that stream's scored column; a literal threshold is read
   in it. Summing or averaging observations keeps their dimension. *)
let rec trigger_sites acc ~scored_of ?subject ~role (t : trigger_expr) =
  match t with
  | TECmp (TQObserved { stream; window = _; reducer = _ }, _cmp, threshold) ->
    let e = match threshold with TTConst v -> Const v | TTParam p -> Param p in
    let obligation = match scored_of stream with
      | Some col -> Equals (K_column (stream, col))
      | None -> Free in
    add acc ?subject ~role:(Printf.sprintf "%s threshold on '%s'" role stream) ~code:"E315"
      ~literals:Whole_terms obligation (E e)
  | TEAnd (a, b) | TEOr (a, b) ->
    trigger_sites acc ~scored_of ?subject ~role a; trigger_sites acc ~scored_of ?subject ~role b
  | TENot a -> trigger_sites acc ~scored_of ?subject ~role a

let intervention_sites acc ~scored_of (iv : intervention) =
  let { name; base_name = _; fire; actions; kind = _ } = iv in
  let subject = None in
  let role = Printf.sprintf "intervention '%s'" name in
  (match fire with
   | Scheduled sched ->
     (match sched with
      | AtTimes _ | Recurring _ -> ()
      | AtTimesExpr es ->
        List.iter (fun e -> add acc ?subject ~role:(role ^ " time") ~code:"E315"
                              ~literals:Whole_terms (Has time) (E e)) es)
   | Reactive { when_; after = _; once = _; cooldown = _ } ->
     trigger_sites acc ~scored_of ?subject ~role when_);
  List.iter (action_sites acc ?subject ~role) actions

let observation_sites acc (obs : observation_model) =
  let { name; obs_source = _; columns; scored; emit_schedule = _;
        stratum = _; covers = _; projection;
        projection_state_grad = _;   (* derivative map *)
        likelihood } = obs in
  let subject = SObservation name in
  let stream = name in
  let site ?(code = "E304") what obligation e =
    add acc ~subject ~stream ~role:(Printf.sprintf "observation '%s': %s" name what)
      ~code obligation (E e) in
  (* A likelihood argument with its gradient: the expression is the position,
     the gradient is autodiff's. *)
  let dexpr ({ expr; grad = _; proj_grad = _ } : diffable) = expr in
  (match projection with
   | DerivedExpr e -> site ~code:"E302" "projection" (Defines (K_stream name)) e
   | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _
   | FlowRatio { numerator = _; denominator = _ } -> ());
  (* A bare literal count (`n = 100`) is a count by context. *)
  let count what e =
    let rec bare e = match[@warning "-4"] e with
      | Const _ -> true
      | UnOp { op = Neg; arg } -> bare arg
      | UnOp _ | BinOp _ | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt
      | Cond _ | TimeFunc _ | TableLookup _ | Reduce _ | BindingRef _
      | PerEvalRef _ | Projected | ObsColumnRef _ | ObsAnchor _ | ForcingBasis _
      | UncheckedDim _ -> false
    in
    site what (if bare e then Free else Has population) e
  in
  let prob what e = site what (Has dimensionless) e in
  match likelihood with
  | Poisson { rate } -> count "Poisson `rate`" (dexpr rate)
  | NegBinomial { mean; dispersion } ->
    count "NegBinomial `mean`" (dexpr mean);
    site ~code:"E307" "NegBinomial `dispersion`" (Has dimensionless) (dexpr dispersion)
  | Normal { mean; sd } ->
    (* The mean has the scored column's dimension. On a `count` column the
       sd's P exponent is free: count variance formulas add a term in the
       count to one in its square (He et al. 2010), consistent only because a
       count is a number of individuals. Its T exponent must still match. *)
    let column = K_column (name, scored) in
    let count_column = List.exists (fun ({ col_name; col_role } : obs_column) ->
      col_name = scored && (match col_role with
        | RoleValue Count -> true
        | RoleValue (Rate | Probability | Positive | Real | Instant | Duration)
        | RoleTime | RoleDim _ | RoleWindowStart | RoleWindowStop -> false)) columns in
    let normal what obligation e =
      add acc ~subject ~stream ~literals:Whole_terms ~code:"E304"
        ~role:(Printf.sprintf "observation '%s': Normal `%s`" name what) obligation (E e) in
    normal "mean" (Equals column) (dexpr mean);
    normal "sd" (if count_column then Equals_t column else Equals column) (dexpr sd)
  | Binomial { n; p } ->
    count "Binomial `n`" n;
    prob "Binomial `p`" (dexpr p)
  | BetaBinomial { n; alpha; beta } ->
    count "BetaBinomial `n`" n;
    prob "BetaBinomial `alpha`" (dexpr alpha);
    prob "BetaBinomial `beta`" (dexpr beta)
  | Beta { mean; concentration } ->
    prob "Beta `mean`" (dexpr mean);
    prob "Beta `concentration`" (dexpr concentration)
  | Bernoulli { p } -> prob "Bernoulli `p`" (dexpr p)
  | ZeroInflatedNegBinomial { mean; dispersion; pi } ->
    count "zero-inflated NegBinomial `mean`" (dexpr mean);
    site ~code:"E307" "zero-inflated NegBinomial `dispersion`" (Has dimensionless)
      (dexpr dispersion);
    prob "zero-inflation `pi`" (dexpr pi)

let parameter_sites acc (p : parameter) =
  let { name; value; param_kind = _; param_dim = _ } = p in
  match value with
  | Fixed _ | Required -> ()
  | Estimated { est_init = _; est_bounds = _; est_prior; est_transform = _ } ->
    (match est_prior with
     | Flat | Dist _ -> ()
     | Hierarchical { hkind = _; hargs; hpool_over = _ } ->
       List.iter (fun (arg, e) ->
         (* Checked after the solve by [Dimcheck]'s prior pass, from every
            prior's arguments; never an equation (§5.9). *)
         add acc ~subject:(SParameter name) ~strength:Checked ~code:"E316"
           ~role:(Printf.sprintf "prior of '%s' `%s`" name arg) Free (E e))
         hargs)

let binding_site acc ({ bname; bexpr } : binding) =
  add acc ~subject:(SBinding bname) ~role:(Printf.sprintf "binding '%s'" bname)
    ~code:"E302" (Defines (K_binding bname)) (E bexpr)

let init_sites acc ((comp, spec) : string * init_spec) =
  let subject = SInit comp in
  let need what e want =
    add acc ~subject ~literals:Literals_free ~code:"E313"
      ~role:(Printf.sprintf "initial value of '%s': %s" comp what) (Has want) (E e) in
  let d ({ expr; grad = _; proj_grad = _ } : diffable) = expr in
  match spec with
  | Deterministic e -> need "the value" e population
  | InitCount (InitPoisson { rate }) -> need "Poisson `rate`" (d rate) population
  | InitCount (InitBinomial { n; p }) ->
    need "Binomial `n`" n population;
    need "Binomial `p`" (d p) dimensionless
  | InitCount (InitNegBinomial { mean; dispersion }) ->
    need "NegBinomial `mean`" (d mean) population;
    need "NegBinomial `dispersion`" (d dispersion) dimensionless
  | InitReal (InitNormal { mean; sd }) ->
    need "Normal `mean`" (d mean) population;
    need "Normal `sd`" (d sd) population

let balance_site acc ({ balance_target; balance_expr } : balance_spec) =
  add acc
    ~role:(Printf.sprintf "balance expression for '%s'" balance_target) ~code:"E305"
    (Has population) (E balance_expr)

let quantity_sites acc (q : quantity) =
  let { q_name; q_stratum = _; q_body; q_dimension = _ (* written by the checker *) } = q in
  let subject = None in
  let role what = Printf.sprintf "quantity '%s' %s" q_name what in
  match q_body with
  | QBDerived se ->
    add acc ?subject ~role:(role "body") ~code:"E297" (Defines (K_quantity q_name)) (Scalar se)
  | QBReduced { source; reduce } ->
    let series = match source with
      | QSState e ->
        add acc ?subject ~role:(role "series") ~code:"E302" (Defines (K_series q_name)) (E e);
        K_series q_name
      | QSObservation stream -> K_stream stream
    in
    (* A threshold is compared with the series, so it has its dimension; a
       literal threshold is read in it. A `value_at` time is a time. *)
    let threshold e =
      add acc ?subject ~role:(role "threshold") ~code:"E317" ~literals:Whole_terms
        (Equals series) (E e) in
    match reduce with
    | None | Some RIntegral -> ()
    | Some (RValue v) ->
      (match v with
       | VFinal | VMax | VMin | VMean -> ()
       | VCountAbove e | VCountBelow e -> threshold e
       | VValueAt (ATime e) ->
         add acc ?subject ~role:(role "`value_at` time") ~code:"E317" ~literals:Whole_terms
           (Has time) (E e)
       | VValueAt (AObs { anchor = _; offset = _ }) -> ())
    | Some (RTime t) ->
      (match t with
       | TimeOfMax | TimeOfMin -> ()
       | FirstAbove e | FirstBelow e | LastAbove e | LastBelow e -> threshold e)

let contrast_site acc ({ c_name; c_body } : contrast) =
  add acc ~subject:(SContrast c_name) ~role:(Printf.sprintf "contrast '%s'" c_name)
    ~code:"E297" Free (Contrast c_body)

let sites (m : model) : site list =
  let { name = _; version = _; time_unit = _; description = _; origin = _;
        origin_rata_die = _; compartments = _; doc_index = _;
        model_structure = _; identity_tracked_compartments = _;
        output = _; simulation = _; presets = _;  (* no expressions: numbers
                                                     already in time_unit *)
        constants = _;            (* declarations, read as K_param *)
        per_eval_bindings = _;    (* created by LICM, which runs after dimcheck *)
        ic_grad = _;              (* derivative map *)
        transitions; ode_equations; time_functions; tables; interventions;
        observations; parameters; bindings; initial_conditions; balance;
        quantities; contrasts } = m in
  let acc = { next = 0; out = [] } in
  (* Bindings first: each defines a name other positions use, and the solver
     blames the later of two disagreeing equations, so a definition must not
     be the one blamed for a use's mistake. *)
  List.iter (binding_site acc) bindings;
  List.iter (transition_sites acc) transitions;
  List.iter (ode_site acc) ode_equations;
  List.iter (forcing_sites acc) time_functions;
  List.iter (table_sites acc) tables;
  let scored_of stream =
    List.find_map (fun (o : observation_model) ->
      if o.name = stream then Some o.scored else None) observations in
  List.iter (intervention_sites acc ~scored_of) interventions;
  List.iter (observation_sites acc) observations;
  List.iter (parameter_sites acc) parameters;
  List.iter (init_sites acc) initial_conditions;
  Option.iter (balance_site acc) balance;
  List.iter (quantity_sites acc) quantities;
  List.iter (contrast_site acc) contrasts;
  List.rev acc.out
