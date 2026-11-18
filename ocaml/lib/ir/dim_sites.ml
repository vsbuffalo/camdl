(** Every position of an expanded model that holds an expression, each with
    the dimension its expression must have: the list the dimension checker
    reads, and the only one.

    {!sites} turns an {!Ir.model} into a list of {!site}s. A site pairs one
    expression with an {!obligation}: a fixed dimension ([Has]: a transition
    rate is [P·T⁻¹], an initial value P, a forcing's period T), the dimension
    of a name ([Equals]: a Normal likelihood's mean has the dimension of the
    column it scores; [Defines]: a binding's body gives the binding its
    dimension), or nothing beyond the operator rules ([Free]). The site also
    says how bare literals in it are read ({!literals}), which diagnostic
    code a violation reports, which construct it belongs to, for locating the
    diagnostic ({!subject}), and which observation stream its [projected]
    and column references read. {!Dimcheck} turns each site into equations.
    In

    {@camdl[
      time_unit = 'days
      compartments { S, I, R }
      parameters {
        beta  : rate
        gamma : rate
      }
      let N = S + I + R
      transitions {
        infection : S --> I @ beta * S * I / N
        recovery  : I --> R @ gamma * I
      }
      init { S = 990  I = 10 }
    ]}

    the sites are the binding [N] (obligation [Defines (K_binding "N")]), the
    two rates ([Has P·T⁻¹], code E300, literals as whole terms) and the two
    initial values ([Has P], code E313).

    The list is complete by construction. {!sites} destructures every IR
    record it visits with a closed pattern and matches every variant without
    a catch-all, with warnings 4, 8 and 9 as errors, so an IR field or
    constructor this module has no rule for is a build error until someone
    writes its rule or an exemption. A field that holds an expression and is bound to [_] would
    still slip through; the coverage test [ocaml/test/test_dim_sites.ml]
    closes that gap by requiring every expression node of the serialized
    model to be the expression of some site, or to lie under an exempt key
    (the derivative maps autodiff writes, the lineage weights the expander
    derives from a checked rate, and the bindings loop-invariant code motion
    creates after the check).

    Sites are listed bindings first, then transitions, ODE equations,
    forcings, tables, interventions, observations, parameter priors, initial
    conditions, the balance expression, quantities and contrasts. The order
    matters: the solver refuses the later of two equations that disagree,
    and a binding's definition should not be the one blamed for a mistake in
    a use of it.

    Some positions are deliberately not checked. A forcing argument or an
    inline table cell that is a constant expression is read in the forcing's
    or table's own declared unit, so it is [Exempt]. A hierarchical prior's
    argument is listed with strength [Checked] and enters no equation;
    {!Dimcheck} checks the unit literals of every prior separately. Schedule
    times, trigger windows, cooldowns and similar times are numbers in the
    IR, not expressions, so they have no site here. *)

[@@@warning "@4@8@9"]
(* 4: a [_] catch-all over a variant is an error, so a new constructor cannot
      fall into an existing arm.
   8: a non-exhaustive match is an error.
   9: a record pattern that omits a field is an error, so a new field cannot be
      skipped by not naming it. *)

open Ir

(** A dimension as exponents of population and time; the same
    representation as {!Dimcheck.dim_vec}, which this module cannot name
    because {!Dimcheck} depends on it. *)
type dim_vec = int array

(** The construct a diagnostic concerns. The IR carries no source locations,
    so the compiler looks up the construct's declaration to place the
    diagnostic. A site without a subject (the balance expression) yields a
    diagnostic with no location. *)
type subject =
  | STransition  of string   (** A transition, by expanded name. *)
  | SOde         of string   (** The ODE equation of the named compartment. *)
  | SObservation of string   (** An observation stream. *)
  | SBinding     of string   (** A shared binding. The compiler has no location for one. *)
  | SContrast    of string   (** A contrast between runs. *)
  | SInit        of string   (** The initial condition of the named compartment. *)
  | SParameter   of string   (** A parameter, for a diagnostic about its prior or bounds. *)
  | SQuantity    of string   (** A generated quantity. *)
  | SIntervention of string   (** An intervention, event or reactive trigger. *)
  | SLet         of string   (** A typed [let], for its annotation. *)
  | SForcing     of string   (** A forcing, by expanded name, for its arguments. *)
  | STable       of string   (** A table, for its inline cells. *)

(** A name whose dimension positions of the model refer to: each key is one
    unknown of the dimension checker's system. *)
type key =
  | K_param    of string   (** A parameter or a typed constant. *)
  | K_table    of string   (** The dimension of a table's cells. *)
  | K_binding  of string   (** A shared binding. *)
  | K_stream   of string   (** The dimension of an observation stream's projection, what [projected] reads. *)
  | K_column   of string * string   (** A column of an observation stream, as (stream, column). *)
  | K_quantity of string   (** The dimension a generated quantity reports. *)
  | K_series   of string   (** The series a reduced quantity reduces, by the quantity's name. *)

(** The expression at a site, in whichever of the IR's expression languages
    the position uses. *)
type site_expr =
  | E        of expr               (** A model expression. *)
  | Scalar   of scalar_expr     (** A derived quantity's body: arithmetic over other quantities and parameters. *)
  | Contrast of contrast_expr      (** A contrast's body: arithmetic over the members of runs. *)

(** Why a position is not read by the checker. *)
type exemption =
  | Constant_argument   (** A forcing argument or inline table cell fixed at
                            compile time is read in the forcing's or table's
                            own declared unit. *)

(** What a site requires of its expression's dimension. *)
type obligation =
  | Has     of dim_vec   (** A fixed dimension: [P·T⁻¹] for a rate, P for a count, T for a time, a pure number for a probability. *)
  | Equals  of key       (** The dimension of [key], which other positions may also determine. *)
  | Equals_t of key      (** The T exponent of [key]'s dimension, with P unconstrained. Every equation the expression generates holds on T alone, except those requiring a function's argument to be a pure number. Used for the [sd] of a Normal likelihood on a [count] column. *)
  | Defines of key       (** The expression gives [key] its dimension: a binding's body, a derived projection, a quantity. The checker treats it as [Equals]; the difference is one of reading. *)
  | Free                 (** Nothing beyond the operator rules inside the expression. *)
  | Exempt  of exemption (** The expression is not read at all. *)

(** How bare numeric literals in a site's expression are read. Zero takes
    any dimension in either case. *)
type literals =
  | Literals_dimensionless   (** Every other literal is a pure number. *)
  | Whole_terms              (** A literal, or a constant expression of
                                literals, that is a whole additive term of
                                the expression takes the position's
                                dimension: [I = 10], [S = N0 - 50],
                                [beta * S * I / N + 0.5] as a rate. Inside a
                                product, quotient or function it is still a
                                pure number. *)

(** Whether a site's equations enter the solve. *)
type strength =
  | Binding   (** A model position: its equations enter the solve. *)
  | Checked   (** A hierarchical prior's argument: listed so the coverage test sees it, and never turned into equations. *)

(** One position of the model that holds an expression, with what its
    dimension must be. *)
type site = {
  subject    : subject option;
  role       : string;           (** The position in words, for messages: ["transition 'infection' rate"]. *)
  expr       : site_expr;
  obligation : obligation;
  literals   : literals;
  code       : string;           (** The diagnostic code a violation reports. *)
  strength   : strength;
  stream     : string option;    (** The observation whose [projected] and column references this expression reads. *)
  note       : string option;    (** Guidance carried as the hint of the diagnostic when the expression breaks a [Has] obligation. *)
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
  | ObsColumnRef _ | ObsAnchor _ | ForcingBasis _ | UncheckedDim _ | UnitLiteral _ -> false

(* ── Building sites ────────────────────────────────────────────────────── *)

(** The sites found so far while walking a model, newest first. *)
type acc = { mutable out : site list }

let add acc ?(subject : subject option) ?(literals = Literals_dimensionless) ?(strength = Binding)
    ?stream ?note ~role ~code obligation expr =
  acc.out <- { subject; role; expr; obligation; literals; code; strength; stream; note }
             :: acc.out


let transition_sites acc (tr : transition) =
  let { name; stoichiometry = _; rate; metadata = _; draw_method;
        rate_grad = _;          (* derivative map: autodiff's, not the user's *)
        rate_state_grad = _;    (* derivative map *)
        lineage } = tr in
  let subject = STransition name in
  add acc ~subject ~role:(Printf.sprintf "transition '%s' rate" name) ~code:"E300"
    ~literals:Whole_terms (Has rate_total) (E rate);
  (match draw_method with
   | DrawOverdispersed { sigma_sq; sigma_sq_grad = _ (* derivative map *) } ->
     (* The runtime draws a unit-mean Gamma multiplier with shape dt/σ² and
        scale σ²/dt (rng.rs), Bretó et al. (2009) gamma white noise: a shape
        is a pure number, so σ² has the dimension of dt. *)
     add acc ~subject ~role:(Printf.sprintf "overdispersion sigma^2 in '%s'" name)
       ~code:"E308"
       ~note:"σ² is the variance of the integrated noise per unit time; with \
              time_unit = 'days it is in days. Declare it `positive dim T`."
       (Has time) (E sigma_sq)
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
    ~literals:Whole_terms (Has rate_total) (E derivative)

let forcing_sites acc (tf : time_function) =
  let { name; kind; dim = (fp, ft); lag; data_source = _ } = tf in
  let subject = Some (SForcing name) in
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
       add acc ?subject ~role:(role "per") ~code:"E312" (Has time) (E per);
       (match link with
        | Link_log ->
          add acc ?subject ~role:(role "level") ~code:"E301" (Has dimensionless)
            (E level);
          add acc ?subject ~role:(role "slope") ~code:"E301" (Has per_time) (E slope);
          add acc ?subject ~role:(role "scale") ~code:"E301" (Has dimensionless) (E scale)
        | Link_identity ->
          (* w = level + slope·(t − t_m) + (scale/κ)·M·z is the forcing's value:
             the penalty Ω integrates B''² over t, so M (MᵀΩM = I) and
             κ = √(K − 2)·per^{3/2} are both T^{3/2} and cancel
             (spline_penalty.rs, compiled_model.rs). *)
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
      add acc ~subject:(STable name) ~role ~code:"E302" obligation (E e)) values

(* A transfer moves a fraction of a compartment, or a number of people; an
   assignment or an addition is a number of people. *)
let action_sites acc ?subject ~role:what (a : action) =
  let role field = Printf.sprintf "%s `%s`" what field in
  let count field e =
    add acc ?subject ~role:(role field) ~code:"E314" ~literals:Whole_terms
      (Has population) (E e) in
  match a with
  | FractionTransfer { src = _; dst = _; fraction; _ } ->
    add acc ?subject ~role:(role "fraction") ~code:"E314" (Has dimensionless) (E fraction)
  | AbsoluteTransfer { src = _; dst = _; count = c } -> count "count" c
  | Set { compartment = _; value; _ } -> count "value" value
  | AddAction { add_compartment = _; add_count; _ } -> count "count" add_count

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
  let subject = Some (SIntervention name) in
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
  (* An expected count or a denominator; a literal term is a count (`n = 100`). *)
  let count what e =
    add acc ~subject ~stream ~literals:Whole_terms ~code:"E304"
      ~role:(Printf.sprintf "observation '%s': %s" name what) (Has population) (E e)
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
  | Estimated { est_init = _; est_bounds = _; est_prior } ->
    (match est_prior with
     | Flat | Dist _ -> ()
     | Hierarchical { hkind = _; hargs; hpool_over = _ } ->
       List.iter (fun (arg, e) ->
         (* Checked after the solve by [Dimcheck]'s prior pass, from every
            prior's arguments; never an equation. *)
         add acc ~subject:(SParameter name) ~strength:Checked ~code:"E316"
           ~role:(Printf.sprintf "prior of '%s' `%s`" name arg) Free (E e))
         hargs)

let binding_site acc ({ bname; bexpr } : binding) =
  add acc ~subject:(SBinding bname) ~role:(Printf.sprintf "binding '%s'" bname)
    ~code:"E302" (Defines (K_binding bname)) (E bexpr)

let init_sites acc ((comp, spec) : string * init_spec) =
  let subject = SInit comp in
  let need what e want =
    add acc ~subject ~literals:Whole_terms ~code:"E313"
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
    ~literals:Whole_terms (Has population) (E balance_expr)

let quantity_sites acc (q : quantity) =
  let { q_name; q_stratum = _; q_body; q_dimension = _ (* written by the checker *) } = q in
  let subject = Some (SQuantity q_name) in
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
  let acc = { out = [] } in
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
