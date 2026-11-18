(** Symbolic differentiation of IR expressions: from the {!Ir.expr} of a
    quantity, the {!Ir.expr} of its derivative, which the runtime evaluates
    with the same evaluator as any other expression.

    The compiler's driver calls one entry point per kind of quantity, after
    validation and dimension checking and before constant folding and
    loop-invariant code motion, and stores each result in a gradient field
    of the {!Ir.model}:

    - {!differentiate_rate} fills a transition's [rate_grad], the derivative
      of its rate with respect to every parameter. The parameter update of
      the particle Gibbs sampler and the ODE gradient read it.
    - {!differentiate_rate_state} fills [rate_state_grad], the derivative of
      a rate with respect to every compartment, and
      {!differentiate_projection} fills an observation's
      [projection_state_grad]. With {!differentiate_ic}, which fills the
      model's [ic_grad] from each initial value, these are what the ODE
      forward-sensitivity gradient integrates.
    - {!differentiate_observations} fills the [grad] and [proj_grad] of
      every differentiable argument of every likelihood (a Poisson rate, a
      negative-binomial mean and dispersion, and so on): its derivative with
      respect to each parameter, and with respect to the projected value.
      {!differentiate_overdispersion} does the same for the variance of an
      overdispersed transition, and {!differentiate_initial_conditions} for
      the arguments of a drawn initial state such as
      [I ~ poisson(rate = I0)].

    One recursion, [differentiate], computes all of them. It takes a
    {!diff_target} saying what the derivative is taken with respect to, and
    returns a {!deriv}: a derivative, or a reason none can be written. The
    target decides which leaves are constant. With respect to a parameter,
    compartments are constant, because the gradient is taken with the
    trajectory held fixed, and so are bindings, which validation keeps free
    of parameters. With respect to a compartment, parameters and forcings
    are constant and a binding is differentiated through its body. A
    forcing whose coefficients are parameters has a real derivative: a
    sinusoid or a Fourier series is differentiated through its closed form,
    and a spline through its coefficients, each multiplying a basis-function
    node ({!Ir.ForcingBasis}) that the runtime evaluates. A table cell
    selected by a constant index is differentiated through the cell's
    expression.

    A key absent from a gradient map means the derivative is exactly zero:
    a derivative that simplifies to [Const 0.0] is dropped. Where no
    derivative can be written, the drivers differ. For a transition's rate
    with respect to a parameter, a parameter in data that is fixed when the
    model is built (the knots or values of a [piecewise] or [interpolated]
    forcing, a spline's knots, window or period, an index into a table, an operand
    of [mod]) is the compile error [E600]. Every other case becomes an
    {!Ir.DEUnsupported} entry naming the reason: a live coefficient whose
    derivative is not written (a [periodic] forcing's values or period, a
    forcing's [lag], a table value selected by an index that is not
    constant), a function that is not smooth in the state ([floor], [abs]
    or [min] of a compartment), and, outside the rate's parameter gradient,
    the structural cases too. The model still compiles, simulates and fits
    by methods that use no gradient; the runtime's gradient gate refuses a
    gradient-based fit that estimates a parameter with such an entry.

    The derivatives are simplified only by {!simplify_fixpoint}, which
    removes additive and multiplicative identities, folds arithmetic on
    numeric literals, and drops zero terms of a sum. It does no algebra
    beyond that, so a quotient keeps the form the quotient rule gives it.
    For the transition

    {@camdl[
      recovery : I --> R @ I / d
    ]}

    with [d] a parameter, the derivative of the rate with respect to [I] is
    written as [d / (d * d)]:

    {@json[
      "rate_state_grad":{"I":{"grad":{"bin_op":{"op":"div","left":{"param":"d"},"right":{"bin_op":{"op":"mul","left":{"param":"d"},"right":{"param":"d"}}}}}}}
    ]}

    Every match over {!Ir.expr} here names each constructor, and the library
    is built with warning 8 as an error, so a new expression constructor is
    a compile error here until its derivative is written, rather than a
    silent zero. A {!Ir.PerEvalRef} cannot reach this module: loop-invariant
    code motion introduces it and runs after differentiation, and meeting
    one raises [Failure]. *)

open Ir

(** The derivative of one expression with respect to one {!diff_target}, or
    the reason the compiler cannot write it.

    The two refusals are kept apart because the drivers treat them
    differently, and a refusal is kept apart from [Known (Const 0.0)] so that
    a derivative the compiler did not write is never mistaken for one that
    is zero. Both carry a [code], the stable reason written to the IR, a
    [node] naming the construct ("forcing `beta_t`", "mod expression"), and
    a [reason], the sentence of the error message. When subexpressions
    combine, [Unsupported] wins over [Omitted], which wins over [Known]. *)
type deriv =
  | Known of expr
      (** The derivative, which may be [Const 0.0]. *)
  | Omitted of { node : string; reason : string; code : unsupported_reason }
      (** The value is a live coefficient that the runtime evaluates from
          the parameters, but its derivative is not written: a [periodic]
          forcing's values or period, a forcing's [lag], or a table value
          selected by an index that is not constant. Every driver turns it
          into an {!Ir.DEUnsupported} entry. *)
  | Unsupported of { node : string; reason : string; code : unsupported_reason }
      (** The derivative does not exist or cannot be written: a parameter in
          data fixed when the model is built, such as forcing knots or a
          table index, an operand of [mod], or a function that is not smooth
          in the target. {!differentiate_rate} turns it into the error
          [E600], the other drivers into an {!Ir.DEUnsupported} entry. *)

let map1 (d : deriv) (f : expr -> expr) : deriv =
  match d with
  | Known e -> Known (f e)
  | (Omitted _ | Unsupported _) as nd -> nd

(* Combine two sub-derivatives under a calculus rule. Precedence:
   [Unsupported] dominates [Omitted] dominates [Known] — a structural refusal
   outranks a live-but-omitted one, which outranks a real derivative. *)
let map2 (da : deriv) (db : deriv) (f : expr -> expr -> expr) : deriv =
  match da, db with
  | (Unsupported _ as u), _ | _, (Unsupported _ as u) -> u
  | (Omitted _ as o), _ | _, (Omitted _ as o) -> o
  | Known a, Known b -> Known (f a b)

(** Does [param] appear syntactically in [e]? Forcings/tables are opaque here
    (their definitions are checked separately by the [differentiate] lookups);
    this is the operand check for [Mod] and table indices. *)
let rec mentions (param : string) (e : expr) : bool =
  match e with
  | Param n -> n = param
  | ConstRef _ -> false   (* a model constant is not a parameter *)
  | Const _ | Pop _ | Time | Dt | Projected | ObsColumnRef _ | ObsAnchor _ -> false
  | PopSum _ -> false
  | BinOp b -> mentions param b.left || mentions param b.right
  | UnOp u -> mentions param u.arg
  | Cond c -> mentions param c.pred || mentions param c.then_ || mentions param c.else_
  | TimeFunc _ | ForcingBasis _ -> false
  | TableLookup (_, args) -> List.exists (mentions param) args
  | UncheckedDim u -> mentions param u.inner
  | UnitLiteral _ -> false
  | Reduce terms -> List.exists (mentions param) terms
  | BindingRef _ -> false   (* hoisted bindings are param-free (state-only) *)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** Does [e] reference the projection output ([Projected])? The [WrtProjected]
    analogue of [mentions]: used to detect a nonsmooth function of the projection
    ([floor]/[min]/… of [projected]) — undifferentiable w.r.t. the projected value.
    [Projected] only appears in likelihood argument expressions, so bindings /
    forcings / tables never carry it. *)
let rec mentions_projected (e : expr) : bool =
  match e with
  | Projected -> true
  | Param _ | ConstRef _ | Const _ | Pop _ | PopSum _ | Time | Dt | ObsColumnRef _
  | ObsAnchor _ -> false
  | BinOp b -> mentions_projected b.left || mentions_projected b.right
  | UnOp u -> mentions_projected u.arg
  | Cond c -> mentions_projected c.pred || mentions_projected c.then_ || mentions_projected c.else_
  | TimeFunc _ | ForcingBasis _ -> false
  | TableLookup (_, args) -> List.exists mentions_projected args
  | UncheckedDim u -> mentions_projected u.inner
  | UnitLiteral _ -> false
  | Reduce terms -> List.exists mentions_projected terms
  | BindingRef _ -> false
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** The variable a derivative is taken with respect to.

    Only the leaves and the functions that are not smooth behave differently
    for each target; the calculus rules are shared. *)
type diff_target =
  | WrtParam of string
      (** The parameter of this name, for every [*_grad] map keyed by
          parameter. Compartments are constant, since the gradient is taken
          with the trajectory fixed, and so are bindings, which hold no
          parameters. *)
  | WrtPop of string
      (** The compartment of this name, for [rate_state_grad] and
          [projection_state_grad]. Parameters and forcings are constant; a
          binding is differentiated through its body; [floor], [ceil],
          [abs], [min], [max] or [mod] of an expression that reads the
          compartment is refused. *)
  | WrtProjected
      (** The projected value ([Projected]) inside a likelihood argument,
          for [proj_grad], so the runtime can chain the derivative of the
          argument with respect to the projection into the derivative of
          the projection with respect to the parameters. Everything else is
          constant. It is meaningful only for a likelihood argument, the one
          place [Projected] occurs. *)

module SS = Set.Make (String)

(* Name lookups. A model can carry thousands of forcings and hundreds of
   bindings, and differentiation looks a name up at every reference, once per
   (rate, compartment) pair, so a linear [List.find_opt] dominated compile time
   on large models (tb_india_baseline_flux: 87% of a 56 s compile). Each list is
   indexed once and the index is reused while the same list (physically) is
   passed again; every list here is immutable, so the index is exact. The first
   entry of a name wins, as with [List.find_opt]. *)
let memo_on_list (build : 'a list -> 'b) : 'a list -> 'b =
  let last = ref None in
  fun l ->
    match !last with
    | Some (l', v) when l' == l -> v
    | _ -> let v = build l in last := Some (l, v); v

let index_by (name : 'a -> string) (l : 'a list) : (string, 'a) Hashtbl.t =
  let h = Hashtbl.create (List.length l) in
  List.iter (fun x -> if not (Hashtbl.mem h (name x)) then Hashtbl.add h (name x) x) l;
  h

let binding_index = memo_on_list (index_by (fun (b : binding) -> b.bname))
let tf_index = memo_on_list (index_by (fun (tf : time_function) -> tf.name))
let table_index = memo_on_list (index_by (fun (t : table) -> t.name))

let find_binding bindings name = Hashtbl.find_opt (binding_index bindings) name
let find_tf tfs name = Hashtbl.find_opt (tf_index tfs) name
let find_table tbls name = Hashtbl.find_opt (table_index tbls) name

(* The compartments a binding's body reaches, directly or through other
   bindings, computed once per binding. Bindings are acyclic by construction
   (topo-ordered; enforced at validate/dimcheck), so the recursion terminates. *)
let binding_reach : binding list -> string -> SS.t =
  memo_on_list (fun bindings ->
    let memo = Hashtbl.create 64 in
    let rec reach name =
      match Hashtbl.find_opt memo name with
      | Some r -> r
      | None ->
        let r = match find_binding bindings name with
          | Some b -> pops SS.empty b.bexpr
          | None -> SS.empty
        in
        Hashtbl.add memo name r; r
    and pops acc (e : expr) =
      match e with
      | Pop n -> SS.add n acc
      | PopSum members -> List.fold_left (fun a m -> SS.add m a) acc members
      | Param _ | ConstRef _ | Const _ | Time | Dt | Projected | ObsColumnRef _
      | ObsAnchor _ | TimeFunc _ | ForcingBasis _ | UnitLiteral _ -> acc
      | BindingRef name -> SS.union (reach name) acc
      | TableLookup (_, args) | Reduce args -> List.fold_left pops acc args
      | BinOp b -> pops (pops acc b.left) b.right
      | UnOp u -> pops acc u.arg
      | Cond cc -> pops (pops (pops acc cc.pred) cc.then_) cc.else_
      | UncheckedDim u -> pops acc u.inner
      | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
    in
    reach)

(** Does compartment [c]'s state reach [e] — directly ([Pop c] or a [PopSum]
    containing [c]) or through a hoisted binding's body? The [WrtPop] analogue of
    [mentions]: forcings and const-indexed table cells are state-free, so a
    non-mention proves ∂e/∂Pop(c) = 0, and a mention inside a nonsmooth op
    (Mod/Floor/Ceil/Abs/Min/Max) is the [WrtPop] refusal trigger. A binding is
    answered from its precomputed reach set. *)
let rec mentions_pop (bindings : binding list) (c : string) (e : expr) : bool =
  match e with
  | Pop n -> n = c
  | PopSum members -> List.mem c members
  | Param _ | ConstRef _ | Const _ | Time | Dt | Projected | ObsColumnRef _
  | ObsAnchor _ -> false
  | TimeFunc _ | ForcingBasis _ -> false   (* forcings are state-free *)
  | BindingRef name -> SS.mem c (binding_reach bindings name)
  | TableLookup (_, args) -> List.exists (mentions_pop bindings c) args
  | BinOp b -> mentions_pop bindings c b.left || mentions_pop bindings c b.right
  | UnOp u -> mentions_pop bindings c u.arg
  | Cond cc ->
    mentions_pop bindings c cc.pred
    || mentions_pop bindings c cc.then_
    || mentions_pop bindings c cc.else_
  | Reduce terms -> List.exists (mentions_pop bindings c) terms
  | UncheckedDim u -> mentions_pop bindings c u.inner
  | UnitLiteral _ -> false
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** Coefficient expressions of a forcing kind — used to decide whether an
    unsupported kind actually depends on the differentiation parameter. *)
let forcing_coeff_exprs (k : time_func_kind) : expr list =
  match k with
  | Sinusoidal s -> [ s.amplitude; s.period; s.phase; s.baseline ]
  | Piecewise p -> p.breakpoints @ p.values
  | Interpolated i -> i.times @ i.values
  | Periodic p -> p.period :: p.values
  | Fourier f -> f.period :: List.concat_map (fun (a, b) -> [ a; b ]) f.harmonics
  | PeriodicSpline ps -> ps.period :: ps.coefs
  | BSpline b ->
    let knots = match b.knots with
      | Knots_uniform { lower; upper; _ } -> [ lower; upper ]
      | Knots_explicit bps -> bps
    in
    let coefs = match b.coefs with
      | Coefs_explicit cs -> cs
      | Coefs_smooth { level; slope; scale; modes; per } -> level :: slope :: scale :: per :: modes
    in
    knots @ coefs

let kind_label (k : time_func_kind) : string =
  match k with
  | Sinusoidal _ -> "sinusoidal"
  | Piecewise _ -> "piecewise"
  | Interpolated _ -> "interpolated"
  | Periodic _ -> "periodic"
  | Fourier _ -> "fourier"
  | PeriodicSpline _ -> "periodic_spline"
  | BSpline _ -> "bspline"

let two_pi = 2.0 *. Float.pi

(** Closed form of a sinusoidal forcing:
    [baseline + amplitude · sin(2π(t − phase)/period)] — matching the Rust
    evaluator (`sinusoidal_value`). Differentiating this w.r.t. a coefficient
    parameter yields the analytic ∂forcing/∂coef via the ordinary rules (the
    coefficient sub-expressions carry the parameter dependence). *)
let sinusoidal_closed ?lag (s : sinusoidal) : expr =
  (* Evaluation-time shift: the runtime evaluates the forcing at
     t - lag, so the differentiated closed form must be built over (Time - lag),
     not bare Time — otherwise the emitted gradient is evaluated at t while the
     value is at t - lag. The lag_mentions guard in
     [differentiate] runs first, so here the differentiation parameter is never
     inside [lag]; the - lag term therefore rides along as a constant shift. *)
  let t = match lag with
    | Some l -> BinOp { op = Sub; left = Time; right = l }
    | None -> Time
  in
  let theta =
    BinOp { op = Div;
            left = BinOp { op = Mul; left = Const two_pi;
                           right = BinOp { op = Sub; left = t; right = s.phase } };
            right = s.period }
  in
  BinOp { op = Add; left = s.baseline;
          right = BinOp { op = Mul; left = s.amplitude;
                          right = UnOp { op = Sin; arg = theta } } }

(** Closed form of a finite Fourier series:
    [Σ_k a_k cos(2π(k+1)t/period) + b_k sin(2π(k+1)t/period)] (k 0-based,
    harmonic k+1) — matching the Rust evaluator (`fourier_value`). No baseline;
    the model author writes `1 + fourier(t)`. *)
let fourier_closed ?lag (f : fourier) : expr =
  (* Same evaluation-time shift as [sinusoidal_closed]: differentiate over
     (Time - lag), not bare Time. *)
  let t = match lag with
    | Some l -> BinOp { op = Sub; left = Time; right = l }
    | None -> Time
  in
  let term k (a, b) =
    let kf = float_of_int (k + 1) in
    let arg =
      BinOp { op = Div;
              left = BinOp { op = Mul; left = Const (two_pi *. kf); right = t };
              right = f.period }
    in
    BinOp { op = Add;
            left  = BinOp { op = Mul; left = a; right = UnOp { op = Cos; arg } };
            right = BinOp { op = Mul; left = b; right = UnOp { op = Sin; arg } } }
  in
  match List.mapi term f.harmonics with
  | [] -> Const 0.0
  | terms -> Reduce terms

(** Symbolic differentiation: ∂expr/∂param → [deriv].

    [tfs]/[tbls] let the [TimeFunc]/[TableLookup] arms reach the forcing/table
    definitions (the IR carries only their names at the use site). *)
let differentiate ?(bindings = []) (top : expr) (target : diff_target)
    (tfs : time_function list) (tbls : table list) : deriv =
  let forcing_mentions param fname =
    match find_tf tfs fname with
    | Some tf -> List.exists (mentions param) (forcing_coeff_exprs tf.kind)
    | None -> false
  in
  let table_value_mentions param name =
    match find_table tbls name with
    | Some { source = Inline vals; _ } -> List.exists (mentions param) vals
    | _ -> false
  in
  (* Does [param] drive the forcing's evaluation-time shift ([lag])?
     The closed forms below differentiate against bare [Time], not [Time − lag],
     so a param in the lag has a live derivative none of them emit. *)
  let lag_mentions param (tf : time_function) =
    match tf.lag with Some l -> mentions param l | None -> false
  in
  (* WrtPop only: does the target compartment reach [e] (through bindings too)?
     A non-mention proves ∂e/∂Pop = 0, and a mention inside a nonsmooth op is the
     refusal trigger. *)
  let hits_state e =
    match target with
    | WrtParam _ | WrtProjected -> false
    | WrtPop c -> mentions_pop bindings c e
  in
  let rec spline_deriv param fname what structural coefs link =
    if List.exists (mentions param) structural then
      Unsupported
        { code = URStructuralForcing;
          node = Printf.sprintf "forcing `%s`" fname;
          reason = Printf.sprintf
            "parameter '%s' drives the %s of spline forcing `%s`, which \
             decides where its basis functions sit and is fixed when the model \
             is built, so it cannot be an estimated parameter. A spline's \
             coefficients can be estimated; its %s must be constant" param what
            fname what }
    else
      let term j c =
        map1 (d c) (fun dc ->
          BinOp { op = Mul; left = dc; right = ForcingBasis { forcing = fname; index = j; space = Space_spline } })
      in
      let sum =
        List.fold_left (fun acc t -> map2 acc t (fun a b -> BinOp { op = Add; left = a; right = b }))
          (Known (Const 0.0)) (List.mapi term coefs)
      in
      match link with
      | Link_identity -> sum
      | Link_log -> map1 sum (fun s -> BinOp { op = Mul; left = TimeFunc fname; right = s })
  (* A smooth spline is g = level + slope (t_c - t_m) + (scale/kappa) sum_i z_i W_i,
     W_i(t) = sum_j M_ji B_j(t). A `whitened` basis node evaluates W_i/kappa and
     the `linear` node t_c - t_m, so
       dg/dtheta = d level + d slope * lin + d scale * sum_i z_i Wh_i
                   + sum_i (d z_i) * scale * Wh_i,
     each coefficient expression differentiated by the ordinary rules (the
     chain rule for free), and under `link = log` times the forcing's value.
     The knots, window and `per` place the basis or set kappa: Unsupported. *)
  and smooth_deriv param fname structural ~level ~slope ~scale ~modes ~per link =
    if List.exists (mentions param) (per :: structural) then
      Unsupported
        { code = URStructuralForcing;
          node = Printf.sprintf "forcing `%s`" fname;
          reason = Printf.sprintf
            "parameter '%s' drives the window, breakpoints or reference \
             duration `per` of smooth spline forcing `%s`, which decide where \
             its basis functions sit and the scale of its penalty and are fixed \
             when the model is built, so they cannot be an estimated parameter"
            param fname }
    else
      let wh i = ForcingBasis { forcing = fname; index = i; space = Space_whitened } in
      let lin = ForcingBasis { forcing = fname; index = 0; space = Space_linear } in
      let mul a b = BinOp { op = Mul; left = a; right = b } in
      let add a b = BinOp { op = Add; left = a; right = b } in
      let z_wh = match List.mapi (fun i z -> mul z (wh i)) modes with
        | [] -> Const 0.0
        | t :: ts -> List.fold_left add t ts
      in
      let terms =
        [ d level;
          map1 (d slope) (fun ds -> mul ds lin);
          map1 (d scale) (fun dsc -> mul dsc z_wh) ]
        @ List.mapi (fun i z -> map1 (d z) (fun dz -> mul dz (mul scale (wh i)))) modes
      in
      let sum = List.fold_left (fun acc t -> map2 acc t add) (Known (Const 0.0)) terms in
      match link with
      | Link_identity -> sum
      | Link_log -> map1 sum (fun s -> mul (TimeFunc fname) s)
  and d (e : expr) : deriv =
    match e with
    (* Constant leaves — zero for every target. `ObsAnchor` is a
       structural knot in a forcing breakpoint list, resolved to a number
       before any gradient is evaluated; it depends on the data, never on a
       parameter or a compartment, so its derivative is a genuine zero. *)
    | Const _ | Time | Dt | ObsColumnRef _ | ObsAnchor _ -> Known (Const 0.0)

    (* A model constant: a literal by another name, not a parameter and not
       state — zero for every target. *)
    | ConstRef _ -> Known (Const 0.0)

    (* The projection output. The target leaf under [WrtProjected] (∂projected/∂projected
       = 1, the source of the observation factor-2 chain); constant w.r.t. a
       parameter or a compartment (it is an independent input to the likelihood). *)
    | Projected ->
      (match target with
       | WrtProjected -> Known (Const 1.0)
       | WrtParam _ | WrtPop _ -> Known (Const 0.0))

    (* Dimensional escape: differentiate the inner, drop the wrapper. *)
    | UncheckedDim u -> d u.inner

    (* A unit literal is a known constant. *)
    | UnitLiteral _ -> Known (Const 0.0)

    (* State. Constant in the θ|X step (WrtParam) and w.r.t. the projection output
       (WrtProjected); a Kronecker delta w.r.t. the target compartment (WrtPop) —
       the source of the on-diagonal J_x, and via PopSum the off-diagonal coupling
       (force of infection). *)
    | Pop n ->
      (match target with
       | WrtParam _ | WrtProjected -> Known (Const 0.0)
       | WrtPop c -> Known (Const (if n = c then 1.0 else 0.0)))
    | PopSum members ->
      (match target with
       | WrtParam _ | WrtProjected -> Known (Const 0.0)
       (* A member may repeat (`I + I` normalises to [PopSum ["I"; "I"]]) and the
          runtime sums every occurrence, so the derivative is the multiplicity. *)
       | WrtPop c ->
         Known (Const (float_of_int (List.length (List.filter (( = ) c) members)))))

    (* Parameter reference — 1 if it's the target param (WrtParam); constant
       w.r.t. state (WrtPop) and w.r.t. the projection output (WrtProjected). *)
    | Param p ->
      (match target with
       | WrtParam param -> Known (if p = param then Const 1.0 else Const 0.0)
       | WrtPop _ | WrtProjected -> Known (Const 0.0))

    (* Forcing. Cases, in order:
       - lag guard (any kind): if [param] drives the forcing's evaluation-time
         shift ([lag]), the derivative with respect to the lag is not
         written. Omitted — a live coefficient with an un-emitted gradient.
         Checked first so a param that also drives a coefficient does not slip
         through to a closed form that ignores the lag.
       - Sinusoidal/Fourier: differentiate through the closed form (real grad).
       - Periodic: period + step values are live scalar coefficients (the Rust
         runtime evaluates them per-step via `resolve_coeff`). When [param]
         actually drives one, the gradient is not emitted: Omitted
         so the model compiles — forward sim and gradient-free IF2/PF use the
         live value, and the runtime's gradient gate refuses a gradient-based
         fit that depends on it. Not a hard error: that would also break
         forward sim and IF2/PF. When [param] does not drive the coefficient,
         the derivative is a genuine zero, exactly as for any constant — the
         common case, e.g. a param multiplying a constant-valued periodic term.
       - PeriodicSpline/BSpline:
         f = h(g), g = Σ_j c_j B_j, so ∂f/∂θ = h′ · Σ_j (∂c_j/∂θ) B_j. Each
         coefficient expression is differentiated by the ordinary rules (a
         corner constraint or a sum-to-zero gets the chain rule), each basis
         function is a [ForcingBasis] node the runtime evaluates with the
         forcing's own span code, and under `link = log` the sum is multiplied
         by the forcing's value (h′ = h). A parameter in a knot, window end or
         period moves the basis itself: Unsupported (E600).
       - Piecewise/Interpolated: a parameter there drives structural data
         (interpolation knots and values, a piecewise step grid) — fixed when
         the model is built, so it cannot be a live coefficient at all. Hard
         compile error (the Rust runtime also rejects it at IR-load via
         `eval_structural`). *)
    | TimeFunc fname ->
      (match target with
       (* forcings are state-free (∂forcing/∂x = 0) and projection-free
          (∂forcing/∂projected = 0). *)
       | WrtPop _ | WrtProjected -> Known (Const 0.0)
       | WrtParam param ->
      (match find_tf tfs fname with
       | Some tf when lag_mentions param tf ->
         Omitted
           { code = URLag;
             node = Printf.sprintf "forcing `%s`" fname;
             reason = Printf.sprintf
               "parameter '%s' drives the evaluation-time shift (`lag`) of \
                forcing `%s`: the forcing is evaluated at t − lag, but the \
                derivative w.r.t. the lag is not emitted (gh#314). Forward \
                simulation and gradient-free IF2/PF use the live value; a NUTS \
                fit that depends on this gradient is refused" param fname }
       | Some ({ kind = Sinusoidal s; _ } as tf) -> d (sinusoidal_closed ?lag:tf.lag s)
       | Some ({ kind = Fourier f; _ } as tf) -> d (fourier_closed ?lag:tf.lag f)
       | Some { kind = Periodic _; _ } ->
         if forcing_mentions param fname then
           Omitted
             { code = URPeriodicCoeff;
               node = Printf.sprintf "forcing `%s`" fname;
               reason = Printf.sprintf
                 "parameter '%s' drives a periodic forcing coefficient (step \
                  value or period) of `%s`: a live coefficient the Rust runtime \
                  evaluates per step, but whose gradient is not emitted \
                  (gh#215). Forward simulation and gradient-free IF2/PF use the \
                  live value; a NUTS fit that depends on this gradient is \
                  refused" param fname }
         else Known (Const 0.0)
       | Some { kind = PeriodicSpline ps; _ } ->
         spline_deriv param fname "period" [ ps.period ] ps.coefs ps.link
       | Some { kind = BSpline b; _ } ->
         let structural = match b.knots with
           | Knots_uniform { lower; upper; _ } -> [ lower; upper ]
           | Knots_explicit bps -> bps
         in
         (match b.coefs with
          | Coefs_explicit coefs ->
            spline_deriv param fname "window or breakpoints" structural coefs b.link
          | Coefs_smooth { level; slope; scale; modes; per } ->
            smooth_deriv param fname structural ~level ~slope ~scale ~modes ~per b.link)
       | Some { kind = (Piecewise _ | Interpolated _) as kind; _ } ->
         if forcing_mentions param fname then
           Unsupported
             { code = URStructuralForcing;
               node = Printf.sprintf "forcing `%s`" fname;
               reason = Printf.sprintf
                 "parameter '%s' drives the knots or values of %s forcing \
                  `%s`, which are structural data — interpolation knots and \
                  values and piecewise step grids are fixed when the model is \
                  built, so they cannot be an estimated parameter. Make them \
                  constants, or use a forcing whose coefficients are live \
                  (sinusoidal, fourier, periodic, periodic_spline, bspline)"
                 param (kind_label kind) fname }
         else Known (Const 0.0)
       | None -> Known (Const 0.0)))

    (* ∂B_k/∂θ: a basis function depends on t alone, so it is zero for every
       target — except a parameter in the forcing's `lag`, which moves the
       evaluation time and whose derivative is not emitted. Gradients
       are not differentiated again, so this arm only keeps the rule total. *)
    | ForcingBasis { forcing; _ } ->
      (match target with
       | WrtPop _ | WrtProjected -> Known (Const 0.0)
       | WrtParam param ->
         (match find_tf tfs forcing with
          | Some tf when lag_mentions param tf ->
            Omitted
              { code = URLag;
                node = Printf.sprintf "forcing `%s`" forcing;
                reason = Printf.sprintf
                  "parameter '%s' drives the evaluation-time shift (`lag`) of \
                   forcing `%s`, whose derivative is not emitted (gh#314)"
                  param forcing }
          | _ -> Known (Const 0.0)))

    (* Table lookup. A constant index selects one cell, so we differentiate the
       selected value expression — this is how per-stratum parameter tables
       (e.g. `sigma_stage[e1] = sigma_e1`) get a real gradient; the all-Const
       fold leaves such param tables as lookups. A non-constant (state- or
       param-dependent) index selects a cell we cannot identify symbolically →
       Unsupported when the param is in the index, Omitted when it is in a
       cell, else a genuine zero. *)
    | TableLookup (name, [ Const fi ]) ->
      (match find_table tbls name with
       | Some { source = Inline vals; _ } ->
         let i = int_of_float (Float.floor fi) in
         (match List.nth_opt vals i with
          | Some v -> d v
          | None -> Known (Const 0.0))  (* OOB — surfaced by validate/runtime *)
       | _ -> Known (Const 0.0))  (* external table: values are runtime data *)
    | TableLookup (name, args) ->
      (match target with
       | WrtParam param ->
      if List.exists (mentions param) args then
        (* The parameter is in a non-constant lookup index — it selects which
           cell, so the lookup is undifferentiable. Reject at compile time. *)
        Unsupported
          { code = URNonConstTableIndex;
            node = Printf.sprintf "table `%s`" name;
            reason = Printf.sprintf
              "parameter '%s' is used as a non-constant lookup index into table \
               `%s`; the lookup selects a cell by its value, so it is not \
               differentiable. Index the table by a constant or a compartment, \
               not by an estimated parameter" param name }
      else if table_value_mentions param name then
        (* The parameter is an inline-table value selected by a non-constant
           index. The value is a live coefficient (the Rust runtime resolves it),
           but the gradient through a runtime-chosen cell is not emitted.
           Omitted so the model compiles — IF2/PF use the live value, and the
           gradient gate refuses a gradient-based fit that depends on it. *)
        Omitted
          { code = URNonConstTableIndex;
            node = Printf.sprintf "table `%s`" name;
            reason = Printf.sprintf
              "parameter '%s' is an inline-table value in `%s` selected by a \
               non-constant index: a live coefficient the Rust runtime \
               resolves, but the gradient through a runtime-chosen cell is not \
               emitted (gh#215). Forward simulation and gradient-free IF2/PF \
               use the live value; a NUTS fit that depends on this gradient is \
               refused" param name }
      else Known (Const 0.0)
       | WrtPop c ->
        (* WrtPop: a non-constant index that reads compartment state is a
           discrete cell-selection by the count — undifferentiable w.r.t. state.
           Otherwise the table is state-free at this use (its values are
           const/param coefficients), so ∂/∂compartment = 0. *)
        if List.exists (mentions_pop bindings c) args then
          Unsupported
            { code = URNonConstTableIndex;
              node = Printf.sprintf "table `%s`" name;
              reason = Printf.sprintf
                "table `%s` is indexed by compartment state — a discrete \
                 cell-selection whose derivative w.r.t. the state is not defined, \
                 so a gradient method cannot use it" name }
        else Known (Const 0.0)
       | WrtProjected ->
        (* An index that reads the projection output is a discrete cell-selection
           by the projected value — undifferentiable; otherwise ∂/∂projected = 0. *)
        if List.exists mentions_projected args then
          Unsupported
            { code = URNonConstTableIndex;
              node = Printf.sprintf "table `%s`" name;
              reason = Printf.sprintf
                "table `%s` is indexed by the projection output — a discrete \
                 cell-selection whose derivative w.r.t. the projected value is not \
                 defined, so a gradient method cannot use it" name }
        else Known (Const 0.0))

    (* Binary operations — standard calculus rules; Unsupported propagates. *)
    | BinOp b -> begin match b.op with
      | Add -> map2 (d b.left) (d b.right)
                 (fun l r -> BinOp { op = Add; left = l; right = r })
      | Sub -> map2 (d b.left) (d b.right)
                 (fun l r -> BinOp { op = Sub; left = l; right = r })
      (* Product rule: d(fg) = f'g + fg' *)
      | Mul -> map2 (d b.left) (d b.right)
                 (fun dl dr -> BinOp { op = Add;
                    left  = BinOp { op = Mul; left = dl; right = b.right };
                    right = BinOp { op = Mul; left = b.left; right = dr } })
      (* Quotient rule: d(f/g) = (f'g - fg') / g² *)
      | Div -> map2 (d b.left) (d b.right)
                 (fun dl dr -> BinOp { op = Div;
                    left = BinOp { op = Sub;
                      left  = BinOp { op = Mul; left = dl; right = b.right };
                      right = BinOp { op = Mul; left = b.left; right = dr } };
                    right = BinOp { op = Mul; left = b.right; right = b.right } })
      (* Power rule, written to stay finite at a zero base:
           d(f^g) = g·f^(g−1)·f′ + g′·[f > 0 ? f^g·ln f : 0]
         The textbook form f^g·(g′ ln f + g f′/f) is equal for f > 0 but
         evaluates ln 0 and 1/0 at f = 0 — NaN at runtime — where the true
         derivative is finite (∂(I²)/∂I = 0, ∂(I^α)/∂α → 0). The guard gives the
         one-sided limit f^g·ln f → 0 (g > 0); the runtime evaluates only the
         taken branch, so ln f is never reached at f ≤ 0. A constant exponent
         has g′ = 0, and simplify reduces the rule to g·f^(g−1)·f′. For
         0 < g < 1, f^(g−1) is infinite at f = 0 and stays non-finite: that
         derivative is genuinely singular. *)
      | Pow -> map2 (d b.left) (d b.right)
                 (fun df dg ->
                    let f = b.left and g = b.right in
                    let f_pow_g_ln_f =
                      Cond { pred = BinOp { op = Gt; left = f; right = Const 0.0 };
                             then_ = BinOp { op = Mul;
                                             left = BinOp { op = Pow; left = f; right = g };
                                             right = UnOp { op = Log; arg = f } };
                             else_ = Const 0.0 } in
                    BinOp { op = Add;
                      left = BinOp { op = Mul;
                        left = BinOp { op = Mul; left = g;
                          right = BinOp { op = Pow; left = f;
                            right = BinOp { op = Sub; left = g; right = Const 1.0 } } };
                        right = df };
                      right = BinOp { op = Mul; left = dg; right = f_pow_g_ln_f } })
      (* Min/Max: subgradient — differentiate the active branch (WrtParam). For
         WrtPop, a min/max of state is nonsmooth at the crossover — refuse;
         state-free min/max has ∂/∂compartment = 0. *)
      | Min ->
        (match target with
         | WrtParam _ ->
           map2 (d b.left) (d b.right)
             (fun dl dr -> Cond { pred = BinOp { op = Lt; left = b.left; right = b.right };
                                  then_ = dl; else_ = dr })
         | WrtPop _ ->
           if hits_state b.left || hits_state b.right then
             Unsupported
               { code = URNonsmoothState; node = "min expression";
                 reason = "derivative of `min` w.r.t. compartment state is not \
                           smooth (a kink at the crossover); a gradient method \
                           cannot use it" }
           else Known (Const 0.0)
         | WrtProjected ->
           if mentions_projected b.left || mentions_projected b.right then
             Unsupported
               { code = URNonsmoothState; node = "min expression";
                 reason = "derivative of `min` w.r.t. the projection output is not \
                           smooth (a kink at the crossover); a gradient method \
                           cannot use it" }
           else Known (Const 0.0))
      | Max ->
        (match target with
         | WrtParam _ ->
           map2 (d b.left) (d b.right)
             (fun dl dr -> Cond { pred = BinOp { op = Gt; left = b.left; right = b.right };
                                  then_ = dl; else_ = dr })
         | WrtPop _ ->
           if hits_state b.left || hits_state b.right then
             Unsupported
               { code = URNonsmoothState; node = "max expression";
                 reason = "derivative of `max` w.r.t. compartment state is not \
                           smooth (a kink at the crossover); a gradient method \
                           cannot use it" }
           else Known (Const 0.0)
         | WrtProjected ->
           if mentions_projected b.left || mentions_projected b.right then
             Unsupported
               { code = URNonsmoothState; node = "max expression";
                 reason = "derivative of `max` w.r.t. the projection output is not \
                           smooth (a kink at the crossover); a gradient method \
                           cannot use it" }
           else Known (Const 0.0))
      (* Mod: a genuine 0 when neither operand depends on the differentiation
         variable; otherwise Unsupported. *)
      | Mod ->
        (match target with
         | WrtParam param ->
           if mentions param b.left || mentions param b.right then
             Unsupported
               { code = URMod;
                 node = "mod expression";
                 reason = Printf.sprintf
                   "derivative of `mod` w.r.t. parameter '%s' is not representable \
                    in the IR grammar (floor is needed); replace mod with a \
                    conditional guard" param }
           else Known (Const 0.0)
         | WrtPop _ ->
           if hits_state b.left || hits_state b.right then
             Unsupported
               { code = URMod; node = "mod expression";
                 reason = "derivative of `mod` w.r.t. compartment state is not \
                           representable (floor is needed) and is nonsmooth at the \
                           wraps; a gradient method cannot use it" }
           else Known (Const 0.0)
         | WrtProjected ->
           if mentions_projected b.left || mentions_projected b.right then
             Unsupported
               { code = URMod; node = "mod expression";
                 reason = "derivative of `mod` w.r.t. the projection output is not \
                           representable (floor is needed) and is nonsmooth at the \
                           wraps; a gradient method cannot use it" }
           else Known (Const 0.0))
      (* Comparison ops: piecewise constant, derivative is 0. *)
      | Eq | Neq | Lt | Gt | Le | Ge -> Known (Const 0.0)
      end

    (* Unary operations — chain rule. *)
    | UnOp u -> begin match u.op with
      | Exp -> map1 (d u.arg)
                 (fun da -> BinOp { op = Mul; left = UnOp { op = Exp; arg = u.arg }; right = da })
      | Log -> map1 (d u.arg)
                 (fun da -> BinOp { op = Div; left = da; right = u.arg })
      | Sqrt -> map1 (d u.arg)
                  (fun da -> BinOp { op = Div; left = da;
                     right = BinOp { op = Mul; left = Const 2.0;
                                     right = UnOp { op = Sqrt; arg = u.arg } } })
      | Neg -> map1 (d u.arg) (fun da -> UnOp { op = Neg; arg = da })
      (* d|f| = f' · sign(f), sign(0) := 0. For WrtPop, |state| is nonsmooth
         at 0 — refuse. *)
      | Abs ->
        (match target with
         | WrtParam _ ->
           let sign =
             Cond { pred = BinOp { op = Gt; left = u.arg; right = Const 0.0 };
                    then_ = Const 1.0;
                    else_ = Cond { pred = BinOp { op = Lt; left = u.arg; right = Const 0.0 };
                                   then_ = Const (-1.0); else_ = Const 0.0 } }
           in
           map1 (d u.arg) (fun da -> BinOp { op = Mul; left = da; right = sign })
         | WrtPop _ ->
           if hits_state u.arg then
             Unsupported
               { code = URNonsmoothState; node = "abs expression";
                 reason = "derivative of `abs` w.r.t. compartment state is not \
                           smooth (a kink at 0); a gradient method cannot use it" }
           else Known (Const 0.0)
         | WrtProjected ->
           if mentions_projected u.arg then
             Unsupported
               { code = URNonsmoothState; node = "abs expression";
                 reason = "derivative of `abs` w.r.t. the projection output is not \
                           smooth (a kink at 0); a gradient method cannot use it" }
           else Known (Const 0.0))
      (* Floor/Ceil: derivative 0 a.e. w.r.t. a parameter; but w.r.t. state it is
         a step function (nonsmooth at each integer) — refuse for WrtPop. *)
      | Floor | Ceil ->
        (match target with
         | WrtParam _ -> Known (Const 0.0)
         | WrtPop _ ->
           if hits_state u.arg then
             Unsupported
               { code = URNonsmoothState; node = "floor/ceil expression";
                 reason = "derivative of `floor`/`ceil` w.r.t. compartment state \
                           is not smooth (a step at each integer); a gradient \
                           method cannot use it" }
           else Known (Const 0.0)
         | WrtProjected ->
           if mentions_projected u.arg then
             Unsupported
               { code = URNonsmoothState; node = "floor/ceil expression";
                 reason = "derivative of `floor`/`ceil` w.r.t. the projection \
                           output is not smooth (a step at each integer); a \
                           gradient method cannot use it" }
           else Known (Const 0.0))
      | Sin -> map1 (d u.arg)
                 (fun da -> BinOp { op = Mul; left = UnOp { op = Cos; arg = u.arg }; right = da })
      | Cos -> map1 (d u.arg)
                 (fun da -> BinOp { op = Mul;
                    left = UnOp { op = Neg; arg = UnOp { op = Sin; arg = u.arg } }; right = da })
      | Tanh ->
        let t = UnOp { op = Tanh; arg = u.arg } in
        map1 (d u.arg)
          (fun da -> BinOp { op = Mul;
             left = BinOp { op = Sub; left = Const 1.0;
                            right = BinOp { op = Mul; left = t; right = t } };
             right = da })
      end

    (* Conditional: differentiate both branches, leave the predicate alone. *)
    | Cond c -> map2 (d c.then_) (d c.else_)
                  (fun dt de -> Cond { pred = c.pred; then_ = dt; else_ = de })

    (* Sum is linear: d/dp (Σ tᵢ) = Σ d/dp tᵢ. Precedence (map2): an Unsupported
       term short-circuits immediately (it dominates); the first Omitted term is
       remembered and returned only once the scan proves no Unsupported term
       follows; otherwise every term is Known and we rebuild the sum. *)
    | Reduce terms ->
      let rec collect acc omitted = function
        | [] ->
          (match omitted with
           | Some o -> o
           | None -> Known (Reduce (List.rev acc)))
        | t :: rest ->
          (match d t with
           | Known e -> collect (e :: acc) omitted rest
           | Omitted _ as o ->
             collect acc (match omitted with None -> Some o | some -> some) rest
           | Unsupported _ as u -> u)
      in
      collect [] None terms

    (* Hoisted bindings are param-free (state-only), so ∂/∂param = 0 (WrtParam).
       For WrtPop the premise inverts — a binding body is a function of state (a
       hoisted force-of-infection is exactly where the coupling lives), so we
       resolve the reference and recurse into its body. Bindings are acyclic by
       construction (topo-ordered; enforced at validate/dimcheck), so the
       recursion terminates; an unresolved name (never emitted for a valid model)
       is a genuine zero. *)
    | BindingRef name ->
      (match target with
       (* Bindings are dynamics surfaces: param-free under WrtParam, and never
          carry the observation-only [Projected] node under WrtProjected. *)
       | WrtParam _ | WrtProjected -> Known (Const 0.0)
       | WrtPop _ ->
         (match find_binding bindings name with
          | Some b -> d b.bexpr
          | None -> Known (Const 0.0)))
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  in
  d top


(** Algebraic simplification: constant folding and identity elimination.
    Reduces expression size after differentiation (product rule creates many
    multiply-by-zero and add-zero terms). Applied to fixed point — repeated
    until the expression stops changing.

    Arithmetic on literals folds only to a finite value. [(0 - 8) ^ 0.5],
    [exp(1000)] or an overflowing product are left as operations: JSON has
    no NaN or infinity, so a folded non-finite literal would make the IR
    unreadable, and the evaluator, not the compiler, decides what an
    operation that is reached does with a degenerate value. The compiler
    reports a non-finite literal sub-expression in a model expression as
    E520 ({!Validate}); a gradient field can still hold one, as the
    derivative of an expression that does not. *)
let rec simplify (e : expr) : expr =
  (* [Const v] when [v] is finite, otherwise the operation as it stands. *)
  let finite v unfolded = if Float.is_finite v then Const v else unfolded in
  let e = match e with
    (* Recurse first, then simplify *)
    | BinOp b ->
      let l = simplify b.left in
      let r = simplify b.right in
      begin match b.op, l, r with
      (* 0 + x = x, x + 0 = x *)
      | Add, Const 0.0, x | Add, x, Const 0.0 -> x
      (* x - 0 = x *)
      | Sub, x, Const 0.0 -> x
      (* 0 - x = -x *)
      | Sub, Const 0.0, x -> UnOp { op = Neg; arg = x }
      (* 0 * x = 0, x * 0 = 0 *)
      | Mul, Const 0.0, _ | Mul, _, Const 0.0 -> Const 0.0
      (* 1 * x = x, x * 1 = x *)
      | Mul, Const 1.0, x | Mul, x, Const 1.0 -> x
      (* 0 / x = 0 *)
      | Div, Const 0.0, _ -> Const 0.0
      (* x / 1 = x *)
      | Div, x, Const 1.0 -> x
      (* x ^ 0 = 1, x ^ 1 = x *)
      | Pow, _, Const 0.0 -> Const 1.0
      | Pow, x, Const 1.0 -> x
      (* Constant folding *)
      | ((Add | Sub | Mul | Div | Pow) as op), Const x, Const y ->
        let unfolded = BinOp { op; left = l; right = r } in
        (match op with
         | Add -> finite (x +. y) unfolded
         | Sub -> finite (x -. y) unfolded
         | Mul -> finite (x *. y) unfolded
         | Div -> if y = 0.0 then unfolded else finite (x /. y) unfolded
         | _   -> finite (x ** y) unfolded)
      | _ -> BinOp { op = b.op; left = l; right = r }
      end

    | UnOp u ->
      let a = simplify u.arg in
      let unfolded = UnOp { op = u.op; arg = a } in
      begin match u.op, a with
      | Neg, Const 0.0 -> Const 0.0
      | Neg, Const c   -> finite (-.c) unfolded
      | Exp, Const c   -> finite (exp c) unfolded
      | Log, Const c when c > 0.0 -> finite (log c) unfolded
      | Sqrt, Const c when c >= 0.0 -> finite (sqrt c) unfolded
      | Abs, Const c   -> finite (abs_float c) unfolded
      | Sin, Const c   -> finite (sin c) unfolded
      | Cos, Const c   -> finite (cos c) unfolded
      | Tanh, Const c  -> finite (tanh c) unfolded
      | _ -> unfolded
      end

    (* [Reduce] is an n-ary sum, so [Const 0.0] is its additive identity and a
       zero summand drops exactly as under `Add, x, Const 0.0 -> x` above.

       This arm is load-bearing for dense mean-field coupling, not a size nicety.
       Differentiating `Σ_h N[h]` (a global denominator over hoisted, param-free
       bindings) yields [Reduce [Const 0.0; ...]] — one zero per term, not
       [Const 0.0]. Left unfolded, the quotient rule's `-f·g'` term never meets
       `Mul, _, Const 0.0`, so `Div, Const 0.0, _` cannot fire and the entire
       O(H) sum rides into every parameter's gradient, squared — including
       parameters absent from the rate. That is an O(H²) [rate_grad]. *)
    | Reduce terms ->
      begin match List.filter (fun t -> t <> Const 0.0) (List.map simplify terms) with
      | []  -> Const 0.0
      | [t] -> t
      | ts  -> Reduce ts
      end

    | Cond c ->
      let p = simplify c.pred in
      let t = simplify c.then_ in
      let e = simplify c.else_ in
      begin match p with
      | Const v -> if v > 0.0 then t else e  (* constant predicate *)
      | _ ->
        (* If both branches are equal, collapse *)
        if t = e then t
        else Cond { pred = p; then_ = t; else_ = e }
      end

    | _ -> e
  in
  e


(** Apply simplify to a fixed point — repeat until the expression stops changing.
    "Stops changing" is [compare], not [=]: an input can already hold a
    [Const nan], and under [=] a tree holding a NaN never equals itself, so
    the loop would not terminate. *)
let simplify_fixpoint (e : expr) : expr =
  let rec go e =
    let e' = simplify e in
    if compare e' e = 0 then e' else go e'
  in
  go e

(** The derivative of a transition's rate with respect to each parameter in
    [param_names], in that order: the transition's [rate_grad].

    The driver passes every parameter of the model, estimated or not. A
    derivative that simplifies to [Const 0.0] is left out, an [Omitted] one
    becomes [DEUnsupported], and the first [Unsupported] one makes the whole
    result [Error msg], which the caller reports as [E600] at the
    transition's declaration. *)
let differentiate_rate (rate : expr) (param_names : string list)
    (tfs : time_function list) (tbls : table list) :
    ((string * deriv_entry) list, string) result =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | p :: rest ->
      (match differentiate rate (WrtParam p) tfs tbls with
       | Unsupported u -> Error (Printf.sprintf "%s: %s" u.node u.reason)
       | Omitted { node; code; _ } -> go ((p, DEUnsupported { node; code }) :: acc) rest
       | Known dexpr ->
         (match simplify_fixpoint dexpr with
          | Const 0.0 -> go acc rest
          | d' -> go ((p, DEGrad d') :: acc) rest))
  in
  go [] param_names

(** The derivative of a transition's rate with respect to each compartment
    in [compartments]: the transition's [rate_state_grad], from which the
    ODE forward-sensitivity gradient builds the Jacobian of the dynamics.

    Unlike {!differentiate_rate}, it never fails: both [Omitted] and
    [Unsupported] become [DEUnsupported], because a rate that is not smooth
    in the state ([floor], [abs] or [min] of a compartment, a table indexed
    by a compartment) must still simulate and fit by methods that use no
    gradient. A derivative that simplifies to [Const 0.0] is left out. *)
let differentiate_rate_state (rate : expr) (compartments : string list)
    (tfs : time_function list) (tbls : table list) (bindings : binding list)
    : (string * deriv_entry) list =
  let entry_of c =
    match differentiate ~bindings rate (WrtPop c) tfs tbls with
    | Known dexpr ->
      (match simplify_fixpoint dexpr with
       | Const 0.0 -> None                              (* genuine zero — drop *)
       | d' -> Some (c, DEGrad d'))
    | Omitted { node; code; _ } -> Some (c, DEUnsupported { node; code })
    | Unsupported { node; code; _ } -> Some (c, DEUnsupported { node; code })
  in
  List.filter_map entry_of compartments

(** The derivative of an observation's projection with respect to each
    compartment: the observation's [projection_state_grad]. The ODE gradient
    chains it with the state sensitivities, [∂proj/∂θ = Σ_j ∂proj/∂x_j ·
    ∂x_j/∂θ].

    Only a [DerivedExpr] projection, an arbitrary expression of the state,
    gets entries, computed and classified as by {!differentiate_rate_state}.
    The other projections select or sum compartments or accumulated flows,
    which the runtime differentiates itself, so they give an empty list. *)
let differentiate_projection (proj : projection) (compartments : string list)
    (tfs : time_function list) (tbls : table list) (bindings : binding list)
    : (string * deriv_entry) list =
  match proj with
  | DerivedExpr e -> differentiate_rate_state e compartments tfs tbls bindings
  (* A flow ratio depends on θ only through its two accumulated bins, exactly
     as a flow sum does; the quotient rule is applied over those bins at
     scoring time, so there is no state gradient to emit here either. *)
  | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _
  | FlowRatio _ -> []


(* ── Observation, overdispersion and initial-state drivers ────────────────────

   These never fail: [Omitted] and [Unsupported] both become [DEUnsupported]
   ([obs_deriv_entry]), and a derivative that simplifies to zero is left out
   of the map. *)

(** Substitute every [Projected] node in [e] with [proj]. Used to inline a
    [DerivedExpr] projection into a likelihood argument before
    differentiation, so ∂arg/∂θ picks up the ∂(projected)/∂θ chain-rule
    term. *)
let rec inline_projected (proj : expr) (e : expr) : expr =
  match e with
  | Projected -> proj
  | Const _ | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _
  | BindingRef _ | ObsColumnRef _ | ObsAnchor _ -> e
  | UnOp u -> UnOp { u with arg = inline_projected proj u.arg }
  | BinOp b ->
    BinOp { b with left = inline_projected proj b.left;
                   right = inline_projected proj b.right }
  | Cond c ->
    Cond { pred = inline_projected proj c.pred;
           then_ = inline_projected proj c.then_;
           else_ = inline_projected proj c.else_ }
  | TableLookup (name, args) -> TableLookup (name, List.map (inline_projected proj) args)
  | UncheckedDim u -> UncheckedDim { u with inner = inline_projected proj u.inner }
  | UnitLiteral _ -> e
  | Reduce terms -> Reduce (List.map (inline_projected proj) terms)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** Inline a projection into a likelihood argument. Only a [DerivedExpr]
    projection makes [projected] a function of θ; substituting it lets the
    chain rule reach ∂(projected)/∂θ. For every other projection kind
    ([CumulativeFlow]/[CurrentPop]/…) [projected] is θ-independent given the
    fixed trajectory X, so [Projected] is left in place and differentiates to a
    genuine zero via the [Projected -> Known (Const 0.0)] arm. *)
let inline_projection (proj : projection) (arg : expr) : expr =
  match proj with
  | DerivedExpr e -> inline_projected e arg
  | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _
  | FlowRatio _ -> arg

(** Adapt a differentiation outcome for an obs/σ² argument into a
    [deriv_entry option]. [None] means "omit the key" — a genuine zero. *)
let obs_deriv_entry (d : deriv) : deriv_entry option =
  match d with
  | Known e ->
    (match simplify_fixpoint e with
     | Const 0.0 -> None                     (* genuine zero → absent key *)
     | e' -> Some (DEGrad e'))
  | Omitted { node; code; _ } -> Some (DEUnsupported { node; code })
  | Unsupported { node; code; _ } -> Some (DEUnsupported { node; code })

(** ∂arg/∂θ for one differentiable likelihood argument, over every parameter.
    The projection is inlined first (so a parametric [DerivedExpr] contributes
    its chain-rule term); genuine zeros are dropped. *)
let differentiate_obs_arg (proj : projection) (arg : expr)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : grad_map =
  let inlined = inline_projection proj arg in
  List.filter_map
    (fun p ->
      match obs_deriv_entry (differentiate inlined (WrtParam p) tfs tbls) with
      | Some de -> Some (p, de)
      | None -> None)
    param_names

(** The derivative of one compartment's initial value with respect to each
    parameter: that compartment's entry in the model's [ic_grad], the
    starting value of the ODE state sensitivities.

    The driver passes the initial value with every compartment it reads
    already replaced by that compartment's own initial value, so the
    derivative follows the order in which the runtime evaluates the
    initial state. Zero derivatives are left out, and a derivative that
    cannot be written becomes [DEUnsupported]. *)
let differentiate_ic (ic_expr : expr) (param_names : string list)
    (tfs : time_function list) (tbls : table list) : grad_map =
  List.filter_map
    (fun p ->
      match obs_deriv_entry (differentiate ic_expr (WrtParam p) tfs tbls) with
      | Some de -> Some (p, de)
      | None -> None)
    param_names

(** Fill the [grad] of every differentiable argument of an initial-state law
    (`I ~ poisson(rate = I0)`), so `∂/∂θ log p(x₀ | θ)` has a compiler-emitted
    per-parameter derivative to chain through.

    A compartment in an argument differentiates to 0, and that is the
    intended convention: the density is scored with the initial state x₀
    held fixed (the conditioning trajectory's initial state), so the only
    dependence on θ is through the law's arguments. The same convention
    makes the emitted gradient checkable by finite differences of
    [initial_state_logpdf] at fixed x₀.

    No projection is involved, so nothing is inlined and [proj_grad] is
    [None]. Each [diffable] is built field by field rather than by a
    functional update, so a new field is a compile error here until it is
    differentiated. *)
let differentiate_initial_conditions (ic : initial_conditions)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : initial_conditions =
  let d (arg : diffable) : diffable =
    { expr = arg.expr;
      grad = differentiate_ic arg.expr param_names tfs tbls;
      proj_grad = None }
  in
  List.map
    (fun (comp, spec) ->
      let spec' =
        match spec with
        | Deterministic _ -> spec
        | InitCount (InitPoisson l) -> InitCount (InitPoisson { rate = d l.rate })
        | InitCount (InitBinomial l) -> InitCount (InitBinomial { n = l.n; p = d l.p })
        | InitCount (InitNegBinomial l) ->
          InitCount (InitNegBinomial { mean = d l.mean; dispersion = d l.dispersion })
        | InitReal (InitNormal l) -> InitReal (InitNormal { mean = d l.mean; sd = d l.sd })
      in
      (comp, spec'))
    ic

(** Differentiate one likelihood argument: keep its [expr] (raw — the projection
    is inlined only inside the gradient, never stored) and fill its [grad]. *)
let differentiate_diffable (proj : projection) (d : diffable)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : diffable =
  { expr = d.expr;
    grad = differentiate_obs_arg proj d.expr param_names tfs tbls;
    (* ∂arg/∂projected: differentiate the argument w.r.t. the
       [Projected] leaf directly, with no projection inlining (that is [grad]'s job,
       for ∂arg/∂θ). A genuine [Const 0.0] (the argument does not read the
       projection output) collapses to [None]. *)
    proj_grad = obs_deriv_entry (differentiate d.expr WrtProjected tfs tbls) }

(** Fill every differentiable argument of a likelihood. The likelihood is
    rebuilt field by field rather than by a functional update, and the match
    names every [likelihood] constructor, so a new argument or a new
    likelihood is a compile error here until it is differentiated.

    The [n] of a binomial or beta-binomial is not a [diffable] and gets no
    gradient: it is rounded to an integer and must not depend on an
    estimated parameter. The runtime's gradient gate, not this module,
    refuses a gradient-based fit in which it does. *)
let differentiate_likelihood (proj : projection) (lik : likelihood)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : likelihood =
  let d arg = differentiate_diffable proj arg param_names tfs tbls in
  match lik with
  | Poisson pl -> Poisson { rate = d pl.rate }
  | NegBinomial nb -> NegBinomial { mean = d nb.mean; dispersion = d nb.dispersion }
  | Normal n -> Normal { mean = d n.mean; sd = d n.sd }
  | Binomial b -> Binomial { n = b.n; p = d b.p }
  | BetaBinomial bb -> BetaBinomial { n = bb.n; alpha = d bb.alpha; beta = d bb.beta }
  | Beta b -> Beta { mean = d b.mean; concentration = d b.concentration }
  | Bernoulli b -> Bernoulli { p = d b.p }
  | ZeroInflatedNegBinomial zi ->
    ZeroInflatedNegBinomial
      { mean = d zi.mean; dispersion = d zi.dispersion; pi = d zi.pi }

(** Differentiate every observation stream's likelihood arguments w.r.t. all
    parameters (the fit reads only the estimated ones; proven zeros are absent).
    Genuine [Const 0.0] gradients are dropped; a live-but-omitted or structural
    coefficient becomes a coded [DEUnsupported] the fit-time gate refuses on. *)
let differentiate_observations (obs : observation_model list)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : observation_model list =
  List.map
    (fun (o : observation_model) ->
      { o with likelihood = differentiate_likelihood o.projection o.likelihood
                              param_names tfs tbls })
    obs

(** Differentiate the σ² expression of every [DrawOverdispersed] transition
    w.r.t. all parameters, filling [sigma_sq_grad]. σ² carries no [Projected]
    node (it is a rate-context overdispersion variance, not an observation), so
    no projection inlining is needed; the adapter is shared with obs. *)
let differentiate_overdispersion (transitions : transition list)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : transition list =
  List.map
    (fun (t : transition) ->
      match t.draw_method with
      | DrawOverdispersed { sigma_sq; _ } ->
        let sigma_sq_grad =
          List.filter_map
            (fun p ->
              match obs_deriv_entry (differentiate sigma_sq (WrtParam p) tfs tbls) with
              | Some de -> Some (p, de)
              | None -> None)
            param_names
        in
        { t with draw_method = DrawOverdispersed { sigma_sq; sigma_sq_grad } }
      | DrawPoisson | DrawDeterministic -> t)
    transitions
