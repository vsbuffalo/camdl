(** Source-to-source symbolic differentiation of camdl expressions.

    Differentiates an [expr] with respect to a named parameter, producing
    a new [expr] representing ∂expr/∂param. The result is a plain expression
    tree that can be evaluated by the existing [eval_expr] in the Rust backend.

    Forcing/table coefficients are expressions over parameters, not data
    (gh#119): a parameter that enters a transition rate only through a forcing
    or a table entry still has a real derivative. [differentiate] looks the
    forcing/table definition up (hence the [time_function]/[table] lists) and
    emits the analytic ∂forcing/∂coef for the kinds it supports (Sinusoidal,
    Fourier, constant-indexed inline tables).

    Three differentiation outcomes, handled differently (gh#215, gh#314):
    - KNOWN — a real derivative expression (including a genuine [Const 0.0] when
      the parameter does not drive the node).
    - OMITTED — a LIVE coefficient the gradient just doesn't cover yet: a
      Periodic step value/period, an inline-table value reached by a
      non-constant index, or a forcing's evaluation-time shift ([lag], gh#314).
      The Rust runtime evaluates these live, so the model must compile (forward
      sim and gradient-free IF2/PF work). [differentiate_rate] drops the
      parameter — byte-identical to a genuine zero — and the Rust NUTS guard
      (coeff_guard.rs) refuses a NUTS fit that depends on the missing gradient.
      (The obs/σ² driver, a later phase, instead refuses with the carried
      [reason], so a live-but-omitted coefficient never masquerades as a genuine
      zero on the observation path.)
    - STRUCTURAL data a parameter cannot drive at all — interpolation knots, a
      piecewise step grid, the spline basis, or a non-constant lookup index.
      These return [Unsupported], which [differentiate_rate] turns into a
      compile-time error (the Rust runtime also rejects them at IR-load).

    Compartment counts (Pop/PopSum), Time, Dt and Projected are constants in
    the PGAS θ|X step (the trajectory X is fixed), so their derivative is 0. *)

open Ir

(** A differentiation result. Three outcomes, distinguished so a live-but-
    omitted coefficient never masquerades as a genuine zero (proposal
    `2026-07-03-unified-obs-gradient-autodiff.md` §4.2; note
    `2026-06-08-static-typing-as-bug-prevention.md` §7):
    - [Known e]      — a real derivative [e] (including a genuine [Const 0.0]).
    - [Omitted]      — a live coefficient whose derivative is not emitted (tier
                       2b: Periodic step/period, inline-table value via a
                       non-constant index, or a forcing's [lag]);
                       [differentiate_rate] drops it, exactly as it drops a
                       proven zero.
    - [Unsupported]  — structural data a parameter cannot drive (tier 3);
                       [differentiate_rate] raises a compile-time error.

    Both non-[Known] cases carry a stable [Ir.unsupported_reason] [code]
    alongside the human [node]/[reason]. The rate path uses [node]/[reason] for
    its E600 message; the obs/σ² driver (P3) uses [code] to build a
    [deriv_entry] [DEUnsupported] — the classification the differentiation site
    already made, carried to the driver rather than re-derived from the free-text
    [reason] (proposal §4.1: the code is canonical, the message is derived). *)
type deriv =
  | Known of expr
  | Omitted of { node : string; reason : string; code : unsupported_reason }
  | Unsupported of { node : string; reason : string; code : unsupported_reason }

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
  | Const _ | Pop _ | Time | Dt | Projected | ObsColumnRef _ -> false
  | PopSum _ -> false
  | BinOp b -> mentions param b.left || mentions param b.right
  | UnOp u -> mentions param u.arg
  | Cond c -> mentions param c.pred || mentions param c.then_ || mentions param c.else_
  | TimeFunc _ -> false
  | TableLookup (_, args) -> List.exists (mentions param) args
  | UncheckedDim u -> mentions param u.inner
  | Reduce terms -> List.exists (mentions param) terms
  | BindingRef _ -> false   (* hoisted bindings are param-free (state-only) *)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** Does [e] reference the projection output ([Projected])? The [WrtProjected]
    analogue of [mentions]: used to detect a nonsmooth function OF the projection
    ([floor]/[min]/… of [projected]) — undifferentiable w.r.t. the projected value.
    [Projected] only appears in likelihood argument expressions, so bindings /
    forcings / tables never carry it. *)
let rec mentions_projected (e : expr) : bool =
  match e with
  | Projected -> true
  | Param _ | Const _ | Pop _ | PopSum _ | Time | Dt | ObsColumnRef _ -> false
  | BinOp b -> mentions_projected b.left || mentions_projected b.right
  | UnOp u -> mentions_projected u.arg
  | Cond c -> mentions_projected c.pred || mentions_projected c.then_ || mentions_projected c.else_
  | TimeFunc _ -> false
  | TableLookup (_, args) -> List.exists mentions_projected args
  | UncheckedDim u -> mentions_projected u.inner
  | Reduce terms -> List.exists mentions_projected terms
  | BindingRef _ -> false
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(** What [differentiate] differentiates with respect to (gh#275). The engine is
    one recursion; only the leaves and the nonsmooth ops branch on the target:
    - [WrtParam p] — ∂/∂param (the [rate_grad] path): state ([Pop]/[PopSum]) is
      constant (fixed in the θ|X step), bindings are param-free (zero).
    - [WrtPop c]   — ∂/∂compartment (the new [rate_state_grad] / ODE-sensitivity
      path): parameters are constant, forcings are state-free, and bindings are
      state-bearing (the premise inverts — recurse into their bodies).
    - [WrtProjected] — ∂/∂projection-output (the observation FACTOR-2 chain,
      gh#275): differentiates a likelihood ARGUMENT (a [diffable]'s [expr]) w.r.t.
      the [Projected] leaf, so [∂arg/∂projected] can chain the obs score against
      [∂projected/∂θ]. [Projected] is the target; parameters, state, forcings, and
      bindings are all constant w.r.t. it. Only valid inside a likelihood
      argument (only there does [Projected] appear). *)
type diff_target =
  | WrtParam of string
  | WrtPop of string
  | WrtProjected

(** Does compartment [c]'s state reach [e] — directly ([Pop c] or a [PopSum]
    containing [c]) or through a hoisted binding's body? The [WrtPop] analogue of
    [mentions]: forcings and const-indexed table cells are state-free, so a
    non-mention proves ∂e/∂Pop(c) = 0, and a mention inside a nonsmooth op
    (Mod/Floor/Ceil/Abs/Min/Max) is the [WrtPop] refusal trigger. Bindings are
    acyclic by construction (topo-ordered; enforced at validate/dimcheck), so the
    binding recursion terminates. *)
let rec mentions_pop (bindings : binding list) (c : string) (e : expr) : bool =
  match e with
  | Pop n -> n = c
  | PopSum members -> List.mem c members
  | Param _ | Const _ | Time | Dt | Projected | ObsColumnRef _ -> false
  | TimeFunc _ -> false   (* forcings are state-free *)
  | BindingRef name ->
    (match List.find_opt (fun (b : binding) -> b.bname = name) bindings with
     | Some b -> mentions_pop bindings c b.bexpr
     | None -> false)
  | TableLookup (_, args) -> List.exists (mentions_pop bindings c) args
  | BinOp b -> mentions_pop bindings c b.left || mentions_pop bindings c b.right
  | UnOp u -> mentions_pop bindings c u.arg
  | Cond cc ->
    mentions_pop bindings c cc.pred
    || mentions_pop bindings c cc.then_
    || mentions_pop bindings c cc.else_
  | Reduce terms -> List.exists (mentions_pop bindings c) terms
  | UncheckedDim u -> mentions_pop bindings c u.inner
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

let kind_label (k : time_func_kind) : string =
  match k with
  | Sinusoidal _ -> "sinusoidal"
  | Piecewise _ -> "piecewise"
  | Interpolated _ -> "interpolated"
  | Periodic _ -> "periodic"
  | Fourier _ -> "fourier"
  | PeriodicSpline _ -> "periodic_spline"

let two_pi = 2.0 *. Float.pi

(** Closed form of a sinusoidal forcing:
    [baseline + amplitude · sin(2π(t − phase)/period)] — matching the Rust
    evaluator (`sinusoidal_value`). Differentiating this w.r.t. a coefficient
    parameter yields the analytic ∂forcing/∂coef via the ordinary rules (the
    coefficient sub-expressions carry the parameter dependence). *)
let sinusoidal_closed ?lag (s : sinusoidal) : expr =
  (* Evaluation-time shift (gh#314): the runtime evaluates the forcing at
     t - lag, so the differentiated closed form must be built over (Time - lag),
     not bare Time — otherwise the emitted gradient is evaluated at t while the
     value is at t - lag (incident 2026-07-05). The lag_mentions guard in
     [differentiate] runs FIRST, so here the differentiation parameter is never
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
  (* Same evaluation-time shift as [sinusoidal_closed] (gh#314, incident
     2026-07-05): differentiate over (Time - lag), not bare Time. *)
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
    match List.find_opt (fun (tf : time_function) -> tf.name = fname) tfs with
    | Some tf -> List.exists (mentions param) (forcing_coeff_exprs tf.kind)
    | None -> false
  in
  let table_value_mentions param name =
    match List.find_opt (fun (t : table) -> t.name = name) tbls with
    | Some { source = Inline vals; _ } -> List.exists (mentions param) vals
    | _ -> false
  in
  (* Does [param] drive the forcing's evaluation-time shift ([lag], gh#314)?
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
  let rec d (e : expr) : deriv =
    match e with
    (* Constant leaves — zero for every target. *)
    | Const _ | Time | Dt | ObsColumnRef _ -> Known (Const 0.0)

    (* The projection output. The target leaf under [WrtProjected] (∂projected/∂projected
       = 1, the source of the observation factor-2 chain); constant w.r.t. a
       parameter or a compartment (it is an independent input to the likelihood). *)
    | Projected ->
      (match target with
       | WrtProjected -> Known (Const 1.0)
       | WrtParam _ | WrtPop _ -> Known (Const 0.0))

    (* Dimensional escape: differentiate the inner, drop the wrapper. *)
    | UncheckedDim u -> d u.inner

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
       | WrtPop c -> Known (Const (if List.mem c members then 1.0 else 0.0)))

    (* Parameter reference — 1 if it's the target param (WrtParam); constant
       w.r.t. state (WrtPop) and w.r.t. the projection output (WrtProjected). *)
    | Param p ->
      (match target with
       | WrtParam param -> Known (if p = param then Const 1.0 else Const 0.0)
       | WrtPop _ | WrtProjected -> Known (Const 0.0))

    (* Forcing. Cases, in order:
       - lag guard (any kind): if [param] drives the forcing's evaluation-time
         shift ([lag], gh#314), the closed forms below cannot express the
         derivative (they differentiate against bare [Time], not [Time − lag]).
         Omitted — a live coefficient with an un-emitted gradient. Checked
         FIRST so a param that also drives a coefficient does not slip through
         to a closed form that silently ignores the lag.
       - Sinusoidal/Fourier: differentiate through the closed form (real grad).
       - Periodic: period + step values are LIVE scalar coefficients (the Rust
         runtime evaluates them per-step via `resolve_coeff`). When [param]
         actually drives one, the gradient is not yet emitted (gh#215): Omitted
         so the model compiles — forward sim and gradient-free IF2/PF use the
         live value, and the Rust NUTS guard (coeff_guard.rs) refuses a NUTS fit
         that depends on it. NOT a hard error: that would also break forward sim
         and IF2/PF. When [param] does NOT drive the coefficient, the derivative
         is a genuine zero (tier 2a), exactly as for any constant — this is the
         common case, e.g. a param multiplying a constant-valued periodic term.
       - Piecewise/Interpolated/PeriodicSpline: a parameter there drives
         STRUCTURAL data (interpolation knots, a piecewise step grid, the
         de-Boor spline basis) — precomputed at construction, so it cannot be a
         live coefficient at all. Hard compile error (the Rust runtime also
         rejects it at IR-load via `eval_structural`). *)
    | TimeFunc fname ->
      (match target with
       (* forcings are state-free (∂forcing/∂x = 0) and projection-free
          (∂forcing/∂projected = 0). *)
       | WrtPop _ | WrtProjected -> Known (Const 0.0)
       | WrtParam param ->
      (match List.find_opt (fun (tf : time_function) -> tf.name = fname) tfs with
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
       | Some { kind = (Piecewise _ | Interpolated _ | PeriodicSpline _) as kind; _ } ->
         if forcing_mentions param fname then
           Unsupported
             { code = URStructuralForcing;
               node = Printf.sprintf "forcing `%s`" fname;
               reason = Printf.sprintf
                 "parameter '%s' drives the %s forcing coefficient, which is \
                  structural data — interpolation knots, piecewise step grids, \
                  and the spline basis are precomputed at construction and \
                  cannot vary per step, so they cannot be an estimated \
                  parameter. Make the coefficient a constant, or use a \
                  sinusoidal, fourier, or periodic forcing (whose coefficients \
                  are live)" param (kind_label kind) }
         else Known (Const 0.0)
       | None -> Known (Const 0.0)))

    (* Table lookup. A constant index selects one cell, so we differentiate the
       selected value expression — this is how per-stratum parameter tables
       (e.g. `sigma_stage[e1] = sigma_e1`) get a real gradient; the all-Const
       fold leaves such param tables as lookups. A non-constant (state- or
       param-dependent) index selects a cell we cannot identify symbolically →
       Unsupported when the param is involved (gh#215), else a genuine zero. *)
    | TableLookup (name, [ Const fi ]) ->
      (match List.find_opt (fun (t : table) -> t.name = name) tbls with
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
        (* The parameter is in a non-constant LOOKUP INDEX — it selects which
           cell, so the lookup is undifferentiable and the index is not a live
           coefficient the NUTS guard covers (it treats indices as body
           sub-expressions). Reject at compile time. *)
        Unsupported
          { code = URNonConstTableIndex;
            node = Printf.sprintf "table `%s`" name;
            reason = Printf.sprintf
              "parameter '%s' is used as a non-constant lookup index into table \
               `%s`; the lookup selects a cell by its value, so it is not \
               differentiable. Index the table by a constant or a compartment, \
               not by an estimated parameter" param name }
      else if table_value_mentions param name then
        (* The parameter is an inline-table VALUE selected by a non-constant
           index. The value is a live coefficient (the Rust runtime resolves it),
           but the gradient through a runtime-chosen cell is not yet emitted
           (gh#215). Omitted so the model compiles — IF2/PF use the live value,
           and the NUTS guard refuses a NUTS fit that depends on it. *)
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
      (* Power rule: d(f^g) = f^g * (g' ln f + g f'/f) *)
      | Pow -> map2 (d b.left) (d b.right)
                 (fun df dg -> BinOp { op = Mul;
                    left = BinOp { op = Pow; left = b.left; right = b.right };
                    right = BinOp { op = Add;
                      left  = BinOp { op = Mul; left = dg;
                                      right = UnOp { op = Log; arg = b.left } };
                      right = BinOp { op = Mul; left = b.right;
                                      right = BinOp { op = Div; left = df;
                                                      right = b.left } } } })
      (* Min/Max: subgradient — differentiate the active branch (WrtParam). For
         WrtPop, a min/max OF STATE is nonsmooth at the crossover — refuse (§1h);
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
      (* Mod: derivative needs floor, absent from the grammar. A genuine 0 when
         neither operand depends on the differentiation variable; otherwise
         Unsupported (was a failwith — M4 in the 2026-04-19 compiler review). *)
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
      (* d|f| = f' · sign(f), sign(0) := 0 (n1 in the 2026-04-19 review). For
         WrtPop, |state| is nonsmooth at 0 — refuse (§1h). *)
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
      (* Floor/Ceil: derivative 0 a.e. w.r.t. a parameter; but w.r.t. STATE it is
         a step function (nonsmooth at each integer) — refuse for WrtPop (§1h). *)
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
       For WrtPop the premise INVERTS — a binding body is a function of state (a
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
         (match List.find_opt (fun (b : binding) -> b.bname = name) bindings with
          | Some b -> d b.bexpr
          | None -> Known (Const 0.0)))
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  in
  d top


(** Algebraic simplification: constant folding and identity elimination.
    Reduces expression size after differentiation (product rule creates many
    multiply-by-zero and add-zero terms). Applied to fixed point — repeated
    until the expression stops changing. *)
let rec simplify (e : expr) : expr =
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
      | Add, Const a, Const b -> Const (a +. b)
      | Sub, Const a, Const b -> Const (a -. b)
      | Mul, Const a, Const b -> Const (a *. b)
      | Div, Const a, Const b when b <> 0.0 -> Const (a /. b)
      | Pow, Const a, Const b -> Const (a ** b)
      | _ -> BinOp { op = b.op; left = l; right = r }
      end

    | UnOp u ->
      let a = simplify u.arg in
      begin match u.op, a with
      | Neg, Const 0.0 -> Const 0.0
      | Neg, Const c   -> Const (-.c)
      | Exp, Const c   -> Const (exp c)
      | Log, Const c when c > 0.0 -> Const (log c)
      | Sqrt, Const c when c >= 0.0 -> Const (sqrt c)
      | Abs, Const c   -> Const (abs_float c)
      | Sin, Const c   -> Const (sin c)
      | Cos, Const c   -> Const (cos c)
      | Tanh, Const c  -> Const (tanh c)
      | _ -> UnOp { op = u.op; arg = a }
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


(** Apply simplify to a fixed point — repeat until the expression stops changing. *)
let simplify_fixpoint (e : expr) : expr =
  let rec go e =
    let e' = simplify e in
    if e' = e then e' else go e'
  in
  go e

(** Differentiate a rate expression w.r.t. each estimated parameter, producing a
    classified [grad_map].

    Returns [Ok grad_map] — [(param_name, deriv_entry)] pairs — or [Error msg] if
    any parameter's derivative is [Unsupported] (a structural coefficient a param
    cannot drive: spline/interp/piecewise knot; the caller turns it into a
    compile-time E600). The classification mirrors the obs driver, differing only
    in the [Unsupported] arm (rate → E600; obs → a serialized refusal):

    - [Known] folding to [Const 0.0] → dropped (absent key = genuine zero).
    - [Known d'] → [DEGrad d'].
    - [Omitted] (live-but-omitted: Periodic step/period, [lag], inline-table value
      via a non-constant index) → [DEUnsupported {node; code}] — was DROPPED
      pre-3b; now serialized so the fit-time preflight refuses on it, subsuming the
      old [coeff_guard] (gh#342). Forward sim / IF2 / PF still use the live value;
      only a gradient-based (NUTS) fit is refused.
    - [Unsupported] → [Error] → E600 (tier-3 is non-runnable regardless of method,
      so the earliest, source-located rejection is best; §3 of the 3b proposal). *)
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

(** ∂rate/∂compartment for each compartment name, producing the transition's
    [rate_state_grad] map — [J_x]'s ingredient for the ODE forward sensitivities
    (gh#275). The [WrtPop] sibling of [differentiate_rate].

    The compile-vs-defer POLICY differs from [differentiate_rate]: there an
    [Unsupported] coefficient is a compile-time E600 (tier-3 is non-runnable by
    any method). Here BOTH [Omitted] and [Unsupported] become a serialized
    [DEUnsupported] the fit-time gradient gate refuses on — a rate with a
    nonsmooth-of-state term (floor/ceil/abs/min/max of a compartment, a
    state-indexed table) must still forward-simulate and fit by the gradient-free
    IF2/PF; only a gradient method (ODE-NUTS) is refused. This mirrors the obs/σ²
    driver, not the rate-θ driver. An absent key is a genuine zero. *)
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

(** ∂projection/∂compartment for an observation projection — the model's
    [projection_state_grad], consumed by the ODE observation gradient's factor-2
    chain (`∂proj/∂θ = Σ_j ∂proj/∂x_j · S[j]`, gh#275 §1h). Only a [DerivedExpr]
    (a nonlinear function of state) has a non-trivial state gradient; it reuses
    [differentiate_rate_state] verbatim (∂expr/∂compartment via [WrtPop], same
    defer POLICY — a nonsmooth-of-state projection like `floor(I/N)` becomes a
    [DEUnsupported] the gradient gate refuses on). The linear projections
    ([CurrentPop*]/[CumulativeFlow*]) are trivial selections the factor-2 chain
    handles directly, so they emit nothing. *)
let differentiate_projection (proj : projection) (compartments : string list)
    (tfs : time_function list) (tbls : table list) (bindings : binding list)
    : (string * deriv_entry) list =
  match proj with
  | DerivedExpr e -> differentiate_rate_state e compartments tfs tbls bindings
  | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _ -> []


(* ── Observation / σ² gradient driver (proposal 2026-07-03, P3) ───────────────

   The obs and σ² densities need ∂arg/∂θ for each differentiable likelihood
   argument (mean, sd, rate, p, α, β) and the overdispersion σ² expression. This
   reuses the single differentiation authority [differentiate]; only the
   compile-vs-defer POLICY differs from the rate path (the "natural seam"):

   - rate  : [Omitted]/[Unsupported] → drop / E600 ([differentiate_rate]).
   - obs/σ²: [Omitted] AND [Unsupported] → a serialized, coded
     [DEUnsupported] the fit-time gate (P5) refuses NUTS on, so a live-but-
     omitted coefficient never masquerades as a genuine zero on this path.

   A genuine zero ([Known] folding to [Const 0.0]) is OMITTED from the map — an
   absent key is a genuine zero, mirroring [differentiate_rate] dropping proven
   zeros. *)

(** Substitute every [Projected] node in [e] with [proj] — a trivial rewrite
    ([Projected] is nullary). Used to inline a [DerivedExpr] projection into a
    likelihood argument BEFORE differentiation, so ∂arg/∂θ picks up the
    ∂(projected)/∂θ chain-rule term. *)
let rec inline_projected (proj : expr) (e : expr) : expr =
  match e with
  | Projected -> proj
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | BindingRef _ | ObsColumnRef _ -> e
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
  | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _ -> arg

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

(** ∂(initial-condition expression)/∂θ for every parameter — one compartment's
    entry in the model's [ic_grad] map (compartment → param → ∂init/∂param), the
    ODE forward-sensitivity seed S(t_start) (gh#275 §1c C-seed).

    Same compile-vs-defer POLICY as the obs/σ² driver ([obs_deriv_entry], the
    "natural seam"): a genuine zero is dropped (an absent key IS a genuine zero),
    and an [Omitted] or [Unsupported] coefficient becomes a serialized
    [DEUnsupported] the fit-time gradient gate refuses ODE-NUTS on — a nonsmooth
    initial condition still forward-simulates and fits by the gradient-free
    backends (IF2/PF), exactly as a nonsmooth rate does. No projection is involved
    (an IC is not an observation argument), so nothing is inlined. *)
let differentiate_ic (ic_expr : expr) (param_names : string list)
    (tfs : time_function list) (tbls : table list) : grad_map =
  List.filter_map
    (fun p ->
      match obs_deriv_entry (differentiate ic_expr (WrtParam p) tfs tbls) with
      | Some de -> Some (p, de)
      | None -> None)
    param_names

(** Differentiate one likelihood argument: keep its [expr] (raw — the projection
    is inlined only inside the gradient, never stored) and fill its [grad]. *)
let differentiate_diffable (proj : projection) (d : diffable)
    (param_names : string list) (tfs : time_function list) (tbls : table list)
    : diffable =
  { expr = d.expr;
    grad = differentiate_obs_arg proj d.expr param_names tfs tbls;
    (* ∂arg/∂projected (gh#275 factor 2): differentiate the argument w.r.t. the
       [Projected] leaf directly — NO projection inlining (that is [grad]'s job,
       for ∂arg/∂θ). A genuine [Const 0.0] (the argument does not read the
       projection output) collapses to [None]. *)
    proj_grad = obs_deriv_entry (differentiate d.expr WrtProjected tfs tbls) }

(** Fill every differentiable position of a likelihood by FULL RECONSTRUCTION
    (not a functional update): a new [diffable] field is a compile error here
    until it is differentiated — the OCaml half of the coverage seal (proposal
    2026-07-06 §4.3). The match is exhaustive over [likelihood], so a new variant
    is likewise a compile error.

    [n] (Binomial/BetaBinomial) is deliberately NOT a [diffable] — it is
    θ-independent (rounded to an integer) and carries no gradient; the refusal of
    an estimated param reaching [n] after inlining is the P5 fit-gate's job
    (proposal §4.4), nothing to emit here. *)
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
  | Bernoulli b -> Bernoulli { p = d b.p }

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
