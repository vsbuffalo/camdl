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

    Two distinct not-yet-supported cases, handled differently (gh#215):
    - LIVE coefficients the gradient just doesn't cover yet — a Periodic step
      value, or an inline-table value reached by a non-constant index. The Rust
      runtime evaluates these live, so the model must compile (forward sim and
      gradient-free IF2/PF work). [differentiate] omits the parameter (a
      [Known (Const 0.0)] that [differentiate_rate] drops); the Rust NUTS guard
      (coeff_guard.rs) refuses a NUTS fit that depends on the missing gradient.
    - STRUCTURAL data a parameter cannot drive at all — interpolation knots, a
      piecewise step grid, the spline basis, or a non-constant lookup index.
      These return [Unsupported], which [differentiate_rate] turns into a
      compile-time error (the Rust runtime also rejects them at IR-load).

    Compartment counts (Pop/PopSum), Time, Dt and Projected are constants in
    the PGAS θ|X step (the trajectory X is fixed), so their derivative is 0. *)

open Ir

(** A differentiation result: a known derivative expression, or an explicit
    "not differentiated" carrying a reason. Replaces the former silent
    [Const 0.0] for forcing/table nodes — a dropped derivative is now a value
    [differentiate_rate] is forced to handle (proposal
    `2026-06-09-const-parametric-forcing.md` §4; note
    `2026-06-08-static-typing-as-bug-prevention.md` §7). *)
type deriv =
  | Known of expr
  | Unsupported of { node : string; reason : string }

let map1 (d : deriv) (f : expr -> expr) : deriv =
  match d with Known e -> Known (f e) | Unsupported _ as u -> u

(* Combine two sub-derivatives under a calculus rule, propagating Unsupported. *)
let map2 (da : deriv) (db : deriv) (f : expr -> expr -> expr) : deriv =
  match da, db with
  | Known a, Known b -> Known (f a b)
  | (Unsupported _ as u), _ -> u
  | _, (Unsupported _ as u) -> u

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
let sinusoidal_closed (s : sinusoidal) : expr =
  let theta =
    BinOp { op = Div;
            left = BinOp { op = Mul; left = Const two_pi;
                           right = BinOp { op = Sub; left = Time; right = s.phase } };
            right = s.period }
  in
  BinOp { op = Add; left = s.baseline;
          right = BinOp { op = Mul; left = s.amplitude;
                          right = UnOp { op = Sin; arg = theta } } }

(** Closed form of a finite Fourier series:
    [Σ_k a_k cos(2π(k+1)t/period) + b_k sin(2π(k+1)t/period)] (k 0-based,
    harmonic k+1) — matching the Rust evaluator (`fourier_value`). No baseline;
    the model author writes `1 + fourier(t)`. *)
let fourier_closed (f : fourier) : expr =
  let term k (a, b) =
    let kf = float_of_int (k + 1) in
    let arg =
      BinOp { op = Div;
              left = BinOp { op = Mul; left = Const (two_pi *. kf); right = Time };
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
let differentiate (top : expr) (param : string)
    (tfs : time_function list) (tbls : table list) : deriv =
  let forcing_mentions fname =
    match List.find_opt (fun (tf : time_function) -> tf.name = fname) tfs with
    | Some tf -> List.exists (mentions param) (forcing_coeff_exprs tf.kind)
    | None -> false
  in
  let table_value_mentions name =
    match List.find_opt (fun (t : table) -> t.name = name) tbls with
    | Some { source = Inline vals; _ } -> List.exists (mentions param) vals
    | _ -> false
  in
  let rec d (e : expr) : deriv =
    match e with
    (* Constants in the θ|X step — derivative is zero. *)
    | Const _ | Pop _ | PopSum _ | Time | Dt | Projected | ObsColumnRef _ -> Known (Const 0.0)

    (* Dimensional escape: differentiate the inner, drop the wrapper. *)
    | UncheckedDim u -> d u.inner

    (* Parameter reference — 1 if it's the target, 0 otherwise. *)
    | Param p -> Known (if p = param then Const 1.0 else Const 0.0)

    (* Forcing. Three cases:
       - Sinusoidal/Fourier: differentiate through the closed form (real grad).
       - Periodic: period + step values are LIVE scalar coefficients (the Rust
         runtime evaluates them per-step via `resolve_coeff`), but the gradient
         is not yet emitted (gh#215). Omit it (Known (Const 0.0)) so the model
         compiles — forward sim and gradient-free IF2/PF use the live value, and
         the Rust NUTS guard (coeff_guard.rs) refuses a NUTS fit that depends on
         it. NOT a hard error: that would also break forward sim and IF2/PF.
       - Piecewise/Interpolated/PeriodicSpline: a parameter there drives
         STRUCTURAL data (interpolation knots, a piecewise step grid, the
         de-Boor spline basis) — precomputed at construction, so it cannot be a
         live coefficient at all. Hard compile error (the Rust runtime also
         rejects it at IR-load via `eval_structural`). *)
    | TimeFunc fname ->
      (match List.find_opt (fun (tf : time_function) -> tf.name = fname) tfs with
       | Some { kind = Sinusoidal s; _ } -> d (sinusoidal_closed s)
       | Some { kind = Fourier f; _ } -> d (fourier_closed f)
       | Some { kind = Periodic _; _ } -> Known (Const 0.0)
       | Some { kind = (Piecewise _ | Interpolated _ | PeriodicSpline _) as kind; _ } ->
         if forcing_mentions fname then
           Unsupported
             { node = Printf.sprintf "forcing `%s`" fname;
               reason = Printf.sprintf
                 "parameter '%s' drives the %s forcing coefficient, which is \
                  structural data — interpolation knots, piecewise step grids, \
                  and the spline basis are precomputed at construction and \
                  cannot vary per step, so they cannot be an estimated \
                  parameter. Make the coefficient a constant, or use a \
                  sinusoidal, fourier, or periodic forcing (whose coefficients \
                  are live)" param (kind_label kind) }
         else Known (Const 0.0)
       | None -> Known (Const 0.0))

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
      if List.exists (mentions param) args then
        (* The parameter is in a non-constant LOOKUP INDEX — it selects which
           cell, so the lookup is undifferentiable and the index is not a live
           coefficient the NUTS guard covers (it treats indices as body
           sub-expressions). Reject at compile time. *)
        Unsupported
          { node = Printf.sprintf "table `%s`" name;
            reason = Printf.sprintf
              "parameter '%s' is used as a non-constant lookup index into table \
               `%s`; the lookup selects a cell by its value, so it is not \
               differentiable. Index the table by a constant or a compartment, \
               not by an estimated parameter" param name }
      else if table_value_mentions name then
        (* The parameter is an inline-table VALUE selected by a non-constant
           index. The value is a live coefficient (the Rust runtime resolves it),
           but the gradient through a runtime-chosen cell is not yet emitted
           (gh#215). Omit it so the model compiles — IF2/PF use the live value,
           and the NUTS guard refuses a NUTS fit that depends on it. *)
        Known (Const 0.0)
      else Known (Const 0.0)

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
      (* Min/Max: subgradient — differentiate the active branch. *)
      | Min -> map2 (d b.left) (d b.right)
                 (fun dl dr -> Cond { pred = BinOp { op = Lt; left = b.left; right = b.right };
                                      then_ = dl; else_ = dr })
      | Max -> map2 (d b.left) (d b.right)
                 (fun dl dr -> Cond { pred = BinOp { op = Gt; left = b.left; right = b.right };
                                      then_ = dl; else_ = dr })
      (* Mod: derivative needs floor, absent from the grammar. A genuine 0 when
         neither operand depends on the param; otherwise Unsupported (was a
         failwith — M4 in the 2026-04-19 compiler review). *)
      | Mod ->
        if mentions param b.left || mentions param b.right then
          Unsupported
            { node = "mod expression";
              reason = Printf.sprintf
                "derivative of `mod` w.r.t. parameter '%s' is not representable \
                 in the IR grammar (floor is needed); replace mod with a \
                 conditional guard" param }
        else Known (Const 0.0)
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
      (* d|f| = f' · sign(f), sign(0) := 0 (n1 in the 2026-04-19 review). *)
      | Abs ->
        let sign =
          Cond { pred = BinOp { op = Gt; left = u.arg; right = Const 0.0 };
                 then_ = Const 1.0;
                 else_ = Cond { pred = BinOp { op = Lt; left = u.arg; right = Const 0.0 };
                                then_ = Const (-1.0); else_ = Const 0.0 } }
        in
        map1 (d u.arg) (fun da -> BinOp { op = Mul; left = da; right = sign })
      | Floor | Ceil -> Known (Const 0.0)
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

    (* Sum is linear: d/dp (Σ tᵢ) = Σ d/dp tᵢ; any Unsupported term propagates. *)
    | Reduce terms ->
      let rec collect acc = function
        | [] -> Known (Reduce (List.rev acc))
        | t :: rest ->
          (match d t with
           | Known e -> collect (e :: acc) rest
           | Unsupported _ as u -> u)
      in
      collect [] terms

    (* Hoisted FOI bindings are param-free (state-only): d/dp BindingRef = 0. *)
    | BindingRef _ -> Known (Const 0.0)
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

(** Differentiate a rate expression w.r.t. each estimated parameter.

    Returns [Ok assoc] — an association list [(param_name, derivative_expr)] —
    or [Error msg] if any parameter's derivative is [Unsupported] (the caller
    turns that into a compile-time diagnostic). Parameters whose derivative is
    a proven [Const 0.0] (absent from the rate) are omitted; the Rust backend
    treats a missing entry as a zero gradient. The [Unsupported] path is the
    only way a non-zero derivative is dropped, and it is never silent. *)
let differentiate_rate (rate : expr) (param_names : string list)
    (tfs : time_function list) (tbls : table list) :
    ((string * expr) list, string) result =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | p :: rest ->
      (match differentiate rate p tfs tbls with
       | Unsupported u -> Error (Printf.sprintf "%s: %s" u.node u.reason)
       | Known dexpr ->
         (match simplify_fixpoint dexpr with
          | Const 0.0 -> go acc rest
          | d' -> go ((p, d') :: acc) rest))
  in
  go [] param_names
