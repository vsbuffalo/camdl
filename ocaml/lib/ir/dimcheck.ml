(* Dimensional analysis checker for camdl IR.

   Base dimensions: P (population, index 0), T (time, index 1).
   A dimension is an int array of exponents: [| p_exp; t_exp |].

   Two-pass approach:
   1. Bottom-up: infer dimensions from leaves (known param kinds, Pop, Time)
   2. Top-down: propagate expected dimensions (P*T^-1 for rates) to resolve unknowns

   Tables and time functions get their dimensions inferred from their
   defining expressions, not hardcoded. *)

open Ir

(* ── Base dimension registry ────────────────────────────────────────────── *)

let n_bases = 2

type dim_vec = int array

let make p t =
  let d = Array.make n_bases 0 in
  d.(0) <- p; d.(1) <- t; d

let dimensionless = make 0 0
let population    = make 1 0
let rate_total    = make 1 (-1)  (* P*T^-1 *)

let dim_mul a b = Array.init n_bases (fun i -> a.(i) + b.(i))
let dim_div a b = Array.init n_bases (fun i -> a.(i) - b.(i))
let dim_scale n a = Array.init n_bases (fun i -> n * a.(i))
let dim_eq  a b = a = b  (* structural equality on int arrays *)
let dim_is_zero a = Array.for_all (fun x -> x = 0) a
let dim_is_even a = Array.for_all (fun x -> x mod 2 = 0) a
let dim_half a = Array.init n_bases (fun i -> a.(i) / 2)

(* ── Display ────────────────────────────────────────────────────────────── *)

let display_dim d =
  match (d.(0), d.(1)) with
  | (0, 0)  -> "dimensionless (probability, ratio)"
  | (1, 0)  -> "population count"
  | (0, 1)  -> "duration"
  | (0, -1) -> "per-capita rate"
  | (1, -1) -> "population-level rate"
  | (1, 1)  -> "population * duration"
  | (-1, 0) -> "inverse population (per-capita)"
  | _       -> Printf.sprintf "P^%d*T^%d" d.(0) d.(1)

let formal_dim d =
  match (d.(0), d.(1)) with
  | (0, 0)  -> "1"
  | (1, 0)  -> "P"
  | (0, 1)  -> "T"
  | (0, -1) -> "T^-1"
  | (1, -1) -> "P*T^-1"
  | (p, 0)  -> Printf.sprintf "P^%d" p
  | (0, t)  -> Printf.sprintf "T^%d" t
  | (p, t)  -> Printf.sprintf "P^%d*T^%d" p t

let dim_display d = Printf.sprintf "%s (%s)" (formal_dim d) (display_dim d)

(* ── Dimension type with unknowns ───────────────────────────────────────── *)

type dim =
  | Known of dim_vec
  | Unknown of int
  | Any                 (* Const 0.0: universal additive identity *)

(* ── Diagnostic results ─────────────────────────────────────────────────── *)

type severity = Error | Info

(* The construct a diagnostic concerns. Dimcheck runs on the IR, which has no
   source spans, so it records the subject and the compiler resolves it to the
   declaration's loc (the same prefix-match the validate/autodiff paths use). *)
type subject =
  | STransition  of string
  | SOde         of string   (* compartment whose ODE derivative *)
  | SObservation of string
  | SBinding     of string
  | SContrast    of string   (* contrast whose operand dimensions conflict *)

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  subject  : subject option;
}

type result = {
  diagnostics : diagnostic list;
  param_dims  : (string * dim_vec) list;
  (* Resolved dimension of each generated quantity, keyed by (name, stratum)
     (prerequisite #5 of the counterfactual-contrasts proposal). Carries only
     the dimensions that resolved to a concrete vector; the compiler writes
     these onto the `Quantity` IR nodes so the Rust contrast reducer can check
     operand-dimension agreement without re-deriving. *)
  quantity_dims : ((string * (string * string) list) * dim_vec) list;
}

(* ── Checker state ──────────────────────────────────────────────────────── *)

type param_dim_entry = {
  stable_dim : dim;
  mutable inferred : dim_vec option;
}

type state = {
  mutable next_var : int;
  resolved : (int, dim_vec) Hashtbl.t;
  links : (int, int) Hashtbl.t;  (* union-find: child -> parent *)
  mutable diags : diagnostic list;
  (* The construct currently being checked, set at each top-level loop so the
     emit helpers can tag diagnostics with their subject for loc resolution. *)
  mutable subject : subject option;
  (* Stable param dim map *)
  param_map : (string, param_dim_entry) Hashtbl.t;
  (* Pre-computed table dims *)
  table_dims : (string, dim) Hashtbl.t;
  (* Pre-computed time function dims *)
  tf_dims : (string, dim) Hashtbl.t;
  (* Cache of per-transition rate dims after inference rounds *)
  rate_cache : (string, dim) Hashtbl.t;
  (* Fix B: dim of each model-level shared binding (name → dim), precomputed
     in topological order before the transition rounds so a `BindingRef` infers
     the binding body's dim instead of an unconstrained fresh var — which would
     hide dimension conflicts (e.g. E303) that the inlined form caught. *)
  binding_dims : (string, dim) Hashtbl.t;
  (* Permissive-dim mode: observation-likelihood expressions use
     variance formulas (e.g. He et al. 2010 eq 6) that mix `1 +
     overdispersion * count` within a single sqrt, so strictly
     enforcing dimensional homogeneity there rejects standard
     epidemiological forms. When this flag is set, Add/Sub-mismatches
     and sqrt-of-odd-dim are silently absorbed (no E302/E304).
     See gh #4. *)
  mutable permissive_dim : bool;
  (* Dimension of the projection expression of the observation currently
     being checked. The `Projected` leaf in a likelihood stands for the
     projection's value, so it must carry the projection's *inferred*
     dimension — a count for `prevalence`/`incidence`, but dimensionless
     for a `DerivedExpr` proportion like `I / N`. None outside an
     observation, where it falls back to a population count. *)
  mutable projected_dim : dim option;
  (* Dimension of each per-observation aux column (`columns { tested : count }`),
     keyed by column name; set per-observation before its likelihood check. An
     [ObsColumnRef name] in a likelihood (e.g. binomial `n = tested`) infers this
     dim, so a declared `count` feeding `n` is verified as a count and a
     dimensionally-wrong column (a `real` feeding `n`) is a located E304. Cleared
     after each observation. (2026-06-10 observation data-entry §3.1.) *)
  obs_col_dims : (string, dim) Hashtbl.t;
}

let create_state () = {
  next_var = 0;
  resolved = Hashtbl.create 32;
  links = Hashtbl.create 32;
  diags = [];
  param_map = Hashtbl.create 32;
  table_dims = Hashtbl.create 16;
  tf_dims = Hashtbl.create 16;
  rate_cache = Hashtbl.create 16;
  binding_dims = Hashtbl.create 16;
  permissive_dim = false;
  subject = None;
  projected_dim = None;
  obs_col_dims = Hashtbl.create 8;
}

let fresh_var st =
  let id = st.next_var in
  st.next_var <- id + 1;
  Unknown id

let emit_error st ~code ~message ?detail ?hint () =
  st.diags <- { severity = Error; code; message; detail; hint; subject = st.subject } :: st.diags

let emit_info st ~code ~message ?detail ?hint () =
  st.diags <- { severity = Info; code; message; detail; hint; subject = st.subject } :: st.diags

(* ── Resolution ─────────────────────────────────────────────────────────── *)

(* Follow the union-find chain to the root variable *)
let rec find_root st id =
  match Hashtbl.find_opt st.links id with
  | None -> id
  | Some parent ->
    let root = find_root st parent in
    if root <> parent then Hashtbl.replace st.links id root;  (* path compression *)
    root

let resolve st = function
  | Known v -> Known v
  | Any -> Any
  | Unknown id ->
    let root = find_root st id in
    match Hashtbl.find_opt st.resolved root with
    | Some v -> Known v
    | None -> Unknown root

let bind st id v =
  let root = find_root st id in
  if not (Hashtbl.mem st.resolved root) then
    Hashtbl.replace st.resolved root v

(* Unify two dimensions. On mismatch with two Known values, emit E302. *)
let unify st ~loc d1 d2 =
  let d1 = resolve st d1 in
  let d2 = resolve st d2 in
  match d1, d2 with
  | Any, d | d, Any -> d
  | Known v1, Known v2 ->
    if dim_eq v1 v2 then Known v1
    else begin
      (* When one side is population and the other dimensionless, hint about
         typed let bindings — this is the iota / obs_floor pattern *)
      let hint =
        if (dim_eq v1 population && dim_eq v2 dimensionless)
        || (dim_eq v1 dimensionless && dim_eq v2 population) then
          Some "if a constant represents a small count (seeding term), declare it with a type:\n\
          \        let iota : count = 1e-6\n\
          \      then use: I + iota"
        else None
      in
      if not st.permissive_dim then
        emit_error st ~code:"E302"
          ~message:(Printf.sprintf "dimension mismatch in %s" loc)
          ~detail:(Printf.sprintf "left has dimension %s, right has dimension %s — cannot combine"
                     (dim_display v1) (dim_display v2))
          ?hint
          ();
      Known v1
    end
  | Known v, Unknown id | Unknown id, Known v ->
    bind st id v; Known v
  | Unknown id1, Unknown id2 ->
    let r1 = find_root st id1 in
    let r2 = find_root st id2 in
    if r1 = r2 then Unknown r1
    else begin
      Hashtbl.replace st.links r2 r1;  (* link r2 -> r1 *)
      Unknown r1
    end

let constrain_known st ~code ~message d expected =
  let d = resolve st d in
  match d with
  | Any -> ()
  | Known v ->
    if not (dim_eq v expected) then
      emit_error st ~code ~message
        ~detail:(Printf.sprintf "expected dimension %s, got %s"
                   (dim_display expected) (dim_display v))
        ()
  | Unknown id -> bind st id expected

(* ── Param dim from kind ────────────────────────────────────────────────── *)

let param_dim_of_kind st (kind : param_kind option) =
  match kind with
  | Some Rate           -> Known (make 0 (-1))
  | Some Probability    -> Known dimensionless
  | Some Count          -> Known population
  (* Both time kinds carry dimension [T] (2026-05-22 calendar-time §6.7):
     `instant` (origin-relative point) and `duration` (relative span). This
     gives the checker time coverage — `rate + instant` is an E302 dimension
     mismatch, `rate * duration` is dimensionless. The instant/duration
     distinction is for *rendering* (date vs span), not for checking. *)
  | Some Instant        -> Known (make 0 1)
  | Some Duration       -> Known (make 0 1)
  (* Positive / Real / unannotated: no declared dimension — inferred. *)
  | Some Positive | Some Real | None -> fresh_var st

(* ── Initialize state from model ────────────────────────────────────────── *)

let init_params st (params : parameter list) =
  List.iter (fun (p : parameter) ->
    (* Explicit [dim] annotation takes highest priority over kind-based inference *)
    let d = match p.param_dim with
      | Some (p_exp, t_exp) -> Known (make p_exp t_exp)
      | None -> param_dim_of_kind st p.param_kind
    in
    Hashtbl.replace st.param_map p.name { stable_dim = d; inferred = None }
  ) params

(* ── Bottom-up inference ────────────────────────────────────────────────── *)

(* A `Reduce [t0; …; tn]` is dimensionally and for display exactly the Add chain
   that `sum(...)` lowered to before Fix D. Desugar to that chain to reuse the
   existing Add logic. The expander does not yet emit Reduce, so these arms are
   currently unreached; they become live when ESum is switched in a later
   increment (proposal 2026-05-29-shared-bindings-and-reduction). *)
let rec reduce_add_chain : expr list -> expr = function
  | []        -> Const 0.0
  | [t]       -> t
  | t :: rest -> BinOp { op = Add; left = t; right = reduce_add_chain rest }

let rec infer st ~ctx (e : expr) : dim =
  match e with
  | Const 0.0 -> Any
  | Const _   -> Known dimensionless
  | Param name ->
    (match Hashtbl.find_opt st.param_map name with
     | Some entry -> resolve st entry.stable_dim
     | None -> fresh_var st)
  | Pop _ | PopSum _ -> Known population
  | Time -> Known (make 0 1)
  | Dt   -> Known (make 0 1)  (* gh#54: dt has dimension T, same as t *)
  | TimeFunc name ->
    (match Hashtbl.find_opt st.tf_dims name with
     | Some d -> resolve st d
     | None -> fresh_var st)
  | TableLookup (name, idx_exprs) ->
    List.iter (fun ie -> ignore (infer st ~ctx ie)) idx_exprs;
    (match Hashtbl.find_opt st.table_dims name with
     | Some d -> resolve st d
     | None -> fresh_var st)
  | Projected ->
    (match st.projected_dim with Some d -> resolve st d | None -> Known population)
  | ObsColumnRef name ->
    (* The aux column's declared dim (e.g. `tested : count` → population),
       registered per-observation in [obs_col_dims]. An unregistered ref
       (invalid model) floors to a fresh var so no spurious mismatch fires. *)
    (match Hashtbl.find_opt st.obs_col_dims name with
     | Some d -> resolve st d
     | None -> fresh_var st)
  | UncheckedDim u ->
    (* Per-expression dimensional escape. The declared dim is
       authoritative; we do NOT recurse into `u.inner` for unification
       purposes. That's the whole point of the escape — the inner
       expression is allowed to be dimensionally irregular, e.g.
       `(I + ι)^α` in the He et al. 2010 measles formulation. *)
    Known (make u.dim_p u.dim_t)
  | BinOp b -> infer_binop st ~ctx b
  | UnOp u -> infer_unop st ~ctx u
  | Cond c -> infer_cond st ~ctx c
  | Reduce terms -> infer st ~ctx (reduce_add_chain terms)
  | BindingRef name ->
    (* Fix B: a binding's dim is its body's dim, precomputed in check_model. *)
    (match Hashtbl.find_opt st.binding_dims name with
     | Some d -> resolve st d
     | None -> fresh_var st)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

and is_bare_const = function
  | Const _ -> true
  | UnOp { op = Neg; arg } -> is_bare_const arg
  | _ -> false

and infer_binop st ~ctx (b : bin_op_expr) : dim =
  let dl = infer st ~ctx b.left in
  let dr = infer st ~ctx b.right in
  match b.op with
  | Add | Sub | Min | Max | Mod ->
    unify st ~loc:ctx dl dr
  | Mul ->
    let dl = resolve st dl in
    let dr = resolve st dr in
    (match dl, dr with
     | Any, d | d, Any -> d
     | Known v1, Known v2 -> Known (dim_mul v1 v2)
     | _ -> fresh_var st)
  | Div ->
    (* Special case: Const/Const (e.g. 1/730) is dimensionally ambiguous.
       In epi models this commonly represents a rate (1/duration).
       Treat as unknown so the solver can infer from context. *)
    if is_bare_const b.left && is_bare_const b.right then
      fresh_var st
    else begin
      let dl = resolve st dl in
      let dr = resolve st dr in
      (match dl, dr with
       | Any, _ -> Any
       | _, Any -> dl
       | Known v1, Known v2 -> Known (dim_div v1 v2)
       | _ -> fresh_var st)
    end
  | Pow -> infer_pow st ~ctx b
  | Eq | Neq | Lt | Gt | Le | Ge ->
    ignore (unify st ~loc:ctx dl dr);
    Known dimensionless

and infer_pow st ~ctx (b : bin_op_expr) : dim =
  let dl = resolve st (infer st ~ctx b.left) in
  let dr = resolve st (infer st ~ctx b.right) in
  (match dr with
   | Known v when not (dim_is_zero v) ->
     emit_error st ~code:"E301"
       ~message:(Printf.sprintf "exponent in '%s' has non-dimensionless dimension" ctx) ()
   | _ -> ());
  match dl with
  | Any -> Any
  | Known v when dim_is_zero v -> Known dimensionless
  | Known v ->
    (match b.right with
     | Const n when Float.is_integer n ->
       Known (dim_scale (Float.to_int n) v)
     | _ ->
       emit_error st ~code:"E301"
         ~message:(Printf.sprintf "non-constant exponent with dimensioned base in '%s'" ctx)
         ~detail:(Printf.sprintf "base has dimension %s" (dim_display v)) ();
       Known dimensionless)
  | _ -> fresh_var st

and infer_unop st ~ctx (u : un_op_expr) : dim =
  let da = infer st ~ctx u.arg in
  match u.op with
  | Neg | Abs | Floor | Ceil -> da
  | Exp | Log | Sin | Cos | Tanh ->
    constrain_known st ~code:"E301"
      ~message:(Printf.sprintf "argument to %s in '%s' must be dimensionless"
                  (match u.op with
                   | Exp -> "exp" | Log -> "log"
                   | Sin -> "sin" | Cos -> "cos" | Tanh -> "tanh"
                   | _ -> "log") ctx)
      da dimensionless;
    Known dimensionless
  | Sqrt ->
    (match resolve st da with
     | Any -> Any
     | Known v ->
       if dim_is_even v then Known (dim_half v)
       else if st.permissive_dim then Any
       else begin
         emit_error st ~code:"E304"
           ~message:(Printf.sprintf "sqrt in '%s' requires even dimension exponents" ctx)
           ~detail:(Printf.sprintf "argument has dimension %s" (dim_display v)) ();
         Known dimensionless
       end
     | Unknown _ -> fresh_var st)

and infer_cond st ~ctx (c : cond_expr) : dim =
  (* M18 in the 2026-04-19 review: the predicate can carry any
     dimension. The IR spec's canonical guard form
     `Cond(Pop("S"), beta * S * I / N, Const 0.0)` — prevent
     division-by-zero by using the population-valued S as a "is S
     empty?" guard — has dim(pred) = population, not dimensionless.
     Previously this site forced pred to be dimensionless via E302,
     producing a spurious false positive for that exact idiom.
     `Cond` semantics is "predicate is truthy iff > 0", which works
     for both population (positive ⇔ non-empty) and boolean (0/1).
     Drop the predicate dim constraint; keep the branch unification
     since then/else must still agree dimensionally. *)
  let _ = infer st ~ctx c.pred in
  let dt = infer st ~ctx c.then_ in
  let de = infer st ~ctx c.else_ in
  unify st ~loc:ctx dt de

(* ── Top-down propagation ───────────────────────────────────────────────── *)

let rec propagate st ~ctx (e : expr) (expected : dim_vec) : unit =
  match e with
  | Const _ -> ()
  | Param name ->
    (match Hashtbl.find_opt st.param_map name with
     | Some entry ->
       (match resolve st entry.stable_dim with
        | Unknown id -> bind st id expected
        | _ -> ())
     | None -> ())
  | Pop _ | PopSum _ | Time | Dt | Projected | ObsColumnRef _ -> ()
  | UncheckedDim _ ->
    (* The escape stops dim-propagation at this boundary — the inner
       is explicitly exempted from checker-driven dimensional
       constraints. *)
    ()
  | TimeFunc name ->
    (* If the time function's dim is unknown, bind it *)
    (match Hashtbl.find_opt st.tf_dims name with
     | Some d ->
       (match resolve st d with
        | Unknown id -> bind st id expected
        | _ -> ())
     | None -> ())
  | TableLookup (name, _) ->
    (match Hashtbl.find_opt st.table_dims name with
     | Some d ->
       (match resolve st d with
        | Unknown id -> bind st id expected
        | _ -> ())
     | None -> ())
  | UnOp u ->
    (match u.op with
     | Neg | Abs | Floor | Ceil -> propagate st ~ctx u.arg expected
     | Exp | Log | Sin | Cos | Tanh -> propagate st ~ctx u.arg dimensionless
     | Sqrt -> propagate st ~ctx u.arg (dim_scale 2 expected))
  | BinOp b ->
    (match b.op with
     | Add | Sub | Min | Max | Mod ->
       propagate st ~ctx b.left expected;
       propagate st ~ctx b.right expected
     | Mul -> propagate_mul st ~ctx b.left b.right expected
     | Div -> propagate_div st ~ctx b.left b.right expected
     | Pow -> ()
     | Eq | Neq | Lt | Gt | Le | Ge -> ())
  | Cond c ->
    propagate st ~ctx c.then_ expected;
    propagate st ~ctx c.else_ expected
  | Reduce terms -> propagate st ~ctx (reduce_add_chain terms) expected
  | BindingRef _ -> ()   (* leaf; inc1b may propagate the expectation into the binding *)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

(* Flatten a multiplicative chain into (numerator_factors, denominator_factors).
   E.g. Div(Mul(Mul(a,b),c), d) → ([a;b;c], [d]) *)
and collect_product_factors (e : expr) : expr list * expr list =
  match e with
  | BinOp { op = Mul; left; right; } ->
    let (nl, dl) = collect_product_factors left in
    let (nr, dr) = collect_product_factors right in
    (nl @ nr, dl @ dr)
  | BinOp { op = Div; left; right; } ->
    let (nl, dl) = collect_product_factors left in
    let (nr, dr) = collect_product_factors right in
    (nl @ dr, dl @ nr)  (* right goes to denominator *)
  | _ -> ([e], [])

and propagate_mul st ~ctx left right expected =
  (* Flatten the product chain and partition into known vs unknown factors *)
  let all_expr = BinOp { op = Mul; left; right } in
  propagate_product st ~ctx all_expr expected

and propagate_div st ~ctx left right expected =
  let all_expr = BinOp { op = Div; left; right } in
  propagate_product st ~ctx all_expr expected

and propagate_product st ~ctx e expected =
  let (num, den) = collect_product_factors e in
  (* Compute the aggregate known dimension from all factors *)
  let known_dim = ref dimensionless in
  let unknown_factors = ref [] in
  List.iter (fun factor ->
    let d = resolve st (infer st ~ctx factor) in
    match d with
    | Known v -> known_dim := dim_mul !known_dim v
    | Any -> ()  (* 0 or dimensionless, skip *)
    | Unknown _ -> unknown_factors := (factor, true) :: !unknown_factors
  ) num;
  List.iter (fun factor ->
    let d = resolve st (infer st ~ctx factor) in
    match d with
    | Known v -> known_dim := dim_div !known_dim v
    | Any -> ()
    | Unknown _ -> unknown_factors := (factor, false) :: !unknown_factors
  ) den;
  let residual = dim_div expected !known_dim in
  match !unknown_factors with
  | [(factor, is_num)] ->
    (* Single unknown factor: its dim is fully determined *)
    let target = if is_num then residual else dim_scale (-1) residual in
    propagate st ~ctx factor target
  | _ ->
    (* Multiple unknowns or none: try to propagate into each unknown factor
       individually. We can't fully resolve, but propagation into each factor
       with the residual may help on subsequent rounds when other constraints
       resolve some of them. *)
    ()

(* ── Infer table dimensions from their values ──────────────────────────── *)

let init_table_dims st (tables : table list) =
  List.iter (fun (tbl : table) ->
    let dim = match tbl.cell_kind with
      | Some _ ->
        (* gh#32: explicit cell-type annotation. Stamp the table with
           the declared dim — exactly the way scalar parameters work.
           [cell_kind] is still a free string (out of scope of the typed
           param_kind enum); an unrecognised kind maps to [None] → a fresh
           dim var, matching the prior catch-all behaviour. *)
        param_dim_of_kind st (Option.bind tbl.cell_kind param_kind_of_name)
      | None ->
        match tbl.source with
        | External _ -> fresh_var st  (* can't know *)
        | Inline exprs ->
          (* If all values are bare constants, the table dimension is ambiguous
             (e.g. age durations in years, contact matrix entries). Treat as
             unknown so the solver can infer from context. *)
          let all_const = List.for_all (fun e ->
            match e with Const _ -> true | _ -> false
          ) exprs in
          if all_const then
            fresh_var st
          else begin
            st.subject <- None;
            let ctx = Printf.sprintf "table '%s'" tbl.name in
            let dims = List.map (fun e -> infer st ~ctx e) exprs in
            (match dims with
             | [] -> Known dimensionless
             | d :: rest ->
               List.fold_left (fun acc d2 -> unify st ~loc:ctx acc d2) d rest)
          end
    in
    Hashtbl.replace st.table_dims tbl.name dim
  ) tables

(* ── Infer time function dimensions ─────────────────────────────────────── *)

let init_tf_dims st (tfs : time_function list) =
  (* GH #8: every forcing declaration carries a tier-3 unit literal
     (e.g. `'per_day`, `'count`, `'ratio`) and the expander populates
     `time_function.dim` from it. The dim-checker uses that as
     authoritative — no value-based inference — because
     value-based inference fails for `interpolated` (data-file values
     are always EConst → dimensionless) and the declared dim is
     strictly more informative for the other three kinds too. *)
  List.iter (fun (tf : time_function) ->
    let (p, t) = tf.dim in
    Hashtbl.replace st.tf_dims tf.name (Known (make p t))
  ) tfs

(* ── Read-only dimension query (no fresh vars, no diagnostics) ─────────── *)

(* Walk an expression and resolve its dimension from already-known state.
   Returns None if any part is unresolved. Used in the check phase to avoid
   side effects from re-running infer. *)
let rec read_dim st (e : expr) : dim =
  match e with
  | Const 0.0 -> Any
  | Const _   -> Known dimensionless
  | Param name ->
    (match Hashtbl.find_opt st.param_map name with
     | Some entry -> resolve st entry.stable_dim
     | None -> Unknown (-1))  (* sentinel: unknown, but don't allocate *)
  | Pop _ | PopSum _ -> Known population
  | Time -> Known (make 0 1)
  | Dt   -> Known (make 0 1)
  | TimeFunc name ->
    (match Hashtbl.find_opt st.tf_dims name with
     | Some d -> resolve st d
     | None -> Unknown (-1))
  | TableLookup (name, _) ->
    (match Hashtbl.find_opt st.table_dims name with
     | Some d -> resolve st d
     | None -> Unknown (-1))
  | Projected ->
    (match st.projected_dim with Some d -> resolve st d | None -> Known population)
  | ObsColumnRef name ->
    (match Hashtbl.find_opt st.obs_col_dims name with
     | Some d -> resolve st d
     | None -> Unknown (-1))
  | UncheckedDim u -> Known (make u.dim_p u.dim_t)
  | BinOp b -> read_dim_binop st b
  | UnOp u -> read_dim_unop st u
  | Cond c ->
    let dt = read_dim st c.then_ in
    let _de = read_dim st c.else_ in
    dt  (* branches already unified during inference *)
  | Reduce terms -> read_dim st (reduce_add_chain terms)
  | BindingRef name ->
    (* Fix B: read-only twin of the infer arm — the binding's precomputed dim,
       so the error-checking phase (E300/E303) sees through a BindingRef. *)
    (match Hashtbl.find_opt st.binding_dims name with
     | Some d -> resolve st d
     | None -> Unknown (-1))
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

and read_dim_binop st (b : bin_op_expr) : dim =
  let dl = read_dim st b.left in
  let dr = read_dim st b.right in
  match b.op with
  | Add | Sub | Min | Max | Mod ->
    (* Already unified; return whichever is known *)
    (match dl, dr with
     | Any, d | d, Any -> d
     | Known _, _ -> dl
     | _, Known _ -> dr
     | _ -> dl)
  | Mul ->
    (match resolve st dl, resolve st dr with
     | Any, d | d, Any -> d
     | Known v1, Known v2 -> Known (dim_mul v1 v2)
     | _ -> Unknown (-1))
  | Div ->
    if is_bare_const b.left && is_bare_const b.right then
      Unknown (-1)
    else
      (match resolve st dl, resolve st dr with
       | Any, _ -> Any
       | _, Any -> dl
       | Known v1, Known v2 -> Known (dim_div v1 v2)
       | _ -> Unknown (-1))
  | Pow ->
    (* M19 in 2026-04-19 review: the inference phase correctly
       emits E301 for Pow with a non-integer-literal exponent
       when the base carries a non-zero dimension. The read-only
       phase used to silently return `Known dimensionless` for the
       same case, masking the error if `read_dim` was the first to
       see this node (via implied_param_dim → read_dim calls during
       the E303 cross-transition check). Return Unknown (-1) here so
       no wrong dimension escapes the read phase; the dimension will
       be inferred / errored on the main pass. *)
    (match resolve st dl with
     | Any -> Any
     | Known v when dim_is_zero v -> Known dimensionless
     | Known v ->
       (match b.right with
        | Const n when Float.is_integer n -> Known (dim_scale (Float.to_int n) v)
        | _ -> Unknown (-1))
     | _ -> Unknown (-1))
  | Eq | Neq | Lt | Gt | Le | Ge -> Known dimensionless

and read_dim_unop st (u : un_op_expr) : dim =
  let da = read_dim st u.arg in
  match u.op with
  | Neg | Abs | Floor | Ceil -> da
  | Exp | Log | Sin | Cos | Tanh -> Known dimensionless
  | Sqrt ->
    (match resolve st da with
     | Any -> Any
     | Known v ->
       if dim_is_even v then Known (dim_half v)
       else Known dimensionless
     | _ -> Unknown (-1))

(* ── Expression printer (for error messages) ────────────────────────────── *)

let rec expr_to_short_string (e : expr) : string =
  match e with
  | Const f ->
    if Float.is_integer f && Float.abs f < 1e9 then
      string_of_int (Float.to_int f)
    else Printf.sprintf "%g" f
  | Param s -> s
  | Pop s -> s
  | PopSum ss -> String.concat " + " ss
  | Time -> "t"
  | Dt   -> "dt"
  | TimeFunc s -> Printf.sprintf "%s(t)" s
  | TableLookup (s, _) -> Printf.sprintf "%s[...]" s
  | Projected -> "projected"
  | ObsColumnRef c -> c
  | UncheckedDim u ->
    Printf.sprintf "unchecked_dim(%s)" (expr_to_short_string u.inner)
  | BinOp b ->
    let op_str = match b.op with
      | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
      | Pow -> "^" | Mod -> "%%" | Min -> "min" | Max -> "max"
      | Eq -> "==" | Neq -> "!=" | Lt -> "<" | Gt -> ">"
      | Le -> "<=" | Ge -> ">="
    in
    Printf.sprintf "(%s %s %s)"
      (expr_to_short_string b.left) op_str (expr_to_short_string b.right)
  | UnOp u ->
    let op_str = match u.op with
      | Neg -> "-" | Exp -> "exp" | Log -> "log" | Sqrt -> "sqrt"
      | Abs -> "abs" | Floor -> "floor" | Ceil -> "ceil"
      | Sin -> "sin" | Cos -> "cos" | Tanh -> "tanh"
    in
    Printf.sprintf "%s(%s)" op_str (expr_to_short_string u.arg)
  | Cond c ->
    Printf.sprintf "if(%s, %s, %s)"
      (expr_to_short_string c.pred)
      (expr_to_short_string c.then_)
      (expr_to_short_string c.else_)
  | Reduce terms ->
    "(" ^ String.concat " + " (List.map expr_to_short_string terms) ^ ")"
  | BindingRef n -> n
  | PerEvalRef n -> n   (* display helper: render the name, never crash in error formatting *)

(* ── Generated quantities + counterfactual contrasts ──────────────────────────
   Run AFTER the transition/observation inference rounds (params resolved). Two
   jobs: (a) compute each quantity's resolved dimension for the #5 stored field
   (read-only — `read_dim` never emits or allocates fresh vars, so adding this
   never surfaces a new error on an existing quantity-bearing model); and (b)
   the contrast dimensional check — operands of `+`/`-` must agree. *)

(* The result dimension of a temporal reduction of a series of dimension
   [series]. Final/Max/Min/Mean preserve it; CountAbove/CountBelow count snapshots
   (dimensionless); a time reduction yields T; an integral multiplies by T; a bare
   series keeps it. *)
let reduced_value_dim (series : dim) (reduce : temporal_reduce option) : dim =
  match reduce with
  | None -> series
  | Some (RValue (VFinal | VMax | VMin | VMean)) -> series
  | Some (RValue (VCountAbove _ | VCountBelow _)) -> Known dimensionless
  | Some (RTime _) -> Known (make 0 1)
  | Some RIntegral ->
    (match series with Known v -> Known (dim_mul v (make 0 1)) | d -> d)

(* The dimension of a declared observation stream — its projection's dimension
   (population for incidence/prevalence; the inferred dim for a DerivedExpr).
   Read-only. *)
let obs_projection_dim st (m : model) (stream : string) : dim =
  match List.find_opt (fun (o : observation_model) -> o.name = stream) m.observations with
  | Some o ->
    (match o.projection with
     | DerivedExpr e -> read_dim st e
     | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _ ->
       Known population)
  | None -> Unknown (-1)

(* Build the (name, stratum) → resolved-dimension table for all quantities, in
   declaration order so a Derived body's QRef reads an earlier leaf's dim. *)
let compute_quantity_dim_table st (m : model)
    : ((string * (string * string) list), dim) Hashtbl.t =
  let resolved : ((string * (string * string) list), dim) Hashtbl.t = Hashtbl.create 16 in
  let lookup_qref (name : string) (stratum : (string * string) list) : dim =
    match Hashtbl.find_opt resolved (name, stratum) with
    | Some d -> d
    | None ->
      (* fall back to any leaf of that name (dim is stratum-invariant) *)
      Hashtbl.fold (fun (n, _) d acc ->
        match acc with Some _ -> acc | None -> if n = name then Some d else None)
        resolved None
      |> (function Some d -> d | None -> Unknown (-1))
  in
  let rec scalar_dim (se : scalar_expr) : dim =
    match se with
    | SConst _ -> Any
    | SParam p ->
      (match Hashtbl.find_opt st.param_map p with
       | Some entry -> resolve st entry.stable_dim
       | None -> Unknown (-1))
    | SQRef q -> lookup_qref q.qref_name q.qref_stratum
    | SUnOp { op; arg } ->
      let da = scalar_dim arg in
      (match op with
       | Neg | Abs | Floor | Ceil -> da
       | Exp | Log | Sin | Cos | Tanh -> Known dimensionless
       | Sqrt -> (match resolve st da with
                  | Any -> Any
                  | Known v when dim_is_even v -> Known (dim_half v)
                  | Known _ -> Known dimensionless
                  | _ -> Unknown (-1)))
    | SBinOp { op; left; right } ->
      let dl = scalar_dim left and dr = scalar_dim right in
      (match op with
       | Add | Sub | Min | Max | Mod ->
         (match dl, dr with Any, d | d, Any -> d | Known _, _ -> dl | _, Known _ -> dr | _ -> dl)
       | Mul -> (match resolve st dl, resolve st dr with
                 | Any, d | d, Any -> d
                 | Known v1, Known v2 -> Known (dim_mul v1 v2) | _ -> Unknown (-1))
       | Div -> (match resolve st dl, resolve st dr with
                 | Any, _ -> Any | _, Any -> dl
                 | Known v1, Known v2 -> Known (dim_div v1 v2) | _ -> Unknown (-1))
       | Pow -> Unknown (-1)
       | Eq | Neq | Lt | Gt | Le | Ge -> Known dimensionless)
    | SCond { then_; _ } -> scalar_dim then_
  in
  List.iter (fun (q : quantity) ->
    let d = match q.q_body with
      | QBReduced { source; reduce } ->
        let sd = match source with
          | QSState e -> read_dim st e
          | QSObservation stream -> obs_projection_dim st m stream
        in reduced_value_dim sd reduce
      | QBDerived se -> scalar_dim se
    in
    Hashtbl.replace resolved (q.q_name, q.q_stratum) (resolve st d)
  ) m.quantities;
  resolved

(* A short rendering of a contrast operand for diagnostics. *)
let rec show_contrast_expr (ce : contrast_expr) : string =
  match ce with
  | CRunMember { run; ns; member } ->
    let nss = match ns with NsQuantities -> "quantities" | NsObservations -> "observations" in
    Printf.sprintf "%s.%s.%s" run nss member
  | CBinOp { op; left; right } ->
    let os = match op with
      | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/" | _ -> "?" in
    Printf.sprintf "(%s %s %s)" (show_contrast_expr left) os (show_contrast_expr right)

(* The dimensional check on every `contrasts { }` entry: the operands of `+`/`-`
   must agree (a `deaths - rate` contrast is meaningless). Permissive on
   unresolved operands (fires only on a Known mismatch), mirroring the
   observation-likelihood `constrain_known` policy. *)
let check_contrasts st (m : model)
    (qdims : ((string * (string * string) list), dim) Hashtbl.t) : unit =
  let member_dim ns member : dim =
    match ns with
    | NsQuantities ->
      Hashtbl.fold (fun (n, _) d acc ->
        match acc with Some _ -> acc | None -> if n = member then Some d else None)
        qdims None
      |> (function Some d -> d | None -> Unknown (-1))
    | NsObservations -> obs_projection_dim st m member
  in
  let rec infer_ce (ce : contrast_expr) : dim =
    match ce with
    | CRunMember { ns; member; _ } -> resolve st (member_dim ns member)
    | CBinOp { op; left; right } ->
      let dl = infer_ce left and dr = infer_ce right in
      (match op with
       | Add | Sub ->
         (match dl, dr with
          | Known v1, Known v2 when not (dim_eq v1 v2) ->
            emit_error st ~code:"E297"
              ~message:(Printf.sprintf
                "contrast operands have incompatible dimensions: \
                 `%s` is %s but `%s` is %s"
                (show_contrast_expr left)  (dim_display v1)
                (show_contrast_expr right) (dim_display v2))
              ~hint:"the two arms of a difference must report the same quantity \
                     (same dimension); compare counts with counts, rates with rates"
              ();
            dl
          | Any, d | d, Any -> d
          | Known _, _ -> dl
          | _ -> dr)
       | Mul -> (match dl, dr with
                 | Any, d | d, Any -> d
                 | Known v1, Known v2 -> Known (dim_mul v1 v2) | _ -> Unknown (-1))
       | Div -> (match dl, dr with
                 | Any, _ -> Any | _, Any -> dl
                 | Known v1, Known v2 -> Known (dim_div v1 v2) | _ -> Unknown (-1))
       | _ -> Unknown (-1))
  in
  List.iter (fun (c : contrast) ->
    st.subject <- Some (SContrast c.c_name);
    ignore (infer_ce c.c_body)
  ) m.contrasts;
  st.subject <- None

(* ── Main check ─────────────────────────────────────────────────────────── *)

let check_model (m : model) : result =
  let st = create_state () in

  (* Initialize parameter dims *)
  init_params st m.parameters;

  (* Initialize table dims from their values *)
  init_table_dims st m.tables;

  (* Initialize time function dims *)
  init_tf_dims st m.time_functions;

  (* gh#314: a forcing's optional `lag` is a duration — it is SUBTRACTED from
     the evaluation time (`t − lag`), so it must carry the time dimension [T].
     A non-duration lag (a rate `'per_day`, a count, a dimensionless literal)
     is a model bug: subtracting it from `t` is meaningless. Validate it the
     same way `period`/`phase` are durations, with its own code (E309) so the
     diagnostic points at the forcing, not at an arithmetic site. *)
  List.iter (fun (tf : time_function) ->
    match tf.lag with
    | None -> ()
    | Some lag_e ->
      st.subject <- None;
      let ctx = Printf.sprintf "forcing '%s' lag" tf.name in
      let d = infer st ~ctx lag_e in
      constrain_known st ~code:"E309"
        ~message:(Printf.sprintf
          "forcing '%s': `lag` must be a duration (dimension T); it is \
           subtracted from the evaluation time. Use a duration literal \
           (e.g. `lag = 10 'days`) or a parameter declared `: duration`."
          tf.name)
        d (make 0 1)
  ) m.time_functions;

  (* Fix B: precompute each shared binding's dim in declaration (topological)
     order, so a BindingRef in a rate infers the body's dim. Bindings are
     param-free, so their dim is fully determined here and stable across the
     transition rounds below. *)
  List.iter (fun (b : binding) ->
    st.subject <- Some (SBinding b.bname);
    let ctx = Printf.sprintf "binding '%s'" b.bname in
    let d = infer st ~ctx b.bexpr in
    Hashtbl.replace st.binding_dims b.bname d
  ) m.bindings;

  (* Pass 1: bottom-up inference + top-down propagation for each transition *)
  (* We do multiple rounds to propagate resolved unknowns across transitions.
     Two rounds suffices for most models (first round resolves most params,
     second round picks up cross-transition effects). *)
  for _round = 1 to 3 do
    List.iter (fun (tr : transition) ->
      st.subject <- Some (STransition tr.name);
      let ctx = Printf.sprintf "transition '%s'" tr.name in
      ignore (infer st ~ctx tr.rate);
      propagate st ~ctx tr.rate rate_total;
      (* Overdispersion *)
      (match tr.draw_method with
       | DrawOverdispersed sigma_sq ->
         ignore (infer st ~ctx sigma_sq);
         propagate st ~ctx sigma_sq dimensionless
       | _ -> ())
    ) m.transitions;

    (* Balance *)
    (match m.balance with
     | Some bal ->
       st.subject <- None;
       let ctx = Printf.sprintf "balance '%s'" bal.balance_target in
       ignore (infer st ~ctx bal.balance_expr);
       propagate st ~ctx bal.balance_expr population
     | None -> ());

    (* ODE *)
    List.iter (fun (eq : ode_equation) ->
      st.subject <- Some (SOde eq.compartment);
      let ctx = Printf.sprintf "ODE d(%s)/dt" eq.compartment in
      ignore (infer st ~ctx eq.derivative);
      propagate st ~ctx eq.derivative rate_total
    ) m.ode_equations;

    (* Observations — permissive-dim covers the *mean / rate / sd*
       positions because standard epi variance formulas (He et al.
       2010 eq 6: sqrt(rho*m*(1 + rho*psi^2*m))) are not strictly
       dim-homogeneous under the P-counting convention. Suppress
       E302/E304 there; downstream likelihood evaluation treats
       those expressions as pure floats regardless.

       gh#116 (2026-05-26 upstream OCaml-compiler review Critical
       #6): the blanket-permissive coverage was too wide — arguments
       with KNOWN dimensional contracts (Binomial.p, Bernoulli.p,
       BetaBinomial.alpha/beta, NegBinomial.dispersion) silently
       accepted dimension-incompatible expressions. A common
       missing-`/N` bug (`binomial(n = N, p = projected)` where
       `projected` is a count) compiled and ran with no signal,
       turning a prevalence likelihood into a count-valued
       probability.

       The fix: keep permissive-dim ON for variance-formula-bearing
       positions (mean / rate / sd) but STRICTLY check the
       known-contract arguments. `constrain_known` inspects the
       *resolved* dim; it fires E304 only when the inferred
       dimension is Known and ≠ expected — so it doesn't false-
       positive on Unknown / Any expressions where the user
       legitimately leaned on inference. *)
    let prev_permissive = st.permissive_dim in
    st.permissive_dim <- true;
    List.iter (fun (obs : observation_model) ->
      st.subject <- Some (SObservation obs.name);
      let ctx = Printf.sprintf "observation '%s'" obs.name in
      (* Register each declared value column's dim so an [ObsColumnRef name] in
         this likelihood (e.g. binomial `n = tested`, Poisson `offset` folded
         into the rate) infers its declared dimension. This is what makes the
         dimcheck of `n`/rate below actually bite: declaring `tested : count`
         registers it as a count, so a `real` column feeding `n` is an E304.
         (2026-06-10 observation data-entry §3.1.) *)
      Hashtbl.reset st.obs_col_dims;
      List.iter (fun (c : obs_column) ->
        match c.col_role with
        | RoleValue k -> Hashtbl.replace st.obs_col_dims c.col_name (param_dim_of_kind st (Some k))
        | RoleTime | RoleDim _ -> ()
      ) obs.columns;
      (* The Projected leaf in this likelihood stands for the projection's
         value, so it carries the projection's dimension: a population count
         for prevalence/incidence, the inferred dim for a DerivedExpr — e.g.
         dimensionless for a proportion `I / N`. This is what lets a
         legitimate proportion-as-probability pass while still catching the
         missing-`/N` bug (a count used as `p`). *)
      st.projected_dim <- Some (match obs.projection with
        | DerivedExpr e -> infer st ~ctx e
        | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _ ->
          Known population);
      (match obs.likelihood with
       | NegBinomial nb ->
         ignore (infer st ~ctx nb.mean);
         let disp_dim = infer st ~ctx nb.dispersion in
         (* Pre-existing constraint, kept for symmetry with the new
            checks below. propagate handles Unknown binding;
            constrain_known catches Known-mismatch. *)
         propagate st ~ctx nb.dispersion dimensionless;
         constrain_known st ~code:"E304"
           ~message:(Printf.sprintf
             "%s: NegBinomial `dispersion` must be dimensionless" ctx)
           disp_dim dimensionless
       | Poisson p ->
         (* The Poisson `rate` is the expected COUNT of events over the
            reporting interval — dimension [population], not a per-time rate.
            A person-time offset is folded into this argument
            (`rate = lambda * person_time`), which keeps the product a count.
            Previously inferred-and-discarded (un-checked); now constrained so a
            dimensionally-wrong rate is a located E304. A bare numeric literal
            (`rate = 100`) carries no real dimension — it is a count by context,
            like a seeding constant — so it is exempt. (2026-06-10 §3.1.) *)
         if not (is_bare_const p.rate) then begin
           let rate_dim = infer st ~ctx p.rate in
           constrain_known st ~code:"E304"
             ~message:(Printf.sprintf
               "%s: Poisson `rate` must have the dimension of a count \
                (expected events over the reporting interval)" ctx)
             rate_dim population
         end else ignore (infer st ~ctx p.rate)
       | Normal n -> ignore (infer st ~ctx n.mean); ignore (infer st ~ctx n.sd)
       | Binomial b ->
         (* The binomial `n` is an external DENOMINATOR — a count (e.g. number
            tested). Previously inferred-and-discarded; now constrained so a
            declared `tested : count` is verified, and a `real`/probability
            column feeding `n` is a located E304. A bare numeric literal
            (`n = 100`) is a count by context and exempt. (2026-06-10 §3.1.) *)
         if not (is_bare_const b.n) then begin
           let n_dim = infer st ~ctx b.n in
           constrain_known st ~code:"E304"
             ~message:(Printf.sprintf
               "%s: Binomial `n` must be a count (the trial denominator); \
                declare the column feeding `n` as `count`." ctx)
             n_dim population
         end else ignore (infer st ~ctx b.n);
         (* gh#116: `p` must be a probability (dimensionless on
            [0, 1]). A count here is the textbook missing-`/N` bug. *)
         let p_dim = infer st ~ctx b.p in
         constrain_known st ~code:"E304"
           ~message:(Printf.sprintf
             "%s: Binomial `p` must be dimensionless (probability); \
              a count here is almost certainly a missing `/N`." ctx)
           p_dim dimensionless
       | BetaBinomial bb ->
         (* BetaBinomial `n` is the same external count denominator as the
            Binomial; constrain it likewise (bare literal exempt). (§3.1.) *)
         if not (is_bare_const bb.n) then begin
           let n_dim = infer st ~ctx bb.n in
           constrain_known st ~code:"E304"
             ~message:(Printf.sprintf
               "%s: BetaBinomial `n` must be a count (the trial denominator); \
                declare the column feeding `n` as `count`." ctx)
             n_dim population
         end else ignore (infer st ~ctx bb.n);
         (* gh#116: alpha/beta are shape parameters of a Beta
            distribution — both dimensionless by definition. *)
         let a_dim = infer st ~ctx bb.alpha in
         let b_dim = infer st ~ctx bb.beta in
         constrain_known st ~code:"E304"
           ~message:(Printf.sprintf
             "%s: BetaBinomial `alpha` must be dimensionless" ctx)
           a_dim dimensionless;
         constrain_known st ~code:"E304"
           ~message:(Printf.sprintf
             "%s: BetaBinomial `beta` must be dimensionless" ctx)
           b_dim dimensionless
       | Bernoulli b ->
         (* gh#116: same as Binomial.p — must be a probability. *)
         let p_dim = infer st ~ctx b.p in
         constrain_known st ~code:"E304"
           ~message:(Printf.sprintf
             "%s: Bernoulli `p` must be dimensionless (probability); \
              a count here is almost certainly a missing `/N`." ctx)
           p_dim dimensionless);
      st.projected_dim <- None;
      Hashtbl.reset st.obs_col_dims
    ) m.observations;
    st.permissive_dim <- prev_permissive
  done;

  (* Cache inferred rate dims for use in the read-only check phase *)
  List.iter (fun (tr : transition) ->
    let d = read_dim st tr.rate in
    Hashtbl.replace st.rate_cache tr.name d
  ) m.transitions;

  (* Pass 2: check constraints and emit errors (read-only — no fresh vars) *)

  (* Transition rates must be P*T^-1 *)
  List.iter (fun (tr : transition) ->
    st.subject <- Some (STransition tr.name);
    let d = resolve st
      (match Hashtbl.find_opt st.rate_cache tr.name with
       | Some cached -> cached
       | None -> read_dim st tr.rate) in
    (match d with
     | Known v when not (dim_eq v rate_total) ->
       emit_error st ~code:"E300"
         ~message:(Printf.sprintf "transition '%s' rate has wrong dimension" tr.name)
         ~detail:(Printf.sprintf "rate = %s\n  expected dimension: %s\n  got dimension: %s"
                    (expr_to_short_string tr.rate)
                    (dim_display rate_total)
                    (dim_display v))
         ()
     | _ -> ());
    (* Overdispersion sigma^2 must be dimensionless *)
    (match tr.draw_method with
     | DrawOverdispersed sigma_sq ->
       let sd = resolve st (read_dim st sigma_sq) in
       (match sd with
        | Known v when not (dim_is_zero v) ->
          emit_error st ~code:"E308"
            ~message:(Printf.sprintf "overdispersion sigma^2 in '%s' must be dimensionless" tr.name)
            ~detail:(Printf.sprintf "got dimension %s" (dim_display v)) ()
        | _ -> ())
     | _ -> ())
  ) m.transitions;

  (* Balance *)
  (match m.balance with
   | Some bal ->
     st.subject <- None;
     let d = resolve st (read_dim st bal.balance_expr) in
     (match d with
      | Known v when not (dim_eq v population) ->
        emit_error st ~code:"E305"
          ~message:(Printf.sprintf "balance expression for '%s' has wrong dimension" bal.balance_target)
          ~detail:(Printf.sprintf "expected %s, got %s" (dim_display population) (dim_display v))
          ()
      | _ -> ())
   | None -> ());

  (* ODE *)
  List.iter (fun (eq : ode_equation) ->
    st.subject <- Some (SOde eq.compartment);
    let d = resolve st (read_dim st eq.derivative) in
    (match d with
     | Known v when not (dim_eq v rate_total) ->
       emit_error st ~code:"E306"
         ~message:(Printf.sprintf "ODE derivative for '%s' has wrong dimension" eq.compartment)
         ~detail:(Printf.sprintf "expected %s, got %s" (dim_display rate_total) (dim_display v))
         ()
     | _ -> ())
  ) m.ode_equations;

  (* Observation dispersion *)
  List.iter (fun (obs : observation_model) ->
    st.subject <- Some (SObservation obs.name);
    (match obs.likelihood with
     | NegBinomial nb ->
       let dd = resolve st (read_dim st nb.dispersion) in
       (match dd with
        | Known v when not (dim_is_zero v) ->
          emit_error st ~code:"E307"
            ~message:(Printf.sprintf "dispersion parameter in observation '%s' must be dimensionless" obs.name)
            ~detail:(Printf.sprintf "got dimension %s" (dim_display v)) ()
        | _ -> ())
     | _ -> ())
  ) m.observations;

  (* E303: cross-transition parameter consistency.
     For each parameter, determine what dimension each transition's rate
     context implies, independently of global resolution. If a parameter
     requires dim A in one transition and dim B in another, emit E303. *)
  let param_transition_dims : (string, (string * dim_vec) list) Hashtbl.t = Hashtbl.create 16 in

  (* Compute the implied dimension of a named parameter in an expression,
     given that the overall expression must have dimension [target].
     Uses read_dim for all sub-expressions except occurrences of [param_name],
     which are treated as the single unknown.
     Returns Some dim_vec if uniquely determined, None otherwise. *)
  let rec implied_param_dim st param_name (e : expr) (target : dim_vec) : dim_vec option =
    match e with
    | Param name when name = param_name -> Some target
    | BinOp { op = Add; left; right; _ }
    | BinOp { op = Sub; left; right; _ }
    | BinOp { op = Min; left; right; _ }
    | BinOp { op = Max; left; right; _ } ->
      (* Both sides must match target *)
      let from_l = implied_param_dim st param_name left target in
      let from_r = implied_param_dim st param_name right target in
      (match from_l, from_r with
       | Some d, None | None, Some d -> Some d
       | Some _, Some _ -> from_l  (* both paths give same answer *)
       | None, None -> None)
    | BinOp { op = Mul; _ } | BinOp { op = Div; _ } ->
      (* Flatten product, find param as the single unknown, compute residual.
         If the param is inside a non-leaf factor (e.g. Add), treat that factor
         as the "unknown" and recurse into it with the residual dim. *)
      let (num, den) = collect_product_factors e in
      let known_dim = ref dimensionless in
      let param_factors = ref [] in  (* (factor, is_num) for factors containing param *)
      let other_unknown = ref 0 in
      let classify_factor factor is_num =
        let has_param = ref false in
        let rec check = function
          | Param name when name = param_name -> has_param := true
          | BinOp b -> check b.left; check b.right
          | UnOp u -> check u.arg
          | Cond c -> check c.pred; check c.then_; check c.else_
          | _ -> ()
        in
        check factor;
        if !has_param then
          param_factors := (factor, is_num) :: !param_factors
        else begin
          let d = resolve st (read_dim st factor) in
          match d with
          | Known v ->
            if is_num then known_dim := dim_mul !known_dim v
            else known_dim := dim_div !known_dim v
          | Any -> ()
          | Unknown _ -> incr other_unknown
        end
      in
      List.iter (fun f -> classify_factor f true) num;
      List.iter (fun f -> classify_factor f false) den;
      if List.length !param_factors = 1 && !other_unknown = 0 then begin
        let (sub_expr, is_num) = List.hd !param_factors in
        let residual = dim_div target !known_dim in
        let sub_target = if is_num then residual else dim_scale (-1) residual in
        (* If the sub_expr is the param itself, we're done *)
        (match sub_expr with
         | Param name when name = param_name -> Some sub_target
         | _ -> implied_param_dim st param_name sub_expr sub_target)
      end else
        None
    | Cond { then_; else_; _ } ->
      let from_t = implied_param_dim st param_name then_ target in
      let from_e = implied_param_dim st param_name else_ target in
      (match from_t, from_e with
       | Some d, None | None, Some d -> Some d
       | Some _, Some _ -> from_t
       | None, None -> None)
    | UnOp { op = Neg; arg; _ } | UnOp { op = Abs; arg; _ }
    | UnOp { op = Floor; arg; _ } | UnOp { op = Ceil; arg; _ } ->
      implied_param_dim st param_name arg target
    | UnOp { op = Sqrt; arg; _ } ->
      implied_param_dim st param_name arg (dim_scale 2 target)
    | _ -> None
  in

  (* Collect param names used in each transition *)
  let rec params_in (e : expr) acc =
    match e with
    | Param name -> if List.mem name acc then acc else name :: acc
    | BinOp b -> params_in b.left (params_in b.right acc)
    | UnOp u -> params_in u.arg acc
    | Cond c -> params_in c.pred (params_in c.then_ (params_in c.else_ acc))
    | TableLookup (_, idxs) -> List.fold_left (fun a e -> params_in e a) acc idxs
    | _ -> acc
  in

  List.iter (fun (tr : transition) ->
    let pnames = params_in tr.rate [] in
    List.iter (fun pname ->
      match implied_param_dim st pname tr.rate rate_total with
      | Some implied ->
        let existing = match Hashtbl.find_opt param_transition_dims pname with
          | Some l -> l | None -> [] in
        if not (List.exists (fun (tn, _) -> tn = tr.name) existing) then
          Hashtbl.replace param_transition_dims pname ((tr.name, implied) :: existing)
      | None -> ()
    ) pnames
  ) m.transitions;

  Hashtbl.iter (fun name entries ->
    match entries with
    | [] | [_] -> ()
    | first :: rest ->
      let (first_tr, first_dim) = first in
      st.subject <- Some (STransition first_tr);
      List.iter (fun (other_tr, other_dim) ->
        if not (dim_eq first_dim other_dim) then
          emit_error st ~code:"E303"
            ~message:(Printf.sprintf "parameter '%s' has conflicting dimensions" name)
            ~detail:(Printf.sprintf
              "In transition '%s': inferred %s (%s)\n  In transition '%s': inferred %s (%s)"
              first_tr (formal_dim first_dim) (display_dim first_dim)
              other_tr (formal_dim other_dim) (display_dim other_dim))
            ()
      ) rest
  ) param_transition_dims;

  (* Collect resolved param dims; emit I300 for undetermined *)
  st.subject <- None;
  let param_dims = ref [] in
  Hashtbl.iter (fun name entry ->
    let d = resolve st entry.stable_dim in
    match d with
    | Known v ->
      param_dims := (name, v) :: !param_dims;
      entry.inferred <- Some v
    | Unknown _ ->
      emit_info st ~code:"I300"
        ~message:(Printf.sprintf "dimension of parameter '%s' could not be determined" name)
        ~hint:"annotate with a more specific kind (rate, probability, count)" ()
    | Any -> ()
  ) st.param_map;

  (* Generated quantities (#5 stored dimension) + the contrast dimensional check.
     Runs last, after all param/table/binding dims are resolved. The quantity
     dims are read-only (`read_dim`); only the contrast check emits. *)
  st.subject <- None;
  let qdim_table = compute_quantity_dim_table st m in
  check_contrasts st m qdim_table;
  let quantity_dims =
    Hashtbl.fold (fun key d acc ->
      match d with Known v -> (key, v) :: acc | _ -> acc)
      qdim_table [] in

  { diagnostics = List.rev st.diags;
    param_dims = !param_dims;
    quantity_dims }
