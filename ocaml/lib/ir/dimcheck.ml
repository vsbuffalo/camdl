(* Dimensional analysis of an expanded model, as one system of linear
   equations over integer exponent vectors.

   Base dimensions: P (population, index 0), T (time, index 1). A dimension is
   an int array of exponents. Dimensions form a free abelian group on the
   bases, so every rule of inference is linear: a product's dimension is the
   sum of its factors', a sum's equals each operand's, a static power [a^n] is
   n times its base's (A. Kennedy, "Dimension types", ESOP 1994, LNCS 788,
   pp. 348–362).

   Three pieces:
   - [Dim_sites.sites] names every expression-bearing position of the model
     with the dimension its expression must have (its obligation).
   - [gen] turns each expression into a linear form over dimension variables
     (one per parameter, table, binding, observation stream, column and
     quantity), emitting an equation wherever two forms must agree (the
     operands of +, a function's argument, a branch of a conditional).
   - [Dim_solve] solves every equation at once, in the order declared facts,
     then expression structure, then obligations, and reports every
     contradiction and every integer infeasibility, which become located
     diagnostics with the code of the position they surface at.

   An operator means the same thing at every position; the positions differ
   only in their obligation (proposal
   docs/dev/proposals/2026-11-06-dimension-checking-as-one-constraint-system.md). *)

open Ir

(* ── Dimensions ────────────────────────────────────────────────────────── *)

let n_bases = 2

type dim_vec = int array

let make p t = [| p; t |]

let dimensionless = make 0 0
let population    = make 1 0
let time_dim      = make 0 1

let dim_mul a b = Array.init n_bases (fun i -> a.(i) + b.(i))
let dim_div a b = Array.init n_bases (fun i -> a.(i) - b.(i))
let dim_eq  a b = a = b

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

let un_op_name : un_op -> string = function
  | Neg -> "-" | Exp -> "exp" | Log -> "log" | Sqrt -> "sqrt"
  | Abs -> "abs" | Floor -> "floor" | Ceil -> "ceil"
  | Sin -> "sin" | Cos -> "cos" | Tanh -> "tanh"

(* ── Results ───────────────────────────────────────────────────────────── *)

type severity = Error | Info

type subject = Dim_sites.subject =
  | STransition  of string
  | SOde         of string
  | SObservation of string
  | SBinding     of string
  | SContrast    of string
  | SInit        of string

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  subject  : subject option;
}

type result = {
  diagnostics   : diagnostic list;
  param_dims    : (string * dim_vec) list;
  quantity_dims : ((string * (string * string) list) * dim_vec) list;
}

(* ── Expression printer (for messages) ─────────────────────────────────── *)

let rec expr_to_short_string (e : expr) : string =
  match e with
  | Const f ->
    if Float.is_integer f && Float.abs f < 1e9 then string_of_int (Float.to_int f)
    else Printf.sprintf "%g" f
  | Param s | ConstRef s -> s
  | Pop s -> s
  | PopSum ss -> String.concat " + " ss
  | Time -> "t"
  | Dt -> "dt"
  | TimeFunc s -> Printf.sprintf "%s(t)" s
  | ForcingBasis { forcing; index; space } ->
    Printf.sprintf "%s_%d(%s)"
      (match space with Space_spline -> "basis" | Space_whitened -> "whitened" | Space_linear -> "linear")
      index forcing
  | TableLookup (s, _) -> Printf.sprintf "%s[...]" s
  | Projected -> "projected"
  | ObsColumnRef c -> c
  | ObsAnchor a ->
    let name = match a.anchor with AnchorFirst -> "first_obs" | AnchorLast -> "last_obs" in
    if a.offset = 0.0 then name
    else if a.offset > 0.0 then Printf.sprintf "%s + %g" name a.offset
    else Printf.sprintf "%s - %g" name (-. a.offset)
  | UncheckedDim u -> Printf.sprintf "unchecked_dim(%s)" (expr_to_short_string u.inner)
  | BinOp b ->
    let op_str = match b.op with
      | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
      | Pow -> "^" | Mod -> "%%" | Min -> "min" | Max -> "max"
      | Eq -> "==" | Neq -> "!=" | Lt -> "<" | Gt -> ">"
      | Le -> "<=" | Ge -> ">="
    in
    Printf.sprintf "(%s %s %s)" (expr_to_short_string b.left) op_str (expr_to_short_string b.right)
  | UnOp u -> Printf.sprintf "%s(%s)" (un_op_name u.op) (expr_to_short_string u.arg)
  | Cond c ->
    Printf.sprintf "if(%s, %s, %s)" (expr_to_short_string c.pred)
      (expr_to_short_string c.then_) (expr_to_short_string c.else_)
  | Reduce terms -> "(" ^ String.concat " + " (List.map expr_to_short_string terms) ^ ")"
  | BindingRef n -> n
  | PerEvalRef n -> n

(* ── Linear forms ──────────────────────────────────────────────────────── *)

(* A dimension as Σ cᵥ·xᵥ + k: [vars] sorted by variable, no zero
   coefficients. [Any] is the dimension of the literal 0, which takes any
   dimension and constrains nothing. *)
type form = { vars : (int * int) list; k : dim_vec }
type value = Any | F of form

let const_form v = { vars = []; k = v }
let var_form x = { vars = [ (x, 1) ]; k = Array.make n_bases 0 }

let rec merge f xs ys =
  match xs, ys with
  | [], r -> List.filter_map (fun (v, c) -> let c = f 0 c in if c = 0 then None else Some (v, c)) r
  | r, [] -> List.filter_map (fun (v, c) -> let c = f c 0 in if c = 0 then None else Some (v, c)) r
  | (vx, cx) :: tx, (vy, cy) :: ty ->
    if vx < vy then (let c = f cx 0 in if c = 0 then merge f tx ys else (vx, c) :: merge f tx ys)
    else if vy < vx then (let c = f 0 cy in if c = 0 then merge f xs ty else (vy, c) :: merge f xs ty)
    else (let c = f cx cy in if c = 0 then merge f tx ty else (vx, c) :: merge f tx ty)

let form_add a b = { vars = merge ( + ) a.vars b.vars; k = dim_mul a.k b.k }
let form_sub a b = { vars = merge ( - ) a.vars b.vars; k = dim_div a.k b.k }
let form_scale n a =
  if n = 0 then const_form dimensionless
  else { vars = List.map (fun (v, c) -> (v, n * c)) a.vars; k = Array.map (( * ) n) a.k }

(* ── Checker state ─────────────────────────────────────────────────────── *)

(* What a diagnostic needs to know about an equation that conflicts. *)
type eq_kind =
  | Declared_fact
  | Operands of { op : string; left : form; right : form }   (* two forms must agree *)
  | Function_arg of { fn : string; arg : form }              (* must be dimensionless *)
  | Even_root of form                                        (* sqrt *)
  | Obligation_of of { expected : dim_vec; got : form }       (* a site's obligation *)
  | Definition of { got : form }                              (* Defines key *)
  | Quantity_reduction

type eq_info = { site : Dim_sites.site option; code : string; kind : eq_kind }

type state = {
  mutable t_only : bool;   (* the site being generated is an [Equals_t] site *)
  t_only_eqs : (int, unit) Hashtbl.t;   (* equations that hold on T alone *)
  mutable next_var : int;
  keys     : (Dim_sites.key, int) Hashtbl.t;
  mutable declared  : Dim_solve.eqn list;   (* each list reversed *)
  mutable structure : Dim_solve.eqn list;
  mutable obligation : Dim_solve.eqn list;
  info     : (int, eq_info) Hashtbl.t;
  mutable next_eq : int;
  tf_dims  : (string, dim_vec) Hashtbl.t;
}

let fresh st = let x = st.next_var in st.next_var <- x + 1; x

let var_of st key =
  match Hashtbl.find_opt st.keys key with
  | Some x -> x
  | None -> let x = fresh st in Hashtbl.replace st.keys key x; x

(* Record [form = rhs] (rhs a constant), in [stage]. *)
let emit st stage ~site ~code kind (f : form) (rhs : dim_vec) =
  let id = st.next_eq in
  st.next_eq <- id + 1;
  Hashtbl.replace st.info id { site; code; kind };
  if st.t_only then Hashtbl.replace st.t_only_eqs id ();
  let e = { Dim_solve.terms = f.vars; rhs = dim_div rhs f.k;
            origin = { site = id; stage; what = code } } in
  match stage with
  | Dim_solve.Declared -> st.declared <- e :: st.declared
  | Structure -> st.structure <- e :: st.structure
  | Obligation -> st.obligation <- e :: st.obligation

(* [a = b] as a structure equation; nothing when either is [Any], or when both
   are the same constant. *)
let equate st ~site ~code ~op a b =
  match a, b with
  | Any, _ | _, Any -> ()
  | F l, F r ->
    if l.vars = [] && r.vars = [] && dim_eq l.k r.k then ()
    else
      emit st Structure ~site ~code (Operands { op; left = l; right = r })
        (form_sub l r) dimensionless

let declare st key v =
  emit st Declared ~site:None ~code:"E302" Declared_fact (var_form (var_of st key)) v

(* ── Declared facts ────────────────────────────────────────────────────── *)

let kind_dim : param_kind -> dim_vec option = function
  | Rate -> Some (make 0 (-1))
  | Probability -> Some dimensionless
  | Count -> Some population
  (* Both time kinds are [T] (2026-05-22 calendar-time §6.7); the
     instant/duration distinction is for rendering, not checking. *)
  | Instant | Duration -> Some time_dim
  | Positive | Real -> None

let declare_model st (m : model) =
  List.iter (fun (p : parameter) ->
    let key = Dim_sites.K_param p.name in
    ignore (var_of st key);
    match p.param_dim, Option.bind p.param_kind kind_dim with
    | Some (dp, dt), _ -> declare st key (make dp dt)
    | None, Some v -> declare st key v
    | None, None -> ()) m.parameters;
  List.iter (fun (c : constant) ->
    let key = Dim_sites.K_param c.c_name in
    ignore (var_of st key);
    Option.iter (declare st key) (kind_dim c.c_kind)) m.constants;
  List.iter (fun (t : table) ->
    match Option.bind (Option.bind t.cell_kind param_kind_of_name) kind_dim with
    | Some v -> declare st (Dim_sites.K_table t.name) v
    | None -> ()) m.tables;
  List.iter (fun (tf : time_function) ->
    let (p, t) = tf.dim in Hashtbl.replace st.tf_dims tf.name (make p t)) m.time_functions;
  List.iter (fun (o : observation_model) ->
    (match o.projection with
     | DerivedExpr _ -> ()
     | CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _ ->
       declare st (Dim_sites.K_stream o.name) population
     (* A ratio of two accumulated counts is a fraction. *)
     | FlowRatio _ -> declare st (Dim_sites.K_stream o.name) dimensionless);
    List.iter (fun (c : obs_column) ->
      match c.col_role with
      | RoleValue k ->
        Option.iter (declare st (Dim_sites.K_column (o.name, c.col_name))) (kind_dim k)
      | RoleTime | RoleWindowStart | RoleWindowStop | RoleDim _ -> ()) o.columns)
    m.observations

(* A quantity reports a reduction of its series: the value keeps the series'
   dimension, a count of snapshots is a pure number, a time is T, an integral
   multiplies by T. *)
let declare_quantities st (m : model) =
  List.iter (fun (q : quantity) ->
    match q.q_body with
    | QBDerived _ -> ()
    | QBReduced { source; reduce } ->
      let series = match source with
        | QSState _ -> var_of st (Dim_sites.K_series q.q_name)
        | QSObservation s -> var_of st (Dim_sites.K_stream s)
      in
      let qv = var_form (var_of st (Dim_sites.K_quantity q.q_name)) in
      let eq f rhs = emit st Structure ~site:None ~code:"E302" Quantity_reduction f rhs in
      match reduce with
      | None | Some (RValue (VFinal | VMax | VMin | VMean | VValueAt _)) ->
        eq (form_sub qv (var_form series)) dimensionless
      | Some (RValue (VCountAbove _ | VCountBelow _)) -> eq qv dimensionless
      | Some (RTime _) -> eq qv time_dim
      | Some RIntegral -> eq (form_sub qv (var_form series)) time_dim) m.quantities

(* ── Expressions ───────────────────────────────────────────────────────── *)

let rec is_bare_const = function
  | Const _ -> true
  | UnOp { op = Neg; arg } -> is_bare_const arg
  | _ -> false

let rec chain_add = function
  | [] -> Const 0.0
  | [ t ] -> t
  | t :: rest -> BinOp { op = Add; left = t; right = chain_add rest }

(* The binary-operator rules, shared by model expressions, derived quantities
   and contrasts. [exponent] is the power's exponent when it is a literal. *)
let binop st ~site ~code ~what (op : bin_op) ?exponent ~fresh_div l r : value =
  let opname = match op with
    | Add -> "+" | Sub -> "-" | Min -> "min" | Max -> "max" | Mod -> "%"
    | Eq -> "==" | Neq -> "!=" | Lt -> "<" | Gt -> ">" | Le -> "<=" | Ge -> ">="
    | Mul -> "*" | Div -> "/" | Pow -> "^" in
  match op with
  | Add | Sub | Min | Max | Mod ->
    equate st ~site ~code ~op:(Printf.sprintf "%s in %s" opname what) l r;
    (match l with Any -> r | F _ -> l)
  | Eq | Neq | Lt | Gt | Le | Ge ->
    equate st ~site ~code ~op:(Printf.sprintf "%s in %s" opname what) l r;
    F (const_form dimensionless)
  | Mul ->
    (match l, r with Any, d | d, Any -> d | F a, F b -> F (form_add a b))
  | Div ->
    if fresh_div then F (var_form (fresh st))
    else (match l, r with
        | Any, _ -> Any
        | _, Any -> l
        | F a, F b -> F (form_sub a b))
  | Pow ->
    (* The exponent is a pure number. *)
    (match r with
     | F e when e.vars = [] && not (dim_eq e.k dimensionless) ->
       emit st Structure ~site ~code:"E301"
         (Function_arg { fn = "the exponent of ^"; arg = e }) e dimensionless
     | Any | F _ -> ());
    (match l with
     | Any -> Any
     | F b when b.vars = [] && dim_eq b.k dimensionless -> F (const_form dimensionless)
     | F b ->
       (match exponent with
        | Some n when Float.is_integer n -> F (form_scale (Float.to_int n) b)
        | Some _ | None ->
          if b.vars = [] then begin
            (* A dimensioned base under a non-integer or variable exponent. *)
            emit st Structure ~site ~code:"E301"
              (Function_arg { fn = Printf.sprintf "a base under a non-integer exponent in %s" what;
                              arg = b }) b dimensionless;
            F (const_form dimensionless)
          end else F (var_form (fresh st))))

let unop st ~site ~what (op : un_op) (a : value) : value =
  match op with
  | Neg | Abs | Floor | Ceil -> a
  | Exp | Log | Sin | Cos | Tanh ->
    (match a with
     | Any -> ()
     | F f ->
       if not (f.vars = [] && dim_eq f.k dimensionless) then
         emit st Structure ~site ~code:"E301"
           (Function_arg { fn = Printf.sprintf "%s in %s" (un_op_name op) what; arg = f })
           f dimensionless);
    F (const_form dimensionless)
  | Sqrt ->
    (match a with
     | Any -> Any
     | F f when f.vars = [] && Array.for_all (fun x -> x mod 2 = 0) f.k ->
       F (const_form (Array.map (fun x -> x / 2) f.k))
     | F f ->
       let x = fresh st in
       (* 2x = a *)
       emit st Structure ~site ~code:"E304" (Even_root f)
         (form_sub (form_scale 2 (var_form x)) f) dimensionless;
       F (var_form x))

(* [term]: [e] is an entire additive term of the site's expression, where a
   literal takes the position's dimension under [Whole_terms]. *)
let rec gen st ~(site : Dim_sites.site) ?(term = false) (e : expr) : value =
  let what = site.role in
  let g = gen st ~site in
  match e with
  | Const 0.0 -> Any
  | Const _ ->
    (match site.literals with
     | Dim_sites.Literals_free -> F (var_form (fresh st))
     | Whole_terms when term -> Any
     | Whole_terms | Literals_dimensionless -> F (const_form dimensionless))
  | BinOp { op = (Add | Sub) as op; left; right }
    when not (match right with UncheckedDim { reason; _ } -> reason = spline_log_shift_reason
                               | _ -> false) ->
    let l = g ~term left and r = g ~term right in
    binop st ~site:(Some site) ~code:"E302" ~what op ~fresh_div:false l r
  | UnOp { op = Neg; arg } -> g ~term arg
  | Param n | ConstRef n -> F (var_form (var_of st (Dim_sites.K_param n)))
  | Pop _ | PopSum _ -> F (const_form population)
  | Time | Dt | ObsAnchor _ -> F (const_form time_dim)
  (* A basis function is a pure number (the basis sums to one). *)
  | ForcingBasis _ -> F (const_form dimensionless)
  | TimeFunc n ->
    (match Hashtbl.find_opt st.tf_dims n with
     | Some d -> F (const_form d)
     | None -> F (var_form (fresh st)))
  | TableLookup (n, idx) ->
    List.iter (fun i -> ignore (g i)) idx;
    F (var_form (var_of st (Dim_sites.K_table n)))
  | Projected ->
    (match site.stream with
     | Some s -> F (var_form (var_of st (Dim_sites.K_stream s)))
     | None -> F (const_form population))
  | ObsColumnRef c ->
    (match site.stream with
     | Some s -> F (var_form (var_of st (Dim_sites.K_column (s, c))))
     | None -> F (var_form (fresh st)))
  | UncheckedDim u -> F (const_form (make u.dim_p u.dim_t))
  (* Under `link = log` the expander adds ln s to a coefficient whose unit
     has scale s, as a marked escape; the coefficient under it is what its
     position constrains. *)
  | BinOp { op = Add; left; right = UncheckedDim { reason; _ } }
    when reason = spline_log_shift_reason -> g left
  | BinOp b ->
    let l = g b.left and r = g b.right in
    let exponent = match b.right with Const n -> Some n | _ -> None in
    let fresh_div =
      (* A ratio of two literals takes the dimension its context needs, until
         the literal rule of §5.2 lands. *)
      b.op = Div && is_bare_const b.left && is_bare_const b.right in
    binop st ~site:(Some site) ~code:"E302" ~what b.op ?exponent ~fresh_div l r
  | UnOp u -> unop st ~site:(Some site) ~what u.op (g u.arg)
  | Cond c ->
    ignore (g c.pred);
    let t = g ~term c.then_ and f = g ~term c.else_ in
    equate st ~site:(Some site) ~code:"E302" ~op:(Printf.sprintf "the branches of a conditional in %s" what) t f;
    (match t with Any -> f | F _ -> t)
  | Reduce terms -> g ~term (chain_add terms)
  | BindingRef n -> F (var_form (var_of st (Dim_sites.K_binding n)))
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"

let rec gen_scalar st ~(site : Dim_sites.site) (se : scalar_expr) : value =
  let g = gen_scalar st ~site in
  match se with
  | SConst _ -> Any
  | SParam p -> F (var_form (var_of st (Dim_sites.K_param p)))
  | SQRef q -> F (var_form (var_of st (Dim_sites.K_quantity q.qref_name)))
  | SUnOp { op; arg } -> unop st ~site:(Some site) ~what:site.role op (g arg)
  | SBinOp { op; left; right } ->
    let exponent = match right with SConst n -> Some n | _ -> None in
    binop st ~site:(Some site) ~code:"E302" ~what:site.role op ?exponent ~fresh_div:false
      (g left) (g right)
  | SCond { pred; then_; else_ } ->
    ignore (g pred);
    let t = g then_ and f = g else_ in
    equate st ~site:(Some site) ~code:"E302" ~op:"the branches of a conditional" t f;
    (match t with Any -> f | F _ -> t)

let rec gen_contrast st ~(site : Dim_sites.site) (ce : contrast_expr) : value =
  match ce with
  | CRunMember { run = _; ns = NsQuantities; member } ->
    F (var_form (var_of st (Dim_sites.K_quantity member)))
  | CRunMember { run = _; ns = NsObservations; member } ->
    F (var_form (var_of st (Dim_sites.K_stream member)))
  | CBinOp { op; left; right } ->
    let l = gen_contrast st ~site left and r = gen_contrast st ~site right in
    (match op with
     | Add | Sub | Mul | Div ->
       binop st ~site:(Some site) ~code:"E297" ~what:site.role op ~fresh_div:false l r
     | Pow | Mod | Min | Max | Eq | Neq | Lt | Gt | Le | Ge -> F (var_form (fresh st)))

(* One site: its expression's equations, then its obligation. *)
let add_site st (s : Dim_sites.site) =
  match s.obligation, s.strength with
  | Dim_sites.Exempt _, _ | _, Dim_sites.Checked -> ()
  | (Has _ | Equals _ | Equals_t _ | Defines _ | Free), Binding ->
    st.t_only <- (match s.obligation with Equals_t _ -> true | _ -> false);
    let v = match s.expr with
      | Dim_sites.E e -> gen st ~site:s ~term:true e
      | Scalar se -> gen_scalar st ~site:s se
      | Contrast ce -> gen_contrast st ~site:s ce
    in
    (match v, s.obligation with
    | Any, _ | _, (Free | Exempt _) -> ()
    | F f, Has d ->
      emit st Obligation ~site:(Some s) ~code:s.code
        (Obligation_of { expected = d; got = f }) f d
    | F f, (Equals key | Defines key) ->
      emit st Structure ~site:(Some s) ~code:s.code (Definition { got = f })
        (form_sub f (var_form (var_of st key))) dimensionless
    | F f, Equals_t key ->
      emit st Obligation ~site:(Some s) ~code:s.code (Definition { got = f })
        (form_sub f (var_form (var_of st key))) dimensionless);
    st.t_only <- false

(* ── Solving ───────────────────────────────────────────────────────────── *)

(* The system decouples by base, so an equation that holds on T alone (count
   noise, §5.6) is honoured by solving each base on its own: P without those
   equations, T with every equation. Without one, a single solve over both
   bases gives the same answer. *)
type solved = {
  value     : int -> dim_vec option;
  conflicts : Dim_solve.conflict list;
  relations : int list -> (Dim_solve.relation * int option) list;
  (* [Some b]: a relation on base b's exponents alone *)
}

let solve_all st (eqs : Dim_solve.eqn list) : solved =
  if Hashtbl.length st.t_only_eqs = 0 then begin
    let out = Dim_solve.solve ~nbases:n_bases eqs in
    { value = Dim_solve.value out; conflicts = Dim_solve.conflicts out;
      relations = (fun vs -> List.map (fun r -> (r, None)) (Dim_solve.relations out vs)) }
  end else begin
    let on_base b keep =
      Dim_solve.solve ~nbases:1
        (List.filter_map (fun (e : Dim_solve.eqn) ->
           if keep e then Some { e with rhs = [| e.rhs.(b) |] } else None) eqs) in
    let p = on_base 0 (fun e -> not (Hashtbl.mem st.t_only_eqs e.origin.site))
    and t = on_base 1 (fun _ -> true) in
    let value v = match Dim_solve.value p v, Dim_solve.value t v with
      | Some a, Some b -> Some [| a.(0); b.(0) |]
      | _ -> None in
    let cp = Dim_solve.conflicts p in
    let seen = List.map (fun (c : Dim_solve.conflict) -> c.culprit.origin.site) cp in
    let ct = List.filter (fun (c : Dim_solve.conflict) ->
      not (List.mem c.culprit.origin.site seen)) (Dim_solve.conflicts t) in
    let conflicts = List.sort (fun (a : Dim_solve.conflict) b ->
      compare a.culprit.origin.site b.culprit.origin.site) (cp @ ct) in
    let relations vs =
      List.map (fun r -> (r, Some 0)) (Dim_solve.relations p vs)
      @ List.map (fun r -> (r, Some 1)) (Dim_solve.relations t vs) in
    { value; conflicts; relations }
  end

(* ── Diagnostics ───────────────────────────────────────────────────────── *)

(* A form's value under the solution, when every variable in it is determined. *)
let form_value out (f : form) : dim_vec option =
  List.fold_left (fun acc (v, c) ->
    match acc, out.value v with
    | Some k, Some d -> Some (Array.mapi (fun i ki -> ki + c * d.(i)) k)
    | _ -> None) (Some (Array.copy f.k)) f.vars

let show out f = match form_value out f with
  | Some d -> dim_display d
  | None -> "not determined by the model"

let count_hint a b =
  if (dim_eq a population && dim_eq b dimensionless)
  || (dim_eq a dimensionless && dim_eq b population) then
    Some "if a constant represents a small count (seeding term), declare it with a type:\n\
         \        let iota : count = 1e-6\n\
         \      then use: I + iota"
  else None

let site_name (s : Dim_sites.site option) = match s with Some s -> s.role | None -> "the model"

let diagnostic_of_conflict st out (c : Dim_solve.conflict) : diagnostic =
  let info = Hashtbl.find st.info c.culprit.origin.site in
  let subject = Option.bind info.site (fun (s : Dim_sites.site) -> s.subject) in
  (* The other positions the culprit was reduced against: where the
     conflicting requirement comes from. *)
  let notes =
    List.filter_map (fun (o : Dim_solve.origin) ->
      match Hashtbl.find_opt st.info o.site with
      (* a use that requires a dimension, not a definition the value flows
         through (a binding, a projection) *)
      | Some { site = Some s; kind = (Operands _ | Function_arg _ | Even_root _
                                     | Obligation_of _); _ }
        when Some s <> info.site -> Some s.Dim_sites.role
      | Some _ | None -> None) c.context
    |> List.sort_uniq compare
    |> (fun l -> List.filteri (fun i _ -> i < 3) l)
  in
  let with_notes d =
    match notes, d with
    | [], d -> d
    | ns, d ->
      let n = "the conflicting requirement comes from: " ^ String.concat "; " ns in
      Some (match d with Some d -> d ^ "\n  " ^ n | None -> n)
  in
  let integrality = match c.kind with
    | Dim_solve.Not_integral ->
      Some "no integer dimension satisfies it: the model requires a fractional \
            exponent (a square root or power of a dimensioned quantity)"
    | Contradiction -> None in
  let mk ?hint code message detail =
    { severity = Error; code; message;
      detail = with_notes (match integrality with Some i -> Some i | None -> detail);
      hint; subject } in
  match info.kind with
  | Obligation_of { expected; got } ->
    let s = Option.get info.site in
    let got_s = show out got in
    (match info.code, s.subject, s.expr with
     | "E300", Some (STransition n), Dim_sites.E rate ->
       mk "E300" (Printf.sprintf "transition '%s' rate has wrong dimension" n)
         (Some (Printf.sprintf "rate = %s\n  expected dimension: %s\n  got dimension: %s"
                  (expr_to_short_string rate) (dim_display expected) got_s))
     | code, _, _ ->
       let message =
         if dim_eq expected dimensionless then Printf.sprintf "%s must be dimensionless" s.role
         else Printf.sprintf "%s must have dimension %s" s.role (dim_display expected) in
       mk ?hint:s.note code message
         (Some (Printf.sprintf "expected dimension %s, got %s" (dim_display expected) got_s)))
  | Operands { op; left; right } ->
    let hint = match form_value out left, form_value out right with
      | Some a, Some b -> count_hint a b | _ -> None in
    mk ?hint info.code (Printf.sprintf "dimension mismatch: %s" op)
      (Some (Printf.sprintf "left has dimension %s, right has dimension %s — cannot combine"
               (show out left) (show out right)))
  | Function_arg { fn; arg } ->
    mk info.code (Printf.sprintf "%s must be dimensionless" fn)
      (Some (Printf.sprintf "it has dimension %s" (show out arg)))
  | Even_root f ->
    mk info.code (Printf.sprintf "sqrt in %s requires even dimension exponents" (site_name info.site))
      (Some (Printf.sprintf "argument has dimension %s" (show out f)))
  | Definition { got } ->
    mk info.code (Printf.sprintf "%s has conflicting dimensions" (site_name info.site))
      (Some (Printf.sprintf "it has dimension %s here" (show out got)))
  | Quantity_reduction ->
    mk info.code "a quantity's reduction has conflicting dimensions" None
  | Declared_fact ->
    mk "E302" "two declarations give one name different dimensions" None

let render_relation names ((r : Dim_solve.relation), base) =
  let lhs = List.map (fun (v, c) ->
    let n = Option.value ~default:"?" (Hashtbl.find_opt names v) in
    if c = 1 then n else Printf.sprintf "%s^%d" n c) r.coeffs in
  match base with
  | None -> Printf.sprintf "%s is %s" (String.concat " · " lhs) (formal_dim r.equals)
  | Some b ->
    Printf.sprintf "the %s exponent of %s is %d" (if b = 0 then "P" else "T")
      (String.concat " · " lhs) r.equals.(0)

(* ── Main check ────────────────────────────────────────────────────────── *)

let check_model (m : model) : result =
  let st = { t_only = false; t_only_eqs = Hashtbl.create 16; next_var = 0; keys = Hashtbl.create 256; declared = []; structure = [];
             obligation = []; info = Hashtbl.create 1024; next_eq = 0;
             tf_dims = Hashtbl.create 16 } in
  declare_model st m;
  declare_quantities st m;
  let sites = Dim_sites.sites m in
  List.iter (add_site st) sites;
  let out = solve_all st
      (List.rev st.declared @ List.rev st.structure @ List.rev st.obligation) in
  let errors =
    out.conflicts
    |> List.map (diagnostic_of_conflict st out)
    (* one diagnostic per position and code *)
    |> List.fold_left (fun acc (d : diagnostic) ->
         if List.exists (fun (x : diagnostic) -> x.code = d.code && x.subject = d.subject
                                  && x.message = d.message) acc then acc else d :: acc) []
    |> List.rev
  in
  (* Parameters and typed constants. *)
  let named = List.map (fun (p : parameter) -> p.name) m.parameters
              @ List.map (fun (c : constant) -> c.c_name) m.constants in
  let names = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.replace names (var_of st (Dim_sites.K_param n)) n) named;
  let resolved, free =
    List.partition_map (fun n ->
      let v = var_of st (Dim_sites.K_param n) in
      match out.value v with
      | Some d -> Left (n, d)
      | None -> Right (n, v)) named in
  let relations = out.relations (List.map snd free) in
  let infos =
    List.map (fun (n, v) ->
      let mine = List.filter (fun ((r : Dim_solve.relation), _) -> List.mem_assoc v r.coeffs) relations in
      { severity = Info; code = "I300";
        message = Printf.sprintf "dimension of parameter '%s' could not be determined" n;
        detail = (match mine with
            | [] -> Some "nothing in the model determines it"
            | rs -> Some ("only this is determined: "
                          ^ String.concat "; " (List.map (render_relation names) rs)));
        hint = Some "give it a dimension with a `dim` clause (`positive dim 1/T`) or a \
                     more specific kind (rate, probability, count)";
        subject = None }) free in
  let quantity_dims =
    List.filter_map (fun (q : quantity) ->
      match out.value (var_of st (Dim_sites.K_quantity q.q_name)) with
      | Some d -> Some ((q.q_name, q.q_stratum), d)
      | None -> None) m.quantities in
  { diagnostics = errors @ infos; param_dims = resolved; quantity_dims }
