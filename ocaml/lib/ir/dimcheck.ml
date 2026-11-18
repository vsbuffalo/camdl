(** The dimension checker of an expanded model: it infers the physical
    dimension of every expression and parameter, and reports each place the
    model's dimensions disagree.

    {2 Dimensions as exponent vectors}

    A dimension is a product [Pᵃ·Tᵇ] of the two base dimensions, population
    (P) and time (T), written as its vector of integer exponents [(a, b)]: a
    count is [(1, 0)], a duration [(0, 1)], a per-capita rate [(0, −1)], a
    transition's rate in individuals per unit time [(1, −1)], and a pure
    number [(0, 0)]. Multiplying quantities adds their vectors, so the
    dimensions form a free abelian group on the bases and every rule of
    inference is linear in the exponents: a product's dimension is the sum of
    its factors', the operands of a sum have one dimension, and a power [a^n]
    with [n] fixed at compile time has [n] times its base's (A. Kennedy,
    "Dimension types", ESOP 1994, LNCS 788, pp. 348–362). The language
    specification, section 2.2, states the rules the user sees; this module
    implements them.

    {2 One system of equations for the whole model}

    The checker treats the model as one system of linear equations and
    solves it at once, so a dimension that follows only from several uses
    together is still found. Its unknowns are dimension variables: one for
    each name whose dimension other positions refer to ({!Dim_sites.key}: a
    parameter or typed constant, a table's cells, a binding, an observation
    stream's projection, an observation column, a generated quantity and the
    series it reduces), and one fresh variable for each intermediate result
    a rule cannot express otherwise (a square root, a fractional power). The
    dimension of an expression is a {!form}, [Σ cᵥ·xᵥ + k]: integer
    coefficients on variables plus a constant vector. {!gen} computes it by
    one walk of the expression and emits an equation wherever two forms must
    agree:

    - a compartment or sum of compartments is P; [t], [dt] and the anchors
      [first_obs] and [last_obs] are T; a unit literal and an [unchecked_dim]
      have the dimension they state; a forcing has its declared dimension; a
      spline basis function is a pure number;
    - the operands of [+], [-], [min], [max], [%] and the comparisons are
      equal (E302), and a comparison is a pure number;
    - [*] adds forms and [/] subtracts them;
    - the argument of [exp], [log], [sin], [cos] and [tanh] is a pure number
      (E301), and so is their result; [abs], [floor], [ceil] and negation
      keep their argument's dimension;
    - [sqrt a] is a fresh [x] with [2·x = a] (E304);
    - in [a ^ e] the exponent is a pure number (E301). When [e] is fixed at
      compile time (literals, typed constants and arithmetic of them) and is
      [p/q] in lowest terms with [q ≤ 12], the result [x] satisfies
      [q·x = p·a]: an integer power scales [a], a fraction needs a fresh
      variable (E304). Any other exponent makes the base a pure number
      (E301);
    - the two branches of a conditional are equal; its condition is walked
      but constrained to nothing;
    - an index of a table lookup is a pure number (E311).

    Literals follow the position, not the neighbours. A literal [0] matches
    any dimension (the {!value} [Any]), and so do a product with a zero
    factor and a quotient with a zero numerator. Any other literal is a pure number,
    except that at a position whose {!Dim_sites.literals} is [Whole_terms]
    (a rate, an initial value, a count) a literal, or a constant expression
    of literals, that forms a whole additive term takes the position's
    dimension: [I = 10] is a count.

    The equations come from three sources, which {!Dim_solve} distinguishes
    as stages. Declared facts come first: a parameter's [dim] clause or else
    its kind (rate, probability, count, instant, duration), the [dim] clause
    or kind of a typed constant, a table's cell kind or unit annotation, the
    dimension of a projection that counts individuals (P) or is a ratio of
    flows (a pure number), and the kind of an observation's value columns.
    The structure of each expression comes next, together with the
    definition of every binding, derived projection and quantity and the
    reduction rule of each quantity. The obligations of the positions
    ({!Dim_sites.obligation}) come last: a transition rate must be [P·T⁻¹], an
    initial value P, and so on. The solver refuses the later of two
    equations that disagree, so a mistake is normally reported at the
    position whose requirement it breaks, with that position's code: a
    per-capita rate written as a transition rate is E300 at the transition,
    not a complaint about the parameter.

    {2 What is reported}

    Each refused equation becomes one error, worded from its {!eq_kind} and
    naming up to three other positions whose requirements it conflicts with.
    A bare number added to, compared with or taken the [min] of a
    dimensioned operand gets a hint naming the two ways to give the number a
    dimension: a typed constant or a unit literal. After the solve, a
    parameter or typed constant whose dimension remains free, and which some
    model position reads, is an I300 info listing what the model does fix
    about it. In the model

    {@camdl[
      time_unit = 'days
      compartments { S, I, R }
      parameters {
        a     : positive
        b     : positive
        gamma : rate
      }
      transitions {
        infection : S --> I @ a * b * S * I / (S + I + R)
        recovery  : I --> R @ gamma * I
      }
      init { S = 990  I = 10 }
    ]}

    neither [a] nor [b] is determined, and each I300 reads
    [only this is determined: a · b is T^-1]. A system whose only solutions
    have fractional exponents, such as [(kappa * w)^2 * I] as a rate, is an
    error at the position whose equation could not be added.

    Priors are checked after the solve and contribute no equation: a prior
    is a distribution over the number a parameter takes in model units, and
    its family says nothing about the parameter's dimension. What is checked
    is each explicit unit literal in a prior's or a bound's argument, which
    is converted to model units and so is right only in the argument's
    conventional dimension ({!convention}); a disagreement is E316. A
    log-normal [mu] or [sigma] that is directly a dimensioned parameter is
    I316.

    {2 Count noise}

    The [sd] of a normal likelihood on a [count] column must match the
    column's T exponent, and its P exponent is free, because variance
    formulas for counts add a term in the count to one in its square. Its
    site has the obligation [Equals_t], and every equation generated inside
    it, except those requiring a function's argument to be a pure number, is
    marked as holding on T alone. When any such equation exists, the
    system is solved twice, once per base: P without the marked equations,
    T with all of them. Because the bases decouple, this gives the same
    answer as one solve when nothing is marked.

    {2 Entry point and guarantees}

    {!check_model} returns a {!result} and changes nothing in the model. An
    operator means the same thing at every position; positions differ only
    in their obligation. Every expression-bearing field of the IR is either a
    site of {!Dim_sites.sites} or deliberately exempt from it, which the
    build and a test enforce together. Exponent arithmetic is checked, here
    as in the solver: an exponent outside the native integer range is the
    single error E319, never a wrapped value. *)

open Ir

(* ── Dimensions ────────────────────────────────────────────────────────── *)

let n_bases = 2

type dim_vec = int array

let make p t = [| p; t |]

let dimensionless = make 0 0
let population    = make 1 0
let time_dim      = make 0 1

(* Exponent arithmetic is checked, as in the solver: an exponent that leaves
   the native integer range raises [Dim_solve.Overflow], reported as E319,
   never a wrapped value. *)
let ( +! ) = Dim_solve.checked_add
let ( *! ) = Dim_solve.checked_mul
let ( -! ) a b = a +! (b *! (-1))

let dim_mul a b = Array.init n_bases (fun i -> a.(i) +! b.(i))
let dim_div a b = Array.init n_bases (fun i -> a.(i) -! b.(i))
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
  | SParameter   of string
  | SQuantity    of string
  | SIntervention of string
  | SLet         of string
  | SForcing     of string
  | STable       of string
  | SBalance     of string

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  subject  : subject option;
}

type prior_source = {
  param  : string;
  family : string;
  args   : (string * expr) list;
}

type transition_source = {
  member  : string;
  rate    : string;
  lowered : string;
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
  (* A sum of several compartments is parenthesised like the other binary
     operations here, so it reads as one operand wherever it stands. *)
  | PopSum [ s ] -> s
  | PopSum ss -> "(" ^ String.concat " + " ss ^ ")"
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
  (* [value] is in the model's time unit, not the literal's, so the unit name
     would misread it; the number alone is what the expression holds. *)
  | UnitLiteral u -> expr_to_short_string (Const u.value)
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

(** The dimension of an expression as a linear form [Σ cᵥ·xᵥ + k] over
    dimension variables. [vars] pairs each variable with its coefficient,
    sorted by variable and with no zero coefficient; the operations on forms
    keep it so, and the type does not enforce it. *)
type form = { vars : (int * int) list; k : dim_vec }

(** The dimension an expression contributes. [Any] is that of the literal
    [0], and of a literal standing as a whole term where it takes the
    position's dimension: it agrees with every dimension and constrains
    nothing. [F] is an ordinary dimension. *)
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

let form_add a b = { vars = merge ( +! ) a.vars b.vars; k = dim_mul a.k b.k }
let form_sub a b = { vars = merge ( -! ) a.vars b.vars; k = dim_div a.k b.k }
let form_scale n a =
  if n = 0 then const_form dimensionless
  else { vars = List.map (fun (v, c) -> (v, n *! c)) a.vars; k = Array.map (( *! ) n) a.k }

(* ── Checker state ─────────────────────────────────────────────────────── *)

(** Why an equation was emitted: what a diagnostic needs to say about it if
    the solver refuses it. *)
type eq_kind =
  | Declared_fact   (** A declared dimension of a name. *)
  | Operands of { op : string; left : form; right : form;
                  literals : string option * string option }
  (** Two operands that must have one dimension. [op] describes the operator and position; [literals] holds the text of an operand written as a bare number. *)
  | Function_arg of { fn : string; arg : form }              (** A value that must be a pure number: the argument of a function, an exponent, a base under a non-constant exponent, a table index. *)
  | Even_root of form                                        (** A square root or fractional power, whose argument's exponents must divide. *)
  | Obligation_of of { expected : dim_vec; got : form }       (** A position's fixed required dimension. *)
  | Definition of { got : form }                              (** A position whose expression defines, or must equal, a name's dimension. *)
  | Quantity_reduction   (** The rule relating a reduced quantity to its series. *)

(** An emitted equation's description: the site it came from (none for a
    declared fact or a quantity's reduction), the code a refusal reports,
    and its kind. *)
type eq_info = { site : Dim_sites.site option; code : string; kind : eq_kind }

(** The checker's state while it generates equations for one model. Each
    equation is numbered in the order it is emitted; the number is the
    [site] field of its {!Dim_solve.origin} and the key of its description
    in [info]. *)
type state = {
  constants : (string, float) Hashtbl.t;   (** Typed constants' values, for exponents fixed at compile time. *)
  used : (string, unit) Hashtbl.t;   (** Parameters and constants read by some model position. *)
  mutable t_only : bool;   (** Whether the site being generated is an [Equals_t] site. *)
  t_only_eqs : (int, unit) Hashtbl.t;   (** The equations that hold on T alone. *)
  mutable next_var : int;
  keys     : (Dim_sites.key, int) Hashtbl.t;   (** The variable of each name. *)
  mutable declared  : Dim_solve.eqn list;   (** Each of the three stage lists is newest first. *)
  mutable structure : Dim_solve.eqn list;
  mutable obligation : Dim_solve.eqn list;
  info     : (int, eq_info) Hashtbl.t;
  mutable next_eq : int;
  tf_dims  : (string, dim_vec) Hashtbl.t;   (** Each forcing's declared dimension. *)
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
  (* A count-noise sd frees the P exponent of its sums, powers and
     roots; a function's argument stays dimensionless in every base, since
     the count-noise reasoning says nothing about exp or log of a count. *)
  let t_only = st.t_only && (match kind with Function_arg _ -> false | _ -> true) in
  if t_only then Hashtbl.replace st.t_only_eqs id ();
  let e = { Dim_solve.terms = f.vars; rhs = dim_div rhs f.k;
            origin = { site = id; stage; what = code } } in
  match stage with
  | Dim_solve.Declared -> st.declared <- e :: st.declared
  | Structure -> st.structure <- e :: st.structure
  | Obligation -> st.obligation <- e :: st.obligation

(* [a = b] as a structure equation; nothing when either is [Any], or when both
   are the same constant. *)
let equate st ~site ~code ~op ?(literals = (None, None)) a b =
  match a, b with
  | Any, _ | _, Any -> ()
  | F l, F r ->
    if l.vars = [] && r.vars = [] && dim_eq l.k r.k then ()
    else
      emit st Structure ~site ~code (Operands { op; left = l; right = r; literals })
        (form_sub l r) dimensionless

let declare st key v =
  emit st Declared ~site:None ~code:"E302" Declared_fact (var_form (var_of st key)) v

(* ── Declared facts ────────────────────────────────────────────────────── *)

let kind_dim : param_kind -> dim_vec option = function
  | Rate -> Some (make 0 (-1))
  | Probability -> Some dimensionless
  | Count -> Some population
  (* Both time kinds are [T]; the instant/duration distinction is for
     rendering, not checking. *)
  | Instant | Duration -> Some time_dim
  | Positive | Real -> None

let declare_model st ~constant_dims ~table_dims (m : model) =
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
    match List.assoc_opt c.c_name constant_dims with
    | Some (dp, dt) -> declare st key (make dp dt)
    | None -> Option.iter (declare st key) (kind_dim c.c_kind)) m.constants;
  List.iter (fun (t : table) ->
    (match Option.bind (Option.bind t.cell_kind param_kind_of_name) kind_dim with
     | Some v -> declare st (Dim_sites.K_table t.name) v
     | None -> ());
    (* A table's unit annotation (`mu : age 'per_day = [...]`) declares its
       cells' dimension; the IR keeps only the scaled values. *)
    match List.assoc_opt t.name table_dims with
    | Some (dp, dt) -> declare st (Dim_sites.K_table t.name) (make dp dt)
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

let rec chain_add = function
  | [] -> Const 0.0
  | [ t ] -> t
  | t :: rest -> BinOp { op = Add; left = t; right = chain_add rest }

(* [x] as p/q in lowest terms with a small denominator, when it is one: a
   compile-time exponent that is a simple fraction. *)
let rational (x : float) : (int * int) option =
  if not (Float.is_finite x) then None
  else
    let rec find q =
      if q > 12 then None
      else
        let p = Float.round (x *. float_of_int q) in
        if Float.abs (p /. float_of_int q -. x) < 1e-12 && Float.abs p < 1e6
        then Some (int_of_float p, q) else find (q + 1)
    in
    find 1

(* The binary-operator rules, shared by model expressions, derived quantities
   and contrasts. [exponent] is the power's exponent when it is a literal. *)
let binop st ~site ~code ~what (op : bin_op) ?exponent ?literals l r : value =
  let opname = match op with
    | Add -> "+" | Sub -> "-" | Min -> "min" | Max -> "max" | Mod -> "%"
    | Eq -> "==" | Neq -> "!=" | Lt -> "<" | Gt -> ">" | Le -> "<=" | Ge -> ">="
    | Mul -> "*" | Div -> "/" | Pow -> "^" in
  match op with
  | Add | Sub | Min | Max | Mod ->
    equate st ~site ~code ~op:(Printf.sprintf "%s in %s" opname what) ?literals l r;
    (match l with Any -> r | F _ -> l)
  | Eq | Neq | Lt | Gt | Le | Ge ->
    equate st ~site ~code ~op:(Printf.sprintf "%s in %s" opname what) ?literals l r;
    F (const_form dimensionless)
  | Mul ->
    (* A factor is never a whole term, so an [Any] factor is a zero (the
       literal, or [0 / x], ...) and the product is a zero of any dimension,
       as [0 / x] is (gh#1212). *)
    (match l, r with Any, _ | _, Any -> Any | F a, F b -> F (form_add a b))
  | Div ->
    (match l, r with
     | Any, _ -> Any
     | _, Any -> l
     | F a, F b -> F (form_sub a b))
  | Pow ->
    (* The exponent is a pure number. *)
    (match r with
     | F e when not (e.vars = [] && dim_eq e.k dimensionless) ->
       emit st Structure ~site ~code:"E301"
         (Function_arg { fn = Printf.sprintf "the exponent of ^ in %s" what; arg = e })
         e dimensionless
     | Any | F _ -> ());
    (* With an exponent known at compile time as p/q, q·dim(result) =
       p·dim(base): an integer power scales the base's dimension, and a
       rational one is infeasible unless the exponents divide. Otherwise the
       result would be d^y with y real and changing during a fit, not a
       dimension at all, so the base must be dimensionless. *)
    (match l with
     | Any -> Any
     | F b ->
       match Option.bind exponent rational with
       | Some (p, 1) -> F (form_scale p b)
       | Some (p, q) ->
         let x = fresh st in
         emit st Structure ~site ~code:"E304" (Even_root b)
           (form_sub (form_scale q (var_form x)) (form_scale p b)) dimensionless;
         F (var_form x)
       | None ->
         if not (b.vars = [] && dim_eq b.k dimensionless) then
           emit st Structure ~site ~code:"E301"
             (Function_arg { fn = Printf.sprintf "a base under a non-constant exponent in %s" what;
                             arg = b }) b dimensionless;
         F (const_form dimensionless))

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

(* The value of an expression fixed at compile time: literals, typed
   constants and arithmetic of them. *)
let rec static_value st (e : expr) : float option =
  let sv = static_value st in
  match e with
  | Const v -> Some v
  | ConstRef c -> Hashtbl.find_opt st.constants c
  | UnOp { op = Neg; arg } -> Option.map Float.neg (sv arg)
  | BinOp { op; left; right } ->
    (match sv left, sv right, op with
     | Some a, Some b, Add -> Some (a +. b)
     | Some a, Some b, Sub -> Some (a -. b)
     | Some a, Some b, Mul -> Some (a *. b)
     | Some a, Some b, Div -> Some (a /. b)
     | _ -> None)
  | _ -> None

(* The text of an operand written as a bare number, for the diagnostic. *)
let literal_text (e : expr) =
  if Dim_sites.is_constant_expr e then Some (expr_to_short_string e) else None

(* [term]: [e] is an entire additive term of the site's expression, where a
   literal takes the position's dimension under [Whole_terms]. *)
let rec gen st ~(site : Dim_sites.site) ?(term = false) (e : expr) : value =
  let what = site.role in
  let g = gen st ~site in
  match e with
  | Const 0.0 -> Any
  (* Arithmetic of literals is a number written as arithmetic: as a whole term
     it takes the position's dimension (`S = 0.95 * 10000`); as a factor it is
     dimensionless (`(1/730) * R` in a rate is not a rate). *)
  | _ when term && site.literals = Dim_sites.Whole_terms && Dim_sites.is_constant_expr e -> Any
  | Const _ ->
    (match site.literals with
     | Dim_sites.Whole_terms when term -> Any
     | Whole_terms | Literals_dimensionless -> F (const_form dimensionless))
  | BinOp { op = (Add | Sub) as op; left; right } ->
    let l = g ~term left and r = g ~term right in
    binop st ~site:(Some site) ~code:"E302" ~what op
      ~literals:(literal_text left, literal_text right) l r
  | UnOp { op = Neg; arg } -> g ~term arg
  | Param n | ConstRef n ->
    Hashtbl.replace st.used n ();
    F (var_form (var_of st (Dim_sites.K_param n)))
  | Pop _ | PopSum _ -> F (const_form population)
  | Time | Dt | ObsAnchor _ -> F (const_form time_dim)
  (* A basis function is a pure number (the basis sums to one). *)
  | ForcingBasis _ -> F (const_form dimensionless)
  | TimeFunc n ->
    (match Hashtbl.find_opt st.tf_dims n with
     | Some d -> F (const_form d)
     | None -> F (var_form (fresh st)))
  | TableLookup (n, idx) ->
    (* an index selects a level: a pure number (E311) *)
    List.iter (fun i ->
      match g i with
      | Any -> ()
      | F f ->
        if not (f.vars = [] && dim_eq f.k dimensionless) then
          emit st Structure ~site:(Some site) ~code:"E311"
            (Function_arg { fn = Printf.sprintf "an index of table '%s' in %s" n what; arg = f })
            f dimensionless) idx;
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
  | UnitLiteral u -> F (const_form (make u.dim_p u.dim_t))
  | BinOp b ->
    let l = g b.left and r = g b.right in
    let exponent = static_value st b.right in
    binop st ~site:(Some site) ~code:"E302" ~what b.op ?exponent
      ~literals:(literal_text b.left, literal_text b.right) l r
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
  | SConst 0.0 -> Any
  | SConst _ -> F (const_form dimensionless)
  | SParam p -> Hashtbl.replace st.used p (); F (var_form (var_of st (Dim_sites.K_param p)))
  | SQRef q -> F (var_form (var_of st (Dim_sites.K_quantity q.qref_name)))
  | SUnOp { op; arg } -> unop st ~site:(Some site) ~what:site.role op (g arg)
  | SBinOp { op; left; right } ->
    let exponent = match right with SConst n -> Some n | _ -> None in
    binop st ~site:(Some site) ~code:"E302" ~what:site.role op ?exponent (g left) (g right)
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
       binop st ~site:(Some site) ~code:"E297" ~what:site.role op l r
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

(** The solution of the model's system, whether it was solved once over both
    bases or once per base for count noise. In the per-base case a variable
    is determined only when both solves determine it, and the conflicts are
    those of both solves, each equation once, in the order the equations
    were emitted. *)
type solved = {
  value     : int -> dim_vec option;
  conflicts : Dim_solve.conflict list;
  relations : int list -> (Dim_solve.relation * int option) list;
  (** The relations among the given variables, each with [None] when it
      holds on both bases at once and [Some b] when it is a relation on base
      [b]'s exponents alone. *)
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
    | Some k, Some d -> Some (Array.mapi (fun i ki -> ki +! (c *! d.(i))) k)
    | _ -> None) (Some (Array.copy f.k)) f.vars

let show out f = match form_value out f with
  | Some d -> dim_display d
  | None -> "not determined by the model"

(* Why a bare number combined with a dimensioned operand is refused, and the
   two ways to say what the number is: a typed constant (a named quantity,
   written once) or a unit literal (a one-off value). *)
let literal_hint (v : string) (d : dim_vec) =
  let kind, unit_ = match d.(0), d.(1) with
    | 1, 0 -> ("count", Some "'count")
    | 0, 1 -> ("duration", Some "'days")
    | 0, -1 -> ("rate", Some "'per_day")
    | _ -> (Printf.sprintf "positive dim %s" (formal_dim d), None) in
  let named = Printf.sprintf "let <name> : %s = %s" kind v in
  let pad s = s ^ String.make (String.length named + 3 - String.length s) ' ' in
  Printf.sprintf
    "a bare number in camdl gets a dimension only from where it stands: as a \
     whole term of a rate, an initial value or a count it takes that \
     position's dimension (`I = 10`, `beta * S * I / N + 0.5`); inside a \
     product, a quotient or a function it is a pure number. The compiler does \
     not guess what `%s` measures from what it is added to, because the same \
     guess would silently accept a constant in the wrong place. Say what `%s` \
     is:\n        %sa named quantity, written once; then use <name>%s"
    v v (pad named)
    (match unit_ with
     | Some u -> Printf.sprintf "\n        %sa one-off value" (pad (v ^ " " ^ u))
     | None -> "")

let site_name (s : Dim_sites.site option) = match s with Some s -> s.role | None -> "the model"

let diagnostic_of_conflict ~sources st out (c : Dim_solve.conflict) : diagnostic =
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
       let shown, lowered = match Hashtbl.find_opt sources n with
         | Some (src : transition_source) -> (src.rate, "\n  " ^ src.lowered)
         | None -> (expr_to_short_string rate, "") in
       mk "E300" (Printf.sprintf "transition '%s' rate has wrong dimension" n)
         (Some (Printf.sprintf "rate = %s\n  expected dimension: %s\n  got dimension: %s%s"
                  shown (dim_display expected) got_s lowered))
     | code, _, _ ->
       let message =
         if dim_eq expected dimensionless then Printf.sprintf "%s must be dimensionless" s.role
         else Printf.sprintf "%s must have dimension %s" s.role (dim_display expected) in
       mk ?hint:s.note code message
         (Some (Printf.sprintf "expected dimension %s, got %s" (dim_display expected) got_s)))
  | Operands { op; left; right; literals } ->
    let lv = form_value out left and rv = form_value out right in
    let literal_case = match literals, lv, rv with
      | (Some v, None), _, Some d | (None, Some v), Some d, _
        when not (dim_eq d dimensionless) -> Some (v, d)
      | _ -> None in
    (match literal_case with
     | Some (v, d) ->
       mk ~hint:(literal_hint v d) info.code (Printf.sprintf "dimension mismatch: %s" op)
         (Some (Printf.sprintf
                  "`%s` is a bare number here, so a pure number, and the other operand \
                   has dimension %s; camdl cannot tell what `%s` measures"
                  v (dim_display d) v))
     | None ->
       mk info.code (Printf.sprintf "dimension mismatch: %s" op)
         (Some (Printf.sprintf "left has dimension %s, right has dimension %s — cannot combine"
                  (show out left) (show out right))))
  | Function_arg { fn; arg } ->
    mk info.code (Printf.sprintf "%s must be dimensionless" fn)
      (Some (Printf.sprintf "it has dimension %s" (show out arg)))
  | Even_root f ->
    mk info.code
      (Printf.sprintf "a square root or fractional power in %s needs dimension exponents it divides"
         (site_name info.site))
      (Some (Printf.sprintf "its argument has dimension %s" (show out f)))
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

(* ── Priors ────────────────────────────────────────────────────────────── *)

(* A prior is a measure on the number θ takes in model units; its family
   neither establishes nor restricts θ's dimension, and nothing here emits an
   equation. What is checked is an explicit unit literal: it is converted to
   model units, and that conversion is right exactly when the literal's
   dimension is the argument's conventional one. *)

(** What the unit literals in a term of a prior's argument assert about its
    dimension. Inside a prior argument a parameter is a number in model
    units (a coordinate), so arithmetic on it is free and asserts nothing.

    {@camdl[
      parameters {
        gamma : rate ~ normal(mu = 0.2 'per_day, sigma = 0.05 'per_day)   # Units
        rho   : probability ~ beta(alpha = 2, beta = 2)                    # Bare
      }
    ]} *)
type prior_value =
  | Bare                (** Bare numbers only: no claim about the dimension. *)
  | Units of dim_vec    (** A dimension fixed by unit literals alone. *)
  | Coord               (** A value that depends on a parameter, a function or another runtime value: no claim. *)

(* [report a b]: explicit units [a] and [b] disagree inside one argument. *)
let rec prior_value ~report (e : expr) : prior_value =
  let pv = prior_value ~report in
  match e with
  | Const _ -> Bare
  | UnitLiteral { value = _; dim_p; dim_t; unit = _ } -> Units (make dim_p dim_t)
  | BinOp { op = (Add | Sub | Min | Max); left; right } ->
    (match pv left, pv right with
     | Units a, Units b -> if dim_eq a b then Units a else (report a b; Coord)
     | Bare, x | x, Bare -> x
     | Coord, _ | _, Coord -> Coord)
  | BinOp { op = (Mul | Div) as op; left; right } ->
    let comb = if op = Mul then dim_mul else dim_div in
    (match pv left, pv right with
     | Units a, Units b -> Units (comb a b)
     | Units a, Bare -> Units a
     | Bare, Units b -> Units (comb dimensionless b)
     | Bare, Bare -> Bare
     | Coord, _ | _, Coord -> Coord)
  | BinOp { op = Pow; left; right = Const n } when Float.is_integer n ->
    (match pv left with
     | Units a -> Units (Array.map (( *! ) (Float.to_int n)) a)
     | x -> x)
  | UnOp { op = Neg; arg } -> pv arg
  | BinOp { op = _; left; right } -> ignore (pv left); ignore (pv right); Coord
  | UnOp { op = _; arg } -> ignore (pv arg); Coord
  | Cond { pred; then_; else_ } -> ignore (pv pred); ignore (pv then_); ignore (pv else_); Coord
  | Reduce ts -> List.iter (fun t -> ignore (pv t)) ts; Coord
  | UncheckedDim _ | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | TableLookup _ | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ | ObsAnchor _
  | ForcingBasis _ -> Coord

(** The dimension a prior argument has in a unit-aware reading of its family,
    relative to the parameter θ it is a prior on: [One] a pure number,
    [Theta] θ's dimension, [Inv_theta] its inverse. A normal's [mu] is
    [Theta], an exponential's [rate] [Inv_theta], and a log-normal's [mu]
    and [sigma], the mean and standard deviation of ln θ, are [One]. An
    argument of a family the function {!convention} does not list has no
    convention, and its unit literals are checked only against each other. *)
type convention = One | Theta | Inv_theta

let convention family arg =
  match family, arg with
  | ("uniform" | "log_uniform" | "bounds"), ("lower" | "upper") -> Some Theta
  | "normal", ("mu" | "sigma") -> Some Theta
  | "truncated_normal", ("mean" | "sd") -> Some Theta
  | "half_normal", "sigma" -> Some Theta
  | "exponential", "rate" -> Some Inv_theta
  | "gamma", "shape" -> Some One
  | "gamma", "rate" -> Some Inv_theta
  | "beta", ("alpha" | "beta") -> Some One
  | "log_normal", ("mu" | "sigma") -> Some One
  | _ -> None

let check_priors (out : solved) st (priors : prior_source list) : diagnostic list =
  let theta_dim p = out.value (var_of st (Dim_sites.K_param p)) in
  List.concat_map (fun { param; family; args } ->
    let subject = Some (SParameter param) in
    List.concat_map (fun (arg, e) ->
      let inner = ref [] in
      let report a b =
        inner := { severity = Error; code = "E316"; subject; hint = None;
                   message = (if family = "bounds" then
                     Printf.sprintf "the `%s` bound of '%s': explicit units disagree" arg param
                   else Printf.sprintf
                     "prior of '%s': explicit units disagree inside `%s`" param arg);
                   detail = Some (Printf.sprintf "%s is combined with %s"
                                    (dim_display a) (dim_display b)) } :: !inner in
      (* Each whole additive term of the argument is read on its own: a unit
         literal term asserts a quantity in the argument's units whatever the
         other terms are, so `m_h + 1 'days` on a rate's `mu` is checked even
         though `m_h` is a coordinate. Terms of explicit units that disagree
         are refused whatever θ is. *)
      let rec terms e = match[@warning "-4"] e with
        | BinOp { op = (Add | Sub); left; right } -> terms left @ terms right
        | UnOp { op = Neg; arg } -> terms arg
        | e -> [ e ] in
      let values = List.map (prior_value ~report) (terms e) in
      let units = List.filter_map (function Units d -> Some d | Bare | Coord -> None) values in
      (match units with
       | a :: rest ->
         (match List.find_opt (fun b -> not (dim_eq a b)) rest with
          | Some b when not (List.exists (fun (d : diagnostic) -> d.subject = subject) !inner) ->
            report a b
          | Some _ | None -> ())
       | [] -> ());
      let v = match units with d :: _ -> Units d | [] -> Bare in
      let top = match v, convention family arg with
        | Units d, Some conv ->
          let expected = match conv with
            | One -> Some dimensionless
            | Theta -> theta_dim param
            | Inv_theta -> Option.map (dim_div dimensionless) (theta_dim param) in
          (match expected with
           | Some x when not (dim_eq x d) ->
             let log_scale = family = "log_normal" in
             [ { severity = Error; code = "E316"; subject;
                 message = (if family = "bounds" then
                   Printf.sprintf "the `%s` bound of '%s' has a unit literal of dimension %s, but '%s' is %s"
                     arg param (dim_display d) param (dim_display x)
                 else Printf.sprintf
                   "prior of '%s': `%s` has a unit literal of dimension %s, but a %s prior's `%s` is %s" param arg (dim_display d) family arg
                   (dim_display x));
                 detail = Some "a unit literal is converted to the model's units, and that conversion is right only in the argument's own dimension";
                 hint = Some (if log_scale then
                   "`mu` and `sigma` of a log-normal are the mean and sd of ln θ, numbers in model units. To give a median with a unit, write `mu = log(0.2 'per_day)`, which converts the value before taking its log; `sigma` is a plain number."
                 else "write the argument in the parameter's own units, or as a bare number in model units") } ]
           | Some _ | None -> [])
        | (Units _ | Bare | Coord), _ -> []
      in
      (* One lint: a log-normal argument that is directly a parameter with a
         dimension is probably a rate used where a log-coordinate belongs. *)
      let lint = match family, arg, e with
        | "log_normal", ("mu" | "sigma"), Param h ->
          (match theta_dim h with
           | Some d when not (dim_eq d dimensionless) ->
             [ { severity = Info; code = "I316"; subject;
                 message = Printf.sprintf
                   "log-normal `%s` of '%s' is the parameter '%s', which has dimension %s"
                   arg param h (dim_display d);
                 detail = Some (Printf.sprintf
                   "`%s` is a log-coordinate in model units: %s"
                   arg (if arg = "mu"
                        then Printf.sprintf "the prior's median is exp(%s) in %s's units" h param
                        else "the sd of ln θ, a pure number"));
                 hint = Some (Printf.sprintf
                   "if '%s' is meant as a median %s, write `%s = log(%s)`; if it is meant as a log-coordinate, declare it `real dim 1`"
                   h (display_dim d) arg h) } ]
           | Some _ | None -> [])
        | _ -> []
      in
      List.rev !inner @ top @ lint) args) priors

(* ── Main check ────────────────────────────────────────────────────────── *)

(* [s] with every occurrence of [sub] replaced by [by]. *)
let replace_all ~sub ~by s =
  let n = String.length sub and b = Buffer.create (String.length s) in
  let rec go i =
    if i > String.length s - n then Buffer.add_string b (String.sub s i (String.length s - i))
    else if String.sub s i n = sub then (Buffer.add_string b by; go (i + n))
    else (Buffer.add_char b s.[i]; go (i + 1)) in
  if n = 0 then s else (go 0; Buffer.contents b)

(* A site of a transition the compiler made from a written one, reported as
   the written one: its subject and role name the member as written, so the
   copies of one member give one diagnostic. *)
let as_written (sources : (string, transition_source) Hashtbl.t) (s : Dim_sites.site) =
  match s.subject with
  | Some (STransition n) ->
    (match Hashtbl.find_opt sources n with
     | Some src ->
       { s with subject = Some (STransition src.member);
                role = replace_all ~sub:(Printf.sprintf "'%s'" n)
                    ~by:(Printf.sprintf "'%s'" src.member) s.role }
     | None -> s)
  | Some (SOde _ | SObservation _ | SBinding _ | SContrast _ | SInit _ | SParameter _
         | SQuantity _ | SIntervention _ | SLet _ | SForcing _ | STable _ | SBalance _)
  | None -> s

let check_model_exn ?prior_args ~constant_dims ~table_dims ~typed_lets
    ~transition_sources (m : model) : result =
  let constants = Hashtbl.create 16 in
  List.iter (fun (c : constant) -> Hashtbl.replace constants c.c_name c.c_value) m.constants;
  let st = { constants; used = Hashtbl.create 64; t_only = false; t_only_eqs = Hashtbl.create 16; next_var = 0; keys = Hashtbl.create 256; declared = []; structure = [];
             obligation = []; info = Hashtbl.create 1024; next_eq = 0;
             tf_dims = Hashtbl.create 16 } in
  declare_model st ~constant_dims ~table_dims m;
  declare_quantities st m;
  let by_name = Hashtbl.create 64 in
  List.iter (fun (n, src) -> Hashtbl.replace by_name n src) transition_sources;
  let sites = List.map (as_written by_name) (Dim_sites.sites m) in
  let sources = Hashtbl.create 16 in
  List.iter (fun (_, (src : transition_source)) -> Hashtbl.replace sources src.member src)
    transition_sources;
  (* A typed `let`'s annotation, at each of its expanded bodies. *)
  let let_sites = List.map (fun (name, (p, t), e) ->
    { Dim_sites.subject = Some (SLet name);
      role = Printf.sprintf "let '%s'" name; expr = Dim_sites.E e;
      obligation = Has (make p t); literals = Whole_terms; code = "E318";
      strength = Binding; stream = None;
      note = Some "the `let` is declared with a kind or `dim` clause, and its \
                   body must have that dimension; fix the body, or the \
                   annotation if it was mistaken" }) typed_lets in
  List.iter (add_site st) (sites @ let_sites);
  let out = solve_all st
      (List.rev st.declared @ List.rev st.structure @ List.rev st.obligation) in
  let errors =
    out.conflicts
    |> List.map (diagnostic_of_conflict ~sources st out)
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
  (* A parameter that no model position reads (one used only in priors, or not
     at all) has no dimension that matters to anything. *)
  let free = List.filter (fun (n, _) -> Hashtbl.mem st.used n) free in
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
  let priors = match prior_args with
    | Some ps -> ps
    | None ->
      List.filter_map (fun (p : parameter) ->
        match p.value with
        | Estimated { est_prior = Hierarchical h; _ } ->
          Some { param = p.name; family = hierarchical_kind_name h.hkind; args = h.hargs }
        | Estimated _ | Fixed _ | Required -> None) m.parameters in
  let prior_diags = check_priors out st priors in
  { diagnostics = errors @ prior_diags @ infos; param_dims = resolved; quantity_dims }

(* An exponent too large for a native integer (a power of a power of a power,
   ...) is reported, never wrapped: the guarantee the solver keeps holds through
   equation generation too. *)
let check_model ?prior_args ?(constant_dims = []) ?(table_dims = []) ?(typed_lets = [])
    ?(transition_sources = []) (m : model) : result =
  try check_model_exn ?prior_args ~constant_dims ~table_dims ~typed_lets ~transition_sources m
  with Dim_solve.Overflow msg ->
    { diagnostics = [ { severity = Error; code = "E319"; subject = None;
                        message = "a dimension exponent in this model is too large to represent";
                        detail = Some msg;
                        hint = Some "a power of a power multiplies exponents; write the \
                                     exponent as one number, or wrap a deliberate term in \
                                     `unchecked_dim(...)`" } ];
      param_dims = []; quantity_dims = [] }
