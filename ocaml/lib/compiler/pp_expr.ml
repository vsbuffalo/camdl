(** A printer of IR expressions for people: infix arithmetic with the fewest
    parentheses that keep the tree, each name coloured by its kind, and, on
    request, compartments shown with their indices.

    {!pp} prints an {!Ir.expr} to a [Format] formatter. It parenthesises a
    subexpression only where precedence requires, treating [+] and [-] as
    left-associative and [^] as right-associative, and colours names
    through {!Term_style}. [~ascii:true] writes [*] for multiplication
    rather than [×]. Under [Dsl], a compartment name the {!split_fn}
    recognises is printed with its levels, [S[child]] for [S_child]; the
    function comes from {!make_split_map}, which enumerates every
    expansion of each base compartment. Only compartments are split:
    parameters and bindings keep their expanded names ([N_a]), and a table
    lookup shows its flattened index ([W[3]] for [W[b, a]]).
    {!to_string} prints in [Ir] mode with ASCII and no colour, for the
    expander's diagnostics; [camdlc inspect] prints in [Dsl] mode.

    The output reads like camdl but is not always camdl that compiles.
    [min], [max] and [mod] are written infix ([a min b]), where camdl
    writes [min(a, b)]; a sum is a parenthesised chain of [+]; a number
    that is not an integer is printed with [%g], six significant digits,
    so [0.123456789] prints as [0.123457]; a unit literal prints as its
    value in the model's time unit, without its unit. An {!Ir.PopSum} of
    more than one compartment binds as a sum, so the rate
    [beta * S * I / (S + I + R)] keeps its parentheses. *)

(** Which names {!pp} prints: [Ir] prints every name as the compiled model
    holds it ([S_child]); [Dsl] prints a compartment name as the model's
    author wrote it ([S[child]]), using a {!split_fn}. *)
type name_mode = Dsl | Ir

(** A way to recover the author's spelling of a compiled compartment name:
    the base name and its levels, in dimension order, such as
    [Some ("S", ["child"])] for ["S_child"], and [None] for a name it does
    not know. {!no_split} knows none. *)
type split_fn = string -> (string * string list) option

let no_split : split_fn = fun _ -> None

(* ── Precedence (higher = tighter binding) ──────────────────────────────── *)

let prec_binop = function
  | Ir.Add | Ir.Sub -> 6
  | Ir.Mul | Ir.Div | Ir.Mod -> 7
  | Ir.Pow          -> 8
  | Ir.Min | Ir.Max -> 5
  | Ir.Eq | Ir.Neq | Ir.Lt | Ir.Gt | Ir.Le | Ir.Ge -> 4

let prec_expr : Ir.expr -> int = function
  | Ir.PopSum (_ :: _ :: _) -> prec_binop Ir.Add   (* printed as a chain of [+] *)
  | Ir.Const _ | Ir.UnitLiteral _ | Ir.Param _ | Ir.ConstRef _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.TimeFunc _ | Ir.ForcingBasis _ | Ir.TableLookup _ | Ir.Projected
  | Ir.ObsColumnRef _ -> 10
  | Ir.ObsAnchor _ -> 10      (* a name (± a number), atomic *)
  | Ir.UncheckedDim _ -> 10   (* function-call-like, atomic *)
  | Ir.Reduce _ -> 10         (* rendered self-parenthesized, atomic *)
  | Ir.BindingRef _ -> 10     (* a name, atomic *)
  | Ir.PerEvalRef _ -> 10     (* a name, atomic *)
  | Ir.BinOp { op; _ } -> prec_binop op
  | Ir.UnOp  _         -> 9
  | Ir.Cond  _         -> 1

let op_str ~ascii = function
  | Ir.Add -> "+"
  | Ir.Sub -> "-"
  | Ir.Mul -> if ascii then "*" else "\xc3\x97"  (* × U+00D7 *)
  | Ir.Div -> "/"
  | Ir.Pow -> "^"
  | Ir.Mod -> "mod"
  | Ir.Min -> "min"
  | Ir.Max -> "max"
  | Ir.Eq  -> "==" | Ir.Neq -> "!=" | Ir.Lt -> "<"
  | Ir.Gt  -> ">"  | Ir.Le  -> "<=" | Ir.Ge -> ">="

let un_op_str = function
  | Ir.Neg   -> "-"
  | Ir.Exp   -> "exp"
  | Ir.Log   -> "log"
  | Ir.Sqrt  -> "sqrt"
  | Ir.Abs   -> "abs"
  | Ir.Floor -> "floor"
  | Ir.Ceil  -> "ceil"
  | Ir.Sin   -> "sin"
  | Ir.Cos   -> "cos"
  | Ir.Tanh  -> "tanh"

(* ── Pop name rendering ──────────────────────────────────────────────────── *)

let pp_pop ~mode ~(split : split_fn) ppf name =
  match mode with
  | Ir -> Term_style.compartment Fmt.string ppf name
  | Dsl ->
    match split name with
    | None -> Term_style.compartment Fmt.string ppf name
    | Some (base, []) ->
      Term_style.compartment Fmt.string ppf base
    | Some (base, idxs) ->
      Term_style.compartment Fmt.string ppf base;
      Term_style.dim_style (fun ppf () ->
        Fmt.pf ppf "[%s]" (String.concat ", " idxs)
      ) ppf ()

(* ── Main printer ────────────────────────────────────────────────────────── *)

let rec pp ~mode ~(split : split_fn) ~ascii ppf e =
  pp_at ~mode ~split ~ascii 0 ppf e

and pp_at ~mode ~split ~ascii min_prec ppf e =
  let p = prec_expr e in
  if p < min_prec then (
    Fmt.pf ppf "(";
    pp_inner ~mode ~split ~ascii ppf e;
    Fmt.pf ppf ")"
  ) else
    pp_inner ~mode ~split ~ascii ppf e

and pp_inner ~mode ~split ~ascii ppf = function
  | Ir.Const f ->
    (* Integer-valued floats up to 1e15 print as bare integers (no
       decimal point or scientific-notation tail); outside that range,
       fall back to %g. *)
    if Float.is_integer f && Float.abs f < 1e15 then
      Fmt.pf ppf "%.0f" f
    else
      Fmt.pf ppf "%g" f
  | Ir.Param name | Ir.ConstRef name ->
    Term_style.param Fmt.string ppf name
  | Ir.Pop name ->
    pp_pop ~mode ~split ppf name
  | Ir.PopSum names ->
    List.iteri (fun i name ->
      if i > 0 then Fmt.pf ppf " + ";
      pp_pop ~mode ~split ppf name
    ) names
  | Ir.Time ->
    Term_style.param Fmt.string ppf "t"
  | Ir.Dt ->
    Term_style.param Fmt.string ppf "dt"
  | Ir.TimeFunc name ->
    Term_style.table Fmt.string ppf name
  (* A derivative-only node: B_k of forcing f, written `basis(f, k)`. *)
  | Ir.ForcingBasis { forcing; index; space } ->
    let fn = match space with
      | Ir.Space_spline -> "basis" | Ir.Space_whitened -> "whitened_basis" | Ir.Space_linear -> "linear_basis" in
    Fmt.pf ppf "%s(%a, %d)" fn (Term_style.table Fmt.string) forcing index
  | Ir.Projected ->
    Fmt.pf ppf "projected"
  | Ir.ObsColumnRef c ->
    Term_style.param Fmt.string ppf c
  | Ir.TableLookup (name, idxs) ->
    Term_style.table Fmt.string ppf name;
    Term_style.dim_style (fun ppf () ->
      Fmt.pf ppf "[";
      List.iteri (fun i idx ->
        if i > 0 then Fmt.pf ppf ", ";
        pp ~mode ~split ~ascii ppf idx
      ) idxs;
      Fmt.pf ppf "]"
    ) ppf ()
  | Ir.BinOp { op; left; right } ->
    let op_p = prec_binop op in
    (* Left-associative by default; Pow is right-associative *)
    let (left_min, right_min) = match op with
      | Ir.Pow -> (op_p + 1, op_p)        (* right-assoc *)
      | _      -> (op_p,     op_p + 1)    (* left-assoc *)
    in
    pp_at ~mode ~split ~ascii left_min ppf left;
    (match op with
     | Ir.Min -> Fmt.pf ppf " min "
     | Ir.Max -> Fmt.pf ppf " max "
     | _ ->
       Fmt.pf ppf " ";
       Term_style.dim_style Fmt.string ppf (op_str ~ascii op);
       Fmt.pf ppf " ");
    pp_at ~mode ~split ~ascii right_min ppf right
  | Ir.UnOp { op; arg } ->
    (match op with
     | Ir.Neg ->
       Term_style.dim_style Fmt.string ppf "-";
       pp_at ~mode ~split ~ascii 10 ppf arg
     | _ ->
       Fmt.pf ppf "%s(" (un_op_str op);
       pp ~mode ~split ~ascii ppf arg;
       Fmt.pf ppf ")")
  | Ir.Cond { pred; then_; else_ } ->
    Fmt.pf ppf "if ";
    pp ~mode ~split ~ascii ppf pred;
    Fmt.pf ppf " then ";
    pp ~mode ~split ~ascii ppf then_;
    Fmt.pf ppf " else ";
    pp ~mode ~split ~ascii ppf else_
  (* The value is already in the model's time unit, so it prints as the number
     it is; the literal's own unit name would misread it. *)
  | Ir.UnitLiteral u -> pp_inner ~mode ~split ~ascii ppf (Ir.Const u.value)
  | Ir.UncheckedDim u ->
    Fmt.pf ppf "unchecked_dim(";
    pp ~mode ~split ~ascii ppf u.inner;
    Fmt.pf ppf ", dim = (%d,%d))" u.dim_p u.dim_t
  | Ir.Reduce terms ->
    Fmt.pf ppf "(";
    List.iteri (fun i t ->
      if i > 0 then Fmt.pf ppf " + ";
      pp ~mode ~split ~ascii ppf t
    ) terms;
    Fmt.pf ppf ")"
  | Ir.BindingRef n ->
    Term_style.param Fmt.string ppf n
  | Ir.PerEvalRef n ->
    Term_style.param Fmt.string ppf n
  | Ir.ObsAnchor a ->
    (* Render as the modeller wrote it, in MODEL time units (the offset was
       folded at compile time, so the original unit is gone). *)
    let name = match a.Ir.anchor with
      | Ir.AnchorFirst -> "first_obs" | Ir.AnchorLast -> "last_obs" in
    Term_style.param Fmt.string ppf name;
    if a.Ir.offset > 0.0 then Fmt.pf ppf " + %g" a.Ir.offset
    else if a.Ir.offset < 0.0 then Fmt.pf ppf " - %g" (-. a.Ir.offset)

(** Render an IR expression to a plain (ASCII, IR-name) string. Used by
    diagnostics that need to quote a sub-expression in error text. *)
let to_string (e : Ir.expr) : string =
  Format.asprintf "%a" (pp ~mode:Ir ~split:no_split ~ascii:true) e

(* ── Convenience: build split_fn from a model + stratification info ──────── *)

(** Build a split function from a flat map of expanded_name → (base, dim_values).
    Call [make_split_map base_comps strats] where:
      base_comps : (base_name * dims list) list
      strats     : (dim_name * values list) list
    Returns a function usable as [split_fn]. *)
let make_split_map
    (base_dims : (string * string list) list)
    (dim_vals  : (string * string list) list)
    : split_fn =
  let tbl : (string, string * string list) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun (base, dims) ->
    let all_vals = List.filter_map (fun d -> List.assoc_opt d dim_vals) dims in
    let rec cart = function
      | [] -> [[]]
      | vs :: rest ->
        let tails = cart rest in
        List.concat_map (fun v -> List.map (fun t -> v :: t) tails) vs
    in
    let combos = if all_vals = [] then [[]] else cart all_vals in
    List.iter (fun combo ->
      let expanded =
        if combo = [] then base
        else String.concat "_" (base :: combo)
      in
      Hashtbl.replace tbl expanded (base, combo)
    ) combos
  ) base_dims;
  fun name -> Hashtbl.find_opt tbl name
