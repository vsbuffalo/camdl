(* Pretty-printer for IR expressions with semantic coloring and
   precedence-aware parenthesization.

   Usage:
     Pp_expr.pp ~mode:Pp_expr.Ir ~split:Pp_expr.no_split ~ascii:false ppf expr
*)

type name_mode = Dsl | Ir

(** Split an IR flat compartment name into (base, dimension_values).
    e.g. "S_child" → Some ("S", ["child"])
    Used in DSL mode to render S[child] instead of S_child. *)
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
  | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
  | Ir.Time | Ir.Dt | Ir.TimeFunc _ | Ir.TableLookup _ | Ir.Projected
  | Ir.ObsColumnRef _ -> 10
  | Ir.UncheckedDim _ -> 10   (* function-call-like, atomic *)
  | Ir.Reduce _ -> 10         (* rendered self-parenthesized, atomic *)
  | Ir.BindingRef _ -> 10     (* a name, atomic *)
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
       fall back to %g. Prior version had both arms identical, losing
       the intended integer-literal rendering — reported as m4 in the
       2026-04-19 compiler review. *)
    if Float.is_integer f && Float.abs f < 1e15 then
      Fmt.pf ppf "%.0f" f
    else
      Fmt.pf ppf "%g" f
  | Ir.Param name ->
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
