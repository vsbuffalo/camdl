(* camdl inspect — model inspection and pretty-printing.
   All output goes to the provided ppf (typically Fmt.stdout or Fmt.stderr). *)

open Ast

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

(** Build a Pp_expr split_fn from the expander context. *)
let make_split ctx =
  let base_dims = List.map (fun cd ->
    let dims = List.filter_map (fun sd ->
      let applies = match sd.sonly with
        | None -> true
        | Some only -> List.mem cd.cname only
      in
      if applies then Some sd.sdim else None
    ) ctx.Expander.stratifies in
    (cd.cname, dims)
  ) ctx.Expander.comp_decls in
  (* `inspect` renders a model that already compiled, so an undeclared
     stratify dimension cannot reach here — E214 would have blocked it. These
     five sites therefore render nothing for `None` rather than diagnose; the
     lookups route through [Expander.dim_values] so the accessor invariant
     (A1) holds file-wide. *)
  let dim_vals = List.filter_map (fun sd ->
    Option.map (fun vs -> (sd.sdim, vs)) (Expander.dim_values ctx sd.sdim)
  ) ctx.Expander.stratifies in
  Pp_expr.make_split_map base_dims dim_vals

let pp_rate ?(ascii=false) ~split ppf expr =
  Pp_expr.pp ~mode:Pp_expr.Dsl ~split ~ascii ppf expr

(** Format a number with thousands separators: 5983740 → "5,983,740" *)
let fmt_number n =
  let s = string_of_int n in
  let len = String.length s in
  let buf = Buffer.create (len + len / 3) in
  String.iteri (fun i c ->
    if i > 0 && (len - i) mod 3 = 0 then Buffer.add_char buf ',';
    Buffer.add_char buf c
  ) s;
  Buffer.contents buf

(** Render a guard in human-readable form. *)
let pp_guard ~ascii ppf g =
  let neq = if ascii then "!=" else "\xe2\x89\xa0" in  (* ≠ *)
  let relop_str = function
    | RLt -> "<" | RLe -> "<=" | RGt -> ">" | RGe -> ">=" | REq -> "==" | RNe -> neq in
  let operand_str = function GoNum f -> Printf.sprintf "%g" f | GoName n -> n in
  let rec pp ppf = function
    | GEq  (a, b) -> Fmt.pf ppf "%s == %s" a b
    | GNeq (a, b) -> Fmt.pf ppf "%s %s %s" a neq b
    | GTab (t, idxs, op, v) ->
      Fmt.pf ppf "%s[%s] %s %s" t (String.concat "," idxs) (relop_str op) (operand_str v)
    | GAnd (g1, g2) -> Fmt.pf ppf "%a and %a" pp g1 pp g2
    | GOr  (g1, g2) -> Fmt.pf ppf "%a or %a"  pp g1 pp g2
  in
  pp ppf g

(** Render index bindings in [v in dim, ...] form. *)
let pp_indices ppf ibs =
  let pp_one ppf ib = match ib with
    | IBind (v, d) ->
      Fmt.pf ppf "%s in " v;
      Term_style.dimension Fmt.string ppf d
    | IConsec (v, vn, d) ->
      Fmt.pf ppf "(%s, %s) in consecutive(" v vn;
      Term_style.dimension Fmt.string ppf d;
      Fmt.pf ppf ")"
    | IComp v ->
      Fmt.pf ppf "%s in " v;
      Term_style.dim_style Fmt.string ppf "compartments"
  in
  Fmt.pf ppf "[";
  List.iteri (fun i ib ->
    if i > 0 then Fmt.pf ppf ", ";
    pp_one ppf ib
  ) ibs;
  Fmt.pf ppf "]"

(** Find all IR transitions whose name starts with [base_name] or equals it. *)
let transitions_for_base (trs : Ir.transition list) base_name =
  List.filter (fun (t : Ir.transition) ->
    Expander.is_expansion_of ~base:base_name t.name) trs

(** Pattern match: glob where * matches any substring. *)
let glob_match pattern s =
  if not (String.contains pattern '*') then pattern = s
  else begin
    (* Simple prefix/suffix glob *)
    let parts = String.split_on_char '*' pattern in
    let rec check s = function
      | [] -> s = ""
      | [last] ->
        let n = String.length last in
        String.length s >= n &&
        String.sub s (String.length s - n) n = last
      | part :: rest ->
        let n = String.length part in
        if String.length s < n then false
        else if String.sub s 0 n = part then
          (* consume part then search for rest *)
          let s' = String.sub s n (String.length s - n) in
          let rec try_pos i =
            if i > String.length s' then false
            else check (String.sub s' i (String.length s' - i)) rest
            || try_pos (i + 1)
          in
          try_pos 0
        else false
    in
    match parts with
    | [] -> true
    | [""] -> true  (* pattern is just "*" or "" *)
    | first :: rest ->
      let n = String.length first in
      if n > 0 then
        String.length s >= n && String.sub s 0 n = first
        && check (String.sub s n (String.length s - n)) rest
      else
        check s rest
  end

(* Render a parsed `#'` doc block inline after a declaration: the prose as
   `# …`, then the @symbol (in parameter colour) and @ref (bracketed), each
   shown only if present. Shared by the parameter / compartment / transition
   listings. *)
let render_doc ppf (d : Ast.doc) =
  (match d.d_text with
   | Some t ->
     Term_style.dim_style Fmt.string ppf "   # ";
     Term_style.dim_style Fmt.string ppf t
   | None -> ());
  (match d.d_symbol with
   | Some s -> Fmt.pf ppf "  "; Term_style.param Fmt.string ppf s
   | None -> ());
  (match d.d_ref with
   | Some r -> Term_style.dim_style Fmt.string ppf (Printf.sprintf "  [%s]" r)
   | None -> ())

(* Human-readable name for a parameter kind. The default summary,
   --parameters, and --dims all group parameters by these labels. *)
let pkind_str (k : Ast.param_type) : string = match k with
  | Ast.PRate -> "rate" | Ast.PProbability -> "probability"
  | Ast.PPositive -> "positive" | Ast.PCount -> "count" | Ast.PReal -> "real"
  | Ast.PInstant -> "instant" | Ast.PDuration -> "duration"

(* Presentation order for the parameter-kind groupings. *)
let kind_order = ["rate"; "probability"; "positive"; "count"; "real"]

(* Find the source AST declaration for an IR parameter: a scalar matches by
   name; an indexed declaration matches its expanded leaves by the `<base>_`
   prefix (and, latently, the bare base name — dead for expanded leaves, kept
   so every caller shares one lookup). *)
let decl_of_param (ctx : Expander.context) (p : Ir.parameter) : Ast.param_decl option =
  List.find_opt (fun pd ->
    match pd with
    | Ast.PScalar s -> s.pname = p.name
    | Ast.PIndexed ix ->
      p.name = ix.pname || Expander.is_indexed_leaf ~base:ix.pname p.name
  ) ctx.Expander.param_decls

(* ── --summary ───────────────────────────────────────────────────────────── *)

let run_summary ppf (model : Ir.model) ctx (sum : Expander.model_summary) =
  (* Model name in bold blue *)
  Term_style.bold (Term_style.transition Fmt.string) ppf model.name;
  Fmt.pf ppf "@\n@\n";
  let lbl s = Term_style.dim_style Fmt.string ppf s in
  let num n = Term_style.bold Fmt.string ppf (fmt_number n) in
  (* Compartments *)
  lbl "  compartments   ";
  (if sum.base_compartment_count = sum.expanded_compartment_count then
     num sum.expanded_compartment_count
   else begin
     num sum.base_compartment_count;
     (* Show dimension breakdown *)
     let dims = List.filter_map (fun sd ->
       Option.map (fun vs -> Printf.sprintf "%d %s" (List.length vs) sd.sdim)
         (Expander.dim_values ctx sd.sdim)
     ) ctx.Expander.stratifies in
     if dims <> [] then (
       Fmt.pf ppf " base";
       List.iter (fun d ->
         Term_style.dim_style Fmt.string ppf " \xc3\x97 ";  (* × *)
         Fmt.pf ppf "%s" d
       ) dims;
       Fmt.pf ppf " = ";
       num sum.expanded_compartment_count;
       Fmt.pf ppf " expanded"
     ) else (
       Fmt.pf ppf " expanded"
     )
   end);
  Fmt.pf ppf "@\n";
  (* Transitions *)
  lbl "  transitions     ";
  num sum.base_transition_count;
  Fmt.pf ppf " base ";
  Term_style.dim_style Fmt.string ppf "\xe2\x86\x92 ";  (* → *)
  num sum.expanded_transition_count;
  Fmt.pf ppf " expanded";
  if sum.filtered_transition_count > 0 then (
    Fmt.pf ppf " (+ ";
    num sum.filtered_transition_count;
    Fmt.pf ppf " filtered by where)"
  ) else
    Fmt.pf ppf " (+ 0 filtered by where)";
  Fmt.pf ppf "@\n";
  (* Parameters — summary groups parameters by declared kind. Full
     per-parameter listing is available via `camdlc inspect --parameters`.
     Previously the default dumped all parameters inline, which scaled
     badly: a Garki model with 17 params wraps to an unreadable blob.
     Reports both declarations (source-level) and expanded count (IR
     params after `p[dim]` stratification), matching the
     compartments line's "N base × dim = M expanded" shape. *)
  lbl "  parameters      ";
  let n_decl = sum.param_count in
  let n_exp  = List.length model.parameters in
  if n_decl = n_exp then (
    num n_exp;
    Fmt.pf ppf " declared"
  ) else (
    num n_decl;
    Fmt.pf ppf " declared";
    Fmt.pf ppf " \xe2\x86\x92 ";  (* → *)
    num n_exp;
    Fmt.pf ppf " expanded"
  );
  if model.parameters <> [] then (
    let kind_of (p : Ir.parameter) =
      match decl_of_param ctx p with
      | Some (Ast.PScalar pd)  -> pkind_str pd.pkind
      | Some (Ast.PIndexed pd) -> pkind_str pd.pkind
      | None -> "?"
    in
    (* Count by kind, preserving a stable declaration-order-like
       presentation: rate, probability, positive, count, real. *)
    let counts_by_kind = List.map (fun k ->
      (k, List.length (List.filter (fun p -> kind_of p = k) model.parameters))
    ) kind_order in
    let nonzero = List.filter (fun (_, n) -> n > 0) counts_by_kind in
    if nonzero <> [] then (
      Fmt.pf ppf " (";
      List.iteri (fun i (k, n) ->
        if i > 0 then Fmt.pf ppf ", ";
        Fmt.pf ppf "%d " n;
        Term_style.param Fmt.string ppf k
      ) nonzero;
      Fmt.pf ppf ")"
    )
  );
  Fmt.pf ppf "@\n";
  (* Tables *)
  lbl "  tables          ";
  num sum.table_count;
  if ctx.table_decls <> [] then (
    Fmt.pf ppf " (";
    List.iteri (fun i td ->
      if i > 0 then Fmt.pf ppf ", ";
      Term_style.table Fmt.string ppf (String.concat ", " td.tnames);
      let dim_names = List.map (function TDim d -> d | TDimUnit (d,_) -> d) td.tdims in
      if dim_names <> [] then (
        Term_style.dim_style Fmt.string ppf ": ";
        Term_style.dim_style Fmt.string ppf (String.concat " \xc3\x97 " dim_names)
      )
    ) ctx.table_decls;
    Fmt.pf ppf ")"
  );
  Fmt.pf ppf "@\n";
  (* Let bindings *)
  lbl "  let bindings    ";
  num sum.let_binding_count;
  if ctx.let_bindings <> [] then (
    Fmt.pf ppf " (";
    List.iteri (fun i lb ->
      if i > 0 then Fmt.pf ppf ", ";
      Term_style.table Fmt.string ppf lb.lname;
      if lb.lindices <> [] then pp_indices ppf lb.lindices
    ) ctx.let_bindings;
    Fmt.pf ppf ")"
  );
  Fmt.pf ppf "@\n";
  (* Dimensions *)
  lbl "  dimensions      ";
  let strats = ctx.Expander.stratifies in
  if strats = [] then
    Term_style.dim_style Fmt.string ppf "none"
  else
    List.iteri (fun i sd ->
      if i > 0 then Fmt.pf ppf ", ";
      Term_style.dimension Fmt.string ppf sd.sdim;
      Fmt.pf ppf " = [";
      let vs = Option.value ~default:[] (Expander.dim_values ctx sd.sdim) in
      List.iteri (fun j v ->
        if j > 0 then Fmt.pf ppf ", ";
        Fmt.pf ppf "%s" v
      ) vs;
      Fmt.pf ppf "]";
      (* Append the dimension's `#'` doc text, if any (dims are not in the IR,
         so the summary is their surfacing point). *)
      (match List.find_opt (fun (de : dimensions_entry) -> de.dename = sd.sdim)
               ctx.Expander.dim_decls with
       | Some { dedoc = Some { d_text = Some t; _ }; _ } ->
         Term_style.dim_style Fmt.string ppf (Printf.sprintf "  # %s" t)
       | _ -> ())
    ) strats;
  Fmt.pf ppf "@\n";
  (* Observations *)
  lbl "  observations    ";
  num sum.obs_count;
  Fmt.pf ppf " streams@\n";
  (* Interventions *)
  lbl "  interventions   ";
  num sum.interv_count;
  Fmt.pf ppf " (0 active by default)@\n"

(* ── --cost-report ───────────────────────────────────────────────────────────
   A read-only cost analysis of the compiled IR: where the per-step
   evaluation work concentrates, what the sparse-coupling fold collapses, how
   much shared bindings are reused, and which rewrite-eligible idioms appear.
   It is the analogue of the runtime `eval_stats` (numerical pathologies) for
   *cost*. Nothing here mutates the model — the constant-fold comparison runs
   on a local copy. Gradient-node costs are NOT reported: inspect compiles the
   front-end only (no autodiff), so `rate_grad` is empty here. *)

(* Total node count of an expression tree (every constructor counts as 1). *)
let rec expr_node_count (e : Ir.expr) : int =
  let open Ir in
  match e with
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> 1
  | BinOp b -> 1 + expr_node_count b.left + expr_node_count b.right
  | UnOp u  -> 1 + expr_node_count u.arg
  | Cond c  -> 1 + expr_node_count c.pred + expr_node_count c.then_ + expr_node_count c.else_
  | TableLookup (_, idxs) -> 1 + List.fold_left (fun a i -> a + expr_node_count i) 0 idxs
  | Reduce terms -> 1 + List.fold_left (fun a t -> a + expr_node_count t) 0 terms
  | UncheckedDim u -> 1 + expr_node_count u.inner

(* Total Reduce-term count across an expr tree (sum of arities of all Reduce
   nodes). Walks into every child so nested Reduces all count. *)
let rec reduce_term_count (e : Ir.expr) : int =
  let open Ir in
  match e with
  | Reduce terms ->
    List.length terms
    + List.fold_left (fun a t -> a + reduce_term_count t) 0 terms
  | BinOp b -> reduce_term_count b.left + reduce_term_count b.right
  | UnOp u  -> reduce_term_count u.arg
  | Cond c  -> reduce_term_count c.pred + reduce_term_count c.then_ + reduce_term_count c.else_
  | TableLookup (_, idxs) -> List.fold_left (fun a i -> a + reduce_term_count i) 0 idxs
  | UncheckedDim u -> reduce_term_count u.inner
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> 0

(* Count BindingRefs to [name] within an expr tree. *)
let rec count_bindingref name (e : Ir.expr) : int =
  let open Ir in
  match e with
  | BindingRef n -> if n = name then 1 else 0
  | PerEvalRef _ -> 0   (* a per-eval ref is not a BindingRef *)
  | BinOp b -> count_bindingref name b.left + count_bindingref name b.right
  | UnOp u  -> count_bindingref name u.arg
  | Cond c  -> count_bindingref name c.pred + count_bindingref name c.then_ + count_bindingref name c.else_
  | TableLookup (_, idxs) -> List.fold_left (fun a i -> a + count_bindingref name i) 0 idxs
  | Reduce terms -> List.fold_left (fun a t -> a + count_bindingref name t) 0 terms
  | UncheckedDim u -> count_bindingref name u.inner
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | Projected | ObsColumnRef _ -> 0

(* All rate exprs of a model (the cost surface inspect can see — gradients are
   absent in front-end-only compilation, and bindings are counted separately
   so their bodies are NOT included here). *)
let rate_exprs (m : Ir.model) : Ir.expr list =
  List.map (fun (t : Ir.transition) -> t.rate) m.transitions

(* `1 - exp(x)` shaped subexprs: the numerically-unstable hazard-probability
   form. As x → 0, exp(x) → 1 and the subtraction loses precision to
   catastrophic cancellation — the form `expm1` / `prob_q_from_rate_dt` exists
   to avoid (inference/numerics.rs). Matches BinOp(Sub, Const 1.0, UnOp(Exp, _))
   for any exponent (the cancellation is in the `1 - ·`, independent of the
   argument's shape). Counts every occurrence anywhere in the tree. *)
let rec count_hazard_idioms (e : Ir.expr) : int =
  let open Ir in
  let here = match e with
    | BinOp { op = Sub; left = Const 1.0;
              right = UnOp { op = Exp; _ } } -> 1
    | _ -> 0
  in
  here + (match e with
    | BinOp b -> count_hazard_idioms b.left + count_hazard_idioms b.right
    | UnOp u  -> count_hazard_idioms u.arg
    | Cond c  -> count_hazard_idioms c.pred + count_hazard_idioms c.then_ + count_hazard_idioms c.else_
    | TableLookup (_, idxs) -> List.fold_left (fun a i -> a + count_hazard_idioms i) 0 idxs
    | Reduce terms -> List.fold_left (fun a t -> a + count_hazard_idioms t) 0 terms
    | UncheckedDim u -> count_hazard_idioms u.inner
    | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
    | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> 0)

(* Structural hash of an expr, for detecting duplicated subexpressions. A
   simple recursive polynomial hash over the constructor shape + leaf payloads.
   Collisions are possible but harmless: this drives an advisory count only. *)
let rec expr_hash (e : Ir.expr) : int =
  let open Ir in
  let mix tag parts = List.fold_left (fun h p -> (h * 31 + p) land max_int) (tag * 2654435761 land max_int) parts in
  match e with
  | Const f -> mix 1 [ Hashtbl.hash f ]
  | Param p -> mix 2 [ Hashtbl.hash p ]
  | Pop c -> mix 3 [ Hashtbl.hash c ]
  | PopSum cs -> mix 4 [ Hashtbl.hash cs ]
  | Time -> mix 5 []
  | Dt -> mix 6 []
  | TimeFunc n -> mix 7 [ Hashtbl.hash n ]
  | BindingRef n -> mix 8 [ Hashtbl.hash n ]
  | PerEvalRef n -> mix 17 [ Hashtbl.hash n ]
  | Projected -> mix 9 []
  | BinOp b -> mix 10 [ Hashtbl.hash b.op; expr_hash b.left; expr_hash b.right ]
  | UnOp u -> mix 11 [ Hashtbl.hash u.op; expr_hash u.arg ]
  | Cond c -> mix 12 [ expr_hash c.pred; expr_hash c.then_; expr_hash c.else_ ]
  | TableLookup (n, idxs) -> mix 13 (Hashtbl.hash n :: List.map expr_hash idxs)
  | Reduce terms -> mix 14 (List.map expr_hash terms)
  | UncheckedDim u -> mix 15 [ expr_hash u.inner ]
  | ObsColumnRef c -> mix 16 [ Hashtbl.hash c ]

(* Number of distinct non-trivial subexpressions that recur ≥ [threshold]
   times across all given roots. "Non-trivial" excludes single-node leaves
   (a repeated `Const 0.0` is not interesting). Uses [expr_hash] as the
   structural key. *)
let count_duplicated_subexprs ?(threshold = 3) (roots : Ir.expr list) : int =
  let counts : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let rec walk (e : Ir.expr) =
    let open Ir in
    (if expr_node_count e > 1 then
       let h = expr_hash e in
       Hashtbl.replace counts h (1 + (Option.value ~default:0 (Hashtbl.find_opt counts h))));
    match e with
    | BinOp b -> walk b.left; walk b.right
    | UnOp u  -> walk u.arg
    | Cond c  -> walk c.pred; walk c.then_; walk c.else_
    | TableLookup (_, idxs) -> List.iter walk idxs
    | Reduce terms -> List.iter walk terms
    | UncheckedDim u -> walk u.inner
    | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
    | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> ()
  in
  List.iter walk roots;
  Hashtbl.fold (fun _ n acc -> if n >= threshold then acc + 1 else acc) counts 0

let run_cost_report ppf (model : Ir.model) _ctx =
  let lbl s = Term_style.dim_style Fmt.string ppf s in
  let num n = Term_style.bold Fmt.string ppf (fmt_number n) in
  (* Header: model name in bold blue (matches run_summary). *)
  Term_style.bold (Term_style.transition Fmt.string) ppf model.name;
  Fmt.pf ppf " cost report@\n@\n";

  (* ── Counts ─────────────────────────────────────────────────────── *)
  let n_tr = List.length model.transitions in
  let n_bind = List.length model.bindings in
  let rates = rate_exprs model in
  let total_nodes = List.fold_left (fun a e -> a + expr_node_count e) 0 rates in
  (* Max rate-expr node count, with the holding transition's name. *)
  let (max_nodes, max_name) =
    List.fold_left (fun (mx, nm) (t : Ir.transition) ->
      let c = expr_node_count t.rate in
      if c > mx then (c, t.name) else (mx, nm)) (0, "") model.transitions
  in
  lbl "  transitions       "; num n_tr; Fmt.pf ppf "@\n";
  lbl "  bindings          "; num n_bind; Fmt.pf ppf "@\n";
  lbl "  rate nodes        "; num total_nodes; Fmt.pf ppf " total, ";
  num max_nodes; Fmt.pf ppf " max";
  if max_name <> "" then (
    Fmt.pf ppf " (";
    Term_style.transition Fmt.string ppf max_name;
    Fmt.pf ppf ")");
  Fmt.pf ppf "@\n";

  (* ── Constant-fold collapse (Reduce terms before/after) ─────────── *)
  let folded = Constant_fold.fold_model model in
  let folded_rates = rate_exprs folded in
  let reduce_before = List.fold_left (fun a e -> a + reduce_term_count e) 0 rates in
  let reduce_after  = List.fold_left (fun a e -> a + reduce_term_count e) 0 folded_rates in
  lbl "  Reduce terms      "; num reduce_before; Fmt.pf ppf " before fold ";
  Term_style.dim_style Fmt.string ppf "\xe2\x86\x92 ";  (* → *)
  num reduce_after; Fmt.pf ppf " after";
  (if reduce_before > 0 then
     let pct = 100.0 *. float_of_int (reduce_before - reduce_after) /. float_of_int reduce_before in
     Fmt.pf ppf " (%.0f%% collapsed)" pct);
  Fmt.pf ppf "@\n";

  (* ── Top bindings by reuse ──────────────────────────────────────── *)
  let bind_dep = Expr_analysis.model_binding_deps model in
  let binding_rows =
    List.map (fun (b : Ir.binding) ->
      let refs = List.fold_left (fun a e -> a + count_bindingref b.bname e) 0 rates in
      let size = expr_node_count b.bexpr in
      let saved = if refs > 1 then (refs - 1) * size else 0 in
      (b.bname, Expr_analysis.dep_name (bind_dep b.bname), size, refs, saved))
      model.bindings
  in
  (* Sort by node-visits saved (descending), then by refs. *)
  let binding_rows =
    List.sort (fun (_, _, _, r1, s1) (_, _, _, r2, s2) ->
      if s2 <> s1 then compare s2 s1 else compare r2 r1) binding_rows
  in
  Fmt.pf ppf "@\n";
  lbl "  top bindings by reuse";
  Fmt.pf ppf "@\n";
  if binding_rows = [] then (
    Fmt.pf ppf "    "; Term_style.dim_style Fmt.string ppf "none"; Fmt.pf ppf "@\n")
  else (
    let top = List.filteri (fun i _ -> i < 8) binding_rows in
    List.iter (fun (name, dep, size, refs, saved) ->
      Fmt.pf ppf "    ";
      Term_style.table Fmt.string ppf name;
      Fmt.pf ppf "  ";
      Term_style.dim_style Fmt.string ppf dep;
      Fmt.pf ppf "  size="; num size;
      Fmt.pf ppf "  refs="; num refs;
      Fmt.pf ppf "  ~saved="; num saved;
      Fmt.pf ppf "@\n") top);

  (* ── Rewrite-eligible idioms ────────────────────────────────────── *)
  (* Idiom search spans rate bodies AND binding bodies (a hazard idiom or a
     shared subexpr may have been hoisted). *)
  let all_exprs = rates @ List.map (fun (b : Ir.binding) -> b.bexpr) model.bindings in
  let hazard = List.fold_left (fun a e -> a + count_hazard_idioms e) 0 all_exprs in
  let dups = count_duplicated_subexprs all_exprs in
  Fmt.pf ppf "@\n";
  lbl "  rewrite-eligible idioms";
  Fmt.pf ppf "@\n";
  Fmt.pf ppf "    1 - exp(x) hazard forms    "; num hazard; Fmt.pf ppf "@\n";
  Fmt.pf ppf "    duplicated subexprs (\xe2\x89\xa53)   "; num dups; Fmt.pf ppf "@\n"

(* ── --compartments ──────────────────────────────────────────────────────── *)

let run_compartments ppf (model : Ir.model) ctx =
  let split = make_split ctx in
  List.iter (fun cd ->
    let base = cd.cname in
    let kind_str = match cd.ckind with
      | Integer -> "integer" | Real -> "real"
    in
    let dims = List.filter_map (fun sd ->
      let applies = match sd.sonly with
        | None -> true
        | Some only -> List.mem base only
      in
      if applies then Some sd.sdim else None
    ) ctx.Expander.stratifies in
    let expanded = List.filter (fun (c : Ir.compartment) ->
      match split c.name with
      | Some (b, _) -> b = base
      | None -> c.name = base
    ) model.compartments in
    (* Name in bold magenta *)
    Term_style.compartment (Term_style.bold Fmt.string) ppf base;
    Fmt.pf ppf "   ";
    Term_style.dim_style Fmt.string ppf kind_str;
    Fmt.pf ppf "   ";
    if dims = [] then
      Term_style.dim_style Fmt.string ppf "[]"
    else (
      Term_style.dim_style (fun ppf () ->
        Fmt.pf ppf "[";
        List.iteri (fun i d ->
          if i > 0 then Fmt.pf ppf ", ";
          Term_style.dimension Fmt.string ppf d
        ) dims;
        Fmt.pf ppf "]"
      ) ppf ()
    );
    Fmt.pf ppf "   ";
    Term_style.dim_style Fmt.string ppf "\xe2\x86\x92 ";  (* → *)
    Term_style.dim_style (fun ppf () ->
      List.iteri (fun i (c : Ir.compartment) ->
        if i > 0 then Fmt.pf ppf ", ";
        (* Show in DSL mode *)
        Pp_expr.pp_pop ~mode:Pp_expr.Dsl ~split ppf c.name
      ) expanded
    ) ppf ();
    (match cd.cdoc with Some d -> render_doc ppf d | None -> ());
    Fmt.pf ppf "@\n"
  ) ctx.comp_decls;
  Fmt.pf ppf "@\n";
  let n_exp = List.length model.compartments in
  let n_base = List.length ctx.comp_decls in
  Term_style.bold Fmt.string ppf (fmt_number n_exp);
  Fmt.pf ppf " expanded compartments (%d base" n_base;
  List.iter (fun sd ->
    let n = List.length
      (Option.value ~default:[] (Expander.dim_values ctx sd.sdim)) in
    Fmt.pf ppf " \xc3\x97 %d " n;
    Term_style.dimension Fmt.string ppf sd.sdim
  ) ctx.Expander.stratifies;
  Fmt.pf ppf ")@\n"

(* ── --parameters ────────────────────────────────────────────────────────── *)

(** Full parameter listing grouped by kind. Shows name, kind, bounds
    (if declared), and any prior / hierarchical structure. Matches the
    default's at-a-glance summary with a detailed view suitable for
    larger models (Garki has 17 params, which wraps the one-line
    summary's listing unreadably). *)
let run_parameters ppf (model : Ir.model) (ctx : Expander.context) =
  let kind_of (p : Ir.parameter) =
    match decl_of_param ctx p with
    | Some (Ast.PScalar pd)  -> pkind_str pd.pkind
    | Some (Ast.PIndexed pd) -> pkind_str pd.pkind
    | None -> "?"
  in
  (* Doc comment for a parameter, looked up from its source declaration
     (scalar by name, indexed by `name_`-prefix). An indexed param's leaves
     all share the one declaration's doc, mirroring shared bounds. *)
  let doc_of (p : Ir.parameter) =
    match decl_of_param ctx p with
    | Some (Ast.PScalar pd)  -> pd.pdoc
    | Some (Ast.PIndexed pd) -> pd.pdoc
    | None -> None
  in
  List.iter (fun kind ->
    let ps = List.filter (fun p -> kind_of p = kind) model.parameters in
    if ps <> [] then begin
      Term_style.dim_style Fmt.string ppf kind;
      Fmt.pf ppf "@\n";
      List.iter (fun (p : Ir.parameter) ->
        Fmt.pf ppf "  ";
        Term_style.param Fmt.string ppf p.name;
        (match Ir.param_bounds p with
         | Some (lo, hi) -> Fmt.pf ppf "   in [%g, %g]" lo hi
         | None -> ());
        (match Ir.param_concrete_value p with
         | Some v when Ir.param_bounds p = None ->
           Term_style.dim_style Fmt.string ppf "  = ";
           Fmt.pf ppf "%g" v
         | _ -> ());
        (match Ir.param_prior_dist p with
         | Some _ ->
           Term_style.dim_style Fmt.string ppf "  ~ prior"
         | None -> ());
        (match Ir.param_hierarchical p with
         | Some h ->
           Term_style.dim_style Fmt.string ppf (Printf.sprintf "  ~ %s | " (Ir.hierarchical_kind_name h.hkind));
           (* Show referenced hyperparameter names *)
           let parents = List.filter_map (fun (_, e) ->
             match e with Ir.Param n -> Some n | _ -> None) h.hargs in
           Fmt.pf ppf "%s" (String.concat ", " parents);
           if h.hpool_over <> "" then
             Fmt.pf ppf " [pool=%s]" h.hpool_over
         | None -> ());
        (match doc_of p with Some d -> render_doc ppf d | None -> ());
        Fmt.pf ppf "@\n"
      ) ps;
      Fmt.pf ppf "@\n"
    end
  ) kind_order;
  (* Catch-all for unknown-kind params (defensive) *)
  let leftover = List.filter (fun p ->
    not (List.mem (kind_of p) kind_order)
  ) model.parameters in
  if leftover <> [] then begin
    Term_style.dim_style Fmt.string ppf "other";
    Fmt.pf ppf "@\n";
    List.iter (fun (p : Ir.parameter) ->
      Fmt.pf ppf "  ";
      Term_style.param Fmt.string ppf p.name;
      Fmt.pf ppf "@\n"
    ) leftover
  end;
  let n = List.length model.parameters in
  Term_style.bold Fmt.string ppf (fmt_number n);
  Fmt.pf ppf " declared parameters@\n"

(* ── --transitions [PATTERN] ────────────────────────────────────────────── *)

let run_transitions ppf (model : Ir.model) ctx (pattern : string option) ~ascii =
  let split = make_split ctx in
  let arrow = if ascii then "->" else "\xe2\x86\x92" in  (* → *)
  let bar   = "\xe2\x94\x82" in                          (* │ *)
  (* For each base/original transition, group the expanded ones *)
  List.iter (fun (orig_tr : transition_decl) ->
    let base = orig_tr.trname in
    let all_expanded = transitions_for_base model.transitions base in
    let matching = match pattern with
      | None -> all_expanded
      | Some pat -> List.filter (fun (t : Ir.transition) -> glob_match pat t.name) all_expanded
    in
    if matching = [] && pattern <> None then ()  (* skip if pattern filters out all *)
    else begin
      (* Group header: infection[a in age] → 2 transitions *)
      Term_style.bold (Term_style.transition Fmt.string) ppf base;
      if orig_tr.trindices <> [] then (
        Term_style.dim_style (fun ppf () ->
          pp_indices ppf orig_tr.trindices
        ) ppf ()
      );
      (match orig_tr.trguard with
       | None -> ()
       | Some g ->
         Term_style.dim_style Fmt.string ppf " where ";
         pp_guard ~ascii ppf g);
      Fmt.pf ppf " %s " arrow;
      Term_style.bold Fmt.string ppf (fmt_number (List.length all_expanded));
      Fmt.pf ppf " transition%s"
        (if List.length all_expanded = 1 then "" else "s");
      (match pattern with
       | Some _ when List.length matching <> List.length all_expanded ->
         Fmt.pf ppf " (%d matching)" (List.length matching)
       | _ -> ());
      (match orig_tr.trdoc with Some d -> render_doc ppf d | None -> ());
      Fmt.pf ppf "@\n";
      (* Render with truncation *)
      let render_tr (t : Ir.transition) =
        (* Find corresponding let bindings referenced in rate *)
        let src_name = Option.map fst
          (List.find_opt (fun (_, d) -> d = -1) t.stoichiometry) in
        let dst_name = Option.map fst
          (List.find_opt (fun (_, d) -> d = 1) t.stoichiometry) in
        Fmt.pf ppf "  ";
        Term_style.dim_style Fmt.string ppf bar;
        Fmt.pf ppf " ";
        Term_style.transition Fmt.string ppf t.name;
        Fmt.pf ppf " : ";
        (match src_name with
         | None -> ()
         | Some s ->
           Pp_expr.pp_pop ~mode:Pp_expr.Dsl ~split ppf s;
           Fmt.pf ppf " ";
           Term_style.dim_style Fmt.string ppf arrow;
           Fmt.pf ppf " ");
        (match dst_name with
         | None -> ()
         | Some d ->
           Pp_expr.pp_pop ~mode:Pp_expr.Dsl ~split ppf d);
        (* Rate: inline if simple, on next line if complex *)
        let rate_str = Format.asprintf "%a" (pp_rate ~ascii ~split) t.rate in
        if String.length rate_str <= 50 then (
          Fmt.pf ppf "   @@ %a@\n" (pp_rate ~ascii ~split) t.rate
        ) else (
          Fmt.pf ppf "@\n  ";
          Term_style.dim_style Fmt.string ppf bar;
          Fmt.pf ppf "   @@ %a@\n" (pp_rate ~ascii ~split) t.rate
        )
      in
      let n_matching = List.length matching in
      if n_matching <= 6 then
        List.iter render_tr matching
      else begin
        let first3 = List.filteri (fun i _ -> i < 3) matching in
        let last1  = List.nth matching (n_matching - 1) in
        List.iter render_tr first3;
        Fmt.pf ppf "  ";
        Term_style.dim_style Fmt.string ppf bar;
        Fmt.pf ppf " ... (%s more)@\n" (fmt_number (n_matching - 4));
        render_tr last1
      end;
      Fmt.pf ppf "@\n"
    end
  ) ctx.Expander.orig_transitions

(** Find let bindings referenced in an AST rate expression. *)
let collect_let_refs_ast ctx ast_rate =
  let found = ref [] in
  let add lb = if not (List.mem lb !found) then found := lb :: !found in
  let rec walk = function
    | EIdent (name, _) ->
      (match List.find_opt (fun lb -> lb.lname = name) ctx.Expander.let_bindings with
       | Some lb -> add lb | None -> ())
    | EIndex (name, _, _) ->
      (match List.find_opt (fun lb -> lb.lname = name) ctx.Expander.let_bindings with
       | Some lb -> add lb | None -> ())
    | EBinOp (_, l, r) -> walk l; walk r
    | EUnOp (_, e) -> walk e
    | ESum (_, _, _, body, _) -> walk body
    | ECond (p, t, el) -> walk p; walk t; walk el
    | EFuncCall (_, args) -> List.iter (fun (_, e) -> walk e) args
    | EList es -> List.iter walk es
    | ERange (a, b) -> walk a; walk b
    | EConst _ | EUnit _ | EObsAccess _ | ERunMember _ -> ()
  in
  walk ast_rate;
  List.rev !found

(* ── --transition NAME --rate ────────────────────────────────────────────── *)

let run_transition_rate ppf (model : Ir.model) ctx name =
  let split = make_split ctx in
  let ascii = false in
  let arrow = "\xe2\x86\x92" in
  match List.find_opt (fun (t : Ir.transition) -> t.name = name) model.transitions with
  | None ->
    (* M30 in 2026-04-19 review: previously this printed an error
       and exited 0, so `camdl inspect --transition foo` with a
       bogus name failed silently from a CI's POV. Exit 1. *)
    Fmt.epr "error: no transition named '%s'@\n" name;
    exit 1
  | Some t ->
    (* Title *)
    Term_style.bold (Term_style.transition Fmt.string) ppf name;
    Fmt.pf ppf "@\n";
    (* Stoichiometry *)
    Fmt.pf ppf "  ";
    Term_style.dim_style Fmt.string ppf "stoichiometry:  ";
    List.iteri (fun i (comp, delta) ->
      if i > 0 then (
        Fmt.pf ppf "  ";
        Term_style.dim_style Fmt.string ppf arrow;
        Fmt.pf ppf "  "
      );
      Pp_expr.pp_pop ~mode:Pp_expr.Dsl ~split ppf comp;
      let sign = if delta > 0 then "+" else "\xe2\x88\x92" in  (* − *)
      Fmt.pf ppf " (%s%d)" sign (abs delta)
    ) t.stoichiometry;
    Fmt.pf ppf "@\n@\n";
    (* Rate *)
    Fmt.pf ppf "  ";
    Term_style.dim_style Fmt.string ppf "rate (total propensity):";
    Fmt.pf ppf "@\n";
    Fmt.pf ppf "    %a@\n@\n" (pp_rate ~ascii ~split) t.rate;
    (* Where: find let bindings referenced in the original AST rate *)
    let ast_rate = match List.find_opt (fun (orig : transition_decl) ->
      Expander.is_expansion_of ~base:orig.trname t.name
    ) ctx.Expander.orig_transitions with
    | Some orig ->
      (* For a `via law(...)` transition there is no single rate expr; reuse the
         func-call walk by reconstructing the law as an EFuncCall over its args
         so any let-bindings referenced in `stages`/`mean`/`rate` still surface. *)
      (match orig.trdyn with
       | Rate e             -> e
       | Via (name, args)   -> EFuncCall (name, args))
    | None -> EConst 0.0
    in
    let refs = collect_let_refs_ast ctx ast_rate in
    if refs <> [] then (
      Fmt.pf ppf "  ";
      Term_style.dim_style Fmt.string ppf "where:";
      Fmt.pf ppf "@\n";
      List.iter (fun (lb : let_binding) ->
        (* Expand the let binding at each index value *)
        let combos = Expander.cartesian_product lb.lindices ctx in
        List.iter (fun env ->
          let idx_vals = List.filter_map (fun ib ->
            match ib with
            | IBind (v, _) -> List.assoc_opt v env
            | IConsec (v, _, _) -> List.assoc_opt v env
            | IComp v -> List.assoc_opt v env
          ) lb.lindices in
          let bound_name =
            if idx_vals = [] then lb.lname
            else lb.lname ^ "[" ^ String.concat ", " idx_vals ^ "]"
          in
          Fmt.pf ppf "    ";
          Term_style.table Fmt.string ppf bound_name;
          Fmt.pf ppf " = ";
          let expanded_body = Expander.normalize_expr
            (Expander.resolve_expr ctx env lb.lbody) in
          Fmt.pf ppf "%a@\n" (pp_rate ~ascii ~split) expanded_body
        ) combos
      ) refs
    );
    (* Origin *)
    (match t.metadata with
     | None -> ()
     | Some m ->
       Fmt.pf ppf "@\n  ";
       Term_style.dim_style Fmt.string ppf "origin:     ";
       (match m.origin_kind with Some s -> Fmt.pf ppf "%s" s | None -> ());
       Fmt.pf ppf "@\n")

(* ── --transition PATTERN --count ───────────────────────────────────────── *)

let run_transition_count ppf (model : Ir.model) ctx (pattern : string option) ~ascii =
  List.iter (fun (orig_tr : transition_decl) ->
    let base = orig_tr.trname in
    let all_expanded = transitions_for_base model.transitions base in
    let matching_n = match pattern with
      | None -> List.length all_expanded
      | Some pat ->
        List.length (List.filter (fun (t : Ir.transition) -> glob_match pat t.name) all_expanded)
    in
    (* Header *)
    Term_style.bold (Term_style.transition Fmt.string) ppf base;
    if orig_tr.trindices <> [] then pp_indices ppf orig_tr.trindices;
    (match orig_tr.trguard with
     | None -> ()
     | Some g ->
       Fmt.pf ppf "@\n  where ";
       pp_guard ~ascii ppf g);
    Fmt.pf ppf "@\n@\n";
    (* Dimension breakdown *)
    List.iter (fun ib ->
      let (var, dim, count) = match ib with
        | IBind (v, d) ->
          let vals = Option.value ~default:[] (Expander.dim_values ctx d) in
          (v, d, List.length vals)
        | IConsec (v, _, d) ->
          let vals = Option.value ~default:[] (Expander.dim_values ctx d) in
          (v, d, max 0 (List.length vals - 1))
        | IComp v ->
          let comps = List.filter (fun cd -> cd.ckind = Integer) ctx.Expander.comp_decls in
          (v, "compartments", List.length comps)
      in
      ignore var;
      Fmt.pf ppf "  ";
      Term_style.dimension Fmt.string ppf dim;
      Fmt.pf ppf "           %d values@\n" count
    ) orig_tr.trindices;
    (* Combinatorial counts.
       M25 in the 2026-04-19 review: the prior version computed
       `all_n = len all_expanded + (len combos - len all_expanded)`
       which simplifies to `combos_len` — a pointless round-trip.
       And it labeled filtered combos as "self-loops", but `where`
       guards can filter on any equality (age == under5, src != dst,
       …) — only src==dst is a self-loop. Both fixed here. *)
    let combos = Expander.cartesian_product orig_tr.trindices ctx in
    let all_n = List.length combos in
    let kept_n = List.length all_expanded in
    Fmt.pf ppf "  all combos     ";
    Term_style.bold Fmt.string ppf (fmt_number all_n);
    Fmt.pf ppf "@\n";
    let filtered_n = all_n - kept_n in
    Fmt.pf ppf "  after where    ";
    Term_style.bold Fmt.string ppf (fmt_number kept_n);
    if filtered_n > 0 then (
      Fmt.pf ppf "  (";
      Term_style.dim_style Fmt.string ppf
        (Printf.sprintf "\xe2\x88\x92%d filtered by where" filtered_n);
      Fmt.pf ppf ")"
    );
    Fmt.pf ppf "@\n";
    (match pattern with
     | Some pat ->
       Fmt.pf ppf "@\nMatching %S: %s transitions@\n"
         pat (fmt_number matching_n)
     | None -> ());
    Fmt.pf ppf "@\n"
  ) ctx.Expander.orig_transitions

(* ── --let NAME ──────────────────────────────────────────────────────────── *)

let run_let ppf ctx name =
  let split = make_split ctx in
  let ascii = false in
  let bar = "\xe2\x94\x82" in
  match List.find_opt (fun lb -> lb.lname = name) ctx.Expander.let_bindings with
  | None ->
    Fmt.epr "error: no let binding named '%s'@\n" name
  | Some lb ->
    (* Header *)
    Term_style.bold (Term_style.table Fmt.string) ppf lb.lname;
    if lb.lindices <> [] then pp_indices ppf lb.lindices;
    Fmt.pf ppf "   ";
    Term_style.dim_style Fmt.string ppf "type: ";
    let dim_names = List.filter_map (function
      | IBind (_, d) -> Some d
      | IConsec (_, _, d) -> Some d
      | IComp _ -> Some "compartments"
    ) lb.lindices in
    if dim_names = [] then
      Term_style.dim_style Fmt.string ppf "scalar"
    else (
      List.iteri (fun i d ->
        if i > 0 then Term_style.dim_style Fmt.string ppf " \xc3\x97 ";
        Term_style.dimension Fmt.string ppf d
      ) dim_names;
      Term_style.dim_style Fmt.string ppf " \xe2\x86\x92 scalar"
    );
    Fmt.pf ppf "@\n@\n";
    (* Expansions *)
    let combos = Expander.cartesian_product lb.lindices ctx in
    let n = List.length combos in
    let show_limit = 6 in
    let to_show =
      if n <= show_limit then combos
      else List.filteri (fun i _ -> i < 3) combos
    in
    let render_combo env =
      let idx_vals = List.filter_map (fun ib ->
        match ib with
        | IBind (v, _)      -> List.assoc_opt v env
        | IConsec (v, _, _) -> List.assoc_opt v env
        | IComp v           -> List.assoc_opt v env
      ) lb.lindices in
      let bound_name =
        if idx_vals = [] then lb.lname
        else
          lb.lname ^ "[" ^ String.concat ", " idx_vals ^ "]"
      in
      Fmt.pf ppf "  ";
      Term_style.dim_style Fmt.string ppf bar;
      Fmt.pf ppf " ";
      Term_style.table Fmt.string ppf bound_name;
      Fmt.pf ppf " = ";
      let body = Expander.normalize_expr (Expander.resolve_expr ctx env lb.lbody) in
      Fmt.pf ppf "%a@\n" (pp_rate ~ascii ~split) body
    in
    List.iter render_combo to_show;
    if n > show_limit then (
      Fmt.pf ppf "  ";
      Term_style.dim_style Fmt.string ppf bar;
      Fmt.pf ppf " ... (%s more)@\n" (fmt_number (n - 4));
      let last = List.nth combos (n - 1) in
      render_combo last
    );
    Fmt.pf ppf "@\n";
    if n > 1 then (
      Term_style.bold Fmt.string ppf (fmt_number n);
      Fmt.pf ppf " entries@\n"
    );
    (* Referenced by.
       M24 in the 2026-04-19 review: previously missed references
       inside EFuncCall / EList / ERange subtrees. A let referenced
       only inside `incidence(N)` or `prevalence(N)` (both
       EFuncCall nodes) didn't show as a reference. Also cleans up
       a dead `rate_str` allocation (n12). *)
    let refs = List.filter_map (fun (orig_tr : transition_decl) ->
      let rec expr_refs_name e =
        match e with
        | EIdent (n, _) when n = lb.lname -> true
        | EIndex (n, _, _) when n = lb.lname -> true
        | EBinOp (_, l, r) -> expr_refs_name l || expr_refs_name r
        | EUnOp (_, e) -> expr_refs_name e
        | ESum (_, _, _, body, _) -> expr_refs_name body
        | ECond (p, t, el) ->
          expr_refs_name p || expr_refs_name t || expr_refs_name el
        | EFuncCall (_, args) ->
          List.exists (fun (_, a) -> expr_refs_name a) args
        | EList es -> List.exists expr_refs_name es
        | ERange (a, b) -> expr_refs_name a || expr_refs_name b
        | _ -> false
      in
      if List.exists expr_refs_name (trans_dynamics_exprs orig_tr.trdyn)
      then Some orig_tr.trname else None
    ) ctx.Expander.orig_transitions in
    if refs <> [] then (
      Fmt.pf ppf "  ";
      Term_style.dim_style Fmt.string ppf "referenced by: ";
      List.iteri (fun i n ->
        if i > 0 then Fmt.pf ppf ", ";
        Term_style.transition Fmt.string ppf n
      ) refs;
      Fmt.pf ppf "@\n"
    )

(* ── --dims ─────────────────────────────────────────────────────────────── *)

let run_dims ppf (model : Ir.model) ctx =
  let dc_result = Dimcheck.check_model model in
  (* Build a lookup: param name → (kind, has_explicit_dim) from AST *)
  let param_info = List.filter_map (fun (p : Ir.parameter) ->
    let ast_decl = decl_of_param ctx p in
    let kind_str = match ast_decl with
      | Some (Ast.PScalar pd)  -> pkind_str pd.pkind
      | Some (Ast.PIndexed pd) -> pkind_str pd.pkind
      | None -> "?"
    in
    (* A dimension is "declared" if the param_kind gives it a known dimension
       (rate, probability, count) or there's an explicit [dim] annotation.
       "positive" and "real" don't declare a dimension — they are inferred. *)
    let declared = match ast_decl with
      | Some (Ast.PScalar pd) ->
        pd.pdim <> None || Ast.(match pd.pkind with
          | PRate | PProbability | PCount | PInstant | PDuration -> true
          | PPositive | PReal -> false)
      | Some (Ast.PIndexed pd) ->
        pd.pdim <> None || Ast.(match pd.pkind with
          | PRate | PProbability | PCount | PInstant | PDuration -> true
          | PPositive | PReal -> false)
      | None -> false
    in
    Some (p.name, kind_str, declared)
  ) model.parameters in
  (* Header *)
  Term_style.bold Fmt.string ppf "parameters (inferred dimensions):";
  Fmt.pf ppf "@\n";
  (* Find the maximum param name length for alignment *)
  let max_name = List.fold_left (fun acc (p : Ir.parameter) ->
    max acc (String.length p.name)
  ) 0 model.parameters in
  let max_kind = List.fold_left (fun acc (_, kind, _) ->
    max acc (String.length kind)
  ) 0 param_info in
  (* Display each parameter *)
  List.iter (fun (p : Ir.parameter) ->
    let name_pad = String.make (max max_name (String.length p.name) - String.length p.name) ' ' in
    let (_, kind_str, declared) = match List.find_opt (fun (n, _, _) -> n = p.name) param_info with
      | Some x -> x | None -> (p.name, "?", false) in
    let kind_pad = String.make (max max_kind (String.length kind_str) - String.length kind_str) ' ' in
    Fmt.pf ppf "  ";
    Term_style.param Fmt.string ppf p.name;
    Fmt.pf ppf "%s" name_pad;
    Term_style.dim_style Fmt.string ppf " : ";
    Fmt.pf ppf "%s%s" kind_str kind_pad;
    (* Look up resolved dimension *)
    (match List.assoc_opt p.name dc_result.param_dims with
     | Some dv ->
       Term_style.dim_style Fmt.string ppf (Printf.sprintf " \xe2\x86\x92 %s" (Dimcheck.formal_dim dv));
       Fmt.pf ppf " (%s)" (Dimcheck.display_dim dv);
       if not declared then
         Term_style.dim_style Fmt.string ppf "  [inferred from context]"
     | None ->
       Term_style.dim_style Fmt.string ppf " \xe2\x86\x92 ?";
       Fmt.pf ppf " (undetermined)");
    Fmt.pf ppf "@\n"
  ) model.parameters

(* ── Tables ──────────────────────────────────────────────────────────────── *)

let dim_of_entry = function
  | TDim d | TDimUnit (d, _) -> d

(** Recover source annotation from the AST table declaration. *)
let table_source_label (td : table_decl) =
  let find_path args =
    List.find_map (fun (kw, e) ->
      if kw = "" then match e with EIdent (s, _) -> Some s | _ -> None
      else None
    ) args
  in
  match td.tvalue with
  | EFuncCall ("read", args)     ->
    (match find_path args with Some p -> `FromFile p | None -> `Inline)
  | EFuncCall ("external", args) ->
    (match find_path args with Some n -> `External n | None -> `External "?")
  | _ -> `Inline

(** Format a compiled table value (always Const f after expansion). *)
let pp_val ppf = function
  | Ir.Const f ->
    let s = Printf.sprintf "%g" f in
    Fmt.string ppf s
  | other ->
    Pp_expr.pp ~mode:Pp_expr.Dsl ~split:Pp_expr.no_split ~ascii:true ppf other

(** Build multi-index label for n-dimensional flat position. *)
let multi_index_label level_lists flat_i =
  let n = List.length level_lists in
  let sizes = List.map List.length level_lists in
  let indices = Array.make n 0 in
  let rem = ref flat_i in
  let strides = Array.make n 1 in
  for k = n - 2 downto 0 do
    strides.(k) <- strides.(k + 1) * List.nth sizes (k + 1)
  done;
  for k = 0 to n - 1 do
    indices.(k) <- !rem / strides.(k);
    rem := !rem mod strides.(k)
  done;
  List.mapi (fun k levels -> List.nth levels indices.(k)) level_lists
  |> String.concat ","

let run_tables ppf (model : Ir.model) ctx (pattern : string option) =
  let faint s  = Term_style.dim_style Fmt.string ppf s in
  let bar ()   = faint "\xe2\x94\x82" in  (* │ *)
  let tables = match pattern with
    | None     -> model.tables
    | Some pat -> List.filter (fun (t : Ir.table) -> glob_match pat t.name) model.tables
  in
  if tables = [] then (
    (match pattern with
     | None     -> Fmt.pf ppf "  (no tables defined)@\n"
     | Some p   -> Fmt.pf ppf "  no tables matching '%s'@\n" p);
    ()
  ) else
  List.iter (fun (t : Ir.table) ->
    let decl_opt = List.find_opt (fun td -> List.mem t.name td.tnames)
                     ctx.Expander.table_decls in
    let tdim_names = match decl_opt with
      | Some td -> List.map dim_of_entry td.tdims
      | None    -> []
    in
    let dim_levels = List.map (fun d ->
      Option.value ~default:[] (Expander.dim_values ctx d)) tdim_names in
    (* ── Header ── *)
    Term_style.bold (Term_style.table Fmt.string) ppf t.name;
    (if tdim_names <> [] then (
      Fmt.pf ppf "  [";
      List.iteri (fun i d ->
        if i > 0 then faint " \xc3\x97 ";   (* × *)
        Term_style.dimension Fmt.string ppf d
      ) tdim_names;
      Fmt.pf ppf "]"
    ));
    (match decl_opt with
     | None    -> ()
     | Some td ->
       match table_source_label td with
       | `Inline     -> faint "  inline"
       | `FromFile p -> faint (Printf.sprintf "  loaded: %s" p)
       | `External _ -> Term_style.warning_style Fmt.string ppf "  runtime: external()");
    Fmt.pf ppf "@\n";
    (* ── Values ── *)
    (match t.source with
     | Ir.External name ->
       Fmt.pf ppf "  (values for '%s' supplied via --table at simulate time)@\n" name
     | Ir.Inline vals ->
       match dim_levels with
       | [] ->
         (match vals with
          | [v] -> Fmt.pf ppf "  "; pp_val ppf v; Fmt.pf ppf "@\n"
          | _   ->
            List.iteri (fun i v ->
              Fmt.pf ppf "  "; bar (); Fmt.pf ppf " [%d]  " i; pp_val ppf v; Fmt.pf ppf "@\n"
            ) vals)
       | [levels] ->
         let w = List.fold_left (fun a s -> max a (String.length s)) 0 levels in
         List.iter2 (fun lv v ->
           let pad = String.make (w - String.length lv) ' ' in
           Fmt.pf ppf "  "; bar (); Fmt.pf ppf " ";
           Term_style.dim_style Fmt.string ppf lv; Fmt.pf ppf "%s  " pad;
           pp_val ppf v; Fmt.pf ppf "@\n"
         ) levels vals
       | [row_lvs; col_lvs] ->
         let nc = List.length col_lvs in
         let nr = List.length row_lvs in
         let val_strs = Array.init (nr * nc) (fun i ->
           Format.asprintf "%a" pp_val (List.nth vals i)) in
         let rw = List.fold_left (fun a s -> max a (String.length s)) 0 row_lvs in
         let cw = List.fold_left (fun a s -> max a (String.length s)) 0 col_lvs in
         let cw = Array.fold_left (fun a s -> max a (String.length s)) cw val_strs in
         (* column header row *)
         Fmt.pf ppf "  "; bar (); Fmt.pf ppf "  %s" (String.make rw ' ');
         List.iter (fun c ->
           let pad = String.make (cw - String.length c) ' ' in
           Fmt.pf ppf "  %s" pad;
           Term_style.dim_style Fmt.string ppf c
         ) col_lvs;
         Fmt.pf ppf "@\n";
         List.iteri (fun ri row ->
           let rpad = String.make (rw - String.length row) ' ' in
           Fmt.pf ppf "  "; bar (); Fmt.pf ppf "  ";
           Term_style.dim_style Fmt.string ppf row; Fmt.pf ppf "%s" rpad;
           List.iteri (fun ci _ ->
             let vs = val_strs.(ri * nc + ci) in
             let pad = String.make (cw - String.length vs) ' ' in
             Fmt.pf ppf "  %s%s" pad vs
           ) col_lvs;
           Fmt.pf ppf "@\n"
         ) row_lvs
       | level_lists ->
         List.iteri (fun i v ->
           let lbl = multi_index_label level_lists i in
           Fmt.pf ppf "  "; bar (); Fmt.pf ppf " [%s]  " lbl;
           pp_val ppf v; Fmt.pf ppf "@\n"
         ) vals);
    Fmt.pf ppf "@\n"
  ) tables

(* ── Main entry point ────────────────────────────────────────────────────── *)

type inspect_cmd =
  | Summary
  | Compartments
  | Parameters
  | Transitions of string option        (* pattern *)
  | TransitionRate of string
  | TransitionCount of string option    (* pattern *)
  | LetBinding of string
  | Dims
  | Tables of string option             (* pattern *)
  | CostReport

type inspect_opts = {
  cmd      : inspect_cmd;
  ir_mode  : bool;   (* --ir: show flat IR names *)
  ascii    : bool;   (* --ascii: no Unicode operators *)
  no_color : bool;   (* --no-color *)
}

let run_inspect path opts =
  let name = Filename.basename path |> Filename.remove_extension in
  let src  = Compiler.read_file path in
  if opts.no_color then (
    Fmt.set_style_renderer Fmt.stdout `None;
    Fmt.set_style_renderer Fmt.stderr `None
  ) else (
    Fmt.set_style_renderer Fmt.stdout `Ansi_tty;
    Fmt.set_style_renderer Fmt.stderr `Ansi_tty
  );
  match Compiler.compile_detail_result ~name ~filename:path src with
  | Error e when e = "compilation failed"
              || (String.length e > 0 && e.[0] = '[') -> exit 1
  | Error e ->
    Fmt.epr "Error: %s@\n" e;
    exit 1
  | Ok { model; ctx; summary; source } ->
    (* compile_detail_result returns Ok only when there are no errors, so any
       remaining diagnostics are warnings/infos — render them and continue. *)
    if ctx.diags.diags <> [] then
      Diagnostics.render_all ctx.diags source Fmt.stderr;
    let ppf = Fmt.stdout in
    (match opts.cmd with
     | Summary ->
       run_summary ppf model ctx summary
     | Compartments ->
       run_compartments ppf model ctx
     | Parameters ->
       run_parameters ppf model ctx
     | Transitions pat ->
       run_transitions ppf model ctx pat ~ascii:opts.ascii
     | TransitionRate name ->
       run_transition_rate ppf model ctx name
     | TransitionCount pat ->
       run_transition_count ppf model ctx pat ~ascii:opts.ascii
     | LetBinding name ->
       run_let ppf ctx name
     | Dims ->
       run_dims ppf model ctx
     | Tables pat ->
       run_tables ppf model ctx pat
     | CostReport ->
       run_cost_report ppf model ctx)

(** Run 'camdl check': the linter. Run the FULL front-end pipeline and report
    the actionable diagnostics — errors (exit 1), warnings, and lints (L4xx) —
    plus a verdict. The structural summary is `camdl inspect --summary`.

    Routed through [Compiler.collect_detail] — the single non-aborting core
    that [compile] and [collect_diagnostics] also use — so `check` runs the
    exact same stages as a real compile (lex → parse → expand → validate →
    dimcheck → lint → autodiff) and can never disagree with it on a model's
    validity. Two earlier divergences (gh#9: dimcheck skipped; gh#170: Validate
    skipped) both came from `check` re-deriving a bespoke sub-pipeline; this
    removes that surface entirely. *)
let run_check path =
  let name = Filename.basename path |> Filename.remove_extension in
  let src  = Compiler.read_file path in
  Fmt.set_style_renderer Fmt.stdout `Ansi_tty;
  Fmt.set_style_renderer Fmt.stderr `Ansi_tty;
  let (detail, diags, source) = Compiler.collect_detail ~name ~filename:path src in
  match detail with
  | None ->
    (* lex/parse/expand structurally failed; [diags] holds the E001.
       Route through [Diagnostics.render] so `check --json-errors` emits
       the JSON array (matching the old front-end-failure path, which
       reached this rendering via [compile_detail_result]). *)
    ignore (Diagnostics.render diags source);
    exit 1
  | Some d ->
    let ctx = d.Compiler.ctx in
    if Diagnostics.has_errors ctx.diags then (
      Diagnostics.render_all ctx.diags source Fmt.stderr;
      exit 1
    );
    (* `check` is the linter: report the actionable diagnostics — warnings and
       lints (L4xx) — and the verdict, nothing else. The structural summary
       (compartments / transitions / …) is `camdl inspect --summary`'s job. *)
    if ctx.diags.diags <> [] then
      Diagnostics.render_all ctx.diags source Fmt.stdout;
    let n_warn = List.length (List.filter
      (fun d -> d.Diagnostics.severity = Diagnostics.Warning) ctx.diags.diags) in
    Fmt.pf Fmt.stdout "@\n  ";
    Term_style.bold Fmt.string Fmt.stdout "\xe2\x9c\x93";  (* ✓ *)
    if n_warn = 0 then
      Fmt.pf Fmt.stdout " no errors, 0 warnings@\n"
    else
      Fmt.pf Fmt.stdout " no errors, %d warning%s@\n"
        n_warn (if n_warn = 1 then "" else "s")
