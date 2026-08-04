(* LaTeX rendering of a camdl model — the mathematician's (indexed) form.
   A read-only projection of the AST, the pre-expansion representation: index
   variables (S[a]), reduction sums, and un-inlined `let` aggregates all survive
   here, because this taps the model BEFORE `expander.ml` flattens it. Never
   touches the IR or execution. Sibling to `pp_expr.ml` (which prints Ir.expr,
   the expanded side). *)

open Ast

(* ── Symbol mapping ─────────────────────────────────────────────────────────── *)

let greek =
  [ "alpha"; "beta"; "gamma"; "delta"; "epsilon"; "zeta"; "eta"; "theta";
    "iota"; "kappa"; "lambda"; "mu"; "nu"; "xi"; "pi"; "rho"; "sigma";
    "tau"; "upsilon"; "phi"; "chi"; "psi"; "omega" ]

let upper_greek =
  [ "Gamma"; "Delta"; "Theta"; "Lambda"; "Xi"; "Pi"; "Sigma"; "Phi"; "Psi"; "Omega" ]

let is_digit c = c >= '0' && c <= '9'
let is_lower c = c >= 'a' && c <= 'z'
let is_upper c = c >= 'A' && c <= 'Z'

(* "r1" -> ("r", ["1"]) ; "N0" -> ("N", ["0"]) ; "beta" -> ("beta", []) *)
let split_trailing_digits s =
  let n = String.length s in
  let i = ref n in
  while !i > 0 && is_digit s.[!i - 1] do decr i done;
  if !i = n || !i = 0 then (s, []) else (String.sub s 0 !i, [ String.sub s !i (n - !i) ])

(* "Iv" -> Some ("I", "v") — a single capital then lowercase (a compartment like
   Sv/Ev/Iv that means I_v). *)
let split_cap_lower s =
  if String.length s >= 2 && is_upper s.[0]
     && String.for_all is_lower (String.sub s 1 (String.length s - 1))
  then Some (String.sub s 0 1, String.sub s 1 (String.length s - 1))
  else None

let escape s = String.concat "\\_" (String.split_on_char '_' s)

(* `@symbol` overrides, keyed by declared name (a compartment / parameter with a
   `#' @symbol …` block). Populated per render; empty means "auto only". *)
let overrides : (string, string) Hashtbl.t = Hashtbl.create 16

(* A camdl identifier -> (base symbol, intrinsic subscripts). Order: an explicit
   `@symbol` wins; then greek (upper + lower), single letters, the `Iv -> I_v`
   capital-lower split, and finally a multi-word name set upright in \mathrm.
   The *parts* form lets callers MERGE an index (S[a]) into ONE subscript rather
   than emitting an invalid double subscript `_{1}_{a}`. *)
let sym_parts (name : string) : string * string list =
  match Hashtbl.find_opt overrides name with
  | Some s -> (s, []) (* modeler-supplied symbol is authoritative *)
  | None ->
    let head, tail =
      match String.split_on_char '_' name with b :: r -> (b, r) | [] -> (name, [])
    in
    let head, dsub = split_trailing_digits head in
    let subs = dsub @ tail in
    if List.mem head upper_greek then ("\\" ^ head, subs)
    else if List.mem head greek then ("\\" ^ head, subs)
    else if String.length head = 1 then (head, subs)
    else if tail = [] && dsub = [] then
      (match split_cap_lower head with
       | Some (b, s) -> (b, [ s ]) (* Iv -> I_v *)
       | None -> ("\\mathrm{" ^ escape name ^ "}", []))
    else ("\\mathrm{" ^ escape name ^ "}", [] (* bite_rate -> \mathrm{bite\_rate} *))

let subs_join = function [] -> "" | subs -> Printf.sprintf "_{%s}" (String.concat "," subs)

(* base symbol with its intrinsic subscripts and an optional extra index string,
   all in ONE subscript group: comp_sym "Y1" "a" -> "Y_{1,a}". *)
let comp_sym name index_str =
  let base, isubs = sym_parts name in
  base ^ subs_join (isubs @ if index_str = "" then [] else [ index_str ])

let sym name = comp_sym name ""

let fmt_num f =
  if Float.is_integer f then string_of_int (int_of_float f) else Printf.sprintf "%g" f

(* ── Expressions ────────────────────────────────────────────────────────────── *)

let cmp_sym = function
  | Eq -> "=" | Neq -> "\\neq" | Lt -> "<" | Gt -> ">" | Le -> "\\le" | Ge -> "\\ge"
  | Add | Sub | Mul | Div | Pow -> "?" (* unreachable: arithmetic handled above *)

(* Flatten a product/quotient tree into (numerator, denominator) factors so
   `beta * S * (I / N)` renders as ONE \frac, not nested. *)
let rec flatten_product (e : expr) : expr list * expr list =
  match e with
  | EBinOp (Mul, l, r) ->
    let ln, ld = flatten_product l and rn, rd = flatten_product r in
    (ln @ rn, ld @ rd)
  | EBinOp (Div, l, r) ->
    let ln, ld = flatten_product l and rn, rd = flatten_product r in
    (ln @ rd, ld @ rn)
  | atom -> ([ atom ], [])

let rec tex ?(prec = 0) (e : expr) : string =
  let paren p s = if p < prec then "\\left(" ^ s ^ "\\right)" else s in
  match e with
  | EConst f -> fmt_num f
  | EUnit (f, _) -> fmt_num f (* a rate literal like 0.5 'per_day — show the number *)
  | EIdent (n, _) -> sym n
  | EIndex (n, items, _) -> comp_sym n (index_tex items)
  | ESum (v, _dim, _guard, body, _) -> Printf.sprintf "\\sum_{%s} %s" v (tex ~prec:7 body)
  | EBinOp ((Mul | Div), _, _) ->
    let num, den = flatten_product e in
    let render fs = String.concat "\\," (List.map (tex ~prec:7) fs) in
    if den = [] then paren 7 (render num)
    else Printf.sprintf "\\frac{%s}{%s}" (render num) (render den)
  | EBinOp (Pow, l, r) -> Printf.sprintf "%s^{%s}" (tex ~prec:9 l) (tex r)
  | EBinOp (Add, l, r) -> paren 6 (Printf.sprintf "%s + %s" (tex ~prec:6 l) (tex ~prec:6 r))
  | EBinOp (Sub, l, r) -> paren 6 (Printf.sprintf "%s - %s" (tex ~prec:6 l) (tex ~prec:6 r))
  | EBinOp (((Eq | Neq | Lt | Gt | Le | Ge) as op), l, r) ->
    Printf.sprintf "%s %s %s" (tex l) (cmp_sym op) (tex r)
  | EUnOp (op, a) -> un_tex op a
  | ECond (p, a, b) ->
    Printf.sprintf "\\begin{cases} %s & \\text{if } %s \\\\ %s & \\text{otherwise} \\end{cases}"
      (tex a) (tex p) (tex b)
  | EFuncCall (f, args) ->
    let a = String.concat ",\\, " (List.map (fun (_, e) -> tex e) args) in
    (match f with
     | "min" | "max" -> Printf.sprintf "\\%s\\!\\left(%s\\right)" f a
     | _ -> Printf.sprintf "\\mathrm{%s}\\!\\left(%s\\right)" (escape f) a)
  | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> "\\,\\cdot\\," (* not valid in a rate *)

and index_tex (items : index_item list) : string =
  String.concat ","
    (List.map (function IPosn e -> tex e | INamed (_, e) -> tex e) items)

and un_tex op a =
  match op with
  | Neg -> "-" ^ tex ~prec:9 a
  | Exp -> Printf.sprintf "e^{%s}" (tex a)
  | Log -> Printf.sprintf "\\ln\\!\\left(%s\\right)" (tex a)
  | Sqrt -> Printf.sprintf "\\sqrt{%s}" (tex a)
  | Abs -> Printf.sprintf "\\left|%s\\right|" (tex a)
  | Floor -> Printf.sprintf "\\lfloor %s \\rfloor" (tex a)
  | Ceil -> Printf.sprintf "\\lceil %s \\rceil" (tex a)
  | Sin -> Printf.sprintf "\\sin\\!\\left(%s\\right)" (tex a)
  | Cos -> Printf.sprintf "\\cos\\!\\left(%s\\right)" (tex a)
  | Tanh -> Printf.sprintf "\\tanh\\!\\left(%s\\right)" (tex a)

(* ── Partial expansion (`--expand dim`) ─────────────────────────────────────── *)

(* Substitute a binder variable with a concrete level throughout an expression —
   the one operation that turns `X[a]` into `X[child]`. Reused to enumerate a
   dimension the reader asked to spell out, while leaving other dimensions
   symbolic. *)
let rec subst (v : string) (level : string) (e : expr) : expr =
  match e with
  | EIdent (n, loc) when n = v -> EIdent (level, loc)
  | EIndex (n, items, loc) -> EIndex (n, List.map (subst_item v level) items, loc)
  | EBinOp (op, l, r) -> EBinOp (op, subst v level l, subst v level r)
  | EUnOp (op, a) -> EUnOp (op, subst v level a)
  | ESum (sv, dim, g, body, l) -> if sv = v then e else ESum (sv, dim, g, subst v level body, l)
  | ECond (p, a, b) -> ECond (subst v level p, subst v level a, subst v level b)
  | EFuncCall (f, args) -> EFuncCall (f, List.map (fun (k, e) -> (k, subst v level e)) args)
  | EList es -> EList (List.map (subst v level) es)
  | ERange (a, b) -> ERange (subst v level a, subst v level b)
  | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> e

and subst_item v level = function
  | IPosn e -> IPosn (subst v level e)
  | INamed (d, e) -> INamed (d, subst v level e)

let subst_ref v level ((n, items) : stoich_ref) : stoich_ref =
  (n, List.map (subst_item v level) items)

(* Expand every binder over a dimension the reader requested (dim -> levels),
   cartesian over multiple such binders; binders over other dimensions stay. *)
let expand_transition (expand : (string * string list) list) (t : transition_decl) : transition_decl list =
  let to_expand, keep =
    List.partition (function IBind (_, d) -> List.mem_assoc d expand | _ -> false) t.trindices
  in
  let apply v level (t : transition_decl) : transition_decl =
    { t with
      trsrc = List.map (subst_ref v level) t.trsrc;
      trdst =
        (match t.trdst with
         | DstSum refs -> DstSum (List.map (subst_ref v level) refs)
         | DstBranch bs -> DstBranch (List.map (fun (r, w) -> (subst_ref v level r, subst v level w)) bs));
      trdyn =
        (match t.trdyn with
         | Rate e -> Rate (subst v level e)
         | Via (l, args) -> Via (l, List.map (fun (k, e) -> (k, subst v level e)) args)) }
  in
  let rec go binders t =
    match binders with
    | [] -> [ t ]
    | IBind (v, d) :: rest ->
      List.concat_map (fun level -> go rest (apply v level t)) (List.assoc d expand)
    | _ :: rest -> go rest t
  in
  List.map (fun t -> { t with trindices = keep }) (go to_expand t)

(* ── Transitions ────────────────────────────────────────────────────────────── *)

let comp_ref ((name, items) : stoich_ref) : string = comp_sym name (index_tex items)

let rate_tex (d : trans_dynamics) : string =
  match d with
  | Rate e -> tex e
  | Via (law, _) -> Printf.sprintf "\\text{via } \\mathrm{%s}" (escape law)

(* The three renderable parts of a transition — reactants, products, rate —
   kept separate so a web consumer can lay them out as a reaction table, while
   the document assembles them into the \xrightarrow arrow. *)
let reaction_parts (t : transition_decl) : string * string * string =
  let lhs = String.concat " + " (List.map comp_ref t.trsrc) in
  let rhs =
    match t.trdst with
    | DstSum refs -> String.concat " + " (List.map comp_ref refs)
    | DstBranch bs ->
      String.concat ",\\ "
        (List.map (fun (r, w) -> Printf.sprintf "%s\\,(%s)" (comp_ref r) (tex w)) bs)
  in
  ( (if lhs = "" then "\\varnothing" else lhs),
    (if rhs = "" then "\\varnothing" else rhs),
    rate_tex t.trdyn )

let reaction (t : transition_decl) : string =
  let reactants, products, rate = reaction_parts t in
  Printf.sprintf "%s &\\xrightarrow{\\; %s \\;} %s" reactants rate products

(* ── Derived dynamics (assembled from stoichiometry, kept indexed) ──────────── *)

let transitions_of = List.concat_map (function DTransitions ts -> ts | _ -> [])
let compartments_of = List.concat_map (function DCompartments cs -> cs | _ -> [])
let lets_of = List.filter_map (function DLet l -> Some l | _ -> None)
let dims_of = List.concat_map (function DDimensions ds -> ds | _ -> [])

let dst_refs = function DstSum refs -> refs | DstBranch bs -> List.map fst bs

(* For a compartment family, one line \dot{X}_a = <signed rates>, collected over
   every transition that produces (+) or consumes (-) it. A catalyst (same
   family on both sides) nets to zero and is skipped. The free index is the
   binder var of a contributing transition, kept symbolic — so the strata are
   quantified away rather than enumerated. *)
(* Distinct compartment instances (base + rendered index) referenced by the
   transitions, in first-occurrence order. Symbolic mode gives one per family
   (X[a]); under --expand, one per enumerated cell (X[child], X[adult]). Deduped
   by (base, rendered index) so source locations never split an instance. *)
let comp_instances (transitions : transition_decl list) (comps : compartment_decl list) : stoich_ref list =
  let names = List.map (fun (c : compartment_decl) -> c.cname) comps in
  let seen = Hashtbl.create 16 in
  List.concat_map (fun t -> t.trsrc @ dst_refs t.trdst) transitions
  |> List.filter_map (fun ((n, items) as r) ->
         if not (List.mem n names) then None
         else
           let key = n ^ "|" ^ index_tex items in
           if Hashtbl.mem seen key then None else (Hashtbl.add seen key (); Some r))

(* One line \dot{X}_i = <signed rates> for a compartment instance, over every
   transition that produces (+) or consumes (-) exactly that instance; a catalyst
   (same instance both sides) nets to zero and is dropped. *)
(* The compartment's derivative split into its LHS (`\dot{X}_i`) and signed-rate
   RHS, plus the compartment base name — so the document assembles the aligned
   `\dot{X}_i &= …` while the web shape keys each equation by its state. *)
let derived_ode_parts (transitions : transition_decl list) ((base, items) : stoich_ref) :
    string * string * string =
  let idx_str = index_tex items in
  let matches (n, its) = n = base && index_tex its = idx_str in
  let terms =
    List.filter_map
      (fun t ->
        let in_src = List.exists matches t.trsrc in
        let in_dst = List.exists matches (dst_refs t.trdst) in
        match (in_src, in_dst) with
        | true, false -> Some (false, rate_tex t.trdyn) (* consumed *)
        | false, true -> Some (true, rate_tex t.trdyn) (* produced *)
        | _ -> None (* untouched, or catalyst (net zero) *))
      transitions
  in
  let bt, isubs = sym_parts base in
  let dot =
    Printf.sprintf "\\dot{%s}%s" bt (subs_join (isubs @ if idx_str = "" then [] else [ idx_str ]))
  in
  let rhs =
    match terms with
    | [] -> "0"
    | _ ->
      String.concat " "
        (List.mapi
           (fun i (plus, r) ->
             if i = 0 then if plus then r else "-" ^ r
             else if plus then "+ " ^ r
             else "- " ^ r)
           terms)
  in
  (base, dot, rhs)

let derived_ode (transitions : transition_decl list) (inst : stoich_ref) : string =
  let _, dot, rhs = derived_ode_parts transitions inst in
  Printf.sprintf "%s &= %s" dot rhs

(* ── Document ───────────────────────────────────────────────────────────────── *)

(* Harvest `#' @symbol …` blocks off compartments and parameters into the
   override table, so the modeler controls the symbol where the auto-heuristic
   would guess (a FOI written \Lambda, a rate written \beta_h). *)
let populate_overrides (decls : declaration list) : unit =
  Hashtbl.clear overrides;
  let add name = function Some { d_symbol = Some s; _ } -> Hashtbl.replace overrides name s | _ -> () in
  List.iter
    (function
      | DCompartments cs -> List.iter (fun (c : compartment_decl) -> add c.cname c.cdoc) cs
      | DParameters ps ->
        List.iter
          (function PScalar { pname; pdoc; _ } | PIndexed { pname; pdoc; _ } -> add pname pdoc)
          ps
      | _ -> ())
    decls

(* ── Structured render: one record, two projections ────────────────────────── *)

type r_param = { rp_name : string; rp_symbol : string; rp_desc : string option }
type r_transition = { rt_name : string; rt_reactants : string; rt_products : string; rt_rate : string }
type r_definition = { rd_name : string; rd_lhs : string; rd_body : string }
type r_dynamics = { ry_state : string; ry_lhs : string; ry_rhs : string }

(* The model rendered to LaTeX, split by section with every equation its own
   KaTeX-renderable string. `to_document` assembles the standalone `.tex`;
   `to_json` emits the web/display shape (`model.render.json`). *)
type rendered_model = {
  rm_name : string;
  rm_mode : string;                       (* "indexed" | "expanded over …" *)
  rm_states : string list;                (* compartment names *)
  rm_dims : (string * string list) list;  (* dimension name, ordered levels *)
  rm_params : r_param list;
  rm_definitions : r_definition list;
  rm_transitions : r_transition list;
  rm_dynamics : r_dynamics list;
}

let param_entry = function
  | PScalar { pname; pdoc; _ } | PIndexed { pname; pdoc; _ } ->
    { rp_name = pname;
      rp_symbol = sym pname;
      rp_desc = (match pdoc with Some { d_text; _ } -> d_text | None -> None) }

let render_model ?(name = "model") ?(expand = []) (decls : declaration list) : rendered_model =
  populate_overrides decls;
  let dims = dims_of decls in
  let lets = lets_of decls in
  let comps = compartments_of decls in
  let expand_levels =
    List.filter_map
      (fun (de : dimensions_entry) ->
        match de.desrc with
        | DInline levels when List.mem de.dename expand -> Some (de.dename, levels)
        | _ -> None)
      dims
  in
  let transitions =
    let raw = transitions_of decls in
    if expand_levels = [] then raw else List.concat_map (expand_transition expand_levels) raw
  in
  let mode =
    if expand_levels = [] then "indexed"
    else "expanded over " ^ String.concat ", " (List.map fst expand_levels)
  in
  let binder_var = function IBind (v, _) | IConsec (v, _, _) -> v | IComp v -> v in
  {
    rm_name = name;
    rm_mode = mode;
    rm_states = List.map (fun (c : compartment_decl) -> c.cname) comps;
    rm_dims =
      List.filter_map
        (fun (de : dimensions_entry) ->
          match de.desrc with DInline levels -> Some (de.dename, levels) | DRead _ -> None)
        dims;
    rm_params = List.concat_map (function DParameters ps -> List.map param_entry ps | _ -> []) decls;
    rm_definitions =
      List.map
        (fun (l : let_binding) ->
          let idx = String.concat "," (List.map binder_var l.lindices) in
          { rd_name = l.lname; rd_lhs = comp_sym l.lname idx; rd_body = tex l.lbody })
        lets;
    rm_transitions =
      List.map
        (fun (t : transition_decl) ->
          let reactants, products, rate = reaction_parts t in
          { rt_name = t.trname; rt_reactants = reactants; rt_products = products; rt_rate = rate })
        transitions;
    rm_dynamics =
      List.map
        (fun inst ->
          let state, lhs, rhs = derived_ode_parts transitions inst in
          { ry_state = state; ry_lhs = lhs; ry_rhs = rhs })
        (comp_instances transitions comps);
  }

let to_document (rm : rendered_model) : string =
  let buf = Buffer.create 4096 in
  let p fmt = Printf.ksprintf (Buffer.add_string buf) fmt in
  p "\\documentclass[12pt]{article}\n";
  p "\\usepackage[utf8]{inputenc}\n\\usepackage{amsmath,amssymb}\n";
  p "\\usepackage[margin=1in]{geometry}\n\\pagestyle{empty}\n";
  p "\\begin{document}\n";
  p "\\begin{center}\\Large\\textbf{Model: \\texttt{%s}}\\end{center}\n\\vspace{0.5em}\n\n" (escape rm.rm_name);
  List.iter
    (fun (name, levels) ->
      p "\\noindent\\textbf{Dimension} $\\mathrm{%s} = \\{%s\\}$.\\par\\medskip\n" (escape name)
        (String.concat ",\\ " (List.map (fun l -> "\\text{" ^ l ^ "}") levels)))
    rm.rm_dims;
  if rm.rm_definitions <> [] then begin
    p "\\noindent\\textbf{Definitions}\n\\begin{align*}\n";
    p "%s\n"
      (String.concat " \\\\\n"
         (List.map (fun d -> Printf.sprintf "%s &= %s" d.rd_lhs d.rd_body) rm.rm_definitions));
    p "\\end{align*}\n\n"
  end;
  p "\\noindent\\textbf{Transitions}\n\\begin{align*}\n";
  p "%s\n"
    (String.concat " \\\\\n"
       (List.map
          (fun t -> Printf.sprintf "%s &\\xrightarrow{\\; %s \\;} %s" t.rt_reactants t.rt_rate t.rt_products)
          rm.rm_transitions));
  p "\\end{align*}\n\n";
  p "\\noindent\\textbf{Derived dynamics} \\; \\small(%s; assembled from stoichiometry)\\normalsize\n" rm.rm_mode;
  p "\\begin{align*}\n";
  p "%s\n"
    (String.concat " \\\\\n"
       (List.map (fun d -> Printf.sprintf "%s &= %s" d.ry_lhs d.ry_rhs) rm.rm_dynamics));
  p "\\end{align*}\n";
  p "\\end{document}\n";
  Buffer.contents buf

(* The web/display shape a KaTeX consumer renders — every math string is a
   standalone expression; transitions are split into parts for a reaction table. *)
let to_json (rm : rendered_model) : string =
  let str s : Yojson.Safe.t = `String s in
  let opt_desc = function Some d -> [ ("description", str d) ] | None -> [] in
  let j : Yojson.Safe.t =
    `Assoc
      [ ("model", str rm.rm_name);
        ("mode", str rm.rm_mode);
        ("states", `List (List.map str rm.rm_states));
        ( "dimensions",
          `List
            (List.map
               (fun (n, ls) -> `Assoc [ ("name", str n); ("levels", `List (List.map str ls)) ])
               rm.rm_dims) );
        ( "parameters",
          `List
            (List.map
               (fun p -> `Assoc ([ ("name", str p.rp_name); ("symbol", str p.rp_symbol) ] @ opt_desc p.rp_desc))
               rm.rm_params) );
        ( "definitions",
          `List
            (List.map
               (fun d -> `Assoc [ ("name", str d.rd_name); ("tex", str (d.rd_lhs ^ " = " ^ d.rd_body)) ])
               rm.rm_definitions) );
        ( "transitions",
          `List
            (List.map
               (fun t ->
                 `Assoc
                   [ ("name", str t.rt_name);
                     ("reactants", str t.rt_reactants);
                     ("products", str t.rt_products);
                     ("rate", str t.rt_rate) ])
               rm.rm_transitions) );
        ( "dynamics",
          `List
            (List.map
               (fun d -> `Assoc [ ("state", str d.ry_state); ("tex", str (d.ry_lhs ^ " = " ^ d.ry_rhs)) ])
               rm.rm_dynamics) );
      ]
  in
  Yojson.Safe.pretty_to_string j

let of_model ?(name = "model") ?(expand = []) (decls : declaration list) : string =
  to_document (render_model ~name ~expand decls)

(* ── Structured render: the compartmental flow graph (model.graph.json) ──────
   The id-based sibling of `to_json`'s LaTeX transitions: a structured node/edge
   graph a viewer can lay out as a compartmental flow diagram. Like the rest of
   this module it is a pure projection of the pre-expansion `Ast` — base
   compartments are the nodes, dimensions are the plates, transitions are the
   edges — so a stratified model stays compact (one edge per transition family,
   not one per stratum). *)

type g_node    = { gn_id : string; gn_label : string }
type g_plate   = { gp_name : string; gp_levels : string list }
type g_edge    = {
  ge_id         : string;
  ge_from       : string option;   (* None ⇒ exogenous inflow (birth) *)
  ge_to         : string option;   (* None ⇒ outflow (death) *)
  ge_rate       : string;          (* KaTeX rate string, as in the reaction table *)
  ge_advances   : string option;   (* the plate a consecutive(dim) binder steps along *)
  ge_reads_pool : bool;            (* rate reads a full-dimension aggregate (mean-field) *)
}
type g_coupling = { gc_edge : string; gc_aggregate : string; gc_over : string list }

type model_graph = {
  mg_name      : string;
  mg_nodes     : g_node list;
  mg_plates    : g_plate list;
  mg_edges     : g_edge list;
  mg_couplings : g_coupling list;
}

let index_item_expr = function IPosn e | INamed (_, e) -> e

(* Dimensions summed over by every `sum(i in dim, …)` (ESum) node anywhere in
   [e], WITHOUT resolving through `let` bindings — nested sums are flattened
   (`sum(a, sum(m, …))` → [a; m]), first-occurrence order, deduped. This is the
   "full-dimension aggregate" predicate the mean-field pool detection is built
   on: a `let` whose body has any ESum is a pool binding. *)
let sum_dims_in (e : expr) : string list =
  let acc = ref [] in
  let add d = if not (List.mem d !acc) then acc := !acc @ [ d ] in
  let rec go = function
    | ESum (_, dim, _, body, _) -> add dim; go body
    | EBinOp (_, l, r) -> go l; go r
    | EUnOp (_, a) -> go a
    | ECond (p, a, b) -> go p; go a; go b
    | EFuncCall (_, args) -> List.iter (fun (_, e) -> go e) args
    | EIndex (_, items, _) -> List.iter (fun it -> go (index_item_expr it)) items
    | EList es -> List.iter go es
    | ERange (a, b) -> go a; go b
    | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> ()
  in
  go e; !acc

(* `let`-binding name → the dimensions its body aggregates over, for every
   binding that IS a pool (its body contains a `sum`). ctl_bb: inf_vil → [age],
   Nvil → [age]; ajura: those over [age; imm; compound]. *)
let pool_bindings (lets : let_binding list) : (string, string list) Hashtbl.t =
  let tbl = Hashtbl.create 16 in
  List.iter
    (fun (l : let_binding) ->
      match sum_dims_in l.lbody with [] -> () | dims -> Hashtbl.replace tbl l.lname dims)
    lets;
  tbl

let let_body_table (lets : let_binding list) : (string, expr) Hashtbl.t =
  let tbl = Hashtbl.create 16 in
  List.iter (fun (l : let_binding) -> Hashtbl.replace tbl l.lname l.lbody) lets;
  tbl

(* The mean-field couplings a rate expression reads, resolving references
   through *transparent* (non-pool) `let` bindings and terminating at either a
   pool binding (aggregate = the binding's name, e.g. "inf_vil") or an inline
   `sum(…)` in the rate itself (aggregate = "sum"). `over` is the summed
   dimension(s). Deduped. A `where`-restricted sum is still reported (its guard
   is not distinguished in v1). *)
let rate_couplings (pools : (string, string list) Hashtbl.t)
    (lets : (string, expr) Hashtbl.t) (rate : expr) : (string * string list) list =
  let acc = ref [] in
  let add name dims =
    if not (List.mem (name, dims) !acc) then acc := !acc @ [ (name, dims) ]
  in
  let rec go visited (e : expr) =
    match e with
    | ESum _ -> add "sum" (sum_dims_in e) (* inline aggregate; sum_dims_in already descends *)
    | EIdent (n, _) ->
      if Hashtbl.mem pools n then add n (Hashtbl.find pools n)
      else if Hashtbl.mem lets n && not (List.mem n visited) then
        go (n :: visited) (Hashtbl.find lets n)
    | EIndex (n, items, _) ->
      (if Hashtbl.mem pools n then add n (Hashtbl.find pools n)
       else if Hashtbl.mem lets n && not (List.mem n visited) then
         go (n :: visited) (Hashtbl.find lets n));
      List.iter (fun it -> go visited (index_item_expr it)) items
    | EBinOp (_, l, r) -> go visited l; go visited r
    | EUnOp (_, a) -> go visited a
    | ECond (p, a, b) -> go visited p; go visited a; go visited b
    | EFuncCall (_, args) -> List.iter (fun (_, e) -> go visited e) args
    | EList es -> List.iter (go visited) es
    | ERange (a, b) -> go visited a; go visited b
    | EConst _ | EUnit _ | EObsAccess _ | ERunMember _ -> ()
  in
  go [] rate; !acc

(* The plate a transition steps along: the dim of its `consecutive(dim)` binder
   (aging `(a, a_next) in consecutive(age)`, immunity `consecutive(imm)`). *)
let advances_of (t : transition_decl) : string option =
  List.find_map (function IConsec (_, _, dim) -> Some dim | _ -> None) t.trindices

(* Base compartment names of a stoichiometry side, first-occurrence, deduped.
   NOTE these are the names *as written*: an ordinary transition yields a
   declared compartment (S_naive), while the family transitions `aging` / `death`
   move a compartment-iteration binder (`c[…] --> c[…]` over `c in compartments`),
   so `from`/`to` there is the iterator variable `c` — signalling "every node".
   We do NOT net-cancel same-name endpoints: `c[a] --> c[a_next]` is a real
   move along the age plate (a self-loop with `advances=age`), not a no-op. *)
let ref_bases (refs : stoich_ref list) : string list =
  let seen = Hashtbl.create 8 in
  List.filter_map
    (fun (n, _) -> if Hashtbl.mem seen n then None else (Hashtbl.add seen n (); Some n))
    refs

(* One transition → its edges and their couplings. A single-source, single-
   destination transition is one edge keyed by the transition name; a multi-
   destination (branch or multi-`+` arrow) or multi-source transition fans out
   to one edge per (source, destination) pair with a disambiguated id. *)
let transition_graph (pools : (string, string list) Hashtbl.t)
    (lets : (string, expr) Hashtbl.t) (t : transition_decl) : g_edge list * g_coupling list =
  let advances = advances_of t in
  let base_rate = rate_tex t.trdyn in
  let couplings =
    List.concat_map (rate_couplings pools lets) (trans_dynamics_exprs t.trdyn)
    |> List.fold_left (fun acc c -> if List.mem c acc then acc else acc @ [ c ]) []
  in
  let reads_pool = couplings <> [] in
  let from_opts =
    match ref_bases t.trsrc with [] -> [ None ] | xs -> List.map (fun x -> Some x) xs
  in
  let to_specs =
    match t.trdst with
    | DstSum refs ->
      (match ref_bases refs with [] -> [ (None, None) ] | xs -> List.map (fun x -> (Some x, None)) xs)
    | DstBranch bs -> List.map (fun ((n, _), w) -> (Some n, Some w)) bs
  in
  let n_edges = List.length from_opts * List.length to_specs in
  let name_or dflt = function Some s -> s | None -> dflt in
  let edges =
    List.concat_map
      (fun from_opt ->
        List.map
          (fun (to_opt, w) ->
            let id =
              if n_edges = 1 then t.trname
              else
                Printf.sprintf "%s__%s__%s" t.trname (name_or "src" from_opt) (name_or "sink" to_opt)
            in
            let rate =
              match w with
              | None -> base_rate
              | Some w -> Printf.sprintf "%s \\cdot %s" (tex w) base_rate
            in
            { ge_id = id; ge_from = from_opt; ge_to = to_opt; ge_rate = rate;
              ge_advances = advances; ge_reads_pool = reads_pool })
          to_specs)
      from_opts
  in
  let couplings_out =
    List.concat_map
      (fun (e : g_edge) ->
        List.map (fun (agg, over) -> { gc_edge = e.ge_id; gc_aggregate = agg; gc_over = over }) couplings)
      edges
  in
  (edges, couplings_out)

let build_graph ?(name = "model") (decls : declaration list) : model_graph =
  populate_overrides decls;
  let comps = compartments_of decls in
  let dims = dims_of decls in
  let pools = pool_bindings (lets_of decls) in
  let lets = let_body_table (lets_of decls) in
  let per_transition = List.map (transition_graph pools lets) (transitions_of decls) in
  {
    mg_name = name;
    mg_nodes = List.map (fun (c : compartment_decl) -> { gn_id = c.cname; gn_label = sym c.cname }) comps;
    mg_plates =
      List.map
        (fun (de : dimensions_entry) ->
          match de.desrc with
          | DInline levels -> { gp_name = de.dename; gp_levels = levels }
          | DRead _ -> { gp_name = de.dename; gp_levels = [] })
        dims;
    mg_edges = List.concat_map fst per_transition;
    mg_couplings = List.concat_map snd per_transition;
  }

(* The flow-graph shape a diagram consumer renders: nodes = base compartments,
   plates = dimensions, edges = transitions (`from`/`to` null ⇒ birth/death),
   couplings = the mean-field pools each edge reads. *)
let to_graph_json (g : model_graph) : string =
  let str s : Yojson.Safe.t = `String s in
  let opt = function Some s -> str s | None -> `Null in
  let j : Yojson.Safe.t =
    `Assoc
      [ ("model", str g.mg_name);
        ("nodes", `List (List.map (fun n -> `Assoc [ ("id", str n.gn_id); ("label", str n.gn_label) ]) g.mg_nodes));
        ( "plates",
          `List
            (List.map
               (fun p -> `Assoc [ ("name", str p.gp_name); ("levels", `List (List.map str p.gp_levels)) ])
               g.mg_plates) );
        ( "edges",
          `List
            (List.map
               (fun e ->
                 `Assoc
                   [ ("id", str e.ge_id);
                     ("from", opt e.ge_from);
                     ("to", opt e.ge_to);
                     ("rate", str e.ge_rate);
                     ("advances", opt e.ge_advances);
                     ("reads_pool", `Bool e.ge_reads_pool) ])
               g.mg_edges) );
        ( "couplings",
          `List
            (List.map
               (fun c ->
                 `Assoc
                   [ ("edge", str c.gc_edge);
                     ("aggregate", str c.gc_aggregate);
                     ("over", `List (List.map str c.gc_over)) ])
               g.mg_couplings) );
      ]
  in
  Yojson.Safe.pretty_to_string j

(* ── Subcommand driver: `camdlc render FILE.camdl` ─────────────────────────── *)

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  s

let run (args : string list) : unit =
  let expand = ref [] and files = ref [] and format = ref `Document in
  let add_dims s = expand := !expand @ List.filter (( <> ) "") (String.split_on_char ',' s) in
  let set_format = function
    | "json" -> format := `Json
    | "document" | "latex" | "tex" -> format := `Document
    | "graph" -> format := `Graph
    | other ->
      Printf.eprintf "camdlc render: unknown --format '%s' (want json|document|graph)\n" other;
      exit 2
  in
  let rec parse = function
    | [] -> ()
    | "--expand" :: dims :: tl -> add_dims dims; parse tl
    | a :: tl when String.length a >= 9 && String.sub a 0 9 = "--expand=" ->
      add_dims (String.sub a 9 (String.length a - 9)); parse tl
    | "--format" :: fmt :: tl -> set_format fmt; parse tl
    | a :: tl when String.length a >= 9 && String.sub a 0 9 = "--format=" ->
      set_format (String.sub a 9 (String.length a - 9)); parse tl
    | a :: _ when String.length a > 0 && a.[0] = '-' ->
      Printf.eprintf "camdlc render: unknown flag '%s'\n" a; exit 2
    | f :: tl -> files := f :: !files; parse tl
  in
  parse args;
  match List.rev !files with
  | [] ->
    prerr_endline "usage: camdlc render [--format json|document|graph] [--expand DIM[,DIM]] FILE.camdl";
    exit 2
  | file :: _ ->
    let src = read_file file in
    let lexbuf = Lexing.from_string src in
    Lexing.set_filename lexbuf file;
    (match try Ok (Parser.file Lexer.token lexbuf) with _ -> Error "parse error" with
     | Error msg ->
       Printf.eprintf "camdlc render: %s in %s\n" msg file;
       exit 1
     | Ok decls ->
       let name = Filename.remove_extension (Filename.basename file) in
       (* The flow graph is a pre-expansion projection (one edge per transition
          family), so `--expand` — which unfolds strata for the LaTeX views —
          does not apply to it. *)
       print_string
         (match !format with
          | `Json -> to_json (render_model ~name ~expand:!expand decls)
          | `Document -> to_document (render_model ~name ~expand:!expand decls)
          | `Graph -> to_graph_json (build_graph ~name decls)))
