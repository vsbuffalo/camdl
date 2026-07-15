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
  | ESum (v, _dim, _guard, body) -> Printf.sprintf "\\sum_{%s} %s" v (tex ~prec:7 body)
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
  | ESum (sv, dim, g, body) -> if sv = v then e else ESum (sv, dim, g, subst v level body)
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
    | other ->
      Printf.eprintf "camdlc render: unknown --format '%s' (want json|document)\n" other;
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
    prerr_endline "usage: camdlc render [--format json|document] [--expand DIM[,DIM]] FILE.camdl";
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
       let rm = render_model ~name ~expand:!expand decls in
       print_string (match !format with `Json -> to_json rm | `Document -> to_document rm))
