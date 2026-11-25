(* The declared-model section the compiler emits (`Declared_emit`), checked on
   every compiled golden model and on small sources written for the purpose.

   "goldens" compiles every `.camdl` under `ocaml/golden` and
   `tests/fixtures/*/` and checks, on the section each produces:

   - invariant 10 of docs/dev/proposals/2026-11-13-model-structure-diff.md:
     every name in every declared tree has exactly one kind, that kind names
     something the section itself declares, and no name is unresolved; and
     each expression's `reads` is the names of its tree, once each, in order
     of first appearance. This walk reads the serialized section, so it does
     not share the emitter's traversal.
   - invariant 11: `ModelStructure` in the compiled model and the flow graph
     from `Latex.build_graph` agree with the section.

   "parse facts" pins what the sameness rule of that proposal relies on:
   redundant parentheses, the spelling of a number, comments, layout and the
   position of a `let` do not change a declared tree; operand order, grouping
   and units do. *)

module D = Declared

let repo_root =
  let is_root dir =
    Sys.file_exists (Filename.concat dir "ir/VERSION")
    && Sys.file_exists (Filename.concat dir "ocaml/dune-project")
  in
  let rec walk dir depth =
    if is_root dir then Some dir
    else if depth = 0 then None
    else
      let parent = Filename.dirname dir in
      if parent = dir then None else walk parent (depth - 1)
  in
  match walk (Sys.getcwd ()) 12 with
  | Some d -> d
  | None ->
    (match walk (Filename.dirname Sys.executable_name) 12 with
     | Some d -> d
     | None -> Alcotest.failf "could not locate repo root")

let read_file path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

let compile ?(filename = "<input>") name src =
  match Compiler.compile_with_declared ~name ~filename src with
  | Ok (m, _reads, d) -> (m, d)
  | Error e -> Alcotest.failf "%s does not compile: %s" name e

(* ── Walking a serialized section ────────────────────────────────────────── *)

type json = Yojson.Safe.t

let is_name_ref = function
  | `Assoc [ ("name", `String _); ("kind", `String _) ] -> true
  | _ -> false

let name_ref = function
  | `Assoc [ ("name", `String n); ("kind", `String k) ] -> (n, k)
  | _ -> Alcotest.fail "not a name reference"

(* The name references of a serialized tree, depth-first in field order, which
   is source order for every node. *)
let rec tree_names (j : json) : (string * string) list =
  match j with
  | `Assoc _ when is_name_ref j -> [ name_ref j ]
  | `Assoc kvs -> List.concat_map (fun (_, v) -> tree_names v) kvs
  | `List xs -> List.concat_map tree_names xs
  | _ -> []

let dedup xs =
  List.fold_left (fun acc x -> if List.mem x acc then acc else acc @ [ x ]) [] xs

(* Every top-level expression `{tree, katex, katex_loss, reads}` of a section, and every
   name reference anywhere in it outside a `reads` list. *)
let rec walk_section (on_expr : json -> json -> unit) (on_name : string * string -> unit)
    (j : json) : unit =
  match j with
  | `Assoc [ ("tree", tree); ("katex", `String _); ("katex_loss", `List _);
             ("reads", reads) ] ->
    on_expr tree reads;
    walk_section on_expr on_name tree
  | `Assoc _ when is_name_ref j -> on_name (name_ref j)
  | `Assoc kvs -> List.iter (fun (_, v) -> walk_section on_expr on_name v) kvs
  | `List xs -> List.iter (walk_section on_expr on_name) xs
  | _ -> ()

(* ── Invariant 10 ────────────────────────────────────────────────────────── *)

let expansion_of bases name =
  List.exists (fun base -> Expander.is_expansion_of ~base name) bases

(* The kinds that name a declaration. A name may not be two of them. *)
let declaring_kinds =
  [ "compartment"; "parameter"; "definition"; "table"; "time_input";
    "quantity"; "stream"; "transition"; "dimension" ]

let check_names file (d : D.declared_model) =
  let j = D.to_json d in
  let comps = List.map (fun (c : D.compartment_decl) -> c.cd_name) d.dm_compartments in
  let params = List.map (fun (p : D.parameter_decl) -> p.pd_name) d.dm_parameters in
  let defs = List.map (fun (l : D.definition_decl) -> l.df_name) d.dm_definitions in
  let tables = List.concat_map (fun (t : D.table_decl) -> t.tb_names) d.dm_tables in
  let inputs = List.map (fun (f : D.time_input_decl) -> f.ti_name) d.dm_time_inputs in
  let quantities = List.map (fun (q : D.quantity_decl) -> q.qd_name) d.dm_quantities in
  let streams = List.map (fun (o : D.observation_decl) -> o.ob_name) d.dm_observations in
  let transitions = List.map (fun (t : D.transition_decl) -> t.tn_name) d.dm_transitions in
  let dims = List.map (fun (x : D.dimension_decl) -> x.dd_name) d.dm_dimensions in
  let levels = List.concat_map (fun (x : D.dimension_decl) -> x.dd_levels) d.dm_dimensions in
  let columns = List.concat_map (fun (o : D.observation_decl) ->
      List.map (fun (c : D.obs_column) -> c.oc_name) o.ob_columns) d.dm_observations in
  let builtins = [ "t"; "dt"; "projected"; "pi"; "e"; "origin"; "t_start"; "t_end" ] in
  let kinds_of : (string, string list) Hashtbl.t = Hashtbl.create 64 in
  let n_names = ref 0 and n_exprs = ref 0 in
  let on_name (name, kind) =
    incr n_names;
    let declared_as_named =
      match kind with
      | "compartment" -> expansion_of comps name
      | "parameter" -> expansion_of params name
      | "definition" -> List.mem name defs
      | "table" -> List.mem name tables
      | "time_input" -> List.mem name inputs
      | "quantity" -> List.mem name quantities
      | "stream" -> List.mem name streams
      | "transition" -> expansion_of transitions name
      | "dimension" -> List.mem name dims
      | "level" -> List.mem name levels
      | "data_column" -> List.mem name columns
      | "builtin" -> List.mem name builtins
      | "obs_anchor" -> name = "first_obs" || name = "last_obs"
      (* Bound by the enclosing declaration or `sum`; a keyword is a word the
         form reads as text. Neither names a declaration. *)
      | "index_var" | "keyword" -> true
      | "unresolved" -> false
      | other -> Alcotest.failf "%s: unknown name kind '%s'" file other
    in
    if not declared_as_named then
      Alcotest.failf "%s: '%s' is tagged %s, but the section declares no such %s"
        file name kind kind;
    if List.mem kind declaring_kinds then begin
      let seen = Option.value ~default:[] (Hashtbl.find_opt kinds_of name) in
      if not (List.mem kind seen) then Hashtbl.replace kinds_of name (kind :: seen)
    end
  in
  let on_expr tree reads =
    incr n_exprs;
    let want = dedup (tree_names tree) in
    let got = match reads with
      | `List xs -> List.map name_ref xs
      | _ -> Alcotest.failf "%s: `reads` is not a list" file in
    if got <> want then
      Alcotest.failf "%s: reads [%s] is not the tree's names [%s]" file
        (String.concat "; " (List.map (fun (n, k) -> n ^ ":" ^ k) got))
        (String.concat "; " (List.map (fun (n, k) -> n ^ ":" ^ k) want))
  in
  walk_section on_expr on_name j;
  Hashtbl.iter (fun name kinds ->
      if List.length kinds > 1 then
        Alcotest.failf "%s: '%s' has more than one kind: %s" file name
          (String.concat ", " kinds)) kinds_of;
  (!n_names, !n_exprs)

(* ── Invariant 11 ────────────────────────────────────────────────────────── *)

(* Names the compiler invents carry a double underscore: a `via` stage
   dimension `__<transition>_stage`, a hyper-Erlang stage compartment
   `<source>__<label>__<i>`. *)
let invented name =
  let n = String.length name in
  let rec go i = i + 1 < n && ((name.[i] = '_' && name.[i + 1] = '_') || go (i + 1)) in
  go 0

let check_model_structure file (m : Ir.model) (d : D.declared_model) =
  match m.model_structure with
  | None -> Alcotest.failf "%s: the compiled model has no model_structure" file
  | Some ms ->
    let declared_levels name =
      match List.find_opt (fun (x : D.dimension_decl) -> x.dd_name = name) d.dm_dimensions with
      | Some x -> x.dd_levels
      | None -> Alcotest.failf "%s: model_structure has dimension '%s', not declared" file name
    in
    (* The structure's dimensions are the stratifying ones, in stratify order. *)
    let ms_dims = List.filter (fun (x : Ir.dimension) -> not (invented x.dim_name)) ms.dimensions in
    Alcotest.(check (list string)) (file ^ ": stratifying dimensions")
      (List.map (fun (s : D.stratify_decl) -> s.st_dim) d.dm_stratify)
      (List.map (fun (x : Ir.dimension) -> x.dim_name) ms_dims);
    List.iter (fun (x : Ir.dimension) ->
        Alcotest.(check (list string)) (file ^ ": levels of " ^ x.dim_name)
          (declared_levels x.dim_name) x.dim_values) ms_dims;
    (* Base compartments: the declared ones, less any a hyper-Erlang `via`
       replaced by per-branch stage compartments. *)
    let declared = List.map (fun (c : D.compartment_decl) -> c.cd_name) d.dm_compartments in
    let ms_base = List.filter (fun c -> not (invented c)) ms.base_compartments in
    List.iter (fun c ->
        if not (List.mem c declared) then
          Alcotest.failf "%s: model_structure has base compartment '%s', not declared" file c)
      ms_base;
    let staged = List.filter (fun c -> not (List.mem c ms_base)) declared in
    List.iter (fun c ->
        let is_staged_source =
          List.exists (fun (t : D.transition_decl) ->
              (match t.tn_dynamics with D.DynVia _ -> true | D.DynRate _ -> false)
              && List.exists (fun (r : D.stoich_ref) -> r.sr_compartment = c) t.tn_from)
            d.dm_transitions in
        if not is_staged_source then
          Alcotest.failf
            "%s: declared compartment '%s' is missing from model_structure and \
             is not the source of a `via` transition" file c) staged;
    List.iter (fun (c : D.compartment_decl) ->
        match List.assoc_opt c.cd_name ms.compartment_dims with
        | None -> ()   (* a staged source, checked above *)
        | Some dims ->
          Alcotest.(check (list string)) (file ^ ": dimensions of " ^ c.cd_name)
            c.cd_dims (List.filter (fun x -> not (invented x)) dims))
      d.dm_compartments

let distinct xs = dedup xs

(* The flow graph has one node per declared compartment, one plate per
   declared dimension, and one edge per (source, destination) pair of each
   declared transition, as `Latex.transition_graph` lays them out. *)
let check_flow_graph file (decls : Ast.declaration list) (d : D.declared_model) =
  let g = Latex.build_graph decls in
  Alcotest.(check (list string)) (file ^ ": graph nodes")
    (List.map (fun (c : D.compartment_decl) -> c.cd_name) d.dm_compartments)
    (List.map (fun (n : Latex.g_node) -> n.gn_id) g.mg_nodes);
  Alcotest.(check (list string)) (file ^ ": graph plates")
    (List.map (fun (x : D.dimension_decl) -> x.dd_name) d.dm_dimensions)
    (List.map (fun (p : Latex.g_plate) -> p.gp_name) g.mg_plates);
  List.iter2 (fun (x : D.dimension_decl) (p : Latex.g_plate) ->
      match x.dd_source with
      | D.DsInline ->
        Alcotest.(check (list string)) (file ^ ": plate levels of " ^ x.dd_name)
          x.dd_levels p.gp_levels
      (* The graph is built without reading files, so a file-backed dimension
         has no levels there; the section has the resolved ones. *)
      | D.DsRead _ ->
        Alcotest.(check (list string)) (file ^ ": file-backed plate " ^ x.dd_name)
          [] p.gp_levels)
    d.dm_dimensions g.mg_plates;
  let expected =
    List.concat_map (fun (t : D.transition_decl) ->
        let bases refs = distinct (List.map (fun (r : D.stoich_ref) -> r.sr_compartment) refs) in
        let froms = match bases t.tn_from with [] -> [ None ] | xs -> List.map Option.some xs in
        (* A `hyper_erlang` law may give each branch its own destination
           (`branch(..., to = D)`); a branch without one goes to the arrow
           target. Read here from the declared arguments. *)
        let branch_tos arrow =
          match t.tn_dynamics with
          | D.DynVia ("hyper_erlang", args) ->
            let per_branch = List.filter_map (fun (a : D.arg) ->
                match a.a_value with
                | D.TCall ("branch", kvs) ->
                  Some (List.find_map (fun (k : D.arg) ->
                      match k.a_name, k.a_value with
                      | "to", (D.TIdent r | D.TIndex (r, _)) -> Some r.nr_name
                      | _ -> None) kvs)
                | _ -> None) args in
            if List.for_all (( = ) None) per_branch then []
            else distinct (List.concat_map (function
                | Some d -> [ Some d ]
                | None -> List.map Option.some arrow) per_branch)
          | _ -> [] in
        let tos = match t.tn_to with
          | D.DstSum refs ->
            (match branch_tos (bases refs) with
             | [] ->
               (match bases refs with [] -> [ None ] | xs -> List.map Option.some xs)
             | xs -> xs)
          | D.DstBranch arms ->
            List.map (fun (a : D.branch_arm) -> Some a.ba_to.sr_compartment) arms in
        let n = List.length froms * List.length tos in
        List.concat_map (fun f ->
            List.map (fun to_ ->
                let id =
                  if n = 1 then t.tn_name
                  else Printf.sprintf "%s__%s__%s" t.tn_name
                      (Option.value ~default:"src" f) (Option.value ~default:"sink" to_) in
                (id, f, to_)) tos) froms) d.dm_transitions in
  let got = List.map (fun (e : Latex.g_edge) -> (e.ge_id, e.ge_from, e.ge_to)) g.mg_edges in
  let show (id, f, t) =
    Printf.sprintf "%s: %s -> %s" id (Option.value ~default:"-" f) (Option.value ~default:"-" t) in
  Alcotest.(check (list string)) (file ^ ": graph edges")
    (List.map show expected) (List.map show got);
  (* Every edge carries the name of the declared transition it draws, and of
     exactly one: its id is that name, or that name followed by the pair
     `__<from>__<to>`. No two edges share an id, so a viewer told "transition
     X changed" finds X's edges and no others. *)
  let ids = List.map (fun (e : Latex.g_edge) -> e.ge_id) g.mg_edges in
  if List.length (distinct ids) <> List.length ids then
    Alcotest.failf "%s: two graph edges share an id: %s" file (String.concat ", " ids);
  let owners (e : Latex.g_edge) =
    List.filter (fun (t : D.transition_decl) ->
        e.ge_id = t.tn_name
        || e.ge_id = Printf.sprintf "%s__%s__%s" t.tn_name
               (Option.value ~default:"src" e.ge_from)
               (Option.value ~default:"sink" e.ge_to)) d.dm_transitions in
  List.iter (fun (e : Latex.g_edge) ->
      match owners e with
      | [ _ ] -> ()
      | ts ->
        Alcotest.failf "%s: graph edge '%s' belongs to %d declared transitions (%s)"
          file e.ge_id (List.length ts)
          (String.concat ", " (List.map (fun (t : D.transition_decl) -> t.tn_name) ts)))
    g.mg_edges;
  let edges_of (t : D.transition_decl) =
    List.filter (fun e -> List.memq t (owners e)) g.mg_edges in
  let via_rate law = Printf.sprintf "\\text{via } \\mathrm{%s}" (Latex.escape law) in
  (* What each edge shows as its rate, against the declaration. *)
  List.iter (fun (t : D.transition_decl) ->
      let es = edges_of t in
      match t.tn_dynamics, t.tn_to with
      (* An ordinary transition: every edge shows the declared rate. *)
      | D.DynRate r, D.DstSum _ ->
        List.iter (fun (e : Latex.g_edge) ->
            Alcotest.(check string) (file ^ ": rate of " ^ e.ge_id) r.de_katex e.ge_rate)
          es
      (* A branching transition: one edge per (source, arm), and the edge to an
         arm shows that arm's weight times the declared rate. *)
      | D.DynRate r, D.DstBranch arms ->
        let n_from = max 1 (List.length (distinct
            (List.map (fun (s : D.stoich_ref) -> s.sr_compartment) t.tn_from))) in
        Alcotest.(check int) (file ^ ": one edge per source and arm of " ^ t.tn_name)
          (n_from * List.length arms) (List.length es);
        List.iter (fun (a : D.branch_arm) ->
            let to_arm = List.filter (fun (e : Latex.g_edge) ->
                e.ge_to = Some a.ba_to.sr_compartment) es in
            if to_arm = [] then
              Alcotest.failf "%s: %s has no edge to its arm %s" file t.tn_name
                a.ba_to.sr_compartment;
            List.iter (fun (e : Latex.g_edge) ->
                Alcotest.(check string) (file ^ ": rate of " ^ e.ge_id)
                  (Printf.sprintf "%s \\cdot %s"
                     (* A weight that is a sum or a difference is a factor
                        here, so it is parenthesised. *)
                     (match a.ba_weight.de_tree with
                      | D.TBinOp ((D.BAdd | D.BSub), _, _) ->
                        "\\left(" ^ a.ba_weight.de_katex ^ "\\right)"
                      | _ -> a.ba_weight.de_katex)
                     r.de_katex)
                  e.ge_rate) to_arm) arms
      (* A staged transition (`via`): the graph draws the declaration and not
         the stages. Each edge runs from the declared source and shows "via
         <law>" as its rate, preceded by a branch's share where the law splits
         the flow; no stage compartment or stage transition appears (the node
         and edge lists above are the declared ones). *)
      | D.DynVia (law, args), dst ->
        if es = [] then Alcotest.failf "%s: staged %s has no edge" file t.tn_name;
        let plain = via_rate law in
        let ends_with ~suffix s =
          let n = String.length suffix and h = String.length s in
          h >= n && String.sub s (h - n) n = suffix in
        List.iter (fun (e : Latex.g_edge) ->
            if not (e.ge_rate = plain || ends_with ~suffix:(" \\cdot " ^ plain) e.ge_rate)
            then
              Alcotest.failf "%s: staged %s shows the rate %S, not \"via %s\""
                file e.ge_id e.ge_rate law) es;
        (* `hyper_erlang` may write no arrow: each destination is then a
           `to =` inside a `branch(...)` argument. The graph draws one edge to
           each such destination, and none to nowhere. *)
        let branch_destinations =
          List.concat_map (fun (a : D.arg) ->
              match a.a_value with
              | D.TCall ("branch", kvs) ->
                List.filter_map (fun (kv : D.arg) ->
                    match kv.a_name, kv.a_value with
                    | "to", D.TIdent r -> Some r.nr_name
                    | "to", D.TIndex (r, _) -> Some r.nr_name
                    | _ -> None) kvs
              | _ -> []) args in
        (match dst, branch_destinations with
         | D.DstSum [], (_ :: _ as dests) ->
           Alcotest.(check (list (option string)))
             (file ^ ": staged " ^ t.tn_name ^ " is drawn to each branch destination")
             (List.map Option.some (distinct dests))
             (List.map (fun (e : Latex.g_edge) -> e.ge_to) es);
           (* Two branches to two destinations: each edge carries its share. *)
           if List.length (distinct dests) = List.length dests then
             List.iter (fun (e : Latex.g_edge) ->
                 if e.ge_rate = plain then
                   Alcotest.failf "%s: branch edge %s shows no share of the flow"
                     file e.ge_id) es
         | D.DstSum [], [] ->
           Alcotest.failf "%s: staged %s declares no destination anywhere" file t.tn_name
         | _ -> ()))
    d.dm_transitions

(* ── Every golden ────────────────────────────────────────────────────────── *)

let camdl_files dir =
  if Sys.file_exists dir && Sys.is_directory dir then
    Sys.readdir dir |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".camdl")
    |> List.sort compare
    |> List.map (fun f -> Filename.concat dir f)
  else []

let golden_models () =
  let fixtures = Filename.concat repo_root "tests/fixtures" in
  let fixture_dirs =
    Sys.readdir fixtures |> Array.to_list |> List.sort compare
    |> List.map (Filename.concat fixtures)
    |> List.filter (fun d ->
        Sys.is_directory d && Sys.file_exists (Filename.concat d "ir")) in
  List.concat_map camdl_files (Filename.concat repo_root "ocaml/golden" :: fixture_dirs)

(* ── What each declaration expanded to ───────────────────────────────────── *)

(* The `expands_to` lists of the declared compartments, taken together, are
   exactly the compiled model's compartments: every compiled compartment is
   claimed by one declaration and by no other, and no declaration claims a
   name the model does not hold. The same for transitions and for observation
   streams. This compares against the finished compiled model, which the
   emitter does not read, so a name the compiler adds, drops or renames after
   expansion fails here. *)
let check_expands_to file (m : Ir.model) (d : D.declared_model) =
  let partition what (claims : (string * string list option) list)
      (compiled : string list) =
    let claimed = List.concat_map (fun (decl, e) ->
        match e with
        | Some names -> List.map (fun n -> (n, decl)) names
        | None ->
          Alcotest.failf "%s: %s '%s' has no expands_to" file what decl) claims in
    List.iter (fun (n, decl) ->
        match List.filter (fun (n', _) -> n' = n) claimed with
        | [ _ ] -> ()
        | owners ->
          Alcotest.failf "%s: compiled %s '%s' is claimed %d times (by %s, ...)"
            file what n (List.length owners) decl) claimed;
    Alcotest.(check (list string))
      (Printf.sprintf "%s: the declared %ss expand to the compiled ones" file what)
      (List.sort compare compiled)
      (List.sort compare (List.map fst claimed))
  in
  partition "compartment"
    (List.map (fun (c : D.compartment_decl) -> (c.cd_name, c.cd_expands_to))
       d.dm_compartments)
    (List.map (fun (c : Ir.compartment) -> c.name) m.compartments);
  partition "transition"
    (List.map (fun (t : D.transition_decl) -> (t.tn_name, t.tn_expands_to))
       d.dm_transitions)
    (List.map (fun (t : Ir.transition) -> t.name) m.transitions);
  partition "observation stream"
    (List.map (fun (o : D.observation_decl) -> (o.ob_name, o.ob_expands_to))
       d.dm_observations)
    (List.map (fun (o : Ir.observation_model) -> o.name) m.observations)

let relative path =
  let n = String.length repo_root + 1 in
  String.sub path n (String.length path - n)

let parse src =
  let lexbuf = Lexing.from_string src in
  try Parser.file Lexer.token lexbuf with _ -> Alcotest.fail "parse failed"

let test_golden path () =
  let file = relative path in
  let src = read_file path in
  let name = Filename.remove_extension (Filename.basename path) in
  let m, d = compile ~filename:path name src in
  let n_names, n_exprs = check_names file d in
  check_model_structure file m d;
  check_flow_graph file (parse src) d;
  (* The section round-trips through its own codec. *)
  Alcotest.(check bool) (file ^ ": round-trips") true (D.of_json (D.to_json d) = d);
  check_expands_to file m d;
  if n_exprs = 0 || n_names = 0 then
    Alcotest.failf "%s: the walk found no expressions or no names" file

(* Recorded paths never leave the model's directory by an absolute route. *)
let test_no_absolute_paths () =
  List.iter (fun path ->
      let _, d = compile ~filename:path
          (Filename.remove_extension (Filename.basename path)) (read_file path) in
      List.iter (fun (f : D.file_input) ->
          if not (Filename.is_relative f.fi_path) then
            Alcotest.failf "%s records an absolute path: %s" (relative path) f.fi_path;
          if f.fi_sha256 = None then
            Alcotest.failf "%s: no digest for %s" (relative path) f.fi_path)
        d.dm_compile_inputs.ci_files)
    (golden_models ())

(* ── Every name kind is exercised by a compiled golden ───────────────────── *)

let kinds_in (d : D.declared_model) : string list =
  let seen = ref [] in
  walk_section (fun _ _ -> ())
    (fun (_, kind) -> if not (List.mem kind !seen) then seen := kind :: !seen)
    (D.to_json d);
  !seen

let section_of path =
  snd (compile ~filename:path
         (Filename.remove_extension (Filename.basename path)) (read_file path))

(* The three kinds that only a hand-written fixture used to reach each have a
   compiled golden of their own (`tests/fixtures/declared_names`). *)
let test_declared_names_fixtures () =
  let dir = Filename.concat repo_root "tests/fixtures/declared_names" in
  List.iter (fun (file, kind) ->
      let kinds = kinds_in (section_of (Filename.concat dir file)) in
      if not (List.mem kind kinds) then
        Alcotest.failf "%s holds no name of kind '%s' (it has: %s)" file kind
          (String.concat ", " (List.sort compare kinds)))
    [ ("obs_anchor.camdl", "obs_anchor");
      ("aux_data_column.camdl", "data_column");
      ("table_time_dimension.camdl", "dimension") ]

(* Every kind the codec knows, other than `unresolved`, appears in at least one
   compiled golden: a kind with no golden is a rule nothing compiled checks. *)
let test_every_kind_has_a_golden () =
  let seen = Hashtbl.create 16 in
  List.iter (fun path ->
      List.iter (fun k -> Hashtbl.replace seen k (relative path))
        (kinds_in (section_of path)))
    (golden_models ());
  List.iter (fun (kind, name) ->
      if kind <> D.KUnresolved && not (Hashtbl.mem seen name) then
        Alcotest.failf "no compiled golden holds a name of kind '%s'" name)
    D.name_kind_table;
  Alcotest.(check bool) "no golden holds an unresolved name" false
    (Hashtbl.mem seen "unresolved")

(* Every path at which two JSON values differ, as "a.b[2].c". *)
let rec json_diff (path : string) (a : json) (b : json) : string list =
  match a, b with
  | `Assoc xs, `Assoc ys
    when List.map fst xs = List.map fst ys ->
    List.concat (List.map2 (fun (k, x) (_, y) ->
      json_diff (if path = "" then k else path ^ "." ^ k) x y) xs ys)
  | `List xs, `List ys when List.length xs = List.length ys ->
    List.concat (List.mapi (fun i (x, y) ->
      json_diff (Printf.sprintf "%s[%d]" path i) x y) (List.combine xs ys))
  | _ -> if a = b then [] else [ path ]

let ends_with ~suffix s =
  let n = String.length suffix and h = String.length s in
  h >= n && String.sub s (h - n) n = suffix

(* ── Parse facts ─────────────────────────────────────────────────────────── *)

let sir rate =
  Printf.sprintf {|
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I  @ %s
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - 1
       I = 1 }
simulate { from = 0 'days
           to = 80 'days }
|} rate

let rate_tree src =
  let _, d = compile "m" src in
  match (List.hd d.dm_transitions).tn_dynamics with
  | D.DynRate r -> r.de_tree
  | D.DynVia _ -> Alcotest.fail "expected a rate"

let same a b = Alcotest.(check bool) "same tree" true (rate_tree a = rate_tree b)
let differ a b = Alcotest.(check bool) "different tree" false (rate_tree a = rate_tree b)

let test_parens () =
  same (sir "(beta * S) * I / N") (sir "beta * S * I / N");
  same (sir "((beta * S * I / N))") (sir "beta * S * I / N");
  same (sir "(((beta * S) * I) / N)") (sir "beta * S * I / N")

let test_number_spelling () =
  same (sir "beta * 1 * S * I / N") (sir "beta * 1.0 * S * I / N");
  same (sir "beta * 1e0 * S * I / N") (sir "beta * 1 * S * I / N")

let test_operand_order () =
  differ (sir "beta * S * I / N") (sir "S * beta * I / N")

let test_grouping () =
  differ (sir "beta * S * (I / N)") (sir "beta * S * I / N")

let test_units () =
  let m span = sir (Printf.sprintf "beta * S * I / N + S / (%s)" span) in
  differ (m "7 'days") (m "1 'weeks")

(* Comments, a doc comment, layout, and where a `let` sits change no declared
   tree: the section is equal apart from the text as written of the one
   declaration whose spacing differs (a `let` is in `definitions` wherever it
   is written, and a comment or a doc block is outside a declaration's text). *)
let test_layout_and_let_position () =
  let base = sir "beta * S * I / N" in
  let _, d0 = compile "m" base in
  let commented = {|
# a comment
time_unit = 'days
compartments { S, I, R }
parameters {
  #' transmission rate
  beta  : rate in [0.001, 2.0]   # trailing
  gamma : rate in [0.001, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I  @ beta*S*I/N
  recovery  : I --> R  @ gamma * I
}
let N = S + I + R
init { S = N0 - 1
       I = 1 }
simulate { from = 0 'days
           to = 80 'days }
|} in
  let _, d1 = compile "m" commented in
  Alcotest.(check (list string))
    "only the text of the declaration that is spaced differently"
    [ "transitions[0].source_text" ]
    (json_diff "" (D.to_json d0) (D.to_json d1))

(* ── The text as written ─────────────────────────────────────────────────── *)

let text_of what = function
  | Some s -> s
  | None -> Alcotest.failf "%s has no text as written" what

(* Each declaration carries its own bytes, from its first token to its last:
   the keyword of a `let` and of a block-level declaration, a `#[lineage]`
   attribute, and interior line breaks are part of it; a doc block above it, a
   trailing comment and the surrounding block are not. *)
let test_source_text () =
  let src = {|
time_unit = 'days
compartments { S, I : integer, R }
let N = S + I + R
parameters {
  #' transmission rate
  beta  : rate in [0.001, 2.0]   # trailing
  gamma : rate in [0.001, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  #' a doc block
  #[lineage]
  infection : S --> I
     @ beta × S * I / N      # the rate
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - 1
       I = 1 }
simulate { from = 0 'days
           to = 80 'days }
|} in
  let _, d = compile "m" src in
  let check what expected actual =
    Alcotest.(check string) what expected (text_of what actual) in
  check "compartment I" "I : integer" (List.nth d.dm_compartments 1).cd_text;
  check "let N" "let N = S + I + R" (List.hd d.dm_definitions).df_text;
  check "parameter beta" "beta  : rate in [0.001, 2.0]"
    (List.hd d.dm_parameters).pd_text;
  check "transition infection"
    "#[lineage]\n  infection : S --> I\n     @ beta × S * I / N"
    (List.hd d.dm_transitions).tn_text;
  check "transition recovery" "recovery  : I --> R  @ gamma * I"
    (List.nth d.dm_transitions 1).tn_text;
  check "initial S" "S = N0 - 1" (List.hd d.dm_initial).in_text;
  check "simulate" "simulate { from = 0 'days\n           to = 80 'days }"
    (match d.dm_simulation with
     | Some s -> s.sm_text
     | None -> Alcotest.fail "no simulate block")

(* On every golden, every declaration has its text, and a declaration with a
   name begins with that name, with `let`, or with an attribute. This reads
   the serialized section, so a declaration kind added without a text fails
   here rather than being recorded as absent. *)
let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let test_golden_source_text path () =
  let file = relative path in
  let n = ref 0 in
  let rec go (j : json) =
    match j with
    | `Assoc kvs ->
      (match List.assoc_opt "source_text" kvs with
       | None -> ()
       | Some (`String text) ->
         incr n;
         if text = "" then Alcotest.failf "%s: an empty text as written" file;
         (match List.assoc_opt "name" kvs with
          | Some (`String name)
            when not (starts_with ~prefix:name text
                      || starts_with ~prefix:"let" text
                      || starts_with ~prefix:"#[" text) ->
            Alcotest.failf "%s: the text of '%s' does not begin with it: %S"
              file name text
          | _ -> ())
       | Some _ ->
         Alcotest.failf "%s: a declaration without its text as written: %s" file
           (Yojson.Safe.to_string j));
      List.iter (fun (_, v) -> go v) kvs
    | `List xs -> List.iter go xs
    | _ -> ()
  in
  go (D.to_json (section_of path));
  if !n = 0 then Alcotest.failf "%s: no declaration carries a text" file

(* A quoted string is a string, not a name, and is not read. *)
let test_string_literal () =
  let src = {|
time_unit = 'days
origin = date("2020-01-01")
compartments { S, I }
parameters { beta : rate in [0.001, 2.0] }
transitions { infection : S --> I @ beta * S }
init { S = 99
       I = 1 }
simulate { from = date("2020-01-01")
           to = date("2020-03-01") }
|} in
  let _, d = compile "m" src in
  match d.dm_simulation with
  | Some { sm_from = { de_tree = D.TCall ("date", [ { a_value = D.TStr "2020-01-01"; _ } ]);
                       de_reads = []; _ }; _ } -> ()
  | _ -> Alcotest.fail "date(\"…\") should hold a string literal and read no name"

(* ── What the rendering does not show ────────────────────────────────────── *)

let loss_names (ls : D.katex_loss list) : string list =
  List.map (fun l -> List.assoc l D.katex_loss_table) ls

let rate_loss src =
  let _, d = compile "m" src in
  match (List.hd d.dm_transitions).tn_dynamics with
  | D.DynRate r -> loss_names r.de_katex_loss
  | D.DynVia _ -> Alcotest.fail "expected a rate"

(* On compiled models: a rate rendered as written has no entry, and each of
   the common shortfalls is named. The pairs in each group are different
   trees that render alike, which is why the second of each is reported. *)
let test_katex_loss_compiled () =
  let check rate expected =
    Alcotest.(check (list string)) rate expected (rate_loss (sir rate)) in
  check "beta * S * I / N" [];
  check "beta * S * (I / N)" [ "grouping_changed" ];
  check "beta * S / N * I" [ "grouping_changed" ];
  check "beta * S * I / N + gamma * I" [];
  check "beta * (S - (I - R))" [ "grouping_changed" ];
  check "beta * (S - I - R)" [];
  check "beta * S * I / N + S / (7 'days)" [ "unit_dropped" ];
  check "0.1234567 * beta * S" [ "number_rounded" ];
  check "0.123457 * beta * S" [];
  check "beta * S * exp(0 - I / N)" []

let located s = Ast.EIdent (s, { Ast.dummy_loc with line = 1; col = 1 })

(* On hand-built trees: every reason, alone, and the shapes that have none.
   This is the table of what `Latex.tex` does not show. *)
let test_katex_loss_reasons () =
  Hashtbl.clear Latex.overrides;
  let a = located "a" and b = located "b" and c = located "c" in
  let bin op l r = Ast.EBinOp (op, l, r) in
  let check what expected e =
    Alcotest.(check (list string)) what expected (loss_names (Latex.tex_loss e)) in
  check "a name" [] a;
  check "a unit literal" [ "unit_dropped" ] (Ast.EUnit (0.5, Ast.PerDay));
  check "a number of seven digits" [ "number_rounded" ] (Ast.EConst 0.1234567);
  check "a number of six digits" [] (Ast.EConst 0.123457);
  check "a whole number" [] (Ast.EConst 100000.0);
  check "a list" [ "placeholder" ] (Ast.EList [ a; b ]);
  check "a range" [ "placeholder" ] (Ast.ERange (a, b));
  check "observations.x" [ "placeholder" ] (Ast.EObsAccess ("x", Ast.dummy_loc));
  check "a quoted string" [ "string_as_name" ] (Ast.EIdent ("a", Ast.dummy_loc));
  check "a sum over a dimension" [ "sum_range_dropped" ]
    (Ast.ESum ("i", "age", None, a, Ast.dummy_loc));
  check "a positional call" [] (Ast.EFuncCall ("min", [ ("", a); ("", b) ]));
  check "a call with a named argument" [ "argument_names_dropped" ]
    (Ast.EFuncCall ("f", [ ("rate", a) ]));
  check "a positional index" []
    (Ast.EIndex ("S", [ Ast.IPosn a ], Ast.dummy_loc));
  check "a named index" [ "index_dimension_dropped" ]
    (Ast.EIndex ("S", [ Ast.INamed ("age", a) ], Ast.dummy_loc));
  (* Grouping. *)
  check "a * b / c" [] (bin Div (bin Mul a b) c);
  check "a / (b * c)" [] (bin Div a (bin Mul b c));
  check "a / b / c" [ "grouping_changed" ] (bin Div (bin Div a b) c);
  check "a * (b * c)" [ "grouping_changed" ] (bin Mul a (bin Mul b c));
  check "a + b - c" [] (bin Sub (bin Add a b) c);
  check "a + (b + c)" [ "grouping_changed" ] (bin Add a (bin Add b c));
  check "(a + b) * c" [] (bin Mul (bin Add a b) c);
  check "a ^ b" [] (bin Pow a b);
  check "(a + b) ^ c" [] (bin Pow (bin Add a b) c);
  check "(-a) ^ b" [ "grouping_changed" ] (bin Pow (Ast.EUnOp (Neg, a)) b);
  check "(a ^ b) ^ c" [ "grouping_changed" ] (bin Pow (bin Pow a b) c);
  check "(a / b) ^ c" [ "grouping_changed" ] (bin Pow (bin Div a b) c);
  check "-(a ^ b)" [] (Ast.EUnOp (Neg, bin Pow a b));
  check "a * -b" [ "grouping_changed" ] (bin Mul a (Ast.EUnOp (Neg, b)));
  check "(a < b) * c" [ "grouping_changed" ] (bin Mul (bin Lt a b) c);
  check "a < b" [] (bin Lt a b);
  check "if a < b then a else b" [] (Ast.ECond (bin Lt a b, a, b));
  check "a * sum(b)" [] (bin Mul a (Ast.EUnOp (Total, b)));
  check "sum(b) * a" [ "grouping_changed" ] (bin Mul (Ast.EUnOp (Total, b)) a);
  (* A `@symbol` tag. *)
  Hashtbl.replace Latex.overrides "a" "\\alpha";
  check "a name with a @symbol" [ "symbol_override" ] a;
  check "several reasons, in the order of the type"
    [ "unit_dropped"; "grouping_changed"; "symbol_override" ]
    (bin Mul a (bin Mul b (Ast.EUnit (1.0, Ast.Days))));
  Hashtbl.clear Latex.overrides

(* ── A doc comment is not a model difference ─────────────────────────────── *)

let documented ~beta_doc ~model_doc =
  Printf.sprintf {|%s
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
%s
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - 1
       I = 1 }
simulate { from = 0 'days
           to = 80 'days }
|} model_doc beta_doc

(* Two sources that differ only in `#'` doc comments, one of which carries an
   `@symbol` tag the KaTeX renderer substitutes. The compiled model is the same
   model (its JSON, which is all `model_ir_hash` reads, is equal); every tree
   and every other field of the declared section is equal; and the only fields
   that differ are `katex` strings and, where the tag's symbol is shown, the
   record that the rendering shows a symbol in place of the name. So a diff that compares trees reports no
   model-layer change for a doc-comment edit. *)
let test_doc_comment_is_not_a_model_difference () =
  let plain = documented ~beta_doc:"" ~model_doc:"" in
  let doc'd = documented
      ~model_doc:"#' A bare SIR model.\n#' @ref Kermack and McKendrick 1927"
      ~beta_doc:"  #' per-capita transmission rate\n  #' @symbol β" in
  let m0, d0 = compile "m" plain and m1, d1 = compile "m" doc'd in
  Alcotest.(check bool) "the compiled model is the same model" true
    (Serde.model_to_json m0 = Serde.model_to_json m1);
  let diffs = json_diff "" (D.to_json d0) (D.to_json d1) in
  Alcotest.(check bool) "the doc comment changed a rendering (the test bites)"
    true (diffs <> []);
  List.iter (fun path ->
    if not (ends_with ~suffix:".katex" path
            || ends_with ~suffix:".katex_loss" path) then
      Alcotest.failf "a doc comment changed a field that is not `katex`: %s" path)
    diffs;
  (* Stated outright: which fields differed. *)
  Alcotest.(check (list string)) "only the renderings that show beta"
    [ "transitions[0].dynamics.rate.rate.katex";
      "transitions[0].dynamics.rate.rate.katex_loss" ] diffs

(* ── An unclassified name is reported (W203) ─────────────────────────────── *)

(* The compiler discards a `timepoints { }` block (gh#1047), so a timepoint
   that refers to another one compiles, and the name it reads is nothing the
   section can classify. That is a valid model producing an unresolved name. *)
let with_timepoints block =
  sir "beta * S * I / N" ^ "\ntimepoints {\n" ^ block ^ "\n}\n"

(* The 1-indexed line of the `later = …` entry in [src]. *)
let later_line src =
  let rec find i = function
    | [] -> Alcotest.fail "no `later` line"
    | l :: rest ->
      if String.length l >= 7 && String.sub l 0 7 = "  later" then i
      else find (i + 1) rest in
  find 1 (String.split_on_char '\n' src)

let w203 src =
  List.filter (fun (d : Diagnostics.diagnostic) -> d.code = "W203")
    (Compiler.collect_diagnostics ~name:"m" ~filename:"m.camdl" src)

let test_unresolved_is_warned () =
  let src = with_timepoints "  midpoint = 40 'days\n  later = midpoint + 10 'days" in
  (* The compile succeeds, and the section says "unresolved". *)
  let _, d = compile "m" src in
  let later = List.find (fun (tp : D.timepoint_decl) -> tp.tp_name = "later")
      d.dm_timepoints in
  Alcotest.(check bool) "the section tags the name unresolved" true
    (List.mem { D.nr_name = "midpoint"; nr_kind = D.KUnresolved }
       later.tp_time.de_reads);
  match w203 src with
  | [ w ] ->
    Alcotest.(check bool) "a warning, not an error" true
      (w.severity = Diagnostics.Warning);
    Alcotest.(check string) "message"
      "the declared-model section could not classify the name 'midpoint' in \
       timepoint 'later'" w.message;
    let detail = Option.value w.detail ~default:"" in
    let contains needle =
      let n = String.length needle and h = String.length detail in
      let rec at i = i + n <= h && (String.sub detail i n = needle || at (i + 1)) in
      at 0 in
    Alcotest.(check bool) "says the compiled model is unaffected" true
      (contains "the compiled model is unaffected");
    Alcotest.(check bool) "says what `camdl diff` will report" true
      (contains "`camdl diff` will report the dependencies of timepoint 'later' \
                 as unknown");
    Alcotest.(check (pair int int)) "points at the name" (later_line src, 11)
      (w.loc.line, w.loc.col)
  | ws -> Alcotest.failf "expected one W203, got %d" (List.length ws)

(* Every name classified: no warning. *)
let test_resolved_is_silent () =
  Alcotest.(check int) "no W203" 0
    (List.length (w203 (with_timepoints "  midpoint = 40 'days")))

(* gh#1117. `inclusive_end = true | false` is a Boolean word chosen by
   spelling, as a reactive intervention's `once` is. The compiled schedule was
   right, and the section tagged the word unresolved, so every model using the
   flag warned and `camdl diff` read the intervention as having unknown
   dependencies. *)
let with_date_range inclusive_end =
  Printf.sprintf {|
time_unit = 'days
origin = date("2020-01-01")
compartments { S, V }
parameters { N0 : count }
transitions { leak : S --> V @ 0.0 'per_day * S }
interventions {
  mo : transfer(from = S, to = V, fraction = 0.1)
       at [date_range(date("2020-01-01"), date("2020-05-01"),
                      calendar_months = 1, inclusive_end = %s)]
}
init { S = N0 }
simulate { from = 0 'days  to = 800 'days }
|} inclusive_end

let test_boolean_keyword_argument word () =
  let src = with_date_range word in
  Alcotest.(check int) "no W203" 0 (List.length (w203 src));
  let _, d = compile "m" src in
  let iv = List.find (fun (i : D.intervention_decl) -> i.iv_name = "mo")
      d.dm_interventions in
  match iv.iv_schedule with
  | D.ShAtTimes [ e ] ->
    Alcotest.(check bool) (word ^ " is a keyword") true
      (List.mem { D.nr_name = word; nr_kind = D.KKeyword } e.de_reads)
  | _ -> Alcotest.fail "expected an `at [...]` schedule"

let () =
  Alcotest.run "declared_emit"
    [ ("goldens",
       List.map (fun path ->
           Alcotest.test_case (relative path) `Quick (test_golden path))
         (golden_models ())
       @ [ Alcotest.test_case "recorded paths are relative, with digests" `Quick
             test_no_absolute_paths;
           Alcotest.test_case "anchor, data column and dimension names each \
                               have a golden" `Quick test_declared_names_fixtures;
           Alcotest.test_case "every name kind appears in a compiled golden"
             `Quick test_every_kind_has_a_golden ]);
      ("parse facts",
       [ Alcotest.test_case "redundant parentheses" `Quick test_parens;
         Alcotest.test_case "the spelling of a number" `Quick test_number_spelling;
         Alcotest.test_case "operand order is kept" `Quick test_operand_order;
         Alcotest.test_case "grouping is kept" `Quick test_grouping;
         Alcotest.test_case "units are kept" `Quick test_units;
         Alcotest.test_case "comments, layout and let position" `Quick
           test_layout_and_let_position;
         Alcotest.test_case "a quoted string is not a name" `Quick test_string_literal ]);
      ("boolean keyword arguments",
       [ Alcotest.test_case "gh#1117 inclusive_end = false" `Quick
           (test_boolean_keyword_argument "false");
         Alcotest.test_case "gh#1117 inclusive_end = true" `Quick
           (test_boolean_keyword_argument "true") ]);
      ("text as written",
       Alcotest.test_case "each declaration carries its own bytes" `Quick
         test_source_text
       :: List.map (fun path ->
           Alcotest.test_case (relative path) `Quick
             (test_golden_source_text path))
         (golden_models ()));
      ("rendering loss",
       [ Alcotest.test_case "on compiled rates" `Quick test_katex_loss_compiled;
         Alcotest.test_case "every reason, on hand-built trees" `Quick
           test_katex_loss_reasons ]);
      ("doc comments",
       [ Alcotest.test_case "a doc comment changes renderings only, and no model"
           `Quick test_doc_comment_is_not_a_model_difference ]);
      ("unclassified names",
       [ Alcotest.test_case "an unresolved name is warned, and the compile succeeds"
           `Quick test_unresolved_is_warned;
         Alcotest.test_case "a model with none is silent" `Quick
           test_resolved_is_silent ]) ]
