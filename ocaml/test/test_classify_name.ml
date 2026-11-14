(* What a name resolves to: `Expander.classify_name` and
   `Expander.classify_indexed_name`, the one precedence chain that
   `resolve_expr` matches on.

   Two groups.

   "precedence" pins each adjacent pair of the chain. On a valid model a name
   lives in one namespace (`check_declaration_names`, E278), so the order
   cannot change a compiled result and no golden can see it; these cases build
   the collision on purpose, take the context the failed compile leaves
   behind, and ask the classifier directly.

   "goldens" compiles every model in `ocaml/golden/` and checks that each name
   the model's expressions use classifies to a declared thing. Covered: the
   rate and branch weights of every transition, every `let` body, every `init`
   value (expression or law arguments) and every observation likelihood
   argument. Not covered: `via` law arguments, forcing and table definitions,
   interventions and events, projections, schedules, guards, quantities and
   contrasts. *)

open Ast

let parse src =
  let lexbuf = Lexing.from_string src in
  try Parser.file Lexer.token lexbuf
  with _ -> Alcotest.fail "parse failed"

let ctx_of ?(source_dir = "") ?(filename = "<input>") name src =
  let (_model, ctx, _summary) =
    Expander.expand_detail ~source_dir ~filename name (parse src) in
  ctx

let classify ?(obs_aux_cols = []) ?(suppress_hoist = false) ctx name =
  Expander.classify_name ctx ~obs_aux_cols ~suppress_hoist name

let class_name : Expander.name_class -> string = function
  | Name_obs_column         -> "obs_column"
  | Name_compartment_cell   -> "compartment_cell"
  | Name_compartment        -> "compartment"
  | Name_param              -> "param"
  | Name_param_cell         -> "param_cell"
  | Name_let_cyclic         -> "let_cyclic"
  | Name_let_indexed_bare _ -> "let_indexed_bare"
  | Name_let_const _        -> "let_const"
  | Name_let_hoisted _      -> "let_hoisted"
  | Name_let_inlined _      -> "let_inlined"
  | Name_forcing            -> "forcing"
  | Name_time               -> "time"
  | Name_dt                 -> "dt"
  | Name_projected          -> "projected"
  | Name_pi                 -> "pi"
  | Name_e                  -> "e"
  | Name_origin             -> "origin"
  | Name_unresolved         -> "unresolved"

let indexed_class_name : Expander.indexed_name_class -> string = function
  | Indexed_table _      -> "table"
  | Indexed_let_cyclic   -> "let_cyclic"
  | Indexed_let _        -> "let"
  | Indexed_shaped_let _ -> "shaped_let"
  | Indexed_forcing _    -> "forcing"
  | Indexed_param        -> "param"
  | Indexed_cell         -> "cell"

let check_class ?obs_aux_cols ?suppress_hoist ctx name expected =
  Alcotest.(check string) name expected
    (class_name (classify ?obs_aux_cols ?suppress_hoist ctx name))

let has_error ctx code =
  List.exists (fun d ->
      d.Diagnostics.severity = Diagnostics.Error && d.Diagnostics.code = code)
    ctx.Expander.diags.Diagnostics.diags

(* ── Precedence ──────────────────────────────────────────────────────────── *)

(* A plain model: every kind of name once, no collisions. *)
let plain = {|
  time_unit = 'days
  dimensions { age = [child, adult] }
  compartments { S, I, R, V }
  stratify(by = age, only = [S])
  let N = S + I + R + V
  let k : rate = 0.5
  let scaled = beta * 2
  let per_age[a in age] = S[a]
  parameters {
    beta  : rate in [0.001, 2.0]
    gamma : rate in [0.001, 1.0]
    sus[age] : probability in [0.0, 1.0]
  }
  forcing {
    season : sinusoidal 'ratio { baseline = 1.0  amplitude = 0.2  period = 365.0  phase = 0.0 }
  }
  transitions {
    infection[a in age] : S[a] --> I @ beta * sus[a] * season * S[a] * I / N
    recovery            : I --> R    @ gamma * I
    waning              : R --> V    @ k * R + 0 * scaled
  }
  init { S[child] = 500  S[adult] = 490  I = 10 }
  simulate { from = 0 'days  to = 10 'days }
|}

let test_each_kind () =
  let ctx = ctx_of "plain" plain in
  Alcotest.(check bool) "the model compiles" false
    (Diagnostics.has_errors ctx.Expander.diags);
  check_class ctx "S" "compartment";            (* stratified: a base name *)
  check_class ctx "S_child" "compartment_cell"; (* its cell, written literally *)
  check_class ctx "beta" "param";
  check_class ctx "sus_child" "param_cell";
  check_class ctx "N" "let_hoisted";
  check_class ctx "k" "let_const";
  check_class ctx "scaled" "let_inlined";
  check_class ctx "per_age" "let_indexed_bare";
  check_class ctx "season" "forcing";
  check_class ctx "t" "time";
  check_class ctx "dt" "dt";
  check_class ctx "projected" "projected";
  check_class ctx "pi" "pi";
  check_class ctx "e" "e";
  check_class ctx "origin" "origin";
  check_class ctx "nope" "unresolved"

(* literal cell before base compartment: an unstratified compartment is in
   both tables under one name, and the chain reports it as the cell. *)
let test_cell_before_base () =
  let ctx = ctx_of "plain" plain in
  check_class ctx "I" "compartment_cell"

(* The hoisting context is an argument: the same `let` is inlined where the
   enclosing transition forbids hoisting. *)
let test_suppress_hoist () =
  let ctx = ctx_of "plain" plain in
  check_class ~suppress_hoist:true ctx "N" "let_inlined"

(* observation column before everything: inside a likelihood whose stream
   declares the column, the name is the column even if the model declares it
   too. (The resolver then reports the collision, E279.) *)
let test_obs_column_first () =
  let ctx = ctx_of "plain" plain in
  check_class ~obs_aux_cols:["beta"] ctx "beta" "obs_column";
  check_class ~obs_aux_cols:["S"] ctx "S" "obs_column";
  check_class ~obs_aux_cols:["t"] ctx "t" "obs_column";
  check_class ~obs_aux_cols:["tested"] ctx "beta" "param"

let collide ~decls =
  ctx_of "collide" (Printf.sprintf {|
  time_unit = 'days
  dimensions { age = [child, adult] }
  compartments { S, I }
  %s
  transitions { infection : S --> I @ 0.1 * S }
  init { S = 99  I = 1 }
  simulate { from = 0 'days  to = 10 'days }
|} decls)

(* compartment before parameter: once for an unstratified compartment (found
   as its own cell) and once for a stratified one (found as the base name),
   since those are two separate steps of the chain. *)
let test_compartment_before_param () =
  let ctx = collide ~decls:"parameters { I : rate in [0.1, 1.0] }" in
  Alcotest.(check bool) "the collision is E278" true (has_error ctx "E278");
  check_class ctx "I" "compartment_cell";
  let ctx = collide ~decls:{|stratify(by = age, only = [S])
    parameters { S : rate in [0.1, 1.0] }|} in
  Alcotest.(check bool) "the collision is E278" true (has_error ctx "E278");
  check_class ctx "S" "compartment"

(* scalar parameter before an expanded indexed-parameter cell of that name *)
let test_param_before_param_cell () =
  let ctx = collide ~decls:{|parameters {
      sus[age]  : probability in [0.0, 1.0]
      sus_child : probability in [0.0, 1.0]
    }|} in
  Alcotest.(check bool) "the collision is E278" true (has_error ctx "E278");
  check_class ctx "sus_child" "param"

(* parameter before let *)
let test_param_before_let () =
  let ctx = collide ~decls:{|parameters { x : rate in [0.1, 1.0] }
    let x = S + I|} in
  Alcotest.(check bool) "the collision is E278" true (has_error ctx "E278");
  check_class ctx "x" "param"

(* let before forcing *)
let test_let_before_forcing () =
  let ctx = collide ~decls:{|let x = S + I
    forcing {
      x : sinusoidal 'ratio { baseline = 1.0  amplitude = 0.2  period = 365.0  phase = 0.0 }
    }|} in
  Alcotest.(check bool) "the collision is E278" true (has_error ctx "E278");
  check_class ctx "x" "let_hoisted"

(* a cyclic let is reported as cyclic before it is looked up as a let *)
let test_cyclic_before_let () =
  let ctx = collide ~decls:"let a = a + 1" in
  Alcotest.(check bool) "the cycle is E201" true (has_error ctx "E201");
  check_class ctx "a" "let_cyclic"

(* A declared name is found before a built-in of the same spelling. *)
let test_declared_before_builtin () =
  let ctx = collide ~decls:"parameters { e : rate in [0.1, 1.0] }" in
  check_class ctx "e" "param"

(* The indexed chain, one case per constructor. *)
let test_indexed_kinds () =
  let ctx = ctx_of "indexed" {|
  time_unit = 'days
  dimensions { age = [child, adult] }
  compartments { S, I }
  stratify(by = age, only = [S])
  tables { w : age = [1.0, 2.0] }
  let per_age[a in age] = S[a]
  let shaped : age = [1.0, 2.0]
  parameters {
    beta : rate in [0.001, 2.0]
    sus[age] : probability in [0.0, 1.0]
  }
  forcing {
    f[a in age] : sinusoidal 'ratio { baseline = 1.0  amplitude = 0.2  period = 365.0  phase = 0.0 }
  }
  transitions {
    infection[a in age] : S[a] --> I @ beta * sus[a] * w[a] * f[a] * S[a]
  }
  init { S[child] = 50  S[adult] = 49  I = 1 }
  simulate { from = 0 'days  to = 10 'days }
|} in
  Alcotest.(check bool) "the model compiles" false
    (Diagnostics.has_errors ctx.Expander.diags);
  let check name expected =
    Alcotest.(check string) name expected
      (indexed_class_name (Expander.classify_indexed_name ctx name)) in
  check "w" "table";
  check "per_age" "let";
  check "shaped" "shaped_let";
  check "f" "forcing";
  check "sus" "param";
  check "S" "cell";
  check "nope" "cell"

(* ── Every golden ────────────────────────────────────────────────────────── *)

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

let binder_vars (ibs : index_binding list) : string list =
  List.concat_map (function
      | IBind (v, _)         -> [v]
      | IConsec (v, next, _) -> [v; next]
      | IComp v              -> [v]) ibs

(* Every bare name and every indexed head in [e], each with the binders in
   scope where it appears. Index items are tokens the expander substitutes
   (`index_item_to_str`), never names it resolves, so they are not visited. *)
let rec names ~(binders : string list) (e : expr)
    : ([ `Bare | `Head ] * string * string list) list =
  match e with
  | EConst _ | EUnit _ | EObsAccess _ | ERunMember _ -> []
  | EIdent (n, _)     -> [ (`Bare, n, binders) ]
  | EIndex (n, _, _)  -> [ (`Head, n, binders) ]
  | EBinOp (_, a, b)  -> names ~binders a @ names ~binders b
  | EUnOp (_, a)      -> names ~binders a
  | ESum (v, _, _, body, _) -> names ~binders:(v :: binders) body
  | ECond (p, a, b)   -> names ~binders p @ names ~binders a @ names ~binders b
  (* `unchecked_dim(<expr>, dim = <keyword>, reason = "...")`: only the
     positional argument is an expression; `dim` is one of a fixed set of
     dimension keywords, read by the expander without name resolution. *)
  | EFuncCall ("unchecked_dim", args) ->
    List.concat_map (fun (k, a) -> if k = "" then names ~binders a else []) args
  | EFuncCall (_, args) -> List.concat_map (fun (_, a) -> names ~binders a) args
  | EList es          -> List.concat_map (names ~binders) es
  | ERange (a, b)     -> names ~binders a @ names ~binders b

(* The expressions this test covers, each with its declaration's binders, the
   auxiliary columns in scope (likelihood arguments only) and a label. *)
let covered_exprs (decls : declaration list)
    : (string * string list * string list * expr) list =
  List.concat_map (function
      | DTransitions trs ->
        List.concat_map (fun (tr : transition_decl) ->
            let binders = binder_vars tr.trindices in
            let rate = match tr.trdyn with Rate e -> [e] | Via _ -> [] in
            let weights = match tr.trdst with
              | DstSum _       -> []
              | DstBranch arms -> List.map snd arms in
            List.map (fun e -> ("transition " ^ tr.trname, binders, [], e))
              (rate @ weights)) trs
      | DLet lb ->
        [ ("let " ^ lb.lname, binder_vars lb.lindices, [], lb.lbody) ]
      | DInit entries ->
        List.concat_map (fun (ie : init_entry) ->
            let binders = binder_vars ie.ibindings in
            let exprs = match ie.ivalue with
              | IVExpr e      -> [e]
              | IVLaw (l, _)  -> List.map snd (likelihood_kwargs l) in
            List.map (fun e -> ("init " ^ ie.icomp, binders, [], e)) exprs)
          entries
      | DObservations obs ->
        List.concat_map (fun (od : obs_decl) ->
            match od.omeasurement with
            | None -> []
            | Some m ->
              (* The auxiliary columns, as the expander computes them: the
                 declared value columns other than the scored one. *)
              let aux = List.filter_map (fun c ->
                  match c.oc_role with
                  | ColValue _ when c.oc_name <> m.om_scored -> Some c.oc_name
                  | _ -> None)
                  (Option.value ~default:[] od.ocolumns) in
              List.map (fun (_, e) ->
                  ("observation " ^ od.oname, binder_vars od.oindices, aux, e))
                (likelihood_kwargs m.om_lik)) obs
      | _ -> []) decls

let declared_compartments (decls : declaration list) : string list =
  List.concat_map (function
      | DCompartments cs -> List.map (fun (c : compartment_decl) -> c.cname) cs
      | _ -> []) decls

let golden_dir () = Filename.concat repo_root "ocaml/golden"

let golden_models () =
  Sys.readdir (golden_dir ())
  |> Array.to_list
  |> List.filter (fun f -> Filename.check_suffix f ".camdl")
  |> List.sort compare

let test_golden file () =
  let dir = golden_dir () in
  let src = read_file (Filename.concat dir file) in
  let decls = parse src in
  let (_model, ctx, _summary) =
    Expander.expand_detail ~source_dir:dir ~filename:file
      (Filename.remove_extension file) decls in
  if Diagnostics.has_errors ctx.Expander.diags then
    Alcotest.failf "%s does not compile through expand_detail" file;
  let declared = declared_compartments decls in
  let n_checked = ref 0 in
  List.iter (fun (where, binders, aux, e) ->
      List.iter (fun (shape, name, binders) ->
          (* A binder is an index token, substituted before resolution. *)
          if not (List.mem name binders) then begin
            incr n_checked;
            match shape with
            | `Bare ->
              (match Expander.classify_name ctx ~obs_aux_cols:aux
                       ~suppress_hoist:false name with
               | Name_unresolved ->
                 (* The context `expand_detail` returns is the one after `via`
                    lowering. A `via hyper_erlang` source compartment is
                    replaced there by its per-branch stage compartments and
                    every reference to it rewritten to their sum, so the
                    declared name is no longer a compartment of that context.
                    That one case is accepted; anything else is a failure. *)
                 let lowered_away =
                   List.mem name declared
                   && not (Hashtbl.mem ctx.Expander.comp_tbl name) in
                 if not lowered_away then
                   Alcotest.failf "%s, %s: bare name '%s' is unresolved"
                     file where name
               | _ -> ())
            | `Head ->
              (match Expander.classify_indexed_name ctx name with
               | Indexed_cell ->
                 (* The catch-all case: a declared model reaches it only for
                    an indexed compartment. *)
                 if not (Hashtbl.mem ctx.Expander.comp_tbl name) then
                   Alcotest.failf
                     "%s, %s: indexed head '%s' is no table, let, forcing, \
                      parameter or compartment" file where name
               | _ -> ())
          end)
        (names ~binders e))
    (covered_exprs decls);
  ignore !n_checked

(* The walk must see names at all: a walker that visited nothing would pass
   every model. *)
let test_walk_is_not_vacuous () =
  let total = List.fold_left (fun acc file ->
      let decls = parse (read_file (Filename.concat (golden_dir ()) file)) in
      acc + List.fold_left (fun n (_, binders, _, e) ->
          n + List.length (names ~binders e)) 0 (covered_exprs decls))
      0 (golden_models ()) in
  Alcotest.(check bool)
    (Printf.sprintf "names visited across the goldens: %d" total)
    true (total > 500)

let () =
  Alcotest.run "classify_name" [
    "precedence", [
      Alcotest.test_case "each kind of name" `Quick test_each_kind;
      Alcotest.test_case "literal cell before base compartment" `Quick test_cell_before_base;
      Alcotest.test_case "hoisting context is an argument" `Quick test_suppress_hoist;
      Alcotest.test_case "observation column first" `Quick test_obs_column_first;
      Alcotest.test_case "compartment before parameter" `Quick test_compartment_before_param;
      Alcotest.test_case "parameter before parameter cell" `Quick test_param_before_param_cell;
      Alcotest.test_case "parameter before let" `Quick test_param_before_let;
      Alcotest.test_case "let before forcing" `Quick test_let_before_forcing;
      Alcotest.test_case "cyclic let before let" `Quick test_cyclic_before_let;
      Alcotest.test_case "declared name before built-in" `Quick test_declared_before_builtin;
      Alcotest.test_case "indexed heads" `Quick test_indexed_kinds;
    ];
    "goldens",
    Alcotest.test_case "the walk visits names" `Quick test_walk_is_not_vacuous
    :: List.map (fun file -> Alcotest.test_case file `Quick (test_golden file))
         (golden_models ());
  ]
