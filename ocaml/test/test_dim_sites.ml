(* Coverage of Dim_sites: every expression of a model is the expression of a
   site, or lies under a key the checker exempts.

   [Dim_sites.sites] destructures the IR with closed patterns, so a new field
   or constructor does not compile there until someone writes its rule. What
   the compiler cannot see is a field bound to [_] although it holds an
   expression. This test sees it, independently of Dim_sites: it serializes
   each model with Serde (the IR writer, whose completeness the round-trip
   tests check), collects every outermost expression node of the JSON, and
   requires the multiset of those nodes to be covered by the multiset of site
   expressions, serialized the same way. An uncovered node is reported with
   its JSON path.

   A node is an object with exactly one key naming an expression constructor.
   The keys exempted are derivative maps written by autodiff (a test of
   autodiff, not of the user's model), the lineage weights the expander
   derives from a rate, and the bindings LICM creates after the checker runs. *)

let expr_keys = [
  "const"; "param"; "const_ref"; "pop"; "pop_sum"; "reduce"; "binding_ref";
  "per_eval_ref"; "time"; "dt"; "projected"; "obs_column_ref"; "obs_anchor";
  "forcing_basis"; "bin_op"; "un_op"; "cond"; "time_func"; "table_lookup";
  "unchecked_dim"; "unit_literal";
]

let exempt_keys = [
  "rate_grad"; "rate_state_grad"; "sigma_sq_grad"; "grad"; "proj_grad";
  "ic_grad"; "projection_state_grad"; "per_eval_bindings"; "lineage";
]

(* The leaves without payload are written with a null value; a reducer such
   as `{"time": {"first_above": ...}}` shares the key and is not one. *)
let null_leaves = [ "time"; "dt"; "projected" ]

let is_expr_node = function
  | `Assoc [ (k, v) ] ->
    List.mem k expr_keys && (not (List.mem k null_leaves) || v = `Null)
  | _ -> false

(* (path, canonical JSON) of every outermost expression node. *)
let expr_nodes (j : Yojson.Safe.t) =
  let out = ref [] in
  let rec walk path j =
    if is_expr_node j then out := (path, Yojson.Safe.to_string j) :: !out
    else match j with
      | `Assoc kvs ->
        List.iter (fun (k, v) ->
          if not (List.mem k exempt_keys) then walk (path ^ "." ^ k) v) kvs
      | `List vs -> List.iteri (fun i v -> walk (Printf.sprintf "%s[%d]" path i) v) vs
      | `Tuple vs -> List.iteri (fun i v -> walk (Printf.sprintf "%s(%d)" path i) v) vs
      | `Variant (_, Some v) -> walk path v
      | `Variant (_, None) | `Null | `Bool _ | `Int _ | `Intlit _ | `Float _ | `String _ -> ()
  in
  walk "" j;
  List.rev !out

let site_json (s : Dim_sites.site) =
  Yojson.Safe.to_string (match s.expr with
    | Dim_sites.E e -> Serde.expr_to_json e
    | Scalar se -> Serde.scalar_expr_to_json se
    | Contrast ce -> Serde.contrast_expr_to_json ce)

(* The nodes no site accounts for, counting multiplicity. *)
let uncovered (m : Ir.model) =
  let available = Hashtbl.create 256 in
  List.iter (fun s ->
    let k = site_json s in
    Hashtbl.replace available k (1 + Option.value ~default:0 (Hashtbl.find_opt available k)))
    (Dim_sites.sites m);
  List.filter (fun (_, j) ->
    match Hashtbl.find_opt available j with
    | Some n when n > 0 -> Hashtbl.replace available j (n - 1); false
    | Some _ | None -> true) (expr_nodes (Serde.model_to_json m))

let repo_root =
  let rec up d n =
    if Sys.file_exists (Filename.concat d "ir/VERSION") then d
    else if n = 0 then failwith "repo root not found"
    else up (Filename.dirname d) (n - 1) in
  up (Sys.getcwd ()) 12

let ir_files () =
  let golden = Filename.concat repo_root "ocaml/golden" in
  let fixtures = Filename.concat repo_root "tests/fixtures" in
  let in_dir d =
    if Sys.file_exists d && Sys.is_directory d then
      Sys.readdir d |> Array.to_list |> List.sort compare
      |> List.filter (fun f -> Filename.check_suffix f ".ir.json")
      |> List.map (Filename.concat d)
    else [] in
  let fixture_dirs =
    if Sys.file_exists fixtures then
      Sys.readdir fixtures |> Array.to_list |> List.sort compare
      |> List.map (fun d -> Filename.concat (Filename.concat fixtures d) "ir")
    else [] in
  in_dir golden @ List.concat_map in_dir fixture_dirs

let load path = Serde.model_of_json (Yojson.Safe.from_file path |> Yojson.Safe.Util.member "model")

let test_corpus_covered () =
  let files = ir_files () in
  Alcotest.(check bool) "found IR files" true (List.length files >= 60);
  let problems =
    List.concat_map (fun f ->
      List.map (fun (path, j) ->
        Printf.sprintf "%s %s: %s" (Filename.basename f) path
          (if String.length j > 80 then String.sub j 0 80 ^ "…" else j))
        (uncovered (load f))) files in
  if problems <> [] then
    Alcotest.failf "expression positions no site covers:\n  %s"
      (String.concat "\n  " problems)

(* The test can fail: an expression position the sites omit is reported. *)
let test_detects_a_missing_position () =
  let m = load (Filename.concat repo_root "ocaml/golden/sir_basic.ir.json") in
  let sites = Dim_sites.sites m in
  let without_rates = List.filter (fun (s : Dim_sites.site) ->
    not (String.length s.role > 0 && Filename.check_suffix s.role " rate")) sites in
  Alcotest.(check bool) "the rates are sites" true (List.length without_rates < List.length sites);
  let available = Hashtbl.create 64 in
  List.iter (fun s -> Hashtbl.replace available (site_json s) ()) without_rates;
  let missed = List.filter (fun (_, j) -> not (Hashtbl.mem available j))
      (expr_nodes (Serde.model_to_json m)) in
  Alcotest.(check bool) "dropping the rate sites leaves rates uncovered" true
    (List.exists (fun (p, _) -> Filename.check_suffix p ".rate") missed)

let () =
  Alcotest.run "dim_sites" [
    "coverage", [
      Alcotest.test_case "every expression of every IR golden is a site" `Quick test_corpus_covered;
      Alcotest.test_case "a missing position is detected" `Quick test_detects_a_missing_position;
    ];
  ]
