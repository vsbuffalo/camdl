(* Phase-2 staged-residence `via erlang(...)` lowering: IR-level tests.

   A `via erlang` transition desugars to EXACTLY the manual stratified-
   `consecutive` staging a user writes today. The load-bearing test (T1) pins
   the desugar against the trusted, validated `seir_erlang.camdl` golden:
   compiling the `via` form yields an IR equal — modulo the stage compartment /
   transition names — to the hand-written staged golden. We assert at the IR
   level (compartments, stoichiometry, constant-folded rate ASTs, init) so it
   cannot pass vacuously. *)

let () = Compiler.no_dim_check := true

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let compile_ok (src : string) : Ir.model =
  match Compiler.compile ~name:"via_test" src with
  | Ok m    -> m
  | Error e -> Alcotest.failf "compile failed: %s" e

let contains_substring ~needle s =
  let nl = String.length needle and sl = String.length s in
  if nl = 0 then true
  else if nl > sl then false
  else
    let rec loop i =
      if i > sl - nl then false
      else if String.sub s i nl = needle then true
      else loop (i + 1)
    in loop 0

let compile_expect_error_code ~code ~contains src =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"via_test" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

(* Canonicalize a model's STAGE names so two structurally-equivalent stagings
   (golden `E_e1,E_e2,E_e3`; via `E_s1,E_s2,E_s3`) compare equal. We rename the
   stage cells of the single staged base compartment to a uniform `<base>_<i>`
   (1-indexed, in compartment-declaration order), rewrite every reference to
   them (stoichiometry, Pop, PopSum, init), and DROP transition names + metadata
   (which encode the stage labels). Everything else — base compartments,
   stoichiometry signs, rate ASTs, init values — is compared verbatim. *)

(* Rename a single compartment name through the map (identity if absent). *)
let rename_name map n = match List.assoc_opt n map with Some r -> r | None -> n

let rec rename_expr map (e : Ir.expr) : Ir.expr =
  match e with
  | Ir.Pop n      -> Ir.Pop (rename_name map n)
  | Ir.PopSum ns  -> Ir.PopSum (List.map (rename_name map) ns)
  | Ir.BinOp b    -> Ir.BinOp { b with left = rename_expr map b.left;
                                        right = rename_expr map b.right }
  | Ir.UnOp u     -> Ir.UnOp { u with arg = rename_expr map u.arg }
  | Ir.Cond c     -> Ir.Cond { pred  = rename_expr map c.pred;
                               then_ = rename_expr map c.then_;
                               else_ = rename_expr map c.else_ }
  | Ir.Reduce ts  -> Ir.Reduce (List.map (rename_expr map) ts)
  | Ir.TableLookup (n, idxs) -> Ir.TableLookup (n, List.map (rename_expr map) idxs)
  | Ir.UncheckedDim u -> Ir.UncheckedDim { u with inner = rename_expr map u.inner }
  | other         -> other

(* A canonical, name-agnostic view of a transition: its renamed, sorted
   stoichiometry and its renamed rate. Name + metadata + draw + grad dropped. *)
type canon_tr = (string * int) list * Ir.expr

let canon_transition map (t : Ir.transition) : canon_tr =
  let stoich =
    List.map (fun (n, d) -> (rename_name map n, d)) t.Ir.stoichiometry
    |> List.sort compare
  in
  (stoich, rename_expr map t.Ir.rate)

(* Find the single staged base compartment (the one with > 1 cell of the form
   `<base>_<level>` for a declared base) and build the rename map cell→`base_i`.
   Returns the map plus the canonical stage cell names. *)
let stage_rename_map (m : Ir.model) : (string * string) list =
  let comp_names = List.map (fun (c : Ir.compartment) -> c.Ir.name) m.Ir.compartments in
  (* The base compartments are in model_structure. *)
  let bases = match m.Ir.model_structure with
    | Some ms -> ms.Ir.base_compartments
    | None -> Alcotest.fail "model has no model_structure"
  in
  (* For each base, its expansion cells in compartment-list order. A staged base
     has > 1 cell; an unstratified base maps to itself (no rename). *)
  List.concat_map (fun base ->
    let cells =
      List.filter (fun n ->
        n = base || (String.length n > String.length base
                     && String.sub n 0 (String.length base + 1) = base ^ "_"))
        comp_names
    in
    match cells with
    | [ _single ] -> []   (* unstratified: no rename *)
    | many ->
      List.mapi (fun i cell -> (cell, Printf.sprintf "%s_%d" base (i + 1))) many
  ) bases

let canon_init map (ic : Ir.initial_conditions) : (string * float) list =
  match ic with
  | Ir.Explicit kvs ->
    List.map (fun (k, v) -> (rename_name map k, v)) kvs |> List.sort compare
  | _ -> Alcotest.fail "expected explicit init"

let canon_comps map (m : Ir.model) : string list =
  List.map (fun (c : Ir.compartment) -> rename_name map c.Ir.name) m.Ir.compartments
  |> List.sort compare

(* ── T1 anchor: via form ≡ seir_erlang golden (modulo stage names) ───────── *)

let golden_dir =
  List.find (fun d -> Sys.file_exists d && Sys.is_directory d)
    [ "../../golden"; "../golden"; "golden" ]

let read_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n; close_in ic; Bytes.to_string s

(* The `via` form of seir_erlang: the exact model in the Phase-2 brief. *)
let seir_via_src =
  "time_unit = 'days\n\
   compartments { S, E, I, R }\n\
   parameters {\n\
  \  beta  : rate in [0.001, 2.0]\n\
  \  sigma : rate in [0.01, 1.0]\n\
  \  gamma : rate in [0.01, 1.0]\n\
   }\n\
   transitions {\n\
  \  infection : S --> E @ beta * S * I / (S + E + I + R)\n\
  \  onset     : E --> I via erlang(stages = 3, rate = sigma)\n\
  \  recovery  : I --> R @ gamma * I\n\
   }\n\
   init { S = 990  E = 5  I = 5 }\n\
   simulate { from = 0 'days  to = 160 'days }\n"

let test_t1_anchor_via_equals_golden () =
  let via_m = compile_ok seir_via_src in
  (* The trusted, hand-written staged golden. *)
  let golden_json = read_file (Filename.concat golden_dir "seir_erlang.ir.json") in
  let golden_m = match Serde.model_of_string golden_json with
    | Ok m    -> m
    | Error e -> Alcotest.failf "bad golden JSON: %s" e
  in
  let map_via    = stage_rename_map via_m in
  let map_golden = stage_rename_map golden_m in
  (* Compartments (canonicalized) must match as a set. *)
  Alcotest.(check (list string)) "compartments (canonical)"
    (canon_comps map_golden golden_m) (canon_comps map_via via_m);
  (* Init must match (canonicalized). *)
  Alcotest.(check (list (pair string (float 1e-9)))) "init (canonical)"
    (canon_init map_golden golden_m.Ir.initial_conditions)
    (canon_init map_via via_m.Ir.initial_conditions);
  (* Transitions: same multiset of (sorted-stoichiometry, rate-AST), modulo
     names. Sorting by the canonical stoichiometry gives a stable order. *)
  let via_canon =
    List.map (canon_transition map_via) via_m.Ir.transitions |> List.sort compare in
  let golden_canon =
    List.map (canon_transition map_golden) golden_m.Ir.transitions |> List.sort compare in
  if List.length via_canon <> List.length golden_canon then
    Alcotest.failf "transition count: golden %d, via %d"
      (List.length golden_canon) (List.length via_canon);
  List.iter2 (fun (g_stoich, g_rate) (v_stoich, v_rate) ->
    if g_stoich <> v_stoich then
      Alcotest.failf "stoichiometry mismatch:\n golden %s\n via    %s"
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) g_stoich))
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) v_stoich));
    if g_rate <> v_rate then
      Alcotest.failf "rate AST mismatch for stoich %s:\n golden %s\n via    %s"
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) g_stoich))
        (Yojson.Safe.to_string (Serde.expr_to_json g_rate))
        (Yojson.Safe.to_string (Serde.expr_to_json v_rate))
  ) golden_canon via_canon

(* ── Stage rate: rate = sigma ⇒ 3*sigma; mean = tau ⇒ 3/tau ─────────────── *)

(* Pull the per-stage chain rate (the first staged transition: E_s1 --> E_s2). *)
let first_chain_rate (m : Ir.model) : Ir.expr =
  let t =
    List.find (fun (t : Ir.transition) ->
      match t.Ir.stoichiometry with
      | [ (a, -1); (b, 1) ] ->
        contains_substring ~needle:"_s1" a && contains_substring ~needle:"_s2" b
      | _ -> false)
      m.Ir.transitions
  in
  t.Ir.rate

let model_with_onset onset =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, E, I, R }\n\
     parameters { beta : rate  sigma : rate  gamma : rate  tau : positive }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S * I / (S + E + I + R)\n\
    \  %s\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     init { S = 990  E = 5  I = 5 }\n" onset

let test_stage_rate_from_rate () =
  let m = compile_ok (model_with_onset
    "onset : E --> I via erlang(stages = 3, rate = sigma)") in
  (* per-stage rate = (3 * sigma) * E_s1 *)
  let expected =
    Ir.BinOp { op = Ir.Mul;
               left  = Ir.BinOp { op = Ir.Mul; left = Ir.Const 3.0; right = Ir.Param "sigma" };
               right = Ir.Pop "E_s1" } in
  Alcotest.(check bool) "rate = sigma ⇒ (3*sigma)*E_s1" true
    (first_chain_rate m = expected)

let test_stage_rate_from_mean () =
  let m = compile_ok (model_with_onset
    "onset : E --> I via erlang(stages = 3, mean = tau)") in
  (* per-stage rate = (3 / tau) * E_s1 *)
  let expected =
    Ir.BinOp { op = Ir.Mul;
               left  = Ir.BinOp { op = Ir.Div; left = Ir.Const 3.0; right = Ir.Param "tau" };
               right = Ir.Pop "E_s1" } in
  Alcotest.(check bool) "mean = tau ⇒ (3/tau)*E_s1" true
    (first_chain_rate m = expected)

(* ── Inflow + init land in stage 1; bare E sums in the FOI ───────────────── *)

let test_inflow_and_init_land_in_stage1 () =
  let m = compile_ok seir_via_src in
  (* infection lands in E_s1, not a bare E. *)
  let inf = List.find (fun (t : Ir.transition) -> t.Ir.name = "infection") m.Ir.transitions in
  Alcotest.(check bool) "infection --> E_s1" true
    (List.mem ("E_s1", 1) inf.Ir.stoichiometry);
  (* init: E_s1 = 5, no bare E key. *)
  (match m.Ir.initial_conditions with
   | Ir.Explicit kvs ->
     Alcotest.(check bool) "init has E_s1 = 5" true (List.mem ("E_s1", 5.0) kvs);
     Alcotest.(check bool) "init has no bare E" false (List.mem_assoc "E" kvs)
   | _ -> Alcotest.fail "expected explicit init")

let test_bare_E_sums_in_foi () =
  let m = compile_ok seir_via_src in
  let inf = List.find (fun (t : Ir.transition) -> t.Ir.name = "infection") m.Ir.transitions in
  (* The FOI denominator (S+E+I+R) must sum all three E stages. *)
  let rec collect_popsums = function
    | Ir.PopSum ns -> [ ns ]
    | Ir.BinOp b   -> collect_popsums b.left @ collect_popsums b.right
    | Ir.UnOp u    -> collect_popsums u.arg
    | Ir.Cond c    -> collect_popsums c.pred @ collect_popsums c.then_ @ collect_popsums c.else_
    | _            -> []
  in
  let sums = collect_popsums inf.Ir.rate in
  Alcotest.(check bool) "FOI denominator sums all E stages" true
    (List.exists (fun ns ->
       List.mem "E_s1" ns && List.mem "E_s2" ns && List.mem "E_s3" ns
       && List.mem "S" ns && List.mem "I" ns && List.mem "R" ns) sums)

(* ── Two staged residences in one model: chain-into-chain redirect ───────── *)

let test_two_staged_residences () =
  let src =
    "time_unit = 'days\n\
     compartments { S, E, I, R }\n\
     parameters { beta : rate  sigma : rate  gamma : rate }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S * I / (S + E + I + R)\n\
    \  onset     : E --> I via erlang(stages = 3, rate = sigma)\n\
    \  recovery  : I --> R via erlang(stages = 3, rate = gamma)\n\
     }\n\
     init { S = 990  E = 5  I = 5 }\n"
  in
  let m = compile_ok src in
  let names = List.map (fun (c : Ir.compartment) -> c.Ir.name) m.Ir.compartments in
  List.iter (fun n -> Alcotest.(check bool) (n ^ " exists") true (List.mem n names))
    [ "E_s1"; "E_s2"; "E_s3"; "I_s1"; "I_s2"; "I_s3" ];
  (* E's chain exits into I_s1 (the first stage of staged I). *)
  let onset = List.find (fun (t : Ir.transition) -> t.Ir.name = "onset") m.Ir.transitions in
  Alcotest.(check bool) "onset E_s3 --> I_s1" true
    (List.mem ("E_s3", -1) onset.Ir.stoichiometry
     && List.mem ("I_s1", 1) onset.Ir.stoichiometry)

(* ── Validation errors: one distinct E-code each, naming the transition ──── *)

let test_err_non_positive_integer_stages () =
  compile_expect_error_code ~code:"E244" ~contains:"onset"
    (model_with_onset "onset : E --> I via erlang(stages = 3.5, rate = sigma)");
  compile_expect_error_code ~code:"E244" ~contains:"onset"
    (model_with_onset "onset : E --> I via erlang(stages = 0, rate = sigma)")

let test_err_both_mean_and_rate () =
  compile_expect_error_code ~code:"E245" ~contains:"onset"
    (model_with_onset "onset : E --> I via erlang(stages = 3, rate = sigma, mean = tau)")

let test_err_neither_mean_nor_rate () =
  compile_expect_error_code ~code:"E245" ~contains:"onset"
    (model_with_onset "onset : E --> I via erlang(stages = 3)")

let test_err_single_exit_violation () =
  (* E drained by both the `via` and an ordinary `@` exit. *)
  let src =
    "time_unit = 'days\n\
     compartments { S, E, I, R, D }\n\
     parameters { beta : rate  sigma : rate  gamma : rate  mu : rate }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S * I / (S + E + I + R)\n\
    \  onset     : E --> I via erlang(stages = 3, rate = sigma)\n\
    \  death     : E --> D @ mu * E\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     init { S = 990  E = 5  I = 5 }\n"
  in
  compile_expect_error_code ~code:"E246" ~contains:"death" src

let test_err_hyper_erlang_deferred () =
  compile_expect_error_code ~code:"E243" ~contains:"not yet supported"
    (model_with_onset "onset : E --> I via hyper_erlang(stages = 3, rate = sigma)")

(* ── Worked SEIR via form simulates and is sane (compiles end-to-end) ────── *)

let test_worked_seir_compiles_clean () =
  let m = compile_ok seir_via_src in
  (* Sanity: 6 compartments, 5 transitions, no via survives. *)
  Alcotest.(check int) "compartment count" 6 (List.length m.Ir.compartments);
  Alcotest.(check int) "transition count" 5 (List.length m.Ir.transitions)

let () =
  Alcotest.run "via_lowering"
    [ ( "t1-anchor",
        [ Alcotest.test_case "via seir_erlang ≡ golden (modulo names)" `Quick
            test_t1_anchor_via_equals_golden ] );
      ( "stage-rate",
        [ Alcotest.test_case "rate = sigma ⇒ 3*sigma per stage" `Quick
            test_stage_rate_from_rate;
          Alcotest.test_case "mean = tau ⇒ 3/tau per stage" `Quick
            test_stage_rate_from_mean ] );
      ( "redirect",
        [ Alcotest.test_case "inflow + init land in stage 1" `Quick
            test_inflow_and_init_land_in_stage1;
          Alcotest.test_case "bare E sums in the FOI" `Quick
            test_bare_E_sums_in_foi;
          Alcotest.test_case "two staged residences chain into each other" `Quick
            test_two_staged_residences ] );
      ( "validation",
        [ Alcotest.test_case "non-positive-integer stages → E244" `Quick
            test_err_non_positive_integer_stages;
          Alcotest.test_case "both mean and rate → E245" `Quick
            test_err_both_mean_and_rate;
          Alcotest.test_case "neither mean nor rate → E245" `Quick
            test_err_neither_mean_nor_rate;
          Alcotest.test_case "single-exit violation → E246" `Quick
            test_err_single_exit_violation;
          Alcotest.test_case "hyper_erlang deferred → E243" `Quick
            test_err_hyper_erlang_deferred ] );
      ( "end-to-end",
        [ Alcotest.test_case "worked SEIR via form compiles clean" `Quick
            test_worked_seir_compiles_clean ] ) ]
