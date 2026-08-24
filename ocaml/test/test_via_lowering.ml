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

(* Every fixture here seeds with literals, so each spec is a constant
   expression; anything else is a lowering bug these tests want to see. *)
let init_const (s : Ir.init_spec) : float =
  match s with
  | Ir.Deterministic (Ir.Const v) -> v
  | Ir.Deterministic _ -> Alcotest.fail "expected a constant initial condition"
  | Ir.InitCount _ | Ir.InitReal _ ->
    Alcotest.fail "expected a constant initial condition, got a drawn one"

let init_consts (ic : Ir.initial_conditions) : (string * float) list =
  List.map (fun (k, s) -> (k, init_const s)) ic

let canon_init map (ic : Ir.initial_conditions) : (string * float) list =
  List.map (fun (k, s) -> (rename_name map k, init_const s)) ic |> List.sort compare

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

(* `stages = 1` is the ordinary exponential dwell (proposal §4: "the exponential
   SEIR … not a no-op"): NO sub-staging, NO stage dimension — it lowers to the
   SAME plain exponential `E --> I @ sigma * E` as writing `@ sigma * E` directly,
   so a `stages = 1,2,3,…` sweep family is uniform. (Erlang(1) = Exponential.) *)
let test_stages_one_is_plain_exponential () =
  let via   = compile_ok (model_with_onset
    "onset : E --> I via erlang(stages = 1, rate = sigma)") in
  let plain = compile_ok (model_with_onset "onset : E --> I @ sigma * E") in
  Alcotest.(check int) "stages=1: no stage compartments (4 base only)"
    (List.length plain.Ir.compartments) (List.length via.Ir.compartments);
  Alcotest.(check int) "stages=1: same transition count as plain exponential"
    (List.length plain.Ir.transitions) (List.length via.Ir.transitions);
  let onset_rate m =
    (List.find (fun (t : Ir.transition) -> t.Ir.name = "onset")
       m.Ir.transitions).Ir.rate in
  Alcotest.(check bool) "stages=1: onset rate identical to `@ sigma * E`" true
    (onset_rate via = onset_rate plain)

(* ── Inflow + init land in stage 1; bare E sums in the FOI ───────────────── *)

let test_inflow_and_init_land_in_stage1 () =
  let m = compile_ok seir_via_src in
  (* infection lands in E_s1, not a bare E. *)
  let inf = List.find (fun (t : Ir.transition) -> t.Ir.name = "infection") m.Ir.transitions in
  Alcotest.(check bool) "infection --> E_s1" true
    (List.mem ("E_s1", 1) inf.Ir.stoichiometry);
  (* init: E_s1 = 5, no bare E key. *)
  let kvs = init_consts m.Ir.initial_conditions in
  Alcotest.(check bool) "init has E_s1 = 5" true (List.mem ("E_s1", 5.0) kvs);
  Alcotest.(check bool) "init has no bare E" false (List.mem_assoc "E" kvs)

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

let test_err_unknown_erlang_keyword () =
  compile_expect_error_code ~code:"E247" ~contains:"onset"
    (model_with_onset
       "onset : E --> I via erlang(stages = 3, rate = sigma, banana = 1)")

let test_err_via_multiple_sources () =
  (* A staged residence stages exactly one compartment. *)
  compile_expect_error_code ~code:"E249" ~contains:"onset"
    (model_with_onset
       "onset : E + S --> I via erlang(stages = 3, rate = sigma)")

let test_err_unsupported_law_deferred () =
  (* `erlang` and `hyper_erlang` now lower; a still-deferred law (`coxian`)
     keeps the repurposed E243 "not yet supported" placeholder. *)
  compile_expect_error_code ~code:"E243" ~contains:"not yet supported"
    (model_with_onset "onset : E --> I via coxian(stages = 3, rate = sigma)")

(* ── Worked SEIR via form simulates and is sane (compiles end-to-end) ────── *)

let test_worked_seir_compiles_clean () =
  let m = compile_ok seir_via_src in
  (* Sanity: 6 compartments, 5 transitions, no via survives. *)
  Alcotest.(check int) "compartment count" 6 (List.length m.Ir.compartments);
  Alcotest.(check int) "transition count" 5 (List.length m.Ir.transitions)

(* ── T6: staging an ALSO-stratified compartment (age × Erlang stages) ─────── *)

(* A two-age SEIR with an Erlang-3 INFECTIOUS period, written two ways. The
   manual form stages `I` by hand into `[age, inf_stage]`, summing `I` over its
   stages EXPLICITLY in the FOI and the per-age total `N_local`; the via form
   writes `I[a]` and `via erlang(stages = 3, rate = gamma)` and relies on the
   staging pass to (a) compose the stage dimension onto `I`'s age stratification,
   (b) thread the age index through the synthesized per-age chain + exit, and
   (c) rewrite every partial `I[a]` reference into the explicit stage-sum. The
   anchor (test below) asserts the two lower to the SAME IR modulo stage names. *)

let age_staged_manual_src =
  "time_unit = 'days\n\
   compartments { S, E, I, R }\n\
   dimensions { age = [child, adult]  inf_stage = [g1, g2, g3] }\n\
   stratify(by = age)\n\
   stratify(by = inf_stage, only = [I])\n\
   let N_local[a in age] = S[a] + E[a] + sum(s in inf_stage, I[a, s]) + R[a]\n\
   parameters {\n\
  \  beta  : rate in [0.001, 0.5]\n\
  \  gamma : rate in [0.01, 1.0]\n\
   }\n\
   tables { C_age : age \xc3\x97 age = [[12.0, 4.0], [4.0, 8.0]] }\n\
   transitions {\n\
  \  infection[a in age] : S[a] --> I[a, g1]\n\
  \    @ beta * S[a] * sum(b in age, C_age[a, b] * sum(s in inf_stage, I[b, s]) / N_local[b])\n\
  \  recovery_stage[a in age, (s, s_next) in consecutive(inf_stage)]\n\
  \    : I[a, s] --> I[a, s_next] @ 3 * gamma * I[a, s]\n\
  \  recovery[a in age] : I[a, g3] --> R[a] @ 3 * gamma * I[a, g3]\n\
   }\n\
   init { S[child] = 4990  S[adult] = 5000  I[child, g1] = 10 }\n\
   simulate { from = 0 'days  to = 100 'days }\n"

let age_staged_via_src =
  "time_unit = 'days\n\
   compartments { S, E, I, R }\n\
   dimensions { age = [child, adult] }\n\
   stratify(by = age)\n\
   let N_local[a in age] = S[a] + E[a] + I[a] + R[a]\n\
   parameters {\n\
  \  beta  : rate in [0.001, 0.5]\n\
  \  gamma : rate in [0.01, 1.0]\n\
   }\n\
   tables { C_age : age \xc3\x97 age = [[12.0, 4.0], [4.0, 8.0]] }\n\
   transitions {\n\
  \  infection[a in age] : S[a] --> I[a]\n\
  \    @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])\n\
  \  recovery[a in age] : I[a] --> R[a] via erlang(stages = 3, rate = gamma)\n\
   }\n\
   init { S[child] = 4990  S[adult] = 5000  I[child] = 10 }\n\
   simulate { from = 0 'days  to = 100 'days }\n"

(* The load-bearing T6 anchor: the via form's IR equals the manual form's IR,
   modulo stage names. Identical machinery to the T1 anchor (it reuses
   [stage_rename_map] — which renames the > 1-cell staged base `I` to a uniform
   `I_<i>` in compartment-list order — and the same canonical-transition compare),
   so an isomorphism failure surfaces as a stoichiometry or rate-AST mismatch.
   Disabling the partial-reference rewrite (`sum_staged_refs`) makes the via FOI
   reference a non-existent `I_child` cell, which fails to compile (E100) — so
   this test goes red→green on the rewrite. *)
let test_t6_anchor_age_staged_via_equals_manual () =
  let via_m = compile_ok age_staged_via_src in
  let man_m = compile_ok age_staged_manual_src in
  let map_via = stage_rename_map via_m in
  let map_man = stage_rename_map man_m in
  Alcotest.(check (list string)) "compartments (canonical)"
    (canon_comps map_man man_m) (canon_comps map_via via_m);
  Alcotest.(check (list (pair string (float 1e-9)))) "init (canonical)"
    (canon_init map_man man_m.Ir.initial_conditions)
    (canon_init map_via via_m.Ir.initial_conditions);
  let via_canon =
    List.map (canon_transition map_via) via_m.Ir.transitions |> List.sort compare in
  let man_canon =
    List.map (canon_transition map_man) man_m.Ir.transitions |> List.sort compare in
  if List.length via_canon <> List.length man_canon then
    Alcotest.failf "transition count: manual %d, via %d"
      (List.length man_canon) (List.length via_canon);
  List.iter2 (fun (m_stoich, m_rate) (v_stoich, v_rate) ->
    if m_stoich <> v_stoich then
      Alcotest.failf "stoichiometry mismatch:\n manual %s\n via    %s"
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) m_stoich))
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) v_stoich));
    if m_rate <> v_rate then
      Alcotest.failf "rate AST mismatch for stoich %s:\n manual %s\n via    %s"
        (String.concat "," (List.map (fun (n,d) -> Printf.sprintf "%s:%d" n d) m_stoich))
        (Yojson.Safe.to_string (Serde.expr_to_json m_rate))
        (Yojson.Safe.to_string (Serde.expr_to_json v_rate))
  ) man_canon via_canon

(* The per-age FOI references the stage-SUM of each age's I, not a dangling
   partial `I_child` / `I_adult`. We confirm at the IR level: the infection
   transition for each age contains a PopSum over exactly that age's three stage
   cells, and no Pop names a bare per-age `I_<age>` (which would not exist as a
   compartment after staging). *)
let test_foi_rewrite_sums_stages_per_age () =
  let m = compile_ok age_staged_via_src in
  let comp_names = List.map (fun (c : Ir.compartment) -> c.Ir.name) m.Ir.compartments in
  (* The staged cells exist; the partial per-age names do NOT. *)
  List.iter (fun n -> Alcotest.(check bool) (n ^ " exists") true (List.mem n comp_names))
    [ "I_child_s1"; "I_child_s2"; "I_child_s3"; "I_adult_s1"; "I_adult_s2"; "I_adult_s3" ];
  Alcotest.(check bool) "no bare per-age I_child compartment" false
    (List.mem "I_child" comp_names);
  (* Collect every PopSum / Pop name appearing anywhere in a rate. *)
  let rec pop_names acc = function
    | Ir.Pop n     -> n :: acc
    | Ir.PopSum ns -> ns @ acc
    | Ir.BinOp b   -> pop_names (pop_names acc b.Ir.left) b.Ir.right
    | Ir.UnOp u    -> pop_names acc u.Ir.arg
    | Ir.Cond c    -> pop_names (pop_names (pop_names acc c.Ir.pred) c.Ir.then_) c.Ir.else_
    | Ir.Reduce ts -> List.fold_left pop_names acc ts
    | _            -> acc
  in
  let rec popsums acc = function
    | Ir.PopSum ns -> ns :: acc
    | Ir.BinOp b   -> popsums (popsums acc b.Ir.left) b.Ir.right
    | Ir.UnOp u    -> popsums acc u.Ir.arg
    | Ir.Cond c    -> popsums (popsums (popsums acc c.Ir.pred) c.Ir.then_) c.Ir.else_
    | Ir.Reduce ts -> List.fold_left popsums acc ts
    | _            -> acc
  in
  let infections =
    List.filter (fun (t : Ir.transition) ->
      String.length t.Ir.name >= 9 && String.sub t.Ir.name 0 9 = "infection")
      m.Ir.transitions
  in
  Alcotest.(check int) "two age-specific infection transitions" 2
    (List.length infections);
  List.iter (fun (t : Ir.transition) ->
    (* No bare per-age I_<age> Pop survives in the FOI. *)
    let names = pop_names [] t.Ir.rate in
    List.iter (fun bad ->
      Alcotest.(check bool) (Printf.sprintf "%s: no dangling %s" t.Ir.name bad) false
        (List.mem bad names)) [ "I_child"; "I_adult" ];
    (* The FOI must sum each age's three I-stages (the b-sum produces per-age
       PopSums over [I_<age>_s1; _s2; _s3]). *)
    let sums = popsums [] t.Ir.rate in
    let sums_age age =
      List.exists (fun ns ->
        List.mem (Printf.sprintf "I_%s_s1" age) ns
        && List.mem (Printf.sprintf "I_%s_s2" age) ns
        && List.mem (Printf.sprintf "I_%s_s3" age) ns) sums
    in
    Alcotest.(check bool) (t.Ir.name ^ ": FOI sums child stages") true (sums_age "child");
    Alcotest.(check bool) (t.Ir.name ^ ": FOI sums adult stages") true (sums_age "adult")
  ) infections

(* Inflow + init for a staged-and-stratified compartment land in (age, stage-1).
   The age-specific infection edge `S[a] --> I[a]` redirects to `I[a, s1]`, and
   `init I[child] = 10` redirects to `I_child_s1`. *)
let test_age_inflow_and_init_land_in_stage1 () =
  let m = compile_ok age_staged_via_src in
  let inf_child =
    List.find (fun (t : Ir.transition) -> t.Ir.name = "infection_child") m.Ir.transitions in
  Alcotest.(check bool) "infection_child --> I_child_s1" true
    (List.mem ("I_child_s1", 1) inf_child.Ir.stoichiometry);
  let kvs = init_consts m.Ir.initial_conditions in
  Alcotest.(check bool) "init has I_child_s1 = 10" true (List.mem ("I_child_s1", 10.0) kvs);
  Alcotest.(check bool) "init has no bare I_child" false (List.mem_assoc "I_child" kvs)

(* ── C4a: the rewrite reaches every expr-bearing container ───────────────── *)

(* Proposal 2026-07-31-aggregation-semantics C4a. The partial-reference rewrite
   used to be applied to five containers by a block written out twice — once for
   `erlang`, once for `hyper_erlang` — and the two copies drifted: gh#463 added
   the action containers to `hyper_erlang` only, and NEITHER ever covered
   `quantities {}`. So `I[child]` on an age × staged compartment resolved to a
   stage-sum in `observations` and raised E287 in `quantities`, from identical
   syntax. Both sites now go through [apply_via_rewrite].

   Asserted at the IR level rather than "it compiles": the quantity's body must
   be the same three-cell stage sum the FOI gets, so removing the rewrite from
   the quantity walk fails here rather than passing vacuously. *)

let staged_container_src body =
  "time_unit = 'days\n\
   compartments { S, E, I, R }\n\
   dimensions { age = [child, adult] }\n\
   stratify(by = age)\n\
   let N_local[a in age] = S[a] + E[a] + I[a] + R[a]\n\
   parameters {\n\
  \  beta  : rate in [0.001, 0.5]\n\
  \  gamma : rate in [0.01, 1.0]\n\
   }\n\
   transitions {\n\
  \  infection[a in age] : S[a] --> I[a] @ beta * S[a] * I[a] / N_local[a]\n\
  \  recovery[a in age] : I[a] --> R[a] via erlang(stages = 3, rate = gamma)\n\
   }\n\
   init { S[child] = 4990  S[adult] = 5000  I[child] = 10 }\n"
  ^ body
  ^ "\nsimulate { from = 0 'days  to = 100 'days }\n"

let stage_cells_of_child = [ "I_child_s1"; "I_child_s2"; "I_child_s3" ]

(* A partial reference `I[child]` on an [age, stage] compartment must lower to
   the sum over that age's stage cells — never to a bare `I_child`, which has no
   cell. *)
let check_is_child_stage_sum ~what (e : Ir.expr) =
  let rec find_popsum = function
    | Ir.PopSum ns -> Some ns
    | Ir.BinOp b   -> (match find_popsum b.Ir.left with
                       | Some x -> Some x | None -> find_popsum b.Ir.right)
    | Ir.UnOp u    -> find_popsum u.Ir.arg
    | Ir.Reduce ts -> List.fold_left
                        (fun acc t -> match acc with Some _ -> acc | None -> find_popsum t)
                        None ts
    | _            -> None
  in
  match find_popsum e with
  | Some ns ->
    Alcotest.(check (list string)) (what ^ ": sums this age's stage cells")
      stage_cells_of_child (List.sort compare ns)
  | None ->
    Alcotest.failf "%s: expected a PopSum over %s, found none — the via rewrite \
                    did not reach this container"
      what (String.concat ", " stage_cells_of_child)

let test_c4a_quantity_sees_the_stage_rewrite () =
  let m = compile_ok (staged_container_src "quantities { prev_child = I[child] }") in
  match List.filter (fun (q : Ir.quantity) -> q.Ir.q_name = "prev_child") m.Ir.quantities with
  | [ { Ir.q_body = Ir.QBReduced { source = Ir.QSState e; reduce = None }; _ } ] ->
    check_is_child_stage_sum ~what:"quantities" e
  | [ _ ] -> Alcotest.fail "prev_child: expected an unreduced state series"
  | []    -> Alcotest.fail "no quantity named prev_child"
  | _     -> Alcotest.fail "multiple prev_child leaves"

(* The same expression, in an intervention's VALUE operand. Endpoints
   (`from =` / `to =`) are deliberately NOT rewritten — a staged source has no
   single cell to name, and that is gh#460. *)
let test_c4a_intervention_value_operand_sees_the_rewrite () =
  let m = compile_ok (staged_container_src
    "interventions { cull : transfer(count = 0.5 * I[child], from = S[child], to = R[child]) \
     at [30] }") in
  let counts =
    List.concat_map (fun (iv : Ir.intervention) ->
      List.filter_map (function
        | Ir.AbsoluteTransfer t -> Some t.Ir.count
        | _ -> None) iv.Ir.actions) m.Ir.interventions
  in
  match counts with
  | [ e ] -> check_is_child_stage_sum ~what:"interventions" e
  | []    -> Alcotest.fail "no AbsoluteTransfer action found"
  | _     -> Alcotest.fail "expected exactly one AbsoluteTransfer action"

let () =
  Alcotest.run "via_lowering"
    [ ( "t1-anchor",
        [ Alcotest.test_case "via seir_erlang ≡ golden (modulo names)" `Quick
            test_t1_anchor_via_equals_golden ] );
      ( "stage-rate",
        [ Alcotest.test_case "rate = sigma ⇒ 3*sigma per stage" `Quick
            test_stage_rate_from_rate;
          Alcotest.test_case "mean = tau ⇒ 3/tau per stage" `Quick
            test_stage_rate_from_mean;
          Alcotest.test_case "stages = 1 ⇒ plain exponential (no staging)" `Quick
            test_stages_one_is_plain_exponential ] );
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
          Alcotest.test_case "unknown erlang keyword → E247" `Quick
            test_err_unknown_erlang_keyword;
          Alcotest.test_case "via with >1 source → E249" `Quick
            test_err_via_multiple_sources;
          Alcotest.test_case "unsupported law (coxian) deferred → E243" `Quick
            test_err_unsupported_law_deferred ] );
      ( "end-to-end",
        [ Alcotest.test_case "worked SEIR via form compiles clean" `Quick
            test_worked_seir_compiles_clean ] );
      ( "t6-stratified",
        [ Alcotest.test_case "age × staged via ≡ manual (modulo stage names)" `Quick
            test_t6_anchor_age_staged_via_equals_manual;
          Alcotest.test_case "FOI rewrite sums each age's stages (no dangling I[a])" `Quick
            test_foi_rewrite_sums_stages_per_age;
          Alcotest.test_case "age inflow + init land in (age, stage 1)" `Quick
            test_age_inflow_and_init_land_in_stage1 ] );
      ( "c4a-containers",
        [ Alcotest.test_case "quantities {} sees the stage rewrite" `Quick
            test_c4a_quantity_sees_the_stage_rewrite;
          Alcotest.test_case "intervention value operand sees the stage rewrite" `Quick
            test_c4a_intervention_value_operand_sees_the_rewrite ] ) ]
