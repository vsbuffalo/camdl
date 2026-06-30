(* Counterfactual-contrasts frontend tests (proposal 2026-06-25): the
   `contrasts {}` surface and its diagnostics. Drives the real compile pipeline
   (Compiler.compile) over inline models and asserts each surviving contrast
   diagnostic — E292 (run-member in a rate), E293 (run-member in a quantities
   recipe, with its ns-branched hint), E294 (undeclared run / member), E295
   (malformed body, located), E297 (operand dimension mismatch), and E298
   (duplicate contrast name, located). Mirrors test_quantities.ml. *)

(* ── String helpers (no Str dependency) ──────────────────────────────────── *)

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

(* ── Compile harness ─────────────────────────────────────────────────────── *)

(* Compile [src] under structured (JSON) diagnostics and return the JSON error
   payload string, asserting the compile FAILED. [dim_check] defaults to false
   (the dimensional pass off, matching the quantities tests) so the
   expander-level contrast diagnostics fire without dimensional noise; the E297
   dimensional test turns it on. *)
let compile_err ?(dim_check = false) src : string =
  let prev = !Compiler.no_dim_check in
  Compiler.no_dim_check := not dim_check;
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"contrast_err" src in
  Diagnostics.json_errors_mode := false;
  Compiler.no_dim_check := prev;
  match result with
  | Ok _ -> Alcotest.fail "expected a compile error, but compile succeeded"
  | Error e -> e

(* Find the (first) diagnostic object carrying [code] in the JSON payload. *)
let find_diag ~code (json : string) : Yojson.Safe.t =
  let open Yojson.Safe.Util in
  let ds = match Yojson.Safe.from_string json with
    | `List ds -> ds
    | _ -> Alcotest.failf "diagnostics payload is not a JSON array: %s" json
  in
  match List.find_opt (fun d -> member "code" d |> to_string = code) ds with
  | Some d -> d
  | None -> Alcotest.failf "no diagnostic with code %s in: %s" code json

(* Assert [src] fails with [code]; return that diagnostic for further checks. *)
let expect_code ?(dim_check = false) ~code src : Yojson.Safe.t =
  find_diag ~code (compile_err ~dim_check src)

let diag_line d = Yojson.Safe.Util.(member "loc" d |> member "line" |> to_int)
let diag_hint d = Yojson.Safe.Util.(member "hint" d |> to_string)
let diag_message d = Yojson.Safe.Util.(member "message" d |> to_string)

(* ── Model scaffolds ─────────────────────────────────────────────────────── *)

(* SIR-ish model; the `quantities` and `contrasts` blocks are spliced in. The
   scaffold is dimensionally clean, so with dim-check on the only dimensional
   error is the one a test deliberately injects. *)
let model_contrasts ?(quantities = "      total = final(D)") contrasts_body =
  Printf.sprintf {|
    time_unit = 'days
    compartments { S, I, R, D, N }
    parameters { beta : rate  gamma : rate }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    quantities {
%s
    }
    contrasts {
%s
    }
    init { S = 990  I = 10  N = 1000 }
    simulate { from = 0 'days  to = 100 'days }
  |} quantities contrasts_body

(* A run-rooted reference in a transition rate — valid only in `contrasts {}`. *)
let e292_src = {|
    time_unit = 'days
    compartments { S, I, R, D, N }
    parameters { beta : rate  gamma : rate }
    quantities { total = final(D) }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * fitted.quantities.total
    }
    init { S = 990  I = 10  N = 1000 }
    simulate { from = 0 'days  to = 100 'days }
  |}

(* A run-rooted reference inside a `quantities {}` recipe (where the run is
   implicit) — splice the offending operand. *)
let e293_src operand = Printf.sprintf {|
    time_unit = 'days
    compartments { S, I, R, D, N }
    parameters { beta : rate  gamma : rate }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    quantities {
      bad = %s
    }
    init { S = 990  I = 10  N = 1000 }
    simulate { from = 0 'days  to = 100 'days }
  |} operand

(* ── E292: run-member used in a rate ──────────────────────────────────────── *)

let test_e292_run_member_in_rate () =
  let d = expect_code ~code:"E292" e292_src in
  Alcotest.(check bool) "E292 names the offending operand" true
    (contains_substring ~needle:"fitted.quantities.total" (diag_message d))

(* ── E293: run-member in a quantities recipe (+ the ns-branched hint) ─────── *)

(* For a quantities-namespace operand the corrected in-recipe form is the BARE
   member name (`quantities.foo` does not parse). *)
let test_e293_quantities_hint () =
  let d = expect_code ~code:"E293" (e293_src "run1.quantities.total") in
  let h = diag_hint d in
  Alcotest.(check bool) "E293 hint suggests the bare member" true
    (contains_substring ~needle:"write `total`" h);
  Alcotest.(check bool) "E293 hint does NOT suggest the non-parsing `quantities.total`" false
    (contains_substring ~needle:"`quantities.total`" h)

(* For an observations-namespace operand the corrected form keeps the
   `observations.` prefix (`observations.afp` parses inside a recipe). *)
let test_e293_observations_hint () =
  let d = expect_code ~code:"E293" (e293_src "run1.observations.afp") in
  Alcotest.(check bool) "E293 hint suggests `observations.afp`" true
    (contains_substring ~needle:"write `observations.afp`" (diag_hint d))

(* ── E294: undeclared run / member ────────────────────────────────────────── *)

let test_e294_undeclared_run () =
  let d = expect_code ~code:"E294"
    (model_contrasts "      averted = madeup.quantities.total - fitted.quantities.total") in
  Alcotest.(check bool) "E294 names the undeclared run" true
    (contains_substring ~needle:"no run named 'madeup'" (diag_message d))

let test_e294_undeclared_member () =
  let d = expect_code ~code:"E294"
    (model_contrasts "      averted = fitted.quantities.nope - fitted.quantities.total") in
  Alcotest.(check bool) "E294 names the undeclared member" true
    (contains_substring ~needle:"no quantity named 'nope'" (diag_message d))

(* ── E295: malformed body (a bare const), and it must be LOCATED ──────────── *)

let test_e295_malformed_body_located () =
  let d = expect_code ~code:"E295" (model_contrasts "      averted = 5.0") in
  Alcotest.(check bool) "E295 is located (line > 0)" true (diag_line d > 0)

(* ── E297: operand dimension mismatch (a count minus a time) ──────────────── *)

let test_e297_dim_mismatch () =
  let m = model_contrasts
    ~quantities:"      total = final(D)\n      peak_t = time_of_max(I)"
    "      bad = fitted.quantities.total - fitted.quantities.peak_t" in
  ignore (expect_code ~dim_check:true ~code:"E297" m)

(* ── E298: duplicate contrast name, located ───────────────────────────────── *)

let test_e298_duplicate_name_located () =
  let body =
    "      averted = fitted.quantities.total - fitted.quantities.total\n\
    \      averted = fitted.quantities.total - fitted.quantities.total" in
  let d = expect_code ~code:"E298" (model_contrasts body) in
  Alcotest.(check bool) "E298 names the duplicated contrast" true
    (contains_substring ~needle:"averted" (diag_message d));
  Alcotest.(check bool) "E298 is located (line > 0)" true (diag_line d > 0)

let () =
  Alcotest.run "contrasts" [
    "diagnostics", [
      Alcotest.test_case "E292 run-member in a transition rate" `Quick
        test_e292_run_member_in_rate;
      Alcotest.test_case "E293 run-member in a recipe → bare-member hint" `Quick
        test_e293_quantities_hint;
      Alcotest.test_case "E293 run-member in a recipe → observations.<m> hint" `Quick
        test_e293_observations_hint;
      Alcotest.test_case "E294 undeclared run" `Quick test_e294_undeclared_run;
      Alcotest.test_case "E294 undeclared member" `Quick test_e294_undeclared_member;
      Alcotest.test_case "E295 malformed body is located" `Quick
        test_e295_malformed_body_located;
      Alcotest.test_case "E297 operand dimension mismatch" `Quick
        test_e297_dim_mismatch;
      Alcotest.test_case "E298 duplicate contrast name is located" `Quick
        test_e298_duplicate_name_located;
    ];
  ]
