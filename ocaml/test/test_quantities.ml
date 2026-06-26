(* Generated-quantities frontend tests (proposal 2026-06-25): DSL `quantities {}`
   → IR `model.quantities`. Exercises the lexer/parser/AST/expander path and the
   quantity classifier (temporal reductions, series, reduction arithmetic, and
   the E288/E289/E290 diagnostics). *)

(* These models exercise expansion + classification, not dimensional analysis;
   some bodies (e.g. I/N where N is a bare compartment) are not the point. *)
let () = Compiler.no_dim_check := true

(** Substring check without the Str library. *)
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

let compile_ok src =
  match Compiler.compile ~name:"q" src with
  | Ok m -> m
  | Error e -> Alcotest.failf "compile failed: %s" e

(* Compile with structured diagnostics so the Error payload carries codes. *)
let compile_expect_error_code ~code ~contains src =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"q_err" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

(* ── Model scaffolds ─────────────────────────────────────────────────────── *)

(* Unstratified SIR-ish model; `body` is spliced into the quantities block.
   Compartments include N and D so `I / N` and `final(D)` resolve to Pops. *)
let model_with body =
  Printf.sprintf {|
    time_unit = 'days
    compartments { S, E, I, R, D, N }
    parameters {
      beta     : rate
      gamma    : rate
      i_thresh : count
    }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = 990  I = 10  N = 1000 }
    simulate { from = 0 'days  to = 100 'days }
    quantities {
%s
    }
  |} body

let stratified_src = {|
    time_unit = 'days
    dimensions { patch = [p0, p1] }
    compartments { S, I }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate }
    let N[l in patch] = S[l] + I[l]
    transitions {
      infection[l in patch] : S[l] --> I[l] @ beta * S[l] * I[l] / N[l]
      recovery[l in patch]  : I[l] --> S[l] @ gamma * I[l]
    }
    init { S_p0 = 990  I_p0 = 10 }
    simulate { from = 0 'days  to = 100 'days }
    quantities {
      peak_time[p in patch] = time_of_max(I[p])
    }
  |}

(* ── Accessors ───────────────────────────────────────────────────────────── *)

let find_q (m : Ir.model) name =
  match List.filter (fun (q : Ir.quantity) -> q.q_name = name) m.Ir.quantities with
  | [q] -> q
  | []  -> Alcotest.failf "no quantity named %s" name
  | _   -> Alcotest.failf "multiple leaves named %s (expected one)" name

(* ── Series / reduction tests ────────────────────────────────────────────── *)

let test_series () =
  let m = compile_ok (model_with "      prevalence = I / N") in
  match (find_q m "prevalence").q_body with
  | Ir.QBReduced { source = Ir.QSState (Ir.BinOp { op = Ir.Div; left; right }); reduce = None } ->
    (match left, right with
     | Ir.Pop "I", Ir.Pop "N" -> ()
     | _ -> Alcotest.failf "prevalence: expected Div(Pop I, Pop N)")
  | _ -> Alcotest.failf "prevalence: expected QBReduced{Div, reduce=None}"

let reduce_of m name =
  match (find_q m name).q_body with
  | Ir.QBReduced { reduce = Some r; _ } -> r
  | Ir.QBReduced { reduce = None; _ } ->
    Alcotest.failf "%s: expected a scalar reduction, got a series" name
  | Ir.QBDerived _ -> Alcotest.failf "%s: expected QBReduced, got QBDerived" name

let test_peak_value () =
  let m = compile_ok (model_with "      peak = max(I / N)") in
  match reduce_of m "peak" with
  | Ir.RValue Ir.VMax -> ()
  | _ -> Alcotest.failf "peak: expected RValue VMax"

let test_time_of_max () =
  let m = compile_ok (model_with "      time_to_peak = time_of_max(I)") in
  match reduce_of m "time_to_peak" with
  | Ir.RTime Ir.TimeOfMax -> ()
  | _ -> Alcotest.failf "time_to_peak: expected RTime TimeOfMax"

let test_first_above () =
  let m = compile_ok (model_with "      takeoff = first_above(I, i_thresh)") in
  (match reduce_of m "takeoff" with
   | Ir.RTime (Ir.FirstAbove (Ir.Param "i_thresh")) -> ()
   | Ir.RTime (Ir.FirstAbove _) ->
     Alcotest.failf "takeoff: FirstAbove threshold not Param i_thresh"
   | _ -> Alcotest.failf "takeoff: expected RTime FirstAbove");
  (* the folded series is the State expr `I` *)
  match (find_q m "takeoff").q_body with
  | Ir.QBReduced { source = Ir.QSState (Ir.Pop "I"); _ } -> ()
  | _ -> Alcotest.failf "takeoff: expected source State(Pop I)"

let test_integral () =
  let m = compile_ok (model_with "      pd = integral(I)") in
  match reduce_of m "pd" with
  | Ir.RIntegral -> ()
  | _ -> Alcotest.failf "pd: expected RIntegral"

let test_final () =
  let m = compile_ok (model_with "      total_deaths = final(D)") in
  (match reduce_of m "total_deaths" with
   | Ir.RValue Ir.VFinal -> ()
   | _ -> Alcotest.failf "total_deaths: expected RValue VFinal");
  match (find_q m "total_deaths").q_body with
  | Ir.QBReduced { source = Ir.QSState (Ir.Pop "D"); _ } -> ()
  | _ -> Alcotest.failf "total_deaths: expected source State(Pop D)"

(* binary max(a, b) is a pointwise operator → a *series* State quantity, NOT a
   reduction. Confirms the arity split. *)
let test_binary_max_is_series () =
  let m = compile_ok (model_with "      capped = max(I, R)") in
  match (find_q m "capped").q_body with
  | Ir.QBReduced { source = Ir.QSState (Ir.BinOp { op = Ir.Max; _ }); reduce = None } -> ()
  | _ -> Alcotest.failf "capped: expected QBReduced{BinOp Max, reduce=None}"

(* ── Stratified ──────────────────────────────────────────────────────────── *)

let test_stratified () =
  let m = compile_ok stratified_src in
  let leaves =
    List.filter (fun (q : Ir.quantity) -> q.q_name = "peak_time") m.Ir.quantities in
  Alcotest.(check int) "two peak_time leaves" 2 (List.length leaves);
  List.iter (fun (q : Ir.quantity) ->
    (match q.q_stratum with
     | [("patch", lvl)] when lvl = "p0" || lvl = "p1" -> ()
     | _ -> Alcotest.failf "peak_time: unexpected stratum");
    match q.q_body with
    | Ir.QBReduced { source = Ir.QSState (Ir.Pop p); reduce = Some (Ir.RTime Ir.TimeOfMax) } ->
      let lvl = snd (List.hd q.q_stratum) in
      if p <> "I_" ^ lvl then
        Alcotest.failf "peak_time[%s]: source Pop %s, expected I_%s" lvl p lvl
    | _ -> Alcotest.failf "peak_time: expected QBReduced{Pop, RTime TimeOfMax}"
  ) leaves

(* ── Reduction arithmetic (Derived) ──────────────────────────────────────── *)

let test_derived () =
  let body =
    "      takeoff = first_above(I, i_thresh)\n\
    \      fadeout = last_above(I, 0)\n\
    \      dur     = fadeout - takeoff" in
  let m = compile_ok (model_with body) in
  match (find_q m "dur").q_body with
  | Ir.QBDerived (Ir.SBinOp { op = Ir.Sub; left = Ir.SQRef l; right = Ir.SQRef r }) ->
    Alcotest.(check string) "dur left QRef" "fadeout" l.qref_name;
    Alcotest.(check string) "dur right QRef" "takeoff" r.qref_name
  | _ -> Alcotest.failf "dur: expected QBDerived(SBinOp Sub (SQRef fadeout)(SQRef takeoff))"

(* ── Diagnostics ─────────────────────────────────────────────────────────── *)

(* A temporal reduction in a transition rate → E290. *)
let test_e290_reduction_in_rate () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters { gamma : rate }
    transitions {
      recovery : I --> R @ gamma * time_of_max(I)
    }
    init { S = 990  I = 10 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  compile_expect_error_code ~code:"E290" ~contains:"time_of_max" src

(* A directly-typed `dt` in a quantity State body → located E288. *)
let test_e288_dt_in_quantity () =
  compile_expect_error_code ~code:"E288" ~contains:"dt"
    (model_with "      bad = I + dt")

let test_e289_total () =
  compile_expect_error_code ~code:"E289" ~contains:"total"
    (model_with "      foo = total(I)")

let test_e289_forward_qref () =
  (* `a` references `b`, declared later → forward QRef. *)
  let body =
    "      a = b - 1.0\n\
    \      b = final(I)" in
  compile_expect_error_code ~code:"E289" ~contains:"before it is declared"
    (model_with body)

let test_e289_qref_to_series () =
  (* `bad` references `prev`, a series quantity → cannot combine in arithmetic. *)
  let body =
    "      prev = I / N\n\
    \      bad  = prev - 1.0" in
  compile_expect_error_code ~code:"E289" ~contains:"series"
    (model_with body)

let () =
  Alcotest.run "quantities" [
    "series_and_reductions", [
      Alcotest.test_case "prevalence = I/N → series State" `Quick test_series;
      Alcotest.test_case "max(I/N) → RValue VMax" `Quick test_peak_value;
      Alcotest.test_case "time_of_max(I) → RTime TimeOfMax" `Quick test_time_of_max;
      Alcotest.test_case "first_above(I, i_thresh) → RTime FirstAbove" `Quick test_first_above;
      Alcotest.test_case "integral(I) → RIntegral" `Quick test_integral;
      Alcotest.test_case "final(D) → RValue VFinal" `Quick test_final;
      Alcotest.test_case "binary max(I,R) stays pointwise series" `Quick test_binary_max_is_series;
    ];
    "stratified", [
      Alcotest.test_case "peak_time[p in patch] → one leaf per patch" `Quick test_stratified;
    ];
    "derived", [
      Alcotest.test_case "dur = fadeout - takeoff → QBDerived SBinOp Sub" `Quick test_derived;
    ];
    "diagnostics", [
      Alcotest.test_case "E290 reduction in a transition rate" `Quick test_e290_reduction_in_rate;
      Alcotest.test_case "E288 dt in a quantity body" `Quick test_e288_dt_in_quantity;
      Alcotest.test_case "E289 total(x) deferred" `Quick test_e289_total;
      Alcotest.test_case "E289 forward QRef" `Quick test_e289_forward_qref;
      Alcotest.test_case "E289 QRef to a series quantity" `Quick test_e289_qref_to_series;
    ];
  ]
