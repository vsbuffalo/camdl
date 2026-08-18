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

(* SIR-ish model WITH an unstratified observation stream `afp`, so the v1.1
   `observations.afp` source resolves; `body` is spliced into the quantities
   block. *)
let model_obs_with body =
  Printf.sprintf {|
    time_unit = 'days
    compartments { S, E, I, R, D, N }
    parameters {
      beta  : rate
      gamma : rate
      rho   : probability
    }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    observations {
      afp {
        columns       { time : time, afp : count }
        projected     = incidence(infection)
        emit_schedule = every 1 'days
        afp           ~ poisson(rate = rho * projected)
      }
    }
    init { S = 990  I = 10  N = 1000 }
    simulate { from = 0 'days  to = 100 'days }
    quantities {
%s
    }
  |} body

(* A model with a STRATIFIED observation stream `afp[p in patch]` — v1.1 defers
   stratified observation sources, so reducing `observations.afp` is E289. *)
let stratified_obs_src body =
  Printf.sprintf {|
    time_unit = 'days
    dimensions { patch = [p0, p1] }
    compartments { S, I }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate  rho : probability }
    let N[l in patch] = S[l] + I[l]
    transitions {
      infection[l in patch] : S[l] --> I[l] @ beta * S[l] * I[l] / N[l]
      recovery[l in patch]  : I[l] --> S[l] @ gamma * I[l]
    }
    observations {
      afp[p in patch] {
        columns       { time : time, patch : dim, afp : count }
        projected     = incidence(infection[p])
        emit_schedule = every 1 'days
        afp           ~ poisson(rate = rho * projected)
      }
    }
    init { S_p0 = 990  I_p0 = 10 }
    simulate { from = 0 'days  to = 100 'days }
    quantities {
%s
    }
  |} body

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

(* value_at with a constant time → RValue (VValueAt (ATime _)) (proposal
   2026-08-17). *)
let test_value_at_const () =
  let m = compile_ok (model_with "      size_at_20 = value_at(I, 20)") in
  match reduce_of m "size_at_20" with
  | Ir.RValue (Ir.VValueAt (Ir.ATime (Ir.Const 20.0))) -> ()
  | _ -> Alcotest.failf "size_at_20: expected RValue (VValueAt (ATime (Const 20)))"

(* value_at with the bare `last_obs` anchor → symbolic ALastObs — the compiled
   model stays data-independent; resolution happens at evaluation time. *)
let test_value_at_last_obs () =
  let m = compile_ok (model_with "      outbreak_size = value_at(I, last_obs)") in
  match reduce_of m "outbreak_size" with
  | Ir.RValue (Ir.VValueAt Ir.ALastObs) -> ()
  | _ -> Alcotest.failf "outbreak_size: expected RValue (VValueAt ALastObs)"

(* Arithmetic over the anchor is rejected (v1): `last_obs` must be the whole
   second argument, so it cannot half-enter the expression language. *)
let test_value_at_anchor_arithmetic_rejected () =
  compile_expect_error_code ~code:"E289" ~contains:"whole second argument"
    (model_with "      bad = value_at(I, last_obs - 7)")

(* `last_obs` is intercepted ONLY inside value_at's second argument; anywhere
   else it stays an unknown identifier (cannot leak into dynamics). *)
let test_last_obs_outside_value_at_is_unknown () =
  compile_expect_error_code ~code:"E100" ~contains:"last_obs"
    (model_with "      bad = final(I + last_obs)")

let test_value_at_wrong_arity () =
  compile_expect_error_code ~code:"E289" ~contains:"two arguments"
    (model_with "      bad = value_at(I)")

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

(* ── v1.1 observation source: observations.<stream> ──────────────────────── *)

(* A bare `observations.afp` body (no reduction) is rejected in v1.1 → E289: an
   observation series has its own observation-time axis and must be reduced. *)
let test_obs_bare_series_rejected () =
  compile_expect_error_code ~code:"E289" ~contains:"must be reduced"
    (model_obs_with "      afp_series = observations.afp")

(* `max(observations.afp)` → a value reduction over y_sim. *)
let test_obs_max () =
  let m = compile_ok (model_obs_with "      peak_afp = max(observations.afp)") in
  match (find_q m "peak_afp").q_body with
  | Ir.QBReduced { source = Ir.QSObservation "afp"; reduce = Some (Ir.RValue Ir.VMax) } -> ()
  | _ -> Alcotest.failf "peak_afp: expected QBReduced{QSObservation afp, Some(RValue VMax)}"

(* `first_above(observations.afp, 0)` → a time reduction over y_sim; threshold
   resolves as an ordinary (state-allowed) expr. *)
let test_obs_first_above () =
  let m = compile_ok (model_obs_with "      first_afp = first_above(observations.afp, 0)") in
  match (find_q m "first_afp").q_body with
  | Ir.QBReduced { source = Ir.QSObservation "afp";
                   reduce = Some (Ir.RTime (Ir.FirstAbove (Ir.Const 0.0))) } -> ()
  | _ ->
    Alcotest.failf
      "first_afp: expected QBReduced{QSObservation afp, Some(RTime FirstAbove(Const 0))}"

(* ── v1.1 observation-source diagnostics ─────────────────────────────────── *)

(* An undeclared observation stream → E289. *)
let test_e289_obs_undeclared () =
  compile_expect_error_code ~code:"E289" ~contains:"no observation stream"
    (model_obs_with "      bad = max(observations.nope)")

(* A stratified observation source is deferred in v1.1 → E289. *)
let test_e289_obs_stratified () =
  compile_expect_error_code ~code:"E289" ~contains:"stratified observation"
    (stratified_obs_src "      peak_afp = max(observations.afp)")

(* An observation source mixed into arithmetic → E289. *)
let test_e289_obs_mixed () =
  compile_expect_error_code ~code:"E289" ~contains:"observation source"
    (model_obs_with "      bad = observations.afp + 1")

(* `observations.afp` used in a transition rate → E290 (only valid in a
   quantities block). *)
let test_e290_obs_in_rate () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters { gamma : rate  rho : probability }
    transitions {
      recovery : I --> R @ gamma * observations.afp
    }
    observations {
      afp {
        columns       { time : time, afp : count }
        projected     = incidence(recovery)
        emit_schedule = every 1 'days
        afp           ~ poisson(rate = rho * projected)
      }
    }
    init { S = 990  I = 10 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  compile_expect_error_code ~code:"E290" ~contains:"observations.afp" src

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

(* A model `let` mixed with a reduced scalar quantity is a shape mismatch (a
   `let` is a per-instant series value, the reduced quantity a whole-trajectory
   scalar). It must report that clearly — NOT the misleading "unknown name '…'
   in reduction arithmetic" — and the hint must correct the misconception that
   lets are out of scope inside quantities. *)
let let_mix_src = {|
  time_unit = 'days
  compartments { S, I, R }
  let M = S + I + R
  parameters { beta : rate  gamma : rate }
  transitions {
    infection : S --> I @ beta * S * I / M
    recovery  : I --> R @ gamma * I
  }
  quantities {
    peak = max(I)
    bad  = peak * M
  }
  init { S = 990  I = 10 }
  simulate { from = 0 to = 100 }
|}

let test_e289_let_mixed_with_scalar () =
  compile_expect_error_code ~code:"E289"
    ~contains:"cannot combine `let` binding 'M'" let_mix_src

let test_e289_let_mix_hint_corrects_scope () =
  compile_expect_error_code ~code:"E289"
    ~contains:"in scope in series quantities" let_mix_src

(* A quantity may not share a `let`'s name (names kept distinct); the hint must
   teach the distinct-name pattern (e.g. `R0_hat = R0`). *)
let test_e289_quantity_let_collision_hint () =
  compile_expect_error_code ~code:"E289" ~contains:"distinct name"
    {|
  time_unit = 'days
  compartments { S, I, R }
  let R0 = 2.0
  parameters { beta : rate  gamma : rate }
  transitions {
    infection : S --> I @ beta * S * I / (S + I + R)
    recovery  : I --> R @ gamma * I
  }
  quantities { R0 = max(I) }
  init { S = 990  I = 10 }
  simulate { from = 0 to = 100 }
|}

let () =
  Alcotest.run "quantities" [
    "series_and_reductions", [
      Alcotest.test_case "prevalence = I/N → series State" `Quick test_series;
      Alcotest.test_case "max(I/N) → RValue VMax" `Quick test_peak_value;
      Alcotest.test_case "time_of_max(I) → RTime TimeOfMax" `Quick test_time_of_max;
      Alcotest.test_case "first_above(I, i_thresh) → RTime FirstAbove" `Quick test_first_above;
      Alcotest.test_case "integral(I) → RIntegral" `Quick test_integral;
      Alcotest.test_case "final(D) → RValue VFinal" `Quick test_final;
      Alcotest.test_case "value_at(I, 20) → RValue VValueAt ATime" `Quick test_value_at_const;
      Alcotest.test_case "value_at(I, last_obs) → RValue VValueAt ALastObs" `Quick
        test_value_at_last_obs;
      Alcotest.test_case "E289 anchor arithmetic rejected" `Quick
        test_value_at_anchor_arithmetic_rejected;
      Alcotest.test_case "last_obs outside value_at stays unknown" `Quick
        test_last_obs_outside_value_at_is_unknown;
      Alcotest.test_case "E289 value_at arity" `Quick test_value_at_wrong_arity;
      Alcotest.test_case "binary max(I,R) stays pointwise series" `Quick test_binary_max_is_series;
    ];
    "stratified", [
      Alcotest.test_case "peak_time[p in patch] → one leaf per patch" `Quick test_stratified;
    ];
    "derived", [
      Alcotest.test_case "dur = fadeout - takeoff → QBDerived SBinOp Sub" `Quick test_derived;
    ];
    "obs_source", [
      Alcotest.test_case "bare observations.afp series rejected (E289)" `Quick
        test_obs_bare_series_rejected;
      Alcotest.test_case "max(observations.afp) → QSObservation RValue VMax" `Quick test_obs_max;
      Alcotest.test_case "first_above(observations.afp, 0) → QSObservation RTime FirstAbove" `Quick test_obs_first_above;
    ];
    "diagnostics", [
      Alcotest.test_case "E290 reduction in a transition rate" `Quick test_e290_reduction_in_rate;
      Alcotest.test_case "E288 dt in a quantity body" `Quick test_e288_dt_in_quantity;
      Alcotest.test_case "E289 total(x) deferred" `Quick test_e289_total;
      Alcotest.test_case "E289 forward QRef" `Quick test_e289_forward_qref;
      Alcotest.test_case "E289 QRef to a series quantity" `Quick test_e289_qref_to_series;
      Alcotest.test_case "E289 let mixed with a reduced scalar (clear, not 'unknown name')" `Quick test_e289_let_mixed_with_scalar;
      Alcotest.test_case "E289 let-mix hint says lets ARE in scope in series" `Quick test_e289_let_mix_hint_corrects_scope;
      Alcotest.test_case "E289 quantity/let name collision hint teaches distinct-name pattern" `Quick test_e289_quantity_let_collision_hint;
      Alcotest.test_case "E289 undeclared observation stream" `Quick test_e289_obs_undeclared;
      Alcotest.test_case "E289 stratified observation source" `Quick test_e289_obs_stratified;
      Alcotest.test_case "E289 observation source mixed into arithmetic" `Quick test_e289_obs_mixed;
      Alcotest.test_case "E290 observations.afp in a transition rate" `Quick test_e290_obs_in_rate;
    ];
  ]
