(** One mistake in an observation's likelihood or projection is reported once,
    by the diagnostic that names it (gh#1322).

    The parser reports a malformed likelihood wrapper — a `diagnostic_test` or
    `zero_inflated` with the wrong base or missing arguments (E253, E254, E324,
    E325), an unknown family (E104), a value that is not a call (E118) — and
    continues so the rest of the model is still checked. What it leaves behind
    for the expander must not draw a second report, such as E250 ("binomial
    missing required argument 'n'") or E251, about arguments of a family the
    author did not write.

    `prevalence(...)` is accepted only as the whole of `projected = ...`.
    Inside a larger expression the report says where it is accepted and writes
    the expression out without it, rather than calling it an undeclared
    function. *)

let codes_of (src : string) : string list =
  Compiler.collect_diagnostics src
  |> List.filter (fun (d : Diagnostics.diagnostic) ->
         d.Diagnostics.severity = Diagnostics.Error)
  |> List.map (fun (d : Diagnostics.diagnostic) -> d.Diagnostics.code)

let diags_with_code (src : string) (code : string) : Diagnostics.diagnostic list =
  Compiler.collect_diagnostics src
  |> List.filter (fun (d : Diagnostics.diagnostic) -> d.Diagnostics.code = code)

let contains ~needle s =
  let n = String.length needle and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = needle || go (i + 1)) in
  go 0

(* The model every case below shares, with one stream whose projection and
   likelihood are spliced in. *)
let model ?(projected = "I") ?(columns = "time : time, cases : count")
    ~likelihood () =
  Printf.sprintf {|
time_unit = 'days
compartments { S, I, R }
parameters {
  beta : rate  gamma : rate
  sens : probability  spec : probability  zpi : probability
  r : positive  N_tested : count
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { %s }
    projected = %s
    emit_schedule = every 1 'weeks
    %s
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 14 'days }
|} columns projected likelihood

let check_codes name ~likelihood expected =
  Alcotest.(check (list string)) name expected
    (List.sort_uniq compare (codes_of (model ~likelihood ())))

(* ── Malformed wrappers: the wrapper's own error, and nothing else ──────── *)

let test_e253_alone () =
  check_codes "diagnostic_test with a poisson base"
    ~likelihood:"cases ~ diagnostic_test(base = poisson(rate = projected), \
                 sens = sens, spec = spec)"
    ["E253"]

let test_e254_alone () =
  check_codes "diagnostic_test without spec"
    ~likelihood:"cases ~ diagnostic_test(base = binomial(n = N_tested, \
                 p = projected), sens = sens)"
    ["E254"]

let test_e254_no_base_alone () =
  check_codes "diagnostic_test without a base"
    ~likelihood:"cases ~ diagnostic_test(sens = sens, spec = spec)"
    ["E254"]

let test_e324_alone () =
  check_codes "zero_inflated with a poisson base"
    ~likelihood:"cases ~ zero_inflated(base = poisson(rate = projected), pi = zpi)"
    ["E324"]

let test_e325_alone () =
  check_codes "zero_inflated without pi"
    ~likelihood:"cases ~ zero_inflated(base = neg_binomial(mean = projected, r = r))"
    ["E325"]

let test_e325_no_base_alone () =
  check_codes "zero_inflated without a base"
    ~likelihood:"cases ~ zero_inflated(pi = zpi)"
    ["E325"]

let test_e104_alone () =
  check_codes "an unknown family"
    ~likelihood:"cases ~ negbin(mean = projected, r = r)"
    ["E104"]

let test_e118_alone () =
  check_codes "the removed `likelihood =` form with a non-call"
    ~likelihood:"likelihood = 3"
    ["E118"; "E273"]

(* The base's own arguments are still checked: a mistake inside a base the
   wrapper accepted is a second, real mistake, and is reported. *)
let test_base_args_still_checked () =
  check_codes "diagnostic_test without spec, base missing n"
    ~likelihood:"cases ~ diagnostic_test(base = binomial(p = projected), sens = sens)"
    ["E250"; "E254"];
  check_codes "zero_inflated without pi, base with a stray argument"
    ~likelihood:"cases ~ zero_inflated(base = neg_binomial(mean = projected, \
                 r = r, rr = r))"
    ["E251"; "E325"]

(* A reported call keeps the arguments the author wrote, so a data column
   named only inside it is still a column the likelihood reads, not a dead one
   (E277). *)
let test_reported_call_keeps_its_names () =
  let columns = "time : time, cases : count, tested : count" in
  let check name ~likelihood expected =
    Alcotest.(check (list string)) name expected
      (List.sort_uniq compare (codes_of (model ~columns ~likelihood ()))) in
  check "an unknown family naming a data column"
    ~likelihood:"cases ~ negbin(mean = tested * projected, r = r)"
    ["E104"];
  check "diagnostic_test with a poisson base naming a data column"
    ~likelihood:"cases ~ diagnostic_test(base = poisson(rate = tested * projected), \
                 sens = sens, spec = spec)"
    ["E253"];
  check "zero_inflated with a poisson base naming a data column"
    ~likelihood:"cases ~ zero_inflated(base = poisson(rate = tested * projected), \
                 pi = zpi)"
    ["E324"]

(* ── prevalence inside arithmetic ─────────────────────────────────────────── *)

let prevalence_in_arithmetic =
  model ~projected:"2 * prevalence(of = I, among = S + I + R)"
    ~likelihood:"cases ~ poisson(rate = projected)" ()

let test_prevalence_in_arithmetic_names_the_position () =
  Alcotest.(check (list string)) "one error" ["E100"]
    (codes_of prevalence_in_arithmetic);
  match diags_with_code prevalence_in_arithmetic "E100" with
  | [d] ->
    if contains ~needle:"undeclared" d.Diagnostics.message then
      Alcotest.failf "still reported as undeclared: %s" d.Diagnostics.message;
    if not (contains ~needle:"whole" d.Diagnostics.message) then
      Alcotest.failf "message does not say where prevalence is accepted: %s"
        d.Diagnostics.message;
    let hint = Option.value ~default:"" d.Diagnostics.hint in
    if not (contains ~needle:"projected = 2 * (I / (S + I + R))" hint) then
      Alcotest.failf "hint does not write the expression out: %s" hint;
    if d.Diagnostics.loc = Diagnostics.no_loc then
      Alcotest.fail "the report has no source location"
  | ds -> Alcotest.failf "expected one E100, got %d" (List.length ds)

(* Outside a projection (here a likelihood argument) the same report is made:
   it says where `prevalence` is accepted rather than that it does not exist. *)
let test_prevalence_in_likelihood_names_the_position () =
  let src =
    model ~likelihood:"cases ~ poisson(rate = 10 * prevalence(of = I, \
                       among = S + I + R))" () in
  match diags_with_code src "E100" with
  | [d] ->
    if contains ~needle:"undeclared" d.Diagnostics.message then
      Alcotest.failf "still reported as undeclared: %s" d.Diagnostics.message;
    if not (contains ~needle:"projected = prevalence(" d.Diagnostics.message) then
      Alcotest.failf "message does not say where prevalence is accepted: %s"
        d.Diagnostics.message
  | ds -> Alcotest.failf "expected one E100, got %d" (List.length ds)

let () =
  Alcotest.run "obs_family_diagnostics" [
    "malformed wrapper reported once", [
      Alcotest.test_case "E253 alone" `Quick test_e253_alone;
      Alcotest.test_case "E254 alone" `Quick test_e254_alone;
      Alcotest.test_case "E254 (no base) alone" `Quick test_e254_no_base_alone;
      Alcotest.test_case "E324 alone" `Quick test_e324_alone;
      Alcotest.test_case "E325 alone" `Quick test_e325_alone;
      Alcotest.test_case "E325 (no base) alone" `Quick test_e325_no_base_alone;
      Alcotest.test_case "E104 alone" `Quick test_e104_alone;
      Alcotest.test_case "E118 alone" `Quick test_e118_alone;
      Alcotest.test_case "base arguments still checked" `Quick
        test_base_args_still_checked;
      Alcotest.test_case "a reported call keeps its names" `Quick
        test_reported_call_keeps_its_names;
    ];
    "prevalence inside arithmetic", [
      Alcotest.test_case "projection: names the position, writes it out" `Quick
        test_prevalence_in_arithmetic_names_the_position;
      Alcotest.test_case "likelihood argument: names the position" `Quick
        test_prevalence_in_likelihood_names_the_position;
    ];
  ]
