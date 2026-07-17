(* gh#402: `beta_binomial(mean, concentration)` parameterization.

   It lowers to the existing `{n, alpha, beta}` IR with
     alpha = mean * concentration
     beta  = (1 - mean) * concentration
   so the obs-autodiff threads the concentration gradient for free (no IR change).
   This pins the lowering shape, the raw form still working, and the two guards:
   E252 (mixing the parameterizations) and E250 (per-form missing argument). *)

let compile_ok (src : string) : Ir.model =
  match Compiler.compile ~name:"bb_test" src with
  | Ok m -> m
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
    in
    loop 0

let compile_expect_error_code ~code ~contains src =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"bb_test" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

let model_src ~lik =
  Printf.sprintf
    {|
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta   : rate
  gamma  : rate
  kappa  : positive
  n_sero : count
  N0     : count
  I0     : count
}
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  seroprev {
    columns       { time : time, seroprev : count }
    projected     = R / N
    emit_schedule = every 90 'days
    seroprev      ~ %s
  }
}
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days to = 1 'years }
|}
    lik

let find_bb (m : Ir.model) : Ir.beta_binomial_likelihood =
  match
    List.find_map
      (fun (o : Ir.observation_model) ->
        match o.likelihood with Ir.BetaBinomial bb -> Some bb | _ -> None)
      m.observations
  with
  | Some bb -> bb
  | None -> Alcotest.fail "no beta_binomial observation in the model"

let test_mean_concentration_lowers () =
  let m =
    compile_ok
      (model_src ~lik:"beta_binomial(n = n_sero, mean = projected, concentration = kappa)")
  in
  let bb = find_bb m in
  (* alpha = mean * concentration *)
  (match bb.alpha.expr with
   | Ir.BinOp { op = Ir.Mul; _ } -> ()
   | _ -> Alcotest.fail "alpha should lower to (mean * concentration)");
  (* beta = (1 - mean) * concentration *)
  match bb.beta.expr with
  | Ir.BinOp { op = Ir.Mul; left = Ir.BinOp { op = Ir.Sub; left = Ir.Const 1.0; _ }; _ } -> ()
  | _ -> Alcotest.fail "beta should lower to ((1 - mean) * concentration)"

let test_raw_form_still_works () =
  let m =
    compile_ok
      (model_src
         ~lik:"beta_binomial(n = n_sero, alpha = projected * kappa, beta = (1 - projected) * kappa)")
  in
  ignore (find_bb m)

let test_mixed_form_e252 () =
  compile_expect_error_code ~code:"E252" ~contains:"mixes parameterizations"
    (model_src ~lik:"beta_binomial(n = n_sero, alpha = projected * kappa, concentration = kappa)")

let test_missing_concentration_e250 () =
  compile_expect_error_code ~code:"E250" ~contains:"concentration"
    (model_src ~lik:"beta_binomial(n = n_sero, mean = projected)")

let () =
  Compiler.no_dim_check := true;
  Alcotest.run "beta_binomial_params"
    [ ( "lowering",
        [ Alcotest.test_case "(mean, concentration) → alpha=mean*conc, beta=(1-mean)*conc"
            `Quick test_mean_concentration_lowers;
          Alcotest.test_case "raw (alpha, beta) still compiles" `Quick test_raw_form_still_works ] );
      ( "errors",
        [ Alcotest.test_case "mixed parameterizations → E252" `Quick test_mixed_form_e252;
          Alcotest.test_case "missing concentration → E250" `Quick test_missing_concentration_e250 ] )
    ]
