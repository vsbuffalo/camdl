(* The `grouped_transfer` IR action (proposal
   2026-11-17-unified-staged-residence, "The grouped transfer action"): its
   JSON, the validator's shape rules (E521), and the derivative of its
   fraction. What the compiler lowers to it is tested in
   `test_staged_transfers.ml`. *)

let contains ~needle s =
  let nl = String.length needle and sl = String.length s in
  let rec loop i = i + nl <= sl && (String.sub s i nl = needle || loop (i + 1)) in
  loop 0

(* A model with a staged compartment written by hand (`E_s1`, `E_s2`, `E_s3`)
   and one intervention whose action each test replaces. *)
let base_src =
  "time_unit = 'days\n\
   dimensions { st = [s1, s2, s3] }\n\
     compartments { S, E, R, W : real }\n\
     stratify(by = st, only = [E])\n\
     parameters { beta : rate  sigma : rate  cov : probability }\n\
     transitions {\n\
    \  infection : S --> E[s1] @ beta * S\n\
    \  clear[s in st] : E[s] --> R @ sigma * E[s]\n\
     }\n\
     ode { W = 0 - sigma * W }\n\
     init { S = 990  E[s1] = 5 }\n\
     interventions { move : transfer(fraction = cov, from = S, to = R) at [10 'days] }\n\
     simulate { from = 0 'days  to = 30 'days }\n"

let base_model () : Ir.model =
  Compiler.no_dim_check := true;
  match Compiler.compile ~name:"grouped_ir" base_src with
  | Ok m -> m
  | Error e -> Alcotest.failf "the base model should compile: %s" e

let with_action (m : Ir.model) (a : Ir.action) : Ir.model =
  { m with Ir.interventions =
      List.map (fun (iv : Ir.intervention) -> { iv with Ir.actions = [ a ] })
        m.Ir.interventions }

let fraction_of pairs =
  Ir.GroupedTransfer
    { pairs; amount = Ir.GroupedFraction { fraction = Ir.Param "cov"; fraction_grad = [] } }

let e_to_r = [ ("E_s1", "R"); ("E_s2", "R"); ("E_s3", "R") ]

let roundtrip (a : Ir.action) = Serde.action_of_json (Serde.action_to_json a)

let test_grouped_transfer_serialization () =
  let frac =
    Ir.GroupedTransfer
      { pairs = [ ("E_pre_s1", "E_sch_s1"); ("E_pre_s2", "E_sch_s2") ];
        amount = Ir.GroupedFraction
            { fraction = Ir.Param "cov";
              fraction_grad = [ ("cov", Ir.DEGrad (Ir.Const 1.0)) ] } } in
  Alcotest.(check string) "the fraction form on the wire"
    {|{"grouped_transfer":{"pairs":[["E_pre_s1","E_sch_s1"],["E_pre_s2","E_sch_s2"]],"fraction":{"param":"cov"},"fraction_grad":{"cov":{"grad":{"const":1.0}}}}}|}
    (Yojson.Safe.to_string (Serde.action_to_json frac));
  Alcotest.(check bool) "the fraction form round-trips" true (roundtrip frac = frac);
  let count =
    Ir.GroupedTransfer
      { pairs = [ ("E_s1", "R"); ("E_s2", "R") ];
        amount = Ir.GroupedCount { count = Ir.Const 5.0 } } in
  Alcotest.(check string) "the count form on the wire"
    {|{"grouped_transfer":{"pairs":[["E_s1","R"],["E_s2","R"]],"count":{"const":5.0}}}|}
    (Yojson.Safe.to_string (Serde.action_to_json count));
  Alcotest.(check bool) "the count form round-trips" true (roundtrip count = count);
  (* Exactly one amount. *)
  List.iter (fun json ->
    match Serde.action_of_json (Yojson.Safe.from_string json) with
    | exception _ -> ()
    | _ -> Alcotest.failf "should not parse: %s" json)
    [ {|{"grouped_transfer":{"pairs":[["E_s1","R"],["E_s2","R"]]}}|};
      {|{"grouped_transfer":{"pairs":[["E_s1","R"],["E_s2","R"]],"fraction":{"const":0.5},"count":{"const":5.0}}}|};
      {|{"grouped_transfer":{"pairs":[["E_s1","R","Q"],["E_s2","R"]],"fraction":{"const":0.5}}}|} ]

(* The validator requires at least two pairs, each source cell once, declared
   cells, and one representation; a well-formed action passes. *)
let test_malformed_grouped_transfer_is_e521 () =
  let m = base_model () in
  let reasons m' = match Validate.validate m' with
    | Ok () -> []
    | Error es -> List.map Validate.error_to_string es in
  Alcotest.(check (list string)) "a well-formed grouped transfer is valid" []
    (reasons (with_action m (fraction_of e_to_r)));
  let expect what pairs needle =
    let rs = reasons (with_action m (fraction_of pairs)) in
    Alcotest.(check bool) (what ^ ": " ^ String.concat "; " rs) true
      (List.exists (fun r -> contains ~needle r) rs) in
  expect "one pair" [ ("E_s1", "R") ] "lists fewer than two";
  expect "a repeated source" [ ("E_s1", "R"); ("E_s1", "S") ]
    "lists the source cell 'E_s1' more than once";
  expect "an undeclared cell" [ ("E_s1", "R"); ("E_s9", "R") ] "unknown compartment: E_s9";
  expect "both representations" [ ("E_s1", "R"); ("E_s2", "W") ]
    "mixes integer and real compartments";
  (* The compiler reports it as E521, naming the intervention. *)
  let detail =
    match Compiler.compile_detail_result ~name:"grouped_ir" base_src with
    | Ok d -> d
    | Error e -> Alcotest.failf "the base model should compile: %s" e in
  match Validate.validate (with_action m (fraction_of [ ("E_s1", "R") ])) with
  | Ok () -> Alcotest.fail "expected a validation error"
  | Error es ->
    let ds = List.map (Compiler.diagnose_validate_error detail.ctx) es in
    Alcotest.(check bool) "E521 names the intervention" true
      (List.exists (fun (d : Diagnostics.diagnostic) ->
         d.code = "E521" && contains ~needle:"'move'" d.message) ds)

(* The fraction is differentiated as a fraction transfer's is; a count is left
   alone, having no smooth jump. *)
let test_differentiate_action () =
  let diff = Autodiff.differentiate_action [ "cov"; "beta" ] [] [] in
  (match diff (fraction_of e_to_r) with
   | Ir.GroupedTransfer { pairs; amount = Ir.GroupedFraction { fraction; fraction_grad } } ->
     Alcotest.(check bool) "pairs unchanged" true (pairs = e_to_r);
     Alcotest.(check bool) "fraction unchanged" true (fraction = Ir.Param "cov");
     Alcotest.(check bool) "d cov / d cov = 1, and no entry for beta" true
       (fraction_grad = [ ("cov", Ir.DEGrad (Ir.Const 1.0)) ])
   | _ -> Alcotest.fail "expected a grouped fraction transfer");
  let count =
    Ir.GroupedTransfer { pairs = e_to_r; amount = Ir.GroupedCount { count = Ir.Param "cov" } } in
  Alcotest.(check bool) "a count is returned unchanged" true (diff count = count)

let () =
  Alcotest.run "grouped_transfer_ir"
    [ ("grouped transfer",
       [ Alcotest.test_case "wire shape and round trip" `Quick
           test_grouped_transfer_serialization;
         Alcotest.test_case "malformed → E521" `Quick
           test_malformed_grouped_transfer_is_e521;
         Alcotest.test_case "the fraction is differentiated, a count is not" `Quick
           test_differentiate_action ]) ]
