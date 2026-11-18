(* Transfers and `add` on a staged compartment (proposal
   2026-11-17-unified-staged-residence, "Residence mapping rules", "Transfers
   and `add`" and "The grouped transfer action").

   A `transfer` endpoint, and an `add` target, may be a residence reference: a
   compartment with an Erlang dwell, written with its population dimensions
   and no stage. These tests assert the compiled actions: which cells each one
   pairs, that the stages of one population cell move under one grouped
   transfer, and each diagnostic. The action itself is tested in
   `test_grouped_transfer_ir.ml`, and its arithmetic, which is the runtime's,
   in `rust/crates/sim/src/effects/`. *)

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let with_switches ?(dim_check = false) f =
  Compiler.no_dim_check := not dim_check;
  Diagnostics.json_errors_mode := true;
  let r = f () in
  Diagnostics.json_errors_mode := false;
  Compiler.no_dim_check := true;
  r

let detail ?dim_check (src : string) : (Ir.model, string) result =
  with_switches ?dim_check (fun () -> Compiler.compile ~name:"staged_transfers" src)

let compile_ok ?dim_check (src : string) : Ir.model =
  match detail ?dim_check src with
  | Ok m    -> m
  | Error e -> Alcotest.failf "compile failed: %s" e

let contains ~needle s =
  let nl = String.length needle and sl = String.length s in
  let rec loop i = i + nl <= sl && (String.sub s i nl = needle || loop (i + 1)) in
  loop 0

(* The compile must fail with [code], the rendered diagnostics must hold every
   string in [mentions], and none in [absent]. *)
let expect_error ?dim_check ?(absent = []) ~code ~mentions src =
  match detail ?dim_check src with
  | Ok _ -> Alcotest.failf "expected error %s but the model compiled" code
  | Error e ->
    List.iter (fun needle ->
      if not (contains ~needle e) then
        Alcotest.failf "expected the %s diagnostics to contain %S, got: %s"
          code needle e) (code :: mentions);
    List.iter (fun needle ->
      if contains ~needle e then
        Alcotest.failf "the %s diagnostics must not contain %S, got: %s"
          code needle e) absent

let warnings_of ~code src : Diagnostics.diagnostic list =
  match with_switches (fun () ->
      Compiler.compile_detail_result ~name:"staged_transfers" src) with
  | Error e -> Alcotest.failf "model should compile: %s" e
  | Ok d ->
    List.filter (fun (dg : Diagnostics.diagnostic) ->
      dg.code = code && dg.severity = Diagnostics.Warning) d.ctx.diags.diags

let intervention (m : Ir.model) name : Ir.intervention =
  match List.find_opt (fun (iv : Ir.intervention) -> iv.Ir.name = name)
          m.Ir.interventions with
  | Some iv -> iv
  | None ->
    Alcotest.failf "no intervention '%s'; the model has: %s" name
      (String.concat ", "
         (List.map (fun (iv : Ir.intervention) -> iv.Ir.name) m.Ir.interventions))

(* One action, as a string a test can compare: the kind, the cells, and the
   amount when it is a constant or a parameter. *)
let show_amount = function
  | Ir.Const c -> Printf.sprintf "%g" c
  | Ir.Param p -> p
  | _ -> "<expr>"

let show_pairs pairs =
  String.concat ", " (List.map (fun (s, d) -> s ^ ">" ^ d) pairs)

let show_action : Ir.action -> string = function
  | Ir.FractionTransfer { src; dst; fraction; _ } ->
    Printf.sprintf "fraction %s [%s>%s]" (show_amount fraction) src dst
  | Ir.AbsoluteTransfer { src; dst; count } ->
    Printf.sprintf "count %s [%s>%s]" (show_amount count) src dst
  | Ir.GroupedTransfer { pairs; amount = Ir.GroupedFraction { fraction; _ } } ->
    Printf.sprintf "grouped fraction %s [%s]" (show_amount fraction) (show_pairs pairs)
  | Ir.GroupedTransfer { pairs; amount = Ir.GroupedCount { count } } ->
    Printf.sprintf "grouped count %s [%s]" (show_amount count) (show_pairs pairs)
  | Ir.Set { compartment; value; _ } ->
    Printf.sprintf "set %s = %s" compartment (show_amount value)
  | Ir.AddAction { add_compartment; add_count; _ } ->
    Printf.sprintf "add %s %s" add_compartment (show_amount add_count)

let actions (m : Ir.model) name : string list =
  List.map show_action (intervention m name).Ir.actions

let check_actions m name expected =
  Alcotest.(check (list string)) name expected (actions m name)

(* ── Models ──────────────────────────────────────────────────────────────── *)

(* Age-stratified S, E, I, R. E has an Erlang dwell of [stages] stages and I
   one of [i_stages]. [effects] is the body of [block]. *)
let age_src ?(block = "interventions") ?(stages = 3) ?(i_stages = 2) effects =
  Printf.sprintf
    "time_unit = 'days\n\
     dimensions { age = [pre, sch] }\n\
     compartments { S, E, I, R }\n\
     stratify(by = age)\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  gamma : rate in [0.01, 1.0]\n\
    \  cov   : probability\n\
    \  n_mv  : count in [0, 100]\n\
     }\n\
     transitions {\n\
    \  infection[a in age] : S[a] --> E[a] @ beta * S[a]\n\
    \  latency[a in age]   : E[a] --> I[a] via erlang(stages = %d, rate = sigma)\n\
    \  recovery[a in age]  : I[a] --> R[a] via erlang(stages = %d, rate = gamma)\n\
     }\n\
     init { S[pre] = 990  S[sch] = 900  E[pre] = 5  E[sch] = 5 }\n\
     %s {\n%s}\n\
     simulate { from = 0 'days  to = 160 'days }\n"
    stages i_stages block effects

(* The same without strata: the residence reference is bare `E`. *)
let flat_src ?(stages = 3) effects =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, E, I, R }\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  gamma : rate in [0.01, 1.0]\n\
    \  cov   : probability\n\
     }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = %d, rate = sigma)\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     init { S = 990  E = 5 }\n\
     interventions {\n%s}\n\
     simulate { from = 0 'days  to = 160 'days }\n"
    stages effects

let e_stages age = List.map (fun s -> Printf.sprintf "E_%s_%s" age s) [ "s1"; "s2"; "s3" ]

let grouped ?(kind = "fraction") amount pairs =
  Printf.sprintf "grouped %s %s [%s]" kind amount (show_pairs pairs)

(* ── Residence mapping ───────────────────────────────────────────────────── *)

(* Between strata of one staged compartment the stage is kept: one grouped
   transfer pairing stage i with stage i. *)
let test_same_compartment_keeps_the_stage () =
  let m = compile_ok (age_src
    "  move : transfer(fraction = 0.2, from = E[pre], to = E[sch]) at [10 'days]\n") in
  check_actions m "move"
    [ grouped "0.2" (List.combine (e_stages "pre") (e_stages "sch")) ]

(* An unstaged source into a staged compartment is an ordinary transfer into
   its first stage. *)
let test_unstaged_source_enters_stage_one () =
  let m = compile_ok (age_src
    "  seed : transfer(fraction = cov, from = S[pre], to = E[pre]) at [10 'days]\n\
    \  seedn : transfer(count = 4, from = S[pre], to = E[pre]) at [11 'days]\n") in
  check_actions m "seed" [ "fraction cov [S_pre>E_pre_s1]" ];
  check_actions m "seedn" [ "count 4 [S_pre>E_pre_s1]" ];
  Alcotest.(check int) "no W107 for an unstaged source" 0
    (List.length (warnings_of ~code:"W107" (age_src
       "  seed : transfer(fraction = cov, from = S[pre], to = E[pre]) at [10 'days]\n")))

(* A staged source into an unstaged compartment draws from every stage. *)
let test_staged_source_draws_from_every_stage () =
  let m = compile_ok (age_src
    "  clear : transfer(fraction = cov, from = E[pre], to = R[pre]) at [10 'days]\n") in
  check_actions m "clear"
    [ grouped "cov" (List.map (fun s -> (s, "R_pre")) (e_stages "pre")) ]

(* A staged source into a different staged compartment draws from every stage
   and enters the destination's first stage, with one W107 saying the dwell
   restarts. *)
let jump = "  jump : transfer(fraction = 0.5, from = E[pre], to = I[pre]) at [10 'days]\n"

let test_other_staged_compartment_restarts_and_warns () =
  let m = compile_ok (age_src jump) in
  check_actions m "jump"
    [ grouped "0.5" (List.map (fun s -> (s, "I_pre_s1")) (e_stages "pre")) ];
  match warnings_of ~code:"W107" (age_src jump) with
  | [ w ] ->
    List.iter (fun needle ->
      Alcotest.(check bool) ("W107 mentions " ^ needle) true
        (contains ~needle (w.message ^ " " ^ Option.value w.hint ~default:"")))
      [ "'jump'"; "from the staged compartment 'E'"; "restarts 'I' at stage 1";
        "stage(...)" ]
  | ws -> Alcotest.failf "expected one W107, got %d" (List.length ws)

(* A family warns once, not once per member; and a source that names its
   stage does not warn. *)
let test_restart_warning_is_per_declaration_and_silenced_by_a_stage () =
  Alcotest.(check int) "one W107 for a family of two" 1
    (List.length (warnings_of ~code:"W107" (age_src
       "  jump[a in age] : transfer(fraction = 0.5, from = E[a], to = I[a]) at [10 'days]\n")));
  let explicit =
    age_src "  late : transfer(fraction = 0.5, from = stage(E[pre], 3), to = I[pre]) at [10 'days]\n" in
  check_actions (compile_ok explicit) "late" [ "fraction 0.5 [E_pre_s3>I_pre_s1]" ];
  Alcotest.(check int) "no W107 when the source names its stage" 0
    (List.length (warnings_of ~code:"W107" explicit))

(* An endpoint that names a stage is that cell only. *)
let test_explicit_stage_is_one_cell () =
  let m = compile_ok (age_src
    "  late : transfer(fraction = 0.5, from = stage(E[pre], 2), to = R[pre]) at [10 'days]\n\
    \  into : transfer(fraction = 0.5, from = E[pre], to = stage(E[sch], 1)) at [11 'days]\n") in
  check_actions m "late" [ "fraction 0.5 [E_pre_s2>R_pre]" ];
  check_actions m "into"
    [ grouped "0.5" (List.map (fun s -> (s, "E_sch_s1")) (e_stages "pre")) ]

(* Without strata the residence reference is the bare name. *)
let test_unstratified_residence_reference_is_the_bare_name () =
  let m = compile_ok (flat_src
    "  clear : transfer(fraction = cov, from = E, to = R) at [10 'days]\n\
    \  seed  : transfer(fraction = cov, from = S, to = E) at [11 'days]\n\
    \  imp   : add(E, 5) at [12 'days]\n") in
  check_actions m "clear"
    [ grouped "cov" [ ("E_s1", "R"); ("E_s2", "R"); ("E_s3", "R") ] ];
  check_actions m "seed" [ "fraction cov [S>E_s1]" ];
  check_actions m "imp" [ "add E_s1 5" ]

(* Named indices are placed as everywhere else. *)
let test_named_index_on_a_residence_reference () =
  let m = compile_ok (age_src
    "  clear : transfer(fraction = cov, from = E[age = sch], to = R[age = sch]) at [10 'days]\n\
    \  imp   : add(E[age = sch], 5) at [11 'days]\n") in
  check_actions m "clear"
    [ grouped "cov" (List.map (fun s -> (s, "R_sch")) (e_stages "sch")) ];
  check_actions m "imp" [ "add E_sch_s1 5" ]

(* ── Population dimensions are never pooled ──────────────────────────────── *)

(* Bare endpoints pair cell for cell over `age`, and group only the stages
   within each age cell: one grouped transfer per age cell, in one
   intervention. *)
let test_bare_endpoints_pair_over_population_dimensions () =
  let m = compile_ok (age_src
    "  clear : transfer(fraction = cov, from = E, to = R) at [10 'days]\n\
    \  shift : transfer(fraction = cov, from = E, to = I) at [11 'days]\n\
    \  seed  : transfer(fraction = cov, from = S, to = E) at [12 'days]\n") in
  check_actions m "clear"
    [ grouped "cov" (List.map (fun s -> (s, "R_pre")) (e_stages "pre"));
      grouped "cov" (List.map (fun s -> (s, "R_sch")) (e_stages "sch")) ];
  check_actions m "shift"
    [ grouped "cov" (List.map (fun s -> (s, "I_pre_s1")) (e_stages "pre"));
      grouped "cov" (List.map (fun s -> (s, "I_sch_s1")) (e_stages "sch")) ];
  check_actions m "seed"
    [ "fraction cov [S_pre>E_pre_s1]"; "fraction cov [S_sch>E_sch_s1]" ]

(* A family over a population dimension fires one grouped transfer per
   member: the effect is not repeated across stages or members. *)
let test_family_over_population_fires_one_grouped_transfer_per_cell () =
  let m = compile_ok (age_src
    "  clear[a in age] : transfer(fraction = cov, from = E[a], to = R[a]) at [10 'days]\n") in
  Alcotest.(check (list string)) "one member per age"
    [ "clear_pre"; "clear_sch" ]
    (List.map (fun (iv : Ir.intervention) -> iv.Ir.name) m.Ir.interventions);
  check_actions m "clear_pre"
    [ grouped "cov" (List.map (fun s -> (s, "R_pre")) (e_stages "pre")) ];
  check_actions m "clear_sch"
    [ grouped "cov" (List.map (fun s -> (s, "R_sch")) (e_stages "sch")) ];
  (* Every stage cell is a source exactly once across the family. *)
  let sources =
    List.concat_map (fun (iv : Ir.intervention) ->
      List.concat_map (function
        | Ir.GroupedTransfer { pairs; _ } -> List.map fst pairs
        | Ir.FractionTransfer _ | Ir.AbsoluteTransfer _ | Ir.Set _ | Ir.AddAction _ -> [])
        iv.Ir.actions) m.Ir.interventions in
  Alcotest.(check (list string)) "each stage cell drained once"
    (e_stages "pre" @ e_stages "sch") sources

(* A bare staged endpoint inside a family would repeat the movement, as a
   bare unstaged one does. *)
let test_bare_staged_endpoint_in_a_family_is_e239 () =
  expect_error ~code:"E239" ~mentions:[ "'E' is bare"; "all 2 cells" ]
    (age_src "  clear[a in age] : transfer(fraction = cov, from = E, to = R) at [10 'days]\n")

(* Endpoints of different population shapes do not pair, and the refusal
   names the population dimensions only: never the generated stage dimension,
   and never a cell that does not exist. *)
let test_shape_mismatch_names_population_dimensions_only () =
  expect_error ~code:"E237"
    ~mentions:[ "'E' stratified by [age]"; "'S' (unstratified, or a single cell)" ]
    ~absent:[ "__latency_stage"; "E_s1" ]
    (age_src "  bad : transfer(fraction = cov, from = E, to = S[pre]) at [10 'days]\n")

(* ── count ───────────────────────────────────────────────────────────────── *)

let test_count_from_a_staged_source_is_one_grouped_count () =
  let m = compile_ok (age_src
    "  clr[a in age] : transfer(count = n_mv, from = E[a], to = R[a]) at [10 'days]\n") in
  check_actions m "clr_pre"
    [ grouped ~kind:"count" "n_mv" (List.map (fun s -> (s, "R_pre")) (e_stages "pre")) ];
  (* The count is not multiplied by the number of stages, and a count over
     bare stratified endpoints is still refused. *)
  expect_error ~code:"E238" ~mentions:[ "2 × count" ] ~absent:[ "6 × count" ]
    (age_src "  clr : transfer(count = 3, from = E, to = R) at [10 'days]\n")

(* ── add and set ─────────────────────────────────────────────────────────── *)

let test_add_enters_stage_one () =
  let m = compile_ok (age_src
    "  imp : add(E[pre], 5) at [10 'days]\n\
    \  imps[a in age] : add(E[a], n_mv) at [11 'days]\n") in
  check_actions m "imp" [ "add E_pre_s1 5" ];
  check_actions m "imps_sch" [ "add E_sch_s1 n_mv" ]

(* `set` on a residence reference is refused: a total has no single meaning
   across the stages. The hint's own suggestion compiles. *)
let test_set_on_a_residence_reference_is_refused () =
  expect_error ~code:"E366"
    ~mentions:[ "sets 'E_pre'"; "Erlang dwell"; "3 stages"; "`stage(E[pre], 1) = ...`" ]
    ~absent:[ "E265"; "__latency_stage" ]
    (age_src "  z : { E[pre] = 5  at = [10 'days] }\n");
  expect_error ~code:"E366" ~mentions:[ "sets 'E'"; "`stage(E, 1) = ...`" ]
    (flat_src "  z : { E = 5  at = [10 'days] }\n");
  let m = compile_ok (age_src "  z : { stage(E[pre], 1) = 5  at = [10 'days] }\n") in
  check_actions m "z" [ "set E_pre_s1 = 5" ]

(* A bare stratified staged compartment is still not one cell for `add`. *)
let test_add_to_a_bare_stratified_staged_compartment_is_e265 () =
  expect_error ~code:"E265" ~mentions:[ "adds to 'E'"; "E_pre_s1" ]
    (age_src "  imp : add(E, 5) at [10 'days]\n")

(* ── A stage into the same compartment's residence ───────────────────────── *)

(* A source that names its stage with a destination in the same compartment
   that does not: refused, and the hint's suggestion compiles. *)
let test_stage_into_same_compartment_residence_is_refused () =
  expect_error ~code:"E367"
    ~mentions:[ "one stage of the staged compartment 'E' (E_pre_s2)";
                "`to = stage(E[sch], 1)`" ]
    (age_src "  x : transfer(fraction = 0.5, from = stage(E[pre], 2), to = E[sch]) at [10 'days]\n");
  let m = compile_ok (age_src
    "  x : transfer(fraction = 0.5, from = stage(E[pre], 2), to = stage(E[sch], 1)) at [10 'days]\n") in
  check_actions m "x" [ "fraction 0.5 [E_pre_s2>E_sch_s1]" ]

(* ── Same text for every stage count ─────────────────────────────────────── *)

let every_form =
  "  move  : transfer(fraction = 0.2, from = E[pre], to = E[sch]) at [10 'days]\n\
  \  seed  : transfer(fraction = cov, from = S[pre], to = E[pre]) at [11 'days]\n\
  \  clear : transfer(fraction = cov, from = E, to = R) at [12 'days]\n\
  \  clr[a in age] : transfer(count = 3, from = E[a], to = R[a]) at [13 'days]\n\
  \  jump  : transfer(fraction = 0.5, from = E[pre], to = I[pre]) at [14 'days]\n\
  \  imp   : add(E[pre], 5) at [15 'days]\n"

(* The text that compiles to grouped transfers at three stages compiles at
   one, where nothing is staged, to the plain transfers. *)
let test_same_text_at_one_and_three_stages () =
  let one = compile_ok (age_src ~stages:1 ~i_stages:1 every_form) in
  check_actions one "move" [ "fraction 0.2 [E_pre>E_sch]" ];
  check_actions one "seed" [ "fraction cov [S_pre>E_pre]" ];
  check_actions one "clear" [ "fraction cov [E_pre>R_pre]"; "fraction cov [E_sch>R_sch]" ];
  check_actions one "clr_pre" [ "count 3 [E_pre>R_pre]" ];
  check_actions one "jump" [ "fraction 0.5 [E_pre>I_pre]" ];
  check_actions one "imp" [ "add E_pre 5" ];
  let three = compile_ok (age_src every_form) in
  Alcotest.(check (list string)) "the same interventions"
    (List.map (fun (iv : Ir.intervention) -> iv.Ir.name) one.Ir.interventions)
    (List.map (fun (iv : Ir.intervention) -> iv.Ir.name) three.Ir.interventions);
  (* One action per population cell at either stage count. *)
  List.iter2 (fun (a : Ir.intervention) (b : Ir.intervention) ->
    Alcotest.(check int) (a.Ir.name ^ ": the same number of actions")
      (List.length a.Ir.actions) (List.length b.Ir.actions))
    one.Ir.interventions three.Ir.interventions

(* ── Events and reactive interventions ───────────────────────────────────── *)

let test_events_lower_the_same_way () =
  let m = compile_ok (age_src ~block:"events" every_form) in
  check_actions m "move"
    [ grouped "0.2" (List.combine (e_stages "pre") (e_stages "sch")) ];
  check_actions m "seed" [ "fraction cov [S_pre>E_pre_s1]" ];
  check_actions m "clr_sch"
    [ grouped ~kind:"count" "3" (List.map (fun s -> (s, "R_sch")) (e_stages "sch")) ];
  check_actions m "imp" [ "add E_pre_s1 5" ];
  Alcotest.(check bool) "an event" true ((intervention m "move").Ir.kind = Ir.Event)

let reactive_src action =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, E, I, R }\n\
     parameters { beta : rate  sigma : rate  gamma : rate  thr : count  cov : probability }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = 3, rate = sigma)\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     observations {\n\
    \  weekly {\n\
    \    columns       { time : time, weekly : count }\n\
    \    covers        = closing_at(time, 7 'days)\n\
    \    projected     = incidence(recovery)\n\
    \    emit_schedule = every 7 'days\n\
    \    weekly        ~ poisson(rate = projected)\n\
    \  }\n\
     }\n\
     reactive_interventions {\n\
    \  resp : when observed(weekly) >= thr {\n\
    \    action = %s\n\
    \  }\n\
     }\n\
     init { S = 990  E = 5 }\n\
     simulate { from = 0 'days  to = 60 'days }\n" action

let test_reactive_interventions_lower_the_same_way () =
  let m = compile_ok (reactive_src "transfer(fraction = cov, from = E, to = R)") in
  check_actions m "resp"
    [ grouped "cov" [ ("E_s1", "R"); ("E_s2", "R"); ("E_s3", "R") ] ];
  let m = compile_ok (reactive_src "transfer(fraction = cov, from = S, to = E)") in
  check_actions m "resp" [ "fraction cov [S>E_s1]" ];
  let m = compile_ok (reactive_src "add(E, 5)") in
  check_actions m "resp" [ "add E_s1 5" ]

(* ── The amount ──────────────────────────────────────────────────────────── *)

(* The fraction of a grouped transfer carries its derivative, as a fraction
   transfer's does; a count carries none. *)
let test_grouped_fraction_carries_its_gradient () =
  let m = compile_ok (age_src
    "  clear : transfer(fraction = cov, from = E[pre], to = R[pre]) at [10 'days]\n\
    \  half  : transfer(fraction = 0.5, from = E[pre], to = R[pre]) at [11 'days]\n") in
  (match (intervention m "clear").Ir.actions with
   | [ Ir.GroupedTransfer { amount = Ir.GroupedFraction { fraction_grad; _ }; _ } ] ->
     Alcotest.(check bool) "d cov / d cov = 1" true
       (fraction_grad = [ ("cov", Ir.DEGrad (Ir.Const 1.0)) ])
   | _ -> Alcotest.fail "expected one grouped fraction transfer");
  match (intervention m "half").Ir.actions with
  | [ Ir.GroupedTransfer { amount = Ir.GroupedFraction { fraction_grad; _ }; _ } ] ->
    Alcotest.(check int) "a constant fraction has no gradient" 0 (List.length fraction_grad)
  | _ -> Alcotest.fail "expected one grouped fraction transfer"

(* The fraction is a pure number and the count a number of people, whether
   the transfer is grouped or not. *)
let test_grouped_amount_is_dimension_checked () =
  expect_error ~dim_check:true ~code:"E314" ~mentions:[ "fraction" ]
    (age_src "  clear : transfer(fraction = sigma, from = E[pre], to = R[pre]) at [10 'days]\n");
  expect_error ~dim_check:true ~code:"E314" ~mentions:[ "count" ]
    (age_src "  clear : transfer(count = sigma, from = E[pre], to = R[pre]) at [10 'days]\n");
  ignore (compile_ok ~dim_check:true (age_src
    "  clear : transfer(fraction = cov, from = E[pre], to = R[pre]) at [10 'days]\n\
    \  clr   : transfer(count = n_mv, from = E[pre], to = R[pre]) at [11 'days]\n"))

(* ── Other refusals ──────────────────────────────────────────────────────── *)

let hyper_src effects =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     parameters { beta : rate  p : probability  tau_a : positive  tau_b : positive }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S\n\
    \  clearance : I --> R via hyper_erlang(\n\
    \    branch(label = typical,   weight = p, stages = 2, mean = tau_a),\n\
    \    branch(label = prolonged,             stages = 1, mean = tau_b)\n\
    \  )\n\
     }\n\
     init { S = 990  I = 5 }\n\
     interventions {\n%s}\n\
     simulate { from = 0 'days  to = 160 'days }\n" effects

(* A `hyper_erlang` compartment has no stage dimension; an action on it is
   refused by name, not as an undeclared compartment. *)
let test_action_on_a_hyper_erlang_compartment_is_refused () =
  List.iter (fun (effect, what) ->
    expect_error ~code:"E368"
      ~mentions:[ what ^ " 'I'"; "hyper_erlang"; "'clearance'" ]
      ~absent:[ "E100"; "E264"; "E265" ]
      (hyper_src effect))
    [ ("  x : transfer(fraction = 0.5, from = I, to = R) at [10 'days]\n", "transfers from or into");
      ("  x : transfer(fraction = 0.5, from = S, to = I) at [10 'days]\n", "transfers from or into");
      ("  x : add(I, 5) at [10 'days]\n", "adds to");
      ("  x : { I = 5  at = [10 'days] }\n", "sets") ]

(* An endpoint that leaves a population dimension unindexed is E287, and only
   that: no second diagnostic about the placeholder it leaves. *)
let test_partial_endpoint_reports_only_e287 () =
  let src =
    "time_unit = 'days\n\
     dimensions { age = [pre, sch]  risk = [lo, hi] }\n\
     compartments { S, V }\n\
     stratify(by = age)\n\
     stratify(by = risk)\n\
     parameters { cov : probability  w : rate }\n\
     transitions { waning[a in age, r in risk] : V[a, r] --> S[a, r] @ w * V[a, r] }\n\
     init { S[a in age, r in risk] = 100 }\n\
     interventions { vacc : transfer(fraction = cov, from = S[pre], to = V[pre]) at [10 'days] }\n\
     simulate { from = 0 'days  to = 10 'days }\n" in
  expect_error ~code:"E287" ~mentions:[ "'S'"; "only 1 of 2" ] ~absent:[ "E264" ] src

let () =
  Alcotest.run "staged_transfers"
    [ ("residence mapping",
       [ Alcotest.test_case "same compartment keeps the stage" `Quick
           test_same_compartment_keeps_the_stage;
         Alcotest.test_case "unstaged source enters stage 1" `Quick
           test_unstaged_source_enters_stage_one;
         Alcotest.test_case "staged source draws from every stage" `Quick
           test_staged_source_draws_from_every_stage;
         Alcotest.test_case "other staged compartment: stage 1, W107" `Quick
           test_other_staged_compartment_restarts_and_warns;
         Alcotest.test_case "W107 once per declaration, silenced by a stage" `Quick
           test_restart_warning_is_per_declaration_and_silenced_by_a_stage;
         Alcotest.test_case "an explicit stage is one cell" `Quick
           test_explicit_stage_is_one_cell;
         Alcotest.test_case "unstratified: the bare name" `Quick
           test_unstratified_residence_reference_is_the_bare_name;
         Alcotest.test_case "named index" `Quick
           test_named_index_on_a_residence_reference ]);
      ("population dimensions",
       [ Alcotest.test_case "bare endpoints pair over age, group stages" `Quick
           test_bare_endpoints_pair_over_population_dimensions;
         Alcotest.test_case "a family fires one grouped transfer per cell" `Quick
           test_family_over_population_fires_one_grouped_transfer_per_cell;
         Alcotest.test_case "bare staged endpoint in a family → E239" `Quick
           test_bare_staged_endpoint_in_a_family_is_e239;
         Alcotest.test_case "shape mismatch → E237, population dimensions only" `Quick
           test_shape_mismatch_names_population_dimensions_only ]);
      ("count",
       [ Alcotest.test_case "one grouped count; bare stratified → E238" `Quick
           test_count_from_a_staged_source_is_one_grouped_count ]);
      ("add and set",
       [ Alcotest.test_case "add enters stage 1" `Quick test_add_enters_stage_one;
         Alcotest.test_case "set on a residence reference → E366" `Quick
           test_set_on_a_residence_reference_is_refused;
         Alcotest.test_case "add to a bare stratified staged compartment → E265" `Quick
           test_add_to_a_bare_stratified_staged_compartment_is_e265 ]);
      ("stage into residence",
       [ Alcotest.test_case "a stage into the same compartment → E367" `Quick
           test_stage_into_same_compartment_residence_is_refused ]);
      ("stage count",
       [ Alcotest.test_case "same text at stages = 1 and stages = 3" `Quick
           test_same_text_at_one_and_three_stages ]);
      ("events and reactive",
       [ Alcotest.test_case "events" `Quick test_events_lower_the_same_way;
         Alcotest.test_case "reactive interventions" `Quick
           test_reactive_interventions_lower_the_same_way ]);
      ("amount",
       [ Alcotest.test_case "fraction gradient" `Quick
           test_grouped_fraction_carries_its_gradient;
         Alcotest.test_case "dimension check" `Quick
           test_grouped_amount_is_dimension_checked ]);
      ("refusals",
       [ Alcotest.test_case "action on a hyper_erlang compartment → E368" `Quick
           test_action_on_a_hyper_erlang_compartment_is_refused;
         Alcotest.test_case "partial endpoint → E287 alone" `Quick
           test_partial_endpoint_reports_only_e287 ]) ]
