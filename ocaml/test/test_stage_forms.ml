(* `stage(...)` and `stages(...)` (proposal
   2026-11-17-unified-staged-residence, "Explicit stages" and "Existing
   spellings").

   `stage(E[a], 2)` names one residence location of a staged compartment and
   `st in stages(E)` binds a variable over its locations. These tests assert
   what each form compiles to, mostly by comparing the whole compiled model
   with the model that names the same cells by their expanded names
   (`E_pre_s2`), that the same text compiles when the stage count is 1, and
   each diagnostic, with the code its hint suggests compiled in turn. *)

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let with_switches f =
  Compiler.no_dim_check := true;
  Diagnostics.json_errors_mode := true;
  let r = f () in
  Diagnostics.json_errors_mode := false;
  r

let detail (src : string) : (Ir.model, string) result =
  with_switches (fun () -> Compiler.compile ~name:"stage_forms" src)

let compile_ok (src : string) : Ir.model =
  match detail src with
  | Ok m    -> m
  | Error e -> Alcotest.failf "compile failed: %s\n--- model ---\n%s" e src

let contains ~needle s =
  let nl = String.length needle and sl = String.length s in
  let rec loop i = i + nl <= sl && (String.sub s i nl = needle || loop (i + 1)) in
  loop 0

(* The compile must fail with [code], the rendered diagnostics must hold every
   string in [mentions], and none in [absent]. *)
let expect_error ?(absent = []) ~code ~mentions src =
  match detail src with
  | Ok _ -> Alcotest.failf "expected error %s but the model compiled:\n%s" code src
  | Error e ->
    List.iter (fun needle ->
      if not (contains ~needle e) then
        Alcotest.failf "expected the %s diagnostics to contain %S, got: %s"
          code needle e) (code :: mentions);
    List.iter (fun needle ->
      if contains ~needle e then
        Alcotest.failf "the %s diagnostics must not contain %S, got: %s"
          code needle e) absent

let compile_detail src : Compiler.compile_detail =
  match with_switches (fun () ->
      Compiler.compile_detail_result ~name:"stage_forms" src) with
  | Error e -> Alcotest.failf "model should compile: %s\n--- model ---\n%s" e src
  | Ok d -> d

let warnings_of ~code src : Diagnostics.diagnostic list =
  List.filter (fun (dg : Diagnostics.diagnostic) ->
    dg.code = code && dg.severity = Diagnostics.Warning)
    (compile_detail src).ctx.diags.diags

let model_text (m : Ir.model) : string =
  Yojson.Safe.to_string (Serde.model_to_json m)

(* Two sources compile to the same model, field for field. *)
let check_same_model what (a : string) (b : string) =
  Alcotest.(check string) what (model_text (compile_ok a)) (model_text (compile_ok b))

let transition (m : Ir.model) name : Ir.transition =
  match List.find_opt (fun (t : Ir.transition) -> t.Ir.name = name) m.Ir.transitions with
  | Some t -> t
  | None ->
    Alcotest.failf "no transition '%s'; the model has: %s" name
      (String.concat ", " (List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions))

let transition_names (m : Ir.model) =
  List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions

let intervention_names (m : Ir.model) =
  List.map (fun (iv : Ir.intervention) -> iv.Ir.name) m.Ir.interventions

let intervention (m : Ir.model) name : Ir.intervention =
  match List.find_opt (fun (iv : Ir.intervention) -> iv.Ir.name = name)
          m.Ir.interventions with
  | Some iv -> iv
  | None ->
    Alcotest.failf "no intervention '%s'; the model has: %s" name
      (String.concat ", " (intervention_names m))

(* One action, as a string a test can compare. *)
let show_action : Ir.action -> string = function
  | Ir.FractionTransfer { src; dst; fraction; _ } ->
    Printf.sprintf "fraction %s [%s>%s]" (Pp_expr.to_string fraction) src dst
  | Ir.AbsoluteTransfer { src; dst; count } ->
    Printf.sprintf "count %s [%s>%s]" (Pp_expr.to_string count) src dst
  | Ir.GroupedTransfer { pairs; _ } ->
    Printf.sprintf "grouped [%s]"
      (String.concat ", " (List.map (fun (a, b) -> a ^ ">" ^ b) pairs))
  | Ir.Set { compartment; value; _ } ->
    Printf.sprintf "set %s = %s" compartment (Pp_expr.to_string value)
  | Ir.AddAction { add_compartment; add_count; _ } ->
    Printf.sprintf "add %s %s" add_compartment (Pp_expr.to_string add_count)

let check_actions m name expected =
  Alcotest.(check (list string)) name expected
    (List.map show_action (intervention m name).Ir.actions)

(* ── Models ──────────────────────────────────────────────────────────────── *)

(* Age-stratified S, E, I, Q, R. E has an Erlang dwell of [stages] stages and
   I one of [i_stages]. [transitions], [init] and [effects] are added to the
   blocks of those names, [top] at top level. *)
let age_src ?(stages = 3) ?(i_stages = 3) ?(block = "interventions")
    ?(top = "") ?(transitions = "") ?(init = "") ?(observations = "") effects =
  Printf.sprintf
    "time_unit = 'days\n\
     dimensions { age = [pre, sch] }\n\
     compartments { S, E, I, Q, R }\n\
     stratify(by = age)\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  gamma : rate in [0.01, 1.0]\n\
    \  mu    : rate in [0.0, 1.0]\n\
    \  cov   : probability\n\
    \  rho   : probability\n\
     }\n\
     %s\n\
     transitions {\n\
    \  infection[a in age] : S[a] --> E[a] @ beta * S[a]\n\
    \  latency[a in age]   : E[a] --> I[a] via erlang(stages = %d, rate = sigma)\n\
    \  recovery[a in age]  : I[a] --> R[a] via erlang(stages = %d, rate = gamma)\n\
     %s}\n\
     init { S[pre] = 990  S[sch] = 900  E[pre] = 5  E[sch] = 5 %s }\n\
     %s {\n%s}\n\
     %s\n\
     simulate { from = 0 'days  to = 160 'days }\n"
    top stages i_stages transitions init block effects observations

(* The same without strata: a residence reference is the bare name. *)
let flat_src ?(stages = 3) ?(i_law = "@ gamma * I") ?(top = "") ?(transitions = "")
    ?(init = "") ?(observations = "") effects =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, E, I, Q, R }\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  gamma : rate in [0.01, 1.0]\n\
    \  mu    : rate in [0.0, 1.0]\n\
    \  cov   : probability\n\
    \  rho   : probability\n\
     }\n\
     %s\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = %d, rate = sigma)\n\
    \  recovery  : I --> R %s\n\
     %s}\n\
     init { S = 990  E = 5 %s }\n\
     interventions {\n%s}\n\
     %s\n\
     simulate { from = 0 'days  to = 160 'days }\n"
    top stages i_law transitions init effects observations

let weekly projected =
  Printf.sprintf
    "observations {\n\
    \  weekly {\n\
    \    columns       { time : time, weekly : count }\n\
    \    projected     = %s\n\
    \    weekly        ~ poisson(rate = rho * projected)\n\
    \    emit_schedule = every 7 'days\n\
    \  }\n\
     }\n" projected

(* ── `stage(...)` as a state expression ──────────────────────────────────── *)

(* Stage-specific infectiousness: the rate reads one stage of I, the cell the
   expanded name `I_s2` names. *)
let test_stage_in_a_rate () =
  let i_law = "via erlang(stages = 3, rate = gamma)" in
  let with_ infectious =
    flat_src ~i_law
      ~top:"let N = S + E + I + Q + R\n"
      ~transitions:(Printf.sprintf
        "  second : S --> E @ beta * S * %s / N\n" infectious) "" in
  check_same_model "a rate reading stage 2 of I"
    (with_ "stage(I, 2)") (with_ "I_s2");
  (* It is one stage, not the compartment. *)
  Alcotest.(check bool) "not the whole compartment" false
    (model_text (compile_ok (with_ "stage(I, 2)"))
     = model_text (compile_ok (with_ "I")))

(* In a `let`, an observation and `balance`. *)
let test_stage_in_let_observation_and_balance () =
  let m = compile_ok (age_src
    ~top:"let late = stage(E[pre], 3)\n"
    ~transitions:"  seek[a in age] : S[a] --> Q[a] @ mu * late\n" "") in
  Alcotest.(check string) "a let reading one stage" "mu * E_pre_s3"
    (Pp_expr.to_string (transition m "seek_sch").Ir.rate);
  let observed projected = age_src ~observations:(weekly projected) "" in
  check_same_model "an observation reading one stage of each stratum"
    (observed "stage(I[pre], 3) + stage(I[sch], 3)") (observed "I_pre_s3 + I_sch_s3");
  let balanced last =
    flat_src ~top:(Printf.sprintf "balance { R = 2000 - S - %s }\n" last) "" in
  check_same_model "balance reading one stage"
    (balanced "stage(E, 2)") (balanced "E_s2")

(* ── `stage(...)` as an endpoint ─────────────────────────────────────────── *)

(* A transition whose source is one stage is an ordinary transition out of
   that one cell: one transition, not one per stage, and no share of a rate. *)
let test_transition_out_of_one_stage () =
  let m = compile_ok (age_src
    ~transitions:"  late_death : stage(E[pre], 3) --> @ mu * stage(E[pre], 3)\n" "") in
  let t = transition m "late_death" in
  Alcotest.(check (list (pair string int))) "one cell loses one individual"
    [ ("E_pre_s3", -1) ] t.Ir.stoichiometry;
  Alcotest.(check string) "the rate reads that cell" "mu * E_pre_s3"
    (Pp_expr.to_string t.Ir.rate);
  Alcotest.(check bool) "no per-stage copies" false
    (List.exists (fun n -> contains ~needle:"late_death_" n) (transition_names m))

(* A destination written with its stage enters that stage, where a
   destination without one enters stage 1. *)
let test_transition_into_one_stage () =
  let m = compile_ok (age_src
    ~transitions:"  skip[a in age] : S[a] --> stage(E[a], 2) @ mu * S[a]\n\
                 \  seed[a in age] : S[a] --> E[a] @ mu * S[a]\n" "") in
  Alcotest.(check (list (pair string int))) "stage 2"
    [ ("E_pre_s2", 1); ("S_pre", -1) ]
    (List.sort compare (transition m "skip_pre").Ir.stoichiometry);
  Alcotest.(check (list (pair string int))) "stage 1"
    [ ("E_pre_s1", 1); ("S_pre", -1) ]
    (List.sort compare (transition m "seed_pre").Ir.stoichiometry)

(* `transfer`, `add`, `set` and `init`, against the expanded cell names. *)
let test_action_and_init_endpoints () =
  let with_ ~late ~q ~s ~into ~added ~set ~placed =
    age_src ~init:(Printf.sprintf "%s = 10" placed)
      (Printf.sprintf
         "  isolate : transfer(fraction = 0.8, from = %s, to = %s) at [10 'days]\n\
         \  seed    : transfer(count = 3, from = %s, to = %s) at [11 'days]\n\
         \  imp     : add(%s, 5) at [12 'days]\n\
         \  z       : { %s = 4  at = [13 'days] }\n"
         late q s into added set) in
  check_same_model "endpoints written with stage(...)"
    (with_ ~late:"stage(I[pre], 3)" ~q:"Q[pre]" ~s:"S[pre]" ~into:"stage(E[pre], 2)"
       ~added:"stage(E[sch], 3)" ~set:"stage(E[sch], 2)" ~placed:"stage(I[sch], 2)")
    (with_ ~late:"I_pre_s3" ~q:"Q_pre" ~s:"S_pre" ~into:"E_pre_s2"
       ~added:"E_sch_s3" ~set:"E_sch_s2" ~placed:"I_sch_s2")

(* `init` on a staged compartment lands in stage 1, and `stage(...)` places
   elsewhere. *)
let test_init_places_in_a_stage () =
  let m = compile_ok (age_src ~init:"stage(I[pre], 2) = 10  I[sch] = 7" "") in
  let cells = List.map fst m.Ir.initial_conditions in
  Alcotest.(check bool) "stage 2 of I[pre]" true (List.mem "I_pre_s2" cells);
  Alcotest.(check bool) "stage 1 of I[sch]" true (List.mem "I_sch_s1" cells);
  Alcotest.(check bool) "stage 1 of E[pre]" true (List.mem "E_pre_s1" cells);
  Alcotest.(check bool) "not stage 1 of I[pre]" false (List.mem "I_pre_s1" cells)

(* A reference that names its dimensions takes the stage the same way. *)
let test_named_index () =
  check_same_model "a named population index"
    (age_src "  x : transfer(fraction = 0.5, from = stage(E[age = pre], 2), to = Q[pre]) at [10 'days]\n")
    (age_src "  x : transfer(fraction = 0.5, from = E_pre_s2, to = Q_pre) at [10 'days]\n")

(* ── `stages(...)` ───────────────────────────────────────────────────────── *)

let stage_family =
  "  move[st in stages(E)] : transfer(fraction = 0.2, from = stage(E[pre], st), \
   to = stage(E[sch], st)) at [10 'days]\n"

(* A family over the stages has one member per stage, each naming its own. *)
let test_binder_on_an_intervention () =
  let m = compile_ok (age_src stage_family) in
  Alcotest.(check (list string)) "one member per stage"
    [ "move_s1"; "move_s2"; "move_s3" ] (intervention_names m);
  List.iter (fun s ->
    check_actions m ("move_" ^ s)
      [ Printf.sprintf "fraction 0.2 [E_pre_%s>E_sch_%s]" s s ])
    [ "s1"; "s2"; "s3" ]

let test_binder_on_an_event () =
  let m = compile_ok (age_src ~block:"events" stage_family) in
  Alcotest.(check (list string)) "one event per stage"
    [ "move_s1"; "move_s2"; "move_s3" ] (intervention_names m);
  List.iter (fun (iv : Ir.intervention) ->
    Alcotest.(check bool) (iv.Ir.name ^ " is an event") true (iv.Ir.kind = Ir.Event))
    m.Ir.interventions

let test_binder_on_a_transition () =
  let m = compile_ok (age_src
    ~transitions:"  shift[st in stages(E)] : stage(E[pre], st) --> stage(E[sch], st) \
                  @ mu * stage(E[pre], st)\n" "") in
  List.iter (fun s ->
    let t = transition m ("shift_" ^ s) in
    Alcotest.(check (list (pair string int))) ("shift_" ^ s)
      [ ("E_pre_" ^ s, -1); ("E_sch_" ^ s, 1) ]
      (List.sort compare t.Ir.stoichiometry);
    Alcotest.(check string) ("rate of shift_" ^ s) ("mu * E_pre_" ^ s)
      (Pp_expr.to_string t.Ir.rate))
    [ "s1"; "s2"; "s3" ]

(* A flow reference to a transition family over the stages reads every stage
   member, and the same text reads the one transition at one stage. The
   members are named with the stage level last, wherever the binder is
   written. *)
let test_flow_of_a_stage_family () =
  let src ?(read = "incidence(shift[pre])") ~stages () =
    age_src ~stages ~i_stages:stages
      ~transitions:"  shift[st in stages(E), a in age] : stage(E[a], st) --> Q[a] \
                    @ mu * stage(E[a], st)\n\
                   \  drop[st in stages(E)] : stage(E[pre], st) --> \
                    @ mu * stage(E[pre], st)\n"
      ~observations:(Printf.sprintf
        "observations {\n\
        \  weekly {\n\
        \    columns       { time : time, weekly : count }\n\
        \    covers        = closing_at(time, 7 'days)\n\
        \    projected     = %s\n\
        \    weekly        ~ poisson(rate = rho * projected)\n\
        \    emit_schedule = every 7 'days\n\
        \  }\n\
         }\n" read)
      "" in
  let projection src =
    match (compile_ok src).Ir.observations with
    | [ o ] -> o.Ir.projection
    | os -> Alcotest.failf "expected one observation, got %d" (List.length os) in
  Alcotest.(check bool) "three stages: the sum of the stage members" true
    (projection (src ~stages:3 ())
     = Ir.CumulativeFlowSum [ "shift_pre_s1"; "shift_pre_s2"; "shift_pre_s3" ]);
  Alcotest.(check bool) "one stage: the transition" true
    (projection (src ~stages:1 ()) = Ir.CumulativeFlow "shift_pre");
  (* A family over the stages alone is one declared flow: read bare, it
     pools no population stratum. *)
  let bare = "incidence(drop)" in
  Alcotest.(check bool) "bare, three stages" true
    (projection (src ~read:bare ~stages:3 ())
     = Ir.CumulativeFlowSum [ "drop_s1"; "drop_s2"; "drop_s3" ]);
  Alcotest.(check bool) "bare, one stage" true
    (projection (src ~read:bare ~stages:1 ()) = Ir.CumulativeFlow "drop")

let reactive_src ?(stages = 3) action_family =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, E, I, Q, R }\n\
     parameters { beta : rate  sigma : rate  gamma : rate  thr : count  cov : probability\n\
    \             rho : probability }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = %d, rate = sigma)\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     observations {\n\
    \  weekly {\n\
    \    columns       { time : time, weekly : count }\n\
    \    covers        = closing_at(time, 7 'days)\n\
    \    projected     = incidence(infection)\n\
    \    weekly        ~ poisson(rate = rho * projected)\n\
    \    emit_schedule = every 7 'days\n\
    \  }\n\
     }\n\
     reactive_interventions {\n%s}\n\
     init { S = 990  E = 5 }\n\
     simulate { from = 0 'days  to = 160 'days }\n"
    stages action_family

let reactive_family =
  "  pull[st in stages(E)] : when observed(weekly) > thr {\n\
  \    action = transfer(fraction = cov, from = stage(E, st), to = Q)\n\
  \    once   = true\n\
  \  }\n"

let test_binder_on_a_reactive_intervention () =
  Alcotest.(check (list string)) "one policy per stage"
    [ "pull_s1"; "pull_s2"; "pull_s3" ]
    (intervention_names (compile_ok (reactive_src reactive_family)));
  Alcotest.(check (list string)) "one policy at one stage" [ "pull" ]
    (intervention_names (compile_ok (reactive_src ~stages:1 reactive_family)))

(* A sum over the stages of a compartment is the compartment. *)
let test_sum_over_stages () =
  let with_ total =
    age_src ~transitions:(Printf.sprintf
      "  seek[a in age] : S[a] --> Q[a] @ mu * %s\n" total) "" in
  check_same_model "the sum over the stages is the residence reference"
    (with_ "sum(st in stages(E), stage(E[a], st))") (with_ "E[a]")

(* A quantity family over the stages: one cell per stage, each with its stage
   as its stratum, and the one quantity with no stratum at one stage. *)
let test_quantity_family_over_stages () =
  let src stages =
    flat_src ~stages ~top:"quantities { peak[st in stages(E)] = max(stage(E, st)) }\n" "" in
  let cells stages =
    List.map (fun (q : Ir.quantity) ->
      q.Ir.q_name ^ "@" ^ String.concat "," (List.map snd q.Ir.q_stratum))
      (compile_ok (src stages)).Ir.quantities in
  Alcotest.(check (list string)) "three stages"
    [ "peak@s1"; "peak@s2"; "peak@s3" ] (cells 3);
  Alcotest.(check (list string)) "one stage" [ "peak@" ] (cells 1)

(* A `let` family over the stages, read with a stage variable. *)
let test_let_family_over_stages () =
  let m = compile_ok (age_src
    ~top:"let occ[st in stages(E)] = stage(E[pre], st) + stage(E[sch], st)\n"
    ~transitions:"  seek[a in age] : S[a] --> Q[a] @ mu * sum(st in stages(E), occ[st])\n" "") in
  ignore (transition m "seek_pre");
  let one = compile_ok (age_src ~stages:1 ~i_stages:1
    ~top:"let occ[st in stages(E)] = stage(E[pre], st) + stage(E[sch], st)\n"
    ~transitions:"  seek[a in age] : S[a] --> Q[a] @ mu * sum(st in stages(E), occ[st])\n" "") in
  ignore (transition one "seek_pre")

(* ── The same text at every stage count ──────────────────────────────────── *)

(* Every form with location 1 or a stage variable. *)
let every_form_transitions =
  "  first[a in age] : stage(E[a], 1) --> Q[a] @ mu * stage(E[a], 1)\n\
  \  shift[st in stages(E)] : stage(E[pre], st) --> stage(E[sch], st) \
   @ mu * stage(E[pre], st)\n\
  \  seek[a in age] : S[a] --> Q[a] @ mu * sum(st in stages(I), stage(I[a], st))\n"

let every_form_effects =
  "  move[st in stages(E)] : transfer(fraction = 0.2, from = stage(E[pre], st), \
   to = stage(E[sch], st)) at [10 'days]\n\
  \  pull : transfer(fraction = 0.8, from = stage(I[pre], 1), to = Q[pre]) at [11 'days]\n\
  \  imp  : add(stage(E[pre], 1), 5) at [12 'days]\n\
  \  z    : { stage(E[sch], 1) = 4  at = [13 'days] }\n"

let every_form ~stages =
  age_src ~stages ~i_stages:stages ~transitions:every_form_transitions
    ~init:"stage(I[pre], 1) = 2" ~observations:(weekly "stage(I[pre], 1)")
    every_form_effects

(* At one stage, `stage(X, 1)` is `X` and a binder over `stages(X)` has one
   element: the model is the one written without either. *)
let test_one_stage_is_the_plain_model () =
  check_same_model "stages = 1"
    (every_form ~stages:1)
    (age_src ~stages:1 ~i_stages:1
       ~transitions:"  first[a in age] : E[a] --> Q[a] @ mu * E[a]\n\
                    \  shift : E[pre] --> E[sch] @ mu * E[pre]\n\
                    \  seek[a in age] : S[a] --> Q[a] @ mu * I[a]\n"
       ~init:"I[pre] = 2" ~observations:(weekly "I[pre]")
       "  move : transfer(fraction = 0.2, from = E[pre], to = E[sch]) at [10 'days]\n\
       \  pull : transfer(fraction = 0.8, from = I[pre], to = Q[pre]) at [11 'days]\n\
       \  imp  : add(E[pre], 5) at [12 'days]\n\
       \  z    : { E[sch] = 4  at = [13 'days] }\n")

let test_same_text_at_one_and_three_stages () =
  let three = compile_ok (every_form ~stages:3) in
  Alcotest.(check (list string)) "the families at three stages"
    [ "move_s1"; "move_s2"; "move_s3"; "pull"; "imp"; "z" ] (intervention_names three);
  ignore (transition three "shift_s3");
  ignore (transition three "first_pre")

(* A compartment with no `via` law has one stage. *)
let test_unstaged_compartment_has_one_stage () =
  check_same_model "stage(S, 1) is S"
    (flat_src ~transitions:"  x : stage(S, 1) --> Q @ mu * stage(S, 1)\n"
       "  v[st in stages(S)] : transfer(fraction = cov, from = stage(S, st), to = R) at [5 'days]\n")
    (flat_src ~transitions:"  x : S --> Q @ mu * S\n"
       "  v : transfer(fraction = cov, from = S, to = R) at [5 'days]\n")

(* ── The restart warning ─────────────────────────────────────────────────── *)

(* An endpoint written with `stage(...)` states its location: no W107. *)
let test_stage_endpoint_silences_w107 () =
  let count src = List.length (warnings_of ~code:"W107" src) in
  Alcotest.(check int) "a transfer between staged compartments warns" 1
    (count (age_src "  jump : transfer(fraction = 0.5, from = E[pre], to = I[pre]) at [10 'days]\n"));
  Alcotest.(check int) "a stage(...) destination" 0
    (count (age_src "  jump : transfer(fraction = 0.5, from = E[pre], to = stage(I[pre], 2)) at [10 'days]\n"));
  Alcotest.(check int) "a stage(...) source" 0
    (count (age_src "  jump : transfer(fraction = 0.5, from = stage(E[pre], 3), to = I[pre]) at [10 'days]\n"));
  Alcotest.(check int) "a competing exit into a staged compartment warns" 1
    (count (age_src ~transitions:"  early[a in age] : E[a] --> I[a] @ mu * E[a]\n" ""));
  Alcotest.(check int) "its destination written with stage(...)" 0
    (count (age_src ~transitions:"  early[a in age] : E[a] --> stage(I[a], 2) @ mu * E[a]\n" ""));
  Alcotest.(check int) "its source written with stage(...)" 0
    (count (age_src ~transitions:"  early[a in age] : stage(E[a], 3) --> I[a] @ mu * stage(E[a], 3)\n" ""));
  (* The staged member of a family over the compartments. *)
  let family dst =
    age_src ~transitions:(Printf.sprintf
      "  out[(a, b) in consecutive(age), c in compartments] : c[a] --> %s @ mu * c[a]\n"
      dst) "" in
  Alcotest.(check int) "a family member into a staged compartment warns" 1
    (count (family "I[b]"));
  Alcotest.(check int) "its destination written with stage(...)" 0
    (count (family "stage(I[b], 2)"));
  (* The destination stage of the competing exit is the one written. *)
  let m = compile_ok (age_src
    ~transitions:"  early[a in age] : E[a] --> stage(I[a], 2) @ mu * E[a]\n" "") in
  Alcotest.(check (list (pair string int))) "enters stage 2"
    [ ("E_pre_s1", -1); ("I_pre_s2", 1) ]
    (List.sort compare (transition m "early_pre_s1").Ir.stoichiometry)

(* The W107 hint names `stage(E, 1)` to `stage(I, 1)`; that movement compiles
   and is not warned about. *)
let test_w107_hint_compiles () =
  let two_staged effects =
    flat_src ~i_law:"via erlang(stages = 2, rate = gamma)" effects in
  (match warnings_of ~code:"W107"
           (two_staged "  jump : transfer(fraction = 0.5, from = E, to = I) at [10 'days]\n") with
   | [ w ] ->
     let hint = Option.value ~default:"" w.Diagnostics.hint in
     List.iter (fun needle ->
       Alcotest.(check bool) ("the hint mentions " ^ needle) true (contains ~needle hint))
       [ "`stage(E, 1)`"; "`stage(I, 1)`" ]
   | ws -> Alcotest.failf "expected one W107, got %d" (List.length ws));
  Alcotest.(check int) "the hinted movement is not warned about" 0
    (List.length (warnings_of ~code:"W107"
       (two_staged "  jump : transfer(fraction = 0.5, from = stage(E, 1), to = stage(I, 1)) at [10 'days]\n")))

(* ── `set` and a stage into the same compartment: the hints ──────────────── *)

let test_e366_hint_compiles () =
  expect_error ~code:"E366" ~mentions:[ "sets 'E_pre'"; "`stage(E[pre], 1) = ...`" ]
    ~absent:[ "E_pre_s1 =" ]
    (age_src "  z : { E[pre] = 5  at = [10 'days] }\n");
  expect_error ~code:"E366" ~mentions:[ "sets 'E'"; "`stage(E, 1) = ...`" ]
    (flat_src "  z : { E = 5  at = [10 'days] }\n");
  ignore (compile_ok (age_src "  z : { stage(E[pre], 1) = 5  at = [10 'days] }\n"));
  ignore (compile_ok (flat_src "  z : { stage(E, 1) = 5  at = [10 'days] }\n"))

let test_e367_hint_compiles () =
  expect_error ~code:"E367"
    ~mentions:[ "one stage of the staged compartment 'E' (E_pre_s2)";
                "`to = stage(E[sch], 1)`" ]
    ~absent:[ "E[sch, s1]" ]
    (age_src "  x : transfer(fraction = 0.5, from = stage(E[pre], 2), to = E[sch]) at [10 'days]\n");
  ignore (compile_ok (age_src
    "  x : transfer(fraction = 0.5, from = stage(E[pre], 2), to = stage(E[sch], 1)) at [10 'days]\n"))

(* ── Locations ───────────────────────────────────────────────────────────── *)

let test_location_out_of_range () =
  expect_error ~code:"E370"
    ~mentions:[ "`stage(E[pre], 4)`"; "'E' has 3 stages";
                "`via erlang(stages = 3, ...)` on 'latency'"; "from 1 to 3" ]
    (age_src "  x : add(stage(E[pre], 4), 5) at [10 'days]\n");
  expect_error ~code:"E370" ~mentions:[ "`stage(E[pre], 0)`"; "'E' has 3 stages" ]
    (age_src "  x : add(stage(E[pre], 0), 5) at [10 'days]\n");
  expect_error ~code:"E370" ~mentions:[ "`stage(E[pre], 1.5)`"; "whole number" ]
    (age_src "  x : add(stage(E[pre], 1.5), 5) at [10 'days]\n");
  (* Not a literal and not a stage variable. *)
  expect_error ~code:"E370" ~mentions:[ "`stage(E[pre], a)`"; "`in stages(E)`" ]
    (age_src "  x[a in age] : add(stage(E[pre], a), 5) at [10 'days]\n")

let test_location_at_one_stage () =
  expect_error ~code:"E370"
    ~mentions:[ "`stage(E[pre], 2)`"; "'E' has 1 stage";
                "`via erlang(stages = 1, ...)` on 'latency'"; "`stage(E[pre], 1)`" ]
    (age_src ~stages:1 "  x : add(stage(E[pre], 2), 5) at [10 'days]\n");
  expect_error ~code:"E370"
    ~mentions:[ "`stage(S[pre], 2)`"; "'S' has no `via` dwell, and so 1 stage" ]
    (age_src "  x : add(stage(S[pre], 2), 5) at [10 'days]\n");
  (* The hint's reference compiles. *)
  ignore (compile_ok (age_src ~stages:1 "  x : add(stage(E[pre], 1), 5) at [10 'days]\n"))

let test_location_of_another_compartment () =
  expect_error ~code:"E371"
    ~mentions:[ "`stage(I[pre], st)`"; "'st' ranges over the stages of 'E'";
                "not of 'I'"; "'I' has 3 stages"; "`stages(I)`" ]
    (age_src "  x[st in stages(E)] : transfer(fraction = 0.5, from = stage(E[pre], st), \
              to = stage(I[pre], st)) at [10 'days]\n");
  (* The hint: a variable of its own, or a number. *)
  ignore (compile_ok (age_src
    "  x[st in stages(E)] : transfer(fraction = 0.5, from = stage(E[pre], st), \
     to = stage(I[pre], 1)) at [10 'days]\n"))

let test_stage_variable_misused () =
  expect_error ~code:"E372"
    ~mentions:[ "'st' is bound by `st in stages(E)`"; "is used as an index of 'S'";
                "`stage(E[...], st)`" ]
    (age_src "  x[st in stages(E)] : transfer(fraction = 0.5, from = S[st], to = Q[pre]) at [10 'days]\n");
  expect_error ~code:"E372" ~mentions:[ "is used as an index of 'E'" ]
    (age_src "  x[st in stages(E)] : add(E[st], 5) at [10 'days]\n");
  expect_error ~code:"E372" ~mentions:[ "is read as a value (`st`)" ]
    (age_src "  x[st in stages(E)] : add(stage(E[pre], st), 5 * st) at [10 'days]\n");
  expect_error ~code:"E372" ~mentions:[ "is compared in a `where` predicate" ]
    (age_src "  x[st in stages(E)] : add(stage(E[pre], st), 5) at [10 'days] where st != s1\n")

(* ── What `stage` and `stages` take ──────────────────────────────────────── *)

let test_stage_needs_a_residence_reference () =
  expect_error ~code:"E369"
    ~mentions:[ "`stage(E, 2)`"; "'E' is stratified by [age]";
                "indexes 0 of its 1 population dimension"; "`stage(E[<age>], ...)`" ]
    (age_src "  x : add(stage(E, 2), 5) at [10 'days]\n");
  expect_error ~code:"E369" ~mentions:[ "`stage(beta, 2)`"; "'beta' is not a compartment" ]
    (age_src ~transitions:"  x[a in age] : S[a] --> Q[a] @ stage(beta, 2) * S[a]\n" "");
  expect_error ~code:"E369" ~mentions:[ "`stage(c[a], 1)`"; "'c' is not a compartment" ]
    (age_src ~transitions:"  x[a in age, c in compartments] : stage(c[a], 1) --> @ mu * c[a]\n" "");
  expect_error ~code:"E369" ~mentions:[ "`stage(E[pre])`"; "is not a stage reference" ]
    (age_src ~transitions:"  x[a in age] : S[a] --> Q[a] @ mu * stage(E[pre])\n" "")

let test_stages_is_a_binder_domain () =
  expect_error ~code:"E374" ~mentions:[ "`st in stages(beta)`"; "'beta' is not a compartment" ]
    (age_src "  x[st in stages(beta)] : add(E[pre], 5) at [10 'days]\n");
  expect_error ~code:"E374" ~mentions:[ "`stages(E)` is read as a value"; "no stage count" ]
    (age_src "  x : transfer(fraction = 0.6 / stages(E), from = E[pre], to = Q[pre]) at [10 'days]\n");
  expect_error ~code:"E374" ~mentions:[ "binds an index of an `init` entry"; "`stage(E, 2) = ...`" ]
    (flat_src ~init:"E[st in stages(E)] = 3" "");
  expect_error ~code:"E374"
    ~mentions:[ "`st in stages(E)` binds an index of an observation stream";
                "`projected = stage(E, 1)`" ]
    (flat_src ~observations:
       "observations {\n\
       \  occ[st in stages(E)] {\n\
       \    columns       { time : time, occ : count }\n\
       \    projected     = stage(E, st)\n\
       \    occ           ~ poisson(rate = rho * projected)\n\
       \    emit_schedule = every 7 'days\n\
       \  }\n\
        }\n" "");
  ignore (compile_ok (flat_src ~observations:(weekly "stage(E, 1)") ""));
  (* After `in`, another call is a parse error that names the forms. *)
  expect_error ~code:"E125" ~mentions:[ "`in levels(...)`"; "`in stages(E)`" ]
    (age_src "  x[st in levels(E)] : add(E[pre], 5) at [10 'days]\n");
  expect_error ~code:"E126" ~mentions:[ "`cell(...)` is not a compartment reference"; "`stage(E, 1)`" ]
    (flat_src "  x : add(cell(E, 1), 5) at [10 'days]\n")

(* `hyper_erlang`: both forms are reserved. *)
let hyper_src effects =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, I, R, D }\n\
     parameters { beta : rate  cfr : probability }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S\n\
    \  outcome   : I via hyper_erlang(\n\
    \    branch(label = fatal, weight = cfr, stages = 2, mean = 8 'days, to = D),\n\
    \    branch(label = recover, stages = 3, mean = 12 'days, to = R))\n\
     }\n\
     init { S = 990  I = 10 }\n\
     interventions {\n%s}\n\
     simulate { from = 0 'days  to = 60 'days }\n" effects

let test_hyper_erlang_is_reserved () =
  expect_error ~code:"E373"
    ~mentions:[ "`stage(I, 1)`"; "`via hyper_erlang(...)` dwell (on 'outcome')";
                "are reserved"; "a branch and a phase" ]
    (hyper_src "  x : transfer(fraction = 0.5, from = stage(I, 1), to = R) at [10 'days]\n");
  expect_error ~code:"E373" ~mentions:[ "`st in stages(I)`"; "are reserved" ]
    (hyper_src "  x[st in stages(I)] : transfer(fraction = 0.5, from = S, to = R) at [10 'days]\n")

(* ── The spellings these forms replace ───────────────────────────────────── *)

(* A generated stage dimension named in a binder, a sum, a named index or a
   declaration: refused, naming `stages(E)`. *)
let test_generated_dimension_by_name_is_refused () =
  expect_error ~code:"E375"
    ~mentions:[ "`st in __latency_stage`"; "'__latency_stage'";
                "Erlang dwell of 'E'"; "`st in stages(E)`"; "`stage(E[...], st)`";
                "camdl docs language-changes" ]
    (age_src "  move[st in __latency_stage] : transfer(fraction = 0.2, \
              from = E[pre, st], to = E[sch, st]) at [10 'days]\n");
  expect_error ~code:"E375"
    ~mentions:[ "`s in __latency_stage`"; "`sum(s in stages(E), ...)`";
                "camdl docs language-changes" ]
    (age_src ~transitions:"  seek[a in age] : S[a] --> Q[a] \
                           @ mu * sum(s in __latency_stage, E[a, s])\n" "");
  expect_error ~code:"E375"
    ~mentions:[ "`__latency_stage = s2`"; "`stage(E[...], <location>)`" ]
    (age_src "  x : add(E[age = pre, __latency_stage = s2], 5) at [10 'days]\n");
  expect_error ~code:"E375" ~mentions:[ "`w[__latency_stage]`"; "written per location" ]
    (age_src ~top:"parameters { w[__latency_stage] : rate in [0, 1] }\n" "");
  (* At one stage no stage dimension exists; the name is refused the same way. *)
  expect_error ~code:"E375" ~mentions:[ "`st in __latency_stage`"; "`st in stages(E)`" ]
    (age_src ~stages:1
       "  move[st in __latency_stage] : add(E[pre], 5) at [10 'days]\n")

(* A stage level as an index: refused, naming the `stage(...)` to write. *)
let test_stage_level_index_is_refused () =
  let refused ~written ~replacement src =
    expect_error ~code:"E376"
      ~mentions:[ Printf.sprintf "`%s` indexes a stage of 'E'" written;
                  Printf.sprintf "write `%s`" replacement;
                  "camdl docs language-changes" ]
      src in
  refused ~written:"E[pre, s2]" ~replacement:"stage(E[pre], 2)"
    (age_src "  x : transfer(fraction = 0.5, from = E[pre, s2], to = Q[pre]) at [10 'days]\n");
  refused ~written:"E[pre, s3]" ~replacement:"stage(E[pre], 3)"
    (age_src ~transitions:"  late : E[pre, s3] --> @ mu * S[pre]\n" "");
  refused ~written:"E[sch, s2]" ~replacement:"stage(E[sch], 2)"
    (age_src ~init:"E[sch, s2] = 4" "");
  refused ~written:"E[a, s1]" ~replacement:"stage(E[a], 1)"
    (age_src ~transitions:"  seek[a in age] : S[a] --> Q[a] @ mu * E[a, s1]\n" "");
  refused ~written:"E[s2]" ~replacement:"stage(E, 2)"
    (flat_src "  x : add(E[s2], 5) at [10 'days]\n");
  (* With the old binder, both halves of the migration are named. *)
  refused ~written:"E[pre, st]" ~replacement:"stage(E[pre], st)"
    (age_src "  move[st in __latency_stage] : transfer(fraction = 0.2, \
              from = E[pre, st], to = E[sch, st]) at [10 'days]\n");
  (* Under a binder over the compartments. *)
  refused ~written:"c[a, s1]" ~replacement:"stage(E[a], 1)"
    (age_src ~transitions:"  x[a in age, c in compartments] : c[a] --> @ mu * c[a, s1]\n" "")

(* Each replacement the two diagnostics name compiles. *)
let test_migrations_compile () =
  ignore (compile_ok (age_src
    "  move[st in stages(E)] : transfer(fraction = 0.2, \
     from = stage(E[pre], st), to = stage(E[sch], st)) at [10 'days]\n\
    \  x : transfer(fraction = 0.5, from = stage(E[pre], 2), to = Q[pre]) at [11 'days]\n"));
  ignore (compile_ok (age_src ~init:"stage(E[sch], 2) = 4"
    ~transitions:"  late : stage(E[pre], 3) --> @ mu * S[pre]\n\
                 \  seek[a in age] : S[a] --> Q[a] @ mu * stage(E[a], 1)\n\
                 \  pool[a in age] : S[a] --> R[a] \
                  @ mu * sum(s in stages(E), stage(E[a], s))\n" ""));
  ignore (compile_ok (flat_src "  x : add(stage(E, 2), 5) at [10 'days]\n"))

(* An expanded cell name is how every stratified cell is named, and stays. *)
let test_expanded_cell_names_stay () =
  let m = compile_ok (age_src ~init:"E_sch_s2 = 4"
    ~transitions:"  seek[a in age] : S[a] --> Q[a] @ mu * E_pre_s3\n"
    "  x : transfer(fraction = 0.5, from = E_pre_s2, to = Q_pre) at [10 'days]\n\
    \  z : { E_pre_s1 = 5  at = [11 'days] }\n") in
  Alcotest.(check bool) "the init entry" true
    (List.mem_assoc "E_sch_s2" m.Ir.initial_conditions)

(* A model's own dimension of that name, with no `via` law to generate one,
   is an ordinary dimension. *)
let test_a_declared_dimension_of_that_name_is_ordinary () =
  ignore (compile_ok
    "time_unit = 'days\n\
     dimensions { __latency_stage = [s1, s2] }\n\
     compartments { S, E, I }\n\
     stratify(by = __latency_stage, only = [E])\n\
     parameters { beta : rate  sigma : rate }\n\
     transitions {\n\
    \  infection : S --> E[s1] @ beta * S\n\
    \  step[(s, n) in consecutive(__latency_stage)] : E[s] --> E[n] @ sigma * E[s]\n\
    \  latency : E[s2] --> I @ sigma * E[s2]\n\
     }\n\
     init { S = 990  E[s1] = 5 }\n\
     simulate { from = 0 'days  to = 60 'days }\n")

(* ── `stage` and `stages` are not reserved words ─────────────────────────── *)

let test_stage_and_stages_are_ordinary_names () =
  ignore (compile_ok
    "time_unit = 'days\n\
     dimensions { stages = [early, late] }\n\
     compartments { S, stage, R }\n\
     stratify(by = stages, only = [stage])\n\
     parameters { beta : rate  gamma : rate }\n\
     transitions {\n\
    \  infection : S --> stage[early] @ beta * S\n\
    \  step      : stage[early] --> stage[late] @ gamma * stage[early]\n\
    \  recovery[st in stages] : stage[st] --> R @ gamma * stage[st]\n\
     }\n\
     init { S = 990  stage[early] = 5 }\n\
     simulate { from = 0 'days  to = 60 'days }\n")

(* ── The declared-model section ──────────────────────────────────────────── *)

(* Every name of a model written with `stage(...)` and `stages(...)` is
   classified: no W203. *)
let test_declared_section_classifies_stage_references () =
  List.iter (fun stages ->
    let d = compile_detail (every_form ~stages) in
    let _, unresolved = Lazy.force d.declared in
    Alcotest.(check (list string))
      (Printf.sprintf "no unclassified name at stages = %d" stages) []
      (List.map (fun (u : Declared_emit.unresolved) -> u.u_name) unresolved))
    [ 1; 3 ]

(* ── Main ────────────────────────────────────────────────────────────────── *)

let () =
  let quick name f = Alcotest.test_case name `Quick f in
  Alcotest.run "stage_forms"
    [ ( "state expressions",
        [ quick "stage(...) in a rate" test_stage_in_a_rate;
          quick "in a let, an observation" test_stage_in_let_observation_and_balance ] );
      ( "endpoints",
        [ quick "a transition out of one stage" test_transition_out_of_one_stage;
          quick "a transition into one stage" test_transition_into_one_stage;
          quick "transfer, add, set and init" test_action_and_init_endpoints;
          quick "init places in a stage" test_init_places_in_a_stage;
          quick "a named population index" test_named_index ] );
      ( "stages(...)",
        [ quick "on an intervention" test_binder_on_an_intervention;
          quick "on an event" test_binder_on_an_event;
          quick "on a transition" test_binder_on_a_transition;
          quick "on a reactive intervention" test_binder_on_a_reactive_intervention;
          quick "a flow reference reads the stage members" test_flow_of_a_stage_family;
          quick "in a sum" test_sum_over_stages;
          quick "on a quantity" test_quantity_family_over_stages;
          quick "on a let" test_let_family_over_stages ] );
      ( "every stage count",
        [ quick "one stage is the plain model" test_one_stage_is_the_plain_model;
          quick "the same text at 1 and 3" test_same_text_at_one_and_three_stages;
          quick "an unstaged compartment" test_unstaged_compartment_has_one_stage ] );
      ( "restart warning",
        [ quick "a stage(...) endpoint silences W107" test_stage_endpoint_silences_w107;
          quick "the W107 hint compiles" test_w107_hint_compiles ] );
      ( "hints",
        [ quick "E366" test_e366_hint_compiles;
          quick "E367" test_e367_hint_compiles ] );
      ( "locations",
        [ quick "out of range → E370" test_location_out_of_range;
          quick "at one stage → E370" test_location_at_one_stage;
          quick "another compartment's → E371" test_location_of_another_compartment;
          quick "a stage variable elsewhere → E372" test_stage_variable_misused ] );
      ( "arguments",
        [ quick "stage(...) takes a residence reference → E369"
            test_stage_needs_a_residence_reference;
          quick "stages(...) is a binder domain → E374, E125, E126"
            test_stages_is_a_binder_domain;
          quick "hyper_erlang is reserved → E373" test_hyper_erlang_is_reserved ] );
      ( "replaced spellings",
        [ quick "a generated dimension by name → E375"
            test_generated_dimension_by_name_is_refused;
          quick "a stage level as an index → E376" test_stage_level_index_is_refused;
          quick "the replacements compile" test_migrations_compile;
          quick "expanded cell names stay" test_expanded_cell_names_stay;
          quick "a declared dimension of that name"
            test_a_declared_dimension_of_that_name_is_ordinary ] );
      ( "names",
        [ quick "stage and stages are not reserved"
            test_stage_and_stages_are_ordinary_names ] );
      ( "declared model",
        [ quick "no W203" test_declared_section_classifies_stage_references ] ) ]
