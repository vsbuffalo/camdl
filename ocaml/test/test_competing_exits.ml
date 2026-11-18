(* Competing exits from a staged compartment (proposal
   2026-11-17-unified-staged-residence, "Competing exits" and the residence
   mapping rules as they apply to ordinary `@` transitions).

   A transition written beside a `via erlang(...)` transition, with the staged
   compartment as its single source, is expanded to one copy per stage. These
   tests assert the compiled IR: the anchor compares a `via` model against the
   same model with the stage dimension written by hand, and the others
   evaluate compiled rates, read the stoichiometry, or read the projection of
   an observation. *)

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let with_switches ?(dim_check = false) f =
  Compiler.no_dim_check := not dim_check;
  Diagnostics.json_errors_mode := true;
  let r = f () in
  Diagnostics.json_errors_mode := false;
  Compiler.no_dim_check := true;
  r

(* The whole pipeline, as `camdlc FILE` runs it. *)
let detail ?dim_check (src : string) : (Ir.model, string) result =
  with_switches ?dim_check (fun () -> Compiler.compile ~name:"competing_exits" src)

let compile_ok ?dim_check (src : string) : Ir.model =
  match detail ?dim_check src with
  | Ok m    -> m
  | Error e -> Alcotest.failf "compile failed: %s" e

let contains ~needle s =
  let nl = String.length needle and sl = String.length s in
  let rec loop i = i + nl <= sl && (String.sub s i nl = needle || loop (i + 1)) in
  loop 0

(* The compile must fail with [code], and the rendered diagnostics must hold
   every string in [mentions]. *)
let expect_error ~code ~mentions src =
  match detail src with
  | Ok _ -> Alcotest.failf "expected error %s but the model compiled" code
  | Error e ->
    List.iter (fun needle ->
      if not (contains ~needle e) then
        Alcotest.failf "expected the %s diagnostics to contain %S, got: %s"
          code needle e) (code :: mentions)

let refuse_error ~code src =
  match detail src with
  | Ok _ -> ()
  | Error e ->
    if contains ~needle:code e then
      Alcotest.failf "did not expect %s, got: %s" code e

let warnings_of ~code src : Diagnostics.diagnostic list =
  match with_switches (fun () ->
      Compiler.compile_detail_result ~name:"competing_exits" src) with
  | Error e -> Alcotest.failf "model should compile: %s" e
  | Ok d ->
    List.filter (fun (dg : Diagnostics.diagnostic) ->
      dg.code = code && dg.severity = Diagnostics.Warning) d.ctx.diags.diags

let transition (m : Ir.model) name : Ir.transition =
  match List.find_opt (fun (t : Ir.transition) -> t.Ir.name = name) m.Ir.transitions with
  | Some t -> t
  | None ->
    Alcotest.failf "no transition '%s'; the model has: %s" name
      (String.concat ", " (List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions))

let transition_names (m : Ir.model) =
  List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions

let stoich (t : Ir.transition) = List.sort compare t.Ir.stoichiometry

(* Evaluate a compiled rate at a state. Covers the nodes these models lower
   to; anything else fails the test rather than being guessed at. *)
let rec eval ~pops ~params (m : Ir.model) (e : Ir.expr) : float =
  let ev = eval ~pops ~params m in
  let pop n = match List.assoc_opt n pops with
    | Some v -> v
    | None -> Alcotest.failf "state has no compartment '%s'" n in
  match e with
  | Ir.Const c -> c
  | Ir.Param p ->
    (match List.assoc_opt p params with
     | Some v -> v
     | None -> Alcotest.failf "no value for parameter '%s'" p)
  | Ir.Pop n -> pop n
  | Ir.PopSum ns -> List.fold_left (fun acc n -> acc +. pop n) 0.0 ns
  | Ir.Reduce ts -> List.fold_left (fun acc t -> acc +. ev t) 0.0 ts
  | Ir.UncheckedDim u -> ev u.inner
  | Ir.BindingRef b ->
    (match List.find_opt (fun (bd : Ir.binding) -> bd.Ir.bname = b) m.Ir.bindings with
     | Some bd -> ev bd.Ir.bexpr
     | None -> Alcotest.failf "no binding '%s'" b)
  | Ir.Cond c -> if ev c.pred > 0.0 then ev c.then_ else ev c.else_
  | Ir.BinOp { op; left; right } ->
    let l = ev left in
    (match op with
     | Ir.Add -> l +. ev right
     | Ir.Sub -> l -. ev right
     | Ir.Mul -> l *. ev right
     | Ir.Div -> l /. ev right
     | Ir.Gt  -> if l > ev right then 1.0 else 0.0
     | _ -> Alcotest.fail "eval: operator not covered by this test evaluator")
  | _ -> Alcotest.fail "eval: expression node not covered by this test evaluator"

(* ── Models ──────────────────────────────────────────────────────────────── *)

(* An age-stratified latent compartment with an Erlang dwell, a linear death
   from it, and ageing within it. [stages] is the only thing a test varies. *)
let age_src ?(extra_params = "") ?(extra = "") ?(obs = "") ~stages () =
  Printf.sprintf
    "time_unit = 'days\n\
     dimensions { age = [pre, sch] }\n\
     compartments { S, E, I, T }\n\
     stratify(by = age)\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  mu    : rate in [0.0, 1.0]\n\
    \  ar    : rate in [0.0, 1.0]\n\
    \  cmax  : rate in [0.0, 50.0]\n\
    \  K     : count in [1, 100]\n\
    \  k_nb  : positive in [0.1, 100]\n\
     %s\
     }\n\
     transitions {\n\
    \  infection[a in age] : S[a] --> E[a] @ beta * S[a]\n\
    \  latency[a in age]   : E[a] --> I[a] via erlang(stages = %d, rate = sigma)\n\
    \  death_E[a in age]   : E[a] --> @ mu * E[a]\n\
    \  aging_E             : E[pre] --> E[sch] @ ar * E[pre]\n\
     %s\
     }\n\
     init { S[pre] = 990  S[sch] = 900  E[pre] = 5  E[sch] = 5 }\n\
     %s\
     simulate { from = 0 'days  to = 160 'days }\n"
    extra_params stages extra obs

(* The same model with the three stages written out: what a modeller writes
   without `via`, using the names the lowering generates. *)
let age_manual_src =
  "time_unit = 'days\n\
   dimensions { age = [pre, sch]  __latency_stage = [s1, s2, s3] }\n\
   compartments { S, E, I, T }\n\
   stratify(by = age)\n\
   stratify(by = __latency_stage, only = [E])\n\
   parameters {\n\
  \  beta  : rate in [0.001, 2.0]\n\
  \  sigma : rate in [0.01, 1.0]\n\
  \  mu    : rate in [0.0, 1.0]\n\
  \  ar    : rate in [0.0, 1.0]\n\
  \  cmax  : rate in [0.0, 50.0]\n\
  \  K     : count in [1, 100]\n\
  \  k_nb  : positive in [0.1, 100]\n\
   }\n\
   transitions {\n\
  \  infection[a in age] : S[a] --> E[a, s1] @ beta * S[a]\n\
  \  latency_stage[a in age, (st, st_next) in consecutive(__latency_stage)]\n\
  \    : E[a, st] --> E[a, st_next] @ 3 * sigma * E[a, st]\n\
  \  latency[a in age]   : E[a, s3] --> I[a] @ 3 * sigma * E[a, s3]\n\
  \  death_E[a in age, st in __latency_stage] : E[a, st] --> @ mu * E[a, st]\n\
  \  aging_E[st in __latency_stage] : E[pre, st] --> E[sch, st] @ ar * E[pre, st]\n\
   }\n\
   init { S[pre] = 990  S[sch] = 900  E[pre, s1] = 5  E[sch, s1] = 5 }\n\
   simulate { from = 0 'days  to = 160 'days }\n"

(* An unstratified latent compartment: the residence reference is bare `E`. *)
let flat_src ?(compartments = "S, E, I, T, D") ?(extra = "") ?(obs = "") ~stages () =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { %s }\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  gamma : rate in [0.01, 1.0]\n\
    \  mu    : rate in [0.0, 1.0]\n\
    \  cmax  : rate in [0.0, 50.0]\n\
    \  K     : count in [1, 100]\n\
    \  k_nb  : positive in [0.1, 100]\n\
     }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = %d, rate = sigma)\n\
     %s\
     }\n\
     init { S = 990  E = 5 }\n\
     %s\
     simulate { from = 0 'days  to = 160 'days }\n"
    compartments stages extra obs

(* ── Manual equivalence ──────────────────────────────────────────────────── *)

(* The `via` model and the hand-written per-stage model compile to the same
   transitions, in the same order, with the same names, stoichiometry, rates
   and metadata, and to the same compartments and initial state. *)
let test_linear_exit_equals_manual () =
  let via = compile_ok (age_src ~stages:3 ()) in
  let man = compile_ok age_manual_src in
  let json_tr (m : Ir.model) =
    List.map (fun t -> Yojson.Safe.to_string (Serde.transition_to_json t)) m.Ir.transitions in
  Alcotest.(check (list string)) "transition names, in order"
    (transition_names man) (transition_names via);
  Alcotest.(check (list string)) "transitions, serialized" (json_tr man) (json_tr via);
  Alcotest.(check (list string)) "compartments"
    (List.map (fun (c : Ir.compartment) -> c.Ir.name) man.Ir.compartments)
    (List.map (fun (c : Ir.compartment) -> c.Ir.name) via.Ir.compartments);
  Alcotest.(check bool) "initial conditions" true
    (man.Ir.initial_conditions = via.Ir.initial_conditions)

(* The linear rate is the written product with the residence reference
   replaced by the stage cell: no quotient, no conditional. *)
let test_linear_rate_is_per_stage_product () =
  let m = compile_ok (age_src ~stages:3 ()) in
  List.iter (fun (age, stage) ->
    let cell = Printf.sprintf "E_%s_%s" age stage in
    let t = transition m (Printf.sprintf "death_E_%s_%s" age stage) in
    Alcotest.(check bool) (t.Ir.name ^ ": rate is mu * " ^ cell) true
      (t.Ir.rate = Ir.BinOp { op = Ir.Mul; left = Ir.Param "mu"; right = Ir.Pop cell });
    Alcotest.(check (list (pair string int))) (t.Ir.name ^ ": drains its own stage")
      [ (cell, -1) ] (stoich t))
    [ ("pre", "s1"); ("pre", "s2"); ("pre", "s3");
      ("sch", "s1"); ("sch", "s2"); ("sch", "s3") ]

(* A quotient whose divisor does not mention the staged compartment is a
   product for the linear rule: the stage rate is the written quotient with the
   residence reference replaced by the stage cell, as a hand-written per-stage
   model has it. A divisor that does mention it takes the guarded quotient. *)
let test_quotient_by_a_source_free_divisor_is_linear () =
  let m = compile_ok (age_src ~stages:3
    ~extra:"  scaled[a in age] : E[a] --> T[a] @ mu * E[a] / K\n\
           \  nested[a in age] : E[a] --> T[a] @ mu * (E[a] / K)\n\
           \  satur[a in age]  : E[a] --> T[a] @ cmax * E[a] / (K + E[a])\n" ()) in
  List.iter (fun (age, stage) ->
    let cell = Printf.sprintf "E_%s_%s" age stage in
    let scaled = transition m (Printf.sprintf "scaled_%s_%s" age stage) in
    Alcotest.(check bool) (scaled.Ir.name ^ ": rate is mu * " ^ cell ^ " / K") true
      (scaled.Ir.rate
       = Ir.BinOp { op = Ir.Div;
                    left = Ir.BinOp { op = Ir.Mul; left = Ir.Param "mu"; right = Ir.Pop cell };
                    right = Ir.Param "K" });
    let nested = transition m (Printf.sprintf "nested_%s_%s" age stage) in
    Alcotest.(check bool) (nested.Ir.name ^ ": rate is mu * (" ^ cell ^ " / K)") true
      (nested.Ir.rate
       = Ir.BinOp { op = Ir.Mul; left = Ir.Param "mu";
                    right = Ir.BinOp { op = Ir.Div; left = Ir.Pop cell; right = Ir.Param "K" } });
    let satur = transition m (Printf.sprintf "satur_%s_%s" age stage) in
    Alcotest.(check bool) (satur.Ir.name ^ ": a divisor reading E takes the guarded quotient")
      true (match satur.Ir.rate with Ir.Cond _ -> true | _ -> false))
    [ ("pre", "s1"); ("pre", "s2"); ("pre", "s3");
      ("sch", "s1"); ("sch", "s2"); ("sch", "s3") ]

(* ── Residence mapping ───────────────────────────────────────────────────── *)

(* A move between strata of the same staged compartment keeps the stage. *)
let test_same_compartment_move_keeps_the_stage () =
  let m = compile_ok (age_src ~stages:3 ()) in
  List.iter (fun s ->
    let t = transition m ("aging_E_" ^ s) in
    Alcotest.(check (list (pair string int))) (t.Ir.name ^ ": stage preserved")
      [ ("E_pre_" ^ s, -1); ("E_sch_" ^ s, 1) ] (stoich t);
    Alcotest.(check bool) (t.Ir.name ^ ": rate is ar * E_pre_" ^ s) true
      (t.Ir.rate = Ir.BinOp { op = Ir.Mul; left = Ir.Param "ar";
                              right = Ir.Pop ("E_pre_" ^ s) }))
    [ "s1"; "s2"; "s3" ]

(* An exit to an unstaged compartment draws from every stage and lands in the
   one destination cell. *)
let test_exit_to_unstaged_destination () =
  let m = compile_ok (flat_src ~stages:3 ~extra:"  die : E --> D @ mu * E\n" ()) in
  List.iter (fun s ->
    Alcotest.(check (list (pair string int))) ("die_" ^ s)
      [ ("D", 1); ("E_" ^ s, -1) ] (stoich (transition m ("die_" ^ s))))
    [ "s1"; "s2"; "s3" ]

let two_staged_src =
  "time_unit = 'days\n\
   compartments { S, E, I, R }\n\
   parameters { beta : rate  sigma : rate  gamma : rate  eps : rate }\n\
   transitions {\n\
  \  infection : S --> E @ beta * S\n\
  \  latency   : E --> I via erlang(stages = 3, rate = sigma)\n\
  \  recovery  : I --> R via erlang(stages = 2, rate = gamma)\n\
  \  early     : E --> I @ eps * E\n\
   }\n\
   init { S = 990  E = 5 }\n"

(* An exit into a different staged compartment enters its first stage, and
   says so: the destination dwell restarts. The `via` transition's own exit
   into the same compartment does not warn. *)
let test_exit_to_other_staged_compartment_restarts_and_warns () =
  let m = compile_ok two_staged_src in
  List.iter (fun s ->
    Alcotest.(check (list (pair string int))) ("early_" ^ s ^ " enters I_s1")
      [ ("E_" ^ s, -1); ("I_s1", 1) ] (stoich (transition m ("early_" ^ s))))
    [ "s1"; "s2"; "s3" ];
  match warnings_of ~code:"W107" two_staged_src with
  | [ w ] ->
    List.iter (fun needle ->
      Alcotest.(check bool) ("W107 mentions " ^ needle) true
        (contains ~needle (w.message ^ " " ^ Option.value w.hint ~default:"")))
      [ "early"; "restarts 'I' at stage 1"; "stage(...)" ]
  | ws -> Alcotest.failf "expected one W107, got %d" (List.length ws)

let test_no_restart_warning_without_a_second_staged_compartment () =
  Alcotest.(check int) "no W107" 0
    (List.length (warnings_of ~code:"W107" (age_src ~stages:3 ())))

(* ── Propensity preservation ─────────────────────────────────────────────── *)

(* Treatment that saturates: the hazard per person falls as the compartment
   fills. *)
let treat = "  treat[a in age] : E[a] --> T[a] @ cmax * E[a] * K / (K + E[a])\n"

(* A rate that is not a product with the residence reference as one factor is
   allocated to the stages in proportion to occupancy. The stage rates sum to
   the written rate at every state, each stage faces the same per-person
   hazard, and an empty compartment gives zero rather than 0/0. *)
let test_nonlinear_exit_preserves_the_propensity () =
  let m = compile_ok (age_src ~stages:3 ~extra:treat ()) in
  let params = [ ("cmax", 7.0); ("K", 20.0) ] in
  let stages = [ "s1"; "s2"; "s3" ] in
  List.iter (fun (e_pre : float list) ->
    let pops =
      List.map2 (fun s v -> ("E_pre_" ^ s, v)) stages e_pre
      @ List.map (fun s -> ("E_sch_" ^ s, 1000.0)) stages in
    let total = List.fold_left ( +. ) 0.0 e_pre in
    let written = 7.0 *. total *. 20.0 /. (20.0 +. total) in
    let rates =
      List.map (fun s ->
        eval ~pops ~params m (transition m ("treat_pre_" ^ s)).Ir.rate) stages in
    List.iter (fun r ->
      Alcotest.(check bool) "stage rate is a finite number" true (Float.is_finite r))
      rates;
    Alcotest.(check (float 1e-12)) "stage rates sum to the written rate"
      written (List.fold_left ( +. ) 0.0 rates);
    List.iter2 (fun r n ->
      Alcotest.(check (float 1e-12)) "stage rate is proportional to occupancy"
        (if total > 0.0 then written *. n /. total else 0.0) r)
      rates e_pre)
    [ [ 3.0; 1.0; 1.0 ]; [ 0.0; 12.0; 0.0 ]; [ 40.0; 7.0; 250.0 ]; [ 0.0; 0.0; 0.0 ] ]

(* A product in which the staged compartment also appears outside the one
   factor is not linear in the proposal's sense, and takes the quotient. *)
let test_product_with_a_second_reference_takes_the_quotient () =
  let m = compile_ok (age_src ~stages:3
    ~extra:"  crowd[a in age] : E[a] --> T[a] @ mu * E[a] * E[a]\n" ()) in
  let pops = [ ("E_pre_s1", 3.0); ("E_pre_s2", 1.0); ("E_pre_s3", 2.0) ] in
  let rates =
    List.map (fun s ->
      eval ~pops ~params:[ ("mu", 0.5) ] m (transition m ("crowd_pre_" ^ s)).Ir.rate)
      [ "s1"; "s2"; "s3" ] in
  Alcotest.(check (list (float 1e-12))) "0.5 * 6 * 6 split 3 : 1 : 2"
    [ 9.0; 3.0; 6.0 ] rates

(* ── Survival fraction ───────────────────────────────────────────────────── *)

(* With a constant per-person hazard mu beside an Erlang-k dwell of stage rate
   k * sigma, a person in any stage advances with probability
   k*sigma / (k*sigma + mu), so the fraction completing the dwell is that to
   the power k. Read from the compiled rates at one person per stage. *)
let test_survival_fraction () =
  List.iter (fun k ->
    let m = compile_ok (flat_src ~stages:k ~extra:"  die : E --> D @ mu * E\n" ()) in
    let sigma = 0.25 and mu = 0.1 in
    let params = [ ("sigma", sigma); ("mu", mu) ] in
    let stages = List.init k (fun i -> Printf.sprintf "s%d" (i + 1)) in
    let pops = List.map (fun s -> ("E_" ^ s, 1.0)) stages in
    let advance s =
      List.find (fun (t : Ir.transition) ->
        List.mem ("E_" ^ s, -1) t.Ir.stoichiometry
        && not (List.mem ("D", 1) t.Ir.stoichiometry)) m.Ir.transitions in
    let survive =
      List.fold_left (fun acc s ->
        let go = eval ~pops ~params m (advance s).Ir.rate in
        let die = eval ~pops ~params m (transition m ("die_" ^ s)).Ir.rate in
        acc *. go /. (go +. die)) 1.0 stages in
    let kf = float_of_int k in
    Alcotest.(check (float 1e-12)) (Printf.sprintf "k = %d" k)
      ((kf *. sigma /. (kf *. sigma +. mu)) ** kf) survive)
    [ 2; 3; 8 ]

(* ── Flows ───────────────────────────────────────────────────────────────── *)

let obs_block ?(index = "") ?(cols = "") projected =
  Printf.sprintf
    "observations {\n\
    \  deaths%s {\n\
    \    columns       { time : time, %sdeaths : count }\n\
    \    covers        = closing_at(time, 7 'days)\n\
    \    projected     = %s\n\
    \    emit_schedule = every 7 'days\n\
    \    deaths ~ neg_binomial(mean = projected, r = k_nb)\n\
    \  }\n\
     }\n" index cols projected

let projections (m : Ir.model) =
  List.map (fun (o : Ir.observation_model) -> o.Ir.projection) m.Ir.observations

let flow_sum names = Ir.CumulativeFlowSum names

(* A flow reference to the transition as written sums its stage copies. *)
let test_indexed_flow_sums_its_stage_copies () =
  let m = compile_ok (age_src ~stages:3
    ~obs:(obs_block ~index:"[a in age]" ~cols:"age : dim, " "incidence(death_E[a])") ()) in
  Alcotest.(check bool) "one row per age, each the sum of its three stage flows" true
    (projections m
     = [ flow_sum [ "death_E_pre_s1"; "death_E_pre_s2"; "death_E_pre_s3" ];
         flow_sum [ "death_E_sch_s1"; "death_E_sch_s2"; "death_E_sch_s3" ] ])

let test_cell_flow_sums_its_stage_copies () =
  let m = compile_ok (age_src ~stages:3 ~obs:(obs_block "incidence(death_E[sch])") ()) in
  Alcotest.(check bool) "incidence(death_E[sch])" true
    (projections m
     = [ flow_sum [ "death_E_sch_s1"; "death_E_sch_s2"; "death_E_sch_s3" ] ])

(* The explicit pooled form sums the stage copies of every stratum. *)
let test_explicit_sum_of_flows_includes_every_stage_copy () =
  let m = compile_ok (age_src ~stages:3
    ~obs:(obs_block "sum(a in age, incidence(death_E[a]))") ()) in
  Alcotest.(check bool) "sum(a in age, incidence(death_E[a]))" true
    (projections m
     = [ flow_sum [ "death_E_pre_s1"; "death_E_pre_s2"; "death_E_pre_s3";
                    "death_E_sch_s1"; "death_E_sch_s2"; "death_E_sch_s3" ] ])

(* The stage axis is not a population stratum: a bare flow reference to a
   transition whose only expansion is the stages is not a silent pooling. *)
let test_bare_flow_over_stages_only_is_not_e280 () =
  let m = compile_ok (flat_src ~stages:3 ~extra:"  die : E --> D @ mu * E\n"
    ~obs:(obs_block "incidence(die)") ()) in
  Alcotest.(check bool) "incidence(die)" true
    (projections m = [ flow_sum [ "die_s1"; "die_s2"; "die_s3" ] ])

let test_flow_ratio_reads_stage_copies () =
  let m = compile_ok (flat_src ~stages:3 ~extra:"  die : E --> D @ mu * E\n"
    ~obs:"observations {\n\
         \  frac {\n\
         \    columns       { time : time, frac : probability }\n\
         \    covers        = closing_at(time, 7 'days)\n\
         \    projected     = incidence(die) / (incidence(die) + incidence(latency))\n\
         \    emit_schedule = every 7 'days\n\
         \    frac ~ beta(mean = projected, concentration = k_nb)\n\
         \  }\n\
          }\n" ()) in
  Alcotest.(check bool) "ratio of flows" true
    (projections m
     = [ Ir.FlowRatio { numerator = [ "die_s1"; "die_s2"; "die_s3" ];
                        denominator = [ "die_s1"; "die_s2"; "die_s3"; "latency" ] } ])

(* A population stratum is still never pooled silently, and the diagnostic
   names the declared dimension only. *)
let test_bare_flow_over_a_population_stratum_is_still_e280 () =
  let src = age_src ~stages:3 ~obs:(obs_block "incidence(death_E)") () in
  expect_error ~code:"E280"
    ~mentions:[ "all 2 strata"; "sum(a in age, incidence(death_E[a]))" ] src;
  match detail src with
  | Ok _ -> Alcotest.fail "expected E280"
  | Error e ->
    Alcotest.(check bool) "the diagnostic does not name the stage dimension" false
      (contains ~needle:"__latency_stage" e)

(* At one stage the flow is the single transition, as for any second exit. *)
let test_flow_at_one_stage_is_the_single_transition () =
  let m = compile_ok (age_src ~stages:1 ~obs:(obs_block "incidence(death_E[sch])") ()) in
  Alcotest.(check bool) "incidence(death_E[sch]) at stages = 1" true
    (projections m = [ Ir.CumulativeFlow "death_E_sch" ])

(* ── stages = 1 ──────────────────────────────────────────────────────────── *)

(* At one stage the source is unstaged: the model equals the one that writes
   the exponential exit directly, and the second exit is not rewritten. *)
let test_one_stage_is_an_ordinary_second_exit () =
  let via = compile_ok (age_src ~stages:1 ~extra:treat ()) in
  let plain =
    let src = age_src ~stages:1 ~extra:treat () in
    let via_line = "E[a] --> I[a] via erlang(stages = 1, rate = sigma)" in
    let i =
      let rec find i =
        if String.sub src i (String.length via_line) = via_line then i else find (i + 1) in
      find 0 in
    compile_ok (String.sub src 0 i ^ "E[a] --> I[a] @ sigma * E[a]"
                ^ String.sub src (i + String.length via_line)
                    (String.length src - i - String.length via_line)) in
  Alcotest.(check bool) "transitions equal the plain exponential model" true
    (via.Ir.transitions = plain.Ir.transitions);
  Alcotest.(check (list string)) "no stage copies"
    [ "infection_pre"; "infection_sch"; "latency_pre"; "latency_sch";
      "death_E_pre"; "death_E_sch"; "aging_E"; "treat_pre"; "treat_sch" ]
    (transition_names via)

(* The same text compiles at one stage and at three, with the dimension
   checker on. *)
let test_same_text_at_one_and_three_stages () =
  let obs = obs_block ~index:"[a in age]" ~cols:"age : dim, " "incidence(death_E[a])" in
  List.iter (fun stages ->
    ignore (compile_ok ~dim_check:true (age_src ~stages ~extra:treat ~obs ())))
    [ 1; 3 ]

(* ── Families over compartments ──────────────────────────────────────────── *)

(* A family bound over the compartment set, beside a staged `E`. *)
let family_src ?(obs = "") ~stages family =
  Printf.sprintf
    "time_unit = 'days\n\
     dimensions { age = [pre, sch] }\n\
     compartments { S, E, I }\n\
     stratify(by = age)\n\
     parameters {\n\
    \  beta  : rate in [0.001, 2.0]\n\
    \  sigma : rate in [0.01, 1.0]\n\
    \  mu    : rate in [0.0, 1.0]\n\
    \  ar    : rate in [0.0, 1.0]\n\
    \  K     : count in [1, 100]\n\
    \  k_nb  : positive in [0.1, 100]\n\
     }\n\
     transitions {\n\
    \  infection[a in age] : S[a] --> E[a] @ beta * S[a]\n\
    \  latency[a in age]   : E[a] --> I[a] via erlang(stages = %d, rate = sigma)\n\
     %s\
     }\n\
     init { S[pre] = 990  S[sch] = 900  E[pre] = 5  E[sch] = 5 }\n\
     %s\
     simulate { from = 0 'days  to = 160 'days }\n"
    stages family obs

let bg = "  bg[a in age, c in compartments] : c[a] --> @ mu * c[a]\n"

(* The member of the family for the staged compartment is a competing exit,
   one copy per stage with the linear per-stage rate; the other members are
   ordinary transitions. *)
let test_family_over_compartments_expands_its_staged_member () =
  let m = compile_ok (family_src ~stages:3 bg) in
  let mu_times cell = Ir.BinOp { op = Ir.Mul; left = Ir.Param "mu"; right = Ir.Pop cell } in
  List.iter (fun age ->
    List.iter (fun s ->
      let cell = Printf.sprintf "E_%s_%s" age s in
      let t = transition m (Printf.sprintf "bg_%s_E_%s" age s) in
      Alcotest.(check (list (pair string int))) (t.Ir.name ^ ": drains its stage")
        [ (cell, -1) ] (stoich t);
      Alcotest.(check bool) (t.Ir.name ^ ": rate is mu * " ^ cell) true
        (t.Ir.rate = mu_times cell))
      [ "s1"; "s2"; "s3" ];
    List.iter (fun c ->
      let cell = Printf.sprintf "%s_%s" c age in
      let t = transition m (Printf.sprintf "bg_%s_%s" age c) in
      Alcotest.(check (list (pair string int))) (t.Ir.name ^ ": ordinary exit")
        [ (cell, -1) ] (stoich t);
      Alcotest.(check bool) (t.Ir.name ^ ": rate is mu * " ^ cell) true
        (t.Ir.rate = mu_times cell))
      [ "S"; "I" ])
    [ "pre"; "sch" ];
  Alcotest.(check bool) "no unexpanded member for E" false
    (List.mem "bg_pre_E" (transition_names m))

(* Ageing written once for every compartment keeps the stage of the staged one. *)
let test_family_over_compartments_keeps_the_stage_within_a_compartment () =
  let m = compile_ok (family_src ~stages:3
    "  ageing[(a, an) in consecutive(age), c in compartments] : c[a] --> c[an] @ ar * c[a]\n") in
  List.iter (fun s ->
    Alcotest.(check (list (pair string int))) ("ageing_pre_E_" ^ s)
      [ ("E_pre_" ^ s, -1); ("E_sch_" ^ s, 1) ]
      (stoich (transition m ("ageing_pre_E_" ^ s))))
    [ "s1"; "s2"; "s3" ];
  Alcotest.(check (list (pair string int))) "ageing_pre_S"
    [ ("S_pre", -1); ("S_sch", 1) ] (stoich (transition m "ageing_pre_S"))

(* A nonlinear rate through the binder is allocated in proportion to
   occupancy, and sums to the written rate, an empty compartment included. *)
let test_family_over_compartments_nonlinear_rate () =
  let m = compile_ok (family_src ~stages:3
    "  bg[a in age, c in compartments] : c[a] --> @ mu * c[a] * K / (K + c[a])\n") in
  let stages = [ "s1"; "s2"; "s3" ] in
  List.iter (fun (e_pre : float list) ->
    let pops = List.map2 (fun s v -> ("E_pre_" ^ s, v)) stages e_pre in
    let total = List.fold_left ( +. ) 0.0 e_pre in
    let written = 0.5 *. total *. 20.0 /. (20.0 +. total) in
    let rates = List.map (fun s ->
      eval ~pops ~params:[ ("mu", 0.5); ("K", 20.0) ] m
        (transition m ("bg_pre_E_" ^ s)).Ir.rate) stages in
    Alcotest.(check (float 1e-12)) "stage rates sum to the written rate"
      written (List.fold_left ( +. ) 0.0 rates);
    List.iter2 (fun r n ->
      Alcotest.(check (float 1e-12)) "stage rate is proportional to occupancy"
        (if total > 0.0 then written *. n /. total else 0.0) r) rates e_pre)
    [ [ 3.0; 1.0; 1.0 ]; [ 0.0; 0.0; 0.0 ] ]

(* A flow reference to the staged member reads its stage copies; one to an
   unstaged member reads the single transition. *)
let test_family_over_compartments_flow_reference () =
  let obs c = obs_block ~index:"[a in age]" ~cols:"age : dim, "
      (Printf.sprintf "incidence(bg[a, %s])" c) in
  Alcotest.(check bool) "incidence(bg[a, E]) sums the stage copies" true
    (projections (compile_ok (family_src ~stages:3 ~obs:(obs "E") bg))
     = [ flow_sum [ "bg_pre_E_s1"; "bg_pre_E_s2"; "bg_pre_E_s3" ];
         flow_sum [ "bg_sch_E_s1"; "bg_sch_E_s2"; "bg_sch_E_s3" ] ]);
  Alcotest.(check bool) "incidence(bg[a, S]) is the single transition" true
    (projections (compile_ok (family_src ~stages:3 ~obs:(obs "S") bg))
     = [ Ir.CumulativeFlow "bg_pre_S"; Ir.CumulativeFlow "bg_sch_S" ])

(* The same family compiles at one stage, where nothing is staged. *)
let test_family_over_compartments_at_one_stage () =
  let m = compile_ok (family_src ~stages:1 bg) in
  Alcotest.(check (list (pair string int))) "bg_pre_E is an ordinary exit"
    [ ("E_pre", -1) ] (stoich (transition m "bg_pre_E"))

(* The refusals of a written competing exit apply to the staged member. *)
let test_family_over_compartments_overdispersed_is_refused () =
  expect_error ~code:"E361" ~mentions:[ "bg"; "`c = E`"; "where c != E" ]
    (family_src ~stages:3
       "  bg[a in age, c in compartments] : c[a] --> @ overdispersed(mu * c[a], k_nb)\n")

(* ── Refusals ────────────────────────────────────────────────────────────── *)

let test_overdispersed_exit_is_refused () =
  let src stages = flat_src ~stages
      ~extra:"  die : E --> D @ overdispersed(mu * E, 0.1)\n" () in
  expect_error ~code:"E361" ~mentions:[ "die"; "overdispersed"; "'E'" ] (src 3);
  refuse_error ~code:"E361" (src 1)

let test_source_stoichiometry_above_one_is_refused () =
  expect_error ~code:"E362" ~mentions:[ "pair"; "'E'" ]
    (flat_src ~stages:3 ~extra:"  pair : E + E --> D @ mu * E\n" ())

let test_two_staged_sources_are_refused () =
  expect_error ~code:"E363" ~mentions:[ "merge"; "'E'"; "'I'" ]
    "time_unit = 'days\n\
     compartments { S, E, I, R }\n\
     parameters { beta : rate  sigma : rate  gamma : rate  mu : rate }\n\
     transitions {\n\
    \  infection : S --> E @ beta * S\n\
    \  latency   : E --> I via erlang(stages = 3, rate = sigma)\n\
    \  recovery  : I --> R via erlang(stages = 2, rate = gamma)\n\
    \  merge     : E + I --> R @ mu * E\n\
     }\n\
     init { S = 990  E = 5 }\n"

let test_exit_with_an_unstaged_second_source_is_refused () =
  expect_error ~code:"E365" ~mentions:[ "meet"; "'E'"; "'S'" ]
    (flat_src ~stages:3 ~extra:"  meet : E + S --> D @ mu * E\n" ())

let hyper_src extra =
  "time_unit = 'days\n\
   compartments { S, I, R, D }\n\
   parameters { beta : rate  g1 : rate  g2 : rate  mu : rate  w : probability }\n\
   transitions {\n\
  \  infection : S --> I @ beta * S\n\
  \  outcome   : I --> R via hyper_erlang(\n\
  \    branch(label = fast, stages = 2, rate = g1, weight = w),\n\
  \    branch(label = slow, stages = 3, rate = g2))\n"
  ^ extra ^
  "}\n\
   init { S = 990  I = 5 }\n"

let test_exit_from_a_hyper_erlang_compartment_is_refused () =
  ignore (compile_ok (hyper_src ""));
  expect_error ~code:"E364" ~mentions:[ "die"; "'I'"; "hyper_erlang" ]
    (hyper_src "  die : I --> D @ mu * I\n")

(* A second `via` on one source keeps E246, and an ordinary second exit no
   longer reports it. *)
let test_second_via_is_e246 () =
  expect_error ~code:"E246" ~mentions:[ "second"; "'E'"; "latency" ]
    (flat_src ~stages:3
       ~extra:"  second : E --> T via erlang(stages = 2, rate = gamma)\n" ());
  refuse_error ~code:"E246"
    (flat_src ~stages:3 ~extra:"  die : E --> D @ mu * E\n" ())

(* A refused exit reports its own diagnostic and nothing that follows from a
   half-applied staging. *)
let test_a_refusal_reports_only_itself () =
  match detail (flat_src ~stages:3 ~extra:"  pair : E + E --> D @ mu * E\n" ()) with
  | Ok _ -> Alcotest.fail "expected E362"
  | Error e ->
    List.iter (fun code ->
      Alcotest.(check bool) ("no " ^ code) false (contains ~needle:code e))
      [ "E100"; "E246"; "E287"; "E503" ]

let () =
  Alcotest.run "competing_exits"
    [ ("manual equivalence",
       [ Alcotest.test_case "linear exit and ageing equal the hand-written model" `Quick
           test_linear_exit_equals_manual;
         Alcotest.test_case "linear rate is the per-stage product" `Quick
           test_linear_rate_is_per_stage_product;
         Alcotest.test_case "quotient by a source-free divisor is linear" `Quick
           test_quotient_by_a_source_free_divisor_is_linear ]);
      ("residence mapping",
       [ Alcotest.test_case "same-compartment move keeps the stage" `Quick
           test_same_compartment_move_keeps_the_stage;
         Alcotest.test_case "exit to an unstaged destination" `Quick
           test_exit_to_unstaged_destination;
         Alcotest.test_case "exit to another staged compartment restarts, W107" `Quick
           test_exit_to_other_staged_compartment_restarts_and_warns;
         Alcotest.test_case "no W107 without a second staged compartment" `Quick
           test_no_restart_warning_without_a_second_staged_compartment ]);
      ("propensity",
       [ Alcotest.test_case "nonlinear exit: stage rates sum to the written rate" `Quick
           test_nonlinear_exit_preserves_the_propensity;
         Alcotest.test_case "product with a second reference takes the quotient" `Quick
           test_product_with_a_second_reference_takes_the_quotient;
         Alcotest.test_case "survival fraction (k sigma / (k sigma + mu))^k" `Quick
           test_survival_fraction ]);
      ("flows",
       [ Alcotest.test_case "incidence(death_E[a]) sums stage copies" `Quick
           test_indexed_flow_sums_its_stage_copies;
         Alcotest.test_case "incidence(death_E[sch]) sums stage copies" `Quick
           test_cell_flow_sums_its_stage_copies;
         Alcotest.test_case "explicit sum over age includes every stage copy" `Quick
           test_explicit_sum_of_flows_includes_every_stage_copy;
         Alcotest.test_case "bare flow over stages only is not E280" `Quick
           test_bare_flow_over_stages_only_is_not_e280;
         Alcotest.test_case "ratio of flows reads stage copies" `Quick
           test_flow_ratio_reads_stage_copies;
         Alcotest.test_case "bare flow over a population stratum is still E280" `Quick
           test_bare_flow_over_a_population_stratum_is_still_e280;
         Alcotest.test_case "flow at stages = 1 is the single transition" `Quick
           test_flow_at_one_stage_is_the_single_transition ]);
      ("stages = 1",
       [ Alcotest.test_case "an ordinary second exit, nothing rewritten" `Quick
           test_one_stage_is_an_ordinary_second_exit;
         Alcotest.test_case "same text at stages = 1 and stages = 3" `Quick
           test_same_text_at_one_and_three_stages ]);
      ("families over compartments",
       [ Alcotest.test_case "staged member is a competing exit" `Quick
           test_family_over_compartments_expands_its_staged_member;
         Alcotest.test_case "a move within the compartment keeps the stage" `Quick
           test_family_over_compartments_keeps_the_stage_within_a_compartment;
         Alcotest.test_case "nonlinear rate sums to the written rate" `Quick
           test_family_over_compartments_nonlinear_rate;
         Alcotest.test_case "flow reference reads the stage copies" `Quick
           test_family_over_compartments_flow_reference;
         Alcotest.test_case "same text at stages = 1" `Quick
           test_family_over_compartments_at_one_stage;
         Alcotest.test_case "overdispersed staged member → E361" `Quick
           test_family_over_compartments_overdispersed_is_refused ]);
      ("refusals",
       [ Alcotest.test_case "overdispersed exit → E361" `Quick
           test_overdispersed_exit_is_refused;
         Alcotest.test_case "source stoichiometry above 1 → E362" `Quick
           test_source_stoichiometry_above_one_is_refused;
         Alcotest.test_case "two staged sources → E363" `Quick
           test_two_staged_sources_are_refused;
         Alcotest.test_case "exit from a hyper_erlang compartment → E364" `Quick
           test_exit_from_a_hyper_erlang_compartment_is_refused;
         Alcotest.test_case "unstaged second source → E365" `Quick
           test_exit_with_an_unstaged_second_source_is_refused;
         Alcotest.test_case "second via on one source → E246" `Quick
           test_second_via_is_e246;
         Alcotest.test_case "a refusal reports only itself" `Quick
           test_a_refusal_reports_only_itself ]) ]
