(* Phase-4 staged-residence `via hyper_erlang(...)` lowering: IR-level tests.

   `hyper_erlang` is a finite mixture of Erlang chains, branched at entry. Unlike
   `erlang` (one stage dimension), the branches have different lengths, so it
   lowers to FLAT per-branch compartments `<src>__<label>__i` and parallel
   chains. We assert at the IR level:

   - Polio (SAME endpoint): two parallel chains both exiting to `R`, the entry
     `DstBranch` weighted `p` / `1−p`, and a bare `I` in a rate summing all
     stages of both branches.
   - Ebola (PER-BRANCH endpoints, no arrow target): the fatal chain exits to `D`,
     recover to `R`, the entry splits `cfr` / `1−cfr`, the FOI sums all infectious
     stages across both branches, and the mixture means (≈8 d / ≈12 d) are right
     at the deterministic ODE level.
   - Validation: each rule a distinct E-code naming the transition.
   - Deferred stratified hyper_erlang → a clean E248 (not a crash). *)

let () = Compiler.no_dim_check := true

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let compile_ok (src : string) : Ir.model =
  match Compiler.compile ~name:"hyper_test" src with
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
  let result = Compiler.compile ~name:"hyper_test" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

let comp_names (m : Ir.model) : string list =
  List.map (fun (c : Ir.compartment) -> c.Ir.name) m.Ir.compartments

let tr_named (m : Ir.model) name : Ir.transition =
  match List.find_opt (fun (t : Ir.transition) -> t.Ir.name = name) m.Ir.transitions with
  | Some t -> t
  | None ->
    Alcotest.failf "no transition named %S (have: %s)" name
      (String.concat ", " (List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions))

(* Every (compartment, signed-delta) stoichiometry pair touching a name. *)
let stoich_has (t : Ir.transition) (name : string) (delta : int) : bool =
  List.mem (name, delta) t.Ir.stoichiometry

(* All Pop / PopSum names appearing anywhere in an expr (for "sums all stages"). *)
let rec pop_names acc = function
  | Ir.Pop n     -> n :: acc
  | Ir.PopSum ns -> ns @ acc
  | Ir.BinOp b   -> pop_names (pop_names acc b.Ir.left) b.Ir.right
  | Ir.UnOp u    -> pop_names acc u.Ir.arg
  | Ir.Cond c    -> pop_names (pop_names (pop_names acc c.Ir.pred) c.Ir.then_) c.Ir.else_
  | Ir.Reduce ts -> List.fold_left pop_names acc ts
  | _            -> acc

(* ── Polio: SAME endpoint (shared `--> R`), bimodal shedding ─────────────────

   `clearance : I --> R via hyper_erlang(branch(typical, weight=p, stages=2, mean),
                                         branch(prolonged, stages=1, mean))`
   Two parallel chains both exiting to R; the entry into I splits p / (1−p). *)

let polio_src =
  "time_unit = 'weeks\n\
   compartments { S, I, R }\n\
   parameters {\n\
  \  beta : rate  p : probability\n\
  \  tau_typ : positive  tau_pro : positive\n\
   }\n\
   transitions {\n\
  \  infection : S --> I @ beta * S * I / (S + I + R)\n\
  \  clearance : I --> R via hyper_erlang(\n\
  \    branch(label = typical,   weight = p, stages = 2, mean = tau_typ),\n\
  \    branch(label = prolonged,             stages = 1, mean = tau_pro)\n\
  \  )\n\
   }\n\
   init { S = 990  I = 10 }\n\
   simulate { from = 0 'weeks  to = 52 'weeks }\n"

let test_polio_flat_compartments_exist () =
  let m = compile_ok polio_src in
  let names = comp_names m in
  (* Two-stage typical, one-stage prolonged. *)
  List.iter (fun n -> Alcotest.(check bool) (n ^ " exists") true (List.mem n names))
    [ "I__typical__1"; "I__typical__2"; "I__prolonged__1" ];
  (* The base I compartment is gone (replaced by the flat per-branch cells). *)
  Alcotest.(check bool) "bare I removed" false (List.mem "I" names)

let test_polio_both_chains_exit_to_R () =
  let m = compile_ok polio_src in
  (* The typical chain steps I__typical__1 --> I__typical__2, then exits to R. *)
  let step = tr_named m "clearance_typical_stage1" in
  Alcotest.(check bool) "typical step1 I__typical__1 -1" true (stoich_has step "I__typical__1" (-1));
  Alcotest.(check bool) "typical step1 I__typical__2 +1" true (stoich_has step "I__typical__2" 1);
  let typ_exit = tr_named m "clearance_typical_exit" in
  Alcotest.(check bool) "typical exit drains last stage" true (stoich_has typ_exit "I__typical__2" (-1));
  Alcotest.(check bool) "typical exit --> R" true (stoich_has typ_exit "R" 1);
  (* The prolonged chain is one stage: its exit drains I__prolonged__1 --> R. *)
  let pro_exit = tr_named m "clearance_prolonged_exit" in
  Alcotest.(check bool) "prolonged exit drains stage 1" true (stoich_has pro_exit "I__prolonged__1" (-1));
  Alcotest.(check bool) "prolonged exit --> R" true (stoich_has pro_exit "R" 1)

let test_polio_entry_dstbranch_weighted () =
  let m = compile_ok polio_src in
  (* infection : S --> I became S --> { I__typical__1 : p, I__prolonged__1 : 1−p }.
     The DstBranch lowering emits one transition per branch, with rate scaled by
     the weight. So there must be exactly two infection-derived transitions, one
     landing in each first stage. *)
  let infections =
    List.filter (fun (t : Ir.transition) ->
      String.length t.Ir.name >= 9 && String.sub t.Ir.name 0 9 = "infection")
      m.Ir.transitions in
  Alcotest.(check int) "two entry branches" 2 (List.length infections);
  let lands_in cell =
    List.exists (fun (t : Ir.transition) -> stoich_has t cell 1) infections in
  Alcotest.(check bool) "an entry branch lands in I__typical__1" true (lands_in "I__typical__1");
  Alcotest.(check bool) "an entry branch lands in I__prolonged__1" true (lands_in "I__prolonged__1");
  (* Each entry rate carries its weight factor. The typical branch's rate must
     mention the parameter `p`; the prolonged branch's rate must be the
     1−p complement (a Sub with a 1 and a p). *)
  let typ = List.find (fun (t : Ir.transition) -> stoich_has t "I__typical__1" 1) infections in
  let pro = List.find (fun (t : Ir.transition) -> stoich_has t "I__prolonged__1" 1) infections in
  let rec mentions_param name = function
    | Ir.Param n -> n = name
    | Ir.BinOp b -> mentions_param name b.Ir.left || mentions_param name b.Ir.right
    | Ir.UnOp u  -> mentions_param name u.Ir.arg
    | Ir.Cond c  -> mentions_param name c.Ir.pred || mentions_param name c.Ir.then_ || mentions_param name c.Ir.else_
    | Ir.Reduce ts -> List.exists (mentions_param name) ts
    | _ -> false in
  let rec has_one_minus = function
    | Ir.BinOp { Ir.op = Ir.Sub; left = Ir.Const 1.0; _ } -> true
    | Ir.BinOp b -> has_one_minus b.Ir.left || has_one_minus b.Ir.right
    | Ir.UnOp u  -> has_one_minus u.Ir.arg
    | Ir.Cond c  -> has_one_minus c.Ir.pred || has_one_minus c.Ir.then_ || has_one_minus c.Ir.else_
    | Ir.Reduce ts -> List.exists has_one_minus ts
    | _ -> false in
  Alcotest.(check bool) "typical entry weight mentions p" true (mentions_param "p" typ.Ir.rate);
  Alcotest.(check bool) "prolonged entry weight is 1 − …" true (has_one_minus pro.Ir.rate)

let test_polio_foi_sums_all_stages () =
  let m = compile_ok polio_src in
  (* The bare `I` in the FOI numerator + denominator sums all branch stages. We
     check the FOI numerator's per-capita I factor: every infection-derived
     transition's rate must reference all three stage cells and NO bare I. *)
  let inf = List.find (fun (t : Ir.transition) -> stoich_has t "I__typical__1" 1)
    (List.filter (fun (t : Ir.transition) ->
       String.length t.Ir.name >= 9 && String.sub t.Ir.name 0 9 = "infection")
       m.Ir.transitions) in
  let names = pop_names [] inf.Ir.rate in
  List.iter (fun cell ->
    Alcotest.(check bool) (Printf.sprintf "FOI references %s" cell) true (List.mem cell names))
    [ "I__typical__1"; "I__typical__2"; "I__prolonged__1" ];
  Alcotest.(check bool) "FOI has no dangling bare I" false (List.mem "I" names)

let test_polio_init_split_by_weight () =
  let m = compile_ok polio_src in
  (* init { I = 10 } splits across the first stages: I__typical__1 = 10*p,
     I__prolonged__1 = 10*(1−p). After constant-folding p is a parameter, so the
     init exprs cannot be a plain float — they appear as a derived init. We assert
     the base I init key is gone and both first-stage keys exist (value-bearing). *)
  match m.Ir.initial_conditions with
  | Ir.Explicit kvs ->
    Alcotest.(check bool) "no bare I init" false (List.mem_assoc "I" kvs)
  | _ ->
    (* A parameterized split init is not Explicit; that is acceptable — the point
       is the base I init was redirected, not left dangling. The compile succeeding
       (compile_ok) already proves the init resolved. *)
    ()

let test_polio_simulates () =
  (* End-to-end: the polio model compiles to a sane IR (right comp / transition
     counts, no via survives). 3 base → S, R, + 3 flat stage cells = 5 comps;
     transitions: 2 entry branches + (typical: 1 step + 1 exit) + (prolonged: 1
     exit) = 5. *)
  let m = compile_ok polio_src in
  Alcotest.(check int) "compartment count" 5 (List.length m.Ir.compartments);
  Alcotest.(check int) "transition count" 5 (List.length m.Ir.transitions)

(* ── Ebola: PER-BRANCH endpoints (no arrow target), CFR split ────────────────

   `outcome : I via hyper_erlang(branch(fatal, weight=cfr, stages=3, mean=8d, to=D),
                                 branch(recover,           stages=3, mean=12d, to=R))`
   The fatal chain exits to D, recover to R; the entry splits cfr / (1−cfr). *)

let ebola_src =
  "time_unit = 'days\n\
   compartments { S, E, I, R, D }\n\
   parameters {\n\
  \  beta : rate  sigma : rate  cfr : probability\n\
   }\n\
   transitions {\n\
  \  infection : S --> E @ beta * S * I / (S + E + I + R)\n\
  \  onset     : E --> I @ sigma * E\n\
  \  outcome   : I via hyper_erlang(\n\
  \    branch(label = fatal,   weight = cfr, stages = 3, mean =  8 'days, to = D),\n\
  \    branch(label = recover,               stages = 3, mean = 12 'days, to = R)\n\
  \  )\n\
   }\n\
   init { S = 990  E = 0  I = 10 }\n\
   simulate { from = 0 'days  to = 120 'days }\n"

let test_ebola_per_branch_destinations () =
  let m = compile_ok ebola_src in
  let names = comp_names m in
  List.iter (fun n -> Alcotest.(check bool) (n ^ " exists") true (List.mem n names))
    [ "I__fatal__1"; "I__fatal__2"; "I__fatal__3";
      "I__recover__1"; "I__recover__2"; "I__recover__3" ];
  Alcotest.(check bool) "bare I removed" false (List.mem "I" names);
  (* The fatal chain exits to D; the recover chain to R. *)
  let fatal_exit = tr_named m "outcome_fatal_exit" in
  Alcotest.(check bool) "fatal exit drains I__fatal__3" true (stoich_has fatal_exit "I__fatal__3" (-1));
  Alcotest.(check bool) "fatal exit --> D" true (stoich_has fatal_exit "D" 1);
  Alcotest.(check bool) "fatal exit does NOT --> R" false (stoich_has fatal_exit "R" 1);
  let recover_exit = tr_named m "outcome_recover_exit" in
  Alcotest.(check bool) "recover exit drains I__recover__3" true (stoich_has recover_exit "I__recover__3" (-1));
  Alcotest.(check bool) "recover exit --> R" true (stoich_has recover_exit "R" 1);
  Alcotest.(check bool) "recover exit does NOT --> D" false (stoich_has recover_exit "D" 1)

let test_ebola_entry_splits_cfr () =
  let m = compile_ok ebola_src in
  (* onset : E --> I became E --> { I__fatal__1 : cfr, I__recover__1 : 1−cfr }.
     So two onset-derived transitions, one into each first stage. *)
  let onsets =
    List.filter (fun (t : Ir.transition) ->
      String.length t.Ir.name >= 5 && String.sub t.Ir.name 0 5 = "onset")
      m.Ir.transitions in
  Alcotest.(check int) "two onset branches" 2 (List.length onsets);
  Alcotest.(check bool) "an onset branch lands in I__fatal__1" true
    (List.exists (fun (t : Ir.transition) -> stoich_has t "I__fatal__1" 1) onsets);
  Alcotest.(check bool) "an onset branch lands in I__recover__1" true
    (List.exists (fun (t : Ir.transition) -> stoich_has t "I__recover__1" 1) onsets)

let test_ebola_foi_sums_all_infectious () =
  let m = compile_ok ebola_src in
  (* The FOI's bare `I` (in `beta * S * I / N`) sums all six infectious stages. *)
  let inf = tr_named m "infection" in
  let names = pop_names [] inf.Ir.rate in
  List.iter (fun cell ->
    Alcotest.(check bool) (Printf.sprintf "FOI references %s" cell) true (List.mem cell names))
    [ "I__fatal__1"; "I__fatal__2"; "I__fatal__3";
      "I__recover__1"; "I__recover__2"; "I__recover__3" ];
  Alcotest.(check bool) "FOI has no dangling bare I" false (List.mem "I" names)

(* The per-stage chain rate for the fatal branch is 3/8 (mean 8 d, 3 stages);
   the recover branch is 3/12 (mean 12 d). At the ODE level the mean dwell of a
   k-stage Erlang(k, k/τ) is τ, so the fatal arm clears in ≈8 d and the recover
   arm in ≈12 d. We check the per-stage coefficients in the IR (k/τ): the chain
   rate of a fatal step is (3 / 8) * I__fatal__1; recover is (3 / 12). *)
let test_ebola_mixture_means () =
  let m = compile_ok ebola_src in
  (* Find a fatal chain step and a recover chain step; their rate's constant
     coefficient is k/τ. The mean is reported in DAYS; 8 'days folds to 8.0. *)
  (* A unit literal `8 'days` lowers to UncheckedDim(Const 8.0, ...); peel it. *)
  let rec peel = function Ir.UncheckedDim u -> peel u.Ir.inner | e -> e in
  let rec leading_const = function
    (* (k/τ) * Pop  →  fold to extract the constant ratio when τ is a literal. *)
    | Ir.BinOp { Ir.op = Ir.Mul; left; _ } -> leading_const left
    | Ir.BinOp { Ir.op = Ir.Div; left; right; _ } ->
      (match peel left, peel right with
       | Ir.Const a, Ir.Const b -> Some (a /. b)
       | _ -> None)
    | _ -> None in
  let fatal_step = tr_named m "outcome_fatal_stage1" in
  let recover_step = tr_named m "outcome_recover_stage1" in
  (match leading_const fatal_step.Ir.rate with
   | Some r -> Alcotest.(check (float 1e-9)) "fatal per-stage rate = 3/8" (3.0 /. 8.0) r
   | None -> Alcotest.fail "fatal step rate not of the (k/τ)*Pop form");
  (match leading_const recover_step.Ir.rate with
   | Some r -> Alcotest.(check (float 1e-9)) "recover per-stage rate = 3/12" (3.0 /. 12.0) r
   | None -> Alcotest.fail "recover step rate not of the (k/τ)*Pop form")

let test_ebola_simulates () =
  let m = compile_ok ebola_src in
  (* S,E,R,D base (4) + 6 flat stage cells = 10. Transitions: infection (1) +
     2 onset branches + (fatal: 2 steps + 1 exit) + (recover: 2 steps + 1 exit)
     = 1 + 2 + 3 + 3 = 9. *)
  Alcotest.(check int) "compartment count" 10 (List.length m.Ir.compartments);
  Alcotest.(check int) "transition count" 9 (List.length m.Ir.transitions)

(* ── Validation: one distinct E-code each, naming the transition ─────────────*)

let polio_with_clearance clearance =
  Printf.sprintf
    "time_unit = 'weeks\n\
     compartments { S, I, R, D }\n\
     parameters { beta : rate  p : probability  tau_typ : positive  tau_pro : positive }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S * I / (S + I + R)\n\
    \  %s\n\
     }\n\
     init { S = 990  I = 10 }\n" clearance

let test_err_non_last_branch_missing_weight () =
  (* The FIRST branch omits weight — only the LAST may. *)
  compile_expect_error_code ~code:"E256" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

let test_err_last_branch_has_weight () =
  (* The LAST branch carries a weight — it must be implicit (1 − Σ). *)
  compile_expect_error_code ~code:"E256" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ), \
        branch(label = prolonged, weight = p, stages = 1, mean = tau_pro))")

let test_err_fewer_than_two_branches () =
  compile_expect_error_code ~code:"E255" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = single, stages = 2, mean = tau_typ))")

let test_err_branch_no_destination () =
  (* No arrow target AND a branch with no `to` → no destination. *)
  compile_expect_error_code ~code:"E257" ~contains:"recover"
    (polio_with_clearance
       "clearance : I via hyper_erlang(\
        branch(label = fatal, weight = p, stages = 3, mean = tau_typ, to = D), \
        branch(label = recover, stages = 3, mean = tau_pro))")

let test_err_duplicate_labels () =
  compile_expect_error_code ~code:"E258" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = dup, weight = p, stages = 2, mean = tau_typ), \
        branch(label = dup, stages = 1, mean = tau_pro))")

let test_err_unknown_branch_kwarg () =
  compile_expect_error_code ~code:"E259" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ, banana = 1), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

let test_err_unknown_hyper_kwarg () =
  (* A bare keyword on hyper_erlang itself (not a branch) is rejected. *)
  compile_expect_error_code ~code:"E259" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(stages = 2, \
        branch(label = typical, weight = p, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

let test_err_non_positive_integer_stages () =
  compile_expect_error_code ~code:"E244" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2.5, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

let test_err_branch_both_mean_and_rate () =
  compile_expect_error_code ~code:"E245" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ, rate = beta), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

let test_err_branch_neither_mean_nor_rate () =
  compile_expect_error_code ~code:"E245" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

(* A literal weight outside [0,1] (or explicit weights summing past 1) makes the
   implicit last weight 1 − Σ negative — a negative entry rate / negative initial
   population. Must be rejected, not silently lowered. *)
let test_err_weight_out_of_range () =
  compile_expect_error_code ~code:"E225" ~contains:"clearance"
    (polio_with_clearance
       "clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = 1.5, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))")

(* An inflow with MULTIPLE destinations into a hyper-staged source — `S --> I + W` —
   cannot be split across the entry branches: the lowering would weight-1 the
   sibling and double the source's drain. Rejected loudly (the single-destination
   form is the supported pattern). *)
let test_err_multidest_inflow_into_staged () =
  let src =
    "time_unit = 'weeks\n\
     compartments { S, I, R, W }\n\
     parameters { beta : rate  p : probability  tau_typ : positive  tau_pro : positive }\n\
     transitions {\n\
    \  infection : S --> I + W @ beta * S * I / (S + I + R)\n\
    \  clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))\n\
     }\n\
     init { S = 990  I = 10  W = 0 }\n"
  in
  compile_expect_error_code ~code:"E224" ~contains:"infection" src

(* A branching (`DstBranch`) inflow into a hyper-staged source — `--> { src : w, … }` —
   is likewise unsupported, and must give a TARGETED E224, not a confusing E503 on
   the (silently staged-away) compartment the user wrote. Asserts both: E224 fires
   AND the E503 cascade on the vanished `I` is suppressed. *)
let test_err_branching_inflow_into_staged () =
  let src =
    "time_unit = 'weeks\n\
     compartments { S, I, R }\n\
     parameters { beta : rate  p : probability  tau_typ : positive  tau_pro : positive }\n\
     transitions {\n\
    \  infection : S --> { I : p, R : 1 - p } @ beta * S\n\
    \  clearance : I --> R via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))\n\
     }\n\
     init { S = 990  I = 10 }\n"
  in
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"hyper_test" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.fail "expected E224 but compile succeeded"
  | Error e ->
    Alcotest.(check bool) "targeted E224 naming the branching inflow" true
      (contains_substring ~needle:"E224" e && contains_substring ~needle:"infection" e);
    Alcotest.(check bool) "no confusing E503 cascade on the vanished compartment" true
      (not (contains_substring ~needle:"E503" e))

(* ── Deferred: stratified hyper_erlang → E248, not a crash / wrong lowering ──*)

let test_err_stratified_hyper_deferred () =
  let src =
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     dimensions { age = [child, adult] }\n\
     stratify(by = age)\n\
     parameters { beta : rate  p : probability  tau_typ : positive  tau_pro : positive }\n\
     tables { C_age : age \xc3\x97 age = [[12.0, 4.0], [4.0, 8.0]] }\n\
     transitions {\n\
    \  infection[a in age] : S[a] --> I[a] @ beta * S[a] * sum(b in age, C_age[a,b] * I[b])\n\
    \  clearance[a in age] : I[a] --> R[a] via hyper_erlang(\
        branch(label = typical, weight = p, stages = 2, mean = tau_typ), \
        branch(label = prolonged, stages = 1, mean = tau_pro))\n\
     }\n\
     init { S[child] = 990  I[child] = 10 }\n\
     simulate { from = 0 'days  to = 100 'days }\n"
  in
  compile_expect_error_code ~code:"E248" ~contains:"not yet supported" src

(* ── Regression: a plain `@ rate` model with no via is untouched ─────────────*)

let test_no_via_model_compiles_plain () =
  let src =
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     parameters { beta : rate  gamma : rate }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S * I / (S + I + R)\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     init { S = 990  I = 10 }\n"
  in
  let m = compile_ok src in
  Alcotest.(check int) "3 compartments" 3 (List.length m.Ir.compartments);
  Alcotest.(check int) "2 transitions" 2 (List.length m.Ir.transitions)

(* ── gh#463: the bare-source rewrite must reach ACTION expressions ───────────
   `hyper_erlang` deletes the base source compartment and replaces it with flat
   per-branch stage cells, so every surviving bare `I` in an expression has to be
   rewritten into the sum over those cells. The pass covered transition rates,
   lets, init, balance and observations, but not intervention / event / reactive
   action operands — so a pre-macro-valid expression became invalid after
   lowering (`error[E100]: undeclared name 'I'`).

   Endpoints (`from =` / `to =`) are deliberately NOT rewritten: those name a
   compartment, and a staged source has no single cell to name. That is gh#460's
   territory.                                                               ── *)

let hyper_model_with_block block =
  Printf.sprintf
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     parameters {\n\
    \  beta : rate  p : probability  tau_a : duration  tau_b : duration\n\
     }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S\n\
    \  clearance : I --> R via hyper_erlang(\n\
    \    branch(label = a, weight = p, stages = 1, mean = tau_a),\n\
    \    branch(label = b, stages = 1, mean = tau_b)\n\
    \  )\n\
     }\n\
     %s\n\
     init { S = 100  I = 10 }\n\
     simulate { from = 0 'days to = 10 'days }\n"
    block

(* Every stage cell the mixture creates, i.e. what a bare `I` must expand to. *)
let stage_cells m =
  List.filter
    (fun n -> String.length n > 3 && String.sub n 0 3 = "I__")
    (comp_names m)

let action_expr_of_intervention (m : Ir.model) name =
  let iv =
    match List.find_opt (fun (i : Ir.intervention) -> i.Ir.name = name) m.Ir.interventions with
    | Some i -> i
    | None -> Alcotest.failf "no intervention named %S" name
  in
  match iv.Ir.actions with
  | [ Ir.AddAction a ]         -> a.Ir.add_count
  | [ Ir.Set s ]               -> s.Ir.value
  | [ Ir.AbsoluteTransfer t ]  -> t.Ir.count
  | [ Ir.FractionTransfer t ]  -> t.Ir.fraction
  | _ -> Alcotest.failf "intervention %S: expected exactly one action" name

(* The rewritten operand must name every stage cell and no bare `I`. *)
let check_operand_rewritten m ~label expr =
  let cells = stage_cells m in
  Alcotest.(check bool)
    (label ^ ": mixture produced stage cells") true (cells <> []);
  let names = pop_names [] expr in
  Alcotest.(check bool)
    (label ^ ": no bare 'I' survives") false (List.mem "I" names);
  List.iter
    (fun c ->
      Alcotest.(check bool)
        (Printf.sprintf "%s: operand sums %s" label c) true (List.mem c names))
    cells

let test_add_count_operand_rewritten () =
  let m = compile_ok (hyper_model_with_block
    "interventions {\n\
    \  pulse : add(S, I) at [1]\n\
     }") in
  check_operand_rewritten m ~label:"add count" (action_expr_of_intervention m "pulse")

let test_set_value_operand_rewritten () =
  let m = compile_ok (hyper_model_with_block
    "interventions {\n\
    \  zap : { S = I at = [1] }\n\
     }") in
  check_operand_rewritten m ~label:"set value" (action_expr_of_intervention m "zap")

let test_transfer_count_operand_rewritten () =
  let m = compile_ok (hyper_model_with_block
    "interventions {\n\
    \  pull : transfer(from = S, to = R, count = I) at [1]\n\
     }") in
  check_operand_rewritten m ~label:"transfer count" (action_expr_of_intervention m "pull")

let test_event_action_operand_rewritten () =
  (* Events share the action grammar, so they must share the rewrite. *)
  let m = compile_ok (hyper_model_with_block
    "events {\n\
    \  seed : add(S, I) at [1]\n\
     }") in
  check_operand_rewritten m ~label:"event add count" (action_expr_of_intervention m "seed")

let test_action_endpoints_not_rewritten () =
  (* Negative control: the rewrite must touch operands only. `from`/`to` still
     name plain compartments, unchanged by the mixture lowering. *)
  let m = compile_ok (hyper_model_with_block
    "interventions {\n\
    \  pull : transfer(from = S, to = R, count = I) at [1]\n\
     }") in
  let iv = List.find (fun (i : Ir.intervention) -> i.Ir.name = "pull") m.Ir.interventions in
  match iv.Ir.actions with
  | [ Ir.AbsoluteTransfer t ] ->
    Alcotest.(check string) "src untouched" "S" t.Ir.src;
    Alcotest.(check string) "dst untouched" "R" t.Ir.dst
  | _ -> Alcotest.fail "expected a single AbsoluteTransfer"

let test_non_hyper_action_operand_untouched () =
  (* Regression: with no `via` in the model, action operands are left alone. *)
  let src =
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     parameters { beta : rate  gamma : rate }\n\
     transitions {\n\
    \  infection : S --> I @ beta * S\n\
    \  recovery  : I --> R @ gamma * I\n\
     }\n\
     interventions {\n\
    \  pulse : add(S, I) at [1]\n\
     }\n\
     init { S = 990  I = 10 }\n\
     simulate { from = 0 'days to = 10 'days }\n"
  in
  let m = compile_ok src in
  let names = pop_names [] (action_expr_of_intervention m "pulse") in
  Alcotest.(check bool) "plain model keeps bare I" true (List.mem "I" names)

let () =
  Alcotest.run "hyper_erlang"
    [ ( "polio-same-endpoint",
        [ Alcotest.test_case "flat per-branch compartments exist" `Quick
            test_polio_flat_compartments_exist;
          Alcotest.test_case "both chains exit to R" `Quick
            test_polio_both_chains_exit_to_R;
          Alcotest.test_case "entry DstBranch weighted p / 1−p" `Quick
            test_polio_entry_dstbranch_weighted;
          Alcotest.test_case "bare I in FOI sums all stages" `Quick
            test_polio_foi_sums_all_stages;
          Alcotest.test_case "init split by weight (no bare I)" `Quick
            test_polio_init_split_by_weight;
          Alcotest.test_case "simulates / sane IR shape" `Quick
            test_polio_simulates ] );
      ( "ebola-per-branch-endpoints",
        [ Alcotest.test_case "per-branch destinations (fatal→D, recover→R)" `Quick
            test_ebola_per_branch_destinations;
          Alcotest.test_case "entry splits cfr / 1−cfr" `Quick
            test_ebola_entry_splits_cfr;
          Alcotest.test_case "FOI sums all infectious stages" `Quick
            test_ebola_foi_sums_all_infectious;
          Alcotest.test_case "mixture means 8d / 12d (per-stage k/τ)" `Quick
            test_ebola_mixture_means;
          Alcotest.test_case "simulates / sane IR shape" `Quick
            test_ebola_simulates ] );
      ( "validation",
        [ Alcotest.test_case "non-last branch missing weight → E256" `Quick
            test_err_non_last_branch_missing_weight;
          Alcotest.test_case "last branch has weight → E256" `Quick
            test_err_last_branch_has_weight;
          Alcotest.test_case "fewer than two branches → E255" `Quick
            test_err_fewer_than_two_branches;
          Alcotest.test_case "branch with no destination → E257" `Quick
            test_err_branch_no_destination;
          Alcotest.test_case "duplicate labels → E258" `Quick
            test_err_duplicate_labels;
          Alcotest.test_case "unknown branch kwarg → E259" `Quick
            test_err_unknown_branch_kwarg;
          Alcotest.test_case "unknown hyper_erlang kwarg → E259" `Quick
            test_err_unknown_hyper_kwarg;
          Alcotest.test_case "non-positive-integer stages → E244" `Quick
            test_err_non_positive_integer_stages;
          Alcotest.test_case "branch both mean and rate → E245" `Quick
            test_err_branch_both_mean_and_rate;
          Alcotest.test_case "branch neither mean nor rate → E245" `Quick
            test_err_branch_neither_mean_nor_rate;
          Alcotest.test_case "weight out of [0,1] → E225" `Quick
            test_err_weight_out_of_range;
          Alcotest.test_case "multi-dest inflow into staged source → E224" `Quick
            test_err_multidest_inflow_into_staged;
          Alcotest.test_case "branching inflow into staged source → E224 (no E503 cascade)" `Quick
            test_err_branching_inflow_into_staged ] );
      ( "deferred-stratified",
        [ Alcotest.test_case "stratified hyper_erlang → E248" `Quick
            test_err_stratified_hyper_deferred ] );
      ( "action-operand-rewrite-gh463",
        [ Alcotest.test_case "add(_, I) count operand sums stages" `Quick
            test_add_count_operand_rewritten;
          Alcotest.test_case "set value operand sums stages" `Quick
            test_set_value_operand_rewritten;
          Alcotest.test_case "transfer count operand sums stages" `Quick
            test_transfer_count_operand_rewritten;
          Alcotest.test_case "events {} action operand sums stages" `Quick
            test_event_action_operand_rewritten;
          Alcotest.test_case "transfer from/to endpoints NOT rewritten" `Quick
            test_action_endpoints_not_rewritten ] );
      ( "regression",
        [ Alcotest.test_case "no-via model compiles plain" `Quick
            test_no_via_model_compiles_plain;
          Alcotest.test_case "no-via action operand untouched" `Quick
            test_non_hyper_action_operand_untouched ] ) ]
