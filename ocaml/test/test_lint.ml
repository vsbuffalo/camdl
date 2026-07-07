(* Unit tests for the model linter (Lint).

   Each test constructs a minimal Ir.model, runs Lint.check_model, and
   asserts the presence (positive) or absence (negative controls) of a
   specific lint code. Lints are Warning severity and never block
   compilation.

   The dead-compartment lint is L402: a compartment declared in
   `model.compartments` whose name is referenced nowhere is flagged. The
   negative controls guard the false-positive risk — a compartment used
   only in a binding body, only in an observation, or only as an init
   target is NOT dead.

   Run with:  cd ocaml && dune runtest *)

open Ir

(* ── Helpers ───────────────────────────────────────────────────────────── *)

let has_lint code (result : Lint.result) =
  List.exists (fun (d : Lint.diagnostic) ->
    d.severity = Lint.Warning && d.code = code
  ) result.diagnostics

(* True iff an L402 lint mentions compartment [name] in its message. The
   message has the shape "compartment 'X' is declared but never used". *)
let has_lint_for code name (result : Lint.result) =
  let needle = Printf.sprintf "'%s'" name in
  List.exists (fun (d : Lint.diagnostic) ->
    d.severity = Lint.Warning
    && d.code = code
    && (let m = d.message in
        let nl = String.length needle and ml = String.length m in
        let rec scan i = i + nl <= ml && (String.sub m i nl = needle || scan (i + 1)) in
        scan 0)
  ) result.diagnostics

let no_lint code (result : Lint.result) = not (has_lint code result)

let lint_count code (result : Lint.result) =
  List.length (List.filter (fun (d : Lint.diagnostic) ->
    d.severity = Lint.Warning && d.code = code
  ) result.diagnostics)

(* Minimal model scaffold — fill in fields as needed. Mirrors the
   test_dimcheck.ml empty_model but exposes the extra fields the
   reference-collection lint must walk (bindings, interventions,
   initial_conditions). *)
let empty_model
    ?(name = "test")
    ?(compartments = [])
    ?(transitions = [])
    ?(parameters = [])
    ?(observations = [])
    ?(ode_equations = [])
    ?(tables = [])
    ?(time_functions = [])
    ?(interventions = [])
    ?(bindings = [])
    ?(initial_conditions = Explicit [])
    ?(balance = None)
    ?(identity_tracked_compartments = [])
    () : model =
  { name;
    version = "1.0";
    time_unit = "days";
    description = None;
    origin = None;
    origin_rata_die = None;
    compartments;
    transitions;
    ode_equations;
    time_functions;
    tables;
    interventions;
    observations;
    parameters;
    bindings;
    per_eval_bindings = [];
    initial_conditions;
    ic_grad = [];
    output = {
      times = OutRegular { start = 0.0; step = 1.0 };
      format = "tsv";
      trajectory = true;
      observations = false;
    };
    simulation = {
      t_start = 0.0;
      t_end = 100.0;
      time_semantics = "continuous";
      dt = None;
      rng_seed = None;
      integrator = Rk4;
    };
    presets = [];
    model_structure = None;
    balance;
    identity_tracked_compartments;
    doc_index = empty_doc_index;
    quantities = [];
    contrasts = [];
  }

let mk_compartment name : compartment = { name; kind = Integer }

let mk_param ?(kind = None) name : parameter =
  { name; value = Required; param_kind = kind; param_dim = None }

let mk_transition ?(stoich = []) ?(metadata = None) name rate : transition =
  { name; stoichiometry = stoich; rate; metadata;
    draw_method = DrawPoisson; rate_grad = []; rate_state_grad = []; lineage = None }

(* Expression shorthands *)
let pop s = Pop s
let param s = Param s
let const f = Const f
let ( *. ) a b = BinOp { op = Mul; left = a; right = b }
let ( /. ) a b = BinOp { op = Div; left = a; right = b }
let ( +. ) a b = BinOp { op = Add; left = a; right = b }

(* A clean SIR base: S -> I (infection), I -> R (recovery). Every
   compartment is referenced (S, I in infection rate + stoich; I, R in
   recovery). N is the binding total. *)
let sir_compartments = [mk_compartment "S"; mk_compartment "I"; mk_compartment "R"]

let infection_tr =
  mk_transition "infection"
    ~stoich:[("S", -1); ("I", 1)]
    ~metadata:(Some { origin_kind = Some "transmission";
                      source_compartment = Some "I";
                      dest_compartment = Some "S" })
    (param "beta" *. pop "S" *. pop "I" /. pop "N")

let recovery_tr =
  mk_transition "recovery"
    ~stoich:[("I", -1); ("R", 1)]
    (param "gamma" *. pop "I")

let n_binding : binding = { bname = "N"; bexpr = pop "S" +. pop "I" +. pop "R" }

(* ── Positive: a genuinely orphan compartment is flagged ─────────────── *)

(* Model with S, I, R wired into transitions, plus a declared-but-unused
   compartment X. X appears in no transition / binding / observation /
   init / intervention / balance / ODE / identity list → L402. *)
let test_dead_compartment_flagged () =
  let m = empty_model
    ~compartments:(sir_compartments @ [mk_compartment "X"])
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[infection_tr; recovery_tr]
    ~bindings:[n_binding]
    () in
  let r = Lint.check_model m in
  Alcotest.(check bool) "L402 fires" true (has_lint "L402" r);
  Alcotest.(check bool) "L402 names X" true (has_lint_for "L402" "X" r);
  Alcotest.(check int) "exactly one L402" 1 (lint_count "L402" r)

(* ── Negative control 1: referenced only in a binding body ───────────── *)

(* R appears nowhere except the `let N = S + I + R` binding. A naive
   transition-only scan would flag R as dead; the comprehensive
   reference-collection must walk binding bodies, so NO L402 for R. To
   isolate R, drop the recovery transition (which would otherwise
   reference R via stoichiometry) and use an SI model whose total N still
   sums over R. *)
let test_binding_only_not_dead () =
  let infection_no_r =
    mk_transition "infection"
      ~stoich:[("S", -1); ("I", 1)]
      (param "beta" *. pop "S" *. pop "I" /. pop "N") in
  let m = empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[infection_no_r]
    ~bindings:[n_binding]   (* N = S + I + R — R lives only here *)
    () in
  let r = Lint.check_model m in
  Alcotest.(check bool) "R (binding-only) is not flagged"
    false (has_lint_for "L402" "R" r)

(* ── Negative control 2: referenced only in an observation ───────────── *)

(* R appears only in an observation projection (CurrentPop "R"). No
   transition / binding / init touches R. Comprehensive collection must
   walk observation projections → NO L402 for R. *)
let test_observation_only_not_dead () =
  let infection_no_r =
    mk_transition "infection"
      ~stoich:[("S", -1); ("I", 1)]
      (param "beta" *. pop "S" *. pop "I") in
  let obs : observation_model =
    { name = "recovered"; obs_source = "recovered";
      columns = [{ col_name = "time"; col_role = RoleTime };
                 { col_name = "recovered"; col_role = RoleValue Count }];
      scored = "recovered";
      emit_schedule = Some (ObsRegular { start = 0.0; step = 1.0; end_ = 100.0 });
      stratum = [];
      projection = CurrentPop "R";
      projection_state_grad = [];
      likelihood = Poisson { rate = { expr = Projected; grad = []; proj_grad = None } } } in
  let m = empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[infection_no_r]
    ~observations:[obs]   (* R lives only in this projection *)
    () in
  let r = Lint.check_model m in
  Alcotest.(check bool) "R (observation-only) is not flagged"
    false (has_lint_for "L402" "R" r)

(* Same as above, but R reached via a DerivedExpr observation (Pop "R"
   inside an arithmetic expression). Exercises the expr-walk path of the
   observation projection. *)
let test_derived_observation_only_not_dead () =
  let infection_no_r =
    mk_transition "infection"
      ~stoich:[("S", -1); ("I", 1)]
      (param "beta" *. pop "S" *. pop "I") in
  let obs : observation_model =
    { name = "frac_recovered"; obs_source = "frac_recovered";
      columns = [{ col_name = "time"; col_role = RoleTime };
                 { col_name = "frac_recovered"; col_role = RoleValue Real }];
      scored = "frac_recovered";
      emit_schedule = Some (ObsRegular { start = 0.0; step = 1.0; end_ = 100.0 });
      stratum = [];
      projection = DerivedExpr (pop "R" /. (pop "S" +. pop "I" +. pop "R"));
      projection_state_grad = [];
      likelihood = Normal { mean = { expr = Projected; grad = []; proj_grad = None };
                            sd = { expr = const 1.0; grad = []; proj_grad = None } } } in
  let m = empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[infection_no_r]
    ~observations:[obs]
    () in
  let r = Lint.check_model m in
  Alcotest.(check bool) "R (derived-obs-only) is not flagged"
    false (has_lint_for "L402" "R" r)

(* ── Negative control 3: referenced only as an init target ───────────── *)

(* R appears only as an initial-condition target. Comprehensive
   collection must include init targets → NO L402 for R. *)
let test_init_target_only_not_dead () =
  let infection_no_r =
    mk_transition "infection"
      ~stoich:[("S", -1); ("I", 1)]
      (param "beta" *. pop "S" *. pop "I") in
  let m = empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[infection_no_r]
    ~initial_conditions:(Explicit [("S", 999.0); ("I", 1.0); ("R", 0.0)])
    () in
  let r = Lint.check_model m in
  Alcotest.(check bool) "R (init-target-only) is not flagged"
    false (has_lint_for "L402" "R" r)

(* ── Negative control 4: clean SIR yields zero L402 ──────────────────── *)

let test_clean_sir_no_lint () =
  let m = empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[infection_tr; recovery_tr]
    ~bindings:[n_binding]
    () in
  let r = Lint.check_model m in
  Alcotest.(check int) "clean SIR has zero L402" 0 (lint_count "L402" r);
  Alcotest.(check bool) "clean SIR no L402" true (no_lint "L402" r)

(* ── Driver ──────────────────────────────────────────────────────────── *)

let () =
  Alcotest.run "lint" [
    "dead_compartment", [
      Alcotest.test_case "orphan compartment flagged"        `Quick test_dead_compartment_flagged;
      Alcotest.test_case "binding-only is not dead"          `Quick test_binding_only_not_dead;
      Alcotest.test_case "observation-only is not dead"      `Quick test_observation_only_not_dead;
      Alcotest.test_case "derived-observation-only not dead" `Quick test_derived_observation_only_not_dead;
      Alcotest.test_case "init-target-only is not dead"      `Quick test_init_target_only_not_dead;
      Alcotest.test_case "clean SIR has no lint"             `Quick test_clean_sir_no_lint;
    ];
  ]
