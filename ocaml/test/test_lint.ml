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

   The shared-projection lint is L404: two or more observation streams
   whose RESOLVED projections are identical, so data bound to both enters
   the joint log-likelihood twice. Its negative controls guard the
   opposite risk from L402's — L404 must fire on the collision even when
   the two streams are legitimate (camdl cannot tell a duplicated file
   from two genuine observation processes), so the controls pin that it
   stays a Warning and that distinct projections never trip it.

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
    ?(initial_conditions = [])
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
      t_end_anchor = None;
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

(* An observation stream over [projection]. [scored] is the `~` LHS — the
   declared value column the likelihood consumes, i.e. the NAME of the quantity
   being measured — and defaults to the stream name. [source] is the
   `from <label>` data-source key (also defaulting to the stream name); [rate]
   is the Poisson mean, which defaults to the bare projection but is overridden
   where a test needs two streams to differ in ascertainment. *)
let mk_obs ?source ?scored ?rate ?(stratum = []) name projection : observation_model =
  let scored = match scored with Some s -> s | None -> name in
  { name;
    obs_source = (match source with Some s -> s | None -> name);
    columns = [{ col_name = "time"; col_role = RoleTime };
               { col_name = scored; col_role = RoleValue Count }];
    scored;
    emit_schedule = Some (ObsRegular { start = 0.0; step = 1.0; end_ = 100.0 });
    stratum;
    projection;
    projection_state_grad = [];
    likelihood =
      Poisson { rate = { expr = (match rate with Some e -> e | None -> Projected);
                         grad = []; proj_grad = None } } }

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
    ~initial_conditions:[("S", Ir.Deterministic (Ir.Const 999.0));
                         ("I", Ir.Deterministic (Ir.Const 1.0));
                         ("R", Ir.Deterministic (Ir.Const 0.0))]
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

(* ── L404: two streams reading one latent quantity ───────────────────── *)

(* The base every L404 test varies: a runnable SIR whose observation list
   the caller supplies. *)
let obs_model observations =
  empty_model
    ~compartments:sir_compartments
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[infection_tr; recovery_tr]
    ~bindings:[n_binding]
    ~observations
    ()

(* The reported instance from gh#853, and the shape the lint exists for: a
   weekly case file and a daily case file are ONE case series at two
   resolutions. Two streams, two `from` labels so they bind different files —
   but one projection and, decisively, ONE scored column name, because both
   files publish the same measured quantity under the same header. Bound
   together, every case is counted twice. *)
let test_one_series_two_resolutions_flagged () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases_weekly" ~source:"cases_weekly" ~scored:"cases"
      (CumulativeFlow "infection");
    mk_obs "cases_daily"  ~source:"cases_daily"  ~scored:"cases"
      (CumulativeFlow "infection");
  ]) in
  Alcotest.(check int) "exactly one L404" 1 (lint_count "L404" r);
  Alcotest.(check bool) "names the first stream"  true
    (has_lint_for "L404" "cases_weekly" r);
  Alcotest.(check bool) "names the second stream" true
    (has_lint_for "L404" "cases_daily" r);
  Alcotest.(check bool) "names the scored quantity" true
    (has_lint_for "L404" "cases" r)

(* Three streams on one measurement collapse to ONE diagnostic naming all
   three, not to a diagnostic per pair. *)
let test_three_streams_one_group () =
  let r = Lint.check_model (obs_model [
    mk_obs "a" ~scored:"cases" (CumulativeFlow "infection");
    mk_obs "b" ~scored:"cases" (CumulativeFlow "infection");
    mk_obs "c" ~scored:"cases" (CumulativeFlow "infection");
  ]) in
  Alcotest.(check int) "one diagnostic for the whole group" 1 (lint_count "L404" r);
  List.iter (fun n ->
    Alcotest.(check bool) (Printf.sprintf "names %s" n) true
      (has_lint_for "L404" n r)) ["a"; "b"; "c"]

(* A flow SUM is commutative, so two streams pooling the same strata in
   opposite order are one projection. Comparing the lists verbatim would
   miss this — the same "one quantity, two spellings" hazard gh#678
   established for the within-stream case. *)
let test_flow_sum_order_insensitive () =
  let r = Lint.check_model (obs_model [
    mk_obs "pooled_a" ~scored:"cases"
      (CumulativeFlowSum ["infection_north"; "infection_south"]);
    mk_obs "pooled_b" ~scored:"cases"
      (CumulativeFlowSum ["infection_south"; "infection_north"]);
  ]) in
  Alcotest.(check int) "commuted flow sums are one projection" 1 (lint_count "L404" r)

(* A one-element sum and the scalar form name the same single flow. *)
let test_singleton_sum_equals_scalar () =
  let r = Lint.check_model (obs_model [
    mk_obs "scalar"    ~scored:"cases" (CumulativeFlow "infection");
    mk_obs "singleton" ~scored:"cases" (CumulativeFlowSum ["infection"]);
  ]) in
  Alcotest.(check int) "singleton sum ≡ scalar flow" 1 (lint_count "L404" r);
  let r = Lint.check_model (obs_model [
    mk_obs "scalar"    ~scored:"prev" (CurrentPop "I");
    mk_obs "singleton" ~scored:"prev" (CurrentPopSum ["I"]);
  ]) in
  Alcotest.(check int) "singleton pop sum ≡ scalar pop" 1 (lint_count "L404" r)

(* The gh#678 shape lifted to the stream layer: a stratified stream whose
   projection never mentions its own binder expands to one leaf per stratum,
   every leaf projecting the SAME flow. Each patch's data rows are then scored
   against the north patch's incidence.

   The narrowed key must not lose this. `scored` is the source `~` LHS token,
   taken once per DECLARATION and copied to every expanded leaf (expander.ml,
   `Ir.scored = meas_v.om_scored`), so the leaves of one stratified stream
   share a scored name by construction — and here they share a projection too,
   which is the defect. *)
let test_stratified_stream_ignoring_its_binder () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases_north" ~source:"cases" ~scored:"cases" ~stratum:["patch", "north"]
      (CumulativeFlow "infection_north");
    mk_obs "cases_south" ~source:"cases" ~scored:"cases" ~stratum:["patch", "south"]
      (CumulativeFlow "infection_north");
  ]) in
  Alcotest.(check int) "leaves of one stream sharing a flow are flagged"
    1 (lint_count "L404" r)

(* Identical derived expressions over the state, scored under one name. *)
let test_identical_derived_expr_flagged () =
  let prevalence () = DerivedExpr (pop "I" /. (pop "S" +. pop "I" +. pop "R")) in
  let r = Lint.check_model (obs_model [
    mk_obs "prev_survey"   ~scored:"prevalence" (prevalence ());
    mk_obs "prev_sentinel" ~scored:"prevalence" (prevalence ());
  ]) in
  Alcotest.(check int) "identical derived exprs are one projection" 1 (lint_count "L404" r)

(* ── L404 negative controls ──────────────────────────────────────────── *)

(* The case that forced the predicate to narrow, measured on a real
   surveillance corpus: cases and deaths driven off ONE confirmation flow,
   each with its own multiplier and its own dispersion, each scoring its own
   named quantity. Two observation processes on one latent state — ordinary
   multi-stream modelling, and the joint is correct. A projection-only key
   fired on 58 of 98 models in that corpus, essentially all of them this
   shape. *)
let test_cases_and_deaths_off_one_flow_not_shared () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases"  ~scored:"cases"
      ~rate:(param "rho" *. Projected)       (CumulativeFlow "confirm");
    mk_obs "deaths" ~scored:"deaths"
      ~rate:(param "rho_d" *. Projected)     (CumulativeFlow "confirm");
  ]) in
  Alcotest.(check int) "distinct measured quantities on one flow" 0
    (lint_count "L404" r)

(* The property most likely to regress: two streams that legitimately share
   one latent quantity AND publish it under one column name — two
   laboratories both reporting a column called `cases` — must keep compiling.
   camdl cannot distinguish that from a duplicated file (whether two files
   hold the same measurements is a fact about how they were produced, not
   about the model), so L404 DOES fire here; what it must never do is
   escalate. Pin the severity, which is what keeps the model runnable. *)
let test_two_labs_one_scored_name_warns_never_errors () =
  let r = Lint.check_model (obs_model [
    mk_obs "lab_a" ~source:"lab_a" ~scored:"cases"
      ~rate:(param "rho_a" *. Projected) (CumulativeFlow "infection");
    mk_obs "lab_b" ~source:"lab_b" ~scored:"cases"
      ~rate:(param "rho_b" *. Projected) (CumulativeFlow "infection");
  ]) in
  Alcotest.(check int) "the legitimate case still warns" 1 (lint_count "L404" r);
  Alcotest.(check bool) "every L404 is a Warning, never blocking" true
    (List.for_all (fun (d : Lint.diagnostic) -> d.severity = Lint.Warning)
       r.diagnostics)

(* Different projection HEADS over the same compartment are different
   quantities: a flow accumulated over the reporting interval is not the
   state read at the instant. *)
let test_flow_and_pop_not_shared () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases"      ~scored:"burden" (CumulativeFlow "infection");
    mk_obs "prevalence" ~scored:"burden" (CurrentPop "I");
  ]) in
  Alcotest.(check int) "incidence and prevalence are distinct" 0 (lint_count "L404" r)

(* The ordinary stratified stream: one leaf per patch, each projecting its
   OWN patch's flow. This is the shape every spatial model has, and a false
   positive here would fire on most of the corpus.

   Note both leaves carry ONE scored name, as the expander produces — so this
   test isolates the projection half of the key: it is the differing
   projections, and nothing else, that keep it quiet. *)
let test_distinct_strata_not_shared () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases_north" ~source:"cases" ~scored:"cases" ~stratum:["patch", "north"]
      (CumulativeFlow "infection_north");
    mk_obs "cases_south" ~source:"cases" ~scored:"cases" ~stratum:["patch", "south"]
      (CumulativeFlow "infection_south");
  ]) in
  Alcotest.(check int) "per-stratum leaves are distinct" 0 (lint_count "L404" r)

(* Disjoint pools over the same family are different quantities. *)
let test_disjoint_flow_sums_not_shared () =
  let r = Lint.check_model (obs_model [
    mk_obs "north_pool" ~scored:"cases"
      (CumulativeFlowSum ["infection_n1"; "infection_n2"]);
    mk_obs "south_pool" ~scored:"cases"
      (CumulativeFlowSum ["infection_s1"; "infection_s2"]);
  ]) in
  Alcotest.(check int) "disjoint pools are distinct" 0 (lint_count "L404" r)

(* Different derived expressions are different quantities. *)
let test_distinct_derived_exprs_not_shared () =
  let r = Lint.check_model (obs_model [
    mk_obs "prev_i" ~scored:"frac"
      (DerivedExpr (pop "I" /. (pop "S" +. pop "I" +. pop "R")));
    mk_obs "prev_r" ~scored:"frac"
      (DerivedExpr (pop "R" /. (pop "S" +. pop "I" +. pop "R")));
  ]) in
  Alcotest.(check int) "distinct derived exprs" 0 (lint_count "L404" r)

(* A single stream, and a model with no observations at all, are clean. *)
let test_single_and_zero_streams_clean () =
  let r = Lint.check_model (obs_model [
    mk_obs "cases" ~scored:"cases" (CumulativeFlow "infection")]) in
  Alcotest.(check int) "one stream cannot collide" 0 (lint_count "L404" r);
  let r = Lint.check_model (obs_model []) in
  Alcotest.(check int) "no streams, no lint" 0 (lint_count "L404" r)

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
    "shared_measurement", [
      Alcotest.test_case "one series at two resolutions"     `Quick test_one_series_two_resolutions_flagged;
      Alcotest.test_case "three streams, one diagnostic"     `Quick test_three_streams_one_group;
      Alcotest.test_case "flow sum is order-insensitive"     `Quick test_flow_sum_order_insensitive;
      Alcotest.test_case "singleton sum ≡ scalar"            `Quick test_singleton_sum_equals_scalar;
      Alcotest.test_case "stratified stream ignoring binder" `Quick test_stratified_stream_ignoring_its_binder;
      Alcotest.test_case "identical derived exprs"           `Quick test_identical_derived_expr_flagged;
      Alcotest.test_case "cases and deaths off one flow"     `Quick test_cases_and_deaths_off_one_flow_not_shared;
      Alcotest.test_case "two labs warn, never error"        `Quick test_two_labs_one_scored_name_warns_never_errors;
      Alcotest.test_case "incidence vs prevalence"           `Quick test_flow_and_pop_not_shared;
      Alcotest.test_case "per-stratum leaves are distinct"   `Quick test_distinct_strata_not_shared;
      Alcotest.test_case "disjoint flow sums"                `Quick test_disjoint_flow_sums_not_shared;
      Alcotest.test_case "distinct derived exprs"            `Quick test_distinct_derived_exprs_not_shared;
      Alcotest.test_case "single / zero streams are clean"   `Quick test_single_and_zero_streams_clean;
    ];
  ]
