(* Unit tests for the dependency lattice (Expr_analysis).

   Asserts dep_of_expr on representative expressions and the join /
   topological-resolution behaviour over a model's bindings.

   Run with:  cd ocaml && dune runtest *)

open Ir
module E = Expr_analysis

(* No-binding environment for the leaf cases. *)
let no_bindings _ = E.Const
let dep e = E.dep_of_expr ~binding_dep:no_bindings e

(* Pretty-print a dep for Alcotest's testable. *)
let dep_t : E.dep Alcotest.testable =
  Alcotest.testable (fun ppf d -> Fmt.string ppf (E.dep_name d)) ( = )

(* ── dep_of_expr on representative exprs ─────────────────────────────────── *)

let test_const () =
  Alcotest.check dep_t "literal is Const" E.Const (dep (Const 3.0))

let test_param_only () =
  (* A pure parameter expression: log(beta) → Param. *)
  Alcotest.check dep_t "param expr is Param"
    E.Param (dep (UnOp { op = Log; arg = Param "beta" }))

let test_foi_is_state () =
  (* beta * S / N → State (the populations dominate the param). *)
  let foi =
    BinOp { op = Div;
            left = BinOp { op = Mul; left = Param "beta"; right = Pop "S" };
            right = Pop "N" } in
  Alcotest.check dep_t "beta*S/N is State" E.State (dep foi)

let test_seasonal_is_time () =
  (* A time function modulated by a parameter: amp * seasonal(t).
     Time outranks Param in the chain, so the join is Time. *)
  let seasonal =
    BinOp { op = Mul; left = Param "amp"; right = TimeFunc "season" } in
  Alcotest.check dep_t "amp*seasonal(t) is Time" E.Time (dep seasonal);
  Alcotest.check dep_t "bare Dt is Time" E.Time (dep Dt);
  Alcotest.check dep_t "bare Time is Time" E.Time (dep Time)

let test_const_table_lookup () =
  (* A constant-indexed table lookup floors to Data (its compile-time
     cells); a state-indexed lookup is pulled up to State. *)
  Alcotest.check dep_t "W[0,1] const-indexed is Data"
    E.Data (dep (TableLookup ("W", [ Const 0.0; Const 1.0 ])));
  Alcotest.check dep_t "W[S] state-indexed is State"
    E.State (dep (TableLookup ("W", [ Pop "S" ])))

let test_projected () =
  Alcotest.check dep_t "Projected is Projected" E.Projected (dep Projected)

let test_reduce_joins () =
  (* A Reduce joins over its terms: a const + a state term → State. *)
  let r = Reduce [ Const 1.0; Pop "I"; Param "k" ] in
  Alcotest.check dep_t "Reduce[const, state, param] is State" E.State (dep r)

(* ── join semantics ──────────────────────────────────────────────────────── *)

let test_join_chain () =
  (* join is commutative and returns the more-dynamic; Const is the unit. *)
  Alcotest.check dep_t "join Const x = x" E.State (E.join E.Const E.State);
  Alcotest.check dep_t "join commutes" E.State (E.join E.State E.Const);
  Alcotest.check dep_t "Param < Time" E.Time (E.join E.Param E.Time);
  Alcotest.check dep_t "State < Projected" E.Projected (E.join E.State E.Projected);
  Alcotest.check dep_t "join_list folds from Const"
    E.State (E.join_list [ E.Const; E.Data; E.State ])

(* ── model_binding_deps: topological BindingRef resolution ───────────────── *)

(* Minimal model carrying just the fields model_binding_deps reads. The
   other fields are stubbed; the analysis only walks `m.bindings`. *)
let model_with_bindings bs =
  {
    name = "t"; version = "1"; time_unit = "days"; description = None;
    origin = None; origin_rata_die = None;
    compartments = []; transitions = []; ode_equations = [];
    time_functions = []; tables = []; interventions = []; observations = [];
    parameters = []; bindings = bs; per_eval_bindings = [];
    initial_conditions = Explicit [];
    ic_grad = [];
    output = { times = OutAtTimes []; format = "tsv";
               trajectory = true; observations = true };
    simulation = { t_start = 0.0; t_end = 1.0; time_semantics = "continuous";
                   dt = None; rng_seed = None;
                   integrator = Rk4 };
    presets = []; model_structure = None; balance = None;
    identity_tracked_compartments = [];
    doc_index = empty_doc_index;
    quantities = [];
    contrasts = [];
  }

let test_binding_deps_topo () =
  (* N = S + I            → State
     ref_N = N            → State (resolved through the earlier binding)
     c = 2.0              → Const *)
  let bs = [
    { bname = "N";     bexpr = BinOp { op = Add; left = Pop "S"; right = Pop "I" } };
    { bname = "ref_N"; bexpr = BindingRef "N" };
    { bname = "c";     bexpr = Const 2.0 };
  ] in
  let bd = E.model_binding_deps (model_with_bindings bs) in
  Alcotest.check dep_t "binding N is State"     E.State (bd "N");
  Alcotest.check dep_t "binding ref_N resolves to State" E.State (bd "ref_N");
  Alcotest.check dep_t "binding c is Const"     E.Const (bd "c");
  Alcotest.check dep_t "unknown binding floors to Const" E.Const (bd "nope")

let () =
  Alcotest.run "expr_analysis" [
    "dep_of_expr", [
      Alcotest.test_case "Const literal"            `Quick test_const;
      Alcotest.test_case "param-only → Param"       `Quick test_param_only;
      Alcotest.test_case "beta*S/N → State"         `Quick test_foi_is_state;
      Alcotest.test_case "seasonal(t) → Time"       `Quick test_seasonal_is_time;
      Alcotest.test_case "table lookup Data/State"  `Quick test_const_table_lookup;
      Alcotest.test_case "Projected → Projected"    `Quick test_projected;
      Alcotest.test_case "Reduce joins terms"       `Quick test_reduce_joins;
    ];
    "join", [
      Alcotest.test_case "chain + unit + commute"   `Quick test_join_chain;
    ];
    "model_binding_deps", [
      Alcotest.test_case "topological BindingRef resolution" `Quick test_binding_deps_topo;
    ];
  ]
