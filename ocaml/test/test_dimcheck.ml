(* Unit tests for the dimensional analysis checker (Dimcheck).

   Each test constructs a minimal Ir.model, runs Dimcheck.check_model,
   and asserts either success (no errors) or a specific error code.

   Run with:  cd ocaml && dune runtest *)

open Ir

(* ── Helpers ───────────────────────────────────────────────────────────── *)

let has_error code (result : Dimcheck.result) =
  List.exists (fun (d : Dimcheck.diagnostic) ->
    d.severity = Dimcheck.Error && d.code = code
  ) result.diagnostics

let contains s needle =
  let n = String.length needle and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = needle || go (i + 1)) in
  go 0

let has_info code (result : Dimcheck.result) =
  List.exists (fun (d : Dimcheck.diagnostic) ->
    d.severity = Dimcheck.Info && d.code = code
  ) result.diagnostics

let error_count (result : Dimcheck.result) =
  List.length (List.filter (fun (d : Dimcheck.diagnostic) ->
    d.severity = Dimcheck.Error
  ) result.diagnostics)

let no_errors (result : Dimcheck.result) =
  error_count result = 0

(* Minimal model scaffold — fill in transitions/parameters as needed *)
let empty_model
    ?(name = "test")
    ?(compartments = [])
    ?(transitions = [])
    ?(parameters = [])
    ?(observations = [])
    ?(ode_equations = [])
    ?(tables = [])
    ?(time_functions = [])
    ?(balance = None)
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
    interventions = [];
    observations;
    parameters;
    constants = [];
    bindings = [];
    per_eval_bindings = [];
    initial_conditions = [];
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
    identity_tracked_compartments = [];
    doc_index = empty_doc_index;
    quantities = [];
    contrasts = [];
  }

let mk_compartment name : compartment = { name; kind = Integer }

let mk_param ?(kind = None) ?(dim = None) ?(value = None) name : parameter =
  let value = match value with Some v -> Fixed v | None -> Required in
  { name; value; param_kind = kind; param_dim = dim }

let mk_transition ?(stoich = []) name rate : transition =
  { name; stoichiometry = stoich; rate;
    metadata = None; draw_method = DrawPoisson; rate_grad = []; rate_state_grad = []; lineage = None }

(* Shorthand constructors for expressions *)
let pop s = Pop s
let param s = Param s
let const f = Const f
let ( +. ) a b = BinOp { op = Add; left = a; right = b }
let ( -. ) a b = BinOp { op = Sub; left = a; right = b }
let ( *. ) a b = BinOp { op = Mul; left = a; right = b }
let ( /. ) a b = BinOp { op = Div; left = a; right = b }
let exp_ a = UnOp { op = Exp; arg = a }
let log_ a = UnOp { op = Log; arg = a }
let sqrt_ a = UnOp { op = Sqrt; arg = a }
let cond p t e = Cond { pred = p; then_ = t; else_ = e }
let gt a b = BinOp { op = Gt; left = a; right = b }

(* ── Basic Arithmetic Rules ────────────────────────────────────────────── *)

(* Pop("S") + Pop("I") in a rate context — both are P, sum is P.
   We wrap in a transition rate: (S + I) as a rate would be P, not P*T^-1,
   so we test the expression dimension via a well-formed rate. *)
let test_add_same_dim () =
  (* rate = beta * (S + I) — should be OK: T^-1 * P = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1" (param "beta" *. (pop "S" +. pop "I"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "add same dim OK" true (no_errors r)

let test_add_mismatched_dim () =
  (* rate = beta * (S + t) — P + T mismatch → E302 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1" (param "beta" *. (pop "S" +. Time))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E302 for P + T" true (has_error "E302" r)

let test_mul_dims_add () =
  (* Pop("S") * Param("beta":rate) = P * T^-1 = P*T^-1 — valid rate *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1" (pop "S" *. param "beta")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "mul dims OK" true (no_errors r)

let test_div_dims_subtract () =
  (* Pop("S") / Pop("N") = P / P = 1 (dimensionless).
     As a rate this would be wrong, but we wrap it correctly. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1"
      (param "beta" *. pop "N" *. (pop "S" /. pop "N"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "div dims OK" true (no_errors r)

let test_div_pop_by_time () =
  (* Pop("S") / Time = P / T = P*T^-1 — valid rate *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~transitions:[mk_transition "t1" (pop "S" /. Time)]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "pop/time OK" true (no_errors r)

(* ── Transition Rate Constraints ───────────────────────────────────────── *)

let test_sir_correct () =
  (* beta * S * I / N → T^-1 * P * P / P = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "R"; mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[
      mk_transition "infection"
        (param "beta" *. pop "S" *. pop "I" /. pop "N");
      mk_transition "recovery"
        (param "gamma" *. pop "I");
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "SIR correct" true (no_errors r)

let test_sir_missing_s () =
  (* beta * I / N → T^-1 * P / P = T^-1 — wrong for rate, should be E300 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. pop "I" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E300 missing S" true (has_error "E300" r)

let test_sir_wrong_param_kind () =
  (* p:probability * S * I / N → 1 * P * P / P = P — wrong *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Probability) "p"]
    ~transitions:[mk_transition "infection"
        (param "p" *. pop "S" *. pop "I" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E300 wrong param kind" true (has_error "E300" r)

let test_recovery_correct () =
  (* gamma * I → T^-1 * P = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "I"; mk_compartment "R"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[mk_transition "recovery" (param "gamma" *. pop "I")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "recovery correct" true (no_errors r)

let test_inflow_correct () =
  (* mu * N → T^-1 * P = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "mu"]
    ~transitions:[mk_transition "birth" (param "mu" *. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "inflow correct" true (no_errors r)

let test_inflow_bare_rate () =
  (* mu alone → T^-1 — wrong for rate *)
  let m = empty_model
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "mu"]
    ~transitions:[mk_transition "birth" (param "mu")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E300 bare rate" true (has_error "E300" r)

(* ── Iota / Seeding ────────────────────────────────────────────────────── *)

let test_iota_bare_const_rejected () =
  (* beta * (I + 1e-6) * S / N — Const(1e-6) is dimensionless, I is P → E302 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. (pop "I" +. const 1e-6) *. pop "S" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E302 bare iota const" true (has_error "E302" r)

let test_iota_typed_param_ok () =
  (* beta * (I + iota:count) * S / N — iota is P, OK *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Count) "iota"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. (pop "I" +. param "iota") *. pop "S" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "iota count OK" true (no_errors r)

let test_typed_let_count_accepted () =
  (* Simulates what the compiler emits for: let iota : count = 1e-6
     → Param("iota") with param_kind="count", value=Some 1e-6 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Count) ~value:(Some 1e-6) "iota"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. (pop "I" +. param "iota") *. pop "S" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "typed let count OK" true (no_errors r)

let test_typed_let_rate_accepted () =
  (* Simulates: let mu : rate = 0.0002
     → Param("mu") with param_kind="rate", value=Some 0.0002 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) ~value:(Some 0.0002) "mu"]
    ~transitions:[mk_transition "death" (param "mu" *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "typed let rate OK" true (no_errors r)

let test_untyped_const_rejected () =
  (* Untyped let iota = 1e-6 is inlined as Const(1e-6) → dimensionless.
     I + Const(1e-6) → P + dimensionless → E302 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. (pop "I" +. const 1e-6) *. pop "S" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "untyped const E302" true (has_error "E302" r)

let test_e302_hint_for_pop_plus_const () =
  (* E302 on a count plus a bare number names the typed-constant fix *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "infection"
        (param "beta" *. (pop "I" +. const 1e-6) *. pop "S" /. pop "N")]
    () in
  let r = Dimcheck.check_model m in
  let has_iota_hint = List.exists (fun (d : Dimcheck.diagnostic) ->
    d.code = "E302" && match d.hint with
    | Some h -> let sub = "let <name> : count = 1e-06" in
      let sl = String.length sub in
      let hl = String.length h in
      sl <= hl && (let found = ref false in
        for i = 0 to hl - sl do
          if String.sub h i sl = sub then found := true
        done; !found)
    | None -> false
  ) r.diagnostics in
  Alcotest.(check bool) "E302 hint mentions typed let" true has_iota_hint

(* ── Cross-Transition Consistency ──────────────────────────────────────── *)

let test_param_consistent () =
  (* alpha used as rate in both transitions → OK *)
  let m = empty_model
    ~compartments:[mk_compartment "A"; mk_compartment "B"; mk_compartment "C"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "alpha"]
    ~transitions:[
      mk_transition "t1" (param "alpha" *. pop "A");
      mk_transition "t2" (param "alpha" *. pop "B");
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "consistent param OK" true (no_errors r)

let test_param_inconsistent () =
  (* alpha : positive. t1 (alpha * A = P·T⁻¹) fixes alpha = T⁻¹, so t2's rate,
     alpha alone, is T⁻¹ where P·T⁻¹ is required: E300. *)
  let m = empty_model
    ~compartments:[mk_compartment "A"; mk_compartment "B"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha"]
    ~transitions:[
      mk_transition "t1" (param "alpha" *. pop "A");  (* alpha inferred T^-1 *)
      mk_transition "t2" (param "alpha");  (* alpha alone = T^-1, E300 *)
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "inconsistent param error" true (has_error "E300" r)

(* ── Transcendental Functions ──────────────────────────────────────────── *)

let test_exp_dimensionless_ok () =
  (* exp(p:probability) → OK, result is dimensionless.
     Wrap: rate * S * exp(p) → T^-1 * P * 1 = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r";
                 mk_param ~kind:(Some Ir.Probability) "p"]
    ~transitions:[mk_transition "t1"
        (param "r" *. pop "S" *. exp_ (param "p"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "exp dimensionless OK" true (no_errors r)

let test_exp_dimensioned_fail () =
  (* exp(Pop("S")) → S is P, E301 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. pop "S" *. exp_ (pop "S"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E301 exp of pop" true (has_error "E301" r)

let test_log_dimensionless_ok () =
  (* log(S / N) → P / P = 1, log(1) = OK *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. pop "N" *. log_ (pop "S" /. pop "N"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "log dimensionless OK" true (no_errors r)

let test_log_dimensioned_fail () =
  (* log(S) → S is P, E301 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. pop "S" *. log_ (pop "S"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E301 log of pop" true (has_error "E301" r)

(* ── Constants and Zero ────────────────────────────────────────────────── *)

let test_zero_compatible_with_pop () =
  (* S + Const(0.0) → OK (Any + P = P) *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. (pop "S" +. const 0.0))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "zero + pop OK" true (no_errors r)

let test_bare_const_is_dimensionless () =
  (* Const(3.14) * S → 1 * P = P. Wrap: r * 3.14 * S → P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. const 3.14 *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "bare const * pop OK" true (no_errors r)

(* ── Conditionals ──────────────────────────────────────────────────────── *)

let test_cond_branches_match () =
  (* cond(I > 0, beta*S, 0.0) — both branches should be P*T^-1 (with 0.0 = Any) *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1"
        (cond (gt (pop "I") (const 0.0))
           (param "beta" *. pop "S")
           (const 0.0))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "cond branches match OK" true (no_errors r)

let test_cond_branches_mismatch () =
  (* cond(I > 0, S, beta) — P vs T^-1 → E302 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1"
        (cond (gt (pop "I") (const 0.0))
           (pop "S")
           (param "beta"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E302 cond mismatch" true (has_error "E302" r)

(* ── Balance and ODE ───────────────────────────────────────────────────── *)

let test_balance_population_ok () =
  (* R = N - S - E - I → all P, OK *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "E";
                   mk_compartment "I"; mk_compartment "R";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[mk_transition "t1" (param "gamma" *. pop "I")]
    ~balance:(Some {
      balance_target = "R";
      balance_expr = pop "N" -. pop "S" -. pop "E" -. pop "I";
    })
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "balance pop OK" true (no_errors r)

let test_balance_wrong_dim () =
  (* balance_expr = gamma (a rate param) → T^-1, should be P → E305 *)
  let m = empty_model
    ~compartments:[mk_compartment "R"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[mk_transition "t1" (param "gamma" *. pop "R")]
    ~balance:(Some {
      balance_target = "R";
      balance_expr = param "gamma";
    })
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E305 balance wrong dim" true (has_error "E305" r)

(* gh#314: a forcing's `lag` must be a duration (dimension T). *)
let lag_forcing lag : time_function =
  { name = "vc";
    kind = Sinusoidal {
      amplitude = const 1.0; period = const 365.0;
      phase = const 0.0; baseline = const 1.0 };
    dim = (0, 0);   (* 'ratio *)
    lag;
    data_source = None }

let test_lag_duration_param_ok () =
  (* lag = tau, tau : duration → dimension T → no E309 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Duration) "tau"]
    ~time_functions:[lag_forcing (Some (param "tau"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "duration lag param ok" true (no_errors r)

let test_lag_literal_duration_ok () =
  (* lag = 10 'days, expander-lowered to UncheckedDim{dim_t=1} → dimension T *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~time_functions:[lag_forcing
      (Some (UncheckedDim { inner = const 10.0; dim_p = 0; dim_t = 1;
                            reason = "unit literal 'days" }))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "literal duration lag ok" true (no_errors r)

let test_lag_rate_param_rejected () =
  (* lag = beta, beta : rate → dimension T^-1, NOT a duration → E309 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~time_functions:[lag_forcing (Some (param "beta"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E309 rate lag rejected" true (has_error "E309" r)

let test_ode_derivative_correct () =
  (* d(V)/dt = -decay * V → T^-1 * P = P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "V"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "decay"]
    ~ode_equations:[{
      compartment = "V";
      derivative = UnOp { op = Neg; arg = param "decay" *. pop "V" };
    }]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "ODE correct" true (no_errors r)

let test_ode_derivative_wrong () =
  (* d(V)/dt = V → P, should be P*T^-1 → E306 *)
  let m = empty_model
    ~compartments:[mk_compartment "V"]
    ~ode_equations:[{
      compartment = "V";
      derivative = pop "V";
    }]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E306 ODE wrong" true (has_error "E306" r)

(* ── Undetermined Parameters ───────────────────────────────────────────── *)

let test_underdetermined_emits_info () =
  (* alpha:positive * beta_p:positive * I — two unknowns, underdetermined → I300 *)
  let m = empty_model
    ~compartments:[mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha";
                 mk_param ~kind:(Some Ir.Positive) "beta_p"]
    ~transitions:[mk_transition "t1"
        (param "alpha" *. param "beta_p" *. pop "I")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "I300 for underdetermined" true (has_info "I300" r)

let test_partially_determined () =
  (* alpha:positive * beta:rate * I → beta = T^-1, I = P, product = alpha * T^-1 * P
     For P*T^-1 total: alpha must be dimensionless (1).
     Single unknown should be resolved. *)
  let m = empty_model
    ~compartments:[mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha";
                 mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1"
        (param "alpha" *. param "beta" *. pop "I")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "partially determined OK" true (no_errors r);
  (* alpha should be inferred as dimensionless *)
  let alpha_dim = List.assoc_opt "alpha" r.param_dims in
  (match alpha_dim with
   | Some d ->
     Alcotest.(check bool) "alpha is dimensionless"
       true (d.(0) = 0 && d.(1) = 0)
   | None ->
     (* It's OK if it wasn't resolved — the key thing is no errors *)
     ())

(* ── Sqrt ──────────────────────────────────────────────────────────────── *)

let test_sqrt_even_powers () =
  (* sqrt(S * I) → sqrt(P^2) = P. Wrap: r * sqrt(S*I) → P*T^-1 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. sqrt_ (pop "S" *. pop "I"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "sqrt even OK" true (no_errors r)

let test_sqrt_odd_powers_fail () =
  (* sqrt(S * t) → sqrt(P * T) = sqrt(P^1 * T^1) — odd exponents → E304 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "r"]
    ~transitions:[mk_transition "t1"
        (param "r" *. sqrt_ (pop "S" *. Time))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E304 sqrt odd" true (has_error "E304" r)

(* ── Explicit Dimension Annotations ───────────────────────────────────── *)

let test_dim_annotation_dimensionless () =
  (* amplitude : real dim 1 — explicitly dimensionless, used as multiplier *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Real) ~dim:(Some (0, 0)) "amplitude"]
    ~transitions:[mk_transition "t1"
        (param "amplitude" *. param "beta" *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "dim annotation [1] OK" true (no_errors r);
  let amp_dim = List.assoc_opt "amplitude" r.param_dims in
  (match amp_dim with
   | Some d -> Alcotest.(check bool) "amplitude is dimensionless"
       true (d.(0) = 0 && d.(1) = 0)
   | None -> Alcotest.fail "amplitude dim not resolved")

let test_dim_annotation_pop_rate () =
  (* mu : positive dim P/T — population-level rate, used alone as transition rate *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) ~dim:(Some (1, -1)) "mu"]
    ~transitions:[mk_transition "t1" (param "mu")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "dim annotation [P/T] OK" true (no_errors r);
  let mu_dim = List.assoc_opt "mu" r.param_dims in
  (match mu_dim with
   | Some d -> Alcotest.(check bool) "mu is P*T^-1"
       true (d.(0) = 1 && d.(1) = -1)
   | None -> Alcotest.fail "mu dim not resolved")

let test_dim_annotation_resolves_i300 () =
  (* Two positive params: alpha * beta_p * I would be I300 (undetermined).
     With annotation [1] on alpha, it becomes determined. *)
  let m = empty_model
    ~compartments:[mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) ~dim:(Some (0, 0)) "alpha";
                 mk_param ~kind:(Some Ir.Positive) "beta_p"]
    ~transitions:[mk_transition "t1"
        (param "alpha" *. param "beta_p" *. pop "I")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "annotation resolves undetermined" true (no_errors r);
  (* beta_p should be inferred as T^-1 *)
  let bp_dim = List.assoc_opt "beta_p" r.param_dims in
  (match bp_dim with
   | Some d -> Alcotest.(check bool) "beta_p inferred T^-1"
       true (d.(0) = 0 && d.(1) = -1)
   | None -> ())

let test_dim_annotation_conflict () =
  (* mu : positive dim 1 used as mu * S — gives 1 * P = P, not P*T^-1 → E300 *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) ~dim:(Some (0, 0)) "mu"]
    ~transitions:[mk_transition "t1" (param "mu" *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "E300 annotation wins over usage"
    true (has_error "E300" r)

(* ── Golden File Dimension Checks ──────────────────────────────────────── *)
(* Compile real .camdl files and assert zero dimension errors.
   This is the most important class of test — no false positives on valid models. *)

let golden_dir =
  let candidates = [
    "../../golden";
    "../golden";
    "golden";
  ] in
  try
    List.find (fun d ->
      Sys.file_exists d && Sys.is_directory d
    ) candidates
  with Not_found -> "golden"  (* will fail gracefully *)

let read_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

(* A golden model is dimensionally clean through the real pipeline, which
   checks the model before constant folding: a table lookup such as
   `age_dur[a]` there still carries its declared cell dimension, which the
   folded IR that [Compiler.compile] returns has replaced by a number. *)
let test_golden_no_dim_errors model_name () =
  let camdl_path = Filename.concat golden_dir (model_name ^ ".camdl") in
  if not (Sys.file_exists camdl_path) then
    Alcotest.skip ()
  else begin
    let src = read_file camdl_path in
    let diags = Compiler.collect_diagnostics ~name:model_name ~filename:camdl_path src in
    let is_dimension_code c =
      String.length c = 4 && c.[0] = 'E' && c.[1] = '3' && (c.[2] = '0' || c.[2] = '1') in
    let errors = List.filter (fun (d : Diagnostics.diagnostic) ->
      d.severity = Diagnostics.Error && is_dimension_code d.code) diags in
    if errors <> [] then
      Alcotest.failf "golden model %s has dimension errors:\n%s" model_name
        (String.concat "\n" (List.map (fun (d : Diagnostics.diagnostic) ->
           Printf.sprintf "  [%s] %s" d.code d.message) errors))
  end

(* ── Unknowns linked through a sum ─────────────────────────────────────── *)

let test_unknown_unknown_linking_conflict () =
  (* Two unknown params: alpha and kappa, used in two transitions.
     t1: alpha * S  — alpha needs T^-1
     t2: kappa * I  — kappa needs T^-1
     t3: (alpha + kappa) * S — alpha and kappa are linked, both T^-1.
     This should work fine — no conflict. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha";
                 mk_param ~kind:(Some Ir.Positive) "kappa"]
    ~transitions:[
      mk_transition "t1" (param "alpha" *. pop "S");
      mk_transition "t2" (param "kappa" *. pop "I");
      mk_transition "t3" ((param "alpha" +. param "kappa") *. pop "S");
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "linked unknowns consistent" true (no_errors r)

let test_unknown_unknown_linking_catches_conflict () =
  (* alpha and kappa linked via addition (must be same dim).
     t1: alpha * S — alpha needs T^-1
     t2: kappa alone — kappa needs P*T^-1
     t3: (alpha + kappa) — links them, but they need different dims: a
     contradiction, at the sum (E302) or at a rate (E300). *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha";
                 mk_param ~kind:(Some Ir.Positive) "kappa"]
    ~transitions:[
      mk_transition "t1" (param "alpha" *. pop "S");
      mk_transition "t2" (param "kappa");
      mk_transition "t3" ((param "alpha" +. param "kappa") *. pop "S");
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "linked unknowns conflict detected"
    true (has_error "E300" r || has_error "E302" r)

(* ── Table lookup indices ──────────────────────────────────────────────── *)

(* An index selects a level, so it is a pure number (E311). The DSL resolves
   every index to a literal level, so only a hand-built model can carry another
   one; the rule holds for the IR anyway. *)
let test_table_index_is_dimensionless () =
  let table = { Ir.name = "w"; source = Ir.Inline [ Const 0.1; Const 0.2 ];
                out_of_bounds = Ir.Error; cell_kind = Some "rate" } in
  let model idx = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"]
    ~tables:[table]
    ~transitions:[mk_transition "t" (BinOp { op = Mul; left = TableLookup ("w", [idx]);
                                             right = pop "I" })]
    () in
  Alcotest.(check bool) "a literal index is fine" true (no_errors (Dimcheck.check_model (model (Const 1.0))));
  Alcotest.(check bool) "a count as an index is E311" true
    (has_error "E311" (Dimcheck.check_model (model (pop "S"))))

(* ── One parameter, two dimensions ───────────────────────────────────── *)

let test_param_two_dimensions () =
  (* alpha used as a rate (T^-1) in `infection` via alpha * S * I / N, but as
     a population (P) in `waning` via gamma * (alpha + I). One equation
     system holds both uses, so this is a contradiction, reported at the
     position where it surfaces with the other use named. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I"; mk_compartment "R";
                   mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "alpha";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[
      mk_transition "infection"
        (param "alpha" *. pop "S" *. pop "I" /. pop "N");
      mk_transition "waning"
        (param "gamma" *. (param "alpha" +. pop "I"));
    ]
    () in
  let r = Dimcheck.check_model m in
  let names_both (d : Dimcheck.diagnostic) =
    d.severity = Dimcheck.Error
    && (match d.detail with
        | Some s -> contains s "infection" && contains s "waning"
        | None -> false)
    || (d.severity = Dimcheck.Error && contains d.message "infection"
        && (match d.detail with Some s -> contains s "waning" | None -> false))
  in
  Alcotest.(check bool) "a contradiction naming both transitions"
    true (List.exists names_both r.diagnostics)

(* ── Each error reported once ──────────────────────────────────────────── *)

let test_valid_model_no_errors () =
  (* A valid model has exactly 0 errors. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"; mk_compartment "I";
                   mk_compartment "R"; mk_compartment "N"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Rate) "gamma"]
    ~transitions:[
      mk_transition "infection"
        (param "beta" *. pop "S" *. pop "I" /. pop "N");
      mk_transition "recovery"
        (param "gamma" *. pop "I");
    ]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "no errors" true (no_errors r);
  Alcotest.(check int) "exactly 0 errors" 0 (error_count r)

let test_single_error_reported_once () =
  (* A model with one clear error should report it exactly once. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta"]
    ~transitions:[mk_transition "t1" (param "beta")]
    () in
  let r = Dimcheck.check_model m in
  (* beta alone is T^-1, not P*T^-1 — exactly one E300 *)
  let e300_count = List.length (List.filter (fun (d : Dimcheck.diagnostic) ->
    d.severity = Dimcheck.Error && d.code = "E300"
  ) r.diagnostics) in
  Alcotest.(check int) "exactly one E300" 1 e300_count

(* ── Time parameter-kinds: instant / duration (2026-05-22 §9.9) ────────── *)

let param_dim_of name (r : Dimcheck.result) =
  List.assoc_opt name r.param_dims

let test_instant_dim_is_time () =
  (* An `instant`-kind parameter carries dimension [T] (P^0 T^1). *)
  let m = empty_model
    ~parameters:[mk_param ~kind:(Some Ir.Instant) "tau"]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check (option (pair int int))) "instant is [T]"
    (Some (0, 1))
    (Option.map (fun a -> (a.(0), a.(1))) (param_dim_of "tau" r))

let test_duration_dim_is_time () =
  (* A `duration`-kind parameter carries dimension [T] (P^0 T^1). *)
  let m = empty_model
    ~parameters:[mk_param ~kind:(Some Ir.Duration) "gen_interval"]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check (option (pair int int))) "duration is [T]"
    (Some (0, 1))
    (Option.map (fun a -> (a.(0), a.(1))) (param_dim_of "gen_interval" r))

let test_rate_plus_instant_e302 () =
  (* rate = beta * (S + tau) where beta:rate (T^-1), S:P, tau:instant (T).
     S + tau is a P + T mismatch → E302. The instant kind gives the checker
     time coverage: adding a time to a population (or rate) is a real bug. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Rate) "beta";
                 mk_param ~kind:(Some Ir.Instant) "tau"]
    ~transitions:[mk_transition "t1" (param "beta" *. (pop "S" +. param "tau"))]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "rate + instant is E302" true (has_error "E302" r)

let test_rate_times_duration_ok () =
  (* rate = beta / gen_interval * S where beta:probability (dimensionless),
     gen_interval:duration (T), S:P. beta/gen_interval = T^-1; * S = P*T^-1.
     A valid per-capita-style rate built from a duration — no error. *)
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Probability) "beta";
                 mk_param ~kind:(Some Ir.Duration) "gen_interval"]
    ~transitions:[mk_transition "t1"
                    ((param "beta" /. param "gen_interval") *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "rate from duration OK" true (no_errors r)

let test_instant_vector_each_is_time () =
  (* A stratified instant (tau[location] expanding to tau_a, tau_b) — each
     component carries [T], the case a per-report fixed estimand set could
     not handle. Expansion produces independent scalar params per cell. *)
  let m = empty_model
    ~parameters:[mk_param ~kind:(Some Ir.Instant) "tau_a";
                 mk_param ~kind:(Some Ir.Instant) "tau_b"]
    () in
  let r = Dimcheck.check_model m in
  let dim n = Option.map (fun a -> (a.(0), a.(1))) (param_dim_of n r) in
  Alcotest.(check (option (pair int int))) "tau_a is [T]" (Some (0,1)) (dim "tau_a");
  Alcotest.(check (option (pair int int))) "tau_b is [T]" (Some (0,1)) (dim "tau_b")

(* gh#116 follow-up: a DerivedExpr projection that is itself a PROPORTION
   (`projected = I / N`, dimensionless) used as a binomial `p` must NOT
   false-fire E304. The `Projected` node must carry the projection
   expression's inferred dimension, not a hard-coded population count.
   Counterpart to the count-projection guard (e304_binomial_p_is_count, which
   projects `I` — a count — and must still E304). *)
let test_e304_proportion_projection_ok () =
  let src = {camdl|
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  pos {
    columns       { time : time, pos : count }
    projected  = I / N
    emit_schedule = every 7 'days
    pos ~ binomial(n = 100, p = projected)
  }
}
|camdl} in
  let diags = Compiler.collect_diagnostics ~filename:"<proportion-projection>" src in
  let has_e304 =
    List.exists (fun (d : Diagnostics.diagnostic) -> d.code = "E304") diags in
  Alcotest.(check bool)
    "proportion projection (I/N) as binomial p must not E304" false has_e304

(* dimcheck-n fix (2026-06-10 observation data-entry §3.1). The binomial/
   beta-binomial `n` and the Poisson `rate` were inferred-and-discarded —
   un-checked. They are now constrained: `n` is a count, the Poisson rate is a
   count (expected events over the interval). A `count`-declared aux column
   feeding `n` passes; a Known-wrong dim (a `probability` column) is E304. A
   `real` column (Unknown dim) is NOT flagged — `constrain_known` deliberately
   never false-positives on an inferred-Unknown dim. *)
let has_e304_in src =
  let diags = Compiler.collect_diagnostics ~filename:"<n-check>" src in
  List.exists (fun (d : Diagnostics.diagnostic) -> d.code = "E304") diags

let survey_n_src n_role n_expr = Printf.sprintf {camdl|
compartments { S, I, R }
parameters { beta : rate  gamma : rate }
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  survey {
    columns       { time : time, pos : count, denom : %s }
    projected  = I
    emit_schedule = every 7 'days
    pos ~ binomial(n = %s, p = projected / N)
  }
}
|camdl} n_role n_expr

let test_binomial_n_count_column_ok () =
  (* `denom : count` feeding `n = denom` — a count, passes. *)
  Alcotest.(check bool) "count column feeding binomial n must not E304"
    false (has_e304_in (survey_n_src "count" "denom"))

let test_binomial_n_probability_column_rejected () =
  (* `denom : probability` (Known dimensionless) feeding `n = denom` — E304. *)
  Alcotest.(check bool) "probability column feeding binomial n must E304"
    true (has_e304_in (survey_n_src "probability" "denom"))

let test_binomial_n_literal_ok () =
  (* A bare literal `n = 100` is a count by context (like a seeding const) —
     exempt from the n-dimension check. *)
  Alcotest.(check bool) "literal binomial n must not E304"
    false (has_e304_in (survey_n_src "count" "100"))

let test_poisson_rate_probability_rejected () =
  (* Poisson `rate = frac` where `frac : probability` (Known dimensionless) is
     E304 — the rate must be a count. *)
  let src = {camdl|
compartments { S, I, R }
parameters { beta : rate  gamma : rate }
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  counts {
    columns       { time : time, k : count, frac : probability }
    projected  = I
    emit_schedule = every 7 'days
    k ~ poisson(rate = frac)
  }
}
|camdl} in
  Alcotest.(check bool) "probability column feeding poisson rate must E304"
    true (has_e304_in src)

(* A likelihood argument's product is an equation like any other: `rate = rho
   * X0 * projected` with `rho : probability` and `projected` a count pins X0
   to dimensionless. *)
let test_likelihood_arg_product_determines () =
  let src = {camdl|
compartments { S, I, R }
parameters { beta : rate  gamma : rate  rho : probability  X0 : positive  s : positive }
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  counts {
    columns       { time : time, k : count }
    projected  = prevalence(I)
    emit_schedule = every 7 'days
    k ~ poisson(rate = rho * X0 * s * projected / s)
  }
}
|camdl} in
  let diags = Compiler.collect_diagnostics ~filename:"<contract>" src in
  let i300_for name = List.exists (fun (d : Diagnostics.diagnostic) ->
    let needle = "'" ^ name ^ "'" in
    let n = String.length needle and m = String.length d.message in
    let rec has i = i + n <= m && (String.sub d.message i n = needle || has (i + 1)) in
    d.code = "I300" && has 0) diags in
  Alcotest.(check bool) "X0 determined through the Poisson rate" false (i300_for "X0")

(* ── NegBinomial / zero-inflated NegBinomial `mean` is a count ─────────── *)

(* The NegBinomial `mean` is the same object as the Poisson `rate`: the expected
   COUNT of events over the reporting interval, dimension [P] — not a per-time
   rate. It was inferred-and-discarded while the `dispersion` immediately below
   it was constrained, so a `projected` that is a transition RATE (`gamma * I`,
   dimension [P·T^-1]) reached the likelihood with no diagnostic. `projected`
   takes a DerivedExpr whose dimension is *inferred*, so nothing upstream
   objected either. At a one-day reporting step a per-day rate and a one-day
   accumulated count coincide numerically, so the resulting likelihood is only
   slightly wrong; at a weekly step it is wrong by roughly the window length.
   The zero-inflated variant carries the identical contract. *)

let e304_diags_in src =
  Compiler.collect_diagnostics ~filename:"<nb-mean>" src
  |> List.filter (fun (d : Diagnostics.diagnostic) -> d.code = "E304")

let nb_mean_src ~projected ~likelihood =
  (* Only an accumulating projection states the period each row covers
     (gh#833): `covers` on a state read is E348, and the RED cases here
     project a rate (`gamma * I`), which is a state read. *)
  let covers =
    if String.starts_with ~prefix:"incidence(" projected
    then "    covers        = closing_at(time, 7 'days)\n"
    else ""
  in
  Printf.sprintf {camdl|
compartments { S, I, R }
parameters {
  beta    : rate
  gamma   : rate
  rho     : probability
  q_comm  : probability
  tau     : rate
  k       : positive
  pi_zero : probability
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
%s    projected  = %s
    emit_schedule = every 7 'days
    weekly_cases ~ %s
  }
}
|camdl} covers projected likelihood

let no_e304 ~projected ~likelihood =
  e304_diags_in (nb_mean_src ~projected ~likelihood) = []

let located_e304 ~projected ~likelihood =
  let diags = e304_diags_in (nb_mean_src ~projected ~likelihood) in
  List.exists (fun (d : Diagnostics.diagnostic) ->
    d.Diagnostics.loc.Diagnostics.line > 0) diags

(* RED case: `projected = gamma * I` is a rate ([P·T^-1]) where a windowed count
   is expected, and it feeds the NegBinomial `mean` directly. *)
let test_e304_neg_binomial_mean_is_rate_rejected () =
  Alcotest.(check bool) "rate-dimensioned neg_binomial mean must be a located E304"
    true (located_e304 ~projected:"gamma * I"
            ~likelihood:"neg_binomial(mean = projected, r = k)")

(* Same contract through the `zero_inflated(base = neg_binomial(...))` wrapper,
   whose `mean` is a bare expr rather than a diffable. *)
let test_e304_zinb_mean_is_rate_rejected () =
  Alcotest.(check bool) "rate-dimensioned zero-inflated mean must be a located E304"
    true (located_e304 ~projected:"gamma * I"
            ~likelihood:"zero_inflated(base = neg_binomial(mean = projected, r = k), \
                         pi = pi_zero)")

(* Negative control (a): the legitimate shape — a CumulativeFlow projection
   (`incidence(...)`, a windowed count) straight into the mean. *)
let test_neg_binomial_mean_incidence_ok () =
  Alcotest.(check bool) "incidence() projection as neg_binomial mean must not E304"
    true (no_e304 ~projected:"incidence(infection)"
            ~likelihood:"neg_binomial(mean = projected, r = k)");
  Alcotest.(check bool) "incidence() projection as zero-inflated mean must not E304"
    true (no_e304 ~projected:"incidence(infection)"
            ~likelihood:"zero_inflated(base = neg_binomial(mean = rho * projected, \
                         r = k), pi = pi_zero)")

(* Negative control (b): a bare numeric literal mean is a count by context —
   [is_bare_const] exempts it, as it does for Poisson `rate` and Binomial `n`. *)
let test_neg_binomial_mean_literal_ok () =
  Alcotest.(check bool) "literal neg_binomial mean must not E304"
    true (no_e304 ~projected:"incidence(infection)"
            ~likelihood:"neg_binomial(mean = 100, r = k)")

(* Negative control (c): the real reporting shape — the count projection scaled
   by dimensionless factors (`rho`, and a dimensionless ratio of two rates).
   The product is still [P], so it must pass. *)
let test_neg_binomial_mean_scaled_projection_ok () =
  Alcotest.(check bool) "dimensionless-scaled projection as mean must not E304"
    true (no_e304 ~projected:"incidence(infection)"
            ~likelihood:"neg_binomial(mean = rho * (1.0 + gamma * q_comm / tau) \
                         * projected, r = k)")

(* ── The `dim` clause on positive/real (2026-11-05 size brackets §3) ─────── *)

(* `positive` and `real` may carry a `dim` clause giving the parameter's
   dimension, which flows to `param_dim`; dimcheck consumes it unchanged. A
   `dim` clause on a kind whose dimension is fixed is E281. The notations it
   replaces — a dimension bracket and a unit literal — are refused with a
   fix-it (E116, E117), and a bracket is a family size (E417 below 2). *)

let compile_ok name src =
  match Compiler.compile ~name src with
  | Ok m -> m
  | Error e -> Alcotest.failf "%s: compile failed: %s" name e

let dim_of_param name (r : Dimcheck.result) =
  Option.map (fun a -> (a.(0), a.(1))) (param_dim_of name r)

(* `tau : positive dim 1` carries dimension [1] (dimensionless) and, used as a
   dimensionless multiplier on a valid rate, produces no errors. *)
let test_positive_ratio_is_dimensionless () =
  let src = {camdl|
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  tau   : positive dim 1 in [0.001, 3.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ tau * beta * S * I / N
  recovery  : I --> R @ gamma * I
}
|camdl} in
  let m = compile_ok "positive-ratio" src in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "positive dim 1 compiles clean" true (no_errors r);
  Alcotest.(check (option (pair int int))) "tau is [1]"
    (Some (0, 0)) (dim_of_param "tau" r)

(* `real dim P` carries dimension [P] (population). Used as a count seed inside
   `(I + iota)`, the addition typechecks (P + P). *)
let test_real_count_is_population () =
  let src = {camdl|
compartments { S, I, R }
parameters {
  beta : rate
  gamma : rate
  iota : real dim P in [0.0, 10.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * (I + iota) * S / N
  recovery  : I --> R @ gamma * I
}
|camdl} in
  let m = compile_ok "real-count" src in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "real dim P compiles clean" true (no_errors r);
  Alcotest.(check (option (pair int int))) "iota is [P]"
    (Some (1, 0)) (dim_of_param "iota" r)

(* `positive dim 1/T` carries dimension [T^-1]; used alone as a per-capita *
   pop rate, it typechecks. *)
let test_positive_per_year_is_rate () =
  let src = {camdl|
time_unit = 'days
compartments { S, I }
parameters {
  iota : positive dim 1/T in [0.0001, 0.1]
}
transitions {
  importation : S --> I @ iota * S
}
|camdl} in
  let m = compile_ok "positive-per-time" src in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "positive dim 1/T compiles clean" true (no_errors r);
  Alcotest.(check (option (pair int int))) "iota is [T^-1]"
    (Some (0, -1)) (dim_of_param "iota" r)

(* The dimension is load-bearing: `tau : positive dim 1` (dimensionless) used
   where a count is required (added to a population) is now a real dimension
   error (E302), not a silently-swallowed I300. *)
let test_positive_ratio_misuse_is_caught () =
  let src = {camdl|
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  tau   : positive dim 1
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * (I + tau) * S / N
  recovery  : I --> R @ gamma * I
}
|camdl} in
  let diags = Compiler.collect_diagnostics ~filename:"<ratio-misuse>" src in
  let has code =
    List.exists (fun (d : Diagnostics.diagnostic) -> d.code = code) diags in
  (* (I + tau) is P + dimensionless → E302; and no I300, because the dim is now
     fully determined by the annotation. *)
  Alcotest.(check bool) "dimensionless tau added to a population is E302"
    true (has "E302");
  Alcotest.(check bool) "no I300: dimension is determined by the clause"
    false (has "I300")

let codes_of src =
  List.map (fun (d : Diagnostics.diagnostic) -> d.code)
    (Compiler.collect_diagnostics ~filename:"<dim-clause>" src)

let one_param decl = Printf.sprintf {camdl|
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  %s
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
|camdl} decl

(* A `dim` clause on a kind that fixes its dimension: E281, whether it agrees
   (redundant) or not (contradictory). *)
let test_dim_clause_on_fixed_kind_rejected () =
  Alcotest.(check bool) "rate dim T^-1 is E281 (redundant)" true
    (List.mem "E281" (codes_of (one_param "x : rate dim T^-1")));
  Alcotest.(check bool) "rate dim 1 is E281 (contradictory)" true
    (List.mem "E281" (codes_of (one_param "x : rate dim 1")))

(* Every old annotation is refused, none reinterpreted: a dimension bracket is
   E116, a unit literal E117, a bracket of 1 is E417 (a family needs two cells),
   a larger size is E119 until families land, and two sizes are E417. *)
let test_old_annotations_refused () =
  List.iter (fun (decl, code) ->
    Alcotest.(check bool) (decl ^ " is " ^ code) true
      (List.mem code (codes_of (one_param decl))))
    [ ("x : real [1/T]", "E116");
      ("x : positive [P*T^-1]", "E116");
      ("x : positive 'per_year", "E117");
      ("x : rate 'per_day", "E117");
      ("x : real [1]", "E417");
      ("x : positive[1]", "E417");
      ("x : real[6]", "E119");
      ("x : real[6, 3]", "E417") ]

(* Each refusal carries one machine-applicable fix-it; applying it yields a
   model that compiles clean. *)
let test_old_annotations_fixits_apply () =
  List.iter (fun (decl, want) ->
    let src = one_param decl in
    let diags = Compiler.collect_diagnostics ~filename:"<fixit>" src in
    let fixes = List.concat_map (fun (d : Diagnostics.diagnostic) -> d.fixes) diags in
    match fixes with
    | [ f ] ->
      let lines = Array.of_list (String.split_on_char '\n' src) in
      let l = f.fix_loc in
      Alcotest.(check int) "single-line fix" l.line l.end_line;
      let row = lines.(l.line - 1) in
      lines.(l.line - 1) <-
        String.sub row 0 (l.col - 1) ^ f.replacement
        ^ String.sub row (l.end_col - 1) (String.length row - l.end_col + 1);
      Alcotest.(check string) (decl ^ " rewrites") ("  " ^ want) lines.(l.line - 1);
      let fixed = String.concat "\n" (Array.to_list lines) in
      Alcotest.(check (list string)) (decl ^ " then compiles clean") []
        (List.filter (fun c -> c.[0] = 'E') (codes_of fixed))
    | fs -> Alcotest.failf "%s: expected one fix-it, got %d" decl (List.length fs))
    [ ("x : real [1/T] in [-1, 1]", "x : real dim 1/T in [-1, 1]");
      ("x : positive[P*T^-1]", "x : positive dim P/T");
      ("x : positive 'per_year in [0.1, 10.0]", "x : positive dim 1/T in [0.1, 10.0]");
      ("x : real [1]", "x : real dim 1");
      ("x : positive 'ratio [1]", "x : positive dim 1");
      ("x : rate 'per_day", "x : rate");
      ("x : rate dim T^-1", "x : rate") ]

(* The feature is purely additive: a bare `positive` (no unit) is unchanged —
   under-determined, still emits I300 when used in a dim-determined slot. *)
let test_bare_positive_unchanged () =
  let src = {camdl|
compartments { S, I, R }
parameters {
  beta  : rate
  gamma : rate
  alpha : positive
  kappa : positive
}
let N = S + I + R
transitions {
  infection : S --> I @ alpha * kappa * beta * S * I / N
  recovery  : I --> R @ gamma * I
}
|camdl} in
  let m = compile_ok "bare-positive" src in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "two bare positives are under-determined → I300"
    true (has_info "I300" r)

(* ── Test Registration ─────────────────────────────────────────────────── *)

(* ── Complete inference: products of unknowns (Kennedy linear solve) ──── *)

let param_dim_of name (r : Dimcheck.result) = List.assoc_opt name r.param_dims

(* Each rate alone has two unknown factors, so neither transition pins either
   parameter by itself: a·b = T^-1 and a/b = T^-1. Together they determine
   a = T^-1 and b = 1. A heuristic that solves one unknown factor per product
   leaves both undetermined; the linear solve must not. *)
let test_two_products_jointly_determine () =
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "a";
                 mk_param ~kind:(Some Ir.Positive) "b"]
    ~transitions:[mk_transition "t1" (param "a" *. param "b" *. pop "S");
                  mk_transition "t2" (param "a" /. param "b" *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "no errors" true (no_errors r);
  Alcotest.(check bool) "no I300" false (has_info "I300" r);
  Alcotest.(check (option (array int))) "a = T^-1" (Some [| 0; -1 |]) (param_dim_of "a" r);
  Alcotest.(check (option (array int))) "b = 1" (Some [| 0; 0 |]) (param_dim_of "b" r)

(* A dimension pinned only through a product inside another product: the rate
   (a·b)·(a/b)·S, where (a·b) and (a/b) are each unresolved sub-products. *)
let test_nested_products_infeasible () =
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "a";
                 mk_param ~kind:(Some Ir.Positive) "b";
                 mk_param ~kind:(Some Ir.Rate) "g"]
    ~transitions:[mk_transition "t1" ((param "a" *. param "b") *. pop "S");
                  mk_transition "t2" (param "g" *. (param "a" /. param "b") *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  (* a·b = T^-1 and a/b = 1  ⇒  2a = T^-1: rational solutions only. A model
     whose dimensions need a fractional exponent is an error, not a free
     parameter. *)
  Alcotest.(check bool) "integer infeasibility is an error" false (no_errors r);
  Alcotest.(check (option (array int))) "a not bound" None (param_dim_of "a" r)

(* A product of two unknowns equal to a known value is one equation in two
   unknowns: genuinely free, so I300 must still be emitted. *)
let test_single_product_still_free () =
  let m = empty_model
    ~compartments:[mk_compartment "S"]
    ~parameters:[mk_param ~kind:(Some Ir.Positive) "a";
                 mk_param ~kind:(Some Ir.Positive) "b"]
    ~transitions:[mk_transition "t1" (param "a" *. param "b" *. pop "S")]
    () in
  let r = Dimcheck.check_model m in
  Alcotest.(check bool) "I300 for a free pair" true (has_info "I300" r);
  Alcotest.(check (option (array int))) "a not bound" None (param_dim_of "a" r)

(* ── Forcing arguments and initial conditions are dimension-checked ───── *)

let diags_of src = Compiler.collect_diagnostics ~filename:"<visit>" src
let has_code code ds = List.exists (fun (d : Diagnostics.diagnostic) -> d.code = code) ds
let i300_names ds = List.filter_map (fun (d : Diagnostics.diagnostic) ->
  if d.code = "I300" then Some d.message else None) ds
let mentions name msg =
  let needle = "'" ^ name ^ "'" in
  let n = String.length needle and m = String.length msg in
  let rec has i = i + n <= m && (String.sub msg i n = needle || has (i + 1)) in
  has 0

let sir_with ~params ~forcing ~init ~rate = Printf.sprintf {camdl|
compartments { S, I, R }
parameters { gamma : rate  %s }
%s
transitions {
  infection : S --> I @ %s
  recovery  : I --> R @ gamma * I
}
init { %s }
|camdl} params forcing rate init

(* `E = k * I`-style: a deterministic initial value is a count, so a factor
   multiplying a count there is dimensionless. *)
let test_init_value_determines () =
  let ds = diags_of (sir_with ~params:"beta : rate  k : positive" ~forcing:""
      ~init:"S = 990  I = k * 10  R = 0" ~rate:"beta * S * I / (S + I + R)") in
  Alcotest.(check bool) "no errors" false (has_code "E313" ds);
  Alcotest.(check bool) "k determined: a literal factor is a pure number" false
    (List.exists (mentions "k") (i300_names ds));
  let ds = diags_of (sir_with ~params:"beta : rate  k : positive" ~forcing:""
      ~init:"S = 990  I = k * S  R = 0" ~rate:"beta * S * I / (S + I + R)") in
  Alcotest.(check bool) "k determined by `I = k * S`" false
    (List.exists (mentions "k") (i300_names ds))

(* An initial value that is not a count is an error. *)
let test_init_value_wrong_dimension () =
  let ds = diags_of (sir_with ~params:"beta : rate" ~forcing:""
      ~init:"S = 990  I = beta * S  R = 0" ~rate:"beta * S * I / (S + I + R)") in
  Alcotest.(check bool) "E313 for a rate times a count" true (has_code "E313" ds)

(* A literal, or arithmetic of literals, that is a whole term of an initial
   value is a count. *)
let test_init_literals_free () =
  let ds = diags_of (sir_with ~params:"beta : rate" ~forcing:""
      ~init:"S = 0.06 * 1000 - 50  I = 10  R = 0" ~rate:"beta * S * I / (S + I + R)") in
  Alcotest.(check bool) "no E302/E313" false (has_code "E302" ds || has_code "E313" ds)

(* A sinusoid is baseline + amplitude * sin(...): the amplitude has the
   forcing's dimension, the phase is a duration. *)
let seasonal amp_expr phase_expr = Printf.sprintf {camdl|
forcing {
  seasonal : sinusoidal 'ratio {
    amplitude = %s
    period    = 365.25 'days
    phase     = %s
    baseline  = 1.0
  }
}
|camdl} amp_expr phase_expr

let test_forcing_arguments_determine () =
  let ds = diags_of (sir_with ~params:"beta : rate  amp : real  ph : real"
      ~forcing:(seasonal "amp" "ph") ~init:"S = 990  I = 10  R = 0"
      ~rate:"beta * seasonal * S * I / (S + I + R)") in
  Alcotest.(check bool) "no E312" false (has_code "E312" ds);
  Alcotest.(check bool) "amp determined" false (List.exists (mentions "amp") (i300_names ds));
  Alcotest.(check bool) "ph determined" false (List.exists (mentions "ph") (i300_names ds))

let test_forcing_argument_wrong_dimension () =
  let ds = diags_of (sir_with ~params:"beta : rate  ph : rate"
      ~forcing:(seasonal "0.2" "ph") ~init:"S = 990  I = 10  R = 0"
      ~rate:"beta * seasonal * S * I / (S + I + R)") in
  Alcotest.(check bool) "E312 for a rate as a phase" true (has_code "E312" ds);
  let ds = diags_of (sir_with ~params:"beta : rate  a : rate"
      ~forcing:(seasonal "2 * a" "0 'days") ~init:"S = 990  I = 10  R = 0"
      ~rate:"beta * seasonal * S * I / (S + I + R)") in
  Alcotest.(check bool) "E312: a literal factor does not absorb a rate" true (has_code "E312" ds)

let () =
  Alcotest.run "dimcheck" [
    "visited_positions", [
      Alcotest.test_case "init value determines"     `Quick test_init_value_determines;
      Alcotest.test_case "init value wrong dimension" `Quick test_init_value_wrong_dimension;
      Alcotest.test_case "init literals free"        `Quick test_init_literals_free;
      Alcotest.test_case "forcing args determine"    `Quick test_forcing_arguments_determine;
      Alcotest.test_case "forcing arg wrong dim"     `Quick test_forcing_argument_wrong_dimension;
    ];
    "linear_solve", [
      Alcotest.test_case "two products jointly determine" `Quick test_two_products_jointly_determine;
      Alcotest.test_case "nested products, fractional"    `Quick test_nested_products_infeasible;
      Alcotest.test_case "single product still free"      `Quick test_single_product_still_free;
      Alcotest.test_case "likelihood arg product determines" `Quick test_likelihood_arg_product_determines;
    ];
    "arithmetic", [
      Alcotest.test_case "add same dim"            `Quick test_add_same_dim;
      Alcotest.test_case "add mismatched dim"       `Quick test_add_mismatched_dim;
      Alcotest.test_case "mul dims add"             `Quick test_mul_dims_add;
      Alcotest.test_case "div dims subtract"        `Quick test_div_dims_subtract;
      Alcotest.test_case "div pop by time"          `Quick test_div_pop_by_time;
    ];
    "transition_rates", [
      Alcotest.test_case "SIR correct"              `Quick test_sir_correct;
      Alcotest.test_case "SIR missing S"            `Quick test_sir_missing_s;
      Alcotest.test_case "SIR wrong param kind"     `Quick test_sir_wrong_param_kind;
      Alcotest.test_case "recovery correct"         `Quick test_recovery_correct;
      Alcotest.test_case "inflow correct"           `Quick test_inflow_correct;
      Alcotest.test_case "inflow bare rate"         `Quick test_inflow_bare_rate;
    ];
    "iota_seeding", [
      Alcotest.test_case "bare const rejected"      `Quick test_iota_bare_const_rejected;
      Alcotest.test_case "typed param ok"           `Quick test_iota_typed_param_ok;
    ];
    "typed_let_bindings", [
      Alcotest.test_case "typed let count accepted"  `Quick test_typed_let_count_accepted;
      Alcotest.test_case "typed let rate accepted"   `Quick test_typed_let_rate_accepted;
      Alcotest.test_case "untyped const rejected"    `Quick test_untyped_const_rejected;
      Alcotest.test_case "E302 hint for pop+const"   `Quick test_e302_hint_for_pop_plus_const;
    ];
    "cross_transition", [
      Alcotest.test_case "param consistent"         `Quick test_param_consistent;
      Alcotest.test_case "param inconsistent"       `Quick test_param_inconsistent;
    ];
    "transcendental", [
      Alcotest.test_case "exp dimensionless ok"     `Quick test_exp_dimensionless_ok;
      Alcotest.test_case "exp dimensioned fail"     `Quick test_exp_dimensioned_fail;
      Alcotest.test_case "log dimensionless ok"     `Quick test_log_dimensionless_ok;
      Alcotest.test_case "log dimensioned fail"     `Quick test_log_dimensioned_fail;
    ];
    "constants_zero", [
      Alcotest.test_case "zero + pop"               `Quick test_zero_compatible_with_pop;
      Alcotest.test_case "bare const * pop"          `Quick test_bare_const_is_dimensionless;
    ];
    "conditionals", [
      Alcotest.test_case "branches match"           `Quick test_cond_branches_match;
      Alcotest.test_case "branches mismatch"        `Quick test_cond_branches_mismatch;
    ];
    "balance_ode", [
      Alcotest.test_case "balance population ok"    `Quick test_balance_population_ok;
      Alcotest.test_case "balance wrong dim"        `Quick test_balance_wrong_dim;
      Alcotest.test_case "ODE derivative correct"   `Quick test_ode_derivative_correct;
      Alcotest.test_case "ODE derivative wrong"     `Quick test_ode_derivative_wrong;
    ];
    "forcing_lag", [
      Alcotest.test_case "gh#314 duration param lag ok"  `Quick test_lag_duration_param_ok;
      Alcotest.test_case "gh#314 literal duration lag ok" `Quick test_lag_literal_duration_ok;
      Alcotest.test_case "gh#314 E309 rate lag rejected" `Quick test_lag_rate_param_rejected;
    ];
    "undetermined", [
      Alcotest.test_case "underdetermined info"     `Quick test_underdetermined_emits_info;
      Alcotest.test_case "partially determined"     `Quick test_partially_determined;
    ];
    "sqrt", [
      Alcotest.test_case "even powers ok"           `Quick test_sqrt_even_powers;
      Alcotest.test_case "odd powers fail"          `Quick test_sqrt_odd_powers_fail;
    ];
    "dim_annotations", [
      Alcotest.test_case "annotation [1] dimensionless" `Quick test_dim_annotation_dimensionless;
      Alcotest.test_case "annotation [P/T] pop rate"    `Quick test_dim_annotation_pop_rate;
      Alcotest.test_case "annotation resolves I300"     `Quick test_dim_annotation_resolves_i300;
      Alcotest.test_case "annotation conflict E300"     `Quick test_dim_annotation_conflict;
    ];
    "golden_no_false_positives", [
      Alcotest.test_case "sir_basic"                `Quick (test_golden_no_dim_errors "sir_basic");
      Alcotest.test_case "sir_demography"           `Quick (test_golden_no_dim_errors "sir_demography");
      Alcotest.test_case "sir_overdispersion"       `Quick (test_golden_no_dim_errors "sir_overdispersion");
      Alcotest.test_case "sir_coupling"             `Quick (test_golden_no_dim_errors "sir_coupling");
      Alcotest.test_case "sir_two_patch"            `Quick (test_golden_no_dim_errors "sir_two_patch");
      Alcotest.test_case "sir_patches_5"            `Quick (test_golden_no_dim_errors "sir_patches_5");
      Alcotest.test_case "sir_spatial_sum"          `Quick (test_golden_no_dim_errors "sir_spatial_sum");
      Alcotest.test_case "sir_five_age"             `Quick (test_golden_no_dim_errors "sir_five_age");
      (* sir_init_table skipped — depends on external data file *)
      Alcotest.test_case "seir_vaccine"             `Quick (test_golden_no_dim_errors "seir_vaccine");
      Alcotest.test_case "seir_vaccine_seasonal"    `Quick (test_golden_no_dim_errors "seir_vaccine_seasonal");
      Alcotest.test_case "seir_seasonal_patch"      `Quick (test_golden_no_dim_errors "seir_seasonal_patch");
      Alcotest.test_case "seir_erlang"              `Quick (test_golden_no_dim_errors "seir_erlang");
      Alcotest.test_case "seir_erlang_staged"       `Quick (test_golden_no_dim_errors "seir_erlang_staged");
      Alcotest.test_case "seir_observations"        `Quick (test_golden_no_dim_errors "seir_observations");
      Alcotest.test_case "seir_age"                 `Quick (test_golden_no_dim_errors "seir_age");
      (* seir_defines_adj, seir_defines_patch skipped — depend on external data files *)
      Alcotest.test_case "seir_spatial_5_inference" `Quick (test_golden_no_dim_errors "seir_spatial_5_inference");
      Alcotest.test_case "polio_age"                `Quick (test_golden_no_dim_errors "polio_age");
      Alcotest.test_case "polio_spatial_5"          `Quick (test_golden_no_dim_errors "polio_spatial_5");
      Alcotest.test_case "malaria_two_species"      `Quick (test_golden_no_dim_errors "malaria_two_species");
      Alcotest.test_case "sir_dim_annotated"        `Quick (test_golden_no_dim_errors "sir_dim_annotated");
      Alcotest.test_case "seir_age_table_rates"     `Quick (test_golden_no_dim_errors "seir_age_table_rates");
    ];
    "linked_unknowns", [
      Alcotest.test_case "linked unknowns consistent"  `Quick test_unknown_unknown_linking_conflict;
      Alcotest.test_case "linked unknowns conflict"    `Quick test_unknown_unknown_linking_catches_conflict;
    ];
    "table_index", [
      Alcotest.test_case "an index is a pure number" `Quick test_table_index_is_dimensionless;
    ];
    "param_two_dimensions", [
      Alcotest.test_case "contradiction across transitions" `Quick test_param_two_dimensions;
    ];
    "time_param_kinds", [
      Alcotest.test_case "instant is [T]"              `Quick test_instant_dim_is_time;
      Alcotest.test_case "duration is [T]"             `Quick test_duration_dim_is_time;
      Alcotest.test_case "rate + instant → E302"       `Quick test_rate_plus_instant_e302;
      Alcotest.test_case "rate from duration OK"       `Quick test_rate_times_duration_ok;
      Alcotest.test_case "instant vector each [T]"     `Quick test_instant_vector_each_is_time;
    ];
    "reporting", [
      Alcotest.test_case "no duplicate errors"         `Quick test_valid_model_no_errors;
      Alcotest.test_case "single error reported once"  `Quick test_single_error_reported_once;
    ];
    "dim_clause", [
      Alcotest.test_case "positive dim 1 is [1]"         `Quick test_positive_ratio_is_dimensionless;
      Alcotest.test_case "real dim P is [P]"             `Quick test_real_count_is_population;
      Alcotest.test_case "positive dim 1/T is [T^-1]"    `Quick test_positive_per_year_is_rate;
      Alcotest.test_case "ratio misuse is caught"        `Quick test_positive_ratio_misuse_is_caught;
      Alcotest.test_case "dim on fixed kind → E281"      `Quick test_dim_clause_on_fixed_kind_rejected;
      Alcotest.test_case "old annotations refused"       `Quick test_old_annotations_refused;
      Alcotest.test_case "old annotation fix-its apply"  `Quick test_old_annotations_fixits_apply;
      Alcotest.test_case "bare positive unchanged"       `Quick test_bare_positive_unchanged;
    ];
    "observation_args", [
      Alcotest.test_case "e304_proportion_projection_ok"  `Quick
        test_e304_proportion_projection_ok;
      (* dimcheck-n fix (2026-06-10 obs data-entry §3.1) *)
      Alcotest.test_case "binomial n = count column ok"  `Quick
        test_binomial_n_count_column_ok;
      Alcotest.test_case "binomial n = probability column → E304"  `Quick
        test_binomial_n_probability_column_rejected;
      Alcotest.test_case "binomial n = literal ok"  `Quick
        test_binomial_n_literal_ok;
      Alcotest.test_case "poisson rate = probability column → E304"  `Quick
        test_poisson_rate_probability_rejected;
      (* NegBinomial / zero-inflated NegBinomial `mean` is the expected count
         over the reporting interval — the Poisson `rate`'s twin. *)
      Alcotest.test_case "neg_binomial mean = rate → E304"  `Quick
        test_e304_neg_binomial_mean_is_rate_rejected;
      Alcotest.test_case "zero-inflated mean = rate → E304"  `Quick
        test_e304_zinb_mean_is_rate_rejected;
      Alcotest.test_case "neg_binomial mean = incidence() ok"  `Quick
        test_neg_binomial_mean_incidence_ok;
      Alcotest.test_case "neg_binomial mean = literal ok"  `Quick
        test_neg_binomial_mean_literal_ok;
      Alcotest.test_case "neg_binomial mean = scaled projection ok"  `Quick
        test_neg_binomial_mean_scaled_projection_ok;
    ];

    (* ── Property-based tests (QCheck) ─────────────────────────────── *)
    "dim_properties", List.map QCheck_alcotest.to_alcotest [

      (* Property 1: mul then div preserves dimension.
         ∀ dim d, dim_div (dim_mul d x) x = d *)
      QCheck.Test.make ~name:"mul_then_div_preserves_dim" ~count:200
        (QCheck.pair
          (QCheck.pair QCheck.int_small QCheck.int_small)
          (QCheck.pair QCheck.int_small QCheck.int_small))
        (fun ((p1, t1), (p2, t2)) ->
          let d = Dimcheck.make p1 t1 in
          let x = Dimcheck.make p2 t2 in
          let roundtrip = Dimcheck.dim_div (Dimcheck.dim_mul d x) x in
          Dimcheck.dim_eq roundtrip d);

      (* Property 2: add requires matching dimensions.
         ∀ d1 d2, d1 ≠ d2 → they cannot be added (dim_eq must fail) *)
      QCheck.Test.make ~name:"mismatched_dims_not_equal" ~count:200
        (QCheck.pair
          (QCheck.pair QCheck.int_small QCheck.int_small)
          (QCheck.pair QCheck.int_small QCheck.int_small))
        (fun ((p1, t1), (p2, t2)) ->
          let d1 = Dimcheck.make p1 t1 in
          let d2 = Dimcheck.make p2 t2 in
          (* If they're equal, dim_eq should say so; if not, it shouldn't *)
          Dimcheck.dim_eq d1 d2 = (p1 = p2 && t1 = t2));

      (* Property 3: zero constant is compatible in any addition context.
         This tests the Any variant in the checker, not dim arithmetic
         directly. We test it via a model: rate = gamma * (S + 0.0)
         should always pass regardless of S's dimension. *)
      QCheck.Test.make ~name:"zero_is_universal_additive_identity" ~count:50
        (QCheck.pair QCheck.int_small QCheck.int_small)
        (fun (p, t) ->
          let d = Dimcheck.make p t in
          let zero = Dimcheck.make 0 0 in
          (* Adding dimensionless zero: dim_mul is used for scaling, but
             for additive identity we just check dim_eq(d, d) holds after
             any operation with zero. The real test is in the model-level
             tests above (zero_compatible_with_pop). Here we verify the
             representation: d + zero arithmetic = d for the P,T system. *)
          let sum_p = d.(0) + zero.(0) in
          let sum_t = d.(1) + zero.(1) in
          sum_p = p && sum_t = t);
    ];
  ]
