(** Round-trip test: deserialise each golden file, re-serialise, deserialise
    again, assert that the two OCaml values are structurally equal.

    Run with:  cd ocaml && dune runtest  *)

open Ir

(* ── Helpers ─────────────────────────────────────────────────────────────── *)

let golden_dir () =
  (* Walk up the directory tree from CWD until we find ir/golden/. *)
  let rec find_up dir =
    let candidate = Filename.concat dir (Filename.concat "ir" "golden") in
    if Sys.file_exists candidate && Sys.is_directory candidate
    then candidate
    else begin
      let parent = Filename.dirname dir in
      if String.equal parent dir
      then failwith ("cannot locate ir/golden (started from " ^ Sys.getcwd () ^ ")")
      else find_up parent
    end
  in
  find_up (Sys.getcwd ())

let read_golden name =
  let path = Filename.concat (golden_dir ()) (name ^ ".ir.json") in
  let ic = open_in path in
  let n  = in_channel_length ic in
  let s  = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

(* ── Equality helpers (structural equality on IR types) ───────────────────── *)
(* OCaml structural equality (=) works on these record/variant types because
   they contain only base types (string, float, int, bool) and recursive
   applications of the same types.  Yojson.Safe.t inside data_contract uses
   polymorphic variants which also support structural equality. *)

let models_equal (a : model) (b : model) : bool = a = b

(* ── Core round-trip assertion ───────────────────────────────────────────── *)

let roundtrip_test name () =
  let json_in = read_golden name in

  (* 1. Deserialise *)
  let m1 = match Serde.model_of_string json_in with
    | Ok m    -> m
    | Error e -> Alcotest.failf "deserialise failed for %s: %s" name e
  in

  (* 2. Check version *)
  Alcotest.(check string) (name ^ " version") "0.3" m1.version;

  (* 3. Re-serialise *)
  let json2 = Serde.model_to_string m1 in

  (* 4. Deserialise again *)
  let m2 = match Serde.model_of_string json2 with
    | Ok m    -> m
    | Error e -> Alcotest.failf "round-trip re-deserialise failed for %s: %s" name e
  in

  (* 5. Structural equality *)
  if not (models_equal m1 m2)
  then Alcotest.failf "round-trip structural equality failed for %s" name;

  (* 6. Basic model sanity checks *)
  Alcotest.(check string) (name ^ " name matches") name m1.name;
  Alcotest.(check bool) (name ^ " has compartments") true (m1.compartments <> []);
  Alcotest.(check bool) (name ^ " has transitions")  true (m1.transitions  <> []);

  (* 7. Validation *)
  match Validate.validate m1 with
  | Ok ()     -> ()
  | Error errs ->
    let msgs = List.map Validate.error_to_string errs in
    Alcotest.failf "validation errors in %s:\n  %s" name (String.concat "\n  " msgs)

(* ── Canonical (compact) vs pretty equivalence ────────────────────────────── *)
(* The default serializer emits compact JSON with one element per line for the
   model's top-level arrays; --pretty emits the indented form. Both render the
   same `envelope_to_json m`, so they must encode identical JSON content — this
   is the divergence guard for the custom compact whitespace policy. *)

let canonical_equiv_test name () =
  let m = match Serde.model_of_string (read_golden name) with
    | Ok m    -> m
    | Error e -> Alcotest.failf "deserialise failed for %s: %s" name e
  in
  let compact = Serde.model_to_string m in
  let pretty  = Serde.model_to_string ~pretty:true m in
  if Yojson.Safe.from_string compact <> Yojson.Safe.from_string pretty then
    Alcotest.failf
      "canonical (compact) and pretty IR JSON diverge in content for %s" name;
  (* The canonical form must also round-trip back to the same model. *)
  match Serde.model_of_string compact with
  | Ok m2 when models_equal m m2 -> ()
  | Ok _    -> Alcotest.failf "canonical IR JSON did not round-trip for %s" name
  | Error e -> Alcotest.failf "canonical IR JSON failed to parse for %s: %s" name e

(* ── Test suite ──────────────────────────────────────────────────────────── *)

let golden_cases =
  [ "sir_basic";
    "sir_demography";
    "sir_vaccination";
    "pure_death";
    "birth_death";
    "two_state";
    "cholera_siwr";
    "seir_age";
  ]

(* ── Deserializer invariant: PriorSpec is a single slot ──────────────────── *)

(* The former "prior and hierarchical mutually exclusive" rejection is now
   STRUCTURAL: an estimated parameter's `value.prior` is one `prior_spec`
   slot (Flat | Dist | Hierarchical), so both-set is unrepresentable (gh#191
   ParamValue ADT). This test verifies each variant round-trips through the
   deserializer — i.e. the single slot faithfully carries either a single-level
   prior or a hierarchical one, never both. *)
let prior_spec_single_slot_test () =
  let mk_param prior_json =
    `Assoc [
      ("name",  `String "fabricated");
      ("value", `Assoc [
        ("mode",      `String "estimated");
        ("bounds",    `List [`Float 0.0; `Float 1.0]);
        ("prior",     prior_json);
        ("transform", `String "identity")]);
      ("param_kind", `Null);
      ("param_dim",  `Null);
    ]
  in
  (* Splice a fabricated parameter into sir_basic's envelope.model.parameters. *)
  let deser_with param =
    let j = Yojson.Safe.from_string (read_golden "sir_basic") in
    let splice_params kvs = `Assoc (List.map (fun (k, v) ->
      if String.equal k "parameters" then
        (k, match v with `List xs -> `List (xs @ [param]) | _ -> v)
      else (k, v)) kvs) in
    let j' = match j with
      | `Assoc kvs -> `Assoc (List.map (fun (k, v) ->
          if String.equal k "model" then
            (k, match v with `Assoc inner -> splice_params inner | _ -> v)
          else (k, v)) kvs)
      | _ -> failwith "sir_basic.ir.json is not a top-level object" in
    Serde.model_of_string (Yojson.Safe.to_string j')
  in
  let find_param m = List.find (fun (p : Ir.parameter) -> p.name = "fabricated") m.Ir.parameters in
  (* Dist round-trips into a single-level prior, with no hierarchical. *)
  let dist = mk_param (`Assoc [("dist", `Assoc [("normal",
    `Assoc [("mean", `Float 0.0); ("sd", `Float 1.0)])])]) in
  (match deser_with dist with
   | Error msg -> Alcotest.failf "rejected a Dist prior_spec: %s" msg
   | Ok m ->
     let p = find_param m in
     Alcotest.(check bool) "Dist → single-level prior present"
       true (Ir.param_prior_dist p <> None);
     Alcotest.(check bool) "Dist → no hierarchical"
       true (Ir.param_hierarchical p = None));
  (* Hierarchical round-trips into a hierarchical prior, with no single-level. *)
  let hier = mk_param (`Assoc [("hierarchical", `Assoc [
    ("kind", `String "normal"); ("args", `Assoc []); ("pool_over", `String "")])]) in
  (match deser_with hier with
   | Error msg -> Alcotest.failf "rejected a Hierarchical prior_spec: %s" msg
   | Ok m ->
     let p = find_param m in
     Alcotest.(check bool) "Hierarchical → hierarchical present"
       true (Ir.param_hierarchical p <> None);
     Alcotest.(check bool) "Hierarchical → no single-level prior"
       true (Ir.param_prior_dist p = None))

(* gh#166: integrator serde — the Rk45 branch was previously exercised by no
   OCaml test. Assert rk45 {atol,rtol} round-trips through the tagged JSON, an
   explicit rk4 tag decodes to Rk4, and an UNKNOWN method is rejected (raises
   DeserError) rather than silently defaulting to rk4 — mirroring the Rust
   internally-tagged enum. *)
let integrator_serde_test () =
  let i = Ir.Rk45 { atol = Some 1e-8; rtol = Some 1e-6 } in
  (match Serde.integrator_of_json (Serde.integrator_to_json i) with
   | Ir.Rk45 { atol = Some a; rtol = Some r } when a = 1e-8 && r = 1e-6 -> ()
   | _ -> Alcotest.fail "rk45 {atol; rtol} did not round-trip");
  (match Serde.integrator_of_json (`Assoc [("method", `String "rk4")]) with
   | Ir.Rk4 -> ()
   | _ -> Alcotest.fail "explicit rk4 tag did not decode to Rk4");
  let rejects_unknown =
    try ignore (Serde.integrator_of_json (`Assoc [("method", `String "euler")])); false
    with Serde.DeserError _ -> true
  in
  Alcotest.(check bool) "unknown integrator method is rejected" true rejects_unknown

(* gh#284: the `PerEvalRef` expr variant (gh#272 LICM) appears in no golden model
   — no golden hoists — so the golden-driven round-trip above never exercises its
   serde path. Assert it directly, both bare and nested in a compound node,
   mirroring the Rust `roundtrips_every_variant` coverage. *)
let expr_serde_test () =
  let open Ir in
  let roundtrips (e : expr) = Serde.expr_of_json (Serde.expr_to_json e) = e in
  Alcotest.(check bool) "bare PerEvalRef round-trips" true
    (roundtrips (PerEvalRef "__licm_0"));
  Alcotest.(check bool) "PerEvalRef nested in a BinOp round-trips" true
    (roundtrips (BinOp { op = Mul; left = PerEvalRef "__licm_1"; right = Param "beta" }))

(* proposal 2026-06-25: generated quantities appear in no golden model — the
   frontend does not yet emit them — so the golden round-trip above never
   exercises their serde. Assert it directly (mirroring the Rust
   `round_trips_*` + `pins_wire_tags` coverage): a Reduced State quantity, a
   Derived reduction-arithmetic quantity, and the exact pinned wire shapes. *)
let quantity_serde_test () =
  let open Ir in
  let roundtrips (q : quantity) =
    Serde.quantity_of_json (Serde.quantity_to_json q) = q in
  (* Reduced State, scalar: peak_prevalence = max(I / N). *)
  let peak = {
    q_name = "peak_prevalence";
    q_stratum = [];
    q_dimension = None;
    q_body = QBReduced {
      source = QSState (BinOp { op = Div; left = Pop "I"; right = Pop "N" });
      reduce = Some (RValue VMax);
    };
  } in
  (* Reduced State, series (reduce = None) with an Expr-threshold time reduction
     in a sibling, plus an Integral — exercise every TemporalReduce arm. *)
  let series = {
    q_name = "prevalence"; q_stratum = [];
    q_dimension = None;
    q_body = QBReduced {
      source = QSState (BinOp { op = Div; left = Pop "I"; right = Pop "N" });
      reduce = None };
  } in
  let onset = {
    q_name = "takeoff_time"; q_stratum = [];
    q_dimension = None;
    q_body = QBReduced {
      source = QSState (Pop "I_total");
      reduce = Some (RTime (FirstAbove (Param "i_thresh"))) };
  } in
  let person_days = {
    q_name = "person_days_inf"; q_stratum = [];
    q_dimension = None;
    q_body = QBReduced { source = QSState (Pop "I"); reduce = Some RIntegral };
  } in
  let counts = {
    q_name = "positive_months";
    q_stratum = [("patch", "p1")];
    q_dimension = None;
    q_body = QBReduced {
      source = QSState (Pop "I_p1");
      reduce = Some (RValue (VCountAbove (Param "i_thresh"))) };
  } in
  (* Derived reduction arithmetic with a stratified QRef, abs UnOp, Sub BinOp,
     and a Cond — exercise every ScalarExpr arm. *)
  let dur = {
    q_name = "outbreak_dur";
    q_stratum = [("patch", "p1")];
    q_dimension = None;
    q_body = QBDerived (SUnOp {
      op = Abs;
      arg = SBinOp {
        op = Sub;
        left = SQRef { qref_name = "fadeout_time";
                       qref_stratum = [("patch", "p1")] };
        right = SCond {
          pred  = SConst 1.0;
          then_ = SQRef { qref_name = "takeoff_time"; qref_stratum = [] };
          else_ = SParam "t0" };
      };
    });
  } in
  (* v1.1: an observation-source quantity (reduces the simulated y_sim). *)
  let obs = {
    q_name = "first_afp"; q_stratum = [];
    q_dimension = None;
    q_body = QBReduced {
      source = QSObservation "afp";
      reduce = Some (RTime (FirstAbove (Const 0.0))) };
  } in
  List.iter (fun (label, q) ->
    Alcotest.(check bool) (label ^ " round-trips") true (roundtrips q))
    [ ("Reduced State max(I/N)", peak);
      ("Reduced State series",   series);
      ("Reduced State first_above", onset);
      ("Reduced State integral", person_days);
      ("Reduced State count_above (stratified)", counts);
      ("Reduced Observation first_above", obs);
      ("Derived reduction arithmetic", dur) ];
  Alcotest.(check string) "observation source wire"
    {|{"observation":{"stream":"afp"}}|}
    (Yojson.Safe.to_string (Serde.quantity_source_to_json (QSObservation "afp")));
  (* Pin the exact on-wire shape the Rust serde fixes (quantity.rs pins_wire_tags). *)
  let pin_q = {
    q_name = "p"; q_stratum = [];
    q_dimension = None;
    q_body = QBReduced {
      source = QSState (Pop "I");
      reduce = Some (RTime TimeOfMax) };
  } in
  Alcotest.(check string) "pinned reduced/time_of_max wire"
    {|{"name":"p","body":{"reduced":{"source":{"state":{"pop":"I"}},"reduce":{"time":"time_of_max"}}}}|}
    (Yojson.Safe.to_string (Serde.quantity_to_json pin_q));
  let pin_d = {
    q_name = "d"; q_stratum = [];
    q_dimension = None;
    q_body = QBDerived (SConst 2.5);
  } in
  Alcotest.(check string) "pinned derived/const wire"
    {|{"name":"d","body":{"derived":{"const":2.5}}}|}
    (Yojson.Safe.to_string (Serde.quantity_to_json pin_d))

let () =
  let tests =
    List.map (fun name ->
      Alcotest.test_case name `Quick (roundtrip_test name)
    ) golden_cases
  in
  let equiv_tests =
    List.map (fun name ->
      Alcotest.test_case name `Quick (canonical_equiv_test name)
    ) golden_cases
  in
  let invariant_tests = [
    Alcotest.test_case "prior_spec single slot" `Quick prior_spec_single_slot_test;
    Alcotest.test_case "integrator serde (rk45 round-trip + strict)" `Quick integrator_serde_test;
    Alcotest.test_case "expr serde (PerEvalRef round-trips)" `Quick expr_serde_test;
    Alcotest.test_case "quantity serde (round-trip + pinned wire)" `Quick quantity_serde_test;
  ] in
  Alcotest.run "IR round-trip" [
    ("golden", tests);
    ("canonical≡pretty", equiv_tests);
    ("deser-invariants", invariant_tests);
  ]
