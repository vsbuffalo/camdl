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
(* Increment B1. The golden corpus contains no `WeightedFlowSum`, so the
   round-trip suite — which walks the goldens — never executes its serde at all.
   Without this the OCaml emitter and reader were shipped unexercised, and the
   FIRST model to use the variant would have been the test.

   Pins the wire shape as well as the round-trip, because the Rust side derives
   its own serde independently: if OCaml emits a key Rust does not accept, the
   two halves of the contract disagree and no OCaml-only test would notice. *)
let weighted_flow_sum_serde_test () =
  let open Ir in
  let terms = [
    { wf_weight = Param "rho_child"; wf_flow = "infection_child" };
    { wf_weight = BinOp { op = Mul; left = Param "rho_adult"; right = Const 0.5 };
      wf_flow   = "infection_adult" };
  ] in
  let p = WeightedFlowSum terms in
  (* 1. round-trip *)
  Alcotest.(check bool) "WeightedFlowSum round-trips" true
    (Serde.projection_of_json (Serde.projection_to_json p) = p);
  (* 2. the wire shape, pinned — this is the half Rust must agree with *)
  let wire = Yojson.Safe.to_string (Serde.projection_to_json p) in
  let expected =
    {|{"weighted_flow_sum":[{"weight":{"param":"rho_child"},"flow":"infection_child"},|} ^
    {|{"weight":{"bin_op":{"op":"mul","left":{"param":"rho_adult"},"right":{"const":0.5}}},|} ^
    {|"flow":"infection_adult"}]}|} in
  Alcotest.(check string) "WeightedFlowSum wire shape" expected wire;
  (* 3. an empty term list must survive the trip rather than collapsing *)
  Alcotest.(check bool) "empty WeightedFlowSum round-trips" true
    (Serde.projection_of_json (Serde.projection_to_json (WeightedFlowSum [])) =
     WeightedFlowSum []);
  (* 4. the sibling variants must be untouched by the new arm *)
  List.iter (fun q ->
    Alcotest.(check bool) "sibling projection round-trips" true
      (Serde.projection_of_json (Serde.projection_to_json q) = q))
    [ CumulativeFlow "infection";
      CumulativeFlowSum ["infection_child"; "infection_adult"];
      CurrentPop "I";
      CurrentPopSum ["I_child"; "I_adult"];
      DerivedExpr (Pop "I") ]

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

(* proposal 2026-06-25: counterfactual contrasts. The `contrast` IR node is new
   and appears in no golden model the round-trip above exercises, so assert its
   serde directly (mirroring quantity_serde_test and the Rust contrast.rs
   coverage): a run-member operand, a bin-op body, the run_namespace tags, and
   the exact pinned wire shape. *)
let contrast_serde_test () =
  let open Ir in
  let roundtrips (c : contrast) =
    Serde.contrast_of_json (Serde.contrast_to_json c) = c in
  (* run_namespace tags round-trip both arms. *)
  Alcotest.(check bool) "NsQuantities round-trips" true
    (Serde.run_namespace_of_json (Serde.run_namespace_to_json NsQuantities)
     = NsQuantities);
  Alcotest.(check bool) "NsObservations round-trips" true
    (Serde.run_namespace_of_json (Serde.run_namespace_to_json NsObservations)
     = NsObservations);
  (* A scalar contrast differencing a quantity across two runs (the showcase
     "deaths averted" shape). *)
  let averted = {
    c_name = "averted";
    c_body = CBinOp {
      op = Sub;
      left  = CRunMember { run = "no_sia";   ns = NsQuantities; member = "total" };
      right = CRunMember { run = "with_sia"; ns = NsQuantities; member = "total" };
    };
  } in
  (* A bare run-member body against the observations namespace. *)
  let raw = {
    c_name = "raw";
    c_body = CRunMember { run = "fitted"; ns = NsObservations; member = "afp" };
  } in
  List.iter (fun (label, c) ->
    Alcotest.(check bool) (label ^ " round-trips") true (roundtrips c))
    [ ("contrast bin_op Sub (quantities)", averted);
      ("contrast bare run_member (observations)", raw) ];
  (* Pin the exact on-wire shape the Rust serde fixes (contrast.rs). *)
  Alcotest.(check string) "pinned contrast wire"
    {|{"name":"averted","body":{"bin_op":{"op":"sub","left":{"run_member":{"run":"no_sia","ns":"quantities","member":"total"}},"right":{"run_member":{"run":"with_sia","ns":"quantities","member":"total"}}}}}|}
    (Yojson.Safe.to_string (Serde.contrast_to_json averted))

(* gh#616: observation anchors. Two things to pin, and the second is the one
   that matters for the 0.32 bump.

   1. The codec itself — both wire spellings, both anchors, and the
      normalisation (an explicit zero-offset object re-encodes to the bare
      string, so the canonical form is unique).

   2. **A model with no anchor must serialise byte-identically to its pre-0.32
      form, modulo the version strings.** The new fields use the
      append-when-present idiom (`integrator`-style), not the adjacent
      null-emitting one (`dt`, `rng_seed`, a preset's `t_end`) — if either had
      been written the null way, every one of the 108 committed IR files would
      have gained a key and every golden diff would be noise that hides the
      real change. This test is what makes that claim checkable rather than
      asserted. *)

let anchor_serde_test () =
  let open Ir in
  let rt (a : anchored_time) =
    Serde.anchored_time_of_json (Serde.anchored_time_to_json a) = a in
  let wire (a : anchored_time) =
    Yojson.Safe.to_string (Serde.anchored_time_to_json a) in
  (* Canonical emission: a zero offset is the BARE STRING — exactly what
     `value_at(…, last_obs)` emitted before gh#616. *)
  Alcotest.(check string) "bare last_obs wire" {|"last_obs"|}
    (wire { anchor = AnchorLast; offset = 0.0 });
  Alcotest.(check string) "bare first_obs wire" {|"first_obs"|}
    (wire { anchor = AnchorFirst; offset = 0.0 });
  Alcotest.(check string) "offset last_obs wire"
    {|{"anchor":"last_obs","offset":28.0}|}
    (wire { anchor = AnchorLast; offset = 28.0 });
  Alcotest.(check string) "offset first_obs wire"
    {|{"anchor":"first_obs","offset":-7.0}|}
    (wire { anchor = AnchorFirst; offset = -7.0 });
  List.iter (fun (label, a) ->
    Alcotest.(check bool) (label ^ " round-trips") true (rt a))
    [ ("bare last_obs",   { anchor = AnchorLast;  offset = 0.0 });
      ("bare first_obs",  { anchor = AnchorFirst; offset = 0.0 });
      ("last_obs + 28",   { anchor = AnchorLast;  offset = 28.0 });
      ("first_obs - 7",   { anchor = AnchorFirst; offset = -7.0 }) ];
  (* The object form with a zero offset decodes, then re-encodes canonically. *)
  let zero_obj = `Assoc [("anchor", `String "last_obs"); ("offset", `Float 0.0)] in
  Alcotest.(check string) "explicit zero offset normalises to the bare string"
    {|"last_obs"|}
    (Yojson.Safe.to_string
       (Serde.anchored_time_to_json (Serde.anchored_time_of_json zero_obj)));
  (* An unknown anchor is rejected, not silently defaulted. *)
  let rejects_unknown =
    try ignore (Serde.anchored_time_of_json (`String "mid_obs")); false
    with Serde.DeserError _ -> true
  in
  Alcotest.(check bool) "unknown anchor is rejected" true rejects_unknown;
  (* An anchored simulation config round-trips through the whole block. *)
  let cfg = { t_start = 0.0; t_end = Float.nan; time_semantics = "continuous";
              dt = None; rng_seed = None; integrator = Rk4;
              t_end_anchor = Some { anchor = AnchorLast; offset = 28.0 } } in
  let cfg' = Serde.simulation_config_of_json (Serde.simulation_config_to_json cfg) in
  Alcotest.(check bool) "anchored simulation config round-trips" true
    (cfg'.t_end_anchor = cfg.t_end_anchor);
  Alcotest.(check bool) "the baked t_end stays NaN across the round-trip" true
    (Float.is_nan cfg'.t_end)

(* (2): the byte-identity claim, per golden. *)

(* Does `hay` contain `needle`? (No Str dependency — this test suite links only
   yojson + alcotest.) *)
let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

let no_anchor_bytes_unchanged_test name () =
  let raw = read_golden name in
  let m = match Serde.model_of_string raw with
    | Ok m -> m
    | Error e -> Alcotest.failf "deserialise failed for %s: %s" name e
  in
  Alcotest.(check bool) (name ^ ": golden declares no horizon anchor") true
    (m.simulation.t_end_anchor = None
     && List.for_all (fun (p : Ir.preset) -> p.preset_t_end_anchor = None) m.presets);
  (* The whole corpus: not one byte of `t_end_anchor` anywhere in the emitted
     IR of a model that declares no anchor. This is what fails if either new
     field is ever switched to the adjacent null-emitting idiom. *)
  let out = Serde.model_to_string m in
  Alcotest.(check bool) (name ^ ": no t_end_anchor key is emitted") true
    (not (contains out "t_end_anchor"))

(* The precise pin: the exact `simulation` and preset objects the 0.31
   serializer emitted for `sir_basic`, copied from the committed 0.31 golden.
   Emitting a `"t_end_anchor": null` — or appending the key in a different
   position — moves these strings and reddens the test. *)
let no_anchor_wire_is_byte_identical_test () =
  let m = match Serde.model_of_string (read_golden "sir_basic") with
    | Ok m -> m | Error e -> Alcotest.failf "deserialise failed: %s" e in
  Alcotest.(check string) "unanchored simulation block is unchanged from 0.31"
    {|{"t_start":0.0,"t_end":80.0,"time_semantics":"continuous","dt":null,"rng_seed":null}|}
    (Yojson.Safe.to_string (Serde.simulation_config_to_json m.simulation));
  let baseline = List.find (fun (p : Ir.preset) -> p.preset_name = "baseline") m.presets in
  Alcotest.(check string) "unanchored preset is unchanged from 0.31"
    {|{"name":"baseline","label":"default  (R0 ≈ 3)","params":{"beta":0.3,"gamma":0.1,"N0":1000.0,"I0":10.0},"enable":[],"disable":[],"t_end":80.0}|}
    (Yojson.Safe.to_string (Serde.preset_to_json baseline));
  (* And the anchored form APPENDS exactly one key at the end, with the horizon
     emitted as `null`.

     `null`, not `NaN`: the compiler bakes NaN for an unresolved anchored
     horizon, but JSON HAS NO NaN LITERAL. Yojson will happily write a bare
     `NaN` token, which `serde_json` then rejects — so an anchored model could
     not be loaded by the runtime at all. The anchor field beside it carries what
     the horizon is, and the reader restores the NaN from that. *)
  let anchored = { m.simulation with
                   t_end = Float.nan;
                   t_end_anchor = Some { Ir.anchor = Ir.AnchorLast; offset = 28.0 } } in
  let anchored_json = Yojson.Safe.to_string (Serde.simulation_config_to_json anchored) in
  Alcotest.(check string) "an anchored simulation block appends one key"
    {|{"t_start":0.0,"t_end":null,"time_semantics":"continuous","dt":null,"rng_seed":null,"t_end_anchor":{"anchor":"last_obs","offset":28.0}}|}
    anchored_json;
  (* The property the string above is a proxy for, asserted directly: what we
     emit must be parseable as JSON. A `NaN` token is not. *)
  (match Yojson.Safe.from_string anchored_json with
   | _ -> ()
   | exception _ ->
     Alcotest.failf "an anchored simulation block must be valid JSON: %s" anchored_json);
  (* And it round-trips back to NaN, so the in-memory invariant survives. *)
  let back = Serde.simulation_config_of_json (Yojson.Safe.from_string anchored_json) in
  Alcotest.(check bool) "the anchored horizon reads back as NaN" true (Float.is_nan back.t_end);
  Alcotest.(check bool) "and keeps its anchor" true (back.t_end_anchor = anchored.t_end_anchor)

(* ir/VERSION 0.33: forcing data provenance. Same two things to pin as the
   0.32 anchor, and again the second is the one that matters for the bump.

   1. The codec — a `data = "path"` forcing carries the path as written plus
      the SHA-256 of the file's bytes, and both survive the round-trip.

   2. **A forcing that is not file-backed must serialise byte-identically to
      its pre-0.33 form.** `data_source` uses the append-when-present idiom
      (the `integrator` one, NOT the adjacent null-emitting `dt`/`rng_seed`
      one) — had it been written the null way, every forcing in the 108
      committed IR files would have gained a `"data_source": null` and the
      golden diff for this bump would have been noise hiding the one real
      change. This is what makes that claim checkable rather than asserted. *)

let data_source_serde_test () =
  let open Ir in
  let sha = "908fb5d7e89d7140b1858bd0c83773e14b3b5c40ff397d04c4031e217249f030" in
  let tf_plain = { name = "clim";
                   kind = Interpolated { times  = [Const 0.0; Const 30.0];
                                         values = [Const 1.4; Const 1.3];
                                         method_ = "linear" };
                   dim  = (0, 0);
                   lag  = None;
                   data_source = None } in
  let tf_file = { tf_plain with
                  data_source = Some { path = "data/flu_forcing.tsv"; sha256 = sha } } in
  (* The pre-0.33 bytes, exactly: no key appears for a forcing with no file. *)
  Alcotest.(check string) "a non-file-backed forcing is unchanged from 0.32"
    {|{"name":"clim","kind":{"interpolated":{"times":[{"const":0.0},{"const":30.0}],"values":[{"const":1.4},{"const":1.3}],"method":"linear"}},"dim":[0,0]}|}
    (Yojson.Safe.to_string (Serde.time_function_to_json tf_plain));
  (* And the file-backed form APPENDS exactly one key, at the end. *)
  Alcotest.(check string) "a file-backed forcing appends one key"
    ({|{"name":"clim","kind":{"interpolated":{"times":[{"const":0.0},{"const":30.0}],"values":[{"const":1.4},{"const":1.3}],"method":"linear"}},"dim":[0,0],|}
     ^ {|"data_source":{"path":"data/flu_forcing.tsv","sha256":"|} ^ sha ^ {|"}}|})
    (Yojson.Safe.to_string (Serde.time_function_to_json tf_file));
  (* Round-trip: both fields come back, and absence stays absence. *)
  let rt tf = Serde.time_function_of_json (Serde.time_function_to_json tf) in
  Alcotest.(check bool) "file-backed forcing round-trips" true (rt tf_file = tf_file);
  Alcotest.(check bool) "no-provenance forcing round-trips" true (rt tf_plain = tf_plain);
  (* `lag` and `data_source` are independent appends, in that order. *)
  let tf_both = { tf_file with lag = Some (Const 10.0) } in
  Alcotest.(check bool) "lag + data_source round-trip together" true (rt tf_both = tf_both);
  Alcotest.(check bool) "lag is emitted before data_source" true
    (contains (Yojson.Safe.to_string (Serde.time_function_to_json tf_both))
       {|"lag":{"const":10.0},"data_source":|})

(* (2): the byte-identity claim, per golden. These goldens declare forcings but
   no `data = "..."` file, so not one byte of `data_source` may appear anywhere
   in what they emit — which is what fails if the field is ever switched to the
   null-emitting idiom.

   The corpus is deliberately NOT [golden_cases]: not one of those eight models
   declares a forcing at all, so the assertion would hold trivially and the
   test would pass whatever the serializer did. The first guard below is what
   keeps that from happening again silently — a model that lost its forcing
   fails here rather than quietly making the second guard vacuous. *)
let forcing_bearing_cases =
  [ "seir_seasonal_patch";       (* two sinusoidals *)
    "seir_vaccine_seasonal";     (* one sinusoidal *)
    "seir_spatial_5_inference";  (* a periodic *)
    "sirv_anchored_calendar";    (* a periodic, under a calendar origin *)
  ]

let no_data_source_bytes_unchanged_test name () =
  let m = match Serde.model_of_string (read_golden name) with
    | Ok m -> m
    | Error e -> Alcotest.failf "deserialise failed for %s: %s" name e
  in
  Alcotest.(check bool)
    (name ^ ": carries a forcing (else this test asserts nothing)") true
    (m.time_functions <> []);
  Alcotest.(check bool) (name ^ ": golden declares no file-backed forcing") true
    (List.for_all (fun (tf : Ir.time_function) -> tf.data_source = None)
       m.time_functions);
  Alcotest.(check bool) (name ^ ": no data_source key is emitted") true
    (not (contains (Serde.model_to_string m) "data_source"))

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
  let byte_identity_tests =
    List.map (fun name ->
      Alcotest.test_case name `Quick (no_anchor_bytes_unchanged_test name)
    ) golden_cases
  in
  let byte_identity_tests =
    byte_identity_tests
    @ List.map (fun name ->
        Alcotest.test_case (name ^ " (no data_source)") `Quick
          (no_data_source_bytes_unchanged_test name)) forcing_bearing_cases
    @ [ Alcotest.test_case "sir_basic simulation + preset wire" `Quick
          no_anchor_wire_is_byte_identical_test ]
  in
  let invariant_tests = [
    Alcotest.test_case "anchor serde (round-trip + pinned wire)" `Quick anchor_serde_test;
    Alcotest.test_case "forcing data_source serde (round-trip + pinned wire)" `Quick
      data_source_serde_test;
    Alcotest.test_case "prior_spec single slot" `Quick prior_spec_single_slot_test;
    Alcotest.test_case "integrator serde (rk45 round-trip + strict)" `Quick integrator_serde_test;
    Alcotest.test_case "expr serde (PerEvalRef round-trips)" `Quick expr_serde_test;
    Alcotest.test_case "quantity serde (round-trip + pinned wire)" `Quick quantity_serde_test;
    Alcotest.test_case "weighted_flow_sum serde (round-trip + pinned wire)" `Quick
      weighted_flow_sum_serde_test;
    Alcotest.test_case "contrast serde (round-trip + pinned wire)" `Quick contrast_serde_test;
  ] in
  Alcotest.run "IR round-trip" [
    ("golden", tests);
    ("canonical≡pretty", equiv_tests);
    ("bytes-unchanged-no-anchor", byte_identity_tests);
    ("deser-invariants", invariant_tests);
  ]
