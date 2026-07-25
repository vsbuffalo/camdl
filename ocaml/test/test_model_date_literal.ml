(* Model-side `date()` literal range validation + symmetry with the
   data-loader path (gh#134).

   The §6 ("parse-don't-validate") design note
   (docs/dev/notes/2026-06-08-static-typing-as-bug-prevention.md) flags
   `date()` model literals as a place where the model surface must
   validate + convert ISO dates the SAME way the data loader does — same
   acceptance set, same day-offset conversion, and no silent / divergent
   error path. The data-loader path catches `Failure msg | Invalid_argument
   msg` and emits a clean located diagnostic (symmetric with the W326
   numeric-time warning); the model `date()` path must match it.

   These cases drive the real compile pipeline (lex → parse → expand) over
   a model that uses `date(...)` in `simulate { from / to }` and assert:

     1. a `date()` literal under a date origin converts to the SAME internal
        time the shared `parse_iso_date` / `days_of_date` machinery (and the
        Rust `caltime` mirror, pinned in ir/golden/caltime.tsv) computes —
        the rata_die day-offset from origin;

     2. an out-of-range / malformed `date()` literal ERRORS (located E223),
        not a silent `0.0` shift. The full rejection set: an impossible day
        (Feb 30), a month out of 1..12 (month 13), and a zero day (day 00),
        each leap-aware via `parse_iso_date`;

     3. a `date()` literal under a non-time `time_unit` (`'count` / `'ratio`,
        both parseable alongside `origin = date(...)`) produces a LOCATED
        diagnostic — NOT an uncaught `Invalid_argument` rendered as a bare
        E001 stack-trace. This is the divergence from the data-loader path,
        which already catches `Invalid_argument` (expander.ml `load_table_data`)
        and emits a clean located error. *)

let codes_of (src : string) : string list =
  Compiler.collect_diagnostics src
  |> List.map (fun (d : Diagnostics.diagnostic) -> d.Diagnostics.code)

let has_error_code (src : string) (code : string) : bool =
  Compiler.collect_diagnostics src
  |> List.exists (fun (d : Diagnostics.diagnostic) ->
       d.Diagnostics.code = code && d.Diagnostics.severity = Diagnostics.Error)

(* A minimal anchored model whose simulate.from/to are spliced in. *)
let model ~time_unit ~from_expr ~to_expr =
  Printf.sprintf
    "time_unit = '%s\n\
     origin    = date(\"1950-01-01\")\n\
     \n\
     compartments { S, I }\n\
     \n\
     parameters {\n\
     \  beta : rate in [0.1, 2.0]\n\
     }\n\
     \n\
     transitions {\n\
     \  infection : S --> I @ beta * S\n\
     }\n\
     \n\
     init {\n\
     \  S = 100\n\
     \  I = 1\n\
     }\n\
     \n\
     simulate {\n\
     \  from = %s\n\
     \  to   = %s\n\
     }\n"
    time_unit from_expr to_expr

(* ── 1. symmetric conversion: date() resolves to the rata_die offset ─────── *)
let test_date_literal_converts_symmetrically () =
  let src =
    model ~time_unit:"days"
      ~from_expr:"date(\"1952-01-01\")" ~to_expr:"date(\"1963-09-08\")"
  in
  match Compiler.compile src with
  | Error e -> Alcotest.failf "expected clean compile, got: %s" e
  | Ok m ->
    (* rata_die("1952-01-01") - rata_die("1950-01-01") = 730 days.
       rata_die("1963-09-08") - rata_die("1950-01-01") = 4998 days.
       (Cross-checked against the OCaml days_of_date and the Rust
       caltime::rata_die mirror; days_per_unit('days) = 1.) *)
    Alcotest.(check (float 1e-9)) "from = 730 days"
      730.0 m.Ir.simulation.Ir.t_start;
    Alcotest.(check (float 1e-9)) "to = 4998 days"
      4998.0 m.Ir.simulation.Ir.t_end

(* ── 2. out-of-range / malformed date() errors (located E223) ────────────── *)
(* The full rejection set: an impossible day for the month, a month outside
   1..12, and a zero day. Each must surface a LOCATED E223 (never a silent
   0.0 shift, never an uncaught Invalid_argument). *)
let test_out_of_range_day_errors () =
  let src =
    model ~time_unit:"days"
      ~from_expr:"date(\"1952-02-30\")" ~to_expr:"date(\"1963-09-08\")"
  in
  Alcotest.(check bool)
    "impossible day date(\"1952-02-30\") emits E223" true
    (has_error_code src "E223");
  Alcotest.(check bool)
    "impossible day date(\"1952-02-30\") does NOT escape as E001" false
    (List.mem "E001" (codes_of src))

let test_out_of_range_month_errors () =
  let src =
    model ~time_unit:"days"
      ~from_expr:"date(\"2020-13-01\")" ~to_expr:"date(\"2021-01-01\")"
  in
  Alcotest.(check bool)
    "month-13 date(\"2020-13-01\") emits E223" true
    (has_error_code src "E223");
  Alcotest.(check bool)
    "month-13 date(\"2020-13-01\") does NOT escape as E001" false
    (List.mem "E001" (codes_of src))

let test_zero_day_errors () =
  let src =
    model ~time_unit:"days"
      ~from_expr:"date(\"2020-01-00\")" ~to_expr:"date(\"2021-01-01\")"
  in
  Alcotest.(check bool)
    "zero-day date(\"2020-01-00\") emits E223" true
    (has_error_code src "E223");
  Alcotest.(check bool)
    "zero-day date(\"2020-01-00\") does NOT escape as E001" false
    (List.mem "E001" (codes_of src))

(* ── 3. date() under a non-time time_unit: located error, not a bare E001 ── *)
let test_nontime_unit_date_is_located () =
  (* `time_unit = 'count` lexes (parser.mly: unit_lit accepts count/ratio) and
     combines with `origin = date(...)`. It must never surface as the generic
     E001 stack-trace carrying
     `Invalid_argument("parse_date_to_float: time_unit must be a time unit")`.

     The reported code is E228 (gh#464): the mistake is the `time_unit`
     declaration, not the `date()` call, so the diagnostic points at the
     declaration. Before E228 existed, the closest available diagnostic was a
     no-loc E223 raised from the `date()` conversion — the symptom rather than
     the cause. The durable invariant across both is "no bare E001". *)
  let src =
    model ~time_unit:"count"
      ~from_expr:"date(\"1952-01-01\")" ~to_expr:"date(\"1963-09-08\")"
  in
  let codes = codes_of src in
  Alcotest.(check bool)
    "non-time time_unit date() does NOT surface a bare E001 stack-trace"
    false (List.mem "E001" codes);
  Alcotest.(check bool)
    "non-time time_unit is rejected at its declaration (E228)"
    true (has_error_code src "E228")

let () =
  Alcotest.run "model_date_literal"
    [ ("symmetry",
       [ Alcotest.test_case "date() converts to rata_die offset" `Quick
           test_date_literal_converts_symmetrically;
         Alcotest.test_case "out-of-range day date() errors (E223)" `Quick
           test_out_of_range_day_errors;
         Alcotest.test_case "out-of-range month date() errors (E223)" `Quick
           test_out_of_range_month_errors;
         Alcotest.test_case "zero-day date() errors (E223)" `Quick
           test_zero_day_errors;
         Alcotest.test_case "non-time time_unit is E228 at its decl, not E001"
           `Quick test_nontime_unit_date_is_located ]) ]
