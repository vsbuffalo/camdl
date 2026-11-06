(* --json-errors structures non-blocking diagnostics, not just errors.

   These tests exercise the REAL render path: they set
   [Diagnostics.json_errors_mode], run [Compiler.compile] (the same
   entry point the CLI calls), and capture what is written to stderr —
   so they observe exactly what `camdlc --json-errors` would print.
   They do NOT go through [Compiler.collect_diagnostics], which never
   renders.

   The invariant under test: a compile emits AT MOST ONE diagnostic
   blob. On the success-with-warnings path that blob is a single JSON
   array (under --json-errors) or a single ANSI box (default); on the
   error path it is the single render [compile] does before returning
   [Error]. The latent double-render bug — non-blocking render firing
   before the autodiff E600 check, then the error path re-rendering
   everything — would emit TWO JSON arrays for an E600-with-warnings
   model. We pin the single-array
   property directly by parsing the captured stderr as a JSON *stream*
   and asserting it holds exactly one value. *)

(* ── stderr capture ──────────────────────────────────────────────────────
   Redirect the real Unix stderr fd to a temp file for the duration of
   [f], then read it back. We flush [Fmt.stderr] and [Stdlib.stderr]
   around the swap so buffered ANSI output lands in the captured file and
   not in the test runner's stderr. *)
let capture_stderr (f : unit -> unit) : string =
  flush stderr;
  let saved = Unix.dup Unix.stderr in
  let tmp = Filename.temp_file "camdl_jsonerr" ".txt" in
  let fd = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o600 in
  Unix.dup2 fd Unix.stderr;
  Unix.close fd;
  let restore () =
    flush stderr;
    Fmt.flush Fmt.stderr ();
    Unix.dup2 saved Unix.stderr;
    Unix.close saved
  in
  (match f () with
   | () -> restore ()
   | exception e -> restore (); raise e);
  let ic = open_in_bin tmp in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  (try Sys.remove tmp with _ -> ());
  s

(* Count top-level JSON values in [s] by parsing it as a stream. Two
   concatenated arrays ("[...]\n[...]\n") yield 2; a single array yields
   1; non-JSON raises. This is the assertion that catches a double-render
   that a naive "starts with '['" check would miss. *)
let json_stream_values (s : string) : Yojson.Safe.t list =
  Yojson.Safe.seq_from_string s |> List.of_seq

(* Extract every (code, severity) pair from a JSON diagnostic array. *)
let codes_of_array (j : Yojson.Safe.t) : (string * string) list =
  match j with
  | `List items ->
    List.map (fun item ->
      match item with
      | `Assoc fields ->
        let s key =
          match List.assoc_opt key fields with
          | Some (`String v) -> v
          | _ -> Alcotest.failf "diagnostic JSON object missing string %S" key
        in
        (s "code", s "severity")
      | _ -> Alcotest.failf "diagnostic array element is not an object") items
  | _ -> Alcotest.failf "expected a JSON array, got: %s" (Yojson.Safe.to_string j)

(* ── Source fixtures (inline, path-independent) ────────────────────────────
   A well-formed SIR that declares a dead compartment `D` — the same
   shape as ocaml/golden/diagnostics/lints/l402_dead_compartment.camdl, inlined so the test
   needs no repo-root resolution. The linter flags `D` (L402, Warning)
   and the compile SUCCEEDS. *)
let l402_src = {camdl|
time_unit = 'days

compartments { S, I, R, D }

let N = S + I + R

parameters {
  beta  : rate  in [0.001, 2.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 100000]
  I0    : count in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

simulate {
  from = 0 'days
  to   = 80 'days
}
|camdl}

(* A syntactically broken model: the front end emits E001 (syntax error)
   and [compile] renders it then returns [Error] — the error path. *)
let error_src = "this is not a valid camdl model {{{"

let with_json_mode b (f : unit -> unit) =
  let prev = !Diagnostics.json_errors_mode in
  Diagnostics.json_errors_mode := b;
  Fun.protect ~finally:(fun () -> Diagnostics.json_errors_mode := prev) f

(* ── Tests ─────────────────────────────────────────────────────────────── *)

(* A warned-but-clean model under --json-errors emits EXACTLY ONE JSON
   array, containing the L402 warning. *)
let test_warning_is_single_json_array () =
  let captured =
    capture_stderr (fun () ->
      with_json_mode true (fun () ->
        match Compiler.compile ~name:"l402" ~filename:"<input>" l402_src with
        | Ok _ -> ()
        | Error e -> Alcotest.failf "expected a successful compile, got Error %S" e))
  in
  let values = json_stream_values captured in
  (* The load-bearing assertion: ONE array, not two. A double-render
     would leave two concatenated arrays here. *)
  Alcotest.(check int) "exactly one JSON array on stderr" 1 (List.length values);
  let pairs = codes_of_array (List.hd values) in
  Alcotest.(check bool) "L402 warning present"
    true (List.mem ("L402", "warning") pairs)

(* Default (non-JSON) output for the SAME warned-but-clean model is
   unchanged: the ANSI box renders, it is NOT a JSON array, and it names
   the L402 warning. *)
let test_warning_default_is_ansi () =
  let captured =
    capture_stderr (fun () ->
      with_json_mode false (fun () ->
        match Compiler.compile ~name:"l402" ~filename:"<input>" l402_src with
        | Ok _ -> ()
        | Error e -> Alcotest.failf "expected a successful compile, got Error %S" e))
  in
  Alcotest.(check bool) "non-empty ANSI render" true (String.length captured > 0);
  (* It must NOT be a JSON array. Stripping ANSI escapes, the first
     non-space char of a JSON emission would be '['. The ANSI box starts
     with "warning[L402]" styling — assert the literal token is present
     and the payload does not parse as a single JSON array. *)
  let mentions_l402 =
    let re = "L402" in
    let rec find i =
      i + String.length re <= String.length captured
      && (String.sub captured i (String.length re) = re || find (i + 1))
    in
    String.length captured >= String.length re && find 0
  in
  Alcotest.(check bool) "ANSI output mentions L402" true mentions_l402;
  let is_json_array =
    match Yojson.Safe.from_string captured with
    | `List _ -> true
    | _ -> false
    | exception _ -> false
  in
  Alcotest.(check bool) "default output is NOT a JSON array" false is_json_array

(* An ERROR model under --json-errors still emits a single valid JSON
   array (unchanged behaviour — the blocking path always JSON-ified). *)
let test_error_is_single_json_array () =
  let captured =
    capture_stderr (fun () ->
      with_json_mode true (fun () ->
        match Compiler.compile ~name:"bad" ~filename:"<input>" error_src with
        | Ok _ -> Alcotest.fail "expected a failed compile on broken source"
        | Error _ -> ()))
  in
  let values = json_stream_values captured in
  Alcotest.(check int) "exactly one JSON array on stderr (error path)"
    1 (List.length values);
  let pairs = codes_of_array (List.hd values) in
  Alcotest.(check bool) "an error diagnostic is present"
    true (List.exists (fun (_, sev) -> sev = "error") pairs)

let () =
  Alcotest.run "json_errors" [
    "render", [
      Alcotest.test_case "warning → single JSON array" `Quick
        test_warning_is_single_json_array;
      Alcotest.test_case "warning default → ANSI box" `Quick
        test_warning_default_is_ansi;
      Alcotest.test_case "error → single JSON array" `Quick
        test_error_is_single_json_array;
    ];
  ]
