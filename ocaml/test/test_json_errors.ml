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

(* ── gh#1215: what the renderers keep and show ─────────────────────────────

   Both renderers used to sort with [List.sort_uniq] on an order that compared
   only (file, line, code, message), so two diagnostics on one line with the
   same code and message (two uses of one undeclared name) compared equal and
   all but one were dropped. The JSON left out [related]; the text left out
   [fixes]. *)

let dup_src = {camdl|
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate  in [0.001, 2.0]
  gamma : rate  in [0.001, 1.0]
  N0    : count in [100, 100000]
  I0    : count in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * Q + gamma * Q
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

let json_items (j : Yojson.Safe.t) =
  match j with
  | `List items -> items
  | _ -> Alcotest.failf "expected a JSON array, got: %s" (Yojson.Safe.to_string j)

(* End to end: the expander reports each use of the undeclared `Q`, and the
   rendered JSON keeps both, at their own columns. *)
let test_same_line_same_message_kept () =
  let emitted =
    Compiler.collect_diagnostics ~name:"dup" ~filename:"dup.camdl" dup_src
    |> List.filter (fun (d : Diagnostics.diagnostic) -> d.code = "E100")
  in
  Alcotest.(check int) "the expander emits one E100 per use of Q"
    2 (List.length emitted);
  let captured =
    capture_stderr (fun () ->
      with_json_mode true (fun () ->
        match Compiler.compile ~name:"dup" ~filename:"dup.camdl" dup_src with
        | Ok _ -> Alcotest.fail "expected a failed compile"
        | Error _ -> ()))
  in
  let cols =
    json_stream_values captured |> List.hd |> json_items
    |> List.filter_map (fun item ->
         match item with
         | `Assoc fs when List.assoc_opt "code" fs = Some (`String "E100") ->
           (match List.assoc_opt "loc" fs with
            | Some (`Assoc l) ->
              (match List.assoc_opt "col" l with Some (`Int c) -> Some c | _ -> None)
            | _ -> None)
         | _ -> None)
  in
  Alcotest.(check (list int)) "both E100s rendered, in column order"
    [34; 46] cols

let mk_loc line col end_col =
  { Diagnostics.file = "m.camdl"; line; col; end_line = line; end_col }

(* An exact duplicate (the same diagnostic emitted twice, as a check run once
   per expanded stratum can) is still shown once. *)
let test_exact_duplicate_dropped () =
  let t = Diagnostics.create () in
  for _ = 1 to 2 do
    Diagnostics.error t ~code:"E100" ~loc:(mk_loc 3 5 6)
      ~message:"undeclared name 'Q'" ()
  done;
  let items = json_items (Yojson.Safe.from_string (Diagnostics.to_json_string t)) in
  Alcotest.(check int) "one JSON entry for an exact duplicate" 1 (List.length items)

let test_related_in_json () =
  let t = Diagnostics.create () in
  Diagnostics.error t ~code:"E278" ~loc:(mk_loc 7 3 8)
    ~message:"'beta' is declared twice"
    ~related:[ (mk_loc 2 3 8, "first declared here") ] ();
  match json_items (Yojson.Safe.from_string (Diagnostics.to_json_string t)) with
  | [ `Assoc fs ] ->
    (match List.assoc_opt "related" fs with
     | Some (`List [ `Assoc r ]) ->
       Alcotest.(check bool) "label carried" true
         (List.assoc_opt "label" r = Some (`String "first declared here"));
       (match List.assoc_opt "loc" r with
        | Some (`Assoc l) ->
          Alcotest.(check bool) "related loc line" true
            (List.assoc_opt "line" l = Some (`Int 2))
        | _ -> Alcotest.fail "related entry has no loc object")
     | _ -> Alcotest.failf "no related array in %s" (Diagnostics.to_json_string t))
  | _ -> Alcotest.fail "expected exactly one diagnostic"

let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = sub || go (i + 1)) in
  go 0

let test_fix_in_text () =
  let t = Diagnostics.create () in
  Diagnostics.error t ~code:"E117" ~loc:(mk_loc 1 1 4)
    ~message:"a unit literal on a parameter declaration"
    ~fixes:[ { Diagnostics.fix_loc = mk_loc 1 10 20;
               replacement = "positive dim 1/T" } ] ();
  let captured =
    capture_stderr (fun () ->
      with_json_mode false (fun () ->
        ignore (Diagnostics.render t
                  (Source_cache.of_string ~filename:"m.camdl" "x\n"))))
  in
  Alcotest.(check bool) "text names the fix's replacement" true
    (contains ~sub:"fix: replace with `positive dim 1/T`" captured)

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
    "gh#1215", [
      Alcotest.test_case "same line, code and message: both kept" `Quick
        test_same_line_same_message_kept;
      Alcotest.test_case "exact duplicate shown once" `Quick
        test_exact_duplicate_dropped;
      Alcotest.test_case "related notes in JSON" `Quick test_related_in_json;
      Alcotest.test_case "fixes in text" `Quick test_fix_in_text;
    ];
  ]
