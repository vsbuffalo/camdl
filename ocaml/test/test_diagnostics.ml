(* Compiler-diagnostics harness.

   Four pieces, all driven through the real compile pipeline via
   [Compiler.collect_diagnostics] (lex → parse → expand → validate →
   dimcheck → lint → autodiff), which returns EVERY diagnostic — errors,
   warnings, and infos — without rendering or aborting:

   1. Fixture-by-code driver: every `.camdl` under `test/lints/` is
      annotated with the diagnostic codes (+severity) it must emit via an
      inline `# expect:` comment, and the driver asserts the emitted
      (code, severity) set EXACTLY matches — catching both misses and
      spurious emissions, for warnings/lints and errors alike.

   2. Clean-corpus regression: every `.camdl` under the model corpora
      (`ocaml/golden/`, `tests/fixtures/`, `tests/recovery/`,
      `tests/external/`) must emit NO diagnostic, modulo a small explicit
      allowlist (empty today). Guards future lints from false-positiving
      on real models.

   3. Catalog consistency: the set of diagnostic codes EMITTED in the
      compiler source equals the set DOCUMENTED in
      `docs/dev/warning-catalog.md`.

   Run with:  cd ocaml && dune runtest *)

(* Disable the constant-fold escape-hatch dependence: collect_diagnostics
   uses the production pipeline, where folding is on by default. Folding
   does not emit diagnostics, so it is harmless here; we leave it at its
   default. *)

(* ── Small string helpers (no Str dependency) ───────────────────────────── *)

let trim = String.trim

let starts_with ~prefix s =
  let pl = String.length prefix in
  String.length s >= pl && String.sub s 0 pl = prefix

let split_on c s = String.split_on_char c s

(* Split a string into whitespace-separated tokens. *)
let words s =
  s |> String.split_on_char ' '
    |> List.concat_map (String.split_on_char '\t')
    |> List.map trim
    |> List.filter (fun w -> w <> "")

let read_file path =
  let ic = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
    let n = in_channel_length ic in
    really_input_string ic n)

(* ── Repo-root resolution (works regardless of dune cwd) ─────────────────── *)

(* The corpus directories `tests/fixtures`, `tests/recovery`, and
   `tests/external` live at the repository root — OUTSIDE the `ocaml/`
   dune project root — so they cannot be pulled in as dune `(deps ...)`.
   Instead we locate the real repo root at runtime by walking up from the
   cwd looking for the `ir/VERSION` marker, then read the corpus files by
   their true source paths. `ocaml/test/lints/` and `ocaml/golden/` live
   inside the project and are also resolved this way for uniformity. *)
let repo_root =
  let is_root dir =
    Sys.file_exists (Filename.concat dir "ir/VERSION")
    && Sys.file_exists (Filename.concat dir "ocaml/dune-project")
  in
  let rec walk dir depth =
    if is_root dir then Some dir
    else if depth = 0 then None
    else
      let parent = Filename.dirname dir in
      if parent = dir then None else walk parent (depth - 1)
  in
  let start = Sys.getcwd () in
  match walk start 12 with
  | Some d -> d
  | None ->
    (* Fallback: walk up from the test executable's directory. *)
    let exe_dir = Filename.dirname Sys.executable_name in
    (match walk exe_dir 12 with
     | Some d -> d
     | None ->
       Alcotest.failf
         "could not locate repo root (no ir/VERSION + ocaml/dune-project) \
          from cwd=%s or exe_dir=%s" start exe_dir)

let root_path rel = Filename.concat repo_root rel

(* List every `*.camdl` under [dir], recursing into subdirectories so
   `tests/recovery/cases/<name>/model.camdl` is found. A subdirectory whose
   basename is in [skip_dirs] is pruned — used to skip
   `ocaml/golden/errors/`, which holds DELIBERATELY broken models (the
   `negative_golden` fixtures) that are not part of any clean corpus.
   Returns absolute paths, sorted; missing directory → []. *)
let camdl_files_under ?(skip_dirs = []) dir : string list =
  let rec collect d acc =
    if not (Sys.file_exists d && Sys.is_directory d) then acc
    else
      Sys.readdir d
      |> Array.to_list
      |> List.sort String.compare
      |> List.fold_left (fun acc entry ->
           let p = Filename.concat d entry in
           if Sys.is_directory p then
             (if List.mem entry skip_dirs then acc else collect p acc)
           else if Filename.check_suffix p ".camdl" then p :: acc
           else acc) acc
  in
  List.rev (collect dir [])

(* ── Expectation annotations ─────────────────────────────────────────────

   A fixture declares its expected diagnostics with one inline comment:

     # expect: L402 Warning
     # expect: E300 Error, E310 Error
     # expect: (none)

   We parse the (code, severity) pairs. Severity tokens are
   Error | Warning | Info (case-insensitive). `(none)` means zero
   diagnostics expected. The annotation is required: a fixture without one
   is a test authoring error. ─────────────────────────────────────────── *)

let severity_of_string s =
  match String.lowercase_ascii (trim s) with
  | "error"   -> Diagnostics.Error
  | "warning" -> Diagnostics.Warning
  | "info"    -> Diagnostics.Info
  | other -> Alcotest.failf "unknown severity token %S in expect: annotation" other

let severity_to_string = function
  | Diagnostics.Error -> "Error"
  | Diagnostics.Warning -> "Warning"
  | Diagnostics.Info -> "Info"

(* The set of expected (code, severity) pairs, sorted+deduped for stable
   comparison and printing. *)
module Pair = struct
  type t = string * Diagnostics.severity
  let compare (c1, s1) (c2, s2) =
    let c = String.compare c1 c2 in
    if c <> 0 then c else compare s1 s2
  let to_string (c, s) = Printf.sprintf "%s/%s" c (severity_to_string s)
end

let pairs_to_string ps =
  if ps = [] then "(none)"
  else String.concat ", " (List.map Pair.to_string ps)

let parse_expectation ~fixture (src : string) : Pair.t list =
  let lines = split_on '\n' src in
  let annotation =
    List.find_map (fun line ->
      let l = trim line in
      (* Accept `# expect:` with any spacing after the hash. *)
      if starts_with ~prefix:"#" l then
        let body = trim (String.sub l 1 (String.length l - 1)) in
        if starts_with ~prefix:"expect:" body then
          Some (trim (String.sub body 7 (String.length body - 7)))
        else None
      else None
    ) lines
  in
  match annotation with
  | None ->
    Alcotest.failf "fixture %s has no `# expect:` annotation" fixture
  | Some rest ->
    let rest = trim rest in
    if rest = "(none)" || rest = "" then []
    else
      rest
      |> split_on ','
      |> List.map trim
      |> List.filter (fun s -> s <> "")
      |> List.map (fun entry ->
           match words entry with
           | [code; sev] -> (code, severity_of_string sev)
           | [_code] ->
             Alcotest.failf
               "fixture %s expect-entry %S is missing a severity \
                (use `CODE Error|Warning|Info`)" fixture entry
           | _ ->
             Alcotest.failf
               "fixture %s has malformed expect-entry %S" fixture entry)
      |> List.sort_uniq Pair.compare

(* Run the real pipeline and collapse to the sorted (code, severity) set. *)
let emitted_pairs path : Pair.t list =
  let src = read_file path in
  Compiler.collect_diagnostics ~name:(Filename.basename path) ~filename:path src
  |> List.map (fun (d : Diagnostics.diagnostic) -> (d.code, d.severity))
  |> List.sort_uniq Pair.compare

(* ── Piece 2: fixture-by-code driver ─────────────────────────────────────── *)

let lints_dir = root_path "ocaml/test/lints"

let test_fixture path () =
  let src = read_file path in
  let expected = parse_expectation ~fixture:(Filename.basename path) src in
  let actual = emitted_pairs path in
  if expected <> actual then
    Alcotest.failf
      "diagnostic mismatch for %s\n  expected: %s\n  actual:   %s"
      (Filename.basename path) (pairs_to_string expected) (pairs_to_string actual)

let fixture_cases () =
  let files = camdl_files_under lints_dir in
  if files = [] then
    Alcotest.failf "no .camdl fixtures found under %s" lints_dir;
  List.map (fun path ->
    Alcotest.test_case (Filename.basename path) `Quick (test_fixture path)
  ) files

(* ── Piece 3: clean-corpus regression ────────────────────────────────────── *)

(* Allowlist of (filename-basename, code) pairs that are intentionally
   expected to fire on the corpus. Empty today: a prior check found zero
   L402 across these models, and the corpus is presumed clean of
   Error/Warning/Lint diagnostics. A future intentional case is added here
   with a one-line rationale. *)
let corpus_allowlist : (string * string) list = [
  (* (basename, code); e.g. ("some_model.camdl", "W301"); *)
  (* W105 (per-(p,q) coupling antipattern): these spatial fixtures predate the
     restricted-sum `where` construct and use the explicit per-pair importation
     form, which W105 correctly flags as O(P²). Allowlisted, not migrated:
     whether each should move to `sum(q in dim where …, …)` or keep the per-pair
     form (to exercise that path) is a per-model follow-up (gh#185). *)
  ("polio_spatial_5.camdl", "W105");
  ("seir_defines_adj.camdl", "W105");
  ("seir_spatial_5_inference.camdl", "W105");
  ("polio_afp_es_2patch.camdl", "W105");
]

(* The model corpora presumed clean. `ocaml/golden/errors/` is pruned: it
   holds deliberately-broken negative fixtures (the dimcheck/semantic error
   suite), not real models. `tests/fixtures/` and `tests/recovery/` carry real
   `.camdl` models (e.g. `polio_afp_es_2patch.camdl`), all scanned here. *)
let corpus_dirs = [
  ("ocaml/golden",   ["errors"; "data"]);
  ("tests/fixtures", []);
  ("tests/recovery", []);
]

(* Severity policy: a clean corpus must emit no Error, Warning, or Lint
   (L4xx is Warning severity). Info (I300, "dimension could not be
   determined") is non-blocking and fires on otherwise-valid models whose
   parameter dimensions are under-annotated — the existing dimcheck
   `golden_no_false_positives` test likewise ignores Info. We therefore do
   NOT treat Info as an offender. *)
let is_offending_severity = function
  | Diagnostics.Error | Diagnostics.Warning -> true
  | Diagnostics.Info -> false

let test_corpus_clean () =
  let allowed basename code =
    List.exists (fun (f, c) -> f = basename && c = code) corpus_allowlist
  in
  let files =
    List.concat_map (fun (rel, skip_dirs) ->
      camdl_files_under ~skip_dirs (root_path rel)) corpus_dirs
  in
  if files = [] then
    Alcotest.failf
      "clean-corpus test found no .camdl files under %s (repo_root=%s)"
      (String.concat ", " (List.map fst corpus_dirs)) repo_root;
  let offenders =
    List.concat_map (fun path ->
      let basename = Filename.basename path in
      emitted_pairs path
      |> List.filter (fun (code, sev) ->
           is_offending_severity sev && not (allowed basename code))
      |> List.map (fun (code, sev) ->
           Printf.sprintf "%s: %s/%s" basename code (severity_to_string sev))
    ) files
  in
  if offenders <> [] then
    Alcotest.failf
      "corpus models emitted unexpected diagnostics (presumed clean):\n  %s"
      (String.concat "\n  " offenders)

(* ── Piece 4: catalog consistency ────────────────────────────────────────── *)

(* Scan the compiler sources for emit-site codes. Codes appear two ways:
   as `~code:"Xnnn"` arguments and as bare `"Xnnn"` data passed to
   Dimcheck/Validate/Lint/Parser_errors helpers. Both reduce to the literal
   `"Xnnn"` string, so a single literal scan over .ml/.mll/.mly catches all
   of them. The .mly grammar carries parser-action emit sites (E1xx, etc.),
   so it MUST be scanned too. *)

(* Match a 4-char code "Xnnn" (uppercase letter + 3 digits) inside a
   double-quoted literal. We scan for the quoted form to avoid matching
   prose like "E300" in comments without quotes — every real emit site
   passes the code as a string literal. *)
let codes_in_source (txt : string) : string list =
  let n = String.length txt in
  let is_upper c = c >= 'A' && c <= 'Z' in
  let is_digit c = c >= '0' && c <= '9' in
  let acc = ref [] in
  let i = ref 0 in
  while !i < n do
    (* look for the pattern: '"' UPPER DIGIT DIGIT DIGIT '"' *)
    if !i + 5 < n
       && txt.[!i] = '"'
       && is_upper txt.[!i + 1]
       && is_digit txt.[!i + 2]
       && is_digit txt.[!i + 3]
       && is_digit txt.[!i + 4]
       && txt.[!i + 5] = '"'
    then begin
      acc := String.sub txt (!i + 1) 4 :: !acc;
      i := !i + 6
    end else
      incr i
  done;
  !acc

let rec source_files_under dir : string list =
  if not (Sys.file_exists dir && Sys.is_directory dir) then []
  else
    Sys.readdir dir
    |> Array.to_list
    |> List.sort String.compare
    |> List.concat_map (fun entry ->
         let p = Filename.concat dir entry in
         if Sys.is_directory p then source_files_under p
         else if List.exists (Filename.check_suffix p) [".ml"; ".mll"; ".mly"]
         then [p] else [])

module SS = Set.Make (String)

let emitted_codes () : SS.t =
  let files = source_files_under (root_path "ocaml/lib") in
  List.fold_left (fun s f ->
    List.fold_left (fun s c -> SS.add c s) s (codes_in_source (read_file f))
  ) SS.empty files

(* Parse the catalog. Two kinds of table rows carry codes in their first
   cell: a single code `| Xnnn | ... |`, and a range `| Xnnn–Xmmm | ... |`
   (en-dash or hyphen). Ranges denote a RESERVED namespace: the emit-side
   need not populate every code in a range, so ranges are exempt from the
   "every catalog code is emitted" direction, but DO cover any emitted code
   that falls inside them. Returns (single_codes, ranges) where a range is
   (prefix_char, lo, hi). *)
let parse_catalog () : SS.t * (char * int * int) list =
  let txt = read_file (root_path "docs/dev/warning-catalog.md") in
  let singles = ref SS.empty in
  let ranges = ref [] in
  List.iter (fun line ->
    let l = trim line in
    if starts_with ~prefix:"|" l then begin
      match split_on '|' l with
      | _ :: cell :: _ ->
        let cell = trim cell in
        let cn = String.length cell in
        let is_upper c = c >= 'A' && c <= 'Z' in
        let is_digit c = c >= '0' && c <= '9' in
        let parse_code_at off =
          if off + 3 < cn + 1
             && off + 3 < cn
             && is_upper cell.[off]
             && is_digit cell.[off + 1]
             && is_digit cell.[off + 2]
             && is_digit cell.[off + 3]
          then Some (cell.[off], int_of_string (String.sub cell (off + 1) 3))
          else None
        in
        (match parse_code_at 0 with
         | None -> ()
         | Some (p1, n1) ->
           (* A range looks like "Xnnn–Xmmm" or "Xnnn-Xmmm"; the dash sits at
              offset 4. The en-dash is multi-byte (UTF-8 E2 80 93), so test
              for an ASCII hyphen at 4 OR a non-ASCII byte (start of en-dash)
              at 4, followed eventually by a second code. *)
           let is_single () =
             (* cell is exactly the 4-char code (after trimming) *)
             cn = 4
           in
           if is_single () then
             singles := SS.add (Printf.sprintf "%c%03d" p1 n1) !singles
           else begin
             (* find a second code anywhere after offset 4 *)
             let second = ref None in
             let j = ref 4 in
             while !second = None && !j + 3 < cn do
               (match parse_code_at !j with
                | Some (p2, n2) when p2 = p1 -> second := Some n2
                | _ -> ());
               incr j
             done;
             (match !second with
              | Some n2 when n2 >= n1 -> ranges := (p1, n1, n2) :: !ranges
              | _ ->
                (* code followed by trailing prose, not a range: treat the
                   leading code as a single. *)
                singles := SS.add (Printf.sprintf "%c%03d" p1 n1) !singles)
           end)
      | _ -> ()
    end
  ) (split_on '\n' txt);
  (!singles, !ranges)

let code_in_range (code : string) (p, lo, hi) =
  String.length code = 4
  && code.[0] = p
  && (let n = try int_of_string (String.sub code 1 3) with _ -> -1 in
      n >= lo && n <= hi)

let test_catalog_consistency () =
  let emitted = emitted_codes () in
  let (singles, ranges) = parse_catalog () in
  let covered code =
    SS.mem code singles || List.exists (code_in_range code) ranges
  in
  (* Direction 1 (load-bearing): every emitted code is documented (by a
     single row or a covering range). Catches a new emit site shipped
     without a catalog entry. *)
  let orphan_emit =
    SS.elements emitted |> List.filter (fun c -> not (covered c))
  in
  (* Direction 2: every SINGLE-code catalog row has a live emit site.
     Range rows are reserved namespaces and exempt. Catches a stale
     catalog row for a code that no longer exists. *)
  let orphan_catalog =
    SS.elements singles |> List.filter (fun c -> not (SS.mem c emitted))
  in
  if orphan_emit <> [] then
    Alcotest.failf
      "emit sites with NO catalog row (add to docs/dev/warning-catalog.md):\n  %s"
      (String.concat ", " orphan_emit);
  if orphan_catalog <> [] then
    Alcotest.failf
      "catalog rows with NO emit site (stale — remove or implement):\n  %s"
      (String.concat ", " orphan_catalog);
  (* Non-vacuity guard: a parsing bug that produced empty sets would make
     both directions trivially pass. Assert we actually found a healthy
     number of codes on both sides. *)
  Alcotest.(check bool) "scanned a non-trivial number of emit codes"
    true (SS.cardinal emitted > 50);
  Alcotest.(check bool) "parsed a non-trivial number of catalog codes"
    true (SS.cardinal singles + List.length ranges > 20)

(* ── Piece 5: check ↔ compile parity (gh#170) ────────────────────────────────

   `camdl check` and a full compile must never disagree on whether a model
   is valid. They diverged twice — gh#9 (`check` skipped dimcheck) and
   gh#170 (`check` skipped Validate, so a dangling observation reference
   E507 passed `check` but failed `fit`/compile). The structural cure was to
   route `check`'s diagnostics through [Compiler.collect_detail], the single
   non-aborting pipeline core that [Compiler.compile] also runs. These tests
   pin that agreement so it can't drift a third time.

   We compare the two production surfaces directly:
     - CHECK side: [Compiler.collect_detail] — the exact function
       `Inspect.run_check` consumes — collapsed to its Error-severity codes.
     - COMPILE side: [Compiler.compile] — the real compile — collapsed to
       its error codes (it surfaces every error, front-end or post-expansion,
       via its [Error] return; gh#181 — it no longer raises).

   A divergence (one accepts, the other rejects; or different code sets) is
   the smell the Defect-B class is made of. *)

(* Pass the real source path as [~filename] so data-dependent models (a
   `patch` dimension `read(...)` from `data/*.tsv`) resolve their files
   relative to their own directory — exactly as [emitted_pairs] does.
   Compiling these out of their source dir would spuriously fail expansion
   (E200/E100/E263) and is not what either pipeline does in practice. *)

(* Error codes (sorted-unique) from the CHECK pipeline — i.e. the diagnostic
   surface `run_check` renders. Only Error severity counts toward
   accept/reject parity; warnings/infos (e.g. L402) don't fail a compile. *)
let check_error_codes ~filename src : string list =
  let (_detail, diags, _src) =
    Compiler.collect_detail ~name:(Filename.basename filename) ~filename src in
  diags.Diagnostics.diags
  |> List.filter (fun (d : Diagnostics.diagnostic) ->
       d.severity = Diagnostics.Error)
  |> List.map (fun (d : Diagnostics.diagnostic) -> d.code)
  |> List.sort_uniq String.compare

(* Error codes (sorted-unique) from the COMPILE pipeline. [Compiler.compile]
   prints to stderr and returns [Error json] for ANY compile failure —
   front-end (lex/parse/expand) or post-expansion (e.g. Validate E507); it
   never raises (gh#181). Under [json_errors_mode] the payload is a JSON
   array, which we parse for `"code"` fields. An [Ok] compile yields []. *)
let compile_error_codes ~filename src : string list =
  let codes_of_json (payload : string) : string list =
    (* payload is a JSON array of diagnostic objects, each with a "code" and a
       "severity". The compile payload carries EVERY diagnostic — errors,
       warnings (e.g. W104), and infos — so we filter to "error" severity to
       compare like-for-like with [check_error_codes] above, which likewise
       keeps only Error-severity codes. Accept/reject (error-set) parity is the
       property under test; a warning that fires alongside an error must not
       register as a divergence. If it isn't valid JSON (the non-json
       "compilation failed" sentinel should never appear here since we set json
       mode), fall back to []. *)
    match Yojson.Safe.from_string payload with
    | exception _ -> []
    | `List items ->
      List.filter_map (fun item ->
        match item with
        | `Assoc fields ->
          (match List.assoc_opt "severity" fields, List.assoc_opt "code" fields with
           | Some (`String "error"), Some (`String c) -> Some c
           | _ -> None)
        | _ -> None) items
      |> List.sort_uniq String.compare
    | _ -> []
  in
  let prev = !Diagnostics.json_errors_mode in
  Diagnostics.json_errors_mode := true;
  let result =
    match Compiler.compile ~name:(Filename.basename filename) ~filename src with
    | Ok _ -> []
    | Error payload -> codes_of_json payload
  in
  Diagnostics.json_errors_mode := prev;
  result

(* Focused red→green guard: the dangling-obs fixture (E507, independent of
   stratified incidence) must be rejected by BOTH pipelines with E507.
   Pre-fix, `check` accepted it (empty code set) while `compile` rejected
   (E507) — the divergence. *)
let dangling_obs_fixture = Filename.concat lints_dir "e507_dangling_obs_transition.camdl"

let test_dangling_obs_parity () =
  let src = read_file dangling_obs_fixture in
  let check_codes   = check_error_codes   ~filename:dangling_obs_fixture src in
  let compile_codes = compile_error_codes ~filename:dangling_obs_fixture src in
  Alcotest.(check bool) "check rejects (E507 present)"
    true (List.mem "E507" check_codes);
  Alcotest.(check bool) "compile rejects (E507 present)"
    true (List.mem "E507" compile_codes);
  Alcotest.(check (list string)) "check and compile agree on the error-code set"
    compile_codes check_codes

(* Structural drift guard: across every fixture (lints/ + the model
   corpora), the CHECK and COMPILE error-code sets must be IDENTICAL. This
   is the meta-test that kills the whole Defect-B class — any future
   `run_check` reimplementation that drifts from `compile` fails here.

   We exclude `ocaml/golden/errors/` (deliberately-broken dimcheck/semantic
   negatives) by reusing the same `skip_dirs` the clean-corpus test uses;
   they're still covered by the lints/ E5xx fixtures, and several are
   crafted to trip a single pass in isolation. The lints/ dir IS included —
   it carries the E507 fixture plus L402/clean models, exercising the
   warning-vs-error boundary of the parity (a lint warns in both; it must
   NOT show up as an error-code divergence). *)
let parity_corpus_dirs = [
  ("ocaml/test/lints", []);
  ("ocaml/golden",     ["errors"; "data"]);
  ("tests/fixtures",   []);
  ("tests/recovery",   []);
]

let test_corpus_check_compile_parity () =
  let files =
    List.concat_map (fun (rel, skip_dirs) ->
      camdl_files_under ~skip_dirs (root_path rel)) parity_corpus_dirs
  in
  if files = [] then
    Alcotest.failf
      "parity test found no .camdl files under %s (repo_root=%s)"
      (String.concat ", " (List.map fst parity_corpus_dirs)) repo_root;
  let divergences =
    List.filter_map (fun path ->
      let src = read_file path in
      let check_codes   = check_error_codes   ~filename:path src in
      let compile_codes = compile_error_codes ~filename:path src in
      if check_codes <> compile_codes then
        Some (Printf.sprintf
          "%s\n      check:   [%s]\n      compile: [%s]"
          (Filename.basename path)
          (String.concat ", " check_codes)
          (String.concat ", " compile_codes))
      else None
    ) files
  in
  if divergences <> [] then
    Alcotest.failf
      "check ↔ compile error-code divergence on %d fixture(s):\n    %s"
      (List.length divergences) (String.concat "\n    " divergences)

(* ── Piece 6: one lookup site for `dim_registry` (A1) ────────────────────────

   `Expander.dim_values` is the single accessor for a dimension's levels, and
   it returns `option` so "no such dimension" cannot masquerade as "a dimension
   with no levels". A hand-inlined `List.assoc_opt <d> ctx.dim_registry` with a
   `[]` fallback re-creates exactly the collapse the accessor exists to remove,
   and the type checker cannot see it — ten such sites existed before A1
   (aggregation-semantics proposal §6). So the invariant is pinned by a source
   scan instead: every value lookup into `dim_registry` goes through the one
   accessor.

   `List.mem_assoc … ctx.dim_registry` is deliberately NOT counted: it is a
   membership test (the E212 redeclaration guard and the E214 stratify check),
   not a levels lookup, and it has no silent default to hide. *)
let dim_registry_lookup_lines () : (string * int * string) list =
  let files =
    List.filter (fun p ->
      List.mem (Filename.basename p) ["expander.ml"; "inspect.ml"])
      (source_files_under (root_path "ocaml/lib/compiler"))
  in
  List.concat_map (fun path ->
    let has needle line = List.exists (fun i ->
      i + String.length needle <= String.length line
      && String.sub line i (String.length needle) = needle)
      (List.init (max 0 (String.length line - String.length needle + 1))
         (fun i -> i))
    in
    String.split_on_char '\n' (read_file path)
    |> List.mapi (fun i l -> (i + 1, l))
    |> List.filter_map (fun (n, l) ->
         if has "dim_registry" l && has "assoc_opt" l
         then Some (Filename.basename path, n, String.trim l) else None)
  ) files

let test_dim_registry_single_lookup_site () =
  let sites = dim_registry_lookup_lines () in
  (* Non-vacuity: the scan must find the accessor itself, or it is not
     looking at the right sources. *)
  Alcotest.(check bool) "the scan reaches expander.ml" true
    (List.exists (fun (f, _, _) -> f = "expander.ml") sites);
  match sites with
  | [(_, _, _)] -> ()
  | _ ->
    Alcotest.failf
      "expected exactly ONE `assoc_opt … dim_registry` lookup (inside \
       Expander.dim_values); found %d — route the others through the accessor:\n  %s"
      (List.length sites)
      (String.concat "\n  "
         (List.map (fun (f, n, l) -> Printf.sprintf "%s:%d  %s" f n l) sites))

(* ── Driver ──────────────────────────────────────────────────────────────── *)

let () =
  Alcotest.run "diagnostics" [
    "dim_registry_accessor_a1", [
      Alcotest.test_case "dim_registry is read through one accessor"
        `Quick test_dim_registry_single_lookup_site;
    ];
    "fixtures", fixture_cases ();
    "corpus", [
      Alcotest.test_case "model corpus is diagnostic-clean" `Quick test_corpus_clean;
    ];
    "catalog", [
      Alcotest.test_case "emit sites ↔ warning-catalog.md" `Quick test_catalog_consistency;
    ];
    "check-compile-parity", [
      Alcotest.test_case "dangling obs (E507) rejected by check AND compile"
        `Quick test_dangling_obs_parity;
      Alcotest.test_case "every fixture: check error-set = compile error-set"
        `Quick test_corpus_check_compile_parity;
    ];
  ]
