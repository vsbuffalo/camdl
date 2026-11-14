(* The declared-model codec (`Declared`), and its agreement with the Rust
   `declared` crate.

   The committed fixture `rust/crates/declared/tests/fixtures/all_variants.json`
   is what both languages read: this test checks it is exactly what
   `Declared_fixture` builds, and `rust/crates/declared/tests/ocaml_fixture.rs`
   reads it through the Rust types and writes it back. The file lives outside
   the `ocaml/` dune root, so it is found by resolving the repo root at run
   time, as `test_caltime_golden.ml` does. *)

open Declared

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
  match walk (Sys.getcwd ()) 12 with
  | Some d -> d
  | None ->
    (match walk (Filename.dirname Sys.executable_name) 12 with
     | Some d -> d
     | None -> Alcotest.failf "could not locate repo root")

let read_file path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

let committed () =
  read_file (Filename.concat repo_root Declared_fixture.fixture_path)

let member key = function
  | `Assoc kvs -> List.assoc key kvs
  | _ -> Alcotest.failf "fixture is not an object"

(* The committed file holds exactly the JSON the generator builds. Compared as
   JSON trees and not as bytes, so the check does not depend on the installed
   yojson's pretty-printer. *)
let test_fixture_is_current () =
  if Yojson.Safe.from_string (committed ()) <> Declared_fixture.fixture_json then
    Alcotest.failf
      "%s is not what Declared_fixture builds; run `make \
       update-declared-fixture` and review the diff"
      Declared_fixture.fixture_path

(* Reading the committed file gives back the value it was written from. *)
let test_reads_committed_fixture () =
  let j = Yojson.Safe.from_string (committed ()) in
  Alcotest.(check bool)
    "all_variants" true
    (of_json (member "all_variants" j) = Declared_fixture.model);
  Alcotest.(check bool)
    "minimal" true
    (of_json (member "minimal" j) = Declared_fixture.minimal)

(* Writing and reading through text, not only through the JSON tree. *)
let test_text_round_trip () =
  List.iter
    (fun m ->
      let text = Yojson.Safe.to_string (to_json m) in
      Alcotest.(check bool)
        "round trip" true
        (of_json (Yojson.Safe.from_string text) = m))
    [ Declared_fixture.model; Declared_fixture.minimal ]

let refused what f =
  match f () with
  | exception Serde.DeserError _ -> ()
  | _ -> Alcotest.failf "%s was accepted" what

(* Replace the value under a path of object keys. *)
let rec replace path f (j : Yojson.Safe.t) : Yojson.Safe.t =
  match (path, j) with
  | [], _ -> f j
  | k :: rest, `Assoc kvs ->
    if not (List.mem_assoc k kvs) then Alcotest.failf "no key '%s'" k;
    `Assoc
      (List.map (fun (k', v) -> if k' = k then (k', replace rest f v) else (k', v)) kvs)
  | k :: _, _ -> Alcotest.failf "no object at '%s'" k

let minimal_json () = to_json Declared_fixture.minimal

(* An unknown field, a missing field, an unknown variant, a wrongly typed
   value and an unknown schema version are each refused. *)
let test_refusals () =
  let add_field = function
    | `Assoc kvs -> `Assoc (kvs @ [ ("extra", `Null) ])
    | j -> j
  in
  let drop key = function
    | `Assoc kvs -> `Assoc (List.remove_assoc key kvs)
    | j -> j
  in
  refused "an unknown top-level field" (fun () ->
      of_json (add_field (minimal_json ())));
  refused "an unknown nested field" (fun () ->
      of_json (replace [ "compile_inputs" ] add_field (minimal_json ())));
  refused "a missing field" (fun () -> of_json (drop "contrasts" (minimal_json ())));
  refused "an unknown unit" (fun () ->
      of_json (replace [ "time_unit" ] (fun _ -> `String "fortnights") (minimal_json ())));
  refused "a number where a string is expected" (fun () ->
      of_json (replace [ "description" ] (fun _ -> `Int 3) (minimal_json ())));
  refused "a schema version this code does not write" (fun () ->
      of_json
        (replace [ "schema_version" ] (fun _ -> `Int (schema_version + 1)) (minimal_json ())));
  refused "an unknown expression variant" (fun () ->
      expr_tree_of_json (`Assoc [ ("paren", `Assoc [ ("arg", `Null) ]) ]));
  refused "a variant with two tags" (fun () ->
      expr_tree_of_json
        (`Assoc
           [ ("const", `Assoc [ ("value", `Float 1.0) ]);
             ("unit", `Assoc [ ("value", `Float 1.0) ]) ]))

(* JSON cannot carry a non-finite number; the writer refuses to try. *)
let test_non_finite_refused () =
  List.iter
    (fun v ->
      match expr_tree_to_json (TConst v) with
      | exception Invalid_argument _ -> ()
      | _ -> Alcotest.failf "a non-finite constant was written")
    [ Float.nan; Float.infinity; Float.neg_infinity ]

(* A whole number is written as a float, so the Rust side reads back a float
   and the two JSON trees compare equal. *)
let test_whole_number_is_a_float () =
  Alcotest.(check string)
    "7 is written 7.0" {|{"const":{"value":7.0}}|}
    (Yojson.Safe.to_string (expr_tree_to_json (TConst 7.0)))

let () =
  Alcotest.run "declared"
    [ ( "fixture",
        [ Alcotest.test_case "committed file is current" `Quick test_fixture_is_current;
          Alcotest.test_case "reads the committed file" `Quick test_reads_committed_fixture ] );
      ( "codec",
        [ Alcotest.test_case "text round trip" `Quick test_text_round_trip;
          Alcotest.test_case "refusals" `Quick test_refusals;
          Alcotest.test_case "non-finite numbers" `Quick test_non_finite_refused;
          Alcotest.test_case "whole numbers" `Quick test_whole_number_is_a_float ] ) ]
