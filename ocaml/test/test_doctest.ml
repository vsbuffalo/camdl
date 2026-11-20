(* Self-test for the doc-doctest classifier (Doctest module in the compiler lib).

   Drives the real parser + classifier over a fixture Markdown file whose blocks
   are ordered to exercise every verdict bucket, including a `preamble=` block
   (whose preamble is a ```camdl-doctest-preamble fence) and a `read()` block
   resolved by inline `camdl-doctest-data`. Non-vacuous: it asserts the FAIL block really reports
   E300, that the preamble block fails when its preamble is stripped, and that
   the data block needs its inline data. *)

let spec = "doctest_fixtures/spec.md"

let verdict_name : Doctest.verdict -> string = function
  | Doctest.Pass          -> "pass"
  | Doctest.Skip_ignore   -> "skip:ignore"
  | Doctest.Skip_parse    -> "skip:parse"
  | Doctest.Skip_data     -> "skip:data"
  | Doctest.Skip_fragment -> "skip:fragment"
  | Doctest.Fail _        -> "fail"
  | Doctest.Ice msg       -> "ice:" ^ msg

(* Expected verdict for each block, in file order. *)
let expected =
  [ "pass"           (* 1. clean model *)
  ; "skip:parse"     (* 2. bare expression *)
  ; "skip:fragment"  (* 3. transitions-only fragment *)
  ; "skip:data"      (* 4. read(), no inline data *)
  ; "fail"           (* 5. complete model, E300 *)
  ; "skip:ignore"    (* 6. ```camdl ignore *)
  ; "pass"           (* 7. fragment + preamble=demo *)
  ; "pass"           (* 8. read() + inline data *)
  ; "fail"           (* 9. complete model, syntax error (gh#1216) *)
  ; "skip:fragment"  (* 10. fragment, `compartments` only in a comment *)
  ; "skip:parse"     (* 11. legend, `compartments` as an index range *)
  ]

let test_classifications () =
  let doc = Doctest.parse_doc spec in
  let basedir = Doctest.materialize_data doc.datas in
  Alcotest.(check int) "block count" (List.length expected) (List.length doc.blocks);
  List.iteri
    (fun i (want, b) ->
       let v = verdict_name (Doctest.classify ~preambles:doc.preambles ~basedir b) in
       Alcotest.(check string)
         (Printf.sprintf "block %d @ L%d" (i + 1) b.Doctest.line) want v)
    (List.combine expected doc.blocks);
  Doctest.rm_rf basedir

let test_non_vacuous () =
  let doc = Doctest.parse_doc spec in
  let basedir = Doctest.materialize_data doc.datas in
  let nth = List.nth doc.blocks in
  (* block 5: a genuine E300, not a vacuous pass *)
  (match Doctest.classify ~preambles:doc.preambles ~basedir (nth 4) with
   | Doctest.Fail errs ->
     let codes = List.map (fun (d : Diagnostics.diagnostic) -> d.code) errs in
     Alcotest.(check bool) "block 5 reports E300" true (List.mem "E300" codes)
   | other -> Alcotest.failf "block 5: expected Fail, got %s" (verdict_name other));
  (* block 7 passes only because of its preamble *)
  let stripped = { (nth 6) with Doctest.preamble = None } in
  Alcotest.(check bool) "preamble block fails without its preamble" false
    (match Doctest.classify ~preambles:doc.preambles ~basedir stripped with
     | Doctest.Pass -> true | _ -> false);
  (* block 8 passes only because of its inline data *)
  let empty = Doctest.make_temp_dir () in
  Alcotest.(check bool) "data block needs its inline data" true
    (match Doctest.classify ~preambles:doc.preambles ~basedir:empty (nth 7) with
     | Doctest.Skip_data -> true | _ -> false);
  Doctest.rm_rf empty;
  Doctest.rm_rf basedir

(* gh#1216: a data comment's path must stay inside the temporary directory. *)
let test_bad_data_paths () =
  let datas =
    [ ("data/pop.tsv", ""); ("../up.tsv", ""); ("data/../../up.tsv", "");
      ("/abs.tsv", ""); ("data/..x.tsv", "") ] in
  Alcotest.(check (list string)) "escaping paths"
    [ "../up.tsv"; "data/../../up.tsv"; "/abs.tsv" ]
    (Doctest.bad_data_paths datas);
  let dir = Doctest.materialize_data [ ("../escaped-by-doctest-test.tsv", "x\n") ] in
  let outside = Filename.concat (Filename.dirname dir) "escaped-by-doctest-test.tsv" in
  let escaped = Sys.file_exists outside in
  if escaped then Sys.remove outside;
  Doctest.rm_rf dir;
  Alcotest.(check bool) "a '..' path is not written" false escaped

(* gh#1324: a preamble written as an HTML comment leaks onto the rendered
   page, because the first `-->` of a transition ends the comment there. The
   comment form is not read as a preamble any more, and is reported. *)
let test_comment_preamble_is_refused () =
  let file = Filename.temp_file "camdl-doctest-gh1324-" ".md" in
  let oc = open_out file in
  output_string oc
    "Text.\n\n\
     <!-- camdl-doctest-preamble: old\n\
     compartments { S, I }\n\
     transitions { infection : S --> I @ S }\n\
     -->\n\n\
     ```camdl preamble=old\n\
     parameters { beta : rate }\n\
     ```\n";
  close_out oc;
  let doc = Doctest.parse_doc file in
  Sys.remove file;
  Alcotest.(check (list int)) "the comment preamble's line is reported" [ 3 ]
    doc.Doctest.comment_preambles;
  Alcotest.(check (list string)) "and it defines no preamble" []
    (List.map fst doc.Doctest.preambles);
  Alcotest.(check int) "the camdl block is still read" 1 (List.length doc.Doctest.blocks)

(* gh#1324: a data comment whose body has `-->` would end on the rendered page
   at that text, as a comment preamble did. *)
let test_data_comment_with_arrow () =
  Alcotest.(check (list string)) "the data comment that has -->" [ "b.tsv" ]
    (Doctest.data_comments_ended_early
       [ ("a.tsv", "x\ty\n1\t2\n"); ("b.tsv", "edge\nS --> I\n") ])

(* gh#1324: a preamble fence defines a preamble and is not itself a block. *)
let test_preamble_fence () =
  let doc = Doctest.parse_doc spec in
  Alcotest.(check (list string)) "the fence defines preamble demo" [ "demo" ]
    (List.map fst doc.Doctest.preambles);
  Alcotest.(check bool) "its text is the model text, arrows included" true
    (Doctest.contains ~sub:"recovery : I --> R @ gamma * I\n}\n"
       (List.assoc "demo" doc.Doctest.preambles))

let () =
  Alcotest.run "doctest"
    [ ("classify",
       [ Alcotest.test_case "every verdict bucket" `Quick test_classifications
       ; Alcotest.test_case "fail / preamble / data are non-vacuous" `Quick test_non_vacuous
       ; Alcotest.test_case "data paths stay in the temporary directory" `Quick test_bad_data_paths
       ; Alcotest.test_case "a preamble fence defines a preamble" `Quick test_preamble_fence
       ; Alcotest.test_case "a comment preamble is refused" `Quick test_comment_preamble_is_refused
       ; Alcotest.test_case "a data comment with --> is refused" `Quick test_data_comment_with_arrow
       ]) ]
