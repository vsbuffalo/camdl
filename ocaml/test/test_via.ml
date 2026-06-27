(* Phase-1 staged-residence `via` clause: parser + AST tests.

   Phase 1 is pure frontend — the `via law(...)` clause parses into a
   `trdyn = Via via_call`, the `@`-XOR-`via` rule is enforced, and a `via`
   transition reaching expansion produces a clean "not yet implemented"
   diagnostic (E243), never a crash or a silent rate-0 transition. No IR /
   Rust / golden change rides in this phase. *)

(* Disable dimcheck — these tests exercise parse + AST + the expansion
   placeholder, not dimensional analysis. *)
let () = Compiler.no_dim_check := true

let contains_substring ~needle s =
  let nl = String.length needle and sl = String.length s in
  if nl = 0 then true
  else if nl > sl then false
  else
    let rec loop i =
      if i > sl - nl then false
      else if String.sub s i nl = needle then true
      else loop (i + 1)
    in loop 0

(* Parse a source string straight to the declaration list (AST), bypassing
   expansion, so we can assert on the parsed `transition_decl`. Raises on a
   lex/parse error (the test then fails loudly). *)
let parse_to_ast (src : string) : Ast.declaration list =
  let lexbuf = Lexing.from_string src in
  Lexing.set_filename lexbuf "<test>";
  Parser.file Lexer.token lexbuf

(* Pull the single transition declaration out of a parsed model that has
   exactly one `transitions {}` block with one transition. *)
let sole_transition (decls : Ast.declaration list) : Ast.transition_decl =
  let trs =
    List.concat_map
      (function Ast.DTransitions ts -> ts | _ -> [])
      decls
  in
  match trs with
  | [ t ] -> t
  | _ ->
    Alcotest.failf "expected exactly one transition, got %d" (List.length trs)

(* A minimal model wrapping a single transition body, so the parser has a
   well-formed file to consume. The transition is the only one. *)
let model_with_transition (tr_src : string) : string =
  Printf.sprintf
    "compartments { S, E, I, R }\n\
     parameters { sigma : rate\n  beta : rate\n  r : rate }\n\
     transitions {\n  %s\n}\n"
    tr_src

(* Compile through the real front end with JSON-diagnostics so the Error
   payload carries the error codes. *)
let compile_expect_error_code ~code ~contains src =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"test_via" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

(* ── 1. inline `via` parses to a Via via_call with NO rate ──────────────── *)

let test_inline_via_parses () =
  let src =
    model_with_transition
      "onset : E --> I via erlang(stages = 3, mean = 7 'days)"
  in
  let tr = sole_transition (parse_to_ast src) in
  (* The dynamics must be a Via carrying the raw law call — NOT a Rate. *)
  match tr.Ast.trdyn with
  | Ast.Rate _ ->
    Alcotest.fail "expected Via dynamics, got Rate (the @-XOR-via encoding leaked a rate)"
  | Ast.Via (law_name, args) ->
    Alcotest.(check string) "law name" "erlang" law_name;
    let arg_keys = List.map fst args in
    Alcotest.(check (list string)) "arg keys, in order"
      [ "stages"; "mean" ] arg_keys;
    (* stages = 3 → an integer constant *)
    (match List.assoc "stages" args with
     | Ast.EConst 3.0 -> ()
     | _ -> Alcotest.fail "stages arg: expected EConst 3.0");
    (* mean = 7 'days → a unit literal *)
    (match List.assoc "mean" args with
     | Ast.EUnit (7.0, Ast.Days) -> ()
     | _ -> Alcotest.fail "mean arg: expected EUnit(7.0, Days)")

(* ── 2. `@`-XOR-`via`: rate AND via, both orderings → a parse error ─────── *)

let test_inline_rate_then_via_rejected () =
  (* `E --> I @ r via erlang(...)` — `@ rate` followed by `via` is not a
     legal inline transition (via REPLACES @). A bare E001 syntax error is
     acceptable here; the point is it does NOT silently parse as one or the
     other. *)
  let src =
    model_with_transition
      "onset : E --> I @ r via erlang(stages = 3, mean = 7 'days)"
  in
  match Compiler.compile ~name:"test_via" src with
  | Ok _ -> Alcotest.fail "expected a parse error for `@ r via ...`, compiled OK"
  | Error _ -> ()

let test_inline_via_then_rate_rejected () =
  (* `E --> I via erlang(...) @ r` — `via` followed by `@ rate` likewise. *)
  let src =
    model_with_transition
      "onset : E --> I via erlang(stages = 3, mean = 7 'days) @ r"
  in
  match Compiler.compile ~name:"test_via" src with
  | Ok _ -> Alcotest.fail "expected a parse error for `via ... @ r`, compiled OK"
  | Error _ -> ()

(* A transition with NEITHER @ nor via still hits the existing missing-rate
   diagnostic (block form is the only place "neither" is reachable — the
   inline grammar requires one of @/via). *)
let test_block_neither_rate_nor_via_rejected () =
  let src = model_with_transition "onset : E --> I { }" in
  compile_expect_error_code ~code:"E213" ~contains:"onset" src

(* ── 3. block form: `via =` parses; `rate = ... via = ...` → error ─────── *)

let test_block_via_parses () =
  let src =
    model_with_transition
      "onset : E --> I { via = erlang(stages = 3, mean = 7 'days) }"
  in
  let tr = sole_transition (parse_to_ast src) in
  match tr.Ast.trdyn with
  | Ast.Rate _ -> Alcotest.fail "block `via =` parsed as Rate, expected Via"
  | Ast.Via (law_name, args) ->
    Alcotest.(check string) "law name" "erlang" law_name;
    Alcotest.(check (list string)) "arg keys"
      [ "stages"; "mean" ] (List.map fst args)

let test_block_both_rate_and_via_rejected () =
  let src =
    model_with_transition
      "onset : E --> I { rate = r  via = erlang(stages = 3, mean = 7 'days) }"
  in
  compile_expect_error_code ~code:"E112" ~contains:"onset" src

(* ── 4. a `via` transition compiled end-to-end hits the clean E243 ──────── *)

let test_via_compile_not_yet_implemented () =
  let src =
    model_with_transition
      "onset : E --> I via erlang(stages = 3, mean = 7 'days)"
  in
  (* Must be a clean located E243, NOT a crash / failwith / rate-0 misbehavior. *)
  compile_expect_error_code ~code:"E243" ~contains:"onset" src

(* The error message must name the not-yet-implemented feature so the user
   knows it parsed but cannot lower yet. *)
let test_via_e288_mentions_via () =
  let src =
    model_with_transition
      "onset : E --> I via erlang(stages = 3, mean = 7 'days)"
  in
  compile_expect_error_code ~code:"E243" ~contains:"not yet implemented" src

(* ── golden-neutrality smoke: a plain `@ rate` model still parses to Rate ── *)

let test_ordinary_rate_still_parses_as_rate () =
  let src = model_with_transition "onset : E --> I @ sigma * E" in
  let tr = sole_transition (parse_to_ast src) in
  match tr.Ast.trdyn with
  | Ast.Rate _ -> ()
  | Ast.Via _ -> Alcotest.fail "ordinary `@ rate` parsed as Via — encoding regression"

let () =
  Alcotest.run "via"
    [ ( "parse",
        [ Alcotest.test_case "inline via parses to Via via_call" `Quick
            test_inline_via_parses;
          Alcotest.test_case "block via parses to Via via_call" `Quick
            test_block_via_parses;
          Alcotest.test_case "ordinary @ rate still parses as Rate" `Quick
            test_ordinary_rate_still_parses_as_rate ] );
      ( "xor",
        [ Alcotest.test_case "inline @ r via ... rejected" `Quick
            test_inline_rate_then_via_rejected;
          Alcotest.test_case "inline via ... @ r rejected" `Quick
            test_inline_via_then_rate_rejected;
          Alcotest.test_case "block neither rate nor via rejected (E213)" `Quick
            test_block_neither_rate_nor_via_rejected;
          Alcotest.test_case "block both rate and via rejected (E112)" `Quick
            test_block_both_rate_and_via_rejected ] );
      ( "lowering-placeholder",
        [ Alcotest.test_case "via compile → E243 not yet implemented" `Quick
            test_via_compile_not_yet_implemented;
          Alcotest.test_case "E243 message names via / not yet implemented"
            `Quick test_via_e288_mentions_via ] ) ]
