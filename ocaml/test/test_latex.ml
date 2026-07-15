(* `camdlc render` — LaTeX projection of the indexed (pre-expansion) model.

   The renderer (Latex.of_model) is a pure Ast -> string projection, a sibling
   to the simulation backend: it taps the AST before stratification expansion so
   the output reads in the mathematician's indexed form (S_{r,a}), with an
   optional --expand to unfold selected dimensions to their literal levels.

   These tests pin the load-bearing pieces of that projection:
   - the @symbol override path (beta -> β) vs the greek fallback (beta -> \beta),
   - product/quotient flattening to a single \frac,
   - derived-ODE assembly from stoichiometry with correct signs,
   - single merged subscripts on indexed refs (guards the double-subscript bug),
   - selective --expand (unfold one dimension, keep the others as binders). *)

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

let parse_to_ast (src : string) : Ast.declaration list =
  let lexbuf = Lexing.from_string src in
  Lexing.set_filename lexbuf "<test>";
  Parser.file Lexer.token lexbuf

let render ?(expand = []) (src : string) : string =
  Latex.of_model ~name:"test" ~expand (parse_to_ast src)

let has ~doc needle out =
  Alcotest.(check bool) (Printf.sprintf "contains %s" doc) true
    (contains_substring ~needle out)

let has_not ~doc needle out =
  Alcotest.(check bool) (Printf.sprintf "absent %s" doc) false
    (contains_substring ~needle out)

(* ── fixtures ─────────────────────────────────────────────────────────── *)

(* Minimal SIR carrying @symbol overrides: beta -> β, gamma -> γ. *)
let sir_symbol = {|
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  #' @symbol β
  beta  : rate in [0.001, 2.0]
  #' @symbol γ
  gamma : rate in [0.001, 1.0]
}
transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}
|}

(* Same model with the @symbol tags stripped: exercises the greek fallback. *)
let sir_plain = {|
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}
transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}
|}

let sir_stratified = {|
time_unit = 'days
compartments { S, I, R }
dimensions {
  region = [north, south]
  age    = [child, adult]
}
stratify(by = region)
stratify(by = age)
let N[r in region, a in age] = S[r, a] + I[r, a] + R[r, a]
parameters {
  beta[region, age] : rate in [0.001, 2.0]
  gamma             : rate in [0.01,  1.0]
}
transitions {
  infection[r in region, a in age] : S[r, a] --> I[r, a]
    @ beta[r, a] * S[r, a] * I[r, a] / N[r, a]
  recovery[r in region, a in age]  : I[r, a] --> R[r, a] @ gamma * I[r, a]
}
|}

(* ── tests ────────────────────────────────────────────────────────────── *)

let test_document_scaffold () =
  let out = render sir_symbol in
  has  ~doc:"document open"  "\\begin{document}" out;
  has  ~doc:"document close" "\\end{document}" out;
  has  ~doc:"transitions heading"     "\\textbf{Transitions}" out;
  has  ~doc:"derived-dynamics heading" "\\textbf{Derived dynamics}" out;
  has  ~doc:"reaction arrow" "\\xrightarrow" out

let test_symbol_override_and_frac () =
  let out = render sir_symbol in
  (* @symbol makes beta/gamma render as literal β/γ … *)
  has ~doc:"β override" "β" out;
  has ~doc:"γ override" "γ" out;
  (* … and the product beta*S*(I/N) flattens to a single fraction. *)
  has ~doc:"flattened FOI fraction" "\\frac{β\\,S\\,I}{N}" out

let test_let_becomes_definition () =
  let out = render sir_symbol in
  has ~doc:"Definitions heading" "\\textbf{Definitions}" out;
  has ~doc:"let N as definition" "N &= S + I + R" out

let test_derived_ode_signs () =
  let out = render sir_symbol in
  (* source loses the flow (negative), sink gains it (positive). *)
  has ~doc:"S loses FOI"  "\\dot{S} &= -\\frac{β\\,S\\,I}{N}" out;
  has ~doc:"R gains gamma*I" "\\dot{R} &= γ\\,I" out

(* Negative control: without @symbol the same names fall back to \beta / \gamma,
   and the literal-unicode forms must NOT appear — this proves the override in
   the tests above is load-bearing, not incidental. *)
let test_greek_fallback () =
  let out = render sir_plain in
  has     ~doc:"\\beta fallback"  "\\beta" out;
  has     ~doc:"\\gamma fallback" "\\gamma" out;
  has_not ~doc:"no β override" "β" out;
  has_not ~doc:"no γ override" "γ" out

(* Indexed refs render one merged subscript S_{r,a}, never S_{r}_{a}. *)
let test_indexed_subscripts () =
  let out = render sir_stratified in
  has     ~doc:"merged compartment subscript" "S_{r,a}" out;
  has     ~doc:"indexed param subscript" "\\beta_{r,a}" out;
  has     ~doc:"indexed FOI fraction"
    "\\frac{\\beta_{r,a}\\,S_{r,a}\\,I_{r,a}}{N_{r,a}}" out;
  (* The let LHS carries ALL its index binders, matching every use site of N —
     not just the first binder (which would render the stale N_{r}). *)
  has     ~doc:"let LHS full index" "N_{r,a} &=" out;
  has_not ~doc:"no double subscript" "S_{r}_{a}" out

(* --expand region unfolds region to its literal levels while age stays a
   binder: S_{north,a} / S_{south,a} appear, bare S_{r,a} is gone. *)
let test_expand_selected_dimension () =
  let out = render ~expand:[ "region" ] sir_stratified in
  (* region unfolds to its literal levels; age stays the binder `a`. *)
  has     ~doc:"north instance" "S_{\\mathrm{north},a}" out;
  has     ~doc:"south instance" "S_{\\mathrm{south},a}" out;
  (* selectivity: age is NOT expanded, so no age level leaks into an index —
     its levels only appear in the \\text{...} dimension listing. *)
  has_not ~doc:"age level not in index" "\\mathrm{child}" out;
  has_not ~doc:"age level not in index" "\\mathrm{adult}" out

let () =
  Alcotest.run "latex"
    [ ( "scaffold",
        [ Alcotest.test_case "document scaffold + headings" `Quick
            test_document_scaffold ] );
      ( "symbols",
        [ Alcotest.test_case "@symbol override + \\frac flattening" `Quick
            test_symbol_override_and_frac;
          Alcotest.test_case "greek fallback (negative control)" `Quick
            test_greek_fallback ] );
      ( "structure",
        [ Alcotest.test_case "let -> Definitions block" `Quick
            test_let_becomes_definition;
          Alcotest.test_case "derived-ODE signs from stoichiometry" `Quick
            test_derived_ode_signs ] );
      ( "indexing",
        [ Alcotest.test_case "merged subscripts on indexed refs" `Quick
            test_indexed_subscripts;
          Alcotest.test_case "--expand unfolds one dimension" `Quick
            test_expand_selected_dimension ] ) ]
