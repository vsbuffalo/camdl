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

let render_json ?(expand = []) (src : string) : string =
  Latex.to_json (Latex.render_model ~name:"test" ~expand (parse_to_ast src))

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

(* An age-stratified SIR with the load-bearing flow-graph features: a
   mean-field pool read through `let` bindings (foi ← inf_tot/ntot, both full-age
   sums), a `consecutive(age)` aging edge over the compartment iterator `c`, an
   exogenous birth inflow (empty LHS), and a death outflow (empty RHS). This is
   the ctl_bb shape reduced to the essentials the emitter must get right. *)
let sir_aging_pool = {|
time_unit = 'days
compartments { S, I, R }
dimensions { age = [young, old] }
stratify(by = age)
let N[a in age] = S[a] + I[a] + R[a]
let inf_tot = sum(a in age, I[a])
let ntot    = sum(a in age, N[a])
let foi     = beta * (inf_tot / ntot)
parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
  mu    : rate in [0.0,   0.1]
}
transitions {
  infection[a in age] : S[a] --> I[a] @ foi * S[a]
  recovery[a in age]  : I[a] --> R[a] @ gamma * I[a]
  aging[c in compartments, (a, a_next) in consecutive(age)] : c[a] --> c[a_next] @ mu * c[a]
  birth : --> S[young] @ mu * ntot
  death[c in compartments, a in age] : c[a] --> @ mu * c[a]
}
|}

(* A stratified SEIR whose FOI reads an INLINE age-mixing sum (not a named `let`
   pool) — exercises the anonymous-aggregate branch (aggregate = "sum"). *)
let seir_inline_sum = {|
time_unit = 'days
compartments { S, E, I, R }
dimensions { age = [child, adult] }
stratify(by = age)
let N_local[a in age] = S[a] + E[a] + I[a] + R[a]
parameters {
  beta  : rate in [0.001, 0.5]
  sigma : rate in [0.01, 1.0]
  gamma : rate in [0.01, 1.0]
}
tables { C_age : age × age = [[12.0, 4.0], [4.0, 8.0]] }
transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
  progression[a in age] : E[a] --> I[a] @ sigma * E[a]
  recovery[a in age]    : I[a] --> R[a] @ gamma * I[a]
}
|}

(* ── graph helpers ────────────────────────────────────────────────────── *)

let graph ?(name = "test") (src : string) : Latex.model_graph =
  Latex.build_graph ~name (parse_to_ast src)

let node_ids (g : Latex.model_graph) : string list =
  List.map (fun (n : Latex.g_node) -> n.gn_id) g.mg_nodes

let plate (g : Latex.model_graph) (name : string) : Latex.g_plate =
  List.find (fun (p : Latex.g_plate) -> p.gp_name = name) g.mg_plates

let edge (g : Latex.model_graph) (id : string) : Latex.g_edge =
  List.find (fun (e : Latex.g_edge) -> e.ge_id = id) g.mg_edges

let couplings (g : Latex.model_graph) (edge_id : string) : (string * string list) list =
  List.filter_map
    (fun (c : Latex.g_coupling) ->
      if c.gc_edge = edge_id then Some (c.gc_aggregate, c.gc_over) else None)
    g.mg_couplings

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

(* gh#527: `@symbol` on a `let`. gh#508 gave `let` a doc slot and the language
   spec advertised the tag as working "as it does elsewhere", but
   `populate_overrides` harvested only compartments and parameters — so the
   override parsed, stored, and was never read. A `let` is where it earns its
   keep: the code name (`N_total`) and the paper symbol (`N`) diverge for the
   same reason a compartment's do. The override must reach BOTH the definition
   line and every use of the name in the derived dynamics, or the document
   contradicts itself. *)
let sir_let_symbol = {|
time_unit = 'days
compartments { S, I, R }
#' total population, the FOI denominator
#' @symbol NTOT
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

let test_let_symbol_override () =
  let out = render sir_let_symbol in
  has ~doc:"let's @symbol on the definition LHS" "NTOT &= S + I + R" out;
  has ~doc:"let's @symbol at its use in the rate"
      "\\frac{\\beta\\,S\\,I}{NTOT}" out;
  (* And the un-overridden name must be gone from those positions, or the
     override is merely additive rather than a substitution. *)
  has_not ~doc:"bare N as the definition LHS" "N &= S + I + R" out

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

(* The JSON projection is well-formed, splits transitions into parts (for a
   reaction table, not a pre-assembled arrow), keys dynamics by state, carries
   the @symbol glossary, and JSON-escapes the LaTeX backslashes. *)
let test_json_shape () =
  let out = render_json sir_symbol in
  (match Yojson.Safe.from_string out with
   | _ -> ()
   | exception _ -> Alcotest.failf "to_json did not produce well-formed JSON:\n%s" out);
  has     ~doc:"reactants part"      "\"reactants\"" out;
  has     ~doc:"rate part"           "\"rate\"" out;
  has_not ~doc:"no assembled arrow"  "xrightarrow" out;
  has     ~doc:"dynamics state key"  "\"state\"" out;
  has     ~doc:"param glossary"      "\"symbol\"" out;
  has     ~doc:"@symbol β in glossary" "β" out;
  has     ~doc:"latex backslash JSON-escaped" "\\\\dot" out

(* ── flow graph (model.graph.json) ────────────────────────────────────── *)

(* Bare SIR: nodes are the base compartments, edges the transitions, and with no
   dimensions and no aggregate there are no plates, couplings, or advances. *)
let test_graph_basic () =
  let g = graph sir_symbol in
  Alcotest.(check (list string)) "base compartments are the nodes" [ "S"; "I"; "R" ] (node_ids g);
  Alcotest.(check int) "no plates (unstratified)" 0 (List.length g.mg_plates);
  Alcotest.(check int) "no couplings (no aggregate)" 0 (List.length g.mg_couplings);
  let inf = edge g "infection" in
  Alcotest.(check (option string)) "infection from S" (Some "S") inf.ge_from;
  Alcotest.(check (option string)) "infection to I" (Some "I") inf.ge_to;
  Alcotest.(check (option string)) "infection advances nothing" None inf.ge_advances;
  Alcotest.(check bool) "infection reads no pool" false inf.ge_reads_pool;
  let rec_ = edge g "recovery" in
  Alcotest.(check (option string)) "recovery from I" (Some "I") rec_.ge_from;
  Alcotest.(check (option string)) "recovery to R" (Some "R") rec_.ge_to

(* The full feature battery on the ctl_bb shape: the age plate with its levels,
   the aging edge advancing along `age` (over the `c` compartment iterator), the
   birth inflow (from=null) and death outflow (to=null), and the mean-field pool
   resolved through `let` bindings into per-edge couplings. *)
let test_graph_aging_pool () =
  let g = graph sir_aging_pool in
  Alcotest.(check (list string)) "nodes" [ "S"; "I"; "R" ] (node_ids g);
  Alcotest.(check (list string)) "age plate levels"
    [ "young"; "old" ] (plate g "age").gp_levels;
  (* aging: a self-transition over every compartment (`c`), stepping along age. *)
  let aging = edge g "aging" in
  Alcotest.(check (option string)) "aging advances along age" (Some "age") aging.ge_advances;
  Alcotest.(check (option string)) "aging from iterator c" (Some "c") aging.ge_from;
  Alcotest.(check (option string)) "aging to iterator c" (Some "c") aging.ge_to;
  (* birth is an exogenous inflow: no source compartment. *)
  let birth = edge g "birth" in
  Alcotest.(check (option string)) "birth has no source (inflow)" None birth.ge_from;
  Alcotest.(check (option string)) "birth into S" (Some "S") birth.ge_to;
  (* death is an outflow: no destination compartment. *)
  let death = edge g "death" in
  Alcotest.(check (option string)) "death from iterator c" (Some "c") death.ge_from;
  Alcotest.(check (option string)) "death has no sink (outflow)" None death.ge_to;
  (* the mean-field pool: infection reads foi ← inf_tot/ntot (both full-age sums),
     birth reads ntot; the ordinary within-cell flows read no pool. *)
  Alcotest.(check bool) "infection reads pool" true (edge g "infection").ge_reads_pool;
  Alcotest.(check bool) "birth reads pool" true birth.ge_reads_pool;
  Alcotest.(check bool) "recovery reads no pool" false (edge g "recovery").ge_reads_pool;
  Alcotest.(check bool) "aging reads no pool" false aging.ge_reads_pool;
  Alcotest.(check bool) "death reads no pool" false death.ge_reads_pool;
  Alcotest.(check (list (pair string (list string))))
    "infection couples over the two age pools"
    [ ("inf_tot", [ "age" ]); ("ntot", [ "age" ]) ] (couplings g "infection");
  Alcotest.(check (list (pair string (list string))))
    "birth couples over the ntot pool"
    [ ("ntot", [ "age" ]) ] (couplings g "birth")

(* An inline `sum(...)` in a rate (no named `let`) surfaces as an anonymous
   aggregate keyed "sum" over its dimension — the seir age-mixing case. *)
let test_graph_inline_sum () =
  let g = graph seir_inline_sum in
  let inf = edge g "infection" in
  Alcotest.(check bool) "infection reads pool (inline sum)" true inf.ge_reads_pool;
  Alcotest.(check (list (pair string (list string))))
    "inline sum couples over age, aggregate=sum"
    [ ("sum", [ "age" ]) ] (couplings g "infection");
  (* negative control: the within-cell flows carry no coupling. *)
  Alcotest.(check bool) "progression reads no pool" false (edge g "progression").ge_reads_pool

(* The JSON serialization is well-formed and carries every top-level key the
   viewer contract promises. *)
let test_graph_json_shape () =
  let out = Latex.to_graph_json (graph sir_aging_pool) in
  (match Yojson.Safe.from_string out with
   | _ -> ()
   | exception _ -> Alcotest.failf "to_graph_json did not produce well-formed JSON:\n%s" out);
  List.iter
    (fun k -> has ~doc:(Printf.sprintf "%s key" k) (Printf.sprintf "\"%s\"" k) out)
    [ "model"; "nodes"; "plates"; "edges"; "couplings"; "advances"; "reads_pool"; "aggregate" ];
  (* birth's null endpoint is a JSON null, not the string "null". *)
  has ~doc:"null endpoint" "\"from\": null" out

let () =
  Alcotest.run "latex"
    [ ( "scaffold",
        [ Alcotest.test_case "document scaffold + headings" `Quick
            test_document_scaffold ] );
      ( "symbols",
        [ Alcotest.test_case "@symbol override + \\frac flattening" `Quick
            test_symbol_override_and_frac;
          Alcotest.test_case "@symbol on a let overrides its symbol (gh#527)" `Quick
            test_let_symbol_override;
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
            test_expand_selected_dimension ] );
      ( "json",
        [ Alcotest.test_case "to_json: split transitions + glossary + escaping"
            `Quick test_json_shape ] );
      ( "graph",
        [ Alcotest.test_case "flow graph: nodes/edges of a bare SIR" `Quick
            test_graph_basic;
          Alcotest.test_case "flow graph: advances, birth/death, mean-field pool"
            `Quick test_graph_aging_pool;
          Alcotest.test_case "flow graph: inline sum aggregate" `Quick
            test_graph_inline_sum;
          Alcotest.test_case "to_graph_json: well-formed + contract keys" `Quick
            test_graph_json_shape ] ) ]
