(* Tests for Dim_solve, the integer solver behind dimension inference.

   Hand-written systems pin each outcome (determined, underdetermined with its
   relation, contradiction, integer infeasibility, a contradiction among rows
   without a unit coefficient, overflow). Two property tests check the solver
   against facts that hold whatever the system: an exhaustive search over a
   box of integer vectors (a solution found in the box means the solver must
   report no conflict; two solutions that differ in a variable mean it must
   call that variable free; every solution must agree with a value the solver
   calls determined), and invariance of the determined values under a
   reordering of a consistent system. *)

open Dim_solve

let o = { site = 0; stage = Structure; what = "test" }
let eq ?(origin = o) terms rhs = { terms; rhs; origin }

(* Bases: (P, T). *)
let vec p t = [| p; t |]

let solve2 = solve ~nbases:2

let show_vec v = "(" ^ String.concat "," (Array.to_list (Array.map string_of_int v)) ^ ")"

let check_value out v expected =
  let got = value out v in
  match got, expected with
  | Some g, Some e when g = e -> ()
  | None, None -> ()
  | _ ->
    Alcotest.failf "variable %d: expected %s, got %s" v
      (match expected with Some e -> show_vec e | None -> "free")
      (match got with Some g -> show_vec g | None -> "free")

let kinds out = List.map (fun c -> c.kind) (conflicts out)

let test_unique () =
  (* x = P; y = x · T⁻¹ *)
  let out = solve2 [ eq [ (0, 1) ] (vec 1 0); eq [ (1, 1); (0, -1) ] (vec 0 (-1)) ] in
  Alcotest.(check int) "no conflict" 0 (List.length (conflicts out));
  check_value out 0 (Some (vec 1 0));
  check_value out 1 (Some (vec 1 (-1)))

let test_underdetermined_relation () =
  (* only κ·w is pinned: κ + w = T⁻¹ *)
  let out = solve2 [ eq [ (0, 1); (1, 1) ] (vec 0 (-1)) ] in
  check_value out 0 None;
  check_value out 1 None;
  match relations out [ 0; 1 ] with
  | [ { coeffs; equals } ] ->
    let c v = Option.value ~default:0 (List.assoc_opt v coeffs) in
    (* the relation is ±(κ + w) = ±T⁻¹ *)
    let s = c 0 in
    Alcotest.(check bool) "coefficients ±(1, 1)" true (abs s = 1 && c 1 = s);
    Alcotest.(check bool) "equals ±T⁻¹" true (equals = vec 0 (-s))
  | rs -> Alcotest.failf "expected one relation, got %d" (List.length rs)

let test_contradiction () =
  let culprit = { o with site = 7; stage = Obligation } in
  let out = solve2 [ eq [ (0, 1) ] (vec 0 (-1));
                     eq ~origin:culprit [ (0, 1) ] (vec 1 (-1)) ] in
  (match conflicts out with
   | [ c ] ->
     Alcotest.(check bool) "contradiction" true (c.kind = Contradiction);
     Alcotest.(check int) "culprit is the second" 7 c.culprit.origin.site;
     Alcotest.(check bool) "context names the first" true
       (List.exists (fun (x : origin) -> x.site = 0) c.context)
   | cs -> Alcotest.failf "expected one conflict, got %d" (List.length cs));
  (* dropped, so it does not cascade: x keeps its first value *)
  check_value out 0 (Some (vec 0 (-1)))

let test_not_integral () =
  (* (κ·w)² in a rate: 2κ + 2w = T⁻¹ *)
  let out = solve2 [ eq [ (0, 2); (1, 2) ] (vec 0 (-1)) ] in
  Alcotest.(check bool) "not integral" true (kinds out = [ Not_integral ])

let test_not_integral_through_substitution () =
  (* y = κ + w (a product), x = 2y (a square), x = T⁻¹ (an obligation) *)
  let out = solve2 [ eq [ (2, 1); (0, -1); (1, -1) ] (vec 0 0);
                     eq [ (3, 1); (2, -2) ] (vec 0 0);
                     eq [ (3, 1) ] (vec 0 (-1)) ] in
  Alcotest.(check bool) "not integral" true (kinds out = [ Not_integral ])

let test_sqrt () =
  (* 2x = a, a = P² → x = P *)
  let out = solve2 [ eq [ (0, 2); (1, -1) ] (vec 0 0); eq [ (1, 1) ] (vec 2 0) ] in
  Alcotest.(check int) "no conflict" 0 (List.length (conflicts out));
  check_value out 0 (Some (vec 1 0));
  let out = solve2 [ eq [ (0, 2); (1, -1) ] (vec 0 0); eq [ (1, 1) ] (vec 1 0) ] in
  Alcotest.(check bool) "sqrt of P is not integral" true (kinds out = [ Not_integral ])

let test_contradiction_among_residual_rows () =
  (* no unit coefficients anywhere; rationally inconsistent *)
  let out = solve2 [ eq [ (0, 2); (1, 4) ] (vec 2 0); eq [ (0, 2); (1, 4) ] (vec 4 0) ] in
  Alcotest.(check bool) "contradiction" true (kinds out = [ Contradiction ]);
  (* consistent and redundant *)
  let out = solve2 [ eq [ (0, 2); (1, 4) ] (vec 2 0); eq [ (0, 4); (1, 8) ] (vec 4 0) ] in
  Alcotest.(check int) "redundant is fine" 0 (List.length (conflicts out))

let test_residual_determines () =
  (* 2x + 3y = 1, 3x + 5y = 2 (no unit coefficient) → x = -1, y = 1 *)
  let out = solve ~nbases:1 [ eq [ (0, 2); (1, 3) ] [| 1 |]; eq [ (0, 3); (1, 5) ] [| 2 |] ] in
  Alcotest.(check int) "no conflict" 0 (List.length (conflicts out));
  check_value out 0 (Some [| -1 |]);
  check_value out 1 (Some [| 1 |])

let test_overflow () =
  (* y + max_int·x = 0, then y − x = 0: substituting y gives −(1 + max_int)·x *)
  match solve ~nbases:1 [ eq [ (1, 1); (0, max_int) ] [| 0 |]; eq [ (1, 1); (0, -1) ] [| 0 |] ] with
  | _ -> Alcotest.fail "expected Overflow"
  | exception Overflow _ -> ()

(* ── Property tests ──────────────────────────────────────────────────── *)

let nvars = 4
let box = 4

(* Every integer vector in [-box, box]^nvars satisfying the system (one base). *)
let brute_solutions (eqs : eqn list) =
  let sols = ref [] in
  let x = Array.make nvars 0 in
  let rec go i =
    if i = nvars then begin
      if List.for_all (fun (e : eqn) ->
          List.fold_left (fun s (v, c) -> s + c * x.(v)) 0 e.terms = e.rhs.(0)) eqs
      then sols := Array.copy x :: !sols
    end else
      for v = -box to box do x.(i) <- v; go (i + 1) done
  in
  go 0; !sols

let gen_system =
  let open QCheck.Gen in
  let term = pair (int_bound (nvars - 1)) (int_range (-3) 3) in
  let equation = pair (list_size (int_range 1 3) term) (int_range (-3) 3) in
  list_size (int_range 1 5) equation

let to_eqns sys = List.mapi (fun i (terms, r) -> eq ~origin:{ o with site = i } terms [| r |]) sys

let print_system sys =
  String.concat "; " (List.map (fun (terms, r) ->
    String.concat " + " (List.map (fun (v, c) -> Printf.sprintf "%d·x%d" c v) terms)
    ^ " = " ^ string_of_int r) sys)

let arb = QCheck.make ~print:print_system gen_system

let prop_oracle =
  QCheck.Test.make ~name:"agrees with exhaustive search in a box" ~count:1500 arb (fun sys ->
    let eqs = to_eqns sys in
    let out = solve ~nbases:1 eqs in
    let sols = brute_solutions eqs in
    let feasible_claimed = conflicts out = [] in
    (* a solution in the box ⇒ the system is feasible *)
    (sols = [] || feasible_claimed)
    && (not feasible_claimed
        || List.for_all (fun v ->
             match value out v with
             | Some d -> List.for_all (fun s -> s.(v) = d.(0)) sols
             | None -> true) (List.init nvars Fun.id))
    && (not feasible_claimed
        || List.for_all (fun v ->
             (* two box solutions differing in v ⇒ v is free *)
             match sols with
             | [] -> true
             | s0 :: _ ->
               not (List.exists (fun s -> s.(v) <> s0.(v)) sols) || value out v = None)
           (List.init nvars Fun.id)))

let shuffle st l =
  let a = Array.of_list l in
  for i = Array.length a - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = a.(i) in a.(i) <- a.(j); a.(j) <- t
  done;
  Array.to_list a

(* A system with a planted solution: right-hand sides computed from it. *)
let gen_planted =
  let open QCheck.Gen in
  let term = pair (int_bound (nvars - 1)) (int_range (-3) 3) in
  pair (array_size (return nvars) (int_range (-3) 3))
    (list_size (int_range 1 6) (list_size (int_range 1 3) term))

let prop_permutation =
  QCheck.Test.make ~name:"determined values are independent of equation order" ~count:500
    (QCheck.make gen_planted) (fun (x, rows) ->
      let sys = List.map (fun terms ->
        (terms, List.fold_left (fun s (v, c) -> s + c * x.(v)) 0 terms)) rows in
      let a = solve ~nbases:1 (to_eqns sys) in
      let b = solve ~nbases:1 (to_eqns (shuffle (Random.State.make [| 42 |]) sys)) in
      conflicts a = [] && conflicts b = []
      && List.for_all (fun v -> value a v = value b v) (List.init nvars Fun.id)
      && List.for_all (fun v -> match value a v with
          | Some d -> d.(0) = x.(v) | None -> true) (List.init nvars Fun.id))

let () =
  Alcotest.run "dim_solve" [
    "outcomes", [
      Alcotest.test_case "unique solution"                 `Quick test_unique;
      Alcotest.test_case "underdetermined, with relation"  `Quick test_underdetermined_relation;
      Alcotest.test_case "contradiction reported, dropped" `Quick test_contradiction;
      Alcotest.test_case "integer infeasibility"           `Quick test_not_integral;
      Alcotest.test_case "infeasibility via substitution"  `Quick test_not_integral_through_substitution;
      Alcotest.test_case "square root"                     `Quick test_sqrt;
      Alcotest.test_case "contradiction, non-unit rows"    `Quick test_contradiction_among_residual_rows;
      Alcotest.test_case "non-unit rows determine"         `Quick test_residual_determines;
      Alcotest.test_case "overflow is an error"            `Quick test_overflow;
    ];
    "properties", List.map QCheck_alcotest.to_alcotest [ prop_oracle; prop_permutation ];
  ]
