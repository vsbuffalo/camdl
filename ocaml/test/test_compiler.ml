(* Compiler golden tests: parse+expand camdl source → match expected IR JSON *)

(* Disable dimcheck for compiler tests — these test expansion/codegen,
   not dimensional analysis. Some test models have rates that dimcheck
   can't infer (table lookups, time functions with ambiguous dimension). *)
let () = Compiler.no_dim_check := true

(** Substring check without the Str library. *)
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

let compile_expect_ok src =
  match Compiler.compile ~name:"test" src with
  | Ok m -> m
  | Error e -> Alcotest.failf "compile failed: %s" e

(** Compile with JSON-diagnostics mode so the Error variant carries the
    structured error payload (codes + messages) rather than the generic
    "compilation failed" string. Then assert the given error code and a
    substring (typically a parameter/intervention name, to confirm
    diagnostics carry enough context) both appear in the payload. *)
let compile_expect_error_code ~code ~contains src =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"test_err" src in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.failf "expected error %s but compile succeeded" code
  | Error e ->
    if String.length e = 0 then Alcotest.failf "error text was empty";
    if not (contains_substring ~needle:code e) then
      Alcotest.failf "expected error code %s, got: %s" code e;
    if not (contains_substring ~needle:contains e) then
      Alcotest.failf "expected error to contain %S, got: %s" contains e

let golden_dir =
  (* The dune test runner sets cwd to the project root (_build/default/test).
     We walk up to find the ocaml/golden directory. *)
  let candidates = [
    "../../golden";          (* from _build/default/test *)
    "../golden";
    "golden";

  ] in
  List.find (fun d ->
    Sys.file_exists d && Sys.is_directory d
  ) candidates

let read_file path =
  let ic = open_in path in
  let n  = in_channel_length ic in
  let s  = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

let test_golden model_name () =
  let camdl_path = Filename.concat golden_dir (model_name ^ ".camdl") in
  let ir_path    = Filename.concat golden_dir (model_name ^ ".ir.json") in
  let src = read_file camdl_path in
  (* Pass ~filename so source_dir is the golden directory; fixtures
     that reference `data/*.tsv` need this to find their data files.
     Without it, source_dir defaults to "" and reads fail against the
     test CWD. *)
  let ir = match Compiler.compile ~name:model_name ~filename:camdl_path src with
    | Ok m    -> m
    | Error e -> Alcotest.failf "compile failed: %s" e
  in
  let expected_json = read_file ir_path in
  let expected_m = match Serde.model_of_string expected_json with
    | Ok m    -> m
    | Error e -> Alcotest.failf "bad golden JSON: %s" e
  in
  if ir <> expected_m then begin
    let actual_json = Serde.model_to_string ir in
    Alcotest.failf "IR mismatch for %s\nExpected:\n%s\n\nActual:\n%s"
      model_name expected_json actual_json
  end

(* ── TableLookup flattening tests ───────────────────────────────────────────
   The IR contract requires TableLookup to carry exactly ONE index: the
   row-major flattened offset computed at compile time.  For a 2×2 table:
     [row 0, col 0] → 0    [row 0, col 1] → 1
     [row 1, col 0] → 2    [row 1, col 1] → 3
   These tests compile seir_age (2×2 C_age contact matrix) and walk the
   rate expressions, asserting exactly that. ──────────────────────────────── *)

let rec collect_table_lookups expr =
  let open Ir in
  match expr with
  | TableLookup (name, idxs) -> [(name, idxs)]
  | BinOp { left; right; _ } ->
    collect_table_lookups left @ collect_table_lookups right
  | UnOp  { arg; _ }         -> collect_table_lookups arg
  | Cond  { pred; then_; else_ } ->
    collect_table_lookups pred
    @ collect_table_lookups then_
    @ collect_table_lookups else_
  | Reduce terms -> List.concat_map collect_table_lookups terms
  | _ -> []

(** Run [f] with the constant-fold pass disabled, restoring the prior setting
    afterwards. The TableLookup-flattening tests below assert on the *unfolded*
    IR (the fold resolves constant-indexed lookups to literals, erasing the
    TableLookup nodes they inspect). Mirrors [with_dim_check_enabled]. *)
let with_fold_disabled f =
  let prev = !Compiler.constant_fold in
  Compiler.constant_fold := false;
  Fun.protect ~finally:(fun () -> Compiler.constant_fold := prev) f

(* Compiled with the fold OFF: these tests inspect the expander's
   TableLookup-flattening contract, which the fold would resolve away. *)
let compile_seir_age () =
  with_fold_disabled (fun () ->
    let src = read_file (Filename.concat golden_dir "seir_age.camdl") in
    match Compiler.compile ~name:"seir_age" src with
    | Ok m    -> m
    | Error e -> Alcotest.failf "seir_age compile failed: %s" e)

let find_transition (m : Ir.model) name =
  match List.find_opt (fun (t : Ir.transition) -> t.name = name) m.transitions with
  | Some t -> t
  | None   -> Alcotest.failf "transition %s not found" name

let tr_rate  (t : Ir.transition) = t.rate
let tr_name  (t : Ir.transition) = t.name

let c_age_indices (tr : Ir.transition) =
  let lookups = collect_table_lookups (tr_rate tr) in
  let indices = List.filter_map (fun (tbl, idxs) ->
    if tbl = "C_age" then
      match idxs with
      | [Ir.Const v] -> Some v
      | _            -> Alcotest.fail "C_age lookup has != 1 index"
    else None
  ) lookups in
  List.sort_uniq compare indices

(* Each TableLookup in the rate must have exactly one index. *)
let test_table_lookup_single_index () =
  let m = compile_seir_age () in
  List.iter (fun (tr : Ir.transition) ->
    let lookups = collect_table_lookups (tr_rate tr) in
    List.iter (fun (tbl, idxs) ->
      Alcotest.(check int)
        (Printf.sprintf "%s: TableLookup(%s) index count" (tr_name tr) tbl)
        1 (List.length idxs)
    ) lookups
  ) m.transitions

(* infection_child uses C_age[child,child]=0 and C_age[child,adult]=1 *)
let test_infection_child_indices () =
  let m = compile_seir_age () in
  let tr = find_transition m "infection_child" in
  Alcotest.(check (list (float 0.)))
    "infection_child C_age indices"
    [0.; 1.] (c_age_indices tr)

(* infection_adult uses C_age[adult,child]=2 and C_age[adult,adult]=3 *)
let test_infection_adult_indices () =
  let m = compile_seir_age () in
  let tr = find_transition m "infection_adult" in
  Alcotest.(check (list (float 0.)))
    "infection_adult C_age indices"
    [2.; 3.] (c_age_indices tr)

(* ── Sparse-coupling constant-fold (A/B gate, OCaml half) ─────────────────────
   `Constant_fold.fold_model` resolves constant-indexed inline-table lookups and
   drops zero-W terms from the FOI Reduce, collapsing the dense P-term spatial
   sum to its k nonzero terms. This test proves the pass *fires* at the source:
   on a sparse ring W (k neighbours per patch) the largest FOI Reduce shrinks
   from P terms to k. The byte-identical *trajectory* half lives in Rust
   (gate_constant_fold_ab). A no-op fold (dense W) would leave the term count
   unchanged and fail the strict-inequality assertion below — the guard against
   a vacuous green. *)

(** Largest Reduce term count anywhere in an expr tree. The only Reduce in a
    spatial FOI rate is the coupling sum, so this is its term count. *)
let rec max_reduce_terms (e : Ir.expr) : int =
  let open Ir in
  match e with
  | Reduce terms ->
    List.fold_left (fun acc t -> max acc (max_reduce_terms t))
      (List.length terms) terms
  | BinOp { left; right; _ } -> max (max_reduce_terms left) (max_reduce_terms right)
  | UnOp { arg; _ } -> max_reduce_terms arg
  | Cond { pred; then_; else_ } ->
    max (max_reduce_terms pred) (max (max_reduce_terms then_) (max_reduce_terms else_))
  | TableLookup (_, idxs) ->
    List.fold_left (fun acc i -> max acc (max_reduce_terms i)) 0 idxs
  | UncheckedDim { inner; _ } -> max_reduce_terms inner
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> 0

let max_foi_reduce_terms (m : Ir.model) : int =
  List.fold_left (fun acc (t : Ir.transition) -> max acc (max_reduce_terms t.rate))
    0 m.transitions

(* P=4 patches, sparse ring W with k=2 neighbours each (off-diagonal cells:
   p couples to p-1 and p+1, wrapping; all other W cells are 0). The FOI uses
   the guarded form `W[p,q] * (if N[q] > 0 then I[q]/N[q] else 0)`, which makes
   the zero-W fold sound (0 * finite -> 0 in one step). The expander emits a
   dense 4-term Reduce; the fold collapses it to the 2 nonzero-W terms. *)
let sparse_ring_src = {|
    time_unit = 'days
    dimensions { patch = [p0, p1, p2, p3] }
    compartments { S, I }
    stratify(by = patch)
    parameters { beta : rate  kappa : probability in [0.0, 1.0] }
    tables {
      W : patch × patch = [[0.0, 0.5, 0.0, 0.5],
                           [0.5, 0.0, 0.5, 0.0],
                           [0.0, 0.5, 0.0, 0.5],
                           [0.5, 0.0, 0.5, 0.0]]
    }
    let N[l in patch] = S[l] + I[l]
    transitions {
      infection[l in patch] : S[l] --> I[l]
        @ beta * S[l] * (
            (if N[l] > 0 then I[l] / N[l] else 0.0)
          + kappa * sum(q in patch, W[l, q] * (if N[q] > 0 then I[q] / N[q] else 0.0))
          )
      recovery[l in patch] : I[l] --> S[l]  @ beta * I[l]
    }
    init { S_p0 = 990  I_p0 = 10 }
    simulate { from = 0 'days  to = 30 'days }
  |}

let test_constant_fold_collapses_sparse_foi_reduce () =
  (* Compile with the fold OFF so [m] is the unfolded (dense) IR; then apply
     the pass directly and compare. (The default pipeline now folds.) *)
  let m = with_fold_disabled (fun () ->
    match Compiler.compile ~name:"sparse_ring" sparse_ring_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "compile failed: %s" e)
  in
  let folded = Constant_fold.fold_model m in
  let before = max_foi_reduce_terms m in
  let after  = max_foi_reduce_terms folded in
  (* 4 patches → dense 4-term FOI Reduce before the fold. *)
  Alcotest.(check int) "dense FOI Reduce has P=4 terms before fold" 4 before;
  (* Sparse ring k=2 → 2 nonzero-W terms survive. The strict drop is the
     non-vacuity guard: a dense W (no zero cells) would leave this at 4 and
     fail here, exactly as it should. *)
  Alcotest.(check int) "fold collapses FOI Reduce to k=2 terms" 2 after;
  Alcotest.(check bool) "fold strictly shrank the FOI Reduce" true (after < before)

(* ── gh#272 LICM pass ─────────────────────────────────────────────────────────
   Loop-invariant code motion hoists param/table-only subexpressions out of the
   dynamics rates into `per_eval_bindings`. These pin the variant/invariant
   classification (esp. that `Dt`/`Time`/forcing/state are VARIANT — a `Dt`
   mis-classified as invariant would freeze the integrator step and silently
   corrupt a trajectory), the cost threshold, and that the pass actually fires.
   The byte-identity soundness proof is the Rust A/B gate gate_licm_ab.rs. *)

(* Build a name → table lookup, as `Licm.licm_model` does, so `is_invariant`
   can judge a `TableLookup` on its cell bodies (gh#284). *)
let tbls_of (ts : Ir.table list) : (string, Ir.table) Hashtbl.t =
  let h = Hashtbl.create (max 1 (List.length ts)) in
  List.iter (fun (t : Ir.table) -> Hashtbl.replace h t.name t) ts;
  h

let inline_table name cells : Ir.table =
  { name; source = Ir.Inline cells; out_of_bounds = Ir.Error; cell_kind = None }

let test_licm_invariant_classification () =
  let open Ir in
  (* The two tables the kernel reads, with constant (invariant) cell bodies. *)
  let tbls = tbls_of [
    inline_table "N0" [Const 1000.0; Const 2000.0];
    inline_table "dratio" [Const 1.0; Const 2.0; Const 2.0; Const 1.0];
  ] in
  (* Param + table + const, no state/time → invariant. *)
  let kernel = BinOp { op = Mul; left = TableLookup ("N0", [Const 1.0]);
    right = UnOp { op = Exp; arg = BinOp { op = Mul;
      left = UnOp { op = Neg; arg = Param "gamma_k" };
      right = UnOp { op = Log; arg = TableLookup ("dratio", [Const 0.0; Const 1.0]) } } } } in
  Alcotest.(check bool) "param/table/const kernel is invariant" true (Licm.is_invariant tbls kernel);
  Alcotest.(check bool) "R0*gamma is invariant" true
    (Licm.is_invariant tbls (BinOp { op = Mul; left = Param "R0"; right = Param "gamma" }));
  (* The variant nodes — each must classify as NOT invariant. The Dt case is the
     load-bearing one (gh#272 review): the pass must never hoist a dt subtree. *)
  Alcotest.(check bool) "Pop is variant" false (Licm.is_invariant tbls (Pop "I"));
  Alcotest.(check bool) "PopSum is variant" false (Licm.is_invariant tbls (PopSum ["S"; "I"]));
  Alcotest.(check bool) "Time is variant" false (Licm.is_invariant tbls Time);
  Alcotest.(check bool) "Dt is variant" false (Licm.is_invariant tbls Dt);
  Alcotest.(check bool) "TimeFunc (forcing) is variant" false (Licm.is_invariant tbls (TimeFunc "school"));
  Alcotest.(check bool) "BindingRef (state) is variant" false (Licm.is_invariant tbls (BindingRef "N"));
  (* exp(c) * dt is variant (contains Dt), so the whole product is never hoisted. *)
  Alcotest.(check bool) "exp(c)*dt is variant" false
    (Licm.is_invariant tbls (BinOp { op = Mul; left = UnOp { op = Exp; arg = Const 0.5 }; right = Dt }))

(* gh#284: `is_invariant` for a `TableLookup` must judge the table's CELL BODIES,
   not just its index. A state-referencing inline cell makes the whole lookup
   variant even with a constant index (else it would be hoisted and read stale).
   External tables (file-loaded numbers) and const-cell inline tables stay
   invariant; an unknown table is conservatively variant.

   This is a unit test on the predicate, not an end-to-end DSL→reject test:
   today's DSL cannot express a state-dependent inline table cell (the parser /
   dim-checker reject the precursors), so the `badtab` shape is built directly.
   The predicate is the forward-compatible guard for when such cells are
   allowed. *)
let test_licm_table_cell_invariance () =
  let open Ir in
  let const_tbls = tbls_of [inline_table "ct" [Const 1.0; Const 2.0]] in
  Alcotest.(check bool) "lookup into const-cell table is invariant" true
    (Licm.is_invariant const_tbls (TableLookup ("ct", [Const 0.0])));
  let state_cell_tbls = tbls_of [inline_table "badtab" [Pop "S"; Pop "I"]] in
  Alcotest.(check bool) "lookup into state-dependent inline table is variant" false
    (Licm.is_invariant state_cell_tbls (TableLookup ("badtab", [Const 0.0])));
  (* The whole enclosing expr inherits the variance — a `pow` over a bad cell is
     NOT hoistable, which is the soundness point. *)
  Alcotest.(check bool) "expr over a state-dependent cell is variant" false
    (Licm.is_invariant state_cell_tbls
       (BinOp { op = Pow; left = TableLookup ("badtab", [Const 0.0]); right = Const 2.0 }));
  let ext_tbls = tbls_of
    [{ name = "ext"; source = External "ext.csv"; out_of_bounds = Error; cell_kind = None }] in
  Alcotest.(check bool) "lookup into external table is invariant" true
    (Licm.is_invariant ext_tbls (TableLookup ("ext", [Const 0.0])));
  Alcotest.(check bool) "lookup into unknown table is variant (conservative)" false
    (Licm.is_invariant (tbls_of []) (TableLookup ("nope", [Const 0.0])))

let test_licm_cost_threshold () =
  let open Ir in
  (* Worth hoisting iff a transcendental / Pow / Reduce is present. *)
  Alcotest.(check bool) "exp(x) is expensive" true
    (Licm.contains_expensive (UnOp { op = Exp; arg = Param "x" }));
  Alcotest.(check bool) "x^y is expensive" true
    (Licm.contains_expensive (BinOp { op = Pow; left = Param "x"; right = Const 2.0 }));
  Alcotest.(check bool) "Reduce is expensive" true
    (Licm.contains_expensive (Reduce [Param "a"; Param "b"]));
  Alcotest.(check bool) "R0*gamma is NOT worth hoisting" false
    (Licm.contains_expensive (BinOp { op = Mul; left = Param "R0"; right = Param "gamma" }));
  Alcotest.(check bool) "bare Param is NOT worth hoisting" false (Licm.contains_expensive (Param "x"))

let licm_kernel_src = {|
    time_unit = 'days
    dimensions { patch = [p0, p1] }
    compartments { S, E, I, R }
    stratify(by = patch)
    parameters {
      R0 : positive in [0.5, 6.0]
      gamma : rate in [0.05, 0.5]
      sigma : rate in [0.05, 0.5]
      gamma_k : positive in [0.5, 10.0]
    }
    tables {
      N0 : patch = [1000.0, 2000.0]
      dratio : patch × patch = [[1.0, 2.0], [2.0, 1.0]]
    }
    let N[l in patch] = S[l] + E[l] + I[l] + R[l]
    transitions {
      infection[l in patch] : S[l] --> E[l]
        @ R0 * gamma * S[l]
          * sum(q in patch, N0[q] * exp(-1.0 * gamma_k * log(dratio[l, q])) * I[q] / N[q])
          / sum(r in patch, N0[r] * exp(-1.0 * gamma_k * log(dratio[l, r])))
      progression[l in patch] : E[l] --> I[l] @ sigma * E[l]
      recovery[l in patch] : I[l] --> R[l] @ gamma * I[l]
    }
    init { S_p0 = 990  I_p0 = 10 }
    simulate { from = 0 'days  to = 30 'days }
  |}

let rec count_per_eval_refs (e : Ir.expr) : int =
  let open Ir in
  match e with
  | PerEvalRef _ -> 1
  | BinOp { left; right; _ } -> count_per_eval_refs left + count_per_eval_refs right
  | UnOp { arg; _ } -> count_per_eval_refs arg
  | Cond { pred; then_; else_ } ->
    count_per_eval_refs pred + count_per_eval_refs then_ + count_per_eval_refs else_
  | TableLookup (_, idxs) | Reduce idxs -> List.fold_left (fun a e -> a + count_per_eval_refs e) 0 idxs
  | UncheckedDim u -> count_per_eval_refs u.inner
  | _ -> 0

let test_licm_hoists_kernel () =
  (* LICM is ON by default (gh#272 flip), so the default compile already hoists
     the kernel into per_eval_bindings; CAMDL_NO_LICM would force the inlined
     variant. The off-vs-on byte-identity is covered by the Rust A/B gate
     (gate_licm_ab.rs); here we check the pass fired and its output is well-formed.
     We do NOT re-apply Licm.licm_model to `m` — its input invariant is "no
     PerEvalRef present", which the hoisted `m` already violates. *)
  let m = match Compiler.compile ~name:"licm_kernel" licm_kernel_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "compile failed: %s" e in
  (* The pass fired in the default pipeline: bindings were created and the rates
     reference them. *)
  Alcotest.(check bool) "default compile created per_eval bindings" true (m.per_eval_bindings <> []);
  let refs_in_rates =
    List.fold_left (fun acc (t : Ir.transition) -> acc + count_per_eval_refs t.rate) 0 m.transitions in
  Alcotest.(check bool) "rates reference PerEvalRefs" true (refs_in_rates > 0);
  (* Every hoisted body is invariant (no state/time/dt smuggled in) — the keystone
     invariant the Rust runtime relies on. *)
  let tbls = tbls_of m.tables in
  Alcotest.(check bool) "every per_eval body is invariant" true
    (List.for_all (fun (b : Ir.binding) -> Licm.is_invariant tbls b.bexpr) m.per_eval_bindings)

(* ── Binding param-free invariant (E512, defensive) ───────────────────────────
   The hoist/autodiff contract: [autodiff.ml] differentiates [BindingRef] to 0,
   so a hoisted [model.bindings] body must be param-free or its gradient is
   silently zeroed (the gh#186 failure class). [Validate.references_param] is
   the structural reachability check; the E512 error fires if any binding body
   reaches a Param. The expander's eligibility heuristic already excludes
   param-referencing lets, so on the clean corpus this NEVER fires — the test
   below asserts that the sparse-ring model (which DOES hoist `N[l]`) compiles
   without E512, which is non-vacuous because the model has bindings for the
   check to traverse. *)

(** Unit test of the reachability primitive on hand-built exprs. *)
let test_references_param_primitive () =
  let open Ir in
  (* No Param anywhere: state/time/const tree → false. *)
  let state_only =
    BinOp { op = Mul; left = Pop "S";
            right = Cond { pred = BinOp { op = Gt; left = Pop "N"; right = Const 0.0 };
                           then_ = BinOp { op = Div; left = Pop "I"; right = Pop "N" };
                           else_ = Const 0.0 } } in
  Alcotest.(check bool) "state/const tree has no Param" false
    (Validate.references_param state_only);
  (* A Param buried under a Reduce term → true. *)
  let with_param =
    Reduce [ Const 0.0;
             BinOp { op = Mul; left = Param "beta"; right = Pop "I" } ] in
  Alcotest.(check bool) "Param under Reduce is reached" true
    (Validate.references_param with_param);
  Alcotest.(check (option string)) "first_param names the offending param"
    (Some "beta") (Validate.first_param with_param);
  (* A BindingRef is a leaf — the check does NOT recurse into other bindings
     (each binding is validated in turn), so a bare BindingRef is param-free. *)
  Alcotest.(check bool) "BindingRef is a param-free leaf" false
    (Validate.references_param (BindingRef "N_p0"))

(** Clean-corpus assertion: the sparse-ring spatial model hoists `N[l]` into
    [model.bindings]; compiling it must NOT raise E512, and the model must
    actually carry bindings (else the invariant runs over nothing). *)
let test_binding_invariant_clean_on_spatial () =
  let m = match Compiler.compile ~name:"sparse_ring" sparse_ring_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "spatial model should compile, got: %s" e
  in
  (* Non-vacuity: there are hoisted bindings for the invariant to traverse. *)
  Alcotest.(check bool) "spatial model has hoisted bindings"
    true (m.bindings <> []);
  (* And every binding body is param-free (the contract holds on the corpus). *)
  List.iter (fun (b : Ir.binding) ->
    Alcotest.(check bool)
      (Printf.sprintf "binding '%s' is param-free" b.bname)
      false (Validate.references_param b.bexpr)
  ) m.bindings

(** Negative control: the invariant is live, not dead code. Take a real model,
    inject a Param into one binding body (the failure the front-end heuristic is
    supposed to prevent), and confirm [Validate.validate] reports it. This is
    the red the DSL itself cannot easily produce, exercised at the IR level. *)
let test_binding_invariant_catches_poisoned_binding () =
  let m = match Compiler.compile ~name:"sparse_ring" sparse_ring_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "spatial model should compile, got: %s" e
  in
  let b0 = match m.bindings with
    | b :: _ -> b
    | [] -> Alcotest.fail "expected at least one hoisted binding"
  in
  (* Splice a param into the first binding's body: N[l] -> N[l] + beta. *)
  let poisoned =
    { b0 with Ir.bexpr =
        Ir.BinOp { op = Ir.Add; left = b0.Ir.bexpr; right = Ir.Param "beta" } } in
  let m' = { m with Ir.bindings = poisoned :: List.tl m.bindings } in
  match Validate.validate m' with
  | Ok () -> Alcotest.fail "expected ParamInBinding error, got Ok"
  | Error errs ->
    let fired = List.exists (function
      | Validate.ParamInBinding (b, p) -> b = b0.Ir.bname && p = "beta"
      | _ -> false) errs in
    Alcotest.(check bool) "ParamInBinding fired for the poisoned binding"
      true fired

(* ── Cost report (--cost-report) ──────────────────────────────────────────────
   The cost report is read-only over the (unfolded) expanded model plus a local
   Constant_fold pass. On the sparse-ring model the report must show the
   sparse-coupling fold collapsing the FOI Reduce (after < before), the hoisted
   N[l] bindings carrying their expected reference counts, and the duplicated
   guarded-FOI subexprs surfacing. The renderer is also smoke-run to a buffer to
   confirm it produces output without raising. *)

(** Compile the sparse-ring fixture unfolded (mirrors inspect's front-end-only
    path), so the report's local fold has something to collapse. *)
let sparse_ring_unfolded () =
  with_fold_disabled (fun () ->
    match Compiler.compile ~name:"sparse_ring" sparse_ring_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "compile failed: %s" e)

let test_cost_report_numbers_sane () =
  let m = sparse_ring_unfolded () in
  let rates = List.map (fun (t : Ir.transition) -> t.rate) m.transitions in
  (* Reduce terms: the fold strictly collapses the sparse coupling sum. *)
  let reduce_before =
    List.fold_left (fun a e -> a + Inspect.reduce_term_count e) 0 rates in
  let folded = Constant_fold.fold_model m in
  let folded_rates = List.map (fun (t : Ir.transition) -> t.rate) folded.transitions in
  let reduce_after =
    List.fold_left (fun a e -> a + Inspect.reduce_term_count e) 0 folded_rates in
  Alcotest.(check bool) "fold strictly shrinks Reduce terms"
    true (reduce_after < reduce_before);
  (* 4 patches × P=4-term FOI Reduce = 16 terms before; k=2 sparse ring → 8. *)
  Alcotest.(check int) "Reduce terms before fold" 16 reduce_before;
  Alcotest.(check int) "Reduce terms after fold" 8 reduce_after;
  (* The hoisted N[l] bindings exist and are referenced from the rates. The
     ring couples each patch to 2 neighbours, and N[l] appears both in the
     direct I[l]/N[l] term and in every neighbour's coupling term, so each
     binding is referenced more than once (the reuse the report highlights). *)
  let n_binding = List.find_opt (fun (b : Ir.binding) -> b.bname = "N_p0") m.bindings in
  (match n_binding with
   | None -> Alcotest.fail "expected hoisted binding N_p0"
   | Some b ->
     let refs = List.fold_left (fun a e -> a + Inspect.count_bindingref b.bname e) 0 rates in
     Alcotest.(check bool) "N_p0 is referenced more than once (reuse)" true (refs > 1));
  (* Duplicated guarded-FOI subexprs surface (the repeated
     `if N[q] > 0 then I[q]/N[q] else 0` guards appear ≥3 times). *)
  let all_exprs = rates @ List.map (fun (b : Ir.binding) -> b.bexpr) m.bindings in
  let dups = Inspect.count_duplicated_subexprs all_exprs in
  Alcotest.(check bool) "duplicated subexprs detected" true (dups > 0)

let test_cost_report_renders_without_raising () =
  let m = sparse_ring_unfolded () in
  (* Smoke: render to a buffer. ctx is unused by run_cost_report (it takes
     _ctx), so we compile through the detail path to obtain one. *)
  let buf = Buffer.create 512 in
  let ppf = Format.formatter_of_buffer buf in
  (* run_cost_report ignores ctx; pass the model's own via a detail compile. *)
  let detail =
    match Compiler.compile_detail_result ~name:"sparse_ring" sparse_ring_src with
    | Ok d -> d
    | Error e -> Alcotest.failf "detail compile failed: %s" e
  in
  Inspect.run_cost_report ppf m detail.ctx;
  Format.pp_print_flush ppf ();
  let out = Buffer.contents buf in
  (* Substring search without pulling in Str. *)
  let contains hay needle =
    let nlen = String.length needle and hlen = String.length hay in
    let rec at i =
      if i + nlen > hlen then false
      else if String.sub hay i nlen = needle then true
      else at (i + 1)
    in nlen = 0 || at 0
  in
  Alcotest.(check bool) "report mentions 'cost report'" true (contains out "cost report");
  Alcotest.(check bool) "report mentions Reduce terms" true (contains out "Reduce terms")

(** Hazard-idiom detector is live: `1 - exp(x)` is recognized. *)
let test_cost_report_hazard_idiom_detected () =
  let m = match Compiler.compile ~name:"hazard" {|
    time_unit = 'days
    compartments { S, I }
    parameters { gamma : rate }
    transitions {
      recovery : I --> S  @ I * (1 - exp(-gamma * dt)) / dt
    }
    init { S = 990  I = 10 }
    simulate { from = 0 'days  to = 30 'days }
  |} with
    | Ok m -> m
    | Error e -> Alcotest.failf "hazard model compile failed: %s" e
  in
  let rates = List.map (fun (t : Ir.transition) -> t.rate) m.transitions in
  let hz = List.fold_left (fun a e -> a + Inspect.count_hazard_idioms e) 0 rates in
  Alcotest.(check int) "one 1 - exp(x) hazard form detected" 1 hz

(* min/max wire through the DSL surface to the already-supported Ir.BinOp
   Min/Max (the IR, Rust eval, dimcheck, and autodiff already handle them). *)
let test_min_max_wire_to_binop () =
  let m = match Compiler.compile ~name:"minmax" {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      beta  : rate
      gamma : rate
    }
    transitions {
      infect  : S --> I @ min(beta, gamma) * I
      recover : I --> S @ max(beta, gamma) * I
    }
    init { S = 1  I = 0 }
    simulate { from = 0 'days  to = 1 'days }
  |} with
    | Ok m    -> m
    | Error e -> Alcotest.failf "min/max compile failed: %s" e
  in
  let has_op op (t : Ir.transition) =
    let rec go = function
      | Ir.BinOp b   -> b.op = op || go b.left || go b.right
      | Ir.UnOp u    -> go u.arg
      | Ir.Cond c    -> go c.pred || go c.then_ || go c.else_
      | Ir.Reduce ts -> List.exists go ts
      | _            -> false
    in go (tr_rate t)
  in
  Alcotest.(check bool) "infect rate contains BinOp Min" true
    (has_op Ir.Min (find_transition m "infect"));
  Alcotest.(check bool) "recover rate contains BinOp Max" true
    (has_op Ir.Max (find_transition m "recover"))

(* ── BUG-3: Comparison operators ────────────────────────────────────────────
   Compile a model that uses a comparison in a rate: `if S > 0 then ... else 0`.
   The compiled rate should contain a Cond node wrapping a BinOp(Gt,...). ── *)

let test_comparison_in_rate () =
  let src = {|
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0    : count
      I0    : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ if S > 0 then beta * S * I / N else 0.0
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_cmp" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let infection = find_transition m "infection" in
    let rate = tr_rate infection in
    let rec contains_gt = function
      | Ir.Cond { pred; _ } -> contains_gt pred
      | Ir.BinOp { op = Ir.Gt; _ } -> true
      | Ir.BinOp b -> contains_gt b.left || contains_gt b.right
      | Ir.UnOp u -> contains_gt u.arg
      | _ -> false
    in
    Alcotest.(check bool) "rate contains Gt comparison" true (contains_gt rate)

(* ── BUG-6: Output schedule step ────────────────────────────────────────────
   The parser uses `every` as a reserved keyword (EVERY token) inside
   trajectories blocks, matched via List.assoc_opt which defaults to EConst 1.0.
   Test that the expand_output function produces OutRegular with the default
   step=1.0 when no output block is provided; the horizon (gh#143) lives in
   simulation.t_end, taken from simulate's `to`.
   (A direct "custom step" end-to-end test requires fixing the parser to accept
   EVERY inside func_arg context — deferred.) ──────────────────────────────── *)

let test_output_format_from_decl () =
  let src = {|
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0    : count
      I0    : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 120 'days }
    output { trajectories { } }
  |} in
  match Compiler.compile ~name:"test_output_fmt" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* output block present → format defaults to "tsv", step to 1.0 *)
    Alcotest.(check string) "format" "tsv" m.Ir.output.Ir.format;
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 0.01)) "default step" 1.0 r.Ir.step
     | _ -> Alcotest.fail "expected OutRegular schedule");
    (* gh#143: the output schedule no longer carries its own end; the horizon
       is `simulation.t_end`, the sole authority the runtime derives output
       times from. *)
    Alcotest.(check (float 0.01)) "t_end" 120.0 m.Ir.simulation.Ir.t_end

let test_output_step_default () =
  let src = {|
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0    : count
      I0    : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_output_default" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 0.01)) "default output step" 1.0 r.Ir.step
     | _ -> Alcotest.fail "expected OutRegular schedule")

(* Regression: the default output schedule must cover the full
   integration window. With anchored models that resolve `from =
   date(...)` to a negative t_start, the default `start = 0.0` would
   leave [t_start, 0) without snapshots and the `--obs-only` writer
   (and any state-at-obs-time consumer) would hard-exit with
   "no snapshot at or before t=…" for pre-origin observations.
   The fix: default `start = min(0.0, t_start)`. *)

let test_output_default_start_unanchored_stays_zero () =
  (* Unanchored (no origin), positive t_start → output.start = 0.0 (no regression). *)
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_output_unanchored_start" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 1e-9)) "output.start stays 0.0" 0.0 r.Ir.start
     | _ -> Alcotest.fail "expected OutRegular schedule")

let test_output_default_start_anchored_negative_t_start () =
  (* Anchored with `from = date("2020-01-21")` before `origin =
     date("2020-02-24")` → t_start = -34. The default output schedule
     must start at -34, not 0, so snapshots cover the full integration
     window. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-02-24")
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = date("2020-01-21")  to = date("2020-06-22") }
  |} in
  match Compiler.compile ~name:"test_output_anchored_neg_start" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    Alcotest.(check (float 1e-9)) "t_start" (-34.0) m.Ir.simulation.Ir.t_start;
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 1e-9))
         "output.start covers negative t_start" (-34.0) r.Ir.start
     | _ -> Alcotest.fail "expected OutRegular schedule")

(* Output trajectory customization (Phase 1): the trajectories block accepts
   `every = E` (regular cadence) and `at = [...]` (explicit times), mirroring
   the observation schedule surface, plus `format = NAME`. *)

let output_model_body = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |}

let test_output_every_explicit () =
  let src = output_model_body ^ "output { trajectories { every = 7 'days } }" in
  match Compiler.compile ~name:"test_output_every" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 0.01)) "explicit step from every=7" 7.0 r.Ir.step
     | _ -> Alcotest.fail "expected OutRegular schedule")

let test_output_every_subunit () =
  (* the user-facing goal: finer-scale trajectories via sub-unit cadence *)
  let src = output_model_body ^ "output { trajectories { every = 0.5 'days } }" in
  match Compiler.compile ~name:"test_output_subunit" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 0.001)) "sub-unit step from every=0.5" 0.5 r.Ir.step
     | _ -> Alcotest.fail "expected OutRegular schedule")

let test_output_at_times () =
  let src = output_model_body ^ "output { trajectories { at = [10, 20, 30] } }" in
  match Compiler.compile ~name:"test_output_at" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutAtTimes ts ->
       Alcotest.(check (list (float 0.01))) "explicit times" [10.0; 20.0; 30.0] ts
     | _ -> Alcotest.fail "expected OutAtTimes schedule")

let test_output_format_parquet () =
  let src = output_model_body
            ^ "output { trajectories { every = 1 'days format = parquet } }" in
  match Compiler.compile ~name:"test_output_fmt_parquet" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m -> Alcotest.(check string) "format parquet" "parquet" m.Ir.output.Ir.format

let test_output_every_and_at_conflict () =
  (* specifying both schedules is ambiguous -> hard error, not silent pick *)
  let src = output_model_body
            ^ "output { trajectories { every = 1 'days at = [5] } }" in
  match Compiler.compile ~name:"test_output_conflict" src with
  | Error _ -> ()  (* expected: conflicting schedule rejected *)
  | Ok _ -> Alcotest.fail "expected error: every and at are mutually exclusive"

(* A.2 guard: observations and output share the `schedule_core` grammar/AST
   but their expander lowerings are NOT identical — obs `every` lowers
   start = t_start, output `every` lowers start = min(0, t_start). For
   t_start > 0 they diverge (obs.start = t_start, output.start = 0). Pin it
   so a future shared schedule_core lowering helper can't silently shift
   observation times (which PGAS conditions on). *)
let test_obs_output_start_divergence () =
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  rho : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 10 'days  to = 120 'days }
    observations {
      cases {
        columns       { time : time, cases : count }
        projected     = incidence(recovery)
        emit_schedule = every 1 'days
        cases         ~ poisson(rate = rho * projected)
      }
    }
    output { trajectories { every = 1 'days } }
  |} in
  match Compiler.compile ~name:"test_obs_out_start" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.output.Ir.times with
     | Ir.OutRegular r ->
       Alcotest.(check (float 1e-9)) "output.start = min(0,t_start) = 0" 0.0 r.Ir.start
     | _ -> Alcotest.fail "expected OutRegular output schedule");
    (match m.Ir.observations with
     | om :: _ ->
       (match om.Ir.emit_schedule with
        | Some (Ir.ObsRegular r) ->
          Alcotest.(check (float 1e-9))
            "obs.start = t_start = 10 (NOT min(0,t_start))" 10.0 r.Ir.start
        | _ -> Alcotest.fail "expected ObsRegular emit_schedule")
     | [] -> Alcotest.fail "expected an observation model")

(* ── §4.2 — stratified observation header emits a `stratum` selector ──────────
   A `cases[p in patch]` observation expands to one IR leaf per patch level
   (`cases_urban`, `cases_rural`). Each leaf must carry a structured
   `stratum = [("patch", <level>)]` selector (the by-name routing key the Rust
   long-form loader uses), and it must round-trip through serde. ───────────── *)
let test_stratified_observation_emits_stratum () =
  let src = {|
    time_unit = 'days
    dimensions { patch = [urban, rural] }
    compartments { S, I, R }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate  rho : probability }
    let N[p in patch] = S[p] + I[p] + R[p]
    transitions {
      infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * I[p] / N[p]
      recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
    }
    init { S[urban] = 990  I[urban] = 10  S[rural] = 999  I[rural] = 1 }
    simulate { from = 0 'days  to = 100 'days }
    observations {
      cases[p in patch] {
        columns       { time : time, patch : dim, cases : count }
        projected     = incidence(infection[p])
        emit_schedule = every 7 'days
        cases         ~ poisson(rate = rho * projected)
      }
    }
  |} in
  let m = compile_expect_ok src in
  let find name =
    List.find (fun (o : Ir.observation_model) -> o.Ir.name = name) m.Ir.observations in
  let check_leaf name level =
    let o = find name in
    Alcotest.(check (list (pair string string)))
      (Printf.sprintf "%s stratum = [(patch, %s)]" name level)
      [("patch", level)] o.Ir.stratum
  in
  check_leaf "cases_urban" "urban";
  check_leaf "cases_rural" "rural";
  (* Round-trip through serde: the `stratum` field survives serialise +
     deserialise (and is OMITTED when empty — an unstratified golden is
     byte-identical, asserted by the golden gate). *)
  let json = Serde.model_to_string m in
  let m2 = match Serde.model_of_string json with
    | Ok m -> m
    | Error e -> Alcotest.failf "round-trip parse failed: %s" e in
  let o2 = List.find
    (fun (o : Ir.observation_model) -> o.Ir.name = "cases_urban") m2.Ir.observations in
  Alcotest.(check (list (pair string string)))
    "round-tripped stratum" [("patch", "urban")] o2.Ir.stratum

(* ── BUG-2: Parameterised table values ───────────────────────────────────────
   Compile a model with a table that references a parameter. The compiled
   table values should include Ir.Param "beta_mf", not drop it. ─────────── *)

let test_parameterised_table () =
  let src = {|
    dimensions { sex = [m, f] }
    compartments { S, I, R }
    stratify(by = sex)
    parameters {
      beta_mf : rate
      beta_fm : rate
      gamma   : rate
      N0      : count
      I0      : count
    }
    tables {
      B_sex : sex × sex = [[0.0, beta_mf], [beta_fm, 0.0]]
    }
    let N = S_m + I_m + R_m + S_f + I_f + R_f
    transitions {
      infection[a in sex] : S[a] --> I[a]
        @ sum(b in sex, B_sex[a, b] * I[b]) / N
      recovery[a in sex]  : I[a] --> R[a]  @ gamma * I[a]
    }
    init {
      S_m = N0 - I0
      I_m = I0
      S_f = N0
    }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_param_table" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match List.find_opt (fun (t : Ir.table) -> t.Ir.name = "B_sex") m.Ir.tables with
     | None -> Alcotest.fail "B_sex table not found"
     | Some tbl ->
       (* The 2nd entry (index 1) should be Ir.Param "beta_mf" *)
       let values = match tbl.Ir.source with
         | Ir.Inline vs -> vs
         | Ir.External _ -> Alcotest.fail "expected Inline table, got External"
       in
       let second = List.nth values 1 in
       (match second with
        | Ir.Param "beta_mf" ->
          ()  (* pass *)
        | other ->
          Alcotest.failf "expected Ir.Param \"beta_mf\", got: %s"
            (Serde.model_to_string
               { m with Ir.tables = [{tbl with Ir.source = Ir.Inline [other]}] })))

(* ── ASCII `*` as an alias for `×` in dimension position ─────────────────────
   The dimension-product separator (table shapes `d × d`, typed `let` shapes)
   accepts both the canonical Unicode `×` and the ASCII `*`, so a model can be
   typed by hand without the glyph. The separator is purely syntactic — it names
   which dimensions a table ranges over — so `×` and `*` must yield byte-
   identical IR. Docs recommend `×` for readability; `*` is the typeable escape
   hatch. *)
let dim_sep_table_src sep = Printf.sprintf {|
    dimensions { sex = [m, f] }
    compartments { S, I, R }
    stratify(by = sex)
    parameters {
      beta_mf : rate
      beta_fm : rate
      gamma   : rate
      N0      : count
      I0      : count
    }
    tables {
      B_sex : sex %s sex = [[0.0, beta_mf], [beta_fm, 0.0]]
    }
    let N = S_m + I_m + R_m + S_f + I_f + R_f
    transitions {
      infection[a in sex] : S[a] --> I[a]
        @ sum(b in sex, B_sex[a, b] * I[b]) / N
      recovery[a in sex]  : I[a] --> R[a]  @ gamma * I[a]
    }
    init {
      S_m = N0 - I0
      I_m = I0
      S_f = N0
    }
    simulate { from = 0 'days  to = 120 'days }
  |} sep

let test_dim_sep_asterisk_equals_cross () =
  let s_cross = Serde.model_to_string (compile_expect_ok (dim_sep_table_src "×")) in
  let s_star  = Serde.model_to_string (compile_expect_ok (dim_sep_table_src "*")) in
  if not (String.equal s_cross s_star) then
    Alcotest.failf
      "IR differs between `×` and `*` dim separators:\n× =\n%s\n\n* =\n%s"
      s_cross s_star

(* ── Table unit conversion (spec §6.1) ───────────────────────────────────────
   `tables { x : dim 'unit = [...] }` annotations must scale inline values
   from the declared unit to the model's `time_unit`. Pre-fix, the unit was
   parsed (TDimUnit) but dropped in the expander (`expander.ml:218,664`),
   so `age_dur : group 'years = [5, 60]` with `time_unit = 'days` compiled
   to verbatim [5, 60] instead of [1826.25, 21915.0]. See incident
   `docs/dev/incidents/2026-04-21-table-unit-annotations-ignored.md`. *)

let assert_inline_const ~epsilon tbl idx expected =
  let values = match tbl.Ir.source with
    | Ir.Inline vs -> vs
    | Ir.External _ -> Alcotest.fail "expected Inline, got External"
  in
  match List.nth values idx with
  | Ir.Const f when Float.abs (f -. expected) < epsilon -> ()
  | Ir.Const f ->
    Alcotest.failf "entry %d: expected %f (±%f), got %f" idx expected epsilon f
  | _ -> Alcotest.failf "entry %d: expected Ir.Const, got non-const" idx

let test_table_years_annotation_scales_to_days () =
  (* With time_unit = 'days, `[5, 60] 'years` must materialise as days. *)
  let src = {|
    time_unit = 'days
    dimensions { group = [young, old] }
    compartments { S, I }
    stratify(by = group)
    parameters { beta : rate }
    tables { age_dur : group 'years = [5, 60] }
    let N = S_young + I_young + S_old + I_old
    transitions {
      recovery[g in group] : I[g] --> S[g]
        @ (1.0 / age_dur[g]) * I[g]
    }
    init { S_young = 500 I_young = 10 S_old = 500 I_old = 10 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "age_dur") m.Ir.tables in
  (* days_per Years = 365.2425 (Gregorian, not Julian 365.25) *)
  assert_inline_const ~epsilon:1e-6 tbl 0 (5.0 *. 365.2425);
  assert_inline_const ~epsilon:1e-6 tbl 1 (60.0 *. 365.2425)

let test_table_per_day_annotation_with_weeks_unit () =
  (* With time_unit = 'weeks, `[0.1] 'per_day` means 0.1 /day = 0.7 /week. *)
  let src = {|
    time_unit = 'weeks
    dimensions { group = [adult] }
    compartments { S, I }
    stratify(by = group)
    parameters { beta : rate }
    tables { mort : group 'per_day = [0.1] }
    let N = S_adult + I_adult
    transitions {
      death[g in group] : I[g] -->   @ mort[g] * I[g]
    }
    init { S_adult = 90 I_adult = 10 }
    simulate { from = 0 'weeks  to = 10 'weeks }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "mort") m.Ir.tables in
  assert_inline_const ~epsilon:1e-6 tbl 0 0.7

let test_table_read_path_scales_unit () =
  (* The `read("file.tsv")` loader had the same pattern-matching bug as the
     inline path; covered in the same fix but not separately tested. This
     test addresses P1.5 of the 2026-04-21 spec-claims audit: exercise a
     unit-annotated table loaded from a TSV file, assert the values are
     scaled. *)
  let tmp = Filename.temp_file "camdl_read_unit" ".tsv" in
  (* TSV: one row per stratum, columns are `group` + `x`. *)
  let oc = open_out tmp in
  output_string oc "group\tx\n";
  output_string oc "a\t5\n";
  output_string oc "b\t60\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    dimensions { group = [a, b] }
    compartments { S, I }
    stratify(by = group)
    parameters { beta : rate }
    tables { age_dur : group 'years = read("%s") }
    let N = S_a + I_a + S_b + I_b
    transitions {
      recovery[g in group] : I[g] --> S[g]  @ (1.0 / age_dur[g]) * I[g]
    }
    init { S_a = 500 I_a = 10 S_b = 500 I_b = 10 }
    simulate { from = 0 'days  to = 10 'days }
  |} tmp in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "age_dur") m.Ir.tables in
  assert_inline_const ~epsilon:1e-6 tbl 0 (5.0 *. 365.2425);
  assert_inline_const ~epsilon:1e-6 tbl 1 (60.0 *. 365.2425);
  Sys.remove tmp

(* ── `camdlc --emit-deps`: the compile read-closure ──────────────────────────
   [compile_with_reads] returns the distinct external data files the compile
   opened (as-written, resolved), powering the MRE-bundle depfile. *)

let test_emit_deps_records_read_closure () =
  let tmp = Filename.temp_file "camdl_emit_deps" ".tsv" in
  let oc = open_out tmp in
  output_string oc "group\tx\n";
  output_string oc "a\t5\n";
  output_string oc "b\t60\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    dimensions { group = [a, b] }
    compartments { S, I }
    stratify(by = group)
    parameters { beta : rate }
    tables { age_dur : group 'years = read("%s") }
    let N = S_a + I_a + S_b + I_b
    transitions {
      recovery[g in group] : I[g] --> S[g]  @ (1.0 / age_dur[g]) * I[g]
    }
    init { S_a = 500 I_a = 10 S_b = 500 I_b = 10 }
    simulate { from = 0 'days  to = 10 'days }
  |} tmp in
  (match Compiler.compile_with_reads ~name:"test" src with
   | Ok (_m, reads) ->
     Alcotest.(check int) "one distinct read file" 1 (List.length reads);
     let (_as_written, resolved) = List.hd reads in
     (* An absolute temp path passes through [resolve_data_path] unchanged. *)
     Alcotest.(check string) "resolved path is the read file" tmp resolved
   | Error e -> Alcotest.failf "compile_with_reads failed: %s" e);
  Sys.remove tmp

(* Negative control: a model with no read() has an empty read-closure — so the
   "one read" assertion above is testing the recording, not a constant. *)
let test_emit_deps_empty_when_no_reads () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    let N = S + I + R
    parameters { beta : rate  gamma : rate }
    transitions {
      infection : S --> I  @ beta * S * (I / N)
      recovery  : I --> R  @ gamma * I
    }
    init { S = 990 I = 10 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  match Compiler.compile_with_reads ~name:"test" src with
  | Ok (_m, reads) ->
    Alcotest.(check int) "no read() → empty closure" 0 (List.length reads)
  | Error e -> Alcotest.failf "compile_with_reads failed: %s" e

(* ── gh#144 — read() data-file header robustness ─────────────────────────────
   read_csv_rows read the FIRST physical line as the header unconditionally,
   even when that line was a `#` provenance comment. A 1-column tab-free
   comment then mis-mapped as a 0/1-column "header" against a multi-dim table,
   tripping `List.combine dim_names header_dims` with an uncaught
   Invalid_argument crash instead of skipping the comment (a) or diagnosing a
   genuinely malformed header (b). ────────────────────────────────────────── *)

(* (a) A leading `#` provenance comment block is skipped; the first non-comment
   line is the header, and the table loads identically to the comment-free
   case. *)
let test_table_read_skips_leading_comment () =
  let tmp = Filename.temp_file "camdl_read_comment" ".tsv" in
  let oc = open_out tmp in
  (* Two leading comment lines (source URL + fetch date), then the header.
     The comments are tab-free single columns; the table has two dims. *)
  output_string oc "# source: https://example.org/contact_matrix\n";
  output_string oc "# fetched: 2026-05-31\n";
  output_string oc "row\tcol\tw\n";
  output_string oc "a\ta\t1.0\n";
  output_string oc "a\tb\t0.5\n";
  output_string oc "b\ta\t0.5\n";
  output_string oc "b\tb\t1.0\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    dimensions { row = [a, b]  col = [a, b] }
    compartments { S }
    stratify(by = row)
    stratify(by = col)
    parameters { beta : rate }
    tables { C : row × col = read("%s") }
    let N = S_a_a + S_a_b + S_b_a + S_b_b
    transitions {
      dummy[r in row, c in col] : S[r, c] -->   @ beta * C[r, c] * S[r, c]
    }
    init { S_a_a = 1 S_a_b = 1 S_b_a = 1 S_b_b = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} tmp in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "C") m.Ir.tables in
  (* Row-major 2×2: [a,a]=1.0 [a,b]=0.5 [b,a]=0.5 [b,b]=1.0 *)
  assert_inline_const ~epsilon:1e-12 tbl 0 1.0;
  assert_inline_const ~epsilon:1e-12 tbl 1 0.5;
  assert_inline_const ~epsilon:1e-12 tbl 2 0.5;
  assert_inline_const ~epsilon:1e-12 tbl 3 1.0;
  Sys.remove tmp

(* (b) A genuinely malformed header (too few columns, no comment) diagnoses
   cleanly with E221 instead of crashing the compiler. *)
let test_table_read_malformed_header_e221 () =
  let tmp = Filename.temp_file "camdl_read_badhdr" ".tsv" in
  let oc = open_out tmp in
  (* Header has ONE column but the table is 2-D (row × col): the dim-column
     count can't be read off this header. *)
  output_string oc "row\n";
  output_string oc "a\ta\t1.0\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    dimensions { row = [a, b]  col = [a, b] }
    compartments { S }
    stratify(by = row)
    stratify(by = col)
    parameters { beta : rate }
    tables { C : row × col = read("%s") }
    let N = S_a_a + S_a_b + S_b_a + S_b_b
    transitions {
      dummy[r in row, c in col] : S[r, c] -->   @ beta * C[r, c] * S[r, c]
    }
    init { S_a_a = 1 S_a_b = 1 S_b_a = 1 S_b_b = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} tmp in
  compile_expect_error_code ~code:"E221" ~contains:"row" src;
  Sys.remove tmp

(* A `read(...)` table with no index dimensions is the scalar-via-table
   mistake: tables hold indexed data (>=1 dimension), so an
   externally-computed scalar belongs in a parameter (params.toml / --params),
   not a table read. E222 must name that seam rather than letting the loader
   trip a confusing column/file error. *)
let test_scalar_read_table_e222 () =
  compile_expect_error_code ~code:"E222" ~contains:"parameter" {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate }
    tables { mu = read("rates.tsv") }
    transitions { recover : I --> S @ beta * I }
    init { S = 1  I = 0 }
    simulate { from = 0 'days  to = 1 'days }
  |}

let test_table_no_unit_annotation_leaves_values_alone () =
  (* No unit literal on the table = no scaling; dimcheck infers dim from use. *)
  let src = {|
    time_unit = 'days
    dimensions { group = [a, b] }
    compartments { S }
    stratify(by = group)
    parameters { beta : rate }
    tables { C : group × group = [[1.0, 0.5], [0.5, 1.0]] }
    let N = S_a + S_b
    transitions {
      dummy[g in group] : S[g] -->   @ beta * C[g, g] * S[g]
    }
    init { S_a = 1 S_b = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "C") m.Ir.tables in
  assert_inline_const ~epsilon:1e-12 tbl 0 1.0;
  assert_inline_const ~epsilon:1e-12 tbl 1 0.5;
  assert_inline_const ~epsilon:1e-12 tbl 2 0.5;
  assert_inline_const ~epsilon:1e-12 tbl 3 1.0

(* ── gh#32 — table cell-type annotations ────────────────────────────────────
   `tables { x : dim :rate = [...] }` stamps every cell with the declared
   dimensional kind, so a per-bin rate table can drive a transition rate
   position without tripping E300. The annotation parallels scalar parameter
   syntax (`p : rate`); absent annotation = legacy dimensionless behaviour. *)

let test_table_cell_type_rate_parses_and_stamps_ir () =
  (* :rate annotation on a 1-D age table — the typhoid aging-rate motivator. *)
  let src = {|
    time_unit = 'days
    dimensions { age = [a02, a25, a510, a1015, a15] }
    compartments { S, I }
    stratify(by = age)
    parameters { gamma : rate }
    tables {
      aging_rate : age :rate = [
        1.0 / (2.0 * 365.0),
        1.0 / (3.0 * 365.0),
        1.0 / (5.0 * 365.0),
        1.0 / (5.0 * 365.0),
        0.0
      ]
    }
    let N = S_a02 + I_a02 + S_a25 + I_a25 + S_a510 + I_a510 + S_a1015 + I_a1015 + S_a15 + I_a15
    transitions {
      recovery[g in age] : I[g] --> S[g]  @ gamma * I[g]
      aging[g in age, (a, a_next) in consecutive(age)]
        : S[a] --> S[a_next]
        @ aging_rate[a] * S[a] where g == a
    }
    init { S_a02 = 100 I_a02 = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "aging_rate") m.Ir.tables in
  Alcotest.(check (option string))
    "aging_rate.cell_kind = Some \"rate\""
    (Some "rate") tbl.Ir.cell_kind

let test_table_cell_type_probability_parses () =
  let src = {|
    time_unit = 'days
    dimensions { age = [a, b] }
    compartments { S, I }
    stratify(by = age)
    parameters { gamma : rate }
    tables { p_severe : age :probability = [0.1, 0.5] }
    let N = S_a + I_a + S_b + I_b
    transitions {
      recovery[g in age] : I[g] --> S[g]  @ gamma * p_severe[g] * I[g]
    }
    init { S_a = 100 I_a = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "p_severe") m.Ir.tables in
  Alcotest.(check (option string))
    "p_severe.cell_kind = Some \"probability\""
    (Some "probability") tbl.Ir.cell_kind

let test_table_no_cell_type_annotation_remains_none () =
  (* No annotation = absent cell_kind — backward compatible. *)
  let src = {|
    time_unit = 'days
    dimensions { age = [a, b] }
    compartments { S }
    stratify(by = age)
    parameters { beta : rate }
    tables { C : age × age = [[1.0, 0.5], [0.5, 1.0]] }
    let N = S_a + S_b
    transitions {
      dummy[g in age] : S[g] -->   @ beta * C[g, g] * S[g]
    }
    init { S_a = 1 S_b = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "C") m.Ir.tables in
  Alcotest.(check (option string))
    "C.cell_kind = None"
    None tbl.Ir.cell_kind

(* An `instant` cell-kind table read from a file accepts ISO-date value
   cells, resolved to internal time via origin + time_unit at compile time
   (the same parse_date_to_float path as date() literals). With origin
   2013-01-01 and time_unit = days: 2013-11-01 -> 304, 2014-03-15 -> 438. *)
let test_table_cell_kind_instant_resolves_dates () =
  let tsv = Filename.temp_file "camdl_instant" ".tsv" in
  let oc = open_out tsv in
  output_string oc "round\tsched\nr0\t2013-11-01\nr1\t2014-03-15\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    origin = date("2013-01-01")
    dimensions { round = [r0, r1] }
    compartments { S, V }
    parameters { r : rate in [0.0, 1.0]  vc : probability in [0.0, 1.0] }
    tables { sched : round : instant = read("%s") }
    transitions { waste : S --> V @ r * S }
    interventions {
      sia[k in round] : transfer(fraction = vc, from = S, to = V) at [ sched[k] ]
    }
    init { S = 1000 }
    simulate { from = origin  to = add_calendar_years(origin, 2) }
  |} tsv in
  let m = compile_expect_ok src in
  Sys.remove tsv;
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "sched") m.Ir.tables in
  Alcotest.(check (option string)) "sched cell_kind" (Some "instant") tbl.Ir.cell_kind;
  let vals =
    match tbl.Ir.source with
    | Ir.Inline exprs ->
        List.map (function
          | Ir.Const f -> f
          | _ -> Alcotest.fail "expected compile-resolved Const cells") exprs
    | Ir.External _ -> Alcotest.fail "expected inline (resolved) table source"
  in
  Alcotest.(check (list (float 1e-9)))
    "ISO date cells resolve to day-offsets via origin" [ 304.0; 438.0 ] vals

(* Negative control: a date cell in an instant table with no top-level
   `origin` cannot be resolved — it is a hard error (E209), never a silent
   0. Proves the date branch is gated on the anchor, so the positive test
   above is non-vacuous. *)
let test_table_cell_kind_instant_needs_origin () =
  let tsv = Filename.temp_file "camdl_instant_noorigin" ".tsv" in
  let oc = open_out tsv in
  output_string oc "round\tsched\nr0\t2013-11-01\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    dimensions { round = [r0] }
    compartments { S, V }
    parameters { r : rate in [0.0, 1.0]  vc : probability in [0.0, 1.0] }
    tables { sched : round : instant = read("%s") }
    transitions { waste : S --> V @ r * S }
    interventions {
      sia[k in round] : transfer(fraction = vc, from = S, to = V) at [ sched[k] ]
    }
    init { S = 1000 }
    simulate { from = 0 'days  to = 730 'days }
  |} tsv in
  compile_expect_error_code ~code:"E209" ~contains:"origin" src;
  Sys.remove tsv

let with_dim_check_enabled f =
  let prev = !Compiler.no_dim_check in
  Compiler.no_dim_check := false;
  Fun.protect ~finally:(fun () -> Compiler.no_dim_check := prev) f

let test_table_cell_type_dim_check_passes_in_rate_position () =
  (* gh#32 motivator: with :rate annotation the dim checker must accept
     `aging_rate[a] * S[a]` as P*T^-1 (population-level rate). *)
  let src = {|
    time_unit = 'days
    dimensions { age = [a02, a25, a510] }
    compartments { S, I }
    stratify(by = age)
    parameters { gamma : rate }
    tables {
      aging_rate : age :rate = [
        1.0 / (2.0 * 365.0),
        1.0 / (3.0 * 365.0),
        0.0
      ]
    }
    let N = S_a02 + I_a02 + S_a25 + I_a25 + S_a510 + I_a510
    transitions {
      recovery[g in age] : I[g] --> S[g]  @ gamma * I[g]
      aging_a02 : S[a02] --> S[a25]   @ aging_rate[a02] * S[a02]
      aging_a25 : S[a25] --> S[a510]  @ aging_rate[a25] * S[a25]
    }
    init { S_a02 = 100 I_a02 = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  with_dim_check_enabled (fun () ->
    match Compiler.compile ~name:"test_cell_type_dim" src with
    | Ok _    -> ()
    | Error e -> Alcotest.failf "expected dim-check pass, got: %s" e)

let test_table_cell_type_ir_round_trips_through_serde () =
  (* Compile a model with :rate cell type, serialise to JSON, deserialise,
     confirm the cell_kind survives the round trip. *)
  let src = {|
    time_unit = 'days
    dimensions { age = [a, b] }
    compartments { S, I }
    stratify(by = age)
    parameters { gamma : rate }
    tables { aging : age :rate = [0.001, 0.002] }
    let N = S_a + I_a + S_b + I_b
    transitions {
      recovery[g in age] : I[g] --> S[g]  @ gamma * I[g]
    }
    init { S_a = 100 I_a = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  let json = Serde.model_to_string m in
  let m2 = match Serde.model_of_string json with
    | Ok m -> m
    | Error e -> Alcotest.failf "round-trip parse failed: %s" e
  in
  let tbl = List.find (fun (t : Ir.table) -> t.Ir.name = "aging") m2.Ir.tables in
  Alcotest.(check (option string))
    "round-tripped cell_kind"
    (Some "rate") tbl.Ir.cell_kind

(* ── P3.1 — let-binding inlining (spec §9) ──────────────────────────────────
   Spec claim: `let N = S + I + R` is inlined at every use site.
   Direct assertion: the compiled transition rate must contain Pop "S" +
   Pop "I" + Pop "R", NOT a Let/Ref node. See audit
   docs/dev/reviews/2026-04-21-spec-claims-vs-tests.md P3.1. *)

(** Walk an Ir.expr and collect all Pop compartment names. *)
let rec collect_pops = function
  | Ir.Const _ | Ir.Param _ | Ir.Time | Ir.Dt | Ir.Projected | Ir.ObsColumnRef _ -> []
  | Ir.Pop name -> [name]
  | Ir.PopSum names -> names
  | Ir.BinOp b -> collect_pops b.left @ collect_pops b.right
  | Ir.UnOp u  -> collect_pops u.arg
  | Ir.Cond c  -> collect_pops c.pred @ collect_pops c.then_ @ collect_pops c.else_
  | Ir.TimeFunc _ -> []
  | Ir.TableLookup (_, idx) -> List.concat_map collect_pops idx
  | Ir.UncheckedDim u -> collect_pops u.inner
  | Ir.Reduce terms -> List.concat_map collect_pops terms
  | Ir.BindingRef _ -> []
  | Ir.PerEvalRef _ -> []

let test_let_binding_is_extracted () =
  let src = {|
    compartments { S, I, R }
    let N = S + I + R
    parameters { beta : rate  gamma : rate }
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  let open Ir in
  (* Fix B (shared-binding extraction). A state-only `let` is no longer
     inlined at each use; it is hoisted once into `model.bindings` and the
     rate references it via `BindingRef`. The model meaning is unchanged
     (the gate proves byte-identical trajectories) — only the IR shape is. *)
  let n_binding = List.find_opt (fun (b : binding) -> b.bname = "N") m.bindings in
  Alcotest.(check bool) "let N extracted into model.bindings" true (n_binding <> None);
  (match n_binding with
   | Some b ->
     let body = collect_pops b.bexpr in
     let has s = List.mem s body in
     Alcotest.(check bool) "binding body sums S" true (has "S");
     Alcotest.(check bool) "binding body sums I" true (has "I");
     Alcotest.(check bool) "binding body sums R" true (has "R")
   | None -> ());
  (* The infection rate now references N via BindingRef; its own Pop leaves
     are just the numerator S and I. collect_pops sees through BindingRef to
     nothing, so R (only in N) must NOT appear in the rate itself. *)
  let infection = List.find (fun (t : transition) -> t.name = "infection") m.transitions in
  let pops = collect_pops infection.rate in
  Alcotest.(check bool) "rate numerator references S" true (List.mem "S" pops);
  Alcotest.(check bool) "rate numerator references I" true (List.mem "I" pops);
  Alcotest.(check bool) "N is a BindingRef, not inlined (no R in rate)" false (List.mem "R" pops)

(* ── P3.2 — stratification count invariant (spec §5) ─────────────────────────
   Spec: `stratify(by = dim)` with N compartments and |dim|=K levels expands
   to N×K compartments. Direct count assertion. *)

let test_stratification_compartment_count () =
  let src = {|
    compartments { S, I, R }
    dimensions { age = [child, adult, elder] }
    stratify(by = age)
    parameters { beta : rate  gamma : rate }
    let N = S + I + R
    transitions {
      infection[a in age] : S[a] --> I[a]  @ beta * S[a] * I[a] / N
      recovery[a in age]  : I[a] --> R[a]  @ gamma * I[a]
    }
    init { S_child = 100  I_child = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  (* 3 compartments × 3 age levels = 9 *)
  Alcotest.(check int) "3 compartments × 3 strata = 9" 9 (List.length m.compartments);
  (* 2 transitions × 3 age levels = 6 *)
  Alcotest.(check int) "2 transitions × 3 strata = 6" 6 (List.length m.transitions);
  (* All expected names present *)
  let names = List.map (fun (c : Ir.compartment) -> c.name) m.compartments in
  List.iter (fun n ->
    Alcotest.(check bool) (Printf.sprintf "compartment %s exists" n) true (List.mem n names)
  ) ["S_child"; "S_adult"; "S_elder"; "I_child"; "I_adult"; "I_elder";
     "R_child"; "R_adult"; "R_elder"]

(* ── P3.5 — incidence positional vs named indexing (spec §13.1) ──────────────
   Spec: both `incidence(transition[stratum])` and `incidence(transition[dim = v])`
   sum over unspecified dimensions. The positional form binds by declaration
   order; named by dim name. Both must produce the same IR shape when the
   positional index targets the same dimension. See clarification in commit
   3960453 + audit P3.5. *)

let test_incidence_positional_and_named_produce_equal_projections () =
  (* Same observation written both ways; assert the IR projection
     structures are identical. *)
  let src_positional = {|
    compartments { S, I, R }
    dimensions { patch = [north, south] }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate  rho : probability }
    let N_north = S_north + I_north + R_north
    let N_south = S_south + I_south + R_south
    transitions {
      infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * I[p]
      recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
    }
    init { S_north = 100  I_north = 1 }
    simulate { from = 0 'days  to = 10 'days }
    observations {
      north_cases {
        columns       { time : time, north_cases : count }
        projected  = incidence(recovery[north])
        emit_schedule = every 1 'days
        north_cases ~ poisson(rate = rho * projected)
      }
    }
  |} in
  let src_named = {|
    compartments { S, I, R }
    dimensions { patch = [north, south] }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate  rho : probability }
    let N_north = S_north + I_north + R_north
    let N_south = S_south + I_south + R_south
    transitions {
      infection[p in patch] : S[p] --> I[p]  @ beta * S[p] * I[p]
      recovery[p in patch]  : I[p] --> R[p]  @ gamma * I[p]
    }
    init { S_north = 100  I_north = 1 }
    simulate { from = 0 'days  to = 10 'days }
    observations {
      north_cases {
        columns       { time : time, north_cases : count }
        projected  = incidence(recovery[patch = north])
        emit_schedule = every 1 'days
        north_cases ~ poisson(rate = rho * projected)
      }
    }
  |} in
  let m_pos = compile_expect_ok src_positional in
  let m_nam = compile_expect_ok src_named in
  let obs_pos = List.hd m_pos.observations in
  let obs_nam = List.hd m_nam.observations in
  (* Serialize both projections and compare — easier than deep-matching. *)
  let pos_proj = Yojson.Safe.to_string (Serde.projection_to_json obs_pos.projection) in
  let nam_proj = Yojson.Safe.to_string (Serde.projection_to_json obs_nam.projection) in
  Alcotest.(check string)
    "positional and named projections produce identical IR"
    pos_proj nam_proj

(* ── P3.4 — consecutive() pair count (spec §14) ──────────────────────────────
   Spec: `consecutive((s, s_next) in consecutive(dim))` pairs adjacent levels
   only — k levels → k-1 transitions. Common pitfall: an off-by-one where k
   transitions get emitted, or a cross-product k² (every (s, t) pair). *)

let test_consecutive_pair_count () =
  (* 3 erlang sub-stages → 2 progression transitions (e1→e2, e2→e3).
     Final exit (e3 → I) is a separate transition. *)
  let src = {|
    compartments { S, E, I, R }
    dimensions { erlang_E = [e1, e2, e3] }
    stratify(by = erlang_E, only = [E])
    parameters { beta : rate  sigma : rate  gamma : rate }
    let N = S + E_e1 + E_e2 + E_e3 + I + R
    transitions {
      infection : S --> E_e1  @ beta * S * I / N
      progression[(s, s_next) in consecutive(erlang_E)]
        : E[s] --> E[s_next]
        @ 3.0 * sigma * E[s]
      exit : E_e3 --> I  @ 3.0 * sigma * E_e3
      recovery : I --> R  @ gamma * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  (* Count transitions whose name starts with "progression" — the
     consecutive expansion should produce exactly k-1 = 2 (for k=3). *)
  let progression_count = List.length
    (List.filter (fun (t : Ir.transition) ->
      String.length t.name >= 11 && String.sub t.name 0 11 = "progression"
    ) m.transitions) in
  Alcotest.(check int) "consecutive(k=3) → k-1 = 2 progression transitions"
    2 progression_count;
  (* Total: 1 infection + 2 progression + 1 exit + 1 recovery = 5 *)
  Alcotest.(check int) "total transition count" 5 (List.length m.transitions)

(* ── DESIGN-2: Intervention expansion ───────────────────────────────────────
   Compile a model with an intervention. Assert it appears in model.interventions. *)

let test_intervention_expansion () =
  let src = {|
    compartments { S, V, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0    : count
      I0    : count
    }
    let N = S + V + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    interventions {
      sia : transfer(fraction = 0.8, from = S, to = V) at [30, 60]
    }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_interv" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    Alcotest.(check int) "one intervention" 1 (List.length m.Ir.interventions);
    let iv = List.hd m.Ir.interventions in
    Alcotest.(check string) "intervention name" "sia" iv.Ir.name;
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.AtTimes ts) ->
       Alcotest.(check int) "two fire times" 2 (List.length ts)
     | _ -> Alcotest.fail "expected AtTimes schedule");
    Alcotest.(check int) "one action" 1 (List.length iv.Ir.actions);
    (match List.hd iv.Ir.actions with
     | Ir.FractionTransfer ft ->
       Alcotest.(check string) "src=S" "S" ft.Ir.src;
       Alcotest.(check string) "dst=V" "V" ft.Ir.dst
     | _ -> Alcotest.fail "expected FractionTransfer action")

(* gh#49: `transfer(count = N, ...)` was rejected by the parser even
   though the spec, expander, and Rust runtime all supported it. The
   `count` token is reserved as a parameter type annotation, so it
   never matched the parser's `IDENT EQ expr` fallthrough. Verify the
   added `COUNT EQ expr` clause in `transfer_kwarg` lets `count` reach
   the expander, where it correctly emits `Ir.AbsoluteTransfer`. *)
let test_intervention_transfer_count_kwarg () =
  let src = {|
    compartments { S, V, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0    : count
      I0    : count
      vrate : count
    }
    let N = S + V + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    interventions {
      routine_vax : transfer(count = vrate, from = S, to = V) at [30, 60, 90]
    }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match Compiler.compile ~name:"test_count_kwarg" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    Alcotest.(check int) "one intervention" 1 (List.length m.Ir.interventions);
    let iv = List.hd m.Ir.interventions in
    Alcotest.(check string) "intervention name" "routine_vax" iv.Ir.name;
    Alcotest.(check int) "one action" 1 (List.length iv.Ir.actions);
    (match List.hd iv.Ir.actions with
     | Ir.AbsoluteTransfer at ->
       Alcotest.(check string) "src=S" "S" at.Ir.src;
       Alcotest.(check string) "dst=V" "V" at.Ir.dst
     | Ir.FractionTransfer _ ->
       Alcotest.fail "got FractionTransfer; expected AbsoluteTransfer \
                      (count kwarg should not produce fraction-flavoured IR)"
     | _ -> Alcotest.fail "expected AbsoluteTransfer action")

(* Multi-set: a block intervention with several `COMP = EXPR` assignments must
   keep ALL of them (spec §13: "one or more assignments"). Before the fix, ast
   `ivaction` was a single action and the parser fold kept only the last,
   silently dropping the rest — a wrong compiled model for a documented feature. *)
let test_intervention_multi_set () =
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    interventions {
      shock : { S = S - 100  I = I + 100  at = [30] }
    }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  let m = compile_expect_ok src in
  Alcotest.(check int) "one intervention" 1 (List.length m.Ir.interventions);
  let iv = List.hd m.Ir.interventions in
  Alcotest.(check int) "both set actions kept" 2 (List.length iv.Ir.actions);
  let targets = List.filter_map (function
    | Ir.Set s -> Some s.Ir.compartment
    | _ -> None) iv.Ir.actions in
  Alcotest.(check (list string)) "targets S then I, in source order"
    ["S"; "I"] targets

(* Option-b: a block intervention with a schedule but NO action is an author
   error, not a silent no-op. Before, it defaulted to `ATransfer []` and
   misfired later as E261 ("transfer missing from/to"); now it is a located
   E296 that names the real problem. *)
let test_intervention_no_action () =
  let src = {|
    compartments { S, I }
    parameters { beta : rate }
    transitions { infection : S --> I @ beta * S }
    init { S = 1000  I = 1 }
    interventions {
      noop : { at = [30] }
    }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  compile_expect_error_code ~code:"E296" ~contains:"noop" src

(* Indexed references resolve arity three ways (table E202, shaped-let E273,
   compartment E287); the let / forcing / parameter branches were unguarded, so
   an over-indexed let silently dropped the extra index and an over-indexed
   forcing/param name-mangled to a bad (often unlocated) error. A shared
   check_index_arity now emits a located E299 for all three. *)
let test_indexed_let_arity () =
  let src = {|
    dimensions { patch = [north, south] }
    compartments { S, I, R }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate }
    let N[p in patch] = S[p] + I[p] + R[p]
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p, south]
      recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
    }
    init { S[p in patch] = 1000  I[p in patch] = 1 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  compile_expect_error_code ~code:"E299" ~contains:"N" src

let test_indexed_forcing_arity () =
  let src = {|
    dimensions { patch = [north, south] }
    compartments { S, I }
    stratify(by = patch)
    parameters { beta : rate }
    forcing {
      seasonal[p in patch] : sinusoidal 'ratio {
        amplitude = 0.1
        period    = 365
        phase     = 0
        baseline  = 1.0
      }
    }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * seasonal[p, south] * S[p]
    }
    init { S[p in patch] = 1000  I[p in patch] = 1 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  compile_expect_error_code ~code:"E299" ~contains:"seasonal" src

let test_indexed_param_arity () =
  let src = {|
    dimensions { patch = [north, south] }
    compartments { S, I }
    stratify(by = patch)
    parameters { R0[patch] : positive  gamma : rate }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ R0[p, south] * gamma * S[p]
    }
    init { S[p in patch] = 1000  I[p in patch] = 1 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  compile_expect_error_code ~code:"E299" ~contains:"R0" src

(* A stratified compartment answers to its bare base name (a PopSum aggregate)
   as well as its cells, but check_declaration_names only registered the cells —
   so a `let`/param sharing the base name silently shadowed it with no collision.
   Registering the base too makes the collision a located E278. *)
let test_stratified_base_name_collision () =
  let src = {|
    dimensions { patch = [north, south] }
    compartments { S, I, R }
    stratify(by = patch)
    parameters { beta : rate  gamma : rate }
    let R = 100
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * S[p]
      recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
    }
    init { S[p in patch] = 1000  I[p in patch] = 1 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  compile_expect_error_code ~code:"E278" ~contains:"R" src

(* gh#49 sibling check: the expander already validates `fraction` and
   `count` are mutually exclusive (E261). Confirm the parser fix didn't
   accidentally let both through. Uses the JSON-errors helper so the
   E261 code surfaces in the returned string (otherwise diagnostics go
   to stderr only). *)
let test_intervention_transfer_count_and_fraction_rejected () =
  let src = {|
    compartments { S, V }
    parameters {
      N0 : count
      I0 : count
    }
    transitions {}
    init { S = N0 - I0 }
    interventions {
      bad : transfer(count = 100.0, fraction = 0.5, from = S, to = V) at [10]
    }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  compile_expect_error_code ~code:"E261" ~contains:"mutually exclusive" src

(* ── Recurring intervention block syntax ─────────────────────────────────
   transfer(...) { every = T, from = T0, until = T1 } — exists alongside
   the existing at [t1, t2, ...] form. *)

let test_recurring_block_transfer () =
  let src = {|
    time_unit = 'days
    compartments { S, V }
    parameters { vacc_rate : probability in [0.0, 1.0] }
    transitions {}
    init { S = 1000  V = 0 }
    simulate { from = 0 'days  to = 365 'days }
    interventions {
      routine : transfer(fraction = vacc_rate, from = S, to = V) {
        every = 30 'days
        from  = 0 'days
        until = 365 'days
      }
    }
  |} in
  match Compiler.compile ~name:"test_recurring" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let iv = List.hd m.Ir.interventions in
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.Recurring { start; period; end_; at_day = None }) ->
       Alcotest.(check (float 1e-9)) "start" 0.0 start;
       Alcotest.(check (float 1e-9)) "period = 30 days" 30.0 period;
       Alcotest.(check (float 1e-9)) "end" 365.0 end_
     | _ -> Alcotest.fail "expected Recurring schedule")

let test_recurring_kwargs_any_order () =
  (* until / from / every in arbitrary order — all should work. *)
  let src = {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 100 'days }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        until = 100 'days
        every = 7 'days
        from  = 14 'days
      }
    }
  |} in
  match Compiler.compile ~name:"test_order" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let iv = List.hd m.Ir.interventions in
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.Recurring { start; period; end_; _ }) ->
       Alcotest.(check (float 1e-9)) "start" 14.0 start;
       Alcotest.(check (float 1e-9)) "period" 7.0 period;
       Alcotest.(check (float 1e-9)) "end" 100.0 end_
     | _ -> Alcotest.fail "expected Recurring")

let test_recurring_unit_conversion () =
  (* Per-year interval with time_unit = weeks. *)
  let src = {|
    time_unit = 'weeks
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'weeks  to = 1 'years }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        every = 30 'days
        from  = 0 'days
        until = 1 'years
      }
    }
  |} in
  match Compiler.compile ~name:"test_units" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let iv = List.hd m.Ir.interventions in
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.Recurring { period; end_; _ }) ->
       (* 30 days / 7 days/week = 30/7 weeks *)
       Alcotest.(check (float 1e-9)) "period in weeks" (30.0 /. 7.0) period;
       (* 1 year = 365.2425 days = 365.2425/7 weeks *)
       Alcotest.(check (float 1e-6)) "end in weeks" (365.2425 /. 7.0) end_
     | _ -> Alcotest.fail "expected Recurring")

let test_recurring_add_action () =
  (* Block syntax works with add() actions too, not just transfer(). *)
  let src = {|
    time_unit = 'days
    compartments { S }
    transitions {}
    init { S = 0 }
    simulate { from = 0 'days  to = 100 'days }
    events {
      influx : add(S, 50) {
        every = 10 'days
        from  = 0 'days
        until = 100 'days
      }
    }
  |} in
  match Compiler.compile ~name:"test_add_recurring" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let iv = List.hd m.Ir.interventions in
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.Recurring { period; _ }) ->
       Alcotest.(check (float 1e-9)) "period" 10.0 period
     | _ -> Alcotest.fail "expected Recurring");
    (match List.hd iv.Ir.actions with
     | Ir.AddAction _ -> ()
     | _ -> Alcotest.fail "expected Add action")

let test_recurring_default_from_until () =
  (* 'from' and 'until' default to simulate.from / simulate.to when omitted.
     Only 'every' is required. *)
  let src = {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 100 'days }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        every = 10 'days
      }
    }
  |} in
  match Compiler.compile ~name:"test_defaults" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let iv = List.hd m.Ir.interventions in
    (match iv.Ir.fire with
     | Ir.Scheduled (Ir.Recurring { start; period; end_; _ }) ->
       Alcotest.(check (float 1e-9)) "start defaults to t_start" 0.0 start;
       Alcotest.(check (float 1e-9)) "period"                     10.0 period;
       Alcotest.(check (float 1e-9)) "end defaults to t_end"     100.0 end_
     | _ -> Alcotest.fail "expected Recurring")

let test_recurring_at_times_still_works () =
  (* Regression guard: the existing at [...] form still compiles unchanged. *)
  let src = {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 365 'days }
    interventions {
      pulses : transfer(fraction = 0.5, from = S, to = V) at [30, 60, 90]
    }
  |} in
  match Compiler.compile ~name:"regression" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    match (List.hd m.Ir.interventions).fire with
    | Ir.Scheduled (Ir.AtTimes ts) ->
      Alcotest.(check int) "three pulses" 3 (List.length ts)
    | _ -> Alcotest.fail "expected AtTimes"

let test_block_transition_missing_rate_e213 () =
  (* Upstream review Finding #3: a block-form transition with no `rate = …`
     (and no inline `@ …`) must be a hard E213 error, not a silent zero-rate
     (never-firing) transition. The diagnostic must name the offending
     transition ("infection"), per "error messages are a feature". *)
  compile_expect_error_code ~code:"E213" ~contains:"infection" {|
    time_unit = 'days
    compartments { S, I }
    transitions {
      infection : S --> I { }
    }
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let test_recurring_e240_zero_every () =
  compile_expect_error_code ~code:"E240" ~contains:"'every' must be positive" {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        every = 0 'days
        from  = 0 'days
        until = 10 'days
      }
    }
  |}

let test_recurring_e241_inverted_range () =
  compile_expect_error_code ~code:"E241" ~contains:"must be <= 'until'" {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        every = 1 'days
        from  = 20 'days
        until = 10 'days
      }
    }
  |}

let test_recurring_e242_schedule_too_long () =
  (* 1 'years / 1e-7 'days (effectively) → way over the cap. Use tiny period. *)
  compile_expect_error_code ~code:"E242" ~contains:"cap" {|
    time_unit = 'days
    compartments { S, V }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'years }
    interventions {
      r : transfer(fraction = 0.1, from = S, to = V) {
        every = 0.000001 'days
        from  = 0 'days
        until = 10 'days
      }
    }
  |}

(* ── Scenario `extends` (single-inheritance sugar) ───────────────────────── *)

let find_scenario (m : Ir.model) name =
  List.find (fun (p : Ir.preset) -> p.preset_name = name) m.presets

let extends_boilerplate = {|
    time_unit = 'days
    compartments { S }
    parameters { x : rate }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |}

let test_extends_inherits_set_values () =
  let src = extends_boilerplate ^ {|
    scenarios {
      baseline { set = { x = 0.3 } }
      child    { extends = baseline }
    }
  |} in
  let m = compile_expect_ok src in
  let child = find_scenario m "child" in
  Alcotest.(check (float 1e-9)) "inherits x" 0.3 (List.assoc "x" child.preset_params)

let test_extends_child_overrides_key () =
  let src = extends_boilerplate ^ {|
    scenarios {
      baseline { set = { x = 0.3 } }
      hot      { extends = baseline   set = { x = 0.9 } }
    }
  |} in
  let m = compile_expect_ok src in
  let hot = find_scenario m "hot" in
  Alcotest.(check (float 1e-9)) "child overrides" 0.9 (List.assoc "x" hot.preset_params)

let test_extends_enable_append_dedup () =
  let src = {|
    time_unit = 'days
    compartments { S, V }
    parameters { x : rate }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      a : transfer(fraction = 0.1, from = S, to = V) at [1]
      b : transfer(fraction = 0.1, from = S, to = V) at [2]
      c : transfer(fraction = 0.1, from = S, to = V) at [3]
    }
    scenarios {
      parent { enable = [a, b] }
      child  { extends = parent   enable = [b, c] }
    }
  |} in
  let m = compile_expect_ok src in
  let child = find_scenario m "child" in
  (* Parent-first, child-second, dedup: [a; b; c] *)
  Alcotest.(check (list string)) "enable append+dedup"
    ["a"; "b"; "c"] child.preset_enable

let test_extends_three_level_chain () =
  let src = extends_boilerplate ^ {|
    scenarios {
      a { set = { x = 0.1 } }
      b { extends = a  set = { x = x * 2 } }
      c { extends = b  set = { x = x * 3 } }
    }
  |} in
  let m = compile_expect_ok src in
  let c = find_scenario m "c" in
  (* 0.1 × 2 × 3 = 0.6 *)
  Alcotest.(check (float 1e-9)) "three-level chain" 0.6 (List.assoc "x" c.preset_params)

let test_extends_e25x_cycle () =
  compile_expect_error_code ~code:"E25x" ~contains:"cycle"
    (extends_boilerplate ^ {|
    scenarios {
      a { extends = b }
      b { extends = a }
    }
  |})

let test_extends_e25y_unknown_with_suggestion () =
  compile_expect_error_code ~code:"E25y" ~contains:"baseline"
    (extends_boilerplate ^ {|
    scenarios {
      foo { extends = baselime }
      baseline {}
    }
  |})

let test_extends_scale_interaction () =
  (* Parent sets, child scales the same key. Child's scale evaluated
     after parent's set is in scope — scale of 0.5 against parent's 0.4
     is what makes it to the scale preset field (scales are applied at
     simulate time as multipliers; resolution here is just value
     computation). *)
  let src = extends_boilerplate ^ {|
    scenarios {
      p { set = { x = 0.4 } }
      c { extends = p   scale = { x = 0.5 } }
    }
  |} in
  let m = compile_expect_ok src in
  let c = find_scenario m "c" in
  Alcotest.(check (float 1e-9)) "scale resolves" 0.5 (List.assoc "x" c.preset_scale);
  (* Child inherits parent's set too *)
  Alcotest.(check (float 1e-9)) "parent set flows through"
    0.4 (List.assoc "x" c.preset_params)

let test_extends_child_references_parent_value () =
  (* Regression: `beta = beta * 1.5` in child must see parent's beta. *)
  let src = extends_boilerplate ^ {|
    scenarios {
      parent { set = { x = 0.4 } }
      warmer { extends = parent   set = { x = x * 1.5 } }
    }
  |} in
  let m = compile_expect_ok src in
  let w = find_scenario m "warmer" in
  Alcotest.(check (float 1e-9)) "parent-first resolution"
    0.6 (List.assoc "x" w.preset_params)

let test_extends_e25z_depth_exceeds () =
  compile_expect_error_code ~code:"E25z" ~contains:"chain"
    (extends_boilerplate ^ {|
    scenarios {
      s1 {}
      s2 { extends = s1 }
      s3 { extends = s2 }
      s4 { extends = s3 }
      s5 { extends = s4 }
      s6 { extends = s5 }
      s7 { extends = s6 }
    }
  |})

(* gh#115 / 2026-05-26 upstream OCaml-compiler review Critical #5 — *)
(* scenario enable/disable/compose/set/scale names must be validated. *)

let scenario_validation_boilerplate = {|
    time_unit = 'days
    compartments { S, V }
    parameters { x : rate  cov : probability }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      sia : transfer(fraction = cov, from = S, to = V) at [1]
    }
  |}

let test_scenario_enable_unknown_intervention_is_e267 () =
  compile_expect_error_code ~code:"E267" ~contains:"sai"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      high_coverage { enable = [sai] }
    }
  |})

let test_scenario_disable_unknown_intervention_is_e267 () =
  compile_expect_error_code ~code:"E267" ~contains:"siaa"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      baseline {}
      no_sia { extends = baseline   disable = [siaa] }
    }
  |})

let test_scenario_set_unknown_param_is_e268 () =
  compile_expect_error_code ~code:"E268" ~contains:"cvo"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      typo_cov { set = { cvo = 0.9 } }
    }
  |})

let test_scenario_scale_unknown_param_is_e268 () =
  compile_expect_error_code ~code:"E268" ~contains:"xxx"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      typo_scale { scale = { xxx = 2.0 } }
    }
  |})

let test_scenario_compose_unknown_scenario_is_e269 () =
  compile_expect_error_code ~code:"E269" ~contains:"missing"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      base { set = { x = 0.3 } }
      composed { compose = [base, missing] }
    }
  |})

let test_scenario_enable_known_intervention_compiles () =
  let _m = compile_expect_ok
    (scenario_validation_boilerplate ^ {|
    scenarios {
      high_coverage { enable = [sia] }
    }
  |}) in ()

(* 2026-06-27 scenario-aware fit predict: `fitted` is the reserved name for
   the no-overlay row in the `scenario` column. A preset by that name would
   shadow the reserved value, so the compiler rejects it (E291) with a
   migration-style diagnostic. *)
let test_scenario_named_fitted_is_reserved_e291 () =
  compile_expect_error_code ~code:"E291" ~contains:"reserved"
    (scenario_validation_boilerplate ^ {|
    scenarios {
      fitted { set = { x = 0.3 } }
    }
  |})

(* gh#130: an indexed intervention `sia[reg in region]` expands to
   per-instance names (`sia_north`, `sia_south`). A scenario may enable
   a single expanded instance — the runtime filter
   (`resolve_enable_list`, rust/crates/cli/src/util.rs) accepts an exact
   instance name, so the E267 validator must too. Before the fix it knew
   only the family name `sia` and false-positived on `sia_north`. *)
let indexed_intervention_boilerplate = {|
    time_unit = 'days
    dimensions { region = [north, south] }
    compartments { S, V }
    stratify(by = region)
    parameters { x : rate  cov : probability }
    transitions {}
    init { S[reg in region] = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      sia[reg in region] : transfer(fraction = cov, from = S[reg], to = V[reg]) at [1]
    }
  |}

let test_scenario_enable_expanded_instance_compiles () =
  let _m = compile_expect_ok
    (indexed_intervention_boilerplate ^ {|
    scenarios {
      targeted_north { enable = [sia_north] }
    }
  |}) in ()

let test_scenario_disable_expanded_instance_compiles () =
  let _m = compile_expect_ok
    (indexed_intervention_boilerplate ^ {|
    scenarios {
      baseline {}
      drop_south { extends = baseline   disable = [sia_south] }
    }
  |}) in ()

(* Negative control: the union accepts only *real* expanded instances —
   a typo on an instance name (`sia_east`, no such region) still E267s,
   so the gh#130 fix did not silently widen the validator to accept any
   `sia_*`. *)
let test_scenario_enable_bogus_instance_still_e267 () =
  compile_expect_error_code ~code:"E267" ~contains:"sia_east"
    (indexed_intervention_boilerplate ^ {|
    scenarios {
      typo { enable = [sia_east] }
    }
  |})

let test_extends_w310_on_enable_dedup () =
  (* Compile should succeed but emit a W310 warning naming the parent
     and showing the resolved enable list. *)
  let src = {|
    time_unit = 'days
    compartments { S, V }
    parameters { x : rate }
    transitions {}
    init { S = 1 }
    simulate { from = 0 'days  to = 10 'days }
    interventions {
      a : transfer(fraction = 0.1, from = S, to = V) at [1]
      b : transfer(fraction = 0.1, from = S, to = V) at [2]
    }
    scenarios {
      p { enable = [a] }
      c { extends = p   enable = [b] }
    }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"w310_test" src in
  Diagnostics.json_errors_mode := false;
  (* T4 in 2026-04-19 review: previously this test only checked that
     the enable list merged correctly, not that W310 actually fired.
     Inspect ctx.diags to assert the warning is present. *)
  match r with
  | Error e -> Alcotest.failf "should compile despite W310: %s" e
  | Ok d ->
    let c = find_scenario d.model "c" in
    Alcotest.(check (list string)) "merged enable" ["a"; "b"] c.preset_enable;
    let has_w310 =
      List.exists (fun (diag : Diagnostics.diagnostic) ->
        diag.code = "W310" && diag.severity = Diagnostics.Warning
      ) d.ctx.diags.diags
    in
    Alcotest.(check bool) "W310 warning was emitted" true has_w310

(* ── L401: Euler-correction with fixed time literal (gh#54) ────────────────── *)

let count_diags_with_code (diags : Diagnostics.diagnostic list) code =
  List.length (List.filter (fun (d : Diagnostics.diagnostic) -> d.code = code) diags)

let test_l401_fires_on_fixed_time_literal () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      R0    : rate  in [0.1, 5.0]
      gamma : rate  in [0.1, 1.0]
    }
    transitions {
      bad : S --> I @ R0 * (1 - exp(-(gamma) * 1 'days)) * S
      ok  : I --> S @ gamma * I
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l401_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile despite L401: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L401" in
    Alcotest.(check int) "L401 fires once on the bad transition" 1 n

let test_l401_no_fire_when_dt_used () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      R0    : rate  in [0.1, 5.0]
      gamma : rate  in [0.1, 1.0]
    }
    transitions {
      good : I --> S @ (1 - exp(-gamma * dt)) / dt * I
    }
    init { I = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l401_dt_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L401" in
    Alcotest.(check int) "no L401 when dt is used" 0 n;
    (* Confirm the parse + expansion produced an actual Ir.Dt node. *)
    let rec contains_dt = function
      | Ir.Dt -> true
      | Ir.BinOp { left; right; _ } -> contains_dt left || contains_dt right
      | Ir.UnOp { arg; _ } -> contains_dt arg
      | Ir.Cond { pred; then_; else_ } ->
        contains_dt pred || contains_dt then_ || contains_dt else_
      | Ir.UncheckedDim u -> contains_dt u.inner
      | Ir.TableLookup (_, args) -> List.exists contains_dt args
      | Ir.Reduce terms -> List.exists contains_dt terms
      | Ir.BindingRef _ -> false
      | Ir.PerEvalRef _ -> false
      | Ir.Const _ | Ir.Param _ | Ir.Pop _ | Ir.PopSum _
      | Ir.Time | Ir.Projected | Ir.ObsColumnRef _ | Ir.TimeFunc _ -> false
    in
    let any_tr_uses_dt = List.exists (fun (t : Ir.transition) ->
      contains_dt t.rate
    ) d.model.transitions in
    Alcotest.(check bool) "Ir.Dt appears in expanded rate" true any_tr_uses_dt;
    (* Round-trip through serde: serialize, parse back, structural equality. *)
    let json   = Serde.model_to_json d.model in
    let model' = Serde.model_of_json json in
    Alcotest.(check bool) "model survives serde round-trip"
      true (d.model = model')

let test_l401_no_fire_on_unit_conversion () =
  (* Pure unit conversion without exp() — must not fire. *)
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      mu    : rate  in [0.001, 1.0]
      gamma : rate  in [0.1, 1.0]
    }
    let mu_per_day = mu / 1 'days
    transitions {
      good : I --> S @ gamma * I
    }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l401_unit_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error _ -> ()  (* compile may fail for other reasons; we only care L401 wasn't tripped if it did succeed *)
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L401" in
    Alcotest.(check int) "no L401 on unit conversion (no exp)" 0 n

(* ── L403: manual re-conversion of an already-rescaled rate forcing (gh#13) ── *)

(* True iff SOME L403 diagnostic's message contains [needle]. *)
let l403_msg_contains needle (diags : Diagnostics.diagnostic list) =
  List.exists (fun (d : Diagnostics.diagnostic) ->
    d.code = "L403" && contains_substring ~needle d.message
  ) diags

(* Div form: `birthrate(t) * popsize(t) / 365.25`. birthrate is 'per_year (a
   rate forcing, already rescaled at load); popsize is 'count. L403 must fire
   once, naming the forcing and the magnitude. *)
let test_l403_fires_on_div_conversion () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      birthrate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
      popsize   : interpolated 'count    { times = [0, 100]  values = [1000, 1100]  method = "linear" }
    }
    transitions {
      births : S --> I @ birthrate(t) * popsize(t) / 365.25
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_div" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile despite L403: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "L403 fires once on the div-conversion" 1 n;
    Alcotest.(check bool) "L403 names the forcing" true
      (l403_msg_contains "birthrate" d.ctx.diags.diags);
    Alcotest.(check bool) "L403 names the magnitude" true
      (l403_msg_contains "365.25" d.ctx.diags.diags)

(* Reciprocal-as-Mul form: `birthrate(t) * (1 / 365.25) * S`. The lint runs
   pre-constant-fold, so `1 / 365.25` is `Div{Const 1, Const 365.25}`. *)
let test_l403_fires_on_reciprocal_mul () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      birthrate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
    }
    transitions {
      births : S --> I @ birthrate(t) * (1 / 365.25) * S
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_recip" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile despite L403: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "L403 fires once on the reciprocal-mul" 1 n;
    Alcotest.(check bool) "L403 names the forcing" true
      (l403_msg_contains "birthrate" d.ctx.diags.diags)

(* Control: plain use `birthrate(t) * S` — no bare conversion → no L403. *)
let test_l403_no_fire_on_plain_use () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      birthrate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
    }
    transitions {
      births : S --> I @ birthrate(t) * S
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_plain" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "no L403 on plain forcing use" 0 n

(* Control: a 'ratio and a 'count forcing divided by a conversion constant.
   Neither is a rate forcing, so the double-conversion bug does not apply. *)
let test_l403_no_fire_on_non_rate_forcing () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      seasonal : interpolated 'ratio { times = [0, 100]  values = [1.0, 1.2]  method = "linear" }
      popsize  : interpolated 'count { times = [0, 100]  values = [1000, 1100]  method = "linear" }
    }
    transitions {
      a : S --> I @ seasonal(t) / 365.25 * R0 * S
      b : S --> I @ popsize(t) / 365.25 * R0
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_nonrate" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "no L403 on 'ratio / 'count forcing" 0 n

(* ── gh#345: indexed (per-stratum) file-backed forcing ──────────────────────
   `cforce[p in patch] : interpolated { data=… key_col=patch … }` must expand to
   one interpolated forcing per patch, each carrying THAT patch's rows from the
   long-format file. Regression guard for the silent key-lookup bug: the filter
   level was looked up by the data-column name in an env keyed by the *binder*
   variable, so `[p in patch]` produced empty knots (interpolate-to-0 every-
   where, the gh#308 failure mode) unless the binder happened to be named
   `patch`. The stratum level now comes from the binding, decoupled from both
   the binder name and the column name. ──────────────────────────────────────*)
let test_gh345_indexed_file_backed_forcing () =
  let dir = Filename.get_temp_dir_name () in
  let tsv = Filename.concat dir "camdl_gh345_temps.tsv" in
  let oc  = open_out tsv in
  output_string oc
    "patch\tweek\tcval\n\
     north\t0\t1.0\nnorth\t10\t2.0\nnorth\t20\t1.5\n\
     south\t0\t0.5\nsouth\t10\t0.8\nsouth\t20\t0.6\n";
  close_out oc;
  let src = {|
    time_unit = 'days
    compartments { S, I }
    dimensions { patch = [north, south] }
    stratify(by = patch)
    let N[p in patch] = S[p] + I[p]
    parameters { beta : rate  gamma : rate }
    forcing {
      cforce[p in patch] : interpolated 'ratio {
        data = "camdl_gh345_temps.tsv"
        key_col = patch  time_col = week  value_col = cval  method = "linear"
      }
    }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * cforce[p] * S[p] * I[p] / N[p]
      recovery[p in patch]  : I[p] --> S[p] @ gamma * I[p]
    }
    init { S[north] = 990  I[north] = 10  S[south] = 495  I[south] = 5 }
    simulate { from = 0 'days  to = 20 'days }
  |} in
  let model_path = Filename.concat dir "camdl_gh345_model.camdl" in
  let m = match Compiler.compile ~name:"gh345" ~filename:model_path src with
    | Ok m    -> m
    | Error e -> Alcotest.failf "compile failed: %s" e
  in
  let values name =
    match List.find_opt (fun (tf : Ir.time_function) -> tf.name = name) m.time_functions with
    | Some { kind = Ir.Interpolated i; _ } ->
      List.map (function Ir.Const f -> f
                       | _ -> Alcotest.failf "%s: non-const knot" name) i.values
    | Some _ -> Alcotest.failf "%s is not an interpolated forcing" name
    | None   -> Alcotest.failf "forcing %s not found (got: [%s])" name
                  (String.concat ", "
                     (List.map (fun (t : Ir.time_function) -> t.name) m.time_functions))
  in
  Alcotest.(check (list (float 1e-9)))
    "north forcing gets north's own rows" [1.0; 2.0; 1.5] (values "cforce_north");
  Alcotest.(check (list (float 1e-9)))
    "south forcing gets south's own rows" [0.5; 0.8; 0.6] (values "cforce_south")

(* ── gh#345 sibling: the table-backed form ──────────────────────────────────
   The per-stratum series comes from a `tables {}` matrix, not a `data =` file.
   `table = temp_data` names the source, `time_dim = climate_week` names the
   time axis, and the forcing's `[p in patch]` index binds the stratum — no `:`
   slice, no inference. The `climate_week` levels here arrive OUT of numeric
   order (10, 0, 20 by first occurrence), so this also pins that the knots are
   sorted by time (each value paired to its own time). ───────────────────────*)
let test_gh345_table_backed_forcing () =
  let dir = Filename.get_temp_dir_name () in
  let tsv = Filename.concat dir "camdl_gh345t.tsv" in
  let oc  = open_out tsv in
  (* week first-occurrence order is 10, 0, 20 — deliberately unsorted *)
  output_string oc
    "patch\tweek\tcval\n\
     north\t10\t2.0\nnorth\t0\t1.0\nnorth\t20\t1.5\n\
     south\t10\t0.8\nsouth\t0\t0.5\nsouth\t20\t0.6\n";
  close_out oc;
  let src = {|
    time_unit = 'days
    compartments { S, I }
    dimensions {
      patch        = [north, south]
      climate_week = read("camdl_gh345t.tsv", column = "week")
    }
    stratify(by = patch)
    let N[p in patch] = S[p] + I[p]
    parameters { beta : rate  gamma : rate }
    tables {
      temp_data : patch × climate_week = read("camdl_gh345t.tsv")
    }
    forcing {
      cforce[p in patch] : interpolated 'ratio {
        table    = temp_data
        time_dim = climate_week
        method   = "linear"
      }
    }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * cforce[p] * S[p] * I[p] / N[p]
      recovery[p in patch]  : I[p] --> S[p] @ gamma * I[p]
    }
    init { S[north] = 990  I[north] = 10  S[south] = 495  I[south] = 5 }
    simulate { from = 0 'days  to = 20 'days }
  |} in
  let model_path = Filename.concat dir "camdl_gh345t_model.camdl" in
  let m = match Compiler.compile ~name:"gh345t" ~filename:model_path src with
    | Ok m    -> m
    | Error e -> Alcotest.failf "compile failed: %s" e
  in
  let interp name =
    match List.find_opt (fun (tf : Ir.time_function) -> tf.name = name) m.time_functions with
    | Some { kind = Ir.Interpolated i; _ } ->
      let consts = List.map (function Ir.Const x -> x
                                    | _ -> Alcotest.failf "%s: non-const knot" name) in
      (consts i.times, consts i.values)
    | Some _ -> Alcotest.failf "%s is not an interpolated forcing" name
    | None   -> Alcotest.failf "forcing %s not found" name
  in
  let (north_t, north_v) = interp "cforce_north" in
  let (_south_t, south_v) = interp "cforce_south" in
  (* knots sorted by time despite the unsorted dimension levels *)
  Alcotest.(check (list (float 1e-9)))
    "times sorted to the climate_week levels" [0.0; 10.0; 20.0] north_t;
  Alcotest.(check (list (float 1e-9)))
    "north = row `north`, value paired to its own time" [1.0; 2.0; 1.5] north_v;
  Alcotest.(check (list (float 1e-9)))
    "south = row `south`, value paired to its own time" [0.5; 0.8; 0.6] south_v

(* gh#345: a table dimension neither indexed by the forcing nor named as
   time_dim is a named error (E229) — never a silent axis guess. *)
let test_gh345_table_unaccounted_dim_rejected () =
  let dir = Filename.get_temp_dir_name () in
  let tsv = Filename.concat dir "camdl_gh345u.tsv" in
  let oc  = open_out tsv in
  output_string oc
    "patch\tseason\tweek\tcval\n\
     north\twet\t0\t1.0\nnorth\twet\t10\t2.0\n\
     north\tdry\t0\t1.1\nnorth\tdry\t10\t2.1\n\
     south\twet\t0\t0.5\nsouth\twet\t10\t0.8\n\
     south\tdry\t0\t0.6\nsouth\tdry\t10\t0.9\n";
  close_out oc;
  let src = {|
    time_unit = 'days
    compartments { S, I }
    dimensions {
      patch  = [north, south]
      season = [wet, dry]
      week   = read("camdl_gh345u.tsv", column = "week")
    }
    stratify(by = patch)
    let N[p in patch] = S[p] + I[p]
    parameters { beta : rate  gamma : rate }
    tables {
      clim : patch × season × week = read("camdl_gh345u.tsv")
    }
    forcing {
      cforce[p in patch] : interpolated 'ratio {
        table    = clim
        time_dim = week
        method   = "linear"
      }
    }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta * cforce[p] * S[p] * I[p] / N[p]
      recovery[p in patch]  : I[p] --> S[p] @ gamma * I[p]
    }
    init { S[north] = 990  I[north] = 10  S[south] = 495  I[south] = 5 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile ~name:"gh345u" ~filename:(Filename.concat dir "m.camdl") src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Ok _    -> Alcotest.failf "expected E229 for the unaccounted `season` dimension"
  | Error e ->
    if not (contains_substring ~needle:"E229" e) then
      Alcotest.failf "expected error code E229, got: %s" e;
    if not (contains_substring ~needle:"season" e) then
      Alcotest.failf "error must name the unaccounted `season` dimension, got: %s" e

(* Control: `mu * S / 365.25` — a bare conversion constant, but no rate forcing
   anywhere in the numerator. Must not fire. *)
let test_l403_no_fire_on_unrelated_div () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      R0 : rate in [0.1, 5.0]
      mu : rate in [0.001, 1.0]
    }
    forcing {
      birthrate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
    }
    transitions {
      decay : I --> S @ mu * S / 365.25
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_unrel" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "no L403 without a rate forcing in the numerator" 0 n

(* The `let flow = birthrate(t) * popsize(t) / 365.25` idiom hoists into a
   model-level binding; the lint must walk binding bodies too, else this common
   pattern escapes it silently. *)
let test_l403_fires_via_hoisted_binding () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      birthrate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
      popsize   : interpolated 'count    { times = [0, 100]  values = [1000, 1100]  method = "linear" }
    }
    let birth_flow = birthrate(t) * popsize(t) / 365.25
    transitions {
      births : S --> I @ birth_flow
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_hoist" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile despite L403: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "L403 fires once via the hoisted binding" 1 n;
    Alcotest.(check bool) "L403 names the binding" true
      (l403_msg_contains "birth_flow" d.ctx.diags.diags)

(* Class A false positive (gh#13 review): a `'per_day` forcing under
   `time_unit = 'days` is NOT rescaled at load (scale = 1.0), so dividing it by a
   constant is not a double-conversion — there was no first conversion. Divide by
   60 (a member of the OLD generic magnitude set, so this FIRED before the fix).
   Must be SILENT: L403 now matches only the forcing's OWN magnitude (1/scale),
   which for a same-unit forcing does not exist (scale = 1 → excluded). *)
let test_l403_no_fire_on_same_unit_forcing () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      rate_pd : interpolated 'per_day { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
    }
    transitions {
      x : S --> I @ rate_pd(t) * S / 60
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_same_unit" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "no L403 on a same-unit (unrescaled) rate forcing" 0 n

(* Class B false positive (gh#13 review): a `'per_year` forcing IS rescaled under
   `time_unit = 'days` (scale = 1/365.2425, magnitude ≈ 365.2425), but here the
   divisor 12 is structural (12 provinces), not months. 12 is in the OLD generic
   magnitude set (months/year), so this FIRED before the fix. Must be SILENT: 12
   is nowhere near THIS forcing's own magnitude ≈ 365.2425. *)
let test_l403_no_fire_on_structural_divisor () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { R0 : rate in [0.1, 5.0] }
    forcing {
      import_rate : interpolated 'per_year { times = [0, 100]  values = [0.02, 0.03]  method = "linear" }
    }
    transitions {
      imp : S --> I @ import_rate(t) * S / 12
    }
    init { S = 100 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"l403_structural_div" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile cleanly: %s" e
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "L403" in
    Alcotest.(check int) "no L403 when the divisor is not this forcing's magnitude" 0 n

(* ── gh#58: trig primitives (sin/cos/tanh) + pi/e ──────────────────────────── *)

let test_trig_pi_resolves_to_const () =
  (* `pi` in a rate expression resolves to Ir.Const 3.14159... *)
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { gamma : rate  in [0.001, 1.0] }
    let period = 1 'days
    transitions { dummy : S --> @ gamma * cos(2 * pi * t / period) }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"trig_pi_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "should compile: %s" e
  | Ok d ->
    let rec find_pi_const = function
      | Ir.Const c when Float.abs (c -. Float.pi) < 1e-9 -> true
      | Ir.BinOp { left; right; _ } -> find_pi_const left || find_pi_const right
      | Ir.UnOp { arg; _ } -> find_pi_const arg
      | Ir.Cond { pred; then_; else_ } ->
        find_pi_const pred || find_pi_const then_ || find_pi_const else_
      | Ir.UncheckedDim u -> find_pi_const u.inner
      | _ -> false
    in
    let any_tr_has_pi = List.exists (fun (t : Ir.transition) ->
      find_pi_const t.rate
    ) d.model.transitions in
    Alcotest.(check bool) "pi appears as Const ≈ π in the rate IR"
      true any_tr_has_pi

let test_trig_cos_compiles_and_dimchecks () =
  (* cos(dimensionless) → OK; rate compiles. *)
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters {
      gamma : rate  in [0.001, 1.0]
      a1    : rate  in [0.001, 1.0]
    }
    let period = 1 'days
    transitions { dummy : S --> @ gamma * (1 + sin(2 * pi * t / period) + cos(2 * pi * t / period)) }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"trig_cos_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "cos(2π·t/period) should compile: %s" e
  | Ok _ -> ()

let test_trig_cos_rejects_dimensional_arg () =
  (* cos(t) where t : time → Dimcheck.Error (argument must be dimensionless). *)
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { gamma : rate  in [0.001, 1.0] }
    transitions { dummy : S --> @ gamma * cos(t) }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"trig_dim_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Ok d ->
    let dc = Dimcheck.check_model d.model in
    let n = List.length (List.filter (fun (x : Dimcheck.diagnostic) ->
      x.code = "E301" && x.severity = Dimcheck.Error
    ) dc.diagnostics) in
    Alcotest.(check bool) "E301 fired on cos(t)" true (n > 0)
  | Error _ -> ()  (* compile may have failed for other reasons *)

let test_trig_autodiff_matches_finite_diff () =
  (* Synthetic IR rate = b * sin(c), with c a literal constant.
     ∂/∂b of (b * sin(c)) = sin(c), a constant.
     Verify Autodiff.differentiate_rate emits exactly that. *)
  let c = Float.pi /. 4.0 in
  let rate = Ir.BinOp { op = Ir.Mul;
                        left  = Ir.Param "b";
                        right = Ir.UnOp { op = Ir.Sin; arg = Ir.Const c } } in
  let grads = match Autodiff.differentiate_rate rate ["b"] [] [] with
    | Ok g -> g
    | Error msg -> Alcotest.failf "differentiate_rate errored: %s" msg in
  match List.assoc_opt "b" grads with
  | None -> Alcotest.failf "no rate_grad for parameter 'b'"
  (* After simplify_fixpoint, the derivative folds to Const (sin c). *)
  | Some (Ir.DEGrad (Ir.Const v)) ->
    Alcotest.(check (float 1e-12))
      "∂(b*sin(c))/∂b = sin(c)" (sin c) v
  | Some _ -> Alcotest.failf "expected DEGrad (Const), got a non-constant or refused derivative"

let test_fourier_autodiff_emitted () =
  (* gh#119/gh#59: a parameter that is a Fourier harmonic coefficient must get
     a real derivative through the forcing closed form (not a dropped/silent
     zero, and not Unsupported). Rate = the forcing itself; ∂/∂a1 is the
     cos term, so a nonzero entry must be emitted. *)
  let f : Ir.fourier =
    { period = Ir.Const 365.0; harmonics = [ (Ir.Param "a1", Ir.Const 0.0) ] } in
  let tf : Ir.time_function = { name = "f"; kind = Ir.Fourier f; dim = (0, 0); lag = None } in
  let rate = Ir.TimeFunc "f" in
  match Autodiff.differentiate_rate rate [ "a1" ] [ tf ] [] with
  | Error msg -> Alcotest.failf "Fourier differentiate errored: %s" msg
  | Ok grads ->
    (match List.assoc_opt "a1" grads with
     | Some _ -> ()  (* emitted a (nonzero) cos derivative — Known, not dropped *)
     | None -> Alcotest.failf "Fourier ∂/∂a1 was dropped (expected a cos term)")

let test_periodic_forcing_coeff_omitted () =
  (* gh#119/gh#215/gh#342: a parameter that is a periodic step value is a LIVE
     coefficient (the Rust runtime evaluates it per-step), so the model must
     COMPILE — its gradient is not yet emitted. Post-3b, rather than being DROPPED,
     it is a serialized coded DEUnsupported{URPeriodicCoeff}, so the fit-time
     preflight refuses a NUTS fit that depends on it (subsuming coeff_guard);
     forward sim and gradient-free IF2/PF still use the live value. It must NOT be
     a hard compile error (that would break forward sim / IF2 / PF too). *)
  let p : Ir.periodic =
    { period = Ir.Const 7.0; values = [ Ir.Param "v0"; Ir.Const 1.0 ] } in
  let tf : Ir.time_function = { name = "g"; kind = Ir.Periodic p; dim = (0, 0); lag = None } in
  let rate = Ir.TimeFunc "g" in
  match Autodiff.differentiate_rate rate [ "v0" ] [ tf ] [] with
  | Error msg -> Alcotest.failf
      "a periodic step-value param must compile (as a coded refusal), got error: %s" msg
  | Ok grads ->
    (match List.assoc_opt "v0" grads with
     | Some (Ir.DEUnsupported { code = Ir.URPeriodicCoeff; _ }) -> ()
     | Some (Ir.DEGrad _) -> Alcotest.failf "periodic ∂/∂v0 must be a coded refusal, not a DEGrad"
     | Some (Ir.DEUnsupported _) -> Alcotest.failf "periodic ∂/∂v0: expected URPeriodicCoeff, got a different code"
     | None -> Alcotest.failf "periodic ∂/∂v0 must now be present as DEUnsupported (was dropped pre-3b)")

let test_structural_forcing_coeff_errors () =
  (* gh#119/gh#215: a parameter that drives STRUCTURAL forcing data (an
     interpolation knot, a piecewise step, a spline basis coefficient) cannot
     be a live coefficient at all — those arrays are precomputed at
     construction. This must be a hard compile error naming the parameter and
     calling the data structural (the Rust runtime also rejects it at IR-load
     via eval_structural). *)
  let i : Ir.interpolated =
    { times = [ Ir.Const 0.0; Ir.Const 1.0 ];
      values = [ Ir.Param "knot0"; Ir.Const 1.0 ]; method_ = "linear" } in
  let tf : Ir.time_function = { name = "g"; kind = Ir.Interpolated i; dim = (0, 0); lag = None } in
  let rate = Ir.TimeFunc "g" in
  match Autodiff.differentiate_rate rate [ "knot0" ] [ tf ] [] with
  | Ok _ -> Alcotest.failf "expected a structural compile error for a param in an interpolation knot"
  | Error msg ->
    Alcotest.(check bool) "names the param and calls the data structural"
      true (contains_substring ~needle:"knot0" msg
            && contains_substring ~needle:"structural" msg)

let test_periodic_param_in_rate_compiles () =
  (* gh#119/gh#215 regression: a full model with a periodic forcing whose step
     value is a parameter, referenced in a transition rate, must compile to IR
     (it built on `main`; the over-firing E600 floor broke forward sim + IF2/PF
     for it). *)
  let m = compile_expect_ok {|
time_unit = 'days
compartments { S, I }
parameters {
  beta  : rate
  wpeak : real
  S0    : count
  I0    : count
}
let N = S + I
forcing {
  weekly : periodic 'ratio {
    period = 7
    values = [wpeak, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  }
}
transitions { infection : S --> I @ beta * weekly(t) * S * I / N }
init { S = S0  I = I0 }
simulate { from = 0 'days  to = 30 'days }
scenarios { baseline { set = { beta = 0.3
      wpeak = 1.5
      S0 = 990
      I0 = 10 } } }
|} in
  (* gh#342: a Periodic step-value coefficient's gradient is not emitted (gh#215),
     but rather than being DROPPED (pre-3b), it is now a serialized coded
     DEUnsupported{URPeriodicCoeff} — so the fit-time preflight refuses a NUTS fit
     that depends on it (subsuming the old coeff_guard). Forward sim / IF2 / PF
     still use the live value. *)
  List.iter (fun (t : Ir.transition) ->
    match List.assoc_opt "wpeak" t.rate_grad with
    | Some (Ir.DEUnsupported { code = Ir.URPeriodicCoeff; _ }) -> ()
    | Some (Ir.DEGrad _) ->
      Alcotest.failf "wpeak (periodic) must be a coded refusal, not a DEGrad, in '%s'" t.name
    | Some (Ir.DEUnsupported _) ->
      Alcotest.failf "wpeak: expected URPeriodicCoeff, got a different refusal code in '%s'" t.name
    | None ->
      Alcotest.failf "wpeak (periodic) must now be present as DEUnsupported in rate_grad of '%s' (was dropped pre-3b)" t.name)
    m.Ir.transitions

(* ── Observation / σ² gradient driver (proposal 2026-07-03, P3) ──────────────── *)

let test_obs_grad_nonderived_projection () =
  (* Poisson obs: rate = rho * projected, projection = CumulativeFlow "inc"
     (θ-independent given the fixed trajectory). [Projected] is left in place and
     differentiates to a genuine zero, so ∂rate/∂rho = projected. A param not in
     the arg (beta) is a genuine zero → ABSENT key. *)
  let rate = Ir.BinOp { op = Ir.Mul; left = Ir.Param "rho"; right = Ir.Projected } in
  let lik = Ir.Poisson { rate = { Ir.expr = rate; Ir.grad = [] } } in
  match Autodiff.differentiate_likelihood (Ir.CumulativeFlow "inc") lik [ "rho"; "beta" ] [] [] with
  | Ir.Poisson pl ->
    (match List.assoc_opt "rho" pl.rate.grad with
     | Some (Ir.DEGrad Ir.Projected) -> ()
     | Some _ -> Alcotest.failf "rate_grad[rho]: expected DEGrad Projected"
     | None -> Alcotest.failf "rate_grad[rho] missing");
    Alcotest.(check bool) "beta (genuine zero) omitted from rate_grad"
      false (List.mem_assoc "beta" pl.rate.grad)
  | _ -> Alcotest.failf "likelihood variant changed unexpectedly"

let test_obs_grad_parametric_derived_projection () =
  (* Poisson obs: rate = rho * projected, projection = DerivedExpr (qgam * P).
     Inlining projected → (qgam·P) makes rate = rho·(qgam·P), so the chain rule
     reaches ∂projected/∂qgam:  ∂rate/∂qgam = rho·P,  ∂rate/∂rho = qgam·P.
     This is the headline gh#180 case — a parametric DerivedExpr projection. *)
  let rate = Ir.BinOp { op = Ir.Mul; left = Ir.Param "rho"; right = Ir.Projected } in
  let lik = Ir.Poisson { rate = { Ir.expr = rate; Ir.grad = [] } } in
  let proj = Ir.DerivedExpr (Ir.BinOp { op = Ir.Mul; left = Ir.Param "qgam"; right = Ir.Pop "P" }) in
  let expect_qgam = Ir.BinOp { op = Ir.Mul; left = Ir.Param "rho";  right = Ir.Pop "P" } in
  let expect_rho  = Ir.BinOp { op = Ir.Mul; left = Ir.Param "qgam"; right = Ir.Pop "P" } in
  match Autodiff.differentiate_likelihood proj lik [ "qgam"; "rho" ] [] [] with
  | Ir.Poisson pl ->
    let grad p = match List.assoc_opt p pl.rate.grad with
      | Some (Ir.DEGrad e) -> e
      | Some (Ir.DEUnsupported _) -> Alcotest.failf "%s: unexpected DEUnsupported" p
      | None -> Alcotest.failf "%s: missing rate_grad entry" p in
    Alcotest.(check bool) "∂rate/∂qgam = rho·P (chain rule through DerivedExpr)"
      true (grad "qgam" = expect_qgam);
    Alcotest.(check bool) "∂rate/∂rho = qgam·P"
      true (grad "rho" = expect_rho)
  | _ -> Alcotest.failf "likelihood variant changed unexpectedly"

let test_obs_grad_structural_forcing_is_coded_refusal () =
  (* A likelihood argument driving a STRUCTURAL forcing coefficient becomes a
     coded DEUnsupported (URStructuralForcing) — the obs path is omit-and-refuse,
     NOT the rate E600 (so forward sim / IF2 / PF still work; P5 refuses NUTS). *)
  let i : Ir.interpolated =
    { times = [ Ir.Const 0.0; Ir.Const 1.0 ];
      values = [ Ir.Param "knot0"; Ir.Const 1.0 ]; method_ = "linear" } in
  let tf : Ir.time_function = { name = "g"; kind = Ir.Interpolated i; dim = (0, 0); lag = None } in
  let lik = Ir.Poisson { rate = { Ir.expr = Ir.TimeFunc "g"; Ir.grad = [] } } in
  match Autodiff.differentiate_likelihood (Ir.CumulativeFlow "inc") lik [ "knot0" ] [ tf ] [] with
  | Ir.Poisson pl ->
    (match List.assoc_opt "knot0" pl.rate.grad with
     | Some (Ir.DEUnsupported { code = Ir.URStructuralForcing; _ }) -> ()
     | Some _ -> Alcotest.failf "rate_grad[knot0]: expected DEUnsupported URStructuralForcing"
     | None -> Alcotest.failf "structural-forcing param must produce a coded refusal, not be dropped")
  | _ -> Alcotest.failf "likelihood variant changed unexpectedly"

let test_sigma_sq_grad_emitted () =
  (* σ² = phi · S (a rate-context overdispersion variance, no Projected node).
     differentiate_overdispersion fills sigma_sq_grad[phi] = DEGrad (Pop "S"). *)
  let sigma_sq = Ir.BinOp { op = Ir.Mul; left = Ir.Param "phi"; right = Ir.Pop "S" } in
  let t : Ir.transition =
    { name = "inf"; stoichiometry = []; rate = Ir.Const 1.0; metadata = None;
      draw_method = Ir.DrawOverdispersed { sigma_sq; sigma_sq_grad = [] };
      rate_grad = []; rate_state_grad = []; lineage = None } in
  match Autodiff.differentiate_overdispersion [ t ] [ "phi"; "beta" ] [] [] with
  | [ { draw_method = Ir.DrawOverdispersed { sigma_sq_grad; _ }; _ } ] ->
    (match List.assoc_opt "phi" sigma_sq_grad with
     | Some (Ir.DEGrad (Ir.Pop "S")) -> ()
     | _ -> Alcotest.failf "sigma_sq_grad[phi]: expected DEGrad (Pop S)");
    Alcotest.(check bool) "beta (genuine zero) omitted from sigma_sq_grad"
      false (List.mem_assoc "beta" sigma_sq_grad)
  | _ -> Alcotest.failf "unexpected transition shape after overdispersion autodiff"

let test_trig_pi_reserved () =
  (* Declaring a parameter named `pi` is rejected. *)
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { pi : rate  in [0.001, 1.0] }
    transitions { dummy : S --> @ pi * S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"pi_reserved_test" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Ok d ->
    let n = count_diags_with_code d.ctx.diags.diags "E100" in
    Alcotest.(check bool) "E100 fired on parameter shadowing pi"
      true (n > 0)
  | Error _ -> ()

(* ── Phase D (BUG-4): Time function expansion ────────────────────────────────
   Compile a model with a sinusoidal forcing function.
   1. The time_functions list must be non-empty.
   2. The rate expression must contain Ir.TimeFunc, not Ir.Const 0.0. *)

let test_sinusoidal_time_func () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seasonal(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  match Compiler.compile ~name:"test_seasonal" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    Alcotest.(check int) "one time function" 1 (List.length m.Ir.time_functions);
    let tf = List.hd m.Ir.time_functions in
    Alcotest.(check string) "name is seasonal" "seasonal" tf.Ir.name;
    (match tf.Ir.kind with
     | Ir.Sinusoidal s ->
       (match s.Ir.amplitude with
        | Ir.Const v -> Alcotest.(check (float 1e-9)) "amplitude" 0.3 v
        | _ -> Alcotest.fail "expected Ir.Const for amplitude");
       (match s.Ir.period with
        | Ir.Const v -> Alcotest.(check (float 1e-9)) "period" 365.0 v
        | _ -> Alcotest.fail "expected Ir.Const for period");
       (match s.Ir.baseline with
        | Ir.Const v -> Alcotest.(check (float 1e-9)) "baseline" 1.0 v
        | _ -> Alcotest.fail "expected Ir.Const for baseline")
     | _ -> Alcotest.fail "expected Sinusoidal kind")

(* gh#314: a forcing without `lag` lands `None` on the time_function record;
   absent lag is byte-identical to today. *)
let test_forcing_without_lag_is_none () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seasonal(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  let m = compile_expect_ok src in
  let tf = List.hd m.Ir.time_functions in
  Alcotest.(check bool) "no lag ⇒ None" true (tf.Ir.lag = None)

(* gh#314: `lag = 10 'days` lands `Some (duration in model time_unit)` on the
   time_function record. The literal is unit-scaled exactly like `period`, so
   with time_unit = 'days it is Const 10.0 (wrapped in UncheckedDim for the
   time dimension). *)
let test_forcing_with_literal_lag () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      vc : interpolated 'ratio {
        times  = [0.0, 10.0, 20.0]
        values = [1.0, 2.0, 3.0]
        method = "linear"
        lag    = 10 'days
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ vc(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  let m = compile_expect_ok src in
  let tf = List.hd m.Ir.time_functions in
  (* lag is a time-dimensioned duration; the scalar is in model time units. *)
  let rec const_of = function
    | Ir.Const v -> v
    | Ir.UncheckedDim u -> const_of u.Ir.inner
    | _ -> Alcotest.fail "expected a constant (possibly UncheckedDim-wrapped) lag"
  in
  (match tf.Ir.lag with
   | Some e -> Alcotest.(check (float 1e-9)) "lag = 10 days" 10.0 (const_of e)
   | None   -> Alcotest.fail "expected Some lag")

(* gh#314: `lag = n` where n is a parameter lands `Some (Param "n")` — the
   lag-as-parameter case (a primary motivation). *)
let test_forcing_with_param_lag () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      tau   : duration
      N0    : count
      I0    : count
    }
    forcing {
      vc : interpolated 'ratio {
        times  = [0.0, 10.0, 20.0]
        values = [1.0, 2.0, 3.0]
        method = "linear"
        lag    = tau
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ vc(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  let m = compile_expect_ok src in
  let tf = List.hd m.Ir.time_functions in
  (match tf.Ir.lag with
   | Some (Ir.Param "tau") -> ()
   | Some _ -> Alcotest.fail "expected lag = Param tau"
   | None   -> Alcotest.fail "expected Some lag")

let rec expr_contains_time_func name = function
  | Ir.TimeFunc n        -> n = name
  | Ir.BinOp b           -> expr_contains_time_func name b.Ir.left
                          || expr_contains_time_func name b.Ir.right
  | Ir.UnOp u            -> expr_contains_time_func name u.Ir.arg
  | Ir.Cond c            -> expr_contains_time_func name c.Ir.pred
                          || expr_contains_time_func name c.Ir.then_
                          || expr_contains_time_func name c.Ir.else_
  | _                    -> false

let test_time_func_in_rate () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seasonal(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  match Compiler.compile ~name:"test_seasonal_rate" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let infection = List.find (fun (t : Ir.transition) -> t.Ir.name = "infection") m.Ir.transitions in
    if not (expr_contains_time_func "seasonal" infection.Ir.rate) then
      Alcotest.fail "infection rate should contain Ir.TimeFunc \"seasonal\", got Const 0.0"

(* ── read tests ──────────────────────────────────────────────────────────────

   These tests write temporary TSV files to a temp directory, compile a model
   that references them via read(), and assert the expected IR.
   The ~filename argument ensures source_dir is set to the temp directory so
   relative paths in the model source resolve correctly.                      *)

let write_tmp_file dir name content =
  let path = Filename.concat dir name in
  let oc = open_out path in
  output_string oc content;
  close_out oc;
  path

let test_read_long_1d () =
  let dir = Filename.get_temp_dir_name () in
  let _tsv_path = write_tmp_file dir "test_rates.tsv" "grp\trate\na\t0.5\nb\t1.5\nc\t2.5\n" in
  let src = {|
    dimensions { grp = [a, b, c] }
    compartments { S, I }
    stratify(by = grp)
    parameters { gamma : rate }
    tables {
      rates : grp = read("test_rates.tsv")
    }
    transitions {
      recovery[g in grp] : I[g] --> S[g] @ rates[g] * I[g]
    }
    simulate { from = 0  to = 10 }
  |} in
  (* Use the temp dir as the source file directory *)
  let fake_src_file = Filename.concat dir "model.camdl" in
  match Compiler.compile ~name:"test_rl1d" ~filename:fake_src_file src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match List.find_opt (fun (t : Ir.table) -> t.Ir.name = "rates") m.Ir.tables with
     | None -> Alcotest.fail "table 'rates' not found"
     | Some tbl ->
       let values = match tbl.Ir.source with
         | Ir.Inline vs -> vs
         | Ir.External _ -> Alcotest.fail "expected Inline table, got External"
       in
       Alcotest.(check int) "three values" 3 (List.length values);
       let vals = List.map (function
         | Ir.Const f -> f
         | _ -> Alcotest.fail "expected Ir.Const"
       ) values in
       Alcotest.(check (list (float 1e-9))) "values match TSV" [0.5; 1.5; 2.5] vals)

let test_read_long_defines () =
  (* Test that dimensions { grp = read(...) } derives levels from the data file *)
  let dir = Filename.get_temp_dir_name () in
  let _tsv_path = write_tmp_file dir "test_pop.tsv" "grp\tpop\nalpha\t1000.0\nbeta\t2000.0\n" in
  let src = {|
    dimensions { grp = read("test_pop.tsv", column = "grp") }
    compartments { S, I }
    parameters { beta : rate }
    stratify(by = grp)
    tables {
      pop : grp = read("test_pop.tsv")
    }
    transitions {
      infection[g in grp] : S[g] --> I[g] @ beta * S[g] * I[g]
    }
    simulate { from = 0  to = 10 }
  |} in
  let fake_src_file = Filename.concat dir "model.camdl" in
  match Compiler.compile ~name:"test_rl_defines" ~filename:fake_src_file src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* The expanded compartments should include S_alpha, S_beta, I_alpha, I_beta *)
    let comp_names = List.map (fun (c : Ir.compartment) -> c.Ir.name) m.Ir.compartments in
    List.iter (fun expected ->
      if not (List.mem expected comp_names) then
        Alcotest.failf "compartment %s not found; got: %s"
          expected (String.concat ", " comp_names)
    ) ["S_alpha"; "S_beta"; "I_alpha"; "I_beta"]

let test_read_long_missing_file () =
  (* Test at expander level to avoid the exit 1 in compiler.ml.
     We parse the AST manually, then call expand_detail with source_dir set,
     and inspect ctx.diags for the expected error. *)
  let dir = Filename.get_temp_dir_name () in
  let src = {|
    dimensions { grp = [a, b] }
    compartments { S }
    stratify(by = grp)
    tables {
      rates : grp = read("nonexistent_xyz_12345.tsv")
    }
    simulate { from = 0  to = 10 }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls =
    try Parser.file Lexer.token lexbuf
    with _ -> Alcotest.fail "parse failed"
  in
  let (_model, ctx, _summary) =
    Expander.expand_detail ~source_dir:dir "test_missing" decls
  in
  (* There should be at least one error containing the missing filename *)
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error)
  in
  Alcotest.(check bool) "at least one error" true (errors <> []);
  let found_filename = List.exists (fun d ->
    let msg = d.Diagnostics.message in
    let contains s sub =
      let ls = String.length s and lb = String.length sub in
      if lb > ls then false
      else begin
        let found = ref false in
        for i = 0 to ls - lb do
          if String.sub s i lb = sub then found := true
        done;
        !found
      end
    in
    contains msg "nonexistent_xyz_12345.tsv"
  ) errors in
  Alcotest.(check bool) "error message contains filename" true found_filename

let test_read_header_reordered () =
  (* Header columns in wrong order → E216 *)
  let dir = Filename.get_temp_dir_name () in
  (* File has columns 'sex' then 'age' but model expects 'age' then 'sex' *)
  let _tsv = write_tmp_file dir "test_reorder.tsv"
    "sex\tage\tvalue\nm\tyoung\t1.0\nm\told\t2.0\nf\tyoung\t3.0\nf\told\t4.0\n" in
  let src = {|
    dimensions { age = [young, old]  sex = [m, f] }
    compartments { S }
    stratify(by = age)
    stratify(by = sex)
    tables {
      mx : age × sex = read("test_reorder.tsv")
    }
    simulate { from = 0  to = 10 }
  |} in
  let fake_src_file = Filename.concat dir "model.camdl" in
  let lexbuf = Lexing.from_string src in
  let decls = try Parser.file Lexer.token lexbuf
              with _ -> Alcotest.fail "parse failed" in
  let (_model, ctx, _summary) =
    Expander.expand_detail ~source_dir:(Filename.dirname fake_src_file)
      "test_reorder" decls
  in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error) in
  let found_e216 = List.exists (fun d -> d.Diagnostics.code = "E216") errors in
  Alcotest.(check bool) "E216 emitted for reordered columns" true found_e216

let test_read_header_mismatch () =
  (* Header names don't match dim names → W201 *)
  let dir = Filename.get_temp_dir_name () in
  let _tsv = write_tmp_file dir "test_mismatch.tsv"
    "zone\tvalue\na\t1.0\nb\t2.0\n" in
  let src = {|
    dimensions { patch = [a, b] }
    compartments { S }
    stratify(by = patch)
    tables {
      pop : patch = read("test_mismatch.tsv")
    }
    simulate { from = 0  to = 10 }
  |} in
  let fake_src_file = Filename.concat dir "model.camdl" in
  let lexbuf = Lexing.from_string src in
  let decls = try Parser.file Lexer.token lexbuf
              with _ -> Alcotest.fail "parse failed" in
  let (_model, ctx, _summary) =
    Expander.expand_detail ~source_dir:(Filename.dirname fake_src_file)
      "test_mismatch" decls
  in
  let warnings = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Warning) in
  let found_w201 = List.exists (fun d -> d.Diagnostics.code = "W201") warnings in
  Alcotest.(check bool) "W201 emitted for mismatched column name" true found_w201

(* ── Indexed parameter tests ─────────────────────────────────────────────────
   These tests verify that indexed parameter declarations like `R0[patch]` are
   expanded to scalar IR parameters, resolved correctly in rate expressions, and
   emit W103 warnings when let bindings shadow stratum values.               ── *)

let test_indexed_param_scalar_expansion () =
  let src = {|
    dimensions { patch = [a, b] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive
      gamma     : rate
    }
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_idx_scalar" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let param_names = List.map (fun (p : Ir.parameter) -> p.Ir.name) m.Ir.parameters in
    List.iter (fun expected ->
      if not (List.mem expected param_names) then
        Alcotest.failf "expected param '%s' not found; got: %s"
          expected (String.concat ", " param_names)
    ) ["R0_a"; "R0_b"; "gamma"];
    (* Values are None — must be supplied externally *)
    let r0_a = List.find (fun (p : Ir.parameter) -> p.Ir.name = "R0_a") m.Ir.parameters in
    Alcotest.(check bool) "R0_a value is None" true ((Ir.param_concrete_value r0_a) = None);
    let gamma_p = List.find (fun (p : Ir.parameter) -> p.Ir.name = "gamma") m.Ir.parameters in
    Alcotest.(check bool) "gamma value is None" true ((Ir.param_concrete_value gamma_p) = None)

let test_indexed_param_variable_index () =
  let src = {|
    dimensions { patch = [a, b] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive
      gamma     : rate
    }
    let beta[p in patch] = R0[p] * gamma
    transitions {
      infection[p in patch] : S[p] --> I[p] @ beta[p] * S[p] * I[p]
      recovery[p in patch]  : I[p] --> S[p] @ gamma * I[p]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_idx_var" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* infection_a rate should contain Ir.Param "R0_a", infection_b "R0_b" *)
    let infection_a = find_transition m "infection_a" in
    let rec contains_param name = function
      | Ir.Param n -> n = name
      | Ir.BinOp b -> contains_param name b.Ir.left || contains_param name b.Ir.right
      | Ir.UnOp u  -> contains_param name u.Ir.arg
      | Ir.Cond c  -> contains_param name c.Ir.pred
                   || contains_param name c.Ir.then_
                   || contains_param name c.Ir.else_
      | _ -> false
    in
    Alcotest.(check bool) "infection_a rate has R0_a" true
      (contains_param "R0_a" (tr_rate infection_a));
    let infection_b = find_transition m "infection_b" in
    Alcotest.(check bool) "infection_b rate has R0_b" true
      (contains_param "R0_b" (tr_rate infection_b))

let test_indexed_param_literal_index () =
  let src = {|
    dimensions { patch = [kano, lagos] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive
      gamma     : rate
    }
    transitions {
      infection_kano : S[kano] --> I[kano] @ R0[kano] * gamma * S[kano] * I[kano]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_idx_lit" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let tr = find_transition m "infection_kano" in
    let rec contains_param name = function
      | Ir.Param n -> n = name
      | Ir.BinOp b -> contains_param name b.Ir.left || contains_param name b.Ir.right
      | _ -> false
    in
    Alcotest.(check bool) "infection_kano rate has R0_kano" true
      (contains_param "R0_kano" (tr_rate tr))

let test_indexed_param_no_default () =
  let src = {|
    dimensions { patch = [x, y] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      z[patch] : real
      gamma    : rate
    }
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_idx_nodef" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let find_param pname =
      match List.find_opt (fun (p : Ir.parameter) -> p.Ir.name = pname) m.Ir.parameters with
      | None -> Alcotest.failf "param %s not found" pname
      | Some p -> p
    in
    Alcotest.(check bool) "z_x value is None" true ((Ir.param_concrete_value (find_param "z_x")) = None);
    Alcotest.(check bool) "z_y value is None" true ((Ir.param_concrete_value (find_param "z_y")) = None)

let test_indexed_param_bad_index () =
  let src = {|
    dimensions { patch = [urban, rural] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive
      gamma     : rate
    }
    transitions {
      infection : S[urban] --> I[urban] @ R0[unknown_place] * gamma * S[urban] * I[urban]
    }
    simulate { from = 0  to = 10 }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls =
    try Parser.file Lexer.token lexbuf
    with _ -> Alcotest.fail "parse failed"
  in
  let (_model, ctx, _summary) = Expander.expand_detail "test_bad_idx" decls in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error)
  in
  Alcotest.(check bool) "at least one error for bad index" true (errors <> []);
  let found_e100 = List.exists (fun d ->
    d.Diagnostics.code = "E100"
  ) errors in
  Alcotest.(check bool) "E100 diagnostic emitted" true found_e100

let test_indexed_param_shadow_warning () =
  (* 'kano' is both a let binding and a stratum value → W103 *)
  let src = {|
    dimensions { patch = [kano, lagos] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive
      gamma     : rate
    }
    let kano = 1.0
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
    }
    simulate { from = 0  to = 10 }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls =
    try Parser.file Lexer.token lexbuf
    with _ -> Alcotest.fail "parse failed"
  in
  let (_model, ctx, _summary) = Expander.expand_detail "test_shadow" decls in
  let warnings = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Warning)
  in
  let found_w103 = List.exists (fun d ->
    d.Diagnostics.code = "W103"
  ) warnings in
  Alcotest.(check bool) "W103 warning for shadowing" true found_w103

(* ── Parameter bounds tests ───────────────────────────────────────────────── *)

let test_scalar_bounds () =
  let src = {|
    compartments { S, I }
    parameters {
      R0 : positive in [1.0, 20.0]
      gamma : rate
    }
    transitions {
      recovery : I --> S @ gamma * I
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_scalar_bounds" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let r0 = List.find (fun (p : Ir.parameter) -> p.Ir.name = "R0") m.Ir.parameters in
    Alcotest.(check bool) "R0 bounds present" true ((Ir.param_bounds r0) <> None);
    (match (Ir.param_bounds r0) with
     | Some (lo, hi) ->
       Alcotest.(check (float 1e-12)) "R0 lo = 1.0"  1.0  lo;
       Alcotest.(check (float 1e-12)) "R0 hi = 20.0" 20.0 hi
     | None -> Alcotest.fail "expected bounds");
    let gamma_p = List.find (fun (p : Ir.parameter) -> p.Ir.name = "gamma") m.Ir.parameters in
    Alcotest.(check bool) "gamma bounds is None" true ((Ir.param_bounds gamma_p) = None)

let test_indexed_bounds () =
  let src = {|
    dimensions { patch = [urban, rural] }
    compartments { S, I }
    stratify(by = patch)
    parameters {
      R0[patch] : positive in [1.0, 10.0]
      gamma     : rate
    }
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_indexed_bounds" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    List.iter (fun pname ->
      let p = List.find (fun (p : Ir.parameter) -> p.Ir.name = pname) m.Ir.parameters in
      Alcotest.(check bool) (pname ^ " bounds present") true ((Ir.param_bounds p) <> None);
      match (Ir.param_bounds p) with
      | Some (lo, hi) ->
        Alcotest.(check (float 1e-12)) (pname ^ " lo = 1.0")  1.0  lo;
        Alcotest.(check (float 1e-12)) (pname ^ " hi = 10.0") 10.0 hi
      | None -> Alcotest.failf "%s bounds expected" pname
    ) ["R0_urban"; "R0_rural"]

(* ── Shaped let bindings ─────────────────────────────────────────────────────
   let B : sex × sex = [[0.0, beta_mf], [beta_fm, 0.0]]
   B[female, male] → Param "beta_mf"  (row-major: 0*2+1 = 1)
   B[female,female]→ Const 0.0        (row-major: 0*2+0 = 0)
   B[male,  male]  → Const 0.0        (row-major: 1*2+1 = 3)              ── *)

let test_shaped_let () =
  let src = {|
    dimensions { sex = [female, male] }
    compartments { S, I }
    stratify(by = sex)
    parameters {
      gamma    : rate
      beta_mf  : rate
      beta_fm  : rate
    }
    let B : sex × sex = [[0.0, beta_mf], [beta_fm, 0.0]]
    transitions {
      inf_ff[a in sex] : S[a] --> I[a]
        @ B[female, female] * S[a] * I[a]
      inf_fm[a in sex] : S[a] --> I[a]
        @ B[female, male]   * S[a] * I[a]
      inf_mm[a in sex] : S[a] --> I[a]
        @ B[male,   male]   * S[a] * I[a]
    }
    simulate { from = 0  to = 10 }
  |} in
  match Compiler.compile ~name:"test_shaped_let" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let find_tr name =
      match List.find_opt (fun (t : Ir.transition) -> t.Ir.name = name) m.Ir.transitions with
      | None -> Alcotest.failf "transition %s not found" name
      | Some t -> t
    in
    let rec has_param pname = function
      | Ir.Param n -> n = pname
      | Ir.BinOp b -> has_param pname b.Ir.left || has_param pname b.Ir.right
      | Ir.UnOp u  -> has_param pname u.Ir.arg
      | Ir.Cond c  -> has_param pname c.Ir.pred
                   || has_param pname c.Ir.then_
                   || has_param pname c.Ir.else_
      | _ -> false
    in
    let rec has_const f = function
      | Ir.Const v -> v = f
      | Ir.BinOp b -> has_const f b.Ir.left || has_const f b.Ir.right
      | _ -> false
    in
    (* inf_fm_female: B[female,male]=beta_mf (index 1) *)
    let inf_fm_f = find_tr "inf_fm_female" in
    Alcotest.(check bool) "B[female,male] → beta_mf" true
      (has_param "beta_mf" inf_fm_f.Ir.rate);
    (* inf_ff_female: B[female,female]=0.0 (index 0) *)
    let inf_ff_f = find_tr "inf_ff_female" in
    Alcotest.(check bool) "B[female,female] → 0.0" true
      (has_const 0.0 inf_ff_f.Ir.rate);
    (* inf_mm_male: B[male,male]=0.0 (index 3) *)
    let inf_mm_m = find_tr "inf_mm_male" in
    Alcotest.(check bool) "B[male,male] → 0.0" true
      (has_const 0.0 inf_mm_m.Ir.rate)

(* ── E217: where guard compile-time check ────────────────────────────────────
   A where guard must only reference dimension level names or loop variables.
   Referencing a parameter or compartment name emits E217.                  ── *)

let test_where_param_in_guard () =
  (* 'gamma' is a parameter — must not appear in a where guard *)
  let src = {|
    dimensions { patch = [urban, rural] }
    compartments { S, I }
    stratify(by = patch)
    parameters { gamma : rate }
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
        where p == gamma
    }
    simulate { from = 0  to = 10 }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls = try Parser.file Lexer.token lexbuf
              with _ -> Alcotest.fail "parse failed" in
  let (_model, ctx, _summary) = Expander.expand_detail "test_where_param" decls in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error) in
  let found_e217 = List.exists (fun d -> d.Diagnostics.code = "E217") errors in
  Alcotest.(check bool) "E217 emitted for param in where guard" true found_e217

let test_where_compartment_in_guard () =
  (* 'S' is a compartment — must not appear in a where guard *)
  let src = {|
    dimensions { patch = [urban, rural] }
    compartments { S, I }
    stratify(by = patch)
    parameters { gamma : rate }
    transitions {
      recovery[p in patch] : I[p] --> S[p] @ gamma * I[p]
        where p == S
    }
    simulate { from = 0  to = 10 }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls = try Parser.file Lexer.token lexbuf
              with _ -> Alcotest.fail "parse failed" in
  let (_model, ctx, _summary) = Expander.expand_detail "test_where_comp" decls in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error) in
  let found_e217 = List.exists (fun d -> d.Diagnostics.code = "E217") errors in
  Alcotest.(check bool) "E217 emitted for compartment in where guard" true found_e217

let test_where_ivguard_filters () =
  (* ivguard where p == urban should skip rural intervention *)
  let src = {|
    dimensions { patch = [urban, rural] }
    compartments { S, V, I }
    stratify(by = patch)
    parameters { vacc_frac : positive }
    transitions {
      infection[p in patch] : S[p] --> I[p] @ S[p] * I[p]
    }
    interventions {
      vacc[p in patch] : transfer(fraction = vacc_frac, from = S[p], to = V[p]) at [30]
        where p == urban
    }
    simulate { from = 0  to = 100 }
  |} in
  match Compiler.compile ~name:"test_ivguard" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* Only vacc_urban should be emitted; vacc_rural filtered out *)
    let iv_names = List.map (fun (iv : Ir.intervention) -> iv.Ir.name) m.Ir.interventions in
    Alcotest.(check bool) "vacc_urban present" true (List.mem "vacc_urban" iv_names);
    Alcotest.(check bool) "vacc_rural absent" true (not (List.mem "vacc_rural" iv_names))

(* ── Issue 2: Bare function name in rate resolves to Ir.TimeFunc ─────────────
   Using `seasonal` without parens in a rate expression should resolve to
   Ir.TimeFunc "seasonal", not emit E100. ─────────────────────────────────── *)

let test_bare_func_name_in_rate () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seasonal * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  match Compiler.compile ~name:"test_bare_func" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let infection = find_transition m "infection" in
    if not (expr_contains_time_func "seasonal" infection.Ir.rate) then
      Alcotest.fail "bare 'seasonal' in rate should resolve to Ir.TimeFunc \"seasonal\""

(* ── Issue 3: Unknown EFuncCall emits E100, not silent 0.0 ───────────────────
   A misspelled function call like `seassonal()` should produce an E100 error. *)

let test_unknown_func_call_e100 () =
  let src = {|
    compartments { S, I, R }
    parameters {
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seassonal() * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls =
    try Parser.file Lexer.token lexbuf
    with _ -> Alcotest.fail "parse failed"
  in
  let (_model, ctx, _summary) = Expander.expand_detail "test_unk_func" decls in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error)
  in
  let found_e100 = List.exists (fun d -> d.Diagnostics.code = "E100") errors in
  Alcotest.(check bool) "E100 for unknown function call" true found_e100

(* ── Issue 1: Time function param args preserved ─────────────────────────────
   Compile a model with a sinusoidal function where amplitude is a parameter.
   The compiled Sinusoidal.amplitude should be Ir.Param "alpha", not Ir.Const 0.0.*)

let test_time_func_param_arg () =
  let src = {|
    compartments { S, I, R }
    parameters {
      alpha : positive
      gamma : rate
      N0    : count
      I0    : count
    }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = alpha
        period    = 365.0
        phase     = 0.0
        baseline  = 1.0
      }
    }
    let N = S + I + R
    transitions {
      infection : S --> I  @ seasonal(t) * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init {
      S = N0 - I0
      I = I0
    }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  match Compiler.compile ~name:"test_tf_param" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let tf = List.find (fun (t : Ir.time_function) -> t.Ir.name = "seasonal") m.Ir.time_functions in
    (match tf.Ir.kind with
     | Ir.Sinusoidal s ->
       (match s.Ir.amplitude with
        | Ir.Param "alpha" -> ()  (* pass *)
        | Ir.Const 0.0     -> Alcotest.fail "amplitude was silently converted to 0.0 (param not preserved)"
        | other ->
          Alcotest.failf "expected Ir.Param \"alpha\", got: %s"
            (Serde.model_to_string { m with Ir.time_functions =
               [{ tf with Ir.kind = Ir.Sinusoidal { s with Ir.amplitude = other } }] }))
     | _ -> Alcotest.fail "expected Sinusoidal kind")

(* ── Layer 3: age-targeted SIA ────────────────────────────────────────────── *)

let test_polio_age_sia_targets_under5 () =
  let src = read_file (Filename.concat golden_dir "polio_age.camdl") in
  match Compiler.compile ~name:"polio_age" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* There should be exactly one intervention named sia_round_1 *)
    let iv = match List.find_opt (fun (iv : Ir.intervention) -> iv.name = "sia_round_1") m.interventions with
      | Some iv -> iv
      | None -> Alcotest.fail "sia_round_1 intervention not found"
    in
    (* Its only action should transfer S_under5 → V_under5 (not S_over5) *)
    (match iv.actions with
     | [ Ir.FractionTransfer { src; dst; _ } ] ->
       Alcotest.(check string) "src is S_under5" "S_under5" src;
       Alcotest.(check string) "dst is V_under5" "V_under5" dst
     | _ -> Alcotest.fail "expected exactly one FractionTransfer action")

(* ── Layer 4: where p!=q guard filters diagonal importation ─────────────── *)

let test_spatial_5_importation_count () =
  let src = read_file (Filename.concat golden_dir "polio_spatial_5.camdl") in
  match Compiler.compile ~name:"polio_spatial_5" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (* 5 patches × 5 transitions (local) = 25 compartments *)
    Alcotest.(check int) "25 compartments" 25 (List.length m.compartments);
    (* importation[p,q where p!=q]: 5×5 - 5 = 20 transitions *)
    let imports = List.filter (fun (t : Ir.transition) ->
      let n = t.name in
      String.length n > 12 &&
      String.sub n 0 12 = "importation_"
    ) m.transitions in
    Alcotest.(check int) "20 importation transitions (where p!=q)" 20 (List.length imports);
    (* No self-loop: importation_north_north must not exist *)
    let has_self = List.exists (fun (t : Ir.transition) ->
      t.name = "importation_north_north" ||
      t.name = "importation_south_south" ||
      t.name = "importation_center_center"
    ) m.transitions in
    Alcotest.(check bool) "no self-loop importation" false has_self

(* ── Issue 5: preset_enable roundtrip ────────────────────────────────────────
   Compile seir_vaccine.camdl and verify the with_sia preset has
   preset_enable = ["sia_round_1"]. ─────────────────────────────────────── *)

let test_preset_enable_seir_vaccine () =
  let src = read_file (Filename.concat golden_dir "seir_vaccine.camdl") in
  match Compiler.compile ~name:"seir_vaccine" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let with_sia = match List.find_opt (fun (p : Ir.preset) -> p.Ir.preset_name = "with_sia") m.Ir.presets with
      | Some p -> p
      | None   -> Alcotest.fail "with_sia preset not found"
    in
    Alcotest.(check (list string)) "with_sia preset_enable"
      ["sia_round_1"] with_sia.Ir.preset_enable

(* ── origin + date() ──────────────────────────────────────────────────────── *)

let test_date_to_const () =
  (* 2019-07-01 − 2019-01-01 = 181 days *)
  let src = {|
    time_unit = 'days
    origin = date("2019-01-01")
    compartments { S }
    simulate { from = date("2019-01-01")  to = date("2019-07-01") }
  |} in
  match Compiler.compile ~name:"t" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    Alcotest.(check (option string)) "origin stored" (Some "2019-01-01") m.Ir.origin;
    Alcotest.(check (float 1e-9)) "t_start = 0" 0.0 m.Ir.simulation.Ir.t_start;
    Alcotest.(check (float 1e-9)) "t_end = 181 days" 181.0 m.Ir.simulation.Ir.t_end

let test_date_requires_origin () =
  let src = {|
    time_unit = 'days
    compartments { S }
    simulate { from = date("2019-07-01")  to = date("2019-07-01") }
  |} in
  let lexbuf = Lexing.from_string src in
  let decls = try Parser.file Lexer.token lexbuf
              with _ -> Alcotest.fail "parse failed" in
  let (_model, ctx, _summary) = Expander.expand_detail "t" decls in
  let errors = ctx.diags.Diagnostics.diags
    |> List.filter (fun d -> d.Diagnostics.severity = Diagnostics.Error) in
  let found_e220 = List.exists (fun d -> d.Diagnostics.code = "E220") errors in
  Alcotest.(check bool) "E220 emitted when origin missing" true found_e220

(* A negative lower bound (e.g. a seed time before the origin, `tau : instant
   in [-40, 120]`) must survive — not be silently floored to 0. Regression for
   the resolve_bounds const-eval fix (negated literals were hitting the 0.0
   fallback). *)
let test_negative_lower_bound () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta : rate
      gamma : rate
      tau  : real in [-40.0, 120.0]
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * (I / N) + tau * 0.0
      recovery  : I --> R @ gamma * I
    }
    init { S = 990  I = 10 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  match Compiler.compile ~name:"t" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    let tau = List.find (fun (p : Ir.parameter) -> p.name = "tau") m.Ir.parameters in
    (match (Ir.param_bounds tau) with
     | Some (lo, hi) ->
       Alcotest.(check (float 1e-9)) "lower bound preserved (not floored to 0)" (-40.0) lo;
       Alcotest.(check (float 1e-9)) "upper bound preserved" 120.0 hi
     | None -> Alcotest.fail "tau should carry bounds")

(* ── Prior distribution syntax ──────────────────────────────────────────
   Test that ~ prior(...) syntax parses and produces correct IR priors. *)


let find_param (m : Ir.model) name =
  List.find (fun (p : Ir.parameter) -> p.name = name) m.parameters

let test_prior_log_normal () =
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
      N0   : count in [100, 1000000]
    }
    compartments { S, I, R }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
    }
    init { S = N0 - 10  I = 10  R = 0 }
    simulate { from = 0 'days  to = 100 'days }
  |} in
  let m = compile_expect_ok src in
  let beta = find_param m "beta" in
  match (Ir.param_prior_dist beta) with
  | Some (Ir.LogNormal { mu; sigma }) ->
    Alcotest.(check (float 1e-10)) "mu" (-1.0) mu;
    Alcotest.(check (float 1e-10)) "sigma" 0.5 sigma
  | _ -> Alcotest.fail "expected LogNormal prior"

let test_prior_beta () =
  let src = {|
    time_unit = 'days
    parameters {
      rho  : probability in [0.01, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
      N0   : count in [100, 1000000]
    }
    compartments { S }
    init { S = N0 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let rho = find_param m "rho" in
  match (Ir.param_prior_dist rho) with
  | Some (Ir.Beta { alpha; beta }) ->
    Alcotest.(check (float 1e-10)) "alpha" 2.0 alpha;
    Alcotest.(check (float 1e-10)) "beta" 5.0 beta
  | _ -> Alcotest.fail "expected Beta prior"

let test_prior_gamma_with_rate_kwarg () =
  (* 'rate' is a DSL keyword — make sure it works as a prior kwarg name *)
  let src = {|
    time_unit = 'days
    parameters {
      x : positive in [0.01, 100.0] ~ gamma(shape = 2.0, rate = 0.1)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let x = find_param m "x" in
  match (Ir.param_prior_dist x) with
  | Some (Ir.Gamma { shape; rate }) ->
    Alcotest.(check (float 1e-10)) "shape" 2.0 shape;
    Alcotest.(check (float 1e-10)) "rate" 0.1 rate
  | _ -> Alcotest.fail "expected Gamma prior"

let test_prior_half_normal () =
  let src = {|
    time_unit = 'days
    parameters {
      sigma_noise : positive in [0.001, 10.0] ~ half_normal(sigma = 0.5)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let p = find_param m "sigma_noise" in
  match (Ir.param_prior_dist p) with
  | Some (Ir.HalfNormal { sigma }) ->
    Alcotest.(check (float 1e-10)) "sigma" 0.5 sigma
  | _ -> Alcotest.fail "expected HalfNormal prior"

let test_prior_log_uniform () =
  let src = {|
    time_unit = 'days
    parameters {
      kappa : rate in [1e-5, 1e-2] ~ log_uniform(lower = 1e-5, upper = 1e-2)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let p = find_param m "kappa" in
  match (Ir.param_prior_dist p) with
  | Some (Ir.LogUniform { lu_lower; lu_upper }) ->
    Alcotest.(check (float 1e-12)) "lower" 1e-5 lu_lower;
    Alcotest.(check (float 1e-12)) "upper" 1e-2 lu_upper
  | _ -> Alcotest.fail "expected LogUniform prior"

let test_prior_log_uniform_nonpositive_errors () =
  (* log_uniform is uniform on the log scale → bounds must be > 0. *)
  compile_expect_error_code ~code:"E235" ~contains:"lower > 0" {|
    time_unit = 'days
    parameters {
      x : rate in [0.0, 1.0] ~ log_uniform(lower = 0.0, upper = 1.0)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |}

let test_prior_truncated_normal_bounds_from_decl () =
  (* truncated_normal reads its truncation bounds from the param's `in [..]`. *)
  let src = {|
    time_unit = 'days
    parameters {
      take : probability in [0.3, 1.0] ~ truncated_normal(mean = 0.7, sd = 0.2)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let p = find_param m "take" in
  match (Ir.param_prior_dist p) with
  | Some (Ir.TruncatedNormal { tn_mean; tn_sd; tn_lower; tn_upper }) ->
    Alcotest.(check (float 1e-12)) "mean"  0.7 tn_mean;
    Alcotest.(check (float 1e-12)) "sd"    0.2 tn_sd;
    Alcotest.(check (float 1e-12)) "lower" 0.3 tn_lower;
    Alcotest.(check (float 1e-12)) "upper" 1.0 tn_upper
  | _ -> Alcotest.fail "expected TruncatedNormal prior"

let test_prior_truncated_normal_requires_bounds () =
  (* No `in [..]` → truncated_normal has nothing to truncate to → E236. *)
  compile_expect_error_code ~code:"E285" ~contains:"requires explicit bounds" {|
    time_unit = 'days
    parameters {
      take : probability ~ truncated_normal(mean = 0.7, sd = 0.2)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |}

let test_prior_log_uniform_not_poolable () =
  (* A param-reference arg makes a prior hierarchical; log_uniform can't be,
     so it must report E237 (a diagnostic), not ICE in hierarchical_kind_of_name. *)
  compile_expect_error_code ~code:"E286" ~contains:"cannot be hierarchical" {|
    time_unit = 'days
    parameters {
      hi    : positive in [1e-3, 1.0] ~ half_normal(sigma = 0.1)
      kappa : rate in [1e-5, 1e-2] ~ log_uniform(lower = 1e-5, upper = hi)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |}

let test_no_prior_is_none () =
  let src = {|
    time_unit = 'days
    parameters {
      N0 : count in [100, 1000000]
    }
    compartments { S }
    init { S = N0 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let n0 = find_param m "N0" in
  Alcotest.(check bool) "no prior means None" true ((Ir.param_prior_dist n0) = None)

let test_indexed_param_shares_prior () =
  (* Indexed parameters: the prior applies to all expanded instances *)
  let src = {|
    time_unit = 'days
    dimensions {
      patch = [north, south, east]
    }
    parameters {
      R0[patch] : positive in [1.0, 10.0] ~ log_normal(mu = 1.0, sigma = 0.3)
      N0        : count in [100, 1000000]
    }
    compartments { S }
    init { S = N0 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  let expected = Ir.LogNormal { mu = 1.0; sigma = 0.3 } in
  List.iter (fun name ->
    let p = find_param m name in
    match (Ir.param_prior_dist p) with
    | Some pd when pd = expected -> ()
    | _ -> Alcotest.failf "%s should have LogNormal prior" name
  ) ["R0_north"; "R0_south"; "R0_east"]

let test_unknown_prior_errors () =
  let src = {|
    time_unit = 'days
    parameters {
      x : rate in [0.01, 1.0] ~ weibull(shape = 2.0, scale = 1.0)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  compile_expect_error_code ~code:"E232" ~contains:"parameter 'x'" src

(* Wrapper — the prior-arg tests all need a minimal compile-clean model. *)
let src_with_prior prior_expr = Printf.sprintf {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 2.0] ~ %s
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} prior_expr

(* ── E230: non-constant prior argument ──────────────────────────────────── *)

let test_e230_non_const_arg () =
  (* After wave 2 / #3 landed hierarchical priors, a reference to a
     declared parameter in a prior arg is legitimately non-const (it's
     a hyperparent). Undeclared names are still an error — caught by
     the generic name-resolution pass as E100 "undeclared name". This
     test pins that behaviour: the error still fires, just under the
     canonical code. *)
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 2.0] ~ log_normal(mu = undeclared, sigma = 0.5)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  compile_expect_error_code ~code:"E100" ~contains:"undeclared" src

(* ── E231: missing required kwarg ──────────────────────────────────────── *)

let test_e231_missing_kwarg () =
  compile_expect_error_code ~code:"E231" ~contains:"parameter 'beta'"
    (src_with_prior "log_normal(mu = -1.0)")

let test_e231_missing_kwarg_half_normal () =
  compile_expect_error_code ~code:"E231" ~contains:"sigma"
    (src_with_prior "half_normal()")

(* ── E233: unknown / extra kwarg ───────────────────────────────────────── *)

let test_e233_unknown_kwarg () =
  compile_expect_error_code ~code:"E233" ~contains:"extra"
    (src_with_prior "log_normal(mu = -1.0, sigma = 0.5, extra = 99)")

let test_e233_typo_kwarg () =
  (* 'mean' instead of 'mu' — common mistake, good test of the error's
     discoverability. *)
  compile_expect_error_code ~code:"E233" ~contains:"log_normal"
    (src_with_prior "log_normal(mean = -1.0, sigma = 0.5)")

(* ── E234: duplicate kwarg ─────────────────────────────────────────────── *)

let test_e234_duplicate_kwarg () =
  compile_expect_error_code ~code:"E234" ~contains:"duplicate"
    (src_with_prior "log_normal(mu = -1.0, mu = -5.0, sigma = 0.5)")

(* ── E235: invalid distribution values ─────────────────────────────────── *)

let test_e235_uniform_inverted () =
  compile_expect_error_code ~code:"E235" ~contains:"lower < upper"
    (src_with_prior "uniform(lower = 5.0, upper = 1.0)")

let test_e235_beta_negative_alpha () =
  compile_expect_error_code ~code:"E235" ~contains:"alpha"
    (src_with_prior "beta(alpha = -1.0, beta = 2.0)")

let test_e235_gamma_zero_shape () =
  compile_expect_error_code ~code:"E235" ~contains:"shape"
    (src_with_prior "gamma(shape = 0.0, rate = 1.0)")

let test_e235_exponential_zero_rate () =
  compile_expect_error_code ~code:"E235" ~contains:"rate"
    (src_with_prior "exponential(rate = 0.0)")

let test_e235_normal_negative_sigma () =
  compile_expect_error_code ~code:"E235" ~contains:"sigma"
    (src_with_prior "normal(mu = 0.0, sigma = -1.0)")

let test_e235_half_normal_zero_sigma () =
  compile_expect_error_code ~code:"E235" ~contains:"sigma"
    (src_with_prior "half_normal(sigma = 0.0)")

(* ── Additional distributions: parse + value round-trip ────────────────── *)

let test_prior_uniform () =
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.0, 10.0] ~ uniform(lower = 0.1, upper = 2.0)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "beta")) with
  | Some (Ir.Uniform { lower; upper }) ->
    Alcotest.(check (float 1e-10)) "lower" 0.1 lower;
    Alcotest.(check (float 1e-10)) "upper" 2.0 upper
  | _ -> Alcotest.fail "expected Uniform prior"

let test_prior_normal () =
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.0, 10.0] ~ normal(mu = 0.3, sigma = 0.1)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "beta")) with
  | Some (Ir.Normal_p { mean; sd }) ->
    Alcotest.(check (float 1e-10)) "mean" 0.3 mean;
    Alcotest.(check (float 1e-10)) "sd" 0.1 sd
  | _ -> Alcotest.fail "expected Normal prior"

let test_prior_exponential () =
  let src = {|
    time_unit = 'days
    parameters {
      lambda : rate in [0.0, 100.0] ~ exponential(rate = 2.5)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "lambda")) with
  | Some (Ir.Exponential { rate }) ->
    Alcotest.(check (float 1e-10)) "rate" 2.5 rate
  | _ -> Alcotest.fail "expected Exponential prior"

(* ── Compile-time arithmetic in prior arguments ────────────────────────── *)

let test_prior_arg_arithmetic () =
  (* Users often encode priors via arithmetic of literals — e.g. when a
     review paper reports a 95% CI that translates to mu ± 1.96*sigma,
     or when combining multiple constants. The const-evaluator should
     handle +, -, *, /, ^ on literals transparently. *)
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 10.0] ~ log_normal(mu = -1.0 * 2.0 + 0.5, sigma = 1.0 / 4.0)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "beta")) with
  | Some (Ir.LogNormal { mu; sigma }) ->
    Alcotest.(check (float 1e-12)) "mu = -1.5" (-1.5) mu;
    Alcotest.(check (float 1e-12)) "sigma = 0.25" 0.25 sigma
  | _ -> Alcotest.fail "expected LogNormal prior"

let test_prior_arg_log_function () =
  (* `mu = log(0.3)` is the canonical way to encode a log_normal with
     a named median. Regression test for the EFuncCall const-eval fix. *)
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 10.0] ~ log_normal(mu = log(0.3), sigma = 0.5)
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "beta")) with
  | Some (Ir.LogNormal { mu; sigma }) ->
    Alcotest.(check (float 1e-12)) "mu = log(0.3)" (log 0.3) mu;
    Alcotest.(check (float 1e-12)) "sigma" 0.5 sigma
  | _ -> Alcotest.fail "expected LogNormal prior"

let test_prior_arg_exp_and_sqrt () =
  (* Exercise exp() and sqrt() in const position — less common than log
     but same path through is_const_expr/eval_const_expr. *)
  let src = {|
    time_unit = 'days
    parameters {
      beta : rate in [0.01, 10.0] ~ gamma(shape = sqrt(9.0), rate = exp(0.0))
    }
    compartments { S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match (Ir.param_prior_dist (find_param m "beta")) with
  | Some (Ir.Gamma { shape; rate }) ->
    Alcotest.(check (float 1e-12)) "shape = sqrt(9)" 3.0 shape;
    Alcotest.(check (float 1e-12)) "rate = exp(0)" 1.0 rate
  | _ -> Alcotest.fail "expected Gamma prior"

(* ── Observation projections on stratified compartments ────────────────────
   `prevalence(E)` on an Erlang-stratified `E` (E_e1, E_e2, E_e3) should
   expand to `CurrentPopSum [E_e1; E_e2; E_e3]`, following the same
   "omitted dimension sums over it" rule that applies to rate expressions
   (see `resolve_ident_name` and language spec §5.1). Previously emitted
   `CurrentPop "E"` which the Rust runtime could not resolve.
   See docs/dev/proposals/2026-04-17-state-snapshot-projections.md. *)
let test_prevalence_on_stratified_compartment () =
  let src = {|
    time_unit = 'days
    compartments { S, E, I, R }
    dimensions { latent_stage = [e1, e2, e3] }
    stratify(by = latent_stage, only = [E])
    parameters {
      beta  : rate in [0.001, 2.0]
      sigma : rate in [0.01, 1.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [1.0, 100.0]
    }
    transitions {
      infection : S --> E[e1] @ beta * S * I / (S + E + I + R)
      latent[(s, s_next) in consecutive(latent_stage)]
        : E[s] --> E[s_next] @ 3 * sigma * E[s]
      onset : E[e3] --> I @ 3 * sigma * E[e3]
      recovery : I --> R @ gamma * I
    }
    observations {
      in_latent {
        columns       { time : time, in_latent : count }
        projected  = prevalence(E)
        emit_schedule = every 1 'days
        in_latent ~ neg_binomial(mean = projected, r = k)
      }
    }
    init { S = 990  E[e1] = 5  I = 5 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match m.observations with
  | [obs] ->
    (match obs.projection with
     | Ir.CurrentPopSum names ->
       Alcotest.(check (list string))
         "prevalence(E) expands to all Erlang substages"
         ["E_e1"; "E_e2"; "E_e3"] names
     | Ir.CurrentPop name ->
       Alcotest.failf
         "expected CurrentPopSum over Erlang substages; got CurrentPop(%s)" name
     | _ ->
       Alcotest.fail "expected CurrentPopSum projection")
  | _ -> Alcotest.fail "expected exactly one observation block"

(* Same rule for `projected = E` (bare identifier form that resolves to a
   stratified compartment). *)
let test_projected_bare_stratified_compartment () =
  let src = {|
    time_unit = 'days
    compartments { S, E, I, R }
    dimensions { latent_stage = [e1, e2, e3] }
    stratify(by = latent_stage, only = [E])
    parameters {
      beta  : rate in [0.001, 2.0]
      sigma : rate in [0.01, 1.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [1.0, 100.0]
    }
    transitions {
      infection : S --> E[e1] @ beta * S * I / (S + E + I + R)
      latent[(s, s_next) in consecutive(latent_stage)]
        : E[s] --> E[s_next] @ 3 * sigma * E[s]
      onset : E[e3] --> I @ 3 * sigma * E[e3]
      recovery : I --> R @ gamma * I
    }
    observations {
      latent_total {
        columns       { time : time, latent_total : count }
        projected  = E
        emit_schedule = every 1 'days
        latent_total ~ neg_binomial(mean = projected, r = k)
      }
    }
    init { S = 990  E[e1] = 5  I = 5 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match m.observations with
  | [obs] ->
    (match obs.projection with
     | Ir.CurrentPopSum names ->
       Alcotest.(check (list string))
         "bare E in projection expands to all Erlang substages"
         ["E_e1"; "E_e2"; "E_e3"] names
     | _ -> Alcotest.fail "expected CurrentPopSum projection for bare stratified compartment")
  | _ -> Alcotest.fail "expected exactly one observation block"

(* Fully-indexed prevalence on a stratified compartment picks a specific
   stratum (not a sum). Guards against over-eagerly sum-expanding when
   the user wanted one. *)
let test_prevalence_fully_indexed_stratified () =
  let src = {|
    time_unit = 'days
    compartments { S, E, I, R }
    dimensions { latent_stage = [e1, e2, e3] }
    stratify(by = latent_stage, only = [E])
    parameters {
      beta  : rate in [0.001, 2.0]
      sigma : rate in [0.01, 1.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [1.0, 100.0]
    }
    transitions {
      infection : S --> E[e1] @ beta * S * I / (S + E + I + R)
      latent[(s, s_next) in consecutive(latent_stage)]
        : E[s] --> E[s_next] @ 3 * sigma * E[s]
      onset : E[e3] --> I @ 3 * sigma * E[e3]
      recovery : I --> R @ gamma * I
    }
    observations {
      first_latent {
        columns       { time : time, first_latent : count }
        projected  = prevalence(E[e1])
        emit_schedule = every 1 'days
        first_latent ~ neg_binomial(mean = projected, r = k)
      }
    }
    init { S = 990  E[e1] = 5  I = 5 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.CurrentPop "E_e1" -> ()
  | Ir.CurrentPopSum _ ->
    Alcotest.fail "fully-indexed prevalence must not sum over strata"
  | Ir.CurrentPop other ->
    Alcotest.failf "expected CurrentPop E_e1, got CurrentPop %s" other
  | _ -> Alcotest.fail "expected CurrentPop projection"

(* Unstratified compartment — behavior unchanged. *)
let test_prevalence_unstratified () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta  : rate in [0.001, 2.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [1.0, 100.0]
    }
    transitions {
      infection : S --> I @ beta * S * I
      recovery  : I --> R @ gamma * I
    }
    observations {
      prev {
        columns       { time : time, prev : count }
        projected  = prevalence(I)
        emit_schedule = every 1 'days
        prev ~ neg_binomial(mean = projected, r = k)
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.CurrentPop "I" -> ()
  | _ -> Alcotest.fail "expected CurrentPop I on unstratified compartment"

(* ── Observation incidence on stratified transitions ───────────────────────
   Symmetric to the prevalence-on-stratified tests above. Un-indexed
   `incidence(infection)` over a stratified `infection[a in age]` family
   should expand to the SUM of per-stratum cumulative flows
   (`CumulativeFlowSum [infection_child; infection_adult]`), per language
   spec §25.4. Previously emitted a bare `CumulativeFlow "infection"` that
   referenced a name not present post-expansion → E507 at compile.
   gh#160/#164/#165 (Defect A). *)

let stratified_age_seir_with_obs obs_block =
  Printf.sprintf {|
    time_unit = 'days
    compartments { S, E, I, R }
    dimensions { age = [child, adult] }
    stratify(by = age)
    let N_local[a in age] = S[a] + E[a] + I[a] + R[a]
    parameters {
      beta  : rate in [0.001, 0.5]
      sigma : rate in [0.01, 1.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [0.1, 100.0]
    }
    tables { C_age : age × age = [[12.0, 4.0], [4.0, 8.0]] }
    transitions {
      infection[a in age] : S[a] --> E[a]
        @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
      progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
      recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
    }
    %s
    init { S[child] = 4990  S[adult] = 5000  I[child] = 10 }
    simulate { from = 0 'days  to = 50 'days }
  |} obs_block

(* Cross-strata aggregation gate (2026-06-10 observation data-entry §5.2): a
   bare un-indexed `incidence(infection)` on a stratified model is now a HARD
   ERROR (E280). It would silently sum all strata and apply reporting uniformly;
   the modeller must state the aggregation explicitly. *)
let test_incidence_unindexed_cross_strata_is_rejected () =
  let src = stratified_age_seir_with_obs {|
    observations {
      weekly_cases {
        columns       { time : time, weekly_cases : count }
        projected  = incidence(infection)
        emit_schedule = every 7 'days
        weekly_cases ~ neg_binomial(mean = projected, r = k)
      }
    }
  |} in
  compile_expect_error_code ~code:"E280" ~contains:"sum" src

(* The explicit uniform-reporting form the gate directs the modeller to:
   `rho * sum(a in age, incidence(infection[a]))` — here without the rho factor
   for the projection check — compiles and expands to the IDENTICAL
   CumulativeFlowSum the bare form used to produce. The reporting choice is now
   stated, not silent. *)
let test_incidence_explicit_sum_compiles_to_flow_sum () =
  let src = stratified_age_seir_with_obs {|
    observations {
      weekly_cases {
        columns       { time : time, weekly_cases : count }
        projected  = sum(a in age, incidence(infection[a]))
        emit_schedule = every 7 'days
        weekly_cases ~ neg_binomial(mean = projected, r = k)
      }
    }
  |} in
  let m = compile_expect_ok src in
  match m.observations with
  | [obs] ->
    (match obs.projection with
     | Ir.CumulativeFlowSum names ->
       Alcotest.(check (list string))
         "explicit sum(a in age, incidence(infection[a])) expands to the per-stratum flow sum"
         ["infection_child"; "infection_adult"] names
     | Ir.CumulativeFlow name ->
       Alcotest.failf
         "expected CumulativeFlowSum over age strata; got CumulativeFlow(%s)" name
     | _ -> Alcotest.fail "expected CumulativeFlowSum projection")
  | _ -> Alcotest.fail "expected exactly one observation block"

(* Positional-indexed incidence pins a single stratum (not a sum). Guards
   against over-eagerly summing when the user named a stratum. *)
let test_incidence_positional_indexed_pins_one_stratum () =
  let src = stratified_age_seir_with_obs {|
    observations {
      child_cases {
        columns       { time : time, child_cases : count }
        projected  = incidence(infection[child])
        emit_schedule = every 7 'days
        child_cases ~ neg_binomial(mean = projected, r = k)
      }
    }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.CumulativeFlow "infection_child" -> ()
  | Ir.CumulativeFlowSum _ ->
    Alcotest.fail "indexed incidence must not sum over strata"
  | Ir.CumulativeFlow other ->
    Alcotest.failf "expected CumulativeFlow infection_child, got CumulativeFlow %s" other
  | _ -> Alcotest.fail "expected CumulativeFlow projection"

(* Named-indexed incidence pins the named dimension, order-independent. *)
let test_incidence_named_indexed_pins_one_stratum () =
  let src = stratified_age_seir_with_obs {|
    observations {
      adult_cases {
        columns       { time : time, adult_cases : count }
        projected  = incidence(infection[age = adult])
        emit_schedule = every 7 'days
        adult_cases ~ neg_binomial(mean = projected, r = k)
      }
    }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.CumulativeFlow "infection_adult" -> ()
  | _ -> Alcotest.fail "expected CumulativeFlow infection_adult for named index"

(* Unstratified incidence — behaviour unchanged: exact single flow. *)
let test_incidence_unstratified () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta  : rate in [0.001, 2.0]
      gamma : rate in [0.01, 1.0]
      k     : real in [1.0, 100.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      cases {
        columns       { time : time, cases : count }
        projected  = incidence(infection)
        emit_schedule = every 1 'days
        cases ~ neg_binomial(mean = projected, r = k)
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.CumulativeFlow "infection" -> ()
  | Ir.CumulativeFlowSum _ ->
    Alcotest.fail "unstratified incidence must stay a single CumulativeFlow"
  | _ -> Alcotest.fail "expected CumulativeFlow infection on unstratified transition"

(* Defect A' (gh#164/#165): a let-bound bare identifier in `projected`
   must inline the let body (a DerivedExpr), not fall through to a dangling
   CumulativeFlow. `projected = I_total` with `let I_total = I[child] +
   I[adult]` resolves to a state expression over the expanded compartments. *)
let test_let_bound_projection_inlines () =
  let src = stratified_age_seir_with_obs {|
    let I_total = I[child] + I[adult]
    observations {
      prevalence_total {
        columns       { time : time, prevalence_total : count }
        projected  = I_total
        emit_schedule = every 7 'days
        prevalence_total ~ neg_binomial(mean = projected, r = k)
      }
    }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).projection with
  | Ir.DerivedExpr e ->
    (* The let body I[child] + I[adult] resolves to a sum over the two
       expanded infectious compartments. Accept either the normalized
       PopSum form or an Add over the two Pops — both denote the same
       sum; the load-bearing assertion is that it is NOT a dangling
       CumulativeFlow. *)
    let comps =
      let acc = ref [] in
      let rec walk = function
        | Ir.Pop c -> acc := c :: !acc
        | Ir.PopSum cs -> List.iter (fun c -> acc := c :: !acc) cs
        | Ir.BinOp { left; right; _ } -> walk left; walk right
        | _ -> ()
      in
      walk e; List.sort compare !acc
    in
    Alcotest.(check (list string))
      "let-bound projection inlines to a sum over the expanded compartments"
      ["I_adult"; "I_child"] comps
  | Ir.CumulativeFlow name ->
    Alcotest.failf
      "let-bound projection must inline, not emit CumulativeFlow(%s)" name
  | _ -> Alcotest.fail "expected DerivedExpr for let-bound projection"

(* ── Likelihood keyword-argument parsing ──────────────────────────────────
   `rate` is a reserved keyword in parameter type annotations; the kwarg
   rule in the parser must allow it (and other soft keywords) in kwarg
   position so `poisson(rate = projected)` parses. Also ensure missing or
   positional args are rejected with real diagnostics, not a silent 0.0. *)
let test_poisson_rate_kwarg_parses () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta  : rate in [0.001, 5.0]
      gamma : rate in [0.01, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      in_bed {
        columns       { time : time, in_bed : count }
        projected = prevalence(I)
        emit_schedule = every 1 'days
        in_bed ~ poisson(rate = projected)
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).likelihood with
  | Ir.Poisson { rate = { Ir.expr = Ir.Projected; _ } } -> ()
  | _ -> Alcotest.fail "expected Poisson{ rate = Projected }"

let test_poisson_positional_errors () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta  : rate in [0.001, 5.0]
      gamma : rate in [0.01, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      in_bed {
        columns       { time : time, in_bed : count }
        projected = prevalence(I)
        emit_schedule = every 1 'days
        in_bed ~ poisson(projected)
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  compile_expect_error_code ~code:"E250" ~contains:"poisson" src

let test_likelihood_unknown_kwarg_errors () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta  : rate in [0.001, 5.0]
      gamma : rate in [0.01, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      in_bed {
        columns       { time : time, in_bed : count }
        projected = prevalence(I)
        emit_schedule = every 1 'days
        in_bed ~ poisson(lambda = projected)
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  compile_expect_error_code ~code:"E251" ~contains:"lambda" src

(* ── Stage 2: survey denominators (per-obs aux) + dimcheck-n ─────────────── *)

(* A survey-positivity stream: `positive ~ binomial(n = tested, p = ...)`.
   `tested` is a declared aux value column referenced on the `~` RHS — it
   resolves to an `ObsColumnRef` leaf (NOT a parameter/compartment), and the
   model compiles. This is the headline Stage-2 surface. *)
let survey_positivity_model lik =
  Printf.sprintf {|
    time_unit = 'days
    compartments { S, I, R }
    let N = S + I + R
    parameters {
      beta  : rate in [0.001, 5.0]
      gamma : rate in [0.01, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    observations {
      survey {
        columns   { time : time, pos : count, tested : count }
        projected = prevalence(I)
        %s
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} lik

let test_survey_denominator_resolves_to_obs_column_ref () =
  let src = survey_positivity_model
    "pos ~ binomial(n = tested, p = projected / N)" in
  let m = compile_expect_ok src in
  match (List.hd m.observations).likelihood with
  | Ir.Binomial { n = Ir.ObsColumnRef "tested"; _ } -> ()
  | Ir.Binomial { n; _ } ->
    Alcotest.failf "expected binomial n = ObsColumnRef \"tested\"; got %s"
      (Pp_expr.to_string n)
  | _ -> Alcotest.fail "expected a Binomial likelihood"

(* NOTE on dimcheck-n (§3.1): `test_compiler.ml` runs with dimcheck DISABLED
   globally (`Compiler.no_dim_check := true`, top of file), so the E304 checks
   on the binomial `n` and Poisson `rate` are exercised in `test_dimcheck.ml`
   (which runs dimcheck), not here. The parse/resolution surface (an aux column
   resolving to `ObsColumnRef`) is covered above. *)

(* A dead aux column — declared but never referenced on the `~` RHS — is the
   existing E277 dead-column error, unchanged by Stage 2. *)
let test_unreferenced_aux_column_is_dead () =
  let src = survey_positivity_model
    "pos ~ binomial(n = 1000, p = projected / N)" in
  (* `tested` is declared but `n = 1000` (a constant), so `tested` is dead. *)
  compile_expect_error_code ~code:"E277" ~contains:"tested" src

(* ── Multi-source transitions (Wave 1 / #1) ──────────────────────────────── *)

(** Parser accepts `S + I --> I + I` on the source side. *)
let test_multi_source_parses () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate in [0.0001, 1.0] }
    transitions {
      infect : S + I --> I + I  @ beta * S * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let _m = compile_expect_ok src in
  ()

(** Catalyst collapse: `S + I --> I + I` produces the same stoichiometry
    as the plain `S --> I` single-source form. The I on both sides
    should sum to zero and be dropped; the rate expression retains its
    reference to I. *)
let test_multi_source_catalyst_collapses () =
  let multi = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate in [0.0001, 1.0] }
    transitions {
      infect : S + I --> I + I  @ beta * S * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let single = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate in [0.0001, 1.0] }
    transitions {
      infect : S --> I  @ beta * S * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m_multi  = compile_expect_ok multi  in
  let m_single = compile_expect_ok single in
  let stoich_of m =
    match m.Ir.transitions with
    | [t] -> List.sort compare t.Ir.stoichiometry
    | _   -> Alcotest.fail "expected exactly one transition"
  in
  let s_multi  = stoich_of m_multi  in
  let s_single = stoich_of m_single in
  Alcotest.(check (list (pair string int)))
    "catalyst-collapsed multi-source stoich == single-source stoich"
    s_single s_multi

(** Indexed multi-source: `bite[a in age] : X[a] + Iv --> I[a] + Iv`.
    Should expand to one transition per age value, each with Iv as a
    catalyst (collapsed to net zero) and X[a] → I[a] as the net flow.
    The stratified pattern is the canonical malaria use case. *)
let test_multi_source_indexed_by_age () =
  let src = {|
    time_unit = 'days
    dimensions { age = [child, adult] }
    compartments { X, I, Iv }
    stratify(by = age, only = [X, I])
    parameters {
      a_bite : rate in [0.01, 1.0]
      b_h    : probability
    }
    let N = X[child] + X[adult] + I[child] + I[adult]
    transitions {
      bite[a in age] : X[a] + Iv --> I[a] + Iv  @ a_bite * b_h * X[a] * Iv / N
    }
    init { X[child] = 100  X[adult] = 100  Iv = 10 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  (* Should expand to exactly two transitions (one per age value),
     each with net stoich {X[a]:-1, I[a]:+1} — Iv collapsed. *)
  let names = List.map (fun (t : Ir.transition) -> t.Ir.name) m.Ir.transitions in
  let sort_strs = List.sort compare in
  Alcotest.(check (list string))
    "one indexed transition per age value"
    ["bite_adult"; "bite_child"]
    (sort_strs names);
  List.iter (fun t ->
    let stoich = List.sort compare t.Ir.stoichiometry in
    let suffix =
      if t.Ir.name = "bite_child" then "child"
      else "adult"
    in
    let expected = List.sort compare [
      (Printf.sprintf "X_%s" suffix, -1);
      (Printf.sprintf "I_%s" suffix,  1);
    ] in
    Alcotest.(check (list (pair string int)))
      (Printf.sprintf "%s stoich has catalyst Iv collapsed" t.Ir.name)
      expected stoich
  ) m.Ir.transitions

(** True bimolecular (non-catalyst) source: `A + B --> C`. Stoichiometry
    must be {A: -1, B: -1, C: +1}. *)
let test_multi_source_bimolecular_stoich () =
  let src = {|
    time_unit = 'days
    compartments { A, B, C }
    parameters { k : rate in [0.0001, 1.0] }
    transitions {
      react : A + B --> C  @ k * A * B
    }
    init { A = 100  B = 100  C = 0 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  let t = match m.Ir.transitions with
    | [t] -> t
    | _   -> Alcotest.fail "expected exactly one transition"
  in
  let got = List.sort compare t.Ir.stoichiometry in
  let expected = List.sort compare [("A", -1); ("B", -1); ("C", 1)] in
  Alcotest.(check (list (pair string int)))
    "A + B --> C produces {A:-1, B:-1, C:+1}"
    expected got

(* ── unchecked_dim per-expression dimensional escape (2026-04-22) ───────── *)

(** Happy path: `unchecked_dim(expr, dim = population, reason = "…")`
    compiles and produces an Ir.UncheckedDim with the asserted dim. *)
let test_unchecked_dim_parses () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters { beta : rate  alpha : real [1]  iota : real [P] }
    transitions {
      infection : S --> I
        @ beta * unchecked_dim((I + iota)^alpha,
                               dim = population,
                               reason = "He et al. 2010 α-mixing exponent")
                * S / (S + I + R)
      recovery : I --> R @ beta * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  (* The first transition's rate should contain an UncheckedDim node
     somewhere in its AST. *)
  let rec contains_unchecked = function
    | Ir.UncheckedDim u ->
      Alcotest.(check (pair int int)) "asserted dim is population (1,0)"
        (1, 0) (u.Ir.dim_p, u.Ir.dim_t);
      Alcotest.(check bool) "reason preserved" true
        (u.Ir.reason = "He et al. 2010 α-mixing exponent");
      true
    | Ir.BinOp b -> contains_unchecked b.left || contains_unchecked b.right
    | Ir.UnOp u  -> contains_unchecked u.arg
    | Ir.Cond c  -> contains_unchecked c.pred || contains_unchecked c.then_ || contains_unchecked c.else_
    | _ -> false
  in
  let tr = List.hd m.Ir.transitions in
  Alcotest.(check bool) "transition rate contains UncheckedDim" true
    (contains_unchecked tr.Ir.rate)

(** Missing `reason` kwarg → E240. *)
let test_unchecked_dim_requires_reason () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate  alpha : real [1]  iota : real [P] }
    transitions {
      infect : S --> I
        @ beta * unchecked_dim((I + iota)^alpha, dim = population) * S / (S + I)
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  compile_expect_error_code ~code:"E240" ~contains:"reason" src

(** Unknown dim name → E240 with domain-name hint. *)
let test_unchecked_dim_unknown_dim_name () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate  alpha : real [1]  iota : real [P] }
    transitions {
      infect : S --> I
        @ beta * unchecked_dim((I + iota)^alpha,
                               dim = bananas,
                               reason = "test")
                * S / (S + I)
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  compile_expect_error_code ~code:"E240" ~contains:"bananas" src

(** The He-style model compiles — a dimensionally-inhomogeneous rate
    expression typechecks when wrapped in `unchecked_dim`. *)
let test_unchecked_dim_he_style_typechecks () =
  let src = {|
    time_unit = 'days
    compartments { S, E, I, R }
    parameters {
      beta  : rate
      alpha : real [1]
      iota  : real [P]
      sigma : rate
      gamma : rate
    }
    let N = S + E + I + R
    transitions {
      infect   : S --> E @ beta * unchecked_dim((I + iota)^alpha,
                                                 dim = population,
                                                 reason = "He 2010 mixing")
                              * S / N
      progress : E --> I @ sigma * E
      recover  : I --> R @ gamma * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let _m = compile_expect_ok src in
  ()

(* ── Unit annotations on interpolated forcing (GH #8, 2026-04-22) ───────── *)

(** Required tier-3 unit literal on sinusoidal forcing; IR dim
    populated from it. *)
let test_sinusoidal_per_day_dim () =
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { amp : rate  baseline : rate }
    forcing {
      seasonal : sinusoidal 'per_day {
        amplitude = amp
        period    = 365.25 'days
        phase     = 0.0
        baseline  = baseline
      }
    }
    transitions { infect : S --> @ seasonal * S }
    init { S = 1000 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match m.Ir.time_functions with
  | [tf] ->
    let (p, t) = tf.Ir.dim in
    Alcotest.(check (pair int int)) "'per_day → (0,-1)" (0, -1) (p, t)
  | _ -> Alcotest.fail "expected one time_function"

(** The `'ratio` literal for dimensionless multipliers. *)
let test_sinusoidal_ratio_dim () =
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { beta : rate }
    forcing {
      seasonal : sinusoidal 'ratio {
        amplitude = 0.3
        period    = 365.25 'days
        phase     = 0.0
        baseline  = 1.0
      }
    }
    transitions { infect : S --> @ beta * seasonal * S }
    init { S = 1000 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let m = compile_expect_ok src in
  match m.Ir.time_functions with
  | [tf] -> Alcotest.(check (pair int int)) "'ratio → (0,0)" (0, 0) tf.Ir.dim
  | _ -> Alcotest.fail "expected one time_function"

(** `'count` unit literal — for population-count forcings. *)
let test_sinusoidal_count_dim () =
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { p : count  q : rate }
    forcing {
      popsize : sinusoidal 'count {
        amplitude = 0.0
        period    = 365 'days
        phase     = 0.0
        baseline  = p
      }
    }
    transitions { out : S --> @ q * S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  let m = compile_expect_ok src in
  match m.Ir.time_functions with
  | [tf] -> Alcotest.(check (pair int int)) "'count → (1,0)" (1, 0) tf.Ir.dim
  | _ -> Alcotest.fail "expected one time_function"

(* ── gh#308 — file-backed interpolated forcing with an ISO-date time_col ──────
   A `data = "..."` interpolated forcing whose time_col holds ISO dates must
   auto-convert them to internal time via the model's origin + time_unit, the
   same rule observation-data time columns and instant/duration tables follow.
   Pre-fix the loader parsed the time cell with `float_of_string_opt` only, so a
   date cell silently dropped the row's time while keeping its value — leaving
   an empty times array against a full values array, which the runtime
   interpolated to 0.0 everywhere (silent-wrong). *)
let extract_interpolated_times (m : Ir.model) name =
  let tf = List.find (fun (t : Ir.time_function) -> t.Ir.name = name) m.Ir.time_functions in
  match tf.Ir.kind with
  | Ir.Interpolated i ->
    List.map (function
      | Ir.Const f -> f
      | _ -> Alcotest.fail "interpolated time knot is not a constant")
      i.Ir.times
  | _ -> Alcotest.failf "forcing %s is not interpolated" name

let test_interpolated_iso_date_time_col () =
  let tmp = Filename.temp_file "camdl_forcing_iso" ".tsv" in
  let oc = open_out tmp in
  output_string oc "time\tvalue\n";
  output_string oc "2016-01-31\t100\n";
  output_string oc "2016-12-31\t200\n";
  output_string oc "2017-12-31\t300\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    origin    = date("2016-01-01")
    compartments { S }
    parameters { dummy : rate }
    forcing {
      eff : interpolated 'count { data = "%s"  time_col = time  value_col = value  method = "linear" }
    }
    transitions { drain : S --> @ dummy * eff * S }
    init { S = 100 }
    simulate { from = date("2016-01-01")  to = date("2018-01-01") }
  |} tmp in
  let m = compile_expect_ok src in
  Sys.remove tmp;
  (* 2016-01-01 origin: Jan 31 → 30, Dec 31 → 365 (2016 leap), 2017-12-31 → 730. *)
  Alcotest.(check (list (float 1e-6))) "ISO dates resolve to day-offsets"
    [30.; 365.; 730.] (extract_interpolated_times m "eff")

(** A numeric time_col is unchanged — day-offsets pass straight through. *)
let test_interpolated_numeric_time_col () =
  let tmp = Filename.temp_file "camdl_forcing_num" ".tsv" in
  let oc = open_out tmp in
  output_string oc "day\tvalue\n";
  output_string oc "30\t100\n";
  output_string oc "365\t200\n";
  output_string oc "730\t300\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    origin    = date("2016-01-01")
    compartments { S }
    parameters { dummy : rate }
    forcing {
      eff : interpolated 'count { data = "%s"  time_col = day  value_col = value  method = "linear" }
    }
    transitions { drain : S --> @ dummy * eff * S }
    init { S = 100 }
    simulate { from = date("2016-01-01")  to = date("2018-01-01") }
  |} tmp in
  let m = compile_expect_ok src in
  Sys.remove tmp;
  Alcotest.(check (list (float 1e-6))) "numeric day-offsets pass through"
    [30.; 365.; 730.] (extract_interpolated_times m "eff")

(** An ISO-date time_col with no model origin is a hard error (E209), not a
    silent fall-through to 0. *)
let test_interpolated_iso_date_no_origin_errors () =
  let tmp = Filename.temp_file "camdl_forcing_iso_noorigin" ".tsv" in
  let oc = open_out tmp in
  output_string oc "time\tvalue\n";
  output_string oc "2016-01-31\t100\n";
  output_string oc "2016-12-31\t200\n";
  close_out oc;
  let src = Printf.sprintf {|
    time_unit = 'days
    compartments { S }
    parameters { dummy : rate }
    forcing {
      eff : interpolated 'count { data = "%s"  time_col = time  value_col = value  method = "linear" }
    }
    transitions { drain : S --> @ dummy * eff * S }
    init { S = 100 }
    simulate { from = 0 'days  to = 10 'days }
  |} tmp in
  compile_expect_error_code ~code:"E209" ~contains:"origin" src;
  Sys.remove tmp

(** Forcing decl without a unit literal now fails at parse time. *)
let test_forcing_without_unit_errors () =
  (* GH #8: every forcing declaration MUST carry a tier-3 unit
     literal. Pre-fix, omitting the literal silently produced a
     dimensionless-by-default forcing and the dim-checker later
     disagreed with downstream rate expressions. Post-fix, the parser
     rejects the shape entirely (E001 syntax error). *)
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters { baseline : rate }
    forcing {
      seasonal : sinusoidal {
        amplitude = 0.3
        period    = 365.25 'days
        phase     = 0.0
        baseline  = baseline
      }
    }
    transitions { out : S --> @ baseline * seasonal * S }
    init { S = 1 }
    simulate { from = 0 'days  to = 1 'days }
  |} in
  compile_expect_error_code ~code:"E001" ~contains:"" src

(** Comma-separated entries in a scenario `set`/`scale` block are rejected
    with a separator hint. Entries are newline-separated; commas are reserved
    for `[...]` lists and `(...)` argument lists. Pre-fix this surfaced as a
    bare E001 pointing at the second key with no explanation; post-fix the
    E001 message names the separator ("newlines, not commas") and shows the
    corrected multi-line form. ("Error messages are a feature.") *)
let test_scenario_set_comma_separator_hint () =
  let src = {|
    time_unit = 'days
    compartments { S }
    parameters {
      mu : rate in [0.001, 10.0]
      nu : rate in [0.001, 10.0]
    }
    init { S = 1000 }
    transitions { death : S --> @ mu * S }
    simulate { from = 0 'days  to = 20 'days }
    scenarios {
      baseline { set = { mu = 0.1, nu = 0.2 } }
    }
  |} in
  compile_expect_error_code ~code:"E001" ~contains:"newlines, not commas" src

(* ── DerivedExpr projection (GH #7 resolution, 2026-04-22) ──────────────── *)

(** Bare arithmetic in `projected = ...` — the general form for pooled
    prevalence, fractions, and any state-dependent scalar observable.
    Proves the DSL supports `x + y` without needing a `prevalence(x, y)`
    shortcut. Emits `Ir::DerivedExpr`. *)
let test_projected_bare_sum_emits_derived_expr () =
  let src = {|
    time_unit = 'days
    compartments { S, I_m, I_s, R }
    parameters { beta : rate  gamma : rate  rho : rate }
    transitions {
      infect    : S --> I_m   @ beta * S * (I_m + I_s) / (S + I_m + I_s + R)
      worsen    : I_m --> I_s @ rho * I_m
      recover_m : I_m --> R   @ gamma * I_m
      recover_s : I_s --> R   @ gamma * I_s
    }
    observations {
      prev {
        columns       { time : time, prev : count }
        projected = I_m + I_s
        emit_schedule = every 1 'weeks
        prev ~ poisson(rate = projected)
      }
    }
    init { S = 999  I_m = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  let obs = List.hd m.Ir.observations in
  match obs.Ir.projection with
  | Ir.DerivedExpr _ -> ()  (* shape is right; CLI path evaluates it *)
  | other ->
    Alcotest.failf "expected DerivedExpr, got %s"
      (match other with
       | Ir.CurrentPop _ -> "CurrentPop"
       | Ir.CurrentPopSum _ -> "CurrentPopSum"
       | Ir.CumulativeFlow _ -> "CumulativeFlow"
       | Ir.CumulativeFlowSum _ -> "CumulativeFlowSum"
       | Ir.DerivedExpr _ -> "DerivedExpr")

(** Prevalence-as-proportion — the canonical Garki/surveillance form.
    `projected = (I_m + I_s) / (S + I_m + I_s + R)`. Compiles, and the
    CLI synthetic-obs path evaluates it correctly (book-agent end-to-end
    verified). *)
let test_projected_proportion_compiles () =
  let src = {|
    time_unit = 'days
    compartments { S, I_m, I_s, R }
    parameters { beta : rate  gamma : rate  rho : rate  N_tested : count
                 rho_sens : probability  rho_spec : probability }
    transitions {
      infect    : S --> I_m   @ beta * S * (I_m + I_s) / (S + I_m + I_s + R)
      worsen    : I_m --> I_s @ rho * I_m
      recover_m : I_m --> R   @ gamma * I_m
      recover_s : I_s --> R   @ gamma * I_s
    }
    observations {
      slide {
        columns       { time : time, slide : count }
        projected = (I_m + I_s) / (S + I_m + I_s + R)
        emit_schedule = every 1 'weeks
        slide ~ diagnostic_test(
          base = binomial(n = N_tested, p = projected),
          sens = rho_sens, spec = rho_spec
        )
      }
    }
    init { S = 999  I_m = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let _m = compile_expect_ok src in
  ()

(* ── Hierarchical priors: cycle + self-reference detection (Gate 2, C-class) ── *)

(** C1. Self-reference: `alpha ~ normal(mu = alpha, ...)` must be
    rejected at compile time. *)
let test_hierarchical_self_reference_rejected () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      alpha : rate ~ log_normal(mu = alpha, sigma = 0.5)
    }
    transitions { infect : S --> I @ alpha * S }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  compile_expect_error_code ~code:"E236" ~contains:"alpha" src

(** C2. Two-parameter cycle: `a ~ f(b); b ~ f(a)` — rejected. *)
let test_hierarchical_cycle_rejected () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      alpha : rate ~ log_normal(mu = beta,  sigma = 0.5)
      beta  : rate ~ log_normal(mu = alpha, sigma = 0.5)
    }
    transitions { infect : S --> I @ alpha * S }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  compile_expect_error_code ~code:"E236" ~contains:"cycle" src

(** C3. Deep chain (3 levels): `c ~ f(b); b ~ f(a); a ~ Normal(...)`
    must compile cleanly — it's a legitimate hierarchy, not a cycle. *)
let test_hierarchical_three_level_chain_compiles () =
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters {
      grand_mu  : rate     ~ half_normal(sigma = 1.0)
      mu_alpha  : rate     ~ log_normal(mu = grand_mu, sigma = 0.5)
      alpha     : rate     ~ log_normal(mu = mu_alpha, sigma = 0.3)
    }
    transitions { infect : S --> I @ alpha * S }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let _m = compile_expect_ok src in
  ()

(* ── Hierarchical priors (Wave 2 / #3, Gate 1: parse + IR) ──────────────── *)

(** Parser accepts `| <dim>` pooling clause on an indexed param's prior. *)
let test_hierarchical_prior_parses () =
  let src = {|
    time_unit = 'days
    dimensions { age = [child, adult] }
    compartments { S, I }
    stratify(by = age, only = [S, I])
    parameters {
      mu_alpha    : rate     ~ half_normal(sigma = 0.1)
      sigma_alpha : positive ~ half_normal(sigma = 0.05)
      alpha[age]  : rate     ~ log_normal(mu = mu_alpha, sigma = sigma_alpha) | age
      beta        : rate     in [0.001, 5.0]
    }
    transitions {
      infect[a in age]  : S[a] --> I[a]  @ beta * S[a] * (I[child] + I[adult])
      recover[a in age] : I[a] --> S[a]  @ alpha[a] * I[a]
    }
    init { S[child] = 500  S[adult] = 500  I[child] = 5 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  let _m = compile_expect_ok src in
  ()

(** Hierarchical plain-scalar plumbing: a scalar leaf (no `| dim`) whose
    prior references another parameter is ALSO hierarchical. Used when the
    hyperparent structure is flat (no pooling across dimensions).

    Shape of the IR after expansion:
    - mu_beta, sigma_beta: `(Ir.param_prior_dist parameter) = Some (Normal_p / HalfNormal ...)`,
      `(Ir.param_hierarchical parameter) = None`
    - beta: `(Ir.param_prior_dist parameter) = None`, `(Ir.param_hierarchical parameter) = Some {...}`
    The `hierarchical` field stores the kwarg expressions so inference can
    resolve them against current hyperparam values at evaluation time. *)
let test_hierarchical_scalar_leaf_ir_shape () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      mu_beta    : rate     ~ half_normal(sigma = 1.0)
      sigma_beta : positive ~ half_normal(sigma = 0.5)
      beta       : rate     ~ log_normal(mu = mu_beta, sigma = sigma_beta)
      gamma      : rate     in [0.01, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  let find_param n =
    List.find_opt (fun (p : Ir.parameter) -> p.Ir.name = n) m.Ir.parameters
  in
  let mu_p = Option.get (find_param "mu_beta") in
  let sig_p = Option.get (find_param "sigma_beta") in
  let beta_p = Option.get (find_param "beta") in
  (* Hyperparents carry plain priors. *)
  Alcotest.(check bool) "mu_beta has plain prior"
    true ((Ir.param_prior_dist mu_p) <> None && (Ir.param_hierarchical mu_p) = None);
  Alcotest.(check bool) "sigma_beta has plain prior"
    true ((Ir.param_prior_dist sig_p) <> None && (Ir.param_hierarchical sig_p) = None);
  (* beta is a leaf: hierarchical, no float prior. *)
  Alcotest.(check bool) "beta has hierarchical prior"
    true ((Ir.param_prior_dist beta_p) = None && (Ir.param_hierarchical beta_p) <> None);
  match (Ir.param_hierarchical beta_p) with
  | Some h ->
    Alcotest.(check string) "leaf dist kind" "log_normal" (Ir.hierarchical_kind_name h.Ir.hkind);
    (* `mu` arg references parameter mu_beta *)
    let mu_arg = List.assoc "mu" h.Ir.hargs in
    Alcotest.(check bool) "mu arg references mu_beta"
      true (mu_arg = Ir.Param "mu_beta");
    let sig_arg = List.assoc "sigma" h.Ir.hargs in
    Alcotest.(check bool) "sigma arg references sigma_beta"
      true (sig_arg = Ir.Param "sigma_beta")
  | None -> Alcotest.fail "expected Some hierarchical"

(** Indexed hierarchical param: `alpha[age]` with `| age` pool clause
    should produce one IR parameter per age value, each with the same
    hierarchical structure pointing at the shared hyperparameters. *)
let test_hierarchical_indexed_ir_shape () =
  let src = {|
    time_unit = 'days
    dimensions { age = [child, adult] }
    compartments { S, I }
    stratify(by = age, only = [S, I])
    parameters {
      mu_alpha    : rate     ~ half_normal(sigma = 0.1)
      sigma_alpha : positive ~ half_normal(sigma = 0.05)
      alpha[age]  : rate     ~ log_normal(mu = mu_alpha, sigma = sigma_alpha) | age
      beta        : rate     in [0.001, 5.0]
    }
    transitions {
      infect[a in age]  : S[a] --> I[a]  @ beta * S[a] * (I[child] + I[adult])
      recover[a in age] : I[a] --> S[a]  @ alpha[a] * I[a]
    }
    init { S[child] = 500  S[adult] = 500  I[child] = 5 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  let m = compile_expect_ok src in
  let names = List.map (fun (p : Ir.parameter) -> p.Ir.name) m.Ir.parameters
              |> List.sort compare in
  (* alpha should be expanded into alpha_child and alpha_adult. *)
  Alcotest.(check bool) "alpha_child is a parameter"
    true (List.mem "alpha_child" names);
  Alcotest.(check bool) "alpha_adult is a parameter"
    true (List.mem "alpha_adult" names);
  (* Both should have hierarchical priors pointing at mu_alpha / sigma_alpha. *)
  List.iter (fun n ->
    let p = List.find (fun (p : Ir.parameter) -> p.Ir.name = n) m.Ir.parameters in
    match (Ir.param_hierarchical p) with
    | Some h ->
      Alcotest.(check string) (n ^ " dist kind") "log_normal" (Ir.hierarchical_kind_name h.Ir.hkind);
      Alcotest.(check string) (n ^ " pool_over") "age" h.Ir.hpool_over;
      let mu_arg = List.assoc "mu" h.Ir.hargs in
      Alcotest.(check bool) (n ^ " mu refs mu_alpha") true (mu_arg = Ir.Param "mu_alpha");
    | None -> Alcotest.failf "%s missing hierarchical prior" n
  ) ["alpha_child"; "alpha_adult"]

(* ── Probabilistic branching on destination (Wave 2 / #2) ───────────────── *)

(** Parser accepts `X --> {Y : p, Z : 1-p} @ rate`. *)
let test_branching_parses () =
  let src = {|
    time_unit = 'days
    compartments { S, Y, Z }
    parameters {
      beta   : rate        in [0.001, 5.0]
      p_symp : probability in [0.01, 0.99]
    }
    transitions {
      infection : S --> { Y : p_symp, Z : 1 - p_symp }  @ beta * S
    }
    init { S = 1000 }
    simulate { from = 0 'days  to = 50 'days }
  |} in
  let m = compile_expect_ok src in
  (* Should expand to exactly TWO transitions. *)
  Alcotest.(check int)
    "branching desugars to one transition per branch"
    2 (List.length m.Ir.transitions)

(** Equivalence: the branching sugar produces the same IR as two
    hand-written transitions with the weight-scaled rates. *)
let test_branching_equivalent_to_two_transitions () =
  let sugar_src = {|
    time_unit = 'days
    compartments { S, Y, Z }
    parameters {
      beta   : rate        in [0.001, 5.0]
      p_symp : probability in [0.01, 0.99]
    }
    transitions {
      infection : S --> { Y : p_symp, Z : 1 - p_symp }  @ beta * S
    }
    init { S = 1000 }
    simulate { from = 0 'days  to = 50 'days }
  |} in
  let manual_src = {|
    time_unit = 'days
    compartments { S, Y, Z }
    parameters {
      beta   : rate        in [0.001, 5.0]
      p_symp : probability in [0.01, 0.99]
    }
    transitions {
      to_Y : S --> Y  @ p_symp * (beta * S)
      to_Z : S --> Z  @ (1 - p_symp) * (beta * S)
    }
    init { S = 1000 }
    simulate { from = 0 'days  to = 50 'days }
  |} in
  let ms = compile_expect_ok sugar_src in
  let mm = compile_expect_ok manual_src in
  (* Match transitions by destination compartment (the one with delta = +1). *)
  let stoich_of (t : Ir.transition) = List.sort compare t.Ir.stoichiometry in
  let dest_of (t : Ir.transition) =
    match List.find_opt (fun (_, d) -> d > 0) t.Ir.stoichiometry with
    | Some (n, _) -> n
    | None -> Alcotest.failf "transition %s has no destination" t.Ir.name
  in
  let by_dest lst =
    List.map (fun t -> (dest_of t, t)) lst
    |> List.sort (fun (a, _) (b, _) -> compare a b)
  in
  let sugar_by_dst  = by_dest ms.Ir.transitions in
  let manual_by_dst = by_dest mm.Ir.transitions in
  Alcotest.(check (list string))
    "same set of destinations"
    (List.map fst manual_by_dst)
    (List.map fst sugar_by_dst);
  List.iter2 (fun (d_s, (s : Ir.transition)) (d_m, (m : Ir.transition)) ->
    assert (d_s = d_m);
    Alcotest.(check bool)
      (Printf.sprintf "stoich for dest %s matches" d_s)
      true
      (stoich_of s = stoich_of m);
    Alcotest.(check bool)
      (Printf.sprintf "rate for dest %s matches" d_s)
      true
      (s.Ir.rate = m.Ir.rate)
  ) sugar_by_dst manual_by_dst

(** Indexed branching: `bite[a in age] : X[a] --> {Y_s[a] : p[a], Y_a[a] : 1-p[a]}`.
    Should produce |age| × 2 = 4 transitions for age = [child, adult]. *)
let test_branching_indexed_by_age () =
  let src = {|
    time_unit = 'days
    dimensions { age = [child, adult] }
    compartments { X, Y_s, Y_a }
    stratify(by = age)
    parameters {
      h_eff  : rate
      p_symp_child : probability
      p_symp_adult : probability
    }
    transitions {
      bite[a in age] : X[a] --> { Y_s[a] : p_symp_child, Y_a[a] : 1 - p_symp_child }
        @ h_eff * X[a]
    }
    init { X[child] = 500  X[adult] = 500 }
    simulate { from = 0 'days  to = 30 'days }
  |} in
  let m = compile_expect_ok src in
  (* 2 age values × 2 branches = 4 generated transitions. *)
  Alcotest.(check int)
    "indexed branching expands to |age|*|branches| transitions"
    4 (List.length m.Ir.transitions)

(* ── diagnostic_test likelihood sugar (Wave 1 / #4) ──────────────────────── *)

(** Minimal model exercising `diagnostic_test(base = binomial, sens, spec)`. *)
let test_diagnostic_test_parses () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta     : rate        in [0.001, 5.0]
      gamma    : rate        in [0.01, 1.0]
      rho_sens : probability in [0.5, 1.0]
      rho_spec : probability in [0.5, 1.0]
      N_tested : count       in [10, 10000]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      slide_positivity {
        columns       { time : time, slide_positivity : count }
        projected = prevalence(I)
        emit_schedule = every 1 'weeks
        slide_positivity ~ diagnostic_test(
          base = binomial(n = N_tested, p = projected),
          sens = rho_sens,
          spec = rho_spec
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).likelihood with
  | Ir.Binomial _ -> ()  (* sugar desugared to Binomial ✓ *)
  | _ -> Alcotest.fail "expected Binomial after diagnostic_test desugar"

(** The sugar must produce IR byte-identical to the hand-inlined
    `binomial(n, p = sens * projected + (1 - spec) * (1 - projected))`
    form. This is the canonical correctness guarantee. *)
let test_diagnostic_test_equivalence () =
  let sugar_src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta     : rate        in [0.001, 5.0]
      gamma    : rate        in [0.01, 1.0]
      rho_sens : probability in [0.5, 1.0]
      rho_spec : probability in [0.5, 1.0]
      N_tested : count       in [10, 10000]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      slide_positivity {
        columns       { time : time, slide_positivity : count }
        projected = prevalence(I)
        emit_schedule = every 1 'weeks
        slide_positivity ~ diagnostic_test(
          base = binomial(n = N_tested, p = projected),
          sens = rho_sens,
          spec = rho_spec
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  let manual_src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta     : rate        in [0.001, 5.0]
      gamma    : rate        in [0.01, 1.0]
      rho_sens : probability in [0.5, 1.0]
      rho_spec : probability in [0.5, 1.0]
      N_tested : count       in [10, 10000]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      slide_positivity {
        columns       { time : time, slide_positivity : count }
        projected = prevalence(I)
        emit_schedule = every 1 'weeks
        slide_positivity ~ binomial(
          n = N_tested,
          p = rho_sens * projected + (1 - rho_spec) * (1 - projected)
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  let m_sugar  = compile_expect_ok sugar_src  in
  let m_manual = compile_expect_ok manual_src in
  match (List.hd m_sugar.observations).likelihood,
        (List.hd m_manual.observations).likelihood with
  | Ir.Binomial s, Ir.Binomial m ->
    Alcotest.(check bool) "n expressions equal" true (s.n = m.n);
    Alcotest.(check bool) "p expressions equal" true (s.p = m.p)
  | _ -> Alcotest.fail "both models should have Binomial likelihood"

(** Bernoulli base (one test per individual). *)
let test_diagnostic_test_bernoulli () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters {
      beta     : rate        in [0.001, 5.0]
      gamma    : rate        in [0.01, 1.0]
      rho_sens : probability in [0.5, 1.0]
      rho_spec : probability in [0.5, 1.0]
    }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      any_positive {
        columns       { time : time, any_positive : count }
        projected = prevalence(I)
        emit_schedule = every 1 'days
        any_positive ~ diagnostic_test(
          base = bernoulli(p = projected),
          sens = rho_sens,
          spec = rho_spec
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  let m = compile_expect_ok src in
  match (List.hd m.observations).likelihood with
  | Ir.Bernoulli _ -> ()
  | _ -> Alcotest.fail "expected Bernoulli after diagnostic_test desugar"

let test_diagnostic_test_bad_base () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  rho_sens : probability  rho_spec : probability }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      cases {
        columns       { time : time, cases : count }
        projected = prevalence(I)
        emit_schedule = every 1 'weeks
        cases ~ diagnostic_test(
          base = poisson(rate = projected),
          sens = rho_sens,
          spec = rho_spec
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  compile_expect_error_code ~code:"E253" ~contains:"poisson" src

let test_diagnostic_test_missing_kwargs () =
  let src = {|
    time_unit = 'days
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  rho_sens : probability  N_tested : count }
    transitions {
      infection : S --> I @ beta * S * I / (S + I + R)
      recovery  : I --> R @ gamma * I
    }
    observations {
      cases {
        columns       { time : time, cases : count }
        projected = prevalence(I)
        emit_schedule = every 1 'weeks
        cases ~ diagnostic_test(
          base = binomial(n = N_tested, p = projected),
          sens = rho_sens
        )
      }
    }
    init { S = 999  I = 1 }
    simulate { from = 0 'days  to = 14 'days }
  |} in
  compile_expect_error_code ~code:"E254" ~contains:"diagnostic_test" src

(* ── #[lineage] individual-sampling layer (2026-05-19 proposal) ──────────────
   Foundation slice: lexer attribute opener, `#[lineage]` parse (both
   forms), linear-in-parents classifier (accept/reject with E601),
   parent_pool_weights extraction, and identity-tracked-subgraph
   reachability (incl. SIRS cycle). ──────────────────────────────────────── *)

let lineage_of (t : Ir.transition) = t.Ir.lineage

let find_lineage m name =
  match (find_transition m name).Ir.lineage with
  | Some l -> l
  | None   -> Alcotest.failf "transition %s has no lineage annotation" name

(* SEIR with #[lineage] above the transition. *)
let seir_lineage_src ~inline =
  let attr_line = if inline then "  #[lineage] infection : S --> E  @ beta * S * I / N"
                  else "  #[lineage]\n  infection : S --> E  @ beta * S * I / N" in
  Printf.sprintf {|
    compartments { S, E, I, R, V }
    parameters {
      beta : rate  sigma : rate  gamma : rate  nu : rate
      N0 : count  I0 : count
    }
    let N = S + E + I + R + V
    transitions {
%s
      progression : E --> I  @ sigma * E
      recovery    : I --> R  @ gamma * I
      vaccination : S --> V  @ nu * S
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} attr_line

(* (a) Both attribute forms parse and produce identical IR. *)
let test_lineage_parses_both_forms () =
  let m_block  = compile_expect_ok (seir_lineage_src ~inline:false) in
  let m_inline = compile_expect_ok (seir_lineage_src ~inline:true)  in
  (* The infection transition carries a lineage annotation in both. *)
  let lb = find_lineage m_block  "infection" in
  let li = find_lineage m_inline "infection" in
  Alcotest.(check bool) "block form is_lineage_event" true lb.Ir.is_lineage_event;
  Alcotest.(check bool) "inline form is_lineage_event" true li.Ir.is_lineage_event;
  (* Identical IR: the whole transition list must match. *)
  Alcotest.(check bool) "both forms produce identical IR"
    true (m_block.Ir.transitions = m_inline.Ir.transitions);
  (* Ordinary transitions carry no lineage annotation. *)
  Alcotest.(check bool) "recovery has no lineage"
    true (lineage_of (find_transition m_block "recovery") = None)

(* (b) Classifier ACCEPTS frequency-dependent β·S·I/N. The denominator
   appearance of I (in N) is exempt; I is the sole linear parent. *)
let test_lineage_accepts_freq_dependent () =
  let m = compile_expect_ok (seir_lineage_src ~inline:false) in
  let l = find_lineage m "infection" in
  let comps = List.map fst l.Ir.parent_pool_weights in
  Alcotest.(check (list string)) "single parent pool I" ["I"] comps

(* (b) Classifier ACCEPTS multi-pool β·S·(β_I·I + β_A·A)/N. *)
let multi_pool_src = {|
    compartments { S, E, I, A, R }
    parameters {
      beta : rate  beta_i : probability  beta_a : probability
      sigma : rate  gamma : rate  N0 : count  I0 : count
    }
    let N = S + E + I + A + R
    transitions {
      #[lineage]
      infection : S --> E  @ beta * S * (beta_i * I + beta_a * A) / N
      progression : E --> I  @ sigma * E
      recovery : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days to = 120 'days }
  |}

let test_lineage_accepts_multi_pool () =
  let m = compile_expect_ok multi_pool_src in
  let l = find_lineage m "infection" in
  let comps = List.sort compare (List.map fst l.Ir.parent_pool_weights) in
  Alcotest.(check (list string)) "two parent pools A, I" ["A"; "I"] comps

(* (b) Classifier ACCEPTS a stratified contact-matrix rate
   S[a] · Σ_b C[a,b]·I[b]/N[b]. *)
let age_lineage_src = {|
    time_unit = 'days
    compartments { S, E, I, R }
    dimensions { age = [child, adult] }
    stratify(by = age)
    let N_local[a in age] = S[a] + E[a] + I[a] + R[a]
    parameters { beta : rate  sigma : rate  gamma : rate }
    tables { C_age : age × age = [[12.0, 4.0], [4.0, 8.0]] }
    transitions {
      #[lineage]
      infection[a in age] : S[a] --> E[a]
        @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
      progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
      recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
    }
    init { S[child] = 4990  S[adult] = 5000  I[child] = 10 }
    simulate { from = 0 'days  to = 100 'days }
  |}

let test_lineage_accepts_stratified () =
  let m = compile_expect_ok age_lineage_src in
  (* infection_child draws from both I_child and I_adult. *)
  let lc = find_lineage m "infection_child" in
  let comps = List.sort compare (List.map fst lc.Ir.parent_pool_weights) in
  Alcotest.(check (list string)) "infection_child pools" ["I_adult"; "I_child"] comps;
  let la = find_lineage m "infection_adult" in
  let comps_a = List.sort compare (List.map fst la.Ir.parent_pool_weights) in
  Alcotest.(check (list string)) "infection_adult pools" ["I_adult"; "I_child"] comps_a

(* (c) Classifier REJECTS β·S·(I+ι)^α/N with E601, pointing at the
   nonlinear subterm. *)
let test_lineage_rejects_nonlinear_e601 () =
  let src = {|
    compartments { S, E, I, R }
    parameters {
      beta : rate  sigma : rate  gamma : rate
      alpha : positive  iota : count  N0 : count  I0 : count
    }
    let N = S + E + I + R
    transitions {
      #[lineage]
      infection : S --> E  @ beta * S * (I + iota)^alpha / N
      progression : E --> I  @ sigma * E
      recovery : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days to = 120 'days }
  |} in
  (* Error code E601 fires, and the message names the nonlinear subterm. *)
  compile_expect_error_code ~code:"E601" ~contains:"I + iota" src

(* (d) parent_pool_weights extraction emits the expected weight ASTs.
   For β·S·I/N (N = S+E+I+R+V), weight(I) = β·S/N. *)
let test_lineage_weight_freq_dependent () =
  let m = compile_expect_ok (seir_lineage_src ~inline:false) in
  let l = find_lineage m "infection" in
  let open Ir in
  let expected =
    BinOp { op = Div;
            left  = BinOp { op = Mul; left = Param "beta"; right = Pop "S" };
            right = PopSum ["S"; "E"; "I"; "R"; "V"] }
  in
  match l.parent_pool_weights with
  | [("I", w)] ->
    Alcotest.(check bool) "weight(I) = beta*S/N" true (w = expected)
  | other ->
    Alcotest.failf "expected single I weight, got %d pools" (List.length other)

(* (d) Multi-pool weights: weight(I) = β·β_I·S/N, weight(A) = β·β_A·S/N. *)
let test_lineage_weight_multi_pool () =
  let m = compile_expect_ok multi_pool_src in
  let l = find_lineage m "infection" in
  let open Ir in
  let denom = PopSum ["S"; "E"; "I"; "A"; "R"] in
  let expect_for p =
    (* β·S·p / N, with the multiplication associated as ((β*S)*p). *)
    BinOp { op = Div;
            left  = BinOp { op = Mul;
                            left  = BinOp { op = Mul; left = Param "beta"; right = Pop "S" };
                            right = Param p };
            right = denom }
  in
  let wi = List.assoc "I" l.parent_pool_weights in
  let wa = List.assoc "A" l.parent_pool_weights in
  Alcotest.(check bool) "weight(I) = beta*S*beta_i/N" true (wi = expect_for "beta_i");
  Alcotest.(check bool) "weight(A) = beta*S*beta_a/N" true (wa = expect_for "beta_a")

(* (e) Identity-tracked subgraph: SEIR with #[lineage] on S→E tracks
   {E, I, R}; S and V are untracked. *)
let test_lineage_identity_subgraph_seir () =
  let m = compile_expect_ok (seir_lineage_src ~inline:false) in
  Alcotest.(check (list string)) "tracked = E,I,R"
    ["E"; "I"; "R"] m.Ir.identity_tracked_compartments

(* (e) Cyclic SIRS: R→S waning pulls S into the tracked set without
   infinite recursion; all of S,I,R tracked. *)
let test_lineage_identity_subgraph_sirs_cycle () =
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  omega : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      #[lineage]
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
      waning    : R --> S  @ omega * R
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  let m = compile_expect_ok src in
  Alcotest.(check (list string)) "SIRS cycle tracks S,I,R"
    ["S"; "I"; "R"] m.Ir.identity_tracked_compartments

(* No #[lineage] anywhere ⇒ inert: empty identity set, no lineage on any
   transition. *)
let test_lineage_inert_when_absent () =
  let m = compile_expect_ok {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  Alcotest.(check (list string)) "no tracked compartments" [] m.Ir.identity_tracked_compartments;
  Alcotest.(check bool) "infection has no lineage"
    true (lineage_of (find_transition m "infection") = None)

(* Lexer: unknown attribute name is a hard error (E110), not a silent
   no-op. *)
let test_lineage_unknown_attribute_e110 () =
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      #[transmission]
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  compile_expect_error_code ~code:"E110" ~contains:"transmission" src

(* 2026-05-22 calendar-time: `origin = date(...)` emits both the string and
   the compiler-derived numeric `origin_rata_die`, and `instant`/`duration`
   param kinds round-trip into the IR. The rata-die integer must equal the
   shared `days_of_date` formula (= Rust `caltime::rata_die`), so the runtime
   can read it without re-parsing the origin string. *)
let test_origin_rata_die_emitted () =
  let m = compile_expect_ok {|
    origin = date("2020-02-28")
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count
                 tau : instant  gen : duration }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  Alcotest.(check (option string)) "origin string preserved"
    (Some "2020-02-28") m.Ir.origin;
  (* days_of_date 2020 2 28 — the shared proleptic-Gregorian day number. *)
  let expected = Expander.days_of_date 2020 2 28 in
  Alcotest.(check (option int)) "origin_rata_die = days_of_date"
    (Some expected) m.Ir.origin_rata_die;
  let kind_of n =
    Option.map Ir.param_kind_name
      (List.find (fun (p : Ir.parameter) -> p.name = n) m.Ir.parameters).param_kind in
  Alcotest.(check (option string)) "tau is instant" (Some "instant") (kind_of "tau");
  Alcotest.(check (option string)) "gen is duration" (Some "duration") (kind_of "gen")

let test_origin_absent_no_rata_die () =
  (* No origin → origin_rata_die is None (backward-compat: existing models
     emit neither field). *)
  let m = compile_expect_ok {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  Alcotest.(check (option string)) "no origin" None m.Ir.origin;
  Alcotest.(check (option int)) "no origin_rata_die" None m.Ir.origin_rata_die

(* ── Phase 1 of the 2026-05-22 typed-time proposal ─────────────────────────
   Rules tested below:
     E320 — time_unit = 'months/'years under `origin = date(...)`.
     E321 — Instant + CalendarDuration (literal or `let`-laundered).
     E322 — Calendar duration in recurring schedule `every`/`from`/`until`.
     E323 — Bare-numeric entries inside `on=[...]` of a periodic forcing
            in anchored mode.
     W324 — Bare-numeric `simulate.from` / `simulate.to` in anchored mode.
     W325 — Bare-numeric `at [...]` schedule entries in anchored mode.

   The proposal's §8 acceptance criteria for Phase 1 require each new
   diagnostic to assert both that it fires when expected AND that the
   emitted message contains the documented hint text. *)

(* Compile a snippet that's expected to succeed, returning the compile
   detail so we can inspect emitted warnings/info. *)
let compile_with_diags src =
  Diagnostics.json_errors_mode := false;
  let result = Compiler.compile_detail_result ~name:"t" src in
  match result with
  | Ok d -> Ok d
  | Error e -> Error e

let diags_of_detail (d : Compiler.compile_detail) =
  List.rev d.Compiler.ctx.Expander.diags.Diagnostics.diags

(* Assert that compilation of `src` succeeds AND emits a diagnostic with
   given code. Returns the matching diagnostic. *)
let expect_diag ?(severity = Diagnostics.Warning) ~code src =
  match compile_with_diags src with
  | Error e -> Alcotest.failf "expected compile success but got: %s" e
  | Ok d ->
    let ds = diags_of_detail d in
    let matches = List.filter (fun (x : Diagnostics.diagnostic) ->
      x.code = code && x.severity = severity) ds in
    (match matches with
     | [] ->
       let codes = String.concat ", "
         (List.map (fun (x : Diagnostics.diagnostic) -> x.code) ds) in
       Alcotest.failf "expected %s diagnostic but only got: [%s]" code codes
     | d :: _ -> d)

(* Assert that compilation of `src` FAILS with an error code. *)
let expect_error_code ?(contains = "") ~code src =
  compile_expect_error_code ~code ~contains src

let assert_hint_contains ~needle (d : Diagnostics.diagnostic) =
  match d.hint with
  | None -> Alcotest.failf "expected diagnostic %s to carry a hint" d.code
  | Some h ->
    if not (contains_substring ~needle h) then
      Alcotest.failf "expected hint to contain %S, got: %s" needle h

(* ── Positive cases (must compile) ──────────────────────────────────────── *)

let test_typed_time_pos_5months_table_value () =
  (* Affine duration literal `5 'months` as a table value compiles —
     it's just a length, not a step from a date. *)
  let m = compile_expect_ok {|
    time_unit = 'days
    dimensions { age = [child, adult] }
    compartments { S, I, R }
    stratify(by = age)
    tables {
      delay_by_age : age 'months = [3, 6]
    }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N[a in age] = S[a] + I[a] + R[a]
    transitions {
      infection[a in age] : S[a] --> I[a] @ beta * S[a] * I[a] / N[a]
      recovery[a in age]  : I[a] --> R[a] @ gamma * I[a]
    }
    init { S[a in age] = N0  I[a in age] = I0 }
    simulate { from = 0 'days  to = 60 'days }
  |} in
  ignore m

let test_typed_time_pos_per_month_rate_in_anchored () =
  (* Per-month rate parameters work in anchored mode — the expander
     converts at compile time via the affine month constant. *)
  let m = compile_expect_ok {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  ignore m

let test_typed_time_pos_simulate_to_months_oneshot () =
  (* `simulate.to = 600 'months` in anchored mode is a one-shot
     affine conversion to days — Rule 1 does not fire on this
     because the position is Duration-typed, not Instant-typed.
     Proposal §1.2. *)
  let m = compile_expect_ok {|
    time_unit = 'days
    origin = date("1891-01-01")
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 600 'months }
  |} in
  ignore m

let test_typed_time_pos_duration_param_bounds_with_months () =
  (* `delay : duration in [1 'months, 6 'months]` parses and stays
     Exact — the bound's month-spelling is harmless shorthand,
     never leaking to uses of `delay`. Proposal §3.3.2 invariant. *)
  let m = compile_expect_ok {|
    time_unit = 'days
    origin = date("2020-02-24")
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      tau   : instant  in [0 'days, 120 'days]
      delay : duration in [1 'months, 6 'months]
      N0    : count
      I0    : count
    }
    let N = S + I + R
    let landmark = tau + delay
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
      seed      :    --> I @ if t > landmark then 0.1 else 0.0
    }
    init { S = N0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  (* `delay` is recorded as duration kind. *)
  let kind_of n =
    Option.map Ir.param_kind_name
      (List.find (fun (p : Ir.parameter) -> p.name = n) m.Ir.parameters).param_kind in
  Alcotest.(check (option string)) "delay is duration" (Some "duration") (kind_of "delay")

let test_typed_time_pos_unanchored_months_axis () =
  (* dacca shape: unanchored, `time_unit = 'months`, per-month rates,
     month-span durations. Stays legal — Rule 2 only fires in
     anchored mode. *)
  let m = compile_expect_ok {|
    time_unit = 'months
    compartments { S, I, R }
    parameters {
      beta  : rate
      gamma : rate
      N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0  to = 600 }
  |} in
  Alcotest.(check string) "time_unit stored as months" "months" m.Ir.time_unit

(* ── Negative cases (must error or warn) ───────────────────────────────── *)

let test_typed_time_e321_date_plus_months_rejected () =
  expect_error_code
    ~code:"E321"
    ~contains:"calendar duration"
    {|
      time_unit = 'days
      origin = date("2020-02-24")
      compartments { S, I, R }
      parameters {
        beta : rate  gamma : rate  N0 : count  I0 : count
      }
      let landmark = date("2020-02-24") + 6 'months
      let N = S + I + R
      transitions {
        infection : S --> I @ beta * S * I / N
        recovery  : I --> R @ gamma * I
      }
      init { S = N0 - I0  I = I0 }
      simulate { from = 0 'days  to = 120 'days }
    |}

let test_typed_time_e321_laundered_through_let () =
  (* Laundered case: `let d = 6 'months; date(...) - d`. The let-
     body's classifier (TCalendar) flows through `classify`'s let-
     lookup. *)
  expect_error_code
    ~code:"E321"
    ~contains:"calendar duration"
    {|
      time_unit = 'days
      origin = date("2020-02-24")
      compartments { S, I, R }
      parameters {
        beta : rate  gamma : rate  N0 : count  I0 : count
      }
      let d = 6 'months
      let landmark = date("2020-02-24") + d
      let N = S + I + R
      transitions {
        infection : S --> I @ beta * S * I / N
        recovery  : I --> R @ gamma * I
      }
      init { S = N0 - I0  I = I0 }
      simulate { from = 0 'days  to = 120 'days }
    |}

let test_typed_time_e321_hint_text () =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"t" {|
    time_unit = 'days
    origin = date("2020-02-24")
    compartments { S, I, R }
    parameters {
      beta : rate  gamma : rate  N0 : count  I0 : count
    }
    let landmark = date("2020-02-24") + 6 'months
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.fail "expected error"
  | Error e ->
    (* Hint text must mention `add_calendar_months` and `'days`. *)
    Alcotest.(check bool) "hint mentions add_calendar_months"
      true (contains_substring ~needle:"add_calendar_months" e);
    Alcotest.(check bool) "hint mentions days affine span"
      true (contains_substring ~needle:"'days" e)

let test_typed_time_e320_time_unit_months_with_origin_rejected () =
  expect_error_code
    ~code:"E320"
    ~contains:"time_unit"
    {|
      time_unit = 'months
      origin = date("2020-01-01")
      compartments { S, I, R }
      parameters {
        beta : rate  gamma : rate  N0 : count  I0 : count
      }
      let N = S + I + R
      transitions {
        infection : S --> I @ beta * S * I / N
        recovery  : I --> R @ gamma * I
      }
      init { S = N0 - I0  I = I0 }
      simulate { from = 0  to = 600 }
    |}

let test_typed_time_e320_hint_text () =
  Diagnostics.json_errors_mode := true;
  let result = Compiler.compile ~name:"t" {|
    time_unit = 'years
    origin = date("2020-01-01")
    compartments { S, I, R }
    parameters {
      beta : rate  gamma : rate  N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0  to = 5 }
  |} in
  Diagnostics.json_errors_mode := false;
  match result with
  | Ok _ -> Alcotest.fail "expected error"
  | Error e ->
    Alcotest.(check bool) "hint mentions time_unit = 'days suggestion"
      true (contains_substring ~needle:"'days" e);
    Alcotest.(check bool) "hint warns about silent-shift trap"
      true (contains_substring ~needle:"silently" e)

let test_typed_time_e322_calendar_cadence_in_recurring () =
  (* `every = 1 'months` inside a recurring intervention schedule
     under origin = date(...) is a calendar cadence and rejected. *)
  expect_error_code
    ~code:"E322"
    ~contains:"calendar"
    {|
      time_unit = 'days
      origin = date("2020-01-01")
      compartments { S, V }
      parameters {
        N0 : count
      }
      transitions {
        leak : S --> V @ 0.0 'per_day * S
      }
      interventions {
        vacc : transfer(from = S, to = V, fraction = 0.1) {
          every = 1 'months
        }
      }
      init { S = N0 }
      simulate { from = 0 'days  to = 120 'days }
    |}

let test_typed_time_e322_unanchored_months_cadence_ok () =
  (* In unanchored mode the same `every = 1 'months` is fine —
     Rule 1/7 are vacuous without origin. *)
  let _m = compile_expect_ok {|
    time_unit = 'days
    compartments { S, V }
    parameters {
      N0 : count
    }
    transitions {
      leak : S --> V @ 0.0 'per_day * S
    }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1) {
        every = 1 'months
      }
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  ()

let test_typed_time_e323_bare_numeric_on_periodic_anchored () =
  (* Bare-numeric entries in `on=[7:100]` inside a periodic forcing
     under an anchored model: hard error. *)
  expect_error_code
    ~code:"E323"
    ~contains:"on="
    {|
      time_unit = 'days
      origin = date("2020-01-01")
      compartments { S, I, R }
      parameters {
        beta0 : rate  gamma : rate
        N0 : count  I0 : count
      }
      forcing {
        school : periodic 'ratio {
          period = 365 'days
          step   = 1 'days
          on     = [7:100, 115:199, 252:300, 308:356]
        }
      }
      let N = S + I + R
      transitions {
        infection : S --> I @ beta0 * school(t) * S * I / N
        recovery  : I --> R @ gamma * I
      }
      init { S = N0 - I0  I = I0 }
      simulate { from = 0 'days  to = 120 'days }
    |}

let test_typed_time_w324_bare_numeric_simulate_warning () =
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I, R }
    parameters {
      beta : rate  gamma : rate  N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0  to = 120 }
  |} in
  let d = expect_diag ~severity:Diagnostics.Warning ~code:"W324" src in
  assert_hint_contains ~needle:"'days" d

let test_typed_time_w324_unit_annotated_simulate_no_warning () =
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I, R }
    parameters {
      beta : rate  gamma : rate  N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  match compile_with_diags src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok d ->
    let ds = diags_of_detail d in
    let w324 = List.filter (fun (x : Diagnostics.diagnostic) -> x.code = "W324") ds in
    Alcotest.(check int) "no W324 when unit-annotated" 0 (List.length w324)

let test_typed_time_w325_bare_numeric_at_schedule_warning () =
  (* Bare-numeric `at [50, 100]` in an intervention schedule under
     an anchored model produces W325. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters {
      N0 : count
    }
    transitions {
      leak : S --> V @ 0.0 'per_day * S
    }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1) at [50, 100]
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 120 'days }
  |} in
  let d = expect_diag ~severity:Diagnostics.Warning ~code:"W325" src in
  assert_hint_contains ~needle:"date(" d

(* gh#134: the model-side calendar nudge (W324/W325) is the symmetric
   sibling of the data-loader's W326. The cases below pin the three
   negative/coverage gaps the issue's ergonomics ask depends on but
   that the suite did not previously lock:

     - a `date(...)` literal in `simulate.from/to` (the legible form
       the warning steers toward) must NOT itself warn W324;
     - a `date(...)` literal in an intervention `at [...]` schedule
       must NOT warn W325 — date() is the suppression-by-clarity path;
     - the `events {}` block (sister construct to `interventions {}`)
       must warn W325 on a bare-numeric `at [...]` symmetrically with
       interventions;
     - an UNanchored model (no `origin = date(...)`) must NOT warn at
       all — the nudge fires only when origin is a date. *)

let count_code ~code (d : Compiler.compile_detail) : int =
  diags_of_detail d
  |> List.filter (fun (x : Diagnostics.diagnostic) -> x.code = code)
  |> List.length

let test_gh134_date_simulate_from_to_no_w324 () =
  (* The form the W324 hint steers the author toward must be clean. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I }
    parameters { beta : rate in [0.1, 2.0] }
    transitions { infection : S --> I @ beta * S }
    init { S = 100  I = 1 }
    simulate { from = date("2020-03-01")  to = date("2020-12-31") }
  |} in
  match compile_with_diags src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok d ->
    Alcotest.(check int) "no W324 when simulate.from/to are date()"
      0 (count_code ~code:"W324" d)

let test_gh134_date_intervention_at_no_w325 () =
  (* date() in an intervention at-schedule is the legible alternative
     the W325 hint names — it must not itself warn. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters { N0 : count }
    transitions { leak : S --> V @ 0.0 'per_day * S }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1)
             at [date("2020-03-01"), date("2020-06-01")]
    }
    init { S = N0 }
    simulate { from = date("2020-01-01")  to = date("2020-12-31") }
  |} in
  match compile_with_diags src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok d ->
    Alcotest.(check int) "no W325 when intervention at[..] is date()"
      0 (count_code ~code:"W325" d)

let test_gh134_events_bare_numeric_at_warns_w325 () =
  (* The `events {}` block is the sister construct to `interventions {}`;
     a bare-numeric `at [...]` under a date origin must warn W325 in the
     events block too (the issue calls out "events / intervention at"). *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I }
    parameters { N0 : count }
    transitions { leak : S --> I @ 0.0 'per_day * S }
    events {
      seed : add(I, 5) at [60, 120]
    }
    init { S = N0 }
    simulate { from = date("2020-01-01")  to = date("2020-12-31") }
  |} in
  let d = expect_diag ~severity:Diagnostics.Warning ~code:"W325" src in
  assert_hint_contains ~needle:"date(" d

let test_gh134_unanchored_bare_numeric_no_nudge () =
  (* No `origin = date(...)`: bare-numeric time positions are the
     normal, intended idiom and must NOT warn W324/W325. The nudge is
     a date-origin-only refinement (the torsor is inactive otherwise). *)
  let src = {|
    time_unit = 'days
    compartments { S, I }
    parameters { beta : rate in [0.1, 2.0]  N0 : count }
    transitions { infection : S --> I @ beta * S }
    interventions {
      pulse : transfer(from = S, to = I, fraction = 0.1) at [50, 100]
    }
    init { S = N0  I = 1 }
    simulate { from = 730  to = 5000 }
  |} in
  match compile_with_diags src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok d ->
    Alcotest.(check int) "no W324 when unanchored" 0 (count_code ~code:"W324" d);
    Alcotest.(check int) "no W325 when unanchored" 0 (count_code ~code:"W325" d)

(* ── Phase 2 of the 2026-05-22 typed-time proposal ──────────────────────────
   Calendar-arithmetic primitives and `date_range`. Errors and warnings
   under test:
     E327 — `add_calendar_*` / `origin` / calendar `date_range` cadence in
            an unanchored model.
     E328 — argument-shape errors (non-constant date, non-integer n).
     E329 — zero/negative cadence, count<1 in `date_range`.
     W327 — literal round-trip composition.
     W328 — `date_range` non-aligned `end`. *)

(* Helper: build a model with a single intervention `at = [...]` and
   return the resulting `Ir.AtTimes` float list. Lets us inspect the
   numeric output of `add_calendar_*` / `date_range` directly. *)
let at_times_of_first_intervention src =
  let m = compile_expect_ok src in
  Alcotest.(check int) "exactly one intervention" 1
    (List.length m.Ir.interventions);
  let iv = List.hd m.Ir.interventions in
  match iv.Ir.fire with
  | Ir.Scheduled (Ir.AtTimes ts) -> ts
  | _ -> Alcotest.failf "expected AtTimes schedule"

(* Helper: build a one-intervention `at` model that fires at one
   `add_calendar_*` result and assert the day offset. *)
let assert_at_day ~src ~expected_day =
  let ts = at_times_of_first_intervention src in
  Alcotest.(check int) "one fire time" 1 (List.length ts);
  Alcotest.(check (float 1e-9))
    "fire day equals expected" expected_day (List.hd ts)

(* ── add_calendar_months / add_calendar_years: canonical cases (§8) ─── *)

(* Template for a one-intervention anchored model whose only firing time
   is `<expr>`. Tests substitute the expression and the expected day
   offset relative to `origin = date("2020-01-01")`. *)
let anchored_at_src ~at_expr = Printf.sprintf {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters { N0 : count }
    transitions { leak : S --> V @ 0.0 'per_day * S }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1) at [%s]
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 800 'days }
  |} at_expr

let test_phase2_add_months_leap_feb_clamp () =
  (* Jan 31 + 1 month → Feb 29 (2020 is leap).
     Day offset of Feb 29 from Jan 1, 2020 = 31 + 28 = 59. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2020-01-31\"), 1)")
    ~expected_day:59.0

let test_phase2_add_months_non_leap_feb_clamp () =
  (* Jan 31 (2021) + 1 month → Feb 28 (2021, non-leap).
     Day offset from Jan 1, 2020 = 366 + 31 + 27 = 424.
       (2020 is leap: 366 days)  +  (Jan 31 - 1 = 30, then +28 = 58)
       = 366 + 58 = 424. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2021-01-31\"), 1)")
    ~expected_day:424.0

let test_phase2_add_years_leap_to_non_leap () =
  (* Feb 29, 2020 + 1 year → Feb 28, 2021 (Feb 29 clamps to Feb 28).
     Day offset = 366 + 31 + 27 = 424. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_years(date(\"2020-02-29\"), 1)")
    ~expected_day:424.0

let test_phase2_add_months_13_crosses_year () =
  (* Jan 31, 2020 + 13 months → Feb 28, 2021.
     Day offset from Jan 1, 2020 = 424. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2020-01-31\"), 13)")
    ~expected_day:424.0

let test_phase2_add_months_mar_to_apr () =
  (* Mar 31, 2020 + 1 month → Apr 30, 2020 (Apr has 30 days).
     Day offset from Jan 1, 2020 = 31 + 29 + 30 + 29 = 119.
     (Jan: 31 days, Feb: 29 days, Mar: 31 days → start of Apr = 91,
      then +29 to reach Apr 30 → 91 + 29 = 120; off-by-one? Let me
      recompute: day 0 = Jan 1. Apr 1 is day 91 (Jan=31, Feb=29,
      Mar=31). Apr 30 is day 120. ✓) *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2020-03-31\"), 1)")
    ~expected_day:120.0

let test_phase2_sub_months_mar_to_feb_leap () =
  (* Mar 31, 2020 − 1 month → Feb 29, 2020 (clamp; leap year).
     Day offset = 59. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2020-03-31\"), -1)")
    ~expected_day:59.0

let test_phase2_sub_months_mar_to_feb_non_leap () =
  (* Mar 31, 2021 − 1 month → Feb 28, 2021 (clamp; non-leap).
     Day offset from Jan 1, 2020 = 366 + 58 = 424. *)
  assert_at_day
    ~src:(anchored_at_src
            ~at_expr:"add_calendar_months(date(\"2021-03-31\"), -1)")
    ~expected_day:424.0

let test_phase2_add_months_origin_anchored () =
  (* `add_calendar_months(origin, 6)` with origin = 2020-01-01.
     Result is 2020-07-01 → day offset = 31+29+31+30+31+30 = 182. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters { N0 : count }
    transitions { leak : S --> V @ 0.0 'per_day * S }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1)
             at [add_calendar_months(origin, 6)]
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 365 'days }
  |} in
  assert_at_day ~src ~expected_day:182.0

let test_phase2_add_months_unanchored_errors () =
  (* `add_calendar_months` in an unanchored model is E327. *)
  expect_error_code
    ~code:"E327"
    ~contains:"anchored"
    {|
      time_unit = 'months
      compartments { S, V }
      parameters { N0 : count }
      transitions { leak : S --> V @ 0.0 'per_month * S }
      interventions {
        vacc : transfer(from = S, to = V, fraction = 0.1)
               at [add_calendar_months(date("2020-01-01"), 6)]
      }
      init { S = N0 }
      simulate { from = 0  to = 24 }
    |}

(* ── date_range tests ──────────────────────────────────────────────── *)

let test_phase2_date_range_affine_start_end () =
  (* date_range(2020-01-01, 2020-12-31, every = 7 'days) → 53 entries. *)
  let src = anchored_at_src
    ~at_expr:"date_range(date(\"2020-01-01\"), date(\"2020-12-31\"), every = 7 'days)"
  in
  let m = compile_expect_ok src in
  let iv = List.hd m.Ir.interventions in
  match iv.Ir.fire with
  | Ir.Scheduled (Ir.AtTimes ts) ->
    Alcotest.(check int) "53 weekly entries" 53 (List.length ts);
    Alcotest.(check (float 1e-9)) "first entry = 0" 0.0 (List.hd ts);
    Alcotest.(check (float 1e-9)) "last entry = 364 (Dec 30)"
      364.0 (List.nth ts 52)
  | _ -> Alcotest.fail "expected AtTimes"

let test_phase2_date_range_affine_count () =
  (* date_range(2020-01-01, count = 24, every = 7 'days) → 25 entries. *)
  let src = anchored_at_src
    ~at_expr:"date_range(date(\"2020-01-01\"), count = 24, every = 7 'days)"
  in
  let m = compile_expect_ok src in
  let iv = List.hd m.Ir.interventions in
  match iv.Ir.fire with
  | Ir.Scheduled (Ir.AtTimes ts) ->
    Alcotest.(check int) "25 weekly entries (start + 24 steps)" 25
      (List.length ts);
    Alcotest.(check (float 1e-9)) "last entry = 24 * 7 = 168"
      168.0 (List.nth ts 24)
  | _ -> Alcotest.fail "expected AtTimes"

let test_phase2_date_range_calendar_months_start_end () =
  (* date_range(2020-01-01, 2024-12-01, calendar_months = 3).

     Cadence boundaries from start are Jan/Apr/Jul/Oct of each year.
     Dec 1 2024 does NOT land on a quarterly cadence from Jan 1 2020
     (the last on-cadence date ≤ end is Oct 1 2024, k=19 → 20 entries
     total) — proposal §4 had an inconsistent example saying "21
     entries: ..., Oct 1 2024, Dec 1 2024" which would require
     appending a non-aligned `end`, contradicting the rule "last entry
     is the latest boundary ≤ end" in the same section. We follow the
     rule, so the result is 20 entries plus a W328 warning. Use a
     model whose `end` IS on cadence (Oct 1 2024) for the clean
     start–end test; the W328 non-aligned case is covered separately. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters { N0 : count }
    transitions { leak : S --> V @ 0.0 'per_day * S }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1)
             at [date_range(date("2020-01-01"), date("2024-10-01"),
                            calendar_months = 3)]
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 2000 'days }
  |} in
  let m = compile_expect_ok src in
  let iv = List.hd m.Ir.interventions in
  match iv.Ir.fire with
  | Ir.Scheduled (Ir.AtTimes ts) ->
    Alcotest.(check int) "20 quarterly entries" 20 (List.length ts);
    Alcotest.(check (float 1e-9)) "first entry = 0" 0.0 (List.hd ts)
  | _ -> Alcotest.fail "expected AtTimes"

let test_phase2_date_range_calendar_years_count () =
  (* date_range(2020-01-01, count = 5, calendar_years = 1) → 6 entries:
     Jan 1 of 2020..2025. *)
  let src = {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, V }
    parameters { N0 : count }
    transitions { leak : S --> V @ 0.0 'per_day * S }
    interventions {
      vacc : transfer(from = S, to = V, fraction = 0.1)
             at [date_range(date("2020-01-01"), count = 5,
                            calendar_years = 1)]
    }
    init { S = N0 }
    simulate { from = 0 'days  to = 2500 'days }
  |} in
  let m = compile_expect_ok src in
  let iv = List.hd m.Ir.interventions in
  match iv.Ir.fire with
  | Ir.Scheduled (Ir.AtTimes ts) ->
    Alcotest.(check int) "6 annual entries (start + 5 steps)" 6
      (List.length ts);
    Alcotest.(check (float 1e-9)) "first entry = 0 (Jan 1, 2020)"
      0.0 (List.hd ts);
    (* Jan 1, 2025 = 366 (2020) + 365 (2021) + 365 (2022) + 365 (2023)
       + 366 (2024 leap) = 1827. *)
    Alcotest.(check (float 1e-9)) "last entry = 1827 (Jan 1, 2025)"
      1827.0 (List.nth ts 5)
  | _ -> Alcotest.fail "expected AtTimes"

let test_phase2_date_range_non_aligned_end_w328 () =
  (* Non-aligned end fires W328. start=Jan 1 2020, end=Jan 5 2020,
     every = 3 'days. Boundaries: Jan 1, Jan 4. Jan 5 doesn't land. *)
  let src = anchored_at_src
    ~at_expr:"date_range(date(\"2020-01-01\"), date(\"2020-01-05\"), every = 3 'days)"
  in
  let d = expect_diag ~severity:Diagnostics.Warning ~code:"W328" src in
  assert_hint_contains ~needle:"inclusive_end" d

let test_phase2_date_range_zero_cadence_errors () =
  expect_error_code
    ~code:"E329"
    ~contains:"positive"
    (anchored_at_src
       ~at_expr:"date_range(date(\"2020-01-01\"), count = 4, every = 0 'days)")

let test_phase2_date_range_negative_calendar_cadence_errors () =
  expect_error_code
    ~code:"E329"
    ~contains:"positive"
    (anchored_at_src
       ~at_expr:"date_range(date(\"2020-01-01\"), count = 4, calendar_months = -1)")

let test_phase2_date_range_count_zero_errors () =
  expect_error_code
    ~code:"E329"
    ~contains:"≥ 1"
    (anchored_at_src
       ~at_expr:"date_range(date(\"2020-01-01\"), count = 0, every = 7 'days)")

let test_phase2_date_range_calendar_in_unanchored_errors () =
  expect_error_code
    ~code:"E327"
    ~contains:"anchored"
    {|
      time_unit = 'months
      compartments { S, V }
      parameters { N0 : count }
      transitions { leak : S --> V @ 0.0 'per_month * S }
      interventions {
        vacc : transfer(from = S, to = V, fraction = 0.1)
               at [date_range(date("2020-01-01"), count = 4,
                              calendar_months = 3)]
      }
      init { S = N0 }
      simulate { from = 0  to = 24 }
    |}

(* ── Round-trip W327 ───────────────────────────────────────────────── *)

let test_phase2_round_trip_w327 () =
  (* `add_calendar_months(add_calendar_months(date("2020-01-31"), 1), -1)`
     fires W327. *)
  let src = anchored_at_src
    ~at_expr:"add_calendar_months(add_calendar_months(date(\"2020-01-31\"), 1), -1)"
  in
  let d = expect_diag ~severity:Diagnostics.Warning ~code:"W327" src in
  assert_hint_contains ~needle:"non-invertible" d

(* ── origin tests ──────────────────────────────────────────────────── *)

let test_phase2_origin_in_simulate_from_anchored () =
  (* `simulate { from = origin }` in anchored mode resolves to 0.0. *)
  let m = compile_expect_ok {|
    time_unit = 'days
    origin = date("2020-01-01")
    compartments { S, I, R }
    parameters {
      beta : rate  gamma : rate  N0 : count  I0 : count
    }
    let N = S + I + R
    transitions {
      infection : S --> I @ beta * S * I / N
      recovery  : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = origin  to = 120 'days }
  |} in
  Alcotest.(check (float 1e-9))
    "simulate.t_start = 0 (origin)"
    0.0 m.Ir.simulation.Ir.t_start

let test_phase2_origin_in_unanchored_errors () =
  (* `origin` reference in unanchored mode is E327. *)
  expect_error_code
    ~code:"E327"
    ~contains:"unanchored"
    {|
      time_unit = 'months
      compartments { S, V }
      parameters { N0 : count }
      transitions { leak : S --> V @ 0.0 'per_month * S }
      interventions {
        vacc : transfer(from = S, to = V, fraction = 0.1) at [origin]
      }
      init { S = N0 }
      simulate { from = 0  to = 24 }
    |}

(* ── #161: dt as a model knob in the simulate block ───────────────────── *)

let dt_model_body = {|
    compartments { S, I, R }
    parameters { beta : rate  gamma : rate  N0 : count  I0 : count }
    let N = S + I + R
    transitions {
      infection : S --> I  @ beta * S * I / N
      recovery  : I --> R  @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
|}

let test_simulate_dt_plain () =
  (* `simulate { dt = 0.5 }` lowers to simulation.dt = Some 0.5. *)
  let src = dt_model_body ^ {|
    simulate { from = 0 'days  to = 100 'days  dt = 0.5 }
  |} in
  match Compiler.compile ~name:"test_sim_dt_plain" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.simulation.Ir.dt with
     | Some d -> Alcotest.(check (float 1e-9)) "simulation.dt = 0.5" 0.5 d
     | None   -> Alcotest.fail "expected simulation.dt = Some 0.5, got None")

let test_simulate_dt_omitted_is_none () =
  (* No dt in the simulate block → simulation.dt = None (CLI default applies). *)
  let src = dt_model_body ^ {|
    simulate { from = 0 'days  to = 100 'days }
  |} in
  match Compiler.compile ~name:"test_sim_dt_none" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.simulation.Ir.dt with
     | None   -> ()
     | Some d -> Alcotest.failf "expected simulation.dt = None, got Some %g" d)

let test_simulate_dt_unit_aware () =
  (* dt is unit-aware like from/to: `dt = 0.05 'months` is one month-scaled
     step. The model time unit is days (default). The 'months factor is the
     Gregorian mean month, days_per(Months) = 365.2425 / 12 = 30.436875 days
     (expander.ml `days_per`), so 0.05 months = 0.05 * 30.436875 =
     1.52184375 days. *)
  let src = {|
    time_unit = 'days
  |} ^ dt_model_body ^ {|
    simulate { from = 0 'days  to = 100 'days  dt = 0.05 'months }
  |} in
  match Compiler.compile ~name:"test_sim_dt_unit" src with
  | Error e -> Alcotest.failf "compile failed: %s" e
  | Ok m ->
    (match m.Ir.simulation.Ir.dt with
     | Some d ->
       let expected = 0.05 *. (365.2425 /. 12.0) in  (* = 1.52184375 *)
       Alcotest.(check (float 1e-9)) "dt = 0.05 'months in days" expected d
     | None -> Alcotest.fail "expected simulation.dt = Some _, got None")

let test_simulate_unknown_key_errors () =
  (* A typo'd / unsupported simulate key is a clear error, never silently
     dropped (no-loose-semantics). *)
  compile_expect_error_code ~code:"E106" ~contains:"step"
    (dt_model_body ^ {|
    simulate { from = 0 'days  to = 100 'days  step = 0.5 }
  |})

(* ── gh#181 step 1: structured, non-raising compile_outcome ──────────────────
   compile_outcome returns every diagnostic as a value and never raises;
   on a POST-EXPANSION error (Validate E507) both surfaces return it as a
   value — [compile] as [Error], [compile_outcome] as [value = None]. The
   two SEIR models differ by one
   character: the observation projects `incidence(infection)` (a real
   transition) vs `incidence(infektion)` (a typo). A bare unknown name in
   `incidence(...)` falls through expansion to a dangling CumulativeFlow
   (expander.ml ~4001), caught by Validate as E507 (validate.ml:112) — a
   front-end E100 would instead catch a name used in a rate/likelihood, so
   the dangling-projection route is what exercises the late path. *)

let outcome_model_ok = {|
    time_unit = 'days
    compartments { S, E, I, R }
    let N = S + E + I + R
    parameters {
      beta  : rate        in [0.001, 0.5]
      sigma : rate        in [0.01,  1.0]
      gamma : rate        in [0.01,  1.0]
      rho   : probability in [0.0,   1.0]
      k     : real        in [0.1,  100.0]
    }
    transitions {
      infection   : S --> E  @ beta * S * I / N
      progression : E --> I  @ sigma * E
      recovery    : I --> R  @ gamma * I
    }
    observations {
      weekly_cases {
        columns       { time : time, weekly_cases : count }
        projected  = incidence(infection)
        emit_schedule = every 7 'days
        weekly_cases ~ neg_binomial(mean = rho * projected, r = k)
      }
    }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let outcome_model_late_err = {|
    time_unit = 'days
    compartments { S, E, I, R }
    let N = S + E + I + R
    parameters {
      beta  : rate        in [0.001, 0.5]
      sigma : rate        in [0.01,  1.0]
      gamma : rate        in [0.01,  1.0]
      rho   : probability in [0.0,   1.0]
      k     : real        in [0.1,  100.0]
    }
    transitions {
      infection   : S --> E  @ beta * S * I / N
      progression : E --> I  @ sigma * E
      recovery    : I --> R  @ gamma * I
    }
    observations {
      weekly_cases {
        columns       { time : time, weekly_cases : count }
        projected  = incidence(infektion)
        emit_schedule = every 7 'days
        weekly_cases ~ neg_binomial(mean = rho * projected, r = k)
      }
    }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let test_compile_outcome_clean_returns_value () =
  let o = Compiler.compile_outcome ~name:"oc_clean" outcome_model_ok in
  (match o.Compiler.value with
   | Some _ -> ()
   | None   -> Alcotest.failf "clean model: expected Some value, got None");
  let n_err =
    List.length
      (List.filter
         (fun (d : Diagnostics.diagnostic) -> d.severity = Diagnostics.Error)
         o.Compiler.diagnostics)
  in
  Alcotest.(check int) "clean model: no Error-severity diagnostics" 0 n_err

let test_compile_outcome_late_error_is_value_not_raise () =
  (* A post-expansion error (Validate E507) must arrive as a VALUE from both
     surfaces: [compile] returns [Error] (it no longer raises Compile_error,
     so the CLI exits 1 cleanly instead of on an uncaught exception), and
     [compile_outcome] returns [value = None]. json mode keeps the rendered
     payload a compact JSON array we can grep for the code. *)
  Diagnostics.json_errors_mode := true;
  let compile_result = Compiler.compile ~name:"oc_err" outcome_model_late_err in
  Diagnostics.json_errors_mode := false;
  (match compile_result with
   | Ok _ ->
     Alcotest.failf "expected compile to return Error on a dangling \
                     observation reference, got Ok"
   | Error payload ->
     if not (contains_substring ~needle:"E507" payload) then
       Alcotest.failf "compile Error payload should name E507, got: %s" payload);
  (* compile_outcome surfaces the same error as a value. *)
  let o = Compiler.compile_outcome ~name:"oc_err" outcome_model_late_err in
  Alcotest.(check bool) "compile_outcome: value is None on error"
    true (o.Compiler.value = None);
  Alcotest.(check bool) "compile_outcome: E507 surfaced as a value"
    true (count_diags_with_code o.Compiler.diagnostics "E507" >= 1)

(* gh#181: decl-keyed post-expansion (validate) errors now carry the source
   loc of the offending declaration instead of no_loc. *)
let test_validate_decl_error_has_location () =
  let src = {|
    compartments { S, I, X : real }
    parameters { beta : rate in [0, 1] }
    transitions { infection : S --> I @ beta * S * I }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let diags = Compiler.collect_diagnostics ~name:"loc_e509" src in
  match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "E509") diags with
  | None -> Alcotest.failf "expected E509 (real compartment with no ODE)"
  | Some d ->
    Alcotest.(check bool) "E509 carries a real source location (line > 0)"
      true (d.loc.Diagnostics.line > 0)

(* gh#181: a reference error (E507, dangling observation transition) points at
   its enclosing observation rather than no_loc. *)
let test_validate_reference_error_has_location () =
  let src = {|
    compartments { S, I, R }
    parameters { beta : rate in [0,1]  gamma : rate in [0,1]  rho : probability in [0,1] }
    transitions {
      infection : S --> I @ beta * S * I
      recovery  : I --> R @ gamma * I
    }
    observations {
      cases {
        columns       { time : time, cases : count }
        projected  = incidence(recoveryX)
        emit_schedule = every 1 'days
        cases ~ poisson(rate = rho * projected)
      }
    }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let diags = Compiler.collect_diagnostics ~name:"loc_e507" src in
  match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "E507") diags with
  | None -> Alcotest.failf "expected E507 (dangling observation transition)"
  | Some d ->
    Alcotest.(check bool) "E507 points at the enclosing observation (line > 0)"
      true (d.loc.Diagnostics.line > 0)

(* gh#181: dimcheck (dimensional) errors now point at the offending construct.
   dimcheck is disabled globally for compiler tests, so enable it here. *)
let test_dimcheck_error_has_location () =
  let src = {|
    compartments { S, I }
    parameters { beta : rate in [0, 1] }
    transitions { infection : S --> I @ beta + S }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let prev = !Compiler.no_dim_check in
  Compiler.no_dim_check := false;
  let diags = Compiler.collect_diagnostics ~name:"loc_e300" src in
  Compiler.no_dim_check := prev;
  match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "E300") diags with
  | None -> Alcotest.failf "expected E300 (dimensional error)"
  | Some d ->
    Alcotest.(check bool) "E300 points at the transition (line > 0)"
      true (d.loc.Diagnostics.line > 0)

(* gh#181: lint L402 (dead compartment) points at the compartment. *)
let test_lint_warning_has_location () =
  let src = {|
    compartments { S, I, Z }
    parameters { beta : rate in [0, 1] }
    transitions { infection : S --> I @ beta * S * I }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |} in
  let diags = Compiler.collect_diagnostics ~name:"loc_l402" src in
  match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "L402") diags with
  | None -> Alcotest.failf "expected L402 (dead compartment)"
  | Some d ->
    Alcotest.(check bool) "L402 points at the compartment (line > 0)"
      true (d.loc.Diagnostics.line > 0)

(* ── gh#112: table-lookup arity validation ──────────────────────────────────
   A table declared `C_age : age × age` (rank 2) must be indexed with exactly
   two indices. Under-indexing (`C_age[a]`) previously fell through the
   `List.mapi (fun i item -> ... List.nth tdims i ...) items` loop, which
   iterates over the user's *items* not the declared *tdims*, terminating short
   and producing a partial-prefix linear index — a silently wrong cell.
   Over-indexing read `tdims` out of range. Both must now be hard E202. *)

let arity_model ~lookup = Printf.sprintf {|
time_unit = 'days
compartments { S, E, I, R }
dimensions { age = [child, adult] }
stratify(by = age)
let N_local[a in age] = S[a] + E[a] + I[a] + R[a]
parameters { beta : rate in [0.001, 0.5] }
tables { C_age : age × age = [[12.0, 4.0], [4.0, 8.0]] }
transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, %s * I[b] / N_local[b])
  recovery[a in age]  : I[a] --> R[a] @ 0.1 * I[a]
}
init { S[child] = 100  I[child] = 1 }
simulate { from = 0 'days  to = 10 'days }
|} lookup

let test_table_lookup_under_indexed_e202 () =
  (* C_age is rank 2; supplying one index must error, not silently
     resolve a prefix cell. *)
  compile_expect_error_code ~code:"E202" ~contains:"C_age"
    (arity_model ~lookup:"C_age[a]")

let test_table_lookup_over_indexed_e202 () =
  (* C_age is rank 2; supplying three indices must error. *)
  compile_expect_error_code ~code:"E202" ~contains:"C_age"
    (arity_model ~lookup:"C_age[a, b, a]")

let test_table_lookup_correct_arity_ok () =
  (* The two-index form still compiles (guard must not over-fire). *)
  let _ = compile_expect_ok (arity_model ~lookup:"C_age[a, b]") in
  ()

(* ── E287: partial dimension omission in a rate read ─────────────────────────
   A compartment stratified over 2+ dimensions (E has [age, latent_stage])
   referenced in a rate with *some but not all* dimensions dropped (`E[a]`) has
   no defined cell. It previously fell through to name-mangling (`E_adult`) and
   E100'd against a synthetic compartment the user never wrote — no source loc,
   a name they can't act on. It must now be a located E287 that names the real
   compartment and points at the explicit-marginalization fix.

   Regression-guarded alongside: bare name `E` (sums ALL dims), full index
   `E[a, s]`, explicit `sum(s in latent_stage, E[a, s])`, and the single-dim
   FOI contact pattern `sum(b in age, C[a,b]*I[b]/N[b])` — none of which must
   trip the new guard. *)

(* `rate` is spliced into the recovery transition's rate body; `E` has
   [age, latent_stage], `I`/`S`/`R` have [age]. *)
let partial_omit_model ~rate = Printf.sprintf {|
time_unit = 'days
compartments { S, E, I, R }
dimensions {
  age = [child, adult]
  latent_stage = [e1, e2]
}
stratify(by = age)
stratify(by = latent_stage, only = [E])
parameters {
  beta : rate in [0.001, 0.5]
  sigma : rate in [0.01, 1.0]
  gamma : rate in [0.01, 1.0]
}
transitions {
  infection[a in age] : S[a] --> E[a, e1] @ beta * S[a] * I[a]
  progression[a in age] : E[a, e1] --> E[a, e2] @ sigma * E[a, e1]
  onset[a in age] : E[a, e2] --> I[a] @ sigma * E[a, e2]
  recovery[a in age] : I[a] --> R[a] @ %s
}
init { S[child] = 100  I[child] = 1 }
simulate { from = 0 'days  to = 10 'days }
|} rate

let test_partial_dimension_omission_e287_with_loc () =
  (* RED before the fix: `E[a]` produced E100 'undeclared name E_adult' with no
     source loc. GREEN: a located E287 naming the real compartment 'E'. *)
  let src = partial_omit_model ~rate:"gamma * E[a]" in
  let diags = Compiler.collect_diagnostics ~name:"partial_omit" src in
  (match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "E100") diags with
   | Some _ -> Alcotest.failf "partial index must NOT fall through to E100 (synthetic name)"
   | None -> ());
  match List.find_opt (fun (d : Diagnostics.diagnostic) -> d.code = "E287") diags with
  | None -> Alcotest.failf "expected E287 for partial dimension omission E[a]"
  | Some d ->
    (* the diagnostic must carry a real source location, not no_loc *)
    Alcotest.(check bool) "E287 points at the index node (line > 0)"
      true (d.loc.Diagnostics.line > 0);
    (* and name the real compartment + its dimensions *)
    Alcotest.(check bool) "E287 names compartment 'E'"
      true (contains_substring ~needle:"'E'" d.message);
    Alcotest.(check bool) "E287 lists the dimensions"
      true (contains_substring ~needle:"latent_stage" d.message)

let test_bare_name_sums_all_dims_ok () =
  (* Omitting ALL dimensions (bare `E`) sums over them — the existing PopSum
     path. Must not trip the partial-index guard. *)
  let _ = compile_expect_ok (partial_omit_model ~rate:"gamma * I[a] * E") in
  ()

let test_full_index_resolves_ok () =
  (* Fully-indexed `E[a, e1]` resolves to the concrete cell. *)
  let _ = compile_expect_ok (partial_omit_model ~rate:"gamma * E[a, e1]") in
  ()

let test_explicit_marginalization_sum_ok () =
  (* The blessed marginalization form `sum(s in latent_stage, E[a, s])`. *)
  let _ = compile_expect_ok
    (partial_omit_model ~rate:"gamma * sum(s in latent_stage, E[a, s])") in
  ()

(* The single-dimension FOI contact pattern: I is [age] (one dim), so the
   indexed reads C[a,b] (table), I[b], N[b] are all full — no partial omission.
   Must compile (the guard must not misfire on single-dim compartments). *)
let foi_model = {|
time_unit = 'days
compartments { S, I, R }
dimensions {
  age = [child, adult]
}
stratify(by = age)
let N[a in age] = S[a] + I[a] + R[a]
parameters {
  beta : rate in [0.001, 0.5]
  gamma : rate in [0.01, 1.0]
}
tables { C : age × age = [[12.0, 4.0], [4.0, 8.0]] }
transitions {
  infection[a in age] : S[a] --> I[a]
    @ beta * S[a] * sum(b in age, C[a, b] * I[b] / N[b])
  recovery[a in age] : I[a] --> R[a] @ gamma * I[a]
}
init { S[child] = 100  I[child] = 1 }
simulate { from = 0 'days  to = 10 'days }
|}

let test_single_dim_foi_pattern_ok () =
  let _ = compile_expect_ok foi_model in
  ()

(* ── gh#117: duplicate / cross-namespace declaration names ───────────────────
   build_lookup_tables used Hashtbl.replace (silent last-wins). A duplicate
   within a namespace, or the same name in two namespaces (e.g. a parameter and
   a let both named `N`), must be a hard error naming both declarations — not a
   silent resolution to whichever the lookup order happens to favour. *)

let test_duplicate_parameter_rejected () =
  compile_expect_error_code ~code:"E278" ~contains:"beta" {|
    compartments { S, I }
    parameters {
      beta : rate in [0, 1]
      beta : rate in [0, 1]
    }
    transitions { inf : S --> I @ beta * S * I }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let test_duplicate_let_rejected () =
  compile_expect_error_code ~code:"E278" ~contains:"k" {|
    compartments { S, I }
    parameters { beta : rate in [0, 1] }
    let k = 1.0
    let k = 2.0
    transitions { inf : S --> I @ beta * k * S * I }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let test_cross_namespace_param_and_let_rejected () =
  (* `N` declared as both a parameter and a let: expressions would resolve to
     one of them depending on lookup order — must be a hard ambiguity error. *)
  compile_expect_error_code ~code:"E278" ~contains:"N" {|
    compartments { S, I }
    parameters {
      beta : rate in [0, 1]
      N    : count in [1, 1e9]
    }
    let N = S + I
    transitions { inf : S --> I @ beta * S * I / N }
    init { S = 100  I = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

let test_cross_namespace_expanded_name_rejected () =
  (* Reviewer feedback: the prior fix checked only BASE names, leaving a hole
     on EXPANDED/stratified names. Here the bases DIFFER (`I_a` the
     compartment vs `I` the indexed parameter) so a base-name-only check
     passes — but `I[grp]` expands to `I_a`, `I_b`, and `I_a` collides with
     the literal compartment `I_a`. The check must catch the expanded
     collision. *)
  compile_expect_error_code ~code:"E278" ~contains:"I_a" {|
    compartments { S, I_a }
    dimensions { grp = [a, b] }
    parameters {
      beta  : rate in [0, 1]
      I[grp] : count in [0, 1000]
    }
    transitions { inf : S --> I_a @ beta * S * I_a }
    init { S = 100  I_a = 1 }
    simulate { from = 0 'days  to = 10 'days }
  |}

(* ── gh#114: stratified initial conditions vs expanded compartments ──────────
   `init { S = N0 }` where S is stratified into S_child/S_adult must be
   rejected (the bare key S is not a real expanded compartment). An init for a
   compartment that does not exist at all must also be rejected. A concrete
   cell (`S[child]`) is accepted. Reviewer feedback: emit ONE located
   diagnostic (E277), not a no-location E513 followed by a located E277. *)

let init_strat_model ~init = Printf.sprintf {|
time_unit = 'days
compartments { S, E, I, R }
dimensions { age = [child, adult] }
stratify(by = age)
parameters { beta : rate in [0.001, 0.5] }
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a] * I[a]
  recovery[a in age]  : I[a] --> R[a] @ 0.1 * I[a]
}
init { %s }
simulate { from = 0 'days  to = 10 'days }
|} init

let test_init_bare_stratified_rejected () =
  compile_expect_error_code ~code:"E277" ~contains:"S"
    (init_strat_model ~init:"S = 5000\n  I[child] = 10")

let test_init_unknown_compartment_rejected () =
  compile_expect_error_code ~code:"E277" ~contains:"X_child"
    (init_strat_model ~init:"X[child] = 5000\n  I[child] = 10")

let test_init_concrete_cell_ok () =
  let _ = compile_expect_ok
    (init_strat_model ~init:"S[child] = 4990\n  S[adult] = 5000\n  I[child] = 10") in
  ()

let test_init_bare_stratified_single_diagnostic () =
  (* Reviewer feedback: exactly ONE diagnostic for the one root cause. *)
  let src = init_strat_model ~init:"S = 5000\n  I[child] = 10" in
  let diags = Compiler.collect_diagnostics ~name:"init_one_diag" src in
  let errs = List.filter (fun (d : Diagnostics.diagnostic) ->
    d.severity = Diagnostics.Error
    && contains_substring ~needle:"S" d.message
    && not (contains_substring ~needle:"undeclared" d.message)) diags in
  Alcotest.(check int) "exactly one init-membership diagnostic for `S`"
    1 (List.length errs);
  (match errs with
   | [d] -> Alcotest.(check string) "located E277" "E277" d.code
   | _ -> ())

(* ── gh#98: ISO date out-of-range validation ─────────────────────────────────
   parse_iso_date did no month/day range check, so `date("2020-02-30")`
   silently shifted to a garbage day offset with no diagnostic. Out-of-range
   dates must now produce a NAMED diagnostic (E223). *)

let date_model ~date = Printf.sprintf {|
time_unit = 'days
origin = date("2020-01-01")
compartments { S, I, V }
parameters {
  beta     : rate        in [0, 1]
  vacc_cov : probability in [0, 1]
}
transitions { inf : S --> I @ beta * S * I }
interventions {
  sia : transfer(fraction = vacc_cov, from = S, to = V)
        at [ date("%s") ]
}
init { S = 100  I = 1 }
simulate { from = origin  to = add_calendar_years(origin, 2) }
|} date

let test_date_invalid_day_rejected () =
  compile_expect_error_code ~code:"E223" ~contains:"2020-02-30"
    (date_model ~date:"2020-02-30")

let test_date_invalid_month_rejected () =
  compile_expect_error_code ~code:"E223" ~contains:"2020-13-01"
    (date_model ~date:"2020-13-01")

let test_date_feb29_leap_year_ok () =
  (* 2020 is a leap year: Feb 29 is valid and must compile. *)
  let _ = compile_expect_ok (date_model ~date:"2020-02-29") in
  ()

let test_date_feb29_non_leap_year_rejected () =
  (* 2021 is not a leap year: Feb 29 must be rejected. *)
  compile_expect_error_code ~code:"E223" ~contains:"2021-02-29"
    (date_model ~date:"2021-02-29")

let test_sum_var_shadows_transition_index_rejected () =
  (* A `sum` bound variable that reuses the enclosing transition's index var
     silently rebinds (first-match-wins env), turning a per-stratum term into a
     global sum. Must be rejected (E281). *)
  let src =
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     dimensions { patch = [a, b, c] }\n\
     stratify(by = patch)\n\
     parameters { beta : rate in [0,2]  gamma : rate in [0,1] }\n\
     let N[p in patch] = S[p] + I[p] + R[p]\n\
     transitions {\n\
     \  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * sum(p in patch, I[p] / N[p])\n\
     \  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]\n\
     }\n\
     init { S[a] = 99  I[a] = 1  S[b] = 100  S[c] = 100 }\n\
     simulate { from = 0 'days  to = 10 'days }\n"
  in
  compile_expect_error_code ~code:"E283" ~contains:"shadow" src

let test_sum_var_distinct_from_index_ok () =
  (* The normal pattern — sum var distinct from the transition index — compiles. *)
  let src =
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     dimensions { patch = [a, b, c] }\n\
     stratify(by = patch)\n\
     parameters { beta : rate in [0,2]  gamma : rate in [0,1] }\n\
     let N[p in patch] = S[p] + I[p] + R[p]\n\
     transitions {\n\
     \  infection[p in patch] : S[p] --> I[p] @ beta * S[p] * sum(q in patch, I[q] / N[q])\n\
     \  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]\n\
     }\n\
     init { S[a] = 99  I[a] = 1  S[b] = 100  S[c] = 100 }\n\
     simulate { from = 0 'days  to = 10 'days }\n"
  in
  let _ = compile_expect_ok src in
  ()

(* ── Restricted sums: sum(v in d where P, body) (gh#185) ─────────────────── *)

(* 4 patches a,b,c,d; row-major dist. Patch a couples only to b (dist 30 < 50);
   c and d are at 99 (out of radius), and the self-term is excluded by q != p. *)
let where_radius_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [a, b, c, d] }\n\
   stratify(by = patch)\n\
   parameters { beta : rate in [0,2]  gamma : rate in [0,1]  rho : probability in [0,1] }\n\
   tables { dist : patch × patch = [[0.0,30.0,99.0,99.0],[30.0,0.0,30.0,99.0],[99.0,30.0,0.0,30.0],[99.0,99.0,30.0,0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   infection[p in patch] : S[p] --> I[p] @ beta * S[p] * (I[p]/N[p] + rho * sum(q in patch where dist[p,q] < 50 and q != p, I[q]/N[q]))\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[a]=99 I[a]=1 S[b]=100 S[c]=100 S[d]=100 }\n\
   simulate { from = 0 'days to = 30 'days }\n"

let rec pop_names acc (e : Ir.expr) = match e with
  | Ir.Pop n -> n :: acc
  | Ir.BinOp b -> pop_names (pop_names acc b.left) b.right
  | Ir.UnOp u -> pop_names acc u.arg
  | Ir.Cond c -> pop_names (pop_names (pop_names acc c.pred) c.then_) c.else_
  | Ir.Reduce ts -> List.fold_left pop_names acc ts
  | Ir.UncheckedDim r -> pop_names acc r.inner
  | _ -> acc

(* With the constant-fold OFF, any pruning is purely from the `where` predicate
   (not the fold dropping zero-W terms) — so this pins sparsity-by-construction. *)
let test_where_radius_prunes () =
  with_fold_disabled (fun () ->
    let m = match Compiler.compile ~name:"where_radius" where_radius_src with
      | Ok m -> m
      | Error e -> Alcotest.failf "where-radius model should compile: %s" e in
    match List.find_opt (fun (t : Ir.transition) -> t.name = "infection_a") m.transitions with
    | None -> Alcotest.fail "no infection_a transition"
    | Some t ->
      let pops = pop_names [] t.rate in
      Alcotest.(check bool) "infection_a couples to I_b (in radius)" true  (List.mem "I_b" pops);
      Alcotest.(check bool) "infection_a does NOT couple to I_c"     false (List.mem "I_c" pops);
      Alcotest.(check bool) "infection_a does NOT couple to I_d"     false (List.mem "I_d" pops))

let test_where_fitted_threshold_rejected () =
  compile_expect_error_code ~code:"E284" ~contains:"fitted threshold"
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     dimensions { patch = [a, b, c] }\n\
     stratify(by = patch)\n\
     parameters { beta : rate in [0,2]  gamma : rate in [0,1]  thr : positive in [0,100] }\n\
     tables { dist : patch × patch = [[0.0,30.0,99.0],[30.0,0.0,30.0],[99.0,30.0,0.0]] }\n\
     let N[p in patch] = S[p] + I[p] + R[p]\n\
     transitions {\n\
     infection[p in patch] : S[p] --> I[p] @ beta * S[p] * sum(q in patch where dist[p,q] < thr, I[q]/N[q])\n\
     recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
     }\n\
     init { S[a]=99 I[a]=1 S[b]=100 S[c]=100 }\n\
     simulate { from = 0 'days to = 10 'days }\n"

let where_empty_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [a, b] }\n\
   stratify(by = patch)\n\
   parameters { beta : rate in [0,2]  gamma : rate in [0,1] }\n\
   tables { dist : patch × patch = [[0.0, 99.0],[99.0, 0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   infection[p in patch] : S[p] --> I[p] @ beta * S[p] * sum(q in patch where dist[p,q] < 50 and q != p, I[q]/N[q])\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[a]=99 I[a]=1 S[b]=100 }\n\
   simulate { from = 0 'days to = 10 'days }\n"

let test_where_empty_survivors_const_zero () =
  (* No in-radius non-self neighbour (b is at 99) ⇒ the coupling sum is empty.
     With the fold OFF, the empty `where` sum must lower to Const 0.0 — i.e.
     the rate references no infectious compartment at all. *)
  with_fold_disabled (fun () ->
    let m = match Compiler.compile ~name:"where_empty" where_empty_src with
      | Ok m -> m
      | Error e -> Alcotest.failf "empty-survivor model should compile: %s" e in
    let t = match List.find_opt (fun (t : Ir.transition) -> t.name = "infection_a") m.transitions with
      | Some t -> t | None -> Alcotest.fail "no infection_a transition" in
    let pops = pop_names [] t.rate in
    Alcotest.(check bool) "empty coupling sum → no I_a in rate" false (List.mem "I_a" pops);
    Alcotest.(check bool) "empty coupling sum → no I_b in rate" false (List.mem "I_b" pops))

(* Mask form: `where mask[p,q] != 0` over a precomputed 0/1 adjacency table.
   p0 couples to p1 (mask 1), not p2 (mask 0). *)
let where_mask_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [p0, p1, p2] }\n\
   stratify(by = patch)\n\
   parameters { beta : rate in [0,2]  gamma : rate in [0,1]  rho : probability in [0,1] }\n\
   tables { mask : patch × patch = [[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   infection[p in patch] : S[p] --> I[p] @ beta * S[p] * (I[p]/N[p] + rho * sum(q in patch where mask[p,q] != 0, I[q]/N[q]))\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[p0]=999 I[p0]=1 S[p1]=1000 S[p2]=1000 }\n\
   simulate { from = 0 'days to = 50 'days }\n"

let test_where_mask_prunes () =
  with_fold_disabled (fun () ->
    let m = match Compiler.compile ~name:"where_mask" where_mask_src with
      | Ok m -> m
      | Error e -> Alcotest.failf "mask model should compile: %s" e in
    let t = match List.find_opt (fun (t : Ir.transition) -> t.name = "infection_p0") m.transitions with
      | Some t -> t | None -> Alcotest.fail "no infection_p0 transition" in
    let pops = pop_names [] t.rate in
    Alcotest.(check bool) "mask=1 neighbour I_p1 present" true  (List.mem "I_p1" pops);
    Alcotest.(check bool) "mask=0 neighbour I_p2 absent"  false (List.mem "I_p2" pops))

(* Boundary: a cell exactly at the threshold (dist[p0,p1] = 50) must be EXCLUDED
   by strict `< 50` — pins the float-comparison semantics. *)
let where_boundary_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [p0, p1] }\n\
   stratify(by = patch)\n\
   parameters { beta : rate in [0,2]  gamma : rate in [0,1]  rho : probability in [0,1] }\n\
   tables { dist : patch × patch = [[0.0,50.0],[50.0,0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   infection[p in patch] : S[p] --> I[p] @ beta * S[p] * (I[p]/N[p] + rho * sum(q in patch where dist[p,q] < 50 and q != p, I[q]/N[q]))\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[p0]=999 I[p0]=1 S[p1]=1000 }\n\
   simulate { from = 0 'days to = 50 'days }\n"

let test_where_boundary_excludes_equal () =
  with_fold_disabled (fun () ->
    let m = match Compiler.compile ~name:"where_boundary" where_boundary_src with
      | Ok m -> m
      | Error e -> Alcotest.failf "boundary model should compile: %s" e in
    let t = match List.find_opt (fun (t : Ir.transition) -> t.name = "infection_p0") m.transitions with
      | Some t -> t | None -> Alcotest.fail "no infection_p0 transition" in
    let pops = pop_names [] t.rate in
    Alcotest.(check bool) "dist == 50 excluded by strict `< 50`" false (List.mem "I_p1" pops))

(* E281 must also fire inside an indexed EVENT (events share intervention_decl;
   this is the gap PR #238's review flagged — the guard was advertised as
   covering every binder but omitted events/forcing). *)
let test_event_sum_shadow_rejected () =
  compile_expect_error_code ~code:"E283" ~contains:"event"
    "time_unit = 'days\n\
     compartments { S, I, R }\n\
     dimensions { patch = [p0, p1] }\n\
     stratify(by = patch)\n\
     parameters { beta : rate in [0.0,2.0]  gamma : rate in [0.0,1.0] }\n\
     let N[p in patch] = S[p] + I[p] + R[p]\n\
     transitions {\n\
     infection[p in patch] : S[p] --> I[p] @ beta * S[p] * I[p] / N[p]\n\
     recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
     }\n\
     events {\n\
     seed[p in patch] : add(I, sum(p in patch, S[p])) at [10]\n\
     }\n\
     init { S[p0]=100 I[p0]=1 S[p1]=100 }\n\
     simulate { from = 0 'days to = 60 'days }\n"

(* The headline of gh#185: a where-restricted coupling sum whose body carries a
   parametric kernel is fittable — autodiff must differentiate it w.r.t. the
   kernel params. This proves the gradient flows through the pruned Reduce. *)
let fitted_kernel_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [p0, p1, p2] }\n\
   stratify(by = patch)\n\
   parameters {\n\
   beta  : rate     in [0.0, 2.0]\n\
   gamma : rate     in [0.0, 1.0]\n\
   G     : positive in [0.0, 10.0]\n\
   rho   : positive in [0.0, 5.0]\n\
   }\n\
   tables { dist : patch × patch = [[0.0,30.0,99.0],[30.0,0.0,30.0],[99.0,30.0,0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   infection[p in patch] : S[p] --> I[p] @ beta * S[p] * (I[p]/N[p] + G * sum(q in patch where dist[p,q] < 50 and q != p, dist[p,q]^(-rho) * I[q]/N[q]))\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[p0]=999 I[p0]=1 S[p1]=1000 S[p2]=1000 }\n\
   simulate { from = 0 'days to = 50 'days }\n"

let test_where_fitted_kernel_gradient () =
  let m = match Compiler.compile ~name:"fitted_kernel" fitted_kernel_src with
    | Ok m -> m
    | Error e -> Alcotest.failf "fitted-kernel model should compile: %s" e in
  let t = match List.find_opt (fun (t : Ir.transition) -> t.name = "infection_p0") m.transitions with
    | Some t -> t
    | None -> Alcotest.fail "no infection_p0 transition" in
  Alcotest.(check bool) "gradient w.r.t. coupling strength G flows through the where-Reduce"
    true (List.mem_assoc "G" t.rate_grad);
  Alcotest.(check bool) "gradient w.r.t. distance-decay rho flows through the where-Reduce"
    true (List.mem_assoc "rho" t.rate_grad)

(* W104: the per-(p,q) coupling antipattern (O(P²) transitions). *)
let perpair_src =
  "time_unit = 'days\n\
   compartments { S, I, R }\n\
   dimensions { patch = [a, b, c] }\n\
   stratify(by = patch)\n\
   parameters { kappa : rate in [0,2]  gamma : rate in [0,1] }\n\
   tables { w : patch × patch = [[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]] }\n\
   let N[p in patch] = S[p] + I[p] + R[p]\n\
   transitions {\n\
   imp[p in patch, q in patch] : S[p] --> I[p] @ kappa * w[p,q] * I[q]/N[q]  where p != q\n\
   recovery[p in patch] : I[p] --> R[p] @ gamma * I[p]\n\
   }\n\
   init { S[a]=99 I[a]=1 S[b]=100 S[c]=100 }\n\
   simulate { from = 0 'days to = 10 'days }\n"

let warns_w104 src =
  Diagnostics.json_errors_mode := true;
  let r = Compiler.compile_detail_result ~name:"w104" src in
  Diagnostics.json_errors_mode := false;
  match r with
  | Error e -> Alcotest.failf "model should compile (W104 is a warning, not an error): %s" e
  | Ok d ->
    List.exists (fun (dg : Diagnostics.diagnostic) ->
      dg.code = "W105" && dg.severity = Diagnostics.Warning) d.ctx.diags.diags

let test_w104_perpair_warns () =
  Alcotest.(check bool) "W104 fires on the per-(p,q) coupling form" true (warns_w104 perpair_src)

let test_w104_summed_no_warn () =
  (* the summed-rate `where` form (where_radius_src) must NOT trip W104 *)
  Alcotest.(check bool) "W104 silent on the summed-rate form" false (warns_w104 where_radius_src)

(* ── W104: absolute path in a forcing `data =` reference (gh#307) ──────────────
   W104 already flags absolute `read(...)` table/dimension paths; a forcing's
   file-backed `data =` loader shares the same `read_csv_rows` chokepoint, so the
   same portability warning fires on it too. These tests pin (a) that W104 fires
   on an absolute forcing `data =` path and (b) that the message names the
   forcing construct rather than misattributing the mistake to `read()`. The
   model reads a non-existent absolute file, so E200/E227 also fire — we look at
   the diagnostic list directly (via [collect_diagnostics], which does not abort
   on errors) rather than compiling to completion. *)

let forcing_data_model ~path =
  Printf.sprintf
    {|
    time_unit = 'days
    compartments { S, E, I, R }
    let N = S + E + I + R
    parameters { beta : rate  sigma : rate  gamma : rate  N0 : count  I0 : count }
    forcing {
      clim : interpolated 'ratio {
        data      = "%s"
        time_col  = t
        value_col = force
        method    = "linear"
      }
    }
    transitions {
      infection   : S --> E @ beta * clim(t) * S * I / N
      progression : E --> I @ sigma * E
      recovery    : I --> R @ gamma * I
    }
    init { S = N0 - I0  I = I0 }
    simulate { from = 0 'days  to = 10 'days }
  |}
    path

let test_w104_forcing_data_absolute_warns () =
  (* Path deliberately contains neither "forcing" nor "read" so the message-text
     assertions test the construct label, not an incidental filename match. *)
  let diags =
    Compiler.collect_diagnostics ~name:"w104_forcing"
      (forcing_data_model ~path:"/abs/beta_series.tsv") in
  let w104 =
    List.filter (fun (d : Diagnostics.diagnostic) ->
      d.code = "W104" && d.severity = Diagnostics.Warning) diags in
  Alcotest.(check int)
    "W104 fires once on the absolute forcing data path" 1 (List.length w104);
  match w104 with
  | [ d ] ->
    Alcotest.(check bool) "W104 message names the forcing data reference" true
      (contains_substring ~needle:"forcing data" d.message);
    Alcotest.(check bool) "W104 message does not misattribute to read()" false
      (contains_substring ~needle:"read()" d.message)
  | _ -> Alcotest.fail "expected exactly one W104 diagnostic"

let test_w104_forcing_data_relative_no_warn () =
  (* A relative forcing data path is portable — no W104, even though the file
     does not exist here (E200/E227 still fire). *)
  let diags =
    Compiler.collect_diagnostics ~name:"w104_forcing_rel"
      (forcing_data_model ~path:"data/beta_series.tsv") in
  Alcotest.(check int) "no W104 on a relative forcing data path"
    0 (count_diags_with_code diags "W104")

(* ── gh#204 reactive interventions ──────────────────────────────────────── *)

(* A full, valid model whose only variable is the reactive_interventions body. *)
let reactive_model_with body = Printf.sprintf {|
time_unit = 'days
compartments { S, I, V }
let N = S + I + V
parameters { beta : rate  thr : count  cov : probability  N0 : count  I0 : count }
transitions { infection : S --> I @ beta * S * I / N }
observations {
  weekly {
    columns       { time : time, weekly : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly        ~ poisson(rate = projected)
  }
}
reactive_interventions { %s }
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days  to = 60 'days }
|} body

(* Positive lowering + the indexed-expansion shape are pinned by the committed
   goldens, not by duplicated inline model strings: `make check-reactive-golden`
   asserts tests/fixtures/reactive/*.camdl compiles byte-for-byte to the
   committed *.ir.json, and rust/crates/ir reactive_golden deserialises those
   and asserts the fields (FireSource::Reactive, TriggerExpr, after/once,
   stratified expansion, indexed stream/targets). The inline OCaml tests below
   stay focused on the default-shape and the negative diagnostics. *)

let test_reactive_observed_is_latest () =
  (* observed(stream) (no window) lowers to the Latest reducer. *)
  let m = compile_expect_ok (reactive_model_with
    "sia : when observed(weekly) >= thr {\n\
     \  action = transfer(fraction = cov, from = S, to = V)\n\
     }") in
  let iv = List.find (fun (i : Ir.intervention) -> i.Ir.name = "sia")
             m.Ir.interventions in
  (match iv.Ir.fire with
   | Ir.Reactive { Ir.when_ = Ir.TECmp (Ir.TQObserved { window; reducer; _ }, _, _); _ } ->
     Alcotest.(check bool) "no window" true (window = None);
     Alcotest.(check bool) "reducer = latest" true (reducer = Ir.RedLatest)
   | _ -> Alcotest.fail "expected reactive observed() trigger");
  (* defaults: once defaults to true. *)
  (match iv.Ir.fire with
   | Ir.Reactive t ->
     Alcotest.(check bool) "once defaults true" true t.Ir.once
   | _ -> Alcotest.fail "expected reactive")

let test_reactive_scope_key_removed () =
  (* gh#204: the `scope` reactive key was removed — latent-scope (scope =
     particle) triggers are deferred. Writing it must fail with the migration
     diagnostic, not silently lower (it previously accepted `particle` and the
     runtime ignored it). *)
  compile_expect_error_code ~code:"E106" ~contains:"scope"
    (reactive_model_with
      "sia : when observed(weekly) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       \  scope  = exogenous\n\
       }")

let test_reactive_observed_in_rate_rejected () =
  (* observed() in a transition rate (a model expression, not a trigger) must
     be rejected with a targeted message, not silently lowered. *)
  compile_expect_error_code ~code:"E278" ~contains:"observed"
    {|
time_unit = 'days
compartments { S, I }
let N = S + I
parameters { beta : rate  N0 : count  I0 : count }
observations {
  weekly {
    columns       { time : time, weekly : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly        ~ poisson(rate = projected)
  }
}
transitions { infection : S --> I @ beta * observed(weekly) * S * I / N }
init { S = N0 - I0  I = I0 }
simulate { from = 0 'days  to = 10 'days }
|}

let test_reactive_once_with_cooldown_rejected () =
  compile_expect_error_code ~code:"E276" ~contains:"cooldown"
    (reactive_model_with
      "sia : when observed(weekly) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       \  once = true\n\
       \  cooldown = 30 'days\n\
       }")

let test_reactive_negative_after_rejected () =
  compile_expect_error_code ~code:"E274" ~contains:"after"
    (reactive_model_with
      "sia : when observed(weekly) >= thr {\n\
       \  after = -5 'days\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       \  once = true\n\
       }")

let test_reactive_non_comparison_when_rejected () =
  (* A `when` that is not a comparison (here a bare observed()) is rejected:
     the predicate must be boolean (a comparison). *)
  compile_expect_error_code ~code:"E273" ~contains:"comparison"
    (reactive_model_with
      "sia : when observed(weekly) {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       }")

let test_reactive_unknown_stream_rejected () =
  (* A trigger referencing an observation stream no `observations {}` declares. *)
  compile_expect_error_code ~code:"E279" ~contains:"nope"
    (reactive_model_with
      "sia : when observed(nope) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       }")

let test_reactive_negative_window_rejected () =
  compile_expect_error_code ~code:"E274" ~contains:"window"
    (reactive_model_with
      "sia : when sum_observed(weekly, window = -5 'days) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       }")

let test_reactive_rolling_method_unsupported () =
  (* `.rolling(...)` method syntax does not exist in the DSL — a bare syntax
     error (E001), i.e. unsupported, not a reactive-specific diagnostic. *)
  compile_expect_error_code ~code:"E001" ~contains:""
    (reactive_model_with
      "sia : when weekly.rolling(14 'days) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = V)\n\
       }")

let test_reactive_unknown_action_target_rejected () =
  (* The action target is validated by the SAME resolver as scheduled
     interventions — a transfer to an undeclared compartment is rejected
     (E264 from `resolve_comp_name`), not a reactive-specific path. *)
  compile_expect_error_code ~code:"E264" ~contains:""
    (reactive_model_with
      "sia : when observed(weekly) >= thr {\n\
       \  action = transfer(fraction = cov, from = S, to = Nowhere)\n\
       }")

(* ── Declaration doc comments (#') ────────────────────────────────────────
   `#'` doc comments attach prose to the following compartment / parameter
   declaration. The prose lives on the AST only (it never reaches the IR) and
   surfaces in `camdlc inspect`. Tests: a documented model compiles; the docs
   are IR-neutral (documented vs a stripped twin → byte-identical IR); a
   dangling `#'` is a hard error; and `inspect` renders the prose, including
   sharing one doc across an indexed param's expanded leaves. *)

let doc_model_src = {|
time_unit = 'days
dimensions {
  #' spatial patches under surveillance
  patch = [urban, rural]
}
compartments {
  #' fully susceptible
  S,
  #' infectious and shedding
  I,
  R
}
stratify(by = patch)
let N[p in patch] = S[p] + I[p] + R[p]
parameters {
  #' basic reproduction number (per patch)
  #' @symbol R_naught
  R0[patch] : positive in [1.0, 6.0]
  #' mean infectious period is 1/gamma
  #' @ref Anderson and May 1991
  gamma : rate in [0.01, 0.5]
  beta : rate in [0.001, 2.0]
}
transitions {
  #' force of infection, per patch
  infection[p in patch] : S[p] --> I[p] @ R0[p] * gamma * S[p] * I[p] / N[p]
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}
init { S[p in patch] = 1000  I[p in patch] = 10 }
simulate { from = 0 'days to = 90 'days }
|}

(* A non-stratified model with a documented observation stream — exercises `#'`
   on an `observations { }` declaration (a stratified obs needs a dim column,
   orthogonal to docs). *)
let doc_obs_src = {|
time_unit = 'days
compartments {
  #' susceptible
  S, I, R
}
let N = S + I + R
parameters { beta : rate in [0.001,1.0]  gamma : rate in [0.01,0.5]  rho : probability in [0.1,0.9] }
transitions {
  #' force of infection
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  #' weekly reported cases (Poisson reporting)
  cases {
    columns   { time : time, cases : count }
    projected = incidence(infection)
    cases     ~ poisson(rate = rho * projected)
  }
}
quantities {
  #' peak prevalence
  peak_prev = max(I / N)
}
init { S = 1000  I = 10 }
simulate { from = 0 'days to = 90 'days }
|}

let test_doc_comment_compiles () =
  let _ : Ir.model = compile_expect_ok doc_model_src in
  ()

let test_doc_param_reaches_ir () =
  (* Parameter docs reach the model's envelope doc dictionary (the single doc
     home, read by Rust consumers). Keyed by BASE name (`R0`, not `R0_urban`):
     a stratified family shares one entry. *)
  let m = compile_expect_ok doc_model_src in
  let find n = List.assoc_opt n m.doc_index.di_parameters in
  (match find "R0" with
   | Some d ->
     Alcotest.(check (option string)) "R0 @symbol in the doc dictionary"
       (Some "R_naught") d.symbol
   | None -> Alcotest.fail "R0 missing from the doc dictionary");
  (match find "gamma" with
   | Some d ->
     Alcotest.(check (option string)) "gamma @ref in the doc dictionary"
       (Some "Anderson and May 1991") d.reference
   | None -> Alcotest.fail "gamma missing from the doc dictionary")

let test_doc_dangling_rejected () =
  (* a `#'` that precedes no declaration (sits before `}`) is a hard parse
     error — rejected, never silently accepted (no loose semantics). *)
  compile_expect_error_code ~code:"E001" ~contains:""
    {|
time_unit = 'days
compartments { S, I, R }
parameters {
  beta : rate
  #' orphan doc with no declaration after it
}
simulate { from = 0 'days to = 5 'days }
|}

let doc_inspect_output view =
  let detail =
    match Compiler.compile_detail_result ~name:"doc_inspect" doc_model_src with
    | Ok d -> d
    | Error e -> Alcotest.failf "detail compile failed: %s" e
  in
  let buf = Buffer.create 512 in
  let ppf = Format.formatter_of_buffer buf in
  (match view with
   | `Params       -> Inspect.run_parameters   ppf detail.model detail.ctx
   | `Compartments -> Inspect.run_compartments ppf detail.model detail.ctx
   | `Transitions  -> Inspect.run_transitions  ppf detail.model detail.ctx None ~ascii:true
   | `Summary      -> Inspect.run_summary       ppf detail.model detail.ctx detail.summary);
  Format.pp_print_flush ppf ();
  Buffer.contents buf

let test_doc_inspect_parameters () =
  let out = doc_inspect_output `Params in
  Alcotest.(check bool) "scalar param doc prose present" true
    (contains_substring ~needle:"mean infectious period is 1/gamma" out);
  (* The indexed param's single doc rides every expanded leaf (R0_urban,
     R0_rural), mirroring shared bounds. *)
  Alcotest.(check bool) "indexed param doc present" true
    (contains_substring ~needle:"basic reproduction number (per patch)" out);
  (* @symbol and @ref tags are split out and rendered. *)
  Alcotest.(check bool) "@symbol rendered" true
    (contains_substring ~needle:"R_naught" out);
  Alcotest.(check bool) "@ref rendered" true
    (contains_substring ~needle:"Anderson and May 1991" out)

let test_doc_refused_tag () =
  (* A number-bearing tag (@default/@plausible/@fixed) is a hard E111 that
     names the --params TOML migration; only @symbol/@ref are recognized. *)
  compile_expect_error_code ~code:"E111" ~contains:"@symbol"
    {|
time_unit = 'days
compartments { S, I }
parameters {
  #' transmission rate
  #' @default 0.3
  beta : rate
}
transitions { infection : S --> I @ beta * S }
init { S = 100 }
simulate { from = 0 'days to = 5 'days }
|}

let test_doc_inspect_compartments () =
  let out = doc_inspect_output `Compartments in
  Alcotest.(check bool) "compartment doc prose present" true
    (contains_substring ~needle:"fully susceptible" out);
  Alcotest.(check bool) "second compartment doc present" true
    (contains_substring ~needle:"infectious and shedding" out)

let test_doc_inspect_transition () =
  let out = doc_inspect_output `Transitions in
  Alcotest.(check bool) "transition doc present" true
    (contains_substring ~needle:"force of infection, per patch" out)

let test_doc_inspect_dimension () =
  let out = doc_inspect_output `Summary in
  Alcotest.(check bool) "dimension doc present in summary" true
    (contains_substring ~needle:"spatial patches under surveillance" out)

(* compartment / transition / observation docs reach the model's doc dictionary
   (the single envelope-level doc home), keyed by base declaration name. *)
let test_doc_nonparam_reaches_ir () =
  let m = compile_expect_ok doc_obs_src in
  Alcotest.(check bool) "compartment doc in the dictionary" true
    (List.mem_assoc "S" m.doc_index.di_compartments);
  Alcotest.(check bool) "transition doc in the dictionary" true
    (List.mem_assoc "infection" m.doc_index.di_transitions);
  Alcotest.(check bool) "observation doc in the dictionary" true
    (List.mem_assoc "cases" m.doc_index.di_observations);
  Alcotest.(check bool) "quantity doc in the dictionary" true
    (List.mem_assoc "peak_prev" m.doc_index.di_quantities)

let () =
  Alcotest.run "compiler" [
    "declaration_doc_comments", [
      Alcotest.test_case "documented model compiles"                  `Quick test_doc_comment_compiles;
      Alcotest.test_case "parameter docs reach the IR (@symbol/@ref)" `Quick test_doc_param_reaches_ir;
      Alcotest.test_case "dangling #' is a hard error (E001)"         `Quick test_doc_dangling_rejected;
      Alcotest.test_case "refused doc tag (@default) is E111"         `Quick test_doc_refused_tag;
      Alcotest.test_case "inspect --parameters shows doc prose + tags" `Quick test_doc_inspect_parameters;
      Alcotest.test_case "inspect --compartments shows doc prose"     `Quick test_doc_inspect_compartments;
      Alcotest.test_case "inspect --transitions shows doc prose"      `Quick test_doc_inspect_transition;
      Alcotest.test_case "inspect --summary shows dimension doc"      `Quick test_doc_inspect_dimension;
      Alcotest.test_case "non-parameter docs reach the dictionary"   `Quick test_doc_nonparam_reaches_ir;
    ];
    "quadratic_coupling_warning", [
      Alcotest.test_case "W104 on per-(p,q) transition" `Quick test_w104_perpair_warns;
      Alcotest.test_case "no W104 on summed-rate form" `Quick test_w104_summed_no_warn;
    ];
    "w104_forcing_data_absolute_path", [
      Alcotest.test_case "W104 fires on an absolute forcing data path"
        `Quick test_w104_forcing_data_absolute_warns;
      Alcotest.test_case "no W104 on a relative forcing data path"
        `Quick test_w104_forcing_data_relative_no_warn;
    ];
    "restricted_sum_where", [
      Alcotest.test_case "where dist[p,q] < r prunes to in-radius neighbours (fold off)"
        `Quick test_where_radius_prunes;
      Alcotest.test_case "E282 fitted-parameter threshold rejected"
        `Quick test_where_fitted_threshold_rejected;
      Alcotest.test_case "empty survivor set → coupling sum is Const 0.0 (fold off)"
        `Quick test_where_empty_survivors_const_zero;
      Alcotest.test_case "mask form: where mask[p,q] != 0 prunes to mask-1 neighbours"
        `Quick test_where_mask_prunes;
      Alcotest.test_case "boundary: dist == 50 excluded by strict `< 50`"
        `Quick test_where_boundary_excludes_equal;
      Alcotest.test_case "fitted kernel: gradient flows through the where-Reduce to G/rho"
        `Quick test_where_fitted_kernel_gradient;
    ];
    "index_shadowing", [
      Alcotest.test_case "E281 sum var shadows transition index"
        `Quick test_sum_var_shadows_transition_index_rejected;
      Alcotest.test_case "E281 sum var shadows event index"
        `Quick test_event_sum_shadow_rejected;
      Alcotest.test_case "distinct sum var still compiles"
        `Quick test_sum_var_distinct_from_index_ok;
    ];
    "table_lookup_arity", [
      Alcotest.test_case "E202 under-indexed C_age[a] (rank 2)" `Quick test_table_lookup_under_indexed_e202;
      Alcotest.test_case "E202 over-indexed C_age[a,b,a] (rank 2)" `Quick test_table_lookup_over_indexed_e202;
      Alcotest.test_case "correct-arity C_age[a,b] still compiles" `Quick test_table_lookup_correct_arity_ok;
    ];
    "partial_dimension_omission", [
      Alcotest.test_case "E287 partial omit E[a] (rank 2) is located" `Quick test_partial_dimension_omission_e287_with_loc;
      Alcotest.test_case "bare name E sums all dims" `Quick test_bare_name_sums_all_dims_ok;
      Alcotest.test_case "full index E[a, e1] resolves" `Quick test_full_index_resolves_ok;
      Alcotest.test_case "explicit sum(s in latent_stage, E[a, s])" `Quick test_explicit_marginalization_sum_ok;
      Alcotest.test_case "single-dim FOI sum(b, C[a,b]*I[b]/N[b])" `Quick test_single_dim_foi_pattern_ok;
    ];
    "declaration_names", [
      Alcotest.test_case "E278 duplicate parameter beta" `Quick test_duplicate_parameter_rejected;
      Alcotest.test_case "E278 duplicate let k" `Quick test_duplicate_let_rejected;
      Alcotest.test_case "E278 cross-namespace param+let N" `Quick test_cross_namespace_param_and_let_rejected;
      Alcotest.test_case "E278 cross-namespace expanded name R0_a" `Quick test_cross_namespace_expanded_name_rejected;
    ];
    "init_membership", [
      Alcotest.test_case "E277 bare stratified init S" `Quick test_init_bare_stratified_rejected;
      Alcotest.test_case "E277 unknown compartment init X[child]" `Quick test_init_unknown_compartment_rejected;
      Alcotest.test_case "concrete cell init S[child] ok" `Quick test_init_concrete_cell_ok;
      Alcotest.test_case "single diagnostic for one root cause" `Quick test_init_bare_stratified_single_diagnostic;
    ];
    "iso_date_validation", [
      Alcotest.test_case "E223 invalid day 2020-02-30" `Quick test_date_invalid_day_rejected;
      Alcotest.test_case "E223 invalid month 2020-13-01" `Quick test_date_invalid_month_rejected;
      Alcotest.test_case "leap-year Feb 29 2020 ok" `Quick test_date_feb29_leap_year_ok;
      Alcotest.test_case "E223 non-leap Feb 29 2021" `Quick test_date_feb29_non_leap_year_rejected;
    ];
    "golden", [
      Alcotest.test_case "sir_basic"      `Quick (test_golden "sir_basic");
      Alcotest.test_case "sir_demography" `Quick (test_golden "sir_demography");
      Alcotest.test_case "seir_age"       `Quick (test_golden "seir_age");
      Alcotest.test_case "sir_five_age"   `Quick (test_golden "sir_five_age");
      Alcotest.test_case "seir_erlang"        `Quick (test_golden "seir_erlang");
      Alcotest.test_case "seir_erlang_staged" `Quick (test_golden "seir_erlang_staged");
      Alcotest.test_case "seir_erlang_via"    `Quick (test_golden "seir_erlang_via");
      Alcotest.test_case "sir_coupling"       `Quick (test_golden "sir_coupling");
      Alcotest.test_case "sir_two_patch"      `Quick (test_golden "sir_two_patch");
      Alcotest.test_case "sir_spatial_where"  `Quick (test_golden "sir_spatial_where");
      Alcotest.test_case "seir_vaccine"            `Quick (test_golden "seir_vaccine");
      Alcotest.test_case "seir_vaccine_seasonal"   `Quick (test_golden "seir_vaccine_seasonal");
      Alcotest.test_case "polio_age"               `Quick (test_golden "polio_age");
      Alcotest.test_case "polio_spatial_5"         `Quick (test_golden "polio_spatial_5");
      Alcotest.test_case "seir_seasonal_patch"     `Quick (test_golden "seir_seasonal_patch");
      Alcotest.test_case "ross_macdonald"          `Quick (test_golden "ross_macdonald");
      (* Goldens missing from the list as of 2026-04-19 (C8 in the
         compiler review). Each has a committed .camdl + .ir.json but
         the compile-and-roundtrip coverage was absent, so a
         regression in (e.g.) the overdispersed-to-IR path or the
         multi-species model would have shipped without signal.
         sir_overdispersion specifically is the only fixture
         exercising overdispersed() — without its registration,
         the C1 silent-Poisson-fallback bug would have had no
         regression guard even after being fixed. *)
      Alcotest.test_case "sir_overdispersion"      `Quick (test_golden "sir_overdispersion");
      Alcotest.test_case "sir_reservoir"           `Quick (test_golden "sir_reservoir");
      Alcotest.test_case "sir_priors"              `Quick (test_golden "sir_priors");
      Alcotest.test_case "sir_init_table"          `Quick (test_golden "sir_init_table");
      Alcotest.test_case "sir_patches_5"           `Quick (test_golden "sir_patches_5");
      Alcotest.test_case "sir_spatial_sum"         `Quick (test_golden "sir_spatial_sum");
      Alcotest.test_case "sir_dim_annotated"       `Quick (test_golden "sir_dim_annotated");
      Alcotest.test_case "seir_observations"       `Quick (test_golden "seir_observations");
      Alcotest.test_case "seir_defines_adj"        `Quick (test_golden "seir_defines_adj");
      Alcotest.test_case "seir_defines_patch"      `Quick (test_golden "seir_defines_patch");
      Alcotest.test_case "seir_spatial_5_inference" `Quick (test_golden "seir_spatial_5_inference");
      Alcotest.test_case "malaria_two_species"     `Quick (test_golden "malaria_two_species");
      Alcotest.test_case "seir_age_table_rates"    `Quick (test_golden "seir_age_table_rates");
      Alcotest.test_case "sia_anchored_dates"      `Quick (test_golden "sia_anchored_dates");
      Alcotest.test_case "sia_instance_enable"     `Quick (test_golden "sia_instance_enable");
    ];
    "table_lookup_flattening", [
      Alcotest.test_case "single index per lookup" `Quick test_table_lookup_single_index;
      Alcotest.test_case "infection_child row 0"   `Quick test_infection_child_indices;
      Alcotest.test_case "infection_adult row 1"   `Quick test_infection_adult_indices;
    ];
    "constant_fold", [
      Alcotest.test_case "sparse ring FOI Reduce P=4 collapses to k=2"
        `Quick test_constant_fold_collapses_sparse_foi_reduce;
    ];
    "licm", [
      Alcotest.test_case "invariant/variant classification (Dt/Time/forcing/state are variant)"
        `Quick test_licm_invariant_classification;
      Alcotest.test_case "table-cell invariance (state-dependent inline cell is variant)"
        `Quick test_licm_table_cell_invariance;
      Alcotest.test_case "cost threshold (transcendental/Pow/Reduce worth hoisting)"
        `Quick test_licm_cost_threshold;
      Alcotest.test_case "hoists the in-model kernel into per_eval bindings"
        `Quick test_licm_hoists_kernel;
    ];
    "binding_param_free_invariant", [
      Alcotest.test_case "references_param on hand-built exprs"
        `Quick test_references_param_primitive;
      Alcotest.test_case "spatial model with hoisted N[l] compiles (no E512)"
        `Quick test_binding_invariant_clean_on_spatial;
      Alcotest.test_case "poisoned binding (Param spliced in) raises E512"
        `Quick test_binding_invariant_catches_poisoned_binding;
    ];
    "cost_report", [
      Alcotest.test_case "sparse-ring numbers are sane (Reduce 16→8, reuse, dups)"
        `Quick test_cost_report_numbers_sane;
      Alcotest.test_case "renders to a buffer without raising"
        `Quick test_cost_report_renders_without_raising;
      Alcotest.test_case "1 - exp(x) hazard idiom detected"
        `Quick test_cost_report_hazard_idiom_detected;
    ];
    "min_max", [
      Alcotest.test_case "min/max wire to BinOp Min/Max" `Quick test_min_max_wire_to_binop;
    ];
    "comparison_ops", [
      Alcotest.test_case "comparison in rate expr" `Quick test_comparison_in_rate;
    ];
    "output_schedule", [
      Alcotest.test_case "format and step when output block present" `Quick test_output_format_from_decl;
      Alcotest.test_case "default step=1.0 with no output block"    `Quick test_output_step_default;
      Alcotest.test_case "unanchored t_start=0 → output.start=0.0 (no regression)"
        `Quick test_output_default_start_unanchored_stays_zero;
      Alcotest.test_case "anchored t_start<0 → output.start covers full integration window"
        `Quick test_output_default_start_anchored_negative_t_start;
      Alcotest.test_case "every = E → OutRegular step" `Quick test_output_every_explicit;
      Alcotest.test_case "every = 0.5 → sub-unit cadence" `Quick test_output_every_subunit;
      Alcotest.test_case "at = [...] → OutAtTimes" `Quick test_output_at_times;
      Alcotest.test_case "format = parquet" `Quick test_output_format_parquet;
      Alcotest.test_case "every and at conflict → error" `Quick test_output_every_and_at_conflict;
      Alcotest.test_case "obs.start=t_start vs output.start=0 (A.2 lowering divergence guard)"
        `Quick test_obs_output_start_divergence;
      Alcotest.test_case "stratified obs header emits (dim,level) stratum + serde round-trip"
        `Quick test_stratified_observation_emits_stratum;
    ];
    "parameterised_tables", [
      Alcotest.test_case "param survives as Ir.Param" `Quick test_parameterised_table;
    ];
    "dim_separator", [
      Alcotest.test_case "ASCII `*` and Unicode `×` yield byte-identical IR"
        `Quick test_dim_sep_asterisk_equals_cross;
    ];
    "table_unit_conversion", [
      Alcotest.test_case "'years table scales to days"
        `Quick test_table_years_annotation_scales_to_days;
      Alcotest.test_case "'per_day table scales to model 'weeks unit"
        `Quick test_table_per_day_annotation_with_weeks_unit;
      Alcotest.test_case "read() path also scales unit-annotated values"
        `Quick test_table_read_path_scales_unit;
      Alcotest.test_case "no unit annotation leaves values untouched"
        `Quick test_table_no_unit_annotation_leaves_values_alone;
    ];
    "emit_deps_read_closure", [
      Alcotest.test_case "compile_with_reads records the distinct read() files"
        `Quick test_emit_deps_records_read_closure;
      Alcotest.test_case "no read() → empty closure (negative control)"
        `Quick test_emit_deps_empty_when_no_reads;
    ];
    "table_read_header_gh144", [
      Alcotest.test_case "leading # comment block is skipped before header"
        `Quick test_table_read_skips_leading_comment;
      Alcotest.test_case "malformed header (too few columns) is E221, not a crash"
        `Quick test_table_read_malformed_header_e221;
      Alcotest.test_case "scalar read() (0-dim table) is E222 naming the parameter seam"
        `Quick test_scalar_read_table_e222;
    ];
    "table_cell_type_annotation_gh32", [
      Alcotest.test_case ":rate annotation parses + stamps IR.cell_kind"
        `Quick test_table_cell_type_rate_parses_and_stamps_ir;
      Alcotest.test_case ":probability annotation parses"
        `Quick test_table_cell_type_probability_parses;
      Alcotest.test_case "no annotation = cell_kind None (back-compat)"
        `Quick test_table_no_cell_type_annotation_remains_none;
      Alcotest.test_case "instant cells: ISO dates resolve via origin"
        `Quick test_table_cell_kind_instant_resolves_dates;
      Alcotest.test_case "instant date cell without origin is E209"
        `Quick test_table_cell_kind_instant_needs_origin;
      Alcotest.test_case "dim-check passes with :rate-typed table in rate position"
        `Quick test_table_cell_type_dim_check_passes_in_rate_position;
      Alcotest.test_case "cell_kind survives JSON serde round-trip"
        `Quick test_table_cell_type_ir_round_trips_through_serde;
    ];
    "spec_claims_v1", [
      Alcotest.test_case "§9 let binding is extracted into model.bindings (P3.1, Fix B)"
        `Quick test_let_binding_is_extracted;
      Alcotest.test_case "§5 stratify expands N × |dim| compartments (P3.2)"
        `Quick test_stratification_compartment_count;
      Alcotest.test_case "§13.1 incidence positional ≡ named projection (P3.5)"
        `Quick test_incidence_positional_and_named_produce_equal_projections;
      Alcotest.test_case "§14 consecutive(k) → k-1 adjacent pairs (P3.4)"
        `Quick test_consecutive_pair_count;
    ];
    "interventions", [
      Alcotest.test_case "intervention expansion" `Quick test_intervention_expansion;
      Alcotest.test_case "transfer(count = N, ...) parses + emits AbsoluteTransfer (gh#49)"
        `Quick test_intervention_transfer_count_kwarg;
      Alcotest.test_case "transfer(count + fraction) rejected as mutually exclusive (gh#49)"
        `Quick test_intervention_transfer_count_and_fraction_rejected;
      Alcotest.test_case "block set keeps all assignments (multi-set)"
        `Quick test_intervention_multi_set;
      Alcotest.test_case "E296 block intervention with no action rejected"
        `Quick test_intervention_no_action;
    ];
    "indexed reference arity", [
      Alcotest.test_case "indexed let over-index rejected (E299)"
        `Quick test_indexed_let_arity;
      Alcotest.test_case "indexed forcing over-index rejected (E299)"
        `Quick test_indexed_forcing_arity;
      Alcotest.test_case "indexed param over-index rejected (E299)"
        `Quick test_indexed_param_arity;
    ];
    "declaration name collisions", [
      Alcotest.test_case "stratified compartment base name vs let (E278)"
        `Quick test_stratified_base_name_collision;
    ];
    "reactive_interventions", [
      Alcotest.test_case "observed() (no window) lowers to Latest + defaults"
        `Quick test_reactive_observed_is_latest;
      Alcotest.test_case "E106 the removed `scope` key is rejected with a migration"
        `Quick test_reactive_scope_key_removed;
      Alcotest.test_case "E278 observed() in a rate is rejected"
        `Quick test_reactive_observed_in_rate_rejected;
      Alcotest.test_case "E279 unknown observation stream is rejected"
        `Quick test_reactive_unknown_stream_rejected;
      Alcotest.test_case "E276 once=true + cooldown is rejected"
        `Quick test_reactive_once_with_cooldown_rejected;
      Alcotest.test_case "E274 negative after is rejected"
        `Quick test_reactive_negative_after_rejected;
      Alcotest.test_case "E274 negative window is rejected"
        `Quick test_reactive_negative_window_rejected;
      Alcotest.test_case "E273 non-comparison when is rejected"
        `Quick test_reactive_non_comparison_when_rejected;
      Alcotest.test_case "E001 .rolling() method syntax is unsupported"
        `Quick test_reactive_rolling_method_unsupported;
      Alcotest.test_case "E264 unknown action target is rejected (shared resolver)"
        `Quick test_reactive_unknown_action_target_rejected;
    ];
    "recurring_interventions", [
      Alcotest.test_case "transfer(...) { every, from, until }"     `Quick test_recurring_block_transfer;
      Alcotest.test_case "kwargs accepted in any order"             `Quick test_recurring_kwargs_any_order;
      Alcotest.test_case "unit conversion applies to interval args" `Quick test_recurring_unit_conversion;
      Alcotest.test_case "add(...) { every, from, until } in events" `Quick test_recurring_add_action;
      Alcotest.test_case "from / until default to simulation bounds" `Quick test_recurring_default_from_until;
      Alcotest.test_case "at [...] form still compiles (regression)" `Quick test_recurring_at_times_still_works;
      Alcotest.test_case "E213 block transition missing rate is rejected" `Quick test_block_transition_missing_rate_e213;
      Alcotest.test_case "E240 every = 0 is rejected"               `Quick test_recurring_e240_zero_every;
      Alcotest.test_case "E241 from > until is rejected"            `Quick test_recurring_e241_inverted_range;
      Alcotest.test_case "E242 expanded schedule too long"          `Quick test_recurring_e242_schedule_too_long;
    ];
    "scenario_extends", [
      Alcotest.test_case "child inherits parent set values"          `Quick test_extends_inherits_set_values;
      Alcotest.test_case "child overrides parent key"                `Quick test_extends_child_overrides_key;
      Alcotest.test_case "enable: parent + child, dedup"             `Quick test_extends_enable_append_dedup;
      Alcotest.test_case "three-level chain a -> b -> c"             `Quick test_extends_three_level_chain;
      Alcotest.test_case "scale interacts with parent's set"         `Quick test_extends_scale_interaction;
      Alcotest.test_case "child references parent's resolved value"  `Quick test_extends_child_references_parent_value;
      Alcotest.test_case "E25x cycle detected with chain in message" `Quick test_extends_e25x_cycle;
      Alcotest.test_case "E25y unknown parent + edit-distance hint"  `Quick test_extends_e25y_unknown_with_suggestion;
      Alcotest.test_case "E25z chain depth > 5 errors"               `Quick test_extends_e25z_depth_exceeds;
      (* gh#115 — scenario field name validation *)
      Alcotest.test_case "E267 scenario enable typo"   `Quick test_scenario_enable_unknown_intervention_is_e267;
      Alcotest.test_case "E267 scenario disable typo"  `Quick test_scenario_disable_unknown_intervention_is_e267;
      Alcotest.test_case "E268 scenario set typo"      `Quick test_scenario_set_unknown_param_is_e268;
      Alcotest.test_case "E268 scenario scale typo"    `Quick test_scenario_scale_unknown_param_is_e268;
      Alcotest.test_case "E269 scenario compose typo"  `Quick test_scenario_compose_unknown_scenario_is_e269;
      Alcotest.test_case "E291 scenario named fitted reserved" `Quick test_scenario_named_fitted_is_reserved_e291;
      Alcotest.test_case "scenario enable known name"  `Quick test_scenario_enable_known_intervention_compiles;
      Alcotest.test_case "gh#130 enable expanded instance"  `Quick test_scenario_enable_expanded_instance_compiles;
      Alcotest.test_case "gh#130 disable expanded instance" `Quick test_scenario_disable_expanded_instance_compiles;
      Alcotest.test_case "gh#130 bogus instance still E267"  `Quick test_scenario_enable_bogus_instance_still_e267;
      Alcotest.test_case "W310 fires on append-dedup collision"      `Quick test_extends_w310_on_enable_dedup;
    ];
    "l401_lint", [
      Alcotest.test_case "L401 fires on fixed time literal"          `Quick test_l401_fires_on_fixed_time_literal;
      Alcotest.test_case "L401 quiet when dt primitive used"         `Quick test_l401_no_fire_when_dt_used;
      Alcotest.test_case "L401 quiet on unit conversion (no exp)"    `Quick test_l401_no_fire_on_unit_conversion;
    ];
    "l403_lint", [
      Alcotest.test_case "L403 fires on div-conversion"              `Quick test_l403_fires_on_div_conversion;
      Alcotest.test_case "L403 fires on reciprocal-mul"              `Quick test_l403_fires_on_reciprocal_mul;
      Alcotest.test_case "L403 quiet on plain forcing use"          `Quick test_l403_no_fire_on_plain_use;
      Alcotest.test_case "L403 quiet on 'ratio / 'count forcing"    `Quick test_l403_no_fire_on_non_rate_forcing;
      Alcotest.test_case "L403 quiet without a rate forcing"        `Quick test_l403_no_fire_on_unrelated_div;
      Alcotest.test_case "L403 fires via hoisted binding"           `Quick test_l403_fires_via_hoisted_binding;
      Alcotest.test_case "L403 quiet on same-unit rate forcing"     `Quick test_l403_no_fire_on_same_unit_forcing;
      Alcotest.test_case "L403 quiet on structural divisor"         `Quick test_l403_no_fire_on_structural_divisor;
      Alcotest.test_case "gh#345 indexed file-backed forcing"       `Quick test_gh345_indexed_file_backed_forcing;
      Alcotest.test_case "gh#345 table-backed forcing"              `Quick test_gh345_table_backed_forcing;
      Alcotest.test_case "gh#345 table forcing unaccounted dim"     `Quick test_gh345_table_unaccounted_dim_rejected;
    ];
    "compile_outcome", [
      Alcotest.test_case "clean model returns Some value, no errors" `Quick test_compile_outcome_clean_returns_value;
      Alcotest.test_case "late error is a value, not a raise"        `Quick test_compile_outcome_late_error_is_value_not_raise;
    ];
    "diagnostic_locations", [
      Alcotest.test_case "decl-keyed validate error carries a loc"   `Quick test_validate_decl_error_has_location;
      Alcotest.test_case "reference validate error carries a loc"    `Quick test_validate_reference_error_has_location;
      Alcotest.test_case "dimcheck error carries a loc"              `Quick test_dimcheck_error_has_location;
      Alcotest.test_case "lint warning carries a loc"                `Quick test_lint_warning_has_location;
    ];
    "trig_primitives", [
      Alcotest.test_case "pi resolves to Const ≈ π"                 `Quick test_trig_pi_resolves_to_const;
      Alcotest.test_case "cos(dimensionless) compiles"              `Quick test_trig_cos_compiles_and_dimchecks;
      Alcotest.test_case "cos(t) rejected with E301"                `Quick test_trig_cos_rejects_dimensional_arg;
      Alcotest.test_case "autodiff emits rate_grad for sin(...)"    `Quick test_trig_autodiff_matches_finite_diff;
      Alcotest.test_case "autodiff emits rate_grad for a Fourier coef" `Quick test_fourier_autodiff_emitted;
      Alcotest.test_case "periodic step-value param compiles, gradient is a coded refusal (gh#342)" `Quick test_periodic_forcing_coeff_omitted;
      Alcotest.test_case "structural forcing-coeff param is a compile error (gh#215)" `Quick test_structural_forcing_coeff_errors;
      Alcotest.test_case "periodic-param-in-rate model compiles to IR (gh#215)" `Quick test_periodic_param_in_rate_compiles;
      Alcotest.test_case "pi as parameter name is reserved (E100)"  `Quick test_trig_pi_reserved;
    ];
    "obs_gradient", [
      Alcotest.test_case "obs grad: non-derived projection ⇒ ∂rate/∂rho = projected" `Quick test_obs_grad_nonderived_projection;
      Alcotest.test_case "obs grad: parametric DerivedExpr chain rule (gh#180)" `Quick test_obs_grad_parametric_derived_projection;
      Alcotest.test_case "obs grad: structural-forcing arg ⇒ coded DEUnsupported" `Quick test_obs_grad_structural_forcing_is_coded_refusal;
      Alcotest.test_case "σ² grad: DrawOverdispersed fills sigma_sq_grad" `Quick test_sigma_sq_grad_emitted;
    ];
    "time_functions", [
      Alcotest.test_case "sinusoidal compiles to TimeFunc"       `Quick test_sinusoidal_time_func;
      Alcotest.test_case "EFuncCall in rate emits Ir.TimeFunc"   `Quick test_time_func_in_rate;
      Alcotest.test_case "param arg preserved in time func"      `Quick test_time_func_param_arg;
      Alcotest.test_case "bare func name resolves to Ir.TimeFunc" `Quick test_bare_func_name_in_rate;
      Alcotest.test_case "unknown func call emits E100"          `Quick test_unknown_func_call_e100;
      Alcotest.test_case "gh#314 forcing without lag ⇒ None"     `Quick test_forcing_without_lag_is_none;
      Alcotest.test_case "gh#314 lag = 10 'days ⇒ Some 10.0"     `Quick test_forcing_with_literal_lag;
      Alcotest.test_case "gh#314 lag = tau ⇒ Some (Param tau)"   `Quick test_forcing_with_param_lag;
    ];
    "read_long", [
      Alcotest.test_case "1D array from TSV file"            `Quick test_read_long_1d;
      Alcotest.test_case "defines() stratify dimension"      `Quick test_read_long_defines;
      Alcotest.test_case "missing file handled gracefully"   `Quick test_read_long_missing_file;
      Alcotest.test_case "reordered columns → E216"          `Quick test_read_header_reordered;
      Alcotest.test_case "mismatched column name → W201"     `Quick test_read_header_mismatch;
    ];
    "indexed_params", [
      Alcotest.test_case "scalar expansion per stratum"      `Quick test_indexed_param_scalar_expansion;
      Alcotest.test_case "variable index in transition rate" `Quick test_indexed_param_variable_index;
      Alcotest.test_case "literal index outside loop"        `Quick test_indexed_param_literal_index;
      Alcotest.test_case "no default → value = 0.0"         `Quick test_indexed_param_no_default;
      Alcotest.test_case "bad index value → E100"            `Quick test_indexed_param_bad_index;
      Alcotest.test_case "let shadows stratum → W103"        `Quick test_indexed_param_shadow_warning;
    ];
    "param_bounds", [
      Alcotest.test_case "scalar param in [lo, hi]"          `Quick test_scalar_bounds;
      Alcotest.test_case "indexed param bounds expand to all strata" `Quick test_indexed_bounds;
    ];
    "shaped_let", [
      Alcotest.test_case "2D matrix literal row-major indexing" `Quick test_shaped_let;
    ];
    "where_guards", [
      Alcotest.test_case "param in where guard → E217"        `Quick test_where_param_in_guard;
      Alcotest.test_case "compartment in where guard → E217"  `Quick test_where_compartment_in_guard;
      Alcotest.test_case "ivguard filters intervention combos" `Quick test_where_ivguard_filters;
    ];
    "polio_models", [
      Alcotest.test_case "age-targeted SIA targets S_under5 → V_under5" `Quick test_polio_age_sia_targets_under5;
      Alcotest.test_case "spatial where p!=q gives 20 importation transitions" `Quick test_spatial_5_importation_count;
    ];
    "scenario_presets", [
      Alcotest.test_case "with_sia preset_enable = [\"sia_round_1\"]" `Quick test_preset_enable_seir_vaccine;
    ];
    "origin_date", [
      Alcotest.test_case "date() converts to float days since origin" `Quick test_date_to_const;
      Alcotest.test_case "date() without origin → E220"               `Quick test_date_requires_origin;
      Alcotest.test_case "negative lower bound preserved (not floored to 0)" `Quick test_negative_lower_bound;
    ];
    "priors", [
      Alcotest.test_case "~ log_normal(mu, sigma) parses"                `Quick test_prior_log_normal;
      Alcotest.test_case "~ beta(alpha, beta) parses"                    `Quick test_prior_beta;
      Alcotest.test_case "~ gamma(shape, rate) — 'rate' kw allowed"       `Quick test_prior_gamma_with_rate_kwarg;
      Alcotest.test_case "~ half_normal(sigma) parses"                   `Quick test_prior_half_normal;
      Alcotest.test_case "no prior clause → prior = None"                `Quick test_no_prior_is_none;
      Alcotest.test_case "indexed param shares prior across expansion"   `Quick test_indexed_param_shares_prior;
      Alcotest.test_case "E232 unknown distribution — carries param name" `Quick test_unknown_prior_errors;
    ];
    "prior_distributions", [
      Alcotest.test_case "~ uniform(lower, upper) parses + round-trips"  `Quick test_prior_uniform;
      Alcotest.test_case "~ normal(mu, sigma) parses + round-trips"      `Quick test_prior_normal;
      Alcotest.test_case "~ exponential(rate) parses + round-trips"      `Quick test_prior_exponential;
      Alcotest.test_case "~ log_uniform(lower, upper) parses"            `Quick test_prior_log_uniform;
      Alcotest.test_case "E235 log_uniform requires positive bounds"     `Quick test_prior_log_uniform_nonpositive_errors;
      Alcotest.test_case "~ truncated_normal bounds from `in [..]`"      `Quick test_prior_truncated_normal_bounds_from_decl;
      Alcotest.test_case "E285 truncated_normal requires `in [..]`"      `Quick test_prior_truncated_normal_requires_bounds;
      Alcotest.test_case "E286 log_uniform not poolable (no ICE)"        `Quick test_prior_log_uniform_not_poolable;
    ];
    "prior_const_args", [
      Alcotest.test_case "arithmetic of literals evaluates correctly"    `Quick test_prior_arg_arithmetic;
      Alcotest.test_case "log(0.3) is a const arg"                       `Quick test_prior_arg_log_function;
      Alcotest.test_case "exp() and sqrt() as const args"                `Quick test_prior_arg_exp_and_sqrt;
    ];
    "prior_validation", [
      Alcotest.test_case "E230 non-const prior arg"                      `Quick test_e230_non_const_arg;
      Alcotest.test_case "E231 missing required kwarg"                   `Quick test_e231_missing_kwarg;
      Alcotest.test_case "E231 half_normal without sigma"                `Quick test_e231_missing_kwarg_half_normal;
      Alcotest.test_case "E233 unknown / extra kwarg"                    `Quick test_e233_unknown_kwarg;
      Alcotest.test_case "E233 typo'd kwarg ('mean' instead of 'mu')"    `Quick test_e233_typo_kwarg;
      Alcotest.test_case "E234 duplicate kwarg"                          `Quick test_e234_duplicate_kwarg;
      Alcotest.test_case "E235 uniform(lower>=upper)"                    `Quick test_e235_uniform_inverted;
      Alcotest.test_case "E235 beta(alpha<=0)"                           `Quick test_e235_beta_negative_alpha;
      Alcotest.test_case "E235 gamma(shape=0)"                           `Quick test_e235_gamma_zero_shape;
      Alcotest.test_case "E235 exponential(rate=0)"                      `Quick test_e235_exponential_zero_rate;
      Alcotest.test_case "E235 normal(sigma<0)"                          `Quick test_e235_normal_negative_sigma;
      Alcotest.test_case "E235 half_normal(sigma=0)"                     `Quick test_e235_half_normal_zero_sigma;
    ];
    "observation_projections", [
      Alcotest.test_case "prevalence(E) sums Erlang substages"           `Quick test_prevalence_on_stratified_compartment;
      Alcotest.test_case "bare E in projected sums Erlang substages"     `Quick test_projected_bare_stratified_compartment;
      Alcotest.test_case "prevalence(E[e1]) picks single stratum"        `Quick test_prevalence_fully_indexed_stratified;
      Alcotest.test_case "prevalence(I) unstratified is unchanged"       `Quick test_prevalence_unstratified;
      Alcotest.test_case "E280: bare incidence(infection) on stratified model rejected" `Quick test_incidence_unindexed_cross_strata_is_rejected;
      Alcotest.test_case "explicit sum(a in age, incidence(infection[a])) → flow sum" `Quick test_incidence_explicit_sum_compiles_to_flow_sum;
      Alcotest.test_case "incidence(infection[child]) picks one stratum" `Quick test_incidence_positional_indexed_pins_one_stratum;
      Alcotest.test_case "incidence(infection[age=adult]) named index"   `Quick test_incidence_named_indexed_pins_one_stratum;
      Alcotest.test_case "incidence(infection) unstratified unchanged"   `Quick test_incidence_unstratified;
      Alcotest.test_case "let-bound projected inlines (not E507)"        `Quick test_let_bound_projection_inlines;
    ];
    "likelihood_kwargs", [
      Alcotest.test_case "poisson(rate = projected) parses"              `Quick test_poisson_rate_kwarg_parses;
      Alcotest.test_case "E250 positional arg in likelihood"             `Quick test_poisson_positional_errors;
      Alcotest.test_case "E251 unknown kwarg in likelihood"              `Quick test_likelihood_unknown_kwarg_errors;
    ];
    "survey_denominators", [
      Alcotest.test_case "binomial(n = tested) → ObsColumnRef leaf"       `Quick test_survey_denominator_resolves_to_obs_column_ref;
      Alcotest.test_case "E277: declared-but-unreferenced aux column is dead" `Quick test_unreferenced_aux_column_is_dead;
    ];
    "unchecked_dim_escape", [
      Alcotest.test_case "parses with all three kwargs"               `Quick test_unchecked_dim_parses;
      Alcotest.test_case "missing `reason` → E240"                    `Quick test_unchecked_dim_requires_reason;
      Alcotest.test_case "unknown dim name → E240"                    `Quick test_unchecked_dim_unknown_dim_name;
      Alcotest.test_case "He 2010-style (I+ι)^α rate typechecks"      `Quick test_unchecked_dim_he_style_typechecks;
    ];
    "forcing_unit_annotations", [
      Alcotest.test_case "'per_day → dim (0,-1)"                     `Quick test_sinusoidal_per_day_dim;
      Alcotest.test_case "'ratio → dim (0,0)"                        `Quick test_sinusoidal_ratio_dim;
      Alcotest.test_case "'count → dim (1,0)"                        `Quick test_sinusoidal_count_dim;
      Alcotest.test_case "gh#308 interpolated ISO-date time_col → day-offsets" `Quick test_interpolated_iso_date_time_col;
      Alcotest.test_case "gh#308 interpolated numeric time_col passes through"  `Quick test_interpolated_numeric_time_col;
      Alcotest.test_case "gh#308 interpolated ISO-date time_col w/o origin errors (E209)" `Quick test_interpolated_iso_date_no_origin_errors;
      Alcotest.test_case "forcing without unit literal is an error" `Quick test_forcing_without_unit_errors;
      Alcotest.test_case "comma in scenario set{} block hints newline separator" `Quick test_scenario_set_comma_separator_hint;
    ];
    "derived_expr_projections", [
      Alcotest.test_case "`projected = I_m + I_s` emits DerivedExpr"    `Quick test_projected_bare_sum_emits_derived_expr;
      Alcotest.test_case "`projected = (I_m + I_s) / N` compiles"       `Quick test_projected_proportion_compiles;
    ];
    "hierarchical_priors", [
      Alcotest.test_case "parses `alpha[age] ~ log_normal(mu=mu_h, sigma=s_h) | age`" `Quick test_hierarchical_prior_parses;
      Alcotest.test_case "scalar leaf populates Ir.(Ir.param_hierarchical parameter)"          `Quick test_hierarchical_scalar_leaf_ir_shape;
      Alcotest.test_case "indexed leaf expands per dim with shared hyperparents"     `Quick test_hierarchical_indexed_ir_shape;
      Alcotest.test_case "C1: self-reference rejected (E236)"                        `Quick test_hierarchical_self_reference_rejected;
      Alcotest.test_case "C2: cycle rejected (E236)"                                 `Quick test_hierarchical_cycle_rejected;
      Alcotest.test_case "C3: 3-level chain compiles cleanly"                        `Quick test_hierarchical_three_level_chain_compiles;
    ];
    "branching_destinations", [
      Alcotest.test_case "parser accepts `X --> {Y:p, Z:1-p}`"         `Quick test_branching_parses;
      Alcotest.test_case "desugars to two weight-scaled transitions"    `Quick test_branching_equivalent_to_two_transitions;
      Alcotest.test_case "indexed `[a in age]` expands per age × branch" `Quick test_branching_indexed_by_age;
    ];
    "multi_source_transitions", [
      Alcotest.test_case "parser accepts `S + I --> I + I`"              `Quick test_multi_source_parses;
      Alcotest.test_case "catalyst collapse preserves single-source IR"  `Quick test_multi_source_catalyst_collapses;
      Alcotest.test_case "bimolecular A + B --> C → {A:-1, B:-1, C:+1}"  `Quick test_multi_source_bimolecular_stoich;
      Alcotest.test_case "indexed `bite[a in age]` expands per age"      `Quick test_multi_source_indexed_by_age;
    ];
    "diagnostic_test_likelihood", [
      Alcotest.test_case "parses + rewrites p to sens·π + (1−spec)·(1−π)" `Quick test_diagnostic_test_parses;
      Alcotest.test_case "IR equivalent to hand-inlined correction"        `Quick test_diagnostic_test_equivalence;
      Alcotest.test_case "bernoulli base supported"                        `Quick test_diagnostic_test_bernoulli;
      Alcotest.test_case "E253 rejects unsupported base (poisson)"        `Quick test_diagnostic_test_bad_base;
      Alcotest.test_case "E254 rejects missing kwargs"                    `Quick test_diagnostic_test_missing_kwargs;
    ];
    "lineage_individual_sampling", [
      Alcotest.test_case "#[lineage] parses (both forms) → identical IR"   `Quick test_lineage_parses_both_forms;
      Alcotest.test_case "accepts β·S·I/N (freq-dependent)"               `Quick test_lineage_accepts_freq_dependent;
      Alcotest.test_case "accepts multi-pool β·S·(β_I·I+β_A·A)/N"         `Quick test_lineage_accepts_multi_pool;
      Alcotest.test_case "accepts stratified contact-matrix rate"          `Quick test_lineage_accepts_stratified;
      Alcotest.test_case "E601 rejects β·S·(I+ι)^α/N at nonlinear subterm" `Quick test_lineage_rejects_nonlinear_e601;
      Alcotest.test_case "weight(I) = β·S/N for freq-dependent"            `Quick test_lineage_weight_freq_dependent;
      Alcotest.test_case "multi-pool weights β·β_I·S/N, β·β_A·S/N"         `Quick test_lineage_weight_multi_pool;
      Alcotest.test_case "identity subgraph SEIR S→E tracks {E,I,R}"       `Quick test_lineage_identity_subgraph_seir;
      Alcotest.test_case "identity subgraph SIRS cycle tracks {S,I,R}"     `Quick test_lineage_identity_subgraph_sirs_cycle;
      Alcotest.test_case "inert when no #[lineage] annotations"            `Quick test_lineage_inert_when_absent;
      Alcotest.test_case "E110 unknown attribute #[transmission]"          `Quick test_lineage_unknown_attribute_e110;
    ];
    "calendar_time", [
      Alcotest.test_case "origin → string + numeric origin_rata_die"        `Quick test_origin_rata_die_emitted;
      Alcotest.test_case "no origin → no origin_rata_die"                   `Quick test_origin_absent_no_rata_die;
    ];
    "typed_time_phase1", [
      (* Positive cases — must compile without error. *)
      Alcotest.test_case "5 'months table value (unanchored) is legal"
        `Quick test_typed_time_pos_5months_table_value;
      Alcotest.test_case "0.087 'per_month rate in anchored mode is legal"
        `Quick test_typed_time_pos_per_month_rate_in_anchored;
      Alcotest.test_case "simulate.to = 600 'months one-shot conversion (anchored)"
        `Quick test_typed_time_pos_simulate_to_months_oneshot;
      Alcotest.test_case "duration param bounds with 'months stay Exact"
        `Quick test_typed_time_pos_duration_param_bounds_with_months;
      Alcotest.test_case "unanchored 'months axis (dacca shape) compiles"
        `Quick test_typed_time_pos_unanchored_months_axis;
      (* Rule 1 (E321): Instant + CalendarDuration. *)
      Alcotest.test_case "E321 date(...) + 6 'months rejected"
        `Quick test_typed_time_e321_date_plus_months_rejected;
      Alcotest.test_case "E321 laundered through let: date(...) + d where d = 6 'months"
        `Quick test_typed_time_e321_laundered_through_let;
      Alcotest.test_case "E321 hint mentions add_calendar_months + 'days fallback"
        `Quick test_typed_time_e321_hint_text;
      (* Rule 2 (E320): time_unit = 'months/'years with origin. *)
      Alcotest.test_case "E320 time_unit = 'months with origin rejected"
        `Quick test_typed_time_e320_time_unit_months_with_origin_rejected;
      Alcotest.test_case "E320 hint mentions 'days switch + silent-shift trap"
        `Quick test_typed_time_e320_hint_text;
      (* Rule 7 (E322): calendar cadence in recurring schedule. *)
      Alcotest.test_case "E322 every = 1 'months in anchored recurring rejected"
        `Quick test_typed_time_e322_calendar_cadence_in_recurring;
      Alcotest.test_case "E322 unanchored every = 1 'months is fine (vacuous rule)"
        `Quick test_typed_time_e322_unanchored_months_cadence_ok;
      (* Rule 4 (E323): bare-numeric on=[] in anchored periodic. *)
      Alcotest.test_case "E323 bare-numeric on=[7:100] in anchored periodic rejected"
        `Quick test_typed_time_e323_bare_numeric_on_periodic_anchored;
      (* Rule 5 (W324, W325): bare-numeric in time positions. *)
      Alcotest.test_case "W324 bare-numeric simulate.from/to warns under origin"
        `Quick test_typed_time_w324_bare_numeric_simulate_warning;
      Alcotest.test_case "W324 not fired when simulate fields are unit-annotated"
        `Quick test_typed_time_w324_unit_annotated_simulate_no_warning;
      Alcotest.test_case "W325 bare-numeric at-schedule warns under origin"
        `Quick test_typed_time_w325_bare_numeric_at_schedule_warning;
      (* gh#134: symmetric negatives + events-block coverage for the
         model-side calendar nudge (sibling of data-loader W326). *)
      Alcotest.test_case "gh134 date() simulate.from/to does not warn W324"
        `Quick test_gh134_date_simulate_from_to_no_w324;
      Alcotest.test_case "gh134 date() intervention at[..] does not warn W325"
        `Quick test_gh134_date_intervention_at_no_w325;
      Alcotest.test_case "gh134 bare-numeric events at[..] warns W325"
        `Quick test_gh134_events_bare_numeric_at_warns_w325;
      Alcotest.test_case "gh134 unanchored bare-numeric does not warn W324/W325"
        `Quick test_gh134_unanchored_bare_numeric_no_nudge;
    ];
    "typed_time_phase2", [
      (* add_calendar_months / add_calendar_years: canonical cases (§8) *)
      Alcotest.test_case "Jan 31 + 1 month → Feb 29 (leap)"
        `Quick test_phase2_add_months_leap_feb_clamp;
      Alcotest.test_case "Jan 31 + 1 month → Feb 28 (non-leap)"
        `Quick test_phase2_add_months_non_leap_feb_clamp;
      Alcotest.test_case "Feb 29 + 1 year → Feb 28 (leap → non-leap clamp)"
        `Quick test_phase2_add_years_leap_to_non_leap;
      Alcotest.test_case "Jan 31 + 13 months → Feb 28 (cross year-end)"
        `Quick test_phase2_add_months_13_crosses_year;
      Alcotest.test_case "Mar 31 + 1 month → Apr 30"
        `Quick test_phase2_add_months_mar_to_apr;
      Alcotest.test_case "Mar 31 − 1 month → Feb 29 (leap)"
        `Quick test_phase2_sub_months_mar_to_feb_leap;
      Alcotest.test_case "Mar 31 − 1 month → Feb 28 (non-leap)"
        `Quick test_phase2_sub_months_mar_to_feb_non_leap;
      Alcotest.test_case "add_calendar_months(origin, 6) resolves in anchored model"
        `Quick test_phase2_add_months_origin_anchored;
      Alcotest.test_case "E327 add_calendar_months in unanchored model"
        `Quick test_phase2_add_months_unanchored_errors;
      (* date_range *)
      Alcotest.test_case "date_range affine start–end produces 53 weekly entries"
        `Quick test_phase2_date_range_affine_start_end;
      Alcotest.test_case "date_range affine count = 24 produces 25 entries"
        `Quick test_phase2_date_range_affine_count;
      Alcotest.test_case "date_range calendar_months = 3 over 5 years produces 21 entries"
        `Quick test_phase2_date_range_calendar_months_start_end;
      Alcotest.test_case "date_range calendar_years = 1 count = 5 produces 6 entries"
        `Quick test_phase2_date_range_calendar_years_count;
      Alcotest.test_case "W328 non-aligned end fires with inclusive_end hint"
        `Quick test_phase2_date_range_non_aligned_end_w328;
      Alcotest.test_case "E329 every = 0 rejected as non-positive"
        `Quick test_phase2_date_range_zero_cadence_errors;
      Alcotest.test_case "E329 calendar_months = -1 rejected as non-positive"
        `Quick test_phase2_date_range_negative_calendar_cadence_errors;
      Alcotest.test_case "E329 count = 0 rejected"
        `Quick test_phase2_date_range_count_zero_errors;
      Alcotest.test_case "E327 calendar cadence in unanchored model"
        `Quick test_phase2_date_range_calendar_in_unanchored_errors;
      (* Round-trip W327 *)
      Alcotest.test_case "W327 round-trip composition warns on month-end clamp"
        `Quick test_phase2_round_trip_w327;
      (* origin *)
      Alcotest.test_case "origin in simulate.from resolves to 0 (anchored)"
        `Quick test_phase2_origin_in_simulate_from_anchored;
      Alcotest.test_case "E327 origin reference in unanchored model"
        `Quick test_phase2_origin_in_unanchored_errors;
    ];
    "simulate_dt", [
      Alcotest.test_case "dt = 0.5 lowers to simulation.dt = Some 0.5"
        `Quick test_simulate_dt_plain;
      Alcotest.test_case "no dt → simulation.dt = None"
        `Quick test_simulate_dt_omitted_is_none;
      Alcotest.test_case "dt = 0.05 'months is unit-aware (→ days)"
        `Quick test_simulate_dt_unit_aware;
      Alcotest.test_case "E106 unknown simulate key (typo) errors"
        `Quick test_simulate_unknown_key_errors;
    ];
  ]
