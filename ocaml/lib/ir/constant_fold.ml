(* Constant-fold pass over a fully-expanded model.

   Resolves a constant-indexed lookup into an *inline literal* table to its
   scalar, applies a few algebraic identities (0*x->0, 1*x->x, 0+x->x, and
   folds Const op Const for total ops), and drops Const 0.0 terms from a
   Reduce. The point: when a coupling matrix W is sparse, the dense P-term FOI
   Reduce the expander emits — each term `W[l,q] * (...)` with `W[l,q]` a
   TableLookup over constant literal indices — collapses to the k nonzero
   terms. O(P^2) FOI -> O(P*k), with no new IR node and no sparse runtime
   evaluator (the runtime already evals Reduce, just with fewer terms).

   Byte-identical, by construction + gate:
   - resolving TableLookup(name,[Const i]) to its literal cell is exact;
   - dropping a `Const 0.0` term from a left-folded Reduce is the additive
     identity (`acc +. 0.0 = acc` for finite acc);
   - `0*x -> 0` is exact when x is finite. In the *guarded* FOI term
     `W[l,q] * (if N>0 then I/N else 0)` the right factor is finite and
     bounded, so a zero-W term folds to exactly 0 in one step (the division
     lives inside the guard, never reached when the term is zeroed).
   Div/Pow/Mod of constants are deliberately NOT folded — their degenerate
   handling lives in the evaluator and folding here could diverge from it.
   The empirical proof is the A/B gate
   `rust/crates/sim/tests/gate_constant_fold_ab.rs`: on a sparse-coupling model
   it compiles the IR both ways and asserts the simulated trajectory is
   byte-identical under every backend (with a non-vacuity guard that the fold
   actually collapsed the FOI Reduce). The OCaml half
   (`test_compiler.ml`, "constant_fold") pins the term-count collapse.

   Runs after expansion + autodiff (so rate_grad folds too), before serialize.
   On by default; set CAMDL_NO_CONSTANT_FOLD to emit the unfolded IR. *)

open Ir

(* table name -> flat (row-major) values, for Inline tables that are all Const. *)
let inline_table_values (tables : table list) : (string, float array) Hashtbl.t =
  let h = Hashtbl.create 16 in
  List.iter
    (fun (t : table) ->
      match t.source with
      | Inline vals ->
          let arr = Array.make (List.length vals) 0.0 in
          let all_const = ref true in
          List.iteri
            (fun i e -> match e with Const f -> arr.(i) <- f | _ -> all_const := false)
            vals;
          if !all_const then Hashtbl.replace h t.name arr
      | External _ -> ())
    tables;
  h

let is_zero = function Const c -> c = 0.0 | _ -> false

(* fold a binary op of two constants — total ops only; Div/Pow/Mod and
   comparisons are left for the evaluator to handle. *)
let fold_bin_consts op a b : expr option =
  match op with
  | Add -> Some (Const (a +. b))
  | Sub -> Some (Const (a -. b))
  | Mul -> Some (Const (a *. b))
  | Min -> Some (Const (Float.min a b))
  | Max -> Some (Const (Float.max a b))
  | _ -> None

let rec fold tbls (e : expr) : expr =
  match e with
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | BindingRef _
  | Projected | ObsColumnRef _ ->
      e
  (* LICM runs after constant_fold, so a PerEvalRef cannot appear here. *)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  | UnOp { op; arg } -> UnOp { op; arg = fold tbls arg }
  | Cond { pred; then_; else_ } ->
      Cond { pred = fold tbls pred; then_ = fold tbls then_; else_ = fold tbls else_ }
  | UncheckedDim r -> UncheckedDim { r with inner = fold tbls r.inner }
  | TableLookup (name, [ idx ]) -> (
      match (fold tbls idx, Hashtbl.find_opt tbls name) with
      | Const fi, Some arr ->
          let i = int_of_float (Float.floor fi) in
          if i >= 0 && i < Array.length arr then Const arr.(i)
            (* OOB index into a literal table: leave it; the runtime's
               out_of_bounds policy (Clamp/Wrap/Error) decides. *)
          else TableLookup (name, [ Const fi ])
      | idx', _ -> TableLookup (name, [ idx' ]))
  | TableLookup (name, idxs) -> TableLookup (name, List.map (fold tbls) idxs)
  | BinOp { op; left; right } -> (
      let l = fold tbls left and r = fold tbls right in
      match (op, l, r) with
      | _, Const a, Const b -> (
          match fold_bin_consts op a b with Some c -> c | None -> BinOp { op; left = l; right = r })
      | Mul, c, _ when is_zero c -> Const 0.0 (* finite-x precondition; gated *)
      | Mul, _, c when is_zero c -> Const 0.0
      | Mul, Const 1.0, x | Mul, x, Const 1.0 -> x
      | Add, c, x when is_zero c -> x
      | Add, x, c when is_zero c -> x
      | Sub, x, c when is_zero c -> x
      | _ -> BinOp { op; left = l; right = r })
  | Reduce terms ->
      let kept = List.filter_map (fun t -> let t' = fold tbls t in if is_zero t' then None else Some t') terms in
      (match kept with [] -> Const 0.0 | [ t ] -> t | _ -> Reduce kept)

(* Fold the expr-bearing fields where a sparse coupling matrix actually
   appears: transition rates + their gradients, model-level bindings, and ODE
   derivatives. Folding a subset is sound (each folded expr keeps its value);
   it just leaves any W-free exprs untouched. *)
let fold_model (m : model) : model =
  let tbls = inline_table_values m.tables in
  if Hashtbl.length tbls = 0 then m
  else
    let fe = fold tbls in
    {
      m with
      transitions =
        List.map
          (fun (t : transition) ->
            { t with rate = fe t.rate; rate_grad = List.map (fun (p, g) -> (p, fe g)) t.rate_grad })
          m.transitions;
      bindings = List.map (fun (b : binding) -> { b with bexpr = fe b.bexpr }) m.bindings;
      ode_equations =
        List.map (fun (eq : ode_equation) -> { eq with derivative = fe eq.derivative }) m.ode_equations;
    }
