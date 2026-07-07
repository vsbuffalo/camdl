(* gh#272 Loop-invariant code motion (LICM) over a fully-expanded,
   already-differentiated model.

   Extracts maximal param/table-only ("invariant") subexpressions out of the
   DYNAMICS surfaces — transition `rate`, transition `rate_grad`, and
   `ode_equations` derivatives — into `model.per_eval_bindings`, replacing each
   with a `PerEvalRef`. The runtime evaluates a per-eval binding once per
   theta-stable scope (a later increment adds the cache; today it is on-demand),
   instead of recomputing the subtree on every integration step. The in-model
   gravity kernel `N0[q]*exp(-gamma*log(dratio))` and its normalization are the
   motivating case — loop-invariant within a trajectory but recomputed every
   step (gh#272).

   Scope is a hard contract: ONLY the three dynamics surfaces, which the runtime
   evaluates with `eval_resolved`. NOT bindings, overdispersion, or observation
   expressions — those are the secondary-gradient surfaces (`eval_resolved_deriv`
   / `collect_param_refs`), kept PerEvalRef-free so those consumers never see the
   node (docs/dev/proposals/2026-06-20-loop-invariant-code-motion.md).

   Value-preserving: substituting a `PerEvalRef` for a structurally-identical
   subtree (CSE-deduped by a BITWISE expr key) does not change evaluation order,
   so trajectories and gradients are byte-identical. Runs AFTER autodiff +
   constant_fold (it hoists already-folded subtrees; constant_fold never sees a
   `PerEvalRef`). On by default; `CAMDL_NO_LICM` forces it off (the inlined
   variant), mirroring constant_fold / `CAMDL_NO_CONSTANT_FOLD`. *)

open Ir

(* An invariant subtree depends only on params/tables/constants — never on
   compartment state, time, dt, a forcing (`TimeFunc`), or a (state) `BindingRef`.
   These are exactly the nodes constant within a single trajectory once theta is
   bound. NOTE this is the variant/invariant predicate the proposal calls for; it
   covers `Dt` explicitly (a deliberate divergence from `expr_is_time_dependent`,
   which omits it).

   `tbls` maps each table name to its definition so a `TableLookup` is judged on
   its CELL BODIES, not just its index (gh#284): an inline cell that references
   state (a future contact table scaling with prevalence) makes the whole lookup
   variant even with a constant index. An external table is file-loaded numeric
   data — constant within a trajectory — so it is invariant. An unknown table is
   treated conservatively as variant (some other pass diagnoses it). Today every
   inline cell is constant-valued — its leaves are only `Const`/`Param`, possibly
   under constant arithmetic (the expander emits no state-dependent cells, and the
   Rust loader evaluates each cell to a number and rejects any state reference) —
   so this is value-preserving: no currently-emitted IR changes which subtrees
   hoist.

   The `visited` set guards the inter-table cell recursion: a table whose inline
   cell looks up a second table (a future possibility, not emittable today) could
   otherwise cycle. A table already on the path is conservatively variant. *)
let is_invariant (tbls : (string, table) Hashtbl.t) (e : expr) : bool =
  let rec go visited (e : expr) : bool =
    match e with
    | Const _ | Param _ -> true
    | PerEvalRef _ -> true   (* its body is invariant; never present in LICM input *)
    | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | BindingRef _
    | Projected | ObsColumnRef _ -> false
    | TableLookup (name, idxs) ->
      List.for_all (go visited) idxs && cells_invariant visited name
    | BinOp { left; right; _ } -> go visited left && go visited right
    | UnOp { arg; _ } -> go visited arg
    | Cond { pred; then_; else_ } -> go visited pred && go visited then_ && go visited else_
    | Reduce ts -> List.for_all (go visited) ts
    | UncheckedDim u -> go visited u.inner
  (* A table's cells are invariant iff every inline cell body is (external tables
     carry only file-loaded numbers). Unknown table → conservatively variant; a
     table already being evaluated (cycle) → conservatively variant. *)
  and cells_invariant visited (name : string) : bool =
    if List.mem name visited then false
    else
      match Hashtbl.find_opt tbls name with
      | Some { source = Inline cells; _ } -> List.for_all (go (name :: visited)) cells
      | Some { source = External _; _ } -> true
      | None -> false
  in
  go [] e

(* Worth hoisting iff the subtree carries a genuinely expensive op — a
   transcendental, a `Pow`, or an n-ary `Reduce`. A bare `Param`/`Const`/
   `TableLookup` or cheap arithmetic (e.g. `R0 * gamma`) is left inline: hoisting
   it would only grow the IR for a negligible per-step saving. *)
let rec contains_expensive (e : expr) : bool =
  match e with
  | UnOp { op = (Exp | Log | Sqrt | Sin | Cos | Tanh); _ } -> true
  | BinOp { op = Pow; _ } -> true
  | Reduce _ -> true
  | BinOp { left; right; _ } -> contains_expensive left || contains_expensive right
  | UnOp { arg; _ } -> contains_expensive arg
  | Cond { pred; then_; else_ } ->
    contains_expensive pred || contains_expensive then_ || contains_expensive else_
  | TableLookup (_, idxs) -> List.exists contains_expensive idxs
  | UncheckedDim u -> contains_expensive u.inner
  | Const _ | Param _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _
  | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ -> false

let binop_tag = function
  | Add -> 0 | Sub -> 1 | Mul -> 2 | Div -> 3 | Pow -> 4 | Mod -> 5
  | Min -> 6 | Max -> 7 | Eq -> 8 | Neq -> 9 | Lt -> 10 | Gt -> 11 | Le -> 12 | Ge -> 13

let unop_tag = function
  | Neg -> 0 | Exp -> 1 | Log -> 2 | Sqrt -> 3 | Abs -> 4 | Floor -> 5
  | Ceil -> 6 | Sin -> 7 | Cos -> 8 | Tanh -> 9

(* Bitwise-faithful canonical key for CSE dedup. Floats are encoded by their
   IEEE-754 bits so `-0.0` and `0.0` (and distinct NaNs) never alias — the
   polymorphic `Hashtbl.hash` / structural `=` that `inspect.expr_hash` uses would
   conflate them and is the WRONG seam here (the `-0.0` Reduce seed is observable).
   Two subtrees share a per-eval binding iff their keys are equal. *)
let rec canon (e : expr) : string =
  match e with
  | Const f -> Printf.sprintf "C%Ld" (Int64.bits_of_float f)
  | Param p -> "P" ^ p ^ ";"
  | Pop p -> "p" ^ p ^ ";"
  | PopSum ps -> "S[" ^ String.concat ";" ps ^ "]"
  | Time -> "t" | Dt -> "d"
  | TimeFunc n -> "F" ^ n ^ ";"
  | Projected -> "@" | ObsColumnRef c -> "O" ^ c ^ ";"
  | BindingRef n -> "Bref" ^ n ^ ";"
  | PerEvalRef n -> "Pref" ^ n ^ ";"
  | TableLookup (t, idxs) -> "T" ^ t ^ "(" ^ String.concat "," (List.map canon idxs) ^ ")"
  | BinOp { op; left; right } -> Printf.sprintf "B%d(%s,%s)" (binop_tag op) (canon left) (canon right)
  | UnOp { op; arg } -> Printf.sprintf "U%d(%s)" (unop_tag op) (canon arg)
  | Cond { pred; then_; else_ } -> Printf.sprintf "?(%s,%s,%s)" (canon pred) (canon then_) (canon else_)
  | Reduce ts -> "R(" ^ String.concat "," (List.map canon ts) ^ ")"
  | UncheckedDim u -> "D(" ^ canon u.inner ^ ")"

type ctx = {
  mutable counter      : int;
  table                : (string, string) Hashtbl.t;   (* canon key -> per-eval binding name *)
  mutable rev_bindings : (string * expr) list;          (* accumulated, newest first *)
  model_tables         : (string, table) Hashtbl.t;     (* table name -> def, for is_invariant *)
}

(* Per-eval binding names use a reserved `__licm_` prefix and a monotonic
   counter, so they are unique by construction (and live in a namespace —
   `per_eval_bindings` — disjoint from regular `bindings`, so a user binding of
   the same name cannot collide). *)
let fresh ctx =
  let n = ctx.counter in
  ctx.counter <- n + 1;
  "__licm_" ^ string_of_int n

(* Hoist `e` (a maximal invariant subtree) into `per_eval_bindings` and return a
   `PerEvalRef` to it. CSE: an already-hoisted, bitwise-identical subtree reuses
   its name. *)
let intern ctx (e : expr) : expr =
  let key = canon e in
  match Hashtbl.find_opt ctx.table key with
  | Some name -> PerEvalRef name
  | None ->
    let name = fresh ctx in
    Hashtbl.add ctx.table key name;
    ctx.rev_bindings <- (name, e) :: ctx.rev_bindings;
    PerEvalRef name

(* Rewrite one expression tree. The use site is a virtual variant parent: a
   maximal invariant subtree (the topmost invariant node on each path, including
   the whole tree if it is invariant) is hoisted iff it is worth hoisting; cheap
   invariant nodes are left inline. We only descend through VARIANT nodes — an
   invariant node is either hoisted whole or left whole, never split. *)
let rec rw ctx (e : expr) : expr =
  if is_invariant ctx.model_tables e then
    (if contains_expensive e then intern ctx e else e)
  else
    match e with
    | BinOp r -> BinOp { r with left = rw ctx r.left; right = rw ctx r.right }
    | UnOp r -> UnOp { r with arg = rw ctx r.arg }
    | Cond r -> Cond { pred = rw ctx r.pred; then_ = rw ctx r.then_; else_ = rw ctx r.else_ }
    | Reduce ts -> Reduce (List.map (rw ctx) ts)
    | UncheckedDim u -> UncheckedDim { u with inner = rw ctx u.inner }
    | TableLookup (t, idxs) -> TableLookup (t, List.map (rw ctx) idxs)
    (* Variant leaves (the invariant leaves Const/Param/PerEvalRef are handled in
       the `is_invariant` branch above and never reach here). *)
    | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | BindingRef _ | PerEvalRef _
    | Projected | ObsColumnRef _ | Const _ | Param _ -> e

(* Hoist invariant subexpressions out of the dynamics surfaces. Folding a subset
   is sound (each rewritten expr keeps its value). The per-eval bindings produced
   here have no inter-binding references (cheap invariant nodes stay inline and
   maximal subtrees are hoisted whole), so they are trivially topologically
   ordered — insertion order. *)
(* LICM a gradient entry: hoist invariant subtrees inside a real derivative
   expression; an [Unsupported] refusal carries no expression, so pass it
   through unchanged. (gh#342: rate_grad now carries classified [deriv_entry].) *)
let rw_deriv_entry ctx (de : deriv_entry) : deriv_entry =
  match de with DEGrad e -> DEGrad (rw ctx e) | DEUnsupported _ -> de

let licm_model (m : model) : model =
  let model_tables = Hashtbl.create (max 1 (List.length m.tables)) in
  List.iter (fun (t : table) -> Hashtbl.replace model_tables t.name t) m.tables;
  let ctx = { counter = 0; table = Hashtbl.create 256; rev_bindings = []; model_tables } in
  let transitions =
    List.map
      (fun (t : transition) ->
        { t with
          rate = rw ctx t.rate;
          rate_grad = List.map (fun (p, de) -> (p, rw_deriv_entry ctx de)) t.rate_grad;
          rate_state_grad = List.map (fun (c, de) -> (c, rw_deriv_entry ctx de)) t.rate_state_grad })
      m.transitions
  in
  let ode_equations =
    List.map (fun (eq : ode_equation) -> { eq with derivative = rw ctx eq.derivative }) m.ode_equations
  in
  let per_eval_bindings =
    List.rev_map (fun (name, e) -> { bname = name; bexpr = e }) ctx.rev_bindings
  in
  { m with transitions; ode_equations; per_eval_bindings }
