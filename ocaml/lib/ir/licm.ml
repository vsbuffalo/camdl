(** Loop-invariant code motion: subexpressions of the dynamics that depend
    only on parameters and tables are moved into named per-evaluation
    bindings, which the runtime computes once per parameter vector instead
    of at every step.

    A transition rate is evaluated at every step of a simulation, but part
    of it is often fixed for the whole trajectory: a distance kernel
    [exp(-gamma * log(d))], a normalising sum over patches. {!licm_model}
    finds, in each transition's [rate], [rate_grad] and [rate_state_grad]
    and in each ODE equation, the largest subtrees that read nothing but
    literals, parameters, model constants and table cells that are
    themselves of that kind. Each such subtree that contains an
    exponential, logarithm, square root, [sin], [cos], [tanh], power or
    [Reduce] becomes an {!Ir.PerEvalRef} to an entry it adds to the model's
    [per_eval_bindings]; a cheap one, such as [R0 * gamma], stays inline.
    Identical subtrees, compared bit for bit including the sign of zero,
    share one binding. For the rate

    {@camdl[
      infection : S --> I @ beta * exp(-k) * S * I / (S + I + R)
    ]}

    the compiler writes the rate over [__licm_0], and its derivatives over
    two more bindings:

    {@json[
      {"name":"__licm_0","expr":{"bin_op":{"op":"mul","left":{"param":"beta"},"right":{"un_op":{"op":"exp","arg":{"un_op":{"op":"neg","arg":{"param":"k"}}}}}}}},
      {"name":"__licm_1","expr":{"un_op":{"op":"exp","arg":{"un_op":{"op":"neg","arg":{"param":"k"}}}}}},
    ]}

    A binding's name is [__licm_] and a counter, in a list separate from
    the model's ordinary bindings, so it cannot collide with a name the
    author chose. A hoisted subtree is moved whole, so no binding refers to
    another. Overlapping subtrees, such as [beta * exp(-k)] and [exp(-k)]
    above, get separate bindings.

    At run time, each simulator and inference loop evaluates every
    per-evaluation binding once, for the parameter vector it is about to
    use, and every reference reads the stored value. Where a caller has not
    stored them, a reference evaluates the binding's body where it occurs,
    to the same value. The Rust loader refuses a binding that reads state
    or time, or refers to a later binding. The rewrite does not change any
    value: the same expression is evaluated, by the same evaluator, from
    the same parameters. The Rust test [gate_licm_ab] compares trajectories
    with and without the pass.

    The pass leaves bindings, observation expressions, overdispersion
    variances and initial conditions alone, so the runtime code that
    differentiates or scans those never meets a {!Ir.PerEvalRef}. It is the
    last pass before serialisation, after differentiation and constant
    folding, so no other pass meets one either. Setting [CAMDL_NO_LICM] in
    the environment turns it off. *)

open Ir

(* An invariant subtree depends only on params/tables/constants — never on
   compartment state, time, dt, a forcing (`TimeFunc`), or a (state) `BindingRef`.
   These are exactly the nodes constant within a single trajectory once theta is
   bound. It covers `Dt` explicitly, unlike `expr_is_time_dependent`, which
   omits it.

   `tbls` maps each table name to its definition so a `TableLookup` is judged on
   its cell bodies, not just its index: an inline cell that references state
   would make the whole lookup variant even with a constant index. An external
   table is file-loaded numeric data — constant within a trajectory — so it is
   invariant. An unknown table is treated conservatively as variant (some other
   pass diagnoses it). The expander emits only constant-valued inline cells
   (leaves `Const`/`Param` under constant arithmetic), and the Rust loader
   rejects a cell that references state.

   The `visited` set guards the inter-table cell recursion: a table whose inline
   cell looks up a second table could otherwise cycle. A table already on the
   path is conservatively variant. *)
let is_invariant (tbls : (string, table) Hashtbl.t) (e : expr) : bool =
  let rec go visited (e : expr) : bool =
    match e with
    | Const _ | UnitLiteral _ | Param _ | ConstRef _ -> true
    | PerEvalRef _ -> true   (* its body is invariant; never present in LICM input *)
    | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _ | BindingRef _
    | Projected | ObsColumnRef _ -> false
    (* Not invariant. Its value comes from the run, not from theta, so
       hoisting it into a theta-stable per-eval binding would cache a value that
       is not a function of theta. It also never reaches LICM: the resolver
       substitutes a Const before compile. *)
    | ObsAnchor _ -> false
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
  | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _
  | BindingRef _ | PerEvalRef _ | Projected | ObsColumnRef _ | ObsAnchor _ -> false

let binop_tag = function
  | Add -> 0 | Sub -> 1 | Mul -> 2 | Div -> 3 | Pow -> 4 | Mod -> 5
  | Min -> 6 | Max -> 7 | Eq -> 8 | Neq -> 9 | Lt -> 10 | Gt -> 11 | Le -> 12 | Ge -> 13

let unop_tag = function
  | Neg -> 0 | Exp -> 1 | Log -> 2 | Sqrt -> 3 | Abs -> 4 | Floor -> 5
  | Ceil -> 6 | Sin -> 7 | Cos -> 8 | Tanh -> 9

(* Bitwise-faithful canonical key for CSE dedup. Floats are encoded by their
   IEEE-754 bits so `-0.0` and `0.0` (and distinct NaNs) never alias — the
   polymorphic `Hashtbl.hash` / structural `=` that `inspect.expr_hash` uses would
   conflate them and is the wrong seam here (the `-0.0` Reduce seed is observable).
   Two subtrees share a per-eval binding iff their keys are equal. *)
let rec canon (e : expr) : string =
  match e with
  | Const f -> Printf.sprintf "C%Ld" (Int64.bits_of_float f)
  | Param p -> "P" ^ p ^ ";"
  | ConstRef c -> "K" ^ c ^ ";"
  | Pop p -> "p" ^ p ^ ";"
  | PopSum ps -> "S[" ^ String.concat ";" ps ^ "]"
  | Time -> "t" | Dt -> "d"
  | TimeFunc n -> "F" ^ n ^ ";"
  | ForcingBasis { forcing; index; space } ->
    Printf.sprintf "Fb%s#%d%s;" forcing index
      (match space with Space_spline -> "" | Space_whitened -> "w" | Space_linear -> "l")
  | Projected -> "@" | ObsColumnRef c -> "O" ^ c ^ ";"
  (* Anchor and offset both enter the key: two knots that differ only in offset
     are different subtrees and must never share a per-eval binding. *)
  | ObsAnchor a ->
    Printf.sprintf "A%s%Ld;"
      (match a.anchor with AnchorFirst -> "f" | AnchorLast -> "l")
      (Int64.bits_of_float a.offset)
  | BindingRef n -> "Bref" ^ n ^ ";"
  | PerEvalRef n -> "Pref" ^ n ^ ";"
  | TableLookup (t, idxs) -> "T" ^ t ^ "(" ^ String.concat "," (List.map canon idxs) ^ ")"
  | BinOp { op; left; right } -> Printf.sprintf "B%d(%s,%s)" (binop_tag op) (canon left) (canon right)
  | UnOp { op; arg } -> Printf.sprintf "U%d(%s)" (unop_tag op) (canon arg)
  | Cond { pred; then_; else_ } -> Printf.sprintf "?(%s,%s,%s)" (canon pred) (canon then_) (canon else_)
  | Reduce ts -> "R(" ^ String.concat "," (List.map canon ts) ^ ")"
  | UncheckedDim u -> "D(" ^ canon u.inner ^ ")"
  | UnitLiteral u -> Printf.sprintf "L%Ld,%d,%d;" (Int64.bits_of_float u.value) u.dim_p u.dim_t

(** The state of one run of {!licm_model}: the bindings hoisted so far, and
    the index that lets an identical subtree reuse one of them. *)
type ctx = {
  mutable counter      : int;  (** The number in the next binding's name, [__licm_N]. *)
  table                : (string, string) Hashtbl.t;   (** The canonical key of each hoisted subtree, mapped to its binding's name. *)
  mutable rev_bindings : (string * expr) list;          (** The hoisted bindings, newest first; the model lists them in the reverse order. *)
  model_tables         : (string, table) Hashtbl.t;     (** Each table's name, mapped to its definition, so a lookup can be judged by its cells. *)
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
   invariant nodes are left inline. We only descend through variant nodes — an
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
    | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _ | BindingRef _ | PerEvalRef _
    | Projected | ObsColumnRef _ | ObsAnchor _ | Const _ | UnitLiteral _ | Param _ | ConstRef _ -> e

(* Hoist invariant subexpressions out of the dynamics surfaces. Folding a subset
   is sound (each rewritten expr keeps its value). The per-eval bindings produced
   here have no inter-binding references (cheap invariant nodes stay inline and
   maximal subtrees are hoisted whole), so they are trivially topologically
   ordered — insertion order. *)
(* LICM a gradient entry: hoist invariant subtrees inside a real derivative
   expression; an [Unsupported] refusal carries no expression, so pass it
   through unchanged. *)
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
