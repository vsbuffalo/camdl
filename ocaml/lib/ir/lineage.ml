(** Linear-in-parents analysis for `#[lineage]` transitions.

    Individual-sampling layer, 2026-05-19 proposal
    (docs/dev/proposals/2026-05-19-individual-sampling-layer.md,
    §"Linear-in-parents requirement" and §"IR change").

    Two responsibilities:

    1. [classify_parents]: decide whether a `#[lineage]` rate is
       linear in every non-source compartment it references. The
       classifier buckets each non-source Pop reference as
       Source / Denominator-only / Linear-parent / Nonlinear-use.
       A Nonlinear-use is rejected (E601 at the call site).

    2. [parent_pool_weights]: for a rate that passes the linearity
       check, emit the per-pool weight sub-expression — the linear
       decomposition of the rate over parent pools. For `β·S·I/N`
       with parent I this is [("I", β·S/N)].

    The key correctness rule (proposal §"Precedence — normalizing
    denominators win"): a parent compartment may appear in both the
    numerator (as a linear parent) and a normalizing denominator.
    Division by a normalizer is a nonlinear function, so a naive
    classifier would bucket the denominator appearance as nonlinear
    and wrongly reject `β·S·I/N` — the most common transmission form.
    The exemption: denominator/normalizer appearances are classified
    Denominator-only and exempt from the Nonlinear-use bucket; the
    exemption is applied before the nonlinear-use check. Linearity is
    required only of the numerator dependence on parent counts; a
    normalizer is a frozen coefficient at the event instant, even when
    it references the parent compartment.

    Thus `β·S·I/N` compiles (I is a linear numerator parent; its N
    appearance is a frozen normalizer), while `β·S·(I+ι)^α/N` is
    rejected (the power is a genuine nonlinear use of I in the
    numerator). *)

open Ir

(* A non-source Pop reference, with the offending sub-expression
   recorded for the diagnostic when it is a nonlinear use. *)
type nonlinear_use = {
  comp:    string;       (* the compartment used nonlinearly *)
  context: expr;         (* the enclosing nonlinear sub-expression *)
}

type classification = {
  parents:    string list;        (* numerator linear-parent compartments, in first-seen order *)
  nonlinear:  nonlinear_use option;  (* Some _ ⇒ reject (E601) *)
}

(* Is this Pow a genuinely nonlinear power, i.e. exponent <> 1?
   `x^1` is linear in x; the dim-checker / normalizer can leave a
   literal `^ 1` and we should not reject it. *)
let pow_is_nonlinear (exponent : expr) : bool =
  match exponent with
  | Const 1.0 -> false
  | _         -> true

(** Walk a rate expression, classifying every non-source Pop reference.

    [sources] is the set of the transition's source compartment names.
    Tracks two descent flags:
      - [in_denom]:     under the right operand of some Div (a frozen
                        normalizer). Denominator appearances are exempt
                        from the nonlinear-use check — the exemption
                        wins (proposal precedence rule).
      - [in_nonlinear]: under a non-unit power, log/exp/sqrt/sin/cos/
                        tanh/abs, Cond, min, max, mod, or a comparison.

    A non-source Pop is:
      - Denominator-only if [in_denom] (exempt; not a parent),
      - Nonlinear-use   if [in_nonlinear] and not [in_denom] (reject),
      - Linear parent   otherwise. *)
let classify_parents ~(sources : string list) (rate : expr) : classification =
  let is_source c = List.mem c sources in
  let parents = ref [] in           (* reverse first-seen order *)
  let add_parent c =
    if not (is_source c) && not (List.mem c !parents) then
      parents := c :: !parents
  in
  let nonlinear = ref None in
  let flag_nonlinear comp context =
    if !nonlinear = None then nonlinear := Some { comp; context }
  in
  (* Record a non-source Pop reference given the current descent flags.
     [enclosing] is the smallest nonlinear sub-expression we are inside,
     used for the diagnostic. *)
  let record_pop ~in_denom ~in_nonlinear ~enclosing comp =
    if is_source comp then ()
    else if in_denom then ()                 (* Denominator-only: exempt, not a parent *)
    else if in_nonlinear then
      flag_nonlinear comp enclosing          (* Nonlinear-use: reject *)
    else
      add_parent comp                        (* Linear parent candidate *)
  in
  let rec walk ~in_denom ~in_nonlinear ~enclosing (e : expr) : unit =
    match e with
    | Pop c -> record_pop ~in_denom ~in_nonlinear ~enclosing c
    | PopSum cs ->
      List.iter (record_pop ~in_denom ~in_nonlinear ~enclosing) cs
    | Const _ | Param _ | Time | Dt | Projected | TimeFunc _ -> ()
    | TableLookup (_, args) ->
      (* Index expressions are evaluated as frozen lookups — a Pop in a
         table index is not a parent pool. Treat as nonlinear context so
         a stray Pop there is rejected rather than silently a parent. *)
      List.iter (walk ~in_denom ~in_nonlinear:true ~enclosing:e) args
    | UncheckedDim u ->
      (* The wrapper is a type-level assertion; descend transparently. *)
      walk ~in_denom ~in_nonlinear ~enclosing u.inner
    | BinOp { op = Div; left; right } ->
      walk ~in_denom ~in_nonlinear ~enclosing left;
      (* Right operand is a normalizer/denominator: frozen, exempt. *)
      walk ~in_denom:true ~in_nonlinear ~enclosing right
    | BinOp { op = (Mul | Add | Sub); left; right } ->
      walk ~in_denom ~in_nonlinear ~enclosing left;
      walk ~in_denom ~in_nonlinear ~enclosing right
    | BinOp { op = Pow; left; right } ->
      let base_nonlinear = in_nonlinear || pow_is_nonlinear right in
      walk ~in_denom ~in_nonlinear:base_nonlinear ~enclosing:e left;
      (* Exponent is always nonlinear context. *)
      walk ~in_denom ~in_nonlinear:true ~enclosing:e right
    | BinOp { op = (Min | Max | Mod | Eq | Neq | Lt | Gt | Le | Ge);
              left; right } ->
      walk ~in_denom ~in_nonlinear:true ~enclosing:e left;
      walk ~in_denom ~in_nonlinear:true ~enclosing:e right
    | UnOp { op = Neg; arg } ->
      (* Negation is linear. *)
      walk ~in_denom ~in_nonlinear ~enclosing arg
    | UnOp { op = (Exp | Log | Sqrt | Abs | Floor | Ceil | Sin | Cos | Tanh);
             arg } ->
      walk ~in_denom ~in_nonlinear:true ~enclosing:e arg
    | Cond { pred; then_; else_ } ->
      (* Cond is in the nonlinear-use bucket (proposal). A parent inside
         any branch (or the predicate) that is not in a denominator is a
         nonlinear use. *)
      walk ~in_denom ~in_nonlinear:true ~enclosing:e pred;
      walk ~in_denom ~in_nonlinear:true ~enclosing:e then_;
      walk ~in_denom ~in_nonlinear:true ~enclosing:e else_
  in
  walk ~in_denom:false ~in_nonlinear:false ~enclosing:rate rate;
  { parents = List.rev !parents; nonlinear = !nonlinear }

(* Factor a rate into (numerator, denominator) by pulling every Div
   right-operand into a single product denominator. Multiplication
   distributes over the factoring; anything that is not a Div/Mul is a
   numerator atom. This keeps the per-pool weight expression in the
   clean `coeff / normalizer` form the proposal shows
   (e.g. β·S/N rather than β·S·I/N − 0/N). The denominator is a frozen
   normalizer at the event instant, so its parent-count dependence is
   preserved verbatim. *)
let rec split_frac (e : expr) : expr * expr =
  match e with
  | BinOp { op = Div; left; right } ->
    let (ln, ld) = split_frac left in
    (ln, BinOp { op = Mul; left = ld; right })
  | BinOp { op = Mul; left; right } ->
    let (ln, ld) = split_frac left in
    let (rn, rd) = split_frac right in
    (BinOp { op = Mul; left = ln; right = rn },
     BinOp { op = Mul; left = ld; right = rd })
  | _ -> (e, Const 1.0)

(* Derivative of a numerator expression with respect to the count of a
   single compartment [comp], treating [Pop comp] as the variable and
   every parameter / time / other compartment as a constant. Because
   the numerator (post [split_frac]) is degree-1 in [comp] (guaranteed
   by [classify_parents]), this derivative is exactly the per-pool
   coefficient, in reduced form — the product rule drops the constant
   factors automatically (∂(β·S·β_i·I)/∂I = β·S·β_i, with no leftover
   I term). This is the v1 specialization of the Phase-4
   ∂rate/∂count(X_k) weighting; here it reduces to the linear
   coefficient. *)
let rec deriv_num_wrt_pop (comp : string) (e : expr) : expr =
  match e with
  | Pop c        -> if c = comp then Const 1.0 else Const 0.0
  | PopSum cs    -> if List.mem comp cs then Const 1.0 else Const 0.0
  | Const _ | Param _ | Time | Dt | Projected | TimeFunc _
  | TableLookup _ -> Const 0.0
  | UncheckedDim u -> deriv_num_wrt_pop comp u.inner
  | BinOp { op = Add; left; right } ->
    BinOp { op = Add; left = deriv_num_wrt_pop comp left;
                       right = deriv_num_wrt_pop comp right }
  | BinOp { op = Sub; left; right } ->
    BinOp { op = Sub; left = deriv_num_wrt_pop comp left;
                       right = deriv_num_wrt_pop comp right }
  | BinOp { op = Mul; left; right } ->
    (* Product rule: d(fg) = f'g + fg'. *)
    BinOp { op = Add;
            left  = BinOp { op = Mul; left = deriv_num_wrt_pop comp left;
                                       right };
            right = BinOp { op = Mul; left; right = deriv_num_wrt_pop comp right } }
  | BinOp { op = Div; left; right } ->
    (* Frozen denominator: ∂(f/g)/∂count = f'/g (g is a frozen
       normalizer at the event instant). split_frac removes top-level
       Divs, so this only fires for a nested normalizer in the
       numerator factoring — still frozen. *)
    BinOp { op = Div; left = deriv_num_wrt_pop comp left; right }
  | BinOp _ | UnOp _ | Cond _ ->
    (* Nonlinear constructs cannot contain a linear parent in the
       numerator (classify_parents would have rejected the rate), so
       the parent does not occur here: derivative is 0. *)
    Const 0.0

(** The per-pool weight expression for parent compartment [comp].

    Factoring rate = numerator / denominator (frozen normalizer), the
    weight is the coefficient of the parent count:

      weight(comp) = (∂numerator/∂count(comp)) / denominator

    For `β·S·I/N`, weight(I) = β·S/N; for the multi-pool
    `β·S·(β_I·I + β_A·A)/N`, weight(I) = β·β_I·S/N and
    weight(A) = β·β_A·S/N. *)
let weight_of_parent (rate : expr) (comp : string) : expr =
  let (num, denom) = split_frac rate in
  let coeff = Autodiff.simplify_fixpoint (deriv_num_wrt_pop comp num) in
  Autodiff.simplify_fixpoint (BinOp { op = Div; left = coeff; right = denom })

(** [parent_pool_weights ~sources rate] returns the linear
    decomposition of [rate] over its parent pools as
    [(compartment, weight_expr)] pairs, in first-seen order.

    Precondition: [rate] passed [classify_parents] with no nonlinear
    use (the caller emits E601 otherwise). *)
let parent_pool_weights ~(sources : string list) (rate : expr)
    : (string * expr) list =
  let { parents; _ } = classify_parents ~sources rate in
  List.map (fun comp -> (comp, weight_of_parent rate comp)) parents
