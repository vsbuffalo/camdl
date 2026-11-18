(** The analysis of a [#[lineage]] transition's rate: whether each firing can
    be attributed to an individual in a parent compartment, and with what
    weights.

    {@camdl[
      compartments { S, I, R }
      parameters { beta : rate  gamma : rate }
      let N = S + I + R
      transitions {
        #[lineage]
        infection : S --> I  @ beta * S * I / N
        recovery  : I --> R  @ gamma * I
      }
    ]}

    A [#[lineage]] transition is one where some existing individual causes
    each event, as an infector causes an infection, so that the runtime can
    record a genealogy. The runtime chooses the cause of a firing in two
    steps: a parent compartment [c] with probability proportional to
    [w_c * count_c], then an individual uniformly within it. That needs the
    rate written as a sum of [w_c * count_c] over the parent compartments,
    and this module finds the [w_c]. For [infection] above the only parent is
    [I], with weight [beta * S / N].

    A parent is a compartment the rate reads, other than the transition's
    sources, that appears in the numerator of the rate outside any nonlinear
    construct. {!classify_parents} walks the rate and puts every such
    reference in one of three buckets. A reference anywhere in the right
    operand of a division is a normaliser, such as [I] inside [N]: it is
    treated as a coefficient frozen at the event and is never a parent. A
    reference inside a nonlinear construct (a power other than [^ 1], [exp],
    [log], [sqrt], [abs], [floor], [ceil], the trigonometric functions,
    [min], [max], [mod], a comparison, or a table index) is a nonlinear use,
    and so is a product whose two operands both read a parent, such as
    [I * I] or [I * A]; the expander refuses the transition with E601. A
    conditional [if p then a else b] is a nonlinear use of every parent in
    it when [p] reads a parent anywhere; when [p] reads none, the branch it
    picks does not depend on the parent counts, and the conditional is
    linear in the parents when [a] and [b] are. Any other reference makes
    the compartment a parent. {!parent_pool_weights} then gives each
    parent's weight: the derivative of the rate's numerator with respect to
    that parent's count, divided by the rate's denominator, after both are
    simplified. For a conditional that derivative is
    [if p then a' else b'], each branch's own.

    [if t > 10 then beta * S * I / N else 0] has the parent [I] with weight
    [if t > 10 then beta * S / N else 0]. This is also the form of a stage
    copy of a competing exit, [if E > 0 then R * E_s / E else 0], whose
    predicate reads only the source.

    The expander calls both, from [Expander.resolve_lineage], on the resolved
    rate of each expanded [#[lineage]] transition, with [let]s inlined rather
    than hoisted into shared bindings so that no parent is hidden behind a
    binding reference. The result becomes the transition's
    {!Ir.transition_lineage}. The module lives in the [ir] library because it
    reads IR expressions.

    The product rule is what makes the numerator of degree one in the
    parents taken together, so that the derivative is the coefficient and
    the weights times the counts sum to the numerator. Without it,
    [beta * S * I * I / (N * N)] would get the weight
    [2 * beta * S * I / (N * N)], twice the rate once multiplied by [I], and
    [beta * S * I * A / N] would attribute the whole rate to each of [I] and
    [A]. *)

open Ir

(** A compartment the rate uses nonlinearly, which makes the transition
    ineligible for lineage, with the construct it is used in, for the E601
    message. In [beta * S * (I + iota)^alpha / N] it is [I] inside
    [(I + iota)^alpha]. *)
type nonlinear_use = {
  comp:    string;       (* the compartment used nonlinearly *)
      (** The compartment. *)
  context: expr;         (* the enclosing nonlinear sub-expression *)
      (** The innermost nonlinear expression around the reference. *)
}

(** What {!classify_parents} found in a rate. *)
type classification = {
  parents:    string list;        (* numerator linear-parent compartments, in first-seen order *)
      (** The parent compartments, each once, in the order the rate first
          reads them. Complete only when [nonlinear] is [None]. *)
  nonlinear:  nonlinear_use option;  (* Some _ ⇒ reject (E601) *)
      (** The first nonlinear use found, if any; the transition is then
          refused. *)
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
                        from the nonlinear-use check, even when they
                        are also inside a nonlinear construct.
      - [in_nonlinear]: under a non-unit power, log/exp/sqrt/sin/cos/
                        tanh/abs, min, max, mod, a comparison, the
                        predicate of a Cond, or either branch of a Cond
                        whose predicate reads a parent.

    A non-source Pop is:
      - Denominator-only if [in_denom] (exempt; not a parent),
      - Nonlinear-use   if [in_nonlinear] and not [in_denom] (reject),
      - Linear parent   otherwise.

    A product outside any denominator whose two operands each read a
    parent is a nonlinear use of the right operand's first parent. *)
let classify_parents ~(sources : string list) (rate : expr) : classification =
  let is_source c = List.mem c sources in
  (* The first parent compartment [e] reads outside a denominator, if any:
     the factor that makes a product of two such operands nonlinear. *)
  let rec first_parent (e : expr) : string option =
    match e with
    | Pop c -> if is_source c then None else Some c
    | PopSum cs -> List.find_opt (fun c -> not (is_source c)) cs
    | BinOp { op = Div; left; _ } -> first_parent left
    | BinOp { left; right; _ } ->
      (match first_parent left with Some _ as p -> p | None -> first_parent right)
    | UnOp { arg; _ } -> first_parent arg
    | UncheckedDim u -> first_parent u.inner
    | Cond { pred; then_; else_ } ->
      List.find_map first_parent [pred; then_; else_]
    | Reduce terms -> List.find_map first_parent terms
    (* A table index is a nonlinear context the walk reports on its own. *)
    | TableLookup _ -> None
    | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Time | Dt | Projected
    | ObsColumnRef _ | TimeFunc _ | ForcingBasis _ | ObsAnchor _ | BindingRef _
    | PerEvalRef _ -> None
  in
  (* Whether [e] reads a compartment other than a source anywhere, a
     denominator included. *)
  let rec reads_parent (e : expr) : bool =
    match e with
    | Pop c -> not (is_source c)
    | PopSum cs -> List.exists (fun c -> not (is_source c)) cs
    | BinOp { left; right; _ } -> reads_parent left || reads_parent right
    | UnOp { arg; _ } -> reads_parent arg
    | UncheckedDim u -> reads_parent u.inner
    | Cond { pred; then_; else_ } -> List.exists reads_parent [pred; then_; else_]
    | Reduce terms -> List.exists reads_parent terms
    | TableLookup (_, args) -> List.exists reads_parent args
    | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Time | Dt | Projected
    | ObsColumnRef _ | TimeFunc _ | ForcingBasis _ | ObsAnchor _ | BindingRef _
    | PerEvalRef _ -> false
  in
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
    (* An anchor is a knot in a forcing breakpoint list, never a
       compartment reference: nothing to record as a parent pool. *)
    | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Time | Dt | Projected | ObsColumnRef _ | TimeFunc _ | ForcingBasis _
    | ObsAnchor _ -> ()
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
    | BinOp { op = Mul; left; right } ->
      (* A product of two parent-reading operands is of degree two in the
         parents: its derivative is not its coefficient (gh#1204). *)
      (if not in_denom && not in_nonlinear && first_parent left <> None then
         match first_parent right with
         | Some c -> flag_nonlinear c e
         | None -> ());
      walk ~in_denom ~in_nonlinear ~enclosing left;
      walk ~in_denom ~in_nonlinear ~enclosing right
    | BinOp { op = (Add | Sub); left; right } ->
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
      (* A predicate that reads no parent picks a branch without regard to
         the parent counts, so the conditional is linear in the parents
         when each branch is. A predicate that reads a parent anywhere,
         even in a denominator, makes the conditional a nonlinear use of
         every parent in it. *)
      walk ~in_denom ~in_nonlinear:true ~enclosing:e pred;
      let branches_nonlinear = in_nonlinear || reads_parent pred in
      walk ~in_denom ~in_nonlinear:branches_nonlinear ~enclosing:e then_;
      walk ~in_denom ~in_nonlinear:branches_nonlinear ~enclosing:e else_
    | Reduce terms ->
      (* A sum is linear (like Add): a parent inside any term is a linear
         parent candidate, in the same denom/nonlinear context. *)
      List.iter (walk ~in_denom ~in_nonlinear ~enclosing) terms
    | BindingRef _ -> ()
      (* A state-bearing BindingRef would hide parent pools, which is why
         the expander does not hoist lets in a #[lineage] rate. *)
    | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  in
  walk ~in_denom:false ~in_nonlinear:false ~enclosing:rate rate;
  { parents = List.rev !parents; nonlinear = !nonlinear }

(* Factor a rate into (numerator, denominator) by pulling every Div
   right-operand into a single product denominator. Multiplication
   distributes over the factoring; anything that is not a Div/Mul is a
   numerator atom. This keeps the per-pool weight expression in the
   `coeff / normalizer` form (β·S/N rather than β·S·I/N − 0/N). The denominator is a frozen
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
   every parameter / time / other compartment as a constant. When the
   numerator (post [split_frac]) is degree-1 in [comp], this derivative is
   exactly the per-pool coefficient, in reduced form — the product rule
   drops the constant factors automatically (∂(β·S·β_i·I)/∂I = β·S·β_i,
   with no leftover I term). [classify_parents] guarantees degree 1: it
   refuses a product of two parent references. *)
let rec deriv_num_wrt_pop (comp : string) (e : expr) : expr =
  match e with
  | Pop c        -> if c = comp then Const 1.0 else Const 0.0
  | PopSum cs    -> if List.mem comp cs then Const 1.0 else Const 0.0
  | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Time | Dt | Projected | ObsColumnRef _ | TimeFunc _ | ForcingBasis _
  | ObsAnchor _ | TableLookup _ -> Const 0.0
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
  | Reduce terms ->
    (* d/dcount (Σ tᵢ) = Σ d/dcount tᵢ (linear, like Add). *)
    Reduce (List.map (deriv_num_wrt_pop comp) terms)
  | BindingRef _ -> Const 0.0   (* lets are not hoisted in a lineage rate *)
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  | Cond { pred; then_; else_ } ->
    (* [classify_parents] admits a parent in a branch only when [pred] reads
       no parent, so the branch [pred] picks is fixed with respect to the
       count: the derivative is that branch's derivative. A conditional
       with no parent in either branch has derivative 0. *)
    let a = Autodiff.simplify_fixpoint (deriv_num_wrt_pop comp then_) in
    let b = Autodiff.simplify_fixpoint (deriv_num_wrt_pop comp else_) in
    if a = Const 0.0 && b = Const 0.0 then Const 0.0
    else Cond { pred; then_ = a; else_ = b }
  | BinOp _ | UnOp _ ->
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
