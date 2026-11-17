(** Constant folding of a compiled model: a lookup into a table of literal
    numbers becomes the number, and the terms it zeroes are removed, so a
    sum over a sparse coupling matrix evaluates only its nonzero terms.

    The expander turns [sum(q in patch, W[p, q] * ...)] into one
    {!Ir.Reduce} with a term for every patch, each holding a
    {!Ir.TableLookup} of [W] at constant indices. {!fold_model} writes the
    cell's value in place of each such lookup, then applies a few identities
    bottom-up: [0 * x] and [x * 0] become [0], [1 * x] and [x * 1] become
    [x], [0 + x], [x + 0] and [x - 0] become [x], [+], [-], [*], [min] and
    [max] of two literals become their value, and a [Reduce] loses its zero
    terms. With [W] sparse, a sum over [P] patches keeps the [k] nonzero
    terms per row. The runtime needs no sparse evaluator: it evaluates the
    same {!Ir.Reduce}, with fewer terms.

    The pass changes the IR only when the model has an inline table whose
    every cell is a numeric literal. Otherwise {!fold_model} returns the
    model unchanged, and the identities are not applied anywhere. A table
    with a parameter in a cell is left as lookups. Division, powers, [mod] and
    comparisons of literals are not folded, so that their handling of
    degenerate values stays the evaluator's, and there is no [0 / x] rule.
    The collapse therefore depends on how the term is written: in

    {@camdl[
      infection[p in patch] : S[p] --> I[p] @ beta * S[p] * sum(q in patch, W[p, q] * I[q] / N[q])
    ]}

    the term parses as [(W[p, q] * I[q]) / N[q]], so a zero cell leaves
    [0 / N_a] behind and the term is kept; written
    [W[p, q] * (I[q] / N[q])], it folds to [0] and is dropped.

    The pass runs after differentiation and before loop-invariant code
    motion, so it folds the gradient fields as well, and never meets an
    {!Ir.PerEvalRef}. It rewrites transition rates and their gradients,
    overdispersion variances and their gradients, bindings, ODE equations,
    observation projections, likelihood arguments with their gradients, and
    [ic_grad]; it leaves initial conditions, forcings, interventions and
    quantities alone.

    Each rewrite keeps the value of a finite expression: a literal cell is
    the value the lookup returns, and a zero term adds nothing to a finite
    sum. [0 * x] is [0] only when [x] is finite, so a factor that would
    evaluate to infinity or NaN, such as an unguarded division by an empty
    patch, folds to [0] where the unfolded model would report a numerical
    error. A lookup whose constant index is out of range is left for the
    runtime's out-of-bounds policy. The Rust test [gate_constant_fold_ab]
    checks, on a sparse-coupling model, that trajectories are identical with
    and without the pass. Setting [CAMDL_NO_CONSTANT_FOLD] in the
    environment turns it off. *)

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
  (* `ObsAnchor` is a leaf here and must stay one — its value is not
     known until a run binds its data, so folding it to a number at compile time
     is exactly the data-dependence this design exists to avoid. *)
  (* A model constant stays a reference, not its literal: folding it would let
     `2 * omega` and `omega * beta` re-associate with neighbouring literals and
     move the arithmetic, and would drop the name every display shows. *)
  (* A unit literal stays a leaf: folding it into neighbouring literals would
     re-associate the arithmetic. *)
  | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Pop _ | PopSum _ | Time | Dt | TimeFunc _ | ForcingBasis _ | BindingRef _
  | Projected | ObsColumnRef _ | ObsAnchor _ ->
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

(* Fold a deriv_entry / grad_map: only a real [DEGrad] expression carries an
   expr to fold; a coded [DEUnsupported] refusal has none. *)
let fold_deriv_entry fe (de : deriv_entry) : deriv_entry =
  match de with DEGrad e -> DEGrad (fe e) | DEUnsupported _ -> de

let fold_grad_map fe (gm : grad_map) : grad_map =
  List.map (fun (p, de) -> (p, fold_deriv_entry fe de)) gm

(* Fold a differentiable likelihood argument: its expression and its gradient
   map. *)
let fold_diffable fe (d : diffable) : diffable =
  { expr = fe d.expr; grad = fold_grad_map fe d.grad;
    proj_grad = Option.map (fold_deriv_entry fe) d.proj_grad }

(* Overdispersion σ² and its gradient. *)
let fold_draw_method fe (dm : draw_method) : draw_method =
  match dm with
  | DrawOverdispersed { sigma_sq; sigma_sq_grad } ->
      DrawOverdispersed
        { sigma_sq = fe sigma_sq; sigma_sq_grad = fold_grad_map fe sigma_sq_grad }
  | (DrawPoisson | DrawDeterministic) as dm -> dm

(* Observation projection ([DerivedExpr] only) + every likelihood argument and
   its gradient map. Exhaustive so a new likelihood variant is a compile error. *)
let fold_projection fe (p : projection) : projection =
  match p with
  | DerivedExpr e -> DerivedExpr (fe e)
  | (CumulativeFlow _ | CurrentPop _ | CurrentPopSum _ | CumulativeFlowSum _
    | FlowRatio _) as p -> p

let fold_likelihood fe (lik : likelihood) : likelihood =
  let fd = fold_diffable fe in
  match lik with
  | Poisson pl -> Poisson { rate = fd pl.rate }
  | NegBinomial nb -> NegBinomial { mean = fd nb.mean; dispersion = fd nb.dispersion }
  | Normal n -> Normal { mean = fd n.mean; sd = fd n.sd }
  | Binomial b -> Binomial { n = fe b.n; p = fd b.p }
  | BetaBinomial bb -> BetaBinomial { n = fe bb.n; alpha = fd bb.alpha; beta = fd bb.beta }
  | Beta b -> Beta { mean = fd b.mean; concentration = fd b.concentration }
  | Bernoulli b -> Bernoulli { p = fd b.p }
  | ZeroInflatedNegBinomial zi ->
    ZeroInflatedNegBinomial { mean = fd zi.mean; dispersion = fd zi.dispersion; pi = fd zi.pi }

(* Fold the expr-bearing fields where a sparse coupling matrix actually
   appears: transition rates + their gradients + σ² draw method, model-level
   bindings, ODE derivatives, and observation projections/likelihood arguments +
   their gradients. Folding a subset is sound (each folded expr keeps its value);
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
            { t with rate = fe t.rate;
                     rate_grad = fold_grad_map fe t.rate_grad;
                     rate_state_grad = fold_grad_map fe t.rate_state_grad;
                     draw_method = fold_draw_method fe t.draw_method })
          m.transitions;
      bindings = List.map (fun (b : binding) -> { b with bexpr = fe b.bexpr }) m.bindings;
      ode_equations =
        List.map (fun (eq : ode_equation) -> { eq with derivative = fe eq.derivative }) m.ode_equations;
      observations =
        List.map
          (fun (o : observation_model) ->
            { o with projection = fold_projection fe o.projection;
                     projection_state_grad = fold_grad_map fe o.projection_state_grad;
                     likelihood = fold_likelihood fe o.likelihood })
          m.observations;
      (* ∂init/∂θ seed: fold each compartment's grad_map like a rate's,
         so a table lookup surviving into an IC derivative collapses consistently
         with the rest of the model. *)
      ic_grad = List.map (fun (c, gm) -> (c, fold_grad_map fe gm)) m.ic_grad;
    }
