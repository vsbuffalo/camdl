(* Model linter for camdl IR.

   Lints catch semantically valid but discouraged patterns — a model that
   compiles and runs but is likely a mistake. They emit at Warning
   severity (see ocaml/lib/compiler/diagnostics.ml `has_errors`, which
   blocks only on Error), so a lint renders with a hint but never fails
   the build.

   Structure mirrors Dimcheck: `check_model : model -> result`, where
   `result` carries a list of this module's own `diagnostic` records.
   `compiler.ml`'s `run_lint` maps each onto `Diagnostics.warning`.

   `check_model` runs a LIST of individual lint-check functions, so a
   future lint (unused-parameter, dead-transition) is a one-line append.
   Today only the dead-compartment check (L402) is implemented. *)

open Ir

(* ── Diagnostic results ─────────────────────────────────────────────────── *)

(* Lints are always Warning severity (never blocking). The single-
   constructor `severity` keeps the type parallel to Dimcheck's and lets
   the compiler-side mapping pattern-match exhaustively. *)
type severity = Warning

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  (* The compartment a compartment-scoped lint (L402) concerns, so the compiler
     can resolve it to a source loc. Lint runs on the IR, which has no spans. *)
  compartment : string option;
}

type result = {
  diagnostics : diagnostic list;
}

(* ── Expression walk: collect compartment references ────────────────────── *)

(* Every compartment a single expression refers to. Pop and PopSum are the
   only constructors that name compartments; the rest either name something
   else (Param, BindingRef, TimeFunc, TableLookup-name) or carry sub-exprs
   we must recurse into. The match is intentionally exhaustive over every
   `Ir.expr` constructor — a non-exhaustive match is a compile error, which
   forces this list to track the AST as new constructors are added.

   Compartment names are accumulated onto [acc] (a fold avoids quadratic
   list concatenation on deep Reduce sums). *)
let rec pops_in_expr (acc : string list) (e : expr) : string list =
  match e with
  | Pop name -> name :: acc
  | PopSum names -> List.rev_append names acc
  (* Leaves with no compartment reference and no sub-expression. *)
  | Const _ | Param _ | Time | Dt | TimeFunc _ | BindingRef _ | Projected
  | ObsColumnRef _ | ObsAnchor _ ->
    acc
  | PerEvalRef _ -> failwith "PerEvalRef before LICM (gh#272 compiler invariant)"
  (* Compound nodes: recurse into every sub-expression. *)
  | BinOp { op = _; left; right } ->
    pops_in_expr (pops_in_expr acc left) right
  | UnOp { op = _; arg } ->
    pops_in_expr acc arg
  | Cond { pred; then_; else_ } ->
    pops_in_expr (pops_in_expr (pops_in_expr acc pred) then_) else_
  (* TableLookup's first field is the table NAME, not a compartment; the
     index expressions can reference compartments (e.g. dynamic strata). *)
  | TableLookup (_name, idx_exprs) ->
    List.fold_left pops_in_expr acc idx_exprs
  | Reduce terms ->
    List.fold_left pops_in_expr acc terms
  | UncheckedDim { inner; _ } ->
    pops_in_expr acc inner

(* ── Reference collection across the whole model ────────────────────────── *)

(* Build the set of compartment names referenced anywhere in [m]. A
   compartment is "referenced" if it appears in ANY of: transition
   stoichiometry / metadata source-dest / rate, ODE equations, intervention
   actions, observation projections + likelihoods, model-level bindings,
   initial conditions, the balance constraint, the identity-tracked list, or
   time-function definitions. A name appearing in any of these is live —
   only a name in NONE of them is dead. *)
let referenced_compartments (m : model) : (string, unit) Hashtbl.t =
  let refs = Hashtbl.create 64 in
  let add name = Hashtbl.replace refs name () in
  let add_opt = function Some s -> add s | None -> () in
  let add_expr e = List.iter add (pops_in_expr [] e) in

  (* Transitions: stoichiometry strings, metadata source/dest, rate expr,
     overdispersion sigma², and (defensively) lineage parent-pool weights. *)
  List.iter (fun (tr : transition) ->
    List.iter (fun (comp, _delta) -> add comp) tr.stoichiometry;
    (match tr.metadata with
     | Some md ->
       add_opt md.source_compartment;
       add_opt md.dest_compartment
     | None -> ());
    add_expr tr.rate;
    (match tr.draw_method with
     | DrawOverdispersed { sigma_sq; _ } -> add_expr sigma_sq
     | DrawPoisson | DrawDeterministic -> ());
    (match tr.lineage with
     | Some lin ->
       List.iter (fun (parent, weight) -> add parent; add_expr weight)
         lin.parent_pool_weights
     | None -> ())
  ) m.transitions;

  (* ODE equations: the compartment whose derivative this defines, plus any
     compartments in the RHS. *)
  List.iter (fun (eq : ode_equation) ->
    add eq.compartment;
    add_expr eq.derivative
  ) m.ode_equations;

  (* Interventions: target compartment(s) of each action + expr operands. *)
  List.iter (fun (iv : intervention) ->
    List.iter (fun (act : action) ->
      match act with
      | FractionTransfer { src; dst; fraction } ->
        add src; add dst; add_expr fraction
      | AbsoluteTransfer { src; dst; count } ->
        add src; add dst; add_expr count
      | Set { compartment; value } ->
        add compartment; add_expr value
      | AddAction { add_compartment; add_count } ->
        add add_compartment; add_expr add_count
    ) iv.actions
  ) m.interventions;

  (* Observations: projection targets + likelihood-parameter exprs.
     CumulativeFlow's string is a FLOW/transition name, NOT a compartment —
     excluded deliberately. *)
  List.iter (fun (obs : observation_model) ->
    (match obs.projection with
     | CumulativeFlow _flow -> ()   (* transition name, not a compartment *)
     | CumulativeFlowSum _flows -> ()  (* transition names, not compartments *)
     | CurrentPop name -> add name
     | CurrentPopSum names -> List.iter add names
     | DerivedExpr e -> add_expr e);
    (match obs.likelihood with
     | Poisson { rate } -> add_expr rate.expr
     | NegBinomial { mean; dispersion } -> add_expr mean.expr; add_expr dispersion.expr
     | Normal { mean; sd } -> add_expr mean.expr; add_expr sd.expr
     | Binomial { n; p } -> add_expr n; add_expr p.expr
     | BetaBinomial { n; alpha; beta } -> add_expr n; add_expr alpha.expr; add_expr beta.expr
     | Beta { mean; concentration } -> add_expr mean.expr; add_expr concentration.expr
     | Bernoulli { p } -> add_expr p.expr
     | ZeroInflatedNegBinomial { mean; dispersion; pi } ->
       add_expr mean; add_expr dispersion; add_expr pi)
  ) m.observations;

  (* Model-level shared bindings: a compartment used only in `let N = S+I+R`
     is live. *)
  List.iter (fun (b : binding) -> add_expr b.bexpr) m.bindings;

  (* Initial conditions: a compartment is live if it has an init target,
     even if it appears nowhere else. *)
  (match m.initial_conditions with
   | Explicit pairs -> List.iter (fun (comp, _v) -> add comp) pairs
   | Parameterized pairs -> List.iter (fun (comp, e) -> add comp; add_expr e) pairs
   | FromDistribution pairs -> List.iter (fun (comp, _d) -> add comp) pairs);

  (* Balance constraint: target compartment + balance expr operands. *)
  (match m.balance with
   | Some bal -> add bal.balance_target; add_expr bal.balance_expr
   | None -> ());

  (* Identity-tracked compartments (lineage subsystem). *)
  List.iter add m.identity_tracked_compartments;

  (* Time-function definitions can reference compartments inside their
     defining expressions — cheap safety to keep the false-positive rate at
     zero. *)
  List.iter (fun (tf : time_function) ->
    List.iter add_expr (Autodiff.forcing_coeff_exprs tf.kind)
  ) m.time_functions;

  refs

(* ── Lint L402: dead compartment ────────────────────────────────────────── *)

(* A compartment declared in `model.compartments` but referenced nowhere is
   almost certainly a leftover from editing — it contributes nothing to the
   dynamics, observations, or initial state. We flag it (one L402 per dead
   compartment, sorted for deterministic output) rather than error, since
   such a model is still well-formed and runnable. *)
let check_dead_compartments (m : model) : diagnostic list =
  let refs = referenced_compartments m in
  let dead =
    List.filter_map (fun (c : compartment) ->
      if Hashtbl.mem refs c.name then None else Some c.name
    ) m.compartments
  in
  let dead = List.sort_uniq String.compare dead in
  List.map (fun name ->
    { severity = Warning;
      code = "L402";
      message =
        Printf.sprintf "compartment '%s' is declared but never used" name;
      detail = None;
      hint =
        Some "remove it, or wire it into a transition / init / observation";
      compartment = Some name;
    }
  ) dead

(* ── Entry point ────────────────────────────────────────────────────────── *)

(* The registry of lint checks. Each is `model -> diagnostic list`; adding a
   new lint is a one-line append here plus the check function above. *)
let checks : (model -> diagnostic list) list = [
  check_dead_compartments;
]

let check_model (m : model) : result =
  let diagnostics = List.concat_map (fun check -> check m) checks in
  { diagnostics }
