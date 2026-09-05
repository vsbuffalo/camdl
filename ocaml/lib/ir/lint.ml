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
   Today: the dead-compartment check (L402) and the shared-measurement
   check (L404). *)

open Ir

(* ── Diagnostic results ─────────────────────────────────────────────────── *)

(* Lints are always Warning severity (never blocking). The single-
   constructor `severity` keeps the type parallel to Dimcheck's and lets
   the compiler-side mapping pattern-match exhaustively. *)
type severity = Warning

(* The declaration a lint concerns, so the compiler can resolve it back to a
   source loc — Lint runs on the IR, which has no spans. Mirrors
   [Dimcheck.subject], which serves the same purpose for the dimension pass;
   one variant per declaration kind a lint can point at, so a lint cannot name
   a compartment and a stream at once. *)
type subject =
  | SCompartment of string
  | SObservation of string

type diagnostic = {
  severity : severity;
  code     : string;
  message  : string;
  detail   : string option;
  hint     : string option;
  subject  : subject option;
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
       add_expr mean.expr; add_expr dispersion.expr; add_expr pi.expr)
  ) m.observations;

  (* Model-level shared bindings: a compartment used only in `let N = S+I+R`
     is live. *)
  List.iter (fun (b : binding) -> add_expr b.bexpr) m.bindings;

  (* Initial conditions: a compartment is live if it has an init target,
     even if it appears nowhere else. *)
  List.iter
    (fun (comp, spec) ->
       add comp;
       (* Every expression the spec evaluates, so a compartment read only
          through a law's argument counts as live. *)
       List.iter add_expr (Ir.init_spec_exprs spec))
    m.initial_conditions;

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
      subject = Some (SCompartment name);
    }
  ) dead

(* ── Lint L404: two streams scoring one measurement ─────────────────────── *)

(* The joint observation log-likelihood is a plain sum over bound streams, so
   two streams that read the same underlying measurements contribute that
   evidence twice. Nothing else notices: the counts are never doubled, so the
   posterior simply concentrates as though there were twice the data, and
   every convergence diagnostic reads clean.

   A shared PROJECTION alone is not that condition, and keying on it is far too
   broad: driving cases and deaths off one confirmation flow, with their own
   multipliers and their own dispersion, is ordinary multi-stream modelling.
   Measured on a 98-model surveillance corpus, a projection-only key fired on
   58 models / 63 collision groups, and not ONE of those groups repeated a
   scored column name — every one was a false positive. A lint that fires on
   three models in five teaches people to ignore its code.

   The condition is a shared projection AND a shared SCORED COLUMN. `scored` is
   the `~` LHS, the declared value column the likelihood consumes; that name is
   the quantity being measured, not merely the latent state behind it. Two
   streams scoring the same named quantity off one projection are reading one
   measurement twice; two streams scoring DIFFERENT named quantities off one
   projection are two observation processes on one latent state, which is
   correct and common.

   This runs on the EXPANDED model — after stratification expansion, index
   resolution and binder substitution — because two different SPELLINGS become
   one concrete projection strictly later. `resolve_index_order` normalises
   named indices to declared order and drops the `INamed` label, so
   `die[child, north]` and `die[patch = north, age = child]` are one flow by
   the time we see them; and `index_item_to_str env` substitutes the stream's
   own binder, so a stratified stream whose projection ignores its binder
   expands to N leaves all naming ONE flow. A syntactic check on what the
   modeller wrote sees none of this (the reasoning gh#678 established for the
   within-projection case). *)

(* A projection's identity for the "same latent quantity?" question. The two
   sum variants are canonicalised — sorted (a sum is commutative, so the term
   order is not part of the quantity) and collapsed to the scalar form at
   length one — so the comparison is over the quantity, not the spelling.
   `KExpr` compares derived projections structurally, which is exact for the
   spellings the expander produces and conservative otherwise: it can miss a
   commuted rewrite, never invent a collision. *)
type projection_key =
  | KFlow of string list      (* accumulated flows, sorted *)
  | KPop  of string list      (* compartments read at the instant, sorted *)
  | KExpr of expr             (* a derived function of the state *)

let projection_key (p : projection) : projection_key =
  let canon names = List.sort String.compare names in
  match p with
  | CumulativeFlow f     -> KFlow [f]
  | CumulativeFlowSum fs -> KFlow (canon fs)
  | CurrentPop c         -> KPop [c]
  | CurrentPopSum cs     -> KPop (canon cs)
  | DerivedExpr e        -> KExpr e

(* A stream's measurement identity: the latent quantity its projection reads,
   paired with the name of the column its likelihood scores. Both halves are
   load-bearing — see the header comment for why the projection alone is not
   the condition. *)
type measurement_key = projection_key * string

let measurement_key (o : observation_model) : measurement_key =
  (projection_key o.projection, o.scored)

(* What the shared projection reads, in the words a modeller would use. *)
let projection_phrase (k : projection_key) : string =
  let quoted names = String.concat " + " (List.map (Printf.sprintf "'%s'") names) in
  match k with
  | KFlow [f]  -> Printf.sprintf "each accumulates the flow '%s'" f
  | KFlow fs   -> Printf.sprintf "each accumulates the same pooled flows (%s)" (quoted fs)
  | KPop  [c]  -> Printf.sprintf "each reads the compartment '%s'" c
  | KPop  cs   -> Printf.sprintf "each reads the same compartments (%s)" (quoted cs)
  | KExpr _    -> "each evaluates the same derived expression over the state"

(* "'a' and 'b'" / "'a', 'b' and 'c'" — the streams of one collision group. *)
let stream_list (names : string list) : string =
  let quoted = List.map (Printf.sprintf "'%s'") names in
  match List.rev quoted with
  | []       -> ""
  | [only]   -> only
  | last :: rev_init ->
    String.concat ", " (List.rev rev_init) ^ " and " ^ last

(* Group the streams by measurement identity and report every group of two or
   more — ONE diagnostic per group, naming all its streams, so three streams
   on one measurement give one report rather than three pairs. Groups are keyed
   by an association list rather than a hashtable because the key carries an
   `expr`, and iteration follows declaration order, so the output is
   deterministic. *)
let check_shared_projections (m : model) : diagnostic list =
  let groups : (measurement_key * string list ref) list ref = ref [] in
  List.iter (fun (o : observation_model) ->
    let k = measurement_key o in
    match List.assoc_opt k !groups with
    | Some names -> names := o.name :: !names
    | None       -> groups := (k, ref [o.name]) :: !groups
  ) m.observations;
  List.filter_map (fun ((proj, scored), names) ->
    match List.rev !names with
    | ([] | [_]) -> None
    | streams ->
      Some
        { severity = Warning;
          code = "L404";
          message =
            Printf.sprintf
              "observation streams %s score the same quantity '%s'"
              (stream_list streams) scored;
          detail =
            Some (Printf.sprintf
              "%s, and each scores it as '%s' — one latent quantity read one \
               way, under one measurement name. The joint log-likelihood is a \
               sum over bound streams, so data bound to all of them adds that \
               evidence once per stream. No count doubles and no convergence \
               diagnostic fires; the posterior just concentrates as if there \
               were that many times the data."
              (projection_phrase proj) scored);
          hint =
            Some "if these are one quantity at two resolutions, or one page \
                  read twice, bind only one of them. If they really are \
                  independent observation processes on the same latent quantity \
                  — two laboratories, a confirmed and a suspected pipeline — \
                  the joint is right; keep both, and distinct scored column \
                  names make that visible.";
          (* Point at the first stream of the group; the message names the rest.
             `obs_loc` maps an expanded leaf back to its base declaration, so a
             stratified collision lands on the declaration line. *)
          subject = Some (SObservation (List.hd streams));
        }
  ) (List.rev !groups)

(* ── Entry point ────────────────────────────────────────────────────────── *)

(* The registry of lint checks. Each is `model -> diagnostic list`; adding a
   new lint is a one-line append here plus the check function above. *)
let checks : (model -> diagnostic list) list = [
  check_dead_compartments;
  check_shared_projections;
]

let check_model (m : model) : result =
  let diagnostics = List.concat_map (fun check -> check m) checks in
  { diagnostics }
