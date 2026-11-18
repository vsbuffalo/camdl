(** The linter of an expanded model: warnings for a model that is valid and
    runs, but is probably not what its author meant.

    {!check_model} reads an {!Ir.model} and returns a {!result} holding a
    list of {!diagnostic}s, all of severity [Warning]; it changes nothing.
    The compiler runs it right after the dimension check, in both a compile
    and [camdlc check], and turns each diagnostic into a warning placed at
    the declaration its {!subject} names. A warning never stops a compile.

    Two lints are implemented here, each a function from the model to its
    diagnostics, listed in [checks]:

    - [L402], a dead compartment: a compartment that nothing in the model
      reads or writes, neither a transition (its stoichiometry, rate,
      overdispersion or lineage weights), an ODE equation, an intervention
      action, an observation's projection or likelihood, a shared binding,
      an initial condition, the balance, the identity-tracked list nor a
      forcing's arguments. Quantities and contrasts are not consulted, so a
      compartment read only by a quantity is reported.
    - [L404], one measurement scored twice: two or more observation streams
      with the same projection, compared after expansion, and the same
      scored column name. The joint log-likelihood is a sum over streams, so
      data bound to both counts the same evidence twice and the posterior is
      too narrow, with no diagnostic of a fit to show it. Streams sharing a
      projection but scoring different columns, such as cases and deaths
      off one flow, are two observation processes on one latent quantity,
      and are not reported.

    {@camdl[
      time_unit = 'days
      compartments { S, I, R, Q }
      parameters {
        beta  : rate
        gamma : rate
        rho   : probability
        k     : positive
      }
      let N = S + I + R
      transitions {
        infection : S --> I @ beta * S * I / N
        recovery  : I --> R @ gamma * I
      }
      observations {
        cases {
          columns       { time : time, cases : count }
          covers        = closing_at(time, 7 'days)
          projected     = incidence(infection)
          emit_schedule = every 7 'days
          cases ~ neg_binomial(mean = rho * projected, r = k)
        }
        cases_again {
          columns       { time : time, cases : count }
          covers        = closing_at(time, 7 'days)
          projected     = incidence(infection)
          emit_schedule = every 7 'days
          cases ~ poisson(rate = rho * projected)
        }
      }
      init { S = 990  I = 10 }
    ]}

    gives [compartment 'Q' is declared but never used] (L402) and
    [observation streams 'cases' and 'cases_again' score the same quantity
    'cases'] (L404).

    Other lints live elsewhere: the expander reports L401 (a fixed time
    literal inside an Euler-correction pattern in a rate) and L403 (a rate
    forcing divided by a time-conversion constant), because they need the
    syntax tree, and {!Dimcheck} reports I316 about log-normal priors. *)

open Ir

(* ── Diagnostic results ─────────────────────────────────────────────────── *)

(** How serious a lint is: always a warning, which never stops a compile.
    The type has one constructor so that it reads like {!Dimcheck.severity}
    and the compiler's mapping of it is an exhaustive match. *)
type severity = Warning

(** The declaration a lint concerns. The IR carries no source locations, so
    the compiler looks up the declaration to place the warning. *)
type subject =
  | SCompartment of string   (** A compartment, by expanded name. *)
  | SObservation of string   (** An observation stream; for a stratified stream the compiler places the warning at its family's declaration. *)

(** One finding of the linter, before the compiler turns it into a located
    warning. *)
type diagnostic = {
  severity : severity;
  code     : string;          (** The lint's code, [L402] or [L404]. *)
  message  : string;
  detail   : string option;   (** Why the pattern is probably a mistake. *)
  hint     : string option;   (** What to do about it. *)
  subject  : subject option;
}

(** Every finding of the linter for one model: the [L402]s, sorted by
    compartment name, then the [L404]s, in the order of each group's first
    stream. *)
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
  | Const _ | UnitLiteral _ | Param _ | ConstRef _ | Time | Dt | TimeFunc _ | ForcingBasis _ | BindingRef _ | Projected
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
  (* TableLookup's first field is the table name, not a compartment; the
     index expressions can reference compartments (e.g. dynamic strata). *)
  | TableLookup (_name, idx_exprs) ->
    List.fold_left pops_in_expr acc idx_exprs
  | Reduce terms ->
    List.fold_left pops_in_expr acc terms
  | UncheckedDim { inner; _ } ->
    pops_in_expr acc inner

(* ── Reference collection across the whole model ────────────────────────── *)

(* Build the set of compartment names referenced anywhere in [m]. A
   compartment is referenced if it appears in any of: transition
   stoichiometry / metadata source-dest / rate, ODE equations, intervention
   actions, observation projections + likelihoods, model-level bindings,
   initial conditions, the balance constraint, the identity-tracked list,
   time-function definitions, or generated quantities. Only a name in none of
   them is dead. *)
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
      | FractionTransfer { src; dst; fraction; _ } ->
        add src; add dst; add_expr fraction
      | AbsoluteTransfer { src; dst; count } ->
        add src; add dst; add_expr count
      | GroupedTransfer { pairs; amount } ->
        List.iter (fun (src, dst) -> add src; add dst) pairs;
        (match amount with
         | GroupedFraction { fraction; _ } -> add_expr fraction
         | GroupedCount { count } -> add_expr count)
      | Set { compartment; value; _ } ->
        add compartment; add_expr value
      | AddAction { add_compartment; add_count; _ } ->
        add add_compartment; add_expr add_count
    ) iv.actions
  ) m.interventions;

  (* Observations: projection targets + likelihood-parameter exprs.
     CumulativeFlow's string is a flow (transition) name, not a compartment,
     and is excluded deliberately. *)
  List.iter (fun (obs : observation_model) ->
    (match obs.projection with
     | CumulativeFlow _flow -> ()   (* transition name, not a compartment *)
     | CumulativeFlowSum _flows -> ()  (* transition names, not compartments *)
     | FlowRatio _ -> ()               (* two flow lists, still not compartments *)
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

  (* Generated quantities: the state series a quantity reduces, and every
     expression of its reduction (a threshold, a `value_at` time). A quantity
     derived from other quantities (QBDerived), or reducing an observation
     stream, names no compartment. Contrasts name no compartment either: an
     operand is a run's quantity or observation stream (Ir.CRunMember). *)
  let add_anchor = function ATime e -> add_expr e | AObs _ -> () in
  List.iter (fun (q : quantity) ->
    match q.q_body with
    | QBDerived _ -> ()
    | QBReduced { source; reduce } ->
      (match source with
       | QSState e -> add_expr e
       | QSObservation _ -> ());
      (match reduce with
       | None | Some RIntegral
       | Some (RValue (VFinal | VMax | VMin | VMean))
       | Some (RTime (TimeOfMax | TimeOfMin)) -> ()
       | Some (RValue (VCountAbove c | VCountBelow c))
       | Some (RTime (FirstAbove c | FirstBelow c | LastAbove c | LastBelow c)) ->
         add_expr c
       | Some (RValue (VValueAt a)) -> add_anchor a)
  ) m.quantities;

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
        Some "remove it, or wire it into a transition / init / observation / quantity";
      subject = Some (SCompartment name);
    }
  ) dead

(* ── Lint L404: two streams scoring one measurement ─────────────────────── *)

(* A shared projection alone is not the condition: driving cases and deaths
   off one confirmation flow, with their own multipliers and dispersions, is
   ordinary multi-stream modelling, and a lint keyed on the projection alone
   fires on such models. The condition is a shared projection and a shared
   scored column. `scored` is the `~` left-hand side, the value column the
   likelihood consumes; that name is the quantity being measured.

   The check runs on the expanded model because two spellings of one flow
   become one name only there: `resolve_index_order` puts named indices in
   declared order, so `die[child, north]` and `die[patch = north, age =
   child]` are one flow, and a stratified stream whose projection ignores its
   binder expands to leaves that all name one flow. *)

(** The latent quantity an observation stream's projection reads, in a form
    where two spellings of one quantity compare equal. A sum of flows or of
    compartments is sorted, since a sum's order is not part of the quantity,
    and a single flow or compartment is a sum of one. A derived projection is
    compared as an expression, structurally: two streams can be missed when
    one commutes the other's expression, but two different quantities never
    compare equal. *)
type projection_key =
  | KFlow  of string list      (** Accumulated flows, sorted. *)
  | KPop   of string list      (** Compartments read at an instant, sorted. *)
  | KExpr  of expr             (** A derived function of the state. *)
  | KRatio of string list * string list
  (** A fraction of accumulated flows, numerator and denominator each
      sorted. It is its own key, so a ratio stream and a count stream on its
      denominator are two measurements, never one. *)

let projection_key (p : projection) : projection_key =
  let canon names = List.sort String.compare names in
  match p with
  | CumulativeFlow f     -> KFlow [f]
  | CumulativeFlowSum fs -> KFlow (canon fs)
  | CurrentPop c         -> KPop [c]
  | CurrentPopSum cs     -> KPop (canon cs)
  | DerivedExpr e        -> KExpr e
  | FlowRatio { numerator; denominator } -> KRatio (canon numerator, canon denominator)

(** What an observation stream measures: the latent quantity its projection
    reads, paired with the name of the column its likelihood scores. Two
    streams with one key are reported by [L404]. *)
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
  | KRatio (num, den) ->
    Printf.sprintf "each accumulates the same fraction of flows (%s over %s)"
      (quoted num) (quoted den)

(* "'a' and 'b'" / "'a', 'b' and 'c'" — the streams of one collision group. *)
let stream_list (names : string list) : string =
  let quoted = List.map (Printf.sprintf "'%s'") names in
  match List.rev quoted with
  | []       -> ""
  | [only]   -> only
  | last :: rev_init ->
    String.concat ", " (List.rev rev_init) ^ " and " ^ last

(* Group the streams by measurement identity and report every group of two or
   more, one diagnostic per group, naming all its streams, so three streams
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
