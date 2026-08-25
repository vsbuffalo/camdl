(* AST for the camdl DSL — mirrors the surface syntax before expansion. *)

(** Source location for error reporting. *)
type loc = {
  file     : string;
  line     : int;   (* 1-indexed *)
  col      : int;   (* 1-indexed *)
  end_line : int;
  end_col  : int;
}

type 'a located = { value : 'a; loc : loc }

let dummy_loc = { file = ""; line = 0; col = 0; end_line = 0; end_col = 0 }

type unit_lit =
  | Days | Weeks | Months | Years
  | PerDay | PerWeek | PerMonth | PerYear
  (* Tier-3 population count (dim P, scale 1). For interpolated
     forcings that carry raw counts (e.g. `pop : interpolated 'count`). *)
  | Count
  (* Tier-3 dimensionless multiplier (dim (0,0), scale 1). The canonical
     choice for forcings that carry a unitless factor around 1.0 —
     seasonal forcing, school-term indicator, reporting multiplier.
     Distinct from the `probability` parameter kind: `'ratio` is the
     unbounded dimensionless case (scalar could be 0.7, 1.3, 50, …),
     `probability` is the bounded [0,1] case. *)
  | Ratio

type bin_op =
  | Add | Sub | Mul | Div | Pow
  | Eq | Neq | Lt | Gt | Le | Ge

type un_op = Neg | Exp | Log | Sqrt | Abs | Floor | Ceil
           | Sin | Cos | Tanh                                  (* gh#58 *)

(* Relational operators usable in a `where` predicate (transition guards and
   restricted sums). A subset of bin_op's comparisons, kept as a separate type
   so the predicate grammar stays restricted (not the full expr language). *)
type relop = RLt | RLe | RGt | RGe | REq | RNe

(* The value a `where` predicate compares a constant table cell against. GoNum
   is a numeric literal (the decidable case); GoName is a bare name, accepted
   only so the decidability check can emit a targeted error when it resolves to
   a parameter (e.g. a fitted radius `dist[p,q] < sparse_thresh`). *)
type guard_operand =
  | GoNum  of float
  | GoName of string

(* `where` predicate. Compile-time-decidable: index comparisons and constant
   table-cell comparisons only — never parameters or compartment state. *)
type guard =
  | GEq  of string * string                              (* index_var == index_val_or_var *)
  | GNeq of string * string
  | GTab of string * string list * relop * guard_operand (* table[idx,…] relop operand *)
  | GAnd of guard * guard
  | GOr  of guard * guard

(** A positional or named index in S[child] or S[age = child] *)
type index_item =
  | IPosn  of expr
  | INamed of string * expr

(** Binding in [a in age], [(a, a_next) in consecutive(age)], [c in compartments] *)
and index_binding =
  | IBind   of string * string               (* var, dim *)
  | IConsec of string * string * string      (* var, var_next, dim *)
  | IComp   of string                        (* compartment-iter var *)

and expr =
  | EConst  of float
  | EUnit   of float * unit_lit
  | EIdent  of string * loc                  (* unresolved name + source loc *)
  | EIndex  of string * index_item list * loc (* S[child] + source loc *)
  | EBinOp  of bin_op * expr * expr
  | EUnOp   of un_op * expr
  (* sum(i in dim [where P], body) + the source loc of the binder `i in dim
     [where P]`. The loc is per-binder, not per-`sum(...)`: the flat form
     `sum(a in age, p in patch, body)` lowers to nested ESum nodes, and a
     diagnostic about one binder's domain must point at that binder. *)
  | ESum    of string * string * guard option * expr * loc
  | ECond   of expr * expr * expr            (* if p then a else b *)
  | EFuncCall of string * (string * expr) list  (* fname(kw=v,...) *)
  | EList   of expr list                     (* [1.0, 2.0] or [[...],[...]] *)
  | ERange  of expr * expr                   (* 7:100 — range literal, only in [...] *)
  (* observations.<stream> — the v1.1 generated-quantity observation source
     (proposal 2026-06-25). Meaningful ONLY inside a `quantities { }` body, where
     the classifier lowers it to Ir.QSObservation; anywhere else resolve_expr
     rejects it (E290). *)
  | EObsAccess of string * loc
  (* <run>.<quantities|observations>.<member> — a run-rooted contrast operand
     (counterfactual-contrasts proposal 2026-06-25). `run` is an explicit
     scenario name or the reserved `fitted`; `ns` selects the sub-namespace;
     `member` names a quantity / observation stream on that run. Meaningful
     ONLY inside a `contrasts { }` body, where the expander lowers it to
     Ir.CRunMember; anywhere else resolve_expr / the quantity classifier reject
     it with a located cross-context diagnostic. *)
  | ERunMember of { run : string; ns : run_namespace; member : string; loc : loc }

(* The two symmetric sub-namespaces of a run member: `<run>.quantities.<q>` and
   `<run>.observations.<stream>`. *)
and run_namespace = NsQuantities | NsObservations

type compartment_kind = Integer | Real

(* A parsed `#'` doc-comment block immediately preceding a declaration: free-text
   prose plus optional structured tags. Non-semantic — it does not affect
   compilation — but it is NOT compilation-local: `Expander.build_doc_index`
   folds it into the IR envelope's `docs` dictionary, which `camdl fit summary`
   reads for its parameter legend. It also surfaces in `camdlc inspect` and, via
   `@symbol`, in `camdlc render`. `None` when the declaration is undocumented.

   A doc attaches to a DECLARATION, never to a block keyword — the parser's
   `doc_opt` slot sits on the member rules (`compartment_decl`, `param_decl`,
   `transition_decl`, `obs_decl`, `quantity_decl`, `dim_entry`) and on the
   top-level `let`. *)
type doc = {
  d_text   : string option;   (* joined prose description (the non-@tag lines) *)
  d_symbol : string option;   (* `@symbol`: display label for plots / reports *)
  d_ref    : string option;   (* `@ref`: citation for the definition *)
}

type compartment_decl = { cname: string; ckind: compartment_kind; cdoc: doc option; cloc: loc }

(* Parameter kinds. The two time kinds (PInstant, PDuration) both carry
   dimension [T] for dimcheck (2026-05-22 calendar-time §6.7): an instant is
   an origin-relative point (renders as a date), a duration is a relative
   span (renders as a span, no origin). *)
type param_type = PRate | PProbability | PPositive | PCount | PReal | PInstant | PDuration

(** Explicit dimension annotation: (P exponent, T exponent) *)
type dim_annotation = int * int

(** Prior distribution specification: ~ name(key = val, ...) [| dim_name]
    The optional `| dim_name` clause marks a hierarchical / partially-
    pooled prior (wave 2 / malaria #3). None = plain prior. *)
type prior_spec = {
  ps_name:      string;                    (** distribution name: "log_normal", "beta", etc. *)
  ps_args:      (string * expr) list;      (** keyword arguments *)
  ps_pool_over: string option;              (** `| <dim>` pooling clause *)
}

(* [punit] is the optional tier-3 unit literal that `positive`/`real` kinds
   may carry (gh#60): `tau : positive 'ratio`. The expander folds it into
   [param_dim] via [unit_lit_to_dim], so it is an alternative spelling of the
   [pdim] bracket annotation (the dimension half of the unit; a parameter's
   scale is always the model time unit). A unit on any other kind, or together
   with a bracket [pdim], is a semantic error (E281 / E282). *)
type param_decl =
  | PScalar  of { pname: string; pkind: param_type; pdim: dim_annotation option; punit: unit_lit option; pbounds: (expr * expr) option; pprior: prior_spec option; pdoc: doc option; ploc: loc }
  | PIndexed of { pname: string; pdims: string list; pkind: param_type; pdim: dim_annotation option; punit: unit_lit option; pbounds: (expr * expr) option; pprior: prior_spec option; pdoc: doc option; ploc: loc }

(** Table dimension entry: bare dim name, or dim + unit *)
type table_dim_entry =
  | TDim     of string
  | TDimUnit of string * unit_lit

(** Table value: inline literal or EFuncCall for read_long/external *)
type table_decl = {
  tnames     : string list;           (* one or more names for multi-value columns *)
  tdims      : table_dim_entry list;
  tcell_kind : param_type option;     (* optional cell-type annotation: rate, probability, ... (gh#32) *)
  tvalue     : expr;
  (* Span of the declared axis list (`age × aeg`), or of the name for the
     dimensionless form. gh#490: an undeclared axis is diagnosed here, at the
     declaration, and the caret has to land on the axes rather than on the
     whole declaration — an inline table's value runs to many lines. *)
  tloc       : loc;
}

(** A stoichiometry reference: compartment name + optional indices *)
type stoich_ref = string * index_item list

(** Transition destination form.
    - [DstSum] is the ordinary case: a `+`-separated list of destination
      compartments, each contributing +1 to stoichiometry. Singleton =
      classic, ≥ 2 = multi-dest (wave 1 / malaria #1).
    - [DstBranch] (wave 2 / malaria #2) is a probabilistic branch:
      `X --> { A : w_A, B : w_B } @ rate`. The expander desugars each
      branch into its own IR transition with rate `w_i * rate`. The
      existing chain-binomial source-grouping machinery
      then performs the correct multinomial split at firing time. *)
type destination_form =
  | DstSum    of stoich_ref list
  | DstBranch of (stoich_ref * expr) list

(* A raw, untyped dwell-law call parsed off a `via` clause (staged-residence
   proposal, 2026-06-26 §3). The pair is the law name and its keyword arguments
   exactly as written — e.g. `erlang(stages = 3, mean = 7 'days)` parses to
   `("erlang", [("stages", ...); ("mean", ...)])`. The typed `via_spec`
   (stages/mean/rate extraction + validation) is built in the EXPANDER's Phase-2
   lowering, not here; Phase 1 stores the call verbatim. *)
type via_call = string * (string * expr) list

(* A compile-time positive integer (the `stages` count of a staged residence).
   The only way to build one is [pos_int_of_float], which rejects anything that
   is not a positive whole-number literal — so a `Pos_int.t` in hand is a proof
   that `stages` is a usable stage count, and no downstream code re-checks. This
   is "parse, don't validate" at the via-spec boundary: the smart constructor is
   the single seam where the invariant is established (staged-residence proposal
   §3, "`stages` is a compile-time pos_int"). *)
module Pos_int : sig
  type t
  val of_float : float -> (t, string) result   (* error message on failure *)
  val to_int   : t -> int
end = struct
  type t = int
  let of_float f =
    if Float.is_integer f && f >= 1.0 then Ok (int_of_float f)
    else if not (Float.is_integer f) then
      Error (Printf.sprintf "must be a whole number, got %g" f)
    else
      Error (Printf.sprintf "must be at least 1, got %g" f)
  let to_int t = t
end

(* The mean-or-rate of a staged residence: EXACTLY one is supplied. Encoding the
   XOR as a sum type makes "both" and "neither" unrepresentable in the typed
   spec — the validation that exactly one keyword is present happens once, at the
   point that builds this value. `Mean τ` ⇒ per-stage rate `k/τ`; `Rate ρ` ⇒
   per-stage rate `k·ρ`. Both are expressions and may reference parameters. *)
type mean_spec = Mean of expr | Rate of expr

(* One arm of a `hyper_erlang` finite mixture (staged-residence proposal §4). A
   self-contained record per branch (no fragile parallel lists): its mixture
   [weight], its Erlang [stages]/[mean], an optional per-branch destination
   [to_] (None ⇒ the transition's arrow target), and a [label] (used for the
   flat per-branch stage compartment names `<src>__<label>__i`, distinct across
   branches by construction). The weight is `None` only on the LAST branch ⇒
   `1 − Σ others`, so the mixture is normalized by construction; a non-last
   branch missing a weight, or the last branch carrying one, is an error caught
   when this record is built. *)
type hyper_branch = {
  hb_weight : expr option;          (* None on the LAST branch ⇒ 1 − Σ others *)
  hb_stages : Pos_int.t;
  hb_mean   : mean_spec;
  hb_to     : stoich_ref option;    (* per-branch destination; None ⇒ the transition's TO *)
  hb_label  : string;
}

(* A typed, validated dwell law (the EXPANDER's view of a `via_call`).
   - [Erlang] (Phase 2): a single chain.
   - [HyperErlang] (Phase 4): a finite mixture of Erlang chains, branched at
     entry. Branches have different lengths, so this lowers to FLAT per-branch
     compartments + parallel chains, NOT one stage dimension. *)
type via_spec =
  | Erlang      of { stages : Pos_int.t; mean : mean_spec }
  | HyperErlang of { branches : hyper_branch list }

(* A transition's dynamics: EITHER an ordinary `@ rate` (exponential, the rate
   IS the propensity) OR a `via law(...)` staged residence (the law supplies the
   per-stage rate). Never both, never neither — the `@`-XOR-`via` rule. Encoding
   it as a sum type makes the illegal "both rate and via" / "neither" states
   unrepresentable: every reader pattern-matches and cannot forget the via case. *)
type trans_dynamics =
  | Rate of expr        (* ordinary exponential transition: `@ rate` *)
  | Via  of via_call    (* staged residence: `via law(...)` (lowered in Phase 2) *)

type transition_decl = {
  trname    : string;
  trindices : index_binding list;
  trsrc     : stoich_ref list;
  trdst     : destination_form;
  trdyn     : trans_dynamics;   (* `@ rate` (Rate) XOR `via law(...)` (Via) *)
  trguard   : guard option;
  (* `#[lineage]` attribute (individual-sampling layer, 2026-05-19
     proposal). True ⇒ this transition has parent-child lineage
     semantics: at firing time a parent is sampled from the
     linear-decomposition parent pool and a tracked child is minted
     in the destination. The compiler verifies the rate is
     linear-in-parents (E601) and emits per-pool weight expressions
     into the IR. *)
  trlineage : bool;
  trdoc     : doc option;   (* `#'` doc block (non-semantic; inspect only) *)
  trloc     : loc;
}

(* Sub-expressions a transition's dynamics carry, for variable / let-reference
   analysis that does not care whether the dynamics is a rate or a via law. A
   `Rate` carries its one rate expr; a `Via` carries its law's keyword-argument
   exprs (`stages`, `mean`, `rate`, …). Readers that walk "the exprs of a
   transition" use this instead of reaching into `trdyn`. *)
let trans_dynamics_exprs (d : trans_dynamics) : expr list =
  match d with
  | Rate e        -> [e]
  | Via (_, args) -> List.map snd args

type let_binding = {
  lname    : string;
  lindices : index_binding list;
  lshape   : string list option;  (* Some dims → shaped literal, None → scalar/indexed *)
  lkind    : param_type option;   (* optional type annotation: count, rate, etc. *)
  lbody    : expr;
  (* gh#508: `#'` prose. A derived quantity is where a modelling assumption
     hides — whether `let N[p] = S[p] + E[p] + I[p]` is "total population" or
     "the currently-infectious denominator" changes the force of infection, and
     the expression alone cannot say which. Surfaced by `camdlc inspect`. *)
  ldoc     : doc option;
}

type stratify_decl = {
  sdim  : string;
  sonly : string list option;
}

type likelihood_kind =
  | LikNegBinomial  of (string * expr) list
  | LikPoisson      of (string * expr) list
  | LikNormal       of (string * expr) list
  | LikBinomial     of (string * expr) list
  | LikBetaBinomial of (string * expr) list
  | LikBeta         of (string * expr) list
  | LikBernoulli    of (string * expr) list
  (* Zero-inflated NB. Surface: `zero_inflated(base = neg_binomial(mean=, r=),
     pi = )`, desugared here at parse time to the base's kwargs (`mean`, `r`)
     plus `pi`. *)
  | LikZeroInflatedNegBinomial of (string * expr) list

(* What one `init { }` entry says its compartment starts at.

   [IVExpr e]   — `S = N0 - I`, a value COMPUTED from parameters, constants and
                  other compartments' seeded values.
   [IVLaw (l, loc)] — `I ~ poisson(rate = I0)`, a value DRAWN from [l] once per
                  particle at t_start. The location is the law's own span, so a
                  kind or placement diagnostic points at the distribution rather
                  than at the whole entry.

   One field, not two: an entry is computed or drawn, never both and never
   neither. *)
type init_value =
  | IVExpr of expr
  | IVLaw  of likelihood_kind * loc

(* The keyword arguments of a distribution call, whichever family it is. *)
let likelihood_kwargs (l : likelihood_kind) : (string * expr) list =
  match l with
  | LikNegBinomial a | LikPoisson a | LikNormal a | LikBinomial a
  | LikBetaBinomial a | LikBeta a | LikBernoulli a
  | LikZeroInflatedNegBinomial a -> a

(* The family keyword as written in the model file (`poisson`,
   `neg_binomial`, ...). One definition, so a diagnostic and the IR tag cannot
   disagree about what a distribution is called. *)
let lik_family_name (l : likelihood_kind) : string =
  match l with
  | LikNegBinomial _  -> "neg_binomial"
  | LikPoisson _      -> "poisson"
  | LikNormal _       -> "normal"
  | LikBinomial _     -> "binomial"
  | LikBetaBinomial _ -> "beta_binomial"
  | LikBeta _         -> "beta"
  | LikBernoulli _    -> "bernoulli"
  | LikZeroInflatedNegBinomial _ -> "zero_inflated_neg_binomial"

(* Rewrite every keyword-argument expression, keeping the family. *)
let map_likelihood_kwargs (f : expr -> expr) (l : likelihood_kind) : likelihood_kind =
  let g = List.map (fun (k, e) -> (k, f e)) in
  match l with
  | LikNegBinomial a  -> LikNegBinomial  (g a)
  | LikPoisson a      -> LikPoisson      (g a)
  | LikNormal a       -> LikNormal       (g a)
  | LikBinomial a     -> LikBinomial     (g a)
  | LikBetaBinomial a -> LikBetaBinomial (g a)
  | LikBeta a         -> LikBeta         (g a)
  | LikBernoulli a    -> LikBernoulli    (g a)
  | LikZeroInflatedNegBinomial a -> LikZeroInflatedNegBinomial (g a)

(* Every expression an init entry evaluates — the RHS, or every argument of the
   law. The passes that walk init expressions (index resolution, the Rule-1
   unit walk, the substitution rewrites) all go through this, so a law's
   arguments get exactly the same treatment as a deterministic RHS. *)
let init_value_exprs (v : init_value) : expr list =
  match v with
  | IVExpr e -> [e]
  | IVLaw (l, _) -> List.map snd (likelihood_kwargs l)

let map_init_value (f : expr -> expr) (v : init_value) : init_value =
  match v with
  | IVExpr e -> IVExpr (f e)
  | IVLaw (l, loc) -> IVLaw (map_likelihood_kwargs f l, loc)

type init_entry = {
  icomp     : string;
  iindices  : index_item list;       (* positional: S[child] *)
  ibindings : index_binding list;    (* loop: [p in patch] *)
  ivalue    : init_value;
  iloc      : loc;
}

(* Shared "specified times" schedule core: `every = E` (regular cadence) or
   `at = [...]` (explicit times). Reused across observation / output (and the
   `at` arm of interventions); each surface lowers it to its own IR variant. *)
type schedule_core =
  | SchedEvery of expr        (* every = E  *)
  | SchedAt    of expr list   (* at = [...] *)

type obs_projection =
  | ProjIncidence  of string * index_item list
  | ProjPrevalence of string * index_item list
  | ProjDerived    of expr

(* A measurement-model statement: `<scored_col> ~ Dist(kw = ..., ...)`.
   The left side is a declared value column (the scored outcome); the right
   side is the distribution family + its keyword args. Distinct from the
   prior `~` (no `| dim` pooling suffix — meaningless on a likelihood). *)
type obs_measurement = {
  om_scored : string;          (* the `~` LHS: a declared value column *)
  om_lik    : likelihood_kind; (* the `~` RHS distribution *)
}

(* The role of a declared file column (the `columns { name : role }` block,
   2026-06-10 observation data-entry §2.2):
   - [ColTime]  — the time axis (exactly one per stream); the FIT time source.
   - [ColDim d] — a model dimension `d`; values bind to that dimension's levels.
   - [ColValue k] — an observed value of type `k` (count/real/probability/…);
     either the `~` LHS (scored) or RHS-referenced auxiliary data. *)
type obs_col_role =
  | ColTime
  | ColDim   of string
  | ColValue of param_type

(* A declared file column: header name + role. *)
type obs_column = {
  oc_name : string;
  oc_role : obs_col_role;
}

type obs_decl = {
  oname       : string;
  oindices    : index_binding list;
  (* `from <label>` — the data SOURCE the stream reads from (§2.4). The
     source label is the data key (`--data label=file`); None defaults the
     key to the stream name. *)
  osource     : string option;
  (* `columns { name : role }` — the full, explicit file schema (§2.2);
     always required. None lets the expander emit the missing-field
     diagnostic. *)
  ocolumns    : obs_column list option;
  (* m12 in 2026-04-19 review: each of measurement/projection is mandatory;
     an empty `cases {}` block previously defaulted to Poisson(rate=1) on an
     incidence projection, silently producing a meaningless but compile-green
     likelihood. Represented as option here so the expander can emit a
     specific diagnostic naming the missing field. *)
  omeasurement : obs_measurement option;
  oprojection  : obs_projection option;
  (* `emit_schedule` (§2.5): the SIMULATE-only emission cadence — when
     `simulate --obs` writes synthetic rows. Optional and never consulted by
     the fit path (the data file's `time` column drives there). *)
  oschedule   : schedule_core option;
  odoc        : doc option;   (* `#'` doc block (non-semantic; inspect only) *)
  oloc        : loc;
}

type action_decl =
  | ATransfer of (string * expr) list      (* kwargs: fraction=, count=, from=, to= *)
  | ASet      of string * index_item list * expr
  | AAdd      of string * index_item list * expr   (* compartment, indices, count expr *)

type schedule_decl =
  | SAtTimes of expr list
  (** Recurring schedule: (every, from?, to?).
      from defaults to simulate.from if None; to defaults to simulate.to. *)
  | SRecurring of expr * expr option * expr option
  | SEveryAtDay of expr * expr          (* period, at_day *)

type intervention_decl = {
  ivname    : string;
  ivindices : index_binding list;   (* [] for non-indexed interventions *)
  ivaction  : action_decl list;     (* one or more; block-form `set` keeps all *)
  ivschedule: schedule_decl;
  ivguard   : guard option;         (* where expr — compile-time filter *)
  ivloc     : loc;
}

(* gh#204 reactive trigger predicate (pre-expansion). A dedicated predicate type
   (not the general expr) because the general expr grammar has comparisons but
   not and/or/not. The leaf [TgAtom] carries a comparison expr (e.g.
   `observed(weekly_afp) >= afp_trigger_threshold`); the expander destructures
   it — recognising observed()/sum_observed() on one side, a static threshold on
   the other — and lowers to Ir.trigger_expr. Keeping the leaf a plain expr
   avoids a grammar conflict with the expr-level comparison productions. *)
type trig_pred =
  | TgAtom of expr
  | TgAnd  of trig_pred * trig_pred
  | TgOr   of trig_pred * trig_pred
  | TgNot  of trig_pred

type reactive_decl = {
  rxname     : string;
  rxindices  : index_binding list;   (* [] for non-indexed policies *)
  rxwhen     : trig_pred;
  rxafter    : expr option;          (* lag before the effect; default 0 *)
  rxonce     : expr option;          (* fire-and-disable; default true *)
  rxcooldown : expr option;          (* min time between firings *)
  rxaction   : action_decl;
  rxguard    : guard option;         (* where expr — compile-time filter *)
  rxloc      : loc;
}

type ode_decl = { ocomp: string; oderiv: expr }

(* A generated-quantity declaration (proposal 2026-06-25): `name [idx]? = body`.
   The body is a plain [expr]; the expander's quantity classifier decides
   whether it is a temporal reduction, a series, or reduction arithmetic over
   earlier scalar quantities (it is not resolved as an ordinary rate expr). *)
type quantity_decl = {
  qd_name    : string;
  qd_indices : index_binding list;   (* [] for an unstratified quantity *)
  qd_body    : expr;
  qd_doc     : doc option;   (* `#'` doc block (non-semantic; inspect + doc_index) *)
  qd_loc     : loc;
}

(* A counterfactual-contrast declaration (proposal 2026-06-25):
   `name = <body>`. The body is arithmetic (reusing [EBinOp]) over run-rooted
   [ERunMember] operands. There is no window — the fork is derived in the reducer
   (the last saved snapshot before the toggled intervention's fire time) and the
   result is shaped over `[fork, run-end]`. Resolved + lowered to Ir.contrast by
   the expander. *)
type contrast_decl = {
  cd_name   : string;
  cd_body   : expr;            (* arithmetic over ERunMember; reuses EBinOp *)
  cd_doc    : doc option;
  cd_loc    : loc;
}

(* A forcing-block keyword-argument value (gh#423). The string-vs-expr
   distinction is carried in the TYPE, confined to the forcing surface — the
   global [expr] AST is unchanged. A quoted STRING names the outside world (a
   data-file path or a file column) and parses to [FStr]; a bare expression
   names something inside the model (a param, a dimension, a table, or a closed
   enum like `method = linear`) and parses to [FExpr]. Consumers require the
   arm that matches the kwarg's role, so `value_col = C` (bare) and
   `table = "T"` (quoted) fail with a signposted diagnostic instead of being
   silently indistinguishable via a `dummy_loc`. Reader's rule: quoted =
   outside the model (file), bare = inside the model or a closed enum. *)
type farg_value =
  | FStr  of string   (* a quoted string: a file path or a file column name *)
  | FExpr of expr     (* a bare expression: model value, model name, or enum *)

(* The bare-expression view of a forcing arg, or [None] for a [FStr]. Readers
   that walk "the exprs of a forcing" (unit / date analysis) use this so a
   string literal — which carries no unit or date content — is skipped. *)
let farg_expr_opt = function FExpr e -> Some e | FStr _ -> None

type func_decl = {
  fname    : string;
  findices : index_binding list;
  fkind    : string;
  (* Required tier-3 unit literal (GH #8): annotates the
     scale/dimension of values produced by this forcing function.
     E.g. `pop : interpolated 'count`, `birthrate : interpolated
     'per_year`, `seasonal : sinusoidal 'ratio`. The dim-checker
     uses this authoritatively — no value-based inference fallback. *)
  funit    : unit_lit;
  fargs    : (string * farg_value) list;
}

type output_traj_decl = {
  otschedule  : schedule_core;
  otformat    : string;
}

type output_decl = {
  out_trajectories: output_traj_decl option;
}

(* gh#166: sim_integrator ("rk4"|"rk45", None -> default rk4); sim_atol/sim_rtol
   are the dimensionless adaptive tolerances (rk45 only). Each carries its source
   span so the expander's semantic diagnostics (unknown method, rk4-takes-no-
   tolerances, dimensioned tolerance) point at the offending token. *)
type simulate_decl = {
  sim_from: expr;
  sim_to: expr;
  sim_dt: expr option;
  sim_integrator: (string * loc) option;
  sim_atol: (expr * loc) option;
  sim_rtol: (expr * loc) option;
}

type timepoint_decl = { tpname: string; tptime: expr }

type scenario_field =
  | ScLabel   of string
  | ScEnable  of string list
  | ScDisable of string list
  | ScSet     of (string * expr) list
  | ScScale   of (string * expr) list
  | ScCompose of string list
  | ScTEnd    of expr
  | ScExtends of string     (** `extends = parent_name` — single-inheritance sugar *)

type scenario_decl = {
  scname   : string;
  scfields : scenario_field list;
}

(** Source of dimension levels: inline list or read from a file column *)
type dim_source =
  | DInline of string list
  (* fn_name: what the user wrote before `(` (expected "read").
     col_kw:  keyword for the column arg (expected "column").
     path:    file path string.
     col:     column name string.
     The parser accepts any `ident(STRING, ident = STRING)` and
     defers the "is it actually `read(…, column = …)`?" check to the
     expander, where a proper E2xx diagnostic can fire — M11 in the
     2026-04-19 review. *)
  | DRead   of { fn_name: string; path: string; col_kw: string; col: string }

type dimensions_entry = {
  dename : string;
  desrc  : dim_source;
  dedoc  : doc option;   (* `#'` doc block (non-semantic; inspect only) *)
}

type balance_decl = { bcomp: string; bexpr: expr }

type declaration =
  | DTimeUnit    of unit_lit * loc
  | DDescription of string
  (* The file-header `#'` block, which documents the MODEL rather than any one
     declaration (gh#750): what it is, what it is fitted to, what it branches
     from. Produced ONLY by the `file` rule, from the MODEL_DOC tokens the lexer
     emits for `#'` lines that precede every declaration — so it can only ever
     sit at the head of the list. Like every other doc it is non-semantic and
     rides the IR envelope's `docs` dictionary, outside the hashed model. *)
  | DModelDoc    of doc
  | DOrigin      of string
  | DDimensions  of dimensions_entry list
  | DCompartments of compartment_decl list
  | DParameters   of param_decl list
  | DTables       of table_decl list
  | DForcing      of func_decl list
  | DTransitions  of transition_decl list
  | DObservations of obs_decl list
  | DInterventions of intervention_decl list
  | DEvents        of intervention_decl list
  | DReactiveInterventions of reactive_decl list   (* gh#204 *)
  | DODE          of ode_decl list
  | DOutput       of output_decl
  | DSimulate     of simulate_decl
  | DInit         of init_entry list
  | DTimepoints   of timepoint_decl list
  | DStratify     of stratify_decl
  | DLet          of let_binding
  | DScenarios    of scenario_decl list
  | DBalance      of balance_decl
  | DQuantities   of quantity_decl list   (* proposal 2026-06-25 *)
  | DContrasts    of contrast_decl list   (* counterfactual contrasts, proposal 2026-06-25 *)

(* ── Cross-file locs ─────────────────────────────────────────────────────────
   A `quantities { }` block can arrive from a SEPARATE compilation unit
   (`camdlc MODEL.camdl --quantities VOCAB.camdl`), and every diagnostic about
   it — an undeclared name, a bad reduction — must name that file, not the
   model. The parser leaves [loc.file] empty and the expander fills it from the
   context's single filename, so the decls parsed out of the vocabulary file are
   stamped with their own path here, before they are spliced into the model's
   declaration list. Only empty [file] fields are filled: a loc that already
   names a file keeps it. *)
let stamp_loc_file ~file (l : loc) : loc =
  if l.file = "" then { l with file } else l

let rec stamp_expr_file ~file (e : expr) : expr =
  let go = stamp_expr_file ~file in
  match e with
  | EConst _ | EUnit _ -> e
  | EIdent (n, l) -> EIdent (n, stamp_loc_file ~file l)
  | EIndex (n, items, l) ->
    EIndex (n, List.map (stamp_index_item_file ~file) items, stamp_loc_file ~file l)
  | EBinOp (op, a, b) -> EBinOp (op, go a, go b)
  | EUnOp (op, a) -> EUnOp (op, go a)
  | ESum (v, d, g, body, l) -> ESum (v, d, g, go body, stamp_loc_file ~file l)
  | ECond (p, a, b) -> ECond (go p, go a, go b)
  | EFuncCall (f, kvs) -> EFuncCall (f, List.map (fun (k, v) -> (k, go v)) kvs)
  | EList xs -> EList (List.map go xs)
  | ERange (a, b) -> ERange (go a, go b)
  | EObsAccess (s, l) -> EObsAccess (s, stamp_loc_file ~file l)
  | ERunMember r -> ERunMember { r with loc = stamp_loc_file ~file r.loc }

and stamp_index_item_file ~file = function
  | IPosn e -> IPosn (stamp_expr_file ~file e)
  | INamed (n, e) -> INamed (n, stamp_expr_file ~file e)

let stamp_quantity_decl_file ~file (qd : quantity_decl) : quantity_decl =
  { qd with qd_body = stamp_expr_file ~file qd.qd_body;
            qd_loc  = stamp_loc_file ~file qd.qd_loc }
