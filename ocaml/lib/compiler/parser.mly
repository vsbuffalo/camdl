%{
  open Ast

  let extract_ident_list = function
    | EList items -> List.filter_map (function EIdent (n, _) -> Some n | _ -> None) items
    | _ -> []

  (* Assemble an [obs_decl] from the header (name, indices, source) and the
     block's key/value entries. The entries are the polymorphic variants
     produced by [obs_kv]; we fold them into the record fields. A bare
     `likelihood = D(...)` on the (rejected) migration path arrives as a
     `Lik`; we wrap it into a measurement with an empty scored column so the
     parse completes — the E273 diagnostic already fired. *)
  (* Desugar a distribution call `D(kw = ..., ...)` into a [likelihood_kind].
     Shared by the `~` measurement form and the (rejected) migration
     `likelihood = D(...)` path. `diagnostic_test(...)` is compile-time sugar
     that reparameterizes a binomial/bernoulli `p` by sensitivity/specificity. *)
  let lik_of_funcall kind args ~sp ~ep =
    match kind with
    | "neg_binomial"  -> LikNegBinomial  args
    | "poisson"       -> LikPoisson      args
    | "normal"        -> LikNormal       args
    | "binomial"      -> LikBinomial     args
    | "beta_binomial" -> LikBetaBinomial args
    | "bernoulli"     -> LikBernoulli    args
    | "diagnostic_test" ->
      let find k = List.assoc_opt k args in
      (match find "base", find "sens", find "spec" with
       | Some (EFuncCall (base_kind, base_args)), Some sens_e, Some spec_e ->
         let one_minus e = EBinOp (Sub, EConst 1.0, e) in
         let rewrite_p =
           List.map (fun (k, v) ->
             if k = "p" then
               let p_adj =
                 EBinOp (Add,
                   EBinOp (Mul, sens_e, v),
                   EBinOp (Mul, one_minus spec_e, one_minus v))
               in (k, p_adj)
             else (k, v))
         in
         (match base_kind with
          | "binomial"  -> LikBinomial  (rewrite_p base_args)
          | "bernoulli" -> LikBernoulli (rewrite_p base_args)
          | other ->
            Parser_errors.push_error ~sp ~ep
              ~code:"E253"
              ~msg:(Printf.sprintf
                "diagnostic_test base must be binomial(...) or bernoulli(...); got %s(...)"
                other);
            LikBinomial [])
       | _ ->
         Parser_errors.push_error ~sp ~ep
           ~code:"E254"
           ~msg:"diagnostic_test requires keyword args base = <binomial|bernoulli>(...), sens = <expr>, spec = <expr>";
         LikBinomial [])
    | s ->
      Parser_errors.push_error ~sp ~ep
        ~code:"E104"
        ~msg:(Printf.sprintf "unknown likelihood '%s': expected one of neg_binomial, poisson, normal, binomial, beta_binomial, bernoulli, diagnostic_test" s);
      LikPoisson args

  let build_obs_decl name ibs src kvs ~sp ~ep =
    let cols  = ref None in
    let sched = ref None in
    let proj  = ref None in
    let meas  = ref None in
    List.iter (function
      | `Columns c     -> cols  := Some c
      | `Schedule s    -> sched := Some s
      | `Proj p        -> proj  := Some p
      | `Measurement m -> meas  := Some m
      | `Lik l         -> meas  := Some { om_scored = ""; om_lik = l }
    ) kvs;
    { oname = name; oindices = ibs;
      osource = src; ocolumns = !cols;
      omeasurement = !meas; oprojection = !proj;
      oschedule = !sched;
      oloc = Parser_errors.ast_loc_of ~sp ~ep }
%}

(* ── Literals & identifiers ────────────────────────────────────────────── *)
%token <string> IDENT
%token <int>    INT
%token <float>  FLOAT
%token <string> STRING
%token <string> UNIT_IDENT   (* 'days, 'per_day, etc. *)

(* ── Punctuation ────────────────────────────────────────────────────────── *)
%token ARROW       (* --> *)
%token AT          (* @ *)
%token TILDE       (* ~ *)
%token EQ          (* = *)
%token COLON       (* : *)
%token COMMA       (* , *)
%token LBRACE RBRACE
%token LBRACKET RBRACKET
%token LPAREN RPAREN
%token PLUS MINUS STAR SLASH CARET
%token EQ2         (* == *)
%token NEQ         (* != *)
%token LT GT LE GE
%token CROSS       (* × *)
%token HASH_LBRACKET (* #[ — attribute opener *)

(* ── Keywords ───────────────────────────────────────────────────────────── *)
%token TIME_UNIT COMPARTMENTS PARAMETERS TABLES FORCING
%token TRANSITIONS OBSERVATIONS INTERVENTIONS ODE OUTPUT SIMULATE
%token INIT TIMEPOINTS SCENARIOS EXTENDS STRATIFY LET FROM TO WHERE SUM
%token CONSECUTIVE IN BY DIMENSIONS ONLY REAL INTEGER RATE PROBABILITY POSITIVE COUNT
%token INSTANT DURATION
%token AND OR NOT IF THEN ELSE EVERY UNTIL AT_KW FORMAT DESCRIPTION NULL TRANSFER LIKELIHOOD ORIGIN BALANCE EVENTS ADD AT_DAY
%token COLUMNS EMIT_SCHEDULE
%token REACTIVE_INTERVENTIONS WHEN ACTION   (* gh#204 *)
%token PIPE

%token EOF

(* ── Precedences (lowest → highest) ────────────────────────────────────── *)
%nonassoc ELSE
%left  OR
%left  AND
%nonassoc EQ2 NEQ LT GT LE GE
%left  PLUS MINUS
%left  STAR SLASH CROSS
%right CARET
%nonassoc UMINUS

%start <Ast.declaration list> file

%%

(* ── Top-level ──────────────────────────────────────────────────────────── *)

file:
  | ds = declaration* EOF { ds }

declaration:
  | TIME_UNIT EQ u = unit_lit
      { DTimeUnit u }
  | DESCRIPTION EQ s = STRING
      { DDescription s }
  | ORIGIN EQ e = expr
      { match e with
        | EFuncCall ("date", [("", EIdent (s, _))]) -> DOrigin s
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E101"
            ~msg:"invalid origin declaration: expected origin = date(\"YYYY-MM-DD\")";
          DOrigin "" }
  | DIMENSIONS LBRACE es = list(dim_entry) RBRACE
      { DDimensions es }
  | COMPARTMENTS LBRACE cs = compartment_list RBRACE
      { DCompartments cs }
  | PARAMETERS LBRACE ps = param_list RBRACE
      { DParameters ps }
  | TABLES LBRACE ts = table_list RBRACE
      { DTables ts }
  | FORCING LBRACE fs = func_list RBRACE
      { DForcing fs }
  | TRANSITIONS LBRACE trs = transition_list RBRACE
      { DTransitions trs }
  | OBSERVATIONS LBRACE obs = obs_list RBRACE
      { DObservations obs }
  | INTERVENTIONS LBRACE ivs = intervention_list RBRACE
      { DInterventions ivs }
  | EVENTS LBRACE evs = intervention_list RBRACE
      { DEvents evs }
  | REACTIVE_INTERVENTIONS LBRACE rxs = list(reactive_decl) RBRACE
      { DReactiveInterventions rxs }
  | ODE LBRACE odes = ode_list RBRACE
      { DODE odes }
  | OUTPUT LBRACE od = output_body RBRACE
      { DOutput od }
  | SIMULATE LBRACE sd = simulate_body RBRACE
      { DSimulate sd }
  | INIT LBRACE ies = init_list RBRACE
      { DInit ies }
  | TIMEPOINTS LBRACE tps = timepoint_list RBRACE
      { DTimepoints tps }
  | STRATIFY LPAREN sa = stratify_args RPAREN
      { DStratify sa }
  | LET name = IDENT ibs = index_bindings_opt COLON pk = param_kind EQ body = expr
      { DLet { lname = name; lindices = ibs; lshape = None; lkind = Some pk; lbody = body } }
  | LET name = IDENT ibs = index_bindings_opt shape = let_shape_opt EQ body = expr
      { DLet { lname = name; lindices = ibs; lshape = shape; lkind = None; lbody = body } }
  | SCENARIOS LBRACE ss = list(scenario_block) RBRACE
      { DScenarios ss }
  | BALANCE LBRACE target = IDENT EQ e = expr RBRACE
      { DBalance { bcomp = target; bexpr = e } }

(* ── Unit literals ──────────────────────────────────────────────────────── *)

unit_lit:
  | u = UNIT_IDENT { match u with
    | "days"      -> Days
    | "weeks"     -> Weeks
    | "months"    -> Months
    | "years"     -> Years
    | "per_day"   -> PerDay
    | "per_week"  -> PerWeek
    | "per_month" -> PerMonth
    | "per_year"  -> PerYear
    | "count"     -> Count
    | "ratio"     -> Ratio
    | s ->
      Parser_errors.push_error ~sp:$startpos ~ep:$endpos
        ~code:"E102"
        ~msg:(Printf.sprintf "unknown unit '%s': expected one of 'days, 'weeks, 'months, 'years, 'per_day, 'per_week, 'per_month, 'per_year, 'count, 'ratio" s);
      Days }

(* ── Compartment block ──────────────────────────────────────────────────── *)

compartment_list:
  | cs = separated_list(COMMA, compartment_decl) { cs }

compartment_decl:
  | name = IDENT kind = compartment_kind_opt
      { { cname = name; ckind = kind;
          cloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

compartment_kind_opt:
  | (* empty *)  { Integer }
  | COLON REAL   { Real }
  | COLON INTEGER { Integer }

(* ── Parameter block ────────────────────────────────────────────────────── *)

param_list:
  | ps = list(param_decl) { ps }

param_decl:
  (* scalar, no bounds, no prior *)
  | name = IDENT COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt
      { PScalar { pname = name; pkind = pk; pdim = da; punit = pu; pbounds = None; pprior = None;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, no bounds, with prior *)
  | name = IDENT COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt TILDE pr = prior_clause
      { PScalar { pname = name; pkind = pk; pdim = da; punit = pu; pbounds = None; pprior = Some pr;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, with bounds, no prior *)
  | name = IDENT COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET
      { PScalar { pname = name; pkind = pk; pdim = da; punit = pu; pbounds = Some (lo, hi); pprior = None;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, with bounds, with prior *)
  | name = IDENT COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET TILDE pr = prior_clause
      { PScalar { pname = name; pkind = pk; pdim = da; punit = pu; pbounds = Some (lo, hi); pprior = Some pr;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, no bounds, no prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; punit = pu; pbounds = None; pprior = None;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, no bounds, with prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt TILDE pr = prior_clause
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; punit = pu; pbounds = None; pprior = Some pr;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, with bounds, no prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; punit = pu; pbounds = Some (lo, hi); pprior = None;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, with bounds, with prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind pu = param_unit_opt da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET TILDE pr = prior_clause
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; punit = pu; pbounds = Some (lo, hi); pprior = Some pr;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

prior_clause:
  (* plain prior: ~ normal(mu = 0, sigma = 1) *)
  | name = prior_name LPAREN args = separated_list(COMMA, prior_kwarg) RPAREN
      { { ps_name = name; ps_args = args; ps_pool_over = None } }
  (* hierarchical / pooled prior: ~ log_normal(mu = mu_h, sigma = sigma_h) | age *)
  | name = prior_name LPAREN args = separated_list(COMMA, prior_kwarg) RPAREN PIPE dim = IDENT
      { { ps_name = name; ps_args = args; ps_pool_over = Some dim } }

(* Distribution names and keyword argument names accept identifiers AND
   common keywords (rate, count, etc.) that conflict with DSL reserved
   words but are natural in statistical contexts. *)
prior_name:
  | k = kw_arg_name { k }

prior_kwarg:
  | k = kw_arg_name EQ v = expr { (k, v) }

dim_annotation_opt:
  | (* empty *) { None }
  | LBRACKET da = dim_literal RBRACKET { Some da }

(* Optional tier-3 unit literal on a parameter kind (gh#60):
   `tau : positive 'ratio`. Syntactically accepted after any kind; the
   expander restricts it to `positive`/`real` (E281) and rejects a clash with
   a `[dim]` bracket (E282). *)
param_unit_opt:
  | (* empty *) { None }
  | u = unit_lit { Some u }

dim_literal:
  (* [1] — dimensionless *)
  | n = INT
      { if n = 1 then (0, 0)
        else begin
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E103"
            ~msg:(Printf.sprintf "unknown dimension '[%d]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" n);
          (0, 0)
        end }
  (* [P] — population *)
  | id = IDENT { match id with
      | "P" -> (1, 0)
      | "T" -> (0, 1)
      | _ ->
        Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E103"
          ~msg:(Printf.sprintf "unknown dimension '[%s]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" id);
        (0, 0) }
  (* [T^-1] — per-capita rate *)
  | id = IDENT CARET MINUS m = INT
      { match id with
      | "P" -> (- m, 0)
      | "T" -> (0, - m)
      | _ ->
        Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E103"
          ~msg:(Printf.sprintf "unknown dimension '[%s^-%d]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" id m);
        (0, 0) }
  (* [P*T^-1] — population-level rate *)
  | id1 = IDENT STAR id2 = IDENT CARET MINUS m = INT
      { match (id1, id2) with
      | ("P", "T") -> (1, - m)
      | ("T", "P") -> (- m, 1)
      | _ ->
        Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E103"
          ~msg:(Printf.sprintf "unknown dimension '[%s*%s^-%d]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" id1 id2 m);
        (0, 0) }
  (* [P/T] — population-level rate (alternative syntax) *)
  | id1 = IDENT SLASH id2 = IDENT
      { match (id1, id2) with
      | ("P", "T") -> (1, -1)
      | ("T", "P") -> (-1, 1)
      | ("P", "P") -> (0, 0)
      | ("T", "T") -> (0, 0)
      | _ ->
        Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E103"
          ~msg:(Printf.sprintf "unknown dimension '[%s/%s]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" id1 id2);
        (0, 0) }
  (* [1/T] — per-capita rate (alternative syntax) *)
  | n = INT SLASH id = IDENT
      { if n = 1 then
        match id with
        | "P" -> (-1, 0)
        | "T" -> (0, -1)
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E103"
            ~msg:(Printf.sprintf "unknown dimension '[1/%s]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" id);
          (0, 0)
      else begin
        Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E103"
          ~msg:(Printf.sprintf "unknown dimension '[%d/%s]' — expected one of: [1], [P], [T], [T^-1], [1/T], [P/T], [P*T^-1]" n id);
        (0, 0)
      end }

param_kind:
  | RATE        { PRate }
  | PROBABILITY { PProbability }
  | POSITIVE    { PPositive }
  | COUNT       { PCount }
  | REAL        { PReal }
  | INSTANT     { PInstant }
  | DURATION    { PDuration }

(* ── Table block ────────────────────────────────────────────────────────── *)

table_list:
  | ts = list(table_decl) { ts }

table_decl:
  | names = separated_nonempty_list(COMMA, IDENT) COLON dims = table_dims_nonempty COLON kind = param_kind EQ v = expr
      { { tnames = names; tdims = dims; tcell_kind = Some kind; tvalue = v } }
  | names = separated_nonempty_list(COMMA, IDENT) COLON dims = table_dims_nonempty EQ v = expr
      { { tnames = names; tdims = dims; tcell_kind = None; tvalue = v } }
  | name = IDENT EQ v = expr
      { { tnames = [name]; tdims = []; tcell_kind = None; tvalue = v } }

table_dims_nonempty:
  | ds = separated_nonempty_list(CROSS, table_dim_entry) { ds }

table_dim_entry:
  | name = IDENT { TDim name }
  | name = IDENT u = unit_lit { TDimUnit (name, u) }

(* ── Function block ─────────────────────────────────────────────────────── *)

func_list:
  | fs = list(func_decl) { fs }

func_decl:
  (* Required tier-3 unit literal between kind and block — e.g.
     `pop : interpolated 'count { … }`, `birthrate : interpolated 'per_year { … }`,
     `seasonal : sinusoidal 'ratio { … }`. Parallels `tables { t :
     dim 'unit = ... }`. The dim-checker uses the declared dim
     authoritatively; no value-based inference fallback (GH #8). *)
  | name = IDENT ibs = index_bindings_opt COLON kind = IDENT u = unit_lit LBRACE args = func_args RBRACE
      { { fname = name; findices = ibs; fkind = kind; funit = u; fargs = args } }

func_args:
  | kvs = list(func_arg) { kvs }

func_arg:
  | k = IDENT EQ v = expr { (k, v) }

(* ── Transitions block ──────────────────────────────────────────────────── *)

transition_list:
  | trs = list(transition_decl) { trs }

transition_decl:
  (* inline: [#[lineage]] name[...] : srcs --> dsts @ rate where guard.
     The optional `#[lineage]` attribute may sit on its own line above
     the transition or inline immediately before it — camdl has no
     statement separators, so both forms are the same production and
     produce identical IR. *)
  | lin = lineage_attr_opt name = IDENT ibs = index_bindings_opt COLON srcs = stoich_ref_list ARROW dsts = stoich_ref_list AT rate = expr guard = where_clause_opt
      { { trname = name; trindices = ibs;
          trsrc = srcs; trdst = DstSum dsts;
          trrate = rate; trguard = guard; trlineage = lin;
          trloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* block form: [#[lineage]] name[...] : srcs --> dsts { rate = ...; where ... } *)
  | lin = lineage_attr_opt name = IDENT ibs = index_bindings_opt COLON srcs = stoich_ref_list ARROW dsts = stoich_ref_list LBRACE tbody = transition_body RBRACE
      { let (rate_opt, guard) = tbody in
        (* A block-form transition with no `rate = …` (and no `@ …`) is a
           hard error, not a silent zero-rate transition. Pushing a
           diagnostic and substituting a placeholder rate lets parsing
           continue so the user sees all errors at once. *)
        let rate = match rate_opt with
          | Some e -> e
          | None ->
            Parser_errors.push_error_hint ~sp:$startpos(name) ~ep:$endpos(name)
              ~code:"E213"
              ~msg:(Printf.sprintf
                "transition '%s' is missing a rate" name)
              ~hint:"add `rate = <expr>` inside the block, or use the \
                     inline form `... --> ... @ <expr>`";
            EConst 0.0
        in
        { trname = name; trindices = ibs;
          trsrc = srcs; trdst = DstSum dsts;
          trrate = rate; trguard = guard; trlineage = lin;
          trloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* branching: [#[lineage]] name[...] : srcs --> { D1 : w1, ... } @ rate where guard *)
  | lin = lineage_attr_opt name = IDENT ibs = index_bindings_opt COLON srcs = stoich_ref_list ARROW LBRACE branches = separated_nonempty_list(COMMA, branch_entry) RBRACE AT rate = expr guard = where_clause_opt
      { { trname = name; trindices = ibs;
          trsrc = srcs; trdst = DstBranch branches;
          trrate = rate; trguard = guard; trlineage = lin;
          trloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

(* Optional transition attribute. Only `#[lineage]` is recognized in
   v1. An unknown attribute name (e.g. `#[transmission]`) is a hard
   error (E110) rather than a silent no-op — "no loose semantics":
   if a construct looks like it means something, it must mean exactly
   that or produce a clear error. *)
lineage_attr_opt:
  | (* empty *) { false }
  | HASH_LBRACKET name = IDENT RBRACKET
      { if name = "lineage" then true
        else begin
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E110"
            ~msg:(Printf.sprintf
              "unknown transition attribute '#[%s]': the only attribute \
               supported in v1 is '#[lineage]'" name);
          false
        end }

stoich_ref_list:
  | (* empty *)                                           { [] }
  | items = separated_nonempty_list(PLUS, stoich_ref_item) { items }

stoich_ref_item:
  | name = IDENT idxs = index_items_opt { (name, idxs) }

branch_entry:
  | dst = stoich_ref_item COLON weight = expr { (dst, weight) }

index_items_opt:
  | (* empty *) { [] }
  | LBRACKET items = separated_list(COMMA, index_item) RBRACKET { items }

index_item:
  | e = expr { IPosn e }
  | name = IDENT EQ e = expr { INamed (name, e) }

where_clause_opt:
  | (* empty *) { None }
  | WHERE g = guard_expr { Some g }

let_shape_opt:
  | (* empty *) { None }
  | COLON ds = separated_nonempty_list(CROSS, IDENT) { Some ds }

transition_body:
  | kvs = list(transition_body_entry)
      { (* `rate` is `expr option`: `None` means no `rate = …` entry was
           given. The block-form production (above) turns that `None` into
           a hard E213 diagnostic — a missing rate must NOT silently
           default to a zero-rate (never-firing) transition. *)
        let rate  = ref None in
        let guard = ref None in
        List.iter (function
          | `Rate e  -> rate := Some e
          | `Guard g -> guard := Some g
        ) kvs;
        (!rate, !guard) }

transition_body_entry:
  | RATE EQ e = expr { `Rate e }
  | WHERE g = guard_expr { `Guard g }

guard_expr:
  | g = guard_atom { g }
  | g1 = guard_expr AND g2 = guard_expr { GAnd (g1, g2) }
  | g1 = guard_expr OR  g2 = guard_expr { GOr  (g1, g2) }

guard_atom:
  | a = IDENT EQ2 b = IDENT { GEq  (a, b) }
  | a = IDENT NEQ  b = IDENT { GNeq (a, b) }
  | t = IDENT LBRACKET idx = separated_nonempty_list(COMMA, IDENT) RBRACKET op = relop v = guard_operand
      { GTab (t, idx, op, v) }
  | LPAREN g = guard_expr RPAREN { g }

relop:
  | LT  { RLt }
  | LE  { RLe }
  | GT  { RGt }
  | GE  { RGe }
  | EQ2 { REq }
  | NEQ { RNe }

guard_operand:
  | i = INT   { GoNum (float_of_int i) }
  | f = FLOAT { GoNum f }
  | n = IDENT { GoName n }

(* ── Index bindings ─────────────────────────────────────────────────────── *)

index_bindings_opt:
  | (* empty *) { [] }
  | LBRACKET ibs = separated_list(COMMA, index_binding) RBRACKET { ibs }

index_binding:
  | v = IDENT IN d = IDENT { IBind (v, d) }
  | v = IDENT IN COMPARTMENTS { IComp v }
  | LPAREN v = IDENT COMMA vn = IDENT RPAREN IN CONSECUTIVE LPAREN d = IDENT RPAREN
      { IConsec (v, vn, d) }

(* ── Observations block ─────────────────────────────────────────────────── *)

obs_list:
  | obs = list(obs_decl) { obs }

(* Header: `name [p in dim] (from <source>)? { ... }` — NO colon (§2.2/§2.4).
   The old `name : { ... }` form is rejected by [obs_decl_colon] below with a
   migration diagnostic. *)
obs_decl:
  | name = IDENT ibs = index_bindings_opt src = obs_source_opt LBRACE obs_kvs = list(obs_kv) RBRACE
      { build_obs_decl name ibs src obs_kvs ~sp:$startpos ~ep:$endpos }
  (* Migration: the stream header colon was dropped (2026-06-10 §9). Reject
     `name : { ... }` with a diagnostic that names the rewrite, not a bare
     E001. *)
  | name = IDENT ibs = index_bindings_opt src = obs_source_opt COLON LBRACE obs_kvs = list(obs_kv) RBRACE
      { Parser_errors.push_error_hint ~sp:$startpos ~ep:$endpos
          ~code:"E270"
          ~msg:(Printf.sprintf
            "observation '%s': the stream-header colon was removed" name)
          ~hint:(Printf.sprintf
            "write `%s { ... }` (no colon) — see `camdl docs language-changes`" name);
        build_obs_decl name ibs src obs_kvs ~sp:$startpos ~ep:$endpos }

obs_source_opt:
  | (* empty *)        { None }
  | FROM src = IDENT   { Some src }

obs_kv:
  (* `columns { name : role }` — the explicit file schema (§2.2). Entries may
     be comma-separated (`{ time : time, cases : count }`) or newline-separated
     (one per line) — the comma after each entry is optional, matching the rest
     of camdl's block style. *)
  | COLUMNS LBRACE cols = list(obs_column) RBRACE { `Columns cols }
  (* `emit_schedule = every N 'unit | at [...] 'unit` — simulate-only cadence
     (§2.5). NOTE the literal form `every N` / `at [...]` (no inner `=`),
     distinct from the `every = ...` field form used by output/interventions. *)
  | EMIT_SCHEDULE EQ s = emit_schedule_spec { `Schedule s }
  (* `<scored_col> ~ Dist(kw = ..., ...)` — the measurement model (§2.1).
     The `| dim` pooling suffix (legal on a prior `~`) is meaningless here and
     is rejected by [obs_measurement_pooled] below. *)
  | scored = IDENT TILDE lik = obs_likelihood
      { `Measurement { om_scored = scored; om_lik = lik } }
  (* Migration: a likelihood `~` does NOT carry the prior's `| dim` pooling
     suffix (§2.1). Point the author at the bracket-index form. *)
  | scored = IDENT TILDE lik = obs_likelihood PIPE dim = IDENT
      { Parser_errors.push_error_hint ~sp:$startpos ~ep:$endpos
          ~code:"E271"
          ~msg:(Printf.sprintf
            "observation '%s ~ ...': the `| %s` pooling suffix is a PRIOR \
             construct and is meaningless on a likelihood" scored dim)
          ~hint:(Printf.sprintf
            "to stratify the observation, index the stream header: \
             `%s[a in %s] from <source> { ... }`" scored dim);
        `Measurement { om_scored = scored; om_lik = lik } }
  (* Migration: `every`/`at` at the top of an observation block was the old
     emission cadence; it is now `emit_schedule = ...` (§2.5). Reject the bare
     form with the rewrite. *)
  | s = schedule_core
      { Parser_errors.push_error_hint ~sp:$startpos ~ep:$endpos
          ~code:"E272"
          ~msg:"the observation emission cadence is now written `emit_schedule = ...`"
          ~hint:(match s with
            | SchedEvery _ -> "write `emit_schedule = every N 'unit` — see `camdl docs language-changes`"
            | SchedAt _    -> "write `emit_schedule = at [t1, t2, ...] 'unit` — see `camdl docs language-changes`");
        `Schedule s }
  | IDENT EQ proj = obs_projection { `Proj proj }
  (* Migration: `likelihood = D(...)` → `<col> ~ D(...)` (§9). Reject with the
     rewrite naming the new operator. *)
  | LIKELIHOOD EQ e = expr
      { Parser_errors.push_error_hint ~sp:$startpos ~ep:$endpos
          ~code:"E273"
          ~msg:"`likelihood = D(...)` was replaced by the `~` form"
          ~hint:"write `<value_col> ~ D(...)` where <value_col> is a declared \
                 column — see `camdl docs language-changes`";
        `Lik (match e with
        | EFuncCall (kind, args) -> lik_of_funcall kind args ~sp:$startpos ~ep:$endpos
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E104"
            ~msg:"likelihood value must be a function call, e.g. cases ~ neg_binomial(mean = projected, r = k)";
          LikPoisson []) }

(* The `emit_schedule` literal (§2.5): `every N 'unit` (regular) or
   `at [t1, t2, ...] 'unit` (explicit times). Lowers to the same
   [schedule_core] the expander already handles. The `'unit` on the `at` form
   rides on the list elements (each `expr` may carry a unit). *)
emit_schedule_spec:
  | EVERY e = expr
      { SchedEvery e }
  | AT_KW LBRACKET ts = separated_list(COMMA, expr) RBRACKET
      { SchedAt ts }

(* The `~` RHS distribution: `D(kw = ..., ...)` — keyword-only (the positional
   case is rejected downstream in the expander as E250). *)
obs_likelihood:
  | name = IDENT LPAREN args = separated_list(COMMA, kw_expr) RPAREN
      { lik_of_funcall name args ~sp:$startpos ~ep:$endpos }

(* `columns { name : role }` — one entry per declared file column (§2.2).
   role ∈ { time, dim, <value-type> }. `time`/`dim` are not reserved
   keywords; they are matched here by text so a model may still use those
   words as identifiers elsewhere. *)
obs_column:
  | name = IDENT COLON role = IDENT comma_opt
      { let r = match role with
          | "time" -> ColTime
          | "dim"  -> ColDim name
          | other  ->
            Parser_errors.push_error_hint ~sp:$startpos ~ep:$endpos
              ~code:"E274"
              ~msg:(Printf.sprintf
                "column '%s': unknown role '%s'" name other)
              ~hint:"role must be `time`, `dim`, or a value type \
                     (count, real, probability, positive)";
            ColValue PReal
        in { oc_name = name; oc_role = r } }
  | name = IDENT COLON pk = param_kind comma_opt
      { { oc_name = name; oc_role = ColValue pk } }

(* Optional separator: columns may be comma- or newline-separated. *)
comma_opt:
  | (* empty *) { () }
  | COMMA       { () }

obs_projection:
  | e = expr { ProjDerived e }

(* ── Interventions block ─────────────────────────────────────────────────── *)

intervention_list:
  | ivs = list(intervention_decl) { ivs }

intervention_decl:
  | name = IDENT ibs = index_bindings_opt COLON LBRACE iv_kvs = list(iv_kv) RBRACE guard = where_clause_opt
      { let action = ref (ATransfer []) in
        let sched  = ref (SAtTimes []) in
        List.iter (function
          | `Action a -> action := a
          | `Schedule s -> sched := s
        ) iv_kvs;
        { ivname = name; ivindices = ibs; ivaction = !action; ivschedule = !sched; ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  | name = IDENT ibs = index_bindings_opt COLON TRANSFER LPAREN kwargs = separated_list(COMMA, transfer_kwarg) RPAREN AT_KW LBRACKET ts = separated_list(COMMA, expr) RBRACKET guard = where_clause_opt
      { { ivname = name; ivindices = ibs; ivaction = ATransfer kwargs; ivschedule = SAtTimes ts; ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* transfer(...) { every = T, from = T0, until = T1 } — recurring schedule *)
  | name = IDENT ibs = index_bindings_opt COLON TRANSFER LPAREN kwargs = separated_list(COMMA, transfer_kwarg) RPAREN LBRACE sched = recurring_body RBRACE guard = where_clause_opt
      { { ivname = name; ivindices = ibs; ivaction = ATransfer kwargs; ivschedule = sched; ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* add(COMP, EXPR) at [...] *)
  | name = IDENT ibs = index_bindings_opt COLON ADD LPAREN comp = IDENT COMMA count = expr RPAREN AT_KW LBRACKET ts = separated_list(COMMA, expr) RBRACKET guard = where_clause_opt
      { { ivname = name; ivindices = ibs; ivaction = AAdd (comp, [], count); ivschedule = SAtTimes ts; ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* add(COMP, EXPR) { every = T, from = T0, until = T1 } — recurring schedule *)
  | name = IDENT ibs = index_bindings_opt COLON ADD LPAREN comp = IDENT COMMA count = expr RPAREN LBRACE sched = recurring_body RBRACE guard = where_clause_opt
      { { ivname = name; ivindices = ibs; ivaction = AAdd (comp, [], count); ivschedule = sched; ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* add(COMP, EXPR) every PERIOD at_day DAY *)
  | name = IDENT ibs = index_bindings_opt COLON ADD LPAREN comp = IDENT COMMA count = expr RPAREN EVERY period = expr AT_DAY day = expr guard = where_clause_opt
      { { ivname = name; ivindices = ibs; ivaction = AAdd (comp, [], count); ivschedule = SEveryAtDay (period, day); ivguard = guard;
          ivloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

(* ── Reactive interventions (gh#204) ─────────────────────────────────────────
   `name[idx]? : when <predicate> { action = .., after = .., once = .., ... }`.
   The predicate is a dedicated boolean grammar over comparison atoms; observed()
   / sum_observed() are ordinary IDENT funcalls recognised only here (the expander
   rejects them in rate expressions). *)
reactive_decl:
  | name = IDENT ibs = index_bindings_opt COLON WHEN pred = trig_pred LBRACE kvs = list(reactive_kv) RBRACE guard = where_clause_opt
      { let action   = ref None in
        let after    = ref None in
        let once     = ref None in
        let cooldown = ref None in
        List.iter (function
          | `Action a   -> action   := Some a
          | `After e    -> after    := Some e
          | `Once e     -> once     := Some e
          | `Cooldown e -> cooldown := Some e
        ) kvs;
        let act = match !action with
          | Some a -> a
          | None ->
            Parser_errors.push_error ~sp:$startpos ~ep:$endpos
              ~code:"E105"
              ~msg:"reactive intervention missing required 'action = ...'";
            ATransfer []
        in
        { rxname = name; rxindices = ibs; rxwhen = pred;
          rxafter = !after; rxonce = !once; rxcooldown = !cooldown;
          rxaction = act; rxguard = guard;
          rxloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

(* Boolean predicate. and/or/not over comparison atoms; the atom is a plain expr
   (a comparison like `observed(a) >= k`) validated/destructured in the expander.
   No `( predicate )` grouping in phase 1 — group inside the comparison via the
   expr's own parens. *)
trig_pred:
  | p1 = trig_pred OR  p2 = trig_pred   { TgOr  (p1, p2) }
  | p1 = trig_pred AND p2 = trig_pred   { TgAnd (p1, p2) }
  | NOT a = trig_atom                    { TgNot a }
  | a = trig_atom                        { a }

trig_atom:
  | e = expr   { TgAtom e }

(* Reactive policy body kvs — newline-separated, any order (camdl block style).
   `action = <action>` needs the ACTION token because its RHS is an action, not
   an expr; the rest are `key = expr`. *)
reactive_kv:
  | ACTION   EQ a = reactive_action  { `Action a }
  | k = IDENT EQ e = expr            { match k with
                                       | "after"    -> `After e
                                       | "once"     -> `Once e
                                       | "cooldown" -> `Cooldown e
                                       | "scope"    ->
                                         Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                                           ~code:"E106"
                                           ~msg:"the `scope` reactive key was removed: latent-scope (scope = particle) triggers are deferred — remove it, exogenous is implicit. See `camdl docs language-changes`";
                                         `After e
                                       | other ->
                                         Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                                           ~code:"E106"
                                           ~msg:(Printf.sprintf
                                             "unknown reactive intervention key '%s' (expected action/after/once/cooldown)" other);
                                         `After e }

(* Reactive action RHS: the same action forms as scheduled interventions, minus
   the schedule (which the trigger replaces). *)
reactive_action:
  | TRANSFER LPAREN kwargs = separated_list(COMMA, transfer_kwarg) RPAREN
      { ATransfer kwargs }
  | ADD LPAREN comp = IDENT COMMA count = expr RPAREN
      { AAdd (comp, [], count) }

(* Recurring schedule body: kwargs in any order, newline-separated
   (matches the rest of camdl's block style — no commas required). *)
recurring_body:
  | kvs = list(recurring_kv)
      { let every = ref None in
        let from_ = ref None in
        let until = ref None in
        List.iter (function
          | `Every e  -> every := Some e
          | `From  e  -> from_  := Some e
          | `Until e  -> until := Some e
        ) kvs;
        let every_e = match !every with
          | Some e -> e
          | None   ->
            Parser_errors.push_error ~sp:$startpos ~ep:$endpos
              ~code:"E105"
              ~msg:"recurring schedule missing required 'every = ...'";
            EConst 1.0
        in
        (* from and until default to simulate.from / simulate.to respectively. *)
        SRecurring (every_e, !from_, !until) }

recurring_kv:
  | EVERY EQ e = expr  { `Every e }
  | FROM  EQ e = expr  { `From  e }
  | UNTIL EQ e = expr  { `Until e }

transfer_kwarg:
  | k = IDENT EQ e = expr { (k, e) }
  | FROM EQ e = expr       { ("from", e) }
  | TO EQ e = expr         { ("to", e) }
  (* gh#49: `count` lexes to the COUNT token (it's reserved as a
     parameter type annotation, e.g. `S0 : count`), so the IDENT
     fallthrough doesn't catch it. Without this clause,
     `transfer(count = N, ...)` fails with E001 syntax error
     pointing at the `count` keyword. The expander has handled the
     "count" kwarg correctly since the IR was specced
     (Ir.AbsoluteTransfer with cap-at-source semantics in the
     runtime); only the parser was blocking. *)
  | COUNT EQ e = expr      { ("count", e) }

iv_kv:
  | AT_KW EQ LBRACKET ts = separated_list(COMMA, expr) RBRACKET
      { `Schedule (SAtTimes ts) }
  | EVERY EQ e = expr FROM EQ f = expr TO EQ t = expr
      { `Schedule (SRecurring (e, Some f, Some t)) }
  | IDENT EQ e = expr
      { (* action hint -- simplified *)
        `Action (ASet ($1, [], e)) }

(* ── ODE block ───────────────────────────────────────────────────────────── *)

ode_list:
  | odes = list(ode_decl) { odes }

ode_decl:
  | comp = IDENT EQ e = expr
      { { ocomp = comp; oderiv = e } }

(* ── Output block ────────────────────────────────────────────────────────── *)

output_body:
  | kvs = list(output_kv)
      { let traj = ref None in
        List.iter (function
          | `Traj t  -> traj  := Some t
        ) kvs;
        { out_trajectories = !traj } }

output_kv:
  | name = IDENT LBRACE fields = list(traj_field) RBRACE
      { match name with
        | "trajectories" ->
          let sched = ref None in
          let fmt   = ref "tsv" in
          let set_sched s =
            match !sched with
            | None   -> sched := Some s
            | Some _ ->
              Parser_errors.push_error ~sp:$startpos ~ep:$endpos ~code:"E106"
                ~msg:"trajectories: specify only one of `every` or `at`"
          in
          List.iter (function
            | `Sched s  -> set_sched s
            | `Format f -> fmt := f
          ) fields;
          let otschedule = Option.value !sched ~default:(SchedEvery (EConst 1.0)) in
          `Traj { otschedule; otformat = !fmt }
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E106"
            ~msg:(Printf.sprintf "unknown output section '%s': expected 'trajectories'" name);
          `Traj { otschedule = SchedEvery (EConst 1.0); otformat = "tsv" } }

traj_field:
  | s = schedule_core      { `Sched s }
  | FORMAT EQ f = IDENT    { `Format f }

(* Shared schedule core (every = E | at = [...]), reused by the surfaces
   that need a "specified times" schedule. *)
schedule_core:
  | EVERY EQ e = expr
      { SchedEvery e }
  | AT_KW EQ LBRACKET ts = separated_list(COMMA, expr) RBRACKET
      { SchedAt ts }

(* ── Simulate block ──────────────────────────────────────────────────────── *)

simulate_body:
  | kvs = list(simulate_kv)
      { let sim_from = ref (EConst 0.0) in
        let sim_to   = ref (EConst 100.0) in
        let sim_dt   = ref None in
        let sim_integrator = ref None in
        let sim_atol = ref None in
        let sim_rtol = ref None in
        List.iter (function
          | `From e -> sim_from := e
          | `To   e -> sim_to   := e
          | `Dt   e -> sim_dt   := Some e
          | `Integrator (meth, mloc, opts) ->
            sim_integrator := Some (meth, mloc);
            List.iter (fun (ok, e, eloc) -> match ok with
              | "atol" -> sim_atol := Some (e, eloc)
              | "rtol" -> sim_rtol := Some (e, eloc)
              | _ -> ()  (* opt-key validity already diagnosed in simulate_kv *)
            ) opts
        ) kvs;
        { sim_from = !sim_from; sim_to = !sim_to; sim_dt = !sim_dt;
          sim_integrator = !sim_integrator; sim_atol = !sim_atol; sim_rtol = !sim_rtol } }

(* `dt` is the discretization step (gh#161). It is a model knob — models are
   sensitive to it (discretization error; Richardson-extrapolation diagnostics
   deliberately vary it) — so it belongs in the model, with `--dt` as the CLI
   override. `dt` is *not* a keyword token: it is a bare identifier in rate
   expressions (`(1 - exp(-rate * dt))`), so it lexes as IDENT and is matched
   here by text. Unknown keys are a hard error (no-loose-semantics), never a
   silent drop. *)
simulate_kv:
  | FROM EQ e = expr { `From e }
  | TO   EQ e = expr { `To   e }
  (* gh#166: TAGGED integrator with an optional tolerance block —
     `integrator = rk45 { atol = 1e-8  rtol = 1e-6 }`. atol/rtol are keys of the
     rk45 block, so they cannot be written without rk45 (illegal-states-
     unrepresentable: the IR type is `Rk4 | Rk45 { atol, rtol }`). Opt-key
     validity is checked here; method validity and "rk4 takes no tolerances" in
     the expander (which builds the enum). *)
  | k = IDENT EQ meth = IDENT LBRACE opts = list(integ_opt) RBRACE
      { if k <> "integrator" then
          Parser_errors.push_error ~sp:$startpos(k) ~ep:$endpos(k) ~code:"E106"
            ~msg:(Printf.sprintf
              "unknown simulate key '%s' { ... }: only `integrator` takes a block" k);
        List.iter (fun (ok, _, _) ->
          if ok <> "atol" && ok <> "rtol" then
            Parser_errors.push_error ~sp:$startpos(meth) ~ep:$endpos ~code:"E106"
              ~msg:(Printf.sprintf
                "unknown integrator option '%s': expected `atol` or `rtol`" ok)) opts;
        `Integrator (meth, Parser_errors.ast_loc_of ~sp:$startpos(meth) ~ep:$endpos(meth), opts) }
  | k = IDENT EQ e = expr
      { match k with
        | "dt"   -> `Dt e
        | "integrator" ->
          (* bare form: `integrator = rk4` / `integrator = rk45` (default tols).
             The method lexes as an EIdent (atom_expr), which carries its loc. *)
          (match e with
           | EIdent (s, l) -> `Integrator (s, l, [])
           | _ ->
             Parser_errors.push_error ~sp:$startpos(e) ~ep:$endpos(e) ~code:"E106"
               ~msg:"`integrator` names an integrator: `integrator = rk4` or \
                     `integrator = rk45 { atol = .., rtol = .. }`";
             `Integrator ("rk4", Parser_errors.ast_loc_of ~sp:$startpos(e) ~ep:$endpos(e), []))
        | _ -> begin
          Parser_errors.push_error ~sp:$startpos(k) ~ep:$endpos(k)
            ~code:"E106"
            ~msg:(Printf.sprintf
              "unknown simulate key '%s': expected one of from, to, dt, integrator" k);
          `Dt e  (* placeholder; the error above aborts compilation *)
        end }

(* A single `key = expr` inside an `integrator = rk45 { ... }` block. The value's
   span rides along so the expander can locate a dimensioned-tolerance error. *)
integ_opt:
  | k = IDENT EQ e = expr
      { (k, e, Parser_errors.ast_loc_of ~sp:$startpos(e) ~ep:$endpos(e)) }

(* ── Init block ──────────────────────────────────────────────────────────── *)

init_list:
  | ies = list(init_entry) { ies }

init_entry:
  | comp = IDENT LBRACKET ibs = separated_nonempty_list(COMMA, index_binding) RBRACKET EQ v = expr
      { { icomp = comp; iindices = []; ibindings = ibs; ivalue = v;
          iloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  | comp = IDENT idxs = index_items_opt EQ v = expr
      { { icomp = comp; iindices = idxs; ibindings = []; ivalue = v;
          iloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

(* ── Timepoints block ────────────────────────────────────────────────────── *)

timepoint_list:
  | tps = list(timepoint_decl) { tps }

timepoint_decl:
  | name = IDENT EQ t = expr { { tpname = name; tptime = t } }

(* ── Dimensions ─────────────────────────────────────────────────────────── *)

dim_entry:
  | name = IDENT EQ src = dim_source_expr { { dename = name; desrc = src } }

dim_source_expr:
  | LBRACKET vs = separated_list(COMMA, IDENT) RBRACKET
      { DInline vs }
  | fn = IDENT LPAREN path = STRING COMMA kwname = IDENT EQ col = STRING RPAREN
      { DRead { fn_name = fn; path; col_kw = kwname; col } }

(* ── Stratify ────────────────────────────────────────────────────────────── *)

stratify_args:
  | kvs = separated_list(COMMA, stratify_kv)
      { let dim  = ref "" in
        let only = ref None in
        List.iter (function
          | `By d    -> dim := d
          | `Only cs -> only := Some cs
        ) kvs;
        { sdim = !dim; sonly = !only } }

stratify_kv:
  | BY EQ d = IDENT { `By d }
  | ONLY EQ LBRACKET cs = separated_list(COMMA, IDENT) RBRACKET { `Only cs }

(* ── Expression grammar ──────────────────────────────────────────────────── *)

expr:
  | IF p = expr THEN a = expr ELSE b = expr
      { ECond (p, a, b) }
  | e1 = expr EQ2   e2 = expr { EBinOp (Eq,  e1, e2) }
  | e1 = expr NEQ   e2 = expr { EBinOp (Neq, e1, e2) }
  | e1 = expr LT    e2 = expr { EBinOp (Lt,  e1, e2) }
  | e1 = expr GT    e2 = expr { EBinOp (Gt,  e1, e2) }
  | e1 = expr LE    e2 = expr { EBinOp (Le,  e1, e2) }
  | e1 = expr GE    e2 = expr { EBinOp (Ge,  e1, e2) }
  | e1 = expr PLUS  e2 = expr { EBinOp (Add, e1, e2) }
  | e1 = expr MINUS e2 = expr { EBinOp (Sub, e1, e2) }
  | e1 = expr STAR  e2 = expr { EBinOp (Mul, e1, e2) }
  | e1 = expr SLASH e2 = expr {
      (* E103: unit literal as right operand of / is always ambiguous.
         20 / 100_000 'per_year — does 'per_year bind to 100_000 or the whole expr?
         The parser binds it to 100_000, which is almost never what the user wants. *)
      (match e2 with
       | EUnit _ ->
         Parser_errors.push_error ~sp:$startpos ~ep:$endpos
           ~code:"E107"
           ~msg:"ambiguous unit literal after '/': the unit suffix binds to the \
                 adjacent number, not the whole expression. Use parentheses: \
                 (20 / 100_000) 'per_year, or pre-compute: 0.0002 'per_year"
       | _ -> ());
      EBinOp (Div, e1, e2)
    }
  | e1 = expr CROSS e2 = expr { EBinOp (Mul, e1, e2) }
  | e1 = expr CARET e2 = expr { EBinOp (Pow, e1, e2) }
  | MINUS e = expr %prec UMINUS { EUnOp (Neg, e) }
  | e = atom_expr { e }

atom_expr:
  | n = INT                    { EConst (float_of_int n) }
  | f = FLOAT                  { EConst f }
  | n = INT    u = unit_lit    { EUnit (float_of_int n, u) }
  | f = FLOAT  u = unit_lit    { EUnit (f, u) }
  | s = STRING                 { EIdent (s, dummy_loc) }   (* string literal usable as path arg *)
  | NULL                       { EConst 0.0 }
  | name = IDENT LPAREN args = separated_list(COMMA, kw_expr) RPAREN
      (* function call with optional keyword args *)
      { EFuncCall (name, args) }
  | SUM LPAREN v = IDENT IN d = IDENT COMMA body = expr RPAREN
      { ESum (v, d, None, body) }
  | SUM LPAREN v = IDENT IN d = IDENT WHERE g = guard_expr COMMA body = expr RPAREN
      { ESum (v, d, Some g, body) }
  | name = IDENT LBRACKET items = separated_list(COMMA, index_item) RBRACKET
      { EIndex (name, items) }
  | name = IDENT
      { let l =
          let open Lexing in
          { file     = $startpos.pos_fname;
            line     = $startpos.pos_lnum;
            col      = $startpos.pos_cnum - $startpos.pos_bol + 1;
            end_line = $endpos.pos_lnum;
            end_col  = $endpos.pos_cnum - $endpos.pos_bol + 1 }
        in
        EIdent (name, l) }
  (* `origin` as a referenceable identifier — Phase 2 of the
     2026-05-22 typed-time proposal §1.1. The ORIGIN keyword is
     consumed by the top-level `origin = date("...")` declaration
     via a separate production; here it appears in expression
     position. The expander resolves `origin` to Ir.Const 0.0 in
     anchored mode (it is the t=0 point) and errors in unanchored
     mode. *)
  | ORIGIN
      { let l =
          let open Lexing in
          { file     = $startpos.pos_fname;
            line     = $startpos.pos_lnum;
            col      = $startpos.pos_cnum - $startpos.pos_bol + 1;
            end_line = $endpos.pos_lnum;
            end_col  = $endpos.pos_cnum - $endpos.pos_bol + 1 }
        in
        EIdent ("origin", l) }
  | LPAREN e = expr RPAREN     { e }
  | LPAREN e = expr RPAREN u = unit_lit
      (* (20 / 100_000) 'per_year — unit applies to the whole expression.
         For durations: multiply by days_per(u). For rates: divide by days_per(u).
         The expander normalizes to the model time unit later. We encode it as
         expr * EUnit(1.0, u) so the expander handles unit conversion. *)
      { EBinOp (Mul, e, EUnit (1.0, u)) }
  | LBRACKET es = separated_list(COMMA, list_element) RBRACKET
      { EList es }

list_element:
  | lo = atom_expr COLON hi = atom_expr { ERange (lo, hi) }
  | e = expr                            { e }

(* A keyword-arg key can be a bare IDENT or one of the soft keywords
   that are reserved elsewhere but unambiguous in kwarg position
   (e.g. `poisson(rate = ...)`, `normal(mean = ..., sd = ...)`).
   Same pattern as prior_name. Extend as new clashes appear. *)
kw_arg_name:
  | id = IDENT  { id }
  | RATE        { "rate" }
  | COUNT       { "count" }
  | PROBABILITY { "probability" }
  | POSITIVE    { "positive" }
  | REAL        { "real" }
  | INTEGER     { "integer" }
  | EVERY       { "every" }  (* date_range(start, end, every = 7 'days) — §4 *)

kw_expr:
  | k = kw_arg_name EQ v = expr { (k, v) }
  | e = expr                     { ("", e) }

(* ── Scenarios block ─────────────────────────────────────────────────────── *)

scenario_block:
  | name = IDENT LBRACE fields = list(scenario_field) RBRACE
      { { Ast.scname = name; scfields = fields } }

scenario_field:
  | SIMULATE LBRACE kvs = list(simulate_kv) RBRACE
      { (* A scenario's `simulate {}` block overrides only the end time
           (`to`); it lowers to ScTEnd. `dt`/`integrator`/`atol`/`rtol` are
           whole-model knobs, not per-scenario overrides, so reject them here
           rather than silently drop them (no-loose-semantics). *)
        (match List.find_map (function
           | `Dt _         -> Some "dt"
           | `Integrator _ -> Some "integrator"
           | _             -> None) kvs with
         | Some key ->
           Parser_errors.push_error ~sp:$startpos ~ep:$endpos
             ~code:"E106"
             ~msg:(Printf.sprintf
               "`%s` is not a per-scenario override: set it once in the \
                top-level `simulate {}` block" key)
         | None -> ());
        let e = match List.find_map (function `To e -> Some e | _ -> None) kvs with
                | Some e -> e | None -> EConst 0.0 in
        Ast.ScTEnd e }
  | k = IDENT EQ LBRACE ps = list(scenario_kv_item) RBRACE
      { match k with
        | "set"   -> Ast.ScSet   ps
        | "scale" -> Ast.ScScale ps
        | _       -> Ast.ScSet   [(k, EConst 0.0)] }
  | EXTENDS EQ v = expr
      { let s = match v with
          | EIdent (s, _)    -> s
          | EFuncCall (s, []) -> s
          | _ ->
            Parser_errors.push_error ~sp:$startpos ~ep:$endpos
              ~code:"E108"
              ~msg:"invalid extends clause: expected a scenario name, e.g. extends = baseline";
            "" in
        Ast.ScExtends s }
  | k = IDENT EQ v = expr
      { match k with
        | "label"   ->
          let s = match v with
            | EIdent (s, _)    -> s   (* quoted string or bare identifier *)
            | EFuncCall (s, []) -> s  (* zero-arg call used as name *)
            | EConst f         -> string_of_float f
            | _ ->
              Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                ~code:"E109"
                ~msg:"invalid scenario label: expected a quoted string or identifier, e.g. label = \"baseline\"";
              "" in
          Ast.ScLabel s
        | "enable"  -> Ast.ScEnable  (extract_ident_list v)
        | "disable" -> Ast.ScDisable (extract_ident_list v)
        | "compose" -> Ast.ScCompose (extract_ident_list v)
        | _         -> Ast.ScSet [(k, v)] }

scenario_kv_item:
  | k = IDENT LBRACKET idxs = separated_nonempty_list(COMMA, IDENT) RBRACKET EQ v = expr
      { (String.concat "_" (k :: idxs), v) }
  | k = IDENT EQ v = expr { (k, v) }
  (* Error production: entries in a scenario `set`/`scale` block are
     separated by NEWLINES, not commas. Commas are reserved for `[...]`
     lists and `(...)` argument lists. Without this clause a comma after
     a complete entry (`set = { mu = 0.1, nu = 0.2 }`) hits no production
     and surfaces as a bare E001 pointing at the next key, with no
     explanation. We consume the stray comma so parsing recovers, and
     buffer an E001 whose message names the separator and shows the fix.
     "Error messages are a feature" — see CLAUDE.md. *)
  | k = IDENT LBRACKET idxs = separated_nonempty_list(COMMA, IDENT) RBRACKET EQ v = expr COMMA
      { Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E001"
          ~msg:"syntax error: entries in a scenario `set`/`scale` block are \
                separated by newlines, not commas. Write each assignment on \
                its own line, e.g.\n  set = {\n    mu = 0.1\n    nu = 0.2\n  }";
        (String.concat "_" (k :: idxs), v) }
  | k = IDENT EQ v = expr COMMA
      { Parser_errors.push_error ~sp:$startpos ~ep:$endpos
          ~code:"E001"
          ~msg:"syntax error: entries in a scenario `set`/`scale` block are \
                separated by newlines, not commas. Write each assignment on \
                its own line, e.g.\n  set = {\n    mu = 0.1\n    nu = 0.2\n  }";
        (k, v) }
