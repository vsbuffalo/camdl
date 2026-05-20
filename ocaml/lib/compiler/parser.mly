%{
  open Ast

  let extract_ident_list = function
    | EList items -> List.filter_map (function EIdent (n, _) -> Some n | _ -> None) items
    | _ -> []
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
%token AND OR NOT IF THEN ELSE EVERY UNTIL AT_KW FORMAT DESCRIPTION TAG NULL TRANSFER LIKELIHOOD ORIGIN BALANCE EVENTS ADD AT_DAY
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
  | name = IDENT COLON pk = param_kind da = dim_annotation_opt
      { PScalar { pname = name; pkind = pk; pdim = da; pbounds = None; pprior = None;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, no bounds, with prior *)
  | name = IDENT COLON pk = param_kind da = dim_annotation_opt TILDE pr = prior_clause
      { PScalar { pname = name; pkind = pk; pdim = da; pbounds = None; pprior = Some pr;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, with bounds, no prior *)
  | name = IDENT COLON pk = param_kind da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET
      { PScalar { pname = name; pkind = pk; pdim = da; pbounds = Some (lo, hi); pprior = None;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* scalar, with bounds, with prior *)
  | name = IDENT COLON pk = param_kind da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET TILDE pr = prior_clause
      { PScalar { pname = name; pkind = pk; pdim = da; pbounds = Some (lo, hi); pprior = Some pr;
                  ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, no bounds, no prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind da = dim_annotation_opt
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; pbounds = None; pprior = None;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, no bounds, with prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind da = dim_annotation_opt TILDE pr = prior_clause
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; pbounds = None; pprior = Some pr;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, with bounds, no prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; pbounds = Some (lo, hi); pprior = None;
                   ploc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* indexed, with bounds, with prior *)
  | name = IDENT LBRACKET dim = IDENT RBRACKET COLON pk = param_kind da = dim_annotation_opt IN LBRACKET lo = expr COMMA hi = expr RBRACKET TILDE pr = prior_clause
      { PIndexed { pname = name; pdims = [dim]; pkind = pk; pdim = da; pbounds = Some (lo, hi); pprior = Some pr;
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
          trrate = rate; trguard = guard; trtag = None; trlineage = lin;
          trloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* block form: [#[lineage]] name[...] : srcs --> dsts { rate = ...; tag = ... } *)
  | lin = lineage_attr_opt name = IDENT ibs = index_bindings_opt COLON srcs = stoich_ref_list ARROW dsts = stoich_ref_list LBRACE tbody = transition_body RBRACE
      { let (rate, guard, tag) = tbody in
        { trname = name; trindices = ibs;
          trsrc = srcs; trdst = DstSum dsts;
          trrate = rate; trguard = guard; trtag = tag; trlineage = lin;
          trloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }
  (* branching: [#[lineage]] name[...] : srcs --> { D1 : w1, ... } @ rate where guard *)
  | lin = lineage_attr_opt name = IDENT ibs = index_bindings_opt COLON srcs = stoich_ref_list ARROW LBRACE branches = separated_nonempty_list(COMMA, branch_entry) RBRACE AT rate = expr guard = where_clause_opt
      { { trname = name; trindices = ibs;
          trsrc = srcs; trdst = DstBranch branches;
          trrate = rate; trguard = guard; trtag = None; trlineage = lin;
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
      { let rate  = ref (EConst 0.0) in
        let guard = ref None in
        let tag   = ref None in
        List.iter (function
          | `Rate e  -> rate := e
          | `Guard g -> guard := Some g
          | `Tag s   -> tag := Some s
        ) kvs;
        (!rate, !guard, !tag) }

transition_body_entry:
  | RATE EQ e = expr { `Rate e }
  | WHERE g = guard_expr { `Guard g }
  | TAG EQ s = STRING { `Tag s }

guard_expr:
  | g = guard_atom { g }
  | g1 = guard_expr AND g2 = guard_expr { GAnd (g1, g2) }
  | g1 = guard_expr OR  g2 = guard_expr { GOr  (g1, g2) }

guard_atom:
  | a = IDENT EQ2 b = IDENT { GEq  (a, b) }
  | a = IDENT NEQ  b = IDENT { GNeq (a, b) }
  | LPAREN g = guard_expr RPAREN { g }

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

obs_decl:
  | name = IDENT ibs = index_bindings_opt COLON LBRACE obs_kvs = list(obs_kv) RBRACE
      { let ds = ref None in
        let sched = ref None in
        let proj = ref None in
        let lik = ref None in
        List.iter (function
          | `DataStream s -> ds := Some s
          | `Schedule sc  -> sched := Some sc
          | `Proj p       -> proj := Some p
          | `Lik l        -> lik := Some l
        ) obs_kvs;
        { oname = name; oindices = ibs; odata_stream = !ds;
          oschedule = !sched; oprojection = !proj; olikelihood = !lik;
          oloc = Parser_errors.ast_loc_of ~sp:$startpos ~ep:$endpos } }

obs_kv:
  | IDENT EQ s = STRING { `DataStream s }
  | EVERY EQ e = expr { `Schedule (ObsEvery e) }
  | AT_KW EQ LBRACKET ts = separated_list(COMMA, expr) RBRACKET { `Schedule (ObsTimes ts) }
  | IDENT EQ proj = obs_projection { `Proj proj }
  | LIKELIHOOD EQ e = expr
      { `Lik (match e with
        | EFuncCall (kind, args) -> (match kind with
            | "neg_binomial"  -> LikNegBinomial  args
            | "poisson"       -> LikPoisson      args
            | "normal"        -> LikNormal       args
            | "binomial"      -> LikBinomial     args
            | "beta_binomial" -> LikBetaBinomial args
            | "bernoulli"     -> LikBernoulli    args
            (* diagnostic_test(base = <binomial|bernoulli>(…, p = π), sens, spec)
               is pure compile-time sugar: it rewrites the inner `p`
               expression from π to  sens·π + (1−spec)·(1−π)
               — the probability that a truly-positive fraction π
               produces a positive observation under an imperfect
               test. The IR sees just Binomial/Bernoulli, so the
               runtime path is identical to hand-inlining the same
               algebra (see wave 1 / malaria #4). *)
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
                    Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                      ~code:"E253"
                      ~msg:(Printf.sprintf
                        "diagnostic_test base must be binomial(...) or bernoulli(...); got %s(...)"
                        other);
                    LikBinomial [])
               | _ ->
                 Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                   ~code:"E254"
                   ~msg:"diagnostic_test requires keyword args base = <binomial|bernoulli>(...), sens = <expr>, spec = <expr>";
                 LikBinomial [])
            | s ->
              Parser_errors.push_error ~sp:$startpos ~ep:$endpos
                ~code:"E104"
                ~msg:(Printf.sprintf "unknown likelihood '%s': expected one of neg_binomial, poisson, normal, binomial, beta_binomial, bernoulli, diagnostic_test" s);
              LikPoisson args)
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E104"
            ~msg:"likelihood value must be a function call, e.g. likelihood = neg_binomial(mean = projected, dispersion = k)";
          LikPoisson []) }

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
        let flows = ref None in
        let summ  = ref None in
        List.iter (function
          | `Traj t  -> traj  := Some t
          | `Flows f -> flows := Some f
          | `Summ s  -> summ  := Some s
        ) kvs;
        { out_trajectories = !traj; out_flows = !flows; out_summary = !summ } }

output_kv:
  | IDENT LBRACE kvs = list(func_arg) RBRACE
      { match $1 with
        | "trajectories" ->
          let every = List.assoc_opt "every" kvs |> Option.value ~default:(EConst 1.0) in
          let fmt   = (match List.assoc_opt "format" kvs with
                       | Some (EIdent (s, _)) | Some (EFuncCall (s, [])) -> s
                       | _ -> "tsv") in
          let rest  = List.filter (fun (k,_) -> k <> "every" && k <> "format") kvs in
          `Traj { otevery = every; otquantities = rest; otformat = fmt }
        | "flows" ->
          let every = List.assoc_opt "every" kvs |> Option.value ~default:(EConst 1.0) in
          let fmt   = (match List.assoc_opt "format" kvs with
                       | Some (EIdent (s, _)) | Some (EFuncCall (s, [])) -> s
                       | _ -> "tsv") in
          let rest  = List.filter (fun (k,_) -> k <> "every" && k <> "format") kvs in
          `Flows { otevery = every; otquantities = rest; otformat = fmt }
        | "summary" ->
          let fmt   = (match List.assoc_opt "format" kvs with
                       | Some (EIdent (s, _)) | Some (EFuncCall (s, [])) -> s
                       | _ -> "tsv") in
          let rest  = List.filter (fun (k,_) -> k <> "format") kvs in
          `Summ { osquantities = rest; osformat = fmt }
        | _ ->
          Parser_errors.push_error ~sp:$startpos ~ep:$endpos
            ~code:"E106"
            ~msg:(Printf.sprintf "unknown output section '%s': expected one of trajectories, flows, summary" $1);
          `Summ { osquantities = []; osformat = "tsv" } }

(* ── Simulate block ──────────────────────────────────────────────────────── *)

simulate_body:
  | kvs = list(simulate_kv)
      { let sim_from = ref (EConst 0.0) in
        let sim_to   = ref (EConst 100.0) in
        List.iter (function
          | `From e -> sim_from := e
          | `To   e -> sim_to   := e
        ) kvs;
        { sim_from = !sim_from; sim_to = !sim_to } }

simulate_kv:
  | FROM EQ e = expr { `From e }
  | TO   EQ e = expr { `To   e }

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
      { ESum (v, d, body) }
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

kw_expr:
  | k = kw_arg_name EQ v = expr { (k, v) }
  | e = expr                     { ("", e) }

(* ── Scenarios block ─────────────────────────────────────────────────────── *)

scenario_block:
  | name = IDENT LBRACE fields = list(scenario_field) RBRACE
      { { Ast.scname = name; scfields = fields } }

scenario_field:
  | SIMULATE LBRACE kvs = list(simulate_kv) RBRACE
      { let e = match List.find_map (function `To e -> Some e | _ -> None) kvs with
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
