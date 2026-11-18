(** Builds the declared-model section of a compiled model, a
    {!Declared.declared_model}, from the parsed declarations of a compile that
    succeeded.

    The section records a model as its author wrote it (see {!Declared} for
    what it holds and who reads it). This module converts each
    {!Ast.declaration} into its declared form, block by block in source
    order: every expression becomes its parsed tree without source locations,
    with a KaTeX rendering from {!Latex}, the ways that rendering falls short
    of the tree, and the names the tree reads; every declaration gets its text
    as written, sliced from the source by its span.

    It reads the declaration list the parser produced, not the expander's
    context, because by the end of expansion the [via] lowering has rewritten
    the context: a compartment that a [via hyper_erlang] transition stages is
    absent from it under its declared name. From the expander it asks only
    what the declarations alone do not say: what each name refers to, through
    the expander's own classifiers ([Expander.classify_name] and
    [Expander.classify_indexed_name]), so that the section and the compiled
    model resolve names by one set of rules; the levels of each dimension,
    including one read from a file; each compartment's dimensions; and the
    expanded names each compartment, transition and observation stream
    produced.

    A name is classified by where it sits as well as by what it spells.
    Binder variables are checked before the classifier sees a name, because
    the expander substitutes them before resolution. A name in an index
    position may be a level. The declared compartment names are consulted
    when the classifier, which sees the context after lowering, does not know
    one. Quantities, streams and transitions are names only where a reporting
    form reads them: a quantity or contrast body, a projection, a reactive
    trigger. A few positions hold words the compiler reads by their spelling,
    [true] and [false] as a call argument, the [dim] of [unchecked_dim(...)],
    the [label] of a [hyper_erlang] branch and a bare forcing argument such as
    [method = linear]; these are tagged {!Declared.KKeyword}.

    A name no rule classifies is tagged {!Declared.KUnresolved}, and
    {!build_detail} returns it, as an {!unresolved}, beside the section; the
    compiler reports each as warning W203. The compiled model never depends
    on the section, so W203 never fails a compile, and a test fails on any
    unresolved name in a compiled golden model.

    {!build_detail} and {!build} take the context of a successful
    [Expander.expand_detail]; {!Compiler} builds the section lazily, once,
    after the analysis passes succeed. {!with_compile_inputs} adds what only
    the command line knows, the [--set] overrides and the [--quantities]
    file, and [camdlc] calls it before writing the envelope. *)

open Ast
module D = Declared

(** A name the section could not classify, as warning W203 reports it. *)
type unresolved = {
  u_site : string;
      (** The declaration the name sits in, in words a diagnostic can print:
          ["transition 'infection'"], or ["the simulate block"]. *)
  u_name : string;
  u_loc  : Ast.loc;
      (** Where the name is written. *)
}

(** What converting one declaration needs to know about the whole model:
    the expander's context, the source text, and the declared names of
    compartments, quantities, streams, transitions, dimensions and levels.
    [site] names the declaration being converted, and [unresolved]
    collects, newest first, the names that could not be classified; both
    are mutated during {!build_detail}. *)
type env = {
  ctx       : Expander.context;
  source    : Source_cache.t;  (* the text of every unit of this compile *)
  comps     : string list;   (* declared compartment names, pre-lowering *)
  quantities : string list;  (* declared quantity names *)
  streams   : string list;   (* declared observation stream names *)
  transitions : string list; (* declared transition names *)
  levels    : (string, unit) Hashtbl.t;   (* every level of every dimension *)
  dims      : string list;   (* declared dimension names *)
  site      : string ref;    (* the declaration being converted *)
  unresolved : unresolved list ref;   (* newest first *)
}

(* Record [name] as unclassified when [kind] says so, and return [kind]. Called
   only where a name enters a tree, so a probe of [kind_of_bare] (a forcing
   argument deciding between a name and a keyword) records nothing. *)
let noted env (name : string) (loc : Ast.loc) (kind : D.name_kind) : D.name_kind =
  if kind = D.KUnresolved then
    env.unresolved :=
      { u_site = !(env.site); u_name = name; u_loc = loc } :: !(env.unresolved);
  kind

(* A declaration's text as written: the bytes its span covers, read back from
   the source the parser was given. A declaration with no span (none is
   parsed without one) or a span outside the cached text has no text; it is
   never approximated. *)
let text env (l : Ast.loc) : D.source_text =
  if l = dummy_loc then None
  else
    Source_cache.slice env.source ~file:l.file ~line:l.line ~col:l.col
      ~end_line:l.end_line ~end_col:l.end_col

(** Where an expression sits, as far as classifying its names depends on it:
    the variables bound around it, the data columns a likelihood or a
    [covers] span may read, and whether it is a reporting position, where a
    quantity, stream or transition may be named. *)
type pos = {
  binders  : string list;   (* index-binder and enclosing `sum` variables *)
  aux_cols : string list;   (* a likelihood's auxiliary data columns *)
  reporting : bool;         (* where a quantity, stream or flow is named: a
                               quantity or contrast body, a projection, a
                               reactive trigger *)
}

let top = { binders = []; aux_cols = []; reporting = false }

let binders_of (bs : index_binding list) : string list =
  List.concat_map (function
    | IBind (v, _)        -> [v]
    | IConsec (v, v2, _)  -> [v; v2]
    | IComp v | IStages (v, _)             -> [v]) bs

let with_binders pos bs = { pos with binders = binders_of bs @ pos.binders }

(* ── Data-less conversions ───────────────────────────────────────────────── *)

let unit_lit : Ast.unit_lit -> D.unit_lit = function
  | Days -> UDays | Weeks -> UWeeks | Months -> UMonths | Years -> UYears
  | PerDay -> UPerDay | PerWeek -> UPerWeek | PerMonth -> UPerMonth
  | PerYear -> UPerYear | Count -> UCount | Ratio -> URatio

let bin_op : Ast.bin_op -> D.bin_op = function
  | Add -> BAdd | Sub -> BSub | Mul -> BMul | Div -> BDiv | Pow -> BPow
  | Eq -> BEq | Neq -> BNeq | Lt -> BLt | Gt -> BGt | Le -> BLe | Ge -> BGe

let un_op : Ast.un_op -> D.un_op = function
  | Neg -> ONeg | Exp -> OExp | Log -> OLog | Sqrt -> OSqrt | Abs -> OAbs
  | Floor -> OFloor | Ceil -> OCeil | Sin -> OSin | Cos -> OCos | Tanh -> OTanh
  | Total -> OTotal

let rel_op : Ast.relop -> D.rel_op = function
  | RLt -> D.RLt | RLe -> D.RLe | RGt -> D.RGt | RGe -> D.RGe
  | REq -> D.REq | RNe -> D.RNe

let rec guard : Ast.guard -> D.guard = function
  | GEq (a, b)  -> D.GEq (a, b)
  | GNeq (a, b) -> D.GNeq (a, b)
  | GTab (t, idx, op, operand) ->
    D.GTable { gt_table = t; gt_indices = idx; gt_op = rel_op op;
               gt_operand = (match operand with
                 | GoNum f  -> D.GoNum f
                 | GoName n -> D.GoName n) }
  | GAnd (a, b) -> D.GAnd (guard a, guard b)
  | GOr (a, b)  -> D.GOr (guard a, guard b)

let index_binding : Ast.index_binding -> D.index_binding = function
  | IBind (v, d)        -> BIn (v, d)
  | IConsec (v, v2, d)  -> BConsecutive (v, v2, d)
  | IComp v             -> BCompartments v
  (* The declared model has no binder of its own for the stages of a
     compartment: the domain is recorded as it is written. *)
  | IStages (v, c)      -> BIn (v, Printf.sprintf "stages(%s)" c)

let bindings = List.map index_binding

let param_kind = Expander.ir_param_kind_of_ast

let dim_annotation ((p, t) : Ast.dim_annotation) : D.dim_annotation =
  { da_population = p; da_time = t }

let family : likelihood_kind -> D.family = function
  | LikNegBinomial _  -> FNegBinomial
  | LikPoisson _      -> FPoisson
  | LikNormal _       -> FNormal
  | LikBinomial _     -> FBinomial
  | LikBetaBinomial _ -> FBetaBinomial
  | LikBeta _         -> FBeta
  | LikBernoulli _    -> FBernoulli
  | LikZeroInflatedNegBinomial _ -> FZeroInflatedNegBinomial

(* ── What a name is ──────────────────────────────────────────────────────── *)

let kind_of_class : Expander.name_class -> D.name_kind option = function
  | Name_obs_column -> Some KDataColumn
  | Name_compartment_cell | Name_compartment -> Some KCompartment
  | Name_param | Name_param_cell -> Some KParameter
  | Name_let_cyclic | Name_let_indexed_bare _ | Name_let_const _
  | Name_let_hoisted _ | Name_let_inlined _ -> Some KDefinition
  | Name_forcing -> Some KTimeInput
  | Name_time | Name_dt | Name_projected | Name_pi | Name_e | Name_origin ->
    Some KBuiltin
  | Name_unresolved -> None

(* The compiler's reserved time names that are not in the classifier's chain:
   `t_start` and `t_end` are read where a schedule or a window is resolved. *)
let reserved_time = [ "t_start"; "t_end" ]

(* A bare name. Binder variables are substituted before the compiler's
   classifier ever sees a name, so they are checked first; then the classifier;
   then the declared names the classifier cannot see. *)
let kind_of_bare env pos (name : string) : D.name_kind =
  if List.mem name pos.binders then KIndexVar
  else
    match kind_of_class
            (Expander.classify_name env.ctx ~obs_aux_cols:pos.aux_cols
               ~suppress_hoist:false name) with
    | Some k -> k
    | None ->
      if List.mem name env.comps then KCompartment
      else if Expander.anchor_of_ident name <> None then KObsAnchor
      else if List.mem name reserved_time then KBuiltin
      else if Option.is_some (Expander.table_dims env.ctx name) then KTable
      (* A parameter family named without its index (`modes = z`). *)
      else if Expander.is_indexed_param env.ctx name then KParameter
      else if pos.reporting && List.mem name env.quantities then KQuantity
      else if pos.reporting && List.mem name env.streams then KStream
      (* A flow is named by its transition, or by one expanded cell of it
         (`incidence(recovery_p1)`). *)
      else if pos.reporting
           && List.exists (fun base -> Expander.is_expansion_of ~base name)
                env.transitions then KTransition
      else KUnresolved

(* The head of `name[...]`. *)
let kind_of_head env pos (name : string) : D.name_kind =
  if List.mem name pos.binders then KIndexVar
  else
    match Expander.classify_indexed_name env.ctx name with
    | Indexed_table _ -> KTable
    | Indexed_let_cyclic | Indexed_let _ | Indexed_shaped_let _ -> KDefinition
    | Indexed_forcing _ -> KTimeInput
    | Indexed_param -> KParameter
    | Indexed_cell -> kind_of_bare env pos name

(* A name in an index position is a bound variable or a level name; the
   compiler substitutes the first and takes the second as text
   (`Expander.index_item_to_str`). *)
let kind_of_index env pos (name : string) : D.name_kind =
  if List.mem name pos.binders then KIndexVar
  else if Hashtbl.mem env.levels name then KLevel
  else kind_of_bare env pos name

(* ── Expressions ─────────────────────────────────────────────────────────── *)

(* Calls whose named arguments include words the compiler reads as text, not
   names: (function, argument name). `unchecked_dim(e, dim = population)`
   names a dimension keyword; a `hyper_erlang` branch's `label` names the
   branch (`Expander`'s via lowering reads the identifier's spelling). *)
let keyword_args = [ ("unchecked_dim", "dim"); ("branch", "label") ]

(* The Boolean words. The compiler matches them by spelling wherever a
   Boolean is accepted (`date_range`'s `inclusive_end`, a reactive
   intervention's `once`), so a `true` or `false` that is a call argument or a
   `once` value is a keyword whatever the call. *)
let is_boolean_word = function "true" | "false" -> true | _ -> false

let rec tree env pos (e : expr) : D.expr_tree =
  match e with
  | EConst f -> TConst f
  | EUnit (f, u) -> TUnit (f, unit_lit u)
  (* A quoted string parses to an identifier with no source location
     (parser.mly, the `STRING` atom). *)
  | EIdent (s, loc) when loc = dummy_loc -> TStr s
  | EIdent (s, loc) ->
    TIdent { nr_name = s; nr_kind = noted env s loc (kind_of_bare env pos s) }
  | EIndex (n, items, loc) ->
    TIndex ({ nr_name = n; nr_kind = noted env n loc (kind_of_head env pos n) },
            List.map (index_item env pos) items)
  | EBinOp (op, a, b) -> TBinOp (bin_op op, tree env pos a, tree env pos b)
  | EUnOp (op, a) -> TUnOp (un_op op, tree env pos a)
  | ESum (v, d, g, body, _) ->
    TSum { ts_var = v; ts_dim = d; ts_guard = Option.map guard g;
           ts_body = tree env { pos with binders = v :: pos.binders } body }
  | ESumStages (v, c, body, _) ->
    TSum { ts_var = v; ts_dim = Printf.sprintf "stages(%s)" c; ts_guard = None;
           ts_body = tree env { pos with binders = v :: pos.binders } body }
  | ECond (p, a, b) -> TCond (tree env pos p, tree env pos a, tree env pos b)
  | EFuncCall (f, kvs) -> TCall (f, List.map (call_arg env pos f) kvs)
  | EList xs -> TList (List.map (tree env pos) xs)
  | ERange (a, b) -> TRange (tree env pos a, tree env pos b)
  | EObsAccess (s, _) -> TObsAccess s
  | ERunMember { run; ns; member; _ } ->
    TRunMember { tr_run = run;
                 tr_namespace = (match ns with
                   | Ast.NsQuantities -> D.NsQuantities
                   | Ast.NsObservations -> D.NsObservations);
                 tr_member = member }

and call_arg env pos f ((k, v) : string * expr) : D.arg =
  match v with
  | EIdent (s, loc) when loc <> dummy_loc
                         && (is_boolean_word s || List.mem (f, k) keyword_args) ->
    { a_name = k; a_value = TIdent { nr_name = s; nr_kind = KKeyword } }
  | _ -> { a_name = k; a_value = tree env pos v }

and index_item env pos : Ast.index_item -> D.index_item = function
  | IPosn e -> IPositional (index_tree env pos e)
  | INamed (d, e) -> D.INamed (d, index_tree env pos e)
  (* `stage(E[a], 2)`: the location, as the call `stage(2)` in the last
     index position. A location is a number or a variable bound by
     `in stages(E)`, never a level name the model declares. *)
  | IStage e ->
    IPositional (TCall ("stage", [ { a_name = ""; a_value = index_tree env pos e } ]))

and index_tree env pos (e : expr) : D.expr_tree =
  match e with
  | EIdent (s, loc) when loc <> dummy_loc ->
    TIdent { nr_name = s; nr_kind = noted env s loc (kind_of_index env pos s) }
  | _ -> tree env pos e

let args env pos (kvs : (string * expr) list) : D.arg list =
  List.map (fun (k, v) -> { D.a_name = k; a_value = tree env pos v }) kvs

(* Every name reference in a tree, in first-occurrence order, without
   repeats. *)
let reads_of (t : D.expr_tree) : D.name_ref list =
  let seen = Hashtbl.create 8 in
  let out = ref [] in
  let add (r : D.name_ref) =
    if not (Hashtbl.mem seen r) then begin
      Hashtbl.add seen r ();
      out := r :: !out
    end
  in
  let rec go : D.expr_tree -> unit = function
    | TConst _ | TUnit _ | TStr _ | TObsAccess _ | TRunMember _ -> ()
    | TIdent r -> add r
    | TIndex (r, items) -> add r; List.iter item items
    | TBinOp (_, a, b) | TRange (a, b) -> go a; go b
    | TUnOp (_, a) -> go a
    | TSum { ts_body; _ } -> go ts_body
    | TCond (p, a, b) -> go p; go a; go b
    | TCall (_, xs) -> List.iter (fun (a : D.arg) -> go a.a_value) xs
    | TList xs -> List.iter go xs
  and item : D.index_item -> unit = function
    | IPositional e | D.INamed (_, e) -> go e
  in
  go t;
  List.rev !out

let dexpr env pos (e : expr) : D.declared_expr =
  let t = tree env pos e in
  { de_tree = t; de_katex = Latex.tex e;
    de_katex_loss = Latex.tex_loss e; de_reads = reads_of t }

let items env pos = List.map (index_item env pos)

let stoich env pos ((c, idx) : stoich_ref) : D.stoich_ref =
  { sr_compartment = c; sr_indices = items env pos idx }

let distribution env pos (l : likelihood_kind) : D.distribution =
  { di_family = family l; di_args = args env pos (likelihood_kwargs l) }

let schedule_core env pos : Ast.schedule_core -> D.schedule_core = function
  | SchedEvery p -> ScEvery (dexpr env pos (Time_pos.expr p))
  | SchedAt ps -> ScAt (List.map (fun p -> dexpr env pos (Time_pos.expr p)) ps)

(* ── Declarations ────────────────────────────────────────────────────────── *)

let dimension env (de : dimensions_entry) : D.dimension_decl =
  { dd_name = de.dename;
    dd_source = (match de.desrc with
      | DInline _ -> DsInline
      | DRead { path; col; _ } -> DsRead { ds_path = path; ds_column = col });
    dd_levels = Expander.dim_levels_or_empty env.ctx de.dename;
    dd_text = text env de.despan }

(* Dimensions the compiler invents (a `via` stage dimension) begin `__`. *)
let invented name = String.length name >= 2 && String.sub name 0 2 = "__"

let compartment env (c : Ast.compartment_decl) : D.compartment_decl =
  { cd_name = c.cname;
    cd_kind = (match c.ckind with Integer -> CkInteger | Real -> CkReal);
    cd_dims = List.filter (fun d -> not (invented d))
        (Expander.comp_dims env.ctx c.cname);
    cd_expands_to =
      Some (Expander.expanded_compartments env.ctx ~declared:c.cname);
    cd_text = text env c.cspan }

let prior env pos (p : prior_spec) : D.prior_decl =
  { pr_family = p.ps_name; pr_args = args env pos p.ps_args;
    pr_pool_over = p.ps_pool_over }

let bounds env pos (b : Ast.bounds) : D.bounds =
  let (lo, hi) = bounds_exprs b in
  { b_lo = dexpr env pos lo; b_hi = dexpr env pos hi }

let parameter env : param_decl -> D.parameter_decl = function
  | PScalar { pname; pkind; pdim; pbounds; pprior; pspan; _ } ->
    { pd_name = pname; pd_dims = []; pd_kind = param_kind pkind;
      pd_dim = Option.map dim_annotation pdim;
      pd_bounds = Option.map (bounds env top) pbounds;
      pd_prior = Option.map (prior env top) pprior;
      pd_text = text env pspan }
  | PIndexed { pname; pdims; pkind; pdim; pbounds; pprior; pspan; _ } ->
    { pd_name = pname; pd_dims = pdims; pd_kind = param_kind pkind;
      pd_dim = Option.map dim_annotation pdim;
      pd_bounds = Option.map (bounds env top) pbounds;
      pd_prior = Option.map (prior env top) pprior;
      pd_text = text env pspan }

let definition env (lb : let_binding) : D.definition_decl =
  let pos = with_binders top lb.lindices in
  { df_name = lb.lname; df_indices = bindings lb.lindices;
    df_shape = lb.lshape; df_kind = Option.map param_kind lb.lkind;
    df_dim = Option.map dim_annotation lb.ldim;
    df_body = dexpr env pos lb.lbody;
    df_text = text env lb.lspan }

let table env (t : Ast.table_decl) : D.table_decl =
  { tb_names = t.tnames;
    tb_dims = List.map (function
      | TDim d -> { D.td_dim = d; td_unit = None }
      | TDimUnit (d, u) -> { D.td_dim = d; td_unit = Some (unit_lit u) }) t.tdims;
    tb_cell_kind = Option.map param_kind t.tcell_kind;
    tb_value = dexpr env top t.tvalue;
    tb_text = text env t.tspan }

(* A forcing argument that is a single bare word naming nothing declared is a
   selection from a closed set (`method = linear`). The compiler reads it as
   text (`Expander`'s forcing expansion matches the identifier's spelling). *)
let forcing_arg env pos ((k, v) : string * farg_value) : D.time_input_arg =
  { ta_name = k;
    ta_value = (match v with
      | FStr s -> TvStr s
      | FExpr (EIdent (s, loc) as e) when loc <> dummy_loc
                                       && kind_of_bare env pos s = KUnresolved ->
        let kind =
          if List.mem s env.dims then D.KDimension else D.KKeyword in
        let t = D.TIdent { nr_name = s; nr_kind = kind } in
        TvExpr { de_tree = t; de_katex = Latex.tex e;
    de_katex_loss = Latex.tex_loss e; de_reads = reads_of t }
      | FExpr e -> TvExpr (dexpr env pos e)) }

let time_input env (f : func_decl) : D.time_input_decl =
  let pos = with_binders top f.findices in
  { ti_name = f.fname; ti_indices = bindings f.findices; ti_kind = forcing_kind_name f.fkind;
    ti_unit = unit_lit f.funit;
    ti_args = List.map (forcing_arg env pos) f.fargs;
    ti_text = text env f.fspan }

let transition env (t : Ast.transition_decl) : D.transition_decl =
  let pos = with_binders top t.trindices in
  { tn_name = t.trname; tn_indices = bindings t.trindices;
    tn_from = List.map (stoich env pos) t.trsrc;
    tn_to = (match t.trdst with
      | DstSum refs -> D.DstSum (List.map (stoich env pos) refs)
      | DstBranch arms ->
        D.DstBranch (List.map (fun (r, w) ->
          { D.ba_to = stoich env pos r; ba_weight = dexpr env pos w }) arms));
    tn_dynamics = (match t.trdyn with
      | Rate e -> DynRate (dexpr env pos e)
      | Via (law, kvs) -> DynVia (law, args env pos kvs));
    tn_guard = Option.map guard t.trguard;
    tn_lineage = t.trlineage;
    tn_expands_to =
      Some (Expander.expanded_transitions env.ctx ~declared:t.trname);
    tn_text = text env t.trspan }

let initial env (i : init_entry) : D.initial_decl =
  let pos = with_binders top i.ibindings in
  { in_compartment = i.icomp; in_indices = items env pos i.iindices;
    in_bindings = bindings i.ibindings;
    in_value = (match i.ivalue with
      | IVExpr e -> IvExpr (dexpr env pos e)
      | IVLaw (l, _) -> IvLaw (distribution env pos l));
    in_text = text env i.ispan }

(* A compile that succeeded has every stream's columns, measurement and
   projection: a missing one is a diagnostic, and no section is built. *)
let required what stream = function
  | Some v -> v
  | None ->
    failwith (Printf.sprintf
      "internal error: observation stream '%s' has no %s after a successful \
       compile" stream what)

let observation env (o : obs_decl) : D.observation_decl =
  let pos = with_binders top o.oindices in
  let columns = required "columns block" o.oname o.ocolumns in
  let meas = required "measurement" o.oname o.omeasurement in
  let proj = required "projection" o.oname o.oprojection in
  let aux_cols =
    List.filter_map (fun (c : Ast.obs_column) ->
      match c.oc_role with
      | ColValue _ when c.oc_name <> meas.om_scored -> Some c.oc_name
      | _ -> None) columns in
  let lik_pos = { pos with aux_cols } in
  { ob_name = o.oname; ob_indices = bindings o.oindices; ob_source = o.osource;
    ob_columns = List.map (fun (c : Ast.obs_column) ->
      { D.oc_name = c.oc_name;
        oc_role = (match c.oc_role with
          | ColTime -> CrTime
          | ColDim d -> CrDim d
          | ColValue k -> CrValue (param_kind k)
          | ColWindowStart -> CrWindowStart
          | ColWindowStop -> CrWindowStop) }) columns;
    ob_measurement = { ms_scored = meas.om_scored;
                       ms_likelihood = distribution env lik_pos meas.om_lik };
    ob_projection = (match proj with
      | ProjIncidence (tr, idx) -> PjIncidence (tr, items env pos idx)
      | ProjDerived e -> PjDerived (dexpr env { pos with reporting = true } e));
    ob_emit_schedule = Option.map (schedule_core env pos) o.oschedule;
    ob_covers = Option.map (fun (c : obs_covers) ->
      (* The span may name a data column for a per-row width. *)
      let span_pos = { pos with aux_cols = List.map (fun (c : Ast.obs_column) ->
        c.oc_name) columns } in
      { D.cv_form = c.ocv_form; cv_column = c.ocv_col;
        cv_span = Option.map (fun p -> dexpr env span_pos (Time_pos.expr p)) c.ocv_span }) o.ocovers;
    ob_expands_to =
      Some (Expander.expanded_observations env.ctx ~declared:o.oname);
    ob_text = text env o.ospan }

let action env pos : action_decl -> D.action = function
  | ATransfer kvs -> AcTransfer (args env pos kvs)
  | ASet (c, idx, e) ->
    AcSet { as_compartment = c; as_indices = items env pos idx;
            as_value = dexpr env pos e }
  | AAdd (c, idx, e) ->
    AcAdd { aa_compartment = c; aa_indices = items env pos idx;
            aa_count = dexpr env pos e }

let intervention env (iv : Ast.intervention_decl) : D.intervention_decl =
  let pos = with_binders top iv.ivindices in
  { iv_name = iv.ivname; iv_indices = bindings iv.ivindices;
    iv_actions = List.map (action env pos) iv.ivaction;
    iv_schedule = (match iv.ivschedule with
      | SAtTimes ps -> ShAtTimes (List.map (fun p -> dexpr env pos (Time_pos.expr p)) ps)
      | SRecurring { every; from; until } ->
        ShRecurring { sh_every = dexpr env pos (Time_pos.expr every);
                      sh_from = Option.map (fun p -> dexpr env pos (Time_pos.expr p)) from;
                      sh_to = Option.map (fun p -> dexpr env pos (Time_pos.expr p)) until }
      | SEveryAtDay { every; at_day } ->
        ShEveryAtDay { sh_period = dexpr env pos (Time_pos.expr every);
                       sh_at_day = dexpr env pos (Time_pos.expr at_day) });
    iv_guard = Option.map guard iv.ivguard;
    iv_text = text env iv.ivspan }

let rec trigger env pos : trig_pred -> D.trigger_pred = function
  | TgAtom e -> D.TgAtom (dexpr env pos e)
  | TgAnd (a, b) -> D.TgAnd (trigger env pos a, trigger env pos b)
  | TgOr (a, b) -> D.TgOr (trigger env pos a, trigger env pos b)
  | TgNot a -> D.TgNot (trigger env pos a)

(* `once = true | false`: the compiler matches the two words by spelling. *)
let once env pos (e : expr) : D.declared_expr =
  match e with
  | EIdent (s, loc) when loc <> dummy_loc && is_boolean_word s ->
    let t = D.TIdent { nr_name = s; nr_kind = KKeyword } in
    { de_tree = t; de_katex = Latex.tex e;
    de_katex_loss = Latex.tex_loss e; de_reads = reads_of t }
  | _ -> dexpr env pos e

let reactive env (r : Ast.reactive_decl) : D.reactive_decl =
  (* A trigger reads streams by name: `observed(weekly_afp)`. *)
  let pos = { (with_binders top r.rxindices) with reporting = true } in
  { rc_name = r.rxname; rc_indices = bindings r.rxindices;
    rc_when = trigger env pos r.rxwhen;
    rc_after = Option.map (fun p -> dexpr env pos (Time_pos.expr p)) r.rxafter;
    rc_once = Option.map (once env pos) r.rxonce;
    rc_cooldown = Option.map (fun p -> dexpr env pos (Time_pos.expr p)) r.rxcooldown;
    rc_action = action env pos r.rxaction;
    rc_guard = Option.map guard r.rxguard;
    rc_text = text env r.rxspan }

let scenario env (s : Ast.scenario_decl) : D.scenario_decl =
  let assigns = List.map (fun (n, e) ->
    { D.an_name = n; an_value = dexpr env top e }) in
  { sn_name = s.scname;
    sn_fields = List.map (function
      | ScLabel l -> D.SfLabel l
      | ScEnable ns -> SfEnable ns
      | ScDisable ns -> SfDisable ns
      | ScSet kvs -> SfSet (assigns kvs)
      | ScScale kvs -> SfScale (assigns kvs)
      | ScCompose ns -> SfCompose ns
      | ScTEnd p -> SfTEnd (dexpr env top (Time_pos.expr p))
      | ScExtends p -> SfExtends p) s.scfields;
    sn_text = text env s.scspan }

let simulation env (s : simulate_decl) : D.simulation_decl =
  { sm_from = dexpr env top (Time_pos.expr s.sim_from);
    sm_to = dexpr env top (Time_pos.expr s.sim_to);
    sm_dt = Option.map (fun p -> dexpr env top (Time_pos.expr p)) s.sim_dt;
    sm_integrator = Option.map fst s.sim_integrator;
    sm_atol = Option.map (fun (e, _) -> dexpr env top e) s.sim_atol;
    sm_rtol = Option.map (fun (e, _) -> dexpr env top e) s.sim_rtol;
    sm_text = text env s.sim_span }

let output env (o : Ast.output_decl) : D.output_decl =
  { ou_trajectories = Option.map (fun (t : output_traj_decl) ->
      { D.to_schedule = schedule_core env top t.otschedule;
        to_format = t.otformat }) o.out_trajectories;
    ou_text = text env o.out_span }

let quantity env (q : Ast.quantity_decl) : D.quantity_decl =
  let pos = { (with_binders top q.qd_indices) with reporting = true } in
  { qd_name = q.qd_name; qd_indices = bindings q.qd_indices;
    qd_body = dexpr env pos q.qd_body;
    qd_text = text env q.qd_span }

let contrast env (c : Ast.contrast_decl) : D.contrast_decl =
  { ct_name = c.cd_name;
    ct_body = dexpr env { top with reporting = true } c.cd_body;
    ct_text = text env c.cd_span }

(* ── The files a compile read ────────────────────────────────────────────── *)

(* [path] as a path relative to the directory [from_dir], by name only (no
   symbolic link is resolved): both are made absolute against the working
   directory, `.` and `..` segments are folded, and the common prefix is
   replaced by the `..` steps that leave [from_dir]. A committed artifact
   never holds an absolute path. *)
let relative_between ~(from_dir : string) (path : string) : string =
  let absolute p =
    if Filename.is_relative p then Filename.concat (Sys.getcwd ()) p else p in
  let normalize p =
    List.fold_left (fun acc seg ->
      match seg, acc with
      | ("" | "."), _ -> acc
      | "..", _ :: rest -> rest
      | "..", [] -> []
      | s, _ -> s :: acc) [] (String.split_on_char '/' (absolute p))
    |> List.rev in
  let rec strip a b =
    match a, b with
    | x :: a', y :: b' when x = y -> strip a' b'
    | _ -> (a, b) in
  let from_rest, path_rest = strip (normalize from_dir) (normalize path) in
  String.concat "/" (List.map (fun _ -> "..") from_rest @ path_rest)

(* A data file as the model wrote it, which is relative to the model file. A
   path the model wrote as absolute is rewritten relative to the model's
   directory. *)
let model_relative ~(source_dir : string) (as_written : string) : string =
  if Filename.is_relative as_written then as_written
  else relative_between ~from_dir:(if source_dir = "" then "." else source_dir)
         as_written

let files (ctx : Expander.context) : D.file_input list =
  List.map (fun (as_written, resolved) ->
    { D.fi_path = model_relative ~source_dir:ctx.source_dir as_written;
      fi_sha256 = Expander.file_sha256 ctx ~resolved })
    (Expander.reads ctx)

(* ── The section ─────────────────────────────────────────────────────────── *)

(** Builds the declared-model section, and lists every name in it that could
    not be classified, in source order. [ctx] is the context a successful
    [Expander.expand_detail] returned for [decls]. [source] holds the text the
    declarations were parsed from, and is where each declaration's text as
    written is read back. The section's compile inputs hold the data files
    the compile read and no [--set] overrides; {!with_compile_inputs} adds
    those and the [--quantities] file. It fills [Latex]'s table of [@symbol]
    overrides from [decls] as a side effect. *)
let build_detail ~(source : Source_cache.t) (ctx : Expander.context)
    (decls : declaration list)
    : D.declared_model * unresolved list =
  Latex.populate_overrides decls;
  let each f = List.concat_map f decls in
  let comps = each (function
    | DCompartments cs -> List.map (fun (c : Ast.compartment_decl) -> c.cname) cs
    | _ -> []) in
  let dim_entries = each (function DDimensions ds -> ds | _ -> []) in
  let dims = List.map (fun (d : dimensions_entry) -> d.dename) dim_entries in
  let levels = Hashtbl.create 64 in
  List.iter (fun d ->
    List.iter (fun l -> Hashtbl.replace levels l ())
      (Expander.dim_levels_or_empty ctx d)) dims;
  let env = {
    ctx; source; comps; dims; levels;
    site = ref ""; unresolved = ref [];
    quantities = each (function
      | DQuantities qs -> List.map (fun (q : Ast.quantity_decl) -> q.qd_name) qs
      | _ -> []);
    streams = each (function
      | DObservations os -> List.map (fun (o : obs_decl) -> o.oname) os
      | _ -> []);
    transitions = each (function
      | DTransitions ts -> List.map (fun (t : Ast.transition_decl) -> t.trname) ts
      | _ -> []);
  } in
  let first f = List.find_map f decls in
  (* Convert one declaration with [env.site] naming it, so an unclassified
     name is reported against the declaration it sits in. *)
  let at what name f x =
    env.site := (if name = "" then what else Printf.sprintf "%s '%s'" what name);
    f env x in
  let section : D.declared_model = {
    dm_schema_version = D.schema_version;
    dm_compile_inputs = { ci_set_values = []; ci_files = files ctx };
    dm_time_unit = first (function DTimeUnit (u, _) -> Some (unit_lit u) | _ -> None);
    dm_description = first (function DDescription (s, _) -> Some s | _ -> None);
    dm_origin = first (function DOrigin (s, _) -> Some s | _ -> None);
    dm_dimensions = List.map (dimension env) dim_entries;
    dm_compartments = each (function
      | DCompartments cs -> List.map (compartment env) cs | _ -> []);
    dm_stratify = each (function
      | DStratify s ->
        [ { D.st_dim = s.sdim; st_only = s.sonly; st_text = text env s.sspan } ]
      | _ -> []);
    dm_parameters = each (function
      | DParameters ps ->
        List.map (fun p ->
          let name = match p with
            | PScalar { pname; _ } | PIndexed { pname; _ } -> pname in
          at "parameter" name parameter p) ps
      | _ -> []);
    dm_definitions = each (function
      | DLet lb -> [ at "let" lb.lname definition lb ] | _ -> []);
    dm_tables = each (function
      | DTables ts ->
        List.map (fun (t : Ast.table_decl) ->
          at "table" (String.concat ", " t.tnames) table t) ts
      | _ -> []);
    dm_time_inputs = each (function
      | DForcing fs ->
        List.map (fun (f : func_decl) -> at "forcing" f.fname time_input f) fs
      | _ -> []);
    dm_timepoints = each (function
      | DTimepoints tps ->
        List.map (fun (tp : Ast.timepoint_decl) ->
          at "timepoint" tp.tpname (fun env (tp : Ast.timepoint_decl) ->
            { D.tp_name = tp.tpname; tp_time = dexpr env top (Time_pos.expr tp.tptime);
              tp_text = text env tp.tpspan }) tp) tps
      | _ -> []);
    dm_transitions = each (function
      | DTransitions ts ->
        List.map (fun (t : Ast.transition_decl) ->
          at "transition" t.trname transition t) ts
      | _ -> []);
    dm_ode_equations = each (function
      | DODE os ->
        List.map (fun (o : Ast.ode_decl) ->
          at "ODE equation for" o.ocomp (fun env (o : Ast.ode_decl) ->
            { D.od_compartment = o.ocomp;
              od_derivative = dexpr env top o.oderiv;
              od_text = text env o.odspan }) o) os
      | _ -> []);
    dm_balance = each (function
      | DBalance b ->
        [ at "balance for" b.bcomp (fun env (b : Ast.balance_decl) ->
            { D.bl_compartment = b.bcomp; bl_expr = dexpr env top b.bexpr;
              bl_text = text env b.bspan }) b ]
      | _ -> []);
    dm_initial = each (function
      | DInit is ->
        List.map (fun (i : init_entry) ->
          at "initial condition for" i.icomp initial i) is
      | _ -> []);
    dm_observations = each (function
      | DObservations os ->
        List.map (fun (o : obs_decl) ->
          at "observation stream" o.oname observation o) os
      | _ -> []);
    dm_interventions = each (function
      | DInterventions ivs ->
        List.map (fun (iv : Ast.intervention_decl) ->
          at "intervention" iv.ivname intervention iv) ivs
      | _ -> []);
    dm_events = each (function
      | DEvents ivs ->
        List.map (fun (iv : Ast.intervention_decl) ->
          at "event" iv.ivname intervention iv) ivs
      | _ -> []);
    dm_reactive_interventions = each (function
      | DReactiveInterventions rs ->
        List.map (fun (r : Ast.reactive_decl) ->
          at "reactive intervention" r.rxname reactive r) rs
      | _ -> []);
    dm_scenarios = each (function
      | DScenarios ss ->
        List.map (fun (s : Ast.scenario_decl) ->
          at "scenario" s.scname scenario s) ss
      | _ -> []);
    dm_simulation = first (function
      | DSimulate s -> Some (at "the simulate block" "" simulation s) | _ -> None);
    dm_output = first (function
      | DOutput o -> Some (at "the output block" "" output o) | _ -> None);
    dm_quantities = each (function
      | DQuantities qs ->
        List.map (fun (q : Ast.quantity_decl) ->
          at "quantity" q.qd_name quantity q) qs
      | _ -> []);
    dm_contrasts = each (function
      | DContrasts cs ->
        List.map (fun (c : Ast.contrast_decl) ->
          at "contrast" c.cd_name contrast c) cs
      | _ -> []);
  } in
  (section, List.rev !(env.unresolved))

(** The declared-model section alone. *)
let build ~source ctx decls : D.declared_model =
  fst (build_detail ~source ctx decls)

(** Add what only the compiler's command line knows: the `--set` overrides, in
    the order given, and a `--quantities` vocabulary file. [model_dir] is the
    directory of the model file; the vocabulary path is recorded relative to
    it. *)
let with_compile_inputs ~(model_dir : string)
    ~(set_values : (string * float) list)
    ~(quantities_file : string option) (d : D.declared_model) : D.declared_model =
  let extra = match quantities_file with
    | None -> []
    | Some path ->
      let sha = match Sha256.hex_of_file path with
        | h -> Some h
        | exception Sys_error _ -> None in
      [ { D.fi_path = relative_between ~from_dir:model_dir path; fi_sha256 = sha } ]
  in
  { d with dm_compile_inputs = {
      ci_set_values = List.map (fun (n, v) ->
        { D.sv_name = n; sv_value = v }) set_values;
      ci_files = d.dm_compile_inputs.ci_files @ extra } }
