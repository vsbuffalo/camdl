(* The census of time positions (proposal
   docs/dev/proposals/2026-11-16-calendar-durations-by-position.md §3.2-3.3).

   [census decls] lists every time position in a model: every declaration field
   typed ['k Ast.Time_pos.t], every forcing keyword whose role is a time, the
   time argument of a temporal reduction (`value_at`), the `window` of a
   `sum_observed` trigger, and the two syntactic positions inside ordinary
   expressions (the operand compared with model time `t`, and an offset added
   to or subtracted from `t`).

   Why it cannot miss a declaration. The census destructures every
   [Ast.declaration] constructor and every record it reads with no wildcard and
   no `; _` in a record pattern, and the census section of this file turns
   warnings 4 (fragile match) and 9 (missing record fields) into errors, beside
   warning 8 (non-exhaustive match), which the library's dune flags already
   make one. Adding a constructor to [Ast.declaration], or a field to a record
   read here, therefore fails the OCaml build until the census says whether the
   new thing holds a time and of which kind.

   To repeat the check by hand (it cannot be a test that runs): add a scratch
   constructor `| DScratch of expr` to [Ast.declaration] and run `dune build`
   in `ocaml/`; the build must fail with warning 8, as an error, in
   `decl_found` here. Add a scratch field to [Ast.simulate_decl] instead and it
   must fail with warning 9, as an error, in `simulate_found` (and in the
   parser, which constructs the record). Both were checked when the census was
   written.

   Test 1.1 (`test_time_census_coverage_*` in ocaml/test/test_compiler.ml)
   compiles models and checks that every position listed here was lowered by
   the expander's [Lower] function for its kind. *)

open Ast

(* ── Temporal reductions (§3.2) ────────────────────────────────────────────── *)

type reduction =
  | Final | Mean | Integral
  | Count_above | Count_below
  | Value_at
  | Time_of_max | Time_of_min
  | First_above | First_below | Last_above | Last_below

(* The role of each positional argument of a reduction. A [Time] argument is a
   time position of the given kind; the quantity classifier obtains it as a
   typed position from [reduction_time_args], never as a bare [expr]. *)
type arg_role =
  | Series
  | Threshold
  | Time of Time_pos.instant Time_pos.kind

let all_reductions =
  [ Final; Mean; Integral; Count_above; Count_below; Value_at;
    Time_of_max; Time_of_min; First_above; First_below; Last_above; Last_below ]

let reduction_name = function
  | Final       -> "final"
  | Mean        -> "mean"
  | Integral    -> "integral"
  | Count_above -> "count_above"
  | Count_below -> "count_below"
  | Value_at    -> "value_at"
  | Time_of_max -> "time_of_max"
  | Time_of_min -> "time_of_min"
  | First_above -> "first_above"
  | First_below -> "first_below"
  | Last_above  -> "last_above"
  | Last_below  -> "last_below"

let reduction_of_name n =
  List.find_opt (fun r -> reduction_name r = n) all_reductions

let reduction_args = function
  | Final | Mean | Integral | Time_of_max | Time_of_min -> [Series]
  | Count_above | Count_below | First_above | First_below
  | Last_above | Last_below -> [Series; Threshold]
  | Value_at -> [Series; Time Time_pos.Instant]

(* The time-role arguments of a reduction call, as typed positions. The
   location is the declaration's: an argument carries no span of its own. *)
let reduction_time_args ~loc (fn : string) (args : (string * expr) list)
  : Time_pos.instant Time_pos.t list =
  match reduction_of_name fn with
  | None -> []
  | Some r ->
    let positional = List.filter_map (fun (k, e) -> if k = "" then Some e else None) args in
    let rec zip roles es =
      match roles, es with
      | Time k :: roles, e :: es -> Time_pos.make k e loc :: zip roles es
      | (Series | Threshold) :: roles, _ :: es -> zip roles es
      | [], _ | _, [] -> []
    in
    zip (reduction_args r) positional

(* ── Forcing keyword roles (§3.2) ──────────────────────────────────────────── *)

module Forcing_schema = struct
  type kwarg_role =
    | Instants       (* a list of instants, one position per entry: `breakpoints`, `times` *)
    | Instant_pair   (* an interval of instants: bspline `window = [a, b]` *)
    | Phase_ranges   (* a list of phase ranges: periodic `on = [lo : hi, ...]` *)
    | Phase_value    (* one phase: sinusoidal `phase` *)
    | Length_value   (* one length: `period`, `step`, `lag` *)
    | Value          (* a model value: `amplitude`, `values`, `coefs`, ... *)
    | Count          (* a compile-time count: `n_basis`, `degree`, `harmonics` *)
    | Enum           (* a closed enum: `method`, `link` *)
    | File           (* a data-file path or column: `data`, `time_col`, ... *)
    | Name           (* a model name: `table`, `time_dim` *)

  (* The keywords each kind accepts, with their roles. `lag` applies to every
     known kind and is not listed here. *)
  let keys : forcing_kind -> (string * kwarg_role) list option = function
    | FSinusoidal ->
      Some [ "amplitude", Value; "period", Length_value; "phase", Phase_value;
             "baseline", Value ]
    | FPiecewise -> Some [ "breakpoints", Instants; "values", Value ]
    | FInterpolated ->
      Some [ "method", Enum; "data", File; "time_col", File; "value_col", File;
             "key_col", File; "table", Name; "time_dim", Name;
             "times", Instants; "values", Value ]
    | FPeriodic ->
      Some [ "period", Length_value; "on", Phase_ranges; "step", Length_value;
             "values", Value ]
    | FFourier -> Some [ "period", Length_value; "harmonics", Count ]
    | FPeriodicSpline ->
      Some [ "period", Length_value; "n_basis", Count; "degree", Count;
             "coefs", Value; "link", Enum ]
    | FBspline ->
      Some [ "window", Instant_pair; "breakpoints", Instants; "n_basis", Count;
             "degree", Count; "link", Enum; "coefs", Value ]
    | FUnknown _ -> None

  let role (k : forcing_kind) (key : string) : kwarg_role option =
    match keys k with
    | None -> None
    | Some ks -> if key = "lag" then Some Length_value else List.assoc_opt key ks

  let is_time_role = function
    | Instants | Instant_pair | Phase_ranges | Phase_value | Length_value -> true
    | Value | Count | Enum | File | Name -> false

  (* The entries of a list-valued keyword: `[a, b]` is two, a bare `a` is one. *)
  let entries (e : expr) : expr list =
    match e with EList es -> es | _ -> [e]

  let instant_entries ~loc (e : expr) : Time_pos.instant Time_pos.t list =
    List.map (fun x -> Time_pos.make Time_pos.Instant x loc) (entries e)

  (* One-position constructors for the expander's forcing sites, so that a
     site builds exactly the position (kind, location, expression) the census
     lists. *)
  let phase ~loc e = Time_pos.make Time_pos.Phase e loc
  let length ~loc e = Time_pos.make Time_pos.Length e loc

  (* The endpoints of the `on` ranges: `lo : hi` gives two, a lone entry one. *)
  let phase_endpoints ~loc (e : expr) : Time_pos.phase Time_pos.t list =
    List.concat_map (fun x ->
      match x with
      | ERange (lo, hi) -> [Time_pos.make Time_pos.Phase lo loc; Time_pos.make Time_pos.Phase hi loc]
      | EConst _ | EUnit _ | EIdent _ | EIndex _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _
      | ECond _ | EFuncCall _ | EList _ | EObsAccess _ | ERunMember _ ->
        [Time_pos.make Time_pos.Phase x loc]) (entries e)

  (* Every time position among one forcing's keyword arguments. *)
  let positions (fd : func_decl) : Time_pos.any list =
    let { fname = _; findices = _; fkind; funit = _; fargs; fspan } = fd in
    List.concat_map (fun (key, fv) ->
      match fv, role fkind key with
      | FStr _, _ | FExpr _, None -> []
      | FExpr e, Some r ->
        (match r with
         | Instants | Instant_pair ->
           List.map (fun p -> Time_pos.Any p) (instant_entries ~loc:fspan e)
         | Phase_ranges ->
           List.map (fun p -> Time_pos.Any p) (phase_endpoints ~loc:fspan e)
         | Phase_value -> [Time_pos.Any (Time_pos.make Time_pos.Phase e fspan)]
         | Length_value -> [Time_pos.Any (Time_pos.make Time_pos.Length e fspan)]
         | Value | Count | Enum | File | Name -> [])) fargs

  (* The keyword expressions that are not time positions, for the walks over
     ordinary expressions. *)
  let value_exprs (fd : func_decl) : expr list =
    List.filter_map (fun (key, fv) ->
      match fv with
      | FStr _ -> None
      | FExpr e ->
        (match role fd.fkind key with
         | Some r when is_time_role r -> None
         | Some _ | None -> Some e)) fd.fargs
end

(* ── Positions inside ordinary expressions (§3.2) ──────────────────────────── *)

let is_comparison = function
  | Lt | Le | Gt | Ge | Eq | Neq -> true
  | Add | Sub | Mul | Div | Pow -> false

(* [a ⋈ b] is [b ⋈' a]. *)
let flip_comparison = function
  | Lt -> Gt | Gt -> Lt | Le -> Ge | Ge -> Le
  | (Eq | Neq | Add | Sub | Mul | Div | Pow) as op -> op

(* Which names hold an instant. The census reads it from the declarations and
   the expander from its context, so the two make the same positions. *)
type instant_env = {
  instant_name : string -> bool;     (* an `instant`-kind parameter *)
  let_body     : string -> expr option;
}

(* Does [e] denote an instant (a point on the axis, not a duration)? `origin`,
   `date(...)`, `add_calendar_*`, model time `t`, an observation anchor, an
   `instant` parameter, a `let` of one, and an instant moved by a duration. *)
let is_instant_operand (env : instant_env) (e : expr) : bool =
  let rec go depth e =
    depth < 32 &&
    match e with
    | EIdent (("origin" | "t" | "first_obs" | "last_obs"), _) -> true
    | EIdent (n, _) | EIndex (n, _, _) ->
      env.instant_name n
      || (match env.let_body n with Some b -> go (depth + 1) b | None -> false)
    | EFuncCall (("date" | "add_calendar_months" | "add_calendar_years"), _) -> true
    | EBinOp (Add, a, b) -> go (depth + 1) a || go (depth + 1) b
    | EBinOp (Sub, a, b) -> go (depth + 1) a && not (go (depth + 1) b)
    | EUnOp (Total, a) -> go (depth + 1) a
    | ECond (_, a, b) -> go (depth + 1) a && go (depth + 1) b
    | EConst _ | EUnit _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _ | EFuncCall _ | EList _
    | ERange _ | EObsAccess _ | ERunMember _ -> false
  in
  go 0 e

(* How one operand of a comparison holds model time `t`, and so what the other
   operand is.
   - [Shifted]: `t`, or `t` moved by a duration (`t ± d`, `d + t`). The other
     operand is a date.
   - [Elapsed]: the time since an instant `X`, `t - X` (or `-(X - t)`). The
     other operand is a duration `D`, and `t - X ⋈ D` is `t ⋈ X + D`.
   - [Remaining]: the time until an instant `X`, `X - t` (or `-(t - X)`).
     `X - t ⋈ D` is `t ⋈' X - D`, with the comparison flipped. *)
type time_side =
  | Shifted of loc
  | Elapsed of { since : expr; tloc : loc }
  | Remaining of { until : expr; tloc : loc }

let time_side env (e : expr) : time_side option =
  let inst = is_instant_operand env in
  match e with
  | EIdent ("t", l) -> Some (Shifted l)
  | EBinOp (Sub, EIdent ("t", l), x)
  | EUnOp (Neg, EBinOp (Sub, x, EIdent ("t", l))) when inst x ->
    Some (Elapsed { since = x; tloc = l })
  | EBinOp (Sub, x, EIdent ("t", l))
  | EUnOp (Neg, EBinOp (Sub, EIdent ("t", l), x)) when inst x ->
    Some (Remaining { until = x; tloc = l })
  | EBinOp ((Add | Sub), EIdent ("t", l), _) | EBinOp (Add, _, EIdent ("t", l)) -> Some (Shifted l)
  | EConst _ | EUnit _ | EIdent _ | EIndex _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _
  | ECond _ | EFuncCall _ | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> None

(* A comparison with model time, read as a comparison of `t` with a date: the
   date (an instant position), the location of `t`, and the comparison with
   the date replaced by a lowered expression for it. For a [Shifted] side the
   date is the other operand as written, and the comparison keeps its shape;
   for an elapsed or remaining time it is `X + D` or `X - D`, and the
   comparison becomes `t ⋈ date`. *)
type time_comparison = { date : expr; tloc : loc; with_date : expr -> expr }

let time_comparison env (e : expr) : time_comparison option =
  match e with
  | EBinOp (op, l, r) when is_comparison op ->
    (* [side] holds `t`, [other] is what it is compared with, and [op'] is the
       comparison read with [side] on the left. *)
    let read side other op' rebuild_shifted =
      match time_side env side with
      | Some (Shifted tloc) -> Some { date = other; tloc; with_date = rebuild_shifted }
      | Some (Elapsed { since; tloc }) ->
        Some { date = EBinOp (Add, since, other); tloc;
               with_date = (fun d -> EBinOp (op', EIdent ("t", tloc), d)) }
      | Some (Remaining { until; tloc }) ->
        Some { date = EBinOp (Sub, until, other); tloc;
               with_date = (fun d -> EBinOp (flip_comparison op', EIdent ("t", tloc), d)) }
      | None -> None
    in
    (match read l r op (fun d -> EBinOp (op, l, d)) with
     | Some c -> Some c
     | None -> read r l (flip_comparison op) (fun d -> EBinOp (op, d, r)))
  | EConst _ | EUnit _ | EIdent _ | EIndex _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _
  | ECond _ | EFuncCall _ | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> None

(* The positions a node itself makes, with their sub-expression: a comparison
   with model time makes its date an instant position ([time_comparison]);
   `t ± d` / `d + t` makes [d] a runtime offset. Syntactic: `t` is the
   reserved model-time name (`reserved_time_names` in the expander), so no
   binding can shadow it. *)
let node_positions env (e : expr) : Time_pos.any list =
  match time_comparison env e with
  | Some { date; tloc; _ } -> [Time_pos.Any (Time_pos.make Time_pos.Instant date tloc)]
  | None ->
    (match e with
     | EBinOp ((Add | Sub), EIdent ("t", l), d) | EBinOp (Add, d, EIdent ("t", l)) ->
       [Time_pos.Any (Time_pos.make Time_pos.Runtime_offset d l)]
     | EConst _ | EUnit _ | EIdent _ | EIndex _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _
     | ECond _ | EFuncCall _ | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> [])

(* The direct sub-expressions of a node that are NOT time positions: the
   children the walks over ordinary expressions (E321 Rule 1, E360) descend
   into. A position's own expression is judged by its [Lower] function, so the
   walk must not judge it a second time. *)
let ordinary_children env (e : expr) : expr list =
  match e with
  | EBinOp (op, l, _) when is_comparison op && time_side env l <> None -> [l]
  | EBinOp (op, _, r) when is_comparison op && time_side env r <> None -> [r]
  | EBinOp ((Add | Sub), (EIdent ("t", _) as t), _) | EBinOp (Add, _, (EIdent ("t", _) as t)) -> [t]
  | EFuncCall (fn, args) ->
    (match reduction_of_name fn with
     | None -> Time_typing.subexprs e
     | Some r ->
       let rec keep roles args =
         match roles, args with
         | Time _ :: roles, ("", _) :: args -> keep roles args
         | (Series | Threshold) :: roles, ("", a) :: args -> a :: keep roles args
         | roles, (k, a) :: args when k <> "" -> a :: keep roles args
         | _, (_, a) :: args -> a :: keep [] args
         | _, [] -> []
       in
       keep (reduction_args r) args)
  | _ -> Time_typing.subexprs e

(* Every `t`-comparison and `t`-offset position in an expression, at any depth
   (a position's own expression is searched too). *)
let rec expression_positions env (e : expr) : Time_pos.any list =
  node_positions env e @ List.concat_map (expression_positions env) (Time_typing.subexprs e)

(* The time arguments of every reduction in an expression, at any depth. *)
let rec reduction_positions ~loc (e : expr) : Time_pos.any list =
  let here = match e with
    | EFuncCall (fn, args) ->
      List.map (fun p -> Time_pos.Any p) (reduction_time_args ~loc fn args)
    | EConst _ | EUnit _ | EIdent _ | EIndex _ | EBinOp _ | EUnOp _ | ESum _ | ESumStages _
    | ECond _ | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> []
  in
  here @ List.concat_map (reduction_positions ~loc) (Time_typing.subexprs e)

(* A prior's location arguments: for an `instant` parameter these are dates
   (`normal(mu = …)`, `uniform(lower = …, upper = …)`); its scale arguments
   (`sigma`, `sd`, `rate`, …) are lengths and keep the average month. *)
let prior_location_args = ["mu"; "mean"; "lower"; "upper"]

(* `window = w` of every `sum_observed(...)` in a trigger predicate: a span. *)
let trigger_windows ~loc (p : trig_pred) : Time_pos.span Time_pos.t list =
  let rec in_expr (e : expr) =
    match e with
    | EFuncCall ("sum_observed", args) ->
      List.filter_map (fun (k, a) ->
        if k = "window" then Some (Time_pos.make Time_pos.Span a loc) else None) args
    | EBinOp (_, a, b) -> in_expr a @ in_expr b
    | EUnOp (_, a) -> in_expr a
    | EConst _ | EUnit _ | EIdent _ | EIndex _ | ESum _ | ESumStages _ | ECond _ | EFuncCall _
    | EList _ | ERange _ | EObsAccess _ | ERunMember _ -> []
  in
  let rec go = function
    | TgAtom e -> in_expr e
    | TgAnd (a, b) | TgOr (a, b) -> go a @ go b
    | TgNot a -> go a
  in
  go p

(* The names of the `instant`-kind parameters, for the values given to them. *)
let instant_param_names (decls : declaration list) : string list =
  List.concat_map (fun d ->
    match d with
    | DParameters ps ->
      List.filter_map (fun pd ->
        match pd with
        | PScalar { pname; pkind; _ } | PIndexed { pname; pkind; _ } ->
          if pkind = PInstant then Some pname else None) ps
    | _ -> []) decls

(* Which names hold an instant, from the declarations. *)
let instant_env_of (decls : declaration list) : instant_env =
  let instant_params = instant_param_names decls in
  let lets = List.filter_map (fun d ->
      match d with DLet lb -> Some (lb.lname, lb.lbody) | _ -> None) decls in
  { instant_name = (fun n -> List.mem n instant_params);
    let_body = (fun n -> List.assoc_opt n lets) }

(* ── The census (§3.3) ─────────────────────────────────────────────────────── *)

(* From here on, a fragile match (4) or a record pattern that omits a field (9)
   is an error: this is what makes a new declaration, or a new field, fail the
   build until the census accounts for it. The expression helpers above are
   not under it: [Ast.expr] and its operators are not where positions are
   declared. *)
[@@@warning "@4@9"]

(* What a declaration holds: [lowered], the positions an expansion site lowers
   through its kind's [Lower] function (declaration fields, forcing keywords,
   reduction arguments, trigger windows); and [in_expr], the ordinary
   expressions whose `t`-comparisons and `t`-offsets are positions. No site
   folds those on its own (the enclosing expression is lowered whole), so the
   expander judges them from the census; which they are depends on which names
   hold an instant ([instant_env]), so they are found once the whole model has
   been read. *)
type found = {
  lowered : Time_pos.any list;
  in_expr : expr list;
  (* A value given to an `instant` parameter (a scenario's `set`, a prior's
     location argument), with the name diagnostics give it. Resolved as an
     ordinary expression where it is used, so, like [in_expr], judged from
     here. *)
  values  : (string * Time_pos.instant Time_pos.t) list;
}

let none = { lowered = []; in_expr = []; values = [] }
let ( ++ ) a b =
  { lowered = a.lowered @ b.lowered; in_expr = a.in_expr @ b.in_expr;
    values = a.values @ b.values }
let cat fs = List.fold_left ( ++ ) none fs
let any p = Time_pos.Any p
let field p = { lowered = [any p]; in_expr = []; values = [] }
let fields ps = { lowered = List.map any ps; in_expr = []; values = [] }
let opt f = function None -> none | Some x -> f x
let in_exprs es = { lowered = []; in_expr = es; values = [] }

let action_exprs (a : action_decl) : expr list =
  match a with
  | ATransfer kws -> List.map snd kws
  | ASet (_, items, e) | AAdd (_, items, e) ->
    e :: List.map (function IPosn x | INamed (_, x) | IStage x -> x) items

let schedule_core_found (s : schedule_core) : found =
  match s with
  | SchedEvery p -> field p
  | SchedAt ps -> fields ps

let schedule_found (s : schedule_decl) : found =
  match s with
  | SAtTimes ps -> fields ps
  | SRecurring { every; from; until } -> field every ++ opt field from ++ opt field until
  | SEveryAtDay { every; at_day } -> field every ++ field at_day

let bounds_found (b : bounds) : found =
  match b with
  | Instant_bounds (lo, hi) -> field lo ++ field hi
  | Length_bounds (lo, hi) -> field lo ++ field hi
  | Value_bounds (lo, hi) -> in_exprs [lo; hi]

let prior_found ~pname ~pkind ~ploc (ps : prior_spec) : found =
  let { ps_name = _; ps_args; ps_pool_over = _ } = ps in
  cat (List.map (fun (k, e) ->
    match pkind with
    | PInstant when List.mem k prior_location_args ->
      { none with values = [ (Printf.sprintf "the prior on '%s' (`%s`)" pname k,
                              Time_pos.make Time_pos.Instant e ploc) ] }
    | PInstant | PRate | PProbability | PPositive | PCount | PReal | PDuration ->
      in_exprs [e]) ps_args)

let param_found (pd : param_decl) : found =
  match pd with
  | PScalar { pname; pkind; pdim = _; pbounds; pprior; pdoc = _; ploc; pspan = _ } ->
    opt bounds_found pbounds ++ opt (prior_found ~pname ~pkind ~ploc) pprior
  | PIndexed { pname; pdims = _; pkind; pdim = _; pbounds; pprior; pdoc = _; ploc; pspan = _ } ->
    opt bounds_found pbounds ++ opt (prior_found ~pname ~pkind ~ploc) pprior

let transition_found (tr : transition_decl) : found =
  let { trname = _; trindices = _; trsrc = _; trdst; trdyn; trguard = _;
        trlineage = _; trdoc = _; trloc = _; trspan = _ } = tr in
  let dst_exprs = match trdst with
    | DstSum _ -> []
    | DstBranch brs -> List.map snd brs
  in
  in_exprs (trans_dynamics_exprs trdyn @ dst_exprs)

let obs_found (od : obs_decl) : found =
  let { oname = _; oindices = _; osource = _; ocolumns = _; omeasurement;
        oprojection; oschedule; ocovers; odoc = _; oloc = _; ospan = _ } = od in
  let covers = match ocovers with
    | None -> none
    | Some { ocv_form = _; ocv_col = _; ocv_span; ocv_loc = _ } -> opt field ocv_span
  in
  let measurement = match omeasurement with
    | None -> none
    | Some { om_scored = _; om_lik; om_accuracy = _ } -> in_exprs (List.map snd (likelihood_kwargs om_lik))
  in
  let projection = match oprojection with
    | None | Some (ProjIncidence _) -> none
    | Some (ProjDerived e) -> in_exprs [e]
  in
  opt schedule_core_found oschedule ++ covers ++ measurement ++ projection

let intervention_found (iv : intervention_decl) : found =
  let { ivname = _; ivindices = _; ivaction; ivschedule; ivguard = _; ivloc = _; ivspan = _ } = iv in
  schedule_found ivschedule ++ in_exprs (List.concat_map action_exprs ivaction)

let reactive_found (rx : reactive_decl) : found =
  let { rxname = _; rxindices = _; rxwhen; rxafter; rxonce; rxcooldown; rxaction;
        rxguard = _; rxloc; rxspan = _ } = rx in
  fields (trigger_windows ~loc:rxloc rxwhen)
  ++ opt field rxafter ++ opt field rxcooldown
  ++ opt (fun e -> in_exprs [e]) rxonce
  ++ in_exprs (action_exprs rxaction)

let simulate_found (sd : simulate_decl) : found =
  let { sim_from; sim_to; sim_to_loc = _; sim_dt; sim_integrator = _; sim_atol;
        sim_rtol; sim_span = _ } = sd in
  field sim_from ++ field sim_to ++ opt field sim_dt
  ++ opt (fun (e, _) -> in_exprs [e]) sim_atol
  ++ opt (fun (e, _) -> in_exprs [e]) sim_rtol

let output_found (od : output_decl) : found =
  let { out_trajectories; out_span = _ } = od in
  match out_trajectories with
  | None -> none
  | Some { otschedule; otformat = _ } -> schedule_core_found otschedule

let scenario_found ~instant_params (sc : scenario_decl) : found =
  let { scname; scfields; scspan } = sc in
  cat (List.map (fun f ->
    match f with
    | ScTEnd p -> field p
    | ScSet kvs ->
      cat (List.map (fun (k, e) ->
        if List.mem k instant_params then
          { none with values = [ (Printf.sprintf "scenario '%s' `set` of '%s'" scname k,
                                  Time_pos.make Time_pos.Instant e scspan) ] }
        else in_exprs [e]) kvs)
    | ScScale kvs -> in_exprs (List.map snd kvs)
    | ScLabel _ | ScEnable _ | ScDisable _ | ScCompose _ | ScExtends _ -> none) scfields)

let forcing_found (fd : func_decl) : found =
  { lowered = Forcing_schema.positions fd; in_expr = []; values = [] }
  ++ in_exprs (Forcing_schema.value_exprs fd)

let decl_found ~instant_params (d : declaration) : found =
  match d with
  | DTimeUnit _ | DDescription _ | DModelDoc _ | DOrigin _ | DDimensions _
  | DCompartments _ | DStratify _ -> none
  | DParameters ps -> cat (List.map param_found ps)
  | DTables tds ->
    cat (List.map (fun { tnames = _; tdims = _; tcell_kind = _; tvalue; tloc = _; tspan = _ } ->
      in_exprs [tvalue]) tds)
  | DForcing fs -> cat (List.map forcing_found fs)
  | DTransitions trs -> cat (List.map transition_found trs)
  | DObservations obs -> cat (List.map obs_found obs)
  | DInterventions ivs | DEvents ivs -> cat (List.map intervention_found ivs)
  | DReactiveInterventions rxs -> cat (List.map reactive_found rxs)
  | DODE odes ->
    cat (List.map (fun { ocomp = _; oderiv; odspan = _ } -> in_exprs [oderiv]) odes)
  | DOutput od -> output_found od
  | DSimulate sd -> simulate_found sd
  | DInit ies ->
    cat (List.map (fun { icomp = _; iindices = _; ibindings = _; ivalue; iloc = _; ispan = _ } ->
      in_exprs (init_value_exprs ivalue)) ies)
  | DTimepoints tps ->
    cat (List.map (fun { tpname = _; tptime; tpspan = _ } -> field tptime) tps)
  | DLet { lname = _; lindices = _; lshape = _; lkind = _; ldim = _; lloc = _; lspan = _;
           lbody; ldoc = _ } -> in_exprs [lbody]
  | DScenarios ss -> cat (List.map (scenario_found ~instant_params) ss)
  | DBalance { bcomp = _; bexpr; bspan = _ } -> in_exprs [bexpr]
  | DQuantities qs ->
    cat (List.map (fun { qd_name = _; qd_indices = _; qd_body; qd_doc = _; qd_loc; qd_span = _ } ->
      { lowered = reduction_positions ~loc:qd_loc qd_body; in_expr = []; values = [] }
      ++ in_exprs [qd_body]) qs)
  | DContrasts cs ->
    cat (List.map (fun { cd_name = _; cd_body; cd_doc = _; cd_loc = _; cd_span = _ } ->
      in_exprs [cd_body]) cs)

let found_of (decls : declaration list) : found =
  let instant_params = instant_param_names decls in
  cat (List.map (decl_found ~instant_params) decls)

let in_expr_positions decls (f : found) =
  List.concat_map (expression_positions (instant_env_of decls)) f.in_expr

(* Every time position in the model. *)
let census (decls : declaration list) : Time_pos.any list =
  let f = found_of decls in
  f.lowered @ in_expr_positions decls f @ List.map (fun (_, p) -> any p) f.values

(* The `t`-comparison and `t`-offset positions alone. *)
let in_expression_positions (decls : declaration list) : Time_pos.any list =
  in_expr_positions decls (found_of decls)

(* The values given to `instant` parameters, with their names. *)
let value_positions (decls : declaration list) : (string * Time_pos.instant Time_pos.t) list =
  (found_of decls).values
