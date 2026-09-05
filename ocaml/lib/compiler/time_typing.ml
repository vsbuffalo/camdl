(* Surface-level time typing — the typed-time / calendar-duration
   classifier on the DSL AST.

   See `docs/dev/proposals/2026-05-22-typed-time-and-dsl-ergonomics.md`
   §3 (Rule 1) and §3.4 (recurring-schedule cadences).

   This pass runs BEFORE `resolve_expr` rewrites unit-literal nodes
   into bare numeric IR constants, so it can still see whether a
   leaf came from a `'months`/`'years` literal or from a `'days`
   / `'weeks` literal / parameter reference.

   The classifier rides alongside the existing dimcheck pass (which
   operates on the post-expansion IR). At the IR level, every
   duration is just an `f64` axis-units value — provenance is lost.
   This pass captures provenance at the surface and emits any
   diagnostics here; the IR pass continues to handle the
   numerical-dimension story unchanged.

   Three classes of values:

     - [TExact]:    a duration derived only from exact-day primitives
                    (`'days`, `'weeks`, `Instant − Instant`,
                    `duration`-kind / `[T]`-annotated param refs).
     - [TCalendar]: a duration whose synthesised type contains a
                    `'months` or `'years` literal leaf.
     - [TInstant]:  an Instant (origin-relative point) —
                    `date(...)`, `origin`, or a let-laundered
                    equivalent.
     - [TOther]:    anything else — non-duration, non-Instant
                    (rates, scalars, populations, …). Time-typing
                    rules don't fire on these.

   Rule 1 fires at exactly one site: an `Instant ± Duration`
   subexpression whose duration side is classified `TCalendar`. The
   single `let`-bound case `let d = 6 'months; date(...) + d` flows
   through the LUB propagation rule below because `let` references
   classify recursively through the binding's body.

   In unanchored mode (no `origin` declared) the rule is vacuous:
   we never emit a diagnostic, because the torsor refinement is
   inactive — see proposal §1.1, decision of record. *)

open Ast

(* ── Classification lattice ─────────────────────────────────────────── *)

type tclass =
  | TExact
  | TCalendar
  | TInstant
  | TOther

(* Least upper bound on the Exact <: Calendar lattice, used at
   arithmetic nodes (Add, Sub, Min, Max). *)
let lub_dur a b =
  match a, b with
  | TCalendar, _ | _, TCalendar -> TCalendar
  | TExact, TExact              -> TExact
  | TExact, TOther | TOther, TExact -> TExact
  | TOther, TOther              -> TOther
  | TInstant, _ | _, TInstant   -> TInstant  (* shouldn't normally reach here *)

(* ── Bound let bindings: an env for recursive classification ─────────── *)

(* We need to look up `let` bindings during classification so that
   `let d = 6 'months; date(...) + d` triggers the rule via
   inheritance of `d`'s classifier from its body. The let-table is
   available on the context AFTER `build_lookup_tables`; our caller
   threads it in. *)

type env = {
  let_tbl : (string, let_binding) Hashtbl.t;
  param_decls : param_decl list;
  origin_set : bool;  (* whether `origin = date(...)` is declared *)
}

let env_of_ctx ~let_tbl ~param_decls ~origin_set =
  { let_tbl; param_decls; origin_set }

(* ── Param kind lookup (for refs to duration/instant-kind params) ────── *)

let param_kind_of_name env name =
  let kind_and_dim = List.find_map (fun pd ->
    match pd with
    | PScalar  { pname; pkind; pdim; _ } when pname = name -> Some (pkind, pdim)
    | PIndexed { pname; pkind; pdim; _ } when pname = name -> Some (pkind, pdim)
    | _ -> None
  ) env.param_decls in
  match kind_and_dim with
  | None -> None
  | Some (PInstant,  _)             -> Some `Instant
  | Some (PDuration, _)             -> Some `Duration
  | Some (_, Some (0, 1))           -> Some `Duration   (* [T]-annotated *)
  | Some (_, _)                     -> None

(* ── The classifier proper ──────────────────────────────────────────── *)

(* Classify a surface expression, returning its time-class. This is
   pure (no diagnostics emitted here) — sinks call this and decide
   what to do with the result.

   Notes:
   - We classify ONLY by structure of the expression. Same
     expression in two contexts gives the same answer; bounds on a
     parameter declaration carry their own classifier, never leaking
     into uses of that parameter (proposal §3.3.2 "one-line
     invariant").
   - Indexed-param references behave identically to scalar refs for
     kind purposes.
   - Function calls other than `date(...)` are conservatively
     classified `TOther`. *)
let rec classify env (e : expr) : tclass =
  match e with
  | EConst _ -> TOther
  | EUnit (_, Days)    | EUnit (_, Weeks)    -> TExact
  | EUnit (_, Months)  | EUnit (_, Years)    -> TCalendar
  | EUnit (_, _)                              -> TOther
  | EIdent ("origin", _) ->
    (* Phase 2 of the 2026-05-22 typed-time proposal §1.1: the
       reserved `origin` identifier is an Instant in anchored
       mode. In unanchored mode the expander emits E327 at the
       resolution site; classifying as TInstant here is correct
       either way — the classifier is structural, and an
       unanchored `origin` reference is malformed in any context. *)
    TInstant
  | EIdent (name, _) ->
    (* Lookup order:
       1. let-binding → classify its body in our own env
          (recursion is bounded because let-bindings are acyclic by
          construction in the AST; cycle detection happens
          elsewhere, not our concern here)
       2. parameter declaration → use its kind/[T] annotation
       3. otherwise → TOther *)
    (match Hashtbl.find_opt env.let_tbl name with
     | Some lb -> classify env lb.lbody
     | None ->
       match param_kind_of_name env name with
       | Some `Instant  -> TInstant
       | Some `Duration -> TExact
       | None           -> TOther)
  | EIndex (name, _, _) ->
    (* Indexed param/let — kind doesn't depend on the index. *)
    (match Hashtbl.find_opt env.let_tbl name with
     | Some lb -> classify env lb.lbody
     | None ->
       match param_kind_of_name env name with
       | Some `Instant  -> TInstant
       | Some `Duration -> TExact
       | None           -> TOther)
  | EFuncCall ("date", _) -> TInstant
  | EFuncCall (("add_calendar_months" | "add_calendar_years"), _) ->
    (* Phase 2 §4: calendar-arithmetic primitives produce an
       Instant. Classified structurally, irrespective of whether the
       expander successfully const-evaluated the call — a malformed
       call still has Instant type from the user's perspective. *)
    TInstant
  | EFuncCall _ -> TOther
  | EBinOp (Sub, l, r) ->
    let cl = classify env l and cr = classify env r in
    (match cl, cr with
     | TInstant, TInstant -> TExact   (* Instant - Instant = exact duration *)
     | TInstant, _        -> TInstant (* Instant - duration = Instant *)
     | _, TInstant        -> TOther   (* duration - Instant: malformed, fall through *)
     | _, _               -> lub_dur cl cr)
  | EBinOp (Add, l, r) ->
    let cl = classify env l and cr = classify env r in
    (match cl, cr with
     | TInstant, _ | _, TInstant -> TInstant
     | _, _ -> lub_dur cl cr)
  (* Note: Ast.bin_op has no Min/Max — those exist in the IR but
     not in the surface AST. Cmp ops (Eq, Lt, …) return bool, never
     reach this code as durations. *)
  | EBinOp (Mul, l, r) | EBinOp (Div, l, r) ->
    (* scalar × duration (or vice versa) preserves the duration's
       class; duration / duration is dimensionless (TOther) but
       Rule 1 only cares about Add/Sub so we don't need to be
       precise. *)
    let cl = classify env l and cr = classify env r in
    (match cl, cr with
     | TCalendar, _ | _, TCalendar -> TCalendar
     | TExact, _ | _, TExact       -> TExact
     | _                            -> TOther)
  | EBinOp _ -> TOther
  | EUnOp (Neg, a) -> classify env a
  | EUnOp _        -> TOther
  | ECond (_, a, b) -> lub_dur (classify env a) (classify env b)
  | ESum (_, _, _, body, _) -> classify env body
  | EList _ | ERange _ -> TOther
  | EObsAccess _ -> TOther
  | ERunMember _ -> TOther   (* a contrast operand — not a time/duration *)

(* ── Sink walk: Rule 1 at Instant ± Duration nodes ──────────────────── *)

(* Walk an expression and call [on_hit] at every Add/Sub node whose
   operands trigger the rule: Instant ± Calendar-duration in
   anchored mode.

   The walk descends through Cond branches, ESum bodies, BinOp
   sub-arms, etc. — anywhere a sub-expression may itself be an
   Instant±Duration expression. *)
let rec walk_rule1 env ~on_hit (e : expr) : unit =
  (match e with
   | EBinOp ((Add | Sub) as _op, l, r) ->
     let cl = classify env l and cr = classify env r in
     (* Two possible orientations: Instant + duration, duration + Instant. *)
     (match cl, cr with
      | TInstant, TCalendar | TCalendar, TInstant ->
        on_hit ~lhs:l ~rhs:r
      | _ -> ())
   | _ -> ());
  walk_subexprs env ~on_hit e

and walk_subexprs env ~on_hit (e : expr) : unit =
  match e with
  | EConst _ | EUnit _ | EIdent _ -> ()
  | EIndex (_, items, _) ->
    List.iter (fun ii ->
      match ii with
      | IPosn e          -> walk_rule1 env ~on_hit e
      | INamed (_, e)    -> walk_rule1 env ~on_hit e
    ) items
  | EBinOp (_, l, r) ->
    walk_rule1 env ~on_hit l; walk_rule1 env ~on_hit r
  | EUnOp (_, a) -> walk_rule1 env ~on_hit a
  | ECond (p, a, b) ->
    walk_rule1 env ~on_hit p;
    walk_rule1 env ~on_hit a;
    walk_rule1 env ~on_hit b
  | ESum (_, _, _, body, _) -> walk_rule1 env ~on_hit body
  | EFuncCall (_, args) ->
    List.iter (fun (_, e) -> walk_rule1 env ~on_hit e) args
  | EList items -> List.iter (walk_rule1 env ~on_hit) items
  | ERange (a, b) ->
    walk_rule1 env ~on_hit a; walk_rule1 env ~on_hit b
  | EObsAccess _ -> ()
  | ERunMember _ -> ()   (* leaf — no sub-expressions to walk *)

(* ── Public diagnostic hints ────────────────────────────────────────── *)

(* [span] is the affine day-equivalent of the offending calendar
   duration paired with a rendering of the duration itself — e.g.
   [(1826, "5 'years")]. The caller folds it, because the days-per-unit
   constants live with the expander's unit conversion and must not be
   restated here.

   [None] when the duration is not a compile-time constant. The hint
   then states the rule without a span: a suggestion the modeller can
   paste has to be *this* model's span, and a guessed one is worse than
   none — pasting a wrong span yields a model that compiles and runs
   over the wrong horizon. *)
let hint_calendar_plus_instant (span : (int * string) option) =
  let rule =
    "calendar months/years aren't invertible spans \
     (e.g. date(\"2021-01-31\") + 1 month = date(\"2021-02-28\") because \
     day-31 clamps to day-28 in Feb 2021). \
     For a calendar-exact date use add_calendar_months(d, N)."
  in
  match span with
  | Some (days, rendered) ->
    Printf.sprintf
      "%s For an explicit affine span use %d 'days (≈ %s)."
      rule days rendered
  | None ->
    rule ^ " For an explicit affine span state the offset in 'days or \
            'weeks, which are fixed spans."

let hint_time_unit_months_with_origin =
  "constant-day axis required for calendar-anchored models. \
   Switch to time_unit = 'days (or 'weeks). \
   When you switch the axis, every *bare-numeric* time position in your \
   model silently changes meaning to the new axis: \
   simulate { from / to }, at [...] schedules on interventions and events, \
   and the time column of any --data file. \
   Annotate each with a unit literal (e.g. to = 600 'months) or a date \
   literal (e.g. to = date(\"1940-12-01\")) to preserve intent. \
   Typed positions are unaffected: rate parameters declared with \
   'per_month continue to work, and duration values like 1 'months \
   continue to work as affine spans (≈ 30.44 days)."

let hint_calendar_cadence_in_recurring =
  "calendar cadence not allowed in an anchored recurring schedule \
   (months/years are average lengths, not invertible spans). \
   Use every = 30 'days for an affine ~monthly recurrence, or list \
   the calendar-aligned firings explicitly via at = [date(...), ...]."

let hint_bare_numeric_on_periodic =
  "bare-numeric entries in `on=[...]` of a periodic forcing under an \
   anchored model are interpreted as internal-time units from origin. \
   Use date(...) entries for calendar-aligned breakpoints, or — if you \
   really mean internal-time offsets — annotate the entries with 'days."

let hint_bare_numeric_simulate =
  "bare number in a time position under `origin = date(...)` is \
   interpreted as the model's `time_unit` from origin. To make the \
   intent explicit, write `<n> 'days` (or another unit literal) or a \
   date literal like `date(\"YYYY-MM-DD\")`. Annotate with `'days` to \
   suppress this warning intentionally."

let hint_bare_numeric_at_schedule =
  "bare number in an `at [...]` schedule entry under `origin = date(...)` \
   is interpreted as the model's `time_unit` from origin. \
   Annotate with `'days` or use `date(\"YYYY-MM-DD\")` to make the \
   intent explicit."

let hint_bare_numeric_data_column =
  "the --data time column is numeric and the model declares \
   `origin = date(...)` — values are interpreted as internal-time units \
   from origin. If that's intentional, pass `--time-format internal-days` \
   to suppress this warning; if you meant calendar dates, switch the \
   column to ISO YYYY-MM-DD form."

(* ── Helper: pretty-printer for the offending sub-expression ─────────── *)

(* Reuse the shared expression printer the dimcheck uses. We don't
   want a second printer for the same job. The format below is a
   coarse approximation good enough for error messages — the real
   printer lives in `Pp_expr` and isn't AST-shaped, so we re-render
   here. *)
let rec show_short (e : expr) : string =
  match e with
  | EConst f ->
    if Float.is_integer f && Float.abs f < 1e9
    then string_of_int (Float.to_int f)
    else Printf.sprintf "%g" f
  | EUnit (f, u) ->
    let us = match u with
      | Days -> "days" | Weeks -> "weeks"
      | Months -> "months" | Years -> "years"
      | PerDay -> "per_day" | PerWeek -> "per_week"
      | PerMonth -> "per_month" | PerYear -> "per_year"
      | Count -> "count" | Ratio -> "ratio"
    in
    let n =
      if Float.is_integer f && Float.abs f < 1e9
      then string_of_int (Float.to_int f)
      else Printf.sprintf "%g" f
    in
    Printf.sprintf "%s '%s" n us
  | EIdent (s, _) -> s
  | EIndex (n, _, _) -> n ^ "[…]"
  | EBinOp (op, l, r) ->
    let os = match op with
      | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
      | Pow -> "^" | Eq -> "==" | Neq -> "!="
      | Lt -> "<" | Gt -> ">" | Le -> "<=" | Ge -> ">="
    in
    Printf.sprintf "(%s %s %s)" (show_short l) os (show_short r)
  | EUnOp (Neg, a) -> "-" ^ show_short a
  | EUnOp (_, a)   -> show_short a
  | EFuncCall (n, _) -> n ^ "(…)"
  | _ -> "…"
