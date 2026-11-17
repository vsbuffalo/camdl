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

   E360 (gh#1070) refuses the arithmetic an Instant does not have:
   Instant + Instant, k · Instant, Duration − Instant, an Instant
   moved by a bare number, and a conditional whose branches are an
   Instant and a duration (gh#1096). See `walk_instant_misuse`.

   In unanchored mode (no `origin` declared) the rules are vacuous:
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
  | EUnOp ((Neg | Total), a) -> classify env a
  | EUnOp _        -> TOther
  | ECond (_, a, b) -> lub_dur (classify env a) (classify env b)
  | ESum (_, _, _, body, _) -> classify env body
  | EList _ | ERange _ -> TOther
  | EObsAccess _ -> TOther
  | ERunMember _ -> TOther   (* a contrast operand — not a time/duration *)

(* ── Sink walk: Rule 1 at Instant ± Duration nodes ──────────────────── *)

(* The direct sub-expressions of a node, for the walks below that judge
   each node on its own and then descend. Observation accesses and
   contrast operands are leaves. *)
let subexprs (e : expr) : expr list =
  match e with
  | EConst _ | EUnit _ | EIdent _ | EObsAccess _ | ERunMember _ -> []
  | EIndex (_, items, _) ->
    List.map (function IPosn e | INamed (_, e) -> e) items
  | EBinOp (_, l, r) | ERange (l, r) -> [l; r]
  | EUnOp (_, a) -> [a]
  | ECond (p, a, b) -> [p; a; b]
  | ESum (_, _, _, body, _) -> [body]
  | EFuncCall (_, args) -> List.map snd args
  | EList items -> items

(* Walk an expression and call [on_hit] at every Add/Sub node whose
   operands trigger the rule: Instant ± Calendar-duration in
   anchored mode.

   The walk descends through Cond branches, ESum bodies, BinOp
   sub-arms, etc. — anywhere a sub-expression may itself be an
   Instant±Duration expression. *)
let rec walk_rule1 ?(children = subexprs) env ~on_hit (e : expr) : unit =
  (match e with
   | EBinOp ((Add | Sub) as _op, l, r) ->
     let cl = classify env l and cr = classify env r in
     (* Two possible orientations: Instant + duration, duration + Instant. *)
     (match cl, cr with
      | TInstant, TCalendar | TCalendar, TInstant ->
        on_hit ~lhs:l ~rhs:r
      | _ -> ())
   | _ -> ());
  List.iter (walk_rule1 ~children env ~on_hit) (children e)

(* ── Instant arithmetic: the forms with no meaning (E360, gh#1070) ──── *)

(* Under an origin an Instant is a point on the calendar. The algebra is
   a torsor: an Instant moves by a duration, and the difference of two
   Instants is a duration. Nothing else is defined. `classify` answers
   these forms with TInstant (or TOther) and lets them through, so the
   refusal is a separate walk.

   The check fires only on a *definite* Instant. `classify` treats model
   time `t` (and any parameter without a time kind) as TOther, so
   `t0 - t` classifies as TInstant although it is a duration in the
   modeller's eyes. Refusing `k * (t0 - t)` on that guess would reject a
   rate that means something. A definite Instant is one built only from
   `date(...)`, `origin`, the calendar primitives, `instant`-kind
   parameters and `let`s of those, moved only by classified durations.
   An Instant moved by a unitless number is refused at that node and is
   not definite for the node above it, so one malformed term gives one
   diagnostic.

   A unitless offset is judged by dimension, not spelling: the caller
   passes [~unitless], true of a constant expression whose dimension is
   1 (`5`, `5 * 2`, `0.5 * 10`). The expander passes the same
   `const_expr_dim` test the observation-anchor offsets use, so the two
   paths share one rule. *)

type instant_misuse =
  | InstantPlusInstant       (* Instant + Instant *)
  | SumOfInstants            (* sum(i in d, <instant>) *)
  | ScaledInstant            (* k · Instant, Instant / k, Instant ^ k, −Instant *)
  | DurationMinusInstant     (* Duration − Instant *)
  | InstantPlusBareNumber    (* Instant ± 5, Instant + 5 * 2, 5 + Instant *)
  | InstantOrDuration        (* if c then Instant else Duration (gh#1096) *)

(* A numeric literal as spelled, for the W324/W325 warnings, which are
   about how a time position is written. E360 judges by dimension
   instead (see [~unitless] above). *)
let is_bare_number (e : expr) : bool =
  match e with
  | EConst _ | EUnOp (Neg, EConst _) -> true
  | _ -> false

let is_classified_duration = function
  | TExact | TCalendar -> true
  | TInstant | TOther  -> false

let rec is_definite_instant env (e : expr) : bool =
  match e with
  | EIdent (name, _) | EIndex (name, _, _) ->
    (match Hashtbl.find_opt env.let_tbl name with
     | Some lb -> is_definite_instant env lb.lbody
     | None -> classify env e = TInstant)
  | EBinOp (Add, l, r) ->
    (is_definite_instant env l && is_classified_duration (classify env r))
    || (is_definite_instant env r && is_classified_duration (classify env l))
  | EBinOp (Sub, l, r) ->
    is_definite_instant env l && is_classified_duration (classify env r)
  | ECond (_, a, b) -> is_definite_instant env a && is_definite_instant env b
  | EFuncCall _ -> classify env e = TInstant
  | _ -> false

(* The misuse at this node, if any. Only the node itself is judged; the
   walk below visits every sub-expression. *)
let instant_misuse_at env ~unitless (e : expr) : instant_misuse option =
  let definite = is_definite_instant env in
  match e with
  | EBinOp (Add, l, r) when definite l && definite r -> Some InstantPlusInstant
  | EBinOp ((Add | Sub), l, r) when definite l && unitless r ->
    Some InstantPlusBareNumber
  | EBinOp (Add, l, r) when unitless l && definite r ->
    Some InstantPlusBareNumber
  | EBinOp (Sub, l, r)
    when definite r
         && (is_classified_duration (classify env l) || unitless l) ->
    Some DurationMinusInstant
  | EBinOp ((Mul | Div | Pow), l, r) when definite l || definite r ->
    Some ScaledInstant
  | EUnOp (Neg, a) when definite a -> Some ScaledInstant
  | ESum (_, _, _, body, _) when definite body -> Some SumOfInstants
  (* `classify` takes the upper bound of a conditional's branches, so an
     Instant on one side and a duration on the other reports TInstant and
     passes every rule above (gh#1096). The two branches must be the same
     kind of time. *)
  | ECond (_, a, b)
    when (definite a && is_classified_duration (classify env b))
         || (definite b && is_classified_duration (classify env a)) ->
    Some InstantOrDuration
  | _ -> None

let rec walk_instant_misuse ?(children = subexprs) env ~unitless ~on_hit (e : expr) : unit =
  (match instant_misuse_at env ~unitless e with
   | Some m -> on_hit m e
   | None -> ());
  List.iter (walk_instant_misuse ~children env ~unitless ~on_hit) (children e)

let describe_instant_misuse = function
  | InstantPlusInstant    -> "adds two instants"
  | SumOfInstants         -> "adds instants in a sum"
  | ScaledInstant         -> "scales an instant"
  | DurationMinusInstant  -> "subtracts an instant from a duration"
  | InstantPlusBareNumber -> "moves an instant by a bare number"
  | InstantOrDuration     -> "chooses between an instant and a duration"

let hint_instant_misuse = function
  | InstantPlusInstant | SumOfInstants ->
    "an instant is a point on the calendar; two of them do not add. \
     The difference of two instants is a duration (date(b) - date(a)), \
     and an instant plus a duration is an instant (origin + 30 'days)."
  | ScaledInstant ->
    "an instant is a point on the calendar, not a quantity, so it cannot \
     be multiplied, divided, raised to a power or negated. Scale a \
     duration instead: origin + 2 * (date(b) - date(a)), or 2 * 30 'days."
  | DurationMinusInstant ->
    "subtracting an instant from a duration has no meaning. Write the \
     instant first: date(b) - 7 'days is an instant, and date(b) - date(a) \
     is a duration."
  | InstantPlusBareNumber ->
    "a bare number has no unit, so the offset would silently be read in \
     the model's time_unit. State the unit: date(\"2020-03-01\") + 5 'days."
  | InstantOrDuration ->
    "the branches of a conditional must be the same kind of time: both \
     instants (if c then date(a) else date(b)) or both durations. To move \
     an instant by a chosen span, put the conditional on the duration: \
     origin + (if c then 5 'days else 7 'days)."

(* ── Public diagnostic hints ────────────────────────────────────────── *)

(* [span] is the affine day-equivalent of the offending calendar
   duration paired with a rendering of the duration itself — e.g.
   [(1826, "5 'years")]. The caller folds it, because the days-per-unit
   constants live with the expander's unit conversion and must not be
   restated here.

   [exact_fn] is the calendar-exact primitive matching the offending
   literal's unit — [add_calendar_years] for a `'years`-flavoured
   duration, [add_calendar_months] otherwise. Naming the wrong one
   reproduces the very failure this hint exists to prevent: a modeller
   who pastes [add_calendar_months(d, 5)] for `5 'years` gets a horizon
   twelve times short, in a model that compiles and runs (gh#843).

   [None] when the duration is not a compile-time constant. The hint
   then states the rule without a span: a suggestion the modeller can
   paste has to be *this* model's span, and a guessed one is worse than
   none — pasting a wrong span yields a model that compiles and runs
   over the wrong horizon. *)
(* Which calendar-exact primitive matches this duration's unit.
   `Calendar` arises only from `'months`/`'years` literals, so a
   duration is years-flavoured iff it mentions `'years` and no
   `'months`; a mixed expression has no single right answer and keeps
   the months default. *)
let calendar_exact_fn (e : expr) : string =
  let saw_years = ref false and saw_months = ref false in
  let rec go e =
    match e with
    | EUnit (_, Years)  -> saw_years := true
    | EUnit (_, Months) -> saw_months := true
    | EBinOp (_, l, r)  -> go l; go r
    | EUnOp (_, x)      -> go x
    | _ -> ()
  in
  go e;
  if !saw_years && not !saw_months then "add_calendar_years" else "add_calendar_months"

let hint_calendar_plus_instant ?(exact_fn = "add_calendar_months") (span : (int * string) option) =
  let rule =
    Printf.sprintf
      "calendar months/years aren't invertible spans \
       (e.g. date(\"2021-01-31\") + 1 month = date(\"2021-02-28\") because \
       day-31 clamps to day-28 in Feb 2021). \
       For a calendar-exact date use %s(d, N)." exact_fn
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
   Annotate each with a date: `to = add_calendar_months(origin, 600)` or \
   `to = date(\"1940-12-01\")`; a calendar duration such as `600 'months` \
   in one of these positions is refused (E321). \
   Typed positions are unaffected: rate parameters declared with \
   'per_month continue to work, and lengths such as `dt = 1 'months` \
   continue to mean the average month (≈ 30.44 days)."

(* E322 on a calendar cadence. [months] is its whole number of calendar
   months, if it has one; [span] its affine day-equivalent, as for E321's
   hint. The `date_range` form compiles at every cadence position: an
   intervention's, an output's and an emit schedule's `at` list all splice it. *)
let hint_calendar_cadence ~(months : int option) (span : (int * string) option) =
  let months = match months with Some m -> string_of_int m | None -> "n" in
  let fixed = match span with
    | Some (days, rendered) ->
      Printf.sprintf "or write a fixed cadence: `every = %d 'days` (≈ %s)." days rendered
    | None -> "or write a fixed cadence in 'days or 'weeks."
  in
  Printf.sprintf
    "a calendar cadence counts whole months from its start, which one fixed \
     step cannot represent. List the calendar dates instead, with \
     `date_range(<start>, <end>, calendar_months = %s)` in an `at` list, %s"
    months fixed

let hint_calendar_offset_in_at_day =
  "an `at_day` offset is one fixed span from the start of every period, \
   but a calendar month or year is 28-31 or 365-366 days depending on \
   which one it is. Write the offset in 'days or 'weeks, or list the \
   calendar-aligned firings explicitly via at = [date(...), ...]."

let hint_calendar_width_in_covers =
  "a `covers` width is one fixed span for every row, but a calendar month \
   or year is 28-31 or 365-366 days depending on which one it is. For \
   calendar periods give the file `window_start` and `window_stop` columns, \
   which state each row's own boundaries; for a fixed span write the width \
   in 'days or 'weeks."

let hint_calendar_span_in_trigger span =
  let rule =
    "a trigger's `window`, `after` and `cooldown` are each folded to one \
     fixed span, but a calendar month or year is 28-31 or 365-366 days \
     depending on which one it is."
  in
  match span with
  | Some (days, rendered) ->
    Printf.sprintf "%s State it in 'days or 'weeks: %d 'days (≈ %s)."
      rule days rendered
  | None -> rule ^ " State it in 'days or 'weeks, which are fixed spans."

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

let hint_bare_numeric_value_at =
  "bare number as `value_at`'s time under `origin = date(...)` is \
   interpreted as the model's `time_unit` from origin. \
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
  | ECond (_, a, b) ->
    Printf.sprintf "(if … then %s else %s)" (show_short a) (show_short b)
  | _ -> "…"
