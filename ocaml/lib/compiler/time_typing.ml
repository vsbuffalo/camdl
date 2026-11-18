(** The rules for time in a model anchored to the calendar: which expressions
    are instants and which are durations, and which arithmetic on them has no
    meaning.

    A model that declares [origin = date("...")] is anchored: model time [0]
    is that date, and [date(...)] and [origin] are instants, points on the
    calendar. An instant moves by a duration, and the difference of two
    instants is a duration; nothing else is defined. Durations come in two
    classes. One written in ['days] or ['weeks] is an exact number of days.
    One written in ['months] or ['years] is calendar-valued: the compiler
    converts it to an average length (a month is 365.2425/12 days), which is
    not the distance from one calendar date to the same date a month later,
    so moving an instant by it would give a date the author did not mean.
    The one exception is a date position, such as [simulate { to }] or an
    [at] entry, where one calendar step from a known date is that calendar
    date; there the expander's [Calendar_reading] decides, in place of
    {!walk_rule1}.

    {@camdl[
      time_unit = 'days
      origin = date("2020-01-01")
      let start = date("2020-03-01")      # an instant
      let span  = start - origin          # an exact duration
      let later = start + 26 'weeks       # an instant moved by an exact duration
    ]}

    Writing [start + 6 'months] instead is E321.

    These distinctions are made on the syntax tree, where an instant is
    still visible: name resolution turns [date(...)] and [origin] into plain
    numbers of time units since the origin, so in the IR an instant cannot be
    told from a duration. The expander uses this module in two places, and
    emits the diagnostics itself. [Expander.check_surface_time_typing] runs
    the walks below over every ordinary expression (transition rates, ODE
    derivatives, [let] bodies, initial values, actions, priors and others)
    after the declaration and [let]-cycle checks and before any expression is
    resolved, skipping the time positions inside them through the walks'
    [children] argument. The lowering function of each time position in
    [Expander.Lower] calls the classifier and the walks on its own position
    when it judges it. This module emits nothing: it supplies {!classify}, which assigns
    an expression a {!tclass}; {!walk_rule1}, which finds an instant moved by
    a calendar duration (E321); {!walk_instant_misuse}, which finds the
    arithmetic an instant does not have (E360, an {!instant_misuse}); and the
    text of the hints the expander attaches.

    Every rule applies only to an anchored model. Without an origin there are
    no instants, and a model with [time_unit = 'months] and per-month rates is
    left alone.

    Classification is structural. A name bound by a [let] takes the class of
    the [let]'s body, so [let d = 6 'months] carries its class to each use; a
    parameter of kind [instant] is an instant, and one of kind [duration] or
    with [dim T] an exact duration. Model time [t], other parameters and every
    function call except [date], [add_calendar_months] and
    [add_calendar_years] are {!TOther}, so an expression built from them is
    never refused. *)

open Ast

(* ── Classification lattice ─────────────────────────────────────────── *)

(** The kind of time an expression denotes, read from how it is written. *)
type tclass =
  | TExact
      (** A duration of an exact number of days: built from ['days] or
          ['weeks] literals, [duration]-kind or [dim T] parameters, or the
          difference of two instants. *)
  | TCalendar
      (** A duration that mentions a ['months] or ['years] literal anywhere
          in it, and so is an average rather than an exact span. *)
  | TInstant
      (** A point on the calendar: [date(...)], [origin],
          [add_calendar_months(...)], [add_calendar_years(...)], an
          [instant]-kind parameter, or one of these moved by a duration. *)
  | TOther
      (** Anything else: a rate, a count, a number, model time [t]. No rule
          applies to it. *)

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

(** What classification needs from the model: its [let]s, so that a name
    takes the class of its definition, and its parameter declarations, for
    their kinds. The expander builds it from its context once the lookup
    tables exist.

    [classify] and [is_definite_instant] follow a [let] into its body and
    stop at a [let] already being followed, which then has the class of an
    expression with no time meaning ([TOther], not definite): a [let]
    defined in terms of itself is left in [let_tbl], and the expander
    reports it (E201) whichever check reaches it first (gh#1211). *)
type env = {
  let_tbl : (string, let_binding) Hashtbl.t;
      (** The model's [let]s by name. *)
  param_decls : param_decl list;
      (** The model's parameter declarations. *)
}

let env_of_ctx ~let_tbl ~param_decls =
  { let_tbl; param_decls }

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

(** The {!tclass} of a surface expression. Emits nothing; the caller decides
    what a class means at its position.

    The answer depends only on the expression's structure and the
    declarations it names, so one expression gets one class wherever it
    appears; the bounds of a parameter declaration never affect the class of
    its uses. An indexed reference has the class of its unindexed name. A
    cyclic [let] classifies as [TOther] at the point it recurs: see {!env}. *)
let rec classify_in env (seen : string list) (e : expr) : tclass =
  (* [seen]: the lets being followed, innermost first. *)
  let classify env e = classify_in env seen e in
  let follow name (lb : let_binding) =
    if List.mem name seen then TOther else classify_in env (name :: seen) lb.lbody
  in
  match e with
  | EConst _ -> TOther
  | EUnit (_, Days)    | EUnit (_, Weeks)    -> TExact
  | EUnit (_, Months)  | EUnit (_, Years)    -> TCalendar
  | EUnit (_, _)                              -> TOther
  | EIdent ("origin", _) ->
    (* `origin` is an Instant. In an unanchored model the expander reports
       the reference (E327) when it resolves it. *)
    TInstant
  | EIdent (name, _) ->
    (* A let takes its body's class (TOther where it recurs; see [env]),
       then a parameter its kind's, then TOther. *)
    (match Hashtbl.find_opt env.let_tbl name with
     | Some lb -> follow name lb
     | None ->
       match param_kind_of_name env name with
       | Some `Instant  -> TInstant
       | Some `Duration -> TExact
       | None           -> TOther)
  | EIndex (name, _, _) ->
    (* Indexed param/let — kind doesn't depend on the index. *)
    (match Hashtbl.find_opt env.let_tbl name with
     | Some lb -> follow name lb
     | None ->
       match param_kind_of_name env name with
       | Some `Instant  -> TInstant
       | Some `Duration -> TExact
       | None           -> TOther)
  | EFuncCall ("date", _) -> TInstant
  | EFuncCall (("add_calendar_months" | "add_calendar_years"), _) ->
    (* An Instant whether or not the expander can evaluate the call: a
       malformed call is still meant as a date. *)
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

let classify env (e : expr) : tclass = classify_in env [] e

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
   Instant±Duration expression. [children] chooses the sub-expressions it
   descends into, all of them by default; the expander passes one that
   skips the time positions inside an ordinary expression. *)
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

(* ── Instant arithmetic: the forms with no meaning (E360) ──────────── *)

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

(** A form of arithmetic on an instant that has no meaning, each reported as
    E360. Judged only on a definite instant: one built from [date(...)],
    [origin], the calendar primitives, [instant]-kind parameters and [let]s
    of those, moved only by classified durations. *)
type instant_misuse =
  | InstantPlusInstant       (** Two instants added: [date(a) + date(b)]. *)
  | SumOfInstants            (** A sum whose body is an instant. *)
  | ScaledInstant            (** An instant multiplied, divided, raised to a power or negated. *)
  | DurationMinusInstant     (** An instant subtracted from a duration. *)
  | InstantPlusBareNumber    (** An instant moved by a number with no unit: [date(a) + 5]. *)
  | InstantOrDuration        (** A conditional with an instant on one branch, a duration on the other. *)

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

(* [seen]: the lets being followed; a let that recurs is not definite (see
   [env]). *)
let rec is_definite_instant_in env (seen : string list) (e : expr) : bool =
  let is_definite_instant env e = is_definite_instant_in env seen e in
  match e with
  | EIdent (name, _) | EIndex (name, _, _) ->
    (match Hashtbl.find_opt env.let_tbl name with
     | Some _ when List.mem name seen -> false
     | Some lb -> is_definite_instant_in env (name :: seen) lb.lbody
     | None -> classify env e = TInstant)
  | EBinOp (Add, l, r) ->
    (is_definite_instant env l && is_classified_duration (classify env r))
    || (is_definite_instant env r && is_classified_duration (classify env l))
  | EBinOp (Sub, l, r) ->
    is_definite_instant env l && is_classified_duration (classify env r)
  | ECond (_, a, b) -> is_definite_instant env a && is_definite_instant env b
  | EFuncCall _ -> classify env e = TInstant
  | _ -> false

let is_definite_instant env (e : expr) : bool = is_definite_instant_in env [] e

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
     passes every rule above. The two branches must be the same
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
   twelve times short, in a model that compiles and runs.

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
       For a calendar-exact date use %s(d, N); the one-step calendar rule \
       reads `d + N 'months` as that date only in a date position (an `at` \
       entry, `simulate { to }`, the value compared with `t`), not inside a \
       `let` or another expression." exact_fn
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
   Annotate each with a unit literal (`to = 600 'months` is the date 600 \
   calendar months after the origin, by the one-step calendar rule) or a \
   date literal (`to = date(\"1940-12-01\")`) to preserve intent. \
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
    "the one-step calendar rule resolves one calendar step to a date, not a \
     repeating schedule: a calendar cadence counts whole months from its \
     start, which one fixed step cannot represent. List the calendar dates \
     instead, with \
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

(* ── Helper: pretty-printer for the offending sub-expression ─────────── *)

(* A short rendering of a surface expression for messages: index lists and
   call arguments are elided. [Pp_expr] prints IR expressions, not the
   syntax tree, so it cannot serve here. *)
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

(* An expression as the modeller wrote it, for a message that quotes the whole
   of a date position (`date("2020-02-26") + 1 'years + 3 'days`): arithmetic
   with the parentheses its precedence needs and no more, `date(...)` with its
   string, and calendar calls with their arguments. Anything else is rendered
   by [show_short]. *)
let show_source (e : expr) : string =
  let prec = function
    | EBinOp ((Add | Sub), _, _) -> 1
    | EBinOp ((Mul | Div), _, _) -> 2
    | EUnOp (Neg, _) -> 3
    | _ -> 4
  in
  let rec go e =
    match e with
    | EBinOp (((Add | Sub | Mul | Div) as op), l, r) ->
      let p = prec e in
      let os = match op with Add -> "+" | Sub -> "-" | Mul -> "*" | _ -> "/" in
      let side ~strict x = if prec x < p || (strict && prec x = p) then "(" ^ go x ^ ")" else go x in
      Printf.sprintf "%s %s %s" (side ~strict:false l) os (side ~strict:true r)
    | EUnOp (Neg, a) -> if prec a < 3 then "-(" ^ go a ^ ")" else "-" ^ go a
    | EFuncCall ("date", [("", EIdent (s, _))]) -> Printf.sprintf "date(%S)" s
    | EFuncCall (("add_calendar_months" | "add_calendar_years") as f, args) ->
      Printf.sprintf "%s(%s)" f
        (String.concat ", " (List.map (fun (k, a) -> if k = "" then go a else k ^ " = " ^ go a) args))
    | _ -> show_short e
  in
  go e
