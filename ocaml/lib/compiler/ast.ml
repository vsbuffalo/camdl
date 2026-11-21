(** The syntax tree of a camdl model: what the parser builds from a [.camdl]
    file, before the expander resolves names, expands families over their
    dimensions and lowers the model to the intermediate representation.

    The tree stays close to the surface syntax. Names are unresolved strings,
    and an indexed family such as [infection[a in age]] is one node, not one
    per level. The language specification, [docs/camdl-language-spec.md],
    defines what each construct means.

    Most declarations carry two source spans. The one named [*loc] is where a
    diagnostic about the declaration points. The one named [*span] is the
    extent of source text the compiled model quotes for the declaration in its
    declared-model section; a declaration the compiler synthesises has
    {!dummy_loc} there and quotes nothing.

    A declaration field whose value is a time, such as [simulate { to }] or
    an intervention's [every], is not a bare {!expr} but a {!Time_pos.t},
    which records the kind of time its position holds (an instant, a
    cadence, a length, ...), because the kind decides what a calendar
    duration written there means. Times inside a forcing's arguments or
    inside an expression are left as expressions, and {!Time_census} finds
    them.

    The module also holds the checked form of a staged-residence law
    ({!via_spec} and the types it uses), which the expander builds from the
    raw call the parser stores. *)

(** A span of source text.

    Lines and columns are 1-based. Columns count bytes, not characters, so a
    multi-byte character such as [×] advances the column by more than one.
    [end_col] is one past the last byte of the span. The parser leaves [file]
    empty; the expander substitutes the model's file name when it reports, and
    {!stamp_loc_file} fills it for declarations read from a second file. *)
type loc = {
  file     : string;
  line     : int;   (** First line. *)
  col      : int;   (** First column. *)
  end_line : int;
  end_col  : int;
}

(** A value paired with the source span it was parsed from. No part of the
    compiler builds one. *)
type 'a located = { value : 'a; loc : loc }

(** The location of a node the compiler built itself rather than parsed: every
    field zero or empty. *)
let dummy_loc = { file = ""; line = 0; col = 0; end_line = 0; end_col = 0 }

(** A unit written after a number with a leading quote, as in [14 'days] or
    [0.1 'per_day].

    Durations ([Days] to [Years]) and rates ([PerDay] to [PerYear]) convert
    to the model's time unit through days: a week is 7 days, a month
    365.2425/12 days and a year 365.2425 days. [Count] and [Ratio] carry no
    time scale. The same type names the model's time unit, the unit of a
    table's cells and the unit of a forcing's values.

    {@camdl[
      time_unit = 'days
      dur : age 'years = [5, 60]
    ]} *)
type unit_lit =
  | Days | Weeks | Months | Years
  | PerDay | PerWeek | PerMonth | PerYear
  | Count
      (** [count]: a number of individuals, dimension P, as in an
          interpolated population forcing [pop : interpolated 'count]. *)
  | Ratio
      (** [ratio]: a dimensionless multiplier with no bound, such as a
          seasonal factor around 1. The [probability] parameter kind is the
          bounded case. *)

(** A binary operator in an expression. Both [×] and [*] parse to [Mul]. The
    comparisons, [Eq] to [Ge], appear in the condition of an [if] and in the
    comparisons of a reactive intervention's trigger. *)
type bin_op =
  | Add | Sub | Mul | Div | Pow
  | Eq | Neq | Lt | Gt | Le | Ge

(** A unary operator in an expression.

    The parser builds only [Neg], from a leading minus, and [Total]. A call
    such as [exp(x)] parses as a function call ([EFuncCall]), and no part of
    the compiler builds [Exp] to [Tanh]; the expander's constant folding
    evaluates them if they occur. *)
type un_op = Neg | Exp | Log | Sqrt | Abs | Floor | Ceil
           | Sin | Cos | Tanh
           | Total
               (** [sum(I)], a [sum] with no binder: the value of its operand,
                   marked as a total the modeller stated. A bare compartment
                   family already means its sum over strata, so [Total] changes
                   no arithmetic. An observation uses the mark to tell a stated
                   pooling from a silent one: [projected = sum(I)] on a
                   stratified [I] is accepted where [projected = I] is error
                   E280. *)

(** A comparison in a [where] predicate between a table cell and a number,
    [dist[p, q] < 20]. It is separate from {!bin_op} so that a predicate cannot
    contain arbitrary arithmetic. *)
type relop = RLt | RLe | RGt | RGe | REq | RNe

(** The right-hand side of a table-cell comparison in a [where] predicate. *)
type guard_operand =
  | GoNum  of float
      (** A number literal: the only operand a predicate can compare
          against. *)
  | GoName of string
      (** A bare name, as in [dist[p, q] < radius]. A predicate must be
          decided at compile time, so the expander always rejects this
          operand (error E284); the parser keeps the name so the error can
          say which name was used. *)

(** A [where] predicate: a condition on index values, decided by the compiler
    for each combination of a declaration's indices, keeping the combinations
    where it holds.

    It filters the members of a transition, intervention or reactive
    intervention family and the terms of a [sum]. It may compare index
    variables with each other or with a dimension's level names, and a
    constant table's cells with a number; never a parameter or a compartment.

    {@camdl[
      wane2[a in age] : R[a] --> S[a] @ mu * R[a] where a == adult
      let near = sum(p in patch, sum(q in patch where dist[p, q] < 20, 1))
    ]} *)
type guard =
  | GEq  of string * string
      (** [a == b]: the two sides name the same level. Each side is an index
          variable of the declaration or a level name. *)
  | GNeq of string * string
      (** [a != b]. *)
  | GTab of string * string list * relop * guard_operand
      (** [dist[p, q] < 20]: the table, the index variables or level names
          that select its cell, the comparison and the operand. *)
  | GAnd of guard * guard
  | GOr  of guard * guard

(** One index inside the brackets of a reference such as [S[a]]: by
    position, or by the dimension's name.

    {@camdl[
      infection[a in age] : S[a] --> E[a] @ lam[a] * S[a]
      let kids_inf = I[age = child]
    ]} *)
type index_item =
  | IPosn  of expr
      (** [S[a]] or [S[child]]: the index in this position. *)
  | INamed of string * expr
      (** [S[age = child]]: the index for the named dimension. *)
  | IStage of expr
      (** The location of [stage(E[a], 2)], which the parser holds as the
          last index of the reference: [E] indexed by [a] and [IStage 2].
          The location is an integer literal or a variable bound by
          [in stages(E)]. The expander replaces every [IStage] before any
          name is resolved ([Expander.lower_stage_forms]). *)

(** A binder that makes a declaration a family with one member per level,
    written in brackets after the declaration's name.

    {@camdl[
      infection[a in age] : S[a] --> E[a] @ lam[a] * S[a]
      ageing[(a, a_next) in consecutive(age)] : S[a] --> S[a_next] @ aging * S[a]
      death[a in age, c in compartments] : c[a] --> @ mu * c[a]
    ]} *)
and index_binding =
  | IBind   of string * string
      (** [a in age]: the variable and the dimension it ranges over. *)
  | IConsec of string * string * string
      (** [(a, a_next) in consecutive(age)]: each pair of adjacent levels in
          declared order, so a dimension of n levels gives n − 1 members.
          The two variables, then the dimension. *)
  | IComp   of string
      (** [c in compartments]: the variable ranges over the model's integer
          compartments; real compartments are excluded. *)
  | IStages of string * string
      (** [st in stages(E)]: the variable ranges over the stages of the
          compartment, one when it has no Erlang dwell of several stages.
          The expander replaces it before any name is resolved
          ([Expander.lower_stage_forms]). *)

(** An expression: a rate, a value, an argument or a predicate, as written.
    Names are not yet resolved, so a parameter, a compartment and a [let]
    binding look alike.

    {@camdl[
      let lam[a in age] = beta * sum(b in age, C[a, b] * I[b]) / N
    ]} *)
and expr =
  | EConst  of float
      (** A number literal. The keyword [null] also parses to
          [EConst 0.0]. *)
  | EUnit   of float * unit_lit
      (** A number with a unit, [14 'days]. A unit after a parenthesised
          expression, [(20 / 100000) 'per_year], parses as that expression
          times [EUnit (1.0, u)]. *)
  | EIdent  of string * loc
      (** A name and its span. The keyword [origin] in an expression parses
          to the name ["origin"]. A quoted string literal also parses to
          [EIdent], with the string's contents as the name and {!dummy_loc}
          as its span. *)
  | EIndex  of string * index_item list * loc
      (** An indexed reference, [S[a]] or [C[a, b]], with the span of the
          whole reference. *)
  | EBinOp  of bin_op * expr * expr
  | EUnOp   of un_op * expr
  | ESum    of string * string * guard option * expr * loc
      (** [sum(a in age where P, body)]: [body] summed over the levels of the
          dimension that pass the optional predicate. The fields are the
          variable, the dimension, the predicate, the body, and the span of
          the binder [a in age where P]. A flat
          [sum(a in age, p in patch, body)] parses to nested [ESum] nodes, one
          per binder, each with its own binder's span. *)
  | ESumStages of string * string * expr * loc
      (** [sum(st in stages(E), body)]: [body] summed over the stages of the
          compartment. The fields are the variable, the compartment, the
          body, and the span of the binder. The expander replaces it before
          any name is resolved ([Expander.lower_stage_forms]). *)
  | ECond   of expr * expr * expr
      (** [if p then a else b]. *)
  | EFuncCall of string * (string * expr) list
      (** A call by name: a built-in function such as [exp], a forcing, a
          distribution, a reduction or a projection such as [incidence]. Each
          argument is paired with its keyword; a positional argument has the
          keyword [""]. *)
  | EList   of expr list
      (** A bracketed list, [[1.0, 2.0]]; a list of lists is a matrix. *)
  | ERange  of expr * expr
      (** [7:100], written only as an element of a list. Only a periodic
          forcing's [on = [...]] accepts it; anywhere else it is an error. *)
  | EObsAccess of string * loc
      (** [observations.cases]: the simulated values of an observation stream,
          as the source of a [quantities] entry, with its span. Anywhere else
          it is error E290. *)
  | ERunMember of { run : string; ns : run_namespace; member : string; loc : loc }
      (** [with_sia.quantities.total_deaths]: a quantity or observation stream
          of one scenario's run, as an operand in a [contrasts] block. [run] is
          a scenario name or [fitted], the model with no scenario applied.
          Anywhere else it is an error. *)

(** Which kind of member a run-rooted reference names: an entry of the
    [quantities] block, or an observation stream. *)
and run_namespace = NsQuantities | NsObservations

(** Whether a compartment holds whole individuals ([Integer], the default) or
    a continuous quantity ([Real], written [W : real]), such as an
    environmental reservoir. A real compartment must have an equation in the
    [ode] block. *)
type compartment_kind = Integer | Real

(** A time written in a declaration, with the kind of time its position
    holds. *)
module Time_pos : sig
  type instant        = Instant_k
  type cadence        = Cadence_k
  type runtime_offset = Runtime_offset_k
  type span           = Span_k
  type phase          = Phase_k
  type length         = Length_k

  type _ kind =
    | Instant        : instant kind
    | Cadence        : cadence kind
    | Runtime_offset : runtime_offset kind
    | Span           : span kind
    | Phase          : phase kind
    | Length         : length kind

  type 'k t = private { kind : 'k kind; expr : expr; loc : loc }

  (* The only way to build a position. *)
  val make : 'k kind -> expr -> loc -> 'k t
  val expr : _ t -> expr
  val loc  : _ t -> loc
  (* The kind's name, as in "instant" or "runtime offset". *)
  val kind_name : _ kind -> string

  type any = Any : 'k t -> any
end = struct
  (** A time written in a declaration, with the kind of time its position
      holds.

      Every expression that holds a time is one of six kinds, and the kind
      belongs to the position the expression is written in, not to the
      expression: [6 'months] is an instant in [simulate { to }], a cadence
      in [every] and a length in [dt]. The kind decides what a calendar
      duration means there. In a model with an [origin], an instant moved by
      one whole calendar step from a date the compiler knows is that
      calendar date, so [to = 6 'months] is the date six calendar months
      after the origin; a cadence, a span or a run-time offset refuses a
      calendar duration; and a length reads one as its average, a month
      being 365.2425/12 days. Without an origin there is no calendar, and
      every month is the average one.

      A declaration field that holds a time has the type ['k Time_pos.t], so
      the type of each lowering function says which kind it accepts. Code
      that handles positions of every kind reads the kind as a value,
      [kind]: the expander does so for the positions {!Time_census} lists
      inside ordinary expressions, and to key its record of the positions it
      has judged. The record is private, so a value is built only with
      [make]: by the parser for every declaration field, by {!Time_census}
      for the positions inside a forcing's arguments, a reduction or an
      ordinary expression, and by the expander for a reactive trigger's
      [window]. The expander's [Lower] module turns a position into a number
      or an IR expression, with one function per kind, each applying its
      kind's rule.

      The module is defined in {!Ast}, not in a file of its own, because the
      declaration records hold a position and a position holds an
      {!Ast.expr}.

      {@camdl[
        simulate { from = 0 'days  to = 6 'months  dt = 1 'days }
      ]} *)

  (** The type index of an instant position. Each of the six index types
      has one constructor, which nothing uses: the type exists only as the
      parameter of {!kind} and {!t}. *)
  type instant        = Instant_k

  (** The type index of a cadence position. *)
  type cadence        = Cadence_k

  (** The type index of a run-time offset position. *)
  type runtime_offset = Runtime_offset_k

  (** The type index of a span position. *)
  type span           = Span_k

  (** The type index of a phase position. *)
  type phase          = Phase_k

  (** The type index of a length position. *)
  type length         = Length_k

  (** The six kinds of time position, each indexed by its own type, so that
      a value of [instant kind] can only be [Instant]. *)
  type _ kind =
    | Instant        : instant kind
        (** A point on the time axis: [simulate { from }] and [{ to }], an
            [at] list's entries, a recurring schedule's [from] and [to], a
            scenario's [to], a timepoint, a [breakpoints] or [times] entry,
            the ends of a forcing's [window], [value_at]'s time, the bounds
            of an [instant] parameter and the value compared with [t]. A
            bare duration [D] is the instant [origin + D]. Under an origin,
            one calendar step from a known date resolves to that date, and
            any other calendar duration is refused (E321). *)
    | Cadence        : cadence kind
        (** The step of a recurring schedule: [every] of an intervention,
            an output or an emit schedule. A calendar duration is refused
            under an origin (E322), since one fixed step cannot follow the
            calendar. *)
    | Runtime_offset : runtime_offset kind
        (** An offset from an instant known only during the run: a reactive
            intervention's [after] and [cooldown], and an offset added to or
            subtracted from [t]. A calendar duration is refused under an
            origin (E321). *)
    | Span           : span kind
        (** The width of a period that tiles the axis or trails a time: the
            duration of a [covers] line and the [window] of a
            [sum_observed] trigger. A calendar duration is refused under an
            origin (E321). *)
    | Phase          : phase kind
        (** An offset within a repeating period: [at_day], a sinusoidal
            forcing's [phase] and a periodic forcing's [on] endpoints. A
            calendar duration is refused in [at_day] (E322) and read as its
            average in [phase]. *)
    | Length         : length kind
        (** A fixed length, in which a calendar duration is its average:
            [simulate { dt }], a forcing's [period], [step] and [lag], and
            the bounds of a [duration] parameter. *)

  (** One time position: its kind, the expression as written, and the span
      of the expression, where a diagnostic about it points. A position the
      parser fills in for an absent field, such as the [from = 0] of a
      [simulate] block that has none, has the span of the enclosing block
      instead. *)
  type 'k t = { kind : 'k kind; expr : expr; loc : loc }

  let make kind expr loc = { kind; expr; loc }
  let expr p = p.expr
  let loc p = p.loc
  let kind_name : type k. k kind -> string = function
    | Instant        -> "instant"
    | Cadence        -> "cadence"
    | Runtime_offset -> "runtime offset"
    | Span           -> "span"
    | Phase          -> "phase"
    | Length         -> "length"

  (** A time position of any kind, the kind known only by inspecting it:
      the element of the census's list of every position in a model. *)
  type any = Any : 'k t -> any
end

(** The text of a [#'] doc-comment block in a model file, with its recognised
    tags.

    A block documents the declaration immediately after it or, at the head of
    the file, the model as a whole. It does not change what the model
    computes, but it is kept: [camdlc inspect] shows it, and the expander
    copies it into the compiled model's documentation index for reports.

    {@camdl[
      #' transmission rate
      #' @symbol \beta
      beta    : rate ~ log_normal(mu = -1, sigma = 0.5)
    ]} *)
type doc = {
  d_text   : string option;
      (** The prose lines, joined with spaces; [None] when the block has only
          tags. In the model's header block, each line that starts with an
          unrecognised [@] tag is kept here on a line of its own. *)
  d_symbol : string option;
      (** [@symbol]: the label to show for the declaration in plots and
          reports. *)
  d_ref    : string option;
      (** [@ref]: a citation for the definition. On a declaration, any tag
          other than [@symbol] and [@ref] is error E111. *)
}

(** One compartment of the [compartments] block: its name, kind and doc
    block. [cloc] and [cspan] both cover the name and the kind.

    {@camdl[
      compartments { S, E, I, R, Q, D, W : real }
    ]} *)
type compartment_decl = { cname: string; ckind: compartment_kind; cdoc: doc option; cloc: loc; cspan: loc }

(** The declared kind of a parameter, a typed [let], a table's cells or an
    observed data column, which fixes its range and, for most kinds, its
    dimension (language specification, section 4.1).

    - [PRate], [rate]: non-negative, dimension 1/time.
    - [PProbability], [probability]: between 0 and 1, dimensionless.
    - [PPositive], [positive]: greater than zero; the dimension comes from a
      [dim] clause or from how the value is used.
    - [PCount], [count]: dimension P, a number of individuals. Whole numbers
      are not enforced.
    - [PReal], [real]: unconstrained; the dimension comes from a [dim]
      clause or from use.
    - [PInstant], [instant]: a point in time, dimension T, shown as a date
      when the model declares an [origin].
    - [PDuration], [duration]: a span of time, dimension T. *)
type param_type = PRate | PProbability | PPositive | PCount | PReal | PInstant | PDuration

(** A dimension stated by a [dim] clause, as the exponents of the two base
    dimensions: population P first, then time T. [dim P/T] is [(1, -1)].

    {@camdl[
      let b0 : positive dim P^-1*T^-1 = 0.0001
    ]} *)
type dim_annotation = int * int

(** A parameter's prior, written after [~] in the [parameters] block: a
    distribution name, its keyword arguments, and an optional pooling clause.

    {@camdl[
      beta    : rate ~ log_normal(mu = -1, sigma = 0.5)
      rho_a[age] : rate ~ log_normal(mu = mu_h, sigma = 0.3) | age
    ]} *)
type prior_spec = {
  ps_name:      string;
      (** The distribution's name as written, such as ["log_normal"]; the
          expander checks it against the known distributions. *)
  ps_args:      (string * expr) list;
      (** The keyword arguments, as written. *)
  ps_pool_over: string option;
      (** [| age]: a hierarchical prior, partially pooling the parameter's
          values over the named dimension. [None] for an ordinary prior. *)
}

(** The two ends of a parameter's [in [lo, hi]], as time positions of the
    parameter's own kind: instants for an [instant] parameter, lengths for a
    [duration] parameter or one declared [dim T], and plain values for every
    other. The parser builds the constructor, with {!make_bounds}, in the
    production that knows the kind.

    {@camdl[
      t_intro : instant in [date("2020-01-01"), date("2020-03-01")]
      delay   : duration in [1 'days, 3 'weeks]
      gamma   : rate in [0.05, 0.5]
    ]} *)
type bounds =
  | Instant_bounds of Time_pos.instant Time_pos.t * Time_pos.instant Time_pos.t
      (** The earliest and latest instant, judged and lowered as instant
          positions, so a calendar step from the origin resolves to its
          date. *)
  | Length_bounds  of Time_pos.length Time_pos.t * Time_pos.length Time_pos.t
      (** The shortest and longest length; a calendar duration is its
          average. *)
  | Value_bounds   of expr * expr
      (** The lower and upper value, in the parameter's units. *)

(** The two bound expressions as written, whatever their kind. *)
let bounds_exprs = function
  | Instant_bounds (lo, hi) -> (Time_pos.expr lo, Time_pos.expr hi)
  | Length_bounds (lo, hi)  -> (Time_pos.expr lo, Time_pos.expr hi)
  | Value_bounds (lo, hi)   -> (lo, hi)

(** The bounds of a parameter of the given kind and [dim] clause, with each
    end's span: [Instant_bounds] for an [instant] parameter,
    [Length_bounds] for a [duration] or a [dim T] one, [Value_bounds]
    otherwise. *)
let make_bounds (kind : param_type) (dim : dim_annotation option) ~lo ~hi ~lo_loc ~hi_loc =
  match kind, dim with
  | PInstant, _ ->
    Instant_bounds (Time_pos.make Time_pos.Instant lo lo_loc,
                    Time_pos.make Time_pos.Instant hi hi_loc)
  | PDuration, _ | _, Some (0, 1) ->
    Length_bounds (Time_pos.make Time_pos.Length lo lo_loc,
                   Time_pos.make Time_pos.Length hi hi_loc)
  | _ -> Value_bounds (lo, hi)

(** One entry of the [parameters] block: a scalar parameter, or a family
    indexed by dimensions.

    In both constructors [pkind] is the kind, [pdim] the dimension a [dim]
    clause states (only [positive] and [real] accept one; every other kind
    fixes its own), [pbounds] the interval of [in [lo, hi]], and [pprior] the
    prior. [ploc] and [pspan] both cover the declaration from its name.

    {@camdl[
      gamma   : rate in [0.05, 0.5]
      rho_a[age] : rate ~ log_normal(mu = mu_h, sigma = 0.3) | age
    ]} *)
type param_decl =
  | PScalar  of { pname: string; pkind: param_type; pdim: dim_annotation option; pbounds: bounds option; pprior: prior_spec option; pdoc: doc option; ploc: loc; pspan: loc }
      (** [gamma : rate]: one value. *)
  | PIndexed of { pname: string; pdims: string list; pkind: param_type; pdim: dim_annotation option; pbounds: bounds option; pprior: prior_spec option; pdoc: doc option; ploc: loc; pspan: loc }
      (** [rho_a[age] : rate]: one value per level of the dimensions listed
          in [pdims]. *)

(** One axis of a table's declared shape: a dimension, which may carry the
    unit of the table's cells. The unit belongs to the cells, not the axis;
    the expander rejects a table that writes a unit on more than one axis
    (error E216). *)
type table_dim_entry =
  | TDim     of string
      (** [age]. *)
  | TDimUnit of string * unit_lit
      (** [age 'years]: the [age] axis, with the cells in years. *)

(** One entry of the [tables] block: constant data indexed by dimensions,
    written inline or read from a file.

    {@camdl[
      C : age × age = [[12.0, 4.0], [4.0, 8.0]]
      ifr : age : probability = [0.001, 0.02]
      dur : age 'years = [5, 60]
    ]} *)
type table_decl = {
  tnames     : string list;
      (** The table's names. Several names, [pop, init_sus : patch = read(...)],
          declare one table per value column of the file. *)
  tdims      : table_dim_entry list;
      (** The axes, in order; empty for the form [name = value], which has
          none. *)
  tcell_kind : param_type option;
      (** The kind of every cell, from [: probability] after the axes. *)
  tvalue     : expr;
      (** The data: a list literal, or a call such as [read("pop.tsv")]. *)
  tloc       : loc;
      (** The span of the axes ([age × patch]), or of the name when there are
          none. A diagnostic about an undeclared axis points here rather than
          at the whole declaration, whose inline value can run to many
          lines. *)
  tspan      : loc;
}

(** A compartment named on either side of a transition's arrow, with its
    indices: [S[a]], [R], [S[child]]. *)
type stoich_ref = string * index_item list

(** Where a transition sends the individuals it moves.

    {@camdl[
      infection : S --> I @ beta * S * I / N                # DstSum [I]
      split     : I --> R + Q @ gamma * I                   # DstSum [R; Q]
      recovery  : I --> { R : 0.9, D : 0.1 } @ gamma * I    # DstBranch
      death     : Q --> @ mu * Q                            # DstSum []
    ]} *)
type destination_form =
  | DstSum    of stoich_ref list
      (** Every listed compartment gains one individual per firing. The empty
          list stands for two things the tree does not tell apart: a
          transition whose individuals leave the model ([Q --> @ mu * Q], a
          death or emigration), and a [via] transition with no arrow target,
          [I via hyper_erlang(...)], each of whose branches names its own
          destination. *)
  | DstBranch of (stoich_ref * expr) list
      (** One flow per branch, at the branch's weight times the transition's
          rate, so the source leaves at the rate times the sum of the
          weights. Weights that are all constants must sum to one (E423). *)

(** A [via] clause's dwell law as written: the law's name and its keyword
    arguments, unchecked. [via erlang(stages = 3, mean = 5 'days)] is
    [("erlang", [("stages", ...); ("mean", ...)])]. The expander checks it and
    builds the typed {!via_spec}.

    {@camdl[
      onset     : E --> I via erlang(stages = 3, mean = 5 'days)
    ]} *)
type via_call = string * (string * expr) list

(** A whole number of at least one, known at compile time: the number of
    stages of a staged residence. *)
module Pos_int : sig
  type t
  val of_float : float -> (t, string) result   (* error message on failure *)
  val to_int   : t -> int
end = struct
  (** A whole number of at least one, known at compile time: the number of
      stages of a staged residence. The only way to make one is [of_float],
      which refuses anything else, so a value of [t] is proof that the count
      is usable and no later code checks it again. *)

  (** A whole number of at least one. Built only by [of_float]. *)
  type t = int
  let of_float f =
    if Float.is_integer f && f >= 1.0 then Ok (int_of_float f)
    else if not (Float.is_integer f) then
      Error (Printf.sprintf "must be a whole number, got %g" f)
    else
      Error (Printf.sprintf "must be at least 1, got %g" f)
  let to_int t = t
end

(** The time scale of a staged residence: its mean dwell time, or the
    reciprocal of that, its exit rate. Either is an expression and may use
    parameters. With [k] stages, each stage's rate is [k / τ] for [Mean τ]
    and [k · ρ] for [Rate ρ]. The expander builds one only when exactly one of
    [mean] and [rate] is given. *)
type mean_spec = Mean of expr | Rate of expr

(** One branch of a [hyper_erlang] mixture: an Erlang chain with its own
    number of stages, its own mean and its own destination, entered with the
    branch's probability.

    {@camdl[
      branch(label = fatal, weight = cfr, stages = 3, mean = 8 'days, to = D),
      branch(label = recover, stages = 2, rate = 1 / tau, to = R)
    ]} *)
type hyper_branch = {
  hb_weight : expr option;
      (** The probability of entering this branch. [None] only on the last
          branch, whose weight is one minus the sum of the others, so the
          mixture is normalised by construction. The type does not hold this:
          the expander refuses a [None] anywhere else and a weight on the last
          branch, and, when the weights are constants, a weight outside 0 to 1
          or weights summing past one. *)
  hb_stages : Pos_int.t;
  hb_mean   : mean_spec;
  hb_to     : stoich_ref option;
      (** The destination, from [to = D]; [None] means the transition's arrow
          target, and a branch with neither is an error. *)
  hb_label  : string;
      (** The branch's name, from [label = fatal]. It names the branch's stage
          compartments, so it must be unique within the mixture. *)
}

(** A checked dwell law: the expander's typed form of a {!via_call}. The
    parser never builds one. *)
type via_spec =
  | Erlang      of { stages : Pos_int.t; mean : mean_spec }
      (** [erlang(stages = k, mean = τ)]: the individual passes through [k]
          exponential stages in turn, so the total dwell time is
          Erlang-distributed with mean [τ]. *)
  | HyperErlang of { branches : hyper_branch list }
      (** [hyper_erlang(branch(...), ...)]: a finite mixture of Erlang chains,
          one chosen on entry with the branches' weights. There are at least
          two branches. Since branches differ in length, each becomes its own
          chain of compartments rather than a level of one stage
          dimension. *)

(** How fast a transition fires: an ordinary rate, or a staged residence whose
    law sets the rate of each stage. A transition has exactly one; writing
    both, or neither, in the block form is a parse error.

    {@camdl[
      wane[a in age] : R[a] --> S[a] {
        rate = mu * R[a]
        where a == adult
      }
    ]} *)
type trans_dynamics =
  | Rate of expr
      (** [@ rate], or [rate = ...] in the block form: the transition's total
          propensity, so the waiting time to a firing is exponential. *)
  | Via  of via_call
      (** [via law(...)], or [via = law(...)] in the block form: a staged
          residence, as written. *)

(** One entry of the [transitions] block: a flow of individuals from source
    compartments to destinations, at a rate or through a staged residence.

    {@camdl[
      birth : --> S[child] @ Pi
      #[lineage]
      infection[a in age] : S[a] --> I[a] @ beta * S[a] * sum(b in age, I[b]) / N
    ]} *)
type transition_decl = {
  trname    : string;
  trindices : index_binding list;
      (** The binders that make the transition a family, one member per
          combination of levels; empty for a single transition. *)
  trsrc     : stoich_ref list;
      (** The compartments that each lose one individual per firing; empty
          for an inflow such as a birth. *)
  trdst     : destination_form;
  trdyn     : trans_dynamics;
  trguard   : guard option;
      (** A [where] predicate that drops members of the family. *)
  trlineage : bool;
      (** [true] when the transition carries the [#[lineage]] attribute: each
          firing records which individual caused the new one, so a simulation
          can produce a transmission tree. The rate must be linear in the
          infector compartments. *)
  trdoc     : doc option;
  trloc     : loc;
      (** From the name to the end of the declaration. *)
  trspan    : loc;
      (** As [trloc], but starting at the [#[lineage]] attribute when there is
          one. *)
}

(** The expressions a transition's dynamics evaluate: the rate, or every
    argument of the [via] law. *)
let trans_dynamics_exprs (d : trans_dynamics) : expr list =
  match d with
  | Rate e        -> [e]
  | Via (_, args) -> List.map snd args

(** A top-level [let]: a named expression, which may be indexed, typed or
    shaped. The parser fills at most one of [lshape] and [lkind].

    {@camdl[
      let N = sum(a in age, S[a] + E[a] + I[a] + R[a]) + Q + D
      let lam[a in age] = beta * sum(b in age, C[a, b] * I[b]) / N
      let b0 : positive dim P^-1*T^-1 = 0.0001
      let w : age × patch = [[1.0, 2.0], [3.0, 4.0]]
    ]} *)
type let_binding = {
  lname    : string;
  lindices : index_binding list;
      (** Binders for an indexed [let], [lam[a in age]]. *)
  lshape   : string list option;
      (** The dimensions of a shaped literal, [let w : age × patch = ...]:
          the body is a nested list laid out over them. *)
  lkind    : param_type option;
      (** The kind of a typed [let], [let b0 : positive = ...]. *)
  ldim     : dim_annotation option;
      (** A [dim] clause on a typed [let]. *)
  lloc     : loc;
      (** From the name to the end of the body. *)
  lspan    : loc;
      (** From the [let] keyword to the end of the body. *)
  lbody    : expr;
  ldoc     : doc option;
      (** The [#'] block. It is often the only place that says which
          quantity a derived expression is meant to be, such as the total
          population or the infectious denominator. *)
}

(** A [stratify(...)] declaration: each compartment it applies to is split
    into one compartment per level of the dimension. Several declarations
    compose, so a compartment can be stratified by age and by patch.

    {@camdl[
      stratify(by = age, only = [S, E, I, R])
    ]} *)
type stratify_decl = {
  sdim  : string;
      (** The dimension, from [by]; [""] when [by] is missing, which the
          expander reports as a dimension with no levels. *)
  sonly : string list option;
      (** [only = [S, I]]: the compartments to stratify; [None] means all. *)
  sspan : loc;
}

(** A distribution, by family, with its keyword arguments as written: an
    observation's measurement model, after [~], or the law an initial value
    is drawn from.

    The parser rewrites two forms into these families.
    [diagnostic_test(base = binomial(...), sens = s, spec = c)] becomes a
    [LikBinomial] (or [LikBernoulli]) whose [p] is replaced by
    [s · p + (1 − c)(1 − p)]. [zero_inflated(base = neg_binomial(...), pi = z)]
    becomes [LikZeroInflatedNegBinomial]. A call the parser reports as
    malformed becomes [LikReported].

    {@camdl[
      cases         ~ neg_binomial(mean = rho * projected, r = r_k)
      I[child] ~ poisson(rate = 5)
    ]} *)
type likelihood_kind =
  | LikNegBinomial  of (string * expr) list
  | LikPoisson      of (string * expr) list
  | LikNormal       of (string * expr) list
  | LikBinomial     of (string * expr) list
  | LikBetaBinomial of (string * expr) list
  | LikBeta         of (string * expr) list
  | LikBernoulli    of (string * expr) list
  | LikZeroInflatedNegBinomial of (string * expr) list
      (** [zero_inflated(base = neg_binomial(mean = m, r = k), pi = z)],
          flattened: the base's arguments followed by [pi], the probability
          that a count is a structural zero. *)
  | LikReported of string * (string * expr) list
      (** A distribution the compiler has already reported as malformed or
          missing, with the family name and the arguments as written: an
          unknown family, a [diagnostic_test] or [zero_inflated] without a
          base it accepts, a value that is not a call, or an observation
          with no measurement. The report has failed the compile, so no
          check that depends on the family reads the arguments and the
          distribution is never lowered. The arguments are kept so that a
          pass reading the names a model uses still sees the ones written
          here. *)

(** A [diagnostic_test(...)] as written. The parser has already folded
    [sens] and [spec] into the base's [p], where no rule can tell the three
    apart; they are kept so the expander can check each argument whose value
    is a probability (E379), and nothing else reads them. *)
type test_accuracy = {
  ta_base : likelihood_kind;
      (** The base, [binomial(...)] or [bernoulli(...)], with its [p] as
          written. *)
  ta_sens : expr;
  ta_spec : expr;
}

(** What one entry of the [init] block says its compartment starts at:
    computed, or drawn. An entry is one or the other, never both. *)
type init_value =
  | IVExpr of expr
      (** [S = N0 - I]: computed from parameters, constants and other
          compartments' initial values. *)
  | IVLaw  of likelihood_kind * test_accuracy option * loc
      (** [I ~ poisson(rate = I0)]: drawn from the law once at the start of
          each run, with the [diagnostic_test(...)] the law was written as,
          if it was, and the span of the law itself, where a diagnostic about
          the distribution points. *)

(** The keyword arguments of a distribution, whatever its family. *)
let likelihood_kwargs (l : likelihood_kind) : (string * expr) list =
  match l with
  | LikNegBinomial a | LikPoisson a | LikNormal a | LikBinomial a
  | LikBetaBinomial a | LikBeta a | LikBernoulli a
  | LikZeroInflatedNegBinomial a | LikReported (_, a) -> a

(** The family's name, as diagnostics and the compiled model spell it:
    the keyword written in the model file (["poisson"], ["neg_binomial"]),
    except that the zero-inflated family is ["zero_inflated_neg_binomial"]
    rather than the [zero_inflated] the model writes, and a reported
    distribution is named as written ([""] when it is not a call). *)
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
  | LikReported (name, _) -> name

(** Rewrites every keyword-argument expression, keeping the family. *)
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
  | LikReported (name, a) -> LikReported (name, g a)

(** Every expression an initial value evaluates: the right-hand side, or
    every argument of the law. Passes over initial values go through this, so
    a law's arguments are treated exactly as a computed value is. *)
let init_value_exprs (v : init_value) : expr list =
  match v with
  | IVExpr e -> [e]
  | IVLaw (l, _, _) -> List.map snd (likelihood_kwargs l)

(** Rewrites every expression of an initial value, keeping its form. *)
let map_init_value (f : expr -> expr) (v : init_value) : init_value =
  match v with
  | IVExpr e -> IVExpr (f e)
  | IVLaw (l, acc, loc) -> IVLaw (map_likelihood_kwargs f l, acc, loc)

(** One entry of the [init] block: a compartment, one of its strata or a
    family of strata, and the value it starts at. The parser fills at most one
    of [iindices] and [ibindings].

    {@camdl[
      S[a in age] = 500
      I[child] ~ poisson(rate = 5)
    ]} *)
type init_entry = {
  icomp     : string;
  iindices  : index_item list;
      (** [I[child]]: the one stratum the entry sets. *)
  ibindings : index_binding list;
      (** [S[a in age]]: binders, one entry per level. *)
  ivalue    : init_value;
  iloc      : loc;
  ispan     : loc;
}

(** A set of times, given by a regular step or listed: when an observation
    stream emits synthetic rows, or when the [output] block records the
    state. *)
type schedule_core =
  | SchedEvery of Time_pos.cadence Time_pos.t
      (** [every = 7 'days], written [every 7 'days] in [emit_schedule]: a
          cadence. *)
  | SchedAt    of Time_pos.instant Time_pos.t list
      (** [at = [0, 30, 60]], written [at [...]] in [emit_schedule]: one
          instant per entry. *)

(** The model quantity an observation stream measures, from its
    [projected = ...] line. *)
type obs_projection =
  | ProjIncidence  of string * index_item list
      (** A transition's cumulative flow, by name. The parser never builds
          it; the expander uses it only as a placeholder after reporting a
          missing projection. [incidence(infection)] parses as
          [ProjDerived]. *)
  | ProjDerived    of expr
      (** The expression on the right of [=]: a compartment, [I / N], or a
          call such as [incidence(infection)].

          {@camdl[
            projected     = incidence(infection)
            projected = I / N
          ]} *)

(** An observation stream's measurement model, [npos ~ binomial(...)]: the
    data column scored, and the distribution it is scored under.

    {@camdl[
      npos      ~ binomial(n = tested, p = projected)
    ]} *)
type obs_measurement = {
  om_scored : string;
      (** The scored column, which must be a declared value column; [""]
          only after a parse error. *)
  om_lik    : likelihood_kind;
  om_accuracy : test_accuracy option;
      (** [Some] when the measurement was written as [diagnostic_test(...)];
          [None] for any other family. *)
  om_loc : loc;
      (** The span of the distribution call. *)
}

(** What one column of an observation's data file holds, from its entry in
    the [columns] block. A stream declares either one [ColTime] column or a
    [ColWindowStart] and [ColWindowStop] pair, never both. *)
type obs_col_role =
  | ColTime
      (** [time : time]: the time each row is labelled with. *)
  | ColDim   of string
      (** [age : dim]: the levels of a dimension. The column is named after
          the dimension, and the string repeats that name. The stream's header
          binders and its [: dim] columns must match one to one. *)
  | ColValue of param_type
      (** [cases : count]: an observed value of that kind, either the scored
          column or data the distribution reads, such as a binomial's [n]. A
          value column that is neither is an error. *)
  | ColWindowStart
      (** [window_start]: the opening boundary of the period a row covers. *)
  | ColWindowStop
      (** [window_stop]: the closing boundary of the period a row covers. *)

(** One entry of an observation's [columns] block: a column of the data file
    and its role.

    {@camdl[
      columns   { time : time, npos : count, tested : count }
    ]} *)
type obs_column = {
  oc_name : string;
  oc_role : obs_col_role;
}

(** An observation's [covers] line, as written: the period each row's count
    accumulates over, stated relative to the row's time label.

    The forms are [day(t)], the one day starting at the label;
    [starting_on(t, d)], the span [d] starting at the label;
    [ending_on(t, d)], the span [d] whose last included day is the label; and
    [closing_at(t, d)], the span [d] ending at the label, the label itself
    excluded. The expander checks the form, that [t] is the stream's time
    column, and that [d] is a fixed duration.

    {@camdl[
      covers        = closing_at(time, 7 'days)
    ]} *)
type obs_covers = {
  ocv_form : string;
      (** The form's name; [""] after a parse error. *)
  ocv_col  : string;
      (** The time column named in the call. *)
  ocv_span : Time_pos.span Time_pos.t option;
      (** The duration, a span position; [None] for [day(t)]. *)
  ocv_loc  : loc;
}

(** One stream of the [observations] block: the columns of its data file, the
    model quantity the data measure, and the distribution the data are scored
    under. A missing [columns], projection or measurement is kept as [None]
    so the expander can name the missing line. *)
type obs_decl = {
  oname       : string;
  oindices    : index_binding list;
      (** Binders for a stratified stream, [cases[a in age]]: one stream per
          level, each binder matched by a [: dim] column. *)
  osource     : string option;
      (** [from label]: the key under which the data file is supplied,
          [--data label=file]; [None] uses the stream's name. *)
  ocolumns    : obs_column list option;
  omeasurement : obs_measurement option;
  oprojection  : obs_projection option;
      (** The projection line. The parser does not check the name on the left
          of its [=], so [foo = incidence(infection)] is accepted as the
          projection. *)
  oschedule   : schedule_core option;
      (** [emit_schedule]: when [simulate --obs] writes synthetic rows. A fit
          never reads it; there the data file's rows give the times. *)
  ocovers     : obs_covers option;
      (** The [covers] line. [None] is right for a stream that reads state at
          an instant, or one with window columns; a stream that accumulates a
          flow with neither is an error. *)
  odoc        : doc option;
  oloc        : loc;
  ospan       : loc;
}

(** What an intervention, event or reactive intervention does to the state
    when it fires. *)
type action_decl =
  | ATransfer of (string * expr) list
      (** [transfer(fraction = f, from = S, to = V)], or with [count = n]:
          moves individuals between compartments. The keyword arguments are
          kept as written. *)
  | ASet      of string * index_item list * expr
      (** [I = I + 10] in the block form: sets a compartment, by name and
          optional indices, to the value. *)
  | AAdd      of string * index_item list * expr
      (** [add(I, 5)]: adds the rounded value to a compartment, by name and
          optional indices; a negative value removes individuals. *)

(** When an intervention or event fires. *)
type schedule_decl =
  | SAtTimes of Time_pos.instant Time_pos.t list
      (** [at [180, 545]], or [at = [...]] in the block form: at each listed
          instant. A block form that states no schedule also parses to
          [SAtTimes []], which never fires. *)
  | SRecurring of { every : Time_pos.cadence Time_pos.t;
                    from  : Time_pos.instant Time_pos.t option;
                    until : Time_pos.instant Time_pos.t option }
      (** [{ every = 30 'days  from = 0 'days  to = 2 'years }]: the cadence
          [every], and the optional window start [from] and end [until]
          (written [to]), which default to the run's start and end. *)
  | SEveryAtDay of { every : Time_pos.cadence Time_pos.t; at_day : Time_pos.phase Time_pos.t }
      (** [every 365 'days at_day 100 'days]: once per period [every], at the
          phase [at_day] from time zero. Only the [add] action accepts
          it. *)

(** One entry of an [interventions] or [events] block: one or more actions and
    when they fire. Interventions fire only when a scenario or [--enable]
    turns them on; events always fire. The two blocks share this grammar.

    {@camdl[
      sia    : transfer(fraction = cov, from = S, to = V) at [180 'days]
      pulse  : { I = I + 10  at = [90 'days] }
      yearly : add(S, 100) every 365 'days at_day 100 'days
    ]} *)
type intervention_decl = {
  ivname    : string;
  ivindices : index_binding list;
      (** Binders for a family, one member per level; empty for a single
          intervention. *)
  ivaction  : action_decl list;
      (** The actions in source order. Only the block form holds more than
          one, and then all are [ASet]; an empty list is an error. *)
  ivschedule: schedule_decl;
  ivguard   : guard option;
      (** A [where] predicate that drops members of the family. *)
  ivloc     : loc;
  ivspan    : loc;
}

(** The condition of a reactive intervention's [when] clause: comparisons of
    observed surveillance data with thresholds, combined with [and], [or] and
    [not]. *)
type trig_pred =
  | TgAtom of expr
      (** A comparison such as [observed(cases) >= threshold], as an
          expression. The expander checks that one side reads [observed] or
          [sum_observed] and the other is a constant or a parameter. *)
  | TgAnd  of trig_pred * trig_pred
  | TgOr   of trig_pred * trig_pred
  | TgNot  of trig_pred
      (** [not]: the grammar puts it only directly on a comparison, and has no
          parentheses for predicates, so a negated conjunction cannot be
          written. *)

(** One entry of the [reactive_interventions] block: an action that fires when
    a condition on observed surveillance data holds, tested at each
    observation-emission time. Like an intervention it is off unless a
    scenario or [--enable] turns it on.

    {@camdl[
      mop_up : when sum_observed(cases, window = 28 'days) >= threshold {
        after    = 21 'days
        action   = transfer(fraction = cov, from = S, to = V)
      }
    ]} *)
type reactive_decl = {
  rxname     : string;
  rxindices  : index_binding list;
      (** Binders for a family; empty for a single policy. *)
  rxwhen     : trig_pred;
  rxafter    : Time_pos.runtime_offset Time_pos.t option;
      (** [after]: the delay from the trigger to the action, a run-time
          offset; [None] means none. *)
  rxonce     : expr option;
      (** [once = true] or [once = false], as an expression; [None] means
          [true]. Any other expression is an error. *)
  rxcooldown : Time_pos.runtime_offset Time_pos.t option;
      (** [cooldown]: the least time between firings of a repeating policy,
          measured from each trigger, a run-time offset; not allowed with
          [once = true]. *)
  rxaction   : action_decl;
      (** The [action]; [ATransfer []] after a parse error for a missing
          one. *)
  rxguard    : guard option;
      (** A [where] predicate that drops members of the family. *)
  rxloc      : loc;
  rxspan     : loc;
}

(** One equation of the [ode] block: the time derivative ([oderiv]) of a real
    compartment ([ocomp]).

    {@camdl[
      W = xi * total_I - delta * W
    ]} *)
type ode_decl = { ocomp: string; oderiv: expr; odspan: loc }

(** One entry of the [quantities] block: a quantity computed from a run, such
    as a final value, a peak, a series, or arithmetic over earlier entries.

    The expander reads the body as a recipe, not as an ordinary expression:
    a call such as [final(D)] or [max(I)] names a reduction over the run, and
    these names are not keywords.

    {@camdl[
      total_deaths = final(D)
      peak_reported = max(observations.cases)
    ]} *)
type quantity_decl = {
  qd_name    : string;
  qd_indices : index_binding list;
      (** Binders for a quantity reported per level; empty for one value. *)
  qd_body    : expr;
  qd_doc     : doc option;
  qd_loc     : loc;
  qd_span    : loc;
}

(** One entry of the [contrasts] block: arithmetic over the quantities or
    observation streams of different scenarios' runs, such as the deaths an
    intervention averted. Its operands are [ERunMember] references.

    {@camdl[
      averted = fitted.quantities.total_deaths - with_sia.quantities.total_deaths
    ]} *)
type contrast_decl = {
  cd_name   : string;
  cd_body   : expr;
  cd_doc    : doc option;
  cd_loc    : loc;
  cd_span   : loc;
}

(** The value of one argument of a forcing declaration: a quoted string, which
    names something outside the model, or an expression, which names
    something inside it. Each argument accepts one of the two, and the other
    is an error.

    {@camdl[
      data      = "data/flu_forcing.tsv"
      method    = linear
    ]} *)
type farg_value =
  | FStr  of string
      (** A quoted string: a data-file path or a column of that file. *)
  | FExpr of expr
      (** A bare expression: a value, a model name such as a table, or a word
          from a fixed set such as [linear]. *)

(** The expression of a forcing argument, or [None] for a quoted string,
    which carries no unit or date. *)
let farg_expr_opt = function FExpr e -> Some e | FStr _ -> None

(** The kind of a forcing, the word after the colon in
    [seasonal : sinusoidal 'ratio], which decides the arguments it takes
    and the function of time it is. The language specification's section
    on forcings defines each kind.

    {@camdl[
      seasonal : sinusoidal 'ratio {
        amplitude = 0.2
        period    = 365 'days
        phase     = 0 'days
        baseline  = 1.0
      }
    ]} *)
type forcing_kind =
  | FSinusoidal | FPiecewise | FInterpolated | FPeriodic | FFourier
  | FPeriodicSpline | FBspline
  | FUnknown of string
      (** A word that is not a forcing kind, kept as written so that the
          expander can name it when it reports the declaration (E408). *)

(** The kind a word names; a word that names none is [FUnknown]. *)
let forcing_kind_of_string = function
  | "sinusoidal"      -> FSinusoidal
  | "piecewise"       -> FPiecewise
  | "interpolated"    -> FInterpolated
  | "periodic"        -> FPeriodic
  | "fourier"         -> FFourier
  | "periodic_spline" -> FPeriodicSpline
  | "bspline"         -> FBspline
  | s                 -> FUnknown s

(** The kind as the model file writes it, the inverse of
    {!forcing_kind_of_string}. *)
let forcing_kind_name = function
  | FSinusoidal     -> "sinusoidal"
  | FPiecewise      -> "piecewise"
  | FInterpolated   -> "interpolated"
  | FPeriodic       -> "periodic"
  | FFourier        -> "fourier"
  | FPeriodicSpline -> "periodic_spline"
  | FBspline        -> "bspline"
  | FUnknown s      -> s

(** One entry of the [forcing] block: a named function of time, such as a
    seasonal factor or a population read from a file, with the unit of its
    values. *)
type func_decl = {
  fname    : string;
  findices : index_binding list;
      (** Binders for a family of forcings, one per level. *)
  fkind    : forcing_kind;
  funit    : unit_lit;
      (** The unit of the forcing's values, required:
          [pop : interpolated 'count], [seasonal : sinusoidal 'ratio]. The
          dimension checker takes the forcing's dimension from it. *)
  fargs    : (string * farg_value) list;
  fspan    : loc;
}

(** The [trajectories] entry of the [output] block: when a simulation records
    the state. *)
type output_traj_decl = {
  otschedule  : schedule_core;
      (** [every = 7 'days] or [at = [...]]. With neither, the parser fills
          [every = 1], in the model's time unit. *)
  otformat    : string;
      (** [format = ...]; ["tsv"] when absent. It is kept in the declared
          model and does not reach the compiled model; the language
          specification states that the writer always emits wide TSV whatever
          it says. *)
}

(** The [output] block. *)
type output_decl = {
  out_trajectories: output_traj_decl option;
      (** [None] keeps the default: a row every 1 time unit over the run. *)
  out_span: loc;
}

(** The [simulate] block: the run's time window, step and ODE integrator.

    {@camdl[
      from = 0 'days
      to   = 365 'days
      dt   = 0.5 'days
      integrator = rk45 { atol = 1e-8  rtol = 1e-6 }
    ]} *)
type simulate_decl = {
  sim_from: Time_pos.instant Time_pos.t;
      (** [from]: the start, an instant; [0] when absent. *)
  sim_to: Time_pos.instant Time_pos.t;
      (** [to]: the end, an instant, which may be an observation anchor such
          as [last_obs + 4 'weeks]. Required: when it is absent the parser
          reports E106 and leaves [100] here as a recovery value. *)
  sim_to_loc: loc option;
      (** The span of the whole [to = ...] entry; [None] when it is
          absent. *)
  sim_dt: Time_pos.length Time_pos.t option;
      (** [dt]: the step of the discrete-time backends, a length. *)
  sim_integrator: (string * loc) option;
      (** [integrator = rk4] or [rk45], as written, with its span; the
          expander checks the name. [None] means rk4. *)
  sim_atol: (expr * loc) option;
      (** [atol] in an [rk45 { ... }] block: a dimensionless absolute
          tolerance, with its span. *)
  sim_rtol: (expr * loc) option;
      (** [rtol] in an [rk45 { ... }] block: a dimensionless relative
          tolerance, with its span. *)
  sim_span: loc;
}

(** One entry of the [timepoints] block: a named instant. The expander
    judges each time as an instant position, so a mistake in one is
    reported, but leaves the block out of the compiled model, so the names
    are not defined anywhere in it. *)
type timepoint_decl = { tpname: string; tptime: Time_pos.instant Time_pos.t; tpspan: loc }

(** One field of a scenario: a change the scenario makes to the model.

    {@camdl[
      extends = baseline
      enable  = [sia]
      scale = { beta = 1.5 }
      simulate { to = 730 'days }
    ]} *)
type scenario_field =
  | ScLabel   of string
      (** [label = "no SIA"]: a display name. A bare word or a number is
          accepted too. *)
  | ScEnable  of string list
      (** [enable = [sia]]: interventions to turn on. The value must be a
          bracketed list of names; [enable = sia], without brackets, is
          E123. *)
  | ScDisable of string list
      (** [disable = [sia]]: interventions to turn off. The list is read as
          for [ScEnable]. *)
  | ScSet     of (string * expr) list
      (** [set = { cov = 0.9 }]: parameter values. An indexed entry
          [beta[child] = 0.3] is stored under the joined name ["beta_child"].
          A field [beta = 0.3] outside a block is a [ScSet] of one parameter.
          A field [beta = { ... }] under any name other than [set] or [scale]
          is stored as setting the parameter of that name to [0], and the
          block's contents are dropped. *)
  | ScScale   of (string * expr) list
      (** [scale = { beta = 1.5 }]: multiplies parameters by factors. *)
  | ScCompose of string list
      (** [compose = [with_sia]]: other scenarios whose changes apply in
          sequence. The list is read as for [ScEnable]. *)
  | ScTEnd    of Time_pos.instant Time_pos.t
      (** [simulate { to = ... }]: the scenario's end, an instant, the only
          setting of [simulate] a scenario may change. *)
  | ScExtends of string
      (** [extends = baseline]: inherit the parent's fields; the parent's
          enable, disable and compose lists come before the child's. *)

(** One scenario of the [scenarios] block: a named set of changes to the
    model, run as one arm of a comparison. *)
type scenario_decl = {
  scname   : string;
  scfields : scenario_field list;
  scspan   : loc;
}

(** Where a dimension's levels come from. *)
type dim_source =
  | DInline of string list
      (** [age = [child, adult]]: the levels, in order. *)
  | DRead   of { fn_name: string; path: string; col_kw: string; col: string }
      (** [patch = read("data/pop.tsv", column = "patch")]: the levels listed
          in a column of a data file. The parser accepts any call of the shape
          [f("...", k = "...")] and keeps [f] as [fn_name] and [k] as
          [col_kw]; the expander requires them to be [read] and [column]. *)

(** One entry of the [dimensions] block: a named set of levels, such as age
    groups or patches, by which compartments are stratified and families are
    indexed.

    {@camdl[
      age   = [child, adult]
      patch = read("data/seir_defines_pop.tsv", column = "patch")
    ]} *)
type dimensions_entry = {
  dename : string;
  desrc  : dim_source;
  dedoc  : doc option;
  despan : loc;
}

(** The [balance] block: a compartment ([bcomp]) overwritten after every step
    with an expression ([bexpr]), so that a population identity holds. Only
    the chain-binomial backend supports it.

    {@camdl[
      R = pop(t) - S - E - I - Q
    ]} *)
type balance_decl = { bcomp: string; bexpr: expr; bspan: loc }

(** One top-level item of a model file. A file parses to a list of these, in
    source order. The expander joins the entries of repeated list blocks
    ([compartments], [parameters], [transitions], ...); [time_unit],
    [origin], [description], [simulate], [output] and [balance] may each be
    written once, and a second one is refused (E140). *)
type declaration =
  | DTimeUnit    of unit_lit * loc
      (** [time_unit = 'days]: the unit of the model's time axis. *)
  | DDescription of string * loc
      (** [description = "..."]: a one-line description of the model, with
          the declaration's span. *)
  | DModelDoc    of doc
      (** The [#'] block at the head of the file, which documents the model
          rather than a declaration. Only the first item of the list can be
          one. *)
  | DOrigin      of string * loc
      (** [origin = date("YYYY-MM-DD")]: the calendar date of time zero, as
          the text inside [date(...)] ([""] after a parse error), with the
          declaration's span. *)
  | DDimensions  of dimensions_entry list
  | DCompartments of compartment_decl list
  | DParameters   of param_decl list
  | DTables       of table_decl list
  | DForcing      of func_decl list
  | DTransitions  of transition_decl list
  | DObservations of obs_decl list
  | DInterventions of intervention_decl list
      (** [interventions]: actions that fire only when enabled. *)
  | DEvents        of intervention_decl list
      (** [events]: actions that always fire. *)
  | DReactiveInterventions of reactive_decl list
  | DODE          of ode_decl list
  | DOutput       of output_decl
  | DSimulate     of simulate_decl
  | DInit         of init_entry list
  | DTimepoints   of timepoint_decl list
  | DStratify     of stratify_decl
      (** One [stratify(...)] line. *)
  | DLet          of let_binding
      (** One [let]. *)
  | DScenarios    of scenario_decl list
  | DBalance      of balance_decl
  | DQuantities   of quantity_decl list
  | DContrasts    of contrast_decl list

(** [l] with [file] set to [file], unless it already names a file.

    A [quantities] block can come from a second file
    ([camdlc MODEL.camdl --quantities VOCAB.camdl]), and a diagnostic about it
    must name that file, not the model. The parser leaves every [file] empty
    and the expander fills it with the model's name, so the declarations read
    from the second file are stamped with its path before they join the
    model's list. *)
let stamp_loc_file ~file (l : loc) : loc =
  if l.file = "" then { l with file } else l

(** [e] with every location in it stamped by {!stamp_loc_file}. *)
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
  | ESumStages (v, c, body, l) -> ESumStages (v, c, go body, stamp_loc_file ~file l)
  | ECond (p, a, b) -> ECond (go p, go a, go b)
  | EFuncCall (f, kvs) -> EFuncCall (f, List.map (fun (k, v) -> (k, go v)) kvs)
  | EList xs -> EList (List.map go xs)
  | ERange (a, b) -> ERange (go a, go b)
  | EObsAccess (s, l) -> EObsAccess (s, stamp_loc_file ~file l)
  | ERunMember r -> ERunMember { r with loc = stamp_loc_file ~file r.loc }

and stamp_index_item_file ~file = function
  | IPosn e -> IPosn (stamp_expr_file ~file e)
  | INamed (n, e) -> INamed (n, stamp_expr_file ~file e)
  | IStage e -> IStage (stamp_expr_file ~file e)

(** A quantity declaration with its body and spans stamped by
    {!stamp_loc_file}. *)
let stamp_quantity_decl_file ~file (qd : quantity_decl) : quantity_decl =
  { qd with qd_body = stamp_expr_file ~file qd.qd_body;
            qd_loc  = stamp_loc_file ~file qd.qd_loc;
            qd_span = stamp_loc_file ~file qd.qd_span }
