(** The declared model: a camdl model as its author wrote it, before
    expansion, and its JSON codec. It is the [declared] section of the
    envelope [camdlc] writes beside the compiled model.

    Expansion keeps the meaning of a model and loses its form. In the
    intermediate representation [infection[p in patch]] has become one
    transition per patch, a [let] may have been inlined into every expression
    that uses it, and every unit literal has been converted into the model's
    time unit. The declared model keeps the form. A {!declared_model} holds one
    entry per declaration, grouped by block, each block's entries in
    declaration order. Every expression is its parsed tree without source
    locations, an {!expr_tree}, wrapped in a {!declared_expr} that adds a
    KaTeX rendering, the ways that rendering falls short of the tree, and the
    names the tree reads. Every name in a tree is a {!name_ref}: the name and
    what the compiler resolved it to, so that a parameter and a compartment
    of the same spelling are told apart. Every declaration also carries its
    text exactly as written, a {!source_text}. A few declarations record what
    the compiler resolved and the declaration alone does not state: the levels
    of a dimension, the dimensions of a compartment, and the expanded names a
    compartment, transition or observation stream produced. [#'] doc blocks
    are not recorded.

    Numbers and units are as written. In a model whose time unit is weeks,
    [at [180 'days]] is recorded as the number 180 with the unit days, and
    its JSON, copied from [camdlc] output, is

    {@json[
      {"tree": {"unit": {"value": 180.0, "unit": "days"}}, "katex": "180",
       "katex_loss": ["unit_dropped"], "reads": []}
    ]}

    The reader is [camdl diff] ([rust/crates/diff]), which compares two models
    declaration by declaration. It treats two declarations as the same when
    their JSON documents are equal once the members that exist only for
    display ([katex], [katex_loss], [reads], [source_text] and [expands_to])
    are left out, so two expressions are the same exactly when they parse to
    the same tree. Nothing is normalised: [beta * S * I] and [S * beta * I]
    are different trees, while [×] and [*] parse to the same one. Which
    declarations a change affects is followed through the {!name_ref}s in each
    declaration, so a name that is not a {!name_ref} (a table named in a
    [where] predicate, a stream named by [observations.cases]) is a
    dependency the diff does not see.

    {!Declared_emit} builds the value from a successful compile; this module
    only defines it and converts it. {!to_json} writes it, and {!of_json}
    reads it back, refusing anything the Rust reader refuses and any
    [schema_version] other than {!schema_version}.

    The Rust crate [rust/crates/declared] owns the format: every type, field
    name and variant tag here mirrors one there, with the OCaml field prefix
    ([nr_], [dd_], ...) dropped on the wire. The two sides are held together
    by one committed fixture,
    [rust/crates/declared/tests/fixtures/all_variants.json], which
    [ocaml/test/test_declared.ml] and
    [rust/crates/declared/tests/ocaml_fixture.rs] both read. The wire
    conventions are those of the Rust serde derives:

    - a record is an object carrying every field, and an absent option is
      [null];
    - a constructor without data is its snake_case name as a string, even in
      a type whose other constructors carry data;
    - a constructor with data is [{"<tag>": {<fields>}}];
    - an unknown field or tag is refused on both sides, because a reader that
      skipped one could call two different models the same.

    Constructors and record fields carry a per-type prefix, as in {!Ast}, so
    the module can be opened beside [Ast] and [Ir] without capture. A
    parameter kind is an {!Ir.param_kind}, the type the compiled model
    uses. *)

(* ── Types ───────────────────────────────────────────────────────────────── *)

(** The schema version this module writes, and the only one {!of_json}
    reads. *)
let schema_version = 1

(** A unit written after a number with a leading quote, as in [14 'days] or
    [0.1 'per_day]; also the model's time unit, the unit on a table's axis
    and the unit of a forcing's values. One constructor per {!Ast.unit_lit},
    with the same meaning. *)
type unit_lit =
  | UDays | UWeeks | UMonths | UYears
  | UPerDay | UPerWeek | UPerMonth | UPerYear
  | UCount | URatio

(** A binary operator of an expression, one per {!Ast.bin_op}. Both [×] and
    [*] are [BMul]. *)
type bin_op =
  | BAdd | BSub | BMul | BDiv | BPow
  | BEq | BNeq | BLt | BGt | BLe | BGe

(** A unary operator of an expression, one per {!Ast.un_op}. The parser
    writes a call such as [exp(x)] as a {!TCall}, so of these only [ONeg]
    and [OTotal] occur in a compiled model's section. *)
type un_op =
  | ONeg | OExp | OLog | OSqrt | OAbs | OFloor | OCeil | OSin | OCos | OTanh
  | OTotal
      (** [sum(I)] with no binder: the value of its operand, marked as a
          total the modeller stated. *)

(** Which kind of member a run reference in a contrast names: an entry of
    the [quantities] block, or an observation stream. *)
type run_namespace = NsQuantities | NsObservations

(** What a name in an expression refers to, as the compiler resolved it at
    that position.

    The constructors on the first line come from the compiler's own name
    resolution ([Expander.classify_name]); the others name the things a
    model reads that are not in that chain. In
    [infection[p in patch] : S[p] --> I[p] @ beta * season * S[p] * ...],
    [beta] is a [KParameter], [season] a [KTimeInput] (a forcing), [S] a
    [KCompartment] and [p] a [KIndexVar]. *)
type name_kind =
  | KCompartment | KParameter | KDefinition | KTable | KTimeInput
  | KIndexVar
      (** A variable bound by the declaration's index binders or by an
          enclosing [sum]. *)
  | KLevel
      (** A level of a dimension written in an index position, as [urban] in
          [S[urban]]. *)
  | KBuiltin
      (** A name the language defines: [t], [dt], [projected], [pi], [e],
          [origin], [t_start] or [t_end]. *)
  | KDataColumn
      (** A column of an observation stream's data file, read by its
          likelihood (the binomial [n = tested]) or by its [covers] span. *)
  | KQuantity
      (** An entry of the [quantities] block, named in a quantity or
          contrast body. *)
  | KStream
      (** An observation stream, named in a quantity or contrast body, a
          projection or a reactive trigger, as in
          [observed(weekly_cases)]. *)
  | KTransition
      (** A transition, or one expanded member of it, named in the same
          positions, as in [incidence(infection[p])]. *)
  | KDimension
      (** A dimension named by a bare word in a forcing argument, as in
          [time_dim = week]. *)
  | KObsAnchor
      (** [first_obs] or [last_obs]: a time read from a run's data, not from
          the model. *)
  | KKeyword
      (** A word the enclosing form reads by its spelling, not as a name:
          [true] and [false] as a call argument or a [once] value, the [dim]
          argument of [unchecked_dim(...)], the [label] of a [hyper_erlang]
          branch, and a bare forcing argument that names nothing declared,
          such as [method = linear]. *)
  | KUnresolved
      (** No rule classifies the name at this position. A compiled model is
          still correct, but the compiler warns (W203), because a diff cannot
          follow the dependencies of the declaration that holds it. *)

(** A name in an expression, with what it refers to. Two references are
    equal only when both the spelling and the kind agree. *)
type name_ref = { nr_name : string; nr_kind : name_kind }

(** A comparison between a table cell and a number in a [where] predicate,
    one per {!Ast.relop}. *)
type rel_op = RLt | RLe | RGt | RGe | REq | RNe

(** The right-hand side of a table-cell comparison in a [where] predicate: a
    number ([GoNum]), or a bare name ([GoName]). The compiler refuses a bare
    name (error E284), so a compiled model's section holds only [GoNum]. *)
type guard_operand = GoNum of float | GoName of string

(** A [where] predicate: a condition on index values that keeps some members
    of a family or some terms of a [sum], decided by the compiler for each
    combination of levels. The names in it are plain strings, not
    {!name_ref}s: an index variable, a level name, or the table of a
    [GTable].

    {@camdl[
      waning[p in patch] : R[p] --> S[p] @ 0.01 'per_day * R[p] where p == rural
    ]} *)
type guard =
  | GEq    of string * string
      (** [p == rural]: the two sides name the same level. Each side is an
          index variable or a level name. *)
  | GNeq   of string * string
      (** [q != p]. *)
  | GTable of { gt_table : string; gt_indices : string list; gt_op : rel_op;
                gt_operand : guard_operand }
      (** [dist[p, q] < 50]: the table, the index variables or level names
          that select its cell, the comparison and the operand. *)
  | GAnd   of guard * guard
  | GOr    of guard * guard

(** A parsed expression, {!Ast.expr}, without its source locations, and with
    each name tagged by what it refers to.

    This is the value on which sameness is decided, so it keeps exactly what
    the parser keeps. [(a * b)] and [a * b] are one tree, because the parser
    has no node for parentheses; a flat [sum(a in age, p in patch, body)] is
    two nested [TSum] nodes. *)
type expr_tree =
  | TConst     of float
      (** A number literal. [1] and [1.0] are the same constant. *)
  | TUnit      of float * unit_lit
      (** A number with a unit, [7 'days], unconverted. *)
  | TIdent     of name_ref
      (** A bare name. *)
  | TIndex     of name_ref * index_item list
      (** An indexed name, [S[p]] or [C[p, q]]. *)
  | TBinOp     of bin_op * expr_tree * expr_tree
  | TUnOp      of un_op * expr_tree
  | TSum       of { ts_var : string; ts_dim : string; ts_guard : guard option;
                    ts_body : expr_tree }
      (** [sum(q in patch where P, body)]: the variable, the dimension, the
          optional predicate and the body. Inside [ts_body] the variable is a
          {!KIndexVar}. *)
  | TCond      of expr_tree * expr_tree * expr_tree
      (** [if pred then a else b]: the predicate, then the two branches. *)
  | TCall      of string * arg list
      (** A call by name, [incidence(infection[p])] or
          [neg_binomial(mean = m, r = 10)]. A positional argument has the
          name [""]. The function name is a plain string, not a
          {!name_ref}. *)
  | TList      of expr_tree list
      (** A bracketed list, [[0.9, 0.1]]; a list of lists is a matrix. *)
  | TRange     of expr_tree * expr_tree
      (** [7:100], written only as an element of a list. *)
  | TObsAccess of string
      (** [observations.cases] in a quantity body: the simulated values of
          the stream named. *)
  | TStr       of string
      (** A quoted string: a file path, a date, a label. It is not a name,
          and is never in a [reads] list. *)
  | TRunMember of { tr_run : string; tr_namespace : run_namespace;
                    tr_member : string }
      (** [with_sia.quantities.total_deaths] in a contrast body: the run (a
          scenario name, or [fitted] for the model with no scenario), the
          namespace and the member. *)

(** One index inside the brackets of a reference: by position, [S[urban]],
    or by the dimension's name, [S[patch = urban]]. *)
and index_item =
  | IPositional of expr_tree
  | INamed      of string * expr_tree
      (** The dimension's name, then the index. *)

(** One argument of a call, a distribution, a prior or a [via] law: its
    keyword, [""] for a positional argument, and its value. *)
and arg = { a_name : string; a_value : expr_tree }

(** A binder that makes a declaration a family, written in brackets after
    its name: [p in patch] ([BIn]: the variable, then the dimension),
    [(a, a_next) in consecutive(age)] ([BConsecutive]: each pair of adjacent
    levels; the two variables, then the dimension), or [c in compartments]
    ([BCompartments]: the variable ranges over the integer compartments). *)
type index_binding =
  | BIn           of string * string
  | BConsecutive  of string * string * string
  | BCompartments of string

(** A declaration's text exactly as its author wrote it, from the first
    token of the declaration to its last; a [#'] doc block above it is not
    part of it. It is for display only: declarations are compared on their
    parsed trees, never on this text, because different spellings parse to
    one tree.

    [None] means the compiler had no text for the declaration. A
    declaration the compiler made itself has none; every declaration of
    every compiled golden model has its text, which a test checks. *)
type source_text = string option

(** One way the KaTeX rendering of an expression fails to show what its tree
    holds: something the tree has that the rendering leaves out, or a
    grouping the rendering does not keep. [Latex.tex_loss] computes it, beside
    the renderer it describes, and errs towards reporting. *)
type katex_loss =
  | LUnitDropped
      (** [0.5 'per_day] is rendered [0.5]. *)
  | LNumberRounded
      (** The rendered digits do not read back as the number. *)
  | LPlaceholder
      (** A list, a range, [observations.x] or a run member is rendered as a
          dot. *)
  | LGroupingChanged
      (** A product or quotient is regrouped into one fraction, or an
          operand is rendered without the parentheses its position needs. *)
  | LSumRangeDropped
      (** A [sum] is rendered without its dimension and its [where]
          predicate. *)
  | LArgNamesDropped
      (** [f(rate = x)] is rendered [f(x)]. *)
  | LIndexDimDropped
      (** [S[age = a]] is rendered as [S[a]] is. *)
  | LStringAsName
      (** A quoted string is rendered as a name is. *)
  | LSymbolOverride
      (** A name is rendered as the [@symbol] its doc block gives it. *)

(** A top-level expression of a declaration: a rate, a bound, a [let] body,
    or any other expression a declaration holds in a field of its own. The
    tree is what sameness is decided on; the other three fields are derived
    from it for display.

    A reader shows the declaration's text as written beside a rendering
    whose [de_katex_loss] is not empty. The arguments of a call, a prior or a
    [via] law are bare {!arg} trees inside their parent and carry no
    rendering of their own. *)
type declared_expr = {
  de_tree  : expr_tree;
  de_katex : string;
      (** The expression rendered as KaTeX, as [camdlc render] renders it. *)
  de_katex_loss : katex_loss list;
      (** Why [de_katex] does not show the whole tree, each reason once, in
          the order of {!katex_loss}; empty when it does. *)
  de_reads : name_ref list;
      (** Every {!name_ref} in the tree, each spelling-and-kind pair once, in
          order of first appearance; index variables included. The names in
          a [sum]'s predicate, a [TObsAccess] stream and a [TRunMember] are
          not {!name_ref}s and are not listed. *)
}

(** One [--set name=value] given to [camdlc]: a parameter pinned to a fixed
    value for this compile. [camdlc] refuses a name that is not a
    parameter. *)
type set_value  = { sv_name : string; sv_value : float }

(** One file the compile read, by its path and the SHA-256 of its bytes.

    [fi_path] is relative to the model file's directory. A data file keeps
    the path the model wrote, unless the model wrote an absolute path, which
    is rewritten relative to the model's directory; the [--quantities] file
    is recorded relative to the model's directory. [fi_sha256] is the
    lowercase hex digest, and [None] only when the file could not be
    re-opened to hash it. *)
type file_input = { fi_path : string; fi_sha256 : string option }

(** What entered the compile from outside the model file. Two compiles of one
    file that differ here can produce different models. *)
type compile_inputs = {
  ci_set_values : set_value list;
      (** The [--set] overrides, in the order given. *)
  ci_files      : file_input list;
      (** The data files the model read (dimension levels, tables,
          forcings), each once, in the order first read, then the
          [--quantities] file when one was given. *)
}

(** Where a dimension's levels come from: listed inline, or read from a
    column of a data file, [read("pop.tsv", column = "patch")] with its
    path and column. *)
type dim_source =
  | DsInline
  | DsRead of { ds_path : string; ds_column : string }

(** One entry of the [dimensions] block.

    {@camdl[
      patch = [urban, rural]
    ]} *)
type dimension_decl = {
  dd_name   : string;
  dd_source : dim_source;
  dd_levels : string list;
      (** The levels the compile used, in order: those written, or those
          read from the file. *)
  dd_text : source_text;
}

(** Whether a compartment holds whole individuals ([CkInteger], the default)
    or a continuous quantity ([CkReal], written [W : real]). *)
type compartment_kind = CkInteger | CkReal

(** One compartment of the [compartments] block, with the stratification the
    compiler resolved for it.

    {@camdl[
      compartments { S, I, R, D }
      stratify(by = patch, only = [S, I, R])
    ]}

    records [S] with dimensions [["patch"]] expanding to
    [["S_urban"; "S_rural"]], and [D] with no dimensions expanding to
    [["D"]]. *)
type compartment_decl = {
  cd_name       : string;
  cd_kind       : compartment_kind;
  cd_dims       : string list;
      (** The dimensions the [stratify] declarations apply to this
          compartment, in declaration order. A stage dimension the compiler
          adds for a [via] law is not listed. *)
  cd_expands_to : string list option;
      (** The compartments of the compiled model this declaration produced,
          in the compiled model's order. For a compartment a [via] law
          stages, these are the stage compartments ([E_s1], [E_s2], ...).
          [camdlc] always records it; [None] means a writer did not. *)
  cd_text : source_text;
}

(** One [stratify(...)] declaration: [st_dim] is the dimension from [by],
    and [st_only] the compartments from [only], [None] meaning all of
    them. *)
type stratify_decl = { st_dim : string; st_only : string list option;
  st_text : source_text }

(** A dimension stated by a [dim] clause, as the exponents of population and
    of time: [dim P^-1*T^-1] is population [-1], time [-1]. *)
type dim_annotation = { da_population : int; da_time : int }

(** The interval of a parameter's [in [lo, hi]]. *)
type bounds = { b_lo : declared_expr; b_hi : declared_expr }

(** A parameter's prior, written after [~]: the distribution's name as
    written ([pr_family], such as ["log_normal"]), its keyword arguments, and
    the dimension of a [| age] pooling clause ([pr_pool_over]).

    {@camdl[
      beta   : rate in [0.01, 2.0] ~ log_normal(mu = -1, sigma = 0.5)
    ]} *)
type prior_decl = {
  pr_family    : string;
  pr_args      : arg list;
  pr_pool_over : string option;
}

(** One entry of the [parameters] block: a value the model reads and a fit
    can estimate. *)
type parameter_decl = {
  pd_name   : string;
  pd_dims   : string list;
      (** The dimensions of an indexed parameter, [k_p[patch]]; empty for a
          scalar. *)
  pd_kind   : Ir.param_kind;
  pd_dim    : dim_annotation option;
      (** A [dim] clause; only the [positive] and [real] kinds accept
          one. *)
  pd_bounds : bounds option;
  pd_prior  : prior_decl option;
  pd_text : source_text;
}

(** A top-level [let]: a named expression, which may be indexed, typed or
    shaped. At most one of [df_shape] and [df_kind] is set.

    {@camdl[
      let N[p in patch] = S[p] + I[p] + R[p]
      let b0 : positive dim P^-1*T^-1 = 0.0001
    ]} *)
type definition_decl = {
  df_name    : string;
  df_indices : index_binding list;
  df_shape   : string list option;
      (** The dimensions of a shaped literal, [let w : age × patch = ...],
          whose body is a nested list laid out over them. *)
  df_kind    : Ir.param_kind option;
      (** The kind of a typed [let], [let b0 : positive = ...]. *)
  df_dim     : dim_annotation option;
  df_body    : declared_expr;
  df_text : source_text;
}

(** One axis of a table's shape: a dimension and, when written on it, the
    unit of the table's cells, as in [age 'years]. *)
type table_dim = { td_dim : string; td_unit : unit_lit option }

(** One entry of the [tables] block: constant data indexed by dimensions,
    written inline or read from a file.

    {@camdl[
      C : patch × patch = [[0.9, 0.1], [0.1, 0.9]]
    ]} *)
type table_decl = {
  tb_names     : string list;
      (** Several names, [pop, init_sus : patch = read(...)], declare one
          table per value column of the file. *)
  tb_dims      : table_dim list;
  tb_cell_kind : Ir.param_kind option;
  tb_value     : declared_expr;
      (** The data: a list literal, or the call that reads it, such as
          [read("dist.tsv", default = 1000000.0)]. *)
  tb_text : source_text;
}

(** The value of one argument of a forcing declaration. A quoted string
    ([TvStr]) names something outside the model, a data file or one of its
    columns; an expression ([TvExpr]) names something inside it, or is a word
    from a fixed set, such as [method = linear], recorded as a {!KKeyword}
    name. *)
type time_input_value =
  | TvStr  of string
  | TvExpr of declared_expr

(** One argument of a forcing declaration, by its keyword. *)
type time_input_arg = { ta_name : string; ta_value : time_input_value }

(** One entry of the [forcing] block: a named function of time supplied to
    the model, such as a seasonal factor.

    {@camdl[
      season : sinusoidal 'ratio {
        baseline  = 1.0
        amplitude = 0.2
        period    = 365.25
        phase     = 0.0
      }
    ]} *)
type time_input_decl = {
  ti_name    : string;
  ti_indices : index_binding list;
  ti_kind    : string;
      (** The kind as written: ["sinusoidal"], ["piecewise"],
          ["interpolated"], ["periodic"], ["fourier"], ["periodic_spline"] or
          ["bspline"]. *)
  ti_unit    : unit_lit;
      (** The unit of the forcing's values. *)
  ti_args    : time_input_arg list;
  ti_text : source_text;
}

(** One entry of the [timepoints] block: a named time. The compiler does not
    use the block, so nothing in the compiled model corresponds to it. *)
type timepoint_decl = { tp_name : string; tp_time : declared_expr;
  tp_text : source_text }

(** A compartment named on either side of a transition's arrow or in an
    action, with its indices: [S[p]], [D]. *)
type stoich_ref = { sr_compartment : string; sr_indices : index_item list }

(** One arm of a branching destination: the compartment and its weight. *)
type branch_arm = { ba_to : stoich_ref; ba_weight : declared_expr }

(** Where a transition sends the individuals it moves.

    {@camdl[
      recovery[p in patch]  : I[p] --> { R[p] : 1 - cfr, D : cfr } @ gamma * I[p]
    ]} *)
type destination =
  | DstSum    of stoich_ref list
      (** Every listed compartment gains one individual per firing. The
          empty list is both a transition whose individuals leave the model
          and a [via hyper_erlang] transition with no arrow target, whose
          branches name their own destinations. *)
  | DstBranch of branch_arm list
      (** One flow per arm, at the arm's weight times the transition's rate.
          The weights are not checked to sum to one. *)

(** How fast a transition fires: a rate ([DynRate]), or a staged residence,
    [via erlang(stages = 3, mean = 5 'days)] ([DynVia]: the law's name and
    its arguments as written, unchecked). *)
type dynamics =
  | DynRate of declared_expr
  | DynVia  of string * arg list

(** One entry of the [transitions] block: a flow of individuals between
    compartments.

    {@camdl[
      infection[p in patch] : S[p] --> I[p] @ beta * season * S[p] * sum(q in patch, C[p, q] * I[q] / N[q])
    ]} *)
type transition_decl = {
  tn_name       : string;
  tn_indices    : index_binding list;
  tn_from       : stoich_ref list;
      (** The compartments that each lose one individual per firing; empty
          for an inflow such as a birth. *)
  tn_to         : destination;
  tn_dynamics   : dynamics;
  tn_guard      : guard option;
      (** A [where] predicate that drops members of the family. *)
  tn_lineage    : bool;
      (** [true] when the transition carries the [#[lineage]] attribute. *)
  tn_expands_to : string list option;
      (** The transitions of the compiled model this declaration produced,
          in the compiled model's order: one per member of the family
          ([infection_urban], [infection_rural]), and for a [via] law the
          stage transitions the compiler made for it. [camdlc] always records
          it; [None] means a writer did not. *)
  tn_text : source_text;
}

(** One equation of the [ode] block: the time derivative of a real
    compartment. *)
type ode_decl     = { od_compartment : string; od_derivative : declared_expr;
  od_text : source_text }

(** A [balance] declaration: a compartment overwritten after every step with
    an expression, so that a population identity holds. *)
type balance_decl = { bl_compartment : string; bl_expr : declared_expr;
  bl_text : source_text }

(** The family of a distribution, one per {!Ast.likelihood_kind}.
    [zero_inflated(base = neg_binomial(...), pi = z)] is
    [FZeroInflatedNegBinomial], its arguments flattened. *)
type family =
  | FNegBinomial | FPoisson | FNormal | FBinomial | FBetaBinomial | FBeta
  | FBernoulli | FZeroInflatedNegBinomial

(** A distribution with its keyword arguments as written: an observation's
    likelihood, or the law an initial value is drawn from. *)
type distribution = { di_family : family; di_args : arg list }

(** What a compartment starts at: computed ([IvExpr], [S = N0 - I]), or drawn
    once at the start of each run ([IvLaw], [I ~ poisson(rate = I0)]). *)
type initial_value =
  | IvExpr of declared_expr
  | IvLaw  of distribution

(** One entry of the [init] block. At most one of [in_indices] and
    [in_bindings] is non-empty.

    {@camdl[
      S[urban] = 990  I[urban] = 10
    ]} *)
type initial_decl = {
  in_compartment : string;
  in_indices     : index_item list;
      (** [S[urban]]: the one stratum the entry sets. *)
  in_bindings    : index_binding list;
      (** [S[a in age]]: binders, one entry per level. *)
  in_value       : initial_value;
  in_text : source_text;
}

(** What one column of an observation's data file holds: the row's time
    ([CrTime]); the levels of the named dimension ([CrDim]); an observed
    value of a kind ([CrValue]); or the opening and closing boundaries of the
    period a row covers ([CrWindowStart], [CrWindowStop]). *)
type column_role =
  | CrTime
  | CrDim   of string
  | CrValue of Ir.param_kind
  | CrWindowStart
  | CrWindowStop

(** One entry of an observation's [columns] block. *)
type obs_column = { oc_name : string; oc_role : column_role }

(** An observation's measurement model, [weekly_cases ~ neg_binomial(...)]:
    the data column scored and the distribution it is scored under. *)
type measurement = { ms_scored : string; ms_likelihood : distribution }

(** The model quantity an observation stream measures, from its
    [projected = ...] line.

    [camdlc] writes only [PjDerived]: [incidence(infection[p])] is a
    {!TCall} inside it. [PjIncidence], a transition's flow by name and
    indices, is part of the format but never written by this compiler. *)
type projection =
  | PjIncidence of string * index_item list
  | PjDerived   of declared_expr

(** A set of times, by a regular step ([ScEvery], [every = 7 'days]) or
    listed ([ScAt], [at = [0, 30, 60]]): when an observation stream emits
    synthetic rows, or when the [output] block records the state. *)
type schedule_core =
  | ScEvery of declared_expr
  | ScAt    of declared_expr list

(** An observation's [covers] line: the period each row's count accumulates
    over, relative to the row's time label. *)
type covers = {
  cv_form   : string;
      (** The form as written: ["day"], ["starting_on"], ["ending_on"] or
          ["closing_at"]. *)
  cv_column : string;
      (** The time column named in the call. *)
  cv_span   : declared_expr option;
      (** The duration; [None] for [day(t)]. *)
}

(** One stream of the [observations] block: the columns of its data file,
    the model quantity the data measure, and the distribution the data are
    scored under.

    {@camdl[
      weekly_cases[p in patch] {
        columns       { time : time, patch : dim, weekly_cases : count }
        covers        = closing_at(time, 7 'days)
        projected     = incidence(infection[p])
        emit_schedule = every 7 'days
        weekly_cases  ~ neg_binomial(mean = rho * projected, r = 10)
      }
    ]} *)
type observation_decl = {
  ob_name          : string;
  ob_indices       : index_binding list;
  ob_source        : string option;
      (** [from label]: the key the data file is supplied under; [None]
          means the stream's name. *)
  ob_columns       : obs_column list;
  ob_measurement   : measurement;
  ob_projection    : projection;
  ob_emit_schedule : schedule_core option;
      (** When [simulate --obs] writes synthetic rows. A fit never reads
          it. *)
  ob_covers        : covers option;
  ob_expands_to    : string list option;
      (** The observation models of the compiled model this declaration
          produced, one per member of the family. [camdlc] always records it;
          [None] means a writer did not. *)
  ob_text : source_text;
}

(** What an intervention, event or reactive intervention does when it fires:
    [transfer(...)] with its keyword arguments as written ([AcTransfer]);
    [I = I + 10] in the block form, setting a compartment ([AcSet]); or
    [add(I[urban], 5)], adding to one ([AcAdd]). *)
type action =
  | AcTransfer of arg list
  | AcSet of { as_compartment : string; as_indices : index_item list;
               as_value : declared_expr }
  | AcAdd of { aa_compartment : string; aa_indices : index_item list;
               aa_count : declared_expr }

(** When an intervention or event fires: at each listed time
    ([ShAtTimes], [at [180 'days]]); every period within an optional window
    ([ShRecurring], [{ every = 30 'days  from = 0 'days  to = 2 'years }]);
    or once per period at an offset from time zero ([ShEveryAtDay],
    [every 365 'days at_day 100 'days]). *)
type schedule =
  | ShAtTimes    of declared_expr list
  | ShRecurring  of { sh_every : declared_expr; sh_from : declared_expr option;
                      sh_to : declared_expr option }
  | ShEveryAtDay of { sh_period : declared_expr; sh_at_day : declared_expr }

(** One entry of an [interventions] or an [events] block; the two blocks
    share this form and are kept apart in {!declared_model}.

    {@camdl[
      sia[p in patch] : transfer(fraction = cov, from = S[p], to = R[p]) at [180 'days]
    ]} *)
type intervention_decl = {
  iv_name     : string;
  iv_indices  : index_binding list;
  iv_actions  : action list;
      (** The actions in source order; only the block form holds more than
          one. *)
  iv_schedule : schedule;
  iv_guard    : guard option;
  iv_text : source_text;
}

(** The condition of a reactive intervention's [when] clause: comparisons
    ([TgAtom]) combined with [and], [or] and [not]. *)
type trigger_pred =
  | TgAtom of declared_expr
  | TgAnd  of trigger_pred * trigger_pred
  | TgOr   of trigger_pred * trigger_pred
  | TgNot  of trigger_pred

(** One entry of the [reactive_interventions] block: an action that fires
    when a condition on observed data holds.

    {@camdl[
      mop_up[p in patch] : when sum_observed(weekly_cases[p], window = 28 'days) >= 20 {
        after    = 14 'days
        action   = transfer(fraction = cov, from = S[p], to = R[p])
        once     = true
      }
    ]} *)
type reactive_decl = {
  rc_name     : string;
  rc_indices  : index_binding list;
  rc_when     : trigger_pred;
  rc_after    : declared_expr option;
      (** The delay from the trigger to the action; [None] means none. *)
  rc_once     : declared_expr option;
      (** [once = true] or [once = false], each word recorded as a
          {!KKeyword} name; [None] means [true]. *)
  rc_cooldown : declared_expr option;
      (** The least time between firings of a repeating policy. *)
  rc_action   : action;
  rc_guard    : guard option;
  rc_text : source_text;
}

(** [name = value] inside a scenario's [set] or [scale]. *)
type assign = { an_name : string; an_value : declared_expr }

(** One field of a scenario: a display label ([SfLabel]); interventions to
    turn on or off ([SfEnable], [SfDisable]); parameter values or factors
    ([SfSet], [SfScale]); other scenarios applied in sequence ([SfCompose]);
    the end time, from [simulate { to = ... }] ([SfTEnd]); or a parent to
    inherit from ([SfExtends]). *)
type scenario_field =
  | SfLabel   of string
  | SfEnable  of string list
  | SfDisable of string list
  | SfSet     of assign list
  | SfScale   of assign list
  | SfCompose of string list
  | SfTEnd    of declared_expr
  | SfExtends of string

(** One scenario of the [scenarios] block, its fields in source order. *)
type scenario_decl = { sn_name : string; sn_fields : scenario_field list;
  sn_text : source_text }

(** The [simulate] block: the run's time window, step and ODE integrator.

    The parser fills a missing [from] with [0] and a missing [to] with
    [100], so a block that leaves them out is recorded as if they were
    written. *)
type simulation_decl = {
  sm_from       : declared_expr;
  sm_to         : declared_expr;
  sm_dt         : declared_expr option;
  sm_integrator : string option;
      (** [rk4] or [rk45], as written; [None] means rk4. *)
  sm_atol       : declared_expr option;
  sm_rtol       : declared_expr option;
  sm_text : source_text;
}

(** The [trajectories] entry of the [output] block: when a simulation records
    the state, and the format named. The parser fills a missing schedule
    with [every = 1] and a missing format with ["tsv"]. *)
type trajectory_output = { to_schedule : schedule_core; to_format : string }

(** The [output] block; [ou_trajectories] is [None] when it has no
    [trajectories] entry. *)
type output_decl = { ou_trajectories : trajectory_output option;
  ou_text : source_text }

(** One entry of the [quantities] block: a quantity computed from a run.

    {@camdl[
      total_deaths = final(D)
    ]} *)
type quantity_decl = {
  qd_name    : string;
  qd_indices : index_binding list;
  qd_body    : declared_expr;
  qd_text : source_text;
}

(** One entry of the [contrasts] block: arithmetic over the quantities or
    streams of different scenarios' runs.

    {@camdl[
      averted = fitted.quantities.total_deaths - with_sia.quantities.total_deaths
    ]} *)
type contrast_decl = { ct_name : string; ct_body : declared_expr;
  ct_text : source_text }

(** One model, as declared: the whole [declared] section.

    Each list holds the entries of one kind of block in declaration order; a
    block written more than once contributes its entries in order. The
    single-valued fields ([dm_time_unit], [dm_description], [dm_origin],
    [dm_simulation], [dm_output]) hold the first declaration of their kind,
    and [None] when there is none. When a model writes one of these twice,
    the compiled model uses the last, without a diagnostic, so the two
    disagree; and [dm_balance] lists every [balance] declaration where the
    compiled model uses only the last. *)
type declared_model = {
  dm_schema_version         : int;
  dm_compile_inputs         : compile_inputs;
  dm_time_unit              : unit_lit option;
      (** [None] when the model does not write one; the compiler then uses
          days. *)
  dm_description            : string option;
  dm_origin                 : string option;
      (** The calendar date of time zero, as the text inside [date(...)]. *)
  dm_dimensions             : dimension_decl list;
  dm_compartments           : compartment_decl list;
  dm_stratify               : stratify_decl list;
  dm_parameters             : parameter_decl list;
  dm_definitions            : definition_decl list;
  dm_tables                 : table_decl list;
  dm_time_inputs            : time_input_decl list;
  dm_timepoints             : timepoint_decl list;
  dm_transitions            : transition_decl list;
  dm_ode_equations          : ode_decl list;
  dm_balance                : balance_decl list;
  dm_initial                : initial_decl list;
  dm_observations           : observation_decl list;
  dm_interventions          : intervention_decl list;
  dm_events                 : intervention_decl list;
  dm_reactive_interventions : reactive_decl list;
  dm_scenarios              : scenario_decl list;
  dm_simulation             : simulation_decl option;
  dm_output                 : output_decl option;
  dm_quantities             : quantity_decl list;
  dm_contrasts              : contrast_decl list;
      (** With [--quantities FILE], the [quantities] block is that file's,
          not the model's. *)
}

(* ── Codec helpers ───────────────────────────────────────────────────────── *)

type json = Yojson.Safe.t

let fail = Serde.fail

let str s : json = `String s
let int n : json = `Int n
let bool b : json = `Bool b
let obj kvs : json = `Assoc kvs
let list f xs : json = `List (List.map f xs)
let opt f : _ -> json = function None -> `Null | Some v -> f v
let strs = list str

(* JSON has no non-finite number, and `1` and `1.0` must read back as the same
   float on the Rust side, so a float is always written as a float. *)
let flt f : json =
  if Float.is_finite f then `Float f
  else invalid_arg "Declared: a non-finite number cannot be written to JSON"

(* `{"<tag>": {<fields>}}` *)
let variant tag fields : json = `Assoc [ (tag, `Assoc fields) ]

(* Strict readers: the Rust side refuses a number where a string is expected,
   so this one does too (`Serde.as_string` coerces an integer). *)
let as_str = function
  | `String s -> s
  | j -> fail "expected string, got %s" (Yojson.Safe.to_string j)

let as_list = Serde.as_list
let as_float = Serde.as_float
let as_bool = Serde.as_bool
let as_int = Serde.as_int

let of_list f j = List.map f (as_list j)
let of_strs = of_list as_str
let of_opt f = function `Null -> None | j -> Some (f j)

(* An object with exactly the fields [keys]: a missing or an unknown field is
   refused. Returns the field getter. *)
let record what keys (j : json) : string -> json =
  match j with
  | `Assoc kvs ->
    List.iter
      (fun (k, _) ->
        if not (List.mem k keys) then fail "%s: unknown field '%s'" what k)
      kvs;
    fun k ->
      (match List.assoc_opt k kvs with
       | Some v -> v
       | None -> fail "%s: missing field '%s'" what k)
  | _ -> fail "%s: expected an object, got %s" what (Yojson.Safe.to_string j)

(* A tagged value: a bare string (a data-less variant) or a one-key object. *)
let tagged what (j : json) : string * json =
  match j with
  | `String t -> (t, `Null)
  | `Assoc [ (t, body) ] -> (t, body)
  | _ ->
    fail "%s: expected a variant (a string or a one-key object), got %s" what
      (Yojson.Safe.to_string j)

(* A data-less enum: one table gives both directions. *)
let enum_to what table v : json =
  match List.find_opt (fun (v', _) -> v' = v) table with
  | Some (_, s) -> str s
  | None -> invalid_arg ("Declared: variant missing from the table of " ^ what)

let enum_of what table (j : json) =
  let s = as_str j in
  match List.find_opt (fun (_, s') -> s' = s) table with
  | Some (v, _) -> v
  | None -> fail "%s: unknown value '%s'" what s

(* ── Data-less enums ─────────────────────────────────────────────────────── *)

let unit_lit_table =
  [ (UDays, "days"); (UWeeks, "weeks"); (UMonths, "months"); (UYears, "years");
    (UPerDay, "per_day"); (UPerWeek, "per_week"); (UPerMonth, "per_month");
    (UPerYear, "per_year"); (UCount, "count"); (URatio, "ratio") ]

let bin_op_table =
  [ (BAdd, "add"); (BSub, "sub"); (BMul, "mul"); (BDiv, "div"); (BPow, "pow");
    (BEq, "eq"); (BNeq, "neq"); (BLt, "lt"); (BGt, "gt"); (BLe, "le");
    (BGe, "ge") ]

let un_op_table =
  [ (ONeg, "neg"); (OExp, "exp"); (OLog, "log"); (OSqrt, "sqrt");
    (OAbs, "abs"); (OFloor, "floor"); (OCeil, "ceil"); (OSin, "sin");
    (OCos, "cos"); (OTanh, "tanh"); (OTotal, "total") ]

let run_namespace_table =
  [ (NsQuantities, "quantities"); (NsObservations, "observations") ]

let name_kind_table =
  [ (KCompartment, "compartment"); (KParameter, "parameter");
    (KDefinition, "definition"); (KTable, "table");
    (KTimeInput, "time_input");
    (KIndexVar, "index_var"); (KLevel, "level"); (KBuiltin, "builtin");
    (KDataColumn, "data_column"); (KQuantity, "quantity");
    (KStream, "stream"); (KTransition, "transition");
    (KDimension, "dimension"); (KObsAnchor, "obs_anchor");
    (KKeyword, "keyword"); (KUnresolved, "unresolved") ]

let katex_loss_table =
  [ (LUnitDropped, "unit_dropped"); (LNumberRounded, "number_rounded");
    (LPlaceholder, "placeholder"); (LGroupingChanged, "grouping_changed");
    (LSumRangeDropped, "sum_range_dropped");
    (LArgNamesDropped, "argument_names_dropped");
    (LIndexDimDropped, "index_dimension_dropped");
    (LStringAsName, "string_as_name"); (LSymbolOverride, "symbol_override") ]

let rel_op_table =
  [ (RLt, "lt"); (RLe, "le"); (RGt, "gt"); (RGe, "ge"); (REq, "eq");
    (RNe, "ne") ]

let compartment_kind_table = [ (CkInteger, "integer"); (CkReal, "real") ]

let family_table =
  [ (FNegBinomial, "neg_binomial"); (FPoisson, "poisson");
    (FNormal, "normal"); (FBinomial, "binomial");
    (FBetaBinomial, "beta_binomial"); (FBeta, "beta");
    (FBernoulli, "bernoulli");
    (FZeroInflatedNegBinomial, "zero_inflated_neg_binomial") ]

let unit_lit_to_json = enum_to "unit" unit_lit_table
let unit_lit_of_json = enum_of "unit" unit_lit_table
let bin_op_to_json = enum_to "bin_op" bin_op_table
let bin_op_of_json = enum_of "bin_op" bin_op_table
let un_op_to_json = enum_to "un_op" un_op_table
let un_op_of_json = enum_of "un_op" un_op_table
let run_namespace_to_json = enum_to "namespace" run_namespace_table
let run_namespace_of_json = enum_of "namespace" run_namespace_table
let name_kind_to_json = enum_to "name kind" name_kind_table
let name_kind_of_json = enum_of "name kind" name_kind_table
let katex_loss_to_json = enum_to "katex loss" katex_loss_table
let katex_loss_of_json = enum_of "katex loss" katex_loss_table
let rel_op_to_json = enum_to "rel_op" rel_op_table
let rel_op_of_json = enum_of "rel_op" rel_op_table
let compartment_kind_to_json = enum_to "compartment kind" compartment_kind_table
let compartment_kind_of_json = enum_of "compartment kind" compartment_kind_table
let family_to_json = enum_to "family" family_table
let family_of_json = enum_of "family" family_table

let param_kind_to_json k = str (Ir.param_kind_name k)

let param_kind_of_json j =
  let s = as_str j in
  match Ir.param_kind_of_name s with
  | Some k -> k
  | None -> fail "parameter kind: unknown value '%s'" s

(* ── Expressions ─────────────────────────────────────────────────────────── *)

let source_text_to_json : source_text -> json = opt str
let source_text_of_json : json -> source_text = of_opt as_str

let name_ref_to_json r =
  obj [ ("name", str r.nr_name); ("kind", name_kind_to_json r.nr_kind) ]

let name_ref_of_json j =
  let g = record "name" [ "name"; "kind" ] j in
  { nr_name = as_str (g "name"); nr_kind = name_kind_of_json (g "kind") }

let guard_operand_to_json = function
  | GoNum v  -> variant "num" [ ("value", flt v) ]
  | GoName n -> variant "name" [ ("name", str n) ]

let guard_operand_of_json j =
  match tagged "guard operand" j with
  | "num", b  -> GoNum (as_float (record "num" [ "value" ] b "value"))
  | "name", b -> GoName (as_str (record "name" [ "name" ] b "name"))
  | t, _ -> fail "guard operand: unknown variant '%s'" t

let rec guard_to_json = function
  | GEq (l, r)  -> variant "eq" [ ("lhs", str l); ("rhs", str r) ]
  | GNeq (l, r) -> variant "neq" [ ("lhs", str l); ("rhs", str r) ]
  | GTable t ->
    variant "table"
      [ ("table", str t.gt_table); ("indices", strs t.gt_indices);
        ("op", rel_op_to_json t.gt_op);
        ("operand", guard_operand_to_json t.gt_operand) ]
  | GAnd (l, r) ->
    variant "and" [ ("lhs", guard_to_json l); ("rhs", guard_to_json r) ]
  | GOr (l, r) ->
    variant "or" [ ("lhs", guard_to_json l); ("rhs", guard_to_json r) ]

let rec guard_of_json j =
  let pair what b f =
    let g = record what [ "lhs"; "rhs" ] b in
    (f (g "lhs"), f (g "rhs"))
  in
  match tagged "guard" j with
  | "eq", b  -> let l, r = pair "eq" b as_str in GEq (l, r)
  | "neq", b -> let l, r = pair "neq" b as_str in GNeq (l, r)
  | "table", b ->
    let g = record "table guard" [ "table"; "indices"; "op"; "operand" ] b in
    GTable
      { gt_table = as_str (g "table"); gt_indices = of_strs (g "indices");
        gt_op = rel_op_of_json (g "op");
        gt_operand = guard_operand_of_json (g "operand") }
  | "and", b -> let l, r = pair "and" b guard_of_json in GAnd (l, r)
  | "or", b  -> let l, r = pair "or" b guard_of_json in GOr (l, r)
  | t, _ -> fail "guard: unknown variant '%s'" t

let rec expr_tree_to_json = function
  | TConst v -> variant "const" [ ("value", flt v) ]
  | TUnit (v, u) ->
    variant "unit" [ ("value", flt v); ("unit", unit_lit_to_json u) ]
  | TIdent n -> variant "ident" [ ("name", name_ref_to_json n) ]
  | TIndex (n, ix) ->
    variant "index"
      [ ("name", name_ref_to_json n); ("indices", list index_item_to_json ix) ]
  | TBinOp (op, l, r) ->
    variant "bin_op"
      [ ("op", bin_op_to_json op); ("lhs", expr_tree_to_json l);
        ("rhs", expr_tree_to_json r) ]
  | TUnOp (op, a) ->
    variant "un_op" [ ("op", un_op_to_json op); ("arg", expr_tree_to_json a) ]
  | TSum s ->
    variant "sum"
      [ ("var", str s.ts_var); ("dim", str s.ts_dim);
        ("guard", opt guard_to_json s.ts_guard);
        ("body", expr_tree_to_json s.ts_body) ]
  | TCond (p, t, e) ->
    variant "cond"
      [ ("pred", expr_tree_to_json p); ("then", expr_tree_to_json t);
        ("else", expr_tree_to_json e) ]
  | TCall (f, args) ->
    variant "call" [ ("function", str f); ("args", list arg_to_json args) ]
  | TList items -> variant "list" [ ("items", list expr_tree_to_json items) ]
  | TRange (lo, hi) ->
    variant "range"
      [ ("lo", expr_tree_to_json lo); ("hi", expr_tree_to_json hi) ]
  | TObsAccess s -> variant "obs_access" [ ("stream", str s) ]
  | TStr s -> variant "str" [ ("value", str s) ]
  | TRunMember m ->
    variant "run_member"
      [ ("run", str m.tr_run);
        ("namespace", run_namespace_to_json m.tr_namespace);
        ("member", str m.tr_member) ]

and index_item_to_json = function
  | IPositional v -> variant "positional" [ ("value", expr_tree_to_json v) ]
  | INamed (d, v) ->
    variant "named" [ ("dim", str d); ("value", expr_tree_to_json v) ]

and arg_to_json a =
  obj [ ("name", str a.a_name); ("value", expr_tree_to_json a.a_value) ]

let rec expr_tree_of_json j =
  match tagged "expression" j with
  | "const", b -> TConst (as_float (record "const" [ "value" ] b "value"))
  | "unit", b ->
    let g = record "unit" [ "value"; "unit" ] b in
    TUnit (as_float (g "value"), unit_lit_of_json (g "unit"))
  | "ident", b -> TIdent (name_ref_of_json (record "ident" [ "name" ] b "name"))
  | "index", b ->
    let g = record "index" [ "name"; "indices" ] b in
    TIndex (name_ref_of_json (g "name"), of_list index_item_of_json (g "indices"))
  | "bin_op", b ->
    let g = record "bin_op" [ "op"; "lhs"; "rhs" ] b in
    TBinOp
      (bin_op_of_json (g "op"), expr_tree_of_json (g "lhs"),
       expr_tree_of_json (g "rhs"))
  | "un_op", b ->
    let g = record "un_op" [ "op"; "arg" ] b in
    TUnOp (un_op_of_json (g "op"), expr_tree_of_json (g "arg"))
  | "sum", b ->
    let g = record "sum" [ "var"; "dim"; "guard"; "body" ] b in
    TSum
      { ts_var = as_str (g "var"); ts_dim = as_str (g "dim");
        ts_guard = of_opt guard_of_json (g "guard");
        ts_body = expr_tree_of_json (g "body") }
  | "cond", b ->
    let g = record "cond" [ "pred"; "then"; "else" ] b in
    TCond
      (expr_tree_of_json (g "pred"), expr_tree_of_json (g "then"),
       expr_tree_of_json (g "else"))
  | "call", b ->
    let g = record "call" [ "function"; "args" ] b in
    TCall (as_str (g "function"), of_list arg_of_json (g "args"))
  | "list", b ->
    TList (of_list expr_tree_of_json (record "list" [ "items" ] b "items"))
  | "range", b ->
    let g = record "range" [ "lo"; "hi" ] b in
    TRange (expr_tree_of_json (g "lo"), expr_tree_of_json (g "hi"))
  | "obs_access", b ->
    TObsAccess (as_str (record "obs_access" [ "stream" ] b "stream"))
  | "str", b -> TStr (as_str (record "str" [ "value" ] b "value"))
  | "run_member", b ->
    let g = record "run_member" [ "run"; "namespace"; "member" ] b in
    TRunMember
      { tr_run = as_str (g "run");
        tr_namespace = run_namespace_of_json (g "namespace");
        tr_member = as_str (g "member") }
  | t, _ -> fail "expression: unknown variant '%s'" t

and index_item_of_json j =
  match tagged "index item" j with
  | "positional", b ->
    IPositional (expr_tree_of_json (record "positional" [ "value" ] b "value"))
  | "named", b ->
    let g = record "named" [ "dim"; "value" ] b in
    INamed (as_str (g "dim"), expr_tree_of_json (g "value"))
  | t, _ -> fail "index item: unknown variant '%s'" t

and arg_of_json j =
  let g = record "argument" [ "name"; "value" ] j in
  { a_name = as_str (g "name"); a_value = expr_tree_of_json (g "value") }

let index_binding_to_json = function
  | BIn (v, d) -> variant "in" [ ("var", str v); ("dim", str d) ]
  | BConsecutive (v, vn, d) ->
    variant "consecutive"
      [ ("var", str v); ("var_next", str vn); ("dim", str d) ]
  | BCompartments v -> variant "compartments" [ ("var", str v) ]

let index_binding_of_json j =
  match tagged "index binding" j with
  | "in", b ->
    let g = record "in" [ "var"; "dim" ] b in
    BIn (as_str (g "var"), as_str (g "dim"))
  | "consecutive", b ->
    let g = record "consecutive" [ "var"; "var_next"; "dim" ] b in
    BConsecutive (as_str (g "var"), as_str (g "var_next"), as_str (g "dim"))
  | "compartments", b ->
    BCompartments (as_str (record "compartments" [ "var" ] b "var"))
  | t, _ -> fail "index binding: unknown variant '%s'" t

let declared_expr_to_json e =
  obj
    [ ("tree", expr_tree_to_json e.de_tree); ("katex", str e.de_katex);
      ("katex_loss", list katex_loss_to_json e.de_katex_loss);
      ("reads", list name_ref_to_json e.de_reads) ]

let declared_expr_of_json j =
  let g = record "expression" [ "tree"; "katex"; "katex_loss"; "reads" ] j in
  { de_tree = expr_tree_of_json (g "tree"); de_katex = as_str (g "katex");
    de_katex_loss = of_list katex_loss_of_json (g "katex_loss");
    de_reads = of_list name_ref_of_json (g "reads") }

(* Short names for the two shapes used on almost every declaration. *)
let dx = declared_expr_to_json
let xd = declared_expr_of_json
let bindings = list index_binding_to_json
let of_bindings = of_list index_binding_of_json
let items = list index_item_to_json
let of_items = of_list index_item_of_json
let args = list arg_to_json
let of_args = of_list arg_of_json

(* ── Declarations ────────────────────────────────────────────────────────── *)

let compile_inputs_to_json c =
  obj
    [ ("set_values",
       list
         (fun s -> obj [ ("name", str s.sv_name); ("value", flt s.sv_value) ])
         c.ci_set_values);
      ("files",
       list
         (fun f ->
           obj [ ("path", str f.fi_path); ("sha256", opt str f.fi_sha256) ])
         c.ci_files) ]

let compile_inputs_of_json j =
  let g = record "compile_inputs" [ "set_values"; "files" ] j in
  { ci_set_values =
      of_list
        (fun j ->
          let g = record "set value" [ "name"; "value" ] j in
          { sv_name = as_str (g "name"); sv_value = as_float (g "value") })
        (g "set_values");
    ci_files =
      of_list
        (fun j ->
          let g = record "file" [ "path"; "sha256" ] j in
          { fi_path = as_str (g "path");
            fi_sha256 = of_opt as_str (g "sha256") })
        (g "files") }

let dim_source_to_json = function
  | DsInline -> str "inline"
  | DsRead r ->
    variant "read" [ ("path", str r.ds_path); ("column", str r.ds_column) ]

let dim_source_of_json j =
  match tagged "dimension source" j with
  | "inline", `Null -> DsInline
  | "read", b ->
    let g = record "read" [ "path"; "column" ] b in
    DsRead { ds_path = as_str (g "path"); ds_column = as_str (g "column") }
  | t, _ -> fail "dimension source: unknown or malformed variant '%s'" t

let dimension_to_json d =
  obj
    [ ("name", str d.dd_name); ("source", dim_source_to_json d.dd_source);
      ("levels", strs d.dd_levels);
      ("source_text", source_text_to_json d.dd_text) ]

let dimension_of_json j =
  let g = record "dimension" [ "name"; "source"; "levels"; "source_text" ] j in
  { dd_name = as_str (g "name"); dd_source = dim_source_of_json (g "source");
    dd_levels = of_strs (g "levels");
    dd_text = source_text_of_json (g "source_text") }

let compartment_to_json c =
  obj
    [ ("name", str c.cd_name); ("kind", compartment_kind_to_json c.cd_kind);
      ("dims", strs c.cd_dims); ("expands_to", opt strs c.cd_expands_to);
      ("source_text", source_text_to_json c.cd_text) ]

let compartment_of_json j =
  let g = record "compartment" [ "name"; "kind"; "dims"; "expands_to"; "source_text" ] j in
  { cd_name = as_str (g "name"); cd_kind = compartment_kind_of_json (g "kind");
    cd_dims = of_strs (g "dims"); cd_expands_to = of_opt of_strs (g "expands_to");
    cd_text = source_text_of_json (g "source_text") }

let stratify_to_json s =
  obj [ ("dim", str s.st_dim); ("only", opt strs s.st_only);
      ("source_text", source_text_to_json s.st_text) ]

let stratify_of_json j =
  let g = record "stratify" [ "dim"; "only"; "source_text" ] j in
  { st_dim = as_str (g "dim"); st_only = of_opt of_strs (g "only");
    st_text = source_text_of_json (g "source_text") }

let dim_annotation_to_json d =
  obj [ ("population", int d.da_population); ("time", int d.da_time) ]

let dim_annotation_of_json j =
  let g = record "dim" [ "population"; "time" ] j in
  { da_population = as_int (g "population"); da_time = as_int (g "time") }

let parameter_to_json p =
  obj
    [ ("name", str p.pd_name); ("dims", strs p.pd_dims);
      ("kind", param_kind_to_json p.pd_kind);
      ("dim", opt dim_annotation_to_json p.pd_dim);
      ("bounds",
       opt (fun b -> obj [ ("lo", dx b.b_lo); ("hi", dx b.b_hi) ]) p.pd_bounds);
      ("prior",
       opt
         (fun r ->
           obj
             [ ("family", str r.pr_family); ("args", args r.pr_args);
               ("pool_over", opt str r.pr_pool_over) ])
         p.pd_prior);
      ("source_text", source_text_to_json p.pd_text) ]

let parameter_of_json j =
  let g =
    record "parameter" [ "name"; "dims"; "kind"; "dim"; "bounds"; "prior"; "source_text" ] j
  in
  { pd_name = as_str (g "name"); pd_dims = of_strs (g "dims");
    pd_kind = param_kind_of_json (g "kind");
    pd_dim = of_opt dim_annotation_of_json (g "dim");
    pd_bounds =
      of_opt
        (fun j ->
          let g = record "bounds" [ "lo"; "hi" ] j in
          { b_lo = xd (g "lo"); b_hi = xd (g "hi") })
        (g "bounds");
    pd_prior =
      of_opt
        (fun j ->
          let g = record "prior" [ "family"; "args"; "pool_over" ] j in
          { pr_family = as_str (g "family"); pr_args = of_args (g "args");
            pr_pool_over = of_opt as_str (g "pool_over") })
        (g "prior");
    pd_text = source_text_of_json (g "source_text") }

let definition_to_json d =
  obj
    [ ("name", str d.df_name); ("indices", bindings d.df_indices);
      ("shape", opt strs d.df_shape); ("kind", opt param_kind_to_json d.df_kind);
      ("dim", opt dim_annotation_to_json d.df_dim); ("body", dx d.df_body);
      ("source_text", source_text_to_json d.df_text) ]

let definition_of_json j =
  let g =
    record "definition" [ "name"; "indices"; "shape"; "kind"; "dim"; "body"; "source_text" ] j
  in
  { df_name = as_str (g "name"); df_indices = of_bindings (g "indices");
    df_shape = of_opt of_strs (g "shape");
    df_kind = of_opt param_kind_of_json (g "kind");
    df_dim = of_opt dim_annotation_of_json (g "dim"); df_body = xd (g "body");
    df_text = source_text_of_json (g "source_text") }

let table_to_json t =
  obj
    [ ("names", strs t.tb_names);
      ("dims",
       list
         (fun d ->
           obj [ ("dim", str d.td_dim); ("unit", opt unit_lit_to_json d.td_unit) ])
         t.tb_dims);
      ("cell_kind", opt param_kind_to_json t.tb_cell_kind);
      ("value", dx t.tb_value);
      ("source_text", source_text_to_json t.tb_text) ]

let table_of_json j =
  let g = record "table" [ "names"; "dims"; "cell_kind"; "value"; "source_text" ] j in
  { tb_names = of_strs (g "names");
    tb_dims =
      of_list
        (fun j ->
          let g = record "table dimension" [ "dim"; "unit" ] j in
          { td_dim = as_str (g "dim");
            td_unit = of_opt unit_lit_of_json (g "unit") })
        (g "dims");
    tb_cell_kind = of_opt param_kind_of_json (g "cell_kind");
    tb_value = xd (g "value");
    tb_text = source_text_of_json (g "source_text") }

let time_input_value_to_json = function
  | TvStr s  -> variant "str" [ ("value", str s) ]
  | TvExpr e -> variant "expr" [ ("value", dx e) ]

let time_input_value_of_json j =
  match tagged "time input value" j with
  | "str", b  -> TvStr (as_str (record "str" [ "value" ] b "value"))
  | "expr", b -> TvExpr (xd (record "expr" [ "value" ] b "value"))
  | t, _ -> fail "time input value: unknown variant '%s'" t

let time_input_to_json t =
  obj
    [ ("name", str t.ti_name); ("indices", bindings t.ti_indices);
      ("kind", str t.ti_kind); ("unit", unit_lit_to_json t.ti_unit);
      ("args",
       list
         (fun a ->
           obj
             [ ("name", str a.ta_name);
               ("value", time_input_value_to_json a.ta_value) ])
         t.ti_args);
      ("source_text", source_text_to_json t.ti_text) ]

let time_input_of_json j =
  let g = record "time input" [ "name"; "indices"; "kind"; "unit"; "args"; "source_text" ] j in
  { ti_name = as_str (g "name"); ti_indices = of_bindings (g "indices");
    ti_kind = as_str (g "kind"); ti_unit = unit_lit_of_json (g "unit");
    ti_args =
      of_list
        (fun j ->
          let g = record "time input argument" [ "name"; "value" ] j in
          { ta_name = as_str (g "name");
            ta_value = time_input_value_of_json (g "value") })
        (g "args");
    ti_text = source_text_of_json (g "source_text") }

let timepoint_to_json t = obj [ ("name", str t.tp_name); ("time", dx t.tp_time);
      ("source_text", source_text_to_json t.tp_text) ]

let timepoint_of_json j =
  let g = record "timepoint" [ "name"; "time"; "source_text" ] j in
  { tp_name = as_str (g "name"); tp_time = xd (g "time");
    tp_text = source_text_of_json (g "source_text") }

let stoich_ref_to_json s =
  obj [ ("compartment", str s.sr_compartment); ("indices", items s.sr_indices) ]

let stoich_ref_of_json j =
  let g = record "compartment reference" [ "compartment"; "indices" ] j in
  { sr_compartment = as_str (g "compartment");
    sr_indices = of_items (g "indices") }

let destination_to_json = function
  | DstSum cs -> variant "sum" [ ("compartments", list stoich_ref_to_json cs) ]
  | DstBranch arms ->
    variant "branch"
      [ ("arms",
         list
           (fun a ->
             obj [ ("to", stoich_ref_to_json a.ba_to); ("weight", dx a.ba_weight) ])
           arms) ]

let destination_of_json j =
  match tagged "destination" j with
  | "sum", b ->
    DstSum
      (of_list stoich_ref_of_json (record "sum" [ "compartments" ] b "compartments"))
  | "branch", b ->
    DstBranch
      (of_list
         (fun j ->
           let g = record "branch arm" [ "to"; "weight" ] j in
           { ba_to = stoich_ref_of_json (g "to"); ba_weight = xd (g "weight") })
         (record "branch" [ "arms" ] b "arms"))
  | t, _ -> fail "destination: unknown variant '%s'" t

let dynamics_to_json = function
  | DynRate r -> variant "rate" [ ("rate", dx r) ]
  | DynVia (law, a) -> variant "via" [ ("law", str law); ("args", args a) ]

let dynamics_of_json j =
  match tagged "dynamics" j with
  | "rate", b -> DynRate (xd (record "rate" [ "rate" ] b "rate"))
  | "via", b ->
    let g = record "via" [ "law"; "args" ] b in
    DynVia (as_str (g "law"), of_args (g "args"))
  | t, _ -> fail "dynamics: unknown variant '%s'" t

let transition_to_json t =
  obj
    [ ("name", str t.tn_name); ("indices", bindings t.tn_indices);
      ("from", list stoich_ref_to_json t.tn_from);
      ("to", destination_to_json t.tn_to);
      ("dynamics", dynamics_to_json t.tn_dynamics);
      ("guard", opt guard_to_json t.tn_guard); ("lineage", bool t.tn_lineage);
      ("expands_to", opt strs t.tn_expands_to);
      ("source_text", source_text_to_json t.tn_text) ]

let transition_of_json j =
  let g =
    record "transition"
      [ "name"; "indices"; "from"; "to"; "dynamics"; "guard"; "lineage";
        "expands_to"; "source_text" ]
      j
  in
  { tn_name = as_str (g "name"); tn_indices = of_bindings (g "indices");
    tn_from = of_list stoich_ref_of_json (g "from");
    tn_to = destination_of_json (g "to");
    tn_dynamics = dynamics_of_json (g "dynamics");
    tn_guard = of_opt guard_of_json (g "guard");
    tn_lineage = as_bool (g "lineage");
    tn_expands_to = of_opt of_strs (g "expands_to");
    tn_text = source_text_of_json (g "source_text") }

let ode_to_json o =
  obj [ ("compartment", str o.od_compartment); ("derivative", dx o.od_derivative);
      ("source_text", source_text_to_json o.od_text) ]

let ode_of_json j =
  let g = record "ode equation" [ "compartment"; "derivative"; "source_text" ] j in
  { od_compartment = as_str (g "compartment");
    od_derivative = xd (g "derivative");
    od_text = source_text_of_json (g "source_text") }

let balance_to_json b =
  obj [ ("compartment", str b.bl_compartment); ("expr", dx b.bl_expr);
      ("source_text", source_text_to_json b.bl_text) ]

let balance_of_json j =
  let g = record "balance" [ "compartment"; "expr"; "source_text" ] j in
  { bl_compartment = as_str (g "compartment"); bl_expr = xd (g "expr");
    bl_text = source_text_of_json (g "source_text") }

let distribution_to_json d =
  obj [ ("family", family_to_json d.di_family); ("args", args d.di_args) ]

let distribution_of_json j =
  let g = record "distribution" [ "family"; "args" ] j in
  { di_family = family_of_json (g "family"); di_args = of_args (g "args") }

let initial_value_to_json = function
  | IvExpr e -> variant "expr" [ ("value", dx e) ]
  | IvLaw d  -> variant "law" [ ("law", distribution_to_json d) ]

let initial_value_of_json j =
  match tagged "initial value" j with
  | "expr", b -> IvExpr (xd (record "expr" [ "value" ] b "value"))
  | "law", b  -> IvLaw (distribution_of_json (record "law" [ "law" ] b "law"))
  | t, _ -> fail "initial value: unknown variant '%s'" t

let initial_to_json i =
  obj
    [ ("compartment", str i.in_compartment); ("indices", items i.in_indices);
      ("bindings", bindings i.in_bindings);
      ("value", initial_value_to_json i.in_value);
      ("source_text", source_text_to_json i.in_text) ]

let initial_of_json j =
  let g =
    record "initial condition" [ "compartment"; "indices"; "bindings"; "value"; "source_text" ] j
  in
  { in_compartment = as_str (g "compartment");
    in_indices = of_items (g "indices");
    in_bindings = of_bindings (g "bindings");
    in_value = initial_value_of_json (g "value");
    in_text = source_text_of_json (g "source_text") }

let column_role_to_json = function
  | CrTime        -> str "time"
  | CrDim d       -> variant "dim" [ ("dim", str d) ]
  | CrValue k     -> variant "value" [ ("kind", param_kind_to_json k) ]
  | CrWindowStart -> str "window_start"
  | CrWindowStop  -> str "window_stop"

let column_role_of_json j =
  match tagged "column role" j with
  | "time", `Null         -> CrTime
  | "window_start", `Null -> CrWindowStart
  | "window_stop", `Null  -> CrWindowStop
  | "dim", b   -> CrDim (as_str (record "dim" [ "dim" ] b "dim"))
  | "value", b -> CrValue (param_kind_of_json (record "value" [ "kind" ] b "kind"))
  | t, _ -> fail "column role: unknown or malformed variant '%s'" t

let projection_to_json = function
  | PjIncidence (t, ix) ->
    variant "incidence" [ ("transition", str t); ("indices", items ix) ]
  | PjDerived e -> variant "derived" [ ("value", dx e) ]

let projection_of_json j =
  match tagged "projection" j with
  | "incidence", b ->
    let g = record "incidence" [ "transition"; "indices" ] b in
    PjIncidence (as_str (g "transition"), of_items (g "indices"))
  | "derived", b -> PjDerived (xd (record "derived" [ "value" ] b "value"))
  | t, _ -> fail "projection: unknown variant '%s'" t

let schedule_core_to_json = function
  | ScEvery e -> variant "every" [ ("every", dx e) ]
  | ScAt ts   -> variant "at" [ ("times", list dx ts) ]

let schedule_core_of_json j =
  match tagged "schedule" j with
  | "every", b -> ScEvery (xd (record "every" [ "every" ] b "every"))
  | "at", b    -> ScAt (of_list xd (record "at" [ "times" ] b "times"))
  | t, _ -> fail "schedule: unknown variant '%s'" t

let observation_to_json o =
  obj
    [ ("name", str o.ob_name); ("indices", bindings o.ob_indices);
      ("source", opt str o.ob_source);
      ("columns",
       list
         (fun c ->
           obj [ ("name", str c.oc_name); ("role", column_role_to_json c.oc_role) ])
         o.ob_columns);
      ("measurement",
       obj
         [ ("scored", str o.ob_measurement.ms_scored);
           ("likelihood", distribution_to_json o.ob_measurement.ms_likelihood) ]);
      ("projection", projection_to_json o.ob_projection);
      ("emit_schedule", opt schedule_core_to_json o.ob_emit_schedule);
      ("covers",
       opt
         (fun c ->
           obj
             [ ("form", str c.cv_form); ("column", str c.cv_column);
               ("span", opt dx c.cv_span) ])
         o.ob_covers);
      ("expands_to", opt strs o.ob_expands_to);
      ("source_text", source_text_to_json o.ob_text) ]

let observation_of_json j =
  let g =
    record "observation"
      [ "name"; "indices"; "source"; "columns"; "measurement"; "projection";
        "emit_schedule"; "covers"; "expands_to"; "source_text" ]
      j
  in
  { ob_name = as_str (g "name"); ob_indices = of_bindings (g "indices");
    ob_source = of_opt as_str (g "source");
    ob_columns =
      of_list
        (fun j ->
          let g = record "column" [ "name"; "role" ] j in
          { oc_name = as_str (g "name"); oc_role = column_role_of_json (g "role") })
        (g "columns");
    ob_measurement =
      (let g = record "measurement" [ "scored"; "likelihood" ] (g "measurement") in
       { ms_scored = as_str (g "scored");
         ms_likelihood = distribution_of_json (g "likelihood") });
    ob_projection = projection_of_json (g "projection");
    ob_emit_schedule = of_opt schedule_core_of_json (g "emit_schedule");
    ob_covers =
      of_opt
        (fun j ->
          let g = record "covers" [ "form"; "column"; "span" ] j in
          { cv_form = as_str (g "form"); cv_column = as_str (g "column");
            cv_span = of_opt xd (g "span") })
        (g "covers");
    ob_expands_to = of_opt of_strs (g "expands_to");
    ob_text = source_text_of_json (g "source_text") }

let action_to_json = function
  | AcTransfer a -> variant "transfer" [ ("args", args a) ]
  | AcSet s ->
    variant "set"
      [ ("compartment", str s.as_compartment); ("indices", items s.as_indices);
        ("value", dx s.as_value) ]
  | AcAdd a ->
    variant "add"
      [ ("compartment", str a.aa_compartment); ("indices", items a.aa_indices);
        ("count", dx a.aa_count) ]

let action_of_json j =
  match tagged "action" j with
  | "transfer", b -> AcTransfer (of_args (record "transfer" [ "args" ] b "args"))
  | "set", b ->
    let g = record "set" [ "compartment"; "indices"; "value" ] b in
    AcSet
      { as_compartment = as_str (g "compartment");
        as_indices = of_items (g "indices"); as_value = xd (g "value") }
  | "add", b ->
    let g = record "add" [ "compartment"; "indices"; "count" ] b in
    AcAdd
      { aa_compartment = as_str (g "compartment");
        aa_indices = of_items (g "indices"); aa_count = xd (g "count") }
  | t, _ -> fail "action: unknown variant '%s'" t

let schedule_to_json = function
  | ShAtTimes ts -> variant "at_times" [ ("times", list dx ts) ]
  | ShRecurring r ->
    variant "recurring"
      [ ("every", dx r.sh_every); ("from", opt dx r.sh_from);
        ("to", opt dx r.sh_to) ]
  | ShEveryAtDay e ->
    variant "every_at_day"
      [ ("period", dx e.sh_period); ("at_day", dx e.sh_at_day) ]

let schedule_of_json j =
  match tagged "intervention schedule" j with
  | "at_times", b ->
    ShAtTimes (of_list xd (record "at_times" [ "times" ] b "times"))
  | "recurring", b ->
    let g = record "recurring" [ "every"; "from"; "to" ] b in
    ShRecurring
      { sh_every = xd (g "every"); sh_from = of_opt xd (g "from");
        sh_to = of_opt xd (g "to") }
  | "every_at_day", b ->
    let g = record "every_at_day" [ "period"; "at_day" ] b in
    ShEveryAtDay { sh_period = xd (g "period"); sh_at_day = xd (g "at_day") }
  | t, _ -> fail "intervention schedule: unknown variant '%s'" t

let intervention_to_json i =
  obj
    [ ("name", str i.iv_name); ("indices", bindings i.iv_indices);
      ("actions", list action_to_json i.iv_actions);
      ("schedule", schedule_to_json i.iv_schedule);
      ("guard", opt guard_to_json i.iv_guard);
      ("source_text", source_text_to_json i.iv_text) ]

let intervention_of_json j =
  let g =
    record "intervention" [ "name"; "indices"; "actions"; "schedule"; "guard"; "source_text" ] j
  in
  { iv_name = as_str (g "name"); iv_indices = of_bindings (g "indices");
    iv_actions = of_list action_of_json (g "actions");
    iv_schedule = schedule_of_json (g "schedule");
    iv_guard = of_opt guard_of_json (g "guard");
    iv_text = source_text_of_json (g "source_text") }

let rec trigger_pred_to_json = function
  | TgAtom e -> variant "atom" [ ("value", dx e) ]
  | TgAnd (l, r) ->
    variant "and"
      [ ("lhs", trigger_pred_to_json l); ("rhs", trigger_pred_to_json r) ]
  | TgOr (l, r) ->
    variant "or"
      [ ("lhs", trigger_pred_to_json l); ("rhs", trigger_pred_to_json r) ]
  | TgNot a -> variant "not" [ ("arg", trigger_pred_to_json a) ]

let rec trigger_pred_of_json j =
  let pair what b =
    let g = record what [ "lhs"; "rhs" ] b in
    (trigger_pred_of_json (g "lhs"), trigger_pred_of_json (g "rhs"))
  in
  match tagged "trigger" j with
  | "atom", b -> TgAtom (xd (record "atom" [ "value" ] b "value"))
  | "and", b  -> let l, r = pair "and" b in TgAnd (l, r)
  | "or", b   -> let l, r = pair "or" b in TgOr (l, r)
  | "not", b  -> TgNot (trigger_pred_of_json (record "not" [ "arg" ] b "arg"))
  | t, _ -> fail "trigger: unknown variant '%s'" t

let reactive_to_json r =
  obj
    [ ("name", str r.rc_name); ("indices", bindings r.rc_indices);
      ("when", trigger_pred_to_json r.rc_when); ("after", opt dx r.rc_after);
      ("once", opt dx r.rc_once); ("cooldown", opt dx r.rc_cooldown);
      ("action", action_to_json r.rc_action);
      ("guard", opt guard_to_json r.rc_guard);
      ("source_text", source_text_to_json r.rc_text) ]

let reactive_of_json j =
  let g =
    record "reactive intervention"
      [ "name"; "indices"; "when"; "after"; "once"; "cooldown"; "action"; "guard"; "source_text" ]
      j
  in
  { rc_name = as_str (g "name"); rc_indices = of_bindings (g "indices");
    rc_when = trigger_pred_of_json (g "when"); rc_after = of_opt xd (g "after");
    rc_once = of_opt xd (g "once"); rc_cooldown = of_opt xd (g "cooldown");
    rc_action = action_of_json (g "action");
    rc_guard = of_opt guard_of_json (g "guard");
    rc_text = source_text_of_json (g "source_text") }

let assign_to_json a = obj [ ("name", str a.an_name); ("value", dx a.an_value) ]

let assign_of_json j =
  let g = record "assignment" [ "name"; "value" ] j in
  { an_name = as_str (g "name"); an_value = xd (g "value") }

let scenario_field_to_json = function
  | SfLabel t   -> variant "label" [ ("text", str t) ]
  | SfEnable n  -> variant "enable" [ ("names", strs n) ]
  | SfDisable n -> variant "disable" [ ("names", strs n) ]
  | SfSet v     -> variant "set" [ ("values", list assign_to_json v) ]
  | SfScale v   -> variant "scale" [ ("values", list assign_to_json v) ]
  | SfCompose n -> variant "compose" [ ("names", strs n) ]
  | SfTEnd e    -> variant "t_end" [ ("value", dx e) ]
  | SfExtends p -> variant "extends" [ ("parent", str p) ]

let scenario_field_of_json j =
  let one what key b = record what [ key ] b key in
  match tagged "scenario field" j with
  | "label", b   -> SfLabel (as_str (one "label" "text" b))
  | "enable", b  -> SfEnable (of_strs (one "enable" "names" b))
  | "disable", b -> SfDisable (of_strs (one "disable" "names" b))
  | "set", b     -> SfSet (of_list assign_of_json (one "set" "values" b))
  | "scale", b   -> SfScale (of_list assign_of_json (one "scale" "values" b))
  | "compose", b -> SfCompose (of_strs (one "compose" "names" b))
  | "t_end", b   -> SfTEnd (xd (one "t_end" "value" b))
  | "extends", b -> SfExtends (as_str (one "extends" "parent" b))
  | t, _ -> fail "scenario field: unknown variant '%s'" t

let scenario_to_json s =
  obj
    [ ("name", str s.sn_name);
      ("fields", list scenario_field_to_json s.sn_fields);
      ("source_text", source_text_to_json s.sn_text) ]

let scenario_of_json j =
  let g = record "scenario" [ "name"; "fields"; "source_text" ] j in
  { sn_name = as_str (g "name");
    sn_fields = of_list scenario_field_of_json (g "fields");
    sn_text = source_text_of_json (g "source_text") }

let simulation_to_json s =
  obj
    [ ("from", dx s.sm_from); ("to", dx s.sm_to); ("dt", opt dx s.sm_dt);
      ("integrator", opt str s.sm_integrator); ("atol", opt dx s.sm_atol);
      ("rtol", opt dx s.sm_rtol);
      ("source_text", source_text_to_json s.sm_text) ]

let simulation_of_json j =
  let g =
    record "simulation" [ "from"; "to"; "dt"; "integrator"; "atol"; "rtol"; "source_text" ] j
  in
  { sm_from = xd (g "from"); sm_to = xd (g "to"); sm_dt = of_opt xd (g "dt");
    sm_integrator = of_opt as_str (g "integrator");
    sm_atol = of_opt xd (g "atol"); sm_rtol = of_opt xd (g "rtol");
    sm_text = source_text_of_json (g "source_text") }

let output_to_json o =
  obj
    [ ("trajectories",
       opt
         (fun t ->
           obj
             [ ("schedule", schedule_core_to_json t.to_schedule);
               ("format", str t.to_format) ])
         o.ou_trajectories);
      ("source_text", source_text_to_json o.ou_text) ]

let output_of_json j =
  let g = record "output" [ "trajectories"; "source_text" ] j in
  { ou_trajectories =
      of_opt
        (fun j ->
          let g = record "trajectories" [ "schedule"; "format" ] j in
          { to_schedule = schedule_core_of_json (g "schedule");
            to_format = as_str (g "format") })
        (g "trajectories");
    ou_text = source_text_of_json (g "source_text") }

let quantity_to_json q =
  obj
    [ ("name", str q.qd_name); ("indices", bindings q.qd_indices);
      ("body", dx q.qd_body);
      ("source_text", source_text_to_json q.qd_text) ]

let quantity_of_json j =
  let g = record "quantity" [ "name"; "indices"; "body"; "source_text" ] j in
  { qd_name = as_str (g "name"); qd_indices = of_bindings (g "indices");
    qd_body = xd (g "body");
    qd_text = source_text_of_json (g "source_text") }

let contrast_to_json c = obj [ ("name", str c.ct_name); ("body", dx c.ct_body);
      ("source_text", source_text_to_json c.ct_text) ]

let contrast_of_json j =
  let g = record "contrast" [ "name"; "body"; "source_text" ] j in
  { ct_name = as_str (g "name"); ct_body = xd (g "body");
    ct_text = source_text_of_json (g "source_text") }

(* ── The document ────────────────────────────────────────────────────────── *)

let model_fields =
  [ "schema_version"; "compile_inputs"; "time_unit"; "description"; "origin";
    "dimensions"; "compartments"; "stratify"; "parameters"; "definitions";
    "tables"; "time_inputs"; "timepoints"; "transitions"; "ode_equations";
    "balance"; "initial"; "observations"; "interventions"; "events";
    "reactive_interventions"; "scenarios"; "simulation"; "output";
    "quantities"; "contrasts" ]

(** The section as the JSON the Rust [declared] crate reads. Raises
    [Invalid_argument] on a non-finite number, which JSON cannot hold. *)
let to_json (m : declared_model) : json =
  obj
    [ ("schema_version", int m.dm_schema_version);
      ("compile_inputs", compile_inputs_to_json m.dm_compile_inputs);
      ("time_unit", opt unit_lit_to_json m.dm_time_unit);
      ("description", opt str m.dm_description);
      ("origin", opt str m.dm_origin);
      ("dimensions", list dimension_to_json m.dm_dimensions);
      ("compartments", list compartment_to_json m.dm_compartments);
      ("stratify", list stratify_to_json m.dm_stratify);
      ("parameters", list parameter_to_json m.dm_parameters);
      ("definitions", list definition_to_json m.dm_definitions);
      ("tables", list table_to_json m.dm_tables);
      ("time_inputs", list time_input_to_json m.dm_time_inputs);
      ("timepoints", list timepoint_to_json m.dm_timepoints);
      ("transitions", list transition_to_json m.dm_transitions);
      ("ode_equations", list ode_to_json m.dm_ode_equations);
      ("balance", list balance_to_json m.dm_balance);
      ("initial", list initial_to_json m.dm_initial);
      ("observations", list observation_to_json m.dm_observations);
      ("interventions", list intervention_to_json m.dm_interventions);
      ("events", list intervention_to_json m.dm_events);
      ("reactive_interventions",
       list reactive_to_json m.dm_reactive_interventions);
      ("scenarios", list scenario_to_json m.dm_scenarios);
      ("simulation", opt simulation_to_json m.dm_simulation);
      ("output", opt output_to_json m.dm_output);
      ("quantities", list quantity_to_json m.dm_quantities);
      ("contrasts", list contrast_to_json m.dm_contrasts) ]

(** Reads a section {!to_json} wrote. Raises [Serde.DeserError] on a missing
    or unknown field or tag, a wrong JSON type, and a [schema_version] other
    than {!schema_version}. An unknown top-level field is reported before the
    version is compared. *)
let of_json (j : json) : declared_model =
  let g = record "declared model" model_fields j in
  let version = as_int (g "schema_version") in
  if version <> schema_version then
    fail "declared model: schema_version %d is not the supported version %d"
      version schema_version;
  { dm_schema_version = version;
    dm_compile_inputs = compile_inputs_of_json (g "compile_inputs");
    dm_time_unit = of_opt unit_lit_of_json (g "time_unit");
    dm_description = of_opt as_str (g "description");
    dm_origin = of_opt as_str (g "origin");
    dm_dimensions = of_list dimension_of_json (g "dimensions");
    dm_compartments = of_list compartment_of_json (g "compartments");
    dm_stratify = of_list stratify_of_json (g "stratify");
    dm_parameters = of_list parameter_of_json (g "parameters");
    dm_definitions = of_list definition_of_json (g "definitions");
    dm_tables = of_list table_of_json (g "tables");
    dm_time_inputs = of_list time_input_of_json (g "time_inputs");
    dm_timepoints = of_list timepoint_of_json (g "timepoints");
    dm_transitions = of_list transition_of_json (g "transitions");
    dm_ode_equations = of_list ode_of_json (g "ode_equations");
    dm_balance = of_list balance_of_json (g "balance");
    dm_initial = of_list initial_of_json (g "initial");
    dm_observations = of_list observation_of_json (g "observations");
    dm_interventions = of_list intervention_of_json (g "interventions");
    dm_events = of_list intervention_of_json (g "events");
    dm_reactive_interventions =
      of_list reactive_of_json (g "reactive_interventions");
    dm_scenarios = of_list scenario_of_json (g "scenarios");
    dm_simulation = of_opt simulation_of_json (g "simulation");
    dm_output = of_opt output_of_json (g "output");
    dm_quantities = of_list quantity_of_json (g "quantities");
    dm_contrasts = of_list contrast_of_json (g "contrasts") }
