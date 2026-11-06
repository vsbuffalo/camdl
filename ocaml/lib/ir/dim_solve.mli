(** Exact integer solver for dimension equations.

    A dimension is a vector of integer exponents, one per base (P, T). Every
    rule of dimension inference is a linear equation over such vectors: a
    product's dimension is the sum of its factors', a sum's is each operand's,
    a static power [a^n] is n times its base's. The system decouples by base,
    so one coefficient matrix serves every base, with one right-hand side per
    base.

    Equations are added in the order given. Each is reduced against the rows
    accepted so far; one that reduces to 0 = 0 is redundant, one that reduces
    to 0 = r with r ≠ 0 is a contradiction, reported and dropped so it does not
    cascade. Otherwise it is pivoted on a coefficient of ±1 (the newest such
    variable, which keeps elimination local), a unimodular step that preserves
    the integer solutions. Rows with no ±1 coefficient (they come only from
    powers and square roots) are set aside and checked together at the end by
    a column Hermite normal form: the system has an integer solution exactly
    when forward substitution in that form divides evenly. That is what finds
    [2·κ + 2·w = T⁻¹], which has rational solutions and no integer one.

    Arithmetic is on native integers with every addition and multiplication
    checked; an overflow raises {!Overflow}, never a wrapped value. *)

type var = int

type stage = Declared | Structure | Obligation
(** Where an equation came from: a declared dimension (a parameter's kind, a
    compartment), an operator rule inside an expression, or the obligation of
    the position the expression sits in. Callers add them in this order, so a
    conflict surfaces at an obligation, where its site's code applies. *)

type origin = {
  site  : int;      (** the site whose expression produced the equation *)
  stage : stage;
  what  : string;   (** "operands of +", "obligation", "declared kind", ... *)
}

type eqn = {
  terms  : (var * int) list;   (** Σ cᵢ·xᵢ, integer coefficients *)
  rhs    : int array;          (** = an exponent vector, one entry per base *)
  origin : origin;
}

type conflict_kind =
  | Contradiction  (** no rational solution: reduced to 0 = r, r ≠ 0 *)
  | Not_integral   (** rational solutions only *)

type conflict = {
  culprit : eqn;                         (** the equation that could not be added *)
  kind    : conflict_kind;
  reduced : (var * int) list * int array;
  (** the culprit after reduction against the accepted rows: [([], r)] for a
      contradiction of the unit system, "0 = r" *)
  context : origin list;
  (** origins of the accepted equations the culprit was reduced with, its own
      first; deduplicated and capped, enough to name what it conflicts with *)
}

type outcome

exception Overflow of string
(** An intermediate coefficient left the native integer range. *)

val solve : nbases:int -> eqn list -> outcome
(** Raises [Invalid_argument] if some [rhs] does not have [nbases] entries, and
    {!Overflow}. *)

val conflicts : outcome -> conflict list
(** In the order the culprits were added. *)

val value : outcome -> var -> int array option
(** [Some v] when every integer solution of the accepted equations gives the
    variable the exponent vector [v]; [None] when it is free. *)

type relation = {
  coeffs : (var * int) list;   (** nonzero, over the variables asked about *)
  equals : int array;
}

val relations : outcome -> var list -> relation list
(** A basis of the integer linear relations Σ aᵢ·xᵢ = c that hold among the
    given variables in every solution. For [κ] and [w] when only [κ·w] is
    pinned: [κ + w = T⁻¹]. Determined variables among the arguments each give
    the trivial relation [x = v]; pass only undetermined ones to learn what
    the model fixes about them jointly. *)
