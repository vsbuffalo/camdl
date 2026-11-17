(** An exact solver for systems of linear equations over integer dimension
    exponents: it finds which dimensions a set of equations determines, which
    it leaves free, and which equations contradict the rest.

    A dimension is a vector of integer exponents, one per base dimension; the
    dimension checker uses two bases, population (P) and time (T). An
    equation is [c₁·x₁ + … + cₙ·xₙ = r], with integer coefficients [cᵢ],
    unknown dimensions [xᵢ] and a known dimension [r]. Because the
    coefficients are scalars, the system splits by base: one coefficient
    matrix, solved once for each base's column of right-hand sides.
    {!solve} takes the equations in order and returns an {!outcome}, which
    answers three questions: the {!conflicts} (equations that could not be
    added), the {!value} of each variable that every solution agrees on, and
    the {!relations} that tie together variables left free. Every addition
    and multiplication is checked, so an exponent outside the native integer
    range raises {!Overflow} rather than wrapping. *)

(** An unknown dimension, numbered by the caller. The solver prefers to
    eliminate the highest-numbered variable of an equation, so numbering
    variables in the order they are created keeps elimination local. *)
type var = int

(** Which kind of fact an equation records: a declared dimension (a
    parameter's kind, a compartment), an operator rule inside an expression,
    or the dimension a position in the model requires.

    The solver carries the stage back in conflicts and does not read it. Its
    use is the order of addition: a caller that adds every [Declared]
    equation, then every [Structure] one, then every [Obligation], has a
    disagreement reported at an obligation, where the position's own
    diagnostic code applies. *)
type stage = Declared | Structure | Obligation

(** Where an equation came from, carried unchanged into any {!conflict} that
    names it. *)
type origin = {
  site  : int;      (** A number the caller chooses to identify the equation. The {!Dimcheck} caller gives each equation its own number, not the number of a model position. *)
  stage : stage;
  what  : string;   (** A label for the caller, such as a diagnostic code. *)
}

(** One equation [Σ cᵢ·xᵢ = rhs]. A variable may appear in [terms] more than
    once; its coefficients are added. *)
type eqn = {
  terms  : (var * int) list;   (** Each variable with its integer coefficient. *)
  rhs    : int array;          (** The known side: one exponent per base, in the base order of the whole system. *)
  origin : origin;
}

(** Why an equation could not be added to those already accepted. *)
type conflict_kind =
  | Contradiction  (** No solution even in rational exponents: the equation reduces to [0 = r] with [r ≠ 0]. *)
  | Not_integral   (** Rational solutions exist, integer ones do not, as in [2·x = T⁻¹]. *)

(** An equation the solver refused, with the accepted equations it was
    found to disagree with. The refused equation is dropped, so one mistake
    is reported once rather than in every equation that depends on it. *)
type conflict = {
  culprit : eqn;                         (** The equation that could not be added. *)
  kind    : conflict_kind;
  context : origin list;
  (** The origin of the culprit, followed by the origins of the accepted
      equations it was reduced against: what it conflicts with. Duplicates
      are removed and the list stops at twelve entries. *)
}

(** The solution of a system: the accepted equations in reduced form, from
    which {!value} and {!relations} read each variable's general solution. *)
type outcome

exception Overflow of string
(** An intermediate coefficient or exponent left the native integer range.
    The message names the operation. *)

val checked_add : int -> int -> int
val checked_mul : int -> int -> int
(** Integer addition and multiplication that raise {!Overflow} rather than
    wrap; for building equations with the same guarantee the solver keeps. *)

val solve : nbases:int -> eqn list -> outcome
(** Solves the equations, added in list order. When equations disagree, the
    later one is refused, with one exception: an equation with no
    coefficient of ±1 (one from a square root or fractional power) is
    checked only after all the others, so it is the one refused when it
    disagrees with any of them. Raises [Invalid_argument] if some [rhs] does
    not have [nbases] entries, and {!Overflow}. *)

val conflicts : outcome -> conflict list
(** The refused equations, in the order they were found: first those
    refused while adding, then those without a coefficient of ±1. *)

val value : outcome -> var -> int array option
(** [Some v] when every integer solution of the accepted equations gives the
    variable the exponent vector [v]; [None] when it is free. A variable no
    accepted equation mentions is free. *)

(** A linear combination of variables that has the same value in every
    solution, [Σ aᵢ·xᵢ = equals]. *)
type relation = {
  coeffs : (var * int) list;   (** Nonzero, over the variables asked about. *)
  equals : int array;          (** One exponent per base. *)
}

val relations : outcome -> var list -> relation list
(** A basis of the integer linear relations [Σ aᵢ·xᵢ = c] that hold among the
    given variables in every solution. For [κ] and [w] when only [κ·w] is
    pinned: [κ + w = T⁻¹]. Determined variables among the arguments each give
    the trivial relation [x = v]; pass only undetermined ones to learn what
    the model fixes about them jointly. *)
