(** The module the others open: its [expr] shares a name with [User.expr]. *)

type expr = Lit of int | Neg of expr

type t = A | B
