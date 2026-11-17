(** Opens [Other], then declares an [expr] of its own. A name written
    before the declaration means [Other.expr]; after it, [User.expr]. *)

open Other

let first (e : expr) : t = match e with Lit _ -> A | Neg _ -> B

type expr = Wrap of Other.expr | Twice of expr

type pair = { a : (* cut *) expr; b : Other.expr }

let eval (e : expr) : Other.expr =
  match e with Wrap x -> x | Twice _ -> Lit 0

module Inner = struct
  type t = { label : string }
end

module I = Inner

type labelled = I.t * Inner.t * (int, int) Stdlib.Hashtbl.t

let s = "type expr"
