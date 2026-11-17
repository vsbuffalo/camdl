(** The fixture module: exercises every declaration shape. *)

[@@@warning "-32"]

(** A variant with per-constructor docs. *)
type shape =
  | Circle of float  (** A circle with a radius. *)
  | Rect of float * float  (* implementation note, never published *)
  | Pt  (** The point. *)
  | Box of { w : float; mutable h : float }  (** Inline record. *)

(** A record with a mutable field. *)
type loc = {
  file : string;  (** The file. *)
  mutable line : int;   (* 1-indexed *)
  col : int;
}

type 'a pair = 'a * Other.expr list

type opaque

(* a comment inside the declaration *)
type tree =
  (* leading note *)
  | Leaf
  | Node of tree * forest   (* trailing note *)

(** Mutually recursive with [tree]. *)
and forest = tree list

type nonrec t = Other.t = A | B

type ext = ..

type derived = { d : int } [@@deriving show]

type _ gadt = G : int -> int gadt | H : unit gadt

(** Falls back to the interface. *)
type documented_in_mli = { x : int }

module Inner = struct
  (** Inner module doc. *)

  (** An inner type. *)
  type item = { name : string }
end
