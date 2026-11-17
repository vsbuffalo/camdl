(** The interface. *)

type shape
(** Interface doc for shape; the implementation's own doc wins. *)

type documented_in_mli = { x : int  (** The x coordinate. *) }
(** Documented only in the interface. *)
