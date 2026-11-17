(** The parsed form of a camdl model, before names are resolved. *)

module Located = struct
  type 'a t = { value : 'a; loc : Loc.t }
end

type stoich_ref = Located.t

type expr =
  | Num of float
  | Call of call
and call = { fn : string; args : expr list }

type destination_form =
  | DstSum    of stoich_ref list
  | DstBranch of (stoich_ref * expr) list

type settings = {
  verbose : bool;
  strict : bool;
  mutable cache : bool;
  backend_kind : string;
  out : Yojson.Safe.t;
}

let arrow = "-->" (* <b> & "quoted" *)
