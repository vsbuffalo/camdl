(* Source file caching for error display.

   A compile has one PRIMARY source (the model), and may pull in further
   compilation units — today just the `--quantities` vocabulary file. Each
   diagnostic carries the file its location belongs to, so the cache is keyed by
   filename: rendering a diagnostic about the vocabulary file must show a line
   from THAT file, never the model's line of the same number. A location whose
   file is unknown to the cache renders its header (file:line:col) with no
   source excerpt, rather than an excerpt from the wrong file. *)

type t = {
  filename : string;
  lines    : string array;  (* 0-indexed; line_no 1 → lines.(0) *)
  extra    : (string * string array) list;  (* additional units, by filename *)
}

let split src = String.split_on_char '\n' src |> Array.of_list

let of_string ~filename src = { filename; lines = split src; extra = [] }

(** Register a further compilation unit's text under its own filename. *)
let add_unit t ~filename src =
  { t with extra = (filename, split src) :: t.extra }

(** Get the source text of line [line_no] (1-indexed) in [file]. An empty
    [file] means the primary source (the parser leaves locs unstamped and the
    expander fills the model's filename, so both spellings reach here). *)
let get_line cache ~file line_no =
  let lines =
    if file = "" || String.equal file cache.filename then Some cache.lines
    else List.assoc_opt file cache.extra
  in
  match lines with
  | Some ls when line_no >= 1 && line_no <= Array.length ls ->
    Some ls.(line_no - 1)
  | _ -> None
