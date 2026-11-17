(** The source text of every file in one compile, kept so that diagnostics can
    quote the line they point at and the declared-model section can quote a
    declaration as written.

    A compile has one primary file, the model, and at most one more: the
    vocabulary file given with [--quantities]. Each diagnostic location names
    the file it belongs to, so the cache is keyed by file name, and a lookup
    answers from that file or not at all: a diagnostic about line 3 of the
    vocabulary file never shows line 3 of the model.

    {!Compiler.front_end_collect} builds the cache with {!of_string} and
    {!add_unit} before parsing. {!Diagnostics.render} reads single lines with
    {!get_line}; a location whose file is not in the cache is rendered with its
    [file:line:col] header and no excerpt. {!Declared_emit} reads whole spans
    with {!slice}. Both lookups treat an empty file name as the primary file,
    because the parser leaves the [file] of every location empty and the
    expander fills in the model's file name only where it reports. *)

(** The text of the files of one compile, split into lines at ['\n']. Line [n]
    of a file, counting from 1, is element [n - 1] of its array. *)
type t = {
  filename : string;
      (** The primary file's name, as the driver was given it. *)
  lines    : string array;
      (** The primary file's lines. *)
  extra    : (string * string array) list;
      (** Every other file, by name: the [--quantities] vocabulary file when
          there is one. *)
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

(** The source text between two positions of [file], exactly as written: from
    column [col] of line [line] up to, not including, column [end_col] of line
    [end_line]. Lines and columns are 1-indexed and columns count bytes, as in
    [Ast.loc]. Interior line breaks, comments and spacing are kept. [None] when
    the file is not in the cache or the positions do not lie inside it; never a
    shortened or a neighbouring text. *)
let slice cache ~file ~line ~col ~end_line ~end_col : string option =
  let lines =
    if file = "" || String.equal file cache.filename then Some cache.lines
    else List.assoc_opt file cache.extra
  in
  match lines with
  | None -> None
  | Some ls ->
    let n = Array.length ls in
    let within l c = c >= 1 && c <= String.length ls.(l - 1) + 1 in
    if line < 1 || end_line > n || line > end_line then None
    else if not (within line col && within end_line end_col) then None
    else if line = end_line then
      if end_col < col then None
      else Some (String.sub ls.(line - 1) (col - 1) (end_col - col))
    else begin
      let first = ls.(line - 1) in
      let buf = Buffer.create 128 in
      Buffer.add_string buf (String.sub first (col - 1) (String.length first - col + 1));
      for l = line + 1 to end_line - 1 do
        Buffer.add_char buf '\n';
        Buffer.add_string buf ls.(l - 1)
      done;
      Buffer.add_char buf '\n';
      Buffer.add_string buf (String.sub ls.(end_line - 1) 0 (end_col - 1));
      Some (Buffer.contents buf)
    end
