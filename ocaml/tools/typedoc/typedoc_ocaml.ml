(* Prints a JSON description of every type declaration, with its doc
   comments, in the OCaml source files given on the command line, and the
   markup of each file's text: keywords, literals, comments, and every type
   name that refers to one of the declarations. Usage:

     typedoc_ocaml [--root DIR] --cmt-dir DIR [--cmt-dir DIR ...] FILE.ml ...

   Paths in the output are relative to --root, by default the enclosing git
   checkout (the nearest ancestor of the working directory with a `.git`).

   Implementation comments and doc comments are cut out of each
   declaration's source text; doc comments are reported separately in `doc`
   fields. Doc comments missing in `foo.ml` fall back to `foo.mli`.

   Which declaration a type name refers to is read from the compiler's typed
   tree, in the `foo.cmt` that compiling `foo.ml` wrote into one of the
   --cmt-dir directories, never guessed from the name: with `open Ast` in
   scope, `expr` means `Ast.expr`, and in `ir.ml` it means `Ir.expr`. A file
   with no .cmt, or whose .cmt was compiled from other text, is an error.

   The JSON contract is documented in rust/crates/typedoc/src/model.rs. *)

open Parsetree

(* ---- paths ---- *)

let normalize path =
  let abs = String.length path > 0 && path.[0] = '/' in
  let segs = String.split_on_char '/' path in
  let out =
    List.fold_left
      (fun acc s ->
        match s with
        | "" | "." -> acc
        | ".." -> ( match acc with _ :: t -> t | [] -> [])
        | s -> s :: acc)
      [] segs
  in
  (if abs then "/" else "") ^ String.concat "/" (List.rev out)

let absolute p =
  normalize
    (if Filename.is_relative p then Filename.concat (Sys.getcwd ()) p else p)

let rec find_git_root dir =
  if Sys.file_exists (Filename.concat dir ".git") then Some dir
  else
    let parent = Filename.dirname dir in
    if parent = dir then None else find_git_root parent

let relativize root p =
  let a = absolute p in
  match root with
  | Some r ->
      let prefix = r ^ "/" in
      let n = String.length prefix in
      if String.length a > n && String.sub a 0 n = prefix then
        String.sub a n (String.length a - n)
      else normalize p
  | None -> normalize p

(* ---- docstrings ---- *)

let read_file f = In_channel.with_open_bin f In_channel.input_all
let is_blank s = String.trim s = ""

let indent_of s =
  let n = String.length s in
  let i = ref 0 in
  while !i < n && (s.[!i] = ' ' || s.[!i] = '\t') do
    incr i
  done;
  !i

let drop n s =
  if String.length s >= n then String.sub s n (String.length s - n) else ""

(* Remove the common indentation of continuation lines, then trim. *)
let normalize_doc s =
  match String.split_on_char '\n' s with
  | [] -> ""
  | first :: rest ->
      let m =
        List.fold_left
          (fun m l -> if is_blank l then m else min m (indent_of l))
          max_int rest
      in
      let m = if m = max_int then 0 else m in
      let rest = List.map (fun l -> if is_blank l then "" else drop m l) rest in
      String.trim (String.concat "\n" (first :: rest))

let string_payload = function
  | PStr
      [ { pstr_desc =
            Pstr_eval
              ({ pexp_desc = Pexp_constant (Pconst_string (s, _, _)); _ }, _);
          _ } ] ->
      Some (normalize_doc s)
  | _ -> None

let doc_of_attr a =
  if a.attr_name.txt = "ocaml.doc" then string_payload a.attr_payload else None

let doc_of_attrs attrs =
  match List.filter_map doc_of_attr attrs with
  | [] -> None
  | ds -> Some (String.concat "\n\n" ds)

(* ---- types as text ---- *)

let print_type (t : core_type) =
  let buf = Buffer.create 64 in
  let ppf = Format.formatter_of_buffer buf in
  Format.pp_set_margin ppf 100000;
  Format.pp_set_max_indent ppf 99999;
  Pprintast.core_type ppf t;
  Format.pp_print_flush ppf ();
  Buffer.contents buf

let print_fields (lds : label_declaration list) =
  let f ld =
    Printf.sprintf "%s%s : %s"
      (match ld.pld_mutable with
       | Asttypes.Mutable -> "mutable "
       | Immutable -> "")
      ld.pld_name.txt (print_type ld.pld_type)
  in
  "{ " ^ String.concat "; " (List.map f lds) ^ " }"

let print_args = function
  | Pcstr_tuple [] -> ""
  | Pcstr_tuple [ t ] -> print_type t
  | Pcstr_tuple ts ->
      let elt t =
        match t.ptyp_desc with
        | Ptyp_arrow _ | Ptyp_tuple _ -> "(" ^ print_type t ^ ")"
        | _ -> print_type t
      in
      String.concat " * " (List.map elt ts)
  | Pcstr_record lds -> print_fields lds

(* ---- references ---- *)

let builtins =
  [ "int"; "float"; "string"; "bool"; "unit"; "char"; "bytes"; "list";
    "option"; "array"; "ref"; "exn"; "int32"; "int64"; "nativeint"; "lazy_t";
    "format"; "format4"; "format6"; "result"; "seq" ]

let refs_of_types (ts : core_type list) =
  let acc = ref [] in
  let add s =
    if (not (List.mem s builtins)) && not (List.mem s !acc) then
      acc := s :: !acc
  in
  let it =
    { Ast_iterator.default_iterator with
      typ =
        (fun self t ->
          Ast_iterator.default_iterator.typ self t;
          match t.ptyp_desc with
          | Ptyp_constr (lid, _) ->
              add (String.concat "." (Longident.flatten lid.txt))
          | _ -> ()) }
  in
  List.iter (it.typ it) ts;
  List.rev !acc

(* ---- comment cutting ---- *)

type cuts = (int * int) list (* byte offsets [lo, hi) *)

let cnum (p : Lexing.position) = p.pos_cnum

(* A declaration's source text with its comments cut out, blank lines
   dropped and the common indentation removed, and where each byte of it
   came from: [origin] maps the source offset of every byte that survives to
   its line (from 1) and byte column (from 0) in [text]. *)
type stripped = { text : string; origin : (int, int * int) Hashtbl.t }

let strip_comments src (cuts : cuts) lo hi col : stripped =
  (* The kept bytes, each with its source offset; -1 for the padding that
     stands in for the text before the declaration on its first line, so
     the first line's indentation compares with the others'. *)
  let kept = ref [] in
  for _ = 1 to col do
    kept := (' ', -1) :: !kept
  done;
  let keep a z =
    for k = a to z - 1 do
      kept := (src.[k], k) :: !kept
    done
  in
  let pos = ref lo in
  List.iter
    (fun (a, z) ->
      if z > !pos && a < hi then begin
        if a > !pos then keep !pos a;
        pos := max !pos (min z hi)
      end)
    (List.sort compare cuts);
  if !pos < hi then keep !pos hi;
  let lines =
    List.fold_left
      (fun (cur, acc) c ->
        match c with
        | '\n', _ -> ([], List.rev cur :: acc)
        | c -> (c :: cur, acc))
      ([], []) (List.rev !kept)
    |> fun (cur, acc) -> List.rev (List.rev cur :: acc)
  in
  let is_space c = c = ' ' || c = '\t' || c = '\r' in
  let rstrip l =
    let rec go = function (c, _) :: t when is_space c -> go t | l -> l in
    List.rev (go (List.rev l))
  in
  let lines = List.map rstrip lines |> List.filter (fun l -> l <> []) in
  let rec indent n = function
    | (c, _) :: t when c = ' ' || c = '\t' -> indent (n + 1) t
    | _ -> n
  in
  let m = List.fold_left (fun m l -> min m (indent 0 l)) max_int lines in
  let m = if m = max_int then 0 else m in
  let rec drop_n n l =
    match l with _ :: t when n > 0 -> drop_n (n - 1) t | l -> l
  in
  let lines = List.map (drop_n m) lines in
  let origin = Hashtbl.create 256 in
  List.iteri
    (fun k l ->
      List.iteri
        (fun j (_, o) -> if o >= 0 then Hashtbl.replace origin o (k + 1, j))
        l)
    lines;
  let text_of l = String.of_seq (List.to_seq (List.map fst l)) in
  { text = String.concat "\n" (List.map text_of lines); origin }

(* ---- errors ---- *)

let fail fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("typedoc_ocaml: " ^ s);
      exit 1)
    fmt

(* ---- spans: the parts of a source file that get markup ---- *)

(* The documented type a span links to: the module's full path and the
   type's name, both as the JSON names them. *)
type target = { t_module : string; t_type : string }

type span_class =
  | Keyword
  | Comment
  | Doc
  | String_lit
  | Char_lit
  | Number
  | Builtin_type  (* a type the language predefines: int, list, option *)
  | Type  (* a type constructor that resolves to a documented type *)

let class_name = function
  | Keyword -> "keyword"
  | Comment -> "comment"
  | Doc -> "doc"
  | String_lit -> "string"
  | Char_lit -> "char"
  | Number -> "number"
  | Builtin_type -> "builtin_type"
  | Type -> "type"

(* A span in byte offsets [lo, hi) of the file; may cross lines. *)
type span = { lo : int; hi : int; cls : span_class; target : target option }

(* One line's part of a span: the form the JSON carries. *)
type piece = {
  off : int;  (* byte offset in the file *)
  line : int;  (* from 1 *)
  col : int;  (* byte column, from 0 *)
  len : int;  (* bytes *)
  p_cls : span_class;
  p_target : target option;
}

let keywords =
  [ "and"; "as"; "assert"; "asr"; "begin"; "class"; "constraint"; "do";
    "done"; "downto"; "else"; "end"; "exception"; "external"; "false"; "for";
    "fun"; "function"; "functor"; "if"; "in"; "include"; "inherit";
    "initializer"; "land"; "lazy"; "let"; "lor"; "lsl"; "lsr"; "lxor";
    "match"; "method"; "mod"; "module"; "mutable"; "new"; "nonrec"; "object";
    "of"; "open"; "or"; "private"; "rec"; "sig"; "struct"; "then"; "to";
    "true"; "try"; "type"; "val"; "virtual"; "when"; "while"; "with" ]

(* Keywords, literals and comments, from OCaml's own lexer. *)
let lex_spans path src : span list =
  let lexbuf = Lexing.from_string src in
  Location.init lexbuf path;
  Lexer.init ();
  let acc = ref [] in
  let add lo hi cls =
    if hi > lo then acc := { lo; hi; cls; target = None } :: !acc
  in
  let of_loc (l : Location.t) cls = add (cnum l.loc_start) (cnum l.loc_end) cls in
  let rec go () =
    match Lexer.token_with_comments lexbuf with
    | Parser.EOF -> ()
    | tok ->
        let lo = cnum lexbuf.lex_start_p and hi = cnum lexbuf.lex_curr_p in
        (match tok with
         | Parser.COMMENT (_, loc) -> of_loc loc Comment
         | DOCSTRING d -> of_loc (Docstrings.docstring_loc d) Doc
         | STRING _ | QUOTED_STRING_EXPR _ | QUOTED_STRING_ITEM _ ->
             add lo hi String_lit
         | CHAR _ -> add lo hi Char_lit
         | INT _ | FLOAT _ -> add lo hi Number
         | LIDENT _ | UIDENT _ | EOL -> ()
         | _ ->
             if List.mem (String.sub src lo (hi - lo)) keywords then
               add lo hi Keyword);
        go ()
  in
  (try go ()
   with Lexer.Error (_, loc) ->
     fail "%s:%d: the lexer refused the file" path loc.loc_start.pos_lnum);
  List.rev !acc

(* ---- the typed tree: which declaration each type name refers to ---- *)

module SSet = Set.Make (String)

(* One file's compiled form, read from the .cmt dune writes beside the
   bytecode. Idents are only meaningful within the file they come from, so
   these tables are per file. *)
type typed = {
  ty_unit : string;  (* the file's module: Ast *)
  ty_fname : string;  (* the file name the compiler recorded in locations *)
  ty_str : Typedtree.structure;
  type_ids : (string * string) Ident.Tbl.t;
      (* a documented declaration's ident -> module path, type name *)
  mod_ids : string Ident.Tbl.t;  (* a documented module's ident -> path *)
  aliases : Path.t Ident.Tbl.t;  (* module L = Located *)
  mutable decls : (string * string * int) list;
      (* module path, type name, line: checked against the parse *)
}

(* Shared by every file: the module paths of the inputs, and for each
   (module path, type name) the ident of the declaration a reader of that
   module sees under that name -- [None] when that name is bound last by
   something other than a documented declaration, such as an [include]. *)
type world = {
  units : SSet.t;
  mutable modules : SSet.t;
  final : (string * string, (string * Ident.t) option) Hashtbl.t;
      (* the ident, with the unit it belongs to *)
}

let rec tstruct_of_mod (me : Typedtree.module_expr) =
  match me.mod_desc with
  | Tmod_structure s -> Some s
  | Tmod_constraint (m, _, _, _) -> tstruct_of_mod m
  | _ -> None

(* Records the declarations and modules of a structure, mirroring [walk]:
   a nested module is documented when it is a structure, possibly under a
   signature constraint. *)
let rec register world t modname (str : Typedtree.structure) =
  world.modules <- SSet.add modname world.modules;
  List.iter
    (fun (item : Typedtree.structure_item) ->
      match item.str_desc with
      | Tstr_type (_, tds) ->
          List.iter
            (fun (td : Typedtree.type_declaration) ->
              let name = td.typ_name.txt in
              Ident.Tbl.replace t.type_ids td.typ_id (modname, name);
              Hashtbl.replace world.final (modname, name)
                (Some (t.ty_unit, td.typ_id));
              t.decls <-
                (modname, name, td.typ_loc.loc_start.pos_lnum) :: t.decls)
            tds
      | Tstr_include incl ->
          List.iter
            (function
              | Types.Sig_type (id, _, _, _) ->
                  Hashtbl.replace world.final (modname, Ident.name id) None
              | _ -> ())
            incl.incl_type
      | Tstr_module { mb_id = Some id; mb_name = { txt = Some n; _ }; mb_expr; _ }
        -> (
          match tstruct_of_mod mb_expr with
          | Some s ->
              let full = modname ^ "." ^ n in
              Ident.Tbl.replace t.mod_ids id full;
              register world t full s
          | None -> (
              match mb_expr.mod_desc with
              | Tmod_ident (p, _) -> Ident.Tbl.replace t.aliases id p
              | _ -> ()))
      | _ -> ())
    str.str_items

(* The documented module a module path names, if any. *)
let rec module_of_path world t (p : Path.t) =
  match p with
  | Pident id when Ident.persistent id ->
      let n = Ident.name id in
      if SSet.mem n world.units then Some n else None
  | Pident id -> (
      match Ident.Tbl.find_opt t.mod_ids id with
      | Some n -> Some n
      | None ->
          Option.bind (Ident.Tbl.find_opt t.aliases id) (module_of_path world t))
  | Pdot (q, s) ->
      Option.bind (module_of_path world t q) (fun m ->
          let n = m ^ "." ^ s in
          if SSet.mem n world.modules then Some n else None)
  | Papply _ | Pextra_ty _ -> None

(* The documented declaration a type path names, if any. A local name
   counts only when it is the declaration its module finally binds that name
   to, since a target names a type by module and name alone. *)
let target_of_path world t (p : Path.t) =
  let final m n = Hashtbl.find_opt world.final (m, n) |> Option.join in
  match p with
  | Pident id -> (
      match Ident.Tbl.find_opt t.type_ids id with
      | Some (m, n) -> (
          match final m n with
          | Some (u, last) when u = t.ty_unit && Ident.same last id ->
              Some { t_module = m; t_type = n }
          | _ -> None)
      | None -> None)
  | Pdot (q, s) -> (
      match module_of_path world t q with
      | Some m when final m s <> None -> Some { t_module = m; t_type = s }
      | _ -> None)
  | Papply _ | Pextra_ty _ -> None

let squeeze s =
  String.concat "" (String.split_on_char ' ' s)
  |> String.split_on_char '\n' |> String.concat ""
  |> String.split_on_char '\t' |> String.concat ""

(* Every type constructor written in the file that is predefined or names a
   documented declaration, and the name of each documented declaration
   where it is defined. A location must lie in the file itself, not be
   compiler-generated, and cover exactly the name the tree records, or the
   file is refused: a link at the wrong place is worse than none. *)
let typed_spans world t path src : span list =
  let acc = ref [] in
  let record (loc : Location.t) expected cls target =
    if (not loc.loc_ghost) && loc.loc_start.pos_fname = t.ty_fname then begin
      let lo = cnum loc.loc_start and hi = cnum loc.loc_end in
      if lo < 0 || hi > String.length src || hi <= lo then
        fail "%s:%d: a type's location lies outside the file" path
          loc.loc_start.pos_lnum;
      let text = String.sub src lo (hi - lo) in
      if squeeze text <> expected then
        fail "%s:%d: the typed tree puts %S where the source has %S" path
          loc.loc_start.pos_lnum expected text;
      acc := { lo; hi; cls; target } :: !acc
    end
  in
  let flatten lid =
    match Longident.flatten lid with
    | parts -> Some (String.concat "." parts)
    | exception _ -> None
  in
  let it =
    { Tast_iterator.default_iterator with
      typ =
        (fun self ct ->
          (match ct.ctyp_desc with
           | Ttyp_constr (p, lid, _) -> (
               match flatten lid.txt with
               | None -> ()
               | Some expected -> (
                   match p with
                   | Pident id when Ident.is_predef id ->
                       record lid.loc expected Builtin_type None
                   | _ -> (
                       match target_of_path world t p with
                       | Some tg -> record lid.loc expected Type (Some tg)
                       | None -> ())))
           | _ -> ());
          Tast_iterator.default_iterator.typ self ct);
      type_declaration =
        (fun self td ->
          (match target_of_path world t (Pident td.typ_id) with
           | Some tg -> record td.typ_name.loc td.typ_name.txt Type (Some tg)
           | None -> ());
          Tast_iterator.default_iterator.type_declaration self td) }
  in
  it.structure it t.ty_str;
  (* The typer can visit one written type twice (a [let x : t = e] is
     checked as a pattern constraint and an expression constraint). Keep one
     of each; two different readings of one place get no link. *)
  let sorted = List.sort_uniq compare !acc in
  let rec dedupe = function
    | a :: b :: rest when a.lo = b.lo && a.hi = b.hi ->
        let rest =
          List.filter (fun s -> not (s.lo = a.lo && s.hi = a.hi)) rest
        in
        Printf.eprintf
          "typedoc_ocaml: %s: two readings of the type at byte %d; not linked\n"
          path a.lo;
        dedupe rest
    | a :: rest -> a :: dedupe rest
    | [] -> []
  in
  dedupe sorted

(* All spans of a file, sorted. Spans never overlap: a type name is no
   keyword, literal or comment. *)
let file_spans world t path src =
  let spans =
    List.sort
      (fun a b -> compare (a.lo, a.hi) (b.lo, b.hi))
      (lex_spans path src @ typed_spans world t path src)
  in
  let rec check = function
    | a :: (b :: _ as rest) ->
        if b.lo < a.hi then
          fail "%s: spans overlap at bytes %d-%d and %d-%d" path a.lo a.hi b.lo
            b.hi;
        check rest
    | _ -> ()
  in
  check spans;
  spans

(* Splits spans at line ends, so each piece lies on one line. *)
let pieces src (spans : span list) : piece array =
  let n = String.length src in
  let starts =
    let acc = ref [ 0 ] in
    String.iteri (fun i c -> if c = '\n' && i + 1 <= n then acc := (i + 1) :: !acc) src;
    Array.of_list (List.rev !acc)
  in
  let line_of off =
    (* the last start <= off *)
    let lo = ref 0 and hi = ref (Array.length starts - 1) in
    while !lo < !hi do
      let mid = (!lo + !hi + 1) / 2 in
      if starts.(mid) <= off then lo := mid else hi := mid - 1
    done;
    !lo
  in
  let out = ref [] in
  List.iter
    (fun s ->
      let pos = ref s.lo in
      while !pos < s.hi do
        let l = line_of !pos in
        let eol = match String.index_from_opt src !pos '\n' with Some e -> e | None -> n in
        let stop = min eol s.hi in
        if stop > !pos then
          out :=
            { off = !pos; line = l + 1; col = !pos - starts.(l);
              len = stop - !pos; p_cls = s.cls; p_target = s.target }
            :: !out;
        pos := eol + 1
      done)
    spans;
  Array.of_list (List.rev !out)

(* A span's JSON: [target] is present only on a span that links. *)
let piece_json ~line ~col (p : piece) =
  let target =
    match p.p_target with
    | None -> []
    | Some t ->
        [ ( "target",
            `Assoc [ ("module", `String t.t_module); ("type", `String t.t_type) ]
          ) ]
  in
  `Assoc
    ([ ("line", `Int line); ("col", `Int col); ("len", `Int p.len);
       ("class", `String (class_name p.p_cls)) ]
    @ target)

(* The pieces inside source bytes [lo, hi), placed in a declaration's
   stripped text. A piece in a cut comment drops out whole; a piece that
   survives only in part would be a fault in the cutting. *)
let decl_spans path (all : piece array) lo hi (st : stripped) =
  Array.to_list all
  |> List.filter (fun p -> p.off >= lo && p.off + p.len <= hi)
  |> List.filter_map (fun p ->
         let places = List.init p.len (fun k -> Hashtbl.find_opt st.origin (p.off + k)) in
         match places with
         | Some (line, col) :: _
           when List.for_all2
                  (fun k pl -> pl = Some (line, col + k))
                  (List.init p.len Fun.id) places ->
             Some (piece_json ~line ~col p)
         | _ when List.for_all Option.is_none places -> None
         | _ -> fail "%s:%d: a span survives comment cutting only in part" path p.line)

(* ---- the .mli doc index ---- *)

type mli_entry = {
  m_doc : string option;
  m_members : (string * string option) list;
}

let mli_members (td : type_declaration) =
  match td.ptype_kind with
  | Ptype_variant cds ->
      List.map (fun cd -> (cd.pcd_name.txt, doc_of_attrs cd.pcd_attributes)) cds
  | Ptype_record lds ->
      List.map (fun ld -> (ld.pld_name.txt, doc_of_attrs ld.pld_attributes)) lds
  | _ -> []

let rec index_sig tbl modname (sg : signature) =
  List.iter
    (fun item ->
      match item.psig_desc with
      | Psig_type (_, tds) ->
          List.iter
            (fun td ->
              Hashtbl.replace tbl
                (modname, td.ptype_name.txt)
                { m_doc = doc_of_attrs td.ptype_attributes;
                  m_members = mli_members td })
            tds
      | Psig_module
          { pmd_name = { txt = Some n; _ };
            pmd_type = { pmty_desc = Pmty_signature s; _ };
            _ } ->
          index_sig tbl (modname ^ "." ^ n) s
      | _ -> ())
    sg

let parse_with f path =
  let src = read_file path in
  let lexbuf = Lexing.from_string src in
  Location.init lexbuf path;
  (src, f lexbuf)

(* ---- per-file extraction ---- *)

type ctx = {
  path : string;  (* as the JSON names it *)
  src : string;
  cuts : cuts;
  pieces : piece array;
  mli : (string * string, mli_entry) Hashtbl.t;
  mutable group : int;
  mutable seen : (string * string * int) list;
      (* module path, type name, line of each declaration documented *)
}

let jstr_opt = function None -> `Null | Some s -> `String s

let member_json ~name ~typ ~mut ~doc =
  `Assoc
    [ ("name", `String name); ("type", `String typ); ("mutable", `Bool mut);
      ("doc", jstr_opt doc) ]

let type_json ctx modname ~group (td : type_declaration) =
  let loc = td.ptype_loc in
  let lo = cnum loc.loc_start and hi = cnum loc.loc_end in
  let col = loc.loc_start.pos_cnum - loc.loc_start.pos_bol in
  let stripped = strip_comments ctx.src ctx.cuts lo hi col in
  ctx.seen <- (modname, td.ptype_name.txt, loc.loc_start.pos_lnum) :: ctx.seen;
  let mli = Hashtbl.find_opt ctx.mli (modname, td.ptype_name.txt) in
  let doc =
    match doc_of_attrs td.ptype_attributes with
    | Some d -> Some d
    | None -> Option.bind mli (fun e -> e.m_doc)
  in
  let mdoc name own =
    match own with
    | Some _ -> own
    | None -> (
        match mli with
        | None -> None
        | Some e -> Option.join (List.assoc_opt name e.m_members))
  in
  let kind, members, member_types =
    match td.ptype_kind with
    | Ptype_variant cds ->
        let ms =
          List.map
            (fun cd ->
              let args = print_args cd.pcd_args in
              let typ =
                match cd.pcd_res with
                | None -> args
                | Some r -> args ^ " : " ^ print_type r
              in
              member_json ~name:cd.pcd_name.txt ~typ ~mut:false
                ~doc:(mdoc cd.pcd_name.txt (doc_of_attrs cd.pcd_attributes)))
            cds
        in
        let ts =
          List.concat_map
            (fun cd ->
              (match cd.pcd_args with
               | Pcstr_tuple ts -> ts
               | Pcstr_record lds -> List.map (fun l -> l.pld_type) lds)
              @ Option.to_list cd.pcd_res)
            cds
        in
        ("variant", ms, ts)
    | Ptype_record lds ->
        let ms =
          List.map
            (fun ld ->
              member_json ~name:ld.pld_name.txt ~typ:(print_type ld.pld_type)
                ~mut:(ld.pld_mutable = Asttypes.Mutable)
                ~doc:(mdoc ld.pld_name.txt (doc_of_attrs ld.pld_attributes)))
            lds
        in
        ("record", ms, List.map (fun l -> l.pld_type) lds)
    | Ptype_open -> ("extensible", [], [])
    | Ptype_abstract ->
        ((if td.ptype_manifest = None then "abstract" else "alias"), [], [])
  in
  let refs = refs_of_types (Option.to_list td.ptype_manifest @ member_types) in
  `Assoc
    [ ("name", `String td.ptype_name.txt);
      ("line", `Int loc.loc_start.pos_lnum);
      ("kind", `String kind);
      ( "params",
        `List (List.map (fun (t, _) -> `String (print_type t)) td.ptype_params)
      );
      ("recursive_group", match group with None -> `Null | Some g -> `Int g);
      ("decl", `String stripped.text);
      ("decl_spans", `List (decl_spans ctx.path ctx.pieces lo hi stripped));
      ("doc", jstr_opt doc);
      ("members", `List members);
      ("refs", `List (List.map (fun s -> `String s) refs)) ]

(* Module doc: the first floating docstring before the first item that is not
   an attribute. *)
let module_doc (str : structure) =
  let rec go = function
    | { pstr_desc = Pstr_attribute a; _ } :: rest ->
        if a.attr_name.txt = "ocaml.text" then
          match string_payload a.attr_payload with
          | Some s -> Some s
          | None -> go rest
        else go rest
    | _ -> None
  in
  go str

let rec struct_of_mod (me : module_expr) =
  match me.pmod_desc with
  | Pmod_structure s -> Some s
  | Pmod_constraint (m, _) -> struct_of_mod m
  | _ -> None

let rec walk ctx path modname (str : structure) : Yojson.Safe.t list =
  let types = ref [] and nested = ref [] in
  List.iter
    (fun item ->
      match item.pstr_desc with
      | Pstr_type (_, tds) ->
          let group =
            match tds with
            | [] | [ _ ] -> None
            | _ ->
                ctx.group <- ctx.group + 1;
                Some ctx.group
          in
          List.iter
            (fun td -> types := type_json ctx modname ~group td :: !types)
            tds
      | Pstr_module { pmb_name = { txt = Some n; _ }; pmb_expr; _ } -> (
          match struct_of_mod pmb_expr with
          | Some s -> nested := walk ctx path (modname ^ "." ^ n) s :: !nested
          | None -> ())
      | _ -> ())
    str;
  `Assoc
    [ ("name", `String modname); ("path", `String path);
      ("doc", jstr_opt (module_doc str));
      ("types", `List (List.rev !types)) ]
  :: List.concat (List.rev !nested)

let unit_of file =
  String.capitalize_ascii (Filename.remove_extension (Filename.basename file))

(* The .cmt of a source file: [foo.ml] compiles to [foo.cmt], which must be
   in exactly one of the directories and must have been compiled from the
   file as it is now. *)
let load_cmt cmt_dirs file : Cmt_format.cmt_infos =
  let name = String.uncapitalize_ascii (Filename.remove_extension (Filename.basename file)) ^ ".cmt" in
  let found =
    List.filter Sys.file_exists (List.map (fun d -> Filename.concat d name) cmt_dirs)
  in
  let cmt =
    match found with
    | [ c ] -> c
    | [] ->
        fail "%s: no %s in any --cmt-dir (%s); build the library first" file name
          (String.concat ", " cmt_dirs)
    | cs -> fail "%s: %s is in several --cmt-dir: %s" file name (String.concat ", " cs)
  in
  let infos = Cmt_format.read_cmt cmt in
  (match infos.cmt_source_digest with
   | Some d when d = Digest.file file -> ()
   | Some _ ->
       fail "%s: %s was compiled from another version of the file; rebuild" file cmt
   | None -> fail "%s: %s records no source digest" file cmt);
  infos

let read_typed world cmt_dirs file =
  let infos = load_cmt cmt_dirs file in
  let str =
    match infos.cmt_annots with
    | Implementation s -> s
    | _ -> fail "%s: its .cmt holds no implementation" file
  in
  let fname =
    match infos.cmt_sourcefile with
    | Some f -> f
    | None -> fail "%s: its .cmt names no source file" file
  in
  let t =
    { ty_unit = unit_of file; ty_fname = fname; ty_str = str;
      type_ids = Ident.Tbl.create 64; mod_ids = Ident.Tbl.create 8;
      aliases = Ident.Tbl.create 8; decls = [] }
  in
  register world t t.ty_unit str;
  t

let process root world (t : typed) file =
  let src, str = parse_with Parse.implementation file in
  let comment_cuts =
    List.map
      (fun (_, (l : Location.t)) -> (cnum l.loc_start, cnum l.loc_end))
      (Lexer.comments ())
  in
  let doc_cuts = ref [] in
  let it =
    { Ast_iterator.default_iterator with
      attribute =
        (fun self a ->
          if a.attr_name.txt = "ocaml.doc" || a.attr_name.txt = "ocaml.text"
          then
            doc_cuts :=
              (cnum a.attr_loc.loc_start, cnum a.attr_loc.loc_end) :: !doc_cuts;
          Ast_iterator.default_iterator.attribute self a) }
  in
  it.structure it str;
  let base = unit_of file in
  let mli = Hashtbl.create 16 in
  let mli_path = Filename.remove_extension file ^ ".mli" in
  if Sys.file_exists mli_path then begin
    let _, sg = parse_with Parse.interface mli_path in
    index_sig mli base sg
  end;
  let path = relativize root file in
  let spans = file_spans world t path src in
  let ctx =
    { path; src; cuts = comment_cuts @ !doc_cuts; pieces = pieces src spans; mli;
      group = 0; seen = [] }
  in
  let modules = walk ctx path base str in
  (* The parse and the typed tree must agree on what is declared where, or
     the typed tree's answers would be attached to the wrong declarations. *)
  if List.sort compare ctx.seen <> List.sort compare t.decls then
    fail "%s: the parse and the .cmt disagree on the type declarations" path;
  let lines =
    let n = String.length src in
    let nl = ref 0 in
    String.iter (fun c -> if c = '\n' then incr nl) src;
    if n > 0 && src.[n - 1] <> '\n' then !nl + 1 else !nl
  in
  let source =
    `Assoc
      [ ("path", `String path); ("lines", `Int lines);
        ("bytes", `Int (String.length src));
        ( "spans",
          `List
            (Array.to_list
               (Array.map (fun p -> piece_json ~line:p.line ~col:p.col p) ctx.pieces)) ) ]
  in
  (modules, source)

let usage =
  "usage: typedoc_ocaml [--root DIR] --cmt-dir DIR [--cmt-dir DIR ...] FILE.ml ..."

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let rec parse root dirs files = function
    | "--root" :: d :: rest -> parse (Some (absolute d)) dirs files rest
    | "--cmt-dir" :: d :: rest -> parse root (d :: dirs) files rest
    | ("--root" | "--cmt-dir") :: [] -> fail "%s" usage
    | f :: rest -> parse root dirs (f :: files) rest
    | [] -> (root, List.rev dirs, List.rev files)
  in
  let root, cmt_dirs, files = parse None [] [] args in
  if cmt_dirs = [] then fail "no --cmt-dir given\n%s" usage;
  let root =
    match root with Some _ -> root | None -> find_git_root (Sys.getcwd ())
  in
  let world =
    { units = SSet.of_list (List.map unit_of files); modules = SSet.empty;
      final = Hashtbl.create 256 }
  in
  let typed = List.map (read_typed world cmt_dirs) files in
  let results = List.map2 (process root world) typed files in
  let modules = List.concat_map fst results in
  let sources = List.map snd results in
  print_string
    (Yojson.Safe.pretty_to_string
       (`Assoc
         [ ("language", `String "ocaml"); ("modules", `List modules);
           ("sources", `List sources) ]));
  print_newline ()
