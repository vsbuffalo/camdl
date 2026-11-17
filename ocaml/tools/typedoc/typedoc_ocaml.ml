(* Prints a JSON description of every type declaration, with its doc
   comments, in the OCaml source files given on the command line. Usage:

     typedoc_ocaml [--root DIR] FILE.ml ...

   Paths in the output are relative to DIR, by default the enclosing git
   checkout (the nearest ancestor of the working directory with a `.git`).

   Implementation comments and doc comments are cut out of each
   declaration's source text; doc comments are reported separately in `doc`
   fields. Doc comments missing in `foo.ml` fall back to `foo.mli`. The JSON
   contract is described in docs/documentation-style.md, section 10. *)

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

let strip_comments src (cuts : cuts) lo hi col =
  let b = Buffer.create (hi - lo + 16) in
  Buffer.add_string b (String.make col ' ');
  let pos = ref lo in
  List.iter
    (fun (a, z) ->
      if z > !pos && a < hi then begin
        if a > !pos then Buffer.add_string b (String.sub src !pos (a - !pos));
        pos := max !pos (min z hi)
      end)
    (List.sort compare cuts);
  if !pos < hi then Buffer.add_string b (String.sub src !pos (hi - !pos));
  let rstrip l =
    let n = ref (String.length l) in
    while
      !n > 0 && (l.[!n - 1] = ' ' || l.[!n - 1] = '\t' || l.[!n - 1] = '\r')
    do
      decr n
    done;
    String.sub l 0 !n
  in
  let lines =
    String.split_on_char '\n' (Buffer.contents b)
    |> List.map rstrip
    |> List.filter (fun l -> l <> "")
  in
  let m = List.fold_left (fun m l -> min m (indent_of l)) max_int lines in
  let m = if m = max_int then 0 else m in
  String.concat "\n" (List.map (drop m) lines)

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
  src : string;
  cuts : cuts;
  mli : (string * string, mli_entry) Hashtbl.t;
  mutable group : int;
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
  let decl = strip_comments ctx.src ctx.cuts lo hi col in
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
      ("decl", `String decl);
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

let process root file =
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
  let base =
    String.capitalize_ascii (Filename.remove_extension (Filename.basename file))
  in
  let mli = Hashtbl.create 16 in
  let mli_path = Filename.remove_extension file ^ ".mli" in
  if Sys.file_exists mli_path then begin
    let _, sg = parse_with Parse.interface mli_path in
    index_sig mli base sg
  end;
  let ctx = { src; cuts = comment_cuts @ !doc_cuts; mli; group = 0 } in
  walk ctx (relativize root file) base str

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let rec parse root files = function
    | "--root" :: d :: rest -> parse (Some (absolute d)) files rest
    | f :: rest -> parse root (f :: files) rest
    | [] -> (root, List.rev files)
  in
  let root, files = parse None [] args in
  let root =
    match root with Some _ -> root | None -> find_git_root (Sys.getcwd ())
  in
  let modules = List.concat_map (process root) files in
  print_string
    (Yojson.Safe.pretty_to_string
       (`Assoc [ ("language", `String "ocaml"); ("modules", `List modules) ]));
  print_newline ()
