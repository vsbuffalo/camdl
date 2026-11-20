(** [camdlc doctest]: compiles every camdl code block of a Markdown document
    with the real compiler, so a documented model that stops compiling is
    caught.

    {!main} takes [--gate], [--verbose] and the files; [make test-docs] runs
    it with [--gate] over the language specification and the user guides.
    {!parse_doc} reads a file in one pass into a {!doc}: every fenced block
    whose info string begins with [camdl], every preamble fence, and every
    data comment. A preamble fence, whose info string is
    [camdl-doctest-preamble LABEL], holds declarations a block may borrow,
    so a fragment can be shown alone and still compile; it is not itself a
    block, and the site drops it ([docs/_web/doctest-preamble.lua]). It is a
    fence rather than an HTML comment because the model text inside it has
    arrows: a comment ends at the first [-->], so the rest of the model
    would show on the page (gh#1324). A data comment holds a file a block
    reads, and ends at a line that is exactly [-->]:

    {v
    ```camdl-doctest-preamble sir
    compartments { S, I, R }
    parameters {
      gamma : rate
    }
    ```

    <!-- camdl-doctest-data: data/pop.tsv
    patch	pop
    north	50000
    -->

    ```camdl preamble=sir
    transitions { recovery : I --> R @ gamma * I }
    ```
    v}

    {!classify} compiles a block, prefixed by its preamble, through
    [Compiler.collect_diagnostics], the full compile that returns its
    diagnostics instead of stopping at the first error. The data files are
    written to a temporary directory, which stands in for the block's own
    directory, so [read("data/pop.tsv")] finds them. The result is a
    {!verdict}. The rules are heuristics over the error codes, applied in
    order: a block marked [ignore] is skipped; a block with no errors
    passes; a block with a data-file error ([E200]) is skipped; a block
    that names a preamble fails on any other error; a block that declares
    no [compartments { }] block is a fragment, skipped as one that cannot
    parse alone when its only errors are syntax errors ([E001]) and as a
    fragment otherwise; a block that declares compartments fails. Whether a
    block declares compartments is read from its code, not its comments
    ({!declares_compartments}). A preamble that is not defined, or an
    exception from the compiler, is a failure too. {!run} prints one line
    per failure (per block under [--verbose]) and a summary, and under
    [--gate] exits 1 if anything failed.

    A data comment's path must be relative and must not contain a [..]
    segment, so every file is written inside the temporary directory; a
    document with any other path fails ({!bad_data_paths}). So does a data
    comment whose contents have [-->], and a preamble written as an HTML
    comment ([<!-- camdl-doctest-preamble: LABEL]): the rendered page would
    end either at that text and show the rest
    ({!data_comments_ended_early}, [doc.comment_preambles]). *)

(* ── string helpers (no Str dependency) ──────────────────────────────────── *)

let find_sub ~sub s =
  let ls = String.length s and n = String.length sub in
  if n = 0 then Some 0
  else begin
    let rec go i =
      if i + n > ls then None
      else if String.sub s i n = sub then Some i
      else go (i + 1)
    in
    go 0
  end

let contains ~sub s = find_sub ~sub s <> None

let lstrip s =
  let n = String.length s in
  let i = ref 0 in
  while !i < n && (s.[!i] = ' ' || s.[!i] = '\t') do incr i done;
  String.sub s !i (n - !i)

let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

(* Split an info string into tokens on spaces, tabs and commas. *)
let tokens s =
  String.map (fun c -> if c = '\t' || c = ',' then ' ' else c) s
  |> String.split_on_char ' '
  |> List.filter (fun t -> t <> "")

(* ── filesystem helpers for inline data ───────────────────────────────────── *)

let rec mkdir_p dir =
  if dir <> "" && dir <> "." && dir <> "/" && not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    (try Sys.mkdir dir 0o755 with Sys_error _ -> ())
  end

let rec rm_rf path =
  if Sys.file_exists path then
    if Sys.is_directory path then begin
      Array.iter (fun n -> rm_rf (Filename.concat path n)) (Sys.readdir path);
      (try Sys.rmdir path with Sys_error _ -> ())
    end
    else (try Sys.remove path with Sys_error _ -> ())

let write_file path content =
  mkdir_p (Filename.dirname path);
  let oc = open_out path in
  output_string oc content;
  close_out oc

let make_temp_dir () =
  let f = Filename.temp_file "camdl-doctest-" "" in
  Sys.remove f;
  Sys.mkdir f 0o755;
  f

(* ── document model ───────────────────────────────────────────────────────── *)

(** One fenced camdl block of a document, as written. *)
type block = {
  file     : string;         (** The document's path, as given on the command line. *)
  line     : int;            (** The 1-based line of the opening fence. *)
  ignore_  : bool;           (** The info string has the word [ignore]: the block is not compiled. *)
  preamble : string option;  (** The label of the preamble the block borrows, from [preamble=LABEL]. *)
  source   : string;         (** The block's text, without the fences. *)
}

(** What {!parse_doc} found in one document. *)
type doc = {
  blocks    : block list;  (** The camdl blocks, in document order. *)
  preambles : (string * string) list;  (** Each preamble fence's label and text. Every block of the document can use every preamble, wherever it appears; when a label is defined twice, the first definition is used. *)
  datas     : (string * string) list;  (** Each data comment's path, relative to the block's directory, and its contents. A path {!bad_data_paths} rejects is never written. *)
  comment_preambles : int list;  (** The 1-based line of each [<!-- camdl-doctest-preamble:] comment. That form is not read as a preamble: on the rendered page the first [-->] of a transition ends the comment and the rest of the model shows as text (gh#1324). {!run} reports each as a failure. *)
}

let parse_preamble toks =
  List.find_map
    (fun t ->
       if starts_with ~prefix:"preamble=" t
       then Some (String.sub t 9 (String.length t - 9))
       else None)
    toks

(* Recognise `<!-- camdl-doctest-KIND: ARG` on a line. ARG is everything after
   the colon, with any trailing `-->` and whitespace stripped. *)
let comment_open line =
  let t = lstrip line in
  let pfx = "<!-- camdl-doctest-" in
  if starts_with ~prefix:pfx t then
    let rest = String.sub t (String.length pfx) (String.length t - String.length pfx) in
    match String.index_opt rest ':' with
    | Some i ->
      let kind = String.trim (String.sub rest 0 i) in
      let arg0 = String.sub rest (i + 1) (String.length rest - i - 1) in
      let arg = match find_sub ~sub:"-->" arg0 with
        | Some j -> String.sub arg0 0 j
        | None -> arg0
      in
      Some (kind, String.trim arg)
    | None -> None
  else None

(* Single pass: fenced ```camdl blocks, ```camdl-doctest-preamble fences and
   HTML-comment data chunks. Fence bodies and comment bodies are kept
   verbatim. *)
let parse_doc file : doc =
  let ic = open_in file in
  let blocks = ref [] and preambles = ref [] and datas = ref [] in
  let comment_preambles = ref [] in
  let in_block = ref false and capturing = ref false in
  (* [Some label] while inside a preamble fence *)
  let defining = ref None in
  let buf = Buffer.create 256 and start = ref 0 and ign = ref false and pre = ref None in
  let in_comment = ref false and ckind = ref "" and carg = ref "" in
  let cbuf = Buffer.create 256 in
  let lineno = ref 0 in
  let flush_comment () =
    (match !ckind with
     | "data"     -> datas := (!carg, Buffer.contents cbuf) :: !datas
     | _ -> ());
    in_comment := false;
    Buffer.clear cbuf
  in
  (try
     while true do
       let line = input_line ic in
       incr lineno;
       let t = lstrip line in
       if !in_comment then begin
         (* The closing `-->` must be on its own line, so that a comment
            preamble's transitions (`S --> I`) are skipped with it rather
            than read as the document. *)
         if String.trim line = "-->" then flush_comment ()
         else begin
           Buffer.add_string cbuf line;
           Buffer.add_char cbuf '\n'
         end
       end
       else if !in_block then begin
         if starts_with ~prefix:"```" t then begin
           (match !defining with
            | Some label -> preambles := (label, Buffer.contents buf) :: !preambles
            | None ->
              if !capturing then
                blocks :=
                  { file; line = !start; ignore_ = !ign; preamble = !pre;
                    source = Buffer.contents buf }
                  :: !blocks);
           in_block := false; capturing := false; defining := None
         end
         else if !capturing then begin
           Buffer.add_string buf line;
           Buffer.add_char buf '\n'
         end
       end
       else begin
         match comment_open line with
         | Some (kind, arg) ->
           if kind = "preamble" then comment_preambles := !lineno :: !comment_preambles;
           in_comment := true; ckind := kind; carg := arg; Buffer.clear cbuf;
           (* a single-line `<!-- … --> ` closes immediately with empty body *)
           if contains ~sub:"-->" line then flush_comment ()
         | None ->
           if starts_with ~prefix:"```" t then begin
             let info = String.trim (String.sub t 3 (String.length t - 3)) in
             match tokens info with
             | "camdl" :: rest ->
               in_block := true; capturing := true;
               start := !lineno; ign := List.mem "ignore" rest;
               pre := parse_preamble rest;
               Buffer.clear buf
             | "camdl-doctest-preamble" :: label :: _ ->
               in_block := true; capturing := true; defining := Some label;
               Buffer.clear buf
             | _ ->
               in_block := true; capturing := false
           end
       end
     done
   with End_of_file -> ());
  close_in ic;
  { blocks = List.rev !blocks;
    preambles = List.rev !preambles;
    datas = List.rev !datas;
    comment_preambles = List.rev !comment_preambles }

(** The data-comment paths that would be written outside the temporary
    directory: an absolute path, or one with a [..] segment. *)
let bad_data_paths (datas : (string * string) list) : string list =
  let escapes path =
    (not (Filename.is_relative path))
    || List.mem ".." (String.split_on_char '/' path)
  in
  List.filter_map (fun (path, _) -> if escapes path then Some path else None) datas

(** The data-comment paths whose contents have [-->]. The doctest reads such
    a comment to its closing line, but the rendered page ends it at that
    text and shows the rest, as a preamble comment's transitions did
    (gh#1324). *)
let data_comments_ended_early (datas : (string * string) list) : string list =
  List.filter_map
    (fun (path, content) -> if contains ~sub:"-->" content then Some path else None)
    datas

(* Materialise inline data chunks into a fresh temp dir; returns its path.
   Block `read("data/x.tsv")` paths resolve against this dir. A path
   [bad_data_paths] rejects is not written; [run] reports it as a failure. *)
let materialize_data datas =
  let dir = make_temp_dir () in
  let bad = bad_data_paths datas in
  List.iter
    (fun (path, content) ->
       if not (List.mem path bad) then write_file (Filename.concat dir path) content)
    datas;
  dir

(** Whether [src] declares a [compartments { }] block, read from its code:
    a [#] comment (other than a [#\[] attribute) is dropped to the end of
    its line, and [compartments] counts only as a whole word followed,
    after any whitespace, by [{], so neither a comment that mentions
    compartments nor an index over them ([c in compartments]) counts. This
    is what separates a complete model from a fragment. *)
let declares_compartments (src : string) : bool =
  let code =
    String.split_on_char '\n' src
    |> List.map (fun line ->
           let n = String.length line in
           let rec cut i =
             if i >= n then line
             else if line.[i] = '#' && not (i + 1 < n && line.[i + 1] = '[') then
               String.sub line 0 i
             else cut (i + 1)
           in
           cut 0)
    |> String.concat "\n"
  in
  let kw = "compartments" in
  let n = String.length code and k = String.length kw in
  let ident c =
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c = '_'
  in
  let rec next_non_space j =
    if j < n && (code.[j] = ' ' || code.[j] = '\t' || code.[j] = '\n' || code.[j] = '\r')
    then next_non_space (j + 1)
    else j
  in
  let rec from i =
    if i + k > n then false
    else if String.sub code i k = kw
         && (i = 0 || not (ident code.[i - 1]))
         && (i + k = n || not (ident code.[i + k]))
         && (let j = next_non_space (i + k) in j < n && code.[j] = '{')
    then true
    else from (i + 1)
  in
  from 0

(* ── classification ───────────────────────────────────────────────────────── *)

(** The outcome of compiling one block. [Fail] and [Ice] count as failures
    under [--gate]; every [Skip_*] counts as a skip. *)
type verdict =
  | Pass                                 (** The block compiled with no errors. *)
  | Skip_ignore                          (** The block is marked [ignore]. *)
  | Skip_parse                           (** Every error is a syntax error ([E001]), and the block names no preamble and declares no compartments. *)
  | Skip_data                            (** An error is a data file that could not be read ([E200]). *)
  | Skip_fragment                        (** The block has errors, names no preamble, and declares no compartments. *)
  | Fail of Diagnostics.diagnostic list  (** The block has errors that no skip rule covers: the error diagnostics. *)
  | Ice of string                        (** The block names an undefined preamble, or the compiler raised an exception: the message. *)

let classify ~preambles ~basedir (b : block) : verdict =
  if b.ignore_ then Skip_ignore
  else
    let prefix =
      match b.preamble with
      | None -> Ok ""
      | Some label ->
        (match List.assoc_opt label preambles with
         | Some src -> Ok (src ^ "\n")
         | None -> Error (Printf.sprintf "preamble '%s' not defined in doc" label))
    in
    match prefix with
    | Error msg -> Ice msg
    | Ok pre ->
      let src = pre ^ b.source in
      (* filename's directory is the data dir, so read("…") resolves there *)
      let filename = Filename.concat basedir "doc.camdl" in
      (match
         (try `Ok (Compiler.collect_diagnostics ~filename src)
          with e -> `Raised (Printexc.to_string e))
       with
       | `Raised msg -> Ice msg
       | `Ok diags ->
         let errors =
           List.filter
             (fun (d : Diagnostics.diagnostic) -> d.severity = Diagnostics.Error) diags
         in
         if errors = [] then Pass
         else begin
           let codes = List.map (fun (d : Diagnostics.diagnostic) -> d.code) errors in
           if List.mem "E200" codes then Skip_data
           (* a preamble asserts the block compiles — residual error is a FAIL *)
           else if b.preamble <> None then Fail errors
           (* a block that declares compartments is a complete model, and a
              complete model that does not compile fails, syntax errors
              included *)
           else if declares_compartments b.source then Fail errors
           else if List.for_all (fun c -> c = "E001") codes then Skip_parse
           else Skip_fragment
         end)

(* ── report / entry point ─────────────────────────────────────────────────── *)

let run ~gate ~verbose files =
  let total = ref 0 and npass = ref 0 and nfail = ref 0 in
  let n_parse = ref 0 and n_frag = ref 0 and n_data = ref 0 and n_ign = ref 0 in
  List.iter
    (fun file ->
       let doc = parse_doc file in
       let basedir = materialize_data doc.datas in
       Printf.printf "\n%s — %d camdl block(s)\n" file (List.length doc.blocks);
       List.iter
         (fun path ->
            incr nfail;
            Printf.printf
              "  FAIL   data comment '%s': the path must be relative and must not \
               contain '..'; it was not written\n"
              path)
         (bad_data_paths doc.datas);
       List.iter
         (fun line ->
            incr nfail;
            Printf.printf
              "  FAIL   L%d  a preamble written as an HTML comment is not read: the \
               first `-->` of its model text ends the comment on the rendered page \
               and shows the rest; write it as a ```camdl-doctest-preamble LABEL \
               fence\n"
              line)
         doc.comment_preambles;
       List.iter
         (fun path ->
            incr nfail;
            Printf.printf
              "  FAIL   data comment '%s': its contents have `-->`, which ends the \
               comment on the rendered page and shows the rest\n"
              path)
         (data_comments_ended_early doc.datas);
       List.iter
         (fun b ->
            incr total;
            match classify ~preambles:doc.preambles ~basedir b with
            | Pass -> incr npass; if verbose then Printf.printf "  pass   L%d\n" b.line
            | Skip_ignore ->
              incr n_ign; if verbose then Printf.printf "  skip   L%d  (ignore)\n" b.line
            | Skip_parse ->
              incr n_parse; if verbose then Printf.printf "  skip   L%d  (parse-only fragment)\n" b.line
            | Skip_data ->
              incr n_data; if verbose then Printf.printf "  skip   L%d  (needs external data file)\n" b.line
            | Skip_fragment ->
              incr n_frag; if verbose then Printf.printf "  skip   L%d  (fragment)\n" b.line
            | Fail errors ->
              incr nfail;
              let codes = List.map (fun (d : Diagnostics.diagnostic) -> d.code) errors in
              let msg = match errors with d :: _ -> d.Diagnostics.message | [] -> "" in
              Printf.printf "  FAIL   L%d  [%s]  %s\n" b.line (String.concat "," codes) msg
            | Ice msg ->
              incr nfail;
              Printf.printf "  FAIL   L%d  [ICE]  %s\n" b.line msg)
         doc.blocks;
       rm_rf basedir)
    files;
  let nskip = !n_parse + !n_frag + !n_data + !n_ign in
  Printf.printf "\n── summary ──\n";
  Printf.printf
    "%d blocks: %d pass, %d skip (%d parse, %d fragment, %d data, %d ignore), %d FAIL\n"
    !total !npass nskip !n_parse !n_frag !n_data !n_ign !nfail;
  if gate && !nfail > 0 then begin
    Printf.printf "\ngate: FAILED (%d block(s) did not compile)\n" !nfail;
    exit 1
  end

let usage_text = {|usage: camdlc doctest [--gate] [--verbose] FILE.md ...

Compiles every ```camdl block of each Markdown file with the real compiler.

  --gate          exit 1 if any block fails
  --verbose, -v   print a line for every block, not only for failures
  --help, -h      print this help

A block's verdict:
  pass            it compiles with no errors
  skip (ignore)   its info string has the word `ignore` (```camdl ignore)
  skip (data)     it reads a data file that is not supplied
  skip (parse)    it declares no `compartments { }`, names no preamble, and
                  its only errors are syntax errors (E001): a legend or an
                  expression shown alone
  skip (fragment) it declares no `compartments { }` and names no preamble
  FAIL            anything else: a block that declares compartments, or names
                  a preamble (```camdl preamble=LABEL), must compile

Two markers, which the rendered page does not show, supply what a block needs:
  ```camdl-doctest-preamble LABEL      a fence of declarations a block may
                                       borrow; not itself a block
  <!-- camdl-doctest-data: PATH        a comment holding a file a block reads,
                                       ending at a line that is exactly `-->`;
                                       PATH must be relative, without `..`, and
                                       the contents must not have `-->`
A preamble written as an HTML comment (<!-- camdl-doctest-preamble: LABEL)
is not read and fails: the first `-->` of its model text ends the comment on
the rendered page.
|}

let main args =
  let gate = ref false and verbose = ref false and files = ref [] in
  List.iter
    (fun a ->
       match a with
       | "--help" | "-h" -> print_string usage_text; exit 0
       | "--gate" -> gate := true
       | "--verbose" | "-v" -> verbose := true
       | s when String.length s > 0 && s.[0] = '-' ->
         Printf.eprintf "doctest: unknown flag %s\n  run `camdlc doctest --help` for supported flags\n" s;
         exit 1
       | s -> files := s :: !files)
    args;
  let files = List.rev !files in
  if files = [] then begin
    print_string usage_text;
    exit 1
  end;
  run ~gate:!gate ~verbose:!verbose files
