(* Doctest: compile the ```camdl fenced code blocks in Markdown docs against the
   real compiler and classify each block's outcome.

   Focused fragments that reference declarations made elsewhere in a section can
   borrow a hidden preamble and inline data that travel WITH the doc as HTML
   comments — invisible in the rendered page, so the spec stays self-contained
   with nothing to drift out of sync:

     <!-- camdl-doctest-preamble: sir
     compartments { S, I, R }
     parameters { gamma : rate }
     -->

     <!-- camdl-doctest-data: data/pop.tsv
     patch<TAB>pop
     north<TAB>50000
     -->

     ```camdl preamble=sir
     transitions { recovery : I --> R @ gamma * I }   <- compiled as preamble ^ block
     ```

   A `preamble=LABEL` block is *asserted* to compile, so any residual error is a
   FAIL, not a skip. `camdl-doctest-data` chunks are materialised into a temp
   directory that the block's `read("…")` paths resolve against.

   The oracle is [Compiler.collect_diagnostics] — the full, non-aborting
   pipeline, returning structured diagnostics as values. *)

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

type block = {
  file     : string;
  line     : int;            (* 1-based line of the opening fence *)
  ignore_  : bool;           (* ```camdl ignore *)
  preamble : string option;  (* ```camdl preamble=LABEL *)
  source   : string;
}

type doc = {
  blocks    : block list;
  preambles : (string * string) list;  (* label -> hidden source *)
  datas     : (string * string) list;  (* relative path -> file content *)
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

(* Single pass: fenced ```camdl blocks + HTML-comment preamble/data chunks.
   Code-fence bodies and comment bodies are kept verbatim. *)
let parse_doc file : doc =
  let ic = open_in file in
  let blocks = ref [] and preambles = ref [] and datas = ref [] in
  let in_block = ref false and capturing = ref false in
  let buf = Buffer.create 256 and start = ref 0 and ign = ref false and pre = ref None in
  let in_comment = ref false and ckind = ref "" and carg = ref "" in
  let cbuf = Buffer.create 256 in
  let lineno = ref 0 in
  let flush_comment () =
    (match !ckind with
     | "preamble" -> preambles := (!carg, Buffer.contents cbuf) :: !preambles
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
         (* The closing `-->` must be on its own line. CAMDL transition syntax
            (`S --> I`) contains `-->`, so matching it mid-line would truncate
            a preamble at its first transition. *)
         if String.trim line = "-->" then flush_comment ()
         else begin
           Buffer.add_string cbuf line;
           Buffer.add_char cbuf '\n'
         end
       end
       else if !in_block then begin
         if starts_with ~prefix:"```" t then begin
           if !capturing then
             blocks :=
               { file; line = !start; ignore_ = !ign; preamble = !pre;
                 source = Buffer.contents buf }
               :: !blocks;
           in_block := false; capturing := false
         end
         else if !capturing then begin
           Buffer.add_string buf line;
           Buffer.add_char buf '\n'
         end
       end
       else begin
         match comment_open line with
         | Some (kind, arg) ->
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
             | _ ->
               in_block := true; capturing := false
           end
       end
     done
   with End_of_file -> ());
  close_in ic;
  { blocks = List.rev !blocks;
    preambles = List.rev !preambles;
    datas = List.rev !datas }

(* Materialise inline data chunks into a fresh temp dir; returns its path.
   Block `read("data/x.tsv")` paths resolve against this dir. *)
let materialize_data datas =
  let dir = make_temp_dir () in
  List.iter (fun (path, content) -> write_file (Filename.concat dir path) content) datas;
  dir

(* ── classification ───────────────────────────────────────────────────────── *)

type verdict =
  | Pass
  | Skip_ignore
  | Skip_parse
  | Skip_data
  | Skip_fragment
  | Fail of Diagnostics.diagnostic list
  | Ice of string

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
           else if List.for_all (fun c -> c = "E001") codes then Skip_parse
           else if not (contains ~sub:"compartments" b.source) then Skip_fragment
           else Fail errors
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

let main args =
  let gate = ref false and verbose = ref false and files = ref [] in
  List.iter
    (fun a ->
       match a with
       | "--gate" -> gate := true
       | "--verbose" | "-v" -> verbose := true
       | s when String.length s > 0 && s.[0] = '-' ->
         Printf.eprintf "doctest: unknown flag %s\n" s; exit 1
       | s -> files := s :: !files)
    args;
  let files = List.rev !files in
  if files = [] then begin
    print_endline "usage: camdlc doctest [--gate] [--verbose] FILE.md ...";
    exit 1
  end;
  run ~gate:!gate ~verbose:!verbose files
