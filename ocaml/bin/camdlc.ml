let usage_text = {|camdlc — the camdl compiler. Expands a .camdl model into the JSON
intermediate representation (IR) the runtime consumes, with unit/dimension
checking and symbolic gradients.

Usage:
  camdlc FILE.camdl [--set NAME=VALUE ...]   compile to IR JSON (stdout)
  camdlc check   FILE.camdl                  parse + type-check; report diagnostics
  camdlc inspect FILE.camdl [OPTIONS]        print model structure (summary, dims, ...)
  camdlc doctest [--gate] FILE.md ...        compile the camdl blocks in Markdown docs
  camdlc render  FILE.camdl [--format json]  render the model as LaTeX (or JSON, for display)

Flags (compile):
  --set NAME=VALUE   override a parameter value
  --json-errors      emit diagnostics as a JSON array to stderr
  --no-dim-check     disable dimensional analysis (only for a confirmed false positive)
  --no-state-grad    skip the state-Jacobian (rate_state_grad); smaller IR for
                     forward sim + gradient-free fits (IF2/PMMH/PF/MH), but the
                     IR can't be fit with `nuts` on the ODE backend
  --quantities FILE  compile with FILE's `quantities { }` block in place of the
                     model's own (a reporting vocabulary; FILE may contain
                     nothing else). Replaces, never merges.
  --camdl-version    print the compiler's git hash

To run models — simulate, fit, profile, survey, browse results — use `camdl`,
which also wraps these compiler commands (camdl compile/check/inspect/doctest).
Run `camdl --help`.
|}

let () =
  let args = Array.to_list Sys.argv |> List.tl in
  try
  match args with
  | [] ->
    print_string usage_text;
    exit 1

  (* ── camdlc --help / -h ───────────────────────────────────────────── *)
  | "--help" :: _ | "-h" :: _ ->
    print_string usage_text;
    exit 0

  (* ── camdlc doctest [--gate] [--verbose] FILE.md ... ──────────────── *)
  | "doctest" :: rest ->
    Doctest.main rest

  (* ── camdlc render FILE.camdl  (LaTeX of the indexed, pre-expansion model) ── *)
  | "render" :: rest ->
    Latex.run rest

  (* ── camdlc --camdl-version ──────────────────────────────────────── *)
  | ["--camdl-version"] | "--camdl-version" :: _ ->
    print_endline Version.git_hash;
    exit 0

  (* ── camdlc check FILE ────────────────────────────────────────────── *)
  | "check" :: rest ->
    (* M26 in 2026-04-19 review: --no-dim-check previously only
       registered on the `compile` subcommand's Arg.Unit handler,
       so `camdlc check --no-dim-check model.camdl` silently
       ignored the flag. Parse it here too. *)
    let path = ref None in
    List.iter (fun a -> match a with
      | "--no-dim-check" -> Compiler.no_dim_check := true
      | s when String.length s > 0 && s.[0] = '-' ->
        Printf.eprintf "error: unknown flag '%s' for `camdlc check`\n" s;
        exit 1
      | s -> path := Some s
    ) rest;
    (match !path with
     | None -> print_endline "usage: camdlc check [--no-dim-check] FILE.camdl"; exit 1
     | Some p -> Inspect.run_check p)

  (* ── camdlc inspect FILE [options] ───────────────────────────────── *)
  | "inspect" :: rest ->
    let files     = ref [] in
    let summary   = ref false in
    let cost_report = ref false in
    let comps     = ref false in
    let transitions_pat = ref None in
    let do_transitions  = ref false in
    let tr_rate   = ref None in
    let tr_count  = ref false in
    let let_name  = ref None in
    let ir_mode   = ref false in
    let ascii     = ref false in
    let no_color  = ref false in
    let dims      = ref false in
    let do_tables = ref false in
    let tables_pat = ref None in
    let do_forcings = ref false in
    let forcings_pat = ref None in
    let do_parameters = ref false in
    let rec parse = function
      | [] -> ()
      | "--summary"      :: tl -> summary := true;         parse tl
      | "--cost-report"  :: tl -> cost_report := true;     parse tl
      | "--dims"         :: tl -> dims    := true;         parse tl
      | "--compartments" :: tl -> comps   := true;         parse tl
      | "--parameters"   :: tl -> do_parameters := true;   parse tl
      | "--transitions"  :: tl ->
        do_transitions := true;
        (match tl with
         | s :: tl2 when not (String.length s > 0 && s.[0] = '-') ->
           transitions_pat := Some s; parse tl2
         | _ -> parse tl)
      | "--transition" :: name :: tl ->
        tr_rate := Some name; parse tl
      | "--count" :: tl ->
        tr_count := true; parse tl
      | "--let" :: name :: tl ->
        let_name := Some name; parse tl
      | "--tables" :: tl ->
        do_tables := true;
        (match tl with
         | s :: tl2 when not (String.length s > 0 && s.[0] = '-') ->
           tables_pat := Some s; parse tl2
         | _ -> parse tl)
      | "--forcings" :: tl ->
        do_forcings := true;
        (match tl with
         | s :: tl2 when not (String.length s > 0 && s.[0] = '-') ->
           forcings_pat := Some s; parse tl2
         | _ -> parse tl)
      | "--ir"       :: tl -> ir_mode   := true; parse tl
      | "--ascii"    :: tl -> ascii     := true; parse tl
      | "--no-color" :: tl -> no_color  := true; parse tl
      | s :: tl when not (String.length s > 0 && s.[0] = '-') ->
        files := s :: !files; parse tl
      | s :: _ ->
        (* Per CLAUDE.md "no loose semantics" — unknown flags are
           typos (e.g. --sumary for --summary) and silently continuing
           produces default output that masks the user's intent.
           Hard exit with the flag named. *)
        Printf.eprintf "error: unknown flag '%s'\n" s;
        Printf.eprintf "  run `camdlc inspect --help` for supported flags\n";
        exit 1
    in
    parse rest;
    let path = match List.rev !files with
      | [] -> print_endline "usage: camdlc inspect FILE.camdl [OPTIONS]"; exit 1
      | p :: _ -> p
    in
    let cmd =
      if !cost_report       then Inspect.CostReport
      else if !dims         then Inspect.Dims
      else if !do_tables    then Inspect.Tables !tables_pat
      else if !do_forcings  then Inspect.Forcings !forcings_pat
      else if !comps             then Inspect.Compartments
      else if !do_parameters then Inspect.Parameters
      else if !do_transitions then Inspect.Transitions !transitions_pat
      else if !tr_count then Inspect.TransitionCount !transitions_pat
      else (match !tr_rate with
        | Some name -> Inspect.TransitionRate name
        | None ->
      match !let_name with
        | Some name -> Inspect.LetBinding name
        | None -> Inspect.Summary)
    in
    let opts : Inspect.inspect_opts = {
      cmd;
      ir_mode  = !ir_mode;
      ascii    = !ascii;
      no_color = !no_color;
    } in
    Inspect.run_inspect path opts

  (* ── camdlc FILE.camdl [--set ...] [-o FILE] (default: compile) ──── *)
  | _ ->
    let usage  = "camdlc FILE.camdl [--set NAME=VALUE ...] [-o FILE]" in
    let files  = ref [] in
    let set_kvs = ref [] in
    let output_path = ref "" in       (* "" → write to stdout *)
    let set_output p = output_path := p in
    let pretty_output = ref false in  (* false → canonical compact IR JSON *)
    let emit_deps_path = ref "" in    (* "" → don't emit a read-closure depfile *)
    (* "" → use the model's own `quantities { }` block *)
    let quantities_path = ref "" in
    let spec = [
      ("--set", Arg.String (fun s ->
        match String.split_on_char '=' s with
        | [k; v] -> set_kvs := (k, float_of_string v) :: !set_kvs
        | _ -> Printf.eprintf "bad --set %s (want NAME=VALUE)\n" s; exit 1
       ), "NAME=VALUE  override a parameter value");
      ("--json-errors", Arg.Unit (fun () ->
        Diagnostics.json_errors_mode := true
       ), " emit diagnostics as JSON array to stderr instead of ANSI text");
      ("--no-dim-check", Arg.Unit (fun () ->
        Compiler.no_dim_check := true
       ), " disable dimensional analysis checking");
      ("--no-state-grad", Arg.Unit (fun () ->
        Compiler.no_state_grad := true
       ), " skip the state-Jacobian (rate_state_grad/projection_state_grad); \
           shrinks the IR for forward simulation and gradient-free fits \
           (IF2/PMMH/PF/MH). A model compiled this way cannot be fit with \
           `nuts` on the ODE backend");
      ("-o", Arg.String set_output,
       "FILE  write IR JSON to FILE instead of stdout");
      ("--output", Arg.String set_output,
       "FILE  write IR JSON to FILE instead of stdout (long form of -o)");
      ("--pretty", Arg.Set pretty_output,
       " emit indented (human-readable) IR JSON instead of the default compact form");
      ("--emit-deps", Arg.String (fun p -> emit_deps_path := p),
       "FILE  also write the compile's external-data read-closure to FILE (JSON)");
      ("--quantities", Arg.String (fun p -> quantities_path := p),
       "FILE  replace the model's `quantities { }` block with FILE's (FILE may \
              contain only a quantities block)");
    ] in
    Arg.parse_argv (Array.of_list ("camdlc" :: args))
      spec (fun f -> files := f :: !files) usage;
    (match List.rev !files with
     | [] -> print_endline usage; exit 1
     | path :: _ ->
       let name = Filename.basename path |> Filename.remove_extension in
       let src =
         let ic = open_in path in
         let n  = in_channel_length ic in
         let s  = Bytes.create n in
         really_input ic s 0 n;
         close_in ic;
         Bytes.to_string s
       in
       (* The reporting vocabulary, as a second compilation unit. Read here so
          a missing/unreadable file surfaces as the same Sys_error diagnostic a
          missing model does. *)
       let quantities =
         if !quantities_path = "" then None
         else
           let qp = !quantities_path in
           let ic = open_in qp in
           let n  = in_channel_length ic in
           let s  = Bytes.create n in
           really_input ic s 0 n;
           close_in ic;
           Some (qp, Bytes.to_string s)
       in
       match Compiler.compile_with_reads ~name ~filename:path ?quantities src with
       | Error e when e = "compilation failed"
                   || (String.length e > 0 && e.[0] = '[') ->
         (* Diagnostics already rendered to stderr (text or JSON) by
            [Compiler.compile] — the sniff matches its rendered payload
            ("compilation failed", or a "["-prefixed JSON array under
            --json-errors). Exit 1 without re-printing on a fresh line
            (m5 in the 2026-04-19 compiler review). *)
         exit 1
       | Error e -> Printf.eprintf "Error: %s\n" e; exit 1
       | Ok (m, reads) ->
         (* Read-closure depfile (`--emit-deps`): the external data files this
            compile opened. Written before the IR so a downstream `camdl mre`
            sees it even if IR streaming is later interrupted. *)
         if !emit_deps_path <> "" then
           Compiler.write_depfile ~path:!emit_deps_path ~model:path reads;
         let overrides = List.rev !set_kvs in
         let m = if overrides = [] then m else
           { m with Ir.parameters =
               List.map (fun (p : Ir.parameter) ->
                 match List.assoc_opt p.name overrides with
                 (* --set pins the parameter to a fixed constant. *)
                 | Some v -> { p with value = Ir.Fixed v }
                 | None   -> p
               ) m.Ir.parameters
           }
         in
         (* Stream the IR JSON straight to the destination channel rather than
            materializing the whole (multi-GB at scale) string first — see
            Serde.model_to_channel. Default is the canonical compact form;
            --pretty selects the indented view. *)
         let pretty = !pretty_output in
         (if !output_path = "" then begin
           (* Default: write to stdout, preserving trailing newline. *)
           Passtime.time "serialize" (fun () -> Serde.model_to_channel ~pretty stdout m);
           print_newline ()
         end else begin
           (* -o / --output FILE: write IR JSON to FILE. Includes the
              trailing newline so file output matches `camdl compile … > FILE`. *)
           let oc = open_out !output_path in
           Passtime.time "serialize" (fun () -> Serde.model_to_channel ~pretty oc m);
           output_char oc '\n';
           close_out oc
         end);
         (* Env-gated per-pass timing breakdown to stderr (no-op unless
            CAMDL_TIME_PASSES is set); never touches the IR on stdout/-o. *)
         Passtime.dump ())
  with Sys_error msg ->
    (* A bad input/output path (e.g. a misspelled .camdl) raises Sys_error from
       open_in / open_out; surface it as a clean diagnostic instead of an
       uncaught "Fatal error: exception …" trace. *)
    Printf.eprintf "error: %s\n" msg;
    exit 1
