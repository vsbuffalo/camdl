(** The driver of a compile: it runs the stages in order, decides when a
    compile stops, and hands back either the compiled model or the rendered
    diagnostics.

    A compile starts from the source text of a model, and of an optional
    vocabulary file given with [--quantities], and goes through three phases.
    The front end, [front_end_collect], parses each file, swaps the model's
    [quantities] block for the vocabulary file's, and calls
    {!Expander.expand_detail}. It then adds the lexer's warnings (W100) and the
    errors that parser actions recorded in {!Parser_errors} to the expander's
    diagnostics. A lexing or syntax error is E001 at the position where
    parsing stopped, or E115 when the cause is a duration written without its
    tick ([80 days] for [80 'days]); an exception raised anywhere in parsing
    or expansion is caught and becomes E001 with no location. The result is a
    {!compile_detail}: the expanded model with the context and inputs that
    produced it.

    The analysis, [run_analysis], runs the checks of the expanded model and
    the differentiation of transition rates: {!Validate}, then {!Dimcheck} and
    {!Lint}, then {!Autodiff} on every rate, then the W203 warnings of the
    declared-model section. It stops after validation if validation reported
    anything, and after the dimension check and lint, or after
    differentiation, if any error has been reported by then. The passes over
    the IR return their diagnostics without source locations; this module
    gives each the location of the declaration it names, through the
    expander's [*_loc] lookups, or {!Diagnostics.no_loc} when there is none.
    The finish, [finish_compile], differentiates the rest of the model (the
    overdispersion variances, the observations, the projections, the initial
    state and the arguments of initial-state laws), writes the dimensions the
    dimension checker resolved onto the generated quantities, and applies
    {!Constant_fold} and then {!Licm}.

    {!compile}, {!compile_with_reads} and {!compile_with_declared} run all
    three; [camdlc] calls the last. They render every diagnostic to standard
    error as a side effect, warnings included on success, and return [Error]
    with the string {!Diagnostics.render} returned: ["compilation failed"], or
    the JSON array under [--json-errors]. [camdlc] recognises those two
    shapes to avoid printing the failure twice. {!collect_detail},
    {!collect_diagnostics} and {!compile_outcome} run the front end and the
    analysis without rendering and return the diagnostics as values; [camdlc
    check], [camdlc doctest] and the tests use them. Because both kinds of
    caller go through [run_analysis], a check added there runs in both. None
    of these functions raises on a bad model.

    Five process-wide switches change what a compile does. {!no_dim_check},
    set by [--no-dim-check], skips the dimension checker, so it reports
    nothing and resolves no quantity dimensions. {!no_state_grad}, set by
    [--no-state-grad], leaves out the derivatives of rates and projections
    with respect to compartments. {!constant_fold}, which only tests set,
    turns constant folding off, as does the environment variable
    [CAMDL_NO_CONSTANT_FOLD]; [CAMDL_NO_LICM] turns off loop-invariant code
    motion. {!Diagnostics.json_errors_mode} selects JSON rendering. The two
    environment variables are read at each compile; the references hold for
    the rest of the process. *)

(** One model after the front end: the expanded model together with what
    produced it, which the later stages need to report against the source. *)
type compile_detail = {
  model   : Ir.model;
      (** The expanded model as {!Expander.expand_detail} returned it: names
          resolved, no gradients, not folded. *)
  ctx     : Expander.context;
      (** The expander's working context, holding the declarations, the
          diagnostics collected so far, and the lookups that map an expanded
          name back to its declaration's location. *)
  summary : Expander.model_summary;
      (** Counts of declared and expanded constructs, for [camdlc inspect]. *)
  source  : Source_cache.t;
      (** The text of every file of the compile, for rendering. *)
  decls   : Ast.declaration list;
      (** The declarations as parsed, after the [--quantities] splice. The
          declared-model section is built from these and never from [ctx],
          whose transitions and compartments the [via] lowering has
          rewritten. *)
  declared : (Declared.declared_model * Declared_emit.unresolved list) Lazy.t;
      (** The declared-model section, and the names in it that could not be
          classified. Built at most once: [run_analysis] forces it to report
          W203, and {!compile_with_declared} returns the same value. *)
}

(** The result of a compile that does not render: every diagnostic, and a
    value only when none of them is an error. {!compile_outcome} returns one
    whose value is the {!compile_detail} after the analysis, not the finished
    model: its transitions carry no gradients. *)
type 'a outcome = {
  value       : 'a option;
      (** [Some] exactly when no diagnostic has severity [Error]. *)
  diagnostics : Diagnostics.diagnostic list;
      (** Every diagnostic, in the order it was reported. *)
  source      : Source_cache.t;
      (** The text of the compile's files, for rendering the diagnostics. *)
}

(** Read the entire contents of a file into a string. *)
let read_file path = In_channel.with_open_bin path In_channel.input_all

(* ── Front-end core ───────────────────────────────────────────────────────

   The single, non-aborting lex/parse/expand front end. It runs the
   pipeline once, accumulating every front-end diagnostic — the E001 on a
   lex/parse/expand failure, the W100 lex warnings, and the parser-action
   errors — into the returned [Diagnostics.t], and never renders or
   aborts. Both consumers build on it:

   - [compile_detail_result] (the production path) wraps it: on a
     front-end error it renders the diagnostics and returns [Error]; on
     a clean expand it returns [Ok].
   - [collect_diagnostics] (test/tooling) uses it directly and continues
     the downstream pipeline.

   Return shape: [(detail, diags, source)]. [detail] is [None] when
   lex/parse/expand structurally failed (then [diags] holds the E001, and
   the W100 / parser-action diagnostics recorded before the failure);
   [Some d] when expansion produced a model (then [diags] is [d.ctx.diags],
   which may itself carry expansion-phase errors/warnings and the drained
   W100 / parser-action diagnostics — the caller decides whether to
   continue). [source] is the [Source_cache] for the input, returned even
   on the [None] path so a renderer can show the offending line. *)
(* Run [body]; on an unexpected [Failure]/exception, record a no-location E001
   into [diags] and return [Error ()]. The parse and expand phases share this
   outer guard (parse's inner located lex/parse errors stay inside [body]). *)
let capture_e001 (diags : Diagnostics.t) (body : unit -> ('a, unit) result)
    : ('a, unit) result =
  try body ()
  with
  | Failure msg ->
    Diagnostics.error diags ~code:"E001" ~loc:Diagnostics.no_loc ~message:msg ();
    Error ()
  | exn ->
    Diagnostics.error diags ~code:"E001" ~loc:Diagnostics.no_loc
      ~message:(Printexc.to_string exn) ();
    Error ()

(* camdl writes durations with a leading tick (`4 'weeks`); the CLI's
   `--to "last_obs + 8 weeks"` writes them as bare words, because a tick is a
   shell-quoting hazard. The asymmetry is deliberate, so it must be
   self-correcting on both sides — the CLI already rejects the tick form by
   name. On this side the bare word is only ever a parse failure, and where it
   fails is not where it reads wrong: inside a `{ key = value }` block,
   `to = 80 days` parses `80` as the value and `days` as the next key, so the
   parser does not complain until the closing brace, pointing at a token the
   author has no reason to suspect.

   Run only on the failure path: re-lex and look for a duration word directly
   after a numeric literal, at or before the position the parse actually failed
   (the parser cannot have failed before consuming it, so a later match belongs
   to an unrelated error and is ignored). Returns the offending word and its
   span. *)
let bare_duration_word_before (src : string) ~(before : Lexing.position)
    : (string * Lexing.position * Lexing.position) option =
  let lexbuf = Lexing.from_string src in
  let rec scan prev_was_number =
    match Lexer.token lexbuf with
    | exception _ -> None
    | Parser.EOF -> None
    | Parser.IDENT w
      when prev_was_number
        && List.mem w ["day"; "days"; "week"; "weeks";
                       "month"; "months"; "year"; "years"] ->
      let sp = lexbuf.Lexing.lex_start_p and ep = lexbuf.Lexing.lex_curr_p in
      if sp.Lexing.pos_cnum <= before.Lexing.pos_cnum then Some (w, sp, ep)
      else None
    | Parser.INT _ | Parser.FLOAT _ -> scan true
    | _ -> scan false
  in
  scan false

(* Parse one compilation unit into its declaration list, recording a located
   lex/parse failure as E001/E115 against [parse_diags] under that unit's own
   filename. A compile can have two units: the model, and the `--quantities`
   vocabulary file. *)
let parse_unit (parse_diags : Diagnostics.t) ~(filename : string) (src : string)
    : (Ast.declaration list, unit) result =
  let lexbuf = Lexing.from_string src in
  Lexing.set_filename lexbuf filename;
  (* Offsets are per unit; a stale entry from another unit could hide E107. *)
  Hashtbl.reset Parser_errors.paren_starts;
  try Ok (Parser.file Lexer.token lexbuf)
  with
  | Lexer.LexError msg ->
    let pos = lexbuf.Lexing.lex_curr_p in
    Diagnostics.error parse_diags ~code:"E001"
      ~loc:(Diagnostics.loc_of_positions ~file:filename pos pos)
      ~message:(Printf.sprintf "lex error: %s" msg) ();
    Error ()
  | Parser.Error ->
    let pos = lexbuf.Lexing.lex_curr_p in
    (match bare_duration_word_before src ~before:pos with
     | Some (w, sp, ep) ->
       let plural =
         if w.[String.length w - 1] = 's' then w else w ^ "s" in
       Diagnostics.error parse_diags ~code:"E115"
         ~loc:(Diagnostics.loc_of_positions ~file:filename sp ep)
         ~message:(Printf.sprintf
           "`%s` is not a unit here: camdl writes durations with a \
            leading tick" w)
         ~hint:(Printf.sprintf
           "write `'%s` — e.g. `to = 80 '%s`, `last_obs + 4 '%s`. \
            (The CLI's `--to \"last_obs + 8 weeks\"` takes bare words \
            instead, because a tick is a shell-quoting hazard.)"
           plural plural plural)
         ()
     | None ->
       Diagnostics.error parse_diags ~code:"E001"
         ~loc:(Diagnostics.loc_of_positions ~file:filename pos pos)
         ~message:"syntax error" ());
    Error ()

(* The English name of a declaration kind, for the E339 message. *)
let decl_kind_name : Ast.declaration -> string = function
  | DTimeUnit _     -> "a `time_unit`"
  | DDescription _  -> "a `description`"
  | DModelDoc _     -> "a model-level `#'` doc block"
  | DOrigin _       -> "an `origin`"
  | DDimensions _   -> "a `dimensions { }` block"
  | DCompartments _ -> "a `compartments { }` block"
  | DParameters _   -> "a `parameters { }` block"
  | DTables _       -> "a `tables { }` block"
  | DForcing _      -> "a `forcing { }` block"
  | DTransitions _  -> "a `transitions { }` block"
  | DObservations _ -> "an `observations { }` block"
  | DInterventions _ -> "an `interventions { }` block"
  | DEvents _       -> "an `events { }` block"
  | DReactiveInterventions _ -> "a `reactive { }` block"
  | DODE _          -> "an `ode { }` block"
  | DOutput _       -> "an `output { }` block"
  | DSimulate _     -> "a `simulate { }` block"
  | DInit _         -> "an `init { }` block"
  | DTimepoints _   -> "a `timepoints { }` block"
  | DStratify _     -> "a `stratify(...)`"
  | DLet _          -> "a top-level `let`"
  | DScenarios _    -> "a `scenarios { }` block"
  | DBalance _      -> "a `balance { }` block"
  | DQuantities _   -> "a `quantities { }` block"
  | DContrasts _    -> "a `contrasts { }` block"

(* Splice a vocabulary file's `quantities { }` into the model's declaration
   list, replacing the model's own block wholesale. Replacement, not merge: a
   merge rule is a silent-precedence surface, and "swap the reporting
   vocabulary" is what the flag means.

   The vocabulary file is a compilation unit of its own, so its declarations are
   stamped with its path before the splice — a diagnostic about a name the model
   does not declare must point at the vocabulary, or the author cannot tell
   which of the two files is wrong. *)
let splice_quantities (diags : Diagnostics.t) ~(file : string)
    (model_decls : Ast.declaration list) (unit_decls : Ast.declaration list)
    : (Ast.declaration list, unit) result =
  let file_loc = { Diagnostics.no_loc with Diagnostics.file = file } in
  match List.find_opt (function Ast.DQuantities _ -> false | _ -> true) unit_decls with
  | Some other ->
    Diagnostics.error diags ~code:"E339" ~loc:file_loc
      ~message:(Printf.sprintf
        "the quantities file '%s' also declares %s; a file supplied to \
         --quantities may contain only a `quantities { }` block"
        file (decl_kind_name other))
      (* A file-header `#'` block reaches here too, and the generic
         "move the declarations back" hint would be wrong advice for it: the
         author wrote a description, not a declaration. Point at the fix that
         applies — the model's header documents the model; a vocabulary file
         describes itself in an ordinary comment. *)
      ~hint:(match other with
        | Ast.DModelDoc _ ->
          "a `#'` block at the top of a file documents the MODEL, and a \
           vocabulary file is not one — describe the vocabulary with an \
           ordinary `#` comment, or move the prose to the model's own header"
        | _ ->
          "a quantities file is a reporting vocabulary applied to a model \
           that already exists — move the model declarations back into the \
           model file")
      ();
    Error ()
  | None ->
    let qds =
      List.concat_map (function Ast.DQuantities qs -> qs | _ -> []) unit_decls in
    if qds = [] then begin
      Diagnostics.error diags ~code:"E340" ~loc:file_loc
        ~message:(Printf.sprintf
          "the quantities file '%s' declares no quantities" file)
        ~hint:"--quantities REPLACES the model's own `quantities { }` block, so \
               an empty vocabulary would report nothing; drop the flag to use \
               the model's own block"
        ();
      Error ()
    end else
      let qds = List.map (Ast.stamp_quantity_decl_file ~file) qds in
      Ok (List.filter (function Ast.DQuantities _ -> false | _ -> true) model_decls
          @ [Ast.DQuantities qds])

(* [quantities] is an optional second compilation unit — (filename, source) —
   whose `quantities { }` block replaces the model's own. See
   [splice_quantities]. *)
let front_end_collect ?(name = "model") ?(filename = "<input>")
    ?(quantities : (string * string) option) (src : string)
    : compile_detail option * Diagnostics.t * Source_cache.t =
  let source = Source_cache.of_string ~filename src in
  let source =
    match quantities with
    | Some (qfile, qsrc) -> Source_cache.add_unit source ~filename:qfile qsrc
    | None -> source
  in
  (* Drain any stale lex-phase warnings from a previous compilation in the
     same process. pending_warnings is a mutable global ref; clearing it
     here ensures we never replay warnings from a prior run. *)
  Lexer.pending_warnings := [];
  Parser_errors.pending_errors := [];
  let parse_diags = Diagnostics.create () in
  (* Move what the lexer and the parser actions recorded into [diags]: the
     W100 lex warnings, and the errors parser actions reported while parsing
     went on (see [Parser_errors]). Every return path drains, so a later
     syntax error or an expansion failure does not hide the errors reported
     before it (gh#1210).

     A position is attributed to the unit it was lexed from: with a
     `--quantities` vocabulary file there are two units in one compile, and
     [Lexing.set_filename] stamped each position with its own path. A
     position with no stamp falls back to the model filename. *)
  let drain_pending (diags : Diagnostics.t) =
    let unit_file (sp : Lexing.position) =
      if sp.Lexing.pos_fname = "" then filename else sp.Lexing.pos_fname in
    List.iter (fun (sp, ep, msg) ->
      Diagnostics.warning diags ~code:"W100"
        ~loc:(Diagnostics.loc_of_positions ~file:(unit_file sp) sp ep)
        ~message:msg ()
    ) (List.rev !Lexer.pending_warnings);
    Lexer.pending_warnings := [];
    List.iter (fun (sp, ep, code, msg, hint, fix) ->
      let fixes = match fix with
        | None -> []
        | Some (fsp, fep, replacement) ->
          [ { Diagnostics.fix_loc =
                Diagnostics.loc_of_positions ~file:(unit_file fsp) fsp fep;
              replacement } ] in
      Diagnostics.error diags ~code
        ~loc:(Diagnostics.loc_of_positions ~file:(unit_file sp) sp ep)
        ~message:msg ?hint ~fixes ()
    ) (List.rev !Parser_errors.pending_errors);
    Parser_errors.pending_errors := []
  in
  match
    capture_e001 parse_diags (fun () ->
       let t_parse = Sys.time () in
       let decls =
         match parse_unit parse_diags ~filename src, quantities with
         | Error (), _ -> Error ()
         | Ok decls, None -> Ok decls
         | Ok decls, Some (qfile, qsrc) ->
           (match parse_unit parse_diags ~filename:qfile qsrc with
            | Error () -> Error ()
            | Ok qdecls -> splice_quantities parse_diags ~file:qfile decls qdecls)
       in
       Passtime.record "parse" (Sys.time () -. t_parse);
       decls)
  with
  | Error () -> drain_pending parse_diags; (None, parse_diags, source)
  | Ok decls ->
    let source_dir =
      if filename = "<input>" then "" else Filename.dirname filename
    in
    (match
       capture_e001 parse_diags (fun () ->
          Ok (Passtime.time "expand"
                (fun () -> Expander.expand_detail ~source_dir ~filename name decls)))
     with
     | Error () -> drain_pending parse_diags; (None, parse_diags, source)
     | Ok (model, ctx, summary) ->
       drain_pending ctx.diags;
       (Some { model; ctx; summary; source; decls;
               declared = lazy (Declared_emit.build_detail ~source ctx decls) },
        ctx.diags, source))

(** Production front end: run [front_end_collect], and on any front-end
    error (lex/parse/expand failure, or a parser-action error drained
    into [ctx.diags]) render the diagnostics and return [Error]; on a
    clean expand return [Ok]. The [Error] payload is the rendered string
    from [Diagnostics.render] — the serialized JSON array under
    [--json-errors], else ["compilation failed"]. CLI entry points
    recognize the payload shape and exit without re-printing a redundant
    Error line. Warnings are not rendered here: callers render once at the
    end of their pipeline, so expansion-phase warnings do not print twice
    when downstream passes also emit diagnostics. *)
let compile_detail_result ?(name = "model") ?(filename = "<input>")
    ?(quantities : (string * string) option) (src : string)
    : (compile_detail, string) result =
  let (detail, diags, source) = front_end_collect ~name ~filename ?quantities src in
  match detail with
  | None ->
    (* Front-end failure (lex/parse/expand): [front_end_collect] captured it
       as an E001 in [diags]. Render to stderr and return the payload string
       as [Error]. A [Failure] raised inside the expander (e.g. a malformed
       date literal) also surfaces here as a rendered E001. *)
    Error (Diagnostics.render diags source)
  | Some d ->
    (* Expansion produced a model; [d.ctx.diags] may carry a drained
       parser-action error. Render and return [Error] on any error, else
       [Ok]. *)
    if Diagnostics.has_errors d.ctx.diags then
      Error (Diagnostics.render d.ctx.diags d.source)
    else
      Ok d

(** Skip the dimension checker. Off unless [--no-dim-check] is given, to
    [camdlc] or [camdlc check]. With it set, the checker reports nothing and
    no generated quantity gets a resolved dimension. *)
let no_dim_check = ref false

(** Leave out the state Jacobian: the derivatives of each transition rate
    ([rate_state_grad]) and of each nonlinear observation projection
    ([projection_state_grad]) with respect to every compartment. Off unless
    [--no-state-grad] is given; the Rust runtime passes that flag on every
    compile except one for an [nl-lbfgs] fit on the ODE backend, the only
    consumer of these maps. On a model with mean-field coupling across many
    strata the maps hold one entry per stratum per coupled transition and
    make up most of the IR, so leaving them out is what keeps such a model
    compilable. The derivatives with respect to parameters ([rate_grad],
    [ic_grad]) are unaffected. {!collect_detail} sets it for its own
    duration, because no diagnostic depends on these maps. *)
let no_state_grad = ref false

(** Run {!Constant_fold}. On by default; tests that inspect the unfolded IR
    set it to [false] for their duration. [CAMDL_NO_CONSTANT_FOLD], set to
    anything, also turns the pass off. *)
let constant_fold = ref true

(** Translate a [Validate.error] into an E5xx diagnostic. The E2xx range
    covers the expander's duplicates and unknowns; [Validate.validate] runs on
    the expanded model and catches what the expander missed, or an IR that
    did not come from it, so its codes have a range of their own. Most of
    them are a compiler invariant on a model compiled from source. *)
let diagnose_validate_error ctx (err : Validate.error) : Diagnostics.diagnostic =
  let open Validate in
  (* Decl-keyed errors map their named symbol back to its declaration's source
     loc (prefix-matched through stratification). Reference errors (E503–E507)
     name a symbol that does NOT exist, so they point at the enclosing
     construct (transition / ODE compartment / observation) via [site_loc]. *)
  let comp_loc = Expander.compartment_loc ctx in
  let tr_loc   = Expander.transition_loc  ctx in
  let par_loc  = Expander.param_loc       ctx in
  let obs_loc  = Expander.obs_loc         ctx in
  (* A reference error's loc is its enclosing construct's declaration. *)
  let interv_loc = Expander.interv_loc ctx in
  let site_loc = function
    | Validate.InTransition   n -> tr_loc n
    | Validate.InOde          c -> comp_loc c
    | Validate.InObservation  n -> obs_loc n
    | Validate.InIntervention n -> interv_loc n
  in
  let (code, message, hint, loc) = match err with
    | DuplicateCompartment s ->
      "E500",
      Printf.sprintf "duplicate compartment after expansion: '%s'" s,
      Some "stratification produced two compartments with the same name",
      comp_loc s
    | DuplicateTransition s ->
      "E501",
      Printf.sprintf "duplicate transition after expansion: '%s'" s,
      Some "stratification produced two transitions with the same name",
      tr_loc s
    | DuplicateParameter s ->
      "E502",
      Printf.sprintf "duplicate parameter: '%s'" s,
      Some "two `parameters` entries (or a stratified family) share this name; \
            rename or remove one",
      par_loc s
    | UnknownCompartment (s, site) ->
      "E503",
      Printf.sprintf "unknown compartment referenced: '%s'" s,
      Some "check stratification / spelling against the compartments block",
      site_loc site
    | UnknownParameter (s, site) ->
      "E504",
      Printf.sprintf "unknown parameter referenced: '%s'" s,
      Some "check the parameters block for a matching declaration",
      site_loc site
    | UnknownConstant (s, site) ->
      "E518",
      Printf.sprintf "unknown constant referenced: '%s'" s,
      Some "a constant reference names no entry of the model's `constants`; \
            this is a compiler invariant, so report it",
      site_loc site
    | UnknownTable (s, site) ->
      "E505",
      Printf.sprintf "unknown table referenced: '%s'" s,
      Some "check the `tables` block for a matching declaration",
      site_loc site
    | UnknownTimeFunction (s, site) ->
      "E506",
      Printf.sprintf "unknown time_function referenced: '%s'" s,
      Some "check the `time_functions` block for a matching declaration",
      site_loc site
    | ForcingBasisOutsideGradient (s, site) ->
      (* No DSL spelling reaches this: autodiff writes the node only into
         gradient maps. A hit is a compiler bug or hand-written IR. *)
      "E519",
      Printf.sprintf "a basis function of forcing '%s' appears outside a gradient" s,
      Some "forcing_basis is a derivative node; only the compiler's gradient entries may hold it",
      site_loc site
    | DuplicateFlowInUnion (s, site) ->
      (* The expander's lowering-site check (E342) catches this on the
         .camdl path with a located span and a binder-aware hint; E514 is the
         backstop for an IR that never passed through it — hand-written,
         generated, or drifted. *)
      "E514",
      Printf.sprintf
        "flow '%s' appears twice in one observation's flow union, so every \
         event on it would be counted twice" s,
      Some "a union of flows must be disjoint. From a .camdl this is E342 with \
            a source span; a bare E514 means the IR was hand-written or has \
            drifted",
      site_loc site
    | UnknownTransition (s, site) ->
      "E507",
      Printf.sprintf "unknown transition referenced in observation: '%s'" s,
      Some "check the transition name against the `transitions` block; \
            stratified transitions expand to `<base>_<stratum>`",
      site_loc site
    | RealCompartmentInStoichiometry (tr, c) ->
      "E508",
      Printf.sprintf "real-valued compartment '%s' cannot appear in \
                      stoichiometry of transition '%s'" c tr,
      Some "real compartments have continuous dynamics (ODE); mixing them \
            into transition stoichiometry is ill-defined",
      tr_loc tr
    | MissingOdeEquation s ->
      "E509",
      Printf.sprintf "real-valued compartment '%s' has no ODE equation" s,
      Some "add an `ode { ... }` block with dX/dt for this compartment",
      comp_loc s
    | OdeForNonRealComp s ->
      "E510",
      Printf.sprintf "ODE equation for '%s', which is not a real-valued \
                      compartment" s,
      Some "only compartments declared `: real` can have ODE equations",
      comp_loc s
    | ZeroDelta (tr, c) ->
      "E511",
      Printf.sprintf "transition '%s' has zero delta for compartment '%s'" tr c,
      Some "a zero-delta stoichiometry entry has no effect; remove it",
      tr_loc tr
    | ParamInBinding (b, p) ->
      "E512",
      Printf.sprintf "hoisted binding '%s' references parameter '%s'" b p,
      Some "shared bindings must be param-free: the gradient pass \
            differentiates a binding reference to 0, so a parameter inside \
            one would be silently frozen during inference (zero gradient). \
            This is a compiler invariant — please file a bug.",
      (* Bindings are synthesized post-expansion with no source span. *)
      Diagnostics.no_loc
    | InitUnknownCompartment s ->
      "E513",
      Printf.sprintf "initial condition '%s' names a compartment that does \
                      not exist in the expanded model" s,
      Some "init keys must be real (expanded) compartment cells; the \
            frontend reports this with a located E277 — a bare E513 here \
            means the IR was hand-written or has drifted",
      (* The IR carries no per-init-entry source span. *)
      Diagnostics.no_loc
    | InitLawKindMismatch (comp, family, seeds) ->
      (* The frontend reports this with a located E344; a bare E516 here means
         the IR was hand-written or has drifted. *)
      "E516",
      Printf.sprintf
        "initial condition '%s' is drawn from `%s`, which seeds %s; '%s' is not one"
        comp family seeds comp,
      Some "a count law (poisson, binomial, neg_binomial) seeds an integer \
            compartment and a continuous law (normal) seeds a real one",
      Diagnostics.no_loc
    | InitLawOnBalanceTarget comp ->
      "E517",
      Printf.sprintf
        "initial condition '%s' is drawn from a law, but '%s' is the `balance { }` \
         target: the balance expression overwrites it after every substep, so the \
         draw is discarded" comp comp,
      Some "draw one of the other compartments and let `balance` absorb it. The \
            frontend reports this with a located E345 — a bare E517 here means the \
            IR was hand-written or has drifted",
      Diagnostics.no_loc
    | InitDependencyCycle cyc ->
      (* An init entry may read another compartment's initial value, so
         the entries are evaluated in dependency order. A cycle has no such
         order — every candidate evaluation reads a compartment that has not
         been built yet. Name the whole cycle: the offending entry is rarely the
         one the author is looking at. *)
      "E515",
      Printf.sprintf
        "initial conditions reference each other in a cycle: %s"
        (String.concat " -> " (cyc @ [List.hd cyc])),
      Some "an initial condition may read another compartment's initial value, \
            but the references must form a chain, not a loop. Break the cycle \
            by writing one of these entries in terms of parameters only",
      (* The IR carries no per-init-entry source span. *)
      Diagnostics.no_loc
  in
  Diagnostics.mk_error ~code ~loc ~message ?hint ()

(** Run the structural validation of the expanded model.

    It runs before the dimension checker, which fails on an unknown
    parameter, so that such a model gets "unknown parameter 'foo'" rather
    than an internal error. *)
let run_validate (d : compile_detail) : Diagnostics.diagnostic list =
  match Validate.validate d.model with
  | Ok () -> []
  | Error errs -> List.map (diagnose_validate_error d.ctx) errs

(** Run the dimension checker, returning its diagnostics with source
    locations and the dimension it resolved for each generated quantity cell.
    [finish_compile] writes the dimensions onto the model's quantities: the
    checker is the one authority on dimensions, and the stored field is a
    copy for the runtime's contrast reducer. *)
let run_dimcheck_full (d : compile_detail)
    : Diagnostics.diagnostic list
      * ((string * (string * string) list) * (int * int)) list =
  if !no_dim_check then ([], [])
  else
    (* dimcheck runs on the IR (no source spans); it tags each diagnostic with
       the construct it concerns, which we resolve to the declaration's loc. *)
    let loc_of_subject = function
      | Some (Dimcheck.STransition n)  -> Expander.transition_loc  d.ctx n
      | Some (Dimcheck.SOde c)         -> Expander.compartment_loc d.ctx c
      | Some (Dimcheck.SObservation n) -> Expander.obs_loc         d.ctx n
      | Some (Dimcheck.SContrast n)    -> Expander.contrast_loc    d.ctx n
      | Some (Dimcheck.SInit c)        -> Expander.compartment_loc d.ctx c
      | Some (Dimcheck.SParameter p)   -> Expander.param_loc       d.ctx p
      | Some (Dimcheck.SQuantity q)    -> Expander.quantity_loc    d.ctx q
      | Some (Dimcheck.SIntervention i) -> Expander.interv_loc     d.ctx i
      | Some (Dimcheck.SLet l)         -> Expander.let_loc         d.ctx l
      | Some (Dimcheck.SBinding _) | None -> Diagnostics.no_loc
    in
    let result =
      Dimcheck.check_model ~prior_args:(Expander.prior_sources d.ctx)
        ~constant_dims:(Expander.constant_dims d.ctx)
        ~table_dims:(Expander.table_unit_dims d.ctx)
        ~typed_lets:(Expander.typed_let_uses d.ctx) d.model in
    let diags =
      List.map (fun (dc : Dimcheck.diagnostic) ->
        let loc = loc_of_subject dc.subject in
        match dc.severity with
        | Dimcheck.Error ->
          Diagnostics.mk_error
            ~code:dc.code ~loc
            ~message:dc.message ?detail:dc.detail ?hint:dc.hint ()
        | Dimcheck.Info ->
          Diagnostics.mk_info
            ~code:dc.code ~loc
            ~message:dc.message ?detail:dc.detail ?hint:dc.hint ()
      ) result.diagnostics
    in
    let qdims =
      List.map (fun ((name, stratum), dv) -> ((name, stratum), (dv.(0), dv.(1))))
        result.quantity_dims
    in
    (diags, qdims)

(* Write each resolved quantity dimension onto its IR node by (name,
   stratum). Quantities without a resolved dimension keep `None` (omitted on the
   wire), so only quantity-bearing models gain the field. *)
let annotate_quantity_dims
    (qdims : ((string * (string * string) list) * (int * int)) list)
    (m : Ir.model) : Ir.model =
  if qdims = [] then m
  else
    let quantities =
      List.map (fun (q : Ir.quantity) ->
        match List.assoc_opt (q.Ir.q_name, q.Ir.q_stratum) qdims with
        | Some dim -> { q with Ir.q_dimension = Some dim }
        | None -> q)
        m.Ir.quantities
    in
    { m with Ir.quantities = quantities }

(** Run the model linter on a compiled model and route its results into
    the diagnostic context as non-blocking warnings. Lints (L4xx) flag
    semantically valid but discouraged patterns (e.g. L402 dead
    compartment, L404 two streams projecting one quantity); they render
    with hint text but never set [has_errors], so the build does not fail
    on a lint. Each lint tags itself with the declaration it concerns
    ([Lint.subject]), which we resolve to that declaration's source loc.
    Runs right after dimcheck in [run_analysis], so both `camdlc compile`
    and `camdlc check` run it. *)
let run_lint (d : compile_detail) : Diagnostics.diagnostic list =
  List.map (fun (l : Lint.diagnostic) ->
    let loc = match l.subject with
      | Some (Lint.SCompartment c) -> Expander.compartment_loc d.ctx c
      | Some (Lint.SObservation n) -> Expander.obs_loc         d.ctx n
      | None                       -> Diagnostics.no_loc
    in
    match l.severity with
    | Lint.Warning ->
      Diagnostics.mk_warning
        ~code:l.code ~loc
        ~message:l.message ?detail:l.detail ?hint:l.hint ()
  ) (Lint.check_model d.model).diagnostics

(** Autodiff pass: differentiate every transition rate w.r.t. all
    parameters, returning the transition list with [rate_grad] filled in,
    paired with any diagnostics produced. A non-differentiable construct
    that a parameter cannot legitimately drive — `mod` over a parameter,
    structural forcing data (interpolation knots / piecewise steps / a
    spline's knots, window or period), or a non-constant table lookup index — yields an E600 (with
    source location) and leaves that transition's [rate_grad] empty.
    Coefficients whose gradient is not emitted (a periodic step value, an
    inline-table value via a non-constant index) are not errors: the
    parameter is omitted from [rate_grad], and the runtime refuses a
    gradient-based fit that depends on it. Pure: it neither emits into a
    context nor renders, so [compile] (which short-circuits on the resulting
    errors) and [collect_diagnostics] (which does not) share it. *)
let differentiate_transitions (d : compile_detail)
    : Ir.transition list * Diagnostics.diagnostic list =
  let param_names = List.map (fun (p : Ir.parameter) -> p.name) d.model.Ir.parameters in
  (* ∂rate/∂compartment (rate_state_grad, the J_x ingredient for the ODE
     forward sensitivities). Computed alongside rate_grad from the same rate, over
     the model's compartments; a hoisted binding body is state-bearing under WrtPop,
     so the binding list is threaded. Unlike rate_grad it never errors (E600): a
     live-but-omitted or nonsmooth-of-state coefficient becomes a serialized
     DEUnsupported the fit-time gradient gate refuses on — the model still forward-
     sims / IF2 / PF. *)
  let comp_names = List.map (fun (c : Ir.compartment) -> c.name) d.model.Ir.compartments in
  let bindings = d.model.Ir.bindings in
  let tr_loc name =
    (* Find the original (pre-expansion) transition declaration by prefix
       match: expanded name "infection_child" → base "infection". *)
    match List.find_opt (fun (td : Ast.transition_decl) ->
      Expander.is_expansion_of ~base:td.trname name
    ) d.ctx.orig_transitions with
    | Some td -> Expander.diag_loc_of_ast_ctx d.ctx td.trloc
    | None -> Diagnostics.no_loc
  in
  let diags = ref [] in
  let transitions =
    Passtime.time "autodiff" (fun () ->
      List.map (fun (t : Ir.transition) ->
        let rate_state_grad =
          if !no_state_grad then []
          else Autodiff.differentiate_rate_state t.rate comp_names
                 d.model.Ir.time_functions d.model.Ir.tables bindings
        in
        match Autodiff.differentiate_rate t.rate param_names
                d.model.Ir.time_functions d.model.Ir.tables with
        | Ok rate_grad -> { t with Ir.rate_grad; Ir.rate_state_grad }
        | Error msg ->
          diags := Diagnostics.mk_error
                     ~code:"E600"
                     ~loc:(tr_loc t.name)
                     ~message:(Printf.sprintf "transition '%s': %s" t.name msg)
                     ~hint:"reparameterize as the message describes — a \
                            parameter cannot drive structural forcing data or a \
                            non-constant lookup index; see \
                            `camdl docs language-changes`"
                     () :: !diags;
          { t with Ir.rate_grad = []; Ir.rate_state_grad }
      ) d.model.Ir.transitions)
  in
  (transitions, List.rev !diags)

(** Sparse-coupling constant-fold (on by default): resolves
    constant-indexed inline-table lookups and drops zero-W terms from FOI
    Reduce sums, collapsing the dense P-term spatial sum to its k nonzero
    terms. A Rust test, [gate_constant_fold_ab], checks that it does not
    change simulated output. Set CAMDL_NO_CONSTANT_FOLD to emit the
    unfolded (dense) IR, for debugging the pass or inspecting the pre-fold
    shape. *)
let maybe_constant_fold (m : Ir.model) : Ir.model =
  let fold_on = !constant_fold && Sys.getenv_opt "CAMDL_NO_CONSTANT_FOLD" = None in
  if fold_on then Passtime.time "constant_fold" (fun () -> Constant_fold.fold_model m)
  else m

(* Loop-invariant code motion. On by default; CAMDL_NO_LICM forces it off
   (debugging / A-B comparison), mirroring constant_fold / CAMDL_NO_CONSTANT_FOLD.
   It is value-preserving (checked by the Rust test gate_licm_ab), so it
   changes speed, never results. Runs AFTER constant_fold (so it hoists already-folded
   subtrees; constant_fold never sees a PerEvalRef) and is the last transform
   before serialization, so the pre-LICM passes (autodiff/dimcheck/validate/lint)
   never encounter the node. *)
let maybe_licm (m : Ir.model) : Ir.model =
  if Sys.getenv_opt "CAMDL_NO_LICM" <> None
  then m
  else Passtime.time "licm" (fun () -> Licm.licm_model m)

(* ── The post-expansion analysis pipeline ────────────────────────────────────

   [run_analysis] is the single definition of "the post-expansion pipeline":
   validate → dimcheck → lint → autodiff-transitions, in that order, with the
   short-circuit structure [compile] and `check` must share. Each stage emits
   its diagnostics into [d.ctx.diags] (the accumulator [has_errors]/[render]
   read); the sequence stops after Validate (dimcheck fails on unknown params,
   so a structural error must halt first), and after dimcheck+lint / autodiff
   on any Error-severity diagnostic.

   Returns [Some (qdims, transitions)] on the all-clear path — the resolved
   quantity dimensions and the gradient-annotated transitions [compile]
   needs to finish building the model — or [None] if a stage short-circuited on
   an error. [compile] ([finish_compile]) builds a model from the [Some];
   `check`/[collect_detail] discards it and keeps only the accumulated
   diagnostics. One place defines the stage sequence, so a stage added here is
   added to both paths at once and they cannot drift.

   The passes are pure (each returns a diagnostic list rather than emitting or
   raising); [run_analysis] emits them, so a late-phase error (validate E5xx,
   dimcheck, autodiff E600) becomes a [None] the caller renders. *)
(* W203: a name the declared-model section could not classify. The compile
   succeeds, since the compiled model does not depend on the section; but the
   section then says "unresolved" where it should say what the name is, and
   `camdl diff` cannot follow that declaration's dependencies. Reported here,
   on the no-error path, so `compile` and `check` both show it. *)
let run_declared_names (d : compile_detail) : Diagnostics.diagnostic list =
  let (_section, unresolved) = Lazy.force d.declared in
  List.map (fun (u : Declared_emit.unresolved) ->
    Diagnostics.mk_warning ~code:"W203"
      ~loc:(Expander.diag_loc_of_ast_ctx d.ctx u.u_loc)
      ~message:(Printf.sprintf
        "the declared-model section could not classify the name '%s' in %s"
        u.u_name u.u_site)
      ~detail:(Printf.sprintf
        "The model compiled, and the compiled model is unaffected. The record \
         of the model as written (the `declared` section of the compiled file) \
         tags '%s' as unresolved, so `camdl diff` will report the dependencies \
         of %s as unknown." u.u_name u.u_site)
      ~hint:"this is a gap in camdl's record of the model, not an error in the \
             model; please report it with the model file"
      ()) unresolved

let run_analysis (d : compile_detail)
    : (((string * (string * string) list) * (int * int)) list
       * Ir.transition list) option =
  let emit_all = List.iter (Diagnostics.emit d.ctx.diags) in
  let vdiags = Passtime.time "validate" (fun () -> run_validate d) in
  emit_all vdiags;
  (* Validate short-circuits before dimcheck (dimcheck ICEs on unknown
     params), matching the original short-circuit ordering. *)
  if vdiags <> [] then None
  else begin
    let (ddiags, qdims) = Passtime.time "dimcheck" (fun () -> run_dimcheck_full d) in
    emit_all ddiags;
    emit_all (Passtime.time "lint" (fun () -> run_lint d));
    if Diagnostics.has_errors d.ctx.diags then None
    else begin
      let (transitions, gdiags) = differentiate_transitions d in
      emit_all gdiags;
      if Diagnostics.has_errors d.ctx.diags then None
      else begin
        emit_all (Passtime.time "declared-names" (fun () -> run_declared_names d));
        Some (qdims, transitions)
      end
    end
  end

(* The post-expansion pipeline, shared by [compile], [compile_with_reads] and
   [compile_with_declared], which therefore produce the same model.

   [run_analysis] runs the shared stage sequence (validate → dimcheck → lint →
   autodiff-transitions), emitting into [d.ctx.diags]. On [None] a stage
   short-circuited on an error, so render + [Error] (the CLI exits 1). On [Some (qdims,
   transitions)] every stage cleared, so finish building the model: the obs/σ²
   autodiff tail, the quantity-dim write-back, and the value-preserving
   constant-fold/LICM transforms. *)
let finish_compile (d : compile_detail) : (Ir.model, string) result =
  match run_analysis d with
  | None -> Error (Diagnostics.render d.ctx.diags d.source)
  | Some (qdims, transitions) ->
    (* Single render of any collected non-blocking diagnostics (expansion
       warnings + dimcheck infos + L4xx lints). The only non-blocking emission,
       on the definitely-succeeding path — it can never co-fire with the [None]
       render above, so warnings never double-print. Routing through
       [Diagnostics.render] gives JSON under [--json-errors] and the ANSI box
       otherwise, matching the error path's shape. *)
    if Diagnostics.has_any d.ctx.diags then
      ignore (Diagnostics.render d.ctx.diags d.source);
    (* Observation + σ² autodiff: differentiate every likelihood argument
       (projection inlined) and every [DrawOverdispersed] σ² w.r.t. all
       parameters, filling the obs [*_grad] and [sigma_sq_grad] maps. Reuses
       the single differentiation authority [Autodiff.differentiate]; unlike
       the rate E600 path it never errors — an omitted or structural
       coefficient becomes a coded [DEUnsupported] that the runtime's
       fit-time gate refuses a gradient-based fit on. *)
    let param_names =
      List.map (fun (p : Ir.parameter) -> p.name) d.model.Ir.parameters in
    let tfs = d.model.Ir.time_functions and tbls = d.model.Ir.tables in
    let transitions =
      Passtime.time "autodiff-sigma"
        (fun () -> Autodiff.differentiate_overdispersion transitions param_names tfs tbls) in
    let observations =
      Passtime.time "autodiff-obs"
        (fun () -> Autodiff.differentiate_observations d.model.Ir.observations param_names tfs tbls) in
    (* Projection autodiff: ∂projection/∂compartment for a
       [DerivedExpr] (nonlinear) projection, filling each obs model's
       [projection_state_grad] — the factor-2 ingredient the ODE observation
       gradient consumes. The linear projection families (current-pop /
       cumulative-flow) emit nothing (a trivial selection handled directly).
       WrtPop, so it threads the compartments + bindings, like [rate_state_grad]. *)
    let observations =
      let comp_names = List.map (fun (c : Ir.compartment) -> c.name) d.model.Ir.compartments in
      let bindings = d.model.Ir.bindings in
      Passtime.time "autodiff-projection" (fun () ->
        List.map
          (fun (om : Ir.observation_model) ->
            { om with Ir.projection_state_grad =
                if !no_state_grad then []
                else Autodiff.differentiate_projection om.projection comp_names tfs tbls bindings })
          observations)
    in
    (* Initial-condition autodiff: ∂(initial_state)/∂θ per
       initial condition, filling the model's [ic_grad] map — the ODE
       forward-sensitivity seed S(t_start). A compartment whose expression has
       no parameter dependence is dropped (empty grad_map), so a block of
       literals emits nothing and a mixed block emits only the parameter-bearing
       compartments.

       [Init_order.closed] first substitutes each compartment reference by that
       compartment's own initial expression: the runtime evaluates the
       entries in dependency order, so `init { A = A0   B = A0 - A }` starts B
       at 0, and the seed has to differentiate that, not the raw `A0 - A` whose
       [Pop] leaf differentiates to zero (which would report ∂B/∂A0 = 1). *)
    let ic_grad =
      Passtime.time "autodiff-ic" (fun () ->
        List.filter_map
          (fun (comp, expr) ->
            match Autodiff.differentiate_ic expr param_names tfs tbls with
            | [] -> None
            | grad -> Some (comp, grad))
          (Init_order.closed d.model.Ir.initial_conditions
             ~bindings:d.model.Ir.bindings))
    in
    (* Initial-state law autodiff: ∂arg/∂θ for every argument of a drawn initial
       condition, which is what `initial_state_logpdf_grad` chains through. A
       law is a sampler, a density and a gradient; emitting the first two and
       leaving the third empty would give NUTS a gradient identically zero on
       that coordinate. Distinct from [ic_grad] above, which differentiates the
       MEAN for the deterministic ODE forward-sensitivity seed. *)
    let initial_conditions =
      Passtime.time "autodiff-init-law" (fun () ->
        Autodiff.differentiate_initial_conditions
          d.model.Ir.initial_conditions param_names tfs tbls)
    in
    (* Write the resolved quantity dimensions back onto the model before the
       value-preserving transforms (constant-fold/LICM never touch quantities). *)
    let m = annotate_quantity_dims qdims
              { d.model with Ir.transitions = transitions;
                             Ir.observations = observations;
                             Ir.initial_conditions;
                             Ir.ic_grad } in
    Ok (maybe_licm (maybe_constant_fold m))

(** Compile a model's source text to the finished IR, rendering every
    diagnostic. [name] becomes the model's name; [filename] is used in
    diagnostics and to resolve data paths relative to the model;
    [quantities] is the vocabulary file, as (file name, text). *)
let compile ?(name = "model") ?(filename = "<input>")
    ?(quantities : (string * string) option) (src : string)
    : (Ir.model, string) result =
  match compile_detail_result ~name ~filename ?quantities src with
  | Ok d -> finish_compile d
  | Error e -> Error e

(** As {!compile}, and also the distinct data files the compile opened, as
    (as written, resolved) pairs, for [camdlc --emit-deps]. The model is the
    one {!compile} returns. *)
let compile_with_reads ?(name = "model") ?(filename = "<input>")
    ?(quantities : (string * string) option) (src : string)
    : (Ir.model * (string * string) list, string) result =
  match compile_detail_result ~name ~filename ?quantities src with
  | Ok d ->
    (match finish_compile d with
     | Ok m -> Ok (m, Expander.reads d.ctx)
     | Error e -> Error e)
  | Error e -> Error e

(** As {!compile_with_reads}, and also the declared-model section of the IR
    envelope: the model as its author wrote it, built from the same compile,
    so the two describe one model. The model is the one {!compile} returns.
    [camdlc] calls this. *)
let compile_with_declared ?(name = "model") ?(filename = "<input>")
    ?(quantities : (string * string) option) (src : string)
    : (Ir.model * (string * string) list * Declared.declared_model, string) result =
  match compile_detail_result ~name ~filename ?quantities src with
  | Ok d ->
    (match finish_compile d with
     | Ok m -> Ok (m, Expander.reads d.ctx, fst (Lazy.force d.declared))
     | Error e -> Error e)
  | Error e -> Error e

(* Write the read-closure depfile for `camdlc --emit-deps`: a JSON sidecar
   listing the distinct external data files the compile opened. Atomic
   (temp + rename) so a failed/killed write never leaves a partial or stale
   depfile. Compile-time provenance only — never part of the IR. *)
let write_depfile ~(path : string) ~(model : string)
    (reads : (string * string) list) : unit =
  let j = `Assoc [
    "schema", `Int 1;
    "model",  `String model;
    "reads",  `List (List.map (fun (w, r) ->
      `Assoc [ "as_written", `String w; "resolved", `String r ]) reads);
  ] in
  let tmp = path ^ ".tmp" in
  let oc = open_out tmp in
  Fun.protect ~finally:(fun () -> close_out_noerr oc) (fun () ->
    output_string oc (Yojson.Safe.pretty_to_string j);
    output_char oc '\n');
  Sys.rename tmp path

(* ── Severity-agnostic diagnostic collection ─────────────────────────────────

   [collect_detail] runs the real compile pipeline (lex → parse → expand →
   validate → dimcheck → lint → autodiff) over a source, accumulating every
   diagnostic — errors, warnings, and infos alike — into the returned
   [Diagnostics.t], without rendering to stderr. It is the shared core behind
   both [collect_diagnostics] (test/tooling: keeps only the diagnostic list)
   and `inspect`'s `run_check` (the CLI: also renders the summary off the
   [compile_detail]). [compile] runs the same stages but renders to stderr and
   returns [Error] on failure; neither raises.

   Routing `run_check` through this core means one place defines the
   pipeline, so `check` and `compile` cannot disagree on a model's validity.
   It takes no vocabulary file, so `check` cannot check one.

   The pipeline short-circuits exactly as [compile] does: a structural
   [Validate] error stops before dimcheck (Validate runs first because
   dimcheck fails on unknown params). On the no-error path, all of
   dimcheck, lint, and autodiff run, so non-blocking warnings/lints (e.g.
   L402 on a clean-compiling model) are captured.

   Return shape mirrors [front_end_collect]: [(detail, diags, source)] with
   [detail = None] when lex/parse/expand structurally failed (then [diags]
   holds the E001), [Some d] otherwise (then [diags] is [d.ctx.diags], now
   also carrying any validate/dimcheck/lint/autodiff diagnostics). *)
let collect_detail ?(name = "model") ?(filename = "<input>") (src : string)
    : compile_detail option * Diagnostics.t * Source_cache.t =
  let (detail, diags, source) = front_end_collect ~name ~filename src in
  (match detail with
   | None -> ()                     (* lex/parse/expand failed; diags has the E001 *)
   | Some d ->
     (* Same post-expansion pipeline as [compile], minus rendering/abort: run the
        shared [run_analysis] stages, discarding its model outputs and keeping
        only the diagnostics it emitted into [d.ctx.diags] (the accumulator this
        function returns).

        Skip the passes if the front end (expander) already emitted errors —
        matching [compile], where [compile_detail_result] returns [Error] before
        [run_analysis] runs. Skipping avoids emitting a *second*, less-located
        diagnostic for a root cause the expander already reported with a located
        code (e.g. an init-membership error: a located E277 from [expand_init]
        would otherwise be shadowed by a no-location E513 from [run_validate]).
        One root cause, one diagnostic. *)
     if not (Diagnostics.has_errors d.ctx.diags) then begin
       (* Collecting diagnostics never needs the state Jacobian
          ([rate_state_grad]): it emits no diagnostic (a problem in it is a
          serialized [DEUnsupported] the fit-time gate refuses on), and only a
          gradient fit on the ODE backend reads it. It is also the one pass
          whose cost grows with transitions x compartments, so `camdlc check`
          on a 4,428-cell model took 26 min with it and 7 s without. *)
       let saved = !no_state_grad in
       no_state_grad := true;
       Fun.protect ~finally:(fun () -> no_state_grad := saved)
         (fun () -> ignore (run_analysis d))
     end);
  (detail, diags, source)

(* [collect_diagnostics] is the thin test/tooling projection of
   [collect_detail]: it discards the model/summary and returns just the
   accumulated diagnostics in the order they were reported. A fixture-driven test over it
   exercises the same diagnostic surface as the CLI. *)
let collect_diagnostics ?(name = "model") ?(filename = "<input>") (src : string)
    : Diagnostics.diagnostic list =
  let (_detail, diags, _source) = collect_detail ~name ~filename src in
  (* diags accumulates newest-first via [emit]; reverse to emission order. *)
  List.rev diags.Diagnostics.diags

(** [compile_outcome]: the non-raising projection of
    [collect_detail] into the structured {!outcome}. [collect_detail] runs the
    full pipeline (expand → validate → dimcheck → lint → autodiff),
    accumulating into [diags] without rendering or aborting; this wraps it.

    [value] is the expanded [compile_detail] exactly when no Error-severity
    diagnostic fired. A structural lex/parse/expand failure already yields
    [detail = None] (with the E001 in [diags]), so the [has_errors] gate
    subsumes that case. A late-phase error (validate E5xx, autodiff E600)
    arrives here as a value in [diagnostics]; [compile] returns the same
    error as a string [Error]. Neither raises. *)
let compile_outcome ?(name = "model") ?(filename = "<input>") (src : string)
    : compile_detail outcome =
  let (detail, diags, source) = collect_detail ~name ~filename src in
  { value       = (if Diagnostics.has_errors diags then None else detail);
    diagnostics = List.rev diags.Diagnostics.diags;
    source }
