(* Compile a camdl source string + optional model name to an Ir.model *)

type compile_detail = {
  model   : Ir.model;
  ctx     : Expander.context;
  summary : Expander.model_summary;
  source  : Source_cache.t;
}

(** Structured, non-raising compile outcome (gh#181 step 1).

    A value-typed surface over [collect_detail]: every diagnostic — errors,
    warnings, infos — is returned as a [diagnostic list] rather than rendered
    and raised. [value] is [Some] exactly when no [Error]-severity diagnostic
    was produced. Nothing in the library raises: [compile] returns [Error]
    on failure, this returns [value = None].

    This is the accumulating shape the gh#181 proposal targets — structurally
    [MaybeT (Writer (diagnostic list))]: the diagnostic log is always present;
    the value is present only on success. Step 1 deliberately carries the
    expanded [compile_detail] (not a fully-finished [Ir.model]) — promoting
    [value] to a gradient-attached, constant-folded model and routing
    [simulate]/[fit]/CLI through this one surface is steps 2–4 of the
    migration. Keeping it a pure addition here means no existing caller
    changes behaviour. *)
type 'a outcome = {
  value       : 'a option;
  diagnostics : Diagnostics.diagnostic list;
  source      : Source_cache.t;
}

(* ── Front-end core ───────────────────────────────────────────────────────

   The single, non-aborting lex/parse/expand front end. It runs the
   pipeline once, accumulating EVERY front-end diagnostic — the E001 on a
   lex/parse/expand failure, the W100 lex warnings, and the parser-action
   errors — into the returned [Diagnostics.t], and never renders or
   aborts. Both consumers build on it:

   - [compile_detail_result] (the production path) wraps it: on a
     front-end error it renders the diagnostics and returns [Error]; on
     a clean expand it returns [Ok].
   - [collect_diagnostics] (test/tooling) uses it directly and continues
     the downstream pipeline.

   Return shape: [(detail, diags, source)]. [detail] is [None] when
   lex/parse/expand structurally failed (then [diags] holds the E001);
   [Some d] when expansion produced a model (then [diags] is [d.ctx.diags],
   which may itself carry expansion-phase errors/warnings and the drained
   W100 / parser-action diagnostics — the caller decides whether to
   continue). [source] is the [Source_cache] for the input, returned even
   on the [None] path so a renderer can show the offending line. *)
let front_end_collect ?(name = "model") ?(filename = "<input>") (src : string)
    : compile_detail option * Diagnostics.t * Source_cache.t =
  let source = Source_cache.of_string ~filename src in
  (* Drain any stale lex-phase warnings from a previous compilation in the
     same process. pending_warnings is a mutable global ref; clearing it
     here ensures we never replay warnings from a prior run. *)
  Lexer.pending_warnings := [];
  Parser_errors.pending_errors := [];
  let parse_diags = Diagnostics.create () in
  match
    (try
       let lexbuf = Lexing.from_string src in
       Lexing.set_filename lexbuf filename;
       let t_parse = Sys.time () in
       let decls =
         (try Ok (Parser.file Lexer.token lexbuf)
          with
          | Lexer.LexError msg ->
            let pos = lexbuf.Lexing.lex_curr_p in
            Diagnostics.error parse_diags ~code:"E001"
              ~loc:(Diagnostics.loc_of_positions ~file:filename pos pos)
              ~message:(Printf.sprintf "lex error: %s" msg) ();
            Error ()
          | Parser.Error ->
            let pos = lexbuf.Lexing.lex_curr_p in
            Diagnostics.error parse_diags ~code:"E001"
              ~loc:(Diagnostics.loc_of_positions ~file:filename pos pos)
              ~message:"syntax error" ();
            Error ())
       in
       Passtime.record "parse" (Sys.time () -. t_parse);
       decls
     with
     | Failure msg ->
       Diagnostics.error parse_diags ~code:"E001" ~loc:Diagnostics.no_loc
         ~message:msg ();
       Error ()
     | exn ->
       Diagnostics.error parse_diags ~code:"E001" ~loc:Diagnostics.no_loc
         ~message:(Printexc.to_string exn) ();
       Error ())
  with
  | Error () -> (None, parse_diags, source)
  | Ok decls ->
    let source_dir =
      if filename = "<input>" then "" else Filename.dirname filename
    in
    (match
       (try Ok (Passtime.time "expand"
                  (fun () -> Expander.expand_detail ~source_dir ~filename name decls))
        with
        | Failure msg ->
          Diagnostics.error parse_diags ~code:"E001" ~loc:Diagnostics.no_loc
            ~message:msg ();
          Error ()
        | exn ->
          Diagnostics.error parse_diags ~code:"E001" ~loc:Diagnostics.no_loc
            ~message:(Printexc.to_string exn) ();
          Error ())
     with
     | Error () -> (None, parse_diags, source)
     | Ok (model, ctx, summary) ->
       (* Drain lex-phase warnings (e.g. inconsistent digit grouping)
          collected before the expander's ctx.diags was available. *)
       List.iter (fun (sp, ep, msg) ->
         Diagnostics.warning ctx.diags ~code:"W100"
           ~loc:(Diagnostics.loc_of_positions ~file:filename sp ep)
           ~message:msg ()
       ) (List.rev !Lexer.pending_warnings);
       Lexer.pending_warnings := [];
       (* Drain parser-action errors collected from semantic actions that
          used to `failwith` (n3 in the 2026-04-19 compiler review). *)
       List.iter (fun (sp, ep, code, msg, hint) ->
         Diagnostics.error ctx.diags ~code
           ~loc:(Diagnostics.loc_of_positions ~file:filename sp ep)
           ~message:msg ?hint ()
       ) (List.rev !Parser_errors.pending_errors);
       Parser_errors.pending_errors := [];
       (Some { model; ctx; summary; source }, ctx.diags, source))

(** Production front end: run [front_end_collect], and on any front-end
    error (lex/parse/expand failure, or a parser-action error drained
    into [ctx.diags]) render the diagnostics and return [Error]; on a
    clean expand return [Ok]. The [Error] payload is the rendered string
    from [Diagnostics.render] — the serialized JSON array under
    [--json-errors], else ["compilation failed"]. CLI entry points
    recognize the payload shape and exit without re-printing a redundant
    Error line (m5 in the 2026-04-19 compiler review). Warnings are NOT
    rendered here: callers render once at the end of their pipeline so
    expansion-phase warnings don't print twice when downstream passes
    (dimcheck) also emit diagnostics (M3). *)
let compile_detail_result ?(name = "model") ?(filename = "<input>") (src : string)
    : (compile_detail, string) result =
  let (detail, diags, source) = front_end_collect ~name ~filename src in
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

let no_dim_check = ref false

(** Run the sparse-coupling constant-fold pass. On by default; the
    CAMDL_NO_CONSTANT_FOLD escape hatch forces it off (see the call site).
    Exposed as a ref so tests that assert on the *unfolded* IR shape (the
    expander's TableLookup-flattening contract) can disable it locally,
    mirroring [no_dim_check]. *)
let constant_fold = ref true

(** Translate a `Validate.error` into an E5xx Diagnostic and attach
    it to the given context. Codes are new (E500–E511) — the existing
    E2xx range covers parser/expansion-phase duplicates and unknowns,
    but `Validate.validate` runs post-expansion and can catch cases
    the parser/expander miss (e.g. unknown reference in a let-binding
    that expands into a rate, or a `Real` compartment with no ODE).
    A separate code range makes that distinction visible in output. *)
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
  let site_loc = function
    | Validate.InTransition  n -> tr_loc n
    | Validate.InOde         c -> comp_loc c
    | Validate.InObservation n -> obs_loc n
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
  in
  Diagnostics.mk_error ~code ~loc ~message ?hint ()

(** Run post-expansion structural validation.

    Wired in per M1 of the 2026-04-19 compiler review — previously
    `Validate.validate` existed in `lib/ir/validate.ml` but was never
    called from the compile pipeline, so its unknown-reference /
    missing-ODE / zero-delta checks ran nowhere. Without this pass
    the `ode_equations = []` hardcoding bug (C5) would have been
    invisible; now C5 is fixed AND the integrity net that would have
    caught it in the first place runs.

    Order: post-expansion, pre-dimcheck. Dimcheck ICEs on unknown
    params, so running Validate first gives the user a clean
    "unknown parameter 'foo'" error instead of a dimcheck trace. *)
let run_validate (d : compile_detail) : Diagnostics.diagnostic list =
  match Validate.validate d.model with
  | Ok () -> []
  | Error errs -> List.map (diagnose_validate_error d.ctx) errs

(** Run Dimcheck on a compiled model and route results into the diagnostic
    context. Exposed so `camdlc check` runs the same pass as `camdlc compile`;
    previously `check` skipped dimcheck entirely (GH #9). *)
let run_dimcheck (d : compile_detail) : Diagnostics.diagnostic list =
  if !no_dim_check then []
  else
    (* dimcheck runs on the IR (no source spans); it tags each diagnostic with
       the construct it concerns, which we resolve to the declaration's loc. *)
    let loc_of_subject = function
      | Some (Dimcheck.STransition n)  -> Expander.transition_loc  d.ctx n
      | Some (Dimcheck.SOde c)         -> Expander.compartment_loc d.ctx c
      | Some (Dimcheck.SObservation n) -> Expander.obs_loc         d.ctx n
      | Some (Dimcheck.SBinding _) | None -> Diagnostics.no_loc
    in
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
    ) (Dimcheck.check_model d.model).diagnostics

(** Run the model linter on a compiled model and route its results into
    the diagnostic context as non-blocking warnings. Lints (L4xx) flag
    semantically valid but discouraged patterns (e.g. L402 dead
    compartment); they render with hint text but never set [has_errors],
    so the build does not fail on a lint. Called right after
    [run_dimcheck] so both `camdlc compile` and `camdlc check` run it. *)
let run_lint (d : compile_detail) : Diagnostics.diagnostic list =
  List.map (fun (l : Lint.diagnostic) ->
    let loc = match l.compartment with
      | Some c -> Expander.compartment_loc d.ctx c
      | None   -> Diagnostics.no_loc
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
    structural forcing data (interpolation knots / spline basis / piecewise
    steps), or a non-constant table lookup index — yields an E600 (with
    source location) and leaves that transition's [rate_grad] empty. Live
    coefficients whose gradient is not yet emitted (a periodic step value, an
    inline-table value via a non-constant index — gh#215) are NOT errors: the
    parameter is simply omitted from [rate_grad] and the Rust NUTS guard
    rejects a NUTS fit that depends on it. Pure: it neither emits into a
    context nor renders, so [compile] (which short-circuits on the resulting
    errors) and [collect_diagnostics] (which does not) share it. *)
let differentiate_transitions (d : compile_detail)
    : Ir.transition list * Diagnostics.diagnostic list =
  let param_names = List.map (fun (p : Ir.parameter) -> p.name) d.model.Ir.parameters in
  let tr_loc name =
    (* Find the original (pre-expansion) transition declaration by prefix
       match: expanded name "infection_child" → base "infection". *)
    match List.find_opt (fun (td : Ast.transition_decl) ->
      let b = td.trname and bl = String.length td.trname in
      let el = String.length name in
      name = b || (el > bl && String.sub name 0 bl = b && name.[bl] = '_')
    ) d.ctx.orig_transitions with
    | Some td -> Expander.diag_loc_of_ast_ctx d.ctx td.trloc
    | None -> Diagnostics.no_loc
  in
  let diags = ref [] in
  let transitions =
    Passtime.time "autodiff" (fun () ->
      List.map (fun (t : Ir.transition) ->
        match Autodiff.differentiate_rate t.rate param_names
                d.model.Ir.time_functions d.model.Ir.tables with
        | Ok rate_grad -> { t with Ir.rate_grad }
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
          { t with Ir.rate_grad = [] }
      ) d.model.Ir.transitions)
  in
  (transitions, List.rev !diags)

(** Sparse-coupling constant-fold (on by default): resolves
    constant-indexed inline-table lookups and drops zero-W terms from FOI
    Reduce sums, collapsing the dense P-term spatial sum to its k nonzero
    terms. Proven byte-identical by the A/B gate (rust
    .../gate_constant_fold_ab). Set CAMDL_NO_CONSTANT_FOLD to emit the
    unfolded (dense) IR — an escape hatch for debugging the pass or
    inspecting the pre-fold shape. *)
let maybe_constant_fold (m : Ir.model) : Ir.model =
  let fold_on = !constant_fold && Sys.getenv_opt "CAMDL_NO_CONSTANT_FOLD" = None in
  if fold_on then Passtime.time "constant_fold" (fun () -> Constant_fold.fold_model m)
  else m

(* gh#272 Loop-invariant code motion. Opt-in via CAMDL_LICM, off by default.
   Runs AFTER constant_fold (so it hoists already-folded subtrees; constant_fold
   never sees a PerEvalRef) and is the last transform before serialization, so
   the pre-LICM passes (autodiff/dimcheck/validate/lint) never encounter the
   node. *)
let maybe_licm (m : Ir.model) : Ir.model =
  if Sys.getenv_opt "CAMDL_LICM" <> None
  then Passtime.time "licm" (fun () -> Licm.licm_model m)
  else m

(* The post-expansion pipeline (validate → dimcheck → lint → autodiff →
   constant-fold), factored out of [compile] so [compile_with_reads] can reuse
   it without duplicating the high-risk pipeline. [compile] stays
   byte-identical — it is now [compile_detail_result] + [finish_compile]. *)
let finish_compile (d : compile_detail) : (Ir.model, string) result =
  (* Post-expansion passes are pure (return diagnostic lists); emit them
     into [d.ctx.diags] — the accumulator [has_errors]/[render] read — and
     on any Error-severity diagnostic return [Error (render …)] rather than
     raising. [compile] never throws (gh#181): a late-phase error (validate
     E5xx, dimcheck, autodiff E600) now arrives as [Error], the same shape
     the front-end path already returns, so the CLI exits cleanly (1)
     instead of on an uncaught [Compile_error] (a Fatal-error trace, exit 2).
     [render] still writes the diagnostics to stderr exactly as before; only
     the control flow changes from raise to return.
     Validate first (M1 / C5 in the 2026-04-19 compiler review). *)
  let emit_all = List.iter (Diagnostics.emit d.ctx.diags) in
  let fail () : (Ir.model, string) result =
    Error (Diagnostics.render d.ctx.diags d.source) in
  let vdiags = Passtime.time "validate" (fun () -> run_validate d) in
  emit_all vdiags;
  (* Validate short-circuits before dimcheck (dimcheck ICEs on unknown
     params), matching the original short-circuit ordering. *)
  if vdiags <> [] then fail ()
  else begin
    emit_all (Passtime.time "dimcheck" (fun () -> run_dimcheck d));
    emit_all (Passtime.time "lint" (fun () -> run_lint d));
    if Diagnostics.has_errors d.ctx.diags then fail ()
    else begin
      let (transitions, gdiags) = differentiate_transitions d in
      emit_all gdiags;
      if Diagnostics.has_errors d.ctx.diags then fail ()
      else begin
        (* Single render of any collected non-blocking diagnostics
           (expansion warnings + dimcheck infos + L4xx lints). The ONLY
           non-blocking emission, on the definitely-succeeding path after
           the final [has_errors] check — it can never co-fire with a
           [fail ()] render above, so warnings never double-print. Routing
           through [Diagnostics.render] gives JSON under [--json-errors] and
           the ANSI box otherwise, matching the error path's shape. *)
        if Diagnostics.has_any d.ctx.diags then
          ignore (Diagnostics.render d.ctx.diags d.source);
        let m = { d.model with Ir.transitions = transitions } in
        Ok (maybe_licm (maybe_constant_fold m))
      end
    end
  end

let compile ?(name = "model") ?(filename = "<input>") (src : string) : (Ir.model, string) result =
  match compile_detail_result ~name ~filename src with
  | Ok d -> finish_compile d
  | Error e -> Error e

(* As [compile], but also returns the read-closure: the distinct external data
   files the compile opened, as (as-written, resolved) pairs, for
   `camdlc --emit-deps`. Reuses [finish_compile], so the compiled model is
   byte-identical to [compile]'s. The reads are populated during expansion
   (before [finish_compile]), so they are read off [d.ctx] on the success
   path. *)
let compile_with_reads ?(name = "model") ?(filename = "<input>") (src : string)
    : (Ir.model * (string * string) list, string) result =
  match compile_detail_result ~name ~filename src with
  | Ok d ->
    (match finish_compile d with
     | Ok m -> Ok (m, Expander.reads d.ctx)
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
   validate → dimcheck → lint → autodiff) over a source, accumulating EVERY
   diagnostic — errors, warnings, and infos alike — into the returned
   [Diagnostics.t], without rendering to stderr. It is the shared core behind
   both [collect_diagnostics] (test/tooling: keeps only the diagnostic list)
   and `inspect`'s `run_check` (the CLI: also renders the summary off the
   [compile_detail]). [compile] runs the same stages but renders to stderr and
   returns [Error] on failure; neither raises.

   Routing `run_check` through this core is the cure for the recurring
   check/compile divergence (gh#9 re dimcheck, gh#170 re validate): there is
   now ONE place that defines "the front-end pipeline", so `check` and
   `compile` cannot disagree on a model's validity.

   The pipeline short-circuits exactly as [compile] does: a structural
   [Validate] error stops before dimcheck (Validate runs first precisely
   because dimcheck ICEs on unknown params). On the no-error path, all of
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
     (* Same staged pipeline as [compile], minus rendering/abort: Validate
        first (it gates dimcheck, which ICEs on unknown params), then
        dimcheck + lint, then autodiff. Short-circuit after Validate matches
        [compile]; downstream passes run only on a structurally-valid model.
        The passes are pure now, so emit their lists into [d.ctx.diags] (the
        accumulator this function returns). *)
     let emit_all = List.iter (Diagnostics.emit d.ctx.diags) in
     (* Short-circuit the post-expansion passes if the front end (expander)
        already emitted errors — matching [compile], where
        [compile_detail_result] returns [Error] before [run_validate] runs.
        Skipping it here avoids emitting a *second*, less-located diagnostic
        for a root cause the expander already reported with a located code
        (e.g. an init-membership error: located E277 from [expand_init] would
        otherwise be shadowed by a no-location E513 from [run_validate]).
        gh#114 reviewer feedback: one root cause → one diagnostic. *)
     if not (Diagnostics.has_errors d.ctx.diags) then begin
       let vdiags = Passtime.time "validate" (fun () -> run_validate d) in
       emit_all vdiags;
       if vdiags = [] then begin
         emit_all (Passtime.time "dimcheck" (fun () -> run_dimcheck d));
         emit_all (Passtime.time "lint" (fun () -> run_lint d));
         if not (Diagnostics.has_errors d.ctx.diags) then
           emit_all (snd (differentiate_transitions d))
       end
     end);
  (detail, diags, source)

(* [collect_diagnostics] is the thin test/tooling projection of
   [collect_detail]: it discards the model/summary and returns just the
   accumulated diagnostics in source order. A fixture-driven test over it
   exercises the same diagnostic surface as the CLI. *)
let collect_diagnostics ?(name = "model") ?(filename = "<input>") (src : string)
    : Diagnostics.diagnostic list =
  let (_detail, diags, _source) = collect_detail ~name ~filename src in
  (* diags accumulates newest-first via [emit]; reverse to source order. *)
  List.rev diags.Diagnostics.diags

(** [compile_outcome] (gh#181 step 1): the non-raising projection of
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
