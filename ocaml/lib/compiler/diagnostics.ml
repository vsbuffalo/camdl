(* Diagnostic collection and rendering for the camdl compiler. *)

(* ── Types ─────────────────────────────────────────────────────────────────── *)

(** Three-level severity — M5 in the 2026-04-19 compiler review.
    Previously only Error | Warning existed, and the dimcheck Info
    diagnostic (I300 "undetermined dimension") was emitted as a
    Warning here, which confused `--json-errors` clients (they saw
    `"severity": "warning"` for what the compiler intended as non-
    blocking informational) and promoted Info to a blocking level in
    any `-Werror`-style workflow. Info sits below Warning, never
    blocks compilation, and renders in a distinct style. *)
type severity = Error | Warning | Info

type loc = {
  file     : string;
  line     : int;
  col      : int;
  end_line : int;
  end_col  : int;
}

type diagnostic = {
  severity : severity;
  code     : string;              (* "E100", "W200", etc. *)
  loc      : loc;
  message  : string;
  detail   : string option;
  hint     : string option;
  related  : (loc * string) list; (* secondary locations + labels *)
}

(* ── Collection ─────────────────────────────────────────────────────────────── *)

type t = { mutable diags : diagnostic list }

let create () = { diags = [] }
let emit t d  = t.diags <- d :: t.diags

let has_errors t =
  List.exists (fun d -> d.severity = Error) t.diags

let has_any t = t.diags <> []

(* ── Locations ───────────────────────────────────────────────────────────── *)

let no_loc = { file = ""; line = 0; col = 0; end_line = 0; end_col = 0 }

let loc_of_positions ~file (sp : Lexing.position) (ep : Lexing.position) =
  { file;
    line     = sp.Lexing.pos_lnum;
    col      = sp.Lexing.pos_cnum - sp.Lexing.pos_bol + 1;
    end_line = ep.Lexing.pos_lnum;
    end_col  = ep.Lexing.pos_cnum - ep.Lexing.pos_bol + 1;
  }

(* ── Rendering helpers ───────────────────────────────────────────────────── *)

let box_tl = "\xe2\x94\x8c"  (* ┌ *)
let box_h  = "\xe2\x94\x80"  (* ─ *)
let box_v  = "\xe2\x94\x82"  (* │ *)

let pp_sev_code ppf (sev, code) =
  match sev with
  | Error ->
    Term_style.error_style
      (Term_style.bold (fun ppf () -> Fmt.pf ppf "error[%s]" code)) ppf ()
  | Warning ->
    Term_style.warning_style
      (Term_style.bold (fun ppf () -> Fmt.pf ppf "warning[%s]" code)) ppf ()
  | Info ->
    (* Info uses dimmed style — non-blocking, informational. *)
    Term_style.dim_style
      (Term_style.bold (fun ppf () -> Fmt.pf ppf "info[%s]" code)) ppf ()

(** Render one ┌─ source block at the given location. *)
let pp_block ppf (cache : Source_cache.t) sev (l : loc) (label : string option) =
  if l.line = 0 then ()
  else begin
    (* Header: ┌─ file:line:col *)
    let file_ref =
      if l.file = "" then Printf.sprintf "line %d" l.line
      else Printf.sprintf "%s:%d:%d" l.file l.line l.col
    in
    Term_style.dim_style (fun ppf () ->
      Fmt.pf ppf "  %s%s %s@\n" box_tl box_h file_ref;
      Fmt.pf ppf "  %s@\n" box_v
    ) ppf ();
    (* Source line *)
    (match Source_cache.get_line cache ~file:l.file l.line with
     | None -> ()
     | Some text ->
       let lno  = string_of_int l.line in
       let pad  = String.make (max 0 (3 - String.length lno)) ' ' in
       (* "  NNN│  text" *)
       Term_style.dim_style Fmt.string ppf pad;
       Term_style.bold (Term_style.transition Fmt.string) ppf lno;
       Term_style.dim_style Fmt.string ppf box_v;
       Fmt.pf ppf "  %s@\n" text;
       (* Underline line:  "  │  ·····~~~~^" *)
       let col0 = max 0 (l.col - 1) in
       let span = if l.end_line = l.line then max 1 (l.end_col - l.col) else 1 in
       let ul   = String.make (span - 1) '~' ^ "^" in
       Fmt.pf ppf "  %s  %s" box_v (String.make col0 ' ');
       (match sev with
        | Error   -> Term_style.error_style   Fmt.string ppf ul
        | Warning -> Term_style.warning_style Fmt.string ppf ul
        | Info    -> Term_style.dim_style     Fmt.string ppf ul);
       (match label with Some s -> Fmt.pf ppf " %s" s | None -> ());
       Fmt.pf ppf "@\n"
    );
    Term_style.dim_style (fun ppf () -> Fmt.pf ppf "  %s@\n" box_v) ppf ()
  end

let render_one ppf cache (d : diagnostic) =
  pp_sev_code ppf (d.severity, d.code);
  Fmt.pf ppf ": %s@\n@\n" d.message;
  pp_block ppf cache d.severity d.loc None;
  (match d.detail with
   | None   -> ()
   | Some s ->
     Term_style.dim_style Fmt.string ppf "  = note: ";
     Fmt.pf ppf "%s@\n" s);
  (match d.hint with
   | None   -> ()
   | Some s ->
     Term_style.dim_style Fmt.string ppf "  = hint: ";
     Fmt.pf ppf "%s@\n" s);
  let related =
    if List.length d.related > 3 then
      let n = List.length d.related - 3 in
      List.filteri (fun i _ -> i < 3) d.related
      @ [(no_loc, Printf.sprintf "... and %d more" n)]
    else d.related
  in
  List.iter (fun (rl, lbl) ->
    if rl.line > 0 then pp_block ppf cache d.severity rl (Some lbl)
    else Fmt.pf ppf "  %s@\n" lbl
  ) related;
  Fmt.pf ppf "@\n"

(* ── JSON serialisation ──────────────────────────────────────────────────── *)

let json_errors_mode = ref false

let severity_string = function
  | Error   -> "error"
  | Warning -> "warning"
  | Info    -> "info"

let loc_to_json (l : loc) : Yojson.Safe.t =
  `Assoc [
    ("file",     `String l.file);
    ("line",     `Int    l.line);
    ("col",      `Int    l.col);
    ("end_line", `Int    l.end_line);
    ("end_col",  `Int    l.end_col);
  ]

let diagnostic_to_json (d : diagnostic) : Yojson.Safe.t =
  let fields : (string * Yojson.Safe.t) list = [
    ("severity", `String (severity_string d.severity));
    ("code",     `String d.code);
    ("message",  `String d.message);
    ("loc",      loc_to_json d.loc);
  ] in
  let fields = match d.detail with
    | None   -> fields
    | Some s -> fields @ [("detail", `String s)]
  in
  let fields = match d.hint with
    | None   -> fields
    | Some s -> fields @ [("hint", `String s)]
  in
  `Assoc fields

(* Total order on diagnostics for stable, source-following output: sort
   ascending by (file, line, code, message) so JSON and text renderers agree
   and de-duplicate identically. *)
let by_source_order (a : diagnostic) (b : diagnostic) : int =
  let c = compare a.loc.file b.loc.file in
  if c <> 0 then c else
  let c = compare a.loc.line b.loc.line in
  if c <> 0 then c else
  let c = compare a.code b.code in
  if c <> 0 then c else
  compare a.message b.message

let to_json_string (t : t) : string =
  (* Sort ascending by (file, line, code, message) so JSON output
     matches text-mode ordering. Pre-fix this used rev_map on the
     raw diag list, giving unsorted, arbitrary-insertion-order
     output that disagreed with the text renderer. *)
  let sorted = List.sort_uniq by_source_order t.diags in
  let arr = `List (List.map diagnostic_to_json sorted) in
  Yojson.Safe.to_string arr

let render_all t cache ppf =
  let sorted = List.sort_uniq by_source_order t.diags in
  List.iter (render_one ppf cache) sorted

(** Render every diagnostic in [t] to stderr, honouring [json_errors_mode]:
    a single JSON array under [--json-errors], otherwise the ANSI
    source-block boxes. Returns the payload string — the JSON array under
    [--json-errors], else ["compilation failed"] — which [Compiler.compile]
    hands back as its [Error] value. The single rendering surface for both
    the error path and the non-blocking warning/lint path, so JSON and ANSI
    stay identical and there is one emission shape per call. The library
    never raises or exits: rendering is a side effect; control flow is the
    caller's [Error]/[outcome] value (gh#181). *)
let render t cache : string =
  if !json_errors_mode then (
    let msg = to_json_string t in
    Printf.eprintf "%s\n" msg;
    msg
  ) else (
    Fmt.set_style_renderer Fmt.stderr `Ansi_tty;
    render_all t cache Fmt.stderr;
    "compilation failed"
  )

(* ── Shorthand constructors ──────────────────────────────────────────────── *)

(* Pure constructors: build a [diagnostic] without emitting. The
   post-expansion passes return [diagnostic list] and let the caller emit
   (gh#181), so they construct via these; the expansion phase still emits
   directly through [error]/[warning]/[info] below. *)
let mk_error ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  { severity=Error; code; loc; message; detail; hint; related }

let mk_warning ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  { severity=Warning; code; loc; message; detail; hint; related }

let mk_info ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  { severity=Info; code; loc; message; detail; hint; related }

let error t ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  emit t (mk_error ~code ~loc ~message ?detail ?hint ~related ())

let warning t ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  emit t (mk_warning ~code ~loc ~message ?detail ?hint ~related ())

(** Info diagnostic — non-blocking, dimmed style, distinct from
    Warning in JSON output. See `severity` type above. *)
let info t ~code ~loc ~message ?detail ?hint ?(related=[]) () =
  emit t (mk_info ~code ~loc ~message ?detail ?hint ~related ())
