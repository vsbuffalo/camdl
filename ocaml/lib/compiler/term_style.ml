(** Terminal styling for the compiler's output: a colour for each kind of
    name a model contains, and the one policy that decides whether a stream
    is coloured at all.

    Each style wraps a [Fmt] printer, so
    [Term_style.compartment Fmt.string ppf "S"] prints [S] in magenta.
    Transitions are blue, compartments magenta, parameters (and the other
    names {!Pp_expr} prints in that colour, such as [t] and bindings)
    green, tables and forcings yellow, dimensions cyan; operators and
    indices are faint, errors red and warnings yellow. The styles only mark
    the text: whether the escape codes are written is decided by the
    renderer installed on the formatter.

    {!renderer} is that decision, highest precedence first: [--no-color]
    turns colour off, then a set [NO_COLOR] variable, then a set
    [CLICOLOR_FORCE] variable turns it on, and otherwise colour is on
    exactly when the stream is a terminal. The Rust command-line tool
    applies the same policy, so [camdlc] and [camdl] agree about a pipe.
    {!apply_policy} installs it on standard output and standard error;
    [camdlc inspect] and [camdlc check] call it, and the diagnostics
    renderer calls {!renderer} for standard error. [--no-color] is recorded
    once, in {!no_color_flag}, by the command that parses it, so a later
    call reads the same decision. *)

(* The eta-expansion (fun pp ppf x -> ...) avoids the OCaml value restriction
   on partially-applied polymorphic functions. *)

let transition   pp ppf x = Fmt.styled (`Fg `Blue)    pp ppf x
let compartment  pp ppf x = Fmt.styled (`Fg `Magenta) pp ppf x
let param        pp ppf x = Fmt.styled (`Fg `Green)   pp ppf x
let table        pp ppf x = Fmt.styled (`Fg `Yellow)  pp ppf x
let dimension    pp ppf x = Fmt.styled (`Fg `Cyan)    pp ppf x
let dim_style    pp ppf x = Fmt.styled `Faint          pp ppf x
let bold         pp ppf x = Fmt.styled `Bold           pp ppf x
let error_style  pp ppf x = Fmt.styled (`Fg `Red)     pp ppf x
let warning_style pp ppf x = Fmt.styled (`Fg `Yellow) pp ppf x

(* ── The colour policy ───────────────────────────────────────────────────

   One function decides whether a stream gets escape codes, and every
   [Fmt.set_style_renderer] call goes through it. Highest precedence first:
   the [--no-color] flag (off), [NO_COLOR] (off; no-color.org),
   [CLICOLOR_FORCE] with any value (on: the convention ls, grep and git use
   when piped through [less -R]), then whether that stream is a terminal.
   This is the policy the Rust CLI applies ([style::Painter::for_stream]),
   so [camdlc] and [camdl] agree about a pipe.

   [--no-color] is recorded once, by the command that parses it, in
   [no_color_flag]; a later [set_style_renderer] call (a diagnostics render
   after [inspect] chose its renderer) reads the same decision instead of
   forcing colour back on. *)

let no_color_flag = ref false

let renderer (stream : [ `Stdout | `Stderr ]) : Fmt.style_renderer =
  if !no_color_flag then `None
  else if Sys.getenv_opt "NO_COLOR" <> None then `None
  else if Sys.getenv_opt "CLICOLOR_FORCE" <> None then `Ansi_tty
  else
    let fd = match stream with `Stdout -> Unix.stdout | `Stderr -> Unix.stderr in
    if Unix.isatty fd then `Ansi_tty else `None

(* Set both standard formatters' renderers from [renderer]. *)
let apply_policy () =
  Fmt.set_style_renderer Fmt.stdout (renderer `Stdout);
  Fmt.set_style_renderer Fmt.stderr (renderer `Stderr)
