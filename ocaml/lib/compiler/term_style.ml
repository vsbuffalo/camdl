(* Semantic color helpers using the Fmt library (Daniel Bünzli).
   Each value is a styled formatter constructor:
     Term_style.compartment Fmt.string ppf "S"  -- prints "S" in magenta

   The eta-expansion (fun pp ppf x -> ...) avoids the OCaml value restriction
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
