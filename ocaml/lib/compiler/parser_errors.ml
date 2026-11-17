(** Errors found inside the parser's semantic actions, held until the driver
    can report them as diagnostics.

    A Menhir semantic action has no diagnostics context to write to, and
    raising from one would abandon the parse with no source location. So an
    action that finds a problem it can name, such as an unknown likelihood, an
    unknown dimension in a [dim] clause, or an origin not written as [date("...")],
    calls {!push_error} (or {!push_error_hint}, {!push_error_fix}) with the
    positions Menhir gives it, returns a placeholder value, and lets the parse
    go on. {!Param_annot}, which the parser calls, records its refusals the
    same way.

    {!Compiler.front_end_collect} clears {!pending_errors} before each compile
    and, once expansion has returned, turns every entry into an error
    diagnostic in the expander's context, in the order the actions pushed
    them. It does that only on that path: when the parse fails with a syntax
    error, or expansion raises, the recorded entries are dropped and only that
    failure is reported.

    The module also holds {!paren_starts}, a second piece of parser state the
    tree cannot carry, and {!ast_loc_of}, which turns Menhir positions into an
    {!Ast.loc}. All of this state is global to the process, so one parse runs
    at a time. *)

(** A machine-applicable fix: replace the source text between the two
    positions with the string. *)
type fix = Lexing.position * Lexing.position * string

(* Each entry: start and end of the offending text, code, message, optional
   hint, optional fix. Newest first. *)
let pending_errors
  : (Lexing.position * Lexing.position * string * string * string option * fix option) list ref
  = ref []

(* Start offsets (pos_cnum) of every parenthesized expression parsed so far.
   The AST keeps no node for parentheses, so the `/` rule reads this to tell
   `I / (5 'days)`, which is unambiguous, from `I / 5 'days` (E107). Offsets
   are per file, so [Compiler.parse_unit] resets it before each parse. *)
let paren_starts : (int, unit) Hashtbl.t = Hashtbl.create 64

let push_error ~sp ~ep ~code ~msg =
  pending_errors := (sp, ep, code, msg, None, None) :: !pending_errors

(* Like [push_error], with hint text. *)
let push_error_hint ~sp ~ep ~code ~msg ~hint =
  pending_errors := (sp, ep, code, msg, Some hint, None) :: !pending_errors

(* Like [push_error_hint], with an optional fix-it. *)
let push_error_fix ~sp ~ep ~code ~msg ~hint ~(fix : fix option) =
  pending_errors := (sp, ep, code, msg, Some hint, fix) :: !pending_errors

(* Convert Menhir's $startpos/$endpos into the [Ast.loc] that declaration
   records carry. The file is left empty; see [Ast.loc]. *)
let ast_loc_of ~(sp : Lexing.position) ~(ep : Lexing.position) : Ast.loc =
  { Ast.file     = "";
    line         = sp.pos_lnum;
    col          = sp.pos_cnum - sp.pos_bol + 1;
    end_line     = ep.pos_lnum;
    end_col      = ep.pos_cnum - ep.pos_bol + 1;
  }
