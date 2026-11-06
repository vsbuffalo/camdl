(* Parser-action error buffering. n3 in the 2026-04-19 compiler review.

   Menhir semantic actions can't thread a Diagnostics context, so actions
   that used to `failwith "..."` (invalid origin, unknown dim literal,
   unknown likelihood, unknown unit, unknown output section, invalid
   extends, missing recurring 'every') now push a
   (start, end, code, message) tuple here and return a placeholder
   value. `compiler.ml` drains this list into `ctx.diags` after parsing,
   giving the user a proper diagnostic with source location instead of
   a bare stack trace. *)

(* A machine-applicable fix-it: replace the source between the two positions
   with the text. *)
type fix = Lexing.position * Lexing.position * string

let pending_errors
  : (Lexing.position * Lexing.position * string * string * string option * fix option) list ref
  = ref []

let push_error ~sp ~ep ~code ~msg =
  pending_errors := (sp, ep, code, msg, None, None) :: !pending_errors

(* Like [push_error] but carries hint text. Kept as a separate entry point
   so the many existing [push_error] call sites need no change. *)
let push_error_hint ~sp ~ep ~code ~msg ~hint =
  pending_errors := (sp, ep, code, msg, Some hint, None) :: !pending_errors

(* Like [push_error_hint], with an optional fix-it. *)
let push_error_fix ~sp ~ep ~code ~msg ~hint ~(fix : fix option) =
  pending_errors := (sp, ep, code, msg, Some hint, fix) :: !pending_errors

(* Helper for parser semantic actions: convert menhir $startpos/$endpos
   into an Ast.loc suitable for stashing in decl records so the
   expander can thread source locations into diagnostics without
   re-walking the parse tree (M9 in the 2026-04-19 compiler review). *)
let ast_loc_of ~(sp : Lexing.position) ~(ep : Lexing.position) : Ast.loc =
  { Ast.file     = "";
    line         = sp.pos_lnum;
    col          = sp.pos_cnum - sp.pos_bol + 1;
    end_line     = ep.pos_lnum;
    end_col      = ep.pos_cnum - ep.pos_bol + 1;
  }
