{
  open Parser

  exception LexError of string

  (* Strip underscore separators from a numeric literal before parsing.
     Allows Rust-style 1_000_000 and 1_000.0. *)
  let strip_us s = String.concat "" (String.split_on_char '_' s)

  (* Pending lex-phase warnings: drained by compiler.ml into ctx.diags after parsing. *)
  let pending_warnings
    : (Lexing.position * Lexing.position * string) list ref = ref []


  (* Warn on suspicious digit grouping with underscores.
     Fires on:
       10_00       — groups (2,2), trailing group not 3 digits
       10_00_000   — groups (2,2,3), inconsistent widths
       1_00        — groups (1,2), trailing group not 3 digits
     Does NOT fire on:
       1_000       — standard thousands separator
       1_000_000   — consistent 3-digit groups
       10_000_000  — leading group can differ, trailing must be 3 *)
  let check_int_grouping lexbuf (raw : string) =
    let int_part =
      let stop =
        let dot = try String.index raw '.'  with Not_found -> max_int in
        let e1  = try String.index raw 'e'  with Not_found -> max_int in
        let e2  = try String.index raw 'E'  with Not_found -> max_int in
        min dot (min e1 e2)
      in
      if stop = max_int then raw else String.sub raw 0 stop
    in
    if String.contains int_part '_' then begin
      let groups = String.split_on_char '_' int_part in
      match groups with
      | [] | [_] -> ()
      | _ :: rest ->
        let sizes = List.map String.length rest in
        (* Trailing groups should all be 3 digits (thousands separator).
           Flag if any trailing group is not 3, or if trailing groups
           are inconsistent with each other. *)
        let bad = List.exists (fun s -> s <> 3) sizes in
        if bad then begin
          let all_sizes = List.map String.length groups in
          let sizes_str =
            String.concat ", " (List.map string_of_int all_sizes)
          in
          let sp = Lexing.lexeme_start_p lexbuf in
          let ep = Lexing.lexeme_end_p lexbuf in
          pending_warnings := (sp, ep,
            Printf.sprintf
              "suspicious digit grouping in '%s' (group widths: %s) — \
               did you mean %s? Use 3-digit groups: 1_000, 10_000, 1_000_000"
              raw sizes_str
              (String.concat "" (List.map (fun g -> g) groups))
          ) :: !pending_warnings
        end
    end

  let kw_table : (string, token) Hashtbl.t =
    let t = Hashtbl.create 64 in
    List.iter (fun (k, v) -> Hashtbl.add t k v) [
      "time_unit",     TIME_UNIT;
      "compartments",  COMPARTMENTS;
      "parameters",    PARAMETERS;
      "tables",        TABLES;
      "forcing",       FORCING;
      "transitions",   TRANSITIONS;
      "via",           VIA;                               (* staged-residence dwell-law clause *)
      "observations",  OBSERVATIONS;
      "quantities",    QUANTITIES;
      "contrasts",     CONTRASTS;                         (* counterfactual contrasts *)
      "interventions", INTERVENTIONS;
      "reactive_interventions", REACTIVE_INTERVENTIONS;  (* gh#204 *)
      "when",          WHEN;                             (* gh#204 trigger head *)
      "action",        ACTION;                           (* gh#204 reactive action kv *)
      "ode",           ODE;
      "output",        OUTPUT;
      "simulate",      SIMULATE;
      "init",          INIT;
      "timepoints",    TIMEPOINTS;
      "scenarios",     SCENARIOS;
      "extends",       EXTENDS;
      "stratify",      STRATIFY;
      "let",           LET;
      "from",          FROM;
      "to",            TO;
      "where",         WHERE;
      "sum",           SUM;
      "consecutive",   CONSECUTIVE;
      "in",            IN;
      "by",            BY;
      "dimensions",    DIMENSIONS;
      "only",          ONLY;
      "real",          REAL;
      "integer",       INTEGER;
      "rate",          RATE;
      "probability",   PROBABILITY;
      "positive",      POSITIVE;
      "count",         COUNT;
      "instant",       INSTANT;
      "duration",      DURATION;
      "and",           AND;
      "or",            OR;
      "not",           NOT;
      "if",            IF;
      "then",          THEN;
      "else",          ELSE;
      "every",         EVERY;
      "until",         UNTIL;
      "at",            AT_KW;
      "format",        FORMAT;
      "description",   DESCRIPTION;
      "null",          NULL;
      "transfer",      TRANSFER;
      "balance",       BALANCE;
      "events",        EVENTS;
      "add",           ADD;
      "at_day",        AT_DAY;
      "likelihood",    LIKELIHOOD;
      "origin",        ORIGIN;
      "columns",       COLUMNS;
      "emit_schedule", EMIT_SCHEDULE;
    ];
    t

  let lookup_kw s =
    match Hashtbl.find_opt kw_table s with
    | Some tok -> tok
    | None     -> IDENT s
}

let digit   = ['0'-'9']
let alpha   = ['a'-'z' 'A'-'Z' '_']
let alnum   = ['a'-'z' 'A'-'Z' '0'-'9' '_']
let ws      = [' ' '\t' '\r']
(* int_lit allows underscore separators between digit groups: 1_000_000 *)
let int_lit = digit+ ('_'+ digit+)*

rule token = parse
  | ws+               { token lexbuf }
  | '\n'              { Lexing.new_line lexbuf; token lexbuf }

  (* Attribute opener: `#[` with NO intervening space begins an
     attribute (e.g. `#[lineage]`). One-character lookahead. The
     comment rules below are written so they never match a `#`
     immediately followed by `[`, so `# [` (with space) and `#foo`
     stay comments. The attribute name is lexed as a following IDENT
     by the parser.

     ocamllex picks the longest match, so the comment rule must be
     structured to NOT consume `#[…]`. We do this by splitting the
     comment alternatives: a bare `#` at end-of-line, or `#` followed
     by a non-`[` character then arbitrary rest-of-line. *)
  | "#["              { HASH_LBRACKET }
  (* Doc comment: `#'` followed by the rest of the line is a roxygen-style
     documentation line that ATTACHES to the following declaration (it is not
     discarded like an ordinary `#` comment). This rule MUST precede the comment
     rules below: a `#' …` line also matches the line-comment rule (its char
     class `[^'\n' '[']` includes `'`) to the same length, and ocamllex breaks a
     longest-match tie by source order — earliest wins. Placed after `#[` so the
     attribute opener still wins for `#[…]`. *)
  | "#'" ([^'\n']* as body)   { DOC (String.trim body) }
  | '#'                       { token lexbuf }   (* lone `#` (e.g. end of line) *)
  | '#' [^'\n' '['] [^'\n']*  { token lexbuf }   (* line comment, first char not `[` *)

  (* Unit literals: 'days, 'per_day, etc. *)
  | "'days"      { UNIT_IDENT "days" }
  | "'weeks"     { UNIT_IDENT "weeks" }
  | "'months"    { UNIT_IDENT "months" }
  | "'years"     { UNIT_IDENT "years" }
  | "'per_day"   { UNIT_IDENT "per_day" }
  | "'per_week"  { UNIT_IDENT "per_week" }
  | "'per_month" { UNIT_IDENT "per_month" }
  | "'per_year"  { UNIT_IDENT "per_year" }
  | "'count"     { UNIT_IDENT "count" }
  | "'ratio"     { UNIT_IDENT "ratio" }

  (* Numbers — underscore separators allowed between digit groups (1_000_000) *)
  | int_lit '.' digit* (['e' 'E'] ['+' '-']? int_lit)?
      { let raw = Lexing.lexeme lexbuf in
        check_int_grouping lexbuf raw;
        FLOAT (float_of_string (strip_us raw)) }
  | '.' digit+ (['e' 'E'] ['+' '-']? int_lit)?
      { FLOAT (float_of_string (strip_us (Lexing.lexeme lexbuf))) }
  | int_lit (['e' 'E'] ['+' '-']? int_lit)
      { let raw = Lexing.lexeme lexbuf in
        check_int_grouping lexbuf raw;
        FLOAT (float_of_string (strip_us raw)) }
  | int_lit
      { let raw = Lexing.lexeme lexbuf in
        check_int_grouping lexbuf raw;
        INT (int_of_string (strip_us raw)) }

  (* String literals *)
  | '"'
      { let buf = Buffer.create 64 in
        string_content buf lexbuf }

  (* Identifiers / keywords *)
  | alpha alnum*
      { lookup_kw (Lexing.lexeme lexbuf) }

  (* Two-character operators — must come before single-char *)
  | "-->"   { ARROW }
  | "=="    { EQ2 }
  | "!="    { NEQ }
  | "<="    { LE }
  | ">="    { GE }

  (* Unicode cross product *)
  | "\xc3\x97" { CROSS }   (* UTF-8 for × *)

  (* Single-character tokens *)
  | '='     { EQ }
  | ':'     { COLON }
  | ','     { COMMA }
  | '.'     { DOT }
  | '{'     { LBRACE }
  | '}'     { RBRACE }
  | '['     { LBRACKET }
  | ']'     { RBRACKET }
  | '('     { LPAREN }
  | ')'     { RPAREN }
  | '+'     { PLUS }
  | '-'     { MINUS }
  | '*'     { STAR }
  | '/'     { SLASH }
  | '^'     { CARET }
  | '@'     { AT }
  | '~'     { TILDE }
  | '|'     { PIPE }
  | '<'     { LT }
  | '>'     { GT }

  | eof     { EOF }

  | _ as c  { raise (LexError (Printf.sprintf "unexpected character '%c'" c)) }

and string_content buf = parse
  | '"'           { STRING (Buffer.contents buf) }
  | '\\' '"'      { Buffer.add_char buf '"'; string_content buf lexbuf }
  | '\\' '\\'     { Buffer.add_char buf '\\'; string_content buf lexbuf }
  | '\\' 'n'      { Buffer.add_char buf '\n'; string_content buf lexbuf }
  | '\\' 't'      { Buffer.add_char buf '\t'; string_content buf lexbuf }
  | [^'"' '\\']+  { Buffer.add_string buf (Lexing.lexeme lexbuf); string_content buf lexbuf }
  | eof           { raise (LexError "unterminated string") }
