(* What follows a parameter's kind: an optional size bracket (`real[6]`) and
   an optional `dim` clause (`real dim T^-1`). Brackets on the type mean size
   and only size; a dimension is written with `dim`. The two notations this
   replaces, a dimension bracket (`real [1/T]`) and a unit literal
   (`positive 'per_day`), are refused here with a machine-applicable fix-it,
   never reinterpreted (proposal 2026-11-05-size-brackets-and-dim-clause §3).

   Called from parser semantic actions, so diagnostics go through
   [Parser_errors] with source positions. *)

open Ast

(* One factor of a dimension expression as written: an integer literal, a
   power of a base dimension, or an unknown name. *)
type factor =
  | FInt  of int
  | FBase of int * int          (* (P exponent, T exponent) *)
  | FBad  of string

(* A dimension expression: factors with a sign, +1 in the numerator and -1 in
   the denominator, in source order (`P/T` is [(1, P); (-1, T)]). *)
type dim_expr = (int * factor) list

let base_factor name exp =
  match name with
  | "P" -> FBase (exp, 0)
  | "T" -> FBase (0, exp)
  | other -> FBad other

(* Canonical source spelling of a dimension, used in fix-its. *)
let render ((p, t) : int * int) : string =
  match p, t with
  | 0, 0 -> "1"
  | 1, 0 -> "P"
  | 0, 1 -> "T"
  | 0, -1 -> "1/T"
  | 1, -1 -> "P/T"
  | -1, 0 -> "1/P"
  | _ ->
    let part b e = if e = 0 then [] else if e = 1 then [b] else [Printf.sprintf "%s^%d" b e] in
    String.concat "*" (part "P" p @ part "T" t)

let accepted = "1, P, T, 1/T, T^-1, P/T, P*T^-1, or integer powers of P and T \
                combined with * and /"

(* The (P, T) exponents of a dimension expression, or the reason it is not
   one. The literal 1 is the dimensionless unit; any other integer is not a
   dimension. *)
let eval (e : dim_expr) : (int * int, string) result =
  List.fold_left (fun acc (sign, f) ->
    match acc, f with
    | Error _, _ -> acc
    | Ok d, FInt 1 -> Ok d
    | Ok _, FInt n -> Error (Printf.sprintf "`%d` is not a dimension" n)
    | Ok (p, t), FBase (a, b) -> Ok (p + sign * a, t + sign * b)
    | Ok _, FBad name -> Error (Printf.sprintf "`%s` is not a base dimension (use P or T)" name))
    (Ok (0, 0)) e

(* The dimension a kind keyword fixes; [None] for the two kinds whose
   dimension is inferred (or given by `dim`). Mirrors [Dimcheck.kind_dim]. *)
let kind_dim : param_type -> (int * int) option = function
  | PRate -> Some (0, -1)
  | PProbability -> Some (0, 0)
  | PCount -> Some (1, 0)
  | PInstant | PDuration -> Some (0, 1)
  | PPositive | PReal -> None

let kind_name : param_type -> string = function
  | PRate -> "rate" | PProbability -> "probability" | PCount -> "count"
  | PInstant -> "instant" | PDuration -> "duration"
  | PPositive -> "positive" | PReal -> "real"

let unit_dim : unit_lit -> int * int = function
  | Days | Weeks | Months | Years -> (0, 1)
  | PerDay | PerWeek | PerMonth | PerYear -> (0, -1)
  | Count -> (1, 0)
  | Ratio -> (0, 0)

let unit_name : unit_lit -> string = function
  | Days -> "days" | Weeks -> "weeks" | Months -> "months" | Years -> "years"
  | PerDay -> "per_day" | PerWeek -> "per_week" | PerMonth -> "per_month"
  | PerYear -> "per_year" | Count -> "count" | Ratio -> "ratio"

let see_changes = "See `camdl docs language-changes`."

type pos = Lexing.position

(* A bracket item is a size when it is a bare integer. *)
let size_of_item (e : dim_expr) : int option =
  match e with [ (1, FInt n) ] -> Some n | _ -> None

(* Resolve the annotations after a kind to the parameter's explicit dimension,
   pushing a refusal for every old form. [kind_end] is the end of the kind
   keyword: each fix-it replaces the text from there to the end of the old
   annotations, so applying one leaves exactly one space before `dim`.

   At most one fix-it is attached per declaration — the one that rewrites
   every old annotation on it at once — so fix-its never overlap. *)
let resolve ~(kind : param_type) ~(kind_end : pos)
    ~(unit_ : (unit_lit * pos * pos) option)
    ~(bracket : (dim_expr list * pos * pos) option)
    ~(clause : (dim_expr * pos * pos) option) : (int * int) option =
  let kname = kind_name kind in
  let inferred_kind = kind_dim kind = None in
  (* The end of the old annotations, and the dimension they meant. *)
  let old_end = match bracket, unit_ with
    | Some (_, _, ep), _ -> Some ep
    | None, Some (_, _, ep) -> Some ep
    | None, None -> None in
  let fix_used = ref false in
  (* The replacement for the old annotations: ` dim D` on a kind that takes
     one (unless a `dim` clause is already written), otherwise nothing. *)
  let fix_for (d : (int * int) option) =
    if !fix_used then None
    else match old_end with
      | None -> None
      | Some ep ->
        fix_used := true;
        let text = match d, clause with
          | Some d, None when inferred_kind -> " dim " ^ render d
          | _ -> "" in
        Some (kind_end, ep, text)
  in
  let push ~sp ~ep ~code ~msg ~hint ?fix () =
    Parser_errors.push_error_fix ~sp ~ep ~code ~msg ~hint ~fix
  in
  (* A unit literal on a parameter. *)
  (match unit_ with
   | None -> ()
   | Some (u, sp, ep) ->
     let d = unit_dim u in
     let hint =
       if inferred_kind then
         Printf.sprintf "write `%s dim %s`. A unit literal carries a scale as \
                         well as a dimension, and a parameter's value is always \
                         in the model's time unit, so the scale would be \
                         silently dropped. %s" kname (render d) see_changes
       else
         Printf.sprintf "drop the unit literal: the `%s` kind already fixes the \
                         dimension. %s" kname see_changes
     in
     (* With a bracket also present, the bracket's refusal carries the fix. *)
     let fix = if bracket = None then fix_for (Some d) else None in
     push ~sp ~ep ~code:"E117"
       ~msg:(Printf.sprintf "a parameter has no scale of its own; a unit literal \
                             ('%s) is not allowed on a parameter declaration" (unit_name u))
       ~hint ?fix ());
  (* A bracket after the kind. *)
  (match bracket with
   | None -> ()
   | Some (items, sp, ep) ->
     let sizes = List.map size_of_item items in
     if List.for_all Option.is_some sizes then begin
       match List.filter_map Fun.id sizes with
       | [ n ] when n < 2 ->
         let fix = if n = 1 then fix_for (Some (0, 0)) else None in
         push ~sp ~ep ~code:"E417"
           ~msg:(Printf.sprintf "a parameter family needs at least two cells; `%s[%d]` \
                                 has %d" kname n n)
           ~hint:(Printf.sprintf "a single cell is a scalar, written `%s`.%s %s" kname
                    (if inferred_kind && n = 1 then
                       Printf.sprintf " If this was a dimension annotation, write `%s dim 1`." kname
                     else "")
                    see_changes) ?fix ()
       | [ _ ] ->
         push ~sp ~ep ~code:"E119"
           ~msg:(Printf.sprintf "sized parameter families (`%s[n]`) are not \
                                 implemented yet" kname)
           ~hint:"declare the cells as separate parameters, or index the \
                  family over a named dimension (`b[province] : real`)" ()
       | _ ->
         push ~sp ~ep ~code:"E417"
           ~msg:"a parameter family has at most one sized axis"
           ~hint:"index the other axes over named dimensions \
                  (`b[province] : real[6]`)" ()
     end else begin
       (* A dimension in brackets: the old annotation. *)
       let d = match items with
         | [ e ] -> (match eval e with Ok d -> Some d | Error _ -> None)
         | _ -> None in
       let fix = fix_for d in
       let written = match d with Some d -> render d | None -> "..." in
       push ~sp ~ep ~code:"E116"
         ~msg:"dimension brackets were replaced by the `dim` clause; brackets \
               after a kind now give a parameter family's size"
         ~hint:(if inferred_kind then
                  Printf.sprintf "write `%s dim %s`. %s" kname written see_changes
                else
                  Printf.sprintf "drop the bracket: the `%s` kind already fixes the \
                                  dimension. %s" kname see_changes) ?fix ()
     end);
  (* The `dim` clause itself. *)
  match clause with
  | None -> None
  | Some (e, sp, ep) ->
    match eval e with
    | Error why ->
      push ~sp ~ep ~code:"E103"
        ~msg:(Printf.sprintf "unknown dimension in `dim` clause: %s" why)
        ~hint:(Printf.sprintf "a dimension is %s" accepted) ();
      None
    | Ok d ->
      match kind_dim kind with
      | None -> Some d
      | Some kd when kd = d ->
        let fix = if old_end = None then Some (kind_end, ep, "") else None in
        push ~sp ~ep ~code:"E281"
          ~msg:(Printf.sprintf "the `%s` kind already has dimension %s; the `dim` \
                                clause is redundant" kname (render kd))
          ~hint:"drop the `dim` clause; only `positive` and `real` take one" ?fix ();
        None
      | Some kd ->
        push ~sp ~ep ~code:"E281"
          ~msg:(Printf.sprintf "`dim %s` contradicts the `%s` kind, whose dimension is %s"
                  (render d) kname (render kd))
          ~hint:(Printf.sprintf "keep `%s` and drop the clause, or declare the \
                                 parameter `positive dim %s` or `real dim %s`"
                   kname (render d) (render d)) ();
        None
